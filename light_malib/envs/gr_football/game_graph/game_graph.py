# Copyright 2022 Digital Brain Laboratory, Yan Song and He jiang
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ..tools import geometry as g
import numpy as np
from .data_structure import Node, Chain, Subgame
from ..tools.tracer import MatchTracer

from .event import GoalEvent, PassingEvent, InterceptingBallEvent, LosingBallEvent

DEBUG = False


class GameGraph:
    def __init__(self, tracer: MatchTracer) -> None:
        self.tracer = tracer

        self.nodes = []
        self.chains = []
        self.subgames = []

        self.goals = {}  # the next step changes the score.
        self.passings = {}  # the next step ball out.
        self.losing_balls = {}  # the next step changes the ownership
        self.intercepting_balls = {}  # the step intercepting balls
        self.build(self.tracer.data)

        self.step2node = []
        self.step2chain = []
        self.step2subgame = []

        for subgame in self.subgames:
            for chain in subgame.chains:
                for node in chain.nodes:
                    for step in range(node.s_step, node.e_step + 1):
                        self.step2node.append(node)
                        self.step2chain.append(chain)
                        self.step2subgame.append(subgame)

    @property
    def events(self):
        def m(obj):
            if isinstance(obj, LosingBallEvent):
                return 2
            if isinstance(obj, InterceptingBallEvent):
                return 1
            return 0

        events = []
        events.extend(list(self.goals.items()))
        events.extend(list(self.passings.items()))
        events.extend(list(self.losing_balls.items()))
        events.extend(list(self.intercepting_balls.items()))
        events = sorted(events, key=lambda x: (x[0], m(x[1])))
        return events

    def _get_closest_player_to_ball(self, obs, team):
        ball = obs["ball"]
        players = obs["{}_team".format(team)]
        dists = g.get_dist(ball[:2], players)
        idx = np.argmin(dists)
        return idx, dists[idx]

    def get_closest_player_to_ball(self, obs, team=None):
        if team is not None:
            idx, dist = self._get_closest_player_to_ball(obs, team)
            return team, idx, dist
        l_idx, l_dist = self._get_closest_player_to_ball(obs, "left")
        r_idx, r_dist = self._get_closest_player_to_ball(obs, "right")
        if l_dist < r_dist:
            return "left", l_idx, l_dist
        else:
            return "right", r_idx, r_dist

    def scored(self, prev_obs, obs):
        if prev_obs is None:
            return 0
        if prev_obs["score"][0] < obs["score"][0]:
            return 1
        elif prev_obs["score"][1] < obs["score"][1]:
            return -1
        return 0

    def get_ball_owned(self, obs, prev_obs=None, next_obs=None):
        if obs["game_mode"] == 0:  # Normal
            if obs["ball_owned_team"] != -1:
                return obs["ball_owned_team"], obs["ball_owned_player"]
            elif obs["ball_owned_team"] == -1 and (
                (prev_obs is None)
                or (prev_obs is not None and self.scored(prev_obs, obs))
            ):
                # another team kick off now
                team = "right" if self.scored(prev_obs, obs) == 1 else "left"
                team, idx, dist = self.get_closest_player_to_ball(obs, team)
                owned_team = 1 if team == "right" else 0
                return owned_team, idx
            elif obs["ball_owned_team"] == -1 and next_obs is not None:
                # NOTE direct passing: this is hard, not sure it is a correct implementation
                team, idx, dist = self.get_closest_player_to_ball(obs)
                ball_coord_speed = obs["ball_direction"]
                next_ball_coord_speed = next_obs["ball_direction"]
                speed_change = g.get_speed(
                    np.array(next_ball_coord_speed) - np.array(ball_coord_speed)
                )
                if (
                    dist < g.BALL_CONTROLLED_DIST * 1.5
                    and g.tz(obs["ball"][-2]) < g.BALL_CONTROLLED_HEIGHT
                    and speed_change > g.BALL_SPEED_VARIATION_THRESH
                ):
                    owned_team = 1 if team == "right" else 0
                    return owned_team, idx
                else:
                    return obs["ball_owned_team"], obs["ball_owned_player"]
            else:
                # just see who is closest to the ball.
                team, idx, _ = self.get_closest_player_to_ball(obs)
                owned_team = 1 if team == "right" else 0
                return owned_team, idx
        else:  # KickOff,GoalKick,FreeKick,ThrowIn,Penalty
            # just see who is closest to the ball.
            team, idx, _ = self.get_closest_player_to_ball(obs)
            owned_team = 1 if team == "right" else 0
            return owned_team, idx

    def build(self, game):

        # get nodes
        last_owned_team = None
        last_owned_player = None
        idx = 0
        node = None
        for step, frame in game.items():
            prev_frame = game[step - 1] if step != 0 else None
            next_frame = game[step + 1] if step < len(game) - 1 else None
            owned_team, owned_player = self.get_ball_owned(
                frame, prev_frame, next_frame
            )
            score = self.scored(prev_frame, frame)
            if score != 0:
                self.goals[step - 1] = GoalEvent(step - 1, score)
            if (
                owned_team != last_owned_team
                or owned_player != last_owned_player
                or score != 0
            ):
                # means a transition
                if node is not None:
                    # complete the last node
                    node.set_e_step(step - 1)
                    self.nodes.append(node)
                    node = None
                # start a new node
                last_owned_team = owned_team
                last_owned_player = owned_player
                node = Node(idx, owned_team, owned_player, step)
                idx += 1
        node.set_e_step(step)
        self.nodes.append(node)
        assert len(self.nodes) == idx

        # build chains
        idx = 0
        cur_team_owner = None
        last_player_node = None
        last2_player_node = None
        for node in self.nodes:
            if cur_team_owner is None:
                # NOTE owned team may be -1,0 or 1, but less likely to be -1 in general cases.
                chain = Chain(idx)
                idx += 1
                chain.append(node)
                cur_team_owner = node.owned_team
                if node.owned_team != -1:
                    last2_player_node = last_player_node
                    last_player_node = node
            else:
                scored = node.s_step - 1 in self.goals
                if (
                    node.owned_team == cur_team_owner or node.owned_team == -1
                ) and not scored:
                    if node.owned_team != -1:
                        if (
                            last_player_node is not None
                            and node.owned_player != last_player_node.owned_player
                        ):
                            # it is a pass
                            self.passings[last_player_node.e_step] = PassingEvent(
                                last_player_node.e_step, last_player_node, node
                            )
                        last2_player_node = last_player_node
                        last_player_node = node
                    chain.append(node)
                else:
                    self.chains.append(chain)

                    if scored:  # prioritized
                        # who shot the goal
                        self.goals[node.s_step - 1].out_node = last_player_node
                    elif cur_team_owner != node.owned_team and node.owned_team != -1:
                        if cur_team_owner != -1:
                            self.losing_balls[node.s_step - 1] = LosingBallEvent(
                                node.s_step - 1,
                                self.data[node.s_step]["game_mode"],
                                last_player_node,
                                passer_node=last2_player_node,
                            )
                            if self.data[node.s_step]["game_mode"] == 0:
                                # directly intercept...
                                self.intercepting_balls[
                                    node.s_step
                                ] = InterceptingBallEvent(node.s_step, node)
                            else:
                                # game mode changes...
                                pass
                    else:
                        raise Exception("wrong implementation!")
                    # start a new chain
                    last_player_node = None
                    last2_player_node = None
                    cur_team_owner = None
                    chain = Chain(idx)
                    idx += 1
                    chain.append(node)
                    cur_team_owner = node.owned_team
                    if node.owned_team != -1:
                        last2_player_node = last_player_node
                        last_player_node = node
        self.chains.append(chain)
        assert len(self.chains) == idx

        # build subgames
        # subgames are divided by goals
        idx = 0
        subgame = Subgame(idx)
        idx += 1
        for chain in self.chains:
            subgame.append(chain)
            if chain.e_step in self.goals:
                self.subgames.append(subgame)
                subgame = Subgame(idx)
                idx += 1
        if len(subgame) != 0:
            self.subgames.append(subgame)
        assert len(self.subgames) == idx

        # TODO bind some control-related information

    def __len__(self):
        return len(self.subgames)

    def __getitem__(self, idx):
        return self.subgames[idx]

    def __str__(self) -> str:
        s = "G[{} {}]: \n".format(self.s_step, self.e_step)
        for idx, subgame in enumerate(self.subgames):
            s += "<{:04d}> {}\n".format(subgame.s_step, str(subgame))
        s += "\nEvents: \n"
        for idx, event in enumerate(self.events):
            # print(type(event[1]))
            s += "<{:04d}> {}\n".format(event[0], str(event[1]))
        return s

    __repr__ = __str__

    @property
    def data(self):
        return self.tracer.data

    @property
    def s_step(self):
        return self.subgames[0].s_step if len(self) > 0 else None

    @property
    def e_step(self):
        return self.subgames[-1].e_step if len(self) > 0 else None

    @property
    def n_steps(self):
        return self.e_step - self.s_step + 1

    @property
    def n_left(self):
        return self.tracer.data[0]["n_left"]

    @property
    def n_right(self):
        return self.tracer.data[0]["n_right"]

    @property
    def n_steps(self):
        return len(self.tracer.data)
