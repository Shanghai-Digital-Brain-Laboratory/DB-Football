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

from light_malib.envs.gr_football.tools import geometry as g
import numpy as np


class State:
    def __init__(self, n_player=11):
        self.obs_list = []
        self.action_list = []
        self.last_ball_owned_team = None
        self.last_ball_owned_player = None

        self.num_player = n_player  # 11 for 11v11 scenarios, 5 for 5v5
        self.last_loffside = np.zeros(self.num_player, np.float32)
        self.last_roffside = np.zeros(self.num_player, np.float32)

    def update_obs(self, obs):
        self.obs_list.append(obs)
        self.update_last_ball_owned()

    def update_action(self, action):
        self.action_list.append(action)

    @property
    def obs(self):
        return self.obs_list[-1] if len(self.obs_list) >= 1 else None

    @property
    def prev_obs(self):
        return self.obs_list[-2] if len(self.obs_list) >= 2 else None

    @property
    def action(self):
        return self.action_list[-1] if len(self.action_list) >= 1 else None

    @property
    def prev_action(self):
        return self.action_list[-2] if len(self.action_list) >= 2 else None

    def get_team_states(self):
        #for qmixer
        left_status = np.concatenate([
            self.obs['left_team'].flatten(),
            self.obs['left_team_direction'].flatten(),
            self.obs['left_team_roles']]
        )           #25
        right_status = np.concatenate([
            self.obs['right_team'].flatten(),
            self.obs['right_team_direction'].flatten(),
            self.obs['right_team_roles']]
        )           # 25
        ball_status = np.concatenate([
            self.obs['ball'],
            self.obs['ball_direction'],
            self.obs['ball_rotation']]
        )
        game_mode = np.zeros(7, dtype=np.float32)
        game_mode[self.obs["game_mode"]] = 1
        score_ratio = self.obs["score"][0] - self.obs["score"][1]
        score_ratio /= 5.0
        score_ratio = min(score_ratio, 1.0)
        score_ratio = max(-1.0, score_ratio)
        steps_left = self.obs["steps_left"]
        match_state = np.concatenate(
            (
                np.array([ steps_left , score_ratio]),
                game_mode,
            )
        )

        team_state = np.concatenate([left_status, right_status, ball_status, match_state])
        return team_state



    def update_last_ball_owned(self):
        last_ball_owned_team, last_ball_owned_player = self.get_last_ball_owned(
            self.obs, self.prev_obs
        )
        if last_ball_owned_team is not None:
            self.last_ball_owned_team = last_ball_owned_team
            self.last_ball_owned_player = last_ball_owned_player
        assert self.last_ball_owned_team != -1 and self.last_ball_owned_player != -1

    def get_last_ball_owned(self, obs, prev_obs=None):
        if prev_obs is None:
            team, idx, dist = self.get_closest_player_to_ball(obs)
            owned_team = 1 if team == "right" else 0
            return owned_team, idx
        if obs["game_mode"] == 0:  # Normal
            if obs["ball_owned_team"] != -1:
                return obs["ball_owned_team"], obs["ball_owned_player"]
            elif obs["ball_owned_team"] == -1 and self.scored(prev_obs, obs):
                # another team kick off now
                team = "right" if self.scored(prev_obs, obs) == 1 else "left"
                team, idx, dist = self.get_closest_player_to_ball(obs, team)
                owned_team = 1 if team == "right" else 0
                return owned_team, idx
            else:
                # TODO jh need code review
                # NOTE direct passing: this is hard, not sure it is a correct implementation
                team, idx, dist = self.get_closest_player_to_ball(obs)
                prev_ball_coord_speed = prev_obs["ball_direction"]
                ball_coord_speed = obs["ball_direction"]
                speed_change = g.get_speed(ball_coord_speed - prev_ball_coord_speed)
                if (
                    dist < g.BALL_CONTROLLED_DIST * 1.5
                    and g.tz(obs["ball"][-2]) < g.BALL_CONTROLLED_HEIGHT
                    and speed_change > g.BALL_SPEED_VARIATION_THRESH
                ):
                    owned_team = 1 if team == "right" else 0
                    return owned_team, idx
                else:
                    return None, None
        else:  # KickOff,GoalKick,FreeKick,ThrowIn,Penalty
            # just see who is closest to the ball.
            team, idx, _ = self.get_closest_player_to_ball(obs)
            owned_team = 1 if team == "right" else 0
            return owned_team, idx

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

    def get_offside(self, obs):
        """
        check offside status, borrow from wekick implementation
        """
        ball = np.array(obs["ball"][:2])
        ally = np.array(obs["left_team"])
        enemy = np.array(obs["right_team"])

        #  任意球、角球等没有越位，只有正常比赛有越位 (free kick and corner has no offside whereas full-game scenario does)
        if obs["game_mode"] != 0:
            self.last_loffside = np.zeros(self.num_player, np.float32)
            self.last_roffside = np.zeros(self.num_player, np.float32)
            return np.zeros(self.num_player, np.float32), np.zeros(
                self.num_player, np.float32
            )

        need_recalc = False
        effective_ownball_team = -1
        effective_ownball_player = -1

        #  当一方控球时才判断是否越位 (check offside if one side of the team owns the ball )
        if obs["ball_owned_team"] > -1:
            effective_ownball_team = obs["ball_owned_team"]
            effective_ownball_player = obs["ball_owned_player"]
            need_recalc = True
        else:
            # 没有控球但是离球很近也要判断越位 (check offside when no control of the ball but stand closed to the ball)
            # 有这种情况比如一脚传球时obs['ball_owned_team'] 时不会显示的 (case exists when pass the ball
            #                                                       and ['ball_owned_team'] term does not display)
            ally_dist = np.linalg.norm(ball - ally, axis=-1)
            enemy_dist = np.linalg.norm(ball - enemy, axis=-1)
            # 我方控球      (we own the ball)
            if np.min(ally_dist) < np.min(enemy_dist):
                if np.min(ally_dist) < 0.017:
                    need_recalc = True
                    effective_ownball_team = 0
                    effective_ownball_player = np.argmin(ally_dist)
            # 对方控球      (opponent owns the ball)
            elif np.min(enemy_dist) < np.min(ally_dist):
                if np.min(enemy_dist) < 0.017:
                    need_recalc = True
                    effective_ownball_team = 1
                    effective_ownball_player = np.argmin(enemy_dist)

        if not need_recalc:
            return self.last_loffside, self.last_roffside

        left_offside = np.zeros(self.num_player, np.float32)
        right_offside = np.zeros(self.num_player, np.float32)

        if effective_ownball_team == 0:
            # 所有对方球员的x坐标加入排序        (all opponent player's x value added for ranking)
            # 取倒数第二名防守球员作为越位线       (pick the second last defender as the offside line)
            right_xs = [obs["right_team"][k][0] for k in range(0, self.num_player)]
            right_xs = np.array(right_xs)
            right_xs.sort()

            # 将倒数第二名防守球员的位置和球比较，更深的成为越位线        (compare second last defender position with ball position)
            offside_line = max(right_xs[-2], ball[0])

            # 己方守门员不参与进攻，不为其计算越位标志，直接用0的初始化     (our GK plays no offense, skip it and initialise with 0)
            # 己方半场不计算越位             (no checking offside at our half)
            for k in range(1, self.num_player):
                if (
                    obs["left_team"][k][0] > offside_line
                    and k != effective_ownball_player
                    and obs["left_team"][k][0] > 0.0
                ):
                    left_offside[k] = 1.0
        else:
            left_xs = [obs["left_team"][k][0] for k in range(0, 5)]
            left_xs = np.array(left_xs)
            left_xs.sort()

            # 左右半场左边相反          (left and right half are opposite)
            offside_line = min(left_xs[1], ball[0])

            # 左右半场左边相反          (left and right half are opposite)
            for k in range(1, self.num_player):
                if (
                    obs["right_team"][k][0] < offside_line
                    and k != effective_ownball_player
                    and obs["right_team"][k][0] < 0.0
                ):
                    right_offside[k] = 1.0

        self.last_loffside = left_offside
        self.last_roffside = right_offside

        return left_offside, right_offside
