# MIT License

# Copyright (c) 2022 DigitalBrain, Yan Song and He jiang
# Copyright (c) 2020 seungeunrho

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''
Our Feature Encoder code is adapated from wekick and liveinparis in the kaggle football competition.

basic_enhanced_11 outputs 217+107-dimension features and has less action masking,
used as an extended FE for 11v11 full-game scenarios
'''

from typing import OrderedDict
import numpy as np
from light_malib.utils.logger import Logger
from gym.spaces import Box

class FeatureEncoder:
    def __init__(self):
        self.active = -1
        self.player_pos_x, self.player_pos_y = 0, 0
        self.action_n = 19
        self.use_action_gramma = False

    def get_feature_dims(self):
        dims = {
            "player": 29,
            "ball": 18,
            "left_team": 7,
            "left_team_closest": 7,
            "right_team": 7,
            "right_team_closest": 7,
            "match_state": 10,
            "offside": 10,
            "card": 20,
            "sticky_action": 10,
            "ball_distance": 9
        }
        return dims

    def encode(self, states):
        feats = []
        for state in states:
            feat = self.encode_each(state)
            feats.append(feat)
        return feats

    @property
    def observation_space(self):
        return Box(low=-1000, high=1000, shape=[217 + 107])

    def encode_each(self, state):
        obs = state.obs
        his_actions = state.action_list

        player_num = obs["active"]

        player_pos_x, player_pos_y = obs["left_team"][player_num]
        player_direction = np.array(obs["left_team_direction"][player_num])
        player_speed = np.linalg.norm(player_direction)
        player_role = obs["left_team_roles"][player_num]
        player_role_onehot = self._encode_role_onehot(player_role)
        player_tired = obs["left_team_tired_factor"][player_num]
        is_dribbling = obs["sticky_actions"][9]
        is_sprinting = obs["sticky_actions"][8]

        ball_x, ball_y, ball_z = obs["ball"]
        ball_x_relative = ball_x - player_pos_x
        ball_y_relative = ball_y - player_pos_y
        ball_x_speed, ball_y_speed, _ = obs["ball_direction"]
        ball_distance = np.linalg.norm([ball_x_relative, ball_y_relative])
        ball_speed = np.linalg.norm([ball_x_speed, ball_y_speed])
        ball_owned = 0.0
        if obs["ball_owned_team"] == -1:
            ball_owned = 0.0
        else:
            ball_owned = 1.0
        ball_owned_by_us = 0.0
        if obs["ball_owned_team"] == 0:
            ball_owned_by_us = 1.0
        elif obs["ball_owned_team"] == 1:
            ball_owned_by_us = 0.0
        else:
            ball_owned_by_us = 0.0

        ball_which_zone = self._encode_ball_which_zone(ball_x, ball_y)

        if ball_distance > 0.03:
            ball_far = 1.0
        else:
            ball_far = 0.0

        avail = self.get_available_actions(obs, ball_distance, his_actions)
        # Logger.log(logging.INFO,"avail: {} his_actions: {}".format(avail,his_actions))
        # print("avail: {}".format(avail))
        # avail = self._get_avail_new(obs, ball_distance, action_num)
        # avail = self._get_avail(obs, ball_distance)
        player_state = np.concatenate(
            (
                # avail[2:],
                obs["left_team"][player_num],
                player_direction * 100,
                [player_speed * 100],
                player_role_onehot,
                [ball_far, player_tired, is_dribbling, is_sprinting],
            )
        )

        ball_state = np.concatenate(
            (
                np.array(obs["ball"]),
                np.array(ball_which_zone),
                np.array([ball_x_relative, ball_y_relative]),
                np.array(obs["ball_direction"]) * 20,
                np.array(
                    [ball_speed * 20, ball_distance, ball_owned, ball_owned_by_us]
                ),
            )
        )

        obs_left_team = np.delete(obs["left_team"], player_num, axis=0)
        obs_left_team_direction = np.delete(
            obs["left_team_direction"], player_num, axis=0
        )
        left_team_relative = obs_left_team
        left_team_distance = np.linalg.norm(
            left_team_relative - obs["left_team"][player_num], axis=1, keepdims=True
        )
        left_team_speed = np.linalg.norm(obs_left_team_direction, axis=1, keepdims=True)
        left_team_tired = np.delete(
            obs["left_team_tired_factor"], player_num, axis=0
        ).reshape(-1, 1)
        left_team_state = np.concatenate(
            (
                left_team_relative * 2,
                obs_left_team_direction * 100,
                left_team_speed * 100,
                left_team_distance * 2,
                left_team_tired,
            ),
            axis=1,
        )
        left_closest_idx = np.argmin(left_team_distance)
        left_closest_state = left_team_state[left_closest_idx]

        obs_right_team = np.array(obs["right_team"])
        obs_right_team_direction = np.array(obs["right_team_direction"])
        right_team_distance = np.linalg.norm(
            obs_right_team - obs["left_team"][player_num], axis=1, keepdims=True
        )
        right_team_speed = np.linalg.norm(
            obs_right_team_direction, axis=1, keepdims=True
        )
        right_team_tired = np.array(obs["right_team_tired_factor"]).reshape(-1, 1)
        right_team_state = np.concatenate(
            (
                obs_right_team * 2,
                obs_right_team_direction * 100,
                right_team_speed * 100,
                right_team_distance * 2,
                right_team_tired,
            ),
            axis=1,
        )
        right_closest_idx = np.argmin(right_team_distance)
        right_closest_state = right_team_state[right_closest_idx]

        steps_left = obs['steps_left']  # steps left till end
        half_steps_left = steps_left
        if half_steps_left > 1500:
            half_steps_left -= 1501  # steps left till halfend
        half_steps_left = 1.0 * min(half_steps_left, 300.0)  # clip
        half_steps_left /= 300.0

        score_ratio = (obs['score'][0] - obs['score'][1])
        score_ratio /= 5.0
        score_ratio = min(score_ratio, 1.0)
        score_ratio = max(-1.0, score_ratio)

        game_mode = np.zeros(7, dtype=np.float32)
        game_mode[obs['game_mode']] = 1
        match_state = np.concatenate(
            (
                np.array([1.0 * steps_left / 3001, half_steps_left, score_ratio]),
                game_mode
            )
        )

        # offside
        l_o, r_o = state.get_offside(obs)
        offside = np.concatenate(
            (
                l_o,
                r_o
            )
        )
        # card
        card = np.concatenate(
            (
                obs['left_team_yellow_card'],
                obs['left_team_active'],
                obs['right_team_yellow_card'],
                obs['right_team_active']
            )
        )

        # sticky_action
        sticky_action = obs['sticky_actions']

        # ball_distance
        left_team_distance = np.linalg.norm(
            obs_left_team - obs["ball"][:2], axis=1, keepdims=False
        )
        right_team_distance = np.linalg.norm(
            obs_right_team - obs["ball"][:2], axis=1, keepdims=False
        )
        ball_distance = np.concatenate(
            (
                left_team_distance,
                right_team_distance
            )
        )
        state_dict = OrderedDict({
            "avail": avail,
            "ball": ball_state,
            "left_closest": left_closest_state,
            "left_team": left_team_state,
            "player": player_state,
            "right_closest": right_closest_state,
            "right_team": right_team_state,
            "match_state": match_state,
            "offside": offside,
            "card": card,
            "sticky_action": sticky_action,
            "ball_distance": ball_distance
        })

        feats = np.hstack(
            [np.array(state_dict[k], dtype=np.float32).flatten() for k in (state_dict)]
        )

        return feats

    def _get_avail(self, obs, ball_distance):
        avail = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        (
            NO_OP,
            MOVE,
            LONG_PASS,
            HIGH_PASS,
            SHORT_PASS,
            SHOT,
            SPRINT,
            RELEASE_MOVE,
            RELEASE_SPRINT,
            SLIDE,
            DRIBBLE,
            RELEASE_DRIBBLE,
        ) = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)

        if obs["ball_owned_team"] == 1:  # opponents owning ball
            (
                avail[LONG_PASS],
                avail[HIGH_PASS],
                avail[SHORT_PASS],
                avail[SHOT],
                avail[DRIBBLE],
            ) = (0, 0, 0, 0, 0)
        elif (
                obs["ball_owned_team"] == -1
                and ball_distance > 0.03
                and obs["game_mode"] == 0
        ):  # Ground ball  and far from me
            (
                avail[LONG_PASS],
                avail[HIGH_PASS],
                avail[SHORT_PASS],
                avail[SHOT],
                avail[DRIBBLE],
            ) = (0, 0, 0, 0, 0)
        else:  # my team owning ball
            avail[SLIDE] = 0

        # Dealing with sticky actions
        sticky_actions = obs["sticky_actions"]
        if sticky_actions[8] == 0:  # sprinting
            avail[RELEASE_SPRINT] = 0

        if sticky_actions[9] == 1:  # dribbling
            avail[SLIDE] = 0
        else:
            avail[RELEASE_DRIBBLE] = 0

        if np.sum(sticky_actions[:8]) == 0:
            avail[RELEASE_MOVE] = 0

        # if too far, no shot
        ball_x, ball_y, _ = obs["ball"]
        if ball_x < 0.64 or ball_y < -0.27 or 0.27 < ball_y:
            avail[SHOT] = 0
        elif (0.64 <= ball_x and ball_x <= 1.0) and (
                -0.27 <= ball_y and ball_y <= 0.27
        ):
            avail[HIGH_PASS], avail[LONG_PASS] = 0, 0

        if obs["game_mode"] == 2 and ball_x < -0.7:  # Our GoalKick
            avail = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            avail[LONG_PASS], avail[HIGH_PASS], avail[SHORT_PASS] = 1, 1, 1
            return np.array(avail)

        elif obs["game_mode"] == 4 and ball_x > 0.9:  # Our CornerKick
            avail = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            avail[LONG_PASS], avail[HIGH_PASS], avail[SHORT_PASS] = 1, 1, 1
            return np.array(avail)

        elif obs["game_mode"] == 6 and ball_x > 0.6:  # Our PenaltyKick
            avail = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            avail[SHOT] = 1
            return np.array(avail)

        return np.array(avail)

    def _get_avail_new(self, obs, ball_distance, action_n):
        # avail = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        assert (action_n == 19 or action_n == 20)  # we dont support full action set
        avail = [1] * action_n

        (
            NO_OP,
            LEFT,
            TOP_LEFT,
            TOP,
            TOP_RIGHT,
            RIGHT,
            BOTTOM_RIGHT,
            BOTTOM,
            BOTTOM_LEFT,
            LONG_PASS,
            HIGH_PASS,
            SHORT_PASS,
            SHOT,
            SPRINT,
            RELEASE_MOVE,
            RELEASE_SPRINT,
            SLIDE,
            DRIBBLE,
            RELEASE_DRIBBLE,
        ) = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)

        if action_n == 20:
            BUILTIN_AI = 19

        if obs["ball_owned_team"] == 1:  # opponents owning ball
            pass

        elif obs["ball_owned_team"] == 0:
            # my team owning ball
            avail[SLIDE] = 0

        # if too far, no shot
        ball_x, ball_y, _ = obs["ball"]

        if obs["game_mode"] == 6 and ball_x > 0.6:  # Our PenaltyKick
            # avail = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            avail = [1] + [0] * (action_n - 1)
            avail[SHOT] = 1
            return np.array(avail)

        return np.array(avail)

    def _encode_ball_which_zone(self, ball_x, ball_y):
        MIDDLE_X, PENALTY_X, END_X = 0.2, 0.64, 1.0
        PENALTY_Y, END_Y = 0.27, 0.42
        if (-END_X <= ball_x and ball_x < -PENALTY_X) and (
                -PENALTY_Y < ball_y and ball_y < PENALTY_Y
        ):
            return [1.0, 0, 0, 0, 0, 0]
        elif (-END_X <= ball_x and ball_x < -MIDDLE_X) and (
                -END_Y < ball_y and ball_y < END_Y
        ):
            return [0, 1.0, 0, 0, 0, 0]
        elif (-MIDDLE_X <= ball_x and ball_x <= MIDDLE_X) and (
                -END_Y < ball_y and ball_y < END_Y
        ):
            return [0, 0, 1.0, 0, 0, 0]
        elif (PENALTY_X < ball_x and ball_x <= END_X) and (
                -PENALTY_Y < ball_y and ball_y < PENALTY_Y
        ):
            return [0, 0, 0, 1.0, 0, 0]
        elif (MIDDLE_X < ball_x and ball_x <= END_X) and (
                -END_Y < ball_y and ball_y < END_Y
        ):
            return [0, 0, 0, 0, 1.0, 0]
        else:
            return [0, 0, 0, 0, 0, 1.0]

    def _encode_role_onehot(self, role_num):
        result = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        result[role_num] = 1.0
        return np.array(result)

    def _get_available_actions_gramma(self, his_actions, action_n):
        (
            NO_OP,
            LEFT,
            TOP_LEFT,
            TOP,
            TOP_RIGHT,
            RIGHT,
            BOTTOM_RIGHT,
            BOTTOM,
            BOTTOM_LEFT,
            LONG_PASS,
            HIGH_PASS,
            SHORT_PASS,
            SHOT,
            SPRINT,
            RELEASE_MOVE,
            RELEASE_SPRINT,
            SLIDE,
            DRIBBLE,
            RELEASE_DRIBBLE,
        ) = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)
        avail = np.zeros(action_n)
        avail[13:] = 1

        directions = [LEFT, TOP_LEFT, TOP, TOP_RIGHT, RIGHT, BOTTOM_RIGHT, BOTTOM_LEFT, BOTTOM]
        if len(his_actions) == 0:
            self.set_on(avail, directions)
        else:
            last_action = his_actions[-1]
            # directions
            if last_action in directions:
                self.set_on(avail, self._get_smooth_directions(his_actions) + [RELEASE_MOVE])
            elif last_action in [LONG_PASS, SHOT, HIGH_PASS, SHORT_PASS, SLIDE]:
                self.set_on(avail, directions + [last_action])
            # we regard release move as an end of a series of commands
            else:
                avail = np.ones(action_n)
                avail[0] = 0

        ret = np.array(avail)
        return ret

    def get_available_actions(self, obs, ball_distance, his_actions):
        # todo further restrict it by automata
        avail1 = self._get_avail_new(obs, ball_distance, self.action_n)
        if self.use_action_gramma:
            avail2 = self._get_available_actions_gramma(his_actions, self.action_n)
            avail = np.minimum(avail1, avail2)
            if np.sum(avail) == 0:
                # logger.warn("no available actions!")
                avail = avail1
        else:
            avail = avail1
        return avail

    def set_on(self, avail, args):
        avail[args] = 1

    def set_off(self, avail, args):
        avail[args] = 0

    def _get_smooth_directions(self, his_actions):
        (
            NO_OP,
            LEFT,
            TOP_LEFT,
            TOP,
            TOP_RIGHT,
            RIGHT,
            BOTTOM_RIGHT,
            BOTTOM,
            BOTTOM_LEFT,
            LONG_PASS,
            HIGH_PASS,
            SHORT_PASS,
            SHOT,
            SPRINT,
            RELEASE_MOVE,
            RELEASE_SPRINT,
            SLIDE,
            DRIBBLE,
            RELEASE_DRIBBLE,
        ) = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)
        last_action = his_actions[-1]
        # last2_action=his_actions[-2]
        assert last_action >= 1 and last_action <= 8
        # thenext direction is allow to be 90 degrees away,totally 5 actions
        s = (last_action + 5) % 8
        avail_ids = np.arange(s, s + 5) % 8 + 1
        return list(avail_ids)