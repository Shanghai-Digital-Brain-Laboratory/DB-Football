#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2023 The TARTRL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

# Area.
THIRD_X = 0.3
BOX_X = 0.7
MAX_X = 1.0
BOX_Y = 0.24
MAX_Y = 0.42

# Actions.
IDLE = 0
LEFT = 1
TOP_LEFT = 2
TOP = 3
TOP_RIGHT = 4
RIGHT = 5
BOTTOM_RIGHT = 6
BOTTOM = 7
BOTTOM_LEFT = 8
LONG_PASS = 9
HIGH_PASS = 10
SHORT_PASS = 11
SHOT = 12
SPRINT = 13
RELEASE_DIRECTION = 14
RELEASE_SPRINT = 15
SLIDING = 16
DRIBBLE = 17
RELEASE_DRIBBLE = 18
STICKY_LEFT = 0
STICKY_TOP_LEFT = 1
STICKY_TOP = 2
STICKY_TOP_RIGHT = 3
STICKY_RIGHT = 4
STICKY_BOTTOM_RIGHT = 5
STICKY_BOTTOM = 6
STICKY_BOTTOM_LEFT = 7

RIGHT_ACTIONS = [TOP_RIGHT, RIGHT, BOTTOM_RIGHT, TOP, BOTTOM]
LEFT_ACTIONS = [TOP_LEFT, LEFT, BOTTOM_LEFT, TOP, BOTTOM]
BOTTOM_ACTIONS = [BOTTOM_LEFT, BOTTOM, BOTTOM_RIGHT, LEFT, RIGHT]
TOP_ACTIONS = [TOP_LEFT, TOP, TOP_RIGHT, LEFT, RIGHT]
ALL_DIRECTION_ACTIONS = [LEFT, TOP_LEFT, TOP, TOP_RIGHT, RIGHT, BOTTOM_RIGHT, BOTTOM, BOTTOM_LEFT]
ALL_DIRECTION_VECS = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]

def get_direction_action(available_action, sticky_actions, forbidden_action, target_action, active_direction, need_sprint):
    available_action = np.zeros(19)
    available_action[forbidden_action] = 0
    available_action[target_action] = 1

    if need_sprint:
        available_action[RELEASE_SPRINT] = 0
        if sticky_actions[8] == 0:
            available_action = np.zeros(19)
            available_action[SPRINT] = 1
    else:
        available_action[SPRINT] = 0
        if sticky_actions[8] == 1:
            available_action = np.zeros(19)
            available_action[RELEASE_SPRINT] = 1
    return available_action

def tartrl_obs_deal(obs):

    direction_x_bound = 0.03
    direction_y_bound = 0.02
    ball_direction_x_bound = 0.15
    ball_direction_y_bound = 0.07
    ball_direction_z_bound = 4
    ball_rotation_x_bound = 0.0005
    ball_rotation_y_bound = 0.0004
    ball_rotation_z_bound = 0.015
    active_id = [obs["active"]]
    assert active_id[0] < 11 and active_id[0] >= 0, "active id is wrong, active id = {}".format(active_id[0])
    # left team 88
    left_position = np.concatenate(obs["left_team"])
    left_direction = np.concatenate(obs["left_team_direction"])
    left_tired_factor = obs["left_team_tired_factor"]
    left_yellow_card = obs["left_team_yellow_card"]
    left_red_card = ~obs["left_team_active"]
    left_offside = np.zeros(11)
    if obs["ball_owned_team"] == 0:
        left_offside_line = max(0, obs["ball"][0], np.sort(obs["right_team"][:, 0])[-2])
        left_offside = obs["left_team"][:, 0] > left_offside_line
        left_offside[obs["ball_owned_player"]] = False

    new_left_direction = left_direction.copy()
    for counting in range(len(new_left_direction)):
        new_left_direction[counting] = new_left_direction[counting] / direction_x_bound if counting % 2 == 0 else new_left_direction[counting] / direction_y_bound

    left_team = np.concatenate([
        left_position,
        new_left_direction,
        left_tired_factor,
        left_yellow_card,
        left_red_card,
        left_offside,
    ]).astype(np.float64)

    # right team 88
    right_position = np.concatenate(obs["right_team"])
    right_direction = np.concatenate(obs["right_team_direction"])
    right_tired_factor = obs["right_team_tired_factor"]
    right_yellow_card = obs["right_team_yellow_card"]
    right_red_card = ~obs["right_team_active"]
    right_offside = np.zeros(11)
    if obs["ball_owned_team"] == 1:
        right_offside_line = min(0, obs["ball"][0], np.sort(obs["left_team"][:, 0])[1])
        right_offside = obs["right_team"][:, 0] < right_offside_line
        right_offside[obs["ball_owned_player"]] = False

    new_right_direction = right_direction.copy()
    for counting in range(len(new_right_direction)):
        new_right_direction[counting] = new_right_direction[counting] / direction_x_bound if counting % 2 == 0 else new_right_direction[counting] / direction_y_bound

    right_team = np.concatenate([
        right_position,
        new_right_direction,
        right_tired_factor,
        right_yellow_card,
        right_red_card,
        right_offside,
    ]).astype(np.float64)

    # active 18
    sticky_actions = obs["sticky_actions"][:10]
    active_position = obs["left_team"][obs["active"]]
    active_direction = obs["left_team_direction"][obs["active"]]
    active_tired_factor = left_tired_factor[obs["active"]]
    active_yellow_card = left_yellow_card[obs["active"]]
    active_red_card = left_red_card[obs["active"]]
    active_offside = left_offside[obs["active"]]

    new_active_direction = active_direction.copy()
    new_active_direction[0] /= direction_x_bound
    new_active_direction[1] /= direction_y_bound

    active_player = np.concatenate([
        sticky_actions,
        active_position,
        new_active_direction,
        [active_tired_factor],
        [active_yellow_card],
        [active_red_card],
        [active_offside],
    ]).astype(np.float64)

    # relative 69
    relative_ball_position = obs["ball"][:2] - active_position
    distance2ball = np.linalg.norm(relative_ball_position)
    relative_left_position = obs["left_team"] - active_position
    distance2left = np.linalg.norm(relative_left_position, axis=1)
    relative_left_position = np.concatenate(relative_left_position)
    relative_right_position = obs["right_team"] - active_position
    distance2right = np.linalg.norm(relative_right_position, axis=1)
    relative_right_position = np.concatenate(relative_right_position)
    relative_info = np.concatenate([
        relative_ball_position,
        [distance2ball],
        relative_left_position,
        distance2left,
        relative_right_position,
        distance2right,
    ]).astype(np.float64)

    active_info = np.concatenate([active_player, relative_info])    # 87

    # ball info 12
    ball_owned_team = np.zeros(3)
    ball_owned_team[obs["ball_owned_team"] + 1] = 1.0
    new_ball_direction = obs["ball_direction"].copy()
    new_ball_rotation = obs['ball_rotation'].copy()
    for counting in range(len(new_ball_direction)):
        if counting % 3 == 0:
            new_ball_direction[counting] /= ball_direction_x_bound
            new_ball_rotation[counting] /= ball_rotation_x_bound
        if counting % 3 == 1:
            new_ball_direction[counting] /= ball_direction_y_bound
            new_ball_rotation[counting] /= ball_rotation_y_bound
        if counting % 3 == 2:
            new_ball_direction[counting] /= ball_direction_z_bound
            new_ball_rotation[counting] /= ball_rotation_z_bound
    ball_info = np.concatenate([
        obs["ball"],
        new_ball_direction,
        ball_owned_team,
        new_ball_rotation
    ]).astype(np.float64)
    # ball owned player 23
    ball_owned_player = np.zeros(23)
    if obs["ball_owned_team"] == 1:     # 对手
        ball_owned_player[11 + obs['ball_owned_player']] = 1.0
        ball_owned_player_pos = obs['right_team'][obs['ball_owned_player']]
        ball_owned_player_direction = obs["right_team_direction"][obs['ball_owned_player']]
        ball_owner_tired_factor = right_tired_factor[obs['ball_owned_player']]
        ball_owner_yellow_card = right_yellow_card[obs['ball_owned_player']]
        ball_owner_red_card = right_red_card[obs['ball_owned_player']]
        ball_owner_offside = right_offside[obs['ball_owned_player']]
    elif obs["ball_owned_team"] == 0:
        ball_owned_player[obs['ball_owned_player']] = 1.0
        ball_owned_player_pos = obs['left_team'][obs['ball_owned_player']]
        ball_owned_player_direction = obs["left_team_direction"][obs['ball_owned_player']]
        ball_owner_tired_factor = left_tired_factor[obs['ball_owned_player']]
        ball_owner_yellow_card = left_yellow_card[obs['ball_owned_player']]
        ball_owner_red_card = left_red_card[obs['ball_owned_player']]
        ball_owner_offside = left_offside[obs['ball_owned_player']]
    else:
        ball_owned_player[-1] = 1.0
        ball_owned_player_pos = np.zeros(2)
        ball_owned_player_direction = np.zeros(2)

    relative_ball_owner_position = np.zeros(2)
    distance2ballowner = np.zeros(1)
    ball_owner_info = np.zeros(4)
    if obs["ball_owned_team"] != -1:
        relative_ball_owner_position = ball_owned_player_pos - active_position
        distance2ballowner = [np.linalg.norm(relative_ball_owner_position)]
        ball_owner_info = np.concatenate([
            [ball_owner_tired_factor],
            [ball_owner_yellow_card],
            [ball_owner_red_card],
            [ball_owner_offside]
        ])

    new_ball_owned_player_direction = ball_owned_player_direction.copy()
    new_ball_owned_player_direction[0] /= direction_x_bound
    new_ball_owned_player_direction[1] /= direction_y_bound

    ball_own_active_info = np.concatenate([
        ball_info,      # 12
        ball_owned_player,  # 23
        active_position,    # 2
        new_active_direction,    # 2
        [active_tired_factor],    # 1
        [active_yellow_card],    # 1
        [active_red_card],    # 1
        [active_offside],    # 1
        relative_ball_position,    # 2
        [distance2ball],    # 1
        ball_owned_player_pos,    # 2
        new_ball_owned_player_direction,    # 2
        relative_ball_owner_position,    # 2
        distance2ballowner,    # 1
        ball_owner_info # 4
    ])

    # match state
    game_mode = np.zeros(7)
    game_mode[obs["game_mode"]] = 1.0
    goal_diff_ratio = (obs["score"][0] - obs["score"][1]) / 5
    steps_left_ratio = obs["steps_left"] / 3001
    match_state = np.concatenate([
        game_mode,
        [goal_diff_ratio],
        [steps_left_ratio],
    ]).astype(np.float64)

    # available action
    available_action = np.ones(19)
    available_action[IDLE] = 0
    available_action[RELEASE_DIRECTION] = 0
    should_left = False


    if obs["game_mode"] == 0:
        active_x = active_position[0]
        counting_right_enemy_num = 0
        counting_right_teammate_num = 0
        counting_left_teammate_num = 0
        for enemy_pos in obs["right_team"][1:]:
            if active_x < enemy_pos[0]:
                counting_right_enemy_num += 1
        for teammate_pos in obs["left_team"][1:]:
            if active_x < teammate_pos[0]:
                counting_right_teammate_num += 1
            if active_x > teammate_pos[0]:
                counting_left_teammate_num += 1
        
        if active_x > obs['ball'][0] + 0.05:

            if counting_left_teammate_num < 2:

                if obs['ball_owned_team'] != 0:
                    should_left = True
    if should_left:
        available_action = get_direction_action(available_action, sticky_actions, RIGHT_ACTIONS, [LEFT, BOTTOM_LEFT, TOP_LEFT], active_direction, True)


    if (abs(relative_ball_position[0]) > 0.75 or abs(relative_ball_position[1]) > 0.5):
        all_directions_vecs = [np.array(v) / np.linalg.norm(np.array(v)) for v in ALL_DIRECTION_VECS]
        best_direction = np.argmax([np.dot(relative_ball_position, v) for v in all_directions_vecs])
        target_direction = ALL_DIRECTION_ACTIONS[best_direction]
        forbidden_actions = ALL_DIRECTION_ACTIONS.copy()
        forbidden_actions.remove(target_direction)
        available_action = get_direction_action(available_action, sticky_actions, forbidden_actions, [target_direction], active_direction, True)


    if_i_hold_ball = (obs["ball_owned_team"] == 0 and obs["ball_owned_player"] == obs['active'])
    ball_pos_offset = 0.05
    no_ball_pos_offset = 0.03

    active_x, active_y = active_position[0], active_position[1]
    if_outside = False
    if active_x <= (-1 + no_ball_pos_offset) or (if_i_hold_ball and active_x <= (-1 + ball_pos_offset)):
        if_outside = True
        action_index = LEFT_ACTIONS
        target_direction = RIGHT
    elif active_x >= (1 - no_ball_pos_offset) or (if_i_hold_ball and active_x >= (1 - ball_pos_offset)):
        if_outside = True
        action_index = RIGHT_ACTIONS
        target_direction = LEFT
    elif active_y >= (0.42 - no_ball_pos_offset) or (if_i_hold_ball and active_y >= (0.42 - ball_pos_offset)):
        if_outside = True
        action_index = BOTTOM_ACTIONS
        target_direction = TOP
    elif active_y <= (-0.42 + no_ball_pos_offset) or (if_i_hold_ball and active_x <= (-0.42 + ball_pos_offset)):
        if_outside = True
        action_index = TOP_ACTIONS
        target_direction = BOTTOM
    if obs["game_mode"] in [1, 2, 3, 4, 5]:
        left2ball = np.linalg.norm(obs["left_team"] - obs["ball"][:2], axis=1)
        right2ball = np.linalg.norm(obs["right_team"] - obs["ball"][:2], axis=1)
        if np.min(left2ball) < np.min(right2ball) and obs["active"] == np.argmin(left2ball):
            if_outside = False
    elif obs["game_mode"] in [6]:
        if obs["ball"][0] > 0 and active_position[0] > BOX_X:
            if_outside = False
    if if_outside:
        available_action = get_direction_action(available_action, sticky_actions, action_index, [target_direction], active_direction, False)

    if np.sum(sticky_actions[:8]) == 0:
        available_action[RELEASE_DIRECTION] = 0
    if sticky_actions[8] == 0:
        available_action[RELEASE_SPRINT] = 0
    else:
        available_action[SPRINT] = 0
    if sticky_actions[9] == 0:
        available_action[RELEASE_DRIBBLE] = 0
    else:
        available_action[DRIBBLE] = 0
    if active_position[0] < 0.4 or abs(active_position[1]) > 0.3:
        available_action[SHOT] = 0

    if obs["game_mode"] == 0:
        if obs["ball_owned_team"] == -1:
            available_action[DRIBBLE] = 0
            if distance2ball >= 0.05:
                available_action[SLIDING] = 0
                available_action[[LONG_PASS, HIGH_PASS, SHORT_PASS, SHOT]] = 0
        elif obs["ball_owned_team"] == 0:
            available_action[SLIDING] = 0
            if distance2ball >= 0.05:
                available_action[[LONG_PASS, HIGH_PASS, SHORT_PASS, SHOT, DRIBBLE]] = 0
        elif obs["ball_owned_team"] == 1:
            available_action[DRIBBLE] = 0
            if distance2ball >= 0.05:
                available_action[[LONG_PASS, HIGH_PASS, SHORT_PASS, SHOT, SLIDING]] = 0
    elif obs["game_mode"] in [1, 2, 3, 4, 5]:
        left2ball = np.linalg.norm(obs["left_team"] - obs["ball"][:2], axis=1)
        right2ball = np.linalg.norm(obs["right_team"] - obs["ball"][:2], axis=1)
        if np.min(left2ball) < np.min(right2ball) and obs["active"] == np.argmin(left2ball):
            available_action[[SPRINT, RELEASE_SPRINT, SLIDING, DRIBBLE, RELEASE_DRIBBLE]] = 0
        else:
            available_action[[LONG_PASS, HIGH_PASS, SHORT_PASS, SHOT]] = 0
            available_action[[SLIDING, DRIBBLE, RELEASE_DRIBBLE]] = 0
    elif obs["game_mode"] == 6:
        if obs["ball"][0] > 0 and active_position[0] > BOX_X:
            available_action[[LONG_PASS, HIGH_PASS, SHORT_PASS]] = 0
            available_action[[SPRINT, RELEASE_SPRINT, SLIDING, DRIBBLE, RELEASE_DRIBBLE]] = 0
        else:
            available_action[[LONG_PASS, HIGH_PASS, SHORT_PASS, SHOT]] = 0
            available_action[[SLIDING, DRIBBLE, RELEASE_DRIBBLE]] = 0


    obs = np.concatenate([
            active_id, # 1
            active_info, # 87
            ball_own_active_info,   # 57
            left_team,  # 88
            right_team, # 88
            match_state, # 9
        ])

    share_obs = np.concatenate([
            ball_info, # 12
            ball_owned_player,   # 23
            left_team, # 88
            right_team, # 88
            match_state, # 9
        ])

    assert available_action.sum() > 0
    return dict(
        obs=obs,
        share_obs=share_obs,
        available_action=available_action,
    )


def _t2n(x):
    return x.detach().cpu().numpy()




