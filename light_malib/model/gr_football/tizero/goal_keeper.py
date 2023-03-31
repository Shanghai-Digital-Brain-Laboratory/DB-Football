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

# original code from https://github.com/Sarvar-Anvarov/Google-Research-Football/blob/main/gfootball.py
# modified by TARTRL team

import math
import random
import numpy as np

from functools import wraps
from enum import Enum
from typing import *



class Action(Enum):
    Idle = 0
    Left = 1
    TopLeft = 2
    Top = 3
    TopRight = 4
    Right = 5
    BottomRight = 6
    Bottom = 7
    BottomLeft = 8
    LongPass= 9
    HighPass = 10
    ShortPass = 11
    Shot = 12
    Sprint = 13
    ReleaseDirection = 14
    ReleaseSprint = 15
    Slide = 16
    Dribble = 17
    ReleaseDribble = 18


ALL_DIRECTION_ACTIONS = [Action.Left, Action.TopLeft, Action.Top, Action.TopRight, Action.Right, Action.BottomRight, Action.Bottom, Action.BottomLeft]
ALL_DIRECTION_VECS = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]

sticky_index_to_action = [
    Action.Left,
    Action.TopLeft,
    Action.Top,
    Action.TopRight,
    Action.Right,
    Action.BottomRight,
    Action.Bottom,
    Action.BottomLeft,
    Action.Sprint,
    Action.Dribble
]

GOAL_BIAS = 0.01

class PlayerRole(Enum):
    GoalKeeper = 0
    CenterBack = 1
    LeftBack = 2
    RightBack = 3
    DefenceMidfield = 4
    CentralMidfield = 5
    LeftMidfield = 6
    RIghtMidfield = 7
    AttackMidfield = 8
    CentralFront = 9


class GameMode(Enum):
    Normal = 0
    KickOff = 1
    GoalKick = 2
    FreeKick = 3
    Corner = 4
    ThrowIn = 5
    Penalty = 6


def human_readable_agent(agent: Callable[[Dict], Action]):
    """
    Decorator allowing for more human-friendly implementation of the agent function.
    @human_readable_agent
    def my_agent(obs):
        ...
        return football_action_set.action_right
    """
    @wraps(agent)
    def agent_wrapper(obs) -> List[int]:
        # Extract observations for the first (and only) player we control.
        # obs = obs['players_raw'][0]
        # Turn 'sticky_actions' into a set of active actions (strongly typed).
        obs['sticky_actions'] = { sticky_index_to_action[nr] for nr, action in enumerate(obs['sticky_actions']) if action }
        # Turn 'game_mode' into an enum.
        obs['game_mode'] = GameMode(obs['game_mode'])
        # In case of single agent mode, 'designated' is always equal to 'active'.
        if 'designated' in obs:
            del obs['designated']
        # Conver players' roles to enum.
        obs['left_team_roles'] = [ PlayerRole(role) for role in obs['left_team_roles'] ]
        obs['right_team_roles'] = [ PlayerRole(role) for role in obs['right_team_roles'] ]

        action = agent(obs)
        return [action.value]

    return agent_wrapper

def find_patterns(obs, player_x, player_y):
    """ find list of appropriate patterns in groups of memory patterns """
    for get_group in groups_of_memory_patterns:
        group = get_group(obs, player_x, player_y)
        if group["environment_fits"](obs, player_x, player_y):
            return group["get_memory_patterns"](obs, player_x, player_y)

        
def get_action_of_agent(obs, player_x, player_y):
    """ get action of appropriate pattern in agent's memory """
    memory_patterns = find_patterns(obs, player_x, player_y)
    # find appropriate pattern in list of memory patterns
    for get_pattern in memory_patterns:
        pattern = get_pattern(obs, player_x, player_y)
        if pattern["environment_fits"](obs, player_x, player_y):
            return pattern["get_action"](obs, player_x, player_y)

        
def get_distance(x1, y1, right_team):
    """ get two-dimensional Euclidean distance, considering y size of the field """
    return math.sqrt((x1 - right_team[0]) ** 2 + (y1 * 2.38 - right_team[1] * 2.38) ** 2)


def run_to_ball_bottom(obs, player_x, player_y):
    """ run to the ball if it is to the bottom from player's position """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # ball is to the bottom from player's position
        if (obs["ball"][1] > player_y and
                abs(obs["ball"][0] - player_x) < 0.01):
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        return Action.Bottom
    
    return {"environment_fits": environment_fits, "get_action": get_action}


def run_to_ball_bottom_left(obs, player_x, player_y):
    """ run to the ball if it is to the bottom left from player's position """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # ball is to the bottom left from player's position
        if (obs["ball"][0] < player_x and
                obs["ball"][1] > player_y):
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        return Action.BottomLeft
    
    return {"environment_fits": environment_fits, "get_action": get_action}


def run_to_ball_bottom_right(obs, player_x, player_y):
    """ run to the ball if it is to the bottom right from player's position """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # ball is to the bottom right from player's position
        if (obs["ball"][0] > player_x and
                obs["ball"][1] > player_y):
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        return Action.BottomRight
    
    return {"environment_fits": environment_fits, "get_action": get_action}


def run_to_ball_left(obs, player_x, player_y):
    """ run to the ball if it is to the left from player's position """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # ball is to the left from player's position
        if (obs["ball"][0] < player_x and
                abs(obs["ball"][1] - player_y) < 0.01):
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        return Action.Left
    
    return {"environment_fits": environment_fits, "get_action": get_action}


def run_to_ball_right(obs, player_x, player_y):
    """ run to the ball if it is to the right from player's position """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # ball is to the right from player's position
        if (obs["ball"][0] > player_x and
                abs(obs["ball"][1] - player_y) < 0.01):
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        return Action.Right
    
    return {"environment_fits": environment_fits, "get_action": get_action}


def run_to_ball_top(obs, player_x, player_y):
    """ run to the ball if it is to the top from player's position """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # ball is to the top from player's position
        if (obs["ball"][1] < player_y and
                abs(obs["ball"][0] - player_x) < 0.01):
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        return Action.Top
    
    return {"environment_fits": environment_fits, "get_action": get_action}


def run_to_ball_top_left(obs, player_x, player_y):
    """ run to the ball if it is to the top left from player's position """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # ball is to the top left from player's position
        if (obs["ball"][0] < player_x and
                obs["ball"][1] < player_y):
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        return Action.TopLeft
    
    return {"environment_fits": environment_fits, "get_action": get_action}


def run_to_ball_top_right(obs, player_x, player_y):
    """ run to the ball if it is to the top right from player's position """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # ball is to the top right from player's position
        if (obs["ball"][0] > player_x and
                obs["ball"][1] < player_y):
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        return Action.TopRight
    
    return {"environment_fits": environment_fits, "get_action": get_action}


def idle(obs, player_x, player_y):
    """ do nothing, release all sticky actions """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        return True
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        return Action.Idle
    
    return {"environment_fits": environment_fits, "get_action": get_action}
 
    
def start_sprinting(obs, player_x, player_y):
    """ make sure player is sprinting """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        if Action.Sprint not in obs["sticky_actions"]:
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        if Action.Dribble in obs['sticky_actions']:
            return Action.ReleaseDribble
        return Action.Sprint
    
    return {"environment_fits": environment_fits, "get_action": get_action}


def corner(obs, player_x, player_y):
    """ perform a shot in corner game mode """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # it is corner game mode
        if obs['game_mode'] == GameMode.Corner:
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        if player_y > 0:
            if Action.TopRight not in obs["sticky_actions"]:
                return Action.TopRight
        else:
            if Action.BottomRight not in obs["sticky_actions"]:
                return Action.BottomRight
        return Action.HighPass
    
    return {"environment_fits": environment_fits, "get_action": get_action}


def free_kick(obs, player_x, player_y):
    """ perform a high pass or a shot in free kick game mode """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # it is free kick game mode
        if obs['game_mode'] == GameMode.FreeKick:
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        # shot if player close to goal
        if player_x > 0.5:
            if player_y > 0:
                if Action.TopRight not in obs["sticky_actions"]:
                    return Action.TopRight
            else:
                if Action.BottomRight not in obs["sticky_actions"]:
                    return Action.BottomRight
            return Action.Shot
        # high pass if player far from goal
        else:
            if player_y > 0:
                if Action.BottomRight not in obs["sticky_actions"]:
                    return Action.BottomRight
            else:
                if Action.TopRight not in obs['sticky_actions']:
                    return Action.TopRight
            return Action.ShortPass
    
    return {"environment_fits": environment_fits, "get_action": get_action}


def goal_kick(obs, player_x, player_y):
    """ perform a short pass in goal kick game mode """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # it is goal kick game mode
        if obs['game_mode'] == GameMode.GoalKick:
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        if Action.BottomRight not in obs["sticky_actions"]:
            return Action.BottomRight
        return Action.ShortPass
    
    return {"environment_fits": environment_fits, "get_action": get_action}


def kick_off(obs, player_x, player_y):
    """ perform a short pass in kick off game mode """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # it is kick off game mode
        if obs['game_mode'] == GameMode.KickOff:
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        if player_y > 0:
            if Action.Top not in obs["sticky_actions"]:
                return Action.Top
        else:
            if Action.Bottom not in obs["sticky_actions"]:
                return Action.Bottom
        return Action.ShortPass
    
    return {"environment_fits": environment_fits, "get_action": get_action}


def penalty(obs, player_x, player_y):
    """ perform a shot in penalty game mode """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # it is penalty game mode
        if obs['game_mode'] == GameMode.Penalty:
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        if (random.random() < 0.5 and
                Action.TopRight not in obs["sticky_actions"] and
                Action.BottomRight not in obs["sticky_actions"]):
            return Action.TopRight
        else:
            if Action.BottomRight not in obs["sticky_actions"]:
                return Action.BottomRight
        return Action.Shot
    
    return {"environment_fits": environment_fits, "get_action": get_action}

def throw_in(obs, player_x, player_y):
    """ perform a short pass in throw in game mode """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # it is throw in game mode
        if obs['game_mode'] == GameMode.ThrowIn:
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        if Action.Right not in obs["sticky_actions"]:
            return Action.Right
        return Action.ShortPass
    
    return {"environment_fits": environment_fits, "get_action": get_action}


def defence_memory_patterns(obs, player_x, player_y):
    """ group of memory patterns for environments in which opponent's team has the ball """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # player don't have the ball
        if obs["ball_owned_team"] != 0:
            return True
        return False
        
    def get_memory_patterns(obs, player_x, player_y):
        """ get list of memory patterns """
        # shift ball position
        obs["ball"][0] += obs["ball_direction"][0] * 7
        obs["ball"][1] += obs["ball_direction"][1] * 3
        # if opponent has the ball and is far from y axis center
        if abs(obs["ball"][1]) > 0.07 and obs["ball_owned_team"] == 1:
            obs["ball"][0] -= 0.01
            if obs["ball"][1] > 0:
                obs["ball"][1] -= 0.01
            else:
                obs["ball"][1] += 0.01
            
        memory_patterns = [
            start_sprinting,
            run_to_ball_right,
            run_to_ball_left,
            run_to_ball_bottom,
            run_to_ball_top,
            run_to_ball_top_right,
            run_to_ball_top_left,
            run_to_ball_bottom_right,
            run_to_ball_bottom_left,
            idle
        ]
        return memory_patterns
        
    return {"environment_fits": environment_fits, "get_memory_patterns": get_memory_patterns}

def goalkeeper_memory_patterns(obs, player_x, player_y):
    """ group of memory patterns for goalkeeper """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # player is a goalkeeper have the ball
        if (obs["ball_owned_player"] == obs["active"] and
                obs["ball_owned_team"] == 0 and
                obs["ball_owned_player"] == 0):
            return True
        return False
        
    def get_memory_patterns(obs, player_x, player_y):
        """ get list of memory patterns """
        memory_patterns = [
            long_pass_forward,
            idle
        ]
        return memory_patterns
        
    return {"environment_fits": environment_fits, "get_memory_patterns": get_memory_patterns}


def offence_memory_patterns(obs, player_x, player_y):
    """ group of memory patterns for environments in which player's team has the ball """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # player have the ball
        if obs["ball_owned_player"] == obs["active"] and obs["ball_owned_team"] == 0:
            return True
        return False
        
    def get_memory_patterns(obs, player_x, player_y):
        """ get list of memory patterns """
        memory_patterns = [
            close_to_goalkeeper_shot,
            spot_shot,
            cross,
            long_pass_forward,
            keep_the_ball,
        idle
        ]
        return memory_patterns
        
    return {"environment_fits": environment_fits, "get_memory_patterns": get_memory_patterns}


def other_memory_patterns(obs, player_x, player_y):
    """ group of memory patterns for all other environments """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        return True
        
    def get_memory_patterns(obs, player_x, player_y):
        """ get list of memory patterns """
        memory_patterns = [
            idle
        ]
        return memory_patterns
        
    return {"environment_fits": environment_fits, "get_memory_patterns": get_memory_patterns}

def special_game_modes_memory_patterns(obs, player_x, player_y):
    """ group of memory patterns for special game mode environments """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # if game mode is not normal
        if obs['game_mode'] != GameMode.Normal:
            return True
        return False
        
    def get_memory_patterns(obs, player_x, player_y):
        """ get list of memory patterns """
        memory_patterns = [
            corner,
            free_kick,
            goal_kick,
            kick_off,
            penalty,
            throw_in,
            idle
        ]
        return memory_patterns
        
    return {"environment_fits": environment_fits, "get_memory_patterns": get_memory_patterns}


def special_spot_shot(obs, player_x, player_y):
    """ group of memory patterns for special game mode environments """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # if game mode is not normal
        if player_x > 0.8 and abs(player_y) < 0.21:
            return True
        return False
        
    def get_memory_patterns(obs, player_x, player_y):
        """ get list of memory patterns """
        memory_patterns = [
            shot,
            idle
        ]
        return memory_patterns
        
    return {"environment_fits": environment_fits, "get_memory_patterns": get_memory_patterns}


def own_goal(obs, player_x, player_y):
    """ group of memory patterns for special game mode environments """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # if game mode is not normal
        if player_x < -0.9 and player_y:
            return True
        return False
        
    def get_memory_patterns(obs, player_x, player_y):
        """ get list of memory patterns """
        memory_patterns = [
            own_goal_2
        ]
        return memory_patterns
        
    return {"environment_fits": environment_fits, "get_memory_patterns": get_memory_patterns}

def get_best_direction(obs, target_direction):
    active_position = obs["left_team"][obs["active"]]
    relative_goal_position = np.array(target_direction) - active_position
    all_directions_vecs = [np.array(v) / np.linalg.norm(np.array(v)) for v in ALL_DIRECTION_VECS]
    best_direction = np.argmax([np.dot(relative_goal_position, v) for v in all_directions_vecs])
    return ALL_DIRECTION_ACTIONS[best_direction]

def get_distance2ball(obs):
    return np.linalg.norm(obs["ball"][:2] - obs["left_team"][obs['active']])

def get_target2line(obs):
    active_position = obs["left_team"][obs["active"]]
    ball_x, ball_y = obs['ball'][0], obs['ball'][1]
    distance2goal = ((ball_x + 1) ** 2 + ball_y ** 2) ** 0.5 + 1e-5
    cos_theta = (ball_x + 1) / distance2goal
    sin_theta = ball_y / distance2goal
    target_pos = np.array([0.03 * cos_theta - 1, 0.03 * sin_theta])
    return target_pos

def already_near_goal(obs, player_x, player_y):
    """ do nothing, release all sticky actions """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        active_position = obs["left_team"][obs["active"]]
        relative_goal_position = np.array([-1 + GOAL_BIAS, 0]) - active_position
        distance2goal = np.linalg.norm(relative_goal_position)
        if distance2goal < 0.02:
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        # print(obs["sticky_actions"])
        if Action.Sprint in obs["sticky_actions"]:
            return Action.ReleaseSprint
        if Action.Dribble in obs["sticky_actions"]:
            return Action.ReleaseDribble
        if len(obs["sticky_actions"]) > 0:
            return Action.ReleaseDirection
        return Action.Idle
    
    return {"environment_fits": environment_fits, "get_action": get_action}

def already_in_line(obs, player_x, player_y):
    """ do nothing, release all sticky actions """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        
        target_pos = get_target2line(obs)
        distance2goal = np.linalg.norm(target_pos - obs['left_team'][obs['active']])
        if distance2goal < 0.02:
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        # print(obs["sticky_actions"])
        if Action.Sprint in obs["sticky_actions"]:
            return Action.ReleaseSprint
        if Action.Dribble in obs["sticky_actions"]:
            return Action.ReleaseDribble
        if len(obs["sticky_actions"]) > 0:
            return Action.ReleaseDirection
        return Action.Idle
    
    return {"environment_fits": environment_fits, "get_action": get_action}

def run_to_goal(obs, player_x, player_y):
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        return True

    def get_action(obs, player_x, player_y):
        # active_position = obs["left_team"][obs["active"]]
        # relative_goal_position = np.array([-1 + GOAL_BIAS, 0]) - active_position
        # all_directions_vecs = [np.array(v) / np.linalg.norm(np.array(v)) for v in ALL_DIRECTION_VECS]
        # best_direction = np.argmax([np.dot(relative_goal_position, v) for v in all_directions_vecs])
        # return ALL_DIRECTION_ACTIONS[best_direction]
        return get_best_direction(obs, [-1 + GOAL_BIAS, 0])

    return {"environment_fits": environment_fits, "get_action": get_action}

def run_to_line(obs, player_x, player_y):
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        return True

    def get_action(obs, player_x, player_y):
        target_pos = get_target2line(obs)
        return get_best_direction(obs, target_pos)

    return {"environment_fits": environment_fits, "get_action": get_action}

def goal_keeper_far_pattern(obs, player_x, player_y):
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # player have the ball
        if (obs["active"] == 0):
            active_position = obs["left_team"][0]
            relative_ball_position = obs["ball"][:2] - active_position
            distance2ball = np.linalg.norm(relative_ball_position)
            if distance2ball > 0.5 or (obs['ball_owned_team'] == 0 and obs['ball_owned_player'] != 0):
                return True
            if active_position[0] > -0.7 or abs(active_position[1]) > 0.25:
                for teammate_pos in obs['left_team'][1:]:
                    teammate_dis = np.linalg.norm(obs["ball"][:2] - teammate_pos)
                    if teammate_dis < distance2ball:
                        return True
        return False
        
    def get_memory_patterns(obs, player_x, player_y):
        """ get list of memory patterns """
        memory_patterns = [
            already_near_goal,
            start_sprinting,
            run_to_goal
        ]
        return memory_patterns
        
    return {"environment_fits": environment_fits, "get_memory_patterns": get_memory_patterns}

def ball_distance_2_5(obs, player_x, player_y):
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # player have the ball
        if (obs["active"] == 0 and obs['ball_owned_team'] != 0):
            distance2ball = get_distance2ball(obs)
            if distance2ball <= 0.5 and distance2ball >= 0.2:
                return True
        return False
        
    def get_memory_patterns(obs, player_x, player_y):
        """ get list of memory patterns """
        memory_patterns = [
            already_in_line,
            start_sprinting,
            run_to_line
        ]
        return memory_patterns
        
    return {"environment_fits": environment_fits, "get_memory_patterns": get_memory_patterns}

def ball_distance_close(obs, player_x, player_y):
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # player have the ball
        if (obs["active"] == 0 and obs['ball_owned_team'] != 0):
            distance2ball = get_distance2ball(obs)
            if distance2ball < 0.25:
                return True
        return False
        
    def get_memory_patterns(obs, player_x, player_y):
        """ get list of memory patterns """
        memory_patterns = [
            shot
        ]
        return memory_patterns
        
    return {"environment_fits": environment_fits, "get_memory_patterns": get_memory_patterns}

# list of groups of memory patterns
groups_of_memory_patterns = [
    goal_keeper_far_pattern,        # 安全
    goalkeeper_memory_patterns,     # 守门员持球
    # special_spot_shot,      # 射门 进不去
    special_game_modes_memory_patterns,     # 特殊game mode
    ball_distance_2_5,
    ball_distance_close,
    # own_goal,
    # offence_memory_patterns,        # 我方持球 进不去
    defence_memory_patterns,
    other_memory_patterns       # idle
]


def keep_the_ball(obs, player_x, player_y):
    def environment_fits(obs, player_x, player_y):
        return True
    
    def get_action(obs, player_x, player_y):
        right_team, left_team = obs['right_team'], obs['left_team']
        dist = [get_distance(player_x, player_y, i) for i in right_team]
        closest = right_team[np.argmin(dist)]
        near = [i for i in right_team if (i[0] < player_x + 0.2) and (i[0] > player_x) and (i[1] > player_y - 0.05)
               and (i[1] < player_y + 0.05)] 
        back = [i for i in right_team if (i[0] > player_x)]
        bottom_right = [i for i in left_team if (i[0] > player_x - 0.05) and (i[0] < player_x + 0.2) and (i[1] < player_y + 0.2) and 
                       (i[1] > player_y)]
        top_right = [i for i in left_team if (i[0] > player_x - 0.05) and (i[0] < player_x + 0.2) and (i[1] > player_y - 0.2) and 
                       (i[1] < player_y)]
        bottom_left = [i for i in left_team if (i[0] < player_x) and (i[0] > player_x - 0.2) and (i[1] < player_y + 0.2) and 
                       (i[1] > player_y)]
        top_left = [i for i in left_team if (i[0] < player_x) and (i[0] > player_x - 0.2) and (i[1] > player_y - 0.2) and 
                       (i[1] < player_y)]
        
    
        if len(near) == 0:
            return Action.Right
        
        if player_y > 0:
            if player_y > 0.35:
                return Action.Right
            if len(bottom_right) > 0:
                if Action.BottomRight not in obs['sticky_actions']:
                    return Action.BottomRight
                return Action.ShortPass
            return Action.BottomRight
        
        if player_y < 0:
            if player_y < -0.35:
                return Action.Right
            if len(top_right) > 0:
                if Action.TopRight not in obs['sticky_actions']:
                    return Action.TopRight
                return Action.ShortPass
            return Action.TopRight
            
    return {'environment_fits': environment_fits, 'get_action': get_action}


def spot_shot(obs, player_x, player_y):
    """ shot if close to the goalkeeper """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # shoot if in spotted location
        if player_x > 0.75 and abs(player_y) < 0.21:
            return True
        return False

    
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        if player_y >= 0:
            if Action.TopRight not in obs["sticky_actions"]:
                return Action.TopRight
        else:
            if Action.BottomRight not in obs["sticky_actions"]:
                return Action.BottomRight
        return Action.Shot

    return {"environment_fits": environment_fits, "get_action": get_action}


def cross(obs, player_x, player_y):
    def environment_fits(obs, player_x, player_y):
        if player_x > 0.7 and abs(player_y) > 0.21:
            return True
        return False
    
    def get_action(obs, player_x, player_y):
        
        if player_x > 0.88:
            if player_y > 0:
                if Action.Top not in obs['sticky_actions']:
                    return Action.Top
            else:
                if Action.Bottom not in obs['sticky_actions']:
                    return Action.Bottom
            return Action.HighPass
        
        if player_x > 0.9:
            if (Action.Right in obs['sticky_actions'] or 
                Action.TopRight in obs['sticky_actions'] or 
                Action.BottomRight in obs['sticky_actions']):
                return Action.ReleaseDirection
            if Action.Right not in obs['sticky_actions']:
                if player_y > 0:
                    if Action.Top not in obs['sticky_actions']:
                        return Action.Top
                if player_y < 0:
                    if Action.Bottom not in obs['sticky_actions']:
                        return Action.Bottom
        return Action.HighPass
                
    return {"environment_fits": environment_fits, "get_action": get_action}


def close_to_goalkeeper_shot(obs, player_x, player_y):
    """ shot if close to the goalkeeper """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        goalkeeper_x = obs["right_team"][0][0] + obs["right_team_direction"][0][0] * 13
        goalkeeper_y = obs["right_team"][0][1] + obs["right_team_direction"][0][1] * 13
        goalkeeper = [goalkeeper_x,goalkeeper_y]
        
        if get_distance(player_x, player_y, goalkeeper) < 0.25:
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        if player_y >= 0:
            if Action.TopRight not in obs["sticky_actions"]:
                return Action.TopRight
        else:
            if Action.BottomRight not in obs["sticky_actions"]:
                return Action.BottomRight
        return Action.Shot
    
    return {"environment_fits": environment_fits, "get_action": get_action}


def long_pass_forward(obs, player_x, player_y):
    """ perform a high pass, if far from opponent's goal """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        right_team = obs["right_team"][1:]
        # player have the ball and is far from opponent's goal
        if player_x < -0.4:
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        right_team, left_team = obs['right_team'], obs['left_team']
        dist = [get_distance(player_x, player_y, i) for i in right_team]
        closest = right_team[np.argmin(dist)]
        
        
        if abs(player_y) > 0.22:
            if Action.Right not in obs["sticky_actions"]:
                return Action.Right
            return Action.HighPass
        
        if np.min(dist) > 0.4:
            if player_y > 0:
                return Action.Bottom
            else:
                return Action.Top
            
        if np.min(dist) < 0.4 and np.min(dist) > 0.2:
            if player_y < 0:
                return Action.TopRight
            else:
                return Action.BottomRight
            
        if np.min(dist) < 0.2:
            if Action.Right not in obs['sticky_actions']:
                return Action.Right
            return Action.HighPass
    
    return {"environment_fits": environment_fits, "get_action": get_action}

def shot(obs, player_x, player_y):
    def environment_fits(obs, player_x, player_y):
        return True
    
    def get_action(obs, player_x, player_y):
        # if player_y > 0:
        #     if Action.TopRight not in obs['sticky_actions']:
        #         return Action.TopRight
        # else:
        #     if Action.BottomRight not in obs['sticky_actions']:
        #         return Action.BottomRight
        return Action.Shot
    
    return {"environment_fits": environment_fits, "get_action": get_action}


def own_goal_2(obs, player_x, player_y):
    def environment_fits(obs, player_x, player_y):
        return True
    
    def get_action(obs, player_x, player_y):
        return Action.Shot
    
    return {"environment_fits": environment_fits, "get_action": get_action}


# @human_readable_agent wrapper modifies raw observations 
# provided by the environment:
# https://github.com/google-research/football/blob/master/gfootball/doc/observation.md#raw-observations
# into a form easier to work with by humans.
# Following modifications are applied:
# - Action, PlayerRole and GameMode enums are introduced.
# - 'sticky_actions' are turned into a set of active actions (Action enum)
#    see usage example below.
# - 'game_mode' is turned into GameMode enum.
# - 'designated' field is removed, as it always equals to 'active'
#    when a single player is controlled on the team.
# - 'left_team_roles'/'right_team_roles' are turned into PlayerRole enums.
# - Action enum is to be returned by the agent function.
@human_readable_agent
def agent_get_action(obs):
    """ Ole ole ole ole """
    # dictionary for Memory Patterns data
    obs["memory_patterns"] = {}
    # We always control left team (observations and actions
    # are mirrored appropriately by the environment).
    controlled_player_pos = obs["left_team"][obs["active"]]
    # get action of appropriate pattern in agent's memory
    action = get_action_of_agent(obs, controlled_player_pos[0], controlled_player_pos[1])
    # return action
    return action