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


""""""
import os
import sys
from pathlib import Path
import numpy as np
import torch

base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))

from tartrl_policy import PolicyNetwork
from tartrl_utils import tartrl_obs_deal, _t2n
from goal_keeper import agent_get_action

class TARTRLAgent():
    def __init__(self):
        rnn_shape = [1,1,1,512]
        self.rnn_hidden_state = [np.zeros(rnn_shape, dtype=np.float32) for _ in range (11)]
        self.model = PolicyNetwork()
        self.model.load_state_dict(torch.load( os.path.dirname(os.path.abspath(__file__)) + '/actor.pt', map_location=torch.device("cpu")))
        self.model.eval()

    def get_action(self,raw_obs,idx):
        # if idx == 0:
        #     re_action = [[0]*19]
        #     re_action_index = agent_get_action(raw_obs)[0]
        #     re_action[0][re_action_index] = 1
        #     return re_action

        tartrl_obs = tartrl_obs_deal(raw_obs)

        obs = tartrl_obs['obs']
        obs = np.concatenate(obs.reshape(1, 1, 330))
        rnn_hidden_state = np.concatenate(self.rnn_hidden_state[idx])
        avail_actions = np.zeros(20)
        avail_actions[:19] = tartrl_obs['available_action']
        avail_actions = np.concatenate(avail_actions.reshape([1, 1, 20]))
        with torch.no_grad():
            actions, rnn_hidden_state = self.model(obs, rnn_hidden_state, available_actions=avail_actions, deterministic=True)
        if actions[0][0] == 17 and raw_obs["sticky_actions"][8] == 1:
            actions[0][0] = 15
        self.rnn_hidden_state[idx] = np.array(np.split(_t2n(rnn_hidden_state), 1))

        re_action = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        re_action[0][actions[0]] = 1

        return re_action

agent = TARTRLAgent()

def my_controller(obs_list, action_space_list, is_act_continuous=False):
    idx = obs_list['controlled_player_index'] % 11
    del obs_list['controlled_player_index']
    action = agent.get_action(obs_list,idx)
    return action

def jidi_controller(obs_list=None):
    if obs_list is None:
        return
    #重命名，防止加载错误
    re = my_controller(obs_list,None)
    assert isinstance(re,list)
    assert isinstance(re[0],list)
    return re