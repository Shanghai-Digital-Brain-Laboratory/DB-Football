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

"""
Built-in AI in a form of actor-critic models
"""

import torch.nn as nn
import torch
from light_malib.envs.gr_football.tools import action_set as act
from light_malib.utils.logger import Logger
import numpy as np
from gym.spaces import Box, Discrete
from light_malib.envs.gr_football.encoders.encoder_dummy import FeatureEncoder

def check_tensor(data):
    if not isinstance(data, torch.Tensor):
        data = torch.as_tensor(data, dtype=torch.float32)
    return data

class Actor(nn.Module):
    def __init__(
        self,
        model_config,
        observation_space,
        action_space,
        custom_config,
        initialization,
    ):
        super().__init__()
        self.rnn_layer_num = 1
        self.rnn_state_size = 1

    def forward(self, observation, rnn_states, rnn_masks, action_masks, explore, actions):
        observation = check_tensor(observation)
        rnn_states = check_tensor(rnn_states)

        shape = list(observation.shape[:-1])
        actions = torch.full(
            shape, fill_value=act.BUILT_IN, dtype=torch.int, device=observation.device
        )

        action_log_probs = torch.zeros_like(actions)

        return actions, rnn_states, action_log_probs, None

    def eval_actions(self):
        raise NotImplementedError

    def eval_values(self):
        raise NotImplementedError

class Critic(nn.Module):
    def __init__(
        self,
        model_config,
        observation_space,
        action_space,
        custom_config,
        initialization,
    ):
        super().__init__()
        self.rnn_layer_num = 1
        self.rnn_state_size = 1

    def forward(self, observation, rnn_states, rnn_masks):
        observation = check_tensor(observation)
        rnn_states = check_tensor(rnn_states)
        shape = list(observation.shape[:-1])
        value = torch.zeros(shape, dtype=observation.dtype, device=observation.device)
        return value, rnn_states