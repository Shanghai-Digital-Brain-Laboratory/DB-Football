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
model with enhanced encoder
"""

import torch
import torch.nn as nn

from light_malib.algorithm.common.rnn_net import RNNNet
from . import encoder_enhanced
from gym.spaces import Discrete


class Actor(RNNNet):
    def __init__(
        self,
        model_config,
        observation_space,
        action_space,
        custom_config,
        initialization,
    ):
        if observation_space is None:
            observation_space = encoder_enhanced.FeatureEncoder.observation_space
        if action_space is None:
            action_space = Discrete(19)
        super().__init__(
            model_config, observation_space, action_space, custom_config, initialization
        )


class Critic(RNNNet):
    def __init__(
        self,
        model_config,
        observation_space,
        action_space,
        custom_config,
        initialization,
    ):
        if observation_space is None:
            observation_space = encoder_enhanced.FeatureEncoder.observation_space
        if action_space is None:
            action_space = Discrete(1)
        super().__init__(
            model_config, observation_space, action_space, custom_config, initialization
        )


share_backbone = False
FeatureEncoder = encoder_enhanced.FeatureEncoder
