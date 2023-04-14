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

### extended model with enhanced encoder

import torch
import torch.nn as nn

from light_malib.algorithm.common import actor
from light_malib.algorithm.common import critic
from light_malib.envs.gr_football.encoders import encoder_basic, encoder_enhanced
from light_malib.utils.logger import Logger
from gym.spaces import Discrete

class PartialLayernorm(nn.Module):
    def __init__(self, in_dim, layer):
        super().__init__()
        self.layer = layer
        self.dim = self.layer.normalized_shape[0]
        self.layer2 = nn.LayerNorm(in_dim - self.dim)

    def forward(self, x):
        x1 = x[..., : self.dim]
        y1 = self.layer(x1)
        x2 = x[..., self.dim :]
        y2 = self.layer2(x2)
        y = torch.concat([y1, y2], dim=-1)
        return y

class Actor(actor.Actor):
    def __init__(
        self,
        model_config,
        observation_space,
        action_space,
        custom_config,
        initialization,
    ):
        if observation_space is None:
            observation_space = encoder_basic.FeatureEncoder.observation_space
        if action_space is None:
            action_space = Discrete(19)

        super().__init__(
            model_config, observation_space, action_space, custom_config, initialization
        )
        self.base._feature_norm = PartialLayernorm(
            encoder_enhanced.FeatureEncoder(num_players=11*2).observation_space.shape[0],
            nn.LayerNorm(encoder_basic.FeatureEncoder(num_players=11*2).observation_space.shape[0]),
        )

class Critic(critic.Critic):
    def __init__(
        self,
        model_config,
        observation_space,
        action_space,
        custom_config,
        initialization,
    ):
        if observation_space is None:
            observation_space = encoder_basic.FeatureEncoder.observation_space
        if action_space is None:
            action_space = Discrete(1)
        super().__init__(
            model_config, observation_space, action_space, custom_config, initialization
        )
        self.base._feature_norm = PartialLayernorm(
            encoder_enhanced.FeatureEncoder(num_players=11*2).observation_space.shape[0],
            nn.LayerNorm(encoder_basic.FeatureEncoder(num_players=11*2).observation_space.shape[0]),
        )
        
class FeatureEncoder(encoder_enhanced.FeatureEncoder):
    def __init__(self, **kwargs):
        kwargs["num_players"]=11*2
        super().__init__(**kwargs)