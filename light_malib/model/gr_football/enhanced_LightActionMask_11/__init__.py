# MIT License

# Copyright (c) 2022 DigitalBrain, Yan Song and He jiang

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

### extended model with enhanced encoder and light action mask

import torch
import torch.nn as nn

from light_malib.algorithm.common.rnn_net import RNNNet
from . import encoder_enhanced_LessActionMask
from . import encoder_basic
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
            observation_space = encoder_basic.FeatureEncoder.observation_space
        if action_space is None:
            action_space = Discrete(19)
        super().__init__(
            model_config, observation_space, action_space, custom_config, initialization
        )
        self.base._feature_norm = PartialLayernorm(
            encoder_enhanced_LessActionMask.FeatureEncoder().observation_space.shape[0],
            nn.LayerNorm(encoder_basic.FeatureEncoder().observation_space.shape[0]),
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
            observation_space = encoder_basic.FeatureEncoder.observation_space
        if action_space is None:
            action_space = Discrete(1)
        super().__init__(
            model_config, observation_space, action_space, custom_config, initialization
        )
        self.base._feature_norm = PartialLayernorm(
            encoder_enhanced_LessActionMask.FeatureEncoder().observation_space.shape[0],
            nn.LayerNorm(encoder_basic.FeatureEncoder().observation_space.shape[0]),
        )


share_backbone = False
FeatureEncoder = encoder_enhanced_LessActionMask.FeatureEncoder
