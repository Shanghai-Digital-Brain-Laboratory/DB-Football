import torch
import torch.nn as nn

from light_malib.algorithm.common.rnn_net import RNNNet
from . import enhanced_LightActionMask_5
from . import encoder_basic_5


class PartialLayernorm(nn.Module):
    def __init__(self, in_dim, layer):
        super().__init__()
        self.layer = layer
        self.dim = self.layer.normalized_shape[0]
        self.layer2 = nn.LayerNorm(in_dim - self.dim)

    def forward(self, x):
        x1 = x[..., :self.dim]
        y1 = self.layer(x1)
        x2 = x[..., self.dim:]
        y2 = self.layer2(x2)
        y = torch.concat([y1, y2], dim=-1)
        return y


class RNNNetEnhanced(RNNNet):
    def __init__(self, model_config, observation_space, action_space, custom_config, initialization):
        super().__init__(model_config, observation_space, action_space, custom_config, initialization)
        self.base._feature_norm = PartialLayernorm(
            enhanced_LightActionMask_5.FeatureEncoder().observation_space.shape[0],
            nn.LayerNorm(encoder_basic_5.FeatureEncoder().observation_space.shape[0])
        )


Actor = RNNNetEnhanced
Critic = RNNNetEnhanced
share_backbone = False
FeatureEncoder = enhanced_LightActionMask_5.FeatureEncoder