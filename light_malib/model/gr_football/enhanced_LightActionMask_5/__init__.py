import torch
import torch.nn as nn

from light_malib.algorithm.common.rnn_net import RNNNet
from . import enhanced_LightActionMask_5
from . import encoder_basic_5
from .partial_layer_norm import PartialLayernorm


# class RNNNetEnhanced(RNNNet):
class Actor(RNNNet):

    def __init__(self, model_config, observation_space, action_space, custom_config, initialization):
        super().__init__(model_config, observation_space, action_space, custom_config, initialization)
        self.base._feature_norm = PartialLayernorm(
            enhanced_LightActionMask_5.FeatureEncoder().observation_space.shape[0],
            nn.LayerNorm(encoder_basic_5.FeatureEncoder().observation_space.shape[0])
        )

    def forward(self, observations, actor_rnn_states, rnn_masks, action_masks, explore, actions):
        logits, actor_rnn_states = super().forward(
            observations, actor_rnn_states, rnn_masks
        )
        illegal_action_mask = 1 - action_masks
        logits = logits - 1e10 * illegal_action_mask

        dist = torch.distributions.Categorical(logits=logits)
        if actions is None:
            actions = dist.sample() if explore else dist.probs.argmax(dim=-1)
            dist_entropy = None
        else:
            dist_entropy = dist.entropy()
        action_log_probs = dist.log_prob(actions)  # num_action

        return actions, actor_rnn_states, action_log_probs, dist_entropy

# class RNNNetEnhanced_critic(RNNNet):
class Critic(RNNNet):
    def __init__(self, model_config, observation_space, action_space, custom_config, initialization):
        super().__init__(model_config, observation_space, action_space, custom_config, initialization)
        self.base._feature_norm = PartialLayernorm(
            enhanced_LightActionMask_5.FeatureEncoder().observation_space.shape[0],
            nn.LayerNorm(encoder_basic_5.FeatureEncoder().observation_space.shape[0])
        )


# Actor = RNNNetEnhanced
# Critic = RNNNetEnhanced_critic
share_backbone = False
FeatureEncoder = enhanced_LightActionMask_5.FeatureEncoder