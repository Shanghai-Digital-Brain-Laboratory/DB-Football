from light_malib.algorithm.common.rnn_net import RNNNet
import torch

class Actor(RNNNet):
    def __init__(
        self,
        model_config,
        observation_space,
        action_space,
        custom_config,
        initialization,
    ):
        super().__init__(
            model_config, observation_space, action_space, custom_config, initialization
        )
        
    def forward(self, observations, actor_rnn_states, rnn_masks, action_masks, explore, actions):
        logits, actor_rnn_states = super().forward(
            observations, actor_rnn_states, rnn_masks
        )
        illegal_action_mask = 1-action_masks
        logits = logits - 1e10 * illegal_action_mask

        dist = torch.distributions.Categorical(logits=logits)
        if actions is None:
            actions = dist.sample() if explore else dist.probs.argmax(dim=-1)
            dist_entropy = None
        else:
            dist_entropy = dist.entropy()
        action_log_probs = dist.log_prob(actions) # num_action
        
        return actions,  actor_rnn_states, action_log_probs, dist_entropy

    def logits(self, obs, rnn_states, masks):
        obs = torch.as_tensor(obs, dtype=torch.float32)
        if rnn_states is not None:
            rnn_states = torch.as_tensor(rnn_states, dtype=torch.float32)
        feat = self.base(obs)
        if self._use_rnn:
            assert masks is not None
            masks = torch.as_tensor(masks, dtype=torch.float32)
            feat, rnn_states = self.rnn(feat, rnn_states, masks)

        act_out = self.out(feat)
        return act_out, rnn_states