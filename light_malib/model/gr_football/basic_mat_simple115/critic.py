import torch.nn as nn

class Critic(nn.Module):
    def __init__(
        self,
        model_config,
        action_space,
        custom_config,
        initialization,
        backbone
    ):
        super().__init__()
        # TODO(jh): remove. legacy.
        self.rnn_layer_num=1
        self.rnn_state_size=1
        
    def forward(self, states, critic_rnn_states, rnn_masks):
        values = states["values"]
        return values, critic_rnn_states