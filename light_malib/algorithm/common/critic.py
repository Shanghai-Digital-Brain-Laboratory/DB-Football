from light_malib.algorithm.common.rnn_net import RNNNet

class Critic(RNNNet):
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