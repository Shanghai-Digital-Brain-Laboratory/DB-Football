import torch
import torch.nn as nn
import numpy as np
import os
from gym.spaces import Box, Discrete

from .tartrl_policy import PolicyNetwork
from .tartrl_utils import tartrl_obs_deal, _t2n

def check_tensor(data):
    if not isinstance(data,torch.Tensor):
        data=torch.as_tensor(data,dtype=torch.float32)
    return data

# class SerializableModel:
#     def __init__(self ,path ,device="cpu"):
#         self.path =path
#         self.device =device
#         self.load()
#
#     def __getstate__(self):
#         return self.path ,self.device
#
#     def __setstate__(self ,state):
#         self.path ,self.device =state
#         self.load()
#
#     def load(self):
#         self.model =torch.jit.load(self.path, map_location=self.device)
#
#     def __call__(self ,features ,amasks):
#         return self.model(features ,amasks)
model = PolicyNetwork()
model.load_state_dict(torch.load( os.path.dirname(os.path.abspath(__file__)) + '/actor.pt', map_location=torch.device("cpu")))
model.eval()

class Actor(nn.Module):
    def __init__(
            self,
            model_config,
            observation_space,
            action_space,
            custom_config,
            initialization
    ):
        super().__init__()
        rnn_shape = [1, 1, 1, 512]
        self.rnn_layer_num = 1
        self.rnn_state_size = 1
        self.rnn_hidden_state = [np.zeros(rnn_shape, dtype=np.float32) for _ in range(11)]
        self.model = model

    def forward(self, obs, rnn_states, rnn_masks, action_masks, explore, actions):

        # encoded_obs = obs[...,19:]
        # avail_actions = np.zeros((obs.shape[0], 20))
        # avail_actions[..., :19] = obs[...,:19]
        # rnn_hidden_state = np.concatenate(self.rnn_hidden_state[1:], -2)
        # with torch.no_grad():
        #     actions, rnn_hidden_state = self.model(encoded_obs, rnn_hidden_state, available_actions=avail_actions, deterministic=True)
        # self.rnn_hidden_state[1:,...] = np.array(np.split(_t2n(rnn_hidden_state), 1))
        #

        action_list = []
        for i in range(obs.shape[0]):
            each_obs = obs[i]
            encoded_obs = each_obs[19:]
            encoded_obs = encoded_obs.reshape(1,330) #np.concatenate(encoded_obs.reshape(1,1,330))
            avail_actions = np.zeros(20)
            avail_actions[:19] = each_obs[:19]
            avail_actions = np.concatenate(avail_actions.reshape([1, 1, 20]))
            rnn_hidden_state = np.concatenate(self.rnn_hidden_state[i+1])
            with torch.no_grad():
                actions, rnn_hidden_state = self.model(encoded_obs, rnn_hidden_state, available_actions=avail_actions, deterministic=not explore)
            if actions[0][0] == 17 and each_obs["sticky_actions"][8] == 1:
                actions[0][0] = 15
            self.rnn_hidden_state[i+1] = np.array(np.split(_t2n(rnn_hidden_state), 1))

            action_list.append(actions)

        action_list = torch.concatenate(action_list).squeeze(-1)
        return action_list,torch.tensor(rnn_states),torch.ones_like(action_list), torch.ones_like(action_list)


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

# TODO(jh): we need a dummy one
class FeatureEncoder:
    def __init__(self):
        pass

    def encode(self, states):
        # at least 19 for action masks
        raw_obs = []
        for i in states:
            current_obs = i.obs_list[-1]
            tartrl_obs = tartrl_obs_deal(current_obs)
            obs = tartrl_obs['obs']
            # avail_actions = np.zeros(20)
            avail_actions = tartrl_obs['available_action']

            obs_new = np.concatenate([avail_actions, obs])

            raw_obs.append(obs_new)
        encoded_obs = np.stack(raw_obs)

        return encoded_obs
    @property
    def global_observation_space(self):
        return Box(
            low=-1,
            high=1,
            shape=[
                20,
            ],
        )
    @property
    def action_space(self):
        return Discrete(19)


    @property
    def observation_space(self):
        return Box(
            low=-1,
            high=1,
            shape=[
                20,
            ],
        )

