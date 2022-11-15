# MIT License

# Copyright (c) 2021 MARL @ SJTU

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

"""
Model factory. Add more description
"""

import copy

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from light_malib.utils.preprocessor import Mode, get_preprocessor
from light_malib.utils.typing import DataTransferType, Dict, Any, List


def mlp(layers_config):
    layers = []
    for j in range(len(layers_config) - 1):
        tmp = [nn.Linear(layers_config[j]["units"], layers_config[j + 1]["units"])]
        if layers_config[j + 1].get("activation"):
            tmp.append(getattr(torch.nn, layers_config[j + 1]["activation"])())
        layers += tmp
    return nn.Sequential(*layers)


class Model(nn.Module):
    def __init__(self, input_space, output_space):
        """
        Create a Model instance.
        Common abstract methods could be added here.

        :param input_space: Input space size, int or gym.spaces.Space.
        :param output_space: Output space size, int or gym.spaces.Space.
        """

        super(Model, self).__init__()
        if isinstance(input_space, gym.spaces.Space):
            self.input_dim = get_preprocessor(input_space)(input_space).size
        else:
            self.input_dim = input_space

        if isinstance(output_space, gym.spaces.Space):
            self.output_dim = get_preprocessor(output_space)(output_space).size
        else:
            self.output_dim = output_space

    def get_initial_state(self) -> List[torch.TensorType]:
        """Return a list of initial rnn state, if current model is rnn"""

        return []


class MLP(Model):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        model_config: Dict[str, Any],
        **kwargs
    ):
        super(MLP, self).__init__(observation_space, action_space)

        layers_config: list = (
            self._default_layers()
            if model_config.get("layers") is None
            else model_config["layers"]
        )
        layers_config.insert(0, {"units": self.input_dim})

        if action_space:
            act_dim = get_preprocessor(action_space)(action_space).size
            layers_config.append(
                {"units": act_dim, "activation": model_config["output"]["activation"]}
            )
        self.use_feature_normalization = kwargs.get("use_feature_normalization", False)
        if self.use_feature_normalization:
            self._feature_norm = nn.LayerNorm(self.input_dim)
        self.net = mlp(layers_config)

    def _default_layers(self):
        return [
            {"units": 256, "activation": "ReLU"},
            {"units": 64, "activation": "ReLU"},
        ]

    def forward(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32)
        if self.use_feature_normalization:
            obs = self._feature_norm(obs)
        pi = self.net(obs)
        return pi


class RNN(Model):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        model_config: Dict[str, Any],
    ):
        super(RNN, self).__init__(observation_space, action_space)
        self.hidden_dims = (
            64 if model_config is None else model_config.get("rnn_hidden_dim", 64)
        )

        # default by flatten
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dims)
        self.rnn = nn.GRUCell(self.hidden_dims, self.hidden_dims)
        self.fc2 = nn.Linear(self.hidden_dims, self.output_dim)

    def _init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.hidden_dims).zero_()

    def get_initial_state(self) -> List[torch.TensorType]:
        return [self._init_hidden()]

    def forward(self, obs, hidden_state):
        obs = torch.as_tensor(obs, dtype=torch.float32)
        x = F.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.hidden_dims)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h


class QMixer(Model):
    def __init__(self, obs_dim, num_agents, model_config=None):
        super(QMixer, self).__init__(obs_dim, 1)
        self.n_agents = num_agents

        self.embed_dim = (
            32 if model_config is None else model_config.get("mixer_embed_dim", 32)
        )
        self.hyper_hidden_dim = (
            64 if model_config is None else model_config.get("hyper_hidden_dim", 64)
        )

        self.hyper_w_1 = nn.Sequential(
            nn.Linear(obs_dim, self.hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hyper_hidden_dim, self.embed_dim * num_agents),
        )
        self.hyper_w_final = nn.Sequential(
            nn.Linear(obs_dim, self.hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hyper_hidden_dim, self.embed_dim),
        )

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(obs_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(
            nn.Linear(obs_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.output_dim),
        )

    def forward(self, agent_qs, obs):
        bs = agent_qs.size(0)
        obs = torch.as_tensor(obs, dtype=torch.float32)
        agent_qs = torch.as_tensor(agent_qs, dtype=torch.float32)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = torch.abs(self.hyper_w_1(obs))
        b1 = self.hyper_b_1(obs)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = torch.abs(self.hyper_w_final(obs))
        w_final = w_final.view(-1, self.embed_dim, 1)
        v = self.V(obs).view(-1, 1, 1)
        y = torch.bmm(hidden, w_final) + v
        q_tot = y.view(bs, -1)
        return q_tot


def get_model(model_config: Dict[str, Any]):
    model_type = model_config["network"]

    if model_type == "mlp":
        handler = MLP
    elif model_type == "rnn":
        handler = RNN
    elif model_type == "cnn":
        raise NotImplementedError
    elif model_type == "rcnn":
        raise NotImplementedError
    else:
        raise NotImplementedError

    def builder(observation_space, action_space, use_cuda=False, **kwargs):
        model = handler(
            observation_space, action_space, copy.deepcopy(model_config), **kwargs
        )
        if use_cuda:
            model.cuda()
        return model

    return builder
