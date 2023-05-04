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

import copy
import os
import pickle
import random
import gym
import torch
import numpy as np

from torch import nn
from light_malib.utils.logger import Logger
from light_malib.utils.typing import DataTransferType, Tuple, Any, Dict, EpisodeID, List
from light_malib.utils.episode import EpisodeKey

from light_malib.algorithm.common.policy import Policy
from gym.spaces import Box, Discrete

from ..utils import PopArt, init_fc_weights
import wrapt
import tree
import importlib
from light_malib.utils.logger import Logger
from light_malib.registry import registry


class MergedFeatureEncoder:
    def __init__(self, behavior_FE, target_FE):
        self.behavior_FE = behavior_FE
        self.target_FE = target_FE

        self.behavior_FE_dim = self.behavior_FE.observation_space.shape
        self.target_FE_dim = self.target_FE.observation_space.shape

    @property
    def observation_space(self):
        return Box(low=-1000, high=1000, shape=[self.behavior_FE_dim + self.target_FE_dim])

    @property
    def action_space(self):
        return self.behavior_FE.action_space

    @property
    def cutoff_idx(self):
        return self.behavior_FE_dim[0]

    def encode(self, states):
        behavior_feat = self.behavior_FE.encode(states)
        target_feat = self.target_FE.encode(states)
        merged_feat = []
        for player_id in range(len(behavior_feat)):
            _feat = np.concatenate([behavior_feat[player_id], target_feat[player_id]])
            merged_feat.append(_feat)

        return merged_feat


@registry.registered(registry.POLICY)
class BC:
    def __init__(
        self,
        registered_name: str,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        model_config: Dict[str, Any] = None,
        custom_config: Dict[str, Any] = None,
        **kwargs,
    ):
        self.registered_name = registered_name

        algo_cls = importlib.import_module(f"light_malib.registry.registration")

        behavior_model_name = model_config['behavior_model_name']
        behavior_pkl_path = model_config['behavior_model_path']
        behavior_policy_cls = getattr(algo_cls, f"{behavior_model_name}")
        self.behavior_policy = behavior_policy_cls.load(behavior_pkl_path)
        self.behavior_policy.eval()

        # super(BC, self).__init__(
        #     registered_name=registered_name,
        #     observation_space=self.behavior_policy.feature_encoder.observation_space,
        #     action_space=self.behavior_policy.feature_encoder.action_space,
        #     model_config=model_config,
        #     custom_config=custom_config,
        # )

        self.target_model_name = model_config['target_model_name']
        target_policy_cls = getattr(algo_cls, f"{self.target_model_name}")
        self.target_policy = target_policy_cls(
            registered_name=self.target_model_name,
            observation_space=observation_space,
            action_space = action_space,
            model_config = model_config,
            custom_config = custom_config
        )
        self.target_feature_encoder = self.target_policy.feature_encoder


        self.merged_feature_encoder = MergedFeatureEncoder(self.behavior_policy.feature_encoder,
                                                           self.target_feature_encoder)
        self.cutoff_idx = self.merged_feature_encoder.cutoff_idx

        self.custom_config = custom_config
        self.share_backbone = False

    @property
    def feature_encoder(self):
        return self.merged_feature_encoder


    def get_initial_state(self, batch_size):
        return self.behavior_policy.get_initial_state(batch_size)


    def train(self):
        self.behavior_policy.train()

    def eval(self):
        self.behavior_policy.eval()

    def to_device(self, device):
        self_copy = copy.deepcopy(self)
        self_copy.device = device
        self_copy.target_policy = self_copy.target_policy.to_device(device)
        return self_copy

    def compute_action(self, **kwargs):
        if kwargs['explore']:
            kwargs[EpisodeKey.CUR_OBS] = kwargs[EpisodeKey.CUR_OBS][:, :self.cutoff_idx]
            return self.behavior_policy.compute_action(**kwargs)
        else:
            kwargs[EpisodeKey.CUR_OBS] = kwargs[EpisodeKey.CUR_OBS][:, self.cutoff_idx:]
            return self.target_policy.compute_action(**kwargs)

    def dump(self, dump_dir):
        # os.makedirs(dump_dir, exist_ok=True)
        # torch.save(self.target_policy.actor, os.path.join(dump_dir, "actor.pt"))
        # pickle.dump(self.description, open(os.path.join(dump_dir, "desc.pkl"), "wb"))
        self.target_policy.dump(dump_dir)



    def value_function(self, *args, **kwargs):
        pass



