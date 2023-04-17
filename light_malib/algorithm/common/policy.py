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

"""
Implementation of basic PyTorch-based policy class
"""

import gym

from abc import ABCMeta, abstractmethod
import torch

import torch.nn as nn

from light_malib.utils import errors
from light_malib.utils.typing import (
    DataTransferType,
    ModelConfig,
    Dict,
    Any,
    Tuple,
    Callable,
    List,
)
from light_malib.utils.preprocessor import get_preprocessor, Mode

DEFAULT_MODEL_CONFIG = {
    "actor": {
        "network": "mlp",
        "layers": [
            {"units": 64, "activation": "ReLU"},
            {"units": 64, "activation": "ReLU"},
        ],
        "output": {"activation": False},
    },
    "critic": {
        "network": "mlp",
        "layers": [
            {"units": 64, "activation": "ReLU"},
            {"units": 64, "activation": "ReLU"},
        ],
        "output": {"activation": False},
    },
}

class Policy(metaclass=ABCMeta):
    def __init__(
        self,
        registered_name: str,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        model_config: ModelConfig = None,
        custom_config: Dict[str, Any] = None,
        **kwargs,
    ):
        """Create a policy instance.

        :param str registered_name: Registered policy name.
        :param gym.spaces.Space observation_space: Raw observation space of related environment agent(s), determines
            the model input space.
        :param gym.spaces.Space action_space: Raw action space of related environment agent(s).
        :param Dict[str,Any] model_config: Model configuration to construct models. Default to None.
        :param Dict[str,Any] custom_config: Custom configuration, includes some hyper-parameters. Default to None.
        """

        self.registered_name = registered_name
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = torch.device("cpu")

        self.custom_config = {
            "gamma": 0.99,
            "use_cuda": False,
            "use_dueling": False,
            "preprocess_mode": Mode.FLATTEN,
        }
        # FIXME(jh): either start from {} or deep copy DEFAUL_MODEL_CONFIG!
        self.model_config = {}  # DEFAULT_MODEL_CONFIG

        if custom_config is None:
            custom_config = {}
        self.custom_config.update(custom_config)

        # FIXME(ming): use deep update rule
        if model_config is None:
            model_config = {}
        self.model_config.update(model_config)

        self.preprocessor = get_preprocessor(
            observation_space, self.custom_config["preprocess_mode"]
        )(observation_space)

    @property
    def description(self):
        """Return a dict of basic attributes to identify policy.

        The essential elements of returned description:

        - registered_name: `self.registered_name`
        - observation_space: `self.observation_space`
        - action_space: `self.action_space`
        - model_config: `self.model_config`
        - custom_config: `self.custom_config`

        :return: A dictionary.
        """

        return {
            "registered_name": self.registered_name,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "model_config": self.model_config,
            "custom_config": self.custom_config,
        }

    @abstractmethod
    def compute_action(
        self, observation: DataTransferType, **kwargs
    ) -> Tuple[DataTransferType, DataTransferType, List[DataTransferType]]:
        """Compute single action when rollout at each step, return 3 elements:
        action, action_dist, a list of rnn_state
        """
    @abstractmethod
    def get_initial_state(self, batch_size: int = None) -> List[DataTransferType]:
        """Return a list of rnn states if models are rnns"""

    @abstractmethod
    def to_device(self, device):
        pass
    
    @abstractmethod
    def value_function(self, *args, **kwargs):
        """Compute values of critic."""

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def eval(self):
        pass