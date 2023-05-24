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
Trainer is a interface for
"""

from abc import ABCMeta, abstractmethod
from typing import Any, Dict

from .policy import Policy
from .loss_func import LossFunc


class Trainer(metaclass=ABCMeta):
    def __init__(self, tid: str):
        """Create a trainer instance.

        :param str tid: Specify trainer id.
        """

        self._tid = tid
        self._training_config = {}
        self._policy = None
        self._loss = None

    def __call__(self, *args, **kwargs):
        return self.optimize(*args, **kwargs)

    @property
    def training_config(self):
        return self._training_config

    @abstractmethod
    def optimize(self, batch, **kwargs) -> Dict[str, Any]:
        """Execution policy optimization then return a dict of statistics"""
        pass

    @property
    def policy(self) -> Policy:
        return self._policy

    @property
    def loss(self) -> LossFunc:
        return self._loss

    def reset(self, policy, training_config):
        """Reset policy, called before optimize, and read training configuration"""

        self._policy = policy
        self._training_config.update(training_config)
        if self._loss is not None:
            self._loss.reset(policy, training_config)
        # else:
        #     raise ValueError("Loss has not been initialized yet.")

    @abstractmethod
    def preprocess(self, batch, **kwargs) -> Any:
        """Preprocess batch if need"""
        pass
