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
    def optimize(self, batch) -> Dict[str, Any]:
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
