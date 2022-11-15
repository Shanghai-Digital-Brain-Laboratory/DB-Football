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

from abc import ABCMeta, abstractmethod
from typing import Dict, Any, Sequence
import torch
from light_malib.utils.general import tensor_cast

class LossFunc(metaclass=ABCMeta):
    """Define loss function and optimizers

    Flowchart:
    1. create a loss func instance with: loss = LossFunc(policy, **kwargs)
    2. setup optimizers: loss.setup_optimizers(**kwargs)
    3. zero grads: loss.zero_grads()
    4. calculate loss and got returned statistics: statistics = loss(batch)
    5. do optimization (step): loss.step()
    **NOTE**: if you wanna calculate policy for another policy, do reset: loss.reset(policy)
    """

    def __init__(self):
        self._policy = None
        self._params = {"device": "cpu", "custom_caster": None}
        self._gradients = []
        self.optimizers = None
        self.loss = []

    @property
    def stacked_gradients(self):
        """Return stacked gradients"""

        return self._gradients

    def push_gradients(self, grad):
        """Push new gradient to gradients"""

        self._gradients.append(grad)

    @property
    def optim_cls(self) -> type:
        """Return default optimizer class. If not specify in params, return Adam as default."""

        return getattr(torch.optim, self._params.get("optimizer", "Adam"))

    @property
    def policy(self):
        return self._policy

    @abstractmethod
    def setup_optimizers(self, *args, **kwargs):
        """Set optimizers and loss function"""

        # self.optimizers.append(...)
        # self.loss.append(...)
        pass

    def __call__(self, *args, **kwargs) -> Dict[str, Any]:
        """Compute loss function here, but not optimize"""
        return tensor_cast(
            custom_caster=self._params["custom_caster"],
            callback=None,
            dtype_mapping=None,
            device=self._params["device"],
        )(self.loss_compute)(*args, **kwargs)

    @abstractmethod
    def loss_compute(self, *args, **kwargs):
        """Implement loss computation here"""

    @abstractmethod
    def step(self) -> Any:
        pass

    def zero_grad(self):
        """Clean stacked gradients and optimizers"""

        self._gradients = []
        if isinstance(self.optimizers, Sequence):
            _ = [p.zero_grad() for p in self.optimizers]
        elif isinstance(self.optimizers, Dict):
            _ = [p.zero_grad() for p in self.optimizers.values()]
        elif isinstance(self.optimizers, torch.optim.Optimizer):
            self.optimizers.zero_grad()
        else:
            raise TypeError(
                f"Unexpected optimizers type: {type(self.optimizers)}, expected are included: Sequence, Dict, and torch.optim.Optimizer"
            )

    def reset(self, policy, configs):
        # reset optimizers
        # self.optimizers = []
        self.loss = []
        self._params.update(configs)
        if self._policy is not policy:
            self._policy = policy
            self.setup_optimizers()
