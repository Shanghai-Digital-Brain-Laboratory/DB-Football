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

from .base import MetaSolver
import nashpy as nash
import numpy as np


class Solver(MetaSolver):
    def __init__(self):
        self.iterations = 20000

    def compute(self, payoff):
        assert len(payoff.shape) == 2 and np.all(
            payoff + payoff.T < 1e-6
        ), "only support two-player zero-sum symetric games now.\n payoff:{}".format(
            payoff
        )
        eqs = self.compute_nash(payoff)
        return eqs

    def compute_nash(self, payoff):
        game = nash.Game(payoff)
        freqs = list(game.fictitious_play(iterations=100000))[-1]
        eqs = tuple(map(lambda x: x / np.sum(x), freqs))
        return eqs
