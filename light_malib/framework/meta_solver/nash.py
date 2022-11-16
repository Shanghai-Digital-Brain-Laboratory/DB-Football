# MIT License

# Copyright (c) 2022 DigitalBrain, Yan Song and He jiang

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
