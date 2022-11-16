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
import numpy as np


class Solver(MetaSolver):
    def __init__(self):
        self.iterations = 20000

    def compute(self, payoff):
        """
        Prioitise Fictitious Self-Play on win rate
        """
        print("---------------------computing PFSP---------------------")
        newest_payoff_entry = payoff[-1, :]
        fn = lambda x: (1 - x) ** 0.5
        fn_payoff_entry = [fn(x) for x in newest_payoff_entry]
        sum_fn = sum(fn_payoff_entry)
        PFSP_dist = [i / sum_fn for i in fn_payoff_entry]

        print(PFSP_dist)
        eqs = (np.array(PFSP_dist), np.array(PFSP_dist))
        return eqs
