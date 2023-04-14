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
Our Feature Encoder code is adapated from wekick and liveinparis in the kaggle football competition.

basic FE outputs 133-dimension features, used for 5v5 full-game scenarios
"""

import numpy as np
from light_malib.utils.logger import Logger
from gym.spaces import Box, Discrete

# TODO(jh): we need a dummy one
class FeatureEncoder:
    def __init__(self,**kwargs):
        pass

    def encode(self, states):
        # at least 19 for action masks
        return np.zeros((len(states), 20), dtype=np.float32)

    @property
    def global_observation_space(self):
        return self.observation_space

    @property
    def observation_space(self):
        return Box(low=-1000, high=1000, shape=[20])
    
    @property
    def action_space(self):
        return Discrete(19)