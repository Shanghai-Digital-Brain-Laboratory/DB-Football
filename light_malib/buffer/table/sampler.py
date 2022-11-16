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

import numpy as np
import time
from light_malib.utils.logger import Logger


class Sampler:
    def __init__(self, table=None):
        self.table = table

    def sample(self):
        raise NotImplementedError


class UniformSampler(Sampler):
    def __init__(self, table):
        super().__init__(table)

    def sample(self, indices, n):
        assert len(indices) >= n
        indices = np.random.choice(indices, size=n, replace=False)
        return indices


class LUMRFSampler(Sampler):
    "Less Usage Most Recent (inserted) First Sampler"

    def __init__(self, table=None):
        super().__init__(table)
        assert hasattr(table, "usage_ctrs")
        assert hasattr(table, "insert_timestamps")

    def sample(self, indices, n):
        assert len(indices) >= n
        usage_ctrs = self.table.usage_ctrs[indices]
        timestamps = self.table.insert_timestamps[indices]
        _indices = np.lexsort([-timestamps, usage_ctrs])[:n]
        indices = indices[_indices]
        assert len(indices) == n
        return indices


class LULRFSampler(Sampler):
    "Less Usage Least Recent (inserted) First Sampler"

    def __init__(self, table=None):
        super().__init__(table)
        assert hasattr(table, "usage_ctrs")
        assert hasattr(table, "insert_timestamps")

    def sample(self, indices, n):
        assert len(indices) >= n
        usage_ctrs = self.table.usage_ctrs[indices]
        timestamps = self.table.insert_timestamps[indices]
        _indices = np.lexsort([timestamps, usage_ctrs])[:n]
        indices = indices[_indices]
        assert len(indices) == n
        return indices
