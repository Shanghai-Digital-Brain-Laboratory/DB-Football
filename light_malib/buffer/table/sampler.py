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
