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

from collections import defaultdict
import time


class Timer:
    def __init__(self):
        self.timestamps = {}
        self.elapses = defaultdict(int)
        self.elapses_ctr = defaultdict(int)

    def clear(self, timestamps=None, elapses=None):
        if timestamps is not None:
            for timestamp in timestamps:
                self.timestamps.pop(timestamp, 0)

        if elapses is not None:
            for elapse in elapses:
                self.elapses.pop(elapse, 0)
                self.elapses_ctr.pop(elapse, 0)

        if timestamps is None and elapses is None:
            self.timestamps = {}
            self.elapses = defaultdict(int)
            self.elapses_ctr = defaultdict(int)

    def record(self, key):
        self.timestamps[key] = time.perf_counter()

    def time(self, okey, nkey=None, name=None):
        if okey is None:
            return None
        t = time.perf_counter()
        res = round(t - self.timestamps[okey], 8)
        if nkey is not None:
            self.timestamps[nkey] = t
        if name is not None:
            self.elapses[name] += res
            self.elapses_ctr[name] += 1
        return res

    def diff(self, key1, key2, name=None):
        if key1 is None:
            return None
        t1 = self.timestamps[key1]
        t2 = self.timestamps[key2]
        res = round(t2 - t1, 8)
        if name is not None:
            self.elapses[name] += res
            self.elapses[name] += 1
        return res

    def elapse(self, name, mode="mean"):
        if mode == "mean":
            return self.elapses[name] / self.elapses_ctr[name]
        elif mode == "sum":
            return self.elapses[name]
        else:
            raise NotImplementedError

    @property
    def mean_elapses(self):
        return {k: v / self.elapses_ctr[k] for k, v in self.elapses.items()}


global_timer = Timer()
