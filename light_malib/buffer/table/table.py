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


from light_malib.utils.logger import Logger
from light_malib.utils.timer import Timer
import queue
from readerwriterlock import rwlock
import numpy as np
from .sampler import UniformSampler, LUMRFSampler, LULRFSampler
import time
from .rate_limiter import RateLimiter
from easydict import EasyDict


class Table:
    """
    1. support uniform sampling & maybe other patterns, e.g. priority sampling.
    2. support FIFO & maybe other patterns, e.g. priority evicting.
    """

    def __init__(self, table_cfg: EasyDict):
        self.table_cfg = table_cfg
        self.capacity = table_cfg.capacity
        self.sample_max_usage = table_cfg.get("sample_max_usage", 1e20)
        self.sampler_type = table_cfg.sampler_type
        self.rate_limiter_cfg = table_cfg.rate_limiter_cfg

        self.data = np.empty(shape=(self.capacity,), dtype=object)

        # TODO(jh): should we use here read-first lock?
        self.lock = rwlock.RWLockRead()

        # meta data/structure
        # TODO(jh): some of followings need to be protected separately if we support multiple writes/reads
        self.avail_indices = queue.Queue()
        # TODO(jh): may use a priority queue here!
        self.used_indices = queue.Queue()
        self.flags = np.zeros_like(self.data, dtype=bool)
        self.usage_ctrs = np.zeros_like(self.data, dtype=int)
        self.insert_timestamps = np.zeros_like(self.data, dtype=float)
        self.last_usage_timestamps = np.zeros_like(self.data, dtype=float)
        self.total_usage = 0
        self.max_usage = float("-inf")
        self.min_usage = float("inf")
        self.total_zero_usage_ctr = 0
        self.total_evict = 0
        self.total_read_wait_time = 0
        self.read_ctr = 0
        self.write_num = 0
        self.read_num = 0
        self.first_round = True
        self.first_write = True
        self.first_read = True

        for i in range(self.capacity):
            self.avail_indices.put_nowait(i)

        self.wait_time = 1
        self.timer = Timer()

        sampler_clses = {
            "uniform": UniformSampler,
            "lumrf": LUMRFSampler,
            "lulrf": LULRFSampler,
        }
        sampler_cls = sampler_clses[self.sampler_type]
        self.sampler = sampler_cls(self)
        self.rate_limiter = RateLimiter(self, **self.rate_limiter_cfg)

    def write(self, samples):
        # TODO(jh): check type of samples
        assert isinstance(samples, list) or isinstance(samples, np.ndarray)
        # cannot support list, because np.ndarray recursively builds...
        assert len(samples) > 0 and not isinstance(samples[0], list)
        n = len(samples)
        assert n <= self.capacity

        with self.lock.gen_wlock():
            self.write_num += len(samples)
            if self.first_write:
                self.timer.record("write_start")
                self.first_write = False

            # get available indices.
            indices = self._get_avail_indices(n)
            # insert data
            indices = np.array(indices, dtype=int)
            samples = np.array(samples, dtype=object)
            self._insert(indices, samples)

    def _get_avail_indices(self, n):
        assert n <= self.capacity
        if self.avail_indices.qsize() < n:
            remained = n - self.avail_indices.qsize()
            # FIFO: pop remained from used_indices
            evicted_indices = [self.used_indices.get_nowait() for i in range(remained)]
            self._evict(evicted_indices)

        indices = [self.avail_indices.get_nowait() for i in range(n)]
        return indices

    def _update_stats_for_eviction(self, indices):
        self.total_evict += len(indices)
        usage_ctrs = self.usage_ctrs[indices]
        self.total_usage += np.sum(usage_ctrs)
        self.max_usage = max(np.max(usage_ctrs), self.max_usage)
        self.min_usage = min(np.min(usage_ctrs), self.min_usage)
        self.total_zero_usage_ctr += np.count_nonzero(usage_ctrs == 0)

        # clear
        self.usage_ctrs[indices] = 0

    def _evict(self, indices):
        # update evicition statistics
        self._update_stats_for_eviction(indices)

        # main operations
        self.data[indices] = None
        self.flags[indices] = False
        self.usage_ctrs[indices] = 0
        for index in indices:
            self.avail_indices.put_nowait(index)

    def _insert(self, indices, samples):
        self.data[indices] = samples
        self.flags[indices] = True
        self.usage_ctrs[indices] = 0
        self.insert_timestamps[indices] = time.time()

        for index in indices:
            self.used_indices.put_nowait(index)

    def read(self, n, timeout=None):
        start_time = time.time()
        assert n <= self.capacity

        # TODO(jh): only use rlock is wrong in the multiprocessing setting. because meta data is changed!
        rlock = self.lock.gen_rlock()

        while (timeout is None) or (time.time() - start_time < timeout):
            rlock.acquire()
            if self.first_read:
                self.timer.record("read_start")
                self.first_read = False
            avail_indices = np.nonzero(self.flags)[0]
            avail_num = len(avail_indices)
            # maybe we could use condition here...
            # it is now fine because we only have a single reader actually...hhh
            if avail_num < n or not self.rate_limiter.is_reading_available(n):
                rlock.release()
                # wait
                self.total_read_wait_time += self.wait_time
                time.sleep(self.wait_time)
            else:
                indices = self.sampler.sample(avail_indices, n)
                samples = list(self.data[indices])

                self.usage_ctrs[indices] += 1
                self.last_usage_timestamps[indices] = time.time()

                # control the max usage
                if self.sample_max_usage is not None:
                    indices_of_indices = np.nonzero(
                        self.usage_ctrs[indices] >= self.sample_max_usage
                    )[0]
                    if len(indices_of_indices) > 0:
                        removed_indices = indices[indices_of_indices]
                        self._evict(removed_indices)

                self.read_ctr += 1
                self.read_num += len(samples)
                rlock.release()

                return samples
        return None

    def get_statistics(self):
        with self.lock.gen_rlock():
            try:
                ret = {}
                if self.total_evict > 0:
                    ret.update(
                        {
                            "max_reusage": self.max_usage,
                            "mean_reusage": self.total_usage / self.total_evict,
                            "min_reusage": self.min_usage,
                            "zero_usage_rate": self.total_zero_usage_ctr
                            / self.total_evict,
                        }
                    )
                if np.count_nonzero(self.flags) > 0:
                    ret.update(
                        {
                            "alive_usage_mean": np.mean(self.usage_ctrs[self.flags]),
                            "alive_usage_std": np.std(self.usage_ctrs[self.flags]),
                        }
                    )
                if self.read_ctr > 0:
                    ret["mean_wait_time"] = self.total_read_wait_time / self.read_ctr
                if self.read_num > 0:
                    ret["sample_per_minute_read"] = (
                        self.read_num / self.timer.time("read_start") * 60
                    )
                if self.write_num > 0:
                    ret["sample_per_minute_write"] = (
                        self.write_num / self.timer.time("write_start") * 60
                    )
                return ret
            except:
                return {}
