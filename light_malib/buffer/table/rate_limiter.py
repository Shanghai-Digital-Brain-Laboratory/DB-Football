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


class RateLimiter:
    def __init__(self, table, min_size=1, r_w_ratio=None):
        self.table = table
        self.min_size = min_size
        self.r_w_ratio = r_w_ratio

    def is_reading_available(self, batch_size):
        if self.table.write_num < self.min_size:
            return False
        if self.r_w_ratio is not None:
            max_read_num = self.table.write_num * self.r_w_ratio
            if self.table.read_num + batch_size >= max_read_num:
                return False
        # else
        return True
