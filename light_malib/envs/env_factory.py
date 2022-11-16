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

from light_malib.registry import registry

MAX_ENV_PER_WORKER = 100


def make_envs(worker_id, worker_seed, cfg):
    envs = {}
    assert len(cfg) < MAX_ENV_PER_WORKER
    for idx, env_cfg in enumerate(cfg):
        env_name = env_cfg["cls"]
        id_prefix = env_cfg["id_prefix"]
        env_id = "{}_{}_{}".format(worker_id, id_prefix, idx)
        assert env_id not in envs
        env_seed = worker_seed * MAX_ENV_PER_WORKER + idx
        env_cls = registry.get(registry.ENV, env_name)
        env = env_cls(env_id, env_seed, env_cfg)
        envs[env_id] = env
    return envs
