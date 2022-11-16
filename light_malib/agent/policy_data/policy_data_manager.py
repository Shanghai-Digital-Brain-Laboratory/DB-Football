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

from itertools import product

from light_malib.agent.agent_manager import AgentManager
from ..agent import Agents
from light_malib.utils.logger import Logger
import numpy as np
import importlib

# from .custom_update_func.gr_football import update_func


class PolicyDataManager:
    def __init__(self, cfg, agent_manager: AgentManager, max_num_policies=100):
        self.id = "PolicyDataManager"
        self.cfg = cfg
        self.agents: Agents = agent_manager.agents
        self.policy_server = agent_manager.policy_server
        self.max_num_policies = max_num_policies
        module = importlib.import_module(
            "light_malib.agent.policy_data.custom_update_func.{}".format(
                self.cfg.update_func
            )
        )
        self.update_func = module.update_func

        self.data = {}
        self.raw_data = {}

        assert "payoff" in cfg.fields
        for field, field_cfg in cfg.fields.items():
            if field_cfg.type == "matrix":
                self.init_matrix_data(field, default_value=field_cfg.missing_value)
            elif field_cfg.type == "array":
                self.init_array_data(field, default_value=field_cfg.missing_value)
            else:
                raise NotImplementedError

    def init_matrix_data(self, key, default_value=-100, dtype=float):
        max_shape = [self.max_num_policies] * len(self.agents)
        arr = np.full(max_shape, fill_value=default_value, dtype=dtype)
        self.data[key] = arr

    def init_array_data(self, key, default_value=-100, dtype=float):
        if not self.agents.share_policies:
            arrays = [
                np.full(self.max_num_policies, fill_value=default_value, dtype=dtype)
                for i in range(len(self.agents))
            ]
        else:
            arrays = [
                np.full(self.max_num_policies, fill_value=default_value, dtype=dtype)
            ] * len(self.agents)
        self.data[key] = {
            agent_id: array for agent_id, array in zip(self.agents, arrays)
        }

    def get_array_data(self, key, agent2policy_indices=None):
        if agent2policy_indices is None:
            indices = [slice(len(agent.policy_ids)) for agent in self.agents.values()]
        else:
            indices = [
                agent2policy_indices[agent_id] for agent_id in agent2policy_indices
            ]
        arrays = self.data[key]
        ret = {}
        for agent_id, index in zip(arrays, indices):
            ret[agent_id] = arrays[agent_id][index]
        return ret

    def get_matrix_data(self, key, agent2policy_indices=None):
        if agent2policy_indices is None:
            indices = tuple(
                slice(len(agent.policy_ids)) for agent in self.agents.values()
            )
        else:
            indices = np.ix_(
                *[agent2policy_indices[agent_id] for agent_id in agent2policy_indices]
            )
        matrix = self.data[key]
        matrix = matrix[indices]
        return matrix

    def set_matrix_data(self, key, value):
        matrix = self.data[key]
        slices = tuple(slice(len(agent.policy_ids)) for agent in self.agents.values())
        matrix[slices] = value

    def update_policy_data(self, eval_results, **kwargs):
        self.update_func(self, eval_results, **kwargs)

    def format_matrices_data(self, keys):
        matrices = {key: self.get_matrix_data(key).flatten() for key in keys}
        policy_ids_list = product(*[agent.policy_ids for agent in self.agents.values()])
        s = ""
        for idx, policy_ids in enumerate(policy_ids_list):
            values = {key: matrices[key][idx] for key in keys}
            s += "[{}:{}],".format(policy_ids, values)
        return s
