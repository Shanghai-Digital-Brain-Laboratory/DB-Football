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

from itertools import product

from light_malib.agent.agent_manager import AgentManager
from ..agent import Agents
from light_malib.utils.logger import Logger
import numpy as np
import importlib
from .custom_update_func.gr_football import update_func

class PolicyDataManager:
    def __init__(self,cfg,agent_manager:AgentManager,max_num_policies=100):
        self.id="PolicyDataManager"
        self.cfg=cfg
        self.agents:Agents=agent_manager.agents
        self.policy_server=agent_manager.policy_server
        self.max_num_policies=max_num_policies
        module=importlib.import_module("light_malib.agent.policy_data.custom_update_func.{}".format(self.cfg.update_func))
        self.update_func=module.update_func
        
        self.data={}        
        self.raw_data={}
        
        assert "payoff" in cfg.fields
        for field,field_cfg in cfg.fields.items():
            if field_cfg.type=="matrix":
                self.init_matrix_data(field,default_value=field_cfg.missing_value)
            elif field_cfg.type=="array":
                self.init_array_data(field,default_value=field_cfg.missing_value)
            else:
                raise NotImplementedError
        
    def init_matrix_data(self,key,default_value=-100,dtype=float):
        max_shape=[self.max_num_policies]*len(self.agents)
        arr=np.full(max_shape,fill_value=default_value,dtype=dtype)
        self.data[key]=arr
        
    def init_array_data(self,key,default_value=-100,dtype=float):
        if not self.agents.share_policies:
            arrays=[np.full(self.max_num_policies,fill_value=default_value,dtype=dtype) for i in range(len(self.agents))]
        else:
            arrays=[np.full(self.max_num_policies,fill_value=default_value,dtype=dtype)]*len(self.agents)
        self.data[key]={
            agent_id:array
            for agent_id,array in zip(self.agents,arrays)
        }
    
    def get_array_data(self,key,agent2policy_indices=None):
        if agent2policy_indices is None:
            indices=[slice(len(agent.policy_ids)) for agent in self.agents.values()]
        else:
            indices=[agent2policy_indices[agent_id] for agent_id in agent2policy_indices]
        arrays=self.data[key]
        ret={}
        for agent_id,index in zip(arrays,indices):
            ret[agent_id]=arrays[agent_id][index]
        return ret
        
    def get_matrix_data(self,key,agent2policy_indices=None):
        if agent2policy_indices is None:
            indices=tuple(slice(len(agent.policy_ids)) for agent in self.agents.values())
        else:
            indices=np.ix_(*[agent2policy_indices[agent_id] for agent_id in agent2policy_indices])
        matrix=self.data[key]
        matrix=matrix[indices]
        return matrix
    
    def set_matrix_data(self,key,value):
        matrix=self.data[key]
        slices=tuple(slice(len(agent.policy_ids)) for agent in self.agents.values())
        matrix[slices]=value
    
    def update_policy_data(self,eval_results, **kwargs):
        self.update_func(self,eval_results, **kwargs)
        
    def format_matrices_data(self,keys):
        matrices={key:self.get_matrix_data(key).flatten() for key in keys}
        policy_ids_list=product(*[agent.policy_ids for agent in self.agents.values()])
        s=""
        for idx,policy_ids in enumerate(policy_ids_list):
            values={key:matrices[key][idx] for key in keys}
            s+="[{}:{}],".format(policy_ids,values) 
        return s