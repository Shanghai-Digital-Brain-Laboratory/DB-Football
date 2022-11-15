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

import copy
from light_malib.utils.logger import Logger
from typing import Dict

from collections import OrderedDict
from light_malib.utils.desc.policy_desc import PolicyDesc
from light_malib.envs.env_factory import make_envs
import importlib
import numpy as np
from ..agent.agent import Agent,Agents
from ..utils.random import set_random_seed
from ..utils.distributed import get_actor
import ray
from light_malib.utils.desc.task_desc import RolloutDesc
from light_malib.utils.timer import global_timer
import random

class RolloutWorker:
    def __init__(self,id,seed,cfg,agents:Agents):
        self.id=id
        self.seed=seed
        #set_random_seed(self.seed)
        
        self.cfg=cfg
        self.agents=agents
        self.policy_server=get_actor(self.id,"PolicyServer")
        self.data_server=get_actor(self.id,"DataServer")
        self.envs:Dict=make_envs(self.id,self.seed,self.cfg.envs)
        
        module=importlib.import_module("light_malib.rollout.{}".format(self.cfg.rollout_func_name))
        self.rollout_func=module.rollout_func

        self.credit_reassign_cfg = self.cfg.credit_reassign

    def rollout(self,rollout_desc:RolloutDesc,eval=False,rollout_epoch=None):
        global_timer.record("rollout_start")
        assert len(self.envs)==1,"jh: currently only single env is supported"
        env=list(self.envs.values())[0]
        
        if self.agents.share_policies:
            # make order-invariant
            rollout_desc=self.random_permute(rollout_desc)
        
        policy_distributions=rollout_desc.policy_distributions
        policy_ids=self.sample_policies(policy_distributions)
        global_timer.time("rollout_start","sample_end","sample")
        
        # pull policies from remote
        self.pull_policies(policy_ids)
        behaving_policies=self.get_policies(policy_ids)
        global_timer.time("sample_end","policy_update_end","policy_update")
        
        rollout_length=self.cfg.rollout_length if not eval else self.cfg.eval_rollout_length
        result=self.rollout_func(
            eval,
            self,
            rollout_desc,
            env,
            behaving_policies,
            self.data_server,
            rollout_length=rollout_length,
            sample_length=self.cfg.sample_length,
            padding_length=self.cfg.padding_length,
            rollout_epoch=rollout_epoch,
            credit_reassign_cfg=self.credit_reassign_cfg
            # decaying_exploration_cfg=self.cfg.decaying_exploration
        )
        global_timer.time("policy_update_end","rollout_end","rollout")
        
        result["timer"]=copy.deepcopy(global_timer.elapses)
        global_timer.clear()
        
        return result
    
    def random_permute(self,rollout_desc:RolloutDesc):
        main_agent_id=rollout_desc.agent_id
        policy_distributions=rollout_desc.policy_distributions
        agent_ids=list(policy_distributions.keys())
        new_agent_ids=np.random.permutation(agent_ids)
        new_policy_distributions={agent_id:policy_distributions[new_agent_ids[idx]] for idx,agent_id in enumerate(agent_ids)}
        new_main_idx=np.where(new_agent_ids==main_agent_id)[0][0]
        new_main_agent_id=agent_ids[new_main_idx]
        rollout_desc.agent_id=new_main_agent_id
        rollout_desc.policy_distributions=new_policy_distributions
        return rollout_desc
    
    def get_policies(self,policy_ids):
        policies=OrderedDict()
        for agent_id,policy_id in policy_ids.items():
            policy=self.agents[agent_id].policy_data[policy_id].policy
            policies[agent_id]=(policy_id,policy)
        return policies
    
    def pull_policies(self,policy_ids):
        for agent_id,policy_id in policy_ids.items():
            if policy_id not in self.agents[agent_id].policy_data:
                policy_desc=ray.get(self.policy_server.pull.remote(self.id,agent_id,policy_id,old_version=None))
                if policy_desc is None:
                    raise Exception("{} {} not found in policy server".format(agent_id,policy_id))
                self.agents[agent_id].policy_data[policy_id]=policy_desc
            else:
                old_policy_desc:PolicyDesc=self.agents[agent_id].policy_data[policy_id]
                policy_desc=ray.get(self.policy_server.pull.remote(self.id,agent_id,policy_id,old_version=old_policy_desc.version))
                if policy_desc is not None:
                    self.agents[agent_id].policy_data[policy_id]=policy_desc
    
    def sample_policies(self,policy_distributions):
        policy_ids=OrderedDict()
        for agent_id,distribution in policy_distributions.items():
            policy_ids[agent_id]=self.sample_policy(distribution)
        return policy_ids
    
    def sample_policy(self,policy_distribution):
        policy_ids=list(policy_distribution.keys())
        policy_probs=np.array([policy_distribution[policy_id] for policy_id in policy_ids],dtype=np.float32)

        policy_probs=policy_probs/np.sum(policy_probs)
        policy_id=np.random.choice(a=policy_ids,p=policy_probs)
        return policy_id
