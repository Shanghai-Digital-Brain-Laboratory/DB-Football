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


import threading
from light_malib.agent.agent import Agents
from light_malib.utils.desc.policy_desc import PolicyDesc
from readerwriterlock import rwlock
from light_malib.utils.logger import Logger

class PolicyServer:
    '''
    TODO(jh) This implementation is still problematic. we should rewrite it in asyncio's way, e.g. should use asyncio.Lock.
    Because there is not yield here, and no resouce contention, no lock is still correct.
    '''
    def __init__(self,id,cfg,agents:Agents):
        self.id=id
        self.cfg=cfg
        self.agents=agents
        locks=[rwlock.RWLockWrite()]*len(self.agents) if self.agents.share_policies else [rwlock.RWLockWrite() for i in range(len(self.agents))]
        self.locks={agent_id:lock for agent_id,lock in zip(self.agents.agent_ids,locks)}
        
        Logger.info("{} initialized".format(self.id))
        
    async def push(self,caller_id,policy_desc:PolicyDesc):
        # Logger.debug("{} try to push({}) to policy server".format(caller_id,str(policy_desc)))
        agent_id=policy_desc.agent_id
        policy_id=policy_desc.policy_id
        lock=self.locks[agent_id]
        with lock.gen_wlock():
            old_policy_desc=self.agents[agent_id].policy_data.get(policy_id,None)
            if old_policy_desc is None or old_policy_desc.version is None or old_policy_desc.version<policy_desc.version:
                self.agents[agent_id].policy_data[policy_id]=policy_desc
            else:
                Logger.debug("{}::push() discard order policy".format(self.id))
        # Logger.debug("{} try to push({}) to policy server ends".format(caller_id,str(policy_desc)))
        
    async def pull(self,caller_id,agent_id,policy_id,old_version=None):
        # Logger.debug("{} try to pull({},{},{}) from policy server".format(caller_id,agent_id,policy_id,old_version))   
        lock=self.locks[agent_id]
        with lock.gen_rlock():
            if policy_id not in self.agents[agent_id].policy_data:
                ret=None     
            else:
                policy_desc:PolicyDesc=self.agents[agent_id].policy_data[policy_id]
                if old_version is None or old_version<policy_desc.version:
                    ret=policy_desc
                else:
                    ret=None
        # Logger.debug("{} try to pull({},{},{}) from policy server ends".format(caller_id,agent_id,policy_id,old_version))
        return ret
        
    def dump_policy(self):
        pass