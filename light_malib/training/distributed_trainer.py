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

import copy
from light_malib.buffer import policy_server
from light_malib.utils.desc.policy_desc import PolicyDesc
from light_malib.utils.desc.task_desc import TrainingDesc
from ..utils.distributed import get_actor
from light_malib.utils.logger import Logger
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import os
import ray
from torch import distributed
import queue
from .data_prefetcher import GPUPreLoadQueueWrapper
from light_malib.utils.timer import global_timer
from light_malib.registry import registry


class DistributedPolicyWrapper:
    """
    TODO much more functionality
    """

    def __init__(self, policy, local_rank):
        Logger.info(
            "local_rank: {} cuda_visible_devices:{}".format(
                local_rank, os.environ["CUDA_VISIBLE_DEVICES"]
            )
        )
        self.device = torch.device("cuda:0")
        self.policy = policy.to_device(self.device)
        # maintain a shallow copy of policy, which shares the underlying models with self.policy
        self._policy = copy.copy(self.policy)
     
        self._wrapping_module_names=["actor","critic","target_critic"]#,"value_normalizer"]
        if policy.share_backbone:
            self._wrapping_module_names.append("backbone")
        
        for key in self._wrapping_module_names:
            self._wrap(key)

        # TODO jh: we need a distributed version of value_normalizer           
        self._wrap("value_normalizer",False)
    
    def _wrap(self, key, distributed=True):
        if hasattr(self.policy,key):
            value=getattr(self.policy,key)
            if isinstance(value,nn.Module) and len(list(value.parameters()))>0:
                setattr(self._policy,key,getattr(self.policy,key))
                if distributed:
                    setattr(self.policy,key,DistributedDataParallel(getattr(self._policy,key), device_ids=[0], find_unused_parameters=False))
                else:
                    setattr(self.policy,key,getattr(self._policy,key))
    
    def get_unwrapped_policy(self,device="cpu"):
        return self._policy.to_device(device)
    
    def __getattr__(self, key: str):
        try:
            return self.policy.__getattribute__(key)
        except AttributeError:
            return self.policy.__getattr__(key)


class DistributedTrainer:
    def __init__(
        self,
        id,
        local_rank,
        world_size,
        master_addr,
        master_port,
        master_ifname,
        gpu_preload,
        local_queue_size,
        policy_server,
    ):

        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        if master_ifname is not None:
            # eth0,eth1,etc. See https://pytorch.org/docs/stable/distributed.html.
            os.environ["GLOO_SOCKET_IFNAME"] = master_ifname
        distributed.init_process_group("gloo", rank=local_rank, world_size=world_size)

        self.id = id
        self.local_rank = local_rank
        self.world_size = world_size
        self.gpu_preload = gpu_preload
        self.device = torch.device("cuda:0")
        self.cfg = None
        self.local_queue_size = local_queue_size
        self.policy_server = policy_server

        Logger.info("{} (local rank: {}) initialized".format(self.id, local_rank))

    def local_queue_put(self, data):
        if self.gpu_preload:
            data = GPUPreLoadQueueWrapper.to_pin_memory(data)
        try:
            self.local_queue.put(data, block=True, timeout=10)
        except queue.Full:
            Logger.warning("queue is full. May have bugs in training.")

    def local_queue_get(self, timeout=60):
        try:
            data = self.local_queue.get(block=True, timeout=timeout)
        except queue.Empty:
            Logger.warning(
                "queue is empty. May have bugs in rollout. For example, there is no enough data in data server."
            )
            data = None
        return data

    def local_queue_init(self):
        Logger.debug("local queue first prefetch")
        self.local_queue._prefetch_next_batch(block=True)

    def reset(self, training_desc: TrainingDesc):
        self.agent_id = training_desc.agent_id
        self.policy_id = training_desc.policy_id
        self.cfg = training_desc.kwargs["cfg"]
        # pull from policy_server
        policy_desc = ray.get(
            self.policy_server.pull.remote(
                self.id, self.agent_id, self.policy_id, old_version=None
            )
        )
        # wrap policies to distributed ones
        self.policy = DistributedPolicyWrapper(policy_desc.policy, self.local_rank)
        # TODO(jh): trainer may be set in training_desc
        trainer_cls = registry.get(
            registry.TRAINER, policy_desc.policy.registered_name + "Trainer"
        )
        self.trainer = trainer_cls(self.id)
        self.trainer.reset(self.policy, self.cfg)

        self.local_queue = queue.Queue(self.local_queue_size)
        if self.gpu_preload:
            self.local_queue = GPUPreLoadQueueWrapper(self.local_queue)
        Logger.warning("{} reset to training_task {}".format(self.id, training_desc))

    def is_main(self):
        return self.local_rank == 0

    def optimize(self, batch=None):
        global_timer.record("trainer_data_start")
        while batch is None:
            batch = self.local_queue_get()
        global_timer.time("trainer_data_start", "trainer_data_end", "trainer_data")
        global_timer.record("trainer_optimize_start")
        training_info = self.trainer.optimize(batch)
        global_timer.time(
            "trainer_optimize_start", "trainer_optimize_end", "trainer_optimize"
        )
        timer_info = copy.deepcopy(global_timer.elapses)
        global_timer.clear()
        return training_info, timer_info

    def get_unwrapped_policy(self,device="cpu"):
        return self.policy.get_unwrapped_policy(device)

    def push_policy(self, version):
        policy=self.get_unwrapped_policy()

        policy_desc = PolicyDesc(
            self.agent_id,
            self.policy_id,
            policy,
            version=version,
        )

        ray.get(self.policy_server.push.remote(self.id, policy_desc))

    def dump_policy(self):
        pass

    def close(self):
        distributed.destroy_process_group()
