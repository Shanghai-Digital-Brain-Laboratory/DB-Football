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

import numpy as np
from collections import defaultdict
from light_malib.training.data_generator import (
    recurrent_generator,
    simple_data_generator,
)
from .loss import MAPPOLoss
import torch
import functools
from light_malib.utils.logger import Logger
from light_malib.utils.timer import global_timer
from ..return_compute import compute_return
from ..common.trainer import Trainer
from light_malib.registry import registry
from light_malib.utils.episode import EpisodeKey

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


@registry.registered(registry.TRAINER)
class MAPPOTrainer(Trainer):
    def __init__(self, tid):
        super().__init__(tid)
        self.id = tid
        # TODO(jh)
        self._loss = MAPPOLoss()

    def optimize(self, batch, **kwargs):
        total_opt_result = defaultdict(lambda: 0)
        policy = self.loss.policy
        
        ppo_epoch = policy.custom_config["ppo_epoch"]
        num_mini_batch = policy.custom_config["num_mini_batch"]  # num_mini_batch
        kl_early_stop = policy.custom_config.get("kl_early_stop", None)
        assert (
            kl_early_stop is None
        ), "TODO(jh): kl early stop is not supported is current distributed implmentation."

        # move data to gpu
        global_timer.record("move_to_gpu_start")
        for key, value in batch.items():
            if isinstance(value, np.ndarray):
                value = torch.FloatTensor(value)
            batch[key] = value.to(policy.device)
        
        if EpisodeKey.CUR_STATE not in batch:
            batch[EpisodeKey.CUR_STATE]=batch[EpisodeKey.CUR_OBS]
        global_timer.time("move_to_gpu_start", "move_to_gpu_end", "move_to_gpu")

        kl_diff = 0
        for i_epoch in range(ppo_epoch):
            # NOTE(jh): for backward compatibility, when return_mode="new_gae", only call return_compute once.
            if i_epoch==0 or policy.custom_config["return_mode"] in ["new_gae_trace"]:
                batch_with_return=self._compute_return(policy, batch)        
            
            data_generator_fn=self._get_data_generator(policy, batch_with_return, num_mini_batch)
                
            for mini_batch in data_generator_fn():
                global_timer.record("loss_start")
                tmp_opt_result = self.loss(mini_batch)
                global_timer.time("loss_start", "loss_end", "loss")
                for k, v in tmp_opt_result.items():
                    total_opt_result[k] = v

            if i_epoch == 0:
                start_kl = tmp_opt_result["approx_kl"]
            else:
                kl_diff += tmp_opt_result["approx_kl"] - start_kl
                start_kl = tmp_opt_result["approx_kl"]

            if (
                kl_early_stop is not None
                and tmp_opt_result["approx_kl"] > kl_early_stop
            ):
                break

            total_opt_result["kl_diff"] = kl_diff
            total_opt_result["training_epoch"] = i_epoch + 1

        # TODO(ziyu & ming): find a way for customize optimizer and scheduler
        # but now it doesn't affect the performance ...

        # TODO(jh)
        # if kwargs["lr_decay"]:
        #     epoch = kwargs["rollout_epoch"]
        #     total_epoch = kwargs["lr_decay_epoch"]
        #     assert total_epoch is not None
        #     update_linear_schedule(
        #         self.loss.optimizers["critic"],
        #         epoch,
        #         total_epoch,
        #         self.loss._params["critic_lr"],
        #     )
        #     update_linear_schedule(
        #         self.loss.optimizers["actor"],
        #         epoch,
        #         total_epoch,
        #         self.loss._params["actor_lr"],
        #     )

        return total_opt_result

    def preprocess(self, batch, **kwargs):
        pass

    def _compute_return(self, policy, batch):
        # compute return
        global_timer.record("compute_return_start")
        new_batch = compute_return(policy, batch)
        global_timer.time(
            "compute_return_start", "compute_return_end", "compute_return"
        )
        return new_batch        

    def _get_data_generator(self, policy, new_batch, num_mini_batch):
        # build data generator
        if policy.custom_config["use_rnn"]:
            data_generator_fn = functools.partial(
                recurrent_generator,
                new_batch,
                num_mini_batch,
                policy.custom_config["rnn_data_chunk_length"],
                policy.device,
            )
        else:
            data_generator_fn = functools.partial(
                simple_data_generator, new_batch, num_mini_batch, policy.device
            )
        
        return data_generator_fn