# MIT License

# Copyright (c) 2022 DigitalBrain, Yan Song and He jiang
# Copyright (c) 2021 MARL @ SJTU

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

        global_timer.record("move_to_gpu_start")
        # move data to gpu
        for key, value in batch.items():
            if isinstance(value, np.ndarray):
                value = torch.FloatTensor(value)
            batch[key] = value.to(policy.device)
        global_timer.time("move_to_gpu_start", "move_to_gpu_end", "move_to_gpu")

        global_timer.record("compute_return_start")
        new_data = compute_return(policy, batch)
        batch.update(new_data)
        global_timer.time(
            "compute_return_start", "compute_return_end", "compute_return"
        )

        ppo_epoch = policy.custom_config["ppo_epoch"]
        num_mini_batch = policy.custom_config["num_mini_batch"]  # num_mini_batch
        kl_early_stop = policy.custom_config.get("kl_early_stop", None)
        assert (
            kl_early_stop is None
        ), "TODO(jh): kl early stop is not supported is current distributed implmentation."

        if policy.custom_config["use_rnn"]:
            data_generator_fn = functools.partial(
                recurrent_generator,
                batch,
                num_mini_batch,
                policy.custom_config["rnn_data_chunk_length"],
                policy.device,
            )
        else:
            data_generator_fn = functools.partial(
                simple_data_generator, batch, num_mini_batch, policy.device
            )

        # jh: special optimization
        if num_mini_batch == 1:
            global_timer.record("data_generator_start")
            mini_batch = next(data_generator_fn())
            global_timer.time(
                "data_generator_start", "data_generator_end", "data_generator"
            )
            kl_diff = 0
            for i_epoch in range(ppo_epoch):
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

        else:
            kl_diff = 0
            for i_epoch in range(ppo_epoch):
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
