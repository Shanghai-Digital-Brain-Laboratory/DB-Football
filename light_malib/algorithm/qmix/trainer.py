import torch


import numpy as np
from collections import defaultdict
from light_malib.registry import registry
from light_malib.training.data_generator import (
    recurrent_generator,
    simple_data_generator,
    simple_team_data_generator
)
from .loss import QMIXLoss
import torch
import functools
from light_malib.utils.logger import Logger
from light_malib.utils.timer import global_timer
from ..return_compute import compute_return
from ..common.trainer import Trainer

from .q_mixer import QMixer

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

@registry.registered(registry.TRAINER)
class QMixTrainer(Trainer):
    def __init__(self, tid):
        super().__init__(tid)
        self.id=tid
        self._loss = QMIXLoss()

    def optimize(self, batch):      #[batch, 1, num_agent, feat_dim]
        total_opt_result = defaultdict(lambda: 0)
        policy = self.loss.policy
        # move data to gpu
        for key,value in batch.items():
            if isinstance(value,np.ndarray):
                value=torch.FloatTensor(value)
            batch[key]=value.to(policy.device)

        num_mini_batch=policy.custom_config["num_mini_batch"]
        data_generator_fn = functools.partial(
            simple_team_data_generator, batch, num_mini_batch, policy.device, shuffle=False
        )

        data_iter = data_generator_fn()
        for i in range(num_mini_batch):
            mini_batch = next(data_iter)
            tmp_opt_result = self.loss(mini_batch)
            for k, v in tmp_opt_result.items():
                total_opt_result[k] = v

        return total_opt_result

    def reset(self, policies, training_config):

        qmix_config = policies.policy.custom_config['qmixer_config']

        if self.loss.mixer is None:
            self.loss.set_mixer(QMixer(qmix_config, None, None))
            self.loss.set_mixer_target(QMixer(qmix_config, None, None))

            with torch.no_grad():
                for target_param, param in zip(self.loss.mixer_target.parameters(), self.loss.mixer.parameters()):
                    target_param.data.copy_(param.data)

        super(QMixTrainer, self).reset(policies, training_config)

    def preprocess(self, batch, **kwargs):
        pass
