import numpy as np
from collections import defaultdict
from light_malib.training.data_generator import (
    recurrent_generator,
    simple_data_generator,
    dummy_data_generator
)
from .loss import BCLoss
import torch
import functools
from light_malib.utils.logger import Logger
from light_malib.utils.timer import global_timer
from ..common.trainer import Trainer
from light_malib.registry import registry
from light_malib.utils.episode import EpisodeKey

def drop_bootstraping_data(batch):
    for key in [
        EpisodeKey.CUR_OBS,
        EpisodeKey.DONE,
        EpisodeKey.CRITIC_RNN_STATE,
        EpisodeKey.CUR_STATE,
    ]:
        if key in batch:
            batch[key] = batch[key][:,:-1,...]

    return batch

@registry.registered(registry.TRAINER)
class BCTrainer(Trainer):
    def __init__(self, tid):
        super().__init__(tid)
        self.id = tid

        self._loss = BCLoss()

    def optimize(self, batch, **kwargs):
        total_opt_result = defaultdict(lambda: 0)
        policy = self.loss.policy

        if len(batch[EpisodeKey.CUR_OBS].shape)==4:         #if traj mode
            batch = drop_bootstraping_data(batch)

        # move data to gpu
        # for key,value in batch.items():
        #     if isinstance(value,np.ndarray):
        #         value=torch.FloatTensor(value)
        #     batch[key]=value.to(policy.device)
        # global_timer.time("move_to_gpu_start","move_to_gpu_end","move_to_gpu")

        epoch = policy.custom_config['bc_epoch']
        num_mini_batch= 1 #policy.custom_config["num_mini_batch"]

        for _ in range(epoch):
            data_generator_fn = functools.partial(
                dummy_data_generator, batch, num_mini_batch, policy.device, shuffle=False
            )

            data_iter = data_generator_fn()
            for i in range(num_mini_batch):
                mini_batch = next(data_iter)

                for key,value in mini_batch.items():
                    if isinstance(value,np.ndarray):
                        value=torch.FloatTensor(value)
                    mini_batch[key]=value.to(policy.device)

                tmp_opt_result = self.loss(mini_batch)

                for k, v in tmp_opt_result.items():
                    total_opt_result[k] = v

        return total_opt_result

    def preprocess(self, batch, **kwargs):
        pass

