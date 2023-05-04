from typing import Union
import torch
from light_malib.utils.episode import EpisodeKey
from light_malib.algorithm.common.loss_func import LossFunc
from light_malib.utils.logger import Logger
from light_malib.registry import registry
import numpy as np
import torch.nn.functional as F




@registry.registered(registry.LOSS)
class BCLoss(LossFunc):
    def __init__(self):
        # TODO: set these values using custom_config
        super(BCLoss, self).__init__()

    def reset(self, policy, config):
        self._params.update(config)
        if policy is not self.policy:
            self._policy = policy
            # self._set_centralized_critic()
            self.setup_optimizers()

    def setup_optimizers(self, *args, **kwargs):
        """Accept training configuration and setup optimizers"""
        optim_cls = getattr(torch.optim, self._params.get("optimizer", "Adam"))

        param_groups = []
        if len(list(self._policy.target_policy.actor.parameters()))>0:
            param_groups.append({
                'params': self.policy.target_policy.actor.parameters(),
                'lr': self._params["lr"]
            })

        self.optimizer = optim_cls(
            param_groups,
            eps=self._params['opti_eps'],
            weight_decay = self._params['weight_decay']
        )

    def loss_compute(self, sample):
        policy = self._policy.target_policy
        policy.train()
        cutoff_idx = self._policy.cutoff_idx
        self.max_grad_norm = policy.custom_config.get("max_grad_norm",10)

        (obs_batch, actions_batch, action_masks) = (
            sample[EpisodeKey.CUR_OBS],
            sample[EpisodeKey.ACTION].long(),
            sample[EpisodeKey.ACTION_MASK]
        )
        obs_batch = obs_batch[...,cutoff_idx:]

        action_logits_batch, _ = self._policy.target_policy.actor.logits(
            obs_batch,
            None,
            None
        )
        policy_loss = F.cross_entropy(action_logits_batch.reshape(-1, action_logits_batch.shape[-1]), actions_batch.reshape(-1))

        self.optimizer.zero_grad()
        policy_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self._policy.target_policy.actor.parameters(), self.max_grad_norm
        )
        self.optimizer.step()

        stats = dict(
            policy_loss=float(policy_loss.detach().item())
        )
        return stats

    def step(self):
        pass
