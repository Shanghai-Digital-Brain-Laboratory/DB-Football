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

import torch
from light_malib.utils.episode import EpisodeKey
from light_malib.algorithm.common.loss_func import LossFunc
from light_malib.utils.logger import Logger
from light_malib.registry import registry

def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a * e**2 / 2 + b * d * (abs(e) - d / 2)


def mse_loss(e):
    return (e**2) / 2


def to_value(tensor: torch.Tensor):
    return tensor.detach().cpu().item()


def basic_stats(name, tensor: torch.Tensor):
    stats = {}
    stats["{}_max".format(name)] = to_value(tensor.max())
    stats["{}_min".format(name)] = to_value(tensor.min())
    stats["{}_mean".format(name)] = to_value(tensor.mean())
    stats["{}_std".format(name)] = to_value(tensor.std())
    return stats


@registry.registered(registry.LOSS)
class MAPPOLoss(LossFunc):
    def __init__(self):
        # TODO: set these values using custom_config
        super(MAPPOLoss, self).__init__()

        self._use_clipped_value_loss = True
        self._use_huber_loss = True
        if self._use_huber_loss:
            self.huber_delta = 10.0
        # self._use_value_active_masks = False
        # self._use_policy_active_masks = False

        self._use_max_grad_norm = True

        # the following are useless now
        self.inner_clip_param = 0.1
        self.use_modified_mappo = False
        self.use_inner_clip = False
        # TODO double clipping Tencent
        self.use_double_clip = False
        self.double_clip_param = 3

    def reset(self, policy, config):
        """Replace critic with a centralized critic"""
        self._params.update(config)
        if policy is not self.policy:
            self._policy = policy
            # self._set_centralized_critic()
            self.setup_optimizers()

    def setup_optimizers(self, *args, **kwargs):
        """Accept training configuration and setup optimizers"""
        optim_cls = getattr(torch.optim, self._params.get("optimizer", "Adam"))
        
        # TODO(jh): update actor and critic simutaneously
        param_groups=[]
        
        if len(list(self._policy.actor.parameters()))>0:
            param_groups.append({'params': self.policy.actor.parameters(), 'lr': self._params["actor_lr"]})
        
        if len(list(self._policy.critic.parameters()))>0:
            param_groups.append({'params': self.policy.critic.parameters(), 'lr': self._params["critic_lr"]})
        
        if self._policy.share_backbone and len(list(self._policy.backbone.parameters()))>0:
            param_groups.append({'params': self.policy.backbone.parameters(), 'lr': self._params["backbone_lr"]})
            
        self.optimizer=optim_cls(
            param_groups,
            eps=self._params["opti_eps"],
            weight_decay=self._params["weight_decay"]
        )

    def loss_compute(self, sample):
        policy = self._policy
        policy.train()
        
        self.clip_param = policy.custom_config.get("clip_param", 0.2)
        self.max_grad_norm = policy.custom_config.get("max_grad_norm", 10)

        self.use_modified_mappo = policy.custom_config.get("use_modified_mappo", False)

        (
            obs_batch,
            actions_batch,
            value_preds_batch,
            return_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            available_actions_batch,
            actor_rnn_states_batch,
            critic_rnn_states_batch,
            dones_batch,
            adv_targ,
            delta,
        ) = (
            sample[EpisodeKey.CUR_OBS],
            sample[EpisodeKey.ACTION].long(),
            sample[EpisodeKey.STATE_VALUE],
            sample[EpisodeKey.RETURN],
            sample.get(EpisodeKey.ACTIVE_MASK, None),
            sample[EpisodeKey.ACTION_LOG_PROB],
            sample[EpisodeKey.ACTION_MASK],
            sample[EpisodeKey.ACTOR_RNN_STATE],
            sample[EpisodeKey.CRITIC_RNN_STATE],
            sample[EpisodeKey.DONE],
            sample[EpisodeKey.ADVANTAGE],
            sample["delta"],
        )
        if EpisodeKey.CUR_STATE in sample:
            share_obs_batch = sample[EpisodeKey.CUR_STATE]
        else:
            share_obs_batch = sample[EpisodeKey.CUR_OBS]

        values, action_log_probs, dist_entropy = self._evaluate_actions(
            share_obs_batch,
            obs_batch,
            actions_batch,
            available_actions_batch,
            actor_rnn_states_batch,
            critic_rnn_states_batch,
            dones_batch,
            active_masks_batch,
        )

        imp_weights = torch.exp(
            action_log_probs - old_action_log_probs_batch
        ).view(-1,1)
        approx_kl = (
            (old_action_log_probs_batch - action_log_probs).mean().item()
        )

        if self.use_modified_mappo:
            if self.use_inner_clip:
                o_imp_weights = imp_weights + 1e-9 * (imp_weights == 0)
            # #env*#agent
            imp_weights = imp_weights.view(-1, n_agent)
            batch_size, n_agent = imp_weights.shape
            imp_weights = torch.prod(imp_weights, dim=-1, keepdim=True)
            imp_weights = torch.tile(imp_weights, (1, n_agent))
            imp_weights = imp_weights.view(batch_size * n_agent, 1)
            if self.use_inner_clip:
                imp_weights /= o_imp_weights
                imp_weights = torch.clamp(
                    imp_weights,
                    1.0 - self.inner_clip_param,
                    1.0 + self.inner_clip_param,
                )
                imp_weights *= o_imp_weights

        surr1 = imp_weights * adv_targ
        surr2 = (
            torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param)
            * adv_targ
        )

        if self.use_double_clip:
            surr3 = self.double_clip_param * adv_targ

        if active_masks_batch is not None:
            surr = torch.min(surr1, surr2)
            if self.use_double_clip:
                surr = torch.max(surr, surr3)
            policy_action_loss = (
                -torch.sum(surr, dim=-1, keepdim=True) * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            surr = torch.min(surr1, surr2)
            if self.use_double_clip:
                mask = (adv_targ < 0).float()
                surr = torch.max(surr, surr3) * mask + surr * (1 - mask)
            policy_action_loss = -torch.sum(surr, dim=-1, keepdim=True).mean()
 
        # ============================== Loss ================================
        policy_loss = (
            policy_action_loss
            - dist_entropy * self._policy.custom_config["entropy_coef"]
        )

        value_loss = self._calc_value_loss(
            values, value_preds_batch, return_batch, active_masks_batch
        )
        
        # TODO(jh): add balacing coefficients?
        total_loss=policy_loss+value_loss

        # ============================== Optimizer ================================
        self.optimizer.zero_grad()
        total_loss.backward()        
        if self._use_max_grad_norm:
            for param_group in self.optimizer.param_groups:
                torch.nn.utils.clip_grad_norm_(
                    param_group["params"], self.max_grad_norm
                )
        self.optimizer.step()

        # ============================== Statistics ================================
        stats = dict(
            ratio=float(imp_weights.detach().mean().cpu().numpy()),
            ratio_std=float(imp_weights.detach().std().cpu().numpy()),
            policy_loss=float(policy_loss.detach().cpu().numpy()),
            value_loss=float(value_loss.detach().cpu().numpy()),
            entropy=float(dist_entropy.detach().cpu().numpy()),
            approx_kl=approx_kl,
        )

        stats.update(basic_stats("imp_weights", imp_weights))
        stats.update(basic_stats("advantages", adv_targ))
        stats.update(basic_stats("V", values))
        stats.update(basic_stats("Old_V", value_preds_batch))
        stats.update(basic_stats("delta", delta))

        stats["upper_clip_ratio"] = to_value(
            (imp_weights > (1 + self.clip_param)).float().mean()
        )
        stats["lower_clip_ratio"] = to_value(
            (imp_weights < (1 - self.clip_param)).float().mean()
        )
        stats["clip_ratio"] = stats["upper_clip_ratio"] + stats["lower_clip_ratio"]
        return stats

    def _evaluate_actions(
        self,
        share_obs_batch,
        obs_batch,
        actions_batch,
        available_actions_batch,
        actor_rnn_states_batch,
        critic_rnn_states_batch,
        dones_batch,
        active_masks_batch,
    ):
        ret = self._policy.compute_action(
            **{
            EpisodeKey.CUR_STATE: share_obs_batch,
            EpisodeKey.CUR_OBS: obs_batch,
            EpisodeKey.ACTION: actions_batch,
            EpisodeKey.ACTOR_RNN_STATE: actor_rnn_states_batch,
            EpisodeKey.CRITIC_RNN_STATE: critic_rnn_states_batch,
            EpisodeKey.DONE: dones_batch,
            EpisodeKey.ACTION_MASK: available_actions_batch  
            },
            explore=False
        )
        
        values=ret[EpisodeKey.STATE_VALUE]
        action_log_probs=ret[EpisodeKey.ACTION_LOG_PROB]
        dist_entropy=ret[EpisodeKey.ACTION_ENTROPY].mean()

        return values, action_log_probs, dist_entropy

    def _calc_value_loss(
        self, values, value_preds_batch, return_batch, active_masks_batch=None
    ):
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
            -self.clip_param, self.clip_param
        )
        error_clipped = return_batch - value_pred_clipped
        error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if active_masks_batch is not None:
            value_loss = (
                value_loss * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def zero_grad(self):
        pass

    def step(self):
        pass
