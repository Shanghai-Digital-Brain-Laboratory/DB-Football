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

from typing import Union
import torch
from light_malib.utils.episode import EpisodeKey
from light_malib.algorithm.common.loss_func import LossFunc
from light_malib.utils.logger import Logger
from light_malib.registry import registry
import numpy as np

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
        self._use_max_grad_norm = True

    def reset(self, policy, config):
        """
        reset should always be called for each training task.
        """
        self._params.update(config)
        if policy is not self.policy:
            self._policy = policy
            # self._set_centralized_critic()
            self.setup_optimizers()
        
        self.clip_param = policy.custom_config.get("clip_param", 0.2)
        self.max_grad_norm = policy.custom_config.get("max_grad_norm", 10)

        self.sub_algorithm_name = policy.custom_config.get("sub_algorithm_name","MAPPO")   
        assert self.sub_algorithm_name in ["MAPPO","CoPPO","HAPPO","A2PO"]
        
        if self.sub_algorithm_name=="MAPPO":
            self._use_seq=False
            self._use_two_stage=False
            self._use_co_ma_ratio=False
            self._clip_before_prod=False
            self._clip_others=False
        elif self.sub_algorithm_name=="CoPPO":
            self._use_seq=False
            self._use_two_stage=False
            self._use_co_ma_ratio=True
            self._clip_before_prod=True
            self._clip_others=True
            self._other_clip_param=policy.custom_config["other_clip_param"]
            self._num_agents=policy.custom_config["num_agents"]
        elif self.sub_algorithm_name=="HAPPO":
            self._use_seq=True
            self._use_two_stage=False
            self._use_co_ma_ratio=True
            self._clip_before_prod=True
            self._clip_others=False
            self._num_agents=policy.custom_config["num_agents"]
            self._seq_strategy=policy.custom_config.get("seq_strategy","random")
            # TODO(jh): check default
            self._one_agent_per_update=False
            self._use_agent_block=policy.custom_config.get("use_agent_block",False)
            if self._use_agent_block:
                self._block_num=policy.custom_config["block_num"]
            self._use_cum_sequence=True
            self._agent_seq=[]
        elif self.sub_algorithm_name=="A2PO":
            self._use_seq=True
            self._use_two_stage=True
            self._use_co_ma_ratio=True
            self._clip_before_prod=False
            self._clip_others=True
            self._other_clip_param=policy.custom_config["other_clip_param"]
            self._num_agents=policy.custom_config["num_agents"]
            self._seq_strategy=policy.custom_config.get("seq_strategy","semi_greedy")
            # TODO(jh): check default
            self._one_agent_per_update=False
            self._use_agent_block=policy.custom_config.get("use_agent_block",False)
            if self._use_agent_block:
                self._block_num=policy.custom_config["block_num"]
            self._use_cum_sequence=True
            self._agent_seq=[]
        else:
            raise NotImplementedError     
            
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
        if self._use_seq:
            return self.loss_compute_sequential(sample)
        else:
            return self.loss_compute_simultaneous(sample)
            
    def _select_data_from_agent_ids(
        self,
        x: Union[np.ndarray, torch.Tensor],
        agent_ids: np.ndarray
    ) -> Union[np.ndarray, torch.Tensor]:
        '''
        we assume x is the shape [#batch_size*#agents,...]
        '''
        if agent_ids is None:
            return x        
        
        if not isinstance(x,(np.ndarray,torch.Tensor)):
            return x
        
        x = x.reshape(-1, self._num_agents, *x.shape[1:])[:, agent_ids]
        x = x.reshape(-1,*x.shape[2:])
        return x

    def loss_compute_simultaneous(
        self, 
        sample,
        agent_ids:np.ndarray=None,
        update_actor:bool=True
    ):
        # agent_ids not None means block update
        if agent_ids is not None:
            assert len(agent_ids.shape)==1
        
        (
            share_obs_batch,
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
            sample[EpisodeKey.CUR_STATE],
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

        if update_actor:
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
                inference=False,
                explore=False
            )
            
            values=ret[EpisodeKey.STATE_VALUE]
            action_log_probs=ret[EpisodeKey.ACTION_LOG_PROB]
            dist_entropy=ret[EpisodeKey.ACTION_ENTROPY]     
            
             # ============================== Policy Loss ================================
            imp_weights = torch.exp(
                action_log_probs - old_action_log_probs_batch
            ).view(-1,1)
            approx_kl = (
                (old_action_log_probs_batch - action_log_probs).mean().item()
            )
        
            # CoPPO, HAPPO, A2PO
            if self._use_co_ma_ratio:
                each_agent_imp_weights = imp_weights.reshape(
                    -1, self._num_agents, 1
                )
                # NOTE(jh): important to detach, so gradients won't flow back from other agents' policy update
                each_agent_imp_weights = each_agent_imp_weights.detach()
                
                mask_self = torch.eye(self._num_agents,device=each_agent_imp_weights.device,dtype=torch.bool)
                mask_self = mask_self[agent_ids]
                
                # (#selected_agents,#agents,1)
                mask_self = mask_self.unsqueeze(-1)
                
                # (#batch,1,#agents,1)
                each_agent_imp_weights = each_agent_imp_weights.unsqueeze(1)
                # (#batch,#selected_agents,#agents,1)
                if agent_ids is None:
                    repeats=self._num_agents
                else:
                    repeats=len(agent_ids)
                each_agent_imp_weights = each_agent_imp_weights.repeat_interleave(repeats,dim=1)
                each_agent_imp_weights[..., mask_self] = 1.0
                
                # (#batch,#selected_agents,1)
                other_agents_prod_imp_weights = each_agent_imp_weights.prod(dim=2)
                
                # CoPPO, A2PO
                if self._clip_others:
                    other_agents_prod_imp_weights = torch.clamp(
                        other_agents_prod_imp_weights,
                        1.0-self._other_clip_param,
                        1.0+self._other_clip_param
                    )

                other_agents_prod_imp_weights = other_agents_prod_imp_weights.reshape(-1, 1)
                
            imp_weights = self._select_data_from_agent_ids(imp_weights, agent_ids)
            adv_targ = self._select_data_from_agent_ids(adv_targ, agent_ids)
            active_masks_batch = self._select_data_from_agent_ids(active_masks_batch,agent_ids)
            dist_entropy = self._select_data_from_agent_ids(dist_entropy, agent_ids)
            
            # CoPPO, A2PO
            if not self._clip_before_prod:
                imp_weights = imp_weights * other_agents_prod_imp_weights
        
            surr1 = imp_weights * adv_targ
            surr2 = (
                torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param)
                * adv_targ
            )
            
            # HAPPO
            if self._clip_before_prod:
                surr1 = surr1 * other_agents_prod_imp_weights
                surr2 = surr2 * other_agents_prod_imp_weights

            if active_masks_batch is not None:
                surr = torch.min(surr1, surr2)
                policy_action_loss = (
                    -torch.sum(surr, dim=-1, keepdim=True) * active_masks_batch
                ).sum() / (active_masks_batch.sum()+1e-20)
                assert dist_entropy.shape==active_masks_batch.shape
                policy_entropy_loss = - (dist_entropy*active_masks_batch).sum()/(active_masks_batch.sum()+1e-20)
            else:
                surr = torch.min(surr1, surr2)
                policy_action_loss = -torch.sum(surr, dim=-1, keepdim=True).mean()
                policy_entropy_loss = - dist_entropy.mean()

            policy_loss = policy_action_loss + policy_entropy_loss * self._policy.custom_config["entropy_coef"]

        else:
            ret = self._policy.value_function(
                **{
                    EpisodeKey.CUR_STATE: share_obs_batch,
                    EpisodeKey.CUR_OBS: obs_batch,
                    EpisodeKey.CRITIC_RNN_STATE: critic_rnn_states_batch,
                    EpisodeKey.DONE: dones_batch
                },
                inference=False
            )
            values=ret[EpisodeKey.STATE_VALUE]
            
            policy_loss = 0
            active_masks_batch = self._select_data_from_agent_ids(active_masks_batch, agent_ids)
        
        # ============================== Value Loss ================================
       
        values = self._select_data_from_agent_ids(values, agent_ids)
        value_preds_batch = self._select_data_from_agent_ids(value_preds_batch, agent_ids)
        return_batch = self._select_data_from_agent_ids(return_batch, agent_ids)
       
        value_loss = self._calc_value_loss(
            values, value_preds_batch, return_batch, active_masks_batch
        )

        # ============================== Total Loss ================================        
        total_loss = policy_loss + value_loss * self._policy.custom_config.get("value_loss_coef",1.0)

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
        if update_actor:
            # TODO(jh): miss active masks?
            stats = dict(
                ratio=float(imp_weights.detach().mean().cpu().numpy()),
                ratio_std=float(imp_weights.detach().std().cpu().numpy()),
                policy_loss=float(policy_loss.detach().cpu().numpy()),
                value_loss=float(value_loss.detach().cpu().numpy()),
                entropy=float(dist_entropy.detach().mean().cpu().numpy()),
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
        else:
            stats = {}
            
        return stats
    
    def loss_compute_sequential(self, sample):
        '''
        NOTE(jh): sharing policy is actually not suggested in sequentially-updating agorithm.
        the reason is the update of one agent will also affect others' policies that is not carefully analized.
        but as an approximation used in practice, it might be acceptable. so we don't restrict it.
        '''
        (
            value_preds_batch,
            adv_targ,
        ) = (
            sample[EpisodeKey.STATE_VALUE],
            sample[EpisodeKey.ADVANTAGE],
        )
        
        if not self._one_agent_per_update:
            self._agent_seq=self._get_agent_sequence(adv_targ, value_preds_batch)
        elif self._one_agent_per_update and len(self._agent_seq) == 0:
            self._agent_seq = self._get_agent_sequence(adv_targ, value_preds_batch)
            
        stats = {}
        for a_ids in self._agent_seq:
            if self._use_two_stage:
                self.loss_compute_simultaneous(sample, a_ids, update_actor=False)
            _stats = self.loss_compute_simultaneous(sample, a_ids)
            for k, v in _stats.items():
                if k in stats:
                    stats[k] += v
                else:
                    stats[k] = v
            if self._one_agent_per_update:
                self._agent_seq.pop(0)
                return stats
        for k, v in stats.items():
            stats[k] = v / len(self._agent_seq)
        return stats

    def _get_agent_sequence(self, adv_targ, value_preds_batch):
        # size (bsz, num_agents, ...)
        if self._seq_strategy == "random":
            seq = np.random.permutation(self._num_agents)
        elif self._seq_strategy in ["semi_greedy","greedy"]:
            adv_targ = adv_targ.reshape(-1, self._num_agents, *adv_targ.shape[1:])
            value_preds_batch = value_preds_batch.reshape(
                -1, self._num_agents, *value_preds_batch.shape[1:]
            )
            score = np.abs(
                adv_targ.cpu().numpy() / (value_preds_batch.cpu().numpy() + 1e-8)
            )
            score = np.mean(score, axis=0)
            score = np.sum(score, axis=score.shape[1:])
            id_scores = [(_i, _s) for (_i, _s) in zip(range(self._num_agents), score)]
            id_scores = sorted(id_scores, key=lambda x: x[1], reverse=True)
            if self._seq_strategy=="semi_greedy":
                # print("semi")
                seq = []
                a_i = 0
                while a_i < self._num_agents:
                    seq.append(id_scores[0][0])
                    id_scores.pop(0)
                    a_i += 1
                    if len(id_scores) > 0:
                        next_i = np.random.choice(len(id_scores))
                        seq.append(id_scores[next_i][0])
                        id_scores.pop(next_i)
                        a_i += 1
                seq = np.array(seq)
            else:
                seq = np.array([_i for (_i, _s) in id_scores])
        else:
            raise NotImplementedError("you can only select random, semi_greedy or greedy as your seq_strategy now.")
        if self._use_agent_block:
            _seq = np.array_split(seq, self._block_num)
        else:
            _seq = seq.reshape(-1, 1)

        if self._use_cum_sequence:
            seq = []
            for s_i in range(len(_seq)):
                seq.append(np.concatenate(_seq[: s_i + 1]))
        else:
            seq = _seq
        return seq

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
            ).sum() / (active_masks_batch.sum()+1e-20)
        else:
            value_loss = value_loss.mean()

        return value_loss

    def zero_grad(self):
        pass

    def step(self):
        pass
