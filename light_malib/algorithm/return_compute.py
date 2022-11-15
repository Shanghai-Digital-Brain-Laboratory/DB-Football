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

import torch
import numpy as np

from light_malib.utils.episode import EpisodeKey
from light_malib.utils.logger import Logger

def compute_return(policy, batch):
    return_mode=policy.custom_config["return_mode"]
    if return_mode == "gae":
        raise NotImplementedError
    elif return_mode == "vtrace":
        raise NotImplementedError
    elif return_mode in ['new_gae','async_gae']:
        return compute_async_gae(policy, batch)
    elif return_mode in ["mc"]:
        return compute_mc(policy,batch)
    else:
        raise ValueError("Unexpected return mode: {}".format(return_mode))

def compute_async_gae(policy,batch):
    '''
    NOTE the last obs,state,done,critic_rnn_states are for bootstraping.
    '''
    with torch.no_grad():
        cfg = policy.custom_config
        gamma, gae_lambda = cfg["gamma"], cfg["gae"]["gae_lambda"]
        rewards=batch[EpisodeKey.REWARD]
        dones=batch[EpisodeKey.DONE]
        cur_obs=batch[EpisodeKey.CUR_OBS]
        rnn_states=batch[EpisodeKey.CRITIC_RNN_STATE]
        
        assert len(rewards.shape) == 4, (rewards.shape, dones.shape)
        B, Tp1, N, _ = cur_obs.shape
        assert rewards.shape[1]==Tp1-1 and dones.shape[1]==Tp1 and rnn_states.shape[1]==Tp1,"{}".format({k:v.shape for k,v in batch.items()})

        obs = cur_obs.reshape((B * Tp1 * N, -1))
        rnn_states = rnn_states.reshape((B * Tp1 * N, *rnn_states.shape[-2:]))
        masks = dones.reshape((B * Tp1 * N, -1))

        normalized_value, _ = policy.critic(
            obs, rnn_states, masks
        )
        normalized_value = normalized_value.reshape((B, Tp1, N, -1)).detach()

        if cfg["use_popart"]:
            values = policy.value_normalizer.denormalize(normalized_value.reshape(-1,normalized_value.shape[-1]))
            values = values.reshape(normalized_value.shape)
        else:
            values = normalized_value


        gae = 0
        advantages = torch.zeros_like(rewards)
        delta_list = torch.zeros_like(rewards)
        for t in reversed(range(Tp1-1)):
            delta = rewards[:,t] + gamma * (1-dones[:,t]) * values[:,t + 1] - values[:,t]
            gae = delta + gamma * gae_lambda * (1-dones[:,t]) * gae
            # TODO(jh): we should differentiate terminal case and truncation case. now we directly follow env's dones
            # gae *= (1-done[t])          #terminal case
            advantages[:,t]=gae
            delta_list[:, t]=delta
        
        returns = advantages + values[:,:-1]
        
        if cfg["use_popart"]:
            normalized_returns = policy.value_normalizer(returns.reshape(-1,rewards.shape[-1]))
            normalized_returns = normalized_returns.reshape(rewards.shape)
        else:
            normalized_returns = returns
        
        advantages = (advantages - advantages.mean()) / (1e-9 + advantages.std())

        ret={
            EpisodeKey.RETURN: normalized_returns,
            EpisodeKey.STATE_VALUE: normalized_value[:,:-1],
            EpisodeKey.ADVANTAGE: advantages,
            'delta': delta_list
        }

        # remove bootstraping data
        for key in [EpisodeKey.CUR_OBS,EpisodeKey.DONE,EpisodeKey.CRITIC_RNN_STATE,EpisodeKey.CUR_STATE]:
            if key in batch:
                ret[key]=batch[key][:,:-1]
            
        return ret
    
def compute_mc(policy,batch):
    with torch.no_grad():
        cfg = policy.custom_config
        gamma = cfg["gamma"]
        rewards=batch[EpisodeKey.REWARD]
        dones=batch[EpisodeKey.DONE]
        cur_obs=batch[EpisodeKey.CUR_OBS]
        rnn_states=batch[EpisodeKey.CRITIC_RNN_STATE]
        
        assert len(rewards.shape) == 4, (rewards.shape, dones.shape)
        B, Tp1, N, _ = cur_obs.shape
        assert rewards.shape[1]==Tp1-1 and dones.shape[1]==Tp1 and rnn_states.shape[1]==Tp1,"{}".format({k:v.shape for k,v in batch.items()})

        # get last step for boostrapping
        obs = cur_obs.reshape((B * Tp1 * N, -1))
        rnn_states = rnn_states.reshape((B * Tp1 * N, *rnn_states.shape[-2:]))
        masks = dones.reshape((B * Tp1 * N, -1))

        normalized_value, _ = policy.critic(
            obs, rnn_states, masks
        )
        normalized_value = normalized_value.reshape((B, Tp1, N, -1)).detach()

        if cfg["use_popart"]:
            values = policy.value_normalizer.denormalize(normalized_value.reshape(-1,normalized_value.shape[-1]))
            values = values.reshape(normalized_value.shape)
        else:
            values = normalized_value


        ret = 0
        advantages = torch.zeros_like(rewards)
        for t in reversed(range(Tp1-1)):
            ret=gamma*(1-dones[:,t])*ret+rewards[:,t]
            if t==Tp1-1-1:
                # bootstrapping values
                ret+=(1-dones[:,t])*values[:,t+1]
            advantages[:,t]=ret-values[:,t]
        
        returns = advantages + values[:,:-1]
        
        if cfg["use_popart"]:
            normalized_returns = policy.value_normalizer(returns.reshape(-1,rewards.shape[-1]))
            normalized_returns = normalized_returns.reshape(rewards.shape)
        else:
            normalized_returns = returns
        
        advantages = (advantages - advantages.mean()) / (1e-9 + advantages.std())

        ret={
            EpisodeKey.RETURN: normalized_returns,
            EpisodeKey.STATE_VALUE: normalized_value[:,:-1],
            EpisodeKey.ADVANTAGE: advantages
        }

        # remove bootstraping data
        for key in [EpisodeKey.CUR_OBS,EpisodeKey.DONE,EpisodeKey.CRITIC_RNN_STATE,EpisodeKey.CUR_STATE]:
            if key in batch:
                ret[key]=batch[key][:,:-1]
            
        return ret