#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2023 The TARTRL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Categorical

import gym

def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output

class FcEncoder(nn.Module):
    def __init__(self, fc_num, input_size, output_size):
        super(FcEncoder, self).__init__()
        self.first_mlp = nn.Sequential(
                nn.Linear(input_size, output_size), nn.ReLU(), nn.LayerNorm(output_size)
            )
        self.mlp = nn.Sequential()
        for _ in range(fc_num - 1):
            self.mlp.append(nn.Sequential(
                nn.Linear(output_size, output_size), nn.ReLU(), nn.LayerNorm(output_size)
            ))

    def forward(self, x):
        output = self.first_mlp(x)
        return self.mlp(output)

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module


class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)

class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(Categorical, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, available_actions=None):
        x = self.linear(x)
        if available_actions is not None:
            x[available_actions == 0] = -1e10
        return FixedCategorical(logits=x)


class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias

class ACTLayer(nn.Module):
    def __init__(self, action_space, inputs_dim, use_orthogonal, gain):
        super(ACTLayer, self).__init__()
        self.multidiscrete_action = False
        self.continuous_action = False
        self.mixed_action = False

        action_dim = action_space.n
        self.action_out = Categorical(inputs_dim, action_dim, use_orthogonal, gain)


    
    def forward(self, x, available_actions=None, deterministic=False):
        if self.mixed_action :
            actions = []
            action_log_probs = []
            for action_out in self.action_outs:
                action_logit = action_out(x)
                action = action_logit.mode() if deterministic else action_logit.sample()
                action_log_prob = action_logit.log_probs(action)
                actions.append(action.float())
                action_log_probs.append(action_log_prob)

            actions = torch.cat(actions, -1)
            action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)

        elif self.multidiscrete_action:
            actions = []
            action_log_probs = []
            for action_out in self.action_outs:
                action_logit = action_out(x)
                action = action_logit.mode() if deterministic else action_logit.sample()
                action_log_prob = action_logit.log_probs(action)
                actions.append(action)
                action_log_probs.append(action_log_prob)

            actions = torch.cat(actions, -1)
            action_log_probs = torch.cat(action_log_probs, -1)
        
        elif self.continuous_action:
            action_logits = self.action_out(x)
            actions = action_logits.mode() if deterministic else action_logits.sample() 
            action_log_probs = action_logits.log_probs(actions)
        
        else:
            action_logits = self.action_out(x, available_actions)
            actions = action_logits.mode() if deterministic else action_logits.sample() 
            action_log_probs = action_logits.log_probs(actions)
        
        return actions, action_log_probs

    def get_probs(self, x, available_actions=None):
        if self.mixed_action or self.multidiscrete_action:
            action_probs = []
            for action_out in self.action_outs:
                action_logit = action_out(x)
                action_prob = action_logit.probs
                action_probs.append(action_prob)
            action_probs = torch.cat(action_probs, -1)
        elif self.continuous_action:
            action_logits = self.action_out(x)
            action_probs = action_logits.probs
        else:
            action_logits = self.action_out(x, available_actions)
            action_probs = action_logits.probs
        
        return action_probs

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None, get_probs=False):
        if self.mixed_action:
            a, b = action.split((2, 1), -1)
            b = b.long()
            action = [a, b] 
            action_log_probs = [] 
            dist_entropy = []
            for action_out, act in zip(self.action_outs, action):
                action_logit = action_out(x)
                action_log_probs.append(action_logit.log_probs(act))
                if active_masks is not None:
                    if len(action_logit.entropy().shape) == len(active_masks.shape):
                        dist_entropy.append((action_logit.entropy() * active_masks).sum()/active_masks.sum()) 
                    else:
                        dist_entropy.append((action_logit.entropy() * active_masks.squeeze(-1)).sum()/active_masks.sum())
                else:
                    dist_entropy.append(action_logit.entropy().mean())
                
            action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)
            dist_entropy = dist_entropy[0] * 0.0025 + dist_entropy[1] * 0.01 

        elif self.multidiscrete_action:
            action = torch.transpose(action, 0, 1)
            action_log_probs = []
            dist_entropy = []
            for action_out, act in zip(self.action_outs, action):
                action_logit = action_out(x)
                action_log_probs.append(action_logit.log_probs(act))
                if active_masks is not None:
                    dist_entropy.append((action_logit.entropy()*active_masks.squeeze(-1)).sum()/active_masks.sum())
                else:
                    dist_entropy.append(action_logit.entropy().mean())

            action_log_probs = torch.cat(action_log_probs, -1) # ! could be wrong
            dist_entropy = torch.tensor(dist_entropy).mean()

        elif self.continuous_action:
            action_logits = self.action_out(x)
            action_log_probs = action_logits.log_probs(action)
            act_entropy = action_logits.entropy()
            # import pdb;pdb.set_trace()
            if active_masks is not None:
                dist_entropy = (act_entropy*active_masks).sum()/active_masks.sum()
            else:
                dist_entropy = act_entropy.mean()

        else:
            action_logits = self.action_out(x, available_actions)
            action_log_probs = action_logits.log_probs(action)
            if active_masks is not None:
                dist_entropy = (action_logits.entropy()*active_masks.squeeze(-1)).sum()/active_masks.sum()
            else:
                dist_entropy = action_logits.entropy().mean()
        if not get_probs:
            return action_log_probs, dist_entropy
        else:
            return action_log_probs, dist_entropy, action_logits

class RNNLayer(nn.Module):
    def __init__(self, inputs_dim, outputs_dim, recurrent_N, use_orthogonal,rnn_type='gru'):
        super(RNNLayer, self).__init__()
        self._recurrent_N = recurrent_N
        self._use_orthogonal = use_orthogonal
        self.rnn_type = rnn_type
        if rnn_type == 'gru':
            self.rnn = nn.GRU(inputs_dim, outputs_dim, num_layers=self._recurrent_N)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(inputs_dim, outputs_dim, num_layers=self._recurrent_N)
        else:
            raise NotImplementedError(f'RNN type {rnn_type} has not been implemented.')

        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                if self._use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
        self.norm = nn.LayerNorm(outputs_dim)

    def rnn_forward(self, x, h):
        if self.rnn_type == 'lstm':
            h = torch.split(h, h.shape[-1] // 2, dim=-1)
            h = (h[0].contiguous(), h[1].contiguous())
        x_, h_ = self.rnn(x, h)
        if self.rnn_type == 'lstm':
            h_ = torch.cat(h_, -1)
        return x_, h_

    def forward(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.rnn_forward(x.unsqueeze(0), (hxs * masks.repeat(1, self._recurrent_N).unsqueeze(-1)).transpose(0, 1).contiguous())
            #x= self.gru(x.unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.transpose(0, 1)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0)
                         .any(dim=-1)
                         .nonzero()
                         .squeeze()
                         .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.transpose(0, 1)

            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]               
                temp = (hxs * masks[start_idx].view(1, -1, 1).repeat(self._recurrent_N, 1, 1)).contiguous()
                rnn_scores, hxs = self.rnn_forward(x[start_idx:end_idx], temp)
                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)

            # flatten
            x = x.reshape(T * N, -1)
            hxs = hxs.transpose(0, 1)

        x = self.norm(x)
        return x, hxs


class InputEncoder(nn.Module):
    def __init__(self):
        super(InputEncoder, self).__init__()
        fc_layer_num = 2
        fc_output_num = 64
        self.active_input_num = 87
        self.ball_owner_input_num = 57
        self.left_input_num = 88
        self.right_input_num = 88
        self.match_state_input_num = 9

        self.active_encoder = FcEncoder(fc_layer_num, self.active_input_num, fc_output_num)
        self.ball_owner_encoder = FcEncoder(fc_layer_num, self.ball_owner_input_num, fc_output_num)
        self.left_encoder = FcEncoder(fc_layer_num, self.left_input_num, fc_output_num)
        self.right_encoder = FcEncoder(fc_layer_num, self.right_input_num, fc_output_num)
        self.match_state_encoder = FcEncoder(fc_layer_num, self.match_state_input_num, self.match_state_input_num)

    def forward(self, x):
        active_vec = x[:, :self.active_input_num]
        ball_owner_vec = x[:, self.active_input_num : self.active_input_num + self.ball_owner_input_num]
        left_vec = x[:, self.active_input_num + self.ball_owner_input_num : self.active_input_num + self.ball_owner_input_num + self.left_input_num]
        right_vec = x[:, self.active_input_num + self.ball_owner_input_num + self.left_input_num : \
            self.active_input_num + self.ball_owner_input_num + self.left_input_num + self.right_input_num]
        match_state_vec = x[:, self.active_input_num + self.ball_owner_input_num + self.left_input_num + self.right_input_num:]

        active_output = self.active_encoder(active_vec)
        ball_owner_output = self.ball_owner_encoder(ball_owner_vec)
        left_output = self.left_encoder(left_vec)
        right_output = self.right_encoder(right_vec)
        match_state_output = self.match_state_encoder(match_state_vec)

        return torch.cat([
            active_output,
            ball_owner_output,
            left_output,
            right_output,
            match_state_output
        ], 1)

def get_fc(input_size, output_size):
    return nn.Sequential(nn.Linear(input_size, output_size), nn.ReLU(), nn.LayerNorm(output_size))

class ObsEncoder(nn.Module):
    def __init__(self, input_embedding_size, hidden_size, _recurrent_N, _use_orthogonal, rnn_type):
        super(ObsEncoder, self).__init__()
        self.input_encoder = InputEncoder()     # input先过一遍input encoder
        self.input_embedding = get_fc(input_embedding_size, hidden_size)  # 将encoder输出进行embedding
        self.rnn = RNNLayer(hidden_size, hidden_size, _recurrent_N, _use_orthogonal, rnn_type=rnn_type)  # embedding输出过一遍rnn
        self.after_rnn_mlp = get_fc(hidden_size, hidden_size)   # 过了rnn后再过该mlp

    def forward(self, obs, rnn_states, masks):
        actor_features = self.input_encoder(obs)
        actor_features = self.input_embedding(actor_features)
        output, rnn_states = self.rnn(actor_features, rnn_states, masks)
        return self.after_rnn_mlp(output), rnn_states


class PolicyNetwork(nn.Module):
    def __init__(self, device=torch.device("cpu")):
        super(PolicyNetwork, self).__init__()
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.device = device
        self.hidden_size = 256
        self._use_policy_active_masks = True
        recurrent_N = 1
        use_orthogonal = True
        rnn_type = 'lstm'
        gain = 0.01
        action_space = gym.spaces.Discrete(20)
        self.action_dim = 19
        input_embedding_size = 64 * 4 + 9
        self.active_id_size = 1
        self.id_max = 11

        self.obs_encoder = ObsEncoder(input_embedding_size, self.hidden_size, recurrent_N, use_orthogonal, rnn_type)

        self.predict_id = get_fc(self.hidden_size + self.action_dim, self.id_max)    # 其他信息(指除了active_id外的信息)过了rnn和一层mlp后，经过该层来预测id
        self.id_embedding = get_fc(self.id_max, self.id_max)     # active id作为输入，输出和其他信息的feature concat

        self.before_act_wrapper = FcEncoder(2, self.hidden_size + self.id_max, self.hidden_size)
        self.act = ACTLayer(action_space, self.hidden_size, use_orthogonal, gain)

        self.to(device)


    def forward(self, obs, rnn_states, masks=np.concatenate(np.ones((1, 1, 1), dtype=np.float32)), available_actions=None, deterministic=False):
        obs = check(obs).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)

        active_id = obs[:,:self.active_id_size].squeeze(1).long()
        id_onehot = torch.eye(self.id_max)[active_id].to(self.device)
        obs = obs[:,self.active_id_size:]
        
        obs_output, rnn_states = self.obs_encoder(obs, rnn_states, masks)
        id_output = self.id_embedding(id_onehot)
        output = torch.cat([id_output, obs_output], 1)
        
        output = self.before_act_wrapper(output)

        actions, action_log_probs = self.act(output, available_actions, deterministic)
        return actions, rnn_states

    def eval_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        obs = check(obs).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)

        id_groundtruth = obs[:,:self.active_id_size].squeeze(1).long()
        id_onehot = torch.eye(self.id_max)[id_groundtruth].to(self.device)
        obs = obs[:,self.active_id_size:]

        obs_output, rnn_states = self.obs_encoder(obs, rnn_states, masks)
        id_output = self.id_embedding(id_onehot)

        action_onehot = torch.eye(self.action_dim)[action.squeeze(1).long()].to(self.device)

        id_prediction = self.predict_id(torch.cat([obs_output, action_onehot], 1))
        output = torch.cat([id_output, obs_output], 1)

        output = self.before_act_wrapper(output)
        action_log_probs, dist_entropy = self.act.evaluate_actions(output, action, available_actions,
                                                                   active_masks=active_masks if self._use_policy_active_masks else None)
        values = None
        return action_log_probs, dist_entropy, values, id_prediction, id_groundtruth

