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

from light_malib.algorithm.common import actor as actor_module
from light_malib.algorithm.common import critic as critic_module
from light_malib.envs.gr_football.encoders.encoder_basic import FeatureEncoder
import torch
import torch.nn as nn

def fold(tensor,num_agents):
    if tensor is None:
        return None
    assert tensor.shape[0]%num_agents==0
    B=tensor.shape[0]//num_agents
    tensor=tensor.reshape(B,num_agents,*tensor.shape[1:])
    return tensor
        
def unfold(tensor,num_agents):
    if tensor is None:
        return None
    assert tensor.shape[1]==num_agents
    tensor=tensor.reshape(-1,*tensor.shape[2:])
    return tensor


class Actor(nn.Module):
    def __init__(
        self,
        model_config,
        observation_space,
        action_space,
        custom_config,
        initialization         
        ) -> None:
        super().__init__()
        
        self.actors=nn.ModuleList()
        
        self.num_agents=custom_config["num_agents"]
        for i in range(self.num_agents):
            actor=actor_module.Actor(model_config,observation_space,action_space,custom_config,initialization)
            self.actors.append(actor)
            
        self.rnn_layer_num=actor.rnn_layer_num
        self.rnn_state_size=actor.rnn_state_size
            
    def forward(self, observations, actor_rnn_states, rnn_masks, action_masks, explore, actions):
        
        observations_list=fold(observations,self.num_agents)
        actor_rnn_states_list=fold(actor_rnn_states,self.num_agents)
        rnn_masks_list=fold(rnn_masks,self.num_agents)
        action_masks_list=fold(action_masks,self.num_agents)
        actions_list=fold(actions,self.num_agents)

        _actions=[]
        _action_log_probs=[]
        if actor_rnn_states_list is not None:
            _actor_rnn_states=[]
        if actions_list is not None:
            _dist_entropy=[]
            
        for i in range(self.num_agents):
            observations_i=observations_list[:,i]
            actor_rnn_states_i = actor_rnn_states_list[:,i] if actor_rnn_states_list is not None else None
            rnn_masks_i = rnn_masks_list[:,i] if rnn_masks_list is not None else None
            action_masks_i = action_masks_list[:,i] if action_masks_list is not None else None
            actions_i = actions_list[:,i] if actions_list is not None else None
            actions_i, actor_rnn_states_i, action_log_probs_i, dist_entropy_i = self.actors[i].forward(
                observations_i,
                actor_rnn_states_i,
                rnn_masks_i,
                action_masks_i, 
                explore,
                actions_i)
            
            if actions_list is not None:
                _dist_entropy.append(dist_entropy_i)
            _actions.append(actions_i)
            _action_log_probs.append(action_log_probs_i)  
            if actor_rnn_states_list is not None:
                _actor_rnn_states.append(actor_rnn_states_i)  
        
        actions=torch.stack(_actions,dim=1)
        actions=unfold(actions,self.num_agents)
        action_log_probs=torch.stack(_action_log_probs,dim=1)
        action_log_probs=unfold(action_log_probs,self.num_agents)
        
        if actor_rnn_states_list is not None:
            actor_rnn_states=torch.stack(_actor_rnn_states,dim=1)
            actor_rnn_states=unfold(actor_rnn_states,self.num_agents)

        if actions_list is not None:
            dist_entropy=torch.stack(_dist_entropy,dim=1)
            dist_entropy=unfold(dist_entropy,self.num_agents)
        else:
            dist_entropy=None
        
        return actions, actor_rnn_states, action_log_probs, dist_entropy
  
class Critic(critic_module.Critic):
    def __init__(self, model_config, observation_space, action_space, custom_config, initialization):
        super().__init__(model_config, observation_space, action_space, custom_config, initialization)
        
        self.num_players=custom_config["num_agents"]
        
        self.out=nn.Linear(self.feat_dim*2,self.act_dim)
        
    def forward(self, obs, rnn_states, masks):
                
        assert len(obs.shape)==2
        
        assert obs.shape[0]%self.num_players==0
        batch_size=obs.shape[0]//self.num_players
        
        feats = self.base(obs)
        
        feats=feats.reshape(batch_size,self.num_players,feats.shape[-1])
        
        # merge: #batch, #feats
        global_feats=feats.mean(dim=1)
        
        # repeat
        global_feats=global_feats.unsqueeze(dim=1).repeat(1,self.num_players,1) 
        feats=torch.concat([global_feats,feats],dim=-1)
        
        # #batch,1
        values=self.out(feats)
        
        # repeat
        # values=values.unsqueeze(dim=1).repeat(1,self.num_players,1)
        
        values=values.reshape(-1,values.shape[-1])
        
        return values, rnn_states