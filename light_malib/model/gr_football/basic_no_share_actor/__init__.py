from light_malib.algorithm.common import actor as actor_module
from light_malib.algorithm.common import critic as critic_module
from light_malib.envs.gr_football.encoders.encoder_basic import FeatureEncoder

import torch.nn as nn
import torch

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
  
Critic=critic_module.Critic

# class Critic(nn.Module):
#     def __init__(
#             self,
#             model_config,
#             observation_space,
#             action_space,
#             custom_config,
#             initialization         
#             ) -> None:
#         super().__init__()
        
#         self.critics=nn.ModuleList()
        
#         self.num_agents=custom_config["num_agents"]
#         for i in range(self.num_agents):
#             critic=critic_module.Critic(model_config,observation_space,action_space,custom_config,initialization)
#             self.critics.append(critic)
            
#         self.rnn_layer_num=critic.rnn_layer_num
#         self.rnn_state_size=critic.rnn_state_size
                        
#     def forward(self,observations, rnn_states, rnn_masks):
#         observations_list=fold(observations,self.num_agents)
#         rnn_states_list=fold(rnn_states,self.num_agents)
#         rnn_masks_list=fold(rnn_masks,self.num_agents)
        
#         _values=[]
#         _rnn_states=[]
            
#         for i in range(self.num_agents):
#             observations_i=observations_list[:,i]
#             rnn_states_i = rnn_states_list[:,i] if rnn_states_list is not None else None
#             rnn_masks_i = rnn_masks_list[:,i] if rnn_masks_list is not None else None
#             values_i, rnn_states_i = self.critics[i].forward(
#                 observations_i,
#                 rnn_states_i,
#                 rnn_masks_i)
            
#             _values.append(values_i)
#             _rnn_states.append(rnn_states_i)
            
#         values=torch.stack(_values,dim=1)
#         values=unfold(values,self.num_agents)
#         rnn_states=torch.stack(_rnn_states,dim=1)
#         rnn_states=unfold(rnn_states,self.num_agents)
        
#         return values, rnn_states