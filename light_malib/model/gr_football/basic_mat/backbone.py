import torch
import torch.nn as nn
from gym.spaces import Box, Discrete
from .ma_transformer import Encoder

class Backbone(nn.Module):
    def __init__(
        self,
        model_config,
        global_observation_space,
        observation_space,
        action_space,
        custom_config,
        initialization,
    ):
        super().__init__()
        
        assert isinstance(global_observation_space, Box) and len(global_observation_space.shape)==1,global_observation_space
        assert isinstance(observation_space,Box) and len(observation_space.shape)==1,observation_space
        
        self.global_observation_space=global_observation_space
        self.observation_space=observation_space
        self.state_dim=global_observation_space.shape[0]
        self.obs_dim=observation_space.shape[0]
        
        self.num_agents=model_config["num_agents"]
        self.embed_dim=model_config["embed_dim"]
        self.num_blocks=model_config["num_blocks"]
        self.num_heads=model_config["num_heads"]
        self.encode_state=model_config["encode_state"]
        
        # TODO: only maintain encoder here
        self.encoder=Encoder(
            state_dim=self.state_dim,
            obs_dim=self.obs_dim,
            n_block=self.num_blocks,
            n_embd=self.embed_dim,
            n_head=self.num_heads,
            n_agent=self.num_agents,
            encode_state=self.encode_state
        )
    
    def forward(
            self,
            states,
            observations,
            critic_rnn_states,
            rnn_masks    
        ):
        '''
        backbone should return observations, but it could be a data structure containing anything.
        '''
        
        assert len(observations)%self.num_agents==0
        batch_size=len(observations)//self.num_agents
        
        states=states.reshape(batch_size,self.num_agents,-1)
        observations=observations.reshape(batch_size,self.num_agents,-1)
        
        values, obs_rep=self.encoder(states, observations)
        
        values=values.reshape(batch_size*self.num_agents,-1)
        obs_rep=obs_rep.reshape(batch_size*self.num_agents,-1)
        observations=observations.reshape(batch_size*self.num_agents,-1)
        
        return {"values": values, "obs_rep": obs_rep, "obs": observations}
        
        