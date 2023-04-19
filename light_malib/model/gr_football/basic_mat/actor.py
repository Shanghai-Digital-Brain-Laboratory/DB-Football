import torch.nn as nn
from gym.spaces import Discrete,Box
from .ma_transformer import Decoder
from .utils.transformer_act import discrete_autoregreesive_act, continuous_autoregreesive_act, discrete_parallel_act, continuous_parallel_act

class Actor(nn.Module):
    def __init__(
        self,
        model_config,
        action_space,
        custom_config,
        initialization,
        backbone
    ):
        super().__init__()
        
        self.action_space=action_space
        if isinstance(action_space,Discrete):
            self.action_dim=action_space.n
            action_type="Discrete"
        elif isinstance(action_space,Box):
            assert len(action_space.shape)==0
            self.action_dim=action_space.shape[0]
            action_type="Continuous"
        else:
            raise NotImplementedError
        
        # TODO(jh): remove. legacy.
        self.rnn_layer_num=1
        self.rnn_state_size=1
        
        self.dec_actor=model_config["dec_actor"]
        self.share_actor=model_config["share_actor"]
        
        self.num_agents=backbone.num_agents
        
        self.decoder=Decoder(
            obs_dim=backbone.obs_dim,
            action_dim=self.action_dim,
            n_block=backbone.num_blocks,
            n_embd=backbone.embed_dim,
            n_head=backbone.num_heads,
            n_agent=backbone.num_agents,
            action_type=action_type,
            dec_actor=self.dec_actor,
            share_actor=self.share_actor
        )
    
    def forward(
            self,
            observations, 
            actor_rnn_states,
            rnn_masks, 
            action_masks, 
            explore=True,
            actions=None
        ):
        
        obs_rep=observations["obs_rep"]
        obs=observations["obs"]
        
        deterministic= not explore
        assert len(obs)%self.num_agents==0
        batch_size=len(obs)//self.num_agents
        
        obs_rep=obs_rep.reshape(batch_size,self.num_agents,-1)
        obs=obs.reshape(batch_size,self.num_agents,-1)
        if actions is not None:
            actions=actions.reshape(batch_size,self.num_agents,-1)
        if action_masks is not None:
            action_masks=action_masks.reshape(batch_size,self.num_agents,-1)
        
        if actions is None:
            # inference: sample actions
            if isinstance(self.action_space,Discrete):
                actions, action_log_probs = discrete_autoregreesive_act(self.decoder, obs_rep, obs, batch_size,
                                                                            self.num_agents, self.action_dim,
                                                                            action_masks, deterministic)
            else:
                actions, action_log_probs = continuous_autoregreesive_act(self.decoder, obs_rep, obs, batch_size,
                                                                                self.num_agents, self.action_dim,
                                                                                deterministic)
            entropy=None  
        else:
            # training: evaluate actions log probs
            if isinstance(self.action_space,Discrete):
                action_log_probs, entropy = discrete_parallel_act(self.decoder, obs_rep, obs, actions, batch_size,
                                                            self.num_agents, self.action_dim, action_masks)
            else:
                action_log_probs, entropy = continuous_parallel_act(self.decoder, obs_rep, obs, actions, batch_size,
                                                            self.num_agents, self.action_dim)
        
        # TODO(jh): actions' shape should be 2-dim as well.
        actions=actions.reshape(batch_size*self.num_agents)
        action_log_probs=action_log_probs.reshape(batch_size*self.num_agents,-1)
        if entropy is not None:
            entropy=entropy.reshape(batch_size*self.num_agents,-1)
        
        return actions, actor_rnn_states, action_log_probs,  entropy