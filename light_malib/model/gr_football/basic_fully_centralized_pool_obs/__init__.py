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

from light_malib.algorithm.common import actor
from light_malib.algorithm.common import critic
from light_malib.envs.gr_football.encoders.encoder_basic import FeatureEncoder
import torch
import torch.nn as nn

class Critic(critic.Critic):
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
    
class Actor(actor.Actor):
    def __init__(self, model_config, observation_space, action_space, custom_config, initialization):
        super().__init__(model_config, observation_space, action_space, custom_config, initialization)
        
        self.num_players=custom_config["num_agents"]
        
        self.out=nn.Linear(self.feat_dim*2,self.act_dim)
        
    def forward(self, obs, actor_rnn_states, rnn_masks, action_masks, explore, actions):
                
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
        
        # #batch,logits
        logits=self.out(feats)
        
        # repeat
        # values=values.unsqueeze(dim=1).repeat(1,self.num_players,1)
        
        logits=logits.reshape(-1,logits.shape[-1])
        
        illegal_action_mask = 1-action_masks
        logits = logits - 1e10 * illegal_action_mask

        dist = torch.distributions.Categorical(logits=logits)
        if actions is None:
            actions = dist.sample() if explore else dist.probs.argmax(dim=-1)
            dist_entropy = None
        else:
            dist_entropy = dist.entropy()
        action_log_probs = dist.log_prob(actions) # num_action
        
        return actions,  actor_rnn_states, action_log_probs, dist_entropy