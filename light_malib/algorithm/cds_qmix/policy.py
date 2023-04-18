import copy
import os
import pickle
import random
import gym
import torch
import numpy as np

from torch import nn
from light_malib.utils.logger import Logger
from light_malib.utils.typing import DataTransferType, Tuple, Any, Dict, EpisodeID, List
from light_malib.utils.episode import EpisodeKey

import wrapt
import tree
import importlib
from light_malib.utils.logger import Logger
from gym.spaces import Discrete
from ..utils import PopArt
from light_malib.registry import registry
from copy import deepcopy
from omegaconf import OmegaConf

from light_malib.model.gr_football.basic_academy import FeatureEncoder
from .mac import BasicMAC



@registry.registered(registry.POLICY)
class CDS_QMix:
    "https://github.com/lich14/CDS.git "
    def __init__(
            self,
            registered_name: str,
            observation_space: gym.spaces.Space,  # legacy
            action_space: gym.spaces.Space,  # legacy
            model_config: Dict[str, Any] = None,
            custom_config: Dict[str, Any] = None,
            **kwargs,
    ):
        super().__init__()

        self.registered_name = registered_name

        self.encoder = FeatureEncoder(num_players=6)
        self.observation_space = self.feature_encoder.observation_space
        self.obs_dim = self.observation_space.shape[0]
        self.action_space = Discrete(19)
        self.act_dim = 19 #action_space.n
        self.output_dim = sum(self.act_dim) if isinstance(self.act_dim, np.ndarray) else self.act_dim

        self.custom_config = custom_config
        self.model_config = model_config

        buffer_scheme = {'state': {'vshape': 26}, 'obs': {'vshape': 26, 'group': 'agents'},
                         'actions': {'vshape': (1,), 'group': 'agents', 'dtype': torch.int64},
                         'avail_actions': {'vshape': (19,), 'group': 'agents', 'dtype': torch.int32},
                         'reward': {'vshape': (1,)}, 'terminated': {'vshape': (1,), 'dtype': torch.uint8},
                         'actions_onehot': {'vshape': (19,), 'dtype': torch.float32, 'group': 'agents'},
                         'filled': {'vshape': (1,), 'dtype': torch.int64}}
        groups = {'agents': 3}
        args = {'runner': 'episode', 'mac': 'basic_mac', 'env':'academy_3_vs_1_with_keeper',
                'env_args':{'dense_reward': False, 'write_full_episode_dumps': False, 'write_goal_dumps': False,
                            'dump_freq': 1000, 'render': False, 'n_agents': 3, 'time_limit': 150, 'time_step': 0,
                            'obs_dim': 26, 'env_name': 'academy_3_vs_1_with_keeper', 'stacked': False, 'representation': 'simple115',
                            'rewards': 'scoring', 'logdir': 'football_dumps', 'write_video': True, 'number_of_right_players_agent_controls': 0,
                            'seed': 426769177},
                'batch_size_run':1, 'test_nepisode': 32, 'test_interval': 10000, 'test_greed':True, 'log_interval':10000,
                'runner_log_interval':10000, 'learner_log_interval':10000, 't_max':4050000, 'use_cuda':True, 'GPU':'cuda:0',
                'buffer_cpu_only':True, 'is_save_buffer':False, 'save_buffer_size':10000, 'save_buffer_interval':1000,
                'is_batch_rl':False, 'load_buffer_id':0, 'save_buffer_id':0, 'is_from_start':True, 'num_circle':1,
                'burn_in_period':32, 'use_tensorboard':True, 'save_model':True, 'save_model_interval':500000, 'checkpoint_path':'',
                'evaluate':False, 'load_step':0, 'save_replay':False, 'local_results_path':'results', 'gamma':0.99, 'batch_size':8,
                'buffer_size':5000, 'lr':0.0005, 'critic_lr':0.0005, 'optim_alpha':0.99, 'optim_eps':1e-05, 'grad_norm_clip':10,
                'agent':'rnn', 'rnn_hidden_dim':64, 'obs_agent_id':False, 'obs_last_action':True, 'repeat_id':1, 'label':'default_label',
                'action_selector':'epsilon_greedy', 'epsilon_start':1.0, 'epsilon_finish':0.05, 'epsilon_anneal_time':50000, 'target_update_interval':200,
                'agent_output_type':'q', 'learner':'CDS_QMIX', 'double_q':True, 'mixer':'qmix', 'mixing_embed_dim':32, 'hypernet_layers':2, 'hypernet_embed':64,
                'on_policy_batch':16, 'predict_epoch':25, 'predict_net_dim':128, 'beta1':0.5, 'beta2':1.0, 'beta':0.05, 'ifaver':True, 'ifon_sample':False,
                'ifaddobs':True, 'alpha':0.3, 'name':'cds_qmix_prior', 'seed':426769177, 'device':'cuda:0', 'unique_token':'academy_3_vs_1_with_keeper/cds_qmix/seed_426769177',
                'obs_shape':26, 'episode_limit':150, 'n_agents':3, 'n_actions':19, 'state_shape':26, 'unit_dim':26}
        self.args = OmegaConf.create(args)

        self.mac = BasicMAC(buffer_scheme, groups, self.args)
        # self.register_module(f"mac", self.mac)

        self.target_mac = copy.deepcopy(self.mac)
        # self.register_module(f"target_mac", self.target_mac)

        self.hidden_states = None


    @property
    def description(self):
        """Return a dict of basic attributes to identify policy.

        The essential elements of returned description:

        - registered_name: `self.registered_name`
        - observation_space: `self.observation_space`
        - action_space: `self.action_space`
        - model_config: `self.model_config`
        - custom_config: `self.custom_config`

        :return: A dictionary.
        """

        return {
            "registered_name": self.registered_name,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "model_config": self.model_config,
            "custom_config": self.custom_config,
        }


    @property
    def feature_encoder(self):  # legacy
        return self.encoder

    # def to_device(self, device):
    #     #self_copy = copy.deepcopy(self)
    #     self.mac.agent.to(device)
    #     self.target_mac.agent.to(device)
    #     self.device = device
    #     return self

    def to_device(self, device):

        if hasattr(self.mac, "hidden_states") and self.mac.hidden_states is not None:
            self.mac.hidden_states = self.mac.hidden_states.detach()
        self_copy = copy.deepcopy(self)


        self_copy.device = device
        self_copy.mac = self_copy.mac.to(device)
        self_copy.target_mac = self_copy.target_mac.to(device)

        if hasattr(self_copy.mac, "hidden_states") and self_copy.mac.hidden_states is not None:
            self_copy.mac.hidden_states = self_copy.mac.hidden_states.to(device)

        if hasattr(self_copy.target_mac, "hidden_states") and self_copy.target_mac.hidden_states is not None:
            self_copy.target_mac.hidden_states = self_copy.target_mac.hidden_states.to(device)

        return self_copy

    # def to_device(self, device):
    #     self_copy = copy.deepcopy(self)
    #     self_copy.to(device)
    #     self_copy.device = device
    #     return self_copy



    def get_initial_state(self, batch_size):

        self.mac.init_hidden(1)

        return {EpisodeKey.ACTOR_RNN_STATE: np.zeros((1,1,1)), EpisodeKey.CRITIC_RNN_STATE: np.zeros((1,1,1))}

    def compute_action(self, **kwargs):
        local_obs = kwargs[EpisodeKey.CUR_OBS]
        action_masks = kwargs[EpisodeKey.ACTION_MASK]
        rollout_step = kwargs['step']
        batch_size = 1  #local_obs.shape[0]

        agent_outs, new_hidden_state, _ = self.mac.agent(torch.FloatTensor(local_obs), self.mac.hidden_states)
        self.mac.hidden_states = new_hidden_state

        if self.mac.agent_output_type=='pi_logits':
            if getattr(self.mac.args, 'mask_before_softmax', True):
                reshaped_avail_actions = action_masks.reshape(batch_size*self.mac.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = torch.nn.functional.softmax(agent_outs, dim=-1)

            if kwargs['explore']:
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.mac.args, "mask_before_softmax", True):
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                agent_outs = ((1-self.mac.action_selector.epsilon)*agent_outs +
                              torch.ones_like(agent_outs)*self.mac.action_selector.epsilon/epsilon_action_num)

                if getattr(self.mac.args, "mask_before_softmax", True):
                    agent_outs[reshaped_avail_actions == 0] = 0.0


        agent_outs.view(batch_size, self.mac.n_agents, -1)
        chosen_actions = self.mac.action_selector.select_action(agent_outs, action_masks, rollout_step)

        return {EpisodeKey.ACTION: chosen_actions.detach().cpu().numpy(),
                EpisodeKey.ACTOR_RNN_STATE: torch.ones(1,1,1).numpy(),
                EpisodeKey.CRITIC_RNN_STATE: torch.ones(1,1,1).numpy()}





