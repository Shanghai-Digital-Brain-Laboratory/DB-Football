import torch

from torch.nn import functional as F

from light_malib.utils.typing import Dict, Any
from light_malib.algorithm.common.loss_func import LossFunc
from light_malib.utils.episode import EpisodeKey
from light_malib.utils.logger import Logger

class CDS_QMixLoss(LossFunc):
    def __init__(self):
        super(CDS_QMixLoss, self).__init__()
        self._cast_to_tensor = None
        self._mixer = None
        self._mixer_target = None
        self._params = {"gamma": 0.9995, "lr": 0.0005, "tau": 0.1,
                        "custom_caster": None, "device": "cpu", "target_update_freq": 5}


        self.args = {'GPU': 'cuda:0',
                'action_selector': 'epsilon_greedy',
                'agent': 'rnn',
                'agent_output_type': 'q',
                'alpha': 0.3,
                'batch_size': 8,
                'batch_size_run': 1,
                'beta': 0.05,
                'beta1': 0.5,
                'beta2': 1.0,
                'buffer_cpu_only': True,
                'buffer_size': 5000,
                'burn_in_period': 32,
                'checkpoint_path': '',
                'critic_lr': 0.0005,
                'double_q': True,
                'env': 'academy_3_vs_1_with_keeper',
                'env_args': {   'dense_reward': False,
                                'dump_freq': 1000,
                                'env_name': 'academy_3_vs_1_with_keeper',
                                'logdir': 'football_dumps',
                                'n_agents': 3,
                                'number_of_right_players_agent_controls': 0,
                                'obs_dim': 26,
                                'render': False,
                                'representation': 'simple115',
                                'rewards': 'scoring',
                                'seed': 689404616,
                                'stacked': False,
                                'time_limit': 150,
                                'time_step': 0,
                                'write_full_episode_dumps': False,
                                'write_goal_dumps': False,
                                'write_video': True},
                'epsilon_anneal_time': 50000,
                'epsilon_finish': 0.05,
                'epsilon_start': 1.0,
                'evaluate': False,
                'gamma': 0.99,
                'grad_norm_clip': 10,
                'hypernet_embed': 64,
                'hypernet_layers': 2,
                'ifaddobs': True,
                'ifaver': True,
                'ifon_sample': False,
                'is_batch_rl': False,
                'is_from_start': True,
                'is_save_buffer': False,
                'label': 'default_label',
                'learner': 'CDS_QMIX',
                'learner_log_interval': 10000,
                'load_buffer_id': 0,
                'load_step': 0,
                'local_results_path': 'results',
                'log_interval': 10000,
                'lr': 0.0005,
                'mac': 'basic_mac',
                'mixer': 'qmix',
                'mixing_embed_dim': 32,
                'name': 'cds_qmix_prior',
                'num_circle': 1,
                'obs_agent_id': False,
                'obs_last_action': True,
                'on_policy_batch': 16,
                'optim_alpha': 0.99,
                'optim_eps': 1e-05,
                'predict_epoch': 25,
                'predict_net_dim': 128,
                'repeat_id': 1,
                'rnn_hidden_dim': 64,
                'runner': 'episode',
                'runner_log_interval': 10000,
                'save_buffer_id': 0,
                'save_buffer_interval': 1000,
                'save_buffer_size': 10000,
                'save_model': True,
                'save_model_interval': 500000,
                'save_replay': False,
                'seed': 689404616,
                't_max': 4050000,
                'target_update_interval': 200,
                'test_greedy': True,
                'test_interval': 10000,
                'test_nepisode': 32,
                'use_cuda': True,
                'use_tensorboard': True}

    @property
    def mixer(self):
        return self._mixer

    @property
    def mixer_target(self):
        return self._mixer_target

    def set_mixer(self, mixer):
        self._mixer = mixer

    def set_mixer_target(self, mixer_target):
        self._mixer_target = mixer_target

    def reset(self, policy, configs):
        self._params.update(configs)
        if policy is not self.policy:
            self._policy = policy
            self._mixer = self.mixer.to(policy.device)
            self._mixer_target = self.mixer_target.to(policy.device)
            self.setup_optimizers()
        self.step_ctr=0

    def step(self):
        pass

    def setup_optimizers(self, *args, **kwargs):
        assert self.mixer is not None, "Mixer has not been set yet!"
        if self.optimizers is None:
            self.optimizers = self.optim_cls(
                self.mixer.parameters(), lr=self._params["lr"]
            )
        else:
            self.optimizers.param_groups = []
            self.optimizers.add_param_group({"params": self.mixer.parameters()})

        # for policy in self.policy.values():
        # if not isinstance(self.policy.critic, list):
        #     self.optimizers.add_param_group({"params": self.policy.critic.parameters()})
        # else:
        #     for i in range(len(self.policy.critic)):
        #         self.optimizers.add_param_group({"params": self.policy.critic[i].parameters()})
        self.optimizers.add_param_group({"params": self.policy.mac.parameters()})


    def loss_compute(self, sample):
        self.step_ctr += 1
        self.loss = []
        policy = self._policy

        (
            state,
            observations,
            action_masks,
            actions,
            rewards,
            dones,
        ) = (
            sample[EpisodeKey.GLOBAL_STATE],
            sample[EpisodeKey.CUR_OBS],
            sample[EpisodeKey.ACTION_MASK],
            sample[EpisodeKey.ACTION].long(),
            sample[EpisodeKey.REWARD],
            sample[EpisodeKey.DONE],
        )

        actions = actions.unsqueeze(-1)
        assert len(state.shape) == 4            #[batch_size, traj_length, num_agent,_]

        batch_size, traj_length, num_agent,_ = observations.shape

        policy.mac.init_hidden(batch_size)
        init_hidden = policy.mac.hidden_states.clone().detach()
        init_hidden = init_hidden.reshape(-1, init_hidden.shape[-1]).to(state.device)


        input_here = observations.permute(0,2,1,3)
        mac_out, hidden_store, local_qs = policy.mac.agent.forward(input_here.clone().detach(),
                                                                   init_hidden.clone().detach())
        hidden_store = hidden_store.reshape(-1, input_here.shape[1], hidden_store.shape[-2],
                                            hidden_store.shape[-1]).permute(0,2,1,3)
        Logger.warning(f"RNNs are implemented wrong,  need to fix this")

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = torch.gather(
            mac_out[:, :-1], dim=3, index=actions[:,:-1]).squeeze(3)  # Remove the last dim, [batch_size, traj_length-1, num_agent]

        x_mac_out = mac_out.clone().detach()
        x_mac_out[action_masks == 0] = -9999999
        max_action_qvals, max_action_index = x_mac_out[:, :-1].max(dim=3)

        max_action_index = max_action_index.detach().unsqueeze(3)
        is_max_action = (max_action_index == actions[:,:-1]).int().float()

        # Calculate the Q-Values necessary for the target
        policy.target_mac.init_hidden(batch_size)
        initial_hidden_target = policy.target_mac.hidden_states.clone().detach()
        initial_hidden_target = initial_hidden_target.reshape(
            -1, initial_hidden_target.shape[-1])
        target_mac_out, _, _ = policy.target_mac.agent.forward(
            input_here.clone().detach(), initial_hidden_target.clone().detach())
        target_mac_out = target_mac_out[:, 1:]

        # Max over target Q-Values
        if policy.custom_config['double_q']:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[action_masks == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = torch.gather(
                target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]


        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(
                chosen_action_qvals, state[:, :-1])
            target_max_qvals = self.mixer_target(
                target_max_qvals, state[:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards[:,:-1,...].sum(-2) + self._params['gamma'] * (1 - dones[:,:-1,:,0]) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = torch.ones_like(td_error) #action_masks[:,:-1].expand_as(td_error)            #filled us a reserved key for masking episode

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()
        update_prior = (masked_td_error ** 2).squeeze().sum(dim=-1,
                                                            keepdim=True) / mask.squeeze().sum(dim=-1, keepdim=True)

        norm_loss = F.l1_loss(local_qs, target=torch.zeros_like(
            local_qs), reduction='none')[:, :-1]
        mask_expand = mask.unsqueeze(-1).expand_as(norm_loss)
        norm_loss = (norm_loss * mask_expand).sum() / mask_expand.sum()
        loss += 0.1 * norm_loss

        masked_hit_prob = torch.mean(is_max_action, dim=2) * mask
        hit_prob = masked_hit_prob.sum() / mask.sum()

        # Optimise
        self.optimizers.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.mac.parameters(), 1)
        mixer_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.mixer.parameters(), 1
        )
        self.optimizers.step()

        if self.step_ctr%self._params['target_update_freq']==0:
            self.policy.target_mac.load_state(self.policy.mac)
            self.mixer_target.load_state_dict(self.mixer.state_dict())

        return {'loss': loss.detach().cpu().numpy()}



