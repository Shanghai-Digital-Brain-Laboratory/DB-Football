import torch

from torch.nn import functional as F

from light_malib.utils.typing import Dict, Any
from light_malib.algorithm.common.loss_func import LossFunc
from light_malib.utils.episode import EpisodeKey
from light_malib.utils.logger import Logger
def soft_update(target, source, tau):
    """Perform DDPG soft update (move target params toward source based on weight factor tau).

    Reference:
        https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L11

    :param torch.nn.Module target: Net to copy parameters to
    :param torch.nn.Module source: Net whose parameters to copy
    :param float tau: Range form 0 to 1, weight factor for update
    """

    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class QMIXLoss(LossFunc):
    def __init__(self):
        super(QMIXLoss, self).__init__()
        self._cast_to_tensor = None
        self._mixer = None
        self._mixer_target = None
        self._params = {"gamma": 0.9995, "lr": 1e-3, "tau": 0.1,
                        "custom_caster": None, "device": "cpu", "target_update_freq": 5}

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

    def update_target(self):
        # for _, p in self.policy.items():
            # assert isinstance(p, DQN), type(p)
            # p.soft_update(self._params["tau"])
        # self.policy.soft_update(self._params['tau'])
        if isinstance(self.policy.critic, list):
            for i in range(len(self.policy.critic)):
                soft_update(self.policy.target_critic[i], self.policy.critic[i],
                            self._params['tau'])
        else:
            soft_update(self.policy.target_critic, self.policy.critic,
                        self._params['tau'])


        with torch.no_grad():
            soft_update(self.mixer_target, self.mixer, self._params["tau"])

    def reset(self, policy, configs):

        self._params.update(configs)

        if policy is not self.policy:
            self._policy = policy
            self._mixer = self.mixer.to(policy.device)
            self._mixer_target = self.mixer_target.to(policy.device)

            self.setup_optimizers()

        self.step_ctr=0

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
        if not isinstance(self.policy.critic, list):
            self.optimizers.add_param_group({"params": self.policy.critic.parameters()})
        else:
            for i in range(len(self.policy.critic)):
                self.optimizers.add_param_group({"params": self.policy.critic[i].parameters()})


    def step(self) -> Any:
        pass

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
            critic_rnn_state
        ) = (
            sample[EpisodeKey.GLOBAL_STATE],
            sample[EpisodeKey.CUR_OBS],
            sample[EpisodeKey.ACTION_MASK],
            sample[EpisodeKey.ACTION].long(),
            sample[EpisodeKey.REWARD],
            sample[EpisodeKey.DONE],
            sample[EpisodeKey.CRITIC_RNN_STATE]         #[batch_size, traj_length, 1, num_agents, hidden_state]
        )

        actions = actions.unsqueeze(-1)
            #[batch_size, traj_length, num_agents, feat_dim]
        bz, traj_length, num_agents, _ = observations.shape
        # Calculate estimated Q-values
        mac_out = []
        target_mac_out = []
        if not policy.custom_config.local_q_config.use_rnn_layer:
            if not isinstance(policy.critic, list):
                obs_mac = observations.reshape(bz*traj_length*num_agents, -1)
                mac_out, _ = policy.critic(obs_mac, torch.ones(1,1,1).to(obs_mac.device))
                mac_out = mac_out.reshape(bz, traj_length, num_agents, -1)
                target_mac_out, _ = policy.target_critic(obs_mac, torch.ones(1,1,1).to(obs_mac.device))
                target_mac_out = target_mac_out.reshape(bz, traj_length, num_agents, -1)

            else:
                qt = []
                qt_target = []
                for agent_idx in range(num_agents):
                    # obs_a_pre = observations[:,:-1,agent_idx, ...]   #[bz, traj_length, feat_dim]

                    _agent_q, _ = policy.critic[agent_idx](observations[:,:,agent_idx,...], torch.ones(1,1,1).to(observations.device))
                    qt.append(_agent_q)

                    # obs_a_post = observations[:,1:,agent_idx, ...]
                    _agent_q_target, _ = policy.target_critic[agent_idx](observations[:,:,agent_idx,...], torch.ones(1,1,1).to(observations.device))
                    qt_target.append(_agent_q_target)
                mac_out = torch.stack(qt, dim=-2)
                target_mac_out = torch.stack(qt_target, dim=-2)
        else:
            if not isinstance(policy.critic, list):
                obs_mac = observations.reshape(bz*traj_length*num_agents, -1).unsqueeze(0)  #[1, bz*traj_length*num_agents, feat_dim]
                h_mac = critic_rnn_state.reshape(bz*traj_length*num_agents, -1).unsqueeze(0)
                q, _ = policy.critic(obs_mac, h_mac)
                q = q[0].reshape(bz, traj_length, num_agents, -1)

                target_q, _ = policy.target_critic(obs_mac, h_mac)
                target_q = target_q[0].reshape(bz, traj_length, num_agents, -1)

                mac_out, target_mac_out = q, target_q
            else:
                raise NotImplementedError
                # qt, qt_target = [], []
                # for agent_idx in range(num_agents):
                #     obs_mac_a = observations[:,:,agent_idx,...].reshape(1,bz*traj_length,-1)
                #     h_mac_a =

        # Pick the Q-Values for the actions taken by each agent, [bz, traj_length-1, num_agents]
        chosen_action_qvals = torch.gather(mac_out[:, :-1, ...], dim=-1, index=actions[:, :-1, ...]).squeeze(-1)

        # Calculate the Q-Values necessary for the target       #[bz, traj_length-1, num_agents, _]
        # target_mac_out = torch.stack(target_mac_out[1:], dim=1)
        target_mac_out = target_mac_out[:, 1:, ...]
        target_mac_out[action_masks[:, 1:, ...] == 0] = -99999

        # Max over target Q-values, if double_q
        if policy.custom_config.double_q:
            cur_max_actions = mac_out[:, 1:].max(dim=-1, keepdim=True)[1]
            target_max_qvals = torch.gather(target_mac_out, -1, cur_max_actions).squeeze(-1)
        else:
            target_max_qvals = target_mac_out.max(dim=-1)[0]

        # Mix
        chosen_action_qvals = self.mixer(chosen_action_qvals, state[:, :-1, ...])
        target_max_qvals = self.mixer_target(target_max_qvals, state[:, 1:, ...])

        # Calculate 1-step Q-Learning targets
        targets = rewards[:, :-1, :, 0].sum(-1) + policy.custom_config.gamma * (
                    1 - dones[:, :-1, 0, 0]) * target_max_qvals

        # TD-error
        td_error = (chosen_action_qvals - targets.detach())

        # Normal L2 loss, take mean over actual data
        loss = (td_error ** 2).sum()

        grad_dict = {}
        # Optimise
        self.optimizers.zero_grad()
        loss.backward()
        if not isinstance(self._policy.critic, list):
            torch.nn.utils.clip_grad_norm_(
                self._policy.critic.parameters(), self._params['max_grad_norm']
            )

            for i, j in self._policy.critic.named_parameters():
                grad_dict[f"grad/{i}"] = j.grad.mean().detach().cpu().numpy()
        else:
            for idx, c in enumerate(self._policy.critic):
                torch.nn.utils.clip_grad_norm_(
                    c.parameters(), self._params['max_grad_norm']
                )
                for i, j in c.named_parameters():
                    grad_dict[f'grad/{idx}th critic/{i}'] = j.grad.mean().detach().cpu().numpy()

        torch.nn.utils.clip_grad_norm_(self.mixer.parameters(), self._params['max_grad_norm'])
        for i, j in self.mixer.named_parameters():
            grad_dict[f"grad/mixer/{i}"] = j.grad.mean().detach().cpu().numpy()

        self.optimizers.step()

        if self.step_ctr % policy.custom_config.target_update_freq == 0:
            self.update_target()

        ret = {
            "mixer_loss": loss.detach().cpu().numpy(),
            "value": chosen_action_qvals.mean().detach().cpu().numpy(),
            "target_value": targets.mean().detach().cpu().numpy(),
        }
        ret.update(grad_dict)

        return ret


def soft_update(target, source, tau):
    """Perform DDPG soft update (move target params toward source based on weight factor tau).

    Reference:
        https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L11

    :param torch.nn.Module target: Net to copy parameters to
    :param torch.nn.Module source: Net whose parameters to copy
    :param float tau: Range form 0 to 1, weight factor for update
    """

    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

