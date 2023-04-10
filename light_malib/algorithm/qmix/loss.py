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
        self._params = {"gamma": 0.99, "lr": 5e-4, "tau": 0.01,
                        "custom_caster": None, "device": "cpu"}

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
        self.optimizers.add_param_group({"params": self.policy.critic.parameters()})

    def step(self) -> Any:
        pass

    def loss_compute(self, sample) -> Dict[str, Any]:
        self.step_ctr += 1
        self.loss = []
        policy = self._policy

        (
            state,
            next_state,
            observations,
            action_masks,
            actions,
            rewards,
            dones,
            next_observations,
            next_action_masks
        ) = (
            sample[EpisodeKey.GLOBAL_STATE],
            sample[EpisodeKey.NEXT_GLOBAL_STATE],
            sample[EpisodeKey.CUR_OBS],
            sample[EpisodeKey.ACTION_MASK],
            sample[EpisodeKey.ACTION].long(),
            sample[EpisodeKey.REWARD],
            sample[EpisodeKey.DONE],
            sample[EpisodeKey.NEXT_OBS],
            sample[EpisodeKey.NEXT_ACTION_MASK]
        )
            #[batch_size, num_agents, feat_dim]

        # ================= handle for each agent ====================================
        q_vals, next_max_q_vals = [], []
        for agent_idx in range(observations.shape[1]):
            _obs = observations[:,agent_idx, ...]
            _next_obs = next_observations[:, agent_idx, ...]
            _act = actions[:, agent_idx, ...]
            _next_action_mask = next_action_masks[:, agent_idx, ...]
            _q, _ = policy.critic(_obs, torch.ones(1,1,1).to(_obs.device))
            q = _q.gather(-1, _act.unsqueeze(1)).squeeze()
            # q = policy.critic(_obs, torch.ones(1,1,1).to(_obs.device)).gather(-1, _act.unsqueeze(1)).squeeze()
            q_vals.append(q)

            next_q, _ = policy.target_critic(_next_obs, torch.ones(1,1,1).to(_next_obs.device))
            next_q[_next_action_mask==0]=-9999999
            next_max_q = next_q.max(1)[0]
            next_max_q_vals.append(next_max_q.detach())


        q_vals = torch.stack(q_vals, dim=-1)
        next_max_q_vals = torch.stack(next_max_q_vals, dim=-1)

        q_tot = self.mixer(q_vals, state)

        next_max_q_tot = self.mixer_target(next_max_q_vals, next_state)

        targets = (
            rewards.sum(1) + self._params["gamma"] * (1.0 - dones[:,0,:]) * next_max_q_tot.detach()
        )           #all agent share the same rewards
        loss = F.smooth_l1_loss(q_tot, targets)
        # self.loss.append(loss)
        self.optimizers.zero_grad()
        loss.backward()
        if self._params['use_max_grad_norm']:
            torch.nn.utils.clip_grad_norm_(
                self._policy.critic.parameters(), self._params['max_grad_norm']
            )
            torch.nn.utils.clip_grad_norm_(
                self.mixer.parameters(), self._params['max_grad_norm']
            )

        for n, p in policy.critic.named_parameters():
            if p.grad is None:
                Logger.error(f'critic {n} has no grad')
        for n, p in self.mixer.named_parameters():
            if p.grad is None:
                Logger.error(f'mixer {n} has no grad')



        self.optimizers.step()
        self.update_target()

        return {
            "mixer_loss": loss.detach().cpu().numpy(),
            "value": q_tot.mean().detach().cpu().numpy(),
            "target_value": targets.mean().detach().cpu().numpy(),
        }


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

