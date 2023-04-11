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

"""
put this file in the directory of gfootball /gfootball/env/players/

and run :

play_with_ai.sh

"""

from light_malib.algorithm.common.misc import hard_update
import os
import gym
import pickle
import numpy as np
import torch
from light_malib.algorithm.mappo.policy import MAPPO
from gfootball.env import football_action_set
from gfootball.env import player_base
import time
from light_malib.utils.episode import EpisodeKey
from light_malib.envs.gr_football.state import State


class Player(player_base.PlayerBase):
    """An agent loaded from torch model."""

    def __init__(self, player_config, env_config):
        player_base.PlayerBase.__init__(self, player_config)
        self.device="cpu"

        self._action_set = (
            env_config["action_set"] if "action_set" in env_config else "default"
        )
        self._policy = load_model(player_config["checkpoint"], device=self.device)
        self._feature_encoder = self._policy.feature_encoder
        self.n_pass = 0
        self.time_stamp = time.time()
        self.builtin_action_count = 0
        self.all_action_count = 0
        self.state = None

        # self._stats_calculator = stats_basic.StatsCaculator()
        # self._stats_calculator.reset()


    def take_action(self, observations):
        batch_obs = []
        batch_action_masks = []

        all_states = [State() for i in range(len(observations))]
        for o, s  in zip(observations, all_states):
            o['sticky_actions'] = o['sticky_actions'][:10]
            s.update_obs(o)

        for i, observation in enumerate(observations):
            # if observation["steps_left"] == 3001:
            #     self.state = State()
            # self.state.update_obs(observation)
            observation = np.array(self._feature_encoder.encode([all_states[i]]))
            action_masks = observation[..., :19]

            batch_obs.append(observation)
            batch_action_masks.append(action_masks)

        batch_obs = np.concatenate(batch_obs)
        batch_action_masks = np.concatenate(batch_action_masks)

        # print(f'batch obs shape = {batch_obs.shape}, batch action mask = {batch_action_masks.shape}, initial state shape = {self._policy.get_initial_state(1).keys()}')

        policy_input={
            EpisodeKey.CUR_OBS: batch_obs,
            EpisodeKey.ACTION_MASK: batch_action_masks,
            EpisodeKey.DONE: np.array([[False]*10]),
            EpisodeKey.ACTOR_RNN_STATE: np.repeat(self._policy.get_initial_state(1)['ACTOR_RNN_STATE'], 10, 0),
            EpisodeKey.CRITIC_RNN_STATE: np.repeat(self._policy.get_initial_state(1)['CRITIC_RNN_STATE'], 10, 0)
        }

        policy_input = to_tensor(policy_input, device=self.device)
        rets = self._policy.compute_action(**policy_input,explore=False)

        actions = rets[EpisodeKey.ACTION]

        # EPS = self._stats_calculator.effective_playing_space(observations[0])
        # lpW = self._stats_calculator.PlpW(observations[0])
        # TS = self._stats_calculator.team_separateness(observations[0])
        # wc = self._stats_calculator.team_centroid(observations[0])

        # if observations[0]['steps_left'] == 2:
            # plt.plot(self._stats_calculator.EPS_list, label = 'EPS')
            # plt.plot(self._stats_calculator.oppo_EPS_list, label='Bot EPS')
            # plt.axhline(y=0.195, color='r',label='initial EPS')
            # plt.axhline(y=1.776, color='grey', label='Max possible EPS')
            # plt.plot(self._stats_calculator.lpw_list, label = 'LpW')
            # plt.plot(self._stats_calculator.oppo_lpw_list, label='oppo LpW')
            # plt.axhline(y=self._stats_calculator.lpw_list[0], color='r')
            # plt.plot(self._stats_calculator.TS_list, label = 'TS')
            # plt.plot(self._stats_calculator.oppo_TS_list, label='Oppo TS')
            # plt.axhline(y=self._stats_calculator.TS_list[0], color='r')
            # plt.plot(self._stats_calculator.weighted_cen_list, label='weighted centroid x')
            # plt.plot(self._stats_calculator.oppo_weighted_cen_list, label='oppo weighted centroid x')

            # plt.legend()
            # plt.show()

        return actions


def load_model(load_path,device="cpu"):
    print('load path = ', load_path)
    with open(os.path.join(load_path, "desc.pkl"), "rb") as f:
        desc_pkl = pickle.load(f)

    res = MAPPO(
        desc_pkl["registered_name"],
        desc_pkl["observation_space"],
        desc_pkl["action_space"],
        desc_pkl["model_config"],
        desc_pkl["custom_config"],
        env_agent_id = 'team_1'
    )
    actor = torch.load(os.path.join(load_path, "actor.pt"), device)
    hard_update(res._actor, actor)
    return res


def to_tensor(data,device="cpu"):
    if isinstance(data,list):
        ret=[]
        for d in data:
            ret.append(to_tensor(d,device))
        return ret
    elif isinstance(data,dict):
        ret={}
        for k,v in data.items():
            ret[k]=to_tensor(v,device)
        return ret
    elif isinstance(data,np.ndarray):
        return torch.from_numpy(data).to(device)
    else:
        return data.to(device)



def concate_observation_from_raw(obs):
    obs_cat = np.hstack(
        [np.array(obs[k], dtype=np.float32).flatten() for k in sorted(obs)]
    )
    return obs_cat
