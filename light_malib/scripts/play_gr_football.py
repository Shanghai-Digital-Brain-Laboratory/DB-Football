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

from light_malib.rollout.rollout_func import rollout_func
from light_malib.utils.desc.task_desc import RolloutDesc
from light_malib.utils.episode import EpisodeKey
from light_malib.envs.gr_football.env import GRFootballEnv
from light_malib.algorithm.mappo.policy import MAPPO
from light_malib.utils.cfg import load_cfg
from light_malib.utils.logger import Logger
import numpy as np
import pickle as pkl
import argparse

parser = argparse.ArgumentParser(
    description="play google research football competition"
)
parser.add_argument(
    "--config", type=str, default="/home/yansong/Desktop/football_new/DB-Football/light_malib/expr/gr_football/expr_4_vs_4_psro.yaml"
)
parser.add_argument(
    "--model_0",
    type=str,
    default="/home/yansong/Desktop/football_new/DB-Football/light_malib/trained_models/gr_football/5_vs_5/GKBug_v3",
)
# parser.add_argument(
#     "--model_1",
#     type=str,
#     default="/home/yansong/Desktop/football_new/DB-Football/light_malib/trained_models/gr_football/5_vs_5/GKBug_v2",
# )
parser.add_argument(
    "--model_1",
    type=str,
    default="/home/yansong/Desktop/football_new/DB-Football/light_malib/trained_models/gr_football/5_vs_5/built_in",
)
parser.add_argument("--render", default=False, action="store_true")
parser.add_argument("--total_run", default=1, type=int)
args = parser.parse_args()

config_path = args.config
model_path_0 = args.model_0
model_path_1 = args.model_1

cfg = load_cfg(config_path)
cfg["rollout_manager"]["worker"]["envs"][0]["scenario_config"]["render"] = args.render

policy_id_0 = "policy_0"
policy_id_1 = "policy_1"
policy_0 = MAPPO.load(model_path_0, env_agent_id="agent_0")
# path = '/home/yansong/Downloads/5v5'


policy_1 = MAPPO.load(model_path_1, env_agent_id="agent_1")

env = GRFootballEnv(0, None, cfg.rollout_manager.worker.envs[0])
rollout_desc = RolloutDesc("agent_0", None, None, None, None, None)

total_run = args.total_run
total_win = 0
offset = np.random.randint(0, 2)
for idx in range(total_run):
    # if (offset + idx) % 2 == 0:
    agent = "agent_0"
    behavior_policies = {
        "agent_0": (policy_id_0, policy_0),
        "agent_1": (policy_id_1, policy_1),
    }
    Logger.info("run {}/{}: model_0 vs model_1".format(idx + 1, total_run))
    # else:
    #     agent = "agent_1"
    #     behavior_policies = {
    #         "agent_0": (policy_id_1, policy_1),
    #         "agent_1": (policy_id_0, policy_0),
    #     }
    #     Logger.info("run {}/{}: model_1 vs model_0".format(idx + 1, total_run))
    rollout_results = rollout_func(
        eval=True,
        rollout_worker=None,
        rollout_desc=rollout_desc,
        env=env,
        behavior_policies=behavior_policies,
        data_server=None,
        rollout_length=3001,
        render=True,
        episode_mode='traj'
    )
    Logger.info("stats of model_0 is {}".format(rollout_results["stats"][agent]))
    total_win += rollout_results["stats"][agent]["win"]
Logger.warning("win rate of model_0 is {}".format(total_win / total_run))
