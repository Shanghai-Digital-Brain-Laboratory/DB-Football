# MIT License

# Copyright (c) 2022 DigitalBrain, Yan Song and He jiang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
    "--config", type=str, default="light_malib/expr/gr_football/expr_10_vs_10_psro.yaml"
)
parser.add_argument(
    "--model_0",
    type=str,
    default="light_malib/trained_models/gr_football/11_vs_11/offensive_passer",
)
parser.add_argument(
    "--model_1",
    type=str,
    default="light_malib/trained_models/gr_football/11_vs_11/built_in",
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
policy_1 = MAPPO.load(model_path_1, env_agent_id="agent_1")

env = GRFootballEnv(0, None, cfg.rollout_manager.worker.envs[0])
rollout_desc = RolloutDesc("agent_0", None, None, None, None, None)

total_run = args.total_run
total_win = 0
offset = np.random.randint(0, 2)
for idx in range(total_run):
    if (offset + idx) % 2 == 0:
        agent = "agent_0"
        behavior_policies = {
            "agent_0": (policy_id_0, policy_0),
            "agent_1": (policy_id_1, policy_1),
        }
        Logger.info("run {}/{}: model_0 vs model_1".format(idx + 1, total_run))
    else:
        agent = "agent_1"
        behavior_policies = {
            "agent_0": (policy_id_1, policy_1),
            "agent_1": (policy_id_0, policy_0),
        }
        Logger.info("run {}/{}: model_1 vs model_0".format(idx + 1, total_run))
    rollout_results = rollout_func(
        eval=True,
        rollout_worker=None,
        rollout_desc=rollout_desc,
        env=env,
        behavior_policies=behavior_policies,
        data_server=None,
        rollout_length=3001,
        render=False,
    )
    Logger.info("stats of model_0 is {}".format(rollout_results["stats"][agent]))
    total_win += rollout_results["stats"][agent]["win"]
Logger.warning("win rate of model_0 is {}".format(total_win / total_run))
