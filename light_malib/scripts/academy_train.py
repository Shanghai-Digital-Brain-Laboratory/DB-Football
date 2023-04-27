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

from light_malib.buffer.data_server import DataServer
from light_malib.utils.naming import default_table_name


parser = argparse.ArgumentParser(
    description="play google research football competition"
)
parser.add_argument(
    "--config", type=str, default="/home/yansong/Desktop/football_new/DB-Football/light_malib/expr/academy/qmix_3_vs_1_with_keeper.yaml"
)
parser.add_argument(
    "--model_0",
    type=str,
    default="/home/yansong/Desktop/football_new/DB-Football/logs/gr_football/academy_3_vs_1_with_keeper/2023-04-03-02-01-10/agent_0/agent_0-default-1/epoch_83000",
)
parser.add_argument(
    "--model_1",
    type=str,
    default="/home/yansong/Desktop/football_new/DB-Football/light_malib/trained_models/gr_football/11_vs_11/built_in",
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

# from light_malib.registry.registration import QMix
# policy_0 = QMix('QMix', None, None, cfg.populations[0].algorithm.model_config, cfg.populations[0].algorithm.custom_config)
# from light_malib.registry.registration import CDS_QMix
# policy_0 = CDS_QMix('CDS_QMix', None, None, cfg.populations[0].algorithm.model_config, cfg.populations[0].algorithm.custom_config)
from light_malib.registry.registration import QMix
policy_0 = QMix('QMix', None, None,cfg.populations[0].algorithm.model_config, cfg.populations[0].algorithm.custom_config)


# policy_0 = MAPPO.load(model_path_0, env_agent_id="agent_0")
policy_1 = MAPPO.load(model_path_1, env_agent_id="agent_1")

env = GRFootballEnv(0, None, cfg.rollout_manager.worker.envs[0])
rollout_desc = RolloutDesc("agent_0", None, None, None, None, None)

cfg.data_server.table_cfg.rate_limiter_cfg.min_size = 1
datasever = DataServer('dataserver_1', cfg.data_server)
table_name = default_table_name(
    rollout_desc.agent_id,
    rollout_desc.policy_id,
    rollout_desc.share_policies,
)
datasever.create_table(table_name)

from light_malib.registry.registration import QMixTrainer
# trainer = CDS_QMixTrainer('trainer_1')
trainer = QMixTrainer('trainer_1')

total_run = args.total_run
total_win = 0
offset = np.random.randint(0, 2)
for idx in range(5):
    env = GRFootballEnv(0, None, cfg.rollout_manager.worker.envs[0])
    rollout_desc = RolloutDesc("agent_0", None, None, None, None, None)

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
        eval=False,
        rollout_worker=None,
        rollout_desc=rollout_desc,
        env=env,
        behavior_policies=behavior_policies,
        data_server=datasever,
        rollout_length=cfg.rollout_manager.worker.rollout_length,
        sample_length=cfg.rollout_manager.worker.sample_length,
        render=False,
        rollout_epoch=100,
        episode_mode=cfg.rollout_manager.worker.episode_mode,
    )
    Logger.info("stats of model_0 is {}".format(rollout_results['results'][0]["stats"][agent]))
    # total_win += rollout_results["stats"][agent]["win"]
Logger.warning("win rate of model_0 is {}".format(total_win / total_run))

data_list = []
for _ in range(1):
    sample, _ = datasever.sample(table_name, batch_size = 2)
    data_list.append(sample)

def stack(samples):
    ret = {}
    for k, v in samples[0].items():
        # recursively stack
        if isinstance(v, dict):
            ret[k] = stack([sample[k] for sample in samples])
        elif isinstance(v, np.ndarray):
            ret[k] = np.stack([sample[k] for sample in samples])
        elif isinstance(v, list):
            ret[k] = [
                stack([sample[k][i] for sample in samples])
                for i in range(len(v))
            ]
        else:
            raise NotImplementedError
    return ret

#merge data
samples = []
for i in range(len(data_list[0])):
    sample = {}
    for data in data_list:
        sample.update(data[i])
    samples.append(sample)

stack_samples = stack(samples)


policy_0 = policy_0.to_device('cuda:0')
class trainer_cfg:
    policy = policy_0
    device = 'cpu'

trainer.reset(policy_0, cfg.training_manager.trainer)
trainer.optimize(stack_samples)

Logger.info(f"Training complete")



