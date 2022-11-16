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

from light_malib import rollout
from light_malib.agent.policy_data.policy_data_manager import PolicyDataManager
from light_malib.utils.desc.task_desc import RolloutEvalDesc
from light_malib.utils.distributed import get_actor
import ray
import numpy as np

from light_malib.utils.logger import Logger
from .elo import EloManager


class EvaluationManager:
    def __init__(self, cfg, agent_manager, policy_data_manager):
        self.cfg = cfg
        self.agents = agent_manager.agents
        self.policy_data_manager = policy_data_manager
        self.rollout_manager = get_actor("EvaluationManager", "RolloutManager")
        self.elo_manager = EloManager(K=16)

    def eval(self):
        # generate tasks from payoff matrix
        rollout_eval_desc = self.generate_rollout_tasks()

        # call rollout_eval remotely
        eval_results = ray.get(
            self.rollout_manager.rollout_eval.remote(rollout_eval_desc)
        )

        for match, value in eval_results.items():
            score_dict = {}
            team_0 = match[0]
            team_1 = match[1]
            aid_0, pid_0 = team_0
            aid_1, pid_1 = team_1
            if pid_0 != pid_1:
                score_dict[pid_0] = value[aid_0]["score"]
                score_dict[pid_1] = value[aid_1]["score"]

                self.elo_manager.record_new_match_result(score_dict, 1)

        print(f"Elo = {self.elo_manager._elo_table.items()}")

        # update policy data
        self.policy_data_manager.update_policy_data(
            eval_results, elo=list(self.elo_manager._elo_table.items())
        )

    def _ordered(self, arr):
        for i in range(len(arr) - 1):
            if arr[i] > arr[i + 1]:
                return False
        return True

    def generate_rollout_tasks(self):
        payoff_matrix = self.policy_data_manager.get_matrix_data("payoff")
        indices = np.nonzero(
            payoff_matrix == self.policy_data_manager.cfg.fields.payoff.missing_value
        )

        policy_combs = []
        for index_comb in zip(*indices):
            if not self.agents.share_policies or self._ordered(index_comb):
                assert len(index_comb) == len(self.agents)
                policy_comb = {
                    agent_id: {agent.policy_ids[index_comb[i]]: 1.0}
                    for i, (agent_id, agent) in enumerate(self.agents.items())
                }
                policy_combs.append(policy_comb)

        Logger.warning(
            "Evaluation rollouts (num: {}) for {} policy combinations: {}".format(
                self.cfg.num_eval_rollouts, len(policy_combs), policy_combs
            )
        )
        rollout_eval_desc = RolloutEvalDesc(
            policy_combs, self.cfg.num_eval_rollouts, self.agents.share_policies
        )
        return rollout_eval_desc
