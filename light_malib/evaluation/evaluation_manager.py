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

from light_malib import rollout
from light_malib.agent.policy_data.policy_data_manager import PolicyDataManager
from light_malib.utils.desc.task_desc import RolloutEvalDesc
from light_malib.utils.distributed import get_actor
import ray
import numpy as np

from light_malib.utils.logger import Logger
from .elo import EloManager
from .melo import MEloManager

from open_spiel.python.egt import alpharank, utils as alpharank_utils
import nashpy as nash


class EvaluationManager:
    def __init__(self, cfg, agent_manager, policy_data_manager):
        self.cfg = cfg
        self.agents = agent_manager.agents
        self.policy_data_manager = policy_data_manager
        self.rollout_manager = get_actor("EvaluationManager", "RolloutManager")
        self.elo_manager = EloManager(K=16)
        self.melo_manager = MEloManager(K=16)

    def eval(self):
        # generate tasks from payoff matrix
        rollout_eval_desc = self.generate_rollout_tasks()

        # call rollout_eval remotely
        eval_results = ray.get(
            self.rollout_manager.rollout_eval.remote(rollout_eval_desc)
        )
        # breakpoint()
        for match, value in eval_results.items():
            # breakpoint()
            score_dict = {}
            win_dict = {}
            team_0 = match[0]
            team_1 = match[1]
            aid_0, pid_0 = team_0
            aid_1, pid_1 = team_1
            if pid_0 != pid_1:
                score_dict[pid_0] = value[aid_0]["score"]
                score_dict[pid_1] = value[aid_1]["score"]
                self.elo_manager.record_new_match_result(score_dict, 1)

                # win_dict[pid_0] = value[aid_0]['win']
                # win_dict[pid_1] = value[aid_1]['win']
                # self.melo_manager.record_new_match_result(win_dict, 1)


        # print(f"Elo = {self.elo_manager._elo_table.items()}")

        # update policy data
        self.policy_data_manager.update_policy_data(
            eval_results, elo=list(self.elo_manager._elo_table.items())
        )

        if self.cfg.eval_only:
            payoff_matrix = self.policy_data_manager.get_matrix_data("payoff")
            # win_matrix = self.policy_data_manager.get_matrix_data('win')
            training_agent_id = self.policy_data_manager.agents.training_agent_ids[0]
            pid = self.policy_data_manager.agents[training_agent_id].policy_ids

            _alpharank_ret = self.alpha_rank([payoff_matrix])
            alpharank_ret = dict(zip(pid, _alpharank_ret))

            nash_avg = self.nash_average(payoff_matrix)
            nash_avg_ret = dict(zip(pid, nash_avg[0]))

            sorted_elo = sorted(self.elo_manager._elo_table.items(), key = lambda x:x[1], reverse=True)
            print(f"Sorted Elo = {sorted_elo}")
            sorted_alpharank = sorted(alpharank_ret.items(), key = lambda x:x[1], reverse=True)
            print(f"Sorted AlphaRank = {sorted_alpharank}")

            sorted_nash = sorted(nash_avg_ret.items(), key = lambda x:x[1], reverse=True)
            print(f"Soted NashAvg = {sorted_nash}")


    def nash_average(self, payoff):
        def compute_nash(_payoff):
            game = nash.Game(_payoff)
            freqs = list(game.fictitious_play(iterations=100000))[-1]
            eqs = tuple(map(lambda x: x / np.sum(x), freqs))
            return eqs
        eqs = compute_nash(payoff)
        return eqs


    def alpha_rank(self, payoffs_seq):
        def remove_epsilon_negative_probs(probs, epsilon=1e-9):
            """Removes negative probabilities that occur due to precision errors."""
            if len(probs[probs < 0]) > 0:  # pylint: disable=g-explicit-length-test
                # Ensures these negative probabilities aren't large in magnitude, as that is
                # unexpected and likely not due to numerical precision issues
                print("Probabilities received were: {}".format(probs[probs < 0]))
                assert np.alltrue(
                    np.min(probs[probs < 0]) > -1.0 * epsilon
                ), "Negative Probabilities received were: {}".format(probs[probs < 0])

                probs[probs < 0] = 0
                probs = probs / np.sum(probs)
            return probs

        def get_alpharank_marginals(payoff_tables, pi):
            """Returns marginal strategy rankings for each player given joint rankings pi.
            Args:
              payoff_tables: List of meta-game payoff tables for a K-player game, where
                each table has dim [n_strategies_player_1 x ... x n_strategies_player_K].
                These payoff tables may be asymmetric.
              pi: The vector of joint rankings as computed by alpharank. Each element i
                corresponds to a unique integer ID representing a given strategy profile,
                with profile_to_id mappings provided by
                alpharank_utils.get_id_from_strat_profile().
            Returns:
              pi_marginals: List of np.arrays of player-wise marginal strategy masses,
                where the k-th player's np.array has shape [n_strategies_player_k].
            """
            num_populations = len(payoff_tables)

            if num_populations == 1:
                return pi
            else:
                num_strats_per_population = (
                    alpharank_utils.get_num_strats_per_population(
                        payoff_tables, payoffs_are_hpt_format=False
                    )
                )
                num_profiles = alpharank_utils.get_num_profiles(
                    num_strats_per_population
                )
                pi_marginals = [np.zeros(n) for n in num_strats_per_population]
                for i_strat in range(num_profiles):
                    strat_profile = alpharank_utils.get_strat_profile_from_id(
                        num_strats_per_population, i_strat
                    )
                    for i_player in range(num_populations):
                        pi_marginals[i_player][strat_profile[i_player]] += pi[i_strat]
                return pi_marginals

        joint_distr = alpharank.sweep_pi_vs_epsilon(payoffs_seq)
        joint_distr = remove_epsilon_negative_probs(joint_distr)
        marginals = get_alpharank_marginals(payoffs_seq, joint_distr)

        return marginals




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
