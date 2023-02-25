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

from light_malib.utils.logger import Logger
from .utils.pretty_print import pformat_table
import numpy as np
from light_malib.utils.distributed import get_actor
import ray
import pickle
import os


def update_func(policy_data_manager, eval_results, **kwargs):
    assert policy_data_manager.agents.share_policies, "jh: assert symmetry"
    for policy_comb, agents_results in eval_results.items():
        agent_id_0, policy_id_0 = policy_comb[0]
        agent_id_1, policy_id_1 = policy_comb[1]
        results_0 = agents_results[agent_id_0]
        results_1 = agents_results[agent_id_1]

        idx_0 = policy_data_manager.agents[agent_id_0].policy_id2idx[policy_id_0]
        idx_1 = policy_data_manager.agents[agent_id_1].policy_id2idx[policy_id_1]

        if (
            policy_data_manager.data["payoff"][idx_0, idx_1]
            == policy_data_manager.cfg.fields.payoff.missing_value
        ):
            for key in ["payoff", "score", "win", "lose", "my_goal", "goal_diff"]:
                policy_data_manager.data[key][idx_0, idx_1] = 0
                policy_data_manager.data[key][idx_1, idx_0] = 0

        for key in ["score", "win", "lose", "my_goal", "goal_diff"]:
            policy_data_manager.data[key][idx_0, idx_1] += results_0[key] / 2
            policy_data_manager.data[key][idx_1, idx_0] += results_1[key] / 2
            if key == "score":
                policy_data_manager.data["payoff"][idx_0, idx_1] += results_0[key] - 0.5
                policy_data_manager.data["payoff"][idx_1, idx_0] += results_1[key] - 0.5

    # print data
    Logger.info(
        "policy_data: {}".format(
            policy_data_manager.format_matrices_data(
                ["payoff", "score", "win", "lose", "my_goal", "goal_diff"]
            )
        )
    )

    # pretty-print
    # support last_k. last_k=0 means showing all
    last_k = 10
    policy_ids_dict = {
        agent_id: agent.policy_ids[-last_k:]
        for agent_id, agent in policy_data_manager.agents.items()
    }
    policy_ids_0 = [
        policy_id[8:] if policy_id.startswith("agent_0") else policy_id
        for policy_id in policy_ids_dict["agent_0"]
    ]  # remove prefix "agent_0_"
    policy_ids_1 = [
        policy_id[8:] if policy_id.startswith("agent_0") else policy_id
        for policy_id in policy_ids_dict["agent_1"]
    ]

    payoff_matrix = policy_data_manager.get_matrix_data("payoff") * 100
    monitor = get_actor(policy_data_manager.id, "Monitor")
    training_agent_id = policy_data_manager.agents.training_agent_ids[0]
    pid = policy_data_manager.agents[training_agent_id].policy_ids
    ray.get(
        monitor.add_array.remote(
            "PSRO/Nash_Equilibrium/Payoff Table",
            payoff_matrix,
            pid,
            pid,
            payoff_matrix.shape[0],
            "bwr",
            show_text=False,
        )
    )
    dump_path = ray.get(monitor.get_expr_log_dir.remote())
    elo = kwargs.get("elo", None)
    if len(elo)>0 and "agent_0" in elo[-1][0]:
        ray.get(monitor.add_scalar.remote("PSRO/Elo", elo[-1][1], int(elo[-1][0][-1])))

    if elo is not None:
        dump_path = os.path.join(dump_path, "elo.pkl")
        with open(dump_path, "wb") as f:
            pickle.dump(elo, f)

    payoff_matrix = payoff_matrix[-last_k:, -last_k:]
    table = pformat_table(
        payoff_matrix, headers=policy_ids_1, row_indices=policy_ids_0, floatfmt="+3.0f"
    )
    Logger.info("payoff table:\n{}".format(table))

    worst_k = 10
    policy_ids = {
        agent_id: agent.policy_ids
        for agent_id, agent in policy_data_manager.agents.items()
    }["agent_0"]
    policy_ids = [
        policy_id[8:] if policy_id.startswith("agent_0") else policy_id
        for policy_id in policy_ids
    ]

    worst_indices = np.argsort(payoff_matrix[-1, :])[:worst_k]
    Logger.info(
        "{}'s top {} worst opponents are:\n{}".format(
            policy_ids[-1],
            worst_k,
            pformat_table(
                payoff_matrix[-1:, worst_indices].T,
                headers=["policy_id", "payoff"],
                row_indices=[policy_ids[idx] for idx in worst_indices],
                floatfmt="+6.2f",
            ),
        )
    )
