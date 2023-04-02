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


from typing import OrderedDict
import numpy as np
from light_malib.utils.logger import Logger
from light_malib.utils.episode import EpisodeKey
from light_malib.envs.base_env import BaseEnv
from light_malib.utils.desc.task_desc import RolloutDesc
from light_malib.utils.timer import global_timer
from light_malib.utils.naming import default_table_name


def rename_field(data, field, new_field):
    for agent_id, agent_data in data.items():
        field_data = agent_data.pop(field)
        agent_data[new_field] = field_data
    return data


def select_fields(data, fields):
    rets = {
        agent_id: {field: agent_data[field] for field in fields if field in agent_data}
        for agent_id, agent_data in data.items()
    }
    return rets


def update_fields(data1, data2):
    def update_dict(dict1, dict2):
        d = {}
        d.update(dict1)
        d.update(dict2)
        return d

    rets = {
        agent_id: update_dict(data1[agent_id], data2[agent_id]) for agent_id in data1
    }
    return rets


def stack_step_data(step_data_list, bootstrap_data):
    episode_data = {}
    for field in step_data_list[0]:
        data_list = [step_data[field] for step_data in step_data_list]
        if field in bootstrap_data:
            data_list.append(bootstrap_data[field])
        episode_data[field] = np.stack(data_list)
    return episode_data


def credit_reassign(episode, info, reward_reassignment_cfg, s_idx, e_idx):
    _reward = episode["reward"]
    goal_info = info["goal_info"]
    assist_info = info["assist_info"]
    loseball_info = info["loseball_info"]
    halt_loseball_info = info["halt_loseball_info"]
    gainball_info = info["gainball_info"]
    tag_dict = {
        "goal": goal_info,
        "assist": assist_info,
        "loseball": loseball_info,
        "gainball": gainball_info,
    }
    for tag, i in tag_dict.items():
        for value in i:
            t_idx = value["t"] - s_idx
            if 0 <= t_idx < len(_reward):
                # if s_idx<=value['t']<=e_idx:
                #     t_idx = value['t']-s_idx
                player_idx = value["player"] - 1
                _reward[t_idx, player_idx, 0] += reward_reassignment_cfg[tag]

    episode["reward"] = _reward
    return episode


def rollout_func(
    eval: bool,
    rollout_worker,
    rollout_desc: RolloutDesc,
    env: BaseEnv,
    behavior_policies,
    data_server,
    rollout_length,
    **kwargs
):
    """
    TODO(jh): modify document

    Rollout in simultaneous mode, support environment vectorization.

    :param VectorEnv env: The environment instance.
    :param Dict[Agent,AgentInterface] agent_interfaces: The dict of agent interfaces for interacting with environment.
    :param ray.ObjectRef dataset_server: The offline dataset server handler, buffering data if it is not None.
    :return: A dict of rollout information.
    """

    sample_length = kwargs.get("sample_length", rollout_length)
    render = kwargs.get("render", False)
    if render:
        env.render()

    record_value = kwargs.get("record_value", False)
    if record_value:
        value_list = []
        pos_list = []

    policy_ids = OrderedDict()
    feature_encoders = OrderedDict()
    for agent_id, (policy_id, policy) in behavior_policies.items():
        feature_encoders[agent_id] = policy.feature_encoder
        policy_ids[agent_id] = policy_id

    custom_reset_config = {
        "feature_encoders": feature_encoders,
        "main_agent_id": rollout_desc.agent_id,
        "rollout_length": rollout_length,
    }

    global_timer.record("env_step_start")
    env_rets = env.reset(custom_reset_config)
    global_timer.time("env_step_start", "env_step_end", "env_step")

    init_rnn_states = {
        agent_id: behavior_policies[agent_id][1].get_initial_state(
            batch_size=env.num_players[agent_id]
        )
        for agent_id in env.agent_ids
    }

    # TODO(jh): support multi-dimensional batched data based on dict & list using sth like NamedIndex.
    step_data = update_fields(env_rets, init_rnn_states)

    step = 0
    step_data_list = []
    while (
        not env.is_terminated()
    ):  # XXX(yan): terminate only when step_length >= fragment_length
        # prepare policy input
        policy_inputs = rename_field(step_data, EpisodeKey.NEXT_OBS, EpisodeKey.CUR_OBS)
        policy_outputs = {}
        global_timer.record("inference_start")
        for agent_id, (policy_id, policy) in behavior_policies.items():
            policy_outputs[agent_id] = policy.compute_action(**policy_inputs[agent_id])
            if record_value and agent_id == "agent_0":
                value_list.append(policy_outputs[agent_id][EpisodeKey.STATE_VALUE])
                pos_list.append(policy_inputs[agent_id][EpisodeKey.CUR_OBS][:, 114:116])

        global_timer.time("inference_start", "inference_end", "inference")

        actions = select_fields(policy_outputs, [EpisodeKey.ACTION])

        global_timer.record("env_step_start")
        env_rets = env.step(actions)
        global_timer.time("env_step_start", "env_step_end", "env_step")

        # record data after env step
        step_data = update_fields(
            step_data, select_fields(env_rets, [EpisodeKey.REWARD, EpisodeKey.DONE])
        )
        step_data = update_fields(
            step_data,
            select_fields(
                policy_outputs,
                [EpisodeKey.ACTION, EpisodeKey.ACTION_DIST, EpisodeKey.STATE_VALUE],
            ),
        )

        # save data of trained agent for training
        step_data_list.append(step_data[rollout_desc.agent_id])

        # record data for next step
        step_data = update_fields(
            env_rets,
            select_fields(
                policy_outputs,
                [EpisodeKey.ACTOR_RNN_STATE, EpisodeKey.CRITIC_RNN_STATE],
            ),
        )

        step += 1
        if not eval:
            assert data_server is not None
            if sample_length == 0 and env_rets['agent_0']['done'][0][0]:            #collect after episode done

                bootstrap_data = select_fields(
                    step_data,
                    [
                        EpisodeKey.NEXT_OBS,
                        EpisodeKey.DONE,
                        EpisodeKey.CRITIC_RNN_STATE,
                        EpisodeKey.CUR_STATE,
                    ],
                )
                bootstrap_data = bootstrap_data[rollout_desc.agent_id]
                bootstrap_data[EpisodeKey.CUR_OBS] = bootstrap_data[EpisodeKey.NEXT_OBS]

                episode = stack_step_data(
                    step_data_list,
                    # TODO CUR_STATE is not supported now
                    bootstrap_data,
                )

                # submit data:

                data_server.save.remote(
                    default_table_name(
                        rollout_desc.agent_id,
                        rollout_desc.policy_id,
                        rollout_desc.share_policies,
                    ),
                    [episode],
                )



            if sample_length > 0 and step % sample_length == 0:

                assist_info = env.get_AssistInfo()

                submit_ctr = step // sample_length
                submit_max_num = rollout_length // sample_length

                s_idx = sample_length * (submit_ctr - 1)
                e_idx = sample_length * submit_ctr

                bootstrap_data = select_fields(
                    step_data,
                    [
                        EpisodeKey.NEXT_OBS,
                        EpisodeKey.DONE,
                        EpisodeKey.CRITIC_RNN_STATE,
                        EpisodeKey.CUR_STATE,
                    ],
                )
                bootstrap_data = bootstrap_data[rollout_desc.agent_id]
                bootstrap_data[EpisodeKey.CUR_OBS] = bootstrap_data[EpisodeKey.NEXT_OBS]

                _episode = stack_step_data(
                    step_data_list[s_idx:e_idx],
                    # TODO CUR_STATE is not supported now
                    bootstrap_data,
                )

                if kwargs.get("credit_reassign_cfg", None) is not None:
                    episode = credit_reassign(
                        _episode,
                        assist_info,
                        kwargs.get("credit_reassign_cfg"),
                        s_idx,
                        e_idx,
                    )
                else:
                    episode = _episode

                # submit data:
                data_server.save.remote(
                    default_table_name(
                        rollout_desc.agent_id,
                        rollout_desc.policy_id,
                        rollout_desc.share_policies,
                    ),
                    [episode],
                )

                if submit_ctr != submit_max_num:
                    # update model:
                    rollout_worker.pull_policies(policy_ids)
                    behavior_policies = rollout_worker.get_policies(policy_ids)

        elif record_value:
            if step % sample_length == 0:
                submit_ctr = step // sample_length
                submit_max_num = rollout_length // sample_length

                s_idx = sample_length * (submit_ctr - 1)
                e_idx = sample_length * submit_ctr

                bootstrap_data = select_fields(
                    step_data,
                    [
                        EpisodeKey.NEXT_OBS,
                        EpisodeKey.DONE,
                        EpisodeKey.CRITIC_RNN_STATE,
                        EpisodeKey.CUR_STATE,
                    ],
                )
                bootstrap_data = bootstrap_data[rollout_desc.agent_id]
                bootstrap_data[EpisodeKey.CUR_OBS] = bootstrap_data[EpisodeKey.NEXT_OBS]

                episode = stack_step_data(
                    step_data_list[s_idx:e_idx],
                    # TODO CUR_STATE is not supported now
                    bootstrap_data,
                )

                assist_info = env.get_AssistInfo()

    stats = env.get_episode_stats()

    if record_value:
        return {
            "main_agent_id": rollout_desc.agent_id,
            "policy_ids": policy_ids,
            "stats": stats,
            "value": value_list,
            "pos": pos_list,
            "episode": episode,
            "assist_info": assist_info,
        }
    else:
        return {
            "main_agent_id": rollout_desc.agent_id,
            "policy_ids": policy_ids,
            "stats": stats,
        }
