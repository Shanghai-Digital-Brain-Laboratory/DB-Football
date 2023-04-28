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


def rename_fields(data, fields, new_fields):
    assert len(fields)==len(new_fields)
    for agent_id, agent_data in data.items():
        for field, new_field in zip(fields,new_fields):
            if field in agent_data:
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
        try:
            episode_data[field] = np.stack(data_list)
        except Exception as e:
            import traceback
            Logger.error(traceback.format_exc())
            first_shape=data_list[0].shape
            for idx,data in enumerate(data_list):
                if data.shape!=first_shape:
                    Logger.error("field {}: first_shape: {}, mismatched_shape: {}, mismatched_idx: {}".format(field,first_shape,data.shape,idx))
                    break
            raise e
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

def pull_policies(rollout_worker,policy_ids):
    rollout_worker.pull_policies(policy_ids)
    behavior_policies = rollout_worker.get_policies(policy_ids)
    return behavior_policies

def env_reset(env, behavior_policies, custom_reset_config):
    global_timer.record("env_step_start")
    env_rets = env.reset(custom_reset_config)
    global_timer.time("env_step_start", "env_step_end", "env_step")

    init_rnn_states = {
        agent_id: behavior_policies[agent_id][1].get_initial_state(
            batch_size=env.num_players[agent_id]
        )
        for agent_id in env.agent_ids
    }

    step_data = update_fields(env_rets, init_rnn_states)
    return step_data

def submit_traj(data_server,step_data_list,last_step_data,rollout_desc,s_idx=None,e_idx=None,credit_reassign_cfg=None,assist_info=None):
    bootstrap_data = select_fields(
        last_step_data,
        [
            EpisodeKey.NEXT_OBS,
            EpisodeKey.DONE,
            EpisodeKey.CRITIC_RNN_STATE,
            EpisodeKey.NEXT_STATE,
        ],
    )
    bootstrap_data = rename_fields(bootstrap_data, [EpisodeKey.NEXT_OBS,EpisodeKey.NEXT_STATE], [EpisodeKey.CUR_OBS,EpisodeKey.CUR_OBS])
    bootstrap_data = bootstrap_data[rollout_desc.agent_id]
    
    _episode = stack_step_data(
        step_data_list[s_idx:e_idx],
        # TODO CUR_STATE is not supported now
        bootstrap_data,
    )

    if credit_reassign_cfg is not None and assist_info is not None:
        episode = credit_reassign(
            _episode,
            assist_info,
            credit_reassign_cfg,
            s_idx,
            e_idx,
        )
    else:
        episode = _episode

    # submit data:
    if hasattr(data_server.save, 'remote'):
        data_server.save.remote(
            default_table_name(
                rollout_desc.agent_id,
                rollout_desc.policy_id,
                rollout_desc.share_policies,
            ),
            [episode],
        )
    else:
        data_server.save(
            default_table_name(
                rollout_desc.agent_id,
                rollout_desc.policy_id,
                rollout_desc.share_policies,
            ),
            [episode],
        )

def submit_batches(data_server,episode, rollout_desc,credit_reassign_cfg=None,assist_info=None):
    transitions = []
    for step in range(len(episode) - 1):
        transition = {
            EpisodeKey.CUR_OBS: episode[step][EpisodeKey.CUR_OBS],  # [np.newaxis, ...],
            EpisodeKey.ACTION_MASK: episode[step][EpisodeKey.ACTION_MASK],  # [np.newaxis, ...],
            EpisodeKey.ACTION: episode[step][EpisodeKey.ACTION],  # [np.newaxis, ...],
            EpisodeKey.REWARD: episode[step][EpisodeKey.REWARD],  # [np.newaxis, ...],
            EpisodeKey.DONE: episode[step][EpisodeKey.DONE],  # [np.newaxis, ...],
            EpisodeKey.NEXT_OBS: episode[step + 1][EpisodeKey.CUR_OBS],  # [np.newaxis, ...],
            EpisodeKey.NEXT_ACTION_MASK: episode[step + 1][EpisodeKey.ACTION_MASK],  # [np.newaxis, ...]
            EpisodeKey.CRITIC_RNN_STATE: episode[step][EpisodeKey.CRITIC_RNN_STATE],
            EpisodeKey.NEXT_CRITIC_RNN_STATE: episode[step + 1][EpisodeKey.CRITIC_RNN_STATE],
            EpisodeKey.GLOBAL_STATE: episode[step][EpisodeKey.GLOBAL_STATE],
            EpisodeKey.NEXT_GLOBAL_STATE: episode[step + 1][EpisodeKey.GLOBAL_STATE]
        }
        transitions.append(transition)
    if hasattr(data_server.save, 'remote'):
        data_server.save.remote(
            default_table_name(
                rollout_desc.agent_id,
                rollout_desc.policy_id,
                rollout_desc.share_policies,
            ),
            transitions
        )
    else:
        data_server.save(
            default_table_name(
                rollout_desc.agent_id,
                rollout_desc.policy_id,
                rollout_desc.share_policies,
            ),
            transitions
        )




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

    episode_mode = kwargs.get('episode_mode','traj')
    record_value = kwargs.get("record_value", False)
    if record_value:
        value_list = []
        pos_list = []

    policy_ids = OrderedDict()
    feature_encoders = OrderedDict()
    for agent_id, (policy_id, policy) in behavior_policies.items():
        feature_encoders[agent_id] = policy.feature_encoder
        policy_ids[agent_id] = policy_id
        policy.eval()

    custom_reset_config = {
        "feature_encoders": feature_encoders,
        "main_agent_id": rollout_desc.agent_id,
        "rollout_length": rollout_length,
    }

    step_data = env_reset(env,behavior_policies,custom_reset_config)

    step = 0
    step_data_list = []
    results = []
    # collect until rollout_length
    while step <= rollout_length:
        # prepare policy input
        policy_inputs = rename_fields(step_data, [EpisodeKey.NEXT_OBS,EpisodeKey.NEXT_STATE], [EpisodeKey.CUR_OBS,EpisodeKey.CUR_OBS])
        policy_outputs = {}
        global_timer.record("inference_start")
        for agent_id, (policy_id, policy) in behavior_policies.items():
            policy_outputs[agent_id] = policy.compute_action(
                inference=True, 
                explore=not eval,
                to_numpy=True,
                step = kwargs.get('rollout_epoch', 0),
                **policy_inputs[agent_id]
            )
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
                [EpisodeKey.ACTION, EpisodeKey.ACTION_LOG_PROB, EpisodeKey.STATE_VALUE],
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
        
        ##### submit samples to server #####
        # used for the full game
        if not eval:
            if episode_mode == 'traj':
                # used for on-policy algorithms
                assert data_server is not None            
                if sample_length > 0 and step % sample_length == 0:
                    assist_info = env.get_AssistInfo()

                    submit_ctr = step // sample_length
                    submit_max_num = rollout_length // sample_length

                    s_idx = sample_length * (submit_ctr - 1)
                    e_idx = sample_length * submit_ctr

                    submit_traj(data_server,step_data_list,step_data,rollout_desc,s_idx,e_idx,
                                credit_reassign_cfg=kwargs.get("credit_reassign_cfg"),
                                assist_info=assist_info)   

                    if submit_ctr != submit_max_num:
                        # update model:
                        behavior_policies=pull_policies(rollout_worker,policy_ids)
                    
            # elif episode_mode == 'time-step':
            #     # used for off-policy algorithms
            #     episode = step_data_list
            #     transitions = []
            #     for step in range(len(episode)-1):
            #         transition = {
            #             EpisodeKey.CUR_OBS: episode[step][EpisodeKey.CUR_OBS][np.newaxis, ...],
            #             EpisodeKey.ACTION_MASK: episode[step][EpisodeKey.ACTION_MASK][np.newaxis, ...],
            #             EpisodeKey.ACTION: episode[step][EpisodeKey.ACTION][np.newaxis, ...],
            #             EpisodeKey.REWARD: episode[step][EpisodeKey.REWARD][np.newaxis, ...],
            #             EpisodeKey.DONE: episode[step][EpisodeKey.DONE][np.newaxis, ...],
            #             EpisodeKey.NEXT_OBS: episode[step + 1][EpisodeKey.CUR_OBS][np.newaxis, ...],
            #             EpisodeKey.NEXT_ACTION_MASK: episode[step + 1][EpisodeKey.ACTION_MASK][np.newaxis, ...]
            #         }
            #         transitions.append(transition)
            #     data_server.save.remote(
            #         default_table_name(
            #             rollout_desc.agent_id,
            #             rollout_desc.policy_id,
            #             rollout_desc.share_policies,
            #         ),
            #         transitions
            #     )
                    
        ##### check if  env ends #####
        if env.is_terminated():
            stats = env.get_episode_stats()
            if record_value:
                result = {
                    "main_agent_id": rollout_desc.agent_id,
                    "policy_ids": policy_ids,
                    "stats": stats,
                    "value": value_list,
                    "pos": pos_list,
                    "assist_info": assist_info,
                }
            else:
                result = {
                    "main_agent_id": rollout_desc.agent_id,
                    "policy_ids": policy_ids,
                    "stats": stats,
                }
            results.append(result)
            
            # reset env
            step_data = env_reset(env,behavior_policies,custom_reset_config)
    
    if not eval and sample_length <= 0:            #collect after rollout done
        # used for the academy
        if episode_mode == 'traj':
            submit_traj(data_server,step_data_list,step_data,rollout_desc)
        elif episode_mode == 'time-step':
            submit_batches(data_server, step_data_list, rollout_desc)



    results={"results":results}            
    return results

