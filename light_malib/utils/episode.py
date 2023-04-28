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


class EpisodeKey:
    """Unlimited buffer"""

    CUR_OBS = "observation"
    NEXT_OBS = "next_observation"
    ACTION = "action"
    ACTION_MASK = "action_mask"
    NEXT_ACTION_MASK = "next_action_mask"
    REWARD = "reward"
    DONE = "done"
    # XXX(ziyu): Change to 'logits' for numerical issues.
    ACTION_DIST = "action_logits"
    ACTION_PROB = "action_prob"
    ACTION_PROBS = "action_probs"
    ACTION_LOG_PROB = "action_log_prob"
    ACTION_ENTROPY = "action_entropy"
    # XXX(ming): seems useless
    INFO = "infos"

    ACTIVE_MASK = "active_mask"

    # optional
    STATE_VALUE = "state_value_estimation"
    STATE_ACTION_VALUE = "state_action_value_estimation"
    CUR_STATE = "state"  # current global state
    NEXT_STATE = "next_state"  # next global state
    LAST_REWARD = "last_reward"
    RETURN = "return"
    GLOBAL_STATE = "global_state"
    NEXT_GLOBAL_STATE = 'next_global_state'

    # post process
    ACC_REWARD = "accumulate_reward"
    ADVANTAGE = "advantage"
    STATE_VALUE_TARGET = "state_value_target"

    # model states
    RNN_STATE = "rnn_state"
    ACTOR_RNN_STATE = "ACTOR_RNN_STATE"
    CRITIC_RNN_STATE = "CRITIC_RNN_STATE"
    NEXT_CRITIC_RNN_STATE = "NEXT_CRITIC_RNN_STATE"

    # expert
    EXPERT_OBS = "expert_obs"
    EXPERT_ACTION = "expert_action"
