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

import sys
import numpy as np
from absl import logging

logging.set_verbosity(logging.INFO)

from gfootball.env import create_environment
from ..tools.tracer import MatchTracer

env = create_environment(
    env_name="11_vs_11_kaggle",
    number_of_left_players_agent_controls=1,
    number_of_right_players_agent_controls=0,
    representation="raw",
    render=True,  # set to True will save large-size RGB frame.
    write_full_episode_dumps=True,
    logdir="./temp",
    other_config_options={"sorted_observations": True},
)

tracer = MatchTracer()

# now, only need to provide those two parameters.
tracer.update_settings({"n_left_control": 1, "n_right_control": 0})

print("test_tracer")
obs = env.reset()
steps = 0
print(steps)
while True:
    try:
        actions = env.action_space.sample()
        if isinstance(actions, int):
            actions = [actions]
        tracer.update(obs, actions)
        obs, rew, done, info = env.step(actions)
        steps += 1
        if steps % 100 == 0:
            print(steps, rew)
        if done:
            tracer.update(obs)
            tracer.save("temp/random_play.trace")
            break
    except:
        tracer.save("temp/random_play.trace")
        break
