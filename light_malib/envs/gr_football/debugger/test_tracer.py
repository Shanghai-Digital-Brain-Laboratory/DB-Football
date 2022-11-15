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
    render=True, # set to True will save large-size RGB frame.
    write_full_episode_dumps=True,
    logdir="./temp",
    other_config_options={
        "sorted_observations":True
    }
)

tracer=MatchTracer()

# now, only need to provide those two parameters.
tracer.update_settings(
    {
        "n_left_control":1,
        "n_right_control":0
    }
)

print("test_tracer")
obs=env.reset()
steps = 0
print(steps)
while True:
    try:
        actions=env.action_space.sample()
        if isinstance(actions,int):
            actions=[actions]
        tracer.update(obs,actions)
        obs, rew, done, info = env.step(actions)
        steps += 1
        if steps % 100 == 0:
            print(steps,rew)
        if done:
            tracer.update(obs)
            tracer.save("temp/random_play.trace")
            break
    except:
        tracer.save("temp/random_play.trace")
        break

