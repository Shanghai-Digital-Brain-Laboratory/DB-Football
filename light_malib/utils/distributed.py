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

import time
import ray
from light_malib.utils.logger import Logger


def get_actor(obj_name, actor_name, max_retries=10):
    actor = None
    ctr = 0
    while ctr <= max_retries:
        ctr += 1
        try:
            if actor is None:
                actor = ray.get_actor(actor_name)
            break
        except Exception as e:
            time.sleep(1)
            Logger.warning(
                "{} retried to get actor {} (ctr: {})".format(obj_name, actor_name, ctr)
            )
            continue
    if actor is None:
        # TODO (jh): how to fail the whole cluster in the error case?
        Logger.error("{} failed to get actor {}".format(obj_name, actor_name))
    else:
        Logger.debug("{} succeeded to get actor {}".format(obj_name, actor_name))
    return actor


def get_resources(resources):
    assert resources is not None
    if "resources" in resources:
        resources["resources"] = dict(resources["resources"])
    return resources
