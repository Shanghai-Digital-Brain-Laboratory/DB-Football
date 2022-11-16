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

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class TrainingDesc:
    agent_id: str
    policy_id: str
    policy_distributions: Dict
    share_policies: bool
    sync: bool
    stopper: Any
    kwargs: Dict = field(default_factory=lambda: {})


@dataclass
class RolloutDesc:
    agent_id: str
    policy_id: str
    # {agent_id:{"policy_ids":np.ndarray,"policy_probs":np.ndarray}}
    policy_distributions: Dict
    share_policies: bool
    sync: bool
    stopper: Any
    kwargs: Dict = field(default_factory=lambda: {})


@dataclass
class RolloutEvalDesc:
    policy_combinations: List[Dict]
    num_eval_rollouts: int
    share_policies: bool
    kwargs: Dict = field(default_factory=lambda: {})


@dataclass
class PrefetchingDesc:
    table_name: str
    batch_size: int
