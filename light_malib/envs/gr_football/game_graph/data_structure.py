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


class Node:
    def __init__(self, idx, t, p, s) -> None:
        self.owned_team = t
        self.owned_player = p
        self.s_step = s
        self.e_step = None  # include
        self.idx = idx

    def set_e_step(self, step):
        self.e_step = step

    def __str__(self) -> str:
        return "N[T{} P{} {} {}]".format(
            self.owned_team, self.owned_player, self.s_step, self.e_step
        )

    __repr__ = __str__


class Chain:
    def __init__(self, idx) -> None:
        self.nodes = []
        self.idx = idx

    def __str__(self):
        s = "C[T{} {} {}]: ".format(self.owned_team, self.s_step, self.e_step)
        s += str(self.nodes[0])
        for i in range(1, len(self)):
            s += "->" + str(self.nodes[i])
        return s

    __repr__ = __str__

    def append(self, node):
        self.nodes.append(node)

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, idx):
        return self.nodes[idx]

    @property
    def owned_team(self):
        return self.nodes[0].owned_team if len(self) > 0 else None

    @property
    def s_step(self):
        return self.nodes[0].s_step if len(self) > 0 else None

    @property
    def e_step(self):
        return self.nodes[-1].e_step if len(self) > 0 else None

    @property
    def n_steps(self):
        return self.e_step - self.s_step + 1 if len(self) > 0 else None


class Subgame:
    def __init__(self, idx) -> None:
        self.chains = []
        self.idx = idx

    def __str__(self):
        s = "S[T{} {} {}]: \n".format(self.owned_team, self.s_step, self.e_step)
        for idx, chain in enumerate(self.chains):
            s += "  <{:04d}> {}\n".format(chain.s_step, str(chain))
        return s

    __repr__ = __str__

    def append(self, chain):
        self.chains.append(chain)

    def __len__(self):
        return len(self.chains)

    def __getitem__(self, idx):
        return self.chains[idx]

    @property
    def owned_team(self):
        return self.chains[0].owned_team if len(self) > 0 else None

    @property
    def s_step(self):
        return self.chains[0].s_step if len(self) > 0 else None

    @property
    def e_step(self):
        return self.chains[-1].e_step if len(self) > 0 else None

    @property
    def n_steps(self):
        return self.e_step - self.s_step + 1 if len(self) > 0 else None
