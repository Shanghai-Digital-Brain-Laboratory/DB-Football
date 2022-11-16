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
