# MIT License

# Copyright (c) 2022 DigitalBrain, Yan Song and He jiang
# Copyright (c) 2021 MARL @ SJTU

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


class ELO:
    def __init__(self):
        pass


from typing import Dict
from collections import defaultdict


class EloManager:
    def __init__(self, K, default_elo=1000):
        """
        Create an Elo Rating Manager, which will manager all player's elo in it.
        https://en.wikipedia.org/wiki/Elo_rating_system#Implementing_Elo's_scheme
        :param K: the K-factor to compute Elo.
        """
        self._default_elo = default_elo
        self._elo_table = defaultdict(lambda: self._default_elo)
        self._K = K

    def __getitem__(self, key):
        return self._elo_table[key]

    def __setitem__(self, key, value):
        self._elo_table[key] = value

    def _check_score(self, input_scores):
        s1, s2 = input_scores
        assert s1 + s2 == 1, (s1, s2)

    def record_new_match_result(self, latest_score: Dict[str, float], iter_cnt=1):
        """Update the ratings of players in zero-sum games using the new result.
        :param latest_score: the score is defined as: win-rate + 0.5 * draw-rate
        :return: the updated rating
        """

        assert len(latest_score) == 2, print("latest score = ", latest_score)

        players = list(latest_score.keys())
        E0, E1 = self._compute_expect_score(*players)
        for (player, score), expected_score in zip(latest_score.items(), [E0, E1]):
            for _ in range(iter_cnt):
                self._update_elo(player, score, expected_score)

    def _update_elo(self, player, score, expected_score):
        old_rating = self._elo_table[player]
        self._elo_table[player] = max(
            0, old_rating + self._K * (score - expected_score)
        )

    def _compute_expect_score(self, player0, player1):
        def func(R0, R1):
            return 1 / (1 + 10 ** ((R1 - R0) / 400))

        R0, R1 = self._elo_table[player0], self._elo_table[player1]
        E0 = func(R0, R1)
        E1 = 1 - E0

        return E0, E1
