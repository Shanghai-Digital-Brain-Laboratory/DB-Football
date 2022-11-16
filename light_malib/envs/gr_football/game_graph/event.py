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

"""
1. Result Events: have a cause
2. Action Events: have a result TODO
"""


class ResultEvent:
    def __init__(self, step):
        self.step = step

    def __str__(self):
        raise NotImplementedError

    __repr__ = __str__


### Results Events
class GoalEvent(ResultEvent):
    """
    注意有可能是乌龙球
    """

    def __init__(self, step, score, out_node=None):
        super().__init__(step)
        self.score = score
        self.out_node = out_node

    @property
    def team(self):
        return self.out_node.owned_team

    @property
    def player(self):
        return self.out_node.owned_player

    @property
    def out_step(self):
        return self.out_node.e_step

    def __str__(self):
        return "T{} P{} acts(step {}) and goals(step {}) with score {}".format(
            self.team, self.player, self.out_node.e_step, self.step, self.score
        )


class PassingEvent(ResultEvent):
    def __init__(self, step, out_node=None, in_node=None) -> None:
        super().__init__(step)
        self.out_node = out_node
        self.in_node = in_node

    @property
    def team(self):
        assert self.out_node.owned_team == self.in_node.owned_team
        return self.out_node.owned_team

    @property
    def player(self):
        return self.out_node.owned_player

    @property
    def out_step(self):
        return self.out_node.e_step

    @property
    def receiver(self):
        return self.in_node.owned_player

    @property
    def in_step(self):
        return self.in_node.s_step

    def __str__(self):
        return "T{} P{} passes(step {}) ball to P{}(step {})".format(
            self.team, self.player, self.out_step, self.receiver, self.in_step
        )


class LosingBallEvent(ResultEvent):
    """
    The team cause game mode changing!
    - `0` = `e_GameMode_Normal`
    x `1` = `e_GameMode_KickOff` [BUG] seems won't happen even someone goals.
    - `2` = `e_GameMode_GoalKick
    - `3` = `e_GameMode_FreeKick
    - `4` = `e_GameMode_Corner`
    - `5` = `e_GameMode_ThrowIn`
    - `6` = `e_GameMode_Penalty`

    """

    def __init__(self, step, next_game_mode, out_node=None, **kwargs):
        super().__init__(step)
        self.out_node = out_node
        self.next_game_mode = next_game_mode
        self.extra_info = kwargs

    @property
    def team(self):
        return self.out_node.owned_team

    @property
    def player(self):
        return self.out_node.owned_player

    @property
    def out_step(self):
        return self.out_node.e_step

    def __str__(self):
        return (
            "T{} P{} acts(step {}) and loses(step {}) ball(next game mode:{})".format(
                self.team, self.player, self.out_step, self.step, self.next_game_mode
            )
        )


class InterceptingBallEvent(ResultEvent):
    def __init__(self, step, in_node=None):
        super().__init__(step)
        self.in_node = in_node

    @property
    def team(self):
        return self.in_node.owned_team

    @property
    def player(self):
        return self.in_node.owned_player

    @property
    def in_step(self):
        return self.in_node.s_step

    def __str__(self):
        return "T{} P{} acts(step {}) and intercepts(step {}) ball".format(
            self.team, self.player, self.in_step, self.step
        )


### Action Events
class ActionEvent:
    """
    result_event=None means that nothing significant happens
    """

    def __init__(self, step, result_event=None) -> None:
        self.step = step
        self.result_event = result_event
        self.ball_touched = False


class ShotActionEvent(ActionEvent):
    def __init__(self, step, result_event=None) -> None:
        super().__init__(step, result_event)


class PassActionEvent(ActionEvent):
    def __init__(self, step, pass_type, result_event=None) -> None:
        super().__init__(step, result_event)
        self.pass_type = pass_type


class SlideActionEvent(ActionEvent):
    def __init__(self, step, result_event=None) -> None:
        super().__init__(step, result_event)
