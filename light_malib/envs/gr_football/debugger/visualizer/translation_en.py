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

TEAMS = lambda x: [
    "left",  # left
    "right",  # right
    "none",  # free
][x]

ROLES = lambda x: ["GK", "CB", "LB", "RB", "DM", "CM", "LM", "RM", "AM", "CF"][x]

SIMPLE_ROLES = lambda x: ["G", "c", "l", "r", "D", "C", "L", "R", "A", "F"][x]

GAME_MODES = lambda x: [
    "normal",
    "kick off",
    "goal kick",
    "free kick",
    "corner",
    "throw-in",
    "penalty",
][x]

ACTIONS = lambda x: [
    "none",
    "\u2190",  # 左 1
    "\u2196",  # 上左
    "\u2191",  # 上
    "\u2197",  # 上右
    "\u2192",  # 右
    "\u2198",  # 下右
    "\u2193",  # 下
    "\u2199",  # 下左 8
    "long pass",  # 9
    "high pass",  # 10
    "short pass",  # 11
    "shot",  # 12
    "sprint",  # 13
    "no direction",  # 14
    "no sprint",  # 15
    "slide",  # 16
    "dribble",  # 17
    "no dribble",  # 18
    "built-in ai",  # 19
    "-",  # not controlled
][x]

DIRECTIONS = lambda x: [
    "\u2190",  # 左
    "\u2196",  # 上左
    "\u2191",  # 上
    "\u2197",  # 上右
    "\u2192",  # 右
    "\u2198",  # 下右
    "\u2193",  # 下
    "\u2199",  # 下左
    "-",
][x]

INSTRUCTIONS = lambda x: ["RUN", "PASS", "SHOT", "SLIDE"][x]

USE_NEW_INSTRUCTIONS = lambda x: ["OLD", "NEW"][x]
