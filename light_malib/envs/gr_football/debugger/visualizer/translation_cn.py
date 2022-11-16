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
    "左",  # left
    "右",  # right
    "无",  # free
][x]

ROLES = lambda x: ["GK", "CB", "LB", "RB", "DM", "CM", "LM", "RM", "AM", "CF"][x]

SIMPLE_ROLES = lambda x: ["G", "c", "l", "r", "D", "C", "L", "R", "A", "F"][x]

GAME_MODES = lambda x: ["正常", "开球", "球门球", "任意球", "角球", "边线球", "点球"][x]

ACTIONS = lambda x: [
    "空",
    "\u2190",  # 左 1
    "\u2196",  # 上左
    "\u2191",  # 上
    "\u2197",  # 上右
    "\u2192",  # 右
    "\u2198",  # 下右
    "\u2193",  # 下
    "\u2199",  # 下左 8
    "长传",  # 9
    "高传",  # 10
    "短传",  # 11
    "射门",  # 12
    "冲刺",  # 13
    "释放方向",  # 14
    "释放冲刺",  # 15
    "滑铲",  # 16
    "盘球",  # 17
    "释放盘球",  # 18
    "内置AI",  # 19
    "门将出击",  # 20
    "施压",  # 21
    "全队施压",  # 22
    "切换",  # 23
    "释放长传",  # 24
    "释放高传",  # 25
    "释放短传",  # 26
    "释放射门",  # 27
    "释放出击",  # 28
    "释放铲球",  # 29
    "释放施压",  # 30
    "释放全压",  # 31
    "释放切换",  # 32
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

INSTRUCTIONS = lambda x: ["跑动", "传球", "射门", "铲球"][x]

USE_NEW_INSTRUCTIONS = lambda x: ["旧指令", "新指令"][x]
