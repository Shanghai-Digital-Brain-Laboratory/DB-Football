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
