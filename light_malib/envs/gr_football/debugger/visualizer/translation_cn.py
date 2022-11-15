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

TEAMS=lambda x: [
    "左", # left
    "右", # right
    "无", # free
][x]

ROLES=lambda x: [
    "GK",
    "CB",
    "LB",
    "RB",
    "DM",
    "CM",
    "LM",
    "RM",
    "AM",
    "CF"    
][x]

SIMPLE_ROLES=lambda x: [
    "G",
    "c",
    "l",
    "r",
    "D",
    "C",
    "L",
    "R",
    "A",
    "F"    
][x]

GAME_MODES= lambda x:[
    "正常",
    "开球",
    "球门球",
    "任意球",
    "角球",
    "边线球",
    "点球"
][x]

ACTIONS=lambda x: [
    "空",
    "\u2190", # 左 1
    "\u2196", # 上左
    "\u2191", # 上
    "\u2197", # 上右
    "\u2192", # 右
    "\u2198", # 下右
    "\u2193", # 下
    "\u2199", # 下左 8
    "长传", # 9
    "高传", # 10
    "短传", # 11
    "射门", # 12
    "冲刺", # 13
    "释放方向", # 14
    "释放冲刺", # 15
    "滑铲", # 16
    "盘球", # 17
    "释放盘球", # 18
    "内置AI", # 19
    "门将出击", # 20
    "施压", # 21
    "全队施压", # 22
    "切换", # 23
    "释放长传", # 24
    "释放高传", # 25
    "释放短传", # 26
    "释放射门", # 27
    "释放出击", # 28
    "释放铲球", # 29 
    "释放施压", # 30
    "释放全压", # 31
    "释放切换", # 32
    "-" # not controlled
][x]

DIRECTIONS=lambda x: [
    "\u2190", # 左
    "\u2196", # 上左
    "\u2191", # 上
    "\u2197", # 上右
    "\u2192", # 右
    "\u2198", # 下右
    "\u2193", # 下
    "\u2199", # 下左
    "-"
][x]

INSTRUCTIONS=lambda x: [
    "跑动",
    "传球",
    "射门",
    "铲球"
][x]

USE_NEW_INSTRUCTIONS=lambda x:[
    "旧指令",
    "新指令"
][x]