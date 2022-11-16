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

N_ACTIONS = 19

NONE = -1

(
    NO_OP,
    LEFT,
    TOP_LEFT,
    TOP,
    TOP_RIGHT,
    RIGHT,
    BOTTOM_RIGHT,
    BOTTOM,
    BOTTOM_LEFT,
    LONG_PASS,
    HIGH_PASS,
    SHORT_PASS,
    SHOT,
    SPRINT,
    RELEASE_DIRECTION,
    RELEASE_SPRINT,
    SLIDE,
    DRIBBLE,
    RELEASE_DRIBBLE,
) = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)


def is_direction(action):
    return action >= LEFT and action <= BOTTOM_LEFT


def is_passing(action):
    return action >= LONG_PASS and action <= SHORT_PASS


DIRECTIONS = list(range(LEFT, LEFT + 8))
PASSINGS = list(range(LONG_PASS, SHORT_PASS + 1))

N_DIRECTIONS = len(DIRECTIONS)

BUILT_IN = 19
