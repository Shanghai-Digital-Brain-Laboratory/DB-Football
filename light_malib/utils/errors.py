# Copyright 2022 DigitalBrain, Yan Song and He jiang
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


class Error(Exception):
    pass


class RepeatLockingError(Error):
    """Raised when lock parameter description"""

    pass


class UnexpectedType(Error):
    """Raised when type is not expected"""

    pass


class NoEnoughSpace(Error):
    """Raised when population size is not enough for new policies registry"""

    pass


class UnexpectedAlgorithm(Error):
    """Raised when registered an unkown algorithm in agent.AgentInterface"""

    pass


class TypeError(Error):
    """Raised when illegal type"""

    pass


class RepeatedAssignError(Error):
    """Raised when repeated assign value to a not-None dict"""

    pass


class OversampleError(Error):
    """Raised when over-sample from a offline data table"""

    pass


class NoEnoughDataError(Error):
    pass


class RegisterFailure(Error):
    pass
