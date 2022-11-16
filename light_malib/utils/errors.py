# MIT License

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
