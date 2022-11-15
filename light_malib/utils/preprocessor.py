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

import operator
from abc import ABCMeta, abstractmethod
from functools import reduce

import numpy as np
from gym import spaces

from .typing import DataTransferType, Dict, Sequence, Tuple, List, Any


def _get_batched(data: Any):
    """Get batch dim, nested data must be numpy array like"""

    res = []
    if isinstance(data, Dict):
        for k, v in data.items():
            cleaned_v = _get_batched(v)
            for i, e in enumerate(cleaned_v):
                if i > len(res):
                    res[i] = {}
                res[i][k] = e
    elif isinstance(data, Sequence):
        for v in data:
            cleaned_v = _get_batched(v)
            for i, e in enumerate(cleaned_v):
                if i > len(res):
                    res[i] = []
                res[i].append(e)
    elif isinstance(data, np.ndarray):
        return data
    else:
        raise TypeError(f"Unexpected nested data type: {type(data)}")


class Preprocessor(metaclass=ABCMeta):
    def __init__(self, space: spaces.Space):
        self._original_space = space

    @abstractmethod
    def transform(self, data, nested=False) -> DataTransferType:
        """Transform original data to feet the preprocessed shape. Nested works for nested array."""
        pass

    @abstractmethod
    def write(self, array: DataTransferType, offset: int, data: Any):
        pass

    @property
    def size(self):
        raise NotImplementedError

    @property
    def shape(self):
        raise NotImplementedError

    @property
    def observation_space(self):
        return spaces.Box(
            np.finfo(np.float32).min,
            np.finfo(np.float32).max,
            self.shape,
            dtype=np.float32,
        )


class DictFlattenPreprocessor(Preprocessor):
    def __init__(self, space: spaces.Dict):
        assert isinstance(space, spaces.Dict), space
        super(DictFlattenPreprocessor, self).__init__(space)
        self._preprocessors = {}

        for k, _space in space.spaces.items():
            self._preprocessors[k] = get_preprocessor(_space)(_space)

        self._size = sum([prep.size for prep in self._preprocessors.values()])

    @property
    def shape(self):
        return (self.size,)

    @property
    def size(self):
        return self._size

    def transform(self, data, nested=False) -> DataTransferType:
        """Transform support multi-instance input"""

        if nested:
            data = _get_batched(data)

        if isinstance(data, Dict):
            array = np.zeros(self.shape)
            self.write(array, 0, data)
        elif isinstance(data, Sequence):
            array = np.zeros((len(data),) + self.shape)
            for i in range(len(array)):
                self.write(array[i], 0, data[i])
        else:
            raise TypeError(f"Unexpected type: {type(data)}")

        return array

    def write(self, array: DataTransferType, offset: int, data: Any):
        if isinstance(data, dict):
            for k, _data in sorted(data.items()):
                size = self._preprocessors[k].size
                array[offset : offset + size] = self._preprocessors[k].transform(_data)
                offset += size
        else:
            raise TypeError(f"Unexpected type: {type(data)}")


class TupleFlattenPreprocessor(Preprocessor):
    def __init__(self, space: spaces.Tuple):
        assert isinstance(space, spaces.Tuple), space
        super(TupleFlattenPreprocessor, self).__init__(space)
        self._preprocessors = []
        for k, _space in enumerate(space.spaces):
            self._preprocessors.append(get_preprocessor(_space)(_space))
        self._size = sum([prep.size for prep in self._preprocessors])

    @property
    def size(self):
        return self._size

    @property
    def shape(self):
        return (self.size,)

    def transform(self, data, nested=False) -> DataTransferType:
        if nested:
            data = _get_batched(data)

        if isinstance(data, List):
            array = np.zeros((len(data),) + self.shape)
            for i in range(len(array)):
                self.write(array[i], 0, data[i])
        else:
            array = np.zeros(self.shape)
            self.write(array, 0, data)
        return array

    def write(self, array: DataTransferType, offset: int, data: Any):
        if isinstance(data, Tuple):
            for _data, prep in zip(data, self._preprocessors):
                array[offset : offset + prep.size] = prep.transform(_data)
        else:
            raise TypeError(f"Unexpected type: {type(data)}")


class BoxFlattenPreprocessor(Preprocessor):
    def __init__(self, space: spaces.Box):
        super(BoxFlattenPreprocessor, self).__init__(space)
        self._size = reduce(operator.mul, space.shape)

    @property
    def size(self):
        return self._size

    @property
    def shape(self):
        return (self._size,)

    def transform(self, data, nested=False) -> np.ndarray:
        if nested:
            data = _get_batched(data)

        # if isinstance(data, list):
        #     array = np.vtack(data)
        #     return array
        # else:
        array = np.asarray(data).reshape((-1,) + self.shape).squeeze()
        return array

    def write(self, array, offset, data):
        pass


class BoxStackedPreprocessor(Preprocessor):
    def __init__(self, space: spaces.Box):
        super(BoxStackedPreprocessor, self).__init__(space)
        assert (
            len(space.shape) >= 3
        ), "Stacked box preprocess can only applied to 3D shape"
        self._size = reduce(operator.mul, space.shape)
        self._shape = space.shape

    @property
    def size(self):
        return self._size

    @property
    def shape(self):
        return self._shape

    def transform(self, data, nested=False) -> DataTransferType:
        if nested:
            raise TypeError("Do not support nested transformation yet")

        if isinstance(data, list):
            array = np.stack(data)
            return array
        else:
            array = np.asarray(data)
            return array

    def write(self, array: DataTransferType, offset: int, data: Any):
        pass


class DiscreteFlattenPreprocessor(Preprocessor):
    def __init__(self, space: spaces.Discrete):
        super(DiscreteFlattenPreprocessor, self).__init__(space)
        self._size = space.n

    @property
    def size(self):
        return self._size

    @property
    def shape(self):
        return (self._size,)

    def transform(self, data, nested=False) -> np.ndarray:
        """Transform to one hot"""

        if nested:
            data = _get_batched(data)

        if isinstance(data, int):
            array = np.zeros(self.size, dtype=np.int32)
            array[data] = 1
            return array
        else:
            raise TypeError(f"Unexpected type: {type(data)}")

    def write(self, array, offset, data):
        pass


class Mode:
    FLATTEN = "flatten"
    STACK = "stack"


def get_preprocessor(space: spaces.Space, mode: str = Mode.FLATTEN):
    if mode == Mode.FLATTEN:
        if isinstance(space, spaces.Dict):
            # logger.debug("Use DictFlattenPreprocessor")
            return DictFlattenPreprocessor
        elif isinstance(space, spaces.Tuple):
            # logger.debug("Use TupleFlattenPreprocessor")
            return TupleFlattenPreprocessor
        elif isinstance(space, spaces.Box):
            # logger.debug("Use BoxFlattenPreprocessor")
            return BoxFlattenPreprocessor
        elif isinstance(space, spaces.Discrete):
            return DiscreteFlattenPreprocessor
        else:
            raise TypeError(f"Unexpected space type: {type(space)}")
    elif mode == Mode.STACK:  # for sequential model like CNN and RNN
        if isinstance(space, spaces.Box):
            return BoxStackedPreprocessor
        else:
            raise NotImplementedError
    else:
        raise ValueError(f"Unexpected mode: {mode}")
