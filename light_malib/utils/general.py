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

from collections import OrderedDict

import torch
import numpy as np
import copy

from .typing import Dict, Callable

# TODO(ming): will be replaced with many dicts
def iter_dicts_recursively(d1, d2):
    """Assuming dicts have the exact same structure."""
    for k, v in d1.items():
        assert k in d2

        if isinstance(v, (dict, OrderedDict)):
            yield from iter_dicts_recursively(d1[k], d2[k])
        else:
            yield d1, d2, k, d1[k], d2[k]


def iter_many_dicts_recursively(*d, history=None):
    """Assuming dicts have the exact same structure, or raise KeyError."""

    for k, v in d[0].items():
        if isinstance(v, (dict, OrderedDict)):
            yield from iter_many_dicts_recursively(
                *[_d[k] for _d in d],
                history=history + [k] if history is not None else None,
            )
        else:
            if history is None:
                yield d, k, tuple([_d[k] for _d in d])
            else:
                # print("fff", k, [list(_d.keys())for _d in d])
                yield history + [k], d, k, tuple([_d[k] for _d in d])


def _default_dtype_mapping(dtype):
    # FIXME(ming): cast 64 to 32?
    if dtype in [np.int32, np.int64, int]:
        return torch.int32
    elif dtype in [float, np.float32, np.float16]:
        return torch.float32
    elif dtype == np.float64:
        return torch.float64
    elif dtype in [bool, np.bool_]:
        return torch.float32
    else:
        raise NotImplementedError(f"dtype: {dtype} has no transmission rule.") from None


# wrap with type checking
def _walk(caster, v):
    if isinstance(v, Dict):
        for k, _v in v.items():
            v[k] = _walk(caster, _v)
    else:
        v = caster(v)
    return v


def tensor_cast(
    custom_caster: Callable = None,
    callback: Callable = None,
    dtype_mapping: Dict = None,
    device="cpu",
):
    """Casting the inputs of a method into tensors if needed.

    Note:
        This function does not support recursive iteration.

    Args:
        custom_caster (Callable, optional): Customized caster. Defaults to None.
        callback (Callable, optional): Callback function, accepts returns of wrapped function as inputs. Defaults to None.
        dtype_mapping (Dict, optional): Specify the data type for inputs which you wanna. Defaults to None.

    Returns:
        Callable: A decorator.
    """
    dtype_mapping = dtype_mapping or _default_dtype_mapping
    cast_to_tensor = custom_caster or (
        lambda x: torch.FloatTensor(x.copy()).to(
            device=device, dtype=dtype_mapping(x.dtype)
        )
        if not isinstance(x, torch.Tensor)
        else x
    )

    def decorator(func):
        def wrap(*args, **kwargs):
            new_args = []
            for i, arg in enumerate(args):
                new_args.append(_walk(cast_to_tensor, arg))
            for k, v in kwargs.items():
                kwargs[k] = _walk(cast_to_tensor, v)
            rets = func(*new_args, **kwargs)
            if callback is not None:
                callback(rets)
            return rets

        return wrap

    return decorator
