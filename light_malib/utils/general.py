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
