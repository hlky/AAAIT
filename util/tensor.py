from aitemplate.frontend import Tensor, IntVar
from aitemplate.compiler import ops
from typing import List, Union

def full(fill_value: float, shape: Union[List[int], List[IntVar], Tensor], dtype="float16") -> Tensor:
    if isinstance(shape, Tensor):
        shape = shape._attrs["shape"]
    elif isinstance(shape, List[int]):
        shape = [IntVar([v, v]) for v in shape]
    elif isinstance(shape, List[IntVar]):
        pass
    else:
        raise ValueError(f"Invalid shape {shape}")
    return ops.full()(shape=shape, fill_value=fill_value, dtype=dtype)

def ones(shape: Union[List[int], List[IntVar], Tensor], dtype="float16") -> Tensor:
    return full(fill_value=1.0, shape=shape, dtype=dtype)

def zeros(shape: Union[List[int], List[IntVar], Tensor], dtype="float16") -> Tensor:
    return full(fill_value=0.0, shape=shape, dtype=dtype)