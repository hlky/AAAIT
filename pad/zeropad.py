from aitemplate.compiler import ops
from aitemplate.frontend import nn, Tensor, IntVar

from typing import Optional, Tuple, Union, Dict, List

def full(fill_value: float, shape: Union[List[int], List[IntVar], Tensor], dtype="float16") -> Tensor:
    if isinstance(shape, Tensor):
        shape = shape._attrs["shape"]
    elif isinstance(shape, list) and isinstance(shape[0], int):
        shape = [IntVar([v, v]) for v in shape]
    elif isinstance(shape, list) and isinstance(shape[0], IntVar):
        pass
    else:
        raise ValueError(f"Invalid shape {shape}")
    return ops.full()(shape=shape, fill_value=fill_value, dtype=dtype)

def ones(shape: Union[List[int], List[IntVar], Tensor], dtype="float16") -> Tensor:
    return full(fill_value=1.0, shape=shape, dtype=dtype)

def zeros(shape: Union[List[int], List[IntVar], Tensor], dtype="float16") -> Tensor:
    return full(fill_value=0.0, shape=shape, dtype=dtype)

class ZeroPad2d(nn.Module):
    """ZeroPad2d module."""

    def __init__(
        self,
        padding: Union[
            int,
            Tuple[int, int], # horizontal, vertical
            Tuple[int, int, int, int], # left, right, top, bottom
            ]
        ):
        """Initialize ZeroPad2d module.

        Args:
            padding (int or tuple[int]): Padding size.
        """
        super().__init__()
        if isinstance(padding, int):
            self.top, self.bottom, self.left, self.right = padding, padding, padding, padding
        elif isinstance(padding, tuple):
            if len(padding) == 2:
                self.left, self.right, self.top, self.bottom = padding[0], padding[0], padding[1], padding[1]
            elif len(padding) == 4:
                self.left, self.right, self.top, self.bottom = padding[0], padding[1], padding[2], padding[3]
            else:
                raise ValueError(f"Invalid padding {padding}")
        else:
            raise ValueError(f"Invalid padding {padding}")
    
    def forward(self, x: Tensor) -> Tensor:
        # x: [batch, height, width, channel]
        shape = [x._attrs['int_var'] for x in ops.size()(x)]
        left_pad = shape.copy()
        left_pad[2] = self.left
        left_pad = zeros(left_pad)
        right_pad = shape.copy()
        right_pad[2] = self.right
        right_pad = zeros(right_pad)
        x = ops.concatenate()([left_pad, x, right_pad], dim=2)
        shape = [x._attrs['int_var'] for x in ops.size()(x)]
        top_pad = shape.copy()
        top_pad[1] = self.top
        top_pad = zeros(top_pad)
        bottom_pad = shape.copy()
        bottom_pad[1] = self.bottom
        bottom_pad = zeros(bottom_pad)
        x = ops.concatenate()([top_pad, x, bottom_pad], dim=1)
        return x
