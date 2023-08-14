from aitemplate.compiler import ops
from aitemplate.frontend import nn, Tensor, IntVar
from aitemplate.frontend.nn.batch_norm import BatchNorm2d

from typing import Optional, Tuple, Union, Dict, List

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

class ZeroPad2d(nn.Module):
    """ZeroPad2d module."""

    def __init__(
        self,
        padding: Union[
            int,
            Tuple[int, int], # height, width
            Tuple[int, int, int, int], # top, bottom, left, right
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
                self.top, self.bottom, self.left, self.right = padding[0], padding[0], padding[1], padding[1]
            elif len(padding) == 4:
                self.top, self.bottom, self.left, self.right = padding[0], padding[1], padding[2], padding[3]
            else:
                raise ValueError(f"Invalid padding {padding}")
        else:
            raise ValueError(f"Invalid padding {padding}")
    
    def forward(self, x: Tensor) -> Tensor:
        # x: [batch, height, width, channel]
        x_shape = [x._attrs['int_var'] for x in ops.size()(x)]
        left_pad = zeros([x_shape[0], x_shape[1], self.left, x_shape[3]])
        right_pad = zeros([x_shape[0], x_shape[1], self.right, x_shape[3]])
        top_pad = zeros([x_shape[0], self.top, x_shape[2] + self.left + self.right, x_shape[3]])
        bottom_pad = zeros([x_shape[0], self.bottom, x_shape[2] + self.left + self.right, x_shape[3]])
        x = ops.concatenate()([left_pad, x, right_pad], axis=2)
        x = ops.concatenate()([top_pad, x, bottom_pad], axis=1)
        return x

        

def build_padding_layer(cfg: Dict, *args, **kwargs) -> nn.Module:
    """Build padding layer.

    Args:
        cfg (dict): The padding layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate a padding layer.

    Returns:
        nn.Module: Created padding layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')

    cfg_ = cfg.copy()
    padding_type = cfg_.pop('type')
    if padding_type not in PADDING_LAYERS:
        raise KeyError(f'Unrecognized padding type {padding_type}.')
    else:
        padding_layer = PADDING_LAYERS.get(padding_type)

    layer = padding_layer(*args, **kwargs, **cfg_)

    return layer