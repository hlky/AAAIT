from aitemplate.compiler import ops
from aitemplate.frontend import nn, Tensor, IntVar

from typing import Optional, Tuple, Union, Dict, List

class ReflectionPad2d(nn.Module):
    """ReflectionPad2d module."""

    def __init__(
        self,
        padding: Union[
            int,
            Tuple[int, int], # horizontal, vertical
            Tuple[int, int, int, int], # left, right, top, bottom
            ]
        ):
        """Initialize ReflectionPad2d module.
        >>> m = nn.ReflectionPad2d(2)
        >>> input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3)
        >>> input
        tensor([[[[0., 1., 2.],
                [3., 4., 5.],
                [6., 7., 8.]]]])
        >>> m(input)
        tensor([[[[8., 7., 6., 7., 8., 7., 6.],
                [5., 4., 3., 4., 5., 4., 3.],
                [2., 1., 0., 1., 2., 1., 0.],
                [5., 4., 3., 4., 5., 4., 3.],
                [8., 7., 6., 7., 8., 7., 6.],
                [5., 4., 3., 4., 5., 4., 3.],
                [2., 1., 0., 1., 2., 1., 0.]]]])
        >>> # using different paddings for different sides
        >>> m = nn.ReflectionPad2d((1, 1, 2, 0))
        >>> m(input)
        tensor([[[[7., 6., 7., 8., 7.],
                [4., 3., 4., 5., 4.],
                [1., 0., 1., 2., 1.],
                [4., 3., 4., 5., 4.],
                [7., 6., 7., 8., 7.]]]])

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
        b, h, w, c = shape
        h = h.upper_bound()
        top_slices = None
        if self.top > 0:
            top_slices = []
            for i in range(self.top):
                s = -(self.top - i)
                e = h + 1 + s
                print(i, s, e)
                top_slices.append(ops.dynamic_slice()(x, [0, s, 0, 0], [b, e, w, c]))
            top_slices = list(reversed(top_slices))
            top_slices = ops.concatenate()(top_slices, dim=1)

        bottom_slices = None
        if self.bottom > 0:
            bottom_slices = []
            for i in range(self.bottom):
                s = i
                e = i + 1
                print(i, s, e)
                bottom_slices.append(ops.dynamic_slice()(x, [0, s, 0, 0], [b, e, w, c]))
            bottom_slices = list(reversed(bottom_slices))
            bottom_slices = ops.concatenate()(bottom_slices, dim=1)
        
        if top_slices is not None:
            x = ops.concatenate()([top_slices, x], dim=1)
        if bottom_slices is not None:
            x = ops.concatenate()([x, bottom_slices], dim=1)

        shape = [x._attrs['int_var'] for x in ops.size()(x)]
        b, h, w, c = shape
        w = w.upper_bound()
        left_slices = None
        if self.left > 0:
            left_slices = []
            for i in range(self.left):
                s = -(self.left - i)
                e = w + 1 + s
                print(i, s, e)
                left_slices.append(ops.dynamic_slice()(x, [0, 0, s, 0], [b, h, e, c]))
            left_slices = list(reversed(left_slices))
            left_slices = ops.concatenate()(left_slices, dim=2)

        right_slices = None
        if self.right > 0:
            right_slices = []
            for i in range(self.right):
                s = i
                e = i + 1
                print(i, s, e)
                right_slices.append(ops.dynamic_slice()(x, [0, 0, s, 0], [b, h, e, c]))
            right_slices = list(reversed(right_slices))
            right_slices = ops.concatenate()(right_slices, dim=2)

        if left_slices is not None:
            x = ops.concatenate()([left_slices, x], dim=2)
        if right_slices is not None:
            x = ops.concatenate()([x, right_slices], dim=2)

        return x
