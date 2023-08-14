from aitemplate.frontend import nn, Tensor, IntVar
from aitemplate.compiler import ops
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

ACT2FN = {
    "gelu": ops.gelu,
    "fast_gelu": ops.fast_gelu,
    "silu": ops.silu,
}

class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act = "gelu",
        dtype: str = "float16",
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim, dtype=dtype)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim, dtype=dtype)
        self.act = ACT2FN[act]

    def forward(self, x: Tensor) -> Tensor:
        return self.lin2(self.act(self.lin1(x)))

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = ones([num_channels])
        self.bias = zeros([num_channels])
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        u = ops.reduce_mean(1, keepdim=True)(x)
        s = ops.reduce_mean(1, keepdim=True)(ops.pow((x - u), 2))
        x = (x - u) / ops.sqrt(s + self.eps)
        weight = ops.expand()(self.weight, shape=[-1, 1, 1])
        bias = ops.expand()(self.bias, shape=[-1, 1, 1])
        x = weight * x + bias
        return x