from typing import List

from aitemplate.compiler import ops
from aitemplate.frontend import nn, Tensor, IntVar

def ones(shape: List[int]):
    shape = [IntVar(values=[dim, dim]) for dim in shape]
    return ops.full()(shape, 1.0)

def rsqrt(x: Tensor):
    return 1 / ops.sqrt(x)

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(ones([hidden_size]))
        self.variance_epsilon = eps

    def forward(self, hidden_states: Tensor):
        input_dtype = hidden_states.dtype()
        hidden_states = ops.cast()(hidden_states, "float32")
        variance = ops.reduce_mean(-1, keepdim=True)(ops.pow(hidden_states, 2))
        hidden_states = hidden_states * rsqrt(variance + self.variance_epsilon)
        return self.weight * ops.cast()(hidden_states, input_dtype)

class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq_arange = Tensor(shape=[self.dim // 2], name="rotary_inv_freq_arange")
        self.inv_freq = 1.0 / (self.base ** (inv_freq_arange / self.dim))
        self.max_seq_len_cached = max_position_embeddings
        t = Tensor(shape=[self.max_seq_len_cached], name="rotary_t")
        # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # outer product
        freqs = []
        for i in range(t.shape()[0]):
            for j in range(self.inv_freq.shape()[0]):
                freqs[i, j] = t[i] * self.inv_freq[j]
        freqs = Tensor(shape=[self.max_seq_len_cached, self.dim // 2], name="rotary_freqs")
    