import torch.nn as nn
import torch
from einops import einsum
import math


def _init_weight(d_in: int, d_out: int, device: torch.device = None, dtype: torch.dtype = None):
    w = nn.Parameter(torch.empty(d_in, d_out, device=device, dtype=dtype))
    return nn.init.trunc_normal_(
        w,
        mean=0.0,
        std=1,
        a=-3,
        b=3,
    )


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        self.w = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = 2 / (in_features + out_features)
        self.w = nn.init.trunc_normal_(
            self.w,
            mean=0.0,
            std=std**2,
            a=-std,
            b=std,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.w, "...  d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        self.w = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        self.w = nn.init.trunc_normal_(
            self.w,
            mean=0.0,
            std=1,
            a=-3,
            b=3,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.w[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.w = nn.Parameter(torch.ones(size=(d_model,), device=device, dtype=dtype))
        self.device = (device,)
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = (x * self.w) / rms
        return x.to(self.dtype)


class Silu(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x * nn.functional.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        if d_ff is None:
            d_ff = int((d_model * 8 / 3) // 64 * 64)
        self.w1 = _init_weight(d_ff, d_model, device, dtype)
        self.w2 = _init_weight(d_model, d_ff, device, dtype)
        self.w3 = _init_weight(d_ff, d_model, device, dtype)
        self.silu = Silu()

    def forward(self, x: torch.Tensor):
        silu_t = self.silu(einsum(x, self.w1, "... d_model, d_ff d_model -> ... d_ff"))
        glu_t = einsum(x, self.w3, "... d_model, d_ff d_model -> ... d_ff")
        t = silu_t * glu_t
        return einsum(t, self.w2, "...  d_ff, d_model d_ff -> ... d_model")


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        freq = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))  # (d_k//2)
        freq = torch.cat([freq, freq])
        pos = torch.arange(0, max_seq_len)
        freq_cis = pos.unsqueeze(1) * freq  # (max_seq_len, d_k)

        self.register_buffer("sin", torch.sin(freq_cis), persistent=False)
        self.register_buffer("cos", torch.cos(freq_cis), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        sin = self.sin[token_positions]  # （..., seq_len, d_k)
        cos = self.cos[token_positions]
        return x * cos + self.rotate_half(x) * sin

    def rotate_half(self, x: torch.Tensor):
        y = torch.empty(x.shape, dtype=x.dtype)
        y[..., 0::2] = -x[..., 1::2]
        y[..., 1::2] = x[..., 0::2]
        return y


def softmax(x: torch.Tensor, dim: int):
    # Handle -inf by replacing with min finite value before max computation
    max_val = torch.max(x, dim=dim, keepdim=True).values
    # Clamp to avoid NaN from -inf - -inf
    max_val = torch.clamp(max_val, min=-1e9)
    val = torch.exp(x - max_val)
    sum_val = torch.sum(val, dim=dim, keepdim=True)
    return val / sum_val
        

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor):
    attention_matrix = einsum(Q, K, "... seq_q dv, ... seq_k dv -> ... seq_q seq_k") / math.sqrt(K.shape[-1])
    attention_matrix = softmax(attention_matrix.masked_fill(~mask, float('-inf')), dim=-1)
    return einsum(attention_matrix, V, "... seq_q seq_kv, ... seq_kv dv -> ... seq_q dv")
        