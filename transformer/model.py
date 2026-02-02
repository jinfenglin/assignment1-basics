import torch.nn as nn
import torch
from einops import einsum


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
        self.w = nn.Parameter(torch.ones( size = (d_model,), device = device, dtype = dtype))
        self.device = device,
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = (x * self.w)/rms
        return x.to(self.dtype) 
