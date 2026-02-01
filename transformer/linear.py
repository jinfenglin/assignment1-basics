import torch.nn as nn
import torch
from einops import einsum

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device = None, dtype: torch.dtype=None):
        super().__init__()
        self.w =  nn.Parameter(torch.empty(out_features, in_features, device = device, dtype = dtype))
        std = 2/ (in_features + out_features)
        self.w = nn.init.trunc_normal_(
            self.w,
            mean = 0.0,
            std = std ** 2,
            a = - std,
            b = std,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.w, "...  d_in, d_out d_in -> ... d_out")