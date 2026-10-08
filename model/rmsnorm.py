import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """y = x / sqrt(mean(x^2) + eps) * g, computed per token over the last dimension."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))   # learned gain g, starts at 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.float()                                              # fp32 for the delicate math
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (self.weight * x).to(in_dtype)                      # gain in fp32, cast back once
