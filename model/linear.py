import torch
import torch.nn as nn


class Linear(nn.Module):
    """y = x @ W^T  (no bias). W has shape (out_features, in_features)."""

    def __init__(self, in_features: int, out_features: int, std: float = 0.02):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T
