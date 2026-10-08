import torch
import torch.nn as nn


class Embedding(nn.Module):
    """A lookup table: token id -> row of a (vocab_size, d_model) matrix."""

    def __init__(self, vocab_size: int, d_model: int, std: float = 0.02):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, d_model) * std)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.weight[ids]          # (B, T) -> (B, T, d_model)
