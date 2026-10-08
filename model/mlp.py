import math

import torch
import torch.nn as nn

from model.config import ModelConfig
from model.linear import Linear


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU / swish: x * sigmoid(x). Smooth, near 0 for very negative x, near x for large x."""
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """down( silu(gate(x)) * up(x) ): a gated feed-forward network applied to each token."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d, h = cfg.d_model, cfg.d_ff
        self.gate_proj = Linear(d, h, std=cfg.init_std)
        self.up_proj = Linear(d, h, std=cfg.init_std)
        # feeds the residual stream, so scaled like the attention output projection
        self.down_proj = Linear(h, d, std=cfg.init_std / math.sqrt(2 * cfg.n_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(silu(self.gate_proj(x)) * self.up_proj(x))
