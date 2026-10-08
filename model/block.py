import torch.nn as nn

from model.attention import GroupedQueryAttention
from model.config import ModelConfig
from model.mlp import SwiGLU
from model.rmsnorm import RMSNorm


class DecoderBlock(nn.Module):
    """Pre-norm transformer block:  x = x + attn(norm(x));  x = x + mlp(norm(x))."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = GroupedQueryAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, rope, start_pos: int = 0, past_kv=None):
        h, kv = self.attn(self.attn_norm(x), rope, start_pos, past_kv)
        x = x + h                                   # residual add: tokens exchange information
        x = x + self.mlp(self.mlp_norm(x))          # residual add: each token is processed alone
        return x, kv
