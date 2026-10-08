import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import ModelConfig
from model.linear import Linear
from model.rope import RotaryEmbedding


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable softmax: subtract the max first so exp() never overflows."""
    x = x - x.max(dim=dim, keepdim=True).values
    e = x.exp()
    return e / e.sum(dim=dim, keepdim=True)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(B, n_kv, S, D) -> (B, n_kv * n_rep, S, D). Query head h reads KV head h // n_rep."""
    if n_rep == 1:
        return x
    B, n_kv, S, D = x.shape
    x = x[:, :, None, :, :].expand(B, n_kv, n_rep, S, D)
    return x.reshape(B, n_kv * n_rep, S, D)


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Scaled dot-product attention with a causal mask.

    q: (B, H, T, D) for the T new tokens.  k, v: (B, H, S, D) for all S tokens so far (S >= T).
    The new tokens sit at absolute positions S-T ... S-1.
    """
    T, S, D = q.shape[-2], k.shape[-2], q.shape[-1]
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(D)                    # (B, H, T, S)
    q_pos = torch.arange(T, device=q.device)[:, None] + (S - T)          # absolute query positions
    k_pos = torch.arange(S, device=q.device)[None, :]
    future = k_pos > q_pos                                               # True where key is ahead of query
    scores = scores.float().masked_fill(future, float("-inf"))           # softmax in fp32
    probs = softmax(scores, dim=-1).to(q.dtype)
    return probs @ v                                                     # (B, H, T, D)


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads, self.n_kv_heads = cfg.n_heads, cfg.n_kv_heads
        self.head_dim, self.n_rep = cfg.head_dim, cfg.n_rep
        self.fused = cfg.fused_attention
        d, hd = cfg.d_model, cfg.head_dim
        self.q_proj = Linear(d, cfg.n_heads * hd, std=cfg.init_std)
        self.k_proj = Linear(d, cfg.n_kv_heads * hd, std=cfg.init_std)
        self.v_proj = Linear(d, cfg.n_kv_heads * hd, std=cfg.init_std)
        # output projection feeds the residual stream: shrink it so 2*n_layers adds stay stable
        self.o_proj = Linear(cfg.n_heads * hd, d, std=cfg.init_std / math.sqrt(2 * cfg.n_layers))

    def forward(self, x, rope: RotaryEmbedding, start_pos: int = 0, past_kv=None):
        """x: (B, T, d_model). Returns (output, (k, v)) where (k, v) is the updated cache."""
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)      # (B, H, T, D)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)   # (B, KV, T, D)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = rope(q, start_pos)
        k = rope(k, start_pos)               # cache stores already-rotated keys

        if past_kv is not None:
            assert past_kv[0].shape[2] == start_pos, "start_pos must equal the cached length"
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        k_all, v_all = repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep)   # expanded copies; cache keeps (k, v)
        S = k_all.shape[2]
        if self.fused and (T == S or T == 1):
            # prefill (T == S) needs the causal mask; a single new token (T == 1) may see everything
            out = F.scaled_dot_product_attention(q, k_all, v_all, is_causal=(T == S))
        else:
            out = causal_attention(q, k_all, v_all)    # my manual version (also handles chunked T < S)
        out = out.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)
        return self.o_proj(out), (k, v)
