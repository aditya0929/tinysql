import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split the last dim into halves (x1, x2) and return (-x2, x1): a 90-degree turn per pair."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings.

    Dimension i is paired with dimension i + head_dim/2. Pair i is rotated by the
    angle  position * theta^(-2i/head_dim).  Applied to queries and keys only.
    """

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10_000.0):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even (dimensions are rotated in pairs)"
        # one frequency per pair: fast-spinning pairs first, slow-spinning pairs last
        inv_freq = 1.0 / theta ** (torch.arange(0, head_dim, 2).float() / head_dim)   # (D/2,)
        positions = torch.arange(max_seq_len).float()                                  # (T,)
        angles = torch.outer(positions, inv_freq)                                      # (T, D/2)
        angles = torch.cat((angles, angles), dim=-1)                                   # (T, D)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """x: (batch, n_heads, seq_len, head_dim). start_pos: position of x's first token
        (non-zero when decoding with a KV cache)."""
        seq_len = x.shape[-2]
        cos = self.cos[start_pos : start_pos + seq_len]    # (T, D), broadcasts over batch/heads
        sin = self.sin[start_pos : start_pos + seq_len]
        out = x.float() * cos + rotate_half(x.float()) * sin
        return out.to(x.dtype)
