import pytest
import torch

from model.attention import GroupedQueryAttention
from model.rope import RotaryEmbedding
from tests.test_attention_backends import small_cfg


@pytest.mark.parametrize("fused", [False, True])
def test_future_tokens_cannot_change_the_past(fused):
    cfg = small_cfg(fused_attention=fused)
    attn = GroupedQueryAttention(cfg)
    rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len)
    x = torch.randn(1, 8, cfg.d_model)
    y1, _ = attn(x, rope)
    x2 = x.clone()
    x2[0, 5] += 10.0                                   # change token 5
    y2, _ = attn(x2, rope)
    assert torch.allclose(y1[0, :5], y2[0, :5], atol=1e-6)      # earlier positions unchanged
    assert not torch.allclose(y1[0, 5:], y2[0, 5:], atol=1e-3)  # token 5 and later do change
