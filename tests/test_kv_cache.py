import pytest
import torch

from model.attention import GroupedQueryAttention
from model.rope import RotaryEmbedding
from tests.test_attention_backends import small_cfg


@pytest.mark.parametrize("fused", [False, True])
def test_incremental_decode_matches_full_forward(fused):
    cfg = small_cfg(fused_attention=fused)
    attn = GroupedQueryAttention(cfg)
    rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len)
    x = torch.randn(2, 12, cfg.d_model)
    full, _ = attn(x, rope)

    # prefill the first 7 tokens, then feed the remaining 5 one at a time
    out, cache = attn(x[:, :7], rope)
    pieces = [out]
    for t in range(7, 12):
        out, cache = attn(x[:, t : t + 1], rope, start_pos=t, past_kv=cache)
        pieces.append(out)
    assert torch.allclose(full, torch.cat(pieces, dim=1), atol=1e-5)


def test_cache_is_smaller_with_gqa():
    cfg = small_cfg()
    attn = GroupedQueryAttention(cfg)
    rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len)
    _, (k, v) = attn(torch.randn(1, 6, cfg.d_model), rope)
    assert k.shape[1] == cfg.n_kv_heads < cfg.n_heads       # cache stores 2 heads, not 4
