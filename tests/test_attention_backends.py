import torch
import torch.nn.functional as F

from model.attention import GroupedQueryAttention, causal_attention, repeat_kv, softmax
from model.config import ModelConfig


def small_cfg(**kw):
    base = dict(vocab_size=100, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=128, max_seq_len=32)
    base.update(kw)
    return ModelConfig(**base)


def test_own_softmax_matches_torch():
    x = torch.randn(3, 5, 7) * 50          # large values would overflow a naive exp()
    assert torch.allclose(softmax(x), torch.softmax(x, dim=-1), atol=1e-6)


def test_causal_attention_matches_torch_reference():
    # PyTorch's built-in attention is used HERE ONLY, as a referee for my implementation.
    q, k, v = (torch.randn(2, 4, 10, 16) for _ in range(3))
    mine = causal_attention(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(mine, ref, atol=1e-5)


def test_repeat_kv_head_mapping():
    x = torch.randn(1, 2, 5, 8)                 # 2 KV heads
    y = repeat_kv(x, 3)                         # -> 6 query heads
    assert y.shape == (1, 6, 5, 8)
    for h in range(6):
        assert torch.equal(y[:, h], x[:, h // 3])


def test_attention_param_count():
    cfg = ModelConfig()
    n = sum(p.numel() for p in GroupedQueryAttention(cfg).parameters())
    assert n == 884_736
