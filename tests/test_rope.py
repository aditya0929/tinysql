import math

import torch

from model.rope import RotaryEmbedding


def test_position_zero_is_identity():
    rope = RotaryEmbedding(head_dim=8, max_seq_len=16)
    x = torch.randn(1, 2, 1, 8)
    assert torch.allclose(rope(x, start_pos=0), x, atol=1e-6)


def test_2d_worked_example():
    # head_dim=2 -> one pair, frequency 1. Vector (1, 0) at position m rotates by m radians.
    rope = RotaryEmbedding(head_dim=2, max_seq_len=8)
    x = torch.tensor([1.0, 0.0]).expand(1, 1, 4, 2)
    y = rope(x)[0, 0]
    for m in range(4):
        assert torch.allclose(y[m], torch.tensor([math.cos(m), math.sin(m)]), atol=1e-6)


def test_preserves_norm():
    rope = RotaryEmbedding(head_dim=64, max_seq_len=128)
    x = torch.randn(2, 9, 128, 64)
    assert torch.allclose(rope(x).norm(dim=-1), x.norm(dim=-1), atol=1e-4)


def test_dot_product_depends_only_on_relative_position():
    rope = RotaryEmbedding(head_dim=64, max_seq_len=128)
    q = torch.randn(64).expand(1, 1, 128, 64)       # same vector at every position
    k = torch.randn(64).expand(1, 1, 128, 64)
    rq, rk = rope(q)[0, 0], rope(k)[0, 0]
    # pairs (m, n) with m - n = 5 must all give the same score
    scores = [torch.dot(rq[m], rk[m - 5]) for m in (5, 20, 77, 127)]
    assert all(torch.allclose(scores[0], s, atol=1e-3) for s in scores)
    # and a different offset gives a different score
    assert not torch.allclose(scores[0], torch.dot(rq[20], rk[10]), atol=1e-3)


def test_start_pos_matches_full_sequence():
    # decoding one token at position 7 must equal row 7 of the full-sequence result
    rope = RotaryEmbedding(head_dim=16, max_seq_len=32)
    x = torch.randn(1, 3, 10, 16)
    full = rope(x)
    step = rope(x[:, :, 7:8], start_pos=7)
    assert torch.allclose(full[:, :, 7:8], step, atol=1e-6)


def test_preserves_dtype_bf16():
    rope = RotaryEmbedding(head_dim=16, max_seq_len=8)
    assert rope(torch.randn(1, 1, 4, 16).to(torch.bfloat16)).dtype == torch.bfloat16
