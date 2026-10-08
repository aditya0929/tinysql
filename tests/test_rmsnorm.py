import torch

from model.config import ModelConfig
from model.rmsnorm import RMSNorm


def test_worked_example():
    # x = [1,2,3,4] -> rms = sqrt(7.5) ~ 2.7386 -> [0.365, 0.730, 1.095, 1.461]
    norm = RMSNorm(4, eps=0.0)
    y = norm(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
    expected = torch.tensor([[1.0, 2.0, 3.0, 4.0]]) / (7.5 ** 0.5)
    assert torch.allclose(y, expected, atol=1e-6)


def test_output_has_unit_rms():
    norm = RMSNorm(576)
    y = norm(torch.randn(2, 5, 576) * 37)
    rms = y.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_scale_invariant():
    norm = RMSNorm(16)
    x = torch.randn(3, 16)
    assert torch.allclose(norm(x), norm(x * 100), atol=1e-4)


def test_tokens_are_independent():
    # Changing one token must not change another token's output (KV-cache safety).
    norm = RMSNorm(16)
    x = torch.randn(1, 4, 16)
    y1 = norm(x)
    x2 = x.clone()
    x2[0, 3] += 5.0
    y2 = norm(x2)
    assert torch.equal(y1[0, :3], y2[0, :3])


def test_preserves_dtype_bf16():
    norm = RMSNorm(32)
    assert norm(torch.randn(2, 32).to(torch.bfloat16)).dtype == torch.bfloat16


def test_param_count_and_config():
    assert sum(p.numel() for p in RMSNorm(576).parameters()) == 576
    cfg = ModelConfig()
    assert cfg.head_dim == 64 and cfg.n_rep == 3
    assert cfg.expected_params() == 125_077_824
