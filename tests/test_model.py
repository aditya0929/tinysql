import math

import pytest
import torch
import torch.nn.functional as F

from model.model import TinySQL, cross_entropy
from tests.test_attention_backends import small_cfg


def test_cross_entropy_matches_torch_with_ignored_positions():
    logits = torch.randn(3, 7, 50) * 5
    targets = torch.randint(0, 50, (3, 7))
    targets[0, :3] = -100                                        # masked prompt tokens
    ref = F.cross_entropy(logits.reshape(-1, 50), targets.reshape(-1), ignore_index=-100)
    assert torch.allclose(cross_entropy(logits, targets), ref, atol=1e-5)


def test_initial_loss_is_near_ln_vocab():
    cfg = small_cfg(vocab_size=1000)
    model = TinySQL(cfg)
    ids = torch.randint(0, 1000, (4, 32))
    targets = torch.randint(0, 1000, (4, 32))      # unrelated to the inputs; targets=ids would leak via tied embeddings
    _, loss, _ = model(ids, targets=targets)
    assert abs(loss.item() - math.log(1000)) < 0.5


@pytest.mark.parametrize("fused", [False, True])
def test_model_cache_decode_matches_full_forward(fused):
    cfg = small_cfg(fused_attention=fused)
    model = TinySQL(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 14))
    with torch.no_grad():
        full, _, _ = model(ids)
        out, _, kvs = model(ids[:, :8])
        pieces = [out]
        for t in range(8, 14):
            out, _, kvs = model(ids[:, t : t + 1], start_pos=t, past_kvs=kvs)
            pieces.append(out)
    assert torch.allclose(full, torch.cat(pieces, dim=1), atol=1e-4)


def test_model_can_overfit_one_batch():
    torch.manual_seed(0)
    cfg = small_cfg(vocab_size=64)
    model = TinySQL(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)   # placeholder: my own AdamW comes later
    ids = torch.randint(0, 64, (2, 17))
    x, y = ids[:, :-1], ids[:, 1:]
    for _ in range(300):
        _, loss, _ = model(x, targets=y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05


def test_cross_entropy_gradient_matches_torch():
    torch.manual_seed(0)
    base = torch.randn(3, 7, 50) * 4
    targets = torch.randint(0, 50, (3, 7))
    targets[1, :4] = -100                                         # masked positions must get exactly zero gradient
    a = base.clone().requires_grad_()
    b = base.clone().requires_grad_()
    mine = cross_entropy(a, targets)
    ref = F.cross_entropy(b.reshape(-1, 50), targets.reshape(-1), ignore_index=-100)
    mine.backward()
    ref.backward()
    assert torch.allclose(mine, ref, atol=1e-6)
    assert torch.allclose(a.grad, b.grad, atol=1e-6)
    assert float(a.grad[1, :4].abs().max()) == 0.0


def test_cross_entropy_bf16_logits_keep_dtype_and_stay_close():
    torch.manual_seed(1)
    base = (torch.randn(2, 5, 40) * 3)
    targets = torch.randint(0, 40, (2, 5))
    a = base.to(torch.bfloat16).requires_grad_()
    loss = cross_entropy(a, targets)
    loss.backward()
    assert a.grad.dtype == torch.bfloat16
    ref = F.cross_entropy(base.to(torch.bfloat16).float().reshape(-1, 40), targets.reshape(-1))
    assert abs(loss.item() - ref.item()) < 1e-5


def test_cross_entropy_upstream_gradient_scales():
    logits = torch.randn(4, 9, requires_grad=True)
    targets = torch.randint(0, 9, (4,))
    (3.0 * cross_entropy(logits, targets)).backward()
    g3 = logits.grad.clone()
    logits.grad = None
    cross_entropy(logits, targets).backward()
    assert torch.allclose(g3, 3.0 * logits.grad, atol=1e-6)
