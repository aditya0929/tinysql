import math

import pytest
import torch

from optim.adamw import AdamW, clip_grad_norm_
from optim.schedule import cosine_lr


def make_model(seed=0):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(6, 12), torch.nn.Tanh(), torch.nn.Linear(12, 3))


def groups(model):
    weights = [p for n, p in model.named_parameters() if p.dim() >= 2]
    biases = [p for n, p in model.named_parameters() if p.dim() < 2]
    return [{"params": weights, "weight_decay": 0.1}, {"params": biases, "weight_decay": 0.0}]


def train(opt_factory, steps):
    model = make_model()
    opt = opt_factory(model)
    gen = torch.Generator().manual_seed(1)
    for _ in range(steps):
        x, y = torch.randn(16, 6, generator=gen), torch.randn(16, 3, generator=gen)
        loss = (model(x) - y).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def test_matches_torch_adamw_over_100_steps():
    mine = train(lambda m: AdamW(groups(m), lr=3e-3, betas=(0.9, 0.95), eps=1e-8), 100)
    ref = train(lambda m: torch.optim.AdamW(groups(m), lr=3e-3, betas=(0.9, 0.95), eps=1e-8), 100)
    for a, b in zip(mine.parameters(), ref.parameters()):
        assert torch.allclose(a, b, atol=1e-6), (a - b).abs().max()


def test_weight_decay_is_decoupled_and_respects_zero_group():
    p = torch.nn.Parameter(torch.ones(4))
    q = torch.nn.Parameter(torch.ones(4))
    opt = AdamW([{"params": [p], "weight_decay": 0.5}, {"params": [q], "weight_decay": 0.0}], lr=0.1)
    p.grad, q.grad = torch.zeros(4), torch.zeros(4)           # zero gradient: only the decay can move the weights
    opt.step()
    assert torch.allclose(p, torch.full((4,), 1 - 0.1 * 0.5)) and torch.equal(q, torch.ones(4))


def test_state_dict_round_trip_continues_identically():
    def run(resume):
        model = make_model()
        opt = AdamW(groups(model), lr=1e-2)
        gen = torch.Generator().manual_seed(4)

        def step():
            x, y = torch.randn(8, 6, generator=gen), torch.randn(8, 3, generator=gen)
            loss = (model(x) - y).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        for _ in range(10):
            step()
        if resume:
            sd = opt.state_dict()
            weights = {k: v.clone() for k, v in model.state_dict().items()}
            model = make_model()
            model.load_state_dict(weights)
            opt = AdamW(groups(model), lr=1e-2)
            opt.load_state_dict(sd)
        for _ in range(10):
            step()
        return torch.cat([p.flatten() for p in model.parameters()])

    assert torch.equal(run(resume=False), run(resume=True))


def test_clip_grad_norm():
    a, b = torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(2))
    a.grad, b.grad = torch.tensor([3.0, 0.0]), torch.tensor([0.0, 4.0])      # global norm 5
    norm = clip_grad_norm_([a, b], max_norm=1.0)
    assert math.isclose(norm.item(), 5.0, rel_tol=1e-6)
    assert math.isclose(torch.sqrt(a.grad.pow(2).sum() + b.grad.pow(2).sum()).item(), 1.0, rel_tol=1e-4)
    a.grad, b.grad = torch.tensor([0.1, 0.0]), torch.tensor([0.0, 0.1])      # already small: untouched
    clip_grad_norm_([a, b], max_norm=1.0)
    assert torch.allclose(a.grad, torch.tensor([0.1, 0.0]))
    a.grad = torch.tensor([float("inf"), 0.0])
    assert not torch.isfinite(clip_grad_norm_([a, b], max_norm=1.0))


def test_cosine_schedule_shape():
    kw = dict(max_steps=1000, peak_lr=1e-3, warmup_steps=100, min_lr_ratio=0.1)
    assert cosine_lr(0, **kw) == pytest.approx(1e-5) and cosine_lr(99, **kw) == pytest.approx(1e-3)
    assert cosine_lr(100, **kw) == pytest.approx(1e-3)
    assert cosine_lr(1000, **kw) == pytest.approx(1e-4) and cosine_lr(5000, **kw) == pytest.approx(1e-4)
    lrs = [cosine_lr(s, **kw) for s in range(100, 1001)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))                          # monotone decay after warmup
