import torch

from model.generate import filter_logits, generate, sample_from_probs, sample_next
from model.model import TinySQL
from tests.test_attention_backends import small_cfg


def test_greedy_with_cache_equals_greedy_without_cache():
    torch.manual_seed(0)
    model = TinySQL(small_cfg()).eval()
    prompt = torch.randint(0, 100, (2, 5))
    fast = generate(model, prompt, max_new_tokens=15, temperature=0)

    ids = prompt                                   # reference: recompute the whole sequence every step
    with torch.no_grad():
        for _ in range(15):
            logits, _, _ = model(ids)
            ids = torch.cat([ids, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    assert torch.equal(fast, ids)


def test_tiny_temperature_is_greedy():
    logits = torch.tensor([[1.0, 5.0, 2.0, 0.5]])
    g = torch.Generator().manual_seed(0)
    assert all(sample_next(logits, temperature=0.01, generator=g).item() == 1 for _ in range(50))


def test_top_k_keeps_exactly_k():
    logits = torch.randn(3, 50)
    out = filter_logits(logits, top_k=5)
    assert ((out > float("-inf")).sum(-1) == 5).all()
    assert torch.equal(out.max(-1).values, logits.max(-1).values)       # best token survives untouched


def test_top_p_nucleus():
    probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    logits = probs.log()
    assert (filter_logits(logits, top_p=0.7) > float("-inf")).sum() == 2    # 0.5 + 0.3 covers 0.7
    assert (filter_logits(logits, top_p=0.8) > float("-inf")).sum() == 3    # the 0.15 token is still needed
    assert (filter_logits(logits, top_p=0.0) > float("-inf")).sum() == 1    # top token always survives


def test_sampler_matches_distribution():
    probs = torch.tensor([[0.6, 0.3, 0.1]]).expand(20000, 3)
    draws = sample_from_probs(probs, torch.Generator().manual_seed(1))
    freq = torch.bincount(draws, minlength=3) / 20000
    assert torch.allclose(freq, torch.tensor([0.6, 0.3, 0.1]), atol=0.02)


def test_sampler_never_picks_filtered_tokens():
    logits = torch.randn(1, 40).expand(5000, 40)
    allowed = set(logits[0].topk(4).indices.tolist())
    picks = sample_next(logits, top_k=4, generator=torch.Generator().manual_seed(2))
    assert set(picks.tolist()) <= allowed


def test_stop_token_ends_generation():
    torch.manual_seed(0)
    model = TinySQL(small_cfg()).eval()
    prompt = torch.randint(0, 100, (1, 4))
    full = generate(model, prompt, max_new_tokens=12, temperature=0)
    stop = full[0, 4 + 3].item()                                  # pick the 4th generated token as "stop"
    first = (full[0, 4:] == stop).nonzero()[0].item()             # its first occurrence
    out = generate(model, prompt, max_new_tokens=12, temperature=0, stop_ids=[stop])
    assert out.shape[1] == 4 + first + 1 and out[0, -1].item() == stop


def test_same_seed_same_text():
    model = TinySQL(small_cfg()).eval()
    prompt = torch.randint(0, 100, (1, 4))
    a = generate(model, prompt, 10, temperature=1.0, top_k=20, generator=torch.Generator().manual_seed(7))
    b = generate(model, prompt, 10, temperature=1.0, top_k=20, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)
