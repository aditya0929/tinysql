import random

import pytest
import torch

transformers = pytest.importorskip("transformers")

from eval.baselines.finetune_hf import IGNORE, accumulate_backward, collate


def tiny_hf_model():
    cfg = transformers.LlamaConfig(vocab_size=50, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    torch.manual_seed(0)
    return transformers.LlamaForCausalLM(cfg).eval()


def make_batch(n=12, seed=0):
    rng = random.Random(seed)
    batch = []
    for _ in range(n):
        p, c = rng.randint(3, 15), rng.randint(2, 9)                  # prompt and completion lengths vary
        ids = [rng.randint(1, 49) for _ in range(p + c)]
        batch.append((ids, [IGNORE] * p + ids[p:]))
    return batch


def test_micro_batched_gradient_equals_the_full_batch_gradient():
    batch = make_batch()
    full = tiny_hf_model()
    ids, labels, mask = collate(batch, pad_id=0)
    full(input_ids=ids, attention_mask=mask, labels=labels).loss.backward()

    micro = tiny_hf_model()
    accumulate_backward(micro, batch, micro_batch=5, pad_id=0, device=torch.device("cpu"))   # 5 + 5 + 2
    for (name, a), (_, b) in zip(full.named_parameters(), micro.named_parameters()):
        assert torch.allclose(a.grad, b.grad, atol=1e-6), name


def test_reported_loss_is_the_full_batch_loss():
    batch = make_batch(seed=1)
    ids, labels, mask = collate(batch, pad_id=0)
    expected = tiny_hf_model()(input_ids=ids, attention_mask=mask, labels=labels).loss.item()
    got = accumulate_backward(tiny_hf_model(), batch, micro_batch=4, pad_id=0, device=torch.device("cpu"))
    assert abs(got - expected) < 1e-5
