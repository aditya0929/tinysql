import json
import os

import numpy as np
import pytest
import torch

from data.loader import SOURCES
from model.config import ModelConfig
from model.model import TinySQL
from train import pretrain
from train.utils import LossScaler, latest_checkpoint, param_groups, save_checkpoint

MODEL = dict(vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=64, max_seq_len=32)


@pytest.fixture(scope="module")
def shards(tmp_path_factory):
    """Learnable data: a repeating pattern, so a tiny model's loss must fall quickly."""
    root = tmp_path_factory.mktemp("shards")
    for split in ("train", "val"):
        os.makedirs(root / split)
        for s in SOURCES:
            (np.arange(4000) % 16 + 1).astype(np.uint16).tofile(root / split / f"{s}__p0.bin")
    return str(root)


def cfg_for(shards, out_dir, **kw):
    cfg = dict(model=MODEL, shards_dir=shards, out_dir=str(out_dir), seq_len=16, micro_batch=4, grad_accum=2,
               max_steps=20, warmup_steps=3, peak_lr=3e-3, log_interval=5, eval_interval=1000, ckpt_interval=1000,
               sample_prompts=[], seed=0)
    cfg.update(kw)
    return cfg


def read_metrics(out_dir):
    with open(os.path.join(out_dir, "metrics.jsonl")) as f:
        return [json.loads(l) for l in f]


def test_loss_goes_down(shards, tmp_path):
    pretrain.train(cfg_for(shards, tmp_path, max_steps=60, log_interval=10))
    losses = [m["loss"] for m in read_metrics(tmp_path) if "loss" in m]
    assert losses[-1] < 0.5 * losses[0], losses


def test_resume_gives_identical_weights(shards, tmp_path):
    straight, _ = pretrain.train(cfg_for(shards, tmp_path / "a"))
    pretrain.train(cfg_for(shards, tmp_path / "b", stop_at_step=10))          # stop half way (checkpoint is written)
    assert latest_checkpoint(str(tmp_path / "b")).endswith("ckpt_0000010.pt")
    resumed, step = pretrain.train(cfg_for(shards, tmp_path / "b"))           # picks the checkpoint up and finishes
    assert step == 20
    for (n, a), (_, b) in zip(straight.state_dict().items(), resumed.state_dict().items()):
        assert torch.equal(a, b), n


def test_non_finite_gradients_skip_the_step(shards, tmp_path, monkeypatch):
    real = pretrain.clip_grad_norm_
    calls = []

    def fake(params, max_norm):
        calls.append(1)
        real(params, max_norm)
        return torch.tensor(float("nan")) if len(calls) == 1 else torch.tensor(1.0)

    monkeypatch.setattr(pretrain, "clip_grad_norm_", fake)
    torch.manual_seed(0)
    initial = {k: v.clone() for k, v in TinySQL(ModelConfig(**MODEL)).state_dict().items()}
    model, _ = pretrain.train(cfg_for(shards, tmp_path, max_steps=1, log_interval=1, warmup_steps=1))
    assert all(torch.equal(initial[k], v) for k, v in model.state_dict().items())          # weights untouched
    assert read_metrics(tmp_path)[0]["skipped_steps"] == 1


def test_param_groups_decay_only_matrices():
    model = TinySQL(ModelConfig(**MODEL))
    decay, no_decay = param_groups(model, 0.1)
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    in_group = lambda group, p: any(q is p for q in group["params"])          # identity, not tensor equality
    assert in_group(no_decay, model.embed.weight) and not in_group(decay, model.embed.weight)
    assert in_group(no_decay, model.final_norm.weight)
    assert in_group(decay, model.blocks[0].attn.q_proj.weight)


def test_loss_scaler():
    s = LossScaler(enabled=True, init_scale=1024.0, growth_interval=3)
    p = torch.nn.Parameter(torch.zeros(2))
    p.grad = torch.tensor([2048.0, 1024.0])
    s.unscale_([p])
    assert torch.equal(p.grad, torch.tensor([2.0, 1.0]))
    s.update(found_inf=True)
    assert s.scale_value == 512.0
    for _ in range(3):
        s.update(found_inf=False)
    assert s.scale_value == 1024.0
    off = LossScaler(enabled=False)
    assert off.scale(torch.tensor(3.0)) == 3.0


def test_checkpoints_keep_only_the_last_two(tmp_path):
    for step in (1, 2, 3, 4):
        save_checkpoint(str(tmp_path), step, {"step": step})
    assert sorted(os.listdir(tmp_path)) == ["ckpt_0000003.pt", "ckpt_0000004.pt"]


def test_override_parsing_handles_scientific_notation_without_a_dot():
    assert pretrain.parse_override("peak_lr=7e-4") == ("peak_lr", 0.0007)
    assert pretrain.parse_override("peak_lr=7.0e-4") == ("peak_lr", 0.0007)
    assert pretrain.parse_override("max_steps=50") == ("max_steps", 50)
    assert pretrain.parse_override("compile=true") == ("compile", True)
    assert pretrain.parse_override("out_dir=runs/x") == ("out_dir", "runs/x")
    assert pretrain.parse_override("sample_prompts=[]") == ("sample_prompts", [])
