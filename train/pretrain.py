"""Pretraining loop.

    python -m train.pretrain --config configs/debug_tiny.yaml
    python -m train.pretrain --config configs/tinysql_125m.yaml        # picks up the latest checkpoint automatically
"""
import argparse
import contextlib
import copy
import math
import os
import time

import torch
import yaml

from data.loader import DEFAULT_SCHEDULE, SOURCES, MixedLoader, TokenLoader
from model.config import ModelConfig
from model.generate import generate
from model.model import TinySQL
from optim.adamw import AdamW, clip_grad_norm_
from optim.schedule import cosine_lr
from train.utils import LossScaler, MetricsLog, latest_checkpoint, param_groups, save_checkpoint

DEFAULTS = dict(
    model={}, shards_dir=os.path.join("data", "shards"), out_dir="runs/default", seed=0,
    seq_len=2048, micro_batch=8, grad_accum=32,                     # 8 x 32 x 2048 = 0.5M tokens per step
    max_steps=10_000, warmup_steps=200, peak_lr=1e-3, min_lr_ratio=0.1,
    weight_decay=0.1, betas=[0.9, 0.95], eps=1e-8, grad_clip=1.0,
    precision="auto", device="auto", compile=False,
    log_interval=10, eval_interval=500, eval_batches=20, ckpt_interval=500,
    peak_tflops=None,                                               # for the MFU readout, e.g. 312 for A100 bf16
    tokenizer=os.path.join("tokenizer", "tinysql_bpe.json"),
    sample_prompts=["The capital of France is",
                    "<schema>CREATE TABLE employees (id INT, name TEXT, salary INT);</schema>\n"
                    "<question>How many employees are there?</question>\n<sql>"],
    sample_tokens=40, tensorboard=False, stop_at_step=None,          # stop_at_step: exit early without changing the schedule
    mix=None,                                                        # None = DEFAULT_SCHEDULE
)


def resolve_precision(cfg, device):
    p = cfg["precision"]
    if p == "auto":
        if device.type == "cuda":
            p = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        else:
            p = "fp32"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[p]
    ctx = (lambda: torch.autocast(device.type, dtype=dtype)) if p != "fp32" else contextlib.nullcontext
    return p, ctx


@torch.no_grad()
def evaluate(model, cfg, device, ctx, sources_present):
    """Validation loss per source on a fixed set of batches (same batches every time, so values are comparable)."""
    model.eval()
    out = {}
    for s in sources_present:
        ld = TokenLoader("val", cfg["seq_len"], cfg["shards_dir"], source=s, seed=123)
        losses = []
        for _ in range(cfg["eval_batches"]):
            x, y = ld.get_batch(cfg["micro_batch"])
            with ctx():
                _, loss, _ = model(x.to(device), targets=y.to(device))
            losses.append(loss.item())
        out[s] = sum(losses) / len(losses)
    model.train()
    return out


@torch.no_grad()
def sample_text(model, cfg, device):
    if not cfg["sample_prompts"] or not os.path.exists(cfg["tokenizer"]):
        return []
    from tokenizer.bpe import BPETokenizer
    tok = BPETokenizer.load(cfg["tokenizer"])
    model.eval()
    texts = []
    for prompt in cfg["sample_prompts"]:
        ids = torch.tensor([tok.encode(prompt)], device=device)
        out = generate(model, ids, cfg["sample_tokens"], temperature=0)
        texts.append(tok.decode(out[0].tolist()))
    model.train()
    return texts


def train(user_cfg: dict):
    cfg = copy.deepcopy(DEFAULTS)
    cfg.update(user_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if cfg["device"] == "auto" else torch.device(cfg["device"])
    precision, ctx = resolve_precision(cfg, device)
    torch.manual_seed(cfg["seed"])

    model = TinySQL(ModelConfig(**cfg["model"])).to(device)
    n_params = model.num_params()
    optim = AdamW(param_groups(model, cfg["weight_decay"]), lr=cfg["peak_lr"], betas=tuple(cfg["betas"]), eps=cfg["eps"])
    schedule = [(u, w) for u, w in (cfg["mix"] or DEFAULT_SCHEDULE)]
    loader = MixedLoader("train", cfg["seq_len"], schedule, cfg["shards_dir"], seed=cfg["seed"])
    scaler = LossScaler(enabled=precision == "fp16")
    log = MetricsLog(cfg["out_dir"], cfg["tensorboard"])
    run_model = torch.compile(model) if cfg["compile"] else model

    step = 0
    ckpt = latest_checkpoint(cfg["out_dir"])
    if ckpt:                                                         # resume: weights, optimizer, data position, RNG
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optim.load_state_dict(state["optim"])
        loader.load_state_dict(state["loader"])
        scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["torch_rng"].cpu())
        step = state["step"]
        print(f"resumed from {ckpt} at step {step}")

    tokens_per_step = cfg["micro_batch"] * cfg["grad_accum"] * cfg["seq_len"]
    print(f"{n_params:,} parameters | device {device} | precision {precision} | {tokens_per_step:,} tokens/step | "
          f"{cfg['max_steps']} steps = {tokens_per_step * cfg['max_steps'] / 1e9:.2f}B tokens")
    params = [p for g in optim.param_groups for p in g["params"]]
    stop_at = cfg["stop_at_step"] or cfg["max_steps"]
    present = [s for s in SOURCES if os.path.exists(cfg["shards_dir"]) and any(
        f.startswith(s + "__") for f in os.listdir(os.path.join(cfg["shards_dir"], "val")))]
    skipped, last_t, tokens_seen = 0, time.time(), step * tokens_per_step

    def checkpoint():
        save_checkpoint(cfg["out_dir"], step, {
            "model": model.state_dict(), "optim": optim.state_dict(), "loader": loader.state_dict(),
            "scaler": scaler.state_dict(), "torch_rng": torch.get_rng_state(), "step": step, "cfg": cfg})

    model.train()
    while step < stop_at:
        lr = cosine_lr(step, cfg["max_steps"], cfg["peak_lr"], cfg["warmup_steps"], cfg["min_lr_ratio"])
        for g in optim.param_groups:
            g["lr"] = lr
        optim.zero_grad()
        loss_sum = 0.0
        for _ in range(cfg["grad_accum"]):
            x, y = loader.get_batch(cfg["micro_batch"], progress=step / cfg["max_steps"])
            with ctx():
                _, loss, _ = run_model(x.to(device), targets=y.to(device))
            scaler.scale(loss / cfg["grad_accum"]).backward()
            loss_sum += loss.item() / cfg["grad_accum"]
        scaler.unscale_(params)
        grad_norm = clip_grad_norm_(params, cfg["grad_clip"])
        if torch.isfinite(grad_norm) and math.isfinite(loss_sum):
            optim.step()
            scaler.update(False)
        else:                                                        # inf/nan: skip this update instead of poisoning the weights
            optim.zero_grad()
            scaler.update(True)
            skipped += 1
        step += 1
        tokens_seen += tokens_per_step

        if step % cfg["log_interval"] == 0 or step == 1:
            now = time.time()
            tps = tokens_per_step * (cfg["log_interval"] if step > 1 else 1) / (now - last_t)
            last_t = now
            mfu = 6 * n_params * tps / (cfg["peak_tflops"] * 1e12) if cfg["peak_tflops"] else None
            mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None
            log.log(step, loss=loss_sum, lr=lr, grad_norm=float(grad_norm), tokens_per_s=tps, tokens=tokens_seen,
                    skipped_steps=skipped, **({"mfu": mfu} if mfu else {}), **({"peak_mem_gb": mem} if mem else {}))
            print(f"step {step:6d} | loss {loss_sum:.4f} | lr {lr:.2e} | |g| {float(grad_norm):.2f} | {tps:,.0f} tok/s"
                  + (f" | MFU {mfu:.1%}" if mfu else "") + (f" | {mem:.1f} GB peak" if mem else "")
                  + (f" | skipped {skipped}" if skipped else ""), flush=True)
        if step % cfg["eval_interval"] == 0 or step == cfg["max_steps"]:
            val = evaluate(model, cfg, device, ctx, present)
            log.log(step, **{f"val_loss/{k}": v for k, v in val.items()})
            print(f"  val loss: " + ", ".join(f"{k} {v:.3f}" for k, v in val.items()), flush=True)
            for text in sample_text(model, cfg, device):
                print("  sample:", repr(text[:200]), flush=True)
        if step % cfg["ckpt_interval"] == 0 or step == stop_at:
            checkpoint()
    return model, step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. max_steps=50 out_dir=runs/x")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    train(cfg)


if __name__ == "__main__":
    main()
