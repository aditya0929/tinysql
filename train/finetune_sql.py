"""Fine-tune a pretrained TinySQL checkpoint on WikiSQL (text-to-SQL), scoring execution accuracy on dev every epoch.

    python -m data.wikisql_to_sql                                   # once: builds data/wikisql_sft/*.jsonl
    python -m train.finetune_sql --config configs/finetune_wikisql.yaml --set base_ckpt=runs/main/ckpt_0008600.pt

Loss is computed only on the completion (the SQL, the closing </sql> and <|endoftext|>); prompt tokens are masked
with -100. The checkpoint with the best dev execution accuracy is kept, not the one with the lowest loss.
"""
import argparse
import contextlib
import copy
import json
import os
import random
import time

import torch
import yaml

from data.wikisql_to_sql import BASE, load_split
from eval.evaluate_sql import evaluate
from eval.execution_acc import WikiSQLExecutor
from model.config import ModelConfig
from model.model import TinySQL
from optim.adamw import AdamW, clip_grad_norm_
from optim.schedule import cosine_lr
from tokenizer.bpe import BPETokenizer
from train.pretrain import parse_override
from train.utils import MetricsLog, param_groups

IGNORE = -100
DEFAULTS = dict(
    base_ckpt=None, tokenizer=os.path.join("tokenizer", "tinysql_bpe.json"), sft_dir=os.path.join("data", "wikisql_sft"),
    out_dir="runs/finetune_wikisql", seed=0, max_len=1024,
    epochs=3, batch_size=32, grad_accum=1, peak_lr=1e-4, min_lr_ratio=0.1, warmup_steps=100,
    weight_decay=0.0, betas=[0.9, 0.95], eps=1e-8, grad_clip=1.0,
    eval_examples=1000, eval_batch=64, device="auto", precision="auto", limit_train=None,
)


def encode_example(tok: BPETokenizer, ex: dict, max_len: int):
    """-> (ids, labels). labels are IGNORE on the prompt, the real token ids on the completion."""
    p = tok.encode(ex["prompt"], allow_special=True)
    c = tok.encode(ex["completion"], allow_special=True) + [tok.special_to_id["<|endoftext|>"]]
    ids = (p + c)[:max_len]
    labels = ([IGNORE] * len(p) + c)[:max_len]
    return ids, labels


def collate(batch, pad_id: int):
    """Pad to the longest sequence in the batch; returns model inputs and shifted targets."""
    n = max(len(ids) for ids, _ in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), n), IGNORE, dtype=torch.long)
    for i, (a, b) in enumerate(batch):
        ids[i, :len(a)] = torch.tensor(a)
        labels[i, :len(b)] = torch.tensor(b)
    return ids[:, :-1], labels[:, 1:]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def finetune(user_cfg: dict):
    cfg = copy.deepcopy(DEFAULTS)
    cfg.update(user_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if cfg["device"] == "auto" else torch.device(cfg["device"])
    use_bf16 = device.type == "cuda" and (cfg["precision"] in ("auto", "bf16"))
    ctx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_bf16 else contextlib.nullcontext
    torch.manual_seed(cfg["seed"])
    rng = random.Random(cfg["seed"])

    tok = BPETokenizer.load(cfg["tokenizer"])
    pad_id = tok.special_to_id["<|pad|>"]
    state = torch.load(cfg["base_ckpt"], map_location="cpu", weights_only=False)
    model = TinySQL(ModelConfig(**state["cfg"]["model"]))
    model.load_state_dict(state["model"])
    model.to(device)

    train_rows = load_jsonl(os.path.join(cfg["sft_dir"], "train.jsonl"))
    if cfg["limit_train"]:
        train_rows = train_rows[:cfg["limit_train"]]
    train = [encode_example(tok, ex, cfg["max_len"]) for ex in train_rows]
    dev_items, dev_tables = load_split("dev")
    dev_items = dev_items[:cfg["eval_examples"]] if cfg["eval_examples"] else dev_items
    executor = WikiSQLExecutor(os.path.join(BASE, "dev.db"), dev_tables)

    steps_per_epoch = max(1, len(train) // (cfg["batch_size"] * cfg["grad_accum"]))
    max_steps = steps_per_epoch * cfg["epochs"]
    optim = AdamW(param_groups(model, cfg["weight_decay"]), lr=cfg["peak_lr"], betas=tuple(cfg["betas"]), eps=cfg["eps"])
    params = [p for g in optim.param_groups for p in g["params"]]
    log = MetricsLog(cfg["out_dir"])
    print(f"{model.num_params():,} parameters | {len(train):,} training examples | {steps_per_epoch} steps/epoch x {cfg['epochs']} "
          f"| device {device}", flush=True)

    best, step, t0 = -1.0, 0, time.time()
    for epoch in range(cfg["epochs"]):
        order = list(range(len(train)))
        rng.shuffle(order)
        model.train()
        for s in range(steps_per_epoch):
            lr = cosine_lr(step, max_steps, cfg["peak_lr"], cfg["warmup_steps"], cfg["min_lr_ratio"])
            for g in optim.param_groups:
                g["lr"] = lr
            optim.zero_grad()
            loss_sum = 0.0
            for a in range(cfg["grad_accum"]):
                lo = (s * cfg["grad_accum"] + a) * cfg["batch_size"]
                x, y = collate([train[i] for i in order[lo:lo + cfg["batch_size"]]], pad_id)
                with ctx():
                    _, loss, _ = model(x.to(device), targets=y.to(device))
                (loss / cfg["grad_accum"]).backward()
                loss_sum += loss.item() / cfg["grad_accum"]
            norm = clip_grad_norm_(params, cfg["grad_clip"])
            if torch.isfinite(norm):
                optim.step()
            step += 1
            if step % 25 == 0 or step == 1:
                log.log(step, loss=loss_sum, lr=lr, grad_norm=float(norm), epoch=epoch)
                print(f"epoch {epoch} step {step:5d}/{max_steps} | loss {loss_sum:.4f} | lr {lr:.2e} | |g| {float(norm):.2f}", flush=True)
        metrics, _ = evaluate(model, tok, dev_items, dev_tables, executor, device=device, max_batch=cfg["eval_batch"])
        log.log(step, epoch=epoch, **{f"dev_{k}": v for k, v in metrics.items() if k != "n"})
        print(f"== epoch {epoch}: dev exec acc {metrics['exec_acc']:.1%} | validity {metrics['validity']:.1%} | "
              f"exact match {metrics['exact_match']:.1%} ({time.time() - t0:.0f}s)", flush=True)
        if metrics["exec_acc"] > best:
            best = metrics["exec_acc"]
            os.makedirs(cfg["out_dir"], exist_ok=True)
            torch.save({"model": model.state_dict(), "cfg": state["cfg"], "dev_metrics": metrics, "epoch": epoch},
                       os.path.join(cfg["out_dir"], "best.pt"))
    return model, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config)) if args.config else {}
    for kv in args.set:
        k, v = parse_override(kv)
        cfg[k] = v
    finetune(cfg)


if __name__ == "__main__":
    main()
