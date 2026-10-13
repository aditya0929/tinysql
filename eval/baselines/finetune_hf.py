"""Baseline: fine-tune a professionally pretrained Hugging Face model with EXACTLY our WikiSQL recipe.

    python -m eval.baselines.finetune_hf --model HuggingFaceTB/SmolLM2-135M --out runs/baseline_smollm2_135m

Same as train/finetune_sql.py in every choice that matters: identical prompt text, loss only on the SQL completion,
batch 32, my AdamW (betas 0.9/0.95, no weight decay), peak LR 1e-4 with 100 warmup steps and cosine decay to 10%,
gradient clipping 1.0, bf16 autocast, 3 epochs, best epoch chosen by execution accuracy on the first 1,000 dev
questions, then the best epoch is scored on the full dev set with the same executor. The only differences are the
model, its tokenizer, and that generation uses Hugging Face's generate (greedy).
"""
import argparse
import json
import os
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.wikisql_to_sql import BASE, load_split, prompt_text
from eval.evaluate_sql import score_predictions
from eval.execution_acc import WikiSQLExecutor
from optim.adamw import AdamW, clip_grad_norm_
from optim.schedule import cosine_lr

IGNORE = -100


def encode(tok, ex, max_len=1024):
    p = tok(ex["prompt"], add_special_tokens=False)["input_ids"]
    c = tok(ex["completion"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    ids, labels = (p + c)[:max_len], ([IGNORE] * len(p) + c)[:max_len]
    return ids, labels


def collate(batch, pad_id):
    n = max(len(i) for i, _ in batch)
    ids = torch.full((len(batch), n), pad_id)
    labels = torch.full((len(batch), n), IGNORE)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    for k, (i, l) in enumerate(batch):
        ids[k, :len(i)], labels[k, :len(l)], mask[k, :len(i)] = torch.tensor(i), torch.tensor(l), 1
    return ids, labels, mask                       # HF shifts labels internally


def accumulate_backward(model, batch, micro_batch, pad_id, device) -> float:
    """Backward pass for one optimiser batch, done in micro-batches to fit in memory, with exactly the gradient of the
    whole batch: each micro-batch's mean loss is weighted by its share of the batch's trained tokens (HF shifts labels
    by one, so a sequence's trained tokens are its labels[1:] that are not IGNORE). Returns the batch's mean loss."""
    chunks = [batch[k:k + micro_batch] for k in range(0, len(batch), micro_batch)]
    counts = [sum(sum(1 for t in l[1:] if t != IGNORE) for _, l in c) for c in chunks]
    total = max(1, sum(counts))
    loss_val = 0.0
    for chunk, cnt in zip(chunks, counts):
        ids, labels, mask = collate(chunk, pad_id)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            part = model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=labels.to(device)).loss
        (part * cnt / total).backward()
        loss_val += part.item() * cnt / total
    return loss_val


@torch.no_grad()
def generate_sql(model, tok, items, tables, device, batch=32, max_new_tokens=160):
    """Greedy, left-padded batches; the answer is the text before '</sql>'."""
    model.eval()
    tok.padding_side = "left"
    prompts = [prompt_text(it["question"], tables[it["table_id"]]["header"], tables[it["table_id"]]["types"]) for it in items]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))          # similar lengths -> less padding
    preds = [None] * len(prompts)
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id,
                                 eos_token_id=tok.eos_token_id, stop_strings=["</sql>"], tokenizer=tok)
        for i, row in zip(idx, out[:, enc["input_ids"].shape[1]:]):
            preds[i] = tok.decode(row, skip_special_tokens=True).split("</sql>")[0]
    model.train()
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sft_dir", default=os.path.join("data", "wikisql_sft"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--micro_batch", type=int, default=8, help="memory only: the optimiser still sees batches of --batch_size")
    ap.add_argument("--peak_lr", type=float, default=1e-4)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--eval_examples", type=int, default=1000)
    ap.add_argument("--limit_train", type=int, default=None)
    ap.add_argument("--full_dev", action="store_true", help="after training, score the best epoch on all dev questions")
    ap.add_argument("--seed", type=int, default=0, help="controls the order of the training data")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    rows = [json.loads(l) for l in open(os.path.join(args.sft_dir, "train.jsonl"), encoding="utf-8")]
    rows = rows[:args.limit_train] if args.limit_train else rows
    train = [encode(tok, r) for r in rows]
    dev_items, dev_tables = load_split("dev")
    executor = WikiSQLExecutor(os.path.join(BASE, "dev.db"), dev_tables)
    sub = dev_items[:args.eval_examples]

    steps_per_epoch = max(1, len(train) // args.batch_size)
    max_steps = steps_per_epoch * args.epochs
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "embed" not in n]
    no_decay = [p for n, p in model.named_parameters() if not (p.dim() >= 2 and "embed" not in n)]
    optim = AdamW([{"params": decay, "weight_decay": 0.0}, {"params": no_decay, "weight_decay": 0.0}], lr=args.peak_lr, betas=(0.9, 0.95))
    params = decay + no_decay
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "metrics.jsonl"), "a")
    print(f"{args.model}: {n_params:,} parameters | {len(train):,} examples | {steps_per_epoch} steps/epoch x {args.epochs}", flush=True)

    best, step, t0 = -1.0, 0, time.time()
    for epoch in range(args.epochs):
        order = list(range(len(train)))
        rng.shuffle(order)
        model.train()
        for s in range(steps_per_epoch):
            lr = cosine_lr(step, max_steps, args.peak_lr, args.warmup_steps, 0.1)
            for g in optim.param_groups:
                g["lr"] = lr
            batch = [train[i] for i in order[s * args.batch_size:(s + 1) * args.batch_size]]
            optim.zero_grad()
            loss = torch.tensor(accumulate_backward(model, batch, args.micro_batch, tok.pad_token_id, device))
            norm = clip_grad_norm_(params, 1.0)
            if torch.isfinite(norm):
                optim.step()
            step += 1
            if step % 25 == 0 or step == 1:
                print(f"epoch {epoch} step {step:5d}/{max_steps} | loss {loss.item():.4f} | lr {lr:.2e} | |g| {float(norm):.2f}", flush=True)
                log.write(json.dumps({"step": step, "loss": loss.item(), "lr": lr, "epoch": epoch}) + "\n")
        m, _ = score_predictions(sub, dev_tables, executor, generate_sql(model, tok, sub, dev_tables, device))
        log.write(json.dumps({"step": step, "epoch": epoch, **{f"dev_{k}": v for k, v in m.items() if k != "n"}}) + "\n")
        log.flush()
        print(f"== epoch {epoch}: dev exec acc {m['exec_acc']:.1%} | validity {m['validity']:.1%} | exact match {m['exact_match']:.1%} ({time.time() - t0:.0f}s)", flush=True)
        if m["exec_acc"] > best:
            best = m["exec_acc"]
            model.save_pretrained(os.path.join(args.out, "best"))
            tok.save_pretrained(os.path.join(args.out, "best"))

    if args.full_dev:
        model = AutoModelForCausalLM.from_pretrained(os.path.join(args.out, "best"), dtype=torch.float32).to(device)
        t1 = time.time()
        metrics, records = score_predictions(dev_items, dev_tables, executor, generate_sql(model, tok, dev_items, dev_tables, device))
        metrics.update({"model": args.model, "params": n_params, "seconds": round(time.time() - t1, 1)})
        json.dump(metrics, open(os.path.join(args.out, "dev_greedy_metrics.json"), "w"), indent=1)
        with open(os.path.join(args.out, "dev_greedy_predictions.jsonl"), "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"dev [greedy]: {metrics['n']:,} examples | execution accuracy {metrics['exec_acc']:.2%} | validity {metrics['validity']:.2%} | "
              f"exact match {metrics['exact_match']:.2%}", flush=True)


if __name__ == "__main__":
    main()
