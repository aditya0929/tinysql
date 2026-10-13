"""Greedy full-split evaluation of a fine-tuned Hugging Face model directory (same prompts, decoding and scorer as
eval/baselines/finetune_hf.py --full_dev, but for a model that is already trained).

    python -m eval.baselines.eval_hf --model_dir runs/bench/gpt2_lr0.0001_s0/best --out runs/bench/gpt2_lr0.0001_s0
"""
import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.wikisql_to_sql import BASE, load_split
from eval.baselines.finetune_hf import generate_sql
from eval.evaluate_sql import score_predictions
from eval.execution_acc import WikiSQLExecutor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="dev")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model_dir, dtype=torch.float32).to(device)
    items, tables = load_split(a.split)
    ex = WikiSQLExecutor(os.path.join(BASE, f"{a.split}.db"), tables)
    t0 = time.time()
    metrics, records = score_predictions(items, tables, ex, generate_sql(model, tok, items, tables, device))
    metrics.update({"model_dir": a.model_dir, "decode": "greedy", "seconds": round(time.time() - t0, 1)})
    os.makedirs(a.out, exist_ok=True)
    json.dump(metrics, open(os.path.join(a.out, f"{a.split}_greedy_metrics.json"), "w"), indent=1)
    with open(os.path.join(a.out, f"{a.split}_greedy_predictions.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{a.split} [greedy] {a.model_dir}: execution accuracy {metrics['exec_acc']:.2%} | validity {metrics['validity']:.2%}", flush=True)


if __name__ == "__main__":
    main()
