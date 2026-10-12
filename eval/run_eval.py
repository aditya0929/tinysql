"""Score a checkpoint on a WikiSQL split and save metrics plus every prediction (for failure analysis).

    python -m eval.run_eval --ckpt runs/finetune_wikisql/best.pt --split dev --out runs/finetune_wikisql
"""
import argparse
import json
import os
import time

import torch

from data.wikisql_to_sql import BASE, load_split
from decode.constrained import make_constrained_generator
from eval.evaluate_sql import evaluate
from eval.execution_acc import WikiSQLExecutor
from model.config import ModelConfig
from model.model import TinySQL
from tokenizer.bpe import BPETokenizer


def load_model(path: str, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = TinySQL(ModelConfig(**state["cfg"]["model"]))
    model.load_state_dict(state["model"])
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="runs/eval")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--tokenizer", default=os.path.join("tokenizer", "tinysql_bpe.json"))
    ap.add_argument("--decode", default="greedy", choices=["greedy", "schema", "values"],
                    help="greedy: unconstrained; schema: grammar + real columns only; values: also values copied from the question")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = BPETokenizer.load(args.tokenizer)
    model = load_model(args.ckpt, device)
    items, tables = load_split(args.split)
    items = items[:args.limit] if args.limit else items
    executor = WikiSQLExecutor(os.path.join(BASE, f"{args.split}.db"), tables)

    t0 = time.time()
    generator = None if args.decode == "greedy" else make_constrained_generator(
        model, tok, items, tables, values_from_question=args.decode == "values", device=device, max_batch=args.batch)
    metrics, records = evaluate(model, tok, items, tables, executor, device=device, max_batch=args.batch, generator=generator)
    metrics["decode"] = args.decode
    metrics["seconds"] = round(time.time() - t0, 1)
    metrics["ckpt"] = args.ckpt
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{args.split}_{args.decode}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    with open(os.path.join(args.out, f"{args.split}_{args.decode}_predictions.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{args.split} [{args.decode}]: {metrics['n']:,} examples | execution accuracy {metrics['exec_acc']:.2%} | "
          f"validity {metrics['validity']:.2%} | exact match {metrics['exact_match']:.2%} | {metrics['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
