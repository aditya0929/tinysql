"""Generate SQL for WikiSQL examples with a model and score it by execution.

Generation uses my KV-cache decoder, which has no padding mask, so prompts are grouped by their exact token length
and each group is decoded as one batch (hundreds of distinct lengths, so it is still fast).

Metrics: execution accuracy (primary), validity (the SQL ran), exact match (normalised text).
"""
import re
from collections import defaultdict

import torch

from data.wikisql_to_sql import prompt_text, to_sql
from eval.execution_acc import WikiSQLExecutor, execution_match
from model.generate import generate


def normalize_sql(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().rstrip(";")).lower()


@torch.no_grad()
def generate_grouped(model, prompts, stop_id: int, max_new_tokens: int = 160, max_batch: int = 64, device="cpu"):
    """prompts: list of token-id lists. Returns, for each, the generated tokens up to (not including) stop_id."""
    out = [None] * len(prompts)
    by_len = defaultdict(list)
    for i, p in enumerate(prompts):
        by_len[len(p)].append(i)
    model.eval()
    for length, idxs in by_len.items():
        for s in range(0, len(idxs), max_batch):
            chunk = idxs[s:s + max_batch]
            ids = torch.tensor([prompts[i] for i in chunk], dtype=torch.long, device=device)
            gen = generate(model, ids, max_new_tokens, temperature=0, stop_ids=[stop_id])[:, length:].tolist()
            for i, row in zip(chunk, gen):
                out[i] = row[:row.index(stop_id)] if stop_id in row else row
    return out


def evaluate(model, tok, items, tables, executor: WikiSQLExecutor, device="cpu", max_new_tokens=160,
             max_batch=64, generator=None):
    """items: WikiSQL dev/test rows. Returns metrics plus per-example records for failure analysis.
    `generator(prompts, stop_id)` can replace the model decoder (used by tests and by beam/constrained decoding)."""
    stop_id = tok.special_to_id["</sql>"]
    prompts = [tok.encode(prompt_text(it["question"], tables[it["table_id"]]["header"], tables[it["table_id"]]["types"]),
                          allow_special=True) for it in items]
    if generator is None:
        gens = generate_grouped(model, prompts, stop_id, max_new_tokens, max_batch, device)
    else:
        gens = generator(prompts, stop_id)
    records, n_valid, n_correct, n_exact = [], 0, 0, 0
    for it, g in zip(items, gens):
        t = tables[it["table_id"]]
        gold = to_sql(it["sql"], t["header"])
        pred = tok.decode(g).strip()
        valid, correct = execution_match(executor, it["table_id"], gold, pred)
        exact = normalize_sql(pred) == normalize_sql(gold)
        n_valid, n_correct, n_exact = n_valid + valid, n_correct + correct, n_exact + exact
        records.append({"table_id": it["table_id"], "question": it["question"], "gold": gold, "pred": pred,
                        "valid": valid, "correct": correct, "exact": exact})
    n = max(1, len(items))
    return {"n": len(items), "exec_acc": n_correct / n, "validity": n_valid / n, "exact_match": n_exact / n}, records
