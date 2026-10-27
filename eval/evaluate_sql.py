"""Generate SQL for WikiSQL examples with a model and score it by execution."""
import re
from collections import defaultdict

import torch

from data.wikisql_to_sql import prompt_text, to_sql
from eval.execution_acc import WikiSQLExecutor, execution_match
from model.generate import generate


def normalize_sql(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().rstrip(";")).lower()


@torch.no_grad()
def generate_grouped(model, prompts, stop_id, max_new_tokens: int = 160, max_batch: int = 64, device="cpu"):
    """prompts: list of token-id lists. stop_id: one id or a list of ids.
    Returns, for each prompt, the generated tokens up to (not including) the first stop id."""
    stops = [stop_id] if isinstance(stop_id, int) else list(stop_id)
    out = [None] * len(prompts)
    by_len = defaultdict(list)
    for i, p in enumerate(prompts):
        by_len[len(p)].append(i)
    model.eval()
    for length, idxs in by_len.items():
        for s in range(0, len(idxs), max_batch):
            chunk = idxs[s:s + max_batch]
            ids = torch.tensor([prompts[i] for i in chunk], dtype=torch.long, device=device)
            gen = generate(model, ids, max_new_tokens, temperature=0, stop_ids=stops)[:, length:].tolist()
            for i, row in zip(chunk, gen):
                cut = next((j for j, t in enumerate(row) if t in stops), len(row))
                out[i] = row[:cut]
    return out


def evaluate(model, tok, items, tables, executor: WikiSQLExecutor, device="cpu", max_new_tokens=160,
             max_batch=64, generator=None):
    """items: WikiSQL dev/test rows. Returns metrics plus per-example records for failure analysis.
    `generator(prompts, stop_id)` can replace the model decoder (used by tests and by beam/constrained decoding)."""
    stop_id = [tok.special_to_id["</sql>"], tok.special_to_id["<|endoftext|>"]]   # whichever the model emits first
    prompts = [tok.encode(prompt_text(it["question"], tables[it["table_id"]]["header"], tables[it["table_id"]]["types"]),
                          allow_special=True) for it in items]
    if generator is None:
        gens = generate_grouped(model, prompts, stop_id, max_new_tokens, max_batch, device)
    else:
        gens = generator(prompts, stop_id)
    return score_predictions(items, tables, executor, [tok.decode(g).strip() for g in gens])


def score_predictions(items, tables, executor: WikiSQLExecutor, preds):
    """Score SQL strings from ANY model (ours or a baseline) the same way. preds[i] answers items[i]."""
    records, n_valid, n_correct, n_exact = [], 0, 0, 0
    for it, pred in zip(items, preds):
        t = tables[it["table_id"]]
        gold = to_sql(it["sql"], t["header"])
        pred = pred.strip()
        valid, correct = execution_match(executor, it["table_id"], gold, pred)
        exact = normalize_sql(pred) == normalize_sql(gold)
        n_valid, n_correct, n_exact = n_valid + valid, n_correct + correct, n_exact + exact
        records.append({"table_id": it["table_id"], "question": it["question"], "gold": gold, "pred": pred,
                        "valid": valid, "correct": correct, "exact": exact})
    n = max(1, len(items))
    return {"n": len(items), "exec_acc": n_correct / n, "validity": n_valid / n, "exact_match": n_exact / n}, records
