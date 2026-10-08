"""Tokenize data/final into packed uint16 token shards: data/shards/{train,val}/<source>__<file>__<part>.bin

The mix is set here:
  fineweb_edu, stackexchange   every document
  stack_sql                    the --sql-keep fraction (default 0.7) of files, chosen by "query score":
                               files with real SELECT/WHERE/JOIN/GROUP BY queries are preferred over pure DDL or data
  gretel_sql                   every example, repeated --gretel-repeat times in train (they are tiny and match our prompt format)

Every document is followed by <|endoftext|>. Special-token text is NOT interpreted in web, code or Q&A documents
(allow_special=False), so crawled text containing "<sql>" cannot inject a control token; only the Gretel examples,
which we formatted ourselves, may produce the <schema>/<question>/<sql> special tokens.

A deterministic slice of documents (hash of the text, --val-rate, default 0.4%) goes to the validation split.

    python -m data.build_shards [--limit-groups N]       # --limit-groups: smoke test with N row groups per file
"""
import argparse
import functools
import glob
import json
import os
import re
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow.parquet as pq
import xxhash

from tokenizer.bpe import BPETokenizer

FINAL = os.path.join("data", "final")
SHARDS = os.path.join("data", "shards")
TOKENIZER = os.path.join("tokenizer", "tinysql_bpe.json")
ROW_GROUPS_PER_TASK = 25
SOURCES = ["fineweb_edu", "stack_sql", "stackexchange", "gretel_sql"]

_CLAUSES = [re.compile(p, re.I) for p in (
    r"\bselect\b", r"\bfrom\b", r"\bwhere\b", r"\bjoin\b", r"\bgroup\s+by\b", r"\border\s+by\b",
    r"\bhaving\b", r"\b(?:count|sum|avg|min|max)\s*\(", r"\blimit\b", r"\bunion\b", r"\bwith\s+\w+\s+as\s*\(")]


def query_score(text: str) -> int:
    """0..11: how many distinct query features the file uses."""
    return sum(1 for rx in _CLAUSES if rx.search(text))


def doc_hash(text: str) -> int:
    return xxhash.xxh64_intdigest(text.encode("utf-8"))


@functools.lru_cache(maxsize=2)
def load_tokenizer(path: str) -> BPETokenizer:
    return BPETokenizer.load(path)


# ----------------------------------------------------------------------------- choosing the SQL files
def score_histogram(path: str) -> Counter:
    texts = pq.read_table(path, columns=["text"]).column("text").to_pylist()
    return Counter(query_score(t) for t in texts)


def choose_threshold(hist: Counter, keep_frac: float):
    """-> (s_star, tie_keep_prob): keep docs scoring above s_star, plus a hash-chosen share of those at s_star,
    so that ~keep_frac of all documents are kept."""
    need, kept_above = keep_frac * sum(hist.values()), 0
    for s in range(max(hist), -1, -1):
        if kept_above + hist.get(s, 0) >= need:
            return s, (need - kept_above) / max(1, hist.get(s, 0))
        kept_above += hist.get(s, 0)
    return 0, 1.0


def sql_threshold(paths, keep_frac: float, workers: int):
    hist = Counter()
    with ProcessPoolExecutor(workers) as pool:
        for h in pool.map(score_histogram, paths):
            hist.update(h)
    return (*choose_threshold(hist, keep_frac), hist)


def keep_sql_doc(text: str, s_star: int, tie_p: float) -> bool:
    s = query_score(text)
    if s != s_star:
        return s > s_star
    return ((doc_hash(text) >> 20) % 10_000) < tie_p * 10_000


# ----------------------------------------------------------------------------- one task = a range of row groups
def build_task(task):
    source, path, g0, g1, cfg = task
    name = f"{source}__{os.path.splitext(os.path.basename(path))[0]}__{g0:04d}"
    paths = {s: os.path.join(cfg["shards_dir"], s, name + ".bin") for s in ("train", "val")}
    meta_path = os.path.join(cfg["shards_dir"], "meta", name + ".json")
    if os.path.exists(meta_path) and not cfg["limit_groups"]:
        with open(meta_path) as f:
            return json.load(f)                                  # already built: resume support
    for p in list(paths.values()) + [meta_path]:
        os.makedirs(os.path.dirname(p), exist_ok=True)
    tok = load_tokenizer(cfg["tokenizer"])
    eos = tok.special_to_id["<|endoftext|>"]
    allow_special = source == "gretel_sql"                       # only our own formatted examples may use control tokens
    stats = Counter()
    pf = pq.ParquetFile(path)
    with open(paths["train"], "wb") as f_train, open(paths["val"], "wb") as f_val:
        for g in range(g0, g1):
            texts = pf.read_row_group(g, columns=["text"]).column("text").to_pylist()
            train_chunks, val_chunks = [], []
            for text in texts:
                if source == "stack_sql" and not keep_sql_doc(text, cfg["s_star"], cfg["tie_p"]):
                    stats["docs_skipped"] += 1
                    continue
                ids = tok.encode(text, allow_special=allow_special)
                ids.append(eos)
                arr = np.asarray(ids, dtype=np.uint16)
                if doc_hash(text) % 10_000 < cfg["val_rate"] * 10_000:
                    val_chunks.append(arr)
                    stats["docs_val"] += 1
                    stats["tokens_val"] += len(arr)
                else:
                    reps = cfg["gretel_repeat"] if source == "gretel_sql" else 1
                    train_chunks += [arr] * reps
                    stats["docs_train"] += 1
                    stats["tokens_train"] += len(arr) * reps
            if train_chunks:
                np.concatenate(train_chunks).tofile(f_train)
            if val_chunks:
                np.concatenate(val_chunks).tofile(f_val)
    stats = {"name": name, "source": source, **{k: int(v) for k, v in stats.items()}}
    if not cfg["limit_groups"]:
        with open(meta_path, "w") as f:
            json.dump(stats, f)
    return stats


def make_tasks(final_dir, cfg, workers):
    sql_paths = sorted(glob.glob(os.path.join(final_dir, "stack_sql", "*.parquet")))
    if sql_paths:
        cfg["s_star"], cfg["tie_p"], hist = sql_threshold(sql_paths, cfg["sql_keep"], workers)
        print(f"stack_sql query-score histogram: {dict(sorted(hist.items()))}")
        print(f"keeping files with score > {cfg['s_star']}, plus {cfg['tie_p']:.0%} of those scoring exactly {cfg['s_star']}")
    tasks = []
    for source in SOURCES:
        for path in sorted(glob.glob(os.path.join(final_dir, source, "*.parquet"))):
            n = pq.ParquetFile(path).num_row_groups
            if cfg["limit_groups"]:
                n = min(n, cfg["limit_groups"])
            tasks += [(source, path, g, min(g + ROW_GROUPS_PER_TASK, n), cfg) for g in range(0, n, ROW_GROUPS_PER_TASK)]
    return tasks


def run(final_dir=FINAL, shards_dir=SHARDS, tokenizer=TOKENIZER, sql_keep=0.7, gretel_repeat=3,
        val_rate=0.004, workers=None, limit_groups=None):
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    cfg = dict(shards_dir=shards_dir, tokenizer=tokenizer, sql_keep=sql_keep, gretel_repeat=gretel_repeat,
               val_rate=val_rate, limit_groups=limit_groups, s_star=-1, tie_p=1.0)
    tasks = make_tasks(final_dir, cfg, workers)
    print(f"{len(tasks)} tasks on {workers} workers")
    t0, results = time.time(), []
    with ProcessPoolExecutor(workers) as pool:
        for r in pool.map(build_task, tasks):
            results.append(r)
    print(f"built in {time.time() - t0:.0f}s\n")

    summary = {}
    for src in SOURCES:
        rows = [r for r in results if r["source"] == src]
        summary[src] = {k: sum(r.get(k, 0) for r in rows) for k in ("docs_train", "docs_val", "docs_skipped", "tokens_train", "tokens_val")}
    total = sum(s["tokens_train"] for s in summary.values())
    print(f"{'source':14s} {'docs (train)':>13s} {'tokens train':>14s} {'share':>7s} {'tokens val':>12s}")
    for src, s in summary.items():
        print(f"{src:14s} {s['docs_train']:>13,} {s['tokens_train']:>14,} {s['tokens_train'] / max(1, total):>7.1%} {s['tokens_val']:>12,}")
    print(f"{'total':14s} {'':>13s} {total:>14,} {'':>7s} {sum(s['tokens_val'] for s in summary.values()):>12,}")
    if not limit_groups:
        with open(os.path.join(shards_dir, "summary.json"), "w") as f:
            json.dump({"cfg": {k: v for k, v in cfg.items() if k != "tokenizer"}, "sources": summary, "total_train_tokens": total}, f, indent=1)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sql-keep", type=float, default=0.7)
    ap.add_argument("--gretel-repeat", type=int, default=3)
    ap.add_argument("--val-rate", type=float, default=0.004)
    ap.add_argument("--limit-groups", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    run(sql_keep=a.sql_keep, gretel_repeat=a.gretel_repeat, val_rate=a.val_rate, workers=a.workers, limit_groups=a.limit_groups)


if __name__ == "__main__":
    main()
