"""Train the BPE tokenizer on a sample of the cleaned corpus, with SQL deliberately over-represented.

    python -m tokenizer.train_tokenizer --scale 0.1          # quick run to measure time and memory
    python -m tokenizer.train_tokenizer                      # full run -> tokenizer/tinysql_bpe.json

Sampling is by random row group (a few thousand documents each) so the sample mixes all files of a source.
Counting chunk frequencies is parallel (one Counter per row-group batch); the merge loop is single-process.
"""
import argparse
import glob
import json
import os
import random
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import pyarrow.parquet as pq

from tokenizer.bpe import BPETokenizer

FINAL = os.path.join("data", "final")
WIKISQL = os.path.join("data", "raw", "wikisql", "extracted", "data")
VOCAB_SIZE = 32_768
# megabytes of text per source in the tokenizer-training sample (SQL-heavy on purpose)
SAMPLE_MB = {"fineweb_edu": 350, "stack_sql": 400, "stackexchange": 150, "gretel_sql": 40}


def row_groups(source: str):
    """All (file, row_group_index, n_rows) of a source."""
    out = []
    for path in sorted(glob.glob(os.path.join(FINAL, source, "*.parquet"))):
        pf = pq.ParquetFile(path)
        out += [(path, i) for i in range(pf.num_row_groups) if pf.metadata.row_group(i).num_rows > 0]
    return out


def read_groups(task):
    """Worker: read row groups, count chunks. Returns (Counter, characters)."""
    groups = task
    texts = []
    for path, i in groups:
        texts += pq.ParquetFile(path).read_row_group(i, columns=["text"]).column("text").to_pylist()
    return BPETokenizer.count_chunks(texts), sum(len(t) for t in texts)


def choose_groups(source: str, target_mb: float, rng: random.Random):
    """Pick random row groups until ~target_mb of text; also return one extra group held out for evaluation."""
    groups = row_groups(source)
    rng.shuffle(groups)
    chosen, size, i = [], 0.0, 0
    while i < len(groups) - 1 and size < target_mb * 1e6:
        path, g = groups[i]
        rows = pq.ParquetFile(path).metadata.row_group(g)
        size += rows.total_byte_size * 4                       # parquet bytes -> rough text bytes (zstd ~4x)
        chosen.append(groups[i])
        i += 1
    return chosen, groups[i]


def wikisql_train_texts():
    """Questions and table headers of WikiSQL TRAIN only (dev/test stay unseen)."""
    texts = []
    with open(os.path.join(WIKISQL, "train.jsonl"), encoding="utf-8") as f:
        texts += [json.loads(line)["question"] for line in f]
    with open(os.path.join(WIKISQL, "train.tables.jsonl"), encoding="utf-8") as f:
        texts += [" ".join(json.loads(line)["header"]) for line in f]
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0, help="multiply the per-source sample sizes")
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    ap.add_argument("--out", default=os.path.join("tokenizer", "tinysql_bpe.json"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    rng = random.Random(0)
    tasks, held_out, plan = [], {}, {}
    for source, mb in SAMPLE_MB.items():
        chosen, held = choose_groups(source, mb * args.scale, rng)
        held_out[source] = held
        plan[source] = len(chosen)
        step = max(1, len(chosen) // args.workers)              # a few row groups per task
        tasks += [chosen[i:i + step] for i in range(0, len(chosen), step)]
    print("row groups per source:", plan)

    t0 = time.time()
    counts, chars = Counter(), 0
    with ProcessPoolExecutor(args.workers) as pool:
        for c, n in pool.map(read_groups, tasks):
            counts.update(c)
            chars += n
    extra = BPETokenizer.count_chunks(wikisql_train_texts())
    counts.update(extra)
    print(f"counted {chars / 1e6:,.0f} MB of text in {time.time() - t0:.0f}s: "
          f"{len(counts):,} distinct chunks, {sum(1 for c in counts.values() if c >= args.min_count):,} with count >= {args.min_count}")

    t1 = time.time()
    tok = BPETokenizer.train_from_counts(counts, args.vocab_size, verbose=True, min_count=args.min_count)
    print(f"trained {tok.vocab_size:,} tokens ({len(tok.merges):,} merges) in {time.time() - t1:.0f}s")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tok.save(args.out)
    print("saved", args.out)

    print("\nbytes per token on held-out text (higher = better compression):")
    for source, (path, g) in held_out.items():
        texts = pq.ParquetFile(path).read_row_group(g, columns=["text"]).column("text").to_pylist()[:300]
        n_bytes = sum(len(t.encode("utf-8")) for t in texts)
        n_tok = sum(len(tok.encode(t, allow_special=False)) for t in texts)
        print(f"  {source:14s} {n_bytes / n_tok:5.2f}")
    print("\nhow SQL pieces tokenize:")
    for s in [" SELECT", " FROM", " WHERE", " GROUP BY", " ORDER BY", " customer_id", "COUNT(*)", " INNER JOIN", " CREATE TABLE"]:
        print(f"  {s!r:16s} -> {[tok.decode([i]) for i in tok.encode(s, allow_special=False)]}")


if __name__ == "__main__":
    main()
