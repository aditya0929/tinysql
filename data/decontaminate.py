"""Remove pretraining documents that contain WikiSQL dev/test questions: data/dedup -> data/final

A document is dropped if it contains, as a contiguous run of word tokens (lowercased, punctuation ignored):
  - any 13-gram of a question that has >= 13 tokens, or
  - the WHOLE question when it has 8..12 tokens (a 13-gram would never fit).
Questions with < 8 tokens are skipped: they are generic phrases and would flag unrelated documents.

    python -m data.decontaminate [--splits dev test]
"""
import argparse
import glob
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from data.dedup import DEDUP, _TOKEN, _token_hash

FINAL = os.path.join("data", "final")
WIKISQL = os.path.join("data", "raw", "wikisql", "extracted", "data")
MAX_N, MIN_N = 13, 8
_PRIME = np.uint64(0x100000001B3)


def token_hashes(text: str) -> np.ndarray:
    toks = _TOKEN.findall(text.lower())
    return np.fromiter((_token_hash(t) for t in toks), dtype=np.uint64, count=len(toks))


def ngram_hashes(th: np.ndarray, max_n: int = MAX_N):
    """Yield (n, hashes of all n-grams) for n = 2..max_n, built incrementally with a rolling polynomial hash."""
    h = th.copy()
    for n in range(2, max_n + 1):
        m = len(th) - n + 1
        if m < 1:
            return
        h = h[:m] * _PRIME + th[n - 1:n - 1 + m]               # uint64 arithmetic wraps on purpose
        yield n, h


def build_eval_index(questions):
    """-> ({n: sorted unique uint64 hashes}, {(n, hash): question}, number of skipped short questions)"""
    per_n, lookup, skipped = defaultdict(list), {}, 0
    for q in questions:
        th = token_hashes(q)
        if len(th) < MIN_N:
            skipped += 1
            continue
        wanted = MAX_N if len(th) >= MAX_N else len(th)         # long: every 13-gram; short: the whole question
        for n, h in ngram_hashes(th, wanted):
            if n == wanted:
                for x in h:
                    per_n[n].append(x)
                    lookup.setdefault((n, int(x)), q)
    return {n: np.unique(np.array(v, dtype=np.uint64)) for n, v in per_n.items()}, lookup, skipped


def find_contamination(text: str, index, lookup):
    """Returns the matched eval question, or None."""
    th = token_hashes(text)
    for n, h in ngram_hashes(th):
        arr = index.get(n)
        if arr is None:
            continue
        pos = np.searchsorted(arr, h)
        pos[pos == len(arr)] = 0
        hit = np.flatnonzero(arr[pos] == h)
        if len(hit):
            return lookup[(n, int(h[hit[0]]))]
    return None


_STATE = {}


def _init(index, lookup):
    _STATE["index"], _STATE["lookup"] = index, lookup


def decontaminate_file(path):
    source = os.path.basename(os.path.dirname(path))
    out_dir = os.path.join(FINAL, source)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, os.path.basename(path))
    stats = {"source": source, "docs_in": 0, "removed": 0, "examples": []}
    with pq.ParquetWriter(out, pq.ParquetFile(path).schema_arrow, compression="zstd") as w:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=5000):
            keep = []
            for text, origin in zip(batch.column("text").to_pylist(), batch.column("origin").to_pylist()):
                q = find_contamination(text, _STATE["index"], _STATE["lookup"])
                keep.append(q is None)
                if q is not None and len(stats["examples"]) < 3:
                    stats["examples"].append({"origin": origin, "question": q})
            stats["docs_in"] += len(keep)
            stats["removed"] += keep.count(False)
            w.write_table(pa.Table.from_batches([batch]).filter(pa.array(keep)))
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="*", default=["dev", "test"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    questions = []
    for s in args.splits:
        with open(os.path.join(WIKISQL, f"{s}.jsonl"), encoding="utf-8") as f:
            questions += [json.loads(line)["question"] for line in f]
    index, lookup, skipped = build_eval_index(questions)
    print(f"{len(questions):,} eval questions ({', '.join(args.splits)}); skipped {skipped:,} with < {MIN_N} tokens; "
          f"{sum(len(v) for v in index.values()):,} n-gram hashes")

    paths = sorted(glob.glob(os.path.join(DEDUP, "*", "*.parquet")))
    t0, results = time.time(), []
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(index, lookup)) as pool:
        results = list(pool.map(decontaminate_file, paths))
    print(f"scanned {len(paths)} files in {time.time() - t0:.0f}s\n")

    report = {}
    for r in results:
        s = report.setdefault(r["source"], {"docs_in": 0, "removed": 0, "examples": []})
        s["docs_in"] += r["docs_in"]
        s["removed"] += r["removed"]
        s["examples"] += r["examples"][: max(0, 3 - len(s["examples"]))]
    print(f"{'source':14s} {'docs in':>11s} {'removed':>9s}")
    for s, r in report.items():
        print(f"{s:14s} {r['docs_in']:>11,} {r['removed']:>9,}")
        for e in r["examples"]:
            print(f"    e.g. {e['origin'][:60]!r} contained: {e['question'][:90]!r}")
    with open(os.path.join(FINAL, "report.json"), "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
