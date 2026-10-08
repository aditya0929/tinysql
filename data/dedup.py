"""Remove duplicate documents: data/clean/<source>/*.parquet -> data/dedup/<source>/*.parquet

Step 1, exact: a document is dropped if its text hash was already seen (across all sources).
Step 2, near (MinHash + LSH), for sources listed in --near (default: stack_sql, where GitHub forks and
        copy-pasted scripts are everywhere):
          shingle   every run of 5 word tokens -> a 32-bit hash
          MinHash   128 hash functions; for each, keep the minimum over all shingles of the document.
                    The fraction of equal slots between two signatures estimates their Jaccard similarity.
          LSH       split the 128 slots into 16 bands of 8; documents sharing an entire band are candidates.
                    P(candidate) = 1 - (1 - s^8)^16, which is ~0.5 at s = 0.7 and ~1 at s >= 0.85.
          verify    a candidate pair is merged only if >= 80% of its slots agree.
          keep      the longest document of each cluster.

    python -m data.dedup
"""
import argparse
import glob
import hashlib
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import xxhash

from data.clean import CLEAN, SCHEMA

DEDUP = os.path.join("data", "dedup")
NUM_PERM, BANDS, ROWS = 128, 16, 8
SHINGLE = 5
VERIFY_THRESHOLD = 0.8
_rng = np.random.default_rng(1234)
_A = _rng.integers(1, 2**32, NUM_PERM, dtype=np.uint64) | np.uint64(1)
_B = _rng.integers(0, 2**32, NUM_PERM, dtype=np.uint64)
_MASK = np.uint64(0xFFFFFFFF)
_TOKEN = re.compile(r"\w+")
_token_hash_cache: dict = {}


# ----------------------------------------------------------------------------- step 1: exact
def text_hash(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "little")


def exact_dedup(sources, clean_dir=CLEAN, out_dir=DEDUP):
    """Stream all files in a fixed order, keeping the first occurrence of each text. Returns per-source counts."""
    seen: set = set()
    counts = {}
    for source in sources:
        c = counts[source] = {"docs_in": 0, "docs_out": 0, "chars_out": 0}
        os.makedirs(os.path.join(out_dir, source), exist_ok=True)
        for path in sorted(glob.glob(os.path.join(clean_dir, source, "*.parquet"))):
            with pq.ParquetWriter(os.path.join(out_dir, source, os.path.basename(path)), SCHEMA, compression="zstd") as w:
                for batch in pq.ParquetFile(path).iter_batches(batch_size=5000):
                    texts = batch.column("text").to_pylist()
                    keep = []
                    for t in texts:
                        h = text_hash(t)
                        keep.append(h not in seen)
                        seen.add(h)
                    c["docs_in"] += len(texts)
                    c["docs_out"] += sum(keep)
                    c["chars_out"] += sum(len(t) for t, k in zip(texts, keep) if k)
                    w.write_table(pa.Table.from_batches([batch]).filter(pa.array(keep)))
    return counts


# ----------------------------------------------------------------------------- step 2: near-duplicates
def _token_hash(tok: str) -> int:
    h = _token_hash_cache.get(tok)
    if h is None:
        h = _token_hash_cache[tok] = xxhash.xxh32_intdigest(tok.encode("utf-8"))
    return h


def shingle_hashes(text: str) -> np.ndarray:
    """Unique 32-bit hashes of every run of SHINGLE consecutive word tokens."""
    toks = _TOKEN.findall(text.lower()) or [text]
    t = np.fromiter((_token_hash(x) for x in toks), dtype=np.uint64, count=len(toks))
    n = len(t) - SHINGLE + 1
    if n < 1:
        n, h = 1, np.array([int(t.sum()) & 0xFFFFFFFF], dtype=np.uint64)
    else:
        h = t[:n].copy()
        for j in range(1, SHINGLE):
            h = h * np.uint64(0x100000001B3) + t[j:n + j]            # uint64 arithmetic wraps on purpose
    return np.unique((h ^ (h >> np.uint64(32))) & _MASK)


def minhash(text: str) -> np.ndarray:
    """(NUM_PERM,) uint32 signature."""
    s = shingle_hashes(text)
    sig = np.full(NUM_PERM, 0xFFFFFFFF, dtype=np.uint64)
    for i in range(0, len(s), 4096):                                  # chunked to bound memory
        chunk = s[i:i + 4096]
        part = ((_A[:, None] * chunk[None, :] + _B[:, None]) & _MASK).min(axis=1)
        sig = np.minimum(sig, part)
    return sig.astype(np.uint32)


def file_signatures(path: str):
    texts = pq.read_table(path, columns=["text"]).column("text").to_pylist()
    sigs = np.empty((len(texts), NUM_PERM), dtype=np.uint32)
    for i, t in enumerate(texts):
        sigs[i] = minhash(t)
    return sigs, np.array([len(t) for t in texts], dtype=np.int64)


def near_dup_keep_mask(sigs: np.ndarray, lengths: np.ndarray, threshold: float = VERIFY_THRESHOLD) -> np.ndarray:
    """Cluster near-duplicates with LSH and return a boolean mask that keeps the longest doc of each cluster."""
    n = len(sigs)
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for b in range(BANDS):
        band = np.ascontiguousarray(sigs[:, b * ROWS:(b + 1) * ROWS])
        keys = band.view(np.dtype((np.void, band.dtype.itemsize * ROWS))).ravel()
        _, inverse = np.unique(keys, return_inverse=True)
        order = np.argsort(inverse, kind="stable")
        sorted_ids = inverse[order]
        starts = np.flatnonzero(np.r_[True, sorted_ids[1:] != sorted_ids[:-1]])
        ends = np.r_[starts[1:], n]
        for s, e in zip(starts, ends):
            if e - s < 2:
                continue
            members = order[s:e]
            first = members[0]
            for other in members[1:]:
                if (sigs[first] == sigs[other]).mean() >= threshold:
                    ra, rb = find(first), find(other)
                    if ra != rb:
                        parent[rb] = ra
    roots = np.array([find(i) for i in range(n)])
    keep = np.zeros(n, dtype=bool)
    best: dict = {}
    for i in range(n):
        r = roots[i]
        if r not in best or lengths[i] > lengths[best[r]]:
            best[r] = i
    keep[list(best.values())] = True
    return keep


def near_dedup(source: str, workers: int, out_dir=DEDUP):
    paths = sorted(glob.glob(os.path.join(out_dir, source, "*.parquet")))
    with ProcessPoolExecutor(workers) as pool:
        results = list(pool.map(file_signatures, paths))
    sigs = np.concatenate([r[0] for r in results])
    lengths = np.concatenate([r[1] for r in results])
    keep = near_dup_keep_mask(sigs, lengths)
    offset, docs_out, chars_out = 0, 0, 0
    for path, (s, _) in zip(paths, results):
        table = pq.read_table(path)
        mask = keep[offset:offset + len(s)]
        offset += len(s)
        kept = table.filter(pa.array(mask))
        pq.write_table(kept, path + ".tmp", compression="zstd")
        os.replace(path + ".tmp", path)
        docs_out += kept.num_rows
        chars_out += int(lengths[offset - len(s):offset][mask].sum())
    return {"docs_in": len(keep), "docs_out": docs_out, "chars_out": chars_out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--near", nargs="*", default=["stack_sql"], help="sources that also get near-duplicate removal")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    sources = sorted(d for d in os.listdir(CLEAN) if os.path.isdir(os.path.join(CLEAN, d)))

    t0 = time.time()
    exact = exact_dedup(sources)
    print(f"exact dedup done in {time.time() - t0:.0f}s")
    report = {s: {"clean_docs": c["docs_in"], "after_exact": c["docs_out"]} for s, c in exact.items()}
    for s in args.near:
        t1 = time.time()
        r = near_dedup(s, args.workers)
        report[s]["after_near"] = r["docs_out"]
        report[s]["chars_out"] = r["chars_out"]
        print(f"near dedup {s}: {r['docs_in']:,} -> {r['docs_out']:,} in {time.time() - t1:.0f}s")
    for s in sources:
        report[s].setdefault("after_near", report[s]["after_exact"])
        report[s].setdefault("chars_out", exact[s]["chars_out"])

    print(f"\n{'source':14s} {'clean':>10s} {'exact':>10s} {'final':>10s} {'removed':>8s} {'MB':>8s}")
    for s in sources:
        r = report[s]
        print(f"{s:14s} {r['clean_docs']:>10,} {r['after_exact']:>10,} {r['after_near']:>10,} "
              f"{1 - r['after_near'] / r['clean_docs']:>8.1%} {r['chars_out'] / 1e6:>8,.0f}")
    with open(os.path.join(DEDUP, "report.json"), "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
