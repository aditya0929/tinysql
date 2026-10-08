import random

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from data.clean import SCHEMA
from data.dedup import exact_dedup, minhash, near_dup_keep_mask


def make_doc(seed: int, n_words: int = 300) -> str:
    rng = random.Random(seed)
    vocab = [f"w{i}" for i in range(5000)]
    return " ".join(rng.choice(vocab) for _ in range(n_words))


def agreement(a, b):
    return float((a == b).mean())


def test_minhash_estimates_similarity():
    a = make_doc(1)
    words = a.split()
    words[150] = "CHANGED"                                  # one word edited -> ~5 of ~300 shingles differ
    b = " ".join(words)
    c = make_doc(2)
    assert agreement(minhash(a), minhash(a)) == 1.0
    assert agreement(minhash(a), minhash(b)) > 0.85
    assert agreement(minhash(a), minhash(c)) < 0.1


def test_near_dup_mask_keeps_the_longest_of_each_cluster():
    a = make_doc(1)
    words = a.split()
    words[10] = "EDIT"
    a_edit = " ".join(words) + " extra tail words appended here"      # longer near-copy of a
    docs = [a, make_doc(2), a_edit, make_doc(3)]
    sigs = np.stack([minhash(d) for d in docs])
    keep = near_dup_keep_mask(sigs, np.array([len(d) for d in docs]))
    assert keep.tolist() == [False, True, True, True]               # a dropped, its longer copy a_edit kept


def test_exact_dedup_across_files_and_sources(tmp_path):
    clean, out = tmp_path / "clean", tmp_path / "out"

    def write(source, name, texts):
        d = clean / source
        d.mkdir(parents=True, exist_ok=True)
        t = pa.table({"text": texts, "source": [source] * len(texts), "license": ["x"] * len(texts),
                      "origin": ["o"] * len(texts)}, schema=SCHEMA)
        pq.write_table(t, d / name)

    write("a", "f1.parquet", ["one", "two", "one"])             # duplicate inside a file
    write("a", "f2.parquet", ["two", "three"])                  # duplicate across files
    write("b", "f1.parquet", ["three", "four"])                 # duplicate across sources
    counts = exact_dedup(["a", "b"], str(clean), str(out))
    kept_a = pq.read_table(out / "a" / "f1.parquet").column("text").to_pylist() + \
        pq.read_table(out / "a" / "f2.parquet").column("text").to_pylist()
    kept_b = pq.read_table(out / "b" / "f1.parquet").column("text").to_pylist()
    assert kept_a == ["one", "two", "three"] and kept_b == ["four"]
    assert counts["a"]["docs_out"] == 3 and counts["b"]["docs_out"] == 1
