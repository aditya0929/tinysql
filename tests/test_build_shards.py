import glob
from collections import Counter
import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data import build_shards
from data.build_shards import keep_sql_doc, query_score
from data.clean import SCHEMA
from data.loader import TokenLoader
from tokenizer.bpe import BPETokenizer

QUERY = "SELECT name, COUNT(*) FROM employees WHERE salary > 5 GROUP BY department ORDER BY 2 LIMIT 3;"
DDL = "CREATE TABLE employees (id INT, name TEXT, salary INT);"


@pytest.fixture(scope="module")
def tok_path(tmp_path_factory):
    tok = BPETokenizer.train([QUERY * 20, DDL * 20, "the quick brown fox jumps over the lazy dog " * 20], 256 + 60 + 12)
    path = str(tmp_path_factory.mktemp("tok") / "tok.json")
    tok.save(path)
    return path


def write_source(final_dir, source, texts):
    d = os.path.join(final_dir, source)
    os.makedirs(d, exist_ok=True)
    pq.write_table(pa.table({"text": texts, "source": [source] * len(texts), "license": ["x"] * len(texts),
                             "origin": ["o"] * len(texts)}, schema=SCHEMA), os.path.join(d, "part0.parquet"))


def test_query_score_prefers_real_queries():
    assert query_score(QUERY) > query_score(DDL) >= 0
    assert query_score("just prose") == 0


def test_sql_selection_keeps_the_requested_fraction_and_the_best_files():
    docs = [QUERY + f" -- {i}" for i in range(300)] + [DDL + f" -- {i}" for i in range(700)]
    s_star, tie_p = build_shards.choose_threshold(Counter(query_score(d) for d in docs), 0.5)
    kept = [d for d in docs if keep_sql_doc(d, s_star, tie_p)]
    assert abs(len(kept) - 500) < 60
    assert sum(d.startswith("SELECT") for d in kept) == 300       # every real query file is kept before any DDL file


def test_build_and_load(tmp_path, tok_path):
    final, shards = str(tmp_path / "final"), str(tmp_path / "shards")
    write_source(final, "fineweb_edu", [f"the quick brown fox {i} jumps over the lazy dog <sql> text" for i in range(400)])
    write_source(final, "stack_sql", [QUERY + f" -- {i}" for i in range(100)] + [DDL + f" -- {i}" for i in range(100)])
    write_source(final, "stackexchange", [f"question {i}: how do I write a query? " * 3 for i in range(50)])
    write_source(final, "gretel_sql", ["<schema>CREATE TABLE t (a INT);</schema>\n<question>q?</question>\n<sql>SELECT 1;</sql>"] * 40)
    summary = build_shards.run(final, shards, tok_path, sql_keep=0.5, gretel_repeat=3, val_rate=0.1, workers=2)
    tok = BPETokenizer.load(tok_path)
    eos = tok.special_to_id["<|endoftext|>"]

    # stack_sql: half of the 200 files kept (train + val + skipped add up), real queries preferred
    s = summary["stack_sql"]
    assert s["docs_skipped"] == 100 and s["docs_train"] + s["docs_val"] == 100
    # gretel: each training example appears 3 times
    g = summary["gretel_sql"]
    assert g["docs_train"] > 0 and g["tokens_train"] % 3 == 0
    # validation is a small, separate slice
    assert 0 < summary["fineweb_edu"]["docs_val"] < 0.3 * 400
    assert summary["fineweb_edu"]["docs_train"] + summary["fineweb_edu"]["docs_val"] == 400

    # control-token safety: literal "<sql>" in web text is plain text; in our Gretel examples it is a special id
    sql_id = tok.special_to_id["<sql>"]
    web = np.concatenate([np.fromfile(p, dtype=np.uint16) for p in glob.glob(os.path.join(shards, "train", "fineweb_edu__*.bin"))])
    gre = np.concatenate([np.fromfile(p, dtype=np.uint16) for p in glob.glob(os.path.join(shards, "train", "gretel_sql__*.bin"))])
    assert sql_id not in web and sql_id in gre
    assert (web == eos).sum() == summary["fineweb_edu"]["docs_train"]            # one EOS per document

    # loader: shapes, shift-by-one targets, determinism, exact resume
    loader = TokenLoader("train", seq_len=16, shards_dir=shards, seed=3)
    x, y = loader.get_batch(8)
    assert x.shape == y.shape == (8, 16) and int(x.max()) < tok.vocab_size
    assert torch_equal_shift(x, y)
    a = TokenLoader("train", 16, shards_dir=shards, seed=3).get_batch(8)[0]
    assert (a == x).all()
    state = loader.state_dict()
    nxt = loader.get_batch(4)[0]
    loader.load_state_dict(state)
    assert (loader.get_batch(4)[0] == nxt).all()
    assert TokenLoader("val", 16, shards_dir=shards, source="fineweb_edu").num_tokens == summary["fineweb_edu"]["tokens_val"]


def torch_equal_shift(x, y):
    return bool((x[:, 1:] == y[:, :-1]).all())
