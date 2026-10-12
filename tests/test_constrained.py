import os

import pytest
import torch

from data.wikisql_to_sql import BASE, column_names, load_split, to_sql
from decode.constrained import constrained_generate
from decode.sql_grammar import SqlGrammar
from model.config import ModelConfig
from model.model import TinySQL

COLS = ["Player", "No.", "School/Club Team"]
Q = "Which player wears no. 42 and went to O'Neil Prep?"


def g(question=Q):
    return SqlGrammar(COLS, question)


def test_complete_queries():
    assert g().check(b'SELECT "Player" FROM t;') == "complete"
    assert g().check(b'SELECT COUNT("Player") FROM t WHERE "No." = \'42\';') == "complete"
    assert g().check(b'SELECT "Player" FROM t WHERE "No." = \'42\' AND "School/Club Team" = \'o\'\'neil prep\';') == "complete"


def test_prefixes_are_accepted():
    full = b'SELECT MAX("No.") FROM t WHERE "Player" = \'which\';'
    for n in range(len(full)):
        assert g().check(full[:n]) == "prefix", full[:n]


def test_invalid_queries_are_rejected():
    assert g().check(b'SELECT "Nope"') == "invalid"                         # hallucinated column
    assert g().check(b'SELECT "Player" FROM users') == "invalid"            # wrong table
    assert g().check(b'SELECT "Player" FROM t WHERE "No." != ') == "invalid"   # unsupported operator
    assert g().check(b'SELECT "Player" FROM t WHERE "No." = \'99') == "invalid"    # 99 is not in the question
    assert g().check(b'SELECT "Player" FROM t; DROP') == "invalid"          # nothing may follow ';'
    assert g().check(b'SELECT "Player" FROM t WHERE "No." = \'\';') == "invalid"   # empty value
    assert SqlGrammar(COLS, None).check(b'SELECT "Player" FROM t WHERE "No." = \'99\';') == "complete"   # values unconstrained


def test_value_check_is_case_insensitive_and_handles_escaped_quotes():
    assert g().check(b'SELECT "Player" FROM t WHERE "School/Club Team" = \'O\'\'NEIL PREP\';') == "complete"
    assert g().check(b'SELECT "Player" FROM t WHERE "School/Club Team" = \'O\'') == "prefix"   # closing or escape


def test_non_ascii_split_across_tokens():
    q = "Who managed Atlético?"
    full = 'SELECT "Player" FROM t WHERE "Player" = \'Atlético\';'.encode("utf-8")
    for n in range(len(full)):                                           # includes cuts inside the 2-byte "é"
        assert SqlGrammar(COLS, q).check(full[:n]) == "prefix"
    assert SqlGrammar(COLS, q).check(full) == "complete"


@pytest.mark.skipif(not os.path.exists(os.path.join(BASE, "dev.jsonl")), reason="WikiSQL data not downloaded")
def test_every_gold_dev_query_is_accepted_by_the_strictest_grammar():
    """The grammar must never block a correct answer: all 8,421 gold queries are complete, and their prefixes valid."""
    items, tables = load_split("dev")
    for n, it in enumerate(items):
        header = tables[it["table_id"]]["header"]
        gold = to_sql(it["sql"], header).encode("utf-8")
        gram = SqlGrammar(column_names(header), it["question"])
        assert gram.check(gold) == "complete", (it["question"], gold)
        if n < 300:
            for cut in range(0, len(gold), 3):
                assert gram.check(gold[:cut]) == "prefix", (gold, cut)


@pytest.mark.skipif(not os.path.exists(os.path.join("tokenizer", "tinysql_bpe.json")), reason="trained tokenizer not present")
def test_even_a_random_model_produces_only_valid_sql():
    from tokenizer.bpe import BPETokenizer
    tok = BPETokenizer.load(os.path.join("tokenizer", "tinysql_bpe.json"))
    torch.manual_seed(0)
    model = TinySQL(ModelConfig(vocab_size=32768, d_model=32, n_layers=1, n_heads=4, n_kv_heads=2, d_ff=64, max_seq_len=512)).eval()
    prompts = [tok.encode("<schema>x</schema>\n<question>q</question>\n<sql>", allow_special=True)] * 3
    grammars = [SqlGrammar(COLS, Q), SqlGrammar(COLS, Q), SqlGrammar(COLS, None)]
    outs = constrained_generate(model, tok, prompts, grammars, max_new_tokens=300)
    for out, gram in zip(outs, grammars):
        text = b"".join(tok.vocab[t] for t in out)
        assert gram.check(text) == "complete", text
