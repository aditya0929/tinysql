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


def test_values_must_be_valid_utf8():
    q = "Who managed Atlético?"
    gram = SqlGrammar(COLS, q)
    head = b'SELECT "Player" FROM t WHERE "Player" = \'Atl'
    assert gram.check(head + "é".encode()[:1]) == "prefix"            # first byte of a 2-byte character: wait for the rest
    assert gram.check(head + b"\x80") == "invalid"                      # a stray continuation byte can never be valid
    assert gram.check(head + "é".encode()[:1] + b"'") == "invalid"      # cannot close a value mid-character
    assert SqlGrammar(COLS, None).check(head + b"\xff") == "invalid"    # the same rule without the question constraint


def test_unfinished_character_must_lead_somewhere_in_the_question():
    head = b'SELECT "Player" FROM t WHERE "Player" = \'H'
    assert SqlGrammar(COLS, "who played for haugar").check(head + "é".encode()[:1]) == "invalid"   # no é after 'h' in question: dead end
    assert SqlGrammar(COLS, "who played for hé fc").check(head + "é".encode()[:1]) == "prefix"     # 'hé' is in the question
    assert SqlGrammar(COLS, "who played for hÉ fc").check(head + "é".encode()) == "prefix"         # case-insensitive


def test_output_rows_without_a_token_are_skipped():
    """HF models (e.g. Pythia) pad the output layer past the tokenizer: those ids must never be chosen."""
    from types import SimpleNamespace
    from decode.constrained import _choose
    tok = SimpleNamespace(vocab={0: b"SELECT", 1: b" x"}, id_to_special={2: "</sql>"})
    logits = torch.tensor([0.0, 1.0, 2.0, 9.0, 8.0])                   # ids 3 and 4 have no token and score highest
    assert _choose(logits, b"", SqlGrammar(["Player"]), tok, stop_id=2, topk=2) == 0
