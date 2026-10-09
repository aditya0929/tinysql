import os

import pytest
import torch

from data.wikisql_to_sql import BASE, make_example, to_sql
from eval.evaluate_sql import evaluate, generate_grouped, normalize_sql
from eval.execution_acc import WikiSQLExecutor
from model.config import ModelConfig
from model.model import TinySQL
from tests.test_wikisql_converter import make_db
from tokenizer.bpe import BPETokenizer
from train.finetune_sql import IGNORE, collate, encode_example, finetune

TOK_PATH = os.path.join("tokenizer", "tinysql_bpe.json")
pytestmark = pytest.mark.skipif(not os.path.exists(TOK_PATH), reason="trained tokenizer not present")

TABLE = {"header": ["Player", "No.", "School"], "types": ["text", "real", "text"]}
ITEM = {"table_id": "1-5", "question": "Which player went to butler cc (ks)?",
        "sql": {"sel": 0, "agg": 0, "conds": [[2, 0, "Butler CC (KS)"]]}}


def test_only_the_completion_is_trained_on():
    tok = BPETokenizer.load(TOK_PATH)
    ex = make_example(ITEM, TABLE)
    ids, labels = encode_example(tok, ex, max_len=1024)
    n_prompt = len(tok.encode(ex["prompt"], allow_special=True))
    assert all(l == IGNORE for l in labels[:n_prompt])                       # schema + question + <sql> are masked
    assert labels[n_prompt:] == ids[n_prompt:]                              # the SQL itself is trained on
    assert ids[-1] == tok.special_to_id["<|endoftext|>"]
    assert ids[-2] == tok.special_to_id["</sql>"] and ids.count(tok.special_to_id["</sql>"]) == 1
    assert ids[n_prompt - 1] == tok.special_to_id["<sql>"]
    assert tok.decode(ids[n_prompt:-2]) == to_sql(ITEM["sql"], TABLE["header"])


def test_truncation_keeps_inputs_and_labels_aligned():
    tok = BPETokenizer.load(TOK_PATH)
    ids, labels = encode_example(tok, make_example(ITEM, TABLE), max_len=20)
    assert len(ids) == len(labels) == 20


def test_collate_pads_and_shifts():
    batch = [([5, 6, 7, 8], [IGNORE, IGNORE, 7, 8]), ([5, 6], [IGNORE, 6])]
    x, y = collate(batch, pad_id=0)
    assert x.tolist() == [[5, 6, 7], [5, 6, 0]]
    assert y.tolist() == [[IGNORE, 7, 8], [6, IGNORE, IGNORE]]              # targets are the next tokens; padding is ignored


def test_grouped_generation_matches_one_at_a_time():
    torch.manual_seed(0)
    model = TinySQL(ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=64, max_seq_len=48)).eval()
    prompts = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10], [11, 12, 13, 14]]        # three of length 4, one of length 2
    grouped = generate_grouped(model, prompts, stop_id=63, max_new_tokens=6)
    single = [generate_grouped(model, [p], stop_id=63, max_new_tokens=6)[0] for p in prompts]
    assert grouped == single


def test_evaluate_scores_correct_wrong_and_invalid_predictions(tmp_path):
    tok = BPETokenizer.load(TOK_PATH)
    db, tables = make_db(tmp_path)
    ex = WikiSQLExecutor(db, tables)
    items = [ITEM] * 4
    t = tables["1-5"]
    gold = to_sql(ITEM["sql"], t["header"])
    answers = [gold,                                                            # correct, exact match
               'SELECT "Player" FROM t WHERE "School" = \'duke\'',              # runs but wrong rows
               'SELECT "Nope" FROM t',                                          # invalid (unknown column)
               'select "Player" from t where "School"=\'butler cc (ks)\'']      # right result, different text
    fake = lambda prompts, stop: [tok.encode(a) for a in answers]
    metrics, records = evaluate(None, tok, items, tables, ex, generator=fake)
    assert metrics["n"] == 4
    # the 4th answer is right by execution but not textually identical, so exact match is stricter than execution accuracy
    assert metrics["exec_acc"] == 0.5 and metrics["validity"] == 0.75 and metrics["exact_match"] == 0.25
    assert [r["correct"] for r in records] == [True, False, False, True]
    assert normalize_sql(gold) == normalize_sql(gold.upper().replace(" ", "  "))


@pytest.mark.skipif(not (os.path.exists(os.path.join("data", "wikisql_sft", "train.jsonl")) and os.path.exists(os.path.join(BASE, "dev.db"))),
                    reason="run python -m data.wikisql_to_sql first")
def test_finetune_runs_end_to_end_on_real_wikisql(tmp_path):
    cfg = dict(vocab_size=32768, d_model=32, n_layers=1, n_heads=4, n_kv_heads=2, d_ff=64, max_seq_len=1024)
    torch.manual_seed(0)
    base = TinySQL(ModelConfig(**cfg))
    ckpt = str(tmp_path / "base.pt")
    torch.save({"model": base.state_dict(), "cfg": {"model": cfg}}, ckpt)
    model, best = finetune(dict(base_ckpt=ckpt, out_dir=str(tmp_path / "ft"), epochs=1, batch_size=8, limit_train=64,
                                eval_examples=16, device="cpu", warmup_steps=2, peak_lr=1e-3))
    assert os.path.exists(tmp_path / "ft" / "best.pt") and 0.0 <= best <= 1.0
    saved = torch.load(tmp_path / "ft" / "best.pt", weights_only=False)
    assert set(saved["dev_metrics"]) >= {"exec_acc", "validity", "exact_match"}
