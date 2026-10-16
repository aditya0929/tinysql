"""The Hugging Face export must be exact: same token ids as the from-scratch tokenizer, same logits as the model."""
import random

import pytest
import torch

transformers = pytest.importorskip("transformers")

from model.config import ModelConfig
from model.model import TinySQL
from scripts.export_hf import export, to_hf_tokenizer, TOKENIZER
from tokenizer.bpe import BPETokenizer

SAMPLES = [
    "Hello world, it's 2026!",
    "SELECT COUNT(\"Player\") FROM t WHERE \"School/Club Team\" = 'Butler CC (KS)';",
    "<schema>CREATE TABLE t (\"No.\" TEXT);</schema>\n<question>How many?</question>\n<sql>",
    "naïve café — Ölçek 尙州 emoji 🙂 tabs\tand\n\n  newlines   ",
    "x = 123456789; y=3.14159   \r\n done",
    "INSERT INTO users VALUES (1, 'a@b.com', NULL);",
    "   leading spaces and trailing   ",
]


def _random_texts(n=300, seed=0):
    rng = random.Random(seed)
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJ0123456789 ,.;:'\"()*_-=<>/\n\tçé尙🙂"
    return ["".join(rng.choice(alphabet) for _ in range(rng.randint(1, 80))) for _ in range(n)]


@pytest.fixture(scope="module")
def toks():
    mine = BPETokenizer.load(TOKENIZER)
    return mine, to_hf_tokenizer(mine)


def test_tokenizer_ids_identical(toks):
    mine, hf = toks
    for text in SAMPLES + _random_texts():
        assert hf.encode(text).ids == mine.encode(text), text


def test_tokenizer_decode_identical(toks):
    mine, hf = toks
    for text in SAMPLES:
        ids = mine.encode(text)
        assert hf.decode(ids, skip_special_tokens=False) == mine.decode(ids) == text


def test_exported_model_has_identical_logits(tmp_path):
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=32768, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=128, max_seq_len=64)
    model = TinySQL(cfg).eval()
    ckpt = tmp_path / "tiny.pt"
    torch.save({"model": model.state_dict(), "cfg": {"model": cfg.__dict__}}, ckpt)
    export(str(ckpt), str(tmp_path / "hf"), sql=True)
    hf = transformers.AutoModelForCausalLM.from_pretrained(tmp_path / "hf", attn_implementation="eager").eval()
    ids = torch.randint(0, 32768, (2, 17))
    with torch.no_grad():
        mine = model(ids)[0].float()
        theirs = hf(input_ids=ids).logits.float()
    assert (mine - theirs).abs().max() < 1e-4
    tok = transformers.AutoTokenizer.from_pretrained(tmp_path / "hf")
    assert tok("<sql>SELECT").input_ids == BPETokenizer.load(TOKENIZER).encode("<sql>SELECT")
