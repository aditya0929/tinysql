import random

import pytest

from tokenizer.bpe import BPETokenizer, merge_ids

SQL = "SELECT name, COUNT(*) FROM employees WHERE salary > 50000 GROUP BY department ORDER BY 2 DESC;\n"
CORPUS = [SQL * 30, "How many orders were placed in March 2024? " * 30, "the quick brown fox jumps over the lazy dog. " * 30]


@pytest.fixture(scope="module")
def tok():
    return BPETokenizer.train(CORPUS, vocab_size=256 + 150 + 12)


def test_merge_ids():
    assert merge_ids([1, 2, 3, 1, 2], (1, 2), 9) == [9, 3, 9]
    assert merge_ids([5, 5, 5], (5, 5), 9) == [9, 5]          # left to right, non-overlapping


def test_first_merge_is_the_most_frequent_pair():
    t = BPETokenizer.train(["aaabdaaabac"], vocab_size=256 + 3 + 12)
    assert t.merges[0] == (97, 97)                             # 'aa' occurs 4 times, more than any other pair
    assert t.vocab_size == 256 + 3 + 12


@pytest.mark.parametrize("text", [
    "", "a", "hello world", "  leading and trailing  ", "tabs\tand\nnewlines\n\n",
    "naïve café", "日本語のテキスト", "emoji 🙂🚀 mix", SQL, "x" * 500, "12345678901234567890",
])
def test_round_trip(tok, text):
    assert tok.decode(tok.encode(text)) == text


def test_round_trip_random_unicode(tok):
    rng = random.Random(0)
    for _ in range(200):
        s = "".join(chr(rng.choice([rng.randint(32, 126), rng.randint(0x80, 0x2FFF), rng.randint(0x1F300, 0x1F64F)]))
                    for _ in range(rng.randint(1, 40)))
        assert tok.decode(tok.encode(s)) == s


def test_training_compresses_seen_text(tok):
    n_bytes = len(SQL.encode())
    assert len(tok.encode(SQL)) < n_bytes / 2


def test_frequent_word_becomes_one_token():
    t = BPETokenizer.train(["hello hello hello hello world world"] * 20, vocab_size=256 + 20 + 12)
    assert len(t.encode(" hello")) == 1


def test_special_tokens_are_single_ids(tok):
    ids = tok.encode("<sql>SELECT 1</sql>")
    assert ids[0] == tok.special_to_id["<sql>"] and ids[-1] == tok.special_to_id["</sql>"]
    assert tok.decode(ids) == "<sql>SELECT 1</sql>"
    assert len(tok.encode("<sql>", allow_special=False)) > 1   # as plain text it is split into pieces
    assert tok.decode(tok.encode("<sql>", allow_special=False)) == "<sql>"


def test_special_tokens_have_fixed_ids_at_the_end(tok):
    assert tok.special_to_id["<|endoftext|>"] == tok.vocab_size - len(tok.special_tokens)
    assert max(tok.special_to_id.values()) == tok.vocab_size - 1


def test_training_is_deterministic():
    a = BPETokenizer.train(CORPUS, vocab_size=256 + 80 + 12)
    b = BPETokenizer.train(CORPUS, vocab_size=256 + 80 + 12)
    assert a.merges == b.merges


def test_save_and_load(tok, tmp_path):
    path = str(tmp_path / "tok.json")
    tok.save(path)
    t2 = BPETokenizer.load(path)
    assert t2.merges == tok.merges and t2.encode(SQL) == tok.encode(SQL)


def test_decode_of_arbitrary_ids_does_not_crash(tok):
    tok.decode([200, 201, 5, 300])                              # may contain a broken UTF-8 sequence
