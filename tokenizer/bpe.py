"""Byte-level BPE tokenizer, written from scratch.

ids 0..255                      the 256 raw bytes
ids 256..256+n_merges-1         learned merges (id 256+r is the r-th merge)
last len(special_tokens) ids    special tokens such as <|endoftext|>, <sql>
"""
import heapq
import json
from collections import Counter, defaultdict

import regex

# Splits text into word-like chunks so merges never cross word / number / whitespace boundaries.
GPT4_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}"""
    r"""| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)

DEFAULT_SPECIAL_TOKENS = [
    "<|endoftext|>", "<|pad|>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>",
    "<schema>", "</schema>", "<question>", "</question>", "<sql>", "</sql>",
]


def merge_ids(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every (left-to-right, non-overlapping) occurrence of `pair` in ids by new_id."""
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, merges, special_tokens=None, pattern: str = GPT4_PATTERN):
        self.merges = [tuple(m) for m in merges]
        self.special_tokens = list(DEFAULT_SPECIAL_TOKENS if special_tokens is None else special_tokens)
        self.pattern = pattern
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for rank, (a, b) in enumerate(self.merges):
            self.vocab[256 + rank] = self.vocab[a] + self.vocab[b]
        first_special = 256 + len(self.merges)
        self.special_to_id = {tok: first_special + i for i, tok in enumerate(self.special_tokens)}
        self.id_to_special = {i: tok for tok, i in self.special_to_id.items()}
        self._chunk_re = regex.compile(pattern)
        longest_first = sorted(self.special_tokens, key=len, reverse=True)
        self._special_re = regex.compile("(" + "|".join(regex.escape(t) for t in longest_first) + ")") if longest_first else None
        self._cache: dict[bytes, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + len(self.special_tokens)

    # ------------------------------------------------------------------ training
    @classmethod
    def train(cls, texts, vocab_size: int, special_tokens=None, pattern: str = GPT4_PATTERN, verbose: bool = False):
        specials = list(DEFAULT_SPECIAL_TOKENS if special_tokens is None else special_tokens)
        n_merges = vocab_size - 256 - len(specials)
        assert n_merges >= 0, "vocab_size too small for bytes + special tokens"
        helper = cls([], specials, pattern)

        # 1. pre-tokenize and count each distinct chunk once (the key speed-up: we merge over the
        #    table of unique words, weighted by frequency, not over the raw text)
        chunk_counts: Counter = Counter()
        for text in texts:
            for part in helper._split_specials(text):
                if part[1]:
                    continue                                    # special tokens are not learned
                for chunk in helper._chunk_re.findall(part[0]):
                    chunk_counts[chunk.encode("utf-8")] += 1
        words = [list(w) for w in chunk_counts]
        freqs = list(chunk_counts.values())

        # 2. count adjacent pairs, remembering which words contain each pair
        pair_counts: dict = defaultdict(int)
        where: dict = defaultdict(set)
        for i, (w, f) in enumerate(zip(words, freqs)):
            for p in zip(w, w[1:]):
                pair_counts[p] += f
                where[p].add(i)
        heap = [(-c, p) for p, c in pair_counts.items()]        # max-heap by count; ties -> smaller pair
        heapq.heapify(heap)

        # 3. repeatedly merge the most frequent pair
        merges: list[tuple[int, int]] = []
        for _ in range(n_merges):
            pair = None
            while heap:
                neg_c, cand = heapq.heappop(heap)
                current = pair_counts.get(cand, 0)
                if current <= 0:
                    continue
                if -neg_c != current:                           # stale entry: re-insert with the true count
                    heapq.heappush(heap, (-current, cand))
                    continue
                pair = cand
                break
            if pair is None:
                break                                           # nothing left to merge
            new_id = 256 + len(merges)
            merges.append(pair)
            changed = set()
            for i in where.pop(pair):
                w, f = words[i], freqs[i]
                for p in zip(w, w[1:]):
                    pair_counts[p] -= f
                w = merge_ids(w, pair, new_id)
                words[i] = w
                for p in zip(w, w[1:]):
                    pair_counts[p] += f
                    where[p].add(i)
                    changed.add(p)
            pair_counts.pop(pair, None)
            for p in changed:
                if pair_counts[p] > 0:
                    heapq.heappush(heap, (-pair_counts[p], p))
            if verbose and len(merges) % 500 == 0:
                print(f"merge {len(merges)}/{n_merges}")
        return cls(merges, specials, pattern)

    # ------------------------------------------------------------------ encoding
    def _split_specials(self, text: str):
        """Yield (piece, is_special) pairs."""
        if self._special_re is None:
            yield text, False
            return
        for piece in self._special_re.split(text):
            if piece:
                yield piece, piece in self.special_to_id

    def _encode_chunk(self, chunk: bytes) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk)
        while len(ids) >= 2:
            # apply the earliest-learned merge present in this chunk
            best_rank, best_pair = None, None
            for p in zip(ids, ids[1:]):
                r = self.merge_ranks.get(p)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_pair = r, p
            if best_pair is None:
                break
            ids = merge_ids(ids, best_pair, 256 + best_rank)
        self._cache[chunk] = ids
        return ids

    def encode(self, text: str, allow_special: bool = True) -> list[int]:
        out: list[int] = []
        parts = self._split_specials(text) if allow_special else [(text, False)]
        for piece, is_special in parts:
            if is_special:
                out.append(self.special_to_id[piece])
            else:
                for chunk in self._chunk_re.findall(piece):
                    out.extend(self._encode_chunk(chunk.encode("utf-8")))
        return out

    # ------------------------------------------------------------------ decoding
    def decode(self, ids) -> str:
        buf = bytearray()
        for i in ids:
            if i in self.id_to_special:
                buf += self.id_to_special[i].encode("utf-8")
            else:
                buf += self.vocab[i]
        return buf.decode("utf-8", errors="replace")           # a partial UTF-8 character becomes U+FFFD

    # ------------------------------------------------------------------ persistence
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"pattern": self.pattern, "merges": self.merges, "special_tokens": self.special_tokens}, f)

    @classmethod
    def load(cls, path: str):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["merges"], d["special_tokens"], d["pattern"])
