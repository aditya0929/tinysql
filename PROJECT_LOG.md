# TinySQL-125M: Project Log

A running, detailed record of what has been built, how, why, what broke, and what was measured. It complements [TINYSQL_PLAN.md](TINYSQL_PLAN.md) (the design and the intended road map). This file records what has actually happened. Every number below was measured in this repository unless marked as an estimate.

Last updated: 2026-10-08.

---

## 1. Status at a glance

| Phase | State |
|---|---|
| A. Corpus: download, clean, dedup, decontaminate | done (decontamination numbers in section 6.4) |
| B. Tokenizer (own byte-level BPE) | implemented and tested; training on the real corpus is the next step |
| C. Data pipeline (token shards, loader) | not started |
| D. Model from scratch + verification | **done**, matches a reference Llama to 1e-4 |
| E. Pretraining (optimizer, loop, run) | not started |
| F. Ablations and scaling | not started |
| G. Instruction tuning | not started |
| H. Text-to-SQL fine-tuning | not started |
| I. Decoding (beam, constrained) | generation done; beam and constrained not started |
| J. Benchmarking and failure analysis | not started |
| K. Demo, model card, README | not started |

Tests: **82 passing** (`python -m pytest tests`).
Code pushed to `github.com/aditya0929/tinysql` (branch `main`).

---

## 2. The "from scratch" contract (what is and is not allowed)

| Layer | Written by hand | Library use allowed |
|---|---|---|
| Tokenizer | byte-level BPE trainer, encoder, decoder | `regex` for the pre-tokenization pattern |
| Model | embedding, linear, RMSNorm, RoPE, GQA attention, SwiGLU, block, KV cache, loss | PyTorch tensors, `nn.Module`, `nn.Parameter`, autograd |
| Generation | greedy, temperature, top-k, top-p, the random draw itself, KV-cache loop | none |
| Data pipeline | cleaning filters, dedup (exact and MinHash/LSH), decontamination | `pyarrow` for parquet I/O, `xxhash` for hashing, `datasets`/`huggingface_hub` only to download |

Not used anywhere in the model path: `AutoModel`, `LlamaConfig`, `LlamaForCausalLM`, pretrained tokenizers or weights, `transformers.Trainer`.

Hugging Face appears in exactly three roles: a test oracle (`tests/test_matches_hf_oracle.py`, `model/convert_hf.py`), a tokenizer comparison (planned), and baseline rows in the final benchmark (planned). PyTorch's fused attention kernel is an optional, tested backend; my manual attention remains the reference it is checked against.

---

## 3. Environment

- Windows 11, Python 3.12.10, virtual environment in `.venv/`, 16 CPU cores, 31 GB RAM.
- **No NVIDIA GPU.** The laptop has an AMD Radeon 860M integrated GPU. PyTorch is the CPU build (`torch 2.14.1+cpu`). The laptop does everything up to and including tiny training runs. Real training needs a rented NVIDIA GPU (see section 9).
- `requirements.txt`: torch, numpy, regex, pytest, tqdm, pyyaml, matplotlib, pandas, datasets, huggingface_hub, tokenizers, transformers, safetensors, tensorboard, gradio. (`pyarrow` and `xxhash` come in as dependencies.)
- Data volumes: about 20 GB on disk across `data/raw`, `data/clean`, `data/dedup`, `data/final`. All are git-ignored.

---

## 4. Architecture decisions and the model

### 4.1 Changes from the original scope

| Original scope | What was built | Why |
|---|---|---|
| Reuse the SmolLM2 tokenizer (49,152 tokens) | **Own byte-level BPE, 32,768 tokens** | the tokenizer is part of "from scratch"; a smaller vocabulary removes about 9.4M embedding parameters, and 32,768 is a power of two that fits `uint16` |
| Name: TinySQL-135M | **TinySQL-125M** (125,077,824 parameters) | follows from the smaller vocabulary |
| NFKC Unicode normalization | **NFC** | NFKC rewrites symbols such as `²` and ligatures, which would corrupt SQL string literals |
| Peak LR 3e-3 | to be swept, not assumed | aggressive for this size; will be chosen from a proxy sweep |

### 4.2 Hyperparameters (`model/config.py`)

| Setting | Value |
|---|---|
| Layers | 30 |
| Hidden size `d_model` | 576 |
| Query heads / KV heads | 9 / 3 (head dim 64; each KV head serves 3 query heads) |
| SwiGLU width | 1536 (= 8/3 x 576, giving the same parameter count as a classic 4x two-matrix MLP) |
| Vocabulary | 32,768 |
| Context | 2048 |
| Positions | RoPE, theta 10,000 |
| Norm | RMSNorm, pre-norm, eps 1e-5 |
| Embeddings | tied input/output |
| Biases | none |
| Init | normal(0, 0.02); residual-output projections (`o_proj`, `down_proj`) scaled by 1/sqrt(2 x n_layers) |

### 4.3 Parameter count, derived by hand and asserted in a test

| Component | Calculation | Parameters |
|---|---|---|
| Embedding (tied) | 32,768 x 576 | 18,874,368 |
| Attention per layer | Q 576x576 + K 576x192 + V 576x192 + O 576x576 | 884,736 |
| SwiGLU per layer | 3 x 576 x 1536 | 2,654,208 |
| RMSNorm per layer | 2 x 576 | 1,152 |
| One layer | | 3,540,096 |
| 30 layers | | 106,202,880 |
| Final RMSNorm | | 576 |
| **Total** | | **125,077,824** |

### 4.4 Modules (all in `model/`)

| File | What it does | Key points |
|---|---|---|
| `config.py` | `ModelConfig` dataclass | derived `head_dim`, `n_rep`; `expected_params()` is the by-hand derivation as code; `fused_attention` flag |
| `linear.py` | `y = x @ W^T`, no bias | weight shape `(out, in)`, the same layout as PyTorch and Hugging Face |
| `embedding.py` | table lookup `weight[ids]` | |
| `rmsnorm.py` | `x / sqrt(mean(x^2) + eps) * g` | computed in fp32, cast back once at the end; per-token, so KV-cache-safe |
| `rope.py` | rotary position embeddings | precomputed cos/sin tables; pairs dimension `i` with `i + d/2` (the Llama convention); `start_pos` offset for cached decoding; rotation done in fp32 |
| `attention.py` | grouped-query attention | own stable softmax; own `repeat_kv` (query head `h` reads KV head `h // n_rep`); manual causal attention plus an optional fused backend; KV cache stores un-expanded K/V (3 heads, not 9) |
| `mlp.py` | SwiGLU: `down(silu(gate(x)) * up(x))` | own `silu` built on `torch.sigmoid` (the naive `1/(1+exp(-x))` gives NaN gradients for very negative inputs) |
| `block.py` | pre-norm decoder block | `x = x + attn(norm(x)); x = x + mlp(norm(x))` |
| `model.py` | full model + own cross-entropy | tied head `x @ E^T`; one shared RoPE table for all layers; loss supports `ignore_index=-100` for masking prompt tokens |
| `generate.py` | sampling and generation | see 4.6 |
| `convert_hf.py` | weight-name mapping to Hugging Face Llama | used only by the oracle test |

### 4.5 Attention, in one paragraph

Each token produces a query, key, and value. Scores are `q . k / sqrt(d)`, masked so a token cannot see later tokens, softmaxed into weights, and used to average the values. Nine query heads share three KV heads, which makes the KV cache 3x smaller. With the fused backend, prefill uses PyTorch's kernel with `is_causal=True`, single-token decoding uses it without a mask, and anything else (a multi-token chunk with a cache) falls back to the manual implementation. All three paths are tested against each other.

### 4.6 Generation (`model/generate.py`)

- `filter_logits`: top-k and top-p (nucleus) by sorting, masking with `-inf`, and scattering back. The top token always survives.
- `sample_from_probs`: **inverse-CDF sampling** (cumulative sum, one uniform number, first index above it). Written by hand instead of `torch.multinomial`; uses `<=` so zero-probability tokens can never be drawn.
- `generate`: prefill once, then one token at a time with the KV cache; per-sequence `finished` flags and stop ids; seedable through a `torch.Generator`.
- Measured on the full 125M model on CPU: 48 tokens in **1.71 s with the cache vs 3.31 s without**, with identical output.

### 4.7 Verification of the model

| Test | Proves |
|---|---|
| parameter count == 125,077,824 (tied) and +18,874,368 (untied) | the architecture table is real |
| **oracle test**: my weights loaded into a random-init Hugging Face `LlamaForCausalLM` (eager attention), logits compared in fp32 | the architecture is correct. Passes at max abs difference < 1e-4 for small models (fused and manual attention, tied and untied) **and for the full 125M model** |
| causality: changing token t leaves positions < t unchanged (both backends) | the mask is right |
| incremental decode with KV cache == one full forward (attention, block, whole model, both backends) | cache and RoPE offsets are right |
| fused attention == manual attention, including the cache path | the fast path is correct |
| RoPE: position 0 is identity, norm-preserving, dot product depends only on relative position, `start_pos` slice equals the full result, 2-D worked example | RoPE is right |
| RMSNorm: worked example, unit RMS output, scale invariance, tokens independent, bf16 dtype preserved | norm is right |
| softmax, silu, cross-entropy (with ignored positions) match PyTorch | the hand-written primitives are right |
| initial loss ~ ln(vocab): 10.47 vs ln(32768) = 10.397 on the full model | the init is sane |
| overfit one batch to loss < 0.05 | gradients flow end to end |
| sampler frequencies match target probabilities (20,000 draws, tolerance 0.02) | the inverse-CDF sampler is right |

### 4.8 Bugs found along the way

| Bug | How it was caught | Fix |
|---|---|---|
| RMSNorm returned fp32 for bf16 input (fp32 gain times bf16 tensor promotes) | dtype test | multiply in fp32, cast once at the end |
| Fused attention path rebound `k` to the 9-head expanded tensor and would have cached it | review while writing, before it ran | keep `k, v` for the cache and use `k_all, v_all` for attention |
| Initial-loss test returned 5.75 instead of about 6.9 | the test itself | the test used `targets = ids`, which is easy with tied embeddings (the stream starts as `E[token]` and the head scores against the same `E`); fixed by using independent random targets |
| `xxhash` rejected `str` input in MinHash | dedup test | encode to bytes first |

---

## 5. Tokenizer (`tokenizer/bpe.py`)

Byte-level BPE, written from scratch.

- **Ids:** 0 to 255 are raw bytes (so nothing is ever "unknown"); 256 and up are learned merges; the 12 special tokens occupy the **last** ids: `<|endoftext|>`, `<|pad|>`, `<|system|>`, `<|user|>`, `<|assistant|>`, `<|end|>`, `<schema>`, `</schema>`, `<question>`, `</question>`, `<sql>`, `</sql>`.
- **Pre-tokenization:** a GPT-4-style regex splits text into word, number (up to 3 digits), punctuation, and whitespace chunks, so merges never cross those boundaries.
- **Training** (`count_chunks` + `train_from_counts`): count each distinct chunk once, then merge over that table weighted by frequency. A pair-count table plus a "which words contain this pair" index make each merge touch only affected words; a max-heap with lazy invalidation picks the best pair; ties break deterministically so training is reproducible. `min_count` ignores chunks seen fewer times than the threshold.
- **Encoding:** repeatedly apply the earliest-learned merge present in a chunk; per-chunk cache.
- **Special tokens** are matched before the regex, so `<sql>` is one id. With `allow_special=False` the same text is encoded as ordinary text. **This must be used for web and code corpora** so a webpage that contains the literal text `<sql>` cannot inject a control token during pretraining.
- **Decode:** bytes joined and decoded as UTF-8 with replacement for broken sequences.

Verification: 21 tests (round-trips on Unicode, emoji, whitespace, random Unicode; first merge is the most frequent pair; a frequent word becomes one token; special-token behavior and fixed ids; determinism; save/load; decode of arbitrary ids).

Early comparison (4.6 MB of Python standard-library text, vocabulary 4,000): **3.64 bytes/token for my tokenizer vs 3.63 for Hugging Face's `tokenizers`** trained on the same data. Training took 2.8 s.

Training on the real corpus is next. The script is `tokenizer/train_tokenizer.py`: a SQL-heavy sample of random row groups (about 400 MB Stack SQL, 350 MB FineWeb-Edu, 150 MB Stack Exchange, 40 MB Gretel) plus WikiSQL **train** questions and column headers only, so dev and test stay unseen. It prints per-source bytes/token on held-out text and how SQL keywords split.

---

## 6. Data pipeline

### 6.1 Sources (`data/download.py`, `data/SOURCES.md`)

| Source | What was taken | Measured size | License (status) | Role |
|---|---|---|---|---|
| FineWeb-Edu `sample/10BT` | files 000 to 004 | 3.66M docs, 3.76B tokens (dataset's count) | ODC-By (Hub metadata) | general text |
| The Stack (dedup), `data/sql` | **all 27 shards** | 994,019 files, 12.2 GB text | gated; per-file permissive licenses kept in the data | SQL code |
| Stack Exchange (`dba`, `datascience`) | 2 sites | 127,963 threads, 484 MB | CC BY-SA (per-post field; wording to verify before release) | SQL / data Q&A |
| `gretelai/synthetic_text_to_sql` | train | 100,000 examples, 37 MB after formatting | Apache-2.0 | schema + question + SQL in our prompt format |
| SmolTalk (subset) | `smol-magpie-ultra` (2 shards) and four small subsets | 355k conversations | to verify on the dataset card | instruction tuning (stage 2) |
| WikiSQL (Salesforce GitHub archive) | everything, including SQLite DBs | 26 MB | to verify | stage-3 fine-tuning and the benchmark |

Deliberately not used: `b-mc2/sql-create-context` (derived from WikiSQL and Spider, contamination risk).

Notes: the Stack requires a Hugging Face account that accepted the terms (done). Total download about 17 GB. Two problems were handled during download: a transient DNS failure on the router (resolved by waiting and retrying), and a wrong shard-name pattern (`train-` vs `data-`) caught by a dry run before downloading.

### 6.2 Cleaning (`data/clean.py`, 7 tests)

Pure per-source filter functions; each dropped document is counted under its reason. Runs in parallel across files and is resumable. Every output row keeps `text, source, license, origin` (repo name, URL, or post id) so attribution survives.

| Source | Rules | Result |
|---|---|---|
| FineWeb-Edu | drop < 200 chars, > 1% replacement characters, < 55% alphabetic, or > 30% repeated lines | 3,661,000 to 3,660,810 (190 dropped: 134 low alphabetic, 27 broken encoding, 20 repeated lines, 9 too short) |
| Stack SQL | drop < 50 chars, any line > 1000 chars, < 30% alphanumeric, or no SQL keyword. **`INSERT` data dumps are truncated** to the first 8,000 characters at a line boundary (keeps the `CREATE TABLE` schema, discards bulk data) | 994,019 to 893,352 (89.9% kept); 12.2 GB to 5.37 GB. Reasons: 38,294 no SQL keyword, 36,681 truncated dumps, 34,805 long lines, 26,331 too short, 1,237 low alphanumeric |
| Stack Exchange | drop negative score or < 300 chars; uses the ready-made markdown `ThreadText` | 127,963 to 123,159 (96.2%); 484 to 468 MB |
| Gretel | keep only `CREATE TABLE` statements as the schema; rewrite to `<schema>..</schema>\n<question>..</question>\n<sql>..</sql>` | 100,000 to 99,561 |

Why dump truncation: about 4% of SQL files were `INSERT`-dominated but they held about 25% of the bytes. Dropping them would lose schemas; keeping them whole would teach the model to emit data rows. A spot check of dropped files confirmed the filters removed non-SQL text, one-line scripts, and `mysqldump` output, with one legitimate `CREATE TABLE` file lost to a long enum line (an accepted small loss).

### 6.3 Deduplication (`data/dedup.py`, 3 tests)

1. **Exact:** a 64-bit hash of each document's text, shared across all sources and files in a fixed order; the first occurrence is kept.
2. **Near-duplicate (Stack SQL only):** MinHash with 128 hash functions over 5-word shingles; LSH with 16 bands of 8 rows (candidate probability about 0.5 at similarity 0.7 and about 1 at 0.85); a candidate pair merges only if at least 80% of signature slots agree; the longest document of each cluster is kept. In tests, a one-word edit in a 300-word document scores above 0.85 and unrelated documents score below 0.1.

| Source | After cleaning | After exact | Final | Removed | Text |
|---|---|---|---|---|---|
| FineWeb-Edu | 3,660,810 | 3,604,108 | 3,604,108 | 1.5% | 17,013 MB |
| Stack SQL | 893,352 | 891,129 | 878,957 | 1.6% | 5,329 MB |
| Stack Exchange | 123,159 | 123,159 | 123,159 | 0% | 468 MB |
| Gretel | 99,561 | 99,561 | 99,561 | 0% | 37 MB |

Exact dedup took 299 s; near-duplicate removal of the SQL files took 175 s. The Stack's "dedup" variant was already near-deduplicated by its authors, which explains the small near-duplicate yield (12,172 files).

### 6.4 Decontamination against WikiSQL (`data/decontaminate.py`, 4 tests)

Goal: no pretraining document may contain a WikiSQL **dev or test** question, so the benchmark stays clean.

- Match on lowercased word tokens, punctuation ignored.
- Questions with **13 or more** tokens: any 13-gram.
- Questions with **8 to 12** tokens (the median question has 11): the **whole question** must appear as a contiguous run.
- Questions under 8 tokens (about 14% of dev/test) are skipped as too generic; this is a documented limitation.
- Implementation: 64-bit rolling polynomial hashes built incrementally for n = 2 to 13, checked against a sorted array with `searchsorted`; parallel over files; flagged documents are dropped; the clean copy is written to `data/final/`.

**Result** (24,299 dev+test questions; 3,473 skipped for having fewer than 8 tokens; 57,359 n-gram hashes in the index; 35 files scanned in 675 s):

| Source | Documents scanned | Removed |
|---|---|---|
| FineWeb-Edu | 3,604,108 | **0** |
| Stack SQL | 878,957 | **0** |
| Stack Exchange | 123,159 | **0** |
| Gretel | 99,561 | **0** |

Zero hits across 4.7M documents is plausible: WikiSQL questions were written by crowd workers about Wikipedia tables and are unlikely to appear word-for-word in web text, GitHub SQL files, or Stack Exchange threads. Because "found nothing" only means something if the scanner can find something, a **positive control** was run through the real file-scanning path: 200 randomly chosen dev/test questions were planted into 1,000 real FineWeb-Edu documents. The scanner flagged exactly those 200 and none of the other 800. The Gretel data is synthetic and was generated independently of WikiSQL.

Limitation: only questions are checked; gold SQL strings will be added once the WikiSQL-to-SQL converter exists.

---

## 7. Key decisions and why

| Decision | Reason |
|---|---|
| Hand-write even `Linear`, `Embedding`, softmax, SiLU, cross-entropy, sampler | the claim "from scratch" must survive inspection |
| Keep manual attention as the reference and allow a fused backend | speed on real hardware, with the manual version proving the fast one correct |
| 32,768 vocabulary | less embedding waste at 125M parameters; fits `uint16` |
| Over-represent SQL in the tokenizer sample and include WikiSQL train text | SQL keywords and typical column names should be single or few tokens |
| Truncate (not drop) INSERT dumps | keeps schemas, removes bulk data |
| Download all 27 Stack shards | after dump truncation the first 8 shards gave only about 0.4B SQL tokens, too thin for a SQL specialist; all 27 give about 1.3B and can be sampled down at mixing time |
| Decontaminate dev and test, not train | train is the fine-tuning set; only the evaluation splits must stay unseen |
| `allow_special=False` for web and code text | prevents control-token injection from crawled text |

---

## 8. Measured corpus after all processing

Estimates use about 4 bytes per token for SQL and Q&A, and the dataset's own token count for FineWeb-Edu.

| Slice | Text | Approx. tokens | Share of mix |
|---|---|---|---|
| FineWeb-Edu | 17.0 GB | 3.7B | about 71% |
| Stack SQL | 5.3 GB | 1.33B | about 26% |
| Stack Exchange | 0.47 GB | 0.12B | about 2% |
| Gretel synthetic SQL | 0.04 GB | 0.01B | about 0.2% |
| **Total** | | **about 5.2B** | |

Plan: at shard-building time the SQL code slice can be sampled down to roughly 15 to 18% (or kept whole for a more SQL-heavy model), and the tiny Gretel slice repeated a few times. Pretraining on about 4 to 5B tokens is well above the compute-optimal point (about 2.5B for 125M parameters).

---

## 9. What is next, in order

1. Train the BPE tokenizer on the real sample: a 10%-scale run first to measure time and memory, then the full run. Report bytes/token per source and the SQL keyword splits, and compare against Hugging Face's trainer.
2. `build_shards.py` and `loader.py`: tokenize everything (web and code with `allow_special=False`), pack into `uint16` shards with a held-out validation shard, memory-mapped random-window batching.
3. Own AdamW (checked against `torch.optim.AdamW`), warmup and cosine schedule, training loop with gradient accumulation, precision switch (bf16, and fp16 with loss scaling for T4-class GPUs), checkpoint and exact resume, logging. Verify with a one-batch overfit and a resume test on CPU.
4. Measure real tokens per second on a free Colab or Kaggle T4, then decide where the full run happens. The laptop's integrated GPU is not usable for training (compute-limited and poorly supported by PyTorch on Windows). The plan is a rented A100 or H100 for about a day for the final pretraining; small ablations can use a free T4.
5. Ablations and scaling on small proxies; the full pretraining run.
6. Instruction tuning (SmolTalk subset), WikiSQL conversion and fine-tuning, constrained and beam decoding, benchmarking against fine-tuned SmolLM2 baselines, failure analysis, demo, model card.

---

## 10. Reproducing what exists

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest tests                          # 82 tests
python -m data.download --dry-run               # list what would be downloaded
python -m data.download                         # download all sources (about 17 GB)
python -m data.clean                            # data/raw -> data/clean
python -m data.dedup                            # data/clean -> data/dedup
python -m data.decontaminate                    # data/dedup -> data/final
python -m tokenizer.train_tokenizer --scale 0.1 # quick tokenizer run (next step)
```

Learning scripts that reproduce the lessons: `learn/01_tensors.py`, `learn/02_generation_demo.py`, `learn/03_tokenizer_demo.py`.
