# TinySQL-125M: Project Log

A running, detailed record of what has been built, how, why, what broke, and what was measured. It complements [TINYSQL_PLAN.md](TINYSQL_PLAN.md) (the design and the intended road map). This file records what has actually happened. Every number below was measured in this repository unless marked as an estimate.

Last updated: 2026-10-08.

---

## 1. Status at a glance

| Phase | State |
|---|---|
| A. Corpus: download, clean, dedup, decontaminate | done (decontamination numbers in section 6.4) |
| B. Tokenizer (own byte-level BPE) | **done and trained** on the real corpus (section 5.1); comparison against Hugging Face's trainer still to do |
| C. Data pipeline (token shards, loader) | **done**: 5.41B training tokens in `data/shards/` (section 6.5) |
| D. Model from scratch + verification | **done**, matches a reference Llama to 1e-4 |
| E. Pretraining (optimizer, loop, run) | optimizer and loop **done** (section 11); GPU throughput **measured** (section 12); the full run is not started |
| F. Ablations and scaling | not started |
| G. Instruction tuning | not started |
| H. Text-to-SQL fine-tuning | not started |
| I. Decoding (beam, constrained) | generation done; beam and constrained not started |
| J. Benchmarking and failure analysis | not started |
| K. Demo, model card, README | not started |

Tests: **106 passing** (`python -m pytest tests`).
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
| Cross-entropy ran the 125M model out of GPU memory at micro-batch 8 (several fp32 copies of the 8 x 2048 x 32,768 logits) | first run on a real GPU | custom autograd function: saves only the original logits and one logsumexp per token, works in place, recomputes the softmax in the backward pass; gradient tests against PyTorch's loss |
| The new loss corrupted its own input when logits were already fp32 (`.float()` returns the same tensor, so in-place ops modified the saved input) | the unit tests, on CPU | always copy explicitly (`to(torch.float32, copy=True)`); the bf16 GPU path would have hidden it |

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

### 5.1 Training on the real corpus (`tokenizer/train_tokenizer.py`)

Sample: random row groups, SQL-heavy on purpose: 15 FineWeb-Edu, 16 Stack SQL, 9 Stack Exchange, 19 Gretel row groups (955 MB of text), plus WikiSQL **train** questions and column headers only, so dev and test stay unseen. Counting runs in parallel; chunks seen fewer than 2 times are ignored in the merge loop.

A 10%-scale test run came first (54 MB, 58 s total) and exposed a sampler bug: Parquet's `total_byte_size` is already uncompressed, but it had been multiplied by 4, so a 350 MB target would have yielded about 90 MB. After the fix the sampler hits its targets (FineWeb-Edu about 354 MB, Stack SQL 418 MB, Stack Exchange 166 MB, Gretel 37 MB).

Full run, result saved to `tokenizer/tinysql_bpe.json` (413 KB, 32,768 tokens = 256 bytes + 32,500 merges + 12 special tokens):

| Measurement | Value |
|---|---|
| Counting | 26 s; 955 MB of text, 2,587,243 distinct chunks (1,207,284 seen at least twice) |
| 32,500 merges | 229 s; about 1.5 GB RAM |

Held-out compression, bytes per token (higher is better):

| Source | Aggregate | Median document |
|---|---|---|
| FineWeb-Edu | 4.49 | 4.56 |
| Stack Exchange | 3.80 | 4.06 |
| Gretel | 4.10 | 4.10 |
| Stack SQL | 2.71 | 3.60 |

The Stack SQL aggregate is dragged down by data-heavy files (`INSERT` rows of UUIDs and numbers compress at about 1.3 bytes/token; random keys are inherently incompressible). The median SQL file compresses at 3.60. On 900 random SQL files, a tokenizer from the 10% test scored a median of 3.47, so the full run improved it.

How SQL tokenizes: ` SELECT`, ` FROM`, ` WHERE` are single tokens; ` GROUP BY`, ` ORDER BY`, ` INNER JOIN`, ` CREATE TABLE` are two tokens (keyword + keyword); `customer_id` is ` customer` + `_id`; `COUNT(*)` is `COUNT` + `(*)`. Digits are split in groups of at most 3 by design.

Still to do for the tokenizer: the comparison against Hugging Face's trainer on identical data (`tokenizer/compare_hf.py`).

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

### 6.5 Token shards and the loader (`data/build_shards.py`, `data/loader.py`, 3 tests)

Every document is tokenized with the trained tokenizer, followed by `<|endoftext|>`, and written to flat `uint16` files (`data/shards/train/` and `data/shards/val/`, one file per source and row-group range, 60 files).

- **Mix in the shards:** FineWeb-Edu and Stack Exchange whole; Gretel repeated 3 times; **70% of the Stack SQL files**, chosen by a "query score" (0 to 11 distinct query features: `SELECT`, `FROM`, `WHERE`, `JOIN`, `GROUP BY`, `ORDER BY`, `HAVING`, aggregates, `LIMIT`, `UNION`, CTEs). The histogram over 878,957 files: 461,218 score 0, then 74,518 / 61,612 / 110,028 / 70,086 / 40,922 / 26,708 / 19,241 / 9,747 / 3,675 / 1,027 / 175 for scores 1 to 11. Keeping 70% means every file with at least one query feature plus 43% of the score-0 files (612,761 files kept).
- **Control-token safety:** web, code and Q&A text is encoded with `allow_special=False`; only the Gretel examples may produce `<schema>`/`<question>`/`<sql>` ids. Verified on the built shards: 0 unexpected special tokens in sampled FineWeb-Edu and Stack SQL shards.
- **Validation split:** a deterministic 0.4% of documents per source (hash of the text), written separately.
- **Loader:** memory-mapped files; a training example is any window of `seq_len + 1` consecutive tokens, drawn uniformly over all tokens (windows never cross a file boundary), so the training mix equals the shard mix; no padding; inputs are `tokens[:-1]`, targets `tokens[1:]`. The sampler state can be saved and restored for exact resume. Documents are packed back to back, so attention can cross a document boundary (standard for pretraining).

Build time: 1,164 s on 14 workers (about 4.7M tokens/s). Real counts:

| Source | Training docs | Training tokens | Share | Validation tokens |
|---|---|---|---|---|
| FineWeb-Edu | 3,589,789 | 3,838,959,370 | 71.0% | 15,515,778 |
| Stack SQL | 612,761 | 1,424,490,027 | 26.3% | 6,090,553 |
| Stack Exchange | 122,659 | 120,646,509 | 2.2% | 488,623 |
| Gretel (x3) | 99,164 | 24,806,583 | 0.5% | 33,030 |
| **Total** | | **5,408,902,489** | | **22,127,984** |

**Correction to an earlier estimate.** Section 8 below originally estimated the SQL code at about 1.33B tokens, using 4 bytes per token. The trained tokenizer compresses SQL code at only about 2.7 bytes per token (aggregate; data-heavy files compress worst), so all 878,957 deduplicated SQL files are about 1.97B tokens, and the 70% selection (which favors larger, query-rich files) holds 1.42B of them. That made the SQL share 26.3% of tokens, not the roughly 19% expected from "70% of the files". Shards are kept as built; the training mix is to be set with sampling weights in the loader (see section 9).

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

Training tokens in the shards (measured with the trained tokenizer; the earlier byte-based estimates were wrong for SQL, see 6.5):

| Slice | Training tokens | Share of shards |
|---|---|---|
| FineWeb-Edu | 3.84B | 71.0% |
| Stack SQL (70% of files, quality-selected) | 1.42B | 26.3% |
| Stack Exchange | 0.12B | 2.2% |
| Gretel synthetic SQL (3 repeats) | 0.025B | 0.5% |
| **Total** | **5.41B** | |

Pretraining on 4 to 5B tokens is well above the compute-optimal point (about 2.5B for 125M parameters). The mix actually trained on is set at training time with per-source sampling weights, not by the shard contents.

---

## 9. What is next, in order

1. Compare the trained tokenizer against Hugging Face's trainer on identical data (`tokenizer/compare_hf.py`).
2. Add per-source sampling weights to the loader so the training mix is chosen at training time (default idea: about 78% FineWeb-Edu, 19% SQL code, 2.5% Stack Exchange, 0.5% Gretel), without rebuilding shards.
3. Own AdamW (checked against `torch.optim.AdamW`), warmup and cosine schedule, training loop with gradient accumulation, precision switch (bf16, and fp16 with loss scaling for T4-class GPUs), checkpoint and exact resume, logging. Verify with a one-batch overfit and a resume test on CPU.
4. Measure real tokens per second on a free Colab or Kaggle T4, then decide where the full run happens. The laptop's integrated GPU is not usable for training (compute-limited and poorly supported by PyTorch on Windows). The plan is a rented A100 or H100 for about a day for the final pretraining; small ablations can use a free T4.
5. Ablations and scaling on small proxies; the full pretraining run.
6. Instruction tuning (SmolTalk subset), WikiSQL conversion and fine-tuning, constrained and beam decoding, benchmarking against fine-tuned SmolLM2 baselines, failure analysis, demo, model card.

---

## 10. Reproducing what exists

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest tests                          # 106 tests
python -m data.download --dry-run               # list what would be downloaded
python -m data.download                         # download all sources (about 17 GB)
python -m data.clean                            # data/raw -> data/clean
python -m data.dedup                            # data/clean -> data/dedup
python -m data.decontaminate                    # data/dedup -> data/final (about 11 min)
python -m data.build_shards                     # data/final -> data/shards (about 19 min)
python -m tokenizer.train_tokenizer             # about 5 minutes -> tokenizer/tinysql_bpe.json
```

Learning scripts that reproduce the lessons: `learn/01_tensors.py`, `learn/02_generation_demo.py`, `learn/03_tokenizer_demo.py`.

---

## 11. Optimizer, schedule, training loop, and the training-time mix

**Own AdamW** (`optim/adamw.py`): decoupled weight decay, betas (0.9, 0.95), fp32 moment buffers, per-group learning rate and decay, `state_dict` for exact resume. Verified against `torch.optim.AdamW`: parameters agree to 1e-6 after 100 steps, including a decay group and a no-decay group. Also in the module: global-norm gradient clipping (returns the pre-clip norm, which may be non-finite). `optim/schedule.py`: linear warmup, then cosine decay to 10% of peak. Weight decay applies to 2-D weight matrices only, not to norm gains and not to the tied embedding table.

**Training loop** (`train/pretrain.py`, `train/utils.py`), 6 tests:
- gradient accumulation (tokens per step = micro-batch x accumulation x sequence length); bf16 autocast on A100/H100/L4, fp16 with my own dynamic loss scaler for T4-class GPUs, fp32 on CPU;
- non-finite gradients skip the update and are counted, instead of poisoning the weights;
- per-source validation loss on fixed batches (comparable across evaluations), greedy text samples from fixed prompts, `metrics.jsonl`, optional TensorBoard, tokens/s, MFU and peak GPU memory readouts, optional `torch.compile`;
- checkpoints are written atomically (temp file, then rename), the last two are kept; resume restores weights, optimizer, loader position, loss scaler and RNG. **Test: stop at step 10 and resume to 20 gives bit-identical weights to training straight through.**

**First run on the real shards** (CPU, 4.93M parameters, `configs/debug_tiny.yaml`, 300 steps, 2.5M tokens): loss 10.40 at step 1 (ln 32768 = 10.397) down to about 6.3 to 6.7; validation loss fineweb_edu 6.68, stack_sql 6.15, stackexchange 6.96, gretel_sql 6.78. Generated samples are still gibberish, as expected for 0.05% of the token budget; the run confirms mechanics, not quality. CPU speed was 2.7k tokens/s.

**Training-time mix** (`MixedLoader` in `data/loader.py`, 6 tests). The shards keep everything; the mix is chosen at training time by sampling weights, so it can change without rebuilding. Default two-phase schedule (specialised data concentrated near the end, while the learning rate is decaying):

| Phase | FineWeb-Edu | SQL code | Stack Exchange | Gretel |
|---|---|---|---|---|
| first 85% of training | 82% | 15% | 2.5% | 0.5% |
| last 15% | 54% | 36% | 8% | 2% |
| overall average | about 78% | about 18% | about 3.3% | about 0.7% |

Repetition check for a 4.5B-token budget, as epochs over the available data (1.0 = each token seen once), columns FineWeb-Edu / SQL / Stack Exchange / Gretel: constant 8% SQL 1.05 / 0.25 / 0.63 / 1.63; constant 17% SQL 0.94 / 0.54 / 1.01 / 1.63; shards as built (26% SQL) 0.83 / 0.83 / 0.82 / 2.72; constant 35% SQL 0.70 / 1.11 / 1.49 / 5.44; the staged schedule above 0.91 / 0.57 / 1.35 / 4.76. The staged mix is a reasoned default, **not yet proven best**: the planned proxy experiment (small models, several mixes, scored on WikiSQL after fine-tuning) will confirm or change it.

---

## 12. Google Cloud setup and GPU throughput

**Account and quota.** Project `tinysql`, billing enabled, Compute Engine API on. The global GPU quota (`GPUS_ALL_REGIONS`) was 0 and was raised to 1 through a support case (approved). L4 quota was already 1 per region. A100 quota is 0 (an A100 request was filed for asia-east1, a region that offers no A100 zones; a request for asia-southeast1 plus the A2 CPU quota would be needed for an A100). vCPU quota is 100 in asia-southeast1, SSD 250 GB, total disk 2,048 GB.

**Region: asia-southeast1 (Singapore).** It was chosen from a zone-availability query: it offers both L4 (zones a, b, c) and A100 40GB, and it is near the developer. us-central1 also offers both.

**Data.** Bucket `gs://tinysql-data-474078649724` (Singapore, public access blocked). The 181 shard files (10,862,071,753 bytes) were uploaded with `gcloud storage rsync` in about 25 minutes (average 6.8 MiB/s, which saturated the home link and caused unrelated API calls to fail meanwhile) and verified byte for byte against the local copy. Inside Google's network the same data downloads to the VM in about a minute. The VM's service account needed an explicit read-only grant on that one bucket (`roles/storage.objectViewer`).

**VM.** `g2-standard-8` (8 vCPU, 31 GB RAM, 1 x NVIDIA L4 with 23 GB, driver 580), 200 GB balanced disk, image `pytorch-2-9-cu129-ubuntu-2204-nvidia-580` (PyTorch 2.9.1, CUDA 12.9, bf16 supported), zone asia-southeast1-b (zone a had no L4 capacity at the time), a 5-hour automatic stop as a safety net, restricted scopes `storage-ro,logging-write`. Code is shipped as a `git archive` tarball, so no GitHub credentials are placed on the VM.

**Throughput benchmark of the real 125M model** (sequence length 2048, bf16 autocast, my training loop, 30 steps, L4 peak assumed 121 TFLOPS; MFU uses 6 x parameters x tokens/s, so it ignores attention FLOPs):

| Config | Steady tokens/s | Peak GPU memory | MFU |
|---|---|---|---|
| micro-batch 8, eager (first attempt) | out of memory | above 22 GB | n/a |
| micro-batch 4, eager | 14.7k | 11.9 GB | 9.1% |
| micro-batch 8, `torch.compile` | **27.1k** | 15.9 GB | 16.8% |

Compilation costs about 85 seconds once, at the first step. The first attempt exposed the cross-entropy memory problem listed in section 4.8; memory at micro-batch 8 is still dominated by activations (fp32 residual stream and RMSNorm intermediates), which compile reduces.

**Projection for the full run on one L4:** 4.5B tokens at 27.1k tokens/s is about **46 hours** (3B tokens: about 31 hours). At roughly $1 per hour (an estimate, to be checked in the pricing calculator) that is on the order of $50 including disk. An A100 should be several times faster at about $3.5 to 4 per hour, likely a similar total cost with far less wall-clock time, but it needs A100 and A2-CPU quota in the chosen region.

**Current state.** The VM is stopped (status TERMINATED), so there are no GPU or CPU charges; the 200 GB disk (about $24 per month, estimate) and the bucket (about $0.25 per month) remain. Costs so far are on the order of a couple of dollars.

**Next:** decide between the L4 for the whole run and requesting A100 quota; run the proxy experiments (learning-rate sweep, data-mix check, architecture ablations); then the full pretraining run with checkpoints and resume.
