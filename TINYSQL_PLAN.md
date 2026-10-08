# TinySQL-125M — A Language Model Built From Scratch, Corpus to Benchmark

> A ~125M-parameter decoder-only transformer where **every layer of the stack is written by hand**: the BPE tokenizer, the data pipeline, the model, the optimizer, the training loop, the sampler, the constrained decoder, and the evaluation harness. Pretrained from random weights on ~5B tokens, instruction-tuned, specialized for text-to-SQL, and benchmarked on WikiSQL with execution accuracy.

---

## 0. The "From Scratch" Contract

This project is only impressive if the claim is airtight. These are the rules.

| Layer | Written by me (own code) | Allowed library use |
|---|---|---|
| Tokenizer | Byte-level BPE trainer + encoder + decoder | `regex` for the pre-tokenization pattern only |
| Data pipeline | Cleaning, dedup, tokenization, packing, memmap loader | `datasets` / `huggingface_hub` for **downloading** raw text only |
| Model | Embedding, RMSNorm, RoPE, GQA attention, SwiGLU, blocks, KV cache, init | `torch` tensors, `nn.Module`, `nn.Parameter`, autograd |
| Optimizer | AdamW (decoupled weight decay), LR schedule, grad clipping | `torch.optim.AdamW` only to **verify** mine matches |
| Training loop | Gradient accumulation, bf16 autocast, checkpoint/resume, logging, DDP-optional | `torch.amp`, `torch.compile` |
| Generation | Greedy, temperature, top-k, top-p, KV-cache decode | none |
| Constrained decoding | SQL grammar state machine + schema trie + logit masking | none |
| Evaluation | WikiSQL → SQL conversion, SQLite execution comparator, failure taxonomy | `sqlite3` |

**Forbidden in the core path:** `AutoModel`, `LlamaConfig`, `LlamaForCausalLM`, `transformers.Trainer`, `trl`, `peft`, pre-trained tokenizers, pre-trained weights.

**Hugging Face is used in exactly three places, always as an oracle or a baseline, never as a dependency of the model:**
1. `tests/` — random-init HF `LlamaForCausalLM` receives *my* weights; logits must match (proves correctness).
2. `tokenizer/` — HF `tokenizers` trained on the same corpus, to compare against my BPE.
3. `eval/baselines/` — fine-tuned SmolLM2 as a comparison row in the results table.

State this contract at the top of the README. A recruiter should be able to verify it with one `grep`.

---

## 1. Problem and Success Criteria

**Task:** given a table schema and a natural-language question, emit a SQL query.

```
<schema>orders(order_id, customer_id, order_date, total)</schema>
<question>How many orders were placed in March 2024?</question>
<sql>SELECT COUNT(*) FROM orders WHERE order_date BETWEEN '2024-03-01' AND '2024-03-31'</sql>
```

**Success criteria (committed up front, so the results are honest):**

| # | Criterion | Target (hypothesis, not a promise) |
|---|---|---|
| 1 | Hand-written model matches reference logits | max abs diff < 1e-4 in fp32 |
| 2 | Pretraining reaches stable low loss | val loss curve smooth, no unrecovered spikes |
| 3 | WikiSQL execution accuracy | beat a "no pretraining, SQL-only" baseline by a clear margin; get within striking distance of fine-tuned SmolLM2-135M |
| 4 | Constrained decoding | validity rate → ~100%, with measured execution-accuracy gain |
| 5 | Every design choice | has an ablation or a cited reason |

**Non-goals:** SOTA text-to-SQL, Spider/BIRD-level cross-database reasoning.

---

## 2. Architecture

Llama-style decoder-only transformer, implemented manually.

| Setting | Value | Reason |
|---|---|---|
| Layers | 30 | Deep-and-narrow is parameter-efficient at small scale |
| Hidden size `d_model` | 576 | |
| Attention heads | 9 (head dim 64) | |
| KV heads | 3 | Grouped-query attention: 3× smaller KV cache |
| FFN (SwiGLU) hidden | 1536 | |
| Vocabulary | 32,768 (own BPE) | Power of 2; fits `uint16` for token storage; smaller than 49k to avoid embedding bloat |
| Context | 2048 | Schema + question + SQL fits easily |
| Positions | RoPE (θ = 10,000) | Relative, no learned table |
| Norm | RMSNorm, pre-norm, ε = 1e-5 | |
| Embeddings | Tied input/output | Saves ~19M params |
| Biases | None | |

### Parameter count (derived by hand, then asserted in code)

| Component | Calculation | Params |
|---|---|---|
| Embedding (tied) | 32,768 × 576 | 18,874,368 |
| Attention / layer | Q 576×576 + K 576×192 + V 576×192 + O 576×576 | 884,736 |
| SwiGLU MLP / layer | 3 × 576 × 1536 | 2,654,208 |
| RMSNorm / layer | 2 × 576 | 1,152 |
| Per layer | | 3,540,096 |
| 30 layers | | 106,202,880 |
| Final RMSNorm | | 576 |
| **Total** | | **125,077,824 (~125.1M)** |

The model is therefore named **TinySQL-125M**. A unit test asserts `sum(p.numel()) == 125_077_824`.

### Modules to write

- `RMSNorm`: `x * rsqrt(mean(x²) + ε) * g`, computed in fp32.
- `RoPE`: precompute `cos/sin` tables; rotate pairs of q/k dims; support a position offset for KV-cache decoding.
- `GroupedQueryAttention`: project, reshape, apply RoPE, expand KV heads over query groups, causal mask, softmax, output projection. Two backends: a **naive** one (explicit `QKᵀ/√d`, mask, softmax) and a fused one (`F.scaled_dot_product_attention`). Tests assert they agree.
- `SwiGLU`: `down(silu(gate(x)) * up(x))`.
- `Block`: `x = x + attn(norm1(x)); x = x + mlp(norm2(x))`.
- `Model`: embeddings → blocks → final norm → tied LM head. Init: normal(0, 0.02), with residual-output projections scaled by `1/√(2·n_layers)`.
- `generate()`: KV cache, greedy / temperature / top-k / top-p, stop tokens.

---

## 3. The Full Pipeline

```
 raw corpora ──► clean + dedup ──► train BPE ──► tokenize ──► packed .bin shards
                                                                   │
        ┌──────────────────────────────────────────────────────────┘
        ▼
  Stage 1 PRETRAIN (5B tokens, random init)
        ▼
  Stage 2 INSTRUCTION TUNE (chat format, masked loss)
        ▼
  Stage 3 SQL SPECIALIZE (WikiSQL + custom schema)
        ▼
  Decode: greedy / beam / schema-constrained
        ▼
  Benchmark: exec acc, validity, latency, failure analysis, baselines
```

---

## 4. Phase A — Corpus

### A1. Sources

Verify each dataset's current name, size, and **license** before downloading, and record them in `data/SOURCES.md`.

| Source | Share of tokens | Why |
|---|---|---|
| FineWeb-Edu (sample) | ~70% | Clean educational English, the backbone of small-model pretraining |
| SQL and code (SQL from The Stack-style data, plus general code for structure) | ~18% | SQL syntax, identifiers, schemas |
| StackExchange / DBA-style Q&A, SQL tutorials and docs | ~7% | Question → query phrasing |
| Synthetic schema+question+SQL (generated programmatically with templates, **not** from the eval set) | ~5% | Teaches the exact prompt shape early |

Do not put WikiSQL train or dev text into pretraining. Keep the benchmark clean (see A4).

### A2. Cleaning (own code)

1. Unicode normalization (NFKC), strip control characters.
2. Length and quality filters: drop very short docs, extreme symbol ratio, boilerplate.
3. Language filter is not needed for FineWeb-Edu; apply a cheap heuristic for the rest.
4. **Deduplication:** exact (hash) at minimum; MinHash-LSH near-dup as the "own code" stretch. Report % removed.

### A3. Corpus statistics

Report: documents, characters, tokens per source, mix ratios, dedup rate. Plot a histogram of document lengths.

### A4. Decontamination

Compute 13-gram overlap between the pretraining corpus and WikiSQL dev/test questions and SQL. Remove or report any hits. This single step signals benchmark hygiene more than almost anything else.

---

## 5. Phase B — Tokenizer (written from scratch)

### B1. Implementation

- Byte-level BPE (256 base tokens, so no unknown tokens).
- Pre-tokenization regex (GPT-style) so merges do not cross word/number/whitespace boundaries.
- Train on a ~1 GB sample of the cleaned mix, with SQL deliberately over-represented so keywords and identifiers get good merges.
- Algorithm: count pre-token frequencies once, then run merges over the **unique-word table** with incremental pair-count updates (fast enough in pure Python/NumPy; port the hot loop to NumPy or Cython only if needed).
- Special tokens: `<|endoftext|>`, `<|pad|>`, chat tokens (`<|system|>`, `<|user|>`, `<|assistant|>`, `<|end|>`), and SQL-prompt tokens (`<schema>`, `</schema>`, `<question>`, `</question>`, `<sql>`, `</sql>`). Reserve these at fixed IDs so they are single tokens.
- Vocab size 32,768. Save as a simple JSON merges file plus vocab.

### B2. Verification

- `decode(encode(x)) == x` on 100k random documents, including emoji, code, and invalid UTF-8 edge cases.
- Compare against HF `tokenizers` trained on the same sample: compression ratio (bytes/token), and the overlap in the first N merges.
- Report tokens per word on English, and **how SQL keywords and typical column names split** (e.g. `SELECT`, `GROUP BY`, `customer_id`). A table of this is a good README figure.

---

## 6. Phase C — Data Pipeline (own code)

1. Tokenize each shard in parallel, appending `<|endoftext|>` between documents.
2. Write flat `uint16` binary shards (`train_000.bin`, …) plus a held-out `val.bin` (~20M tokens, from every source).
3. **Loader:** memory-map the shards; each step samples random offsets of length 2049 (inputs/targets shifted by one). No padding, because documents are packed back-to-back.
4. Shuffle at shard level; track tokens-seen in the checkpoint for exact resume.
5. Sanity checks: decode a random batch and read it; assert no token ID ≥ vocab; verify the mix ratios.

---

## 7. Phase D — Model Verification (before spending GPU money)

These tests are the backbone of the repo. All run on CPU in under a minute with a tiny config.

| Test | What it proves |
|---|---|
| Param count == 125,077,824 | Architecture matches the design |
| **Oracle test:** random-init HF `LlamaForCausalLM` with the same config, copy *my* weights in, compare logits (fp32, atol 1e-4) | The architecture is correct, with no HF code in my model |
| Causality test: perturb token *t*, assert logits at positions < *t* are unchanged | The mask is right |
| KV-cache test: full-forward logits == incremental-decode logits | The cache and RoPE offset are right |
| RoPE properties: norm-preserving; `q_m·k_n` depends only on `m−n` | RoPE is right |
| Naive attention == fused attention | Both backends agree |
| GQA: with `n_kv = n_heads`, equals standard MHA | Grouping is right |
| Custom AdamW == `torch.optim.AdamW` for 100 steps | Optimizer is right |
| Overfit one batch to ~0 loss | End-to-end learning works |
| Initial loss ≈ `ln(32768) ≈ 10.4` | Init is sane |
| Tokenizer round-trip | Lossless |

---

## 8. Phase E — Stage 1: Pretraining

### E1. Budget

- Compute-optimal (Chinchilla, ~20 tokens/param): **~2.5B tokens**.
- Target: **5B tokens** (~40 tokens/param), with a stretch to 8–10B if credits allow. Small models keep improving past the optimum.
- FLOPs ≈ 6 × 125M × 5B ≈ **3.8 × 10¹⁸**.

| Hardware | Rough time for 5B tokens (assumes ~35–40% MFU) | 
|---|---|
| 1× H100 | ~8–12 hrs |
| 1× A100 80GB | ~15–22 hrs |
| 1× RTX 4090 | ~2–3 days |

Measure real tokens/sec in the first 200 steps and recompute; treat the table as an estimate, not a quote.

### E2. Hyperparameters (starting point, then swept)

| Setting | Value |
|---|---|
| Optimizer | Own AdamW, β = (0.9, 0.95), ε = 1e-8, weight decay 0.1 (no decay on norms/embeddings/biases) |
| Peak LR | Sweep {6e-4, 1e-3, 2e-3} on a 30M proxy; **do not trust 3e-3 blindly** |
| Schedule | Linear warmup 1–2% → cosine to 10% of peak (own implementation) |
| Batch | ~0.5M tokens/step via gradient accumulation |
| Sequence | 2048, packed |
| Precision | bf16 autocast, fp32 master weights and norms |
| Grad clip | 1.0 |
| Compile | `torch.compile`, with a numerical check against eager |

### E3. Training loop features (own code)

- Gradient accumulation, loss scaled correctly.
- Checkpoint every N steps (model, optimizer, step, RNG, data position); **resume-and-continue test** proves bit-reasonable continuity.
- Loss-spike guard: log and auto-skip a step if grad-norm is non-finite.
- Eval every N steps on `val.bin`.
- Logging (W&B or TensorBoard): train/val loss, perplexity, LR, grad norm, tokens/sec, MFU, GPU memory, and generated text samples from fixed prompts (including SQL-ish ones) so training progress is *readable*.

### E4. Pre-flight sequence (cheap → expensive)

1. Overfit one batch (CPU).
2. 10M-param run for ~100M tokens on a free/consumer GPU: loss falls, samples go from garbage to words.
3. LR sweep + ablations on the proxy (Phase F).
4. Short 125M burn-in (~50M tokens): confirm throughput, memory, no NaNs.
5. Full run.

---

## 9. Phase F — Ablations and Scaling (on small proxies)

Run on ~20–30M-param configs with identical data and token budget. Each ablation gets one plot and one sentence of conclusion.

| Ablation | Compare |
|---|---|
| Positions | RoPE vs. learned absolute |
| Attention | GQA vs. full MHA (val loss **and** KV-cache bytes) |
| Embeddings | Tied vs. untied |
| LR | Peak LR sweep; warmup length |
| Tokenizer | My BPE (32k) vs. smaller 16k vs. byte-level: loss per **byte** (the fair metric) |
| Data mix | With vs. without the SQL slice in pretraining (measured after Stage 3) |
| Norm | Pre-norm RMSNorm vs. LayerNorm (optional) |

**Scaling study:** train ~10M, ~25M, ~60M, and 125M on the same data; plot val loss vs. compute (log-log) and fit a power law. A rough curve is enough to demonstrate understanding.

Note: compare loss across tokenizers per byte, not per token.

---

## 10. Phase G — Stage 2: Instruction Tuning

- **Data:** a small general instruction/chat set (e.g. smoltalk subset; check license), reformatted into my chat template.
- **Loss masking:** compute loss only on assistant tokens.
- **Packing:** pack conversations with attention reset between samples is optional; simplest is one conversation per sequence with padding masked.
- **Hyperparams:** LR ~5e-5 to 1e-4 cosine, 2–3 epochs, small batch.
- **Experiment:** SQL fine-tune **with vs. without** this stage (answers "does general instruction tuning help a tiny model?").

---

## 11. Phase H — Stage 3: Text-to-SQL Specialization

### H1. Data

| Dataset | Use |
|---|---|
| **WikiSQL** (56k train / 8.4k dev / 15.9k test) | Primary benchmark, single-table queries with SQLite databases for execution |
| **Custom e-commerce database** (multi-table, fixed schema, ~2–5k pairs) | Demo + "harder" evaluation. Generate pairs with a large model, keep only those whose SQL executes and returns a non-trivial result. Hand-verify a ≥200 sample test set |
| Spider (stretch) | Only a small subset, and only to report an honest "this is where it breaks" number |

### H2. WikiSQL → SQL conversion (own code)

WikiSQL stores queries as a structured form (`sel`, `agg`, `conds`). Write a converter to SQL text with proper quoting, and **verify the converter by executing the converted gold SQL and checking it reproduces the dataset's stored answers**. Normalize whitespace and casing consistently so the model is not learning noise.

### H3. Prompt format

Use the special tokens from Phase B:

```
<schema>table: t(col1, col2, col3)</schema>
<question>...</question>
<sql>SELECT ... FROM t WHERE ...</sql>
```

Optionally include 1–3 sample cell values per column, which often helps with value-matching. Treat it as an ablation.

### H4. Training

- Loss only on the SQL span plus `</sql>`.
- LR ~1e-4 → cosine, 2–4 epochs, evaluate **execution accuracy on dev every epoch** and keep the best checkpoint by that, not by loss.
- Data augmentation (optional, measure it): shuffle column order, paraphrase questions.

---

## 12. Phase I — Decoding

1. **Greedy** (baseline).
2. **Beam search** (own implementation, width 4) with a validity re-rank.
3. **Schema-constrained decoding** (own implementation):
   - A small SQL grammar state machine for the supported subset (`SELECT [agg](col) FROM t WHERE col op value [AND …]`).
   - At each step, compute the set of allowed token IDs from the grammar state and the schema (column-name token trie, table name, operators, aggregate keywords), then mask logits to `-inf` outside it.
   - Value literals: allow free text inside quotes, with the closing quote and `</sql>` handled by the grammar.
4. Also test **execution-guided decoding**: run the top-k beams, discard those that error, return the first that executes.

Report all of them in the same results table.

---

## 13. Phase J — Benchmarking

### J1. Metrics

| Metric | Definition |
|---|---|
| **Execution accuracy** (primary) | Result set of predicted SQL equals gold result set (order-insensitive unless `ORDER BY`) |
| Logical-form / exact-match accuracy | Normalized SQL string match; stricter, reported for comparability with WikiSQL literature |
| Validity rate | % of outputs that parse and execute without error |
| Schema-faithfulness | % of outputs referencing only real columns/tables |
| Latency | ms/query and tokens/sec: CPU (fp32) and GPU (bf16), with and without KV cache |
| Memory | Peak RAM/VRAM, and KV-cache bytes at 2048 ctx (GQA vs. MHA) |

Evaluate on WikiSQL **test** only once, at the end. Use dev for all model selection. Report bootstrap 95% confidence intervals.

### J2. Systems compared

| System | Role |
|---|---|
| TinySQL-125M, **no pretraining** (random init → SQL fine-tune) | Shows what pretraining buys |
| TinySQL-125M, pretrain → SQL (no instruction tuning) | Ablation |
| **TinySQL-125M, full pipeline** | The project |
| TinySQL-125M + constrained decoding | The decoding gain |
| SmolLM2-135M, same fine-tune recipe | Professionally pretrained (~2T tokens) baseline |
| SmolLM2-360M, same fine-tune recipe | Size effect |
| Large LLM, few-shot prompted, no fine-tuning | Upper reference |

Use the **same** training data, prompts, epochs budget, and eval script for every row. Fairness is the whole point of the table.

### J3. Results table (fill in)

| System | Exec acc | Logical acc | Validity | Latency (CPU) | Latency (GPU) |
|---|---|---|---|---|---|
| No-pretrain baseline | | | | | |
| TinySQL-125M (greedy) | | | | | |
| TinySQL-125M (+constrained) | | | | | |
| TinySQL-125M (+exec-guided beam) | | | | | |
| SmolLM2-135M (FT) | | | | | |
| SmolLM2-360M (FT) | | | | | |
| Large LLM (prompted) | | | | | |

### J4. Failure analysis

Categorize **≥150** dev failures by hand-assisted script into: wrong select column, wrong aggregation, wrong condition column, wrong operator, wrong/misspelled value, missing condition, extra condition, syntax error, hallucinated identifier. Show a bar chart per system. Add 5 annotated qualitative examples (input, gold, prediction, why it failed).

### J5. Statistical hygiene

Fixed seeds, 3 fine-tuning seeds for the headline rows (mean ± std), confidence intervals, and a note on contamination checks (Phase A4).

---

## 14. Phase K — Demo and Packaging

- **Demo (Gradio on Hugging Face Spaces):** paste a schema, ask a question, see the SQL, execute it on a sample SQLite DB, show a result table; toggle constrained decoding; show tokens/sec.
- **Release:** weights as `safetensors`, a model card (data, limits, intended use, eval numbers, license), the tokenizer files, and a standalone `inference.py` with no dependencies beyond PyTorch.
- **CPU inference:** confirm it runs on a laptop at usable speed.

---

## 15. Repository Structure

```
tinysql/
├── README.md                    # results table, plots, architecture diagram, FROM-SCRATCH CONTRACT
├── TINYSQL_PLAN.md
├── tokenizer/
│   ├── bpe.py                   # trainer, encoder, decoder (own)
│   ├── train_tokenizer.py
│   └── compare_hf.py            # oracle comparison + stats
├── data/
│   ├── SOURCES.md               # datasets, versions, licenses
│   ├── download.py
│   ├── clean.py                 # normalization, filters
│   ├── dedup.py                 # exact + MinHash
│   ├── decontaminate.py         # 13-gram overlap vs WikiSQL
│   ├── build_shards.py          # tokenize + pack to uint16 .bin
│   ├── loader.py                # memmap batch sampler
│   ├── sft_data.py
│   ├── wikisql_to_sql.py        # + execution verification of converter
│   └── make_custom_db.py
├── model/
│   ├── config.py
│   ├── rmsnorm.py
│   ├── rope.py
│   ├── attention.py             # naive + fused backends, GQA, KV cache
│   ├── mlp.py                   # SwiGLU
│   ├── model.py
│   └── generate.py              # sampling + KV-cache decode
├── optim/
│   ├── adamw.py                 # own AdamW
│   └── schedule.py              # warmup + cosine
├── train/
│   ├── pretrain.py
│   ├── sft.py
│   ├── finetune_sql.py
│   └── utils.py                 # checkpointing, logging, MFU
├── decode/
│   ├── beam.py
│   ├── sql_grammar.py           # state machine
│   └── constrained.py           # schema trie + logit masking
├── eval/
│   ├── execution_acc.py
│   ├── latency.py
│   ├── failure_analysis.py
│   ├── bootstrap_ci.py
│   └── baselines/               # SmolLM2 fine-tune scripts (HF allowed here only)
├── experiments/
│   ├── ablations/               # one config per ablation
│   ├── scaling/
│   └── sweeps/
├── tests/
│   ├── test_param_count.py
│   ├── test_matches_hf_oracle.py
│   ├── test_causality.py
│   ├── test_kv_cache.py
│   ├── test_rope.py
│   ├── test_attention_backends.py
│   ├── test_adamw.py
│   ├── test_tokenizer.py
│   └── test_wikisql_converter.py
├── configs/                     # YAML per run (tiny, 25m, 60m, 125m, sft, sql)
├── demo/app.py
└── notebooks/                   # plots only; no logic lives here
```

---

## 16. Timeline (7–8 weeks, part-time)

| Week | Deliverable | Gate to pass before moving on |
|---|---|---|
| 1 | Model modules + all Phase D tests | Oracle test passes at fp32 atol 1e-4 |
| 2 | Corpus download, clean, dedup; own BPE trained and verified | Tokenizer round-trip + HF compression comparison done |
| 3 | Shards, loader, training loop, AdamW; tiny-config run | One-batch overfit; resume test; samples look like language |
| 4 | Ablations + LR sweep + scaling runs on proxies | Plots produced; peak LR chosen |
| 5 | **Full 125M pretraining** (rented GPU) | Val loss on track vs. scaling-curve prediction |
| 6 | Instruction tuning; WikiSQL conversion verified; SQL fine-tune; baselines trained | Dev exec acc measured for all rows |
| 7 | Constrained/beam decoding; full benchmarking; failure analysis | Results table filled, CIs computed |
| 8 | Demo, model card, README, write-up/video | Fresh-clone reproduction works |

Buffer: the pretraining run will not go perfectly. Keep ~1 week of slack and checkpoint often.

---

## 17. Budget

| Item | Estimate |
|---|---|
| Final 125M pretraining (one H100/A100, ~1 day) | ~$30–80 |
| Proxy ablations / scaling (many small runs) | ~$30–60 |
| SQL fine-tunes + 3 baselines × seeds | ~$20–40 |
| **Total** | **~$100–200** |

Estimates only; check current rental prices. Debug on CPU or a free T4 so paid GPU time is spent only on verified code.

---

## 18. Risks

| Risk | Mitigation |
|---|---|
| Loss spikes / divergence | Warmup, clip, lower LR, checkpoint often, step-skip on non-finite grads |
| Subtle architecture bug | Oracle test + causality + cache tests before any real run |
| Pretraining too short → weak SQL | SQL slice in mix; more tokens; measure the no-pretrain baseline to show the effect either way |
| Own BPE is slow | Merge over unique-word table; train on a sample, not the full corpus |
| Benchmark contamination | Decontamination step; report overlap |
| GPU cost overrun | Proxy-first workflow; fixed budget gate before the full run |
| WikiSQL label noise caps accuracy | Report the known ceiling honestly; supplement with the custom DB |
| Hallucinated identifiers | Schema-constrained decoding, reported with and without |

---

## 19. README Checklist (what a recruiter sees in 60 seconds)

1. One-paragraph pitch plus the **From-Scratch Contract** table.
2. Architecture diagram and the hand-derived parameter table.
3. **Headline results table** (Section 13.3) with CIs.
4. Training curves (loss, LR, grad-norm), and the scaling-law plot.
5. Ablation figure grid.
6. Failure-analysis chart.
7. Demo GIF + link to the live Space and the weights.
8. "What broke and what I learned" section (loss spikes, tokenizer pitfalls, contamination finding, etc.).
9. Reproduce: exact commands, configs, seeds, hardware, cost.

### Resume bullet (fill in real numbers only)

> Built a 125M-parameter Llama-style LLM entirely from scratch in PyTorch — custom BPE tokenizer, data pipeline, transformer (GQA, RoPE, SwiGLU, KV cache), AdamW, and training loop — and pretrained it on X B tokens; fine-tuned for text-to-SQL to X% execution accuracy on WikiSQL (vs. Y% for fine-tuned SmolLM2-135M trained on ~2T tokens); added grammar/schema-constrained decoding (+Z pts validity); validated against a reference implementation to 1e-4 logit tolerance.

---

## 20. Interview Talking Points This Project Earns You

- Why tied embeddings, GQA, RoPE, pre-norm, and SwiGLU, with the ablation to back each.
- Where the 125M parameters live, derived on a whiteboard.
- How BPE works and why byte-level avoids unknown tokens.
- Chinchilla vs. over-training small models.
- Why loss per token is not comparable across tokenizers.
- How the KV cache works and what GQA saves.
- How constrained decoding works at the logit level.
- Why execution accuracy beats exact match, and how benchmark contamination is checked.
- What the gap to SmolLM2 teaches about pretraining scale.

---

## 21. Follow-up Project

Use TinySQL-125M as the cheap SQL tool inside a data-analyst agent: a larger model plans, TinySQL writes queries, and the planner takes over after two failed attempts. Evaluate success rate, steps, cost, and the share of queries the small model handled alone.
