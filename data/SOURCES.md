# Data sources

Downloaded 2026-10-07 into `data/raw/` (git-ignored) with `python -m data.download`. Sizes are measured on disk; token counts are before cleaning/dedup.

| Name | Hugging Face repo / URL | What we took | Measured size | License (status) | Role |
|---|---|---|---|---|---|
| `fineweb_edu` | HuggingFaceFW/fineweb-edu | `sample/10BT`, files 000-004 | 3.66M docs, **3.76B tokens** (dataset's own count), 10.8 GB | ODC-By 1.0 (from Hub metadata) | main pretraining text |
| `stack_sql` | bigcode/the-stack-dedup | `data/sql`, shards 0-7 of 27 | 294,528 files, 3.64 GB text, ~0.9B tokens est. | gated; each file carries its own permissive license in `max_stars_repo_licenses` (keep this field; attribution may be required) | SQL code |
| `stackexchange` | HuggingFaceTB/stackexchange_2025_md | `dba`, `datascience` | 128k threads, 484 MB text, ~0.12B tokens est. | CC BY-SA (each post has a `ContentLicense` field; **verify wording on the dataset card before release**) | SQL / data Q&A |
| `gretel_sql` | gretelai/synthetic_text_to_sql | train + test | 100k examples, 49 MB | Apache-2.0 (from Hub metadata) | schema + question + SQL in pretraining mix |
| `smoltalk` | HuggingFaceTB/smoltalk | `smol-magpie-ultra` (2 shards), `smol-constraints`, `smol-rewrite`, `smol-summarize`, `systemchats-30k` | 355k conversations | **not stated in Hub metadata, verify on the dataset card** | instruction tuning (stage 2) |
| `wikisql` | github.com/salesforce/WikiSQL (`data.tar.bz2`) | everything | train/dev/test `.jsonl` + `.tables.jsonl` + SQLite `.db` (26 MB archive) | **verify on the repo** | stage-3 fine-tuning data and the benchmark |

## Rules

- WikiSQL train/dev/test text must **not** appear in the pretraining mix. `data/decontaminate.py` checks 13-gram overlap against WikiSQL questions and SQL before shards are built.
- Not used on purpose: `b-mc2/sql-create-context` (built from WikiSQL and Spider, contamination risk).
- The Stack is gated: access is tied to the account that accepted the terms. Do not redistribute raw files; release only the trained weights, the tokenizer, and the code.
- Mix at pretraining time (target): ~80% FineWeb-Edu, ~15-19% SQL code, ~3% Q&A, plus the synthetic SQL examples (upsampled, they are tiny). Final proportions are set after cleaning, from the measured token counts.
