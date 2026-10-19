# TinySQL-125M

[![Paper DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23396770.svg)](https://doi.org/10.5281/zenodo.23396770)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-TinySQL--125M--WikiSQL-yellow)](https://huggingface.co/adiitya0929/TinySQL-125M-WikiSQL)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MQ70s8txcYtgMd99cH_WHvYwgcmJd117?usp=sharing)

A 125M-parameter text-to-SQL language model in which every part of the model path is written from scratch: the byte-level BPE tokenizer, the Llama-style transformer, the optimizer, the training loop, the data pipeline, a grammar-constrained decoder and the evaluation harness. It was pretrained for about 46 hours on one NVIDIA L4 GPU on 4.5B tokens of web text and SQL code, then fine-tuned on WikiSQL.

| | |
|---|---|
| **Paper** | [TinySQL: A 125M-Parameter Text-to-SQL Model Built and Pretrained From Scratch on 4.5B Tokens](https://doi.org/10.5281/zenodo.23396770) (Zenodo, 2026, DOI 10.5281/zenodo.23396770). PDF also in [`paper/`](paper/TinySQL-technical-report-v1.pdf) |
| **Models** (Apache-2.0) | [TinySQL-125M-WikiSQL](https://huggingface.co/adiitya0929/TinySQL-125M-WikiSQL) (text-to-SQL) and [TinySQL-125M-base](https://huggingface.co/adiitya0929/TinySQL-125M-base) (pretrained base) |
| **Try it** | [Colab notebook](https://colab.research.google.com/drive/1MQ70s8txcYtgMd99cH_WHvYwgcmJd117?usp=sharing) and the notebooks in [`playground/`](playground/) |

## Quick start

The models load with plain `transformers`; no code from this repository is needed.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("adiitya0929/TinySQL-125M-WikiSQL")
model = AutoModelForCausalLM.from_pretrained("adiitya0929/TinySQL-125M-WikiSQL")

prompt = ('<schema>CREATE TABLE t ("Player" TEXT, "Position" TEXT, "School/Club Team" TEXT);</schema>\n'
          '<question>What position does the player who played for butler cc (ks) play?</question>\n<sql>')
ids = tok(prompt, return_tensors="pt").input_ids
print(tok.decode(model.generate(ids, max_new_tokens=64)[0, ids.shape[1]:], skip_special_tokens=True))
# SELECT "Position" FROM t WHERE "School/Club Team" = 'butler cc (ks)';
```

**Prompt format:**
- The table is always called `t`.
- Column names are double-quoted, and each column is typed `REAL` (numeric) or `TEXT`.
- The question goes inside `<question>…</question>`, and the prompt ends with `<sql>`.
- The model writes one `SELECT` statement and then `</sql>`.

## Playground

| Notebook | What it shows |
|---|---|
| [`TinySQL_125M_WikiSQL`](playground/Copy_of_TinySQL_125M_WikiSQL.ipynb) | Load the model from the Hub, ask questions about a pandas table, and run the generated SQL with SQLite. |
| [`Constrained decoding + real data`](<playground/add_constrained_decoding_notebook_(greedy_vs_grammar_constrained).ipynb>) | The paper's constrained decoder as a self-contained `transformers` `LogitsProcessor`, applied to 344 real penguin records. A small grounding step maps each value onto a real cell (`'Gentoo penguin'` becomes `'Gentoo'`). The notebook also shows honestly what the model gets wrong. |

**Why constrain:** with plain greedy decoding the model sometimes invents a column, such as `SUM(" salary")`. SQLite silently treats an unknown quoted name as a text string, so the query runs and returns a wrong answer. The constrained decoder only lets the model write real column names, and values copied from the question, so every query is valid. The notebook version produced the same SQL as the repository's [`decode/`](decode/) on all 9 questions we compared.

## Results (WikiSQL test, 15,878 questions, execution accuracy %)

| Model | Pretraining tokens | Greedy | + Schema + values constraints |
|---|---|---|---|
| **TinySQL-125M (this repo)** | **4.5B** | **77.95 ± 0.18** | **80.39 ± 0.20** |
| GPT-2 124M | not published | 78.93 | 82.36 |
| Pythia-160M | 300B | 60.78 | 67.63 |
| SmolLM2-135M | 2T | 82.57 ± 0.37 | 85.22 ± 0.21 |
| SmolLM2-360M | 4T | 87.03 ± 0.19 | 88.01 ± 0.12 |
| Same architecture, no pretraining | 0 | 37.39 | 48.33 |

Every model was fine-tuned with the identical recipe, after its own learning-rate sweep, and scored once on the test set. Results are mean ± std over 3 seeds where shown. The official WikiSQL evaluator agrees with our executor within 0.11 points for these systems; it gives 78.03% execution and 72.28% logical-form accuracy for our model at greedy.

**Main findings:**
- **Pretraining is most of the result:** it adds 41 points over the same network trained from random weights.
- **Close to GPT-2, far ahead of Pythia:** within about 1 point of GPT-2 (124M), and 17 points above Pythia-160M despite 67× less pretraining data.
- **Constraints help weaker models more:** +2.4 points for ours, +6.9 for Pythia and +1.0 for SmolLM2-360M (means over seeds). They never turned one of our model's correct answers into a wrong one in any paired comparison.
- **Ablations on a 25M-parameter version:** RoPE (+8.5 points) and tied embeddings (+5.0) matter far more than grouped-query attention (+0.75, not significant) or an SQL-heavy pretraining mix (+0.9).
- **Errors:** accuracy falls as queries need more conditions. A hand audit of 50 aggregation errors found that in 62% of them the WikiSQL label itself does not match the question.

The paper has the confidence intervals, McNemar tests, failure analysis and figures; [`PROJECT_LOG.md`](PROJECT_LOG.md), sections 15 to 18, has every number.

## What is built from scratch

| Part | Where |
|---|---|
| Byte-level BPE tokenizer (32,768 tokens) | [`tokenizer/`](tokenizer/) |
| Transformer: RMSNorm, RoPE, grouped-query attention, SwiGLU, KV cache, memory-lean cross-entropy | [`model/`](model/) |
| AdamW, cosine schedule, gradient clipping | [`optim/`](optim/) |
| Training loop with bf16, `torch.compile`, atomic checkpoints and exact resume | [`train/`](train/) |
| Data pipeline: cleaning, exact and MinHash-LSH deduplication, benchmark decontamination (0 leaks in 4.7M documents), token shards | [`data/`](data/) |
| Schema- and value-constrained decoding | [`decode/`](decode/) |
| Sandboxed SQL execution, statistics (bootstrap, McNemar), failure analysis | [`eval/`](eval/) |

No pretrained weights, pretrained tokenizers or Hugging Face model classes are used to build or train the model. Hugging Face code appears in only four places:
- as a test oracle: our model reproduces a reference Llama's logits to within 1e-4;
- to download data;
- to run the baselines;
- to publish the finished model ([`scripts/export_hf.py`](scripts/export_hf.py)).

The export is checked to be exact: all 48,598 WikiSQL dev and test strings tokenize identically, and greedy SQL matches on 300 of 300 dev questions. All 169 tests run with `python -m pytest tests`.

## Reproduce

```bash
pip install -r requirements.txt
python -m pytest tests                                   # 169 tests

# data (about 17 GB download)
python -m data.download
python -m data.clean
python -m data.dedup
python -m data.decontaminate
python -m tokenizer.train_tokenizer
python -m data.build_shards

# pretraining (about 46 h on one L4) and fine-tuning
python -m train.pretrain --config configs/tinysql_125m.yaml
python -m data.wikisql_to_sql
python -m train.finetune_sql --config configs/finetune_wikisql.yaml --set base_ckpt=runs/main/ckpt_0008600.pt

# evaluation: greedy, or constrained with --decode schema / values
python -m eval.run_eval --ckpt runs/finetune_wikisql/best.pt --split dev --decode values --out runs/finetune_wikisql

# export to Hugging Face format and check it is exact
python -m scripts.export_hf --ckpt runs/finetune_wikisql/best.pt --out release/TinySQL-125M-WikiSQL --sql
python -m scripts.verify_hf_export --ckpt runs/finetune_wikisql/best.pt --hf release/TinySQL-125M-WikiSQL
```

[`scripts/benchmark_queue.py`](scripts/benchmark_queue.py) runs the full benchmark unattended: learning-rate sweep, seeds, baselines, no-pretraining control, ablations and the test set. [`PROJECT_LOG.md`](PROJECT_LOG.md) records every run, every measurement and every bug found along the way.

## Limitations

- **WikiSQL shape only:** one table, one selected column, an optional aggregate, and up to four `AND` conditions. No joins, `GROUP BY`, `ORDER BY` or nesting.
- **Real-world wording:** the model often copies too much of the question into a value. The grounding step in the playground fixes many of these cases.
- **English only.** Generated SQL should never be run with write permissions on a real database.

## Citation

```bibtex
@techreport{jha2026tinysql,
  title       = {{TinySQL}: A 125M-Parameter Text-to-SQL Model Built and Pretrained From Scratch on 4.5B Tokens},
  author      = {Jha, Aditya Narayan},
  year        = {2026},
  institution = {Zenodo},
  doi         = {10.5281/zenodo.23396770},
  url         = {https://doi.org/10.5281/zenodo.23396770}
}
```

Author: Aditya Narayan Jha ([ORCID 0009-0000-0770-886X](https://orcid.org/0009-0000-0770-886X)).
