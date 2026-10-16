---
license: apache-2.0
language:
- en
library_name: transformers
pipeline_tag: text-generation
base_model: HF_USER/TinySQL-125M-base
datasets:
- Salesforce/wikisql
tags:
- text-to-sql
- sql
- wikisql
- small-language-model
- from-scratch
---

# TinySQL-125M-WikiSQL

A 125M-parameter text-to-SQL model. Given a table schema and a question, it writes the SQL query. Every part of it was built from scratch: the tokenizer, the transformer, the optimizer, the training loop and the data pipeline. It was pretrained on 4.5B tokens on one GPU, then fine-tuned on WikiSQL.

- **Paper:** [TinySQL: A 125M-Parameter Text-to-SQL Model Built and Pretrained From Scratch on 4.5B Tokens](https://doi.org/10.5281/zenodo.23396770) (Zenodo, 2026)
- **Code:** [github.com/aditya0929/tinysql](https://github.com/aditya0929/tinysql)
- **Base model:** [HF_USER/TinySQL-125M-base](https://huggingface.co/HF_USER/TinySQL-125M-base)

## Quick start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "HF_USER/TinySQL-125M-WikiSQL"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo)

prompt = (
    '<schema>CREATE TABLE t ("Player" TEXT, "No." TEXT, "Nationality" TEXT, "Position" TEXT, '
    '"Years in Toronto" TEXT, "School/Club Team" TEXT);</schema>\n'
    "<question>What position does the player who played for butler cc (ks) play?</question>\n"
    "<sql>"
)
ids = tok(prompt, return_tensors="pt").input_ids
out = model.generate(ids, max_new_tokens=64)          # greedy; stops at </sql>
print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
# SELECT "Position" FROM t WHERE "School/Club Team" = 'butler cc (ks)';
```

**Prompt format.** The model was trained on exactly this layout. The table is always called `t`. Column names are double-quoted, and each column is typed `REAL` (numeric) or `TEXT`. The question goes in `<question>…</question>`, and the prompt ends with `<sql>`. The model writes one `SELECT` statement and then `</sql>`.

## Results (WikiSQL test, 15,878 questions)

| Model | Pretraining tokens | Greedy | + Schema + values constraints |
|---|---|---|---|
| **TinySQL-125M (this model, mean of 3 seeds)** | **4.5B** | **77.95 ± 0.18** | **80.39 ± 0.20** |
| GPT-2 124M | not published | 78.93 | 82.36 |
| Pythia-160M | 300B | 60.78 | 67.63 |
| SmolLM2-135M | 2T | 82.57 ± 0.37 | 85.22 ± 0.21 |
| SmolLM2-360M | 4T | 87.03 ± 0.19 | 88.01 ± 0.12 |

The table reports execution accuracy (%): the generated query is run against the table and its result compared with the gold query's. Every model was fine-tuned with the same recipe after its own learning-rate sweep. This checkpoint is seed 0, which scores 78.01 greedy and 80.44 constrained. The official WikiSQL evaluator gives 78.08 execution and 72.38 logical-form accuracy for it.

`model.generate()` above is plain greedy decoding. The constrained numbers use the grammar-constrained decoder in the [GitHub repository](https://github.com/aditya0929/tinysql) (`decode/`). It only allows the table's real column names and values copied from the question, which gives 100% valid SQL.

## Training

| | |
|---|---|
| Architecture | Llama-style decoder: 30 layers, hidden size 576, 9 query / 3 key-value heads (GQA), SwiGLU 1536, RoPE, RMSNorm, tied embeddings, context 2048 |
| Parameters | 125,077,824 |
| Tokenizer | own byte-level BPE, 32,768 tokens |
| Pretraining | 4.51B tokens (FineWeb-Edu, The Stack SQL, Stack Exchange, Gretel synthetic text-to-SQL), 8,600 steps, about 46 h on one NVIDIA L4 |
| Fine-tuning | WikiSQL train (56,355 examples), loss on the SQL only, 3 epochs, batch 32, peak LR 5e-5, best epoch by dev execution accuracy |

The architecture matches Llama, so the weights are distributed in the standard `LlamaForCausalLM` format. The model was built and trained with the repository's own code, not with `transformers`. The export is checked to produce the same token ids, logits and generated SQL as the original implementation.

## Limitations

- **WikiSQL shape only:** one table, one selected column, an optional aggregate, and up to four `AND` conditions. No joins, `GROUP BY`, `ORDER BY` or nesting. Do not expect it to handle Spider-style or production SQL.
- **English only;** small model; greedy decoding can produce column names that don't exist. Use the constrained decoder from the repository to rule that out.
- **Label noise:** WikiSQL's labels are noisy. A hand audit in the paper suggests that most remaining aggregation errors are label errors.
- **Not for production databases** without review: generated SQL should never be executed with write permissions.

## Training data licenses

FineWeb-Edu (ODC-By 1.0), The Stack (dedup), SQL subset (permissively licensed source files, subject to The Stack's terms of use), Stack Exchange dba and datascience (CC BY-SA; 2.5, 3.0 or 4.0 depending on the post date), gretelai/synthetic_text_to_sql (Apache-2.0), WikiSQL (Salesforce, BSD-3-Clause). The pretraining data was decontaminated against WikiSQL dev and test questions.

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
