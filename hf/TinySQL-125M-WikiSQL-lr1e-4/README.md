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

# TinySQL-125M-WikiSQL, branch `lr1e-4-seed0`

This branch holds a learning-rate sweep run: TinySQL-125M-base fine-tuned on WikiSQL with peak LR 1e-4, seed 0. **The paper's model is on the `main` branch** (LR 5e-5, chosen because it scored higher on the first 1,000 dev questions: 74.3% against 73.4% for this run).

- **Paper:** [TinySQL: A 125M-Parameter Text-to-SQL Model Built and Pretrained From Scratch on 4.5B Tokens](https://doi.org/10.5281/zenodo.23396770) (Zenodo, 2026)
- **Code:** [github.com/aditya0929/tinysql](https://github.com/aditya0929/tinysql)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo, rev = "HF_USER/TinySQL-125M-WikiSQL", "lr1e-4-seed0"
tok = AutoTokenizer.from_pretrained(repo, revision=rev)
model = AutoModelForCausalLM.from_pretrained(repo, revision=rev)
```

The prompt format is the same as on `main`.

## Results (WikiSQL dev, 8,421 questions)

| Decoding | Execution accuracy | Exact match |
|---|---|---|
| Greedy | 78.46% | 70.64% |
| Schema + values constraints | 81.46% | 73.03% |

This run was scored on dev only, never on test. Best epoch: 3 of 3 (68.3 / 70.8 / 73.4% on the first 1,000 dev questions). The export to this format was checked the same way as `main`: identical tokenization of all WikiSQL dev and test strings, and identical greedy SQL on the first 300 dev questions.

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
