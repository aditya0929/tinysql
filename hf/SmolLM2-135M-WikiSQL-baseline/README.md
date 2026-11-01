---
license: apache-2.0
language:
- en
library_name: transformers
pipeline_tag: text-generation
base_model: HuggingFaceTB/SmolLM2-135M
datasets:
- Salesforce/wikisql
tags:
- text-to-sql
- sql
- wikisql
- baseline
---

# SmolLM2-135M-WikiSQL-baseline

[SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) (by Hugging Face) fine-tuned on WikiSQL with exactly the recipe used for TinySQL-125M, as a baseline in the TinySQL paper. It answers the question "what does the same fine-tuning give on a model pretrained on about 2T tokens instead of 4.5B?". For the from-scratch model, see [HF_USER/TinySQL-125M-WikiSQL](https://huggingface.co/HF_USER/TinySQL-125M-WikiSQL).

- **Paper:** [TinySQL: A 125M-Parameter Text-to-SQL Model Built and Pretrained From Scratch on 4.5B Tokens](https://doi.org/10.5281/zenodo.23396770) (Zenodo, 2026)
- **Code:** [github.com/aditya0929/tinysql](https://github.com/aditya0929/tinysql) (`eval/baselines/finetune_hf.py`)

## Quick start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "HF_USER/SmolLM2-135M-WikiSQL-baseline"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo)

prompt = ('<schema>CREATE TABLE t ("Player" TEXT, "Position" TEXT, "School/Club Team" TEXT);</schema>\n'
          '<question>What position does the player who played for butler cc (ks) play?</question>\n<sql>')
ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
out = model.generate(ids, max_new_tokens=64, do_sample=False, stop_strings=["</sql>"], tokenizer=tok)
print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).split("</sql>")[0])
# SELECT "Position" FROM t WHERE "School/Club Team" = 'butler cc (ks)';
```

The prompt format is the same as TinySQL's. Here `<schema>`, `<question>` and `<sql>` are plain text, not special tokens.

## Training

WikiSQL train (56,355 examples), loss on the SQL only, batch 32, AdamW (the repository's own), peak LR 1e-4 with warmup and cosine decay, 3 epochs, seed 0. The best epoch was picked by execution accuracy on the first 1,000 dev questions: 76.3 / **78.3** / 77.1%, so epoch 2.

## Results (WikiSQL dev, 8,421 questions)

| Decoding | Execution accuracy | Exact match |
|---|---|---|
| Greedy | 82.72% | 74.79% |
| Schema constraints | 83.46% | 75.05% |
| Schema + values constraints | 85.19% | 76.62% |

The constrained decoder for Hugging Face models is `eval/baselines/constrained_hf.py` in the repository.

This is the dev-phase run. The paper's test-set numbers for SmolLM2-135M (82.57 ± 0.37% greedy, 85.22 ± 0.21% constrained) come from three later runs with the same recipe and LR (seeds 0, 1 and 2), whose weights were not kept.

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

SmolLM2 itself: Ben Allal et al., [*SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language Model*](https://arxiv.org/abs/2502.02737), 2025.
