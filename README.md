# TinySQL-125M

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23396770.svg)](https://doi.org/10.5281/zenodo.23396770)

A 125M-parameter text-to-SQL language model in which every part of the model path is written from scratch: the byte-level BPE tokenizer, the Llama-style transformer, the optimizer, the training loop, the data pipeline, a grammar-constrained decoder and the evaluation harness. It was pretrained for about 46 hours on one NVIDIA L4 GPU on 4.5B tokens of web text and SQL code, then fine-tuned on WikiSQL.

**Technical report:** [TinySQL: A 125M-Parameter Text-to-SQL Model Built and Pretrained From Scratch on 4.5B Tokens](https://doi.org/10.5281/zenodo.23396770) (Zenodo, 2026). The PDF is also in [`paper/`](paper/TinySQL-technical-report-v1.pdf).

**Models on Hugging Face** (Apache-2.0, load with plain `transformers`):
[TinySQL-125M-WikiSQL](https://huggingface.co/adiitya0929/TinySQL-125M-WikiSQL) (text-to-SQL) ·
[TinySQL-125M-base](https://huggingface.co/adiitya0929/TinySQL-125M-base) (pretrained base)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("adiitya0929/TinySQL-125M-WikiSQL")
model = AutoModelForCausalLM.from_pretrained("adiitya0929/TinySQL-125M-WikiSQL")
prompt = ('<schema>CREATE TABLE t ("Player" TEXT, "Position" TEXT, "School/Club Team" TEXT);</schema>\n'
          '<question>What position does the player who played for butler cc (ks) play?</question>\n<sql>')
ids = tok(prompt, return_tensors="pt").input_ids
print(tok.decode(model.generate(ids, max_new_tokens=64)[0, ids.shape[1]:], skip_special_tokens=True))
```

## Results (WikiSQL test, 15,878 questions, execution accuracy %)

| Model | Pretraining tokens | Greedy | + Schema + values constraints |
|---|---|---|---|
| **TinySQL-125M (this repo)** | **4.5B** | **77.95 ± 0.18** | **80.39 ± 0.20** |
| GPT-2 124M | not published | 78.93 | 82.36 |
| Pythia-160M | 300B | 60.78 | 67.63 |
| SmolLM2-135M | 2T | 82.57 ± 0.37 | 85.22 ± 0.21 |
| SmolLM2-360M | 4T | 87.03 ± 0.19 | 88.01 ± 0.12 |
| Same architecture, no pretraining | 0 | 37.39 | 48.33 |

Every model is fine-tuned with the identical recipe, after its own learning-rate sweep. Results are the mean ± standard deviation over 3 seeds where shown. The official WikiSQL evaluator agrees with our executor within 0.11 points for these systems. Confidence intervals, paired significance tests, ablations and the failure analysis are in the report and in [`PROJECT_LOG.md`](PROJECT_LOG.md), sections 15 to 17.

## What is built from scratch

| Part | Where |
|---|---|
| Byte-level BPE tokenizer (32,768 tokens) | [`tokenizer/`](tokenizer/) |
| Transformer: RMSNorm, RoPE, grouped-query attention, SwiGLU, KV cache, memory-lean cross-entropy | [`model/`](model/) |
| AdamW, cosine schedule, gradient clipping | [`optim/`](optim/) |
| Training loop with bf16, `torch.compile`, atomic checkpoints and exact resume | [`train/`](train/) |
| Data pipeline: cleaning, exact and MinHash deduplication, benchmark decontamination, token shards | [`data/`](data/) |
| Schema- and value-constrained decoding | [`decode/`](decode/) |
| Sandboxed SQL execution, statistics (bootstrap, McNemar), failure analysis | [`eval/`](eval/) |

No pretrained weights, pretrained tokenizers or Hugging Face model classes are used in the model path. Hugging Face code is used only as a test oracle (our model reproduces a reference Llama's logits to within 1e-4), to download data, and to run the baselines. The 169 tests run with `python -m pytest tests`.

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
```

The full benchmark (learning-rate sweep, seeds, baselines, control, ablations, test set) runs unattended with [`scripts/benchmark_queue.py`](scripts/benchmark_queue.py). [`PROJECT_LOG.md`](PROJECT_LOG.md) records every run, every measurement and every bug found along the way.

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
