Weights for TinySQL-125M in this repository's own checkpoint format, plus the paper. The same models in standard Hugging Face format: [TinySQL-125M-WikiSQL](https://huggingface.co/adiitya0929/TinySQL-125M-WikiSQL) and [TinySQL-125M-base](https://huggingface.co/adiitya0929/TinySQL-125M-base).

**Paper:** [TinySQL: A 125M-Parameter Text-to-SQL Model Built and Pretrained From Scratch on 4.5B Tokens](https://doi.org/10.5281/zenodo.23396770) (Zenodo, DOI 10.5281/zenodo.23396770)

| File | Size | What it is |
|---|---|---|
| `tinysql_125m_wikisql_lr5e-05_s0.pt` | 477 MB | **The paper's model.** Fine-tuned on WikiSQL, LR 5e-5, seed 0. Test execution accuracy 78.01% greedy, 80.44% with schema + values constraints |
| `tinysql_125m_base.pt` | 477 MB | Pretrained base model, step 8,600 (4.51B tokens), weights only |
| `tinysql_125m_pretrain_step8600_full.pt` | 1.40 GB | Full training state at step 8,600: weights, AdamW moments, loss scaler, data-loader and torch random states |
| `tinysql_125m_pretrain_step8500_full.pt` | 1.40 GB | The same at step 8,500; resuming from it reruns the last 100 steps |
| `tinysql_125m_wikisql_lr1e-04_s0.pt` | 477 MB | Learning-rate sweep run, LR 1e-4, seed 0. Scored on dev only: 78.46% greedy, 81.46% constrained (8,421 questions) |
| `TinySQL-technical-report-v1.pdf` | 0.8 MB | The paper |
| `SHA256SUMS.txt` | | Checksums: `sha256sum -c SHA256SUMS.txt` |

Every file loads with `torch.load(path, weights_only=True)`. The model config is stored inside: `TinySQL(ModelConfig(**ckpt["cfg"]["model"]))`, then `load_state_dict(ckpt["model"])`.

```bash
# score the paper's model on test (greedy, or --decode schema / values)
python -m data.download --only wikisql
python -m eval.run_eval --ckpt tinysql_125m_wikisql_lr5e-05_s0.pt --split test --decode values --out runs/check

# fine-tune the base model on WikiSQL yourself
python -m train.finetune_sql --config configs/finetune_wikisql.yaml --set base_ckpt=tinysql_125m_base.pt

# resume pretraining: the trainer picks up the latest ckpt_*.pt in out_dir (needs the token shards from the data pipeline)
mkdir -p runs/main && cp tinysql_125m_pretrain_step8500_full.pt runs/main/ckpt_0008500.pt
python -m train.pretrain --config configs/tinysql_125m.yaml
```

Training past step 8,600 means raising `max_steps`, which also stretches the cosine learning-rate schedule from that point on.

**Also on Hugging Face:** the learning-rate 1e-4 run as branch [`lr1e-4-seed0`](https://huggingface.co/adiitya0929/TinySQL-125M-WikiSQL/tree/lr1e-4-seed0), and the SmolLM2 baselines fine-tuned with the same recipe: [SmolLM2-135M-WikiSQL-baseline](https://huggingface.co/adiitya0929/SmolLM2-135M-WikiSQL-baseline) and [SmolLM2-360M-WikiSQL-baseline](https://huggingface.co/adiitya0929/SmolLM2-360M-WikiSQL-baseline).

**Not available:** the weights of TinySQL seeds 1 and 2, the no-pretraining control, the GPT-2, Pythia and SmolLM2 test-set runs, and the 25M ablation models were not kept. Their scores are in `results/benchmark_test.json`, `results/benchmark_dev.json` and `PROJECT_LOG.md`.

The weights are released under Apache-2.0, as on Hugging Face.
