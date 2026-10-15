# Zenodo upload: values to paste

**File to upload:** `paper/TinySQL-technical-report-v1.pdf`

| Field | Value |
|---|---|
| Resource type | Publication → Report |
| Title | TinySQL: A 125M-Parameter Text-to-SQL Model Built and Pretrained From Scratch on 4.5B Tokens |
| Publication date | 2026-10-15 |
| Creator | Jha, Aditya. Affiliation: Coherent Market Insights. Role (if asked): AI Engineer. Add your ORCID iD if you have one. |
| License | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| Version | v1.0 |
| Language | English |
| Related works | Relation "Is supplemented by", identifier `https://github.com/aditya0929/tinysql`, scheme URL, resource type Software |

## Description

We describe TinySQL-125M, a decoder-only language model for text-to-SQL in which every component of the model path is written from scratch: a byte-level BPE tokenizer, the transformer itself (grouped-query attention, rotary embeddings, SwiGLU, RMSNorm), the optimizer, the training loop, the data pipeline, a grammar-constrained decoder and the evaluation harness. The model is pretrained for about 46 hours on a single NVIDIA L4 GPU on 4.5B tokens of educational web text and SQL code, then fine-tuned on WikiSQL.

To make the result interpretable, we fine-tune four public models of similar size (GPT-2 124M, Pythia-160M, SmolLM2-135M and SmolLM2-360M) with an identical recipe, sweep the learning rate for every model, average over three seeds, train our architecture without pretraining as a control, and report confidence intervals and paired significance tests throughout.

On the WikiSQL test set, TinySQL reaches 77.95 ± 0.18% execution accuracy with greedy decoding and 80.39 ± 0.20% with schema- and value-constrained decoding (official evaluator: 78.03% and 80.50%). This is about one point below GPT-2 and 17 points above Pythia-160M, which was pretrained on 67× more tokens; SmolLM2-135M, pretrained on 2T tokens (about 440× more), is 4.6 points ahead. Pretraining accounts for 41 points of accuracy. Constrained decoding never turned one of our model's correct answers into a wrong one in any paired comparison we ran. Small-scale ablations show that rotary embeddings and tied embeddings matter far more than grouped-query attention or an SQL-heavy pretraining mix, and a hand audit suggests that most remaining aggregation errors are label errors in WikiSQL.

Code, results and tests: https://github.com/aditya0929/tinysql

## Keywords (one per box)

text-to-SQL · small language models · pretraining from scratch · WikiSQL · constrained decoding · grouped-query attention · rotary position embeddings · tokenizer · benchmark · reproducibility
