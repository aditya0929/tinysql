# Humanised rewrite: side by side


## Manual review of the 6 accepted rewrites

Numbers, citations and LaTeX are guaranteed unchanged by the placeholder check. Meaning was checked by hand:

| Unit | Verdict |
|---|---|
| 11 | small error: adds "up to" (micro-batches are exactly 8x2048) |
| 16 | fine (the colon introducing the listing became a full stop) |
| 19 | loses detail: "our AdamW" (own implementation) becomes "AdamW optimiser"; "As a control" dropped |
| 22 | loses a key fact: the official evaluator is run **unmodified**; also describes execution accuracy as queries "matching" rather than results being compared |
| 32 | fine |
| 39 | fine |

The 34 rejected outputs included invented facts, e.g. unit 24 claimed random-init training "reaches about the same performance" and that pretraining on "1.3 trillion tokens adds about five percentage points" (the paper: 4.5B tokens, +40.6 points).

6 paragraphs rewritten, 34 kept as the original (failed a safety check).

## Unit 0 (line 30): KEPT ORIGINAL, placeholders changed (missing [1, 2, 3, 4, 5, 6], found 0 of 37)

We describe TinySQL-125M, a decoder-only language model for text-to-SQL in which every component of the model path is written from scratch: a byte-level BPE tokenizer, the transformer itself (grouped-query attention, rotary embeddings, SwiGLU, RMSNorm), the optimizer, the training loop, the data pipeline, a grammar-constrained decoder and the evaluation harness. The model is pretrained for about 46 hours on a single NVIDIA L4 GPU on 4.5B tokens of educational web text and SQL code, then fine-tuned on WikiSQL. To make the result interpretable, we fine-tune four public models of similar size (GPT-2 124M, Pythia-160M, SmolLM2-135M and SmolLM2-360M) with an identical recipe, sweep the learning rate for every model, average over three seeds, train our architecture without pretraining as a control, and report confidence intervals and paired significance tests throughout. On the WikiSQL test set, TinySQL reaches 77.95\,$\pm$\,0.18\% execution accuracy with greedy decoding and 80.39\,$\pm$\,0.20\% with schema- and value-constrained decoding (official evaluator: 78.03\% and 80.50\%). This is about one point below GPT-2 and 17 points above Pythia-160M, which was pretrained on 67$\times$ more tokens; SmolLM2-135M, pretrained on 2T tokens (about 440$\times$ more), is 4.6 points ahead. Pretraining accounts for 41 points of accuracy. Constrained decoding never turned one of our model's correct answers into a wrong one in any paired comparison we ran. Small-scale ablations show that rotary embeddings and tied embeddings matter far more than grouped-query attention or an SQL-heavy pretraining mix, and a hand audit suggests that most remaining aggregation errors are label errors in WikiSQL. All code, results and the 166 tests that verify the components are public.

## Unit 1 (line 35): KEPT ORIGINAL, placeholders changed (missing [1], found 0 of 1)

Small language models are attractive for text-to-SQL: they run on a CPU, can be deployed next to a database, and are cheap to fine-tune. Most small models available today are pretrained by well-resourced teams on trillions of tokens. This report asks a narrower question: \emph{how far can a single person get on a standard text-to-SQL benchmark with a model whose every part they built themselves, on a pretraining budget of a few billion tokens and one rented GPU?}

## Unit 2 (line 37): KEPT ORIGINAL, placeholders changed (missing [1, 2], found 0 of 2)

To answer it credibly, the system is built without any pretrained weights, pretrained tokenizers or high-level modeling libraries (Section~\ref{sec:system}), and the evaluation is designed so that differences can be attributed (Section~\ref{sec:setup}): every baseline gets the same fine-tuning recipe and its own learning-rate sweep, main results are averaged over seeds, the test set is used once, and comparisons are paired and tested for significance.

## Unit 3 (line 41): KEPT ORIGINAL, placeholders changed (missing [1, 2], found 0 of 2)

A complete, tested, from-scratch text-to-SQL stack: tokenizer, Llama-style model~\cite{touvron2023llama}, optimizer, a data pipeline with deduplication and benchmark decontamination, a grammar-constrained decoder, and evaluation code (Section~\ref{sec:system}).

## Unit 4 (line 42): KEPT ORIGINAL, placeholders changed (missing [1, 2], found 0 of 2)

A controlled comparison on WikiSQL~\cite{zhong2017seq2sql} against four public models of similar size, with learning-rate sweeps, three seeds, a no-pretraining control, bootstrap confidence intervals and McNemar tests (Section~\ref{sec:results}).

## Unit 5 (line 43): KEPT ORIGINAL, placeholders changed (missing [1, 2, 3], found 0 of 3)

Evidence that a 4.5B-token model trained on a deliberate web+SQL mix comes within about one point of GPT-2 and well above Pythia-160M at this task, and that constrained decoding helps weaker models more without ever hurting ours.

## Unit 6 (line 56): KEPT ORIGINAL, placeholders changed (missing [1, 2, 3, 4, 5, 6], found 0 of 6)

WikiSQL~\cite{zhong2017seq2sql} pairs 80,654 crowd-written questions with SQL queries over single Wikipedia tables. Its queries select one column, optionally aggregate it, and filter with up to four conjunctive conditions. Strong specialised systems built on large pretrained encoders report test execution accuracy in the high 80s, and their authors attribute a large share of the remaining errors to annotation errors~\cite{hwang2019sqlova}; recent work cleans the dataset for the LLM era~\cite{pihulski2025llmsql}. Multi-table benchmarks such as Spider~\cite{yu2018spider} are harder and are not addressed here.

## Unit 7 (line 58): KEPT ORIGINAL, placeholders changed (missing [1, 2], found 0 of 2)

Restricting generation to well-formed SQL is an established technique, either by checking partial queries against a parser (PICARD~\cite{scholak2021picard}) or by executing candidate queries (execution-guided decoding~\cite{wang2018execution}). We use a simple byte-level grammar specialised to the WikiSQL query shape, with a stricter level that only admits values copied from the question.

## Unit 8 (line 60): KEPT ORIGINAL, placeholders changed (missing [1, 2, 3, 4, 5, 6], found 0 of 16)

GPT-2~\cite{radford2019gpt2}, Pythia~\cite{biderman2023pythia} and SmolLM2~\cite{benallal2025smollm2} are open models in the 100--400M range with very different pretraining budgets (Pythia: 300B tokens; SmolLM2-135M/360M: 2T/4T tokens; GPT-2: not published). Our architecture follows Llama~\cite{touvron2023llama}: RoPE~\cite{su2021roformer}, RMSNorm~\cite{zhang2019rmsnorm}, SwiGLU~\cite{shazeer2020glu}, grouped-query attention~\cite{ainslie2023gqa} and tied embeddings~\cite{press2017tying}.

## Unit 9 (line 67): KEPT ORIGINAL, placeholders changed (missing [1, 2, 3, 4, 5, 6], found 0 of 6)

All model code is written by hand on top of PyTorch tensors and autograd: linear and embedding layers, RMSNorm, rotary embeddings, grouped-query attention with a KV cache, SwiGLU, a memory-efficient cross-entropy, sampling, AdamW and the training loop. No pretrained weights, pretrained tokenizers or modeling classes from \texttt{transformers} are used anywhere in our model's path. Hugging Face code appears in three roles only: as a test oracle (our weights loaded into a reference Llama implementation produce the same logits to within $10^{-4}$, for the full 125M model), to download datasets, and to run the baselines. PyTorch's fused attention kernel is an optional backend that is tested against our own attention implementation. The repository contains 166 tests, including exact-resume of training, optimizer agreement with \texttt{torch.optim.AdamW} to $10^{-6}$, and a positive control for decontamination.

## Unit 10 (line 71): KEPT ORIGINAL, placeholders changed (missing [1, 3, 6, 7, 8, 11], found 7 of 17)

A byte-level BPE~\cite{sennrich2016bpe} tokenizer with 32,768 tokens (256 bytes, 32,500 merges, 12 special tokens such as \texttt{<schema>}, \texttt{<question>} and \texttt{<sql>}). It was trained on 955\,MB of text deliberately weighted towards SQL, plus WikiSQL \emph{training} questions and column names. Common SQL keywords become single tokens (\texttt{\textvisiblespace SELECT}, \texttt{\textvisiblespace FROM}, \texttt{\textvisiblespace WHERE}). Held-out compression is 4.49 bytes/token on FineWeb-Edu and 3.60 bytes/token on the median SQL file. Web, code and forum text is encoded with special tokens disabled, so crawled text containing the literal string \texttt{<sql>} cannot inject a control token.

## Unit 11 (line 100): rewritten

**Before:** Table~\ref{tab:arch} and Figure~\ref{fig:arch} give the architecture. Weights are initialised from $\mathcal{N}(0, 0.02)$ with residual output projections scaled by $1/\sqrt{2L}$. The loss is a custom autograd function that stores only the logits and one log-sum-exp per token, which was needed to fit micro-batches of 8$\times$2048 tokens with a 32,768-way output on a 24\,GB GPU.

**After:** Table~\ref{tab:arch} and Figure~\ref{fig:arch} show the model architecture. Weights start from $\mathcal{N}(0, 0.02)$, with residual output projections scaled by $1/\sqrt{2L}$. The loss is a custom autograd function that keeps only the logits and one log-sum-exp per token. This lets us fit micro-batches of up to 8$\times$2048 tokens through a 32,768-way-output layer on a 24\,GB GPU.

## Unit 12 (line 123): KEPT ORIGINAL, placeholders changed (missing [2, 3, 4, 5, 6, 7], found 1 of 8)

Table~\ref{tab:data} lists the sources. The pipeline applies per-source quality filters (for example, \texttt{INSERT}-heavy SQL dumps are truncated to keep the schema and drop bulk data rows), exact deduplication across all sources, and MinHash-LSH near-duplicate removal~\cite{broder1997minhash} for SQL files. From the SQL files, the 70\% with the most query features (\texttt{SELECT}, \texttt{JOIN}, \texttt{GROUP BY}, aggregates, and so on) are kept.

## Unit 13 (line 125): KEPT ORIGINAL, placeholders changed (missing [1, 2, 3, 4, 5, 6], found 0 of 10)

Every pretraining document was scanned for WikiSQL dev and test questions: any shared 13-gram for questions of 13 or more words, or the whole question for questions of 8 to 12 words. No document matched in 4.7M scanned. Because a scanner that finds nothing proves little by itself, we ran a positive control through the same code: 200 dev/test questions planted into 1,000 real documents were all flagged, with no false positives. Questions shorter than 8 words (14\%) are too generic to scan and gold SQL strings were not scanned; both are limitations.

## Unit 14 (line 130): KEPT ORIGINAL, output was not the expected JSON

We train with our own AdamW~\cite{loshchilov2019adamw} ($\beta = (0.9, 0.95)$, weight decay 0.1 on weight matrices only), gradient clipping at 1.0, bf16 autocast and \texttt{torch.compile}. The peak learning rate of $1.2\times10^{-3}$ comes from a sweep on a 25M-parameter proxy (optimum near $1.4$--$1.5\times10^{-3}$, lowered for the larger model and longer run), with 200 warmup steps and cosine decay to 10\%. Each step is 524,288 tokens; the run is 8,600 steps, i.e.\ 4.51B tokens or about $3.4\times10^{18}$ FLOPs, roughly 1.8$\times$ the compute-optimal token count for this size~\cite{hoffmann2022chinchilla}. Data is mixed in two phases (Figure~\ref{fig:mix}): 82/15/2.5/0.5\% (web/SQL/Stack Exchange/Gretel) for the first 85\% of steps, then 54/36/8/2\% while the learning rate decays, so the specialised data is concentrated where the learning rate is low.

## Unit 15 (line 139): KEPT ORIGINAL, placeholders changed (missing [1, 2, 3, 8], found 4 of 8)

The run took about 46 hours on one NVIDIA L4 at 27k tokens/s (about 17\% model FLOPs utilisation), with zero skipped steps. Final validation loss is 2.94 nats/token on FineWeb-Edu and 0.96 on SQL code; every source improved at every checkpoint (Figure~\ref{fig:loss}).

## Unit 16 (line 150): rewritten

**Before:** Each WikiSQL example becomes a prompt with the table schema as a \texttt{CREATE TABLE} statement and the question, and a completion with the SQL; the loss is computed on the completion only:

**After:** Each WikiSQL example turns into a pair: a prompt with the table schema as a \texttt{CREATE TABLE} statement and the question, and a completion with the SQL. The loss is calculated from the completion alone.

## Unit 17 (line 156): KEPT ORIGINAL, placeholders changed (missing [1, 2, 3, 4, 5, 6], found 0 of 7)

Fine-tuning uses batch 32, 100 warmup steps, cosine decay, no weight decay and 3 epochs (5,283 steps, about 30 minutes on the L4). After each epoch, execution accuracy is measured on the first 1,000 dev questions and the best epoch is kept.

## Unit 18 (line 158): KEPT ORIGINAL, placeholders changed (missing [1, 2, 3, 4, 5, 6], found 0 of 10)

Two facts about WikiSQL motivate it. 95\% of our model's invalid greedy outputs reference a column that does not exist, and 100\% of gold condition values (11,545 of 11,545 on dev) occur verbatim, ignoring case, in the question. We therefore define two levels. \emph{Schema} is a byte-level grammar for the WikiSQL query shape that only admits the table's real column names. \emph{Schema + values} additionally requires every condition value to be a substring of the question (Unicode-aware, valid UTF-8). At each step, candidate tokens are checked best-first and the first that keeps the output a valid prefix is taken, which is equivalent to masking every invalid token and taking the argmax (Figure~\ref{fig:constrained}). The same decoder runs on the baselines through a byte-level adapter for their tokenizers.

## Unit 19 (line 170): rewritten

**Before:** GPT-2 124M~\cite{radford2019gpt2}, Pythia-160M~\cite{biderman2023pythia}, SmolLM2-135M and SmolLM2-360M~\cite{benallal2025smollm2}, fine-tuned with our recipe: the same prompt text, loss masking, batch size, optimiser (our AdamW), schedule, epochs and epoch selection. For memory, the baselines accumulate gradients over token-weighted micro-batches; a test shows this gives the same gradient as the full batch to $10^{-6}$. As a control, our architecture is fine-tuned from random weights with the same recipe.

**After:** We fine-tuned models from GPT-2 124M~\cite{radford2019gpt2}, Pythia-160M~\cite{biderman2023pythia}, SmolLM2-135M and SmolLM2-360M~\cite{benallal2025smollm2} using the same prompt text, loss masking, batch size, AdamW optimiser, schedule, epochs and epoch selection. The baselines accumulate gradients over token-weighted micro-batches, which a test shows gives the same gradient as the full batch to $10^{-6}$. Our architecture was also fine-tuned from random weights with the same recipe.

## Unit 20 (line 174): KEPT ORIGINAL, placeholders changed (missing [1, 2, 4], found 3 of 6)

\textbf{Learning rate}, per model, from $\{5\times10^{-5}, 10^{-4}, 2\times10^{-4}\}$ (control: $\{10^{-4}, 10^{-3}\}$), chosen on the first 1,000 dev questions (Figure~\ref{fig:lr}; Table~\ref{tab:lr} in the appendix).

## Unit 21 (line 176): KEPT ORIGINAL, placeholders changed (missing [1], found 0 of 1)

\textbf{Test set, once:} the dev-phase weights had been deleted after scoring, so every system was fine-tuned again with its dev-chosen configuration and scored on test exactly once. No decision was made after seeing a test number.

## Unit 22 (line 186): rewritten

**Before:** Our main metric is execution accuracy: the predicted and gold queries are executed on the table, in a sandboxed read-only SQLite connection, and their results compared. We also run Salesforce's official WikiSQL evaluator, unmodified, which reports execution and logical-form accuracy. Fed the gold queries as predictions, it scores exactly 100\% on dev and test, and its execution accuracy agrees with ours within 0.11 points on every run of the four main systems.

**After:** Our main measure is how accurately the predicted queries match the gold queries when both are run against the table in a sandboxed read-only SQLite connection. We also use Salesforce's official WikiSQL evaluator, which gives scores for both execution and logical-form accuracy. When we feed the gold queries as predictions, it scores exactly 100\% on the dev and test sets. Its execution accuracy matches ours within 0.11 points across all runs of the four main systems.

## Unit 23 (line 188): KEPT ORIGINAL, placeholders changed (missing [1, 2, 3, 4, 5, 6], found 0 of 8)

For each system we report the mean and standard deviation over seeds and a 95\% percentile-bootstrap confidence interval~\cite{efron1993bootstrap} over questions, using 10,000 resamples of the seed-averaged per-question accuracy. Comparisons between systems pair their seed-0 predictions question by question, using a paired bootstrap~\cite{koehn2004significance} and the exact McNemar test~\cite{mcnemar1947} on the questions where exactly one system is correct, following~\cite{dror2018hitchhiker}.

## Unit 24 (line 236): KEPT ORIGINAL, placeholders changed (missing [1, 2, 3, 4, 5, 6], found 0 of 7)

\textbf{Pretraining is most of the result.} The same network fine-tuned from random weights reaches 37.4\%. Pretraining on 4.5B tokens adds 40.6 points, and constrained decoding cannot substitute for it (48.3\% even with both constraints).

## Unit 25 (line 237): KEPT ORIGINAL, placeholders changed (missing [1, 2], found 6 of 8)

\textbf{About one point behind GPT-2.} GPT-2 is ahead by 0.9 points at greedy ($p = 0.004$) and 2.0 points with both constraints. On dev the greedy gap was 0.5 points and not significant ($p = 0.26$), so we report the test result: GPT-2 is slightly better.

## Unit 26 (line 238): KEPT ORIGINAL, placeholders changed (missing [2, 4, 5, 6, 8], found 4 of 9)

\textbf{Far ahead of Pythia-160M,} by 17 points at greedy, despite Pythia's 67$\times$ larger pretraining budget. Pythia invents column names far more often (8.5\% of test questions produce invalid SQL, vs 2.0\% for ours). Our ablation (Section~\ref{sec:ablations}) suggests that the SQL-heavy mix explains only a small part of this gap, so we do not claim a cause.

## Unit 27 (line 239): KEPT ORIGINAL, placeholders changed (missing [1, 2, 6], found 4 of 7)

\textbf{Scale still wins at equal size:} SmolLM2-135M, with about 440$\times$ our pretraining tokens, is 4.6 points ahead (mean over seeds), and SmolLM2-360M is 9.1 points ahead.

## Unit 28 (line 240): KEPT ORIGINAL, placeholders changed (missing [1, 2, 3, 4, 5], found 0 of 5)

\textbf{Results are stable.} Seed standard deviations are 0.1--0.4 points, far smaller than every gap between models. Our dev-to-test drop is 0.7 points, similar to the baselines' 0.4--0.5 (Pythia, whose fine-tuning is unstable, drops 5 points).

## Unit 29 (line 280): KEPT ORIGINAL, placeholders changed (missing [1, 2, 3], found 0 of 3)

The value constraint is worth more than the schema constraint for every model: copying filter values exactly is where small models slip most. For our model, constrained decoding \emph{never} turned a correct answer into a wrong one in any paired comparison we ran (0 broken on test at both levels; 0 on dev). For every baseline it sometimes does, plausibly because their tokenizers can be forced into token splits they rarely produce; we did not verify this.

## Unit 30 (line 310): KEPT ORIGINAL, placeholders changed (missing [4, 5, 7, 8, 9, 10], found 6 of 16)

Table~\ref{tab:ablations} and Figure~\ref{fig:ablations} vary one design choice at a time on a small proxy model. Rotary embeddings matter most: without them, accuracy drops 8.5 points and pretraining loss is clearly worse (4.51 vs 4.09 nats on web text). Tying embeddings helps by 5 points even though the untied model has 50\% more parameters. Grouped-query attention costs nothing measurable while shrinking the KV cache 3$\times$. Most surprisingly, removing all SQL, code and Q\&A data from pretraining costs only 0.9 points after fine-tuning, even though the web-only model's loss on SQL code is far higher (4.40 vs 1.81 nats). Fine-tuning on 56k WikiSQL examples appears to recover most of what the SQL data teaches. These are single-seed results at 150M tokens, so we treat them as indicative.

## Unit 31 (line 322): KEPT ORIGINAL, placeholders changed (missing [1, 2, 3, 4], found 0 of 4)

Each wrong prediction is parsed back into WikiSQL's structure (selected column, aggregation, conditions) and compared with gold, giving exactly one category (Figure~\ref{fig:failures}). Compared with stronger models, our model loses most on condition columns, values and multi-part errors, the skills that depend on reading the table and the question. Aggregation errors are different: they stay at 3.8--4.3\% of test questions for every pretrained model, although overall accuracy spans 26 points.

## Unit 32 (line 324): rewritten

**Before:** Accuracy falls as the gold query needs more conditions (Figure~\ref{fig:conds}). On the 10,832 test questions with a single condition, our model and GPT-2 are level (80.3\% and 80.2\%, seed 0). The gap between them opens with two conditions (73.5\% vs 76.2\% on 3,879 questions), which matches the multi-part errors in Figure~\ref{fig:failures}. Pythia falls fastest, from 64.5\% to 35.8\% at four conditions.

**After:** Accuracy falls when the gold query needs more conditions (Figure~\ref{fig:conds}). On the 10,832 single-condition test questions, our model and GPT-2 are level (80.3\% and 80.2\%, seed 0). The gap widens with two conditions (73.5\% versus 76.2\% on 3,879 questions), which matches the multi-part errors in Figure~\ref{fig:failures}. Pythia drops fastest, from 64.5\% to 35.8\% at four conditions.

## Unit 33 (line 333): KEPT ORIGINAL, placeholders changed (missing [10, 11, 12, 13, 14, 15], found 9 of 26)

To find out why, we hand-checked 50 randomly sampled dev questions where our model was wrong \emph{only} in the aggregation. In 31 (62\%; 95\% Wilson CI~\cite{wilson1927} 48--74\%) the gold aggregation is not supported by the question. For example, ``How many people were in the crowd when Essendon was the home team?'' is labelled \texttt{COUNT(Crowd)} rather than the crowd figure, and ``For record 25-24, what is the sum of attendance?'' is labelled \texttt{COUNT}. A further 12 (24\%) are ambiguous, and only 7 (14\%) are clear model errors. Extrapolated, roughly 2 points of dev accuracy are lost to aggregation labels for every model, which is consistent with earlier reports of annotation errors in WikiSQL~\cite{hwang2019sqlova,pihulski2025llmsql}. The audit had a single annotator (the author), not blinded; all 50 judgements are released for re-checking.

## Unit 34 (line 338): KEPT ORIGINAL, placeholders changed (missing [1, 2], found 0 of 2)

\textbf{One benchmark, one query shape.} WikiSQL queries are single-table with no joins, grouping or nesting. We do not evaluate on harder benchmarks such as Spider~\cite{yu2018spider}.

## Unit 35 (line 339): KEPT ORIGINAL, placeholders changed (missing [1], found 1 of 2)

\textbf{Uncontrolled baseline pretraining.} The baselines differ in pretraining data, tokenizers and architecture, not just in token count, so Figure~\ref{fig:tokens} shows association, not a scaling law.

## Unit 36 (line 341): KEPT ORIGINAL, placeholders changed (missing [1, 2, 3, 4, 5, 6], found 0 of 6)

\textbf{Noisy selection.} Learning rates and epochs are selected on 1,000 dev questions, whose 95\% CI is about $\pm$2.8 points, wider than the differences between adjacent learning rates.

## Unit 37 (line 342): KEPT ORIGINAL, placeholders changed (missing [1], found 0 of 1)

\textbf{Reproducibility on GPU.} Fine-tuning is not bit-reproducible on GPU even with fixed seeds. Pythia's re-fine-tune for test, for example, differs from its dev run.

## Unit 38 (line 348): KEPT ORIGINAL, placeholders changed (missing [1, 2], found 2 of 4)

A single developer can build every component of a text-to-SQL language model, pretrain it on 4.5B tokens in two days on one GPU, and land within about one point of GPT-2 on WikiSQL while outperforming a model pretrained on 67$\times$ more data. Constrained decoding is a cheap and safe addition at this scale. Getting the last points still requires much more pretraining, and on WikiSQL a measurable share of the remaining errors are in the labels rather than in the models.

## Unit 39 (line 352): rewritten

**Before:** Code, configurations, all prediction-level results, statistics and figures are at \url{https://github.com/aditya0929/tinysql}. \texttt{PROJECT\_LOG.md} records every run, every number and every bug found along the way. Total compute was about 46 L4 GPU-hours for pretraining and about 45 for the benchmark (fine-tuning, evaluation and ablations), all on NVIDIA L4 GPUs.

**After:** The code, configurations, prediction-level results, statistics and figures are in \url{https://github.com/aditya0929/tinysql}. Every run, every number and every bug found along the way are recorded in \texttt{PROJECT\_LOG.md}. Pretraining took about 46 L4 GPU-hours, and the benchmark (fine-tuning, evaluation and ablations) took about 45, both on NVIDIA L4 GPUs.
