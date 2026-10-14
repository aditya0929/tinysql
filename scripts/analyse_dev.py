"""Statistics and failure analysis for every benchmark system on WikiSQL dev (8,421 questions).

Reads the prediction files written by scripts/benchmark_queue.py (downloaded to runs_local/benchmark/) and, for the
schema-only decoding of the three main models, the earlier full runs (runs_local/rescored, runs_local/baselines). All were
scored by the same executor (eval/execution_acc.py, after the identifier-lexer fix).

Per system and decoding: accuracy of each seed, mean and std over seeds, and a 95% bootstrap CI over questions of the
seed-averaged accuracy (each question counts as the fraction of seeds that got it right). Paired tests (paired bootstrap
and exact McNemar) compare seed-0 predictions question by question. Failure categories use seed 0, greedy.

Writes results/benchmark_dev.json and results/failures_dev.json.

    python -m scripts.analyse_dev
"""
import json
from pathlib import Path

from data.wikisql_to_sql import load_split
from eval.failure_analysis import analyse
from eval.stats import bootstrap_ci, mcnemar, mean_std, paired_bootstrap

B = Path("runs_local/benchmark")
# name -> {decoding: [prediction files, seed 0 first]}
SYSTEMS = {
    "TinySQL-125M (ours)": {
        "greedy": [B / f"tinysql_lr5e-05_s{s}/dev_greedy_predictions.jsonl" for s in range(3)],
        "schema": [Path("runs_local/rescored/dev_schema_predictions.jsonl")],
        "values": [B / f"tinysql_lr5e-05_s{s}/dev_values_predictions.jsonl" for s in range(3)]},
    "SmolLM2-135M": {
        "greedy": [B / f"smollm2-135m_lr0.0001_s{s}/dev_greedy_predictions.jsonl" for s in range(3)],
        "schema": [Path("runs_local/baselines/smollm2-135m/dev_schema_predictions.jsonl")],
        "values": [B / f"smollm2-135m_lr0.0001_s{s}/dev_values_predictions.jsonl" for s in range(3)]},
    "SmolLM2-360M": {
        "greedy": [B / f"smollm2-360m_lr5e-05_s{s}/dev_greedy_predictions.jsonl" for s in range(3)],
        "schema": [Path("runs_local/baselines/smollm2-360m/dev_schema_predictions.jsonl")],
        "values": [B / f"smollm2-360m_lr5e-05_s{s}/dev_values_predictions.jsonl" for s in range(3)]},
    "GPT-2 124M": {d: [B / f"gpt2_lr0.0001_s0/dev_{d}_predictions.jsonl"] for d in ("greedy", "schema", "values")},
    "Pythia-160M": {d: [B / f"pythia-160m_lr0.0001_s0/dev_{d}_predictions.jsonl"] for d in ("greedy", "schema", "values")},
    "No pretraining (ours, random init)": {d: [B / f"control_lr0.0001_s0/dev_{d}_predictions.jsonl"] for d in ("greedy", "values")},
}
# Where the schema-only run came from different weights than the best-LR seed 0, its paired test uses greedy from the
# SAME weights (the original LR-1e-4 runs).
SAME_WEIGHTS_GREEDY = {
    "TinySQL-125M (ours)": B / "tinysql_lr0.0001_s0/dev_greedy_predictions.jsonl",
    "SmolLM2-360M": B / "smollm2-360m_lr0.0001_s0/dev_greedy_predictions.jsonl",
}
ABLATIONS = ["base", "learned_pos", "mha", "untied", "general_only"]          # 25M proxy, greedy, 1 seed


def load(path, items):
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    assert len(recs) == len(items) and all(r["table_id"] == it["table_id"] for r, it in zip(recs, items)), path
    return recs


def main():
    items, tables = load_split("dev")
    n = len(items)
    out, seed0, failures = {"n": n, "systems": {}, "ablations": {}, "pairwise": {}}, {}, {}
    for name, decs in SYSTEMS.items():
        out["systems"][name] = {}
        for dec, paths in decs.items():
            runs = [[int(r["correct"]) for r in load(p, items)] for p in paths]
            accs = [sum(c) / n for c in runs]
            m, s = mean_std(accs)
            per_q = [sum(col) / len(runs) for col in zip(*runs)]
            _, lo, hi = bootstrap_ci(per_q)
            out["systems"][name][dec] = {"seeds": accs, "mean": m, "std": s, "ci_low": lo, "ci_high": hi}
            seed0[f"{name} / {dec}"] = runs[0]
            print(f"{name:36s} {dec:7s} {100 * m:6.2f} ± {100 * s:.2f}  [{100 * lo:.2f}, {100 * hi:.2f}]  seeds={len(runs)}")
        recs = load(decs["greedy"][0], items)
        failures[name], _ = analyse(recs, items, tables)

    for a in ABLATIONS:
        recs = load(B / f"abl_{a}/dev_greedy_predictions.jsonl", items)
        c = [int(r["correct"]) for r in recs]
        acc, lo, hi = bootstrap_ci(c)
        logs = [json.loads(l) for l in open(B / f"pretrain_{a}/metrics.jsonl")]
        val = [{k.split("/")[1]: round(v, 4) for k, v in r.items() if k.startswith("val_loss/")} for r in logs]
        val = [v for v in val if v]
        out["ablations"][a] = {"acc": acc, "ci_low": lo, "ci_high": hi, "final_val_loss": val[-1] if val else None}
        seed0[f"ablation {a}"] = c
        print(f"ablation {a:14s} {100 * acc:6.2f}  [{100 * lo:.2f}, {100 * hi:.2f}]  val loss {out['ablations'][a]['final_val_loss']}")

    ours = "TinySQL-125M (ours)"
    pairs = []
    for b in ["SmolLM2-135M", "SmolLM2-360M", "GPT-2 124M", "Pythia-160M"]:
        for dec in ("greedy", "values"):
            pairs.append((f"{b} / {dec}", f"{ours} / {dec}"))
    pairs.append((f"{ours} / greedy", "No pretraining (ours, random init) / greedy"))
    for name, path in SAME_WEIGHTS_GREEDY.items():
        seed0[f"{name} / greedy (LR 1e-4 run)"] = [int(r["correct"]) for r in load(path, items)]
    for name, decs in SYSTEMS.items():
        for dec in decs:
            if dec == "schema" and name in SAME_WEIGHTS_GREEDY:
                pairs.append((f"{name} / {dec}", f"{name} / greedy (LR 1e-4 run)"))
            elif dec != "greedy":
                pairs.append((f"{name} / {dec}", f"{name} / greedy"))
    pairs += [(f"ablation base", f"ablation {a}") for a in ABLATIONS[1:]]
    for a, b in pairs:
        d, lo, hi, p = paired_bootstrap(seed0[a], seed0[b])
        oa, ob, pm = mcnemar(seed0[a], seed0[b])
        out["pairwise"][f"{a}  vs  {b}"] = {"diff": d, "low": lo, "high": hi, "p_bootstrap": p,
                                             "only_a": oa, "only_b": ob, "p_mcnemar": pm}
        print(f"{a:44s} vs {b:44s} {100 * d:+6.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]  {oa}/{ob}  p={pm:.2g}")

    Path("results/benchmark_dev.json").write_text(json.dumps(out, indent=1))
    Path("results/failures_dev.json").write_text(json.dumps(failures, indent=1))
    for k, f in failures.items():
        print(f"{k:36s} wrong {f['wrong']:5d}", {c: v for c, v in f["categories"].items() if v})


if __name__ == "__main__":
    main()
