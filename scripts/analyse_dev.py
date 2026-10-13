"""Re-score every dev prediction file with the current executor, then compute confidence intervals, paired significance
tests and failure categories. Writes results/stats_dev.json and results/failures_dev.json.

    python -m scripts.analyse_dev
"""
import json
import os
from pathlib import Path

from data.wikisql_to_sql import BASE, load_split
from eval.evaluate_sql import score_predictions
from eval.execution_acc import WikiSQLExecutor
from eval.failure_analysis import analyse
from eval.stats import bootstrap_ci, mcnemar, paired_bootstrap

SYSTEMS = {
    "TinySQL-125M (ours)": "runs_local/rescored",
    "SmolLM2-135M": "runs_local/baselines/smollm2-135m",
    "SmolLM2-360M": "runs_local/baselines/smollm2-360m",
}
DECODES = ["greedy", "schema", "values"]


def main():
    items, tables = load_split("dev")
    executor = WikiSQLExecutor(os.path.join(BASE, "dev.db"), tables)
    correct, failures = {}, {}
    for name, d in SYSTEMS.items():
        for dec in DECODES:
            recs = [json.loads(l) for l in open(Path(d) / f"dev_{dec}_predictions.jsonl", encoding="utf-8")]
            assert len(recs) == len(items) and all(r["table_id"] == it["table_id"] for r, it in zip(recs, items))
            _, scored = score_predictions(items, tables, executor, [r["pred"] for r in recs])
            key = f"{name} / {dec}"
            correct[key] = [int(r["correct"]) for r in scored]
            failures[key], _ = analyse(scored, items, tables)
            print(f"{key:36s} {sum(correct[key]) / len(items):.4f}", flush=True)

    stats = {"n": len(items), "accuracy_ci95": {}, "pairwise": {}}
    for k, c in correct.items():
        acc, lo, hi = bootstrap_ci(c)
        stats["accuracy_ci95"][k] = {"acc": acc, "low": lo, "high": hi}
    pairs = []
    for dec in DECODES:                                                # ours vs each baseline, same decoding
        for b in ["SmolLM2-135M", "SmolLM2-360M"]:
            pairs.append((f"{b} / {dec}", f"TinySQL-125M (ours) / {dec}"))
    for name in SYSTEMS:                                               # effect of constraints within each model
        pairs.append((f"{name} / schema", f"{name} / greedy"))
        pairs.append((f"{name} / values", f"{name} / greedy"))
    for a, b in pairs:
        d, lo, hi, p = paired_bootstrap(correct[a], correct[b])
        oa, ob, pm = mcnemar(correct[a], correct[b])
        stats["pairwise"][f"{a}  vs  {b}"] = {"diff": d, "low": lo, "high": hi, "p_bootstrap": p,
                                               "only_a": oa, "only_b": ob, "p_mcnemar": pm}
    Path("results/stats_dev.json").write_text(json.dumps(stats, indent=1))
    Path("results/failures_dev.json").write_text(json.dumps(failures, indent=1))
    print(json.dumps(stats["pairwise"], indent=1))
    for k, f in failures.items():
        print(k, f["wrong"], f["categories"], "lucky", f["right_but_structurally_different"])


if __name__ == "__main__":
    main()
