"""Score every test prediction file with Salesforce's official WikiSQL evaluator (third_party/wikisql, unmodified).

First a sanity check: the gold queries, passed as predictions, must score exactly 100% logical form and execution.
Then for each system / decoding / seed: official execution and logical-form accuracy next to our executor's number.
Writes results/wikisql_test_official.json.

    python -m scripts.official_test
"""
import json
from pathlib import Path

from data.wikisql_to_sql import load_split, to_sql
from eval.wikisql_official import official_scores

T = Path("runs_local/test")


def main():
    items, tables = load_split("test")
    gold = [to_sql(it["sql"], tables[it["table_id"]]["header"]) for it in items]
    g = official_scores(gold, "test")
    print("gold as predictions:", g, flush=True)
    assert g["ex_accuracy"] == 1.0 and g["lf_accuracy"] == 1.0 and g["parsed"] == 1.0, g

    out = {"gold_check": g, "runs": {}}
    for path in sorted(T.glob("*/test_*_predictions.jsonl")):
        recs = [json.loads(l) for l in open(path, encoding="utf-8")]
        m = official_scores([r["pred"] for r in recs], "test")
        ours = sum(r["correct"] for r in recs) / len(recs)
        key = f"{path.parent.name} / {path.name.split('_')[1]}"
        out["runs"][key] = {"ours_exec": ours, "official_ex": m["ex_accuracy"], "official_lf": m["lf_accuracy"],
                            "parsed": m["parsed"]}
        print(f"{key:42s} ours {100 * ours:6.2f}  official ex {100 * m['ex_accuracy']:6.2f}  "
              f"lf {100 * m['lf_accuracy']:6.2f}  diff {100 * (m['ex_accuracy'] - ours):+.2f}", flush=True)
        Path("results/wikisql_test_official.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
