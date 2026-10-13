"""Categorise every wrong answer by what part of the query went wrong.

Each prediction is parsed into WikiSQL's structured form (eval/wikisql_official.parse_sql) and compared with the gold
query component by component. A wrong answer gets exactly one category:

  invalid             the SQL did not parse/execute
  select column       only the selected column is wrong
  aggregation         only the aggregation (none / MAX / MIN / COUNT / SUM / AVG) is wrong
  condition column    only a filtered column is wrong
  operator            only an operator (= > <) is wrong
  value               only a condition value is wrong
  missing condition   the prediction has fewer conditions than gold, everything present is right
  extra condition     the prediction has more conditions than gold, everything gold needs is right
  multiple            more than one of the above
  other               the query matches gold structurally but the result still differs (rare)

Separately counted: answers that are right by execution but structurally different from gold ("lucky").

    python -m eval.failure_analysis --pred runs_local/final/dev_values_predictions.jsonl
"""
import argparse
import json
from collections import Counter

from data.wikisql_to_sql import load_split
from eval.wikisql_official import parse_sql

CATEGORIES = ["invalid", "select column", "aggregation", "condition column", "operator", "value",
              "missing condition", "extra condition", "multiple", "other"]


def _norm(v):
    return str(v).strip().lower()


def diagnose(pred_sql: str, gold: dict, header):
    """-> one category name, or None if the structure matches gold exactly."""
    try:
        p = parse_sql(pred_sql, header)
    except ValueError:
        return "invalid"
    wrong = []
    if p["sel"] != gold["sel"]:
        wrong.append("select column")
    if p["agg"] != gold["agg"]:
        wrong.append("aggregation")
    gc = [(c, o, _norm(v)) for c, o, v in gold["conds"]]
    pc = [(c, o, _norm(v)) for c, o, v in p["conds"]]
    if sorted(gc) != sorted(pc):
        if len(pc) < len(gc) and set(pc) <= set(gc):
            wrong.append("missing condition")
        elif len(pc) > len(gc) and set(gc) <= set(pc):
            wrong.append("extra condition")
        elif len(pc) == len(gc):
            # pair conditions up by position after sorting by column, then see which field differs
            for (c1, o1, v1), (c2, o2, v2) in zip(sorted(gc), sorted(pc)):
                if c1 != c2:
                    wrong.append("condition column")
                elif o1 != o2:
                    wrong.append("operator")
                elif v1 != v2:
                    wrong.append("value")
        else:
            wrong.append("multiple")                              # different number of conditions AND different content
    wrong = list(dict.fromkeys(wrong))
    if not wrong:
        return None
    return wrong[0] if len(wrong) == 1 else "multiple"


def analyse(records, items, tables):
    """records: per-question dicts with 'pred' and 'correct', aligned with items."""
    cats, lucky, examples = Counter(), 0, {}
    for r, it in zip(records, items):
        header = tables[it["table_id"]]["header"]
        d = diagnose(r["pred"], it["sql"], header)
        if r["correct"]:
            lucky += d is not None
            continue
        cat = d or "other"
        cats[cat] += 1
        examples.setdefault(cat, []).append({"question": it["question"], "pred": r["pred"], "gold": r.get("gold")})
    n = len(records)
    return {"n": n, "wrong": sum(cats.values()), "categories": {c: cats.get(c, 0) for c in CATEGORIES},
            "right_but_structurally_different": lucky}, examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--split", default="dev")
    a = ap.parse_args()
    items, tables = load_split(a.split)
    recs = [json.loads(l) for l in open(a.pred, encoding="utf-8")]
    summary, _ = analyse(recs, items, tables)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
