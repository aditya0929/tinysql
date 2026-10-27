"""Score predictions with the OFFICIAL WikiSQL evaluator (third_party/wikisql, unmodified Salesforce code).

    python -m eval.wikisql_official --pred runs_local/final/dev_values_predictions.jsonl --split dev
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

from data.wikisql_to_sql import AGG_OPS, BASE, COND_OPS, column_names, load_split
from eval.execution_acc import lex

OFFICIAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "wikisql")
_HEAD = re.compile(r"^\s*SELECT\s+(?:(MAX|MIN|COUNT|SUM|AVG)\s*\(\s*)?$", re.I)
_FROM = re.compile(r"^\s*(\))?\s*FROM\s+t\s*(WHERE\s*)?$", re.I)
_OP = re.compile(r"^\s*(=|>|<)\s*$")
_JOIN = re.compile(r"^\s*(AND\s*)?$", re.I)


def parse_sql(sql: str, header):
    """Our SQL text -> {"sel", "agg", "conds"} in WikiSQL's structured form. Raises ValueError if it does not fit."""
    parts = lex(sql.strip().rstrip(";").strip())
    if parts and parts[-1][0] == "code" and not parts[-1][1].strip():
        parts = parts[:-1]
    names = column_names(header)

    def col_index(p):
        if p[0] != "ident" or not p[2] or p[1] not in names:
            raise ValueError(f"not a column of this table: {p}")
        return names.index(p[1])

    if len(parts) < 3 or parts[0][0] != "code":
        raise ValueError("does not start with SELECT")
    head = _HEAD.match(parts[0][1])
    if not head:
        raise ValueError("bad SELECT clause")
    agg = AGG_OPS.index(head.group(1).upper()) if head.group(1) else 0
    sel = col_index(parts[1])
    if parts[2][0] != "code" or not (m := _FROM.match(parts[2][1])) or bool(m.group(1)) != bool(agg):
        raise ValueError("bad FROM clause")
    conds, rest = [], parts[3:]
    if m.group(2):
        while True:
            if len(rest) < 3 or rest[1][0] != "code" or rest[2][0] != "string" or not rest[2][2]:
                raise ValueError("bad condition")
            op = _OP.match(rest[1][1])
            if not op:
                raise ValueError("bad operator")
            conds.append([col_index(rest[0]), COND_OPS.index(op.group(1)), rest[2][1]])
            rest = rest[3:]
            if not rest:
                break
            if rest[0][0] != "code" or not (j := _JOIN.match(rest[0][1])) or not j.group(1):
                raise ValueError("bad condition separator")
            rest = rest[1:]
    elif rest:
        raise ValueError("unexpected text after FROM t")
    return {"sel": sel, "agg": agg, "conds": conds}


def official_scores(preds, split="dev"):
    """preds: SQL strings aligned with the split's .jsonl lines. Runs the official evaluate.py; returns its metrics."""
    items, tables = load_split(split)
    assert len(preds) == len(items), "predictions must align with the dataset"
    n_parsed = 0
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for it, sql in zip(items, preds):
            try:
                f.write(json.dumps({"query": parse_sql(sql, tables[it["table_id"]]["header"])}) + "\n")
                n_parsed += 1
            except ValueError as e:
                f.write(json.dumps({"error": str(e)}) + "\n")
        pred_file = f.name
    try:
        # UTF-8 mode: the official script opens files with the platform default encoding (cp1252 on Windows),
        # which cannot read WikiSQL's non-English text; on Linux UTF-8 is already the default.
        # The official code parses numbers with babel, which takes its locale from the environment; the original
        # evaluation ran with an English locale ("1,200.5" -> 1200.5), so set that explicitly (Windows sets none).
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "LC_NUMERIC": "en_US.UTF-8", "LANG": "en_US.UTF-8"}
        out = subprocess.run([sys.executable, "evaluate.py", os.path.abspath(os.path.join(BASE, f"{split}.jsonl")),
                              os.path.abspath(os.path.join(BASE, f"{split}.db")), pred_file],
                             cwd=OFFICIAL, capture_output=True, encoding="utf-8", errors="replace", env=env)
    finally:
        os.remove(pred_file)
    if out.returncode != 0:
        raise RuntimeError("official evaluate.py failed:\n" + "\n".join(out.stderr.strip().splitlines()[-12:]))
    metrics = json.loads(out.stdout[out.stdout.index("{"):])
    metrics["parsed"] = n_parsed / len(preds)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="our predictions .jsonl (field 'pred'), aligned with the split")
    ap.add_argument("--split", default="dev")
    a = ap.parse_args()
    preds = [json.loads(line)["pred"] for line in open(a.pred, encoding="utf-8")]
    m = official_scores(preds, a.split)
    print(json.dumps({"pred_file": a.pred, **m}, indent=1))


if __name__ == "__main__":
    main()
