"""WikiSQL structured queries -> SQL text, and prompt/target pairs for fine-tuning.

    python -m data.wikisql_to_sql            # writes data/wikisql_sft/{train,dev,test}.jsonl
"""
import json
import os

AGG_OPS = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
COND_OPS = ["=", ">", "<", "OP"]
BASE = os.path.join("data", "raw", "wikisql", "extracted", "data")
OUT = os.path.join("data", "wikisql_sft")


def column_names(header):
    """Real header names; the (rare) empty header becomes colN so the SQL stays well formed."""
    return [h if h.strip() else f"col{i}" for i, h in enumerate(header)]


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def quote_value(value) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def schema_text(header, types) -> str:
    cols = ", ".join(f"{quote_ident(n)} {'REAL' if t == 'real' else 'TEXT'}" for n, t in zip(column_names(header), types))
    return f"CREATE TABLE t ({cols});"


def to_sql(sql: dict, header) -> str:
    names = [quote_ident(n) for n in column_names(header)]
    select = names[sql["sel"]]
    if sql["agg"]:
        select = f"{AGG_OPS[sql['agg']]}({select})"
    out = f"SELECT {select} FROM t"
    if sql["conds"]:
        out += " WHERE " + " AND ".join(f"{names[c]} {COND_OPS[o]} {quote_value(v)}" for c, o, v in sql["conds"])
    return out + ";"


def prompt_text(question: str, header, types) -> str:
    return f"<schema>{schema_text(header, types)}</schema>\n<question>{question.strip()}</question>\n<sql>"


def make_example(item: dict, table: dict) -> dict:
    return {
        "table_id": item["table_id"],
        "question": item["question"],
        "prompt": prompt_text(item["question"], table["header"], table["types"]),
        "completion": to_sql(item["sql"], table["header"]) + "</sql>",
    }


def load_split(split: str, base: str = BASE):
    tables = {}
    with open(os.path.join(base, f"{split}.tables.jsonl"), encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            tables[t["id"]] = t
    with open(os.path.join(base, f"{split}.jsonl"), encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    return items, tables


def build(split: str, base: str = BASE, out_dir: str = OUT) -> int:
    items, tables = load_split(split, base)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{split}.jsonl"), "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(make_example(it, tables[it["table_id"]]), ensure_ascii=False) + "\n")
    return len(items)


if __name__ == "__main__":
    for split in ("train", "dev", "test"):
        print(f"{split}: {build(split):,} examples")
