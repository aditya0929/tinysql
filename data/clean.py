"""Clean the raw corpora: data/raw/<source>/*.parquet -> data/clean/<source>/*.parquet

Every output row has: text, source, license, origin.  Each filter is a small pure function that returns
(text_or_None, reason). A dropped document is counted under its reason, so the report shows exactly what
each filter removed.

    python -m data.clean                      # all sources, all CPU cores
    python -m data.clean --only stack_sql     # one source
    python -m data.clean --limit-batches 2    # smoke test on the first 2 batches of each file
"""
import argparse
import glob
import json
import os
import re
import time
import unicodedata
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import pyarrow as pa
import pyarrow.parquet as pq

RAW, CLEAN = os.path.join("data", "raw"), os.path.join("data", "clean")
SCHEMA = pa.schema([("text", pa.string()), ("source", pa.string()), ("license", pa.string()), ("origin", pa.string())])
BATCH = 2000

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def normalize(text: str) -> str:
    """NFC (not NFKC: NFKC would rewrite symbols inside SQL string literals), unix newlines, no control chars."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return _CONTROL.sub("", text).strip()


# ----------------------------------------------------------------------------- web text (FineWeb-Edu)
def clean_web(text: str):
    t = normalize(text)
    n = len(t)
    if n < 200:
        return None, "too_short"
    if t.count("�") / n > 0.01:
        return None, "replacement_chars"           # broken encoding
    if sum(map(str.isalpha, t)) / n < 0.55:
        return None, "low_alpha_fraction"          # tables, number dumps, markup soup
    lines = [l for l in t.split("\n") if l.strip()]
    if len(lines) >= 10 and 1 - len(set(lines)) / len(lines) > 0.3:
        return None, "repeated_lines"              # boilerplate repeated over and over
    return t, None


# ----------------------------------------------------------------------------- SQL code (The Stack)
_SQL_KEYWORD = re.compile(r"\b(select|create|insert|update|delete|alter|drop)\b", re.I)
_INSERT_LINE = re.compile(r"^\s*insert\b", re.I | re.M)
DUMP_KEEP_CHARS = 8000


def clean_sql(content: str):
    """Returns (text, reason). reason is a drop reason, or 'truncated_dump' when the text was kept but shortened."""
    t = normalize(content)
    if len(t) < 50:
        return None, "too_short"
    lines = t.split("\n")
    reason = None
    if len(_INSERT_LINE.findall(t)) / len(lines) > 0.5:         # INSERT data dump: keep only the head (schema)
        t = t[:DUMP_KEEP_CHARS].rsplit("\n", 1)[0]
        lines = t.split("\n")
        reason = "truncated_dump"
    if max(map(len, lines)) > 1000:
        return None, "long_lines"                               # minified / generated / one-line data
    if sum(map(str.isalnum, t)) / max(1, len(t)) < 0.3:
        return None, "low_alnum_fraction"
    if not _SQL_KEYWORD.search(t):
        return None, "no_sql_keyword"
    return t, reason


# ----------------------------------------------------------------------------- Stack Exchange
def clean_se(thread_text: str, score):
    if score is not None and score < 0:
        return None, "negative_score"
    t = normalize(thread_text or "")
    if len(t) < 300:
        return None, "too_short"
    return t, None


# ----------------------------------------------------------------------------- Gretel synthetic text-to-SQL
_CREATE_TABLE = re.compile(r"CREATE\s+TABLE\b[^;]*;", re.I | re.S)


def format_gretel(sql_context: str, sql_prompt: str, sql: str):
    """Teach the exact prompt layout used later: <schema>..</schema><question>..</question><sql>..</sql>.
    Only the CREATE TABLE statements are kept as the schema (INSERT rows are dropped)."""
    tables = _CREATE_TABLE.findall(sql_context or "")
    if not tables or not sql_prompt or not sql:
        return None, "missing_field"
    schema = "\n".join(" ".join(t.split()) for t in tables)
    return f"<schema>{schema}</schema>\n<question>{normalize(sql_prompt)}</question>\n<sql>{normalize(sql)}</sql>", None


# ----------------------------------------------------------------------------- per-file driver
SOURCES = {
    "fineweb_edu": {"glob": "sample/10BT/*.parquet", "columns": ["text", "url"]},
    "stack_sql": {"glob": "data/sql/*.parquet", "columns": ["content", "max_stars_repo_licenses", "max_stars_repo_name"]},
    "stackexchange": {"glob": "*.stackexchange.com/*.parquet", "columns": ["ThreadText", "Score", "ContentLicense", "Id"]},
    "gretel_sql": {"glob": "synthetic_text_to_sql_train*.parquet", "columns": ["sql_context", "sql_prompt", "sql", "id"]},
}


def process_row(source: str, r: dict, path: str):
    """-> (row_or_None, reason)"""
    if source == "fineweb_edu":
        text, why = clean_web(r["text"])
        return (None if text is None else (text, "ODC-By", r["url"] or "")), why
    if source == "stack_sql":
        text, why = clean_sql(r["content"])
        return (None if text is None else (text, str(r["max_stars_repo_licenses"]), r["max_stars_repo_name"] or "")), why
    if source == "stackexchange":
        text, why = clean_se(r["ThreadText"], r["Score"])
        site = os.path.basename(os.path.dirname(path))
        return (None if text is None else (text, r["ContentLicense"] or "CC BY-SA", f"{site}:{r['Id']}")), why
    text, why = format_gretel(r["sql_context"], r["sql_prompt"], r["sql"])
    return (None if text is None else (text, "Apache-2.0", f"gretel:{r['id']}")), why


def clean_file(task):
    source, path, limit_batches = task
    out_dir = os.path.join(CLEAN, source)
    os.makedirs(out_dir, exist_ok=True)
    tag = os.path.basename(os.path.dirname(path)) + "_" if source == "stackexchange" else ""
    out_path = os.path.join(out_dir, tag + os.path.basename(path))
    stats = {"source": source, "file": os.path.basename(path), "docs_in": 0, "docs_out": 0,
             "chars_in": 0, "chars_out": 0, "reasons": Counter()}
    if limit_batches is None and os.path.exists(out_path + ".stats.json"):
        with open(out_path + ".stats.json") as f:
            return json.load(f)                                  # already done: resume support
    tmp = out_path + ".tmp"
    pf = pq.ParquetFile(path)
    key_col = SOURCES[source]["columns"][0]
    with pq.ParquetWriter(tmp, SCHEMA, compression="zstd") as writer:
        for b, batch in enumerate(pf.iter_batches(batch_size=BATCH, columns=SOURCES[source]["columns"])):
            if limit_batches is not None and b >= limit_batches:
                break
            rows_out = {"text": [], "source": [], "license": [], "origin": []}
            for r in batch.to_pylist():
                stats["docs_in"] += 1
                stats["chars_in"] += len(r[key_col] or "") if source != "gretel_sql" else 0
                row, why = process_row(source, r, path)
                if row is None:
                    stats["reasons"][why] += 1
                    continue
                if why:
                    stats["reasons"][why] += 1                   # kept but modified (e.g. truncated_dump)
                text, lic, origin = row
                stats["docs_out"] += 1
                stats["chars_out"] += len(text)
                for k, v in zip(("text", "source", "license", "origin"), (text, source, lic, origin)):
                    rows_out[k].append(v)
            writer.write_table(pa.table(rows_out, schema=SCHEMA))
    os.replace(tmp, out_path)
    stats["reasons"] = dict(stats["reasons"])
    if limit_batches is None:
        with open(out_path + ".stats.json", "w") as f:
            json.dump(stats, f)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(SOURCES))
    ap.add_argument("--limit-batches", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    tasks = []
    for name in args.only or SOURCES:
        for path in sorted(glob.glob(os.path.join(RAW, name, SOURCES[name]["glob"]))):
            tasks.append((name, path, args.limit_batches))
    print(f"{len(tasks)} files, {args.workers} workers")
    t0, all_stats = time.time(), []
    with ProcessPoolExecutor(args.workers) as pool:
        for s in pool.map(clean_file, tasks):
            all_stats.append(s)
            print(f"  {s['source']:14s} {s['file'][:44]:44s} {s['docs_in']:>9,} -> {s['docs_out']:>9,}", flush=True)

    print(f"\ndone in {time.time() - t0:.0f}s\n")
    print(f"{'source':14s} {'docs in':>11s} {'docs out':>11s} {'kept':>6s} {'MB in':>9s} {'MB out':>9s}   drop/modify reasons")
    report = {}
    for name in args.only or SOURCES:
        rows = [s for s in all_stats if s["source"] == name]
        if not rows:
            continue
        agg = Counter()
        for s in rows:
            agg.update(s["reasons"])
        d_in, d_out = sum(s["docs_in"] for s in rows), sum(s["docs_out"] for s in rows)
        c_in, c_out = sum(s["chars_in"] for s in rows), sum(s["chars_out"] for s in rows)
        report[name] = {"docs_in": d_in, "docs_out": d_out, "chars_in": c_in, "chars_out": c_out, "reasons": dict(agg)}
        print(f"{name:14s} {d_in:>11,} {d_out:>11,} {d_out / max(1, d_in):>6.1%} {c_in / 1e6:>9,.0f} {c_out / 1e6:>9,.0f}   {dict(agg.most_common())}")
    if args.limit_batches is None:
        with open(os.path.join(CLEAN, "report.json"), "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
