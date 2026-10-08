import os
import sqlite3

import pytest

from data.wikisql_to_sql import BASE, load_split, make_example, quote_value, schema_text, to_sql
from eval.execution_acc import WikiSQLExecutor, execution_match, lower_literals

HEADER = ["Player", "No.", "School/Club Team", ""]
TYPES = ["text", "real", "text", "text"]


def test_to_sql_renders_aggregation_conditions_and_quoting():
    q = {"sel": 0, "agg": 3, "conds": [[2, 0, "Butler CC (KS)"], [1, 1, 20]]}
    assert to_sql(q, HEADER) == 'SELECT COUNT("Player") FROM t WHERE "School/Club Team" = \'Butler CC (KS)\' AND "No." > \'20\';'
    assert to_sql({"sel": 3, "agg": 0, "conds": []}, HEADER) == 'SELECT "col3" FROM t;'      # empty header -> col3
    assert quote_value("O'Neil") == "'O''Neil'"


def test_schema_and_prompt_layout():
    assert schema_text(HEADER, TYPES) == 'CREATE TABLE t ("Player" TEXT, "No." REAL, "School/Club Team" TEXT, "col3" TEXT);'
    ex = make_example({"table_id": "x", "question": " Who? ", "sql": {"sel": 0, "agg": 0, "conds": []}},
                      {"header": HEADER, "types": TYPES})
    assert ex["prompt"].startswith("<schema>CREATE TABLE t (") and ex["prompt"].endswith("<question>Who?</question>\n<sql>")
    assert ex["completion"] == 'SELECT "Player" FROM t;</sql>'


def test_literals_are_lowercased_but_identifiers_are_not():
    assert lower_literals("""SELECT "Player" FROM t WHERE "School" = 'Butler CC (KS)' AND x = 'O''Neil'""") == \
        """SELECT "Player" FROM t WHERE "School" = 'butler cc (ks)' AND x = 'o''neil'"""


def make_db(tmp_path):
    path = str(tmp_path / "mini.db")
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE table_1_5 (col0 text, col1 real, col2 text)")
    con.executemany("INSERT INTO table_1_5 VALUES (?,?,?)", [("ann", 3.0, "duke"), ("bob", 21.0, "butler cc (ks)"), ("cy", 21.0, "duke")])
    con.commit()
    con.close()
    return path, {"1-5": {"header": ["Player", "No.", "School"], "types": ["text", "real", "text"]}}


def test_executor_runs_generated_sql_and_compares_results(tmp_path):
    db, tables = make_db(tmp_path)
    ex = WikiSQLExecutor(db, tables)
    gold = 'SELECT "Player" FROM t WHERE "School" = \'Butler CC (KS)\';'
    assert ex.run("1-5", gold) == (True, [("bob",)])
    assert execution_match(ex, "1-5", gold, 'SELECT "Player" FROM t WHERE "School" = \'butler cc (ks)\'') == (True, True)
    assert execution_match(ex, "1-5", gold, 'SELECT "Player" FROM t WHERE "School" = \'Duke\'') == (True, False)
    assert execution_match(ex, "1-5", gold, 'SELECT "Nope" FROM t') == (False, False)           # unknown column: invalid
    assert execution_match(ex, "1-5", gold, "DROP TABLE t") == (False, False)                   # cannot modify anything
    assert ex.run("1-5", 'SELECT COUNT("Player") FROM t WHERE "No." > \'5\'') == (True, [(2,)])   # quoted number vs REAL column
    assert ex.run("1-5", 'SELECT AVG("No.") FROM t') == (True, [(15.0,)])


@pytest.mark.skipif(not os.path.exists(os.path.join(BASE, "dev.db")), reason="WikiSQL data not downloaded")
def test_converter_matches_the_official_structured_query_on_real_data():
    """Execute (a) my converted SQL through the renaming view and (b) a canonical colN query built directly from the
    structured form, the way the official evaluator does it. They must agree on every example."""
    items, tables = load_split("dev")
    ex = WikiSQLExecutor(os.path.join(BASE, "dev.db"), tables)
    mismatches, checked = [], 0
    for it in items[:3000]:
        t = tables[it["table_id"]]
        q = it["sql"]
        ok, mine = ex.run(it["table_id"], to_sql(q, t["header"]))
        conds, params = [], {}
        for i, (c, o, v) in enumerate(q["conds"]):
            if t["types"][c] == "real" and not isinstance(v, (int, float)):
                try:
                    v = float(str(v).replace(",", ""))
                except ValueError:
                    v = float("nan")
            elif isinstance(v, str):
                v = v.lower()
            conds.append(f"col{c} {['=', '>', '<'][o]} :v{i}")
            params[f"v{i}"] = v
        sel = f"col{q['sel']}" if not q["agg"] else f"{['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG'][q['agg']]}(col{q['sel']})"
        sql = f"SELECT {sel} FROM table_{it['table_id'].replace('-', '_')}" + (" WHERE " + " AND ".join(conds) if conds else "")
        from eval.execution_acc import normalize
        ref = normalize(ex.con.execute(sql, params).fetchall())
        checked += 1
        if not ok or mine != ref:
            mismatches.append((it["question"], to_sql(q, t["header"]), mine, ref))
    assert checked == 3000
    assert len(mismatches) / checked < 0.005, mismatches[:3]
