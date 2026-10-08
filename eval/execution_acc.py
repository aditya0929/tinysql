"""Execute generated SQL against the official WikiSQL SQLite databases and compare results (execution accuracy).

The databases name their columns col0, col1, ... and store text lower-cased. For every table we expose a temporary
VIEW called `t` that renames the columns to the real header names, so the model's SQL runs unchanged. String literals
in the SQL are lower-cased before execution to match the lower-cased data (this is also what the official evaluator does).
"""
import re
import sqlite3
import time

from data.wikisql_to_sql import column_names, quote_ident

_STRING = re.compile(r"'((?:[^']|'')*)'")
_READ_ONLY_ACTIONS = (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION)


def _authorizer(action, *_):
    """While model-generated SQL runs, allow reading only: anything else (INSERT, DROP, CREATE, ATTACH, PRAGMA...) is denied."""
    return sqlite3.SQLITE_OK if action in _READ_ONLY_ACTIONS else sqlite3.SQLITE_DENY
_NUM_PRECISION = 6


def lower_literals(sql: str) -> str:
    return _STRING.sub(lambda m: "'" + m.group(1).lower() + "'", sql)


def normalize(rows):
    """Order-insensitive, float-tolerant canonical form of a result set."""
    out = []
    for row in rows:
        out.append(tuple(round(v, _NUM_PRECISION) if isinstance(v, float) else v for v in row))
    return sorted(out, key=repr)


class WikiSQLExecutor:
    def __init__(self, db_path: str, tables: dict, timeout_s: float = 2.0):
        self.con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)      # the database file itself is read-only
        # SQLite quirk: an unknown "double-quoted name" silently becomes a string literal. Turn that off so a
        # hallucinated column is a real error and the validity rate is honest.
        self.con.setconfig(sqlite3.SQLITE_DBCONFIG_DQS_DML, False)
        self.con.setconfig(sqlite3.SQLITE_DBCONFIG_DQS_DDL, False)
        self.tables = tables
        self.timeout_s = timeout_s

    def _use_table(self, table_id: str):
        t = self.tables[table_id]
        cols = ", ".join(f"col{i} AS {quote_ident(n)}" for i, n in enumerate(column_names(t["header"])))
        self.con.execute("DROP VIEW IF EXISTS temp.t")
        self.con.execute(f"CREATE TEMP VIEW t AS SELECT {cols} FROM table_{table_id.replace('-', '_')}")

    def run(self, table_id: str, sql: str):
        """-> (ok, normalized rows or error text). Never raises on bad SQL; aborts runaway queries."""
        try:
            self.con.set_authorizer(None)
            self._use_table(table_id)                       # rebuilt for every query, so a stray DROP VIEW cannot break later ones
            self.con.set_authorizer(_authorizer)
            deadline = time.time() + self.timeout_s
            self.con.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 20_000)
            rows = self.con.execute(lower_literals(sql.strip().rstrip(";"))).fetchall()
            return True, normalize(rows)
        except Exception as e:                      # syntax error, unknown column, timeout, ...
            return False, f"{type(e).__name__}: {e}"
        finally:
            self.con.set_progress_handler(None, 0)
            self.con.set_authorizer(None)


def execution_match(executor: WikiSQLExecutor, table_id: str, gold_sql: str, pred_sql: str):
    """-> (valid, correct). valid = predicted SQL executed; correct = same result set as the gold SQL."""
    ok_g, gold = executor.run(table_id, gold_sql)
    ok_p, pred = executor.run(table_id, pred_sql)
    return ok_p, bool(ok_g and ok_p and gold == pred)
