"""Execute generated SQL against the official WikiSQL SQLite databases and compare results (execution accuracy).

The databases name their columns col0, col1, ... and store text lower-cased. For every table we expose a temporary
VIEW called `t` that renames the columns to the real header names, so the model's SQL runs unchanged. String literals
in the SQL are lower-cased before execution to match the lower-cased data (this is also what the official evaluator does).
"""
import sqlite3
import time

from data.wikisql_to_sql import column_names, quote_ident

_READ_ONLY_ACTIONS = (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION)


def _authorizer(action, *_):
    """While model-generated SQL runs, allow reading only: anything else (INSERT, DROP, CREATE, ATTACH, PRAGMA...) is denied."""
    return sqlite3.SQLITE_OK if action in _READ_ONLY_ACTIONS else sqlite3.SQLITE_DENY


def _allow_all(*_):
    """Used to lift the read-only guard. (set_authorizer(None) only works from Python 3.11, so it is not used.)"""
    return sqlite3.SQLITE_OK


_NUM_PRECISION = 6


def lex(sql: str):
    """Split SQL into ("code", text), ("ident", name, closed) and ("string", text, closed) pieces, reading left to
    right so that an apostrophe inside a "Driver's Age" identifier, or a double quote inside a 'string', is handled
    correctly (regexes for one quote kind cannot see the other). Doubled quotes are escapes in both."""
    out, buf, i, n = [], [], 0, len(sql)
    while i < n:
        ch = sql[i]
        if ch not in "'\"":
            buf.append(ch)
            i += 1
            continue
        if buf:
            out.append(("code", "".join(buf)))
            buf = []
        j, content = i + 1, []
        while j < n:
            if sql[j] == ch:
                if j + 1 < n and sql[j + 1] == ch:
                    content.append(ch)
                    j += 2
                    continue
                break
            content.append(sql[j])
            j += 1
        out.append(("ident" if ch == '"' else "string", "".join(content), j < n))
        i = j + 1
    if buf:
        out.append(("code", "".join(buf)))
    return out


def lower_literals(sql: str) -> str:
    """Lower-case the contents of 'string literals' only (identifiers and keywords keep their case)."""
    parts = []
    for p in lex(sql):
        if p[0] == "code":
            parts.append(p[1])
        else:
            q = '"' if p[0] == "ident" else "'"
            text = p[1] if p[0] == "ident" else p[1].lower()
            parts.append(q + text.replace(q, q + q) + (q if p[2] else ""))
    return "".join(parts)


def unknown_identifiers(sql: str, allowed) -> list:
    """Double-quoted names in the SQL that are not real columns (text inside 'string literals' is ignored).
    SQLite would otherwise silently treat an unknown "name" as a string, hiding a hallucinated column."""
    names = {p[1] for p in lex(sql) if p[0] == "ident"}
    return sorted(n for n in names if n not in allowed and n != "t")


def normalize(rows):
    """Order-insensitive, float-tolerant canonical form of a result set."""
    out = []
    for row in rows:
        out.append(tuple(round(v, _NUM_PRECISION) if isinstance(v, float) else v for v in row))
    return sorted(out, key=repr)


class WikiSQLExecutor:
    def __init__(self, db_path: str, tables: dict, timeout_s: float = 2.0):
        self.con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)      # the database file itself is read-only
        # SQLite quirk: an unknown "double-quoted name" silently becomes a string literal. Turn that off where the
        # Python version allows it (3.12+); unknown_identifiers() below enforces the same rule on any version.
        if hasattr(self.con, "setconfig"):
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
            bad = unknown_identifiers(sql, set(column_names(self.tables[table_id]["header"])))
            if bad:
                return False, f"unknown column(s): {bad}"
            self.con.set_authorizer(_allow_all)
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
            self.con.set_authorizer(_allow_all)


def execution_match(executor: WikiSQLExecutor, table_id: str, gold_sql: str, pred_sql: str):
    """-> (valid, correct). valid = predicted SQL executed; correct = same result set as the gold SQL."""
    ok_g, gold = executor.run(table_id, gold_sql)
    ok_p, pred = executor.run(table_id, pred_sql)
    return ok_p, bool(ok_g and ok_p and gold == pred)
