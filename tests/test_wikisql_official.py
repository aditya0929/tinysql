import os

import pytest

from data.wikisql_to_sql import BASE, load_split, to_sql
from eval.wikisql_official import official_scores, parse_sql

HEADER = ["Player", "No.", "Driver's Age", ""]
needs_data = pytest.mark.skipif(not os.path.exists(os.path.join(BASE, "dev.jsonl")), reason="WikiSQL data not downloaded")


def test_parse_simple_and_aggregated_queries():
    assert parse_sql('SELECT "Player" FROM t;', HEADER) == {"sel": 0, "agg": 0, "conds": []}
    q = parse_sql('SELECT COUNT("No.") FROM t WHERE "Driver\'s Age" > \'24\' AND "col3" = \'O\'\'Neil\';', HEADER)
    assert q == {"sel": 1, "agg": 3, "conds": [[2, 1, "24"], [3, 0, "O'Neil"]]}


@pytest.mark.parametrize("bad", [
    'SELECT "Nope" FROM t;',                       # unknown column
    'SELECT COUNT("Player" FROM t;',               # missing parenthesis
    'SELECT "Player" FROM users;',                 # wrong table
    'SELECT "Player" FROM t WHERE "No." != \'1\';',  # unsupported operator
    'SELECT "Player" FROM t WHERE "No." = \'1',    # unterminated value
    'DROP TABLE t;',
])
def test_unparseable_predictions_raise(bad):
    with pytest.raises(ValueError):
        parse_sql(bad, HEADER)


@needs_data
def test_every_gold_query_round_trips_to_the_structured_form():
    items, tables = load_split("dev")
    for it in items:
        header = tables[it["table_id"]]["header"]
        got = parse_sql(to_sql(it["sql"], header), header)
        want = it["sql"]
        assert (got["sel"], got["agg"]) == (want["sel"], want["agg"]), it["question"]
        assert [[c, o, str(v)] for c, o, v in got["conds"]] == [[c, o, str(v)] for c, o, v in want["conds"]], it["question"]


@needs_data
def test_official_evaluator_scores_gold_as_perfect():
    items, tables = load_split("dev")
    gold = [to_sql(it["sql"], tables[it["table_id"]]["header"]) for it in items]
    m = official_scores(gold, "dev")
    assert m["parsed"] == 1.0 and m["ex_accuracy"] == 1.0 and m["lf_accuracy"] == 1.0, m
