import pytest

from eval.failure_analysis import diagnose

H = ["Player", "No.", "School", "Year"]
GOLD = {"sel": 0, "agg": 0, "conds": [[2, 0, "Duke"], [3, 1, "1990"]]}
BASE = 'SELECT "Player" FROM t WHERE "School" = \'duke\' AND "Year" > \'1990\';'


@pytest.mark.parametrize("pred, category", [
    (BASE, None),                                                                          # matches (case-insensitive values, any order)
    ('SELECT "Player" FROM t WHERE "Year" > \'1990\' AND "School" = \'DUKE\';', None),
    ('SELECT "Nope" FROM t;', "invalid"),
    ('SELECT "No." FROM t WHERE "School" = \'duke\' AND "Year" > \'1990\';', "select column"),
    ('SELECT COUNT("Player") FROM t WHERE "School" = \'duke\' AND "Year" > \'1990\';', "aggregation"),
    ('SELECT "Player" FROM t WHERE "No." = \'duke\' AND "Year" > \'1990\';', "condition column"),
    ('SELECT "Player" FROM t WHERE "School" = \'duke\' AND "Year" < \'1990\';', "operator"),
    ('SELECT "Player" FROM t WHERE "School" = \'duke university\' AND "Year" > \'1990\';', "value"),
    ('SELECT "Player" FROM t WHERE "School" = \'duke\';', "missing condition"),
    ('SELECT "Player" FROM t WHERE "School" = \'duke\' AND "Year" > \'1990\' AND "No." = \'4\';', "extra condition"),
    ('SELECT COUNT("No.") FROM t WHERE "School" = \'duke\' AND "Year" > \'1990\';', "multiple"),
    ('SELECT "Player" FROM t WHERE "No." = \'4\';', "multiple"),
])
def test_diagnose(pred, category):
    assert diagnose(pred, GOLD, H) == category
