import math
import random

from eval.stats import bootstrap_ci, mcnemar, mean_std, paired_bootstrap


def test_bootstrap_ci_brackets_the_accuracy_and_has_the_right_width():
    rng = random.Random(0)
    x = [rng.random() < 0.8 for _ in range(2000)]
    acc, lo, hi = bootstrap_ci(x, n_boot=2000)
    assert lo < acc < hi
    expected_half_width = 1.96 * math.sqrt(acc * (1 - acc) / len(x))        # normal approximation
    assert abs((hi - lo) / 2 - expected_half_width) < 0.004


def test_paired_bootstrap_identical_systems_show_no_difference():
    x = [1, 0, 1, 1, 0] * 200
    d, lo, hi, p = paired_bootstrap(x, x, n_boot=1000)
    assert d == 0 and lo == 0 and hi == 0 and p == 1.0


def test_paired_bootstrap_detects_a_real_difference():
    rng = random.Random(1)
    a = [rng.random() < 0.85 for _ in range(3000)]
    b = [x and rng.random() > 0.08 for x in a]                                 # b loses ~8% of a's correct answers
    d, lo, hi, p = paired_bootstrap(a, b, n_boot=2000)
    assert d > 0 and lo > 0 and p < 0.001


def test_mcnemar_matches_hand_computed_values():
    a = [1] * 10 + [0] * 2 + [1] * 5                                           # 10 only-a, 2 only-b, 5 both
    b = [0] * 10 + [1] * 2 + [1] * 5
    only_a, only_b, p = mcnemar(a, b)
    assert (only_a, only_b) == (10, 2)
    exact = 2 * sum(math.comb(12, k) for k in range(3)) / 2 ** 12               # = 0.0386
    assert abs(p - exact) < 1e-12
    assert mcnemar([1, 0, 1], [1, 0, 1])[2] == 1.0
    assert mcnemar([1] * 3 + [0] * 3, [0] * 3 + [1] * 3)[2] == 1.0             # balanced disagreement


def test_mcnemar_is_stable_for_large_samples():
    only_a, only_b, p = mcnemar([1] * 900 + [0] * 600, [0] * 900 + [1] * 600)
    assert (only_a, only_b) == (900, 600) and 0 < p < 1e-12


def test_mean_std():
    m, s = mean_std([0.78, 0.80, 0.79])
    assert abs(m - 0.79) < 1e-12 and abs(s - 0.01) < 1e-12
    assert mean_std([0.5]) == (0.5, 0.0)
