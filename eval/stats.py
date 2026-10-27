"""Statistics for comparing systems evaluated on the SAME questions (paired data), written from scratch."""
import math

import numpy as np


def _mean(xs):
    return sum(xs) / len(xs)


def _resampled_means(x, n_boot, seed, chunk=500):
    """Means of n_boot resamples (with replacement) of x, sorted. Vectorised in chunks to bound memory."""
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    out = np.concatenate([x[rng.integers(0, len(x), size=(min(chunk, n_boot - i), len(x)))].mean(axis=1)
                          for i in range(0, n_boot, chunk)])
    return np.sort(out).tolist()


def bootstrap_ci(correct, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0):
    """-> (accuracy, low, high). `correct` may also hold per-question fractions (e.g. averaged over seeds)."""
    x = [float(c) for c in correct]
    stats = _resampled_means(x, n_boot, seed)
    return _mean(x), stats[int(alpha / 2 * n_boot)], stats[int((1 - alpha / 2) * n_boot) - 1]


def paired_bootstrap(a, b, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0):
    """-> (difference a - b, low, high, two-sided p-value). Resamples questions, keeping each question's pair together."""
    d = [int(x) - int(y) for x, y in zip(a, b)]
    diffs = _resampled_means(d, n_boot, seed)
    obs = _mean(d)
    # p-value: how often the resampled difference lands on the other side of zero (shifted-null form)
    centered = [x - obs for x in diffs]
    p = sum(1 for x in centered if abs(x) >= abs(obs)) / n_boot
    return obs, diffs[int(alpha / 2 * n_boot)], diffs[int((1 - alpha / 2) * n_boot) - 1], max(p, 1 / n_boot)


def _log_binom_pmf(k: int, n: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) - n * math.log(2)


def mcnemar(a, b):
    """-> (only_a_right, only_b_right, exact two-sided p-value). Under the null, discordant pairs split 50/50."""
    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    only_b = sum(1 for x, y in zip(a, b) if y and not x)
    n = only_a + only_b
    if n == 0:
        return 0, 0, 1.0
    k = min(only_a, only_b)
    logs = [_log_binom_pmf(i, n) for i in range(k + 1)]
    m = max(logs)
    tail = math.exp(m) * sum(math.exp(v - m) for v in logs)
    return only_a, only_b, min(1.0, 2 * tail)


def mean_std(values):
    """-> (mean, sample standard deviation); std is 0 for a single value."""
    m = _mean(values)
    if len(values) < 2:
        return m, 0.0
    return m, math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))
