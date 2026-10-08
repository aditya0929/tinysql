import os

import numpy as np
import pytest

from data.loader import DEFAULT_SCHEDULE, SOURCES, MixedLoader, TokenLoader

SEQ = 8


@pytest.fixture(scope="module")
def shards(tmp_path_factory):
    """Tiny shards where every token of source i has the value i + 1, so a window identifies its source."""
    root = tmp_path_factory.mktemp("shards")
    for split in ("train", "val"):
        os.makedirs(root / split)
        for i, s in enumerate(SOURCES):
            for part in range(2):
                np.full(500, i + 1, dtype=np.uint16).tofile(root / split / f"{s}__p{part}.bin")
    return str(root)


def composition(x):
    return np.bincount((x[:, 0].numpy() - 1), minlength=len(SOURCES)) / len(x)


def test_weights_switch_between_phases():
    ld_weights = MixedLoader.weights_at
    class Stub: schedule = DEFAULT_SCHEDULE
    assert ld_weights(Stub, 0.0)["stack_sql"] == 0.15
    assert ld_weights(Stub, 0.84)["stack_sql"] == 0.15
    assert ld_weights(Stub, 0.85)["stack_sql"] == 0.36
    assert ld_weights(Stub, 1.0)["stack_sql"] == 0.36


def test_default_schedule_averages_to_the_documented_mix():
    avg = {s: sum((hi - lo) * w[s] for (hi, w), lo in zip(DEFAULT_SCHEDULE, [0.0, 0.85])) for s in SOURCES}
    assert abs(avg["fineweb_edu"] - 0.775) < 0.01 and abs(avg["stack_sql"] - 0.18) < 0.01
    assert abs(sum(avg.values()) - 1) < 1e-9


def test_batches_follow_the_phase_mix(shards):
    ld = MixedLoader("train", SEQ, shards_dir=shards, seed=1)
    early = np.mean([composition(ld.get_batch(200, progress=0.3)[0]) for _ in range(20)], axis=0)
    late = np.mean([composition(ld.get_batch(200, progress=0.95)[0]) for _ in range(20)], axis=0)
    assert np.allclose(early, [0.82, 0.15, 0.025, 0.005], atol=0.02)
    assert np.allclose(late, [0.54, 0.36, 0.08, 0.02], atol=0.02)


def test_rows_are_shuffled_across_sources_and_targets_are_shifted(shards):
    ld = MixedLoader("train", SEQ, shards_dir=shards, seed=2)
    x, y = ld.get_batch(64, progress=0.95)
    first_half = set(x[:32, 0].tolist())
    assert len(first_half) > 1                                   # a micro-batch is not a single source
    assert x.shape == y.shape == (64, SEQ) and bool((x[:, 1:] == y[:, :-1]).all())


def test_deterministic_and_exactly_resumable(shards):
    a = MixedLoader("train", SEQ, shards_dir=shards, seed=5)
    b = MixedLoader("train", SEQ, shards_dir=shards, seed=5)
    assert bool((a.get_batch(16, 0.5)[0] == b.get_batch(16, 0.5)[0]).all())
    state = a.state_dict()
    nxt = a.get_batch(16, 0.9)[0]
    a.load_state_dict(state)
    assert bool((a.get_batch(16, 0.9)[0] == nxt).all())


def test_per_source_validation_loader(shards):
    ld = TokenLoader("val", SEQ, shards_dir=shards, source="stack_sql", seed=0)
    x, _ = ld.get_batch(10)
    assert set(x.flatten().tolist()) == {2}
