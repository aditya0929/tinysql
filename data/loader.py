"""Batch loader over the packed token shards.

Documents are packed back to back (separated by <|endoftext|>), so there is no padding: a training example is any
window of seq_len + 1 consecutive tokens. Inputs are tokens[:-1] and targets tokens[1:]. Windows are drawn uniformly
over ALL tokens, so the data mix equals the mix of the shards.
"""
import glob
import os

import numpy as np
import torch


class TokenLoader:
    def __init__(self, split: str, seq_len: int, shards_dir: str = os.path.join("data", "shards"),
                 source: str | None = None, seed: int = 0):
        pattern = os.path.join(shards_dir, split, (f"{source}__" if source else "") + "*.bin")
        self.arrays = [np.memmap(p, dtype=np.uint16, mode="r") for p in sorted(glob.glob(pattern))]
        self.arrays = [a for a in self.arrays if len(a) > seq_len + 1]
        if not self.arrays:
            raise FileNotFoundError(f"no usable shards for {pattern}")
        self.seq_len = seq_len
        # number of valid window starts in each file; a window never crosses a file boundary
        self.starts = np.array([len(a) - seq_len - 1 for a in self.arrays], dtype=np.int64)
        self.cum = np.cumsum(self.starts)
        self.rng = np.random.default_rng(seed)

    @property
    def num_tokens(self) -> int:
        return int(sum(len(a) for a in self.arrays))

    def get_batch(self, batch_size: int):
        """-> (x, y), each (batch_size, seq_len) int64."""
        idx = self.rng.integers(0, self.cum[-1], size=batch_size)
        files = np.searchsorted(self.cum, idx, side="right")
        offsets = idx - np.where(files > 0, self.cum[files - 1], 0)
        windows = np.stack([self.arrays[f][o:o + self.seq_len + 1] for f, o in zip(files, offsets)]).astype(np.int64)
        windows = torch.from_numpy(windows)
        return windows[:, :-1], windows[:, 1:]

    # exact resume: save and restore the sampler state in checkpoints
    def state_dict(self):
        return {"rng": self.rng.bit_generator.state}

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state["rng"]


# ----------------------------------------------------------------------------- training-time mixing
SOURCES = ["fineweb_edu", "stack_sql", "stackexchange", "gretel_sql"]

# (phase ends at this fraction of training, sampling weights). Specialised data is concentrated near the end,
# while the learning rate is decaying. Averaged over the run: ~78% web, ~18% SQL code, ~3.3% Q&A, ~0.7% Gretel.
DEFAULT_SCHEDULE = [
    (0.85, {"fineweb_edu": 0.82, "stack_sql": 0.15, "stackexchange": 0.025, "gretel_sql": 0.005}),
    (1.00, {"fineweb_edu": 0.54, "stack_sql": 0.36, "stackexchange": 0.08, "gretel_sql": 0.02}),
]


class MixedLoader:
    """Draws every batch from the four sources according to the mix at the current training progress."""

    def __init__(self, split: str, seq_len: int, schedule=DEFAULT_SCHEDULE, shards_dir: str = os.path.join("data", "shards"),
                 seed: int = 0):
        for _, w in schedule:
            assert abs(sum(w.values()) - 1) < 1e-6, "mix weights must sum to 1"
        self.schedule = schedule
        self.sources = sorted({s for _, w in schedule for s in w})
        self.loaders = {s: TokenLoader(split, seq_len, shards_dir, source=s, seed=seed + i) for i, s in enumerate(self.sources)}
        self.rng = np.random.default_rng(seed + 1000)

    def weights_at(self, progress: float):
        for until, w in self.schedule:
            if progress < until:
                return w
        return self.schedule[-1][1]

    def get_batch(self, batch_size: int, progress: float = 0.0):
        """progress in [0, 1] = fraction of training completed. Returns (x, y), rows in random source order."""
        w = self.weights_at(progress)
        counts = self.rng.multinomial(batch_size, [w.get(s, 0.0) for s in self.sources])
        xs, ys = [], []
        for s, c in zip(self.sources, counts):
            if c:
                x, y = self.loaders[s].get_batch(int(c))
                xs.append(x)
                ys.append(y)
        order = torch.from_numpy(self.rng.permutation(batch_size))      # so micro-batches are not single-source
        return torch.cat(xs)[order], torch.cat(ys)[order]

    def state_dict(self):
        return {"rng": self.rng.bit_generator.state, "loaders": {s: l.state_dict() for s, l in self.loaders.items()}}

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state["rng"]
        for s, l in self.loaders.items():
            l.load_state_dict(state["loaders"][s])
