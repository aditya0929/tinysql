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
