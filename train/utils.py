import glob
import json
import os
import time

import torch


def param_groups(model, weight_decay: float):
    """Weight decay on 2-D weight matrices only: not on norm gains and not on the (tied) embedding table."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (decay if p.dim() >= 2 and not name.startswith("embed") else no_decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


class LossScaler:
    """Dynamic loss scaling for fp16 (bf16 and fp32 do not need it: pass enabled=False).
    Scale the loss up before backward so small fp16 gradients do not underflow, divide the gradients back before
    the optimizer step, halve the scale when an inf/nan appears, double it after `growth_interval` good steps."""

    def __init__(self, enabled: bool, init_scale: float = 2.0 ** 16, growth_interval: int = 2000):
        self.enabled, self.scale_value, self.growth_interval, self.good_steps = enabled, init_scale, growth_interval, 0

    def scale(self, loss):
        return loss * self.scale_value if self.enabled else loss

    def unscale_(self, params):
        if self.enabled:
            for p in params:
                if p.grad is not None:
                    p.grad.div_(self.scale_value)

    def update(self, found_inf: bool):
        if not self.enabled:
            return
        if found_inf:
            self.scale_value, self.good_steps = max(self.scale_value / 2, 1.0), 0
        else:
            self.good_steps += 1
            if self.good_steps >= self.growth_interval:
                self.scale_value, self.good_steps = self.scale_value * 2, 0

    def state_dict(self):
        return {"scale": self.scale_value, "good": self.good_steps}

    def load_state_dict(self, sd):
        self.scale_value, self.good_steps = sd["scale"], sd["good"]


# ----------------------------------------------------------------------------- checkpoints
def save_checkpoint(out_dir: str, step: int, state: dict, keep: int = 2) -> str:
    """Atomic write (temp file then rename) so a crash mid-save never corrupts the latest checkpoint."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"ckpt_{step:07d}.pt")
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)
    for old in sorted(glob.glob(os.path.join(out_dir, "ckpt_*.pt")))[:-keep]:
        os.remove(old)
    return path


def latest_checkpoint(out_dir: str):
    found = sorted(glob.glob(os.path.join(out_dir, "ckpt_*.pt")))
    return found[-1] if found else None


# ----------------------------------------------------------------------------- logging
class MetricsLog:
    """Prints, appends to metrics.jsonl, and (optionally) writes TensorBoard scalars."""

    def __init__(self, out_dir: str, tensorboard: bool = False):
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, "metrics.jsonl")
        self.tb = None
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(os.path.join(out_dir, "tb"))

    def log(self, step: int, **metrics):
        with open(self.path, "a") as f:
            f.write(json.dumps({"step": step, "time": time.time(), **metrics}) + "\n")
        if self.tb:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.tb.add_scalar(k, v, step)
