import math


def cosine_lr(step: int, max_steps: int, peak_lr: float, warmup_steps: int, min_lr_ratio: float = 0.1) -> float:
    """Linear warmup to peak_lr, then cosine decay to min_lr_ratio * peak_lr at max_steps."""
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, max_steps - warmup_steps))
    min_lr = peak_lr * min_lr_ratio
    return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))
