"""Plot the pretraining curves from a metrics.jsonl (the one synced to the bucket or runs/main/metrics.jsonl).

    python scripts/plot_loss.py --metrics runs_local/main/metrics.jsonl --out docs/figures/loss_curves.png

Four small charts, each with its own axis (no dual axes): training loss, per-source validation loss,
learning rate, gradient norm. Colors come from a validated categorical palette (slots 1-4).
"""
import argparse
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
BLUE_TINT = "#9ec5f4"
SOURCES = [("fineweb_edu", "FineWeb-Edu", BLUE), ("stack_sql", "SQL code", ORANGE),
           ("stackexchange", "Stack Exchange", AQUA), ("gretel_sql", "Gretel", YELLOW)]
LN_VOCAB = math.log(32768)
MIX_SWITCH_STEP = 7310        # 85% of 8,600: the data mix becomes SQL-heavy here (SQL 15% -> 36%)


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def smooth(values, steps=None, half=4, from_step=200):
    """Centered moving average (window 2*half+1, shrinking at the ends). Unlike an EMA it does not lag the curve.
    Before `from_step` the loss is falling too steeply for a window to be honest, so raw values are kept."""
    out = []
    for i in range(len(values)):
        if steps is not None and steps[i] < from_step:
            out.append(values[i])
            continue
        lo, hi = max(0, i - half), min(len(values), i + half + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10, fontweight="semibold")
    ax.set_xlabel(xlabel, fontsize=9, color=INK2)
    ax.set_ylabel(ylabel, fontsize=9, color=INK2)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out", default="docs/figures/loss_curves.png")
    args = ap.parse_args()
    rows = load(args.metrics)
    train = [r for r in rows if "loss" in r]
    vals = [r for r in rows if "val_loss/fineweb_edu" in r]
    steps = [r["step"] for r in train]
    toks = [r["tokens"] / 1e9 for r in train]
    last = train[-1]

    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.2), dpi=150, facecolor=SURFACE)
    (a, b), (c, d) = axes

    # A: training loss
    style(a, "Training loss", "tokens seen (billions, log scale)", "cross-entropy (nats per token)")
    a.set_xscale("log")
    a.plot(toks, [r["loss"] for r in train], color=BLUE_TINT, linewidth=0.9, label="each logged step")
    a.plot(toks, smooth([r["loss"] for r in train], steps), color=BLUE, linewidth=2.0, label="smoothed")
    a.axhline(LN_VOCAB, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
    a.text(0.0015, LN_VOCAB + 0.14, "random guessing: ln(32,768) = 10.4", fontsize=8.5, color=INK2, va="bottom")
    a.annotate(f"{smooth([r['loss'] for r in train], steps)[-1]:.2f}", (toks[-1], smooth([r["loss"] for r in train], steps)[-1]), xytext=(6, 8),
               textcoords="offset points", fontsize=9, color=INK)
    if last["step"] > MIX_SWITCH_STEP:
        sw = MIX_SWITCH_STEP * 524288 / 1e9
        a.axvline(sw, color=ORANGE, linewidth=1.2, linestyle=(0, (4, 3)))
        a.annotate("mix switch: SQL share 15% to 36%\n(the loss drop is not learning)", (sw, 6.2), xytext=(-8, 0),
                   textcoords="offset points", fontsize=8.5, color=INK2, ha="right", va="center")
    a.set_ylim(2.0, 11)
    a.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower left", bbox_to_anchor=(0.02, 0.14))

    # B: validation loss per source
    style(b, "Validation loss by source (every 500 steps)", "training step", "cross-entropy (nats per token)")
    vs = [r["step"] for r in vals]
    for key, label, color in SOURCES:
        ys = [r[f"val_loss/{key}"] for r in vals]
        b.plot(vs, ys, color=color, linewidth=2.0, marker="o", markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=1.2, label=label)
        b.annotate(f"{label}  {ys[-1]:.2f}", (vs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points", fontsize=8.5, color=INK, va="center")
    b.set_xlim(vs[0] - 100, vs[-1] + (vs[-1] - vs[0]) * 0.42)
    b.set_xticks([v for v in vs if v % 1000 == 0] or vs)
    b.set_ylim(0.9, 4.6)
    b.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left", ncol=4, bbox_to_anchor=(0.0, 1.0))

    # C: learning rate
    style(c, "Learning rate", "training step", "learning rate")
    c.plot(steps, [r["lr"] for r in train], color=BLUE, linewidth=2.0)
    c.set_ylim(0, 1.4e-3)
    if last["step"] > MIX_SWITCH_STEP:
        c.axvline(MIX_SWITCH_STEP, color=ORANGE, linewidth=1.2, linestyle=(0, (4, 3)))
    c.text(steps[0] + 10, 1.22e-3, "peak 1.2e-3 after a 200-step warmup", fontsize=8.5, color=INK2)

    # D: gradient norm
    style(d, "Gradient norm (before clipping)", "training step", "global L2 norm")
    d.set_yscale("log")
    d.set_ylim(0.1, 12)
    if last["step"] > MIX_SWITCH_STEP:
        d.axvline(MIX_SWITCH_STEP, color=ORANGE, linewidth=1.2, linestyle=(0, (4, 3)))
    d.plot(steps, [r["grad_norm"] for r in train], color=BLUE, linewidth=1.6)
    d.axhline(1.0, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
    d.text(1500, 1.12, "clip threshold 1.0", fontsize=8.5, color=INK2, ha="left")

    fig.suptitle(f"TinySQL-125M pretraining, steps 1 to {last['step']:,} of 8,600  ({last['tokens'] / 1e9:.2f}B tokens)",
                 x=0.012, y=0.985, ha="left", fontsize=13, color=INK, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, facecolor=SURFACE)
    print("saved", args.out)


if __name__ == "__main__":
    main()
