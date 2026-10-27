"""Extra figures for the technical report (paper/figures/).

    python -m scripts.plot_paper
"""
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.wikisql_to_sql import load_split

BLUE, ORANGE, AQUA, YELLOW, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#7a7a7a"
COLOURS = {"TinySQL-125M (ours)": BLUE, "SmolLM2-135M": ORANGE, "SmolLM2-360M": AQUA, "Pythia-160M": YELLOW,
           "GPT-2 124M": GREY, "No pretraining (ours, random init)": "black"}
OUT = Path("paper/figures")


def _spread(ys, gap):
    """Label positions: the same order as ys, at least `gap` apart (end-of-line labels must not overlap)."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    pos = [0.0] * len(ys)
    last = None
    for i in order:
        pos[i] = ys[i] if last is None else max(ys[i], last + gap)
        last = pos[i]
    return pos


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="both", alpha=0.25, linewidth=0.6)


def data_mix():
    # two-phase schedule (PROJECT_LOG section 11): first 85% of 8,600 steps, then the SQL-heavy phase
    phases = [(0, 7310, [82, 15, 2.5, 0.5]), (7310, 8600, [54, 36, 8, 2])]
    names = ["FineWeb-Edu (web)", "The Stack (SQL code)", "Stack Exchange (Q&A)", "Gretel (synthetic text-to-SQL)"]
    cols = [BLUE, ORANGE, AQUA, YELLOW]
    fig, ax = plt.subplots(figsize=(7.0, 2.9), dpi=170)
    for a, b, shares in phases:
        left = 0
        for share, c in zip(shares, cols):
            ax.barh(0, b - a, left=a, height=0.0, color=c)              # placeholder for legend order
        bottom = 0
        for share, c in zip(shares, cols):
            ax.fill_between([a, b], bottom, bottom + share, color=c, linewidth=0)
            if share >= 8:
                ax.text((a + b) / 2, bottom + share / 2, f"{share:g}%", ha="center", va="center", color="white",
                        fontsize=8.5, fontweight="bold")
            bottom += share
    ax.axvline(7310, color="black", lw=0.8, ls="--")
    ax.text(7250, 103, "mix switch (85%)", ha="right", fontsize=8)
    ax.set_xlim(0, 8600)
    ax.set_ylim(0, 100)
    ax.set_xlabel("pretraining step (524,288 tokens each)")
    ax.set_ylabel("share of sampled tokens (%)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cols]
    ax.legend(handles, names, frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "data_mix.png")


def lr_sweep():
    # first-1,000-dev-question scores at selection time (Pythia's 1e-4 run was later repeated; this is the original)
    sweep = {"TinySQL-125M (ours)": [74.3, 73.4, 68.8], "SmolLM2-135M": [78.0, 78.3, 75.1],
             "SmolLM2-360M": [84.6, 84.5, 80.5], "GPT-2 124M": [72.4, 75.2, 74.5], "Pythia-160M": [56.9, 58.5, 45.7]}
    lrs = [5e-5, 1e-4, 2e-4]
    fig, ax = plt.subplots(figsize=(6.2, 3.8), dpi=170)
    labels = _spread([ys[2] for ys in sweep.values()], 2.2)
    for (name, ys), y_label in zip(sweep.items(), labels):
        c = COLOURS[name]
        ax.plot(lrs, ys, marker="o", color=c, lw=1.8)
        best = max(range(3), key=lambda i: ys[i])
        ax.plot(lrs[best], ys[best], marker="o", ms=11, mfc="none", mec=c, mew=1.5)
        ax.text(2.1e-4, y_label, name, color=c, fontsize=7.5, va="center")
    ax.set_xscale("log")
    ax.set_xticks(lrs, ["5e-5", "1e-4", "2e-4"])
    ax.minorticks_off()
    ax.set_xlim(4.2e-5, 4.2e-4)
    ax.set_xlabel("peak fine-tuning learning rate")
    ax.set_ylabel("execution accuracy, first 1,000 dev (%)")
    ax.set_title("Learning-rate sweep (best epoch; circled = chosen)", fontsize=10)
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "lr_sweep.png")


def decoding_gains(res):
    order = ["Pythia-160M", "TinySQL-125M (ours)", "GPT-2 124M", "SmolLM2-135M", "SmolLM2-360M"]
    fig, ax = plt.subplots(figsize=(6.6, 3.4), dpi=170)
    for j, name in enumerate(order):
        r = res["systems"][name]
        g, s, v = (100 * r[d]["mean"] for d in ("greedy", "schema", "values"))
        c = COLOURS[name]
        ax.plot([g, v], [j, j], color=c, lw=2.5, alpha=0.35, solid_capstyle="round")
        ax.plot(g, j, "o", mfc="white", mec=c, mew=1.6, ms=7)
        ax.plot(s, j, "s", color=c, ms=5)
        ax.plot(v, j, "o", color=c, ms=8)
        ax.text(v + 0.6, j, f"+{v - g:.1f}", va="center", fontsize=8, color=c)
    ax.set_yticks(range(len(order)), order, fontsize=8.5)
    ax.set_xlabel("WikiSQL test execution accuracy (%)")
    ax.set_xlim(58, 92)
    ax.set_title("Constrained decoding: ○ greedy  ■ + schema  ● + schema + values", fontsize=9.5)
    _style(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "decoding_gains.png")


def accuracy_by_conditions():
    items, _ = load_split("test")
    nconds = [len(it["sql"]["conds"]) for it in items]
    runs = {"TinySQL-125M (ours)": "tinysql_lr5e-05_s0", "GPT-2 124M": "gpt2_lr0.0001_s0", "Pythia-160M": "pythia-160m_lr0.0001_s0",
            "SmolLM2-135M": "smollm2-135m_lr0.0001_s0", "SmolLM2-360M": "smollm2-360m_lr5e-05_s0"}
    ks = [1, 2, 3, 4]
    counts = {k: sum(1 for n in nconds if n == k) for k in ks}
    fig, ax = plt.subplots(figsize=(6.2, 3.8), dpi=170)
    accs = {}
    for name, job in runs.items():
        recs = [json.loads(l) for l in open(f"runs_local/test/{job}/test_greedy_predictions.jsonl", encoding="utf-8")]
        accs[name] = [100 * sum(r["correct"] for r, n in zip(recs, nconds) if n == k) / counts[k] for k in ks]
    for (name, acc), y_label in zip(accs.items(), _spread([a[-1] for a in accs.values()], 3.0)):
        c = COLOURS[name]
        ax.plot(ks, acc, marker="o", color=c, lw=1.8)
        ax.text(4.1, y_label, name, color=c, fontsize=7.5, va="center")
    ax.set_xticks(ks, [f"{k}\n(n={counts[k]:,})" for k in ks])
    ax.set_xlim(0.8, 5.3)
    ax.set_xlabel("WHERE conditions in the gold query")
    ax.set_ylabel("test execution accuracy (%)")
    ax.set_title("Accuracy falls with query complexity (greedy, seed 0)", fontsize=10)
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "accuracy_by_conditions.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data_mix()
    lr_sweep()
    decoding_gains(json.loads(Path("results/benchmark_test.json").read_text()))
    accuracy_by_conditions()
    shutil.copy("docs/figures/ablations.png", OUT / "ablations.png")


if __name__ == "__main__":
    main()
