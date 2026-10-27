"""Result figures from results/benchmark_dev.json and results/failures_dev.json (run scripts/analyse_dev.py first).

    python -m scripts.plot_results
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, YELLOW, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#7a7a7a"
MODELS = [  # (name in benchmark_dev.json, parameters, pretraining tokens or None, colour)
    ("TinySQL-125M (ours)", 125.1e6, 8600 * 524288, BLUE),
    ("SmolLM2-135M", 134.5e6, 2e12, ORANGE),
    ("SmolLM2-360M", 361.8e6, 4e12, AQUA),
    ("Pythia-160M", 162.3e6, 299.9e9, YELLOW),
    ("GPT-2 124M", 124.4e6, None, GREY),
]
OUT = Path("docs/figures")


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="both", alpha=0.25, linewidth=0.6)


def accuracy_vs_tokens(res, split="dev", suffix=""):
    sysres = res["systems"]
    fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=160)
    for name, params, tokens, colour in MODELS:
        g = sysres[name]["greedy"]
        acc = 100 * g["mean"]
        if tokens is None:
            ax.axhline(acc, color=colour, lw=1, ls="--")
            ax.text(1.5e10, acc + 0.8, f"{name}: {acc:.1f}% (tokens unpublished)", fontsize=7.5, color=colour)
            continue
        ax.errorbar(tokens, acc, yerr=[[acc - 100 * g["ci_low"]], [100 * g["ci_high"] - acc]], fmt="o", ms=7,
                    color=colour, capsize=3)
        seeds = len(g["seeds"])
        above = name == "SmolLM2-135M"                                 # keeps its label clear of the GPT-2 line
        ax.annotate(f"{name}\n{acc:.1f}%" + (f" (mean of {seeds} seeds)" if seeds > 1 else ""), (tokens, acc),
                    textcoords="offset points", xytext=(-9, 14) if above else (9, -12), fontsize=7.5, color=colour,
                    va="center", ha="right" if above else "left")
    ctrl = 100 * sysres["No pretraining (ours, random init)"]["greedy"]["mean"]
    ax.axhline(ctrl, color="black", lw=1, ls=":")
    ax.text(1.1e9, ctrl + 0.8, f"same model, no pretraining: {ctrl:.1f}%", fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlim(1e9, 6e13)
    ax.set_ylim(30, 92)
    ax.set_xlabel("pretraining tokens (log scale)")
    ax.set_ylabel(f"WikiSQL {split} execution accuracy (%)")
    ax.set_title("Accuracy against pretraining data (greedy decoding, 95% CI)", fontsize=10)
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUT / f"accuracy_vs_tokens{suffix}.png")


def failure_categories(failures, split="dev", suffix=""):
    cats = ["invalid", "select column", "aggregation", "condition column", "operator", "value",
            "missing condition", "extra condition", "multiple", "other"]
    fig, ax = plt.subplots(figsize=(7.4, 5.6), dpi=160)
    h = 0.8 / len(MODELS)
    for i, (name, _, _, colour) in enumerate(MODELS):
        f = failures[name]
        ys = [j + (i - (len(MODELS) - 1) / 2) * h for j in range(len(cats))]
        ax.barh(ys, [100 * f["categories"][c] / f["n"] for c in cats], height=h, color=colour,
                label=f"{name} ({100 * f['wrong'] / f['n']:.1f}% wrong)")
    ax.set_yticks(range(len(cats)), cats)
    ax.invert_yaxis()
    ax.set_xlabel(f"share of all {split} questions (%)")
    ax.set_title("Where the wrong answers come from (greedy decoding, seed 0)", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.4, -0.1), ncol=3)
    _style(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / f"failure_categories{suffix}.png")


def ablations(res):
    labels = {"base": "base (RoPE, GQA, tied, mixed data)", "learned_pos": "learned positions instead of RoPE",
              "mha": "full multi-head attention", "untied": "untied input/output embeddings",
              "general_only": "web text only (no SQL/code)"}
    rows = list(labels)
    fig, ax = plt.subplots(figsize=(6.8, 3.0), dpi=160)
    for j, k in enumerate(rows):
        a = res["ablations"][k]
        acc = 100 * a["acc"]
        ax.barh(j, acc, color=BLUE if k == "base" else GREY, height=0.6)
        ax.errorbar(acc, j, xerr=[[acc - 100 * a["ci_low"]], [100 * a["ci_high"] - acc]], color="black", capsize=3, lw=1)
        ax.text(100 * a["ci_high"] + 0.6, j, f"{acc:.1f}%", va="center", fontsize=8)
    ax.set_yticks(range(len(rows)), [labels[k] for k in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 65)
    ax.set_xlabel("WikiSQL dev execution accuracy (%), 25M-parameter proxy, greedy")
    ax.set_title("Ablations: one change at a time from the base design", fontsize=10)
    _style(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "ablations.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = json.loads(Path("results/benchmark_dev.json").read_text())
    accuracy_vs_tokens(res)
    failure_categories(json.loads(Path("results/failures_dev.json").read_text()))
    ablations(res)
    if Path("results/benchmark_test.json").exists():                    # test: same figures, no ablations
        accuracy_vs_tokens(json.loads(Path("results/benchmark_test.json").read_text()), "test", "_test")
        failure_categories(json.loads(Path("results/failures_test.json").read_text()), "test", "_test")


if __name__ == "__main__":
    main()
