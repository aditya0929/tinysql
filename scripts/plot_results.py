"""Result figures from results/stats_dev.json and results/failures_dev.json (run scripts/analyse_dev.py first).

  docs/figures/accuracy_vs_tokens.png   dev execution accuracy (greedy, 95% CI) against pretraining tokens, log scale
  docs/figures/failure_categories.png   what goes wrong in each model's wrong answers (greedy)

Pretraining token counts are from the model cards: SmolLM2-135M 2T, SmolLM2-360M 4T (huggingface.co/HuggingFaceTB),
Pythia-160M 299.9B (github.com/EleutherAI/pythia). GPT-2's token count was never published, so it is left off the
token axis. Ours: 8,600 steps x 524,288 tokens.

    python -m scripts.plot_results
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MODELS = [  # (name in stats_dev.json, label, parameters, pretraining tokens, colour)
    ("TinySQL-125M (ours)", "TinySQL-125M (ours)", 125.1e6, 8600 * 524288, BLUE),
    ("SmolLM2-135M", "SmolLM2-135M", 134.5e6, 2e12, ORANGE),
    ("SmolLM2-360M", "SmolLM2-360M", 361.8e6, 4e12, AQUA),
    ("Pythia-160M", "Pythia-160M", 162.3e6, 299.9e9, YELLOW),
]
OUT = Path("docs/figures")


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="both", alpha=0.25, linewidth=0.6)


def accuracy_vs_tokens(stats):
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=160)
    for key, label, params, tokens, colour in MODELS:
        ci = stats["accuracy_ci95"].get(f"{key} / greedy")
        if ci is None:
            continue                                                  # not evaluated yet
        acc = 100 * ci["acc"]
        ax.errorbar(tokens, acc, yerr=[[acc - 100 * ci["low"]], [100 * ci["high"] - acc]], fmt="o", ms=7,
                    color=colour, capsize=3)
        ax.annotate(f"{label}\n{params / 1e6:.0f}M params, {acc:.1f}%", (tokens, acc), textcoords="offset points",
                    xytext=(9, -4), fontsize=8, color=colour, va="center")
    ax.set_xscale("log")
    ax.set_xlim(1e9, 3e13)
    ax.set_xlabel("pretraining tokens (log scale)")
    ax.set_ylabel("WikiSQL dev execution accuracy (%)")
    ax.set_title("Accuracy against pretraining data (greedy decoding, 95% CI)", fontsize=10)
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "accuracy_vs_tokens.png")


def failure_categories(failures):
    cats = ["invalid", "select column", "aggregation", "condition column", "operator", "value",
            "missing condition", "extra condition", "multiple", "other"]
    present = [m for m in MODELS if f"{m[0]} / greedy" in failures]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    h = 0.8 / len(present)
    for i, (key, label, _, _, colour) in enumerate(present):
        f = failures[f"{key} / greedy"]
        ys = [j + (i - (len(present) - 1) / 2) * h for j in range(len(cats))]
        ax.barh(ys, [100 * f["categories"][c] / f["n"] for c in cats], height=h, color=colour,
                label=f"{label} ({100 * f['wrong'] / f['n']:.1f}% wrong)")
    ax.set_yticks(range(len(cats)), cats)
    ax.invert_yaxis()
    ax.set_xlabel("share of all dev questions (%)")
    ax.set_title("Where the wrong answers come from (greedy decoding)", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _style(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "failure_categories.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    accuracy_vs_tokens(json.loads(Path("results/stats_dev.json").read_text()))
    failure_categories(json.loads(Path("results/failures_dev.json").read_text()))


if __name__ == "__main__":
    main()
