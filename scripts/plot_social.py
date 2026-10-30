"""Plain-language versions of two result figures for sharing (docs/linkedin/), from results/benchmark_test.json.

Highlight design: TinySQL in the accent colour, the other models in neutral ink, every model labelled directly.

    python -m scripts.plot_social
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ACCENT, OTHER, INK, MUTED, SURFACE = "#2a78d6", "#8a8984", "#0b0b0b", "#52514e", "#fcfcfb"
OUT = Path("docs/linkedin")
OURS = "TinySQL-125M (ours)"
PEERS = {"GPT-2 124M": None, "SmolLM2-135M": 2e12, "Pythia-160M": 299.9e9}      # pretraining tokens (GPT-2: not published)
GRAMMAR = "grammar check (constrained decoding)"


def _axes(fig, ax):
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c3")
    ax.tick_params(colors=MUTED)
    ax.grid(alpha=0.35, linewidth=0.6, color="#d9d8d3")


def accuracy_vs_data(S):
    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=200)
    _axes(fig, ax)
    pts = [(OURS, 8600 * 524288)] + [(k, v) for k, v in PEERS.items() if v]
    for name, tok in pts:
        acc = 100 * S[name]["greedy"]["mean"]
        c = ACCENT if name == OURS else OTHER
        ax.plot(tok, acc, "o", ms=10 if name == OURS else 8, color=c, mec=SURFACE, mew=2, zorder=3)
        label = "TinySQL (ours)" if name == OURS else name
        ax.annotate(f"{label}  {acc:.1f}%", (tok, acc), textcoords="offset points", xytext=(10, -14 if name == OURS else 8),
                    fontsize=9, color=INK, fontweight="bold" if name == OURS else "normal")
    gpt2 = 100 * S["GPT-2 124M"]["greedy"]["mean"]
    ax.axhline(gpt2, color=OTHER, lw=1.2, ls="--", zorder=1)
    ax.text(1.5e10, gpt2 + 1.0, f"GPT-2 124M  {gpt2:.1f}%  (training data size not published)", fontsize=8.5, color=MUTED)
    ctrl = 100 * S["No pretraining (ours, random init)"]["greedy"]["mean"]
    ax.axhline(ctrl, color=OTHER, lw=1.2, ls=":", zorder=1)
    ax.text(1.15e9, ctrl + 1.0, f"Same model with no pretraining  {ctrl:.1f}%", fontsize=8.5, color=MUTED)
    ax.set_xscale("log")
    ax.set_xlim(1e9, 1e13)
    ax.set_ylim(30, 90)
    ax.set_xlabel("Pretraining data (tokens, log scale)", color=MUTED)
    ax.set_ylabel("WikiSQL test accuracy (%)", color=MUTED)
    fig.suptitle("TinySQL vs similar-sized models", x=0.08, ha="left", fontsize=13, color=INK, fontweight="bold")
    ax.set_title(f"Test accuracy without the {GRAMMAR}, after identical fine-tuning", loc="left", fontsize=9, color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT / "1_accuracy_vs_pretraining_data.png", facecolor=SURFACE)


def grammar_gains(S):
    rows = ["Pythia-160M", OURS, "GPT-2 124M", "SmolLM2-135M"]          # low to high
    fig, ax = plt.subplots(figsize=(7.6, 3.8), dpi=200)
    _axes(fig, ax)
    ax.grid(axis="y", visible=False)
    for j, name in enumerate(rows):
        g, v = 100 * S[name]["greedy"]["mean"], 100 * S[name]["values"]["mean"]
        c = ACCENT if name == OURS else OTHER
        ax.plot([g, v], [j, j], color=c, lw=2, alpha=0.5, solid_capstyle="round", zorder=1)
        ax.plot(g, j, "o", ms=8, mfc=SURFACE, mec=c, mew=2, zorder=2)
        ax.plot(v, j, "o", ms=9, color=c, mec=SURFACE, mew=1.5, zorder=3)
        ax.text(v + 0.8, j, f"{g:.1f} → {v:.1f}%  (+{v - g:.1f})", va="center", fontsize=9, color=INK,
                fontweight="bold" if name == OURS else "normal")
    ax.set_yticks(range(len(rows)), ["TinySQL (ours)" if r == OURS else r for r in rows], fontsize=9.5)
    for lbl in ax.get_yticklabels():
        lbl.set_color(INK)
        if lbl.get_text() == "TinySQL (ours)":
            lbl.set_fontweight("bold")
    ax.set_xlim(55, 96)
    ax.set_xlabel("WikiSQL test accuracy (%)", color=MUTED)
    fig.suptitle(f"What the {GRAMMAR} adds", x=0.03, ha="left", fontsize=13, color=INK, fontweight="bold")
    ax.set_title("○ without   ● with  (only real column names and values from the question allowed)",
                 loc="left", fontsize=9, color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT / "3_grammar_check_gains.png", facecolor=SURFACE)


def main():
    S = json.loads(Path("results/benchmark_test.json").read_text())["systems"]
    OUT.mkdir(parents=True, exist_ok=True)
    accuracy_vs_data(S)
    grammar_gains(S)


if __name__ == "__main__":
    main()
