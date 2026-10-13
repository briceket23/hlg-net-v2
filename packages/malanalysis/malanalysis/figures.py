"""Figures 2-4 (Step 17): paired per-fold differences, pooled confusion matrices, reliability diagrams.
Colour-blind-safe palette; vector PDF output."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from malcore import STAGES, STAGE_NAMES

PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]


def fig_paired_differences(contrasts: dict, out: Path, name="fig2_paired_differences.pdf"):
    """One panel per contrast: one point per fold (coloured by repeat), the mean and the corrected 95 % CI."""
    n = len(contrasts)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n + 1, 3.4), sharey=True, squeeze=False)
    for ax, (title, c) in zip(axes[0], contrasts.items()):
        d = c["per_fold"]; t = c["test"]
        reps = sorted({row["repeat"] for row in d})
        for j, r in enumerate(reps):
            xs = [row["fold"] + 0.12 * (j - (len(reps) - 1) / 2) for row in d if row["repeat"] == r]
            ys = [row["d"] for row in d if row["repeat"] == r]
            ax.scatter(xs, ys, s=28, color=PALETTE[j % len(PALETTE)], label=f"repeat {r}", zorder=3)
        ax.axhline(0, color="0.5", lw=0.8)
        ax.axhspan(t["ci_lo"], t["ci_hi"], color="0.85", zorder=1, label="corrected 95 % CI")
        ax.axhline(t["mean"], color="black", lw=1.2, label="mean")
        ax.set_title(title, fontsize=10); ax.set_xlabel("outer fold")
        ax.set_xticks(sorted({row["fold"] for row in d}))
    axes[0][0].set_ylabel("difference in per-fold macro-F1")
    axes[0][-1].legend(fontsize=7, loc="best", frameon=False)
    fig.tight_layout(); fig.savefig(out / name); plt.close(fig)
    return out / name


def fig_confusions(cms: dict[str, np.ndarray], out: Path, name="fig3_confusion.pdf"):
    """Row-normalised pooled confusion matrices, e.g. {'B* (MP-IDB)': cm, 'M-full (MP-IDB)': cm, ...}."""
    items = [(k, v) for k, v in cms.items() if v is not None]
    n = len(items)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n + 0.5, 3.2), squeeze=False)
    for ax, (title, cm) in zip(axes[0], items):
        cm = np.asarray(cm, dtype=float); norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", fontsize=7, color="white" if norm[i, j] > 0.5 else "black")
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels([STAGE_NAMES[s][:5] for s in STAGES], fontsize=7); ax.set_yticklabels([STAGE_NAMES[s][:5] for s in STAGES], fontsize=7)
        ax.set_xlabel("predicted"); ax.set_title(title, fontsize=9)
    axes[0][0].set_ylabel("true")
    fig.tight_layout(); fig.savefig(out / name); plt.close(fig)
    return out / name


def fig_reliability(curves: dict, out: Path, name="fig4_reliability.pdf"):
    """Reliability diagrams after temperature scaling, one curve per arm, pooled over all folds."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], color="0.5", lw=0.8, ls="--")
    for j, (arm, rel) in enumerate(curves.items()):
        b = [x for x in rel["bins"] if x["n"] > 0]
        ax.plot([x["conf"] for x in b], [x["acc"] for x in b], marker="o", ms=3, color=PALETTE[j % len(PALETTE)],
                label=f"{arm} (ECE {rel['ece']:.3f})")
    ax.set_xlabel("confidence (temperature-scaled)"); ax.set_ylabel("accuracy"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(out / name); plt.close(fig)
    return out / name
