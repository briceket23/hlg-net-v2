"""Step 12: zero-shot evaluation of the 15 frozen fold models of each arm on BBBC041.

Unit of replication = the fold model. Each model gives one external macro-F1 over all four stages; the
arm's score is the mean over models with a descriptive t-interval (models share one external set, so it
is not a test). The paired contrast M-full - B* is reported per (r, k) with mean and range; sampling
uncertainty of the target comes from an image-grouped bootstrap over BBBC041 images applied to the
fold-averaged probabilities of each arm. Results are also given per official subset (train / test).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from malcore import STAGES
from malcore.metrics import summary, per_class, macro_f1
from .loaders import RunSet, PROB
from .stats import t_interval, group_confusions, grouped_bootstrap


def _rows(p: pd.DataFrame, arm, r, k, subset: str, low_n: int) -> dict:
    s = summary(p.y_true.to_numpy(), p.y_pred.to_numpy(), p[PROB].to_numpy())
    cm = s.pop("confusion")
    low = {st: int(s[f"support_{st}"]) < low_n for st in STAGES}
    return {"arm": arm, "repeat": r, "fold": k, "subset": subset, "regime": "per-model", **s,
            **{f"low_support_{st}": low[st] for st in STAGES}, "confusion": cm}


def by_model(rs: RunSet, arms, repeats, folds, subsets=("train", "test"), low_n: int = 30) -> pd.DataFrame:
    rows = []
    for arm in arms:
        for r in repeats:
            for k in folds:
                p = rs.preds(arm, r, k, "bbbc041")
                if p is None:
                    continue
                rows.append(_rows(p, arm, r, k, "pooled", low_n))
                if "subset" in p:
                    for sub in subsets:
                        g = p[p.subset == sub]
                        if len(g):
                            rows.append(_rows(g, arm, r, k, sub, low_n))
    return pd.DataFrame(rows)


def arm_summary(bm: pd.DataFrame, metric: str = "macro_f1") -> pd.DataFrame:
    rows = []
    for (arm, sub), g in bm.groupby(["arm", "subset"]):
        ti = t_interval(g[metric])
        rows.append({"arm": arm, "subset": sub, "metric": metric, "regime": "mean over models (descriptive t-interval)", **ti,
                     **{f"mean_f1_{st}": float(g[f"f1_{st}"].mean()) for st in STAGES},
                     **{f"mean_recall_{st}": float(g[f"recall_{st}"].mean()) for st in STAGES}})
    return pd.DataFrame(rows)


def paired_contrast(bm: pd.DataFrame, a: str, b: str, subset: str = "pooled", metric: str = "macro_f1") -> dict:
    x = bm[(bm.arm == a) & (bm.subset == subset)][["repeat", "fold", metric]].rename(columns={metric: "a"})
    y = bm[(bm.arm == b) & (bm.subset == subset)][["repeat", "fold", metric]].rename(columns={metric: "b"})
    d = x.merge(y, on=["repeat", "fold"]); diff = (d.a - d.b).to_numpy()
    return {"a": a, "b": b, "subset": subset, "metric": metric, "n_models": int(len(d)),
            "mean": float(diff.mean()) if len(diff) else float("nan"),
            "min": float(diff.min()) if len(diff) else float("nan"), "max": float(diff.max()) if len(diff) else float("nan"),
            "per_fold": [{"repeat": int(rr), "fold": int(kk), "d": float(dd)} for rr, kk, dd in zip(d.repeat, d.fold, diff)]}


def fold_averaged(rs: RunSet, arm: str, repeats, folds) -> pd.DataFrame | None:
    """Average the calibrated probabilities of the fold models of one arm per BBBC041 cell; argmax = prediction."""
    parts = [rs.preds(arm, r, k, "bbbc041") for r in repeats for k in folds]
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    base = parts[0][["cell_id", "y_true"] + [c for c in ("image_group", "image_id", "subset") if c in parts[0]]].copy()
    probs = np.mean([p.set_index("cell_id").loc[base.cell_id, PROB].to_numpy() for p in parts], axis=0)
    for c in range(4):
        base[f"prob_{c}"] = probs[:, c]
    base["y_pred"] = probs.argmax(1)
    base["n_models"] = len(parts)
    return base


def bootstrap_contrast(rs: RunSet, a: str, b: str, repeats, folds, n_boot: int = 2000, seed: int = 0, stat="macro_f1") -> dict:
    """Image-grouped bootstrap over BBBC041 images on the fold-averaged predictions of the two arms."""
    fa, fb = fold_averaged(rs, a, repeats, folds), fold_averaged(rs, b, repeats, folds)
    if fa is None or fb is None:
        return {"note": "missing external predictions"}
    fb = fb.set_index("cell_id").loc[fa.cell_id].reset_index()
    gcol = "image_group" if "image_group" in fa else "image_id"
    cm_a, keys = group_confusions(fa.y_true, fa.y_pred, fa[gcol]); cm_b, _ = group_confusions(fb.y_true, fb.y_pred, fb[gcol])
    out = grouped_bootstrap([(cm_a, cm_b)], stat, n_boot, seed)
    out.update({"a": a, "b": b, "n_images": int(len(keys)), "n_cells": int(len(fa)),
                "pooled_a": float(macro_f1(fa.y_true, fa.y_pred)), "pooled_b": float(macro_f1(fb.y_true, fb.y_pred))})
    return out


def pooled_confusion(rs: RunSet, arm: str, repeats, folds, subset: str | None = None) -> np.ndarray | None:
    fa = fold_averaged(rs, arm, repeats, folds)
    if fa is None:
        return None
    if subset and "subset" in fa:
        fa = fa[fa.subset == subset]
    return per_class(fa.y_true.to_numpy(), fa.y_pred.to_numpy())["cm"]
