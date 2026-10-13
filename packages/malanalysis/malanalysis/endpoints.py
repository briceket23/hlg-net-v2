"""Step 13 endpoints from the prediction files.

Per fold (15 values per arm): stage macro-F1 (primary), balanced accuracy, ECE, Brier, NLL, accuracy.
Pooled per repeat (3 values per arm): per-stage precision / recall / F1 and the confusion matrix. Rare-
stage F1 is pooled because a fold may hold only a few schizonts (fixed here, not after seeing results).
Pooled values are labelled "pooled" in every table and never share a column with per-fold values.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from malcore import STAGES
from malcore.metrics import summary, per_class, reliability
from .loaders import RunSet, PROB


def per_fold_metrics(rs: RunSet, arms, repeats, folds) -> pd.DataFrame:
    rows = []
    for arm in arms:
        for r in repeats:
            for k in folds:
                p = rs.preds(arm, r, k)
                if p is None:
                    continue
                s = summary(p.y_true.to_numpy(), p.y_pred.to_numpy(), p[PROB].to_numpy())
                s.pop("confusion")
                rows.append({"arm": arm, "repeat": r, "fold": k, "regime": "per-fold",
                             "bstar_arm": rs.bstar.get((r, k)), "T": float(p["T"].iloc[0]), **s})
    return pd.DataFrame(rows)


def pooled_per_repeat(rs: RunSet, arms, repeats, folds) -> tuple[pd.DataFrame, dict]:
    """Pooled per-stage metrics per (arm, repeat) and the pooled confusion matrices {arm: {repeat: cm}}."""
    rows, cms = [], {}
    for arm in arms:
        for r in repeats:
            parts = [rs.preds(arm, r, k) for k in folds]
            parts = [p for p in parts if p is not None]
            if not parts:
                continue
            p = pd.concat(parts, ignore_index=True)
            pc = per_class(p.y_true.to_numpy(), p.y_pred.to_numpy())
            cms.setdefault(arm, {})[r] = pc["cm"]
            row = {"arm": arm, "repeat": r, "regime": "pooled", "n": int(len(p))}
            for i, st in enumerate(STAGES):
                row[f"f1_{st}"] = float(pc["f1"][i]); row[f"precision_{st}"] = float(pc["precision"][i])
                row[f"recall_{st}"] = float(pc["recall"][i]); row[f"support_{st}"] = int(pc["support"][i])
            rows.append(row)
    return pd.DataFrame(rows), cms


def pooled_by_species(rs: RunSet, arm: str, repeats, folds) -> pd.DataFrame:
    """Per-species stage confusion matrices pooled over the repeats (confound report of Step 2 decisions)."""
    rows = []
    for r in repeats:
        for k in folds:
            p = rs.preds(arm, r, k)
            if p is not None and "species" in p:
                rows.append(p[["species", "y_true", "y_pred"]].assign(repeat=r))
    if not rows:
        return pd.DataFrame()
    p = pd.concat(rows, ignore_index=True); out = []
    for sp, g in p.groupby("species"):
        pc = per_class(g.y_true.to_numpy(), g.y_pred.to_numpy())
        out.append({"arm": arm, "species": sp, "n": int(len(g)), "regime": "pooled (all repeats)",
                    **{f"f1_{st}": float(pc["f1"][i]) for i, st in enumerate(STAGES)},
                    **{f"recall_{st}": float(pc["recall"][i]) for i, st in enumerate(STAGES)},
                    "confusion": pc["cm"].tolist()})
    return pd.DataFrame(out)


def species_macro_f1(rs: RunSet, arm: str, repeats, folds) -> pd.DataFrame:
    """Descriptive species macro-F1 of the species head (B5 only); species is an image-level label."""
    from malcore import SPECIES
    rows = []
    for r in repeats:
        for k in folds:
            p = rs.preds(arm, r, k)
            if p is None or "species_pred" not in p:
                continue
            yt = p.species.map({s: i for i, s in enumerate(SPECIES)}).fillna(-1).astype(int).to_numpy()
            m = yt >= 0
            pc = per_class(yt[m], p.species_pred.to_numpy()[m])
            rows.append({"arm": arm, "repeat": r, "fold": k, "species_macro_f1": float(pc["f1"].mean()), "n": int(m.sum())})
    return pd.DataFrame(rows)


def reliability_curves(rs: RunSet, arms, repeats, folds) -> dict:
    out = {}
    for arm in arms:
        parts = [rs.preds(arm, r, k) for r in repeats for k in folds]
        parts = [p for p in parts if p is not None]
        if parts:
            p = pd.concat(parts, ignore_index=True)
            out[arm] = reliability(p[PROB].to_numpy(), p.y_true.to_numpy())
    return out


def paired_differences(pf: pd.DataFrame, a: str, b: str, metric: str = "macro_f1") -> pd.DataFrame:
    """One row per (repeat, fold) with metric(a) - metric(b); folds missing for either arm are dropped."""
    x = pf[pf.arm == a][["repeat", "fold", metric]].rename(columns={metric: "a"})
    y = pf[pf.arm == b][["repeat", "fold", metric]].rename(columns={metric: "b"})
    d = x.merge(y, on=["repeat", "fold"])
    d["d"] = d.a - d.b
    return d.sort_values(["repeat", "fold"]).reset_index(drop=True)
