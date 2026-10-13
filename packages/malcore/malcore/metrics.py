"""Metrics of protocol Step 13 (pure numpy; cross-checked against scikit-learn in the tests).

All four stage labels [0, 1, 2, 3] are always scored (`zero_division=0`), so macro-F1 is the same
endpoint on MP-IDB and on BBBC041.
"""
from __future__ import annotations
import numpy as np

from . import STAGES

LABELS = [0, 1, 2, 3]


def confusion(y_true, y_pred, n: int = 4) -> np.ndarray:
    cm = np.zeros((n, n), dtype=int)
    yt = np.asarray(y_true, dtype=int); yp = np.asarray(y_pred, dtype=int)
    np.add.at(cm, (yt, yp), 1)
    return cm


def per_class(y_true, y_pred, n: int = 4) -> dict:
    cm = confusion(y_true, y_pred, n)
    tp = np.diag(cm).astype(float); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    return {"precision": prec, "recall": rec, "f1": f1, "support": cm.sum(1), "cm": cm}


def macro_f1(y_true, y_pred, n: int = 4) -> float:
    return float(per_class(y_true, y_pred, n)["f1"].mean())


def balanced_accuracy(y_true, y_pred, n: int = 4) -> float:
    r = per_class(y_true, y_pred, n)
    present = r["support"] > 0
    return float(r["recall"][present].mean()) if present.any() else 0.0


def ece(probs, y_true, n_bins: int = 15) -> float:
    """Top-label expected calibration error, 15 equal-width right-closed bins."""
    probs = np.asarray(probs, dtype=float); y = np.asarray(y_true)
    conf = probs.max(1); pred = probs.argmax(1); acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1); e = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def reliability(probs, y_true, n_bins: int = 15) -> dict:
    """Per-bin accuracy, confidence and count (for reliability diagrams, Figure 4)."""
    probs = np.asarray(probs, dtype=float); y = np.asarray(y_true)
    conf = probs.max(1); acc = (probs.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        rows.append({"bin": i, "lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                     "acc": float(acc[m].mean()) if m.any() else float("nan"),
                     "conf": float(conf[m].mean()) if m.any() else float("nan")})
    return {"bins": rows, "ece": ece(probs, y, n_bins)}


def brier(probs, y_true) -> float:
    probs = np.asarray(probs, dtype=float); y = np.asarray(y_true)
    onehot = np.eye(probs.shape[1])[y]
    return float(((probs - onehot) ** 2).sum(1).mean())


def nll(probs, y_true) -> float:
    probs = np.asarray(probs, dtype=float); y = np.asarray(y_true)
    return float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1)).mean())


def summary(y_true, y_pred, probs=None) -> dict:
    r = per_class(y_true, y_pred)
    out = {"macro_f1": macro_f1(y_true, y_pred), "balanced_accuracy": balanced_accuracy(y_true, y_pred),
           "accuracy": float((np.asarray(y_true) == np.asarray(y_pred)).mean()) if len(y_true) else 0.0,
           "n": int(len(y_true))}
    for i, st in enumerate(STAGES):
        out[f"f1_{st}"] = float(r["f1"][i]); out[f"recall_{st}"] = float(r["recall"][i])
        out[f"precision_{st}"] = float(r["precision"][i]); out[f"support_{st}"] = int(r["support"][i])
    if probs is not None:
        out["ece"] = ece(probs, y_true); out["brier"] = brier(probs, y_true); out["nll"] = nll(probs, y_true)
    out["confusion"] = r["cm"].tolist()
    return out
