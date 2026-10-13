"""Statistics of protocol Step 14.

Corrected resampled t-test (Nadeau & Bengio 2003; repeated k-fold form, Bouckaert & Frank 2004):
    t = dbar / sqrt((1/J + rho) * s_d^2),  df = J - 1,  CI95 = dbar +- t_{0.975, J-1} * sqrt((1/J + rho) * s_d^2)
with rho = n_test / n_train (~0.25 for a 5-fold outer split).
"""
from __future__ import annotations
import math
import numpy as np
from scipy import stats as sps

from malcore.metrics import per_class

DECISION_WORDING = {
    "supported": "M-full improves stage macro-F1 over B* (H1 supported)",
    "below_delta": "Detectable but below the smallest effect of interest",
    "no_improvement": "No meaningful improvement",
    "inconclusive": "Inconclusive at this sample size",
    "worse": "M-full is worse than B*",
}


def corrected_resampled_t(d, rho: float = 0.25, alpha: float = 0.05) -> dict:
    d = np.asarray(d, dtype=float); J = len(d)
    if J < 2:
        return {"J": J, "mean": float(d.mean()) if J else float("nan"), "note": "fewer than 2 differences"}
    mean = float(d.mean()); var = float(d.var(ddof=1))
    se = math.sqrt((1.0 / J + rho) * var)
    df = J - 1
    t = mean / se if se > 0 else (math.inf if mean > 0 else (-math.inf if mean < 0 else 0.0))
    p = float(2 * sps.t.sf(abs(t), df)) if math.isfinite(t) else 0.0
    q = float(sps.t.ppf(1 - alpha / 2, df))
    return {"J": J, "mean": mean, "sd": math.sqrt(var), "se_corrected": se, "rho": rho, "t": float(t), "df": df,
            "p": p, "ci_lo": mean - q * se, "ci_hi": mean + q * se, "alpha": alpha,
            "naive_se": math.sqrt(var / J)}


def decision(mean: float, ci_lo: float, ci_hi: float, delta_min: float = 0.02) -> dict:
    """The protocol's decision rule (Step 14), wording used verbatim in the paper."""
    if ci_lo > 0 and mean >= delta_min:
        key = "supported"
    elif ci_lo > 0:
        key = "below_delta"
    elif ci_hi < 0:
        key = "worse"
    elif ci_hi < delta_min:
        key = "no_improvement"
    else:
        key = "inconclusive"
    return {"verdict": key, "wording": DECISION_WORDING[key], "delta_min": delta_min}


def holm(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm step-down adjustment within one family; returns {name: {p, p_adj, reject}}."""
    items = sorted(((p, n) for n, p in pvals.items() if p is not None and not math.isnan(p)), key=lambda t: t[0])
    m = len(items); out = {}; running = 0.0
    for i, (p, n) in enumerate(items):
        adj = min(1.0, (m - i) * p); running = max(running, adj)
        out[n] = {"p": p, "p_adj": running, "reject": running < alpha}
    for n, p in pvals.items():
        if n not in out:
            out[n] = {"p": p, "p_adj": None, "reject": None}
    return out


def sign_test(d) -> dict:
    """Exact two-sided sign test on the paired differences (zeros dropped)."""
    d = np.asarray(d, dtype=float); nz = d[d != 0]
    n = len(nz); pos = int((nz > 0).sum())
    p = float(sps.binomtest(pos, n, 0.5).pvalue) if n else float("nan")
    return {"n_nonzero": n, "n_positive": pos, "n_negative": n - pos, "p": p}


def power_statement(rho: float = 0.25, J: int = 15, sigma_d=(0.02, 0.03), alpha: float = 0.05, power: float = 0.8) -> dict:
    """Minimum detectable effect under the corrected SE: MDE ~ (t_{1-alpha/2} + t_{power}) * sqrt(1/J + rho) * sigma_d."""
    df = J - 1; factor = math.sqrt(1.0 / J + rho)
    mult = float(sps.t.ppf(1 - alpha / 2, df) + sps.t.ppf(power, df))
    return {"J": J, "df": df, "rho": rho, "se_factor": factor, "t_multiplier": mult, "alpha": alpha, "power": power,
            "mde_per_sigma_d": mult * factor,
            "mde": {str(s): mult * factor * s for s in sigma_d},
            "se_factor_25_folds": math.sqrt(1 / 25 + rho),
            "statement": (f"The corrected standard error is sigma_d * sqrt(1/{J} + {rho}) = {factor:.2f} sigma_d. For "
                          f"{int(power * 100)} % power at alpha = {alpha} (two-sided, df {df}) the minimum detectable effect is "
                          f"{mult:.2f} x {factor:.2f} sigma_d = {mult * factor:.2f} sigma_d: "
                          + "; ".join(f"sigma_d = {s} -> MDE = {mult * factor * s:.3f}" for s in sigma_d) + ".")}


# ------------------------------------------------------------------ grouped bootstrap on pooled predictions
def _f1_from_cm(cm: np.ndarray) -> np.ndarray:
    """cm: (B, 4, 4) -> per-class F1 (B, 4), zero_division = 0."""
    tp = np.einsum("bii->bi", cm).astype(float); fp = cm.sum(1) - tp; fn = cm.sum(2) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0); rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        return np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)


def _recall_from_cm(cm: np.ndarray) -> np.ndarray:
    tp = np.einsum("bii->bi", cm).astype(float); fn = cm.sum(2) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(tp + fn > 0, tp / (tp + fn), np.nan)


def group_confusions(y_true, y_pred, groups) -> tuple[np.ndarray, np.ndarray]:
    """Per-group 4x4 confusion matrices (G, 4, 4) and the sorted group keys."""
    keys, inv = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    cm = np.zeros((len(keys), 4, 4), dtype=np.int64)
    np.add.at(cm, (inv, np.asarray(y_true, int), np.asarray(y_pred, int)), 1)
    return cm, keys


def stat_from_cm(cm: np.ndarray, stat: str) -> np.ndarray:
    """stat: macro_f1 | f1_S | f1_G | balanced_accuracy on a (B, 4, 4) stack."""
    if stat == "macro_f1":
        return _f1_from_cm(cm).mean(1)
    if stat.startswith("f1_"):
        return _f1_from_cm(cm)[:, "RTSG".index(stat[-1])]
    if stat == "balanced_accuracy":
        return np.nanmean(_recall_from_cm(cm), axis=1)
    raise ValueError(stat)


def grouped_bootstrap(pairs: list[tuple[np.ndarray, np.ndarray]], stat: str, n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired, image-grouped bootstrap of a pooled statistic.

    pairs: per repeat, (cm_a, cm_b) group-confusion stacks over the SAME groups (paired across arms).
    Groups are resampled with replacement within each repeat; the difference (a - b) is averaged over
    repeats; the CI is the percentile interval of that repeat-averaged difference.
    """
    rng = np.random.default_rng(seed)
    diffs = np.zeros(n_boot); point = []
    for cm_a, cm_b in pairs:
        G = cm_a.shape[0]
        idx = rng.integers(0, G, size=(n_boot, G))
        mult = np.stack([np.bincount(row, minlength=G) for row in idx]).astype(np.int64)      # (B, G)
        boot_a = np.einsum("bg,gij->bij", mult, cm_a); boot_b = np.einsum("bg,gij->bij", mult, cm_b)
        diffs += stat_from_cm(boot_a, stat) - stat_from_cm(boot_b, stat)
        point.append(float(stat_from_cm(cm_a.sum(0)[None], stat)[0] - stat_from_cm(cm_b.sum(0)[None], stat)[0]))
    diffs /= len(pairs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return {"stat": stat, "mean_diff": float(np.mean(point)), "boot_mean": float(diffs.mean()), "ci_lo": float(lo),
            "ci_hi": float(hi), "p_boot": min(1.0, p), "n_boot": n_boot, "n_repeats": len(pairs), "per_repeat_diff": point}


def t_interval(x, alpha: float = 0.05) -> dict:
    """Descriptive t-interval across models (Step 12: not a test, since models share the external set)."""
    x = np.asarray(x, dtype=float); n = len(x)
    if n < 2:
        return {"n": n, "mean": float(x.mean()) if n else float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    se = x.std(ddof=1) / math.sqrt(n); q = float(sps.t.ppf(1 - alpha / 2, n - 1))
    return {"n": n, "mean": float(x.mean()), "sd": float(x.std(ddof=1)), "ci_lo": float(x.mean() - q * se),
            "ci_hi": float(x.mean() + q * se), "min": float(x.min()), "max": float(x.max())}
