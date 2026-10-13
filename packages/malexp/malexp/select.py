"""Nested selection (protocol Step 9) and the per-fold job plans.

Per outer fold (r, k):
  1. train B0..B4 x 2 configs; pick each arm's config on inner-val macro-F1 (tie within 0.005 -> the
     smaller lr / weight, i.e. grid index 0)
  2. B*(r, k) = best of B1..B4 on inner val (tie within 0.005 -> simpler: B2 < B1 < B4 < B3)
  3. B5, B6, M-aux, M-full, M-prob x 2 configs built from B*(r, k); ablations from the selected M-full
  4. only the selected run of each arm has its test predictions produced (finalize)
Every decision is appended to runs/<unit>/selection.csv; test predictions are never read here.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from malcore import ARMS_BASE, ARMS_DERIVED, ARMS_ABLATION
from .config import ArmConfig, base_arms, derive_from_bstar, candidates

SIMPLICITY = {"B2": 0, "B1": 1, "B4": 2, "B3": 3}
TIE = 0.005
DIVERGENCE_MARGIN = 0.10          # Step 10: inner-val macro-F1 below B0's score - 0.10 counts as diverged
TIER1 = list(ARMS_BASE)
TIER2 = list(ARMS_DERIVED)
ABL = list(ARMS_ABLATION)
KEY = ["unit", "backbone", "repeat", "fold", "arm", "cfg_idx", "selected"]


def backbone_tag(backbone: str) -> str:
    return backbone.split(".")[0]


def run_id(arm: str, cfg_idx: int, backbone_tag_: str, r: int, k: int) -> str:
    return f"{arm}/{backbone_tag_}/r{r}_f{k}/c{cfg_idx}"


def pick(arm: str, scored: list[tuple[int, float, ArmConfig]]) -> tuple[int, ArmConfig]:
    """scored: (cfg_idx, inner_val_macro_f1, cfg). Tie within TIE -> the smaller tuned value (lower index)."""
    best = max(scored, key=lambda t: t[1])
    tied = [t for t in scored if best[1] - t[1] <= TIE]
    chosen = min(tied, key=lambda t: t[0])
    return chosen[0], chosen[2]


def pick_bstar(arm_scores: dict[str, float]) -> str:
    """arm_scores: inner-val macro-F1 of the selected config of B1..B4."""
    best = max(arm_scores.values())
    tied = [a for a, s in arm_scores.items() if best - s <= TIE]
    return min(tied, key=lambda a: SIMPLICITY[a])


def arm_order(arms: list[str], r: int, k: int, seed: int = 7) -> list[str]:
    """Step 10: the arm order within each (r, k) is randomised with seed 7 (within a tier; B0 stays
    first in tier 1 because B4 and the divergence rule depend on it)."""
    rng = np.random.default_rng([seed, r, k])
    rest = [a for a in arms if a != "B0"]
    rest = [rest[i] for i in rng.permutation(len(rest))]
    return (["B0"] if "B0" in arms else []) + rest


class SelectionLog:
    """analysis/selection.csv: one row per (r, k, arm, config) plus one 'selected' row per arm. Idempotent
    on resume: a row with the same key replaces the earlier one instead of duplicating it."""
    def __init__(self, path: str | Path):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = pd.read_csv(self.path).to_dict("records") if self.path.exists() else []

    def add(self, **row):
        key = tuple(row.get(k) for k in KEY)
        self.rows = [r for r in self.rows if tuple(r.get(k) for k in KEY) != key]
        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(self.path, index=False, lineterminator="\n")

    def df(self):
        return pd.DataFrame(self.rows)


def build_fold_plan(unit: str, r: int, k: int, backbone: str, input_size: int, arms_dir=None) -> list[dict]:
    """Tier-1 jobs of a fold (independent of any result). Tier 2 needs B*, so it is planned after tier 1."""
    tag = backbone_tag(backbone)
    jobs = []
    arms = base_arms(arms_dir)
    for arm in arm_order(TIER1, r, k):
        base = arms[arm].with_(backbone=backbone, input_size=input_size)
        for i, cfg in enumerate(candidates(arm, base, arms_dir)):
            jobs.append({"arm": arm, "cfg_idx": i, "cfg": cfg, "tier": 1, "unit": unit, "r": r, "k": k,
                         "run_id": run_id(arm, i, tag, r, k)})
    return jobs


def tier2_plan(unit: str, r: int, k: int, bstar: ArmConfig, include_prob: bool = False, arms_dir=None) -> list[dict]:
    tag = backbone_tag(bstar.backbone)
    jobs = []
    for arm in arm_order(TIER2, r, k):
        if arm == "M-prob" and not include_prob:
            continue
        base = derive_from_bstar(arm, bstar)
        for i, cfg in enumerate(candidates(arm, base, arms_dir)):
            jobs.append({"arm": arm, "cfg_idx": i, "cfg": cfg, "tier": 2, "unit": unit, "r": r, "k": k,
                         "run_id": run_id(arm, i, tag, r, k)})
    return jobs


def ablation_plan(unit: str, r: int, k: int, mfull: ArmConfig) -> list[dict]:
    """One run each, inheriting the fold's selected M-full settings (Step 15); no extra tuning."""
    tag = backbone_tag(mfull.backbone)
    jobs = []
    neutral = mfull.with_(use_gate=False, use_boost=False, detach_gate=False, use_coordattn=False)
    for arm in arm_order(ABL, r, k):
        cfg = derive_from_bstar(arm, neutral, lambda_gate=mfull.lambda_gate if arm != "ABL-noaux" else 0.0)
        jobs.append({"arm": arm, "cfg_idx": 0, "cfg": cfg, "tier": 3, "unit": unit, "r": r, "k": k,
                     "run_id": run_id(arm, 0, tag, r, k)})
    return jobs
