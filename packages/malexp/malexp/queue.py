"""Step 10: the run matrix as runs/<unit>/queue.csv.

Ordered by tier, then repeat, then fold; the arm order within each (r, k) is randomised with seed 7
(select.arm_order). Folds are completed for all arms of a tier before the next fold starts, so if
compute runs out every arm has the same folds and the paired analysis stays valid. The runner follows
this order; the file documents it and is what `status` compares against.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

from malcore.io import write_csv
from .select import TIER1, TIER2, ABL, arm_order, backbone_tag

TIER_ARMS = {1: TIER1, 2: TIER2, 3: ABL}


def build_queue(unit: str, backbone: str, repeats=(1, 2, 3), folds=(0, 1, 2, 3, 4), tiers=(1, 2, 3),
                with_prob: bool = False) -> pd.DataFrame:
    rows = []
    for tier in tiers:
        for r in repeats:
            for k in folds:
                for arm in arm_order(TIER_ARMS[tier], r, k):
                    if arm == "M-prob" and not with_prob:
                        continue
                    n_cfg = 1 if tier == 3 else 2
                    for i in range(n_cfg):
                        rows.append({"tier": tier, "repeat": r, "fold": k, "arm": arm, "cfg_idx": i, "unit": unit,
                                     "backbone": backbone, "run_id": f"{arm}/{backbone_tag(backbone)}/r{r}_f{k}/c{i}"})
    q = pd.DataFrame(rows)
    q["order"] = range(len(q))
    return q


def write_queue(runs_root: str | Path, **kw) -> Path:
    q = build_queue(**kw)
    p = Path(runs_root) / "queue.csv"
    write_csv(q, p)
    return p


def status(runs_root: str | Path) -> pd.DataFrame:
    """Which queued runs are trained (fit.json) and which are finalized (meta.json + predictions)."""
    runs_root = Path(runs_root)
    q = pd.read_csv(runs_root / "queue.csv")
    q["trained"] = [(runs_root / rid / "fit.json").exists() for rid in q.run_id]
    q["finalized"] = [(runs_root / rid / "meta.json").exists() and (runs_root / rid / "preds_test.parquet").exists()
                      for rid in q.run_id]
    return q
