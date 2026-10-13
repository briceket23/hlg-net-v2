"""Read-side API used by the training code: the cells of one (repeat, fold) partition.

The training code must request the "test" partition only inside finalize() (protocol Step 7);
malexp enforces this with its HeldoutGuard.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from malcore import RARE
from .splits import load_split, split_path
from .io_utils import out_dir, sha256_file

PARTS = ("inner_train", "inner_val", "test")


def get_partition(cfg: dict, unit: str, r: int, k: int, part: str, with_crops: bool = True) -> pd.DataFrame:
    if part not in PARTS:
        raise ValueError(part)
    sp = load_split(cfg, unit, r)
    f = sp["folds"][k]
    cells = pd.read_csv(sp["manifest"], dtype={"slide": str, "group_slide": str, "group_image": str})
    df = cells[cells[sp["group_column"]].astype(str).isin(set(map(str, f[part])))].copy()
    df["image_group"] = df[sp["group_column"]].astype(str)
    if with_crops:
        crops = pd.read_csv(out_dir(cfg, "manifests") / "crops.csv")
        cols = ["cell_id", "crop_path", "mask_path", "crop_sha256"] + \
               [c for c in ("n_other_in_window", "n_other_stage_in_window") if c in crops]
        df = df.merge(crops[cols], on="cell_id", how="left")
    df["repeat"], df["fold"], df["partition"] = r, k, part
    return df.sort_values("cell_id").reset_index(drop=True)


def fold_digests(cfg: dict, unit: str, r: int) -> dict:
    """SHA-256 of the split file and of the manifest it was built from (for meta.json / preds)."""
    p = split_path(cfg, unit, r)
    sp = load_split(cfg, unit, r)
    return {"split_file": p.as_posix(), "split_sha256": sha256_file(p),
            "manifest": sp["manifest"], "manifest_sha256": sp["manifest_sha256"],
            "manifest_sha256_now": sha256_file(sp["manifest"]) if Path(sp["manifest"]).exists() else None}


def sampled_prevalence(labels, weights, late=RARE) -> float:
    """Expected share of the rare-stage group under a WeightedRandomSampler (sets the gate BCE pos_weight)."""
    y = np.asarray(labels)
    if weights is None:
        return float(np.isin(y, late).mean())
    w = np.asarray(weights, float)
    return float(w[np.isin(y, late)].sum() / w.sum())
