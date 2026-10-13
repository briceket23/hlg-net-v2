"""Run discovery, prediction loading, the per-fold B* arm, and the Step 11 completeness check."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

from malcore.hashing import sha256_file

PROB = [f"prob_{c}" for c in range(4)]
LOGIT = [f"logit_{c}" for c in range(4)]
FILES = ("preds_test.parquet", "preds_innerval.parquet", "preds_bbbc041.parquet")


def discover(runs_root: str | Path) -> pd.DataFrame:
    """One row per finalized run (meta.json present): arm, backbone tag, repeat, fold, cfg_idx, run_dir."""
    rows = []
    for m in sorted(Path(runs_root).glob("*/*/r*_f*/c*/meta.json")):
        d = json.loads(m.read_text())
        rows.append({"arm": d["arm"], "backbone": d["backbone"], "backbone_tag": m.parents[2].name,
                     "repeat": int(d["repeat"]), "fold": int(d["fold"]), "cfg_idx": int(m.parent.name[1:]),
                     "run_dir": str(m.parent), "innerval_macro_f1": d.get("innerval_macro_f1_at_best"),
                     "best_epoch": d.get("best_epoch"), "hit_cap": d.get("hit_cap"), "diverged": d.get("diverged", False),
                     "temperature": d.get("temperature"), "manifest_sha256": d.get("manifest_sha256"),
                     "split_sha256": d.get("split_sha256"), "config_sha": d.get("config_sha"),
                     "lambda_gate": (d.get("config") or {}).get("lambda_gate"), "lr": (d.get("config") or {}).get("lr"),
                     "commit": d.get("commit"), "elapsed_s": d.get("elapsed_s")})
    return pd.DataFrame(rows)


def selection(runs_root: str | Path) -> pd.DataFrame:
    p = Path(runs_root) / "selection.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p}: the selection log is required to identify B* per fold")
    s = pd.read_csv(p)
    s["selected"] = s.selected.astype(str).str.lower().isin(["true", "1"])
    return s


def bstar_map(sel: pd.DataFrame) -> dict[tuple[int, int], str]:
    """(repeat, fold) -> the arm selected as B* in that fold (from the selection log)."""
    s = sel[sel.selected & sel.bstar.notna() & (sel.bstar.astype(str) != "")]
    out = {}
    for (r, k), g in s.groupby(["repeat", "fold"]):
        vals = set(g.bstar.astype(str))
        if len(vals) != 1:
            raise ValueError(f"inconsistent B* for r{r} k{k}: {vals}")
        out[(int(r), int(k))] = vals.pop()
    return out


def load_preds(run_dir: str | Path, kind: str = "test") -> pd.DataFrame:
    return pd.read_parquet(Path(run_dir) / f"preds_{kind}.parquet")


class RunSet:
    """All finalized runs of one runs root with the virtual arm 'B*' resolved per fold."""
    def __init__(self, runs_root: str | Path, backbone_tag: str | None = None):
        self.root = Path(runs_root)
        self.runs = discover(self.root)
        if backbone_tag is not None and len(self.runs):
            self.runs = self.runs[self.runs.backbone_tag == backbone_tag].reset_index(drop=True)
        self.sel = selection(self.root)
        self.bstar = bstar_map(self.sel)
        self._cache: dict = {}

    def arms(self) -> list[str]:
        return sorted(set(self.runs.arm)) + (["B*"] if self.bstar else [])

    def run_dir(self, arm: str, r: int, k: int) -> Path | None:
        if arm == "B*":
            arm = self.bstar.get((r, k))
            if arm is None:
                return None
        g = self.runs[(self.runs.arm == arm) & (self.runs.repeat == r) & (self.runs.fold == k)]
        if len(g) == 0:
            return None
        if len(g) > 1:
            raise ValueError(f"more than one finalized run for {arm} r{r} k{k}: {g.run_dir.tolist()}")
        return Path(g.run_dir.iloc[0])

    def preds(self, arm: str, r: int, k: int, kind: str = "test") -> pd.DataFrame | None:
        key = (arm, r, k, kind)
        if key not in self._cache:
            rd = self.run_dir(arm, r, k)
            self._cache[key] = None if rd is None else load_preds(rd, kind).assign(arm_label=arm)
        return self._cache[key]

    def folds(self, repeats, folds):
        return [(r, k) for r in repeats for k in folds]


def completeness(rs: RunSet, arms: list[str], repeats, folds, manifest: str | Path | None, splits_dir=None,
                 unit: str = "slide") -> dict:
    """Step 11 check_completeness: every (arm, r, k) has the three files; within each repeat the union of
    test cell_ids equals the manifest with no duplicates; digests match the frozen files."""
    problems = []
    for arm in arms:
        for r in repeats:
            for k in folds:
                rd = rs.run_dir(arm, r, k)
                if rd is None:
                    problems.append(f"{arm} r{r} k{k}: no finalized run"); continue
                for f in FILES:
                    if not (rd / f).exists():
                        problems.append(f"{arm} r{r} k{k}: missing {f}")
    manifest_ids = None
    if manifest and Path(manifest).exists():
        man = pd.read_csv(manifest, usecols=["cell_id"])
        manifest_ids = set(man.cell_id.astype(str))
        man_sha = sha256_file(manifest)
        for arm in arms:
            for r in repeats:
                ids = []
                for k in folds:
                    p = rs.preds(arm, r, k)
                    if p is not None:
                        ids += p.cell_id.astype(str).tolist()
                if len(ids) != len(set(ids)):
                    problems.append(f"{arm} r{r}: duplicate test cell_ids across folds")
                if ids and set(ids) != manifest_ids:
                    problems.append(f"{arm} r{r}: union of test cells != manifest ({len(set(ids))} vs {len(manifest_ids)})")
        bad = rs.runs[rs.runs.manifest_sha256.notna() & (rs.runs.manifest_sha256 != man_sha)]
        if len(bad):
            problems.append(f"{len(bad)} runs record a manifest SHA-256 different from {manifest}")
    if splits_dir:
        for r in repeats:
            sp = Path(splits_dir) / unit / f"outer_r{r}.json"
            if sp.exists():
                sha = sha256_file(sp)
                bad = rs.runs[(rs.runs.repeat == r) & rs.runs.split_sha256.notna() & (rs.runs.split_sha256 != sha)]
                if len(bad):
                    problems.append(f"{len(bad)} runs of repeat {r} record a split SHA-256 different from {sp}")
    # non-selected configs must not carry test predictions
    for c in Path(rs.root).glob("*/*/r*_f*/c*"):
        if c.is_dir() and not (c / "meta.json").exists() and (c / "preds_test.parquet").exists():
            problems.append(f"{c}: test predictions for a non-selected config")
    return {"ok": not problems, "problems": problems, "n_runs": int(len(rs.runs)),
            "arms": arms, "repeats": list(repeats), "folds": list(folds), "manifest_cells": len(manifest_ids) if manifest_ids else None}
