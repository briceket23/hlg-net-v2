"""Step 3.5: Table 1 (data) and the derived ratios, generated only from the frozen manifest."""
from __future__ import annotations
import pandas as pd

from malcore import STAGES, SPECIES
from .io_utils import out_dir, write_csv, write_json, append_digest, assert_not_frozen, write_text


def _need(path):
    from pathlib import Path as _P
    if not _P(path).exists():
        raise SystemExit(f"{path} not found: complete visual QC first (qc-mark, then qc-apply). "
                         "For a development preview only, pass --manifest cells.csv.")


def _md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.astype(object).values]
    return "\n".join(lines) + "\n"


def table1(cfg: dict, manifest: str = "cells_final.csv") -> dict:
    assert_not_frozen(cfg, "tables")
    md = out_dir(cfg, "manifests"); td = out_dir(cfg, "tables")
    _need(md / manifest)
    c = pd.read_csv(md / manifest, dtype={"slide": str, "group_slide": str})
    unit = "group_" + cfg["mpidb"]["group_unit"] if "group_" + cfg["mpidb"]["group_unit"] in c else "slide"
    sp = [s for s in SPECIES if s in set(c.species)]
    cells = pd.crosstab(c.species, c.stage).reindex(index=sp, columns=STAGES, fill_value=0)
    t = cells.copy()
    t["cells"] = cells.sum(axis=1)
    t["images"] = c.groupby("species").image_id.nunique().reindex(sp)
    t["groups"] = c.groupby("species")[unit].nunique().reindex(sp)
    for st in ["S", "G"]:
        t[f"images_with_{st}"] = c[c.stage == st].groupby("species").image_id.nunique().reindex(sp).fillna(0).astype(int)
        t[f"groups_with_{st}"] = c[c.stage == st].groupby("species")[unit].nunique().reindex(sp).fillna(0).astype(int)
    tot = t.sum(numeric_only=True)
    tot["images"] = c.image_id.nunique(); tot["groups"] = c[unit].nunique()
    for st in ["S", "G"]:
        tot[f"images_with_{st}"] = c[c.stage == st].image_id.nunique()
        tot[f"groups_with_{st}"] = c[c.stage == st][unit].nunique()
    t.loc["total"] = tot
    t = t.reset_index().rename(columns={"index": "species"})
    n = len(c)
    derived = {
        "n_cells": n, "grouping_unit": unit,
        "share_ring_plus_troph": round((c.stage.isin(["R", "T"])).mean(), 4),
        "share_schizont_plus_gametocyte": round((c.stage.isin(["S", "G"])).mean(), 4),
        "imbalance_ring_to_schizont": round((c.stage == "R").sum() / max(1, (c.stage == "S").sum()), 2),
        "imbalance_ring_to_gametocyte": round((c.stage == "R").sum() / max(1, (c.stage == "G").sum()), 2),
        "falciparum_share_of_rings": round(((c.stage == "R") & (c.species == "falciparum")).sum()
                                           / max(1, (c.stage == "R").sum()), 4),
        "max_cells_one_group": int(c.groupby(unit).size().max()),
        "box_side_median_px_by_stage": c.groupby("stage").box_side.median().round(1).reindex(STAGES).to_dict(),
    }
    sha = write_csv(t, td / "table1_data.csv"); append_digest(cfg, "table1_data.csv", sha)
    write_text(_md(t), td / "table1_data.md")
    write_json(derived, td / "derived_ratios.json")
    ex = md / ("exclusions_final.csv" if (md / "exclusions_final.csv").exists() else "exclusions.csv")
    e = pd.read_csv(ex)
    if len(e):
        es = e.groupby(["reason", "stage"]).size().unstack(fill_value=0).reset_index()
        write_csv(es, td / "table1_exclusions.csv")
    return derived
