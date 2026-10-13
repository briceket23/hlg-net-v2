"""Step 3.4: visual QC sheets, the QC decision file, and the final cell manifest.

qc-sheets: writes contact sheets (every schizont and gametocyte; a random sample of rings and
           trophozoites) with the polygon outline drawn, and data/qc/qc.csv to fill in.
           Existing decisions in qc.csv are kept when sheets are regenerated.
qc-apply : reads qc.csv and writes data/manifests/cells_final.csv
           decision values:  ok | exclude | relabel:<R|T|S|G> | (blank = not reviewed)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from malcore import STAGES
from .io_utils import out_dir, write_csv, append_digest, assert_not_frozen

TILE = 160
VALID = {"ok", "exclude"} | {f"relabel:{s}" for s in STAGES}


def _tile(crop_path: str, mask_path: str, caption: str) -> Image.Image:
    im = Image.open(crop_path).convert("RGB").resize((TILE, TILE), Image.Resampling.LANCZOS)
    m = Image.open(mask_path).convert("L").resize((TILE, TILE), Image.Resampling.NEAREST)
    edge = np.asarray(m.filter(ImageFilter.FIND_EDGES)) > 0
    a = np.asarray(im).copy(); a[edge] = (255, 0, 0)
    t = Image.new("RGB", (TILE, TILE + 14), (255, 255, 255))
    t.paste(Image.fromarray(a), (0, 0))
    ImageDraw.Draw(t).text((2, TILE + 1), caption[-24:], fill=(0, 0, 0), font=ImageFont.load_default())
    return t


def make_sheets(cfg: dict) -> dict:
    assert_not_frozen(cfg, "qc-sheets")
    md = out_dir(cfg, "manifests"); qd = out_dir(cfg, "qc"); sd = out_dir(cfg, "qc", "sheets")
    cells = pd.read_csv(md / "cells.csv", dtype={"slide": str, "field": str})
    crops = pd.read_csv(md / "crops.csv")
    df = cells.merge(crops[["cell_id", "crop_path", "mask_path"]], on="cell_id")
    Q = cfg["qc"]; rng = np.random.default_rng(Q["seed"])
    chosen = []
    for st in STAGES:
        g = df[df.stage == st].sort_values("cell_id")
        if st in ("R", "T") and len(g) > Q["random_common_per_stage"]:
            g = g.iloc[np.sort(rng.choice(len(g), Q["random_common_per_stage"], replace=False))]
        chosen.append(g)
    sel = pd.concat(chosen)
    cols = Q["sheet_cols"]; per_sheet = cols * 6
    sheet_of = {}
    for st, g in sel.groupby("stage", sort=False):
        g = g.sort_values(["species", "cell_id"])
        for s in range(0, len(g), per_sheet):
            part = g.iloc[s:s + per_sheet]
            rows = int(np.ceil(len(part) / cols))
            sheet = Image.new("RGB", (cols * (TILE + 4), rows * (TILE + 18)), (255, 255, 255))
            for j, r in enumerate(part.itertuples()):
                sheet.paste(_tile(r.crop_path, r.mask_path, r.cell_id),
                            ((j % cols) * (TILE + 4), (j // cols) * (TILE + 18)))
            name = f"{st}_{s // per_sheet + 1:02d}.png"
            sheet.save(sd / name, format="PNG")
            for cid in part.cell_id:
                sheet_of[cid] = name
    qc_path = qd / "qc.csv"
    new = sel[["cell_id", "image_id", "species", "stage"]].assign(
        sheet=sel.cell_id.map(sheet_of), decision="", reviewer="", note="")
    if qc_path.exists():                                   # keep earlier decisions
        old = read_qc(qc_path)
        new = new.drop(columns=["decision", "reviewer", "note"]).merge(
            old[["cell_id", "decision", "reviewer", "note"]], on="cell_id", how="left").fillna("")
    write_csv(new, qc_path)
    return {"sheets": len(set(sheet_of.values())), "cells_for_review": int(len(new)),
            "rare_cells_for_review": int(new.stage.isin(["S", "G"]).sum())}


def read_qc(path) -> pd.DataFrame:
    """Accepts ',' or ';' separators (Excel in French/European locales saves CSV with ';')."""
    q = pd.read_csv(path, dtype=str, sep=None, engine="python").fillna("")
    q.columns = [c.strip() for c in q.columns]
    for c in q.columns:
        q[c] = q[c].str.strip()
    return q


def mark(cfg: dict, stages: list[str], decision: str, reviewer: str, exclude=(), relabel=(),
         only_blank: bool = True) -> dict:
    assert_not_frozen(cfg, "qc-mark")
    """Record decisions without opening qc.csv in a spreadsheet program.
    stages: set `decision` for every crop of these stages (only blank ones unless only_blank=False).
    exclude: cell_ids to mark 'exclude'; relabel: 'cell_id=X' items to mark 'relabel:X'."""
    qd = out_dir(cfg, "qc"); q = read_qc(qd / "qc.csv")
    if decision and decision not in VALID:
        raise ValueError(f"decision must be one of {sorted(VALID)}")
    sel = q.stage.isin(stages) & ((q.decision == "") if only_blank else True)
    q.loc[sel, "decision"] = decision; q.loc[sel, "reviewer"] = reviewer
    unknown = []
    for cid in exclude:
        m = q.cell_id == cid
        if not m.any(): unknown.append(cid); continue
        q.loc[m, ["decision", "reviewer"]] = ["exclude", reviewer]
    for item in relabel:
        cid, st = item.split("=")
        m = q.cell_id == cid
        if not m.any() or st not in STAGES: unknown.append(item); continue
        q.loc[m, ["decision", "reviewer"]] = [f"relabel:{st}", reviewer]
    if unknown:
        raise ValueError(f"unknown cell_id or stage (nothing written): {unknown}")
    write_csv(q, qd / "qc.csv")
    return {"marked_by_stage": int(sel.sum()), "excluded": len(exclude), "relabelled": len(relabel),
            "decisions": q.decision.replace("", "(blank)").value_counts().to_dict()}


def apply_qc(cfg: dict, require_complete: bool = True) -> dict:
    assert_not_frozen(cfg, "qc-apply")
    md = out_dir(cfg, "manifests"); qd = out_dir(cfg, "qc")
    cells = pd.read_csv(md / "cells.csv", dtype={"slide": str, "field": str})
    qc = read_qc(qd / "qc.csv")
    bad = qc[(qc.decision != "") & ~qc.decision.isin(VALID)]
    if len(bad):
        raise ValueError(f"invalid QC decisions: {bad[['cell_id', 'decision']].values.tolist()[:10]}")
    missing_rare = qc[(qc.stage.isin(["S", "G"])) & (qc.decision == "")]
    if require_complete and len(missing_rare):
        raise RuntimeError(
            f"{len(missing_rare)} schizont/gametocyte crops have no QC decision (gate G1).\n"
            "Review data/qc/sheets/S_*.png and G_*.png, then record decisions, e.g.\n"
            "  python -m malprep qc-mark --stages S,G --decision ok --reviewer <initials> "
            "[--exclude <cell_id> ...] [--relabel <cell_id>=T ...]\n"
            "and rerun qc-apply.")
    dec = dict(zip(qc.cell_id, qc.decision))
    cells["qc_decision"] = cells.cell_id.map(dec).fillna("not_sampled")
    cells["stage_original"] = cells.stage
    rel = cells.qc_decision.str.startswith("relabel:")
    cells.loc[rel, "stage"] = cells.loc[rel, "qc_decision"].str.split(":").str[1]
    cells["stage_idx"] = cells.stage.map(STAGES.index)
    cells["label_changed"] = rel
    drop = cells.qc_decision == "exclude"
    excl = pd.read_csv(md / "exclusions.csv")
    excl = pd.concat([excl, cells[drop].assign(reason="QC", detail="excluded at visual QC")], ignore_index=True)
    final = cells[~drop].reset_index(drop=True)
    shas = {"cells_final.csv": write_csv(final, md / "cells_final.csv"),
            "exclusions_final.csv": write_csv(excl, md / "exclusions_final.csv"),
            "qc.csv": write_csv(qc, qd / "qc.csv")}
    for k, v in shas.items():
        append_digest(cfg, k, v)
    return {"final_cells": int(len(final)), "excluded_at_qc": int(drop.sum()), "relabelled": int(rel.sum()),
            "rare_unreviewed": int(len(missing_rare)),
            "stage_counts": final.stage.value_counts().reindex(STAGES, fill_value=0).to_dict()}
