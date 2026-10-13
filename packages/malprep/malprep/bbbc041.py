"""Step 4: BBBC041 external stage set (zero-shot only; never read by training code).

Layout: <bbbc041>/training.json, <bbbc041>/test.json, <bbbc041>/images/ (pathnames in the JSON are
'/images/<uuid>.png|jpg'). __MACOSX folders are ignored. Checked on the released files (12 Oct 2026):
  training: 1,208 PNG images, 1600x1200;  test: 120 JPG images, 1944x1383
  categories: red blood cell, ring, trophozoite, schizont, gametocyte, leukocyte, difficult

Geometry differs from MP-IDB: BBBC041 boxes enclose the whole infected red cell, MP-IDB polygons
outline the parasite. Crops are centred on the box either way; scale is compared via red cells.
Two crop sets are written:
  crops/bbbc041          as acquired (primary external analysis; same 256 px native window)
  crops/bbbc041_matched  window rescaled so red cells have the MP-IDB diameter (declared sensitivity)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

from malcore import STAGES
from .io_utils import out_dir, write_csv, write_json, append_digest, posix, assert_not_frozen, write_text
from .crops import extract

CAT2STAGE = {"ring": "R", "trophozoite": "T", "schizont": "S", "gametocyte": "G"}
SUBSETS = [("train", "training.json"), ("test", "test.json")]


def _side(bb):
    return max(bb["maximum"]["r"] - bb["minimum"]["r"], bb["maximum"]["c"] - bb["minimum"]["c"])


def parse(cfg: dict) -> tuple[pd.DataFrame, dict]:
    root = Path(cfg["paths"]["bbbc041"])
    keep = {CAT2STAGE[k] for k in cfg["bbbc041"]["keep"]}
    rows, cats, missing, rbc, shapes = [], {}, [], {}, {}
    for subset, fn in SUBSETS:
        f = root / fn
        if not f.exists():
            raise FileNotFoundError(f"{f} (set paths.bbbc041 to the folder holding training.json)")
        for entry in json.loads(f.read_text()):
            im = entry["image"]
            rel = im["pathname"].lstrip("/")
            ip = root / rel
            if not ip.exists():
                missing.append(rel)
            H, W = im["shape"]["r"], im["shape"]["c"]
            shapes.setdefault(subset, set()).add((W, H))
            stem = Path(rel).stem
            for i, o in enumerate(entry.get("objects", [])):
                cat = o["category"].strip().lower()
                cats[(subset, cat)] = cats.get((subset, cat), 0) + 1
                if cat == "red blood cell":
                    rbc.setdefault(subset, []).append(_side(o["bounding_box"]))
                st = CAT2STAGE.get(cat)
                if st is None or st not in keep:
                    continue
                bb = o["bounding_box"]
                y0, x0 = bb["minimum"]["r"], bb["minimum"]["c"]
                y1, x1 = bb["maximum"]["r"], bb["maximum"]["c"]
                rows.append({"cell_id": f"bbbc_{subset}_{stem}__{i:03d}", "image_id": stem, "subset": subset,
                             "image_path": posix(ip), "format": ip.suffix.lower().lstrip("."),
                             "image_md5": im.get("checksum", ""),
                             "species": "vivax", "stage": st, "stage_idx": STAGES.index(st),
                             "img_w": W, "img_h": H, "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                             "box_w": x1 - x0, "box_h": y1 - y0, "box_side": max(x1 - x0, y1 - y0),
                             "cx": (x0 + x1) / 2, "cy": (y0 + y1) / 2})
    df = pd.DataFrame(rows)
    rbc_med = {s: float(np.median(v)) for s, v in rbc.items()}
    df["rbc_median_side_subset"] = df.subset.map(rbc_med)
    report = {"category_counts": {f"{s}:{c}": n for (s, c), n in sorted(cats.items())},
              "image_sizes": {s: sorted(v) for s, v in shapes.items()},
              "rbc_median_box_side": rbc_med,
              "test_over_train_rbc_ratio": round(rbc_med.get("test", np.nan) / rbc_med.get("train", np.nan), 3),
              "missing_images": missing[:20], "n_missing_images": len(missing)}
    return df, report


def run(cfg: dict, make_crops: bool = True) -> dict:
    assert_not_frozen(cfg, "bbbc041")
    md = out_dir(cfg, "manifests"); td = out_dir(cfg, "tables")
    df, report = parse(cfg)
    C = cfg["crop"]
    if make_crops:
        if report["n_missing_images"]:
            raise FileNotFoundError(f"{report['n_missing_images']} BBBC041 images missing, e.g. {report['missing_images'][:3]}")
        idx = extract(df, "image_path", out_dir(cfg, "crops", "bbbc041"), C["size"], C["oversize_margin"])
        from .crops import window_neighbours
        df = df.merge(idx, on="cell_id")
        df = df.merge(window_neighbours(df, idx), on="cell_id", how="left")
        report["crops"] = {"scaled_oversize": int(df.oversize.sum()), "padded": int(df.padded.sum()),
                           "with_other_infected_cell_in_window": int((df.n_other_in_window > 0).sum())}
        # magnification-matched sensitivity crops: window side = 256 * (BBBC041 RBC / MP-IDB RBC)
        mp_rbc = cfg["bbbc041"].get("mpidb_rbc_diameter_px")
        if mp_rbc:
            # RBC box side ~ diameter; a ratio > 1 means BBBC041 is more magnified -> larger native window
            df["match_factor"] = df.rbc_median_side_subset / float(mp_rbc)
            keep = [c for c in df.columns if c not in idx.columns or c == "cell_id"]
            keep = [c for c in keep if c not in ("n_other_in_window", "n_other_stage_in_window")]
            idx2 = extract(df[keep], "image_path",
                           out_dir(cfg, "crops", "bbbc041_matched"), C["size"], C["oversize_margin"],
                           scale_col="match_factor")
            df = df.merge(idx2.add_suffix("_matched").rename(columns={"cell_id_matched": "cell_id"}), on="cell_id")
            report["matched_crops"] = {"mpidb_rbc_diameter_px": mp_rbc,
                                       "factor_by_subset": df.groupby("subset").match_factor.first().round(3).to_dict()}
        else:
            report["matched_crops"] = "skipped: set bbbc041.mpidb_rbc_diameter_px (run `python -m malprep rbc-scale`)"
    sha = write_csv(df, md / "bbbc041_cells.csv"); append_digest(cfg, "bbbc041_cells.csv", sha)
    # Table 5a
    t = pd.crosstab(df.stage, df.subset).reindex(index=STAGES, columns=["train", "test"], fill_value=0)
    t["total"] = t.sum(axis=1)
    for s in ("train", "test"):
        t[f"images_{s}"] = df[df.subset == s].groupby("stage").image_id.nunique().reindex(STAGES).fillna(0).astype(int)
    t["low_support_flag_test"] = t["test"] < cfg["bbbc041"]["min_stage_cells_flag"]
    t["median_box_side"] = df.groupby("stage").box_side.median().reindex(STAGES)
    write_csv(t.reset_index(), td / "table5a_external_data.csv")
    report["cells_per_stage_subset"] = {s: df[df.subset == s].stage.value_counts().reindex(STAGES, fill_value=0).to_dict()
                                        for s in ("train", "test")}
    report["note_geometry"] = ("BBBC041 boxes enclose the infected red cell; MP-IDB polygons outline the parasite. "
                               "Compare magnification via red-cell diameter, not object boxes.")
    write_json(report, td / "bbbc041_report.json")
    exp = md / "bbbc041_exposure.md"
    if not exp.exists():
        write_text(
            "# BBBC041 exposure statement (fill in and sign before gate G1)\n\n"
            "- Pretrained weights used by any arm (identifier + SHA-256):\n"
            "- Were any BBBC041 images used for pretraining, fine-tuning, model selection, crop-rule or\n"
            "  hyperparameter decisions for the models of this study?  yes / no\n"
            "- Earlier studies that used BBBC041 (e.g. the transfer study) and why they do not affect these models:\n"
            "- Conclusion: zero-shot / external transfer after exposure\n"
            "- Signed / date:\n", exp)
    return report
