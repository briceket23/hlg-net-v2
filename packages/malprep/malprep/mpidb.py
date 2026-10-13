"""Step 2: MP-IDB per-parasite manifest from YOLO-segmentation labels + original images.

Outputs (data/manifests/):
  cells.csv       one row per kept parasite (polygon geometry in original pixels)
  exclusions.csv  one row per dropped parasite with a reason code
  images.csv      one row per image (size, hash, filename stage codes vs labelled stages)
"""
from __future__ import annotations
import os, re
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from PIL import Image

from malcore import STAGES
from .io_utils import sha256_file, write_csv, out_dir, append_digest, posix, assert_not_frozen

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
SKIP_DIRS = {"gt", "masks", "mask", "labels", "__macosx"}   # GT masks share the image stems; macOS zip junk


def find_label_names(cfg: dict) -> Path:
    """data.yaml given in the config, else looked up next to / above the label folder."""
    P = cfg["paths"]
    cands = [P.get("label_names")] if P.get("label_names") else []
    lab = Path(P["mpidb_labels"])
    cands += [lab / "data.yaml", lab.parent / "data.yaml", lab.parent.parent / "data.yaml"]
    for c in cands:
        if c and Path(c).exists():
            return Path(c)
    raise FileNotFoundError(f"class-name file (data.yaml) not found; tried {cands}")


def load_class_names(path: str | Path) -> list[tuple[str, str]]:
    names = yaml.safe_load(open(path))["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    out = []
    for n in names:
        sp, st = n.split("_")
        if st not in STAGES:
            raise ValueError(f"unknown stage code in class name {n}")
        out.append((sp.lower(), st))
    return out


def parse_stem(stem: str) -> dict:
    """'1305121398-0012-R_T' -> slide '1305121398', field '0012', codes {'R','T'}.
    Tolerates '-' instead of '_' between codes (e.g. '1701151546-0008-R-T')."""
    parts = stem.split("-")
    slide = parts[0]
    field = parts[1] if len(parts) > 1 else ""
    codes = set()
    for p in parts[2:]:
        codes |= {c for c in re.split(r"[_\s]+", p) if c}
    anomalous = bool(len(parts) != 3 or not re.fullmatch(r"\d+-\d+-[A-Z](_[A-Z])*", stem))
    return {"slide": slide, "field": field, "codes": codes, "name_anomaly": anomalous}


def index_images(root: str | Path) -> dict[str, Path]:
    found: dict[str, list[Path]] = {}
    for dp, dns, fns in os.walk(root):
        dns[:] = sorted(d for d in dns if d.lower() not in SKIP_DIRS)
        for fn in sorted(fns):
            p = Path(dp) / fn
            if fn.startswith("._"):          # macOS resource forks
                continue
            if p.suffix.lower() in IMG_EXT:
                found.setdefault(p.stem, []).append(p)
    dup = {k: v for k, v in found.items() if len(v) > 1}
    if dup:
        raise RuntimeError(f"image stems found more than once (resolve before freezing): {list(dup.items())[:5]}")
    return {k: v[0] for k, v in found.items()}


def polygon_geometry(xy: np.ndarray) -> dict:
    x, y = xy[:, 0], xy[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    x0, y0, x1, y1 = float(x.min()), float(y.min()), float(x.max()), float(y.max())
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "box_w": x1 - x0, "box_h": y1 - y0,
            "box_side": max(x1 - x0, y1 - y0), "cx": (x0 + x1) / 2, "cy": (y0 + y1) / 2,
            "area_px": float(area), "n_vertices": int(len(xy))}


def build_manifest(cfg: dict, use_images: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    P, M = cfg["paths"], cfg["mpidb"]
    classes = load_class_names(find_label_names(cfg))
    images = index_images(P["mpidb_images"]) if use_images else {}
    exp_w, exp_h = M["expected_image_size"]
    cells, excl, imgs = [], [], []
    label_files = sorted(Path(P["mpidb_labels"]).glob("*.txt"))
    if not label_files:
        raise FileNotFoundError(f"no label files in {P['mpidb_labels']}")
    for lf in label_files:
        stem = lf.stem
        meta = parse_stem(stem)
        ip = images.get(stem)
        if use_images and ip is None:
            raise FileNotFoundError(f"no image for label file {lf.name}")
        if ip is not None:
            with Image.open(ip) as im:
                W, H = im.size
            img_sha = sha256_file(ip)
        else:
            W, H, img_sha = exp_w, exp_h, ""
        lines = [l.split() for l in lf.read_text().splitlines() if l.strip()]
        sp_set, st_set = set(), set()
        for i, tok in enumerate(lines):
            c = int(tok[0])
            sp, st = classes[c]
            coords = np.asarray(tok[1:], dtype=float)
            row = {"cell_id": f"{stem}__{i:03d}", "image_id": stem, "slide": meta["slide"], "field": meta["field"],
                   "species": sp, "stage": st, "stage_idx": STAGES.index(st), "class_id": c, "obj_index": i,
                   "img_w": W, "img_h": H, "label_file": lf.name}
            if coords.size < 6 or coords.size % 2:
                excl.append({**row, "reason": "POLY", "detail": f"{coords.size} coordinates"}); continue
            xy = coords.reshape(-1, 2) * np.array([W, H])
            g = polygon_geometry(xy)
            row.update(g)
            row["touches_border"] = bool(g["x0"] <= 1 or g["y0"] <= 1 or g["x1"] >= W - 1 or g["y1"] >= H - 1)
            row["polygon"] = " ".join(f"{v:.6f}" for v in coords)   # normalised, as in the label file
            if g["area_px"] < M["min_area_px"]:
                excl.append({**row, "reason": "AREA", "detail": f"{g['area_px']:.0f} px2"}); continue
            sp_set.add(sp); st_set.add(st)
            cells.append(row)
        imgs.append({"image_id": stem, "slide": meta["slide"], "field": meta["field"],
                     "image_path": posix(ip), "image_sha256": img_sha, "img_w": W, "img_h": H,
                     "size_ok": (W, H) == (exp_w, exp_h), "n_objects": len(lines),
                     "species": ",".join(sorted(sp_set)), "n_species": len(sp_set),
                     "name_codes": "_".join(sorted(meta["codes"])), "label_stages": "_".join(sorted(st_set)),
                     "codes_match": meta["codes"] == st_set, "name_anomaly": meta["name_anomaly"]})
    cells, excl, imgs = pd.DataFrame(cells), pd.DataFrame(excl), pd.DataFrame(imgs)
    # images mixing species are excluded whole (species is an image-level property in MP-IDB)
    mixed = set(imgs.loc[imgs.n_species > 1, "image_id"])
    if mixed:
        m = cells.image_id.isin(mixed)
        excl = pd.concat([excl, cells[m].assign(reason="SPECIES_MIX", detail="image has >1 species")])
        cells = cells[~m]
    images_unlabelled = sorted(set(images) - {f.stem for f in label_files})
    imgs.attrs["unlabelled_images"] = images_unlabelled
    return cells.reset_index(drop=True), excl.reset_index(drop=True), imgs


def write_manifest(cfg: dict, use_images: bool = True) -> dict:
    assert_not_frozen(cfg, "manifest")
    cells, excl, imgs = build_manifest(cfg, use_images)
    md = out_dir(cfg, "manifests")
    shas = {"cells.csv": write_csv(cells, md / "cells.csv"),
            "exclusions.csv": write_csv(excl if len(excl) else pd.DataFrame(columns=["cell_id", "reason", "detail"]),
                                        md / "exclusions.csv"),
            "images.csv": write_csv(imgs, md / "images.csv")}
    for k, v in shas.items():
        append_digest(cfg, k, v)
    report = {
        "label_files": int(len(imgs)), "cells_kept": int(len(cells)), "cells_excluded": int(len(excl)),
        "exclusions_by_reason": excl.reason.value_counts().to_dict() if len(excl) else {},
        "images_without_labels": imgs.attrs.get("unlabelled_images", []),
        "images_wrong_size": imgs.loc[~imgs.size_ok, "image_id"].tolist(),
        "filename_codes_mismatch": imgs.loc[~imgs.codes_match, "image_id"].tolist(),
        "filename_anomalies": imgs.loc[imgs.name_anomaly, "image_id"].tolist(),
        "multi_stage_images": int((imgs.label_stages.str.count("_") > 0).sum()),
        "stage_counts": cells.stage.value_counts().reindex(STAGES, fill_value=0).to_dict(),
        "used_images": use_images,
    }
    return report
