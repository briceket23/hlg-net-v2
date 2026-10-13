"""Step 3: fixed-window crops at native resolution (one rule for MP-IDB and BBBC041).

Rule: a square window of `size` px centred on the object's box centre, no resizing, so absolute
object size is preserved. Pixels outside the image take the per-image median RGB. An object whose
box side * margin exceeds `size` gets a window of that side, downscaled to `size` (flag `scaled`).
Crops and masks are lossless PNG written with fixed parameters, so reruns are byte-identical.
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from .io_utils import out_dir, write_csv, append_digest, sha256_file, posix, assert_not_frozen

PNG_KW = dict(format="PNG", optimize=False, compress_level=6)


def image_median_rgb(im: Image.Image) -> tuple[int, int, int]:
    small = im.convert("RGB").reduce(8)
    a = np.asarray(small).reshape(-1, 3)
    return tuple(int(v) for v in np.median(a, axis=0))


def window_for(cx: float, cy: float, box_side: float, size: int, margin: float,
               base_side: int | None = None) -> tuple[int, int, int, float]:
    """base_side: native window side before resizing to `size` (default = size, i.e. no rescaling).
    A magnification-matched crop uses base_side = size / (target px per um / source px per um)."""
    base = base_side or size
    side = base if box_side * margin <= base else int(math.ceil(box_side * margin))
    x0 = int(round(cx - side / 2)); y0 = int(round(cy - side / 2))
    return x0, y0, side, size / side


def cut(im: Image.Image, x0: int, y0: int, side: int, fill: tuple[int, int, int]) -> Image.Image:
    canvas = Image.new("RGB", (side, side), fill)
    W, H = im.size
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, x0 + side), min(H, y0 + side)
    if sx1 > sx0 and sy1 > sy0:
        canvas.paste(im.crop((sx0, sy0, sx1, sy1)), (sx0 - x0, sy0 - y0))
    return canvas


def object_mask(poly_px: np.ndarray | None, box: tuple[float, float, float, float], x0: int, y0: int,
                side: int) -> Image.Image:
    m = Image.new("L", (side, side), 0)
    d = ImageDraw.Draw(m)
    if poly_px is not None:
        d.polygon([(float(x - x0), float(y - y0)) for x, y in poly_px], fill=255)
    else:
        bx0, by0, bx1, by1 = box
        d.rectangle([bx0 - x0, by0 - y0, bx1 - x0, by1 - y0], fill=255)
    return m


def extract(df: pd.DataFrame, path_col: str, crop_dir: Path, size: int, margin: float,
            poly_col: str | None = None, scale_col: str | None = None) -> pd.DataFrame:
    """scale_col: optional per-row factor f; the native window is size*f px, resized to size."""
    crop_dir = Path(crop_dir); (crop_dir / "img").mkdir(parents=True, exist_ok=True)
    (crop_dir / "mask").mkdir(parents=True, exist_ok=True)
    rows = []
    for img_path, g in df.groupby(path_col, sort=True):
        with Image.open(img_path) as im0:
            im = im0.convert("RGB"); im.load()
        fill = image_median_rgb(im)
        W, H = im.size
        for r in g.sort_values("cell_id").itertuples():
            base = int(round(size * getattr(r, scale_col))) if scale_col else None
            x0, y0, side, scale = window_for(r.cx, r.cy, r.box_side, size, margin, base)
            crop = cut(im, x0, y0, side, fill)
            poly = None
            if poly_col and isinstance(getattr(r, poly_col), str) and getattr(r, poly_col):
                poly = np.asarray(getattr(r, poly_col).split(), dtype=float).reshape(-1, 2) * np.array([W, H])
            mask = object_mask(poly, (r.x0, r.y0, r.x1, r.y1), x0, y0, side)
            if side != size:
                crop = crop.resize((size, size), Image.Resampling.LANCZOS)
                mask = mask.resize((size, size), Image.Resampling.NEAREST)
            cp = crop_dir / "img" / f"{r.cell_id}.png"; mp = crop_dir / "mask" / f"{r.cell_id}.png"
            crop.save(cp, **PNG_KW); mask.save(mp, **PNG_KW)
            pad = (x0 < 0) or (y0 < 0) or (x0 + side > W) or (y0 + side > H)
            rows.append({"cell_id": r.cell_id, "crop_path": posix(cp), "mask_path": posix(mp),
                         "win_x0": x0, "win_y0": y0, "win_side": side, "scale": round(scale, 6),
                         "scaled": side != size, "oversize": bool(r.box_side * margin > (base or size)), "padded": pad, "pad_rgb": "%d,%d,%d" % fill,
                         "crop_sha256": sha256_file(cp), "mask_sha256": sha256_file(mp)})
    return pd.DataFrame(rows)


def window_neighbours(df: pd.DataFrame, idx: pd.DataFrame, group_col: str = "image_id") -> pd.DataFrame:
    """Other annotated objects whose centre lies inside each crop window (label-noise flag).
    n_other_in_window: any stage; n_other_stage_in_window: a different stage than the cell's own."""
    w = df[["cell_id", group_col, "cx", "cy", "stage"]].merge(idx[["cell_id", "win_x0", "win_y0", "win_side"]], on="cell_id")
    out = []
    for _, g in w.groupby(group_col, sort=False):
        cx, cy, st = g.cx.to_numpy(), g.cy.to_numpy(), g.stage.to_numpy()
        for r in g.itertuples():
            inside = (cx >= r.win_x0) & (cx < r.win_x0 + r.win_side) & (cy >= r.win_y0) & (cy < r.win_y0 + r.win_side)
            inside &= g.cell_id.to_numpy() != r.cell_id
            out.append((r.cell_id, int(inside.sum()), int((inside & (st != r.stage)).sum())))
    return pd.DataFrame(out, columns=["cell_id", "n_other_in_window", "n_other_stage_in_window"])


def run_mpidb(cfg: dict) -> dict:
    assert_not_frozen(cfg, "crops")
    md = out_dir(cfg, "manifests")
    cells = pd.read_csv(md / "cells.csv", dtype={"slide": str, "field": str})
    imgs = pd.read_csv(md / "images.csv", dtype={"slide": str})
    cells = cells.merge(imgs[["image_id", "image_path"]], on="image_id", how="left")
    if cells.image_path.isna().any() or (cells.image_path == "").any():
        raise RuntimeError("crops need the original images: rerun `manifest` without --no-images")
    C = cfg["crop"]
    idx = extract(cells, "image_path", out_dir(cfg, "crops", "mpidb"), C["size"], C["oversize_margin"], "polygon")
    idx = idx.merge(window_neighbours(cells, idx), on="cell_id", how="left")
    sha = write_csv(idx, md / "crops.csv"); append_digest(cfg, "crops.csv", sha)
    st = cells.set_index("cell_id").stage
    other = idx.assign(stage=idx.cell_id.map(st))
    return {"crops": int(len(idx)), "scaled": int(idx.scaled.sum()), "padded": int(idx.padded.sum()),
            "with_other_object_in_window": int((idx.n_other_in_window > 0).sum()),
            "with_other_stage_in_window_by_stage":
                other[other.n_other_stage_in_window > 0].stage.value_counts().to_dict()}


def verify_reproducible(cfg: dict, n: int = 25, seed: int = 0) -> dict:
    """Re-extract a random sample into a temp dir and compare bytes with the stored crops."""
    import tempfile
    md = out_dir(cfg, "manifests")
    cells = pd.read_csv(md / "cells.csv", dtype={"slide": str, "field": str})
    imgs = pd.read_csv(md / "images.csv", dtype={"slide": str})
    idx = pd.read_csv(md / "crops.csv")
    s = cells.sample(min(n, len(cells)), random_state=seed).merge(imgs[["image_id", "image_path"]], on="image_id")
    with tempfile.TemporaryDirectory() as td:
        new = extract(s, "image_path", Path(td), cfg["crop"]["size"], cfg["crop"]["oversize_margin"], "polygon")
    j = new.merge(idx, on="cell_id", suffixes=("_new", "_old"))
    bad = j[(j.crop_sha256_new != j.crop_sha256_old) | (j.mask_sha256_new != j.mask_sha256_old)]
    return {"checked": int(len(j)), "mismatches": bad.cell_id.tolist()}
