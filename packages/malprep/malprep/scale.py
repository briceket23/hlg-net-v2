"""Magnification check between MP-IDB and BBBC041, using red-blood-cell diameter as the ruler.

MP-IDB polygons outline the parasite, while BBBC041 boxes enclose the whole infected red cell, so
object boxes cannot be compared across the two datasets. Red cells can: BBBC041 annotates every red
cell, and for MP-IDB we estimate the diameter of isolated red cells on a sample of fields
(Otsu threshold, hole filling, connected components, roundness filter). Inspect the overlay PNGs.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from .io_utils import out_dir, write_json


def otsu(a: np.ndarray) -> float:
    hist, edges = np.histogram(a, bins=256, range=(0, 256))
    p = hist / hist.sum(); w = np.cumsum(p); mu = np.cumsum(p * np.arange(256)); mt = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (mt * w - mu) ** 2 / (w * (1 - w))
    return float(np.nanargmax(sb))


def rbc_diameters(path: str, reduce: int = 4, min_d: float = 40, max_d: float = 250) -> tuple[np.ndarray, list]:
    """Diameters (native px) of isolated, round, cell-sized components."""
    with Image.open(path) as im:
        g = np.asarray(im.convert("RGB").reduce(reduce), dtype=np.float32)
    green = ndi.gaussian_filter(g[..., 1], 1.5)            # red cells absorb green; background is bright
    fg = green < otsu(green)
    fg = ndi.binary_opening(fg, iterations=1)
    fg = ndi.binary_fill_holes(fg)
    lab, n = ndi.label(fg)
    if n == 0:
        return np.array([]), []
    idx = np.arange(1, n + 1)
    area = ndi.sum(fg, lab, idx)
    sl = ndi.find_objects(lab)
    out, boxes = [], []
    H, W = fg.shape
    for i, s in enumerate(sl):
        if s is None:
            continue
        h, w = s[0].stop - s[0].start, s[1].stop - s[1].start
        if s[0].start == 0 or s[1].start == 0 or s[0].stop == H or s[1].stop == W:
            continue                                            # touches the border
        d_eq = 2 * np.sqrt(area[i] / np.pi) * reduce
        aspect = min(h, w) / max(h, w)
        extent = area[i] / (h * w)                              # disc = pi/4 = 0.785
        if min_d <= d_eq <= max_d and aspect > 0.8 and 0.7 < extent < 0.85:
            out.append(d_eq); boxes.append((s[1].start * reduce, s[0].start * reduce, s[1].stop * reduce, s[0].stop * reduce))
    return np.asarray(out), boxes


def mpidb_rbc_scale(cfg: dict, per_slide: int = 3, seed: int = 0) -> dict:
    md = out_dir(cfg, "manifests"); qd = out_dir(cfg, "qc", "rbc_scale")
    im = pd.read_csv(md / "images.csv", dtype={"slide": str})
    im = im[im.image_path.fillna("") != ""]
    rng = np.random.default_rng(seed)
    # stratify the sample by slide so one large slide does not dominate
    pick = pd.concat([g.sample(min(len(g), per_slide), random_state=seed)
                      for _, g in im.groupby("slide", sort=True)])
    rows, allds = [], []
    for j, r in enumerate(pick.itertuples()):
        d, boxes = rbc_diameters(r.image_path)
        rows.append({"image_id": r.image_id, "slide": r.slide, "n_rbc": len(d),
                     "median_d": float(np.median(d)) if len(d) else np.nan})
        allds += list(d)
        if j < 6:                                               # overlays for visual verification
            with Image.open(r.image_path) as x:
                ov = x.convert("RGB").reduce(4); dr = ImageDraw.Draw(ov)
                for b in boxes:
                    dr.rectangle([v / 4 for v in b], outline=(0, 200, 0), width=2)
                ov.save(qd / f"{r.image_id}.png")
    per = pd.DataFrame(rows)
    per.to_csv(qd / "per_image.csv", index=False)
    res = {"n_images": int(len(per)), "n_rbc": int(len(allds)),
           "mpidb_rbc_diameter_px": round(float(np.median(allds)), 1) if allds else None,
           "iqr": [round(float(np.percentile(allds, q)), 1) for q in (25, 75)] if allds else None,
           "between_slide_sd": round(float(per.groupby("slide").median_d.median().std()), 1) if len(per) > 1 else None,
           "note": "verify overlays in data/qc/rbc_scale/ before using this value"}
    write_json(res, out_dir(cfg, "tables") / "mpidb_rbc_scale.json")
    return res
