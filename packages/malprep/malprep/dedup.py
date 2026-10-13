"""Step 2.4: near-duplicate images (64-bit DCT pHash) and final grouping columns.

Adds to cells.csv:
  dup_group    images linked by near-duplicate pHash (Hamming <= threshold)
  group_slide  slide prefix, merged across slides that share a near-duplicate image
  group_image  image, merged across near-duplicates
"""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn

from .io_utils import out_dir, write_csv, append_digest, assert_not_frozen


def phash(path: str, hash_size: int = 8, highfreq: int = 4) -> int:
    n = hash_size * highfreq
    with Image.open(path) as im:
        a = np.asarray(im.convert("L").resize((n, n), Image.Resampling.LANCZOS), dtype=np.float64)
    d = dctn(a, norm="ortho")[:hash_size, :hash_size]
    med = np.median(d.ravel()[1:])          # exclude DC term
    bits = (d > med).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)


class UnionFind:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)    # deterministic representative


def thumb(path: str, size=(160, 120)) -> np.ndarray:
    with Image.open(path) as im:
        a = np.asarray(im.convert("L").resize(size, Image.Resampling.BILINEAR), dtype=np.float64)
    a -= a.mean()
    return a / (np.linalg.norm(a) + 1e-9)


def near_duplicates(images: pd.DataFrame, max_hamming: int, min_corr: float = 0.95) -> pd.DataFrame:
    """Two-stage test. pHash alone over-merges smear fields (mostly uniform pink background), so a
    pHash candidate is confirmed only if the mean-centred thumbnails also correlate >= min_corr."""
    hashes = {r.image_id: phash(r.image_path) for r in images.itertuples() if r.image_path}
    paths = dict(zip(images.image_id, images.image_path))
    thumbs: dict = {}
    pairs = []
    for (a, ha), (b, hb) in itertools.combinations(sorted(hashes.items()), 2):
        d = bin(ha ^ hb).count("1")
        if d <= max_hamming:
            for x in (a, b):
                if x not in thumbs:
                    thumbs[x] = thumb(paths[x])
            corr = float((thumbs[a] * thumbs[b]).sum())
            pairs.append({"image_a": a, "image_b": b, "hamming": d, "thumb_corr": round(corr, 4),
                          "confirmed": corr >= min_corr})
    images = images.copy()
    images["phash"] = images.image_id.map(lambda i: f"{hashes[i]:016x}" if i in hashes else "")
    return pd.DataFrame(pairs, columns=["image_a", "image_b", "hamming", "thumb_corr", "confirmed"]), images


def assign_groups(cfg: dict, use_images: bool = True) -> dict:
    assert_not_frozen(cfg, "dedup")
    md = out_dir(cfg, "manifests")
    cells = pd.read_csv(md / "cells.csv", dtype={"slide": str, "field": str})
    images = pd.read_csv(md / "images.csv", dtype={"slide": str, "field": str}).fillna({"image_path": ""})
    if use_images:
        pairs, images = near_duplicates(images, cfg["mpidb"]["phash_hamming_max"], cfg["mpidb"].get("dup_min_corr", 0.95))
    else:
        pairs = pd.DataFrame(columns=["image_a", "image_b", "hamming", "thumb_corr", "confirmed"])
        images["phash"] = ""
    conf = pairs[pairs.confirmed.astype(bool)] if len(pairs) else pairs
    uf_img = UnionFind(images.image_id)
    for p in conf.itertuples():
        uf_img.union(p.image_a, p.image_b)
    slide_of = dict(zip(images.image_id, images.slide))
    uf_sl = UnionFind(sorted(set(images.slide)))
    for p in conf.itertuples():
        uf_sl.union(slide_of[p.image_a], slide_of[p.image_b])
    images["dup_group"] = images.image_id.map(uf_img.find)
    images["group_image"] = images["dup_group"]
    images["group_slide"] = images.slide.map(uf_sl.find)
    m = images.set_index("image_id")
    for c in ["dup_group", "group_image", "group_slide"]:
        cells[c] = cells.image_id.map(m[c])
    shas = {"cells.csv": write_csv(cells, md / "cells.csv"),
            "images.csv": write_csv(images, md / "images.csv"),
            "near_duplicates.csv": write_csv(pairs, md / "near_duplicates.csv")}
    for k, v in shas.items():
        append_digest(cfg, k, v)
    cross = [p for p in conf.itertuples() if slide_of[p.image_a] != slide_of[p.image_b]]
    return {"phash_candidates": int(len(pairs)), "confirmed_duplicates": int(len(conf)),
            "cross_slide_duplicates": len(cross),
            "n_groups_slide": int(images.group_slide.nunique()), "n_groups_image": int(images.group_image.nunique()),
            "phash_computed": bool(use_images)}
