"""Synthetic MP-IDB-like and BBBC041-like data for dry runs and unit tests (never for results).

render_mpidb_images(): draws an image for every real label file, so the full pipeline can be
dry-run on the real labels before the original images are copied in.
make_toy(): a small self-contained dataset (labels + images + BBBC041 JSON) used by the tests.
"""
from __future__ import annotations
import json, shutil
from pathlib import Path
import numpy as np
import yaml
from PIL import Image, ImageDraw

NAMES = [f"{sp}_{st}" for sp in ["falciparum", "vivax", "ovale", "malariae"] for st in ["R", "S", "T", "G"]]
SPECIES_DIR = {"falciparum": "Falciparum", "vivax": "Vivax", "ovale": "Ovale", "malariae": "Malariae"}
STAGE_COL = {"R": (120, 40, 140), "T": (90, 30, 120), "S": (70, 20, 100), "G": (50, 20, 90)}


def _background(W, H, rng):
    base = np.array([226, 196, 205], dtype=np.float32)
    small = rng.normal(0, 6, (H // 16 + 1, W // 16 + 1, 3)).astype(np.float32)
    img = Image.fromarray(np.clip(base + small, 0, 255).astype(np.uint8)).resize((W, H), Image.Resampling.BILINEAR)
    return img


def render(label_path: Path, out_path: Path, names: list[str], W=2592, H=1944, seed=0):
    rng = np.random.default_rng(seed)
    im = _background(W, H, rng); d = ImageDraw.Draw(im)
    for line in label_path.read_text().splitlines():
        t = line.split()
        if not t:
            continue
        st = names[int(t[0])].split("_")[1]
        xy = (np.asarray(t[1:], float).reshape(-1, 2) * [W, H]).tolist()
        d.polygon([tuple(p) for p in xy], fill=STAGE_COL[st])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path, quality=92)


def render_mpidb_images(label_dir: str, names_yaml: str, out_root: str, W=2592, H=1944):
    names = yaml.safe_load(open(names_yaml))["names"]
    for i, lf in enumerate(sorted(Path(label_dir).glob("*.txt"))):
        first = lf.read_text().split()
        sp = names[int(first[0])].split("_")[0] if first else "falciparum"
        render(lf, Path(out_root) / SPECIES_DIR[sp] / "img" / f"{lf.stem}.jpg", names, W, H, seed=i)


def _poly(cx, cy, r, n=24):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + r * np.cos(a), cy + r * 0.85 * np.sin(a)], 1)


def make_toy(root: str | Path, W=800, H=600, seed=0) -> dict:
    """16 images on 6 slides; rare stages on several slides; one exact duplicate image."""
    root = Path(root); rng = np.random.default_rng(seed)
    lab = root / "labels"; img = root / "mpidb"; lab.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(yaml.safe_dump({"names": NAMES, "nc": 16}))
    plan = [  # slide, field, species, stages per object
        ("1001", "0001", "falciparum", "RRRRT"), ("1001", "0002", "falciparum", "RRRS"), ("1001", "0003", "falciparum", "RRG"),
        ("1002", "0001", "vivax", "RRTG"), ("1002", "0002", "vivax", "RS"), ("1002", "0003", "vivax", "RRRR"),
        ("1003", "0001", "ovale", "TTG"), ("1003", "0002", "ovale", "RS"), ("1003", "0003", "ovale", "RRT"),
        ("1004", "0001", "malariae", "TS"), ("1004", "0002", "malariae", "TG"), ("1004", "0003", "malariae", "RRTT"),
        ("1005", "0001", "falciparum", "RRTSG"), ("1005", "0002", "falciparum", "RRR"),
        ("1006", "0001", "vivax", "RTSG"), ("1006", "0002", "vivax", "RRT"),
    ]
    size = {"R": 30, "T": 40, "S": 55, "G": 75}
    for i, (sl, fd, sp, sts) in enumerate(plan):
        codes = "_".join(sorted(set(sts), key="RTSG".index))
        stem = f"{sl}-{fd}-{codes}"
        lines = []
        for j, st in enumerate(sts):
            cx, cy = 100 + 130 * j, 120 + 60 * (j % 3)
            p = _poly(cx, cy, size[st] / 2) / [W, H]
            lines.append(f"{NAMES.index(f'{sp}_{st}')} " + " ".join(f"{v:.6f}" for v in p.ravel()))
        (lab / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        render(lab / f"{stem}.txt", img / SPECIES_DIR[sp] / "img" / f"{stem}.jpg", NAMES, W, H, seed=i)
        # MP-IDB ground-truth masks share the stem and must be ignored by the image index
        (img / SPECIES_DIR[sp] / "gt").mkdir(parents=True, exist_ok=True)
        Image.new("L", (W, H)).save(img / SPECIES_DIR[sp] / "gt" / f"{stem}.jpg")
    # exact duplicate of slide 1001 field 0001 filed under slide 1003 -> must be merged into one group
    dup_stem = "1003-0009-R_T"
    shutil.copy(lab / "1001-0001-R_T.txt", lab / f"{dup_stem}.txt")
    shutil.copy(img / "Falciparum" / "img" / "1001-0001-R_T.jpg", img / "Ovale" / "img" / f"{dup_stem}.jpg")
    # the duplicate keeps falciparum labels; fine for grouping tests
    # BBBC041-like
    b = root / "bbbc041"; (b / "images").mkdir(parents=True, exist_ok=True)
    entries = {"training.json": [], "test.json": []}
    for i in range(4):
        fn = f"img{i}.png"; im = _background(400, 300, rng); d = ImageDraw.Draw(im)
        objs = []
        for j, cat in enumerate(["red blood cell", "ring", "trophozoite", "schizont", "gametocyte", "difficult"]):
            x0, y0 = 20 + 60 * j, 40 + 20 * (i % 2); s = 30 + 5 * j
            d.ellipse([x0, y0, x0 + s, y0 + s], fill=(120, 40, 140))
            objs.append({"bounding_box": {"minimum": {"r": y0, "c": x0}, "maximum": {"r": y0 + s, "c": x0 + s}},
                         "category": cat})
        im.save(b / "images" / fn)
        entries["training.json" if i < 3 else "test.json"].append(
            {"image": {"checksum": "", "pathname": f"/images/{fn}", "shape": {"r": 300, "c": 400, "channels": 3}},
             "objects": objs})
    for k, v in entries.items():
        (b / k).write_text(json.dumps(v))
    cfg = {
        "paths": {"mpidb_images": str(img), "mpidb_labels": str(lab), "label_names": str(root / "data.yaml"),
                  "bbbc041": str(b), "out": str(root / "data"), "splits": str(root / "splits")},
        "label_source": "synthetic toy data",
        "mpidb": {"expected_image_size": [W, H], "min_area_px": 100, "phash_hamming_max": 4, "group_unit": "slide"},
        "crop": {"size": 128, "oversize_margin": 1.1, "pad": "median"},
        "bbbc041": {"keep": ["ring", "trophozoite", "schizont", "gametocyte"], "min_stage_cells_flag": 30, "mpidb_rbc_diameter_px": 40},
        "qc": {"random_common_per_stage": 5, "sheet_cols": 5, "seed": 7},
        "splits": {"K": 2, "repeats": [1, 2, 3], "split_seed_base": 100, "inner_K": 2, "n_candidates": 30, "n_inner_candidates": 20,
                   "rare_stages": ["G", "S"]},
    }
    (root / "data_cfg.yaml").write_text(yaml.safe_dump(cfg))
    return cfg
