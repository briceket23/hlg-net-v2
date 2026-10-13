"""End-to-end tests on a synthetic toy dataset: python -m pytest -q"""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from malprep.synthetic import make_toy
from malprep.io_utils import load_config
from malprep import mpidb, dedup, crops, qc, tables, bbbc041, splits
from malprep.mpidb import parse_stem
from malprep.splits import assign_groups, group_table


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    root = tmp_path_factory.mktemp("toy")
    make_toy(root)
    c = load_config(root / "data_cfg.yaml")
    mpidb.write_manifest(c); dedup.assign_groups(c); crops.run_mpidb(c); qc.make_sheets(c)
    # mark every reviewed crop ok, exclude one ring, relabel one trophozoite
    q = pd.read_csv(Path(c["paths"]["out"]) / "qc" / "qc.csv", dtype=str).fillna("")
    q["decision"] = "ok"
    q.loc[q.index[q.stage == "R"][0], "decision"] = "exclude"
    q.loc[q.index[q.stage == "T"][0], "decision"] = "relabel:R"
    q.to_csv(Path(c["paths"]["out"]) / "qc" / "qc.csv", index=False)
    qc.apply_qc(c); tables.table1(c); bbbc041.run(c)
    for u in ("slide", "image"):
        splits.make_splits(c, unit=u)
    return c


def md(c, f):
    return pd.read_csv(Path(c["paths"]["out"]) / "manifests" / f, dtype={"slide": str, "group_slide": str})


def test_parse_stem_tolerates_hyphen_codes():
    assert parse_stem("1701151546-0008-R-T")["codes"] == {"R", "T"}
    assert parse_stem("1701151546-0008-R-T")["name_anomaly"]
    assert parse_stem("1305121398-0012-R_T")["codes"] == {"R", "T"}
    assert not parse_stem("1305121398-0012-R_T")["name_anomaly"]


def test_manifest_counts_and_gt_folders_ignored(cfg):
    cells, im = md(cfg, "cells.csv"), md(cfg, "images.csv")
    assert len(im) == 17 and im.codes_match.all()
    assert cells.cell_id.is_unique
    assert set(cells.stage) == {"R", "T", "S", "G"}
    assert all("/gt/" not in p for p in im.image_path)


def test_duplicate_image_merges_slides(cfg):
    nd = md(cfg, "near_duplicates.csv")
    assert len(nd) >= 1
    im = md(cfg, "images.csv")
    g = im.set_index("image_id").group_slide
    assert g["1003-0009-R_T"] == g["1001-0001-R_T"]          # slides 1001 and 1003 merged


def test_crops_size_centre_and_reproducible(cfg):
    cr = md(cfg, "crops.csv"); cells = md(cfg, "cells.csv").merge(cr, on="cell_id")
    r = cells.iloc[0]
    im = Image.open(r.crop_path); assert im.size == (128, 128)
    m = np.asarray(Image.open(r.mask_path)) > 0
    ys, xs = np.nonzero(m)
    assert abs(xs.mean() - 64) < 3 and abs(ys.mean() - 64) < 3
    rep = crops.verify_reproducible(cfg, n=len(cells))
    assert rep["mismatches"] == [] and rep["checked"] == len(cells)


def test_qc_apply(cfg):
    f = md(cfg, "cells_final.csv"); c = md(cfg, "cells.csv")
    assert len(f) == len(c) - 1 and f.label_changed.sum() == 1


def test_table1_total(cfg):
    t = pd.read_csv(Path(cfg["paths"]["out"]) / "tables" / "table1_data.csv")
    assert int(t.loc[t.species == "total", "cells"].iloc[0]) == len(md(cfg, "cells_final.csv"))


def test_bbbc041(cfg):
    b = md(cfg, "bbbc041_cells.csv")
    assert set(b.stage) == {"R", "T", "S", "G"} and len(b) == 16
    assert set(b.subset) == {"train", "test"}


@pytest.mark.parametrize("unit", ["slide", "image"])
def test_splits_pass_verification(cfg, unit):
    assert splits.verify(cfg, unit) == []


def test_assign_groups_deterministic_and_complete(cfg):
    g = group_table(md(cfg, "cells_final.csv"), "group_image")
    a, b = assign_groups(g, 2, 5), assign_groups(g, 2, 5)
    assert a.equals(b) and a.notna().all() and set(a) == {0, 1}
    for k in (0, 1):  # every fold receives gametocytes and schizonts
        assert g.loc[a.index[a == k], "G"].sum() > 0 and g.loc[a.index[a == k], "S"].sum() > 0


def test_tampered_manifest_detected(cfg, tmp_path):
    p = Path(cfg["paths"]["out"]) / "manifests" / "cells_final.csv"
    orig = p.read_bytes()
    try:
        p.write_bytes(orig + b"\n")
        assert any("manifest changed" in f for f in splits.verify(cfg, "slide"))
    finally:
        p.write_bytes(orig)


def test_bbbc041_rbc_ratio_and_matched_crops(cfg):
    import json
    rep = json.loads((Path(cfg["paths"]["out"]) / "tables" / "bbbc041_report.json").read_text())
    assert rep["rbc_median_box_side"]["train"] > 0
    b = md(cfg, "bbbc041_cells.csv")
    assert "crop_path_matched" in b and b.match_factor.gt(0).all()
    r = b.iloc[0]
    assert Image.open(r.crop_path_matched).size == (128, 128)
    assert r.win_side_matched == int(round(128 * r.match_factor)) or r.scaled_matched


def test_rbc_diameter_estimator(tmp_path):
    from PIL import ImageDraw
    from malprep.scale import rbc_diameters
    im = Image.new("RGB", (1600, 1200), (235, 215, 220)); d = ImageDraw.Draw(im)
    for i in range(5):
        for j in range(4):
            cx, cy = 200 + 280 * i, 200 + 260 * j
            d.ellipse([cx - 50, cy - 50, cx + 50, cy + 50], fill=(200, 120, 150))   # 100 px discs
    p = tmp_path / "f.png"; im.save(p)
    ds, _ = rbc_diameters(str(p))
    assert len(ds) >= 15 and abs(np.median(ds) - 100) < 6


def test_index_skips_macos_junk(tmp_path):
    from malprep.mpidb import index_images
    (tmp_path / "__MACOSX").mkdir(); (tmp_path / "img").mkdir()
    Image.new("RGB", (4, 4)).save(tmp_path / "img" / "a-1-R.jpg")
    Image.new("RGB", (4, 4)).save(tmp_path / "__MACOSX" / "a-1-R.jpg")
    (tmp_path / "img" / "._a-1-R.jpg").write_bytes(b"junk")
    assert list(index_images(tmp_path)) == ["a-1-R"]


def test_qc_mark_and_semicolon_csv(cfg):
    q = Path(cfg["paths"]["out"]) / "qc" / "qc.csv"
    orig = q.read_bytes()
    try:
        df = pd.read_csv(q, dtype=str).fillna(""); df["decision"] = ""
        df.to_csv(q, index=False, sep=";")                     # what French-locale Excel writes
        g = df[df.stage == "G"].cell_id.iloc[0]
        r = qc.mark(cfg, ["S", "G"], "ok", "T", exclude=[g])
        assert r["decisions"]["exclude"] == 1
        out = qc.read_qc(q)
        assert (out[out.stage.isin(["S", "G"])].decision != "").all()
        with pytest.raises(ValueError):
            qc.mark(cfg, [], "ok", "T", exclude=["no_such_cell"])
    finally:
        q.write_bytes(orig)


def test_nih_manifest_merge_and_splits(tmp_path):
    from malprep import nih
    root = tmp_path / "cell_images"
    rows = {"parasitized": {}, "uninfected": {}}
    ids = [f"C{i}P{i}ThinF" for i in range(1, 25)] + ["C47P8thinOriginalOlympusCX21", "C47P8thin_Original_Motic"] \
        + [f"C{i}ThinF" for i in range(201, 211)]
    for pid in ids:
        for lab, sub in (("parasitized", "Parasitized"), ("uninfected", "Uninfected")):
            if lab == "parasitized" and "P" not in pid[1:].split("Thin")[0].lstrip("0123456789"):
                continue
            (root / sub).mkdir(parents=True, exist_ok=True)
            names = [f"{pid}_IMG_2015091{f}_cell_{c}.png" for f in range(2) for c in range(3)]
            for n in names:
                Image.new("RGB", (8, 8)).save(root / sub / n)
            rows[lab][pid] = names
    for lab in rows:   # spreadsheet-export format with trailing commas
        with open(root / f"map_{lab}.csv", "w") as f:
            for pid, names in rows[lab].items():
                f.write(pid + ',"' + str(names) + '"' + "," * 30 + "\n")
    c = {"paths": {"out": str(tmp_path / "data"), "splits": str(tmp_path / "splits")},
         "nih": {"root": str(root), "mapping_parasitized": str(root / "map_parasitized.csv"),
                 "mapping_uninfected": str(root / "map_uninfected.csv"),
                 "merge_patients": [["C47P8thinOriginalOlympusCX21", "C47P8thin_Original_Motic"]],
                 "split": {"fractions": [0.6, 0.2, 0.2], "seed": 1, "import_csv": None}},
         "_path": "x", "_sha256": "x"}
    rep = nih.build(c)
    assert rep["patient_ids"] == 36 and rep["patient_groups"] == 35
    assert rep["issue_counts"].get("unmapped", 0) == 0
    leak = nih.make_splits(c)
    assert leak["part_patient"]["patients_in_>1_partition"] == 0
    assert leak["part_field"]["fields_in_>1_partition"] == 0
    assert leak["part_cell"]["patients_in_>1_partition"] > 0     # cell-level split leaks patients
    assert nih.verify(c) == []


def test_inner_validation_share_in_range(cfg):
    import json
    for unit in ("slide", "image"):
        for r in cfg["splits"]["repeats"]:
            sp = json.loads((Path(cfg["paths"]["splits"]) / unit / f"outer_r{r}.json").read_text())
            assert all("\\\\" not in sp["manifest"] for _ in [0])
            for f in sp["folds"]:
                assert f["n_cells"]["inner_train"] > 0 and f["inner_val_share"] > 0


def test_paths_are_posix(cfg):
    cr = md(cfg, "crops.csv"); im = md(cfg, "images.csv")
    assert not cr.crop_path.str.contains("\\\\", regex=False).any()
    assert not im.image_path.str.contains("\\\\", regex=False).any()


def test_gate_freeze_and_lock(cfg, capsys):
    import os
    from malprep.__main__ import gate, freeze
    out = Path(cfg["paths"]["out"])
    (out / "manifests" / "bbbc041_exposure.md").write_text("no exposure. Signed / date: T 2026-10-12\n")
    assert gate(cfg) == 0
    assert freeze(cfg) == 0
    with pytest.raises(SystemExit):                      # writers are locked after freeze
        tables.table1(cfg)
    t = out / "tables" / "table1_data.md"; orig = t.read_bytes()
    try:
        t.write_bytes(orig + b"x")                       # tampering is detected
        assert gate(cfg) == 1
    finally:
        t.write_bytes(orig)
    assert gate(cfg) == 0
    (out / "manifests" / "FROZEN.json").unlink()


def test_stale_qc_detected(cfg):
    from malprep.__main__ import gate
    from malprep.qc import read_qc
    qp = Path(cfg["paths"]["out"]) / "qc" / "qc.csv"; orig = qp.read_bytes()
    try:
        q = read_qc(qp); q.loc[q.index[q.stage == "S"][0], "decision"] = "exclude"; q.to_csv(qp, index=False)
        assert gate(cfg) == 1                            # cells_final no longer matches qc.csv
    finally:
        qp.write_bytes(orig)
