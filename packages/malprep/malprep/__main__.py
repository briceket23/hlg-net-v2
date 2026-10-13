"""Command line: python -m malprep <step> --config configs/data.yaml

Steps, in protocol order (Steps 2-5):
  manifest    per-parasite manifest from labels + images            -> data/manifests/cells.csv ...
  dedup       pHash near-duplicates, group_slide / group_image       -> cells.csv, near_duplicates.csv
  crops       fixed 256 px native-resolution crops + masks           -> data/crops/mpidb, crops.csv
  qc-sheets   contact sheets + qc.csv to fill in                     -> data/qc/
  qc-mark     record QC decisions from the command line (avoids Excel ';' CSV problems)
              e.g. qc-mark --stages S,G --decision ok --reviewer CBFT --exclude ID1 --relabel ID2=T
  qc-apply    apply QC decisions                                     -> cells_final.csv
  tables      Table 1 + derived ratios                               -> data/tables/
  rbc-scale   MP-IDB red-cell diameter (magnification ruler)         -> data/tables/mpidb_rbc_scale.json
  bbbc041     external set manifest, crops (+ matched), Table 5a     -> bbbc041_cells.csv
  nih         NIH patient-grouped manifest + mapping audit            -> nih_cells.csv, nih_report.json
  nih-splits  patient / field / cell split protocols (same sizes)    -> splits/nih/
  splits      outer folds + inner validation (--unit slide|image)    -> splits/<unit>/outer_r*.json
  coverage    rewrite the per-fold coverage tables (thesis Table 4.x / paper Table 2 input)
  verify      split leakage/coverage tests + crop reproducibility
  gate        pass/fail list for gates G1 and G2 (after freeze: also detects any changed file)
  freeze      run gate; if it passes, record SHA-256 of every output and lock all writing steps
              freeze --amend --reason "..."  re-records after a deliberate, logged change
  all         manifest..tables, then splits for both units (QC must already be complete)
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

from .io_utils import load_config, out_dir

PLACEHOLDER = re.compile(r"to fill|fill in|<[^>]*>|tbd|todo", re.I)


def _p(obj):
    print(json.dumps(obj, indent=1, default=str))


def _tracked_files(cfg: dict) -> list[Path]:
    out = Path(cfg["paths"]["out"])
    files = [p for d in ("manifests", "tables") for p in sorted((out / d).glob("*")) if p.is_file() and p.name != "FROZEN.json"]
    files += [out / "qc" / "qc.csv"]
    files += sorted(p for p in Path(cfg["paths"]["splits"]).rglob("*") if p.is_file())
    files += [Path(cfg["_path"])]
    return [p for p in files if p.exists()]


def gate(cfg: dict) -> int:
    """REQUIRED checks fail the gate; ADVISORY checks print WARN and do not."""
    from .splits import verify
    from .crops import verify_reproducible
    from .qc import read_qc
    from .io_utils import sha256_file, frozen_path
    import pandas as pd
    md = out_dir(cfg, "manifests"); checks = []

    def chk(name, ok, detail="", required=True):
        checks.append((name, bool(ok), detail, required))

    # --- frozen state: nothing may have changed since `freeze`
    fz = frozen_path(cfg)
    if fz.exists():
        rec = json.loads(fz.read_text())
        changed = [k for k, v in rec["files"].items() if not Path(k).exists() or sha256_file(k) != v]
        chk("frozen files unchanged since freeze", not changed, changed[:5])
        if (md / "crops.csv").exists():
            cr = pd.read_csv(md / "crops.csv")
            bad = [r.cell_id for r in cr.itertuples() if not Path(r.crop_path).exists() or sha256_file(r.crop_path) != r.crop_sha256]
            chk("every crop file matches its recorded SHA-256", not bad, bad[:5])
    # --- declarations (G1: label provenance and BBBC041 exposure statement recorded)
    ls = str(cfg.get("label_source", ""))
    chk("label_source recorded (no placeholder left)", ls and not PLACEHOLDER.search(ls), ls)
    exp = md / "bbbc041_exposure.md"
    chk("BBBC041 exposure statement filled and signed", exp.exists() and "Signed / date:\n" not in exp.read_text()
        and "yes / no" not in exp.read_text(), "answer yes/no and sign")
    # --- artefacts exist and are mutually consistent (no stale step)
    for f in ["cells.csv", "images.csv", "crops.csv", "cells_final.csv", "digests.json", "bbbc041_cells.csv"]:
        chk(f"{f} exists", (md / f).exists())
    if all((md / f).exists() for f in ["cells.csv", "crops.csv", "cells_final.csv", "images.csv"]):
        cells = pd.read_csv(md / "cells.csv"); crops = pd.read_csv(md / "crops.csv")
        fin = pd.read_csv(md / "cells_final.csv"); im = pd.read_csv(md / "images.csv")
        chk("crops.csv matches cells.csv (no stale crops)", set(cells.cell_id) == set(crops.cell_id))
        chk("all images have the expected size", im.size_ok.all(), im.loc[~im.size_ok, "image_id"].tolist()[:5])
        chk("filename codes match labelled stages", im.codes_match.all(), im.loc[~im.codes_match, "image_id"].tolist()[:5])
        chk("stored paths use forward slashes", not crops.crop_path.str.contains("\\", regex=False).any()
            and not im.image_path.fillna("").astype(str).str.contains("\\", regex=False).any(), "rerun manifest/crops with this version")
        qcp = out_dir(cfg, "qc") / "qc.csv"
        if qcp.exists():
            q = read_qc(qcp)
            miss = q[q.stage.isin(["S", "G"]) & (q.decision == "")]
            chk("every schizont/gametocyte crop QC-reviewed", len(miss) == 0, f"{len(miss)} missing")
            ex = set(q.cell_id[q.decision == "exclude"]); rel = q[q.decision.str.startswith("relabel:")]
            fs = fin.set_index("cell_id").stage
            stale = bool(ex & set(fin.cell_id)) or any(fs.get(r.cell_id) != r.decision.split(":")[1] for r in rel.itertuples())
            chk("cells_final.csv reflects the current qc.csv (rerun qc-apply after any QC edit)", not stale)
        else:
            chk("qc.csv exists", False)
        rep = verify_reproducible(cfg)
        chk("crops byte-identical on re-extraction (25-cell sample)", not rep["mismatches"], rep)
        t1 = out_dir(cfg, "tables") / "table1_data.csv"
        if t1.exists():
            t = pd.read_csv(t1)
            chk("Table 1 total equals cells_final rows", int(t.loc[t.species == "total", "cells"].iloc[0]) == len(fin))
        else:
            chk("Table 1 written", False)
        for st in cfg["splits"]["rare_stages"]:      # feasibility stop rule (Step 3): rare stage in >= K groups
            unit = "group_" + cfg["mpidb"]["group_unit"]
            col = unit if unit in fin else "slide"
            n = fin[fin.stage == st][col].nunique()
            chk(f"rare stage {st} occurs in >= K={cfg['splits']['K']} groups (feasibility)", n >= cfg["splits"]["K"], f"{n} groups")
    for unit in ("slide", "image"):
        sd = Path(cfg["paths"]["splits"]) / unit
        if sd.exists():
            f = verify(cfg, unit)
            chk(f"splits[{unit}] pass leakage and coverage tests", not f, f[:5])
        else:
            chk(f"splits[{unit}] generated", False)
    # --- advisory
    chk("BBBC041 magnification-matched crops written (sensitivity analysis)",
        "crop_path_matched" in (pd.read_csv(md / "bbbc041_cells.csv", nrows=1).columns if (md / "bbbc041_cells.csv").exists() else []),
        "set bbbc041.mpidb_rbc_diameter_px and rerun bbbc041", required=False)
    if (Path(cfg["paths"]["splits"]) / "nih" / "nih_splits.csv").exists():
        from . import nih
        f = nih.verify(cfg)
        chk("NIH split files pass patient/field disjointness", not f, f)
        chk("NIH patient split imported from the transfer study (thesis comparability)",
            bool(cfg.get("nih", {}).get("split", {}).get("import_csv")), "set nih.split.import_csv", required=False)
    width = max(len(c[0]) for c in checks)
    for name, ok, detail, req in checks:
        tag = "PASS" if ok else ("FAIL" if req else "WARN")
        print(f"{tag}  {name.ljust(width)}  {detail if not ok and detail else ''}")
    return 0 if all(c[1] for c in checks if c[3]) else 1


def freeze(cfg: dict, amend: bool = False, reason: str = "") -> int:
    """Record the SHA-256 of every output and lock all writers. `--amend` re-records after a deliberate
    change (e.g. filling in label_source); the reason is stored in FROZEN.json and must also go into
    logs/deviations.md."""
    from .io_utils import sha256_file, frozen_path, write_json, provenance
    fp = frozen_path(cfg)
    if fp.exists() and not amend:
        print(f"already frozen: {fp} (use `freeze --amend --reason ...` for a logged re-freeze)"); return 1
    if amend and not reason:
        print("--amend needs --reason"); return 1
    import os
    os.environ["MALPREP_UNFREEZE"] = "1" if amend else os.environ.get("MALPREP_UNFREEZE", "")
    if gate(cfg) != 0:
        print("\nnot frozen: fix the FAIL lines first"); return 1
    files = {p.as_posix(): sha256_file(p) for p in _tracked_files(cfg)}
    rec = {"files": files, "provenance": provenance(cfg), "amendments": []}
    if amend:
        old = json.loads(fp.read_text())
        rec["amendments"] = old.get("amendments", []) + [
            {"when": rec["provenance"]["created"], "reason": reason,
             "changed": sorted(k for k, v in files.items() if old["files"].get(k) != v)}]
        rec["first_frozen"] = old.get("first_frozen", old["provenance"]["created"])
    else:
        rec["first_frozen"] = rec["provenance"]["created"]
    write_json(rec, fp)
    print(f"\nfrozen {len(files)} files -> {fp.as_posix()}; commit and tag (e.g. data-frozen-v1)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="malprep", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step")
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--no-images", action="store_true", help="labels-only dry run (no pHash, no crops)")
    ap.add_argument("--unit", choices=["slide", "image"], help="grouping unit for `splits` (default: config)")
    ap.add_argument("--manifest", default="cells_final.csv")
    ap.add_argument("--allow-incomplete-qc", action="store_true", help="development only; never for the frozen run")
    ap.add_argument("--stages", default="", help="qc-mark: comma list of stages, e.g. S,G")
    ap.add_argument("--decision", default="ok", help="qc-mark: ok | exclude | relabel:X")
    ap.add_argument("--reviewer", default="", help="qc-mark: reviewer initials")
    ap.add_argument("--exclude", nargs="*", default=[], help="qc-mark: cell_ids to exclude")
    ap.add_argument("--relabel", nargs="*", default=[], help="qc-mark: cell_id=STAGE items")
    ap.add_argument("--amend", action="store_true", help="freeze: re-record after a deliberate, logged change")
    ap.add_argument("--reason", default="", help="freeze --amend: why (also log it in logs/deviations.md)")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    use_img = not a.no_images
    from . import mpidb, dedup, crops, qc, tables, bbbc041, splits
    s = a.step
    if s == "manifest": _p(mpidb.write_manifest(cfg, use_img))
    elif s == "dedup": _p(dedup.assign_groups(cfg, use_img))
    elif s == "crops": _p(crops.run_mpidb(cfg))
    elif s == "qc-sheets": _p(qc.make_sheets(cfg))
    elif s == "qc-mark":
        if not a.reviewer: ap.error("--reviewer is required")
        _p(qc.mark(cfg, [x for x in a.stages.split(",") if x], a.decision, a.reviewer, a.exclude, a.relabel))
    elif s == "qc-apply": _p(qc.apply_qc(cfg, require_complete=not a.allow_incomplete_qc))
    elif s == "tables": _p(tables.table1(cfg, a.manifest))
    elif s == "rbc-scale":
        from . import scale; _p(scale.mpidb_rbc_scale(cfg))
    elif s == "bbbc041": _p(bbbc041.run(cfg, make_crops=use_img))
    elif s == "nih":
        from . import nih; _p(nih.build(cfg))
    elif s == "nih-splits":
        from . import nih; _p(nih.make_splits(cfg)); _p({"verify": nih.verify(cfg) or "PASS"})
    elif s == "splits": _p(splits.make_splits(cfg, a.manifest, a.unit))
    elif s == "coverage":
        for unit in ("slide", "image"):
            if (Path(cfg["paths"]["splits"]) / unit).exists():
                _p({unit: splits.coverage_summary(splits.coverage(cfg, unit))})
    elif s == "verify":
        for unit in ("slide", "image"):
            if (Path(cfg["paths"]["splits"]) / unit).exists():
                _p({unit: splits.verify(cfg, unit) or "PASS"})
        if use_img: _p(crops.verify_reproducible(cfg))
    elif s == "gate": return gate(cfg)
    elif s == "freeze": return freeze(cfg, a.amend, a.reason)
    elif s == "all":
        _p(mpidb.write_manifest(cfg, use_img)); _p(dedup.assign_groups(cfg, use_img))
        if use_img:
            _p(crops.run_mpidb(cfg)); _p(qc.make_sheets(cfg))
        _p(qc.apply_qc(cfg, require_complete=not a.allow_incomplete_qc) if use_img else "qc skipped (no images)")
        man = a.manifest if use_img else "cells.csv"
        _p(tables.table1(cfg, man))
        for unit in ("slide", "image"):
            _p(splits.make_splits(cfg, man, unit))
    else:
        ap.error(f"unknown step {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
