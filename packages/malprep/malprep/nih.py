"""NIH cell set (raw_data/cell_images): patient-grouped manifest and split files.

Not used by the stage-only HLG-Net study. It serves the transfer study and the thesis
split-protocol experiment (patient- vs field- vs cell-level splitting).

Checked on the mapping files (12 Oct 2026):
  - 27,558 cells (13,779 parasitized + 13,779 uninfected), 201 patient IDs, 1,408 fields.
  - 151 IDs carry a patient number (C<n>P<k>...); these are the patients with parasitized cells.
    The 50 IDs without one (C<n>ThinF, ...) contribute uninfected cells only.
  - C47P8thinOriginalOlympusCX21 and C47P8thin_Original_Motic are the SAME patient (P8) imaged on
    two microscopes -> merged into one group (200 patient groups), else the patient leaks.
  - The mapping CSVs are spreadsheet exports (trailing commas); file names are parsed by regex.
"""
from __future__ import annotations
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

from .io_utils import out_dir, write_csv, write_json, append_digest, sha256_file, provenance, posix, assert_not_frozen, write_text

PNG = re.compile(r"'([^']+\.png)'")


def parse_mapping(path: str | Path) -> dict[str, list[str]]:
    m: dict[str, list[str]] = {}
    for line in open(path, encoding="utf-8-sig"):
        if not line.strip():
            continue
        pid = line.split(",", 1)[0].strip()
        m.setdefault(pid, []).extend(PNG.findall(line))
    return m


def field_of(fname: str) -> str:
    return fname.rsplit("_cell_", 1)[0] if "_cell_" in fname else Path(fname).stem


def build(cfg: dict) -> dict:
    assert_not_frozen(cfg, "nih")
    N = cfg["nih"]; root = Path(N["root"])
    maps = {"parasitized": parse_mapping(N["mapping_parasitized"]),
            "uninfected": parse_mapping(N["mapping_uninfected"])}
    owner: dict[tuple, set] = defaultdict(set)
    for lab, m in maps.items():
        for pid, files in m.items():
            for f in files:
                owner[(lab, f)].add(pid)
    # patient groups: union of IDs listed together in merge_patients
    group = {pid: pid for m in maps.values() for pid in m}
    for ids in N.get("merge_patients", []):
        rep = sorted(ids)[0]
        for i in ids:
            if i in group:
                group[i] = rep
    has_p = {pid: bool(re.match(r"C\d+[A-Z]?P\d+", pid)) for pid in group}
    infected_patients = {group[p] for p in maps["parasitized"]}
    rows, issues = [], defaultdict(list)
    for lab, sub in (("parasitized", "Parasitized"), ("uninfected", "Uninfected")):
        d = root / sub
        files = sorted(p for p in d.glob("*.png")) if d.exists() else []
        if not d.exists():
            issues["missing_folder"].append(str(d))
        on_disk = {p.name for p in files}
        for p in files:
            pids = owner.get((lab, p.name), set())
            if len(pids) != 1:
                issues["unmapped" if not pids else "multi_mapped"].append(p.name); continue
            pid = next(iter(pids))
            if not p.name.startswith(pid + "_"):
                issues["prefix_mismatch"].append(p.name)
            g = group[pid]
            rows.append({"cell_id": p.stem, "path": posix(p), "label": int(lab == "parasitized"), "class": lab,
                         "patient_id": pid, "patient_group": g, "field_id": field_of(p.name),
                         "patient_type": "infected" if g in infected_patients else "uninfected_only"})
        listed = {f for (l, f) in owner if l == lab}
        issues[f"listed_not_on_disk_{lab}"] = sorted(listed - on_disk)[:50]
        issues[f"n_listed_not_on_disk_{lab}"] = [len(listed - on_disk)]
    both = {f for (l, f) in owner if l == "parasitized"} & {f for (l, f) in owner if l == "uninfected"}
    df = pd.DataFrame(rows)
    if len(df):
        fp = df.groupby("field_id").patient_group.nunique()
        issues["fields_spanning_patients"] = sorted(fp.index[fp > 1])
    md = out_dir(cfg, "manifests")
    sha = write_csv(df, md / "nih_cells.csv"); append_digest(cfg, "nih_cells.csv", sha)
    rep = {"cells": int(len(df)), "parasitized": int(df.label.sum()) if len(df) else 0,
           "patient_ids": len(group), "patient_groups": int(df.patient_group.nunique()) if len(df) else 0,
           "fields": int(df.field_id.nunique()) if len(df) else 0,
           "patient_types": df.drop_duplicates("patient_group").patient_type.value_counts().to_dict() if len(df) else {},
           "ids_with_patient_number": int(sum(has_p.values())),
           "merged": [ids for ids in N.get("merge_patients", [])],
           "files_in_both_classes": len(both),
           "issue_counts": {k: (v[0] if k.startswith("n_") else len(v)) for k, v in issues.items()
                            if not k.startswith("listed_not_on_disk")},
           "issue_examples": {k: v[:10] for k, v in issues.items() if v and not k.startswith("n_")},
           "cells_per_patient": df.groupby("patient_group").size().describe().round(1).to_dict() if len(df) else {}}
    write_json(rep, out_dir(cfg, "tables") / "nih_report.json")
    return rep


def _allocate(units: list, sizes: list[int], rng) -> dict:
    units = list(units); rng.shuffle(units)
    out, i = {}, 0
    for part, n in zip(("train", "val", "test"), sizes):
        for u in units[i:i + n]:
            out[u] = part
        i += n
    return out


def _sizes(n: int, frac: list[float]) -> list[int]:
    a = [int(round(n * f)) for f in frac[:2]]
    return a + [n - sum(a)]


def make_splits(cfg: dict) -> dict:
    """Three nested split protocols with the SAME partition sizes (in cells, approximately):
    patient (primary, leakage-free), field (image-level in the field sense), cell (fully random).
    If nih.split.import_csv is set (columns patient_group|patient_id, partition), the patient split is
    imported from the transfer study instead of generated, so the thesis experiments stay comparable."""
    assert_not_frozen(cfg, "nih-splits")
    N = cfg["nih"]; S = N["split"]; md = out_dir(cfg, "manifests")
    mpath = md / "nih_cells.csv"
    cells = pd.read_csv(mpath)
    rng = np.random.default_rng(S["seed"])
    imp = S.get("import_csv")
    if imp:
        t = pd.read_csv(imp)
        key = "patient_group" if "patient_group" in t else "patient_id"
        mp = dict(zip(t[key].astype(str), t.partition))
        if key == "patient_id":
            g = cells.drop_duplicates("patient_id").set_index("patient_id").patient_group
            mp = {g[k]: v for k, v in mp.items() if k in g}
        missing = set(cells.patient_group) - set(mp)
        if missing:
            raise ValueError(f"imported split lacks {len(missing)} patient groups, e.g. {sorted(missing)[:5]}")
        pat = mp
    else:
        pat = {}
        for _, g in cells.drop_duplicates("patient_group").groupby("patient_type"):   # stratify by type
            units = sorted(g.patient_group)
            pat.update(_allocate(units, _sizes(len(units), S["fractions"]), rng))
    cells["part_patient"] = cells.patient_group.map(pat)
    # field- and cell-level splits matched to the patient split's cell counts per partition and class
    target = cells.groupby(["part_patient", "label"]).size()
    fields = sorted(cells.field_id.unique()); rng.shuffle(fields)
    fsize = cells.groupby("field_id").size()
    fpart, acc = {}, {p: 0 for p in ("train", "val", "test")}
    ftot = {p: int(target.loc[p].sum()) for p in ("train", "val", "test")}
    for f in fields:                                     # greedy fill to the patient split's sizes
        p = max(acc, key=lambda q: (ftot[q] - acc[q]) / max(1, ftot[q]))
        fpart[f] = p; acc[p] += int(fsize[f])
    cells["part_field"] = cells.field_id.map(fpart)
    cells["part_cell"] = ""
    for lab, g in cells.groupby("label"):
        idx = g.index.to_numpy().copy(); rng.shuffle(idx); i = 0
        for p in ("train", "val", "test"):
            n = int(target.get((p, lab), 0)); cells.loc[idx[i:i + n], "part_cell"] = p; i += n
    sdir = Path(cfg["paths"]["splits"]) / "nih"; sdir.mkdir(parents=True, exist_ok=True)
    out = cells[["cell_id", "label", "patient_group", "field_id", "part_patient", "part_field", "part_cell"]]
    write_csv(out, sdir / "nih_splits.csv")
    leak = {}
    for proto in ("part_patient", "part_field", "part_cell"):
        pp = out.groupby("patient_group")[proto].nunique()
        ff = out.groupby("field_id")[proto].nunique()
        leak[proto] = {"patients_in_>1_partition": int((pp > 1).sum()), "fields_in_>1_partition": int((ff > 1).sum()),
                       "cells": out.groupby(proto).size().to_dict(),
                       "patients_per_partition": out.drop_duplicates(["patient_group", proto]).groupby(proto).size().to_dict()}
    meta = {"seed": S["seed"], "fractions": S["fractions"], "imported_from": imp or None,
            "manifest_sha256": sha256_file(mpath), "leakage": leak, "provenance": provenance(cfg)}
    write_json(meta, sdir / "nih_splits_meta.json")
    write_text(f"{sha256_file(sdir / 'nih_splits.csv')}  nih_splits.csv\n"
               f"{sha256_file(mpath)}  nih_cells.csv\n", sdir / "digests.txt")
    return leak


def verify(cfg: dict) -> list[str]:
    s = pd.read_csv(Path(cfg["paths"]["splits"]) / "nih" / "nih_splits.csv")
    fails = []
    if (s.groupby("patient_group").part_patient.nunique() > 1).any():
        fails.append("patient split: a patient group is in more than one partition")
    if (s.groupby("field_id").part_field.nunique() > 1).any():
        fails.append("field split: a field is in more than one partition")
    fp = s.groupby("field_id").patient_group.nunique()
    if (fp > 1).any():
        fails.append("a field belongs to more than one patient group")
    for c in ("part_patient", "part_field", "part_cell"):
        if s[c].isna().any() or (s[c] == "").any():
            fails.append(f"{c}: unassigned cells")
    return fails
