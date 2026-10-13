"""Analysis configuration (configs/analysis.yaml), frozen with the analysis plan before Step 10."""
from __future__ import annotations
from pathlib import Path
from malcore.io import load_yaml
from malcore.hashing import sha256_file

DEFAULTS = {
    "runs_root": "runs/slide",
    "unit": "slide",
    "backbone": "mobilevit_xxs",
    "repeats": [1, 2, 3],
    "folds": [0, 1, 2, 3, 4],
    "manifest": "data/manifests/cells_final.csv",
    "splits_dir": "splits",
    "coverage_table": "data/tables/fold_coverage_slide.csv",
    "complexity_table": "analysis/complexity.csv",
    "out": "analysis",
    "arms": {"treatment": "M-full", "control": "B*", "mechanism": "M-aux",
             "ladder": ["B0", "B1", "B2", "B3", "B4", "B5", "B6"],
             "ablations": ["ABL-noaux", "ABL-detach", "ABL-coordattn"],
             "optional": ["M-prob"]},
    "stats": {"rho": 0.25, "alpha": 0.05, "delta_min": 0.02, "sigma_d_assumed": [0.02, 0.03],
              "n_boot": 2000, "boot_seed": 2026, "power": 0.8},
    "external": {"low_support_n": 30, "subsets": ["train", "test"]},
    "sensitivity": {"replication_runs_root": None, "image_unit_runs_root": None},
}


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in (b or {}).items():
        out[k] = _merge(a[k], v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
    return out


def load_analysis_config(path: str | Path | None) -> dict:
    cfg = _merge(DEFAULTS, load_yaml(path) if path and Path(path).exists() else {})
    cfg["_path"] = str(path) if path else None
    cfg["_sha256"] = sha256_file(path) if path and Path(path).exists() else None
    return cfg
