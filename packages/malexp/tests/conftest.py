"""Shared toy dataset (synthetic MP-IDB-like crops, frozen splits) for the malexp tests."""
from pathlib import Path
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def toy(tmp_path_factory):
    from malprep.synthetic import make_toy
    from malprep.io_utils import load_config
    from malprep import mpidb, dedup, crops, qc, tables, bbbc041, splits
    root = tmp_path_factory.mktemp("toy"); make_toy(root)
    cfg = load_config(root / "data_cfg.yaml")
    mpidb.write_manifest(cfg); dedup.assign_groups(cfg); crops.run_mpidb(cfg); qc.make_sheets(cfg)
    q = pd.read_csv(Path(cfg["paths"]["out"]) / "qc" / "qc.csv", dtype=str).fillna(""); q["decision"] = "ok"
    q.to_csv(Path(cfg["paths"]["out"]) / "qc" / "qc.csv", index=False)
    qc.apply_qc(cfg); tables.table1(cfg); bbbc041.run(cfg); splits.make_splits(cfg, unit="slide")
    return cfg
