"""A synthetic run tree with the real layout (meta.json, selection.csv, three prediction files per run) for
testing the analysis end to end. 12 arms x 3 repeats x 5 folds = 180 runs by default. Never for results."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

from malcore import STAGES
from malcore.io import write_csv, write_json
from malcore.hashing import sha256_file

ARMS12 = ["B0", "B1", "B2", "B3", "B4", "B5", "B6", "M-aux", "M-full", "ABL-noaux", "ABL-detach", "ABL-coordattn"]


def _probs(rng, y, acc_rare, acc_common, n):
    """Calibrated-ish probabilities whose argmax hits the truth with stage-dependent accuracy."""
    p = rng.dirichlet(np.ones(4) * 0.7, n)
    hit = rng.random(n) < np.where(y >= 2, acc_rare, acc_common)
    for i in range(n):
        if hit[i]:
            p[i, y[i]] += 1.5
        else:
            p[i, rng.choice([c for c in range(4) if c != y[i]])] += 1.0
    return p / p.sum(1, keepdims=True)


def make_synthetic_runs(root: str | Path, repeats=(1, 2, 3), folds=(0, 1, 2, 3, 4), arms=ARMS12, n_cells=1500,
                        n_groups=17, n_external=400, effect=0.04, noise=0.01, seed=0, backbone_tag="tiny", unit="slide") -> dict:
    root = Path(root); rng = np.random.default_rng(seed)
    # manifest + splits (groups -> folds, different per repeat)
    y = rng.choice(4, n_cells, p=[0.80, 0.10, 0.05, 0.05])
    groups = rng.integers(0, n_groups, n_cells)
    man = pd.DataFrame({"cell_id": [f"c{i:04d}" for i in range(n_cells)], "stage": [STAGES[i] for i in y], "stage_idx": y,
                        "species": rng.choice(["falciparum", "vivax"], n_cells, p=[0.8, 0.2]), "group_slide": groups.astype(str)})
    man_path = root / "data" / "manifests" / "cells_final.csv"; write_csv(man, man_path)
    man_sha = sha256_file(man_path)
    split_sha = {}
    for r in repeats:
        g2f = rng.permutation(n_groups) % len(folds)
        obj = {"repeat": r, "group_column": "group_slide", "manifest": str(man_path), "manifest_sha256": man_sha,
               "folds": [{"k": k, "test": [str(g) for g in range(n_groups) if g2f[g] == k]} for k in folds]}
        p = root / "splits" / unit / f"outer_r{r}.json"; write_json(obj, p); split_sha[r] = sha256_file(p)
    ext_y = rng.choice(4, n_external, p=[0.15, 0.65, 0.1, 0.1]); ext_img = rng.integers(0, 60, n_external)
    ext_sub = np.where(ext_img < 50, "train", "test")
    base = {a: 0.55 for a in arms}; base.update({"B0": 0.45, "M-full": 0.55 + effect, "M-aux": 0.56, "ABL-noaux": 0.55, "ABL-detach": 0.54})
    sel_rows = []
    for r in repeats:
        sp = json.loads((root / "splits" / unit / f"outer_r{r}.json").read_text())
        for k in folds:
            test_groups = set(sp["folds"][k]["test"])
            m = man[man.group_slide.isin(test_groups)]
            bstar = ["B1", "B2", "B3", "B4"][(r + k) % 4]
            for arm in arms:
                acc_r = base[arm] + rng.normal(0, noise); acc_c = 0.9
                rd = root / "runs" / unit / arm / backbone_tag / f"r{r}_f{k}" / "c0"; rd.mkdir(parents=True, exist_ok=True)
                for kind, yy, ids, grp, sub in (("test", m.stage_idx.to_numpy(), m.cell_id, m.group_slide, None),
                                                ("innerval", y[:80], man.cell_id[:80], groups[:80].astype(str), None),
                                                ("bbbc041", ext_y, [f"b{i}" for i in range(n_external)], ext_img.astype(str), ext_sub)):
                    pr = _probs(rng, yy, acc_r, acc_c, len(yy))
                    df = pd.DataFrame({"cell_id": list(ids), "y_true": yy, "y_pred": pr.argmax(1), "T": 1.2, "gate": np.nan,
                                       "image_group": list(grp), "species": "vivax" if sub is not None else "falciparum",
                                       "arm": arm, "backbone": backbone_tag, "unit": unit, "repeat": r, "fold": k, "seed": 1000 * r + k,
                                       "config_sha": f"cfg-{arm}", "split_sha": split_sha[r], "manifest_sha": man_sha, "commit": "synthetic"})
                    for c in range(4):
                        df[f"logit_{c}"] = np.log(pr[:, c]); df[f"prob_{c}"] = pr[:, c]
                    if sub is not None:
                        df["subset"] = sub
                    if arm == "B5":
                        df["species_pred"] = 0
                    df.to_parquet(rd / f"preds_{kind}.parquet", index=False)
                meta = {"arm": arm, "backbone": backbone_tag, "unit": unit, "repeat": r, "fold": k, "seed": 1000 * r + k,
                        "config": {"lambda_gate": 0.15 if "M-" in arm or "ABL" in arm else None, "lr": 1e-3},
                        "config_sha": f"cfg-{arm}", "best_epoch": 20, "hit_cap": False, "innerval_macro_f1_at_best": 0.6,
                        "temperature": 1.2, "manifest_sha256": man_sha, "split_sha256": split_sha[r], "commit": "synthetic",
                        "elapsed_s": 100.0, "diverged": False, "run_id": f"{arm}/{backbone_tag}/r{r}_f{k}/c0"}
                (rd / "meta.json").write_text(json.dumps(meta, indent=1))
                sel_rows.append({"unit": unit, "backbone": backbone_tag, "repeat": r, "fold": k, "arm": arm, "cfg_idx": 0,
                                 "selected": True, "inner_val_macro_f1": 0.6, "bstar": bstar, "tier": 1})
                # a non-selected candidate without test predictions
                (rd.parent / "c1").mkdir(exist_ok=True); (rd.parent / "c1" / "fit.json").write_text("{}")
    write_csv(pd.DataFrame(sel_rows), root / "runs" / unit / "selection.csv")
    return {"runs_root": str(root / "runs" / unit), "manifest": str(man_path), "splits_dir": str(root / "splits"),
            "n_runs": len(arms) * len(repeats) * len(folds)}
