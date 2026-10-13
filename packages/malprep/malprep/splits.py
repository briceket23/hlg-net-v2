"""Step 5: grouped, rare-bucket stratified outer folds with nested inner validation.

assign_groups(): groups containing a gametocyte are placed first, then groups with a schizont,
then the rest. Within a bucket, groups go in order of decreasing count of the bucket's stage
(randomly perturbed by the seed) to the fold with the fewest cells of that stage, ties broken by
fewest total cells, then fewest cells of the group's species, then at random.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

from malcore import STAGES, SPECIES
from .io_utils import out_dir, write_json, write_csv, sha256_file, provenance


def _need(path):
    from pathlib import Path as _P
    if not _P(path).exists():
        raise SystemExit(f"{path} not found: complete visual QC first (qc-mark, then qc-apply). "
                         "For a development preview only, pass --manifest cells.csv.")


def group_table(cells: pd.DataFrame, unit_col: str) -> pd.DataFrame:
    g = pd.crosstab(cells[unit_col], cells.stage).reindex(columns=STAGES, fill_value=0)
    g["cells"] = g[STAGES].sum(axis=1)
    sp = cells.groupby(unit_col).species.agg(lambda s: s.mode().iloc[0])
    g["species"] = sp
    return g.sort_index()


def assign_groups(g: pd.DataFrame, K: int, seed: int, rare=("G", "S"), jitter: float = 0.2) -> pd.Series:
    """One greedy rare-bucket assignment (numpy implementation)."""
    rng = np.random.default_rng(seed)
    idx = list(g.index)
    G, Sx, C = g["G"].to_numpy(int), g["S"].to_numpy(int), g["cells"].to_numpy(int)
    sp = g["species"].to_numpy()
    sp_code = {s: i for i, s in enumerate(sorted(set(sp)))}
    spi = np.array([sp_code[x] for x in sp])
    load = np.zeros((K, 3 + len(sp_code)))          # columns: G, S, cells, species...
    col = {"G": 0, "S": 1, "cells": 2}
    stage_arr = {"G": G, "S": Sx, "cells": C}
    placed = np.full(len(idx), -1)
    remaining = np.ones(len(idx), bool)
    buckets = []
    for st in rare:
        b = remaining & (stage_arr[st] > 0); buckets.append((st, np.flatnonzero(b))); remaining &= ~b
    buckets.append(("cells", np.flatnonzero(remaining)))
    for key, members in buckets:
        if len(members) == 0:
            continue
        w = stage_arr[key][members] * rng.uniform(1 - jitter, 1 + jitter, len(members))
        for i in members[np.argsort(-w, kind="stable")]:
            noise = rng.random(K)
            order = np.lexsort((noise, load[:, 3 + spi[i]], load[:, 2], load[:, col[key]]))
            k = order[0]; placed[i] = k
            load[k, 0] += G[i]; load[k, 1] += Sx[i]; load[k, 2] += C[i]; load[k, 3 + spi[i]] += C[i]
    return pd.Series(placed, index=idx)


def _signature(a: pd.Series) -> frozenset:
    return frozenset(frozenset(map(str, a.index[a == k])) for k in sorted(a.unique()))


def _cv(x) -> float:
    x = np.asarray(x, float)
    return float(x.std() / x.mean()) if x.mean() > 0 else 0.0


def outer_score(g: pd.DataFrame, a: pd.Series, K: int, rare) -> tuple[bool, float]:
    per = g.groupby(a.values)[["cells", *rare]].sum().reindex(range(K), fill_value=0)
    feasible = all((per[st] > 0).all() for st in rare if (g[st] > 0).sum() >= K)
    return feasible, _cv(per["cells"]) + 0.5 * sum(_cv(per[st]) for st in rare)


def inner_score(train: pd.DataFrame, iv: list, target: float, rare) -> tuple[bool, float]:
    it = train.index.difference(iv)
    share = train.loc[iv, "cells"].sum() / train["cells"].sum()
    ok = (all(train.loc[iv, st].sum() > 0 for st in rare)
          and all(train.loc[it, st].sum() > 0 for st in STAGES)
          and 0.5 * target <= share <= 1.6 * target)
    dev = abs(share - target) + 0.5 * sum(abs(train.loc[iv, st].sum() / max(1, train[st].sum()) - target) for st in rare)
    return ok, dev


def search_outer(g, K, seed, rare, n_cand, exclude, used_folds=frozenset(), overlap_weight: float = 0.25):
    """Best candidate by (infeasible, score + overlap_weight * test folds identical to earlier repeats).
    The overlap term keeps repeats from collapsing onto the same partition when groups are few."""
    best = None
    for c in range(n_cand):
        a = assign_groups(g, K, seed * 100003 + c, rare, jitter=0.6)
        sig = _signature(a)
        if sig in exclude:
            continue
        feas, sc = outer_score(g, a, K, rare)
        n_same = sum(1 for f in sig if f in used_folds)
        cand = (not feas, sc + overlap_weight * n_same, c, a, sig, n_same, sc)
        if best is None or cand[:3] < best[:3]:
            best = cand
    return best


def search_inner(train, K_in, seed, rare, n_cand):
    target = 1.0 / K_in
    best = None
    for c in range(n_cand):
        a = assign_groups(train, K_in, seed * 100003 + c, rare, jitter=0.6)
        for f in range(K_in):                    # any fold of the inner assignment may serve
            iv = sorted(a.index[a == f])
            ok, dev = inner_score(train, iv, target, rare)
            cand = (not ok, dev, c, f, iv)
            if best is None or cand[:4] < best[:4]:
                best = cand
    return best


def make_splits(cfg: dict, manifest: str = "cells_final.csv", unit: str | None = None) -> dict:
    """Outer folds: best of `n_candidates` randomised rare-bucket assignments (every test fold holds every
    rare stage; minimal spread of fold sizes and rare counts), never repeating an earlier repeat's partition.
    Inner validation: best candidate with both rare stages, all four stages left for inner training and a
    cell share close to 1/inner_K (accepted range 0.5-1.6 x target)."""
    from .io_utils import assert_not_frozen
    assert_not_frozen(cfg, "splits")
    S = cfg["splits"]
    unit = unit or cfg["mpidb"]["group_unit"]
    md = out_dir(cfg, "manifests")
    mpath = md / manifest
    _need(mpath)
    cells = pd.read_csv(mpath, dtype={"slide": str, "group_slide": str, "group_image": str})
    unit_col = f"group_{unit}" if f"group_{unit}" in cells else ("slide" if unit == "slide" else "image_id")
    g = group_table(cells, unit_col)
    rare = tuple(S["rare_stages"])
    n_out, n_in = S.get("n_candidates", 300), S.get("n_inner_candidates", 100)
    sdir = Path(cfg["paths"]["splits"]) / unit; sdir.mkdir(parents=True, exist_ok=True)
    digests, used, used_folds, warnings, overlap = {}, set(), set(), [], {}
    for r in S["repeats"]:
        seed = S["split_seed_base"] + r
        best = search_outer(g, S["K"], seed, rare, n_out, used, frozenset(used_folds),
                            S.get("repeat_overlap_weight", 0.25))
        if best is None:
            raise RuntimeError("no outer candidate left that differs from earlier repeats")
        infeasible, _, ocand, fold, sig, n_same, oscore = best
        used.add(sig); used_folds |= set(sig); overlap[f"r{r}"] = n_same
        if infeasible:
            warnings.append(f"r{r}: no candidate places every rare stage in every test fold")
        folds = []
        for k in range(S["K"]):
            test = sorted(fold.index[fold == k]); train = g.drop(index=test)
            ib = search_inner(train, S["inner_K"], 10 * seed + k, rare, n_in)
            bad, dev, icand, ifold, iv = ib
            if bad:
                warnings.append(f"r{r} k{k}: inner validation misses a constraint (best deviation {dev:.2f})")
            it = sorted(set(train.index) - set(iv))
            folds.append({"k": k, "test": [str(x) for x in test], "inner_val": [str(x) for x in iv],
                          "inner_train": [str(x) for x in it],
                          "inner_seed": 10 * seed + k, "inner_candidate": int(icand), "inner_fold_used": int(ifold),
                          "inner_val_share": round(float(train.loc[iv, "cells"].sum() / train["cells"].sum()), 3),
                          "n_cells": {"test": int(g.loc[test, "cells"].sum()), "inner_val": int(g.loc[iv, "cells"].sum()),
                                      "inner_train": int(g.loc[it, "cells"].sum())}})
        obj = {"repeat": r, "seed": seed, "K": S["K"], "group_unit": unit, "group_column": unit_col,
               "selection": {"outer_candidates": n_out, "chosen_candidate": int(ocand), "outer_score": round(oscore, 4),
                             "test_folds_identical_to_earlier_repeats": int(n_same),
                             "inner_candidates": n_in},
               "manifest": Path(mpath).as_posix(), "manifest_sha256": sha256_file(mpath), "folds": folds,
               "provenance": provenance(cfg)}
        p = sdir / f"outer_r{r}.json"
        write_json(obj, p)
        digests[p.name] = sha256_file(p)
    from .io_utils import write_text
    write_text("".join(f"{v}  {k}\n" for k, v in sorted(digests.items()))
               + f"{sha256_file(mpath)}  {manifest}\n", sdir / "digests.txt")
    cov = coverage(cfg, unit)
    t = cov[cov.partition == "test"]; iv_ = cov[cov.partition == "inner_val"]; it_ = cov[cov.partition == "inner_train"]
    share = iv_.cells.values / (iv_.cells.values + it_.cells.values)
    return {"unit": unit, "groups": int(len(g)), "distinct_partitions": len(used), "repeats": len(S["repeats"]),
            "test_cells_min_max": [int(t.cells.min()), int(t.cells.max())],
            "test_min_G_S": [int(t.G_cells.min()), int(t.S_cells.min())],
            "inner_val_share_min_max": [round(float(share.min()), 3), round(float(share.max()), 3)],
            "inner_train_cells_min": int(it_.cells.min()),
            "test_folds_repeating_earlier_repeats": overlap, "warnings": warnings}


def split_path(cfg: dict, unit: str, r: int) -> Path:
    return Path(cfg["paths"]["splits"]) / unit / f"outer_r{r}.json"


def load_split(cfg: dict, unit: str, r: int) -> dict:
    return json.loads(split_path(cfg, unit, r).read_text())


def coverage(cfg: dict, unit: str) -> pd.DataFrame:
    rows = []
    for r in cfg["splits"]["repeats"]:
        sp = load_split(cfg, unit, r)
        cells = pd.read_csv(sp["manifest"], dtype={"slide": str, "group_slide": str, "group_image": str})
        col = sp["group_column"]
        for f in sp["folds"]:
            for part in ("test", "inner_val", "inner_train"):
                c = cells[cells[col].astype(str).isin(set(map(str, f[part])))]
                row = {"repeat": r, "fold": f["k"], "partition": part, "groups": len(f[part]), "cells": len(c)}
                for st in STAGES:
                    row[f"{st}_cells"] = int((c.stage == st).sum())
                    row[f"{st}_groups"] = int(c[c.stage == st][col].nunique())
                for s in SPECIES:
                    row[f"{s}_cells"] = int((c.species == s).sum())
                rows.append(row)
    df = pd.DataFrame(rows)
    write_csv(df, out_dir(cfg, "tables") / f"fold_coverage_{unit}.csv")
    return df


def verify(cfg: dict, unit: str) -> list[str]:
    """Returns a list of failures (empty list = gate G2 split checks pass)."""
    fails = []
    for r in cfg["splits"]["repeats"]:
        if not split_path(cfg, unit, r).exists():
            fails.append(f"r{r}: {split_path(cfg, unit, r)} missing"); continue
        sp = load_split(cfg, unit, r)
        if not Path(sp["manifest"]).exists():
            fails.append(f"r{r}: manifest {sp['manifest']} missing (copy the frozen data/manifests here)"); continue
        cells = pd.read_csv(sp["manifest"], dtype={"slide": str, "group_slide": str, "group_image": str})
        if sha256_file(sp["manifest"]) != sp["manifest_sha256"]:
            fails.append(f"r{r}: manifest changed since split was made")
        col = sp["group_column"]
        all_groups = set(cells[col].astype(str))
        seen_test = []
        for f in sp["folds"]:
            T, V, I = (set(map(str, f[p])) for p in ("test", "inner_val", "inner_train"))
            if T & V or T & I or V & I:
                fails.append(f"r{r} k{f['k']}: partitions overlap")
            if T | V | I != all_groups:
                fails.append(f"r{r} k{f['k']}: partitions do not cover all groups")
            seen_test += list(T)
            # image-level and near-duplicate leakage check (stronger than group check)
            img_part = {}
            for part, G in (("test", T), ("inner_val", V), ("inner_train", I)):
                for im in cells[cells[col].astype(str).isin(G)].image_id.unique():
                    img_part.setdefault(im, set()).add(part)
                for dg in cells[cells[col].astype(str).isin(G)].dup_group.unique() if "dup_group" in cells else []:
                    img_part.setdefault("dup:" + str(dg), set()).add(part)
            leak = [k for k, v in img_part.items() if len(v) > 1]
            if leak:
                fails.append(f"r{r} k{f['k']}: images/near-duplicates in two partitions: {leak[:3]}")
            if set(cells[cells[col].astype(str).isin(I)].stage) != set(STAGES):
                fails.append(f"r{r} k{f['k']}: inner training set lacks a stage")
            for st in cfg["splits"]["rare_stages"]:
                if not (cells[cells[col].astype(str).isin(T)].stage == st).any():
                    fails.append(f"r{r} k{f['k']}: no {st} in test fold")
                if not (cells[cells[col].astype(str).isin(V)].stage == st).any():
                    fails.append(f"r{r} k{f['k']}: no {st} in inner validation")
        if sorted(seen_test) != sorted(all_groups):
            fails.append(f"r{r}: groups not in exactly one test fold")
    return fails


def coverage_summary(cov: pd.DataFrame) -> dict:
    """Per-repeat summary of the coverage table: test-fold sizes, rare counts, species-share deviation
    (reported, not enforced: protocol Step 5 asks for +-10 percentage points)."""
    t = cov[cov.partition == "test"].copy()
    sp_cols = [c for c in t.columns if c.endswith("_cells") and c.split("_")[0] in SPECIES]
    tot = t.groupby("repeat")[sp_cols].transform("sum")
    share = t[sp_cols].div(t.cells, axis=0); overall = tot.div(tot.sum(axis=1), axis=0)
    t["max_species_share_dev_pp"] = ((share - overall).abs().max(axis=1) * 100).round(1)
    out = {}
    for r, g in t.groupby("repeat"):
        out[f"r{int(r)}"] = {"test_cells_min_max": [int(g.cells.min()), int(g.cells.max())],
                             "G_cells_min": int(g.G_cells.min()), "S_cells_min": int(g.S_cells.min()),
                             "G_groups_min": int(g.G_groups.min()), "S_groups_min": int(g.S_groups.min()),
                             "max_species_share_dev_pp": float(g.max_species_share_dev_pp.max()),
                             "species_share_within_10pp": bool((g.max_species_share_dev_pp <= 10).all())}
    return out
