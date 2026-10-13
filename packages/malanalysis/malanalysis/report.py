"""run_analysis(): the one entry point that reads the prediction files and writes every number of the
paper (Steps 13-14) plus the exhibits (Step 17). Writes analysis/stats_log.json.

Order: completeness check (Step 11) -> endpoints -> primary contrast and decision rule -> secondary
families with Holm -> sensitivity analyses -> external evaluation (Step 12) -> tables and figures.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

from malcore import STAGES
from malcore.io import write_csv, write_json
from malcore.provenance import provenance
from .config import load_analysis_config
from .loaders import RunSet, completeness
from .endpoints import (per_fold_metrics, pooled_per_repeat, pooled_by_species, species_macro_f1, reliability_curves,
                        paired_differences)
from .stats import corrected_resampled_t, decision, holm, sign_test, power_statement, group_confusions, grouped_bootstrap
from . import external as ext
from . import tables, figures


def _contrast(pf: pd.DataFrame, a: str, b: str, metric: str, S: dict) -> dict:
    d = paired_differences(pf, a, b, metric)
    t = corrected_resampled_t(d.d.to_numpy(), S["rho"], S["alpha"])
    return {"a": a, "b": b, "metric": metric, "test": t,
            "per_fold": [{"repeat": int(r), "fold": int(k), "d": float(x)} for r, k, x in zip(d.repeat, d.fold, d.d)]}


def _family(pf, contrasts: list[tuple[str, str, str]], S: dict) -> dict:
    cs = {f"{a} - {b} ({m})" if m != "macro_f1" else f"{a} - {b}": _contrast(pf, a, b, m, S) for a, b, m in contrasts}
    cs = {k: v for k, v in cs.items() if "p" in v["test"]}
    return {"contrasts": cs, "holm": holm({k: v["test"]["p"] for k, v in cs.items()}, S["alpha"])}


def _pooled_pairs(rs: RunSet, a: str, b: str, repeats, folds):
    """Per repeat, paired group-confusion stacks of the pooled test predictions of the two arms."""
    pairs = []
    for r in repeats:
        pa = [rs.preds(a, r, k) for k in folds]; pb = [rs.preds(b, r, k) for k in folds]
        if any(p is None for p in pa + pb):
            continue
        pa = pd.concat(pa, ignore_index=True).set_index("cell_id"); pb = pd.concat(pb, ignore_index=True).set_index("cell_id")
        pb = pb.loc[pa.index]
        cm_a, keys = group_confusions(pa.y_true, pa.y_pred, pa.image_group); cm_b, _ = group_confusions(pb.y_true, pb.y_pred, pb.image_group)
        pairs.append((cm_a, cm_b))
    return pairs


def run_analysis(cfg_path: str | Path | None = None, cfg: dict | None = None, skip_completeness: bool = False) -> dict:
    cfg = cfg or load_analysis_config(cfg_path)
    S, A, R, F = cfg["stats"], cfg["arms"], cfg["repeats"], cfg["folds"]
    out = Path(cfg["out"]); out.mkdir(parents=True, exist_ok=True)
    rs = RunSet(cfg["runs_root"], cfg.get("backbone"))
    arms_needed = [A["treatment"], A["control"], A["mechanism"]] + list(A["ladder"]) + list(A["ablations"])
    present = [a for a in arms_needed if a in rs.arms()]
    log = {"provenance": provenance(cfg), "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
           "arms_present": present, "arms_missing": [a for a in arms_needed if a not in present]}
    # ---- Step 11 completeness (tier 1 = treatment, control, mechanism must be complete)
    tier1 = [a for a in (A["treatment"], A["control"], A["mechanism"]) if a in rs.arms()]
    comp = completeness(rs, tier1, R, F, cfg.get("manifest"), cfg.get("splits_dir"), cfg["unit"])
    log["completeness_tier1"] = comp
    if not comp["ok"] and not skip_completeness:
        write_json(log, out / "stats_log.json")
        raise SystemExit("completeness check failed (gate G5): " + "; ".join(comp["problems"][:5]))
    log["completeness_all"] = completeness(rs, present, R, F, cfg.get("manifest"), cfg.get("splits_dir"), cfg["unit"])
    # ---- Step 13 endpoints
    pf = per_fold_metrics(rs, present, R, F); write_csv(pf, out / "per_fold_metrics.csv")
    pooled, cms = pooled_per_repeat(rs, present, R, F); write_csv(pooled, out / "pooled_per_repeat.csv")
    if "B5" in present:
        write_csv(species_macro_f1(rs, "B5", R, F), out / "species_macro_f1_B5.csv")
    for arm in (A["control"], A["treatment"]):
        if arm in present:
            sp = pooled_by_species(rs, arm, R, F)
            if len(sp):
                write_csv(sp.drop(columns=["confusion"]), out / f"pooled_by_species_{arm.replace('*', 'star')}.csv")
    # ---- Step 14 primary contrast
    T, C, M = A["treatment"], A["control"], A["mechanism"]
    prim = _contrast(pf, T, C, "macro_f1", S)
    prim["decision"] = decision(prim["test"]["mean"], prim["test"]["ci_lo"], prim["test"]["ci_hi"], S["delta_min"])
    prim["observed_sigma_d"] = prim["test"].get("sd")
    log["primary"] = prim
    log["power"] = power_statement(S["rho"], len(R) * len(F), S["sigma_d_assumed"], S["alpha"], S["power"])
    # ---- secondary families
    famA = {"contrasts": {}, "bootstrap": {}}
    bal = _contrast(pf, T, C, "balanced_accuracy", S); famA["contrasts"]["balanced_accuracy"] = bal
    pvals = {"balanced_accuracy": bal["test"]["p"]}
    pairs = _pooled_pairs(rs, T, C, R, F)
    for stat in ("f1_S", "f1_G"):
        if pairs:
            b = grouped_bootstrap(pairs, stat, S["n_boot"], S["boot_seed"]); famA["bootstrap"][stat] = b; pvals[stat] = b["p_boot"]
    famA["holm"] = holm(pvals, S["alpha"]); log["family_A"] = famA
    log["family_B"] = _family(pf, [(T, b, "macro_f1") for b in A["ladder"] if b in present], S)
    log["family_C"] = _family(pf, [(T, M, "macro_f1")] + [(T, b, "macro_f1") for b in A["ablations"] if b in present], S)
    # ---- sensitivity analyses (reported, not used for decisions)
    sens = {}
    if pairs:
        sens["grouped_bootstrap_macro_f1"] = grouped_bootstrap(pairs, "macro_f1", S["n_boot"], S["boot_seed"])
    sens["sign_test"] = sign_test([x["d"] for x in prim["per_fold"]])
    d1 = [x["d"] for x in prim["per_fold"] if x["repeat"] == R[0]]
    sens["repeat_1_only"] = corrected_resampled_t(d1, S["rho"], S["alpha"]) if len(d1) > 1 else {"note": "not available"}
    sens["tta"] = {"note": "not run: no TTA predictions are produced by the frozen recipe"}
    for key, root in (("replication_efficientnet_b0", cfg["sensitivity"].get("replication_runs_root")),
                      ("image_grouped_splits", cfg["sensitivity"].get("image_unit_runs_root"))):
        if root and Path(root).exists():
            try:
                rs2 = RunSet(root); pf2 = per_fold_metrics(rs2, [T, C], R, F)
                c2 = _contrast(pf2, T, C, "macro_f1", S)
                c2["decision"] = decision(c2["test"]["mean"], c2["test"]["ci_lo"], c2["test"]["ci_hi"], S["delta_min"])
                sens[key] = c2
            except Exception as e:                            # a missing sensitivity run is reported, never fatal
                sens[key] = {"note": f"not available: {e}"}
        else:
            sens[key] = {"note": "not run"}
    log["sensitivity"] = sens
    # ---- Step 12 external
    E = cfg["external"]
    bm = ext.by_model(rs, present, R, F, E["subsets"], E["low_support_n"])
    if len(bm):
        write_csv(bm.drop(columns=["confusion"]), out / "external_by_model.csv")
        asum = ext.arm_summary(bm); write_csv(asum, out / "external_by_arm.csv")
        contrast = ext.paired_contrast(bm, T, C)
        boot = ext.bootstrap_contrast(rs, T, C, R, F, S["n_boot"], S["boot_seed"])
        write_csv(pd.DataFrame([boot]), out / "external_bootstrap.csv")
        by_subset = {sub: ext.paired_contrast(bm, T, C, sub) for sub in E["subsets"]}
        low = {st: bool(bm[(bm.subset == "pooled")][f"low_support_{st}"].any()) for st in STAGES}
        log["external"] = {"paired_contrast": contrast, "paired_contrast_by_subset": by_subset, "bootstrap": boot,
                           "low_support_flags": low, "by_arm": asum.to_dict("records"),
                           "note": "secondary endpoint; cannot rescue a failed primary; models share one external set"}
        tables.table_external(asum, cfg.get("external_data_table"), contrast, out)
    else:
        log["external"] = {"note": "no BBBC041 predictions found"}
    # ---- exhibits
    comp_csv = cfg.get("complexity_table"); complexity = pd.read_csv(comp_csv) if comp_csv and Path(comp_csv).exists() else None
    tables.table_fold_coverage(cfg.get("coverage_table"), out)
    order = [a for a in A["ladder"] + [C, M, T] if a in present]
    tables.table_main(pf, pooled, order, prim, complexity, out)
    tables.table_ablation(log["family_C"], out)
    tables.table_ablation(log["family_B"], out, label="tab:ladder", caption="M-full against the baseline ladder (Holm-adjusted within the family).")
    log["selection_summary"] = tables.table_selection(rs.sel, rs.runs, out)
    figs = {f"{T} - {C}": prim}
    if f"{T} - {M}" in log["family_C"]["contrasts"]:
        figs[f"{T} - {M}"] = log["family_C"]["contrasts"][f"{T} - {M}"]
    figures.fig_paired_differences(figs, out)
    conf = {}
    for arm in (C, T):
        if arm in cms:
            conf[f"{arm} (MP-IDB, pooled)"] = sum(cms[arm].values())
        if len(bm):
            conf[f"{arm} (BBBC041)"] = ext.pooled_confusion(rs, arm, R, F)
    figures.fig_confusions(conf, out)
    figures.fig_reliability(reliability_curves(rs, [a for a in (C, T, M) if a in present], R, F), out)
    log["distinct_test_folds"] = _distinct_test_folds(cfg)
    log["verdict"] = prim["decision"]["wording"]
    write_json(log, out / "stats_log.json")
    return log


def _distinct_test_folds(cfg) -> dict | None:
    """With few slide groups the repeats are not independent resplits: report the number of distinct test folds."""
    sd = Path(cfg.get("splits_dir") or "") / cfg["unit"]
    if not sd.exists():
        return None
    folds = []
    for r in cfg["repeats"]:
        p = sd / f"outer_r{r}.json"
        if p.exists():
            folds += [frozenset(f["test"]) for f in json.loads(p.read_text())["folds"]]
    return {"n_test_folds": len(folds), "n_distinct": len(set(folds))} if folds else None
