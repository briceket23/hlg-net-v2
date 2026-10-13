"""malanalysis tests: statistics against reference formulas and the end-to-end run on a synthetic 180-run tree."""
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from malanalysis.stats import (corrected_resampled_t, decision, holm, sign_test, power_statement, grouped_bootstrap,
                               group_confusions, stat_from_cm, DECISION_WORDING)
from malanalysis.synthetic import make_synthetic_runs
from malanalysis.config import load_analysis_config
from malanalysis.loaders import RunSet, completeness
from malanalysis.report import run_analysis


def test_corrected_t_matches_formula():
    rng = np.random.default_rng(0); d = rng.normal(0.03, 0.02, 15)
    r = corrected_resampled_t(d, rho=0.25)
    se = math.sqrt((1 / 15 + 0.25) * d.var(ddof=1))
    assert abs(r["se_corrected"] - se) < 1e-12 and abs(r["t"] - d.mean() / se) < 1e-9 and r["df"] == 14
    assert abs(r["ci_hi"] - (d.mean() + sps.t.ppf(0.975, 14) * se)) < 1e-9
    assert abs(r["p"] - 2 * sps.t.sf(abs(d.mean() / se), 14)) < 1e-12
    assert r["se_corrected"] > r["naive_se"]                      # the correction widens the interval


def test_decision_rule_wording():
    assert decision(0.03, 0.01, 0.05)["wording"] == DECISION_WORDING["supported"]
    assert decision(0.015, 0.005, 0.03)["verdict"] == "below_delta"
    assert decision(0.005, -0.005, 0.015)["verdict"] == "no_improvement"
    assert decision(0.01, -0.01, 0.03)["verdict"] == "inconclusive"
    assert decision(-0.03, -0.05, -0.01)["verdict"] == "worse"


def test_holm_and_sign_test():
    h = holm({"a": 0.01, "b": 0.04, "c": 0.03}, 0.05)
    assert abs(h["a"]["p_adj"] - 0.03) < 1e-12 and abs(h["c"]["p_adj"] - 0.06) < 1e-12 and abs(h["b"]["p_adj"] - 0.06) < 1e-12
    assert h["a"]["reject"] and not h["b"]["reject"]
    s = sign_test([1, 1, 1, 1, 1, -1, 0])
    assert s["n_nonzero"] == 6 and s["n_positive"] == 5 and abs(s["p"] - sps.binomtest(5, 6, 0.5).pvalue) < 1e-12


def test_power_statement_matches_protocol_numbers():
    p = power_statement(0.25, 15, (0.02, 0.03))
    assert abs(p["se_factor"] - 0.5627) < 1e-3 and abs(p["t_multiplier"] - 3.01) < 0.02
    assert abs(p["mde"]["0.02"] - 0.034) < 0.002 and abs(p["mde"]["0.03"] - 0.051) < 0.002
    assert abs(p["se_factor_25_folds"] - 0.539) < 1e-3


def test_grouped_bootstrap_recovers_pooled_difference():
    rng = np.random.default_rng(1); n = 2000
    y = rng.choice(4, n, p=[0.8, 0.1, 0.05, 0.05]); g = rng.integers(0, 40, n)
    pa = np.where(rng.random(n) < 0.8, y, rng.integers(0, 4, n)); pb = np.where(rng.random(n) < 0.6, y, rng.integers(0, 4, n))
    cm_a, _ = group_confusions(y, pa, g); cm_b, _ = group_confusions(y, pb, g)
    from malcore.metrics import macro_f1
    true = macro_f1(y, pa) - macro_f1(y, pb)
    assert abs(float(stat_from_cm(cm_a.sum(0)[None], "macro_f1")[0]) - macro_f1(y, pa)) < 1e-12
    b = grouped_bootstrap([(cm_a, cm_b)], "macro_f1", 500, 0)
    assert abs(b["mean_diff"] - true) < 1e-12 and b["ci_lo"] < true < b["ci_hi"] and b["ci_lo"] > 0


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth")
    info = make_synthetic_runs(root, effect=0.30, seed=0)
    assert info["n_runs"] == 180
    return root, info


def test_completeness_check(synth):
    root, info = synth
    rs = RunSet(info["runs_root"], "tiny")
    assert "B*" in rs.arms() and len(rs.runs) == 180
    rep = completeness(rs, ["M-full", "B*", "M-aux"], [1, 2, 3], [0, 1, 2, 3, 4], info["manifest"], info["splits_dir"])
    assert rep["ok"], rep["problems"]
    p = Path(rs.run_dir("M-full", 2, 3)) / "preds_bbbc041.parquet"; p.rename(p.with_suffix(".bak"))
    try:
        rep = completeness(rs, ["M-full"], [1, 2, 3], [0, 1, 2, 3, 4], info["manifest"], info["splits_dir"])
        assert not rep["ok"] and any("preds_bbbc041" in x for x in rep["problems"])
    finally:
        p.with_suffix(".bak").rename(p)


def test_run_analysis_end_to_end(synth, tmp_path):
    root, info = synth
    cfg = load_analysis_config(None)
    cfg.update(runs_root=info["runs_root"], backbone="tiny", manifest=info["manifest"], splits_dir=info["splits_dir"],
               out=str(tmp_path / "analysis"), coverage_table=None, complexity_table=None)
    cfg["stats"]["n_boot"] = 200
    log = run_analysis(cfg=cfg)
    out = tmp_path / "analysis"
    for f in ["stats_log.json", "per_fold_metrics.csv", "pooled_per_repeat.csv", "table3_main.tex", "table3_main.csv",
              "table4_ablation.tex", "table4_ladder.tex", "table5b_external_results.tex", "external_by_model.csv",
              "external_bootstrap.csv", "fig2_paired_differences.pdf", "fig3_confusion.pdf", "fig4_reliability.pdf",
              "supp_selection_log.csv", "supp_bstar_counts.csv"]:
        assert (out / f).exists(), f
    assert log["primary"]["test"]["J"] == 15 and log["verdict"] in DECISION_WORDING.values()
    assert log["primary"]["test"]["mean"] > 0 and log["primary"]["decision"]["verdict"] == "supported"
    assert set(log["family_B"]["contrasts"]) == {f"M-full - {b}" for b in ["B0", "B1", "B2", "B3", "B4", "B5", "B6"]}
    assert set(log["family_C"]["contrasts"]) == {"M-full - M-aux", "M-full - ABL-noaux", "M-full - ABL-detach", "M-full - ABL-coordattn"}
    assert "f1_S" in log["family_A"]["bootstrap"] and log["family_A"]["holm"]["f1_S"]["p_adj"] is not None
    assert log["external"]["paired_contrast"]["n_models"] == 15 and "train" in log["external"]["paired_contrast_by_subset"]
    assert log["distinct_test_folds"]["n_test_folds"] == 15
    pf = pd.read_csv(out / "per_fold_metrics.csv")
    assert (pf.regime == "per-fold").all() and len(pf[pf.arm == "B*"]) == 15
    assert (pd.read_csv(out / "pooled_per_repeat.csv").regime == "pooled").all()
    assert log["selection_summary"]["bstar_counts"] and sum(log["selection_summary"]["bstar_counts"].values()) == 15
    tex = (out / "table3_main.tex").read_text()
    assert "pooled" in tex and "per-fold" in tex and "Verdict:" in tex


def test_plan_is_hashed(tmp_path):
    from malanalysis.plan import write_plan
    cfg = load_analysis_config(None)
    rep = write_plan(cfg, tmp_path)
    assert (tmp_path / "analysis_plan.md").exists() and rep["sha256"] in (tmp_path / "analysis_plan.sha256").read_text()
    assert "Delta_min = 0.02" in (tmp_path / "analysis_plan.md").read_text()
