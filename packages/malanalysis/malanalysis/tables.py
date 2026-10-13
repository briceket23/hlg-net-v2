"""Paper exhibits as CSV and LaTeX (Step 17). Every table is generated; none is typed by hand."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from malcore import STAGES, STAGE_NAMES
from malcore.io import write_csv, write_text


def _esc(s) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def to_latex(df: pd.DataFrame, caption: str, label: str, fmt: str = "{:.3f}", note: str = "") -> str:
    cols = list(df.columns)
    lines = [r"\begin{table}[t]", r"\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             r"\begin{tabular}{l" + "r" * (len(cols) - 1) + "}", r"\toprule",
             " & ".join(_esc(c) for c in cols) + r" \\", r"\midrule"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append("--" if np.isnan(v) else fmt.format(v))
            else:
                cells.append(_esc(v))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if note:
        lines.append(f"\\par\\footnotesize {note}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def ms(x) -> str:
    x = np.asarray(x, dtype=float)
    return "--" if len(x) == 0 else f"{x.mean():.3f} $\\pm$ {x.std(ddof=1) if len(x) > 1 else 0:.3f}"


def table_fold_coverage(coverage_csv: str | Path | None, out: Path) -> pd.DataFrame | None:
    """Table 2: rare-stage cells and groups per test fold (summary; the full table goes to the supplement)."""
    if not coverage_csv or not Path(coverage_csv).exists():
        return None
    cov = pd.read_csv(coverage_csv); t = cov[cov.partition == "test"]
    rows = []
    for r, g in t.groupby("repeat"):
        rows.append({"repeat": int(r), "test cells (min-max)": f"{int(g.cells.min())}-{int(g.cells.max())}",
                     "schizont cells (min-max)": f"{int(g.S_cells.min())}-{int(g.S_cells.max())}",
                     "schizont groups (min)": int(g.S_groups.min()),
                     "gametocyte cells (min-max)": f"{int(g.G_cells.min())}-{int(g.G_cells.max())}",
                     "gametocyte groups (min)": int(g.G_groups.min())})
    df = pd.DataFrame(rows)
    write_csv(df, out / "table2_fold_coverage.csv"); write_csv(cov, out / "supp_fold_coverage_full.csv")
    write_text(to_latex(df, "Fold coverage: rare-stage cells and image groups per outer test fold (summary per repeat).",
                        "tab:fold-coverage"), out / "table2_fold_coverage.tex")
    return df


def table_main(pf: pd.DataFrame, pooled: pd.DataFrame, arms: list[str], primary: dict, complexity: pd.DataFrame | None,
               out: Path) -> pd.DataFrame:
    """Table 3: per-fold macro-F1 (mean +- SD), pooled schizont / gametocyte F1 (mean of the per-repeat pooled
    values), balanced accuracy, ECE, params and latency, then the primary contrast row."""
    rows = []
    for arm in arms:
        f = pf[pf.arm == arm]; p = pooled[pooled.arm == arm]
        if len(f) == 0:
            continue
        row = {"arm": arm, "folds": int(len(f)), "macro-F1 per-fold (mean ± SD)": ms(f.macro_f1),
               "schizont F1 (pooled, mean of repeats)": f"{p.f1_S.mean():.3f}" if len(p) else "--",
               "gametocyte F1 (pooled, mean of repeats)": f"{p.f1_G.mean():.3f}" if len(p) else "--",
               "balanced acc. per-fold (mean ± SD)": ms(f.balanced_accuracy), "ECE per-fold (mean ± SD)": ms(f.ece)}
        if complexity is not None and len(complexity):
            key = "B*" if arm == "B*" else arm
            c = complexity[complexity.arm == key]
            if len(c) == 0 and arm.startswith(("B1", "B2", "B3", "B4", "B5", "B*", "M-aux", "ABL-noaux", "ABL-detach")):
                c = complexity[complexity.arm == ("M-full" if arm.startswith(("M-", "ABL-")) else "B0")]
            row["params (M)"] = f"{float(c.params_M.iloc[0]):.2f}" if len(c) else "--"
            row["CPU latency ms (median)"] = f"{float(c.median_ms.iloc[0]):.1f}" if len(c) else "--"
        rows.append(row)
    df = pd.DataFrame(rows)
    write_csv(df, out / "table3_main.csv")
    pr = primary["test"]; dec = primary["decision"]
    note = (f"Primary contrast {primary['a']} $-$ {primary['b']}: $\\bar d$ = {pr['mean']:.3f}, corrected 95\\% CI "
            f"[{pr['ci_lo']:.3f}, {pr['ci_hi']:.3f}] (J = {pr['J']}, $\\rho$ = {pr['rho']}, df = {pr['df']}, p = {pr['p']:.3f}). "
            f"Verdict: {_esc(dec['wording'])}.")
    write_text(to_latex(df, "Main result on MP-IDB (3 repeats x 5 folds). Per-fold values are mean $\\pm$ SD over 15 folds; "
                        "pooled values are computed on the predictions pooled within each repeat and averaged over the 3 repeats.",
                        "tab:main", note=note), out / "table3_main.tex")
    return df


def table_ablation(family: dict, out: Path, label="tab:ablation", caption="Ablations relative to M-full (Holm-adjusted within the family)."):
    """Table 4: each contrast with mean difference, corrected CI, raw and Holm-adjusted p, and the interpretation rule."""
    rows = []
    for name, c in family["contrasts"].items():
        t = c["test"]; h = family["holm"].get(name, {})
        contributes = (h.get("reject") is True) and t["mean"] > 0 and t["ci_lo"] > 0
        rows.append({"contrast": name, "mean difference": t["mean"], "CI low": t["ci_lo"], "CI high": t["ci_hi"],
                     "p": t["p"], "p (Holm)": h.get("p_adj"),
                     "interpretation": "contributes" if contributes else "no evidence that it contributes"})
    df = pd.DataFrame(rows)
    stem = label.split(":")[-1]
    write_csv(df, out / f"table4_{stem}.csv"); write_text(to_latex(df, caption, label), out / f"table4_{stem}.tex")
    return df


def table_external(arm_sum: pd.DataFrame, data_table: str | Path | None, contrast: dict, out: Path) -> pd.DataFrame:
    """Table 5: (a) external data (copied from malprep's table5a) and (b) results by arm and subset."""
    if data_table and Path(data_table).exists():
        d = pd.read_csv(data_table); write_csv(d, out / "table5a_external_data.csv")
        write_text(to_latex(d, "BBBC041 external data: cells per stage, images and official subset.", "tab:external-data"),
                   out / "table5a_external_data.tex")
    df = arm_sum[["arm", "subset", "n", "mean", "ci_lo", "ci_hi"] + [f"mean_f1_{s}" for s in STAGES]].copy()
    df.columns = ["arm", "subset", "models", "macro-F1 (mean)", "CI low", "CI high"] + [f"{STAGE_NAMES[s]} F1" for s in STAGES]
    write_csv(df, out / "table5b_external_results.csv")
    note = (f"Paired contrast {contrast['a']} $-$ {contrast['b']} (pooled subsets): mean {contrast['mean']:.3f}, "
            f"range [{contrast['min']:.3f}, {contrast['max']:.3f}] over {contrast['n_models']} fold models. "
            "Intervals are descriptive t-intervals across the fold models, which share one external set.")
    write_text(to_latex(df, "BBBC041 zero-shot results by arm and official subset (15 fold models per arm).",
                        "tab:external-results", note=note), out / "table5b_external_results.tex")
    return df


def table_selection(sel: pd.DataFrame, runs: pd.DataFrame, out: Path) -> dict:
    """Supplement: how often each arm was B*, and the gate weights / learning rates selected."""
    s = sel[sel.selected.astype(bool)]
    bst = s.drop_duplicates(["repeat", "fold"])[["repeat", "fold", "bstar"]]
    counts = bst.bstar.value_counts().to_dict()
    gate = runs[runs.arm.isin(["M-aux", "M-full"]) & runs.lambda_gate.notna()].groupby("arm").lambda_gate.value_counts().to_dict()
    lrs = runs[runs.arm.isin(["B0", "B1", "B2", "B3", "B6"]) & runs.lr.notna()].groupby("arm").lr.value_counts().to_dict()
    write_csv(sel, out / "supp_selection_log.csv")
    rep = {"bstar_counts": {str(k): int(v) for k, v in counts.items()},
           "gate_weights_selected": {f"{a}:{g}": int(n) for (a, g), n in gate.items()},
           "lr_selected": {f"{a}:{l}": int(n) for (a, l), n in lrs.items()}}
    df = pd.DataFrame([{"arm": k, "times selected as B*": v} for k, v in sorted(counts.items())])
    write_csv(df, out / "supp_bstar_counts.csv")
    return rep
