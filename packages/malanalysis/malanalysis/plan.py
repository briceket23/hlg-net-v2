"""The timestamped analysis plan (gate G4): fixed decisions, assumed sigma_d, power statement, and the
SHA-256 of configs/analysis.yaml and of the analysis package itself. Committed (signed tag) before Step 10."""
from __future__ import annotations
import hashlib
from pathlib import Path

from malcore.io import write_text
from malcore.provenance import provenance
from .stats import power_statement, DECISION_WORDING


def package_sha256() -> str:
    h = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob("*.py")):
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()


def render(cfg: dict) -> str:
    S = cfg["stats"]; A = cfg["arms"]
    pw = power_statement(S["rho"], len(cfg["repeats"]) * len(cfg["folds"]), S["sigma_d_assumed"], S["alpha"], S["power"])
    prov = provenance(cfg)
    lines = [
        "# Analysis plan (frozen before any confirmatory run; protocol Steps 13-14)", "",
        f"Timestamp: {prov['created']}  ·  commit: {prov['commit']}{' (dirty)' if prov['commit_dirty'] else ''}",
        f"configs/analysis.yaml SHA-256: {cfg.get('_sha256')}", f"malanalysis package SHA-256: {package_sha256()}", "",
        "## Primary endpoint and contrast",
        f"- Endpoint: per-fold four-class stage macro-F1 on the outer test fold (`zero_division=0`), {len(cfg['repeats'])} repeats x {len(cfg['folds'])} folds.",
        f"- Contrast H1: {A['treatment']} - {A['control']} per (r, k), B* = the best selected rebalanced baseline of each fold (inner validation).",
        f"- Test: corrected resampled t-test, rho = {S['rho']}, df = J - 1, alpha = {S['alpha']} two-sided.",
        f"- Decision rule (smallest effect of interest Delta_min = {S['delta_min']}):",
        *[f"    - {k}: \"{v}\"" for k, v in DECISION_WORDING.items()], "",
        "## Secondary contrasts (Holm within each family)",
        "- Family A (H2), M-full vs B*: schizont F1 and gametocyte F1 pooled per repeat with grouped-bootstrap CIs on the repeat-averaged difference; balanced accuracy (corrected t); external macro-F1 (descriptive).",
        f"- Family B, the ladder: M-full vs each of {', '.join(A['ladder'])}.",
        f"- Family C (H3) and ablations: M-full vs {A['mechanism']}, and M-full vs each of {', '.join(A['ablations'])}.", "",
        "## Sensitivity analyses (reported, not used for decisions)",
        f"1. Image-grouped bootstrap on pooled predictions ({S['n_boot']} resamples within each repeat, paired across arms, seed {S['boot_seed']}).",
        "2. Exact sign test on the paired fold differences.", "3. Primary contrast restricted to repeat 1.",
        "4. Primary contrast with TTA applied to both arms (only if TTA predictions are produced).",
        "5. Primary contrast on EfficientNet-B0 (replication; same decision rule, reported separately).",
        "6. Image-grouped splits (sensitivity for the slide grouping unit).", "",
        "## Power statement", pw["statement"],
        f"Assumed sigma_d range: {S['sigma_d_assumed']}. The observed sigma_d is reported afterwards. Under the correction, extra repeats barely help "
        f"(sqrt(1/J + rho) falls from {pw['se_factor']:.2f} to {pw['se_factor_25_folds']:.2f} with 25 folds); true effects between "
        f"{S['delta_min']} and the MDE will often read as inconclusive, and the paper says so in advance.", "",
        "## Rules", "- Test predictions are opened only once a tier is complete (gate G5/G6); `malanalysis check` must pass first.",
        "- Pooled values are labelled 'pooled' and never share a column with per-fold values.",
        "- External macro-F1 always covers four stages; a stage with fewer than "
        f"{cfg['external']['low_support_n']} cells is flagged low-support and its per-stage F1 shown with its interval.",
        "- Every number in the paper is traceable to stats_log.json or a generated table (gate G7).", ""]
    return "\n".join(lines)


def write_plan(cfg: dict, out_dir: str | Path) -> dict:
    out = Path(out_dir); text = render(cfg)
    sha = write_text(text, out / "analysis_plan.md")
    write_text(f"{sha}  analysis_plan.md\n", out / "analysis_plan.sha256")
    return {"path": str(out / "analysis_plan.md"), "sha256": sha}
