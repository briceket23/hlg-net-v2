"""python -m malanalysis <command> [--config configs/analysis.yaml]

  check   Step 11 completeness check on the runs tree (gate G5/G6); exit 1 on any problem
  plan    write the timestamped analysis plan (analysis/analysis_plan.md + .sha256) (gate G4)
  run     Steps 13-14 + 17: metrics, statistics, decision rule, tables, figures -> analysis/stats_log.json
  synth   write a synthetic 180-run tree for testing (--out <folder>)
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="malanalysis", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["check", "plan", "run", "synth"])
    ap.add_argument("--config", default="configs/analysis.yaml")
    ap.add_argument("--out", default=None)
    ap.add_argument("--runs-root", default=None)
    ap.add_argument("--skip-completeness", action="store_true", help="development only; never for the paper")
    a = ap.parse_args(argv)
    from .config import load_analysis_config
    cfg = load_analysis_config(a.config)
    if a.runs_root: cfg["runs_root"] = a.runs_root
    if a.out: cfg["out"] = a.out
    if a.command == "synth":
        from .synthetic import make_synthetic_runs
        print(json.dumps(make_synthetic_runs(a.out or "synthetic_runs"), indent=1)); return 0
    if a.command == "plan":
        from .plan import write_plan
        print(json.dumps(write_plan(cfg, cfg["out"]), indent=1)); return 0
    if a.command == "check":
        from .loaders import RunSet, completeness
        rs = RunSet(cfg["runs_root"], cfg.get("backbone"))
        A = cfg["arms"]; arms = [x for x in [A["treatment"], A["control"], A["mechanism"]] + A["ladder"] + A["ablations"] if x in rs.arms()]
        rep = completeness(rs, arms, cfg["repeats"], cfg["folds"], cfg.get("manifest"), cfg.get("splits_dir"), cfg["unit"])
        print(json.dumps(rep, indent=1)); return 0 if rep["ok"] else 1
    from .report import run_analysis
    log = run_analysis(cfg=cfg, skip_completeness=a.skip_completeness)
    print(json.dumps({"verdict": log["verdict"], "primary": log["primary"]["test"], "out": cfg["out"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
