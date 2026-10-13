"""python -m malexp <command> [--data-config configs/data.yaml]

  run          nested protocol on the frozen splits (Steps 9-10), resumable; follows runs/<unit>/queue.csv order
  queue        write runs/<unit>/queue.csv (Step 10) without training
  status       finished / pending runs against the queue
  arms-export  write configs/arms/*.yaml, grids.yaml, derived.yaml, digests.txt (gate G4 artefacts)
  pilot        one full-length M-full run on the inner split of (r1, k0): batch fit, epoch time, GPU memory
  smoke        2-epoch nested run of every arm on one fold with the tiny backbone (Step 8 smoke)
  preflight    one bf16 forward/backward/inference step of every arm variant on the device (no data)
  determinism  two same-seed runs of one arm on the inner split (Step 1 check) -> determinism.json
  complexity   params, MACs, CPU latency per arm (Step 13) -> analysis/complexity.csv

Run from the project root (configs/, data/, splits/ resolve), or pass --data-config with an absolute path.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path


def _chdir_to_project(a):
    root = Path(a.data_config).resolve().parent.parent          # <project>/configs/data.yaml -> <project>
    if (root / "data").exists() and (root / "splits").exists() and Path.cwd().resolve() != root:
        os.chdir(root); a.data_config = str(Path("configs") / Path(a.data_config).name)
        print(f"working directory: {root}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="malexp", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["run", "queue", "status", "arms-export", "pilot", "smoke", "preflight",
                                        "determinism", "complexity"])
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--arms-dir", default="configs/arms", help="frozen arm YAMLs; 'builtin' to ignore")
    ap.add_argument("--unit", default="slide", choices=["slide", "image"])
    ap.add_argument("--repeats", type=int, nargs="*", default=[1, 2, 3])
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--tiers", type=int, nargs="*", default=[1, 2, 3])
    ap.add_argument("--backbone", default="mobilevit_xxs.cvnets_in1k")
    ap.add_argument("--input-size", type=int, default=256)
    ap.add_argument("--runs", default=None, help="runs root (default runs/<unit>)")
    ap.add_argument("--with-prob", action="store_true", help="include the optional M-prob arm")
    ap.add_argument("--matched-external", action="store_true", help="use magnification-matched BBBC041 crops")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--tiny", action="store_true", help="test-only backbone")
    ap.add_argument("--device", default=None)
    ap.add_argument("--arm", default="M-full", help="determinism / complexity: arm name")
    ap.add_argument("--out", default="analysis", help="complexity / determinism output folder")
    a = ap.parse_args(argv)

    if a.command == "arms-export":
        from .config import export_arms
        shas = export_arms(a.arms_dir)
        print(json.dumps(shas, indent=1)); return 0

    arms_dir = None if a.arms_dir == "builtin" else a.arms_dir
    if a.command == "preflight":
        from .preflight import run as preflight
        res = preflight("tiny" if a.tiny else a.backbone, a.input_size, a.device, arms_dir=arms_dir if arms_dir and Path(arms_dir).exists() else None)
        for name, st in res:
            print(f"{name:<45} {st}")
        bad = [r for r in res if r[1] != "ok"]
        print(f"{len(res) - len(bad)}/{len(res)} variants pass")
        return 1 if bad else 0

    if a.command == "complexity":
        from .complexity import measure
        from .config import base_arms, derive_from_bstar
        import pandas as pd
        arms = base_arms(arms_dir if arms_dir and Path(arms_dir).exists() else None)
        rows = []
        for name in ["B0", "B3", "B5", "B6", "M-aux", "M-full", "ABL-coordattn"]:
            cfg = arms[name] if name in arms else derive_from_bstar(name, arms["B2"])
            cfg = cfg.with_(backbone="tiny" if a.tiny else a.backbone, input_size=a.input_size)
            rows.append(measure(cfg)); print(rows[-1])
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out / "complexity.csv", index=False, lineterminator="\n")
        return 0

    _chdir_to_project(a)
    from .runner import load_data_cfg, load_external, FoldRunner
    from .select import SelectionLog
    from .config import verify_arms
    data_cfg = load_data_cfg(a.data_config)
    runs_root = Path(a.runs or f"runs/{a.unit}")
    backbone = "tiny" if a.tiny else a.backbone
    if arms_dir is not None:
        if not Path(arms_dir).exists():
            print(f"{arms_dir} not found: run `python -m malexp arms-export` (gate G4) first"); return 1
        bad = verify_arms(arms_dir)
        if bad:
            print(f"arm YAMLs differ from configs/arms/digests.txt: {bad}"); return 1

    if a.command == "queue":
        from .queue import write_queue
        p = write_queue(runs_root, unit=a.unit, backbone=backbone, repeats=a.repeats, folds=a.folds, tiers=a.tiers,
                        with_prob=a.with_prob)
        print(p); return 0

    if a.command == "status":
        from .queue import status
        if not (runs_root / "queue.csv").exists():
            print("no queue.csv: run `python -m malexp queue` first"); return 1
        q = status(runs_root)
        for tier, g in q.groupby("tier"):
            print(f"tier {tier}: {int(g.trained.sum())}/{len(g)} trained, {int(g.finalized.sum())}/{len(g)} finalized")
        for m in sorted(runs_root.rglob("meta.json")):
            d = json.loads(m.read_text())
            print(f"{d['arm']:<14} r{d['repeat']} f{d['fold']} c{d.get('cfg_idx', '?')} best_ep={d['best_epoch']:>3} "
                  f"innerval={d['innerval_macro_f1_at_best']:.3f} {d['elapsed_s']:.0f}s")
        return 0

    external = load_external(data_cfg, a.matched_external)

    if a.command == "pilot":
        from .config import ArmConfig
        from .train import Trainer
        import torch
        cfg = ArmConfig("M-full", backbone=backbone, input_size=a.input_size, use_gate=True, use_boost=True,
                        loss="logit_adjusted")
        t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        tr = Trainer(cfg, data_cfg, a.unit, 1, 0, runs_root / "_pilot", device=a.device, max_epochs=a.max_epochs, external=None)
        tr.fit()
        rep = {"epochs": len(tr.log), "elapsed_s": round(time.time() - t0, 1),
               "s_per_epoch": round((time.time() - t0) / max(1, len(tr.log)), 1), "batch": cfg.batch_size,
               "peak_gpu_mem_MB": round(torch.cuda.max_memory_allocated() / 2 ** 20) if torch.cuda.is_available() else None,
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "torch": torch.__version__,
               "best_epoch": tr.best["epoch"], "note": "pilot scores are never reported (Step 8)"}
        print(json.dumps(rep, indent=1)); (runs_root / "_pilot" / "pilot.json").write_text(json.dumps(rep, indent=1))
        return 0

    if a.command == "determinism":
        from .config import base_arms, derive_from_bstar
        from .determinism import run as det
        arms = base_arms(arms_dir)
        cfg = arms[a.arm] if a.arm in arms else derive_from_bstar(a.arm, arms["B2"])
        cfg = cfg.with_(backbone=backbone, input_size=a.input_size)
        rep = det(cfg, data_cfg, a.unit, a.repeats[0], a.folds[0], runs_root / "_determinism", a.device, a.max_epochs)
        print(json.dumps(rep, indent=1)); return 0 if rep["within_tolerance"] else 1

    if a.command == "smoke":
        fr = FoldRunner(data_cfg, a.unit, runs_root / "_smoke", "tiny", 64, tiers=a.tiers, with_prob=True,
                        max_epochs=a.max_epochs or 2, device=a.device, external=external, arms_dir=arms_dir)
        print(json.dumps(fr.run_fold(a.repeats[0], a.folds[0]), indent=1)); return 0

    # run: follow the queue order (tier-major outside, fold-complete inside)
    from .queue import write_queue
    write_queue(runs_root, unit=a.unit, backbone=backbone, repeats=a.repeats, folds=a.folds, tiers=a.tiers, with_prob=a.with_prob)
    fr = FoldRunner(data_cfg, a.unit, runs_root, backbone, a.input_size, tiers=a.tiers, with_prob=a.with_prob,
                    max_epochs=a.max_epochs, device=a.device, external=external,
                    log=SelectionLog(runs_root / "selection.csv"), arms_dir=arms_dir)
    for r in a.repeats:
        for k in a.folds:
            print(json.dumps(fr.run_fold(r, k)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
