"""Executes the nested protocol fold by fold (Steps 9-10). Resumable: finished runs are skipped.

python -m malexp run --data-config configs/data.yaml --unit slide --repeats 1 2 3 --folds 0 1 2 3 4
                     [--backbone mobilevit_xxs.cvnets_in1k --input-size 256] [--tiers 1 2 3] [--with-prob]

Failure rules (Step 10): a crashed run resumes from its files; a run that diverges (non-finite loss, or
inner-val macro-F1 below the fold's selected B0 score - 0.10) is rerun once with the same seed and the
rerun is logged; if it diverges again it is kept and reported. Test scores never replace a run.
"""
from __future__ import annotations
import json, shutil
from pathlib import Path
import pandas as pd

from .config import ArmConfig, arms_digest
from .select import (SelectionLog, build_fold_plan, tier2_plan, ablation_plan, pick, pick_bstar,
                     TIER1, TIER2, DIVERGENCE_MARGIN)
from .train import Trainer


def load_data_cfg(path):
    from malprep.io_utils import load_config
    return load_config(path)


def load_external(data_cfg, matched: bool = False) -> pd.DataFrame | None:
    """BBBC041 manifest for finalize() only (Step 12). Never used by fit()."""
    p = Path(data_cfg["paths"]["out"]) / "manifests" / "bbbc041_cells.csv"
    if not p.exists():
        return None
    b = pd.read_csv(p)
    if matched and "crop_path_matched" in b:
        b = b.assign(crop_path=b.crop_path_matched)
    b = b.assign(image_group=b.image_id.astype(str))
    return b[["cell_id", "image_id", "image_group", "subset", "species", "stage", "stage_idx", "crop_path"]]


def _done(run_dir: Path) -> bool:
    return (run_dir / "meta.json").exists() and (run_dir / "preds_innerval.parquet").exists()


class FoldRunner:
    def __init__(self, data_cfg, unit, runs_root: Path, backbone, input_size, tiers=(1, 2, 3), with_prob=False,
                 max_epochs=None, device=None, external=None, log: SelectionLog | None = None, arms_dir=None):
        self.data_cfg, self.unit, self.runs_root = data_cfg, unit, Path(runs_root)
        self.backbone, self.input_size, self.tiers, self.with_prob = backbone, input_size, tiers, with_prob
        self.max_epochs, self.device, self.external, self.arms_dir = max_epochs, device, external, arms_dir
        self.log = log or SelectionLog(self.runs_root / "selection.csv")
        self.arms_digest = arms_digest(arms_dir)

    # ------------------------------------------------------------------ one candidate
    def _trainer(self, job, rd, init_from=None):
        return Trainer(job["cfg"], self.data_cfg, job["unit"], job["r"], job["k"], rd, device=self.device,
                       max_epochs=self.max_epochs, external=self.external, init_from=init_from,
                       arms_digest=self.arms_digest)

    def train_candidate(self, job, init_from=None, b0_score=None) -> dict:
        rd = self.runs_root / job["run_id"]
        if (rd / "best.pt").exists() and (rd / "fit.json").exists():
            return json.loads((rd / "fit.json").read_text())
        reruns = 0
        while True:
            tr = self._trainer(job, rd, init_from)
            best = tr.fit()
            diverged = bool(best.get("diverged")) or (b0_score is not None and best["score"] < b0_score - DIVERGENCE_MARGIN)
            fit = {"best": best, "config_sha": job["cfg"].sha256, "diverged": diverged, "reruns": reruns,
                   "init_from": init_from and str(init_from), "phases": tr.phase_log,
                   "train_start_time": tr.start_time, "train_elapsed_s": round(__import__("time").time() - tr.t0, 1)}
            pd.DataFrame(tr.log).to_csv(rd / "log.csv", index=False, lineterminator="\n")
            job["cfg"].save(rd / "config.yaml")
            if diverged and reruns == 0:                      # rerun once with the same seed, keep the log
                shutil.copy(rd / "log.csv", rd / "log_diverged_attempt1.csv")
                (rd / "fit_diverged_attempt1.json").write_text(json.dumps(fit, indent=1))
                reruns += 1
                continue
            (rd / "fit.json").write_text(json.dumps(fit, indent=1))
            return fit

    def finalize(self, job, init_from=None) -> dict:
        rd = self.runs_root / job["run_id"]
        if _done(rd):
            return json.loads((rd / "meta.json").read_text())
        tr = self._trainer(job, rd, init_from)
        if not (rd / "best.pt").exists():
            tr.fit()
        fit = json.loads((rd / "fit.json").read_text())
        tr.best = {k: v for k, v in fit["best"].items() if k != "diverged"}
        tr.diverged = bool(fit.get("diverged", False))
        if not tr.phase_log:                                 # restore the training record of the resumed run
            tr.phase_log = fit.get("phases", [])
            tr.start_time = fit.get("train_start_time", tr.start_time)
        if not hasattr(tr, "val_loader"):                    # restore the val loader for temperature fitting
            from .data import CropDataset, make_loader
            tr.val_ds = CropDataset(tr.val_df, tr.cfg, train=False); tr.val_loader = make_loader(tr.val_ds, tr.cfg, False, 0)
        tr.finalize({"run_id": job["run_id"], "tier": job["tier"], "cfg_idx": job["cfg_idx"],
                     "selection_record": {"file": str(self.log.path), "unit": job["unit"], "repeat": job["r"],
                                          "fold": job["k"], "arm": job["arm"], "cfg_idx": job["cfg_idx"]},
                     "reruns": fit.get("reruns", 0)})
        return json.loads((rd / "meta.json").read_text())

    def _log(self, job, fit, selected, bstar, score=None):
        self.log.add(unit=job["unit"], backbone=job["cfg"].backbone, repeat=job["r"], fold=job["k"], arm=job["arm"],
                     cfg_idx=job["cfg_idx"], selected=selected, config_sha=job["cfg"].sha256,
                     inner_val_macro_f1=score if score is not None else fit["best"]["score"],
                     best_epoch=fit["best"]["epoch"] if fit else -1, diverged=fit.get("diverged", False) if fit else False,
                     reruns=fit.get("reruns", 0) if fit else 0, tier=job["tier"], bstar=bstar, run_id=job["run_id"])

    # ------------------------------------------------------------------ one fold
    def run_fold(self, r: int, k: int) -> dict:
        unit = self.unit
        result = {"unit": unit, "r": r, "k": k}
        jobs = build_fold_plan(unit, r, k, self.backbone, self.input_size, self.arms_dir)
        scores, fits = {}, {}
        b0_dir = b0_score = None
        for arm in [j["arm"] for j in jobs if j["cfg_idx"] == 0]:         # B0 first (arm_order guarantees it)
            scored = []
            for job in [j for j in jobs if j["arm"] == arm]:
                init = None
                if arm == "B4":                                           # cRT phase 1 = the fold's selected B0
                    if b0_dir is None:
                        raise RuntimeError("B4 needs the selected B0 run of the fold")
                    init = b0_dir / "best.pt"
                    job["cfg"] = job["cfg"].with_(lr=scores["B0"][1].lr)   # inherit the selected B0 lr
                fit = self.train_candidate(job, init_from=init, b0_score=b0_score if arm != "B0" else None)
                fits[job["run_id"]] = fit
                scored.append((job["cfg_idx"], fit["best"]["score"], job["cfg"]))
                self._log(job, fit, False, "")
            idx, cfg = pick(arm, scored)
            scores[arm] = (idx, cfg, max(s for i, s, c in scored if i == idx))
            if arm == "B0":
                b0_dir = self.runs_root / next(j["run_id"] for j in jobs if j["arm"] == "B0" and j["cfg_idx"] == idx)
                b0_score = scores["B0"][2]
        bstar_arm = pick_bstar({a: scores[a][2] for a in ["B1", "B2", "B3", "B4"]})
        bstar = scores[bstar_arm][1]
        result["bstar"] = bstar_arm
        result["b0_run"] = str(b0_dir)
        for arm in TIER1:
            idx, cfg, sc = scores[arm]
            job = next(j for j in jobs if j["arm"] == arm and j["cfg_idx"] == idx)
            self._log(job, fits[job["run_id"]], True, bstar_arm, sc)
            if 1 in self.tiers:
                self.finalize(job, init_from=(b0_dir / "best.pt") if arm == "B4" else None)
        # ---- tier 2: arms built on B*
        if 2 in self.tiers:
            jobs2 = tier2_plan(unit, r, k, bstar, self.with_prob, self.arms_dir)
            sel2 = {}
            for arm in [a for a in dict.fromkeys(j["arm"] for j in jobs2)]:
                scored = []
                for job in [j for j in jobs2 if j["arm"] == arm]:
                    fit = self.train_candidate(job, b0_score=b0_score)
                    scored.append((job["cfg_idx"], fit["best"]["score"], job["cfg"]))
                    self._log(job, fit, False, bstar_arm)
                idx, cfg = pick(arm, scored)
                sel2[arm] = cfg
                job = next(j for j in jobs2 if j["arm"] == arm and j["cfg_idx"] == idx)
                self._log(job, json.loads((self.runs_root / job["run_id"] / "fit.json").read_text()), True, bstar_arm,
                          max(s for i, s, c in scored if i == idx))
                self.finalize(job)
            result["selected"] = {a: c.sha256 for a, c in sel2.items()}
            # ---- tier 3: ablations from the selected M-full
            if 3 in self.tiers and "M-full" in sel2:
                for job in ablation_plan(unit, r, k, sel2["M-full"]):
                    fit = self.train_candidate(job, b0_score=b0_score)
                    self.finalize(job)
                    self._log(job, fit, True, bstar_arm)
        return result


def run_fold(data_cfg, unit, r, k, runs_root: Path, backbone, input_size, tiers=(1, 2, 3), with_prob=False,
             max_epochs=None, device=None, external=None, log: SelectionLog | None = None, arms_dir=None) -> dict:
    """Functional entry point (used by the smoke test and the CLI)."""
    fr = FoldRunner(data_cfg, unit, runs_root, backbone, input_size, tiers, with_prob, max_epochs, device, external,
                    log, arms_dir)
    return fr.run_fold(r, k)
