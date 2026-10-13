"""One run = one (arm, config, repeat, fold): protocol Step 7 "Run procedure" and Step 11 prediction files.

    tr = Trainer(cfg, data_cfg, unit, r, k, run_dir)
    tr.fit()                 # inner_train / inner_val only; a test loader cannot be built here
    tr.finalize()            # best checkpoint -> temperature on inner_val -> test (+ BBBC041) predictions

Schedules (Step 6 "Rules", Step 7):
  single  early stopping on inner-val macro-F1 (patience 15, min delta 0.002), 60-epoch cap;
          fixed_budget=True (M-prob) disables early stopping, best checkpoint over the whole run
  drw     B3: phase 1 = LDAM for the first 80 % of the cap, phase 2 = LDAM + DRW re-weighting for the
          rest; no early stopping; the checkpoint is selected within phase 2 only
  crt     B4: phase 1 = the fold's selected B0 run (loaded through `init_from`; trained here only when
          no B0 checkpoint is given), phase 2 = backbone frozen, new linear head, q = 1 sampler,
          15 epochs, head lr from the grid; checkpoint selected within phase 2 only. Arms derived from
          a B4 B* train phase 1 themselves (with their gate / heads) and keep gate and boost frozen in
          phase 2.
"""
from __future__ import annotations
import json, math, os, platform, random, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from malcore import STAGES
from malcore.provenance import git_commit, gpu_info, now, package_versions
from malcore.hashing import read_digests, sha256_file
from .config import ArmConfig
from .data import CropDataset, make_loader, heldout_loader, sample_weights, sampled_rare_prevalence, HeldoutGuard
from .losses import total_loss
from .metrics import summary, macro_f1
from .models import StageNet, state_sha256


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def run_seed(r: int, k: int, offset: int = 0) -> int:
    """Initialisation seed of run (r, k) = 1000 r + k (Step 5); split seeds are 100 + r."""
    return 1000 * r + k + offset


def param_groups(model: StageNet, cfg: ArmConfig):
    head_ids = {id(p) for p in model.head_parameters()}
    groups = {"bb_decay": [], "bb_nodecay": [], "head_decay": [], "head_nodecay": []}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        head = id(p) in head_ids
        nodecay = p.ndim <= 1 or n.endswith(".bias")          # no decay on bias and norm parameters
        groups[("head" if head else "bb") + ("_nodecay" if nodecay else "_decay")].append(p)
    lr_h = cfg.lr * cfg.head_lr_mult
    return [{"params": groups["bb_decay"], "lr": cfg.lr, "weight_decay": cfg.weight_decay},
            {"params": groups["bb_nodecay"], "lr": cfg.lr, "weight_decay": 0.0},
            {"params": groups["head_decay"], "lr": lr_h, "weight_decay": cfg.weight_decay},
            {"params": groups["head_nodecay"], "lr": lr_h, "weight_decay": 0.0}]


def lr_factor(epoch: float, warmup: int, total: int) -> float:
    """Linear warm-up over `warmup` epochs, then cosine decay to 1 % of the base rate."""
    if epoch < warmup:
        return (epoch + 1) / warmup
    t = (epoch - warmup) / max(1, total - warmup)
    return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))


def fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> float:
    """One temperature T minimising the NLL on inner-validation logits (L-BFGS). Changes probabilities, not argmax."""
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad(); loss = F.cross_entropy(logits / logT.exp(), y); loss.backward(); return loss
    opt.step(closure)
    return float(logT.exp().detach())


def check_split_digests(data_cfg: dict, unit: str) -> dict:
    """Step 7 run procedure (1): assert the split files and the manifest match splits/<unit>/digests.txt."""
    sdir = Path(data_cfg["paths"]["splits"]) / unit
    dig = sdir / "digests.txt"
    if not dig.exists():
        raise FileNotFoundError(f"{dig} missing: splits are not frozen (gate G2)")
    rec = read_digests(dig)
    md = Path(data_cfg["paths"]["out"]) / "manifests"
    bad = []
    for name, sha in rec.items():
        p = sdir / name if name.startswith("outer_") else md / name
        if not p.exists() or sha256_file(p) != sha:
            bad.append(name)
    if bad:
        raise RuntimeError(f"split/manifest digests differ from {dig}: {bad}")
    return rec


class Trainer:
    def __init__(self, cfg: ArmConfig, data_cfg: dict, unit: str, r: int, k: int, run_dir: str | Path,
                 device: str | None = None, max_epochs: int | None = None, external: pd.DataFrame | None = None,
                 init_from: str | Path | None = None, arms_digest: str | None = None):
        from malprep.folds import get_partition, fold_digests
        self.cfg, self.data_cfg, self.unit, self.r, self.k = cfg, data_cfg, unit, r, k
        self.run_dir = Path(run_dir); self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = max_epochs or cfg.epochs
        self.seed = run_seed(r, k, cfg.seed_offset)
        self.start_time = now(); self.t0 = time.time()
        self.split_digests = check_split_digests(data_cfg, unit)
        self.digests = fold_digests(data_cfg, unit, r)
        self.arms_digest = arms_digest
        self.init_from = str(init_from) if init_from else None
        seed_everything(self.seed)
        self.train_df = get_partition(data_cfg, unit, r, k, "inner_train")
        self.val_df = get_partition(data_cfg, unit, r, k, "inner_val")
        self.external = external
        self._test_df = None                                 # built only in finalize()
        y = self.train_df.stage_idx.to_numpy()
        self.counts = torch.as_tensor(np.bincount(y, minlength=4))
        self.weights = sample_weights(y, cfg)
        self.gate_prev = sampled_rare_prevalence(y, self.weights)
        self.gate_pos_weight = (1 - self.gate_prev) / max(self.gate_prev, 1e-6)
        self.model = StageNet(cfg).to(self.device)
        self.pretrained_sha256 = state_sha256(self.model.backbone) if cfg.pretrained and cfg.backbone != "tiny" else None
        self.log: list[dict] = []
        self.best = {"epoch": -1, "score": -1.0, "phase": None}
        self.diverged = False
        self.phase_log: list[dict] = []

    # ------------------------------------------------------------------ training
    def _epoch(self, loader, opt, epoch, dtype):
        self.model.train(); tot = 0.0; n = 0; comps_sum = {}
        for x, y, sp, _ in loader:
            x, y, sp = x.to(self.device), y.to(self.device), sp.to(self.device)
            with torch.autocast(self.device.split(":")[0], dtype=dtype, enabled=dtype is not None):
                out = self.model(x)
                loss, comps = total_loss(self.cfg, out, y, sp, self.counts, epoch / self.epochs, self.gate_pos_weight)
            opt.zero_grad(set_to_none=True); loss.backward()
            if self.cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            opt.step()
            tot += float(loss) * len(y); n += len(y)
            for c, v in comps.items():
                comps_sum[c] = comps_sum.get(c, 0.0) + v * len(y)
        return tot / max(n, 1), {c: v / max(n, 1) for c, v in comps_sum.items()}

    @torch.no_grad()
    def _infer(self, loader):
        self.model.eval(); L, G, Y, S, SP = [], [], [], [], []
        for x, y, sp, _ in loader:
            out = self.model(x.to(self.device))
            L.append(self.model.predict_logits(out).float().cpu()); Y.append(y)
            G.append(out["gate"].float().cpu() if out["gate"] is not None else torch.full((len(y),), float("nan")))
            S.append(out["s"].float().cpu())
            SP.append(out["species"].float().cpu() if out["species"] is not None else torch.full((len(y), 4), float("nan")))
        return torch.cat(L), torch.cat(Y), torch.cat(G), torch.cat(S), torch.cat(SP)

    def _val_score(self):
        logits, y, g, _, _ = self._infer(self.val_loader)
        pred = logits.argmax(1).numpy()
        gm = {}
        if not torch.isnan(g).all():
            for i, st in enumerate(STAGES):
                m = y == i
                gm[f"gate_mean_{st}"] = float(g[m].mean()) if m.any() else float("nan")
        return macro_f1(y.numpy(), pred), gm

    def fit(self) -> dict:
        cfg = self.cfg
        self.train_ds = CropDataset(self.train_df, cfg, train=True, seed=self.seed)
        self.val_ds = CropDataset(self.val_df, cfg, train=False)
        self.train_loader = make_loader(self.train_ds, cfg, True, self.seed, self.weights)
        self.val_loader = make_loader(self.val_ds, cfg, False, 0)
        dtype = torch.bfloat16 if (cfg.bf16 and self.device.startswith("cuda")) else None
        fam = cfg.phases
        if fam == "single":
            self._phase(self.train_loader, self.epochs, cfg.lr, dtype, phase=1, early_stop=not cfg.fixed_budget)
        elif fam == "drw":                                   # B3 and arms derived from a B3 B*
            start2 = min(int(round(cfg.drw_start_frac * self.epochs)), self.epochs - 1)
            self._phase(self.train_loader, self.epochs, cfg.lr, dtype, phase=1, early_stop=False,
                        select_from=start2, phase2_from=start2)
        elif fam == "crt":                                   # B4 and arms derived from a B4 B*
            if self.init_from:
                sd = torch.load(self.init_from, map_location=self.device)
                missing, unexpected = self.model.load_state_dict(sd, strict=False)
                self.phase_log.append({"phase": 1, "loaded_from": self.init_from, "epochs": 0,
                                       "missing_keys": list(missing), "unexpected_keys": list(unexpected)})
            else:
                self._phase(self.train_loader, self.epochs, cfg.lr, dtype, phase=1, early_stop=True)
                self._load_best()
            self.model.freeze_backbone(); self.model.freeze_gate(); self.model.reset_head()
            w = sample_weights(self.train_df.stage_idx.to_numpy(), cfg.with_(sampler="class_balanced"), q=1.0)
            loader = make_loader(self.train_ds, cfg, True, self.seed + 1, w)
            self.best = {"epoch": -1, "score": -1.0, "phase": None}     # select within the final phase only
            self._phase(loader, cfg.crt_epochs, cfg.crt_head_lr, dtype, phase=2, head_only=True, early_stop=False,
                        epoch_offset=len(self.log))
        else:
            raise ValueError(fam)
        return dict(self.best, diverged=self.diverged)

    def _phase(self, loader, epochs, lr, dtype, phase, head_only=False, early_stop=True, select_from=0,
               phase2_from=None, epoch_offset=0):
        cfg = self.cfg
        if head_only:
            params = [p for p in self.model.head_parameters(phase2=True) if p.requires_grad]
            groups = [{"params": params, "lr": lr, "weight_decay": 0.0}]
        else:
            groups = param_groups(self.model, cfg)
        opt = torch.optim.AdamW(groups)
        base = [g["lr"] for g in opt.param_groups]
        since_best = 0; score = float("nan"); last_ep = -1
        for ep in range(epochs):
            f = lr_factor(ep, cfg.warmup_epochs if not head_only else 1, epochs)
            for g, b in zip(opt.param_groups, base):
                g["lr"] = b * f
            loss, comps = self._epoch(loader, opt, ep, dtype)
            score, gm = self._val_score()
            cur_phase = phase if phase2_from is None or ep < phase2_from else 2
            rec = {"phase": cur_phase, "epoch": ep + epoch_offset, "phase_epoch": ep, "train_loss": loss,
                   **{f"loss_{c}": v for c, v in comps.items()}, "val_macro_f1": score, "lr_factor": f, **gm}
            if self.model.boost is not None:
                rec["boost_b"] = float(self.model.boost.detach())
            self.log.append(rec); last_ep = ep
            if not math.isfinite(loss):                      # Step 10 failure rule: a diverged run is reported
                self.diverged = True
                break
            if ep >= select_from and score > self.best["score"] + cfg.min_delta:
                self.best = {"epoch": ep + epoch_offset, "score": score, "phase": cur_phase}; since_best = 0
                torch.save(self.model.state_dict(), self.run_dir / "best.pt")
            else:
                since_best += 1
            if early_stop and since_best >= cfg.patience:
                break
        if self.best["epoch"] < 0:                           # never improved above -1+delta: keep the last state
            torch.save(self.model.state_dict(), self.run_dir / "best.pt")
            self.best = {"epoch": last_ep + epoch_offset, "score": score, "phase": phase}
        self.phase_log.append({"phase": phase, "epochs": last_ep + 1, "budget": epochs, "early_stop": early_stop,
                               "select_from": select_from, "head_only": head_only})

    def _load_best(self):
        self.model.load_state_dict(torch.load(self.run_dir / "best.pt", map_location=self.device))

    # ------------------------------------------------------------------ finalize
    def finalize(self, provenance: dict | None = None) -> dict:
        from malprep.folds import get_partition
        self._load_best()
        vl, vy, _, _, _ = self._infer(self.val_loader)
        T = fit_temperature(vl, vy)
        HeldoutGuard.allowed = True
        try:
            self._test_df = get_partition(self.data_cfg, self.unit, self.r, self.k, "test")
            tl = heldout_loader(self._test_df, self.cfg)
            res = {"T": T, "best": self.best}
            for name, df, loader in (("test", self._test_df, tl), ("innerval", self.val_df, self.val_loader)):
                res[name] = self._write_preds(name, df, loader, T)
            if self.external is not None:
                el = heldout_loader(self.external, self.cfg)
                res["bbbc041"] = self._write_preds("bbbc041", self.external, el, T)
        finally:
            HeldoutGuard.allowed = False
        self._write_meta(res, provenance)
        if self.log:                                          # a resumed finalize keeps the training log
            pd.DataFrame(self.log).to_csv(self.run_dir / "log.csv", index=False, lineterminator="\n")
        return res

    def _write_preds(self, name, df, loader, T):
        logits, y, g, s, spl = self._infer(loader)
        probs = F.softmax(logits / T, 1).numpy()
        pred = logits.argmax(1).numpy()
        out = pd.DataFrame({"cell_id": df.cell_id.to_numpy(), "y_true": y.numpy(), "y_pred": pred, "T": T, "gate": g.numpy()})
        for c in range(4):
            out[f"logit_{c}"] = logits[:, c].numpy(); out[f"prob_{c}"] = probs[:, c]
        if not torch.isnan(spl).all():                        # B5 only: species head output (descriptive endpoint)
            for c in range(spl.shape[1]):
                out[f"species_logit_{c}"] = spl[:, c].numpy()
            out["species_pred"] = spl.argmax(1).numpy()
        for col in ("image_group", "image_id", "group_slide", "group_image", "species", "subset"):
            if col in df:
                out[col] = df[col].to_numpy()
        out["arm"], out["backbone"], out["unit"] = self.cfg.arm, self.cfg.backbone, self.unit
        out["repeat"], out["fold"], out["seed"] = self.r, self.k, self.seed
        out["config_sha"] = self.cfg.sha256
        out["split_sha"] = self.digests["split_sha256"]; out["manifest_sha"] = self.digests["manifest_sha256"]
        g = git_commit(); out["commit"] = g["commit"] + ("-dirty" if g["dirty"] else "")
        out.to_parquet(self.run_dir / f"preds_{name}.parquet", index=False)
        return summary(y.numpy(), pred, probs)

    def _write_meta(self, res, provenance):
        g = git_commit(); gpu = gpu_info(); vers = package_versions()
        meta = {"arm": self.cfg.arm, "backbone": self.cfg.backbone, "unit": self.unit, "repeat": self.r, "fold": self.k,
                "seed": self.seed, "config": self.cfg.__dict__, "config_sha": self.cfg.sha256, "arms_digest": self.arms_digest,
                "commit": g["commit"], "commit_dirty": g["dirty"],
                "manifest": self.digests["manifest"], "manifest_sha256": self.digests["manifest_sha256"],
                "split_file": self.digests["split_file"], "split_sha256": self.digests["split_sha256"],
                "best_epoch": self.best["epoch"], "best_phase": self.best.get("phase"), "epoch_cap": self.epochs,
                "hit_cap": self.best["epoch"] >= self.epochs - 1, "phases": self.phase_log, "schedule": self.cfg.phases,
                "diverged": self.diverged, "init_from": self.init_from,
                "innerval_macro_f1_at_best": self.best["score"], "temperature": res["T"],
                "gate_sampled_prevalence": self.gate_prev, "gate_pos_weight": self.gate_pos_weight,
                "stage_counts_inner_train": self.counts.tolist(),
                "n_cells": {"inner_train": len(self.train_df), "inner_val": len(self.val_df),
                            "test": int(len(self._test_df)) if self._test_df is not None else None,
                            "bbbc041": int(len(self.external)) if self.external is not None else None},
                "device": self.device, "gpu": gpu["gpu"], "gpu_driver": gpu["driver"], "cuda": gpu["cuda"],
                "torch": vers.get("torch"), "timm": vers.get("timm"), "pretrained_sha256": self.pretrained_sha256,
                "python": sys.version.split()[0], "platform": platform.platform(),
                "start_time": self.start_time, "end_time": now(), "elapsed_s": round(time.time() - self.t0, 1),
                "innerval_summary": {k: v for k, v in res["innerval"].items() if k != "confusion"},
                "test_summary_written": True,
                **(provenance or {})}
        with open(self.run_dir / "meta.json", "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(meta, indent=1, default=str) + "\n")
