"""Step 1 check: two runs of the same config and seed on the same machine give inner-validation
macro-F1 within 0.005. If not, the measured nondeterminism is recorded (deviations log) and becomes part
of the noise floor; above 0.02 the protocol's stop rule applies (fix it, or add outer repeats r4, r5)."""
from __future__ import annotations
import json
from pathlib import Path

from .config import ArmConfig
from .train import Trainer

TOLERANCE = 0.005
STOP_RULE = 0.02


def run(cfg: ArmConfig, data_cfg: dict, unit: str, r: int, k: int, out_dir: str | Path, device=None, max_epochs=None,
        n: int = 2) -> dict:
    out_dir = Path(out_dir); scores = []
    for i in range(n):
        tr = Trainer(cfg, data_cfg, unit, r, k, out_dir / f"rep{i}", device=device, max_epochs=max_epochs, external=None)
        best = tr.fit()
        scores.append(best["score"])
    spread = max(scores) - min(scores)
    rep = {"arm": cfg.arm, "backbone": cfg.backbone, "unit": unit, "repeat": r, "fold": k, "seed": tr.seed,
           "innerval_macro_f1": scores, "spread": spread, "within_tolerance": spread <= TOLERANCE,
           "stop_rule_triggered": spread > STOP_RULE, "device": tr.device,
           "note": "GPU runs with memory-efficient attention are seed-fixed but not bit-reproducible (PyTorch); "
                   "record the spread as part of the noise floor."}
    (out_dir / "determinism.json").write_text(json.dumps(rep, indent=1) + "\n")
    return rep
