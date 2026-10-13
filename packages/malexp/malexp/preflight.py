"""GPU preflight (Step 8): one forward/backward/inference step of every arm variant the matrix can
produce, on the training device with the training precision (bf16 autocast on CUDA). Catches device-
or precision-only failures (the boost index-put and the C-HMCNN BCE, Deviation 1) before a long run
reaches them. Random inputs; no data, no files written, no effect on any run."""
from __future__ import annotations
import torch

from .config import base_arms, derive_from_bstar
from .losses import total_loss
from .models import StageNet
from .select import TIER2, ABL


def variants(backbone, input_size, arms_dir=None):
    base = {a: c.with_(backbone=backbone, input_size=input_size, pretrained=False) for a, c in base_arms(arms_dir).items()}
    out = list(base.values())
    for b in ["B1", "B2", "B3", "B4"]:
        for arm in TIER2:
            out.append(derive_from_bstar(arm, base[b]))
        mfull = derive_from_bstar("M-full", base[b])
        neutral = mfull.with_(use_gate=False, use_boost=False, detach_gate=False, use_coordattn=False)
        for arm in ABL:
            out.append(derive_from_bstar(arm, neutral))
    return out


def run(backbone="mobilevit_xxs.cvnets_in1k", input_size=256, device=None, batch=8, arms_dir=None) -> list[tuple[str, str]]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    counts = torch.tensor([600, 40, 20, 15])
    y = torch.tensor([0, 1, 2, 3] * (batch // 4), device=device)
    sp = torch.zeros_like(y)
    x = torch.randn(len(y), 3, input_size, input_size, device=device)
    res = []
    for cfg in variants(backbone, input_size, arms_dir):
        name = f"{cfg.arm} (B*={cfg.loss}/{cfg.sampler}{'/cRT' if cfg.crt else ''})"
        m = None
        try:
            torch.manual_seed(0)
            m = StageNet(cfg).to(device).train()
            with torch.autocast(device.split(":")[0], dtype=dtype):
                out = m(x)
                loss, _ = total_loss(cfg, out, y, sp, counts, 0.9, 3.0)    # 0.9 > DRW start: DRW weights exercised
            loss.backward()
            assert torch.isfinite(loss), "non-finite loss"
            m.eval()
            with torch.no_grad(), torch.autocast(device.split(":")[0], dtype=dtype):
                logits = m.predict_logits(m(x))
            assert logits.shape == (len(y), 4) and torch.isfinite(logits.float()).all(), "bad inference logits"
            res.append((name, "ok"))
        except Exception as e:                                             # report every failure, not just the first
            res.append((name, f"FAIL {type(e).__name__}: {str(e).splitlines()[0][:160]}"))
        finally:
            del m
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    return res
