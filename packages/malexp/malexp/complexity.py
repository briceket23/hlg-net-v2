"""Step 13 descriptive endpoints per arm: parameters, MACs and CPU latency.

Latency: batch 1, 4 threads, 50 warm-up and 200 timed forward passes, median and IQR, on one named CPU.
MACs = FLOPs / 2 counted with torch.utils.flop_counter over every aten op (attention included)."""
from __future__ import annotations
import platform, statistics, time
import numpy as np
import torch

from .config import ArmConfig
from .models import StageNet


def count_macs(model: torch.nn.Module, input_size: int) -> int:
    from torch.utils.flop_counter import FlopCounterMode
    x = torch.randn(1, 3, input_size, input_size)
    with FlopCounterMode(display=False) as fc:
        model(x)
    return int(fc.get_total_flops() // 2)


def cpu_latency_ms(model: torch.nn.Module, input_size: int, threads: int = 4, warmup: int = 50, timed: int = 200) -> dict:
    torch.set_num_threads(threads)
    x = torch.randn(1, 3, input_size, input_size)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        t = []
        for _ in range(timed):
            t0 = time.perf_counter(); model(x); t.append((time.perf_counter() - t0) * 1000)
    q = np.percentile(t, [25, 50, 75])
    return {"median_ms": float(q[1]), "iqr_ms": [float(q[0]), float(q[2])], "threads": threads, "warmup": warmup, "timed": timed}


def measure(cfg: ArmConfig, warmup: int = 50, timed: int = 200) -> dict:
    model = StageNet(cfg.with_(pretrained=False)).eval()
    n_params = sum(p.numel() for p in model.parameters())
    return {"arm": cfg.arm, "backbone": cfg.backbone, "input_size": cfg.input_size, "params": n_params,
            "params_M": round(n_params / 1e6, 3), "macs": count_macs(model, cfg.input_size),
            "cpu": platform.processor() or platform.machine(), "torch": torch.__version__,
            **cpu_latency_ms(model, cfg.input_size, warmup=warmup, timed=timed)}
