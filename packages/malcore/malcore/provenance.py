"""Provenance: git commit + dirty flag, package versions, platform and GPU description."""
from __future__ import annotations
import datetime, platform, re, subprocess, sys


def git_commit(cwd: str | None = None) -> dict:
    """{'commit': <sha or 'no-git'>, 'dirty': bool}."""
    try:
        c = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, cwd=cwd).stdout.strip()
        d = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5, cwd=cwd).stdout.strip()
        if not re.fullmatch(r"[0-9a-f]{40}", c):            # no commits yet: rev-parse prints "HEAD"
            return {"commit": "no-git", "dirty": bool(d)}
        return {"commit": c, "dirty": bool(d)}
    except Exception:
        return {"commit": "no-git", "dirty": False}


def git_commit_str(cwd: str | None = None) -> str:
    g = git_commit(cwd)
    return g["commit"] + ("-dirty" if g["dirty"] else "")


def package_versions(names=("numpy", "pandas", "PIL", "scipy", "torch", "timm", "sklearn", "pyarrow")) -> dict:
    out = {}
    for n in names:
        try:
            m = __import__(n)
            out[n] = getattr(m, "__version__", "?")
        except Exception:
            pass
    return out


def now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def gpu_info() -> dict:
    """GPU model, CUDA version and driver (from nvidia-smi when available). All None on CPU."""
    info = {"gpu": None, "cuda": None, "driver": None}
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
    except Exception:
        pass
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            info["driver"] = r.stdout.strip().splitlines()[0]
    except Exception:
        pass
    return info


def provenance(cfg: dict | None = None) -> dict:
    """The record written into every generated artefact."""
    g = git_commit()
    rec = {"created": now(), "commit": g["commit"], "commit_dirty": g["dirty"],
           "python": sys.version.split()[0], "platform": platform.platform(), **package_versions()}
    if cfg is not None:
        rec["config"] = cfg.get("_path")
        rec["config_sha256"] = cfg.get("_sha256")
    return rec
