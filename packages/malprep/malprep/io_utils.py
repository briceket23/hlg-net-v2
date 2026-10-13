"""Config loading, output folders, freeze lock and digest registry (hashing/IO live in malcore)."""
from __future__ import annotations
import json, os
from pathlib import Path

from malcore.hashing import sha256_file, sha256_bytes            # noqa: F401 (re-exported)
from malcore.io import write_text, write_json, write_csv, posix, load_yaml   # noqa: F401
from malcore.provenance import provenance as _provenance, git_commit_str as git_commit   # noqa: F401


def load_config(path: str | Path) -> dict:
    cfg = load_yaml(path)
    cfg["_path"] = str(path)
    cfg["_sha256"] = sha256_file(path)
    return cfg


def out_dir(cfg: dict, *parts: str) -> Path:
    p = Path(cfg["paths"]["out"]).joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


FROZEN = "FROZEN.json"


def frozen_path(cfg: dict) -> Path:
    return Path(cfg["paths"]["out"]) / "manifests" / FROZEN


def assert_not_frozen(cfg: dict, step: str) -> None:
    """Writers refuse to run once the data are frozen (gate G1/G2), unless MALPREP_UNFREEZE=1."""
    if frozen_path(cfg).exists() and os.environ.get("MALPREP_UNFREEZE") != "1":
        raise SystemExit(f"data are frozen ({frozen_path(cfg)}); `{step}` would overwrite frozen outputs. "
                         "Record a deviation (logs/deviations.md) and set MALPREP_UNFREEZE=1 to proceed deliberately.")


def provenance(cfg: dict) -> dict:
    return _provenance(cfg)


def append_digest(cfg: dict, name: str, sha: str) -> None:
    """One digest registry for all generated artefacts (data/manifests/digests.json)."""
    p = out_dir(cfg, "manifests") / "digests.json"
    d = json.loads(p.read_text()) if p.exists() else {}
    d[name] = sha
    write_text(json.dumps(d, sort_keys=True, indent=1) + "\n", p)
