"""Portable file writing: LF endings, forward-slash paths, canonical JSON, so hashes agree across OSes."""
from __future__ import annotations
import json
from pathlib import Path
import yaml

from .hashing import sha256_file


def write_text(s: str, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)
    return sha256_file(path)


def write_json(obj, path: str | Path) -> str:
    """Canonical JSON (sorted keys, fixed indent) so digests are stable."""
    return write_text(json.dumps(obj, sort_keys=True, indent=1, default=str) + "\n", path)


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(df, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")
    return sha256_file(path)


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(obj, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(obj, f, sort_keys=True)
    return sha256_file(path)


def posix(p) -> str:
    """Paths are stored with forward slashes so manifests made on Windows work on Linux servers."""
    return Path(p).as_posix() if p not in (None, "") else ""
