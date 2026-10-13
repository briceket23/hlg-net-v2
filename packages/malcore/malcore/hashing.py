"""SHA-256 helpers and digest files (`<sha>  <name>` lines, as sha256sum writes them)."""
from __future__ import annotations
import hashlib, json
from pathlib import Path


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(s: str) -> str:
    return sha256_bytes(s.encode("utf-8"))


def sha256_obj(obj) -> str:
    """Digest of a JSON-serialisable object (sorted keys), e.g. a configuration."""
    return sha256_text(json.dumps(obj, sort_keys=True, default=str))


def read_digests(path: str | Path) -> dict[str, str]:
    """Parse a digests.txt file into {name: sha}."""
    out = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        sha, _, name = line.partition("  ")
        out[name.strip()] = sha.strip()
    return out


def format_digests(d: dict[str, str]) -> str:
    return "".join(f"{v}  {k}\n" for k, v in sorted(d.items()))


def verify_digests(path: str | Path, root: str | Path | None = None) -> list[str]:
    """Return the names whose current SHA-256 differs from the recorded one (or that are missing)."""
    root = Path(root) if root else Path(path).parent
    bad = []
    for name, sha in read_digests(path).items():
        p = root / name
        if not p.exists() or sha256_file(p) != sha:
            bad.append(name)
    return bad
