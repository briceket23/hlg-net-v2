"""Arm configuration: one YAML per arm, one boolean per component (protocol Step 6, Rules).

`ArmConfig` is immutable; its SHA-256 goes into every run's meta.json and prediction file.
The base arms B0-B4 and the tuning grids live in configs/arms/ (frozen at gate G4, digests in
configs/arms/digests.txt). `base_arms()` without a directory returns the built-in definitions,
which `export_arms()` writes out; the two must agree (tested).
"""
from __future__ import annotations
import dataclasses, hashlib, json
from dataclasses import dataclass, asdict
from pathlib import Path
import yaml

from malcore import ARMS, ARMS_BASE, ARMS_DERIVED, ARMS_ABLATION   # noqa: F401
from malcore.hashing import sha256_file, format_digests, read_digests
from malcore.io import write_text


@dataclass(frozen=True)
class ArmConfig:
    arm: str
    # backbone
    backbone: str = "mobilevit_xxs.cvnets_in1k"      # timm id; "tiny" = test-only CNN
    input_size: int = 256
    pretrained: bool = True
    use_coordattn: bool = False
    # objective / rebalancing (the B* family)
    loss: str = "ce"                 # ce | logit_adjusted | ldam_drw
    tau: float = 1.0                 # logit adjustment strength (B2)
    ldam_max_margin: float = 0.5     # B3
    ldam_scale: float = 30.0
    drw_beta: float = 0.9999
    drw_start_frac: float = 0.8      # DRW phase starts at this fraction of the epoch cap (B3 phase 2)
    sampler: str = "instance"        # instance | class_balanced
    sampler_q: float = 0.5           # class-balanced weight ~ n_c^-q
    crt: bool = False                # B4: phase 2 = frozen backbone, new head on a q = 1 sampler
    crt_epochs: int = 15
    crt_head_lr: float = 1e-3
    # auxiliary heads
    use_species_head: bool = False   # B5
    lambda_species: float = 0.3
    use_chmcnn: bool = False         # B6
    # the proposed mechanism
    use_gate: bool = False
    lambda_gate: float = 0.15
    detach_gate: bool = False
    use_boost: bool = False
    boost_max: float = 5.0
    boost_beta0: float = -2.94       # b0 = 5*sigmoid(-2.94) ~ 0.25
    use_prob: bool = False           # M-prob: log p(c) = log p(group) + log softmax within group
    lambda_prior: float = 0.0
    # schedule (shared recipe, Step 7)
    lr: float = 1e-3
    head_lr_mult: float = 10.0
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    epochs: int = 60
    fixed_budget: bool = False       # True: no early stopping (B3, B4, M-prob use fixed phase budgets)
    batch_size: int = 64
    grad_clip: float = 1.0
    bf16: bool = True
    patience: int = 15
    min_delta: float = 0.002
    num_workers: int = 4
    seed_offset: int = 0

    @property
    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    def with_(self, **kw) -> "ArmConfig":
        return dataclasses.replace(self, **kw)

    @property
    def phases(self) -> str:
        """Schedule family: 'single' | 'drw' (B3) | 'crt' (B4 and arms derived from a B4 B*)."""
        if self.crt:
            return "crt"
        if self.loss == "ldam_drw" and not (self.use_prob or self.use_chmcnn):   # the NLL / MCLoss objectives have no DRW phase
            return "drw"
        return "single"

    @staticmethod
    def load(path: str | Path) -> "ArmConfig":
        d = yaml.safe_load(open(path)) or {}
        unknown = set(d) - {f.name for f in dataclasses.fields(ArmConfig)}
        if unknown:
            raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
        return ArmConfig(**d)

    def save(self, path: str | Path) -> str:
        return write_text(yaml.safe_dump(asdict(self), sort_keys=True), path)


def _builtin_base_arms() -> dict[str, ArmConfig]:
    B = ArmConfig
    return {
        "B0": B("B0"),
        "B1": B("B1", sampler="class_balanced", sampler_q=0.5),
        "B2": B("B2", loss="logit_adjusted", tau=1.0),
        "B3": B("B3", loss="ldam_drw", fixed_budget=True),
        "B4": B("B4", crt=True, fixed_budget=True),
    }


# tuning grids (2 configs per arm per fold, protocol Step 6 / Step 9); index 0 = the smaller value
_BUILTIN_GRIDS = {
    "B0": [dict(lr=3e-4), dict(lr=1e-3)],
    "B1": [dict(lr=3e-4), dict(lr=1e-3)],
    "B2": [dict(lr=3e-4), dict(lr=1e-3)],
    "B3": [dict(lr=3e-4), dict(lr=1e-3)],
    "B4": [dict(crt_head_lr=1e-3), dict(crt_head_lr=1e-2)],
    "B5": [dict(lambda_species=0.3), dict(lambda_species=1.0)],
    "B6": [dict(lr=3e-4), dict(lr=1e-3)],
    "M-aux": [dict(lambda_gate=0.15), dict(lambda_gate=0.5)],
    "M-full": [dict(lambda_gate=0.15), dict(lambda_gate=0.5)],
    "M-prob": [dict(lambda_prior=0.0), dict(lambda_prior=1.0)],
}

# what each derived arm changes relative to the fold's B* (protocol Step 6 table)
DERIVED = {
    "B5": dict(use_species_head=True),
    "B6": dict(use_chmcnn=True),
    "M-aux": dict(use_gate=True, use_boost=False),
    "M-full": dict(use_gate=True, use_boost=True),
    "M-prob": dict(use_gate=True, use_prob=True, fixed_budget=True),
    "ABL-noaux": dict(use_gate=True, use_boost=True, lambda_gate=0.0),
    "ABL-detach": dict(use_gate=True, use_boost=True, detach_gate=True),
    "ABL-coordattn": dict(use_gate=True, use_boost=True, use_coordattn=True),
}

GRIDS = _BUILTIN_GRIDS   # alias kept for the tests / older call sites


def base_arms(arms_dir: str | Path | None = None) -> dict[str, ArmConfig]:
    """The B0-B4 definitions: from configs/arms/<arm>.yaml when a directory is given, else built in."""
    if arms_dir is None:
        return _builtin_base_arms()
    d = Path(arms_dir)
    out = {}
    for arm in ARMS_BASE:
        p = d / f"{arm}.yaml"
        if not p.exists():
            raise FileNotFoundError(f"{p} (run `python -m malexp arms-export` to write the frozen arm YAMLs)")
        out[arm] = ArmConfig.load(p)
        if out[arm].arm != arm:
            raise ValueError(f"{p}: arm field is {out[arm].arm!r}")
    return out


def grids(arms_dir: str | Path | None = None) -> dict[str, list[dict]]:
    if arms_dir is None:
        return _BUILTIN_GRIDS
    p = Path(arms_dir) / "grids.yaml"
    g = yaml.safe_load(open(p)) or {}
    for arm, cands in g.items():
        if len(cands) != 2:
            raise ValueError(f"{p}: arm {arm} must have exactly 2 configs (protocol Step 9)")
    return g


def derive_from_bstar(arm: str, bstar: ArmConfig, **kw) -> ArmConfig:
    """Arms that inherit the fold's selected B* rebalancing and learning rate (Step 6 table).

    B6 rule: 'B* rebalancing + C-HMCNN; if B* is loss-based (B2 logit adjustment, B3 LDAM), the B1
    sampler with B1's q'. The MCLoss replaces the stage objective, so a loss-based rebalancing cannot
    be kept; the class-balanced sampler (q = 0.5) carries the rebalancing instead.
    """
    base = bstar.with_(arm=arm)
    changes = dict(DERIVED[arm])
    if arm == "B6" and bstar.loss in ("logit_adjusted", "ldam_drw"):
        changes.update(loss="ce", sampler="class_balanced", sampler_q=0.5)
    changes.update(kw)
    return base.with_(**changes)


def candidates(arm: str, base: ArmConfig, arms_dir: str | Path | None = None) -> list[ArmConfig]:
    """The 2 configs of an arm; index 0 is the smaller tuned value (tie rule of Step 9)."""
    return [base.with_(arm=arm, **g) for g in grids(arms_dir)[arm]]


def export_arms(arms_dir: str | Path) -> dict[str, str]:
    """Write configs/arms/{B0..B4}.yaml, grids.yaml, derived.yaml and digests.txt (gate G4 artefacts)."""
    d = Path(arms_dir); d.mkdir(parents=True, exist_ok=True)
    shas = {}
    for arm, cfg in _builtin_base_arms().items():
        shas[f"{arm}.yaml"] = cfg.save(d / f"{arm}.yaml")
    shas["grids.yaml"] = write_text(yaml.safe_dump(_BUILTIN_GRIDS, sort_keys=True), d / "grids.yaml")
    shas["derived.yaml"] = write_text(yaml.safe_dump(DERIVED, sort_keys=True), d / "derived.yaml")
    write_text(format_digests(shas), d / "digests.txt")
    return shas


def arms_digest(arms_dir: str | Path | None) -> str | None:
    """SHA-256 over the arm YAML digests (recorded in every meta.json); None for built-in arms."""
    if arms_dir is None or not (Path(arms_dir) / "digests.txt").exists():
        return None
    return hashlib.sha256(Path(arms_dir, "digests.txt").read_bytes()).hexdigest()


def verify_arms(arms_dir: str | Path) -> list[str]:
    """Names in configs/arms whose content no longer matches digests.txt (must be empty after G4)."""
    d = Path(arms_dir)
    return [n for n, sha in read_digests(d / "digests.txt").items() if not (d / n).exists() or sha256_file(d / n) != sha]
