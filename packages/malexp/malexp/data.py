"""Datasets, transforms and samplers (protocol Step 7). Reads frozen partitions through malprep.folds.

The test partition is wrapped in a guard: constructing its loader before finalize() raises.

Augmentation randomness: every sample draws its own generator seeded from (run seed, the DataLoader
worker's torch seed, the sample index, a per-worker call counter). The worker seed comes from the
loader's generator and changes every epoch, so two workers never share a stream and the same run seed
gives the same augmentations (tested). The earlier version shared one generator across forked workers.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from malcore import SPECIES, RARE
from .config import ArmConfig

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# Ruifrok & Johnston H-E-DAB stain vectors (used for stain jitter in optical-density space)
_HED = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]], dtype=np.float32)
_HED_INV = np.linalg.inv(_HED)


class HeldoutAccessError(RuntimeError):
    pass


class HeldoutGuard:
    """Set by Trainer.finalize(); any attempt to build a held-out (test / external) loader before that raises."""
    allowed = False


def hed_jitter(img: np.ndarray, rng: np.random.Generator, sigma: float = 0.03) -> np.ndarray:
    od = -np.log(np.clip(img.astype(np.float32) / 255, 1e-3, 1))
    hed = od.reshape(-1, 3) @ _HED_INV
    hed = hed * rng.uniform(1 - sigma, 1 + sigma, 3) + rng.uniform(-sigma, sigma, 3)
    od2 = hed @ _HED
    return (np.exp(-od2).reshape(img.shape) * 255).clip(0, 255).astype(np.uint8)


class Transform:
    """Train: flips, 90-degree rotations, RandomResizedCrop(0.85-1, ratio 0.9-1.1), HED jitter sigma 0.03,
    brightness/contrast +-0.1, ImageNet normalisation. Eval: resize only, no TTA."""
    def __init__(self, size: int, train: bool):
        self.size, self.train = size, train

    def __call__(self, im: Image.Image, rng: np.random.Generator | None = None) -> torch.Tensor:
        if self.train:
            rng = rng if rng is not None else np.random.default_rng()
            if rng.random() < 0.5: im = im.transpose(Image.FLIP_LEFT_RIGHT)
            if rng.random() < 0.5: im = im.transpose(Image.FLIP_TOP_BOTTOM)
            k = int(rng.integers(0, 4))
            if k: im = im.rotate(90 * k)
            w, h = im.size; area = w * h * rng.uniform(0.85, 1.0); ratio = rng.uniform(0.9, 1.1)
            cw = int(round(min(w, (area * ratio) ** 0.5))); ch = int(round(min(h, (area / ratio) ** 0.5)))
            x0 = int(rng.integers(0, w - cw + 1)); y0 = int(rng.integers(0, h - ch + 1))
            im = im.crop((x0, y0, x0 + cw, y0 + ch)).resize((self.size, self.size), Image.Resampling.BILINEAR)
            a = hed_jitter(np.asarray(im), rng)
            a = a.astype(np.float32) * rng.uniform(0.9, 1.1) + 255 * rng.uniform(-0.1, 0.1)
            a = a.clip(0, 255) / 255
        else:
            if im.size != (self.size, self.size):
                im = im.resize((self.size, self.size), Image.Resampling.BILINEAR)
            a = np.asarray(im).astype(np.float32) / 255
        a = (a - MEAN) / STD
        return torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))


class CropDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: ArmConfig, train: bool, seed: int = 0):
        self.df = df.reset_index(drop=True)
        self.paths = self.df.crop_path.tolist()
        self.tf = Transform(cfg.input_size, train)
        self.train, self.seed = train, int(seed)
        self.y = self.df.stage_idx.to_numpy().astype(int)
        self.species = self.df.species.map({s: i for i, s in enumerate(SPECIES)}).fillna(-1).astype(int).to_numpy()
        self._calls = 0

    def __len__(self):
        return len(self.df)

    def _rng(self, i: int) -> np.random.Generator:
        self._calls += 1
        return np.random.default_rng([self.seed, int(torch.initial_seed() % 2 ** 32), int(i), self._calls])

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            x = self.tf(im.convert("RGB"), self._rng(i) if self.train else None)
        return x, int(self.y[i]), int(self.species[i]), i


def sample_weights(y: np.ndarray, cfg: ArmConfig, q: float | None = None) -> np.ndarray | None:
    """Per-cell sampling weight ~ n_y^(-q) for the class-balanced sampler; None for the instance sampler."""
    if cfg.sampler == "instance":
        return None
    q = cfg.sampler_q if q is None else q
    counts = np.bincount(y, minlength=4).astype(float)
    return counts[y] ** -q


def sampled_rare_prevalence(y: np.ndarray, weights: np.ndarray | None) -> float:
    """Expected share of the rare group in sampled batches (sets the gate BCE pos_weight, Step 6)."""
    if weights is None:
        return float(np.isin(y, list(RARE)).mean())
    return float(weights[np.isin(y, list(RARE))].sum() / weights.sum())


def _workers(cfg: ArmConfig) -> int:
    """Tiny (test/smoke) runs use no worker processes."""
    if cfg.backbone == "tiny":
        return 0
    return max(0, min(cfg.num_workers, (os.cpu_count() or 1) - 1))


def _winit(worker_id):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed((s + worker_id) % 2 ** 32)


def make_loader(ds: CropDataset, cfg: ArmConfig, train: bool, seed: int, weights=None, batch_size=None) -> DataLoader:
    g = torch.Generator(); g.manual_seed(seed)
    bs = batch_size or cfg.batch_size
    nw = _workers(cfg)
    if train:
        if weights is not None:
            sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(ds),
                                            replacement=True, generator=g)
            return DataLoader(ds, batch_size=bs, sampler=sampler, num_workers=nw, drop_last=True,
                              worker_init_fn=_winit, generator=g, persistent_workers=nw > 0)
        return DataLoader(ds, batch_size=bs, shuffle=True, num_workers=nw, drop_last=True,
                          worker_init_fn=_winit, generator=g, persistent_workers=nw > 0)
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, persistent_workers=nw > 0)


def heldout_loader(df: pd.DataFrame, cfg: ArmConfig) -> DataLoader:
    if not HeldoutGuard.allowed:
        raise HeldoutAccessError("held-out partition accessed before finalize()")
    return make_loader(CropDataset(df, cfg, train=False), cfg, train=False, seed=0)
