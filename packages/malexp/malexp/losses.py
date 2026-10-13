"""Losses of protocol Step 6, each a pure function of (model output, targets, class counts).

total_loss() returns the scalar loss and a dict of its components for logging. The objective always
sees out["s_boost"], so in M-full the boost is applied before any logit adjustment; for every other
arm s_boost == s exactly (tested).
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F

from malcore import RARE, EARLY
from .config import ArmConfig


def logit_adjusted_ce(s, y, prior, tau):
    """Menon et al. 2021: CE on s + tau*log(pi). Prediction stays argmax s (no adjustment at test time)."""
    return F.cross_entropy(s + tau * torch.log(prior).unsqueeze(0), y)


def ldam_margins(counts, max_margin):
    m = 1.0 / counts.float().clamp(min=1) ** 0.25
    return m * (max_margin / m.max())


def ldam_loss(s_cos, y, margins, scale, weight=None):
    """Cao et al. 2019. s_cos already = scale * cos(theta); subtract scale*margin on the true class."""
    s_m = s_cos - scale * margins[y].unsqueeze(1) * F.one_hot(y, s_cos.shape[1]).float()
    return F.cross_entropy(s_m, y, weight=weight)


def drw_weights(counts, beta):
    eff = (1 - beta ** counts.float()) / (1 - beta)
    w = 1.0 / eff
    return w / w.sum() * len(counts)


def chmcnn_project(s):
    """C-HMCNN (Giunchiglia & Lukasiewicz 2020) with the two-level hierarchy early/late -> stage.
    Leaf probabilities p_c = sigmoid(s_c); group probability = max over its children (constraint module)."""
    p = torch.sigmoid(s)
    g_early = p[:, list(EARLY)].max(1).values
    g_late = p[:, list(RARE)].max(1).values
    return p, torch.stack([g_early, g_late], 1)


def chmcnn_loss(s, y):
    """MCLoss: BCE on leaves with the hierarchy-consistent group output.

    Computed in float32 with autocast off: BCE on probabilities is refused under CUDA autocast, and
    the max-constraint needs probabilities (not logits), so BCE-with-logits cannot be used here."""
    with torch.autocast(device_type=s.device.type, enabled=False):
        p, g = chmcnn_project(s.float())
        t_leaf = F.one_hot(y, 4).float()
        t_group = torch.stack([(y < 2).float(), (y >= 2).float()], 1)
        eps = 1e-6
        return (F.binary_cross_entropy(p.clamp(eps, 1 - eps), t_leaf)
                + F.binary_cross_entropy(g.clamp(eps, 1 - eps), t_group))


def gate_bce(gate_logit, y, pos_weight):
    t = (y >= 2).float()
    return F.binary_cross_entropy_with_logits(gate_logit, t, pos_weight=torch.as_tensor(pos_weight, device=gate_logit.device))


def total_loss(cfg: ArmConfig, out: dict, y, species, counts, epoch_frac: float, gate_pos_weight: float):
    """counts: per-stage counts in the inner training set (for priors, LDAM margins and DRW)."""
    comps = {}
    s = out["s_boost"]                       # the objective sees the boosted logits (M-full); = s otherwise
    prior = counts.float() / counts.float().sum()
    if cfg.use_prob:
        comps["nll"] = F.nll_loss(out["logp"], y)
        if cfg.lambda_prior > 0:
            comps["prior"] = cfg.lambda_prior * logit_adjusted_ce(out["logp"], y, prior.to(s.device), 1.0)
    elif cfg.use_chmcnn:
        comps["mc"] = chmcnn_loss(s, y)
    elif cfg.loss == "ce":
        comps["ce"] = F.cross_entropy(s, y)
    elif cfg.loss == "logit_adjusted":
        comps["la"] = logit_adjusted_ce(s, y, prior.to(s.device), cfg.tau)
    elif cfg.loss == "ldam_drw":
        w = drw_weights(counts, cfg.drw_beta).to(s.device) if epoch_frac >= cfg.drw_start_frac else None
        comps["ldam"] = ldam_loss(s, y, ldam_margins(counts, cfg.ldam_max_margin).to(s.device), cfg.ldam_scale, w)
    else:
        raise ValueError(cfg.loss)
    if cfg.use_species_head and species is not None:
        comps["species"] = cfg.lambda_species * F.cross_entropy(out["species"], species)
    if cfg.use_gate and cfg.lambda_gate > 0:
        comps["gate"] = cfg.lambda_gate * gate_bce(out["gate_logit"], y, gate_pos_weight)
    total = sum(comps.values())
    return total, {k: float(v.detach()) for k, v in comps.items()}
