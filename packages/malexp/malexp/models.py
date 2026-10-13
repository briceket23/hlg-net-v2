"""Network: backbone -> pooled feature z -> stage head (4 logits) [+ species head] [+ gate -> boost].

Forward returns a dict so every quantity the protocol logs is available:
  z          pooled features
  s          raw stage logits
  s_boost    boosted logits s' (equals s when use_boost is False)        -> prediction = argmax
  gate       g = sigmoid(h(z)) in [0, 1], or None
  gate_logit h(z)
  species    species logits or None
  logp       M-prob composed log-probabilities or None

Boost (Step 6): s'_c = s_c + b*g for c in L = {schizont, gametocyte}, b = boost_max * sigmoid(beta),
beta_0 = -2.94 so b_0 ~ 0.25. It is applied before any logit adjustment (losses.py).
"""
from __future__ import annotations
import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F

from malcore import RARE, EARLY
from .config import ArmConfig


class CoordAttention(nn.Module):
    """Hou et al. 2021, applied once on the backbone's last feature map (ablation only)."""
    def __init__(self, c: int, reduction: int = 8):
        super().__init__()
        m = max(8, c // reduction)
        self.conv1 = nn.Conv2d(c, m, 1); self.bn = nn.BatchNorm2d(m); self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(m, c, 1); self.conv_w = nn.Conv2d(m, c, 1)

    def forward(self, x):
        n, c, h, w = x.shape
        xh = x.mean(3, keepdim=True)                       # n,c,h,1
        xw = x.mean(2, keepdim=True).permute(0, 1, 3, 2)   # n,c,w,1
        y = self.act(self.bn(self.conv1(torch.cat([xh, xw], 2))))
        yh, yw = torch.split(y, [h, w], 2)
        ah = torch.sigmoid(self.conv_h(yh)); aw = torch.sigmoid(self.conv_w(yw.permute(0, 1, 3, 2)))
        return x * ah * aw


class TinyBackbone(nn.Module):
    """Test-only backbone (no download, runs on CPU in seconds)."""
    def __init__(self, c=16):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, c, 5, 2, 2), nn.ReLU(), nn.Conv2d(c, 2 * c, 3, 2, 1), nn.ReLU(),
                                 nn.Conv2d(2 * c, 2 * c, 3, 2, 1), nn.ReLU())
        self.num_features = 2 * c

    def forward_features(self, x):
        return self.net(x)


def make_backbone(cfg: ArmConfig):
    if cfg.backbone == "tiny":
        return TinyBackbone()
    import timm
    return timm.create_model(cfg.backbone, pretrained=cfg.pretrained, num_classes=0, global_pool="")


def state_sha256(module: nn.Module) -> str:
    """SHA-256 of a module's parameters (records the pretrained weights actually loaded)."""
    h = hashlib.sha256()
    for name, t in sorted(module.state_dict().items()):
        h.update(name.encode()); h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class StageNet(nn.Module):
    def __init__(self, cfg: ArmConfig, n_species: int = 4):
        super().__init__()
        self.cfg = cfg
        self.backbone = make_backbone(cfg)
        d = self.backbone.num_features
        self.coordattn = CoordAttention(d) if cfg.use_coordattn else nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.d = d
        self.ldam = cfg.loss == "ldam_drw"
        if self.ldam:                                   # normalised cosine classifier (Cao et al. 2019)
            self.head = nn.Parameter(torch.empty(4, d)); nn.init.kaiming_uniform_(self.head, a=5 ** 0.5)
        else:
            self.head = nn.Linear(d, 4); nn.init.zeros_(self.head.bias)
        self.species_head = nn.Linear(d, n_species) if cfg.use_species_head else None
        self.gate = (nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1))
                     if cfg.use_gate else None)
        self.boost_beta = nn.Parameter(torch.tensor(float(cfg.boost_beta0))) if cfg.use_boost else None
        self.register_buffer("rare_mask", torch.zeros(4).index_fill_(0, torch.tensor(list(RARE)), 1.0), persistent=False)

    @property
    def boost(self) -> torch.Tensor | None:
        return None if self.boost_beta is None else self.cfg.boost_max * torch.sigmoid(self.boost_beta)

    def stage_head_parameters(self) -> list:
        return [self.head] if self.ldam else list(self.head.parameters())

    def head_parameters(self, phase2: bool = False) -> list:
        """Parameters trained at the head learning rate. In a cRT phase 2 the gate and the boost are
        frozen (protocol Step 6: 'add the gate in phase 1 and keep it frozen in phase 2'); the
        Coordinate Attention block belongs to the feature path and is frozen with the backbone."""
        ps = self.stage_head_parameters()
        if self.species_head is not None:
            ps += list(self.species_head.parameters())
        if not phase2:
            if not isinstance(self.coordattn, nn.Identity):
                ps += list(self.coordattn.parameters())
            if self.gate is not None:
                ps += list(self.gate.parameters())
            if self.boost_beta is not None:
                ps.append(self.boost_beta)
        return ps

    def features(self, x):
        f = self.backbone.forward_features(x)
        f = self.coordattn(f)
        return self.pool(f).flatten(1)

    def forward(self, x):
        z = self.features(x)
        if self.ldam:
            s = self.cfg.ldam_scale * F.normalize(z, dim=1) @ F.normalize(self.head, dim=1).t()
        else:
            s = self.head(z)
        out = {"z": z, "s": s, "gate": None, "gate_logit": None, "species": None, "logp": None}
        if self.species_head is not None:
            out["species"] = self.species_head(z)
        if self.gate is not None:
            zg = z.detach() if self.cfg.detach_gate else z
            out["gate_logit"] = self.gate(zg).squeeze(1)
            out["gate"] = torch.sigmoid(out["gate_logit"])
        s_b = s
        if self.boost_beta is not None:
            # mask product instead of index_put: broadcasting index assignment fails on CUDA under autocast
            s_b = s + (self.boost * out["gate"]).unsqueeze(1).to(s.dtype) * self.rare_mask.to(s.dtype).unsqueeze(0)
        out["s_boost"] = s_b
        if self.cfg.use_prob:
            g = out["gate"].clamp(1e-6, 1 - 1e-6)
            lp_e = torch.log(1 - g).unsqueeze(1) + F.log_softmax(s[:, list(EARLY)], 1)
            lp_r = torch.log(g).unsqueeze(1) + F.log_softmax(s[:, list(RARE)], 1)
            cols = {c: lp_e[:, i] for i, c in enumerate(EARLY)}; cols.update({c: lp_r[:, i] for i, c in enumerate(RARE)})
            out["logp"] = torch.stack([cols[c] for c in range(s.shape[1])], 1)
        return out

    def predict_logits(self, out: dict) -> torch.Tensor:
        """The logits used for argmax prediction (and for temperature scaling)."""
        return out["logp"] if self.cfg.use_prob else out["s_boost"]

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        if not isinstance(self.coordattn, nn.Identity):
            for p in self.coordattn.parameters():
                p.requires_grad_(False)

    def freeze_gate(self):
        if self.gate is not None:
            for p in self.gate.parameters():
                p.requires_grad_(False)
        if self.boost_beta is not None:
            self.boost_beta.requires_grad_(False)

    def reset_head(self):
        if self.ldam:
            nn.init.kaiming_uniform_(self.head, a=5 ** 0.5)
        else:
            self.head.reset_parameters(); nn.init.zeros_(self.head.bias)
