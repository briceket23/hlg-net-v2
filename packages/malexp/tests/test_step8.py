"""Protocol Step 8 code-verification tests. Run: python -m pytest -q packages/malexp/tests  (CPU, ~2 min)."""
import inspect, json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from malexp.config import ArmConfig, base_arms, derive_from_bstar, GRIDS, export_arms, verify_arms, candidates
from malexp.models import StageNet
from malexp.losses import total_loss, logit_adjusted_ce, ldam_loss, ldam_margins, chmcnn_project, drw_weights, chmcnn_loss
from malexp.metrics import macro_f1, balanced_accuracy, per_class, ece, confusion
from malexp.data import (sample_weights, sampled_rare_prevalence, HeldoutGuard, HeldoutAccessError, heldout_loader,
                         CropDataset, make_loader)
from malexp.select import pick, pick_bstar, SelectionLog, arm_order, ablation_plan

torch.manual_seed(0)
TINY = ArmConfig("T", backbone="tiny", input_size=64, pretrained=False, num_workers=0, batch_size=8)
ALL_ARMS = {"B0", "B1", "B2", "B3", "B4", "B5", "B6", "M-aux", "M-full", "M-prob", "ABL-noaux", "ABL-detach", "ABL-coordattn"}


def _batch(n=16, size=64):
    g = torch.Generator().manual_seed(0)
    return torch.randn(n, 3, size, size, generator=g), torch.randint(0, 4, (n,), generator=g), torch.randint(0, 4, (n,), generator=g)


# ---------------------------------------------------------------- metrics (vs scikit-learn)
def test_metrics_match_sklearn():
    from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support
    rng = np.random.default_rng(0)
    for _ in range(100):
        n = int(rng.integers(5, 60)); y = rng.integers(0, 4, n); p = rng.integers(0, 4, n)
        if rng.random() < 0.3:
            y = rng.integers(0, 2, n); p = rng.integers(0, 3, n)
        assert abs(macro_f1(y, p) - f1_score(y, p, labels=[0, 1, 2, 3], average="macro", zero_division=0)) < 1e-9
        assert (confusion(y, p) == confusion_matrix(y, p, labels=[0, 1, 2, 3])).all()
        pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=[0, 1, 2, 3], zero_division=0)
        r = per_class(y, p)
        assert np.allclose(r["precision"], pr) and np.allclose(r["recall"], rc) and np.allclose(r["f1"], f1)
        if len(set(y)) == 4:
            assert abs(balanced_accuracy(y, p) - balanced_accuracy_score(y, p)) < 1e-9


def test_ece_reference():
    rng = np.random.default_rng(1)
    probs = rng.dirichlet(np.ones(4), 500); y = rng.integers(0, 4, 500)
    conf = probs.max(1); acc = probs.argmax(1) == y
    bins = np.where(conf == 1.0, 14, np.ceil(conf * 15).astype(int) - 1); ref = 0.0
    for b in range(15):
        m = bins == b
        if m.any(): ref += m.mean() * abs(acc[m].mean() - conf[m].mean())
    assert abs(ece(probs, y) - ref) < 1e-6


# ---------------------------------------------------------------- flags
def test_flags_toggle_exactly_one_thing():
    base = TINY
    names = lambda m: {n for n, _ in m.named_parameters()}
    n0 = names(StageNet(base))
    assert names(StageNet(base.with_(use_gate=True))) - n0 == {f"gate.{i}.{w}" for i in (0, 3) for w in ("weight", "bias")}
    assert names(StageNet(base.with_(use_gate=True, use_boost=True))) - names(StageNet(base.with_(use_gate=True))) == {"boost_beta"}
    assert names(StageNet(base.with_(use_species_head=True))) - n0 == {"species_head.weight", "species_head.bias"}
    assert names(StageNet(base.with_(detach_gate=True, use_gate=True))) == names(StageNet(base.with_(use_gate=True)))
    assert names(StageNet(base.with_(use_chmcnn=True))) == n0            # C-HMCNN changes the loss only
    assert all(n.startswith("coordattn.") for n in names(StageNet(base.with_(use_coordattn=True))) - n0)
    x, y, sp = _batch()
    for flag, key in [("use_species_head", "species"), ("use_gate", "gate"), ("use_chmcnn", "mc"), ("use_prob", "nll")]:
        kw = {flag: True}
        if flag == "use_prob": kw["use_gate"] = True
        cfg = base.with_(**kw)
        out = StageNet(cfg)(x)
        _, comps = total_loss(cfg, out, y, sp, torch.tensor([8, 4, 2, 2]), 0.0, 1.0)
        assert key in comps, (flag, comps)


def test_boost_off_and_zero_boost_equivalence():
    x, y, _ = _batch()
    m_aux = StageNet(TINY.with_(use_gate=True, use_boost=False))
    out = m_aux(x); assert torch.equal(out["s_boost"], out["s"])
    m_full = StageNet(TINY.with_(use_gate=True, use_boost=True)); m_full.load_state_dict(m_aux.state_dict(), strict=False)
    with torch.no_grad():
        m_full.boost_beta.fill_(-1e4)                                     # b -> 0
    o2 = m_full(x)
    assert torch.allclose(o2["s_boost"], out["s"], atol=1e-6) and torch.allclose(o2["s"], out["s"])
    with torch.no_grad():
        m_full.boost_beta.fill_(0.0)
    o3 = m_full(x); d = o3["s_boost"] - o3["s"]
    assert torch.allclose(d[:, :2], torch.zeros_like(d[:, :2])) and torch.allclose(d[:, 2], d[:, 3])
    assert torch.allclose(d[:, 2], 2.5 * o3["gate"])                      # b = 5*sigmoid(0)


def test_boost_initial_value_and_applied_before_logit_adjustment():
    m = StageNet(TINY.with_(use_gate=True, use_boost=True))
    assert abs(float(m.boost.detach()) - 0.25) < 0.01
    src = inspect.getsource(total_loss)
    assert 's = out["s_boost"]' in src and src.index('s = out["s_boost"]') < src.index("logit_adjusted_ce(s")


def test_gate_gradient_reaches_backbone_only_when_not_detached():
    x, y, _ = _batch()
    for detach in (False, True):
        cfg = TINY.with_(use_gate=True, detach_gate=detach)
        m = StageNet(cfg); out = m(x)
        loss = F.binary_cross_entropy_with_logits(out["gate_logit"], (y >= 2).float())
        m.zero_grad(); loss.backward()
        g = sum(float(p.grad.abs().sum()) for p in m.backbone.parameters() if p.grad is not None)
        assert (g == 0) == detach, (detach, g)


def test_crt_phase2_freezes_gate_and_boost_but_trains_heads():
    m = StageNet(TINY.with_(use_gate=True, use_boost=True, use_species_head=True, use_coordattn=True))
    m.freeze_backbone(); m.freeze_gate()
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert trainable == {"head.weight", "head.bias", "species_head.weight", "species_head.bias"}
    ids = {id(p) for p in m.head_parameters(phase2=True)}
    assert ids == {id(p) for n, p in m.named_parameters() if n in trainable}


# ---------------------------------------------------------------- losses
def test_logit_adjust_tau0_is_ce():
    s = torch.randn(10, 4); y = torch.randint(0, 4, (10,)); prior = torch.tensor([.7, .2, .05, .05])
    assert torch.allclose(logit_adjusted_ce(s, y, prior, 0.0), F.cross_entropy(s, y))
    assert not torch.allclose(logit_adjusted_ce(s, y, prior, 1.0), F.cross_entropy(s, y))


def test_ldam_zero_margin_is_scaled_cosine_ce():
    s = torch.randn(10, 4); y = torch.randint(0, 4, (10,))
    assert torch.allclose(ldam_loss(s, y, torch.zeros(4), 30.0), F.cross_entropy(s, y))
    m = ldam_margins(torch.tensor([1000, 100, 10, 10]), 0.5)
    assert abs(float(m.max()) - 0.5) < 1e-6 and m[0] < m[1] < m[2] and m[2] == m[3]
    w = drw_weights(torch.tensor([1000, 100, 10, 10]), 0.9999); assert w[2] > w[0] and abs(float(w.sum()) - 4) < 1e-5


def test_shift_invariance():
    """Adding one constant to all four logits changes neither CE nor argmax: the old suppression term was a no-op."""
    s = torch.randn(32, 4); y = torch.randint(0, 4, (32,)); c = torch.randn(32, 1) * 5
    assert torch.allclose(F.cross_entropy(s + c, y), F.cross_entropy(s, y), atol=1e-5)
    assert torch.equal((s + c).argmax(1), s.argmax(1))


def test_chmcnn_group_consistency_and_float32_loss():
    s = torch.randn(20, 4); p, g = chmcnn_project(s)
    assert torch.all(g[:, 0] >= p[:, :2].max(1).values - 1e-7) and torch.all(g[:, 1] >= p[:, 2:].max(1).values - 1e-7)
    torch.manual_seed(0)
    s = torch.randn(16, 4); y = torch.randint(0, 4, (16,))
    p, g = chmcnn_project(s); eps = 1e-6
    ref = (F.binary_cross_entropy(p.clamp(eps, 1 - eps), F.one_hot(y, 4).float())
           + F.binary_cross_entropy(g.clamp(eps, 1 - eps), torch.stack([(y < 2).float(), (y >= 2).float()], 1)))
    assert torch.allclose(chmcnn_loss(s, y), ref)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = chmcnn_loss(s.bfloat16(), y)
    assert out.dtype == torch.float32


def test_mprob_is_a_distribution():
    x, _, _ = _batch()
    out = StageNet(TINY.with_(use_gate=True, use_prob=True))(x)
    assert torch.allclose(out["logp"].exp().sum(1), torch.ones(len(x)), atol=1e-5)


# ---------------------------------------------------------------- sampler
def test_sampler_frequencies():
    from torch.utils.data import WeightedRandomSampler
    y = np.array([0] * 900 + [1] * 60 + [2] * 25 + [3] * 15)
    w = sample_weights(y, TINY.with_(sampler="class_balanced", sampler_q=1.0))
    target = w.reshape(-1) / w.sum(); exp = np.bincount(y, weights=target, minlength=4)
    g = torch.Generator().manual_seed(0)
    draws = list(WeightedRandomSampler(torch.as_tensor(w), 20000, replacement=True, generator=g))
    emp = np.bincount(y[draws], minlength=4) / 20000
    assert np.abs(emp - exp).max() < 0.02 and np.allclose(exp, 0.25, atol=1e-9)
    assert abs(sampled_rare_prevalence(y, w) - 0.5) < 1e-9 and abs(sampled_rare_prevalence(y, None) - 0.04) < 1e-9


# ---------------------------------------------------------------- configuration and selection rules
def test_selection_rules():
    c = TINY
    assert pick("B0", [(0, 0.70, c), (1, 0.703, c)])[0] == 0           # tie -> lower index (smaller lr)
    assert pick("B0", [(0, 0.70, c), (1, 0.71, c)])[0] == 1
    assert pick_bstar({"B1": 0.70, "B2": 0.70, "B3": 0.704, "B4": 0.60}) == "B2"   # tie -> simplest
    assert pick_bstar({"B1": 0.70, "B2": 0.70, "B3": 0.71, "B4": 0.60}) == "B3"
    d = derive_from_bstar("M-full", base_arms()["B2"].with_(lr=3e-4))
    assert d.loss == "logit_adjusted" and d.lr == 3e-4 and d.use_gate and d.use_boost
    assert all(len(v) == 2 for v in GRIDS.values())
    assert all(cands[0] != cands[1] for cands in [candidates(a, base_arms()["B0"]) for a in GRIDS])


def test_b6_rule_and_fixed_budgets():
    A = base_arms()
    assert A["B3"].fixed_budget and A["B4"].fixed_budget and not A["B0"].fixed_budget
    assert A["B3"].phases == "drw" and A["B4"].phases == "crt" and A["B1"].phases == "single"
    b6 = derive_from_bstar("B6", A["B2"]); assert b6.loss == "ce" and b6.sampler == "class_balanced" and b6.sampler_q == 0.5
    b6 = derive_from_bstar("B6", A["B3"]); assert b6.loss == "ce" and b6.sampler == "class_balanced" and b6.phases == "single"
    b6 = derive_from_bstar("B6", A["B1"]); assert b6.sampler == "class_balanced" and b6.loss == "ce"
    b6 = derive_from_bstar("B6", A["B4"]); assert b6.crt and b6.use_chmcnn
    assert derive_from_bstar("M-prob", A["B2"]).fixed_budget
    assert derive_from_bstar("M-prob", A["B3"]).phases == "single"       # the NLL objective has no DRW phase


def test_arm_order_and_ablation_plan():
    o = arm_order(["B0", "B1", "B2", "B3", "B4"], 1, 0)
    assert o[0] == "B0" and sorted(o) == ["B0", "B1", "B2", "B3", "B4"] and o == arm_order(["B0", "B1", "B2", "B3", "B4"], 1, 0)
    mfull = derive_from_bstar("M-full", base_arms()["B1"]).with_(lambda_gate=0.5)
    plan = {j["arm"]: j["cfg"] for j in ablation_plan("slide", 1, 0, mfull)}
    assert plan["ABL-noaux"].lambda_gate == 0.0 and plan["ABL-noaux"].use_boost
    assert plan["ABL-detach"].detach_gate and plan["ABL-detach"].lambda_gate == 0.5
    assert plan["ABL-coordattn"].use_coordattn and not plan["ABL-coordattn"].detach_gate


def test_arms_export_matches_builtin(tmp_path):
    export_arms(tmp_path)
    assert verify_arms(tmp_path) == []
    assert base_arms(tmp_path) == base_arms()
    (tmp_path / "B0.yaml").write_text((tmp_path / "B0.yaml").read_text() + "\n# edited\n")
    assert verify_arms(tmp_path) == ["B0.yaml"]


def test_selection_log_is_idempotent(tmp_path):
    log = SelectionLog(tmp_path / "sel.csv")
    row = dict(unit="slide", backbone="tiny", repeat=1, fold=0, arm="B0", cfg_idx=0, selected=False, inner_val_macro_f1=0.5)
    log.add(**row); log.add(**dict(row, inner_val_macro_f1=0.6)); log.add(**dict(row, cfg_idx=1))
    df = pd.read_csv(tmp_path / "sel.csv")
    assert len(df) == 2 and float(df[df.cfg_idx == 0].inner_val_macro_f1.iloc[0]) == 0.6


# ---------------------------------------------------------------- data: guard, augmentation determinism
def test_test_access_guard(toy):
    from malprep.folds import get_partition
    df = get_partition(toy, "slide", 1, 0, "test")
    HeldoutGuard.allowed = False
    with pytest.raises(HeldoutAccessError):
        heldout_loader(df, TINY)


def test_augmentation_streams_are_per_sample_and_reproducible(toy):
    from malprep.folds import get_partition
    df = get_partition(toy, "slide", 1, 0, "inner_train").head(16); n = len(df)
    cfg = TINY.with_(input_size=32)
    def epoch(seed):
        torch.manual_seed(seed)
        ds = CropDataset(df, cfg, train=True, seed=seed)
        return torch.stack([ds[i][0] for i in range(n)])
    a, b, c = epoch(3), epoch(3), epoch(4)
    assert torch.equal(a, b) and not torch.equal(a, c)
    ds = CropDataset(df, cfg, train=True, seed=3)
    assert not torch.equal(ds[0][0], ds[0][0])          # a second pass over the same sample draws new augmentation
    assert torch.equal(CropDataset(df, cfg, train=False)[0][0], CropDataset(df, cfg, train=False)[0][0])


def test_dataset_and_loader_pickle_for_spawn_workers(toy):
    """Windows DataLoader workers start with `spawn`: the dataset (and its transform) must pickle."""
    import pickle
    from malprep.folds import get_partition
    df = get_partition(toy, "slide", 1, 0, "inner_train").head(8)
    ds = CropDataset(df, TINY.with_(input_size=32), train=True, seed=1)
    ds2 = pickle.loads(pickle.dumps(ds))
    assert len(ds2) == len(ds) and ds2[0][0].shape == ds[0][0].shape


def test_training_code_never_touches_bbbc041():
    """Step 12 check: no BBBC041 path is read by training or tuning code."""
    import malexp.data, malexp.losses, malexp.models, malexp.select, malexp.config, malexp.train
    for mod in (malexp.data, malexp.losses, malexp.models, malexp.select, malexp.config):
        assert "bbbc" not in inspect.getsource(mod).lower(), mod.__name__
    from malexp.train import Trainer
    for fn in (Trainer.fit, Trainer._phase, Trainer._epoch, Trainer._val_score):
        assert "external" not in inspect.getsource(fn) and "bbbc" not in inspect.getsource(fn).lower()


# ---------------------------------------------------------------- overfit + nested smoke on toy data
def test_overfit_every_arm(toy):
    """64 training crops reach > 0.95 training accuracy within 100 steps for every arm (Step 8)."""
    from malprep.folds import get_partition
    df = get_partition(toy, "slide", 1, 0, "inner_train").head(64)
    ds = CropDataset(df, TINY.with_(input_size=64), train=False)
    x = torch.stack([ds[i][0] for i in range(len(ds))]); y = torch.as_tensor(ds.y); sp = torch.as_tensor(ds.species)
    counts = torch.bincount(y, minlength=4)
    arms = {**base_arms(), **{a: derive_from_bstar(a, base_arms()["B2"]) for a in ["B5", "B6", "M-aux", "M-full", "M-prob"]}}
    for name, cfg in arms.items():
        cfg = cfg.with_(backbone="tiny", input_size=64, pretrained=False)
        torch.manual_seed(0); m = StageNet(cfg); opt = torch.optim.Adam(m.parameters(), 3e-3)
        for step in range(100):
            out = m(x); loss, _ = total_loss(cfg, out, y, sp, counts, step / 100, 1.0)
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval(); acc = float((m.predict_logits(m(x)).argmax(1) == y).float().mean())
        assert acc > 0.95, (name, acc)


def test_smoke_full_fold(toy, tmp_path):
    """3-epoch nested run of all arms on one fold: fit -> select -> finalize -> prediction files (Steps 9-11)."""
    from malexp.runner import FoldRunner, load_external
    from malprep.folds import get_partition
    ext = load_external(toy)
    runs = tmp_path / "runs"
    fr = FoldRunner(toy, "slide", runs, "tiny", 64, tiers=(1, 2, 3), with_prob=True, max_epochs=3, device="cpu", external=ext)
    res = fr.run_fold(1, 0)
    assert res["bstar"] in {"B1", "B2", "B3", "B4"}
    metas = {json.loads(m.read_text())["arm"]: json.loads(m.read_text()) for m in runs.rglob("meta.json")}
    assert ALL_ARMS <= set(metas)
    # Step 1 meta.json fields
    need = {"commit", "commit_dirty", "config_sha", "manifest_sha256", "split_sha256", "arm", "backbone", "repeat", "fold",
            "seed", "gpu", "gpu_driver", "start_time", "end_time", "best_epoch", "innerval_macro_f1_at_best",
            "selection_record", "schedule", "phases"}
    assert need <= set(metas["M-full"]), need - set(metas["M-full"])
    # B4: phase 1 loaded from the fold's selected B0; checkpoint selected in phase 2; fixed budget
    assert metas["B4"]["init_from"] and metas["B4"]["init_from"].startswith(str(runs / "B0"))
    assert metas["B4"]["best_phase"] == 2 and metas["B4"]["phases"][0]["loaded_from"]
    # B3: DRW phase 2 and selection inside it; no early stopping
    assert metas["B3"]["schedule"] == "drw" and metas["B3"]["best_phase"] == 2 and metas["B3"]["phases"][0]["early_stop"] is False
    assert metas["M-prob"]["phases"][0]["early_stop"] is False and metas["B0"]["phases"][0]["early_stop"] is True
    # selection log: complete, idempotent on resume, names B*
    sel = pd.read_csv(runs / "selection.csv")
    assert (sel[sel.arm == "B4"].bstar.dropna() != "").any() and "backbone" in sel
    n_rows = len(sel)
    fr.run_fold(1, 0)                                                     # resume: nothing retrained, no duplicate rows
    assert len(pd.read_csv(runs / "selection.csv")) == n_rows
    # prediction files: one row per test cell, probabilities sum to 1, provenance columns, no test preds for non-selected
    HeldoutGuard.allowed = True; n_test = len(get_partition(toy, "slide", 1, 0, "test")); HeldoutGuard.allowed = False
    cols = {"cell_id", "image_group", "species", "y_true", "y_pred", "T", "gate", "arm", "backbone", "repeat", "fold",
            "seed", "config_sha", "split_sha", "manifest_sha", "commit"} | {f"logit_{c}" for c in range(4)} | {f"prob_{c}" for c in range(4)}
    for arm, m in metas.items():
        d = Path(m["run_id"]); p = pd.read_parquet(runs / d / "preds_test.parquet")
        assert len(p) == n_test and np.allclose(p[[f"prob_{c}" for c in range(4)]].sum(axis=1), 1, atol=1e-5)
        assert cols <= set(p.columns), cols - set(p.columns)
        assert (runs / d / "preds_bbbc041.parquet").exists() and (runs / d / "preds_innerval.parquet").exists()
        assert p.split_sha.iloc[0] == m["split_sha256"] and p.manifest_sha.iloc[0] == m["manifest_sha256"]
    assert "species_pred" in pd.read_parquet(runs / metas["B5"]["run_id"] / "preds_test.parquet")
    non_sel = [d for d in runs.rglob("c*") if d.is_dir() and not (d / "meta.json").exists()]
    assert all(not (d / "preds_test.parquet").exists() for d in non_sel) and len(non_sel) > 0
    log = pd.read_csv(runs / metas["B0"]["run_id"] / "log.csv")
    assert "val_macro_f1" in log and not any(c.startswith("test") for c in log.columns)
    # queue + status
    from malexp.queue import write_queue, status
    write_queue(runs, unit="slide", backbone="tiny", repeats=(1,), folds=(0,), tiers=(1, 2, 3), with_prob=True)
    st = status(runs)
    assert st.finalized.sum() == len(ALL_ARMS) and st.trained.all()


def test_digest_mismatch_blocks_training(toy, tmp_path):
    from malexp.train import Trainer
    dig = Path(toy["paths"]["splits"]) / "slide" / "digests.txt"; orig = dig.read_text()
    try:
        dig.write_text(orig.replace(orig[:4], "0000" if orig[:4] != "0000" else "1111", 1))
        with pytest.raises(RuntimeError):
            Trainer(TINY, toy, "slide", 1, 0, tmp_path / "x", device="cpu", max_epochs=1)
    finally:
        dig.write_text(orig)


def test_preflight_tiny_all_variants():
    from malexp.preflight import run
    res = run("tiny", 64, "cpu", batch=8)
    assert len(res) == 37 and all(st == "ok" for _, st in res), [r for r in res if r[1] != "ok"]


def test_complexity_tiny():
    from malexp.complexity import measure
    r = measure(TINY.with_(arm="B0"), warmup=2, timed=5)
    assert r["params"] > 0 and r["macs"] > 0 and r["median_ms"] > 0


def test_determinism_tiny(toy, tmp_path):
    from malexp.determinism import run
    rep = run(TINY.with_(arm="B0"), toy, "slide", 1, 0, tmp_path, device="cpu", max_epochs=2)
    assert rep["within_tolerance"] and (tmp_path / "determinism.json").exists()
