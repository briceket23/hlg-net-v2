import numpy as np
import pytest

from malcore import STAGES, RARE
from malcore.hashing import sha256_text, read_digests, format_digests, verify_digests
from malcore.io import write_text, write_json, read_json
from malcore.metrics import macro_f1, balanced_accuracy, confusion, per_class, ece, brier, nll, reliability
from malcore.provenance import provenance, git_commit


def test_constants():
    assert STAGES == ["R", "T", "S", "G"] and RARE == (2, 3)


def test_digest_roundtrip(tmp_path):
    a = tmp_path / "a.txt"; sha = write_text("hello\n", a)
    assert sha == sha256_text("hello\n")
    d = tmp_path / "digests.txt"; write_text(format_digests({"a.txt": sha}), d)
    assert read_digests(d) == {"a.txt": sha} and verify_digests(d) == []
    a.write_text("changed")
    assert verify_digests(d) == ["a.txt"]


def test_json_lf(tmp_path):
    p = tmp_path / "x.json"; write_json({"b": 1, "a": [1, 2]}, p)
    assert p.read_bytes().endswith(b"\n") and b"\r" not in p.read_bytes() and read_json(p) == {"a": [1, 2], "b": 1}


def test_metrics_vs_sklearn():
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


def test_ece_reference_and_calibration_metrics():
    rng = np.random.default_rng(1)
    probs = rng.dirichlet(np.ones(4), 500); y = rng.integers(0, 4, 500)
    conf = probs.max(1); acc = probs.argmax(1) == y
    bins = np.where(conf == 1.0, 14, np.ceil(conf * 15).astype(int) - 1); ref = 0.0
    for b in range(15):
        m = bins == b
        if m.any(): ref += m.mean() * abs(acc[m].mean() - conf[m].mean())
    assert abs(ece(probs, y) - ref) < 1e-6
    assert abs(reliability(probs, y)["ece"] - ref) < 1e-6 and sum(b["n"] for b in reliability(probs, y)["bins"]) == 500
    onehot = np.eye(4)[y]
    assert abs(brier(onehot, y)) < 1e-12 and abs(nll(onehot, y)) < 1e-9


def test_provenance_fields():
    p = provenance({"_path": "c.yaml", "_sha256": "x"})
    assert {"created", "commit", "commit_dirty", "python", "platform", "config", "config_sha256"} <= set(p)
    assert isinstance(git_commit()["dirty"], bool)
