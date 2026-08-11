"""Tests for the head-only cached-embedding experiment.

Locks the numerically-verifiable contract of ``scripts/train_head_only_cached.py``
and ``scripts/build_embedding_cache.py``: cache fingerprints, train/val
separation, manifest alignment, cached-logit reproduction of the full model,
head init from the checkpoint, head-only parameter counts, deterministic
verdict classification, boundary groups, and the integrated artifacts (sanity
gate, report, deployable checkpoint with bit-identical backbone).

The experiment optimizes ONLY the final ``Linear(1280, 29)`` on frozen cached
embeddings. No test here asserts an outcome — the verdict is allowed to be any
of the four; the tests assert the experiment was run correctly.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.build_embedding_cache import (
    file_sha256,
    seeded_train_transform,
    state_dict_sha256,
    tensor_sha256,
)
from scripts.train_head_only_cached import (
    DEFAULT_HEAD_OUT,
    DEFAULT_MANIFEST,
    DEFAULT_OUT_JSON,
    DEFAULT_OUT_MD,
    FEATURE_DIM,
    N_CLASSES,
    SCHEMA,
    build_deployable_checkpoint,
    boundary_groups,
    cached_logits,
    classify,
    cosine_warmup_lr,
    evaluate_cm,
    init_linear,
    s_e_metrics,
    train_candidate,
    verify_cache_against_manifest,
    verify_cache_pairs,
)
from src.data import CLASSES, Sample, SplitManifest

CHECKPOINT = Path("checkpoints/efficientnet_b0_targeted.pt")
BASELINE = Path("docs/reports/dev_val_targeted_current.json")
TRAIN_CACHE = Path("data/embedding_cache/targeted_train_embeddings.pt")
VAL_CACHE = Path("data/embedding_cache/targeted_val_embeddings.pt")
HEAD_OUT = Path(DEFAULT_HEAD_OUT)
DEPLOY = Path("checkpoints/efficientnet_b0_head_ft.pt")
OUT_JSON = Path(DEFAULT_OUT_JSON)


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _fake_cache(n: int, rng, cls_names: list[str] | None = None) -> dict:
    if cls_names is None:
        cls_names = [CLASSES[i % N_CLASSES] for i in range(n)]
    labels = [CLASSES.index(c) for c in cls_names]
    return {
        "embeddings": rng.standard_normal((n, FEATURE_DIM), dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.int64),
        "class_names": list(cls_names),
        "paths": [f"/tmp/fake/{i:04d}.jpg" for i in range(n)],
        "frame_ids": list(range(n)),
    }


# --- hashing helpers ----------------------------------------------------------
def test_tensor_sha256_deterministic_and_sensitive():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    b = a.copy()
    assert tensor_sha256(a) == tensor_sha256(b)
    a[0, 0] += 1
    assert tensor_sha256(a) != tensor_sha256(b)


def test_file_sha256_matches_reference_sha256sum(tmp_path: Path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"dlq-head-only\x00\x01")
    import hashlib
    assert file_sha256(str(p)) == hashlib.sha256(b"dlq-head-only\x00\x01").hexdigest()


def test_state_dict_sha256_is_key_ordered_and_sensitive():
    sd = {"b": torch.zeros(2), "a": torch.ones(3)}
    assert state_dict_sha256(sd) == state_dict_sha256(
        {"a": torch.ones(3), "b": torch.zeros(2)})
    sd2 = {"b": torch.ones(2), "a": torch.ones(3)}
    assert state_dict_sha256(sd) != state_dict_sha256(sd2)


# --- seeded train transform ---------------------------------------------------
def test_seeded_train_transform_is_deterministic():
    img = _rng(1).integers(0, 256, (224, 224, 3), dtype=np.uint8)
    t1 = seeded_train_transform("default", 20260809)
    t2 = seeded_train_transform("default", 20260809)
    a, b = t1(img).numpy(), t2(img).numpy()
    assert a.shape == (3, 224, 224)
    assert np.array_equal(a, b)
    t3 = seeded_train_transform("default", 1)
    assert not np.array_equal(a, t3(img).numpy())


# --- cached logits ------------------------------------------------------------
def test_cached_logits_matches_torch_linear():
    rng = _rng(2)
    embs = rng.standard_normal((50, FEATURE_DIM), dtype=np.float32)
    W = rng.standard_normal((N_CLASSES, FEATURE_DIM), dtype=np.float32)
    b = rng.standard_normal((N_CLASSES,), dtype=np.float32)
    np_logits = cached_logits(embs, W, b)
    torch_logits = torch.nn.functional.linear(
        torch.from_numpy(embs), torch.from_numpy(W), torch.from_numpy(b))
    assert np.allclose(np_logits, torch_logits.numpy(), atol=1e-4)
    assert np.array_equal(np_logits.argmax(1),
                          torch_logits.argmax(1).numpy())


# --- LR schedule --------------------------------------------------------------
def test_cosine_warmup_lr_warmup_then_cosine_decay():
    total, warmup, base = 1000, 100, 1e-3
    vals = [cosine_warmup_lr(s, total, warmup, base) for s in range(total + 1)]
    assert vals[0] == base / warmup
    assert vals[warmup - 1] == pytest.approx(base)
    assert vals[total] == pytest.approx(0.0, abs=1e-12)
    tail = vals[warmup:]
    assert all(tail[i] >= tail[i + 1] for i in range(len(tail) - 1))


# --- metrics ------------------------------------------------------------------
def test_evaluate_cm_accuracy_f1_errors():
    cm = np.zeros((29, 29), dtype=np.int64)
    for i in range(29):
        cm[i, i] = 20
    cm[18, 4] = 5
    cm[18, 18] -= 5
    ev = evaluate_cm(np.arange(29).repeat(20).astype(np.int64),
                     np.arange(29).repeat(20).astype(np.int64))
    ev = evaluate_cm(np.array([18] * 20), np.array([4] * 5 + [18] * 15))
    assert ev["total_errors"] == 5
    assert ev["accuracy"] == 15 / 20
    assert ev["per_class"]["S"]["recall"] == 0.75
    assert ev["per_class"]["S"]["accuracy"] == 0.75


def test_s_e_metrics_pairs_counted_from_cm():
    cm = np.zeros((29, 29), dtype=np.int64)
    for i in range(29):
        cm[i, i] = 100
    cm[CLASSES.index("S"), CLASSES.index("E")] = 7
    cm[CLASSES.index("T"), CLASSES.index("L")] = 3
    sm = s_e_metrics(cm)
    assert sm["S->E"] == 7
    assert sm["T->L"] == 3
    assert sm["E->S"] == 0


# --- boundary groups ----------------------------------------------------------
def test_boundary_groups_assigns_by_prediction_and_true_class():
    s_idx, e_idx = CLASSES.index("S"), CLASSES.index("E")
    W = np.zeros((N_CLASSES, FEATURE_DIM), dtype=np.float32)
    b = np.zeros(N_CLASSES, dtype=np.float32)
    W[s_idx, 0] = 1.0
    W[e_idx, 0] = -1.0  # margin_SE = (w_S-w_E) x = 2*x0; denom = ||w_S-w_E|| = 2

    rng = _rng(3)
    x_pos = np.full((50, FEATURE_DIM), 0.0, dtype=np.float32)
    x_pos[:, 0] = 2.0  # S side: margin +4, dist +2
    x_neg = np.full((50, FEATURE_DIM), 0.0, dtype=np.float32)
    x_neg[:, 0] = -2.0  # E side: margin -4, dist -2
    emb = np.concatenate([x_pos, x_neg, x_neg])
    names = ["S"] * 100 + ["E"] * 50
    val = _fake_cache(150, rng, cls_names=names)
    val["embeddings"] = emb
    val["labels"] = np.array([CLASSES.index(c) for c in names], dtype=np.int64)
    val["class_names"] = names

    groups = boundary_groups(val, W, b)
    err = groups["S->E errors"]
    ok = groups["correct S"]
    te = groups["true E"]
    assert err["n"] == 50 and err["margin_mean"] == pytest.approx(-4.0)
    assert err["distance_mean"] == pytest.approx(-2.0)
    assert err["n_on_S_side"] == 0
    assert ok["n"] == 50 and ok["margin_mean"] == pytest.approx(4.0)
    assert ok["n_on_S_side"] == 50
    assert te["n"] == 50 and te["margin_mean"] == pytest.approx(-4.0)


# --- cache verification -------------------------------------------------------
def test_verify_cache_against_manifest_alignment(tmp_path: Path):
    rng = _rng(4)
    manifest = SplitManifest(0.8, 0, str(tmp_path))
    manifest.val = [Sample(f"/tmp/fake/{i:04d}.jpg", i % N_CLASSES,
                           CLASSES[i % N_CLASSES]) for i in range(20)]
    val = _fake_cache(20, rng)
    verify_cache_against_manifest(val, manifest)  # must not raise
    val["paths"][3] = "/tmp/fake/XXXX.jpg"
    with pytest.raises(SystemExit):
        verify_cache_against_manifest(val, manifest)


def test_verify_cache_pairs_split_counts_and_overlap(tmp_path: Path):
    rng = _rng(5)
    manifest = SplitManifest(0.8, 0, str(tmp_path))
    manifest.train = [Sample(f"/tmp/fake/t{i}.jpg", i % N_CLASSES,
                             CLASSES[i % N_CLASSES]) for i in range(40)]
    manifest.val = [Sample(f"/tmp/fake/v{i}.jpg", i % N_CLASSES,
                           CLASSES[i % N_CLASSES]) for i in range(20)]
    sha = "aa" * 32
    (tmp_path / "manifest.json").write_text(
        json.dumps({"content_sha256": sha, "root": str(tmp_path)}))
    train = _fake_cache(40, rng)
    train.update({"split": "train", "n": 40, "manifest_sha256": sha,
                  "parent_checkpoint": "ckpt.pt"})
    val = _fake_cache(20, rng)
    val.update({"split": "val", "n": 20, "manifest_sha256": sha,
                "parent_checkpoint": "ckpt.pt"})
    val["frame_ids"] = [i for i in range(40, 60)]  # disjoint from train 0..39
    out = verify_cache_pairs(train, val, manifest, str(tmp_path / "manifest.json"),
                             "ckpt.pt")
    assert out["train_val_frame_overlap"] == 0
    assert out["train_n"] == 40 and out["val_n"] == 20

    val["manifest_sha256"] = "bb" * 32
    with pytest.raises(SystemExit):
        verify_cache_pairs(train, val, manifest, str(tmp_path / "manifest.json"),
                           "ckpt.pt")
    val["manifest_sha256"] = sha
    val["frame_ids"] = [i for i in range(40, 60)]  # force an overlap with train
    val["frame_ids"][0] = 0
    with pytest.raises(SystemExit):
        verify_cache_pairs(train, val, manifest, str(tmp_path / "manifest.json"),
                           "ckpt.pt")


# --- head init + head-only training ------------------------------------------
def test_init_linear_copies_w0_b0():
    rng = _rng(6)
    W0 = rng.standard_normal((N_CLASSES, FEATURE_DIM), dtype=np.float32)
    b0 = rng.standard_normal(N_CLASSES, dtype=np.float32)
    linear = init_linear(torch.nn.Linear(FEATURE_DIM, N_CLASSES), W0, b0)
    assert np.array_equal(linear.weight.detach().numpy(), W0)
    assert np.array_equal(linear.bias.detach().numpy(), b0)


def test_train_candidate_head_only_moves_weights_off_init():
    rng = _rng(7)
    # cleanly separable 29-class problem on axis features (margin 5)
    def separable(n: int):
        labels = rng.integers(0, N_CLASSES, n)
        x = np.zeros((n, FEATURE_DIM), dtype=np.float32)
        x[np.arange(n), labels] = 5.0
        x += 0.05 * rng.standard_normal((n, FEATURE_DIM), dtype=np.float32)
        return x, labels.astype(np.int64)

    x_tr, y_tr = separable(464)
    x_va, y_va = separable(232)
    train = _fake_cache(len(y_tr), rng)
    val = _fake_cache(len(y_va), rng)
    train["embeddings"], train["labels"] = x_tr, y_tr
    val["embeddings"], val["labels"] = x_va, y_va
    W0 = -np.eye(N_CLASSES, FEATURE_DIM, dtype=np.float32)  # deliberately wrong
    b0 = np.zeros(N_CLASSES, dtype=np.float32)
    start_ev = evaluate_cm(y_va, cached_logits(x_va, W0, b0).argmax(1))
    assert start_ev["macro_f1"] < 0.1
    cw = [1.0] * N_CLASSES
    out = train_candidate(train, val, W0, b0, cw, 5e-2, epochs=30, patience=6,
                          batch_size=64, seed=20260809, device="cpu")
    assert not np.allclose(out["W"], W0)
    assert out["W"].shape == (N_CLASSES, FEATURE_DIM)
    assert out["b"].shape == (N_CLASSES,)
    assert out["best_macro_f1"] > start_ev["macro_f1"]
    assert out["best_macro_f1"] > 0.9
    assert 1 <= out["best_epoch"] <= 30


def test_train_candidate_class_weights_flow_through():
    rng = _rng(8)
    train = _fake_cache(48, rng)
    val = _fake_cache(24, rng)
    W0 = rng.standard_normal((N_CLASSES, FEATURE_DIM), dtype=np.float32)
    b0 = rng.standard_normal(N_CLASSES, dtype=np.float32)
    cw = np.full(N_CLASSES, 2.0)
    cw[CLASSES.index("S")] = 5.0
    out = train_candidate(train, val, W0, b0, cw.tolist(), 1e-3, epochs=2,
                          patience=1, batch_size=16, seed=20260809,
                          device="cpu")
    assert out["W"].shape == (N_CLASSES, FEATURE_DIM)


# --- verdict classification --------------------------------------------------
def _base_current(s_to_e_before, s_to_e_after, acc=0.98, f1=0.98,
                  e_recall=0.99):
    cm = np.zeros((29, 29), dtype=np.int64)
    for i in range(29):
        cm[i, i] = 598
    cm[CLASSES.index("S"), CLASSES.index("E")] = s_to_e_before
    cm[CLASSES.index("S"), CLASSES.index("S")] = 598 - s_to_e_before
    cm2 = cm.copy()
    cm2[CLASSES.index("S"), CLASSES.index("E")] = s_to_e_after
    cm2[CLASSES.index("S"), CLASSES.index("S")] = 598 - s_to_e_after
    ev = evaluate_cm(np.zeros(600, dtype=np.int64), np.zeros(600, dtype=np.int64))
    base = {"accuracy": acc, "macro_f1": f1, "total_errors": 391, "cm": cm,
            "per_class": {c: {"accuracy": 1.0, "precision": 1.0, "recall": 1.0,
                              "f1": 1.0} for c in CLASSES},
            **s_e_metrics(cm)}
    base["E recall"] = e_recall
    cur = {"accuracy": acc, "macro_f1": f1, "total_errors": 391 - (s_to_e_before - s_to_e_after),
           "cm": cm2,
           "per_class": base["per_class"],
           **s_e_metrics(cm2)}
    cur["E recall"] = e_recall
    return base, cur


def test_classify_clean_improvement():
    base, cur = _base_current(138, 60)
    v = classify(base, cur)
    assert v["verdict"] == "CLEAN IMPROVEMENT"
    assert v["rules"]["S_to_E_delta"] == 78
    assert v["rules"]["S_to_E_drop_material"] is True


def test_classify_tradeoff_when_accuracy_drops():
    base, cur = _base_current(138, 60, acc=0.98)
    cur["accuracy"] = 0.975
    cur["macro_f1"] = 0.98
    v = classify(base, cur)
    assert v["verdict"] == "TRADEOFF"


def test_classify_regression():
    base, cur = _base_current(138, 138)  # no S->E improvement
    cur["accuracy"] = 0.97
    cur["macro_f1"] = 0.969
    v = classify(base, cur)
    assert v["verdict"] == "REGRESSION"


def test_classify_no_meaningful_change():
    base, cur = _base_current(138, 135)
    v = classify(base, cur)
    assert v["verdict"] == "NO MEANINGFUL CHANGE"


def test_classify_watch_pair_regression_is_not_clean():
    base, cur = _base_current(138, 60)
    t_idx, l_idx = CLASSES.index("T"), CLASSES.index("L")
    cur["cm"][t_idx, l_idx] = 45
    cur["cm"][t_idx, t_idx] -= 45
    v = classify(base, cur)
    assert v["verdict"] == "TRADEOFF"
    assert "T->L" in v["rules"]["material_pair_regressions"]


# --- deployable checkpoint reconstruction -------------------------------------
def test_build_deployable_checkpoint_keeps_verdict_and_hash(tmp_path: Path):
    from src.model import build_efficientnet

    rng = _rng(9)
    W = rng.standard_normal((N_CLASSES, FEATURE_DIM), dtype=np.float32)
    b = rng.standard_normal(N_CLASSES, dtype=np.float32)
    best = {"lr": 1e-4, "best_epoch": 3, "best_macro_f1": 0.98}
    verdict = {"verdict": "CLEAN IMPROVEMENT"}
    train_cache = {"parent_checkpoint_sha256": "aa" * 32,
                   "backbone_state_dict_sha256": "bb" * 32}
    parent_model = build_efficientnet(pretrained=False)
    parent_model.eval()
    parent = tmp_path / "parent.pt"
    torch.save({"model": parent_model.state_dict(), "arch": "efficientnet",
                "class_weights": [1.0] * N_CLASSES},
               parent)
    deploy = tmp_path / "deploy.pt"
    info = build_deployable_checkpoint(str(parent), train_cache, W, b, best,
                                       verdict, str(deploy))
    assert deploy.is_file()
    ckpt = torch.load(deploy, map_location="cpu")
    assert ckpt["schema"] == "asl-head-only-ft/v1"
    assert ckpt["verdict"] == "CLEAN IMPROVEMENT"
    assert ckpt["head_lr"] == 1e-4
    assert "1.1.weight" in ckpt["model"]
    assert info["file_sha256"] == file_sha256(str(deploy))


# --- integrated artifacts (skipped when artifacts are absent) -----------------
@pytest.mark.skipif(not VAL_CACHE.is_file(), reason="val cache not built yet")
def test_val_cache_file_invariants():
    d = torch.load(VAL_CACHE, map_location="cpu", weights_only=False)
    assert d["schema"] == "asl-embedding-cache/v1"
    assert d["split"] == "val" and d["n"] == 17400
    assert d["embeddings"].shape == (17400, FEATURE_DIM)
    assert d["labels"].shape == (17400,)
    assert tensor_sha256(d["embeddings"]) == d["embeddings_sha256"]
    per = {c: d["class_names"].count(c) for c in CLASSES}
    assert set(per.values()) == {600} and len(per) == 29


@pytest.mark.skipif(not TRAIN_CACHE.is_file(), reason="train cache not built yet")
def test_train_cache_file_invariants():
    d = torch.load(TRAIN_CACHE, map_location="cpu", weights_only=False)
    assert d["schema"] == "asl-embedding-cache/v1"
    assert d["split"] == "train" and d["n"] == 69600
    assert d["embeddings"].shape == (69600, FEATURE_DIM)
    assert tensor_sha256(d["embeddings"]) == d["embeddings_sha256"]


@pytest.mark.skipif(not (CHECKPOINT.is_file() and VAL_CACHE.is_file()
                         and BASELINE.is_file()),
                    reason="checkpoint/cache/baseline missing")
def test_sanity_gate_reproduces_baseline_exactly():
    from scripts.analyze_s_to_e_head import extract_head
    from src.evaluate import load_checkpoint
    from scripts.train_head_only_cached import recompute_baseline, load_cache

    model = load_checkpoint(str(CHECKPOINT), "cpu")
    W0, b0, _ = extract_head(model)
    val = load_cache(str(VAL_CACHE))
    ev, bcm = recompute_baseline(val, W0, b0, str(BASELINE))
    assert ev["accuracy"] == pytest.approx(0.9775287356321839, abs=1e-9)
    assert ev["macro_f1"] == pytest.approx(0.9775082322752521, abs=1e-9)
    assert s_e_metrics(bcm)["S->E"] == 138


@pytest.mark.skipif(not (CHECKPOINT.is_file() and VAL_CACHE.is_file()
                         and Path(DEFAULT_MANIFEST).is_file()),
                    reason="checkpoint/cache/manifest missing")
def test_boundary_groups_before_matches_known_baseline():
    from scripts.analyze_s_to_e_head import extract_head
    from src.evaluate import load_checkpoint
    from scripts.train_head_only_cached import load_cache

    model = load_checkpoint(str(CHECKPOINT), "cpu")
    W0, b0, _ = extract_head(model)
    val = load_cache(str(VAL_CACHE))
    groups = boundary_groups(val, W0, b0)
    err = groups["S->E errors"]
    assert err["n"] == 138
    assert err["margin_mean"] == pytest.approx(-1.208, abs=0.02)
    assert err["margin_median"] == pytest.approx(-1.047, abs=0.02)
    assert err["distance_mean"] == pytest.approx(-0.340, abs=0.02)
    assert groups["correct S"]["n"] == 462
    assert groups["correct S"]["margin_mean"] == pytest.approx(2.880, abs=0.02)
    assert groups["true E"]["n"] == 600
    assert groups["true E"]["margin_mean"] == pytest.approx(-5.317, abs=0.02)


@pytest.mark.skipif(not (CHECKPOINT.is_file() and VAL_CACHE.is_file()
                         and Path(DEFAULT_MANIFEST).is_file()),
                    reason="checkpoint/cache/manifest missing")
def test_verify_cached_logits_reproduce_full_model():
    from scripts.train_head_only_cached import load_cache, verify_cached_logits
    from src.data import SplitManifest

    manifest = SplitManifest.from_json(DEFAULT_MANIFEST)
    val = load_cache(str(VAL_CACHE))
    verif = verify_cached_logits(str(CHECKPOINT), val, manifest, "cpu",
                                 n_per_class=2)
    assert verif["max_abs_logit_diff"] <= 1e-3


@pytest.mark.skipif(not OUT_JSON.is_file(), reason="experiment report missing")
def test_experiment_report_artifacts_and_verdict():
    report = json.loads(OUT_JSON.read_text())
    assert report["schema"] == SCHEMA
    assert report["sanity_gate"]["passed"] is True
    assert report["sanity_gate"]["S_to_E"] == 138
    assert report["classification"]["verdict"] in (
        "CLEAN IMPROVEMENT", "TRADEOFF", "NO MEANINGFUL CHANGE", "REGRESSION")
    assert report["current"]["accuracy"] >= report["baseline"]["accuracy"] - 0.005
    assert report["before_after_pairs"]["S->E"]["before"] == 138
    assert report["cache"]["train"]["n"] == 69600
    assert report["cache"]["val"]["n"] == 17400
    deploy = Path(report["artifacts"]["deployable_checkpoint"]["path"])
    assert deploy.is_file()


@pytest.mark.skipif(not (CHECKPOINT.is_file() and DEPLOY.is_file()),
                    reason="parent + deployable checkpoint missing")
def test_deployable_backbone_is_bit_identical_to_parent():
    parent = torch.load(str(CHECKPOINT), map_location="cpu")
    deploy = torch.load(str(DEPLOY), map_location="cpu")
    p_keys = {k for k in parent["model"] if k.startswith("0.")}
    d_keys = {k for k in deploy["model"] if k.startswith("0.")}
    assert p_keys == d_keys
    psd = {k: parent["model"][k] for k in sorted(p_keys)}
    dsd = {k: deploy["model"][k] for k in sorted(d_keys)}
    assert state_dict_sha256(psd) == state_dict_sha256(dsd)
    assert deploy["schema"] == "asl-head-only-ft/v1"
    assert deploy["verdict"] in (
        "CLEAN IMPROVEMENT", "TRADEOFF", "NO MEANINGFUL CHANGE", "REGRESSION")


@pytest.mark.skipif(not (CHECKPOINT.is_file() and DEPLOY.is_file()),
                    reason="parent + deployable checkpoint missing")
def test_deployable_checkpoint_loads_via_load_checkpoint():
    from src.evaluate import load_checkpoint
    model = load_checkpoint(str(DEPLOY), "cpu")
    assert isinstance(model, torch.nn.Sequential) and len(model) == 2
    x = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, N_CLASSES)


@pytest.mark.skipif(not (TRAIN_CACHE.is_file() and VAL_CACHE.is_file()
                         and CHECKPOINT.is_file()),
                    reason="cache + checkpoint missing")
def test_cache_hashes_unchanged_after_experiment():
    """The experiment must never rewrite the embedding cache files."""
    from scripts.train_head_only_cached import load_cache
    for path in (TRAIN_CACHE, VAL_CACHE):
        d = load_cache(str(path))
        assert tensor_sha256(d["embeddings"]) == d["embeddings_sha256"]
