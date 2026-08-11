"""Tests for the partial fine-tuning experiment.

Locks the numerically-verifiable contract of
``scripts/train_partial_finetune.py`` and ``scripts/analyze_partial_finetune.py``:
parent provenance, manifest identity, the inspected model structure, the exact
trainable set (final MBConv block ``blocks[6]`` conv/SE + ``Linear(1280,29)``,
with every BatchNorm tensor frozen), the differential-LR optimizer groups, BN
eval-mode freezing, parameter-change verification (only allowlisted tensors may
differ), the fixed diagnostic groups, embedding-movement/nearest-neighbour
computations, the deployable checkpoint schema, and the integrated artifacts.

No test asserts an outcome — the verdict may be any of the four; the tests
assert the experiment was run correctly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.analyze_partial_finetune import (
    DEFAULT_DEPLOY_OUT,
    DEFAULT_MANIFEST,
    DEFAULT_OUT_JSON,
    DEFAULT_OUT_MD,
    FEATURE_DIM,
    N_CLASSES,
    build_deployable,
    group_arrays,
    l2_normalize,
    margin_rows,
    nearest_stats,
    summarize_movement,
)
from scripts.train_partial_finetune import (
    DEFAULT_CANDIDATE_OUT,
    DEFAULT_GATE,
    DEFAULT_PARENT,
    EXPECTED_GEOMETRY_SAFE,
    MANIFEST_SHA,
    PARENT_FILE_SHA,
    apply_freeze_policy,
    bn_eval,
    bn_param_names,
    is_bn_module,
    verify_manifest,
    verify_param_changes,
    verify_parent,
)
from src.data import CLASSES, SplitManifest
from src.model import build_efficientnet

CHECKPOINT = Path(DEFAULT_PARENT)
MANIFEST = Path(DEFAULT_MANIFEST)
GATE = Path(DEFAULT_GATE)
CANDIDATE = Path(DEFAULT_CANDIDATE_OUT)
DEPLOY = Path(DEFAULT_DEPLOY_OUT)
OUT_JSON = Path(DEFAULT_OUT_JSON)
TRAIN_JSON = Path("docs/reports/partial_finetune_training.json")

S_IDX = CLASSES.index("S")
E_IDX = CLASSES.index("E")


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _build_model():
    return build_efficientnet(pretrained=False).eval()


# --- BN detection ------------------------------------------------------------
def test_is_bn_module_detects_batchnorm_variants():
    m = _build_model()
    bns = 0
    for mod in m.modules():
        if is_bn_module(mod):
            bns += 1
    assert bns == 49  # 7 stages * 3 bn + conv_stem-side bn1 + conv_head-side bn2
    assert not is_bn_module(m[0].conv_stem)
    assert is_bn_module(m[0].bn1)


def test_bn_param_names_covers_affine_and_running_stats():
    m = _build_model()
    names = bn_param_names(m)
    assert "0.bn1.weight" in names and "0.bn1.bias" in names
    assert "0.blocks.6.0.bn1.weight" in names
    assert all(not n.endswith("running_mean") for n in names)  # buffers, not params


def test_bn_eval_forces_all_batchnorm_to_eval():
    m = _build_model().train()
    assert any(mod.training for mod in m.modules() if is_bn_module(mod))
    bn_eval(m)
    assert not any(mod.training for mod in m.modules() if is_bn_module(mod))


# --- freeze policy -----------------------------------------------------------
def test_apply_freeze_policy_exact_trainable_set():
    m = _build_model()
    frz = apply_freeze_policy(m)
    names = set(frz["trainable_names"])
    expected = {
        "0.blocks.6.0.conv_pw.weight",
        "0.blocks.6.0.conv_dw.weight",
        "0.blocks.6.0.se.conv_reduce.weight",
        "0.blocks.6.0.se.conv_reduce.bias",
        "0.blocks.6.0.se.conv_expand.weight",
        "0.blocks.6.0.se.conv_expand.bias",
        "0.blocks.6.0.conv_pwl.weight",
        "1.1.weight", "1.1.bias",
    }
    assert names == expected
    assert frz["bn_affine_frozen"]
    # every BN-affine param stays frozen
    bns = bn_param_names(m)
    assert not any(n in bns for n in names)


def test_apply_freeze_policy_param_counts():
    m = _build_model()
    frz = apply_freeze_policy(m)
    total = frz["total_params"]
    assert frz["trainable_params"] + frz["frozen_params"] == total
    assert frz["trainable_fraction"] == pytest.approx(
        frz["trainable_params"] / total)
    assert 0.1 < frz["trainable_fraction"] < 0.3


def test_apply_freeze_policy_overrides_tampering():
    """The policy is authoritative: any pre-existing requires_grad outside the
    allowlist is wiped, so no unexpected module can stay trainable."""
    m = _build_model()
    m[0].blocks[5][0].conv_pw.weight.requires_grad = True  # tamper before
    frz = apply_freeze_policy(m)
    assert not m[0].blocks[5][0].conv_pw.weight.requires_grad
    expected = {"0.blocks.6.0.conv_pw.weight", "0.blocks.6.0.conv_dw.weight",
                "0.blocks.6.0.conv_pwl.weight",
                "0.blocks.6.0.se.conv_reduce.weight",
                "0.blocks.6.0.se.conv_reduce.bias",
                "0.blocks.6.0.se.conv_expand.weight",
                "0.blocks.6.0.se.conv_expand.bias",
                "1.1.weight", "1.1.bias"}
    assert set(frz["trainable_names"]) == expected


def test_apply_freeze_policy_blocks6_conv_only_no_bn():
    """No BN affine in blocks[6] may be trainable (conservative BN policy)."""
    m = _build_model()
    apply_freeze_policy(m)
    for n, p in m.named_parameters():
        if n.startswith("0.blocks.6.") and is_bn_tensor_name(n):
            assert not p.requires_grad


def is_bn_tensor_name(n: str) -> bool:
    head = n.split(".")[-2] if "." in n else ""
    return head.startswith("bn") or ".bn" in n


# --- parameter-change verification ------------------------------------------
def test_verify_param_changes_only_trainable_differ():
    m = _build_model()
    frz = apply_freeze_policy(m)
    parent_sd = {k: v.clone() for k, v in m.state_dict().items()}
    cand_sd = {k: v.clone() for k, v in parent_sd.items()}
    for n in frz["trainable_names"]:
        cand_sd[n].add_(1e-4)
    ver = verify_param_changes(parent_sd, cand_sd, frz["trainable_names"])
    assert ver["n_changed"] == len(frz["trainable_names"])
    assert ver["bn_tensors_unchanged"] is True
    assert ver["n_changed"] + ver["n_unchanged"] == len(parent_sd)


def test_verify_param_changes_fails_when_frozen_tensor_differs():
    m = _build_model()
    frz = apply_freeze_policy(m)
    parent_sd = {k: v.clone() for k, v in m.state_dict().items()}
    cand_sd = {k: v.clone() for k, v in parent_sd.items()}
    cand_sd["0.blocks.5.0.conv_pw.weight"].add_(1e-4)  # frozen block!
    with pytest.raises(SystemExit, match="outside the trainable set"):
        verify_param_changes(parent_sd, cand_sd, frz["trainable_names"])


def test_verify_param_changes_fails_when_bn_changes():
    m = _build_model()
    frz = apply_freeze_policy(m)
    parent_sd = {k: v.clone() for k, v in m.state_dict().items()}
    cand_sd = {k: v.clone() for k, v in parent_sd.items()}
    for n in frz["trainable_names"]:
        cand_sd[n].add_(1e-4)
    cand_sd["0.bn1.running_mean"].add_(0.01)  # frozen BN statistic!
    with pytest.raises(SystemExit, match="BatchNorm tensors changed"):
        verify_param_changes(parent_sd, cand_sd, frz["trainable_names"])


def test_verify_param_changes_fails_when_trainable_unchanged():
    m = _build_model()
    frz = apply_freeze_policy(m)
    parent_sd = {k: v.clone() for k, v in m.state_dict().items()}
    with pytest.raises(SystemExit, match="did not change"):
        verify_param_changes(parent_sd, {**parent_sd}, frz["trainable_names"])


# --- manifest + parent provenance -------------------------------------------
@pytest.mark.skipif(not MANIFEST.is_file(), reason="manifest missing")
def test_verify_manifest_identity_and_counts():
    manifest = SplitManifest.from_json(str(MANIFEST))
    out = verify_manifest(str(MANIFEST), manifest)
    assert out["manifest_sha256"].startswith(MANIFEST_SHA)
    assert out["n_train"] == 69600 and out["n_val"] == 17400
    assert set(out["val_per_class"].values()) == {600}


@pytest.mark.skipif(not CHECKPOINT.is_file(), reason="parent checkpoint missing")
def test_verify_parent_provenance():
    out = verify_parent(str(CHECKPOINT))
    assert out["file_sha256"].startswith(PARENT_FILE_SHA)
    assert out["rebalance"] == "loss"
    assert out["geometry_safe_classes"] == EXPECTED_GEOMETRY_SAFE
    assert out["backgrounds"] is None
    assert len(out["class_weights"]) == N_CLASSES


# --- analyze-side pure helpers ----------------------------------------------
def test_l2_normalize_unit_norm():
    rng = _rng(1)
    x = rng.standard_normal((10, 128))
    n = l2_normalize(x)
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0)
    assert l2_normalize(np.zeros((3, 5))).sum() == 0.0


def test_group_arrays_fixed_groups():
    names = ["S"] * 600 + ["E"] * 600
    preds = np.array([E_IDX] * 138 + [S_IDX] * (600 - 138) + [E_IDX] * 600,
                     dtype=np.int64)
    m = group_arrays(names, preds)
    assert int(m["S->E errors"].sum()) == 138
    assert int(m["correct S"].sum()) == 462
    assert int(m["true E"].sum()) == 600


def test_nearest_stats_counts_sides():
    rng = _rng(2)
    pool_s = rng.standard_normal((5, 16))
    pool_e = rng.standard_normal((5, 16))
    q_near_s = pool_s[:2] + 1e-6
    q_near_e = pool_e[:2] + 1e-6
    out = nearest_stats(np.concatenate([q_near_s, q_near_e]), pool_s, pool_e)
    assert out["closer_to_S"] == 2
    assert out["closer_to_E"] == 2
    assert out["n"] == 4


def test_summarize_movement_range():
    cos = np.array([1.0, 0.5, 0.0, -1.0])  # movement 0, 0.5, 1, 2
    s = summarize_movement(cos)
    assert s["n"] == 4
    assert s["movement_mean"] == pytest.approx(0.875)
    assert s["movement_median"] == pytest.approx(0.75)
    assert s["cosine_min"] == pytest.approx(-1.0)


def test_margin_rows_sides():
    W = np.zeros((N_CLASSES, FEATURE_DIM), dtype=np.float32)
    b = np.zeros(N_CLASSES, dtype=np.float32)
    W[S_IDX, 0], W[E_IDX, 0] = 1.0, -1.0  # margin_SE = 2*x0; errors have x0<0
    emb = np.zeros((200, FEATURE_DIM), dtype=np.float32)
    emb[:50, 0] = 2.0    # S frame on S side (margin +4) -> correct S
    emb[50:100, 0] = -2.0  # S frame on E side (margin -4) -> S->E error
    emb[100:, 0] = -2.0    # true E (margin -4)
    names = ["S"] * 100 + ["E"] * 100
    preds = np.array([S_IDX] * 50 + [E_IDX] * 50 + [E_IDX] * 100, dtype=np.int64)
    masks = group_arrays(names, preds, expected=(50, 50, 100))
    rows = margin_rows(emb, W, b, masks, names)
    assert rows["S->E errors"]["n"] == 50
    assert rows["S->E errors"]["margin_mean"] == pytest.approx(-4.0)
    assert rows["correct S"]["n"] == 50
    assert rows["correct S"]["margin_mean"] == pytest.approx(4.0)
    assert rows["true E"]["n"] == 100
    assert rows["S->E errors"]["n_on_S_side"] == 0


# --- deployable checkpoint ---------------------------------------------------
def test_build_deployable_schema_and_verdict(tmp_path):
    from scripts.build_embedding_cache import file_sha256

    parent_model = _build_model()
    parent = tmp_path / "parent.pt"
    torch.save({"model": parent_model.state_dict(), "arch": "efficientnet",
                "class_weights": [1.0] * N_CLASSES}, parent)
    cand = tmp_path / "cand.pt"
    torch.save({"model": parent_model.state_dict(),
                "lr_backbone": 1e-5, "lr_head": 1e-4, "best_epoch": 2,
                "best_macro_f1": 0.98, "epochs_run": 3}, cand)
    verdict = {"verdict": "NO MEANINGFUL CHANGE"}
    out = tmp_path / "deploy.pt"
    info = build_deployable(str(parent), str(cand), "aa" * 32, "bb" * 32,
                            verdict, str(out))
    assert out.is_file()
    ckpt = torch.load(out, map_location="cpu")
    assert ckpt["schema"] == "asl-partial-ft/v1"
    assert ckpt["verdict"] == "NO MEANINGFUL CHANGE"
    assert ckpt["parent_checkpoint_sha256"] == "aa" * 32
    assert info["file_sha256"] == file_sha256(str(out))


# --- integrated artifacts (skipped when artifacts are absent) ----------------
@pytest.mark.skipif(not GATE.is_file(), reason="baseline gate report missing")
def test_gate_reproduces_established_baseline_exactly():
    gate = json.loads(GATE.read_text())
    est = json.loads(Path("docs/reports/dev_val_targeted_current.json").read_text())
    from src.evaluate import report_matrix
    assert np.array_equal(report_matrix(gate), report_matrix(est))
    assert gate["overall_accuracy"] == pytest.approx(0.9775287356321839, abs=1e-9)
    assert gate["macro_f1"] == pytest.approx(0.9775082322752521, abs=1e-9)
    cm = report_matrix(gate)
    assert int(cm.sum() - cm.trace()) == 391
    assert int(cm[S_IDX, E_IDX]) == 138


@pytest.mark.skipif(not CANDIDATE.is_file(), reason="candidate checkpoint missing")
def test_candidate_metadata():
    cand = torch.load(CANDIDATE, map_location="cpu")
    assert cand["schema"] == "asl-partial-ft-candidate/v1"
    assert cand["parent_checkpoint_sha256"].startswith(PARENT_FILE_SHA)
    assert cand["lr_backbone"] in (3e-6, 1e-5, 3e-5)
    assert cand["lr_head"] in (3e-5, 1e-4, 3e-4)
    assert 1 <= cand["best_epoch"] <= cand["epochs_run"]
    assert "0.blocks.6.0.conv_pw.weight" in cand["model"]


@pytest.mark.skipif(not (CANDIDATE.is_file() and CHECKPOINT.is_file()),
                    reason="candidate + parent missing")
def test_candidate_changed_only_allowlisted_tensors():
    from scripts.train_partial_finetune import verify_param_changes
    cand = torch.load(CANDIDATE, map_location="cpu")
    parent = torch.load(str(CHECKPOINT), map_location="cpu")
    ver = verify_param_changes(parent["model"], cand["model"],
                               cand["trainable_names"])
    assert ver["bn_tensors_unchanged"] is True
    assert ver["n_changed"] == len(cand["trainable_names"])


@pytest.mark.skipif(not TRAIN_JSON.is_file(), reason="training report missing")
def test_training_report_consistency():
    report = json.loads(TRAIN_JSON.read_text())
    assert report["schema"] == "asl-partial-ft-training/v1"
    assert report["gate"]["S_to_E"] == 138
    assert report["gate"]["total_errors"] == 391
    assert report["parent"]["geometry_safe_classes"] == EXPECTED_GEOMETRY_SAFE
    assert report["param_change"]["bn_tensors_unchanged"] is True
    selected = report["selected"]
    assert selected["lr_backbone"] in (3e-6, 1e-5, 3e-5)
    assert report["structure"]["n_blocks"] == 7
    assert report["structure"]["blocks_params"][6] == 717232


@pytest.mark.skipif(not OUT_JSON.is_file(), reason="final report missing")
def test_final_report_artifacts_and_verdict():
    report = json.loads(OUT_JSON.read_text())
    assert report["schema"] == "asl-partial-ft/v1"
    assert report["classification"]["verdict"] in (
        "CLEAN IMPROVEMENT", "TRADEOFF", "NO MEANINGFUL CHANGE", "REGRESSION")
    assert report["before_after_pairs"]["S->E"]["before"] == 138
    assert report["baseline"]["accuracy"] == pytest.approx(
        0.9775287356321839, abs=1e-9)
    assert report["diagnostics"]["movement"]["all val"]["n"] == 17400
    assert report["diagnostics"]["corrections"]["n"] == 138
    assert report["three_way"]["partial_finetune"]["S_to_E"] == \
        report["current"]["S->E"]
    deploy = Path(report["artifacts"]["deployable"]["path"])
    assert deploy.is_file()
    for p in (report["artifacts"]["confusion_png"],
              report["artifacts"]["confusion_csv"],
              report["artifacts"]["rep_png"]):
        assert Path(p).is_file()


@pytest.mark.skipif(not (CHECKPOINT.is_file() and DEPLOY.is_file()),
                    reason="parent + deployable missing")
def test_deployable_partial_ft_only_blocks6_and_head_changed():
    parent = torch.load(str(CHECKPOINT), map_location="cpu")
    deploy = torch.load(str(DEPLOY), map_location="cpu")
    assert deploy["schema"] == "asl-partial-ft/v1"
    assert deploy["parent_checkpoint_sha256"].startswith(PARENT_FILE_SHA)
    trainable = {
        "0.blocks.6.0.conv_pw.weight", "0.blocks.6.0.conv_dw.weight",
        "0.blocks.6.0.conv_pwl.weight", "0.blocks.6.0.se.conv_reduce.weight",
        "0.blocks.6.0.se.conv_reduce.bias",
        "0.blocks.6.0.se.conv_expand.weight",
        "0.blocks.6.0.se.conv_expand.bias",
        "1.1.weight", "1.1.bias",
    }
    changed = [k for k in parent["model"]
               if not torch.equal(parent["model"][k], deploy["model"][k])]
    assert set(changed) == trainable
    # BN running stats + affine frozen
    for k in parent["model"]:
        if "running_mean" in k or "running_var" in k or "num_batches" in k:
            assert torch.equal(parent["model"][k], deploy["model"][k])


@pytest.mark.skipif(not DEPLOY.is_file(), reason="deployable missing")
def test_deployable_partial_ft_loads_via_load_checkpoint():
    from src.evaluate import load_checkpoint
    model = load_checkpoint(str(DEPLOY), "cpu")
    assert isinstance(model, torch.nn.Sequential) and len(model) == 2
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 224, 224))
    assert out.shape == (1, N_CLASSES)
