"""Targeted fixes for the worst-confused classes (#12).

The #6 run passed the overall dev-val target (0.9545 > 95%) while three classes
sat below the 85% per-class bar — X at 0.507, V at 0.630, S at 0.723 — and 86%
of all errors lived in those three. Everything tested here exists to aim the next
run at exactly those classes and, just as importantly, to prove afterwards that
fixing them did not break a fourth.

The machinery is tested on synthetic confusion matrices rather than on a real
run, because the properties that matter are arithmetic (which classes get picked,
how hard, which mechanism applies them) and a GPU run cannot make those any more
true than a 29x29 array can.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.data import CLASS_TO_IDX, CLASSES, Sample
from src.evaluate import (
    PER_CLASS_TARGET,
    REPORT_SCHEMA,
    build_report,
    compare_reports,
    load_confusion_csv,
    load_report,
    report_matrix,
    save_confusion_csv,
    write_report,
)
from src.train import (
    PARTNER_ERROR_SHARE,
    TrainConfig,
    assert_not_webcam_derived,
    auto_geometry_safe_classes,
    derive_class_weights,
    loss_class_weights,
    resolve_geometry_safe_classes,
    sampler_weights,
)

N = len(CLASSES)


def _cm(per_class_errors: dict, n_per_class: int = 600) -> np.ndarray:
    """A confusion matrix where every class is perfect except those named.

    ``per_class_errors`` maps a true class to ``{predicted_class: count}``.
    """
    cm = np.zeros((N, N), dtype=np.int64)
    for i, c in enumerate(CLASSES):
        cm[i, i] = n_per_class
    for true, errors in per_class_errors.items():
        i = CLASS_TO_IDX[true]
        for pred, count in errors.items():
            cm[i, CLASS_TO_IDX[pred]] = count
            cm[i, i] -= count
    return cm


def _report(per_class_errors: dict, **kw) -> dict:
    return build_report(_cm(per_class_errors), **kw)


# The measured #6 failure shape, in miniature: one clean collision, one diffuse
# class, one mild pair that stays above the bar.
BASELINE_ERRORS = {
    "V": {"K": 219, "W": 3},
    "S": {"E": 160, "X": 4, "A": 2},
    "X": {"A": 120, "L": 78, "R": 65, "I": 16},
    "M": {"N": 41},
}


# --------------------------------------------------------------------------
# which classes are failing
# --------------------------------------------------------------------------


def test_below_target_finds_exactly_the_failing_classes():
    """M loses 41 frames but stays at 0.932 — the bar is per-class, not per-error."""
    report = _report(BASELINE_ERRORS)
    assert report["below_target"] == ["S", "V", "X"]
    assert report["per_class"]["M"]["accuracy"] == pytest.approx(0.932, abs=5e-4)
    assert report["per_class"]["X"]["accuracy"] == pytest.approx(0.535, abs=5e-4)


def test_committed_baseline_matches_the_published_numbers():
    """``docs/reports/dev_val_baseline_6.json`` must still say what RESULTS.md says.

    It is the 'before' half of the #12 deliverable, and it was reconstructed from
    a CSV that a later ``evaluate --figures`` run will overwrite — so if the two
    ever drift, the comparison is being made against the wrong run.
    """
    report = load_report("docs/reports/dev_val_baseline_6.json")
    assert report["overall_accuracy"] == pytest.approx(0.9545, abs=5e-5)
    assert report["macro_f1"] == pytest.approx(0.9518, abs=5e-5)
    assert report["below_target"] == ["S", "V", "X"]
    assert report["n_images"] == 17400
    assert report["source"]["kind"] == "dev_val"
    assert report["is_reported_metric"] is False  # never quotable as the result


# --------------------------------------------------------------------------
# class weights
# --------------------------------------------------------------------------


def test_weights_track_error_not_frequency():
    """Every class has 600 val frames, so only difficulty can move the weights."""
    weights = derive_class_weights(_report(BASELINE_ERRORS), alpha=1.0)
    by_class = dict(zip(CLASSES, weights))

    # Ordered by how badly each class failed: X worst, then V, then S.
    assert by_class["X"] > by_class["V"] > by_class["S"] > by_class["M"]
    # A solved class is not suppressed, just not boosted.
    assert by_class["Q"] == pytest.approx(by_class["Z"])
    assert by_class["X"] / by_class["Q"] == pytest.approx(1.465, abs=0.01)


def test_alpha_zero_is_a_no_op():
    """``--rebalance-alpha 0`` must leave every class equal, whatever the errors."""
    weights = derive_class_weights(_report(BASELINE_ERRORS), alpha=0.0)
    assert weights == pytest.approx([1.0] * N)


def test_alpha_scales_the_emphasis():
    mild = dict(zip(CLASSES, derive_class_weights(_report(BASELINE_ERRORS), alpha=1.0)))
    hard = dict(zip(CLASSES, derive_class_weights(_report(BASELINE_ERRORS), alpha=4.0)))
    assert hard["X"] / hard["Q"] > mild["X"] / mild["Q"]


def test_cap_binds_before_normalisation():
    """A pathological class cannot run away with the whole loss."""
    hopeless = _report({"X": {"A": 600}})  # X at 0.000
    weights = dict(zip(CLASSES, derive_class_weights(hopeless, alpha=50.0, cap=3.0)))
    assert weights["X"] / weights["Q"] == pytest.approx(3.0, abs=0.01)


def test_weights_are_mean_one():
    """Cosmetic but load-bearing for readability: 1.0 means 'not touched'."""
    weights = derive_class_weights(_report(BASELINE_ERRORS), alpha=2.0)
    assert sum(weights) / len(weights) == pytest.approx(1.0)


def test_class_missing_from_the_report_gets_the_neutral_weight():
    """A class with no val frames must not be guessed at."""
    report = _report(BASELINE_ERRORS)
    report["per_class"]["Q"] = {"accuracy": None, "n": 0}
    weights = dict(zip(CLASSES, derive_class_weights(report, alpha=1.0)))
    assert weights["Q"] == pytest.approx(weights["Z"])


@pytest.mark.parametrize("alpha,cap", [(-0.1, 3.0), (1.0, 0.5)])
def test_nonsense_weight_settings_are_rejected(alpha, cap):
    with pytest.raises(ValueError):
        derive_class_weights(_report(BASELINE_ERRORS), alpha=alpha, cap=cap)


# --------------------------------------------------------------------------
# aimed augmentation: which classes get the gentle policy
# --------------------------------------------------------------------------


def test_auto_picks_failing_classes_and_their_partners():
    """S is softened together with E, because the S/E boundary needs both.

    Includes X's partners A (41% of its errors) and L (26%) and R (22%), but not
    I (5%) — softening a class over a 5% leak would spread the change to classes
    that are not the problem.
    """
    chosen = auto_geometry_safe_classes(_report(BASELINE_ERRORS))
    assert set(chosen) == {"S", "V", "X", "E", "K", "A", "L", "R"}
    assert "I" not in chosen
    assert "M" not in chosen  # above the bar, so not targeted at all
    assert chosen == [c for c in CLASSES if c in set(chosen)]  # CLASSES order


def test_auto_partner_threshold_is_a_share_of_that_class_errors():
    """A big absolute count in a healthy class is not a reason to soften it."""
    # Y fails, leaking 250 to I and 50 to U; I is 83% of Y's errors, U is 17%.
    chosen = auto_geometry_safe_classes(_report({"Y": {"I": 250, "U": 50}}))
    assert "I" in chosen and "U" not in chosen
    assert 50 / 300 < PARTNER_ERROR_SHARE <= 250 / 300


def test_auto_is_empty_when_every_class_passes():
    """Nothing below the bar -> nothing to aim at, and no silent default set."""
    assert auto_geometry_safe_classes(_report({"M": {"N": 41}})) == []


def test_auto_honours_a_stricter_target():
    assert auto_geometry_safe_classes(_report({"M": {"N": 41}}), target=0.95) == ["M", "N"]


def test_explicit_class_list_is_normalised_to_classes_order():
    assert resolve_geometry_safe_classes("V, K ,S", None) == ["K", "S", "V"]


@pytest.mark.parametrize("spec", [None, "", "none", "NONE"])
def test_no_spec_means_no_gentle_classes(spec):
    assert resolve_geometry_safe_classes(spec, None) == []


def test_typo_in_class_list_is_rejected_not_ignored():
    """Silently dropping 'v' would produce a run that looks targeted but isn't."""
    with pytest.raises(ValueError, match=r"not ASL class names.*\['v'\]"):
        resolve_geometry_safe_classes("v,K", None)


def test_auto_without_a_report_explains_itself():
    with pytest.raises(ValueError, match="needs --rebalance-from"):
        resolve_geometry_safe_classes("auto", None)


# --------------------------------------------------------------------------
# loss weighting vs oversampling: exactly one applies
# --------------------------------------------------------------------------


def _samples(counts: dict) -> list:
    return [
        Sample(f"{c}{i}.jpg", CLASS_TO_IDX[c], c)
        for c, n in counts.items() for i in range(n)
    ]


def test_loss_mode_weights_the_loss_and_not_the_sampler():
    weights = derive_class_weights(_report(BASELINE_ERRORS), alpha=1.0)
    cfg = TrainConfig(root="x", class_weights=weights, rebalance="loss")
    assert loss_class_weights(cfg) == pytest.approx(weights)
    assert sampler_weights(cfg, _samples({"X": 2})) is None


def test_sampler_mode_weights_the_sampler_and_not_the_loss():
    """Applying both would square the emphasis nobody asked to square."""
    weights = derive_class_weights(_report(BASELINE_ERRORS), alpha=1.0)
    cfg = TrainConfig(root="x", class_weights=weights, rebalance="sampler")
    assert loss_class_weights(cfg) is None

    drawn = sampler_weights(cfg, _samples({"X": 2, "Q": 2}))
    assert drawn[0] == drawn[1] == pytest.approx(weights[CLASS_TO_IDX["X"]])
    assert drawn[2] == drawn[3] == pytest.approx(weights[CLASS_TO_IDX["Q"]])
    assert drawn[0] > drawn[2]  # the failing class really is drawn more often


def test_no_weights_means_neither_mechanism_engages():
    """The default run must be byte-for-byte the old behaviour."""
    for mode in ("loss", "sampler"):
        cfg = TrainConfig(root="x", rebalance=mode)
        assert loss_class_weights(cfg) is None
        assert sampler_weights(cfg, _samples({"X": 1})) is None


# --------------------------------------------------------------------------
# report round-trips
# --------------------------------------------------------------------------


def test_report_json_round_trips(tmp_path):
    original = _report(BASELINE_ERRORS, checkpoint="ckpt.pt", arch="efficientnet")
    path = tmp_path / "report.json"
    write_report(original, str(path))
    loaded = load_report(str(path))
    assert loaded["per_class"] == original["per_class"]
    np.testing.assert_array_equal(report_matrix(loaded), _cm(BASELINE_ERRORS))


def test_confusion_csv_round_trips(tmp_path):
    """The CSV is the only record older runs left, so it has to load exactly."""
    cm = _cm(BASELINE_ERRORS)
    path = tmp_path / "confusion_matrix.csv"
    save_confusion_csv(cm, str(path))
    np.testing.assert_array_equal(load_confusion_csv(str(path)), cm)


def test_csv_columns_are_matched_by_name_not_position(tmp_path):
    """A CSV written under a different class order still has to mean the same thing."""
    path = tmp_path / "reordered.csv"
    # Two classes, columns deliberately reversed relative to CLASSES.
    path.write_text("Q,B,A\nA,3,7\nB,5,2\n")
    cm = load_confusion_csv(str(path))
    a, b = CLASS_TO_IDX["A"], CLASS_TO_IDX["B"]
    assert cm[a, a] == 7 and cm[a, b] == 3
    assert cm[b, b] == 5 and cm[b, a] == 2


def test_unknown_class_in_csv_is_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("Q,A,NOPE\nA,1,2\n")
    with pytest.raises(ValueError, match="unknown classes"):
        load_confusion_csv(str(path))


def test_foreign_json_is_rejected(tmp_path):
    """A split manifest and a class report are both JSON with a 'classes' key."""
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema": "asl-frame-range-split/v1",
                                "classes": list(CLASSES)}))
    with pytest.raises(ValueError, match=REPORT_SCHEMA):
        load_report(str(path))


def test_webcam_report_cannot_aim_a_training_run():
    """The quarantine rule, enforced instead of merely written down.

    Class weights and the geometry-safe class list are augmentation settings, and
    CONTRIBUTING says the webcam set never picks one. A run aimed by a webcam
    report would look exactly like a legitimate one, which is why this is a hard
    failure rather than a warning.
    """
    web = _report(BASELINE_ERRORS, source_kind="webcam")
    with pytest.raises(SystemExit, match="quarantine rule"):
        assert_not_webcam_derived(web, "webcam.json")


def test_dev_val_report_may_aim_a_training_run(capsys):
    assert_not_webcam_derived(_report(BASELINE_ERRORS, source_kind="dev_val"), "dev.json")
    assert "warning" not in capsys.readouterr().out


def test_unprovenanced_csv_is_warned_about_not_refused(capsys):
    """A CSV records no source, so the check is as strong as its input and no more.

    Refusing outright would be worse: the #6 baseline was CSV-only. Warning keeps
    it usable while making the unverifiable step visible.
    """
    assert_not_webcam_derived(load_report("docs/figures/confusion_matrix.csv"), "cm.csv")
    assert "quarantine check cannot verify" in capsys.readouterr().out


def test_dev_val_and_webcam_reports_are_distinguishable():
    """Nothing downstream may have to guess which number it is holding."""
    dev = _report(BASELINE_ERRORS, source_kind="dev_val")
    web = _report(BASELINE_ERRORS, source_kind="webcam")
    assert dev["is_reported_metric"] is False
    assert web["is_reported_metric"] is True
    assert "NOT the reported number" in dev["note"]


# --------------------------------------------------------------------------
# before/after
# --------------------------------------------------------------------------


def test_compare_reports_buckets_fixed_and_still_broken():
    before = _report(BASELINE_ERRORS, label="before")
    after = _report({"V": {"K": 30}, "S": {"E": 20}, "X": {"A": 150, "L": 90}},
                    label="after")
    result = compare_reports(before, after, target=PER_CLASS_TARGET)
    assert set(result["fixed"]) == {"S", "V"}
    assert result["still_below"] == ["X"]
    assert result["newly_below"] == []


def test_compare_reports_catches_a_new_regression():
    """The failure mode this whole comparison exists for: fixing V by breaking K."""
    before = _report(BASELINE_ERRORS, label="before")
    after = _report({"K": {"V": 200}}, label="after")
    result = compare_reports(before, after)
    assert result["newly_below"] == ["K"]  # K was at 1.000, now 0.667
    assert set(result["fixed"]) == {"S", "V", "X"}


def test_compare_reports_flags_a_soft_regression_that_stays_above_the_bar():
    """A class that drops 1.000 -> 0.90 still passes, and still wants saying."""
    before = _report({})
    after = _report({"T": {"L": 60}})  # 1.000 -> 0.900
    result = compare_reports(before, after)
    assert result["newly_below"] == []
    assert "T" in result["softly_regressed"]


def test_compare_reports_survives_a_class_absent_from_one_side():
    """A partial webcam capture must not crash the comparison."""
    before = _report(BASELINE_ERRORS)
    after = _report(BASELINE_ERRORS)
    after["per_class"]["Q"] = {"accuracy": None, "n": 0}
    result = compare_reports(before, after)
    assert "Q" not in result["still_below"] + result["newly_below"]
