"""The val-tail brightness check (#12).

The property worth guarding here is not the arithmetic — it is *which classes get
compared*. The first version of this experiment hardcoded the failing group as
the classes the docs discussed (S, E, M, N) when only S was actually below the
bar, and the three passing classes' large deltas produced a confident false
positive. So the tests below pin the grouping to the report's ``below_target``
field and check that a shift confined to passing classes is not reported as a
shift.

Synthetic flat-grey frames throughout: the statistic is a median grey level, and
a constant-valued array exercises it exactly as a photograph would, without
needing the 87k-image dataset present.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from scripts.check_val_brightness import (  # noqa: E402
    class_delta,
    failing_from_report,
    median_grey,
    stride_sample,
    verdict,
)


def write_class(root, cls, n=100, train_level=120, val_level=120, train_frac=0.8):
    """A class folder whose last ``1 - train_frac`` frames sit at ``val_level``."""
    d = root / cls
    d.mkdir(parents=True, exist_ok=True)
    cut = int(n * train_frac)
    for i in range(1, n + 1):
        level = train_level if i <= cut else val_level
        cv2.imwrite(str(d / f"{cls}{i}.jpg"), np.full((16, 16), level, np.uint8))
    return d


def test_median_grey_skips_unreadable_files(tmp_path):
    """One corrupt frame must not take down the sweep."""
    d = write_class(tmp_path, "S", n=4, train_frac=1.0)
    bad = d / "S999.jpg"
    bad.write_bytes(b"not a jpeg")

    mean, n = median_grey([str(p) for p in sorted(d.iterdir())])

    assert n == 4, "the corrupt frame should be skipped, not counted"
    assert mean == pytest.approx(120.0)


def test_median_grey_all_unreadable_is_nan_not_crash(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"nope")
    mean, n = median_grey([str(bad)])
    assert n == 0 and np.isnan(mean)


def test_stride_sample_spreads_and_caps():
    items = list(range(1000))
    got = stride_sample(items, 40)
    assert len(got) == 40
    assert got[0] == 0 and got[-1] > 900, "should span the range, not clump at the start"
    assert stride_sample([], 40) == []
    assert stride_sample(items, 0) == []


def test_class_delta_measures_the_val_tail(tmp_path):
    """A darker tail must show up as a negative delta of the right size."""
    write_class(tmp_path, "S", n=100, train_level=140, val_level=90)

    tr, va, delta, n_tr, n_va = class_delta(str(tmp_path), "S", per_class=40, train_frac=0.8)

    assert tr == pytest.approx(140.0)
    assert va == pytest.approx(90.0)
    assert delta == pytest.approx(-50.0)
    assert n_tr > 0 and n_va > 0


def test_class_delta_missing_class_returns_none(tmp_path):
    assert class_delta(str(tmp_path), "S", per_class=40, train_frac=0.8) is None


def test_failing_classes_come_from_the_report(tmp_path):
    """The grouping is read, never assumed -- this is the bug that got it wrong."""
    p = tmp_path / "report.json"
    p.write_text(json.dumps({"below_target": ["S"]}), encoding="utf-8")
    assert failing_from_report(str(p)) == ["S"]


def test_report_without_below_target_is_an_error(tmp_path):
    """Silently defaulting to 'nothing failed' would invert the verdict."""
    p = tmp_path / "report.json"
    p.write_text(json.dumps({"per_class": {}}), encoding="utf-8")
    with pytest.raises(KeyError):
        failing_from_report(str(p))


def test_verdict_flags_a_real_class_specific_shift():
    assert verdict(fail_d=-40.0, pass_d=-2.0).startswith("DISTRIBUTION SHIFT")


def test_verdict_calls_global_drift_what_it_is():
    """The 2026-08-06 case: everything darkens, failing classes no worse."""
    assert verdict(fail_d=-17.4, pass_d=-28.0).startswith("GLOBAL DRIFT")


def test_verdict_flat_when_nothing_moves():
    assert verdict(fail_d=-1.0, pass_d=-1.0).startswith("FLAT")


def test_shift_confined_to_passing_classes_is_not_a_shift():
    """A passing class darkening hard must not implicate the failing one.

    This is the exact shape of the false positive: E darkened -50.9 and scored
    1.000, and grouping it with S produced a 'distribution shift' verdict.
    """
    assert not verdict(fail_d=-17.4, pass_d=-50.9).startswith("DISTRIBUTION SHIFT")
