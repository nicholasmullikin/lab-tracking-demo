"""Mask helpers: the shared `mask_ops` functions and every caller's edge-case contract.

The caller tests were written against the per-module copies before those copies were
replaced, so they pin the behaviour each module had (None vs 0.0 vs NaN on an empty union,
tuple vs array centroids, the half-pixel offset) rather than the shared function's defaults.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from battle import (
    correction_acceptance_search,
    detector_scorecard,
    egoexo_correspondence,
    ensemble_reference,
    interaction_review,
    mask_ops,
    multiview_consensus,
    multiview_seed_transfer,
    review_metrics,
    segmentation_disagreement,
)


def _masks() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.zeros((6, 8), dtype=bool)
    b = np.zeros((6, 8), dtype=bool)
    a[1:4, 1:5] = True  # 12 px
    b[2:5, 3:7] = True  # 12 px, overlap 2x2 = 4, union 20
    empty = np.zeros((6, 8), dtype=bool)
    return a, b, empty


# --------------------------------------------------------------------------- mask_ops itself


def test_mask_iou_value_and_empty_union_parameter() -> None:
    a, b, empty = _masks()
    assert mask_ops.mask_iou(a, b) == pytest.approx(4 / 20)
    assert mask_ops.mask_iou(a, a) == 1.0
    assert mask_ops.mask_iou(a, empty) == 0.0
    assert mask_ops.mask_iou(empty, empty) == 0.0
    assert mask_ops.mask_iou(empty, empty, empty_union=None) is None
    assert math.isnan(mask_ops.mask_iou(empty, empty, empty_union=float("nan")))
    assert mask_ops.mask_iou(None, a) is None
    assert mask_ops.mask_iou(a, None) is None
    with pytest.raises(ValueError, match="mask shapes differ"):
        mask_ops.mask_iou(a, np.zeros((3, 4), dtype=bool))


def test_mask_iou_matches_the_direct_formula_bitwise() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = rng.random((30, 40)) > 0.6
        b = rng.random((30, 40)) > 0.6
        union = np.logical_or(a, b).sum()
        expected = float(np.logical_and(a, b).sum() / union)
        assert mask_ops.mask_iou(a, b) == expected


def test_mask_area_and_centroid() -> None:
    a, _, empty = _masks()
    assert mask_ops.mask_area(a) == 12
    assert mask_ops.mask_area(None) == 0
    assert mask_ops.mask_area(empty) == 0
    assert mask_ops.mask_centroid(a) == (2.5, 2.0)
    assert mask_ops.mask_centroid(a, pixel_center=True) == (3.0, 2.5)
    assert mask_ops.mask_centroid(a, pixel_center=True, scale=1.5) == (4.5, 3.75)
    assert mask_ops.mask_centroid(empty) is None
    assert mask_ops.mask_centroid(empty, pixel_center=True, scale=2.0) is None


def test_overlap_fraction_single_and_union() -> None:
    a, b, empty = _masks()
    assert mask_ops.overlap_fraction(a, b) == pytest.approx(4 / 12)
    assert mask_ops.overlap_fraction(empty, b) == 0.0
    assert math.isnan(mask_ops.overlap_fraction(empty, b, empty=float("nan")))
    assert mask_ops.overlap_fraction_union(a, [b]) == pytest.approx(4 / 12)
    assert mask_ops.overlap_fraction_union(a, [None, empty]) == 0.0
    assert mask_ops.overlap_fraction_union(a, [np.ones((2, 2), dtype=bool)]) == 0.0
    assert math.isnan(mask_ops.overlap_fraction_union(empty, [a]))
    assert math.isnan(mask_ops.overlap_fraction_union(None, [a]))


def test_decode_mask_png_is_mask_cache_decode(tmp_path) -> None:
    from PIL import Image

    from battle import mask_cache

    a, _, _ = _masks()
    path = tmp_path / "m.png"
    Image.fromarray(np.where(a, 255, 0).astype(np.uint8), mode="L").save(path)
    decoded = mask_ops.decode_mask_png(path)
    assert decoded.dtype == bool
    assert np.array_equal(decoded, a)
    assert np.array_equal(decoded, mask_cache.decode_mask_png(path))


# --------------------------------------------------------------------------- caller contracts


def test_interaction_review_iou_is_zero_on_an_empty_union() -> None:
    a, b, empty = _masks()
    assert interaction_review.mask_iou is mask_ops.mask_iou
    assert interaction_review.mask_iou(a, b) == pytest.approx(4 / 20)
    assert interaction_review.mask_iou(empty, empty) == 0.0


def test_ensemble_reference_iou_is_none_when_a_mask_is_missing_and_zero_when_empty() -> None:
    a, b, empty = _masks()
    assert ensemble_reference.mask_iou(a, b) == pytest.approx(4 / 20)
    assert ensemble_reference.mask_iou(None, b) is None
    assert ensemble_reference.mask_iou(a, None) is None
    assert ensemble_reference.mask_iou(empty, empty) == 0.0
    assert ensemble_reference.mask_area(a) == 12
    assert ensemble_reference.mask_area(None) == 0
    assert ensemble_reference.mask_centroid(a) == (2.5, 2.0)
    assert ensemble_reference.mask_centroid(empty) is None


def test_segmentation_disagreement_iou_labels_each_absence() -> None:
    a, b, empty = _masks()
    assert segmentation_disagreement.mask_iou(a, b) == (pytest.approx(4 / 20), "both_present")
    assert segmentation_disagreement.mask_iou(a, None) == (0.0, "missing_in_b")
    assert segmentation_disagreement.mask_iou(None, b) == (0.0, "missing_in_a")
    assert segmentation_disagreement.mask_iou(None, None) == (None, "both_missing")
    assert segmentation_disagreement.mask_iou(empty, empty) == (None, "both_missing")


def test_detector_scorecard_iou_is_nan_when_both_absent_zero_when_one_absent() -> None:
    a, b, empty = _masks()
    assert detector_scorecard._iou(a, b) == pytest.approx(4 / 20)
    assert math.isnan(detector_scorecard._iou(None, None))
    assert math.isnan(detector_scorecard._iou(empty, None))
    assert math.isnan(detector_scorecard._iou(empty, empty))
    assert detector_scorecard._iou(a, None) == 0.0
    assert detector_scorecard._iou(a, empty) == 0.0
    # A differently sized second mask is resized nearest onto the first.
    big = np.kron(b, np.ones((2, 2), dtype=bool))
    assert detector_scorecard._iou(a, big) == pytest.approx(4 / 20)


def test_egoexo_iou_is_none_on_an_empty_union_and_refuses_shape_mismatch() -> None:
    a, b, empty = _masks()
    assert egoexo_correspondence.iou(a, b) == pytest.approx(4 / 20)
    assert egoexo_correspondence.iou(empty, empty) is None
    assert egoexo_correspondence.iou(a, empty) == 0.0
    with pytest.raises(ValueError, match="mask shapes differ"):
        egoexo_correspondence.iou(a, np.zeros((3, 4), dtype=bool))


def test_multiview_seed_transfer_iou_is_zero_on_an_empty_union() -> None:
    a, b, empty = _masks()
    assert multiview_seed_transfer.iou(a, b) == pytest.approx(4 / 20)
    assert multiview_seed_transfer.iou(empty, empty) == 0.0


def test_multiview_seed_transfer_centroid_is_pixel_centred_and_refuses_empty() -> None:
    a, _, empty = _masks()
    centroid = multiview_seed_transfer.mask_centroid(a)
    assert isinstance(centroid, np.ndarray)
    assert centroid.tolist() == [3.0, 2.5]
    with pytest.raises(ValueError, match="mask is empty"):
        multiview_seed_transfer.mask_centroid(empty)


def test_multiview_consensus_raw_centroid_scales_the_pixel_centre() -> None:
    a, _, empty = _masks()
    centroid = multiview_consensus.mask_centroid_raw(a, 1.5)
    assert isinstance(centroid, np.ndarray)
    assert centroid.tolist() == [4.5, 3.75]
    assert multiview_consensus.mask_centroid_raw(empty, 1.0) is None


def test_overlap_fraction_callers_keep_their_empty_semantics() -> None:
    a, b, empty = _masks()
    assert review_metrics.overlap_fraction(a, b) == pytest.approx(4 / 12)
    assert review_metrics.overlap_fraction(empty, b) == 0.0
    assert correction_acceptance_search.overlap_fraction(a, b) == pytest.approx(4 / 12)
    assert correction_acceptance_search.overlap_fraction(empty, b) == 0.0
    assert detector_scorecard.overlap_fraction(a, [b]) == pytest.approx(4 / 12)
    assert detector_scorecard.overlap_fraction(a, [None, empty]) == 0.0
    assert math.isnan(detector_scorecard.overlap_fraction(empty, [a]))
    assert math.isnan(detector_scorecard.overlap_fraction(None, [a]))
