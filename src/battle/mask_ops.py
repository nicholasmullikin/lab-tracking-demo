"""Boolean-mask arithmetic shared by the builders: decode, area, centroid, IoU, overlap.

Every function takes masks as `bool` arrays of one image shape.  The per-module copies these
replace agreed on the arithmetic and differed only at the edges, so the edge behaviour is a
parameter here and each former copy is a one-line call with its own value:

IoU on a missing (`None`) mask, on an empty union (neither mask has a set pixel) and on a
shape mismatch, per former copy:

| former copy                          | missing mask   | empty union | shapes differ    |
|--------------------------------------|----------------|-------------|------------------|
| `interaction_review._mask_iou`       | not accepted   | 0.0         | numpy error      |
| `ensemble_reference.mask_iou`        | None           | 0.0         | numpy error      |
| `segmentation_disagreement.mask_iou` | labelled [1]   | labelled [1]| numpy error      |
| `detector_scorecard._iou`            | NaN / 0.0 [2]  | NaN         | resize b nearest |
| `egoexo_correspondence.iou`          | not accepted   | None        | ValueError       |
| `multiview_seed_transfer.iou`        | not accepted   | 0.0         | numpy error      |

[1] returns `(value, reason)`: one mask missing -> `(0.0, "missing_in_a" | "missing_in_b")`,
    both missing or an empty union -> `(None, "both_missing")`, else `(iou, "both_present")`.
[2] both missing -> NaN, one missing -> 0.0, where an all-False mask counts as missing.

:func:`mask_iou` returns None for a missing mask, `empty_union` (default 0.0; callers pass
None or NaN) for an empty union, and raises ValueError on a shape mismatch (numpy would have
raised or, worse, broadcast); `segmentation_disagreement` keeps its reason labels and
`detector_scorecard` its presence test and nearest resize as thin wrappers around it.

Centroids: `ensemble_reference.mask_centroid` returned `(mean x, mean y)` as floats or None
on an empty mask; `multiview_seed_transfer.mask_centroid` the same plus 0.5 (pixel centre) as
an array, raising on empty; `multiview_consensus.mask_centroid_raw` the pixel centre through
`cv2.moments`, scaled to raw pixels, None on empty.  :func:`mask_centroid` takes
`pixel_center` and `scale`; the moments and the mean agree bit for bit on all 7200 pm-append
masks (measured Sep 22), so one arithmetic serves all three.

Overlap: `review_metrics.overlap_fraction` and `correction_acceptance_search.overlap_fraction`
divide |mask & other| by |mask|, 0.0 on an empty mask; `detector_scorecard.overlap_fraction`
takes a sequence of others (None and mis-shaped entries skipped), NaN on an empty or missing
mask.  Two functions: :func:`overlap_fraction` (one region, `empty` parameter) and
:func:`overlap_fraction_union`.

Decoding: `mask_cache.decode_mask_png` (PIL `L` > 0) is the one PNG-to-bool reader; the cv2
`IMREAD_GRAYSCALE > 0` copies decode identically on all 7200 pm-append masks (measured Sep 22).
:func:`decode_mask_png` re-exports it so callers import one module for mask arithmetic.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .mask_cache import decode_mask_png as _decode_mask_png


def decode_mask_png(path: Path) -> np.ndarray:
    """Bool mask from a PNG: any non-zero luminance pixel is set (`mask_cache.decode_mask_png`)."""
    return _decode_mask_png(path)


def mask_area(mask: np.ndarray | None) -> int:
    """Number of set pixels; 0 for a missing mask."""
    return int(np.count_nonzero(mask)) if mask is not None else 0


def mask_centroid(
    mask: np.ndarray, *, pixel_center: bool = False, scale: float = 1.0
) -> tuple[float, float] | None:
    """`(x, y)` mean of the set pixel indices, or None when the mask is empty.

    `pixel_center` adds 0.5 to both coordinates (the centre of the index pixel rather than its
    corner) and `scale` multiplies the result, e.g. proxy pixels to raw sensor pixels.
    """
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    offset = 0.5 if pixel_center else 0.0
    return float((xs.mean() + offset) * scale), float((ys.mean() + offset) * scale)


def mask_iou(
    a: np.ndarray | None, b: np.ndarray | None, *, empty_union: float | None = 0.0
) -> float | None:
    """Intersection over union of two bool masks of one shape.

    None when either mask is None; `empty_union` when neither mask has a set pixel; ValueError
    when the shapes differ.
    """
    if a is None or b is None:
        return None
    if a.shape != b.shape:
        raise ValueError(f"mask shapes differ: {a.shape} vs {b.shape}")
    union = np.logical_or(a, b).sum()
    if union == 0:
        return empty_union
    return float(np.logical_and(a, b).sum() / union)


def overlap_fraction(mask: np.ndarray, region: np.ndarray, *, empty: float = 0.0) -> float:
    """|mask & region| / |mask|; `empty` when the mask has no set pixel."""
    total = int(np.count_nonzero(mask))
    if total == 0:
        return empty
    return float(np.count_nonzero(np.logical_and(mask, region)) / total)


def overlap_fraction_union(
    mask: np.ndarray | None, regions: Sequence[np.ndarray | None], *, empty: float = float("nan")
) -> float:
    """Fraction of `mask` covered by the union of `regions` of the same shape (None skipped).

    `empty` (NaN by default) when the mask is None or has no set pixel.
    """
    if mask is None or not mask.any():
        return empty
    shared = np.zeros(mask.shape, dtype=bool)
    for region in regions:
        if region is not None and region.shape == mask.shape:
            shared |= region
    return overlap_fraction(mask, shared, empty=empty)
