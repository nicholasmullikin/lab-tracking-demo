"""Label-free review metrics and ranked review triggers for the first-minute review.

This module turns the already retained v4 review inputs (corrected SAM3 masks, stabilized
and raw WiLoR, MediaPipe, BoxMOT, fused Kineo, contact diagnostics, agent-authored substeps,
coarse GT) into per-frame proxy metrics and ranked review episodes.  It runs no model, never
touches the GPU, and asserts nothing about accuracy: every output is a prompt for a human
to look at a frame range, not a verdict about what the masks or hands really are.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import interaction_review as review
from .exploratory_comparison import _file_fingerprint, validate_artifact_fingerprint
from .fine_substep_contract import load_contract as load_fine_substep_contract
from .fine_substep_contract import substep_for_frame
from .four_part_contract import ANALYSIS_FPS, TARGETS
from .interaction_review_v4 import (
    COARSE_GT,
    FINE_LABELS,
    SEGMENTATION_CONTACT_ELIGIBLE_THROUGH,
    SOURCE_START_SECONDS,
)
from .mask_ops import mask_iou, overlap_fraction
from .review_metrics_schemas import (
    ColorBand,
    ContactIntervalRecord,
    ContactSummary,
    HandGapFrameProxy,
    HandGapRecord,
    HandJitterFrameMetric,
    KineoFrameMetric,
    KineoProvenanceSummary,
    PartAppearanceFrameMetric,
    PartFrameStat,
    PartGrowthFrameMetric,
    PartPairFrameMetric,
    PhaseJitterSummary,
    ReviewMetricsConfig,
    ReviewMetricsManifest,
    ReviewTriggerEpisode,
    StabilizedHandProvenanceRow,
)
from .schemas import (
    ArtifactFingerprint,
    FrameObservations,
    InteractionContactDiagnostic,
    InteractionReviewIndexManifest,
    KineoBoxProvenance,
    TimeInterval,
)

FRAME_COUNT = 1800
OUTPUT_ROOT = Path("runs/review-metrics-first-minute-v2")
CONFIG_PATH = Path("configs/review_metrics/first_minute_v1.json")
V4_INDEX_PATH = Path("runs/interaction-review-first-minute-v4/interaction_review_index.json")
METRICS_NAME = "metrics.json"
TRIGGERS_NAME = "triggers.json"
REPORT_NAME = "metrics_report.md"
RRD_NAME = "review_metrics_first_minute.rrd"
# Reference segmentation methods the metrics accept; the run directory itself always comes
# from the v4 index so the metrics measure exactly the masks the review package displays.
ACCEPTED_REFERENCE_METHODS = ("baseline_sam3", "ensemble_reference")
SHEET_DIR = "contact_sheets"
OVERVIEW_SHEET_NAME = "top_episodes_contact_sheet.png"
PART_PAIRS: tuple[tuple[str, str], ...] = tuple(combinations(TARGETS, 2))
DARK_PARTS = ("chassis", "interior")
YELLOW_PARTS = ("rear_body", "cabin")
PART_COLORS = {
    "chassis": (70, 130, 255),
    "interior": (60, 210, 150),
    "rear_body": (255, 190, 45),
    "cabin": (255, 95, 100),
}
TIP_INDICES = review.TIP_INDICES
SKIN_SAMPLE_STRIDE = 5
SKIN_SAMPLE_CAP = 2_000_000
BACKGROUND_STRIDE = 8
CONTEXT_FRAMES = 15
CLAIM_BOUNDARIES = (
    "All metrics are label-free geometry, appearance, and provenance proxies, not accuracy.",
    "Episodes are ranked review prompts with frame ranges; none decides mask or hand semantics.",
    "Skin-color evidence near a last known pose is a visibility proxy, not a hand detection.",
    "Expected touched parts are agent-authored assumptions on agent-authored substeps.",
    "Contact intervals are debounced geometry candidates from the v4 index, never touch labels.",
)


def _r(value: float, digits: int = 4) -> float:
    """Round a finite float for compact JSON; NaN/inf are programming errors here."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("metric values must be finite; represent absence with None")
    return round(result, digits)


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def cluster_frames(frames: Iterable[int], max_gap: int) -> list[tuple[int, int]]:
    """Group frame indices into inclusive (start, end) runs.

    Two hits belong to one run when at most `max_gap` non-hit frames separate them, so
    `max_gap=0` clusters exactly consecutive frames.
    """
    if max_gap < 0:
        raise ValueError("max_gap must be non-negative")
    runs: list[tuple[int, int]] = []
    start = previous = None
    for frame in sorted(set(frames)):
        if start is None or previous is None:
            start = previous = frame
        elif frame - previous > max_gap + 1:
            runs.append((start, previous))
            start = previous = frame
        else:
            previous = frame
    if start is not None:
        runs.append((start, previous))
    return runs


def mask_stats(mask: np.ndarray) -> tuple[int, float, float] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return int(xs.size), float(xs.mean()), float(ys.mean())


def box_pixels(box: object, dimensions: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = dimensions
    x0 = int(math.floor(getattr(box, "x") * width))
    y0 = int(math.floor(getattr(box, "y") * height))
    x1 = int(math.ceil((getattr(box, "x") + getattr(box, "width")) * width))
    y1 = int(math.ceil((getattr(box, "y") + getattr(box, "height")) * height))
    return max(0, x0), max(0, y0), min(width, max(x0 + 1, x1)), min(height, max(y0 + 1, y1))


def hand_scale_pixels(hand: object, dimensions: tuple[int, int]) -> float:
    box = getattr(hand, "box")
    return max(1.0, math.sqrt(box.width * dimensions[0] * box.height * dimensions[1]))


def hand_box_union(hands: Sequence[object], dimensions: tuple[int, int]) -> np.ndarray:
    union = np.zeros((dimensions[1], dimensions[0]), dtype=bool)
    for hand in hands:
        x0, y0, x1, y1 = box_pixels(getattr(hand, "box"), dimensions)
        union[y0:y1, x0:x1] = True
    return union


# --------------------------------------------------------------------------------------
# 1. Segmentation identity-swap geometry
# --------------------------------------------------------------------------------------


@dataclass
class FrameGeometry:
    """Raw per-frame part geometry collected during the single I/O pass."""

    frame: int
    areas: dict[str, int | None]
    centroids: dict[str, tuple[float, float] | None]
    pair_iou: dict[tuple[str, str], float | None]
    exchange_iou: dict[tuple[str, str], float | None] = field(default_factory=dict)


def frame_geometry(
    frame: int,
    masks: dict[str, np.ndarray | None],
    previous: dict[str, np.ndarray | None] | None,
) -> FrameGeometry:
    areas: dict[str, int | None] = {}
    centroids: dict[str, tuple[float, float] | None] = {}
    for part in TARGETS:
        stats = mask_stats(masks[part]) if masks.get(part) is not None else None
        areas[part] = stats[0] if stats else None
        centroids[part] = (stats[1], stats[2]) if stats else None
    pair_iou: dict[tuple[str, str], float | None] = {}
    exchange: dict[tuple[str, str], float | None] = {}
    for a, b in PART_PAIRS:
        left, right = masks.get(a), masks.get(b)
        pair_iou[(a, b)] = mask_iou(left, right) if left is not None and right is not None else None
        prior_a = previous.get(a) if previous else None
        prior_b = previous.get(b) if previous else None
        if left is not None and right is not None and prior_a is not None and prior_b is not None:
            exchange[(a, b)] = min(mask_iou(left, prior_b), mask_iou(right, prior_a))
        else:
            exchange[(a, b)] = None
    return FrameGeometry(frame, areas, centroids, pair_iou, exchange)


def part_area_medians(geometries: Sequence[FrameGeometry]) -> dict[str, float]:
    medians: dict[str, float] = {}
    for part in TARGETS:
        values = [item.areas[part] for item in geometries if item.areas[part] is not None]
        medians[part] = float(np.median(values)) if values else 0.0
    return medians


def part_stats(
    geometries: Sequence[FrameGeometry], medians: dict[str, float]
) -> tuple[PartFrameStat, ...]:
    records: list[PartFrameStat] = []
    for item in geometries:
        for part in TARGETS:
            area = item.areas[part]
            centroid = item.centroids[part]
            if area is None or centroid is None:
                records.append(
                    PartFrameStat(
                        analysis_frame_index=item.frame, part_id=part, state="missing_mask"
                    )
                )
                continue
            records.append(
                PartFrameStat(
                    analysis_frame_index=item.frame,
                    part_id=part,
                    state="observed",
                    area_pixels=area,
                    centroid_x=_r(centroid[0], 2),
                    centroid_y=_r(centroid[1], 2),
                    area_ratio_vs_median=_r(area / medians[part]) if medians[part] else 0.0,
                )
            )
    return tuple(records)


def _absorption(ratio_a: float, ratio_b: float, distance: float, area_b: int) -> float:
    """Score for `b` having grown into the footprint of a collapsing `a`."""
    growth = _clip01(ratio_b - 1.0)
    collapse = _clip01((0.5 - ratio_a) / 0.5)
    proximity = _clip01(1.0 - distance / (2.0 * math.sqrt(max(1, area_b))))
    return growth * collapse * proximity


def pair_metrics(
    geometries: Sequence[FrameGeometry],
    medians: dict[str, float],
    config: ReviewMetricsConfig,
) -> tuple[PartPairFrameMetric, ...]:
    """Per-frame swap evidence for every unordered part pair."""
    by_frame = {item.frame: item for item in geometries}
    records: list[PartPairFrameMetric] = []
    window = config.crossing_window_frames
    for item in geometries:
        for a, b in PART_PAIRS:
            area_a, area_b = item.areas[a], item.areas[b]
            ca, cb = item.centroids[a], item.centroids[b]
            if area_a is None or area_b is None or ca is None or cb is None:
                records.append(
                    PartPairFrameMetric(
                        analysis_frame_index=item.frame, part_a=a, part_b=b, state="missing_mask"
                    )
                )
                continue
            ratio_a = area_a / medians[a] if medians[a] else 0.0
            ratio_b = area_b / medians[b] if medians[b] else 0.0
            iou = item.pair_iou[(a, b)] or 0.0
            distance = math.hypot(ca[0] - cb[0], ca[1] - cb[1])
            large_a = _clip01(ratio_a / config.swap_large_area_fraction)
            large_b = _clip01(ratio_b / config.swap_large_area_fraction)
            convergence = iou * large_a * large_b
            absorb_by_b = _absorption(ratio_a, ratio_b, distance, area_b)
            absorb_by_a = _absorption(ratio_b, ratio_a, distance, area_a)
            absorbing = None
            absorption = max(absorb_by_a, absorb_by_b)
            if absorption > 0:
                absorbing = b if absorb_by_b >= absorb_by_a else a
            exchange = item.exchange_iou.get((a, b))
            crossing = _label_crossing(by_frame, item.frame, a, b, medians, config, window)
            swap = max(convergence, absorption, exchange or 0.0)
            records.append(
                PartPairFrameMetric(
                    analysis_frame_index=item.frame,
                    part_a=a,
                    part_b=b,
                    state="observed",
                    iou=_r(iou),
                    centroid_distance_pixels=_r(distance, 2),
                    exchange_score=None if exchange is None else _r(exchange),
                    convergence_score=_r(convergence),
                    absorption_score=_r(absorption),
                    absorbing_part=absorbing,
                    swap_score=_r(swap),
                    label_crossing=crossing,
                )
            )
    return tuple(records)


def _label_crossing(
    by_frame: dict[int, FrameGeometry],
    frame: int,
    a: str,
    b: str,
    medians: dict[str, float],
    config: ReviewMetricsConfig,
    window: int,
) -> bool:
    """Two centroid trajectories swap sides over `window` frames while both stay large."""
    earlier = by_frame.get(frame - window)
    current = by_frame[frame]
    if earlier is None:
        return False
    for step in range(frame - window, frame + 1):
        item = by_frame.get(step)
        if item is None:
            return False
        for part in (a, b):
            area = item.areas[part]
            if area is None or item.centroids[part] is None:
                return False
            if medians[part] and area < config.crossing_min_area_fraction * medians[part]:
                return False
    ca0, cb0 = earlier.centroids[a], earlier.centroids[b]
    ca1, cb1 = current.centroids[a], current.centroids[b]
    assert ca0 and cb0 and ca1 and cb1
    d0 = np.asarray(ca0) - np.asarray(cb0)
    d1 = np.asarray(ca1) - np.asarray(cb1)
    if (
        np.linalg.norm(d0) < config.crossing_min_separation_pixels
        or np.linalg.norm(d1) < config.crossing_min_separation_pixels
    ):
        return False
    return bool(np.dot(d0, d1) < 0)


# --------------------------------------------------------------------------------------
# 2. Appearance consistency
# --------------------------------------------------------------------------------------


def derive_yellow_band(hsv: np.ndarray, masks: dict[str, np.ndarray | None]) -> ColorBand:
    """Derive the yellow band from the frame-0 rear_body/cabin masks, not from a constant."""
    pixels = [hsv[masks[part]] for part in YELLOW_PARTS if masks.get(part) is not None]
    if not pixels:
        raise ValueError("frame 0 must contain at least one yellow-part mask")
    sample = np.concatenate(pixels).astype(np.float64)
    hue = sample[:, 0]
    wraps = False
    if np.percentile(hue, 95) - np.percentile(hue, 5) > 90:
        # Hue spans the 0/180 seam; rotate by half a turn to measure the compact range.
        hue = (hue + 90) % 180
        wraps = True
    low_h, high_h = np.percentile(hue, 5), np.percentile(hue, 95)
    if wraps:
        low_h, high_h = (low_h - 90) % 180, (high_h - 90) % 180
    return ColorBand(
        space="hsv",
        channel_low=(
            int(low_h),
            int(np.percentile(sample[:, 1], 10)),
            int(np.percentile(sample[:, 2], 10)),
        ),
        channel_high=(int(high_h), 255, 255),
        hue_wraps=wraps,
        derived_from=(
            "frame-0 rear_body and cabin corrected-SAM3 mask pixels (HSV p5/p95 hue, p10 S/V)"
        ),
        sample_pixels=int(sample.shape[0]),
    )


def band_membership(image: np.ndarray, band: ColorBand) -> np.ndarray:
    """Boolean membership for an image already in the band's color space."""
    low = np.asarray(band.channel_low)
    high = np.asarray(band.channel_high)
    first = image[..., 0]
    if band.hue_wraps:
        hue_ok = (first >= low[0]) | (first <= high[0])
    else:
        hue_ok = (first >= low[0]) & (first <= high[0])
    rest = np.all((image[..., 1:] >= low[1:]) & (image[..., 1:] <= high[1:]), axis=-1)
    return hue_ok & rest


def appearance_metrics_for_frame(
    frame: int,
    hsv: np.ndarray,
    lab: np.ndarray,
    masks: dict[str, np.ndarray | None],
    hand_union: np.ndarray,
    frame0_lab_means: dict[str, np.ndarray],
    yellow: np.ndarray,
    config: ReviewMetricsConfig,
) -> list[PartAppearanceFrameMetric]:
    records: list[PartAppearanceFrameMetric] = []
    for part in TARGETS:
        mask = masks.get(part)
        if mask is None or not mask.any():
            records.append(
                PartAppearanceFrameMetric(
                    analysis_frame_index=frame, part_id=part, state="missing_mask"
                )
            )
            continue
        hsv_pixels = hsv[mask]
        lab_mean = lab[mask].astype(np.float64).mean(axis=0)
        reference = frame0_lab_means.get(part)
        distance = float(np.linalg.norm(lab_mean - reference)) if reference is not None else 0.0
        yellow_fraction = float(yellow[mask].mean())
        hand_fraction = overlap_fraction(mask, hand_union)
        median = np.median(hsv_pixels, axis=0)
        # The robust z and the leakage flag need the whole-minute distribution; see
        # `finalize_appearance`.  Provisional zeros keep the record NaN-free meanwhile.
        records.append(
            PartAppearanceFrameMetric(
                analysis_frame_index=frame,
                part_id=part,
                state="observed",
                median_h=_r(median[0], 1),
                median_s=_r(median[1], 1),
                median_v=_r(median[2], 1),
                mean_l=_r(lab_mean[0], 1),
                mean_a=_r(lab_mean[1], 1),
                mean_b=_r(lab_mean[2], 1),
                lab_distance_from_frame0=_r(distance, 2),
                lab_distance_robust_z=0.0,
                yellow_fraction=_r(yellow_fraction),
                hand_overlap_fraction=_r(hand_fraction),
                leakage_suspect=False,
            )
        )
    return records


def robust_z(values: np.ndarray) -> np.ndarray:
    """Median/MAD z-score; a degenerate MAD falls back to a tiny scale, never NaN."""
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad if mad > 0 else 1e-6
    return (values - median) / scale


def finalize_appearance(
    records: Sequence[PartAppearanceFrameMetric], config: ReviewMetricsConfig
) -> tuple[PartAppearanceFrameMetric, ...]:
    """Flag leakage against each part's own distance distribution, not hand occlusion.

    A hand box covering a part is legitimate handling and is kept only as context; leakage is
    a color shift that is both large in absolute Lab terms and unusual for that part over the
    minute, or a dark part turning yellow.
    """
    output = list(records)
    for part in TARGETS:
        indices = [
            i for i, item in enumerate(output) if item.part_id == part and item.state == "observed"
        ]
        if not indices:
            continue
        distances = np.asarray(
            [output[i].lab_distance_from_frame0 or 0.0 for i in indices], dtype=float
        )
        zs = robust_z(distances)
        for i, z in zip(indices, zs, strict=True):
            item = output[i]
            drift = (
                z >= config.appearance_robust_z_threshold
                and float(item.lab_distance_from_frame0 or 0.0)
                >= config.appearance_distance_threshold
            )
            yellow = part in DARK_PARTS and (
                float(item.yellow_fraction or 0.0) > config.yellow_fraction_threshold_dark_parts
            )
            output[i] = item.model_copy(
                update={"lab_distance_robust_z": _r(z, 2), "leakage_suspect": bool(drift or yellow)}
            )
    return tuple(output)


# --------------------------------------------------------------------------------------
# 3. Mask growth vs hand proximity
# --------------------------------------------------------------------------------------


def growth_metrics_for_frame(
    current: FrameGeometry,
    previous: FrameGeometry | None,
    masks: dict[str, np.ndarray | None],
    hand_union: np.ndarray,
    config: ReviewMetricsConfig,
    window: FrameGeometry | None = None,
) -> list[PartGrowthFrameMetric]:
    """Frame-to-frame and windowed area growth, classified by stabilized-hand overlap.

    `window` is the geometry `growth_window_frames` earlier; it lets a slow ramp that never
    exceeds the per-frame threshold still register as growth.
    """
    records: list[PartGrowthFrameMetric] = []
    for part in TARGETS:
        mask = masks.get(part)
        area = current.areas[part]
        centroid = current.centroids[part]
        if mask is None or area is None or centroid is None:
            records.append(
                PartGrowthFrameMetric(
                    analysis_frame_index=current.frame, part_id=part, state="missing_mask"
                )
            )
            continue
        prior_area = previous.areas[part] if previous else None
        prior_centroid = previous.centroids[part] if previous else None
        if prior_area is None or prior_centroid is None:
            records.append(
                PartGrowthFrameMetric(
                    analysis_frame_index=current.frame, part_id=part, state="no_previous_mask"
                )
            )
            continue
        ratio = area / max(1, prior_area)
        window_area = window.areas[part] if window else None
        window_ratio = area / max(1, window_area) if window_area is not None else None
        velocity = math.hypot(centroid[0] - prior_centroid[0], centroid[1] - prior_centroid[1])
        hand_fraction = overlap_fraction(mask, hand_union)
        overlaps = hand_fraction > 0.0
        growth = ratio >= config.growth_area_ratio_threshold or (
            window_ratio is not None and window_ratio >= config.growth_window_ratio_threshold
        )
        growth_class = None
        if growth:
            growth_class = (
                "hand_capture_suspect"
                if hand_fraction >= config.hand_capture_overlap_fraction
                else "unexplained"
            )
        records.append(
            PartGrowthFrameMetric(
                analysis_frame_index=current.frame,
                part_id=part,
                state="observed",
                area_ratio_vs_previous=_r(ratio),
                area_ratio_vs_window=None if window_ratio is None else _r(window_ratio),
                centroid_velocity_pixels=_r(velocity, 2),
                hand_box_overlaps_mask=bool(overlaps),
                hand_overlap_fraction=_r(hand_fraction),
                growth_event=bool(growth),
                growth_class=growth_class,
            )
        )
    return records


# --------------------------------------------------------------------------------------
# 4. Hands: gaps, visibility proxy, re-entry, jitter
# --------------------------------------------------------------------------------------


@dataclass
class GapWindowSample:
    """Deferred crop so the skin band can be derived after the single video pass."""

    frame: int
    last_known_frame: int
    hand_id: str
    center: tuple[int, int]
    size: int
    crop_ycrcb: np.ndarray
    reference_ycrcb: np.ndarray
    background_ycrcb: np.ndarray
    fingertips_in_frame: int


def hand_gaps(
    observations: dict[int, FrameObservations], frame_count: int
) -> list[tuple[int, int]]:
    """Maximal runs of frames with no stabilized hand at all, as (start, end_exclusive)."""
    missing = [frame for frame in range(frame_count) if not observations[frame].hands]
    return [(start, end + 1) for start, end in cluster_frames(missing, 0)]


def fingertips_in_frame(hand: object, margin: float = 0.005) -> int:
    count = 0
    for index in TIP_INDICES:
        point = getattr(hand, "landmarks")[index]
        if margin <= point.x <= 1 - margin and margin <= point.y <= 1 - margin:
            count += 1
    return count


def palm_center_pixels(hand: object, dimensions: tuple[int, int]) -> tuple[int, int]:
    points = np.asarray([(p.x, p.y) for p in getattr(hand, "landmarks")], dtype=np.float64)
    center = points.mean(axis=0)
    return (
        int(min(dimensions[0] - 1, max(0, round(center[0] * (dimensions[0] - 1))))),
        int(min(dimensions[1] - 1, max(0, round(center[1] * (dimensions[1] - 1))))),
    )


def crop_window(image: np.ndarray, center: tuple[int, int], size: int) -> np.ndarray:
    half = max(1, size // 2)
    x0, x1 = max(0, center[0] - half), min(image.shape[1], center[0] + half + 1)
    y0, y1 = max(0, center[1] - half), min(image.shape[0], center[1] + half + 1)
    return image[y0:y1, x0:x1]


def derive_skin_band(
    samples: Sequence[np.ndarray], background: Sequence[np.ndarray] = ()
) -> ColorBand:
    """Percentile YCrCb band from confident raw-WiLoR hand-box centers; a proxy only.

    `background` pixels (frame content outside hand boxes) are only measured against the
    band so the report can state how skin-like the scene itself is.
    """
    if not samples:
        raise ValueError("skin calibration requires at least one confident hand sample")
    pixels = np.concatenate([item.reshape(-1, 3) for item in samples]).astype(np.float64)
    band = ColorBand(
        space="ycrcb",
        channel_low=(
            int(np.percentile(pixels[:, 0], 5)),
            int(np.percentile(pixels[:, 1], 10)),
            int(np.percentile(pixels[:, 2], 10)),
        ),
        channel_high=(
            255,
            int(np.percentile(pixels[:, 1], 90)),
            int(np.percentile(pixels[:, 2], 90)),
        ),
        derived_from=(
            "central 50% of raw-WiLoR hand boxes with confidence >= configured minimum, "
            "every 5th frame (YCrCb p5 Y floor, p10/p90 Cr and Cb)"
        ),
        sample_pixels=int(pixels.shape[0]),
    )
    if background:
        scene = np.concatenate([item.reshape(-1, 3) for item in background])
        band = band.model_copy(
            update={"background_fraction": _r(float(band_membership(scene, band).mean()))}
        )
    return band


def gap_proxies(
    samples: Sequence[GapWindowSample], skin: ColorBand, config: ReviewMetricsConfig
) -> tuple[HandGapFrameProxy, ...]:
    records: list[HandGapFrameProxy] = []
    for item in samples:
        skin_fraction = (
            float(band_membership(item.crop_ycrcb, skin).mean()) if item.crop_ycrcb.size else 0.0
        )
        reference = (
            float(band_membership(item.reference_ycrcb, skin).mean())
            if item.reference_ycrcb.size
            else 0.0
        )
        background = (
            float(band_membership(item.background_ycrcb, skin).mean())
            if item.background_ycrcb.size
            else 0.0
        )
        suspect = (
            skin_fraction >= config.skin_fraction_min
            and skin_fraction >= config.skin_fraction_reference_ratio * reference
            and skin_fraction >= config.skin_background_ratio * background
            and item.fingertips_in_frame >= config.fingertips_in_frame_min
        )
        records.append(
            HandGapFrameProxy(
                analysis_frame_index=item.frame,
                last_known_frame=item.last_known_frame,
                last_known_hand_id=item.hand_id,
                window_center_x=item.center[0],
                window_center_y=item.center[1],
                window_size_pixels=item.size,
                skin_fraction_in_window=_r(skin_fraction),
                reference_skin_fraction=_r(reference),
                background_skin_fraction=_r(background),
                fingertips_in_frame_count=item.fingertips_in_frame,
                visible_but_undetected_suspect=bool(suspect),
            )
        )
    return tuple(records)


def gap_records(
    observations: dict[int, FrameObservations],
    gaps: Sequence[tuple[int, int]],
    proxies: Sequence[HandGapFrameProxy],
    dimensions: tuple[int, int],
) -> tuple[HandGapRecord, ...]:
    """Re-entry jump: nearest re-entering wrist to any last known wrist, per gap."""
    suspects: dict[tuple[int, int], int] = defaultdict(int)
    for gap in gaps:
        suspects[gap] = sum(
            gap[0] <= item.analysis_frame_index < gap[1] and item.visible_but_undetected_suspect
            for item in proxies
        )
    records: list[HandGapRecord] = []
    for start, end in gaps:
        last_known = start - 1 if start > 0 and observations[start - 1].hands else None
        reentry = end if end in observations and observations[end].hands else None
        jump = normalized = None
        if last_known is not None and reentry is not None:
            previous = [
                (
                    np.asarray(review._pixel(h.landmarks[0], dimensions), dtype=float),
                    hand_scale_pixels(h, dimensions),
                )
                for h in observations[last_known].hands
            ]
            entering = [
                np.asarray(review._pixel(h.landmarks[0], dimensions), dtype=float)
                for h in observations[reentry].hands
            ]
            best = min(
                (float(np.linalg.norm(new - old)), scale)
                for old, scale in previous
                for new in entering
            )
            jump, normalized = _r(best[0], 2), _r(best[0] / best[1])
        records.append(
            HandGapRecord(
                start_frame=start,
                end_frame_exclusive=end,
                length_frames=end - start,
                last_known_frame=last_known,
                reentry_frame=reentry,
                reentry_jump_pixels=jump,
                reentry_jump_normalized=normalized,
                suspect_frames=suspects[(start, end)],
            )
        )
    return tuple(records)


def jitter_metrics(
    observations: dict[int, FrameObservations],
    provenance: dict[tuple[int, str], str],
    dimensions: tuple[int, int],
    frame_count: int,
) -> tuple[HandJitterFrameMetric, ...]:
    """Nearest previous-frame wrist displacement per hand, normalized by that hand's scale."""
    records: list[HandJitterFrameMetric] = []
    for frame in range(frame_count):
        hands = observations[frame].hands
        previous = observations[frame - 1].hands if frame > 0 else ()
        prior_wrists = [
            np.asarray(review._pixel(h.landmarks[0], dimensions), dtype=float) for h in previous
        ]
        for hand in hands:
            wrist = np.asarray(review._pixel(hand.landmarks[0], dimensions), dtype=float)
            scale = hand_scale_pixels(hand, dimensions)
            displacement = (
                min(float(np.linalg.norm(wrist - old)) for old in prior_wrists)
                if prior_wrists
                else None
            )
            records.append(
                HandJitterFrameMetric(
                    analysis_frame_index=frame,
                    hand_id=hand.hand_id,
                    provenance_state=provenance.get((frame, hand.hand_id), "unknown"),
                    hand_scale_pixels=_r(scale, 2),
                    wrist_displacement_pixels=None if displacement is None else _r(displacement, 2),
                    jitter_normalized=None if displacement is None else _r(displacement / scale),
                )
            )
    return tuple(records)


def jitter_by_phase(
    jitter: Sequence[HandJitterFrameMetric],
    phases: Sequence[tuple[str, str, int, int]],
    observations: dict[int, FrameObservations],
) -> tuple[PhaseJitterSummary, ...]:
    summaries: list[PhaseJitterSummary] = []
    for kind, phase_id, start, end in phases:
        values = [
            item.jitter_normalized
            for item in jitter
            if start <= item.analysis_frame_index < end and item.jitter_normalized is not None
        ]
        array = np.asarray(values, dtype=float)
        summaries.append(
            PhaseJitterSummary(
                phase_kind=kind,  # type: ignore[arg-type]
                phase_id=phase_id,
                start_frame=start,
                end_frame_exclusive=end,
                sample_count=len(values),
                median_jitter_normalized=_r(np.median(array)) if values else None,
                p90_jitter_normalized=_r(np.percentile(array, 90)) if values else None,
                mean_jitter_normalized=_r(array.mean()) if values else None,
                missing_hand_frames=sum(
                    not observations[frame].hands
                    for frame in range(start, end)
                    if frame in observations
                ),
            )
        )
    return tuple(summaries)


# --------------------------------------------------------------------------------------
# 5. Contact intervals
# --------------------------------------------------------------------------------------


def contact_intervals(
    diagnostics: Sequence[InteractionContactDiagnostic],
    *,
    frame_count: int,
    substep_lookup: dict[int, str],
    expected_parts: dict[str, tuple[str, ...]],
    coarse_gt: Sequence[tuple[str, int, int]],
    flicker_max_frames: int,
) -> tuple[ContactIntervalRecord, ...]:
    """Runs of debounced candidates per (lane, part); a run is closed when it ends observed."""
    series: dict[tuple[str, str], dict[int, bool | None]] = defaultdict(dict)
    for item in diagnostics:
        series[(item.hand_source_id, item.part_id)][item.analysis_frame_index] = (
            item.debounced_contact_candidate if item.observation_state == "observed" else None
        )
    records: list[ContactIntervalRecord] = []
    for (lane, part), values in sorted(series.items()):
        start: int | None = None
        for frame in range(frame_count + 1):
            value = values.get(frame) if frame < frame_count else None
            if value is True and start is None:
                start = frame
            elif value is not True and start is not None:
                closed = value is False
                substep = substep_lookup.get(start)
                expected = None if substep is None else part in expected_parts.get(substep, ())
                action = next((name for name, s, e in coarse_gt if s <= start < e), None)
                records.append(
                    ContactIntervalRecord(
                        hand_source_id=lane,
                        part_id=part,  # type: ignore[arg-type]
                        start_frame=start,
                        end_frame_exclusive=frame,
                        duration_frames=frame - start,
                        closed=closed,
                        substep_id=substep,
                        expected_for_substep=expected,
                        coarse_gt_action=action,
                        flicker=frame - start < flicker_max_frames,
                    )
                )
                start = None
    return tuple(
        sorted(records, key=lambda item: (item.start_frame, item.hand_source_id, item.part_id))
    )


def contact_summary(intervals: Sequence[ContactIntervalRecord]) -> ContactSummary:
    durations = np.asarray([item.duration_frames for item in intervals], dtype=float)
    bins = ((1, 4), (5, 14), (15, 29), (30, 89), (90, 10**6))
    histogram = {
        f"{low}-{high if high < 10**6 else 'inf'}": int(
            ((durations >= low) & (durations <= high)).sum()
        )
        for low, high in bins
    }
    per_substep: dict[str, int] = defaultdict(int)
    for item in intervals:
        if item.expected_for_substep is False and item.substep_id is not None:
            per_substep[item.substep_id] += 1
    return ContactSummary(
        interval_count=len(intervals),
        sub_5_frame_count=int((durations < 5).sum()) if intervals else 0,
        duration_min=int(durations.min()) if intervals else None,
        duration_median=_r(np.median(durations)) if intervals else None,
        duration_p90=_r(np.percentile(durations, 90)) if intervals else None,
        duration_max=int(durations.max()) if intervals else None,
        duration_histogram=histogram,
        expected_count=sum(item.expected_for_substep is True for item in intervals),
        unexpected_count=sum(item.expected_for_substep is False for item in intervals),
        unknown_substep_count=sum(item.expected_for_substep is None for item in intervals),
        per_substep_unexpected=dict(sorted(per_substep.items())),
    )


# --------------------------------------------------------------------------------------
# 6. Kineo by box provenance
# --------------------------------------------------------------------------------------


def kineo_metrics(
    observations: dict[int, FrameObservations],
    provenance: dict[int, KineoBoxProvenance],
    dimensions: tuple[int, int],
    frame_count: int,
) -> tuple[KineoFrameMetric, ...]:
    records: list[KineoFrameMetric] = []
    previous: dict[str, np.ndarray] = {}
    for frame in range(frame_count):
        bodies = observations[frame].nlf_body_2d
        source = provenance[frame].source if frame in provenance else "missing"
        if not bodies:
            records.append(
                KineoFrameMetric(analysis_frame_index=frame, box_source=source, body_present=False)
            )
            previous = {}
            continue
        current: dict[str, np.ndarray] = {}
        confidences: list[float] = []
        jitters: list[float] = []
        for body in bodies:
            points = np.asarray(
                [(p.x * dimensions[0], p.y * dimensions[1]) for p in body.landmarks], dtype=float
            )
            current[body.subject_id] = points
            confidences.append(float(np.mean([p.confidence for p in body.landmarks])))
            prior = previous.get(body.subject_id)
            if prior is not None and prior.shape == points.shape:
                jitters.append(float(np.linalg.norm(points - prior, axis=1).mean()))
        box = provenance[frame].box if frame in provenance else None
        height = box.height * dimensions[1] if box is not None else None
        jitter = float(np.mean(jitters)) if jitters else None
        records.append(
            KineoFrameMetric(
                analysis_frame_index=frame,
                box_source=source,
                body_present=True,
                mean_joint_confidence=_r(float(np.mean(confidences))),
                joint_jitter_pixels=None if jitter is None else _r(jitter, 2),
                joint_jitter_normalized=(
                    None if jitter is None or not height else _r(jitter / height)
                ),
            )
        )
        previous = current
    return tuple(records)


def kineo_by_provenance(metrics: Sequence[KineoFrameMetric]) -> tuple[KineoProvenanceSummary, ...]:
    summaries: list[KineoProvenanceSummary] = []
    for source in ("detected_native", "boxmot_fallback", "interpolated", "held", "missing"):
        rows = [item for item in metrics if item.box_source == source]
        confidences = [
            item.mean_joint_confidence for item in rows if item.mean_joint_confidence is not None
        ]
        jitters = [
            item.joint_jitter_pixels for item in rows if item.joint_jitter_pixels is not None
        ]
        normalized = [
            item.joint_jitter_normalized
            for item in rows
            if item.joint_jitter_normalized is not None
        ]
        summaries.append(
            KineoProvenanceSummary(
                box_source=source,  # type: ignore[arg-type]
                frame_count=len(rows),
                body_frames=sum(item.body_present for item in rows),
                mean_joint_confidence=_r(np.mean(confidences)) if confidences else None,
                jitter_sample_count=len(jitters),
                median_jitter_pixels=_r(np.median(jitters), 2) if jitters else None,
                p90_jitter_pixels=_r(np.percentile(jitters, 90), 2) if jitters else None,
                median_jitter_normalized=_r(np.median(normalized)) if normalized else None,
            )
        )
    return tuple(summaries)


# --------------------------------------------------------------------------------------
# 7. Episodes and ranking
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DraftEpisode:
    episode_type: str
    start_frame: int
    end_frame: int
    peak_frame: int
    score: float
    subjects: tuple[str, ...]
    rationale: str


def _episodes_from_scores(
    episode_type: str,
    subjects: tuple[str, ...],
    scores: dict[int, float],
    threshold: float,
    max_gap: int,
    rationale: str,
) -> list[DraftEpisode]:
    """Cluster frames whose score meets `threshold`; score is reported in threshold units."""
    hits = [frame for frame, value in scores.items() if value >= threshold]
    drafts: list[DraftEpisode] = []
    for start, end in cluster_frames(hits, max_gap):
        peak = max(range(start, end + 1), key=lambda f: (scores.get(f, -1.0), -f))
        drafts.append(
            DraftEpisode(
                episode_type,
                start,
                end,
                peak,
                scores[peak] / threshold if threshold else scores[peak],
                subjects,
                rationale.format(peak=peak, value=scores[peak], start=start, end=end),
            )
        )
    return drafts


def swap_episodes(
    metrics: Sequence[PartPairFrameMetric], config: ReviewMetricsConfig
) -> list[DraftEpisode]:
    drafts: list[DraftEpisode] = []
    for a, b in PART_PAIRS:
        rows = [m for m in metrics if (m.part_a, m.part_b) == (a, b) and m.state == "observed"]
        swap_scores = {m.analysis_frame_index: float(m.swap_score or 0.0) for m in rows}
        absorbing = {m.analysis_frame_index: m.absorbing_part for m in rows}
        for draft in _episodes_from_scores(
            "segmentation_identity_swap",
            (a, b),
            swap_scores,
            config.swap_score_threshold,
            config.episode_max_frame_gap,
            f"{a}/{b} swap score peaks at {{value:.2f}} on f{{peak}} "
            "(max of label exchange, large-mask convergence, absorption).",
        ):
            part = absorbing.get(draft.peak_frame)
            drafts.append(
                draft
                if part is None
                else DraftEpisode(
                    draft.episode_type,
                    draft.start_frame,
                    draft.end_frame,
                    draft.peak_frame,
                    draft.score,
                    draft.subjects,
                    draft.rationale[:-1]
                    + f"; {part} appears to absorb the other part's footprint.",
                )
            )
        crossing = {m.analysis_frame_index: 1.0 for m in rows if m.label_crossing}
        drafts.extend(
            _episodes_from_scores(
                "segmentation_label_crossing",
                (a, b),
                crossing,
                1.0,
                config.episode_max_frame_gap,
                f"{a} and {b} centroid trajectories swapped sides within "
                f"{config.crossing_window_frames} frames ending f{{peak}} while both masks stayed "
                f">= {config.crossing_min_area_fraction:.0%} of their median areas.",
            )
        )
    return drafts


def leakage_episodes(
    metrics: Sequence[PartAppearanceFrameMetric], config: ReviewMetricsConfig
) -> list[DraftEpisode]:
    drafts: list[DraftEpisode] = []
    for part in TARGETS:
        rows = [m for m in metrics if m.part_id == part and m.state == "observed"]
        scores: dict[int, float] = {}
        for m in rows:
            if not m.leakage_suspect:
                continue
            components = [
                float(m.lab_distance_robust_z or 0) / config.appearance_robust_z_threshold
            ]
            if part in DARK_PARTS:
                components.append(
                    float(m.yellow_fraction or 0) / config.yellow_fraction_threshold_dark_parts
                )
            scores[m.analysis_frame_index] = max(1.0, max(components))
        drafts.extend(
            _episodes_from_scores(
                "appearance_leakage",
                (part,),
                scores,
                1.0,
                config.episode_max_frame_gap,
                f"{part} mask color is unusually far from its frame-0 statistics for this "
                "part (robust z) or a dark part turns yellow (peak {value:.2f}x threshold at "
                "f{peak}); hand overlap is recorded as context only.",
            )
        )
    return drafts


def growth_episodes(
    metrics: Sequence[PartGrowthFrameMetric], config: ReviewMetricsConfig
) -> list[DraftEpisode]:
    drafts: list[DraftEpisode] = []
    for part in TARGETS:
        for growth_class, episode_type in (
            ("hand_capture_suspect", "mask_growth_hand_capture_suspect"),
            ("unexplained", "mask_growth_unexplained"),
        ):
            # Score in threshold units, whichever of the step or windowed ratio is stronger.
            scores = {
                m.analysis_frame_index: max(
                    float(m.area_ratio_vs_previous or 0) / config.growth_area_ratio_threshold,
                    float(m.area_ratio_vs_window or 0) / config.growth_window_ratio_threshold,
                )
                for m in metrics
                if m.part_id == part and m.state == "observed" and m.growth_class == growth_class
            }
            drafts.extend(
                _episodes_from_scores(
                    episode_type,
                    (part,),
                    scores,
                    1.0,
                    config.episode_max_frame_gap,
                    f"{part} area grows {{value:.2f}}x threshold (frame-to-frame >= "
                    f"{config.growth_area_ratio_threshold} or {config.growth_window_frames}-frame "
                    f"window >= {config.growth_window_ratio_threshold}) at f{{peak}}; "
                    + (
                        "a stabilized hand box overlaps the mask (possible hand capture)."
                        if growth_class == "hand_capture_suspect"
                        else "no stabilized hand box overlaps the mask (unexplained growth)."
                    ),
                )
            )
    return drafts


def area_anomaly_episodes(
    stats: Sequence[PartFrameStat], config: ReviewMetricsConfig
) -> list[DraftEpisode]:
    """Sustained runs where a part is far larger or smaller than its own median area."""
    drafts: list[DraftEpisode] = []
    for part in TARGETS:
        rows = [s for s in stats if s.part_id == part and s.state == "observed"]
        enlarged = {
            s.analysis_frame_index: float(s.area_ratio_vs_median or 0)
            / config.area_enlarged_ratio_threshold
            for s in rows
        }
        collapsed = {
            s.analysis_frame_index: config.area_collapsed_ratio_threshold
            / max(1e-6, float(s.area_ratio_vs_median or 0))
            for s in rows
        }
        for scores, kind in ((enlarged, "enlarged to"), (collapsed, "collapsed to")):
            for draft in _episodes_from_scores(
                "mask_area_anomaly_vs_median",
                (part,),
                scores,
                1.0,
                config.episode_max_frame_gap,
                f"{part} mask {kind} {{value:.2f}}x the anomaly threshold relative to its "
                "own median area, sustained over f{start}–f{end} (peak f{peak}).",
            ):
                if draft.end_frame - draft.start_frame + 1 >= config.area_anomaly_min_frames:
                    drafts.append(draft)
    return drafts


def hand_episodes(
    gaps: Sequence[HandGapRecord],
    proxies: Sequence[HandGapFrameProxy],
    config: ReviewMetricsConfig,
) -> list[DraftEpisode]:
    drafts: list[DraftEpisode] = []
    for gap in gaps:
        rows = [
            p
            for p in proxies
            if gap.start_frame <= p.analysis_frame_index < gap.end_frame_exclusive
        ]
        suspects = [p for p in rows if p.visible_but_undetected_suspect]
        if suspects:
            peak = max(suspects, key=lambda p: (p.skin_fraction_in_window, -p.analysis_frame_index))
            drafts.append(
                DraftEpisode(
                    "hand_visible_but_undetected",
                    gap.start_frame,
                    gap.end_frame_exclusive - 1,
                    peak.analysis_frame_index,
                    len(suspects) / max(1, gap.length_frames) + peak.skin_fraction_in_window,
                    (peak.last_known_hand_id,),
                    f"Stabilized WiLoR missing for {gap.length_frames} frame(s); skin-like pixels "
                    f"fill {peak.skin_fraction_in_window:.0%} of the last-pose window at "
                    f"f{peak.analysis_frame_index} with {peak.fingertips_in_frame_count}/5 "
                    "fingertips in frame (proxy, not a detection).",
                )
            )
        if (
            gap.reentry_jump_normalized is not None
            and gap.reentry_frame is not None
            and gap.reentry_jump_normalized >= config.reentry_jump_normalized_threshold
        ):
            drafts.append(
                DraftEpisode(
                    "hand_reentry_jump",
                    gap.start_frame,
                    gap.reentry_frame,
                    gap.reentry_frame,
                    gap.reentry_jump_normalized / config.reentry_jump_normalized_threshold,
                    ("stabilized_wilor",),
                    f"Hand re-enters at f{gap.reentry_frame} {gap.reentry_jump_normalized:.2f} "
                    f"hand-scales ({gap.reentry_jump_pixels:.0f} px) from the last known wrist "
                    f"after a {gap.length_frames}-frame gap.",
                )
            )
    return drafts


def contact_episodes(intervals: Sequence[ContactIntervalRecord]) -> list[DraftEpisode]:
    drafts: list[DraftEpisode] = []
    for item in intervals:
        if item.expected_for_substep is False:
            drafts.append(
                DraftEpisode(
                    "contact_unexpected_for_substep",
                    item.start_frame,
                    item.end_frame_exclusive - 1,
                    item.start_frame,
                    1.0 + min(1.0, item.duration_frames / 30.0),
                    (item.hand_source_id, item.part_id),
                    f"{item.hand_source_id} contact candidate with {item.part_id} for "
                    f"{item.duration_frames} frame(s) during {item.substep_id}, whose "
                    "agent-assumed expected parts exclude it.",
                )
            )
        if item.flicker:
            drafts.append(
                DraftEpisode(
                    "contact_flicker",
                    item.start_frame,
                    item.end_frame_exclusive - 1,
                    item.start_frame,
                    1.0,
                    (item.hand_source_id, item.part_id),
                    f"{item.hand_source_id}/{item.part_id} contact candidate lasts only "
                    f"{item.duration_frames} frame(s); likely geometry flicker rather than a hold.",
                )
            )
    return drafts


def rank_episodes(drafts: Sequence[DraftEpisode]) -> tuple[ReviewTriggerEpisode, ...]:
    """Interleave episode types by within-type score so the top list spans every detector.

    Scores are only comparable within a type (each is in its own threshold units), so a
    global sort by score would let one prolific detector crowd out the rest.  Round `k` of
    the ranking holds the k-th strongest episode of every type, ordered by score.
    """
    by_type: dict[str, list[DraftEpisode]] = defaultdict(list)
    for draft in sorted(
        drafts, key=lambda d: (-d.score, d.start_frame, d.episode_type, d.subjects)
    ):
        by_type[draft.episode_type].append(draft)
    ordered: list[DraftEpisode] = []
    depth = 0
    while any(depth < len(items) for items in by_type.values()):
        round_items = [items[depth] for items in by_type.values() if depth < len(items)]
        ordered.extend(sorted(round_items, key=lambda d: (-d.score, d.start_frame, d.episode_type)))
        depth += 1
    return tuple(
        ReviewTriggerEpisode(
            rank=index + 1,
            episode_type=draft.episode_type,  # type: ignore[arg-type]
            start_frame=draft.start_frame,
            end_frame=draft.end_frame,
            peak_frame=draft.peak_frame,
            score=_r(draft.score),
            subjects=draft.subjects,
            rationale=draft.rationale,
        )
        for index, draft in enumerate(ordered)
    )


def episodes_covering(
    episodes: Sequence[ReviewTriggerEpisode], start: int, end: int
) -> tuple[ReviewTriggerEpisode, ...]:
    """Episodes overlapping the inclusive frame range [start, end]."""
    return tuple(e for e in episodes if e.start_frame <= end and e.end_frame >= start)


# --------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------


def _verify_against_index(
    index: InteractionReviewIndexManifest, repository_root: Path, uri: str
) -> ArtifactFingerprint:
    declared = next((item for item in index.input_artifacts if item.uri == uri), None)
    if declared is None:
        raise ValueError(f"v4 index does not declare the input {uri}")
    validate_artifact_fingerprint(declared, repository_root, label=f"v4 input {uri}")
    return declared


def config_fingerprint(config_file: Path, repository_root: Path) -> ArtifactFingerprint:
    """Fingerprint the config actually used; an out-of-tree copy keeps its checked-in URI."""
    if config_file.resolve().is_relative_to(repository_root.resolve()):
        return _file_fingerprint(config_file, repository_root)
    measured = _file_fingerprint(config_file, config_file.parent)
    return measured.model_copy(update={"uri": CONFIG_PATH.as_posix()})


def load_config(path: Path) -> ReviewMetricsConfig:
    config = ReviewMetricsConfig.model_validate_json(path.read_text(encoding="utf-8"))
    if config.provenance_tag != "agent_authored_assumption":
        raise ValueError("review metrics config must be tagged as agent-authored assumptions")
    return config


def load_hand_provenance(path: Path) -> dict[tuple[int, str], str]:
    rows = [
        StabilizedHandProvenanceRow.model_validate(item) for item in json.loads(path.read_text())
    ]
    return {(row.analysis_frame_index, row.output_hand_id): row.state for row in rows}


def load_kineo_provenance(path: Path) -> dict[int, KineoBoxProvenance]:
    rows = [KineoBoxProvenance.model_validate(item) for item in json.loads(path.read_text())]
    return {row.analysis_frame_index: row for row in rows}


def _phases(fine_contract: object) -> list[tuple[str, str, int, int]]:
    phases = [
        ("agent_substep", step.substep_id, step.start_frame, step.end_frame_exclusive)
        for step in getattr(fine_contract, "substeps")
    ]
    phases.extend(
        ("coarse_gt", f"{index}:{action}", start, end)
        for index, (action, start, end) in enumerate(COARSE_GT)
    )
    return phases


# --------------------------------------------------------------------------------------
# Rendering and reporting
# --------------------------------------------------------------------------------------


def _read_frame(capture: cv2.VideoCapture, frame: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, image = capture.read()
    if not ok:
        raise RuntimeError(f"could not decode frame {frame}")
    return image


def _thumbnail(
    bgr: np.ndarray,
    masks: dict[str, np.ndarray | None],
    hands: Sequence[object],
    dimensions: tuple[int, int],
    title: str,
    size: tuple[int, int] = (426, 240),
) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    for part in TARGETS:
        mask = masks.get(part)
        if mask is None:
            continue
        color = np.asarray(PART_COLORS[part], dtype=np.float32)
        rgb[mask] = 0.55 * rgb[mask] + 0.45 * color
    image = Image.fromarray(rgb.clip(0, 255).astype(np.uint8)).resize(size)
    draw = ImageDraw.Draw(image)
    sx, sy = size[0] / dimensions[0], size[1] / dimensions[1]
    for hand in hands:
        x0, y0, x1, y1 = box_pixels(getattr(hand, "box"), dimensions)
        draw.rectangle((x0 * sx, y0 * sy, x1 * sx, y1 * sy), outline=(65, 169, 245), width=1)
        wx, wy = review._pixel(getattr(hand, "landmarks")[0], dimensions)
        draw.ellipse((wx * sx - 3, wy * sy - 3, wx * sx + 3, wy * sy + 3), fill=(65, 169, 245))
    draw.rectangle((0, 0, size[0], 14), fill=(0, 0, 0))
    draw.text((3, 1), title[: size[0] // 6], fill=(255, 255, 255))
    return image


def _episode_sheet(
    episode: ReviewTriggerEpisode,
    capture: cv2.VideoCapture,
    reference: review.LoadedSource,
    hands: review.LoadedSource,
    dimensions: tuple[int, int],
) -> Image.Image:
    frames = {episode.start_frame, episode.peak_frame, episode.end_frame}
    # Short episodes get before/after context so the tile row still shows a transition.
    for candidate in (episode.start_frame - CONTEXT_FRAMES, episode.end_frame + CONTEXT_FRAMES):
        if len(frames) < 3 and 0 <= candidate < FRAME_COUNT:
            frames.add(candidate)
    tiles = []
    for frame in sorted(frames):
        masks = {
            part: review._mask_for_part(
                reference.observations[frame], part, reference.run_directory, dimensions
            )
            for part in TARGETS
        }
        roles = [
            name
            for name, value in (
                ("start", episode.start_frame),
                ("peak", episode.peak_frame),
                ("end", episode.end_frame),
            )
            if frame == value
        ]
        label = "/".join(roles) or "context"
        tiles.append(
            _thumbnail(
                _read_frame(capture, frame),
                masks,
                hands.observations[frame].hands,
                dimensions,
                f"#{episode.rank} {episode.episode_type} f{frame} ({label})",
            )
        )
    width, height = tiles[0].size
    sheet = Image.new("RGB", (width * 3, height + 16), color=(20, 20, 20))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, (index * width, 16))
    ImageDraw.Draw(sheet).text(
        (3, 2),
        f"#{episode.rank} {episode.episode_type} f{episode.start_frame}-f{episode.end_frame} "
        f"score {episode.score:.2f}: {episode.rationale}"[: width * 3 // 6],
        fill=(255, 230, 50),
    )
    return sheet


def render_contact_sheets(
    episodes: Sequence[ReviewTriggerEpisode],
    video_path: Path,
    reference: review.LoadedSource,
    hands: review.LoadedSource,
    dimensions: tuple[int, int],
    output_dir: Path,
    overview_path: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    paths: list[Path] = []
    sheets: list[Image.Image] = []
    try:
        for episode in episodes:
            sheet = _episode_sheet(episode, capture, reference, hands, dimensions)
            path = output_dir / f"episode_{episode.rank:02d}_{episode.episode_type}.png"
            sheet.save(path)
            paths.append(path)
            sheets.append(sheet)
    finally:
        capture.release()
    if sheets:
        width, height = sheets[0].size
        overview = Image.new("RGB", (width, height * len(sheets)), color=(20, 20, 20))
        for index, sheet in enumerate(sheets):
            overview.paste(sheet, (0, index * height))
        overview.save(overview_path)
    return paths


def _fmt(value: float | None, digits: int) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _episode_refs(episodes: Sequence[ReviewTriggerEpisode]) -> str:
    return "; ".join(
        f"#{e.rank} `{e.episode_type}` f{e.start_frame}–f{e.end_frame} ({', '.join(e.subjects)})"
        for e in episodes
    )


def _eligibility_description(index: InteractionReviewIndexManifest) -> str:
    """Describe the index's contact-eligible frame intervals; falls back to the v4 constant."""
    eligible = [
        f"[{item.start_frame},{item.end_frame_exclusive})"
        for item in index.segmentation_validity_intervals
        if item.state == "contact_eligible"
    ]
    if not eligible:
        return f"`[0,{SEGMENTATION_CONTACT_ELIGIBLE_THROUGH})`"
    return " and ".join(f"`{item}`" for item in eligible)


def _report(
    manifest: ReviewMetricsManifest,
    config: ReviewMetricsConfig,
    sheet_paths: Sequence[Path],
    output_root: Path,
    repository_root: Path,
    *,
    eligibility: str | None = None,
) -> str:
    eligibility = eligibility or f"`[0,{SEGMENTATION_CONTACT_ELIGIBLE_THROUGH})`"
    episodes = manifest.episodes
    counts: dict[str, int] = defaultdict(int)
    for item in episodes:
        counts[item.episode_type] += 1
    top = episodes[: config.top_episode_count]
    swap_late = episodes_covering(
        [e for e in episodes if e.episode_type.startswith("segmentation_")], 1100, 1200
    )
    interior_370 = episodes_covering([e for e in episodes if "interior" in e.subjects], 355, 385)
    interval = manifest.source_interval
    yellow, skin, contacts = manifest.yellow_band, manifest.skin_band, manifest.contact_summary
    suspect_frames = sum(p.visible_but_undetected_suspect for p in manifest.hand_gap_proxies)
    lines = [
        "# First-minute review metrics (v1)",
        "",
        "Label-free proxies over analysis frames `[0,1800)` / source "
        f"{interval.start_seconds:.3f}–{interval.end_seconds:.3f} s. Every number below is a "
        "geometry, appearance, or provenance measurement of retained review artifacts. None is an "
        "accuracy metric, a ground-truth label, or a claim about what the worker touched; episodes "
        "are ranked prompts for a human to look at a frame range.",
        "",
        "## Claim boundaries",
        "",
        *[f"- {item}" for item in manifest.claim_boundaries],
        *[f"- {item}" for item in config.claim_boundaries],
        "",
        "## Ranked review episodes",
        "",
        f"{len(episodes)} episodes in total: "
        + ", ".join(f"`{kind}` {count}" for kind, count in sorted(counts.items()))
        + ".",
        "",
        "| rank | type | frames | peak | score | subjects | rationale |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        *[
            f"| {e.rank} | `{e.episode_type}` | {e.start_frame}–{e.end_frame} | {e.peak_frame} | "
            f"{e.score:.2f} | {', '.join(e.subjects)} | {e.rationale} |"
            for e in top
        ],
        "",
        "Scores are in threshold units (1.0 = exactly at the configured threshold) except "
        "`hand_visible_but_undetected` (suspect-frame fraction plus peak skin fraction) and the "
        "contact types (1.0 plus a duration term). Scores are only comparable within a type, so "
        "the rank interleaves types: round k holds the k-th strongest episode of every type.",
        "",
        "## Reported ranges",
        "",
        "- Reported chassis/interior swap around frames 1100–1200: "
        + (
            f"caught by {_episode_refs(swap_late)}"
            if swap_late
            else "no segmentation episode overlaps this range."
        ),
        "- Reported transient interior growth near frame 370: "
        + (
            f"overlapping interior episodes: {_episode_refs(interior_370)}"
            if interior_370
            else "no interior episode overlaps frames 355–385."
        ),
        "",
        "Thresholds were applied uniformly to the whole minute; all episodes are listed in "
        f"`{TRIGGERS_NAME}`.",
        "",
        "## Segmentation geometry",
        "",
        "Part area medians (pixels): "
        + ", ".join(f"{part} {value:.0f}" for part, value in manifest.part_area_medians.items())
        + ".",
        "",
        "Swap score per pair per frame = max(label exchange IoU with the other part's previous "
        "mask, same-frame IoU weighted by both parts being large, absorption of a collapsing "
        "part's footprint by an enlarged neighbor). Label crossing = the centroid difference "
        f"vector reverses over {config.crossing_window_frames} frames while both areas stay >= "
        f"{config.crossing_min_area_fraction:.0%} of median. Area anomalies = runs of at least "
        f"{config.area_anomaly_min_frames} frames where a part is >= "
        f"{config.area_enlarged_ratio_threshold}x or <= {config.area_collapsed_ratio_threshold}x "
        "its own median area. Growth events use the frame-to-frame ratio or the "
        f"{config.growth_window_frames}-frame windowed ratio.",
        "",
        "## Appearance",
        "",
        f"Yellow band ({yellow.space}, low {yellow.channel_low}, high {yellow.channel_high}) "
        f"was derived from {yellow.derived_from}; {yellow.sample_pixels} sample pixels.",
        f"Skin band ({skin.space}, low {skin.channel_low}, high {skin.channel_high}) was derived "
        f"from {skin.derived_from}; {skin.sample_pixels} sample pixels. "
        f"{_fmt(skin.background_fraction, 3)} of sampled non-hand scene pixels also fall inside "
        "the band, which bounds how discriminative this visibility proxy can be; a gap frame "
        "is a suspect only when its window beats the frame background by the configured ratio.",
        "",
        "Leakage flags use each part's own Lab-distance distribution (robust z >= "
        f"{config.appearance_robust_z_threshold} and distance >= "
        f"{config.appearance_distance_threshold}) or a dark part turning yellow. Hand-box "
        "overlap is recorded per frame as context and drives the growth classifier, not the "
        "leakage flag.",
        "",
        "## Hands",
        "",
        f"{len(manifest.hand_gaps)} frame-level stabilized-WiLoR gaps covering "
        f"{sum(g.length_frames for g in manifest.hand_gaps)} frames; {suspect_frames} gap frames "
        "carry the visible-but-undetected proxy.",
        "",
        "| phase | frames | samples | median jitter | p90 jitter | missing-hand frames |",
        "| --- | --- | --- | --- | --- | --- |",
        *[
            f"| {p.phase_kind} {p.phase_id} | {p.start_frame}–{p.end_frame_exclusive} | "
            f"{p.sample_count} | {_fmt(p.median_jitter_normalized, 4)} | "
            f"{_fmt(p.p90_jitter_normalized, 4)} | {p.missing_hand_frames} |"
            for p in manifest.jitter_by_phase
        ],
        "",
        "Jitter is nearest-wrist displacement divided by hand scale (sqrt of box area); it "
        "includes real hand motion and is not a pose-accuracy measure.",
        "",
        "## Contact candidates",
        "",
        f"{contacts.interval_count} debounced intervals on contact-eligible frames "
        f"{eligibility}; {contacts.sub_5_frame_count} shorter "
        f"than 5 frames; duration median {contacts.duration_median}, p90 {contacts.duration_p90}, "
        f"max {contacts.duration_max}. Histogram: {contacts.duration_histogram}. Against "
        f"agent-assumed expected parts: {contacts.expected_count} expected, "
        f"{contacts.unexpected_count} unexpected, {contacts.unknown_substep_count} outside the "
        f"labeled `[0,600)`; unexpected per substep {contacts.per_substep_unexpected}.",
        "",
        "## Kineo by box provenance",
        "",
        "| source | frames | body frames | mean joint conf | jitter samples | median jitter px "
        "| p90 jitter px |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        *[
            f"| {k.box_source} | {k.frame_count} | {k.body_frames} | "
            f"{_fmt(k.mean_joint_confidence, 3)} | {k.jitter_sample_count} | "
            f"{_fmt(k.median_jitter_pixels, 2)} | {_fmt(k.p90_jitter_pixels, 2)} |"
            for k in manifest.kineo_by_provenance
        ],
        "",
        "## Contact sheets",
        "",
        *[f"- `{path.relative_to(repository_root).as_posix()}`" for path in sheet_paths],
        "",
        "## Outputs",
        "",
        f"- `{(output_root / METRICS_NAME).as_posix()}` typed per-frame metrics",
        f"- `{(output_root / TRIGGERS_NAME).as_posix()}` ranked episodes",
        *(
            [f"- `{manifest.output_rrd.uri}` inference-free metric time series (v4 clocks)"]
            if manifest.output_rrd
            else []
        ),
        "",
        "## TODO / follow-up (GPU, not run here)",
        "",
        "- **TODO (GPU):** run DAM4SAM on the 60 s window and compute the cross-method mask "
        "disagreement against corrected SAM3 as an additional swap/leakage trigger. This requires "
        "model inference and was deliberately not executed in this CPU-only pass.",
        "",
    ]
    return "\n".join(lines)


def _log_rrd(
    rrd_path: Path,
    clip_id: str,
    manifest: ReviewMetricsManifest,
) -> None:
    import rerun as rr
    import rerun.blueprint as rrb

    entity = f"world/{clip_id}/review_metrics_v1"
    rr.init("battle-review-metrics", recording_id="review_metrics_first_minute_v1")
    rr.save(rrd_path)
    rr.log(
        f"{entity}/metadata/summary",
        rr.TextDocument(
            manifest.model_copy(
                update={
                    key: ()
                    for key in (
                        "part_stats",
                        "pair_metrics",
                        "appearance_metrics",
                        "growth_metrics",
                        "hand_jitter",
                        "kineo_metrics",
                    )
                }
            ).model_dump_json(indent=2),
            media_type="application/json",
        ),
        static=True,
    )
    by_frame_pairs: dict[int, list[PartPairFrameMetric]] = defaultdict(list)
    for item in manifest.pair_metrics:
        by_frame_pairs[item.analysis_frame_index].append(item)
    by_frame_appearance: dict[int, list[PartAppearanceFrameMetric]] = defaultdict(list)
    for item in manifest.appearance_metrics:
        by_frame_appearance[item.analysis_frame_index].append(item)
    by_frame_growth: dict[int, list[PartGrowthFrameMetric]] = defaultdict(list)
    for item in manifest.growth_metrics:
        by_frame_growth[item.analysis_frame_index].append(item)
    by_frame_stats: dict[int, list[PartFrameStat]] = defaultdict(list)
    for item in manifest.part_stats:
        by_frame_stats[item.analysis_frame_index].append(item)
    by_frame_jitter: dict[int, list[float]] = defaultdict(list)
    for item in manifest.hand_jitter:
        if item.jitter_normalized is not None:
            by_frame_jitter[item.analysis_frame_index].append(item.jitter_normalized)
    suspects = {
        item.analysis_frame_index
        for item in manifest.hand_gap_proxies
        if item.visible_but_undetected_suspect
    }
    kineo = {item.analysis_frame_index: item for item in manifest.kineo_metrics}
    for frame in range(FRAME_COUNT):
        time = frame / ANALYSIS_FPS
        rr.set_time("analysis_frame", sequence=frame)
        rr.set_time("analysis_time", duration=time)
        rr.set_time("source_time", duration=SOURCE_START_SECONDS + time)
        for item in by_frame_pairs[frame]:
            review._log_scalar_or_clear(
                f"{entity}/segmentation/swap_score/{item.part_a}__{item.part_b}", item.swap_score
            )
        for item in by_frame_stats[frame]:
            review._log_scalar_or_clear(
                f"{entity}/segmentation/area_ratio_vs_median/{item.part_id}",
                item.area_ratio_vs_median,
            )
        for item in by_frame_appearance[frame]:
            review._log_scalar_or_clear(
                f"{entity}/appearance/lab_distance_from_frame0/{item.part_id}",
                item.lab_distance_from_frame0,
            )
            review._log_scalar_or_clear(
                f"{entity}/appearance/yellow_fraction/{item.part_id}", item.yellow_fraction
            )
            review._log_scalar_or_clear(
                f"{entity}/appearance/hand_overlap_fraction/{item.part_id}",
                item.hand_overlap_fraction,
            )
        for item in by_frame_growth[frame]:
            review._log_scalar_or_clear(
                f"{entity}/growth/area_ratio_vs_previous/{item.part_id}",
                item.area_ratio_vs_previous,
            )
        values = by_frame_jitter.get(frame)
        review._log_scalar_or_clear(
            f"{entity}/hands/mean_jitter_normalized", float(np.mean(values)) if values else None
        )
        rr.log(
            f"{entity}/hands/visible_but_undetected_suspect", rr.Scalars([float(frame in suspects)])
        )
        item = kineo[frame]
        review._log_scalar_or_clear(
            f"{entity}/kineo/mean_joint_confidence", item.mean_joint_confidence
        )
        review._log_scalar_or_clear(f"{entity}/kineo/joint_jitter_pixels", item.joint_jitter_pixels)
        rr.log(
            f"{entity}/episodes/active_count",
            rr.Scalars([float(len(episodes_covering(manifest.episodes, frame, frame)))]),
        )
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Vertical(
                rrb.TimeSeriesView(
                    origin=f"{entity}/segmentation", name="Swap scores / area ratios"
                ),
                rrb.TimeSeriesView(origin=f"{entity}/appearance", name="Appearance proxies"),
                rrb.Horizontal(
                    rrb.TimeSeriesView(origin=f"{entity}/growth", name="Growth ratios"),
                    rrb.TimeSeriesView(origin=f"{entity}/hands", name="Hands"),
                    rrb.TimeSeriesView(origin=f"{entity}/kineo", name="Kineo"),
                    rrb.TimeSeriesView(origin=f"{entity}/episodes", name="Active episodes"),
                ),
                rrb.TextDocumentView(origin=f"{entity}/metadata/summary", name="Metrics summary"),
                row_shares=[3, 2, 2, 2],
            ),
            rrb.TimePanel(timeline="analysis_time", fps=ANALYSIS_FPS),
            auto_layout=False,
            auto_views=False,
        )
    )
    rr.disconnect()


# --------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------


def build_review_metrics(
    *,
    repository_root: Path,
    output_root: Path = OUTPUT_ROOT,
    config_path: Path = CONFIG_PATH,
    v4_index_path: Path = V4_INDEX_PATH,
    write_rrd: bool = True,
    overwrite: bool = False,
    reference_run: Path | None = None,
) -> Path:
    """Compute every metric in one CPU pass over retained artifacts and write the package.

    `reference_run` swaps the part masks for those of any four-part C10379 run over the same
    first minute (a tracker-policy arm, for instance) while every other input still comes
    from the v4 index; the manifest records the override so the package cannot be mistaken
    for the review reference's metrics.
    """
    repository_root = repository_root.resolve()
    config_file = (repository_root / config_path).resolve()
    config = load_config(config_file)
    index_file = (repository_root / v4_index_path).resolve()
    index = InteractionReviewIndexManifest.model_validate_json(
        index_file.read_text(encoding="utf-8")
    )
    if (
        index.frame_count != FRAME_COUNT
        or index.reference_segmentation_method not in ACCEPTED_REFERENCE_METHODS
    ):
        raise ValueError(
            "review metrics require a first-minute v4 index whose reference is one of "
            f"{ACCEPTED_REFERENCE_METHODS}"
        )
    specs = dict(review.FIRST_MINUTE_SOURCES)
    reference_spec = review.SourceSpec(
        index.reference_segmentation_method,
        Path(index.reference_segmentation_manifest.uri).parent,
    )
    indexed_specs = [*specs.values(), reference_spec]
    reference_override: ArtifactFingerprint | None = None
    if reference_run is not None:
        override_directory = (repository_root / reference_run).resolve()
        reference_spec = review.SourceSpec(
            f"override:{override_directory.name}", override_directory
        )
        reference_override = _file_fingerprint(
            override_directory / "manifest.json", repository_root
        )
        indexed_specs = list(specs.values())
    for spec in indexed_specs:
        for name in ("manifest.json", "observations.jsonl"):
            _verify_against_index(index, repository_root, (spec.run_directory / name).as_posix())
    sources = {
        name: review._validate_run(spec, repository_root, frame_count=FRAME_COUNT)
        for name, spec in specs.items()
    }
    reference = review._validate_run(reference_spec, repository_root, frame_count=FRAME_COUNT)
    review._validate_shared_sources([*sources.values(), reference])
    video_path = validate_artifact_fingerprint(
        index.bounded_video, repository_root, label="bounded video"
    )
    frames, fps, dimensions = review._video_info(video_path)
    if (frames, fps, dimensions) != (FRAME_COUNT, ANALYSIS_FPS, review.DIMENSIONS):
        raise ValueError(
            f"expected one 1280x720 1800-frame 30-fps video, got {(frames, fps, dimensions)}"
        )
    hand_provenance_path = sources["stabilized_wilor"].run_directory / "hand_provenance.json"
    kineo_provenance_path = sources["kineo"].run_directory / "box_fusion.json"
    hand_provenance = load_hand_provenance(hand_provenance_path)
    kineo_provenance = load_kineo_provenance(kineo_provenance_path)
    fine_contract = load_fine_substep_contract(repository_root / FINE_LABELS)
    root = (repository_root / output_root).resolve()
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise FileExistsError(root)
    root.mkdir(parents=True, exist_ok=True)

    stabilized = sources["stabilized_wilor"].observations
    raw_wilor = sources["wilor"].observations
    geometries: list[FrameGeometry] = []
    appearance: list[PartAppearanceFrameMetric] = []
    growth: list[PartGrowthFrameMetric] = []
    skin_samples: list[np.ndarray] = []
    background_samples: list[np.ndarray] = []
    gap_samples: list[GapWindowSample] = []
    frame0_lab: dict[str, np.ndarray] = {}
    yellow_band: ColorBand | None = None
    previous_masks: dict[str, np.ndarray | None] | None = None
    previous_geometry: FrameGeometry | None = None
    previous_bgr: np.ndarray | None = None
    gap_anchor: tuple[int, np.ndarray] | None = None
    capture = cv2.VideoCapture(str(video_path))
    try:
        for frame in range(FRAME_COUNT):
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f"bounded video ended before frame {frame}")
            masks = {
                part: review._mask_for_part(
                    reference.observations[frame], part, reference.run_directory, dimensions
                )
                for part in TARGETS
            }
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            if frame == 0:
                yellow_band = derive_yellow_band(hsv, masks)
                for part in TARGETS:
                    if masks[part] is not None and masks[part].any():
                        frame0_lab[part] = lab[masks[part]].astype(np.float64).mean(axis=0)
            assert yellow_band is not None
            yellow = band_membership(hsv, yellow_band)
            hands = stabilized[frame].hands
            hand_union = hand_box_union(hands, dimensions)
            geometry = frame_geometry(frame, masks, previous_masks)
            geometries.append(geometry)
            appearance.extend(
                appearance_metrics_for_frame(
                    frame, hsv, lab, masks, hand_union, frame0_lab, yellow, config
                )
            )
            growth.extend(
                growth_metrics_for_frame(
                    geometry,
                    previous_geometry,
                    masks,
                    hand_union,
                    config,
                    window=(
                        geometries[frame - config.growth_window_frames]
                        if frame >= config.growth_window_frames
                        else None
                    ),
                )
            )
            if frame % SKIN_SAMPLE_STRIDE == 0:
                ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
                raw_union = hand_box_union(raw_wilor[frame].hands, dimensions)
                background_samples.append(
                    ycrcb[::BACKGROUND_STRIDE, ::BACKGROUND_STRIDE][
                        ~raw_union[::BACKGROUND_STRIDE, ::BACKGROUND_STRIDE]
                    ]
                )
                for hand in raw_wilor[frame].hands:
                    if hand.confidence < config.skin_calibration_min_confidence:
                        continue
                    x0, y0, x1, y1 = box_pixels(hand.box, dimensions)
                    qx, qy = (x1 - x0) // 4, (y1 - y0) // 4
                    crop = ycrcb[y0 + qy : y1 - qy, x0 + qx : x1 - qx]
                    if crop.size and sum(s.size for s in skin_samples) < SKIN_SAMPLE_CAP * 3:
                        skin_samples.append(crop.reshape(-1, 3))
            if hands:
                gap_anchor = (frame, bgr.copy())
            elif gap_anchor is not None and previous_bgr is not None:
                anchor_frame = gap_anchor[0]
                ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
                reference_ycrcb = cv2.cvtColor(gap_anchor[1], cv2.COLOR_BGR2YCrCb)
                background_ycrcb = ycrcb[::BACKGROUND_STRIDE, ::BACKGROUND_STRIDE].copy()
                for hand in stabilized[anchor_frame].hands:
                    center = palm_center_pixels(hand, dimensions)
                    size = max(
                        8,
                        int(round(config.skin_window_scale * hand_scale_pixels(hand, dimensions))),
                    )
                    gap_samples.append(
                        GapWindowSample(
                            frame,
                            anchor_frame,
                            hand.hand_id,
                            center,
                            size,
                            crop_window(ycrcb, center, size),
                            crop_window(reference_ycrcb, center, size),
                            background_ycrcb,
                            fingertips_in_frame(hand),
                        )
                    )
            previous_masks = masks
            previous_geometry = geometry
            previous_bgr = bgr
    finally:
        capture.release()
    assert yellow_band is not None
    medians = part_area_medians(geometries)
    stats = part_stats(geometries, medians)
    pairs = pair_metrics(geometries, medians, config)
    skin_band = derive_skin_band(skin_samples, background_samples)
    proxies = gap_proxies(gap_samples, skin_band, config)
    appearance = list(finalize_appearance(appearance, config))
    gaps = hand_gaps(stabilized, FRAME_COUNT)
    gap_rows = gap_records(stabilized, gaps, proxies, dimensions)
    jitter = jitter_metrics(stabilized, hand_provenance, dimensions, FRAME_COUNT)
    phases = jitter_by_phase(jitter, _phases(fine_contract), stabilized)
    substep_lookup = {
        frame: substep_for_frame(fine_contract, frame).substep_id
        for frame in range(fine_contract.frame_count)
    }
    expected = {item.substep_id: item.expected_parts for item in config.expected_touched_parts}
    intervals = contact_intervals(
        index.contact_diagnostics,
        frame_count=FRAME_COUNT,
        substep_lookup=substep_lookup,
        expected_parts=expected,
        coarse_gt=COARSE_GT,
        flicker_max_frames=config.contact_flicker_max_frames,
    )
    summary = contact_summary(intervals)
    kineo_rows = kineo_metrics(
        sources["kineo"].observations, kineo_provenance, dimensions, FRAME_COUNT
    )
    kineo_summary = kineo_by_provenance(kineo_rows)
    episodes = rank_episodes(
        [
            *swap_episodes(pairs, config),
            *leakage_episodes(appearance, config),
            *growth_episodes(growth, config),
            *area_anomaly_episodes(stats, config),
            *hand_episodes(gap_rows, proxies, config),
            *contact_episodes(intervals),
        ]
    )
    artifacts = [
        _file_fingerprint(video_path, repository_root),
        _file_fingerprint(hand_provenance_path, repository_root),
        _file_fingerprint(kineo_provenance_path, repository_root),
        _file_fingerprint(repository_root / FINE_LABELS, repository_root),
        *[item for source in [*sources.values(), reference] for item in source.artifacts],
    ]
    manifest = ReviewMetricsManifest(
        manifest_kind="review_metrics_first_minute_v1",
        frame_count=FRAME_COUNT,
        analysis_fps=ANALYSIS_FPS,
        source_interval=TimeInterval(
            start_seconds=SOURCE_START_SECONDS,
            end_seconds=SOURCE_START_SECONDS + FRAME_COUNT / ANALYSIS_FPS,
        ),
        v4_index=_file_fingerprint(index_file, repository_root),
        config=config_fingerprint(config_file, repository_root),
        reference_run_override=reference_override,
        input_artifacts=tuple({(a.uri, a.sha256): a for a in artifacts}.values()),
        claim_boundaries=CLAIM_BOUNDARIES,
        part_area_medians={part: _r(value, 1) for part, value in medians.items()},
        yellow_band=yellow_band,
        skin_band=skin_band,
        part_stats=stats,
        pair_metrics=pairs,
        appearance_metrics=tuple(appearance),
        growth_metrics=tuple(growth),
        hand_gap_proxies=proxies,
        hand_gaps=gap_rows,
        hand_jitter=jitter,
        jitter_by_phase=phases,
        contact_intervals=intervals,
        contact_summary=summary,
        kineo_metrics=kineo_rows,
        kineo_by_provenance=kineo_summary,
        episodes=episodes,
    )
    sheet_paths = render_contact_sheets(
        episodes[: config.top_episode_count],
        video_path,
        reference,
        sources["stabilized_wilor"],
        dimensions,
        root / SHEET_DIR,
        root / OVERVIEW_SHEET_NAME,
    )
    if write_rrd:
        rrd_path = root / RRD_NAME
        _log_rrd(rrd_path, reference.manifest.clip.clip_id, manifest)
        manifest = manifest.model_copy(
            update={"output_rrd": _file_fingerprint(rrd_path, repository_root)}
        )
    (root / TRIGGERS_NAME).write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "manifest_kind": "review_metrics_triggers_v1",
                "claim_boundaries": list(CLAIM_BOUNDARIES),
                "episodes": [item.model_dump(mode="json") for item in episodes],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / REPORT_NAME).write_text(
        _report(
            manifest,
            config,
            sheet_paths,
            output_root,
            repository_root,
            eligibility=_eligibility_description(index),
        ),
        encoding="utf-8",
    )
    metrics_path = root / METRICS_NAME
    metrics_path.write_text(manifest.model_dump_json(indent=1) + "\n", encoding="utf-8")
    return metrics_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--v4-index", type=Path, default=V4_INDEX_PATH)
    parser.add_argument("--skip-rrd", action="store_true", help="Do not write the metric RRD.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing package.")
    parser.add_argument(
        "--reference-run",
        type=Path,
        help=(
            "Take the part masks from this four-part C10379 run instead of the v4 reference; "
            "the package then measures that run and says so."
        ),
    )
    args = parser.parse_args()
    print(
        build_review_metrics(
            repository_root=args.repository_root,
            output_root=args.output_root,
            config_path=args.config,
            v4_index_path=args.v4_index,
            write_rrd=not args.skip_rrd,
            overwrite=args.overwrite,
            reference_run=args.reference_run,
        )
    )


if __name__ == "__main__":
    main()
