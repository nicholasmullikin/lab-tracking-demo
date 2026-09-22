"""Detector scorecard: which label-free signals notice a segmentation failure?

Truth comes from the 52 human review-anchor cells (13 frames x 4 parts, C10379): a cell
is "failed" when the scored run's anchor IoU is below 0.5, or when the run has more than
300 px on the one cell the human marked hidden. Every detector is a per-frame per-part
scalar computed from artifacts that already exist (observations, masks, a DAM4SAM run on
the same clip, the cross-view consensus and the visual hull). Detectors are scored by
rank AUROC and by precision / recall with counts, overall and per failure class, then the
top-3 by AUROC are rank-averaged into one confidence series with an abstain flag; the
combination is also scored leave-one-frame-out so the detector choice is not fitted to the
frame it is scored on.

The anchors are review evidence on 13 frames of one view, not ground truth; every table
says so and reports counts, not only rates.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from . import mask_cache
from .mask_ops import mask_iou
from .mask_ops import overlap_fraction_union as overlap_fraction
from .muggled_smoke import relative_uri
from .review_anchors import (
    MASK_SET_NAME,
    AnchorRunScore,
    ReviewAnchorMaskSet,
    load_mask_set,
    resolve_run_directory,
    score_run,
)
from .schemas import (
    DetectorConfidenceRow,
    DetectorOperatingPoint,
    DetectorScore,
    DetectorScorecardManifest,
    DetectorTruthCell,
    LeaveOneFrameOutFold,
    LeaveOneFrameOutResult,
    ProposedAnchorFrame,
    ProposedAnchorFrames,
    fingerprint,
)

CLAIM_BOUNDARY = (
    "13 frames, one view (C10379), review evidence not ground truth: the human review anchors "
    "rank detectors against each other on 52 cells; they are not a dataset and support no "
    "accuracy claim. Counts are reported with every rate because the intervals are wide."
)

FAILED_IOU_BELOW = 0.5
HIDDEN_FALSE_POSITIVE_PIXELS = 300
AREA_JUMP_WINDOW = 15
RECALL_FLOOR = 0.8
TOP_K = 3
COMBINED_NAME = "combined_rank_average"
ALL_SUBSET = "all"
OUTSIDE_CLASS = "outside"
DISTRACTOR_CLASS = "distractor"
DEFAULT_CLASS_BY_WINDOW: dict[str, str] = {
    "279-408": "occlusion_leak",
    "573-722": "rotation_swap",
    "1020-1172": "rotation_swap",
}
# The screwdriver window: rear_body latched onto a yellow screwdriver from ~1660; the human
# named the cell (1700, rear_body) `distractor_confusion`. 1200 and 1500 are the other late
# frames outside every reported window. No anchor frame lies in the 1235 correction region.
DEFAULT_DISTRACTOR_FRAMES: tuple[int, ...] = (1200, 1500, 1700)

DETECTORS: tuple[str, ...] = (
    "sam3_score",
    "area_jump",
    "area_vs_seed",
    "tracker_disagreement_large",
    "tracker_disagreement_tiny",
    "consensus_error_px",
    "consensus_contradicted",
    "hull_disagreement",
    "hull_episode",
    "overlap_other_parts",
)
DETECTOR_DEFINITIONS: dict[str, str] = {
    "sam3_score": (
        "1 - sigmoid(object_score) from observations.jsonl (SAM3's per-object logit); higher "
        "= less confident. NaN where the run logs no score (frame 0 seeds)."
    ),
    "area_jump": (
        f"|area_t / median(area_(t-{AREA_JUMP_WINDOW})..(t-1)) - 1|; the median is floored at 1 "
        "px; NaN at frame 0."
    ),
    "area_vs_seed": "|log((area_t + 1) / (area_0 + 1))| against the frame-0 seed mask.",
    "tracker_disagreement_large": (
        "1 - IoU(run mask, dam4sam-large-1024-sched mask) on the same frame; 1 when exactly one "
        "is empty, NaN when both are."
    ),
    "tracker_disagreement_tiny": (
        "Same as tracker_disagreement_large against dam4sam-tiny-1024-sched."
    ),
    "consensus_error_px": (
        "The view's error px from the cross-view consensus per_frame (0 inside its own mask); "
        "NaN where no consensus exists (interior: no other view tracks it)."
    ),
    "consensus_contradicted": (
        "1 inside a consensus episode for this view and part that contradicts the majority, "
        "else 0; NaN where consensus_error_px is NaN."
    ),
    "hull_disagreement": (
        "1 - IoU(hull projection, run mask) for this view from hull_series.npz; NaN where no "
        "hull exists."
    ),
    "hull_episode": (
        "1 inside a hull-disagreement episode for this view and part, else 0; NaN as hull."
    ),
    "overlap_other_parts": (
        "Fraction of the part's mask pixels shared with any other part's mask on the same "
        "frame (the swap signature); NaN when the mask is empty."
    ),
}
TRUTH_RULE = (
    f"failed = anchor IoU < {FAILED_IOU_BELOW} on a labeled cell (a missing run mask counts as "
    f"IoU 0); on the hidden cell failed = run area > {HIDDEN_FALSE_POSITIVE_PIXELS} px. The "
    "failure class is per frame: explicit distractor frames first, then the window the frame "
    "lies in, else outside."
)


# ------------------------------------------------------------------------- failure classes


def failure_class_for_frame(
    frame: int,
    windows: Mapping[str, tuple[int, int]],
    *,
    class_by_window: Mapping[str, str] | None = None,
    distractor_frames: Iterable[int] = DEFAULT_DISTRACTOR_FRAMES,
) -> str:
    """Explicit distractor frames win, then the first window containing the frame, else outside."""
    if frame in set(distractor_frames):
        return DISTRACTOR_CLASS
    mapping = DEFAULT_CLASS_BY_WINDOW if class_by_window is None else class_by_window
    for name, (low, high) in windows.items():
        if low <= frame < high:
            return mapping.get(name, name)
    return OUTSIDE_CLASS


def cell_failed(outcome: str, iou: float | None, run_area: int | None) -> bool | None:
    if outcome in ("scored", "run_mask_missing"):
        return (iou if iou is not None else 0.0) < FAILED_IOU_BELOW
    if outcome == "hidden_false_positive":
        return (run_area or 0) > HIDDEN_FALSE_POSITIVE_PIXELS
    if outcome == "hidden_correct":
        return False
    return None


def truth_cells(
    score: AnchorRunScore,
    windows: Mapping[str, tuple[int, int]],
    *,
    distractor_frames: Iterable[int] = DEFAULT_DISTRACTOR_FRAMES,
) -> tuple[DetectorTruthCell, ...]:
    distractors = tuple(distractor_frames)
    return tuple(
        DetectorTruthCell(
            analysis_frame_index=cell.analysis_frame_index,
            target=cell.target,
            failure_class=failure_class_for_frame(
                cell.analysis_frame_index, windows, distractor_frames=distractors
            ),
            anchor_state=cell.anchor_state,
            outcome=cell.outcome,
            iou=cell.iou,
            run_area=cell.run_area,
            failed=cell_failed(cell.outcome, cell.iou, cell.run_area),
        )
        for cell in score.cells
    )


# ------------------------------------------------------------------------- rank statistics


def average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks with ties given their mean rank (no scipy)."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        stop = start
        while stop + 1 < values.size and sorted_values[stop + 1] == sorted_values[start]:
            stop += 1
        ranks[order[start : stop + 1]] = (start + 1 + stop + 1) / 2.0
        start = stop + 1
    return ranks


def _finite_subset(scores: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=float)
    truth = np.asarray(truth, dtype=bool)
    keep = np.isfinite(scores)
    return scores[keep], truth[keep]


def auroc(scores: np.ndarray, truth: np.ndarray) -> float | None:
    """Mann-Whitney AUROC of `scores` for `truth`; NaN scores dropped; None without both classes."""
    scores, truth = _finite_subset(scores, truth)
    positives = int(truth.sum())
    negatives = int(truth.size - positives)
    if positives == 0 or negatives == 0:
        return None
    ranks = average_ranks(scores)
    return float((ranks[truth].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def confusion_at(scores: np.ndarray, truth: np.ndarray, threshold: float) -> DetectorOperatingPoint:
    """Predict failed where score >= threshold; NaN scores are excluded from the counts."""
    scores, truth = _finite_subset(scores, truth)
    predicted = scores >= threshold
    tp = int(np.sum(predicted & truth))
    fp = int(np.sum(predicted & ~truth))
    fn = int(np.sum(~predicted & truth))
    tn = int(np.sum(~predicted & ~truth))
    return _operating_point(threshold, tp, fp, fn, tn)


def _operating_point(
    threshold: float | None, tp: int, fp: int, fn: int, tn: int
) -> DetectorOperatingPoint:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else (0.0 if precision is not None and recall is not None else None)
    )
    return DetectorOperatingPoint(
        threshold=threshold, tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall, f1=f1
    )


def best_f1_threshold(scores: np.ndarray, truth: np.ndarray) -> float | None:
    """The score value whose >= rule maximises F1; ties go to the higher threshold (fewer flags)."""
    scores, truth = _finite_subset(scores, truth)
    if scores.size == 0 or not truth.any():
        return None
    best: tuple[float, float] | None = None
    for candidate in np.unique(scores):
        point = confusion_at(scores, truth, float(candidate))
        f1 = point.f1 or 0.0
        if (
            best is None
            or f1 > best[0] + 1e-12
            or (abs(f1 - best[0]) <= 1e-12 and candidate > best[1])
        ):
            best = (f1, float(candidate))
    return None if best is None else best[1]


def recall_floor_threshold(
    scores: np.ndarray, truth: np.ndarray, floor: float = RECALL_FLOOR
) -> float | None:
    """The highest score value whose >= rule still recalls at least `floor` of the failures."""
    scores, truth = _finite_subset(scores, truth)
    if scores.size == 0 or not truth.any():
        return None
    for candidate in sorted(np.unique(scores), reverse=True):
        point = confusion_at(scores, truth, float(candidate))
        if (point.recall or 0.0) >= floor:
            return float(candidate)
    return None


def score_detector(
    name: str,
    subset: str,
    scores: np.ndarray,
    truth: np.ndarray,
    *,
    overall_best_f1: float | None = None,
    overall_recall_floor: float | None = None,
    recall_floor: float = RECALL_FLOOR,
) -> DetectorScore:
    scores = np.asarray(scores, dtype=float)
    truth = np.asarray(truth, dtype=bool)
    finite = np.isfinite(scores)
    f1_threshold = best_f1_threshold(scores, truth)
    floor_threshold = recall_floor_threshold(scores, truth, floor=recall_floor)
    return DetectorScore(
        detector=name,
        subset=subset,
        cells=int(finite.sum()),
        positives=int((truth & finite).sum()),
        negatives=int((~truth & finite).sum()),
        undefined=int((~finite).sum()),
        auroc=auroc(scores, truth),
        best_f1=None if f1_threshold is None else confusion_at(scores, truth, f1_threshold),
        recall_floor=(
            None if floor_threshold is None else confusion_at(scores, truth, floor_threshold)
        ),
        at_overall_best_f1=(
            None if overall_best_f1 is None else confusion_at(scores, truth, overall_best_f1)
        ),
        at_overall_recall_floor=(
            None
            if overall_recall_floor is None
            else confusion_at(scores, truth, overall_recall_floor)
        ),
    )


def normalized_ranks(values: np.ndarray) -> np.ndarray:
    """Map finite values to [0, 1] by average rank; NaN stays NaN. One finite value maps to 0.5."""
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan)
    finite = np.isfinite(values)
    count = int(finite.sum())
    if count == 1:
        out[finite] = 0.5
    elif count > 1:
        out[finite] = (average_ranks(values[finite]) - 1.0) / (count - 1.0)
    return out


def rank_average(normalized: Mapping[str, np.ndarray], detectors: Sequence[str]) -> np.ndarray:
    """Mean of the selected detectors' normalized ranks, ignoring NaN; NaN where none is defined."""
    if not detectors:
        raise ValueError("rank_average needs at least one detector")
    stack = np.stack([normalized[name] for name in detectors], axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(stack, axis=0)


def select_top_detectors(aurocs: Mapping[str, float | None], k: int = TOP_K) -> tuple[str, ...]:
    ranked = sorted(
        ((name, value) for name, value in aurocs.items() if value is not None),
        key=lambda item: (-item[1], item[0]),
    )
    return tuple(name for name, _ in ranked[:k])


def leave_one_frame_out(
    *,
    frames: np.ndarray,
    truth: np.ndarray,
    raw: Mapping[str, np.ndarray],
    normalized: Mapping[str, np.ndarray],
    k: int = TOP_K,
    recall_floor: float = RECALL_FLOOR,
) -> LeaveOneFrameOutResult:
    """Select the top-k detectors and both thresholds on the other frames, score the held-out one.

    `raw` and `normalized` are per-detector arrays over the anchor cells; the normalized
    ranks come from the full series so thresholds stay in the confidence series' units.
    """
    frames = np.asarray(frames)
    truth = np.asarray(truth, dtype=bool)
    held_scores = np.full(truth.shape, np.nan)
    counts_f1 = np.zeros(4, dtype=int)
    counts_floor = np.zeros(4, dtype=int)
    folds: list[LeaveOneFrameOutFold] = []
    for frame in sorted(set(int(f) for f in frames)):
        test = frames == frame
        train = ~test
        aurocs = {name: auroc(raw[name][train], truth[train]) for name in raw}
        chosen = select_top_detectors(aurocs, k=k)
        if not chosen:
            folds.append(LeaveOneFrameOutFold(held_out_frame=frame, selected_detectors=()))
            continue
        combined = rank_average(normalized, chosen)
        held_scores[test] = combined[test]
        threshold_f1 = best_f1_threshold(combined[train], truth[train])
        threshold_floor = recall_floor_threshold(combined[train], truth[train], floor=recall_floor)
        for threshold, counts in ((threshold_f1, counts_f1), (threshold_floor, counts_floor)):
            if threshold is None:
                continue
            point = confusion_at(combined[test], truth[test], threshold)
            counts += np.array([point.tp, point.fp, point.fn, point.tn])
        folds.append(
            LeaveOneFrameOutFold(
                held_out_frame=frame,
                selected_detectors=chosen,
                best_f1_threshold=threshold_f1,
                recall_floor_threshold=threshold_floor,
            )
        )
    return LeaveOneFrameOutResult(
        folds=tuple(folds),
        pooled_auroc=auroc(held_scores, truth),
        best_f1=_operating_point(None, *(int(v) for v in counts_f1)),
        recall_floor=_operating_point(None, *(int(v) for v in counts_floor)),
    )


# ------------------------------------------------------------------------- feature series


@dataclass(frozen=True)
class DetectorSeries:
    """Per-frame per-part detector values for one run: `values[name][frame, target_index]`."""

    targets: tuple[str, ...]
    frame_count: int
    values: dict[str, np.ndarray]

    def cell_values(self, name: str, frames: Sequence[int], targets: Sequence[str]) -> np.ndarray:
        columns = [self.targets.index(t) for t in targets]
        return self.values[name][np.asarray(frames), np.asarray(columns)]


def _iou(a: np.ndarray | None, b: np.ndarray | None) -> float:
    a_present = a is not None and bool(a.any())
    b_present = b is not None and bool(b.any())
    if not a_present and not b_present:
        return float("nan")
    if not a_present or not b_present:
        return 0.0
    assert a is not None and b is not None
    if a.shape != b.shape:
        from .review_anchors import _resize_nearest

        b = _resize_nearest(b, a.shape)
    value = mask_iou(a, b, empty_union=float("nan"))
    assert value is not None
    return value


def area_jump_series(areas: np.ndarray, window: int = AREA_JUMP_WINDOW) -> np.ndarray:
    """|area_t / median(previous `window` areas) - 1| per frame; NaN at frame 0."""
    areas = np.asarray(areas, dtype=float)
    out = np.full(areas.shape, np.nan)
    for t in range(1, areas.size):
        previous = areas[max(0, t - window) : t]
        median = max(float(np.median(previous)), 1.0)
        out[t] = abs(areas[t] / median - 1.0)
    return out


def area_vs_seed_series(areas: np.ndarray) -> np.ndarray:
    areas = np.asarray(areas, dtype=float)
    return np.abs(np.log((areas + 1.0) / (areas[0] + 1.0)))


def read_observations(
    run_directory: Path, frame_count: int
) -> dict[int, dict[str, tuple[str | None, float | None]]]:
    """Frame -> label -> (mask uri, object_score) for frames below `frame_count`."""
    found: dict[int, dict[str, tuple[str | None, float | None]]] = {}
    with (run_directory / "observations.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            frame = int(row["analysis_frame_index"])
            if frame >= frame_count:
                continue
            entry = found.setdefault(frame, {})
            for item in row.get("objects", ()):
                label = item.get("label")
                if not label:
                    continue
                mask = item.get("mask") or {}
                score = item.get("object_score")
                entry[str(label)] = (
                    str(mask["uri"]) if mask.get("uri") else None,
                    float(score) if score is not None else None,
                )
    return found


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def consensus_error_series(
    consensus_dir: Path, *, view: str, targets: Sequence[str], frame_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """(error px, contradicted flag) per frame and target for one view; NaN without consensus."""
    error = np.full((frame_count, len(targets)), np.nan)
    with (consensus_dir / "per_frame.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            frame = int(row["analysis_frame_index"])
            if frame >= frame_count:
                continue
            for column, target in enumerate(targets):
                part = row.get("parts", {}).get(target) or {}
                if part.get("consensus_world_mm") is None:
                    continue
                value = (part.get("view_error_px") or {}).get(view)
                if value is not None:
                    error[frame, column] = float(value)
    contradicted = np.where(np.isfinite(error), 0.0, np.nan)
    manifest = json.loads((consensus_dir / "manifest.json").read_text(encoding="utf-8"))
    for episode in manifest.get("episodes", ()):
        if episode.get("view") != view or not episode.get("contradicts_majority"):
            continue
        if episode["target"] not in targets:
            continue
        column = targets.index(episode["target"])
        start, stop = (
            int(episode["start_frame"]),
            min(int(episode["end_frame_exclusive"]), frame_count),
        )
        span = slice(start, stop)
        contradicted[span, column] = np.where(np.isfinite(error[span, column]), 1.0, np.nan)
    return error, contradicted


def hull_disagreement_series(
    hull_dir: Path, *, view: str, targets: Sequence[str], frame_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """(1 - hull-vs-mask IoU, hull episode flag) per frame and target for one view."""
    disagreement = np.full((frame_count, len(targets)), np.nan)
    with np.load(hull_dir / "hull_series.npz") as series:
        for column, target in enumerate(targets):
            key = f"iou/{target}/{view}"
            if key in series.files:
                iou = np.asarray(series[key], dtype=float)[:frame_count]
                disagreement[: iou.size, column] = 1.0 - iou
    episode_flag = np.where(np.isfinite(disagreement), 0.0, np.nan)
    manifest = json.loads((hull_dir / "manifest.json").read_text(encoding="utf-8"))
    for episode in manifest.get("episodes", ()):
        if episode.get("view") != view or episode["target"] not in targets:
            continue
        column = targets.index(episode["target"])
        start, stop = (
            int(episode["start_frame"]),
            min(int(episode["end_frame_exclusive"]), frame_count),
        )
        span = slice(start, stop)
        episode_flag[span, column] = np.where(np.isfinite(disagreement[span, column]), 1.0, np.nan)
    return disagreement, episode_flag


def compute_series(
    *,
    run_directory: Path,
    targets: Sequence[str],
    frame_count: int,
    view: str,
    dam4sam_large: Path | None,
    dam4sam_tiny: Path | None,
    consensus_dir: Path | None,
    hull_dir: Path | None,
) -> DetectorSeries:
    """Every detector for every frame and part of the run, from existing artifacts only."""
    targets = tuple(targets)
    shape = (frame_count, len(targets))
    values = {name: np.full(shape, np.nan) for name in DETECTORS}
    areas = np.zeros(shape, dtype=float)
    run_rows = read_observations(run_directory, frame_count)
    run_cache = mask_cache.cache_for(run_directory)
    comparators: list[tuple[str, Path, dict[int, dict[str, tuple[str | None, float | None]]]]] = []
    for name, directory in (
        ("tracker_disagreement_large", dam4sam_large),
        ("tracker_disagreement_tiny", dam4sam_tiny),
    ):
        if directory is not None:
            comparators.append((name, directory, read_observations(directory, frame_count)))
    for frame in range(frame_count):
        row = run_rows.get(frame, {})
        masks: dict[str, np.ndarray | None] = {}
        for column, target in enumerate(targets):
            uri, score = row.get(target, (None, None))
            mask = run_cache.mask(uri) if uri is not None else None
            masks[target] = mask
            areas[frame, column] = float(mask.sum()) if mask is not None else 0.0
            if score is not None:
                values["sam3_score"][frame, column] = 1.0 - _sigmoid(score)
        for column, target in enumerate(targets):
            others = [masks[t] for t in targets if t != target]
            values["overlap_other_parts"][frame, column] = overlap_fraction(masks[target], others)
        for name, directory, rows in comparators:
            cache = mask_cache.cache_for(directory)
            other_row = rows.get(frame, {})
            for column, target in enumerate(targets):
                uri, _ = other_row.get(target, (None, None))
                other = cache.mask(uri) if uri is not None else None
                iou = _iou(masks[target], other)
                values[name][frame, column] = 1.0 - iou if np.isfinite(iou) else np.nan
    for column in range(len(targets)):
        values["area_jump"][:, column] = area_jump_series(areas[:, column])
        values["area_vs_seed"][:, column] = area_vs_seed_series(areas[:, column])
    if consensus_dir is not None:
        error, contradicted = consensus_error_series(
            consensus_dir, view=view, targets=targets, frame_count=frame_count
        )
        values["consensus_error_px"] = error
        values["consensus_contradicted"] = contradicted
    if hull_dir is not None:
        disagreement, flag = hull_disagreement_series(
            hull_dir, view=view, targets=targets, frame_count=frame_count
        )
        values["hull_disagreement"] = disagreement
        values["hull_episode"] = flag
    return DetectorSeries(targets=targets, frame_count=frame_count, values=values)


# ------------------------------------------------------------------------------ scorecard


@dataclass(frozen=True)
class Scorecard:
    manifest: DetectorScorecardManifest
    rows: tuple[DetectorConfidenceRow, ...]
    series: DetectorSeries
    combined_series: np.ndarray


def _class_counts(cells: Sequence[DetectorTruthCell]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for cell in cells:
        for key in (cell.failure_class, ALL_SUBSET):
            bucket = counts.setdefault(key, {"cells": 0, "failed": 0, "undefined": 0})
            bucket["cells"] += 1
            if cell.failed is None:
                bucket["undefined"] += 1
            elif cell.failed:
                bucket["failed"] += 1
    return counts


def build_scorecard(
    *,
    run_name: str,
    run_directory: Path,
    mask_set: ReviewAnchorMaskSet,
    anchors_root: Path,
    series: DetectorSeries,
    comparators: Mapping[str, Path],
    view: str,
    repository_root: Path,
    output_dir: Path,
    distractor_frames: Iterable[int] = DEFAULT_DISTRACTOR_FRAMES,
    recall_floor: float = RECALL_FLOOR,
    top_k: int = TOP_K,
) -> Scorecard:
    anchor_score = score_run(
        run_name=run_name, run_directory=run_directory, mask_set=mask_set, anchors_root=anchors_root
    )
    cells = truth_cells(anchor_score, mask_set.windows, distractor_frames=distractor_frames)
    defined = [cell for cell in cells if cell.failed is not None]
    frames = np.array([cell.analysis_frame_index for cell in defined])
    cell_targets = [cell.target for cell in defined]
    truth = np.array([bool(cell.failed) for cell in defined])
    classes = np.array([cell.failure_class for cell in defined])

    normalized_full = {name: normalized_ranks(series.values[name]) for name in DETECTORS}
    raw = {name: series.cell_values(name, frames, cell_targets) for name in DETECTORS}
    normalized = {
        name: normalized_full[name][frames, [series.targets.index(t) for t in cell_targets]]
        for name in DETECTORS
    }

    scores: list[DetectorScore] = []
    overall_auroc: dict[str, float | None] = {}
    for name in DETECTORS:
        overall = score_detector(name, ALL_SUBSET, raw[name], truth, recall_floor=recall_floor)
        overall_auroc[name] = overall.auroc
        scores.append(overall)
        scores.extend(_class_scores(name, raw[name], truth, classes, overall, recall_floor))

    top = select_top_detectors(overall_auroc, k=top_k)
    combined_cells = rank_average(normalized, top) if top else np.full(truth.shape, np.nan)
    combined_overall = score_detector(
        COMBINED_NAME, ALL_SUBSET, combined_cells, truth, recall_floor=recall_floor
    )
    combined_scores = [combined_overall]
    combined_scores.extend(
        _class_scores(COMBINED_NAME, combined_cells, truth, classes, combined_overall, recall_floor)
    )
    loo = leave_one_frame_out(
        frames=frames,
        truth=truth,
        raw=raw,
        normalized=normalized,
        k=top_k,
        recall_floor=recall_floor,
    )

    combined_series = (
        rank_average(normalized_full, top)
        if top
        else np.full((series.frame_count, len(series.targets)), np.nan)
    )
    suspicion_threshold = (
        combined_overall.recall_floor.threshold
        if combined_overall.recall_floor is not None
        else None
    )
    abstain_confidence = None if suspicion_threshold is None else 1.0 - suspicion_threshold
    truth_lookup = {(c.analysis_frame_index, c.target): c.failed for c in cells}
    anchor_frames = set(mask_set.frames)
    rows: list[DetectorConfidenceRow] = []
    for frame in range(series.frame_count):
        for column, target in enumerate(series.targets):
            combined = float(combined_series[frame, column])
            defined_value = math.isfinite(combined)
            confidence = 1.0 - combined if defined_value else None
            rows.append(
                DetectorConfidenceRow(
                    analysis_frame_index=frame,
                    target=target,
                    detectors={
                        name: (
                            float(series.values[name][frame, column])
                            if math.isfinite(float(series.values[name][frame, column]))
                            else None
                        )
                        for name in DETECTORS
                    },
                    combined_suspicion=_clip01(combined) if defined_value else None,
                    confidence=_clip01(confidence) if confidence is not None else None,
                    abstain=(
                        True
                        if confidence is None
                        else (abstain_confidence is not None and confidence <= abstain_confidence)
                    ),
                    is_anchor_frame=frame in anchor_frames,
                    anchor_truth_failed=truth_lookup.get((frame, target)),
                )
            )
    class_frames: dict[str, list[int]] = {}
    for frame in mask_set.frames:
        label = failure_class_for_frame(
            frame, mask_set.windows, distractor_frames=distractor_frames
        )
        class_frames.setdefault(label, []).append(frame)
    manifest = DetectorScorecardManifest(
        manifest_kind="detector_scorecard",
        run_name=run_name,
        run_directory=relative_uri(run_directory.resolve(), repository_root),
        run_observations=fingerprint(run_directory / "observations.jsonl", repository_root),
        anchor_set=fingerprint(anchors_root / MASK_SET_NAME, repository_root),
        comparators={
            name: relative_uri(path.resolve(), repository_root)
            for name, path in comparators.items()
        },
        view=view,
        frame_count=series.frame_count,
        targets=series.targets,
        anchor_frames=mask_set.frames,
        truth_rule=TRUTH_RULE,
        failure_class_frames={k: tuple(v) for k, v in class_frames.items()},
        class_counts=_class_counts(cells),
        truth_cells=cells,
        detectors=DETECTORS,
        detector_definitions=DETECTOR_DEFINITIONS,
        scores=tuple(scores),
        top_detectors=top,
        combined=tuple(combined_scores),
        leave_one_frame_out=loo,
        abstain_confidence_threshold=abstain_confidence,
        confidence_uri=relative_uri((output_dir / "confidence.jsonl").resolve(), repository_root),
        plot_uri=relative_uri((output_dir / "confidence_timeline.png").resolve(), repository_root),
        generated_at=datetime.now(UTC),
        claim_boundary=CLAIM_BOUNDARY,
    )
    return Scorecard(
        manifest=manifest, rows=tuple(rows), series=series, combined_series=combined_series
    )


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _class_scores(
    name: str,
    values: np.ndarray,
    truth: np.ndarray,
    classes: np.ndarray,
    overall: DetectorScore,
    recall_floor: float,
) -> list[DetectorScore]:
    out = []
    # Classes in order of first appearance (cells are in frame order), not alphabetical.
    for label in dict.fromkeys(classes.tolist()):
        keep = classes == label
        out.append(
            score_detector(
                name,
                label,
                values[keep],
                truth[keep],
                overall_best_f1=overall.best_f1.threshold if overall.best_f1 else None,
                overall_recall_floor=(
                    overall.recall_floor.threshold if overall.recall_floor else None
                ),
                recall_floor=recall_floor,
            )
        )
    return out


# ------------------------------------------------------------------------ anchor proposals


def propose_anchor_frames(
    *,
    run_name: str,
    scorecard: Scorecard,
    scorecard_uri: str,
    existing_anchor_frames: Sequence[int],
    seed: int,
    frame_step: int = 10,
    exclusion_radius: int = 20,
    ranked_count: int = 8,
    random_count: int = 5,
    min_spacing: int | None = None,
) -> ProposedAnchorFrames:
    """Top-`ranked_count` non-anchored frames by combined suspicion plus a seeded random draw.

    Frames within `exclusion_radius` of an existing anchor are never proposed; proposals keep
    at least `min_spacing` (default: the exclusion radius) between each other.
    """
    if min_spacing is None:
        min_spacing = exclusion_radius
    series = scorecard.series
    combined = scorecard.combined_series
    existing = tuple(sorted(set(existing_anchor_frames)))
    eligible = [
        frame
        for frame in range(0, series.frame_count, frame_step)
        if all(abs(frame - anchor) > exclusion_radius for anchor in existing)
    ]
    top = scorecard.manifest.top_detectors
    normalized = {name: normalized_ranks(series.values[name]) for name in top}

    def frame_suspicion(frame: int) -> tuple[float, int]:
        row = combined[frame]
        if not np.isfinite(row).any():
            return (float("-inf"), 0)
        column = int(np.nanargmax(row))
        return (float(row[column]), column)

    ranked = sorted(eligible, key=lambda f: (-frame_suspicion(f)[0], f))
    # Greedy spacing: two frames 10 apart in the same episode are one piece of evidence.
    chosen: list[int] = []
    for frame in ranked:
        if len(chosen) >= ranked_count or not np.isfinite(frame_suspicion(frame)[0]):
            break
        if all(abs(frame - other) >= min_spacing for other in chosen):
            chosen.append(frame)
    proposals: list[ProposedAnchorFrame] = []
    for rank, frame in enumerate(chosen, start=1):
        suspicion, column = frame_suspicion(frame)
        target = series.targets[column]
        contributions = {
            name: float(normalized[name][frame, column])
            for name in top
            if np.isfinite(normalized[name][frame, column])
        }
        lead = max(contributions, key=contributions.get) if contributions else None
        lead_text = (
            f"; led by {lead} = {series.values[lead][frame, column]:.3g} "
            f"(rank {contributions[lead]:.2f})"
            if lead is not None
            else ""
        )
        proposals.append(
            ProposedAnchorFrame(
                analysis_frame_index=frame,
                selection="detector_ranked",
                rank=rank,
                combined_suspicion=_clip01(suspicion),
                target=target,
                reason=(
                    f"{target}: combined suspicion {suspicion:.2f} (rank-average of "
                    f"{', '.join(top)}){lead_text}"
                ),
            )
        )
    remaining = [f for f in eligible if all(abs(f - other) >= min_spacing for other in chosen)]
    rng = np.random.default_rng(seed)
    draw = sorted(
        int(f) for f in rng.choice(remaining, size=min(random_count, len(remaining)), replace=False)
    )
    for frame in draw:
        suspicion, column = frame_suspicion(frame)
        proposals.append(
            ProposedAnchorFrame(
                analysis_frame_index=frame,
                selection="random",
                combined_suspicion=_clip01(suspicion) if np.isfinite(suspicion) else None,
                target=series.targets[column] if np.isfinite(suspicion) else None,
                reason=(
                    f"uniform random draw over eligible frames (seed {seed}); tests the detectors "
                    "rather than confirming them"
                ),
            )
        )
    return ProposedAnchorFrames(
        manifest_kind="proposed_anchor_frames",
        run_name=run_name,
        scorecard_uri=scorecard_uri,
        frame_step=frame_step,
        exclusion_radius_frames=exclusion_radius,
        min_spacing_frames=min_spacing,
        existing_anchor_frames=existing,
        random_seed=seed,
        frames=tuple(proposals),
        claim_boundary=CLAIM_BOUNDARY,
    )


# --------------------------------------------------------------------------------- reports


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    return f"{value:.{digits}f}"


def _fmt_threshold(value: float | None, detector: str | None = None) -> str:
    if value is None:
        return "-"
    if detector == "sam3_score" and 0.0 < value < 1.0:
        # The detector is 1 - sigmoid(logit); the logit is what the observations carry.
        return f"{value:.3g} (logit {math.log((1.0 - value) / value):.2f})"
    return f"{value:.3g}"


def _point_cells(
    point: DetectorOperatingPoint | None,
    *,
    with_threshold: bool = True,
    detector: str | None = None,
) -> list[str]:
    if point is None:
        return ["-"] * (5 if with_threshold else 4)
    cells = [_fmt_threshold(point.threshold, detector)] if with_threshold else []
    cells += [
        _fmt(point.precision),
        _fmt(point.recall),
        _fmt(point.f1),
        f"{point.tp}/{point.fp}/{point.fn}/{point.tn}",
    ]
    return cells


def _score_table(
    scores: Sequence[DetectorScore], *, title: str, show_subset: bool = False
) -> list[str]:
    header = [
        "detector",
        *(["subset"] if show_subset else []),
        "cells (fail / ok / undefined)",
        "AUROC",
        "F1-max thr",
        "P",
        "R",
        "F1",
        "TP/FP/FN/TN",
        f"R>={RECALL_FLOOR:g} thr",
        "P",
        "R",
        "F1",
        "TP/FP/FN/TN",
    ]
    lines = [f"### {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for score in scores:
        row = [
            f"`{score.detector}`",
            *([score.subset] if show_subset else []),
            f"{score.cells} ({score.positives} / {score.negatives} / {score.undefined})",
            _fmt(score.auroc),
            *_point_cells(score.best_f1, detector=score.detector),
            *_point_cells(score.recall_floor, detector=score.detector),
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def _at_overall_table(scores: Sequence[DetectorScore], classes: Sequence[str]) -> list[str]:
    header = ["detector", *[f"{c} TP/FP/FN/TN" for c in classes]]
    lines = [
        f"### Per-class counts at the overall R>={RECALL_FLOOR:g} threshold",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    by_key = {(s.detector, s.subset): s for s in scores}
    for detector in sorted(
        {s.detector for s in scores}, key=lambda d: DETECTORS.index(d) if d in DETECTORS else 99
    ):
        row = [f"`{detector}`"]
        for label in classes:
            score = by_key.get((detector, label))
            point = score.at_overall_recall_floor if score else None
            row.append("-" if point is None else f"{point.tp}/{point.fp}/{point.fn}/{point.tn}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def _failed_cell_text(cell: DetectorTruthCell) -> str:
    detail = _fmt(cell.iou, 2) if cell.iou is not None else f"{cell.run_area} px hidden"
    return f"{cell.analysis_frame_index} {cell.target} ({detail})"


def markdown_report(manifest: DetectorScorecardManifest) -> str:
    counts = manifest.class_counts
    classes = [c for c in counts if c != ALL_SUBSET]
    lines = [
        f"# Detector scorecard: `{manifest.run_name}`",
        "",
        f"**{manifest.claim_boundary}**",
        "",
        f"- run: `{manifest.run_directory}` ({manifest.frame_count} frames, view {manifest.view})",
        f"- anchors: `{manifest.anchor_set.uri}` (sha256 `{manifest.anchor_set.sha256[:12]}...`)",
        *[f"- {name}: `{path}`" for name, path in manifest.comparators.items()],
        f"- truth rule: {manifest.truth_rule}",
        f"- generated: {manifest.generated_at.isoformat()}",
        "",
        "## Truth (small counts, say so)",
        "",
        "| class | anchor frames | cells | failed | ok | undefined |",
        "|---|---|---|---|---|---|",
    ]
    for label in [*classes, ALL_SUBSET]:
        bucket = counts[label]
        frames = manifest.failure_class_frames.get(label, ())
        lines.append(
            f"| {label} | {', '.join(str(f) for f in frames) if frames else '(all)'} | "
            f"{bucket['cells']} | {bucket['failed']} | "
            f"{bucket['cells'] - bucket['failed'] - bucket['undefined']} | {bucket['undefined']} |"
        )
    failed_cells = [c for c in manifest.truth_cells if c.failed]
    lines += [
        "",
        "Failed cells: "
        + (", ".join(_failed_cell_text(c) for c in failed_cells) if failed_cells else "none")
        + ".",
        "",
        "## Detectors",
        "",
    ]
    for name in manifest.detectors:
        lines.append(f"- `{name}`: {manifest.detector_definitions.get(name, '')}")
    lines += [
        "",
        "Thresholds: `F1-max` is the score value maximising F1 on the subset; "
        f"`R>={RECALL_FLOOR:g}` is the highest score value still recalling at least "
        f"{RECALL_FLOOR:g} of the failures. Predict failed where score >= threshold. "
        "Undefined (NaN) cells are excluded from that detector's counts.",
        "",
    ]
    overall = [s for s in manifest.scores if s.subset == ALL_SUBSET]
    lines += _score_table(overall, title="Overall (52 cells, one view)")
    for label in classes:
        subset = [s for s in manifest.scores if s.subset == label]
        bucket = counts[label]
        lines += _score_table(
            subset,
            title=(
                f"Class `{label}` ({bucket['cells']} cells, {bucket['failed']} failed; thresholds "
                "chosen inside the class)"
            ),
        )
    lines += _at_overall_table(manifest.scores, classes)
    lines += [
        "## Combined detector",
        "",
        f"Top-{len(manifest.top_detectors)} by overall AUROC: "
        + ", ".join(f"`{d}`" for d in manifest.top_detectors)
        + ". Combined = mean of their normalized ranks over the full "
        f"{manifest.frame_count}-frame series (NaN ignored); confidence = 1 - combined; "
        f"abstain where confidence <= {_fmt(manifest.abstain_confidence_threshold)} "
        f"(the R>={RECALL_FLOOR:g} point).",
        "",
    ]
    lines += _score_table(
        manifest.combined,
        title="Combined, in-sample (detector choice and thresholds saw all 13 frames)",
        show_subset=True,
    )
    loo = manifest.leave_one_frame_out
    lines += [
        "### Combined, leave-one-frame-out (top-3 and thresholds chosen on 12 frames, "
        "scored on the 13th)",
        "",
        "| pooled AUROC | F1-max P | R | F1 | TP/FP/FN/TN | "
        f"R>={RECALL_FLOOR:g} P | R | F1 | TP/FP/FN/TN |",
        "|---|---|---|---|---|---|---|---|---|",
        "| "
        + " | ".join(
            [
                _fmt(loo.pooled_auroc),
                *_point_cells(loo.best_f1, with_threshold=False),
                *_point_cells(loo.recall_floor, with_threshold=False),
            ]
        )
        + " |",
        "",
        "| held-out frame | selected detectors | F1-max thr | R>=0.8 thr |",
        "|---|---|---|---|",
    ]
    for fold in loo.folds:
        lines.append(
            f"| {fold.held_out_frame} | {', '.join(fold.selected_detectors) or '-'} | "
            f"{_fmt_threshold(fold.best_f1_threshold)} | "
            f"{_fmt_threshold(fold.recall_floor_threshold)} |"
        )
    lines += [
        "",
        "## Artifacts",
        "",
        f"- `{manifest.confidence_uri}`: per frame, per part detector values, combined suspicion, "
        "confidence, abstain.",
    ]
    if manifest.plot_uri:
        lines.append(
            f"- `{manifest.plot_uri}`: per-part confidence timeline with anchor frames marked."
        )
    if manifest.proposed_anchors_uri:
        lines.append(
            f"- `{manifest.proposed_anchors_uri}`: proposed extra anchor frames for the human."
        )
    lines.append("")
    return "\n".join(lines)


def render_timeline(
    scorecard: Scorecard, output_path: Path, *, abstain_confidence: float | None
) -> Path:
    """Per-part confidence over the run with anchor frames marked and truth coloured."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = scorecard.series
    manifest = scorecard.manifest
    truth = {(c.analysis_frame_index, c.target): c.failed for c in manifest.truth_cells}
    frames = np.arange(series.frame_count)
    figure, axes = plt.subplots(
        len(series.targets), 1, figsize=(14, 2.2 * len(series.targets)), sharex=True
    )
    axes = np.atleast_1d(axes)
    for column, (axis, target) in enumerate(zip(axes, series.targets, strict=True)):
        confidence = 1.0 - scorecard.combined_series[:, column]
        axis.plot(frames, confidence, color="#333333", linewidth=0.8, label="confidence")
        if abstain_confidence is not None:
            axis.axhline(abstain_confidence, color="#888888", linestyle=":", linewidth=0.8)
            axis.fill_between(
                frames,
                0,
                1,
                where=np.isfinite(confidence) & (confidence <= abstain_confidence),
                color="#f4c7c3",
                alpha=0.35,
                linewidth=0,
                label="abstain",
            )
        for frame in manifest.anchor_frames:
            failed = truth.get((frame, target))
            colour = "#bbbbbb" if failed is None else ("#d62728" if failed else "#2ca02c")
            axis.axvline(frame, color=colour, linewidth=0.9, alpha=0.9)
            value = confidence[frame] if np.isfinite(confidence[frame]) else 0.5
            axis.plot(frame, value, marker="o", color=colour, markersize=5)
        axis.set_ylim(-0.02, 1.02)
        axis.set_ylabel(target)
        axis.grid(True, alpha=0.2)
    axes[-1].set_xlabel("analysis frame")
    axes[0].set_title(
        f"{manifest.run_name}: confidence = 1 - rank-average of "
        f"{', '.join(manifest.top_detectors)}; anchor frames: red = failed, green = ok "
        "(13 frames, one view, review evidence not ground truth)",
        fontsize=9,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=110)
    plt.close(figure)
    return output_path


def write_scorecard(
    scorecard: Scorecard, output_dir: Path, *, proposals: ProposedAnchorFrames | None = None
) -> DetectorScorecardManifest:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = scorecard.manifest
    with (output_dir / "confidence.jsonl").open("w", encoding="utf-8") as handle:
        for row in scorecard.rows:
            handle.write(row.model_dump_json() + "\n")
    render_timeline(
        scorecard,
        output_dir / "confidence_timeline.png",
        abstain_confidence=manifest.abstain_confidence_threshold,
    )
    if proposals is not None:
        (output_dir / "proposed_anchor_frames.json").write_text(
            proposals.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
    (output_dir / "scorecard.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "scorecard.md").write_text(markdown_report(manifest), encoding="utf-8")
    return manifest


def write_confidence_only(
    *,
    run_name: str,
    run_directory: Path,
    series: DetectorSeries,
    comparators: Mapping[str, Path],
    view: str,
    repository_root: Path,
    output_dir: Path,
    top_detectors: Sequence[str],
    abstain_confidence: float | None,
    thresholds_from: Path,
) -> Path:
    """Confidence series for a run without anchors (`--no-truth`).

    Nothing is scored: the detector choice and the abstain threshold are *carried over* from a
    scored scorecard on another run (`thresholds_from`, e.g. recording 1's `pm-append`), and
    the rank normalisation is over this run's own frames. The output is the raw input for a
    gate on a recording with no labels yet, not a calibrated detector.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_full = {name: normalized_ranks(series.values[name]) for name in DETECTORS}
    combined_series = rank_average(normalized_full, tuple(top_detectors))
    rows: list[DetectorConfidenceRow] = []
    for frame in range(series.frame_count):
        for column, target in enumerate(series.targets):
            combined = float(combined_series[frame, column])
            defined = math.isfinite(combined)
            confidence = 1.0 - combined if defined else None
            rows.append(
                DetectorConfidenceRow(
                    analysis_frame_index=frame,
                    target=target,
                    detectors={
                        name: (
                            float(series.values[name][frame, column])
                            if math.isfinite(float(series.values[name][frame, column]))
                            else None
                        )
                        for name in DETECTORS
                    },
                    combined_suspicion=_clip01(combined) if defined else None,
                    confidence=_clip01(confidence) if confidence is not None else None,
                    abstain=(
                        True
                        if confidence is None
                        else (abstain_confidence is not None and confidence <= abstain_confidence)
                    ),
                )
            )
    with (output_dir / "confidence.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = np.arange(series.frame_count)
    figure, axes = plt.subplots(
        len(series.targets), 1, figsize=(14, 2.2 * len(series.targets)), sharex=True
    )
    axes = np.atleast_1d(axes)
    for column, (axis, target) in enumerate(zip(axes, series.targets, strict=True)):
        confidence_series = 1.0 - combined_series[:, column]
        axis.plot(frames, confidence_series, color="#333333", linewidth=0.8)
        if abstain_confidence is not None:
            axis.axhline(abstain_confidence, color="#888888", linestyle=":", linewidth=0.8)
            axis.fill_between(
                frames,
                0,
                1,
                where=np.isfinite(confidence_series) & (confidence_series <= abstain_confidence),
                color="#f4c7c3",
                alpha=0.35,
                linewidth=0,
            )
        axis.set_ylim(-0.02, 1.02)
        axis.set_ylabel(target)
        axis.grid(True, alpha=0.2)
    axes[-1].set_xlabel("analysis frame")
    axes[0].set_title(
        f"{run_name}: confidence = 1 - rank-average of {', '.join(top_detectors)}; no anchors on "
        "this run (thresholds carried over, nothing scored)",
        fontsize=9,
    )
    figure.tight_layout()
    figure.savefig(output_dir / "confidence_timeline.png", dpi=110)
    plt.close(figure)
    abstain_by_target = {
        target: int(sum(1 for r in rows if r.target == target and r.abstain))
        for target in series.targets
    }
    summary = {
        "manifest_kind": "detector_confidence_only",
        "run_name": run_name,
        "run_directory": relative_uri(run_directory.resolve(), repository_root),
        "run_observations": fingerprint(
            run_directory / "observations.jsonl", repository_root
        ).model_dump(mode="json"),
        "comparators": {
            name: relative_uri(path.resolve(), repository_root)
            for name, path in comparators.items()
        },
        "view": view,
        "frame_count": series.frame_count,
        "targets": list(series.targets),
        "top_detectors": list(top_detectors),
        "abstain_confidence_threshold": abstain_confidence,
        "thresholds_carried_from": fingerprint(thresholds_from, repository_root).model_dump(
            mode="json"
        ),
        "rows": len(rows),
        "abstain_rows_by_target": abstain_by_target,
        "abstain_fraction": sum(abstain_by_target.values()) / max(1, len(rows)),
        "confidence_uri": relative_uri(output_dir / "confidence.jsonl", repository_root),
        "plot_uri": relative_uri(output_dir / "confidence_timeline.png", repository_root),
        "generated_at": datetime.now(UTC).isoformat(),
        "claim_boundary": (
            "No anchors exist on this run: nothing here is scored. The detector set and the "
            "abstain threshold come from a scorecard on another run and may not transfer (the "
            "recording-1 scorecard found its thresholds do not transfer leave-one-frame-out); "
            "the series is the raw input for a gate, not the gate."
        ),
    }
    path = output_dir / "confidence_only.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return path


# ------------------------------------------------------------------------------------ CLI


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score label-free failure detectors against the human review anchors (13 frames, "
            "one view, review evidence not ground truth) and write a per-frame per-part "
            "confidence series with an abstain flag."
        )
    )
    parser.add_argument(
        "--run", required=True, metavar="NAME=PATH", help="the tracked run to score"
    )
    parser.add_argument(
        "--anchors", type=Path, default=Path("runs/human-review-anchors-first-minute")
    )
    parser.add_argument("--dam4sam-large", type=Path, default=None)
    parser.add_argument("--dam4sam-tiny", type=Path, default=None)
    parser.add_argument("--consensus", type=Path, default=None, help="multiview consensus run dir")
    parser.add_argument("--hull", type=Path, default=None, help="visual hull run dir")
    parser.add_argument("--view", default="C10379")
    parser.add_argument("--frame-count", type=int, default=1800)
    parser.add_argument(
        "--output-root", type=Path, default=Path("runs/detector-scorecard-20260920")
    )
    parser.add_argument(
        "--propose-anchors",
        action="store_true",
        help="also rank non-anchored frames for the human and add a seeded random draw",
    )
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument(
        "--distractor-frame",
        type=int,
        action="append",
        default=None,
        help=f"anchor frames classed distractor (default {DEFAULT_DISTRACTOR_FRAMES})",
    )
    parser.add_argument(
        "--no-truth",
        type=Path,
        default=None,
        metavar="SCORECARD_JSON",
        help=(
            "write only confidence.jsonl and the timeline for a run without anchors, carrying "
            "the top detectors and abstain threshold from this scored scorecard.json"
        ),
    )
    parser.add_argument(
        "--target",
        action="append",
        default=None,
        help="targets (parts) of the run when --no-truth (default: the four parts)",
    )
    args = parser.parse_args()
    root = Path.cwd().resolve()
    if args.no_truth is not None:
        run_name, run_directory = resolve_run_directory(args.run)
        reference = json.loads(args.no_truth.read_text(encoding="utf-8"))
        comparators_nt: dict[str, Path] = {}
        for key, value in (("consensus", args.consensus), ("hull", args.hull)):
            if value is not None:
                comparators_nt[key] = value.resolve()
        targets = (
            tuple(args.target) if args.target else ("chassis", "interior", "rear_body", "cabin")
        )
        series = compute_series(
            run_directory=run_directory,
            targets=targets,
            frame_count=args.frame_count,
            view=args.view,
            dam4sam_large=None,
            dam4sam_tiny=None,
            consensus_dir=comparators_nt.get("consensus"),
            hull_dir=comparators_nt.get("hull"),
        )
        output_dir = (args.output_root / run_name).resolve()
        path = write_confidence_only(
            run_name=run_name,
            run_directory=run_directory,
            series=series,
            comparators=comparators_nt,
            view=args.view,
            repository_root=root,
            output_dir=output_dir,
            top_detectors=tuple(reference["top_detectors"]),
            abstain_confidence=reference.get("abstain_confidence_threshold"),
            thresholds_from=args.no_truth.resolve(),
        )
        print(path.read_text(encoding="utf-8"), end="")
        return
    try:
        run_name, run_directory = resolve_run_directory(args.run)
        mask_set, anchors_root = load_mask_set(args.anchors)
        comparators: dict[str, Path] = {}
        resolved: dict[str, Path | None] = {}
        for key, value in (
            ("dam4sam_large", args.dam4sam_large),
            ("dam4sam_tiny", args.dam4sam_tiny),
        ):
            resolved[key] = resolve_run_directory(str(value))[1] if value is not None else None
            if resolved[key] is not None:
                comparators[key] = resolved[key]
        for key, value in (("consensus", args.consensus), ("hull", args.hull)):
            resolved[key] = value.resolve() if value is not None else None
            if value is not None:
                comparators[key] = value.resolve()
        output_dir = (args.output_root / run_name).resolve()
        series = compute_series(
            run_directory=run_directory,
            targets=mask_set.targets,
            frame_count=args.frame_count,
            view=args.view,
            dam4sam_large=resolved["dam4sam_large"],
            dam4sam_tiny=resolved["dam4sam_tiny"],
            consensus_dir=resolved["consensus"],
            hull_dir=resolved["hull"],
        )
        scorecard = build_scorecard(
            run_name=run_name,
            run_directory=run_directory,
            mask_set=mask_set,
            anchors_root=anchors_root,
            series=series,
            comparators=comparators,
            view=args.view,
            repository_root=root,
            output_dir=output_dir,
            distractor_frames=(
                tuple(args.distractor_frame) if args.distractor_frame else DEFAULT_DISTRACTOR_FRAMES
            ),
        )
        proposals = None
        if args.propose_anchors:
            proposals_uri = relative_uri(output_dir / "proposed_anchor_frames.json", root)
            proposals = propose_anchor_frames(
                run_name=run_name,
                scorecard=scorecard,
                scorecard_uri=relative_uri(output_dir / "scorecard.json", root),
                existing_anchor_frames=mask_set.frames,
                seed=args.seed,
            )
            scorecard = Scorecard(
                manifest=scorecard.manifest.model_copy(
                    update={"proposed_anchors_uri": proposals_uri}
                ),
                rows=scorecard.rows,
                series=scorecard.series,
                combined_series=scorecard.combined_series,
            )
        manifest = write_scorecard(scorecard, output_dir, proposals=proposals)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(markdown_report(manifest), end="")
    print(f"Scorecard: {output_dir / 'scorecard.md'}")


if __name__ == "__main__":
    main()
