"""Adapters from the two FineBio producers to `FineBioObservation` rows (plan `p2-seeds`).

The tracker (`battle-multiview-tracks`) and the slice consume one row shape, in raw pixels
and raw frame indices.  Two tools produce the evidence in their own formats:

* `battle-finebio-detect` writes one JSONL per view with raw ``frame_index`` and
  ``detections[{class, class_id, score, box_xyxy_px}]`` (plus ``interpolated`` and
  ``source_frames`` on rows the CPU fallback filled in).  :func:`detections_to_observations`
  turns every box at or above a score into a ``detector`` row with ``slot = <class>#<rank>``
  (rank = same-class score rank in that frame, not an identity) and keeps ``interpolated`` in
  the provenance, so a reader can tell a detector box from a linear guess.
* The SAM3 worker (`battle-muggled-arms`, both modes) writes ``observations.jsonl`` in analysis
  frames (frame 0 = ``--start-frame``) with normalised boxes and mask PNGs.
  :func:`worker_to_observations` reads the masks back, drops isolated speckle components the
  way the worker's own box rule does (:func:`filter_mask_components`, recorded in the
  provenance), computes the **area centroid** (the preflight fixtures carry the bbox centre;
  the arms carry the real one), the mask bbox and area, maps analysis frame ``k`` to raw frame
  ``start_frame + k``, keeps the decoder's IoU
  or the tracker's presence logit as ``sam3_object_score`` and names the source
  (``sam3_decode`` for the memory-free decode, ``sam3_video`` for the video-memory tracker).
  A box-decode row carries the detector box that prompted it; a video row is matched to a
  same-class detector row of the same frame by IoU when detector rows are supplied.
  Sep 28 (`p0-axis-observations`): :func:`mask_axis_measurements` adds the mask's principal
  axis (two endpoints in pixels, elongation, width, skeleton residual) beside the centroid,
  so the pipettes can be tracked as 3D lines; the old fields are computed by the same code
  as before and a compact mask simply carries no axis.
  Sep 29 (v4, from `finebio_tipseg`): an elongated mask also carries ``tip_side``, the axis
  end with the longer thin tail (:func:`tip_side_from_tails`; None when the tails do not
  differ enough), and ``body_end_px``, the terminal centroid at each end
  (:func:`terminal_centroids`: the mean of the mask's last pixels along the axis, which sits
  on a thin tip where the fitted axis endpoint can be 30 px off it sideways).

The fpv observes only with a valid shipped pose: :func:`fpv_pose_validity` (from the trial's
pose file) or :func:`pose_validity_from_fixture` (from ``fpv_poses.json``) give the lookup,
:func:`fill_pose_valid` applies it.  Nothing here reads a frame of video; masks are read only
to measure them and are never copied (FineBio, non-commercial research).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .finebio_detect import box_iou
from .multiview_schemas import (
    FINEBIO_FPV_VIEW,
    BoxXYXY,
    FineBioObservation,
    ObservationSource,
    Vec2,
    write_jsonl,
)
from .observations import load_observations
from .schemas import FrameObservations, PerFrameObject

VIEWS: tuple[str, ...] = ("fpv", "T1", "T2", "T3", "T4", "T5")
DEFAULT_MIN_SCORE = 0.3
DEFAULT_MATCH_IOU = 0.3
# The worker derives its own box from the mask's 8-connected components with area >= this
# fraction of the largest (`muggled_worker.BOX_COMPONENT_KEEP_FRACTION`). SAM3.1 video-memory
# masks carry a few isolated positive pixels far from the object (Sep 25, p4-arms: 1-15 single
# pixels per mask), which would otherwise set the bbox; the same rule is applied here before a
# mask is measured, and the dropped pixels are recorded in the row's provenance.
MASK_COMPONENT_KEEP_FRACTION = 0.20
# Sep 28 (p0-axis-observations): the mask axis. A mask whose elongation (sqrt of the ratio of
# the second-moment eigenvalues) is under the threshold is compact and gets no axis; the
# pixel cloud is stride-sampled above AXIS_MAX_POINTS (1080p masks, ~180k arm-b rows); the
# RANSAC inlier band on the skeleton is a quarter of the PCA minor width, clamped.
AXIS_ELONGATION_THRESHOLD = 2.5
AXIS_MAX_POINTS = 50_000
AXIS_MIN_PIXELS = 10
AXIS_MIN_SKELETON_POINTS = 5
AXIS_RANSAC_ITERATIONS = 200
AXIS_RANSAC_MIN_INLIER_PX = 1.5
AXIS_RANSAC_MAX_INLIER_PX = 6.0
AXIS_WIDTH_BINS = 48
AXIS_WIDTH_MIN_BIN_PX = 2.0
AXIS_WIDTH_MIN_BIN_PIXELS = 3
# The fraction of the axis length at each end over which the end width is measured.
AXIS_END_BAND = 0.2
AXIS_END_MIN_PIXELS = 6
# Sep 29 (v4): the long-thin-tail tip side, moved here from `finebio_tipseg` so every row
# carries it. The across-axis width is read every TAIL_STEP_PX along the axis; walked in from
# each end, the bins under TAIL_FRACTION of the body width (the 75th percentile of the
# profile) are that end's tail, ending at the first two consecutive bins at or over it. The
# end with the longer tail is the tip when the tails differ by more than
# TAIL_MIN_DIFFERENCE_WIDTHS body widths (tipseg asked for 1 cm at the pipette's depth; the
# observation layer has no depth, and on trial 1's rows 1 cm is 0.28 body widths at the
# median) and the longer is at least TAIL_SIDE_RATIO times the shorter. The shaft and cone
# make a long thin run and the plunger stem a short one, so this picks the tip of a pipette
# resting grip-up where the thinner end band alone picks the plunger. Validated by eye on
# the single-channel pipettes; the 8-channel's manifold is its wide end and the tracker keeps
# the class rule for it. The terminal centroid is the mean of the mask pixels within
# TERMINAL_DEPTH_PX of the far extreme along the axis at each end.
TAIL_STEP_PX = 2.0
TAIL_FRACTION = 0.4
TAIL_MIN_DIFFERENCE_WIDTHS = 0.3
TAIL_SIDE_RATIO = 1.5
TAIL_MIN_BIN_PIXELS = 2
TERMINAL_DEPTH_PX = 3.0
PoseValidity = Callable[[int], bool]


def _box(values: Sequence[float]) -> BoxXYXY:
    x0, y0, x1, y1 = (round(float(v), 1) for v in values)
    return (x0, y0, x1, y1)


def _score(value: Any) -> float | None:
    if value is None:
        return None
    return round(min(max(float(value), 0.0), 1.0), 4)


def slot_label(object_class: str, rank: int) -> str:
    return f"{object_class}#{rank}"


def class_of_slot(slot: str) -> str:
    """``<class>#<k>`` -> ``<class>``; a label without ``#`` is its own class."""
    return slot.rsplit("#", 1)[0] if "#" in slot else slot


# -------------------------------------------------------------------------------- pose validity


def fpv_pose_validity(trial: str) -> PoseValidity:
    """Validity of the shipped per-frame fpv pose, from the trial's pose file under data/."""
    from .finebio_cameras import fpv_poses

    rets = fpv_poses(trial)[0]

    def valid(frame: int) -> bool:
        return 0 <= frame < len(rets) and bool(rets[frame])

    return valid


def pose_validity_from_fixture(path: Path) -> PoseValidity:
    """Validity from a fixtures-shaped ``fpv_poses.json`` (``frames: {"<frame>": {valid}}``);
    a frame the file does not list is invalid."""
    frames = json.loads(Path(path).read_text(encoding="utf-8"))["frames"]
    valid_frames = {int(frame) for frame, pose in frames.items() if pose.get("valid")}
    return lambda frame: frame in valid_frames


def fill_pose_valid(
    rows: Iterable[FineBioObservation], validity: PoseValidity
) -> list[FineBioObservation]:
    """Copy of `rows` with every fpv row's ``pose_valid`` set from `validity`."""
    out = []
    for row in rows:
        if row.view == FINEBIO_FPV_VIEW:
            row = row.model_copy(update={"pose_valid": bool(validity(row.frame_index))})
        out.append(row)
    return out


# ------------------------------------------------------------------------- detector -> rows


def read_detection_rows(path: Path) -> list[dict[str, Any]]:
    """The per-frame records of one `battle-finebio-detect` view file, in file order."""
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def detection_row_to_observations(
    record: Mapping[str, Any],
    *,
    min_score: float = DEFAULT_MIN_SCORE,
    pose_valid: bool = True,
) -> list[FineBioObservation]:
    """Every box of one per-frame detector record at ``score >= min_score`` as a detector row."""
    view = str(record["view"])
    frame = int(record["frame_index"])
    interpolated = bool(record.get("interpolated", False))
    provenance: dict[str, Any] = {}
    if interpolated:
        provenance = {"interpolated": True}
        if record.get("source_frames") is not None:
            provenance["source_frames"] = [int(f) for f in record["source_frames"]]
    kept = [d for d in record["detections"] if float(d["score"]) >= min_score]
    kept.sort(key=lambda d: -float(d["score"]))
    rank: dict[str, int] = {}
    rows = []
    for det in kept:
        cls = str(det["class"])
        k = rank.get(cls, 0)
        rank[cls] = k + 1
        rows.append(
            FineBioObservation(
                view=view,
                frame_index=frame,
                slot=slot_label(cls, k),
                object_class=cls,
                detector_score=_score(det["score"]),
                box_xyxy_px=_box(det["box_xyxy_px"]),
                pose_valid=pose_valid,
                source="detector",
                provenance=dict(provenance),
            )
        )
    return rows


def detections_to_observations(
    detections_dir: Path,
    views: Sequence[str] = VIEWS,
    min_score: float = DEFAULT_MIN_SCORE,
    *,
    pose_valid: PoseValidity | None = None,
    frames: Collection[int] | None = None,
) -> list[FineBioObservation]:
    """`battle-finebio-detect` output -> detector rows for `views` (files ``<view>.jsonl``).

    ``pose_valid`` is required when the fpv is among the views (the fpv observes only with a
    valid shipped pose; see :func:`fpv_pose_validity`).  ``frames`` restricts the raw frames.
    Boxes are rounded to 0.1 px and scores to four decimals, as the preflight fixtures are.
    """
    if FINEBIO_FPV_VIEW in views and pose_valid is None:
        raise ValueError("fpv rows need a pose validity lookup (fpv_pose_validity(trial))")
    detections_dir = Path(detections_dir)
    rows: list[FineBioObservation] = []
    wanted = None if frames is None else {int(f) for f in frames}
    for view in views:
        for record in read_detection_rows(detections_dir / f"{view}.jsonl"):
            frame = int(record["frame_index"])
            if wanted is not None and frame not in wanted:
                continue
            valid = bool(pose_valid(frame)) if view == FINEBIO_FPV_VIEW and pose_valid else True
            rows.extend(
                detection_row_to_observations(record, min_score=min_score, pose_valid=valid)
            )
    return rows


# --------------------------------------------------------------------------- worker -> rows


def mask_measurements(mask: np.ndarray) -> tuple[BoxXYXY, tuple[float, float], int] | None:
    """(mask bbox, area centroid, area) of a boolean mask, None when it is empty.

    The bbox is ``[x_min, y_min, x_max + 1, y_max + 1]`` and coordinates use pixel centres
    (``+ 0.5``), so a one-pixel mask at (10, 10) has bbox (10, 10, 11, 11) and centroid
    (10.5, 10.5), the same convention as a box centre.
    """
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    bbox = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
    centroid = (round(float(xs.mean()) + 0.5, 1), round(float(ys.mean()) + 0.5, 1))
    return bbox, centroid, int(xs.size)


@dataclass(frozen=True)
class MaskAxis:
    """The principal axis of one mask (plan `p0-axis-observations`, Sep 28).

    `axis_px` is the two endpoints in raw pixels, ordered by image y then x (top first);
    the order says nothing about tip or butt. `elongation` is sqrt(largest / smallest
    second-moment eigenvalue) of the mask as a region (1.0 is round; a w x L rectangle gives
    exactly L / w). `width_px` is the median across-axis extent sampled along the axis.
    `residual_px` is the RMS distance of the skeleton points to the fitted axis, the quality
    number a merged mask raises. `method` is ``ransac_skeleton`` (RANSAC line on the thinned
    mask, PCA as the seed), ``pca`` (the PCA axis: elongated with no usable skeleton, or a
    compact mask with no axis) or ``none`` (too few pixels); `reason` names why `axis_px` is
    None (``compact``, ``too_few_pixels``).
    """

    axis_px: tuple[Vec2, Vec2] | None
    elongation: float | None
    width_px: float | None
    residual_px: float | None
    method: str
    reason: str | None = None
    # Sep 29 (tip / butt by the width profile): the median across-axis width over the band
    # `AXIS_END_BAND` of the axis length at each end, in the order of `axis_px`; None on a
    # compact mask or when a band holds too few pixels.
    end_widths_px: tuple[float, float] | None = None
    # Sep 29 (v4): the axis end (index into `axis_px`) with the longer thin tail, None when
    # the tails do not differ enough (`tip_side_from_tails`); the terminal centroid at each
    # end, in the order of `axis_px` (`terminal_centroids`); and the tail lengths behind the
    # decision (provenance only).
    tip_side: int | None = None
    body_ends_px: tuple[Vec2, Vec2] | None = None
    tails_px: tuple[float, float] | None = None

    def provenance(self) -> dict[str, Any]:
        out: dict[str, Any] = {"axis_method": self.method}
        if self.reason is not None:
            out["axis_reason"] = self.reason
        if self.tails_px is not None:
            out["tails_px"] = list(self.tails_px)
        return out


NO_AXIS = MaskAxis(None, None, None, None, "none", "too_few_pixels")


def _region_moments(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre, eigenvalues (ascending) and eigenvectors of the pixel cloud's second moments as
    a region: each pixel is a unit square, so 1/12 is added on the diagonal (a one-pixel-wide
    line then has a finite minor moment and an L x w rectangle an elongation of exactly L/w)."""
    centre = points.mean(axis=0)
    centred = points - centre
    cov = centred.T @ centred / len(points) + np.eye(2) / 12.0
    evals, evecs = np.linalg.eigh(cov)
    return centre, evals, evecs


def _axis_frame(
    points: np.ndarray, centre: np.ndarray, direction: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(along, across) coordinates of `points` in the frame of a line through `centre`."""
    normal = np.array([-direction[1], direction[0]])
    rel = points - centre
    return rel @ direction, rel @ normal


def _width_along_axis(along: np.ndarray, across: np.ndarray, bins: int = AXIS_WIDTH_BINS) -> float:
    """Median over bins along the axis of the across-axis extent (max - min + 1 px)."""
    t0, t1 = float(along.min()), float(along.max())
    count = max(1, min(bins, int(np.ceil((t1 - t0) / AXIS_WIDTH_MIN_BIN_PX))))
    edges = np.linspace(t0, t1, count + 1)
    index = np.clip(np.searchsorted(edges, along, side="right") - 1, 0, count - 1)
    low = np.full(count, np.inf)
    high = np.full(count, -np.inf)
    filled = np.zeros(count, dtype=np.int64)
    np.minimum.at(low, index, across)
    np.maximum.at(high, index, across)
    np.add.at(filled, index, 1)
    keep = filled >= AXIS_WIDTH_MIN_BIN_PIXELS
    if not keep.any():
        keep = filled > 0
    return float(np.median(high[keep] - low[keep] + 1.0))


def _skeleton_points(mask: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray | None:
    """Zhang-Suen thinning of the mask's bbox crop (cv2.ximgproc), as pixel-centre points in
    full-image coordinates; None when the thinning is unavailable or finds nothing."""
    import cv2

    thinning = getattr(getattr(cv2, "ximgproc", None), "thinning", None)
    if thinning is None:
        return None
    # The thinning never removes a pixel on the image border, so a shape that touches the
    # crop's edge would keep its full outline: pad the crop by two background pixels.
    pad = 2
    x0, y0 = int(xs.min()), int(ys.min())
    crop = np.pad(mask[y0 : int(ys.max()) + 1, x0 : int(xs.max()) + 1], pad).astype(np.uint8)
    thin = thinning(crop * 255, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    sy, sx = np.nonzero(thin)
    if sx.size == 0:
        return None
    return np.column_stack([sx + x0 - pad, sy + y0 - pad]).astype(np.float64) + 0.5


def _ransac_line(
    points: np.ndarray,
    *,
    inlier_px: float,
    iterations: int = AXIS_RANSAC_ITERATIONS,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """The line through the most points within `inlier_px`, from pairs at least a quarter of
    the cloud's extent apart, refitted twice by total least squares on its inliers. Returns
    (centre, unit direction) or None with fewer than three usable points."""
    n = len(points)
    if n < 3:
        return None
    rng = np.random.default_rng(seed)
    a = points[rng.integers(0, n, iterations)]
    b = points[rng.integers(0, n, iterations)]
    span = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
    gap = np.linalg.norm(b - a, axis=1)
    usable = gap >= max(0.25 * span, 1.0)
    if not usable.any():
        return None
    a, b, gap = a[usable], b[usable], gap[usable]
    direction = (b - a) / gap[:, None]
    normal = np.column_stack([-direction[:, 1], direction[:, 0]])
    # (hypotheses, points): across-axis distance of every point to every candidate line.
    dist = np.abs(normal @ points.T - np.sum(normal * a, axis=1)[:, None])
    best = int(np.argmax((dist <= inlier_px).sum(axis=1)))
    centre, unit = a[best], direction[best]
    for _ in range(2):
        along, across = _axis_frame(points, centre, unit)
        inliers = points[np.abs(across) <= inlier_px]
        if len(inliers) < 3:
            break
        centre, _, evecs = _region_moments(inliers)
        unit = evecs[:, 1]
    return centre, unit


def _axis_endpoints(
    points: np.ndarray, centre: np.ndarray, direction: np.ndarray, band_px: float
) -> tuple[Vec2, Vec2]:
    """Extreme projections onto the axis of the pixels within `band_px` of it, pushed half a
    pixel outward to the pixel edge, ordered by image y then x."""
    along, across = _axis_frame(points, centre, direction)
    inside = np.abs(across) <= band_px
    if not inside.any():
        inside = np.ones(len(points), dtype=bool)
    t0, t1 = float(along[inside].min()) - 0.5, float(along[inside].max()) + 0.5
    ends = sorted(
        (
            (round(float(p[0]), 1), round(float(p[1]), 1))
            for p in (centre + t * direction for t in (t0, t1))
        ),
        key=lambda p: (p[1], p[0]),
    )
    return ends[0], ends[1]


def mask_axis_measurements(
    mask: np.ndarray,
    *,
    elongation_threshold: float = AXIS_ELONGATION_THRESHOLD,
    max_points: int = AXIS_MAX_POINTS,
) -> MaskAxis:
    """Principal axis, elongation, width and axis residual of a boolean mask (:class:`MaskAxis`).

    PCA on the pixel cloud (a deterministic stride sample of at most `max_points`) seeds the
    axis and gives the elongation; below `elongation_threshold` the mask is compact and gets
    no axis (elongation and width are still filled). An elongated mask is thinned and a
    RANSAC line on the skeleton replaces the PCA axis (the branch to a blob stuck on one side
    loses the vote); the endpoints are the extreme projections of the mask pixels within
    three quarters of the width of the axis, the width the median across-axis extent along
    it, the residual the RMS skeleton distance to the axis. Fewer than
    ``AXIS_MIN_PIXELS`` pixels give :data:`NO_AXIS`.
    """
    ys, xs = np.nonzero(mask)
    n = int(xs.size)
    if n < AXIS_MIN_PIXELS:
        return NO_AXIS
    points = np.column_stack([xs, ys]).astype(np.float64) + 0.5
    sample = points[np.linspace(0, n - 1, max_points).astype(int)] if n > max_points else points
    centre, evals, evecs = _region_moments(sample)
    elongation = float(np.sqrt(evals[1] / evals[0]))
    direction = evecs[:, 1]
    minor_width = float(np.sqrt(12.0 * evals[0]))
    if elongation < elongation_threshold:
        along, across = _axis_frame(sample, centre, direction)
        width = _width_along_axis(along, across)
        return MaskAxis(None, round(elongation, 3), round(width, 1), None, "pca", "compact")
    method, residual = "pca", None
    skeleton = _skeleton_points(mask, xs, ys)
    if skeleton is not None and len(skeleton) >= AXIS_MIN_SKELETON_POINTS:
        inlier_px = min(
            max(0.25 * minor_width, AXIS_RANSAC_MIN_INLIER_PX), AXIS_RANSAC_MAX_INLIER_PX
        )
        fit = _ransac_line(skeleton, inlier_px=inlier_px)
        if fit is not None:
            centre, direction = fit
            method = "ransac_skeleton"
            _, across = _axis_frame(skeleton, centre, direction)
            residual = round(float(np.sqrt(np.mean(across**2))), 2)
    along, across = _axis_frame(sample, centre, direction)
    width = _width_along_axis(along, across)
    ends = _axis_endpoints(sample, centre, direction, band_px=max(0.75 * width, 1.0))
    end_widths = _end_widths(centre, direction, ends, along, across)
    # The tail rule and the terminal centroids read every pixel: a thin tip is few pixels.
    tip_side: int | None = None
    tails: tuple[float, float] | None = None
    profile = width_profile_along_axis(points, ends)
    if profile is not None:
        widths, step = profile
        body_width = float(np.percentile(widths, 75))
        tails = tail_lengths_px(widths, step)
        tip_side = tip_side_from_tails(tails, TAIL_MIN_DIFFERENCE_WIDTHS * body_width)
    return MaskAxis(
        ends,
        round(elongation, 3),
        round(width, 1),
        residual,
        method,
        end_widths_px=end_widths,
        tip_side=tip_side,
        body_ends_px=terminal_centroids(points, ends),
        tails_px=None if tails is None else (round(tails[0], 1), round(tails[1], 1)),
    )


def width_profile_along_axis(
    points: np.ndarray,
    ends: Sequence[Sequence[float]],
    step: float = TAIL_STEP_PX,
    *,
    min_bin_pixels: int = TAIL_MIN_BIN_PIXELS,
) -> tuple[np.ndarray, float] | None:
    """The across-axis extent (max - min + 1 px) of the pixel-centre `points` in bins of
    `step` px along the axis from `ends[0]` to `ends[1]`, bins with fewer than
    `min_bin_pixels` dropped: ``(widths, step)``, or None for an axis shorter than a bin
    (`finebio_tipseg.width_profile`, the same binning)."""
    ends_arr = np.asarray(ends, dtype=np.float64).reshape(2, 2)
    segment = ends_arr[1] - ends_arr[0]
    length = float(np.linalg.norm(segment))
    if points.size == 0 or length < step:
        return None
    unit = segment / length
    normal = np.array([-unit[1], unit[0]])
    rel = np.asarray(points, dtype=np.float64) - ends_arr[0]
    along, across = rel @ unit, rel @ normal
    count = max(1, int(math.ceil(length / step)))
    index = np.clip(np.floor(along / step).astype(int), 0, count - 1)
    low = np.full(count, np.inf)
    high = np.full(count, -np.inf)
    filled = np.zeros(count, dtype=np.int64)
    np.minimum.at(low, index, across)
    np.maximum.at(high, index, across)
    np.add.at(filled, index, 1)
    keep = filled >= min_bin_pixels
    if not keep.any():
        return None
    return high[keep] - low[keep] + 1.0, step


def tail_length_px(widths: np.ndarray, step: float, side: int, *, fraction: float) -> float:
    """The thin run from the profile's end `side` inward: bins under `fraction` of the body
    width (the 75th percentile of the profile) until the first two consecutive bins at or
    over it, in px (`finebio_tipseg.junction_from_profile`, the tail alone)."""
    body = float(np.percentile(widths, 75)) if widths.size else 0.0
    ordered = widths[::-1] if side == 1 else widths
    limit = fraction * body
    tail = 0
    for i, width in enumerate(ordered):
        if width >= limit and (i + 1 >= len(ordered) or ordered[i + 1] >= limit):
            break
        tail = i + 1
    return tail * step


def tail_lengths_px(
    widths: np.ndarray, step: float, *, fraction: float = TAIL_FRACTION
) -> tuple[float, float]:
    """(tail from end 0, tail from end 1) of a width profile, px."""
    return (
        tail_length_px(widths, step, 0, fraction=fraction),
        tail_length_px(widths, step, 1, fraction=fraction),
    )


def tip_side_from_tails(
    tails: Sequence[float], min_difference_px: float, *, ratio: float = TAIL_SIDE_RATIO
) -> int | None:
    """The long-thin-tail rule (Sep 29, `finebio_tipseg`): the end whose thin tail is longer
    by more than `min_difference_px` and at least `ratio` times the other's is the tip (the
    shaft and cone make a long thin run, the plunger stem a short one); None when the tails
    do not differ enough, so a caller may fall back to a hand box or the lower end."""
    t0, t1 = float(tails[0]), float(tails[1])
    longer, shorter = max(t0, t1), min(t0, t1)
    if longer > 0 and longer - shorter > min_difference_px and longer >= ratio * shorter:
        return 0 if t0 > t1 else 1
    return None


def terminal_centroids(
    points: np.ndarray, ends: Sequence[Sequence[float]], depth_px: float = TERMINAL_DEPTH_PX
) -> tuple[Vec2, Vec2] | None:
    """The centroid of the pixel-centre `points` within `depth_px` of the far extreme along
    the axis at each end, in the order of `ends`: the mask's own end on a thin tip, where the
    fitted axis endpoint (the extreme within a band of the axis line) can sit off it sideways
    (`finebio_tipseg.terminal_centroid`, both ends). None for a degenerate axis."""
    ends_arr = np.asarray(ends, dtype=np.float64).reshape(2, 2)
    segment = ends_arr[1] - ends_arr[0]
    norm = float(np.linalg.norm(segment))
    if points.size == 0 or norm < 1e-9:
        return None
    unit = segment / norm
    along = (np.asarray(points, dtype=np.float64) - ends_arr[0]) @ unit
    out = []
    for keep in (along <= along.min() + depth_px, along >= along.max() - depth_px):
        centroid = points[keep].mean(axis=0)
        out.append((round(float(centroid[0]), 1), round(float(centroid[1]), 1)))
    return out[0], out[1]


def _end_widths(
    centre: np.ndarray,
    direction: np.ndarray,
    ends: tuple[Vec2, Vec2],
    along: np.ndarray,
    across: np.ndarray,
    band: float = AXIS_END_BAND,
) -> tuple[float, float] | None:
    """The median across-axis width over the first and last `band` of the axis length, in
    the order of `ends` (Sep 29: the wide end of an 8-channel pipette is its manifold, the
    wide end of a single-channel one its grip; either names the tip / butt)."""
    t_ends = [float((np.asarray(e, dtype=np.float64) - centre) @ direction) for e in ends]
    span = abs(t_ends[1] - t_ends[0])
    if span < 1e-6:
        return None
    widths = []
    for t_end, t_other in (t_ends, t_ends[::-1]):
        sign = 1.0 if t_other > t_end else -1.0
        inside = (along - t_end) * sign <= band * span
        inside &= (along - t_end) * sign >= -0.5
        if int(inside.sum()) < AXIS_END_MIN_PIXELS:
            return None
        widths.append(round(_width_along_axis(along[inside], across[inside]), 1))
    return widths[0], widths[1]


def filter_mask_components(
    mask: np.ndarray, keep_fraction: float = MASK_COMPONENT_KEEP_FRACTION
) -> tuple[np.ndarray, int, int]:
    """Keep the 8-connected components with area >= `keep_fraction` x the largest one, the
    worker's own box rule; returns (filtered mask, pixels dropped, components found)."""
    import cv2

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    components = max(int(count) - 1, 0)
    if components <= 1:
        return mask, 0, components
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.flatnonzero(areas >= keep_fraction * areas.max()) + 1
    kept = np.isin(labels, keep)
    return kept, int(np.count_nonzero(mask) - np.count_nonzero(kept)), components


def read_mask(path: Path) -> np.ndarray | None:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None if image is None else image > 0


def _slot_index(object_id: str) -> int | None:
    tail = object_id.rsplit("-", 1)[-1]
    return int(tail) if tail.isdigit() else None


def _resolve_slot_label(
    obj: PerFrameObject, slot_labels: Sequence[str] | Mapping[int, str] | None
) -> str:
    index = _slot_index(obj.object_id)
    if slot_labels is not None and index is not None:
        if isinstance(slot_labels, Mapping):
            if index in slot_labels:
                return str(slot_labels[index])
        elif index < len(slot_labels):
            return str(slot_labels[index])
    if "#" in obj.label:
        return obj.label
    return slot_label(obj.label, index if index is not None else 0)


def _normalised_to_px(box: Any, width: int, height: int) -> BoxXYXY:
    return _box(
        (box.x * width, box.y * height, (box.x + box.width) * width, (box.y + box.height) * height)
    )


def worker_to_observations(
    run_dir: Path,
    view: str,
    start_frame: int,
    slot_labels: Sequence[str] | Mapping[int, str] | None = None,
    *,
    pose_valid: PoseValidity | None = None,
    detector_rows: Iterable[FineBioObservation] | None = None,
    match_iou: float = DEFAULT_MATCH_IOU,
    image_size: tuple[int, int] | None = None,
) -> list[FineBioObservation]:
    """A SAM3 worker run (``observations.jsonl`` + ``masks/``) -> SAM3 rows in raw frames.

    ``slot_labels`` maps the worker's multiplex slot index (``object_id`` ``sam3-NN``) to the
    ``<class>#<k>`` slot label; without it a label that already carries ``#`` is used as is
    and a plain class label becomes ``<class>#<slot index>``.  ``detector_rows`` (same view,
    raw frames) attach the same-class detector box with the best IoU >= ``match_iou`` to
    video rows; box-decode rows carry the box that prompted them.  ``image_size`` (w, h) is
    read from the first mask when not given.
    """
    run_dir = Path(run_dir)
    frames: tuple[FrameObservations, ...] = load_observations(run_dir / "observations.jsonl")
    det_index: dict[tuple[int, str], list[FineBioObservation]] = {}
    for det in detector_rows or ():
        if det.view == view and det.source == "detector" and det.box_xyxy_px is not None:
            det_index.setdefault((det.frame_index, det.object_class), []).append(det)
    size = image_size
    rows: list[FineBioObservation] = []
    for frame in frames:
        raw_frame = start_frame + frame.analysis_frame_index
        valid = bool(pose_valid(raw_frame)) if view == FINEBIO_FPV_VIEW and pose_valid else True
        for obj in frame.objects:
            mask = read_mask(run_dir / obj.mask.uri) if obj.mask is not None else None
            if mask is not None and size is None:
                size = (int(mask.shape[1]), int(mask.shape[0]))
            if size is None:
                raise ValueError(
                    f"{run_dir}: no mask to read the image size from; pass image_size=(w, h)"
                )
            width, height = size
            slot = _resolve_slot_label(obj, slot_labels)
            cls = class_of_slot(slot)
            source: ObservationSource = (
                "sam3_decode" if obj.source == "sam3_decode" else "sam3_video"
            )
            provenance: dict[str, Any] = {"worker_label": obj.label, "object_id": obj.object_id}
            if mask is not None:
                mask, dropped, components = filter_mask_components(mask)
                if dropped:
                    provenance["mask_speckle_pixels_dropped"] = dropped
                    provenance["mask_components"] = components
            measured = mask_measurements(mask) if mask is not None else None
            axis = NO_AXIS
            if measured is None:
                mask_bbox = _normalised_to_px(obj.box, width, height)
                centroid = None
                area = None
                provenance["mask"] = "absent" if mask is None else "empty"
            else:
                mask_bbox, centroid, area = measured
                assert mask is not None
                axis = mask_axis_measurements(mask)
            if obj.iou_prediction is not None:
                provenance["decoder_iou_pred"] = round(float(obj.iou_prediction), 4)
            if source == "sam3_decode":
                provenance["decoder_iou_pred"] = round(float(obj.confidence), 4)
            box: BoxXYXY | None = None
            det_score: float | None = None
            if obj.prompt_box is not None:
                box = _normalised_to_px(obj.prompt_box, width, height)
                det_score = _score(obj.prompt_score)
                provenance["prompt_source"] = obj.prompt_source
            elif det_index:
                best, best_iou = None, 0.0
                for det in det_index.get((raw_frame, cls), ()):
                    assert det.box_xyxy_px is not None
                    iou = box_iou(list(mask_bbox), list(det.box_xyxy_px))
                    if iou > best_iou:
                        best, best_iou = det, iou
                if best is not None and best_iou >= match_iou:
                    box, det_score = best.box_xyxy_px, best.detector_score
                    provenance["detector_box_iou"] = round(best_iou, 4)
                    provenance["detector_slot"] = best.slot
            if box is not None:
                provenance.setdefault(
                    "detector_box_iou", round(box_iou(list(mask_bbox), list(box)), 4)
                )
            if measured is not None:
                # Last, so `remeasure` (which appends to a Sep 25 row) writes the same bytes.
                provenance.update(axis.provenance())
            rows.append(
                FineBioObservation(
                    view=view,
                    frame_index=raw_frame,
                    slot=slot,
                    object_class=cls,
                    detector_score=det_score,
                    box_xyxy_px=box,
                    mask_bbox_px=mask_bbox,
                    mask_centroid_px=centroid,
                    mask_area_px=area,
                    mask_axis_px=axis.axis_px,
                    mask_elongation=axis.elongation,
                    mask_width_px=axis.width_px,
                    mask_axis_residual_px=axis.residual_px,
                    mask_end_widths_px=axis.end_widths_px,
                    tip_side=axis.tip_side,
                    body_end_px=axis.body_ends_px,
                    sam3_object_score=(
                        None if obj.object_score is None else round(float(obj.object_score), 4)
                    ),
                    pose_valid=valid,
                    source=source,
                    provenance=provenance,
                )
            )
    return rows


def write_observations(rows: Iterable[FineBioObservation], path: Path) -> int:
    """Rows sorted by frame, view, source and slot, written compactly; returns the count."""
    ordered = sorted(
        rows,
        key=lambda r: (
            r.frame_index,
            VIEWS.index(r.view) if r.view in VIEWS else len(VIEWS),
            r.source,
            r.slot,
        ),
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return write_jsonl(ordered, Path(path), compact=True)
