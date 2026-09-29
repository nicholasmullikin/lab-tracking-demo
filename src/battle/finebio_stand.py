"""`battle-finebio-stand`: the pipettes at rest as 3D segments (plan `p0-stand-slice`, Sep 28).

The first slice of the pipettes-as-lines plan, on the frames where the pipettes sit still,
using only the per-view 2D mask axes (`FineBioObservation.mask_axis_px`, Sep 28) and the rig's
fixed cameras; no 3D line module is involved.

* **Stand frames.** A pipette is *static* on a frame when the top same-class detector box
  centre of a fixed view moves at most `max_step_px` per frame over a run of at least
  `min_run_frames` frames, in at least `min_static_views` fixed views. The *stand window* of
  a trial is the longest run of frames on which the most single-channel pipettes are static
  at once (with a tie broken by length); every pipette's own static stretches are kept too,
  since the pipette in use also rests somewhere for a while.
* **Geometry.** On every static frame of a pipette, the two axis endpoints of every fixed view
  with an axis are matched across views and back-projected with
  `finebio_slice.triangulate_pixels` (undistort, then DLT), giving two 3D ends, hence the
  length, the direction, the angle to the bench normal (the board plane is the bench, ``z``
  into it), the per-view reprojection residual of each end and, with three or more views,
  the leave-one-view-out residual. Ends are first labelled by the projected world-up
  direction in the most reliable view (the plan's "vertical order"); every further view's
  pairing is the one with the smaller two-view triangulation residual, and a pairing that
  disagrees with the up-labelling is counted (`up_order_disagreements`): the pipettes on this
  bench lie flat, so image "up" alone cannot order their ends.
* **Report.** `stand_report.json` and `.md`: per pipette the median length (p10, p90), the
  angle to the bench normal, the height of both ends, the drift of each end over the window,
  the per-view residuals; pairwise angle and spacing (distance between the segments' midlines)
  over the stand window; a shared length (median of the single-channel medians, with their
  spread); and the tracks-ext cross-check (the point tracker's observed rows of each class
  over the window, their median position and drift). The `config` subcommand writes
  `configs/finebio/pipettes.json` (the length prior with provenance) from one or two trials'
  reports: when the trials disagree by more than the spread, trial 1's value is taken and the
  disagreement is written down.

Claim boundary: the ends are SAM3 mask extremes, not physical tips; a mask that stops short
(a tip in shadow, a hand over the head) shortens the segment, and a merged mask lengthens it.
Nothing here is ground truth; FineBio is non-commercial research data and nothing under
`runs/` or `data/` is committed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .finebio_cameras import Camera, cameras_from_config, read_camera_config
from .finebio_slice import triangulate_pixels
from .multiview_schemas import (
    FINEBIO_FIXED_VIEWS,
    FineBioObservation,
    Track3D,
    read_jsonl,
)

SCHEMA = "battle-finebio-stand/1"
CONFIG_KIND = "finebio_pipettes"
SINGLE_CHANNEL_PIPETTES: tuple[str, ...] = ("blue_pipette", "yellow_pipette", "red_pipette")
PIPETTE_CLASSES: tuple[str, ...] = (*SINGLE_CHANNEL_PIPETTES, "8_channel_pipette")
SAM3_SOURCES = ("sam3_decode", "sam3_video")
# World up in board centimetres (z into the bench); the bench plane is z = 0.
BENCH_UP = np.array([0.0, 0.0, -1.0])
# The plan's merged-mask gates, carried into the config for the tracker extension.
ELONGATION_THRESHOLD = 2.5
MERGED_WIDTH_FACTOR = 1.6
MERGED_EXTENT_FACTOR = 1.2
CLAIM_BOUNDARY = (
    "Segment ends are the extremes of SAM3 masks along their fitted axis, not physical tips: a "
    "mask that stops short shortens the segment and a merged mask lengthens it. The static "
    "frames are the detector's, the cameras the rig's; no number here is ground truth."
)
LICENCE_NOTE = (
    "FineBio is licensed for non-commercial research; frames, masks and videos derived from it "
    "stay under runs/ or data/ and are never committed or redistributed"
)


@dataclass(frozen=True)
class StandSettings:
    min_run_frames: int = 30
    max_step_px: float = 3.0
    min_static_views: int = 2
    min_axis_views: int = 2
    # A static frame is *at rest* when every view's endpoint residual is within this (the
    # rig's association gate); with three or more views the worst view is dropped first. A
    # pipette held still in a hand is static by its boxes and fails this by tens of pixels.
    rest_residual_px: float = 30.0
    # The resting place is on the bench (a rack or the bench itself): the lower end within
    # this height of the bench plane, above or below (nothing sits under the bench, so a lower
    # end further below is a triangulation error). A pipette held still in the air is static,
    # and not at rest.
    max_rest_height_cm: float = 10.0
    fixed_views: tuple[str, ...] = FINEBIO_FIXED_VIEWS
    classes: tuple[str, ...] = PIPETTE_CLASSES
    shared_classes: tuple[str, ...] = SINGLE_CHANNEL_PIPETTES

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_run_frames": self.min_run_frames,
            "max_step_px": self.max_step_px,
            "min_static_views": self.min_static_views,
            "min_axis_views": self.min_axis_views,
            "rest_residual_px": self.rest_residual_px,
            "max_rest_height_cm": self.max_rest_height_cm,
            "fixed_views": list(self.fixed_views),
            "classes": list(self.classes),
            "shared_classes": list(self.shared_classes),
        }


# --------------------------------------------------------------------------- static frames


def top_box_centres(
    rows: Iterable[FineBioObservation], settings: StandSettings
) -> dict[tuple[str, str], dict[int, tuple[float, float]]]:
    """(class, fixed view) -> frame -> centre of the top-scoring detector box."""
    best: dict[tuple[str, str], dict[int, tuple[float, tuple[float, float]]]] = defaultdict(dict)
    for r in rows:
        if r.source != "detector" or r.view not in settings.fixed_views:
            continue
        if r.object_class not in settings.classes or r.box_xyxy_px is None:
            continue
        score = r.detector_score or 0.0
        key = (r.object_class, r.view)
        current = best[key].get(r.frame_index)
        if current is None or score > current[0]:
            box = r.box_xyxy_px
            best[key][r.frame_index] = (score, ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
    return {key: {f: c for f, (_, c) in by.items()} for key, by in best.items()}


def stationary_runs(
    centres: Mapping[int, tuple[float, float]], *, max_step_px: float, min_run_frames: int
) -> list[tuple[int, int]]:
    """Maximal runs of consecutive frames whose centre steps at most `max_step_px`, of at
    least `min_run_frames` frames, as inclusive (start, end) pairs."""
    frames = sorted(centres)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    prev: int | None = None
    for f in frames:
        continues = (
            prev is not None
            and f == prev + 1
            and float(np.hypot(*(np.subtract(centres[f], centres[prev])))) <= max_step_px
        )
        if not continues:
            if start is not None and prev is not None and prev - start + 1 >= min_run_frames:
                runs.append((start, prev))
            start = f
        prev = f
    if start is not None and prev is not None and prev - start + 1 >= min_run_frames:
        runs.append((start, prev))
    return runs


def static_frames(
    rows: Iterable[FineBioObservation], settings: StandSettings
) -> dict[str, dict[int, tuple[str, ...]]]:
    """class -> frame -> the fixed views in which the class is static on that frame (only
    frames with at least `min_static_views` such views are kept)."""
    centres = top_box_centres(rows, settings)
    views_by_frame: dict[str, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    for (cls, view), by_frame in centres.items():
        for start, end in stationary_runs(
            by_frame, max_step_px=settings.max_step_px, min_run_frames=settings.min_run_frames
        ):
            for f in range(start, end + 1):
                views_by_frame[cls][f].append(view)
    out: dict[str, dict[int, tuple[str, ...]]] = {}
    for cls, frames in views_by_frame.items():
        kept = {
            f: tuple(sorted(views))
            for f, views in sorted(frames.items())
            if len(views) >= settings.min_static_views
        }
        if kept:
            out[cls] = kept
    return out


def _runs(frames: Iterable[int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for f in sorted(set(frames)):
        if out and f == out[-1][1] + 1:
            out[-1] = (out[-1][0], f)
        else:
            out.append((f, f))
    return out


@dataclass(frozen=True)
class StandWindow:
    start: int
    end: int  # inclusive
    classes: tuple[str, ...]
    static_count: int

    @property
    def frames(self) -> range:
        return range(self.start, self.end + 1)

    def as_dict(self) -> dict[str, Any]:
        return {
            "start_frame": self.start,
            "end_frame": self.end,
            "frames": self.end - self.start + 1,
            "classes": list(self.classes),
            "static_pipettes": self.static_count,
        }


def stand_window(
    resting: Mapping[str, Iterable[int]], settings: StandSettings
) -> StandWindow | None:
    """The longest run of frames on which the most shared-class pipettes are at rest at once
    (at least `min_run_frames` long); None when no pipette ever rests. `resting` maps a class
    to the frames on which it is static and its segment passes the rest gate."""
    count: Counter[int] = Counter()
    for cls in settings.shared_classes:
        for f in resting.get(cls, ()):
            count[f] += 1
    if not count:
        return None
    for target in range(max(count.values()), 0, -1):
        frames = [f for f, n in count.items() if n >= target]
        runs = [r for r in _runs(frames) if r[1] - r[0] + 1 >= settings.min_run_frames]
        if not runs:
            continue
        start, end = max(runs, key=lambda r: (r[1] - r[0], -r[0]))
        classes = tuple(
            cls
            for cls in settings.shared_classes
            if len(set(resting.get(cls, ())) & set(range(start, end + 1)))
            >= 0.9 * (end - start + 1)
        )
        return StandWindow(start, end, classes, target)
    return None


# --------------------------------------------------------------------------- geometry


def _pixel_residual(cam: Camera, point: np.ndarray, pixel: np.ndarray) -> float:
    return float(np.linalg.norm(cam.project(point)[0] - pixel))


def _pair_residual(
    cams: Mapping[str, Camera], views: Sequence[str], pixels: Sequence[np.ndarray]
) -> float:
    point = triangulate_pixels([cams[v] for v in views], pixels)
    if not np.all(np.isfinite(point)):
        return float("inf")
    return sum(_pixel_residual(cams[v], point, p) for v, p in zip(views, pixels))


def match_endpoints(
    cams: Mapping[str, Camera],
    axes: Mapping[str, tuple[np.ndarray, np.ndarray]],
    anchor: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Label each view's two axis endpoints as end A / end B consistently across views.

    End A is the higher end by the projected world-up direction at `anchor` in the view where
    the axis is most aligned with that direction; every other view takes the pairing with the
    smaller two-view triangulation residual against it. Returns (A per view, B per view, info:
    the anchor view, each view's |cos| to up, the views whose pairing disagreed with up).
    """
    up_score: dict[str, float] = {}
    by_up: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for view, (p0, p1) in axes.items():
        cam = cams[view]
        up = cam.project(anchor + BENCH_UP)[0] - cam.project(anchor)[0]
        axis = p1 - p0
        norm = float(np.linalg.norm(up) * np.linalg.norm(axis))
        cos = float(np.dot(up, axis) / norm) if norm > 0 else 0.0
        up_score[view] = abs(cos)
        by_up[view] = (p1, p0) if cos > 0 else (p0, p1)
    order = sorted(axes, key=lambda v: -up_score[v])
    first = order[0]
    ends_a = {first: by_up[first][0]}
    ends_b = {first: by_up[first][1]}
    disagreed: list[str] = []
    for view in order[1:]:
        a, b = by_up[view]
        keep = _pair_residual(cams, [first, view], [ends_a[first], a]) + _pair_residual(
            cams, [first, view], [ends_b[first], b]
        )
        swap = _pair_residual(cams, [first, view], [ends_a[first], b]) + _pair_residual(
            cams, [first, view], [ends_b[first], a]
        )
        if swap < keep:
            a, b = b, a
            disagreed.append(view)
        ends_a[view] = a
        ends_b[view] = b
    info = {
        "anchor_view": first,
        "up_alignment": {v: round(up_score[v], 3) for v in order},
        "up_order_disagreements": disagreed,
    }
    return ends_a, ends_b, info


@dataclass
class Segment:
    frame_index: int
    object_class: str
    end_a_cm: np.ndarray
    end_b_cm: np.ndarray
    views: tuple[str, ...]
    residual_a_px: dict[str, float]
    residual_b_px: dict[str, float]
    loo_a_px: dict[str, float] = field(default_factory=dict)
    loo_b_px: dict[str, float] = field(default_factory=dict)
    up_order_disagreements: tuple[str, ...] = ()
    anchor_view: str = ""
    views_dropped: tuple[str, ...] = ()

    @property
    def max_residual_px(self) -> float:
        return max([*self.residual_a_px.values(), *self.residual_b_px.values()], default=0.0)

    @property
    def lower_end_height_cm(self) -> float:
        """Height above the bench (board plane) of the lower end, cm; negative is below."""
        return min(float(-self.end_a_cm[2]), float(-self.end_b_cm[2]))

    def within_rest_height(self, settings: StandSettings) -> bool:
        return abs(self.lower_end_height_cm) <= settings.max_rest_height_cm

    def at_rest(self, settings: StandSettings) -> bool:
        return self.max_residual_px <= settings.rest_residual_px and self.within_rest_height(
            settings
        )

    @property
    def length_cm(self) -> float:
        return float(np.linalg.norm(self.end_a_cm - self.end_b_cm))

    @property
    def direction(self) -> np.ndarray:
        d = self.end_a_cm - self.end_b_cm
        return d / (np.linalg.norm(d) or 1.0)

    @property
    def angle_to_bench_normal_deg(self) -> float:
        return float(np.degrees(np.arccos(min(1.0, abs(float(self.direction @ BENCH_UP))))))

    @property
    def midpoint_cm(self) -> np.ndarray:
        return (self.end_a_cm + self.end_b_cm) / 2

    def to_record(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "object_class": self.object_class,
            "end_a_cm": [round(float(x), 3) for x in self.end_a_cm],
            "end_b_cm": [round(float(x), 3) for x in self.end_b_cm],
            "length_cm": round(self.length_cm, 3),
            "angle_to_bench_normal_deg": round(self.angle_to_bench_normal_deg, 2),
            "views": list(self.views),
            "residual_a_px": {v: round(r, 2) for v, r in self.residual_a_px.items()},
            "residual_b_px": {v: round(r, 2) for v, r in self.residual_b_px.items()},
            "loo_a_px": {v: round(r, 2) for v, r in self.loo_a_px.items()},
            "loo_b_px": {v: round(r, 2) for v, r in self.loo_b_px.items()},
            "up_order_disagreements": list(self.up_order_disagreements),
            "anchor_view": self.anchor_view,
            "views_dropped": list(self.views_dropped),
            "max_residual_px": round(self.max_residual_px, 2),
        }


def _loo(
    cams: Mapping[str, Camera], views: Sequence[str], pixels: Mapping[str, np.ndarray]
) -> dict[str, float]:
    out: dict[str, float] = {}
    if len(views) < 3:
        return out
    for v in views:
        others = [u for u in views if u != v]
        point = triangulate_pixels([cams[u] for u in others], [pixels[u] for u in others])
        if np.all(np.isfinite(point)):
            out[v] = _pixel_residual(cams[v], point, pixels[v])
    return out


def _triangulate_segment(
    frame: int,
    object_class: str,
    cams: Mapping[str, Camera],
    axes: Mapping[str, tuple[np.ndarray, np.ndarray]],
    anchor: np.ndarray,
    views: Sequence[str],
    dropped: Sequence[str],
) -> Segment | None:
    ends_a, ends_b, info = match_endpoints(cams, {v: axes[v] for v in views}, anchor)
    a = triangulate_pixels([cams[v] for v in views], [ends_a[v] for v in views])
    b = triangulate_pixels([cams[v] for v in views], [ends_b[v] for v in views])
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return None
    return Segment(
        frame_index=frame,
        object_class=object_class,
        end_a_cm=a,
        end_b_cm=b,
        views=tuple(views),
        residual_a_px={v: _pixel_residual(cams[v], a, ends_a[v]) for v in views},
        residual_b_px={v: _pixel_residual(cams[v], b, ends_b[v]) for v in views},
        loo_a_px=_loo(cams, views, ends_a),
        loo_b_px=_loo(cams, views, ends_b),
        up_order_disagreements=tuple(info["up_order_disagreements"]),
        anchor_view=str(info["anchor_view"]),
        views_dropped=tuple(dropped),
    )


def segment_from_views(
    frame: int,
    object_class: str,
    cams: Mapping[str, Camera],
    per_view: Mapping[str, FineBioObservation],
    *,
    min_views: int = 2,
    rest_residual_px: float | None = None,
) -> Segment | None:
    """One frame's 3D segment of a pipette from the fixed views that carry an axis. With
    `rest_residual_px`, while an endpoint residual exceeds it and more than `min_views` views
    remain, the view with the largest leave-one-out residual is dropped and the segment
    refitted (`views_dropped`); the returned segment may still exceed the gate, which the
    caller reads from `max_residual_px`."""
    axes = {
        v: (
            np.array(r.mask_axis_px[0], dtype=np.float64),
            np.array(r.mask_axis_px[1], dtype=np.float64),
        )
        for v, r in per_view.items()
        if r.mask_axis_px is not None and v in cams
    }
    if len(axes) < min_views:
        return None
    views = sorted(axes)
    anchor = triangulate_pixels(
        [cams[v] for v in views], [np.array(per_view[v].point_px, dtype=np.float64) for v in views]
    )
    if not np.all(np.isfinite(anchor)):
        return None
    dropped: list[str] = []
    while True:
        segment = _triangulate_segment(frame, object_class, cams, axes, anchor, views, dropped)
        if segment is None or rest_residual_px is None:
            return segment
        if segment.max_residual_px <= rest_residual_px or len(views) <= min_views:
            return segment
        if not segment.loo_a_px:
            return segment
        worst = max(
            views, key=lambda v: segment.loo_a_px.get(v, 0.0) + segment.loo_b_px.get(v, 0.0)
        )
        dropped.append(worst)
        views = [v for v in views if v != worst]


def sam3_axis_rows(
    rows: Iterable[FineBioObservation], settings: StandSettings
) -> dict[tuple[str, int], dict[str, FineBioObservation]]:
    """(class, frame) -> fixed view -> the SAM3 row with an axis (the highest SAM3 score when
    a view carries several rows of the class)."""
    out: dict[tuple[str, int], dict[str, FineBioObservation]] = defaultdict(dict)
    for r in rows:
        if r.source not in SAM3_SOURCES or r.view not in settings.fixed_views:
            continue
        if r.object_class not in settings.classes or r.mask_axis_px is None:
            continue
        current = out[(r.object_class, r.frame_index)].get(r.view)
        if current is None or (r.sam3_object_score or 0.0) > (current.sam3_object_score or 0.0):
            out[(r.object_class, r.frame_index)][r.view] = r
    return out


def segments_on_static_frames(
    rows: Sequence[FineBioObservation],
    cams: Mapping[str, Camera],
    static: Mapping[str, Mapping[int, Sequence[str]]],
    settings: StandSettings,
) -> dict[str, list[Segment]]:
    """class -> its 3D segments on every frame where it is static and carries axes in at
    least `min_axis_views` fixed views."""
    axis_rows = sam3_axis_rows(rows, settings)
    out: dict[str, list[Segment]] = defaultdict(list)
    for cls in settings.classes:
        for f in sorted(static.get(cls, {})):
            per_view = axis_rows.get((cls, f))
            if not per_view:
                continue
            seg = segment_from_views(
                f,
                cls,
                cams,
                per_view,
                min_views=settings.min_axis_views,
                rest_residual_px=settings.rest_residual_px,
            )
            if seg is not None:
                out[cls].append(seg)
    return dict(out)


def resting_segments(
    segments: Mapping[str, Sequence[Segment]], settings: StandSettings
) -> dict[str, list[Segment]]:
    """The segments within the rest gates (endpoint residual, end height), per class."""
    return {cls: [s for s in segs if s.at_rest(settings)] for cls, segs in segments.items()}


# --------------------------------------------------------------------------- statistics


def _pct(values: Sequence[float], digits: int = 2) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "median": None, "p10": None, "p90": None}
    return {
        "n": int(arr.size),
        "median": round(float(np.median(arr)), digits),
        "p10": round(float(np.percentile(arr, 10)), digits),
        "p90": round(float(np.percentile(arr, 90)), digits),
    }


def _drift(points: np.ndarray) -> dict[str, Any]:
    """Distance of each frame's point from the window median, cm."""
    if len(points) == 0:
        return {"n": 0}
    median = np.median(points, axis=0)
    d = np.linalg.norm(points - median, axis=1)
    return {
        "n": int(len(points)),
        "median_point_cm": [round(float(x), 2) for x in median],
        "p50_cm": round(float(np.median(d)), 2),
        "p90_cm": round(float(np.percentile(d, 90)), 2),
        "max_cm": round(float(d.max()), 2),
    }


def _angle_between(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.degrees(np.arccos(min(1.0, abs(float(u @ v))))))


def _line_spacing(
    seg_i: tuple[np.ndarray, np.ndarray], seg_j: tuple[np.ndarray, np.ndarray]
) -> float:
    """Distance between two nearly parallel segments' midlines: the mean over both of the
    distance from one's midpoint to the other's infinite line."""

    def point_to_line(p: np.ndarray, a: np.ndarray, d: np.ndarray) -> float:
        return float(np.linalg.norm(np.cross(p - a, d)))

    (a_i, d_i), (a_j, d_j) = seg_i, seg_j
    return 0.5 * (point_to_line(a_j, a_i, d_i) + point_to_line(a_i, a_j, d_j))


def pipette_summary(segments: Sequence[Segment], window: StandWindow | None) -> dict[str, Any]:
    lengths = [s.length_cm for s in segments]
    angles = [s.angle_to_bench_normal_deg for s in segments]
    a = np.array([s.end_a_cm for s in segments]) if segments else np.zeros((0, 3))
    b = np.array([s.end_b_cm for s in segments]) if segments else np.zeros((0, 3))
    in_window = (
        [s for s in segments if window.start <= s.frame_index <= window.end] if window else []
    )
    a_w = np.array([s.end_a_cm for s in in_window]) if in_window else np.zeros((0, 3))
    b_w = np.array([s.end_b_cm for s in in_window]) if in_window else np.zeros((0, 3))
    per_view_a: dict[str, list[float]] = defaultdict(list)
    per_view_b: dict[str, list[float]] = defaultdict(list)
    loo_a: dict[str, list[float]] = defaultdict(list)
    loo_b: dict[str, list[float]] = defaultdict(list)
    views_used: Counter[str] = Counter()
    disagreements: Counter[str] = Counter()
    for s in segments:
        for v, r in s.residual_a_px.items():
            per_view_a[v].append(r)
        for v, r in s.residual_b_px.items():
            per_view_b[v].append(r)
        for v, r in s.loo_a_px.items():
            loo_a[v].append(r)
        for v, r in s.loo_b_px.items():
            loo_b[v].append(r)
        views_used[",".join(s.views)] += 1
        for v in s.up_order_disagreements:
            disagreements[v] += 1
    direction = None
    if segments:
        # Directions are sign-ambiguous frame to frame: align each with the first before the
        # median.
        reference = segments[0].direction
        aligned = [s.direction * (1.0 if s.direction @ reference >= 0 else -1.0) for s in segments]
        direction = np.median(np.array(aligned), axis=0)
        direction = direction / (np.linalg.norm(direction) or 1.0)
    return {
        "frames_with_segment": len(segments),
        "frames_in_stand_window": len(in_window),
        "frame_runs": _runs(s.frame_index for s in segments)[:20],
        "length_cm": _pct(lengths),
        "angle_to_bench_normal_deg": _pct(angles, 1),
        "end_a_height_cm": _pct([-float(p[2]) for p in a]),
        "end_b_height_cm": _pct([-float(p[2]) for p in b]),
        "end_a_drift_window": _drift(a_w),
        "end_b_drift_window": _drift(b_w),
        "end_a_drift_all_static": _drift(a),
        "end_b_drift_all_static": _drift(b),
        "median_direction": (
            None if direction is None else [round(float(x), 3) for x in direction]
        ),
        "residual_a_px_per_view": {v: _pct(r, 1) for v, r in sorted(per_view_a.items())},
        "residual_b_px_per_view": {v: _pct(r, 1) for v, r in sorted(per_view_b.items())},
        "loo_a_px_per_view": {v: _pct(r, 1) for v, r in sorted(loo_a.items())},
        "loo_b_px_per_view": {v: _pct(r, 1) for v, r in sorted(loo_b.items())},
        "view_sets": dict(views_used.most_common()),
        "up_order_disagreements_by_view": dict(disagreements),
    }


def pairwise_summary(
    segments: Mapping[str, Sequence[Segment]], window: StandWindow | None, classes: Sequence[str]
) -> list[dict[str, Any]]:
    """Angle and spacing between every pair of pipettes over the frames both have a segment
    in the stand window."""
    out: list[dict[str, Any]] = []
    if window is None:
        return out
    by_frame: dict[str, dict[int, Segment]] = {
        cls: {s.frame_index: s for s in segs if window.start <= s.frame_index <= window.end}
        for cls, segs in segments.items()
    }
    present = [c for c in classes if by_frame.get(c)]
    for i, ci in enumerate(present):
        for cj in present[i + 1 :]:
            common = sorted(set(by_frame[ci]) & set(by_frame[cj]))
            if not common:
                continue
            angles, spacings, mids = [], [], []
            for f in common:
                si, sj = by_frame[ci][f], by_frame[cj][f]
                angles.append(_angle_between(si.direction, sj.direction))
                spacings.append(
                    _line_spacing((si.midpoint_cm, si.direction), (sj.midpoint_cm, sj.direction))
                )
                mids.append(float(np.linalg.norm(si.midpoint_cm - sj.midpoint_cm)))
            out.append(
                {
                    "pair": [ci, cj],
                    "frames": len(common),
                    "angle_deg": _pct(angles, 1),
                    "spacing_cm": _pct(spacings),
                    "midpoint_distance_cm": _pct(mids),
                }
            )
    return out


def tracks_cross_check(
    tracks: Iterable[Track3D], window: StandWindow | None, classes: Sequence[str]
) -> dict[str, Any]:
    """The point tracker's observed rows of each class over the stand window: track ids,
    observed frames, median position and drift (the tracker never wrote its mover flag, so
    the drift stands in for it)."""
    if window is None:
        return {}
    by_class: dict[str, dict[str, list[Track3D]]] = defaultdict(lambda: defaultdict(list))
    for t in tracks:
        if t.object_class in classes and window.start <= t.frame_index <= window.end:
            by_class[t.object_class][t.track_id].append(t)
    out: dict[str, Any] = {}
    for cls in classes:
        entries = []
        for tid, rows in sorted(by_class.get(cls, {}).items()):
            observed = [r for r in rows if r.state == "observed"]
            positions = (
                np.array([r.position_cm for r in observed]) if observed else np.zeros((0, 3))
            )
            entries.append(
                {
                    "track_id": tid,
                    "rows": len(rows),
                    "observed_rows": len(observed),
                    "states": dict(Counter(r.state for r in rows)),
                    "position_drift": _drift(positions),
                }
            )
        out[cls] = {
            "tracks": entries,
            "observed_fraction_of_window": round(
                sum(e["observed_rows"] for e in entries) / len(window.frames), 3
            ),
        }
    return out


def shared_length(
    per_pipette: Mapping[str, dict[str, Any]], classes: Sequence[str], *, min_frames: int = 30
) -> dict[str, Any]:
    """Median of the single-channel pipettes' median resting lengths, over the pipettes with
    at least `min_frames` resting frames; the spread is their max - min."""
    medians = {
        cls: per_pipette[cls]["length_cm"]["median"]
        for cls in classes
        if cls in per_pipette
        and per_pipette[cls]["length_cm"]["median"] is not None
        and per_pipette[cls]["length_cm"]["n"] >= min_frames
    }
    excluded = [cls for cls in classes if cls in per_pipette and cls not in medians]
    if not medians:
        return {"length_cm": None, "spread_cm": None, "per_pipette_median_cm": {}, "pipettes": 0}
    values = np.array(list(medians.values()))
    return {
        "length_cm": round(float(np.median(values)), 2),
        "spread_cm": round(float(values.max() - values.min()), 2),
        "per_pipette_median_cm": {k: round(float(v), 2) for k, v in medians.items()},
        "pipettes": len(medians),
        "excluded_too_few_resting_frames": excluded,
        "rule": (
            "median of the single-channel pipettes' median resting lengths (pipettes with at "
            f"least {min_frames} resting frames); spread = max - min"
        ),
    }


# --------------------------------------------------------------------------- report


def stand_report(
    rows: Sequence[FineBioObservation],
    cams: Mapping[str, Camera],
    *,
    trial: str,
    settings: StandSettings | None = None,
    tracks: Iterable[Track3D] | None = None,
    frames: range | None = None,
) -> tuple[dict[str, Any], dict[str, list[Segment]]]:
    settings = settings or StandSettings()
    if frames is not None:
        wanted = set(frames)
        rows = [r for r in rows if r.frame_index in wanted]
    static = static_frames(rows, settings)
    all_segments = segments_on_static_frames(rows, cams, static, settings)
    segments = resting_segments(all_segments, settings)
    window = stand_window(
        {cls: [s.frame_index for s in segs] for cls, segs in segments.items()}, settings
    )
    per_pipette = {}
    for cls, segs in sorted(all_segments.items()):
        summary = pipette_summary(segments.get(cls, []), window)
        summary["frames_static_with_axes"] = len(segs)
        summary["frames_failed_rest_gate"] = len(segs) - len(segments.get(cls, []))
        summary["frames_failed_residual_gate"] = sum(
            1 for s in segs if s.max_residual_px > settings.rest_residual_px
        )
        summary["frames_failed_height_gate"] = sum(
            1 for s in segs if not s.within_rest_height(settings)
        )
        summary["max_residual_px_all_static"] = _pct([s.max_residual_px for s in segs], 1)
        summary["lower_end_height_cm_all_static"] = _pct([s.lower_end_height_cm for s in segs], 1)
        summary["views_dropped"] = dict(Counter(v for s in segs for v in s.views_dropped))
        per_pipette[cls] = summary
    static_summary = {
        cls: {
            "static_frames": len(frames_),
            "runs": _runs(frames_)[:20],
            "views": dict(Counter(v for views in frames_.values() for v in views)),
        }
        for cls, frames_ in sorted(static.items())
    }
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "trial": trial,
        "settings": settings.as_dict(),
        "bench_normal": [float(x) for x in BENCH_UP],
        "stand_window": None if window is None else window.as_dict(),
        "static_by_class": static_summary,
        "per_pipette": per_pipette,
        "pairwise": pairwise_summary(segments, window, settings.classes),
        "shared_length": shared_length(
            per_pipette, settings.shared_classes, min_frames=settings.min_run_frames
        ),
        "tracks_cross_check": (
            tracks_cross_check(tracks, window, settings.classes) if tracks is not None else None
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "licence_note": LICENCE_NOTE,
    }
    return report, segments


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _pct_cell(block: Mapping[str, Any], digits: int = 2) -> str:
    if not block or block.get("n", 0) == 0:
        return "-"
    median, p10, p90 = (_fmt(block[k], digits) for k in ("median", "p10", "p90"))
    return f"{median} ({p10}–{p90})"


def report_markdown(report: dict[str, Any]) -> str:
    window = report["stand_window"]
    lines = [f"# Pipettes at rest, {report['trial']}", ""]
    if window is None:
        lines += ["No pipette is static in the window by the detector's boxes.", ""]
    else:
        settings = report["settings"]
        lines += [
            f"Stand window: raw frames {window['start_frame']}–{window['end_frame']} "
            f"({window['frames']} frames), {window['static_pipettes']} single-channel pipettes "
            f"static at once ({', '.join(window['classes']) or 'none for 90% of it'}). A pipette "
            f"is static when its detector box centre moves at most {settings['max_step_px']} px "
            f"a frame for {settings['min_run_frames']} frames in at least "
            f"{settings['min_static_views']} fixed views.",
            "",
        ]
    lines += [
        "## Static stretches per class",
        "",
        "| class | static frames | views | first runs |",
        "|---|---|---|---|",
    ]
    for cls, s in report["static_by_class"].items():
        runs = ", ".join(f"{a}–{b}" for a, b in s["runs"][:6])
        lines.append(f"| {cls} | {s['static_frames']} | {s['views']} | {runs} |")
    gate = report["settings"]["rest_residual_px"]
    height = report["settings"]["max_rest_height_cm"]
    lines += [
        "",
        f"## Per pipette (static frames with axes in >= 2 fixed views, every endpoint "
        f"residual within {gate} px, the lower end within {height} cm of the bench plane)",
        "",
        "| class | static frames with axes | failed the rest gate (residual / height) | "
        "resting frames | length cm | "
        "angle to bench normal deg | end A height cm | end B height cm | "
        "end A drift p90 / max cm (window) | end B drift p90 / max cm | view sets |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cls, p in report["per_pipette"].items():
        da, db = p["end_a_drift_window"], p["end_b_drift_window"]
        lines.append(
            f"| {cls} | {p['frames_static_with_axes']} | {p['frames_failed_rest_gate']} "
            f"({p['frames_failed_residual_gate']} / {p['frames_failed_height_gate']}) | "
            f"{p['frames_with_segment']} | {_pct_cell(p['length_cm'])} | "
            f"{_pct_cell(p['angle_to_bench_normal_deg'], 1)} | "
            f"{_pct_cell(p['end_a_height_cm'], 1)} | {_pct_cell(p['end_b_height_cm'], 1)} | "
            f"{_fmt(da.get('p90_cm'))} / {_fmt(da.get('max_cm'))} | "
            f"{_fmt(db.get('p90_cm'))} / {_fmt(db.get('max_cm'))} | {p['view_sets']} |"
        )
    lines += [
        "",
        "## Per-view endpoint residuals (px, median (p10–p90); LOO with >= 3 views)",
        "",
        "| class | view | end A residual | end B residual | end A LOO | end B LOO | "
        "up-order disagreements |",
        "|---|---|---|---|---|---|---|",
    ]
    for cls, p in report["per_pipette"].items():
        for view in sorted(set(p["residual_a_px_per_view"]) | set(p["loo_a_px_per_view"])):
            lines.append(
                f"| {cls} | {view} | {_pct_cell(p['residual_a_px_per_view'].get(view, {}), 1)} | "
                f"{_pct_cell(p['residual_b_px_per_view'].get(view, {}), 1)} | "
                f"{_pct_cell(p['loo_a_px_per_view'].get(view, {}), 1)} | "
                f"{_pct_cell(p['loo_b_px_per_view'].get(view, {}), 1)} | "
                f"{p['up_order_disagreements_by_view'].get(view, 0)} |"
            )
    if report["pairwise"]:
        lines += [
            "",
            "## Pairs over the stand window",
            "",
            "| pair | frames | angle deg | spacing cm (midline) | midpoint distance cm |",
            "|---|---|---|---|---|",
        ]
        for pair in report["pairwise"]:
            lines.append(
                f"| {pair['pair'][0]} / {pair['pair'][1]} | {pair['frames']} | "
                f"{_pct_cell(pair['angle_deg'], 1)} | {_pct_cell(pair['spacing_cm'])} | "
                f"{_pct_cell(pair['midpoint_distance_cm'])} |"
            )
    shared = report["shared_length"]
    lines += [
        "",
        "## Shared length",
        "",
        f"**{_fmt(shared['length_cm'])} cm**, spread {_fmt(shared['spread_cm'])} cm across "
        f"{shared.get('pipettes', 0)} single-channel pipettes ({shared['per_pipette_median_cm']}"
        f"; excluded for too few resting frames: "
        f"{shared.get('excluded_too_few_resting_frames', [])}).",
    ]
    cross = report.get("tracks_cross_check")
    if cross:
        lines += [
            "",
            "## Point tracker over the stand window (tracks-ext)",
            "",
            "| class | observed fraction | tracks (id: observed rows, drift p90 cm) |",
            "|---|---|---|",
        ]
        for cls, c in cross.items():
            tracks = "; ".join(
                f"{t['track_id']}: {t['observed_rows']}, {_fmt(t['position_drift'].get('p90_cm'))}"
                for t in c["tracks"]
            )
            lines.append(f"| {cls} | {c['observed_fraction_of_window']} | {tracks or '-'} |")
    lines += ["", report["claim_boundary"], "", report["licence_note"], ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- config


def pipettes_config(
    reports: Sequence[dict[str, Any]],
    *,
    run_dirs: Sequence[str],
    primary: int = 0,
) -> dict[str, Any]:
    """`configs/finebio/pipettes.json`: the shared length prior from `reports[primary]` with
    the other trials as a check; a disagreement beyond the spread is written down and the
    primary trial's value kept."""
    main = reports[primary]
    shared = main["shared_length"]
    spacings = [
        p["spacing_cm"]["median"]
        for p in main["pairwise"]
        if all(c in SINGLE_CHANNEL_PIPETTES for c in p["pair"]) and p["spacing_cm"]["median"]
    ]
    per_trial = {}
    notes = []
    for i, (rep, run_dir) in enumerate(zip(reports, run_dirs)):
        s = rep["shared_length"]
        per_trial[rep["trial"]] = {
            "length_cm": s["length_cm"],
            "spread_cm": s["spread_cm"],
            "per_pipette_median_cm": s["per_pipette_median_cm"],
            "stand_window": rep["stand_window"],
            "run_dir": run_dir,
            "primary": i == primary,
        }
        if i != primary and s["length_cm"] is not None and shared["length_cm"] is not None:
            gap = abs(s["length_cm"] - shared["length_cm"])
            tolerance = max(shared["spread_cm"] or 0.0, s["spread_cm"] or 0.0)
            if gap > tolerance:
                notes.append(
                    f"{rep['trial']} measures {s['length_cm']} cm against {main['trial']}'s "
                    f"{shared['length_cm']} cm, a gap of {gap:.2f} cm over the spread of "
                    f"{tolerance:.2f} cm; {main['trial']}'s value is kept"
                )
            else:
                notes.append(
                    f"{rep['trial']} measures {s['length_cm']} cm, within the spread of "
                    f"{main['trial']}'s {shared['length_cm']} cm"
                )
    return {
        "schema_version": "1.0",
        "config_kind": CONFIG_KIND,
        "length_cm": shared["length_cm"],
        "length_spread_cm": shared["spread_cm"],
        "stand_spacing_cm": round(float(np.median(spacings)), 2) if spacings else None,
        "stand_spacing_pairs_cm": {
            f"{p['pair'][0]}/{p['pair'][1]}": p["spacing_cm"]["median"] for p in main["pairwise"]
        },
        "elongation_threshold": ELONGATION_THRESHOLD,
        "merged_width_factor": MERGED_WIDTH_FACTOR,
        "merged_extent_factor": MERGED_EXTENT_FACTOR,
        "units": "board centimetres; pixels are raw video pixels",
        "provenance": {
            "trial": main["trial"],
            "frames": (
                [main["stand_window"]["start_frame"], main["stand_window"]["end_frame"]]
                if main["stand_window"]
                else None
            ),
            "run_dir": run_dirs[primary],
            "method": (
                "battle-finebio-stand: per static frame, the two SAM3 mask-axis endpoints of "
                "every fixed view with an axis are matched across views (projected world-up in "
                "the best-aligned view, then the pairing with the smaller two-view residual) and "
                "triangulated (undistort + DLT); length = |end A - end B|; the shared length is "
                "the median of the single-channel pipettes' median lengths and the spread their "
                "max - min; spacing is the distance between segment midlines over the stand window"
            ),
            "per_trial": per_trial,
            "notes": notes,
            "claim_boundary": CLAIM_BOUNDARY,
        },
    }


# --------------------------------------------------------------------------- cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    rep = sub.add_parser("report", help="stand frames and geometry of one trial")
    rep.add_argument(
        "--observations", type=Path, required=True, help="remeasured observations.jsonl"
    )
    rep.add_argument("--cameras", type=Path, required=True, help="FineBioCameraConfig JSON")
    rep.add_argument(
        "--tracks", type=Path, default=None, help="tracks-ext/tracks.jsonl (cross-check)"
    )
    rep.add_argument("--output", type=Path, required=True)
    rep.add_argument("--frames", default=None, help="a-b raw frame range (inclusive)")
    rep.add_argument("--min-run-frames", type=int, default=StandSettings.min_run_frames)
    rep.add_argument("--max-step-px", type=float, default=StandSettings.max_step_px)
    rep.add_argument("--min-static-views", type=int, default=StandSettings.min_static_views)
    rep.add_argument("--min-axis-views", type=int, default=StandSettings.min_axis_views)
    cfg = sub.add_parser("config", help="pipettes.json from one or two stand reports")
    cfg.add_argument(
        "--report",
        action="append",
        required=True,
        help="stand_report.json (first = primary trial); repeat for a second trial",
    )
    cfg.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "report":
        config = read_camera_config(args.cameras)
        cams = cameras_from_config(config)
        rows = list(read_jsonl(args.observations, FineBioObservation))
        tracks = list(read_jsonl(args.tracks, Track3D)) if args.tracks is not None else None
        frames = None
        if args.frames:
            a, b = (int(x) for x in args.frames.split("-"))
            frames = range(a, b + 1)
        settings = StandSettings(
            min_run_frames=args.min_run_frames,
            max_step_px=args.max_step_px,
            min_static_views=args.min_static_views,
            min_axis_views=args.min_axis_views,
            fixed_views=tuple(config.fixed),
        )
        report, segments = stand_report(
            rows, cams, trial=config.trial, settings=settings, tracks=tracks, frames=frames
        )
        report["inputs"] = {
            "observations": str(args.observations),
            "cameras": str(args.cameras),
            "tracks": None if args.tracks is None else str(args.tracks),
        }
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "stand_report.json").write_text(
            json.dumps(report, indent=1) + "\n", encoding="utf-8"
        )
        with (args.output / "segments.jsonl").open("w", encoding="utf-8") as handle:
            for cls in sorted(segments):
                for seg in segments[cls]:
                    handle.write(json.dumps(seg.to_record()) + "\n")
        markdown = report_markdown(report)
        (args.output / "stand_report.md").write_text(markdown, encoding="utf-8")
        print(markdown)
        return 0
    if args.command == "config":
        reports = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.report]
        run_dirs = [str(Path(p).parent) for p in args.report]
        doc = pipettes_config(reports, run_dirs=run_dirs)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(doc, indent=1))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
