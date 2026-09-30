"""The label-free scoreboard of the pipettes-as-lines extension (plan `p3-metrics`, Sep 28).

`battle-finebio-arms lines-scoreboard` reads three tracker outputs over one trial window (the
Sep 27 point tracker with the extensions, the same point tracker on the remeasured rows, and
the line extension on those rows), the remeasured observations, the clip's cameras, the rig
and the length prior, and writes `lines_scoreboard.{json,md}`. Every row names what it is
measured against, because no 3D ground truth exists for these trials:

* **ids per pipette class**, lifetimes and births a minute, for all three runs; the line
  tracker's tracks are counted by their plurality `observed_class`, the point tracker's by
  class.
* **leave-one-camera-out (LOO) residual**: the line tracker's own (`extensions.lines`), the
  same recomputed from the rows, and the **box-centre baseline over the same cells**: for
  every frame where a line track's fit had three or more axis views (planes), the centroids of
  all but one view are triangulated (`centroid_point`), projected into the held-out view and
  measured against that view's axis midpoint and against its axis line. The line has to beat
  the axis-line number, which is the like-for-like one; the midpoint number is the point
  method's honest error.
* **length**: the visible extent per class against the per-class prior.
* **flat-rest geometry**: on frames where a line track is stationary (midpoint speed under
  `REST_SPEED_CM_PER_FRAME` for `REST_MIN_FRAMES` frames or more) the angle to the bench
  normal and the drift of each end, against the stand slice's report and the rig's static
  residual turned into centimetres at the resting depth.
* **plausibility**: tip and midpoint speed, the fraction of steps over `STEP_JUMP_CM`, and
  the butt-to-hand distance while `held`, against physics.
* **ambiguities**: frames outside the rest frames where two pipette lines sit within
  `AMBIGUITY_PERPENDICULAR_CM` and `AMBIGUITY_ANGLE_DEG`, the near-duplicate pairs, and the
  tracker's id switches with the note that a colour-cross association counts as a switch
  under the geometric class.
* **colour**: the vote's identity, confidence and entropy per track and the tracks whose
  colour disagrees with their plurality detector class, with the slots they came from.
* **merged views**, line / aided / fallback frame fractions, the tip-resolved fraction.
* **negative controls** from `lines-negative-controls` (the camera-6 shipped pose; one view
  shifted a frame), each against the normal run over the same frames.
* **the pre-registered rule** as booleans with their numbers.

Claim boundary: the detector's classes and SAM3's masks are model output; the stand report,
the rig and the colour vote are label-free checks of the same pipeline; nothing here is
accuracy against ground truth. Tip error waits for the clicked anchors (`p3-tip-anchors`).
FineBio is non-commercial research data; nothing under `runs/` or `data/` is committed.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from .finebio_cameras import (
    Camera,
    cameras_from_config,
    fixed_camera,
    read_camera_config,
)
from .finebio_slice import depth_cm, resolve_fpv_source
from .multiview_lines import (
    ELONGATION_THRESHOLD,
    MIN_PAIR_ANGLE_DEG,
    AxisObs,
    Line3D,
    centroid_loo_residual,
    is_elongated,
    line_distance,
    line_endpoints,
    loo_residual,
)
from .multiview_schemas import FINEBIO_FPV_VIEW, FineBioObservation, Track3D, read_jsonl
from .multiview_tracks import LinePrior

SCHEMA = "battle-finebio-lines-scoreboard/1"
FPS = 30000 / 1001
GEOMETRIC_CLASS = "pipette"
PIPETTE_CLASSES: tuple[str, ...] = (
    "blue_pipette",
    "yellow_pipette",
    "red_pipette",
    "8_channel_pipette",
)
CLASS_COLOUR: dict[str, str] = {
    "blue_pipette": "blue",
    "yellow_pipette": "yellow",
    "red_pipette": "red",
}
LOCALISED_STATES = ("observed", "single_view")
LIVE_STATES = ("observed", "single_view", "coasting", "held", "contained")
BENCH_NORMAL = np.array([0.0, 0.0, -1.0])
# A line track is at rest when its midpoint moves under this for at least this many frames.
REST_SPEED_CM_PER_FRAME = 0.2
REST_MIN_FRAMES = 30
# Two pipette lines this close and this parallel outside the rest frames are an ambiguity.
AMBIGUITY_PERPENDICULAR_CM = 1.5
AMBIGUITY_ANGLE_DEG = 15.0
# A tip or midpoint step over this in one frame is implausible.
STEP_JUMP_CM = 10.0
# The pre-registered rule: ids per held pipette fall at least this factor.
ID_FALL_FACTOR = 5.0
# A negative control "rises clearly" when its LOO median is at least this factor over normal.
NEGATIVE_CONTROL_RISE = 1.5
DEFAULT_CONTROL_FRAMES = (600, 899)
CLAIM_BOUNDARY = (
    "No 3D ground truth exists for these trials. The LOO residuals are held-out camera "
    "consistency of the same masks; ids, lifetimes and ambiguities are the tracker's own; the "
    "stand report, the rig and the colour vote are label-free checks of the same pipeline; "
    "colour agreement is against the detector's class, itself an appearance vote. Tip error "
    "waits for the clicked anchors (p3-tip-anchors)."
)
LICENCE_NOTE = (
    "FineBio is licensed for non-commercial research; frames, masks and videos derived from it "
    "stay under runs/ or data/ and are never committed or redistributed"
)


# --------------------------------------------------------------------------- io


def resolve_tracks_dir(path: Path, default_name: str) -> Path:
    """`path` when it holds `tracks.jsonl`, else `path/<default_name>`."""
    path = Path(path)
    if (path / "tracks.jsonl").is_file():
        return path
    candidate = path / default_name
    if (candidate / "tracks.jsonl").is_file():
        return candidate
    raise FileNotFoundError(f"no tracks.jsonl under {path} or {candidate}")


def tracks_path(tracks_dir: Path) -> Path:
    """`tracks_oriented.jsonl` when it sits beside `tracks.jsonl`, else `tracks.jsonl`.

    A v3 or v4 directory has no sibling, so those boards keep reading `tracks.jsonl`.
    """
    directory = Path(tracks_dir)
    oriented = directory / "tracks_oriented.jsonl"
    if oriented.is_file():
        return oriented
    return directory / "tracks.jsonl"


def load_tracks(tracks_dir: Path) -> list[Track3D]:
    return list(read_jsonl(tracks_path(tracks_dir), Track3D))


def load_metrics(tracks_dir: Path) -> dict[str, Any]:
    return json.loads((Path(tracks_dir) / "identity_metrics.json").read_text(encoding="utf-8"))


def pipette_observations(
    path: Path, classes: Iterable[str] = PIPETTE_CLASSES
) -> dict[tuple[str, int, str], FineBioObservation]:
    """The pipette-class rows of an observations file (a directory or the JSONL itself)
    keyed by ``(view, frame, slot)``; when a detector row and a SAM3 row share a key the
    SAM3 row wins, as the tracker's source rule does."""
    path = Path(path)
    if path.is_dir():
        path = path / "observations.jsonl"
    wanted = set(classes)
    out: dict[tuple[str, int, str], FineBioObservation] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            if raw.get("object_class") not in wanted:
                continue
            row = FineBioObservation.model_validate(raw)
            key = (row.view, row.frame_index, row.slot)
            current = out.get(key)
            if current is None or (current.source == "detector" and row.source != "detector"):
                out[key] = row
    return out


def _percentiles(values: Sequence[float], digits: int = 3) -> dict[str, Any]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "median": None, "p10": None, "p90": None, "max": None}
    return {
        "n": int(arr.size),
        "median": round(float(np.median(arr)), digits),
        "p10": round(float(np.percentile(arr, 10)), digits),
        "p90": round(float(np.percentile(arr, 90)), digits),
        "max": round(float(arr.max()), digits),
    }


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _stats_cells(block: Mapping[str, Any], digits: int = 2) -> str:
    """`median | p90 | max | n` cells of a `_percentiles` block."""
    return (
        f"{_fmt(block.get('median'), digits)} | {_fmt(block.get('p90'), digits)} | "
        f"{_fmt(block.get('max'), digits)} | {block.get('n', '-')}"
    )


def _pct_cell(block: Mapping[str, Any], digits: int = 2) -> str:
    if not block or not block.get("n"):
        return "-"
    median, p10, p90 = (_fmt(block[k], digits) for k in ("median", "p10", "p90"))
    return f"{median} ({p10}–{p90})"


# --------------------------------------------------------------------------- tracks


def is_pipette_row(row: Track3D, geometric: bool) -> bool:
    if geometric:
        return row.object_class == GEOMETRIC_CLASS
    return row.object_class in PIPETTE_CLASSES


def rows_by_track(rows: Iterable[Track3D], keep: Callable[[Track3D], bool]) -> dict[str, list]:
    by: dict[str, list[Track3D]] = defaultdict(list)
    for row in rows:
        if keep(row):
            by[row.track_id].append(row)
    for trows in by.values():
        trows.sort(key=lambda r: r.frame_index)
    return dict(by)


def plurality_class(trows: Sequence[Track3D], geometric: bool) -> str | None:
    """The line tracker writes the running plurality on every row: the last row carries the
    final one. A point track's class is its own."""
    if not trows:
        return None
    if geometric:
        return trows[-1].observed_class
    return trows[-1].object_class


def lifetimes(by_track: Mapping[str, Sequence[Track3D]]) -> dict[str, Any]:
    spans = [t[-1].frame_index - t[0].frame_index + 1 for t in by_track.values() if t]
    return _percentiles(spans, digits=1)


def ids_summary(
    metrics: Mapping[str, Any],
    by_track: Mapping[str, Sequence[Track3D]],
    *,
    geometric: bool,
    minutes: float,
) -> dict[str, Any]:
    """Per pipette class: ids born (the tracker's `per_class` when present, else counted from
    the rows), lifetimes and births a minute; the geometric class's own count for a line run."""
    per_class = metrics.get("per_class", {})
    classes: dict[str, list[str]] = defaultdict(list)
    for tid, trows in by_track.items():
        cls = plurality_class(trows, geometric)
        classes[cls or "undecided"].append(tid)
    out: dict[str, Any] = {"by_class": {}, "counted_by": None, "minutes": round(minutes, 3)}
    for cls in PIPETTE_CLASSES:
        own = per_class.get(cls, {})
        from_rows = len(classes.get(cls, ()))
        ids = int(own.get("tracks_born", from_rows))
        out["by_class"][cls] = {
            "ids": ids,
            "ids_from_rows": from_rows,
            "fragmentation": own.get("fragmentation"),
            "max_simultaneous": own.get("max_simultaneous"),
            "lifetime_frames": lifetimes({t: by_track[t] for t in classes.get(cls, ())}),
            "births_per_minute": round(ids / minutes, 2) if minutes > 0 else None,
        }
        if geometric and own.get("counted_by"):
            out["counted_by"] = own["counted_by"]
    if geometric:
        geo = per_class.get(GEOMETRIC_CLASS, {})
        out["geometric"] = {
            "ids": int(geo.get("tracks_born", len(by_track))),
            "fragmentation": geo.get("fragmentation"),
            "max_simultaneous": geo.get("max_simultaneous"),
            "lifetime_frames": lifetimes(by_track),
            "births_per_minute": (
                round(int(geo.get("tracks_born", len(by_track))) / minutes, 2)
                if minutes > 0
                else None
            ),
            "undecided_class": len(classes.get("undecided", ())),
        }
    out["pipette_ids_total"] = sum(v["ids"] for v in out["by_class"].values())
    return out


# --------------------------------------------------------------------------- LOO and length


def line_from_row(row: Track3D) -> Line3D:
    assert row.endpoints_cm is not None and row.direction is not None
    ends = np.asarray(row.endpoints_cm, dtype=np.float64)
    return Line3D(point=ends.mean(axis=0), direction=np.asarray(row.direction), endpoints=ends)


def frame_members(
    row: Track3D,
    observations: Mapping[tuple[str, int, str], FineBioObservation],
    cams: Mapping[str, Camera],
    *,
    fpv_weight: float,
) -> dict[str, tuple[Camera, AxisObs]]:
    """The views of a line-fit row (the keys of `line_residual_px`, the members the tracker
    kept) as `multiview_lines` records, looked up by the row's support slots."""
    members: dict[str, tuple[Camera, AxisObs]] = {}
    for view in row.line_residual_px or {}:
        slot = row.support_slots.get(view)
        if slot is None or view not in cams:
            continue
        obs = observations.get((view, row.frame_index, slot))
        if obs is None:
            continue
        axis = AxisObs(
            view,
            None if obs.mask_axis_px is None else np.asarray(obs.mask_axis_px, dtype=np.float64),
            np.asarray(obs.point_px, dtype=np.float64),
            obs.mask_elongation,
            (fpv_weight if view == FINEBIO_FPV_VIEW else 1.0)
            * (0.25 if obs.provenance.get("reprojection_disagrees") else 1.0),
        )
        members[view] = (cams[view], axis)
    return members


def loo_and_length(
    lines_rows: Sequence[Track3D],
    observations: Mapping[tuple[str, int, str], FineBioObservation],
    fixed_cams: Mapping[str, Camera],
    fpv_source: Callable[[int], Camera | None],
    prior: LinePrior,
    *,
    fpv_weight: float = 0.5,
    elongation_threshold: float = ELONGATION_THRESHOLD,
    min_pair_angle_deg: float = MIN_PAIR_ANGLE_DEG,
) -> dict[str, Any]:
    """One pass over the line-fit rows: the box-centre LOO and the recomputed line LOO on the
    frames with three or more planes (the cells the tracker's own LOO covers), and the
    visible length per class on every line-fit row with a plane."""
    box_distance: list[float] = []
    box_perpendicular: list[float] = []
    line_perpendicular: list[float] = []
    line_angle: list[float] = []
    box_by_view: dict[str, list[float]] = defaultdict(list)
    line_by_view: dict[str, list[float]] = defaultdict(list)
    lengths: dict[str, list[float]] = defaultdict(list)
    deviations: dict[str, list[float]] = defaultdict(list)
    loo_frames = 0
    rows_with_members = 0
    rows_missing = 0
    for row in lines_rows:
        if row.object_class != GEOMETRIC_CLASS or not row.line_residual_px:
            continue
        if row.endpoints_cm is None or row.direction is None:
            continue
        cams = dict(fixed_cams)
        fpv = fpv_source(row.frame_index)
        if fpv is not None:
            cams[FINEBIO_FPV_VIEW] = fpv
        members = frame_members(row, observations, cams, fpv_weight=fpv_weight)
        if len(members) < len(row.line_residual_px):
            rows_missing += 1
        if not members:
            continue
        rows_with_members += 1
        pairs = list(members.values())
        planes = [v for v, (_, axis) in members.items() if is_elongated(axis, elongation_threshold)]
        cls = row.observed_class or "undecided"
        extent = line_endpoints(
            line_from_row(row), pairs, elongation_threshold=elongation_threshold
        )
        if extent is not None:
            visible = extent.visible_length_cm
            lengths[cls].append(visible)
            deviations[cls].append(visible - prior.length_for(row.observed_class))
        if len(planes) < 3:
            continue
        loo_frames += 1
        plane_set = set(planes)
        for record in centroid_loo_residual(pairs):
            if record["view"] not in plane_set or not record["fitted"]:
                continue
            if record["distance_px"] is not None and np.isfinite(record["distance_px"]):
                box_distance.append(float(record["distance_px"]))
                box_by_view[record["view"]].append(float(record["distance_px"]))
            if record["perpendicular_px"] is not None and np.isfinite(record["perpendicular_px"]):
                box_perpendicular.append(float(record["perpendicular_px"]))
        for record in loo_residual(pairs, elongation_threshold, min_pair_angle_deg):
            if record["fitted"] and np.isfinite(record["perpendicular_px"]):
                line_perpendicular.append(float(record["perpendicular_px"]))
                line_by_view[record["view"]].append(float(record["perpendicular_px"]))
                if record["angle_deg"] is not None and np.isfinite(record["angle_deg"]):
                    line_angle.append(float(record["angle_deg"]))
    return {
        "frames_with_3_planes": loo_frames,
        "line_fit_rows": rows_with_members,
        "line_fit_rows_with_missing_observations": rows_missing,
        "box_centre": {
            "distance_to_axis_midpoint_px": _percentiles(box_distance),
            "perpendicular_to_axis_px": _percentiles(box_perpendicular),
            "by_view_distance_median_px": {
                v: round(float(np.median(x)), 3) for v, x in sorted(box_by_view.items())
            },
            "method": (
                "for each frame of a line track whose fit had >= 3 axis views, the centroids "
                "of all but one view triangulated (centroid_point, the point tracker's DLT with "
                "its view weights), projected into the held-out view; distance to that view's "
                "axis midpoint and perpendicular distance to its axis line, undistorted px"
            ),
        },
        "line_recomputed": {
            "perpendicular_px": _percentiles(line_perpendicular),
            "angle_deg": _percentiles(line_angle),
            "by_view_median_px": {
                v: round(float(np.median(x)), 3) for v, x in sorted(line_by_view.items())
            },
            "method": (
                "loo_residual over the same members and frames: the line fitted from the other "
                "views, the held-out axis endpoints' mean distance to its projection (px) and "
                "the angle between axis and projection (deg)"
            ),
        },
        "length_by_class": {
            cls: {
                "visible_cm": _percentiles(lengths[cls]),
                "spread_p10_p90_cm": (
                    round(
                        float(np.percentile(lengths[cls], 90) - np.percentile(lengths[cls], 10)), 3
                    )
                    if lengths[cls]
                    else None
                ),
                "prior_cm": round(prior.length_for(cls if cls != "undecided" else None), 3),
                "deviation_from_prior_cm": _percentiles(deviations[cls]),
                "abs_deviation_from_prior_cm": _percentiles([abs(d) for d in deviations[cls]]),
            }
            for cls in sorted(lengths)
        },
        "length_method": (
            "the row's line (endpoints and direction) with the frame's kept views: each axis "
            "view's endpoint rays give an interval of the line parameter, the visible extent is "
            "their robust union (line_endpoints without the prior); deviation is visible minus "
            "the per-class prior"
        ),
    }


# --------------------------------------------------------------------------- rest geometry


def stationary_runs(
    trows: Sequence[Track3D],
    *,
    speed_cm_per_frame: float = REST_SPEED_CM_PER_FRAME,
    min_frames: int = REST_MIN_FRAMES,
) -> list[list[Track3D]]:
    """Maximal runs of consecutive localised line rows whose midpoint moves under the speed
    on every step, at least `min_frames` long. Coasting rows are left out (the stationary
    prior would make every occlusion look like rest)."""
    rows = [r for r in trows if r.state in LOCALISED_STATES and r.endpoints_cm is not None]
    runs: list[list[Track3D]] = []
    current: list[Track3D] = []
    for prev, row in zip([None, *rows], rows):
        if prev is None:
            current = [row]
            continue
        gap = row.frame_index - prev.frame_index
        step = float(np.linalg.norm(np.subtract(row.position_cm, prev.position_cm)))
        if gap == 1 and step < speed_cm_per_frame:
            current.append(row)
        else:
            if len(current) >= min_frames:
                runs.append(current)
            current = [row]
    if len(current) >= min_frames:
        runs.append(current)
    return runs


def _aligned_endpoints(run: Sequence[Track3D]) -> np.ndarray:
    """(n, 2, 3) endpoints with each frame's order chosen to follow the first frame's, so an
    unresolved tip does not flip an end between frames."""
    ends = np.asarray([r.endpoints_cm for r in run], dtype=np.float64)
    reference = ends[0]
    out = ends.copy()
    for i in range(1, len(ends)):
        same = np.linalg.norm(ends[i] - reference, axis=1).sum()
        swapped = np.linalg.norm(ends[i][::-1] - reference, axis=1).sum()
        if swapped < same:
            out[i] = ends[i][::-1]
    return out


def rest_geometry(
    lines_rows: Sequence[Track3D],
    stand_report: Mapping[str, Any] | None,
    *,
    fixed_cams: Mapping[str, Camera] | None = None,
    rig_static_loo_px: float | None = None,
    prior: LinePrior | None = None,
) -> tuple[dict[str, Any], dict[str, set[int]]]:
    """Per class over the stationary runs: the angle between the line and the bench normal
    (0 upright, 90 flat) and each end's drift from its run median; beside them the stand
    report's numbers for the class and the rig's static residual in centimetres at the
    resting depth (static LOO median px times the median depth over focal length of the fixed
    views that supported the rest rows). Returns the block and the rest frames per track."""
    by_track = rows_by_track(lines_rows, lambda r: r.object_class == GEOMETRIC_CLASS)
    angles: dict[str, list[float]] = defaultdict(list)
    drift_a: dict[str, list[float]] = defaultdict(list)
    drift_b: dict[str, list[float]] = defaultdict(list)
    runs_by_class: Counter = Counter()
    frames_by_class: Counter = Counter()
    rest_frames: dict[str, set[int]] = defaultdict(set)
    cm_per_px: list[float] = []
    for tid, trows in by_track.items():
        cls = plurality_class(trows, True) or "undecided"
        for run in stationary_runs(trows):
            runs_by_class[cls] += 1
            frames_by_class[cls] += len(run)
            rest_frames[tid].update(r.frame_index for r in run)
            ends = _aligned_endpoints(run)
            directions = ends[:, 1] - ends[:, 0]
            norms = np.linalg.norm(directions, axis=1)
            ok = norms > 1e-9
            cosines = np.abs(directions[ok] @ BENCH_NORMAL) / norms[ok]
            angles[cls].extend(np.degrees(np.arccos(np.clip(cosines, 0.0, 1.0))).tolist())
            for k, sink in ((0, drift_a), (1, drift_b)):
                median = np.median(ends[:, k], axis=0)
                sink[cls].extend(np.linalg.norm(ends[:, k] - median, axis=1).tolist())
            if fixed_cams:
                for row in run[:: max(1, len(run) // 10)]:
                    for v in row.support_views:
                        cam = fixed_cams.get(v)
                        if cam is None:
                            continue
                        depth = depth_cm(cam, np.asarray(row.position_cm))
                        if depth > 0:
                            cm_per_px.append(depth / float(cam.K[0, 0]))
    noise_cm = None
    if rig_static_loo_px is not None and cm_per_px:
        noise_cm = round(float(rig_static_loo_px) * float(np.median(cm_per_px)), 3)
    stand_per_pipette = (stand_report or {}).get("per_pipette", {}) if stand_report else {}
    by_class: dict[str, Any] = {}
    for cls in sorted(set(angles) | set(runs_by_class)):
        stand = stand_per_pipette.get(cls, {})
        angle = _percentiles(angles[cls], digits=1)
        end_a, end_b = _percentiles(drift_a[cls]), _percentiles(drift_b[cls])
        stand_angle = (stand.get("angle_to_bench_normal_deg") or {}).get("median")
        stand_drift = [
            (stand.get(key) or {}).get("p90_cm")
            for key in ("end_a_drift_window", "end_b_drift_window")
        ]
        stand_drift = [d for d in stand_drift if d is not None]
        drift_p90 = max(x for x in (end_a.get("p90"), end_b.get("p90")) if x is not None)
        angle_noise = None
        if noise_cm is not None and prior is not None:
            angle_noise = round(
                math.degrees(
                    math.atan2(noise_cm, prior.length_for(cls if cls in PIPETTE_CLASSES else None))
                ),
                2,
            )
        by_class[cls] = {
            "runs": runs_by_class[cls],
            "frames": frames_by_class[cls],
            "angle_to_bench_normal_deg": angle,
            "end_a_drift_cm": end_a,
            "end_b_drift_cm": end_b,
            "end_drift_p90_cm": round(drift_p90, 3),
            "stand_report": {
                "angle_to_bench_normal_median_deg": stand_angle,
                "end_drift_p90_cm": max(stand_drift) if stand_drift else None,
                "resting_frames": stand.get("frames_with_segment"),
            },
            "angle_difference_deg": (
                round(abs(angle["median"] - stand_angle), 2)
                if stand_angle is not None and angle["median"] is not None
                else None
            ),
            "angle_noise_deg": angle_noise,
            "within_rig_noise": (
                (drift_p90 <= noise_cm)
                and (
                    stand_angle is None
                    or angle["median"] is None
                    or angle_noise is None
                    or abs(angle["median"] - stand_angle) <= angle_noise
                )
                if noise_cm is not None
                else None
            ),
        }
    verdict = None
    if noise_cm is not None and by_class:
        verdict = all(v["within_rig_noise"] for v in by_class.values())
    block = {
        "rest_rule": (
            f"midpoint speed under {REST_SPEED_CM_PER_FRAME} cm a frame for at least "
            f"{REST_MIN_FRAMES} consecutive localised frames"
        ),
        "rig_static_loo_median_px": rig_static_loo_px,
        "rig_static_noise_cm": noise_cm,
        "noise_method": (
            "the rig's static LOO median in px times the median depth / focal length (cm per "
            "px) of the fixed views supporting the rest rows; the angle tolerance is that "
            "distance over the class's prior length"
        ),
        "by_class": by_class,
        "within_rig_noise_all_classes": verdict,
        "stand_report_rest_rule": (
            (stand_report or {}).get("settings", {}).get("max_step_px")
            and "detector box centre static within max_step_px for min_run_frames in >= 2 fixed "
            "views; segments from matched axis endpoints at rest on the bench"
        ),
    }
    return block, dict(rest_frames)


# --------------------------------------------------------------------------- plausibility


def plausibility(lines_rows: Sequence[Track3D], prior: LinePrior | None = None) -> dict[str, Any]:
    """Tip speed (frames where both rows resolve the tip), midpoint speed (every consecutive
    localised pair), the fraction of steps over `STEP_JUMP_CM`, the butt-to-hand distance
    while `held` (the butt is the far end from the tip when resolved, else the nearer end),
    and the written segment length per row against the prior (a segment over twice the prior
    is a runaway extent, not a pipette)."""
    by_track = rows_by_track(lines_rows, lambda r: r.object_class == GEOMETRIC_CLASS)
    row_lengths: list[float] = []
    runaway = 0
    single_view_big_steps = 0
    hands: dict[tuple[str, int], np.ndarray] = {}
    for row in lines_rows:
        if row.object_class in ("left_hand", "right_hand"):
            hands[(row.track_id, row.frame_index)] = np.asarray(row.position_cm)
    tip_steps: list[float] = []
    mid_steps: list[float] = []
    butt_hand: list[float] = []
    tip_nearer_than_butt = 0
    butt_resolved = 0
    held_rows = 0
    held_without_hand_row = 0
    # Sep 29 row fields (absent on the first run's rows): the update that moved each
    # localised row and the clamp flag.
    updates: Counter = Counter()
    clamped_rows = 0
    localised_line_rows = 0
    for trows in by_track.values():
        for row in trows:
            if row.endpoints_cm is not None:
                length = float(np.linalg.norm(np.subtract(*row.endpoints_cm)))
                row_lengths.append(length)
                if prior is not None and length > 2.0 * prior.length_for(row.observed_class):
                    runaway += 1
                if row.state in LOCALISED_STATES:
                    localised_line_rows += 1
                    updates[row.line_update or "unknown"] += 1
                    if row.extent_clamped:
                        clamped_rows += 1
            if (
                row.state == "held"
                and row.butt_to_hand_cm is not None
                and row.tip_to_hand_cm is not None
                and row.tip_to_hand_cm < row.butt_to_hand_cm
            ):
                tip_nearer_than_butt += 1
        localised = [r for r in trows if r.state in LOCALISED_STATES]
        for prev, row in zip(localised, localised[1:]):
            if row.frame_index - prev.frame_index != 1:
                continue
            mid_step = float(np.linalg.norm(np.subtract(row.position_cm, prev.position_cm)))
            mid_steps.append(mid_step)
            if mid_step > STEP_JUMP_CM and "single_view" in (prev.state, row.state):
                single_view_big_steps += 1
            if (
                row.tip_resolved
                and prev.tip_resolved
                and row.endpoints_cm is not None
                and prev.endpoints_cm is not None
            ):
                tip_steps.append(
                    float(np.linalg.norm(np.subtract(row.endpoints_cm[0], prev.endpoints_cm[0])))
                )
        for row in trows:
            if row.state != "held" or row.endpoints_cm is None:
                continue
            held_rows += 1
            hand = hands.get((row.held_by or "", row.frame_index))
            if hand is None:
                held_without_hand_row += 1
                continue
            ends = np.asarray(row.endpoints_cm)
            if row.tip_resolved:
                butt_resolved += 1
                butt_hand.append(float(np.linalg.norm(ends[1] - hand)))
            else:
                butt_hand.append(float(np.linalg.norm(ends - hand, axis=1).min()))
    tip = np.asarray(tip_steps)
    mid = np.asarray(mid_steps)
    aided_like = sum(updates[k] for k in ("aided", "capped", "predicted"))
    return {
        "tip_speed_cm_per_frame": _percentiles(tip_steps),
        "tip_steps_over_10cm_fraction": (
            round(float(np.mean(tip > STEP_JUMP_CM)), 4) if tip.size else None
        ),
        "midpoint_speed_cm_per_frame": _percentiles(mid_steps),
        "midpoint_steps_over_10cm_fraction": (
            round(float(np.mean(mid > STEP_JUMP_CM)), 4) if mid.size else None
        ),
        "midpoint_steps_over_10cm": int((mid > STEP_JUMP_CM).sum()) if mid.size else 0,
        "midpoint_big_steps_touching_a_single_view_row": single_view_big_steps,
        "row_segment_length_cm": _percentiles(row_lengths),
        "rows_over_twice_the_prior": runaway,
        "rows_over_twice_the_prior_fraction": (
            round(runaway / len(row_lengths), 4) if row_lengths else None
        ),
        "butt_to_hand_cm_while_held": _percentiles(butt_hand),
        "held_rows": held_rows,
        "held_rows_with_resolved_butt": butt_resolved,
        "held_rows_without_hand_row": held_without_hand_row,
        "held_rows_with_the_tip_nearer_the_hand_than_the_butt": tip_nearer_than_butt,
        "localised_line_rows": localised_line_rows,
        "rows_by_update": dict(sorted(updates.items())),
        "prediction_aided_row_fraction": (
            round(aided_like / localised_line_rows, 4) if localised_line_rows else None
        ),
        "rows_extent_clamped": clamped_rows,
        "measured_against": "physics: a pipette in a hand moves a few cm a frame at most and "
        "its butt sits near the hand",
    }


# --------------------------------------------------------------------------- ambiguities


def ambiguities(
    lines_rows: Sequence[Track3D],
    rest_frames: Mapping[str, set[int]],
    metrics: Mapping[str, Any],
    *,
    perpendicular_cm: float = AMBIGUITY_PERPENDICULAR_CM,
    angle_deg: float = AMBIGUITY_ANGLE_DEG,
) -> dict[str, Any]:
    """Frames where two live pipette lines sit within the perpendicular and angle gates,
    outside the frames where both are at rest; the near-duplicate pairs; the tracker's id
    switches, with the geometric-class note."""
    by_frame: dict[int, list[Track3D]] = defaultdict(list)
    classes: dict[str, str | None] = {}
    for row in lines_rows:
        if row.object_class == GEOMETRIC_CLASS and row.state in LIVE_STATES:
            if row.endpoints_cm is not None and row.direction is not None:
                by_frame[row.frame_index].append(row)
                classes[row.track_id] = row.observed_class
    pair_frames: Counter = Counter()
    pair_frames_at_rest: Counter = Counter()
    frames_ambiguous: set[int] = set()
    for frame, rows in by_frame.items():
        for a, b in combinations(rows, 2):
            angle, perpendicular, _ = line_distance(line_from_row(a), line_from_row(b))
            if angle >= angle_deg or perpendicular >= perpendicular_cm:
                continue
            key = tuple(sorted((a.track_id, b.track_id)))
            both_rest = frame in rest_frames.get(a.track_id, ()) and frame in rest_frames.get(
                b.track_id, ()
            )
            if both_rest:
                pair_frames_at_rest[key] += 1
            else:
                pair_frames[key] += 1
                frames_ambiguous.add(frame)
    pairs = [
        {
            "tracks": list(key),
            "classes": [classes.get(key[0]), classes.get(key[1])],
            "frames": n,
            "frames_at_rest": pair_frames_at_rest.get(key, 0),
        }
        for key, n in pair_frames.most_common()
    ]
    return {
        "rule": (
            f"two live pipette lines within {perpendicular_cm} cm perpendicular and "
            f"{angle_deg} deg, on frames where not both are at rest"
        ),
        "frames_with_an_ambiguity": len(frames_ambiguous),
        "pair_frames": int(sum(pair_frames.values())),
        "pair_frames_at_rest_excluded": int(sum(pair_frames_at_rest.values())),
        "near_duplicate_pairs": len(pairs),
        "pairs": pairs[:20],
        "tracker_ambiguities": metrics.get("ambiguities"),
        "tracker_duplicate_pair_frames": metrics.get("duplicate_pair_frames"),
        "id_switches": metrics.get("id_switches"),
        "id_switch_reference": metrics.get("id_switch_reference"),
        "id_switch_note": (
            "SAM3 per-view slots are the proxy identities; under the geometric class a slot of "
            "one colour associated to a track whose other views carry another colour (a "
            "colour-cross association) counts as a switch, so this number is not comparable "
            "with the point tracker's per-class count"
        ),
    }


# --------------------------------------------------------------------------- colour


def colour_summary(
    colour_rows: Sequence[Mapping[str, Any]] | None,
    lines_rows: Sequence[Track3D],
) -> dict[str, Any]:
    """Per track the vote's identity, confidence, entropy and samples; agreement with the
    plurality detector class (the three single-channel classes carry a colour, the 8-channel
    head none); the disagreeing tracks with the slots they came from."""
    by_track = rows_by_track(lines_rows, lambda r: r.object_class == GEOMETRIC_CLASS)
    slots: dict[str, Counter] = {}
    for tid, trows in by_track.items():
        counter: Counter = Counter()
        for row in trows:
            for view, slot in row.support_slots.items():
                counter[f"{view}/{slot}"] += 1
        slots[tid] = counter
    if colour_rows is None:
        return {"available": False, "note": "no tracks_colour.jsonl beside the line tracks"}
    per_track: dict[str, dict[str, Any]] = {}
    for row in colour_rows:
        if row.get("object_class") != GEOMETRIC_CLASS:
            continue
        tid = str(row["track_id"])
        entry = per_track.setdefault(
            tid,
            {
                "colour_identity": row.get("colour_identity"),
                "colour_confidence": row.get("colour_confidence"),
                "colour_entropy_bits": row.get("colour_entropy"),
                "colour_samples": row.get("colour_samples"),
                "observed_class": None,
                "rows": 0,
            },
        )
        entry["rows"] += 1
        if row.get("observed_class"):
            entry["observed_class"] = row["observed_class"]
    agree = disagree = no_colour = no_samples = 0
    disagreements: list[dict[str, Any]] = []
    confidences: list[float] = []
    entropies: list[float] = []
    for tid, entry in per_track.items():
        trows = by_track.get(tid)
        cls = plurality_class(trows, True) if trows else entry["observed_class"]
        entry["observed_class"] = cls
        expected = CLASS_COLOUR.get(cls or "")
        entry["expected_colour"] = expected
        if not entry["colour_samples"]:
            entry["agreement"] = None
            no_samples += 1
            continue
        confidences.append(float(entry["colour_confidence"] or 0.0))
        entropies.append(float(entry["colour_entropy_bits"] or 0.0))
        if expected is None:
            entry["agreement"] = None
            no_colour += 1
            continue
        entry["agreement"] = entry["colour_identity"] == expected
        if entry["agreement"]:
            agree += 1
        else:
            disagree += 1
            disagreements.append(
                {
                    "track_id": tid,
                    "observed_class": cls,
                    "colour_identity": entry["colour_identity"],
                    "colour_confidence": entry["colour_confidence"],
                    "colour_samples": entry["colour_samples"],
                    "rows": entry["rows"],
                    "slots": [f"{s} x{n}" for s, n in slots.get(tid, Counter()).most_common(4)],
                }
            )
    return {
        "available": True,
        "tracks": len(per_track),
        "tracks_with_samples": len(per_track) - no_samples,
        "tracks_without_samples": no_samples,
        "tracks_without_a_class_colour": no_colour,
        "agree": agree,
        "disagree": disagree,
        "agreement_fraction": round(agree / (agree + disagree), 4) if agree + disagree else None,
        "confidence": _percentiles(confidences),
        "entropy_bits": _percentiles(entropies),
        "disagreements": disagreements,
        "per_track": dict(sorted(per_track.items())),
        "measured_against": (
            "the plurality detector class of the same track; the 8-channel head has no colour "
            "and tracks without a ring sample have no identity"
        ),
    }


# --------------------------------------------------------------------------- lines metrics


def line_frames_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    lines = metrics.get("extensions", {}).get("lines", {})
    frames = lines.get("frames", {})
    line = int(frames.get("line") or 0)
    aided = int(frames.get("line_prediction_aided") or 0)
    fallback = int(frames.get("point_fallback") or 0)
    total = line + fallback
    return {
        "line_frames": line,
        "line_prediction_aided_frames": aided,
        "point_fallback_frames": fallback,
        "single_view_frames": frames.get("single_view"),
        "degenerate_fits": frames.get("degenerate_fits"),
        "extended_by_prior": frames.get("extended_by_prior"),
        "line_fraction": round(line / total, 4) if total else None,
        "aided_fraction_of_line_frames": round(aided / line, 4) if line else None,
        "fallback_fraction": round(fallback / total, 4) if total else None,
        # Sep 29 corrections (None on a first-run metrics file).
        "extent_clamped": frames.get("extent_clamped"),
        "step_capped": frames.get("step_capped"),
        "single_view_vetoed": frames.get("single_view_vetoed"),
        "aided_rejected": frames.get("aided_rejected"),
        "aided_extent_from_prediction": frames.get("aided_extent_from_prediction"),
        "far_rays_dropped": frames.get("far_rays_dropped"),
        "held_corrections": lines.get("held"),
        "class_veto": lines.get("class_veto"),
        "tip_resolved_fraction": lines.get("tip_resolved_fraction"),
        "tip_resolutions_by_basis": lines.get("tip_resolutions_by_basis"),
        "merged_views_by_view": lines.get("merged_views_by_view"),
        "line_births": lines.get("line_births"),
        "point_births": lines.get("point_births"),
        "tracker_loo_residual_px": lines.get("loo_residual_px"),
        "tracker_loo_angle_deg": lines.get("loo_angle_deg"),
        "tracker_length_cm": lines.get("length_cm"),
        "class_agreement": lines.get("class_agreement"),
        "prior": lines.get("prior"),
    }


# --------------------------------------------------------------------------- disposable tips


def tips_summary(
    lines_rows: Sequence[Track3D],
    metrics: Mapping[str, Any],
    prior: LinePrior,
    tip_events: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Sep 29: how often a `*_tip` box attached (per line frame and view, from the tracker's
    metrics), the `tip_attached` state by tracker state over the rows, the written segment
    length per class split by the state against the two-state prior (bare, bare plus tip),
    the tip / butt bases, and the tip events when `battle-finebio-events --tip-events` ran."""
    lines = metrics.get("extensions", {}).get("lines", {})
    tips = lines.get("tips") or {}
    by_state: dict[str, Counter] = defaultdict(Counter)
    lengths: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    attached_views_rows = 0
    line_rows = 0
    for row in lines_rows:
        if row.object_class != GEOMETRIC_CLASS or row.endpoints_cm is None:
            continue
        state_key = {True: "attached", False: "bare", None: "undecided"}[row.tip_attached]
        by_state[row.state][state_key] += 1
        if row.state in LOCALISED_STATES:
            line_rows += 1
            if row.tip_attached_views:
                attached_views_rows += 1
            cls = row.observed_class or "undecided"
            ends = np.asarray(row.endpoints_cm, dtype=np.float64)
            lengths[cls][state_key].append(float(np.linalg.norm(ends[1] - ends[0])))
    modes: dict[str, Any] = {}
    for cls in sorted(lengths):
        pipette = cls if cls in PIPETTE_CLASSES else None
        bare = prior.bare_for(pipette)
        tip_class = prior.tip_class_for(pipette)
        expected_tip = bare + prior.tip_for(tip_class)
        block: dict[str, Any] = {
            "prior_bare_cm": round(bare, 2),
            "prior_with_tip_cm": round(expected_tip, 2),
            "tip_class": tip_class,
        }
        for key in ("bare", "attached", "undecided"):
            block[f"written_length_{key}_cm"] = _percentiles(lengths[cls][key])
        modes[cls] = block
    out: dict[str, Any] = {
        "enabled": bool(tips.get("enabled")),
        "rule": tips.get("rule"),
        "line_frames": tips.get("line_frames"),
        "line_frames_with_attached_tip": tips.get("line_frames_with_attached_tip"),
        "attached_fraction": tips.get("attached_fraction"),
        "attached_by_view": tips.get("attached_by_view"),
        "attached_by_mode": tips.get("attached_by_mode"),
        "attached_by_tip_class": tips.get("attached_by_tip_class"),
        "state_transitions": tips.get("state_transitions"),
        "tip_class_votes": tips.get("tip_class_votes"),
        "tracks_ever_attached": tips.get("tracks_ever_attached"),
        "rows_with_attached_views_fraction": (
            round(attached_views_rows / line_rows, 4) if line_rows else None
        ),
        "tip_attached_by_tracker_state": {
            state: dict(counter) for state, counter in sorted(by_state.items())
        },
        "written_length_by_class_and_state": modes,
        "tip_basis_frames": lines.get("tip_basis_frames"),
        "tip_resolutions_by_basis": lines.get("tip_resolutions_by_basis"),
        "width_basis": lines.get("width_basis"),
        "prior": {
            "two_state": prior.two_state,
            "bare_length_cm": dict(sorted(prior.bare.items())),
            "tip_length_cm": dict(sorted(prior.tip.items())),
        },
        "events": None,
    }
    if tip_events:
        out["events"] = {
            "counts": tip_events.get("counts"),
            "unmatched_flips": tip_events.get("unmatched_flips"),
            "frames": [
                {k: e.get(k) for k in ("kind", "track_id", "frame_index", "target", "tip_class")}
                for e in tip_events.get("events", [])
            ],
            "volumes_from_rig": tip_events.get("volumes_from_rig"),
            "volumes_from_boxes": tip_events.get("volumes_from_boxes"),
            "volumes_skipped": tip_events.get("volumes_skipped"),
            "rule": tip_events.get("rule"),
        }
    return out


# --------------------------------------------------------------------------- rule


def held_pipette_class(baseline_metrics: Mapping[str, Any]) -> str | None:
    """The class the baseline fragments most: the held pipette of the trial."""
    per_class = baseline_metrics.get("per_class", {})
    candidates = {c: per_class[c].get("tracks_born", 0) for c in PIPETTE_CLASSES if c in per_class}
    if not candidates:
        return None
    return max(sorted(candidates), key=candidates.__getitem__)


def evaluate_rule(
    *,
    held_class: str | None,
    ids: Mapping[str, Any],
    loo: Mapping[str, Any],
    line_frames: Mapping[str, Any],
    rest: Mapping[str, Any],
    other_trial: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The plan's pre-registered rule for one trial, and for both when the other trial's
    scoreboard is given."""
    base_ids = ids["baseline"]["by_class"].get(held_class or "", {}).get("ids")
    line_ids = ids["lines"]["by_class"].get(held_class or "", {}).get("ids")
    ratio = (base_ids / line_ids) if base_ids and line_ids else None
    line_median = (line_frames.get("tracker_loo_residual_px") or {}).get("median")
    box_perp = loo["box_centre"]["perpendicular_to_axis_px"].get("median")
    box_mid = loo["box_centre"]["distance_to_axis_midpoint_px"].get("median")
    this = {
        "held_pipette_class": held_class,
        "ids_baseline": base_ids,
        "ids_lines": line_ids,
        "ids_fall_factor": round(ratio, 2) if ratio is not None else None,
        "ids_fall_at_least_5x": (ratio >= ID_FALL_FACTOR) if ratio is not None else None,
        "loo_line_median_px": line_median,
        "loo_box_centre_perpendicular_median_px": box_perp,
        "loo_box_centre_midpoint_median_px": box_mid,
        "loo_line_beats_box_centre": (
            (line_median < box_perp) if line_median is not None and box_perp is not None else None
        ),
        "loo_line_beats_box_centre_midpoint": (
            (line_median < box_mid) if line_median is not None and box_mid is not None else None
        ),
        "tip_error_median_cm": None,
        "tip_error_under_2cm": "pending human anchors",
        "flat_rest_within_rig_noise": rest.get("within_rig_noise_all_classes"),
    }
    out: dict[str, Any] = {
        "this_trial": this,
        "rule": (
            "adopt the extension if, on both trials with zero tuning: ids per held pipette fall "
            f"at least {ID_FALL_FACTOR:.0f}x; the LOO line residual median beats the box-centre "
            "LOO median (perpendicular to the held-out axis, like for like); tip error median "
            "under 2 cm on the clicked anchors. Flat-rest geometry must stay within the rig's "
            "static residual noise."
        ),
    }
    if other_trial is not None:
        other = other_trial.get("rule", {}).get("this_trial", {})
        out["other_trial"] = other
        out["both_trials"] = {
            "ids_fall_at_least_5x": _both(this, other, "ids_fall_at_least_5x"),
            "loo_line_beats_box_centre": _both(this, other, "loo_line_beats_box_centre"),
            "tip_error_under_2cm": "pending human anchors",
            "flat_rest_within_rig_noise": _both(this, other, "flat_rest_within_rig_noise"),
        }
    return out


def orientation_summary(rows: Sequence[Track3D]) -> dict[str, Any]:
    """Episodes, online flips, the tip-confidence distribution, and per-cue agreement
    with the retrofit sign. Empty when the rows are not a vote run."""
    from .finebio_orientation import CUE_NAMES, cue_agreement

    voted = [row for row in rows if row.orientation_episode is not None and row.endpoints_cm]
    if not voted:
        return {"enabled": False}
    episodes = {(row.track_id, row.orientation_episode) for row in voted}
    flips = 0
    by_track: dict[tuple[str, int], list[Track3D]] = defaultdict(list)
    for row in voted:
        by_track[(row.track_id, int(row.orientation_episode or 0))].append(row)
    for group in by_track.values():
        signs = [
            row.orientation_online_sign
            for row in sorted(group, key=lambda item: item.frame_index)
            if row.orientation_online_sign in (1, -1)
        ]
        flips += sum(1 for left, right in zip(signs, signs[1:]) if left != right)
    agreement: dict[str, Counter] = {cue: Counter() for cue in CUE_NAMES}
    for row in voted:
        for cue, verdict in cue_agreement(row.tip_votes, row.orientation_retrofit_sign).items():
            agreement[cue][verdict] += 1
    per_cue = {}
    for cue, counter in agreement.items():
        total = counter["agree"] + counter["differ"]
        per_cue[cue] = {
            "agree": counter["agree"],
            "differ": counter["differ"],
            "agreement": round(counter["agree"] / total, 4) if total else None,
        }
    return {
        "enabled": True,
        "source": (
            "tracks_oriented.jsonl"
            if any(row.orientation_retrofit_sign for row in voted)
            else "tracks.jsonl"
        ),
        "episodes": len(episodes),
        "flips_within_episode": flips,
        "tip_confidence": _percentiles(
            [float(row.tip_confidence) for row in voted if row.tip_confidence is not None]
        ),
        "cue_agreement": per_cue,
    }


def _both(a: Mapping[str, Any], b: Mapping[str, Any], key: str) -> bool | None:
    x, y = a.get(key), b.get(key)
    if x is None or y is None:
        return None
    return bool(x) and bool(y)


# --------------------------------------------------------------------------- the board


def reprojection_coverage(
    before: Sequence[FineBioObservation],
    after: Sequence[FineBioObservation],
    baseline_lines: Sequence[Track3D],
    held_class: str,
    window: tuple[int, int],
    *,
    min_score: float = 0.3,
    after_lines: Sequence[Track3D] | None = None,
) -> dict[str, Any]:
    """Compare masks on a frozen detector denominator and baseline held-frame subset.

    Counts are distinct cameras, not masks or tracks. A valid axis must pass the same
    elongation and 25 px skeleton-residual gates on both sides. These are mask coverage
    measures, not a claim that the views describe the same physical pipette.
    """
    frames = {
        row.frame_index
        for row in before
        if window[0] <= row.frame_index < window[1]
        and row.object_class == held_class
        and row.source == "detector"
        and row.pose_valid
        and (row.detector_score or 0) >= min_score
    }
    held = {
        row.frame_index
        for row in baseline_lines
        if row.state == "held" and (row.observed_class or row.object_class) == held_class
    } & frames

    def summarize(rows, track_rows):
        views = defaultdict(set)
        consistent = defaultdict(set)
        axes = {}
        for row in rows:
            if (
                row.frame_index not in frames
                or row.object_class != held_class
                or row.source == "detector"
                or not row.pose_valid
                or row.mask_axis_px is None
                or (row.mask_elongation or 0) < ELONGATION_THRESHOLD
                or row.mask_axis_residual_px is None
                or row.mask_axis_residual_px > 25
            ):
                continue
            views[row.frame_index].add(row.view)
            axes[(row.view, row.frame_index, row.slot)] = row
            if not row.provenance.get("reprojection_disagrees"):
                consistent[row.frame_index].add(row.view)

        def counts(denominator):
            n = len(denominator)
            one = sum(bool(views[frame]) for frame in denominator)
            two = sum(len(views[frame]) >= 2 for frame in denominator)
            two_consistent = sum(len(consistent[frame]) >= 2 for frame in denominator)
            return {
                "frames": n,
                "one_or_more_axis_views": one,
                "one_or_more_fraction": one / n if n else None,
                "two_or_more_axis_views": two,
                "two_or_more_fraction": two / n if n else None,
                "two_or_more_without_disagreement": two_consistent,
                "two_or_more_without_disagreement_fraction": two_consistent / n if n else None,
            }

        supported = defaultdict(int)
        supported_consistent = defaultdict(int)
        for row in track_rows:
            if (
                row.frame_index not in held
                or (row.observed_class or row.object_class) != held_class
                or row.state == "lost"
            ):
                continue
            members = [
                axes[(view, row.frame_index, slot)]
                for view, slot in row.support_slots.items()
                if (view, row.frame_index, slot) in axes
            ]
            supported[row.frame_index] = max(supported[row.frame_index], len(members))
            supported_consistent[row.frame_index] = max(
                supported_consistent[row.frame_index],
                sum(not member.provenance.get("reprojection_disagrees") for member in members),
            )
        n = len(held)
        one = sum(supported[f] >= 1 for f in held)
        two = sum(supported[f] >= 2 for f in held)
        two_consistent = sum(supported_consistent[f] >= 2 for f in held)
        return {
            "detector_frames": counts(frames),
            "baseline_held_frames": counts(held),
            "baseline_held_supported": {
                "frames": n,
                "one_or_more_axis_views": one,
                "one_or_more_fraction": one / n if n else None,
                "two_or_more_axis_views": two,
                "two_or_more_fraction": two / n if n else None,
                "two_or_more_without_disagreement": two_consistent,
                "two_or_more_without_disagreement_fraction": two_consistent / n if n else None,
            },
        }

    return {
        "class": held_class,
        "denominator": "original valid-pose detector frames at the tracker score threshold",
        "held_denominator": "same frames held in the baseline orientation-vote run",
        "supported_numerator": "maximum axis support on one live track of this class per frame",
        "axis_gate": "elongated mask, skeleton residual at most 25 px, valid pose",
        "claim_boundary": "camera coverage of class masks; identity is not verified",
        "before": summarize(before, baseline_lines),
        "after": summarize(after, after_lines if after_lines is not None else baseline_lines),
    }


def build_scoreboard(
    *,
    lines_tracks_dir: Path,
    ext_tracks_dir: Path,
    baseline_tracks_dir: Path,
    observations: Path,
    camera_config: Path,
    window: tuple[int, int],
    rig: Mapping[str, Any],
    prior: LinePrior,
    trial: str,
    stand_report: Mapping[str, Any] | None = None,
    colour_tracks: Path | None = None,
    negative_controls: Sequence[Mapping[str, Any]] | None = None,
    other_trial: Mapping[str, Any] | None = None,
    fpv_poses: Path | None = None,
    inputs: Mapping[str, Any] | None = None,
    tip_events: Mapping[str, Any] | None = None,
    reprojection_baseline_lines: Path | None = None,
    reprojection_requests: Path | None = None,
) -> dict[str, Any]:
    lines_rows = load_tracks(lines_tracks_dir)
    lines_metrics = load_metrics(lines_tracks_dir)
    ext_rows = load_tracks(ext_tracks_dir)
    ext_metrics = load_metrics(ext_tracks_dir)
    base_rows = load_tracks(baseline_tracks_dir)
    base_metrics = load_metrics(baseline_tracks_dir)
    minutes = (window[1] - window[0]) / FPS / 60.0
    ids = {
        "baseline": ids_summary(
            base_metrics,
            rows_by_track(base_rows, lambda r: is_pipette_row(r, False)),
            geometric=False,
            minutes=minutes,
        ),
        "point_on_new_rows": ids_summary(
            ext_metrics,
            rows_by_track(ext_rows, lambda r: is_pipette_row(r, False)),
            geometric=False,
            minutes=minutes,
        ),
        "lines": ids_summary(
            lines_metrics,
            rows_by_track(lines_rows, lambda r: is_pipette_row(r, True)),
            geometric=True,
            minutes=minutes,
        ),
    }
    ids["point_reproduces_baseline"] = {
        cls: ids["baseline"]["by_class"][cls]["ids"]
        == ids["point_on_new_rows"]["by_class"][cls]["ids"]
        for cls in PIPETTE_CLASSES
    }
    config = read_camera_config(camera_config)
    fixed_cams = cameras_from_config(config)
    fpv_source = resolve_fpv_source(config, fpv_poses)
    obs = pipette_observations(observations)
    params = lines_metrics.get("params", {})
    ext_params = params.get("extensions", {})
    loo = loo_and_length(
        lines_rows,
        obs,
        fixed_cams,
        fpv_source,
        prior,
        fpv_weight=float(params.get("fpv_weight", 0.5)),
        min_pair_angle_deg=float(ext_params.get("line_min_pair_angle_deg", MIN_PAIR_ANGLE_DEG)),
    )
    static_loo = (rig.get("gates", {}).get("inputs", {}) or {}).get("static_loo_median_px")
    if static_loo is None:
        static_loo = (rig.get("static_loo_px") or {}).get("median")
    rest, rest_frames = rest_geometry(
        lines_rows,
        stand_report,
        fixed_cams=fixed_cams,
        rig_static_loo_px=static_loo,
        prior=prior,
    )
    colour_rows = None
    if colour_tracks is not None and Path(colour_tracks).is_file():
        colour_rows = [
            json.loads(line)
            for line in Path(colour_tracks).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    line_frames = line_frames_summary(lines_metrics)
    held = held_pipette_class(base_metrics)
    board: dict[str, Any] = {
        "schema": SCHEMA,
        "trial": trial,
        "window": list(window),
        "inputs": dict(inputs or {}),
        "ids": ids,
        "loo": loo,
        "line_frames": line_frames,
        "rest": rest,
        "plausibility": plausibility(lines_rows, prior),
        "ambiguities": ambiguities(lines_rows, rest_frames, lines_metrics),
        "colour": colour_summary(colour_rows, lines_rows),
        "tips": tips_summary(lines_rows, lines_metrics, prior, tip_events),
        "orientation": orientation_summary(lines_rows),
        "negative_controls": [dict(nc) for nc in negative_controls] if negative_controls else None,
        "gates": params.get("gates"),
        "claim_boundary": CLAIM_BOUNDARY,
        "licence_note": LICENCE_NOTE,
    }
    observations_dir = observations if observations.is_dir() else observations.parent
    merge_path = observations_dir / "merge_summary.json"
    if reprojection_baseline_lines is not None and merge_path.is_file():
        from .finebio_reprojection import line_rows, pipette_rows

        merge = json.loads(merge_path.read_text())
        board["reprojection"] = {
            "merge": merge,
            "requests": (
                json.loads((reprojection_requests / "requests_summary.json").read_text())
                if reprojection_requests is not None
                else None
            ),
            "coverage": reprojection_coverage(
                pipette_rows(Path(merge["source"])),
                pipette_rows(observations),
                line_rows(reprojection_baseline_lines),
                held,
                window,
                min_score=float(params.get("min_detector_score", 0.3)),
                after_lines=lines_rows,
            ),
        }
    board["rule"] = evaluate_rule(
        held_class=held,
        ids=ids,
        loo=loo,
        line_frames=line_frames,
        rest=rest,
        other_trial=other_trial,
    )
    return board


# --------------------------------------------------------------------------- markdown


def scoreboard_markdown(board: Mapping[str, Any]) -> str:
    ids = board["ids"]
    loo = board["loo"]
    frames = board["line_frames"]
    rest = board["rest"]
    plaus = board["plausibility"]
    amb = board["ambiguities"]
    colour = board["colour"]
    rule = board["rule"]["this_trial"]
    lines = [
        f"# Pipettes as lines: scoreboard, {board['trial']}",
        "",
        f"Window raw frames {board['window'][0]}–{board['window'][1]}. Three runs of the same "
        "observations: the Sep 27 point tracker with the extensions (baseline), the same point "
        "tracker on the remeasured rows, and the line extension on those rows. Every row names "
        "what it is measured against; nothing is ground truth.",
        "",
        "## Ids per pipette class",
        "",
        "| class | baseline | point on new rows | lines | lifetime median / p90 (frames), lines | "
        "births a minute, lines | measured against |",
        "|---|---|---|---|---|---|---|",
    ]
    for cls in PIPETTE_CLASSES:
        b = ids["baseline"]["by_class"][cls]
        p = ids["point_on_new_rows"]["by_class"][cls]
        ln = ids["lines"]["by_class"][cls]
        life = ln["lifetime_frames"]
        lines.append(
            f"| {cls} | {b['ids']} | {p['ids']} | **{ln['ids']}** | "
            f"{_fmt(life['median'], 0)} / {_fmt(life['p90'], 0)} | "
            f"{_fmt(ln['births_per_minute'])} | the tracker's own ids; lines counted by "
            "plurality observed_class |"
        )
    geo = ids["lines"].get("geometric", {})
    lines += [
        f"| pipette (geometric) | - | - | **{geo.get('ids')}** | "
        f"{_fmt(geo.get('lifetime_frames', {}).get('median'), 0)} / "
        f"{_fmt(geo.get('lifetime_frames', {}).get('p90'), 0)} | "
        f"{_fmt(geo.get('births_per_minute'))} | all line-class tracks |",
        "",
        f"Point tracker on the remeasured rows reproduces the baseline per class: "
        f"{all(ids['point_reproduces_baseline'].values())}. Baseline lifetimes median / p90: "
        + ", ".join(
            f"{cls} {_fmt(ids['baseline']['by_class'][cls]['lifetime_frames']['median'], 0)} / "
            f"{_fmt(ids['baseline']['by_class'][cls]['lifetime_frames']['p90'], 0)}"
            for cls in PIPETTE_CLASSES
        )
        + ".",
        "",
        "## Leave-one-camera-out residual",
        "",
        "| measure | median | p90 | n | measured against |",
        "|---|---|---|---|---|",
    ]
    tl = frames.get("tracker_loo_residual_px") or {}
    ta = frames.get("tracker_loo_angle_deg") or {}
    lr = loo["line_recomputed"]
    bc = loo["box_centre"]
    lines += [
        f"| line, tracker's own (px) | **{_fmt(tl.get('median'))}** | {_fmt(tl.get('p90'))} | "
        f"{tl.get('n', '-')} | the held-out camera's own mask axis |",
        f"| line, tracker's own (deg) | {_fmt(ta.get('median'))} | {_fmt(ta.get('p90'))} | "
        f"{ta.get('n', '-')} | same |",
        f"| line, recomputed from the rows (px) | {_fmt(lr['perpendicular_px']['median'])} | "
        f"{_fmt(lr['perpendicular_px']['p90'])} | {lr['perpendicular_px']['n']} | same cells |",
        f"| box centre, perpendicular to the held-out axis (px) | "
        f"**{_fmt(bc['perpendicular_to_axis_px']['median'])}** | "
        f"{_fmt(bc['perpendicular_to_axis_px']['p90'])} | {bc['perpendicular_to_axis_px']['n']} | "
        "same cells; the like-for-like number the line has to beat |",
        f"| box centre, distance to the held-out axis midpoint (px) | "
        f"{_fmt(bc['distance_to_axis_midpoint_px']['median'])} | "
        f"{_fmt(bc['distance_to_axis_midpoint_px']['p90'])} | "
        f"{bc['distance_to_axis_midpoint_px']['n']} | same cells; the point method's error |",
        "",
        f"Frames with three or more planes: {loo['frames_with_3_planes']} of "
        f"{loo['line_fit_rows']} line-fit rows. Box-centre distance by held-out view (median px): "
        + ", ".join(f"{v} {x:.1f}" for v, x in bc["by_view_distance_median_px"].items())
        + ". Line by view: "
        + ", ".join(f"{v} {x:.1f}" for v, x in lr["by_view_median_px"].items())
        + ".",
        "",
        "## Length",
        "",
        "| class | visible median (p10–p90) cm | spread p10–p90 cm | prior cm | "
        "deviation median cm | abs deviation p90 cm | measured against |",
        "|---|---|---|---|---|---|---|",
    ]
    for cls, block in loo["length_by_class"].items():
        lines.append(
            f"| {cls} | {_pct_cell(block['visible_cm'])} | {_fmt(block['spread_p10_p90_cm'])} | "
            f"{_fmt(block['prior_cm'])} | {_fmt(block['deviation_from_prior_cm']['median'])} | "
            f"{_fmt(block['abs_deviation_from_prior_cm']['p90'])} | the per-class prior from "
            "the stand slice |"
        )
    lines += [
        "",
        "## Flat-rest geometry",
        "",
        f"Rest rule: {rest['rest_rule']}. Rig static residual noise: "
        f"{_fmt(rest['rig_static_loo_median_px'], 1)} px, about "
        f"{_fmt(rest['rig_static_noise_cm'])} cm at the resting depth.",
        "",
        "| class | runs | frames | angle to bench normal median (p10–p90) deg | stand report "
        "angle deg | end drift p90 cm (A / B) | stand report drift p90 cm | within rig noise | "
        "measured against |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for cls, block in rest["by_class"].items():
        s = block["stand_report"]
        lines.append(
            f"| {cls} | {block['runs']} | {block['frames']} | "
            f"{_pct_cell(block['angle_to_bench_normal_deg'], 1)} | "
            f"{_fmt(s['angle_to_bench_normal_median_deg'], 1)} | "
            f"{_fmt(block['end_a_drift_cm']['p90'])} / {_fmt(block['end_b_drift_cm']['p90'])} | "
            f"{_fmt(s['end_drift_p90_cm'])} | {_fmt(block['within_rig_noise'])} | the stand "
            "report's resting segments and the rig's static residual |"
        )
    lines += [
        "",
        "## Plausibility",
        "",
        "| measure | median | p90 | max | n | measured against |",
        "|---|---|---|---|---|---|",
        f"| tip speed, cm a frame | {_stats_cells(plaus['tip_speed_cm_per_frame'])} | physics |",
        f"| midpoint speed, cm a frame | {_stats_cells(plaus['midpoint_speed_cm_per_frame'])} | "
        "physics |",
        f"| butt to hand while held, cm | {_stats_cells(plaus['butt_to_hand_cm_while_held'])} | "
        "the hand track's position |",
        f"| written segment length, cm | {_stats_cells(plaus['row_segment_length_cm'])} | the "
        "length prior; over twice it is a runaway extent |",
        "",
        f"Steps over {STEP_JUMP_CM:.0f} cm: tip {_fmt(plaus['tip_steps_over_10cm_fraction'], 4)}, "
        f"midpoint {_fmt(plaus['midpoint_steps_over_10cm_fraction'], 4)} of the steps "
        f"({plaus['midpoint_steps_over_10cm']} midpoint steps, "
        f"{plaus['midpoint_big_steps_touching_a_single_view_row']} of them on a single-view "
        f"row). Rows over twice the prior: {plaus['rows_over_twice_the_prior']} "
        f"({_fmt(plaus['rows_over_twice_the_prior_fraction'], 4)}). Held rows "
        f"{plaus['held_rows']}, {plaus['held_rows_with_resolved_butt']} with a resolved butt.",
        "",
    ]
    if plaus.get("rows_by_update"):
        by_update = plaus["rows_by_update"]
        lines += [
            "Localised line rows by the update that moved them: "
            + ", ".join(f"{k} {n}" for k, n in by_update.items())
            + f" ({plaus['localised_line_rows']} rows); prediction-aided rows (aided, capped, "
            f"predicted) {_fmt(plaus['prediction_aided_row_fraction'], 4)}; rows with the "
            f"extent clamped {plaus['rows_extent_clamped']}; held rows with the tip nearer the "
            f"hand than the butt "
            f"{plaus['held_rows_with_the_tip_nearer_the_hand_than_the_butt']}.",
            "",
        ]
    lines += [
        "## Ambiguities",
        "",
        f"Rule: {amb['rule']}.",
        "",
        "| measure | value | measured against |",
        "|---|---|---|",
        f"| frames with an ambiguity | {amb['frames_with_an_ambiguity']} | the tracker's own "
        "lines |",
        f"| pair frames (at rest, excluded) | {amb['pair_frames']} "
        f"({amb['pair_frames_at_rest_excluded']}) | same |",
        f"| near-duplicate pairs | {amb['near_duplicate_pairs']} | same |",
        f"| tracker ambiguities / duplicate-pair frames | {amb['tracker_ambiguities']} / "
        f"{amb['tracker_duplicate_pair_frames']} | the tracker's re-acquisition rule |",
        f"| id switches (proxy) | {amb['id_switches']} | SAM3 slots; a colour-cross association "
        "counts as a switch under the geometric class |",
        "",
    ]
    if amb["pairs"]:
        lines += ["Near-duplicate pairs, most frames first:", ""]
        for pair in amb["pairs"][:8]:
            lines.append(
                f"- {pair['tracks'][0]} ({pair['classes'][0]}) and {pair['tracks'][1]} "
                f"({pair['classes'][1]}): {pair['frames']} frames "
                f"({pair['frames_at_rest']} at rest)"
            )
        lines.append("")
    lines += ["## Colour vote", ""]
    if not colour.get("available"):
        lines += [colour.get("note", "not available"), ""]
    else:
        lines += [
            "| measure | value | measured against |",
            "|---|---|---|",
            f"| tracks annotated / with samples | {colour['tracks']} / "
            f"{colour['tracks_with_samples']} | the proxy frames at the plunger end |",
            f"| agree / disagree with the plurality detector class | {colour['agree']} / "
            f"**{colour['disagree']}** | the detector's class, itself an appearance vote |",
            f"| tracks without a class colour (8-channel) | "
            f"{colour['tracks_without_a_class_colour']} | - |",
            f"| confidence median (p10–p90) | {_pct_cell(colour['confidence'])} | top share |",
            f"| entropy bits median (p10–p90) | {_pct_cell(colour['entropy_bits'])} | three bins |",
            "",
        ]
        if colour["disagreements"]:
            lines += ["Disagreements and the slots they came from:", ""]
            for d in colour["disagreements"]:
                lines.append(
                    f"- {d['track_id']}: detector {d['observed_class']}, colour "
                    f"{d['colour_identity']} ({_fmt(d['colour_confidence'])}, "
                    f"{d['colour_samples']} samples, {d['rows']} rows); slots "
                    f"{', '.join(d['slots'])}"
                )
            lines.append("")
    merged = frames.get("merged_views_by_view") or {}
    lines += [
        "## Line frames and merged views",
        "",
        "| measure | value | measured against |",
        "|---|---|---|",
        f"| line frames / prediction-aided / point fallback | {frames['line_frames']} / "
        f"{frames['line_prediction_aided_frames']} / {frames['point_fallback_frames']} | the "
        "tracker's own fits |",
        f"| line fraction / aided fraction of line frames / fallback fraction | "
        f"{_fmt(frames['line_fraction'], 4)} / "
        f"{_fmt(frames['aided_fraction_of_line_frames'], 4)} / "
        f"{_fmt(frames['fallback_fraction'], 4)} | same |",
        f"| single-view frames / degenerate fits / extended by the prior | "
        f"{frames['single_view_frames']} / {frames['degenerate_fits']} / "
        f"{frames['extended_by_prior']} | same |",
        f"| tip resolved fraction | {_fmt(frames['tip_resolved_fraction'], 4)} | hand coupling "
        f"({frames.get('tip_resolutions_by_basis')}) |",
        "| merged views by view | "
        + (", ".join(f"{v} {n}" for v, n in merged.items()) or "none")
        + " | width and extent against the running median and the prior |",
    ]
    if frames.get("aided_rejected") is not None:
        held_fix = frames.get("held_corrections") or {}
        veto = frames.get("class_veto") or {}
        lines += [
            f"| corrections: extent clamped / step capped / single-view vetoed / aided "
            f"rejected / far rays dropped | {frames['extent_clamped']} / "
            f"{frames['step_capped']} / {frames['single_view_vetoed']} / "
            f"{frames['aided_rejected']} / {frames['far_rays_dropped']} | the Sep 29 "
            "geometry corrections, times each acted |",
            f"| holds refused (hand far from the butt) / offsets from a localised frame | "
            f"{held_fix.get('refused_hand_far_from_butt')} / "
            f"{held_fix.get('offset_from_a_localised_frame')} | the held offset rule |",
            f"| class veto: observations refused / id splits | "
            f"{veto.get('observations_refused')} / {veto.get('splits')} | {veto.get('rule')} |",
        ]
    lines += [""]
    tips = board.get("tips") or {}
    lines += ["## Disposable tips", ""]
    if not tips.get("enabled"):
        lines += ["The tip boxes were not attached on this run (a v1 / v2 board).", ""]
    else:
        by_view = tips.get("attached_by_view") or {}
        by_mode = tips.get("attached_by_mode") or {}
        transitions = tips.get("state_transitions") or {}
        lines += [
            f"Rule: {tips.get('rule')}.",
            "",
            "| measure | value | measured against |",
            "|---|---|---|",
            f"| line frames with an attached tip box | {tips.get('line_frames_with_attached_tip')} "
            f"of {tips.get('line_frames')} ({_fmt(tips.get('attached_fraction'), 4)}) | the "
            "detector's *_tip boxes on the mask axis |",
            "| attachments by view | "
            + (", ".join(f"{v} {n}" for v, n in by_view.items()) or "none")
            + " | same |",
            "| attachments by mode (axis: the observation's own axis; track: the projected "
            "body of a box-only view) | "
            + (", ".join(f"{k} {n}" for k, n in by_mode.items()) or "none")
            + " | same |",
            f"| tip state turned on / off | {transitions.get('on')} / {transitions.get('off')} | "
            "the hysteresis |",
            f"| tip class votes cast / tracks that ever carried a tip | "
            f"{tips.get('tip_class_votes')} / {tips.get('tracks_ever_attached')} | the class "
            "vote |",
            "",
            "`tip_attached` by tracker state (rows):",
            "",
            "| tracker state | attached | bare | undecided |",
            "|---|---|---|---|",
        ]
        for state, counter in (tips.get("tip_attached_by_tracker_state") or {}).items():
            lines.append(
                f"| {state} | {counter.get('attached', 0)} | {counter.get('bare', 0)} | "
                f"{counter.get('undecided', 0)} |"
            )
        lines += [
            "",
            "Written segment length by class and tip state, against the two-state prior:",
            "",
            "| class | bare rows median cm (n) | attached rows median cm (n) | undecided rows "
            "median cm (n) | prior bare cm | prior with tip cm | measured against |",
            "|---|---|---|---|---|---|---|",
        ]
        for cls, block in (tips.get("written_length_by_class_and_state") or {}).items():
            cells = []
            for key in ("bare", "attached", "undecided"):
                b = block[f"written_length_{key}_cm"]
                cells.append(f"{_fmt(b.get('median'))} ({b.get('n', 0)})")
            lines.append(
                f"| {cls} | {' | '.join(cells)} | {_fmt(block['prior_bare_cm'])} | "
                f"{_fmt(block['prior_with_tip_cm'])} | `configs/finebio/pipettes.json` |"
            )
        basis = tips.get("tip_basis_frames") or {}
        flips = tips.get("tip_resolutions_by_basis") or {}
        lines += [
            "",
            "Tip / butt bases over the line rows: "
            + (", ".join(f"{k} {n}" for k, n in basis.items()) or "none")
            + "; decisions (changes of the flag) by basis: "
            + (", ".join(f"{k} {n}" for k, n in flips.items()) or "none")
            + f". Width basis: {(tips.get('width_basis') or {}).get('rule')}; votes cast "
            f"{(tips.get('width_basis') or {}).get('votes_cast')}.",
            "",
        ]
        events = tips.get("events")
        if events:
            counts = events.get("counts") or {}
            unmatched = events.get("unmatched_flips") or {}
            lines += [
                f"Tip events (`battle-finebio-events --tip-events`): tip_picked "
                f"**{counts.get('tip_picked')}**, tip_ejected **{counts.get('tip_ejected')}**; "
                f"flips that met no volume: {unmatched.get('tip_picked')} on, "
                f"{unmatched.get('tip_ejected')} off. Volumes from the rig: "
                + (", ".join(events.get("volumes_from_rig") or []) or "none")
                + "; from detector boxes: "
                + (", ".join(events.get("volumes_from_boxes") or []) or "none")
                + "; skipped: "
                + (
                    ", ".join(
                        f"{k} ({v})" for k, v in (events.get("volumes_skipped") or {}).items()
                    )
                    or "none"
                )
                + ".",
                "",
            ]
            for e in events.get("frames") or []:
                lines.append(
                    f"- {e['kind']} `{e['track_id']}` at frame {e['frame_index']} in "
                    f"{e['target']} ({e['tip_class']})"
                )
            if events.get("frames"):
                lines.append("")
        else:
            lines += ["Tip events: not run for this board.", ""]
    orientation = board.get("orientation") or {}
    lines += ["## Orientation vote", ""]
    if not orientation.get("enabled"):
        lines += ["Not a vote run. The tip bases above are the priority list.", ""]
    else:
        confidence = orientation.get("tip_confidence") or {}
        lines += [
            f"Episodes **{orientation.get('episodes')}**, flips within an episode "
            f"**{orientation.get('flips_within_episode')}**. Tip confidence "
            f"(sigmoid of the online log-odds) median {_fmt(confidence.get('median'))}, "
            f"p10 {_fmt(confidence.get('p10'))}, p90 {_fmt(confidence.get('p90'))} "
            f"(n {confidence.get('n', 0)}). A value near 0 or 1 is a decided end.",
            "",
            "| cue | agree | differ | agreement | measured against |",
            "|---|---|---|---|---|",
        ]
        for cue, block in (orientation.get("cue_agreement") or {}).items():
            lines.append(
                f"| {cue} | {block.get('agree')} | {block.get('differ')} | "
                f"{_fmt(block.get('agreement'), 4)} | the episode's retrofit sign |"
            )
        lines.append("")
    reprojection = board.get("reprojection")
    if reprojection:
        lines += ["## Reprojection masks", ""]
        merge = reprojection["merge"]
        counts = merge["counts"]
        lines += [
            f"Requests {merge['requests']}, returned masks {merge['returned_masks']}, "
            f"accepted {counts.get('accepted', 0)}, disagreeing {counts.get('disagrees', 0)}.",
            "",
            "| denominator | run | frames | at least one axis view | at least two axis views | "
            "two without flagged disagreement | measured against |",
            "|---|---|---|---|---|---|---|",
        ]
        coverage = reprojection["coverage"]
        for name in ("detector_frames", "baseline_held_frames", "baseline_held_supported"):
            for run in ("before", "after"):
                block = coverage[run][name]
                lines.append(
                    f"| {name} | {run} | {block['frames']} | "
                    f"{_fmt(block['one_or_more_fraction'], 4)} | "
                    f"{_fmt(block['two_or_more_fraction'], 4)} | "
                    f"{_fmt(block['two_or_more_without_disagreement_fraction'], 4)} | "
                    "original detector frames, same class and axis gates |"
                )
        lines += ["", coverage["claim_boundary"], ""]
    controls = board.get("negative_controls") or []
    lines += ["## Negative controls", ""]
    if not controls:
        lines += ["Not run for this trial (trial 1 only).", ""]
    for nc in controls:
        lines += [
            f"Frames {nc['frames'][0]}–{nc['frames'][1]}, the line tracker with the same flags. "
            "Cells = LOO cells with three or more planes, as a fraction of the normal run's.",
            "",
            "| run | LOO median px | p90 | n (cells) | ratio to normal | rises clearly | "
            "measured against |",
            "|---|---|---|---|---|---|---|",
        ]
        for name, run in nc["runs"].items():
            r = run.get("loo_residual_px") or {}
            lines.append(
                f"| {name} | {_fmt(r.get('median'))} | {_fmt(r.get('p90'))} | {r.get('n', '-')} "
                f"({_fmt(run.get('cells_fraction_of_normal'))}) | "
                f"{_fmt(run.get('ratio_to_normal'))} | {_fmt(run.get('rises_clearly'))} | "
                f"{run.get('description', '')} |"
            )
        lines += ["", nc.get("verdict", ""), ""]
    lines += [
        "## The pre-registered rule",
        "",
        "| clause | this trial | numbers |",
        "|---|---|---|",
        f"| ids per held pipette ({rule['held_pipette_class']}) fall >= 5x | "
        f"**{_fmt(rule['ids_fall_at_least_5x'])}** | {rule['ids_baseline']} -> {rule['ids_lines']} "
        f"({_fmt(rule['ids_fall_factor'])}x) |",
        f"| LOO line median < box-centre LOO median | "
        f"**{_fmt(rule['loo_line_beats_box_centre'])}** | {_fmt(rule['loo_line_median_px'])} vs "
        f"{_fmt(rule['loo_box_centre_perpendicular_median_px'])} px perpendicular "
        f"({_fmt(rule['loo_box_centre_midpoint_median_px'])} px to the midpoint) |",
        f"| tip error median < 2 cm | {rule['tip_error_under_2cm']} | - |",
        f"| flat-rest geometry within the rig's static noise | "
        f"**{_fmt(rule['flat_rest_within_rig_noise'])}** | drift p90 against "
        f"{_fmt(rest['rig_static_noise_cm'])} cm |",
        "",
    ]
    both = board["rule"].get("both_trials")
    if both:
        lines += [
            "Both trials: ids fall >= 5x "
            f"**{_fmt(both['ids_fall_at_least_5x'])}**, LOO line beats box centre "
            f"**{_fmt(both['loo_line_beats_box_centre'])}**, tip error "
            f"{both['tip_error_under_2cm']}, flat rest within noise "
            f"**{_fmt(both['flat_rest_within_rig_noise'])}**.",
            "",
        ]
    lines += [f"Claim boundary: {board['claim_boundary']}", "", board["licence_note"], ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- negative controls


def shipped_camera_config(
    config_path: Path,
    view: str,
    output: Path,
    shipped_pose: tuple[Sequence[float], Sequence[float]] | None = None,
) -> Path:
    """The clip's camera config with `view`'s pose replaced by the shipped extrinsics (from
    the shipped pose file of the recording day and the view's camera id, or the given
    ``(rvec, tvec)``), provenance `shipped`; written to `output`."""
    config = read_camera_config(Path(config_path))
    fixed = config.fixed[view]
    if shipped_pose is None:
        cam = fixed_camera(config.recording_day, fixed.camera_id, view)
        rvec, tvec = cam.rvec, cam.tvec
    else:
        rvec, tvec = shipped_pose
    replaced = fixed.model_copy(
        update={
            "provenance": "shipped",
            "rvec": tuple(float(x) for x in rvec),
            "tvec": tuple(float(x) for x in tvec),
            "marker_fit_residual_px": fixed.shipped_marker_residual_px,
        }
    )
    doc = config.model_copy(
        update={
            "fixed": {**config.fixed, view: replaced},
            "provenance": {
                **dict(config.provenance),
                "negative_control": f"{view} pose replaced by the shipped extrinsics",
            },
        }
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(doc.model_dump_json(indent=1) + "\n", encoding="utf-8")
    return output


def write_control_observations(
    observations: Path,
    output_dir: Path,
    frames: tuple[int, int],
    shift_view: str,
) -> tuple[Path, Path, dict[str, int]]:
    """The window's rows as written (`observations-window.jsonl`) and with `shift_view`'s
    rows moved one frame later (`observations-shift-<view>.jsonl`): the row of frame f is
    presented at f + 1, rows leaving the window dropped. Rows are copied as text so nothing
    else changes."""
    observations = Path(observations)
    if observations.is_dir():
        observations = observations / "observations.jsonl"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    window_path = output_dir / "observations-window.jsonl"
    shift_path = output_dir / f"observations-shift-{shift_view}.jsonl"
    lo, hi = frames
    counts = {"window_rows": 0, "shifted_rows": 0, "shift_view_rows": 0}
    with (
        observations.open(encoding="utf-8") as source,
        window_path.open("w", encoding="utf-8") as window_out,
        shift_path.open("w", encoding="utf-8") as shift_out,
    ):
        for line in source:
            if not line.strip():
                continue
            raw = json.loads(line)
            frame = int(raw["frame_index"])
            if lo <= frame <= hi:
                window_out.write(line if line.endswith("\n") else line + "\n")
                counts["window_rows"] += 1
            if raw.get("view") == shift_view:
                if lo <= frame + 1 <= hi:
                    raw["frame_index"] = frame + 1
                    shift_out.write(json.dumps(raw) + "\n")
                    counts["shift_view_rows"] += 1
                    counts["shifted_rows"] += 1
            elif lo <= frame <= hi:
                shift_out.write(line if line.endswith("\n") else line + "\n")
                counts["shifted_rows"] += 1
    return window_path, shift_path, counts


def run_negative_controls(
    *,
    observations: Path,
    camera_config: Path,
    gates: Path | None,
    tracker_flags: Sequence[str],
    output: Path,
    frames: tuple[int, int] = DEFAULT_CONTROL_FRAMES,
    shipped_view: str = "T5",
    shift_views: Sequence[str] = ("T3",),
    shipped_pose: tuple[Sequence[float], Sequence[float]] | None = None,
    fpv_poses: Path | None = None,
    run_tracker: Callable[..., Mapping[str, Any]],
) -> dict[str, Any]:
    """Line-tracker runs over a few hundred frames: as is, with `shipped_view`'s pose
    replaced by the shipped one, and with each of `shift_views`' observations one frame
    late. The LOO line residual median of each control against the normal run; a control
    rises clearly when its median is at least `NEGATIVE_CONTROL_RISE` times the normal one."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    counts: dict[str, Any] = {}
    shift_paths: dict[str, Path] = {}
    window_path: Path | None = None
    for view in shift_views:
        window_path, shift_paths[view], counts[view] = write_control_observations(
            observations, output, frames, view
        )
    if window_path is None:
        window_path, _, counts["window"] = write_control_observations(
            observations, output, frames, "none"
        )
    shipped_config = shipped_camera_config(
        camera_config, shipped_view, output / f"cameras-shipped-{shipped_view}.json", shipped_pose
    )
    runs: dict[str, dict[str, Any]] = {}
    camera_id = read_camera_config(Path(camera_config)).fixed[shipped_view].camera_id
    specs = [
        ("normal", window_path, Path(camera_config), "the run as is"),
        (
            f"shipped-{shipped_view}",
            window_path,
            shipped_config,
            f"{shipped_view} (camera {camera_id}) with the shipped pose instead of the "
            "marker-PnP one",
        ),
        *(
            (
                f"shift-{view}",
                shift_paths[view],
                Path(camera_config),
                f"{view}'s observations presented one frame late",
            )
            for view in shift_views
        ),
    ]
    for name, obs_path, cams_path, description in specs:
        metrics = run_tracker(
            obs_path, cams_path, gates, output / name, list(tracker_flags), fpv_poses
        )
        lines = metrics.get("extensions", {}).get("lines", {})
        runs[name] = {
            "description": description,
            "loo_residual_px": lines.get("loo_residual_px"),
            "loo_angle_deg": lines.get("loo_angle_deg"),
            "line_frames": (lines.get("frames") or {}).get("line"),
            "pipette_tracks": (metrics.get("per_class", {}).get(GEOMETRIC_CLASS) or {}).get(
                "tracks_born"
            ),
        }
    normal = (runs["normal"]["loo_residual_px"] or {}).get("median")
    normal_n = (runs["normal"]["loo_residual_px"] or {}).get("n") or 0
    for name, run in runs.items():
        median = (run["loo_residual_px"] or {}).get("median")
        ratio = (median / normal) if normal and median is not None else None
        run["ratio_to_normal"] = round(ratio, 3) if ratio is not None else None
        cells = (run["loo_residual_px"] or {}).get("n") or 0
        # The cells the control still fits with three planes: a pose that breaks the gate
        # can raise the median while most cells vanish, and the README should say so.
        run["cells_fraction_of_normal"] = round(cells / normal_n, 3) if normal_n else None
        run["rises_clearly"] = (
            None if name == "normal" or ratio is None else bool(ratio >= NEGATIVE_CONTROL_RISE)
        )
    controls = [n for n in runs if n != "normal"]
    clear = [n for n in controls if runs[n]["rises_clearly"]]
    unclear = [n for n in controls if not runs[n]["rises_clearly"]]
    verdict = (
        f"Both controls raise the LOO median by at least {NEGATIVE_CONTROL_RISE}x."
        if not unclear
        else (
            f"{', '.join(unclear)} does not raise the LOO median clearly (under "
            f"{NEGATIVE_CONTROL_RISE}x); {', '.join(clear) or 'neither control'} does."
        )
    )
    report = {
        "schema": SCHEMA,
        "frames": list(frames),
        "rise_factor": NEGATIVE_CONTROL_RISE,
        "row_counts": counts,
        "tracker_flags": list(tracker_flags),
        "runs": runs,
        "verdict": verdict,
        "inputs": {
            "observations": str(observations),
            "camera_config": str(camera_config),
            "gates": None if gates is None else str(gates),
            "shipped_config": str(shipped_config),
        },
    }
    (output / "negative_controls.json").write_text(
        json.dumps(report, indent=1) + "\n", encoding="utf-8"
    )
    return report
