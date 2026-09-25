"""Thin vertical slice of the FineBio 3D-tracking chain on one window (`p0-slice`).

Observations (detector boxes and SAM3 masks in raw pixels, `FineBioObservation`) -> per-frame
same-class triangulation across the fixed views (`Camera.undistort` then `dlt_triangulate` on
the projection matrices, as the preflight did) -> 3D points in board centimetres with per-view
reprojection residuals, leave-one-view-out residuals, height above the bench (``-z``) and the
reprojection into the first-person camera where its shipped pose is valid -> one Rerun
recording carrying the seven preflight cross-checks as standing entities. No identity is
claimed here: a point is one frame's agreement between views, and the top-scoring detector box
per class per view is the only association rule. This is the floor demo's shape; the tracker
(`battle.multiview_tracks`) adds identity on top of the same geometry helpers.

Two observation sets go through the same code: ``detector`` (top box per class per view at
score >= `min_detector_score`) and ``sam3`` (`sam3_decode` and `sam3_video` rows, one per
class per view, using `point_px`, the mask centroid). A class is triangulated when it is seen
in at least `min_fixed_views` fixed views, or in `fixed_views_with_fpv` fixed views plus the
fpv with a valid pose (the raised in-hand object has three fixed views on 14/61 frames).

Units: board centimetres, z into the bench. Pixels: raw video pixels (fixed 1920x1080, fpv
1920x1440). Frame indices: raw. FineBio is non-commercial research data; the recording under
``runs/`` holds video frames only when ``--video-root`` is given and is never committed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field

from .finebio_cameras import (
    POSES,
    Camera,
    cameras_from_config,
    fpv_camera_from_config,
    fpv_poses,
    marker_points,
    read_camera_config,
)
from .multiview_geometry import dlt_triangulate
from .multiview_schemas import (
    FINEBIO_FIXED_VIEWS,
    FINEBIO_FPV_VIEW,
    FineBioCameraConfig,
    FineBioObservation,
    read_jsonl,
    write_jsonl,
)
from .schemas import VersionedModel

STATIC_CLASSES: tuple[str, ...] = (
    "centrifuge",
    "pcr_machine",
    "vortex_mixer",
    "trash_can",
    "8_tube_stripes_rack",
    "micro_tube_rack",
    "magnetic_rack",
    "blue_tip_rack",
    "red_tip_rack",
    "yellow_tip_rack",
    "8_channel_tip_rack",
)
HAND_CLASSES: tuple[str, ...] = ("left_hand", "right_hand")
MOVING_CLASSES: tuple[str, ...] = ("cell_culture_plate", "blue_pipette", *HAND_CLASSES)
HANDOFF_CLASS = "cell_culture_plate"
SAM3_SOURCES = ("sam3_decode", "sam3_video")
DEFAULT_MIN_DETECTOR_SCORE = 0.3
DEFAULT_MIN_FIXED_VIEWS = 3
DEFAULT_FIXED_VIEWS_WITH_FPV = 2
# P03's association gate from the preflight (3x the static LOO median): a class whose median
# all-view residual exceeds it under the top-box-per-class rule is not one object across views.
INSTANCE_CONSISTENCY_PX = 30.0
# Room-1 bench outline at z=0 in cm, as drawn by the preflight recording.
BENCH_OUTLINE_CM = np.array(
    [[-75, -45, 0], [75, -45, 0], [75, 55, 0], [-75, 55, 0], [-75, -45, 0]], dtype=float
)

Rows = dict[int, dict[str, list[FineBioObservation]]]
FpvCameraSource = Callable[[int], Camera | None]


# --------------------------------------------------------------------------- geometry


def triangulate_pixels(
    cams: Sequence[Camera], pixels: Sequence[np.ndarray], weights: Sequence[float] | None = None
) -> np.ndarray:
    """Undistort each view's pixel, then DLT on the projection matrices. `weights` scale each
    view's two constraint rows (equal weights reproduce `dlt_triangulate` exactly)."""
    pts = np.stack([c.undistort(p).reshape(2) for c, p in zip(cams, pixels)])
    P = np.stack([c.projection for c in cams])
    if weights is None:
        return dlt_triangulate(pts[:, None, :], P)[0]
    rows = []
    for (x, y), matrix, w in zip(pts, P, weights):
        rows.append(w * (x * matrix[2] - matrix[0]))
        rows.append(w * (y * matrix[2] - matrix[1]))
    _, _, vh = np.linalg.svd(np.stack(rows), full_matrices=True)
    h = vh[-1]
    return h[:3] / h[3]


def reprojection_residuals(
    cams: dict[str, Camera], pixels: dict[str, np.ndarray], point: np.ndarray
) -> dict[str, float]:
    return {
        v: float(np.linalg.norm(cams[v].project(point)[0] - pixels[v])) for v in pixels if v in cams
    }


def leave_one_out_residuals(
    cams: dict[str, Camera], pixels: dict[str, np.ndarray], views: Sequence[str]
) -> dict[str, float]:
    """Residual of each view against the triangulation of the other views (>= 2 others)."""
    out = {}
    for v in views:
        others = [u for u in views if u != v]
        if len(others) < 2:
            continue
        point = triangulate_pixels([cams[u] for u in others], [pixels[u] for u in others])
        out[v] = float(np.linalg.norm(cams[v].project(point)[0] - pixels[v]))
    return out


def inside_box(point: np.ndarray, box: Sequence[float]) -> bool:
    return bool(box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3])


def depth_cm(cam: Camera, point: np.ndarray) -> float:
    """Distance along the optical axis; a point behind the camera has no valid projection."""
    return float((cam.R @ np.asarray(point, dtype=np.float64).reshape(3) + cam.tvec)[2])


# --------------------------------------------------------------------------- observations


def group_rows(rows: Iterable[FineBioObservation]) -> Rows:
    grouped: Rows = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row.frame_index][row.view].append(row)
    return {frame: dict(views) for frame, views in sorted(grouped.items())}


def top_per_class(
    rows: Iterable[FineBioObservation], *, source_set: str, min_detector_score: float
) -> dict[str, FineBioObservation]:
    """One observation per class for one view and frame: the top-scoring detector box, or the
    SAM3 row with the highest object score (decode rows carry none and rank by detector score)."""
    best: dict[str, FineBioObservation] = {}
    for row in rows:
        if source_set == "detector":
            if row.source != "detector" or (row.detector_score or 0.0) < min_detector_score:
                continue
            score = row.detector_score or 0.0
        else:
            if row.source not in SAM3_SOURCES:
                continue
            score = row.sam3_object_score if row.sam3_object_score is not None else -1.0
            score = score if score >= 0 else (row.detector_score or 0.0) - 2.0
        current = best.get(row.object_class)
        if current is None or score > _rank(current, source_set):
            best[row.object_class] = row
    return best


def _rank(row: FineBioObservation, source_set: str) -> float:
    if source_set == "detector":
        return row.detector_score or 0.0
    if row.sam3_object_score is not None:
        return row.sam3_object_score
    return (row.detector_score or 0.0) - 2.0


def consecutive_frames(frames: Iterable[int]) -> list[int]:
    present = set(frames)
    return sorted(f for f in present if f + 1 in present or f - 1 in present)


def parse_frames(spec: str | None, available: Iterable[int]) -> list[int]:
    """``start:count``, ``a-b``, comma list, or None for the consecutive frames present."""
    available = sorted(set(available))
    if spec is None or spec == "consecutive":
        return consecutive_frames(available)
    if spec == "all":
        return available
    if ":" in spec:
        start, count = (int(x) for x in spec.split(":"))
        return [f for f in range(start, start + count) if f in set(available)]
    if "-" in spec:
        a, b = (int(x) for x in spec.split("-"))
        return [f for f in available if a <= f <= b]
    wanted = {int(x) for x in spec.split(",")}
    return [f for f in available if f in wanted]


# --------------------------------------------------------------------------- fpv poses


def fpv_source_from_fixture_json(config: FineBioCameraConfig, path: Path) -> FpvCameraSource:
    """Per-frame fpv camera from the fixtures' ``fpv_poses.json`` (``frames`` keyed by raw
    frame with ``valid``, ``rvec``, ``tvec``)."""
    poses = json.loads(path.read_text(encoding="utf-8"))["frames"]
    K = np.array(config.fpv.K, dtype=np.float64)
    dist = np.array(config.fpv.distortion, dtype=np.float64)
    size = tuple(config.fpv.image_size)

    def source(frame: int) -> Camera | None:
        pose = poses.get(str(frame))
        if not pose or not pose["valid"]:
            return None
        return Camera("fpv", K, dist, np.array(pose["rvec"]), np.array(pose["tvec"]), size)

    return source


def fpv_source_from_shipped_poses(config: FineBioCameraConfig) -> FpvCameraSource | None:
    """Per-frame fpv camera from the shipped pose file named in the config, when on disk."""
    if not (POSES / "first_person_camera_poses" / f"{config.trial}.npz").exists():
        return None
    poses = fpv_poses(config.trial)
    return lambda frame: fpv_camera_from_config(config, frame, poses)


def resolve_fpv_source(config: FineBioCameraConfig, fpv_json: Path | None) -> FpvCameraSource:
    if fpv_json is not None and fpv_json.exists():
        return fpv_source_from_fixture_json(config, fpv_json)
    source = fpv_source_from_shipped_poses(config)
    if source is None:
        print("no fpv pose source: fpv reprojection skipped", file=sys.stderr)
        return lambda frame: None
    return source


# --------------------------------------------------------------------------- the slice


class SlicePoint(VersionedModel):
    """One frame's same-class triangulation across views (a row of points3d.jsonl)."""

    frame_index: int = Field(ge=0)
    object_class: str = Field(min_length=1)
    source_set: str = Field(pattern=r"^(detector|sam3)$")
    point_cm: tuple[float, float, float]
    height_cm: float
    views_used: tuple[str, ...] = Field(min_length=2)
    fpv_used: bool
    residual_px: dict[str, float]
    loo_px: dict[str, float] = Field(default_factory=dict)
    fpv_residual_px: float | None = Field(default=None, ge=0)
    fpv_inside_box: bool | None = None
    scores: dict[str, float] = Field(default_factory=dict)


@dataclass
class SliceSettings:
    min_detector_score: float = DEFAULT_MIN_DETECTOR_SCORE
    min_fixed_views: int = DEFAULT_MIN_FIXED_VIEWS
    fixed_views_with_fpv: int = DEFAULT_FIXED_VIEWS_WITH_FPV
    fixed_views: tuple[str, ...] = FINEBIO_FIXED_VIEWS
    source_sets: tuple[str, ...] = ("detector", "sam3")


@dataclass
class SliceResult:
    points: list[SlicePoint]
    frames: list[int]
    settings: SliceSettings
    per_view_selected: dict[tuple[int, str, str], dict[str, FineBioObservation]] = field(
        default_factory=dict
    )

    def by_class(self, source_set: str) -> dict[str, list[SlicePoint]]:
        out: dict[str, list[SlicePoint]] = defaultdict(list)
        for p in self.points:
            if p.source_set == source_set:
                out[p.object_class].append(p)
        return dict(out)

    def median_point(self, object_class: str, source_set: str = "detector") -> np.ndarray | None:
        pts = [
            p.point_cm
            for p in self.points
            if p.object_class == object_class and p.source_set == source_set
        ]
        return np.median(np.array(pts), axis=0) if pts else None


def slice_frame(
    frame: int,
    per_view: dict[str, list[FineBioObservation]],
    cams: dict[str, Camera],
    fpv_cam: Camera | None,
    settings: SliceSettings,
    source_set: str,
) -> tuple[list[SlicePoint], dict[str, dict[str, FineBioObservation]]]:
    """Triangulate every class of one frame for one observation set. Returns the points and
    the per-view selected observation per class (what the recording draws)."""
    selected = {
        view: top_per_class(
            rows, source_set=source_set, min_detector_score=settings.min_detector_score
        )
        for view, rows in per_view.items()
    }
    classes = sorted({cls for chosen in selected.values() for cls in chosen})
    points: list[SlicePoint] = []
    for cls in classes:
        fixed = [v for v in settings.fixed_views if cls in selected.get(v, {})]
        fpv_obs = selected.get(FINEBIO_FPV_VIEW, {}).get(cls)
        fpv_ok = fpv_cam is not None and fpv_obs is not None and fpv_obs.pose_valid
        use_fpv = False
        if len(fixed) >= settings.min_fixed_views:
            views = fixed
        elif len(fixed) >= settings.fixed_views_with_fpv and fpv_ok:
            views = [*fixed, FINEBIO_FPV_VIEW]
            use_fpv = True
        else:
            continue
        all_cams = dict(cams)
        if fpv_cam is not None:
            all_cams[FINEBIO_FPV_VIEW] = fpv_cam
        pixels = {v: np.array(selected[v][cls].point_px) for v in views}
        point = triangulate_pixels([all_cams[v] for v in views], [pixels[v] for v in views])
        if not np.all(np.isfinite(point)):
            continue
        residual = reprojection_residuals(all_cams, pixels, point)
        loo = leave_one_out_residuals(all_cams, pixels, views)
        fpv_residual, fpv_inside = None, None
        if fpv_ok and not use_fpv and depth_cm(fpv_cam, point) > 0:
            projected = fpv_cam.project(point)[0]
            fpv_residual = float(np.linalg.norm(projected - np.array(fpv_obs.point_px)))
            box = fpv_obs.box_xyxy_px or fpv_obs.mask_bbox_px
            fpv_inside = inside_box(projected, box) if box else None
        points.append(
            SlicePoint(
                frame_index=frame,
                object_class=cls,
                source_set=source_set,
                point_cm=tuple(round(float(x), 3) for x in point),
                height_cm=round(float(-point[2]), 3),
                views_used=tuple(views),
                fpv_used=use_fpv,
                residual_px={v: round(r, 2) for v, r in residual.items()},
                loo_px={v: round(r, 2) for v, r in loo.items()},
                fpv_residual_px=None if fpv_residual is None else round(fpv_residual, 2),
                fpv_inside_box=fpv_inside,
                scores={
                    v: round(_rank(selected[v][cls], source_set), 4)
                    for v in views
                    if _rank(selected[v][cls], source_set) >= 0
                },
            )
        )
    return points, selected


def run_slice(
    rows: Iterable[FineBioObservation],
    cams: dict[str, Camera],
    fpv_source: FpvCameraSource,
    frames: Sequence[int] | None = None,
    settings: SliceSettings | None = None,
) -> SliceResult:
    settings = settings or SliceSettings()
    grouped = group_rows(rows)
    frame_list = list(frames) if frames is not None else consecutive_frames(grouped)
    result = SliceResult(points=[], frames=frame_list, settings=settings)
    for frame in frame_list:
        per_view = grouped.get(frame, {})
        if not per_view:
            continue
        fpv_cam = fpv_source(frame)
        for source_set in settings.source_sets:
            points, selected = slice_frame(frame, per_view, cams, fpv_cam, settings, source_set)
            result.points.extend(points)
            for view, chosen in selected.items():
                if chosen:
                    result.per_view_selected[(frame, view, source_set)] = chosen
    return result


# --------------------------------------------------------------------------- summary


def _stats(values: Sequence[float]) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "median": round(float(np.median(arr)), 2),
        "p90": round(float(np.percentile(arr, 90)), 2),
        "max": round(float(arr.max()), 2),
    }


def summarise(result: SliceResult, rig_reference: dict[str, Any] | None) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "frames": {"count": len(result.frames), "first": None, "last": None},
        "settings": {
            "min_detector_score": result.settings.min_detector_score,
            "min_fixed_views": result.settings.min_fixed_views,
            "fixed_views_with_fpv": result.settings.fixed_views_with_fpv,
        },
        "classes": {},
    }
    if result.frames:
        summary["frames"].update(first=result.frames[0], last=result.frames[-1])
    for source_set in result.settings.source_sets:
        per_class: dict[str, Any] = {}
        for cls, pts in sorted(result.by_class(source_set).items()):
            residuals = [r for p in pts for r in p.residual_px.values()]
            loo = [r for p in pts for r in p.loo_px.values()]
            per_view_loo: dict[str, list[float]] = defaultdict(list)
            for p in pts:
                for v, r in p.loo_px.items():
                    per_view_loo[v].append(r)
            fpv = [p.fpv_residual_px for p in pts if p.fpv_residual_px is not None]
            inside = [p.fpv_inside_box for p in pts if p.fpv_inside_box is not None]
            entry: dict[str, Any] = {
                "frames_triangulated": len(pts),
                "one_object_under_top_box_rule": bool(
                    np.median(residuals) <= INSTANCE_CONSISTENCY_PX
                ),
                "frames_with_fpv_in_triangulation": sum(p.fpv_used for p in pts),
                "views_used_median": float(np.median([len(p.views_used) for p in pts])),
                "residual_px": _stats(residuals),
                "loo_px": _stats(loo),
                "loo_px_per_view": {v: _stats(r) for v, r in sorted(per_view_loo.items())},
                "height_cm_median": round(float(np.median([p.height_cm for p in pts])), 2),
                "point_cm_median": [
                    round(float(x), 2) for x in np.median([p.point_cm for p in pts], axis=0)
                ],
                "fpv_residual_px": _stats(fpv),
            }
            if inside:
                entry["fpv_inside_box_fraction"] = round(float(np.mean(inside)), 3)
            per_class[cls] = entry
        summary["classes"][source_set] = per_class
    if rig_reference and "static" in rig_reference:
        comparison = {}
        detector_classes = summary["classes"].get("detector", {})
        for static in rig_reference["static"]:
            cls = static["class"]
            if cls not in detector_classes:
                comparison[cls] = {"frames_triangulated": 0}
                continue
            ref = np.array(static["point_cm"], dtype=np.float64)
            pts = np.array(
                [
                    p.point_cm
                    for p in result.points
                    if p.object_class == cls and p.source_set == "detector"
                ]
            )
            per_frame = np.linalg.norm(pts - ref, axis=1)
            comparison[cls] = {
                "frames_triangulated": int(len(pts)),
                "median_point_to_reference_cm": round(
                    float(np.linalg.norm(np.median(pts, axis=0) - ref)), 2
                ),
                "per_frame_distance_cm": _stats(per_frame),
                "reference_point_cm": static["point_cm"],
                "reference_height_cm": round(float(static["height_cm"]), 2),
                "slice_height_cm_median": detector_classes[cls]["height_cm_median"],
            }
        distances = [
            c["median_point_to_reference_cm"]
            for c in comparison.values()
            if "median_point_to_reference_cm" in c
        ]
        summary["static_vs_preflight"] = {
            "per_class": comparison,
            "median_point_to_reference_cm": _stats(distances),
            "note": (
                "the preflight triangulated the median box centre over 78 frames at score >= 0.5;"
                " the slice triangulates every frame at score >= "
                f"{result.settings.min_detector_score} and compares the window median"
            ),
        }
    return summary


def summary_markdown(summary: dict[str, Any], trial: str, extras: Sequence[str] = ()) -> str:
    lines = [
        f"# Thin slice, {trial}",
        "",
        f"Frames {summary['frames'].get('first')}..{summary['frames'].get('last')} "
        f"({summary['frames']['count']} frames). Same-class triangulation per frame across the "
        f"fixed views (>= {summary['settings']['min_fixed_views']} fixed views, or "
        f">= {summary['settings']['fixed_views_with_fpv']} fixed views plus the fpv with a valid "
        f"pose), detector boxes at score >= {summary['settings']['min_detector_score']}. "
        "Residuals in raw pixels (1920 wide); heights in board centimetres above the bench.",
        "",
    ]
    for source_set, per_class in summary["classes"].items():
        lines += [
            f"## Observation set `{source_set}`",
            "",
            "| class | frames | views (median) | residual median / p90 px | LOO median / p90 px "
            "| height cm | fpv residual median / p90 px (n, inside) | one object |",
            "|---|---|---|---|---|---|---|---|",
        ]
        inconsistent = []
        for cls, e in per_class.items():
            res, loo, fpv = e["residual_px"], e["loo_px"], e["fpv_residual_px"]
            fpv_cell = (
                f"{fpv['median']} / {fpv['p90']} ({fpv['n']}, "
                f"{e.get('fpv_inside_box_fraction', 'n/a')})"
                if fpv["n"]
                else "n/a"
            )
            loo_cell = f"{loo['median']} / {loo['p90']}" if loo["n"] else "n/a"
            one = e["one_object_under_top_box_rule"]
            if not one:
                inconsistent.append(cls)
            lines.append(
                f"| {cls} | {e['frames_triangulated']} | {e['views_used_median']:.0f} | "
                f"{res['median']} / {res['p90']} | {loo_cell} | {e['height_cm_median']} | "
                f"{fpv_cell} | {'yes' if one else 'no'} |"
            )
        lines.append("")
        if inconsistent:
            lines += [
                f"`one object` = median all-view residual within {INSTANCE_CONSISTENCY_PX:.0f} px "
                "(P03's association gate). The classes that fail it ("
                + ", ".join(inconsistent)
                + ") "
                "have several instances on the bench or a different instance as the top box in "
                "each view (the SAM3 `50ml_tube` slots were seeded per view from the top box, so "
                "the views track different tubes); their points are not one object and are the "
                "case the tracker's pairwise-assignment birth exists for.",
                "",
            ]
    if "static_vs_preflight" in summary:
        block = summary["static_vs_preflight"]
        lines += [
            "## Static classes against the preflight (`rig_reference.json`)",
            "",
            "| class | frames | window-median point to reference cm | per-frame distance "
            "median / p90 cm | height cm slice / reference |",
            "|---|---|---|---|---|",
        ]
        for cls, c in block["per_class"].items():
            if "median_point_to_reference_cm" not in c:
                lines.append(f"| {cls} | 0 | not triangulated | | |")
                continue
            d = c["per_frame_distance_cm"]
            lines.append(
                f"| {cls} | {c['frames_triangulated']} | {c['median_point_to_reference_cm']} | "
                f"{d['median']} / {d['p90']} | {c['slice_height_cm_median']} / "
                f"{c['reference_height_cm']} |"
            )
        agg = block["median_point_to_reference_cm"]
        lines += [
            "",
            f"Over {agg.get('n', 0)} static classes the window-median point sits "
            f"{agg.get('median', 'n/a')} cm (median) / {agg.get('p90', 'n/a')} cm (p90) from the "
            "preflight's point. " + block["note"] + ".",
            "",
        ]
    lines += list(extras)
    return "\n".join(lines) + "\n"


MISSING_FOR_FLOOR = (
    "## What the floor demo still needs",
    "",
    "- SAM3 per-frame masks over the whole window in every view (arm b, per-frame box-prompted "
    "decode); here the `sam3` set has masks in fpv/T2/T4/T5 only, from the preflight's "
    "video-memory run, and the mask centroid is the mask-bbox centre (the preflight kept no "
    "mask pixels).",
    "- Identity: a point here is one frame's agreement between views under the top-box-per-"
    "class rule; nothing links frames, nothing survives an occlusion, and two instances of one "
    "class collapse onto the top-scoring box. `battle-multiview-tracks` owns that.",
    "- Detections on every frame of the trial window (the fixtures cover 78 frames), the "
    "per-trial camera solve and rig gates from lane B, and the trial-1 window itself.",
    "",
)


# --------------------------------------------------------------------------- rerun


def class_colour(name: str) -> list[int]:
    """Stable, saturated colour per class name."""
    h = 0
    for ch in name:
        h = (h * 131 + ord(ch)) % 2_147_483_647
    rng = np.random.default_rng(h)
    hue = rng.random()
    i = int(hue * 6)
    f = hue * 6 - i
    p, q, t = 0.25, 1 - 0.75 * f, 0.25 + 0.75 * f
    rgb = [(1, t, p), (q, 1, p), (p, 1, t), (p, q, 1), (t, p, 1), (1, p, q)][i % 6]
    return [int(255 * c) for c in rgb]


@dataclass
class RerunInputs:
    trial: str
    cams: dict[str, Camera]
    fpv_source: FpvCameraSource
    result: SliceResult
    summary: dict[str, Any]
    rig_reference: dict[str, Any] | None
    markers: np.ndarray | None
    video_root: Path | None
    image_every: int
    handoff_class: str = HANDOFF_CLASS


def _video_path(root: Path, trial: str, view: str) -> Path:
    if view == FINEBIO_FPV_VIEW:
        return root / "finebio_videos_fpv_test/finebio_videos" / f"{trial}.mp4"
    return root / "finebio_videos_tpv_test/finebio_videos" / f"{trial}_{view}.mp4"


def _clock_scan_text(rig_reference: dict[str, Any] | None) -> str:
    if not rig_reference or "clock" not in rig_reference:
        return (
            "# Clock scan\n\nNot computed for this window; the preflight's scan "
            "(`rig_reference.json`) is only available for P03_01_01."
        )
    lines = [
        "# Clock scan (preflight, P03_01_01)",
        "",
        "Best offset in frames of each fixed view against the others' triangulation, "
        "-15..+15; residuals in px. Flat classes (plate, left hand) are uninformative; the "
        "moving right hand puts T3/T4 at +1 and T5 at -1: synchronised to +/-1 frame.",
        "",
        "| view | class | best offset | residual at best | residual at 0 |",
        "|---|---|---|---|---|",
    ]
    for view, classes in rig_reference["clock"].items():
        for cls, entry in classes.items():
            lines.append(
                f"| {view} | {cls} | {entry['best_offset']:+d} | "
                f"{entry['residual_px_at_best']:.1f} | {entry['residual_px_at_0']:.1f} |"
            )
    return "\n".join(lines)


HAND_COLOURS: tuple[tuple[str, list[int]], ...] = (
    ("left_hand", [255, 128, 0]),
    ("right_hand", [0, 128, 255]),
)


def log_world_static(cams: dict[str, Camera], markers: np.ndarray | None) -> None:
    """The world frame (z down), bench outline, board origin, the day's markers, one static
    frustum per fixed camera, and cross-check 1 for the fixed views: the markers projected
    through each camera's pose (`world/<view>/markers_projected`)."""
    import rerun as rr

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_DOWN, static=True)
    rr.log(
        "world/bench",
        rr.LineStrips3D([BENCH_OUTLINE_CM], colors=[[120, 120, 120]], radii=0.3),
        static=True,
    )
    rr.log(
        "world/board_origin",
        rr.Points3D([[0, 0, 0]], colors=[[255, 255, 255]], radii=0.8, labels=["board origin"]),
        static=True,
    )
    if markers is not None:
        rr.log(
            "world/markers",
            rr.LineStrips3D(
                [np.vstack([m, m[:1]]) for m in markers],
                colors=[[255, 60, 60]],
                radii=0.25,
                labels=[f"marker {i}" for i in range(len(markers))],
            ),
            static=True,
        )
    for v, cam in cams.items():
        log_camera_frustum(f"world/{v}", cam, image_plane_distance=15.0, static=True)
        if markers is not None:
            log_markers_projected(f"world/{v}/markers_projected", cam, markers, static=True)


def log_camera_frustum(
    entity: str, cam: Camera, *, image_plane_distance: float, static: bool = False
) -> None:
    """`Transform3D` + `Pinhole` of one camera at `entity` (a fixed view static, the fpv per
    frame)."""
    import rerun as rr

    rr.log(entity, rr.Transform3D(translation=cam.centre, mat3x3=cam.R.T), static=static)
    rr.log(
        entity,
        rr.Pinhole(
            image_from_camera=cam.K,
            resolution=list(cam.size),
            camera_xyz=rr.ViewCoordinates.RDF,
            image_plane_distance=image_plane_distance,
        ),
        static=static,
    )


def log_markers_projected(
    entity: str,
    cam: Camera,
    markers: np.ndarray,
    *,
    colour: list[int] | None = None,
    static: bool = False,
) -> None:
    """Cross-check 1: the day's marker corners projected through a camera pose."""
    import rerun as rr

    proj = cam.project(markers.reshape(-1, 3)).reshape(-1, 4, 2)
    rr.log(
        entity,
        rr.LineStrips2D(
            [np.vstack([m, m[:1]]) for m in proj],
            colors=[colour or [60, 255, 60]],
            radii=1.5,
            labels=[f"m{i}" for i in range(len(proj))],
        ),
        static=static,
    )


def log_static_objects(
    points: Sequence[np.ndarray],
    labels: Sequence[str],
    cams: dict[str, Camera],
    reference: dict[str, Any] | None = None,
) -> None:
    """Cross-check 3: the static objects' triangulated points (`world/static_objects`) and
    their reprojection into every fixed view, beside the preflight's reference points."""
    import rerun as rr

    if points:
        rr.log(
            "world/static_objects",
            rr.Points3D(np.array(points), colors=[[80, 220, 80]], radii=1.0, labels=list(labels)),
            static=True,
        )
        for v, cam in cams.items():
            rr.log(
                f"world/{v}/static_reprojected",
                rr.Points2D(
                    cam.project(np.array(points)),
                    colors=[[80, 220, 80]],
                    radii=4,
                    labels=list(labels),
                ),
                static=True,
            )
    if reference and "static" in reference:
        ref = reference["static"]
        rr.log(
            "world/static_reference",
            rr.Points3D(
                np.array([r["point_cm"] for r in ref]),
                colors=[[220, 80, 220]],
                radii=0.7,
                labels=[f"{r['class']} (preflight)" for r in ref],
            ),
            static=True,
        )


def log_clock_scan(text: str) -> None:
    """Cross-check 5: the clock scan as a markdown document at `checks/clock_scan`."""
    import rerun as rr

    rr.log(
        "checks/clock_scan", rr.TextDocument(text, media_type=rr.MediaType.MARKDOWN), static=True
    )


def log_fpv_frame(
    fpv_cam: Camera,
    cams: dict[str, Camera],
    markers: np.ndarray | None,
    trail: list[np.ndarray],
) -> None:
    """Per frame: the fpv frustum, its trail (appended in place), the markers through its
    pose (cross-check 1) and its camera centre in every fixed view (cross-check 2)."""
    import rerun as rr

    log_camera_frustum("world/fpv", fpv_cam, image_plane_distance=10.0)
    trail.append(fpv_cam.centre)
    rr.log(
        "world/fpv_trail",
        rr.LineStrips3D([np.array(trail)], colors=[[255, 200, 0]], radii=0.2),
    )
    if markers is not None:
        log_markers_projected("world/fpv/markers_projected", fpv_cam, markers)
    for v, cam in cams.items():
        rr.log(
            f"world/{v}/fpv_camera_centre",
            rr.Points2D(
                cam.project(fpv_cam.centre),
                colors=[[255, 255, 0]],
                radii=8,
                labels=["fpv camera"],
            ),
        )


def clear_fpv_frame(cams: dict[str, Camera]) -> None:
    """A frame without a valid fpv pose clears the per-frame fpv entities."""
    import rerun as rr

    rr.log("world/fpv", rr.Clear(recursive=False))
    rr.log("world/fpv/markers_projected", rr.Clear(recursive=False))
    for v in cams:
        rr.log(f"world/{v}/fpv_camera_centre", rr.Clear(recursive=False))


def log_hand_probe(
    cls: str, point: Sequence[float], colour: list[int], cams: dict[str, Camera]
) -> None:
    """Cross-check 4: a hand triangulated as a probe, in 3D and reprojected into every view."""
    import rerun as rr

    rr.log(f"world/{cls}", rr.Points3D([point], colors=[colour], radii=1.5, labels=[cls]))
    for v, cam in cams.items():
        rr.log(
            f"world/{v}/{cls}_triangulated",
            rr.Points2D(cam.project(np.array(point)), colors=[colour], radii=6, labels=[cls]),
        )


def clear_hand_probe(cls: str, cams: dict[str, Camera]) -> None:
    import rerun as rr

    rr.log(f"world/{cls}", rr.Clear(recursive=False))
    for v in cams:
        rr.log(f"world/{v}/{cls}_triangulated", rr.Clear(recursive=False))


def log_recording(inputs: RerunInputs, path: Path) -> Path:
    """One `.rrd`: the world (bench, markers, frusta, fpv trail, per-frame points, static
    reference), the seven cross-checks as entities, six camera tiles, time series. No viewer."""
    import cv2
    import rerun as rr
    import rerun.blueprint as rrb

    from .rerun_logging import init_and_save

    views = (*FINEBIO_FIXED_VIEWS, FINEBIO_FPV_VIEW)
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(origin="world", name="Rig (cm, z down)"),
                rrb.Grid(
                    *[rrb.Spatial2DView(origin=f"world/{v}", name=v) for v in views],
                    grid_columns=3,
                ),
                column_shares=[1, 2],
            ),
            rrb.Horizontal(
                rrb.TimeSeriesView(origin="checks/loo", name="LOO residual px (detector set)"),
                rrb.TimeSeriesView(
                    origin="checks/handoff",
                    name=f"fixed -> fpv hand-off px ({inputs.handoff_class})",
                ),
                rrb.TimeSeriesView(origin="checks/support", name="fixed views per class"),
                rrb.Tabs(
                    rrb.TextDocumentView(origin="checks/clock_scan", name="clock scan"),
                    rrb.TextDocumentView(origin="checks/summary", name="summary"),
                ),
            ),
            row_shares=[3, 1],
        ),
        collapse_panels=True,
    )
    init_and_save(f"finebio-slice-{inputs.trial}", path, default_blueprint=blueprint)
    log_world_static(inputs.cams, inputs.markers)
    # Cross-check 3: static objects, the slice's window medians beside the preflight's points.
    static_pts, static_labels = [], []
    for cls in STATIC_CLASSES:
        point = inputs.result.median_point(cls)
        if point is not None:
            static_pts.append(point)
            static_labels.append(cls)
    log_static_objects(static_pts, static_labels, inputs.cams, inputs.rig_reference)
    log_clock_scan(_clock_scan_text(inputs.rig_reference))
    rr.log(
        "checks/summary",
        rr.TextDocument(
            summary_markdown(inputs.summary, inputs.trial), media_type=rr.MediaType.MARKDOWN
        ),
        static=True,
    )

    caps = {}
    if inputs.video_root is not None:
        for v in views:
            video = _video_path(inputs.video_root, inputs.trial, v)
            if video.exists():
                caps[v] = cv2.VideoCapture(str(video))
    by_frame: dict[int, list[SlicePoint]] = defaultdict(list)
    for p in inputs.result.points:
        by_frame[p.frame_index].append(p)
    fpv_trail: list[np.ndarray] = []
    hands_seen: set[tuple[str, str]] = set()
    logged_images = 0
    for index, frame in enumerate(inputs.result.frames):
        rr.set_time("frame", sequence=frame)
        rr.set_time("source_time", duration=frame * 1001 / 30000)
        fpv_cam = inputs.fpv_source(frame)
        all_cams = dict(inputs.cams)
        if fpv_cam is not None:
            all_cams[FINEBIO_FPV_VIEW] = fpv_cam
            log_fpv_frame(fpv_cam, inputs.cams, inputs.markers, fpv_trail)
        else:
            clear_fpv_frame(inputs.cams)
        if caps and index % inputs.image_every == 0:
            for v, cap in caps.items():
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, img = cap.read()
                if not ok:
                    continue
                ok_enc, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok_enc:
                    rr.log(
                        f"world/{v}/image",
                        rr.EncodedImage(contents=jpg.tobytes(), media_type="image/jpeg"),
                    )
                    logged_images += 1
        points = by_frame.get(frame, [])
        for source_set in inputs.result.settings.source_sets:
            pts = [p for p in points if p.source_set == source_set]
            entity = f"world/points3d/{source_set}"
            if not pts:
                rr.log(entity, rr.Clear(recursive=False))
            else:
                rr.log(
                    entity,
                    rr.Points3D(
                        np.array([p.point_cm for p in pts]),
                        colors=[class_colour(p.object_class) for p in pts],
                        radii=1.2 if source_set == "detector" else 0.8,
                        labels=[
                            p.object_class
                            if source_set == "detector"
                            else f"{p.object_class} (sam3)"
                            for p in pts
                        ],
                    ),
                )
            for v, cam in all_cams.items():
                entity_2d = f"world/{v}/points_reprojected_{source_set}"
                if not pts:
                    rr.log(entity_2d, rr.Clear(recursive=False))
                    continue
                rr.log(
                    entity_2d,
                    rr.Points2D(
                        cam.project(np.array([p.point_cm for p in pts])),
                        colors=[class_colour(p.object_class) for p in pts],
                        radii=5 if source_set == "detector" else 3.5,
                        labels=[p.object_class for p in pts],
                    ),
                )
            for v in views:
                chosen = inputs.result.per_view_selected.get((frame, v, source_set))
                entity_boxes = (
                    f"world/{v}/{'detector' if source_set == 'detector' else 'sam3_masks'}"
                )
                if not chosen:
                    rr.log(entity_boxes, rr.Clear(recursive=False))
                    continue
                boxes, labels, colours = [], [], []
                for cls, obs in sorted(chosen.items()):
                    box = obs.box_xyxy_px if source_set == "detector" else obs.mask_bbox_px
                    if box is None:
                        continue
                    boxes.append(box)
                    score = _rank(obs, source_set)
                    labels.append(f"{cls} {score:.2f}" if score >= 0 else cls)
                    colours.append(class_colour(cls))
                rr.log(
                    entity_boxes,
                    rr.Boxes2D(
                        array=np.array(boxes),
                        array_format=rr.Box2DFormat.XYXY,
                        labels=labels,
                        colors=colours,
                    ),
                )
        # Cross-check 4: hands as probes (detector set).
        for cls, col in HAND_COLOURS:
            hand = next(
                (p for p in points if p.object_class == cls and p.source_set == "detector"), None
            )
            if hand is None:
                if (cls, "3d") in hands_seen:
                    clear_hand_probe(cls, inputs.cams)
                continue
            hands_seen.add((cls, "3d"))
            log_hand_probe(cls, hand.point_cm, col, inputs.cams)
        # Cross-checks 6 and 7: LOO series per class and view, hand-off residual for the plate.
        for p in points:
            if p.source_set != "detector":
                continue
            for v, r in p.loo_px.items():
                rr.log(f"checks/loo/{p.object_class}/{v}", rr.Scalars(r))
            rr.log(
                f"checks/support/{p.object_class}",
                rr.Scalars(len([v for v in p.views_used if v != FINEBIO_FPV_VIEW])),
            )
            if p.object_class == inputs.handoff_class and p.fpv_residual_px is not None:
                rr.log(f"checks/handoff/{p.object_class}", rr.Scalars(p.fpv_residual_px))
                rr.log(
                    f"checks/handoff/{p.object_class}_inside",
                    rr.Scalars(1.0 if p.fpv_inside_box else 0.0),
                )
    for cap in caps.values():
        cap.release()
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB, {logged_images} frame images)")
    return path


# --------------------------------------------------------------------------- cli


def load_inputs(
    args: argparse.Namespace,
) -> tuple[list[FineBioObservation], FineBioCameraConfig, FpvCameraSource, dict | None]:
    if args.fixtures is not None:
        root = args.fixtures
        observations = root / "observations.jsonl"
        cameras = root / "cameras.json"
        fpv_json = root / "fpv_poses.json"
        reference_path = root / "rig_reference.json"
    else:
        if args.observations is None or args.cameras is None:
            raise SystemExit(
                "give --fixtures <dir> or both --observations <jsonl> and --cameras <json>"
            )
        observations, cameras, fpv_json = args.observations, args.cameras, args.fpv_poses
        reference_path = args.rig_reference
    config = read_camera_config(cameras)
    rows = list(read_jsonl(observations, FineBioObservation))
    fpv_source = resolve_fpv_source(config, fpv_json)
    reference = (
        json.loads(reference_path.read_text(encoding="utf-8"))
        if reference_path is not None and reference_path.exists()
        else None
    )
    return rows, config, fpv_source, reference


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=None,
        help="fixture directory: observations.jsonl, cameras.json, fpv_poses.json, "
        "rig_reference.json",
    )
    parser.add_argument("--observations", type=Path, default=None, help="FineBioObservation JSONL")
    parser.add_argument("--cameras", type=Path, default=None, help="FineBioCameraConfig JSON")
    parser.add_argument(
        "--fpv-poses",
        type=Path,
        default=None,
        help="fixture-format fpv_poses.json; default: the shipped pose file named in the "
        "config when on disk",
    )
    parser.add_argument(
        "--rig-reference",
        type=Path,
        default=None,
        help="rig_reference.json to compare static points against",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--frames",
        default=None,
        help="start:count, a-b, comma list, 'all'; default: the consecutive frames present",
    )
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_DETECTOR_SCORE)
    parser.add_argument("--min-fixed-views", type=int, default=DEFAULT_MIN_FIXED_VIEWS)
    parser.add_argument("--fixed-views-with-fpv", type=int, default=DEFAULT_FIXED_VIEWS_WITH_FPV)
    parser.add_argument(
        "--video-root",
        type=Path,
        default=None,
        help="data/raw/finebio; logs JPEG frames every --image-every frames into the "
        "recording (never committed)",
    )
    parser.add_argument("--image-every", type=int, default=10)
    parser.add_argument("--no-rerun", action="store_true")
    args = parser.parse_args(argv)

    rows, config, fpv_source, reference = load_inputs(args)
    cams = cameras_from_config(config)
    detector_frames = {r.frame_index for r in rows if r.source == "detector"}
    frames = parse_frames(args.frames, detector_frames or {r.frame_index for r in rows})
    settings = SliceSettings(
        min_detector_score=args.min_score,
        min_fixed_views=args.min_fixed_views,
        fixed_views_with_fpv=args.fixed_views_with_fpv,
        fixed_views=tuple(config.fixed),
    )
    result = run_slice(rows, cams, fpv_source, frames, settings)
    args.output.mkdir(parents=True, exist_ok=True)
    count = write_jsonl(result.points, args.output / "points3d.jsonl")
    summary = summarise(result, reference)
    summary["trial"] = config.trial
    summary["points_written"] = count
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=1) + "\n", encoding="utf-8"
    )
    markdown = summary_markdown(summary, config.trial, MISSING_FOR_FLOOR)
    (args.output / "summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    if not args.no_rerun:
        markers = None
        marker_file = (
            POSES / f"third_person_camera_poses/{config.recording_day}/params/marker_points.npy"
        )
        if marker_file.exists():
            markers = marker_points(config.recording_day)
        log_recording(
            RerunInputs(
                trial=config.trial,
                cams=cams,
                fpv_source=fpv_source,
                result=result,
                summary=summary,
                rig_reference=reference,
                markers=markers,
                video_root=args.video_root,
                image_every=max(1, args.image_every),
            ),
            args.output / "slice.rrd",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
