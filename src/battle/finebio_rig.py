"""``battle-finebio-rig``: the standing rig check of a FineBio trial and its gates (p1-rig).

The preflight's ``rig`` subcommand generalised to any camera config and any detector pass.
Inputs: a `FineBioCameraConfig` JSON, a detections directory (one ``<view>.jsonl`` per view,
rows ``{"frame_index": raw frame, "detections": [{"class", "score", "box_xyxy_px"}]}``, the
shape of ``runs/preflight-finebio-20260924/detections`` and of ``battle-finebio-detect``) and a
frame list. Every triangulation goes through `Camera.undistort` and `dlt_triangulate` on the
projection matrices; reprojection uses the distortion.

The seven cross-checks, as in the preflight:

1. static objects: per-view median detector box centre over the frames, five-view (>= 3)
   triangulation, all-view and leave-one-view-out residuals, height above the bench (half
   the object's physical height when the board plane is the bench);
2. hands as probes: per-frame triangulation from >= 3 fixed views, residual and height;
3. clock scan: per view and moving class, the residual of the other views' triangulation
   reprojected into the view shifted by -15..+15 frames;
4. leave-one-view-out on moving objects per frame, and whether the reprojection lands inside
   the held-out box;
5. fixed -> fpv hand-off: the plate (and the hands) triangulated from the fixed views and
   projected through the shipped fpv pose against the fpv detector box;
6. the fpv camera centre projected into the fixed views (the head-mounted GoPro);
7. marker reprojection per view (read from the camera config, measured by the camera solve).

**Gates are formulas, never constants**, written into ``rig.json["gates"]``:

    association_px = clamp(3 * static_loo_median_px, floor, cap)
    handoff_px     = clamp(moving_loo_p90_px, floor, cap)

with ``floor = 15`` and ``cap = 80`` px at 1920 (scaled by image width / 1920). The floor is
1.5x the preflight's static leave-one-out median: box-centre parallax alone puts true matches
10 px off, so a tighter gate would reject them. The cap is about 4% of the frame width, below
the spacing of neighbouring same-class bench objects in the fixed views (tip racks 80-150 px
apart) and inside a plate's detector box in the fpv (250-360 px), so a wider gate would start
merging identities. ``moving_loo_p90_px`` is the **max over views** of the p90 of the
per-frame held-out residuals of the tracked moving objects (plate, pipette; hands are probes
and stay out) reprojected into that view: the fixed-view leave-one-out cells and the fixed ->
fpv hand-off. One gate is applied to every view, so the widest view sets it; the pooled p90
would refuse a fifth of the correct hand-offs into the head camera. On P03_01_01 the formulas
give 31 px and 52 px (the preflight's 30 / 50-60 px). The clock offset per view is the median
best offset over the informative (class, view) scans with a +/-1 frame uncertainty; birth
needs 3 fixed views, or 2 fixed views plus the fpv when its pose is valid.

``--negative-control`` re-evaluates every ``marker_pnp`` view with its **shipped** pose and
reports the static, hand and marker residuals side by side (the Evidence preset's control).

    uv run battle-finebio-rig --config configs/finebio/cameras/P03_01_01.json \
        --detections runs/preflight-finebio-20260924/detections \
        --output runs/finebio-rig-P03_01_01-20260925 --negative-control

Regression: on the preflight detections this reproduces ``rig_reference.json`` (static points
within 0.05 cm, LOO residuals within 0.5 px, hands, clock best offsets, plate hand-off).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .finebio_cameras import (
    Camera,
    cameras_from_config,
    fixed_camera,
    fpv_camera_from_config,
    fpv_poses,
    read_camera_config,
)
from .multiview_geometry import dlt_triangulate
from .multiview_schemas import FINEBIO_FIXED_VIEWS, FineBioCameraConfig

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
MOVING_CLASSES: tuple[str, ...] = ("left_hand", "right_hand", "cell_culture_plate", "blue_pipette")
HAND_CLASSES: tuple[str, ...] = ("left_hand", "right_hand")
# The moving classes the tracker will own; hands are probes and stay out of the gate.
TRACKED_MOVING_CLASSES: tuple[str, ...] = ("cell_culture_plate", "blue_pipette")
HANDOFF_CLASS = "cell_culture_plate"
GATE_MIN_RESIDUALS_PER_VIEW = 10
STATIC_MIN_SCORE = 0.5
HAND_MIN_SCORE = 0.5
MOVING_MIN_SCORE = 0.4
CLOCK_OFFSETS: tuple[int, ...] = tuple(range(-15, 16))
CLOCK_MIN_FRAMES = 10
GATE_FLOOR_PX = 15.0
GATE_CAP_PX = 80.0
GATE_REFERENCE_WIDTH_PX = 1920
ASSOCIATION_FACTOR = 3.0
BIRTH_MIN_FIXED_VIEWS = 3
BIRTH_FIXED_VIEWS_WITH_FPV = 2
CLOCK_UNCERTAINTY_FRAMES = 1
# A clock scan is informative when the best offset beats both +/-3 neighbours by this factor.
CLOCK_INFORMATIVE_RATIO = 0.8

Detections = dict[str, dict[int, list[dict[str, Any]]]]
FpvSource = Callable[[int], Camera | None]


# --------------------------------------------------------------------------- inputs


def load_detections(root: Path, views: Iterable[str]) -> Detections:
    """``{view: {raw frame: detections}}`` from ``<view>.jsonl`` files; missing views skipped."""
    out: Detections = {}
    for view in views:
        path = root / f"{view}.jsonl"
        if not path.exists():
            continue
        per_frame: dict[int, list[dict[str, Any]]] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rec = json.loads(line)
                    per_frame[int(rec["frame_index"])] = rec["detections"]
        out[view] = per_frame
    return out


def frames_in_detections(
    dets: Detections, views: Iterable[str], *, min_views: int = 3
) -> list[int]:
    """Frames detected in at least `min_views` of `views`."""
    counts: dict[int, int] = {}
    for view in views:
        for frame in dets.get(view, {}):
            counts[frame] = counts.get(frame, 0) + 1
    return sorted(f for f, n in counts.items() if n >= min_views)


def consecutive_frames(frames: list[int]) -> list[int]:
    present = set(frames)
    return [f for f in frames if f + 1 in present or f - 1 in present]


def subsample(frames: list[int], limit: int) -> list[int]:
    if limit <= 0 or len(frames) <= limit:
        return list(frames)
    picks = np.linspace(0, len(frames) - 1, limit).round().astype(int)
    return [frames[i] for i in sorted(set(picks.tolist()))]


def parse_frames(text: str) -> list[int]:
    frames: set[int] = set()
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            first, last, step = (int(v) for v in token.split(":"))
            if step <= 0:
                raise ValueError(f"expected A:B:S with S > 0, got {token!r}")
            frames.update(range(first, last, step))
        elif "-" in token:
            first, last = (int(v) for v in token.split("-", 1))
            frames.update(range(first, last + 1))
        else:
            frames.add(int(token))
    if not frames or min(frames) < 0:
        raise ValueError("frame indices must be >= 0 and non-empty")
    return sorted(frames)


# --------------------------------------------------------------------------- geometry


def top_box(dets: list[dict[str, Any]], cls: str, min_score: float) -> dict[str, Any] | None:
    cands = [d for d in dets if d["class"] == cls and d["score"] >= min_score]
    return max(cands, key=lambda d: d["score"]) if cands else None


def centre(box: list[float]) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


def inside(box: list[float], point: np.ndarray) -> bool:
    x0, y0, x1, y1 = box
    return bool(x0 <= point[0] <= x1 and y0 <= point[1] <= y1)


def triangulate(cams: list[Camera], pixels: list[np.ndarray]) -> np.ndarray:
    pts = np.stack([c.undistort(p).reshape(2) for c, p in zip(cams, pixels)])[:, None, :]
    P = np.stack([c.projection for c in cams])
    return dlt_triangulate(pts, P)[0]


def _reproject_residual(cam: Camera, X: np.ndarray, pixel: np.ndarray) -> float:
    return float(np.linalg.norm(cam.project(X)[0] - pixel))


def _summary(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    return {
        "n": len(values),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
    }


# --------------------------------------------------------------------------- the checks


def static_objects(
    dets: Detections,
    cams: dict[str, Camera],
    frames: list[int],
    *,
    classes: tuple[str, ...] = STATIC_CLASSES,
    min_score: float = STATIC_MIN_SCORE,
) -> list[dict[str, Any]]:
    rows = []
    for cls in classes:
        per_view: dict[str, np.ndarray] = {}
        for v in cams:
            cs = []
            for f in frames:
                box = top_box(dets.get(v, {}).get(f, []), cls, min_score)
                if box:
                    cs.append(centre(box["box_xyxy_px"]))
            if len(cs) >= max(3, len(frames) // 3):
                per_view[v] = np.median(np.stack(cs), axis=0)
        if len(per_view) < 3:
            continue
        vs = list(per_view)
        X = triangulate([cams[v] for v in vs], [per_view[v] for v in vs])
        resid = {v: _reproject_residual(cams[v], X, per_view[v]) for v in vs}
        loo = {}
        for v in vs:
            others = [u for u in vs if u != v]
            if len(others) < 2:
                continue
            Xo = triangulate([cams[u] for u in others], [per_view[u] for u in others])
            loo[v] = _reproject_residual(cams[v], Xo, per_view[v])
        rows.append(
            {
                "class": cls,
                "views": vs,
                "point_cm": X.round(2).tolist(),
                "height_cm": float(-X[2]),
                "residual_px": resid,
                "loo_px": loo,
                "median_centre_px": {v: per_view[v].round(1).tolist() for v in vs},
            }
        )
    return rows


def hands(
    dets: Detections,
    cams: dict[str, Camera],
    frames: list[int],
    *,
    classes: tuple[str, ...] = HAND_CLASSES,
    min_score: float = HAND_MIN_SCORE,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[int, np.ndarray]]]:
    rows: dict[str, list[dict[str, Any]]] = {cls: [] for cls in classes}
    points: dict[str, dict[int, np.ndarray]] = {cls: {} for cls in classes}
    for cls in classes:
        for f in frames:
            vs, px = [], []
            for v in cams:
                b = top_box(dets.get(v, {}).get(f, []), cls, min_score)
                if b:
                    vs.append(v)
                    px.append(centre(b["box_xyxy_px"]))
            if len(vs) < 3:
                continue
            X = triangulate([cams[v] for v in vs], px)
            resid = {v: _reproject_residual(cams[v], X, p) for v, p in zip(vs, px)}
            rows[cls].append(
                {"frame": f, "views": vs, "point_cm": X.round(2).tolist(), "residual_px": resid}
            )
            points[cls][f] = X
    return rows, points


def clock_scan(
    dets: Detections,
    cams: dict[str, Camera],
    frames: list[int],
    *,
    classes: tuple[str, ...] = MOVING_CLASSES,
    offsets: tuple[int, ...] = CLOCK_OFFSETS,
    min_score: float = MOVING_MIN_SCORE,
    min_frames: int = CLOCK_MIN_FRAMES,
) -> dict[str, dict[str, Any]]:
    clock: dict[str, dict[str, Any]] = {}
    for v in cams:
        clock[v] = {}
        others_all = [u for u in cams if u != v]
        for cls in classes:
            # The other views' triangulation per frame does not depend on the offset.
            base: dict[int, np.ndarray] = {}
            for f in frames:
                others, px = [], []
                for u in others_all:
                    b = top_box(dets.get(u, {}).get(f, []), cls, min_score)
                    if b:
                        others.append(u)
                        px.append(centre(b["box_xyxy_px"]))
                if len(others) >= 2:
                    base[f] = triangulate([cams[u] for u in others], px)
            per_offset = {}
            for k in offsets:
                res = []
                for f, X in base.items():
                    if f + k not in dets.get(v, {}):
                        continue
                    bv = top_box(dets[v][f + k], cls, min_score)
                    if not bv:
                        continue
                    res.append(_reproject_residual(cams[v], X, centre(bv["box_xyxy_px"])))
                if len(res) >= min_frames:
                    per_offset[k] = float(np.median(res))
            if not per_offset:
                continue
            best_k = min(per_offset, key=per_offset.get)
            neighbours = [per_offset.get(best_k + d) for d in (-3, 3)]
            informative = all(
                r is not None and per_offset[best_k] <= CLOCK_INFORMATIVE_RATIO * r
                for r in neighbours
            )
            clock[v][cls] = {
                "best_offset": best_k,
                "residual_px_at_best": per_offset[best_k],
                "residual_px_at_0": per_offset.get(0),
                "informative": informative,
                "residual_px_by_offset": {str(k): r for k, r in per_offset.items()},
            }
    return clock


def loo_moving(
    dets: Detections,
    cams: dict[str, Camera],
    frames: list[int],
    *,
    classes: tuple[str, ...] = MOVING_CLASSES,
    gate_classes: tuple[str, ...] = TRACKED_MOVING_CLASSES,
    min_score: float = MOVING_MIN_SCORE,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[float]]]:
    """Per class and held-out view: the 4-view (>= 3) triangulation reprojected into the
    held-out view against its box centre. Also returns, per held-out view, every per-frame
    residual of the `gate_classes` (the tracked objects; hands are probes) for the gate."""
    result: dict[str, dict[str, Any]] = {}
    pooled: dict[str, list[float]] = {v: [] for v in cams}
    for cls in classes:
        result[cls] = {}
        for v in cams:
            res, inside_flags = [], []
            for f in frames:
                bv = top_box(dets.get(v, {}).get(f, []), cls, min_score)
                if not bv:
                    continue
                others, px = [], []
                for u in cams:
                    if u == v:
                        continue
                    b = top_box(dets.get(u, {}).get(f, []), cls, min_score)
                    if b:
                        others.append(u)
                        px.append(centre(b["box_xyxy_px"]))
                if len(others) < 3:
                    continue
                X = triangulate([cams[u] for u in others], px)
                p = cams[v].project(X)[0]
                res.append(float(np.linalg.norm(p - centre(bv["box_xyxy_px"]))))
                inside_flags.append(inside(bv["box_xyxy_px"], p))
            if res:
                result[cls][v] = {
                    "frames": len(res),
                    "median_px": float(np.median(res)),
                    "p90_px": float(np.percentile(res, 90)),
                    "inside_fraction": float(np.mean(inside_flags)),
                }
                if cls in gate_classes:
                    pooled[v].extend(res)
    return result, pooled


def fpv_handoff(
    dets: Detections,
    cams: dict[str, Camera],
    frames: list[int],
    fpv_camera: FpvSource,
    hand_points: dict[str, dict[int, np.ndarray]],
    *,
    handoff_class: str = HANDOFF_CLASS,
    min_score: float = MOVING_MIN_SCORE,
) -> list[dict[str, Any]]:
    rows = []
    for f in frames:
        cam_f = fpv_camera(f)
        if cam_f is None:
            continue
        row: dict[str, Any] = {
            "frame": f,
            "fpv_centre_cm": cam_f.centre.round(2).tolist(),
            "centre_in_view_px": {
                v: cams[v].project(cam_f.centre)[0].round(1).tolist() for v in cams
            },
        }
        others, px = [], []
        for u in cams:
            b = top_box(dets.get(u, {}).get(f, []), handoff_class, min_score)
            if b:
                others.append(u)
                px.append(centre(b["box_xyxy_px"]))
        bf = top_box(dets.get("fpv", {}).get(f, []), handoff_class, min_score)
        if len(others) >= 3 and bf:
            X = triangulate([cams[u] for u in others], px)
            p = cam_f.project(X)[0]
            row["plate_fixed_to_fpv"] = {
                "projected_px": p.round(1).tolist(),
                "fpv_box_centre_px": centre(bf["box_xyxy_px"]).round(1).tolist(),
                "residual_px": float(np.linalg.norm(p - centre(bf["box_xyxy_px"]))),
                "inside_box": inside(bf["box_xyxy_px"], p),
                "fixed_views": others,
            }
        for cls, pts in hand_points.items():
            if f in pts:
                bf2 = top_box(dets.get("fpv", {}).get(f, []), cls, min_score)
                if bf2:
                    p = cam_f.project(pts[f])[0]
                    row[f"{cls}_fixed_to_fpv_px"] = float(
                        np.linalg.norm(p - centre(bf2["box_xyxy_px"]))
                    )
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- gates


def clamp(value: float, floor: float, cap: float) -> float:
    return float(min(max(value, floor), cap))


def clock_offsets(clock: dict[str, dict[str, Any]]) -> dict[str, Any]:
    per_view = {}
    for v, scans in clock.items():
        informative = [s["best_offset"] for s in scans.values() if s["informative"]]
        offset = int(round(float(np.median(informative)))) if informative else 0
        per_view[v] = {
            "offset_frames": offset,
            "uncertainty_frames": CLOCK_UNCERTAINTY_FRAMES,
            # An offset inside the uncertainty is reported, not applied.
            "significant": abs(offset) > CLOCK_UNCERTAINTY_FRAMES,
            "informative_scans": len(informative),
            "informative_best_offsets": informative,
            "basis": "median best offset over informative scans"
            if informative
            else "no informative scan; assumed synchronised",
        }
    return per_view


def compute_gates(
    static_rows: list[dict[str, Any]],
    moving_residuals_by_view: dict[str, list[float]],
    clock: dict[str, dict[str, Any]],
    *,
    image_width_px: int = GATE_REFERENCE_WIDTH_PX,
    floor_px: float = GATE_FLOOR_PX,
    cap_px: float = GATE_CAP_PX,
) -> dict[str, Any]:
    """The gates block. `moving_residuals_by_view` holds, per view the object is handed into
    (fixed views and ``fpv``), the per-frame held-out residuals of the tracked moving objects."""
    scale = image_width_px / GATE_REFERENCE_WIDTH_PX
    floor, cap = floor_px * scale, cap_px * scale
    static_loo = [r for row in static_rows for r in row["loo_px"].values()]
    static_median = float(np.median(static_loo)) if static_loo else None
    per_view_p90 = {
        v: float(np.percentile(res, 90))
        for v, res in moving_residuals_by_view.items()
        if len(res) >= GATE_MIN_RESIDUALS_PER_VIEW
    }
    pooled = [r for v, res in moving_residuals_by_view.items() if v in per_view_p90 for r in res]
    moving_p90 = max(per_view_p90.values()) if per_view_p90 else None
    return {
        "image_width_px": image_width_px,
        "floor_px": floor,
        "cap_px": cap,
        "formula": {
            "association_px": f"clamp({ASSOCIATION_FACTOR:g} * static_loo_median_px, floor, cap)",
            "handoff_px": "clamp(moving_loo_p90_px, floor, cap)",
            "moving_loo_p90_px": (
                "max over views of the p90 of the per-frame held-out residuals of the tracked "
                "moving objects (plate, pipette; hands are probes) reprojected into that view: "
                "the fixed-view leave-one-out cells and the fixed -> fpv hand-off; one gate is "
                f"applied to every view, so the widest view sets it; views with fewer than "
                f"{GATE_MIN_RESIDUALS_PER_VIEW} residuals are skipped"
            ),
            "floor_cap_rationale": (
                "floor 15 px at 1920 = 1.5x the static LOO median of box-centre parallax, so true "
                "matches are not rejected; cap 80 px = ~4% of the width, under the spacing of "
                "neighbouring same-class bench objects and inside an fpv plate box, so identities "
                "do not merge; both scale with image width"
            ),
        },
        "inputs": {
            "static_loo_median_px": static_median,
            "static_loo_cells": len(static_loo),
            "moving_loo_p90_px": moving_p90,
            "moving_loo_p90_px_by_view": per_view_p90,
            "moving_loo_pooled_p90_px": float(np.percentile(pooled, 90)) if pooled else None,
            "moving_residuals_by_view": {
                v: len(res) for v, res in moving_residuals_by_view.items()
            },
        },
        "association_px": None
        if static_median is None
        else clamp(ASSOCIATION_FACTOR * static_median, floor, cap),
        "handoff_px": None if moving_p90 is None else clamp(moving_p90, floor, cap),
        "clock_offset_frames": clock_offsets(clock),
        "birth_min_fixed_views": BIRTH_MIN_FIXED_VIEWS,
        "birth_fixed_views_with_fpv": BIRTH_FIXED_VIEWS_WITH_FPV,
    }


# --------------------------------------------------------------------------- negative control


def negative_control(
    config: FineBioCameraConfig,
    cams: dict[str, Camera],
    dets: Detections,
    frames: list[int],
    consecutive: list[int],
) -> dict[str, Any]:
    """Every ``marker_pnp`` view re-evaluated with its shipped pose (other views unchanged):
    static all-view and LOO residuals, hand residuals and the marker fit, side by side."""
    out: dict[str, Any] = {}
    for view, fixed in config.fixed.items():
        if fixed.provenance != "marker_pnp":
            continue
        shipped = fixed_camera(config.recording_day, fixed.camera_id, view)
        alt = {**cams, view: shipped}
        pnp_static = static_objects(dets, cams, frames)
        shipped_static = static_objects(dets, alt, frames)
        pnp_hands, _ = hands(dets, cams, consecutive)
        shipped_hands, _ = hands(dets, alt, consecutive)

        def view_loo(rows: list[dict[str, Any]]) -> dict[str, float]:
            return {row["class"]: row["loo_px"][view] for row in rows if view in row["loo_px"]}

        def view_all(rows: list[dict[str, Any]]) -> dict[str, float]:
            return {
                row["class"]: row["residual_px"][view] for row in rows if view in row["residual_px"]
            }

        def hand_res(rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
            return {
                cls: _summary([r["residual_px"][view] for r in items if view in r["residual_px"]])
                for cls, items in rows.items()
            }

        pnp_loo, shipped_loo = view_loo(pnp_static), view_loo(shipped_static)
        out[view] = {
            "camera_id": fixed.camera_id,
            "marker_rms_px": {
                "marker_pnp": fixed.marker_fit_residual_px,
                "shipped": fixed.shipped_marker_residual_px,
            },
            "centre_cm": {
                "marker_pnp": cams[view].centre.round(2).tolist(),
                "shipped": shipped.centre.round(2).tolist(),
                "distance": float(np.linalg.norm(cams[view].centre - shipped.centre)),
            },
            "static_loo_px": {
                "marker_pnp": pnp_loo,
                "shipped": shipped_loo,
                "median": {
                    "marker_pnp": float(np.median(list(pnp_loo.values()))) if pnp_loo else None,
                    "shipped": float(np.median(list(shipped_loo.values())))
                    if shipped_loo
                    else None,
                },
            },
            "static_all_view_px": {
                "marker_pnp": view_all(pnp_static),
                "shipped": view_all(shipped_static),
            },
            "hands_px": {"marker_pnp": hand_res(pnp_hands), "shipped": hand_res(shipped_hands)},
            "static_heights_cm": {
                "marker_pnp": {row["class"]: row["height_cm"] for row in pnp_static},
                "shipped": {row["class"]: row["height_cm"] for row in shipped_static},
            },
        }
    return out


# --------------------------------------------------------------------------- run


def run_rig(
    config: FineBioCameraConfig,
    dets: Detections,
    frames: list[int],
    fpv_camera: FpvSource,
    *,
    clock_max_frames: int = 600,
    with_negative_control: bool = False,
    frames_source: str = "given",
) -> dict[str, Any]:
    cams = cameras_from_config(config)
    consecutive = consecutive_frames(frames)
    clock_frames = subsample(consecutive, clock_max_frames)
    static_rows = static_objects(dets, cams, frames)
    hand_rows, hand_points = hands(dets, cams, consecutive)
    clock = clock_scan(dets, cams, clock_frames)
    moving, pooled = loo_moving(dets, cams, consecutive)
    fpv_rows = fpv_handoff(dets, cams, consecutive, fpv_camera, hand_points)
    plate_res = [
        r["plate_fixed_to_fpv"]["residual_px"] for r in fpv_rows if "plate_fixed_to_fpv" in r
    ]
    plate_in = [
        r["plate_fixed_to_fpv"]["inside_box"] for r in fpv_rows if "plate_fixed_to_fpv" in r
    ]
    hand_handoff = {
        cls: [r[f"{cls}_fixed_to_fpv_px"] for r in fpv_rows if f"{cls}_fixed_to_fpv_px" in r]
        for cls in HAND_CLASSES
    }
    moving_by_view = {**pooled, "fpv": list(plate_res)}
    width = (
        next(iter(config.fixed.values())).image_size[0] if config.fixed else GATE_REFERENCE_WIDTH_PX
    )
    gates = compute_gates(static_rows, moving_by_view, clock, image_width_px=width)
    static_loo = [r for row in static_rows for r in row["loo_px"].values()]
    report: dict[str, Any] = {
        "kind": "finebio_rig",
        "trial": config.trial,
        "day": config.recording_day,
        "camera_config": {
            "trial": config.trial,
            "frame_index_offset": config.frame_index_offset,
            "views": {
                v: {
                    "camera_id": c.camera_id,
                    "provenance": c.provenance,
                    "marker_fit_residual_px": c.marker_fit_residual_px,
                    "shipped_marker_residual_px": c.shipped_marker_residual_px,
                }
                for v, c in config.fixed.items()
            },
            "dropped_views": config.provenance.get("dropped_views", {}),
        },
        "camera_ids": {v: c.camera_id for v, c in config.fixed.items()},
        "camera_provenance": {v: c.provenance for v, c in config.fixed.items()},
        "frames": {
            "source": frames_source,
            "count": len(frames),
            "first": frames[0] if frames else None,
            "last": frames[-1] if frames else None,
            "consecutive": len(consecutive),
            "clock_scan_frames": len(clock_frames),
        },
        "static": static_rows,
        "static_loo_px": _summary(static_loo),
        "hands": hand_rows,
        "hands_summary": {
            cls: {
                "frames_with_3_views": len(rows),
                "consecutive_frames": len(consecutive),
                "residual_px": _summary([r for row in rows for r in row["residual_px"].values()]),
                "height_cm_median": float(np.median([-row["point_cm"][2] for row in rows]))
                if rows
                else None,
            }
            for cls, rows in hand_rows.items()
        },
        "clock": clock,
        "loo_moving": moving,
        "fpv": fpv_rows,
        "fpv_summary": {
            "pose_valid_consecutive_frames": len(fpv_rows),
            "consecutive_frames": len(consecutive),
            "plate_fixed_to_fpv_px": {
                **(_summary(plate_res) or {"n": 0}),
                "inside_box_fraction": float(np.mean(plate_in)) if plate_in else None,
            },
            **{f"{cls}_fixed_to_fpv_px": _summary(values) for cls, values in hand_handoff.items()},
        },
        "gates": gates,
    }
    if with_negative_control:
        report["negative_control"] = negative_control(config, cams, dets, frames, consecutive)
    return report


# --------------------------------------------------------------------------- report text


def _fmt(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def rig_markdown(report: dict[str, Any]) -> str:
    cams = report["camera_config"]["views"]
    frames = report["frames"]
    camera_text = ", ".join(
        f"{v}=cam{c['camera_id']} ({c['provenance']}, markers "
        f"{_fmt(c['marker_fit_residual_px'])} px)"
        for v, c in cams.items()
    )
    if report["camera_config"]["dropped_views"]:
        camera_text += f"; dropped: {', '.join(report['camera_config']['dropped_views'])}"
    lines = [
        f"# Rig checks, {report['trial']} (day {report['day']})",
        "",
        "Cameras: " + camera_text,
        f"Frames: {frames['count']} ({frames['source']}), raw {frames['first']}..{frames['last']}; "
        f"consecutive {frames['consecutive']}; clock scan on {frames['clock_scan_frames']}.",
        "",
        "## Static objects (median box centre over the frames, score >= 0.5)",
        "",
        "| class | views | height above bench (cm) | all-view residual px per view "
        "| LOO residual px per view |",
        "|---|---|---|---|---|",
    ]
    for row in report["static"]:
        lines.append(
            f"| {row['class']} | {len(row['views'])} | {row['height_cm']:.1f} | "
            + " ".join(f"{v}:{r:.0f}" for v, r in row["residual_px"].items())
            + " | "
            + " ".join(f"{v}:{r:.0f}" for v, r in row["loo_px"].items())
            + " |"
        )
    s = report["static_loo_px"]
    lines += [
        "",
        (
            f"Static LOO residual over {s['n']} (object, view) cells: median {s['median']:.1f} px, "
            f"p90 {s['p90']:.1f} px."
            if s
            else "no static objects triangulated"
        ),
        "",
        "## Hands per frame (consecutive frames, score >= 0.5, >= 3 views)",
        "",
    ]
    for cls, h in report["hands_summary"].items():
        if h["residual_px"]:
            lines.append(
                f"- **{cls}**: {h['frames_with_3_views']}/{h['consecutive_frames']} frames with "
                f">= 3 views; residual median {h['residual_px']['median']:.1f} px, p90 "
                f"{h['residual_px']['p90']:.1f}; height above bench median "
                f"{h['height_cm_median']:.1f} cm"
            )
    lines += [
        "",
        "## Clock scan per view (-15..+15 frames; residual of the other views' triangulation "
        "reprojected into the shifted view)",
        "",
        "| view | class | best offset | residual at best (px) | residual at 0 (px) | informative |",
        "|---|---|---|---|---|---|",
    ]
    for v, scans in report["clock"].items():
        for cls, scan in scans.items():
            lines.append(
                f"| {v} | {cls} | {scan['best_offset']:+d} | {scan['residual_px_at_best']:.1f} | "
                f"{_fmt(scan['residual_px_at_0'])} | {'yes' if scan['informative'] else 'flat'} |"
            )
    lines += [
        "",
        "## Leave-one-view-out, moving objects (consecutive frames)",
        "",
        "| class | held-out view | frames | centre residual median px | p90 "
        "| inside held-out box |",
        "|---|---|---|---|---|---|",
    ]
    for cls, views in report["loo_moving"].items():
        for v, cell in views.items():
            lines.append(
                f"| {cls} | {v} | {cell['frames']} | {cell['median_px']:.1f} | "
                f"{cell['p90_px']:.1f} | {cell['inside_fraction']:.2f} |"
            )
    fpv = report["fpv_summary"]
    plate = fpv["plate_fixed_to_fpv_px"]
    lines += [
        "",
        "## First-person camera",
        "",
        f"- fpv pose valid on {fpv['pose_valid_consecutive_frames']}/{fpv['consecutive_frames']} "
        "consecutive frames.",
    ]
    if plate.get("n"):
        lines.append(
            "- Plate triangulated from the fixed views and projected into the fpv: residual vs "
            f"the fpv detector box centre median {plate['median']:.0f} px, p90 {plate['p90']:.0f}; "
            f"inside the fpv box on {plate['inside_box_fraction']:.2f} of {plate['n']} frames."
        )
    for cls in HAND_CLASSES:
        h = fpv.get(f"{cls}_fixed_to_fpv_px")
        if h:
            lines.append(
                f"- {cls} (fixed-view triangulation -> fpv): residual median {h['median']:.0f} px, "
                f"p90 {h['p90']:.0f} over {h['n']} frames."
            )
    g = report["gates"]
    inputs = g["inputs"]
    lines += [
        "",
        "## Gates (formulas evaluated on this trial)",
        "",
        f"- `association_px = {g['formula']['association_px']}` = "
        f"clamp({ASSOCIATION_FACTOR:g} * {_fmt(inputs['static_loo_median_px'])}, "
        f"{g['floor_px']:.0f}, {g['cap_px']:.0f}) = **{_fmt(g['association_px'])} px**",
        f"- `handoff_px = {g['formula']['handoff_px']}` = "
        f"clamp({_fmt(inputs['moving_loo_p90_px'])}, {g['floor_px']:.0f}, {g['cap_px']:.0f}) = "
        f"**{_fmt(g['handoff_px'])} px** (per-view p90: "
        + ", ".join(f"{v} {r:.1f}" for v, r in inputs["moving_loo_p90_px_by_view"].items())
        + f"; pooled p90 {_fmt(inputs['moving_loo_pooled_p90_px'])})",
        "- clock offsets (frames, +/-1): "
        + ", ".join(
            f"{v} {c['offset_frames']:+d} ({c['informative_scans']} informative)"
            for v, c in g["clock_offset_frames"].items()
        ),
        f"- birth: >= {g['birth_min_fixed_views']} fixed views, or >= "
        f"{g['birth_fixed_views_with_fpv']} fixed views plus the fpv with a valid pose",
        f"- {g['formula']['floor_cap_rationale']}",
    ]
    if report.get("negative_control"):
        lines += ["", "## Negative control: marker-PnP pose vs the shipped pose", ""]
        for v, nc in report["negative_control"].items():
            hands_text = ", ".join(
                f"{cls} {_fmt((nc['hands_px']['marker_pnp'][cls] or {}).get('median'))} vs "
                f"{_fmt((nc['hands_px']['shipped'][cls] or {}).get('median'))} px"
                for cls in nc["hands_px"]["marker_pnp"]
            )
            distance = nc["centre_cm"]["distance"]
            lines += [
                f"- **{v}** (camera {nc['camera_id']}, centres {distance:.1f} cm "
                f"apart): markers {_fmt(nc['marker_rms_px']['marker_pnp'], 2)} vs "
                f"{_fmt(nc['marker_rms_px']['shipped'])} px; static LOO median "
                f"{_fmt(nc['static_loo_px']['median']['marker_pnp'])} vs "
                f"{_fmt(nc['static_loo_px']['median']['shipped'])} px; hands {hands_text}",
                "",
                "  | class | LOO marker_pnp (px) | LOO shipped (px) | height marker_pnp (cm) "
                "| height shipped (cm) |",
                "  |---|---|---|---|---|",
            ]
            for cls in nc["static_loo_px"]["marker_pnp"]:
                lines.append(
                    f"  | {cls} | {nc['static_loo_px']['marker_pnp'][cls]:.1f} | "
                    f"{_fmt(nc['static_loo_px']['shipped'].get(cls))} | "
                    f"{_fmt(nc['static_heights_cm']['marker_pnp'].get(cls))} | "
                    f"{_fmt(nc['static_heights_cm']['shipped'].get(cls))} |"
                )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- cli


def fpv_source_from_data(config: FineBioCameraConfig, trial: str) -> FpvSource:
    poses = fpv_poses(trial)

    def source(frame: int) -> Camera | None:
        return fpv_camera_from_config(config, frame, poses)

    return source


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="battle-finebio-rig",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True, help="FineBioCameraConfig JSON")
    parser.add_argument("--detections", type=Path, required=True, help="per-view JSONL directory")
    parser.add_argument(
        "--frames",
        default=None,
        help="raw frames N,A-B,A:B:S (default: frames.json or the detections)",
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="default runs/finebio-rig-<trial>-<date>"
    )
    parser.add_argument("--trial", default=None, help="fpv pose file trial (default: the config's)")
    parser.add_argument("--clock-max-frames", type=int, default=600)
    parser.add_argument("--negative-control", action="store_true")
    args = parser.parse_args(argv)

    config = read_camera_config(args.config)
    trial = args.trial or config.trial
    views = (*FINEBIO_FIXED_VIEWS, "fpv")
    dets = load_detections(args.detections, views)
    if args.frames:
        frames, source = parse_frames(args.frames), "--frames"
    elif (args.detections / "frames.json").exists():
        frames = sorted(json.loads((args.detections / "frames.json").read_text())["frames"])
        source = "frames.json"
    else:
        frames, source = frames_in_detections(dets, config.fixed), "detections"
    output = args.output or Path("runs") / f"finebio-rig-{trial}-{datetime.now(UTC):%Y%m%d}"
    output.mkdir(parents=True, exist_ok=True)
    report = run_rig(
        config,
        dets,
        frames,
        fpv_source_from_data(config, trial),
        clock_max_frames=args.clock_max_frames,
        with_negative_control=args.negative_control,
        frames_source=source,
    )
    report["provenance"] = {
        "command": " ".join(sys.argv),
        "camera_config": str(args.config),
        "detections": str(args.detections),
        "written": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (output / "rig.json").write_text(json.dumps(report, indent=1) + "\n")
    text = rig_markdown(report)
    (output / "rig.md").write_text(text)
    print(text)
    print(f"wrote {output / 'rig.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
