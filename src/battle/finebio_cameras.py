"""FineBio camera library: shipped calibration, ArUco markers, marker PnP, per-trial config.

Moved out of ``scripts/finebio_preflight.py`` on Sep 24 (p0-contracts) without changing a
code path, so the preflight's ``mapping`` and ``rig`` outputs are reproduced byte for byte.

Data: ``data/raw/finebio/misc/finebio_camera_poses`` (FineBio, non-commercial research).
Intrinsics were calibrated at 4000x3000 (fpv) and 3840x2160 (fixed); the shipped videos are
1920x1440 and 1920x1080, so `intrinsics` rescales K by 0.48 / 0.5 (no crop; the preflight
reproduced the markers to 2-7 px with this). Extrinsics are per recording day and camera id
(``<day>/extrinsics/<id>_board.npz``, world -> camera ``rvec``/``tvec``); the fpv pose is a
per-frame marker PnP (``first_person_camera_poses/<trial>.npz``: ``rets``, ``rots``, ``trans``,
one entry per raw video frame). Units are the calibration board's centimetres with z into the
bench, so height above the bench is ``-z``.

Fixed views ``T1..T5`` are cameras ``1,2,3,4,6`` on P03 (day 221013); the top-down camera 6
does not fit its shipped pose (94 px) and `rig_cameras` replaces it by the marker PnP
(`provenance: marker_pnp`). `camera_config_from_mapping` turns a preflight mapping report into
a `FineBioCameraConfig`, and `cameras_from_config` rebuilds the `Camera` objects from it.

Sep 25 (p0-cameras): `solve_mapping` is the preflight's ``mapping`` subcommand as a library
call (`solve_view`, `decide_day`, `marker_pnp`, `chosen_day_record`), `fpv_pose_check` its
``fpv-pose`` subcommand, and `camera_config_from_mapping` gates on the chosen day's residual
and drops a view that fits neither the shipped pose nor the marker PnP within 10 px. The CLI
is ``battle-finebio-cameras`` (`finebio_cameras_cli.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .multiview_schemas import (
    FINEBIO_FIXED_VIEWS,
    FineBioCameraConfig,
    FineBioFixedCamera,
    FineBioFpvCamera,
)

RAW = Path("/home/nick/src/battle/data/raw/finebio")
POSES = RAW / "misc/finebio_camera_poses"
FIXED_VIEWS = FINEBIO_FIXED_VIEWS
CAMERA_IDS = (1, 2, 3, 4, 6)
ARUCO_DICT = cv2.aruco.DICT_6X6_50
# A shipped fixed-camera pose is kept only when its median corner RMS is within this.
SHIPPED_POSE_MAX_RMS_PX = 10.0
# Default fpv validity gate parameters (FineBioFpvCamera); the preflight saw 3/196 frames over
# 20 px and a p99 centre step of 2.5 cm, so these pass normal head motion and cut the outliers.
FPV_MARKER_RESIDUAL_GATE_PX = 20.0
FPV_VELOCITY_GATE_CM_PER_FRAME = 5.0


# --------------------------------------------------------------------------- calibration io


@dataclass(frozen=True)
class Camera:
    name: str
    K: np.ndarray  # (3, 3) for the shipped video resolution
    dist: np.ndarray  # (5,)
    rvec: np.ndarray  # (3,) world -> camera
    tvec: np.ndarray  # (3,)
    size: tuple[int, int]  # (w, h)

    @property
    def R(self) -> np.ndarray:
        return cv2.Rodrigues(self.rvec.reshape(3, 1))[0]

    @property
    def centre(self) -> np.ndarray:
        return (-self.R.T @ self.tvec.reshape(3, 1)).ravel()

    @property
    def projection(self) -> np.ndarray:
        return self.K @ np.hstack([self.R, self.tvec.reshape(3, 1)])

    def project(self, points: np.ndarray) -> np.ndarray:
        pts, _ = cv2.projectPoints(
            np.asarray(points, dtype=np.float64).reshape(-1, 1, 3),
            self.rvec.astype(np.float64),
            self.tvec.astype(np.float64),
            self.K,
            self.dist,
        )
        return pts.reshape(-1, 2)

    def undistort(self, pixels: np.ndarray) -> np.ndarray:
        pts = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.undistortPoints(pts, self.K, self.dist, P=self.K).reshape(-1, 2)


def intrinsics(kind: str) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Shipped chessboard intrinsics rescaled from the calibration resolution to the video."""
    if kind == "fpv":
        d = np.load(POSES / "intrinsic_parameters/gopro9_5_wide_4k_43_0.50.npz")
        sx, sy, size = 1920 / 4000, 1440 / 3000, (1920, 1440)
    else:
        d = np.load(POSES / "intrinsic_parameters/gopro9_6_linear_4k_169_0.50.npz")
        sx, sy, size = 1920 / 3840, 1080 / 2160, (1920, 1080)
    K = d["intrinsic_matrix"].astype(np.float64).copy()
    K[0, 0] *= sx
    K[0, 2] *= sx
    K[1, 1] *= sy
    K[1, 2] *= sy
    return K, d["distCoeff"].astype(np.float64).ravel(), size


def days() -> list[str]:
    return sorted(p.name for p in (POSES / "third_person_camera_poses").iterdir() if p.is_dir())


def fixed_camera(day: str, camera_id: int, name: str | None = None) -> Camera:
    d = np.load(POSES / f"third_person_camera_poses/{day}/extrinsics/{camera_id}_board.npz")
    K, dist, size = intrinsics("tpv")
    return Camera(
        name or f"cam{camera_id}",
        K,
        dist,
        d["r"].astype(np.float64).ravel(),
        d["t"].astype(np.float64).ravel(),
        size,
    )


def rig_cameras(
    mapping: dict, *, pnp_over_px: float = SHIPPED_POSE_MAX_RMS_PX
) -> tuple[dict[str, Camera], dict[str, str]]:
    """Cameras for the five fixed views from a mapping report: the shipped extrinsics where
    they fit the markers within `pnp_over_px`, otherwise the marker-PnP pose (provenance
    recorded per view as ``shipped`` or ``marker_pnp``)."""
    day = mapping["day_decision"]["chosen"]
    cams, provenance = {}, {}
    for v in FIXED_VIEWS:
        info = mapping["views"][v]
        cam = fixed_camera(day, info["best"]["camera_id"], v)
        if info["best"]["median_corner_rms_px"] > pnp_over_px and info.get("pnp_pose"):
            cam = Camera(
                v,
                cam.K,
                cam.dist,
                np.array(info["pnp_pose"]["rvec"]),
                np.array(info["pnp_pose"]["tvec"]),
                cam.size,
            )
            provenance[v] = "marker_pnp"
        else:
            provenance[v] = "shipped"
        cams[v] = cam
    return cams, provenance


def marker_points(day: str) -> np.ndarray:
    pts = np.load(POSES / f"third_person_camera_poses/{day}/params/marker_points.npy")
    return pts.astype(np.float64).reshape(-1, 4, 3)


def fpv_pose_path(trial: str) -> Path:
    return POSES / f"first_person_camera_poses/{trial}.npz"


def fpv_poses(trial: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(rets, rots, trans)``: one entry per raw fpv frame; ``rets[f]`` is pose validity."""
    d = np.load(fpv_pose_path(trial))
    return (
        d["rets"].astype(bool),
        d["rots"].reshape(-1, 3).astype(np.float64),
        d["trans"].reshape(-1, 3).astype(np.float64),
    )


def fpv_camera(trial: str, frame: int) -> Camera | None:
    rets, rots, trans = fpv_poses(trial)
    if frame >= len(rets) or not rets[frame]:
        return None
    K, dist, size = intrinsics("fpv")
    return Camera("fpv", K, dist, rots[frame], trans[frame], size)


# --------------------------------------------------------------------------- markers


def detect_markers(img: np.ndarray) -> dict[int, np.ndarray]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(ARUCO_DICT), cv2.aruco.DetectorParameters()
    )
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2).astype(np.float64) for i, c in zip(ids.ravel(), corners)}


def match_markers(
    detected: dict[int, np.ndarray], projected: np.ndarray
) -> list[tuple[int, int, float, np.ndarray]]:
    """For each detected marker: (aruco id, index into projected markers, corner RMS px,
    projected corners in the matching corner order). Nearest projected centroid wins; the
    corner order is the cyclic shift with the smallest RMS."""
    out = []
    centroids = projected.mean(axis=1)
    for marker_id, corners in detected.items():
        j = int(np.argmin(np.linalg.norm(centroids - corners.mean(axis=0), axis=1)))
        best = None
        for shift in range(4):
            rolled = np.roll(projected[j], shift, axis=0)
            rms = float(np.sqrt(((rolled - corners) ** 2).sum(axis=1).mean()))
            if best is None or rms < best[0]:
                best = (rms, rolled)
        out.append((marker_id, j, best[0], best[1]))
    return out


def marker_corner_rms(camera: Camera, day: str, img: np.ndarray) -> float | None:
    """Median corner RMS (px) of the day's markers projected through `camera` against the
    ArUco corners detected in `img`; None when no marker is detected."""
    detected = detect_markers(img)
    if not detected:
        return None
    projected = camera.project(marker_points(day).reshape(-1, 3)).reshape(-1, 4, 2)
    return float(np.median([m[2] for m in match_markers(detected, projected)]))


# --------------------------------------------------------------------------- marker solve


def seconds_to_frame(seconds: float) -> int:
    """Raw frame index of a time in seconds at the native 30000/1001 rate (preflight rounding)."""
    return int(round(seconds * 30000 / 1001))


def marker_pnp(
    detections: list[dict[int, np.ndarray]], camera: Camera, markers: np.ndarray
) -> dict[str, Any] | None:
    """One PnP over every detected corner of every frame against `markers` (a day's
    ``marker_points``); `camera` supplies the intrinsics and the pose used to associate
    detected markers with projected ones (nearest centroid, best cyclic corner order).

    Returns ``{"rvec", "tvec", "corner_count", "corner_rms_px", "centre_cm"}`` or None when no
    marker was detected or PnP failed.
    """
    projected = camera.project(markers.reshape(-1, 3)).reshape(-1, 4, 2)
    object_points, image_points = [], []
    for detected in detections:
        for marker_id, j, _, rolled in match_markers(detected, projected):
            shift = next(
                k for k in range(4) if np.allclose(np.roll(projected[j], k, axis=0), rolled)
            )
            object_points.append(np.roll(markers[j], shift, axis=0))
            image_points.append(detected[marker_id])
    if not object_points:
        return None
    obj = np.concatenate(object_points).reshape(-1, 1, 3)
    img = np.concatenate(image_points).reshape(-1, 1, 2)
    ok, rvec, tvec = cv2.solvePnP(obj, img, camera.K, camera.dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    pnp = Camera("pnp", camera.K, camera.dist, rvec.ravel(), tvec.ravel(), camera.size)
    errors = pnp.project(obj.reshape(-1, 3)) - img.reshape(-1, 2)
    rms = float(np.sqrt((errors**2).sum(1).mean()))
    return {
        "rvec": rvec.ravel().tolist(),
        "tvec": tvec.ravel().tolist(),
        "corner_count": int(obj.shape[0]),
        "corner_rms_px": rms,
        "centre_cm": pnp.centre.tolist(),
    }


def _residual_table(
    detections: list[dict[int, np.ndarray]], all_days: list[str], camera_ids: tuple[int, ...]
) -> dict[tuple[str, int], float]:
    """Median corner RMS of every (day, camera) shipped pose against the detected corners."""
    table: dict[tuple[str, int], float] = {}
    for day in all_days:
        mp = marker_points(day)
        for cam_id in camera_ids:
            cam = fixed_camera(day, cam_id)
            rms_all: list[float] = []
            for detected in detections:
                if not detected:
                    continue
                proj = cam.project(mp.reshape(-1, 3)).reshape(-1, 4, 2)
                rms_all.extend(m[2] for m in match_markers(detected, proj))
            table[(day, cam_id)] = float(np.median(rms_all)) if rms_all else float("nan")
    return table


def solve_view(
    detections: list[dict[int, np.ndarray]],
    *,
    all_days: list[str],
    camera_ids: tuple[int, ...] = CAMERA_IDS,
) -> dict[str, Any]:
    """The preflight's per-view mapping record from the ArUco detections of one fixed view:
    every (day, camera) shipped pose ranked by median corner RMS, the best, the best other
    camera, the same camera on other days, and an independent PnP against the best day's
    markers (the regression reference; the chosen-day PnP is added by `solve_mapping`)."""
    table = _residual_table(detections, all_days, camera_ids)
    ranked = sorted(table.items(), key=lambda kv: kv[1])
    (best_day, best_cam), best_rms = ranked[0]
    runner = next(((d, c), r) for (d, c), r in ranked if c != best_cam)
    same_cam_other_day = [((d, c), r) for (d, c), r in ranked if c == best_cam and d != best_day]
    cam = fixed_camera(best_day, best_cam)
    pnp = marker_pnp(detections, cam, marker_points(best_day))
    return {
        "best": {"day": best_day, "camera_id": best_cam, "median_corner_rms_px": best_rms},
        "runner_up_other_camera": {
            "day": runner[0][0],
            "camera_id": runner[0][1],
            "median_corner_rms_px": runner[1],
        },
        "same_camera_other_days": [
            {"day": d, "median_corner_rms_px": r} for (d, _), r in same_cam_other_day
        ],
        "markers_detected_per_frame": [len(d) for d in detections],
        "shipped_centre_cm": cam.centre.round(2).tolist(),
        "pnp_centre_cm": None if pnp is None else np.array(pnp["centre_cm"]).round(2).tolist(),
        "pnp_vs_shipped_cm": None
        if pnp is None
        else float(np.linalg.norm(np.array(pnp["centre_cm"]) - cam.centre)),
        "pnp_corner_rms_px": None if pnp is None else pnp["corner_rms_px"],
        "pnp_pose": None
        if pnp is None
        else {"rvec": pnp["rvec"], "tvec": pnp["tvec"], "corner_count": pnp["corner_count"]},
        "table": {f"{d}/cam{c}": r for (d, c), r in ranked},
    }


def decide_day(views: dict[str, dict[str, Any]], all_days: list[str]) -> dict[str, Any]:
    """The recording day that minimises the summed residual of the five chosen cameras
    (the preflight's rule), plus the residual-weighted vote for the record."""
    day_sum: dict[str, float] = {}
    for day in all_days:
        vals = [
            info["table"].get(f"{day}/cam{info['best']['camera_id']}", np.nan)
            for info in views.values()
        ]
        day_sum[day] = float(np.nansum(vals)) if not all(np.isnan(vals)) else float("nan")
    ranked_days = sorted(day_sum.items(), key=lambda kv: kv[1])
    votes: dict[str, float] = {}
    for info in views.values():
        best = info["best"]
        votes[best["day"]] = votes.get(best["day"], 0.0) + 1.0 / max(
            best["median_corner_rms_px"], 0.5
        )
        for other in info["same_camera_other_days"]:
            votes[other["day"]] = votes.get(other["day"], 0.0) + 1.0 / max(
                other["median_corner_rms_px"], 0.5
            )
    return {
        "by_summed_residual": [list(kv) for kv in ranked_days],
        "chosen": ranked_days[0][0],
        "weighted_vote": max(votes.items(), key=lambda kv: kv[1])[0],
    }


def chosen_day_record(
    detections: list[dict[int, np.ndarray]], day: str, camera_id: int
) -> dict[str, Any]:
    """The camera's shipped pose on the chosen day against the markers, and the PnP in the
    chosen day's world frame (the pose a `marker_pnp` view actually uses)."""
    cam = fixed_camera(day, camera_id)
    mp = marker_points(day)
    rms_all: list[float] = []
    for detected in detections:
        if detected:
            proj = cam.project(mp.reshape(-1, 3)).reshape(-1, 4, 2)
            rms_all.extend(m[2] for m in match_markers(detected, proj))
    pnp = marker_pnp(detections, cam, mp)
    return {
        "day": day,
        "camera_id": camera_id,
        "median_corner_rms_px": float(np.median(rms_all)) if rms_all else None,
        "shipped_centre_cm": cam.centre.round(2).tolist(),
        "pnp": pnp,
        "pnp_vs_shipped_cm": None
        if pnp is None
        else float(np.linalg.norm(np.array(pnp["centre_cm"]) - cam.centre)),
    }


def solve_mapping(
    trial: str,
    frames: list[int],
    read: Any,
    *,
    seconds: list[float] | None = None,
    views: tuple[str, ...] = FIXED_VIEWS,
    camera_ids: tuple[int, ...] = CAMERA_IDS,
    all_days: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, list[tuple[int, np.ndarray, dict[int, np.ndarray]]]]]:
    """The preflight ``mapping`` report for `trial` from ArUco markers on `frames` of every
    fixed view (`read(view, frame)` returns the BGR frame), extended with a ``chosen_day``
    block per view. Returns the report and the per-view ``(frame, image, detected)`` list
    for overlays."""
    all_days = days() if all_days is None else all_days
    report: dict[str, Any] = {
        "trial": trial,
        "seconds": seconds,
        "frames": list(frames),
        "views": {},
    }
    kept: dict[str, list[tuple[int, np.ndarray, dict[int, np.ndarray]]]] = {}
    for view in views:
        kept[view] = []
        for frame in frames:
            img = read(view, frame)
            kept[view].append((frame, img, detect_markers(img)))
        report["views"][view] = solve_view(
            [d for _, _, d in kept[view]], all_days=all_days, camera_ids=camera_ids
        )
    report["day_decision"] = decide_day(report["views"], all_days)
    day = report["day_decision"]["chosen"]
    for view in views:
        info = report["views"][view]
        info["chosen_day"] = chosen_day_record(
            [d for _, _, d in kept[view]], day, info["best"]["camera_id"]
        )
    cams = [report["views"][v]["best"]["camera_id"] for v in views]
    report["camera_permutation_ok"] = len(set(cams)) == len(views)
    return report, kept


def fpv_velocity_gate_fraction(
    poses: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    K: np.ndarray,
    dist: np.ndarray,
    size: tuple[int, int],
    gate_cm_per_frame: float = FPV_VELOCITY_GATE_CM_PER_FRAME,
) -> dict[str, Any]:
    """Consecutive valid-pose frames whose camera-centre step exceeds the velocity gate."""
    rets, rots, trans = poses
    centres = np.full((len(rets), 3), np.nan)
    for f in np.flatnonzero(rets):
        centres[f] = Camera("fpv", K, dist, rots[f], trans[f], size).centre
    both = rets[1:] & rets[:-1]
    steps = np.linalg.norm(centres[1:] - centres[:-1], axis=1)[both]
    return {
        "consecutive_valid_pairs": int(both.sum()),
        "pairs_over_gate": int((steps > gate_cm_per_frame).sum()),
        "fraction_over_gate": float((steps > gate_cm_per_frame).mean()) if steps.size else 0.0,
        "step_cm_p99": float(np.percentile(steps, 99)) if steps.size else None,
        "step_cm_max": float(steps.max()) if steps.size else None,
    }


def fpv_pose_check(
    trial: str,
    day: str,
    read: Any,
    *,
    step: int = 250,
    residual_gate_px: float = FPV_MARKER_RESIDUAL_GATE_PX,
) -> dict[str, Any]:
    """The shipped fpv pose against ArUco markers on every `step`-th valid frame
    (`read(frame)` returns the BGR fpv frame): corner RMS per frame, an independent marker
    PnP centre where >= 2 markers are seen, and the summary the preflight's ``fpv-pose``
    subcommand printed."""
    rets, rots, trans = fpv_poses(trial)
    mp = marker_points(day)
    K, dist, size = intrinsics("fpv")
    rows = []
    for f in range(0, len(rets), step):
        if not rets[f]:
            continue
        img = read(f)
        detected = detect_markers(img)
        cam = Camera("fpv", K, dist, rots[f], trans[f], size)
        proj = cam.project(mp.reshape(-1, 3)).reshape(-1, 4, 2)
        matches = match_markers(detected, proj) if detected else []
        rms = [m[2] for m in matches]
        pnp = marker_pnp([detected], cam, mp) if len(matches) >= 2 else None
        rows.append(
            {
                "frame": f,
                "markers_detected": len(detected),
                "median_corner_rms_px": float(np.median(rms)) if rms else None,
                "max_corner_rms_px": float(np.max(rms)) if rms else None,
                "pnp_vs_shipped_cm": None
                if pnp is None
                else float(np.linalg.norm(np.array(pnp["centre_cm"]) - cam.centre)),
                "height_cm": float(-cam.centre[2]),
            }
        )
    med = [r["median_corner_rms_px"] for r in rows if r["median_corner_rms_px"] is not None]
    pnp_d = [r["pnp_vs_shipped_cm"] for r in rows if r["pnp_vs_shipped_cm"] is not None]
    return {
        "trial": trial,
        "day": day,
        "step": step,
        "frames_checked": len(rows),
        "frames_with_markers": len(med),
        "corner_rms_px": {
            "median": float(np.median(med)),
            "p90": float(np.percentile(med, 90)),
            "max": float(np.max(med)),
        }
        if med
        else None,
        "frames_over_gate": int(sum(m > residual_gate_px for m in med)),
        "residual_gate_px": residual_gate_px,
        "pnp_vs_shipped_cm": {
            "median": float(np.median(pnp_d)),
            "p90": float(np.percentile(pnp_d, 90)),
            "max": float(np.max(pnp_d)),
            "n": len(pnp_d),
        }
        if pnp_d
        else None,
        "velocity_gate": fpv_velocity_gate_fraction((rets, rots, trans), K=K, dist=dist, size=size),
        "rows": rows,
    }


# --------------------------------------------------------------------------- per-trial config


def _mat3(K: np.ndarray) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(float(x) for x in row) for row in np.asarray(K, dtype=np.float64))


def _vec(values: np.ndarray) -> tuple[float, ...]:
    return tuple(float(x) for x in np.asarray(values, dtype=np.float64).ravel())


def _pose_decision(
    info: dict[str, Any], day: str, *, pnp_over_px: float, drop_over_px: float
) -> tuple[str, float | None, dict[str, Any] | None, float | None]:
    """``(provenance, shipped residual, pnp pose or None, pnp residual)`` for one view.

    The residual gated is that of the pose the config would use: the chosen day's shipped
    pose, read from the ``chosen_day`` block (`solve_mapping`) or from the report's
    ``table`` (the preflight's ``mapping.json``), not the best (day, camera) residual, which
    may belong to another day. The PnP pose comes from the chosen day's markers when the
    block exists, else from the preflight's best-day PnP. ``provenance`` is ``shipped``,
    ``marker_pnp`` or ``dropped`` (neither fits within the gates: the plan's stop rule).
    """
    chosen = info.get("chosen_day")
    camera_id = info["best"]["camera_id"]
    if chosen:
        shipped_rms = chosen["median_corner_rms_px"]
        pnp = chosen["pnp"]
        pnp_rms = None if pnp is None else pnp["corner_rms_px"]
    else:
        shipped_rms = info.get("table", {}).get(
            f"{day}/cam{camera_id}", info["best"]["median_corner_rms_px"]
        )
        pnp = info.get("pnp_pose")
        pnp_rms = info.get("pnp_corner_rms_px")
    if shipped_rms is not None and np.isfinite(shipped_rms) and shipped_rms <= pnp_over_px:
        return "shipped", float(shipped_rms), pnp, pnp_rms
    if pnp is not None and pnp_rms is not None and pnp_rms <= drop_over_px:
        return "marker_pnp", None if shipped_rms is None else float(shipped_rms), pnp, pnp_rms
    return "dropped", None if shipped_rms is None else float(shipped_rms), pnp, pnp_rms


def camera_config_from_mapping(
    mapping: dict[str, Any],
    trial: str,
    *,
    pnp_over_px: float = SHIPPED_POSE_MAX_RMS_PX,
    drop_over_px: float = SHIPPED_POSE_MAX_RMS_PX,
    provenance: dict[str, Any] | None = None,
) -> FineBioCameraConfig:
    """The trial's camera config from a mapping report (`solve_mapping`, or the preflight's
    ``mapping.json``): per fixed view the shipped pose of the chosen day where its median
    corner RMS is within `pnp_over_px`, else the marker PnP where that fits within
    `drop_over_px`, else the view is **dropped** (absent from ``fixed``, listed with its
    numbers under ``provenance["dropped_views"]``); the fpv gets the rescaled intrinsics,
    its pose file and the default validity gate parameters. Raw frames, offset 0."""
    day = str(mapping["day_decision"]["chosen"])
    fixed: dict[str, FineBioFixedCamera] = {}
    dropped: dict[str, Any] = {}
    per_view: dict[str, Any] = {}
    for view in mapping["views"]:
        info = mapping["views"][view]
        camera_id = int(info["best"]["camera_id"])
        decision, shipped_rms, pnp, pnp_rms = _pose_decision(
            info, day, pnp_over_px=pnp_over_px, drop_over_px=drop_over_px
        )
        per_view[view] = {
            "best_day": info["best"]["day"],
            "best_day_median_corner_rms_px": info["best"]["median_corner_rms_px"],
            "chosen_day_median_corner_rms_px": shipped_rms,
            "pnp_corner_rms_px": pnp_rms,
            "pnp_vs_shipped_cm": (info.get("chosen_day") or info).get("pnp_vs_shipped_cm"),
            "markers_detected_per_frame": info.get("markers_detected_per_frame"),
        }
        if decision == "dropped":
            dropped[view] = {
                **per_view[view],
                "camera_id": camera_id,
                "reason": (
                    f"shipped pose {shipped_rms} px > {pnp_over_px} and marker PnP "
                    f"{pnp_rms} px > {drop_over_px} (or no markers)"
                ),
            }
            continue
        cam = fixed_camera(day, camera_id, view)
        if decision == "marker_pnp":
            assert pnp is not None
            cam = Camera(
                view, cam.K, cam.dist, np.array(pnp["rvec"]), np.array(pnp["tvec"]), cam.size
            )
        in_use_rms = pnp_rms if decision == "marker_pnp" else shipped_rms
        fixed[view] = FineBioFixedCamera(
            view=view,
            camera_id=camera_id,
            provenance=decision,
            K=_mat3(cam.K),
            distortion=_vec(cam.dist),
            rvec=_vec(cam.rvec),
            tvec=_vec(cam.tvec),
            image_size=cam.size,
            marker_fit_residual_px=None if in_use_rms is None else float(in_use_rms),
            shipped_marker_residual_px=None if shipped_rms is None else float(shipped_rms),
        )
    rets, _, _ = fpv_poses(trial)
    K_f, dist_f, size_f = intrinsics("fpv")
    fpv = FineBioFpvCamera(
        K=_mat3(K_f),
        distortion=_vec(dist_f),
        image_size=size_f,
        pose_source=str(fpv_pose_path(trial).relative_to(RAW)),
        pose_frame_count=int(rets.size),
        valid_pose_fraction=float(rets.mean()),
        marker_residual_gate_px=FPV_MARKER_RESIDUAL_GATE_PX,
        velocity_gate_cm_per_frame=FPV_VELOCITY_GATE_CM_PER_FRAME,
    )
    return FineBioCameraConfig(
        trial=trial,
        recording_day=day,
        fixed=fixed,
        fpv=fpv,
        provenance={
            "mapping_trial": mapping.get("trial"),
            "mapping_seconds": mapping.get("seconds"),
            "mapping_frames": mapping.get("frames"),
            "shipped_pose_max_rms_px": pnp_over_px,
            "pnp_drop_over_px": drop_over_px,
            "intrinsics_rescale": {"fixed": 0.5, "fpv": 0.48},
            "views": per_view,
            "dropped_views": dropped,
            **(provenance or {}),
        },
    )


def write_camera_config(config: FineBioCameraConfig, path: Path) -> Path:
    """Committable JSON (indent 1, repr floats, trailing newline); numbers only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config.model_dump_json(indent=1) + "\n", encoding="utf-8")
    return path


def read_camera_config(path: Path) -> FineBioCameraConfig:
    return FineBioCameraConfig.model_validate_json(path.read_text(encoding="utf-8"))


def camera_from_fixed_config(camera: FineBioFixedCamera) -> Camera:
    return Camera(
        camera.view,
        np.array(camera.K, dtype=np.float64),
        np.array(camera.distortion, dtype=np.float64),
        np.array(camera.rvec, dtype=np.float64),
        np.array(camera.tvec, dtype=np.float64),
        tuple(camera.image_size),
    )


def cameras_from_config(config: FineBioCameraConfig) -> dict[str, Camera]:
    """The fixed cameras of a config by view (the fpv is per frame: `fpv_camera_from_config`)."""
    return {view: camera_from_fixed_config(cam) for view, cam in config.fixed.items()}


def fpv_camera_from_config(
    config: FineBioCameraConfig, frame: int, poses: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> Camera | None:
    """The fpv camera at raw frame `frame` from the config's intrinsics and `fpv_poses` arrays,
    or None when the shipped pose is missing for that frame."""
    rets, rots, trans = poses
    if frame < 0 or frame >= len(rets) or not rets[frame]:
        return None
    return Camera(
        "fpv",
        np.array(config.fpv.K, dtype=np.float64),
        np.array(config.fpv.distortion, dtype=np.float64),
        rots[frame],
        trans[frame],
        tuple(config.fpv.image_size),
    )
