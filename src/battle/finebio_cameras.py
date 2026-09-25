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


# --------------------------------------------------------------------------- per-trial config


def _mat3(K: np.ndarray) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(float(x) for x in row) for row in np.asarray(K, dtype=np.float64))


def _vec(values: np.ndarray) -> tuple[float, ...]:
    return tuple(float(x) for x in np.asarray(values, dtype=np.float64).ravel())


def camera_config_from_mapping(
    mapping: dict[str, Any],
    trial: str,
    *,
    pnp_over_px: float = SHIPPED_POSE_MAX_RMS_PX,
    provenance: dict[str, Any] | None = None,
) -> FineBioCameraConfig:
    """The trial's camera config from a preflight ``mapping.json`` report: `rig_cameras`
    decides shipped vs marker PnP per fixed view; the fpv gets the rescaled intrinsics, its
    pose file and the default validity gate parameters. Raw frames, offset 0."""
    day = mapping["day_decision"]["chosen"]
    cams, pose_provenance = rig_cameras(mapping, pnp_over_px=pnp_over_px)
    fixed: dict[str, FineBioFixedCamera] = {}
    for view in FIXED_VIEWS:
        info = mapping["views"][view]
        cam = cams[view]
        shipped_rms = info["best"]["median_corner_rms_px"]
        in_use_rms = (
            info.get("pnp_corner_rms_px") if pose_provenance[view] == "marker_pnp" else shipped_rms
        )
        fixed[view] = FineBioFixedCamera(
            view=view,
            camera_id=int(info["best"]["camera_id"]),
            provenance=pose_provenance[view],
            K=_mat3(cam.K),
            distortion=_vec(cam.dist),
            rvec=_vec(cam.rvec),
            tvec=_vec(cam.tvec),
            image_size=cam.size,
            marker_fit_residual_px=None if in_use_rms is None else float(in_use_rms),
            shipped_marker_residual_px=float(shipped_rms),
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
        recording_day=str(day),
        fixed=fixed,
        fpv=fpv,
        provenance={
            "mapping_trial": mapping.get("trial"),
            "mapping_seconds": mapping.get("seconds"),
            "shipped_pose_max_rms_px": pnp_over_px,
            "intrinsics_rescale": {"fixed": 0.5, "fpv": 0.48},
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
