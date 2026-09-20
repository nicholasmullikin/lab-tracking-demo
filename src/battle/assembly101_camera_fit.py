"""Fit a camera model per Assembly101 view from the dataset's own 2D/3D landmark pairs.

The archive ships camera-to-world extrinsics (constant for the eight static cameras, per
frame for the four head-mounted ones) but no intrinsics.  The shipped 2D landmarks are exact
projections of the shipped 3D landmarks, so `cv2.calibrateCamera` on frame groups of those
pairs recovers the projection model the dataset used: a five-coefficient Brown model for the
static views and OpenCV's eight-coefficient rational model for the ego views.  The result is
verified by re-projecting the 3D joints through the *shipped* extrinsics with the fitted
intrinsics; that residual is the number to trust.

Everything written here is an estimate of the dataset's internal projection, labelled as such.
It is not a physical calibration and no accuracy claim rests on it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from .assembly101_clock_offset import is_ego, npz_key, view_key
from .assembly101_fetch_view import EGO_VIEWS, STATIC_VIEWS
from .assembly101_pose_schemas import (
    BROWN_COEFFICIENTS,
    RATIONAL_COEFFICIENTS,
    Assembly101CameraModel,
)
from .assembly101_recordings import RECORDING_1, Assembly101Recording, get_recording

CONFIG_ROOT = Path(RECORDING_1.camera_config_root)
STATIC_RAW_SIZE = (1920, 1080)
EGO_RAW_SIZE = (636, 480)
FRAME_STEP = 50
CONFIDENCE_FLOOR = 0.8
MIN_POINTS_PER_FRAME = 8
STATIC_INITIAL_K = np.array([[1250.0, 0.0, 960.0], [0.0, 1250.0, 540.0], [0.0, 0.0, 1.0]])
EGO_INITIAL_K = np.array([[300.0, 0.0, 318.0], [0.0, 300.0, 240.0], [0.0, 0.0, 1.0]])
BROWN_FLAGS = cv2.CALIB_USE_INTRINSIC_GUESS
RATIONAL_FLAGS = (
    cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_ZERO_TANGENT_DIST
)
# A static view whose Brown fit leaves more than this through the shipped pose gets a rational
# refit as well; the better one is kept and the fallback is recorded in `fit_notes`.
BROWN_ACCEPT_RMS_PIXELS = 0.05


def camera_estimate_path(
    view: str,
    config_root: Path | None = None,
    *,
    recording: Assembly101Recording = RECORDING_1,
) -> Path:
    root = config_root if config_root is not None else Path(recording.camera_config_root)
    return root / f"{view.lower()}_camera_estimate.json"


def raw_image_size(view: str) -> tuple[int, int]:
    return EGO_RAW_SIZE if is_ego(view) else STATIC_RAW_SIZE


class PoseMembers:
    """The pose-archive members every fit needs, loaded once for all views."""

    def __init__(
        self, repository_root: Path, recording: Assembly101Recording = RECORDING_1
    ) -> None:
        self.recording = recording

        def member(kind: str) -> dict:
            return json.loads(
                (repository_root / recording.poses_member(kind)).read_text(encoding="utf-8")
            )

        self.landmarks3d = member("landmarks3D")
        self.confidences = member("hand_confidences")
        self.extrinsics_fixed = member("camera_extrinsics_fixed")
        self.extrinsics_ego = member("camera_extrinsics_ego")

    def camera_to_world(self, view: str, pose_frame: int) -> np.ndarray:
        key = view_key(view)
        if is_ego(view):
            return np.asarray(self.extrinsics_ego[str(pose_frame)][key], dtype=np.float64)
        return np.asarray(self.extrinsics_fixed[key], dtype=np.float64)


def gather_correspondences(
    view: str,
    members: PoseMembers,
    landmarks2d: np.ndarray,
    *,
    frame_step: int = FRAME_STEP,
    window_start_pose_frame: int | None = None,
    window_frame_count: int | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[int]]:
    """Frame groups of (3D world mm, 2D raw px, world->camera 4x4, pose frame)."""
    width, height = raw_image_size(view)
    start = (
        window_start_pose_frame
        if window_start_pose_frame is not None
        else members.recording.window_start_raw_frame
    )
    count = (
        window_frame_count
        if window_frame_count is not None
        else members.recording.window_raw_frame_count
    )
    objects: list[np.ndarray] = []
    images: list[np.ndarray] = []
    world_to_camera: list[np.ndarray] = []
    frames: list[int] = []
    for pose_frame in range(start, start + count, frame_step):
        row = pose_frame - start
        key = str(pose_frame)
        if key not in members.confidences or key not in members.landmarks3d:
            continue
        xyz: list[np.ndarray] = []
        uv: list[np.ndarray] = []
        for hand in (0, 1):
            if float(members.confidences[key][str(hand)]) < CONFIDENCE_FLOOR:
                continue
            xyz.append(np.asarray(members.landmarks3d[key][str(hand)], dtype=np.float64))
            uv.append(landmarks2d[row, hand])
        if not xyz:
            continue
        points = np.concatenate(xyz)
        pixels = np.concatenate(uv)
        inside = (
            np.isfinite(pixels).all(axis=1)
            & (pixels[:, 0] >= 0)
            & (pixels[:, 0] < width)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < height)
        )
        if inside.sum() < MIN_POINTS_PER_FRAME:
            continue
        objects.append(points[inside].reshape(-1, 1, 3).astype(np.float32))
        images.append(pixels[inside].reshape(-1, 1, 2).astype(np.float32))
        world_to_camera.append(np.linalg.inv(members.camera_to_world(view, pose_frame)))
        frames.append(pose_frame)
    return objects, images, world_to_camera, frames


def shipped_extrinsics_residuals(
    objects: list[np.ndarray],
    images: list[np.ndarray],
    world_to_camera: list[np.ndarray],
    intrinsic: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    """Per-point reprojection error using the dataset's own pose and the fitted intrinsics."""
    errors: list[np.ndarray] = []
    for xyz, uv, matrix in zip(objects, images, world_to_camera, strict=True):
        rotation, _ = cv2.Rodrigues(matrix[:3, :3])
        projected, _ = cv2.projectPoints(
            xyz.astype(np.float64), rotation, matrix[:3, 3], intrinsic, distortion
        )
        errors.append(np.linalg.norm(projected.reshape(-1, 2) - uv.reshape(-1, 2), axis=1))
    return np.concatenate(errors)


def _calibrate(
    objects: list[np.ndarray],
    images: list[np.ndarray],
    size: tuple[int, int],
    initial: np.ndarray,
    flags: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    rms, intrinsic, distortion, _, _ = cv2.calibrateCamera(
        objects, images, size, initial.copy(), None, flags=flags
    )
    return float(rms), np.asarray(intrinsic, dtype=np.float64), np.asarray(distortion).ravel()


def fit_view(
    view: str,
    members: PoseMembers,
    landmarks2d: np.ndarray,
) -> Assembly101CameraModel:
    """Fit one view; static views try Brown first and fall back to rational if it fails."""
    objects, images, world_to_camera, frames = gather_correspondences(view, members, landmarks2d)
    if len(objects) < 3:
        raise ValueError(f"{view}: only {len(objects)} usable frame groups; cannot fit")
    size = raw_image_size(view)
    ego = is_ego(view)
    attempts: list[tuple[Literal["brown", "rational"], float, np.ndarray, np.ndarray, np.ndarray]]
    attempts = []
    if ego:
        rms, intrinsic, distortion = _calibrate(
            objects, images, size, EGO_INITIAL_K, RATIONAL_FLAGS
        )
        distortion = distortion[:RATIONAL_COEFFICIENTS]
        residual = shipped_extrinsics_residuals(
            objects, images, world_to_camera, intrinsic, distortion
        )
        attempts.append(("rational", rms, intrinsic, distortion, residual))
    else:
        rms, intrinsic, distortion = _calibrate(
            objects, images, size, STATIC_INITIAL_K, BROWN_FLAGS
        )
        distortion = distortion[:BROWN_COEFFICIENTS]
        residual = shipped_extrinsics_residuals(
            objects, images, world_to_camera, intrinsic, distortion
        )
        attempts.append(("brown", rms, intrinsic, distortion, residual))
        if float(np.sqrt(np.mean(residual**2))) > BROWN_ACCEPT_RMS_PIXELS:
            rms, intrinsic, distortion = _calibrate(
                objects, images, size, STATIC_INITIAL_K, RATIONAL_FLAGS
            )
            distortion = distortion[:RATIONAL_COEFFICIENTS]
            residual = shipped_extrinsics_residuals(
                objects, images, world_to_camera, intrinsic, distortion
            )
            attempts.append(("rational", rms, intrinsic, distortion, residual))
    best = min(attempts, key=lambda item: float(np.sqrt(np.mean(item[4] ** 2))))
    model_name, rms, intrinsic, distortion, residual = best
    window_start = members.recording.window_start_raw_frame
    window_end = window_start + members.recording.window_raw_frame_count
    notes = (
        f"{len(frames)} frame groups every {FRAME_STEP} pose frames in "
        f"[{window_start}, {window_end}), hands "
        f"with confidence >= {CONFIDENCE_FLOOR}, joints inside the image; "
        f"cv2.calibrateCamera {model_name}"
        + (" with per-frame camera_extrinsics_ego" if ego else " with camera_extrinsics_fixed")
    )
    if len(attempts) > 1:
        rejected = ", ".join(
            f"{name} {np.sqrt(np.mean(res**2)):.4f} px" for name, _, _, _, res in attempts
        )
        notes += f"; Brown exceeded {BROWN_ACCEPT_RMS_PIXELS} px so both were tried ({rejected})"
    camera_to_world = None if ego else members.camera_to_world(view, frames[0])
    return Assembly101CameraModel(
        view_key=view_key(view),
        raw_image_size=size,
        intrinsic_matrix=tuple(tuple(float(v) for v in row) for row in intrinsic),
        distortion=tuple(float(v) for v in distortion),
        distortion_model=model_name,
        extrinsics_kind="per_frame_ego" if ego else "fixed",
        camera_to_world=(
            None
            if camera_to_world is None
            else tuple(tuple(float(v) for v in row) for row in camera_to_world)
        ),
        provenance="estimated_from_dataset_landmark_projection",
        fit_rms_pixels=rms,
        fit_point_count=int(residual.size),
        fit_frame_count=len(frames),
        shipped_extrinsics_rms_pixels=float(np.sqrt(np.mean(residual**2))),
        shipped_extrinsics_max_pixels=float(residual.max()),
        fit_notes=notes,
    )


def load_window_landmarks(
    repository_root: Path, view: str, recording: Assembly101Recording = RECORDING_1
) -> np.ndarray:
    with np.load(repository_root / recording.shipped_2d_window) as archive:
        return np.asarray(archive[npz_key(view)], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", action="append", help="e.g. C10095 or HMC_21179183; repeatable")
    parser.add_argument("--all", action="store_true", help="fit every static and ego view")
    parser.add_argument(
        "--recording",
        help="Registry label or recording id (configs/assembly101/recordings.json); "
        "default: recording 1.",
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config-root", type=Path, help="default: the recording's camera_config_root"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing estimate (the C10379 file is fingerprinted by built runs).",
    )
    args = parser.parse_args()
    recording = get_recording(args.recording, args.repository_root)
    views = list(args.view or [])
    if args.all:
        views = (
            [*STATIC_VIEWS, *EGO_VIEWS]
            if recording.recording_id == RECORDING_1.recording_id
            else list(recording.all_views)
        )
    if not views:
        parser.error("pass --view or --all")
    repository_root = args.repository_root.resolve()
    started = time.monotonic()
    members = PoseMembers(repository_root, recording)
    print(f"loaded pose members in {time.monotonic() - started:.1f} s")
    for view in views:
        output = repository_root / camera_estimate_path(view, args.config_root, recording=recording)
        if output.exists() and not args.overwrite:
            print(f"{view}: {output} exists; skipped (pass --overwrite)")
            continue
        view_started = time.monotonic()
        model = fit_view(view, members, load_window_landmarks(repository_root, view, recording))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(model.model_dump_json(indent=2) + "\n", encoding="utf-8")
        k = model.intrinsic_matrix
        print(
            f"{view}: {model.distortion_model} fx={k[0][0]:.2f} fy={k[1][1]:.2f} "
            f"cx={k[0][2]:.2f} cy={k[1][2]:.2f}; calib RMS {model.fit_rms_pixels:.5f} px; "
            f"shipped-extrinsics RMS {model.shipped_extrinsics_rms_pixels:.5f} px "
            f"(max {model.shipped_extrinsics_max_pixels:.4f}) over {model.fit_point_count} "
            f"points / {model.fit_frame_count} frames; {time.monotonic() - view_started:.1f} s"
        )


if __name__ == "__main__":
    main()
