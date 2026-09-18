"""battle-kineo-multiview: Kineo on all eight static Assembly101 views, two arms.

Track 4 of the overnight multicam pass.  `prepare` is CPU-only and writes everything a queued
GPU job needs: per-view first-minute trims with identical frame counts (the measured per-view
clock offsets absorbed by shifting each view's start frame), a headless Kineo YAML generated
from `configs/demo/offline/nlf_single_person_sam2.yaml` (rtmlib `best_bbox_only` replaces the
interactive SAM2 stage, `shared_intrinsics: false`), the dataset cameras as Kineo annotation
PKLs, a queue job for `scripts/overnight_queue.py`, and a typed `prepare.json`.  The job itself
runs `scripts/kineo_multiview_runner.py` under Kineo's pixi environment; the runner is the
only place that touches the GPU and it is never called from here.

Two arms:

* `selfcal`: Kineo's full offline stage list (SfM initialisation, three bundle-adjustment
  passes, triangulation, SMPL scale, reorientation, SMPL fitting, BVH, Rerun).  The dataset
  extrinsics enter the run only as `gt_annotations`, which the estimation stages never read
  (only `rerun_export` uses them, for its own visual alignment).  `evaluate` similarity-aligns
  Kineo's estimated camera centres to the dataset's (Umeyama, with scale) and reports per-camera
  rotation and translation error after alignment plus the recovered scale.  This is a
  calibration comparison against dataset context, not pose accuracy.
* `known`: `transfer_gt_annotations` injects our fitted intrinsics (scaled to the 1280x720
  proxies) and the inverted dataset camera-to-world poses as world-to-camera OpenCV `R, t`
  (metres), and the SfM/BA/scale/reorientation stages are removed, so Kineo only detects and
  triangulates.  `evaluate` compares the triangulated body wrists with the dataset hand wrists
  directly (x1000, no alignment).

Both wrist numbers are cross-source disagreement between two estimates.  Ego cameras are
excluded on purpose: Kineo assumes static cameras.  Assembly101 is CC BY-NC 4.0; Kineo is
research/evaluation-only per its upstream licence.
"""

from __future__ import annotations

import argparse
import builtins
import enum
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import Field

from . import assembly101_reference as a101
from .assembly101_fetch_view import RECORDING_ID, STATIC_VIEWS, proxy_path
from .assembly101_pose_schemas import (
    ASSEMBLY101_JOINT_NAMES,
    Assembly101ClockRule,
    Assembly101HandFrame,
)
from .kineo_nlf import (
    DEFAULT_KINEO_REVISION,
    DEFAULT_KINEO_ROOT,
    MOGE_MODEL,
    NLF_CHECKPOINT,
    RTMLIB_MODEL_URL,
    _git_fingerprint,
)
from .multiview_geometry import CameraRig
from .schemas import ArtifactFingerprint, VersionedModel

Arm = Literal["selfcal", "known"]
ARMS: tuple[Arm, ...] = ("selfcal", "known")

DEFAULT_FRAME_COUNT = 1800
ANALYSIS_FPS = 30
PROXY_RESOLUTION_WH = (1280, 720)
RAW_RESOLUTION_WH = (1920, 1080)
DATASET_REFERENCE_RUN = Path("runs/assembly101-reference-first-minute-v1")
RUN_ROOT_TEMPLATE = "runs/kineo-multiview-{arm}-first-minute-20260918"
QUEUE_DIR = Path("runs/overnight-multicam-20260918")
QUEUE_LOG = QUEUE_DIR / "queue.log"
RUNNER_SCRIPT = Path("scripts/kineo_multiview_runner.py")
KINEO_STOCK_CONFIG = Path("configs/demo/offline/nlf_single_person_sam2.yaml")
TIMEOUT_S: dict[Arm, int] = {"selfcal": 3600, "known": 1800}
DATASET_WRIST = ASSEMBLY101_JOINT_NAMES.index("wrist")
SIDES: tuple[str, ...] = ("left", "right")
KINEO_WRIST_NAMES: dict[str, str] = {"left": "left_wrist", "right": "right_wrist"}
SMPLX_BODY_JOINTS = 55
MODEL_FREE_STAGES: tuple[str, ...] = (
    "TransferGroundTruthAnnotationsStage",
    "GlobalTimeResamplingStage",
    "KeypointsPairsSamplingStage",
    "SfMCameraExtrinsicsInitializationStage",
    "BundleAdjustmentSamplingStage",
    "BundleAdjustmentStage",
    "MVSTriangulationStage",
    "SMPLGlobalScaleEstimationStage",
    "GlobalScaleApplicationStage",
    "SceneReorientationStage",
    "BundleAdjustmentHistoryRerunExportStage",
    "ExportBvhStage",
    "RerunExportStage",
    "AnnotationsExportStage",
)
CLAIM_BOUNDARIES: tuple[str, ...] = (
    "Self-calibration arm: Kineo's estimated extrinsics are compared with the dataset's shipped "
    "extrinsics after a similarity alignment of camera centres; this is a calibration comparison "
    "against dataset context, not pose accuracy, and the dataset extrinsics are the only "
    "reference used.",
    "Wrist numbers compare Kineo's body wrists (NLF 2D through Kineo's own or the dataset's "
    "cameras) with the dataset's hand-tracker wrists: cross-source disagreement between two "
    "estimates, never accuracy of either.",
    "Known-camera arm: the intrinsics are fitted estimates of the dataset's own projection and "
    "the extrinsics are the dataset's; Kineo only detects and triangulates.",
    "Views are aligned by shifting each trim's start frame by the measured per-view clock "
    "offset rounded to whole 30 fps frames; the residual is at most one 60 fps pose frame "
    "(16.7 ms) per view and nothing is resampled.  Ego cameras are excluded (Kineo assumes "
    "static cameras).",
    "Assembly101 is CC BY-NC 4.0 (attribution required, non-commercial); Kineo is licensed for "
    "research and evaluation only and its checkout was dirty (fingerprinted in prepare.json).",
)

# -- geometry helpers --------------------------------------------------------------------------


def scale_intrinsics(intrinsic_matrix: np.ndarray, scale: float) -> np.ndarray:
    """K for an image resampled by `scale` (fx, fy, cx, cy scale linearly; distortion is free)."""
    scaled = np.array(intrinsic_matrix, dtype=np.float64)
    scaled[0, :] *= scale
    scaled[1, :] *= scale
    return scaled


def world_to_camera(camera_to_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV world-to-camera `(R, t)` with `X_cam = R X_world + t` from a 4x4 camera-to-world.

    The dataset's rotation blocks are orthonormal only to JSON precision (~1e-7), so the
    nearest proper rotation is taken first; Kineo inverts `[R | t]` by transposition and would
    otherwise move the camera centre by ~1e-4 mm.
    """
    pose = np.asarray(camera_to_world, dtype=np.float64)
    u, _, vt = np.linalg.svd(pose[:3, :3])
    rotation_c2w = u @ vt
    if np.linalg.det(rotation_c2w) < 0:
        rotation_c2w = u @ np.diag([1.0, 1.0, -1.0]) @ vt
    rotation = rotation_c2w.T
    translation = -rotation @ pose[:3, 3]
    return rotation, translation


def camera_center(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Camera position in world coordinates from world-to-camera `(R, t)`."""
    return -np.asarray(rotation).T @ np.asarray(translation)


def umeyama(
    source: np.ndarray, target: np.ndarray, *, with_scale: bool = True
) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity `target ~ s R source + t` (Umeyama 1991), no reflection."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("umeyama needs matching (N, 3) point sets")
    if source.shape[0] < 3:
        raise ValueError("umeyama needs at least three correspondences")
    mean_source = source.mean(axis=0)
    mean_target = target.mean(axis=0)
    centred_source = source - mean_source
    centred_target = target - mean_target
    covariance = centred_target.T @ centred_source / source.shape[0]
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2, 2] = -1.0
    rotation = u @ sign @ vt
    if with_scale:
        variance_source = np.mean(np.sum(centred_source**2, axis=1))
        scale = float(np.trace(np.diag(singular) @ sign) / variance_source)
    else:
        scale = 1.0
    translation = mean_target - scale * rotation @ mean_source
    return scale, rotation, translation


def apply_similarity(
    points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    return scale * np.asarray(points, dtype=np.float64) @ np.asarray(rotation).T + translation


def rotation_angle_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    """Geodesic angle between two rotation matrices in degrees."""
    relative = np.asarray(rotation_a).T @ np.asarray(rotation_b)
    cosine = (np.trace(relative) - 1.0) / 2.0
    axis = np.array(
        [
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ]
    )
    sine = np.linalg.norm(axis) / 2.0
    # atan2 keeps full precision near zero where arccos((tr - 1) / 2) loses it.
    return float(np.degrees(np.arctan2(sine, cosine)))


# -- clock alignment ---------------------------------------------------------------------------


class ViewStart(VersionedModel):
    """Where a view's trim starts so that trimmed frame q shows the same instant in every view."""

    view: str
    pose_offset_frames: int
    shift_pose_frames: int
    start_frame: int = Field(ge=0)
    residual_pose_frames: float
    residual_seconds: float


def view_start_frames(
    rules: Mapping[str, Assembly101ClockRule], views: Sequence[str]
) -> tuple[str, dict[str, ViewStart]]:
    """Per-view start frames that absorb the measured clock offsets.

    The reference is the view whose video started latest (largest offset: its frame 0 shows the
    latest scene instant), so every other view skips `round(shift / raw_frames_per_proxy_frame)`
    frames where `shift = reference.pose_frame(0) - view.pose_frame(0)` in 60 fps pose frames.
    Halves round up (`floor(x + 0.5)`), matching the nearest-frame rule of Track 3b; the
    residual per view is reported in pose frames and seconds and never exceeds half an
    analysis frame.
    """
    if not views:
        raise ValueError("at least one view is required")
    reference = max(views, key=lambda view: rules[view].pose_frame(0))
    reference_pose = rules[reference].pose_frame(0)
    starts: dict[str, ViewStart] = {}
    for view in views:
        rule = rules[view]
        shift = reference_pose - rule.pose_frame(0)
        if shift < 0:
            raise ValueError("reference must be the latest-starting view")
        start = int(np.floor(shift / rule.raw_frames_per_proxy_frame + 0.5))
        residual = shift - start * rule.raw_frames_per_proxy_frame
        starts[view] = ViewStart(
            view=view,
            pose_offset_frames=rule.pose_offset_frames,
            shift_pose_frames=shift,
            start_frame=start,
            residual_pose_frames=float(residual),
            residual_seconds=float(residual / rule.pose_fps),
        )
    return reference, starts


# -- Kineo annotation dictionaries -------------------------------------------------------------


def intrinsics_annotation_dict(
    view: str,
    intrinsic_matrix: np.ndarray,
    distortion: Sequence[float],
    resolution_hw: tuple[int, int],
) -> dict[str, Any]:
    coefficients = [float(v) for v in distortion]
    if len(coefficients) != 5:
        raise ValueError("Kineo's brown_conrady model takes exactly five coefficients")
    return {
        "view_id": view,
        "frame_idx": 0,
        "K": np.asarray(intrinsic_matrix, dtype=np.float64).tolist(),
        "distortion_coefficients": coefficients,
        "distortion_model": "brown_conrady",
        "resolution_hw": [int(resolution_hw[0]), int(resolution_hw[1])],
    }


def extrinsics_annotation_dict(view: str, rotation: np.ndarray, translation: np.ndarray) -> dict:
    return {
        "view_id": view,
        "frame_idx": 0,
        "R": np.asarray(rotation, dtype=np.float64).tolist(),
        "t": np.asarray(translation, dtype=np.float64).tolist(),
    }


def build_gt_annotations(
    rig: CameraRig, views: Sequence[str], *, image_scale: float, translation_scale: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Kineo `camera_intrinsics` / `camera_extrinsics` dicts (as `Annotations.to_dict()` writes
    them, with the distortion model as its string value) for the given static views.

    `image_scale` maps raw sensor pixels to the proxy (2/3 for 1920x1080 -> 1280x720);
    `translation_scale` maps dataset millimetres to Kineo's world unit (1e-3 for metres).
    """
    intrinsics = []
    extrinsics = []
    for view in views:
        camera = rig.camera(view)
        if camera.is_ego:
            raise ValueError(f"{view} is an ego camera; Kineo assumes static cameras")
        if camera.distortion_model != "brown":
            raise ValueError(f"{view} is not a Brown camera")
        width, height = camera.raw_image_size
        scaled = scale_intrinsics(np.asarray(camera.intrinsic_matrix), image_scale)
        resolution_hw = (round(height * image_scale), round(width * image_scale))
        intrinsics.append(
            intrinsics_annotation_dict(view, scaled, camera.distortion[:5], resolution_hw)
        )
        rotation, translation = world_to_camera(rig.camera_to_world(view))
        extrinsics.append(
            extrinsics_annotation_dict(view, rotation, translation * translation_scale)
        )
    metadata = {"version": "1"}
    return (
        {"metadata": metadata, "annotations": intrinsics},
        {"metadata": metadata, "annotations": extrinsics},
    )


# -- Kineo PKL reading -------------------------------------------------------------------------


class KineoDistortionModel(enum.Enum):
    """Stand-in for `kineo.annotations.camera_intrinsics.CameraDistortionModel`."""

    BROWN_CONRADY = "brown_conrady"
    OPENCV_FISHEYE = "opencv_fisheye"


_SAFE_BUILTINS = {"dict", "list", "tuple", "set", "frozenset", "int", "float", "str", "bool"}


class _KineoUnpickler(pickle.Unpickler):
    """Unpickle Kineo's exported annotation dicts without importing Kineo.

    The dicts hold lists and scalars (`Annotations.to_dict()` calls `tolist()`), plus the
    distortion-model enum, which is mapped onto a local stand-in.  Anything else is refused.
    """

    def find_class(self, module: str, name: str) -> Any:
        if module == "builtins" and name in _SAFE_BUILTINS:
            return getattr(builtins, name)
        if module == "kineo.annotations.camera_intrinsics" and name == "CameraDistortionModel":
            return KineoDistortionModel
        raise pickle.UnpicklingError(f"refusing to unpickle {module}.{name}")


def load_kineo_pkl(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return _KineoUnpickler(handle).load()


def parse_extrinsics(data: Mapping[str, Any]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """`view_id -> (R, t)` world-to-camera; static cameras carry one row per view."""
    cameras: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for row in data["annotations"]:
        view = str(row["view_id"])
        if view in cameras:
            raise ValueError(f"{view} has more than one extrinsics row; Kineo output is not static")
        rotation = np.asarray(row["R"], dtype=np.float64)
        translation = np.asarray(row["t"], dtype=np.float64)
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError(f"malformed extrinsics row for {view}")
        cameras[view] = (rotation, translation)
    return cameras


def parse_intrinsics(data: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    cameras: dict[str, dict[str, Any]] = {}
    for row in data["annotations"]:
        view = str(row["view_id"])
        model = row["distortion_model"]
        cameras[view] = {
            "K": np.asarray(row["K"], dtype=np.float64),
            "distortion_coefficients": np.asarray(row["distortion_coefficients"], dtype=np.float64),
            "distortion_model": model.value if isinstance(model, enum.Enum) else str(model),
            "resolution_hw": tuple(int(v) for v in row["resolution_hw"]),
        }
    return cameras


class Keypoints3D:
    """Kineo `keypoints_3d` for one subject as dense arrays over the global frame index."""

    def __init__(
        self,
        frames: np.ndarray,
        xyz: np.ndarray,
        scores: np.ndarray,
        names: tuple[str, ...],
        edges: tuple[tuple[int, int], ...],
        subject_id: str,
        subject_ids: tuple[str, ...],
    ) -> None:
        self.frames = frames
        self.xyz = xyz
        self.scores = scores
        self.names = names
        self.edges = edges
        self.subject_id = subject_id
        self.subject_ids = subject_ids


def parse_keypoints_3d(
    data: Mapping[str, Any], *, joint_limit: int = SMPLX_BODY_JOINTS, subject_id: str | None = None
) -> Keypoints3D:
    """Dense `(F, J, 3)` body joints for one subject (`joint_limit` keeps the 55 SMPL-X body
    joints and drops NLF's 1024 surface vertices)."""
    formats = data["metadata"]["formats"]
    if not formats:
        raise ValueError("keypoints_3d metadata carries no format")
    names = tuple(str(n) for n in formats[0]["keypoints_names"][:joint_limit])
    edges = tuple(
        (int(a), int(b))
        for a, b in formats[0].get("keypoints_connectivity", [])
        if int(a) < joint_limit and int(b) < joint_limit
    )
    rows = data["annotations"]
    subject_ids = tuple(sorted({str(row["subject_id"]) for row in rows}))
    if not subject_ids:
        raise ValueError("keypoints_3d carries no annotations")
    chosen = subject_id or subject_ids[0]
    rows = [row for row in rows if str(row["subject_id"]) == chosen]
    frame_indices = sorted({int(row["frame_idx"]) for row in rows})
    frame_count = frame_indices[-1] + 1
    xyz = np.full((frame_count, len(names), 3), np.nan)
    scores = np.zeros((frame_count, len(names)))
    for row in rows:
        frame = int(row["frame_idx"])
        xyz[frame] = np.asarray(row["xyz"], dtype=np.float64)[: len(names)]
        scores[frame] = np.asarray(row["scores"], dtype=np.float64)[: len(names)]
    return Keypoints3D(np.arange(frame_count), xyz, scores, names, edges, chosen, subject_ids)


def parse_stage_timings(data: Mapping[str, Any]) -> tuple[StageTiming, ...]:
    return tuple(
        StageTiming(
            stage_name=str(row["stage_name"]),
            stage_idx=int(row["stage_idx"]),
            duration_seconds=float(row["duration_seconds"]),
        )
        for row in data["annotations"]
    )


# -- manifests ---------------------------------------------------------------------------------


class KineoGitState(VersionedModel):
    head: str = Field(min_length=7)
    dirty: bool
    status_short: str
    diff_stat: str
    dirty_fingerprint_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class ViewTrim(VersionedModel):
    view: str
    source_proxy: ArtifactFingerprint
    trimmed: ArtifactFingerprint
    start: ViewStart
    frame_count: int = Field(ge=1)
    verified_frame_count: int = Field(ge=1)


class PrepareManifest(VersionedModel):
    manifest_kind: Literal["kineo_multiview_prepare"] = "kineo_multiview_prepare"
    arm: Arm
    run_id: str = Field(min_length=1)
    recording_id: str = Field(min_length=1)
    prepared_at_utc: str
    sequence_name: str = Field(min_length=1)
    views: tuple[str, ...] = Field(min_length=2, description="Kineo view ids in argv order")
    reference_view: str
    frame_count: int = Field(ge=1)
    analysis_fps: Literal[30] = 30
    offset_handling: str
    trims: tuple[ViewTrim, ...]
    kineo_config: ArtifactFingerprint
    kineo_stages: tuple[str, ...]
    removed_stock_stages: tuple[str, ...]
    gt_camera_intrinsics: ArtifactFingerprint
    gt_camera_extrinsics: ArtifactFingerprint
    gt_annotations_role: str
    intrinsics_image_scale: float
    extrinsics_convention: str
    extrinsics_translation_unit: Literal["metres"]
    camera_estimates: tuple[ArtifactFingerprint, ...]
    clock_rules: ArtifactFingerprint
    runner_script: ArtifactFingerprint
    kineo_repository: str
    kineo_pinned_revision: str
    kineo_git: KineoGitState
    model_identities: dict[str, Any]
    kineo_output_root: str
    kineo_cache_root: str
    queue_job: str
    queue_timeout_s: int
    claim_boundaries: tuple[str, ...] = Field(min_length=1)


class StageTiming(VersionedModel):
    stage_name: str
    stage_idx: int
    duration_seconds: float


class SimilarityAlignment(VersionedModel):
    method: str
    scale_mm_per_kineo_unit: float
    rotation: tuple[tuple[float, float, float], ...] = Field(min_length=3, max_length=3)
    translation_mm: tuple[float, float, float]
    camera_center_rms_mm: float


class CameraComparison(VersionedModel):
    view: str
    rotation_error_deg: float
    translation_error_mm: float
    kineo_center_aligned_mm: tuple[float, float, float]
    dataset_center_mm: tuple[float, float, float]


class WristComparison(VersionedModel):
    side: str
    frames_compared: int = Field(ge=0)
    median_mm: float | None
    p90_mm: float | None
    mean_mm: float | None
    swapped_side_closer_fraction: float | None


class PreAccuracyMeasures(VersionedModel):
    """The five measures every track reports before any accuracy talk."""

    coverage_frames_with_body_3d: int = Field(ge=0)
    coverage_frames_with_both_wrists: int = Field(ge=0)
    frame_count: int = Field(ge=1)
    time_to_first_3d_output_seconds: float | None
    runtime_seconds: float | None
    kineo_pipeline_seconds: float | None
    gpu_peak_vram_bytes: int | None
    id_resets: int | None
    id_resets_note: str
    notes: tuple[str, ...] = ()


class QueueRecord(VersionedModel):
    job: str
    state: str
    exit_code: int | None
    duration_s: float
    stdout_log: str | None
    time: str


class KineoMultiviewManifest(VersionedModel):
    manifest_kind: Literal["kineo_multiview_evaluation"] = "kineo_multiview_evaluation"
    arm: Arm
    run_id: str
    evaluated_at_utc: str
    prepare: ArtifactFingerprint
    views: tuple[str, ...]
    frame_count: int
    kineo_native_artifacts: tuple[ArtifactFingerprint, ...]
    alignment: SimilarityAlignment
    cameras: tuple[CameraComparison, ...]
    camera_rotation_error_deg_median: float
    camera_translation_error_mm_median: float
    kineo_global_scale: float | None
    wrists: tuple[WristComparison, ...]
    body_joint_names: tuple[str, ...]
    stage_timings: tuple[StageTiming, ...]
    queue_record: QueueRecord | None
    measures: PreAccuracyMeasures
    claim_boundaries: tuple[str, ...] = Field(min_length=1)


# -- YAML generation ---------------------------------------------------------------------------


def _fingerprint(path: Path, repository_root: Path) -> ArtifactFingerprint:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    resolved = path.resolve()
    root = repository_root.resolve()
    try:
        uri = resolved.relative_to(root).as_posix()
    except ValueError:
        uri = resolved.as_posix()
    return ArtifactFingerprint(uri=uri, sha256=digest.hexdigest(), source="measured")


def _rtmlib_stage(frame_step: int) -> dict[str, Any]:
    return {
        "_target_": "kineo.pipeline.stages.rtmlib.rtmlib_bbox_detection.RtmlibBboxDetectionStage",
        "name": "Rtmlib Bbox Detection",
        "order": 5,
        "bbox_model": RTMLIB_MODEL_URL,
        "bbox_model_input_shape_hw": [416, 416],
        "runtime_cfg": {
            "_target_": (
                "kineo.pipeline.stages.rtmlib.rtmlib_bbox_detection.RtmlibBboxDetectionRuntimeConfig"
            ),
            "use_cache": "${use_cache}",
            "cache_output_path_template": "${cache_root_dir}/{sequence_name}/{annotation_key}.pkl",
            "show": False,
            "frame_step": frame_step,
            "best_bbox_only": True,
        },
    }


def _transfer_stage() -> dict[str, Any]:
    return {
        "_target_": (
            "kineo.pipeline.stages.transfer_gt_annotations.TransferGroundTruthAnnotationsStage"
        ),
        "name": "Transfer dataset cameras",
        "order": 0,
        "runtime_cfg": {
            "_target_": (
                "kineo.pipeline.stages.transfer_gt_annotations."
                "TransferGroundTruthAnnotationsRuntimeConfig"
            ),
            "annotations_keys": ["camera_intrinsics", "camera_extrinsics"],
            "raise_error_if_not_found": True,
            "overwrite": True,
        },
    }


SELFCAL_REMOVED = ("sam2_semiauto_bbox_detection_rtmlib",)
KNOWN_REMOVED = (
    "sam2_semiauto_bbox_detection_rtmlib",
    "moge_intrinsics_estimation",
    "keypoints_pairs_sampling",
    "sfm_camera_extrinsics_initialization",
    "bundle_adjustment_sampling",
    "bundle_adjustment_1",
    "bundle_adjustment_2",
    "bundle_adjustment_3",
    "smpl_global_scale_estimation",
    "global_scale_application",
    "scene_reorientation",
    "bundle_adjustment_history_rerun_export",
    "smpl_fitting",
    "background_subtraction",
    "moge_scene_reconstruction",
    "export_bvh",
)


def generate_kineo_config(
    stock: Mapping[str, Any],
    *,
    arm: Arm,
    output_root_dir: str,
    cache_root_dir: str,
    rtmlib_frame_step: int = 5,
    target_fps: int = ANALYSIS_FPS,
) -> dict[str, Any]:
    """Battle's headless multi-view YAML derived from Kineo's stock offline demo config.

    The stock file is read, never edited: the SAM2 GUI stage is replaced by rtmlib
    `best_bbox_only`, `shared_intrinsics` is false, the time-resampling target is the proxy
    rate, Rerun/annotation exports point into the run directory and log the dataset cameras,
    and (known arm) `transfer_gt_annotations` replaces every camera-estimation stage.
    """
    stages = json.loads(json.dumps(dict(stock["pipeline"]["stages"])))
    removed = SELFCAL_REMOVED if arm == "selfcal" else KNOWN_REMOVED
    for name in removed:
        if name not in stages:
            raise KeyError(f"stock Kineo config lacks stage {name}")
        del stages[name]
    stages["rtmlib_bbox_detection"] = _rtmlib_stage(rtmlib_frame_step)
    if arm == "known":
        stages["transfer_gt_annotations"] = _transfer_stage()
    stages["global_time_resampling"]["runtime_cfg"]["target_fps"] = target_fps
    if "bundle_adjustment_history_rerun_export" in stages:
        stages["bundle_adjustment_history_rerun_export"]["runtime_cfg"]["output_path_template"] = (
            "${output_root_dir}/{sequence_name}_ba_history.rrd"
        )
    if "export_bvh" in stages:
        stages["export_bvh"]["runtime_cfg"]["output_path_template"] = (
            "${output_root_dir}/{sequence_name}.bvh"
        )
    rerun = stages["rerun_export"]["runtime_cfg"]
    rerun["output_path_template"] = "${output_root_dir}/{sequence_name}.rrd"
    rerun["log_gt_cameras"] = True
    if arm == "known":
        rerun["log_pred_smpl"] = False
        rerun["log_pred_smpl_skeleton_2d"] = False
        rerun["log_world_reconstruction"] = False
    stages["annotations_export"]["runtime_cfg"]["output_path_template"] = (
        "${output_root_dir}/annotations/{sequence_name}/{annotation_key}.pkl"
    )
    stages["annotations_export"]["runtime_cfg"]["not_found_error"] = False
    ordered = dict(sorted(stages.items(), key=lambda item: (item[1]["order"], item[0])))
    config: dict[str, Any] = {
        "output_root_dir": output_root_dir,
        "cache_root_dir": cache_root_dir,
        "use_cache": False,
        "batch_size": int(stock.get("batch_size", 32)),
        "use_half_precision": bool(stock.get("use_half_precision", True)),
        "shared_intrinsics": False,
        "rtmlib_bbox_detection_frame_step": rtmlib_frame_step,
    }
    if arm == "selfcal":
        config["smplx_joints_indices"] = list(stock["smplx_joints_indices"])
    config["pipeline"] = {"seed": int(stock["pipeline"]["seed"]), "stages": ordered}
    return config


def dump_yaml(config: Mapping[str, Any]) -> str:
    return yaml.safe_dump(dict(config), sort_keys=False, width=100, default_flow_style=False)


def stage_names(config: Mapping[str, Any]) -> tuple[str, ...]:
    stages = config["pipeline"]["stages"]
    return tuple(sorted(stages, key=lambda name: (stages[name]["order"], name)))


# -- trims -------------------------------------------------------------------------------------


def trim_command(source: Path, output: Path, start_frame: int, frame_count: int) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vf",
        f"trim=start_frame={start_frame}:end_frame={start_frame + frame_count},setpts=PTS-STARTPTS",
        "-frames:v",
        str(frame_count),
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(ANALYSIS_FPS),
        str(output),
    ]


def count_frames(video: Path) -> int:
    """Decoded frame count (what Kineo's `VideoLoader` and `ffprobe -count_frames` agree on)."""
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(completed.stdout.strip())


def _trim_view(
    source: Path, output: Path, start_frame: int, frame_count: int, *, overwrite: bool
) -> int:
    if not output.exists() or overwrite:
        subprocess.run(trim_command(source, output, start_frame, frame_count), check=True)
    verified = count_frames(output)
    if verified != frame_count:
        raise ValueError(f"{output.name}: {verified} frames, expected {frame_count}")
    return verified


# -- prepare -----------------------------------------------------------------------------------


def run_id_for(arm: Arm) -> str:
    return Path(RUN_ROOT_TEMPLATE.format(arm=arm)).name


def runner_argv(
    *,
    repository_root: Path,
    run_directory: Path,
    config_path: Path,
    sequence_name: str,
    gt_directory: Path,
    runtime_json: Path,
    views: Sequence[str],
    trims: Mapping[str, Path],
) -> list[str]:
    argv = [
        "pixi",
        "run",
        "python",
        str((repository_root / RUNNER_SCRIPT).resolve()),
        "--config-file",
        str(config_path.resolve()),
        "--sequence-name",
        sequence_name,
        "--gt-annotations-dir",
        str(gt_directory.resolve()),
        "--runtime-json",
        str(runtime_json.resolve()),
    ]
    argv.extend(f"{view}={trims[view].resolve()}" for view in views)
    return argv


def queue_job(
    *, name: str, argv: Sequence[str], kineo_root: Path, timeout_s: int
) -> dict[str, Any]:
    return {
        "jobs": [
            {
                "name": name,
                "argv": list(argv),
                "cwd": str(kineo_root.resolve()),
                "timeout_s": timeout_s,
                "env": {"CUDA_VISIBLE_DEVICES": "0"},
                "interpreter": [],
            }
        ]
    }


def prepare(
    repository_root: Path,
    *,
    arm: Arm,
    views: Sequence[str] = STATIC_VIEWS,
    frame_count: int = DEFAULT_FRAME_COUNT,
    kineo_root: Path = DEFAULT_KINEO_ROOT,
    run_root: Path | None = None,
    queue_dir: Path = QUEUE_DIR,
    overwrite: bool = False,
    workers: int = 4,
) -> Path:
    repository_root = repository_root.resolve()
    kineo_root = kineo_root.resolve()
    views = tuple(views)
    if len(views) < 2:
        raise ValueError("Kineo needs at least two views")
    if any(view not in STATIC_VIEWS for view in views):
        raise ValueError(f"only the static views {STATIC_VIEWS} are supported")
    run_directory = repository_root / (run_root or Path(RUN_ROOT_TEMPLATE.format(arm=arm)))
    if (run_directory / "prepare.json").exists() and not overwrite:
        raise FileExistsError(f"{run_directory} is already prepared; pass --overwrite")
    run_directory.mkdir(parents=True, exist_ok=True)
    trims_dir = run_directory / "trims"
    gt_dir = run_directory / "gt_annotations"
    trims_dir.mkdir(exist_ok=True)
    gt_dir.mkdir(exist_ok=True)
    run_id = run_directory.name
    sequence_name = f"assembly101_{arm}_first_minute"

    rig = CameraRig.load(repository_root, views=views)
    reference, starts = view_start_frames(rig.clock_rules, views)

    trim_paths = {view: trims_dir / f"{view}_first_minute_{frame_count}f.mp4" for view in views}
    sources = {view: repository_root / proxy_path(view) for view in views}
    for view, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(f"proxy for {view} is missing: {source}")
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            view: pool.submit(
                _trim_view,
                sources[view],
                trim_paths[view],
                starts[view].start_frame,
                frame_count,
                overwrite=overwrite,
            )
            for view in views
        }
        verified = {view: future.result() for view, future in futures.items()}

    image_scale = PROXY_RESOLUTION_WH[0] / RAW_RESOLUTION_WH[0]
    intrinsics, extrinsics = build_gt_annotations(
        rig, views, image_scale=image_scale, translation_scale=1e-3
    )
    intrinsics_path = gt_dir / "camera_intrinsics.pkl"
    extrinsics_path = gt_dir / "camera_extrinsics.pkl"
    intrinsics_path.write_bytes(pickle.dumps(intrinsics, protocol=4))
    extrinsics_path.write_bytes(pickle.dumps(extrinsics, protocol=4))
    (gt_dir / "camera_intrinsics.json").write_text(
        json.dumps(intrinsics, indent=2) + "\n", encoding="utf-8"
    )
    (gt_dir / "camera_extrinsics.json").write_text(
        json.dumps(extrinsics, indent=2) + "\n", encoding="utf-8"
    )

    stock = yaml.safe_load((kineo_root / KINEO_STOCK_CONFIG).read_text(encoding="utf-8"))
    output_root = run_directory / "kineo_outputs"
    cache_root = run_directory / "kineo_cache"
    config = generate_kineo_config(
        stock, arm=arm, output_root_dir=str(output_root), cache_root_dir=str(cache_root)
    )
    config_path = run_directory / f"kineo_{arm}.yaml"
    config_path.write_text(dump_yaml(config), encoding="utf-8")

    runtime_json = run_directory / "kineo_runtime.json"
    argv = runner_argv(
        repository_root=repository_root,
        run_directory=run_directory,
        config_path=config_path,
        sequence_name=sequence_name,
        gt_directory=gt_dir,
        runtime_json=runtime_json,
        views=views,
        trims=trim_paths,
    )
    job = queue_job(
        name=f"kineo-multiview-{arm}", argv=argv, kineo_root=kineo_root, timeout_s=TIMEOUT_S[arm]
    )
    job_path = repository_root / queue_dir / f"jobs_t4_kineo_{arm}.json"
    job_path.parent.mkdir(parents=True, exist_ok=True)
    job_path.write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
    (run_directory / "queue_job.json").write_text(
        json.dumps(job, indent=2) + "\n", encoding="utf-8"
    )

    git_state = _git_fingerprint(kineo_root)
    checkpoint = kineo_root / NLF_CHECKPOINT
    if arm == "selfcal":
        removed = SELFCAL_REMOVED
        gt_role = (
            "gt_annotations only; no estimation stage reads them (transfer_gt_annotations is "
            "absent); rerun_export uses them to align its own recording and to log the dataset "
            "cameras.  The exported camera_extrinsics.pkl is Kineo's unaligned estimate."
        )
    else:
        removed = KNOWN_REMOVED
        gt_role = (
            "transfer_gt_annotations (order 0) copies camera_intrinsics and camera_extrinsics into "
            "the working annotations; every SfM/BA/scale/reorientation stage is removed, so Kineo "
            "detects (rtmlib + NLF) and triangulates through these cameras only."
        )
    manifest = PrepareManifest(
        arm=arm,
        run_id=run_id,
        recording_id=RECORDING_ID,
        prepared_at_utc=datetime.now(UTC).isoformat(),
        sequence_name=sequence_name,
        views=views,
        reference_view=reference,
        frame_count=frame_count,
        offset_handling=(
            f"Reference {reference} (latest-starting camera, pose offset "
            f"{starts[reference].pose_offset_frames:+d}); every view's trim starts at "
            "floor((reference.pose_frame(0) - view.pose_frame(0)) / 2 + 0.5) proxy frames so "
            "trimmed frame q shows pose frame 17640 + 2q + 9 in every view to within half an "
            "analysis frame; residuals are per view in `trims[].start`.  No `camera_temporal` "
            "annotation is passed, so Kineo's global_time_resampling takes its identity path "
            "(all offsets zero, identical frame counts) and Kineo frame q == analysis frame q on "
            "the C10379 clock."
        ),
        trims=tuple(
            ViewTrim(
                view=view,
                source_proxy=_fingerprint(sources[view], repository_root),
                trimmed=_fingerprint(trim_paths[view], repository_root),
                start=starts[view],
                frame_count=frame_count,
                verified_frame_count=verified[view],
            )
            for view in views
        ),
        kineo_config=_fingerprint(config_path, repository_root),
        kineo_stages=stage_names(config),
        removed_stock_stages=removed,
        gt_camera_intrinsics=_fingerprint(intrinsics_path, repository_root),
        gt_camera_extrinsics=_fingerprint(extrinsics_path, repository_root),
        gt_annotations_role=gt_role,
        intrinsics_image_scale=image_scale,
        extrinsics_convention=(
            "world-to-camera OpenCV (X_cam = R X_world + t), R = C2W[:3,:3].T, "
            "t = -R C2W[:3,3] / 1000 from the dataset camera_to_world (millimetres)"
        ),
        extrinsics_translation_unit="metres",
        camera_estimates=tuple(
            _fingerprint(
                repository_root / f"configs/assembly101/{view.lower()}_camera_estimate.json",
                repository_root,
            )
            for view in views
        ),
        clock_rules=_fingerprint(
            repository_root / "configs/assembly101/clock_rules.json", repository_root
        ),
        runner_script=_fingerprint(repository_root / RUNNER_SCRIPT, repository_root),
        kineo_repository=str(kineo_root),
        kineo_pinned_revision=DEFAULT_KINEO_REVISION,
        kineo_git=KineoGitState(**git_state),
        model_identities={
            "nlf_torchscript": {
                "path": NLF_CHECKPOINT,
                "sha256": (
                    hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                    if checkpoint.is_file()
                    else None
                ),
            },
            "moge_model": MOGE_MODEL if arm == "selfcal" else None,
            "rtmlib_bbox_model_url": RTMLIB_MODEL_URL,
            "smplx_body_model": "body_models/smplx/SMPLX_NEUTRAL.npz" if arm == "selfcal" else None,
        },
        kineo_output_root=str(output_root),
        kineo_cache_root=str(cache_root),
        queue_job=job_path.relative_to(repository_root).as_posix(),
        queue_timeout_s=TIMEOUT_S[arm],
        claim_boundaries=CLAIM_BOUNDARIES,
    )
    manifest_path = run_directory / "prepare.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return manifest_path


def load_prepare(run_directory: Path) -> PrepareManifest:
    return PrepareManifest.model_validate_json(
        (run_directory / "prepare.json").read_text(encoding="utf-8")
    )


# -- validate (CPU dry run in the pixi env) ----------------------------------------------------


def validate(
    repository_root: Path, run_directory: Path, *, kineo_root: Path = DEFAULT_KINEO_ROOT
) -> Path:
    """Run the runner's `--validate-only` path under pixi with CUDA hidden; no model, no GPU."""
    repository_root = repository_root.resolve()
    run_directory = (repository_root / run_directory).resolve()
    manifest = load_prepare(run_directory)
    job = json.loads((run_directory / "queue_job.json").read_text(encoding="utf-8"))["jobs"][0]
    argv = [*job["argv"], "--validate-only"]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1"}
    completed = subprocess.run(
        argv, cwd=kineo_root, env=env, capture_output=True, text=True, check=False
    )
    report_path = run_directory / "validation.json"
    payload: dict[str, Any] = {
        "run_id": manifest.run_id,
        "validated_at_utc": datetime.now(UTC).isoformat(),
        "argv": argv,
        "exit_code": completed.returncode,
        "stderr_tail": completed.stderr[-4000:],
    }
    try:
        payload["report"] = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        payload["stdout_tail"] = completed.stdout[-4000:]
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"Kineo config validation failed; see {report_path}")
    return report_path


# -- evaluate ----------------------------------------------------------------------------------


def _stats(values: np.ndarray) -> tuple[int, float | None, float | None, float | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0, None, None, None
    return (
        int(finite.size),
        float(np.median(finite)),
        float(np.percentile(finite, 90)),
        float(np.mean(finite)),
    )


def dataset_wrists(
    frames: Mapping[int, Assembly101HandFrame], frame_count: int, *, min_confidence: float = 0.5
) -> np.ndarray:
    """`(F, 2, 3)` dataset wrist world mm by side (NaN when absent or below confidence)."""
    wrists = np.full((frame_count, 2, 3), np.nan)
    for frame in range(frame_count):
        entry = frames.get(frame)
        if entry is None:
            continue
        for hand in entry.hands:
            if hand.confidence < min_confidence:
                continue
            joint = hand.joints_world_mm[DATASET_WRIST]
            wrists[frame, SIDES.index(hand.side)] = (joint.x, joint.y, joint.z)
    return wrists


def compare_wrists(
    kineo_wrists_mm: np.ndarray, dataset_wrists_mm: np.ndarray
) -> tuple[WristComparison, ...]:
    """Per side: distance between Kineo's body wrist and the dataset's hand wrist, plus how
    often the opposite dataset side would have been closer (a handedness sanity signal)."""
    results = []
    for s, side in enumerate(SIDES):
        kineo = kineo_wrists_mm[:, s]
        same = np.linalg.norm(kineo - dataset_wrists_mm[:, s], axis=1)
        other = np.linalg.norm(kineo - dataset_wrists_mm[:, 1 - s], axis=1)
        count, median, p90, mean = _stats(same)
        both = np.isfinite(same) & np.isfinite(other)
        swapped = float(np.mean(other[both] < same[both])) if both.any() else None
        results.append(
            WristComparison(
                side=side,
                frames_compared=count,
                median_mm=median,
                p90_mm=p90,
                mean_mm=mean,
                swapped_side_closer_fraction=swapped,
            )
        )
    return tuple(results)


def align_cameras(
    kineo: Mapping[str, tuple[np.ndarray, np.ndarray]],
    rig: CameraRig,
    views: Sequence[str],
    *,
    arm: Arm,
) -> tuple[SimilarityAlignment, tuple[CameraComparison, ...]]:
    """Similarity from Kineo's world to the dataset's (mm) and per-camera errors after it.

    Self-calibration: Umeyama with scale on the camera centres.  Known cameras: the fixed
    metres-to-millimetres scale with identity rotation (the world *is* the dataset's), so the
    per-camera numbers are a round-trip check that must be ~0.
    """
    kineo_centers = np.array([camera_center(*kineo[view]) for view in views])
    dataset_centers = np.array([rig.camera_position(view) for view in views])
    if arm == "selfcal":
        scale, rotation, translation = umeyama(kineo_centers, dataset_centers, with_scale=True)
        method = "umeyama_with_scale_on_camera_centres"
    else:
        scale, rotation, translation = 1000.0, np.eye(3), np.zeros(3)
        method = "fixed_metres_to_millimetres_identity"
    aligned_centers = apply_similarity(kineo_centers, scale, rotation, translation)
    rms = float(np.sqrt(np.mean(np.sum((aligned_centers - dataset_centers) ** 2, axis=1))))
    comparisons = []
    for index, view in enumerate(views):
        rotation_w2c, _ = kineo[view]
        kineo_c2w = rotation @ rotation_w2c.T
        dataset_c2w = rig.camera_to_world(view)[:3, :3]
        comparisons.append(
            CameraComparison(
                view=view,
                rotation_error_deg=rotation_angle_deg(dataset_c2w, kineo_c2w),
                translation_error_mm=float(
                    np.linalg.norm(aligned_centers[index] - dataset_centers[index])
                ),
                kineo_center_aligned_mm=tuple(float(v) for v in aligned_centers[index]),
                dataset_center_mm=tuple(float(v) for v in dataset_centers[index]),
            )
        )
    alignment = SimilarityAlignment(
        method=method,
        scale_mm_per_kineo_unit=float(scale),
        rotation=tuple(tuple(float(v) for v in row) for row in rotation),
        translation_mm=tuple(float(v) for v in translation),
        camera_center_rms_mm=rms,
    )
    return alignment, tuple(comparisons)


def read_queue_record(queue_log: Path, job_name: str) -> QueueRecord | None:
    """Last `job_end` record for the job in the overnight queue log (None when it has not run)."""
    if not queue_log.is_file():
        return None
    record: QueueRecord | None = None
    for line in queue_log.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") == "job_end" and event.get("name") == job_name:
            record = QueueRecord(
                job=job_name,
                state=str(event.get("state")),
                exit_code=event.get("exit_code"),
                duration_s=float(event.get("duration_s", 0.0)),
                stdout_log=event.get("stdout_log"),
                time=str(event.get("time")),
            )
    return record


def time_to_first_3d_output(timings: Sequence[StageTiming]) -> float | None:
    """Cumulative pipeline time through the triangulation stage (the first 3D body output)."""
    total = 0.0
    for timing in sorted(timings, key=lambda t: t.stage_idx):
        total += timing.duration_seconds
        if timing.stage_name == "MVS Triangulation":
            return total
    return None


def evaluate(
    repository_root: Path,
    run_directory: Path,
    *,
    queue_log: Path = QUEUE_LOG,
    dataset_reference: Path = DATASET_REFERENCE_RUN,
) -> Path:
    repository_root = repository_root.resolve()
    run_directory = (repository_root / run_directory).resolve()
    manifest = load_prepare(run_directory)
    annotations_dir = Path(manifest.kineo_output_root) / "annotations" / manifest.sequence_name
    required = ("camera_extrinsics.pkl", "camera_intrinsics.pkl", "keypoints_3d.pkl")
    for name in required:
        if not (annotations_dir / name).is_file():
            raise FileNotFoundError(f"Kineo has not exported {name} under {annotations_dir}")

    views = manifest.views
    rig = CameraRig.load(repository_root, views=views)
    extrinsics = parse_extrinsics(load_kineo_pkl(annotations_dir / "camera_extrinsics.pkl"))
    if set(extrinsics) != set(views):
        raise ValueError(f"Kineo exported cameras {sorted(extrinsics)} but the run has {views}")
    intrinsics = parse_intrinsics(load_kineo_pkl(annotations_dir / "camera_intrinsics.pkl"))
    body = parse_keypoints_3d(load_kineo_pkl(annotations_dir / "keypoints_3d.pkl"))
    timings: tuple[StageTiming, ...] = ()
    if (annotations_dir / "stage_timings.pkl").is_file():
        timings = parse_stage_timings(load_kineo_pkl(annotations_dir / "stage_timings.pkl"))
    global_scale: float | None = None
    if (annotations_dir / "global_scale.pkl").is_file():
        rows = load_kineo_pkl(annotations_dir / "global_scale.pkl").get("annotations", [])
        if rows and "scale" in rows[0]:
            global_scale = float(rows[0]["scale"])

    alignment, cameras = align_cameras(extrinsics, rig, views, arm=manifest.arm)
    rotation = np.asarray(alignment.rotation)
    translation = np.asarray(alignment.translation_mm)
    scale = alignment.scale_mm_per_kineo_unit

    frame_count = manifest.frame_count
    body_frames = min(frame_count, body.xyz.shape[0])
    aligned = np.full((frame_count, len(body.names), 3), np.nan)
    aligned[:body_frames] = apply_similarity(
        body.xyz[:body_frames].reshape(-1, 3), scale, rotation, translation
    ).reshape(body_frames, len(body.names), 3)
    scores = np.zeros((frame_count, len(body.names)))
    scores[:body_frames] = body.scores[:body_frames]
    # Kineo zeroes both position and score for joints it could not triangulate.
    aligned[scores <= 0] = np.nan
    wrist_indices = [body.names.index(KINEO_WRIST_NAMES[side]) for side in SIDES]
    kineo_wrists = aligned[:, wrist_indices]

    dataset = a101.load_reference(
        dataset_reference, repository_root, frame_count=frame_count, verify=False
    )
    dataset_w = dataset_wrists(
        dataset.frames, frame_count, min_confidence=dataset.manifest.draw_confidence_threshold
    )
    wrists = compare_wrists(kineo_wrists, dataset_w)

    job_name = f"kineo-multiview-{manifest.arm}"
    queue_record = read_queue_record(repository_root / queue_log, job_name)
    runtime_json = run_directory / "kineo_runtime.json"
    runtime: dict[str, Any] = (
        json.loads(runtime_json.read_text(encoding="utf-8")) if runtime_json.is_file() else {}
    )
    notes = []
    if queue_record is None:
        notes.append("no job_end record in the queue log; runtime is the runner's own timer")
    if not runtime:
        notes.append("kineo_runtime.json missing; VRAM unknown")
    else:
        notes.append(
            "gpu_peak_vram_bytes is torch.cuda.max_memory_reserved in the runner process; "
            "ONNX Runtime (rtmlib) allocations are outside it"
        )
    has_body = np.isfinite(aligned[..., 0]).any(axis=1)
    both_wrists = np.isfinite(kineo_wrists[..., 0]).all(axis=1)
    measures = PreAccuracyMeasures(
        coverage_frames_with_body_3d=int(has_body.sum()),
        coverage_frames_with_both_wrists=int(both_wrists.sum()),
        frame_count=frame_count,
        time_to_first_3d_output_seconds=time_to_first_3d_output(timings),
        runtime_seconds=(
            queue_record.duration_s if queue_record is not None else runtime.get("elapsed_seconds")
        ),
        kineo_pipeline_seconds=(
            float(sum(t.duration_seconds for t in timings)) if timings else None
        ),
        gpu_peak_vram_bytes=runtime.get("gpu_peak_vram_bytes"),
        id_resets=None,
        id_resets_note=(
            f"not applicable: rtmlib best_bbox_only yields one subject per frame "
            f"(subject ids exported: {', '.join(body.subject_ids)})"
        ),
        notes=tuple(notes),
    )
    native = [
        _fingerprint(annotations_dir / name, repository_root)
        for name in sorted(os.listdir(annotations_dir))
        if name.endswith(".pkl")
    ]
    for extra in (
        Path(manifest.kineo_output_root) / f"{manifest.sequence_name}.rrd",
        Path(manifest.kineo_output_root) / f"{manifest.sequence_name}.bvh",
        Path(manifest.kineo_output_root) / f"{manifest.sequence_name}_ba_history.rrd",
    ):
        if extra.is_file():
            native.append(_fingerprint(extra, repository_root))

    evaluation = KineoMultiviewManifest(
        arm=manifest.arm,
        run_id=manifest.run_id,
        evaluated_at_utc=datetime.now(UTC).isoformat(),
        prepare=_fingerprint(run_directory / "prepare.json", repository_root),
        views=views,
        frame_count=frame_count,
        kineo_native_artifacts=tuple(native),
        alignment=alignment,
        cameras=cameras,
        camera_rotation_error_deg_median=float(np.median([c.rotation_error_deg for c in cameras])),
        camera_translation_error_mm_median=float(
            np.median([c.translation_error_mm for c in cameras])
        ),
        kineo_global_scale=global_scale,
        wrists=wrists,
        body_joint_names=body.names,
        stage_timings=timings,
        queue_record=queue_record,
        measures=measures,
        claim_boundaries=CLAIM_BOUNDARIES,
    )
    np.savez_compressed(
        run_directory / "body_aligned_mm.npz",
        frames=np.arange(frame_count),
        xyz=aligned,
        scores=scores,
        names=np.array(body.names),
        edges=np.array(body.edges, dtype=np.int64).reshape(-1, 2),
        dataset_wrists=dataset_w,
        kineo_camera_centers_aligned_mm=np.array([c.kineo_center_aligned_mm for c in cameras]),
        kineo_camera_rotations_c2w_aligned=np.array(
            [rotation @ extrinsics[view][0].T for view in views]
        ),
        kineo_intrinsics=np.array([intrinsics[view]["K"] for view in views]),
        kineo_resolution_hw=np.array([intrinsics[view]["resolution_hw"] for view in views]),
        views=np.array(views),
    )
    manifest_path = run_directory / "manifest.json"
    manifest_path.write_text(evaluation.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return manifest_path


def load_evaluation(run_directory: Path) -> KineoMultiviewManifest:
    return KineoMultiviewManifest.model_validate_json(
        (run_directory / "manifest.json").read_text(encoding="utf-8")
    )


def print_evaluation(manifest: KineoMultiviewManifest) -> None:
    print(
        f"{manifest.run_id} ({manifest.arm}): alignment {manifest.alignment.method}, scale "
        f"{manifest.alignment.scale_mm_per_kineo_unit:.2f} mm/unit, camera-centre RMS "
        f"{manifest.alignment.camera_center_rms_mm:.1f} mm"
    )
    for camera in manifest.cameras:
        print(
            f"  {camera.view}: rotation {camera.rotation_error_deg:.2f} deg, translation "
            f"{camera.translation_error_mm:.1f} mm"
        )
    for wrist in manifest.wrists:
        median = "n/a" if wrist.median_mm is None else f"{wrist.median_mm:.1f}"
        p90 = "n/a" if wrist.p90_mm is None else f"{wrist.p90_mm:.1f}"
        print(
            f"  {wrist.side} wrist vs dataset: median {median} mm, p90 {p90} mm over "
            f"{wrist.frames_compared} frames"
        )
    m = manifest.measures
    print(
        f"  coverage {m.coverage_frames_with_body_3d}/{m.frame_count} frames with body 3D, "
        f"runtime {m.runtime_seconds}, pipeline {m.kineo_pipeline_seconds}, "
        f"first 3D {m.time_to_first_3d_output_seconds}, VRAM {m.gpu_peak_vram_bytes}"
    )


# -- CLI ---------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--kineo-root", type=Path, default=DEFAULT_KINEO_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)

    prep = commands.add_parser("prepare", help="trims, YAML, GT PKLs, queue job (CPU only)")
    prep.add_argument("--arm", choices=ARMS, required=True)
    prep.add_argument("--views", nargs="+", default=list(STATIC_VIEWS))
    prep.add_argument("--frame-count", type=int, default=DEFAULT_FRAME_COUNT)
    prep.add_argument("--run-root", type=Path, default=None)
    prep.add_argument("--queue-dir", type=Path, default=QUEUE_DIR)
    prep.add_argument("--overwrite", action="store_true")
    prep.add_argument("--workers", type=int, default=4)

    val = commands.add_parser("validate", help="CPU dry run of the YAML under pixi (no models)")
    val.add_argument("run_directory", type=Path)

    ev = commands.add_parser("evaluate", help="compare Kineo's exported PKLs with the dataset")
    ev.add_argument("run_directory", type=Path)
    ev.add_argument("--queue-log", type=Path, default=QUEUE_LOG)
    ev.add_argument("--dataset-reference", type=Path, default=DATASET_REFERENCE_RUN)

    rr_cmd = commands.add_parser("rerun", help="inference-free Rerun recording of the comparison")
    rr_cmd.add_argument("run_directory", type=Path)
    rr_cmd.add_argument("--output", type=Path, default=None)

    args = parser.parse_args(argv)
    started = time.monotonic()
    if args.command == "prepare":
        path = prepare(
            args.repository_root,
            arm=args.arm,
            views=args.views,
            frame_count=args.frame_count,
            kineo_root=args.kineo_root,
            run_root=args.run_root,
            queue_dir=args.queue_dir,
            overwrite=args.overwrite,
            workers=args.workers,
        )
    elif args.command == "validate":
        path = validate(args.repository_root, args.run_directory, kineo_root=args.kineo_root)
    elif args.command == "evaluate":
        path = evaluate(
            args.repository_root,
            args.run_directory,
            queue_log=args.queue_log,
            dataset_reference=args.dataset_reference,
        )
        print_evaluation(load_evaluation(path.parent))
    else:
        from .kineo_multiview_review import build_recording

        path = build_recording(args.repository_root, args.run_directory, output=args.output)
    print(f"{path} ({time.monotonic() - started:.1f} s)", file=sys.stderr)


if __name__ == "__main__":
    main()
