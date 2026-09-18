"""Multi-view hand triangulation for the pinned Assembly101 recording (`battle-athena-hands`).

Per-view 2D hands from MediaPipe (`battle-mediapipe-hands`) or WiLoR (`battle-wilor-hands`)
runs are lifted onto the calibrated `CameraRig`, matched to a hand side, undistorted with each
view's fitted Brown/rational model and triangulated with ATHENA's DLT + reprojection-filter
loop (`athena.triangulaterefine._triangulate_with_filtering`, executed in ATHENA's own
virtualenv through `scripts/athena_triangulate_worker.py`) followed by ATHENA's Savitzky-Golay
`_smooth3d`.  The rig's own leave-one-out DLT is available as `--triangulator rig`.

Time.  The analysis timeline is the C10379 proxy clock used by the dataset reference window
(`pose_frame = 17649 + 2 p`).  The other cameras started at different instants (measured
per-view offsets 5..9 pose frames, ego 0), so for frame `p` each view contributes the analysis
frame whose pose frame is nearest; the residual is 0 or +-1 pose frame (16.7 ms) per view and
is recorded in the manifest.  Nothing is resampled.

Hand sides.  Method handedness is unreliable, so each detected hand is assigned the side of the
nearest dataset wrist projected into that view (handedness-agnostic, within
`--match-threshold-px` raw pixels).  The correspondence therefore borrows dataset context; the
triangulated *positions* do not.

Joint mapping (dataset MS-G3D order -> MediaPipe/WiLoR order).  Twenty joints are common:

    dataset  0 thumb_tip   -> mp  4        dataset 11 middle_mcp -> mp  9
    dataset  1 index_tip   -> mp  8        dataset 12 middle_pip -> mp 10
    dataset  2 middle_tip  -> mp 12        dataset 13 middle_dip -> mp 11
    dataset  3 ring_tip    -> mp 16        dataset 14 ring_mcp   -> mp 13
    dataset  4 pinky_tip   -> mp 20        dataset 15 ring_pip   -> mp 14
    dataset  5 wrist       -> mp  0        dataset 16 ring_dip   -> mp 15
    dataset  6 thumb_cmc   -> mp  2 (*)    dataset 17 pinky_mcp  -> mp 17
    dataset  7 thumb_ip    -> mp  3        dataset 18 pinky_pip  -> mp 18
    dataset  8 index_mcp   -> mp  5        dataset 19 pinky_dip  -> mp 19
    dataset  9 index_pip   -> mp  6        dropped: dataset 20 palm, mp 1 thumb_cmc
    dataset 10 index_dip   -> mp  7

(*) The dataset thumb has three joints; its "thumb_cmc" sits at the base of the thumb
metacarpal like MediaPipe's thumb_mcp (2), so MediaPipe's carpal joint (1) is the one dropped.
The two thumb joints are the least certain correspondences; per-joint statistics keep them
visible.

Claims.  Triangulated hands compared with the dataset's `landmarks3D` are cross-source
disagreement between two estimates (the dataset's own multi-view tracker with a fixed-scale
hand model against a DLT on MediaPipe/WiLoR detections), never accuracy.  Intrinsics are
fitted estimates.  CC BY-NC 4.0 attribution applies to the dataset; ATHENA is MIT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field

from .assembly101_camera_fit import PoseMembers, raw_image_size
from .assembly101_clock_offset import is_ego
from .assembly101_fetch_view import RECORDING_ID, STATIC_VIEWS
from .assembly101_pose_schemas import (
    ASSEMBLY101_JOINT_NAMES,
    ASSEMBLY101_WRIST_INDEX,
    Assembly101ClockRule,
)
from .athena_triangulation import ATHENA_REVISION, ATHENA_ROOT
from .multiview_geometry import CameraRig
from .schemas import ArtifactFingerprint, VersionedModel

HandSource = Literal["mediapipe", "wilor"]
Triangulator = Literal["athena", "rig"]
SIDES: tuple[str, ...] = ("left", "right")
DATASET_SIDE_INDEX = {"left": 0, "right": 1}

# (name, dataset index, MediaPipe/WiLoR index); see the module docstring.
COMMON_JOINTS: tuple[tuple[str, int, int], ...] = (
    ("thumb_tip", 0, 4),
    ("index_tip", 1, 8),
    ("middle_tip", 2, 12),
    ("ring_tip", 3, 16),
    ("pinky_tip", 4, 20),
    ("wrist", 5, 0),
    ("thumb_cmc", 6, 2),
    ("thumb_ip", 7, 3),
    ("index_mcp", 8, 5),
    ("index_pip", 9, 6),
    ("index_dip", 10, 7),
    ("middle_mcp", 11, 9),
    ("middle_pip", 12, 10),
    ("middle_dip", 13, 11),
    ("ring_mcp", 14, 13),
    ("ring_pip", 15, 14),
    ("ring_dip", 16, 15),
    ("pinky_mcp", 17, 17),
    ("pinky_pip", 18, 18),
    ("pinky_dip", 19, 19),
)
COMMON_JOINT_NAMES: tuple[str, ...] = tuple(name for name, _, _ in COMMON_JOINTS)
DATASET_INDICES = np.array([d for _, d, _ in COMMON_JOINTS])
METHOD_INDICES = np.array([m for _, _, m in COMMON_JOINTS])
WRIST = COMMON_JOINT_NAMES.index("wrist")
FINGERTIPS = tuple(
    COMMON_JOINT_NAMES.index(f"{f}_tip") for f in ("thumb", "index", "middle", "ring", "pinky")
)
# Edges in common-joint order (wrist to each mcp, finger chains, palm arc).
COMMON_EDGES: tuple[tuple[int, int], ...] = (
    (5, 6),
    (6, 7),
    (7, 0),
    (5, 8),
    (8, 9),
    (9, 10),
    (10, 1),
    (5, 11),
    (11, 12),
    (12, 13),
    (13, 2),
    (5, 14),
    (14, 15),
    (15, 16),
    (16, 3),
    (5, 17),
    (17, 18),
    (18, 19),
    (19, 4),
    (6, 8),
    (8, 11),
    (11, 14),
    (14, 17),
)
assert all(ASSEMBLY101_JOINT_NAMES[d] == name for name, d, _ in COMMON_JOINTS)
assert ASSEMBLY101_JOINT_NAMES[ASSEMBLY101_WRIST_INDEX] == "wrist"

ATHENA_PYTHON = ATHENA_ROOT / ".venv" / "bin" / "python"
ATHENA_WORKER = Path("scripts/athena_triangulate_worker.py")
DEFAULT_REFERENCE_VIEW = "C10379"
DEFAULT_REPROJECTION_FILTER_PX = 30.0
DEFAULT_MATCH_THRESHOLD_PX = 150.0
DEFAULT_FRAME_COUNT = 1800
ANALYSIS_FPS = 30.0
DEFAULT_RUN_TEMPLATES: dict[str, str] = {
    "mediapipe": "runs/mediapipe-hands-{view_lower}-60s-20260918",
    "wilor": "runs/wilor-hands-{view_lower}-60s-20260918",
}
# The selected C10379 MediaPipe run predates the per-view naming.
DEFAULT_RUN_OVERRIDES: dict[str, dict[str, str]] = {
    "mediapipe": {"C10379": "runs/mediapipe-hands-static-60s-fused-dedup-th035-20260916t0430z"},
    "wilor": {},
}
DATASET_REFERENCE_RUN = Path("runs/assembly101-reference-first-minute-v1")
DEFAULT_OUTPUT_ROOT = Path("runs/athena-hands-first-minute-mediapipe")
SOURCE_MANIFEST_KEY = {"mediapipe": "mediapipe_hands", "wilor": "wilor_hands"}
CLAIM_BOUNDARIES: tuple[str, ...] = (
    "Triangulated hands versus the dataset's landmarks3D is cross-source disagreement between "
    "two estimates, not accuracy: the dataset poses are the output of the dataset's own "
    "multi-view tracker with a fixed-scale hand model.",
    "Camera intrinsics are estimates fitted to the dataset's own 2D/3D projection; extrinsics "
    "and clock offsets are dataset context and measurements, not calibration ground truth.",
    "Hand-side correspondence borrows the dataset wrist projected into each view; the "
    "triangulated positions come only from the method's 2D detections.",
    "Views are aligned to the nearest analysis frame on the C10379 clock; the residual is at "
    "most one 60 fps pose frame (16.7 ms) per view and nothing is resampled.",
    "Assembly101 is CC BY-NC 4.0 (attribution required, non-commercial); ATHENA is MIT.",
)


# -- loading per-view 2D hands --------------------------------------------------------------


@dataclass
class ViewHands:
    """One view's detections: analysis frame -> list of (21, 2) raw-pixel landmark arrays."""

    view: str
    run_directory: Path
    source: HandSource
    frames: dict[int, list[np.ndarray]] = field(default_factory=dict)
    confidences: dict[int, list[float]] = field(default_factory=dict)
    camera_relative_3d: dict[int, list[np.ndarray | None]] = field(default_factory=dict)
    proxy_fingerprint: ArtifactFingerprint | None = None
    frame_count: int = 0
    camera_translation_attached: bool = False

    @property
    def frames_with_hands(self) -> int:
        return sum(1 for hands in self.frames.values() if hands)


def camera_name_from_view_id(view_id: str) -> str:
    kind, _, rest = view_id.partition("-")
    if kind == "static":
        return rest.upper()
    if kind == "ego" and rest.startswith("hmc"):
        return f"HMC_{rest[3:]}"
    raise ValueError(f"cannot map view id {view_id!r} onto a camera name")


def run_directory_for(view: str, source: HandSource, overrides: Mapping[str, str]) -> Path:
    if view in overrides:
        return Path(overrides[view])
    if view in DEFAULT_RUN_OVERRIDES[source]:
        return Path(DEFAULT_RUN_OVERRIDES[source][view])
    return Path(DEFAULT_RUN_TEMPLATES[source].format(view_lower=view.lower().replace("_", "")))


def load_view_hands(
    repository_root: Path, view: str, run_directory: Path, source: HandSource
) -> ViewHands:
    """Read a MediaPipe/WiLoR run's `observations.jsonl` as raw-pixel landmarks.

    Normalised landmarks are relative to the proxy frame, and every proxy is a uniform 1.5x
    downscale of the sensor, so raw pixels are `normalised * raw_image_size`.  The run's
    proxy fingerprint must name this camera.
    """
    root = repository_root / run_directory
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    metadata = manifest.get(SOURCE_MANIFEST_KEY[source])
    if metadata is None:
        raise ValueError(f"{root / 'manifest.json'} is not a {source} run")
    proxy = ArtifactFingerprint.model_validate(metadata["proxy_fingerprint"])
    if view not in proxy.uri:
        raise ValueError(f"run {run_directory} is for {proxy.uri}, not {view}")
    view_ids = {obs["view_id"] for obs in manifest.get("observations", [])[:1]}
    if view_ids and {camera_name_from_view_id(v) for v in view_ids} != {view}:
        raise ValueError(f"run {run_directory} observations are for {view_ids}, not {view}")
    width, height = raw_image_size(view)
    result = ViewHands(
        view=view, run_directory=run_directory, source=source, proxy_fingerprint=proxy
    )
    with (root / "observations.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            observation = json.loads(line)
            frame = int(observation["analysis_frame_index"])
            hands: list[np.ndarray] = []
            confidences: list[float] = []
            relative: list[np.ndarray | None] = []
            for hand in observation.get("hands", ()):
                pixels = np.array(
                    [[p["x"] * width, p["y"] * height] for p in hand["landmarks"]], dtype=np.float64
                )
                if pixels.shape != (21, 2):
                    raise ValueError(f"hand without 21 landmarks in {run_directory} frame {frame}")
                hands.append(pixels)
                confidences.append(float(hand.get("confidence", 0.0)))
                joints = hand.get("joints_3d_camera_relative")
                relative.append(
                    None
                    if joints is None
                    else np.array([[j["x"], j["y"], j["z"]] for j in joints], dtype=np.float64)
                )
            result.frames[frame] = hands
            result.confidences[frame] = confidences
            result.camera_relative_3d[frame] = relative
    result.frame_count = len(result.frames)
    native_root = root / "native_evidence"
    if source == "wilor" and native_root.is_dir():
        _attach_wilor_camera_translation(result, root)
    return result


def _attach_wilor_camera_translation(result: ViewHands, root: Path) -> None:
    """Replace the wrist-rooted `joints_3d_camera_relative` with camera-frame joints by adding
    WiLoR's `pred_cam_t_full` from the native evidence (matched by hand id).  Units are WiLoR's
    own (metres under its scaled focal length), still not calibrated to this camera."""
    with (root / "observations.jsonl").open(encoding="utf-8") as handle:
        hand_ids = {
            int(obs["analysis_frame_index"]): [hand["hand_id"] for hand in obs.get("hands", ())]
            for obs in map(json.loads, filter(str.strip, handle))
        }
    for frame, ids in hand_ids.items():
        native_path = root / "native_evidence" / f"frame_{frame:06d}.json"
        if not native_path.is_file():
            continue
        native = {
            hand["hand_id"]: hand
            for hand in json.loads(native_path.read_text(encoding="utf-8")).get("hands", ())
        }
        joints = result.camera_relative_3d.get(frame, [])
        for index, hand_id in enumerate(ids):
            record = native.get(hand_id)
            if record is None or index >= len(joints) or joints[index] is None:
                continue
            translation = np.asarray(record.get("pred_cam_t_full", ()), dtype=np.float64)
            if translation.shape == (3,):
                joints[index] = joints[index] + translation
        result.camera_translation_attached = True


# -- clocks -----------------------------------------------------------------------------------


def nearest_analysis_frame(rule: Assembly101ClockRule, pose_frame: int) -> tuple[int, int]:
    """View analysis frame whose pose frame is nearest to `pose_frame`, and the residual."""
    exact = (pose_frame - rule.proxy_start_raw_frame - rule.pose_offset_frames) / (
        rule.raw_frames_per_proxy_frame
    )
    frame = int(np.floor(exact + 0.5))
    frame = max(frame, 0)
    return frame, rule.pose_frame(frame) - pose_frame


# -- side matching ------------------------------------------------------------------------------


def match_hands_to_dataset(
    detections: list[np.ndarray],
    dataset_wrists: Mapping[str, np.ndarray],
    *,
    threshold_px: float,
) -> dict[str, tuple[int, float]]:
    """Assign each dataset side at most one detection by wrist proximity (min total distance).

    Returns side -> (detection index, distance px).  Handedness-agnostic by design.
    """
    if not detections or not dataset_wrists:
        return {}
    sides = list(dataset_wrists)
    distance = np.array(
        [[np.linalg.norm(det[0] - dataset_wrists[side]) for det in detections] for side in sides]
    )
    best: tuple[float, dict[str, tuple[int, float]]] | None = None
    from itertools import permutations

    detection_indices = list(range(len(detections)))
    for count in range(min(len(sides), len(detections)), 0, -1):
        for chosen_sides in permutations(range(len(sides)), count):
            for chosen_dets in permutations(detection_indices, count):
                pairs = list(zip(chosen_sides, chosen_dets, strict=True))
                if any(distance[s, d] > threshold_px for s, d in pairs):
                    continue
                total = float(sum(distance[s, d] for s, d in pairs))
                if best is None or total < best[0]:
                    best = (total, {sides[s]: (d, float(distance[s, d])) for s, d in pairs})
        if best is not None:
            break
    return best[1] if best is not None else {}


# -- ATHENA worker ------------------------------------------------------------------------------


def athena_available(repository_root: Path) -> bool:
    return ATHENA_PYTHON.is_file() and (repository_root / ATHENA_WORKER).is_file()


def run_athena_worker(
    repository_root: Path, payload: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Execute ATHENA's triangulation/smoothing in its own venv and read the results back."""
    with tempfile.TemporaryDirectory(prefix="athena-hands-") as tmp:
        inputs = Path(tmp) / "inputs.npz"
        outputs = Path(tmp) / "outputs.npz"
        np.savez(inputs, **payload)
        completed = subprocess.run(
            [str(ATHENA_PYTHON), str(repository_root / ATHENA_WORKER), str(inputs), str(outputs)],
            capture_output=True,
            text=True,
            cwd=str(ATHENA_ROOT),
        )
        if completed.returncode != 0:
            raise RuntimeError(f"ATHENA worker failed:\n{completed.stderr[-4000:]}")
        with np.load(outputs) as archive:
            return {key: np.asarray(archive[key]) for key in archive}


# -- schemas ------------------------------------------------------------------------------------


class ViewAlignment(VersionedModel):
    view: str
    run_directory: str
    proxy_fingerprint: ArtifactFingerprint | None
    pose_offset_frames: int
    analysis_frame_shift: float = Field(
        description="view analysis frame minus reference frame (exact, may be half-integral)"
    )
    residual_pose_frames_used: tuple[int, ...]
    frames_with_detections: int = Field(ge=0)
    detections_matched: int = Field(ge=0)
    detections_unmatched: int = Field(ge=0)
    match_distance_median_px: float | None = None
    match_distance_p90_px: float | None = None


class JointStat(VersionedModel):
    joint: str
    count: int = Field(ge=0)
    median_mm: float | None = None
    p90_mm: float | None = None


class GroupStat(VersionedModel):
    group: str
    count: int = Field(ge=0)
    median_mm: float | None = None
    p90_mm: float | None = None
    mean_mm: float | None = None


class ViewReprojection(VersionedModel):
    view: str
    points_observed: int = Field(ge=0)
    points_used: int = Field(ge=0)
    rms_px_used: float | None = None
    median_px_used: float | None = None
    rms_px_observed: float | None = None


class SteadinessStat(VersionedModel):
    series: str
    unit: str
    count: int = Field(ge=0)
    median_step: float | None = None
    p90_step: float | None = None
    note: str | None = None


class HandSummary(VersionedModel):
    side: str
    frames_with_solution: int = Field(ge=0)
    frames_with_wrist: int = Field(ge=0)
    frames_dataset_present: int = Field(ge=0)
    mean_contributing_views: float | None = None
    disagreement_raw: tuple[GroupStat, ...]
    disagreement_smoothed: tuple[GroupStat, ...]
    per_joint_raw: tuple[JointStat, ...]
    per_view_reprojection: tuple[ViewReprojection, ...]
    steadiness: tuple[SteadinessStat, ...]


class AthenaHandsManifest(VersionedModel):
    manifest_kind: Literal["athena_multiview_hands"] = "athena_multiview_hands"
    run_id: str
    recording_id: str
    hand_source: HandSource
    triangulator: Triangulator
    triangulator_detail: str
    athena_revision: str | None
    views: tuple[str, ...]
    reference_view: str
    frame_count: int = Field(ge=1)
    analysis_fps: float
    reprojection_filter_px: float
    match_threshold_px: float
    smoothing: str
    joint_names: tuple[str, ...]
    joint_mapping_dataset_to_method: tuple[tuple[int, int], ...]
    alignments: tuple[ViewAlignment, ...]
    hands: tuple[HandSummary, ...]
    hands_path: str
    dataset_reference: ArtifactFingerprint
    input_artifacts: tuple[ArtifactFingerprint, ...]
    runtime_seconds: float = Field(ge=0)
    time_to_first_output_seconds: float = Field(ge=0)
    claim_boundaries: tuple[str, ...] = Field(min_length=1)


# -- core ---------------------------------------------------------------------------------------


def _fingerprint(path: Path, repository_root: Path) -> ArtifactFingerprint:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return ArtifactFingerprint(
        uri=path.resolve().relative_to(repository_root.resolve()).as_posix(),
        sha256=digest.hexdigest(),
        source="measured",
    )


def _stats(values: np.ndarray) -> tuple[int, float | None, float | None, float | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0, None, None, None
    return (
        int(finite.size),
        float(np.median(finite)),
        float(np.quantile(finite, 0.9)),
        float(finite.mean()),
    )


def _group_stats(errors: np.ndarray) -> tuple[GroupStat, ...]:
    """`errors` (F, J) mm with NaN gaps -> wrist / fingertips / all groups."""
    groups = {
        "wrist": errors[:, WRIST],
        "fingertips": errors[:, list(FINGERTIPS)],
        "all_joints": errors,
    }
    result = []
    for name, values in groups.items():
        count, median, p90, mean = _stats(np.asarray(values).ravel())
        result.append(
            GroupStat(group=name, count=count, median_mm=median, p90_mm=p90, mean_mm=mean)
        )
    return tuple(result)


def _steadiness(
    series: np.ndarray, name: str, unit: str, note: str | None = None
) -> SteadinessStat:
    """Median/p90 of |x[t+1] - x[t]| over consecutive frames where both exist."""
    if series.shape[0] < 2:
        return SteadinessStat(series=name, unit=unit, count=0, note=note)
    steps = np.linalg.norm(series[1:] - series[:-1], axis=-1)
    count, median, p90, _ = _stats(steps)
    return SteadinessStat(
        series=name, unit=unit, count=count, median_step=median, p90_step=p90, note=note
    )


@dataclass
class TriangulatedHands:
    """Arrays indexed [frame, side, joint]; NaN where unsolved."""

    views: tuple[str, ...]
    frames: np.ndarray  # (F,) analysis frames on the reference clock
    pose_frames: np.ndarray  # (F,)
    points: np.ndarray  # (F, 2, J, 3) world mm, raw DLT
    smoothed: np.ndarray  # (F, 2, J, 3)
    observed: np.ndarray  # (F, 2, J, V) bool: view had a matched detection for the joint
    used: np.ndarray  # (F, 2, J, V) bool: view contributed to the final point
    reprojection_px: np.ndarray  # (F, 2, J, V) final point's error in each observing view
    dataset: np.ndarray  # (F, 2, J, 3) dataset world mm at the reference pose frame (NaN gaps)
    dataset_confidence: np.ndarray  # (F, 2)
    match_distance_px: np.ndarray  # (F, 2, V)


def triangulate_hands(
    repository_root: Path,
    *,
    views: Iterable[str],
    hands: Mapping[str, ViewHands],
    rig: CameraRig,
    members: PoseMembers,
    reference_view: str = DEFAULT_REFERENCE_VIEW,
    frame_count: int = DEFAULT_FRAME_COUNT,
    triangulator: Triangulator = "athena",
    reprojection_filter_px: float = DEFAULT_REPROJECTION_FILTER_PX,
    match_threshold_px: float = DEFAULT_MATCH_THRESHOLD_PX,
    dataset_confidence_floor: float = 0.5,
) -> tuple[TriangulatedHands, dict[str, ViewAlignment]]:
    views = tuple(views)
    view_count = len(views)
    joint_count = len(COMMON_JOINTS)
    reference_rule = rig.clock_rule(reference_view)
    frames = np.arange(frame_count)
    pose_frames = np.array([reference_rule.pose_frame(int(p)) for p in frames])

    observed_px = np.full((frame_count, 2, joint_count, view_count, 2), np.nan)
    match_distance = np.full((frame_count, 2, view_count), np.nan)
    residuals: dict[str, set[int]] = {view: set() for view in views}
    matched = {view: 0 for view in views}
    unmatched = {view: 0 for view in views}
    view_pose_frames = np.zeros((frame_count, view_count), dtype=np.int64)
    dataset_points = np.full((frame_count, 2, joint_count, 3), np.nan)
    dataset_confidence = np.zeros((frame_count, 2))

    def dataset_hand(pose_frame: int, side: str) -> tuple[np.ndarray, float] | None:
        key = str(pose_frame)
        index = str(DATASET_SIDE_INDEX[side])
        landmarks = members.landmarks3d.get(key)
        if landmarks is None:
            return None
        confidence = float(members.confidences[key][index])
        return np.asarray(landmarks[index], dtype=np.float64), confidence

    for f, (frame, pose_frame) in enumerate(zip(frames, pose_frames, strict=True)):
        for s, side in enumerate(SIDES):
            hand = dataset_hand(int(pose_frame), side)
            if hand is not None:
                dataset_points[f, s] = hand[0][DATASET_INDICES]
                dataset_confidence[f, s] = hand[1]
        for v, view in enumerate(views):
            rule = rig.clock_rule(view)
            view_frame, residual = nearest_analysis_frame(rule, int(pose_frame))
            residuals[view].add(int(residual))
            view_pose_frame = rule.pose_frame(view_frame)
            view_pose_frames[f, v] = view_pose_frame
            detections = hands[view].frames.get(view_frame, [])
            if not detections:
                continue
            wrists: dict[str, np.ndarray] = {}
            for side in SIDES:
                hand = dataset_hand(view_pose_frame, side)
                if hand is None or hand[1] < dataset_confidence_floor:
                    continue
                pose_arg = view_pose_frame if is_ego(view) else None
                wrists[side] = rig.project(view, hand[0][ASSEMBLY101_WRIST_INDEX], pose_arg)[0]
            assignment = match_hands_to_dataset(detections, wrists, threshold_px=match_threshold_px)
            matched[view] += len(assignment)
            unmatched[view] += len(detections) - len(assignment)
            for side, (index, distance) in assignment.items():
                s = DATASET_SIDE_INDEX[side]
                observed_px[f, s, :, v] = detections[index][METHOD_INDICES]
                match_distance[f, s, v] = distance

    observed = ~np.isnan(observed_px[..., 0])
    points = np.full((frame_count, 2, joint_count, 3), np.nan)
    used = np.zeros(observed.shape, dtype=bool)
    reprojection = np.full(observed.shape, np.nan)
    ego_present = any(is_ego(view) for view in views)

    def reproject_all(f: int, estimate: np.ndarray) -> np.ndarray:
        """(2, J, V) pixel error of `estimate` (2, J, 3) in every observing view."""
        errors = np.full((2, joint_count, view_count), np.nan)
        flat = estimate.reshape(-1, 3)
        valid = ~np.isnan(flat[:, 0])
        if not valid.any():
            return errors
        for v, view in enumerate(views):
            pose_arg = int(view_pose_frames[f, v]) if is_ego(view) else None
            projected = np.full((flat.shape[0], 2), np.nan)
            projected[valid] = rig.project(view, flat[valid], pose_arg)
            error = np.linalg.norm(projected - observed_px[f, :, :, v].reshape(-1, 2), axis=1)
            errors[:, :, v] = error.reshape(2, joint_count)
        return errors

    if triangulator == "rig":
        for f in range(frame_count):
            for s in range(2):
                if not observed[f, s].any():
                    continue
                pose_map = {
                    view: int(view_pose_frames[f, v])
                    for v, view in enumerate(views)
                    if is_ego(view)
                }
                result = rig.triangulate(
                    {view: observed_px[f, s, :, v] for v, view in enumerate(views)},
                    pose_map or None,
                    reproj_filter_px=reprojection_filter_px,
                    min_views=2,
                )
                points[f, s] = result.points
                used[f, s] = result.used
                reprojection[f, s] = result.reprojection_px
        smoothed = _smooth_rig(points)
    else:
        payload: dict[str, np.ndarray] = {}
        intrinsics = np.stack([rig._intrinsics[view] for view in views])
        if not ego_present:
            # One batch: static extrinsics are shared by every frame.
            extrinsics = np.stack([rig.projection_matrix(view) for view in views])
            flat_px = observed_px.reshape(-1, view_count, 2).transpose(1, 0, 2)  # (V, N, 2)
            undist = np.stack([rig.undistort(view, flat_px[v]) for v, view in enumerate(views)])
            payload["batch_count"] = np.array(1)
            payload["undist_0"] = undist
            payload["px_0"] = _undistorted_pixels(undist, intrinsics)
            payload["extrinsics_0"] = extrinsics
            payload["intrinsics_0"] = intrinsics
        else:
            payload["batch_count"] = np.array(frame_count)
            for f in range(frame_count):
                extrinsics = np.stack(
                    [
                        rig.projection_matrix(
                            view, int(view_pose_frames[f, v]) if is_ego(view) else None
                        )
                        for v, view in enumerate(views)
                    ]
                )
                flat_px = observed_px[f].reshape(-1, view_count, 2).transpose(1, 0, 2)
                undist = np.stack([rig.undistort(view, flat_px[v]) for v, view in enumerate(views)])
                payload[f"undist_{f}"] = undist
                payload[f"px_{f}"] = _undistorted_pixels(undist, intrinsics)
                payload[f"extrinsics_{f}"] = extrinsics
                payload[f"intrinsics_{f}"] = intrinsics
        payload["reproj_threshold"] = np.array(reprojection_filter_px)
        payload["fps"] = np.array(ANALYSIS_FPS)
        payload["smooth_shape"] = np.array([frame_count, 2 * joint_count, 3])
        result = run_athena_worker(repository_root, payload)
        if not ego_present:
            points = result["points3d_0"].reshape(frame_count, 2, joint_count, 3)
        else:
            points = np.stack(
                [result[f"points3d_{f}"].reshape(2, joint_count, 3) for f in range(frame_count)]
            )
        smoothed = result["smoothed"].reshape(frame_count, 2, joint_count, 3)
        for f in range(frame_count):
            reprojection[f] = reproject_all(f, points[f])
        # ATHENA returns only the points; a view counts as contributing when its final
        # reprojection error is within the filter threshold (the loop's own acceptance rule).
        used = observed & (reprojection <= reprojection_filter_px)
        solved = ~np.isnan(points[..., 0])
        used &= solved[..., None]

    alignments: dict[str, ViewAlignment] = {}
    for v, view in enumerate(views):
        rule = rig.clock_rule(view)
        distances = match_distance[:, :, v].ravel()
        count, median, p90, _ = _stats(distances)
        alignments[view] = ViewAlignment(
            view=view,
            run_directory=hands[view].run_directory.as_posix(),
            proxy_fingerprint=hands[view].proxy_fingerprint,
            pose_offset_frames=rule.pose_offset_frames,
            analysis_frame_shift=(reference_rule.pose_offset_frames - rule.pose_offset_frames) / 2,
            residual_pose_frames_used=tuple(sorted(residuals[view])),
            frames_with_detections=hands[view].frames_with_hands,
            detections_matched=matched[view],
            detections_unmatched=unmatched[view],
            match_distance_median_px=median,
            match_distance_p90_px=p90,
        )
    return (
        TriangulatedHands(
            views=views,
            frames=frames,
            pose_frames=pose_frames,
            points=points,
            smoothed=smoothed,
            observed=observed,
            used=used,
            reprojection_px=reprojection,
            dataset=dataset_points,
            dataset_confidence=dataset_confidence,
            match_distance_px=match_distance,
        ),
        alignments,
    )


def _undistorted_pixels(undist: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Normalised undistorted coords (V, N, 2) -> pixel coords of the *undistorted* image.

    ATHENA's filter reprojects with `K @ E` and no distortion, so its pixel-space residual is
    only meaningful against pixels expressed in the same undistorted image.
    """
    result = np.full_like(undist, np.nan)
    for v in range(undist.shape[0]):
        k = intrinsics[v]
        result[v, :, 0] = k[0, 0] * undist[v, :, 0] + k[0, 2]
        result[v, :, 1] = k[1, 1] * undist[v, :, 1] + k[1, 2]
    return result


def savgol(data: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """Savitzky-Golay filter (scipy's default `interp` edge mode) in numpy for the rig path."""
    half = window // 2
    offsets = np.arange(-half, half + 1, dtype=np.float64)
    vandermonde = np.vander(offsets, polyorder + 1, increasing=True)
    coefficients = np.linalg.pinv(vandermonde)[0]  # evaluates the fitted polynomial at 0
    smoothed = np.convolve(data, coefficients[::-1], mode="same")
    if data.shape[0] >= window:
        for start, sl in ((0, slice(0, half)), (data.shape[0] - window, slice(-half, None))):
            fit = np.polyfit(np.arange(window), data[start : start + window], polyorder)
            positions = np.arange(data.shape[0])[sl] - start
            smoothed[sl] = np.polyval(fit, positions)
    return smoothed


def _smooth_rig(
    points: np.ndarray, fps: float = ANALYSIS_FPS, cutoff_hz: float = 20.0
) -> np.ndarray:
    """Savitzky-Golay smoothing matching ATHENA's `_smooth3d` rules (window from the cutoff,
    NaN gaps interpolated first, gaps longer than five frames restored)."""
    frame_count = points.shape[0]
    window = int(fps / cutoff_hz * 2 + 1)
    window = max(3, window | 1)
    window = min(window, frame_count | 1)
    flat = points.reshape(frame_count, -1)
    out = flat.copy()
    for column in range(flat.shape[1]):
        data = flat[:, column]
        valid = ~np.isnan(data)
        if valid.sum() < window:
            continue
        indices = np.arange(frame_count)
        interpolated = np.interp(indices, indices[valid], data[valid])
        filtered = savgol(interpolated, window, 3)
        filtered[_long_nan_runs(~valid, 5)] = np.nan
        out[:, column] = filtered
    return out.reshape(points.shape)


def _long_nan_runs(nan_mask: np.ndarray, min_length: int) -> np.ndarray:
    padded = np.concatenate(([0], nan_mask.astype(np.int8), [0]))
    edges = np.where(np.abs(np.diff(padded)) == 1)[0].reshape(-1, 2)
    result = np.zeros_like(nan_mask, dtype=bool)
    for start, end in edges:
        if end - start > min_length:
            result[start:end] = True
    return result


# -- reporting ------------------------------------------------------------------------------------


def summarize(
    result: TriangulatedHands,
    *,
    reference_view: str,
    rig: CameraRig,
    wilor_reference: ViewHands | None,
) -> tuple[HandSummary, ...]:
    summaries = []
    for s, side in enumerate(SIDES):
        points = result.points[:, s]
        smoothed = result.smoothed[:, s]
        dataset = result.dataset[:, s]
        dataset_ok = result.dataset_confidence[:, s] >= 0.5
        raw_error = np.linalg.norm(points - dataset, axis=-1)
        raw_error[~dataset_ok] = np.nan
        smooth_error = np.linalg.norm(smoothed - dataset, axis=-1)
        smooth_error[~dataset_ok] = np.nan
        solved = ~np.isnan(points[..., 0])
        counts = result.used[:, s].sum(axis=-1).astype(float)
        counts[~solved] = np.nan
        per_view = []
        for v, view in enumerate(result.views):
            observed = result.observed[:, s, :, v]
            used = result.used[:, s, :, v]
            errors = result.reprojection_px[:, s, :, v]
            used_errors = errors[used]
            observed_errors = errors[observed & np.isfinite(errors)]
            per_view.append(
                ViewReprojection(
                    view=view,
                    points_observed=int(observed.sum()),
                    points_used=int(used.sum()),
                    rms_px_used=float(np.sqrt(np.mean(used_errors**2)))
                    if used_errors.size
                    else None,
                    median_px_used=float(np.median(used_errors)) if used_errors.size else None,
                    rms_px_observed=(
                        float(np.sqrt(np.mean(observed_errors**2)))
                        if observed_errors.size
                        else None
                    ),
                )
            )
        per_joint = []
        for j, name in enumerate(COMMON_JOINT_NAMES):
            count, median, p90, _ = _stats(raw_error[:, j])
            per_joint.append(JointStat(joint=name, count=count, median_mm=median, p90_mm=p90))
        wrist_raw = points[:, WRIST]
        wrist_smooth = smoothed[:, WRIST]
        wrist_dataset = np.where(dataset_ok[:, None], dataset[:, WRIST], np.nan)
        steadiness = [
            _steadiness(wrist_raw, "triangulated_wrist_raw", "mm/frame"),
            _steadiness(wrist_smooth, "triangulated_wrist_smoothed", "mm/frame"),
            _steadiness(wrist_dataset, "dataset_wrist", "mm/frame"),
        ]
        # Like-for-like in the reference view's raw pixels.
        pose_arg = None
        valid = ~np.isnan(wrist_raw[:, 0])
        projected = np.full((wrist_raw.shape[0], 2), np.nan)
        if valid.any():
            projected[valid] = rig.project(reference_view, wrist_raw[valid], pose_arg)
        steadiness.append(
            _steadiness(projected, f"triangulated_wrist_raw_projected_{reference_view}", "px/frame")
        )
        if wilor_reference is not None:
            wilor_2d, wilor_3d = _wilor_wrist_tracks(wilor_reference, result, s, rig)
            steadiness.append(
                _steadiness(
                    wilor_2d,
                    f"wilor_wrist_2d_{reference_view}",
                    "px/frame",
                    note="WiLoR detection matched to the dataset wrist per frame",
                )
            )
            if wilor_reference.camera_translation_attached:
                steadiness.append(
                    _steadiness(
                        wilor_3d * 1000.0,
                        f"wilor_wrist_camera_frame_{reference_view}",
                        "WiLoR-metres x1000 per frame",
                        note=(
                            "WiLoR joint 0 plus pred_cam_t_full from the native evidence: the "
                            "model's own camera-frame wrist under its scaled focal length, not "
                            "calibrated to this camera and not millimetres"
                        ),
                    )
                )
            else:
                steadiness.append(
                    _steadiness(
                        wilor_3d,
                        f"wilor_wrist_wrist_rooted_{reference_view}",
                        "native units/frame",
                        note="WiLoR joints_3d_camera_relative without the camera translation",
                    )
                )
        summaries.append(
            HandSummary(
                side=side,
                frames_with_solution=int(solved.any(axis=-1).sum()),
                frames_with_wrist=int(solved[:, WRIST].sum()),
                frames_dataset_present=int(dataset_ok.sum()),
                mean_contributing_views=float(np.nanmean(counts)) if solved.any() else None,
                disagreement_raw=_group_stats(raw_error),
                disagreement_smoothed=_group_stats(smooth_error),
                per_joint_raw=tuple(per_joint),
                per_view_reprojection=tuple(per_view),
                steadiness=tuple(steadiness),
            )
        )
    return tuple(summaries)


def _wilor_wrist_tracks(
    wilor: ViewHands, result: TriangulatedHands, side_index: int, rig: CameraRig
) -> tuple[np.ndarray, np.ndarray]:
    """Per reference frame, the WiLoR detection nearest the projected dataset wrist of this
    side (handedness-agnostic, same threshold as the triangulation matching): 2D raw px and
    the camera-relative joint 0."""
    frame_count = result.frames.shape[0]
    track_2d = np.full((frame_count, 2), np.nan)
    track_3d = np.full((frame_count, 3), np.nan)
    for f in range(frame_count):
        detections = wilor.frames.get(int(result.frames[f]), [])
        target = result.dataset[f, side_index, WRIST]
        if not detections or np.isnan(target[0]) or result.dataset_confidence[f, side_index] < 0.5:
            continue
        wrist_px = rig.project(wilor.view, target, None)[0]
        distances = [float(np.linalg.norm(det[0] - wrist_px)) for det in detections]
        best = int(np.argmin(distances))
        if distances[best] > DEFAULT_MATCH_THRESHOLD_PX:
            continue
        track_2d[f] = detections[best][0]
        relative = wilor.camera_relative_3d[int(result.frames[f])][best]
        if relative is not None:
            track_3d[f] = relative[0]
    return track_2d, track_3d


# -- run ----------------------------------------------------------------------------------------


def write_hands_jsonl(path: Path, result: TriangulatedHands) -> None:
    def clean(array: np.ndarray) -> list:
        return [
            None if np.isnan(row[0]) else [round(float(x), 3) for x in row]
            for row in np.asarray(array).reshape(-1, array.shape[-1])
        ]

    with path.open("w", encoding="utf-8") as handle:
        for f in range(result.frames.shape[0]):
            hands = []
            for s, side in enumerate(SIDES):
                if not (~np.isnan(result.points[f, s, :, 0])).any():
                    continue
                solved = ~np.isnan(result.points[f, s, :, 0])
                dataset_ok = result.dataset_confidence[f, s] >= 0.5
                error = np.linalg.norm(result.points[f, s] - result.dataset[f, s], axis=-1)
                hands.append(
                    {
                        "side": side,
                        "joints_world_mm": clean(result.points[f, s]),
                        "joints_smoothed_world_mm": clean(result.smoothed[f, s]),
                        "contributing_views": result.used[f, s].sum(axis=-1).tolist(),
                        "reprojection_px": {
                            view: [
                                None if np.isnan(e) else round(float(e), 2)
                                for e in result.reprojection_px[f, s, :, v]
                            ]
                            for v, view in enumerate(result.views)
                        },
                        "dataset_confidence": round(float(result.dataset_confidence[f, s]), 4),
                        "disagreement_vs_dataset_mm": (
                            [
                                None if (np.isnan(e) or not ok) else round(float(e), 2)
                                for e, ok in zip(error, solved, strict=True)
                            ]
                            if dataset_ok
                            else None
                        ),
                    }
                )
            handle.write(
                json.dumps(
                    {
                        "analysis_frame_index": int(result.frames[f]),
                        "pose_frame_index": int(result.pose_frames[f]),
                        "hands": hands,
                    }
                )
                + "\n"
            )


def build_run(
    repository_root: Path,
    *,
    views: tuple[str, ...],
    hand_source: HandSource,
    output_root: Path,
    run_overrides: Mapping[str, str] | None = None,
    triangulator: Triangulator = "athena",
    reprojection_filter_px: float = DEFAULT_REPROJECTION_FILTER_PX,
    match_threshold_px: float = DEFAULT_MATCH_THRESHOLD_PX,
    frame_count: int = DEFAULT_FRAME_COUNT,
    reference_view: str = DEFAULT_REFERENCE_VIEW,
    wilor_reference_run: Path | None = None,
    overwrite: bool = False,
) -> Path:
    started = time.monotonic()
    repository_root = repository_root.resolve()
    if len(views) < 2:
        raise ValueError("triangulation needs at least two views")
    overrides = dict(run_overrides or {})
    rig = CameraRig.load(repository_root, views=tuple(dict.fromkeys((*views, reference_view))))
    members = PoseMembers(repository_root)
    hands = {
        view: load_view_hands(
            repository_root, view, run_directory_for(view, hand_source, overrides), hand_source
        )
        for view in views
    }
    if triangulator == "athena" and not athena_available(repository_root):
        raise FileNotFoundError(
            f"ATHENA interpreter {ATHENA_PYTHON} or worker {ATHENA_WORKER} missing; "
            "use --triangulator rig"
        )
    result, alignments = triangulate_hands(
        repository_root,
        views=views,
        hands=hands,
        rig=rig,
        members=members,
        reference_view=reference_view,
        frame_count=frame_count,
        triangulator=triangulator,
        reprojection_filter_px=reprojection_filter_px,
        match_threshold_px=match_threshold_px,
    )
    first_output = time.monotonic() - started
    wilor_reference = None
    if wilor_reference_run is not None:
        wilor_reference = load_view_hands(
            repository_root, reference_view, wilor_reference_run, "wilor"
        )
    summaries = summarize(
        result, reference_view=reference_view, rig=rig, wilor_reference=wilor_reference
    )

    root = repository_root / output_root
    if root.exists() and not overwrite:
        raise FileExistsError(f"{root} exists; pass --overwrite")
    root.mkdir(parents=True, exist_ok=True)
    hands_path = root / "hands.jsonl"
    write_hands_jsonl(hands_path, result)
    np.savez_compressed(
        root / "hands.npz",
        frames=result.frames,
        pose_frames=result.pose_frames,
        points=result.points,
        smoothed=result.smoothed,
        observed=result.observed,
        used=result.used,
        reprojection_px=result.reprojection_px,
        dataset=result.dataset,
        dataset_confidence=result.dataset_confidence,
        views=np.array(result.views),
    )
    dataset_manifest = repository_root / DATASET_REFERENCE_RUN / "manifest.json"
    inputs = [
        _fingerprint(
            repository_root / hands[view].run_directory / "observations.jsonl", repository_root
        )
        for view in views
    ]
    inputs.append(
        _fingerprint(repository_root / "configs/assembly101/clock_rules.json", repository_root)
    )
    for view in rig.views:
        inputs.append(
            _fingerprint(
                repository_root / f"configs/assembly101/{view.lower()}_camera_estimate.json",
                repository_root,
            )
        )
    if wilor_reference is not None:
        inputs.append(
            _fingerprint(
                repository_root / wilor_reference.run_directory / "observations.jsonl",
                repository_root,
            )
        )
    detail = (
        "athena.triangulaterefine._triangulate_with_filtering (equal-weight DLT; drop the worst "
        f"camera while its error exceeds {reprojection_filter_px:g} px, <= 3 passes, min 2 cams) "
        "on undistorted normalised coordinates, reprojection residuals in the undistorted image; "
        "athena.triangulaterefine._smooth3d Savitzky-Golay (20 Hz cutoff, order 3); executed by "
        f"{ATHENA_PYTHON}"
        if triangulator == "athena"
        else "multiview_geometry.CameraRig.triangulate (equal-weight DLT, leave-one-out drop while "
        f"the worst error exceeds {reprojection_filter_px:g} px, min 2 views) with an in-repo "
        "Savitzky-Golay matching ATHENA's window rule"
    )
    manifest = AthenaHandsManifest(
        run_id=root.name,
        recording_id=RECORDING_ID,
        hand_source=hand_source,
        triangulator=triangulator,
        triangulator_detail=detail,
        athena_revision=ATHENA_REVISION if triangulator == "athena" else None,
        views=views,
        reference_view=reference_view,
        frame_count=frame_count,
        analysis_fps=ANALYSIS_FPS,
        reprojection_filter_px=reprojection_filter_px,
        match_threshold_px=match_threshold_px,
        smoothing="savitzky_golay_20hz_order3_nan_runs_over_5_frames_restored",
        joint_names=COMMON_JOINT_NAMES,
        joint_mapping_dataset_to_method=tuple((d, m) for _, d, m in COMMON_JOINTS),
        alignments=tuple(alignments[view] for view in views),
        hands=summaries,
        hands_path="hands.jsonl",
        dataset_reference=_fingerprint(dataset_manifest, repository_root),
        input_artifacts=tuple(inputs),
        runtime_seconds=time.monotonic() - started,
        time_to_first_output_seconds=first_output,
        claim_boundaries=CLAIM_BOUNDARIES,
    )
    manifest_path = root / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return manifest_path


def load_manifest(run_directory: Path) -> AthenaHandsManifest:
    return AthenaHandsManifest.model_validate_json(
        (run_directory / "manifest.json").read_text(encoding="utf-8")
    )


def print_summary(manifest: AthenaHandsManifest) -> None:
    print(
        f"{manifest.run_id}: {manifest.hand_source} on {'+'.join(manifest.views)} via "
        f"{manifest.triangulator}; {manifest.runtime_seconds:.1f} s"
    )
    for alignment in manifest.alignments:
        print(
            f"  {alignment.view:13s} shift {alignment.analysis_frame_shift:+.1f} frames, residual "
            f"{alignment.residual_pose_frames_used}, detections in "
            f"{alignment.frames_with_detections} frames, matched {alignment.detections_matched}, "
            f"unmatched {alignment.detections_unmatched}, match median "
            f"{alignment.match_distance_median_px} px"
        )
    for hand in manifest.hands:
        groups = {g.group: g for g in hand.disagreement_raw}
        smooth = {g.group: g for g in hand.disagreement_smoothed}
        print(
            f"  {hand.side}: solved {hand.frames_with_solution}/{manifest.frame_count} frames "
            f"(wrist {hand.frames_with_wrist}), mean views {hand.mean_contributing_views}"
        )
        for name in ("wrist", "fingertips", "all_joints"):
            g, sm = groups[name], smooth[name]
            print(
                f"    {name:11s} raw median {g.median_mm} p90 {g.p90_mm} mm (n={g.count}); "
                f"smoothed median {sm.median_mm} p90 {sm.p90_mm}"
            )
        for view in hand.per_view_reprojection:
            print(
                f"    {view.view:13s} used {view.points_used}/{view.points_observed} points, "
                f"rms used {view.rms_px_used} px, rms observed {view.rms_px_observed} px"
            )
        for stat in hand.steadiness:
            print(
                f"    step {stat.series}: median {stat.median_step} p90 {stat.p90_step} {stat.unit}"
            )


# -- per-view 2D check ----------------------------------------------------------------------------


class View2DCheck(VersionedModel):
    """One view's detections against the dataset's own 2D landmarks (quality proxy, not
    accuracy: the dataset 2D is its tracker's projection)."""

    manifest_kind: Literal["hands_2d_vs_dataset_check"] = "hands_2d_vs_dataset_check"
    view: str
    hand_source: HandSource
    run_directory: str
    frame_count: int = Field(ge=1)
    frames_with_detection: int = Field(ge=0)
    frames_with_two_detections: int = Field(ge=0)
    dataset_hands_in_view: int = Field(ge=0)
    dataset_hands_with_detection_within_threshold: int = Field(ge=0)
    wrist_distance_median_px: float | None
    wrist_distance_p90_px: float | None
    threshold_px: float
    elapsed_seconds: float | None
    time_to_first_output_seconds: float | None
    roi: str | None
    roi_source: str | None
    claim_boundary: str


def check_view_2d(
    repository_root: Path,
    view: str,
    run_directory: Path,
    *,
    hand_source: HandSource = "mediapipe",
    frame_count: int = DEFAULT_FRAME_COUNT,
    threshold_px: float = DEFAULT_MATCH_THRESHOLD_PX,
    rig: CameraRig | None = None,
    members: PoseMembers | None = None,
) -> View2DCheck:
    repository_root = repository_root.resolve()
    rig = rig or CameraRig.load(repository_root, views=(view,))
    members = members or PoseMembers(repository_root)
    hands = load_view_hands(repository_root, view, run_directory, hand_source)
    rule = rig.clock_rule(view)
    width, height = raw_image_size(view)
    distances: list[float] = []
    dataset_in_view = 0
    within = 0
    for frame in range(frame_count):
        pose_frame = rule.pose_frame(frame)
        landmarks = members.landmarks3d.get(str(pose_frame))
        if landmarks is None:
            continue
        detections = hands.frames.get(frame, [])
        for index in ("0", "1"):
            if float(members.confidences[str(pose_frame)][index]) < 0.5:
                continue
            world = np.asarray(landmarks[index], dtype=np.float64)[ASSEMBLY101_WRIST_INDEX]
            wrist = rig.project(view, world, pose_frame if is_ego(view) else None)[0]
            if not (0 <= wrist[0] < width and 0 <= wrist[1] < height):
                continue
            dataset_in_view += 1
            if not detections:
                continue
            nearest = min(float(np.linalg.norm(det[0] - wrist)) for det in detections)
            distances.append(nearest)
            within += nearest <= threshold_px
    manifest = json.loads((repository_root / run_directory / "manifest.json").read_text("utf-8"))
    metadata = manifest[SOURCE_MANIFEST_KEY[hand_source]]
    settings = metadata.get("runtime_settings", {})
    measurements = metadata.get("measurements", {})
    array = np.asarray(distances)
    count, median, p90, _ = _stats(array) if array.size else (0, None, None, None)
    check = View2DCheck(
        view=view,
        hand_source=hand_source,
        run_directory=run_directory.as_posix(),
        frame_count=frame_count,
        frames_with_detection=sum(1 for f in range(frame_count) if hands.frames.get(f)),
        frames_with_two_detections=sum(
            1 for f in range(frame_count) if len(hands.frames.get(f, [])) >= 2
        ),
        dataset_hands_in_view=dataset_in_view,
        dataset_hands_with_detection_within_threshold=int(within),
        wrist_distance_median_px=median,
        wrist_distance_p90_px=p90,
        threshold_px=threshold_px,
        elapsed_seconds=measurements.get("elapsed_seconds"),
        time_to_first_output_seconds=measurements.get("time_to_first_usable_output_seconds"),
        roi=settings.get("roi"),
        roi_source=settings.get("roi_source"),
        claim_boundary=(
            "Nearest detected wrist to each dataset wrist (raw sensor px, handedness-agnostic); "
            "the dataset 2D is its own tracker's projection, so this is cross-source distance, "
            "not accuracy."
        ),
    )
    (repository_root / run_directory / "dataset_2d_check.json").write_text(
        check.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return check


def check_main() -> None:
    parser = argparse.ArgumentParser(description="Per-view 2D hands vs dataset 2D landmarks.")
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--views", nargs="+", default=list(STATIC_VIEWS))
    parser.add_argument("--hand-source", choices=("mediapipe", "wilor"), default="mediapipe")
    parser.add_argument("--run-dir", type=_parse_override, action="append", default=[])
    parser.add_argument("--frame-count", type=int, default=DEFAULT_FRAME_COUNT)
    parser.add_argument("--threshold-px", type=float, default=DEFAULT_MATCH_THRESHOLD_PX)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    overrides = dict(args.run_dir)
    rig = CameraRig.load(root, views=tuple(args.views))
    members = PoseMembers(root)
    for view in args.views:
        check = check_view_2d(
            root,
            view,
            run_directory_for(view, args.hand_source, overrides),
            hand_source=args.hand_source,
            frame_count=args.frame_count,
            threshold_px=args.threshold_px,
            rig=rig,
            members=members,
        )
        print(
            f"{view:13s} frames with hand {check.frames_with_detection}/{check.frame_count} "
            f"(two {check.frames_with_two_detections}); dataset wrists in view "
            f"{check.dataset_hands_in_view}, within {check.threshold_px:g} px "
            f"{check.dataset_hands_with_detection_within_threshold}; wrist median "
            f"{check.wrist_distance_median_px} px p90 {check.wrist_distance_p90_px}; "
            f"runtime {check.elapsed_seconds} s, first output {check.time_to_first_output_seconds}"
        )


def _parse_override(value: str) -> tuple[str, str]:
    view, _, path = value.partition("=")
    if not view or not path:
        raise argparse.ArgumentTypeError("expected VIEW=RUN_DIRECTORY")
    return view, path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--views", nargs="+", default=list(STATIC_VIEWS))
    parser.add_argument("--hand-source", choices=("mediapipe", "wilor"), default="mediapipe")
    parser.add_argument("--run-dir", type=_parse_override, action="append", default=[])
    parser.add_argument("--triangulator", choices=("athena", "rig"), default="athena")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--reprojection-filter-px", type=float, default=DEFAULT_REPROJECTION_FILTER_PX
    )
    parser.add_argument("--match-threshold-px", type=float, default=DEFAULT_MATCH_THRESHOLD_PX)
    parser.add_argument("--frame-count", type=int, default=DEFAULT_FRAME_COUNT)
    parser.add_argument("--reference-view", default=DEFAULT_REFERENCE_VIEW)
    parser.add_argument(
        "--wilor-reference-run",
        type=Path,
        default=None,
        help="Single-view WiLoR run on the reference view for the steadiness comparison",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest_path = build_run(
        args.repository_root,
        views=tuple(args.views),
        hand_source=args.hand_source,
        output_root=args.output_root,
        run_overrides=dict(args.run_dir),
        triangulator=args.triangulator,
        reprojection_filter_px=args.reprojection_filter_px,
        match_threshold_px=args.match_threshold_px,
        frame_count=args.frame_count,
        reference_view=args.reference_view,
        wilor_reference_run=args.wilor_reference_run,
        overwrite=args.overwrite,
    )
    print_summary(load_manifest(manifest_path.parent))
    print(manifest_path)


if __name__ == "__main__":
    main()
