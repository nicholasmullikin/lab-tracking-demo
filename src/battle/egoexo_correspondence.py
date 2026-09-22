"""battle-egoexo-correspondence: LM-EEC ego-exo correspondence on twelve first-minute keyframes.

Track 7 of the overnight multicam pass.  Given the human/agent-corrected C10379 mask of a
part, LM-EEC (NeurIPS 2025, SAM 2.1 base-plus with a dual long-term memory and a
memory-view mixture of experts) predicts the mask of the same part in the ego view at the
same instant, with no table-plane or hull assumption.  The prediction is then compared with
two other estimates of the same region: the ego SAM3 mask of the Sep 16 ego run and, when
Track 5's visual hull exists, the hull projected into the ego camera through the calibrated
rig.  Both comparisons are cross-source disagreement; nothing here is accuracy.

Four subcommands, three of them CPU-only:

* `prepare` writes `runs/egoexo-correspondence-first-minute-20260918/`: the twelve keyframe
  pairs as JPEGs in the per-view frame directories LM-EEC's predictor reads
  (`frames/<view>/<key>.jpg`, the same `<key>` in both views), the exo query masks per part,
  the ego SAM3 masks per part for comparison, a typed `pairs.json`, and the queue job file.
* `run` is the GPU step.  It is never executed here directly: the queue calls
  `scripts/lm_eec_driver.py` under LM-EEC's own interpreter.  `run` prints or launches that
  same argv and refuses to run with CUDA hidden unless `--cpu-smoke` is given.
* `evaluate` reads the driver's `predictions.json`, projects the hull into the ego view for
  every keyframe that has voxels, computes IoU per part per keyframe against both references
  and writes a typed `manifest.json` with claim boundaries and the five pre-accuracy measures.
* `rerun` writes `correspondence.rrd`: per keyframe the exo frame with the query cut-out
  beside the ego frame with the prediction, the hull projection and the ego SAM3 cut-out.

Clock.  The static C10379 video lags the pose clock by 9 pose frames (60 fps), the HMC cameras
by 0, so exo analysis frame `p` shows pose frame `17649 + 2p` and the ego frame showing that
instant is `p + 4.5`.  The half frame is rounded *down* (`p + 4`): the ego frame is one pose
frame (16.7 ms) before the exo frame, the same integer-timeline convention the Track 1
synchronized recordings use, and the residual is recorded on every pair.

Licences.  LM-EEC's released checkpoints are research artefacts of a NeurIPS paper built on
SAM 2 (Apache-2.0 code); Assembly101 is CC BY-NC 4.0.  Both are noted in `docs/LICENSES.md`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image
from pydantic import Field

from . import mask_cache
from .assembly101_fetch_view import RECORDING_ID, proxy_path
from .assembly101_pose_schemas import Assembly101ClockRule
from .digest_cache import sha256_file as _sha256
from .fs_common import relative_uri
from .mask_ops import mask_iou
from .multiview_geometry import CameraRig
from .observations import observations_by_frame
from .schemas import ArtifactFingerprint, FrameObservations, VersionedModel, fingerprint

FrameReader = Callable[[Path, int], np.ndarray]

RUN_ID = "egoexo-correspondence-first-minute-20260918"
RUN_ROOT = Path("runs") / RUN_ID
QUEUE_DIR = Path("runs/overnight-multicam-20260918")
QUEUE_LOG = QUEUE_DIR / "queue.log"
QUEUE_JOB_FILE = QUEUE_DIR / "jobs_t7_egoexo.json"
QUEUE_JOB_NAME = "lm-eec-egoexo-correspondence"
QUEUE_TIMEOUT_S = 1800
DRIVER_SCRIPT = Path("scripts/lm_eec_driver.py")

EXO_CAMERA = "C10379"
EGO_CAMERA = "HMC_21110305"
EXO_VIEW_ID = "static-c10379"
EGO_VIEW_ID = "ego-hmc21110305"
PARTS: tuple[str, ...] = ("chassis", "interior", "rear_body", "cabin")
HANDS: tuple[str, ...] = ("left_hand", "right_hand")
KEYFRAME_STEP = 150
KEYFRAME_COUNT = 12
ANALYSIS_FPS = 30
JPEG_QUALITY = 95

EXO_QUERY_RUN = Path("runs/ensemble-reference-first-minute-v1")
EGO_REFERENCE_RUN = Path(
    "runs/muggledsam-sam3-four-part-ego-focused-reassembly-ego-hmc21110305-20260916t031515z"
)
HULL_RUN = Path("runs/multiview-visual-hull-first-minute")

LM_EEC_ROOT = Path("/home/nick/src/LM-EEC")
LM_EEC_REPOSITORY = "https://github.com/juneyeeHu/LM-EEC"
LM_EEC_PINNED_COMMIT = "b37e50e50fd03ae8625e6100da37bad3dfeb6aa4"
LM_EEC_INTERPRETER = LM_EEC_ROOT / ".venv/bin/python"
LM_EEC_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
LM_EEC_IMAGE_SIZE = 480
# The authors release two checkpoints with no documentation of which direction each serves;
# their metadata is identical (epoch 60, 11,280 steps).  The file names are read literally:
# `ExoEgo` for a query in the exo view answered in the ego view.  Recorded as an assumption.
LM_EEC_CHECKPOINTS: dict[str, Path] = {
    "exo_to_ego": LM_EEC_ROOT / "checkpoints/LM-EEC-checkpoint/ExoEgo_checkpoint.pt",
    "ego_to_exo": LM_EEC_ROOT / "checkpoints/LM-EEC-checkpoint/EgoExo_checkpoint.pt",
}

Direction = Literal["exo_to_ego", "ego_to_exo"]
DIRECTIONS: tuple[Direction, ...] = ("exo_to_ego", "ego_to_exo")

PART_COLORS: dict[str, tuple[int, int, int]] = dict(mask_cache.REVIEW_COLORS)
PART_COLORS.update({"left_hand": (255, 230, 90), "right_hand": (120, 255, 200)})
PREDICTION_COLOR = (255, 60, 220)
HULL_COLOR = (255, 255, 255)

CLAIM_BOUNDARIES: tuple[str, ...] = (
    "Every IoU here compares two estimates of the same region (LM-EEC prediction, ego SAM3 "
    "mask, projected visual hull): cross-source disagreement, never accuracy of any of them. "
    "No ego-view ground truth exists for these parts.",
    "The exo query masks are the human/agent-corrected C10379 masks of the ensemble reference "
    "run; the ego SAM3 masks were seeded by a human on frame 0 and propagated by one tracker; "
    "the hull is carved from agent-seeded static-view masks and projected through fitted "
    "intrinsics and dataset extrinsics.  None of the three is ground truth.",
    "The ego video is monochrome (HMC_21110305, 10-bit mono decoded to three identical "
    "channels); LM-EEC was trained on Ego-Exo4D colour Aria frames and never saw a grey "
    "ego image or a 954x720 sensor, and both views are squashed to 480x480 by the model.",
    "The two released checkpoints carry no direction label; `ExoEgo_checkpoint.pt` is used "
    "for exo->ego on the strength of its file name only.  Both checkpoints are research "
    "artefacts of a NeurIPS 2025 paper with no separate licence file; SAM 2 code is Apache-2.0; "
    "Assembly101 is CC BY-NC 4.0 (attribution, non-commercial).",
    "Keyframe pairs are matched on the measured per-view clock rules and the half analysis "
    "frame is rounded down, so every ego frame is 16.7 ms before its exo frame; nothing is "
    "resampled.",
    "LM-EEC is a video model: in `sequence` mode the twelve keyframes of one part are fed as "
    "one twelve-frame clip so its long-term memory spans 150-frame gaps it was not trained on; "
    "`independent` mode resets the state per keyframe.  The mode is recorded in the run.",
)


# -- typed manifests ---------------------------------------------------------------------------


class ViewSpec(VersionedModel):
    view_id: str
    camera: str
    proxy: ArtifactFingerprint
    proxy_size_wh: tuple[int, int]
    raw_size_wh: tuple[int, int]
    pose_offset_frames: int
    frames_dir: str = Field(min_length=1)


class KeyframePair(VersionedModel):
    key: str = Field(pattern=r"^\d{6}$")
    exo_analysis_frame: int = Field(ge=0)
    ego_analysis_frame: int = Field(ge=0)
    exo_pose_frame: int = Field(ge=0)
    ego_pose_frame: int = Field(ge=0)
    residual_pose_frames: int = Field(
        description="ego_pose_frame - exo_pose_frame; negative means the ego frame is earlier"
    )
    exo_frame_uri: str
    ego_frame_uri: str
    exo_query_masks: dict[str, str | None]
    ego_reference_masks: dict[str, str | None]
    ego_query_masks: dict[str, str | None] = Field(default_factory=dict)
    hull_voxels_available: dict[str, bool]


class DirectionSpec(VersionedModel):
    direction: Direction
    source_view: str
    target_view: str
    source_frames_dir: str
    target_frames_dir: str
    query_mask_root: str
    objects: tuple[str, ...]
    checkpoint: str
    checkpoint_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    skipped_reason: str | None = None


class ModelIdentity(VersionedModel):
    name: Literal["LM-EEC"] = "LM-EEC"
    repository: str
    pinned_commit: str = Field(pattern=r"^[a-f0-9]{40}$")
    checkout_head: str | None = None
    config: str
    image_size: int = Field(gt=0)
    interpreter: str
    driver: ArtifactFingerprint
    install_script: ArtifactFingerprint
    backbone: str = "SAM 2.1 hiera base-plus (fine-tuned weights inside the LM-EEC checkpoint)"
    mode: Literal["sequence", "independent"]


class PairsManifest(VersionedModel):
    manifest_kind: Literal["egoexo_correspondence_pairs"] = "egoexo_correspondence_pairs"
    run_id: str
    prepared_at_utc: str
    recording_id: str
    exo: ViewSpec
    ego: ViewSpec
    frame_mapping_rule: str
    keyframe_step: int = Field(gt=0)
    keyframes: tuple[KeyframePair, ...] = Field(min_length=1)
    parts: tuple[str, ...]
    hands: tuple[str, ...]
    directions: tuple[DirectionSpec, ...] = Field(min_length=1)
    sources: dict[str, ArtifactFingerprint | None]
    clock_rules: ArtifactFingerprint
    model: ModelIdentity
    queue_job: str
    queue_job_name: str
    queue_timeout_s: int
    claim_boundaries: tuple[str, ...] = Field(min_length=1)


class Prediction(VersionedModel):
    direction: Direction
    object: str
    key: str
    mask_uri: str
    mask_area_px: int = Field(ge=0)
    empty: bool
    iou_prediction: float | None
    object_score_logit: float | None
    object_score: float | None = Field(default=None, ge=0.0, le=1.0)
    max_logit: float | None
    seconds: float = Field(ge=0.0)


class DriverRuntime(VersionedModel):
    device: str
    device_name: str | None
    torch_version: str
    cuda_version: str | None
    mode: Literal["sequence", "independent"]
    cpu_smoke: bool
    lm_eec_head: str | None
    model_load_seconds: dict[str, float]
    time_to_first_output_seconds: float | None
    total_seconds: float
    gpu_peak_vram_reserved_bytes: int | None
    gpu_peak_vram_allocated_bytes: int | None


class PredictionsFile(VersionedModel):
    manifest_kind: Literal["egoexo_correspondence_predictions"] = (
        "egoexo_correspondence_predictions"
    )
    run_id: str
    predicted_at_utc: str
    pairs_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    checkpoints: dict[str, str]
    predictions: tuple[Prediction, ...]
    skipped: tuple[str, ...]
    runtime: DriverRuntime


class PairComparison(VersionedModel):
    direction: Direction
    object: str
    key: str
    exo_analysis_frame: int
    ego_analysis_frame: int
    prediction_uri: str
    prediction_area_px: int
    prediction_empty: bool
    iou_prediction: float | None
    object_score: float | None
    reference_uri: str | None
    reference_area_px: int | None
    iou_vs_reference_mask: float | None
    hull_projection_uri: str | None
    hull_projection_area_px: int | None
    iou_vs_hull_projection: float | None
    iou_reference_vs_hull: float | None
    notes: tuple[str, ...] = ()


class IoUSummary(VersionedModel):
    count: int = Field(ge=0)
    median: float | None
    mean: float | None
    min: float | None
    max: float | None
    at_or_above_0_5: int = Field(ge=0)


class ObjectSummary(VersionedModel):
    direction: Direction
    object: str
    pairs: int
    predicted_nonempty: int
    model_iou_prediction_median: float | None
    vs_reference_mask: IoUSummary
    vs_hull_projection: IoUSummary
    reference_vs_hull: IoUSummary


class Measures(VersionedModel):
    """The five measures every track reports before any accuracy talk."""

    coverage_pairs_predicted_nonempty: int = Field(ge=0)
    coverage_pairs_total: int = Field(ge=0)
    coverage_pairs_with_reference_mask: int = Field(ge=0)
    coverage_pairs_with_hull_projection: int = Field(ge=0)
    time_to_first_output_seconds: float | None
    runtime_seconds: float | None
    driver_seconds: float | None
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


class EvaluationManifest(VersionedModel):
    manifest_kind: Literal["egoexo_correspondence_evaluation"] = "egoexo_correspondence_evaluation"
    run_id: str
    evaluated_at_utc: str
    pairs: ArtifactFingerprint
    predictions: ArtifactFingerprint
    model: ModelIdentity
    hull_source: ArtifactFingerprint | None
    hull_projection_rule: str
    comparisons: tuple[PairComparison, ...]
    summaries: tuple[ObjectSummary, ...]
    skipped: tuple[str, ...]
    measures: Measures
    queue_record: QueueRecord | None
    claim_boundaries: tuple[str, ...] = Field(min_length=1)


# -- small helpers -----------------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def write_mask_png(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").save(path)


def read_mask_png(path: Path) -> np.ndarray:
    return mask_cache.decode_mask_png(path)


def write_jpeg(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(path, format="JPEG", quality=JPEG_QUALITY)


def read_video_frame(video: Path, index: int) -> np.ndarray:
    """Decode frame `index` of `video` as RGB uint8 (H, W, 3)."""
    import cv2

    capture = cv2.VideoCapture(str(video))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise ValueError(f"could not decode frame {index} of {video}")
    return np.ascontiguousarray(frame[:, :, ::-1])


def iou(a: np.ndarray, b: np.ndarray) -> float | None:
    """Intersection over union of two boolean masks; None when both are empty."""
    return mask_iou(a, b, empty_union=None)


def keyframes(step: int = KEYFRAME_STEP, count: int = KEYFRAME_COUNT) -> tuple[int, ...]:
    return tuple(range(0, step * count, step))


def mapped_analysis_frame(
    source_rule: Assembly101ClockRule, target_rule: Assembly101ClockRule, source_frame: int
) -> tuple[int, int, int, int]:
    """Target analysis frame showing the instant of `source_frame`, half frames rounded down.

    Returns `(target_frame, source_pose_frame, target_pose_frame, residual_pose_frames)` where
    the residual is `target_pose_frame - source_pose_frame` (negative: target is earlier).
    """
    source_pose = source_rule.pose_frame(source_frame)
    target_frame = int(
        np.floor(
            (source_pose - target_rule.proxy_start_raw_frame - target_rule.pose_offset_frames)
            / target_rule.raw_frames_per_proxy_frame
        )
    )
    if target_frame < 0:
        raise ValueError(f"source frame {source_frame} precedes the target proxy")
    target_pose = target_rule.pose_frame(target_frame)
    return target_frame, source_pose, target_pose, target_pose - source_pose


def load_observations(run_directory: Path) -> dict[int, FrameObservations]:
    return observations_by_frame(run_directory / "observations.jsonl")


def mask_uri_for(observation: FrameObservations | None, label: str) -> str | None:
    if observation is None:
        return None
    for item in observation.objects:
        if item.label == label and item.mask is not None:
            return item.mask.uri
    return None


def hand_mask_uri_for(observation: FrameObservations | None, hand: str) -> str | None:
    """Mask URI of a hand: an object labelled for that side, or a hand row carrying a mask.

    Neither current run has one (`PerFrameHand` has no mask field and the four-part runs
    label only parts), so this mostly returns None and the ego->exo direction is skipped.
    """
    if observation is None:
        return None
    side = hand.split("_")[0]
    labels = {hand, f"hand_{side}", f"{side}-hand", f"{side} hand"}
    for item in observation.objects:
        if item.label.lower() in labels and item.mask is not None:
            return item.mask.uri
    for item in observation.hands:
        mask = getattr(item, "mask", None)
        if mask is not None and str(item.side).lower().endswith(side):
            return mask.uri
    return None


# -- prepare -----------------------------------------------------------------------------------


def driver_argv(
    *,
    repository_root: Path,
    run_directory: Path,
    mode: str,
    device: str = "cuda",
    cpu_smoke: bool = False,
) -> list[str]:
    argv = [
        str((repository_root / DRIVER_SCRIPT).resolve()),
        "--run-dir",
        str(run_directory.resolve()),
        "--lm-eec-root",
        str(LM_EEC_ROOT),
        "--config",
        LM_EEC_CONFIG,
        "--mode",
        mode,
        "--device",
        device,
    ]
    if cpu_smoke:
        argv.append("--cpu-smoke")
    return argv


def queue_job(*, argv: Sequence[str], timeout_s: int = QUEUE_TIMEOUT_S) -> dict[str, Any]:
    return {
        "jobs": [
            {
                "name": QUEUE_JOB_NAME,
                "argv": list(argv),
                # The predictor resolves its Hydra config module relative to the `sam2`
                # package, so the job runs from the LM-EEC checkout.
                "cwd": str(LM_EEC_ROOT),
                "timeout_s": timeout_s,
                "env": {"CUDA_VISIBLE_DEVICES": "0", "PYTHONUNBUFFERED": "1"},
                "interpreter": [str(LM_EEC_INTERPRETER)],
            }
        ]
    }


def _view_spec(
    rig: CameraRig,
    *,
    camera: str,
    view_id: str,
    proxy: Path,
    proxy_size_wh: tuple[int, int],
    frames_dir: Path,
    root: Path,
) -> ViewSpec:
    raw_w, raw_h = rig.image_size(camera)
    return ViewSpec(
        view_id=view_id,
        camera=camera,
        proxy=fingerprint(proxy, root),
        proxy_size_wh=proxy_size_wh,
        raw_size_wh=(int(raw_w), int(raw_h)),
        pose_offset_frames=rig.clock_rule(camera).pose_offset_frames,
        frames_dir=relative_uri(frames_dir, root),
    )


def _lm_eec_head() -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(LM_EEC_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def prepare(
    repository_root: Path,
    *,
    run_directory: Path = RUN_ROOT,
    exo_query_run: Path = EXO_QUERY_RUN,
    ego_reference_run: Path = EGO_REFERENCE_RUN,
    hull_run: Path | None = HULL_RUN,
    exo_proxy: Path | None = None,
    ego_proxy: Path | None = None,
    frames: Sequence[int] | None = None,
    mode: str = "sequence",
    frame_reader: FrameReader = read_video_frame,
    rig: CameraRig | None = None,
    queue_job_path: Path = QUEUE_JOB_FILE,
    checkpoints: Mapping[str, Path] = LM_EEC_CHECKPOINTS,
    hash_checkpoints: bool = True,
) -> PairsManifest:
    root = repository_root.resolve()
    run_directory = (root / run_directory).resolve()
    run_directory.mkdir(parents=True, exist_ok=True)
    rig = rig or CameraRig.load(root, views=(EXO_CAMERA, EGO_CAMERA))
    exo_rule = rig.clock_rule(EXO_CAMERA)
    ego_rule = rig.clock_rule(EGO_CAMERA)
    exo_proxy = (root / (exo_proxy or proxy_path(EXO_CAMERA))).resolve()
    ego_proxy = (root / (ego_proxy or proxy_path(EGO_CAMERA))).resolve()
    exo_query_dir = (root / exo_query_run).resolve()
    ego_reference_dir = (root / ego_reference_run).resolve()
    hull_dir = (root / hull_run).resolve() if hull_run is not None else None

    exo_observations = load_observations(exo_query_dir)
    ego_observations = load_observations(ego_reference_dir)
    exo_cache = mask_cache.cache_for(exo_query_dir)
    ego_cache = mask_cache.cache_for(ego_reference_dir)
    hull_voxels = None
    if hull_dir is not None and (hull_dir / "hull_voxels_1fps.npz").is_file():
        hull_voxels = np.load(hull_dir / "hull_voxels_1fps.npz")

    frames_root = run_directory / "frames"
    exo_frames_dir = frames_root / EXO_VIEW_ID
    ego_frames_dir = frames_root / EGO_VIEW_ID
    query_root = run_directory / "query_masks"
    reference_root = run_directory / "reference_masks"

    pairs: list[KeyframePair] = []
    any_hand_query = False
    sizes_wh: dict[str, tuple[int, int]] = {}
    for exo_frame in frames if frames is not None else keyframes():
        ego_frame, exo_pose, ego_pose, residual = mapped_analysis_frame(
            exo_rule, ego_rule, exo_frame
        )
        key = f"{exo_frame:06d}"
        exo_rgb = frame_reader(exo_proxy, exo_frame)
        ego_rgb = frame_reader(ego_proxy, ego_frame)
        for view_id, rgb in ((EXO_VIEW_ID, exo_rgb), (EGO_VIEW_ID, ego_rgb)):
            size = (int(rgb.shape[1]), int(rgb.shape[0]))
            if sizes_wh.setdefault(view_id, size) != size:
                raise ValueError(f"{view_id} frame size changed: {sizes_wh[view_id]} vs {size}")
        exo_jpeg = exo_frames_dir / f"{key}.jpg"
        ego_jpeg = ego_frames_dir / f"{key}.jpg"
        write_jpeg(exo_jpeg, exo_rgb)
        write_jpeg(ego_jpeg, ego_rgb)

        exo_observation = exo_observations.get(exo_frame)
        ego_observation = ego_observations.get(ego_frame)
        exo_query: dict[str, str | None] = {}
        ego_reference: dict[str, str | None] = {}
        ego_query: dict[str, str | None] = {}
        hull_available: dict[str, bool] = {}
        for part in PARTS:
            uri = mask_uri_for(exo_observation, part)
            if uri is None:
                exo_query[part] = None
            else:
                mask = exo_cache.mask(uri)
                if mask.shape != exo_rgb.shape[:2]:
                    raise ValueError(
                        f"exo query mask {uri} is {mask.shape}, frame is {exo_rgb.shape[:2]}"
                    )
                target = query_root / EXO_VIEW_ID / part / f"{key}.png"
                write_mask_png(target, mask)
                exo_query[part] = relative_uri(target, run_directory)
            uri = mask_uri_for(ego_observation, part)
            if uri is None:
                ego_reference[part] = None
            else:
                mask = ego_cache.mask(uri)
                if mask.shape != ego_rgb.shape[:2]:
                    raise ValueError(
                        f"ego reference mask {uri} is {mask.shape}, frame is {ego_rgb.shape[:2]}"
                    )
                target = reference_root / EGO_VIEW_ID / part / f"{key}.png"
                write_mask_png(target, mask)
                ego_reference[part] = relative_uri(target, run_directory)
            hull_available[part] = bool(
                hull_voxels is not None
                and f"{part}/{key}" in hull_voxels.files
                and len(hull_voxels[f"{part}/{key}"]) > 0
            )
        for hand in HANDS:
            uri = hand_mask_uri_for(ego_observation, hand)
            if uri is None:
                ego_query[hand] = None
                continue
            mask = ego_cache.mask(uri)
            target = query_root / EGO_VIEW_ID / hand / f"{key}.png"
            write_mask_png(target, mask)
            ego_query[hand] = relative_uri(target, run_directory)
            any_hand_query = True
        pairs.append(
            KeyframePair(
                key=key,
                exo_analysis_frame=exo_frame,
                ego_analysis_frame=ego_frame,
                exo_pose_frame=exo_pose,
                ego_pose_frame=ego_pose,
                residual_pose_frames=residual,
                exo_frame_uri=relative_uri(exo_jpeg, run_directory),
                ego_frame_uri=relative_uri(ego_jpeg, run_directory),
                exo_query_masks=exo_query,
                ego_reference_masks=ego_reference,
                ego_query_masks=ego_query,
                hull_voxels_available=hull_available,
            )
        )

    checkpoint_hashes = {
        direction: (_sha256(path) if hash_checkpoints and path.is_file() else None)
        for direction, path in checkpoints.items()
    }
    directions = (
        DirectionSpec(
            direction="exo_to_ego",
            source_view=EXO_VIEW_ID,
            target_view=EGO_VIEW_ID,
            source_frames_dir=relative_uri(exo_frames_dir, run_directory),
            target_frames_dir=relative_uri(ego_frames_dir, run_directory),
            query_mask_root=relative_uri(query_root / EXO_VIEW_ID, run_directory),
            objects=PARTS,
            checkpoint=str(checkpoints["exo_to_ego"]),
            checkpoint_sha256=checkpoint_hashes["exo_to_ego"],
            skipped_reason=None,
        ),
        DirectionSpec(
            direction="ego_to_exo",
            source_view=EGO_VIEW_ID,
            target_view=EXO_VIEW_ID,
            source_frames_dir=relative_uri(ego_frames_dir, run_directory),
            target_frames_dir=relative_uri(exo_frames_dir, run_directory),
            query_mask_root=(query_root / EGO_VIEW_ID).relative_to(run_directory).as_posix(),
            objects=HANDS,
            checkpoint=str(checkpoints["ego_to_exo"]),
            checkpoint_sha256=checkpoint_hashes["ego_to_exo"],
            skipped_reason=(
                None
                if any_hand_query
                else (
                    f"no ego hand masks: the ego reference run {ego_reference_run.as_posix()} "
                    "carries no hand masks on any keyframe, so ego->exo has no query"
                )
            ),
        ),
    )

    argv = driver_argv(repository_root=root, run_directory=run_directory, mode=mode)
    queue_job_path = (root / queue_job_path).resolve()
    queue_job_path.parent.mkdir(parents=True, exist_ok=True)
    queue_job_path.write_text(json.dumps(queue_job(argv=argv), indent=1) + "\n", encoding="utf-8")

    hull_manifest = hull_dir / "manifest.json" if hull_dir is not None else None
    manifest = PairsManifest(
        run_id=run_directory.name,
        prepared_at_utc=_utc_now(),
        recording_id=RECORDING_ID,
        exo=_view_spec(
            rig,
            camera=EXO_CAMERA,
            view_id=EXO_VIEW_ID,
            proxy=exo_proxy,
            proxy_size_wh=sizes_wh[EXO_VIEW_ID],
            frames_dir=exo_frames_dir,
            root=run_directory,
        ),
        ego=_view_spec(
            rig,
            camera=EGO_CAMERA,
            view_id=EGO_VIEW_ID,
            proxy=ego_proxy,
            proxy_size_wh=sizes_wh[EGO_VIEW_ID],
            frames_dir=ego_frames_dir,
            root=run_directory,
        ),
        frame_mapping_rule=(
            f"ego_frame = floor((pose_frame({EXO_CAMERA}, p) - {ego_rule.proxy_start_raw_frame} - "
            f"{ego_rule.pose_offset_frames}) / {ego_rule.raw_frames_per_proxy_frame}) with "
            f"pose_frame({EXO_CAMERA}, p) = "
            f"{exo_rule.proxy_start_raw_frame} + {exo_rule.raw_frames_per_proxy_frame} p + "
            f"{exo_rule.pose_offset_frames}; the half frame is rounded down, so the ego frame "
            "is one 60 fps pose frame (16.7 ms) before the exo frame"
        ),
        keyframe_step=KEYFRAME_STEP,
        keyframes=tuple(pairs),
        parts=PARTS,
        hands=HANDS,
        directions=directions,
        sources={
            "exo_query_run_manifest": fingerprint(exo_query_dir / "manifest.json", root),
            "ego_reference_run_manifest": fingerprint(ego_reference_dir / "manifest.json", root),
            "hull_manifest": (
                fingerprint(hull_manifest, root)
                if hull_manifest is not None and hull_manifest.is_file()
                else None
            ),
        },
        clock_rules=fingerprint(root / "configs/assembly101/clock_rules.json", root),
        model=ModelIdentity(
            repository=LM_EEC_REPOSITORY,
            pinned_commit=LM_EEC_PINNED_COMMIT,
            checkout_head=_lm_eec_head(),
            config=LM_EEC_CONFIG,
            image_size=LM_EEC_IMAGE_SIZE,
            interpreter=str(LM_EEC_INTERPRETER),
            driver=fingerprint(root / DRIVER_SCRIPT, root),
            install_script=fingerprint(root / "scripts/install_lm_eec.sh", root),
            mode=mode,  # type: ignore[arg-type]
        ),
        queue_job=relative_uri(queue_job_path, root),
        queue_job_name=QUEUE_JOB_NAME,
        queue_timeout_s=QUEUE_TIMEOUT_S,
        claim_boundaries=CLAIM_BOUNDARIES,
    )
    (run_directory / "pairs.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def load_pairs(run_directory: Path) -> PairsManifest:
    return PairsManifest.model_validate_json(
        (run_directory / "pairs.json").read_text(encoding="utf-8")
    )


def load_predictions(run_directory: Path) -> PredictionsFile:
    return PredictionsFile.model_validate_json(
        (run_directory / "predictions.json").read_text(encoding="utf-8")
    )


# -- hull projection ---------------------------------------------------------------------------


def voxel_corners_mm(
    indices: np.ndarray, grid_origin_mm: Sequence[float], voxel_size_mm: float
) -> np.ndarray:
    """(N, 3) voxel indices -> (N, 8, 3) world-mm corners."""
    centres = np.asarray(grid_origin_mm, dtype=np.float64) + (
        np.asarray(indices, dtype=np.float64) + 0.5
    ) * float(voxel_size_mm)
    half = float(voxel_size_mm) / 2.0
    offsets = np.array(
        [[sx, sy, sz] for sx in (-half, half) for sy in (-half, half) for sz in (-half, half)]
    )
    return centres[:, None, :] + offsets[None, :, :]


def rasterize_boxes(corner_px: np.ndarray, image_wh: tuple[int, int]) -> np.ndarray:
    """Fill the axis-aligned pixel box of every (8, 2) corner set into one boolean image."""
    width, height = image_wh
    mask = np.zeros((height, width), dtype=bool)
    if corner_px.size == 0:
        return mask
    finite = np.all(np.isfinite(corner_px), axis=(1, 2))
    corner_px = corner_px[finite]
    if corner_px.size == 0:
        return mask
    x0 = np.clip(np.floor(corner_px[:, :, 0].min(axis=1)), 0, width).astype(int)
    x1 = np.clip(np.ceil(corner_px[:, :, 0].max(axis=1)), 0, width).astype(int)
    y0 = np.clip(np.floor(corner_px[:, :, 1].min(axis=1)), 0, height).astype(int)
    y1 = np.clip(np.ceil(corner_px[:, :, 1].max(axis=1)), 0, height).astype(int)
    for a, b, c, d in zip(x0, x1, y0, y1, strict=True):
        if b > a and d > c:
            mask[c:d, a:b] = True
    return mask


def project_hull_to_view(
    rig: CameraRig,
    *,
    camera: str,
    pose_frame: int,
    voxel_indices: np.ndarray,
    grid_origin_mm: Sequence[float],
    voxel_size_mm: float,
    image_wh: tuple[int, int],
) -> np.ndarray:
    """Project every hull voxel (as its eight corners) into `camera` at proxy resolution."""
    corners = voxel_corners_mm(voxel_indices, grid_origin_mm, voxel_size_mm)
    flat = corners.reshape(-1, 3)
    depth = rig.depth(camera, flat, pose_frame=pose_frame).reshape(-1, 8)
    pixels_raw = rig.project(camera, flat, pose_frame=pose_frame).reshape(-1, 8, 2)
    raw_w, raw_h = rig.image_size(camera)
    scale = np.array([image_wh[0] / raw_w, image_wh[1] / raw_h], dtype=np.float64)
    pixels = pixels_raw * scale
    in_front = np.all(depth > 0, axis=1)
    return rasterize_boxes(pixels[in_front], image_wh)


# -- evaluate ----------------------------------------------------------------------------------


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


def summarize(values: Iterable[float | None]) -> IoUSummary:
    present = np.array([v for v in values if v is not None], dtype=np.float64)
    if present.size == 0:
        return IoUSummary(count=0, median=None, mean=None, min=None, max=None, at_or_above_0_5=0)
    return IoUSummary(
        count=int(present.size),
        median=float(np.median(present)),
        mean=float(present.mean()),
        min=float(present.min()),
        max=float(present.max()),
        at_or_above_0_5=int(np.count_nonzero(present >= 0.5)),
    )


def evaluate(
    repository_root: Path,
    *,
    run_directory: Path = RUN_ROOT,
    hull_run: Path | None = HULL_RUN,
    rig: CameraRig | None = None,
    queue_log: Path = QUEUE_LOG,
) -> EvaluationManifest:
    root = repository_root.resolve()
    run_directory = (root / run_directory).resolve()
    pairs = load_pairs(run_directory)
    predictions = load_predictions(run_directory)
    pairs_by_key = {pair.key: pair for pair in pairs.keyframes}
    hull_dir = (root / hull_run).resolve() if hull_run is not None else None
    hull_voxels = None
    hull_meta: dict[str, Any] | None = None
    hull_fingerprint: ArtifactFingerprint | None = None
    if hull_dir is not None and (hull_dir / "hull_voxels_1fps.npz").is_file():
        hull_voxels = np.load(hull_dir / "hull_voxels_1fps.npz")
        hull_meta = json.loads((hull_dir / "manifest.json").read_text(encoding="utf-8"))
        hull_fingerprint = fingerprint(hull_dir / "manifest.json", root)
        rig = rig or CameraRig.load(root, views=(EXO_CAMERA, EGO_CAMERA))
    hull_root = run_directory / "hull_projection"
    view_by_id = {pairs.exo.view_id: pairs.exo, pairs.ego.view_id: pairs.ego}

    comparisons: list[PairComparison] = []
    for prediction in predictions.predictions:
        pair = pairs_by_key[prediction.key]
        direction = next(d for d in pairs.directions if d.direction == prediction.direction)
        target_view = view_by_id[direction.target_view]
        predicted = read_mask_png(run_directory / prediction.mask_uri)
        notes: list[str] = []

        reference_uri: str | None = None
        reference_mask: np.ndarray | None = None
        if prediction.direction == "exo_to_ego":
            reference_uri = pair.ego_reference_masks.get(prediction.object)
        else:
            reference_uri = pair.exo_query_masks.get(prediction.object)
        if reference_uri is not None:
            reference_mask = read_mask_png(run_directory / reference_uri)
        else:
            notes.append("no reference mask on the target view for this object at this keyframe")

        hull_uri: str | None = None
        hull_mask: np.ndarray | None = None
        if (
            hull_voxels is not None
            and hull_meta is not None
            and rig is not None
            and pair.hull_voxels_available.get(prediction.object, False)
        ):
            pose_frame = (
                pair.ego_pose_frame if target_view.camera == EGO_CAMERA else pair.exo_pose_frame
            )
            hull_mask = project_hull_to_view(
                rig,
                camera=target_view.camera,
                pose_frame=pose_frame,
                voxel_indices=hull_voxels[f"{prediction.object}/{prediction.key}"],
                grid_origin_mm=hull_meta["grid_origin_mm"],
                voxel_size_mm=float(hull_meta["voxel_size_mm"]),
                image_wh=target_view.proxy_size_wh,
            )
            hull_path = hull_root / target_view.view_id / prediction.object / f"{pair.key}.png"
            write_mask_png(hull_path, hull_mask)
            hull_uri = relative_uri(hull_path, run_directory)
            if not hull_mask.any():
                notes.append("hull projects outside the target image at this keyframe")
        elif hull_voxels is None:
            notes.append("hull run not present; hull comparison skipped")
        else:
            notes.append("hull has no voxels for this object at this keyframe")

        comparisons.append(
            PairComparison(
                direction=prediction.direction,
                object=prediction.object,
                key=prediction.key,
                exo_analysis_frame=pair.exo_analysis_frame,
                ego_analysis_frame=pair.ego_analysis_frame,
                prediction_uri=prediction.mask_uri,
                prediction_area_px=int(np.count_nonzero(predicted)),
                prediction_empty=not bool(predicted.any()),
                iou_prediction=prediction.iou_prediction,
                object_score=prediction.object_score,
                reference_uri=reference_uri,
                reference_area_px=(
                    int(np.count_nonzero(reference_mask)) if reference_mask is not None else None
                ),
                iou_vs_reference_mask=(
                    iou(predicted, reference_mask) if reference_mask is not None else None
                ),
                hull_projection_uri=hull_uri,
                hull_projection_area_px=(
                    int(np.count_nonzero(hull_mask)) if hull_mask is not None else None
                ),
                iou_vs_hull_projection=iou(predicted, hull_mask) if hull_mask is not None else None,
                iou_reference_vs_hull=(
                    iou(reference_mask, hull_mask)
                    if reference_mask is not None and hull_mask is not None
                    else None
                ),
                notes=tuple(notes),
            )
        )

    summaries: list[ObjectSummary] = []
    for direction in pairs.directions:
        for obj in direction.objects:
            rows = [
                c for c in comparisons if c.direction == direction.direction and c.object == obj
            ]
            if not rows:
                continue
            summaries.append(
                ObjectSummary(
                    direction=direction.direction,
                    object=obj,
                    pairs=len(rows),
                    predicted_nonempty=sum(not r.prediction_empty for r in rows),
                    model_iou_prediction_median=(
                        float(
                            np.median(
                                [r.iou_prediction for r in rows if r.iou_prediction is not None]
                            )
                        )
                        if any(r.iou_prediction is not None for r in rows)
                        else None
                    ),
                    vs_reference_mask=summarize(r.iou_vs_reference_mask for r in rows),
                    vs_hull_projection=summarize(r.iou_vs_hull_projection for r in rows),
                    reference_vs_hull=summarize(r.iou_reference_vs_hull for r in rows),
                )
            )

    queue_record = read_queue_record(root / queue_log, pairs.queue_job_name)
    runtime = predictions.runtime
    measures = Measures(
        coverage_pairs_predicted_nonempty=sum(not c.prediction_empty for c in comparisons),
        coverage_pairs_total=len(comparisons),
        coverage_pairs_with_reference_mask=sum(
            c.iou_vs_reference_mask is not None for c in comparisons
        ),
        coverage_pairs_with_hull_projection=sum(
            c.iou_vs_hull_projection is not None for c in comparisons
        ),
        time_to_first_output_seconds=runtime.time_to_first_output_seconds,
        runtime_seconds=queue_record.duration_s if queue_record is not None else None,
        driver_seconds=runtime.total_seconds,
        gpu_peak_vram_bytes=runtime.gpu_peak_vram_reserved_bytes,
        id_resets=None,
        id_resets_note=(
            "n/a: one object per query, no identities are tracked; an empty prediction is a "
            "dropped slot, not a judged failure"
        ),
        notes=(
            f"runtime_seconds is the queue job_end duration ({pairs.queue_job_name}); "
            "driver_seconds is the driver's own wall time including model load",
            f"device {runtime.device_name or runtime.device}, torch {runtime.torch_version}, "
            f"cuda {runtime.cuda_version}, mode {runtime.mode}"
            + (", CPU SMOKE (not a GPU result)" if runtime.cpu_smoke else ""),
        ),
    )
    manifest = EvaluationManifest(
        run_id=pairs.run_id,
        evaluated_at_utc=_utc_now(),
        pairs=fingerprint(run_directory / "pairs.json", root),
        predictions=fingerprint(run_directory / "predictions.json", root),
        model=pairs.model,
        hull_source=hull_fingerprint,
        hull_projection_rule=(
            "each hull voxel's eight corners are projected through the rig (fitted intrinsics, "
            "dataset extrinsics, per-frame ego pose at the ego frame's pose frame), raw sensor "
            "pixels scaled to the proxy, and the axis-aligned pixel box of each voxel is filled; "
            "voxels with any corner behind the camera are dropped"
        ),
        comparisons=tuple(comparisons),
        summaries=tuple(summaries),
        skipped=tuple(
            dict.fromkeys(
                (
                    *predictions.skipped,
                    *(
                        f"{d.direction}: {d.skipped_reason}"
                        for d in pairs.directions
                        if d.skipped_reason is not None
                    ),
                )
            )
        ),
        measures=measures,
        queue_record=queue_record,
        claim_boundaries=CLAIM_BOUNDARIES,
    )
    (run_directory / "manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def load_evaluation(run_directory: Path) -> EvaluationManifest:
    return EvaluationManifest.model_validate_json(
        (run_directory / "manifest.json").read_text(encoding="utf-8")
    )


def print_evaluation(manifest: EvaluationManifest) -> None:
    m = manifest.measures
    print(
        f"{manifest.run_id}: {m.coverage_pairs_predicted_nonempty}/{m.coverage_pairs_total} "
        f"pairs predicted non-empty; reference masks on {m.coverage_pairs_with_reference_mask}, "
        f"hull projections on {m.coverage_pairs_with_hull_projection}; first output "
        f"{m.time_to_first_output_seconds} s, driver {m.driver_seconds} s, queue "
        f"{m.runtime_seconds} s, peak VRAM {m.gpu_peak_vram_bytes}"
    )
    for s in manifest.summaries:
        print(
            f"  {s.direction} {s.object:10s} pairs {s.pairs:2d} "
            f"non-empty {s.predicted_nonempty:2d} "
            f"IoU vs ref median {s.vs_reference_mask.median} (n={s.vs_reference_mask.count}) "
            f"vs hull median {s.vs_hull_projection.median} (n={s.vs_hull_projection.count}) "
            f"ref vs hull {s.reference_vs_hull.median}"
        )
    for line in manifest.skipped:
        print(f"  skipped: {line}")


# -- rerun -------------------------------------------------------------------------------------


def _rgba_png(mask: np.ndarray, color: tuple[int, int, int]) -> bytes:
    return mask_cache.encode_rgba_mask_png(mask, color)


def build_recording(
    repository_root: Path, *, run_directory: Path = RUN_ROOT, output: Path | None = None
) -> Path:
    import rerun as rr
    import rerun.blueprint as rrb

    from .rerun_logging import init_and_save, log_rgba_mask, time_series_view

    root = repository_root.resolve()
    run_directory = (root / run_directory).resolve()
    pairs = load_pairs(run_directory)
    evaluation = load_evaluation(run_directory)
    output = output or (run_directory / "correspondence.rrd")
    entity = "world/egoexo_correspondence"
    exo_root = f"{entity}/views/{pairs.exo.view_id}"
    ego_root = f"{entity}/views/{pairs.ego.view_id}"
    comparisons = {(c.direction, c.object, c.key): c for c in evaluation.comparisons}

    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial2DView(
                    origin=exo_root,
                    name=f"{pairs.exo.camera} exo: human query masks (part colours)",
                    contents="$origin/**",
                ),
                rrb.Spatial2DView(
                    origin=ego_root,
                    name=(
                        f"{pairs.ego.camera} ego: LM-EEC prediction (magenta), hull projection "
                        "(white), ego SAM3 (part colours)"
                    ),
                    contents="$origin/**",
                ),
            ),
            rrb.Horizontal(
                time_series_view(
                    f"{entity}/diagnostics/iou",
                    "IoU per part: prediction vs ego SAM3 / vs hull / SAM3 vs hull",
                ),
                rrb.TextDocumentView(origin=f"{entity}/metadata/manifest", name="Manifest"),
            ),
            row_shares=[3, 2],
        ),
        collapse_panels=True,
    )
    init_and_save(
        "battle-egoexo-correspondence",
        output,
        recording_id=pairs.run_id,
        default_blueprint=blueprint,
    )
    rr.log(
        f"{entity}/metadata/manifest",
        rr.TextDocument(evaluation.model_dump_json(indent=2), media_type="application/json"),
        static=True,
    )
    for pair in pairs.keyframes:
        rr.set_time("keyframe", sequence=int(pair.key) // pairs.keyframe_step)
        rr.set_time("exo_analysis_frame", sequence=pair.exo_analysis_frame)
        rr.set_time("analysis_time", duration=pair.exo_analysis_frame / ANALYSIS_FPS)
        rr.log(
            f"{exo_root}/frame",
            rr.EncodedImage(
                contents=(run_directory / pair.exo_frame_uri).read_bytes(),
                media_type="image/jpeg",
            ),
        )
        rr.log(
            f"{ego_root}/frame",
            rr.EncodedImage(
                contents=(run_directory / pair.ego_frame_uri).read_bytes(),
                media_type="image/jpeg",
            ),
        )
        for part in pairs.parts:
            query_uri = pair.exo_query_masks.get(part)
            path = f"{exo_root}/query/{part}"
            if query_uri is None:
                rr.log(path, rr.Clear(recursive=False))
            else:
                log_rgba_mask(
                    path,
                    _rgba_png(read_mask_png(run_directory / query_uri), PART_COLORS[part]),
                    opacity=0.5,
                )
            reference_uri = pair.ego_reference_masks.get(part)
            path = f"{ego_root}/ego_sam3/{part}"
            if reference_uri is None:
                rr.log(path, rr.Clear(recursive=False))
            else:
                log_rgba_mask(
                    path,
                    _rgba_png(read_mask_png(run_directory / reference_uri), PART_COLORS[part]),
                    opacity=0.35,
                )
            comparison = comparisons.get(("exo_to_ego", part, pair.key))
            prediction_path = f"{ego_root}/prediction/{part}"
            hull_path = f"{ego_root}/hull_projection/{part}"
            if comparison is None:
                rr.log(prediction_path, rr.Clear(recursive=False))
                rr.log(hull_path, rr.Clear(recursive=False))
                continue
            log_rgba_mask(
                prediction_path,
                _rgba_png(
                    read_mask_png(run_directory / comparison.prediction_uri), PREDICTION_COLOR
                ),
                opacity=0.55,
                draw_order=3.0,
            )
            if comparison.hull_projection_uri is None:
                rr.log(hull_path, rr.Clear(recursive=False))
            else:
                log_rgba_mask(
                    hull_path,
                    _rgba_png(
                        read_mask_png(run_directory / comparison.hull_projection_uri), HULL_COLOR
                    ),
                    opacity=0.3,
                    draw_order=2.0,
                )
            for name, value in (
                ("prediction_vs_ego_sam3", comparison.iou_vs_reference_mask),
                ("prediction_vs_hull", comparison.iou_vs_hull_projection),
                ("ego_sam3_vs_hull", comparison.iou_reference_vs_hull),
            ):
                series = f"{entity}/diagnostics/iou/{part}/{name}"
                if value is None:
                    rr.log(series, rr.Clear(recursive=False))
                else:
                    rr.log(series, rr.Scalars([value]))
        for hand in pairs.hands:
            comparison = comparisons.get(("ego_to_exo", hand, pair.key))
            path = f"{exo_root}/prediction/{hand}"
            if comparison is None:
                rr.log(path, rr.Clear(recursive=False))
                continue
            log_rgba_mask(
                path,
                _rgba_png(
                    read_mask_png(run_directory / comparison.prediction_uri), PREDICTION_COLOR
                ),
                opacity=0.55,
                draw_order=3.0,
            )
    return output


# -- CLI ---------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare", help="write keyframe pairs, masks, pairs.json, queue job")
    p_prepare.add_argument("--run-dir", type=Path, default=RUN_ROOT)
    p_prepare.add_argument("--mode", choices=("sequence", "independent"), default="sequence")
    p_prepare.add_argument("--no-hull", action="store_true", help="ignore the Track 5 hull run")
    p_prepare.add_argument(
        "--skip-checkpoint-hash", action="store_true", help="do not SHA-256 the 1 GB checkpoints"
    )

    p_run = sub.add_parser("run", help="GPU step: launch or print the LM-EEC driver argv")
    p_run.add_argument("--run-dir", type=Path, default=RUN_ROOT)
    p_run.add_argument("--print-only", action="store_true")
    p_run.add_argument(
        "--cpu-smoke",
        action="store_true",
        help="run the driver on CPU with the model's hard-coded cuda calls shimmed (smoke only)",
    )

    p_eval = sub.add_parser("evaluate", help="CPU: IoU vs ego SAM3 and hull projection, manifest")
    p_eval.add_argument("--run-dir", type=Path, default=RUN_ROOT)
    p_eval.add_argument("--no-hull", action="store_true")

    p_rerun = sub.add_parser("rerun", help="CPU: write correspondence.rrd")
    p_rerun.add_argument("--run-dir", type=Path, default=RUN_ROOT)
    p_rerun.add_argument("--output", type=Path, default=None)

    args = parser.parse_args(argv)
    root = args.repository_root.resolve()

    if args.command == "prepare":
        started = time.perf_counter()
        manifest = prepare(
            root,
            run_directory=args.run_dir,
            hull_run=None if args.no_hull else HULL_RUN,
            mode=args.mode,
            hash_checkpoints=not args.skip_checkpoint_hash,
        )
        with_query = sum(
            sum(uri is not None for uri in pair.exo_query_masks.values())
            for pair in manifest.keyframes
        )
        print(
            f"{manifest.run_id}: {len(manifest.keyframes)} keyframe pairs, {with_query} exo query "
            f"masks, queue job {manifest.queue_job}, {time.perf_counter() - started:.1f} s"
        )
        for direction in manifest.directions:
            state = (
                "ready"
                if direction.skipped_reason is None
                else f"skipped: {direction.skipped_reason}"
            )
            print(f"  {direction.direction}: {state}")
        return

    if args.command == "run":
        run_directory = (root / args.run_dir).resolve()
        pairs = load_pairs(run_directory)
        command = [
            pairs.model.interpreter,
            *driver_argv(
                repository_root=root,
                run_directory=run_directory,
                mode=pairs.model.mode,
                device="cpu" if args.cpu_smoke else "cuda",
                cpu_smoke=args.cpu_smoke,
            ),
        ]
        print(" ".join(command))
        if args.print_only:
            return
        if os.environ.get("CUDA_VISIBLE_DEVICES", "unset") == "" and not args.cpu_smoke:
            raise SystemExit(
                "CUDA is hidden (CUDA_VISIBLE_DEVICES=''); the GPU step belongs to the queue. "
                "Use --print-only, --cpu-smoke, or scripts/overnight_queue.py."
            )
        completed = subprocess.run(command, cwd=str(LM_EEC_ROOT), check=False)
        sys.exit(completed.returncode)

    if args.command == "evaluate":
        manifest = evaluate(
            root, run_directory=args.run_dir, hull_run=None if args.no_hull else HULL_RUN
        )
        print_evaluation(manifest)
        return

    if args.command == "rerun":
        output = build_recording(root, run_directory=args.run_dir, output=args.output)
        print(f"{output} ({output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
