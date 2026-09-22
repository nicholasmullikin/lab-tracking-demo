"""Shared steps of the bounded external-worker video drivers.

`dam4sam_video`, `samurai_video`, `grounding_dino_sam2_video`, `boxmot_track`, `wilor_hands`
and `mediapipe_hands` were written as clones of one another (0.73-1.00 similar): verify the
approved proxy against its config, run the method's worker under its own interpreter, cut the
same bounded `input.mp4` with ffmpeg, draw the same three-frame contact sheet, flatten the
worker's `runtime_settings`, and assemble a `RunManifest` with the same two `MethodStatus`
records and the same zero-overlap `ChunkContinuityPolicy`.  `drop_dtw_align`,
`fine_substep_align` and `four_part_segmentation` carried the ffmpeg step as well.  This
module holds those steps once; each method keeps its worker argv, metadata model, notes and
constants in its own driver.

Nothing here changes bytes.  The ffmpeg argv, the drawing calls and their order, the
`runtime_settings` flattening (`json.dumps(..., sort_keys=True)` for non-scalar values, then
`analysis_fps`, then the driver's extra worker keys) and the manifest fields are the copies
verbatim; where two copies differed the difference is a parameter (`verify_inputs`'s seconds
range and checkpoint list, `load_worker_observations`'s `exact`, `run_external_worker`'s
`record_command`, the contact sheet's per-observation drawer).  `tests/test_video_driver.py`
pins the manifest built here against the former inline code, and the pass-2 ledger entry
records the six 10 s GPU smokes whose masks and manifests were compared before and after.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import mask_cache
from .digest_cache import sha256_file
from .exporter import HAND_CONNECTIONS, export_run
from .fs_common import relative_uri
from .schemas import (
    ArtifactFingerprint,
    ChunkContinuityPolicy,
    EncodedAssetInput,
    FrameObservations,
    FrameRange,
    FullDurationCoverage,
    G2PreprocessingManifest,
    MethodState,
    MethodStatus,
    RunManifest,
    RuntimeMeasurements,
    TimeInterval,
    VideoProxy,
    fingerprint,
)

__all__ = [
    "HAND_CONNECTIONS",
    "assemble_run_manifest",
    "bounded_video",
    "common_metadata_fields",
    "contact_sheet",
    "export_manifest",
    "hands_contact_sheet",
    "load_worker_observations",
    "prepend_pythonpath",
    "run_external_worker",
    "tracker_contact_sheet",
    "verify_inputs",
    "worker_measurements",
    "worker_runtime_settings",
    "worker_status",
]

WorkerResult = Mapping[str, Any]
Drawer = Callable[[np.ndarray, FrameObservations, int, int], None]
ScalarSetting = str | int | float | bool | None

# Zero-overlap, single-stream policy every bounded driver declares (the two-argument form
# some drivers used relies on the same `preserve_track_ids` / `carry_context` defaults).
ZERO_OVERLAP_CHUNK_POLICY: dict[str, float | bool] = {
    "overlap_seconds": 0.0,
    "max_allowed_gap_seconds": 0.0,
    "preserve_track_ids": True,
    "carry_context_across_chunks": True,
}


# --------------------------------------------------------------------------- inputs


def verify_inputs(
    *,
    repository_root: Path,
    config_path: Path,
    view_id: str,
    seconds: float,
    min_seconds: float | None,
    max_seconds: float,
    checkpoints: Sequence[tuple[Path, str, str]] = (),
    min_frames: int | None = None,
) -> tuple[G2PreprocessingManifest, VideoProxy, Path, int]:
    """Check the request against the approved G2 config; return (config, proxy, path, frames).

    `min_seconds=None` is the `(0, max]` range of the hands and BoxMOT drivers, a number the
    `[min, max]` range of the SAM2 trackers.  Each `(path, sha256, label)` in `checkpoints`
    must exist and hash to `sha256` (`ValueError("<label> checksum mismatch for <path>")`).
    `min_frames` is the trackers' "at least 30 frames" rule.  The frame count is
    `min(round(seconds * fps), frame_count)`.
    """
    if min_seconds is None:
        if seconds <= 0 or seconds > max_seconds:
            raise ValueError(f"seconds must be in (0, {max_seconds}]")
    elif seconds < min_seconds or seconds > max_seconds:
        raise ValueError(f"seconds must be in [{min_seconds}, {max_seconds}]")
    config = G2PreprocessingManifest.model_validate_json(config_path.read_text())
    proxy = next((item for item in config.proxies if item.view_id == view_id), None)
    if proxy is None:
        raise ValueError(f"view {view_id} not found in config")
    proxy_path = (repository_root / proxy.proxy_uri).resolve()
    if not proxy_path.is_file():
        raise FileNotFoundError(proxy_path)
    if sha256_file(proxy_path) != proxy.checksum_sha256:
        raise ValueError(f"proxy checksum mismatch for {proxy_path}")
    for path, expected_sha256, label in checkpoints:
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != expected_sha256:
            raise ValueError(f"{label} checksum mismatch for {path}")
    requested_frames = min(round(seconds * proxy.fps), proxy.frame_count)
    if min_frames is not None and requested_frames < min_frames:
        raise ValueError(f"requested frame count must be at least {min_frames}")
    return config, proxy, proxy_path, requested_frames


# --------------------------------------------------------------------------- worker


def prepend_pythonpath(root: Path, environment: Mapping[str, str] | None = None) -> str:
    """`root` in front of the current `PYTHONPATH` (or alone when it is unset or empty)."""
    existing = (os.environ if environment is None else environment).get("PYTHONPATH")
    return os.pathsep.join([str(root), existing]) if existing else str(root)


def run_external_worker(
    python: Path | str,
    argv: Sequence[str],
    *,
    run_directory: Path,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    record_command: bool = False,
) -> dict[str, Any]:
    """Run `python argv...` on GPU 0 and return its `worker_result.json` plus the exit code.

    The child sees the caller's environment with `CUDA_VISIBLE_DEVICES=0` and `env` applied
    on top; stdout and stderr go to `worker.stdout.log` / `worker.stderr.log` in the run
    directory and, with `record_command`, the argv to `worker_command.txt`.  A worker that
    exits without writing `worker_result.json` yields the failed-state record the drivers
    have always returned (`state`, `reason`, `frames_processed`, `elapsed_seconds`).
    """
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    if env:
        environment.update(env)
    command = [str(python), *argv]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=environment,
        check=False,
        cwd=cwd,
    )
    (run_directory / "worker.stdout.log").write_text(completed.stdout)
    (run_directory / "worker.stderr.log").write_text(completed.stderr)
    if record_command:
        (run_directory / "worker_command.txt").write_text(" ".join(command) + "\n")
    result_path = run_directory / "worker_result.json"
    if not result_path.is_file():
        return {
            "state": "failed",
            "reason": (
                f"worker exited {completed.returncode} without worker_result.json; "
                "see worker.stdout.log and worker.stderr.log"
            ),
            "frames_processed": 0,
            "elapsed_seconds": 0.0,
        }
    result = json.loads(result_path.read_text())
    result["worker_exit_code"] = completed.returncode
    return result


def load_worker_observations(
    run_directory: Path,
    worker_result: WorkerResult,
    requested_frames: int,
    *,
    loader: Callable[[Path], tuple[FrameObservations, ...]],
    exact: bool,
) -> tuple[Path, tuple[FrameObservations, ...]]:
    """Read the worker's `observations.jsonl`; `exact` demands the requested count, else at least
    that many."""
    observations_path = run_directory / "observations.jsonl"
    if not observations_path.is_file():
        raise RuntimeError(worker_result.get("reason", "worker produced no observations"))
    observations = loader(observations_path)
    short = len(observations) != requested_frames if exact else len(observations) < requested_frames
    if short:
        raise RuntimeError(
            f"worker produced {len(observations)} observations; expected {requested_frames}"
        )
    return observations_path, observations


# --------------------------------------------------------------------------- media


def bounded_video(proxy_path: Path, output_path: Path, frame_count: int) -> Path:
    """Re-encode the first `frame_count` frames of the proxy (libx264 crf 18, no audio)."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(proxy_path),
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
            str(output_path),
        ],
        check=True,
    )
    return output_path


def contact_sheet(
    *,
    video_path: Path,
    observations: tuple[FrameObservations, ...],
    output_path: Path,
    draw: Drawer,
) -> Path:
    """First, middle and last frame stacked vertically, each annotated by `draw` then labelled."""
    selected = sorted({0, len(observations) // 2, len(observations) - 1})
    capture = cv2.VideoCapture(str(video_path))
    panels = []
    try:
        for frame_index in selected:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"could not decode contact-sheet frame {frame_index}")
            height, width = frame.shape[:2]
            draw(frame, observations[frame_index], width, height)
            cv2.putText(
                frame,
                f"analysis frame {frame_index}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            panels.append(frame)
    finally:
        capture.release()
    sheet = cv2.vconcat(panels)
    if not cv2.imwrite(str(output_path), sheet):
        raise RuntimeError(f"could not write contact sheet: {output_path}")
    return output_path


def _draw_objects(
    frame: np.ndarray, observation: FrameObservations, width: int, height: int
) -> None:
    for obj in observation.objects:
        x1 = round(obj.box.x * width)
        y1 = round(obj.box.y * height)
        x2 = round((obj.box.x + obj.box.width) * width)
        y2 = round((obj.box.y + obj.box.height) * height)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 2)
        cv2.putText(
            frame,
            f"{obj.object_id}: {obj.label}",
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 180, 255),
            2,
        )


def _draw_hands(frame: np.ndarray, observation: FrameObservations, width: int, height: int) -> None:
    for hand in observation.hands:
        points = [(round(point.x * width), round(point.y * height)) for point in hand.landmarks]
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (0, 180, 255), 2)
        for point in points:
            cv2.circle(frame, point, 3, (0, 255, 255), -1)
        x1 = round(hand.box.x * width)
        y1 = round(hand.box.y * height)
        x2 = round((hand.box.x + hand.box.width) * width)
        y2 = round((hand.box.y + hand.box.height) * height)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 2)
        cv2.putText(
            frame,
            f"{hand.hand_id}: {hand.side}",
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 180, 255),
            2,
        )


def tracker_contact_sheet(
    *,
    video_path: Path,
    observations: tuple[FrameObservations, ...],
    output_path: Path,
) -> Path:
    """Contact sheet with each object's box and `object_id: label` (the tracker drivers)."""
    return contact_sheet(
        video_path=video_path,
        observations=observations,
        output_path=output_path,
        draw=_draw_objects,
    )


def hands_contact_sheet(
    *,
    video_path: Path,
    observations: tuple[FrameObservations, ...],
    output_path: Path,
) -> Path:
    """Contact sheet with each hand's skeleton, landmarks, box and `hand_id: side`."""
    return contact_sheet(
        video_path=video_path, observations=observations, output_path=output_path, draw=_draw_hands
    )


# --------------------------------------------------------------------------- manifest


def worker_status(worker_result: WorkerResult) -> tuple[MethodState, str | None]:
    """`(SUCCEEDED, None)` for a succeeded worker, else `(FAILED, str(reason))`."""
    if worker_result.get("state") == "succeeded":
        return MethodState.SUCCEEDED, None
    return MethodState.FAILED, str(worker_result.get("reason"))


def worker_measurements(
    worker_result: WorkerResult, *, known_unavailable_measures: tuple[str, ...]
) -> RuntimeMeasurements:
    """The worker's elapsed / first-output / peak-VRAM numbers as `RuntimeMeasurements`."""
    return RuntimeMeasurements(
        elapsed_seconds=float(worker_result.get("elapsed_seconds", 0.0)),
        time_to_first_usable_output_seconds=worker_result.get(
            "time_to_first_usable_output_seconds"
        ),
        gpu_peak_vram_bytes=worker_result.get("gpu_peak_vram_bytes"),
        known_unavailable_measures=known_unavailable_measures,
    )


def worker_runtime_settings(
    worker_result: WorkerResult,
    *,
    analysis_fps: int | float,
    extra_keys: Sequence[str] = (),
) -> dict[str, ScalarSetting]:
    """Flatten the worker's `runtime_settings` for the manifest.

    Scalars and None pass through; anything else becomes `json.dumps(value, sort_keys=True)`.
    `analysis_fps` is then written (replacing the worker's value in place when it wrote one)
    and each key in `extra_keys` is copied from the top level of the worker result.
    """
    runtime_settings: dict[str, ScalarSetting] = {}
    for key, value in dict(worker_result.get("runtime_settings", {})).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            runtime_settings[key] = value
        else:
            runtime_settings[key] = json.dumps(value, sort_keys=True)
    runtime_settings["analysis_fps"] = analysis_fps
    for key in extra_keys:
        runtime_settings[key] = worker_result.get(key)
    return runtime_settings


def common_metadata_fields(
    *,
    proxy: VideoProxy,
    config_path: Path,
    repository_root: Path,
    requested_frames: int,
    seconds: float,
    observations_path: Path,
    rerun_path: Path,
    qa_path: Path,
) -> dict[str, Any]:
    """The eight metadata fields every bounded driver's `*RunMetadata` model shares.

    Frame range and requested seconds, the three approved-input fingerprints (raw source and
    proxy from the config, the config file measured), and the observations / Rerun / QA URIs.
    """
    return {
        "requested_analysis_frame_range": FrameRange(
            start_frame=0, end_frame_exclusive=requested_frames
        ),
        "requested_seconds": seconds,
        "source_fingerprint": ArtifactFingerprint(
            uri=proxy.raw_source.raw_uri,
            sha256=proxy.raw_source.checksum_sha256,
            source="approved_config",
        ),
        "proxy_fingerprint": ArtifactFingerprint(
            uri=proxy.proxy_uri, sha256=proxy.checksum_sha256, source="approved_config"
        ),
        "config_fingerprint": fingerprint(config_path, repository_root),
        "observations_uri": relative_uri(observations_path, repository_root),
        "rerun_artifact_uri": relative_uri(rerun_path, repository_root),
        "qa_artifact_uri": relative_uri(qa_path, repository_root),
    }


def assemble_run_manifest(
    *,
    run_id: str,
    config: G2PreprocessingManifest,
    seconds: float,
    view_id: str,
    repository_root: Path,
    observations: tuple[FrameObservations, ...],
    observations_path: Path,
    rerun_path: Path,
    method_name: str,
    stage: str,
    export_method_name: str,
    export_measured_on: str,
    state: MethodState,
    blocker: str | None = None,
    **metadata: Any,
) -> RunManifest:
    """The bounded drivers' `RunManifest`: clip cut to `seconds`, one covered interval, the
    zero-overlap chunk policy, the method's status (`stage`, `state`, `blocker`) beside its
    Rerun export status (`SUCCEEDED` when the method succeeded, else `NOT_RUN`), and the
    method's metadata under its own `RunManifest` field (`dam4sam_video=...`, `boxmot=...`)."""
    method_statuses = (
        MethodStatus(
            method_name=method_name,
            stage=stage,
            state=state,
            artifact_uri=relative_uri(observations_path, repository_root),
            measured_on=f"{view_id}; approved {seconds:g}-second proxy prefix",
            blocker=blocker,
        ),
        MethodStatus(
            method_name=export_method_name,
            stage="export",
            state=(
                MethodState.SUCCEEDED if state is MethodState.SUCCEEDED else MethodState.NOT_RUN
            ),
            artifact_uri=relative_uri(rerun_path, repository_root),
            measured_on=export_measured_on,
        ),
    )
    return RunManifest(
        run_id=run_id,
        clip=config.clip.model_copy(update={"source_duration_seconds": seconds}),
        coverage=FullDurationCoverage(
            source_duration_seconds=seconds,
            covered_intervals=(TimeInterval(start_seconds=0.0, end_seconds=seconds),),
        ),
        chunk_policy=ChunkContinuityPolicy(**ZERO_OVERLAP_CHUNK_POLICY),
        method_statuses=method_statuses,
        observations=observations,
        **metadata,
    )


def export_manifest(
    manifest: RunManifest,
    rerun_path: Path,
    *,
    video_path: Path,
    proxy: VideoProxy,
    mask_artifact_root: Path | None = None,
) -> None:
    """`export_run` with the bounded video and the proxy as the asset reference.

    With `mask_artifact_root` (the mask trackers) the referenced masks are embedded and the
    logged-colour sidecar is written beside them.
    """
    export_run(
        manifest,
        rerun_path,
        video_path=video_path,
        video_dimensions=(proxy.dimensions.width, proxy.dimensions.height),
        asset_reference=EncodedAssetInput(
            uri=proxy.proxy_uri,
            media_type="video/mp4",
            checksum_sha256=proxy.checksum_sha256,
        ),
        mask_artifact_root=mask_artifact_root,
    )
    if mask_artifact_root is not None:
        mask_cache.write_sidecar(
            mask_artifact_root, mask_cache.logged_colors_by_uri(mask_artifact_root)
        )
