"""Run a bounded, provenance-checked WiLoR hand-pose smoke over an approved proxy."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .cli_common import add_output_root, add_repository_root
from .fs_common import relative_uri, run_timestamp
from .observations import rebuild_tracker_observations
from .schemas import (
    AdapterMetadata,
    ArtifactFingerprint,
    CameraRelativePoint3D,
    ClockName,
    FrameObservations,
    G2PreprocessingManifest,
    HandSide,
    MethodState,
    NormalizedBox,
    NormalizedPoint,
    PerFrameHand,
    RunManifest,
    VideoProxy,
    WiLoRHandsRunMetadata,
)
from .video_driver import (
    assemble_run_manifest,
    bounded_video,
    common_metadata_fields,
    export_manifest,
    hands_contact_sheet,
    load_worker_observations,
    prepend_pythonpath,
    run_external_worker,
    verify_inputs,
    worker_measurements,
    worker_status,
)

DEFAULT_CONFIG = Path("configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_g2.json")
DEFAULT_VIEW_ID = "static-c10379"
DEFAULT_SECONDS = 20.0
MAX_SECONDS = 60.0
WILOR_SOURCE = Path("/home/nick/src/WiLoR")
WILOR_PYTHON = Path("/home/nick/.pyenv/versions/wilor/bin/python")
DEFAULT_CHECKPOINT = WILOR_SOURCE / "pretrained_models" / "wilor_final.ckpt"
DEFAULT_CHECKPOINT_CONFIG = WILOR_SOURCE / "pretrained_models" / "model_config.yaml"
DEFAULT_DETECTOR = WILOR_SOURCE / "pretrained_models" / "detector.pt"
CHECKPOINT_SHA256 = "3e97aafc7dd08d883a4cc5a027df61fdb6fda6136dbd1319405413862ada6bb2"
DETECTOR_SHA256 = "5ef3df44e42d2db52d4ffe91f83a22ce9925e2acc9abebf453f2c5d22e380033"
METHOD_NAME = "wilor-hand-pose-static-smoke"
RERUN_NAME = "hands.rrd"


def _side(value: str) -> HandSide:
    normalized = value.casefold()
    if normalized == "left":
        return HandSide.LEFT
    if normalized == "right":
        return HandSide.RIGHT
    return HandSide.UNKNOWN


def _verify_inputs(
    *,
    repository_root: Path,
    config_path: Path,
    checkpoint_path: Path,
    detector_path: Path,
    view_id: str,
    seconds: float,
) -> tuple[G2PreprocessingManifest, VideoProxy, Path, int]:
    return verify_inputs(
        repository_root=repository_root,
        config_path=config_path,
        view_id=view_id,
        seconds=seconds,
        min_seconds=None,
        max_seconds=MAX_SECONDS,
        checkpoints=(
            (checkpoint_path, CHECKPOINT_SHA256, "checkpoint"),
            (detector_path, DETECTOR_SHA256, "detector"),
        ),
    )


def _hand_from_record(hand: Mapping[str, Any]) -> PerFrameHand:
    return PerFrameHand(
        hand_id=hand["hand_id"],
        side=_side(hand["side"]),
        confidence=float(hand["confidence"]),
        landmarks=tuple(NormalizedPoint(x=point["x"], y=point["y"]) for point in hand["landmarks"]),
        box=NormalizedBox(**hand["box"]),
        model_side=_side(hand["model_side"]),
        model_handedness_confidence=float(hand["model_handedness_confidence"]),
        joints_3d_camera_relative=tuple(
            CameraRelativePoint3D(**point) for point in hand.get("joints_3d_camera_relative", ())
        )
        or None,
    )


def _load_observations(path: Path) -> tuple[FrameObservations, ...]:
    return rebuild_tracker_observations(path, hand_builder=_hand_from_record)


def _run_worker(
    *,
    worker_path: Path,
    run_directory: Path,
    proxy_path: Path,
    view_id: str,
    source_offset_seconds: float,
    checkpoint_path: Path,
    checkpoint_config_path: Path,
    detector_path: Path,
    max_frames: int,
    analysis_fps: float,
    save_native_evidence: bool,
) -> dict[str, object]:
    argv = [
        str(worker_path.resolve()),
        "--run-directory",
        str(run_directory.resolve()),
        "--video",
        str(proxy_path.resolve()),
        "--view-id",
        view_id,
        "--source-offset-seconds",
        str(source_offset_seconds),
        "--checkpoint",
        str(checkpoint_path.resolve()),
        "--checkpoint-config",
        str(checkpoint_config_path.resolve()),
        "--detector",
        str(detector_path.resolve()),
        "--max-frames",
        str(max_frames),
        "--analysis-fps",
        str(analysis_fps),
    ]
    if save_native_evidence:
        argv.append("--save-native-evidence")
    return run_external_worker(
        WILOR_PYTHON,
        argv,
        run_directory=run_directory,
        env={"PYTHONPATH": prepend_pythonpath(WILOR_SOURCE)},
        cwd=WILOR_SOURCE,
    )


def _external_source_state() -> tuple[str | None, bool, str | None, tuple[str, ...]]:
    try:
        revision = subprocess.run(
            ["git", "-C", str(WILOR_SOURCE), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        status = subprocess.run(
            ["git", "-C", str(WILOR_SOURCE), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
        diff = subprocess.run(
            ["git", "-C", str(WILOR_SOURCE), "diff", "--binary"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None, False, None, ()
    status_lines = tuple(line for line in status.stdout.splitlines() if line)
    return (
        revision.stdout.strip() or None,
        bool(status_lines),
        hashlib.sha256(diff.stdout.encode()).hexdigest() if status_lines else None,
        status_lines,
    )


def _build_manifest(
    *,
    run_id: str,
    repository_root: Path,
    run_directory: Path,
    config: G2PreprocessingManifest,
    config_path: Path,
    proxy: VideoProxy,
    view_id: str,
    seconds: float,
    requested_frames: int,
    worker_result: Mapping[str, Any],
    observations: tuple[FrameObservations, ...],
    observations_path: Path,
    qa_path: Path,
    checkpoint_path: Path,
    detector_path: Path,
    save_native_evidence: bool,
    external_source_state: tuple[str | None, bool, str | None, tuple[str, ...]],
) -> tuple[RunManifest, MethodState]:
    rerun_path = run_directory / RERUN_NAME
    # WiLoR's worker writes scalar settings only, so they are taken as they are (no
    # JSON flattening) before the source-state fields are added.
    runtime_settings = dict(worker_result.get("runtime_settings", {}))
    runtime_settings["analysis_fps"] = proxy.fps
    external_revision, source_dirty, source_diff_sha256, source_status = external_source_state
    runtime_settings["external_source_dirty"] = source_dirty
    runtime_settings["external_source_diff_sha256"] = source_diff_sha256
    runtime_settings["external_source_status"] = json.dumps(source_status)
    metadata = WiLoRHandsRunMetadata(
        **common_metadata_fields(
            proxy=proxy,
            config_path=config_path,
            repository_root=repository_root,
            requested_frames=requested_frames,
            seconds=seconds,
            observations_path=observations_path,
            rerun_path=rerun_path,
            qa_path=qa_path,
        ),
        checkpoint_fingerprint=ArtifactFingerprint(
            uri=relative_uri(checkpoint_path, WILOR_SOURCE.parent),
            sha256=CHECKPOINT_SHA256,
            source="measured",
        ),
        detector_fingerprint=ArtifactFingerprint(
            uri=relative_uri(detector_path, WILOR_SOURCE.parent),
            sha256=DETECTOR_SHA256,
            source="measured",
        ),
        adapter=AdapterMetadata(
            name="wilor-hand-pose",
            version="0.1.0",
            implementation_basis=(
                "WiLoR frame-wise demo with batch size one and mesh export disabled"
            ),
            external_source_uri=str(WILOR_SOURCE),
            external_revision=external_revision,
        ),
        runtime_settings=runtime_settings,
        measurements=worker_measurements(
            worker_result,
            known_unavailable_measures=(
                "ground-truth hand-pose accuracy",
                "metric 3D reconstruction",
            ),
        ),
        native_evidence_uri=(
            relative_uri(run_directory / "native_evidence", repository_root)
            if save_native_evidence
            else None
        ),
    )
    state, blocker = worker_status(worker_result)
    manifest = assemble_run_manifest(
        run_id=run_id,
        config=config,
        seconds=seconds,
        view_id=view_id,
        repository_root=repository_root,
        observations=observations,
        observations_path=observations_path,
        rerun_path=rerun_path,
        method_name=METHOD_NAME,
        stage="pose",
        export_method_name="rerun-wilor-hands-export",
        export_measured_on="normalized WiLoR observations; bounded input video logged once",
        state=state,
        blocker=blocker,
        wilor_hands=metadata,
    )
    return manifest, state


def run(args: argparse.Namespace) -> Path:
    repository_root = args.repository_root.resolve()
    config_path = (repository_root / args.config).resolve()
    checkpoint_path = args.checkpoint.resolve()
    checkpoint_config_path = args.checkpoint_config.resolve()
    detector_path = args.detector.resolve()
    config, proxy, proxy_path, requested_frames = _verify_inputs(
        repository_root=repository_root,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        detector_path=detector_path,
        view_id=args.view,
        seconds=args.seconds,
    )
    run_id = args.run_id or f"wilor-hands-static-{args.seconds:g}s-{run_timestamp()}"
    run_directory = (repository_root / args.output_root / run_id).resolve()
    run_directory.mkdir(parents=True, exist_ok=False)
    worker_path = Path(__file__).with_name("wilor_worker.py")
    worker_result = _run_worker(
        worker_path=worker_path,
        run_directory=run_directory,
        proxy_path=proxy_path,
        view_id=args.view,
        source_offset_seconds=config.proxy_timing.source_seconds_for_frame(ClockName.ANALYSIS, 0),
        checkpoint_path=checkpoint_path,
        checkpoint_config_path=checkpoint_config_path,
        detector_path=detector_path,
        max_frames=requested_frames,
        analysis_fps=float(proxy.fps),
        save_native_evidence=args.save_native_evidence,
    )
    observations_path, observations = load_worker_observations(
        run_directory, worker_result, requested_frames, loader=_load_observations, exact=False
    )
    video_path = bounded_video(proxy_path, run_directory / "input.mp4", requested_frames)
    qa_path = hands_contact_sheet(
        video_path=video_path,
        observations=observations,
        output_path=run_directory / "contact_sheet.png",
    )
    manifest, state = _build_manifest(
        run_id=run_id,
        repository_root=repository_root,
        run_directory=run_directory,
        config=config,
        config_path=config_path,
        proxy=proxy,
        view_id=args.view,
        seconds=args.seconds,
        requested_frames=requested_frames,
        worker_result=worker_result,
        observations=observations,
        observations_path=observations_path,
        qa_path=qa_path,
        checkpoint_path=checkpoint_path,
        detector_path=detector_path,
        save_native_evidence=args.save_native_evidence,
        external_source_state=_external_source_state(),
    )
    manifest_path = run_directory / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n")
    if state is MethodState.SUCCEEDED:
        export_manifest(manifest, run_directory / RERUN_NAME, video_path=video_path, proxy=proxy)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_repository_root(parser)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--view", default=DEFAULT_VIEW_ID)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--checkpoint-config", type=Path, default=DEFAULT_CHECKPOINT_CONFIG)
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    add_output_root(parser, Path("runs"))
    parser.add_argument("--run-id")
    parser.add_argument("--save-native-evidence", action="store_true")
    args = parser.parse_args()
    print(run(args))


if __name__ == "__main__":
    main()
