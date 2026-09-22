"""Run bounded BoxMOT association over independent per-frame YOLO detections."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .digest_cache import sha256_file
from .fs_common import relative_uri, run_timestamp
from .observations import rebuild_tracker_observations as _load_observations
from .schemas import (
    AdapterMetadata,
    ArtifactFingerprint,
    BoxMOTRunMetadata,
    ClockName,
    FrameObservations,
    G2PreprocessingManifest,
    MethodState,
    RunManifest,
    VideoProxy,
)
from .video_driver import (
    assemble_run_manifest,
    bounded_video,
    common_metadata_fields,
    export_manifest,
    load_worker_observations,
    run_external_worker,
    tracker_contact_sheet,
    verify_inputs,
    worker_measurements,
    worker_runtime_settings,
    worker_status,
)

DEFAULT_CONFIG = Path("configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_g2.json")
DEFAULT_VIEW_ID = "static-c10379"
DEFAULT_SECONDS = 20.0
MAX_SECONDS = 60.0
WILOR_PYTHON = Path("/home/nick/.pyenv/versions/wilor/bin/python")
DEFAULT_DETECTOR = Path("models/yolo/yolov8n.pt")
METHOD_NAME = "boxmot-yolo-static-smoke"
RERUN_NAME = "tracks.rrd"


def _verify_inputs(
    *,
    repository_root: Path,
    config_path: Path,
    detector_path: Path,
    view_id: str,
    seconds: float,
) -> tuple[G2PreprocessingManifest, VideoProxy, Path, int, str]:
    config, proxy, proxy_path, requested_frames = verify_inputs(
        repository_root=repository_root,
        config_path=config_path,
        view_id=view_id,
        seconds=seconds,
        min_seconds=None,
        max_seconds=MAX_SECONDS,
    )
    if not detector_path.is_file():
        raise FileNotFoundError(detector_path)
    detector_sha256 = sha256_file(detector_path)
    return config, proxy, proxy_path, requested_frames, detector_sha256


def _run_worker(
    *,
    worker_path: Path,
    run_directory: Path,
    proxy_path: Path,
    view_id: str,
    source_offset_seconds: float,
    detector_path: Path,
    max_frames: int,
    analysis_fps: float,
    detector_confidence: float,
    detector_classes: str,
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
        "--detector-weights",
        str(detector_path.resolve()),
        "--max-frames",
        str(max_frames),
        "--analysis-fps",
        str(analysis_fps),
        "--detector-confidence",
        str(detector_confidence),
        "--detector-classes",
        detector_classes,
    ]
    return run_external_worker(WILOR_PYTHON, argv, run_directory=run_directory)


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
    detector_path: Path,
    detector_sha256: str,
    detector_classes: str,
) -> tuple[RunManifest, MethodState]:
    rerun_path = run_directory / RERUN_NAME
    metadata = BoxMOTRunMetadata(
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
        detector_fingerprint=ArtifactFingerprint(
            uri=relative_uri(detector_path, repository_root),
            sha256=detector_sha256,
            source="measured",
        ),
        adapter=AdapterMetadata(
            name="boxmot-botsort",
            version="25.0.0",
            implementation_basis=(
                "Ultralytics YOLO per-frame detections associated by BoxMOT BotSort"
            ),
            external_source_uri="https://github.com/mikel-brostrom/boxmot",
        ),
        runtime_settings=worker_runtime_settings(
            worker_result,
            analysis_fps=proxy.fps,
            extra_keys=("frames_with_tracks", "total_track_observations"),
        ),
        measurements=worker_measurements(
            worker_result, known_unavailable_measures=("ground-truth association accuracy",)
        ),
        detector_source=(
            "ultralytics YOLOv8n COCO per-frame detections (class filter "
            f"{detector_classes}); no MuggledSAM track IDs"
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
        stage="objects",
        export_method_name="rerun-boxmot-export",
        export_measured_on="normalized BoxMOT observations; bounded input video logged once",
        state=state,
        blocker=blocker,
        boxmot=metadata,
    )
    return manifest, state


def run(args: argparse.Namespace) -> Path:
    repository_root = args.repository_root.resolve()
    config_path = (repository_root / args.config).resolve()
    detector_path = (repository_root / args.detector).resolve()
    config, proxy, proxy_path, requested_frames, detector_sha256 = _verify_inputs(
        repository_root=repository_root,
        config_path=config_path,
        detector_path=detector_path,
        view_id=args.view,
        seconds=args.seconds,
    )
    run_id = args.run_id or f"boxmot-yolo-static-{args.seconds:g}s-{run_timestamp()}"
    run_directory = (repository_root / args.output_root / run_id).resolve()
    run_directory.mkdir(parents=True, exist_ok=False)
    worker_path = Path(__file__).with_name("boxmot_worker.py")
    worker_result = _run_worker(
        worker_path=worker_path,
        run_directory=run_directory,
        proxy_path=proxy_path,
        view_id=args.view,
        source_offset_seconds=config.proxy_timing.source_seconds_for_frame(ClockName.ANALYSIS, 0),
        detector_path=detector_path,
        max_frames=requested_frames,
        analysis_fps=float(proxy.fps),
        detector_confidence=args.detector_confidence,
        detector_classes=args.detector_classes,
    )
    observations_path, observations = load_worker_observations(
        run_directory, worker_result, requested_frames, loader=_load_observations, exact=False
    )
    video_path = bounded_video(proxy_path, run_directory / "input.mp4", requested_frames)
    qa_path = tracker_contact_sheet(
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
        detector_path=detector_path,
        detector_sha256=detector_sha256,
        detector_classes=args.detector_classes,
    )
    manifest_path = run_directory / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n")
    if state is MethodState.SUCCEEDED:
        export_manifest(manifest, run_directory / RERUN_NAME, video_path=video_path, proxy=proxy)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--view", default=DEFAULT_VIEW_ID)
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--run-id")
    parser.add_argument("--detector-confidence", type=float, default=0.25)
    parser.add_argument("--detector-classes", default="0")
    args = parser.parse_args()
    print(run(args))


if __name__ == "__main__":
    main()
