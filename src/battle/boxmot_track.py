"""Run bounded BoxMOT association over independent per-frame YOLO detections."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import cv2

from .exporter import export_run
from .schemas import (
    AdapterMetadata,
    ArtifactFingerprint,
    BoxMOTRunMetadata,
    ChunkContinuityPolicy,
    ClockName,
    EncodedAssetInput,
    FrameObservations,
    FrameRange,
    FullDurationCoverage,
    G2PreprocessingManifest,
    MethodState,
    MethodStatus,
    NormalizedBox,
    PerFrameObject,
    RunManifest,
    RuntimeMeasurements,
    TimeInterval,
)

DEFAULT_CONFIG = Path("configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_g2.json")
DEFAULT_VIEW_ID = "static-c10379"
DEFAULT_SECONDS = 20.0
MAX_SECONDS = 60.0
WILOR_PYTHON = Path("/home/nick/.pyenv/versions/wilor/bin/python")
DEFAULT_DETECTOR = Path("models/yolo/yolov8n.pt")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_uri(path: Path, repository_root: Path) -> str:
    return path.resolve().relative_to(repository_root.resolve()).as_posix()


def _verify_inputs(
    *,
    repository_root: Path,
    config_path: Path,
    detector_path: Path,
    view_id: str,
    seconds: float,
) -> tuple[G2PreprocessingManifest, object, Path, int, str]:
    if seconds <= 0 or seconds > MAX_SECONDS:
        raise ValueError(f"seconds must be in (0, {MAX_SECONDS}]")
    config = G2PreprocessingManifest.model_validate_json(config_path.read_text())
    proxy = next((item for item in config.proxies if item.view_id == view_id), None)
    if proxy is None:
        raise ValueError(f"view {view_id} not found in config")
    proxy_path = (repository_root / proxy.proxy_uri).resolve()
    if not proxy_path.is_file():
        raise FileNotFoundError(proxy_path)
    if sha256_file(proxy_path) != proxy.checksum_sha256:
        raise ValueError(f"proxy checksum mismatch for {proxy_path}")
    if not detector_path.is_file():
        raise FileNotFoundError(detector_path)
    detector_sha256 = sha256_file(detector_path)
    requested_frames = min(round(seconds * proxy.fps), proxy.frame_count)
    return config, proxy, proxy_path, requested_frames, detector_sha256


def _load_observations(path: Path) -> tuple[FrameObservations, ...]:
    observations: list[FrameObservations] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        objects = tuple(
            PerFrameObject(
                object_id=obj["object_id"],
                label=obj["label"],
                confidence=float(obj["confidence"]),
                box=NormalizedBox(**obj["box"]),
            )
            for obj in payload.get("objects", ())
        )
        observations.append(
            FrameObservations(
                view_id=payload["view_id"],
                analysis_frame_index=int(payload["analysis_frame_index"]),
                source_seconds=float(payload["source_seconds"]),
                objects=objects,
            )
        )
    return tuple(observations)


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
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    command = [
        str(WILOR_PYTHON),
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
    completed = subprocess.run(
        command, capture_output=True, text=True, env=environment, check=False
    )
    (run_directory / "worker.stdout.log").write_text(completed.stdout)
    (run_directory / "worker.stderr.log").write_text(completed.stderr)
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


def _bounded_video(proxy_path: Path, output_path: Path, frame_count: int) -> Path:
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


def _contact_sheet(
    *,
    video_path: Path,
    observations: tuple[FrameObservations, ...],
    output_path: Path,
) -> Path:
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
            observation = observations[frame_index]
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
    run_id = (
        args.run_id or f"boxmot-yolo-static-{args.seconds:g}s-{datetime.now(UTC):%Y%m%dt%H%M%Sz}"
    )
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
    observations_path = run_directory / "observations.jsonl"
    if not observations_path.is_file():
        raise RuntimeError(worker_result.get("reason", "worker produced no observations"))
    observations = _load_observations(observations_path)
    if len(observations) < requested_frames:
        raise RuntimeError(
            f"worker produced {len(observations)} observations; expected {requested_frames}"
        )
    video_path = _bounded_video(proxy_path, run_directory / "input.mp4", requested_frames)
    qa_path = _contact_sheet(
        video_path=video_path,
        observations=observations,
        output_path=run_directory / "contact_sheet.png",
    )
    measurements = RuntimeMeasurements(
        elapsed_seconds=float(worker_result.get("elapsed_seconds", 0.0)),
        time_to_first_usable_output_seconds=worker_result.get(
            "time_to_first_usable_output_seconds"
        ),
        gpu_peak_vram_bytes=worker_result.get("gpu_peak_vram_bytes"),
        known_unavailable_measures=("ground-truth association accuracy",),
    )
    runtime_settings: dict[str, str | int | float | bool | None] = {}
    for key, value in dict(worker_result.get("runtime_settings", {})).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            runtime_settings[key] = value
        else:
            runtime_settings[key] = json.dumps(value, sort_keys=True)
    runtime_settings["analysis_fps"] = proxy.fps
    runtime_settings["frames_with_tracks"] = worker_result.get("frames_with_tracks")
    runtime_settings["total_track_observations"] = worker_result.get("total_track_observations")
    metadata = BoxMOTRunMetadata(
        requested_analysis_frame_range=FrameRange(
            start_frame=0, end_frame_exclusive=requested_frames
        ),
        requested_seconds=args.seconds,
        source_fingerprint=ArtifactFingerprint(
            uri=proxy.raw_source.raw_uri,
            sha256=proxy.raw_source.checksum_sha256,
            source="approved_config",
        ),
        proxy_fingerprint=ArtifactFingerprint(
            uri=proxy.proxy_uri, sha256=proxy.checksum_sha256, source="approved_config"
        ),
        config_fingerprint=ArtifactFingerprint(
            uri=relative_uri(config_path, repository_root),
            sha256=sha256_file(config_path),
            source="measured",
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
        runtime_settings=runtime_settings,
        measurements=measurements,
        observations_uri=relative_uri(observations_path, repository_root),
        rerun_artifact_uri=relative_uri(run_directory / "tracks.rrd", repository_root),
        qa_artifact_uri=relative_uri(qa_path, repository_root),
        detector_source=(
            "ultralytics YOLOv8n COCO per-frame detections (class filter "
            f"{args.detector_classes}); no MuggledSAM track IDs"
        ),
    )
    worker_state = (
        MethodState.SUCCEEDED if worker_result.get("state") == "succeeded" else MethodState.FAILED
    )
    method_statuses = (
        MethodStatus(
            method_name="boxmot-yolo-static-smoke",
            stage="objects",
            state=worker_state,
            artifact_uri=relative_uri(observations_path, repository_root),
            measured_on=f"{args.view}; approved {args.seconds:g}-second proxy prefix",
            blocker=(
                str(worker_result.get("reason")) if worker_state is MethodState.FAILED else None
            ),
        ),
        MethodStatus(
            method_name="rerun-boxmot-export",
            stage="export",
            state=(
                MethodState.SUCCEEDED
                if worker_state is MethodState.SUCCEEDED
                else MethodState.NOT_RUN
            ),
            artifact_uri=relative_uri(run_directory / "tracks.rrd", repository_root),
            measured_on="normalized BoxMOT observations; bounded input video logged once",
        ),
    )
    manifest = RunManifest(
        run_id=run_id,
        clip=config.clip.model_copy(update={"source_duration_seconds": args.seconds}),
        coverage=FullDurationCoverage(
            source_duration_seconds=args.seconds,
            covered_intervals=(TimeInterval(start_seconds=0.0, end_seconds=args.seconds),),
        ),
        chunk_policy=ChunkContinuityPolicy(overlap_seconds=0.0, max_allowed_gap_seconds=0.0),
        method_statuses=method_statuses,
        observations=observations,
        boxmot=metadata,
    )
    manifest_path = run_directory / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n")
    if worker_state is MethodState.SUCCEEDED:
        export_run(
            manifest,
            run_directory / "tracks.rrd",
            video_path=video_path,
            video_dimensions=(proxy.dimensions.width, proxy.dimensions.height),
            asset_reference=EncodedAssetInput(
                uri=proxy.proxy_uri,
                media_type="video/mp4",
                checksum_sha256=proxy.checksum_sha256,
            ),
        )
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
