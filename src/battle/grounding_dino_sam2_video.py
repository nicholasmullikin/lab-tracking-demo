"""Run bounded HF Grounding DINO + SAM2 video propagation on the approved static proxy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import cv2

from . import mask_cache
from .digest_cache import sha256_file
from .exporter import export_run
from .fs_common import relative_uri, run_timestamp
from .schemas import (
    AdapterMetadata,
    ArtifactFingerprint,
    ChunkContinuityPolicy,
    ClockName,
    EncodedAssetInput,
    FrameObservations,
    FrameRange,
    FullDurationCoverage,
    G2PreprocessingManifest,
    GroundingDinoSam2VideoRunMetadata,
    MaskReference,
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
DEFAULT_SECONDS = 10.0
MIN_SECONDS = 1.0
MAX_SECONDS = 20.0
GROUNDED_SAM2_PYTHON = Path("/home/nick/.pyenv/versions/grounded_sam2/bin/python")
GROUNDED_SAM2_ROOT = Path("/home/nick/src/Grounded-SAM-2")
GROUNDED_SAM2_REVISION = "b7a9c29f196edff0eb54dbe14588d7ae5e3dde28"
SAM2_CHECKPOINT = GROUNDED_SAM2_ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt"
SAM2_CHECKPOINT_SHA256 = "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
GROUNDING_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
GROUNDING_MODEL_REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
TEXT_PROMPT = "hand."
METHOD_NAME = "transformers_grounding_dino_plus_sam2_video_smoke"


def _normalize_box(
    x1: float, y1: float, x2: float, y2: float, width: int, height: int
) -> NormalizedBox:
    left = min(max(x1 / width, 0.0), 1.0)
    top = min(max(y1 / height, 0.0), 1.0)
    right = min(max(x2 / width, 0.0), 1.0)
    bottom = min(max(y2 / height, 0.0), 1.0)
    box_width = min(max(right - left, 1e-4), 1.0 - left)
    box_height = min(max(bottom - top, 1e-4), 1.0 - top)
    return NormalizedBox(x=left, y=top, width=box_width, height=box_height)


def _verify_inputs(
    *,
    repository_root: Path,
    config_path: Path,
    view_id: str,
    seconds: float,
) -> tuple[G2PreprocessingManifest, object, Path, int]:
    if seconds < MIN_SECONDS or seconds > MAX_SECONDS:
        raise ValueError(f"seconds must be in [{MIN_SECONDS}, {MAX_SECONDS}]")
    config = G2PreprocessingManifest.model_validate_json(config_path.read_text())
    proxy = next((item for item in config.proxies if item.view_id == view_id), None)
    if proxy is None:
        raise ValueError(f"view {view_id} not found in config")
    proxy_path = (repository_root / proxy.proxy_uri).resolve()
    if not proxy_path.is_file():
        raise FileNotFoundError(proxy_path)
    if sha256_file(proxy_path) != proxy.checksum_sha256:
        raise ValueError(f"proxy checksum mismatch for {proxy_path}")
    if not SAM2_CHECKPOINT.is_file():
        raise FileNotFoundError(SAM2_CHECKPOINT)
    if sha256_file(SAM2_CHECKPOINT) != SAM2_CHECKPOINT_SHA256:
        raise ValueError(f"SAM2 checkpoint checksum mismatch for {SAM2_CHECKPOINT}")
    requested_frames = min(round(seconds * proxy.fps), proxy.frame_count)
    if requested_frames < 30:
        raise ValueError("requested frame count must be at least 30")
    return config, proxy, proxy_path, requested_frames


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
                mask=MaskReference(**obj["mask"]) if obj.get("mask") else None,
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
    max_frames: int,
    analysis_fps: float,
) -> dict[str, object]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    command = [
        str(GROUNDED_SAM2_PYTHON),
        str(worker_path.resolve()),
        "--run-directory",
        str(run_directory.resolve()),
        "--video",
        str(proxy_path.resolve()),
        "--view-id",
        view_id,
        "--source-offset-seconds",
        str(source_offset_seconds),
        "--max-frames",
        str(max_frames),
        "--analysis-fps",
        str(analysis_fps),
        "--sam2-checkpoint",
        str(SAM2_CHECKPOINT.resolve()),
        "--grounded-sam2-root",
        str(GROUNDED_SAM2_ROOT.resolve()),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=environment,
        check=False,
        cwd=GROUNDED_SAM2_ROOT,
    )
    (run_directory / "worker.stdout.log").write_text(completed.stdout)
    (run_directory / "worker.stderr.log").write_text(completed.stderr)
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
    config, proxy, proxy_path, requested_frames = _verify_inputs(
        repository_root=repository_root,
        config_path=config_path,
        view_id=args.view,
        seconds=args.seconds,
    )
    run_id = args.run_id or f"{METHOD_NAME}-{args.seconds:g}s-{run_timestamp()}"
    run_directory = (repository_root / args.output_root / run_id).resolve()
    run_directory.mkdir(parents=True, exist_ok=False)
    worker_path = Path(__file__).with_name("grounding_dino_sam2_video_worker.py")
    worker_result = _run_worker(
        worker_path=worker_path,
        run_directory=run_directory,
        proxy_path=proxy_path,
        view_id=args.view,
        source_offset_seconds=config.proxy_timing.source_seconds_for_frame(ClockName.ANALYSIS, 0),
        max_frames=requested_frames,
        analysis_fps=float(proxy.fps),
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
        known_unavailable_measures=("ground-truth mask quality",),
    )
    runtime_settings: dict[str, str | int | float | bool | None] = {}
    for key, value in dict(worker_result.get("runtime_settings", {})).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            runtime_settings[key] = value
        else:
            runtime_settings[key] = json.dumps(value, sort_keys=True)
    runtime_settings["analysis_fps"] = proxy.fps
    runtime_settings["frames_with_masks"] = worker_result.get("frames_with_masks")
    runtime_settings["unique_mask_hashes"] = worker_result.get("unique_mask_hashes")
    metadata = GroundingDinoSam2VideoRunMetadata(
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
        grounding_model_fingerprint=ArtifactFingerprint(
            uri=f"hf://{GROUNDING_MODEL_ID}@{GROUNDING_MODEL_REVISION}",
            sha256=hashlib.sha256(
                f"{GROUNDING_MODEL_ID}@{GROUNDING_MODEL_REVISION}".encode()
            ).hexdigest(),
            source="measured",
        ),
        sam2_checkpoint_fingerprint=ArtifactFingerprint(
            uri=str(SAM2_CHECKPOINT),
            sha256=SAM2_CHECKPOINT_SHA256,
            source="measured",
        ),
        adapter=AdapterMetadata(
            name=METHOD_NAME,
            version="0.1.0",
            implementation_basis=(
                "Hugging Face Grounding DINO Tiny frame-0 detection plus SAM2.1 tiny "
                "video predictor propagation"
            ),
            external_source_uri=str(GROUNDED_SAM2_ROOT),
            external_revision=GROUNDED_SAM2_REVISION,
        ),
        runtime_settings=runtime_settings,
        measurements=measurements,
        observations_uri=relative_uri(observations_path, repository_root),
        native_masks_uri=relative_uri(run_directory / "native" / "masks", repository_root),
        rerun_artifact_uri=relative_uri(run_directory / "propagation.rrd", repository_root),
        qa_artifact_uri=relative_uri(qa_path, repository_root),
        text_prompt=TEXT_PROMPT,
        sam2_model_config=SAM2_CONFIG,
        initialization_note=(
            "Frame 0 uses the pinned HF Grounding DINO fallback after the local CUDA "
            "extension build failed; subsequent frames are SAM2 video propagation only."
        ),
    )
    worker_state = (
        MethodState.SUCCEEDED if worker_result.get("state") == "succeeded" else MethodState.FAILED
    )
    method_statuses = (
        MethodStatus(
            method_name=METHOD_NAME,
            stage="objects",
            state=worker_state,
            artifact_uri=relative_uri(observations_path, repository_root),
            measured_on=f"{args.view}; approved {args.seconds:g}-second proxy prefix",
            blocker=(
                str(worker_result.get("reason")) if worker_state is MethodState.FAILED else None
            ),
        ),
        MethodStatus(
            method_name="rerun-grounding-dino-sam2-export",
            stage="export",
            state=(
                MethodState.SUCCEEDED
                if worker_state is MethodState.SUCCEEDED
                else MethodState.NOT_RUN
            ),
            artifact_uri=relative_uri(run_directory / "propagation.rrd", repository_root),
            measured_on="normalized mask observations; bounded input video logged once",
        ),
    )
    manifest = RunManifest(
        run_id=run_id,
        clip=config.clip.model_copy(update={"source_duration_seconds": args.seconds}),
        coverage=FullDurationCoverage(
            source_duration_seconds=args.seconds,
            covered_intervals=(TimeInterval(start_seconds=0.0, end_seconds=args.seconds),),
        ),
        chunk_policy=ChunkContinuityPolicy(
            overlap_seconds=0.0,
            max_allowed_gap_seconds=0.0,
            preserve_track_ids=True,
            carry_context_across_chunks=True,
        ),
        method_statuses=method_statuses,
        observations=observations,
        grounding_dino_sam2_video=metadata,
    )
    manifest_path = run_directory / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n")
    if worker_state is MethodState.SUCCEEDED:
        export_run(
            manifest,
            run_directory / "propagation.rrd",
            video_path=video_path,
            video_dimensions=(proxy.dimensions.width, proxy.dimensions.height),
            asset_reference=EncodedAssetInput(
                uri=proxy.proxy_uri,
                media_type="video/mp4",
                checksum_sha256=proxy.checksum_sha256,
            ),
            mask_artifact_root=run_directory,
        )
        mask_cache.write_sidecar(run_directory, mask_cache.logged_colors_by_uri(run_directory))
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--view", default=DEFAULT_VIEW_ID)
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--run-id")
    args = parser.parse_args()
    print(run(args))


if __name__ == "__main__":
    main()
