"""Run bounded CLIP + Drop-DTW alignment against an Assembly101 GT transcript."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import rerun as rr

from .assembly101_gt_transcript import parse_coarse_transcript
from .cli_common import add_output_root, add_repository_root
from .digest_cache import sha256_file
from .fs_common import relative_uri, run_timestamp
from .rerun_logging import init_and_save
from .schemas import (
    AdapterMetadata,
    ArtifactFingerprint,
    ChunkContinuityPolicy,
    ClockName,
    DropDTWRunMetadata,
    FullDurationCoverage,
    G2PreprocessingManifest,
    MethodState,
    MethodStatus,
    RunManifest,
    RuntimeMeasurements,
    TimeInterval,
)
from .video_driver import bounded_video as _bounded_video

DEFAULT_CONFIG = Path("configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_g2.json")
DEFAULT_LABELS = Path(
    "data/raw/assembly101/nusar-2021_action_both_9033-c02a_9033_user_id_2021-02-04_140532/"
    "annotations/coarse-annotations/coarse_labels/"
    "assembly_nusar-2021_action_both_9033-c02a_9033_user_id_2021-02-04_140532.txt"
)
DEFAULT_VIEW_ID = "static-c10379"
DEFAULT_SECONDS = 20.0
MAX_SECONDS = 60.0
WILOR_PYTHON = Path("/home/nick/.pyenv/versions/wilor/bin/python")
DROP_DTW_SOURCE = Path("/home/nick/src/Drop-DTW")
DROP_DTW_REVISION = "32ce9c82c6a0d717a94f4139b1902ad146923444"
OPENCLIP_REPOSITORY = "timm/vit_base_patch32_clip_224.openai"
OPENCLIP_REVISION = "a6f597a30f7b82c51704746581f9a4e41421e878"
DEFAULT_OPENCLIP_CHECKPOINT = Path(
    "/home/nick/.cache/huggingface/hub/models--timm--vit_base_patch32_clip_224.openai/"
    f"snapshots/{OPENCLIP_REVISION}/open_clip_model.safetensors"
)
DEFAULT_OPENCLIP_CACHE = Path("/home/nick/.cache/huggingface/hub")


def _export_alignment_rrd(
    *,
    alignment: dict[str, object],
    output_path: Path,
    run_id: str,
    seconds: float,
) -> None:
    init_and_save(f"battle-{run_id}", output_path)
    rr.log("manifest/weak_supervision_note", rr.TextLog(str(alignment["weak_supervision_note"])))
    rr.log("alignment/cost", rr.Scalars([float(alignment["alignment_cost"])]))
    for interval in alignment["intervals"]:
        action = str(interval["action"])
        for second in interval["matched_seconds"]:
            rr.set_time("analysis", duration=float(second))
            rr.log(f"intervals/{action}", rr.Scalars([1.0]))
    rr.set_time("analysis", duration=seconds)


def run(args: argparse.Namespace) -> Path:
    repository_root = args.repository_root.resolve()
    config_path = (repository_root / args.config).resolve()
    labels_path = (repository_root / args.labels).resolve()
    checkpoint_path = args.openclip_checkpoint.resolve()
    config = G2PreprocessingManifest.model_validate_json(config_path.read_text())
    proxy = next((item for item in config.proxies if item.view_id == args.view), None)
    if proxy is None:
        raise ValueError(f"view {args.view} not found in config")
    proxy_path = (repository_root / proxy.proxy_uri).resolve()
    if not proxy_path.is_file():
        raise FileNotFoundError(proxy_path)
    if sha256_file(proxy_path) != proxy.checksum_sha256:
        raise ValueError("proxy checksum mismatch")
    if not labels_path.is_file():
        raise FileNotFoundError(labels_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    requested_frames = min(round(args.seconds * proxy.fps), proxy.frame_count)
    source_offset_seconds = config.proxy_timing.source_seconds_for_frame(ClockName.ANALYSIS, 0)
    annotation_start = round(source_offset_seconds * 30)
    annotation_end = round((source_offset_seconds + args.seconds) * 30)
    steps = parse_coarse_transcript(
        labels_path,
        annotation_start_frame=annotation_start,
        annotation_end_frame=annotation_end,
    )
    if not steps:
        raise RuntimeError(
            "no coarse transcript steps overlap the requested interval; "
            "cannot run Drop-DTW weak supervision"
        )

    run_id = args.run_id or f"drop-dtw-static-{args.seconds:g}s-{run_timestamp()}"
    run_directory = (repository_root / args.output_root / run_id).resolve()
    run_directory.mkdir(parents=True, exist_ok=False)
    transcript_path = run_directory / "gt_transcript.json"
    transcript_payload = {
        "source": "assembly101_coarse_annotations",
        "annotation_fps": 30,
        "annotation_start_frame": annotation_start,
        "annotation_end_frame": annotation_end,
        "steps": [
            {
                "action": step.action,
                "annotation_start_frame": step.annotation_start_frame,
                "annotation_end_frame": step.annotation_end_frame,
            }
            for step in steps
        ],
        "weak_supervision_note": (
            "Ordered text is derived from Assembly101 shipped coarse labels, not model output."
        ),
    }
    transcript_path.write_text(json.dumps(transcript_payload, indent=2) + "\n")
    video_path = _bounded_video(proxy_path, run_directory / "input.mp4", requested_frames)

    worker_path = Path(__file__).with_name("drop_dtw_worker.py")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    environment["HF_HUB_OFFLINE"] = "1"
    command = [
        str(WILOR_PYTHON),
        str(worker_path.resolve()),
        "--run-directory",
        str(run_directory.resolve()),
        "--video",
        str(video_path.resolve()),
        "--transcript-json",
        str(transcript_path.resolve()),
        "--sample-fps",
        str(args.sample_fps),
        "--keep-percentile",
        str(args.keep_percentile),
        "--openclip-checkpoint",
        str(checkpoint_path),
        "--openclip-cache-dir",
        str(args.openclip_cache_dir.resolve()),
    ]
    completed = subprocess.run(
        command, capture_output=True, text=True, env=environment, check=False
    )
    (run_directory / "worker.stdout.log").write_text(completed.stdout)
    (run_directory / "worker.stderr.log").write_text(completed.stderr)
    worker_result = json.loads((run_directory / "worker_result.json").read_text())
    alignment_path = run_directory / "alignment.json"
    if worker_result.get("state") != "succeeded" or not alignment_path.is_file():
        raise RuntimeError(worker_result.get("reason", "Drop-DTW worker failed"))
    alignment = json.loads(alignment_path.read_text())
    rrd_path = run_directory / "alignment.rrd"
    _export_alignment_rrd(
        alignment=alignment,
        output_path=rrd_path,
        run_id=run_id,
        seconds=args.seconds,
    )

    metadata = DropDTWRunMetadata(
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
        transcript_fingerprint=ArtifactFingerprint(
            uri=relative_uri(labels_path, repository_root),
            sha256=sha256_file(labels_path),
            source="measured",
        ),
        openclip_checkpoint_fingerprint=ArtifactFingerprint(
            uri=(f"hf://{OPENCLIP_REPOSITORY}@{OPENCLIP_REVISION}/open_clip_model.safetensors"),
            sha256=sha256_file(checkpoint_path),
            source="measured",
        ),
        adapter=AdapterMetadata(
            name="clip-drop-dtw",
            version="0.1.0",
            implementation_basis=(
                "OpenCLIP ViT-B-32 frame embeddings aligned to Assembly101 coarse transcript "
                f"via SamsungLabs/Drop-DTW @ {DROP_DTW_REVISION}"
            ),
            external_source_uri=str(DROP_DTW_SOURCE),
            external_revision=DROP_DTW_REVISION,
        ),
        runtime_settings={
            "analysis_fps": proxy.fps,
            "sample_fps": args.sample_fps,
            "keep_percentile": args.keep_percentile,
            "transcript_step_count": len(steps),
            "alignment_cost": alignment["alignment_cost"],
            "openclip_architecture": "ViT-B-32",
            "openclip_pretrained": "openai",
            "openclip_repository": OPENCLIP_REPOSITORY,
            "openclip_revision": OPENCLIP_REVISION,
        },
        measurements=RuntimeMeasurements(
            elapsed_seconds=float(worker_result["elapsed_seconds"]),
            known_unavailable_measures=("temporal action segmentation accuracy",),
        ),
        alignment_uri=relative_uri(alignment_path, repository_root),
        rerun_artifact_uri=relative_uri(rrd_path, repository_root),
        drop_dtw_revision=DROP_DTW_REVISION,
    )
    manifest = RunManifest(
        run_id=run_id,
        clip=config.clip.model_copy(update={"source_duration_seconds": args.seconds}),
        coverage=FullDurationCoverage(
            source_duration_seconds=args.seconds,
            covered_intervals=(TimeInterval(start_seconds=0.0, end_seconds=args.seconds),),
        ),
        chunk_policy=ChunkContinuityPolicy(overlap_seconds=0.0, max_allowed_gap_seconds=0.0),
        method_statuses=(
            MethodStatus(
                method_name="clip-drop-dtw-weak-supervision",
                stage="alignment",
                state=MethodState.SUCCEEDED,
                artifact_uri=relative_uri(alignment_path, repository_root),
                measured_on=(
                    f"{args.view}; Assembly101 coarse GT transcript; {args.seconds:g}s prefix"
                ),
            ),
            MethodStatus(
                method_name="rerun-drop-dtw-export",
                stage="export",
                state=MethodState.SUCCEEDED,
                artifact_uri=relative_uri(rrd_path, repository_root),
                measured_on="alignment cost and exploratory interval scalars only",
            ),
        ),
        drop_dtw=metadata,
    )
    manifest_path = run_directory / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_repository_root(parser)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--view", default=DEFAULT_VIEW_ID)
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--keep-percentile", type=float, default=0.3)
    parser.add_argument("--openclip-checkpoint", type=Path, default=DEFAULT_OPENCLIP_CHECKPOINT)
    parser.add_argument("--openclip-cache-dir", type=Path, default=DEFAULT_OPENCLIP_CACHE)
    add_output_root(parser, Path("runs"))
    parser.add_argument("--run-id")
    args = parser.parse_args()
    print(run(args))


if __name__ == "__main__":
    main()
