"""Run one real, bounded four-part segmentation arm with a common reviewed contract."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2

from .four_part_contract import (
    ANALYSIS_FPS,
    FRAME_COUNT,
    TARGETS,
    FourPartContract,
    load_contract,
    relative_uri,
    seed_manifest,
)
from .schemas import (
    AdapterMetadata,
    ArtifactFingerprint,
    ChunkContinuityPolicy,
    FourPartSegmentationRunMetadata,
    FourPartTargetInitialization,
    FrameObservations,
    FrameRange,
    FullDurationCoverage,
    G2PreprocessingManifest,
    MaskReference,
    MethodState,
    MethodStatus,
    NormalizedBox,
    PerFrameObject,
    RunManifest,
    RuntimeMeasurements,
    TimeInterval,
)

DEFAULT_CONTRACT = Path("configs/four_part_segmentation_comparison.json")
METHODS = {
    "grounding_dino_sam2_open_vocabulary": {
        "python": Path("/home/nick/.pyenv/versions/grounded_sam2/bin/python"),
        "root": Path("/home/nick/src/Grounded-SAM-2"),
        "revision": "b7a9c29f196edff0eb54dbe14588d7ae5e3dde28",
        "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "checkpoint": Path("/home/nick/src/Grounded-SAM-2/checkpoints/sam2.1_hiera_tiny.pt"),
        "basis": "independent Grounding-DINO prompts on frame zero followed by SAM2 propagation",
    },
    "reviewed_seed_sam2_control": {
        "python": Path("/home/nick/.pyenv/versions/grounded_sam2/bin/python"),
        "root": Path("/home/nick/src/Grounded-SAM-2"),
        "revision": "b7a9c29f196edff0eb54dbe14588d7ae5e3dde28",
        "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "checkpoint": Path("/home/nick/src/Grounded-SAM-2/checkpoints/sam2.1_hiera_tiny.pt"),
        "basis": "SAM2 propagation initialized from the shared reviewed frame-zero masks",
    },
    "samurai": {
        "python": Path("/home/nick/.pyenv/versions/samurai/bin/python"),
        "root": Path("/home/nick/src/samurai"),
        "revision": "76ba195984892b0d1e3db5d9c90bb62175680a",
        "config": "configs/samurai/sam2.1_hiera_t.yaml",
        "checkpoint": Path("/home/nick/src/samurai/sam2/checkpoints/sam2.1_hiera_tiny.pt"),
        "basis": "SAMURAI config with samurai_mode=true and multi-object SAM2 mask prompts",
    },
    "dam4sam": {
        "python": Path("/home/nick/.pyenv/versions/samurai/bin/python"),
        "root": Path("/home/nick/src/DAM4SAM"),
        "revision": "9c954504b39ebca4c412f207be0787c26bfac85a",
        "config": "sam21pp_hiera_t.yaml",
        "checkpoint": Path("/home/nick/src/DAM4SAM/checkpoints/sam2.1_hiera_tiny.pt"),
        "basis": "four independent DAM4SAMTracker DRM streams initialized from reviewed masks",
    },
}


def _load_observations(path: Path) -> tuple[FrameObservations, ...]:
    output: list[FrameObservations] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        output.append(
            FrameObservations(
                view_id=payload["view_id"],
                analysis_frame_index=payload["analysis_frame_index"],
                source_seconds=payload["source_seconds"],
                objects=tuple(
                    PerFrameObject(
                        object_id=item["object_id"],
                        label=item["label"],
                        confidence=item["confidence"],
                        box=NormalizedBox(**item["box"]),
                        mask=MaskReference(**item["mask"]) if item.get("mask") else None,
                    )
                    for item in payload["objects"]
                ),
            )
        )
    return tuple(output)


def _bounded_video(proxy: Path, destination: Path, frame_count: int = FRAME_COUNT) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(proxy),
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
            str(destination),
        ],
        check=True,
    )


def _contact_sheet(
    video: Path, observations: tuple[FrameObservations, ...], destination: Path
) -> None:
    losses = [
        observation.analysis_frame_index
        for observation in observations
        if len(observation.objects) < len(TARGETS)
    ]
    frame_count = len(observations)
    selected = sorted({0, frame_count // 2, frame_count - 1, *losses[:3]})
    capture = cv2.VideoCapture(str(video))
    panels: list[Any] = []
    colors = {
        "chassis": (255, 120, 50),
        "interior": (60, 220, 255),
        "rear_body": (50, 220, 80),
        "cabin": (210, 80, 255),
    }
    try:
        for index in selected:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"failed to decode contact sheet frame {index}")
            height, width = frame.shape[:2]
            observation = observations[index]
            for item in observation.objects:
                if item.mask:
                    mask = cv2.imread(str(destination.parent / item.mask.uri), cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        overlay = frame.copy()
                        overlay[mask > 0] = colors[item.label]
                        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
                x1, y1 = round(item.box.x * width), round(item.box.y * height)
                x2 = round((item.box.x + item.box.width) * width)
                y2 = round((item.box.y + item.box.height) * height)
                cv2.rectangle(frame, (x1, y1), (x2, y2), colors[item.label], 2)
                cv2.putText(
                    frame,
                    item.label,
                    (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    colors[item.label],
                    2,
                )
            missing = sorted(set(TARGETS) - {item.label for item in observation.objects})
            cv2.putText(
                frame,
                f"frame {index}; missing: {', '.join(missing) or 'none'}",
                (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
            panels.append(frame)
    finally:
        capture.release()
    if not cv2.imwrite(str(destination), cv2.vconcat(panels)):
        raise RuntimeError(f"failed to write {destination}")


def _initializations(
    method: str, contract: FourPartContract, result: dict[str, Any], repository_root: Path
) -> tuple[FourPartTargetInitialization, ...]:
    report = result.get("initialization", {})
    output = []
    for seed in contract.seeds:
        item = report.get(seed.target_id, {})
        if method == "grounding_dino_sam2_open_vocabulary":
            succeeded = item.get("status") == "succeeded"
            output.append(
                FourPartTargetInitialization(
                    target_id=seed.target_id,
                    source="open_vocabulary_detection",
                    state="succeeded" if succeeded else "failed",
                    prompts=contract.open_vocabulary_prompts[seed.target_id],
                    selected_prompt=item.get("selected_prompt"),
                    selected_score=item.get("selected_score"),
                    derived_box_xyxy=tuple(round(value) for value in item["selected_box_xyxy"])
                    if succeeded
                    else None,
                    failure_reason=None
                    if succeeded
                    else "no Grounding-DINO detection above thresholds",
                )
            )
        else:
            output.append(
                FourPartTargetInitialization(
                    target_id=seed.target_id,
                    source="reviewed_mask",
                    state="succeeded",
                    derived_box_xyxy=seed.box_xyxy,
                    reviewed_mask_fingerprint=ArtifactFingerprint(
                        uri=relative_uri(seed.mask_path, repository_root),
                        sha256=seed.mask_sha256,
                        source="measured",
                    ),
                )
            )
    return tuple(output)


def run(args: argparse.Namespace) -> Path:
    repository_root = args.repository_root.resolve()
    contract = load_contract(repository_root, args.contract)
    spec = METHODS[args.method]
    if not spec["python"].is_file() or not spec["checkpoint"].is_file():
        raise FileNotFoundError(f"{args.method} environment or checkpoint is unavailable")
    config = G2PreprocessingManifest.model_validate_json(
        (
            repository_root
            / "configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_g2.json"
        ).read_text()
    )
    proxy = config.proxies[0]
    frame_count = int(args.frame_count)
    if frame_count not in (FRAME_COUNT, 3 * FRAME_COUNT):
        raise ValueError("four-part arms run over exactly 600 (20 s) or 1800 (60 s) frames")
    requested_seconds = frame_count / ANALYSIS_FPS
    run_id = args.run_id or f"{args.method}-{datetime.now(UTC):%Y%m%dt%H%M%Sz}"
    run_directory = repository_root / args.output_root / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    input_video = run_directory / "input.mp4"
    _bounded_video(contract.proxy_path, input_video, frame_count)
    worker_contract = seed_manifest(contract, repository_root)
    worker_contract["open_vocabulary_prompts"] = contract.open_vocabulary_prompts
    for seed in worker_contract["reviewed_frame_zero_seeds"]:
        seed["mask_path"] = str((repository_root / seed["mask_fingerprint"]["uri"]).resolve())
        seed["mask_sha256"] = seed["mask_fingerprint"]["sha256"]
    worker_contract_path = run_directory / "reviewed_seed_contract.json"
    worker_contract_path.write_text(json.dumps(worker_contract, indent=2) + "\n")
    command = [
        str(spec["python"]),
        str(Path(__file__).with_name("four_part_video_worker.py").resolve()),
        "--method",
        args.method,
        "--run-directory",
        str(run_directory),
        "--video",
        str(input_video),
        "--contract",
        str(worker_contract_path),
        "--source-offset-seconds",
        str(contract.source_offset_seconds),
        "--analysis-fps",
        str(ANALYSIS_FPS),
        "--frame-count",
        str(frame_count),
        "--sam2-root",
        str(spec["root"] / "sam2" if args.method == "samurai" else spec["root"]),
        "--sam2-config",
        str(spec["config"]),
        "--checkpoint",
        str(spec["checkpoint"]),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    completed = subprocess.run(
        command, cwd=spec["root"], env=environment, capture_output=True, text=True
    )
    (run_directory / "worker.stdout.log").write_text(completed.stdout)
    (run_directory / "worker.stderr.log").write_text(completed.stderr)
    (run_directory / "worker_command.txt").write_text(" ".join(command) + "\n")
    result = json.loads((run_directory / "worker_result.json").read_text())
    observations_path = run_directory / "observations.jsonl"
    if not observations_path.is_file():
        raise RuntimeError(result.get("reason", f"worker exited {completed.returncode}"))
    observations = _load_observations(observations_path)
    if len(observations) != frame_count or [
        item.analysis_frame_index for item in observations
    ] != list(range(frame_count)):
        raise RuntimeError(f"worker did not write exactly {frame_count} ordered observations")
    contact_sheet = run_directory / "contact_sheet.png"
    _contact_sheet(input_video, observations, contact_sheet)
    metadata = FourPartSegmentationRunMetadata(
        method_arm=args.method,
        requested_analysis_frame_range=FrameRange(start_frame=0, end_frame_exclusive=frame_count),
        requested_seconds=requested_seconds,
        target_order=TARGETS,
        source_fingerprint=ArtifactFingerprint(
            uri=proxy.raw_source.raw_uri,
            sha256=proxy.raw_source.checksum_sha256,
            source="approved_config",
        ),
        proxy_fingerprint=ArtifactFingerprint(
            uri=proxy.proxy_uri, sha256=proxy.checksum_sha256, source="approved_config"
        ),
        contract_fingerprint=ArtifactFingerprint(
            uri=relative_uri(contract.path, repository_root),
            sha256=contract.fingerprint,
            source="measured",
        ),
        reviewed_schedule_fingerprint=ArtifactFingerprint(
            uri="runs/muggledsam-sam3-four-part-focused-corrections-327-1235-20260916t022433z/multi_keyframe_correction_schedule.json",
            sha256="e5981586452cd2f934ee527e89f2ca3f7d43c1ae8fa726bb72add68d20746a99",
            source="measured",
        ),
        adapter=AdapterMetadata(
            name=args.method,
            version="1.0.0",
            implementation_basis=str(spec["basis"]),
            external_source_uri=str(spec["root"]),
            external_revision=str(spec["revision"]),
        ),
        runtime_settings={
            "analysis_fps": ANALYSIS_FPS,
            "frame_count": frame_count,
            "sam2_config": str(spec["config"]),
            "samurai_mode": args.method == "samurai",
            "initialization": "independent_detection"
            if args.method == "grounding_dino_sam2_open_vocabulary"
            else "shared_reviewed_masks",
        },
        measurements=RuntimeMeasurements(
            elapsed_seconds=float(result["elapsed_seconds"]),
            gpu_peak_vram_bytes=result.get("gpu_peak_vram_bytes"),
            known_unavailable_measures=("ground-truth mask quality",),
        ),
        observations_uri=relative_uri(observations_path, repository_root),
        native_masks_uri=relative_uri(run_directory / "native" / "masks", repository_root),
        qa_artifact_uri=relative_uri(contact_sheet, repository_root),
        target_initializations=_initializations(args.method, contract, result, repository_root),
        drm_memory_additions=result.get("drm_memory_additions")
        if args.method == "dam4sam"
        else None,
    )
    state = MethodState.SUCCEEDED if result["state"] == "succeeded" else MethodState.FAILED
    manifest = RunManifest(
        run_id=run_id,
        clip=config.clip.model_copy(update={"source_duration_seconds": requested_seconds}),
        coverage=FullDurationCoverage(
            source_duration_seconds=requested_seconds,
            covered_intervals=(TimeInterval(start_seconds=0.0, end_seconds=requested_seconds),),
        ),
        chunk_policy=ChunkContinuityPolicy(
            overlap_seconds=0.0,
            max_allowed_gap_seconds=0.0,
            preserve_track_ids=True,
            carry_context_across_chunks=True,
        ),
        method_statuses=(
            MethodStatus(
                method_name=args.method,
                stage="objects",
                state=state,
                artifact_uri=relative_uri(observations_path, repository_root),
                measured_on=f"focused static RGB frames [0,{frame_count})",
            ),
        ),
        observations=observations,
        four_part_segmentation=metadata,
    )
    output = run_directory / "manifest.json"
    output.write_text(manifest.model_dump_json(indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=METHODS)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--run-id")
    parser.add_argument(
        "--frame-count",
        type=int,
        default=FRAME_COUNT,
        choices=(FRAME_COUNT, 3 * FRAME_COUNT),
        help="600 keeps the 20 s comparison contract; 1800 extends the same seeds to 60 s.",
    )
    print(run(parser.parse_args()))


if __name__ == "__main__":
    main()
