"""Run one real, bounded four-part segmentation arm with a common reviewed contract."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import cv2

from . import digest_cache
from .cli_common import add_output_root, add_repository_root
from .dam4sam_streaming import (
    DEFAULT_INPUT_IMAGE_SIZE,
    SAM2_MODELS,
    SUPPORTED_INPUT_IMAGE_SIZES,
    corrections_by_frame,
    extrapolate_vram,
    parse_probe_frames,
)
from .four_part_contract import (
    ANALYSIS_FPS,
    FRAME_COUNT,
    TARGETS,
    FourPartContract,
    load_contract,
    relative_uri,
    seed_manifest,
)
from .fs_common import run_timestamp
from .observations import rebuild_tracker_observations as _load_observations
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
    MethodState,
    MethodStatus,
    RunManifest,
    RuntimeMeasurements,
    Sam2ArmSettings,
    TimeInterval,
    VramExtrapolation,
    VramProbe,
)
from .video_driver import bounded_video as _bounded_video

DEFAULT_CONTRACT = Path("configs/four_part_segmentation_comparison.json")
SMOKE_FRAME_COUNT = FRAME_COUNT // 2
FRAME_COUNT_CHOICES = (SMOKE_FRAME_COUNT, FRAME_COUNT, 3 * FRAME_COUNT)
DEFAULT_SAM2_MODEL = "tiny"
DEFAULT_VRAM_PROBE_FRAMES = "30,300"
DEFAULT_EXTRAPOLATE_TO_FRAMES = 3 * FRAME_COUNT
DEFAULT_VRAM_LIMIT_BYTES = 12 * 1024**3
RESOLVED_SCHEDULE_NAME = "correction_schedule.resolved.json"
# `config`/`checkpoint` stay the tiny pair every earlier run used; `sam2` maps
# `--sam2-model` onto each checkout's own copy of the SAM2.1 Hiera checkpoint + yaml.
METHODS = {
    "grounding_dino_sam2_open_vocabulary": {
        "python": Path("/home/nick/.pyenv/versions/grounded_sam2/bin/python"),
        "root": Path("/home/nick/src/Grounded-SAM-2"),
        "revision": "b7a9c29f196edff0eb54dbe14588d7ae5e3dde28",
        "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "checkpoint": Path("/home/nick/src/Grounded-SAM-2/checkpoints/sam2.1_hiera_tiny.pt"),
        "sam2": {
            "tiny": (
                "configs/sam2.1/sam2.1_hiera_t.yaml",
                Path("/home/nick/src/Grounded-SAM-2/checkpoints/sam2.1_hiera_tiny.pt"),
            ),
            "large": (
                "configs/sam2.1/sam2.1_hiera_l.yaml",
                Path("/home/nick/src/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"),
            ),
        },
        "basis": "independent Grounding-DINO prompts on frame zero followed by SAM2 propagation",
    },
    "reviewed_seed_sam2_control": {
        "python": Path("/home/nick/.pyenv/versions/grounded_sam2/bin/python"),
        "root": Path("/home/nick/src/Grounded-SAM-2"),
        "revision": "b7a9c29f196edff0eb54dbe14588d7ae5e3dde28",
        "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "checkpoint": Path("/home/nick/src/Grounded-SAM-2/checkpoints/sam2.1_hiera_tiny.pt"),
        "sam2": {
            "tiny": (
                "configs/sam2.1/sam2.1_hiera_t.yaml",
                Path("/home/nick/src/Grounded-SAM-2/checkpoints/sam2.1_hiera_tiny.pt"),
            ),
            "large": (
                "configs/sam2.1/sam2.1_hiera_l.yaml",
                Path("/home/nick/src/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"),
            ),
        },
        "basis": "SAM2 propagation initialized from the shared reviewed frame-zero masks",
    },
    "samurai": {
        "python": Path("/home/nick/.pyenv/versions/samurai/bin/python"),
        "root": Path("/home/nick/src/samurai"),
        "revision": "76ba195984892b0d1e3db5d9c90bb62175680a",
        "config": "configs/samurai/sam2.1_hiera_t.yaml",
        "checkpoint": Path("/home/nick/src/samurai/sam2/checkpoints/sam2.1_hiera_tiny.pt"),
        "sam2": {
            "tiny": (
                "configs/samurai/sam2.1_hiera_t.yaml",
                Path("/home/nick/src/samurai/sam2/checkpoints/sam2.1_hiera_tiny.pt"),
            ),
            "large": (
                "configs/samurai/sam2.1_hiera_l.yaml",
                Path("/home/nick/src/samurai/sam2/checkpoints/sam2.1_hiera_large.pt"),
            ),
        },
        "basis": "SAMURAI config with samurai_mode=true and multi-object SAM2 mask prompts",
    },
    "dam4sam": {
        "python": Path("/home/nick/.pyenv/versions/samurai/bin/python"),
        "root": Path("/home/nick/src/DAM4SAM"),
        "revision": "9c954504b39ebca4c412f207be0787c26bfac85a",
        "config": "sam21pp_hiera_t.yaml",
        "checkpoint": Path("/home/nick/src/DAM4SAM/checkpoints/sam2.1_hiera_tiny.pt"),
        "sam2": {
            model: (entry["config"], Path("/home/nick/src/DAM4SAM") / entry["checkpoint"])
            for model, entry in SAM2_MODELS.items()
        },
        "basis": (
            "four DAM4SAMTracker DRM streams sharing one SAM2 predictor, each with its own "
            "inference state, initialized from reviewed masks"
        ),
    },
}


class ExtrapolatedVramOverLimit(RuntimeError):
    """The smoke's projected peak VRAM at the full frame count exceeds the configured limit."""


def method_sam2_files(method: str, sam2_model: str) -> tuple[str, Path]:
    """(config, checkpoint) for one method arm at one SAM2 model size."""
    spec = METHODS[method]
    try:
        return spec["sam2"][sam2_model]
    except KeyError as error:
        raise ValueError(f"{method} has no SAM2 {sam2_model!r} checkpoint registered") from error


def select_proxy(config: G2PreprocessingManifest, view_id: str | None) -> Any:
    """The config's first proxy by default, or the one registered under `view_id`."""
    if view_id is None:
        return config.proxies[0]
    for proxy in config.proxies:
        if proxy.view_id == view_id:
            return proxy
    raise ValueError(f"proxy view {view_id!r} is not registered in the clip config")


def filter_corrections(
    corrections: list[dict[str, Any]],
    *,
    frame_count: int,
    smoke_correction_frame: int | None = None,
) -> tuple[list[dict[str, Any]], tuple[int, ...], int | None]:
    """Keep later corrections inside `[1, frame_count)`; a smoke may rebase the earliest.

    Returns `(applied, dropped_frame_indices, smoke_source_frame)`.  With
    `smoke_correction_frame`, only the earliest scheduled keyframe's masks are applied,
    at that frame, so a 10 s run exercises `correct()`; every scheduled frame counts as
    dropped because none is applied where it was authored.
    """
    later = [item for item in corrections if int(item["frame_index"]) > 0]
    scheduled = sorted({int(item["frame_index"]) for item in later})
    if smoke_correction_frame is not None:
        if not 0 < smoke_correction_frame < frame_count:
            raise ValueError("the smoke correction frame must lie inside the run")
        if not scheduled:
            raise ValueError("the schedule has no later correction to rebase for the smoke")
        source = scheduled[0]
        applied = [
            {**item, "frame_index": smoke_correction_frame, "source_frame_index": source}
            for item in later
            if int(item["frame_index"]) == source
        ]
        return applied, tuple(scheduled), source
    applied = [item for item in later if int(item["frame_index"]) < frame_count]
    dropped = tuple(frame for frame in scheduled if frame >= frame_count)
    return applied, dropped, None


def worker_flags(
    args: argparse.Namespace, *, view_id: str | None, schedule: Path | None
) -> list[str]:
    """Only non-default knobs are appended, so the tiny/1024 command stays byte-identical.

    `--sam2-model` reaches the worker only for DAM4SAM, which builds its own predictor from
    the model name; the offline arms (SAMURAI, Grounded-SAM-2, the reviewed-seed control)
    already receive the resolved `--sam2-config` / `--checkpoint` pair from `worker_command`
    and refuse the DAM4SAM-only knob.
    """
    flags: list[str] = []
    if view_id is not None:
        flags += ["--view-id", view_id]
    if args.method == "dam4sam" and args.sam2_model != DEFAULT_SAM2_MODEL:
        flags += ["--sam2-model", args.sam2_model]
    if int(args.input_size) != DEFAULT_INPUT_IMAGE_SIZE:
        flags += ["--input-size", str(int(args.input_size))]
    if schedule is not None:
        flags += ["--multi-keyframe-correction-schedule", str(schedule)]
    if args.add_correction_to_drm:
        flags += ["--add-correction-to-drm"]
    if args.vram_probe_frames != DEFAULT_VRAM_PROBE_FRAMES:
        flags += ["--vram-probe-frames", args.vram_probe_frames]
    return flags


def resolve_correction_schedule(
    *,
    schedule_path: Path,
    repository_root: Path,
    clip_config_path: Path,
    authoring_proxy: Any,
    contract: FourPartContract,
    frame_count: int,
    smoke_correction_frame: int | None,
    destination: Path,
) -> tuple[Path, dict[str, Any]]:
    """Validate the SAM3 schedule exactly as `battle-muggled-smoke` does and write the
    worker-readable resolution (mask paths + SHA-256 per later correction) next to the run.

    The schedule is verified against the proxy it was authored on (the contract's), even
    when the run itself uses another registered proxy; the worker resizes the masks.
    """
    from .muggled_smoke import _load_multi_keyframe_correction_schedule

    payload, metadata = _load_multi_keyframe_correction_schedule(
        schedule_path=schedule_path.resolve(),
        repository_root=repository_root,
        config_path=clip_config_path,
        proxy=authoring_proxy,
        analysis_fps=float(ANALYSIS_FPS),
        max_frame_exclusive=3 * FRAME_COUNT,
        drop_out_of_range_corrections=True,
    )
    applied, dropped, smoke_source = filter_corrections(
        list(payload["corrections"]),
        frame_count=frame_count,
        smoke_correction_frame=smoke_correction_frame,
    )
    corrections_by_frame(applied)
    contract_seed_shas = {seed.target_id: seed.mask_sha256 for seed in contract.seeds}
    schedule_seed_shas = {
        str(seed["target"]): str(seed["mask_sha256"]) for seed in payload.get("seeds", [])
    }
    resolved = {
        "schedule_fingerprint": metadata.schedule_fingerprint.model_dump(),
        "correction_policy_fingerprint": metadata.correction_policy_fingerprint.model_dump(),
        "memory_semantics_authored_for_sam3": payload.get("memory_semantics"),
        "correction_api": "add_new_mask",
        "seeds": payload.get("seeds", []),
        "corrections": applied,
        "scheduled_correction_frame_indices": sorted(
            {int(item["frame_index"]) for item in applied}
        ),
        "dropped_correction_frame_indices": sorted(
            set(dropped) | set(metadata.dropped_correction_frame_indices)
        ),
        "smoke_correction_frame": smoke_correction_frame,
        "smoke_correction_source_frame": smoke_source,
        "schedule_frame_zero_seeds_match_contract": schedule_seed_shas == contract_seed_shas,
    }
    destination.write_text(json.dumps(resolved, indent=2) + "\n", encoding="utf-8")
    return destination, resolved


def _sam2_settings(
    *,
    method: str,
    args: argparse.Namespace,
    config: str,
    checkpoint: Path,
    result: dict[str, Any],
    resolved_schedule: dict[str, Any] | None,
    spec_root: Path,
) -> Sam2ArmSettings:
    worker_sam2 = dict(result.get("sam2") or {})
    if method == "dam4sam" and worker_sam2:
        checkpoint_fingerprint = ArtifactFingerprint(
            uri=str(worker_sam2["sam2_checkpoint_uri"]),
            sha256=str(worker_sam2["sam2_checkpoint_sha256"]),
            source="measured",
        )
        pinned = bool(worker_sam2.get("sam2_checkpoint_sha256_pinned", False))
        config_fingerprint = ArtifactFingerprint(
            uri=str(worker_sam2["sam2_config_uri"]),
            sha256=str(worker_sam2["sam2_config_sha256"]),
            source="measured",
        )
        config_source = str(worker_sam2.get("sam2_config_source", "checkout"))
        input_image_size = int(worker_sam2.get("input_image_size", args.input_size))
        shared = bool(worker_sam2.get("shared_predictor", True))
    else:
        checkpoint_fingerprint = ArtifactFingerprint(
            uri=str(checkpoint), sha256=digest_cache.sha256_file(checkpoint), source="measured"
        )
        pinned = False
        # Hydra resolves these names inside each checkout's `sam2` package directory.
        package_dir = spec_root / "sam2" / "sam2" if method == "samurai" else spec_root / "sam2"
        config_path = package_dir / config
        if not config_path.is_file():
            raise FileNotFoundError(f"SAM2 yaml for {method} is missing: {config_path}")
        config_fingerprint = ArtifactFingerprint(
            uri=str(config_path), sha256=digest_cache.sha256_file(config_path), source="measured"
        )
        config_source = "checkout"
        input_image_size = DEFAULT_INPUT_IMAGE_SIZE
        shared = False
    scheduled: tuple[int, ...] = ()
    dropped: tuple[int, ...] = ()
    schedule_fingerprint = None
    timing = None
    api = None
    smoke_frame = smoke_source = None
    seeds_match = None
    if resolved_schedule is not None:
        scheduled = tuple(
            int(item) for item in resolved_schedule["scheduled_correction_frame_indices"]
        )
        dropped = tuple(int(item) for item in resolved_schedule["dropped_correction_frame_indices"])
        schedule_fingerprint = ArtifactFingerprint(**resolved_schedule["schedule_fingerprint"])
        api = "add_new_mask"
        timing = (
            "mid_stream_after_track"
            if method == "dam4sam"
            else "all_conditioning_frames_before_propagation"
        )
        smoke_frame = resolved_schedule.get("smoke_correction_frame")
        smoke_source = resolved_schedule.get("smoke_correction_source_frame")
        seeds_match = resolved_schedule.get("schedule_frame_zero_seeds_match_contract")
    return Sam2ArmSettings(
        sam2_model=args.sam2_model,
        sam2_checkpoint_fingerprint=checkpoint_fingerprint,
        sam2_checkpoint_sha256_pinned=pinned,
        sam2_config_fingerprint=config_fingerprint,
        sam2_config_source=config_source,
        input_image_size=input_image_size,
        shared_predictor=shared,
        correction_schedule_fingerprint=schedule_fingerprint,
        scheduled_correction_frame_indices=scheduled,
        dropped_correction_frame_indices=dropped,
        correction_api=api,
        correction_timing=timing,
        add_correction_to_drm=bool(args.add_correction_to_drm) and method == "dam4sam",
        correction_masks_resized_to_frame=bool(result.get("correction_masks_resized_to_frame")),
        seed_masks_resized_to_frame=bool(result.get("seed_masks_resized_to_frame")),
        smoke_correction_frame=smoke_frame,
        smoke_correction_source_frame=smoke_source,
        schedule_frame_zero_seeds_match_contract=seeds_match,
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


def worker_command(
    *,
    method: str,
    run_directory: Path,
    input_video: Path,
    worker_contract_path: Path,
    source_offset_seconds: float,
    frame_count: int,
    sam2_model: str,
    extra_flags: list[str],
) -> list[str]:
    spec = METHODS[method]
    config, checkpoint = method_sam2_files(method, sam2_model)
    return [
        str(spec["python"]),
        str(Path(__file__).with_name("four_part_video_worker.py").resolve()),
        "--method",
        method,
        "--run-directory",
        str(run_directory),
        "--video",
        str(input_video),
        "--contract",
        str(worker_contract_path),
        "--source-offset-seconds",
        str(source_offset_seconds),
        "--analysis-fps",
        str(ANALYSIS_FPS),
        "--frame-count",
        str(frame_count),
        "--sam2-root",
        str(spec["root"] / "sam2" if method == "samurai" else spec["root"]),
        "--sam2-config",
        str(config),
        "--checkpoint",
        str(checkpoint),
        *extra_flags,
    ]


def run(args: argparse.Namespace) -> Path:
    repository_root = args.repository_root.resolve()
    contract = load_contract(repository_root, args.contract)
    spec = METHODS[args.method]
    sam2_config, sam2_checkpoint = method_sam2_files(args.method, args.sam2_model)
    if not spec["python"].is_file() or not sam2_checkpoint.is_file():
        raise FileNotFoundError(f"{args.method} environment or checkpoint is unavailable")
    if int(args.input_size) not in SUPPORTED_INPUT_IMAGE_SIZES:
        raise ValueError(f"--input-size must be one of {SUPPORTED_INPUT_IMAGE_SIZES}")
    if args.method != "dam4sam" and int(args.input_size) != DEFAULT_INPUT_IMAGE_SIZE:
        raise ValueError(
            "--input-size is a DAM4SAM knob; the offline SAM2 arms take image_size from their yaml"
        )
    if args.add_correction_to_drm and args.multi_keyframe_correction_schedule is None:
        raise ValueError("--add-correction-to-drm needs --multi-keyframe-correction-schedule")
    if args.smoke_correction_frame is not None and (
        args.multi_keyframe_correction_schedule is None
        or int(args.frame_count) != SMOKE_FRAME_COUNT
    ):
        raise ValueError(
            "--smoke-correction-frame needs --multi-keyframe-correction-schedule and the "
            f"{SMOKE_FRAME_COUNT}-frame smoke"
        )
    parse_probe_frames(args.vram_probe_frames)  # fail early on a malformed probe list
    clip_config_path = (
        repository_root
        / "configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_g2.json"
    )
    config = G2PreprocessingManifest.model_validate_json(clip_config_path.read_text())
    authoring_proxy = config.proxies[0]
    proxy_override = args.proxy_clip_config is not None or args.proxy_view_id is not None
    if proxy_override:
        proxy_config = (
            G2PreprocessingManifest.model_validate_json(
                (repository_root / args.proxy_clip_config).read_text()
            )
            if args.proxy_clip_config is not None
            else config
        )
        proxy = select_proxy(proxy_config, args.proxy_view_id or authoring_proxy.view_id)
        proxy_path = (repository_root / proxy.proxy_uri).resolve()
        if (
            not proxy_path.is_file()
            or digest_cache.sha256_file(proxy_path) != proxy.checksum_sha256
        ):
            raise ValueError(f"registered proxy is unavailable or has changed: {proxy_path}")
        # The schedule and seeds are analysis-frame indices on the authoring proxy's clock;
        # another proxy may only change pixels, not timing (same fps, same raw source).
        if (
            proxy.fps != authoring_proxy.fps
            or proxy.raw_source.raw_uri != authoring_proxy.raw_source.raw_uri
        ):
            raise ValueError(
                f"proxy {proxy.view_id!r} must share the authoring proxy's raw source and fps"
            )
    else:
        proxy = authoring_proxy
        proxy_path = contract.proxy_path
    frame_count = int(args.frame_count)
    if frame_count not in FRAME_COUNT_CHOICES:
        raise ValueError(
            "four-part arms run over exactly 600 (20 s) or 1800 (60 s) frames, "
            "or 300 (10 s) for the SAM2 VRAM smoke"
        )
    requested_seconds = frame_count / ANALYSIS_FPS
    run_id = args.run_id or f"{args.method}-{run_timestamp()}"
    run_directory = repository_root / args.output_root / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    input_video = run_directory / "input.mp4"
    _bounded_video(proxy_path, input_video, frame_count)
    worker_contract = seed_manifest(contract, repository_root)
    worker_contract["open_vocabulary_prompts"] = contract.open_vocabulary_prompts
    for seed in worker_contract["reviewed_frame_zero_seeds"]:
        seed["mask_path"] = str((repository_root / seed["mask_fingerprint"]["uri"]).resolve())
        seed["mask_sha256"] = seed["mask_fingerprint"]["sha256"]
    worker_contract_path = run_directory / "reviewed_seed_contract.json"
    worker_contract_path.write_text(json.dumps(worker_contract, indent=2) + "\n")
    resolved_schedule: dict[str, Any] | None = None
    resolved_schedule_path: Path | None = None
    if args.multi_keyframe_correction_schedule is not None:
        resolved_schedule_path, resolved_schedule = resolve_correction_schedule(
            schedule_path=repository_root / args.multi_keyframe_correction_schedule,
            repository_root=repository_root,
            clip_config_path=clip_config_path,
            authoring_proxy=authoring_proxy,
            contract=contract,
            frame_count=frame_count,
            smoke_correction_frame=args.smoke_correction_frame,
            destination=run_directory / RESOLVED_SCHEDULE_NAME,
        )
        if args.add_correction_to_drm and not resolved_schedule["corrections"]:
            raise ValueError(
                "--add-correction-to-drm has nothing to act on: every scheduled correction "
                f"lies outside the {frame_count}-frame run"
            )
    command = worker_command(
        method=args.method,
        run_directory=run_directory,
        input_video=input_video,
        worker_contract_path=worker_contract_path,
        source_offset_seconds=contract.source_offset_seconds,
        frame_count=frame_count,
        sam2_model=args.sam2_model,
        extra_flags=worker_flags(
            args,
            view_id=proxy.view_id if proxy.view_id != authoring_proxy.view_id else None,
            schedule=resolved_schedule_path,
        ),
    )
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
    probes = tuple(VramProbe(**probe) for probe in result.get("vram_probes", []))
    extrapolation = None
    if len(probes) >= 2:
        extrapolation = VramExtrapolation(
            **extrapolate_vram(
                [probe.model_dump() for probe in probes],
                to_frames=int(args.extrapolate_to_frames),
                limit_bytes=int(args.fail_if_extrapolated_vram_over_bytes),
            )
        )
    sam2_settings = _sam2_settings(
        method=args.method,
        args=args,
        config=sam2_config,
        checkpoint=sam2_checkpoint,
        result=result,
        resolved_schedule=resolved_schedule,
        spec_root=spec["root"],
    )
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
            version="1.1.0",
            implementation_basis=str(spec["basis"]),
            external_source_uri=str(spec["root"]),
            external_revision=str(spec["revision"]),
        ),
        runtime_settings={
            "analysis_fps": ANALYSIS_FPS,
            "frame_count": frame_count,
            "sam2_config": str(sam2_config),
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
        sam2_settings=sam2_settings,
        vram_probes=probes,
        vram_extrapolation=extrapolation,
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
    if extrapolation is not None:
        print(vram_projection_line(extrapolation, frame_count=frame_count))
        if frame_count < extrapolation.extrapolate_to_frames and not extrapolation.within_limit:
            raise ExtrapolatedVramOverLimit(
                f"projected peak {extrapolation.projected_peak_bytes / 1024**3:.2f} GiB at "
                f"{extrapolation.extrapolate_to_frames} frames exceeds the "
                f"{extrapolation.limit_bytes / 1024**3:.2f} GiB limit; manifest written to {output}"
            )
    return output


def vram_projection_line(extrapolation: VramExtrapolation, *, frame_count: int) -> str:
    gib = 1024**3
    first, last = extrapolation.from_frames
    return (
        f"vram: slope {extrapolation.slope_bytes_per_frame / 2**20:.3f} MiB/frame between "
        f"frames {first} and {last}; projected at {extrapolation.extrapolate_to_frames} frames: "
        f"allocated {extrapolation.projected_allocated_bytes / gib:.2f} GiB, peak "
        f"{extrapolation.projected_peak_bytes / gib:.2f} GiB; limit "
        f"{extrapolation.limit_bytes / gib:.2f} GiB; "
        f"{'within' if extrapolation.within_limit else 'OVER'} limit"
        + (
            " (this run already covered the full frame count)"
            if frame_count >= extrapolation.extrapolate_to_frames
            else ""
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=METHODS)
    add_repository_root(parser)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    add_output_root(parser, Path("runs"))
    parser.add_argument("--run-id")
    parser.add_argument(
        "--frame-count",
        type=int,
        default=FRAME_COUNT,
        choices=FRAME_COUNT_CHOICES,
        help=(
            "600 keeps the 20 s comparison contract; 1800 extends the same seeds to 60 s; "
            "300 is the 10 s VRAM smoke for the SAM2 knobs (never compared)."
        ),
    )
    parser.add_argument(
        "--sam2-model",
        choices=tuple(SAM2_MODELS),
        default=DEFAULT_SAM2_MODEL,
        help="SAM2.1 Hiera checkpoint + yaml pair; tiny reproduces every earlier arm.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=DEFAULT_INPUT_IMAGE_SIZE,
        choices=SUPPORTED_INPUT_IMAGE_SIZES,
        help="DAM4SAM only: tracker input size and yaml image_size; 1536 needs the 1080p proxy.",
    )
    parser.add_argument(
        "--multi-keyframe-correction-schedule",
        type=Path,
        default=None,
        help="SAM3 correction schedule JSON; later corrections are applied via add_new_mask.",
    )
    parser.add_argument(
        "--add-correction-to-drm",
        action="store_true",
        help="DAM4SAM: count a correction frame as a DRM addition (bumps last_added).",
    )
    parser.add_argument(
        "--smoke-correction-frame",
        type=int,
        default=None,
        help=(
            "Smoke only: apply the earliest scheduled keyframe's masks at this frame so a "
            "300-frame run exercises correct(); the masks are geometrically stale there."
        ),
    )
    parser.add_argument(
        "--proxy-clip-config",
        type=Path,
        default=None,
        help=(
            "Sibling G2 config whose proxy replaces the contract's pixels, e.g. "
            "configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_1080p_g2.json; "
            "the schedule stays validated against the 720p proxy it was authored on."
        ),
    )
    parser.add_argument(
        "--proxy-view-id",
        default=None,
        help="View ID of the proxy inside the (proxy) clip config; default the contract's.",
    )
    parser.add_argument(
        "--vram-probe-frames",
        default=DEFAULT_VRAM_PROBE_FRAMES,
        help="Frames-processed counts at which the worker records torch.cuda counters.",
    )
    parser.add_argument(
        "--extrapolate-to-frames",
        type=int,
        default=DEFAULT_EXTRAPOLATE_TO_FRAMES,
        help="Frame count the two probes' slope is projected to.",
    )
    parser.add_argument(
        "--fail-if-extrapolated-vram-over-bytes",
        type=int,
        default=DEFAULT_VRAM_LIMIT_BYTES,
        help="After a short run, exit non-zero when the projected peak exceeds this (12 GiB).",
    )
    return parser


def main() -> None:
    print(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
