"""Headless, fixed-budget adapter for a MuggledSAM/SAM3 core-method smoke test."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import digest_cache, mask_cache
from .exporter import export_run
from .schemas import (
    AdapterMetadata,
    ArtifactFingerprint,
    ChunkContinuityPolicy,
    ClockName,
    E4CandidateRunMetadata,
    EncodedAssetInput,
    FourPartFocusedRunMetadata,
    FourPartFullRunMetadata,
    FourPartPilotRunMetadata,
    FrameObservations,
    FrameRange,
    FullDurationCoverage,
    FullEgoManualSeedRunMetadata,
    G2PreprocessingManifest,
    G3CandidateRunMetadata,
    HybridInitializationMetadata,
    HybridSmokeHumanApproval,
    HybridTargetInitializationProvenance,
    ManualSeedCandidateProvenance,
    ManualSeedMultiplexMetadata,
    MethodState,
    MethodStatus,
    MuggledSAMBoxCalibrationManifest,
    MuggledSAMEgoCondition,
    MuggledSAMEgoConditionConfig,
    MuggledSAMHybridInitializationConfig,
    MuggledSAMManualSeedTargetConfig,
    MuggledSAMMultiKeyframeCorrectionPolicy,
    MuggledSAMMultiKeyframeCorrectionSchedule,
    MuggledSAMProposedTrackingPromptConfig,
    MuggledSAMTextTargetConfig,
    MultiKeyframeCorrectionScheduleMetadata,
    RunManifest,
    RuntimeMeasurements,
    SmokeRunMetadata,
    StreamContinuityPolicy,
    TimeInterval,
)

ADAPTER_VERSION = "0.3.0"
CONCEPTS = ("hand", "yellow toy body", "toy wheel")
MANUAL_SEED_TARGETS = (
    ("left_hand", "t000000-b01", 0),
    ("yellow_toy_body", "t000000-b03", 0),
    ("toy_wheel", "t000000-b04", 0),
)
E4_ZERO_SHOT_BASELINE = Path("runs/muggledsam-sam3-smoke-ego-hmc21179183-20260909t033125z")
SMOKE_FRAMES = 300
SMOKE_SECONDS = 10.0
# Worker defaults, restated so a changed condition is visible in the run manifest.
DEFAULT_MAX_SIDE_LENGTH = 504
DEFAULT_MAX_FRAME_MEMORY = 4
# Correction schedules store analysis frame indices, so they bind to one analysis rate.
SCHEDULE_AUTHORING_FPS = 30.0
G3_STATIC_FRAMES = 5400
G3_STATIC_SECONDS = 180.0
E4_CANDIDATE_FRAMES = 1800
E4_CANDIDATE_SECONDS = 60.0
FULL_EGO_MANUAL_SEED_FRAMES = 5400
FULL_EGO_MANUAL_SEED_SECONDS = 180.0
FOUR_PART_PILOT_FRAMES = 600
FOUR_PART_PILOT_SECONDS = 20.0
FOUR_PART_FULL_FRAMES = 5901
FOUR_PART_FULL_SECONDS = 196.7
FOUR_PART_FOCUSED_FRAMES = 2781
FOUR_PART_FOCUSED_SECONDS = 92.7
MUGGLED_SAM_SOURCE = Path("/home/nick/src/muggled_sam")
MUGGLED_SAM_PYTHON = Path("/home/nick/.pyenv/versions/muggled_sam/bin/python")
DEFAULT_MODEL = MUGGLED_SAM_SOURCE / "model_weights" / "sam3.1_multiplex.pt"


def sha256_file(path: Path) -> str:
    """Hash a file, reusing a cached digest while its size and mtime are unchanged."""
    return digest_cache.sha256_file(path)


def relative_uri(path: Path, repository_root: Path) -> str:
    """Prefer portable repository-relative paths in persisted metadata."""
    try:
        return path.relative_to(repository_root).as_posix()
    except ValueError:
        return path.as_posix()


def load_manual_seed_target_config(
    *,
    target_config_path: Path,
    repository_root: Path,
    g2_config_path: Path,
    view_id: str,
    also_permitted_g2_config_path: Path | None = None,
) -> MuggledSAMManualSeedTargetConfig:
    """Load a named e4 target policy bound to the selected approved G2 configuration.

    A second configuration is permitted only once a frame-zero calibration transfer has
    been verified, so the policy may follow its seeds onto an equivalent proxy.
    """
    target_config = MuggledSAMManualSeedTargetConfig.model_validate_json(
        target_config_path.read_text()
    )
    if target_config.view_id != view_id:
        raise ValueError("manual-seed target config view must match the selected approved proxy")
    permitted = {g2_config_path.resolve()}
    if also_permitted_g2_config_path is not None:
        permitted.add(also_permitted_g2_config_path.resolve())
    if (repository_root / target_config.base_g2_config).resolve() not in permitted:
        raise ValueError("manual-seed target config must name the selected G2 configuration")
    return target_config


def load_text_target_config(
    *,
    target_config_path: Path,
    repository_root: Path,
    g2_config_path: Path,
    view_id: str,
) -> MuggledSAMTextTargetConfig:
    """Load a versioned static prompt-to-output-label contract."""
    target_config = MuggledSAMTextTargetConfig.model_validate_json(target_config_path.read_text())
    if target_config.view_id != view_id:
        raise ValueError("text-target config view must match the selected approved proxy")
    if (repository_root / target_config.base_g2_config).resolve() != g2_config_path.resolve():
        raise ValueError("text-target config must name the selected G2 configuration")
    return target_config


def make_run_id(view_id: str, now: datetime | None = None, *, profile: str = "smoke") -> str:
    timestamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ").lower()
    return f"muggledsam-sam3-{profile}-{view_id.lower()}-{timestamp}"


def smoke_frame_count(analysis_fps: float) -> int:
    """Express the approved bound in seconds, so it holds at either analysis rate."""
    return round(SMOKE_SECONDS * analysis_fps)


def selected_frame_budget(args: argparse.Namespace, analysis_fps: float) -> int:
    """Resolve one CLI profile to its frame budget before loading prompt artifacts."""
    if getattr(args, "g3_full_static", False):
        return G3_STATIC_FRAMES
    if getattr(args, "g4_e4_candidate", False):
        return E4_CANDIDATE_FRAMES
    if getattr(args, "full_ego_manual_seed", False):
        return FULL_EGO_MANUAL_SEED_FRAMES
    if getattr(args, "four_part_static_pilot", False):
        return FOUR_PART_PILOT_FRAMES
    if getattr(args, "four_part_static_full", False):
        return FOUR_PART_FULL_FRAMES
    if (
        getattr(args, "four_part_static_focused", False)
        or getattr(args, "four_part_ego_focused", False)
    ):
        return FOUR_PART_FOCUSED_FRAMES
    return smoke_frame_count(analysis_fps)


def require_smoke_range(
    start_frame: int, max_frames: int, analysis_fps: float = 30.0
) -> FrameRange:
    """Refuse ranges other than the user-approved first ten seconds."""
    permitted = smoke_frame_count(analysis_fps)
    if start_frame != 0 or max_frames != permitted:
        raise ValueError(
            f"smoke policy permits only the first {SMOKE_SECONDS:.0f} seconds, which is "
            f"proxy frames [0, {permitted}) at {analysis_fps:g} fps; "
            f"received [{start_frame}, {start_frame + max_frames})"
        )
    return FrameRange(start_frame=start_frame, end_frame_exclusive=permitted)


def require_g3_static_range(view_id: str, start_frame: int, max_frames: int) -> FrameRange:
    """Permit only the exact human-approved full static candidate range."""
    if view_id != "static-c10379" or start_frame != 0 or max_frames != G3_STATIC_FRAMES:
        raise ValueError("G3 permits only static-c10379 proxy frames [0, 5400)")
    return FrameRange(start_frame=start_frame, end_frame_exclusive=G3_STATIC_FRAMES)


def require_e4_candidate_range(view_id: str, start_frame: int, max_frames: int) -> FrameRange:
    """Permit only the human-approved first 60 seconds of e4."""
    if view_id != "ego-hmc21179183" or start_frame != 0 or max_frames != E4_CANDIDATE_FRAMES:
        raise ValueError("e4 candidate permits only ego-hmc21179183 proxy frames [0, 1800)")
    return FrameRange(start_frame=start_frame, end_frame_exclusive=E4_CANDIDATE_FRAMES)


def require_full_ego_manual_seed_range(
    view_id: str, start_frame: int, max_frames: int
) -> FrameRange:
    """Permit only the human-approved full ego manual-seed baseline range."""
    if (
        view_id != "ego-hmc21179183"
        or start_frame != 0
        or max_frames != FULL_EGO_MANUAL_SEED_FRAMES
    ):
        raise ValueError(
            "full ego manual-seed baseline permits only ego-hmc21179183 proxy frames [0, 5400)"
        )
    return FrameRange(start_frame=start_frame, end_frame_exclusive=FULL_EGO_MANUAL_SEED_FRAMES)


def require_four_part_pilot_range(view_id: str, start_frame: int, max_frames: int) -> FrameRange:
    """Permit only the approved first 20 seconds of the static four-part clip."""
    if view_id != "static-c10379" or start_frame != 0 or max_frames != FOUR_PART_PILOT_FRAMES:
        raise ValueError("four-part pilot permits only static-c10379 proxy frames [0, 600)")
    return FrameRange(start_frame=start_frame, end_frame_exclusive=FOUR_PART_PILOT_FRAMES)


def require_four_part_full_range(view_id: str, start_frame: int, max_frames: int) -> FrameRange:
    """Permit only the entire approved source-aligned static four-part proxy."""
    if view_id != "static-c10379" or start_frame != 0 or max_frames != FOUR_PART_FULL_FRAMES:
        raise ValueError("four-part full run permits only static-c10379 proxy frames [0, 5901)")
    return FrameRange(start_frame=start_frame, end_frame_exclusive=FOUR_PART_FULL_FRAMES)


def require_four_part_focused_range(
    view_id: str, start_frame: int, max_frames: int
) -> FrameRange:
    """Permit only the separated-to-assembled focused four-part proxy."""
    if (
        view_id != "static-c10379"
        or start_frame != 0
        or max_frames != FOUR_PART_FOCUSED_FRAMES
    ):
        raise ValueError(
            "focused four-part run permits only static-c10379 proxy frames [0, 2781)"
        )
    return FrameRange(start_frame=start_frame, end_frame_exclusive=FOUR_PART_FOCUSED_FRAMES)


def require_four_part_ego_focused_range(
    view_id: str, start_frame: int, max_frames: int
) -> FrameRange:
    """Permit only the focused monochrome ego four-part proxy."""
    if (
        view_id != "ego-hmc21110305"
        or start_frame != 0
        or max_frames != FOUR_PART_FOCUSED_FRAMES
    ):
        raise ValueError(
            "focused ego four-part run permits only ego-hmc21110305 proxy frames [0, 2781)"
        )
    return FrameRange(start_frame=start_frame, end_frame_exclusive=FOUR_PART_FOCUSED_FRAMES)


def checkpoint_path_for(run_directory: Path, frame_index: int) -> Path:
    return run_directory / "native" / "checkpoints" / f"f{frame_index:06d}.pt"


def prepare_resume(
    *, prior_run: Path, resume_at: int, run_directory: Path, repository_root: Path
) -> tuple[Path, ArtifactFingerprint]:
    """Copy a prior run's frames before `resume_at` so a rerun only re-steps the tail.

    The copied prefix is bit-identical to the prior run, which is the whole point: the
    tracker state saved at `resume_at` was produced by exactly those frames, so the run
    may not recompute them and may not claim to have stepped them.
    """
    checkpoint = checkpoint_path_for(prior_run, resume_at)
    if not checkpoint.is_file():
        raise ValueError(
            f"{prior_run} has no tracker checkpoint for frame {resume_at}; rerun it with "
            "--checkpoint-at or --checkpoint-every to make one"
        )
    kept: list[tuple[str, FrameObservations]] = []
    for line in (prior_run / "observations.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        observation = FrameObservations.model_validate_json(line)
        if observation.analysis_frame_index < resume_at:
            kept.append((line, observation))
    if len(kept) != resume_at:
        raise ValueError(
            f"{prior_run} holds {len(kept)} rows before frame {resume_at}; a resume needs "
            "one row for every earlier frame"
        )
    run_directory.mkdir(parents=True, exist_ok=True)
    (run_directory / "masks").mkdir(exist_ok=True)
    with (run_directory / "observations.jsonl").open("w") as output:
        for line, observation in kept:
            for item in observation.objects:
                if item.mask is None:
                    continue
                source = prior_run / item.mask.uri
                if not source.is_file():
                    raise FileNotFoundError(f"prior run mask is unavailable: {source}")
                destination = run_directory / item.mask.uri
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
            # The prior line is written back verbatim: the prefix must be the prior run's
            # bytes, not a re-serialization that happens to validate the same way.
            output.write(line + "\n")
    return checkpoint, ArtifactFingerprint(
        uri=relative_uri(checkpoint, repository_root),
        sha256=sha256_file(checkpoint),
        source="measured",
    )


def load_observations(path: Path) -> tuple[FrameObservations, ...]:
    """Validate streaming worker records one line at a time."""
    observations: list[FrameObservations] = []
    with path.open() as file:
        for line_number, line in enumerate(file, start=1):
            if line.strip():
                try:
                    observations.append(FrameObservations.model_validate_json(line))
                except ValueError as error:
                    raise ValueError(
                        f"invalid observation at {path}:{line_number}: {error}"
                    ) from error
    return tuple(observations)


def _external_revision() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(MUGGLED_SAM_SOURCE), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _run_worker(
    *,
    external_python: Path,
    worker_path: Path,
    run_directory: Path,
    proxy_path: Path,
    view_id: str,
    source_offset_seconds: float,
    model_path: Path,
    max_frames: int,
    max_side_length: int,
    max_frame_memory: int,
    analysis_fps: float,
    condition: MuggledSAMEgoCondition | None = None,
    text_targets: list[dict[str, str]] | None = None,
    hybrid_initialization: dict[str, Any] | None = None,
    manual_seeds: dict[str, Any] | None = None,
    multi_keyframe_schedule: dict[str, Any] | None = None,
    resume_from_checkpoint: Path | None = None,
    checkpoint_every: int = 0,
) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    source_paths = [str(MUGGLED_SAM_SOURCE)]
    if existing_pythonpath := environment.get("PYTHONPATH"):
        source_paths.append(existing_pythonpath)
    environment["PYTHONPATH"] = os.pathsep.join(source_paths)
    command = [
        str(external_python),
        str(worker_path),
        "--run-directory",
        str(run_directory),
        "--video",
        str(proxy_path),
        "--view-id",
        view_id,
        "--source-offset-seconds",
        str(source_offset_seconds),
        "--model",
        str(model_path),
        "--max-frames",
        str(max_frames),
        "--max-side-length",
        str(max_side_length),
        "--max-frame-memory",
        str(max_frame_memory),
        "--analysis-fps",
        str(analysis_fps),
        "--checkpoint-every",
        str(checkpoint_every),
    ]
    if resume_from_checkpoint is not None:
        command.extend(["--resume-from-checkpoint", str(resume_from_checkpoint)])
    if condition is not None:
        command.extend(
            [
                "--concepts-json",
                json.dumps(condition.concepts),
                "--preprocessing",
                condition.preprocessing.mode,
                "--prompt-mode",
                condition.prompt_mode,
                "--condition-input-video",
                str(run_directory / "condition_input_300f.mp4"),
            ]
        )
        if condition.preprocessing.mode == "gray_p01_p99_clahe":
            command.extend(
                [
                    "--lower-percentile",
                    str(condition.preprocessing.lower_percentile),
                    "--upper-percentile",
                    str(condition.preprocessing.upper_percentile),
                    "--clahe-clip-limit",
                    str(condition.preprocessing.clahe_clip_limit),
                    "--clahe-tile-grid-size",
                    str(condition.preprocessing.clahe_tile_grid_size),
                ]
            )
        if condition.manual_box_seed is not None:
            command.extend(
                [
                    "--manual-box-json",
                    condition.manual_box_seed.model_dump_json(),
                ]
            )
    if text_targets is not None:
        command.extend(
            [
                "--concepts-json",
                json.dumps([target["output_label"] for target in text_targets]),
                "--text-targets-json",
                json.dumps(text_targets, sort_keys=True),
                "--prompt-mode",
                "text_detection",
            ]
        )
    if hybrid_initialization is not None:
        command.extend(
            [
                "--concepts-json",
                json.dumps([target["output_label"] for target in hybrid_initialization["targets"]]),
                "--hybrid-initialization-json",
                json.dumps(hybrid_initialization, sort_keys=True),
                "--prompt-mode",
                "hybrid_text_and_manual_mask",
            ]
        )
    if manual_seeds is not None:
        command.extend(
            [
                "--manual-seeds-json",
                json.dumps(manual_seeds, sort_keys=True),
                "--concepts-json",
                json.dumps([seed["target"] for seed in manual_seeds["seeds"]]),
                "--prompt-mode",
                "manual_seed_multiplexed",
            ]
        )
    if multi_keyframe_schedule is not None:
        command.extend(
            [
                "--multi-keyframe-schedule-json",
                json.dumps(multi_keyframe_schedule, sort_keys=True),
                "--concepts-json",
                json.dumps([seed["target"] for seed in multi_keyframe_schedule["seeds"]]),
                "--prompt-mode",
                "manual_seed_multiplexed_keyframes",
            ]
        )
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
            "time_to_first_usable_output_seconds": None,
            "gpu_peak_vram_bytes": None,
            "masks_written": 0,
            "known_unavailable_measures": [
                "time_to_first_usable_output_seconds",
                "gpu_peak_vram_bytes",
            ],
            "runtime_settings": {},
        }
    result = json.loads(result_path.read_text())
    result["worker_exit_code"] = completed.returncode
    return result


def _require_transferable_frame_zero_calibration(
    *,
    proposal: MuggledSAMProposedTrackingPromptConfig,
    calibration: MuggledSAMBoxCalibrationManifest,
    repository_root: Path,
    selected_config_path: Path,
    proxy: Any,
) -> str:
    """Permit frame-zero seeds on another proxy of the same moment, and say so in the manifest.

    A frame-zero mask is a set of pixels on one decoded frame. It stays valid on a different
    proxy only if that proxy's frame zero is the same source instant at the same geometry,
    which is exactly what distinguishes a re-encoded or re-sampled proxy of the approved
    interval from genuinely different footage. Later-frame seeds are never transferable,
    because their frame index means different instants on different clocks.
    """
    authored_config_path = (repository_root / calibration.base_g2_config).resolve()
    if (
        not authored_config_path.is_file()
        or sha256_file(authored_config_path) != calibration.base_g2_config_sha256
    ):
        raise ValueError("manual-seed calibration must match the selected G2 configuration")
    authored = G2PreprocessingManifest.model_validate_json(authored_config_path.read_text())
    selected = G2PreprocessingManifest.model_validate_json(selected_config_path.read_text())
    authored_proxy = next((p for p in authored.proxies if p.view_id == proxy.view_id), None)
    if authored_proxy is None:
        raise ValueError("manual-seed calibration was not authored against the selected view")

    non_zero = [
        seed.reference_frame.analysis_frame_index
        for seed in proposal.seeds
        if seed.reference_frame.analysis_frame_index != 0
    ]
    if non_zero:
        raise ValueError(
            "only frame-zero seeds transfer between proxies, because a later frame index names "
            f"a different instant on a different clock; received frames {sorted(set(non_zero))}"
        )
    mismatches = []
    if authored_proxy.raw_source.checksum_sha256 != proxy.raw_source.checksum_sha256:
        mismatches.append("raw recording")
    if authored.source_interval.start_seconds != selected.source_interval.start_seconds:
        mismatches.append("source interval start")
    if (authored_proxy.dimensions.width, authored_proxy.dimensions.height) != (
        proxy.dimensions.width,
        proxy.dimensions.height,
    ):
        mismatches.append("proxy dimensions")
    if authored.scaling_policy != selected.scaling_policy:
        mismatches.append("scaling policy")
    if mismatches:
        raise ValueError(
            "manual-seed calibration must match the selected G2 configuration; frame-zero "
            f"transfer is refused because these differ: {', '.join(mismatches)}"
        )
    return (
        f"frame-zero seeds authored against {calibration.base_g2_config} at "
        f"{authored_proxy.fps:g} fps, transferred to a {proxy.fps:g} fps proxy of the same "
        "recording, interval start, dimensions, and scaling policy"
    )


def _load_manual_seed_multiplex(
    *,
    proposal_path: Path,
    repository_root: Path,
    config_path: Path,
    proxy: Any,
    manual_seed_target_config_path: Path | None = None,
) -> tuple[dict[str, Any], ManualSeedMultiplexMetadata]:
    """Validate the approved proposal and construct a policy-sized worker payload."""
    proposal = MuggledSAMProposedTrackingPromptConfig.model_validate_json(proposal_path.read_text())
    calibration_path = (repository_root / proposal.calibration_manifest_uri).resolve()
    if not calibration_path.is_file():
        raise ValueError(f"calibration manifest is unavailable: {calibration_path}")
    calibration_sha256 = sha256_file(calibration_path)
    if proposal.calibration_manifest_sha256 != calibration_sha256:
        raise ValueError("proposal calibration-manifest SHA-256 does not match")
    target_config_path = manual_seed_target_config_path
    if proposal.manual_seed_target_config_fingerprint is not None:
        proposal_target_config_path = (
            repository_root / proposal.manual_seed_target_config_fingerprint.uri
        ).resolve()
        if target_config_path is None:
            target_config_path = proposal_target_config_path
        elif target_config_path.resolve() != proposal_target_config_path:
            raise ValueError("manual-seed target config must match the proposal's target config")
        if (
            not proposal_target_config_path.is_file()
            or sha256_file(proposal_target_config_path)
            != proposal.manual_seed_target_config_fingerprint.sha256
        ):
            raise ValueError("proposal manual-seed target config is unavailable or has changed")
    calibration = MuggledSAMBoxCalibrationManifest.model_validate_json(calibration_path.read_text())
    if proposal.view_id != proxy.view_id or calibration.view_id != proxy.view_id:
        raise ValueError("manual-seed proposal and calibration must name the selected view")
    calibration_transfer_note: str | None = None
    if calibration.base_g2_config != relative_uri(
        config_path, repository_root
    ) or calibration.base_g2_config_sha256 != sha256_file(config_path):
        calibration_transfer_note = _require_transferable_frame_zero_calibration(
            proposal=proposal,
            calibration=calibration,
            repository_root=repository_root,
            selected_config_path=config_path,
            proxy=proxy,
        )
    if (
        calibration.source.uri != proxy.raw_source.raw_uri
        or calibration.source.sha256 != proxy.raw_source.checksum_sha256
    ):
        raise ValueError("manual-seed calibration source fingerprint does not match G2")
    proxy_changed = (
        calibration.proxy.uri != proxy.proxy_uri
        or calibration.proxy.sha256 != proxy.checksum_sha256
    )
    if proxy_changed and calibration_transfer_note is None:
        raise ValueError("manual-seed calibration proxy fingerprint does not match G2")

    target_config = (
        load_manual_seed_target_config(
            target_config_path=target_config_path,
            repository_root=repository_root,
            g2_config_path=config_path,
            view_id=proxy.view_id,
            also_permitted_g2_config_path=(
                (repository_root / calibration.base_g2_config).resolve()
                if calibration_transfer_note is not None
                else None
            ),
        )
        if target_config_path is not None
        else None
    )
    if target_config is None:
        received_seeds = tuple(
            (seed.intended_target, seed.candidate_id, seed.human_selected_candidate_index)
            for seed in proposal.seeds
        )
        if received_seeds != MANUAL_SEED_TARGETS:
            raise ValueError(
                "manual-seed proposal must contain exactly left_hand/t000000-b01/mask-0, "
                "yellow_toy_body/t000000-b03/mask-0, and toy_wheel/t000000-b04/mask-0"
            )
        if any(seed.candidate_id == "t000000-b18" for seed in proposal.seeds):
            raise ValueError("right_hand candidate t000000-b18 is explicitly excluded")
        ordered_seeds = proposal.seeds
    else:
        if (
            len(proposal.seeds) != len(target_config.targets)
            or len({seed.candidate_id for seed in proposal.seeds}) != len(proposal.seeds)
            or {seed.intended_target for seed in proposal.seeds} != set(target_config.targets)
        ):
            raise ValueError(
                "manual-seed proposal must contain exactly "
                f"{len(target_config.targets)} distinct human-selected frame-0 masks for: "
                f"{', '.join(target_config.targets)}"
            )
        seeds_by_target = {seed.intended_target: seed for seed in proposal.seeds}
        ordered_seeds = tuple(seeds_by_target[target] for target in target_config.targets)

    candidates = {candidate.candidate_id: candidate for candidate in calibration.candidates}
    worker_seeds: list[dict[str, Any]] = []
    provenance: list[ManualSeedCandidateProvenance] = []
    selected_mask_uris: set[str] = set()
    descriptors_by_target = (
        {descriptor.target_id: descriptor for descriptor in target_config.target_descriptors}
        if target_config is not None
        else {}
    )
    for slot, proposal_seed in enumerate(ordered_seeds):
        candidate = candidates.get(proposal_seed.candidate_id)
        if candidate is None:
            raise ValueError(
                f"proposal references missing calibration candidate {proposal_seed.candidate_id}"
            )
        if (
            candidate.intended_target != proposal_seed.intended_target
            or candidate.frame != proposal_seed.reference_frame
            or candidate.pixel_box != proposal_seed.pixel_box
            or candidate.normalized_box != proposal_seed.normalized_box
            or candidate.human_selected_candidate_index
            != proposal_seed.human_selected_candidate_index
            or not candidate.human_accepted
            or not candidate.selected_for_finalization
            or candidate.frame.analysis_frame_index != 0
        ):
            raise ValueError(
                f"proposal seed does not match human-finalized calibration candidate "
                f"{proposal_seed.candidate_id}"
            )
        selected = next(
            (
                item
                for item in candidate.decoder_result.candidates
                if item.candidate_index == proposal_seed.human_selected_candidate_index
            ),
            None,
        )
        if selected is None:
            raise ValueError(f"selected mask is absent from {proposal_seed.candidate_id}")
        if selected.mask_uri in selected_mask_uris:
            raise ValueError("manual-seed proposal must select distinct mask artifacts")
        selected_mask_uris.add(selected.mask_uri)
        mask_path = (calibration_path.parent / selected.mask_uri).resolve()
        if not mask_path.is_file():
            raise ValueError(f"selected calibration mask is unavailable: {mask_path}")
        mask_uri = relative_uri(mask_path, repository_root)
        mask_sha256 = sha256_file(mask_path)
        worker_seeds.append(
            {
                "candidate_id": proposal_seed.candidate_id,
                "target": proposal_seed.intended_target,
                "human_selected_candidate_index": proposal_seed.human_selected_candidate_index,
                "mask_path": str(mask_path),
                "mask_sha256": mask_sha256,
                "initial_multiplex_slot": slot,
            }
        )
        provenance.append(
            ManualSeedCandidateProvenance(
                candidate_id=proposal_seed.candidate_id,
                intended_target=proposal_seed.intended_target,
                human_selected_candidate_index=proposal_seed.human_selected_candidate_index,
                calibration_mask_fingerprint=ArtifactFingerprint(
                    uri=mask_uri, sha256=mask_sha256, source="measured"
                ),
                initial_multiplex_slot=slot,
                display_alias=(
                    descriptors_by_target[proposal_seed.intended_target].display_alias
                    if proposal_seed.intended_target in descriptors_by_target
                    else None
                ),
                mask_semantics=(
                    descriptors_by_target[proposal_seed.intended_target].mask_semantics
                    if proposal_seed.intended_target in descriptors_by_target
                    else None
                ),
            )
        )
    return (
        {"seeds": worker_seeds},
        ManualSeedMultiplexMetadata(
            proposal_fingerprint=ArtifactFingerprint(
                uri=relative_uri(proposal_path, repository_root),
                sha256=sha256_file(proposal_path),
                source="measured",
            ),
            calibration_manifest_fingerprint=ArtifactFingerprint(
                uri=relative_uri(calibration_path, repository_root),
                sha256=calibration_sha256,
                source="measured",
            ),
            seeds=tuple(provenance),
            initialization_api="encode_prompt_memory_from_mask",
            excluded_candidate_ids=("t000000-b18",) if target_config is None else (),
            calibration_transfer_note=calibration_transfer_note,
        ),
    )


def _require_fingerprint(repository_root: Path, fingerprint: ArtifactFingerprint) -> Path:
    """Resolve and verify one content-addressed hybrid configuration input."""
    path = (repository_root / fingerprint.uri).resolve()
    if not path.is_file() or sha256_file(path) != fingerprint.sha256:
        raise ValueError(f"hybrid input is unavailable or changed: {fingerprint.uri}")
    return path


def _load_hybrid_initialization(
    *,
    hybrid_config_path: Path,
    repository_root: Path,
    config_path: Path,
    proxy: Any,
) -> tuple[
    dict[str, Any],
    HybridInitializationMetadata,
    MuggledSAMTextTargetConfig,
]:
    """Validate the explicit static hybrid contract and build its worker payload."""
    hybrid = MuggledSAMHybridInitializationConfig.model_validate_json(
        hybrid_config_path.read_text()
    )
    if hybrid.view_id != proxy.view_id:
        raise ValueError("hybrid config view must match the selected approved proxy")
    if (repository_root / hybrid.base_g2_config).resolve() != config_path.resolve():
        raise ValueError("hybrid config must name the selected G2 configuration")

    text_config_path = _require_fingerprint(repository_root, hybrid.text_target_config_fingerprint)
    proposal_path = _require_fingerprint(repository_root, hybrid.manual_seed_proposal_fingerprint)
    manual_target_config_path = _require_fingerprint(
        repository_root, hybrid.manual_seed_target_config_fingerprint
    )
    text_config = load_text_target_config(
        target_config_path=text_config_path,
        repository_root=repository_root,
        g2_config_path=config_path,
        view_id=proxy.view_id,
    )
    text_labels = tuple(target.output_label for target in text_config.targets)
    expected_text_labels = tuple(
        target.output_label
        for target in hybrid.targets
        if target.initialization_source == "text_prompt"
    )
    if text_labels != expected_text_labels:
        raise ValueError("hybrid text config must contain exactly the ordered text targets")

    manual_payload, manual_metadata = _load_manual_seed_multiplex(
        proposal_path=proposal_path,
        repository_root=repository_root,
        config_path=config_path,
        proxy=proxy,
        manual_seed_target_config_path=manual_target_config_path,
    )
    manual_labels = tuple(seed["target"] for seed in manual_payload["seeds"])
    expected_manual_labels = tuple(
        target.output_label
        for target in hybrid.targets
        if target.initialization_source == "human_reviewed_mask"
    )
    if manual_labels != expected_manual_labels or set(text_labels) & set(manual_labels):
        raise ValueError("hybrid text and manual targets must be exact, unique, and disjoint")

    slots_by_label = {target.output_label: slot for slot, target in enumerate(hybrid.targets)}
    worker_manual_seeds = [
        {**seed, "initial_multiplex_slot": slots_by_label[seed["target"]]}
        for seed in manual_payload["seeds"]
    ]
    manual_by_label = {seed.intended_target: seed for seed in manual_metadata.seeds}
    target_provenance: list[HybridTargetInitializationProvenance] = []
    for slot, target in enumerate(hybrid.targets):
        if target.initialization_source == "text_prompt":
            configured = next(
                item for item in text_config.targets if item.output_label == target.output_label
            )
            target_provenance.append(
                HybridTargetInitializationProvenance(
                    output_label=target.output_label,
                    initial_multiplex_slot=slot,
                    initialization_source="text_prompt",
                    source_fingerprint=hybrid.text_target_config_fingerprint,
                    text_prompt=configured.text_prompt,
                )
            )
        else:
            seed = manual_by_label[target.output_label]
            target_provenance.append(
                HybridTargetInitializationProvenance(
                    output_label=target.output_label,
                    initial_multiplex_slot=slot,
                    initialization_source="human_reviewed_mask",
                    source_fingerprint=seed.calibration_mask_fingerprint,
                    candidate_id=seed.candidate_id,
                    human_selected_candidate_index=seed.human_selected_candidate_index,
                )
            )

    metadata = HybridInitializationMetadata(
        contract_fingerprint=ArtifactFingerprint(
            uri=relative_uri(hybrid_config_path, repository_root),
            sha256=sha256_file(hybrid_config_path),
            source="measured",
        ),
        text_target_config_fingerprint=hybrid.text_target_config_fingerprint,
        manual_seed_proposal_fingerprint=hybrid.manual_seed_proposal_fingerprint,
        manual_seed_target_config_fingerprint=(hybrid.manual_seed_target_config_fingerprint),
        calibration_manifest_fingerprint=(manual_metadata.calibration_manifest_fingerprint),
        targets=tuple(target_provenance),
        initialization_api="encode_prompt_memory_from_mask",
        method_label="hybrid_text_and_human_reviewed_mask",
    )
    payload = {
        "targets": [
            {
                "output_label": target.output_label,
                "initialization_source": target.initialization_source,
                "initial_multiplex_slot": slot,
            }
            for slot, target in enumerate(hybrid.targets)
        ],
        "text_targets": [target.model_dump(mode="json") for target in text_config.targets],
        "manual_seeds": worker_manual_seeds,
    }
    return payload, metadata, text_config


def _load_hybrid_smoke_approval(
    *,
    approved_smoke_manifest_path: Path,
    repository_root: Path,
    hybrid_metadata: HybridInitializationMetadata,
    config_path: Path,
    proxy: Any,
    max_side_length: int,
    max_frame_memory: int,
    approved_at: datetime,
    approved_by: str,
    approval_statement: str,
) -> HybridSmokeHumanApproval:
    """Bind a full run to the exact technically valid smoke evidence the human reviewed."""
    if not approved_smoke_manifest_path.is_file():
        raise ValueError("approved hybrid smoke manifest is unavailable")
    approved_run = RunManifest.model_validate_json(approved_smoke_manifest_path.read_text())
    smoke = approved_run.smoke
    if smoke is None or smoke.hybrid_initialization is None:
        raise ValueError("approved smoke must declare hybrid initialization")
    if (
        smoke.requested_analysis_frame_range
        != FrameRange(start_frame=0, end_frame_exclusive=SMOKE_FRAMES)
        or smoke.requested_seconds != SMOKE_SECONDS
        or smoke.concepts != ("left_hand", "right_hand", "yellow_toy_top", "black_toy_top_base")
    ):
        raise ValueError("approved smoke does not use the exact bounded aligned target contract")
    if (
        smoke.hybrid_initialization.contract_fingerprint != hybrid_metadata.contract_fingerprint
        or tuple(
            (
                target.output_label,
                target.initial_multiplex_slot,
                target.initialization_source,
                target.initialized_at_frame_zero,
            )
            for target in smoke.hybrid_initialization.targets
        )
        != (
            ("left_hand", 0, "text_prompt", True),
            ("right_hand", 1, "text_prompt", True),
            ("yellow_toy_top", 2, "text_prompt", True),
            ("black_toy_top_base", 3, "human_reviewed_mask", True),
        )
        or smoke.hybrid_initialization.targets[3].candidate_id != "t000000-b03"
        or smoke.hybrid_initialization.ground_truth_accuracy_claim is not False
    ):
        raise ValueError("approved smoke hybrid provenance differs from the requested full run")
    if (
        smoke.config_fingerprint.uri != relative_uri(config_path, repository_root)
        or smoke.config_fingerprint.sha256 != sha256_file(config_path)
        or smoke.proxy_fingerprint.uri != proxy.proxy_uri
        or smoke.proxy_fingerprint.sha256 != proxy.checksum_sha256
        or smoke.source_fingerprint.uri != proxy.raw_source.raw_uri
        or smoke.source_fingerprint.sha256 != proxy.raw_source.checksum_sha256
    ):
        raise ValueError("approved smoke source/config/proxy provenance differs from G2")
    if (
        smoke.runtime_settings.get("analysis_fps") != 30.0
        or smoke.runtime_settings.get("max_side_length") != max_side_length
        or smoke.runtime_settings.get("max_frame_memory") != max_frame_memory
    ):
        raise ValueError("full hybrid settings must exactly match the approved smoke")
    succeeded_objects = [
        status
        for status in approved_run.method_statuses
        if status.stage == "objects" and status.state is MethodState.SUCCEEDED
    ]
    if len(succeeded_objects) != 1 or smoke.qa_artifact_uri is None:
        raise ValueError("approved smoke must have succeeded output and persisted QA evidence")
    qa_path = (repository_root / smoke.qa_artifact_uri).resolve()
    if not qa_path.is_file():
        raise ValueError("approved smoke QA evidence is unavailable")
    return HybridSmokeHumanApproval(
        approved_smoke_manifest_fingerprint=ArtifactFingerprint(
            uri=relative_uri(approved_smoke_manifest_path, repository_root),
            sha256=sha256_file(approved_smoke_manifest_path),
            source="measured",
        ),
        approved_smoke_qa_fingerprint=ArtifactFingerprint(
            uri=relative_uri(qa_path, repository_root),
            sha256=sha256_file(qa_path),
            source="measured",
        ),
        approved_at=approved_at,
        approved_by=approved_by,
        approval_statement=approval_statement,
    )


def _resolve_artifact_uri(repository_root: Path, uri: str) -> Path:
    path = Path(uri)
    return path if path.is_absolute() else (repository_root / path).resolve()


def _require_correction_frame_in_range(frame_index: int, max_frame_exclusive: int) -> None:
    if not 0 <= frame_index < max_frame_exclusive:
        raise ValueError(
            f"correction schedule permits only frames [0, {max_frame_exclusive}): {frame_index}"
        )


def _load_multi_keyframe_correction_schedule(
    *,
    schedule_path: Path,
    repository_root: Path,
    config_path: Path,
    proxy: Any,
    analysis_fps: float,
    max_frame_exclusive: int = SMOKE_FRAMES,
) -> tuple[dict[str, Any], MultiKeyframeCorrectionScheduleMetadata]:
    """Validate a proposed schedule and prepare its frame-zero and correction mask payload."""
    if analysis_fps != SCHEDULE_AUTHORING_FPS:
        raise ValueError(
            "correction schedules store analysis frame indices, which are only meaningful on the "
            f"{SCHEDULE_AUTHORING_FPS:g} fps clock they were authored against; reinterpreting them "
            f"at {analysis_fps:g} fps would silently move every correction to a different "
            "timestamp. Re-author the schedule against this clock instead."
        )
    schedule = MuggledSAMMultiKeyframeCorrectionSchedule.model_validate_json(
        schedule_path.read_text()
    )
    if schedule.view_id != proxy.view_id:
        raise ValueError("correction schedule must name the selected view")
    if schedule.correction_memory_semantics != "replace_prompt_memory_and_reset_frame_memory":
        raise ValueError("unsupported correction schedule memory semantics")

    calibration_path = _resolve_artifact_uri(
        repository_root, schedule.calibration_manifest_fingerprint.uri
    )
    if (
        not calibration_path.is_file()
        or sha256_file(calibration_path) != schedule.calibration_manifest_fingerprint.sha256
    ):
        raise ValueError("correction schedule calibration manifest is unavailable or has changed")
    calibration = MuggledSAMBoxCalibrationManifest.model_validate_json(calibration_path.read_text())
    if (
        calibration.base_g2_config != relative_uri(config_path, repository_root)
        or calibration.base_g2_config_sha256 != sha256_file(config_path)
        or calibration.view_id != proxy.view_id
        or calibration.proxy.uri != proxy.proxy_uri
        or calibration.proxy.sha256 != proxy.checksum_sha256
    ):
        raise ValueError("correction schedule calibration does not match the selected G2 proxy")

    target_config_path = _resolve_artifact_uri(
        repository_root, schedule.manual_seed_target_config_fingerprint.uri
    )
    if (
        not target_config_path.is_file()
        or sha256_file(target_config_path) != schedule.manual_seed_target_config_fingerprint.sha256
    ):
        raise ValueError("correction schedule target configuration is unavailable or has changed")
    target_config = load_manual_seed_target_config(
        target_config_path=target_config_path,
        repository_root=repository_root,
        g2_config_path=config_path,
        view_id=proxy.view_id,
    )

    policy_path = _resolve_artifact_uri(repository_root, schedule.correction_policy_fingerprint.uri)
    if (
        not policy_path.is_file()
        or sha256_file(policy_path) != schedule.correction_policy_fingerprint.sha256
    ):
        raise ValueError("correction schedule policy is unavailable or has changed")
    policy = MuggledSAMMultiKeyframeCorrectionPolicy.model_validate_json(policy_path.read_text())
    if (
        policy.view_id != proxy.view_id
        or policy.targets != target_config.targets
        or policy.correction_memory_semantics != schedule.correction_memory_semantics
        or policy.manual_seed_target_config_fingerprint.sha256
        != schedule.manual_seed_target_config_fingerprint.sha256
    ):
        raise ValueError("correction schedule policy does not match its target configuration")

    slots = tuple(sorted(schedule.slots, key=lambda slot: slot.multiplex_slot))
    if tuple(slot.target_id for slot in slots) != target_config.targets:
        raise ValueError("correction schedule slot order must match the named target configuration")
    candidates = {candidate.candidate_id: candidate for candidate in calibration.candidates}
    payload_corrections: list[dict[str, Any]] = []
    later_per_target: dict[str, int] = {}
    for correction in schedule.corrections:
        _require_correction_frame_in_range(
            correction.frame.analysis_frame_index, max_frame_exclusive
        )
        candidate = candidates.get(correction.candidate_id)
        if candidate is None:
            raise ValueError(
                "correction schedule references missing calibration candidate "
                f"{correction.candidate_id}"
            )
        if (
            candidate.intended_target != correction.target_id
            or candidate.frame != correction.frame
            or candidate.human_selected_candidate_index != correction.human_selected_candidate_index
            or candidate.selected_by != correction.selected_by
            or not candidate.human_accepted
            or (
                candidate.frame.analysis_frame_index == 0
                and not candidate.selected_for_finalization
            )
            or (candidate.frame.analysis_frame_index != 0 and not candidate.selected_for_correction)
        ):
            raise ValueError(
                "correction schedule entry does not match an eligible recorded selection: "
                f"{correction.candidate_id}"
            )
        selected = next(
            (
                mask
                for mask in candidate.decoder_result.candidates
                if mask.candidate_index == correction.human_selected_candidate_index
            ),
            None,
        )
        if selected is None:
            raise ValueError(f"selected correction mask is absent: {correction.candidate_id}")
        mask_path = _resolve_artifact_uri(
            repository_root, correction.calibration_mask_fingerprint.uri
        )
        expected_mask_path = (calibration_path.parent / selected.mask_uri).resolve()
        if (
            mask_path != expected_mask_path
            or not mask_path.is_file()
            or sha256_file(mask_path) != correction.calibration_mask_fingerprint.sha256
        ):
            raise ValueError(f"correction schedule mask is unavailable or has changed: {mask_path}")
        if correction.frame.analysis_frame_index:
            later_per_target[correction.target_id] = (
                later_per_target.get(correction.target_id, 0) + 1
            )
        payload_corrections.append(
            {
                "candidate_id": correction.candidate_id,
                "target": correction.target_id,
                "object_id": correction.object_id,
                "multiplex_slot": correction.multiplex_slot,
                "frame_index": correction.frame.analysis_frame_index,
                "mask_path": str(mask_path),
                "mask_sha256": correction.calibration_mask_fingerprint.sha256,
                "selected_by": correction.selected_by,
            }
        )
    if any(
        count > policy.maximum_later_correction_keyframes_per_target
        for count in later_per_target.values()
    ):
        raise ValueError("correction schedule exceeds its policy's per-target keyframe limit")
    initial = [item for item in payload_corrections if item["frame_index"] == 0]
    initial.sort(key=lambda item: int(item["multiplex_slot"]))
    _targets = [item["target"] for item in initial]
    if _targets != list(target_config.targets):
        raise ValueError("correction schedule needs one ordered frame-0 mask for every target")
    later = [item for item in payload_corrections if item["frame_index"] != 0]
    later.sort(key=lambda item: (int(item["frame_index"]), int(item["multiplex_slot"])))
    for slot, item in enumerate(initial):
        item["initial_multiplex_slot"] = slot
    return (
        {
            "seeds": initial,
            "corrections": later,
            "memory_semantics": schedule.correction_memory_semantics,
        },
        MultiKeyframeCorrectionScheduleMetadata(
            schedule_fingerprint=ArtifactFingerprint(
                uri=relative_uri(schedule_path, repository_root),
                sha256=sha256_file(schedule_path),
                source="measured",
            ),
            correction_policy_fingerprint=schedule.correction_policy_fingerprint,
            correction_memory_semantics=schedule.correction_memory_semantics,
            scheduled_correction_frame_indices=tuple(
                sorted({int(item["frame_index"]) for item in later})
            ),
            agent_selected_correction_frame_indices=tuple(
                sorted(
                    {int(item["frame_index"]) for item in later if item["selected_by"] == "agent"}
                )
            ),
        ),
    )


def _write_mask_cache_sidecar(run_directory: Path) -> None:
    """Pack this run's masks so review builders skip re-decoding every PNG.

    This runs after inference on the CPU and is never load-bearing: a run that cannot
    write its cache is still complete, and builders fall back to the PNGs.
    """
    try:
        mask_cache.write_sidecar(run_directory, mask_cache.logged_colors_by_uri(run_directory))
    except Exception as error:  # noqa: BLE001 - a cache is an optimisation, not a result
        print(f"mask cache sidecar skipped: {type(error).__name__}: {error}")


def _bounded_video_stamp_path(output_path: Path) -> Path:
    return output_path.with_suffix(f"{output_path.suffix}.source.json")


def _reusable_bounded_video(proxy_path: Path, output_path: Path, frame_count: int) -> bool:
    """Report whether a previously trimmed asset still describes this exact request.

    The stamp records the source, its size and mtime, and the requested frame count, so
    a re-encode is skipped only when the same proxy is trimmed the same way. Reading the
    frame count back with ffprobe would cost a full decode of the file it is checking.
    """
    stamp_path = _bounded_video_stamp_path(output_path)
    if not output_path.is_file() or not stamp_path.is_file():
        return False
    try:
        stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    status = proxy_path.stat()
    return stamp == {
        "source": str(proxy_path.resolve()),
        "source_size_bytes": status.st_size,
        "source_mtime_ns": status.st_mtime_ns,
        "frame_count": frame_count,
    }


def _create_bounded_rerun_video(
    *, proxy_path: Path, output_path: Path, run_directory: Path, frame_count: int
) -> Path:
    """Create the exact approved-range video asset embedded once in a Rerun recording."""
    if _reusable_bounded_video(proxy_path, output_path, frame_count):
        return output_path
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(proxy_path),
        "-map",
        "0:v:0",
        "-frames:v",
        str(frame_count),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    (run_directory / "rerun_video.stdout.log").write_text(completed.stdout)
    (run_directory / "rerun_video.stderr.log").write_text(completed.stderr)
    if completed.returncode != 0 or not output_path.is_file():
        raise RuntimeError(
            f"could not make the bounded {frame_count}-frame Rerun input video "
            f"(ffmpeg exit {completed.returncode})"
        )
    status = proxy_path.stat()
    _bounded_video_stamp_path(output_path).write_text(
        json.dumps(
            {
                "source": str(proxy_path.resolve()),
                "source_size_bytes": status.st_size,
                "source_mtime_ns": status.st_mtime_ns,
                "frame_count": frame_count,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return output_path


def _flat_runtime_settings(settings: dict[str, Any]) -> dict[str, str | int | float | bool | None]:
    """Keep nested worker provenance in schema-compatible JSON strings."""
    return {
        key: value
        if isinstance(value, str | int | float | bool) or value is None
        else json.dumps(value, sort_keys=True)
        for key, value in settings.items()
    }


def _missing_ranges(present: list[bool]) -> list[dict[str, int]]:
    """Return half-open missing-output runs for one ordered analysis stream."""
    gaps: list[dict[str, int]] = []
    start: int | None = None
    for frame_index, is_present in enumerate(present):
        if not is_present and start is None:
            start = frame_index
        elif is_present and start is not None:
            gaps.append({"start_frame": start, "end_frame_exclusive": frame_index})
            start = None
    if start is not None:
        gaps.append({"start_frame": start, "end_frame_exclusive": len(present)})
    return gaps


def _manual_seed_metrics(
    *,
    observations: tuple[FrameObservations, ...],
    worker_result: dict[str, Any],
    manual_seed_metadata: ManualSeedMultiplexMetadata,
    source_fingerprint: ArtifactFingerprint,
    proxy_fingerprint: ArtifactFingerprint,
    frames_requested: int,
    requested_seconds: float,
    analysis_fps: float,
) -> dict[str, Any]:
    """Summarize emissions and identity facts without inferring target accuracy."""
    targets = [seed.intended_target for seed in manual_seed_metadata.seeds]
    by_target: dict[str, Any] = {}
    for seed in manual_seed_metadata.seeds:
        observed = [
            next(
                (
                    object_
                    for object_ in observation.objects
                    if object_.label == seed.intended_target
                ),
                None,
            )
            for observation in observations
        ]
        present = [object_ is not None for object_ in observed]
        initial_ids = {object_.object_id for object_ in observed[:1] if object_ is not None}
        all_ids = {object_.object_id for object_ in observed if object_ is not None}
        mask_frames = [
            frame_index for frame_index, object_ in enumerate(observed) if object_ and object_.mask
        ]
        by_target[seed.intended_target] = {
            "initial_multiplex_slot": seed.initial_multiplex_slot,
            "normalized_object_id": f"sam3-{seed.initial_multiplex_slot:02d}",
            "emitted_frame_count": sum(present),
            "emission_coverage_fraction": sum(present) / len(observations) if observations else 0.0,
            "output_gaps": _missing_ranges(present),
            "unique_normalized_object_ids": sorted(all_ids),
            "id_restarts_after_initialization": len(all_ids - initial_ids),
            "external_mask_frames": mask_frames,
            "external_mask_count": len(mask_frames),
        }
    overall_present = [bool(observation.objects) for observation in observations]
    all_mask_frames = sorted(
        {
            frame_index
            for frame_index, observation in enumerate(observations)
            if any(object_.mask for object_ in observation.objects)
        }
    )
    mask_period_frames = int(worker_result.get("runtime_settings", {}).get("mask_period_frames", 6))
    return {
        "schema_version": "1.0",
        "measurement_scope": (
            "manual-seed multiplexed SAM3 output/continuity measures over the fixed "
            f"{requested_seconds:.1f}-second proxy; not ground-truth accuracy"
        ),
        "run_label": "manual-seed multiplexed; not out-of-box/text zero-shot",
        "frames_requested": frames_requested,
        "frames_processed": len(observations),
        "analysis_fps": analysis_fps,
        "source_fingerprint": source_fingerprint.model_dump(mode="json"),
        "proxy_fingerprint": proxy_fingerprint.model_dump(mode="json"),
        "proposal_fingerprint": manual_seed_metadata.proposal_fingerprint.model_dump(mode="json"),
        "calibration_manifest_fingerprint": (
            manual_seed_metadata.calibration_manifest_fingerprint.model_dump(mode="json")
        ),
        "seed_provenance": [seed.model_dump(mode="json") for seed in manual_seed_metadata.seeds],
        "explicitly_excluded_candidate_ids": list(manual_seed_metadata.excluded_candidate_ids),
        "initialization_api": manual_seed_metadata.initialization_api,
        "overall_output_coverage_fraction": (
            sum(overall_present) / len(observations) if observations else 0.0
        ),
        "overall_output_gaps": _missing_ranges(overall_present),
        "per_target": by_target,
        "time_to_first_usable_output_seconds": worker_result.get(
            "time_to_first_usable_output_seconds"
        ),
        "runtime_seconds": worker_result["elapsed_seconds"],
        "peak_vram_bytes": worker_result.get("gpu_peak_vram_bytes"),
        "external_mask_cadence": {
            "mask_period_frames": mask_period_frames,
            "maximum_hz": analysis_fps / mask_period_frames,
            "frames_with_any_external_mask": all_mask_frames,
            "all_masks_on_declared_cadence": all(
                frame_index % mask_period_frames == 0 for frame_index in all_mask_frames
            ),
        },
        "intentional_id_resets": False,
        "ground_truth_accuracy_claim": False,
        "targets": targets,
    }


def _render_manual_seed_comparison(
    *,
    repository_root: Path,
    run_directory: Path,
    external_python: Path,
) -> Path:
    """Render the fixed manual-seed versus preserved e4 zero-shot review sheet."""
    baseline_directory = (repository_root / E4_ZERO_SHOT_BASELINE).resolve()
    if not (baseline_directory / "observations.jsonl").is_file():
        raise FileNotFoundError(
            f"preserved e4 zero-shot baseline is unavailable: {baseline_directory}"
        )
    output_path = run_directory / "qa" / "manual_seed_multiplexed_vs_e4_zero_shot.png"
    metrics_path = run_directory / "qa" / "manual_seed_multiplexed_vs_e4_zero_shot_summary.json"
    output_path.parent.mkdir(exist_ok=True)
    command = [
        str(external_python),
        str(Path(__file__).with_name("ego_diagnostic.py")),
        "compare",
        "--repository-root",
        str(repository_root),
        "--baseline-run",
        str(baseline_directory),
        "--condition-run",
        "MANUAL-SEED MULTIPLEXED (NOT TEXT ZERO-SHOT)",
        str(run_directory),
        "--output",
        str(output_path),
        "--metrics-output",
        str(metrics_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    (run_directory / "qa.stdout.log").write_text(completed.stdout)
    (run_directory / "qa.stderr.log").write_text(completed.stderr)
    if completed.returncode != 0 or not output_path.is_file():
        raise RuntimeError(
            f"could not render manual-seed QA comparison (exit {completed.returncode})"
        )
    return output_path


def _render_full_ego_manual_seed_contact_sheet(
    *,
    repository_root: Path,
    run_directory: Path,
    external_python: Path,
) -> Path:
    """Render the fixed full-duration review sheet without invoking a model."""
    command = [
        str(external_python),
        str(Path(__file__).with_name("g3_contact_sheet.py")),
        "--run-directory",
        str(run_directory),
        "--repository-root",
        str(repository_root),
        "--timestamps",
        "0.0",
        "90.0",
        str((FULL_EGO_MANUAL_SEED_FRAMES - 1) / 30.0),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    (run_directory / "qa.stdout.log").write_text(completed.stdout)
    (run_directory / "qa.stderr.log").write_text(completed.stderr)
    output_path = (
        run_directory / "g3_review" / "ego-hmc21179183_full_ego_manual_seed_multiplexed_qa.png"
    )
    if completed.returncode != 0 or not output_path.is_file():
        raise RuntimeError(
            f"could not render full ego manual-seed QA contact sheet (exit {completed.returncode})"
        )
    return output_path


def _render_four_part_contact_sheet(
    *,
    repository_root: Path,
    run_directory: Path,
    external_python: Path,
    view_id: str,
    frame_count: int,
    duration_seconds: float,
) -> Path:
    """Render fixed beginning/middle/end evidence for a four-part run."""
    timestamps = ("0.0", str(duration_seconds / 2), str((frame_count - 1) / 30.0))
    command = [
        str(external_python),
        str(Path(__file__).with_name("g3_contact_sheet.py")),
        "--run-directory",
        str(run_directory),
        "--repository-root",
        str(repository_root),
        "--timestamps",
        *timestamps,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    (run_directory / "qa.stdout.log").write_text(completed.stdout)
    (run_directory / "qa.stderr.log").write_text(completed.stderr)
    output_path = run_directory / "g3_review" / f"{view_id}_contact_sheet.png"
    if completed.returncode != 0 or not output_path.is_file():
        raise RuntimeError(
            f"could not render four-part contact sheet (exit {completed.returncode})"
        )
    return output_path


def _render_hybrid_contact_sheet(
    *,
    repository_root: Path,
    run_directory: Path,
    external_python: Path,
    source_offset_seconds: float,
    full_run: bool,
) -> Path:
    """Render fixed hybrid semantic evidence in the OpenCV-capable model environment."""
    timestamps = (
        ("0.0", "90.0", str((G3_STATIC_FRAMES - 1) / 30.0))
        if full_run
        else ("0.0", "5.0", str((SMOKE_FRAMES - 1) / 30.0))
    )
    output_path = (
        run_directory
        / "g3_review"
        / ("static-c10379_full_hybrid_qa.png" if full_run else "static-c10379_contact_sheet.png")
    )
    command = [
        str(external_python),
        str(Path(__file__).with_name("g3_contact_sheet.py")),
        "--run-directory",
        str(run_directory),
        "--repository-root",
        str(repository_root),
        "--timestamps",
        *timestamps,
        "--source-offset-seconds",
        str(source_offset_seconds),
        "--output",
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    (run_directory / "qa.stdout.log").write_text(completed.stdout)
    (run_directory / "qa.stderr.log").write_text(completed.stderr)
    if completed.returncode != 0 or not output_path.is_file():
        raise RuntimeError(
            f"could not render hybrid QA contact sheet (exit {completed.returncode})"
        )
    return output_path


def _qa_method_name(
    *,
    hybrid: bool,
    full_ego: bool,
    four_part_pilot: bool,
    four_part_full: bool,
    four_part_static_focused: bool,
    four_part_ego_focused: bool,
) -> str:
    if hybrid:
        return "static-hybrid-semantic-gate"
    if full_ego:
        return "full-ego-manual-seed-multiplexed-qa"
    if four_part_pilot:
        return "four-part-static-pilot-qa"
    if four_part_full:
        return "four-part-static-full-exploratory-qa"
    if four_part_static_focused:
        return "four-part-static-focused-reassembly-qa"
    if four_part_ego_focused:
        return "four-part-ego-focused-reassembly-qa"
    return "manual-seed-multiplexed-e4-zero-shot-qa"


def run_smoke(args: argparse.Namespace) -> Path:
    repository_root = Path.cwd().resolve()
    config_path = args.config.resolve()
    preprocessing = G2PreprocessingManifest.model_validate_json(config_path.read_text())
    proxy = next((proxy for proxy in preprocessing.proxies if proxy.view_id == args.view), None)
    if proxy is None:
        raise ValueError(f"view {args.view!r} is not present in {config_path}")
    condition = None
    condition_config_path = None
    text_target_config = None
    text_target_config_path = None
    text_target_payload = None
    hybrid_payload = None
    hybrid_metadata = None
    hybrid_smoke_approval = None
    manual_seed_payload = None
    manual_seed_metadata = None
    multi_keyframe_schedule_payload = None
    multi_keyframe_correction_metadata = None
    if args.condition_config is not None:
        condition_config_path = args.condition_config.resolve()
        condition_config = MuggledSAMEgoConditionConfig.model_validate_json(
            condition_config_path.read_text()
        )
        if condition_config.view_id != proxy.view_id:
            raise ValueError("condition config view must match the selected approved proxy")
        if (repository_root / condition_config.base_g2_config).resolve() != config_path:
            raise ValueError("condition config must name the selected G2 configuration")
        condition = next(
            (
                item
                for item in condition_config.conditions
                if item.condition_id == args.condition_id
            ),
            None,
        )
        if condition is None:
            raise ValueError(
                f"condition {args.condition_id!r} is absent from {condition_config_path}"
            )
    elif args.condition_id is not None:
        raise ValueError("--condition-id requires --condition-config")
    analysis_fps = float(preprocessing.proxy_timing.clocks.fps_for(ClockName.ANALYSIS))
    if args.hybrid_config is not None:
        if any(
            value is not None
            for value in (
                condition,
                args.text_target_config,
                args.manual_seed_proposal,
                args.manual_seed_target_config,
                args.multi_keyframe_correction_schedule,
            )
        ):
            raise ValueError(
                "--hybrid-config is an explicit prompt mode and cannot be combined "
                "with condition, text, manual-seed, or correction flags"
            )
        (
            hybrid_payload,
            hybrid_metadata,
            text_target_config,
        ) = _load_hybrid_initialization(
            hybrid_config_path=args.hybrid_config.resolve(),
            repository_root=repository_root,
            config_path=config_path,
            proxy=proxy,
        )
        text_target_config_path = _resolve_artifact_uri(
            repository_root, hybrid_metadata.text_target_config_fingerprint.uri
        )
        text_target_payload = hybrid_payload["text_targets"]
    if args.text_target_config is not None:
        if condition is not None:
            raise ValueError("text-target config cannot be combined with a condition experiment")
        text_target_config_path = args.text_target_config.resolve()
        text_target_config = load_text_target_config(
            target_config_path=text_target_config_path,
            repository_root=repository_root,
            g2_config_path=config_path,
            view_id=proxy.view_id,
        )
        text_target_payload = [
            target.model_dump(mode="json") for target in text_target_config.targets
        ]
    if args.manual_seed_proposal is not None:
        if condition is not None or text_target_payload is not None:
            raise ValueError("manual-seed multiplexing cannot be combined with another prompt mode")
        manual_seed_payload, manual_seed_metadata = _load_manual_seed_multiplex(
            proposal_path=args.manual_seed_proposal.resolve(),
            repository_root=repository_root,
            config_path=config_path,
            proxy=proxy,
            manual_seed_target_config_path=(
                args.manual_seed_target_config.resolve()
                if args.manual_seed_target_config is not None
                else None
            ),
        )
    elif args.manual_seed_target_config is not None:
        raise ValueError("--manual-seed-target-config requires --manual-seed-proposal")
    if args.multi_keyframe_correction_schedule is not None:
        if (
            condition is not None
            or text_target_payload is not None
            or manual_seed_payload is not None
        ):
            raise ValueError(
                "a multi-keyframe correction schedule cannot be combined with another prompt mode"
            )
        (
            multi_keyframe_schedule_payload,
            multi_keyframe_correction_metadata,
        ) = _load_multi_keyframe_correction_schedule(
            schedule_path=args.multi_keyframe_correction_schedule.resolve(),
            repository_root=repository_root,
            config_path=config_path,
            proxy=proxy,
            analysis_fps=analysis_fps,
            max_frame_exclusive=selected_frame_budget(args, analysis_fps),
        )
    is_g3_candidate = args.g3_full_static
    is_e4_candidate = args.g4_e4_candidate
    is_full_ego_manual_seed = args.full_ego_manual_seed
    is_four_part_pilot = args.four_part_static_pilot
    is_four_part_full = args.four_part_static_full
    is_four_part_focused = getattr(args, "four_part_static_focused", False)
    is_four_part_ego_focused = getattr(args, "four_part_ego_focused", False)
    if (
        sum(
            (
                is_g3_candidate,
                is_e4_candidate,
                is_full_ego_manual_seed,
                is_four_part_pilot,
                is_four_part_full,
                is_four_part_focused,
                is_four_part_ego_focused,
            )
        )
        > 1
    ):
        raise ValueError("only one approved candidate profile may be selected")
    four_part_targets = (
        tuple(seed.intended_target for seed in manual_seed_metadata.seeds)
        if manual_seed_metadata is not None
        else tuple(seed["target"] for seed in multi_keyframe_schedule_payload["seeds"])
        if multi_keyframe_schedule_payload is not None
        else ()
    )
    if is_four_part_pilot and (
        condition is not None
        or text_target_payload is not None
        or hybrid_payload is not None
        or (manual_seed_payload is None and multi_keyframe_schedule_payload is None)
        or four_part_targets != ("chassis", "interior", "rear_body", "cabin")
    ):
        raise ValueError(
            "four-part pilot requires only the ordered "
            "chassis/interior/rear_body/cabin manual-seed or correction-schedule payload"
        )
    if is_four_part_full and (
        condition is not None
        or text_target_payload is not None
        or hybrid_payload is not None
        or manual_seed_payload is not None
        or multi_keyframe_schedule_payload is None
        or four_part_targets != ("chassis", "interior", "rear_body", "cabin")
    ):
        raise ValueError(
            "four-part full run requires only the ordered "
            "chassis/interior/rear_body/cabin correction schedule"
        )
    if (is_four_part_focused or is_four_part_ego_focused) and (
        condition is not None
        or text_target_payload is not None
        or hybrid_payload is not None
        or manual_seed_payload is not None
        or multi_keyframe_schedule_payload is None
        or four_part_targets != ("chassis", "interior", "rear_body", "cabin")
    ):
        raise ValueError(
            "focused four-part run requires only the ordered "
            "chassis/interior/rear_body/cabin correction schedule"
        )
    approval_values = (
        args.approved_smoke_manifest,
        args.human_approved_at,
        args.human_approved_by,
        args.human_approval_statement,
    )
    if is_g3_candidate and hybrid_metadata is not None:
        if any(value is None for value in approval_values):
            raise ValueError(
                "full hybrid G3 requires approved smoke manifest, approval time, "
                "approver, and statement"
            )
        hybrid_smoke_approval = _load_hybrid_smoke_approval(
            approved_smoke_manifest_path=args.approved_smoke_manifest.resolve(),
            repository_root=repository_root,
            hybrid_metadata=hybrid_metadata,
            config_path=config_path,
            proxy=proxy,
            max_side_length=args.max_side_length,
            max_frame_memory=args.max_frame_memory,
            approved_at=args.human_approved_at,
            approved_by=args.human_approved_by,
            approval_statement=args.human_approval_statement,
        )
    elif any(value is not None for value in approval_values):
        raise ValueError("hybrid smoke approval metadata is only valid for a full hybrid G3 run")
    if (is_g3_candidate or is_e4_candidate) and (
        condition is not None
        or manual_seed_payload is not None
        or multi_keyframe_schedule_payload is not None
    ):
        raise ValueError(
            "condition and manual-seed experiments are restricted to the 300-frame smoke budget"
        )
    if is_full_ego_manual_seed and (
        condition is not None
        or text_target_payload is not None
        or hybrid_payload is not None
        or manual_seed_payload is None
        or multi_keyframe_schedule_payload is not None
    ):
        raise ValueError(
            "full ego baseline requires the approved manual-seed multiplex payload only"
        )
    requested_frames = selected_frame_budget(args, analysis_fps)
    requested_seconds = requested_frames / analysis_fps
    requested_range = (
        require_g3_static_range(proxy.view_id, args.start_frame, args.max_frames)
        if is_g3_candidate
        else require_e4_candidate_range(proxy.view_id, args.start_frame, args.max_frames)
        if is_e4_candidate
        else require_full_ego_manual_seed_range(proxy.view_id, args.start_frame, args.max_frames)
        if is_full_ego_manual_seed
        else require_four_part_pilot_range(proxy.view_id, args.start_frame, args.max_frames)
        if is_four_part_pilot
        else require_four_part_full_range(proxy.view_id, args.start_frame, args.max_frames)
        if is_four_part_full
        else require_four_part_focused_range(proxy.view_id, args.start_frame, args.max_frames)
        if is_four_part_focused
        else require_four_part_ego_focused_range(
            proxy.view_id, args.start_frame, args.max_frames
        )
        if is_four_part_ego_focused
        else require_smoke_range(args.start_frame, args.max_frames, analysis_fps)
    )

    profile = (
        "g3-full-hybrid-static"
        if is_g3_candidate and hybrid_payload is not None
        else "g3-full"
        if is_g3_candidate
        else "g4-e4-candidate"
        if is_e4_candidate
        else "full-ego-manual-seed-multiplexed"
        if is_full_ego_manual_seed
        else "four-part-static-pilot"
        if is_four_part_pilot
        else "four-part-static-full-exploratory"
        if is_four_part_full
        else "four-part-static-focused-reassembly"
        if is_four_part_focused
        else "four-part-ego-focused-reassembly"
        if is_four_part_ego_focused
        else f"smoke-{condition.condition_id}"
        if condition
        else "smoke-manual-seed-multiplexed"
        if manual_seed_payload is not None
        else "smoke-multi-keyframe-corrections"
        if multi_keyframe_schedule_payload is not None
        else "smoke-hybrid-static"
        if hybrid_payload is not None
        else "smoke"
    )
    run_id = make_run_id(proxy.view_id, profile=profile)
    run_directory = (args.run_root / run_id).resolve()
    suffix = 2
    while run_directory.exists():
        run_directory = (args.run_root / f"{run_id}-{suffix}").resolve()
        suffix += 1
    run_directory.mkdir(parents=True)
    resume_at = 0
    resume_checkpoint: Path | None = None
    resume_continuity: dict[str, Any] = {}
    if args.resume_run is not None:
        if args.resume_at is None:
            raise ValueError("--resume-run requires --resume-at")
        resume_at = args.resume_at
        prior_run = args.resume_run.resolve()
        resume_checkpoint, checkpoint_fingerprint = prepare_resume(
            prior_run=prior_run,
            resume_at=resume_at,
            run_directory=run_directory,
            repository_root=repository_root,
        )
        resume_continuity = {
            "mode": "checkpoint_resumed",
            "resumed_from_run": relative_uri(prior_run, repository_root),
            "resumed_at_frame": resume_at,
            "checkpoint_fingerprint": checkpoint_fingerprint,
        }
    proxy_path = (repository_root / proxy.proxy_uri).resolve()
    model_path = args.model.resolve()
    worker_path = Path(__file__).with_name("muggled_worker.py")
    configured_concepts = (
        tuple(target["output_label"] for target in hybrid_payload["targets"])
        if hybrid_payload is not None
        else tuple(target.output_label for target in text_target_config.targets)
        if text_target_config is not None
        else condition.concepts
        if condition
        else tuple(seed["target"] for seed in manual_seed_payload["seeds"])
        if manual_seed_payload is not None
        else tuple(seed["target"] for seed in multi_keyframe_schedule_payload["seeds"])
        if multi_keyframe_schedule_payload is not None
        else CONCEPTS
    )
    runtime_invocation = {
        "run_id": run_directory.name,
        "config": relative_uri(config_path, repository_root),
        "view_id": proxy.view_id,
        "proxy": proxy.proxy_uri,
        "run_profile": (
            "g3_full_static_hybrid_candidate"
            if is_g3_candidate and hybrid_payload is not None
            else "g3_full_static_candidate"
            if is_g3_candidate
            else "g4_e4_60_second_candidate"
            if is_e4_candidate
            else "full_ego_manual_seed_multiplexed_baseline"
            if is_full_ego_manual_seed
            else "four_part_static_20_second_pilot"
            if is_four_part_pilot
            else "four_part_static_full_exploratory"
            if is_four_part_full
            else "four_part_static_focused_reassembly"
            if is_four_part_focused
            else "four_part_ego_focused_reassembly"
            if is_four_part_ego_focused
            else "smoke_hybrid_static"
            if hybrid_payload is not None
            else "smoke"
        ),
        "approved_range": {"start_frame": 0, "end_frame_exclusive": requested_frames},
        "approved_seconds": requested_seconds,
        "concepts": list(configured_concepts),
        "external_python": str(args.external_python),
        "muggled_sam_source": str(MUGGLED_SAM_SOURCE),
        "model_path": str(model_path),
        "cuda_visible_devices": "0",
    }
    if is_four_part_full:
        runtime_invocation["known_pilot_failure"] = (
            "chassis/cabin identity merge after frame-65 correction"
        )
    if is_four_part_focused or is_four_part_ego_focused:
        runtime_invocation["rescope_reason"] = (
            "old proxy frame 3120 starts with four separated parts before reassembly"
        )
    if condition is not None:
        runtime_invocation.update(
            {
                "condition_config": relative_uri(condition_config_path, repository_root),
                "condition_config_sha256": sha256_file(condition_config_path),
                "condition_id": condition.condition_id,
                "display_label": condition.display_label,
                "prompt_mode": condition.prompt_mode,
                "preprocessing": condition.preprocessing.model_dump(mode="json"),
                "manual_box_seed": (
                    condition.manual_box_seed.model_dump(mode="json")
                    if condition.manual_box_seed is not None
                    else None
                ),
            }
        )
    if text_target_config is not None:
        runtime_invocation.update(
            {
                "prompt_mode": "text_detection",
                "text_target_config": relative_uri(text_target_config_path, repository_root),
                "text_target_config_sha256": sha256_file(text_target_config_path),
                "text_prompt_mapping": text_target_payload,
            }
        )
    if hybrid_payload is not None:
        runtime_invocation.update(
            {
                "prompt_mode": "hybrid_text_and_manual_mask",
                "hybrid_config": relative_uri(args.hybrid_config.resolve(), repository_root),
                "hybrid_config_sha256": sha256_file(args.hybrid_config.resolve()),
                "hybrid_initialization": hybrid_payload,
                "label": (
                    "hybrid: three text detections plus one human-reviewed mask; "
                    "not pure zero-shot and not all-manual"
                ),
            }
        )
        if hybrid_smoke_approval is not None:
            runtime_invocation["hybrid_smoke_human_approval"] = hybrid_smoke_approval.model_dump(
                mode="json"
            )
    if manual_seed_payload is not None:
        runtime_invocation.update(
            {
                "prompt_mode": "manual_seed_multiplexed",
                "manual_seed_multiplex": manual_seed_payload,
                "initialization_api": "encode_prompt_memory_from_mask",
                "label": "manual-seed multiplexed; not out-of-box/text zero-shot",
            }
        )
    if multi_keyframe_schedule_payload is not None:
        runtime_invocation.update(
            {
                "prompt_mode": "manual_seed_multiplexed_keyframes",
                "multi_keyframe_correction_schedule": multi_keyframe_schedule_payload,
                "correction_memory_semantics": multi_keyframe_schedule_payload["memory_semantics"],
                "label": (
                    "multi-keyframe correction schedule with human-selected seeds and "
                    "agent-selected later corrections; not text zero-shot"
                    if any(
                        item.get("selected_by") == "agent"
                        for item in multi_keyframe_schedule_payload["corrections"]
                    )
                    else "human-selected multi-keyframe correction schedule; not text zero-shot"
                ),
            }
        )
    (run_directory / "runtime_settings.json").write_text(
        json.dumps(runtime_invocation, indent=2, sort_keys=True) + "\n"
    )

    if not args.external_python.is_file():
        worker_result: dict[str, Any] = {
            "state": "blocked",
            "reason": f"configured MuggledSAM interpreter does not exist: {args.external_python}",
            "frames_processed": 0,
            "elapsed_seconds": 0.0,
            "time_to_first_usable_output_seconds": None,
            "gpu_peak_vram_bytes": None,
            "masks_written": 0,
            "known_unavailable_measures": [
                "time_to_first_usable_output_seconds",
                "gpu_peak_vram_bytes",
            ],
            "runtime_settings": {},
        }
        (run_directory / "worker.stderr.log").write_text(worker_result["reason"] + "\n")
        (run_directory / "worker.stdout.log").write_text("")
    else:
        worker_result = _run_worker(
            external_python=args.external_python,
            worker_path=worker_path,
            run_directory=run_directory,
            proxy_path=proxy_path,
            view_id=proxy.view_id,
            source_offset_seconds=preprocessing.proxy_timing.source_seconds_for_frame(
                ClockName.ANALYSIS, 0
            ),
            model_path=model_path,
            max_frames=requested_frames,
            max_side_length=args.max_side_length,
            max_frame_memory=args.max_frame_memory,
            analysis_fps=analysis_fps,
            condition=condition,
            text_targets=text_target_payload if hybrid_payload is None else None,
            hybrid_initialization=hybrid_payload,
            manual_seeds=manual_seed_payload,
            multi_keyframe_schedule=multi_keyframe_schedule_payload,
            resume_from_checkpoint=resume_checkpoint,
            checkpoint_every=args.checkpoint_every,
        )

    observations_path = run_directory / "observations.jsonl"
    observations = load_observations(observations_path) if observations_path.is_file() else ()
    # A resumed run copies the prior prefix, which it did not step itself.
    frames_processed = min(int(worker_result["frames_processed"]) + resume_at, requested_frames)
    if len(observations) != frames_processed:
        worker_result["state"] = "failed"
        worker_result["reason"] = (
            f"worker reported {frames_processed} frames but emitted "
            f"{len(observations)} valid observations"
        )
        frames_processed = 0
        observations = ()
    if hybrid_metadata is not None and observations:
        initialized_labels = {object_.label for object_ in observations[0].objects}
        hybrid_metadata = hybrid_metadata.model_copy(
            update={
                "targets": tuple(
                    target.model_copy(
                        update={
                            "initialized_at_frame_zero": (target.output_label in initialized_labels)
                        }
                    )
                    for target in hybrid_metadata.targets
                )
            }
        )
    coverage = FullDurationCoverage(
        source_duration_seconds=preprocessing.clip.source_duration_seconds,
        covered_intervals=(
            (TimeInterval(start_seconds=0.0, end_seconds=frames_processed / analysis_fps),)
            if frames_processed
            else ()
        ),
    )
    method_state = MethodState(worker_result["state"])
    measurements_path = run_directory / "manual_seed_multiplex_metrics.json"
    measurements_written = False
    if manual_seed_metadata is not None and method_state is MethodState.SUCCEEDED:
        metrics = _manual_seed_metrics(
            observations=observations,
            worker_result=worker_result,
            manual_seed_metadata=manual_seed_metadata,
            source_fingerprint=ArtifactFingerprint(
                uri=proxy.raw_source.raw_uri,
                sha256=proxy.raw_source.checksum_sha256,
                source="approved_config",
            ),
            proxy_fingerprint=ArtifactFingerprint(
                uri=proxy.proxy_uri,
                sha256=proxy.checksum_sha256,
                source="approved_config",
            ),
            frames_requested=requested_frames,
            requested_seconds=requested_seconds,
            analysis_fps=analysis_fps,
        )
        measurements_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        measurements_written = True
    method_statuses = [
        MethodStatus(
            method_name=(
                "muggledsam-sam3-g3-full-static-hybrid-candidate"
                if is_g3_candidate and hybrid_payload is not None
                else "muggledsam-sam3-g3-full-static-candidate"
                if is_g3_candidate
                else "muggledsam-sam3-g4-e4-60-second-candidate"
                if is_e4_candidate
                else f"muggledsam-sam3-{condition.condition_id}"
                if condition
                else "muggledsam-sam3-full-ego-manual-seed-multiplexed-baseline"
                if is_full_ego_manual_seed
                else "muggledsam-sam3-four-part-static-20-second-pilot"
                if is_four_part_pilot
                else "muggledsam-sam3-four-part-static-full-exploratory"
                if is_four_part_full
                else "muggledsam-sam3-four-part-static-focused-reassembly"
                if is_four_part_focused
                else "muggledsam-sam3-four-part-ego-focused-reassembly"
                if is_four_part_ego_focused
                else "muggledsam-sam3-manual-seed-multiplexed-smoke"
                if manual_seed_payload is not None
                else "muggledsam-sam3-multi-keyframe-correction-smoke"
                if multi_keyframe_schedule_payload is not None
                else "muggledsam-sam3-static-hybrid-smoke"
                if hybrid_payload is not None
                else "muggledsam-sam3-core-method-smoke"
            ),
            stage="objects",
            state=method_state,
            artifact_uri=relative_uri(observations_path, repository_root)
            if observations_path.is_file()
            else None,
            blocker=worker_result.get("reason"),
            measured_on=(
                f"{proxy.view_id}; approved {requested_seconds:.1f}-second "
                f"{runtime_invocation['run_profile']}"
                + (f"; {condition.display_label}" if condition else "")
            ),
        )
    ]
    if manual_seed_metadata is not None:
        method_statuses.append(
            MethodStatus(
                method_name="manual-seed-multiplexed-output-measurements",
                stage="measurement",
                state=MethodState.SUCCEEDED if measurements_written else MethodState.NOT_RUN,
                artifact_uri=relative_uri(measurements_path, repository_root)
                if measurements_written
                else None,
                measured_on=(
                    "fixed [0, 5400) manual-seed multiplexed SAM3 stream; not accuracy"
                    if is_full_ego_manual_seed
                    else "fixed [0, 600) four-part static pilot; not accuracy"
                    if is_four_part_pilot
                    else "fixed [0, 300) manual-seed multiplexed SAM3 stream; not accuracy"
                ),
                blocker=(
                    None
                    if measurements_written
                    else "core manual-seed multiplexed smoke did not produce observations"
                ),
            )
        )
    rerun_path = run_directory / (
        "g3_full_static.rrd"
        if is_g3_candidate
        else "g4_e4_60_second_candidate.rrd"
        if is_e4_candidate
        else "full_ego_manual_seed_multiplexed_baseline.rrd"
        if is_full_ego_manual_seed
        else "four_part_static_pilot.rrd"
        if is_four_part_pilot
        else "four_part_static_full_exploratory.rrd"
        if is_four_part_full
        else "four_part_static_focused_reassembly.rrd"
        if is_four_part_focused
        else "four_part_ego_focused_reassembly.rrd"
        if is_four_part_ego_focused
        else "smoke.rrd"
    )
    if method_state is MethodState.SUCCEEDED:
        try:
            condition_input = run_directory / "condition_input_300f.mp4"
            smoke_video_path = (
                condition_input
                if condition is not None and condition_input.is_file()
                else _create_bounded_rerun_video(
                    proxy_path=proxy_path,
                    output_path=run_directory / f"input_{requested_frames}f.mp4",
                    run_directory=run_directory,
                    frame_count=requested_frames,
                )
            )
            export_run(
                RunManifest(
                    run_id=run_directory.name,
                    clip=preprocessing.clip,
                    coverage=coverage,
                    chunk_policy=ChunkContinuityPolicy(
                        overlap_seconds=0.0,
                        max_allowed_gap_seconds=0.0,
                    ),
                    method_statuses=tuple(method_statuses),
                    observations=observations,
                ),
                rerun_path,
                video_path=smoke_video_path,
                video_dimensions=(proxy.dimensions.width, proxy.dimensions.height),
                asset_reference=EncodedAssetInput(
                    uri=proxy.proxy_uri,
                    media_type="video/mp4",
                    checksum_sha256=proxy.checksum_sha256,
                ),
                mask_artifact_root=run_directory,
            )
            _write_mask_cache_sidecar(run_directory)
            method_statuses.append(
                MethodStatus(
                    method_name="rerun-g3-candidate-export"
                    if is_g3_candidate
                    else "rerun-g4-e4-candidate-export"
                    if is_e4_candidate
                    else "rerun-full-ego-manual-seed-baseline-export"
                    if is_full_ego_manual_seed
                    else "rerun-four-part-static-full-exploratory-export"
                    if is_four_part_full
                    else "rerun-four-part-static-focused-reassembly-export"
                    if is_four_part_focused
                    else "rerun-four-part-ego-focused-reassembly-export"
                    if is_four_part_ego_focused
                    else "rerun-smoke-export",
                    stage="export",
                    state=MethodState.SUCCEEDED,
                    artifact_uri=relative_uri(rerun_path, repository_root),
                    measured_on=(
                        "normalized SAM3 observations; approved-range input video logged once"
                    ),
                )
            )
        except Exception as error:
            method_statuses.append(
                MethodStatus(
                    method_name=(
                        "rerun-g3-candidate-export"
                        if is_g3_candidate
                        else "rerun-g4-e4-candidate-export"
                        if is_e4_candidate
                        else "rerun-full-ego-manual-seed-baseline-export"
                        if is_full_ego_manual_seed
                        else "rerun-four-part-static-full-exploratory-export"
                        if is_four_part_full
                        else "rerun-four-part-static-focused-reassembly-export"
                        if is_four_part_focused
                        else "rerun-four-part-ego-focused-reassembly-export"
                        if is_four_part_ego_focused
                        else "rerun-smoke-export"
                    ),
                    stage="export",
                    state=MethodState.FAILED,
                    blocker=f"{type(error).__name__}: {error}",
                )
            )
    else:
        method_statuses.append(
            MethodStatus(
                method_name=(
                    "rerun-g3-candidate-export"
                    if is_g3_candidate
                    else "rerun-g4-e4-candidate-export"
                    if is_e4_candidate
                    else "rerun-full-ego-manual-seed-baseline-export"
                    if is_full_ego_manual_seed
                    else "rerun-four-part-static-full-exploratory-export"
                    if is_four_part_full
                    else "rerun-four-part-static-focused-reassembly-export"
                    if is_four_part_focused
                    else "rerun-four-part-ego-focused-reassembly-export"
                    if is_four_part_ego_focused
                    else "rerun-smoke-export"
                ),
                stage="export",
                state=MethodState.NOT_RUN,
                blocker="core method did not produce successful normalized observations",
            )
        )

    qa_path = None
    if (
        manual_seed_metadata is not None
        or multi_keyframe_correction_metadata is not None
        or hybrid_metadata is not None
    ) and method_state is MethodState.SUCCEEDED:
        try:
            qa_path = (
                _render_hybrid_contact_sheet(
                    repository_root=repository_root,
                    run_directory=run_directory,
                    external_python=args.external_python,
                    source_offset_seconds=preprocessing.proxy_timing.source_seconds_for_frame(
                        ClockName.ANALYSIS, 0
                    ),
                    full_run=is_g3_candidate,
                )
                if hybrid_metadata is not None
                else _render_four_part_contact_sheet(
                    repository_root=repository_root,
                    run_directory=run_directory,
                    external_python=args.external_python,
                    view_id=proxy.view_id,
                    frame_count=requested_frames,
                    duration_seconds=requested_seconds,
                )
                if (
                    is_four_part_pilot
                    or is_four_part_full
                    or is_four_part_focused
                    or is_four_part_ego_focused
                )
                else _render_full_ego_manual_seed_contact_sheet(
                    repository_root=repository_root,
                    run_directory=run_directory,
                    external_python=args.external_python,
                )
                if is_full_ego_manual_seed
                else _render_manual_seed_comparison(
                    repository_root=repository_root,
                    run_directory=run_directory,
                    external_python=args.external_python,
                )
            )
            method_statuses.append(
                MethodStatus(
                    method_name=_qa_method_name(
                        hybrid=hybrid_metadata is not None,
                        full_ego=is_full_ego_manual_seed,
                        four_part_pilot=is_four_part_pilot,
                        four_part_full=is_four_part_full,
                        four_part_static_focused=is_four_part_focused,
                        four_part_ego_focused=is_four_part_ego_focused,
                    ),
                    stage="review",
                    state=MethodState.SUCCEEDED,
                    artifact_uri=relative_uri(qa_path, repository_root),
                    measured_on=(
                        "recorded outputs at 0.000, 90.000, and 179.967 seconds; "
                        "review-only, not accuracy"
                        if hybrid_metadata is not None and is_g3_candidate
                        else "recorded outputs at 0.000, 5.000, and 9.967 seconds; "
                        "review-only, not accuracy"
                        if hybrid_metadata is not None
                        else "recorded outputs at 0.000, 90.000, and 179.967 seconds; "
                        "review-only, not accuracy"
                        if is_full_ego_manual_seed
                        else "recorded outputs at 0.000, 10.000, and 19.967 seconds; "
                        "review-only, not accuracy"
                        if is_four_part_pilot
                        else "recorded outputs at 0.000, 98.350, and 196.667 seconds; "
                        "review-only, not accuracy"
                        if is_four_part_full
                        else "recorded outputs at 0.000, 46.350, and 92.667 seconds; "
                        "review-only, not accuracy"
                        if is_four_part_focused or is_four_part_ego_focused
                        else "recorded outputs at 0.000, 5.000, and 9.967 seconds; "
                        "review-only, not accuracy"
                    ),
                )
            )
        except Exception as error:
            method_statuses.append(
                MethodStatus(
                    method_name=_qa_method_name(
                        hybrid=hybrid_metadata is not None,
                        full_ego=is_full_ego_manual_seed,
                        four_part_pilot=is_four_part_pilot,
                        four_part_full=is_four_part_full,
                        four_part_static_focused=is_four_part_focused,
                        four_part_ego_focused=is_four_part_ego_focused,
                    ),
                    stage="review",
                    state=MethodState.FAILED,
                    blocker=f"{type(error).__name__}: {error}",
                )
            )

    metadata_common = dict(
        concepts=configured_concepts,
        source_fingerprint=ArtifactFingerprint(
            uri=proxy.raw_source.raw_uri,
            sha256=proxy.raw_source.checksum_sha256,
            source="approved_config",
        ),
        proxy_fingerprint=ArtifactFingerprint(
            uri=proxy.proxy_uri,
            sha256=proxy.checksum_sha256,
            source="approved_config",
        ),
        config_fingerprint=ArtifactFingerprint(
            uri=relative_uri(config_path, repository_root),
            sha256=sha256_file(config_path),
            source="measured",
        ),
        adapter=AdapterMetadata(
            name="battle.muggled_smoke",
            version=ADAPTER_VERSION,
            implementation_basis="MuggledSAM simple_examples/video_segmentation_multiplexed.py",
            external_source_uri=str(MUGGLED_SAM_SOURCE),
            external_revision=_external_revision(),
        ),
        continuity=StreamContinuityPolicy(
            max_prompt_memory_entries=1,
            max_frame_memory_entries=args.max_frame_memory,
            detected_object_limit=len(configured_concepts),
            **resume_continuity,
        ),
        runtime_settings={
            "external_python": str(args.external_python),
            "model_path": str(model_path),
            "cuda_visible_devices": "0",
            **_flat_runtime_settings(worker_result.get("runtime_settings", {})),
        },
        measurements=RuntimeMeasurements(
            elapsed_seconds=float(worker_result["elapsed_seconds"]),
            time_to_first_usable_output_seconds=worker_result.get(
                "time_to_first_usable_output_seconds"
            ),
            gpu_peak_vram_bytes=worker_result.get("gpu_peak_vram_bytes"),
            known_unavailable_measures=tuple(worker_result.get("known_unavailable_measures", [])),
        ),
        observations_uri=relative_uri(observations_path, repository_root)
        if observations_path.is_file()
        else None,
        mask_artifact_uri=relative_uri(run_directory / "masks", repository_root)
        if (run_directory / "masks").is_dir()
        else None,
        mask_artifact_count=int(worker_result["masks_written"]),
        rerun_artifact_uri=relative_uri(rerun_path, repository_root)
        if rerun_path.is_file()
        else None,
    )
    text_target_config_fingerprint = (
        ArtifactFingerprint(
            uri=relative_uri(text_target_config_path, repository_root),
            sha256=sha256_file(text_target_config_path),
            source="measured",
        )
        if text_target_config_path is not None
        else None
    )
    smoke = None
    g3_candidate = None
    e4_candidate = None
    full_ego_manual_seed = None
    four_part_pilot = None
    four_part_full = None
    four_part_focused = None
    if is_g3_candidate:
        g3_candidate = G3CandidateRunMetadata(
            requested_analysis_frame_range=requested_range,
            requested_seconds=G3_STATIC_SECONDS,
            view_id=proxy.view_id,
            text_target_config_fingerprint=text_target_config_fingerprint,
            text_prompt_mapping=(
                text_target_config.targets if text_target_config is not None else ()
            ),
            hybrid_initialization=hybrid_metadata,
            hybrid_smoke_human_approval=hybrid_smoke_approval,
            **metadata_common,
        )
    elif is_e4_candidate:
        e4_candidate = E4CandidateRunMetadata(
            requested_analysis_frame_range=requested_range,
            requested_seconds=E4_CANDIDATE_SECONDS,
            view_id=proxy.view_id,
            **metadata_common,
        )
    elif is_full_ego_manual_seed:
        full_ego_manual_seed = FullEgoManualSeedRunMetadata(
            requested_analysis_frame_range=requested_range,
            requested_seconds=FULL_EGO_MANUAL_SEED_SECONDS,
            view_id=proxy.view_id,
            measurements_artifact_uri=relative_uri(measurements_path, repository_root)
            if measurements_written
            else None,
            qa_artifact_uri=relative_uri(qa_path, repository_root) if qa_path is not None else None,
            manual_seed_multiplex=manual_seed_metadata,
            **metadata_common,
        )
    elif is_four_part_pilot:
        four_part_pilot = FourPartPilotRunMetadata(
            requested_analysis_frame_range=requested_range,
            requested_seconds=FOUR_PART_PILOT_SECONDS,
            view_id=proxy.view_id,
            measurements_artifact_uri=relative_uri(measurements_path, repository_root)
            if measurements_written
            else None,
            qa_artifact_uri=relative_uri(qa_path, repository_root) if qa_path is not None else None,
            manual_seed_multiplex=manual_seed_metadata,
            multi_keyframe_corrections=multi_keyframe_correction_metadata,
            **metadata_common,
        )
    elif is_four_part_full:
        four_part_full = FourPartFullRunMetadata(
            requested_analysis_frame_range=requested_range,
            requested_seconds=FOUR_PART_FULL_SECONDS,
            view_id=proxy.view_id,
            qa_artifact_uri=relative_uri(qa_path, repository_root) if qa_path is not None else None,
            multi_keyframe_corrections=multi_keyframe_correction_metadata,
            known_pilot_failure="chassis/cabin identity merge after frame-65 correction",
            **metadata_common,
        )
    elif is_four_part_focused or is_four_part_ego_focused:
        four_part_focused = FourPartFocusedRunMetadata(
            requested_analysis_frame_range=requested_range,
            requested_seconds=FOUR_PART_FOCUSED_SECONDS,
            view_id=proxy.view_id,
            qa_artifact_uri=relative_uri(qa_path, repository_root) if qa_path is not None else None,
            multi_keyframe_corrections=multi_keyframe_correction_metadata,
            rescope_reason=(
                "old proxy frame 3120 starts with four separated parts before reassembly"
            ),
            **metadata_common,
        )
    else:
        smoke = SmokeRunMetadata(
            requested_analysis_frame_range=requested_range,
            requested_seconds=SMOKE_SECONDS,
            measurements_artifact_uri=relative_uri(measurements_path, repository_root)
            if measurements_written
            else None,
            qa_artifact_uri=relative_uri(qa_path, repository_root) if qa_path is not None else None,
            manual_seed_multiplex=manual_seed_metadata,
            text_target_config_fingerprint=text_target_config_fingerprint,
            text_prompt_mapping=(
                text_target_config.targets if text_target_config is not None else ()
            ),
            hybrid_initialization=hybrid_metadata,
            **metadata_common,
            multi_keyframe_corrections=multi_keyframe_correction_metadata,
        )
    manifest = RunManifest(
        run_id=run_directory.name,
        clip=preprocessing.clip,
        coverage=coverage,
        chunk_policy=ChunkContinuityPolicy(overlap_seconds=0.0, max_allowed_gap_seconds=0.0),
        method_statuses=tuple(method_statuses),
        observations=observations,
        smoke=smoke,
        g3_candidate=g3_candidate,
        e4_candidate=e4_candidate,
        full_ego_manual_seed=full_ego_manual_seed,
        four_part_pilot=four_part_pilot,
        four_part_full=four_part_full,
        four_part_focused=four_part_focused,
    )
    (run_directory / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
    return run_directory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a fixed headless SAM3 smoke or approved G3 candidate."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/clips/assembly101_nusar_9033_g2.json"),
    )
    parser.add_argument(
        "--view",
        choices=(
            "static-c10379",
            "ego-hmc21110305",
            "ego-hmc21176875",
            "ego-hmc21176623",
            "ego-hmc21179183",
        ),
        required=True,
    )
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=SMOKE_FRAMES)
    parser.add_argument(
        "--max-side-length",
        type=int,
        default=DEFAULT_MAX_SIDE_LENGTH,
        help=(
            "Longest encoded input side given to the model. The default downscales the "
            "954x720 ego proxy, so raising it trades VRAM and time for small-target detail."
        ),
    )
    parser.add_argument(
        "--max-frame-memory",
        type=int,
        default=DEFAULT_MAX_FRAME_MEMORY,
        help=(
            "Frame-memory entries kept per stream. Its span in seconds is this count divided "
            "by the analysis frame rate, so it must be raised alongside any frame-rate change."
        ),
    )
    parser.add_argument(
        "--g3-full-static",
        action="store_true",
        help="Enable only the user-approved static 5,400-frame candidate range.",
    )
    parser.add_argument(
        "--g4-e4-candidate",
        action="store_true",
        help="Enable only the user-approved e4 1,800-frame / 60-second candidate range.",
    )
    parser.add_argument(
        "--full-ego-manual-seed",
        action="store_true",
        help="Enable only the approved 5,400-frame manual-seed multiplexed ego baseline.",
    )
    parser.add_argument(
        "--four-part-static-pilot",
        action="store_true",
        help="Enable only the approved 600-frame/20-second static four-part pilot.",
    )
    parser.add_argument(
        "--four-part-static-full",
        action="store_true",
        help="Run the known-imperfect full 5,901-frame static four-part exploration.",
    )
    parser.add_argument(
        "--four-part-static-focused",
        action="store_true",
        help="Run the 2,781-frame separated-to-assembled focused four-part proxy.",
    )
    parser.add_argument(
        "--four-part-ego-focused",
        action="store_true",
        help="Run the aligned 2,781-frame monochrome ego four-part proxy.",
    )
    parser.add_argument("--external-python", type=Path, default=MUGGLED_SAM_PYTHON)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--condition-config", type=Path)
    parser.add_argument("--condition-id")
    parser.add_argument(
        "--text-target-config",
        type=Path,
        help=(
            "Versioned static zero-shot mapping from normalized output labels to "
            "human-readable SAM3 text prompts."
        ),
    )
    parser.add_argument(
        "--hybrid-config",
        type=Path,
        help=(
            "Versioned static contract that exclusively binds text targets, a reviewed "
            "manual proposal, source fingerprints, and canonical combined ordering."
        ),
    )
    parser.add_argument(
        "--approved-smoke-manifest",
        type=Path,
        help="Exact reviewed hybrid smoke manifest authorizing a full G3 run.",
    )
    parser.add_argument(
        "--human-approved-at",
        type=datetime.fromisoformat,
        help="Timezone-aware ISO-8601 time of the explicit hybrid smoke approval.",
    )
    parser.add_argument("--human-approved-by")
    parser.add_argument("--human-approval-statement")
    parser.add_argument(
        "--manual-seed-proposal",
        type=Path,
        help="Validated proposal whose selected masks initialize one SAM3 multiplex stream.",
    )
    parser.add_argument(
        "--manual-seed-target-config",
        type=Path,
        help="Optional named target policy; must match the proposal when it embeds one.",
    )
    parser.add_argument(
        "--multi-keyframe-correction-schedule",
        type=Path,
        help=("Integrity-validated frame-0 seed plus later correction schedule."),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help=(
            "Save tracker state every N frames in addition to the correction keyframes, "
            "so a later rerun can resume instead of re-streaming from frame zero."
        ),
    )
    parser.add_argument(
        "--resume-run",
        type=Path,
        help=(
            "Prior run whose frames before --resume-at are copied unchanged and whose "
            "tracker checkpoint continues this stream."
        ),
    )
    parser.add_argument(
        "--resume-at",
        type=int,
        help="First frame this run steps itself; requires a checkpoint at that frame.",
    )
    args = parser.parse_args()
    if args.resume_at is not None and args.resume_at < 1:
        parser.error("--resume-at must be at least 1")
    if args.resume_at is not None and args.resume_run is None:
        parser.error("--resume-at requires --resume-run")
    if args.g3_full_static and args.max_frames == SMOKE_FRAMES:
        args.max_frames = G3_STATIC_FRAMES
    if args.g4_e4_candidate and args.max_frames == SMOKE_FRAMES:
        args.max_frames = E4_CANDIDATE_FRAMES
    if args.full_ego_manual_seed and args.max_frames == SMOKE_FRAMES:
        args.max_frames = FULL_EGO_MANUAL_SEED_FRAMES
    if args.four_part_static_pilot and args.max_frames == SMOKE_FRAMES:
        args.max_frames = FOUR_PART_PILOT_FRAMES
    if args.four_part_static_full and args.max_frames == SMOKE_FRAMES:
        args.max_frames = FOUR_PART_FULL_FRAMES
    if args.four_part_static_focused and args.max_frames == SMOKE_FRAMES:
        args.max_frames = FOUR_PART_FOCUSED_FRAMES
    if args.four_part_ego_focused and args.max_frames == SMOKE_FRAMES:
        args.max_frames = FOUR_PART_FOCUSED_FRAMES
    try:
        run_directory = run_smoke(args)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(f"Wrote MuggledSAM/SAM3 run: {run_directory}")


if __name__ == "__main__":
    main()
