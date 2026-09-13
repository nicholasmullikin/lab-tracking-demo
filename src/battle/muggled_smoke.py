"""Headless, fixed-budget adapter for a MuggledSAM/SAM3 core-method smoke test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .exporter import export_run
from .schemas import (
    AdapterMetadata,
    ArtifactFingerprint,
    ChunkContinuityPolicy,
    ClockName,
    E4CandidateRunMetadata,
    EncodedAssetInput,
    FrameObservations,
    FrameRange,
    FullDurationCoverage,
    FullEgoManualSeedRunMetadata,
    G2PreprocessingManifest,
    G3CandidateRunMetadata,
    ManualSeedCandidateProvenance,
    ManualSeedMultiplexMetadata,
    MethodState,
    MethodStatus,
    MuggledSAMBoxCalibrationManifest,
    MuggledSAMEgoCondition,
    MuggledSAMEgoConditionConfig,
    MuggledSAMManualSeedTargetConfig,
    MuggledSAMMultiKeyframeCorrectionPolicy,
    MuggledSAMMultiKeyframeCorrectionSchedule,
    MuggledSAMProposedTrackingPromptConfig,
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
G3_STATIC_FRAMES = 5400
G3_STATIC_SECONDS = 180.0
E4_CANDIDATE_FRAMES = 1800
E4_CANDIDATE_SECONDS = 60.0
FULL_EGO_MANUAL_SEED_FRAMES = 5400
FULL_EGO_MANUAL_SEED_SECONDS = 180.0
MUGGLED_SAM_SOURCE = Path("/home/nick/src/muggled_sam")
MUGGLED_SAM_PYTHON = Path("/home/nick/.pyenv/versions/muggled_sam/bin/python")
DEFAULT_MODEL = MUGGLED_SAM_SOURCE / "model_weights" / "sam3.1_multiplex.pt"


def sha256_file(path: Path) -> str:
    """Hash a file incrementally, never loading media/configuration fully into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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
) -> MuggledSAMManualSeedTargetConfig:
    """Load a named e4 target policy bound to the selected approved G2 configuration."""
    target_config = MuggledSAMManualSeedTargetConfig.model_validate_json(
        target_config_path.read_text()
    )
    if target_config.view_id != view_id:
        raise ValueError("manual-seed target config view must match the selected approved proxy")
    if (repository_root / target_config.base_g2_config).resolve() != g2_config_path.resolve():
        raise ValueError("manual-seed target config must name the selected G2 configuration")
    return target_config


def make_run_id(view_id: str, now: datetime | None = None, *, profile: str = "smoke") -> str:
    timestamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ").lower()
    return f"muggledsam-sam3-{profile}-{view_id.lower()}-{timestamp}"


def require_smoke_range(start_frame: int, max_frames: int) -> FrameRange:
    """Refuse ranges other than the user-approved first ten seconds."""
    if start_frame != 0 or max_frames != SMOKE_FRAMES:
        raise ValueError(
            f"smoke policy permits only proxy frames [0, {SMOKE_FRAMES}); "
            f"received [{start_frame}, {start_frame + max_frames})"
        )
    return FrameRange(start_frame=start_frame, end_frame_exclusive=SMOKE_FRAMES)


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
    condition: MuggledSAMEgoCondition | None = None,
    manual_seeds: dict[str, Any] | None = None,
    multi_keyframe_schedule: dict[str, Any] | None = None,
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
    ]
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
    if calibration.base_g2_config != relative_uri(
        config_path, repository_root
    ) or calibration.base_g2_config_sha256 != sha256_file(config_path):
        raise ValueError("manual-seed calibration must match the selected G2 configuration")
    if (
        calibration.proxy.uri != proxy.proxy_uri
        or calibration.proxy.sha256 != proxy.checksum_sha256
        or calibration.source.uri != proxy.raw_source.raw_uri
        or calibration.source.sha256 != proxy.raw_source.checksum_sha256
    ):
        raise ValueError("manual-seed calibration source/proxy fingerprints do not match G2")

    target_config = (
        load_manual_seed_target_config(
            target_config_path=target_config_path,
            repository_root=repository_root,
            g2_config_path=config_path,
            view_id=proxy.view_id,
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
        ),
    )


def _resolve_artifact_uri(repository_root: Path, uri: str) -> Path:
    path = Path(uri)
    return path if path.is_absolute() else (repository_root / path).resolve()


def _load_multi_keyframe_correction_schedule(
    *,
    schedule_path: Path,
    repository_root: Path,
    config_path: Path,
    proxy: Any,
) -> tuple[dict[str, Any], MultiKeyframeCorrectionScheduleMetadata]:
    """Validate a proposed schedule and prepare its frame-zero and correction mask payload."""
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
        if correction.frame.analysis_frame_index >= SMOKE_FRAMES:
            raise ValueError(
                f"bounded correction smoke permits only frames [0, {SMOKE_FRAMES}): "
                f"{correction.frame.analysis_frame_index}"
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
            or candidate.human_selected_candidate_index
            != correction.human_selected_candidate_index
            or not candidate.human_accepted
            or (
                candidate.frame.analysis_frame_index == 0
                and not candidate.selected_for_finalization
            )
            or (
                candidate.frame.analysis_frame_index != 0
                and not candidate.selected_for_correction
            )
        ):
            raise ValueError(
                f"correction schedule entry does not match an eligible human selection: "
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
        ),
    )


def _create_bounded_rerun_video(
    *, proxy_path: Path, output_path: Path, run_directory: Path, frame_count: int
) -> Path:
    """Create the exact approved-range video asset embedded once in a Rerun recording."""
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
    return {
        "schema_version": "1.0",
        "measurement_scope": (
            "manual-seed multiplexed SAM3 output/continuity measures over the fixed "
            f"{requested_seconds:.1f}-second proxy; not ground-truth accuracy"
        ),
        "run_label": "manual-seed multiplexed; not out-of-box/text zero-shot",
        "frames_requested": frames_requested,
        "frames_processed": len(observations),
        "analysis_fps": 30,
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
            "mask_period_frames": 6,
            "maximum_hz": 5.0,
            "frames_with_any_external_mask": all_mask_frames,
            "all_masks_on_declared_cadence": all(
                frame_index % 6 == 0 for frame_index in all_mask_frames
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
        run_directory
        / "g3_review"
        / "ego-hmc21179183_full_ego_manual_seed_multiplexed_qa.png"
    )
    if completed.returncode != 0 or not output_path.is_file():
        raise RuntimeError(
            f"could not render full ego manual-seed QA contact sheet (exit {completed.returncode})"
        )
    return output_path


def run_smoke(args: argparse.Namespace) -> Path:
    repository_root = Path.cwd().resolve()
    config_path = args.config.resolve()
    preprocessing = G2PreprocessingManifest.model_validate_json(config_path.read_text())
    proxy = next((proxy for proxy in preprocessing.proxies if proxy.view_id == args.view), None)
    if proxy is None:
        raise ValueError(f"view {args.view!r} is not present in {config_path}")
    condition = None
    condition_config_path = None
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
    if args.manual_seed_proposal is not None:
        if condition is not None:
            raise ValueError(
                "manual-seed multiplexing cannot be combined with a condition experiment"
            )
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
        if condition is not None or manual_seed_payload is not None:
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
        )
    is_g3_candidate = args.g3_full_static
    is_e4_candidate = args.g4_e4_candidate
    is_full_ego_manual_seed = args.full_ego_manual_seed
    if sum((is_g3_candidate, is_e4_candidate, is_full_ego_manual_seed)) > 1:
        raise ValueError("only one approved candidate profile may be selected")
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
        or manual_seed_payload is None
        or multi_keyframe_schedule_payload is not None
    ):
        raise ValueError(
            "full ego baseline requires the approved manual-seed multiplex payload only"
        )
    requested_frames = (
        G3_STATIC_FRAMES
        if is_g3_candidate
        else E4_CANDIDATE_FRAMES
        if is_e4_candidate
        else FULL_EGO_MANUAL_SEED_FRAMES
        if is_full_ego_manual_seed
        else SMOKE_FRAMES
    )
    requested_seconds = (
        G3_STATIC_SECONDS
        if is_g3_candidate
        else E4_CANDIDATE_SECONDS
        if is_e4_candidate
        else FULL_EGO_MANUAL_SEED_SECONDS
        if is_full_ego_manual_seed
        else SMOKE_SECONDS
    )
    requested_range = (
        require_g3_static_range(proxy.view_id, args.start_frame, args.max_frames)
        if is_g3_candidate
        else require_e4_candidate_range(proxy.view_id, args.start_frame, args.max_frames)
        if is_e4_candidate
        else require_full_ego_manual_seed_range(proxy.view_id, args.start_frame, args.max_frames)
        if is_full_ego_manual_seed
        else require_smoke_range(args.start_frame, args.max_frames)
    )

    profile = (
        "g3-full"
        if is_g3_candidate
        else "g4-e4-candidate"
        if is_e4_candidate
        else "full-ego-manual-seed-multiplexed"
        if is_full_ego_manual_seed
        else f"smoke-{condition.condition_id}"
        if condition
        else "smoke-manual-seed-multiplexed"
        if manual_seed_payload is not None
        else "smoke-multi-keyframe-corrections"
        if multi_keyframe_schedule_payload is not None
        else "smoke"
    )
    run_id = make_run_id(proxy.view_id, profile=profile)
    run_directory = (args.run_root / run_id).resolve()
    suffix = 2
    while run_directory.exists():
        run_directory = (args.run_root / f"{run_id}-{suffix}").resolve()
        suffix += 1
    run_directory.mkdir(parents=True)
    proxy_path = (repository_root / proxy.proxy_uri).resolve()
    model_path = args.model.resolve()
    worker_path = Path(__file__).with_name("muggled_worker.py")
    runtime_invocation = {
        "run_id": run_directory.name,
        "config": relative_uri(config_path, repository_root),
        "view_id": proxy.view_id,
        "proxy": proxy.proxy_uri,
        "run_profile": (
            "g3_full_static_candidate"
            if is_g3_candidate
            else "g4_e4_60_second_candidate"
            if is_e4_candidate
            else "full_ego_manual_seed_multiplexed_baseline"
            if is_full_ego_manual_seed
            else "smoke"
        ),
        "approved_range": {"start_frame": 0, "end_frame_exclusive": requested_frames},
        "approved_seconds": requested_seconds,
        "concepts": list(
            condition.concepts
            if condition
            else tuple(seed["target"] for seed in manual_seed_payload["seeds"])
            if manual_seed_payload is not None
            else tuple(seed["target"] for seed in multi_keyframe_schedule_payload["seeds"])
            if multi_keyframe_schedule_payload is not None
            else CONCEPTS
        ),
        "external_python": str(args.external_python),
        "muggled_sam_source": str(MUGGLED_SAM_SOURCE),
        "model_path": str(model_path),
        "cuda_visible_devices": "0",
    }
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
                "label": "human-selected multi-keyframe correction schedule; not text zero-shot",
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
            condition=condition,
            manual_seeds=manual_seed_payload,
            multi_keyframe_schedule=multi_keyframe_schedule_payload,
        )

    observations_path = run_directory / "observations.jsonl"
    observations = load_observations(observations_path) if observations_path.is_file() else ()
    frames_processed = min(int(worker_result["frames_processed"]), requested_frames)
    if len(observations) != frames_processed:
        worker_result["state"] = "failed"
        worker_result["reason"] = (
            f"worker reported {frames_processed} frames but emitted "
            f"{len(observations)} valid observations"
        )
        frames_processed = 0
        observations = ()
    coverage = FullDurationCoverage(
        source_duration_seconds=preprocessing.clip.source_duration_seconds,
        covered_intervals=(
            (TimeInterval(start_seconds=0.0, end_seconds=frames_processed / 30.0),)
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
        )
        measurements_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        measurements_written = True
    method_statuses = [
        MethodStatus(
            method_name=(
                "muggledsam-sam3-g3-full-static-candidate"
                if is_g3_candidate
                else "muggledsam-sam3-g4-e4-60-second-candidate"
                if is_e4_candidate
                else f"muggledsam-sam3-{condition.condition_id}"
                if condition
                else "muggledsam-sam3-full-ego-manual-seed-multiplexed-baseline"
                if is_full_ego_manual_seed
                else "muggledsam-sam3-manual-seed-multiplexed-smoke"
                if manual_seed_payload is not None
                else "muggledsam-sam3-multi-keyframe-correction-smoke"
                if multi_keyframe_schedule_payload is not None
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
            method_statuses.append(
                MethodStatus(
                    method_name="rerun-g3-candidate-export"
                    if is_g3_candidate
                    else "rerun-g4-e4-candidate-export"
                    if is_e4_candidate
                    else "rerun-full-ego-manual-seed-baseline-export"
                    if is_full_ego_manual_seed
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
                    else "rerun-smoke-export"
                ),
                stage="export",
                state=MethodState.NOT_RUN,
                blocker="core method did not produce successful normalized observations",
            )
        )

    qa_path = None
    if manual_seed_metadata is not None and method_state is MethodState.SUCCEEDED:
        try:
            qa_path = (
                _render_full_ego_manual_seed_contact_sheet(
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
                    method_name=(
                        "full-ego-manual-seed-multiplexed-qa"
                        if is_full_ego_manual_seed
                        else "manual-seed-multiplexed-e4-zero-shot-qa"
                    ),
                    stage="review",
                    state=MethodState.SUCCEEDED,
                    artifact_uri=relative_uri(qa_path, repository_root),
                    measured_on=(
                        "recorded outputs at 0.000, 90.000, and 179.967 seconds; "
                        "review-only, not accuracy"
                        if is_full_ego_manual_seed
                        else "recorded outputs at 0.000, 5.000, and 9.967 seconds; "
                        "review-only, not accuracy"
                    ),
                )
            )
        except Exception as error:
            method_statuses.append(
                MethodStatus(
                    method_name=(
                        "full-ego-manual-seed-multiplexed-qa"
                        if is_full_ego_manual_seed
                        else "manual-seed-multiplexed-e4-zero-shot-qa"
                    ),
                    stage="review",
                    state=MethodState.FAILED,
                    blocker=f"{type(error).__name__}: {error}",
                )
            )

    metadata_common = dict(
        concepts=(
            condition.concepts
            if condition
            else tuple(seed["target"] for seed in manual_seed_payload["seeds"])
            if manual_seed_payload is not None
            else tuple(seed["target"] for seed in multi_keyframe_schedule_payload["seeds"])
            if multi_keyframe_schedule_payload is not None
            else CONCEPTS
        ),
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
            detected_object_limit=len(
                condition.concepts
                if condition
                else manual_seed_payload["seeds"]
                if manual_seed_payload is not None
                else CONCEPTS
            ),
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
        measurements_artifact_uri=relative_uri(measurements_path, repository_root)
        if measurements_written
        else None,
        qa_artifact_uri=relative_uri(qa_path, repository_root) if qa_path is not None else None,
        manual_seed_multiplex=manual_seed_metadata,
    )
    smoke = None
    g3_candidate = None
    e4_candidate = None
    full_ego_manual_seed = None
    if is_g3_candidate:
        g3_candidate = G3CandidateRunMetadata(
            requested_analysis_frame_range=requested_range,
            requested_seconds=G3_STATIC_SECONDS,
            view_id=proxy.view_id,
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
            **metadata_common,
        )
    else:
        smoke = SmokeRunMetadata(
            requested_analysis_frame_range=requested_range,
            requested_seconds=SMOKE_SECONDS,
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
    parser.add_argument("--external-python", type=Path, default=MUGGLED_SAM_PYTHON)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--condition-config", type=Path)
    parser.add_argument("--condition-id")
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
        help=(
            "Integrity-validated frame-0 seed plus later correction schedule; "
            "permitted only for the 300-frame smoke budget."
        ),
    )
    args = parser.parse_args()
    if args.g3_full_static and args.max_frames == SMOKE_FRAMES:
        args.max_frames = G3_STATIC_FRAMES
    if args.g4_e4_candidate and args.max_frames == SMOKE_FRAMES:
        args.max_frames = E4_CANDIDATE_FRAMES
    if args.full_ego_manual_seed and args.max_frames == SMOKE_FRAMES:
        args.max_frames = FULL_EGO_MANUAL_SEED_FRAMES
    try:
        run_directory = run_smoke(args)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(f"Wrote MuggledSAM/SAM3 run: {run_directory}")


if __name__ == "__main__":
    main()
