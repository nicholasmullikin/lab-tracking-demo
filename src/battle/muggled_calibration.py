"""Interactive, image-only SAM3 box-prompt calibration for the approved e4 proxy."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from .fs_common import run_timestamp
from .muggled_smoke import (
    DEFAULT_MODEL,
    MUGGLED_SAM_PYTHON,
    MUGGLED_SAM_SOURCE,
    load_manual_seed_target_config,
    relative_uri,
    sha256_file,
)
from .schemas import (
    ArtifactFingerprint,
    CalibrationFrameReference,
    G2PreprocessingManifest,
    MuggledSAMBoxCalibrationManifest,
    MuggledSAMManualSeedTargetConfig,
    MuggledSAMMultiKeyframeCorrection,
    MuggledSAMMultiKeyframeCorrectionPolicy,
    MuggledSAMMultiKeyframeCorrectionSchedule,
    MuggledSAMMultiplexSlot,
    MuggledSAMProposedTrackingPromptConfig,
    ProposedTrackingSeed,
    VideoDimensions,
)

TOOL_VERSION = "0.6.0"
DEFAULT_TIMESTAMPS_SECONDS = (0.0, 10.0, 30.0, 50.0)
DEFAULT_CONFIG = Path("configs/clips/assembly101_nusar_9033_ego_viewpoint_screen_g2.json")
E4_VIEW_ID = "ego-hmc21179183"


def parse_timestamps(value: str) -> tuple[float, ...]:
    """Parse an ordered comma-separated timestamp list without silently reordering it."""
    try:
        timestamps = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise ValueError(
            "--timestamps must be comma-separated seconds, such as 0,10,30,50"
        ) from error
    if not timestamps:
        raise ValueError("--timestamps must contain at least one timestamp")
    if (
        any(timestamp < 0 for timestamp in timestamps)
        or tuple(sorted(set(timestamps))) != timestamps
    ):
        raise ValueError("--timestamps must be non-negative, strictly increasing, and unique")
    return timestamps


def frame_reference(
    requested_proxy_seconds: float, *, fps: int, source_offset_seconds: float, frame_count: int
) -> CalibrationFrameReference:
    """Map a requested proxy time to the exact decoded analysis frame and its source time."""
    frame_index = round(requested_proxy_seconds * fps)
    if frame_index < 0 or frame_index >= frame_count:
        raise ValueError(
            f"timestamp {requested_proxy_seconds:.6f}s maps outside proxy frames [0, {frame_count})"
        )
    proxy_seconds = frame_index / fps
    return CalibrationFrameReference(
        analysis_frame_index=frame_index,
        proxy_seconds=proxy_seconds,
        analysis_seconds=proxy_seconds,
        source_seconds=source_offset_seconds + proxy_seconds,
    )


def make_calibration_id(now: datetime | None = None) -> str:
    timestamp = run_timestamp(now)
    return f"muggledsam-sam3-e4-box-calibration-{timestamp}"


def calibration_id_from_output_directory(output_directory: Path) -> str:
    """Derive a schema-compatible calibration ID from a workspace directory name."""
    calibration_id = re.sub(r"[^a-z0-9_-]", "-", output_directory.name.lower())
    if not calibration_id:
        return "calibration"
    if not calibration_id[0].isalnum():
        calibration_id = f"calibration-{calibration_id.lstrip('_-')}"
    return calibration_id


def legacy_default_view(config_path: Path) -> str | None:
    """The e4 default applies only to the original ego screen config this tool was built for."""
    return E4_VIEW_ID if Path(config_path).name == DEFAULT_CONFIG.name else None


def resolve_view_id(
    config: G2PreprocessingManifest, view_id: str | None, *, config_path: Path | None = None
) -> str:
    """The proxy view to calibrate: the named one, or the only one the config has.

    There is no built-in default view for other configs: one with several proxies (the
    all-static clip config) needs an explicit `--view`, and the chosen id must be one of its
    proxies. The legacy e4 screen config keeps its e4 default so old invocations still work.
    """
    available = [item.view_id for item in config.proxies]
    if view_id is None and config_path is not None:
        view_id = legacy_default_view(config_path)
    if view_id is None:
        if len(available) != 1:
            raise ValueError(
                "selected G2 configuration has several proxies; pass --view, one of: "
                + ", ".join(available)
            )
        return available[0]
    if view_id not in available:
        raise ValueError(
            f"selected G2 configuration does not contain {view_id} (has: {', '.join(available)})"
        )
    return view_id


def build_manifest(
    *,
    repository_root: Path,
    config_path: Path,
    timestamps: tuple[float, ...],
    result_directory: Path,
    calibration_id: str,
    view_id: str | None = None,
) -> MuggledSAMBoxCalibrationManifest:
    """Create a schema-validated calibration header for one approved proxy view.

    `view_id=None` is only legal for a single-proxy config, where it resolves to that proxy.
    """
    config = G2PreprocessingManifest.model_validate_json(config_path.read_text())
    view_id = resolve_view_id(config, view_id, config_path=config_path)
    proxy = next(item for item in config.proxies if item.view_id == view_id)
    source_offset_seconds = config.proxy_timing.source_seconds_for_frame("analysis", 0)
    for timestamp in timestamps:
        frame_reference(
            timestamp,
            fps=proxy.fps,
            source_offset_seconds=source_offset_seconds,
            frame_count=proxy.frame_count,
        )
    return MuggledSAMBoxCalibrationManifest(
        manifest_kind="muggledsam_sam3_box_calibration",
        calibration_id=calibration_id,
        base_g2_config=relative_uri(config_path, repository_root),
        base_g2_config_sha256=sha256_file(config_path),
        view_id=view_id,
        proxy=ArtifactFingerprint(
            uri=proxy.proxy_uri, sha256=proxy.checksum_sha256, source="approved_config"
        ),
        source=ArtifactFingerprint(
            uri=proxy.raw_source.raw_uri,
            sha256=proxy.raw_source.checksum_sha256,
            source="approved_config",
        ),
        proxy_dimensions=VideoDimensions(
            width=proxy.dimensions.width,
            height=proxy.dimensions.height,
        ),
        proxy_fps=proxy.fps,
        proxy_frame_count=proxy.frame_count,
        source_offset_seconds=source_offset_seconds,
        tool_version=TOOL_VERSION,
        image_representation="original_bgr",
        prompt_api="boxes_fg_points_bg_points",
        requested_proxy_timestamps_seconds=timestamps,
        result_directory_uri=relative_uri(result_directory, repository_root),
    )


def finalize_prompt(
    *,
    manifest_path: Path,
    candidate_ids: tuple[str, ...],
    proposal_path: Path,
    repository_root: Path,
    manual_seed_target_config_path: Path | None = None,
    manifest: MuggledSAMBoxCalibrationManifest | None = None,
    calibration_manifest_sha256: str | None = None,
) -> MuggledSAMProposedTrackingPromptConfig:
    """Write a non-authoritative future-tracker proposal from explicit human selections only."""
    if not candidate_ids or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("provide one or more unique --candidate-id values")
    manifest = manifest or MuggledSAMBoxCalibrationManifest.model_validate_json(
        manifest_path.read_text()
    )
    candidates = {candidate.candidate_id: candidate for candidate in manifest.candidates}
    missing = [candidate_id for candidate_id in candidate_ids if candidate_id not in candidates]
    if missing:
        raise ValueError(
            f"candidate IDs are absent from the calibration manifest: {', '.join(missing)}"
        )
    ineligible = [
        candidate_id
        for candidate_id in candidate_ids
        if not candidates[candidate_id].selected_for_finalization
        or not candidates[candidate_id].human_accepted
        or candidates[candidate_id].human_selected_candidate_index is None
        or candidates[candidate_id].frame.analysis_frame_index != 0
        or candidates[candidate_id].rejected
    ]
    if ineligible:
        raise ValueError(
            "candidate IDs must be non-rejected, human-accepted frame-0 masks explicitly "
            "marked for finalization: " + ", ".join(ineligible)
        )
    target_config: MuggledSAMManualSeedTargetConfig | None = None
    if manual_seed_target_config_path is not None:
        target_config = load_manual_seed_target_config(
            target_config_path=manual_seed_target_config_path.resolve(),
            repository_root=repository_root,
            g2_config_path=(repository_root / manifest.base_g2_config).resolve(),
            view_id=manifest.view_id,
        )
        selected_targets = [
            candidates[candidate_id].intended_target for candidate_id in candidate_ids
        ]
        if (
            len(candidate_ids) != len(target_config.targets)
            or len(set(selected_targets)) != len(target_config.targets)
            or set(selected_targets) != set(target_config.targets)
        ):
            raise ValueError(
                "finalization requires exactly "
                f"{len(target_config.targets)} distinct human-selected frame-0 masks for: "
                + ", ".join(target_config.targets)
            )
        candidates_by_target = {
            candidates[candidate_id].intended_target: candidates[candidate_id]
            for candidate_id in candidate_ids
        }
        selected_candidates = tuple(
            candidates_by_target[target] for target in target_config.targets
        )
    else:
        selected_candidates = tuple(candidates[candidate_id] for candidate_id in candidate_ids)
    seeds = tuple(
        ProposedTrackingSeed(
            candidate_id=candidate.candidate_id,
            intended_target=candidate.intended_target,
            reference_frame=candidate.frame,
            pixel_box=candidate.pixel_box,
            normalized_box=candidate.normalized_box,
            pixel_fg_points=candidate.pixel_fg_points,
            pixel_bg_points=candidate.pixel_bg_points,
            normalized_fg_points=candidate.normalized_fg_points,
            normalized_bg_points=candidate.normalized_bg_points,
            decoder_best_candidate_index=candidate.decoder_result.deterministic_best_candidate_index,
            human_selected_candidate_index=candidate.human_selected_candidate_index,
        )
        for candidate in selected_candidates
    )
    proposal = MuggledSAMProposedTrackingPromptConfig(
        manifest_kind="muggledsam_sam3_proposed_tracking_prompt",
        authority="proposed_non_authoritative",
        calibration_manifest_uri=relative_uri(manifest_path, repository_root),
        calibration_manifest_sha256=calibration_manifest_sha256 or sha256_file(manifest_path),
        view_id=manifest.view_id,
        seeds=seeds,
        tracker_initialization_limitations=(
            "This is a user-selected prompt proposal, not a ground-truth accuracy result.",
            "MuggledSAM's direct box-prompt memory API is non-multiplexed; multiple boxes "
            "cannot be assumed to initialize independent continuous tracker IDs as-is.",
            "Only human-accepted frame-0 masks are eligible; later timestamps are "
            "decoder checks and this utility did not run a tracking policy.",
            "Image-decoder IoU values are model estimates, not measured segmentation accuracy.",
        ),
        manual_seed_target_config_fingerprint=(
            ArtifactFingerprint(
                uri=relative_uri(manual_seed_target_config_path.resolve(), repository_root),
                sha256=sha256_file(manual_seed_target_config_path.resolve()),
                source="measured",
            )
            if manual_seed_target_config_path is not None
            else None
        ),
    )
    proposal_path.write_text(proposal.model_dump_json(indent=2) + "\n")
    return proposal


def load_correction_policy(
    *,
    correction_policy_path: Path,
    manual_seed_target_config_path: Path,
    repository_root: Path,
    g2_config_path: Path,
    view_id: str,
) -> MuggledSAMMultiKeyframeCorrectionPolicy:
    """Load a policy only when it is bound to the selected four-target e4 configuration."""
    policy = MuggledSAMMultiKeyframeCorrectionPolicy.model_validate_json(
        correction_policy_path.read_text()
    )
    target_config = load_manual_seed_target_config(
        target_config_path=manual_seed_target_config_path,
        repository_root=repository_root,
        g2_config_path=g2_config_path,
        view_id=view_id,
    )
    if policy.view_id != view_id or policy.targets != target_config.targets:
        raise ValueError("correction policy targets must match the selected target configuration")
    expected_uri = relative_uri(manual_seed_target_config_path.resolve(), repository_root)
    fingerprint = policy.manual_seed_target_config_fingerprint
    if fingerprint.uri != expected_uri or fingerprint.sha256 != sha256_file(
        manual_seed_target_config_path
    ):
        raise ValueError("correction policy target configuration fingerprint does not match")
    return policy


def finalize_correction_schedule(
    *,
    manifest_path: Path,
    candidate_ids: tuple[str, ...],
    schedule_path: Path,
    correction_policy_path: Path,
    manual_seed_target_config_path: Path,
    repository_root: Path,
    manifest: MuggledSAMBoxCalibrationManifest | None = None,
    calibration_manifest_sha256: str | None = None,
) -> MuggledSAMMultiKeyframeCorrectionSchedule:
    """Create a bounded, mask-fingerprinted e4 multi-keyframe correction schedule."""
    if not candidate_ids or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("provide one or more unique --candidate-id values")
    manifest = manifest or MuggledSAMBoxCalibrationManifest.model_validate_json(
        manifest_path.read_text()
    )
    policy = load_correction_policy(
        correction_policy_path=correction_policy_path.resolve(),
        manual_seed_target_config_path=manual_seed_target_config_path.resolve(),
        repository_root=repository_root,
        g2_config_path=(repository_root / manifest.base_g2_config).resolve(),
        view_id=manifest.view_id,
    )
    candidates = {candidate.candidate_id: candidate for candidate in manifest.candidates}
    missing = [candidate_id for candidate_id in candidate_ids if candidate_id not in candidates]
    if missing:
        raise ValueError(
            "candidate IDs are absent from the calibration manifest: " + ", ".join(missing)
        )

    slots = tuple(
        MuggledSAMMultiplexSlot(target_id=target, object_id=f"sam3-{slot:02d}", multiplex_slot=slot)
        for slot, target in enumerate(policy.targets)
    )
    slots_by_target = {slot.target_id: slot for slot in slots}
    corrections: list[MuggledSAMMultiKeyframeCorrection] = []
    later_per_target: dict[str, int] = {}
    for candidate_id in candidate_ids:
        candidate = candidates[candidate_id]
        if candidate.intended_target not in slots_by_target:
            raise ValueError(f"candidate target is not in the correction policy: {candidate_id}")
        if (
            candidate.rejected
            or not candidate.human_accepted
            or candidate.human_selected_candidate_index is None
        ):
            raise ValueError(
                f"candidate must be non-rejected and explicitly human-accepted: {candidate_id}"
            )
        if candidate.frame.analysis_frame_index == 0:
            if not candidate.selected_for_finalization:
                raise ValueError(
                    f"frame-0 candidate must be marked initialization eligible: {candidate_id}"
                )
        elif not candidate.selected_for_correction:
            raise ValueError(f"later candidate must be marked correction eligible: {candidate_id}")
        selected_mask = next(
            (
                item
                for item in candidate.decoder_result.candidates
                if item.candidate_index == candidate.human_selected_candidate_index
            ),
            None,
        )
        if selected_mask is None:
            raise ValueError(f"human-selected mask is absent from candidate: {candidate_id}")
        mask_path = (manifest_path.parent / selected_mask.mask_uri).resolve()
        if manifest_path.parent.resolve() not in mask_path.parents or not mask_path.is_file():
            raise ValueError(f"selected calibration mask is unavailable: {mask_path}")
        slot = slots_by_target[candidate.intended_target]
        if candidate.frame.analysis_frame_index:
            later_per_target[slot.target_id] = later_per_target.get(slot.target_id, 0) + 1
        corrections.append(
            MuggledSAMMultiKeyframeCorrection(
                candidate_id=candidate.candidate_id,
                human_selected_candidate_index=candidate.human_selected_candidate_index,
                selected_by=candidate.selected_by,
                target_id=slot.target_id,
                object_id=slot.object_id,
                multiplex_slot=slot.multiplex_slot,
                frame=candidate.frame,
                calibration_mask_fingerprint=ArtifactFingerprint(
                    uri=relative_uri(mask_path, repository_root),
                    sha256=sha256_file(mask_path),
                    source="measured",
                ),
            )
        )
    expected_targets = set(policy.targets)
    frame_zero_targets = {
        correction.target_id
        for correction in corrections
        if correction.frame.analysis_frame_index == 0
    }
    if frame_zero_targets != expected_targets or sum(
        correction.frame.analysis_frame_index == 0 for correction in corrections
    ) != len(policy.targets):
        raise ValueError(
            "correction schedule requires exactly one marked frame-0 initialization mask for: "
            + ", ".join(policy.targets)
        )
    if any(
        count > policy.maximum_later_correction_keyframes_per_target
        for count in later_per_target.values()
    ):
        raise ValueError(
            "correction schedule exceeds the policy's later correction-keyframe limit per target"
        )
    corrections.sort(key=lambda item: (item.frame.analysis_frame_index, item.multiplex_slot))
    schedule = MuggledSAMMultiKeyframeCorrectionSchedule(
        manifest_kind="muggledsam_sam3_multi_keyframe_correction_schedule",
        authority="proposed_non_authoritative",
        schedule_version="1",
        view_id=manifest.view_id,
        calibration_manifest_fingerprint=ArtifactFingerprint(
            uri=relative_uri(manifest_path, repository_root),
            sha256=calibration_manifest_sha256 or sha256_file(manifest_path),
            source="measured",
        ),
        correction_policy_fingerprint=ArtifactFingerprint(
            uri=relative_uri(correction_policy_path.resolve(), repository_root),
            sha256=sha256_file(correction_policy_path),
            source="measured",
        ),
        manual_seed_target_config_fingerprint=ArtifactFingerprint(
            uri=relative_uri(manual_seed_target_config_path.resolve(), repository_root),
            sha256=sha256_file(manual_seed_target_config_path),
            source="measured",
        ),
        correction_memory_semantics=policy.correction_memory_semantics,
        slots=slots,
        corrections=tuple(corrections),
    )
    _write_schedule(schedule_path, schedule)
    return schedule


def _write_schedule(path: Path, schedule: MuggledSAMMultiKeyframeCorrectionSchedule) -> None:
    """Atomically write an integrity-bound correction schedule."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as temporary:
        temporary.write(schedule.model_dump_json(indent=2) + "\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def _write_manifest(path: Path, manifest: MuggledSAMBoxCalibrationManifest) -> None:
    """Atomically persist validated state so a browser or terminal session can resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as temporary:
        temporary.write(manifest.model_dump_json(indent=2) + "\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def next_candidate_id(manifest: MuggledSAMBoxCalibrationManifest, frame_index: int) -> str:
    """Allocate a stable candidate ID without reusing accepted artifact names."""
    prefix = f"t{frame_index:06d}-b"
    used = [
        int(candidate.candidate_id.removeprefix(prefix))
        for candidate in manifest.candidates
        if candidate.candidate_id.startswith(prefix)
    ]
    return f"{prefix}{max(used, default=0) + 1:02d}"


def next_pending_box_id(manifest: MuggledSAMBoxCalibrationManifest, frame_index: int) -> str:
    """Allocate a stable prompt ID without reusing historical artifact suffixes."""
    prefix = f"p{frame_index:06d}-b"
    candidate_prefix = f"t{frame_index:06d}-b"
    used = [
        int(box.box_id.removeprefix(prefix))
        for box in manifest.workspace.pending_boxes
        if box.box_id.startswith(prefix)
    ]
    used.extend(
        int(candidate.candidate_id.removeprefix(candidate_prefix))
        for candidate in manifest.candidates
        if candidate.candidate_id.startswith(candidate_prefix)
    )
    return f"{prefix}{max(used, default=0) + 1:02d}"


def run_interactive(args: argparse.Namespace) -> Path:
    """Prepare a manifest then delegate only GUI/image decoding to MuggledSAM's interpreter."""
    repository_root = Path.cwd().resolve()
    config_path = args.config.resolve()
    timestamps = parse_timestamps(args.timestamps)
    calibration_id = make_calibration_id()
    output_directory = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (args.run_root / calibration_id).resolve()
    )
    manifest_path = output_directory / "calibration_manifest.json"
    if args.resume:
        if not manifest_path.is_file():
            raise ValueError(f"--resume requires an existing calibration manifest: {manifest_path}")
        manifest = MuggledSAMBoxCalibrationManifest.model_validate_json(manifest_path.read_text())
        if manifest.base_g2_config_sha256 != sha256_file(config_path):
            raise ValueError("cannot resume: selected G2 configuration fingerprint changed")
        if manifest.requested_proxy_timestamps_seconds != timestamps:
            raise ValueError("cannot resume with a different --timestamps set")
    else:
        if output_directory.exists():
            raise ValueError(
                f"output directory already exists; use --resume to continue: {output_directory}"
            )
        output_directory.mkdir(parents=True)
        manifest = build_manifest(
            repository_root=repository_root,
            config_path=config_path,
            timestamps=timestamps,
            result_directory=output_directory / "results",
            calibration_id=calibration_id,
            view_id=args.view,
        )
        _write_manifest(manifest_path, manifest)
    (output_directory / "results" / "masks").mkdir(parents=True, exist_ok=True)
    if not args.external_python.is_file():
        raise ValueError(
            f"configured MuggledSAM interpreter does not exist: {args.external_python}"
        )
    proxy_path = (repository_root / manifest.proxy.uri).resolve()
    if not proxy_path.is_file():
        raise ValueError(f"approved e4 proxy does not exist: {proxy_path}")
    if sha256_file(proxy_path) != manifest.proxy.sha256:
        raise ValueError("approved e4 proxy checksum does not match the selected G2 manifest")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(MUGGLED_SAM_SOURCE), environment["PYTHONPATH"]]
        if environment.get("PYTHONPATH")
        else [str(MUGGLED_SAM_SOURCE)]
    )
    worker_path = Path(__file__).with_name("muggled_calibration_worker.py")
    completed = subprocess.run(
        [
            str(args.external_python),
            str(worker_path),
            "--manifest",
            str(manifest_path),
            "--proxy",
            str(proxy_path),
            "--model",
            str(args.model.resolve()),
            "--timestamps-json",
            json.dumps(timestamps),
            "--device",
            args.device,
        ],
        env=environment,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"calibration worker exited {completed.returncode}; manifest is preserved"
        )
    # The external interpreter intentionally has no Battle dependency; validate its saved JSON here.
    persisted = MuggledSAMBoxCalibrationManifest.model_validate_json(manifest_path.read_text())
    _write_manifest(manifest_path, persisted)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Interactively calibrate e4 SAM3 image box prompts; never starts video tracking."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--view",
        default=None,
        help="proxy view id in --config; required when the config holds several proxies",
    )
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timestamps", default="0,10,30,50")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--external-python", type=Path, default=MUGGLED_SAM_PYTHON)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--finalize", type=Path, metavar="CALIBRATION_MANIFEST")
    parser.add_argument("--candidate-id", action="append", default=[])
    parser.add_argument("--proposal-output", type=Path)
    parser.add_argument(
        "--manual-seed-target-config",
        type=Path,
        help="Require exactly the named human-selected frame-0 masks in this target policy.",
    )
    args = parser.parse_args()
    try:
        if args.finalize is not None:
            manifest_path = args.finalize.resolve()
            proposal_path = (
                args.proposal_output.resolve()
                if args.proposal_output is not None
                else manifest_path.with_name("proposed_tracking_prompt.json")
            )
            finalize_prompt(
                manifest_path=manifest_path,
                candidate_ids=tuple(args.candidate_id),
                proposal_path=proposal_path,
                repository_root=Path.cwd().resolve(),
                manual_seed_target_config_path=args.manual_seed_target_config,
            )
            print(f"Wrote non-authoritative tracking prompt proposal: {proposal_path}")
        else:
            if (
                args.candidate_id
                or args.proposal_output is not None
                or args.manual_seed_target_config is not None
            ):
                raise ValueError(
                    "--candidate-id, --proposal-output, and --manual-seed-target-config "
                    "require --finalize"
                )
            manifest_path = run_interactive(args)
            print(f"Wrote calibration manifest: {manifest_path}")
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
