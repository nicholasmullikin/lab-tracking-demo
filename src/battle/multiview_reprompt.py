"""Consensus-driven re-prompting of one view: geometry says where, SAM3 says which pixels.

Where the eight-view consensus says a tracked view has drifted from the other cameras (an
episode that *contradicts the majority*), this tool projects the consensus sphere into that
view as box prompts plus negative points at the other parts, decodes them with the same
isolated SAM3 image decoder the calibration workspace uses, accepts one candidate by the
seed-transfer geometry rule (area band against the other views' expected area, centroid ray
within 1.5 radii of the consensus point) and writes the accepted mask into a *derived*
multi-keyframe correction schedule as an agent-authored correction (`selected_by: agent`,
provenance `multiview_consensus`).  A thin `run` step then emits the exact tracker command for
two arms: `consensus-only` (frame-0 seeds plus consensus corrections, no human correction) and
`human-plus-consensus`.

Three sub-commands, one per stage, so the GPU step is isolated:

- `plan` (CPU): consensus run -> `reprompt_plan.json` with one onset per merged contradiction.
- `decode` (GPU): one warm image-decoder worker -> `reprompt_decisions.json`, the derived
  calibration and its two schedules, and a provenance sidecar.  `--candidate-ranking ray`
  (default) keeps the seed-transfer pick (closest centroid ray); `decoder_score` keeps the same
  filters and takes the decoder's own IoU estimate first (rule (g) of the Sep 20 acceptance
  search); the ranking is recorded on every decision, provenance record and agent correction.
- `run` (CPU): the `battle-muggled-smoke` commands, resumed from the nearest checkpoint before
  the earliest onset when one exists and is consistent with the arm, else a full run.

Claim boundary: nothing here is human review.  The consensus measures disagreement between
views of one tracker, not accuracy; a correction accepted here says the other cameras agree
with the new mask, not that it is right.  CC BY-NC 4.0 applies to the dataset assets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .assembly101_camera_fit import PoseMembers
from .assembly101_clock_offset import is_ego
from .digest_cache import sha256_file
from .four_part_contract import TARGETS
from .multiview_consensus import (
    ViewRun,
    distance_to_mask_px,
    episodes_from_errors,
    load_consensus,
    load_view_run,
    relative_uri,
)
from .multiview_geometry import CameraRig
from .multiview_schemas import MultiviewConsensusManifest
from .multiview_seed_transfer import (
    BOX_MARGINS,
    dataset_hands,
    fit_first_minute_plane,
    inside_box,
    load_mask,
    mask_centroid,
    project_to_proxy,
    proxy_focal_px,
    proxy_to_raw_scale,
    ray_point_distance,
    square_box,
    view_id_for,
)
from .multiview_seed_transfer import (
    RULES as SEED_RULES,
)
from .schemas import (
    MULTIVIEW_CONSENSUS_PROVENANCE,
    MULTIVIEW_REPROMPT_ARMS,
    MULTIVIEW_REPROMPT_CANDIDATE_RANKINGS,
    MULTIVIEW_REPROMPT_MAX_ITERATIONS,
    ArtifactFingerprint,
    G2PreprocessingManifest,
    MuggledSAMBoxCalibrationManifest,
    MuggledSAMCalibrationCandidate,
    MuggledSAMImageDecoderResult,
    MuggledSAMMultiKeyframeCorrectionPolicy,
    MuggledSAMMultiKeyframeCorrectionSchedule,
    MultiviewAgentCorrection,
    MultiviewAgentCorrectionSchedule,
    MultiviewConsensusCorrectionProvenance,
    MultiviewConsensusProvenanceFile,
    MultiviewContradictionOnset,
    MultiviewDistractorGuard,
    MultiviewDistractorMark,
    MultiviewRepromptCandidateRanking,
    MultiviewRepromptDecisions,
    MultiviewRepromptDetectorConfig,
    MultiviewRepromptPlan,
    MultiviewRepromptPrompt,
    MultiviewRepromptRunCommand,
    NormalizedBox,
    NormalizedPoint,
    PixelBox,
    PixelPoint,
    RepromptCandidateScore,
    RepromptDecision,
    RunManifest,
)

OUTPUT_ROOT = Path("runs/multiview-reprompt-20260920")
DEFAULT_TARGET_VIEW = "C10379"
DEFAULT_CONSENSUS_ROOT = Path("runs/multiview-part-consensus-first-minute-r1280-pm-append")
DEFAULT_SOURCE_RUN = Path(
    "runs/sam3-memory-arms-20260919/arms/pm-append/"
    "muggledsam-sam3-four-part-static-focused-reassembly-static-c10379-20260920t033947z-r1280-pm-append"
)
DEFAULT_CORRECTION_POLICY = Path(
    "configs/muggledsam_static_four_part_reassembly_focused_correction_policy_v4.json"
)
PLAN_NAME = "reprompt_plan.json"
DECISIONS_NAME = "reprompt_decisions.json"
DECODE_RESULT_NAME = "decode_result.json"
COMMANDS_NAME = "run_commands.json"
COMMANDS_SCRIPT_NAME = "run_commands.sh"
CALIBRATION_DIR_NAME = "calibration"
PROVENANCE_NAME = "reprompt_provenance.json"
SCHEDULE_NAMES = {
    "human-plus-consensus": "multi_keyframe_correction_schedule.json",
    "consensus-only": "multi_keyframe_correction_schedule.consensus-only.json",
}
AGENT_SCHEDULE_NAME = "agent_correction_schedule.consensus-only.json"
DEFAULT_CHECKPOINT_EVERY = 300
MAX_SIDE_LENGTH = 1280
FIRST_MINUTE_FRAMES = 1800
# A negative point at another part's centroid is placed when that centroid falls inside the
# prompt box grown by this fraction on every side (the seed transfer used 0.25 at frame 0;
# a contradiction usually means the mask has leaked onto a neighbour, so the net is wider).
NEGATIVE_POINT_BOX_SLACK = 0.5

DEFAULT_DETECTOR = MultiviewRepromptDetectorConfig(
    threshold_px=40.0,
    agreement_px=30.0,
    min_agreeing_static_views=3,
    min_episode_frames=5,
    merge_gap_frames=15,
    min_run_frames=10,
    description=(
        "An onset is the first frame of a run where the target view's mask sits > 40 raw px "
        "from a consensus it was dropped from (contradicts the majority) for >= 5 consecutive "
        "frames, while >= 3 static views agree with the consensus within the builder's 30 raw "
        "px reprojection filter at that frame; contradiction runs closer than 15 frames are "
        "merged and merged runs shorter than 10 frames are ignored (the ensemble v2 fallback "
        "rule). Track A may replace these numbers with a calibrated detector file."
    ),
)

CLAIM_BOUNDARIES: tuple[str, ...] = (
    "Every correction proposed here is agent-authored (selected_by agent, provenance "
    "multiview_consensus); no human reviewed any mask it writes.",
    "The consensus measures disagreement between views of one tracker seeded by geometry, not "
    "accuracy; an accepted candidate agrees with the other cameras, which does not make it right.",
    "Dataset poses and extrinsics are external context; intrinsics and the table plane are "
    "estimates; nothing here is ground truth for any method.",
    "Assembly101 is CC BY-NC 4.0; attribution applies to every derived artifact.",
)


def fingerprint(path: Path, repository_root: Path) -> ArtifactFingerprint:
    return ArtifactFingerprint(
        uri=relative_uri(path, repository_root), sha256=sha256_file(path), source="measured"
    )


def _resolve(repository_root: Path, uri: str) -> Path:
    path = Path(uri)
    return path if path.is_absolute() else (repository_root / path).resolve()


def _verify(repository_root: Path, item: ArtifactFingerprint, what: str) -> Path:
    path = _resolve(repository_root, item.uri)
    if not path.is_file() or sha256_file(path) != item.sha256:
        raise ValueError(f"{what} is unavailable or has changed: {item.uri}")
    return path


# -- detector config ---------------------------------------------------------------------------


DETECTOR_KEYS = (
    "threshold_px",
    "agreement_px",
    "min_agreeing_static_views",
    "min_episode_frames",
    "merge_gap_frames",
    "min_run_frames",
)


def load_detector_config(path: Path, repository_root: Path) -> MultiviewRepromptDetectorConfig:
    """Read a detector file that carries the default config's keys (extra keys are ignored).

    Track A's scorecard may write such a file with calibrated thresholds; only the keys the
    gate understands are read, and the file is fingerprinted into the plan.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"detector config must be a JSON object: {path}")
    values = {key: payload[key] for key in DETECTOR_KEYS if key in payload}
    if not values:
        raise ValueError(f"detector config carries none of the keys {DETECTOR_KEYS}: {path}")
    description = str(payload.get("description") or "").strip() or (
        f"calibrated detector config from {relative_uri(path.resolve(), repository_root)} "
        f"overriding {sorted(values)}; other keys keep the defaults"
    )
    return DEFAULT_DETECTOR.model_copy(
        update={
            **values,
            "source": "file",
            "source_fingerprint": fingerprint(path.resolve(), repository_root),
            "description": description,
        }
    )


# -- onset detection (pure) --------------------------------------------------------------------


def merge_with_gaps(
    intervals: Iterable[tuple[int, int]], gap_frames: int
) -> tuple[tuple[int, int], ...]:
    """Merge half-open intervals whose gap is shorter than `gap_frames`."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if merged and start - merged[-1][1] < gap_frames:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)


@dataclass(frozen=True)
class OnsetSeed:
    """A contradiction onset before any geometry: where and how far the target view drifted."""

    target: str
    onset_frame: int
    end_frame_exclusive: int
    source_episodes: tuple[tuple[int, int], ...]
    target_error_px: float
    max_target_error_px: float
    static_views_used: int
    gate_reason: str | None  # None when the detector gate passes


def detect_onsets(
    *,
    target_view: str,
    errors: Mapping[str, Mapping[str, np.ndarray]],
    consensus_exists: Mapping[str, np.ndarray],
    dropped: Mapping[str, Mapping[str, np.ndarray]],
    used_counts: Mapping[str, np.ndarray],
    static_used_counts: Mapping[str, np.ndarray],
    config: MultiviewRepromptDetectorConfig,
    min_majority: int = 2,
) -> tuple[OnsetSeed, ...]:
    """Run the consensus builder's episode logic for one view and apply the re-prompt gate.

    `errors[target][view]`, `dropped[target][view]`, `consensus_exists[target]`,
    `used_counts[target]` and `static_used_counts[target]` are per-frame arrays as the
    consensus builder writes them.  Episodes must contradict the majority; runs are merged
    and filtered by the config; the static-majority gate is evaluated at the onset frame.
    """
    seeds: list[OnsetSeed] = []
    for target, per_view in errors.items():
        if target_view not in per_view:
            continue
        episodes = episodes_from_errors(
            {target_view: per_view[target_view]},
            consensus_exists[target],
            {target_view: dropped[target][target_view]},
            used_counts[target],
            target=target,
            threshold_px=config.threshold_px,
            min_frames=config.min_episode_frames,
            min_majority=min_majority,
        )
        contradicting = [
            (e.start_frame, e.end_frame_exclusive) for e in episodes if e.contradicts_majority
        ]
        for start, end in merge_with_gaps(contradicting, config.merge_gap_frames):
            if end - start < config.min_run_frames:
                continue
            members = tuple((s, e) for s, e in contradicting if s >= start and e <= end)
            error = per_view[target_view]
            static_used = int(static_used_counts[target][start])
            reason = None
            if start < 1:
                reason = "an onset at frame 0 cannot be corrected (frame 0 is the seed)"
            elif static_used < config.min_agreeing_static_views:
                reason = (
                    f"only {static_used} static views agree at the onset; "
                    f"{config.min_agreeing_static_views} required"
                )
            seeds.append(
                OnsetSeed(
                    target=target,
                    onset_frame=start,
                    end_frame_exclusive=end,
                    source_episodes=members,
                    target_error_px=float(error[start]),
                    max_target_error_px=float(np.nanmax(error[start:end])),
                    static_views_used=static_used,
                    gate_reason=reason,
                )
            )
    seeds.sort(key=lambda seed: (seed.onset_frame, TARGETS.index(seed.target)))
    return tuple(seeds)


# -- distractor guard (human hidden marks as evidence) -----------------------------------------


DISTRACTOR_GUARD_WINDOW_FRAMES = 60
HUMAN_RECORD_GLOB = "docs/qa/first-minute-review-anchors*.human-record.json"


def default_human_records(repository_root: Path) -> tuple[Path, ...]:
    return tuple(sorted(repository_root.glob(HUMAN_RECORD_GLOB)))


def collect_hidden_marks(
    repository_root: Path,
    records: Sequence[Path],
    *,
    rig: CameraRig,
    target_view: str,
) -> tuple[MultiviewDistractorMark, ...]:
    """Every human `hidden` mark in the records, mapped onto `target_view`'s frame clock.

    A record's view comes from its anchor config; a mapped config carries the C10379 frame each
    anchor came from (`source_analysis_frame_index`), the C10379 config is its own source. The
    C10379 frame is then mapped onto the target view through the clock rules (constant shift).
    """
    from .anchor_frames_for_view import mapped_frame
    from .review_anchors import load_config

    source_rule = rig.clock_rule(DEFAULT_TARGET_VIEW)
    target_rule = rig.clock_rule(target_view)
    marks: list[MultiviewDistractorMark] = []
    for record_path in records:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        config = load_config(_resolve(repository_root, record["config"]["uri"]))
        view_id = config.view_id
        view = view_from_view_id(view_id)
        by_frame = {frame.analysis_frame_index: frame for frame in config.frames}
        for entry in record["anchors"]:
            if entry.get("state") != "hidden":
                continue
            frame = by_frame.get(int(entry["analysis_frame_index"]))
            if frame is None:
                continue
            source_frame = (
                frame.source_analysis_frame_index
                if frame.source_analysis_frame_index is not None
                else (
                    frame.analysis_frame_index
                    if view == DEFAULT_TARGET_VIEW
                    else mapped_frame(
                        rig.clock_rule(view), source_rule, frame.analysis_frame_index
                    )[0]
                )
            )
            try:
                target_frame = mapped_frame(source_rule, target_rule, source_frame)[0]
            except ValueError:
                continue
            marks.append(
                MultiviewDistractorMark(
                    view=view,
                    view_id=view_id,
                    target=str(entry["target"]),
                    view_frame=int(entry["analysis_frame_index"]),
                    source_frame=int(source_frame),
                    target_view_frame=int(target_frame),
                    failure_case=entry.get("failure_case"),
                    note=entry.get("note"),
                )
            )
    marks.sort(key=lambda m: (m.target, m.target_view_frame, m.view))
    return tuple(marks)


def distractor_marks_near(
    marks: Sequence[MultiviewDistractorMark], target: str, frame: int, window_frames: int
) -> tuple[MultiviewDistractorMark, ...]:
    return tuple(
        m for m in marks if m.target == target and abs(m.target_view_frame - frame) <= window_frames
    )


def distractor_guard_reason(marks: Sequence[MultiviewDistractorMark], frame: int) -> str:
    where = "; ".join(
        f"{m.view} frame {m.view_frame} (C10379 clock {m.source_frame}, this view "
        f"{m.target_view_frame}{', ' + m.failure_case if m.failure_case else ''})"
        for m in marks
    )
    return (
        f"distractor_guard: human anchors mark {marks[0].target} hidden within "
        f"{max(abs(m.target_view_frame - frame) for m in marks)} frames of the onset ({where}); "
        "review evidence that a tracker here would latch onto a distractor, not ground truth"
    )


# -- prompt geometry (pure, rig injected) ------------------------------------------------------


@dataclass(frozen=True)
class OnsetGeometry:
    radius_mm: float
    expected_area_px: float
    depth_mm: float
    centroid_proxy_px: np.ndarray
    per_view_radius_mm: dict[str, float] = field(default_factory=dict)


def onset_geometry(
    rig: CameraRig,
    *,
    target_view: str,
    point_world_mm: np.ndarray,
    masks_by_view: Mapping[str, np.ndarray],
    pose_frames: Mapping[str, int | None],
    target_pose_frame: int | None = None,
) -> OnsetGeometry | None:
    """Sphere radius and expected target-view area from the other views' masks at depth.

    Each contributing view gives the equivalent-circle radius of its mask carried to
    millimetres by its depth over its proxy focal length, and its area carried into the
    target view by the squared focal/depth ratio; the medians are kept so one grazing view
    cannot inflate the prompt.  None when no view contributes.
    """
    point = np.asarray(point_world_mm, dtype=np.float64).reshape(1, 3)
    depth_target = float(rig.depth(target_view, point, target_pose_frame)[0])
    if depth_target <= 0:
        return None
    focal_target = proxy_focal_px(rig, target_view)
    radii: dict[str, float] = {}
    areas: list[float] = []
    for view, mask in masks_by_view.items():
        area = float(np.count_nonzero(mask))
        if area <= 0:
            continue
        depth = float(rig.depth(view, point, pose_frames.get(view))[0])
        if depth <= 0:
            continue
        focal = proxy_focal_px(rig, view)
        radii[view] = float(np.sqrt(area / np.pi) * depth / focal)
        areas.append(area * (focal_target * depth / (focal * depth_target)) ** 2)
    if not radii:
        return None
    centroid = project_to_proxy(rig, target_view, point, target_pose_frame)[0]
    return OnsetGeometry(
        radius_mm=float(np.median(list(radii.values()))),
        expected_area_px=float(np.median(areas)),
        depth_mm=depth_target,
        centroid_proxy_px=centroid,
        per_view_radius_mm=radii,
    )


def build_prompts(
    *,
    target: str,
    frame: int,
    centroid_proxy_px: np.ndarray,
    half_size_px: float,
    shape: tuple[int, int],
    other_centroids: Mapping[str, np.ndarray],
    hand_joints: Mapping[str, np.ndarray] | None = None,
    margins: Sequence[float] = BOX_MARGINS,
    counter_start: int = 1,
) -> tuple[MultiviewRepromptPrompt, ...]:
    """Two square boxes around the projected sphere with negatives at the other parts.

    `other_centroids` are the other parts' consensus centroids projected into the target
    view (NaN when absent); `hand_joints` are projected dataset joints per hand, used as
    extra negatives when they fall inside the box.
    """
    height, width = shape
    prompts: list[MultiviewRepromptPrompt] = []
    counter = counter_start
    for margin in margins:
        box = square_box(centroid_proxy_px, half_size_px, margin, shape)
        if box is None:
            continue
        points: list[PixelPoint] = []
        sources: list[str] = []
        seen: set[tuple[int, int]] = set()

        def add(point: np.ndarray, source: str, slack: float) -> None:
            if np.isnan(point).any() or not inside_box(point, box, slack):
                return
            x, y = int(round(float(point[0]))), int(round(float(point[1])))
            if not (0 <= x < width and 0 <= y < height) or (x, y) in seen:
                return
            seen.add((x, y))
            points.append(PixelPoint(x=x, y=y))
            sources.append(source)

        for other, other_centroid in other_centroids.items():
            if other != target:
                add(
                    np.asarray(other_centroid),
                    f"{other}_consensus_centroid",
                    NEGATIVE_POINT_BOX_SLACK,
                )
        for hand, joints in (hand_joints or {}).items():
            for index, joint in enumerate(np.asarray(joints).reshape(-1, 2)):
                add(joint, f"{hand}_joint_{index:02d}", 0.0)
        prompts.append(
            MultiviewRepromptPrompt(
                prompt_id=f"t{frame:06d}-b{counter:02d}",
                target=target,
                variant=f"consensus_sphere_box_margin_{margin:.2f}",
                pixel_box=box,
                background_points=tuple(points),
                background_point_sources=tuple(sources),
            )
        )
        counter += 1
    return tuple(prompts)


# -- source run introspection ------------------------------------------------------------------


@dataclass(frozen=True)
class SourceRun:
    directory: Path
    manifest: RunManifest
    manifest_fingerprint: ArtifactFingerprint
    profile: str
    view_id: str
    config_fingerprint: ArtifactFingerprint
    proxy_fingerprint: ArtifactFingerprint
    schedule: ArtifactFingerprint | None
    calibration_manifest: ArtifactFingerprint | None
    geometric_seed_manifest: ArtifactFingerprint | None
    checkpoint_frames: tuple[int, ...]
    correction_frames: tuple[int, ...]
    agent_schedule: ArtifactFingerprint | None = None

    @property
    def view(self) -> str:
        return view_from_view_id(self.view_id)


def view_from_view_id(view_id: str) -> str:
    """`static-c10379` -> `C10379`, `ego-hmc21179183` -> `HMC_21179183`."""
    kind, _, rest = view_id.partition("-")
    if kind == "ego":
        return "HMC_" + rest.removeprefix("hmc").upper()
    return rest.upper()


def inspect_source_run(repository_root: Path, run_directory: Path) -> SourceRun:
    directory = _resolve(repository_root, str(run_directory))
    manifest_path = directory / "manifest.json"
    manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    checkpoints = tuple(
        sorted(
            int(path.stem[1:])
            for path in (directory / "native" / "checkpoints").glob("f*.pt")
            if path.stem[1:].isdigit()
        )
    )
    if manifest.four_part_focused is not None:
        focused = manifest.four_part_focused
        corrections = focused.multi_keyframe_corrections
        schedule_path = _verify(
            repository_root, corrections.schedule_fingerprint, "source schedule"
        )
        schedule = MuggledSAMMultiKeyframeCorrectionSchedule.model_validate_json(
            schedule_path.read_text(encoding="utf-8")
        )
        return SourceRun(
            directory=directory,
            manifest=manifest,
            manifest_fingerprint=fingerprint(manifest_path, repository_root),
            profile="four_part_static_focused",
            view_id=focused.view_id,
            config_fingerprint=focused.config_fingerprint,
            proxy_fingerprint=focused.proxy_fingerprint,
            schedule=corrections.schedule_fingerprint,
            calibration_manifest=schedule.calibration_manifest_fingerprint,
            geometric_seed_manifest=None,
            checkpoint_frames=checkpoints,
            correction_frames=tuple(corrections.scheduled_correction_frame_indices),
        )
    if manifest.four_part_multiview is not None:
        multiview = manifest.four_part_multiview
        return SourceRun(
            directory=directory,
            manifest=manifest,
            manifest_fingerprint=fingerprint(manifest_path, repository_root),
            profile="four_part_multiview",
            view_id=multiview.view_id,
            config_fingerprint=multiview.config_fingerprint,
            proxy_fingerprint=multiview.proxy_fingerprint,
            schedule=None,
            calibration_manifest=None,
            geometric_seed_manifest=multiview.seed_manifest_fingerprint,
            checkpoint_frames=checkpoints,
            correction_frames=tuple(multiview.agent_correction_frames),
            agent_schedule=multiview.agent_correction_schedule_fingerprint,
        )
    raise ValueError(f"{directory} is neither a focused nor a multiview four-part run")


def resolve_recording(repository_root: Path, recording: str | None) -> Any | None:
    """The registry record for `recording`, or None when it is absent or names recording 1."""
    if recording is None:
        return None
    from .assembly101_recordings import RECORDING_1, get_recording

    record = get_recording(recording, repository_root)
    if record.recording_id == RECORDING_1.recording_id:
        return None
    return record


def proxy_for(repository_root: Path, source: SourceRun) -> Any:
    config_path = _verify(repository_root, source.config_fingerprint, "source clip config")
    config = G2PreprocessingManifest.model_validate_json(config_path.read_text(encoding="utf-8"))
    proxy = next((item for item in config.proxies if item.view_id == source.view_id), None)
    if proxy is None:
        raise ValueError(f"{source.view_id} is not in {config_path}")
    if (
        proxy.proxy_uri != source.proxy_fingerprint.uri
        or proxy.checksum_sha256 != source.proxy_fingerprint.sha256
    ):
        raise ValueError("source run proxy does not match its clip config")
    return proxy


# -- planning ---------------------------------------------------------------------------------


def next_iteration(previous: MultiviewRepromptPlan | None, requested: int | None) -> int:
    """Iteration index for a new plan; refuses to pass the cap."""
    if previous is not None and requested is not None and requested != previous.iteration + 1:
        raise ValueError(
            f"--iteration {requested} disagrees with the previous plan (iteration "
            f"{previous.iteration}); the next iteration is {previous.iteration + 1}"
        )
    iteration = (previous.iteration + 1) if previous is not None else (requested or 1)
    if iteration > MULTIVIEW_REPROMPT_MAX_ITERATIONS:
        raise ValueError(
            f"the re-prompt loop is capped at {MULTIVIEW_REPROMPT_MAX_ITERATIONS} iterations; "
            f"iteration {iteration} was requested"
        )
    if iteration < 1:
        raise ValueError("iterations start at 1")
    return iteration


def load_plan(path: Path) -> MultiviewRepromptPlan:
    return MultiviewRepromptPlan.model_validate_json(path.read_text(encoding="utf-8"))


def load_decisions(path: Path) -> MultiviewRepromptDecisions:
    return MultiviewRepromptDecisions.model_validate_json(path.read_text(encoding="utf-8"))


def iteration_directory(
    repository_root: Path, output_root: Path, view: str, iteration: int
) -> Path:
    return repository_root / output_root / view / f"iter{iteration}"


def _static_used_counts(
    archive: Mapping[str, np.ndarray], views: Sequence[str], frame_count: int
) -> dict[str, np.ndarray]:
    counts: dict[str, np.ndarray] = {}
    for target in TARGETS:
        total = np.zeros(frame_count, dtype=np.int64)
        for view in views:
            if is_ego(view):
                continue
            key = f"used/{target}/{view}"
            if key in archive:
                total += np.asarray(archive[key], dtype=bool).astype(np.int64)
        counts[target] = total
    return counts


def _target_errors_from_run(
    rig: CameraRig,
    *,
    target_view: str,
    target_run: ViewRun,
    consensus: Mapping[str, np.ndarray],
    frame_count: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Per-frame error and observation of a view the consensus was built without.

    The consensus point is projected into the view and measured against the view's own mask
    exactly as the builder does for member views (`distance_to_mask_px`, raw px).
    """
    scale = proxy_to_raw_scale(target_view)
    errors = {target: np.full(frame_count, np.nan) for target in TARGETS}
    observed = {target: np.zeros(frame_count, dtype=bool) for target in TARGETS}
    for target in TARGETS:
        if target not in target_run.targets:
            continue
        points = consensus[target]
        for frame in range(frame_count):
            if np.isnan(points[frame, 0]):
                continue
            mask = target_run.mask(frame, target)
            if mask is None:
                continue
            observed[target][frame] = True
            pose = rig.pose_frame(target_view, frame) if is_ego(target_view) else None
            projected = rig.project(target_view, points[frame].reshape(1, 3), pose)[0] / scale
            errors[target][frame] = distance_to_mask_px(mask, projected) * scale
    return errors, observed


def plan_reprompt(
    repository_root: Path,
    *,
    target_view: str = DEFAULT_TARGET_VIEW,
    consensus_root: Path = DEFAULT_CONSENSUS_ROOT,
    source_run: Path = DEFAULT_SOURCE_RUN,
    output_root: Path = OUTPUT_ROOT,
    detector: MultiviewRepromptDetectorConfig = DEFAULT_DETECTOR,
    iteration: int | None = None,
    previous_plan: Path | None = None,
    hand_negatives: bool = False,
    box_margins: Sequence[float] = BOX_MARGINS,
    overwrite: bool = False,
    recording: str | None = None,
    distractor_guard: bool = True,
    distractor_guard_window: int = DISTRACTOR_GUARD_WINDOW_FRAMES,
    distractor_guard_records: Sequence[Path] | None = None,
) -> Path:
    """Write `reprompt_plan.json` for `target_view` from a consensus run; returns its path.

    `recording` (registry label) selects another recording's rig, clock rules and poses;
    the default is recording 1, whose first minute bounds the plan at 1800 frames.  The
    distractor guard (default on, recording 1 only) blocks an onset when a human anchor record
    on any view marks the part hidden within `distractor_guard_window` frames of it.
    """
    repository_root = repository_root.resolve()
    if is_ego(target_view):
        raise ValueError("re-prompt targets are static views; ego views only feed the consensus")
    previous = load_plan(_resolve(repository_root, str(previous_plan))) if previous_plan else None
    if previous is not None and previous.target_view != target_view:
        raise ValueError("the previous plan must be for the same target view")
    iteration_index = next_iteration(previous, iteration)

    consensus_dir = _resolve(repository_root, str(consensus_root))
    manifest_path = consensus_dir / "manifest.json"
    points_path = consensus_dir / "consensus_points.npz"
    consensus: MultiviewConsensusManifest = load_consensus(manifest_path)
    source = inspect_source_run(repository_root, source_run)
    if source.view != target_view:
        raise ValueError(f"source run is for {source.view}, not the target view {target_view}")
    proxy = proxy_for(repository_root, source)
    shape = (proxy.dimensions.height, proxy.dimensions.width)
    recording_record = resolve_recording(repository_root, recording)
    recording_label = recording_record.label if recording_record is not None else None
    frame_count = (
        consensus.frame_count
        if recording_record is not None
        else min(consensus.frame_count, FIRST_MINUTE_FRAMES)
    )

    rig = (
        CameraRig.load(repository_root, recording=recording_record)
        if recording_record is not None
        else CameraRig.load(repository_root)
    )
    members = (
        PoseMembers(repository_root, recording_record)
        if recording_record is not None
        else PoseMembers(repository_root)
    )
    plane = fit_first_minute_plane(rig, members)
    sources = {item.view: item for item in consensus.sources}
    for item in consensus.sources:
        _verify(repository_root, item.observations, f"consensus source {item.view} observations")
    includes_target = target_view in sources
    reference_source = sources.get(consensus.reference_view)

    with np.load(points_path) as archive:
        views = tuple(str(v) for v in archive["views"])
        consensus_points = {t: np.asarray(archive[f"consensus/{t}"])[:frame_count] for t in TARGETS}
        used_counts = {t: np.asarray(archive[f"used_counts/{t}"])[:frame_count] for t in TARGETS}
        exists = {t: ~np.isnan(consensus_points[t][:, 0]) for t in TARGETS}
        static_used = _static_used_counts(archive, views, frame_count)
        used = {
            t: {v: np.asarray(archive[f"used/{t}/{v}"], dtype=bool)[:frame_count] for v in views}
            for t in TARGETS
        }
        all_errors = {
            t: {v: np.asarray(archive[f"error/{t}/{v}"])[:frame_count] for v in views}
            for t in TARGETS
        }
        if includes_target:
            observed = {
                t: np.asarray(archive[f"observed/{t}/{target_view}"], dtype=bool)[:frame_count]
                for t in TARGETS
            }
            target_errors = {t: all_errors[t][target_view] for t in TARGETS}
    runs: dict[str, ViewRun] = {}

    def view_run(view: str) -> ViewRun:
        if view not in runs:
            if view == target_view and not includes_target:
                directory = source.directory
            else:
                directory = _resolve(repository_root, sources[view].run_directory_uri)
            runs[view] = load_view_run(
                repository_root, directory, view=view, frame_count=frame_count
            )
        return runs[view]

    if not includes_target:
        target_errors, observed = _target_errors_from_run(
            rig,
            target_view=target_view,
            target_run=view_run(target_view),
            consensus=consensus_points,
            frame_count=frame_count,
        )
        dropped = {t: {target_view: observed[t]} for t in TARGETS}
    else:
        dropped = {t: {target_view: observed[t] & ~used[t][target_view]} for t in TARGETS}
    seeds = detect_onsets(
        target_view=target_view,
        errors={t: {target_view: target_errors[t]} for t in TARGETS},
        consensus_exists=exists,
        dropped=dropped,
        used_counts=used_counts,
        static_used_counts=static_used,
        config=detector,
        min_majority=consensus.rules.min_views,
    )

    guard: MultiviewDistractorGuard | None = None
    hidden_marks: tuple[MultiviewDistractorMark, ...] = ()
    if recording_record is None:
        record_paths = (
            tuple(_resolve(repository_root, str(p)) for p in distractor_guard_records)
            if distractor_guard_records is not None
            else default_human_records(repository_root)
        )
        if distractor_guard:
            hidden_marks = collect_hidden_marks(
                repository_root, record_paths, rig=rig, target_view=target_view
            )
        guard = MultiviewDistractorGuard(
            enabled=distractor_guard,
            window_frames=distractor_guard_window,
            records=tuple(fingerprint(p, repository_root) for p in record_paths),
            hidden_marks=hidden_marks,
            description=(
                f"an onset of a part is blocked when a human anchor record on any view marks "
                f"that part hidden within +-{distractor_guard_window} frames of it (frames "
                "mapped through the clock rules); the marks are review evidence, not truth"
                if distractor_guard
                else "disabled (--no-distractor-guard)"
            ),
        )
    suppressed: list[tuple[str, int]] = []

    onsets: list[MultiviewContradictionOnset] = []
    for seed in seeds:
        frame = seed.onset_frame
        point = consensus_points[seed.target][frame]
        views_used = tuple(v for v in views if v != target_view and used[seed.target][v][frame])
        view_error = {
            v: round(float(all_errors[seed.target][v][frame]), 2)
            for v in views
            if v != target_view and np.isfinite(all_errors[seed.target][v][frame])
        }
        common: dict[str, Any] = dict(
            target=seed.target,
            onset_frame=frame,
            interval_end_frame_exclusive=seed.end_frame_exclusive,
            source_episodes=seed.source_episodes,
            target_error_px=seed.target_error_px,
            max_target_error_px=seed.max_target_error_px,
            consensus_world_mm=tuple(float(v) for v in point),
            views_used=views_used,
            static_views_used=seed.static_views_used,
            view_error_px=view_error,
            height_above_table_mm=float(plane.signed_distance(point.reshape(1, 3))[0]),
        )
        if seed.gate_reason is not None:
            onsets.append(
                MultiviewContradictionOnset(
                    status="blocked", blocked_reason=seed.gate_reason, **common
                )
            )
            continue
        near = distractor_marks_near(hidden_marks, seed.target, frame, distractor_guard_window)
        if guard is not None and guard.enabled and near:
            suppressed.append((seed.target, frame))
            onsets.append(
                MultiviewContradictionOnset(
                    status="blocked", blocked_reason=distractor_guard_reason(near, frame), **common
                )
            )
            continue
        masks: dict[str, np.ndarray] = {}
        poses: dict[str, int | None] = {}
        for view in views_used:
            mask = view_run(view).mask(frame, seed.target)
            if mask is not None:
                masks[view] = mask
                poses[view] = rig.pose_frame(view, frame) if is_ego(view) else None
        geometry = onset_geometry(
            rig,
            target_view=target_view,
            point_world_mm=point,
            masks_by_view=masks,
            pose_frames=poses,
        )
        if geometry is None:
            onsets.append(
                MultiviewContradictionOnset(
                    status="blocked",
                    blocked_reason="no other view's mask gives the part a size at this frame",
                    **common,
                )
            )
            continue
        centroid = geometry.centroid_proxy_px
        common.update(
            radius_mm=geometry.radius_mm,
            depth_mm=geometry.depth_mm,
            expected_area_px=geometry.expected_area_px,
            projected_centroid_proxy_px=(
                None if np.isnan(centroid).any() else (float(centroid[0]), float(centroid[1]))
            ),
        )
        in_frame = (
            not np.isnan(centroid).any()
            and 0 <= centroid[0] < shape[1]
            and 0 <= centroid[1] < shape[0]
        )
        if not in_frame:
            onsets.append(
                MultiviewContradictionOnset(
                    status="blocked",
                    blocked_reason="projected consensus centroid falls outside the target view",
                    **common,
                )
            )
            continue
        other_centroids = {
            other: project_to_proxy(rig, target_view, consensus_points[other][frame].reshape(1, 3))[
                0
            ]
            for other in TARGETS
            if other != seed.target and exists[other][frame]
        }
        hands = None
        if hand_negatives:
            hands = {
                name: project_to_proxy(rig, target_view, joints)
                for name, joints in dataset_hands(
                    members, rig.pose_frame(target_view, frame)
                ).items()
            }
        half_size = geometry.radius_mm * proxy_focal_px(rig, target_view) / geometry.depth_mm
        prompts = build_prompts(
            target=seed.target,
            frame=frame,
            centroid_proxy_px=centroid,
            half_size_px=half_size,
            shape=shape,
            other_centroids=other_centroids,
            hand_joints=hands,
            margins=box_margins,
        )
        if not prompts:
            onsets.append(
                MultiviewContradictionOnset(
                    status="blocked",
                    blocked_reason="projected footprint is degenerate in the target view",
                    **common,
                )
            )
            continue
        onsets.append(MultiviewContradictionOnset(status="planned", prompts=prompts, **common))

    reference_note = reference_dependency_note(
        consensus, includes_target=includes_target, target_view=target_view
    )
    plan = MultiviewRepromptPlan(
        manifest_kind="multiview_reprompt_plan",
        target_view=target_view,
        target_view_id=view_id_for(target_view),
        iteration=iteration_index,
        previous_plan=(
            fingerprint(_resolve(repository_root, str(previous_plan)), repository_root)
            if previous_plan
            else None
        ),
        consensus_root_uri=relative_uri(consensus_dir, repository_root),
        consensus_manifest=fingerprint(manifest_path, repository_root),
        consensus_points=fingerprint(points_path, repository_root),
        consensus_includes_target_view=includes_target,
        consensus_reference_run_uri=(
            reference_source.run_directory_uri if reference_source is not None else None
        ),
        reference_dependency_note=reference_note,
        source_run_uri=relative_uri(source.directory, repository_root),
        source_run_manifest=source.manifest_fingerprint,
        source_run_profile=source.profile,  # type: ignore[arg-type]
        source_schedule=source.schedule,
        source_agent_schedule=source.agent_schedule,
        source_calibration_manifest=source.calibration_manifest,
        source_geometric_seed_manifest=source.geometric_seed_manifest,
        source_checkpoint_frames=source.checkpoint_frames,
        source_correction_frames=source.correction_frames,
        clip_config=source.config_fingerprint,
        proxy=source.proxy_fingerprint,
        proxy_dimensions=(proxy.dimensions.width, proxy.dimensions.height),
        proxy_to_raw_scale=proxy_to_raw_scale(target_view),
        frame_count=frame_count,
        detector=detector,
        box_margins=tuple(float(m) for m in box_margins),
        hand_negatives=hand_negatives,
        recording_label=recording_label,
        distractor_guard=(
            guard.model_copy(update={"suppressed_onsets": tuple(suppressed)})
            if guard is not None
            else None
        ),
        onsets=tuple(onsets),
        claim_boundaries=CLAIM_BOUNDARIES,
    )
    directory = iteration_directory(repository_root, output_root, target_view, iteration_index)
    path = directory / PLAN_NAME
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(plan.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def reference_dependency_note(
    consensus: MultiviewConsensusManifest, *, includes_target: bool, target_view: str
) -> str:
    seeds = ", ".join(
        f"{s.view} ({s.seed_provenance})" for s in consensus.sources if s.view != target_view
    )
    if includes_target:
        return (
            f"The consensus includes {target_view} itself as a member (its run "
            f"{next(s.run_directory_uri for s in consensus.sources if s.view == target_view)}, "
            "which carried the human corrections): consensus points near frames where "
            f"{target_view} agreed with the majority partly rest on the human-corrected masks, "
            "and the other views' frame-0 seeds were transferred from the C10379 human masks "
            f"(sources: {seeds}). A consensus rebuilt with --exclude-view {target_view} removes "
            "the first dependency but not the second."
        )
    return (
        f"The consensus was built without {target_view} (sources: {seeds}); its points do not "
        f"use {target_view}'s masks. The other views' frame-0 seeds were still transferred from "
        "the C10379 human frame-0 masks (geometric_seed_transfer), so the loop is free of the "
        "human *corrections* but not of the human *seeds*."
    )


# -- decoding and acceptance -------------------------------------------------------------------


def decode_requests(
    plan: MultiviewRepromptPlan, candidate_ids: Mapping[str, str]
) -> list[dict[str, Any]]:
    """Worker `batch_decode` prompts (normalised) for every planned onset prompt."""
    width, height = plan.proxy_dimensions
    requests: list[dict[str, Any]] = []
    for onset in plan.planned_onsets:
        for prompt in onset.prompts:
            box = prompt.pixel_box
            candidate_id = candidate_ids[prompt.prompt_id]
            suffix = int(candidate_id.rsplit("b", 1)[1])
            requests.append(
                {
                    "box_id": f"p{onset.onset_frame:06d}-b{suffix:02d}",
                    "candidate_id": candidate_id,
                    "prompt_id": prompt.prompt_id,
                    "frame_index": onset.onset_frame,
                    "pixel_box": box.model_dump(mode="json"),
                    "intended_target": prompt.target,
                    "boxes": [
                        [[box.x1 / width, box.y1 / height], [box.x2 / width, box.y2 / height]]
                    ],
                    "fg_points": [],
                    "bg_points": [[p.x / width, p.y / height] for p in prompt.background_points],
                }
            )
    return requests


def score_candidates(
    rig: CameraRig,
    *,
    target_view: str,
    onset: MultiviewContradictionOnset,
    candidates: Sequence[tuple[RepromptCandidateScore, np.ndarray]],
    max_ray_radii: float = SEED_RULES.max_centroid_ray_distance_radii,
    min_area_ratio: float = SEED_RULES.min_area_ratio,
    max_area_ratio: float = SEED_RULES.max_area_ratio,
    ranking: MultiviewRepromptCandidateRanking = "ray",
) -> RepromptDecision:
    """The seed-transfer acceptance filters (area band, centroid ray), then one ranking.

    `ray` (default): closest centroid ray wins, ties by the decoder's IoU estimate.
    `decoder_score`: highest decoder IoU estimate wins, ties by the ray. The filters are
    identical; only the pick among passing candidates differs, and the decision records which.
    """
    if ranking not in MULTIVIEW_REPROMPT_CANDIDATE_RANKINGS:
        raise ValueError(
            f"unknown candidate ranking {ranking!r}; one of {MULTIVIEW_REPROMPT_CANDIDATE_RANKINGS}"
        )
    point = np.asarray(onset.consensus_world_mm, dtype=np.float64)
    radius = onset.radius_mm or 0.0
    expected = onset.expected_area_px
    scored: list[RepromptCandidateScore] = []
    for candidate, mask in candidates:
        notes: list[str] = []
        area = int(np.count_nonzero(mask))
        passed = True
        ratio = (area / expected) if expected else None
        ray_mm: float | None = None
        ray_radii: float | None = None
        if area == 0:
            passed = False
            notes.append("empty mask")
        else:
            ray_mm = ray_point_distance(rig, target_view, mask_centroid(mask), point, None)
            if np.isinf(ray_mm):
                ray_mm = None
                passed = False
                notes.append("centroid ray points away from the consensus point")
            else:
                ray_radii = ray_mm / radius if radius > 0 else None
        if ratio is None:
            passed = False
            notes.append("no expected area")
        elif not min_area_ratio <= ratio <= max_area_ratio:
            passed = False
            notes.append(f"area ratio {ratio:.2f} outside [{min_area_ratio}, {max_area_ratio}]")
        if ray_radii is None and area > 0 and ray_mm is not None:
            passed = False
            notes.append("no radius to compare the centroid ray against")
        elif ray_radii is not None and ray_radii > max_ray_radii:
            passed = False
            notes.append(
                f"centroid ray {ray_mm:.0f} mm = {ray_radii:.2f} radii from the consensus point "
                f"(limit {max_ray_radii})"
            )
        scored.append(
            candidate.model_copy(
                update={
                    "mask_area_px": area,
                    "area_ratio_vs_expected": ratio,
                    "centroid_ray_distance_mm": ray_mm,
                    "centroid_ray_distance_radii": ray_radii,
                    "passed": passed,
                    "notes": tuple(notes),
                }
            )
        )
    passing = [c for c in scored if c.passed]
    if passing:
        if ranking == "decoder_score":
            best = max(
                passing,
                key=lambda c: (c.decoder_iou_estimate, -(c.centroid_ray_distance_mm or 0.0)),
            )
            how = (
                f"best by decoder IoU {best.decoder_iou_estimate:.2f} (centroid ray "
                f"{best.centroid_ray_distance_mm or 0.0:.0f} mm = "
                f"{best.centroid_ray_distance_radii or 0.0:.2f} radii), area x"
                f"{best.area_ratio_vs_expected or 0.0:.2f}"
            )
        else:
            best = max(
                passing,
                key=lambda c: (-(c.centroid_ray_distance_mm or 0.0), c.decoder_iou_estimate),
            )
            how = (
                f"best by centroid ray {best.centroid_ray_distance_mm or 0.0:.0f} mm "
                f"({best.centroid_ray_distance_radii or 0.0:.2f} radii), area x"
                f"{best.area_ratio_vs_expected or 0.0:.2f}, decoder IoU "
                f"{best.decoder_iou_estimate:.2f}"
            )
        return RepromptDecision(
            target=onset.target,
            onset_frame=onset.onset_frame,
            decision="accepted",
            reason=f"{len(passing)} of {len(scored)} candidates pass; {how}",
            accepted=best,
            candidates=tuple(scored),
            candidate_ranking=ranking,
        )
    best_ray = min(
        (c.centroid_ray_distance_radii for c in scored if c.centroid_ray_distance_radii),
        default=None,
    )
    return RepromptDecision(
        target=onset.target,
        onset_frame=onset.onset_frame,
        decision="rejected",
        reason=(
            f"no candidate passed the acceptance rule over {len(scored)} candidates"
            + (f" (closest centroid ray {best_ray:.2f} radii)" if best_ray is not None else "")
        ),
        candidates=tuple(scored),
        candidate_ranking=ranking,
    )


def _worker(
    *,
    proxy_path: Path,
    results_directory: Path,
    stderr_path: Path,
    external_python: Path | None,
    model: Path | None,
    device: str,
) -> Any:
    from .muggled_calibration_web import WorkerClient
    from .muggled_smoke import DEFAULT_MODEL, MUGGLED_SAM_PYTHON, MUGGLED_SAM_SOURCE

    external_python = external_python or MUGGLED_SAM_PYTHON
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(MUGGLED_SAM_SOURCE), environment["PYTHONPATH"]]
        if environment.get("PYTHONPATH")
        else [str(MUGGLED_SAM_SOURCE)]
    )
    return WorkerClient(
        [
            str(external_python),
            str(Path(__file__).with_name("muggled_calibration_worker.py")),
            "--serve-jsonl",
            "--proxy",
            str(proxy_path),
            "--model",
            str((model or DEFAULT_MODEL).resolve()),
            "--results-directory",
            str(results_directory),
            "--device",
            device,
        ],
        environment=environment,
        stderr_path=stderr_path,
    )


def _normalized_box(box: PixelBox, width: int, height: int) -> NormalizedBox:
    return NormalizedBox(
        x=box.x1 / width,
        y=box.y1 / height,
        width=(box.x2 - box.x1) / width,
        height=(box.y2 - box.y1) / height,
    )


def _normalized_points(
    points: Iterable[PixelPoint], width: int, height: int
) -> tuple[NormalizedPoint, ...]:
    return tuple(NormalizedPoint(x=p.x / width, y=p.y / height) for p in points)


def _consensus_ids_carried(
    calibration_dir: Path,
) -> tuple[MultiviewConsensusProvenanceFile | None, tuple[str, ...]]:
    path = calibration_dir / PROVENANCE_NAME
    if not path.is_file():
        return None, ()
    carried = MultiviewConsensusProvenanceFile.model_validate_json(path.read_text(encoding="utf-8"))
    return carried, carried.candidate_ids


def decode_plan(
    plan_path: Path,
    *,
    repository_root: Path,
    decoder: Any | None = None,
    external_python: Path | None = None,
    model: Path | None = None,
    device: str = "cuda:0",
    correction_policy: Path = DEFAULT_CORRECTION_POLICY,
    rig: CameraRig | None = None,
    candidate_ranking: MultiviewRepromptCandidateRanking = "ray",
) -> Path:
    """Decode every planned prompt with one warm worker, accept, derive the schedules.

    GPU step (the worker is the calibration workspace's isolated image decoder); with an
    injected `decoder` (tests) nothing touches a GPU.  Writes, beside the plan:
    `decode_result.json`, `reprompt_decisions.json`, and when the source run applied a
    correction schedule, `calibration/` (derived calibration with the new agent candidates,
    the `human-plus-consensus` and `consensus-only` schedules, `reprompt_provenance.json`,
    `agent_acceptances.jsonl`).  `candidate_ranking` (see `score_candidates`) is recorded on
    every decision, every provenance record, every agent correction and the decisions file.
    """
    if candidate_ranking not in MULTIVIEW_REPROMPT_CANDIDATE_RANKINGS:
        raise ValueError(
            f"unknown candidate ranking {candidate_ranking!r}; one of "
            f"{MULTIVIEW_REPROMPT_CANDIDATE_RANKINGS}"
        )
    from .muggled_agent_correction import derive_calibration
    from .muggled_calibration import (
        finalize_correction_schedule,
        frame_reference,
        next_candidate_id,
    )

    started = time.monotonic()
    repository_root = repository_root.resolve()
    plan_path = plan_path.resolve()
    plan = load_plan(plan_path)
    iteration_dir = plan_path.parent
    if rig is None:
        record = resolve_recording(repository_root, plan.recording_label)
        rig = (
            CameraRig.load(repository_root, recording=record)
            if record is not None
            else CameraRig.load(repository_root)
        )
    width, height = plan.proxy_dimensions

    schedule_capable = (
        plan.source_calibration_manifest is not None and plan.source_schedule is not None
    )
    agent_schedule_capable = (
        not schedule_capable
        and plan.source_run_profile == "four_part_multiview"
        and plan.source_geometric_seed_manifest is not None
    )
    blocked_reason: str | None = None
    calibration_dir: Path | None = None
    manifest: MuggledSAMBoxCalibrationManifest | None = None
    manifest_path: Path | None = None
    if agent_schedule_capable:
        # A geometry-seeded run has no human calibration to derive from; its corrections go
        # into a `MultiviewAgentCorrectionSchedule` bound to the seed manifest instead.
        calibration_dir = iteration_dir / CALIBRATION_DIR_NAME
        results_directory = calibration_dir / "results"
        results_directory.mkdir(parents=True, exist_ok=True)
    elif schedule_capable:
        source_manifest_path = _verify(
            repository_root,
            plan.source_calibration_manifest,
            "source calibration manifest",  # type: ignore[arg-type]
        )
        calibration_dir = iteration_dir / CALIBRATION_DIR_NAME
        manifest_path = derive_calibration(
            source_dir=source_manifest_path.parent,
            output_dir=calibration_dir,
            repository_root=repository_root,
        )
        manifest = MuggledSAMBoxCalibrationManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        if manifest.view_id != plan.target_view_id:
            raise ValueError("the source calibration is not for the target view")
        results_directory = calibration_dir / "results"
    else:
        blocked_reason = (
            f"the source run ({plan.source_run_profile}) applied no correction schedule, so there "
            "is no calibration to derive a schedule from; the multiview first-minute profile "
            "takes only its geometric seed manifest and the schedule/policy contracts name "
            "static-c10379 and the two ego views only. Decisions and masks are written; the run "
            "step needs a schedule-capable profile for this view."
        )
        results_directory = iteration_dir / "results"
        results_directory.mkdir(parents=True, exist_ok=True)

    # Candidate IDs: unique within the derived calibration (human candidates already occupy
    # `t000327-b01..` etc.), reserved in plan order so the worker's batch never collides.
    candidate_ids: dict[str, str] = {}
    working = manifest
    for onset in plan.planned_onsets:
        for prompt in onset.prompts:
            if working is not None:
                candidate_id = next_candidate_id(working, onset.onset_frame)
                working = working.model_copy(
                    update={
                        "candidates": (
                            *working.candidates,
                            _placeholder_candidate(
                                working, candidate_id, prompt, onset, width, height
                            ),
                        )
                    }
                )
            else:
                taken = sum(
                    1 for v in candidate_ids.values() if v.startswith(f"t{onset.onset_frame:06d}-")
                )
                candidate_id = f"t{onset.onset_frame:06d}-b{taken + 1:02d}"
            candidate_ids[prompt.prompt_id] = candidate_id
    requests = decode_requests(plan, candidate_ids)
    if not requests:
        raise ValueError("the plan has no planned onsets to decode")

    owns = decoder is None
    if decoder is None:
        proxy_path = _verify(repository_root, plan.proxy, "target proxy")
        decoder = _worker(
            proxy_path=proxy_path,
            results_directory=results_directory,
            stderr_path=iteration_dir / "decode_worker.stderr.log",
            external_python=external_python,
            model=model,
            device=device,
        )
    decoded: list[dict[str, Any]] = []
    try:
        for frame in sorted({int(r["frame_index"]) for r in requests}):
            batch = [r for r in requests if int(r["frame_index"]) == frame]
            decoder.request("frame_preview", {"frame_index": frame}, timeout=600)
            response = decoder.request("batch_decode", {"prompts": batch}, timeout=900)
            decoded.extend(response["decoded"])
    finally:
        if owns:
            decoder.close()
    (iteration_dir / DECODE_RESULT_NAME).write_text(
        json.dumps(
            {"target_view": plan.target_view, "prompt_count": len(requests), "decoded": decoded},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    decoded_by_id = {item["candidate_id"]: item["decoder_result"] for item in decoded}
    mask_root = results_directory.parent

    # Acceptance per onset, then the policy's per-target keyframe budget in onset order.
    policy_path = _resolve(repository_root, str(correction_policy))
    policy = MuggledSAMMultiKeyframeCorrectionPolicy.model_validate_json(
        policy_path.read_text(encoding="utf-8")
    )
    source_schedule: MuggledSAMMultiKeyframeCorrectionSchedule | None = None
    source_later: list[Any] = []
    dropped_out_of_range: tuple[int, ...] = ()
    if schedule_capable:
        schedule_path = _verify(repository_root, plan.source_schedule, "source schedule")  # type: ignore[arg-type]
        source_schedule = MuggledSAMMultiKeyframeCorrectionSchedule.model_validate_json(
            schedule_path.read_text(encoding="utf-8")
        )
        source_later = [c for c in source_schedule.corrections if c.frame.analysis_frame_index != 0]
        dropped_out_of_range = tuple(
            sorted(
                {
                    c.frame.analysis_frame_index
                    for c in source_later
                    if c.frame.analysis_frame_index >= plan.frame_count
                }
            )
        )
        source_later = [c for c in source_later if c.frame.analysis_frame_index < plan.frame_count]
    later_per_target: dict[str, int] = {}
    for correction in source_later:
        later_per_target[correction.target_id] = later_per_target.get(correction.target_id, 0) + 1
    carried_agent: list[MultiviewAgentCorrection] = []
    if agent_schedule_capable and plan.source_agent_schedule is not None:
        previous_schedule = MultiviewAgentCorrectionSchedule.model_validate_json(
            _verify(repository_root, plan.source_agent_schedule, "source agent schedule").read_text(
                encoding="utf-8"
            )
        )
        carried_agent = [
            c for c in previous_schedule.corrections if c.analysis_frame_index < plan.frame_count
        ]
        for correction in carried_agent:
            later_per_target[correction.target] = later_per_target.get(correction.target, 0) + 1

    decisions: list[RepromptDecision] = []
    accepted_ids: dict[tuple[str, int], str] = {}
    for onset in plan.planned_onsets:
        candidates: list[tuple[RepromptCandidateScore, np.ndarray]] = []
        for prompt in onset.prompts:
            result = decoded_by_id.get(candidate_ids[prompt.prompt_id])
            if result is None:
                continue
            for candidate in result["candidates"]:
                mask_path = (mask_root / candidate["mask_uri"]).resolve()
                mask = load_mask(mask_path)
                candidates.append(
                    (
                        RepromptCandidateScore(
                            prompt_id=prompt.prompt_id,
                            calibration_candidate_id=candidate_ids[prompt.prompt_id],
                            candidate_index=int(candidate["candidate_index"]),
                            mask=fingerprint(mask_path, repository_root),
                            decoder_iou_estimate=float(candidate["iou_score"]),
                            mask_area_px=int(mask.sum()),
                            passed=False,
                        ),
                        mask,
                    )
                )
        decision = score_candidates(
            rig,
            target_view=plan.target_view,
            onset=onset,
            candidates=candidates,
            ranking=candidate_ranking,
        )
        if decision.decision == "accepted" and (schedule_capable or agent_schedule_capable):
            count = later_per_target.get(onset.target, 0) + 1
            if count > policy.maximum_later_correction_keyframes_per_target:
                decision = decision.model_copy(
                    update={
                        "decision": "rejected",
                        "accepted": None,
                        "reason": (
                            f"policy_keyframe_limit: {onset.target} already has "
                            f"{count - 1} later corrections in the human-plus-consensus arm; "
                            "policy "
                            f"{policy.policy_id} v{policy.policy_version} allows "
                            f"{policy.maximum_later_correction_keyframes_per_target}. Candidate "
                            f"{decision.accepted.calibration_candidate_id} "
                            "passed the acceptance rule but is not scheduled."
                        ),
                    }
                )
            else:
                later_per_target[onset.target] = count
        if decision.decision == "accepted" and decision.accepted is not None:
            accepted_ids[(onset.target, onset.onset_frame)] = (
                decision.accepted.calibration_candidate_id or ""
            )
        decisions.append(decision)

    schedules: dict[str, ArtifactFingerprint] = {}
    provenance_uri: str | None = None
    derived_fingerprint: ArtifactFingerprint | None = None
    if schedule_capable:
        assert manifest is not None and manifest_path is not None and calibration_dir is not None
        assert source_schedule is not None
        accepted_by_id = {v: k for k, v in accepted_ids.items()}
        candidates_out = list(manifest.candidates)
        timestamps = set(manifest.requested_proxy_timestamps_seconds)
        for onset in plan.planned_onsets:
            decision = next(
                d
                for d in decisions
                if (d.target, d.onset_frame) == (onset.target, onset.onset_frame)
            )
            frame = frame_reference(
                onset.onset_frame / manifest.proxy_fps,
                fps=manifest.proxy_fps,
                source_offset_seconds=manifest.source_offset_seconds,
                frame_count=manifest.proxy_frame_count,
            )
            timestamps.add(frame.proxy_seconds)
            for prompt in onset.prompts:
                candidate_id = candidate_ids[prompt.prompt_id]
                result = decoded_by_id.get(candidate_id)
                if result is None:
                    continue
                accepted_here = (
                    decision.accepted is not None
                    and decision.accepted.calibration_candidate_id == candidate_id
                    and candidate_id in accepted_by_id
                )
                candidates_out.append(
                    MuggledSAMCalibrationCandidate(
                        candidate_id=candidate_id,
                        intended_target=prompt.target,
                        frame=frame,
                        pixel_box=prompt.pixel_box,
                        normalized_box=_normalized_box(prompt.pixel_box, width, height),
                        pixel_bg_points=prompt.background_points,
                        normalized_bg_points=_normalized_points(
                            prompt.background_points, width, height
                        ),
                        decoder_result=MuggledSAMImageDecoderResult.model_validate(result),
                        selected_by="agent",
                        human_selected_candidate_index=(
                            decision.accepted.candidate_index
                            if accepted_here and decision.accepted
                            else None
                        ),
                        human_accepted=accepted_here,
                        selected_for_correction=accepted_here,
                    )
                )
        human_plus_path = calibration_dir / SCHEDULE_NAMES["human-plus-consensus"]
        consensus_only_path = calibration_dir / SCHEDULE_NAMES["consensus-only"]
        final_manifest = manifest.model_copy(
            update={
                "candidates": tuple(candidates_out),
                "requested_proxy_timestamps_seconds": tuple(sorted(timestamps)),
                "final_correction_schedule_uri": relative_uri(human_plus_path, repository_root),
                "plan_revision": 1,
            }
        )
        manifest_bytes = (final_manifest.model_dump_json(indent=2) + "\n").encode()
        manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
        carried, carried_ids = _consensus_ids_carried(source_manifest_path.parent)
        source_ids = [
            c.candidate_id for c in source_schedule.corrections if c.frame.analysis_frame_index == 0
        ]
        source_ids += [c.candidate_id for c in source_later]
        new_ids = [
            accepted_ids[key]
            for key in sorted(accepted_ids, key=lambda k: (k[1], TARGETS.index(k[0])))
        ]
        arm_ids = {
            "human-plus-consensus": tuple(source_ids + new_ids),
            "consensus-only": tuple(
                [
                    c.candidate_id
                    for c in source_schedule.corrections
                    if c.frame.analysis_frame_index == 0
                ]
                + [c.candidate_id for c in source_later if c.candidate_id in carried_ids]
                + new_ids
            ),
        }
        target_config_path = _verify(
            repository_root, source_schedule.manual_seed_target_config_fingerprint, "target config"
        )
        # The manifest bytes are hashed once and both schedules bind to that hash; the bytes are
        # written last so neither schedule can name a manifest that differs from the file.
        manifest_path.write_bytes(manifest_bytes)
        for arm, path in (
            (("human-plus-consensus", human_plus_path)),
            ("consensus-only", consensus_only_path),
        ):
            finalize_correction_schedule(
                manifest_path=manifest_path,
                candidate_ids=arm_ids[arm],
                schedule_path=path,
                correction_policy_path=policy_path,
                manual_seed_target_config_path=target_config_path,
                repository_root=repository_root,
                manifest=final_manifest,
                calibration_manifest_sha256=manifest_sha,
            )
            schedules[arm] = fingerprint(path, repository_root)
        derived_fingerprint = ArtifactFingerprint(
            uri=relative_uri(manifest_path, repository_root), sha256=manifest_sha, source="measured"
        )
        provenance_path = calibration_dir / PROVENANCE_NAME
        records = list(carried.corrections) if carried is not None else []
        log_lines: list[str] = []
        for decision in decisions:
            if decision.decision != "accepted" or decision.accepted is None:
                continue
            onset = next(
                o
                for o in plan.onsets
                if (o.target, o.onset_frame) == (decision.target, decision.onset_frame)
            )
            records.append(
                MultiviewConsensusCorrectionProvenance(
                    candidate_id=decision.accepted.calibration_candidate_id or "",
                    target=decision.target,
                    onset_frame=decision.onset_frame,
                    iteration=plan.iteration,
                    consensus_manifest=plan.consensus_manifest,
                    consensus_includes_target_view=plan.consensus_includes_target_view,
                    consensus_reference_run_uri=plan.consensus_reference_run_uri,
                    reference_dependency_note=plan.reference_dependency_note,
                    views_used=onset.views_used,
                    target_error_px=onset.target_error_px,
                    accepted=decision.accepted,
                    rejected_alternatives=tuple(
                        c for c in decision.candidates if c is not decision.accepted
                    ),
                    mask=decision.accepted.mask,
                    candidate_ranking=decision.candidate_ranking,
                )
            )
            log_lines.append(_acceptance_log_line(decision, plan.iteration))
        provenance_path.write_text(
            MultiviewConsensusProvenanceFile(
                manifest_kind="multiview_consensus_correction_provenance",
                derived_calibration_manifest=derived_fingerprint,
                carried_from=(
                    fingerprint(source_manifest_path.parent / PROVENANCE_NAME, repository_root)
                    if carried is not None
                    else None
                ),
                corrections=tuple(records),
            ).model_dump_json(indent=2)
            + "\n",
            encoding="utf-8",
        )
        provenance_uri = relative_uri(provenance_path, repository_root)
        if log_lines:
            with (calibration_dir / "agent_acceptances.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write("\n".join(log_lines) + "\n")

    if agent_schedule_capable:
        assert calibration_dir is not None
        new_records: list[MultiviewAgentCorrection] = []
        provenance_records: list[MultiviewConsensusCorrectionProvenance] = []
        log_lines = []
        for decision in decisions:
            if decision.decision != "accepted" or decision.accepted is None:
                continue
            onset = next(
                o
                for o in plan.onsets
                if (o.target, o.onset_frame) == (decision.target, decision.onset_frame)
            )
            new_records.append(
                MultiviewAgentCorrection(
                    target=decision.target,
                    multiplex_slot=_seed_slot(
                        repository_root, plan.source_geometric_seed_manifest, decision.target
                    ),
                    analysis_frame_index=decision.onset_frame,
                    candidate_id=decision.accepted.calibration_candidate_id or "",
                    candidate_index=decision.accepted.candidate_index,
                    mask=decision.accepted.mask,
                    iteration=plan.iteration,
                    candidate_ranking=decision.candidate_ranking,
                )
            )
            provenance_records.append(
                MultiviewConsensusCorrectionProvenance(
                    candidate_id=decision.accepted.calibration_candidate_id or "",
                    target=decision.target,
                    onset_frame=decision.onset_frame,
                    iteration=plan.iteration,
                    consensus_manifest=plan.consensus_manifest,
                    consensus_includes_target_view=plan.consensus_includes_target_view,
                    consensus_reference_run_uri=plan.consensus_reference_run_uri,
                    reference_dependency_note=plan.reference_dependency_note,
                    views_used=onset.views_used,
                    target_error_px=onset.target_error_px,
                    accepted=decision.accepted,
                    rejected_alternatives=tuple(
                        c for c in decision.candidates if c is not decision.accepted
                    ),
                    mask=decision.accepted.mask,
                    candidate_ranking=decision.candidate_ranking,
                )
            )
            log_lines.append(_acceptance_log_line(decision, plan.iteration))
        if new_records:
            provenance_path = calibration_dir / PROVENANCE_NAME
            provenance_path.write_text(
                MultiviewConsensusProvenanceFile(
                    manifest_kind="multiview_consensus_correction_provenance",
                    derived_calibration_manifest=None,
                    carried_from=None,
                    corrections=tuple(provenance_records),
                ).model_dump_json(indent=2)
                + "\n",
                encoding="utf-8",
            )
            provenance_uri = relative_uri(provenance_path, repository_root)
            with (calibration_dir / "agent_acceptances.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write("\n".join(log_lines) + "\n")
            schedule_path = calibration_dir / AGENT_SCHEDULE_NAME
            schedule_path.write_text(
                MultiviewAgentCorrectionSchedule(
                    manifest_kind="multiview_agent_correction_schedule",
                    view_id=plan.target_view_id,
                    clip_config=plan.clip_config,
                    proxy=plan.proxy,
                    seed_manifest=plan.source_geometric_seed_manifest,  # type: ignore[arg-type]
                    plan=fingerprint(plan_path, repository_root),
                    provenance_file=fingerprint(provenance_path, repository_root),
                    corrections=tuple(
                        sorted(
                            (*carried_agent, *new_records),
                            key=lambda c: (c.analysis_frame_index, c.multiplex_slot),
                        )
                    ),
                    claim_boundaries=CLAIM_BOUNDARIES,
                ).model_dump_json(indent=2)
                + "\n",
                encoding="utf-8",
            )
            schedules["consensus-only"] = fingerprint(schedule_path, repository_root)
        else:
            blocked_reason = (
                "no correction was accepted, so no agent correction schedule was written for "
                "this geometry-seeded run"
            )

    accepted_count = sum(d.decision == "accepted" for d in decisions)
    output = MultiviewRepromptDecisions(
        manifest_kind="multiview_reprompt_decisions",
        plan=fingerprint(plan_path, repository_root),
        target_view=plan.target_view,
        iteration=plan.iteration,
        decode_state="decoded",
        decisions=tuple(decisions),
        accepted_count=accepted_count,
        rejected_count=len(decisions) - accepted_count,
        derived_calibration_manifest=derived_fingerprint,
        schedules=schedules,
        schedule_blocked_reason=blocked_reason if not schedules else None,
        provenance_uri=provenance_uri,
        correction_policy=fingerprint(policy_path, repository_root) if schedule_capable else None,
        source_corrections_dropped_out_of_range=dropped_out_of_range,
        runtime_seconds=time.monotonic() - started,
        candidate_ranking=candidate_ranking,
        claim_boundaries=CLAIM_BOUNDARIES,
    )
    if schedule_capable and not schedules:
        raise RuntimeError("schedule derivation produced no schedule")
    path = iteration_dir / DECISIONS_NAME
    path.write_text(output.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def _acceptance_log_line(decision: RepromptDecision, iteration: int) -> str:
    assert decision.accepted is not None
    return json.dumps(
        {
            "candidate_id": decision.accepted.calibration_candidate_id,
            "candidate_index": decision.accepted.candidate_index,
            "intended_target": decision.target,
            "analysis_frame_index": decision.onset_frame,
            "provenance": MULTIVIEW_CONSENSUS_PROVENANCE,
            "selected_by": "agent",
            "iteration": iteration,
            "candidate_ranking": decision.candidate_ranking,
            "rationale": decision.reason,
        }
    )


def _seed_slot(
    repository_root: Path, seed_manifest: ArtifactFingerprint | None, target: str
) -> int:
    """Multiplex slot of `target` in the geometric seed manifest (accepted parts in order).

    Mirrors `muggled_smoke._load_geometric_seed_manifest`, which assigns slots to the accepted
    parts in manifest order; the tracker re-validates the whole manifest before running.
    """
    if seed_manifest is None:
        raise ValueError("a geometry-seeded run names its seed manifest")
    payload = json.loads(
        _verify(repository_root, seed_manifest, "geometric seed manifest").read_text(
            encoding="utf-8"
        )
    )
    accepted = [
        str(part["target"])
        for part in payload.get("parts", [])
        if part.get("status") == "accepted" and part.get("accepted") is not None
    ]
    if target not in accepted:
        raise ValueError(f"{target} is not an accepted seed in {seed_manifest.uri}")
    return accepted.index(target)


def _placeholder_candidate(
    manifest: MuggledSAMBoxCalibrationManifest,
    candidate_id: str,
    prompt: MultiviewRepromptPrompt,
    onset: MultiviewContradictionOnset,
    width: int,
    height: int,
) -> MuggledSAMCalibrationCandidate:
    """Schema-valid stand-in that only reserves a candidate ID before decoding."""
    from .muggled_calibration import frame_reference

    return MuggledSAMCalibrationCandidate(
        candidate_id=candidate_id,
        intended_target=prompt.target,
        frame=frame_reference(
            onset.onset_frame / manifest.proxy_fps,
            fps=manifest.proxy_fps,
            source_offset_seconds=manifest.source_offset_seconds,
            frame_count=manifest.proxy_frame_count,
        ),
        pixel_box=prompt.pixel_box,
        normalized_box=_normalized_box(prompt.pixel_box, width, height),
        decoder_result={
            "api": "muggledsam_sam3_interactive",
            "candidate_count": 1,
            "deterministic_best_candidate_index": 0,
            "candidates": [
                {
                    "candidate_index": 0,
                    "iou_score": 0.0,
                    "mask_uri": "results/masks/pending.png",
                    "is_deterministic_best": True,
                }
            ],
            "overlay_uri": "results/pending.png",
        },
        selected_by="agent",
    )


# -- run commands ------------------------------------------------------------------------------


def choose_resume_frame(
    *,
    checkpoint_frames: Sequence[int],
    earliest_onset: int,
    source_correction_frames: Sequence[int],
    arm_correction_frames: Sequence[int],
) -> int | None:
    """Largest checkpoint frame before the earliest onset whose prefix matches the arm.

    A checkpoint at `k` holds the state ready to step `k`, produced by frames `[0, k)` with
    the source run's corrections before `k`; the arm may resume there only when every one of
    those corrections is also in the arm's schedule (the consensus-only arm therefore cannot
    resume past the first human correction).
    """
    arm_frames = set(arm_correction_frames)
    for frame in sorted((f for f in checkpoint_frames if 1 <= f <= earliest_onset), reverse=True):
        prefix = {f for f in source_correction_frames if f < frame}
        if prefix <= arm_frames:
            return frame
    return None


def smoke_argv(
    *,
    plan: MultiviewRepromptPlan,
    schedule_uri: str,
    run_root: str,
    checkpoint_every: int,
    resume_run_uri: str | None = None,
    resume_at: int | None = None,
) -> tuple[str, ...]:
    argv: list[str] = [
        "battle-muggled-smoke",
        "--config",
        plan.clip_config.uri,
        "--view",
        plan.target_view_id,
    ]
    if plan.source_run_profile == "four_part_static_focused":
        argv += ["--four-part-static-focused", "--max-frames", str(plan.frame_count)]
        schedule_flag = "--multi-keyframe-correction-schedule"
    else:
        argv += ["--four-part-multiview-first-minute"]
        if plan.source_geometric_seed_manifest is not None:
            argv += ["--geometric-seed-manifest", plan.source_geometric_seed_manifest.uri]
        if plan.recording_label is not None:
            argv += ["--recording", plan.recording_label, "--max-frames", str(plan.frame_count)]
        schedule_flag = "--agent-correction-schedule"
    argv += [
        "--max-side-length",
        str(MAX_SIDE_LENGTH),
        schedule_flag,
        schedule_uri,
        "--prompt-memory-semantics",
        "append",
        "--checkpoint-every",
        str(checkpoint_every),
        "--run-root",
        run_root,
    ]
    if resume_run_uri is not None and resume_at is not None:
        argv += ["--resume-run", resume_run_uri, "--resume-at", str(resume_at)]
    return tuple(argv)


def shell_line(argv: Sequence[str]) -> str:
    return "uv run " + " ".join(shlex.quote(a) for a in argv)


def run_commands(
    iteration_dir: Path,
    *,
    repository_root: Path,
    arms: Sequence[str] = MULTIVIEW_REPROMPT_ARMS,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
    prefer_resume: bool = False,
    run_root: Path | None = None,
) -> tuple[MultiviewRepromptRunCommand, ...]:
    """Emit the tracker command per arm; writes `run_commands.json` and `run_commands.sh`.

    Full runs are the default even when a checkpoint qualifies: the worker's stream identity
    hashes the whole schedule payload, so a resume under a schedule with added corrections is
    refused until that identity is scoped to the corrections before the resume frame.
    `prefer_resume` swaps the resume form into `argv` for the day that change lands.
    """
    repository_root = repository_root.resolve()
    iteration_dir = iteration_dir.resolve()
    plan = load_plan(iteration_dir / PLAN_NAME)
    decisions = load_decisions(iteration_dir / DECISIONS_NAME)
    accepted = [d for d in decisions.decisions if d.decision == "accepted"]
    if not accepted:
        raise ValueError("no accepted correction; nothing to run")
    earliest = min(d.onset_frame for d in accepted)
    consensus_frames = tuple(sorted({d.onset_frame for d in accepted}))
    source_later: dict[str, int] = {}
    if plan.source_schedule is not None:
        source_schedule = MuggledSAMMultiKeyframeCorrectionSchedule.model_validate_json(
            _verify(repository_root, plan.source_schedule, "source schedule").read_text(
                encoding="utf-8"
            )
        )
        source_later = {
            c.candidate_id: c.frame.analysis_frame_index
            for c in source_schedule.corrections
            if c.frame.analysis_frame_index
        }
    commands: list[MultiviewRepromptRunCommand] = []
    root_uri = relative_uri((run_root or (iteration_dir / "arms")).resolve(), repository_root)
    for arm in arms:
        if arm not in MULTIVIEW_REPROMPT_ARMS:
            raise ValueError(f"unknown arm {arm!r}; arms are {MULTIVIEW_REPROMPT_ARMS}")
        schedule = decisions.schedules.get(arm)
        if schedule is None:
            placeholder = ArtifactFingerprint(
                uri=f"<no schedule for {arm}>", sha256="0" * 64, source="measured"
            )
            commands.append(
                MultiviewRepromptRunCommand(
                    arm=arm,  # type: ignore[arg-type]
                    schedule=placeholder,
                    mode="blocked",
                    argv=smoke_argv(
                        plan=plan,
                        schedule_uri=f"<derived schedule for {arm}>",
                        run_root=f"{root_uri}/{arm}",
                        checkpoint_every=checkpoint_every,
                    ),
                    consensus_correction_frames=consensus_frames,
                    note=(
                        "BLOCKED: "
                        + (
                            decisions.schedule_blocked_reason
                            or (
                                "a geometry-seeded view has no human corrections to add; the "
                                "consensus-only arm is the only arm on this view"
                                if plan.source_run_profile == "four_part_multiview"
                                else "no schedule was derived for this arm"
                            )
                        )
                    ),
                )
            )
            continue
        schedule_path = _verify(repository_root, schedule, f"{arm} schedule")
        if plan.source_run_profile == "four_part_multiview":
            agent_model = MultiviewAgentCorrectionSchedule.model_validate_json(
                schedule_path.read_text(encoding="utf-8")
            )
            later = tuple(sorted({c.analysis_frame_index for c in agent_model.corrections}))
            arm_ids = {c.candidate_id for c in agent_model.corrections}
        else:
            model = MuggledSAMMultiKeyframeCorrectionSchedule.model_validate_json(
                schedule_path.read_text(encoding="utf-8")
            )
            later = tuple(
                sorted(
                    {
                        c.frame.analysis_frame_index
                        for c in model.corrections
                        if c.frame.analysis_frame_index
                    }
                )
            )
            arm_ids = {c.candidate_id for c in model.corrections}
        dropped_human = tuple(
            sorted(
                {
                    frame
                    for candidate_id, frame in source_later.items()
                    if candidate_id not in arm_ids and frame < plan.frame_count
                }
            )
        )
        resume_at = choose_resume_frame(
            checkpoint_frames=plan.source_checkpoint_frames,
            earliest_onset=earliest,
            source_correction_frames=plan.source_correction_frames,
            arm_correction_frames=later,
        )
        full = smoke_argv(
            plan=plan,
            schedule_uri=schedule.uri,
            run_root=f"{root_uri}/{arm}",
            checkpoint_every=checkpoint_every,
        )
        resume = (
            smoke_argv(
                plan=plan,
                schedule_uri=schedule.uri,
                run_root=f"{root_uri}/{arm}",
                checkpoint_every=checkpoint_every,
                resume_run_uri=plan.source_run_uri,
                resume_at=resume_at,
            )
            if resume_at is not None
            else None
        )
        if resume_at is None:
            note = (
                f"full run: no checkpoint of the source run at or before the earliest onset "
                f"{earliest} is consistent with this arm (source checkpoints "
                f"{list(plan.source_checkpoint_frames)}, "
                f"source corrections {list(plan.source_correction_frames)})"
            )
            mode = "full_run"
            argv = full
        elif prefer_resume:
            note = (
                f"resumed from the source run's checkpoint at {resume_at} (nearest at or before "
                "the "
                f"earliest onset {earliest}); requires the worker's stream identity to hash only "
                "the corrections before the resume frame"
            )
            mode = "checkpoint_resumed"
            argv = resume or full
        else:
            note = (
                f"full run emitted although checkpoint {resume_at} precedes the earliest onset "
                f"{earliest}: the worker's stream identity hashes the whole schedule payload and "
                "would refuse the resume (resume_argv kept for when that is scoped)"
            )
            mode = "full_run"
            argv = full
        if arm == "consensus-only":
            note += (
                f"; human-loop corrections at {list(dropped_human)} are absent from this schedule "
                "(frame-0 seeds plus consensus corrections only), so no --drop-correction-frame "
                "is needed"
            )
        commands.append(
            MultiviewRepromptRunCommand(
                arm=arm,  # type: ignore[arg-type]
                schedule=schedule,
                mode=mode,  # type: ignore[arg-type]
                argv=argv,
                resume_run_uri=plan.source_run_uri if (mode == "checkpoint_resumed") else None,
                resume_at=resume_at if (mode == "checkpoint_resumed") else None,
                resume_argv=resume,
                later_correction_frames=later,
                consensus_correction_frames=consensus_frames,
                dropped_human_correction_frames=dropped_human,
                note=note,
            )
        )
    (iteration_dir / COMMANDS_NAME).write_text(
        json.dumps([c.model_dump(mode="json") for c in commands], indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(repository_root))}",
        "",
    ]
    for command in commands:
        lines.append(f"# {command.arm}: {command.mode}. {command.note}")
        prefix = "# " if command.mode == "blocked" else ""
        lines.append(prefix + shell_line(command.argv))
        if command.resume_argv is not None and command.mode != "checkpoint_resumed":
            lines.append("# resume variant: " + shell_line(command.resume_argv))
        lines.append("")
    (iteration_dir / COMMANDS_SCRIPT_NAME).write_text("\n".join(lines), encoding="utf-8")
    return tuple(commands)


# -- CLI --------------------------------------------------------------------------------------


def _print_plan(plan: MultiviewRepromptPlan, path: Path) -> None:
    print(
        f"{plan.target_view} iteration {plan.iteration}/{plan.max_iterations}: "
        f"{len(plan.planned_onsets)} planned, "
        f"{len(plan.onsets) - len(plan.planned_onsets)} blocked (consensus "
        f"{'includes' if plan.consensus_includes_target_view else 'excludes'} the target)"
    )
    for onset in plan.onsets:
        line = (
            f"  {onset.target} onset {onset.onset_frame} [{onset.onset_frame},"
            f"{onset.interval_end_frame_exclusive}) error {onset.target_error_px:.0f} px "
            f"(max {onset.max_target_error_px:.0f}), {onset.static_views_used} static views"
        )
        if onset.status == "planned":
            line += (
                f", radius {onset.radius_mm or 0:.0f} mm, expected area "
                f"{onset.expected_area_px or 0:.0f} px, "
                f"{len(onset.prompts)} prompts, "
                f"{sum(len(p.background_points) for p in onset.prompts)} negatives"
            )
        else:
            line += f": blocked ({onset.blocked_reason})"
        print(line)
    if plan.distractor_guard is not None:
        guard = plan.distractor_guard
        print(
            f"  distractor guard {'on' if guard.enabled else 'off'}: {len(guard.hidden_marks)} "
            f"hidden marks from {len(guard.records)} records, +-{guard.window_frames} frames, "
            f"{len(guard.suppressed_onsets)} onset(s) suppressed {list(guard.suppressed_onsets)}"
        )
    print(f"-> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="consensus contradictions -> geometric prompts (CPU)")
    plan.add_argument("--target-view", default=DEFAULT_TARGET_VIEW)
    plan.add_argument("--consensus-root", type=Path, default=DEFAULT_CONSENSUS_ROOT)
    plan.add_argument(
        "--consensus-run",
        type=Path,
        default=None,
        help="alias of --consensus-root, for a consensus built with --exclude-view <target>",
    )
    plan.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    plan.add_argument(
        "--detector-config", type=Path, default=None, help="JSON with the detector keys"
    )
    plan.add_argument("--iteration", type=int, default=None)
    plan.add_argument("--previous-plan", type=Path, default=None)
    plan.add_argument("--hand-negatives", action="store_true")
    plan.add_argument("--box-margin", type=float, action="append", default=None)
    plan.add_argument("--overwrite", action="store_true")
    plan.add_argument(
        "--distractor-guard",
        dest="distractor_guard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Block an onset when a human anchor record on any view marks the part hidden "
            "within --distractor-guard-window frames of it (default on; recording 1 only)."
        ),
    )
    plan.add_argument("--distractor-guard-window", type=int, default=DISTRACTOR_GUARD_WINDOW_FRAMES)
    plan.add_argument(
        "--distractor-guard-record",
        type=Path,
        action="append",
        default=None,
        help=(
            "Human anchor record to use as evidence (repeatable); default: every "
            f"{HUMAN_RECORD_GLOB}"
        ),
    )
    plan.add_argument(
        "--recording",
        default=None,
        help="registry label of the recording when it is not recording 1 (rig, clocks, poses)",
    )

    decode = commands.add_parser(
        "decode", help="decode the plan with one warm worker (GPU), accept, derive"
    )
    decode.add_argument("--plan", type=Path, required=True)
    decode.add_argument("--external-python", type=Path, default=None)
    decode.add_argument("--model", type=Path, default=None)
    decode.add_argument("--device", default="cuda:0")
    decode.add_argument("--correction-policy", type=Path, default=DEFAULT_CORRECTION_POLICY)
    decode.add_argument(
        "--candidate-ranking",
        choices=MULTIVIEW_REPROMPT_CANDIDATE_RANKINGS,
        default="ray",
        help=(
            "how the accepted candidate is chosen among those passing the filters: ray = "
            "closest centroid ray (default, the seed-transfer rule); decoder_score = the "
            "decoder's own IoU estimate first, ties by ray (rule (g))"
        ),
    )

    run = commands.add_parser("run", help="emit the battle-muggled-smoke command per arm (CPU)")
    run.add_argument("--iteration-dir", type=Path, required=True)
    run.add_argument("--arm", action="append", default=None, choices=MULTIVIEW_REPROMPT_ARMS)
    run.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    run.add_argument("--prefer-resume", action="store_true")
    run.add_argument("--run-root", type=Path, default=None)

    args = parser.parse_args()
    root = args.repository_root.resolve()
    if args.command == "plan":
        detector = (
            load_detector_config(args.detector_config.resolve(), root)
            if args.detector_config is not None
            else DEFAULT_DETECTOR
        )
        path = plan_reprompt(
            root,
            target_view=args.target_view,
            consensus_root=args.consensus_run or args.consensus_root,
            source_run=args.source_run,
            output_root=args.output_root,
            detector=detector,
            iteration=args.iteration,
            previous_plan=args.previous_plan,
            hand_negatives=args.hand_negatives,
            box_margins=tuple(args.box_margin) if args.box_margin else BOX_MARGINS,
            overwrite=args.overwrite,
            recording=args.recording,
            distractor_guard=args.distractor_guard,
            distractor_guard_window=args.distractor_guard_window,
            distractor_guard_records=args.distractor_guard_record,
        )
        _print_plan(load_plan(path), path)
    elif args.command == "decode":
        path = decode_plan(
            args.plan,
            repository_root=root,
            external_python=args.external_python,
            model=args.model,
            device=args.device,
            correction_policy=args.correction_policy,
            candidate_ranking=args.candidate_ranking,
        )
        decisions = load_decisions(path)
        for decision in decisions.decisions:
            print(
                f"  {decision.target} @ {decision.onset_frame}: {decision.decision} "
                f"({decision.reason})"
            )
        print(
            f"candidate ranking {decisions.candidate_ranking}; "
            f"{decisions.accepted_count} accepted, {decisions.rejected_count} rejected; schedules "
            + (
                ", ".join(f"{k}={v.uri}" for k, v in decisions.schedules.items())
                or f"blocked ({decisions.schedule_blocked_reason})"
            )
        )
        print(f"-> {path}")
    else:
        emitted = run_commands(
            args.iteration_dir,
            repository_root=root,
            arms=tuple(args.arm) if args.arm else MULTIVIEW_REPROMPT_ARMS,
            checkpoint_every=args.checkpoint_every,
            prefer_resume=args.prefer_resume,
            run_root=args.run_root,
        )
        for command in emitted:
            print(f"# {command.arm} ({command.mode}): {command.note}")
            print(shell_line(command.argv))
        print(f"-> {args.iteration_dir.resolve() / COMMANDS_SCRIPT_NAME}")


if __name__ == "__main__":
    main()
