"""Versioned, inference-agnostic artifact contracts."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0"


class VersionedModel(BaseModel):
    """Base contract that rejects undeclared artifact fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: str = Field(default=SCHEMA_VERSION, pattern=r"^\d+\.\d+$")


class ClockName(StrEnum):
    SOURCE = "source"
    ANALYSIS = "analysis"
    ANNOTATION = "annotation"
    POSE = "pose"


class MethodState(StrEnum):
    APPROVED = "approved"
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    BLOCKED = "blocked"
    FAILED = "failed"
    NOT_RUN = "not_run"


class HumanQADisposition(StrEnum):
    """A disposition authored only by a human reviewer."""

    PENDING = "pending"
    PASS = "pass"
    FLAG = "flag"
    FAIL = "fail"


class HumanQACheckpointRole(StrEnum):
    EASY = "easy_manipulation"
    HARD = "hard_or_occluded_manipulation"


class HandSide(StrEnum):
    LEFT = "left"
    RIGHT = "right"
    UNKNOWN = "unknown"


class MaskStorage(StrEnum):
    EXTERNAL_ARTIFACT = "external_artifact"
    NATIVE_ARTIFACT = "native_artifact"


class ClockSpec(VersionedModel):
    name: ClockName
    fps: int = Field(gt=0)


class ClockSet(VersionedModel):
    """The four independently named time bases used in the lab.

    Every rate is pinned to the dataset's own cadences so a clock cannot drift
    unnoticed. Analysis is the one exception: it may sample either every source
    frame or every second one, because the frame rate is a run condition we
    compare. Sampling at 60 fps decouples analysis from the 30 fps annotation
    cadence, so an annotation comparison has to resample rather than assume a
    shared index.
    """

    clocks: tuple[ClockSpec, ...]

    @model_validator(mode="after")
    def require_planned_clocks(self) -> ClockSet:
        fixed = {
            ClockName.SOURCE: 60,
            ClockName.ANNOTATION: 30,
            ClockName.POSE: 60,
        }
        permitted_analysis_fps = (30, 60)
        received = {clock.name: clock.fps for clock in self.clocks}
        if {name: received.get(name) for name in fixed} != fixed:
            raise ValueError(f"source, annotation, and pose clocks must be exactly {fixed}")
        if received.get(ClockName.ANALYSIS) not in permitted_analysis_fps:
            raise ValueError(f"analysis clock must be one of {permitted_analysis_fps} fps")
        if set(received) != set(fixed) | {ClockName.ANALYSIS}:
            raise ValueError("clock set must name exactly source, analysis, annotation, and pose")
        return self

    def fps_for(self, clock_name: ClockName) -> int:
        return next(clock.fps for clock in self.clocks if clock.name == clock_name)


class ClockMapping(VersionedModel):
    """Affine mapping from a stream frame to source-video seconds."""

    clock: ClockName
    source_offset_seconds: float = Field(ge=0)
    scale: float = Field(gt=0)

    def source_seconds_for_frame(self, frame_index: int, clocks: ClockSet) -> float:
        if frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        return self.source_offset_seconds + self.scale * (frame_index / clocks.fps_for(self.clock))


class TimingModel(VersionedModel):
    clocks: ClockSet
    mappings: tuple[ClockMapping, ...]

    @model_validator(mode="after")
    def require_one_mapping_per_clock(self) -> TimingModel:
        mapped = {mapping.clock for mapping in self.mappings}
        expected = {clock.name for clock in self.clocks.clocks}
        if mapped != expected or len(mapped) != len(self.mappings):
            raise ValueError("mappings must contain one entry for every clock")
        return self

    def source_seconds_for_frame(self, clock: ClockName, frame_index: int) -> float:
        mapping = next(mapping for mapping in self.mappings if mapping.clock == clock)
        return mapping.source_seconds_for_frame(frame_index, self.clocks)


class TimeInterval(VersionedModel):
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)

    @model_validator(mode="after")
    def require_positive_duration(self) -> TimeInterval:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be after start_seconds")
        return self


class FullDurationCoverage(VersionedModel):
    """Coverage intervals for every source second claimed by a run."""

    source_duration_seconds: float = Field(gt=0)
    covered_intervals: tuple[TimeInterval, ...]

    @model_validator(mode="after")
    def require_ordered_bounded_intervals(self) -> FullDurationCoverage:
        previous_end = 0.0
        for interval in self.covered_intervals:
            if interval.end_seconds > self.source_duration_seconds:
                raise ValueError("coverage cannot extend beyond source duration")
            if interval.start_seconds < previous_end:
                raise ValueError("coverage intervals must be monotonic and non-overlapping")
            previous_end = interval.end_seconds
        return self

    @property
    def covered_seconds(self) -> float:
        return sum(
            interval.end_seconds - interval.start_seconds for interval in self.covered_intervals
        )

    @property
    def ratio(self) -> float:
        return self.covered_seconds / self.source_duration_seconds


class ChunkContinuityPolicy(VersionedModel):
    """Compatibility contract; streaming runs explicitly use no chunks."""

    chunk_duration_seconds: float | None = Field(default=None, gt=0)
    overlap_seconds: float = Field(ge=0)
    max_allowed_gap_seconds: float = Field(ge=0)
    preserve_track_ids: bool = True
    carry_context_across_chunks: bool = True

    @model_validator(mode="after")
    def require_overlap_smaller_than_chunk(self) -> ChunkContinuityPolicy:
        if self.chunk_duration_seconds is None:
            if self.overlap_seconds != 0:
                raise ValueError("unchunked streams cannot have overlap")
            if not self.carry_context_across_chunks:
                raise ValueError("unchunked streams must retain stream context")
            return self
        if self.overlap_seconds >= self.chunk_duration_seconds:
            raise ValueError("overlap_seconds must be less than chunk_duration_seconds")
        return self


class EncodedAssetInput(VersionedModel):
    """A media reference contract; this session deliberately never reads its payload."""

    uri: str = Field(min_length=1)
    media_type: str = Field(pattern=r"^video/")
    checksum_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


class ClipManifest(VersionedModel):
    clip_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    source_name: str
    source_license: str
    asset: EncodedAssetInput
    timing: TimingModel
    source_duration_seconds: float = Field(gt=0)
    views: tuple[str, ...] = Field(min_length=1)
    provenance_status: MethodState = MethodState.PENDING

    @field_validator("views")
    @classmethod
    def require_unique_views(cls, views: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(views)) != len(views):
            raise ValueError("view IDs must be unique")
        return views


class FrameRange(VersionedModel):
    """A half-open range of integer frames on a named clock."""

    start_frame: int = Field(ge=0)
    end_frame_exclusive: int = Field(gt=0)

    @model_validator(mode="after")
    def require_positive_frame_count(self) -> FrameRange:
        if self.end_frame_exclusive <= self.start_frame:
            raise ValueError("end_frame_exclusive must be after start_frame")
        return self

    @property
    def frame_count(self) -> int:
        return self.end_frame_exclusive - self.start_frame


class VideoDimensions(VersionedModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class ClipGateApproval(VersionedModel):
    """An explicit human approval gate for real-data preparation."""

    state: Literal["approved", "pending"]
    approved_by: str = Field(min_length=1)
    scope: str = Field(min_length=1)


class RawVideoSource(VersionedModel):
    view_id: str = Field(min_length=1)
    raw_uri: str = Field(min_length=1)
    checksum_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    dimensions: VideoDimensions
    fps: int = Field(gt=0)
    raw_frame_range: FrameRange


class VideoProxy(VersionedModel):
    view_id: str = Field(min_length=1)
    raw_source: RawVideoSource
    proxy_uri: str = Field(min_length=1)
    checksum_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    dimensions: VideoDimensions
    fps: int = Field(gt=0)
    frame_count: int = Field(gt=0)
    codec: Literal["h264"]
    crf: int = Field(ge=0, le=51)
    preset: str = Field(min_length=1)
    pixel_format: Literal["yuv420p"]
    has_audio: Literal[False] = False

    @model_validator(mode="after")
    def require_matching_view_id(self) -> VideoProxy:
        if self.raw_source.view_id != self.view_id:
            raise ValueError("proxy and raw source view IDs must match")
        return self


class G2PreprocessingManifest(VersionedModel):
    """Concrete, multi-view contract for an approved real-data proxy cut."""

    manifest_kind: Literal["assembly101_g2_preprocessing"]
    clip: ClipManifest
    hf_dataset: str = Field(min_length=1)
    hf_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    provenance_ledger_uri: str = Field(min_length=1)
    g1: ClipGateApproval
    g2: ClipGateApproval
    raw_timing: TimingModel
    proxy_timing: TimingModel
    source_interval: TimeInterval
    raw_frame_range: FrameRange
    analysis_frame_range: FrameRange
    proxy_frame_range: FrameRange
    scaling_policy: Literal[
        "preserve_aspect_ratio_height_720", "preserve_aspect_ratio_height_1080"
    ]
    proxies: tuple[VideoProxy, ...] = Field(min_length=1)
    annotations_or_poses_downloaded_by_g2: Literal[False] = False
    annotations_or_poses_used_by_g2: Literal[False] = False

    @model_validator(mode="after")
    def require_consistent_clock_and_proxy_ranges(self) -> G2PreprocessingManifest:
        source_fps = self.raw_timing.clocks.fps_for(ClockName.SOURCE)
        analysis_fps = self.raw_timing.clocks.fps_for(ClockName.ANALYSIS)
        start_seconds = self.source_interval.start_seconds
        end_seconds = self.source_interval.end_seconds
        if self.raw_frame_range.start_frame != round(
            start_seconds * source_fps
        ) or self.raw_frame_range.end_frame_exclusive != round(end_seconds * source_fps):
            raise ValueError("raw frame range must match the source interval")
        if self.analysis_frame_range.start_frame != round(
            start_seconds * analysis_fps
        ) or self.analysis_frame_range.end_frame_exclusive != round(end_seconds * analysis_fps):
            raise ValueError("analysis frame range must match the source interval")
        if self.proxy_frame_range.start_frame != 0:
            raise ValueError("proxy frame range must start at zero")
        if self.proxy_frame_range.frame_count != self.analysis_frame_range.frame_count:
            raise ValueError("proxy and analysis frame counts must match")
        if self.proxy_timing.source_seconds_for_frame(ClockName.ANALYSIS, 0) != start_seconds:
            raise ValueError("proxy analysis clock must start at the source interval")
        if {proxy.view_id for proxy in self.proxies} != set(self.clip.views):
            raise ValueError("proxy view IDs must exactly match clip views")
        if any(
            proxy.frame_count != self.proxy_frame_range.frame_count
            or proxy.fps != analysis_fps
            or proxy.has_audio
            for proxy in self.proxies
        ):
            raise ValueError("proxies must be audio-free and match the analysis range")
        return self


class MethodStatus(VersionedModel):
    method_name: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    state: MethodState
    artifact_uri: str | None = None
    blocker: str | None = None
    measured_on: str | None = None

    @model_validator(mode="after")
    def require_blocker_for_blocked_method(self) -> MethodStatus:
        if self.state == MethodState.BLOCKED and not self.blocker:
            raise ValueError("blocked methods require a blocker")
        return self


class NormalizedPoint(VersionedModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)


class CameraRelativePoint3D(VersionedModel):
    """One joint in WiLoR's camera-relative frame.

    Units are model-native and non-metric; they must not be treated as millimetres or
    compared across methods without an explicit calibration bridge.
    """

    x: float
    y: float
    z: float


class NormalizedBox(VersionedModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def require_box_inside_normalized_frame(self) -> NormalizedBox:
        if self.x + self.width > 1 or self.y + self.height > 1:
            raise ValueError("normalized box must stay inside [0, 1]²")
        return self


class MaskReference(VersionedModel):
    uri: str = Field(min_length=1)
    storage: MaskStorage
    format: str = Field(min_length=1)

    @field_validator("uri")
    @classmethod
    def reject_absolute_local_paths(cls, uri: str) -> str:
        if PurePosixPath(uri).is_absolute():
            raise ValueError("mask references must be portable URIs, not absolute local paths")
        return uri


class PerFrameObject(VersionedModel):
    object_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    box: NormalizedBox
    mask: MaskReference | None = None
    object_score: float | None = Field(
        default=None,
        description=(
            "Raw unbounded tracker presence logit. Upstream treats values at or below zero as "
            "a lost object. It is a diagnostic trace, not a calibrated probability or accuracy."
        ),
    )
    iou_prediction: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description=(
            "Tracker self-estimate of its own mask quality. It is a diagnostic trace and is "
            "not measured against ground truth."
        ),
    )


class ImageLandmark2D(VersionedModel):
    """One named 2D landmark in normalized image coordinates."""

    name: str = Field(min_length=1)
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)


class PerFrameNlfBody2D(VersionedModel):
    """NLF SMPL-X body joints projected to image pixels, not Battle hand landmarks."""

    subject_id: str = Field(min_length=1)
    coordinate_frame: Literal["image_normalized_top_left"] = "image_normalized_top_left"
    landmarks: tuple[ImageLandmark2D, ...] = Field(min_length=1)


class KineoBoxProvenance(VersionedModel):
    """One fused person crop input, retained separately from NLF pose output."""

    analysis_frame_index: int = Field(ge=0, lt=1800)
    source: Literal["detected_native", "boxmot_fallback", "interpolated", "held", "missing"]
    box: NormalizedBox | None = None
    native_box: NormalizedBox | None = None
    boxmot_box: NormalizedBox | None = None
    boxmot_object_id: str | None = None
    source_mapping_verified: bool
    nearest_native_iou: float | None = Field(default=None, ge=0, le=1)
    nearest_native_center_distance: float | None = Field(default=None, ge=0)
    residual_gap_length: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def require_source_evidence(self) -> KineoBoxProvenance:
        if self.source == "missing":
            if self.box is not None:
                raise ValueError("missing Kineo box provenance cannot contain a box")
        elif self.box is None:
            raise ValueError("non-missing Kineo box provenance requires a fused box")
        if self.source == "detected_native" and self.native_box is None:
            raise ValueError("native Kineo provenance requires its native box")
        if self.source == "boxmot_fallback" and (
            self.boxmot_box is None or not self.source_mapping_verified
        ):
            raise ValueError("BoxMOT fallback requires a verified BoxMOT source box")
        if self.source in ("interpolated", "held") and self.residual_gap_length is None:
            raise ValueError("inferred Kineo box provenance requires residual gap length")
        return self


class SegmentationReviewEpisode(VersionedModel):
    """A compact bookmark over adjacent raw geometry review triggers."""

    target_id: Literal["chassis", "interior", "rear_body", "cabin"]
    start_frame: int = Field(ge=0, lt=1800)
    end_frame: int = Field(ge=0, lt=1800)
    trigger_count: int = Field(ge=1)
    trigger_types: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_ordered_range(self) -> SegmentationReviewEpisode:
        if self.end_frame < self.start_frame:
            raise ValueError("segmentation review episode must have an ordered frame range")
        return self


class PerFrameHand(VersionedModel):
    hand_id: str = Field(min_length=1)
    side: HandSide
    confidence: float = Field(ge=0, le=1)
    landmarks: tuple[NormalizedPoint, ...] = Field(min_length=21, max_length=21)
    box: NormalizedBox
    model_side: HandSide
    model_handedness_confidence: float = Field(ge=0, le=1)
    joints_3d_camera_relative: tuple[CameraRelativePoint3D, ...] | None = Field(
        default=None,
        min_length=21,
        max_length=21,
        description=(
            "Optional 21-joint camera-relative 3D pose in the method's native non-metric "
            "frame. Present for WiLoR and absent for 2D-only baselines."
        ),
    )


MemoryGateReason = Literal[
    "ok", "warmup", "low_object_score", "low_iou", "contested", "area_jump", "corrected"
]


class TrackerSlotDiagnostic(VersionedModel):
    """One multiplex slot's raw tracker state, recorded whether or not it produced an object.

    Objects are filtered out once the tracker reports them lost, so these traces are kept
    separately to stay gapless across exactly the frames where tracking fails.
    """

    object_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    multiplex_slot: int = Field(ge=0)
    object_score: float
    iou_prediction: float | None = Field(default=None, ge=0, le=1)
    active: bool
    corrected: bool = False
    # Present only on runs with a non-default tracker memory policy; `object_score` stays the
    # raw presence logit, and these say what the policy did with it on this frame.
    contested_fraction: float | None = Field(default=None, ge=0, le=1)
    memory_written: bool | None = None
    memory_gate_reason: MemoryGateReason | None = None


class TrackerMemoryPolicy(VersionedModel):
    """The worker's slot exclusivity and score-gated memory condition, as run.

    Recorded in every run's runtime settings so the condition is stated even when it is the
    default (both off), which reproduces the unpoliced tracker exactly.  It is a run
    condition, not an accuracy claim.
    """

    slot_exclusivity: Literal["off", "argmax"] = "off"
    memory_gate: Literal["off", "on"] = "off"
    exclusivity_loser_logit: float = Field(default=-8.0, le=0)
    gate_min_object_score: float = 0.0
    gate_min_iou: float = Field(default=0.5, ge=0, le=1)
    gate_max_contested_fraction: float = Field(default=0.2, ge=0, le=1)
    gate_area_band: tuple[float, float] = (0.5, 2.0)
    gate_area_history_frames: int = Field(default=30, ge=1)

    @model_validator(mode="after")
    def require_ordered_area_band(self) -> TrackerMemoryPolicy:
        low, high = self.gate_area_band
        if low <= 0 or high < low:
            raise ValueError("gate_area_band must be 0 < low <= high")
        return self

    @property
    def is_default(self) -> bool:
        return self.slot_exclusivity == "off" and self.memory_gate == "off"

    def run_id_suffix(self) -> str:
        """Short arm label for run ids: empty for the default policy."""
        parts = []
        if self.slot_exclusivity != "off":
            parts.append(f"x{self.slot_exclusivity}")
        if self.memory_gate != "off":
            parts.append(f"g{self.memory_gate}")
        return "-".join(parts)

    def worker_arguments(self) -> list[str]:
        return [
            "--slot-exclusivity",
            self.slot_exclusivity,
            "--exclusivity-loser-logit",
            str(self.exclusivity_loser_logit),
            "--memory-gate",
            self.memory_gate,
            "--gate-min-object-score",
            str(self.gate_min_object_score),
            "--gate-min-iou",
            str(self.gate_min_iou),
            "--gate-max-contested-fraction",
            str(self.gate_max_contested_fraction),
            "--gate-area-band",
            f"{self.gate_area_band[0]},{self.gate_area_band[1]}",
            "--gate-area-history-frames",
            str(self.gate_area_history_frames),
        ]


CorrectionMemorySemantics = Literal[
    "replace_prompt_memory_and_reset_frame_memory",
    "append_prompt_memory_and_reset_frame_memory",
    "append_prompt_memory_and_keep_frame_memory",
    "replace_prompt_memory_and_keep_frame_memory",
]

# MuggledSAM's SAM3 memory fusion has learned position offsets for six frame-memory entries;
# a larger bank runs, but the older entries share the sixth offset.
TRAINED_MAX_FRAME_MEMORY_ENTRIES = 6


class CorrectionMemorySettings(VersionedModel):
    """The worker's memory-bank condition: what a correction does and how big the banks are.

    Recorded in every run's runtime settings, default or not.  The defaults reproduce the
    runs made before the flags existed (one replaced prompt entry, frame memory cleared at
    a correction, frame memories declared oldest-first).  It is a run condition, not an
    accuracy claim.
    """

    prompt_memory_semantics: Literal["replace", "append"] = "replace"
    max_prompt_memory_entries: int = Field(ge=1)
    max_frame_memory_entries: int = Field(default=4, ge=1)
    keep_frame_memory_at_correction: bool = False
    is_recent_first: bool = False

    @model_validator(mode="after")
    def require_one_entry_for_replace(self) -> CorrectionMemorySettings:
        if self.prompt_memory_semantics == "replace" and self.max_prompt_memory_entries != 1:
            raise ValueError("replace semantics keeps exactly one prompt memory entry")
        return self

    @classmethod
    def default_prompt_memory_entries(cls, prompt_memory_semantics: str) -> int:
        return 32 if prompt_memory_semantics == "append" else 1

    @property
    def correction_memory_semantics(self) -> str:
        frame_action = "keep" if self.keep_frame_memory_at_correction else "reset"
        return f"{self.prompt_memory_semantics}_prompt_memory_and_{frame_action}_frame_memory"

    @property
    def frame_memory_position_encoding(self) -> str:
        if self.max_frame_memory_entries > TRAINED_MAX_FRAME_MEMORY_ENTRIES:
            return "clamped_beyond_6"
        return "within_trained_range"

    def is_default_except_frame_memory(self) -> bool:
        return (
            self.prompt_memory_semantics == "replace"
            and self.max_prompt_memory_entries == 1
            and not self.keep_frame_memory_at_correction
            and not self.is_recent_first
        )

    def run_id_suffix(self, *, reference_frame_memory_entries: int | None = None) -> str:
        """Short arm label for run ids: empty for the default condition."""
        parts = []
        if (
            reference_frame_memory_entries is not None
            and self.max_frame_memory_entries != reference_frame_memory_entries
        ):
            parts.append(f"fm{self.max_frame_memory_entries}")
        if self.prompt_memory_semantics != "replace":
            parts.append(f"pm-{self.prompt_memory_semantics}")
            if self.max_prompt_memory_entries != self.default_prompt_memory_entries(
                self.prompt_memory_semantics
            ):
                parts.append(f"pm{self.max_prompt_memory_entries}")
        if self.keep_frame_memory_at_correction:
            parts.append("keepfm")
        if self.is_recent_first:
            parts.append("recent-first")
        return "-".join(parts)

    def worker_arguments(self) -> list[str]:
        """Worker flags for the non-default settings; `--max-frame-memory` is passed separately."""
        arguments: list[str] = []
        if self.prompt_memory_semantics != "replace":
            arguments.extend(["--prompt-memory-semantics", self.prompt_memory_semantics])
            arguments.extend(["--max-prompt-memory", str(self.max_prompt_memory_entries)])
        if self.keep_frame_memory_at_correction:
            arguments.append("--keep-frame-memory-at-correction")
        if self.is_recent_first:
            arguments.append("--recent-first")
        return arguments


class FrameObservations(VersionedModel):
    view_id: str = Field(min_length=1)
    analysis_frame_index: int = Field(ge=0)
    source_seconds: float = Field(ge=0)
    objects: tuple[PerFrameObject, ...] = ()
    hands: tuple[PerFrameHand, ...] = ()
    nlf_body_2d: tuple[PerFrameNlfBody2D, ...] = ()
    tracker_diagnostics: tuple[TrackerSlotDiagnostic, ...] = ()


class ArtifactFingerprint(VersionedModel):
    """A portable content identity supplied or measured for a run input."""

    uri: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source: Literal["approved_config", "measured"]


class AdapterMetadata(VersionedModel):
    """Version and provenance of an adapter without embedding third-party code."""

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    implementation_basis: str = Field(min_length=1)
    external_source_uri: str = Field(min_length=1)
    external_revision: str | None = None


class StreamContinuityPolicy(VersionedModel):
    """Bounded rolling memory with no chunk boundary or intentional ID reset."""

    mode: Literal["continuous_stream", "checkpoint_resumed"] = "continuous_stream"
    intentional_id_resets: Literal[False] = False
    max_prompt_memory_entries: int = Field(ge=1)
    max_frame_memory_entries: int = Field(ge=1)
    detected_object_limit: int = Field(ge=1)
    resumed_from_run: str | None = Field(
        default=None,
        description="Run whose observations and masks this run reuses before its resume frame.",
    )
    resumed_at_frame: int | None = Field(
        default=None,
        gt=0,
        description="First frame this run stepped itself; earlier frames are copied unchanged.",
    )
    checkpoint_fingerprint: ArtifactFingerprint | None = None

    @model_validator(mode="after")
    def require_complete_resume_provenance(self) -> StreamContinuityPolicy:
        resume_fields = (self.resumed_from_run, self.resumed_at_frame, self.checkpoint_fingerprint)
        if self.mode == "continuous_stream":
            if any(field is not None for field in resume_fields):
                raise ValueError("a continuous stream cannot claim resume provenance")
            return self
        if any(field is None for field in resume_fields):
            raise ValueError(
                "a resumed stream must name its prior run, resume frame, and checkpoint"
            )
        return self


class RuntimeMeasurements(VersionedModel):
    elapsed_seconds: float = Field(ge=0)
    time_to_first_usable_output_seconds: float | None = Field(default=None, ge=0)
    gpu_peak_vram_bytes: int | None = Field(default=None, ge=0)
    known_unavailable_measures: tuple[str, ...] = ()


class PixelBox(VersionedModel):
    """Integer image coordinates retained with a manually supplied prompt."""

    x1: int = Field(ge=0)
    y1: int = Field(ge=0)
    x2: int = Field(gt=0)
    y2: int = Field(gt=0)

    @model_validator(mode="after")
    def require_positive_box(self) -> PixelBox:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("pixel box must have positive width and height")
        return self


class PixelPoint(VersionedModel):
    """One integer source-image click retained for an interactive prompt."""

    x: int = Field(ge=0)
    y: int = Field(ge=0)


class MuggledSAMPreprocessing(VersionedModel):
    """Explicit decoded-image representation supplied to MuggledSAM."""

    mode: Literal["original_bgr", "gray_p01_p99_clahe"]
    lower_percentile: float | None = Field(default=None, ge=0, le=100)
    upper_percentile: float | None = Field(default=None, ge=0, le=100)
    clahe_clip_limit: float | None = Field(default=None, gt=0)
    clahe_tile_grid_size: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def require_mode_specific_settings(self) -> MuggledSAMPreprocessing:
        if self.mode == "original_bgr":
            if any(
                setting is not None
                for setting in (
                    self.lower_percentile,
                    self.upper_percentile,
                    self.clahe_clip_limit,
                    self.clahe_tile_grid_size,
                )
            ):
                raise ValueError("original_bgr must not specify normalization or CLAHE settings")
            return self
        if None in (
            self.lower_percentile,
            self.upper_percentile,
            self.clahe_clip_limit,
            self.clahe_tile_grid_size,
        ):
            raise ValueError("gray_p01_p99_clahe requires all normalization and CLAHE settings")
        if self.lower_percentile >= self.upper_percentile:
            raise ValueError("lower_percentile must be less than upper_percentile")
        return self


class MuggledSAMManualBoxSeed(VersionedModel):
    """A human-specified first-frame box, never a detector-derived pseudo-prompt."""

    analysis_frame_index: Literal[0]
    pixel_box: PixelBox
    normalized_box: NormalizedBox
    target_description: str = Field(min_length=1)


class MuggledSAMEgoCondition(VersionedModel):
    """A deliberately narrow, named ego-view condition."""

    condition_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    display_label: str = Field(min_length=1)
    prompt_mode: Literal["text_detection", "manual_box"]
    concepts: tuple[str, ...] = Field(min_length=1)
    preprocessing: MuggledSAMPreprocessing
    manual_box_seed: MuggledSAMManualBoxSeed | None = None

    @model_validator(mode="after")
    def require_explicit_condition_semantics(self) -> MuggledSAMEgoCondition:
        if self.prompt_mode == "text_detection":
            if self.manual_box_seed is not None:
                raise ValueError("text_detection conditions cannot include a manual seed")
            if self.concepts != ("hand",):
                raise ValueError("text_detection ego conditions must use exactly the hand concept")
        elif self.manual_box_seed is None:
            raise ValueError("manual_box conditions require a first-frame manual seed")
        elif self.concepts != ("hand",):
            raise ValueError("manual_box ego conditions must be labeled exactly hand")
        return self


class MuggledSAMEgoConditionConfig(VersionedModel):
    """Versioned configuration for evidence-driven 300-frame ego experiments."""

    manifest_kind: Literal["muggledsam_sam3_ego_conditions"]
    base_g2_config: str = Field(min_length=1)
    view_id: Literal["ego-hmc21110305"]
    conditions: tuple[MuggledSAMEgoCondition, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_unique_condition_ids(self) -> MuggledSAMEgoConditionConfig:
        condition_ids = [condition.condition_id for condition in self.conditions]
        if len(set(condition_ids)) != len(condition_ids):
            raise ValueError("condition IDs must be unique")
        return self


class CalibrationFrameReference(VersionedModel):
    """One decoded proxy frame used for a manual image-prompt calibration."""

    analysis_frame_index: int = Field(ge=0)
    proxy_seconds: float = Field(ge=0)
    analysis_seconds: float = Field(ge=0)
    source_seconds: float = Field(ge=0)


class MuggledSAMImageCandidate(VersionedModel):
    """A single interactive-image decoder hypothesis retained for review."""

    candidate_index: int = Field(ge=0)
    iou_score: float
    predicted_box: NormalizedBox | None = None
    mask_uri: str = Field(min_length=1)
    review_uri: str | None = None
    is_deterministic_best: bool = False

    @field_validator("mask_uri")
    @classmethod
    def require_relative_mask_uri(cls, uri: str) -> str:
        if PurePosixPath(uri).is_absolute():
            raise ValueError("calibration masks must use repository-portable relative URIs")
        return uri


class MuggledSAMImageDecoderResult(VersionedModel):
    """Image-only SAM3 response for one manually drawn rectangle."""

    api: Literal["muggledsam_sam3_interactive"]
    candidate_count: int = Field(ge=1)
    deterministic_best_candidate_index: int = Field(ge=0)
    candidates: tuple[MuggledSAMImageCandidate, ...] = Field(min_length=1)
    overlay_uri: str = Field(min_length=1)
    stability_score_available: Literal[False] = False
    limitations: tuple[str, ...] = ()

    @field_validator("overlay_uri")
    @classmethod
    def require_relative_overlay_uri(cls, uri: str) -> str:
        if PurePosixPath(uri).is_absolute():
            raise ValueError("calibration overlays must use repository-portable relative URIs")
        return uri

    @model_validator(mode="after")
    def require_complete_deterministic_candidates(self) -> MuggledSAMImageDecoderResult:
        candidate_indices = [candidate.candidate_index for candidate in self.candidates]
        best_candidates = [
            candidate.candidate_index
            for candidate in self.candidates
            if candidate.is_deterministic_best
        ]
        if self.candidate_count != len(self.candidates):
            raise ValueError("candidate_count must equal the retained image-decoder candidates")
        if len(set(candidate_indices)) != len(candidate_indices):
            raise ValueError("image-decoder candidate indices must be unique")
        if best_candidates != [self.deterministic_best_candidate_index]:
            raise ValueError("exactly the deterministic best candidate must be labeled")
        return self


class MuggledSAMCalibrationCandidate(VersionedModel):
    """A decoded image-prompt and its SAM3 evidence, including review disposition."""

    candidate_id: str = Field(pattern=r"^t\d{6}-b\d{2,}$")
    intended_target: str = Field(min_length=1)
    frame: CalibrationFrameReference
    pixel_box: PixelBox
    normalized_box: NormalizedBox
    pixel_fg_points: tuple[PixelPoint, ...] = ()
    pixel_bg_points: tuple[PixelPoint, ...] = ()
    normalized_fg_points: tuple[NormalizedPoint, ...] = ()
    normalized_bg_points: tuple[NormalizedPoint, ...] = ()
    decoder_result: MuggledSAMImageDecoderResult
    source_box_id: str | None = Field(default=None, pattern=r"^p\d{6}-b\d{2,}$")
    live_preview: bool = False
    human_selected_candidate_index: int | None = Field(default=None, ge=0)
    human_accepted: bool = False
    # Who made the acceptance recorded in the historical `human_*` fields. Existing records
    # predate this field and were human-reviewed, hence the default; agent-authored visual
    # review must say so explicitly and may only supply later-frame corrections.
    selected_by: Literal["human", "agent"] = "human"
    legacy_finalization_requested: bool = False
    selected_for_finalization: bool = False
    selected_for_correction: bool = False
    rejected: bool = False

    @model_validator(mode="before")
    @classmethod
    def preserve_legacy_unverified_finalization(cls, value: object) -> object:
        """Keep older manifests readable without treating their model choice as human review."""
        if isinstance(value, dict) and "human_selected_candidate_index" not in value:
            migrated = value.copy()
            if migrated.get("selected_for_finalization", False):
                migrated["legacy_finalization_requested"] = True
                migrated["selected_for_finalization"] = False
            return migrated
        return value

    @model_validator(mode="after")
    def require_explicit_human_candidate_selection(self) -> MuggledSAMCalibrationCandidate:
        available_indices = {
            candidate.candidate_index for candidate in self.decoder_result.candidates
        }
        if self.human_selected_candidate_index is not None and (
            self.human_selected_candidate_index not in available_indices
        ):
            raise ValueError("human-selected candidate index must be returned by the decoder")
        if self.human_accepted != (self.human_selected_candidate_index is not None):
            raise ValueError("human acceptance requires exactly one selected decoder candidate")
        if self.live_preview != (self.source_box_id is not None):
            raise ValueError("live preview candidates must identify their editable source box")
        if self.selected_for_finalization and not self.human_accepted:
            raise ValueError("finalization eligibility requires explicit human mask acceptance")
        if self.selected_for_correction and not self.human_accepted:
            raise ValueError("correction eligibility requires explicit human mask acceptance")
        if self.selected_for_correction and self.frame.analysis_frame_index == 0:
            raise ValueError("correction eligibility requires a later-frame mask")
        if self.selected_by == "agent" and (
            self.selected_for_finalization or self.frame.analysis_frame_index == 0
        ):
            raise ValueError("agent-selected masks are later-frame corrections only, never seeds")
        if self.rejected and (
            self.human_accepted
            or self.human_selected_candidate_index is not None
            or self.selected_for_finalization
            or self.selected_for_correction
        ):
            raise ValueError(
                "rejected candidates cannot retain human acceptance or proposal eligibility"
            )
        return self


class MuggledSAMCalibrationPendingBox(VersionedModel):
    """A browser-drawn box-and-point prompt retained before image decoding."""

    box_id: str = Field(pattern=r"^p\d{6}-b\d{2,}$")
    intended_target: str = Field(min_length=1)
    frame: CalibrationFrameReference
    pixel_box: PixelBox
    normalized_box: NormalizedBox
    pixel_fg_points: tuple[PixelPoint, ...] = ()
    pixel_bg_points: tuple[PixelPoint, ...] = ()
    normalized_fg_points: tuple[NormalizedPoint, ...] = ()
    normalized_bg_points: tuple[NormalizedPoint, ...] = ()
    stage: Literal["pending", "queued"] = "pending"


class MuggledSAMCalibrationWorkspace(VersionedModel):
    """Resumable local-workspace state that contains no inference result."""

    active_proxy_timestamp_seconds: float = Field(default=0, ge=0)
    selected_box_id: str | None = None
    custom_label: str = ""
    pending_boxes: tuple[MuggledSAMCalibrationPendingBox, ...] = ()


def _require_normalized_prompt_points(
    *,
    pixel_fg_points: tuple[PixelPoint, ...],
    pixel_bg_points: tuple[PixelPoint, ...],
    normalized_fg_points: tuple[NormalizedPoint, ...],
    normalized_bg_points: tuple[NormalizedPoint, ...],
    width: int,
    height: int,
    item_name: str,
) -> None:
    """Ensure retained source clicks and decoder-ready normalized clicks agree."""
    for kind, pixel_points, normalized_points in (
        ("foreground", pixel_fg_points, normalized_fg_points),
        ("background", pixel_bg_points, normalized_bg_points),
    ):
        if len(pixel_points) != len(normalized_points):
            raise ValueError(f"{item_name} {kind} point counts must match")
        for pixel, normalized in zip(pixel_points, normalized_points, strict=True):
            if pixel.x > width or pixel.y > height:
                raise ValueError(f"{item_name} {kind} point must be inside the proxy dimensions")
            if (
                abs(normalized.x - pixel.x / width) > 1e-9
                or abs(normalized.y - pixel.y / height) > 1e-9
            ):
                raise ValueError(f"{item_name} normalized {kind} point must match its pixel point")


class MuggledSAMCalibrationHiddenTarget(VersionedModel):
    """A human's explicit statement that a target is not visible on a calibration frame.

    A hidden mark is review evidence in its own right (the reviewer looked and found no
    visible surface), distinct from a cell that was simply never labelled. It never enters
    a proposal or correction schedule: those need masks, and this says there is none.
    """

    intended_target: str = Field(min_length=1)
    frame: CalibrationFrameReference
    state: Literal["hidden"] = "hidden"
    marked_by: Literal["human"] = "human"


class MuggledSAMSupersededTrackingPlan(VersionedModel):
    """A finalized plan a human reopened for further editing.

    The artifacts named here are never rewritten: a later finalization writes its own
    revision-stamped files, so every plan a reviewer once saw stays byte-identical and
    keeps the calibration-manifest hash it was sealed with.
    """

    plan_revision: int = Field(ge=1)
    proposal_uri: str | None = None
    correction_schedule_uri: str | None = None

    @model_validator(mode="after")
    def require_a_superseded_artifact(self) -> MuggledSAMSupersededTrackingPlan:
        if self.proposal_uri is None and self.correction_schedule_uri is None:
            raise ValueError("a superseded plan must name at least one written artifact")
        return self


# Any approved proxy view id (`static-c10119`, `ego-hmc21179183`, ...). The calibration
# workspace and the manual-seed target policy accept every view of a clip config; the
# correction policy/schedule and the run profiles below still name the views they permit.
CALIBRATION_VIEW_ID_PATTERN = r"^(static|ego)-[a-z0-9]+$"


class MuggledSAMBoxCalibrationManifest(VersionedModel):
    """Gitignored, reviewable manual-box calibration for one approved proxy view."""

    manifest_kind: Literal["muggledsam_sam3_box_calibration"]
    calibration_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    base_g2_config: str = Field(min_length=1)
    base_g2_config_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    view_id: str = Field(pattern=CALIBRATION_VIEW_ID_PATTERN)
    proxy: ArtifactFingerprint
    source: ArtifactFingerprint
    proxy_dimensions: VideoDimensions
    proxy_fps: int = Field(gt=0)
    proxy_frame_count: int = Field(gt=0)
    source_offset_seconds: float = Field(ge=0)
    tool_version: str = Field(min_length=1)
    image_representation: Literal["original_bgr"]
    prompt_api: Literal["boxes_fg_points_bg_points"]
    requested_proxy_timestamps_seconds: tuple[float, ...] = Field(min_length=1)
    candidates: tuple[MuggledSAMCalibrationCandidate, ...] = ()
    workspace: MuggledSAMCalibrationWorkspace = Field(
        default_factory=MuggledSAMCalibrationWorkspace
    )
    result_directory_uri: str = Field(min_length=1)
    final_proposal_uri: str | None = None
    final_correction_schedule_uri: str | None = None
    # Revision 1 uses the historical unsuffixed artifact names; a plan reopened and
    # finalized again writes revision 2 onwards beside it instead of over it.
    plan_revision: int = Field(default=0, ge=0)
    superseded_plans: tuple[MuggledSAMSupersededTrackingPlan, ...] = ()
    # Set when a calibration was copied from a finalized one so extra later-frame
    # corrections could be added without rewriting the original's fingerprinted manifest.
    derived_from_calibration: ArtifactFingerprint | None = None
    # Frame/target cells a human explicitly marked as not visible (no mask to draw).
    hidden_targets: tuple[MuggledSAMCalibrationHiddenTarget, ...] = ()

    @field_validator("requested_proxy_timestamps_seconds")
    @classmethod
    def require_ordered_unique_timestamps(cls, timestamps: tuple[float, ...]) -> tuple[float, ...]:
        if any(timestamp < 0 for timestamp in timestamps):
            raise ValueError("requested proxy timestamps must be non-negative")
        if tuple(sorted(set(timestamps))) != timestamps:
            raise ValueError("requested proxy timestamps must be ordered and unique")
        return timestamps

    @model_validator(mode="after")
    def require_unique_candidate_ids(self) -> MuggledSAMBoxCalibrationManifest:
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("calibration candidate IDs must be unique")
        pending_ids = [box.box_id for box in self.workspace.pending_boxes]
        if len(set(pending_ids)) != len(pending_ids):
            raise ValueError("pending calibration box IDs must be unique")
        if (
            self.workspace.selected_box_id is not None
            and self.workspace.selected_box_id not in pending_ids
        ):
            raise ValueError("workspace selected box must remain pending")
        for candidate in self.candidates:
            frame = candidate.frame
            pixel_box = candidate.pixel_box
            normalized_box = candidate.normalized_box
            expected_seconds = frame.analysis_frame_index / self.proxy_fps
            if frame.analysis_frame_index >= self.proxy_frame_count:
                raise ValueError("calibration candidate frame must be inside the proxy")
            if (
                abs(frame.proxy_seconds - expected_seconds) > 1e-9
                or abs(frame.analysis_seconds - expected_seconds) > 1e-9
                or abs(frame.source_seconds - (self.source_offset_seconds + expected_seconds))
                > 1e-9
            ):
                raise ValueError("calibration candidate time must match its analysis frame")
            if not any(
                abs(frame.proxy_seconds - timestamp) <= 1e-9
                for timestamp in self.requested_proxy_timestamps_seconds
            ):
                raise ValueError("calibration candidate must use a requested proxy timestamp")
            if (
                pixel_box.x2 > self.proxy_dimensions.width
                or pixel_box.y2 > self.proxy_dimensions.height
            ):
                raise ValueError(
                    "calibration candidate pixel box must be inside the proxy dimensions"
                )
            expected_normalized = (
                pixel_box.x1 / self.proxy_dimensions.width,
                pixel_box.y1 / self.proxy_dimensions.height,
                (pixel_box.x2 - pixel_box.x1) / self.proxy_dimensions.width,
                (pixel_box.y2 - pixel_box.y1) / self.proxy_dimensions.height,
            )
            received_normalized = (
                normalized_box.x,
                normalized_box.y,
                normalized_box.width,
                normalized_box.height,
            )
            if any(
                abs(received - expected) > 1e-9
                for received, expected in zip(received_normalized, expected_normalized, strict=True)
            ):
                raise ValueError("calibration candidate normalized box must match its pixel box")
            _require_normalized_prompt_points(
                pixel_fg_points=candidate.pixel_fg_points,
                pixel_bg_points=candidate.pixel_bg_points,
                normalized_fg_points=candidate.normalized_fg_points,
                normalized_bg_points=candidate.normalized_bg_points,
                width=self.proxy_dimensions.width,
                height=self.proxy_dimensions.height,
                item_name="calibration candidate",
            )
            if candidate.selected_for_finalization and (
                candidate.frame.analysis_frame_index != 0 or not candidate.human_accepted
            ):
                raise ValueError("only human-accepted frame-0 masks can be finalization eligible")
        for pending in self.workspace.pending_boxes:
            frame = pending.frame
            if frame.analysis_frame_index >= self.proxy_frame_count:
                raise ValueError("pending calibration box frame must be inside the proxy")
            expected_seconds = frame.analysis_frame_index / self.proxy_fps
            if (
                abs(frame.proxy_seconds - expected_seconds) > 1e-9
                or abs(frame.analysis_seconds - expected_seconds) > 1e-9
                or abs(frame.source_seconds - (self.source_offset_seconds + expected_seconds))
                > 1e-9
            ):
                raise ValueError("pending calibration box time must match its analysis frame")
            if not any(
                abs(frame.proxy_seconds - timestamp) <= 1e-9
                for timestamp in self.requested_proxy_timestamps_seconds
            ):
                raise ValueError("pending calibration box must use a requested proxy timestamp")
            if (
                pending.pixel_box.x2 > self.proxy_dimensions.width
                or pending.pixel_box.y2 > self.proxy_dimensions.height
            ):
                raise ValueError("pending calibration box must be inside the proxy dimensions")
            expected_normalized = (
                pending.pixel_box.x1 / self.proxy_dimensions.width,
                pending.pixel_box.y1 / self.proxy_dimensions.height,
                (pending.pixel_box.x2 - pending.pixel_box.x1) / self.proxy_dimensions.width,
                (pending.pixel_box.y2 - pending.pixel_box.y1) / self.proxy_dimensions.height,
            )
            actual_normalized = (
                pending.normalized_box.x,
                pending.normalized_box.y,
                pending.normalized_box.width,
                pending.normalized_box.height,
            )
            if any(
                abs(actual - expected) > 1e-9
                for actual, expected in zip(actual_normalized, expected_normalized, strict=True)
            ):
                raise ValueError("pending normalized box must match its pixel box")
            _require_normalized_prompt_points(
                pixel_fg_points=pending.pixel_fg_points,
                pixel_bg_points=pending.pixel_bg_points,
                normalized_fg_points=pending.normalized_fg_points,
                normalized_bg_points=pending.normalized_bg_points,
                width=self.proxy_dimensions.width,
                height=self.proxy_dimensions.height,
                item_name="pending calibration",
            )
        return self

    @model_validator(mode="after")
    def require_ordered_plan_revisions(self) -> MuggledSAMBoxCalibrationManifest:
        revisions = [plan.plan_revision for plan in self.superseded_plans]
        if revisions != sorted(set(revisions)):
            raise ValueError("superseded plan revisions must be increasing and unique")
        if revisions and max(revisions) > self.plan_revision:
            raise ValueError("a superseded plan cannot claim a revision the plan never reached")
        return self

    @model_validator(mode="after")
    def require_consistent_hidden_targets(self) -> MuggledSAMBoxCalibrationManifest:
        cells = [
            (mark.frame.analysis_frame_index, mark.intended_target) for mark in self.hidden_targets
        ]
        if len(set(cells)) != len(cells):
            raise ValueError("a target can be marked hidden at most once per frame")
        accepted = {
            (candidate.frame.analysis_frame_index, candidate.intended_target)
            for candidate in self.candidates
            if candidate.human_accepted and not candidate.rejected
        }
        for mark in self.hidden_targets:
            frame = mark.frame
            expected_seconds = frame.analysis_frame_index / self.proxy_fps
            if frame.analysis_frame_index >= self.proxy_frame_count:
                raise ValueError("hidden target frame must be inside the proxy")
            if (
                abs(frame.proxy_seconds - expected_seconds) > 1e-9
                or abs(frame.analysis_seconds - expected_seconds) > 1e-9
                or abs(frame.source_seconds - (self.source_offset_seconds + expected_seconds))
                > 1e-9
            ):
                raise ValueError("hidden target time must match its analysis frame")
            if not any(
                abs(frame.proxy_seconds - timestamp) <= 1e-9
                for timestamp in self.requested_proxy_timestamps_seconds
            ):
                raise ValueError("hidden target must use a requested proxy timestamp")
            if (frame.analysis_frame_index, mark.intended_target) in accepted:
                raise ValueError(
                    "a target cannot be both hidden and have an accepted mask on one frame"
                )
        return self


class ProposedTrackingSeed(VersionedModel):
    """A user-selected calibration prompt proposed for a future tracker policy."""

    candidate_id: str = Field(pattern=r"^t\d{6}-b\d{2,}$")
    intended_target: str = Field(min_length=1)
    reference_frame: CalibrationFrameReference
    pixel_box: PixelBox
    normalized_box: NormalizedBox
    pixel_fg_points: tuple[PixelPoint, ...] = ()
    pixel_bg_points: tuple[PixelPoint, ...] = ()
    normalized_fg_points: tuple[NormalizedPoint, ...] = ()
    normalized_bg_points: tuple[NormalizedPoint, ...] = ()
    decoder_best_candidate_index: int = Field(ge=0)
    human_selected_candidate_index: int = Field(ge=0)


class MuggledSAMManualSeedTargetDescriptor(VersionedModel):
    """Stable dataset identity plus its display and pixel-visibility contract."""

    target_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    display_alias: str = Field(min_length=1)
    mask_semantics: Literal["visible_surface_only"]


class MuggledSAMManualSeedTargetConfig(VersionedModel):
    """Named target policy for a human-selected e4 manual-seed run."""

    manifest_kind: Literal["muggledsam_sam3_manual_seed_targets"]
    config_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    base_g2_config: str = Field(min_length=1)
    view_id: str = Field(pattern=CALIBRATION_VIEW_ID_PATTERN)
    targets: tuple[str, ...] = Field(min_length=1)
    target_descriptors: tuple[MuggledSAMManualSeedTargetDescriptor, ...] = ()

    @model_validator(mode="after")
    def require_distinct_targets(self) -> MuggledSAMManualSeedTargetConfig:
        if len(set(self.targets)) != len(self.targets):
            raise ValueError("manual-seed target config targets must be distinct")
        descriptor_ids = [descriptor.target_id for descriptor in self.target_descriptors]
        if len(set(descriptor_ids)) != len(descriptor_ids):
            raise ValueError("manual-seed target descriptors must be distinct")
        if self.target_descriptors and tuple(descriptor_ids) != self.targets:
            raise ValueError(
                "manual-seed target descriptors must match the ordered target contract"
            )
        return self


class MuggledSAMTextTarget(VersionedModel):
    """One stable output identity and its human-readable SAM3 detector prompt."""

    output_label: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    text_prompt: str = Field(min_length=1)

    @field_validator("text_prompt")
    @classmethod
    def require_trimmed_text_prompt(cls, prompt: str) -> str:
        if prompt != prompt.strip():
            raise ValueError("SAM3 text prompts must not contain leading or trailing whitespace")
        return prompt


class MuggledSAMTextTargetConfig(VersionedModel):
    """Versioned prompt-to-output-label contract for static zero-shot tracking."""

    manifest_kind: Literal["muggledsam_sam3_text_targets"]
    config_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    base_g2_config: str = Field(min_length=1)
    view_id: Literal["static-c10379"]
    targets: tuple[MuggledSAMTextTarget, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_unambiguous_target_mapping(self) -> MuggledSAMTextTargetConfig:
        labels = [target.output_label for target in self.targets]
        prompts = [target.text_prompt for target in self.targets]
        if len(set(labels)) != len(labels):
            raise ValueError("text-target output labels must be distinct")
        if len(set(prompts)) != len(prompts):
            raise ValueError("text-target SAM3 prompts must be distinct")
        return self


class MuggledSAMHybridTarget(VersionedModel):
    """One canonical target and its declared initialization source."""

    output_label: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    initialization_source: Literal["text_prompt", "human_reviewed_mask"]


class MuggledSAMHybridInitializationConfig(VersionedModel):
    """Pinned static contract combining text detections and one reviewed mask."""

    manifest_kind: Literal["muggledsam_sam3_hybrid_initialization"]
    config_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    base_g2_config: str = Field(min_length=1)
    view_id: Literal["static-c10379"]
    text_target_config_fingerprint: ArtifactFingerprint
    manual_seed_proposal_fingerprint: ArtifactFingerprint
    manual_seed_target_config_fingerprint: ArtifactFingerprint
    targets: tuple[MuggledSAMHybridTarget, ...]
    ground_truth_accuracy_claim: Literal[False] = False

    @model_validator(mode="after")
    def require_aligned_static_contract(self) -> MuggledSAMHybridInitializationConfig:
        expected = (
            ("left_hand", "text_prompt"),
            ("right_hand", "text_prompt"),
            ("yellow_toy_top", "text_prompt"),
            ("black_toy_top_base", "human_reviewed_mask"),
        )
        received = tuple(
            (target.output_label, target.initialization_source) for target in self.targets
        )
        if received != expected:
            raise ValueError(
                "hybrid static targets must use the exact canonical order and provenance"
            )
        return self


class MuggledSAMMultiKeyframeCorrectionPolicy(VersionedModel):
    """Declared, conservative memory semantics for human correction keyframes."""

    manifest_kind: Literal["muggledsam_sam3_multi_keyframe_correction_policy"]
    policy_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    policy_version: Literal["1", "2", "3", "4"]
    view_id: Literal["static-c10379", "ego-hmc21110305", "ego-hmc21179183"]
    manual_seed_target_config_fingerprint: ArtifactFingerprint
    targets: tuple[str, ...] = Field(min_length=1)
    maximum_later_correction_keyframes_per_target: int = Field(ge=0, le=8)
    correction_memory_semantics: Literal["replace_prompt_memory_and_reset_frame_memory"]

    @model_validator(mode="after")
    def require_distinct_policy_targets(self) -> MuggledSAMMultiKeyframeCorrectionPolicy:
        if len(set(self.targets)) != len(self.targets):
            raise ValueError("correction policy targets must be distinct")
        if self.policy_version == "1" and self.maximum_later_correction_keyframes_per_target > 3:
            raise ValueError("correction policy v1 permits at most three later keyframes")
        if self.policy_version == "2" and self.maximum_later_correction_keyframes_per_target > 5:
            raise ValueError("correction policy v2 permits at most five later keyframes")
        if self.policy_version == "3" and self.maximum_later_correction_keyframes_per_target > 6:
            raise ValueError("correction policy v3 permits at most six later keyframes")
        return self


class MuggledSAMMultiplexSlot(VersionedModel):
    """Stable human target to SAM3 multiplex-slot association."""

    target_id: str = Field(min_length=1)
    object_id: str = Field(pattern=r"^sam3-\d{2,}$")
    multiplex_slot: int = Field(ge=0)


class MuggledSAMMultiKeyframeCorrection(VersionedModel):
    """One selected source-size mask applied to one multiplex slot at one frame."""

    candidate_id: str = Field(pattern=r"^t\d{6}-b\d{2,}$")
    human_selected_candidate_index: int = Field(ge=0)
    selected_by: Literal["human", "agent"] = "human"
    target_id: str = Field(min_length=1)
    object_id: str = Field(pattern=r"^sam3-\d{2,}$")
    multiplex_slot: int = Field(ge=0)
    frame: CalibrationFrameReference
    calibration_mask_fingerprint: ArtifactFingerprint

    @model_validator(mode="after")
    def require_human_frame_zero_seed(self) -> MuggledSAMMultiKeyframeCorrection:
        if self.selected_by == "agent" and self.frame.analysis_frame_index == 0:
            raise ValueError("frame-0 seeds must be human-selected")
        return self


class MuggledSAMMultiKeyframeCorrectionSchedule(VersionedModel):
    """Integrity-bound, deterministic schedule of frame-zero seeds and later corrections."""

    manifest_kind: Literal["muggledsam_sam3_multi_keyframe_correction_schedule"]
    authority: Literal["proposed_non_authoritative"]
    schedule_version: Literal["1"]
    view_id: Literal["static-c10379", "ego-hmc21110305", "ego-hmc21179183"]
    calibration_manifest_fingerprint: ArtifactFingerprint
    correction_policy_fingerprint: ArtifactFingerprint
    manual_seed_target_config_fingerprint: ArtifactFingerprint
    correction_memory_semantics: Literal["replace_prompt_memory_and_reset_frame_memory"]
    slots: tuple[MuggledSAMMultiplexSlot, ...] = Field(min_length=1)
    corrections: tuple[MuggledSAMMultiKeyframeCorrection, ...] = Field(min_length=1)
    ground_truth_accuracy_claim: Literal[False] = False

    @model_validator(mode="after")
    def require_unambiguous_corrections(self) -> MuggledSAMMultiKeyframeCorrectionSchedule:
        slots_by_index = {slot.multiplex_slot: slot for slot in self.slots}
        if len(slots_by_index) != len(self.slots):
            raise ValueError("correction schedule multiplex slots must be unique")
        if set(slots_by_index) != set(range(len(self.slots))):
            raise ValueError("correction schedule multiplex slots must be contiguous from zero")
        targets = [slot.target_id for slot in self.slots]
        object_ids = [slot.object_id for slot in self.slots]
        if len(set(targets)) != len(targets) or len(set(object_ids)) != len(object_ids):
            raise ValueError("correction schedule targets and object IDs must be unique")

        prior_key: tuple[int, int] | None = None
        seen_candidate_ids: set[str] = set()
        seen_mask_fingerprints: set[tuple[str, str]] = set()
        frame_zero_slots: set[int] = set()
        for correction in self.corrections:
            key = (correction.frame.analysis_frame_index, correction.multiplex_slot)
            if prior_key is not None and key <= prior_key:
                raise ValueError(
                    "correction schedule entries must be ordered and unique by frame and slot"
                )
            prior_key = key
            slot = slots_by_index.get(correction.multiplex_slot)
            if (
                slot is None
                or correction.target_id != slot.target_id
                or correction.object_id != slot.object_id
            ):
                raise ValueError("correction target/object ID must match its multiplex slot")
            if correction.candidate_id in seen_candidate_ids:
                raise ValueError("correction schedule candidate IDs must be unique")
            seen_candidate_ids.add(correction.candidate_id)
            mask_key = (
                correction.calibration_mask_fingerprint.uri,
                correction.calibration_mask_fingerprint.sha256,
            )
            if mask_key in seen_mask_fingerprints:
                raise ValueError("correction schedule masks must not be reused across slots")
            seen_mask_fingerprints.add(mask_key)
            if correction.frame.analysis_frame_index == 0:
                frame_zero_slots.add(correction.multiplex_slot)
        if frame_zero_slots != set(slots_by_index):
            raise ValueError("correction schedule requires exactly one frame-0 mask per slot")
        return self


class MultiKeyframeCorrectionScheduleMetadata(VersionedModel):
    """Run provenance for a schedule applied by the bounded correction runner."""

    schedule_fingerprint: ArtifactFingerprint
    correction_policy_fingerprint: ArtifactFingerprint
    # The semantics the worker ran with.  Authored schedules only ever declare
    # `replace_prompt_memory_and_reset_frame_memory`; a memory arm overrides it from the CLI.
    correction_memory_semantics: CorrectionMemorySemantics
    scheduled_correction_frame_indices: tuple[int, ...] = ()
    # Frames whose correction masks were chosen by agent visual review rather than a human.
    agent_selected_correction_frame_indices: tuple[int, ...] = ()
    # Later corrections the run deliberately did not apply: those past a bounded frame range,
    # those named by `--drop-correction-frame`, or all of them when the run was asked to
    # track from the frame-0 seeds alone.
    dropped_correction_frame_indices: tuple[int, ...] = ()
    # The subset of `dropped_correction_frame_indices` dropped by explicit CLI request.
    cli_dropped_correction_frame_indices: tuple[int, ...] = ()
    frame_zero_seeds_only: bool = False
    ground_truth_accuracy_claim: Literal[False] = False

    @model_validator(mode="after")
    def require_cli_drops_among_dropped(self) -> MultiKeyframeCorrectionScheduleMetadata:
        if not set(self.cli_dropped_correction_frame_indices) <= set(
            self.dropped_correction_frame_indices
        ):
            raise ValueError("CLI-dropped correction frames must be listed among the dropped")
        return self


class MuggledSAMProposedTrackingPromptConfig(VersionedModel):
    """Non-authoritative configuration output from explicit human selections."""

    manifest_kind: Literal["muggledsam_sam3_proposed_tracking_prompt"]
    authority: Literal["proposed_non_authoritative"]
    calibration_manifest_uri: str = Field(min_length=1)
    calibration_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    view_id: Literal["static-c10379", "ego-hmc21110305", "ego-hmc21179183"]
    seeds: tuple[ProposedTrackingSeed, ...] = Field(min_length=1)
    tracker_initialization_limitations: tuple[str, ...] = Field(min_length=1)
    manual_seed_target_config_fingerprint: ArtifactFingerprint | None = None
    ground_truth_accuracy_claim: Literal[False] = False


class ManualSeedCandidateProvenance(VersionedModel):
    """One human-selected calibration mask used to initialize a multiplexed stream."""

    candidate_id: str = Field(pattern=r"^t\d{6}-b\d{2,}$")
    intended_target: str = Field(min_length=1)
    human_selected_candidate_index: int = Field(ge=0)
    calibration_mask_fingerprint: ArtifactFingerprint
    initial_multiplex_slot: int = Field(ge=0)
    display_alias: str | None = Field(default=None, min_length=1)
    mask_semantics: Literal["visible_surface_only"] | None = None


class ManualSeedMultiplexMetadata(VersionedModel):
    """Auditable initialization details for a manual-mask SAM3 multiplex smoke."""

    proposal_fingerprint: ArtifactFingerprint
    calibration_manifest_fingerprint: ArtifactFingerprint
    seeds: tuple[ManualSeedCandidateProvenance, ...] = Field(min_length=1)
    initialization_api: Literal["encode_prompt_memory_from_mask"]
    excluded_candidate_ids: tuple[str, ...] = ()
    ground_truth_accuracy_claim: Literal[False] = False
    calibration_transfer_note: str | None = Field(
        default=None,
        description=(
            "Set when frame-zero seeds authored against one proxy were reused on another proxy "
            "of the same source instant and geometry, naming what was verified to be equal."
        ),
    )

    @model_validator(mode="after")
    def require_unique_targets_and_slots(self) -> ManualSeedMultiplexMetadata:
        targets = [seed.intended_target for seed in self.seeds]
        slots = [seed.initial_multiplex_slot for seed in self.seeds]
        if len(set(targets)) != len(targets):
            raise ValueError("manual multiplex seed targets must be unique")
        if len(set(slots)) != len(slots):
            raise ValueError("manual multiplex seed slots must be unique")
        return self


class HybridTargetInitializationProvenance(VersionedModel):
    """Persisted source and slot for one hybrid-initialized target."""

    output_label: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    initial_multiplex_slot: int = Field(ge=0)
    initialization_source: Literal["text_prompt", "human_reviewed_mask"]
    source_fingerprint: ArtifactFingerprint
    initialized_at_frame_zero: bool | None = None
    text_prompt: str | None = None
    candidate_id: str | None = None
    human_selected_candidate_index: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def require_source_specific_details(self) -> HybridTargetInitializationProvenance:
        if self.initialization_source == "text_prompt":
            if not self.text_prompt or self.candidate_id is not None:
                raise ValueError("text initialization requires only a text prompt")
            if self.human_selected_candidate_index is not None:
                raise ValueError("text initialization cannot claim a human-selected mask")
        elif (
            self.text_prompt is not None
            or self.candidate_id is None
            or self.human_selected_candidate_index is None
        ):
            raise ValueError("reviewed-mask initialization requires candidate provenance")
        return self


class HybridInitializationMetadata(VersionedModel):
    """Auditable mixed text/reviewed-mask initialization for one SAM3 stream."""

    contract_fingerprint: ArtifactFingerprint
    text_target_config_fingerprint: ArtifactFingerprint
    manual_seed_proposal_fingerprint: ArtifactFingerprint
    manual_seed_target_config_fingerprint: ArtifactFingerprint
    calibration_manifest_fingerprint: ArtifactFingerprint
    targets: tuple[HybridTargetInitializationProvenance, ...]
    initialization_api: Literal["encode_prompt_memory_from_mask"]
    method_label: Literal["hybrid_text_and_human_reviewed_mask"]
    ground_truth_accuracy_claim: Literal[False] = False

    @model_validator(mode="after")
    def require_canonical_targets_and_slots(self) -> HybridInitializationMetadata:
        expected = (
            ("left_hand", 0, "text_prompt"),
            ("right_hand", 1, "text_prompt"),
            ("yellow_toy_top", 2, "text_prompt"),
            ("black_toy_top_base", 3, "human_reviewed_mask"),
        )
        received = tuple(
            (
                target.output_label,
                target.initial_multiplex_slot,
                target.initialization_source,
            )
            for target in self.targets
        )
        if received != expected:
            raise ValueError("hybrid initialization metadata must preserve canonical target slots")
        return self


class HybridSmokeHumanApproval(VersionedModel):
    """Exact reviewed smoke evidence authorizing one full hybrid candidate."""

    approved_smoke_manifest_fingerprint: ArtifactFingerprint
    approved_smoke_qa_fingerprint: ArtifactFingerprint
    approved_at: datetime
    approved_by: str = Field(min_length=1)
    approval_statement: str = Field(min_length=1)

    @field_validator("approved_at")
    @classmethod
    def require_timezone(cls, approved_at: datetime) -> datetime:
        if approved_at.tzinfo is None or approved_at.utcoffset() is None:
            raise ValueError("hybrid smoke approval time must be timezone-aware")
        return approved_at


class SmokeRunMetadata(VersionedModel):
    """Audit data specific to one deliberately bounded model smoke run."""

    requested_analysis_frame_range: FrameRange
    requested_seconds: float = Field(gt=0, le=10.0)
    concepts: tuple[str, ...] = Field(min_length=1)
    source_fingerprint: ArtifactFingerprint
    proxy_fingerprint: ArtifactFingerprint
    config_fingerprint: ArtifactFingerprint
    adapter: AdapterMetadata
    continuity: StreamContinuityPolicy
    runtime_settings: dict[str, str | int | float | bool | None]
    measurements: RuntimeMeasurements
    observations_uri: str | None = None
    mask_artifact_uri: str | None = None
    mask_artifact_count: int = Field(ge=0)
    rerun_artifact_uri: str | None = None
    measurements_artifact_uri: str | None = None
    qa_artifact_uri: str | None = None
    manual_seed_multiplex: ManualSeedMultiplexMetadata | None = None
    multi_keyframe_corrections: MultiKeyframeCorrectionScheduleMetadata | None = None
    text_target_config_fingerprint: ArtifactFingerprint | None = None
    text_prompt_mapping: tuple[MuggledSAMTextTarget, ...] = ()
    hybrid_initialization: HybridInitializationMetadata | None = None

    @model_validator(mode="after")
    def require_exact_smoke_budget(self) -> SmokeRunMetadata:
        if self.requested_analysis_frame_range.start_frame != 0:
            raise ValueError("smoke range must start at proxy frame zero")
        if self.requested_seconds != 10.0:
            raise ValueError("smoke duration must be exactly 10.0 seconds")
        analysis_fps = self.runtime_settings.get("analysis_fps")
        permitted_analysis_fps = (30, 60)
        if (
            isinstance(analysis_fps, bool)
            or not isinstance(analysis_fps, (int, float))
            or analysis_fps not in permitted_analysis_fps
        ):
            raise ValueError(
                "runtime_settings.analysis_fps must be numeric and one of "
                f"{permitted_analysis_fps} fps"
            )
        expected_frame_count = round(self.requested_seconds * analysis_fps)
        if self.requested_analysis_frame_range.frame_count != expected_frame_count:
            raise ValueError(
                "smoke frame count must equal requested_seconds * analysis_fps; "
                f"expected {expected_frame_count}, received "
                f"{self.requested_analysis_frame_range.frame_count}"
            )
        if self.hybrid_initialization is not None:
            if self.manual_seed_multiplex is not None:
                raise ValueError("hybrid smoke cannot be labeled as all-manual initialization")
            if tuple(target.output_label for target in self.hybrid_initialization.targets) != (
                self.concepts
            ):
                raise ValueError("hybrid provenance must cover the ordered smoke concepts")
            text_labels = tuple(
                target.output_label
                for target in self.hybrid_initialization.targets
                if target.initialization_source == "text_prompt"
            )
            if tuple(target.output_label for target in self.text_prompt_mapping) != text_labels:
                raise ValueError("hybrid text prompt mapping must cover only text targets")
            if (
                self.text_target_config_fingerprint
                != self.hybrid_initialization.text_target_config_fingerprint
            ):
                raise ValueError("hybrid text target fingerprints must match")
        elif self.text_prompt_mapping:
            labels = tuple(target.output_label for target in self.text_prompt_mapping)
            if labels != self.concepts or self.text_target_config_fingerprint is None:
                raise ValueError(
                    "text prompt mapping must cover the ordered concepts and fingerprint its config"
                )
        elif self.text_target_config_fingerprint is not None:
            raise ValueError("text target config fingerprint requires a prompt mapping")
        return self


class G3CandidateRunMetadata(VersionedModel):
    """Audit data for the explicitly human-approved static full-duration candidate."""

    requested_analysis_frame_range: FrameRange
    requested_seconds: Literal[180.0]
    view_id: Literal["static-c10379"]
    concepts: tuple[str, ...] = Field(min_length=1)
    source_fingerprint: ArtifactFingerprint
    proxy_fingerprint: ArtifactFingerprint
    config_fingerprint: ArtifactFingerprint
    adapter: AdapterMetadata
    continuity: StreamContinuityPolicy
    runtime_settings: dict[str, str | int | float | bool | None]
    measurements: RuntimeMeasurements
    observations_uri: str | None = None
    mask_artifact_uri: str | None = None
    mask_artifact_count: int = Field(ge=0)
    rerun_artifact_uri: str | None = None
    text_target_config_fingerprint: ArtifactFingerprint | None = None
    text_prompt_mapping: tuple[MuggledSAMTextTarget, ...] = ()
    hybrid_initialization: HybridInitializationMetadata | None = None
    hybrid_smoke_human_approval: HybridSmokeHumanApproval | None = None

    @model_validator(mode="after")
    def require_exact_g3_static_budget(self) -> G3CandidateRunMetadata:
        if (
            self.requested_analysis_frame_range.start_frame != 0
            or self.requested_analysis_frame_range.frame_count != 5400
        ):
            raise ValueError("G3 candidate must cover exactly static proxy frames [0, 5400)")
        if self.hybrid_initialization is not None:
            if self.hybrid_smoke_human_approval is None:
                raise ValueError("full hybrid G3 metadata requires exact smoke approval evidence")
            if tuple(target.output_label for target in self.hybrid_initialization.targets) != (
                self.concepts
            ):
                raise ValueError("hybrid provenance must cover the ordered G3 concepts")
            text_labels = tuple(
                target.output_label
                for target in self.hybrid_initialization.targets
                if target.initialization_source == "text_prompt"
            )
            if tuple(target.output_label for target in self.text_prompt_mapping) != text_labels:
                raise ValueError("hybrid text prompt mapping must cover only text targets")
            if (
                self.text_target_config_fingerprint
                != self.hybrid_initialization.text_target_config_fingerprint
            ):
                raise ValueError("hybrid text target fingerprints must match")
        elif self.text_prompt_mapping:
            labels = tuple(target.output_label for target in self.text_prompt_mapping)
            if labels != self.concepts or self.text_target_config_fingerprint is None:
                raise ValueError(
                    "text prompt mapping must cover the ordered concepts and fingerprint its config"
                )
        elif self.text_target_config_fingerprint is not None:
            raise ValueError("text target config fingerprint requires a prompt mapping")
        if self.hybrid_smoke_human_approval is not None and self.hybrid_initialization is None:
            raise ValueError("hybrid smoke approval cannot label a non-hybrid G3 run")
        return self


class E4CandidateRunMetadata(VersionedModel):
    """Audit data for the explicitly human-approved 60-second e4 candidate."""

    requested_analysis_frame_range: FrameRange
    requested_seconds: Literal[60.0]
    view_id: Literal["ego-hmc21179183"]
    concepts: tuple[str, ...] = Field(min_length=1)
    source_fingerprint: ArtifactFingerprint
    proxy_fingerprint: ArtifactFingerprint
    config_fingerprint: ArtifactFingerprint
    adapter: AdapterMetadata
    continuity: StreamContinuityPolicy
    runtime_settings: dict[str, str | int | float | bool | None]
    measurements: RuntimeMeasurements
    observations_uri: str | None = None
    mask_artifact_uri: str | None = None
    mask_artifact_count: int = Field(ge=0)
    rerun_artifact_uri: str | None = None

    @model_validator(mode="after")
    def require_exact_e4_candidate_budget(self) -> E4CandidateRunMetadata:
        if (
            self.requested_analysis_frame_range.start_frame != 0
            or self.requested_analysis_frame_range.frame_count != 1800
        ):
            raise ValueError("e4 candidate must cover exactly proxy frames [0, 1800)")
        return self


class FullEgoManualSeedRunMetadata(VersionedModel):
    """Audit data for the approved full ego manual-seed multiplexed baseline."""

    requested_analysis_frame_range: FrameRange
    requested_seconds: Literal[180.0]
    view_id: Literal["ego-hmc21179183"]
    concepts: tuple[str, ...] = Field(min_length=1)
    source_fingerprint: ArtifactFingerprint
    proxy_fingerprint: ArtifactFingerprint
    config_fingerprint: ArtifactFingerprint
    adapter: AdapterMetadata
    continuity: StreamContinuityPolicy
    runtime_settings: dict[str, str | int | float | bool | None]
    measurements: RuntimeMeasurements
    observations_uri: str | None = None
    mask_artifact_uri: str | None = None
    mask_artifact_count: int = Field(ge=0)
    rerun_artifact_uri: str | None = None
    measurements_artifact_uri: str | None = None
    qa_artifact_uri: str | None = None
    manual_seed_multiplex: ManualSeedMultiplexMetadata

    @model_validator(mode="after")
    def require_exact_full_ego_manual_seed_budget(self) -> FullEgoManualSeedRunMetadata:
        if (
            self.requested_analysis_frame_range.start_frame != 0
            or self.requested_analysis_frame_range.frame_count != 5400
        ):
            raise ValueError(
                "full ego manual-seed baseline must cover exactly proxy frames [0, 5400)"
            )
        return self


class FourPartPilotRunMetadata(VersionedModel):
    """Audit data for the approved 20-second static four-part pilot."""

    requested_analysis_frame_range: FrameRange
    requested_seconds: Literal[20.0]
    view_id: Literal["static-c10379"]
    concepts: tuple[str, ...] = Field(min_length=1)
    source_fingerprint: ArtifactFingerprint
    proxy_fingerprint: ArtifactFingerprint
    config_fingerprint: ArtifactFingerprint
    adapter: AdapterMetadata
    continuity: StreamContinuityPolicy
    runtime_settings: dict[str, str | int | float | bool | None]
    measurements: RuntimeMeasurements
    observations_uri: str | None = None
    mask_artifact_uri: str | None = None
    mask_artifact_count: int = Field(ge=0)
    rerun_artifact_uri: str | None = None
    measurements_artifact_uri: str | None = None
    qa_artifact_uri: str | None = None
    manual_seed_multiplex: ManualSeedMultiplexMetadata | None = None
    multi_keyframe_corrections: MultiKeyframeCorrectionScheduleMetadata | None = None

    @model_validator(mode="after")
    def require_exact_four_part_pilot_budget(self) -> FourPartPilotRunMetadata:
        if (
            self.requested_analysis_frame_range.start_frame != 0
            or self.requested_analysis_frame_range.frame_count != 600
        ):
            raise ValueError("four-part pilot must cover exactly static proxy frames [0, 600)")
        if self.concepts != ("chassis", "interior", "rear_body", "cabin"):
            raise ValueError("four-part pilot requires chassis/interior/rear_body/cabin ordering")
        if (self.manual_seed_multiplex is None) == (self.multi_keyframe_corrections is None):
            raise ValueError(
                "four-part pilot requires exactly one manual-seed or correction-schedule mode"
            )
        return self


class FourPartFullRunMetadata(VersionedModel):
    """Audit data for the known-imperfect full static four-part exploration."""

    requested_analysis_frame_range: FrameRange
    requested_seconds: Literal[196.7]
    view_id: Literal["static-c10379"]
    concepts: tuple[str, ...] = Field(min_length=1)
    source_fingerprint: ArtifactFingerprint
    proxy_fingerprint: ArtifactFingerprint
    config_fingerprint: ArtifactFingerprint
    adapter: AdapterMetadata
    continuity: StreamContinuityPolicy
    runtime_settings: dict[str, str | int | float | bool | None]
    measurements: RuntimeMeasurements
    observations_uri: str | None = None
    mask_artifact_uri: str | None = None
    mask_artifact_count: int = Field(ge=0)
    rerun_artifact_uri: str | None = None
    qa_artifact_uri: str | None = None
    multi_keyframe_corrections: MultiKeyframeCorrectionScheduleMetadata
    known_pilot_failure: Literal["chassis/cabin identity merge after frame-65 correction"]

    @model_validator(mode="after")
    def require_exact_four_part_full_budget(self) -> FourPartFullRunMetadata:
        if (
            self.requested_analysis_frame_range.start_frame != 0
            or self.requested_analysis_frame_range.frame_count != 5901
        ):
            raise ValueError("four-part full run must cover exactly static proxy frames [0, 5901)")
        if self.concepts != ("chassis", "interior", "rear_body", "cabin"):
            raise ValueError(
                "four-part full run requires chassis/interior/rear_body/cabin ordering"
            )
        return self


class FourPartFocusedRunMetadata(VersionedModel):
    """Audit data for a separated-to-assembled focused four-part view.

    The full profile covers proxy frames [0, 2781); the first-minute bound [0, 1800) is the
    same stream stopped early, used for tracker-policy arms compared against the first minute
    of the full reference run.
    """

    requested_analysis_frame_range: FrameRange
    requested_seconds: Literal[92.7, 60.0]
    view_id: Literal["static-c10379", "ego-hmc21110305"]
    concepts: tuple[str, ...] = Field(min_length=1)
    source_fingerprint: ArtifactFingerprint
    proxy_fingerprint: ArtifactFingerprint
    config_fingerprint: ArtifactFingerprint
    adapter: AdapterMetadata
    continuity: StreamContinuityPolicy
    runtime_settings: dict[str, str | int | float | bool | None]
    measurements: RuntimeMeasurements
    observations_uri: str | None = None
    mask_artifact_uri: str | None = None
    mask_artifact_count: int = Field(ge=0)
    rerun_artifact_uri: str | None = None
    qa_artifact_uri: str | None = None
    multi_keyframe_corrections: MultiKeyframeCorrectionScheduleMetadata
    rescope_reason: Literal[
        "old proxy frame 3120 starts with four separated parts before reassembly"
    ]

    @model_validator(mode="after")
    def require_exact_four_part_focused_budget(self) -> FourPartFocusedRunMetadata:
        permitted = {92.7: 2781, 60.0: 1800}
        if (
            self.requested_analysis_frame_range.start_frame != 0
            or self.requested_analysis_frame_range.frame_count != permitted[self.requested_seconds]
        ):
            raise ValueError(
                "focused four-part run must cover exactly focused proxy frames [0, 2781), "
                "or its first-minute bound [0, 1800) declared as 60.0 seconds"
            )
        if self.concepts != ("chassis", "interior", "rear_body", "cabin"):
            raise ValueError(
                "focused four-part run requires chassis/interior/rear_body/cabin ordering"
            )
        return self


class GeometricSeedProvenance(VersionedModel):
    """One agent-authored frame-0 seed transferred from the human masks by geometry."""

    target: str = Field(min_length=1)
    multiplex_slot: int = Field(ge=0)
    mask_fingerprint: ArtifactFingerprint
    prompt_id: str = Field(pattern=r"^t\d{6}-b\d{2,}$")
    candidate_index: int = Field(ge=0)
    backprojection_iou: float | None = Field(default=None, ge=0, le=1)
    area_ratio_vs_expected: float | None = Field(default=None, ge=0)
    selected_by: Literal["agent"] = "agent"
    provenance: Literal["geometric_seed_transfer"] = "geometric_seed_transfer"


class FourPartMultiviewRunMetadata(VersionedModel):
    """Audit data for a first-minute four-part run on a view seeded by geometry, not a human.

    The seeds came from `battle-multiview-seed-transfer` (agent-authored, back-projection IoU
    acceptance); parts the transfer could not seed are listed as blocked and the run tracks
    only the accepted subset.  No later corrections are applied: the run is cross-view
    comparison evidence for the human-corrected C10379 reference, never a reference itself.
    """

    requested_analysis_frame_range: FrameRange
    requested_seconds: Literal[60.0]
    view_id: str = Field(min_length=1)
    concepts: tuple[str, ...] = Field(min_length=2)
    source_fingerprint: ArtifactFingerprint
    proxy_fingerprint: ArtifactFingerprint
    config_fingerprint: ArtifactFingerprint
    adapter: AdapterMetadata
    continuity: StreamContinuityPolicy
    runtime_settings: dict[str, str | int | float | bool | None]
    measurements: RuntimeMeasurements
    observations_uri: str | None = None
    mask_artifact_uri: str | None = None
    mask_artifact_count: int = Field(ge=0)
    rerun_artifact_uri: str | None = None
    qa_artifact_uri: str | None = None
    seed_manifest_fingerprint: ArtifactFingerprint
    seeds: tuple[GeometricSeedProvenance, ...] = Field(min_length=2)
    blocked_targets: dict[str, str] = Field(default_factory=dict)
    seed_provenance: Literal["geometric_seed_transfer"] = "geometric_seed_transfer"
    later_corrections: Literal["none"] = "none"
    ground_truth_accuracy_claim: Literal[False] = False

    @model_validator(mode="after")
    def require_first_minute_and_ordered_seeds(self) -> FourPartMultiviewRunMetadata:
        if (
            self.requested_analysis_frame_range.start_frame != 0
            or self.requested_analysis_frame_range.frame_count != 1800
        ):
            raise ValueError("multiview four-part runs cover exactly analysis frames [0, 1800)")
        if tuple(seed.target for seed in self.seeds) != self.concepts:
            raise ValueError("multiview seeds must match the ordered concepts")
        if [seed.multiplex_slot for seed in self.seeds] != list(range(len(self.seeds))):
            raise ValueError("multiview seeds must fill multiplex slots from zero")
        return self


class FineSubstepRunMetadata(VersionedModel):
    """Audit data for one bounded fine-substep crop CLIP experiment."""

    requested_seconds: float = Field(gt=0, le=60.0)
    source_fingerprint: ArtifactFingerprint
    proxy_fingerprint: ArtifactFingerprint
    config_fingerprint: ArtifactFingerprint
    label_contract_fingerprint: ArtifactFingerprint
    openclip_checkpoint_fingerprint: ArtifactFingerprint
    adapter: AdapterMetadata
    runtime_settings: dict[str, str | int | float | bool | None]
    measurements: RuntimeMeasurements
    scores_uri: str
    evaluation_uri: str
    rerun_artifact_uri: str | None = None
    contact_sheet_uri: str | None = None
    review_guide_uri: str | None = None
    provenance_tag: Literal["agent_authored_visual_review"] = "agent_authored_visual_review"
    agent_review_note: str = Field(
        min_length=1,
        default=(
            "Predicted substeps are compared only to agent-authored visual-review labels "
            "tagged agent_authored_visual_review; this is not benchmark accuracy."
        ),
    )


class DropDTWRunMetadata(VersionedModel):
    """Audit data for one bounded CLIP + Drop-DTW weak-supervision alignment."""

    requested_seconds: float = Field(gt=0, le=60.0)
    source_fingerprint: ArtifactFingerprint
    proxy_fingerprint: ArtifactFingerprint
    config_fingerprint: ArtifactFingerprint
    transcript_fingerprint: ArtifactFingerprint
    openclip_checkpoint_fingerprint: ArtifactFingerprint
    adapter: AdapterMetadata
    runtime_settings: dict[str, str | int | float | bool | None]
    measurements: RuntimeMeasurements
    alignment_uri: str
    rerun_artifact_uri: str | None = None
    weak_supervision_note: str = Field(
        min_length=1,
        default=(
            "Ordered text derives from Assembly101 coarse ground-truth annotations and is "
            "weak supervision only; alignment intervals and cost are not accuracy claims."
        ),
    )
    drop_dtw_revision: str = Field(min_length=1)


class BoxMOTRunMetadata(VersionedModel):
    """Audit data for one bounded BoxMOT association run over independent detections."""

    requested_analysis_frame_range: FrameRange
    requested_seconds: float = Field(gt=0, le=60.0)
    source_fingerprint: ArtifactFingerprint
    proxy_fingerprint: ArtifactFingerprint
    config_fingerprint: ArtifactFingerprint
    detector_fingerprint: ArtifactFingerprint
    adapter: AdapterMetadata
    runtime_settings: dict[str, str | int | float | bool | None]
    measurements: RuntimeMeasurements
    observations_uri: str
    rerun_artifact_uri: str | None = None
    qa_artifact_uri: str | None = None
    detector_source: str = Field(
        min_length=1,
        description="Independent per-frame detector; must not be MuggledSAM track IDs.",
    )
    association_coverage_note: str = Field(
        min_length=1,
        default=(
            "Track identities are conditional on the declared detector source and class "
            "filter; they are not comparable to SAM3 masks or ground-truth object labels."
        ),
    )

    @model_validator(mode="after")
    def require_consistent_budget(self) -> BoxMOTRunMetadata:
        if self.requested_analysis_frame_range.start_frame != 0:
            raise ValueError("BoxMOT range must start at proxy frame zero")
        analysis_fps = self.runtime_settings.get("analysis_fps")
        if isinstance(analysis_fps, bool) or not isinstance(analysis_fps, (int, float)):
            raise ValueError("runtime_settings.analysis_fps must be numeric")
        if self.requested_analysis_frame_range.frame_count != round(
            self.requested_seconds * analysis_fps
        ):
            raise ValueError("BoxMOT frame count must equal requested_seconds * analysis_fps")
        return self


class WiLoRHandsRunMetadata(VersionedModel):
    """Audit data for one bounded WiLoR hand-pose video run."""

    requested_analysis_frame_range: FrameRange
    requested_seconds: float = Field(gt=0, le=60.0)
    source_fingerprint: ArtifactFingerprint
    proxy_fingerprint: ArtifactFingerprint
    config_fingerprint: ArtifactFingerprint
    checkpoint_fingerprint: ArtifactFingerprint
    detector_fingerprint: ArtifactFingerprint
    adapter: AdapterMetadata
    runtime_settings: dict[str, str | int | float | bool | None]
    measurements: RuntimeMeasurements
    observations_uri: str
    native_evidence_uri: str | None = None
    rerun_artifact_uri: str | None = None
    qa_artifact_uri: str | None = None
    license_caveat: str = Field(
        min_length=1,
        default=(
            "WiLoR checkpoints are CC-BY-NC-ND; MANO and Ultralytics carry separate "
            "licenses. Outputs are model estimates in a non-metric camera-relative "
            "frame and are not multi-view reconstruction or ground-truth pose."
        ),
    )

    @model_validator(mode="after")
    def require_consistent_budget(self) -> WiLoRHandsRunMetadata:
        if self.requested_analysis_frame_range.start_frame != 0:
            raise ValueError("WiLoR range must start at proxy frame zero")
        analysis_fps = self.runtime_settings.get("analysis_fps")
        if isinstance(analysis_fps, bool) or not isinstance(analysis_fps, (int, float)):
            raise ValueError("runtime_settings.analysis_fps must be numeric")
        if self.requested_analysis_frame_range.frame_count != round(
            self.requested_seconds * analysis_fps
        ):
            raise ValueError("WiLoR frame count must equal requested_seconds * analysis_fps")
        return self


class GroundingDinoSam2VideoRunMetadata(VersionedModel):
    """Audit data for one bounded HF Grounding DINO + SAM2 video propagation run."""

    requested_analysis_frame_range: FrameRange
    requested_seconds: float = Field(gt=0, le=20.0)
    source_fingerprint: ArtifactFingerprint
    proxy_fingerprint: ArtifactFingerprint
    config_fingerprint: ArtifactFingerprint
    grounding_model_fingerprint: ArtifactFingerprint
    sam2_checkpoint_fingerprint: ArtifactFingerprint
    adapter: AdapterMetadata
    runtime_settings: dict[str, str | int | float | bool | None]
    measurements: RuntimeMeasurements
    observations_uri: str
    native_masks_uri: str
    rerun_artifact_uri: str | None = None
    qa_artifact_uri: str | None = None
    text_prompt: str = Field(min_length=1)
    sam2_model_config: str = Field(min_length=1)
    initialization_note: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_consistent_budget(self) -> GroundingDinoSam2VideoRunMetadata:
        if self.requested_analysis_frame_range.start_frame != 0:
            raise ValueError("Grounding DINO + SAM2 range must start at proxy frame zero")
        analysis_fps = self.runtime_settings.get("analysis_fps")
        if isinstance(analysis_fps, bool) or not isinstance(analysis_fps, (int, float)):
            raise ValueError("runtime_settings.analysis_fps must be numeric")
        if self.requested_analysis_frame_range.frame_count != round(
            self.requested_seconds * analysis_fps
        ):
            raise ValueError(
                "Grounding DINO + SAM2 frame count must equal requested_seconds * analysis_fps"
            )
        return self


class SamuraiVideoRunMetadata(VersionedModel):
    """Audit data for one bounded SAMURAI SAM2 video propagation run."""

    requested_analysis_frame_range: FrameRange
    requested_seconds: float = Field(gt=0, le=20.0)
    source_fingerprint: ArtifactFingerprint
    proxy_fingerprint: ArtifactFingerprint
    config_fingerprint: ArtifactFingerprint
    sam2_checkpoint_fingerprint: ArtifactFingerprint
    adapter: AdapterMetadata
    runtime_settings: dict[str, str | int | float | bool | None]
    measurements: RuntimeMeasurements
    observations_uri: str
    native_masks_uri: str
    rerun_artifact_uri: str | None = None
    qa_artifact_uri: str | None = None
    sam2_model_config: str = Field(min_length=1)
    init_bbox_xywh: tuple[int, int, int, int]
    initialization_note: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_consistent_budget(self) -> SamuraiVideoRunMetadata:
        if self.requested_analysis_frame_range.start_frame != 0:
            raise ValueError("SAMURAI range must start at proxy frame zero")
        analysis_fps = self.runtime_settings.get("analysis_fps")
        if isinstance(analysis_fps, bool) or not isinstance(analysis_fps, (int, float)):
            raise ValueError("runtime_settings.analysis_fps must be numeric")
        if self.requested_analysis_frame_range.frame_count != round(
            self.requested_seconds * analysis_fps
        ):
            raise ValueError("SAMURAI frame count must equal requested_seconds * analysis_fps")
        return self


class Dam4samVideoRunMetadata(VersionedModel):
    """Audit data for one bounded DAM4SAM headless bbox-init tracking run."""

    requested_analysis_frame_range: FrameRange
    requested_seconds: float = Field(gt=0, le=20.0)
    source_fingerprint: ArtifactFingerprint
    proxy_fingerprint: ArtifactFingerprint
    config_fingerprint: ArtifactFingerprint
    dam4sam_config_fingerprint: ArtifactFingerprint
    sam2_checkpoint_fingerprint: ArtifactFingerprint
    adapter: AdapterMetadata
    runtime_settings: dict[str, str | int | float | bool | None]
    measurements: RuntimeMeasurements
    observations_uri: str
    native_masks_uri: str
    rerun_artifact_uri: str | None = None
    qa_artifact_uri: str | None = None
    tracker_name: str = Field(min_length=1)
    sam2_model_config: str = Field(min_length=1)
    init_bbox_xywh: tuple[int, int, int, int]
    initialization_note: str = Field(min_length=1)
    compatibility_note: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_consistent_budget(self) -> Dam4samVideoRunMetadata:
        if self.requested_analysis_frame_range.start_frame != 0:
            raise ValueError("DAM4SAM range must start at proxy frame zero")
        analysis_fps = self.runtime_settings.get("analysis_fps")
        if isinstance(analysis_fps, bool) or not isinstance(analysis_fps, (int, float)):
            raise ValueError("runtime_settings.analysis_fps must be numeric")
        if self.requested_analysis_frame_range.frame_count != round(
            self.requested_seconds * analysis_fps
        ):
            raise ValueError("DAM4SAM frame count must equal requested_seconds * analysis_fps")
        return self


class ExternalPartialRunMetadata(VersionedModel):
    """Provenance for an imported upstream smoke whose original runner is external.

    This deliberately distinguishes an inference-free Battle import from an integrated
    Battle runner.  It records only observed native outputs and never upgrades the
    upstream method's semantics.
    """

    classification: Literal[
        "single_frame_smoke",
        "external_partial",
        "nlf_only_partial",
        "kineo_nlf_only_partial",
        "fixture_smoke",
    ]
    requested_input_fingerprint: ArtifactFingerprint
    native_artifact_fingerprints: tuple[ArtifactFingerprint, ...] = Field(min_length=1)
    adapter: AdapterMetadata
    reproduced_command: str = Field(min_length=1)
    decoded_frame_count: int | None = Field(default=None, ge=1)
    frames_with_normalized_output: int | None = Field(default=None, ge=0)
    source_offset_seconds: float | None = Field(default=None, ge=0)
    measurements: RuntimeMeasurements | None = None
    normalized_artifact_uri: str = Field(min_length=1)
    rerun_artifact_uri: str | None = None
    limitations: tuple[str, ...] = Field(min_length=1)


class MediaPipeHandsRunMetadata(VersionedModel):
    """Audit data for one bounded MediaPipe Hand Landmarker video run."""

    requested_analysis_frame_range: FrameRange
    requested_seconds: float = Field(gt=0, le=60.0)
    source_fingerprint: ArtifactFingerprint
    proxy_fingerprint: ArtifactFingerprint
    config_fingerprint: ArtifactFingerprint
    model_fingerprint: ArtifactFingerprint
    adapter: AdapterMetadata
    runtime_settings: dict[str, str | int | float | bool | None]
    measurements: RuntimeMeasurements
    observations_uri: str
    rerun_artifact_uri: str | None = None
    qa_artifact_uri: str | None = None

    @model_validator(mode="after")
    def require_consistent_budget(self) -> MediaPipeHandsRunMetadata:
        if self.requested_analysis_frame_range.start_frame != 0:
            raise ValueError("MediaPipe Hands range must start at proxy frame zero")
        analysis_fps = self.runtime_settings.get("analysis_fps")
        if isinstance(analysis_fps, bool) or not isinstance(analysis_fps, (int, float)):
            raise ValueError("runtime_settings.analysis_fps must be numeric")
        if self.requested_analysis_frame_range.frame_count != round(
            self.requested_seconds * analysis_fps
        ):
            raise ValueError(
                "MediaPipe Hands frame count must equal requested_seconds * analysis_fps"
            )
        return self


class FourPartTargetInitialization(VersionedModel):
    """The actual source and result of one frame-zero target initialization."""

    target_id: Literal["chassis", "interior", "rear_body", "cabin"]
    source: Literal["open_vocabulary_detection", "reviewed_mask"]
    state: Literal["succeeded", "failed"]
    prompts: tuple[str, ...] = ()
    selected_prompt: str | None = None
    selected_score: float | None = Field(default=None, ge=0, le=1)
    derived_box_xyxy: tuple[int, int, int, int] | None = None
    reviewed_mask_fingerprint: ArtifactFingerprint | None = None
    failure_reason: str | None = None

    @model_validator(mode="after")
    def require_source_specific_evidence(self) -> FourPartTargetInitialization:
        if self.source == "reviewed_mask":
            if self.state != "succeeded" or self.reviewed_mask_fingerprint is None:
                raise ValueError(
                    "reviewed-mask initialization must retain a successful mask fingerprint"
                )
        elif self.reviewed_mask_fingerprint is not None:
            raise ValueError("open-vocabulary initialization cannot claim a reviewed mask")
        if self.state == "failed" and not self.failure_reason:
            raise ValueError("failed initialization requires a recorded reason")
        return self


class Sam2ArmSettings(VersionedModel):
    """Model size, input size and correction-schedule provenance of one SAM2-family arm.

    Present on runs made after the shared-predictor wrapper; older manifests omit it and
    implicitly mean tiny at 1024 with four independent models and no corrections.
    """

    sam2_model: Literal["tiny", "large"]
    sam2_checkpoint_fingerprint: ArtifactFingerprint
    # True when the measured checkpoint SHA-256 was compared against a hard-coded pin.
    sam2_checkpoint_sha256_pinned: bool = False
    # The yaml actually loaded: the checkout's own, or the battle-owned copy whose
    # `image_size` was rewritten for a non-default input size.
    sam2_config_fingerprint: ArtifactFingerprint
    sam2_config_source: Literal["checkout", "battle_input_size_copy"] = "checkout"
    input_image_size: int = Field(ge=256)
    shared_predictor: bool
    correction_schedule_fingerprint: ArtifactFingerprint | None = None
    scheduled_correction_frame_indices: tuple[int, ...] = ()
    dropped_correction_frame_indices: tuple[int, ...] = ()
    correction_api: Literal["add_new_mask"] | None = None
    # DAM4SAM corrects mid-stream right after tracking the frame; the offline SAM2 arms
    # (SAMURAI, control) add every scheduled mask as a conditioning frame before
    # `propagate_in_video`, which is the only order their API allows.
    correction_timing: (
        Literal["mid_stream_after_track", "all_conditioning_frames_before_propagation"] | None
    ) = None
    add_correction_to_drm: bool = False
    correction_masks_resized_to_frame: bool = False
    seed_masks_resized_to_frame: bool = False
    # Smoke-only: the earliest scheduled keyframe's masks applied at this frame instead,
    # so a 10 s run exercises `correct()`; the masks are then geometrically stale.
    smoke_correction_frame: int | None = Field(default=None, ge=1)
    smoke_correction_source_frame: int | None = Field(default=None, ge=1)
    schedule_frame_zero_seeds_match_contract: bool | None = None

    @model_validator(mode="after")
    def require_consistent_correction_provenance(self) -> Sam2ArmSettings:
        scheduled = bool(self.scheduled_correction_frame_indices)
        if scheduled and (
            self.correction_schedule_fingerprint is None
            or self.correction_api is None
            or self.correction_timing is None
        ):
            raise ValueError("scheduled corrections must name their schedule, API and timing")
        if (self.smoke_correction_frame is None) != (self.smoke_correction_source_frame is None):
            raise ValueError("a smoke correction records both its applied and source frame")
        if self.smoke_correction_frame is not None and self.scheduled_correction_frame_indices != (
            self.smoke_correction_frame,
        ):
            raise ValueError("a smoke correction run applies exactly its rebased frame")
        if self.add_correction_to_drm and not scheduled and self.smoke_correction_frame is None:
            raise ValueError("add_correction_to_drm is meaningless without corrections")
        return self


class VramProbe(VersionedModel):
    """`torch.cuda` allocator counters after `frames_processed` analysis frames."""

    frames_processed: int = Field(ge=1)
    memory_allocated_bytes: int = Field(ge=0)
    max_memory_allocated_bytes: int = Field(ge=0)
    memory_reserved_bytes: int | None = Field(default=None, ge=0)


class VramExtrapolation(VersionedModel):
    """Linear projection of the per-frame memory slope between two probes."""

    basis: Literal["linear_between_two_probes"]
    from_frames: tuple[int, int]
    slope_bytes_per_frame: float
    extrapolate_to_frames: int = Field(ge=1)
    projected_allocated_bytes: int = Field(ge=0)
    projected_peak_bytes: int = Field(ge=0)
    limit_bytes: int = Field(ge=0)
    within_limit: bool

    @model_validator(mode="after")
    def require_consistent_projection(self) -> VramExtrapolation:
        if self.from_frames[0] >= self.from_frames[1]:
            raise ValueError("VRAM probes must be ordered by frames processed")
        if self.within_limit != (self.projected_peak_bytes <= self.limit_bytes):
            raise ValueError("within_limit must agree with the projected peak and the limit")
        return self


class FourPartSegmentationRunMetadata(VersionedModel):
    """Provenance for one exact 20 s (or 60 s extension) arm in the four-part comparison.

    A 10 s (300-frame) run is the VRAM smoke for the SAM2 knobs; it is never compared.
    """

    method_arm: Literal[
        "grounding_dino_sam2_open_vocabulary",
        "reviewed_seed_sam2_control",
        "samurai",
        "dam4sam",
    ]
    requested_analysis_frame_range: FrameRange
    requested_seconds: Literal[10.0, 20.0, 60.0]
    target_order: tuple[
        Literal["chassis"], Literal["interior"], Literal["rear_body"], Literal["cabin"]
    ]
    source_fingerprint: ArtifactFingerprint
    proxy_fingerprint: ArtifactFingerprint
    contract_fingerprint: ArtifactFingerprint
    reviewed_schedule_fingerprint: ArtifactFingerprint
    adapter: AdapterMetadata
    runtime_settings: dict[str, str | int | float | bool | None]
    measurements: RuntimeMeasurements
    observations_uri: str
    native_masks_uri: str
    qa_artifact_uri: str | None = None
    target_initializations: tuple[FourPartTargetInitialization, ...]
    drm_memory_additions: dict[str, int] | None = None
    sam2_settings: Sam2ArmSettings | None = None
    vram_probes: tuple[VramProbe, ...] = ()
    vram_extrapolation: VramExtrapolation | None = None
    ground_truth_accuracy_claim: Literal[False] = False

    @model_validator(mode="after")
    def require_exact_ordered_comparison_contract(self) -> FourPartSegmentationRunMetadata:
        if (
            self.requested_analysis_frame_range.start_frame != 0
            or self.requested_analysis_frame_range.frame_count != round(self.requested_seconds * 30)
            or self.target_order != ("chassis", "interior", "rear_body", "cabin")
        ):
            raise ValueError(
                "four-part segmentation arms require ordered frames [0, 600) or [0, 1800) "
                "(or [0, 300) for the SAM2 VRAM smoke)"
            )
        if self.vram_extrapolation is not None and len(self.vram_probes) < 2:
            raise ValueError("a VRAM extrapolation must retain the probes it was computed from")
        received = tuple(item.target_id for item in self.target_initializations)
        if received != self.target_order:
            raise ValueError("initialization records must cover every ordered target")
        if self.method_arm == "grounding_dino_sam2_open_vocabulary":
            if any(
                item.source != "open_vocabulary_detection" for item in self.target_initializations
            ):
                raise ValueError("the open-vocabulary arm cannot use reviewed-mask seeds")
        elif any(item.source != "reviewed_mask" for item in self.target_initializations):
            raise ValueError("control/tracker arms must use the shared reviewed masks")
        if self.method_arm == "dam4sam" and self.drm_memory_additions is None:
            raise ValueError("DAM4SAM arm must retain available DRM diagnostics")
        return self


EnsembleMaskProvenance = Literal[
    "sam3_corrected", "dam4sam_fallback", "hidden_agent_label", "missing"
]


class EnsembleSourceRunReference(VersionedModel):
    """One retained segmentation run the ensemble reference copies masks from."""

    label: Literal["sam3_corrected", "dam4sam_fallback"]
    run_directory_uri: str = Field(min_length=1)
    manifest_fingerprint: ArtifactFingerprint
    observations_fingerprint: ArtifactFingerprint


class EnsembleReferenceRunMetadata(VersionedModel):
    """Provenance for a per-target ensemble review reference assembled from retained runs.

    The ensemble copies whole masks (never blends them) from a primary run, substitutes a
    fallback run's mask only where a checked-in policy rule fires and the substitute passes
    sanity, and writes explicit empty masks for agent-labelled hidden intervals.  It is a
    review reference; cross-method fallback is not an accuracy claim.
    """

    policy_fingerprint: ArtifactFingerprint
    source_fingerprint: ArtifactFingerprint
    proxy_fingerprint: ArtifactFingerprint
    primary_run: EnsembleSourceRunReference
    fallback_run: EnsembleSourceRunReference
    requested_analysis_frame_range: FrameRange
    view_id: str = Field(min_length=1)
    target_order: tuple[
        Literal["chassis"], Literal["interior"], Literal["rear_body"], Literal["cabin"]
    ]
    observations_uri: str = Field(min_length=1)
    masks_uri: str = Field(min_length=1)
    provenance_uri: str = Field(min_length=1)
    provenance_counts: dict[str, dict[str, int]]
    no_blend: Literal[True] = True
    ground_truth_accuracy_claim: Literal[False] = False
    claim_boundaries: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_distinct_labelled_runs(self) -> EnsembleReferenceRunMetadata:
        if self.primary_run.label != "sam3_corrected":
            raise ValueError("the ensemble primary run must be the corrected SAM3 reference")
        if self.fallback_run.label != "dam4sam_fallback":
            raise ValueError("the ensemble fallback run must be the DAM4SAM arm")
        if self.primary_run.run_directory_uri == self.fallback_run.run_directory_uri:
            raise ValueError("primary and fallback runs must differ")
        if self.target_order != ("chassis", "interior", "rear_body", "cabin"):
            raise ValueError("ensemble targets must keep the contract order")
        for target, counts in self.provenance_counts.items():
            if target not in self.target_order:
                raise ValueError(f"provenance counts name an unknown target {target}")
            if sum(counts.values()) != self.requested_analysis_frame_range.frame_count:
                raise ValueError(f"provenance counts for {target} must cover every frame")
        return self


class RunManifest(VersionedModel):
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    clip: ClipManifest
    coverage: FullDurationCoverage
    chunk_policy: ChunkContinuityPolicy
    method_statuses: tuple[MethodStatus, ...]
    observations: tuple[FrameObservations, ...] = ()
    smoke: SmokeRunMetadata | None = None
    g3_candidate: G3CandidateRunMetadata | None = None
    e4_candidate: E4CandidateRunMetadata | None = None
    full_ego_manual_seed: FullEgoManualSeedRunMetadata | None = None
    four_part_pilot: FourPartPilotRunMetadata | None = None
    four_part_full: FourPartFullRunMetadata | None = None
    four_part_focused: FourPartFocusedRunMetadata | None = None
    four_part_multiview: FourPartMultiviewRunMetadata | None = None
    mediapipe_hands: MediaPipeHandsRunMetadata | None = None
    wilor_hands: WiLoRHandsRunMetadata | None = None
    boxmot: BoxMOTRunMetadata | None = None
    drop_dtw: DropDTWRunMetadata | None = None
    fine_substep: FineSubstepRunMetadata | None = None
    grounding_dino_sam2_video: GroundingDinoSam2VideoRunMetadata | None = None
    samurai_video: SamuraiVideoRunMetadata | None = None
    dam4sam_video: Dam4samVideoRunMetadata | None = None
    four_part_segmentation: FourPartSegmentationRunMetadata | None = None
    external_partial: ExternalPartialRunMetadata | None = None
    ensemble_reference: EnsembleReferenceRunMetadata | None = None

    @model_validator(mode="after")
    def require_monotonic_observations(self) -> RunManifest:
        per_view: dict[str, tuple[int, float]] = {}
        for observation in self.observations:
            prior = per_view.get(observation.view_id)
            current = (observation.analysis_frame_index, observation.source_seconds)
            if prior is not None and current <= prior:
                raise ValueError("observations must increase monotonically per view")
            per_view[observation.view_id] = current
        return self


class ExploratoryTimelineAlignment(StrEnum):
    """Whether an exploratory artifact may share the real source timeline."""

    SOURCE_ALIGNED = "source_aligned"
    METADATA_ONLY = "metadata_only"


class ExploratoryComparisonMethod(VersionedModel):
    """One independently interpreted layer in a bounded comparison recording."""

    method_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    display_name: str = Field(min_length=1)
    state: str = Field(min_length=1)
    run_directory_uri: str | None = None
    input_manifest: ArtifactFingerprint
    input_artifacts: tuple[ArtifactFingerprint, ...] = Field(min_length=1)
    inference_input_fingerprints: tuple[ArtifactFingerprint, ...] = ()
    view_id: str | None = None
    analysis_fps: int | None = Field(default=None, gt=0)
    source_offset_seconds: float | None = Field(default=None, ge=0)
    coverage: FullDurationCoverage | None = None
    coordinate_semantics: tuple[str, ...] = Field(min_length=1)
    comparability_limits: tuple[str, ...] = Field(min_length=1)
    timeline_alignment: ExploratoryTimelineAlignment
    default_visible: bool = False

    @model_validator(mode="after")
    def require_timeline_specific_fields(self) -> ExploratoryComparisonMethod:
        if self.timeline_alignment is ExploratoryTimelineAlignment.SOURCE_ALIGNED:
            if (
                self.view_id is None
                or self.analysis_fps is None
                or self.source_offset_seconds is None
                or self.coverage is None
            ):
                raise ValueError("source-aligned methods require view, clock, offset, and coverage")
        elif any(
            value is not None
            for value in (
                self.view_id,
                self.analysis_fps,
                self.source_offset_seconds,
                self.coverage,
            )
        ):
            raise ValueError("metadata-only methods cannot claim a source-timeline coordinate")
        return self


class ExploratoryComparisonIndexManifest(VersionedModel):
    """Reproducible, inference-free index of inputs included in a unified exploration."""

    manifest_kind: Literal["exploratory_first_20s_comparison"]
    comparison_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    clip_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    source_video: ArtifactFingerprint
    bounded_video: ArtifactFingerprint
    bounded_video_frame_count: int = Field(gt=0)
    bounded_video_fps: int = Field(gt=0)
    source_interval: TimeInterval
    methods: tuple[ExploratoryComparisonMethod, ...] = Field(min_length=1)
    output_rrd: ArtifactFingerprint | None = None
    build_notes: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_distinct_method_ids_and_athena_boundary(
        self,
    ) -> ExploratoryComparisonIndexManifest:
        method_ids = [method.method_id for method in self.methods]
        if len(set(method_ids)) != len(method_ids):
            raise ValueError("exploratory comparison method IDs must be unique")
        athena = next((method for method in self.methods if method.method_id == "athena"), None)
        if (
            athena is not None
            and athena.timeline_alignment is not ExploratoryTimelineAlignment.METADATA_ONLY
        ):
            raise ValueError("ATHENA must remain metadata-only until real calibration is available")
        return self


class FourPartSegmentationComparisonMethod(VersionedModel):
    """One mask-producing arm included in the focused four-target visual comparison."""

    method_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    display_name: str = Field(min_length=1)
    run_manifest: ArtifactFingerprint
    observations: ArtifactFingerprint
    native_masks: ArtifactFingerprint | None = None
    initialization_summary: tuple[str, ...] = Field(min_length=1)
    target_coverage: dict[str, int]

    @model_validator(mode="after")
    def require_exact_target_coverage_keys(self) -> FourPartSegmentationComparisonMethod:
        if tuple(self.target_coverage) != ("chassis", "interior", "rear_body", "cabin"):
            raise ValueError("four-part comparison coverage must preserve target order")
        if any(value < 0 or value > 600 for value in self.target_coverage.values()):
            raise ValueError("four-part coverage must be within the 600-frame contract")
        return self


class FourPartSegmentationComparisonIndex(VersionedModel):
    """Inference-free index for the fixed static RGB four-part comparison."""

    manifest_kind: Literal["four_part_segmentation_comparison"]
    comparison_id: Literal["four_part_segmentation_comparison"]
    contract_fingerprint: ArtifactFingerprint
    source_video: ArtifactFingerprint
    bounded_video: ArtifactFingerprint
    frame_count: Literal[600]
    analysis_fps: Literal[30]
    source_interval: TimeInterval
    methods: tuple[FourPartSegmentationComparisonMethod, ...] = Field(min_length=2)
    prior_unified_comparison_uri: str
    ground_truth_accuracy_claim: Literal[False] = False

    @model_validator(mode="after")
    def require_expected_method_roots(self) -> FourPartSegmentationComparisonIndex:
        roots = {method.method_id for method in self.methods}
        if "baseline_sam3" not in roots:
            raise ValueError("comparison requires the existing SAM3 baseline")
        if not roots & {
            "grounding_dino_sam2_open_vocabulary",
            "reviewed_seed_sam2_control",
            "samurai",
            "dam4sam",
        }:
            raise ValueError("comparison requires at least one applicable new arm")
        return self


class InteractionContactDiagnostic(VersionedModel):
    """One source-pixel hand-to-part proximity measurement for review only."""

    analysis_frame_index: int = Field(ge=0, lt=1800)
    hand_source_id: str = Field(min_length=1)
    part_id: Literal["chassis", "interior", "rear_body", "cabin"]
    observation_state: Literal["observed", "missing_hand", "missing_mask", "invalid_mask"]
    palm_distance_pixels: float | None = Field(default=None, ge=0)
    fingertip_distance_pixels: float | None = Field(default=None, ge=0)
    minimum_distance_pixels: float | None = Field(default=None, ge=0)
    inside_mask: bool | None = None
    raw_contact_candidate: bool | None = None
    debounced_contact_candidate: bool | None = None

    @model_validator(mode="after")
    def require_distances_only_when_observed(self) -> InteractionContactDiagnostic:
        values = (
            self.palm_distance_pixels,
            self.fingertip_distance_pixels,
            self.minimum_distance_pixels,
            self.inside_mask,
            self.raw_contact_candidate,
            self.debounced_contact_candidate,
        )
        if self.observation_state == "observed":
            if any(value is None for value in values):
                raise ValueError("observed contact diagnostics require every derived value")
        elif any(value is not None for value in values):
            raise ValueError("missing or invalid masks cannot retain stale contact values")
        return self


class InteractionContactEvent(VersionedModel):
    """A debounced geometry transition, never a ground-truth interaction label."""

    analysis_frame_index: int = Field(ge=0, lt=1800)
    hand_source_id: str = Field(min_length=1)
    part_id: Literal["chassis", "interior", "rear_body", "cabin"]
    event_type: Literal["contact_candidate_start", "contact_candidate_end"]


class InteractionHandDisagreement(VersionedModel):
    """Same-frame spatial assignment between two frame-local hand detections."""

    analysis_frame_index: int = Field(ge=0, lt=1800)
    assignment_state: Literal["matched", "mediapipe_only", "wilor_only"]
    mediapipe_hand_id: str | None = None
    wilor_hand_id: str | None = None
    mean_landmark_distance_pixels: float | None = Field(default=None, ge=0)
    max_landmark_distance_pixels: float | None = Field(default=None, ge=0)
    handedness_disagrees: bool | None = None

    @model_validator(mode="after")
    def require_assignment_evidence(self) -> InteractionHandDisagreement:
        if self.assignment_state == "matched":
            if (
                self.mediapipe_hand_id is None
                or self.wilor_hand_id is None
                or self.mean_landmark_distance_pixels is None
                or self.max_landmark_distance_pixels is None
                or self.handedness_disagrees is None
            ):
                raise ValueError("matched hands require both IDs and disagreement values")
        elif (
            self.mean_landmark_distance_pixels is not None
            or self.max_landmark_distance_pixels is not None
        ):
            raise ValueError("unmatched hands cannot claim landmark disagreement")
        return self


class InteractionReviewPinnedMoment(VersionedModel):
    """A deterministic review bookmark; human disposition intentionally remains pending."""

    analysis_frame_index: int = Field(ge=0, lt=1800)
    source_seconds: float = Field(ge=0)
    categories: tuple[str, ...] = Field(min_length=1)
    rationale: str = Field(min_length=1)
    disposition: Literal["pending"] = "pending"


class SegmentationReviewTrigger(VersionedModel):
    """A geometry review trigger, not a semantic correctness decision."""

    analysis_frame_index: int = Field(ge=0, lt=1800)
    target_id: Literal["chassis", "interior", "rear_body", "cabin"]
    trigger_type: Literal[
        "temporal_iou_lt_0_5",
        "area_ratio_gt_2",
        "centroid_jump_gt_25px",
        "cross_method_iou_lt_0_5_during_hand_presence",
    ]
    value: float = Field(ge=0)


class SegmentationValidityInterval(VersionedModel):
    """Explicit review eligibility for a bounded segmentation interval."""

    start_frame: int = Field(ge=0, lt=1800)
    end_frame_exclusive: int = Field(gt=0, le=1800)
    state: Literal["contact_eligible", "not_contact_eligible"]
    rationale: str = Field(min_length=1)
    provenance: Literal[
        "agent_authored_visual_review", "human_feedback_report", "ensemble_reference_policy"
    ]
    target_id: Literal["chassis", "interior", "rear_body", "cabin"] | None = Field(
        default=None,
        description="Target the interval applies to; None means every reference target.",
    )

    @model_validator(mode="after")
    def require_ordered_range(self) -> SegmentationValidityInterval:
        if self.end_frame_exclusive <= self.start_frame:
            raise ValueError("segmentation validity interval must be positive")
        return self


class InteractionReviewIndexManifest(VersionedModel):
    """Complete reproducibility index for the focused interaction review package."""

    manifest_kind: Literal["interaction_review_first_20s", "interaction_review_first_minute_v4"]
    comparison_id: Literal["interaction_review_first_20s", "interaction_review_first_minute_v4"]
    source_video: ArtifactFingerprint
    bounded_video: ArtifactFingerprint
    frame_count: Literal[600, 1800]
    analysis_fps: Literal[30]
    source_interval: TimeInterval
    reference_segmentation_method: Literal[
        "reviewed_seed_sam2_control", "baseline_sam3", "ensemble_reference"
    ]
    reference_segmentation_manifest: ArtifactFingerprint
    reference_provenance_sidecar: ArtifactFingerprint | None = Field(
        default=None,
        description="Per-frame, per-target mask provenance when the reference is an ensemble.",
    )
    reference_provenance_counts: dict[str, dict[str, int]] | None = None
    assembly101_reference: ArtifactFingerprint | None = Field(
        default=None,
        description=(
            "Manifest of the Assembly101 dataset reference window (hand poses and "
            "fine-grained labels) when the package includes the dataset layer."
        ),
    )
    multiview_consensus: ArtifactFingerprint | None = Field(
        default=None,
        description=(
            "Manifest of the cross-view part consensus (battle-build-multiview-part-consensus) "
            "when the package includes the assembly101_multiview layer."
        ),
    )
    input_artifacts: tuple[ArtifactFingerprint, ...] = Field(min_length=1)
    contact_heuristic: str = Field(min_length=1)
    hand_matching_rule: str = Field(min_length=1)
    coordinate_semantics: tuple[str, ...] = Field(min_length=1)
    claim_boundaries: tuple[str, ...] = Field(min_length=1)
    coverage: dict[str, int]
    contact_diagnostics: tuple[InteractionContactDiagnostic, ...] = ()
    hand_disagreements: tuple[InteractionHandDisagreement, ...] = ()
    contact_events: tuple[InteractionContactEvent, ...] = ()
    segmentation_review_triggers: tuple[SegmentationReviewTrigger, ...] = ()
    segmentation_review_episodes: tuple[SegmentationReviewEpisode, ...] = ()
    segmentation_validity_intervals: tuple[SegmentationValidityInterval, ...] = ()
    pinned_moments: tuple[InteractionReviewPinnedMoment, ...] = Field(min_length=3)
    logged_layers: tuple[str, ...] | None = Field(
        default=None,
        description=(
            "Layers this recording actually logged when a build was narrowed for "
            "iteration. A complete package leaves this unset."
        ),
    )
    output_rrd: ArtifactFingerprint | None = None
    review_guide: ArtifactFingerprint | None = None
    contact_sheet: ArtifactFingerprint | None = None
    candidate_arms: dict[str, ArtifactFingerprint] | None = Field(
        default=None,
        description=(
            "Further C10379 segmentation runs logged beside the reference "
            "(`--candidate-arm NAME=RUN_DIR`), keyed by display name -> run manifest."
        ),
    )
    confidence_series: ArtifactFingerprint | None = Field(
        default=None,
        description="`confidence.jsonl` of the detector scorecard logged as a time series.",
    )
    anchor_marks: tuple[ArtifactFingerprint, ...] | None = Field(
        default=None,
        description="Human anchor config (and exported mask set) marked on the timeline.",
    )
    application_id: str | None = None
    recording_id: str | None = None


class AgentAuthoredVisualFinding(VersionedModel):
    """A non-human review observation; never an approval or ground-truth claim."""

    analysis_frame_index: int = Field(ge=0, lt=1800)
    subject: str = Field(min_length=1)
    finding: str = Field(min_length=1)
    evidence_kind: Literal["audit", "generated_contact_sheet", "derived_metric"]
    disposition: Literal["agent_authored_visual_review"]


class SegmentationCorrectionCandidate(VersionedModel):
    """A correction triage row that retains human and agent provenance separately."""

    analysis_frame_index: int = Field(ge=0, lt=1800)
    target_id: Literal["chassis", "interior", "rear_body", "cabin"]
    source_method: str = Field(min_length=1)
    provenance: Literal["human_verified_correction", "agent_authored_visual_review"]
    state: Literal["retained", "agent_proposed", "not_selected"]
    rationale_fingerprint: str = Field(min_length=1)
    semantic_claim: str = Field(min_length=1)


class OvernightReviewRecord(VersionedModel):
    """Structured mixed-provenance review record for the overnight rebuild."""

    manifest_kind: Literal["overnight_interaction_review_record"]
    provenance_tag: Literal["agent_authored_visual_review"]
    human_feedback: tuple[str, ...] = Field(min_length=1)
    agent_findings: tuple[AgentAuthoredVisualFinding, ...] = ()
    correction_candidates: tuple[SegmentationCorrectionCandidate, ...] = ()
    human_decisions_pending: Literal[True] = True
    ground_truth_accuracy_claim: Literal[False] = False


class HumanFeedbackItem(VersionedModel):
    """One verbatim human observation; the agent adds only a subject and frame hint."""

    subject: str = Field(min_length=1)
    verbatim: str = Field(min_length=1)
    frame_hint: str | None = None
    agent_action_subjects: tuple[str, ...] = ()


class HumanFeedbackReviewRecord(VersionedModel):
    """A human-authored review of a delivered package, kept apart from agent findings.

    The `feedback` text is the human's own words and carries no pass/fail beyond what was
    said. `agent_actions` are the agent's responses and stay tagged as agent-authored.
    """

    manifest_kind: Literal["human_feedback_review_record"]
    author_type: Literal["human"]
    provenance_tag: Literal["human_feedback_report"]
    reviewed_package: str = Field(min_length=1)
    reviewed_recording_uri: str = Field(min_length=1)
    reviewed_on: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    feedback: tuple[HumanFeedbackItem, ...] = Field(min_length=1)
    agent_actions: tuple[AgentAuthoredVisualFinding, ...] = ()
    human_decisions_pending: Literal[True] = True
    ground_truth_accuracy_claim: Literal[False] = False

    @model_validator(mode="after")
    def require_actions_to_reference_feedback(self) -> HumanFeedbackReviewRecord:
        subjects = {item.subject for item in self.agent_actions}
        for item in self.feedback:
            missing = set(item.agent_action_subjects) - subjects
            if missing:
                raise ValueError(f"feedback references unknown agent actions: {sorted(missing)}")
        return self


class HumanQAEvidence(VersionedModel):
    """One portable, content-addressed visual artifact presented to a reviewer."""

    uri: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    artifact_kind: Literal["two_checkpoint_contact_sheet", "checkpoint_image"]

    @field_validator("uri")
    @classmethod
    def require_portable_evidence_uri(cls, uri: str) -> str:
        path = PurePosixPath(uri)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("human QA evidence must use a repository-relative portable URI")
        return uri


class HumanQACheckpoint(VersionedModel):
    """One fixed source-timeline instant and its human-authored disposition."""

    role: HumanQACheckpointRole
    clock: Literal[ClockName.SOURCE]
    source_seconds: float = Field(ge=0)
    source_frame_index: int = Field(ge=0)
    analysis_frame_index: int = Field(ge=0)
    evidence: tuple[HumanQAEvidence, ...] = Field(min_length=1)
    disposition: HumanQADisposition = HumanQADisposition.PENDING
    notes: str | None = None
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None

    @model_validator(mode="after")
    def require_human_attribution_for_decisions(self) -> HumanQACheckpoint:
        if self.disposition is HumanQADisposition.PENDING:
            if self.reviewed_by is not None or self.reviewed_at is not None:
                raise ValueError("pending human QA checkpoints cannot claim reviewer attribution")
            return self
        if not self.reviewed_by or self.reviewed_at is None:
            raise ValueError("non-pending human QA decisions require reviewer identity and time")
        if self.reviewed_at.tzinfo is None or self.reviewed_at.utcoffset() is None:
            raise ValueError("human QA review time must include a timezone")
        return self


def derive_human_qa_status(
    checkpoints: tuple[HumanQACheckpoint, ...],
) -> HumanQADisposition:
    """Aggregate without weakening a known human flag or failure."""

    dispositions = {checkpoint.disposition for checkpoint in checkpoints}
    if HumanQADisposition.FAIL in dispositions:
        return HumanQADisposition.FAIL
    if HumanQADisposition.FLAG in dispositions:
        return HumanQADisposition.FLAG
    if dispositions == {HumanQADisposition.PASS}:
        return HumanQADisposition.PASS
    return HumanQADisposition.PENDING


class FixedTimestampHumanQARecord(VersionedModel):
    """Auditable hard-gate record for the plan's two fixed semantic checkpoints."""

    manifest_kind: Literal["fixed_timestamp_human_qa"]
    qa_protocol: Literal["assembly101_easy_hard_source_timestamps_v1"]
    selection_rule: Literal["one_easy_manipulation_and_one_hard_or_occluded_manipulation"]
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    run_profile: str = Field(min_length=1)
    method_id: str = Field(min_length=1)
    clip_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    view_id: str = Field(min_length=1)
    run_manifest_fingerprint: ArtifactFingerprint
    config_fingerprint: ArtifactFingerprint
    source_video_fingerprint: ArtifactFingerprint
    timing: TimingModel
    run_analysis_frame_range: FrameRange
    checkpoints: tuple[HumanQACheckpoint, HumanQACheckpoint]
    overall_status: HumanQADisposition
    ground_truth_accuracy_claim: Literal[False] = False

    @model_validator(mode="after")
    def require_fixed_source_checkpoints(self) -> FixedTimestampHumanQARecord:
        expected_roles = (
            HumanQACheckpointRole.EASY,
            HumanQACheckpointRole.HARD,
        )
        if tuple(checkpoint.role for checkpoint in self.checkpoints) != expected_roles:
            raise ValueError("human QA requires exactly one ordered easy and one hard checkpoint")

        source_fps = self.timing.clocks.fps_for(ClockName.SOURCE)
        seen_instants: set[tuple[float, int]] = set()
        for checkpoint in self.checkpoints:
            if not (
                self.run_analysis_frame_range.start_frame
                <= checkpoint.analysis_frame_index
                < self.run_analysis_frame_range.end_frame_exclusive
            ):
                raise ValueError("human QA analysis checkpoint must be inside the completed run")
            expected_seconds = self.timing.source_seconds_for_frame(
                ClockName.ANALYSIS, checkpoint.analysis_frame_index
            )
            if abs(checkpoint.source_seconds - expected_seconds) > 1e-9:
                raise ValueError(
                    "human QA source timestamp must match its analysis frame and source mapping"
                )
            expected_source_frame = round(checkpoint.source_seconds * source_fps)
            if (
                abs(checkpoint.source_seconds - expected_source_frame / source_fps) > 1e-9
                or checkpoint.source_frame_index != expected_source_frame
            ):
                raise ValueError("human QA source timestamp must match its source frame")
            instant = (checkpoint.source_seconds, checkpoint.source_frame_index)
            if instant in seen_instants:
                raise ValueError("human QA checkpoints must use distinct source timestamps")
            seen_instants.add(instant)

        expected_status = derive_human_qa_status(self.checkpoints)
        if self.overall_status is not expected_status:
            raise ValueError(
                f"human QA overall_status must be conservatively derived as {expected_status.value}"
            )

        for fingerprint in (
            self.run_manifest_fingerprint,
            self.config_fingerprint,
            self.source_video_fingerprint,
        ):
            path = PurePosixPath(fingerprint.uri)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("human QA provenance must use repository-relative portable URIs")
        return self


# -- detector scorecard (Track A) ----------------------------------------------------------
#
# Every number below is scored against the human review anchors: 13 frames of one view,
# review evidence and not ground truth. Counts travel with every rate.


class DetectorOperatingPoint(VersionedModel):
    """Confusion counts at one suspicion threshold (predict failed when score >= threshold)."""

    # None when the threshold differs per fold (leave-one-frame-out pooled counts).
    threshold: float | None = None
    tp: int = Field(ge=0)
    fp: int = Field(ge=0)
    fn: int = Field(ge=0)
    tn: int = Field(ge=0)
    precision: float | None = Field(default=None, ge=0, le=1)
    recall: float | None = Field(default=None, ge=0, le=1)
    f1: float | None = Field(default=None, ge=0, le=1)


class DetectorScore(VersionedModel):
    """One detector on one cell subset (`all` or a failure class)."""

    detector: str = Field(min_length=1)
    subset: str = Field(min_length=1)
    cells: int = Field(ge=0)
    positives: int = Field(ge=0)
    negatives: int = Field(ge=0)
    undefined: int = Field(ge=0)
    auroc: float | None = Field(default=None, ge=0, le=1)
    best_f1: DetectorOperatingPoint | None = None
    recall_floor: DetectorOperatingPoint | None = None
    # For class subsets: the same cells at the thresholds chosen on `all`.
    at_overall_best_f1: DetectorOperatingPoint | None = None
    at_overall_recall_floor: DetectorOperatingPoint | None = None


class DetectorTruthCell(VersionedModel):
    analysis_frame_index: int = Field(ge=0)
    target: str = Field(min_length=1)
    failure_class: str = Field(min_length=1)
    anchor_state: str = Field(min_length=1)
    outcome: str = Field(min_length=1)
    iou: float | None = Field(default=None, ge=0, le=1)
    run_area: int | None = Field(default=None, ge=0)
    failed: bool | None = None


class LeaveOneFrameOutFold(VersionedModel):
    held_out_frame: int = Field(ge=0)
    selected_detectors: tuple[str, ...]
    best_f1_threshold: float | None = None
    recall_floor_threshold: float | None = None


class LeaveOneFrameOutResult(VersionedModel):
    folds: tuple[LeaveOneFrameOutFold, ...]
    pooled_auroc: float | None = Field(default=None, ge=0, le=1)
    best_f1: DetectorOperatingPoint | None = None
    recall_floor: DetectorOperatingPoint | None = None


class DetectorConfidenceRow(VersionedModel):
    """One frame x part row of the confidence series (`confidence.jsonl`)."""

    analysis_frame_index: int = Field(ge=0)
    target: str = Field(min_length=1)
    detectors: dict[str, float | None]
    combined_suspicion: float | None = Field(default=None, ge=0, le=1)
    confidence: float | None = Field(default=None, ge=0, le=1)
    abstain: bool
    is_anchor_frame: bool = False
    anchor_truth_failed: bool | None = None


class DetectorScorecardManifest(VersionedModel):
    manifest_kind: Literal["detector_scorecard"]
    run_name: str = Field(min_length=1)
    run_directory: str = Field(min_length=1)
    run_observations: ArtifactFingerprint
    anchor_set: ArtifactFingerprint
    comparators: dict[str, str]
    view: str = Field(min_length=1)
    frame_count: int = Field(ge=1)
    targets: tuple[str, ...] = Field(min_length=1)
    anchor_frames: tuple[int, ...] = Field(min_length=1)
    truth_rule: str = Field(min_length=1)
    failure_class_frames: dict[str, tuple[int, ...]]
    class_counts: dict[str, dict[str, int]]
    truth_cells: tuple[DetectorTruthCell, ...] = Field(min_length=1)
    detectors: tuple[str, ...] = Field(min_length=1)
    detector_definitions: dict[str, str]
    scores: tuple[DetectorScore, ...]
    top_detectors: tuple[str, ...]
    combined: tuple[DetectorScore, ...]
    leave_one_frame_out: LeaveOneFrameOutResult
    abstain_confidence_threshold: float | None = Field(default=None, ge=0, le=1)
    confidence_uri: str = Field(min_length=1)
    plot_uri: str | None = None
    proposed_anchors_uri: str | None = None
    generated_at: datetime
    claim_boundary: str = Field(min_length=1)


class ProposedAnchorFrame(VersionedModel):
    analysis_frame_index: int = Field(ge=0)
    selection: Literal["detector_ranked", "random"]
    rank: int | None = Field(default=None, ge=1)
    combined_suspicion: float | None = Field(default=None, ge=0, le=1)
    target: str | None = None
    reason: str = Field(min_length=1)


class ProposedAnchorFrames(VersionedModel):
    manifest_kind: Literal["proposed_anchor_frames"]
    run_name: str = Field(min_length=1)
    scorecard_uri: str = Field(min_length=1)
    frame_step: int = Field(ge=1)
    exclusion_radius_frames: int = Field(ge=0)
    min_spacing_frames: int = Field(default=0, ge=0)
    existing_anchor_frames: tuple[int, ...]
    random_seed: int
    frames: tuple[ProposedAnchorFrame, ...]
    claim_boundary: str = Field(min_length=1)


# -- multiview re-prompt loop (B4) ------------------------------------------------------------
# Agent-authored later-frame corrections proposed by the eight-view consensus: a contradiction
# onset becomes geometric prompts in the target view, the SAM3 image decoder answers, a
# geometric rule picks one candidate, and the mask enters a derived correction schedule with
# `selected_by: agent`, provenance `multiview_consensus`.  No human reviews any of it.

MULTIVIEW_CONSENSUS_PROVENANCE = "multiview_consensus"
MULTIVIEW_REPROMPT_MAX_ITERATIONS = 3
MULTIVIEW_REPROMPT_ARMS = ("consensus-only", "human-plus-consensus")
MultiviewRepromptArm = Literal["consensus-only", "human-plus-consensus"]


class MultiviewRepromptDetectorConfig(VersionedModel):
    """The contradiction gate that turns consensus episodes into re-prompt onsets.

    The defaults are the consensus builder's own rules plus a majority size; a Track A
    calibrated detector file carries the same keys and replaces them (`source: file`).
    """

    threshold_px: float = Field(gt=0)
    agreement_px: float = Field(gt=0)
    min_agreeing_static_views: int = Field(ge=2)
    min_episode_frames: int = Field(ge=1)
    merge_gap_frames: int = Field(ge=0)
    min_run_frames: int = Field(ge=1)
    source: Literal["default", "file"] = "default"
    source_fingerprint: ArtifactFingerprint | None = None
    description: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_fingerprint_for_files(self) -> MultiviewRepromptDetectorConfig:
        if (self.source == "file") != (self.source_fingerprint is not None):
            raise ValueError("a detector config read from a file must fingerprint that file")
        return self


class MultiviewRepromptPrompt(VersionedModel):
    """One box prompt (plus negative points) for the target view at a contradiction onset."""

    prompt_id: str = Field(pattern=r"^t\d{6}-b\d{2,}$")
    target: str = Field(min_length=1)
    variant: str = Field(min_length=1)
    pixel_box: PixelBox
    background_points: tuple[PixelPoint, ...] = ()
    background_point_sources: tuple[str, ...] = ()

    @model_validator(mode="after")
    def require_one_source_per_point(self) -> MultiviewRepromptPrompt:
        if len(self.background_point_sources) != len(self.background_points):
            raise ValueError("every background point names what it marks")
        return self


class MultiviewContradictionOnset(VersionedModel):
    """The first frame of a merged run where the target view contradicts the majority."""

    target: str = Field(min_length=1)
    onset_frame: int = Field(ge=1)
    interval_end_frame_exclusive: int = Field(gt=0)
    source_episodes: tuple[tuple[int, int], ...] = Field(min_length=1)
    target_error_px: float = Field(ge=0)
    max_target_error_px: float = Field(ge=0)
    consensus_world_mm: tuple[float, float, float]
    views_used: tuple[str, ...] = Field(min_length=2)
    static_views_used: int = Field(ge=0)
    view_error_px: dict[str, float] = Field(default_factory=dict)
    radius_mm: float | None = Field(default=None, ge=0)
    depth_mm: float | None = None
    projected_centroid_proxy_px: tuple[float, float] | None = None
    expected_area_px: float | None = Field(default=None, ge=0)
    height_above_table_mm: float | None = None
    prompts: tuple[MultiviewRepromptPrompt, ...] = ()
    status: Literal["planned", "blocked"]
    blocked_reason: str | None = None

    @model_validator(mode="after")
    def require_consistent_status(self) -> MultiviewContradictionOnset:
        if self.status == "planned" and (not self.prompts or self.blocked_reason):
            raise ValueError("a planned onset carries prompts and no blocked reason")
        if self.status == "blocked" and (self.prompts or not self.blocked_reason):
            raise ValueError("a blocked onset carries a reason and no prompts")
        if self.interval_end_frame_exclusive <= self.onset_frame:
            raise ValueError("an onset's interval ends after it starts")
        return self


class MultiviewRepromptPlan(VersionedModel):
    """Geometric re-prompts for one target view, derived from a consensus run (CPU)."""

    manifest_kind: Literal["multiview_reprompt_plan"]
    target_view: str = Field(min_length=1)
    target_view_id: str = Field(min_length=1)
    iteration: int = Field(ge=1, le=MULTIVIEW_REPROMPT_MAX_ITERATIONS)
    max_iterations: Literal[3] = MULTIVIEW_REPROMPT_MAX_ITERATIONS
    previous_plan: ArtifactFingerprint | None = None
    consensus_root_uri: str = Field(min_length=1)
    consensus_manifest: ArtifactFingerprint
    consensus_points: ArtifactFingerprint
    consensus_includes_target_view: bool
    consensus_reference_run_uri: str | None = None
    reference_dependency_note: str = Field(min_length=1)
    source_run_uri: str = Field(min_length=1)
    source_run_manifest: ArtifactFingerprint
    source_run_profile: Literal["four_part_static_focused", "four_part_multiview"]
    source_schedule: ArtifactFingerprint | None = None
    source_calibration_manifest: ArtifactFingerprint | None = None
    source_geometric_seed_manifest: ArtifactFingerprint | None = None
    source_checkpoint_frames: tuple[int, ...] = ()
    source_correction_frames: tuple[int, ...] = ()
    clip_config: ArtifactFingerprint
    proxy: ArtifactFingerprint
    proxy_dimensions: tuple[int, int]
    proxy_to_raw_scale: float = Field(gt=0)
    frame_count: int = Field(ge=1)
    detector: MultiviewRepromptDetectorConfig
    box_margins: tuple[float, ...] = Field(min_length=1)
    hand_negatives: bool = False
    onsets: tuple[MultiviewContradictionOnset, ...] = ()
    selected_by: Literal["agent"] = "agent"
    provenance: Literal["multiview_consensus"] = MULTIVIEW_CONSENSUS_PROVENANCE
    claim_boundaries: tuple[str, ...] = Field(min_length=1)

    @property
    def planned_onsets(self) -> tuple[MultiviewContradictionOnset, ...]:
        return tuple(onset for onset in self.onsets if onset.status == "planned")


class RepromptCandidateScore(VersionedModel):
    """One decoded mask and the geometric evidence the acceptance rule saw."""

    prompt_id: str = Field(pattern=r"^t\d{6}-b\d{2,}$")
    calibration_candidate_id: str | None = Field(default=None, pattern=r"^t\d{6}-b\d{2,}$")
    candidate_index: int = Field(ge=0)
    mask: ArtifactFingerprint
    decoder_iou_estimate: float
    mask_area_px: int = Field(ge=0)
    area_ratio_vs_expected: float | None = Field(default=None, ge=0)
    centroid_ray_distance_mm: float | None = Field(default=None, ge=0)
    centroid_ray_distance_radii: float | None = Field(default=None, ge=0)
    passed: bool
    notes: tuple[str, ...] = ()


class RepromptDecision(VersionedModel):
    """Accept or reject, with every alternative, for one onset of one target."""

    target: str = Field(min_length=1)
    onset_frame: int = Field(ge=1)
    decision: Literal["accepted", "rejected"]
    reason: str = Field(min_length=1)
    accepted: RepromptCandidateScore | None = None
    candidates: tuple[RepromptCandidateScore, ...] = ()
    selected_by: Literal["agent"] = "agent"
    provenance: Literal["multiview_consensus"] = MULTIVIEW_CONSENSUS_PROVENANCE

    @model_validator(mode="after")
    def require_consistent_decision(self) -> RepromptDecision:
        if (self.decision == "accepted") != (self.accepted is not None):
            raise ValueError("an accepted decision names its candidate; a rejection names none")
        if self.accepted is not None and not self.accepted.passed:
            raise ValueError("an accepted candidate must pass the acceptance rule")
        return self


class MultiviewConsensusCorrectionProvenance(VersionedModel):
    """Why one agent-authored mask sits in a derived correction schedule."""

    candidate_id: str = Field(pattern=r"^t\d{6}-b\d{2,}$")
    target: str = Field(min_length=1)
    onset_frame: int = Field(ge=1)
    iteration: int = Field(ge=1, le=MULTIVIEW_REPROMPT_MAX_ITERATIONS)
    selected_by: Literal["agent"] = "agent"
    provenance: Literal["multiview_consensus"] = MULTIVIEW_CONSENSUS_PROVENANCE
    consensus_manifest: ArtifactFingerprint
    consensus_includes_target_view: bool
    consensus_reference_run_uri: str | None = None
    reference_dependency_note: str = Field(min_length=1)
    views_used: tuple[str, ...] = Field(min_length=2)
    target_error_px: float = Field(ge=0)
    accepted: RepromptCandidateScore
    rejected_alternatives: tuple[RepromptCandidateScore, ...] = ()
    mask: ArtifactFingerprint


class MultiviewConsensusProvenanceFile(VersionedModel):
    """Sidecar beside a derived calibration: every consensus correction it carries."""

    manifest_kind: Literal["multiview_consensus_correction_provenance"]
    derived_calibration_manifest: ArtifactFingerprint
    carried_from: ArtifactFingerprint | None = None
    corrections: tuple[MultiviewConsensusCorrectionProvenance, ...] = ()

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(item.candidate_id for item in self.corrections)


class MultiviewRepromptDecisions(VersionedModel):
    """Output of `battle-multiview-reprompt decode`: decisions plus the derived schedules."""

    manifest_kind: Literal["multiview_reprompt_decisions"]
    plan: ArtifactFingerprint
    target_view: str = Field(min_length=1)
    iteration: int = Field(ge=1, le=MULTIVIEW_REPROMPT_MAX_ITERATIONS)
    decode_state: Literal["decoded"]
    decisions: tuple[RepromptDecision, ...] = ()
    accepted_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    derived_calibration_manifest: ArtifactFingerprint | None = None
    schedules: dict[str, ArtifactFingerprint] = Field(default_factory=dict)
    schedule_blocked_reason: str | None = None
    provenance_uri: str | None = None
    correction_policy: ArtifactFingerprint | None = None
    source_corrections_dropped_out_of_range: tuple[int, ...] = ()
    runtime_seconds: float = Field(ge=0)
    selected_by: Literal["agent"] = "agent"
    provenance: Literal["multiview_consensus"] = MULTIVIEW_CONSENSUS_PROVENANCE
    claim_boundaries: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_counts_to_match(self) -> MultiviewRepromptDecisions:
        accepted = sum(d.decision == "accepted" for d in self.decisions)
        if accepted != self.accepted_count or len(self.decisions) - accepted != self.rejected_count:
            raise ValueError("decision counts must match the decisions listed")
        if bool(self.schedules) == (self.schedule_blocked_reason is not None):
            raise ValueError("either schedules were written or a reason says why not")
        return self


class MultiviewRepromptRunCommand(VersionedModel):
    """The exact tracker command for one arm, as emitted by `battle-multiview-reprompt run`."""

    arm: MultiviewRepromptArm
    schedule: ArtifactFingerprint
    mode: Literal["full_run", "checkpoint_resumed", "blocked"]
    argv: tuple[str, ...] = Field(min_length=1)
    resume_run_uri: str | None = None
    resume_at: int | None = Field(default=None, ge=1)
    resume_argv: tuple[str, ...] | None = None
    later_correction_frames: tuple[int, ...] = ()
    consensus_correction_frames: tuple[int, ...] = ()
    dropped_human_correction_frames: tuple[int, ...] = ()
    note: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_resume_fields_together(self) -> MultiviewRepromptRunCommand:
        if (self.resume_run_uri is None) != (self.resume_at is None):
            raise ValueError("a resume names both the prior run and the frame")
        if self.mode == "checkpoint_resumed" and self.resume_at is None:
            raise ValueError("a checkpoint-resumed command names where it resumes")
        return self
