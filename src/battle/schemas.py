"""Versioned, inference-agnostic artifact contracts."""

from __future__ import annotations

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
    """The four independently named time bases used in the lab."""

    clocks: tuple[ClockSpec, ...]

    @model_validator(mode="after")
    def require_planned_clocks(self) -> ClockSet:
        expected = {
            ClockName.SOURCE: 60,
            ClockName.ANALYSIS: 30,
            ClockName.ANNOTATION: 30,
            ClockName.POSE: 60,
        }
        received = {clock.name: clock.fps for clock in self.clocks}
        if received != expected:
            raise ValueError(f"clock set must be exactly {expected}")
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
    scaling_policy: Literal["preserve_aspect_ratio_height_720"]
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


class PerFrameHand(VersionedModel):
    hand_id: str = Field(min_length=1)
    side: HandSide
    confidence: float = Field(ge=0, le=1)
    landmarks: tuple[NormalizedPoint, ...] = Field(min_length=1)


class FrameObservations(VersionedModel):
    view_id: str = Field(min_length=1)
    analysis_frame_index: int = Field(ge=0)
    source_seconds: float = Field(ge=0)
    objects: tuple[PerFrameObject, ...] = ()
    hands: tuple[PerFrameHand, ...] = ()


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

    mode: Literal["continuous_stream"] = "continuous_stream"
    intentional_id_resets: Literal[False] = False
    max_prompt_memory_entries: int = Field(ge=1)
    max_frame_memory_entries: int = Field(ge=1)
    detected_object_limit: int = Field(ge=1)


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
    human_selected_candidate_index: int | None = Field(default=None, ge=0)
    human_accepted: bool = False
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
        if self.selected_for_finalization and not self.human_accepted:
            raise ValueError("finalization eligibility requires explicit human mask acceptance")
        if self.selected_for_correction and not self.human_accepted:
            raise ValueError("correction eligibility requires explicit human mask acceptance")
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
                raise ValueError(
                    f"{item_name} normalized {kind} point must match its pixel point"
                )


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


class MuggledSAMBoxCalibrationManifest(VersionedModel):
    """Gitignored, reviewable manual-box calibration for the selected e4 proxy."""

    manifest_kind: Literal["muggledsam_sam3_box_calibration"]
    calibration_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    base_g2_config: str = Field(min_length=1)
    base_g2_config_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    view_id: Literal["ego-hmc21179183"]
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


class MuggledSAMManualSeedTargetConfig(VersionedModel):
    """Named target policy for a human-selected e4 manual-seed run."""

    manifest_kind: Literal["muggledsam_sam3_manual_seed_targets"]
    config_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    base_g2_config: str = Field(min_length=1)
    view_id: Literal["ego-hmc21179183"]
    targets: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_distinct_targets(self) -> MuggledSAMManualSeedTargetConfig:
        if len(set(self.targets)) != len(self.targets):
            raise ValueError("manual-seed target config targets must be distinct")
        return self


class MuggledSAMMultiKeyframeCorrectionPolicy(VersionedModel):
    """Declared, conservative memory semantics for human correction keyframes."""

    manifest_kind: Literal["muggledsam_sam3_multi_keyframe_correction_policy"]
    policy_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    policy_version: Literal["1"]
    view_id: Literal["ego-hmc21179183"]
    manual_seed_target_config_fingerprint: ArtifactFingerprint
    targets: tuple[str, ...] = Field(min_length=1)
    maximum_later_correction_keyframes_per_target: int = Field(ge=0, le=3)
    correction_memory_semantics: Literal["replace_prompt_memory_and_reset_frame_memory"]

    @model_validator(mode="after")
    def require_distinct_policy_targets(self) -> MuggledSAMMultiKeyframeCorrectionPolicy:
        if len(set(self.targets)) != len(self.targets):
            raise ValueError("correction policy targets must be distinct")
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
    target_id: str = Field(min_length=1)
    object_id: str = Field(pattern=r"^sam3-\d{2,}$")
    multiplex_slot: int = Field(ge=0)
    frame: CalibrationFrameReference
    calibration_mask_fingerprint: ArtifactFingerprint


class MuggledSAMMultiKeyframeCorrectionSchedule(VersionedModel):
    """Integrity-bound, deterministic schedule of frame-zero seeds and later corrections."""

    manifest_kind: Literal["muggledsam_sam3_multi_keyframe_correction_schedule"]
    authority: Literal["proposed_non_authoritative"]
    schedule_version: Literal["1"]
    view_id: Literal["ego-hmc21179183"]
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
    correction_memory_semantics: Literal["replace_prompt_memory_and_reset_frame_memory"]
    scheduled_correction_frame_indices: tuple[int, ...] = ()
    ground_truth_accuracy_claim: Literal[False] = False


class MuggledSAMProposedTrackingPromptConfig(VersionedModel):
    """Non-authoritative configuration output from explicit human selections."""

    manifest_kind: Literal["muggledsam_sam3_proposed_tracking_prompt"]
    authority: Literal["proposed_non_authoritative"]
    calibration_manifest_uri: str = Field(min_length=1)
    calibration_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    view_id: Literal["ego-hmc21179183"]
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


class ManualSeedMultiplexMetadata(VersionedModel):
    """Auditable initialization details for a manual-mask SAM3 multiplex smoke."""

    proposal_fingerprint: ArtifactFingerprint
    calibration_manifest_fingerprint: ArtifactFingerprint
    seeds: tuple[ManualSeedCandidateProvenance, ...] = Field(min_length=1)
    initialization_api: Literal["encode_prompt_memory_from_mask"]
    excluded_candidate_ids: tuple[str, ...] = ()
    ground_truth_accuracy_claim: Literal[False] = False

    @model_validator(mode="after")
    def require_unique_targets_and_slots(self) -> ManualSeedMultiplexMetadata:
        targets = [seed.intended_target for seed in self.seeds]
        slots = [seed.initial_multiplex_slot for seed in self.seeds]
        if len(set(targets)) != len(targets):
            raise ValueError("manual multiplex seed targets must be unique")
        if len(set(slots)) != len(slots):
            raise ValueError("manual multiplex seed slots must be unique")
        return self


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

    @model_validator(mode="after")
    def require_exact_smoke_budget(self) -> SmokeRunMetadata:
        if self.requested_analysis_frame_range.start_frame != 0:
            raise ValueError("smoke range must start at proxy frame zero")
        if self.requested_analysis_frame_range.frame_count != 300:
            raise ValueError("smoke range must contain exactly 300 analysis frames")
        if self.requested_seconds != 10.0:
            raise ValueError("smoke duration must be exactly 10.0 seconds")
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

    @model_validator(mode="after")
    def require_exact_g3_static_budget(self) -> G3CandidateRunMetadata:
        if (
            self.requested_analysis_frame_range.start_frame != 0
            or self.requested_analysis_frame_range.frame_count != 5400
        ):
            raise ValueError("G3 candidate must cover exactly static proxy frames [0, 5400)")
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
