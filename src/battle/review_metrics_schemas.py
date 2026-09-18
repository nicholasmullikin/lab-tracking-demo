"""Typed, NaN-free records for the label-free first-minute review metrics.

Every value here is a geometry, appearance, or provenance proxy computed from already
retained review artifacts.  None of them is an accuracy measure, a ground-truth label, or a
claim about what the worker touched; missing evidence is always explicit rather than
encoded as NaN, zero, or a carried-forward value.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .schemas import ArtifactFingerprint, TimeInterval, VersionedModel

PartId = Literal["chassis", "interior", "rear_body", "cabin"]
MaskState = Literal["observed", "missing_mask"]
KineoSource = Literal["detected_native", "boxmot_fallback", "interpolated", "held", "missing"]
EpisodeType = Literal[
    "segmentation_identity_swap",
    "segmentation_label_crossing",
    "appearance_leakage",
    "mask_growth_hand_capture_suspect",
    "mask_growth_unexplained",
    "mask_area_anomaly_vs_median",
    "hand_visible_but_undetected",
    "hand_reentry_jump",
    "contact_unexpected_for_substep",
    "contact_flicker",
]


class ExpectedTouchedParts(VersionedModel):
    """Agent-authored assumption about which parts a substep plausibly touches."""

    substep_id: str = Field(pattern=r"^S\d{2}$")
    expected_parts: tuple[PartId, ...] = ()
    note: str = Field(min_length=1)


class ReviewMetricsConfig(VersionedModel):
    """Checked-in thresholds and assumptions; nothing here is tuned to a single episode."""

    manifest_kind: Literal["review_metrics_config"]
    provenance_tag: Literal["agent_authored_assumption"]
    claim_boundaries: tuple[str, ...] = Field(min_length=1)
    episode_max_frame_gap: int = Field(default=6, ge=0)
    swap_score_threshold: float = Field(default=0.35, gt=0, le=1)
    swap_large_area_fraction: float = Field(default=0.5, gt=0, le=1)
    crossing_window_frames: int = Field(default=10, ge=1)
    crossing_min_separation_pixels: float = Field(default=10.0, gt=0)
    crossing_min_area_fraction: float = Field(default=0.3, gt=0, le=1)
    appearance_distance_threshold: float = Field(default=18.0, gt=0)
    appearance_robust_z_threshold: float = Field(default=3.0, gt=0)
    yellow_fraction_threshold_dark_parts: float = Field(default=0.25, gt=0, le=1)
    hand_overlap_fraction_threshold: float = Field(default=0.35, gt=0, le=1)
    growth_area_ratio_threshold: float = Field(default=1.3, gt=1)
    growth_window_frames: int = Field(default=15, ge=1)
    growth_window_ratio_threshold: float = Field(default=1.75, gt=1)
    hand_capture_overlap_fraction: float = Field(default=0.05, ge=0, le=1)
    area_enlarged_ratio_threshold: float = Field(default=1.75, gt=1)
    area_collapsed_ratio_threshold: float = Field(default=0.35, gt=0, lt=1)
    area_anomaly_min_frames: int = Field(default=5, ge=1)
    skin_calibration_min_confidence: float = Field(default=0.8, ge=0, le=1)
    skin_window_scale: float = Field(default=1.5, gt=0)
    skin_fraction_min: float = Field(default=0.15, ge=0, le=1)
    skin_fraction_reference_ratio: float = Field(default=0.5, ge=0, le=1)
    skin_background_ratio: float = Field(default=2.0, ge=1)
    fingertips_in_frame_min: int = Field(default=3, ge=0, le=5)
    reentry_jump_normalized_threshold: float = Field(default=1.0, gt=0)
    contact_flicker_max_frames: int = Field(default=5, ge=1)
    top_episode_count: int = Field(default=12, ge=1)
    expected_touched_parts: tuple[ExpectedTouchedParts, ...] = Field(min_length=1)


class StabilizedHandProvenanceRow(VersionedModel):
    """One row of the stabilized WiLoR layer's `hand_provenance.json`."""

    analysis_frame_index: int = Field(ge=0, lt=1800)
    output_hand_id: str
    source: Literal["wilor", "mediapipe_fallback", "missing"]
    state: Literal["raw", "smoothed", "low_confidence_continuation", "fallback", "missing"]
    reason: str = Field(min_length=1)
    raw_hand_id: str | None = None


class PartFrameStat(VersionedModel):
    """Single-part per-frame geometry; centroids are in source pixels."""

    analysis_frame_index: int = Field(ge=0, lt=1800)
    part_id: PartId
    state: MaskState
    area_pixels: int | None = Field(default=None, ge=0)
    centroid_x: float | None = Field(default=None, ge=0)
    centroid_y: float | None = Field(default=None, ge=0)
    area_ratio_vs_median: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def require_geometry_only_when_observed(self) -> PartFrameStat:
        values = (self.area_pixels, self.centroid_x, self.centroid_y, self.area_ratio_vs_median)
        if self.state == "observed" and any(value is None for value in values):
            raise ValueError("observed part stats require area, centroid, and median ratio")
        if self.state != "observed" and any(value is not None for value in values):
            raise ValueError("missing masks cannot carry stale geometry")
        return self


class PartPairFrameMetric(VersionedModel):
    """Identity-swap evidence for one unordered part pair on one frame."""

    analysis_frame_index: int = Field(ge=0, lt=1800)
    part_a: PartId
    part_b: PartId
    state: MaskState
    iou: float | None = Field(default=None, ge=0, le=1)
    centroid_distance_pixels: float | None = Field(default=None, ge=0)
    exchange_score: float | None = Field(default=None, ge=0, le=1)
    convergence_score: float | None = Field(default=None, ge=0, le=1)
    absorption_score: float | None = Field(default=None, ge=0, le=1)
    absorbing_part: PartId | None = None
    swap_score: float | None = Field(default=None, ge=0, le=1)
    label_crossing: bool | None = None

    @model_validator(mode="after")
    def require_scores_only_when_observed(self) -> PartPairFrameMetric:
        values = (
            self.iou,
            self.centroid_distance_pixels,
            self.convergence_score,
            self.absorption_score,
            self.swap_score,
            self.label_crossing,
        )
        if self.state == "observed" and any(value is None for value in values):
            raise ValueError("observed pair metrics require every same-frame score")
        if self.state != "observed" and any(value is not None for value in values):
            raise ValueError("pairs with a missing mask cannot carry stale scores")
        return self


class ColorBand(VersionedModel):
    """A derived color band in OpenCV 8-bit ranges; recorded so it can be audited."""

    space: Literal["hsv", "ycrcb"]
    channel_low: tuple[int, int, int]
    channel_high: tuple[int, int, int]
    hue_wraps: bool = False
    derived_from: str = Field(min_length=1)
    sample_pixels: int = Field(ge=0)
    background_fraction: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Fraction of sampled non-target pixels that also fall inside the band.",
    )


class PartAppearanceFrameMetric(VersionedModel):
    """Per-mask color statistics relative to the same part on frame 0."""

    analysis_frame_index: int = Field(ge=0, lt=1800)
    part_id: PartId
    state: MaskState
    median_h: float | None = Field(default=None, ge=0, le=180)
    median_s: float | None = Field(default=None, ge=0, le=255)
    median_v: float | None = Field(default=None, ge=0, le=255)
    mean_l: float | None = Field(default=None, ge=0, le=255)
    mean_a: float | None = Field(default=None, ge=0, le=255)
    mean_b: float | None = Field(default=None, ge=0, le=255)
    lab_distance_from_frame0: float | None = Field(default=None, ge=0)
    lab_distance_robust_z: float | None = Field(
        default=None,
        description="(distance - part median) / (1.4826 * MAD) over the whole minute.",
    )
    yellow_fraction: float | None = Field(default=None, ge=0, le=1)
    hand_overlap_fraction: float | None = Field(default=None, ge=0, le=1)
    leakage_suspect: bool | None = None

    @model_validator(mode="after")
    def require_stats_only_when_observed(self) -> PartAppearanceFrameMetric:
        values = (
            self.median_h,
            self.median_s,
            self.median_v,
            self.mean_l,
            self.mean_a,
            self.mean_b,
            self.lab_distance_from_frame0,
            self.lab_distance_robust_z,
            self.yellow_fraction,
            self.hand_overlap_fraction,
            self.leakage_suspect,
        )
        if self.state == "observed" and any(value is None for value in values):
            raise ValueError("observed appearance metrics require every statistic")
        if self.state != "observed" and any(value is not None for value in values):
            raise ValueError("missing masks cannot carry stale appearance statistics")
        return self


GrowthClass = Literal["hand_capture_suspect", "unexplained"]


class PartGrowthFrameMetric(VersionedModel):
    """Area change against the previous frame with hand-overlap context."""

    analysis_frame_index: int = Field(ge=0, lt=1800)
    part_id: PartId
    state: Literal["observed", "missing_mask", "no_previous_mask"]
    area_ratio_vs_previous: float | None = Field(default=None, ge=0)
    area_ratio_vs_window: float | None = Field(
        default=None,
        ge=0,
        description="Area ratio against the mask `growth_window_frames` earlier, when present.",
    )
    centroid_velocity_pixels: float | None = Field(default=None, ge=0)
    hand_box_overlaps_mask: bool | None = None
    hand_overlap_fraction: float | None = Field(default=None, ge=0, le=1)
    growth_event: bool | None = None
    growth_class: GrowthClass | None = None

    @model_validator(mode="after")
    def require_consistency(self) -> PartGrowthFrameMetric:
        values = (
            self.area_ratio_vs_previous,
            self.centroid_velocity_pixels,
            self.hand_box_overlaps_mask,
            self.hand_overlap_fraction,
            self.growth_event,
        )
        if self.state == "observed":
            if any(value is None for value in values):
                raise ValueError("observed growth metrics require every derived value")
            if bool(self.growth_event) != (self.growth_class is not None):
                raise ValueError("growth class is present exactly for growth events")
        elif (
            any(value is not None for value in values)
            or self.growth_class is not None
            or self.area_ratio_vs_window is not None
        ):
            raise ValueError("frames without two masks cannot carry stale growth values")
        return self


class HandGapFrameProxy(VersionedModel):
    """Skin-color proxy near the last known hand while the stabilized layer is missing.

    This is explicitly a proxy: skin-like pixels near the last pose are neither a hand
    detection nor evidence that the detector was wrong.
    """

    analysis_frame_index: int = Field(ge=0, lt=1800)
    last_known_frame: int = Field(ge=0, lt=1800)
    last_known_hand_id: str = Field(min_length=1)
    window_center_x: int = Field(ge=0)
    window_center_y: int = Field(ge=0)
    window_size_pixels: int = Field(ge=1)
    skin_fraction_in_window: float = Field(ge=0, le=1)
    reference_skin_fraction: float = Field(ge=0, le=1)
    background_skin_fraction: float = Field(
        ge=0, le=1, description="Skin-band fraction over the whole downsampled frame."
    )
    fingertips_in_frame_count: int = Field(ge=0, le=5)
    visible_but_undetected_suspect: bool


class HandGapRecord(VersionedModel):
    """One frame-level stabilized-hand outage and the re-entry jump that closes it."""

    start_frame: int = Field(ge=0, lt=1800)
    end_frame_exclusive: int = Field(gt=0, le=1800)
    length_frames: int = Field(ge=1)
    last_known_frame: int | None = Field(default=None, ge=0, lt=1800)
    reentry_frame: int | None = Field(default=None, ge=0, lt=1800)
    reentry_jump_pixels: float | None = Field(default=None, ge=0)
    reentry_jump_normalized: float | None = Field(default=None, ge=0)
    suspect_frames: int = Field(ge=0)

    @model_validator(mode="after")
    def require_ordered(self) -> HandGapRecord:
        if self.end_frame_exclusive <= self.start_frame:
            raise ValueError("hand gap must be a positive interval")
        if self.length_frames != self.end_frame_exclusive - self.start_frame:
            raise ValueError("hand gap length must match its interval")
        if (self.reentry_frame is None) != (self.reentry_jump_pixels is None):
            raise ValueError("re-entry jump requires a re-entry frame")
        return self


class HandJitterFrameMetric(VersionedModel):
    """Nearest-wrist displacement normalized by hand scale; includes real motion."""

    analysis_frame_index: int = Field(ge=0, lt=1800)
    hand_id: str = Field(min_length=1)
    provenance_state: str = Field(min_length=1)
    hand_scale_pixels: float = Field(gt=0)
    wrist_displacement_pixels: float | None = Field(default=None, ge=0)
    jitter_normalized: float | None = Field(default=None, ge=0)


class PhaseJitterSummary(VersionedModel):
    phase_kind: Literal["agent_substep", "coarse_gt"]
    phase_id: str = Field(min_length=1)
    start_frame: int = Field(ge=0, lt=1800)
    end_frame_exclusive: int = Field(gt=0, le=1800)
    sample_count: int = Field(ge=0)
    median_jitter_normalized: float | None = Field(default=None, ge=0)
    p90_jitter_normalized: float | None = Field(default=None, ge=0)
    mean_jitter_normalized: float | None = Field(default=None, ge=0)
    missing_hand_frames: int = Field(ge=0)


class ContactIntervalRecord(VersionedModel):
    """One debounced geometry-only contact-candidate interval, never a touch label."""

    hand_source_id: str = Field(min_length=1)
    part_id: PartId
    start_frame: int = Field(ge=0, lt=1800)
    end_frame_exclusive: int = Field(gt=0, le=1800)
    duration_frames: int = Field(ge=1)
    closed: bool
    substep_id: str | None = None
    expected_for_substep: bool | None = None
    coarse_gt_action: str | None = None
    flicker: bool


class ContactSummary(VersionedModel):
    interval_count: int = Field(ge=0)
    sub_5_frame_count: int = Field(ge=0)
    duration_min: int | None = Field(default=None, ge=1)
    duration_median: float | None = Field(default=None, ge=0)
    duration_p90: float | None = Field(default=None, ge=0)
    duration_max: int | None = Field(default=None, ge=1)
    duration_histogram: dict[str, int]
    expected_count: int = Field(ge=0)
    unexpected_count: int = Field(ge=0)
    unknown_substep_count: int = Field(ge=0)
    per_substep_unexpected: dict[str, int]


class KineoFrameMetric(VersionedModel):
    analysis_frame_index: int = Field(ge=0, lt=1800)
    box_source: KineoSource
    body_present: bool
    mean_joint_confidence: float | None = Field(default=None, ge=0, le=1)
    joint_jitter_pixels: float | None = Field(default=None, ge=0)
    joint_jitter_normalized: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def require_confidence_when_present(self) -> KineoFrameMetric:
        if self.body_present and self.mean_joint_confidence is None:
            raise ValueError("present bodies require a mean joint confidence")
        if not self.body_present and (
            self.mean_joint_confidence is not None or self.joint_jitter_pixels is not None
        ):
            raise ValueError("absent bodies cannot carry stale confidence or jitter")
        return self


class KineoProvenanceSummary(VersionedModel):
    box_source: KineoSource
    frame_count: int = Field(ge=0)
    body_frames: int = Field(ge=0)
    mean_joint_confidence: float | None = Field(default=None, ge=0, le=1)
    jitter_sample_count: int = Field(ge=0)
    median_jitter_pixels: float | None = Field(default=None, ge=0)
    p90_jitter_pixels: float | None = Field(default=None, ge=0)
    median_jitter_normalized: float | None = Field(default=None, ge=0)


class ReviewTriggerEpisode(VersionedModel):
    """A ranked review prompt with an auditable frame range; never a correctness verdict."""

    rank: int = Field(ge=1)
    episode_type: EpisodeType
    start_frame: int = Field(ge=0, lt=1800)
    end_frame: int = Field(ge=0, lt=1800)
    peak_frame: int = Field(ge=0, lt=1800)
    score: float = Field(ge=0)
    subjects: tuple[str, ...] = Field(min_length=1)
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_ordered(self) -> ReviewTriggerEpisode:
        if not self.start_frame <= self.peak_frame <= self.end_frame:
            raise ValueError("episode peak must fall inside its frame range")
        return self


class ReviewMetricsManifest(VersionedModel):
    """Everything needed to audit one metrics run without re-deriving it."""

    manifest_kind: Literal["review_metrics_first_minute_v1"]
    frame_count: Literal[1800]
    analysis_fps: Literal[30]
    source_interval: TimeInterval
    v4_index: ArtifactFingerprint
    config: ArtifactFingerprint
    # Set when the part masks were read from a run other than the v4 index's reference (a
    # tracker-policy arm under comparison); the hand, Kineo and contact inputs stay the index's.
    reference_run_override: ArtifactFingerprint | None = None
    input_artifacts: tuple[ArtifactFingerprint, ...] = Field(min_length=1)
    claim_boundaries: tuple[str, ...] = Field(min_length=1)
    part_area_medians: dict[str, float]
    yellow_band: ColorBand
    skin_band: ColorBand
    part_stats: tuple[PartFrameStat, ...]
    pair_metrics: tuple[PartPairFrameMetric, ...]
    appearance_metrics: tuple[PartAppearanceFrameMetric, ...]
    growth_metrics: tuple[PartGrowthFrameMetric, ...]
    hand_gap_proxies: tuple[HandGapFrameProxy, ...]
    hand_gaps: tuple[HandGapRecord, ...]
    hand_jitter: tuple[HandJitterFrameMetric, ...]
    jitter_by_phase: tuple[PhaseJitterSummary, ...]
    contact_intervals: tuple[ContactIntervalRecord, ...]
    contact_summary: ContactSummary
    kineo_metrics: tuple[KineoFrameMetric, ...]
    kineo_by_provenance: tuple[KineoProvenanceSummary, ...]
    episodes: tuple[ReviewTriggerEpisode, ...]
    output_rrd: ArtifactFingerprint | None = None
