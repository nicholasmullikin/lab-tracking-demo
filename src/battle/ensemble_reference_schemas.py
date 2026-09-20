"""Typed policy and provenance contracts for the per-target ensemble review reference.

The policy is checked in and agent-authored; every threshold is a review assumption, not a
measured accuracy.  Provenance records make each frame x target mask's origin explicit so a
reviewer can see exactly where a fallback or a hidden label replaced the primary tracker.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .schemas import ArtifactFingerprint, EnsembleMaskProvenance, FrameRange, VersionedModel

TargetId = Literal["chassis", "interior", "rear_body", "cabin"]
RuleName = Literal[
    "primary_missing",
    "area_below_rolling_median_fraction",
    "other_target_iou_above_threshold",
    "discontinuous_with_last_accepted",
]
PROVENANCE_CODES: dict[str, int] = {
    "missing": 0,
    "sam3_corrected": 1,
    "dam4sam_fallback": 2,
    "hidden_agent_label": 3,
}


class EnsembleFallbackRules(VersionedModel):
    """When the primary mask is suspect enough to consider the fallback run's mask.

    A rule that is `None` is disabled.  Rules are evaluated on every frame for diagnostics but
    only trigger a substitution inside a target's explicit fallback intervals.
    """

    area_fraction_of_rolling_median_below: float | None = Field(default=0.5, gt=0, lt=1)
    other_target_iou_above: float | None = Field(default=0.3, gt=0, le=1)
    discontinuity_with_last_accepted: bool = Field(
        default=True,
        description=(
            "Fires when the primary mask fails the same continuity test a fallback mask must "
            "pass: IoU with the last sane accepted mask below the sanity minimum and a centroid "
            "jump beyond the per-frame allowance."
        ),
    )
    substitute_when_primary_missing: bool = True


class EnsembleSanityBounds(VersionedModel):
    """What a fallback mask must satisfy before it may replace the primary mask."""

    area_fraction_bounds: tuple[float, float] = (0.4, 2.0)
    min_iou_with_last_accepted: float = Field(default=0.5, ge=0, le=1)
    max_centroid_jump_pixels: float = Field(default=30.0, gt=0)
    centroid_jump_pixels_per_frame: float = Field(
        default=5.0,
        ge=0,
        description="Extra centroid allowance per frame since the last sane accepted mask.",
    )
    max_continuity_allowance_pixels: float = Field(
        default=90.0,
        gt=0,
        description="Cap on the grown allowance so a long unsane stretch cannot accept anything.",
    )

    @model_validator(mode="after")
    def require_ordered_bounds(self) -> EnsembleSanityBounds:
        low, high = self.area_fraction_bounds
        if not 0 < low < 1 <= high:
            raise ValueError("area fraction bounds must satisfy 0 < low < 1 <= high")
        if self.max_continuity_allowance_pixels < self.max_centroid_jump_pixels:
            raise ValueError("the continuity cap must not be below the base centroid allowance")
        return self

    def continuity_allowance(self, frames_since_last_accepted: int) -> float:
        grown = self.max_centroid_jump_pixels + self.centroid_jump_pixels_per_frame * max(
            0, frames_since_last_accepted
        )
        return min(grown, self.max_continuity_allowance_pixels)


IntervalProvenance = Literal[
    # v1: an agent looked at frames and drew the bounds.
    "agent_authored_visual_review",
    # v2 fallback intervals: C10379 contradicted by the majority of the other views in the
    # label-free cross-view consensus; the bounds never saw the human anchors.
    "multiview_consensus_contradiction",
    # v2 distractor interval: bounds read off the primary run's own mask centroid/area
    # trajectory; the failure case name and the rationale come from the human's anchor record.
    "primary_mask_trajectory",
]
FailureCase = Literal["distractor_confusion"]


class EnsembleLabeledInterval(FrameRange):
    """A half-open frame interval with a rationale and the provenance of its bounds."""

    rationale: str = Field(min_length=1)
    provenance: IntervalProvenance = "agent_authored_visual_review"
    human_confirmation_pending: Literal[True] = True
    failure_case: FailureCase | None = Field(
        default=None,
        description=(
            "Named failure case from the human review anchor record that this interval "
            "documents (e.g. the yellow screwdriver next to the rear body at frame 1700)."
        ),
    )


class EnsembleTargetPolicy(VersionedModel):
    target_id: TargetId
    fallback_intervals: tuple[EnsembleLabeledInterval, ...] = ()
    rules: EnsembleFallbackRules = Field(default_factory=EnsembleFallbackRules)
    sanity: EnsembleSanityBounds = Field(default_factory=EnsembleSanityBounds)
    hidden_intervals: tuple[EnsembleLabeledInterval, ...] = ()
    not_contact_eligible_intervals: tuple[EnsembleLabeledInterval, ...] = ()

    @model_validator(mode="after")
    def require_disjoint_hidden_and_fallback(self) -> EnsembleTargetPolicy:
        for hidden in self.hidden_intervals:
            for fallback in self.fallback_intervals:
                if (
                    hidden.start_frame < fallback.end_frame_exclusive
                    and fallback.start_frame < hidden.end_frame_exclusive
                ):
                    raise ValueError(
                        f"{self.target_id}: hidden and fallback intervals must not overlap"
                    )
        return self

    def in_intervals(self, frame: int, intervals: tuple[EnsembleLabeledInterval, ...]) -> bool:
        return any(item.start_frame <= frame < item.end_frame_exclusive for item in intervals)


class EnsembleSourceRunPolicy(VersionedModel):
    label: Literal["sam3_corrected", "dam4sam_fallback"]
    run_directory: str = Field(min_length=1)
    arm: str | None = Field(
        default=None, description="Scoreboard arm name of the run (e.g. `pm-append`)."
    )
    anchor_iou_all: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="The arm's `IoU all` against the human review anchors when it was chosen.",
    )


class EnsembleArmSelection(VersionedModel):
    """How the primary and fallback runs were chosen: by the human review anchors."""

    method: Literal["human_review_anchor_iou"]
    scoreboard_uri: str = Field(min_length=1)
    anchors_uri: str = Field(min_length=1)
    anchor_frame_count: int = Field(gt=0)
    anchor_view: str = Field(min_length=1)
    tie_rule: str = Field(min_length=1)
    rationale: str = Field(min_length=1)


class EnsembleIntervalSelection(VersionedModel):
    """How the fallback intervals were chosen: label-free, from cross-view consensus only."""

    method: Literal["multiview_consensus_contradiction"]
    consensus_root_uri: str = Field(min_length=1)
    consensus_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    consensus_reference_run: str = Field(min_length=1)
    merge_gap_below_frames: int = Field(ge=0)
    min_interval_frames: int = Field(ge=1)
    raw_contradiction_intervals: dict[str, tuple[tuple[int, int], ...]]
    rationale: str = Field(min_length=1)


class EnsembleSelectionProvenance(VersionedModel):
    """Which evidence chose what, and the bias that leaves in the anchor score."""

    arm_selection: EnsembleArmSelection
    interval_selection: EnsembleIntervalSelection
    bias_statement: str = Field(min_length=1)


class EnsembleReferencePolicy(VersionedModel):
    """Checked-in ensemble policy; every value is an agent-authored review assumption."""

    policy_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    provenance_tag: Literal["agent_authored_assumption"]
    frame_count: Literal[1800]
    primary: EnsembleSourceRunPolicy
    fallback: EnsembleSourceRunPolicy
    rolling_median_window_frames: int = Field(default=300, ge=30)
    rolling_median_min_samples: int = Field(default=30, ge=1)
    contact_eligible_through_frame: int = Field(default=1200, gt=0, le=1800)
    late_ineligibility_rationale: str = Field(min_length=1)
    targets: tuple[EnsembleTargetPolicy, ...] = Field(min_length=4, max_length=4)
    no_blend: Literal[True] = True
    claim_boundaries: tuple[str, ...] = Field(min_length=1)
    selection_provenance: EnsembleSelectionProvenance | None = Field(
        default=None,
        description=(
            "v2+: records that the arms were chosen by the human anchors and the fallback "
            "intervals label-free. Absent in v1, whose intervals were drawn by visual review."
        ),
    )

    @model_validator(mode="after")
    def require_label_free_intervals_when_selection_is_recorded(self) -> EnsembleReferencePolicy:
        if self.selection_provenance is None:
            return self
        for target in self.targets:
            for interval in target.fallback_intervals:
                if interval.provenance != "multiview_consensus_contradiction":
                    raise ValueError(
                        f"{target.target_id}: with selection_provenance recorded every fallback "
                        "interval must come from the cross-view consensus, not from review"
                    )
        return self

    @model_validator(mode="after")
    def require_ordered_targets_and_distinct_runs(self) -> EnsembleReferencePolicy:
        order = tuple(item.target_id for item in self.targets)
        if order != ("chassis", "interior", "rear_body", "cabin"):
            raise ValueError("ensemble policy must list the four targets in contract order")
        if self.primary.label != "sam3_corrected" or self.fallback.label != "dam4sam_fallback":
            raise ValueError("primary must be sam3_corrected and fallback dam4sam_fallback")
        if self.primary.run_directory == self.fallback.run_directory:
            raise ValueError("primary and fallback runs must differ")
        for target in self.targets:
            for interval in (
                *target.fallback_intervals,
                *target.hidden_intervals,
                *target.not_contact_eligible_intervals,
            ):
                if interval.end_frame_exclusive > self.frame_count:
                    raise ValueError(f"{target.target_id}: interval exceeds the frame count")
        return self

    def target(self, target_id: str) -> EnsembleTargetPolicy:
        return next(item for item in self.targets if item.target_id == target_id)


class EnsembleFrameProvenance(VersionedModel):
    """Origin and sanity of one frame x target mask in the ensemble reference."""

    analysis_frame_index: int = Field(ge=0, lt=1800)
    target_id: TargetId
    provenance: EnsembleMaskProvenance
    mask_uri: str | None = None
    area_pixels: int = Field(ge=0)
    rules_fired: tuple[RuleName, ...] = ()
    inside_fallback_interval: bool
    sane: bool
    contact_eligible: bool
    primary_area_pixels: int | None = Field(default=None, ge=0)
    fallback_area_pixels: int | None = Field(default=None, ge=0)
    rolling_median_area_pixels: float | None = Field(default=None, ge=0)
    iou_with_last_accepted: float | None = Field(default=None, ge=0, le=1)
    centroid_jump_pixels: float | None = Field(default=None, ge=0)
    rationale: str | None = None

    @model_validator(mode="after")
    def require_consistent_states(self) -> EnsembleFrameProvenance:
        if self.provenance in ("hidden_agent_label", "missing"):
            if self.area_pixels != 0 or self.sane or self.contact_eligible:
                raise ValueError("hidden or missing masks are empty, unsane and ineligible")
        elif self.area_pixels == 0:
            raise ValueError("a sourced mask must contain pixels")
        if self.contact_eligible and not self.sane:
            raise ValueError("contact eligibility requires a sane mask")
        if self.provenance == "dam4sam_fallback" and not self.inside_fallback_interval:
            raise ValueError("fallback masks are only allowed inside fallback intervals")
        return self


class EnsembleSubstitutionRecord(VersionedModel):
    """One accepted fallback substitution or one declined attempt, with its rationale."""

    analysis_frame_index: int = Field(ge=0, lt=1800)
    target_id: TargetId
    outcome: Literal["substituted", "declined"]
    rules_fired: tuple[RuleName, ...] = Field(min_length=1)
    primary_area_pixels: int | None = Field(default=None, ge=0)
    fallback_area_pixels: int | None = Field(default=None, ge=0)
    rolling_median_area_pixels: float | None = Field(default=None, ge=0)
    iou_with_last_accepted: float | None = Field(default=None, ge=0, le=1)
    centroid_jump_pixels: float | None = Field(default=None, ge=0)
    rationale: str = Field(min_length=1)


class EnsembleTargetSummary(VersionedModel):
    target_id: TargetId
    provenance_counts: dict[str, int]
    substituted_intervals: tuple[tuple[int, int], ...] = ()
    declined_intervals: tuple[tuple[int, int], ...] = ()
    hidden_intervals: tuple[tuple[int, int], ...] = ()
    missing_intervals: tuple[tuple[int, int], ...] = ()
    contact_eligible_intervals: tuple[tuple[int, int], ...] = ()
    rule_firings_outside_fallback_intervals: int = Field(ge=0)


class EnsembleProvenanceSidecar(VersionedModel):
    manifest_kind: Literal["ensemble_reference_provenance_v1"]
    policy: ArtifactFingerprint
    primary_manifest: ArtifactFingerprint
    fallback_manifest: ArtifactFingerprint
    frame_count: Literal[1800]
    claim_boundaries: tuple[str, ...] = Field(min_length=1)
    no_blend: Literal[True] = True
    summaries: tuple[EnsembleTargetSummary, ...] = Field(min_length=4, max_length=4)
    substitutions: tuple[EnsembleSubstitutionRecord, ...] = ()
    declined: tuple[EnsembleSubstitutionRecord, ...] = ()
    frames: tuple[EnsembleFrameProvenance, ...]

    @model_validator(mode="after")
    def require_complete_frames(self) -> EnsembleProvenanceSidecar:
        if len(self.frames) != self.frame_count * 4:
            raise ValueError("the sidecar must hold one record per frame and target")
        return self
