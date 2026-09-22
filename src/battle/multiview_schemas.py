"""Typed contracts for the multi-view pass: geometric seed transfer, part consensus, hulls.

Every record here is cross-view geometry on dataset context (shipped extrinsics, per-frame
ego poses) and fitted estimates (intrinsics, table plane).  Seeds on views other than the
human-reviewed C10379 are agent-authored (`selected_by="agent"`, provenance
`geometric_seed_transfer`); consensus and hull numbers measure cross-view disagreement,
never accuracy; CC BY-NC 4.0 attribution applies to the dataset assets.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .multiview_geometry import TablePlane
from .schemas import ArtifactFingerprint, PixelBox, PixelPoint, VersionedModel

GEOMETRIC_SEED_PROVENANCE = "geometric_seed_transfer"
# Sep 20 recording-2 seeding: DINOv2 exemplars of recording 1's human masks rank SAM3 grid
# candidates, accepted by >= 3-view triangulation; no geometry from other views' masks.
EXEMPLAR_SEED_PROVENANCE = "exemplar_multiview_consistency"
# Sep 21: a seed-search proposal the human accepted in the decisions file (the mask is the
# agent decoder's, the choice is the human's); an interior seed decoded from the sphere two
# human masks (C10379 frame 0, C10119 frame 41) triangulate, accepted by consistency and held
# `until the human confirms`; and the manifest-level value when parts carry different ones.
HUMAN_ACCEPTED_SEED_PROVENANCE = "agent_proposed_human_accepted"
TWO_HUMAN_VIEWS_SEED_PROVENANCE = "geometric_from_two_human_views"
MIXED_SEED_PROVENANCE = "mixed_per_part"
SeedProvenance = Literal[
    "geometric_seed_transfer",
    "exemplar_multiview_consistency",
    "agent_proposed_human_accepted",
    "geometric_from_two_human_views",
    "mixed_per_part",
]
MULTIVIEW_CLAIM_BOUNDARIES: tuple[str, ...] = (
    "Seeds on views other than C10379 were chosen by an agent from geometry (table-plane "
    "transfer of the human frame-0 masks) and a back-projection IoU rule; no human reviewed them.",
    "Dataset poses and extrinsics are external context; intrinsics and the table plane are "
    "estimates; nothing here is ground truth for any method.",
    "Cross-view consensus, agreement scores and hulls measure disagreement between views of "
    "the same tracker, not accuracy.",
    "Assembly101 is CC BY-NC 4.0; attribution applies to every derived artifact.",
)

SeedStatus = Literal["accepted", "blocked"]


class SeedPrompt(VersionedModel):
    """One box prompt in a view's proxy pixels, derived from transferred 3D points."""

    prompt_id: str = Field(pattern=r"^t\d{6}-b\d{2,}$")
    target: str = Field(min_length=1)
    variant: str = Field(min_length=1)
    pixel_box: PixelBox
    background_points: tuple[PixelPoint, ...] = ()
    foreground_points: tuple[PixelPoint, ...] = ()


class SeedCandidate(VersionedModel):
    """One decoded mask candidate and its geometric acceptance evidence."""

    prompt_id: str = Field(pattern=r"^t\d{6}-b\d{2,}$")
    candidate_index: int = Field(ge=0)
    mask: ArtifactFingerprint
    decoder_iou_estimate: float
    mask_area_px: int = Field(ge=0)
    backprojection_iou: float | None = Field(default=None, ge=0, le=1)
    centroid_ray_distance_mm: float | None = Field(default=None, ge=0)
    area_ratio_vs_expected: float | None = Field(default=None, ge=0)
    joints_inside_fraction: float | None = Field(default=None, ge=0, le=1)
    sanity_pass: bool
    acceptance_basis: (
        Literal[
            "plane_warp_iou",
            "centroid_ray",
            "dataset_joints",
            # Sep 20 seed search: the candidate's centroid triangulates with >= 3 views'
            # masks at the same pose frame within the reprojection filter, area in band.
            "multiview_consistency",
            # The Sep 18 agent seed carried over unchanged because the searched strategy
            # for that part did not pass the C10379 held-out gate.
            "carried_over_sep18_seed",
            # Sep 20 recording 2: exemplar-ranked candidate whose centroid triangulates with
            # >= 3 static views' top candidates at the seed frame, radii in band.
            "exemplar_multiview_consistency",
            # Sep 21: the human accepted this candidate in the seed-proposal decisions file.
            "human_accepted_proposal",
        ]
        | None
    ) = None
    sanity_notes: tuple[str, ...] = ()
    # Multi-view consistency evidence (Sep 20 transfer); None on the Sep 18 records.
    consistency_views_used: tuple[str, ...] | None = None
    consistency_reprojection_px: float | None = Field(default=None, ge=0)
    iou_vs_sep18_seed: float | None = Field(default=None, ge=0, le=1)


class SeedTransferPart(VersionedModel):
    """Transfer record for one target in one view."""

    target: str = Field(min_length=1)
    kind: Literal["part", "hand"]
    status: SeedStatus
    blocked_reason: str | None = None
    selected_by: Literal["agent"] = "agent"
    provenance: SeedProvenance = GEOMETRIC_SEED_PROVENANCE
    source_points_world_mm: tuple[tuple[float, float, float], ...] = ()
    triangulation_reprojection_px: dict[str, float] = Field(default_factory=dict)
    height_above_table_mm: float | None = None
    radius_mm: float | None = Field(default=None, ge=0)
    projected_centroid_proxy_px: tuple[float, float] | None = None
    depth_mm: float | None = None
    expected_area_px: float | None = Field(default=None, ge=0)
    prompts: tuple[SeedPrompt, ...] = ()
    candidates: tuple[SeedCandidate, ...] = ()
    accepted: SeedCandidate | None = None

    @model_validator(mode="after")
    def require_consistent_status(self) -> SeedTransferPart:
        if self.status == "accepted" and self.accepted is None:
            raise ValueError("an accepted seed must name its accepted candidate")
        if self.status == "blocked" and (self.accepted is not None or not self.blocked_reason):
            raise ValueError("a blocked seed carries a reason and no accepted candidate")
        if self.accepted is not None and not self.accepted.sanity_pass:
            raise ValueError("an accepted candidate must pass the sanity rules")
        return self


class SeedAcceptanceRules(VersionedModel):
    min_area_ratio: float = Field(gt=0)
    max_area_ratio: float = Field(gt=0)
    min_backprojection_iou: float = Field(ge=0, le=1)
    max_centroid_ray_distance_radii: float = Field(gt=0)
    min_hand_joints_inside_fraction: float = Field(ge=0, le=1)
    min_parts_to_run: int = Field(ge=1)
    description: str = Field(min_length=1)


class MultiviewSeedTransferManifest(VersionedModel):
    """Agent-authored frame-0 seeds for one view, with every rejected alternative kept."""

    manifest_kind: Literal["multiview_geometric_seed_transfer"]
    view: str = Field(min_length=1)
    view_id: str = Field(min_length=1)
    analysis_frame_index: int = Field(ge=0)
    pose_frame_index: int = Field(ge=0)
    is_ego: bool
    reference_view: str = Field(min_length=1)
    reference_run_manifest: ArtifactFingerprint
    reference_seed_masks: tuple[ArtifactFingerprint, ...] = Field(min_length=1)
    secondary_reference_view: str = Field(min_length=1)
    secondary_reference_pose_frame: int = Field(ge=0)
    secondary_reference_run_manifest: ArtifactFingerprint
    secondary_reference_seed_masks: tuple[ArtifactFingerprint, ...] = Field(min_length=1)
    transfer_method: str = Field(min_length=1)
    clip_config: ArtifactFingerprint
    proxy: ArtifactFingerprint
    proxy_dimensions: tuple[int, int]
    proxy_to_raw_scale: float = Field(gt=0)
    table_plane: TablePlane
    rules: SeedAcceptanceRules
    parts: tuple[SeedTransferPart, ...] = Field(min_length=1)
    hands: tuple[SeedTransferPart, ...] = ()
    decode_state: Literal["planned", "decoded"]
    run_decision: Literal["run", "skip", "pending"]
    run_decision_reason: str = Field(min_length=1)
    selected_by: Literal["agent"] = "agent"
    provenance: SeedProvenance = GEOMETRIC_SEED_PROVENANCE
    claim_boundaries: tuple[str, ...] = Field(min_length=1)

    @property
    def accepted_parts(self) -> tuple[SeedTransferPart, ...]:
        return tuple(part for part in self.parts if part.status == "accepted")


# -- consensus ------------------------------------------------------------------------------


class ConsensusViewSource(VersionedModel):
    view: str = Field(min_length=1)
    view_id: str = Field(min_length=1)
    run_directory_uri: str = Field(min_length=1)
    manifest: ArtifactFingerprint
    observations: ArtifactFingerprint
    is_ego: bool
    seed_provenance: Literal[
        "human_reviewed",
        "geometric_seed_transfer",
        "exemplar_multiview_consistency",
        "mixed_per_part",
    ]
    targets: tuple[str, ...] = Field(min_length=1)
    frame_count: int = Field(ge=1)
    proxy_to_raw_scale: float = Field(gt=0)


class DisagreementEpisode(VersionedModel):
    """A view whose mask sits away from the consensus for consecutive frames."""

    view: str = Field(min_length=1)
    target: str = Field(min_length=1)
    start_frame: int = Field(ge=0)
    end_frame_exclusive: int = Field(gt=0)
    max_reprojection_error_px: float = Field(ge=0)
    mean_reprojection_error_px: float = Field(ge=0)
    contradicts_majority: bool
    kind: Literal["multiview_disagreement", "hull_disagreement"] = "multiview_disagreement"

    @property
    def frame_count(self) -> int:
        return self.end_frame_exclusive - self.start_frame


class ProposedValidityInterval(VersionedModel):
    """A proposed (never applied) contact-eligibility change for the C10379 reference."""

    target: str = Field(min_length=1)
    start_frame: int = Field(ge=0)
    end_frame_exclusive: int = Field(gt=0)
    proposed_state: Literal["not_contact_eligible"]
    trigger: Literal["multiview_disagreement", "hull_disagreement"]
    rationale: str = Field(min_length=1)
    applied: Literal[False] = False


class ConsensusEpisodeRules(VersionedModel):
    reprojection_filter_px: float = Field(gt=0)
    min_views: int = Field(ge=2)
    disagreement_threshold_px: float = Field(gt=0)
    min_episode_frames: int = Field(ge=1)
    ego_wrist_gate_px: float | None = Field(default=None, gt=0)
    description: str = Field(min_length=1)


class WristTriangulationCheck(VersionedModel):
    frames: int = Field(ge=0)
    points: int = Field(ge=0)
    median_residual_mm: float = Field(ge=0)
    p95_residual_mm: float = Field(ge=0)
    max_residual_mm: float = Field(ge=0)


class PartConsensusSummary(VersionedModel):
    target: str = Field(min_length=1)
    frames_with_consensus: int = Field(ge=0)
    mean_views_used: float = Field(ge=0)
    per_view_mean_error_px: dict[str, float]
    per_view_agreement: dict[str, float]
    per_view_frames_observed: dict[str, int]
    episodes: int = Field(ge=0)
    reference_contradicted_frames: int = Field(ge=0)


class MultiviewConsensusManifest(VersionedModel):
    manifest_kind: Literal["multiview_part_consensus"]
    frame_count: int = Field(ge=1)
    analysis_fps: Literal[30]
    reference_view: str = Field(min_length=1)
    targets: tuple[str, ...] = Field(min_length=1)
    sources: tuple[ConsensusViewSource, ...] = Field(min_length=2)
    rules: ConsensusEpisodeRules
    per_frame_uri: str = Field(min_length=1)
    per_frame_fingerprint: ArtifactFingerprint | None = None
    summaries: tuple[PartConsensusSummary, ...]
    episodes: tuple[DisagreementEpisode, ...]
    reference_contradiction_intervals: tuple[tuple[str, int, int], ...]
    proposed_validity_intervals: tuple[ProposedValidityInterval, ...]
    wrist_triangulation: WristTriangulationCheck
    ego_pose_gate: dict[str, int] | None = None
    runtime_seconds: float = Field(ge=0)
    claim_boundaries: tuple[str, ...] = Field(min_length=1)


# -- visual hull ----------------------------------------------------------------------------


class HullViewComparison(VersionedModel):
    view: str = Field(min_length=1)
    target: str = Field(min_length=1)
    frames_compared: int = Field(ge=0)
    median_iou: float | None = Field(default=None, ge=0, le=1)
    p10_iou: float | None = Field(default=None, ge=0, le=1)
    median_area_ratio_mask_over_hull: float | None = Field(default=None, ge=0)
    mask_larger_than_hull_fraction: float | None = Field(default=None, ge=0, le=1)


class HullPartSummary(VersionedModel):
    target: str = Field(min_length=1)
    frames_with_hull: int = Field(ge=0)
    median_voxel_count: float = Field(ge=0)
    p10_voxel_count: float = Field(ge=0)
    p90_voxel_count: float = Field(ge=0)
    median_views_used: float = Field(ge=0)
    episodes: int = Field(ge=0)


class VisualHullManifest(VersionedModel):
    manifest_kind: Literal["multiview_visual_hull"]
    frame_count: int = Field(ge=1)
    analysis_fps: Literal[30]
    voxel_size_mm: float = Field(gt=0)
    grid_origin_mm: tuple[float, float, float]
    grid_shape: tuple[int, int, int]
    grid_axes: Literal["world_xyz"]
    table_plane: TablePlane
    volume_bounds_source: str = Field(min_length=1)
    views_used: tuple[str, ...] = Field(min_length=2)
    consensus_manifest: ArtifactFingerprint
    sources: tuple[ConsensusViewSource, ...] = Field(min_length=2)
    per_frame_uri: str = Field(min_length=1)
    voxels_npz_uri: str = Field(min_length=1)
    voxels_npz_format: str = Field(min_length=1)
    voxels_fps: int = Field(ge=1)
    hull_projection_masks_uri: str = Field(min_length=1)
    part_summaries: tuple[HullPartSummary, ...]
    view_comparisons: tuple[HullViewComparison, ...]
    episodes: tuple[DisagreementEpisode, ...]
    proposed_validity_intervals: tuple[ProposedValidityInterval, ...]
    disagreement_rules: ConsensusEpisodeRules
    runtime_seconds: float = Field(ge=0)
    not_run: dict[str, str]
    claim_boundaries: tuple[str, ...] = Field(min_length=1)
