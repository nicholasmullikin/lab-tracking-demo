"""Typed contracts for the multi-view pass: geometric seed transfer, part consensus, hulls.

Every record here is cross-view geometry on dataset context (shipped extrinsics, per-frame
ego poses) and fitted estimates (intrinsics, table plane).  Seeds on views other than the
human-reviewed C10379 are agent-authored (`selected_by="agent"`, provenance
`geometric_seed_transfer`); consensus and hull numbers measure cross-view disagreement,
never accuracy; CC BY-NC 4.0 attribution applies to the dataset assets.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Literal

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


# -- FineBio 3D object tracking (Sep 24 contracts, p0-contracts) ----------------------------
#
# Frame indices are raw-video frame indices of the shipped FineBio mp4s (all six videos of a
# trial have the same count at 30000/1001 fps); a window proxy built by
# `finebio_frames.proxy_ffmpeg_args(start_frame=S)` has proxy frame k == raw frame S+k, so
# `frame_index = S + k`. Pixels are raw-video pixels (fixed views 1920x1080, fpv 1920x1440).
# World units are the calibration board's centimetres with z into the bench (height = -z).
# Detector scores and SAM3 masks are model output, not truth; FineBio is non-commercial
# research data and no frame, mask or video is stored in these records.

FINEBIO_FIXED_VIEWS: tuple[str, ...] = ("T1", "T2", "T3", "T4", "T5")
FINEBIO_FPV_VIEW = "fpv"
FINEBIO_WORLD_UNITS = "board centimetres, z into the bench"

ObservationSource = Literal["detector", "sam3_decode", "sam3_video"]
CameraPoseProvenance = Literal["shipped", "marker_pnp"]
TrackState = Literal["observed", "single_view", "coasting", "held", "contained", "lost"]
TrackEventKind = Literal[
    "birth",
    "lost",
    "coasting",
    "reacquired",
    "ambiguous",
    "held",
    "contained",
    "handoff_reseed",
    "detector_reseed",
    # Sep 25 (p3-tracker-ext): identical-instance group tracks (formed inside a container
    # footprint; a member born as its own track; an individual whose identity joins a group).
    "group_formed",
    "group_split",
    "group_joined",
    # Sep 29 (p2-tracker-lines, defects): a line track whose observed classes flipped to a
    # second colour continues under a new id.
    "class_split",
    # Sep 29 (disposable tips, `battle-finebio-events --tip-events`): a pipette's tip state
    # turned on with its tip end in a tip rack, or off with its tip end at the trash can.
    "tip_picked",
    "tip_ejected",
]

Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]
Mat3 = tuple[Vec3, Vec3, Vec3]
BoxXYXY = tuple[float, float, float, float]


def _require_box(box: BoxXYXY, label: str) -> None:
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"{label} must have positive width and height, got {box}")


class FineBioObservation(VersionedModel):
    """One per-view, per-frame report about one slot: a detector box and/or a SAM3 mask.

    `slot` is the per-view instance id (``<class>#<k>``; for `detector` rows k is the
    same-class score rank in that frame, not an identity; for SAM3 rows it is the seeded slot).
    `point_px` is what the tracker consumes: the mask centroid when a mask is present, else the
    box centre. A row with neither box nor mask bbox is not an observation and is rejected.
    """

    view: str = Field(min_length=1)
    frame_index: int = Field(ge=0)
    slot: str = Field(min_length=1)
    object_class: str = Field(min_length=1)
    detector_score: float | None = Field(default=None, ge=0, le=1)
    box_xyxy_px: BoxXYXY | None = None
    mask_bbox_px: BoxXYXY | None = None
    mask_centroid_px: Vec2 | None = None
    mask_area_px: int | None = Field(default=None, ge=0)
    # Sep 28 (p0-axis-observations), all optional so every earlier row still validates: the
    # mask's principal axis as two endpoints in raw pixels (order unresolved: not tip / butt;
    # None on a compact mask), sqrt of the ratio of the largest to smallest second-moment
    # eigenvalue (1.0 is round), the median across-axis width along the axis, and the RMS
    # distance of the skeleton points to the fitted axis (a quality number). The provenance
    # carries `axis_method` (ransac_skeleton | pca | none) and `axis_reason` when None.
    mask_axis_px: tuple[Vec2, Vec2] | None = None
    mask_elongation: float | None = Field(default=None, ge=0)
    mask_width_px: float | None = Field(default=None, ge=0)
    mask_axis_residual_px: float | None = Field(default=None, ge=0)
    # Sep 29 (tip / butt by the width profile), optional: the median across-axis width over
    # the outer fifth of the axis at each end, in the order of `mask_axis_px`.
    mask_end_widths_px: tuple[float, float] | None = None
    # Sep 29 (v4, the long-thin-tail rule moved in from `finebio_tipseg`), optional: the
    # index into `mask_axis_px` of the end with the longer thin tail (the tip of a
    # single-channel pipette; None when the tails do not differ enough), and the terminal
    # centroid at each end in the order of `mask_axis_px` (the mask's own end, which sits on
    # a thin tip where the axis endpoint can be 30 px off it sideways).
    tip_side: int | None = Field(default=None, ge=0, le=1)
    body_end_px: tuple[Vec2, Vec2] | None = None
    sam3_object_score: float | None = None
    pose_valid: bool
    source: ObservationSource
    provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_a_box_or_a_mask(self) -> FineBioObservation:
        if self.box_xyxy_px is None and self.mask_bbox_px is None:
            raise ValueError("an observation carries a detector box, a mask bbox, or both")
        if self.box_xyxy_px is not None:
            _require_box(self.box_xyxy_px, "box_xyxy_px")
        if self.mask_bbox_px is not None:
            _require_box(self.mask_bbox_px, "mask_bbox_px")
        if self.source == "detector" and self.box_xyxy_px is None:
            raise ValueError("a detector observation carries its box")
        if self.source != "detector" and self.mask_bbox_px is None:
            raise ValueError("a SAM3 observation carries its mask bbox")
        return self

    @property
    def point_px(self) -> Vec2:
        if self.mask_centroid_px is not None:
            return self.mask_centroid_px
        box = self.mask_bbox_px if self.box_xyxy_px is None else self.box_xyxy_px
        assert box is not None
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


class FineBioFixedCamera(VersionedModel):
    """One fixed camera of a trial, in the shipped video's pixels, board centimetres."""

    view: str = Field(min_length=1)
    camera_id: int = Field(ge=1)
    provenance: CameraPoseProvenance
    K: Mat3
    distortion: tuple[float, float, float, float, float]
    rvec: Vec3
    tvec: Vec3
    image_size: tuple[int, int]
    # Median ArUco corner RMS of the pose in use, and of the shipped pose (equal when the
    # shipped pose is the one in use; the pair is the negative control for camera 6).
    marker_fit_residual_px: float | None = Field(default=None, ge=0)
    shipped_marker_residual_px: float | None = Field(default=None, ge=0)


class FineBioFpvCamera(VersionedModel):
    """The head camera: rescaled intrinsics plus where its per-frame shipped pose lives."""

    K: Mat3
    distortion: tuple[float, float, float, float, float]
    image_size: tuple[int, int]
    pose_source: str = Field(min_length=1)
    pose_frame_count: int = Field(ge=1)
    valid_pose_fraction: float | None = Field(default=None, ge=0, le=1)
    # Validity gate parameters for the shipped pose (the preflight's ~1.5% outlier frames):
    # a frame fails when the marker reprojection exceeds the first where markers are seen,
    # or the camera centre moves more than the second between consecutive frames.
    marker_residual_gate_px: float = Field(gt=0)
    velocity_gate_cm_per_frame: float = Field(gt=0)


class FineBioCameraConfig(VersionedModel):
    """Per-trial camera set: five fixed views plus the fpv, with pose provenance per view."""

    config_kind: Literal["finebio_camera_config"] = "finebio_camera_config"
    trial: str = Field(min_length=1)
    recording_day: str = Field(pattern=r"^\d{6}$")
    fixed: dict[str, FineBioFixedCamera]
    fpv: FineBioFpvCamera
    units: str = FINEBIO_WORLD_UNITS
    # raw_frame = proxy_frame + frame_index_offset. The trial-level config is written in raw
    # frames (offset 0); a per-window copy may set it to the proxy's start frame.
    frame_index_offset: int = Field(default=0, ge=0)
    provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_views_keyed_by_name(self) -> FineBioCameraConfig:
        for view, camera in self.fixed.items():
            if camera.view != view:
                raise ValueError(f"fixed camera {camera.view!r} is stored under key {view!r}")
        return self


class Track3D(VersionedModel):
    """One 3D track's state at one frame (a row of tracks.jsonl)."""

    frame_index: int = Field(ge=0)
    track_id: str = Field(min_length=1)
    object_class: str = Field(min_length=1)
    position_cm: Vec3
    uncertainty_cm: float = Field(ge=0)
    support_views: tuple[str, ...] = ()
    state: TrackState
    confidence: float = Field(ge=0, le=1)
    abstain: bool
    possibly_same_as: tuple[str, ...] = ()
    # Sep 25 (p3-tracker), all optional so earlier rows still validate: the per-view
    # reprojection residual of the state, the per-view slot associated this frame, the views
    # whose SAM3 slot differs from the slot this track had there before (a confidence signal,
    # never resolved silently), and the frames since the last observation while coasting.
    residual_px: dict[str, float] = Field(default_factory=dict)
    support_slots: dict[str, str] = Field(default_factory=dict)
    slot_disagreement_views: tuple[str, ...] = ()
    frames_unobserved: int = Field(default=0, ge=0)
    # Sep 25 (p3-tracker-ext), optional and None unless the extension that sets them is on:
    # the member count of an identical-instance group track, the group a track split from,
    # the container volume a `contained` track sits in, the hand track a `held` track follows.
    group_size: int | None = Field(default=None, ge=1)
    split_from: str | None = None
    container_id: str | None = None
    held_by: str | None = None
    # Sep 28 (p2-tracker-lines), None unless the track is a line track of the `--line-classes`
    # extension: the unit direction from `endpoints_cm[0]` to `endpoints_cm[1]`; the two
    # endpoints in cm (`endpoints_cm[0]` is the tip when `tip_resolved`, else the order is the
    # track's internal one); the plurality of the per-view observed classes (the geometric
    # class is `object_class`); the per-view perpendicular residual of the fitted line this
    # frame (undistorted px); the views dropped from this frame's fit as merged masks; and the
    # colour vote's identity and confidence (p2-colour-vote, filled by `finebio_colour`).
    direction: Vec3 | None = None
    endpoints_cm: tuple[Vec3, Vec3] | None = None
    tip_resolved: bool | None = None
    observed_class: str | None = None
    line_residual_px: dict[str, float] | None = None
    merged_views: tuple[str, ...] | None = None
    colour_identity: str | None = None
    colour_confidence: float | None = Field(default=None, ge=0, le=1)
    # Sep 29 (p2-tracker-lines, defects found on the first run), None unless a line track:
    # what moved the segment this frame (`fit` a multi-view line fit, `aided` one plane plus
    # the predicted direction, `point` the centroid fallback, `lateral` a single-view
    # update, `predicted` no geometric update, `capped` an update cut to the step cap);
    # whether the written extent was cut to the merged factor times the prior; and the
    # distance from the resolved butt / tip to the holding or nearest hand track (cm).
    line_update: str | None = None
    extent_clamped: bool | None = None
    butt_to_hand_cm: float | None = Field(default=None, ge=0)
    tip_to_hand_cm: float | None = Field(default=None, ge=0)
    # Sep 29 (disposable tips), None unless a line track with the tip boxes on: whether a
    # disposable tip is attached to the pipette (two-state with hysteresis; None while the
    # windows have not decided), the detector class of that tip (`blue_tip`, ...), the views
    # whose `*_tip` box attached to this frame's observation, and the basis of the tip / butt
    # resolution (`tip_box`, `hand_track`, `tail`, `hand_box`, `width`).
    tip_attached: bool | None = None
    tip_class: str | None = None
    tip_attached_views: tuple[str, ...] | None = None
    tip_basis: str | None = None
    # Sep 29 (orientation vote), optional so priority-mode rows stay valid and byte-identical.
    # `tip_confidence` is the sigmoid of the online log-odds (near 1 when endpoint 0 of the
    # track is the tip). `tip_votes` is that frame's per-cue signed log-odds in the track's
    # continuous endpoint order (positive: endpoint 0 is the tip). The episode index and the
    # online sign are what the retrofit and the scoreboard count; `orientation_retrofit_sign`
    # is set on `tracks_oriented.jsonl` (positive: the track's endpoint 0 is the tip written
    # back over the episode).
    tip_confidence: float | None = Field(default=None, ge=0, le=1)
    tip_votes: dict[str, float] | None = None
    orientation_episode: int | None = Field(default=None, ge=0)
    orientation_online_sign: int | None = None
    orientation_retrofit_sign: int | None = None
    # Sep 29 (v4): the median visible butt-to-tip length (cm) the 3D-length tip rule read on
    # this frame; None while too few line frames measured one, and on rows of a tracker
    # without the rule.
    tip_rule_length_cm: float | None = Field(default=None, ge=0)


class TrackEvent(VersionedModel):
    """One identity event of the tracker (a row of events.jsonl)."""

    frame_index: int = Field(ge=0)
    track_id: str = Field(min_length=1)
    kind: TrackEventKind
    payload: dict[str, Any] = Field(default_factory=dict)


def write_jsonl(rows: Iterable[VersionedModel], path: Path, *, compact: bool = False) -> int:
    """One JSON object per line, None fields omitted; `compact` also omits fields at their
    declared default (schema_version, empty provenance), which `read_jsonl` restores.
    Returns the row count."""
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.model_dump_json(exclude_none=True, exclude_defaults=compact))
            handle.write("\n")
            count += 1
    return count


def read_jsonl[T: VersionedModel](path: Path, model: type[T]) -> Iterator[T]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield model.model_validate(json.loads(line))
