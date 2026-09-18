"""Typed contracts for the Assembly101 dataset hand poses and fine-grained labels.

Everything here describes *dataset-shipped* reference data for one recording, resampled onto
the project's 30 FPS analysis clock.  It is external context for review, not a method output:
the poses were produced by the dataset's own multi-view tracker with a fixed-scale hand model,
the intrinsics are an estimate recovered from the dataset's own 2D/3D projection, and the
fine-grained segments are human annotations at 30 FPS.  None of it is ground truth for the
methods compared in this repository, and none of it may back an accuracy claim.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .schemas import ArtifactFingerprint, VersionedModel

ASSEMBLY101_LICENSE = "CC BY-NC 4.0"
ASSEMBLY101_CITATION = (
    "Sener et al., Assembly101: A Large-Scale Multi-View Video Dataset for Understanding "
    "Procedural Activities, CVPR 2022"
)

# Joint order of the dataset's 21-joint hand (official MS-G3D `assembly101_hands.py` graph).
# It is deliberately not remapped onto the MediaPipe order used by `PerFrameHand`: the
# dataset thumb has three joints plus a palm centre where MediaPipe has four thumb joints.
ASSEMBLY101_JOINT_NAMES: tuple[str, ...] = (
    "thumb_tip",
    "index_tip",
    "middle_tip",
    "ring_tip",
    "pinky_tip",
    "wrist",
    "thumb_cmc",
    "thumb_ip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "palm",
)
ASSEMBLY101_WRIST_INDEX = 5
ASSEMBLY101_EDGES: tuple[tuple[int, int], ...] = (
    (5, 17),
    (17, 18),
    (18, 19),
    (19, 4),
    (5, 14),
    (14, 15),
    (15, 16),
    (16, 3),
    (5, 11),
    (11, 12),
    (12, 13),
    (13, 2),
    (5, 8),
    (8, 9),
    (9, 10),
    (10, 1),
    (5, 6),
    (6, 7),
    (7, 0),
    (6, 8),
    (8, 11),
    (11, 14),
    (14, 17),
)
# Dataset hand index -> side (official HandFormer / MS-G3D preprocessing).
ASSEMBLY101_HAND_SIDES: dict[int, str] = {0: "left", 1: "right"}

HandIndex = Literal[0, 1]
HandSideName = Literal["left", "right"]


class Assembly101ClockRule(VersionedModel):
    """How an analysis-proxy frame maps onto the dataset's 60 FPS pose clock for one view.

    `pose_frame = proxy_start_raw_frame + raw_frames_per_proxy_frame * proxy_frame +
    pose_offset_frames`.  The offset is the measured lag of this camera's video behind the
    pose clock (zero for the ego view, +9 for the static C10379 view of this recording).
    """

    view_key: str = Field(min_length=1)
    proxy_start_raw_frame: int = Field(ge=0)
    raw_frames_per_proxy_frame: Literal[2]
    pose_offset_frames: int
    pose_fps: Literal[60]
    analysis_fps: Literal[30]
    offset_uncertainty_frames: int = Field(ge=0)
    offset_evidence: str = Field(min_length=1)

    def pose_frame(self, proxy_frame: int) -> int:
        if proxy_frame < 0:
            raise ValueError("proxy frame must be non-negative")
        return (
            self.proxy_start_raw_frame
            + self.raw_frames_per_proxy_frame * proxy_frame
            + self.pose_offset_frames
        )


DistortionModelName = Literal["brown", "rational"]
ExtrinsicsKind = Literal["fixed", "per_frame_ego"]
BROWN_COEFFICIENTS = 5
RATIONAL_COEFFICIENTS = 8


class Assembly101CameraModel(VersionedModel):
    """Estimated pinhole + distortion for one view at raw sensor resolution.

    The dataset ships extrinsics only.  These intrinsics were recovered by fitting the
    dataset's own 2D landmarks against its 3D landmarks through the shipped camera-to-world
    pose, so they reproduce the dataset's internal projection, not a physical calibration.

    Static views use OpenCV's five-coefficient Brown model and carry their constant
    `camera_to_world`.  Ego views need the eight-coefficient rational model and have no
    constant pose: `camera_to_world` is `None` and the per-frame pose comes from the dataset's
    `camera_extrinsics_ego` member (see `multiview_geometry.CameraRig`).
    """

    view_key: str = Field(min_length=1)
    raw_image_size: tuple[int, int]
    intrinsic_matrix: tuple[tuple[float, float, float], ...] = Field(min_length=3, max_length=3)
    distortion: tuple[float, ...] = Field(min_length=BROWN_COEFFICIENTS, max_length=14)
    distortion_model: DistortionModelName = "brown"
    extrinsics_kind: ExtrinsicsKind = "fixed"
    camera_to_world: tuple[tuple[float, float, float, float], ...] | None = Field(
        default=None, min_length=4, max_length=4
    )
    provenance: Literal["estimated_from_dataset_landmark_projection"]
    fit_rms_pixels: float = Field(ge=0)
    fit_point_count: int = Field(ge=1)
    fit_frame_count: int | None = Field(default=None, ge=1)
    shipped_extrinsics_rms_pixels: float | None = Field(default=None, ge=0)
    shipped_extrinsics_max_pixels: float | None = Field(default=None, ge=0)
    fit_notes: str | None = None

    @model_validator(mode="after")
    def require_consistent_model(self) -> Assembly101CameraModel:
        expected = BROWN_COEFFICIENTS if self.distortion_model == "brown" else RATIONAL_COEFFICIENTS
        if len(self.distortion) != expected:
            raise ValueError(
                f"{self.distortion_model} distortion needs {expected} coefficients, "
                f"got {len(self.distortion)}"
            )
        if (self.extrinsics_kind == "fixed") != (self.camera_to_world is not None):
            raise ValueError("fixed extrinsics need camera_to_world; per-frame ego views omit it")
        return self

    @property
    def is_ego(self) -> bool:
        return self.extrinsics_kind == "per_frame_ego"


class Assembly101Point3D(VersionedModel):
    """One joint in the dataset's world frame, millimetres."""

    x: float
    y: float
    z: float


class Assembly101Point2D(VersionedModel):
    """One projected joint in analysis-proxy pixels; may lie outside the image."""

    x: float
    y: float


class Assembly101Hand(VersionedModel):
    hand_index: HandIndex
    side: HandSideName
    confidence: float = Field(ge=0, le=1)
    joints_world_mm: tuple[Assembly101Point3D, ...] = Field(min_length=21, max_length=21)
    joints_proxy_pixels: tuple[Assembly101Point2D, ...] = Field(min_length=21, max_length=21)
    joints_inside_image: int = Field(ge=0, le=21)

    @model_validator(mode="after")
    def require_consistent_side(self) -> Assembly101Hand:
        if ASSEMBLY101_HAND_SIDES[self.hand_index] != self.side:
            raise ValueError("dataset hand index and side disagree")
        return self


class Assembly101HandFrame(VersionedModel):
    """Dataset hands for one analysis frame, taken from the mapped 60 FPS pose frame."""

    analysis_frame_index: int = Field(ge=0)
    pose_frame_index: int = Field(ge=0)
    pose_timestamp_seconds: float
    source_seconds: float = Field(ge=0)
    hands: tuple[Assembly101Hand, ...] = Field(max_length=2)


class Assembly101FineSegment(VersionedModel):
    """One fine-grained action segment, annotation clock (30 FPS) and proxy clock."""

    annotation_id: str = Field(min_length=1)
    action_id: int = Field(ge=0)
    verb: str = Field(min_length=1)
    noun: str = Field(min_length=1)
    action: str = Field(min_length=1)
    annotation_start_frame: int = Field(ge=0)
    annotation_end_frame: int = Field(ge=0)
    proxy_start_frame: int
    proxy_end_frame_exclusive: int
    clipped_to_window: bool

    @model_validator(mode="after")
    def require_ordered(self) -> Assembly101FineSegment:
        if self.annotation_end_frame <= self.annotation_start_frame:
            raise ValueError("fine-grained segment must end after it starts")
        if self.proxy_end_frame_exclusive <= self.proxy_start_frame:
            raise ValueError("clipped proxy range must be non-empty")
        return self


class Assembly101ProjectionCheck(VersionedModel):
    """Residual between our projection of the 3D joints and the dataset's shipped 2D."""

    compared_points: int = Field(ge=1)
    rms_pixels: float = Field(ge=0)
    max_pixels: float = Field(ge=0)
    shipped_2d_window: ArtifactFingerprint


class Assembly101ReferenceManifest(VersionedModel):
    """Index of one derived dataset-reference window for the review builders."""

    manifest_kind: Literal["assembly101_reference_window"]
    recording_id: str = Field(min_length=1)
    dataset_revision: str = Field(min_length=1)
    license: Literal["CC BY-NC 4.0"]
    citation: str = Field(min_length=1)
    view_key: str = Field(min_length=1)
    proxy_dimensions: tuple[int, int]
    frame_count: int = Field(ge=1)
    analysis_fps: Literal[30]
    source_start_seconds: float = Field(ge=0)
    annotation_start_frame: int = Field(ge=0)
    clock_rule: Assembly101ClockRule
    camera: Assembly101CameraModel
    draw_confidence_threshold: float = Field(ge=0, le=1)
    input_artifacts: tuple[ArtifactFingerprint, ...] = Field(min_length=1)
    hands_path: str = Field(min_length=1)
    fine_segments: tuple[Assembly101FineSegment, ...]
    projection_check: Assembly101ProjectionCheck | None = None
    coverage: dict[str, int]
    claim_boundaries: tuple[str, ...] = Field(min_length=1)
    hands_fingerprint: ArtifactFingerprint | None = None
