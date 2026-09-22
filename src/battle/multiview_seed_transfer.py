"""Agent-authored frame-0 seed transfer from the human-reviewed masks to the other views.

Two views of this recording carry human-reviewed frame-0 masks for the four parts: the static
C10379 run and the ego HMC_21110305 (e3) run.  C10379 sits at table height and sees the parts
at a grazing angle, so lifting its masks onto the table plane is ill-conditioned (a mask
centroid a few centimetres above the table sends the ray past the plane).  Instead each
part's frame-0 centroid is *triangulated* from the two human masks (`CameraRig.triangulate`,
e3 with its per-frame pose), its size is taken from the masks' pixel extent at the recovered
depth, and the point is projected into every other view as a box prompt (plus background
points at the other parts' projected centroids).  Prompts are decoded through the same
isolated MuggledSAM image decoder the calibration workspace uses.  The accepted candidate is
the one whose mask, warped through a table-parallel plane at the part's height into e3, has
the best IoU with the e3 human mask, subject to an area sanity band and an IoU floor.  Every
decoded alternative is kept.  Hands are seeded from the dataset joints and recorded only.

Nothing here is human review: seeds are `selected_by="agent"`, provenance
`geometric_seed_transfer`, and the run that consumes them says so.  CC BY-NC 4.0 applies.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from . import mask_ops
from .assembly101_camera_fit import PoseMembers
from .assembly101_clock_offset import is_ego
from .assembly101_pose_schemas import ASSEMBLY101_HAND_SIDES
from .cli_common import add_output_root, add_repository_root
from .digest_cache import sha256_file
from .four_part_contract import TARGETS
from .mask_ops import decode_mask_png as load_mask
from .mask_ops import mask_iou as iou
from .multiview_geometry import CameraRig, TablePlane
from .multiview_schemas import (
    MULTIVIEW_CLAIM_BOUNDARIES,
    MultiviewSeedTransferManifest,
    SeedAcceptanceRules,
    SeedCandidate,
    SeedPrompt,
    SeedTransferPart,
)
from .schemas import ArtifactFingerprint, G2PreprocessingManifest, PixelBox, PixelPoint, fingerprint

OUTPUT_ROOT = Path("runs/multiview-seed-transfer-20260918")
REFERENCE_VIEW = "C10379"
REFERENCE_RUN = Path(
    "runs/muggledsam-sam3-four-part-static-focused-reassembly-static-c10379-20260918t001210z"
)
SECONDARY_VIEW = "HMC_21110305"
SECONDARY_RUN = Path(
    "runs/muggledsam-sam3-four-part-ego-focused-reassembly-ego-hmc21110305-20260916t031515z"
)
ALL_STATIC_CONFIG = Path(
    "configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_all_static_g2.json"
)
FIRST_MINUTE_FRAMES = 1800
STATIC_PROXY_TO_RAW = 1.5  # 1920x1080 sensor, 1280x720 proxy
EGO_PROXY_TO_RAW = 1.0 / 1.5  # 636x480 sensor, 954x720 proxy
BOX_MARGINS = (0.25, 0.60)
HAND_BOX_MARGIN = 0.25
HAND_CONFIDENCE_FLOOR = 0.5
TRANSFER_METHOD = (
    "two-view centroid triangulation (C10379 human mask x e3 human mask, e3 at its own pose "
    "frame) -> sphere of the masks' pixel radius at depth -> per-view box prompt; acceptance by "
    "warp through a table-parallel plane at the part's height into e3"
)
RULES = SeedAcceptanceRules(
    min_area_ratio=0.3,
    max_area_ratio=3.0,
    min_backprojection_iou=0.25,
    max_centroid_ray_distance_radii=1.5,
    min_hand_joints_inside_fraction=0.6,
    min_parts_to_run=2,
    description=(
        "A part candidate passes when its area lies within [0.3, 3.0] x the e3 human mask area "
        "scaled by the squared focal-length/depth ratio of the two cameras at the triangulated "
        "centroid, and either (a) its warp through a table-parallel plane at the centroid "
        "height into e3 reaches IoU >= 0.25 with the e3 human mask, or (b) the ray through its "
        "own centroid passes within 1.5 part radii of the triangulated centroid (the plane warp "
        "is ill-conditioned for the low, grazing cameras). Candidates passing (a) rank by IoU "
        "and outrank (b); (b) ranks by ray distance. A hand candidate passes when it covers "
        ">= 60 % of the projected dataset joints and lies within [0.1, 1.0] of its prompt box "
        "area. A view runs when at least two parts pass."
    ),
)


def view_id_for(view: str) -> str:
    return ("ego-" if is_ego(view) else "static-") + view.replace("HMC_", "hmc").lower()


def proxy_to_raw_scale(view: str) -> float:
    return EGO_PROXY_TO_RAW if is_ego(view) else STATIC_PROXY_TO_RAW


def proxy_focal_px(rig: CameraRig, view: str) -> float:
    return float(rig.camera(view).intrinsic_matrix[0][0]) / proxy_to_raw_scale(view)


def mask_centroid(mask: np.ndarray) -> np.ndarray:
    """Pixel-centred `[x, y]` centroid in proxy pixels; an empty mask is an error here."""
    centroid = mask_ops.mask_centroid(mask, pixel_center=True)
    if centroid is None:
        raise ValueError("mask is empty")
    return np.array(centroid)


# -- geometry ---------------------------------------------------------------------------------


def rays_batch(rig: CameraRig, view: str, pixels_raw: np.ndarray, pose_frame: int | None):
    """World origin and unit directions for many raw pixels of one view."""
    normalised = rig.undistort(view, pixels_raw)
    directions_camera = np.concatenate([normalised, np.ones((normalised.shape[0], 1))], axis=1)
    pose = rig.camera_to_world(view, pose_frame)
    directions = directions_camera @ pose[:3, :3].T
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return pose[:3, 3], directions


def intersect_plane_batch(
    normal: np.ndarray, offset: float, origin: np.ndarray, directions: np.ndarray
) -> np.ndarray:
    """Hits of many rays with `normal . x = offset`; NaN where a ray misses or points away."""
    denominator = directions @ normal
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (offset - origin @ normal) / denominator
    points = origin[None, :] + t[:, None] * directions
    bad = (np.abs(denominator) < 1e-9) | ~(t > 0)
    points[bad] = np.nan
    return points


def lift_pixels_to_plane(
    rig: CameraRig,
    normal: np.ndarray,
    offset: float,
    view: str,
    pixels_proxy: np.ndarray,
    pose_frame: int | None = None,
) -> np.ndarray:
    pixels_raw = np.asarray(pixels_proxy, dtype=np.float64).reshape(-1, 2) * proxy_to_raw_scale(
        view
    )
    origin, directions = rays_batch(rig, view, pixels_raw, pose_frame)
    return intersect_plane_batch(normal, offset, origin, directions)


def project_to_proxy(
    rig: CameraRig, view: str, points_world: np.ndarray, pose_frame: int | None = None
) -> np.ndarray:
    """World mm -> proxy pixels of `view`; NaN for NaN input or points behind the camera."""
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    result = np.full((points.shape[0], 2), np.nan)
    good = ~np.isnan(points).any(axis=1)
    if good.any():
        depth = rig.depth(view, points[good], pose_frame)
        raw = rig.project(view, points[good], pose_frame)
        raw[depth <= 0] = np.nan
        result[good] = raw / proxy_to_raw_scale(view)
    return result


def warp_mask_via_plane(
    rig: CameraRig,
    normal: np.ndarray,
    offset: float,
    *,
    source_view: str,
    source_mask: np.ndarray,
    target_view: str,
    target_shape: tuple[int, int],
    target_region: tuple[int, int, int, int],
    source_pose_frame: int | None = None,
    target_pose_frame: int | None = None,
) -> np.ndarray:
    """Resample `source_mask` (source proxy px) into `target_view` proxy pixels via a plane.

    Inverse warping over `target_region` (x0, y0, x1, y1): every target pixel is lifted to the
    plane and looked up in the source mask, which leaves no holes.
    """
    x0, y0, x1, y1 = target_region
    height, width = target_shape
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    warped = np.zeros(target_shape, dtype=bool)
    if x1 <= x0 or y1 <= y0:
        return warped
    grid_y, grid_x = np.mgrid[y0:y1, x0:x1]
    pixels = np.stack([grid_x.ravel() + 0.5, grid_y.ravel() + 0.5], axis=1)
    world = lift_pixels_to_plane(rig, normal, offset, target_view, pixels, target_pose_frame)
    source_px = project_to_proxy(rig, source_view, world, source_pose_frame)
    finite = np.isfinite(source_px).all(axis=1)
    sx = np.zeros(pixels.shape[0], dtype=np.int64)
    sy = np.zeros(pixels.shape[0], dtype=np.int64)
    # Points far outside the source frustum project to huge coordinates; clip before casting.
    clipped = np.clip(source_px[finite], -1.0, 1.0e6)
    sx[finite] = np.floor(clipped[:, 0]).astype(np.int64)
    sy[finite] = np.floor(clipped[:, 1]).astype(np.int64)
    inside = (
        finite & (sx >= 0) & (sx < source_mask.shape[1]) & (sy >= 0) & (sy < source_mask.shape[0])
    )
    hit = np.zeros(pixels.shape[0], dtype=bool)
    hit[inside] = source_mask[sy[inside], sx[inside]]
    warped[y0:y1, x0:x1] = hit.reshape(y1 - y0, x1 - x0)
    return warped


def ray_point_distance(
    rig: CameraRig, view: str, pixel_proxy: np.ndarray, point: np.ndarray, pose_frame: int | None
) -> float:
    """Distance (mm) from `point` to the world ray through one proxy pixel of `view`."""
    origin, directions = rays_batch(
        rig, view, np.asarray(pixel_proxy).reshape(1, 2) * proxy_to_raw_scale(view), pose_frame
    )
    offset = point - origin
    along = float(offset @ directions[0])
    if along <= 0:
        return float("inf")
    return float(np.linalg.norm(offset - along * directions[0]))


def bounding_box(
    points_proxy: np.ndarray, margin: float, shape: tuple[int, int]
) -> PixelBox | None:
    valid = points_proxy[~np.isnan(points_proxy).any(axis=1)]
    if valid.shape[0] < 2:
        return None
    x_min, y_min = valid.min(axis=0)
    x_max, y_max = valid.max(axis=0)
    width, height = x_max - x_min, y_max - y_min
    x_min -= width * margin
    x_max += width * margin
    y_min -= height * margin
    y_max += height * margin
    h, w = shape
    x1, y1 = int(np.clip(np.floor(x_min), 0, w - 1)), int(np.clip(np.floor(y_min), 0, h - 1))
    x2, y2 = int(np.clip(np.ceil(x_max), 1, w)), int(np.clip(np.ceil(y_max), 1, h))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return PixelBox(x1=x1, y1=y1, x2=x2, y2=y2)


def square_box(
    centre: np.ndarray, half_size: float, margin: float, shape: tuple[int, int]
) -> PixelBox | None:
    half = half_size * (1.0 + margin)
    corners = np.array([centre - half, centre + half])
    return bounding_box(corners, 0.0, shape)


def inside_box(point: np.ndarray, box: PixelBox, slack: float = 0.25) -> bool:
    width, height = box.x2 - box.x1, box.y2 - box.y1
    return bool(
        box.x1 - slack * width <= point[0] <= box.x2 + slack * width
        and box.y1 - slack * height <= point[1] <= box.y2 + slack * height
    )


# -- reference inputs -------------------------------------------------------------------------


def human_seed_masks(
    repository_root: Path, run: Path
) -> tuple[dict[str, Path], list[ArtifactFingerprint], ArtifactFingerprint]:
    """The human frame-0 seed masks a four-part run was initialised with, fingerprinted."""
    run_dir = repository_root / run
    settings = json.loads((run_dir / "runtime_settings.json").read_text(encoding="utf-8"))
    masks: dict[str, Path] = {}
    fingerprints: list[ArtifactFingerprint] = []
    for seed in settings["multi_keyframe_correction_schedule"]["seeds"]:
        # Records written before `selected_by` existed were human-reviewed by construction.
        if seed["frame_index"] != 0 or seed.get("selected_by", "human") != "human":
            raise ValueError("reference frame-0 seeds must be human-selected")
        path = Path(seed["mask_path"])
        if sha256_file(path) != seed["mask_sha256"]:
            raise ValueError(f"reference seed mask changed: {path}")
        masks[seed["target"]] = path
        fingerprints.append(fingerprint(path, repository_root))
    if tuple(masks) != TARGETS:
        raise ValueError(f"reference seeds must be the ordered targets {TARGETS}")
    return masks, fingerprints, fingerprint(run_dir / "manifest.json", repository_root)


def fit_first_minute_plane(rig: CameraRig, members: PoseMembers) -> TablePlane:
    start = rig.pose_frame(REFERENCE_VIEW, 0)
    return rig.fit_table_plane(
        members.landmarks3d,
        members.confidences,
        pose_frames=range(start, start + 2 * FIRST_MINUTE_FRAMES),
    )


def dataset_hands(
    members: PoseMembers, pose_frame: int, confidence_floor: float = HAND_CONFIDENCE_FLOOR
) -> dict[str, np.ndarray]:
    key = str(pose_frame)
    hands: dict[str, np.ndarray] = {}
    frame = members.landmarks3d.get(key)
    if frame is None:
        return hands
    for index, side in ASSEMBLY101_HAND_SIDES.items():
        if float(members.confidences[key][str(index)]) < confidence_floor:
            continue
        hands[f"{side}_hand"] = np.asarray(frame[str(index)], dtype=np.float64)
    return hands


# -- planning ---------------------------------------------------------------------------------


class SeedTransferPlanner:
    """Triangulate the four frame-0 part centroids once; plan prompts for any view."""

    def __init__(self, repository_root: Path, *, config_path: Path = ALL_STATIC_CONFIG) -> None:
        self.repository_root = repository_root.resolve()
        self.rig = CameraRig.load(self.repository_root)
        self.members = PoseMembers(self.repository_root)
        self.plane = fit_first_minute_plane(self.rig, self.members)
        self.config_path = self.repository_root / config_path
        self.config = G2PreprocessingManifest.model_validate_json(
            self.config_path.read_text(encoding="utf-8")
        )
        masks, self.reference_fingerprints, self.reference_manifest = human_seed_masks(
            self.repository_root, REFERENCE_RUN
        )
        secondary, self.secondary_fingerprints, self.secondary_manifest = human_seed_masks(
            self.repository_root, SECONDARY_RUN
        )
        self.reference_masks = {target: load_mask(path) for target, path in masks.items()}
        self.secondary_masks = {target: load_mask(path) for target, path in secondary.items()}
        self.secondary_pose_frame = self.rig.pose_frame(SECONDARY_VIEW, 0)
        self.centroids_world: dict[str, np.ndarray] = {}
        self.reprojection_px: dict[str, dict[str, float]] = {}
        self.radius_mm: dict[str, float] = {}
        for target in TARGETS:
            self._triangulate(target)

    def _triangulate(self, target: str) -> None:
        reference_px = mask_centroid(self.reference_masks[target]) * proxy_to_raw_scale(
            REFERENCE_VIEW
        )
        secondary_px = mask_centroid(self.secondary_masks[target]) * proxy_to_raw_scale(
            SECONDARY_VIEW
        )
        result = self.rig.triangulate(
            {
                REFERENCE_VIEW: reference_px.reshape(1, 2),
                SECONDARY_VIEW: secondary_px.reshape(1, 2),
            },
            pose_frame={SECONDARY_VIEW: self.secondary_pose_frame},
            reproj_filter_px=None,
        )
        point = result.points[0]
        if np.isnan(point).any():
            raise ValueError(f"{target}: the two human masks do not triangulate")
        self.centroids_world[target] = point
        self.reprojection_px[target] = {
            view: float(result.reprojection_px[0, index]) for index, view in enumerate(result.views)
        }
        # Sphere radius from each mask's equivalent-circle radius at the recovered depth; the
        # larger of the two views is kept so the prompt box does not cut a part short.
        radii = []
        for view, mask, pose in (
            (REFERENCE_VIEW, self.reference_masks[target], None),
            (SECONDARY_VIEW, self.secondary_masks[target], self.secondary_pose_frame),
        ):
            depth = float(self.rig.depth(view, point.reshape(1, 3), pose)[0])
            radius_px = np.sqrt(mask.sum() / np.pi)
            radii.append(radius_px * depth / proxy_focal_px(self.rig, view))
        self.radius_mm[target] = float(max(radii))

    def proxy_for(self, view: str, config: G2PreprocessingManifest | None = None) -> Any:
        config = config or self.config
        view_id = view_id_for(view)
        proxy = next((item for item in config.proxies if item.view_id == view_id), None)
        if proxy is None:
            raise ValueError(f"{view_id} is not in {self.config_path}")
        return proxy

    def expected_area(self, view: str, target: str, pose_frame: int | None) -> float | None:
        """e3 human mask area carried to `view` by the squared focal/depth ratio."""
        point = self.centroids_world[target].reshape(1, 3)
        z_secondary = float(self.rig.depth(SECONDARY_VIEW, point, self.secondary_pose_frame)[0])
        z_view = float(self.rig.depth(view, point, pose_frame)[0])
        if z_secondary <= 0 or z_view <= 0:
            return None
        scale = (
            proxy_focal_px(self.rig, view)
            * z_secondary
            / (proxy_focal_px(self.rig, SECONDARY_VIEW) * z_view)
        ) ** 2
        return float(self.secondary_masks[target].sum() * scale)

    def plan_view(
        self, view: str, *, config_path: Path | None = None, analysis_frame: int = 0
    ) -> MultiviewSeedTransferManifest:
        config = (
            G2PreprocessingManifest.model_validate_json(
                (self.repository_root / config_path).read_text(encoding="utf-8")
            )
            if config_path is not None
            else self.config
        )
        proxy = self.proxy_for(view, config)
        shape = (proxy.dimensions.height, proxy.dimensions.width)
        pose_frame = self.rig.pose_frame(view, analysis_frame)
        ego_pose = pose_frame if is_ego(view) else None
        centroids = {
            target: project_to_proxy(self.rig, view, point.reshape(1, 3), ego_pose)[0]
            for target, point in self.centroids_world.items()
        }
        parts: list[SeedTransferPart] = []
        counter = 1
        for target in TARGETS:
            point = self.centroids_world[target]
            centroid = centroids[target]
            depth = float(self.rig.depth(view, point.reshape(1, 3), ego_pose)[0])
            expected = self.expected_area(view, target, ego_pose)
            common = dict(
                target=target,
                kind="part",
                source_points_world_mm=(tuple(float(v) for v in point),),
                triangulation_reprojection_px=self.reprojection_px[target],
                height_above_table_mm=float(self.plane.signed_distance(point.reshape(1, 3))[0]),
                radius_mm=self.radius_mm[target],
                projected_centroid_proxy_px=(
                    None if np.isnan(centroid).any() else (float(centroid[0]), float(centroid[1]))
                ),
                depth_mm=depth,
                expected_area_px=expected,
            )
            in_frame = (
                not np.isnan(centroid).any()
                and depth > 0
                and 0 <= centroid[0] < shape[1]
                and 0 <= centroid[1] < shape[0]
            )
            if not in_frame:
                parts.append(
                    SeedTransferPart(
                        status="blocked",
                        blocked_reason="projected centroid falls outside the view",
                        **common,
                    )
                )
                continue
            half_size = self.radius_mm[target] * proxy_focal_px(self.rig, view) / depth
            prompts: list[SeedPrompt] = []
            for margin in BOX_MARGINS:
                box = square_box(centroid, half_size, margin, shape)
                if box is None:
                    continue
                background = tuple(
                    PixelPoint(x=int(round(other[0])), y=int(round(other[1])))
                    for other_target, other in centroids.items()
                    if other_target != target
                    and not np.isnan(other).any()
                    and inside_box(other, box)
                    and 0 <= other[0] < shape[1]
                    and 0 <= other[1] < shape[0]
                )
                prompts.append(
                    SeedPrompt(
                        prompt_id=f"t{analysis_frame:06d}-b{counter:02d}",
                        target=target,
                        variant=f"triangulated_sphere_box_margin_{margin:.2f}",
                        pixel_box=box,
                        background_points=background,
                    )
                )
                counter += 1
            if not prompts:
                parts.append(
                    SeedTransferPart(
                        status="blocked",
                        blocked_reason="projected footprint is degenerate in this view",
                        **common,
                    )
                )
                continue
            parts.append(
                SeedTransferPart(
                    status="blocked",
                    blocked_reason="not decoded yet",
                    prompts=tuple(prompts),
                    **common,
                )
            )
        hands: list[SeedTransferPart] = []
        for name, joints in dataset_hands(self.members, pose_frame).items():
            projected = project_to_proxy(self.rig, view, joints, ego_pose)
            box = bounding_box(projected, HAND_BOX_MARGIN, shape)
            wrist = projected[5]
            common = dict(
                target=name,
                kind="hand",
                source_points_world_mm=tuple(tuple(float(v) for v in row) for row in joints),
                projected_centroid_proxy_px=(
                    None if np.isnan(wrist).any() else (float(wrist[0]), float(wrist[1]))
                ),
            )
            if box is None or np.isnan(wrist).any():
                hands.append(
                    SeedTransferPart(
                        status="blocked", blocked_reason="hand projects outside the view", **common
                    )
                )
                continue
            hands.append(
                SeedTransferPart(
                    status="blocked",
                    blocked_reason="not decoded yet",
                    prompts=(
                        SeedPrompt(
                            prompt_id=f"t{analysis_frame:06d}-b{counter:02d}",
                            target=name,
                            variant="dataset_joint_box",
                            pixel_box=box,
                            foreground_points=(
                                PixelPoint(x=int(round(wrist[0])), y=int(round(wrist[1]))),
                            ),
                        ),
                    ),
                    expected_area_px=float((box.x2 - box.x1) * (box.y2 - box.y1)),
                    **common,
                )
            )
            counter += 1
        return MultiviewSeedTransferManifest(
            manifest_kind="multiview_geometric_seed_transfer",
            view=view,
            view_id=view_id_for(view),
            analysis_frame_index=analysis_frame,
            pose_frame_index=pose_frame,
            is_ego=is_ego(view),
            reference_view=REFERENCE_VIEW,
            reference_run_manifest=self.reference_manifest,
            reference_seed_masks=tuple(self.reference_fingerprints),
            secondary_reference_view=SECONDARY_VIEW,
            secondary_reference_pose_frame=self.secondary_pose_frame,
            secondary_reference_run_manifest=self.secondary_manifest,
            secondary_reference_seed_masks=tuple(self.secondary_fingerprints),
            transfer_method=TRANSFER_METHOD,
            clip_config=fingerprint(
                self.repository_root / (config_path or self.config_path), self.repository_root
            ),
            proxy=ArtifactFingerprint(
                uri=proxy.proxy_uri, sha256=proxy.checksum_sha256, source="approved_config"
            ),
            proxy_dimensions=(proxy.dimensions.width, proxy.dimensions.height),
            proxy_to_raw_scale=proxy_to_raw_scale(view),
            table_plane=self.plane,
            rules=RULES,
            parts=tuple(parts),
            hands=tuple(hands),
            decode_state="planned",
            run_decision="skip" if not any(part.prompts for part in parts) else "pending",
            run_decision_reason=(
                "seed_transfer_failed: no part projects inside the view at frame "
                f"{analysis_frame} ({'; '.join(f'{p.target}: {p.blocked_reason}' for p in parts)})"
                if not any(part.prompts for part in parts)
                else "prompts planned; decode has not run"
            ),
            claim_boundaries=MULTIVIEW_CLAIM_BOUNDARIES,
        )


def write_manifest(path: Path, manifest: MultiviewSeedTransferManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_manifest(path: Path) -> MultiviewSeedTransferManifest:
    return MultiviewSeedTransferManifest.model_validate_json(path.read_text(encoding="utf-8"))


def decode_requests(manifest: MultiviewSeedTransferManifest) -> list[dict[str, Any]]:
    """Worker `batch_decode` prompts in MuggledSAM's normalised shapes."""
    width, height = manifest.proxy_dimensions
    requests: list[dict[str, Any]] = []
    for part in (*manifest.parts, *manifest.hands):
        for prompt in part.prompts:
            box = prompt.pixel_box
            requests.append(
                {
                    "box_id": f"p{manifest.analysis_frame_index:06d}-b{prompt.prompt_id[-2:]}",
                    "candidate_id": prompt.prompt_id,
                    "frame_index": manifest.analysis_frame_index,
                    "pixel_box": box.model_dump(mode="json"),
                    "intended_target": prompt.target,
                    "boxes": [
                        [[box.x1 / width, box.y1 / height], [box.x2 / width, box.y2 / height]]
                    ],
                    "fg_points": [[p.x / width, p.y / height] for p in prompt.foreground_points],
                    "bg_points": [[p.x / width, p.y / height] for p in prompt.background_points],
                }
            )
    return requests


# -- decoding (GPU) ---------------------------------------------------------------------------


def decode_view(
    manifest_path: Path,
    *,
    repository_root: Path,
    external_python: Path | None = None,
    model: Path | None = None,
    device: str = "cuda:0",
    decoder: Any | None = None,
) -> Path:
    """Decode every planned prompt with one warm image decoder; write `decode_result.json`."""
    from .muggled_calibration_web import WorkerClient
    from .muggled_smoke import DEFAULT_MODEL, MUGGLED_SAM_PYTHON, MUGGLED_SAM_SOURCE

    manifest = load_manifest(manifest_path)
    view_dir = manifest_path.parent
    proxy_path = (repository_root / manifest.proxy.uri).resolve()
    if sha256_file(proxy_path) != manifest.proxy.sha256:
        raise ValueError("proxy checksum no longer matches the approved config")
    requests = decode_requests(manifest)
    if not requests:
        raise ValueError(f"{manifest.view} has no prompts to decode")
    owns = decoder is None
    if decoder is None:
        external_python = external_python or MUGGLED_SAM_PYTHON
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = "0"
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(MUGGLED_SAM_SOURCE), environment["PYTHONPATH"]]
            if environment.get("PYTHONPATH")
            else [str(MUGGLED_SAM_SOURCE)]
        )
        decoder = WorkerClient(
            [
                str(external_python),
                str(Path(__file__).with_name("muggled_calibration_worker.py")),
                "--serve-jsonl",
                "--proxy",
                str(proxy_path),
                "--model",
                str((model or DEFAULT_MODEL).resolve()),
                "--results-directory",
                str(view_dir / "results"),
                "--device",
                device,
            ],
            environment=environment,
            stderr_path=view_dir / "decode_worker.stderr.log",
        )
    started = time.monotonic()
    try:
        preview = decoder.request(
            "frame_preview", {"frame_index": manifest.analysis_frame_index}, timeout=600
        )
        response = decoder.request("batch_decode", {"prompts": requests}, timeout=900)
    finally:
        if owns:
            decoder.close()
    result = {
        "view": manifest.view,
        "frame_preview": preview,
        "decoded": response["decoded"],
        "elapsed_seconds": time.monotonic() - started,
        "prompt_count": len(requests),
    }
    output = view_dir / "decode_result.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


# -- acceptance (CPU) -------------------------------------------------------------------------


def _score_part(
    part: SeedTransferPart,
    candidates: list[tuple[SeedCandidate, np.ndarray]],
    *,
    rig: CameraRig,
    manifest: MultiviewSeedTransferManifest,
    secondary_mask: np.ndarray,
) -> SeedTransferPart:
    rules = manifest.rules
    normal = np.asarray(manifest.table_plane.normal)
    point = np.asarray(part.source_points_world_mm[0])
    offset = float(normal @ point)
    ys, xs = np.nonzero(secondary_mask)
    span = int(max(xs.max() - xs.min(), ys.max() - ys.min()))
    region = (
        int(xs.min()) - 2 * span,
        int(ys.min()) - 2 * span,
        int(xs.max()) + 2 * span,
        int(ys.max()) + 2 * span,
    )
    ego_pose = manifest.pose_frame_index if manifest.is_ego else None
    scored: list[SeedCandidate] = []
    for candidate, mask in candidates:
        notes: list[str] = []
        area_ratio = (
            candidate.mask_area_px / part.expected_area_px if part.expected_area_px else None
        )
        warped = warp_mask_via_plane(
            rig,
            normal,
            offset,
            source_view=manifest.view,
            source_mask=mask,
            target_view=SECONDARY_VIEW,
            target_shape=secondary_mask.shape,
            target_region=region,
            source_pose_frame=ego_pose,
            target_pose_frame=manifest.secondary_reference_pose_frame,
        )
        overlap = iou(warped, secondary_mask)
        ray_distance = ray_point_distance(rig, manifest.view, mask_centroid(mask), point, ego_pose)
        radius = part.radius_mm or 0.0
        passed = True
        if area_ratio is None:
            passed = False
            notes.append("no expected area (centroid depth unavailable)")
        elif not rules.min_area_ratio <= area_ratio <= rules.max_area_ratio:
            passed = False
            notes.append(f"area ratio {area_ratio:.2f} outside the sanity band")
        basis = None
        if overlap >= rules.min_backprojection_iou:
            basis = "plane_warp_iou"
        elif ray_distance <= rules.max_centroid_ray_distance_radii * radius:
            basis = "centroid_ray"
            notes.append(f"back-projection IoU {overlap:.2f} below the floor; ray distance used")
        else:
            passed = False
            notes.append(
                f"back-projection IoU {overlap:.2f} below the floor and centroid ray "
                f"{ray_distance:.0f} mm from the triangulated point (radius {radius:.0f} mm)"
            )
        scored.append(
            candidate.model_copy(
                update={
                    "backprojection_iou": overlap,
                    "centroid_ray_distance_mm": None if np.isinf(ray_distance) else ray_distance,
                    "area_ratio_vs_expected": area_ratio,
                    "sanity_pass": passed,
                    "acceptance_basis": basis if passed else None,
                    "sanity_notes": tuple(notes),
                }
            )
        )
    passing = [c for c in scored if c.sanity_pass]
    if passing:

        def rank(c: SeedCandidate) -> tuple[float, float, float]:
            if c.acceptance_basis == "plane_warp_iou":
                return (1.0, c.backprojection_iou or 0.0, c.decoder_iou_estimate)
            return (0.0, -(c.centroid_ray_distance_mm or 0.0), c.decoder_iou_estimate)

        best = max(passing, key=rank)
        return part.model_copy(
            update={
                "status": "accepted",
                "blocked_reason": None,
                "candidates": tuple(scored),
                "accepted": best,
            }
        )
    best_iou = max((c.backprojection_iou or 0.0 for c in scored), default=0.0)
    return part.model_copy(
        update={
            "status": "blocked",
            "blocked_reason": (
                "seed_transfer_failed: no candidate passed sanity "
                f"(best back-projection IoU {best_iou:.2f} over {len(scored)} candidates)"
            ),
            "candidates": tuple(scored),
            "accepted": None,
        }
    )


def _score_hand(
    hand: SeedTransferPart,
    candidates: list[tuple[SeedCandidate, np.ndarray]],
    *,
    rig: CameraRig,
    manifest: MultiviewSeedTransferManifest,
) -> SeedTransferPart:
    rules = manifest.rules
    width, height = manifest.proxy_dimensions
    ego_pose = manifest.pose_frame_index if manifest.is_ego else None
    joints = np.asarray(hand.source_points_world_mm, dtype=np.float64)
    projected = project_to_proxy(rig, manifest.view, joints, ego_pose)
    finite = ~np.isnan(projected).any(axis=1)
    px = np.zeros(projected.shape, dtype=np.int64)
    px[finite] = np.floor(projected[finite]).astype(np.int64)
    valid = finite & (px[:, 0] >= 0) & (px[:, 0] < width) & (px[:, 1] >= 0) & (px[:, 1] < height)
    scored: list[SeedCandidate] = []
    for candidate, mask in candidates:
        inside = float(mask[px[valid, 1], px[valid, 0]].mean()) if valid.any() else 0.0
        area_ratio = (
            candidate.mask_area_px / hand.expected_area_px if hand.expected_area_px else None
        )
        notes: list[str] = []
        passed = inside >= rules.min_hand_joints_inside_fraction
        if not passed:
            notes.append(f"only {inside:.2f} of projected joints inside")
        if area_ratio is None or not 0.1 <= area_ratio <= 1.0:
            passed = False
            notes.append("area outside [0.1, 1.0] of the prompt box")
        scored.append(
            candidate.model_copy(
                update={
                    "joints_inside_fraction": inside,
                    "area_ratio_vs_expected": area_ratio,
                    "sanity_pass": passed,
                    "acceptance_basis": "dataset_joints" if passed else None,
                    "sanity_notes": tuple(notes),
                }
            )
        )
    passing = [c for c in scored if c.sanity_pass]
    if passing:
        best = max(passing, key=lambda c: (c.joints_inside_fraction or 0.0, c.decoder_iou_estimate))
        return hand.model_copy(
            update={
                "status": "accepted",
                "blocked_reason": None,
                "candidates": tuple(scored),
                "accepted": best,
            }
        )
    return hand.model_copy(
        update={
            "status": "blocked",
            "blocked_reason": "seed_transfer_failed: no hand candidate passed sanity",
            "candidates": tuple(scored),
        }
    )


def accept_view(
    manifest_path: Path, *, repository_root: Path, rig: CameraRig | None = None
) -> MultiviewSeedTransferManifest:
    """Score every decoded candidate geometrically and pick one seed per target."""
    manifest = load_manifest(manifest_path)
    view_dir = manifest_path.parent
    decoded_path = view_dir / "decode_result.json"
    if not decoded_path.is_file():
        raise FileNotFoundError(f"decode has not run for {manifest.view}: {decoded_path}")
    decoded = {
        item["candidate_id"]: item["decoder_result"]
        for item in json.loads(decoded_path.read_text(encoding="utf-8"))["decoded"]
    }
    rig = rig or CameraRig.load(repository_root)
    secondary_paths, _, _ = human_seed_masks(repository_root, SECONDARY_RUN)
    secondary = {target: load_mask(path) for target, path in secondary_paths.items()}

    def candidates_for(part: SeedTransferPart) -> list[tuple[SeedCandidate, np.ndarray]]:
        out: list[tuple[SeedCandidate, np.ndarray]] = []
        for prompt in part.prompts:
            result = decoded.get(prompt.prompt_id)
            if result is None:
                continue
            for candidate in result["candidates"]:
                mask_path = view_dir / candidate["mask_uri"]
                mask = load_mask(mask_path)
                out.append(
                    (
                        SeedCandidate(
                            prompt_id=prompt.prompt_id,
                            candidate_index=int(candidate["candidate_index"]),
                            mask=fingerprint(mask_path, repository_root),
                            decoder_iou_estimate=float(candidate["iou_score"]),
                            mask_area_px=int(mask.sum()),
                            sanity_pass=False,
                        ),
                        mask,
                    )
                )
        return out

    parts = tuple(
        _score_part(
            part,
            candidates_for(part),
            rig=rig,
            manifest=manifest,
            secondary_mask=secondary[part.target],
        )
        if part.prompts
        else part
        for part in manifest.parts
    )
    hands = tuple(
        _score_hand(hand, candidates_for(hand), rig=rig, manifest=manifest)
        if hand.prompts
        else hand
        for hand in manifest.hands
    )
    accepted_count = sum(part.status == "accepted" for part in parts)
    run = accepted_count >= manifest.rules.min_parts_to_run
    updated = manifest.model_copy(
        update={
            "parts": parts,
            "hands": hands,
            "decode_state": "decoded",
            "run_decision": "run" if run else "skip",
            "run_decision_reason": (
                f"{accepted_count} of {len(TARGETS)} parts passed sanity"
                + ("" if run else f"; fewer than {manifest.rules.min_parts_to_run}, view skipped")
            ),
        }
    )
    write_manifest(manifest_path, updated)
    return updated


def summary_line(manifest: MultiviewSeedTransferManifest) -> str:
    parts = ", ".join(
        f"{part.target}={'ok' if part.status == 'accepted' else 'blocked'}"
        + (
            f"(iou {part.accepted.backprojection_iou:.2f}, "
            f"ray {part.accepted.centroid_ray_distance_mm or float('nan'):.0f} mm, "
            f"area x{part.accepted.area_ratio_vs_expected:.2f}, {part.accepted.acceptance_basis})"
            if part.accepted is not None
            else ""
        )
        for part in manifest.parts
    )
    hands = ", ".join(
        f"{hand.target}={'ok' if hand.status == 'accepted' else 'blocked'}"
        for hand in manifest.hands
    )
    return f"{manifest.view}: {manifest.run_decision} [{parts}] hands [{hands}]"


def planned_views(repository_root: Path, root: Path = OUTPUT_ROOT) -> Iterable[Path]:
    yield from sorted((repository_root / root).glob("*/seed_manifest.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_repository_root(parser)
    add_output_root(parser, OUTPUT_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="write prompts for the given views (CPU)")
    plan.add_argument("--view", action="append", required=True)
    plan.add_argument("--config", type=Path, default=ALL_STATIC_CONFIG)

    decode = commands.add_parser("decode", help="decode one view's prompts (GPU) then accept")
    decode.add_argument("--view", required=True)
    decode.add_argument("--external-python", type=Path, default=None)
    decode.add_argument("--model", type=Path, default=None)
    decode.add_argument("--device", default="cuda:0")
    decode.add_argument("--no-accept", action="store_true")

    accept = commands.add_parser("accept", help="re-run acceptance on a decoded view (CPU)")
    accept.add_argument("--view", required=True)

    args = parser.parse_args()
    root = args.repository_root.resolve()
    output_root = root / args.output_root
    if args.command == "plan":
        planner = SeedTransferPlanner(root, config_path=args.config)
        for target in TARGETS:
            errors = planner.reprojection_px[target]
            point = planner.centroids_world[target]
            height = float(planner.plane.signed_distance(point.reshape(1, 3))[0])
            print(
                f"{target}: centroid {np.round(point, 1).tolist()} mm, "
                f"radius {planner.radius_mm[target]:.0f} mm, height above table {height:.0f} mm, "
                "reprojection " + ", ".join(f"{v} {e:.1f} px" for v, e in errors.items())
            )
        for view in args.view:
            manifest = planner.plan_view(view, config_path=args.config)
            path = output_root / view / "seed_manifest.json"
            write_manifest(path, manifest)
            print(
                f"{view}: {sum(bool(p.prompts) for p in manifest.parts)} parts and "
                f"{sum(bool(h.prompts) for h in manifest.hands)} hands planned -> {path}"
            )
    elif args.command == "decode":
        path = output_root / args.view / "seed_manifest.json"
        result = decode_view(
            path,
            repository_root=root,
            external_python=args.external_python,
            model=args.model,
            device=args.device,
        )
        print(f"decoded -> {result}")
        if not args.no_accept:
            print(summary_line(accept_view(path, repository_root=root)))
    else:
        path = output_root / args.view / "seed_manifest.json"
        print(summary_line(accept_view(path, repository_root=root)))


if __name__ == "__main__":
    main()
