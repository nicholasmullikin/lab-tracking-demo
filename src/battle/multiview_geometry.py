"""One calibrated rig for every Assembly101 view of the pinned recording.

`CameraRig` loads the per-view camera estimates (`configs/assembly101/*_camera_estimate.json`),
the measured clock rules (`configs/assembly101/clock_rules.json`) and, for the head-mounted
cameras, the dataset's per-frame `camera_extrinsics_ego`.  Time is expressed on the dataset's
60 fps *pose clock*: static cameras have one constant pose, ego cameras one pose per pose
frame, and each view's clock rule maps its own 30 fps analysis frames onto that clock.

Operations: project world millimetres into a view's raw sensor pixels (with distortion), cast
the world ray of a pixel, triangulate one point from two or more views (DLT on undistorted
normalised coordinates, then iteratively drop the view with the worst reprojection error while
it exceeds a threshold, the ATHENA scheme), the epipolar distance between two observations,
and a table-plane fit from the dataset's lowest fingertips with "down" derived from the static
camera poses rather than assumed.

Everything here is geometry on dataset context and fitted estimates.  Triangulated points are
cross-view agreement, never accuracy; CC BY-NC 4.0 attribution applies to the dataset assets.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from pydantic import Field

from .assembly101_camera_fit import CONFIG_ROOT, PoseMembers, camera_estimate_path
from .assembly101_clock_offset import (
    CLOCK_RULES_CONFIG,
    POSES_ROOT,
    SHIPPED_2D_WINDOW,
    WINDOW_FRAME_COUNT,
    WINDOW_START_POSE_FRAME,
    Assembly101ClockRuleSet,
    is_ego,
    load_clock_rules,
    npz_key,
    view_key,
)
from .assembly101_fetch_view import EGO_VIEWS, RECORDING_ID, STATIC_VIEWS
from .assembly101_pose_schemas import Assembly101CameraModel, Assembly101ClockRule
from .schemas import VersionedModel

ALL_VIEWS: tuple[str, ...] = (*STATIC_VIEWS, *EGO_VIEWS)
FINGERTIP_JOINTS: tuple[int, ...] = (0, 1, 2, 3, 4)
DEFAULT_REPROJECTION_FILTER_PX = 30.0
CHECK_OUTPUT_ROOT = Path("runs/assembly101-multiview-rig-check")


def _as_matrix(rows: Iterable[Iterable[float]]) -> np.ndarray:
    return np.asarray([list(row) for row in rows], dtype=np.float64)


@dataclass(frozen=True)
class Ray:
    """A world-frame ray: `origin + t * direction`, direction unit length, millimetres."""

    origin: np.ndarray
    direction: np.ndarray

    def point_at(self, t: float) -> np.ndarray:
        return self.origin + t * self.direction


@dataclass(frozen=True)
class TriangulationResult:
    """Points (N, 3) world mm (NaN where fewer than two views survived) and bookkeeping."""

    points: np.ndarray
    views: tuple[str, ...]
    used: np.ndarray  # (N, V) bool: view contributed to the final estimate
    reprojection_px: np.ndarray  # (N, V) pixel error of the final point in each input view (NaN)

    @property
    def view_counts(self) -> np.ndarray:
        return self.used.sum(axis=1)


class TablePlane(VersionedModel):
    """Plane through the resting fingertips: `normal . x = offset`, normal points up."""

    normal: tuple[float, float, float]
    offset_mm: float
    point_on_plane_mm: tuple[float, float, float]
    down_vector: tuple[float, float, float]
    down_vector_source: str = Field(min_length=1)
    down_vector_alternatives_deg: dict[str, float]
    normal_vs_down_deg: float
    fingertip_count: int = Field(ge=1)
    fitted_point_count: int = Field(ge=3)
    lowest_fraction: float = Field(gt=0, le=1)
    residual_rms_mm: float = Field(ge=0)
    residual_p95_mm: float = Field(ge=0)
    fingertips_below_plane_by_20mm_fraction: float = Field(ge=0, le=1)
    claim_boundary: str = Field(min_length=1)

    def signed_distance(self, points_mm: np.ndarray) -> np.ndarray:
        return np.asarray(points_mm, dtype=np.float64) @ np.asarray(self.normal) - self.offset_mm

    def intersect_ray(self, ray: Ray) -> np.ndarray | None:
        normal = np.asarray(self.normal)
        denominator = float(normal @ ray.direction)
        if abs(denominator) < 1e-9:
            return None
        t = (self.offset_mm - float(normal @ ray.origin)) / denominator
        return None if t <= 0 else ray.point_at(t)


def dlt_triangulate(points_2d: np.ndarray, projections: np.ndarray) -> np.ndarray:
    """DLT for a batch: `points_2d` (V, N, 2) normalised coords with NaN gaps, `projections`
    (V, 3, 4).  Returns (N, 3); NaN where fewer than two views observe the point.  Equal
    weight per view, as in ATHENA's `triangulaterefine`.
    """
    view_count, point_count, _ = points_2d.shape
    result = np.full((point_count, 3), np.nan)
    good = ~np.isnan(points_2d[:, :, 0])
    patterns = np.zeros(point_count, dtype=np.int64)
    for view in range(view_count):
        patterns |= good[view].astype(np.int64) << view
    for pattern in np.unique(patterns):
        active = [view for view in range(view_count) if (pattern >> view) & 1]
        if len(active) < 2:
            continue
        indices = np.where(patterns == pattern)[0]
        constraint = np.zeros((len(indices), 2 * len(active), 4))
        for slot, view in enumerate(active):
            x = points_2d[view, indices, 0]
            y = points_2d[view, indices, 1]
            matrix = projections[view]
            constraint[:, 2 * slot] = x[:, None] * matrix[2][None, :] - matrix[0][None, :]
            constraint[:, 2 * slot + 1] = y[:, None] * matrix[2][None, :] - matrix[1][None, :]
        _, _, vh = np.linalg.svd(constraint, full_matrices=True)
        homogeneous = vh[:, -1, :]
        result[indices] = homogeneous[:, :3] / homogeneous[:, 3:4]
    return result


class CameraRig:
    def __init__(
        self,
        cameras: Mapping[str, Assembly101CameraModel],
        clock_rules: Mapping[str, Assembly101ClockRule],
        ego_poses: Mapping[int, Mapping[str, np.ndarray]] | None = None,
    ) -> None:
        self.cameras: dict[str, Assembly101CameraModel] = dict(cameras)
        self.clock_rules: dict[str, Assembly101ClockRule] = dict(clock_rules)
        self.ego_poses: dict[int, dict[str, np.ndarray]] = {
            int(frame): {key: np.asarray(pose, dtype=np.float64) for key, pose in poses.items()}
            for frame, poses in (ego_poses or {}).items()
        }
        for view, camera in self.cameras.items():
            if camera.view_key != view_key(view):
                raise ValueError(f"camera file for {view} carries view_key {camera.view_key}")
        self._intrinsics = {
            view: _as_matrix(camera.intrinsic_matrix) for view, camera in self.cameras.items()
        }
        self._distortion = {
            view: np.asarray(camera.distortion, dtype=np.float64)
            for view, camera in self.cameras.items()
        }

    # -- loading ------------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        repository_root: Path,
        *,
        views: Iterable[str] = ALL_VIEWS,
        config_root: Path = CONFIG_ROOT,
        clock_rules_path: Path = CLOCK_RULES_CONFIG,
        pose_frames: range | None = range(
            WINDOW_START_POSE_FRAME, WINDOW_START_POSE_FRAME + WINDOW_FRAME_COUNT
        ),
    ) -> CameraRig:
        """Load the tracked estimates; ego poses are read for `pose_frames` only (None = all)."""
        repository_root = repository_root.resolve()
        views = tuple(views)
        cameras = {
            view: Assembly101CameraModel.model_validate_json(
                (repository_root / camera_estimate_path(view, config_root)).read_text(
                    encoding="utf-8"
                )
            )
            for view in views
        }
        rules: Assembly101ClockRuleSet = load_clock_rules(repository_root, clock_rules_path)
        clock_rules = {
            view: rules.views[view].clock_rule
            for view in views
            if view in rules.views and rules.views[view].clock_rule is not None
        }
        ego_poses: dict[int, dict[str, np.ndarray]] = {}
        if any(is_ego(view) for view in views):
            raw = json.loads(
                (
                    repository_root / POSES_ROOT / "camera_extrinsics_ego" / f"{RECORDING_ID}.json"
                ).read_text(encoding="utf-8")
            )
            keys = pose_frames if pose_frames is not None else [int(k) for k in raw]
            wanted = {view_key(view) for view in views if is_ego(view)}
            for frame in keys:
                entry = raw.get(str(frame))
                if entry is None:
                    continue
                ego_poses[int(frame)] = {
                    key: np.asarray(pose, dtype=np.float64)
                    for key, pose in entry.items()
                    if key in wanted
                }
        return cls(cameras, clock_rules, ego_poses)

    # -- bookkeeping --------------------------------------------------------------------

    @property
    def views(self) -> tuple[str, ...]:
        return tuple(self.cameras)

    @property
    def static_views(self) -> tuple[str, ...]:
        return tuple(view for view in self.cameras if not self.cameras[view].is_ego)

    @property
    def ego_views(self) -> tuple[str, ...]:
        return tuple(view for view in self.cameras if self.cameras[view].is_ego)

    def camera(self, view: str) -> Assembly101CameraModel:
        try:
            return self.cameras[view]
        except KeyError as error:
            raise KeyError(f"view {view} is not in the rig ({', '.join(self.views)})") from error

    def clock_rule(self, view: str) -> Assembly101ClockRule:
        rule = self.clock_rules.get(view)
        if rule is None:
            raise KeyError(f"view {view} has no measured clock rule (ambiguous or unscanned)")
        return rule

    def pose_frame(self, view: str, analysis_frame: int) -> int:
        """Pose-clock frame shown by `view`'s analysis (proxy) frame."""
        return self.clock_rule(view).pose_frame(analysis_frame)

    def image_size(self, view: str) -> tuple[int, int]:
        return tuple(self.camera(view).raw_image_size)  # type: ignore[return-value]

    # -- poses --------------------------------------------------------------------------

    def camera_to_world(self, view: str, pose_frame: int | None = None) -> np.ndarray:
        camera = self.camera(view)
        if not camera.is_ego:
            assert camera.camera_to_world is not None
            return _as_matrix(camera.camera_to_world)
        if pose_frame is None:
            raise ValueError(f"{view} is an ego camera; a pose_frame is required")
        try:
            return self.ego_poses[int(pose_frame)][camera.view_key]
        except KeyError as error:
            raise KeyError(f"no ego pose for {view} at pose frame {pose_frame}") from error

    def world_to_camera(self, view: str, pose_frame: int | None = None) -> np.ndarray:
        return np.linalg.inv(self.camera_to_world(view, pose_frame))

    def camera_position(self, view: str, pose_frame: int | None = None) -> np.ndarray:
        return self.camera_to_world(view, pose_frame)[:3, 3].copy()

    def projection_matrix(self, view: str, pose_frame: int | None = None) -> np.ndarray:
        """Normalised (K = I) 3x4 projection `[R | t]` from world to camera."""
        return self.world_to_camera(view, pose_frame)[:3, :]

    # -- projection and rays ------------------------------------------------------------

    def project(
        self, view: str, points_world_mm: np.ndarray, pose_frame: int | None = None
    ) -> np.ndarray:
        """World mm (N, 3) -> raw sensor pixels (N, 2) of `view`, distortion applied."""
        points = np.asarray(points_world_mm, dtype=np.float64).reshape(-1, 3)
        if points.size == 0:
            return np.zeros((0, 2))
        extrinsic = self.world_to_camera(view, pose_frame)
        rotation, _ = cv2.Rodrigues(extrinsic[:3, :3])
        projected, _ = cv2.projectPoints(
            points.reshape(-1, 1, 3),
            rotation,
            extrinsic[:3, 3],
            self._intrinsics[view],
            self._distortion[view],
        )
        return projected.reshape(-1, 2)

    def depth(
        self, view: str, points_world_mm: np.ndarray, pose_frame: int | None = None
    ) -> np.ndarray:
        points = np.asarray(points_world_mm, dtype=np.float64).reshape(-1, 3)
        extrinsic = self.world_to_camera(view, pose_frame)
        return (extrinsic[:3, :3] @ points.T + extrinsic[:3, 3:4])[2]

    def undistort(self, view: str, pixels: np.ndarray) -> np.ndarray:
        """Raw pixels (N, 2) -> normalised image coordinates (N, 2); NaN passes through."""
        pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
        result = np.full_like(pixels, np.nan)
        valid = ~np.isnan(pixels).any(axis=1)
        if valid.any():
            normalised = cv2.undistortPoints(
                pixels[valid].reshape(-1, 1, 2),
                self._intrinsics[view],
                self._distortion[view],
            )
            result[valid] = normalised.reshape(-1, 2)
        return result

    def ray(self, view: str, pixel: np.ndarray, pose_frame: int | None = None) -> Ray:
        """World ray through one raw pixel of `view`."""
        normalised = self.undistort(view, np.asarray(pixel, dtype=np.float64).reshape(1, 2))[0]
        direction_camera = np.array([normalised[0], normalised[1], 1.0])
        pose = self.camera_to_world(view, pose_frame)
        direction = pose[:3, :3] @ direction_camera
        return Ray(origin=pose[:3, 3].copy(), direction=direction / np.linalg.norm(direction))

    # -- multi-view ---------------------------------------------------------------------

    def _pose_for(self, view: str, pose_frame: int | Mapping[str, int] | None) -> int | None:
        if isinstance(pose_frame, Mapping):
            return pose_frame.get(view)
        return pose_frame

    def triangulate(
        self,
        points_by_view: Mapping[str, np.ndarray],
        pose_frame: int | Mapping[str, int] | None = None,
        *,
        reproj_filter_px: float | None = DEFAULT_REPROJECTION_FILTER_PX,
        min_views: int = 2,
    ) -> TriangulationResult:
        """Triangulate N points seen in several views.

        `points_by_view[view]` is (N, 2) raw pixels with NaN for unobserved points.  After the
        equal-weight DLT, while a point's largest reprojection error exceeds `reproj_filter_px`
        and more than `min_views` views remain, one view is dropped: the one whose removal
        leaves the smallest worst-case error among the survivors (leave-one-out, which is
        more robust than dropping the largest residual when the compromise solution spreads a
        gross error over the good views).  A point left with fewer than `min_views` views is
        NaN.  `reprojection_px` reports the final point's error in every input view, so a
        two-view point that still disagrees is visible to the caller.
        """
        views = tuple(points_by_view)
        if not views:
            raise ValueError("triangulate needs at least one view")
        count = next(iter(points_by_view.values())).shape[0]
        observed = np.stack(
            [np.asarray(points_by_view[view], dtype=np.float64).reshape(count, 2) for view in views]
        )
        normalised = np.stack(
            [self.undistort(view, observed[index]) for index, view in enumerate(views)]
        )
        poses = [self._pose_for(view, pose_frame) for view in views]
        projections = np.stack(
            [self.projection_matrix(view, pose) for view, pose in zip(views, poses, strict=True)]
        )

        def solve(rows: np.ndarray, active: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            masked = normalised[:, rows].copy()
            masked[~active[rows].T] = np.nan
            estimate = dlt_triangulate(masked, projections)
            errs = np.full((rows.size, len(views)), np.nan)
            valid = ~np.isnan(estimate[:, 0])
            for index, view in enumerate(views):
                if not valid.any():
                    break
                reprojected = self.project(view, estimate[valid], poses[index])
                errs[valid, index] = np.linalg.norm(
                    reprojected - observed[index, rows[valid]], axis=1
                )
            return estimate, errs

        active = ~np.isnan(observed[:, :, 0]).T  # (N, V)
        rows = np.arange(count)
        points, errors = solve(rows, active)
        if reproj_filter_px is not None:
            for row in rows:
                while active[row].sum() > min_views:
                    row_errors = np.where(active[row], errors[row], -np.inf)
                    if row_errors.max() <= reproj_filter_px:
                        break
                    best: tuple[float, int, np.ndarray, np.ndarray] | None = None
                    for candidate in np.where(active[row])[0]:
                        trial = active.copy()
                        trial[row, candidate] = False
                        estimate, errs = solve(np.array([row]), trial)
                        worst = float(np.nanmax(np.where(trial[row], errs[0], -np.inf)))
                        if best is None or worst < best[0]:
                            best = (worst, int(candidate), estimate[0], errs[0])
                    assert best is not None
                    active[row, best[1]] = False
                    points[row] = best[2]
                    errors[row] = best[3]
        insufficient = active.sum(axis=1) < min_views
        points[insufficient] = np.nan
        active[insufficient] = False
        return TriangulationResult(points=points, views=views, used=active, reprojection_px=errors)

    def epipolar_distance(
        self,
        view_a: str,
        pixel_a: np.ndarray,
        view_b: str,
        pixel_b: np.ndarray,
        pose_frame: int | Mapping[str, int] | None = None,
    ) -> float:
        """Distance (px, undistorted image of `view_b`) from `pixel_b` to the epipolar line of
        `pixel_a`."""
        ray_a = self.ray(view_a, pixel_a, self._pose_for(view_a, pose_frame))
        pose_b = self._pose_for(view_b, pose_frame)
        extrinsic_b = self.world_to_camera(view_b, pose_b)
        intrinsic_b = self._intrinsics[view_b]

        def to_undistorted_pixels(point_world: np.ndarray) -> np.ndarray:
            camera = extrinsic_b[:3, :3] @ point_world + extrinsic_b[:3, 3]
            homogeneous = intrinsic_b @ camera
            return homogeneous

        p0 = to_undistorted_pixels(ray_a.origin)
        p1 = to_undistorted_pixels(ray_a.point_at(1000.0))
        line = np.cross(p0, p1)
        normalised_b = self.undistort(view_b, np.asarray(pixel_b, dtype=np.float64).reshape(1, 2))[
            0
        ]
        pixel_b_undistorted = intrinsic_b @ np.array([normalised_b[0], normalised_b[1], 1.0])
        norm = np.hypot(line[0], line[1])
        if norm == 0:
            return float("nan")
        return float(abs(line @ pixel_b_undistorted) / norm)

    # -- table plane --------------------------------------------------------------------

    def down_vector_candidates(self) -> dict[str, np.ndarray]:
        """Two extrinsics-derived 'down' estimates: mean static camera y-axis (OpenCV image
        down) and the direction from the mean static camera position toward the scene."""
        axes = np.stack([self.camera_to_world(view)[:3, 1] for view in self.static_views])
        mean_axis = axes.mean(axis=0)
        return {"mean_static_camera_y_axis": mean_axis / np.linalg.norm(mean_axis)}

    def fit_table_plane(
        self,
        landmarks3d: Mapping[str, Mapping[str, list[list[float]]]],
        confidences: Mapping[str, Mapping[str, float]],
        *,
        pose_frames: Iterable[int],
        lowest_fraction: float = 0.05,
        confidence_floor: float = 0.8,
        fingertips: tuple[int, ...] = FINGERTIP_JOINTS,
    ) -> TablePlane:
        """Fit a plane to the lowest `lowest_fraction` of dataset fingertips over the frames.

        "Down" is not an assumed axis: it is the mean image-down (y) axis of the static
        cameras, cross-checked against the direction from the mean static camera position to
        the fingertip centroid (cameras look down at the table).
        """
        tips: list[np.ndarray] = []
        for frame in pose_frames:
            key = str(frame)
            frame_landmarks = landmarks3d.get(key)
            if frame_landmarks is None:
                continue
            for hand in ("0", "1"):
                if float(confidences[key][hand]) < confidence_floor:
                    continue
                tips.append(np.asarray(frame_landmarks[hand], dtype=np.float64)[list(fingertips)])
        if not tips:
            raise ValueError("no fingertips pass the confidence floor")
        points = np.concatenate(tips)
        candidates = self.down_vector_candidates()
        down = candidates["mean_static_camera_y_axis"]
        camera_mean = np.mean([self.camera_position(view) for view in self.static_views], axis=0)
        toward_scene = points.mean(axis=0) - camera_mean
        toward_scene /= np.linalg.norm(toward_scene)
        alternatives = {
            "mean_static_camera_position_to_fingertip_centroid": float(
                np.degrees(np.arccos(np.clip(down @ toward_scene, -1.0, 1.0)))
            )
        }
        heights = points @ down
        cutoff = np.quantile(heights, 1.0 - lowest_fraction)
        lowest = points[heights >= cutoff]
        centroid = lowest.mean(axis=0)
        _, _, vh = np.linalg.svd(lowest - centroid, full_matrices=False)
        normal = vh[-1]
        if normal @ down > 0:
            normal = -normal
        offset = float(normal @ centroid)
        residual = (lowest - centroid) @ normal
        signed_all = points @ normal - offset
        return TablePlane(
            normal=tuple(float(v) for v in normal),
            offset_mm=offset,
            point_on_plane_mm=tuple(float(v) for v in centroid),
            down_vector=tuple(float(v) for v in down),
            down_vector_source="mean_static_camera_y_axis",
            down_vector_alternatives_deg=alternatives,
            normal_vs_down_deg=float(np.degrees(np.arccos(np.clip(-(normal @ down), -1.0, 1.0)))),
            fingertip_count=int(points.shape[0]),
            fitted_point_count=int(lowest.shape[0]),
            lowest_fraction=lowest_fraction,
            residual_rms_mm=float(np.sqrt(np.mean(residual**2))),
            residual_p95_mm=float(np.quantile(np.abs(residual), 0.95)),
            fingertips_below_plane_by_20mm_fraction=float(np.mean(signed_all < -20.0)),
            claim_boundary=(
                "Plane through the lowest dataset fingertips; a geometric convenience for seed "
                "transfer between views, not a measured table and not an accuracy statement."
            ),
        )


# -- real-data check ----------------------------------------------------------------------


class ViewProjectionCheck(VersionedModel):
    view: str
    frames: int = Field(ge=0)
    points: int = Field(ge=0)
    rms_pixels: float = Field(ge=0)
    max_pixels: float = Field(ge=0)


class TriangulationCheck(VersionedModel):
    views: tuple[str, ...]
    frames: int = Field(ge=0)
    points: int = Field(ge=0)
    median_error_mm: float = Field(ge=0)
    p95_error_mm: float = Field(ge=0)
    max_error_mm: float = Field(ge=0)
    mean_views_used: float = Field(ge=0)


class RigCheckReport(VersionedModel):
    manifest_kind: str = "assembly101_multiview_rig_check"
    recording_id: str
    views: tuple[str, ...]
    projection: tuple[ViewProjectionCheck, ...]
    triangulation: tuple[TriangulationCheck, ...]
    table_plane: TablePlane
    runtime_seconds: float = Field(ge=0)
    claim_boundaries: tuple[str, ...]


def run_rig_check(
    repository_root: Path,
    *,
    frame_step: int = 25,
    output_root: Path = CHECK_OUTPUT_ROOT,
) -> RigCheckReport:
    """Project dataset 3D into every view, triangulate its 2D back, fit the table plane."""
    started = time.monotonic()
    repository_root = repository_root.resolve()
    rig = CameraRig.load(repository_root)
    members = PoseMembers(repository_root)
    with np.load(repository_root / SHIPPED_2D_WINDOW) as archive:
        landmarks2d = {
            view: np.asarray(archive[npz_key(view)], dtype=np.float64) for view in rig.views
        }
    frames = list(
        range(WINDOW_START_POSE_FRAME, WINDOW_START_POSE_FRAME + WINDOW_FRAME_COUNT, frame_step)
    )

    def hands(frame: int) -> list[tuple[int, np.ndarray]]:
        key = str(frame)
        return [
            (hand, np.asarray(members.landmarks3d[key][str(hand)], dtype=np.float64))
            for hand in (0, 1)
            if float(members.confidences[key][str(hand)]) >= 0.8
        ]

    projection_checks: list[ViewProjectionCheck] = []
    for view in rig.views:
        width, height = rig.image_size(view)
        residuals: list[np.ndarray] = []
        used_frames = 0
        for frame in frames:
            present = hands(frame)
            if not present:
                continue
            used_frames += 1
            for hand, world in present:
                shipped = landmarks2d[view][frame - WINDOW_START_POSE_FRAME, hand]
                inside = (
                    (shipped[:, 0] >= 0)
                    & (shipped[:, 0] < width)
                    & (shipped[:, 1] >= 0)
                    & (shipped[:, 1] < height)
                )
                if not inside.any():
                    continue
                ours = rig.project(view, world[inside], frame)
                residuals.append(np.linalg.norm(ours - shipped[inside], axis=1))
        stacked = np.concatenate(residuals) if residuals else np.zeros(0)
        projection_checks.append(
            ViewProjectionCheck(
                view=view,
                frames=used_frames,
                points=int(stacked.size),
                rms_pixels=float(np.sqrt(np.mean(stacked**2))) if stacked.size else 0.0,
                max_pixels=float(stacked.max()) if stacked.size else 0.0,
            )
        )

    def triangulation_check(views: tuple[str, ...]) -> TriangulationCheck:
        errors: list[np.ndarray] = []
        counts: list[np.ndarray] = []
        used_frames = 0
        for frame in frames:
            present = hands(frame)
            if not present:
                continue
            used_frames += 1
            world = np.concatenate([w for _, w in present])
            observations: dict[str, np.ndarray] = {}
            for view in views:
                width, height = rig.image_size(view)
                pixels = np.concatenate(
                    [
                        landmarks2d[view][frame - WINDOW_START_POSE_FRAME, hand]
                        for hand, _ in present
                    ]
                ).copy()
                outside = (
                    (pixels[:, 0] < 0)
                    | (pixels[:, 0] >= width)
                    | (pixels[:, 1] < 0)
                    | (pixels[:, 1] >= height)
                )
                pixels[outside] = np.nan
                observations[view] = pixels
            result = rig.triangulate(observations, frame)
            valid = ~np.isnan(result.points[:, 0])
            errors.append(np.linalg.norm(result.points[valid] - world[valid], axis=1))
            counts.append(result.view_counts[valid])
        stacked = np.concatenate(errors)
        return TriangulationCheck(
            views=views,
            frames=used_frames,
            points=int(stacked.size),
            median_error_mm=float(np.median(stacked)),
            p95_error_mm=float(np.quantile(stacked, 0.95)),
            max_error_mm=float(stacked.max()),
            mean_views_used=float(np.concatenate(counts).mean()),
        )

    triangulation_checks = [
        triangulation_check(rig.static_views),
        triangulation_check(("C10379", "C10395")),
        triangulation_check(("C10115", "C10404")),
        triangulation_check((*rig.static_views, "HMC_21110305")),
    ]
    plane = rig.fit_table_plane(
        members.landmarks3d,
        members.confidences,
        pose_frames=range(WINDOW_START_POSE_FRAME, WINDOW_START_POSE_FRAME + WINDOW_FRAME_COUNT),
    )
    report = RigCheckReport(
        recording_id=RECORDING_ID,
        views=rig.views,
        projection=tuple(projection_checks),
        triangulation=tuple(triangulation_checks),
        table_plane=plane,
        runtime_seconds=time.monotonic() - started,
        claim_boundaries=(
            "Projection and triangulation residuals compare the rig against the dataset's own "
            "landmarks, which were generated by the same projection; they verify the rig "
            "reproduces the dataset's geometry, not that either is accurate.",
            "Dataset poses/extrinsics are external context; intrinsics are estimates; "
            "CC BY-NC 4.0 attribution applies.",
        ),
    )
    root = repository_root / output_root
    root.mkdir(parents=True, exist_ok=True)
    (root / "report.json").write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--frame-step", type=int, default=25)
    parser.add_argument("--output-root", type=Path, default=CHECK_OUTPUT_ROOT)
    args = parser.parse_args()
    report = run_rig_check(
        args.repository_root, frame_step=args.frame_step, output_root=args.output_root
    )
    for check in report.projection:
        print(
            f"{check.view:13s} project 3D->2D vs shipped: rms {check.rms_pixels:.5f} px, "
            f"max {check.max_pixels:.4f} px over {check.points} points"
        )
    for check in report.triangulation:
        print(
            f"triangulate {'+'.join(check.views)}: median {check.median_error_mm:.4f} mm, "
            f"p95 {check.p95_error_mm:.4f} mm, max {check.max_error_mm:.3f} mm over "
            f"{check.points} points, mean views {check.mean_views_used:.2f}"
        )
    plane = report.table_plane
    print(
        f"table plane: normal {np.round(plane.normal, 4).tolist()} offset "
        f"{plane.offset_mm:.1f} mm, residual rms {plane.residual_rms_mm:.2f} mm "
        f"(p95 {plane.residual_p95_mm:.2f}) over {plane.fitted_point_count}/"
        f"{plane.fingertip_count} fingertips; normal vs down {plane.normal_vs_down_deg:.1f} deg; "
        f"down alternatives {plane.down_vector_alternatives_deg}"
    )
    print(f"{report.runtime_seconds:.1f} s")


if __name__ == "__main__":
    main()
