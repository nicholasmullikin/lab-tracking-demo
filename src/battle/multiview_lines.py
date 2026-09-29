"""Pipettes as 3D lines: a line fit from per-view mask axes (`p1-line-geometry`, Sep 28).

The point tracker reduces every view to one pixel and triangulates; for a 30 cm pipette the
views' pixels are different places on the shaft and the residual blows the gate. Here a
mask's principal axis in one view back-projects to a **plane** through that camera's centre
(`plane_from_axis`), and a compact mask (a camera looking along the shaft: T5 over a vertical
pipette, the head camera down the barrel) back-projects to a **ray** through its centroid
(`ray_from_point`). `fit_line` takes the direction as the null space of the stacked plane
normals (SVD; each plane weighted by its observation weight times the RMS sine of its angle
to the other planes, so a plane nearly parallel to the rest counts less), then the point as
the least-squares solution in the plane perpendicular to the direction, where each plane is
one row (its perpendicular offset in cm) and each ray two rows with a free depth (a ray along
the line pins both perpendicular coordinates; a ray across it pins one). With one plane and
two or more rays the rays are intersected with the plane and the line fitted through the
intersections; with rays only there is no line to fit and None is returned (the point
tracker's job). The fit is None when the best pair of planes meets under `min_pair_angle_deg`.

`line_endpoints` puts the visible extent on the line (each axis view's endpoint rays give
their closest points on the line; robust extremes over views) and completes it to a length
prior from the better-supported end, flagging an extent over 1.2x the prior as two masks
merged. `loo_residual` is the label-free score of the plan (every axis view against the line
from the others, in px and degrees); `centroid_point` / `centroid_loo_residual` run today's
box-centre method through `triangulate_pixels` for the same views so the scoreboard can
compare. `line_distance` and `line_cost` are the association gate for the tracker extension.

Conventions, exactly those of `finebio_cameras` / `finebio_slice` / `multiview_tracks`: world
in board centimetres with z into the bench (height is ``-z``); pixels are raw video pixels
and are undistorted first (`Camera.undistort`, P=K), then ``K^-1`` gives the camera ray and
``R^T`` the world direction from `Camera.centre`, as `multiview_tracks.ray_point` does. Pixel
residuals here are measured in the undistorted image (the space `triangulate_pixels` solves
in; within about 1% of raw pixels on these GoPro linear-mode intrinsics); `reproject_line`
returns raw pixels for drawing over frames. A plane is ``(normal, d)`` with
``normal @ X == d`` for X on it, normal unit. Directions and endpoint order carry no sign:
which end is the tip is the tracker's decision (colour ring, hand coupling), not geometry's.
Inputs are `AxisObs` records so this module does not import the observation schema; a thin
adapter from `FineBioObservation` (``mask_axis_px``, ``mask_elongation``, ``point_px``) is
the caller's.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .finebio_cameras import Camera
from .finebio_slice import triangulate_pixels

# A mask with elongation (sqrt of the covariance eigenvalue ratio) at or above this is a
# plane constraint; below it the mask is compact and contributes its centroid's ray.
ELONGATION_THRESHOLD = 2.5
# The fit is degenerate when no two planes meet at this angle (and a ray meeting the single
# plane under it is dropped).
MIN_PAIR_ANGLE_DEG = 5.0
# An axis shorter than this in the image is compact whatever its elongation says.
MIN_AXIS_LENGTH_PX = 4.0
# A visible extent longer than this factor times the length prior is two masks merged.
MERGED_FACTOR = 1.2
# A view supports an end of the extent when its own extent reaches within this of it.
END_TOLERANCE_CM = 2.0
# One plane plus rays: the ray-plane intersections must span at least this to give a direction.
MIN_RAY_SEPARATION_CM = 1.0
# Half-length drawn for a line without endpoints, and the lever arm turning degrees into cm
# in `line_cost` (half a pipette).
DEFAULT_HALF_LENGTH_CM = 15.0


# --------------------------------------------------------------------------- records


@dataclass(frozen=True)
class AxisObs:
    """One view's evidence for a line: the mask axis endpoints in raw pixels (order
    unresolved; None when the mask has no axis), the centroid or box centre in raw pixels, the
    elongation (1.0 round; None means unknown, and endpoints are then trusted) and a weight
    that scales the view's constraint rows."""

    view: str
    endpoints_px: np.ndarray | None
    centroid_px: np.ndarray
    elongation: float | None = None
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.endpoints_px is not None:
            ends = np.asarray(self.endpoints_px, dtype=np.float64).reshape(2, 2)
            object.__setattr__(self, "endpoints_px", ends)
        object.__setattr__(
            self, "centroid_px", np.asarray(self.centroid_px, dtype=np.float64).reshape(2)
        )

    @property
    def axis_midpoint_px(self) -> np.ndarray:
        return self.centroid_px if self.endpoints_px is None else self.endpoints_px.mean(axis=0)

    @property
    def axis_length_px(self) -> float:
        if self.endpoints_px is None:
            return 0.0
        return float(np.linalg.norm(self.endpoints_px[1] - self.endpoints_px[0]))


@dataclass(frozen=True)
class ViewResidual:
    """One view against a line, in undistorted pixels: for an axis view (``kind == "plane"``)
    the mean distance of its endpoints to the projected line and the angle between axis and
    projected line; for a compact view (``kind == "ray"``) the centroid's distance, no angle."""

    kind: str
    perpendicular_px: float
    angle_deg: float | None


@dataclass(frozen=True)
class Line3D:
    """A fitted line: `point` (cm, the midpoint of the visible extent when known), unit
    `direction` (sign arbitrary), optional `endpoints` (2x3, in increasing order along
    `direction`), the views that contributed as planes and as rays, the in-sample residual
    per view, and `condition` in [0, 1] (1 ideal: the ratio of the two non-null singular
    values of the weighted normals, about tan(angle/2) for two planes; for one plane plus
    rays the sine of the smallest ray-plane angle used)."""

    point: np.ndarray
    direction: np.ndarray
    endpoints: np.ndarray | None = None
    support_views: tuple[str, ...] = ()
    plane_views: tuple[str, ...] = ()
    ray_views: tuple[str, ...] = ()
    residuals: dict[str, ViewResidual] = field(default_factory=dict)
    condition: float = 0.0

    def __post_init__(self) -> None:
        direction = np.asarray(self.direction, dtype=np.float64).reshape(3)
        object.__setattr__(self, "direction", direction / np.linalg.norm(direction))
        object.__setattr__(self, "point", np.asarray(self.point, dtype=np.float64).reshape(3))
        if self.endpoints is not None:
            ends = np.asarray(self.endpoints, dtype=np.float64).reshape(2, 3)
            object.__setattr__(self, "endpoints", ends)

    @property
    def midpoint(self) -> np.ndarray:
        return self.point if self.endpoints is None else self.endpoints.mean(axis=0)

    @property
    def length_cm(self) -> float | None:
        if self.endpoints is None:
            return None
        return float(np.linalg.norm(self.endpoints[1] - self.endpoints[0]))

    def point_at(self, t: float) -> np.ndarray:
        return self.point + t * self.direction

    def parameter(self, x: np.ndarray) -> float:
        """Signed distance along `direction` from `point` to the foot of `x`."""
        return float((np.asarray(x, dtype=np.float64).reshape(3) - self.point) @ self.direction)

    def closest_point(self, x: np.ndarray) -> np.ndarray:
        return self.point_at(self.parameter(x))

    def parameter_closest_to_ray(self, origin: np.ndarray, direction: np.ndarray) -> float:
        """Parameter of the point on this line closest to the (two-sided) ray; when the ray is
        parallel to the line, the foot of the ray origin."""
        r = np.asarray(direction, dtype=np.float64).reshape(3)
        r = r / np.linalg.norm(r)
        w = self.point - np.asarray(origin, dtype=np.float64).reshape(3)
        b = float(self.direction @ r)
        d, e = float(self.direction @ w), float(r @ w)
        denominator = 1.0 - b * b
        if denominator < 1e-12:
            return -d
        return (b * e - d) / denominator


@dataclass(frozen=True)
class Extent:
    """`line_endpoints` result: `endpoints` after the length prior (2x3, increasing along the
    line's direction), the `visible` extremes before it, the counts of views reaching each
    end, and the flags: `extended` (the prior reconstructed more than the tolerance of a
    truncated extent; `extended_end` is the index of the end that moved, None when both did)
    and `merged` (the visible extent exceeds `MERGED_FACTOR` times the prior: two masks in
    one)."""

    endpoints: np.ndarray
    visible: np.ndarray
    support: tuple[int, int]
    per_view: dict[str, tuple[float, float]]
    extended: bool = False
    extended_end: int | None = None
    merged: bool = False

    @property
    def length_cm(self) -> float:
        return float(np.linalg.norm(self.endpoints[1] - self.endpoints[0]))

    @property
    def visible_length_cm(self) -> float:
        return float(np.linalg.norm(self.visible[1] - self.visible[0]))


CamObs = tuple[Camera, AxisObs]


# --------------------------------------------------------------------------- back-projection


def ray_from_point(cam: Camera, pixel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World ray ``(origin, unit direction)`` through a raw pixel: undistort, ``K^-1``, ``R^T``
    (the `multiview_tracks.ray_point` convention)."""
    undistorted = cam.undistort(np.asarray(pixel, dtype=np.float64)).reshape(2)
    direction_cam = np.linalg.solve(cam.K, np.array([undistorted[0], undistorted[1], 1.0]))
    direction = cam.R.T @ direction_cam
    return cam.centre, direction / np.linalg.norm(direction)


def plane_from_axis(cam: Camera, endpoints_px: np.ndarray) -> tuple[np.ndarray, float]:
    """The plane through the camera centre and the two back-projected axis endpoints (raw
    pixels, undistorted first): ``(unit normal, d)`` with ``normal @ X == d`` on the plane.
    Raises ValueError when the two endpoints back-project to one ray."""
    ends = np.asarray(endpoints_px, dtype=np.float64).reshape(2, 2)
    undistorted = cam.undistort(ends)
    directions_cam = np.linalg.solve(cam.K, np.column_stack([undistorted, np.ones(2)]).T)
    directions = cam.R.T @ directions_cam
    normal = np.cross(directions[:, 0], directions[:, 1])
    norm = np.linalg.norm(normal)
    sine = norm / (np.linalg.norm(directions[:, 0]) * np.linalg.norm(directions[:, 1]))
    if sine < 1e-9:
        raise ValueError(f"{cam.name}: axis endpoints {ends.tolist()} back-project to one ray")
    normal = normal / norm
    return normal, float(normal @ cam.centre)


def is_elongated(obs: AxisObs, elongation_threshold: float = ELONGATION_THRESHOLD) -> bool:
    """A plane constraint: an axis at least `MIN_AXIS_LENGTH_PX` long whose elongation is at or
    above the threshold (an unknown elongation trusts the axis)."""
    if obs.endpoints_px is None or obs.axis_length_px < MIN_AXIS_LENGTH_PX:
        return False
    return obs.elongation is None or obs.elongation >= elongation_threshold


def _perpendicular_basis(direction: np.ndarray) -> np.ndarray:
    """(3, 2) orthonormal basis of the plane perpendicular to a unit direction."""
    seed = np.zeros(3)
    seed[int(np.argmin(np.abs(direction)))] = 1.0
    e1 = seed - (seed @ direction) * direction
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(direction, e1)
    return np.column_stack([e1, e2])


def _linear_project(cam: Camera, points: np.ndarray) -> np.ndarray:
    """Undistorted pixels of world points through ``K[R|t]``; NaN for points at or behind
    the camera."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    homogeneous = (cam.projection @ np.column_stack([pts, np.ones(len(pts))]).T).T
    out = np.full((len(pts), 2), np.nan)
    front = homogeneous[:, 2] > 1e-9
    out[front] = homogeneous[front, :2] / homogeneous[front, 2:3]
    return out


# --------------------------------------------------------------------------- the fit


def _solve_planes(
    planes: list[tuple[str, np.ndarray, float, float]],
    rays: list[tuple[str, np.ndarray, np.ndarray, float]],
    min_pair_angle_deg: float,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Two or more planes (plus rays): ``(point, direction, condition)`` or None when the best
    pair of planes meets under the angle gate."""
    normals = np.stack([n for _, n, _, _ in planes])
    weights = np.array([w for _, _, _, w in planes], dtype=np.float64)
    cosines = np.clip(np.abs(normals @ normals.T), 0.0, 1.0)
    np.fill_diagonal(cosines, 1.0)
    angles = np.degrees(np.arccos(cosines))
    if float(angles.max()) < min_pair_angle_deg:
        return None
    # Each plane's weight carries the RMS sine of its angle to the other planes (weighted by
    # theirs): a plane nearly parallel to everything else constrains the direction poorly.
    others = weights[None, :] * (1.0 - np.eye(len(planes)))
    sines2 = np.clip(1.0 - cosines**2, 0.0, 1.0)
    total = others.sum(axis=1)
    pair_factor = np.sqrt((sines2 * others).sum(axis=1) / np.where(total > 0, total, 1.0))
    rows = (weights * pair_factor)[:, None] * normals
    _, singular, vh = np.linalg.svd(rows, full_matrices=True)
    direction = vh[-1]
    condition = float(singular[1] / singular[0]) if singular[0] > 0 else 0.0

    basis = _perpendicular_basis(direction)
    n_rays = len(rays)
    matrix = np.zeros((len(planes) + 2 * n_rays, 2 + n_rays))
    rhs = np.zeros(len(planes) + 2 * n_rays)
    for i, (_, normal, offset, weight) in enumerate(planes):
        matrix[i, :2] = weight * (basis.T @ normal)
        rhs[i] = weight * offset
    for k, (_, origin, ray_direction, weight) in enumerate(rays):
        row = len(planes) + 2 * k
        matrix[row : row + 2, :2] = weight * np.eye(2)
        matrix[row : row + 2, 2 + k] = -weight * (basis.T @ ray_direction)
        rhs[row : row + 2] = weight * (basis.T @ origin)
    solution = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
    return basis @ solution[:2], direction, condition


def _solve_one_plane(
    plane: tuple[str, np.ndarray, float, float],
    rays: list[tuple[str, np.ndarray, np.ndarray, float]],
    min_pair_angle_deg: float,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """One plane plus rays: the rays that cross the plane at or above the angle gate are
    intersected with it and the line fitted through the intersections (weighted PCA)."""
    _, normal, offset, _ = plane
    min_sine = math.sin(math.radians(min_pair_angle_deg))
    points, weights, sines = [], [], []
    for _, origin, direction, weight in rays:
        denominator = float(normal @ direction)
        if abs(denominator) < min_sine:
            continue
        t = (offset - float(normal @ origin)) / denominator
        if t <= 0:
            continue
        points.append(origin + t * direction)
        weights.append(weight)
        sines.append(abs(denominator))
    if len(points) < 2:
        return None
    pts, w = np.array(points), np.array(weights, dtype=np.float64)
    mean = (w[:, None] * pts).sum(axis=0) / w.sum()
    _, _, vh = np.linalg.svd(np.sqrt(w)[:, None] * (pts - mean), full_matrices=False)
    direction = vh[0] - (vh[0] @ normal) * normal
    if np.linalg.norm(direction) < 1e-12:
        return None
    direction /= np.linalg.norm(direction)
    along = (pts - mean) @ direction
    if float(along.max() - along.min()) < MIN_RAY_SEPARATION_CM:
        return None
    return mean, direction, float(min(sines))


def fit_line(
    cams_and_obs: Sequence[CamObs],
    elongation_threshold: float = ELONGATION_THRESHOLD,
    min_pair_angle_deg: float = MIN_PAIR_ANGLE_DEG,
) -> Line3D | None:
    """Fit one 3D line to one frame's per-view observations (module docstring for the
    method). Elongated observations are planes, compact ones rays. None with no plane, with
    one plane and fewer than two usable rays, or when the planes are degenerate (best pair
    angle under `min_pair_angle_deg`). The returned line carries the visible extent
    (`line_endpoints` without a prior), `point` at its midpoint, and every view's residual."""
    planes: list[tuple[str, np.ndarray, float, float]] = []
    rays: list[tuple[str, np.ndarray, np.ndarray, float]] = []
    for cam, obs in cams_and_obs:
        if is_elongated(obs, elongation_threshold):
            assert obs.endpoints_px is not None
            try:
                normal, offset = plane_from_axis(cam, obs.endpoints_px)
            except ValueError:
                pass
            else:
                planes.append((obs.view, normal, offset, obs.weight))
                continue
        origin, direction = ray_from_point(cam, obs.centroid_px)
        rays.append((obs.view, origin, direction, obs.weight))
    if not planes:
        return None
    if len(planes) >= 2:
        solved = _solve_planes(planes, rays, min_pair_angle_deg)
    else:
        solved = _solve_one_plane(planes[0], rays, min_pair_angle_deg)
    if solved is None:
        return None
    point, direction, condition = solved
    plane_views = tuple(v for v, _, _, _ in planes)
    ray_views = tuple(v for v, _, _, _ in rays)
    line = Line3D(
        point=point,
        direction=direction,
        support_views=plane_views + ray_views,
        plane_views=plane_views,
        ray_views=ray_views,
        condition=condition,
    )
    extent = line_endpoints(line, cams_and_obs, elongation_threshold=elongation_threshold)
    if extent is not None:
        line = replace(line, point=extent.visible.mean(axis=0), endpoints=extent.visible)
    residuals = {
        obs.view: view_residual(cam, obs, line, elongation_threshold) for cam, obs in cams_and_obs
    }
    return replace(line, residuals=residuals)


# --------------------------------------------------------------------------- extent


def line_endpoints(
    line: Line3D,
    cams_and_obs: Sequence[CamObs],
    length_prior_cm: float | None = None,
    *,
    elongation_threshold: float = ELONGATION_THRESHOLD,
    end_tolerance_cm: float = END_TOLERANCE_CM,
    merged_factor: float = MERGED_FACTOR,
) -> Extent | None:
    """Endpoints on `line` from the elongated views: each axis endpoint's ray gives the closest
    point on the line, so each view spans an interval of the line parameter; the visible
    extent is the min/max over views (up to three views) or the 10th/90th percentile of the
    low and high ends (more).     With a `length_prior_cm`: a shorter visible extent is completed
    to the prior from the end more views reach (both ends by half when tied) and flagged
    `extended` when more than `end_tolerance_cm` was missing; a visible extent over
    `merged_factor` times the prior is flagged `merged` and left as seen. None when no view
    is elongated. Endpoint order is increasing along `line.direction`."""
    per_view: dict[str, tuple[float, float]] = {}
    for cam, obs in cams_and_obs:
        if not is_elongated(obs, elongation_threshold):
            continue
        assert obs.endpoints_px is not None
        params = [line.parameter_closest_to_ray(*ray_from_point(cam, e)) for e in obs.endpoints_px]
        per_view[obs.view] = (min(params), max(params))
    if not per_view:
        return None
    lows = np.array([lo for lo, _ in per_view.values()])
    highs = np.array([hi for _, hi in per_view.values()])
    if len(per_view) <= 3:
        lo, hi = float(lows.min()), float(highs.max())
    else:
        lo, hi = float(np.percentile(lows, 10)), float(np.percentile(highs, 90))
    support = (
        int((lows <= lo + end_tolerance_cm).sum()),
        int((highs >= hi - end_tolerance_cm).sum()),
    )
    visible = np.stack([line.point_at(lo), line.point_at(hi)])
    extended, extended_end, merged = False, None, False
    if length_prior_cm is not None:
        length = hi - lo
        if length < length_prior_cm:
            missing = length_prior_cm - length
            # Completing to the prior is always done; the flag says an end was reconstructed.
            extended = missing > end_tolerance_cm
            if support[1] > support[0]:
                lo, extended_end = hi - length_prior_cm, 0
            elif support[0] > support[1]:
                hi, extended_end = lo + length_prior_cm, 1
            else:
                lo, hi = lo - missing / 2, hi + missing / 2
        elif length > merged_factor * length_prior_cm:
            merged = True
    return Extent(
        endpoints=np.stack([line.point_at(lo), line.point_at(hi)]),
        visible=visible,
        support=support,
        per_view=per_view,
        extended=extended,
        extended_end=extended_end,
        merged=merged,
    )


# --------------------------------------------------------------------------- residuals


def projected_line(
    cam: Camera, line: Line3D, near_pixels: np.ndarray | None = None
) -> np.ndarray | None:
    """Two undistorted pixels on the projection of `line` into `cam`, local to the observation
    when `near_pixels` (raw) are given (the closest points on the line to their rays), else
    `DEFAULT_HALF_LENGTH_CM` either side of `point`. None when the line is behind the camera
    or projects to a point."""
    if near_pixels is not None:
        params = [line.parameter_closest_to_ray(*ray_from_point(cam, p)) for p in near_pixels]
        lo, hi = min(params), max(params)
        if hi - lo < 1e-6:
            lo, hi = lo - DEFAULT_HALF_LENGTH_CM, hi + DEFAULT_HALF_LENGTH_CM
    else:
        lo, hi = -DEFAULT_HALF_LENGTH_CM, DEFAULT_HALF_LENGTH_CM
    pixels = _linear_project(cam, np.stack([line.point_at(lo), line.point_at(hi)]))
    if not np.all(np.isfinite(pixels)) or np.linalg.norm(pixels[1] - pixels[0]) < 1e-9:
        return None
    return pixels


def axis_residual(cam: Camera, endpoints_px: np.ndarray, line: Line3D) -> tuple[float, float]:
    """An observed axis (raw pixels) against a line in one view: mean distance of the
    undistorted endpoints to the projected line (px) and the angle between the two (deg,
    0..90). NaN when the line does not project."""
    ends = cam.undistort(np.asarray(endpoints_px, dtype=np.float64).reshape(2, 2))
    projected = projected_line(cam, line, np.asarray(endpoints_px, dtype=np.float64).reshape(2, 2))
    if projected is None:
        return float("nan"), float("nan")
    segment = projected[1] - projected[0]
    unit = segment / np.linalg.norm(segment)
    normal = np.array([-unit[1], unit[0]])
    perpendicular = float(np.mean(np.abs((ends - projected[0]) @ normal)))
    observed = ends[1] - ends[0]
    observed_norm = np.linalg.norm(observed)
    if observed_norm < 1e-9:
        return perpendicular, float("nan")
    cosine = min(1.0, abs(float(observed @ unit)) / observed_norm)
    return perpendicular, math.degrees(math.acos(cosine))


def point_residual(cam: Camera, pixel: np.ndarray, line: Line3D) -> float:
    """Distance (undistorted px) from a raw pixel to the projected line."""
    target = cam.undistort(np.asarray(pixel, dtype=np.float64)).reshape(2)
    projected = projected_line(cam, line, np.asarray(pixel, dtype=np.float64).reshape(1, 2))
    if projected is None:
        return float("nan")
    segment = projected[1] - projected[0]
    unit = segment / np.linalg.norm(segment)
    return float(abs((target - projected[0]) @ np.array([-unit[1], unit[0]])))


def view_residual(
    cam: Camera,
    obs: AxisObs,
    line: Line3D,
    elongation_threshold: float = ELONGATION_THRESHOLD,
) -> ViewResidual:
    if is_elongated(obs, elongation_threshold):
        assert obs.endpoints_px is not None
        perpendicular, angle = axis_residual(cam, obs.endpoints_px, line)
        return ViewResidual("plane", perpendicular, angle)
    return ViewResidual("ray", point_residual(cam, obs.centroid_px, line), None)


def reproject_line(cam: Camera, line: Line3D) -> tuple[float, float, float, float]:
    """``(x0, y0, x1, y1)`` raw pixels (distortion applied, for drawing over the frame) of the
    line's endpoints, or `DEFAULT_HALF_LENGTH_CM` either side of `point` without endpoints;
    NaN where an end is behind the camera."""
    if line.endpoints is not None:
        ends = line.endpoints
    else:
        ends = np.stack(
            [line.point_at(-DEFAULT_HALF_LENGTH_CM), line.point_at(DEFAULT_HALF_LENGTH_CM)]
        )
    depth = (cam.R @ ends.T + cam.tvec.reshape(3, 1))[2]
    pixels = cam.project(ends)
    pixels[depth <= 0] = np.nan
    return float(pixels[0, 0]), float(pixels[0, 1]), float(pixels[1, 0]), float(pixels[1, 1])


def loo_residual(
    cams_and_obs: Sequence[CamObs],
    elongation_threshold: float = ELONGATION_THRESHOLD,
    min_pair_angle_deg: float = MIN_PAIR_ANGLE_DEG,
) -> list[dict[str, Any]]:
    """Leave-one-camera-out: for each elongated view, the line fitted from the other views and
    that view's axis against it. One dict per axis view: ``view``, ``fitted`` (False when the
    others give no line; the residual keys are then None), ``perpendicular_px``, ``angle_deg``,
    ``support_views`` and ``condition`` of the held-out fit."""
    out: list[dict[str, Any]] = []
    for i, (cam, obs) in enumerate(cams_and_obs):
        if not is_elongated(obs, elongation_threshold):
            continue
        assert obs.endpoints_px is not None
        others = [pair for j, pair in enumerate(cams_and_obs) if j != i]
        line = fit_line(others, elongation_threshold, min_pair_angle_deg)
        if line is None:
            out.append(
                {
                    "view": obs.view,
                    "fitted": False,
                    "perpendicular_px": None,
                    "angle_deg": None,
                    "support_views": (),
                    "condition": None,
                }
            )
            continue
        perpendicular, angle = axis_residual(cam, obs.endpoints_px, line)
        out.append(
            {
                "view": obs.view,
                "fitted": True,
                "perpendicular_px": perpendicular,
                "angle_deg": angle,
                "support_views": line.support_views,
                "condition": line.condition,
            }
        )
    return out


# --------------------------------------------------------------------------- distances


def point_to_line_cm(point: np.ndarray, line: Line3D) -> float:
    x = np.asarray(point, dtype=np.float64).reshape(3)
    return float(np.linalg.norm(x - line.closest_point(x)))


def line_distance(a: Line3D, b: Line3D) -> tuple[float, float, float]:
    """``(angle_deg, perpendicular_cm, midpoint_cm)``: the angle between directions (0..90,
    sign-free), the mean of each line's midpoint's perpendicular distance to the other line
    (the separation of parallel lines; near zero for lines crossing at their midpoints), and
    the distance between the midpoints (the point of each line when it has no endpoints)."""
    cosine = min(1.0, abs(float(a.direction @ b.direction)))
    angle = math.degrees(math.acos(cosine))
    mid_a, mid_b = a.midpoint, b.midpoint
    perpendicular = 0.5 * (point_to_line_cm(mid_b, a) + point_to_line_cm(mid_a, b))
    return angle, perpendicular, float(np.linalg.norm(mid_a - mid_b))


def line_cost(
    a: Line3D,
    b: Line3D,
    *,
    half_length_cm: float = DEFAULT_HALF_LENGTH_CM,
    along_weight: float = 0.5,
) -> float:
    """One scalar in cm for gating: perpendicular offset, plus the angle turned into the
    endpoint displacement it causes over `half_length_cm`, plus `along_weight` times the
    midpoints' offset along the line (down-weighted because views seeing different portions
    of a shaft move the visible midpoint without moving the line)."""
    angle, perpendicular, _ = line_distance(a, b)
    along = abs(float((b.midpoint - a.midpoint) @ a.direction))
    return perpendicular + half_length_cm * math.sin(math.radians(angle)) + along_weight * along


# --------------------------------------------------------------------------- baseline


def centroid_point(cams_and_obs: Sequence[CamObs]) -> np.ndarray | None:
    """Today's method on the same views: `triangulate_pixels` on the centroids (box centres),
    weighted by the observation weights. None with fewer than two views or a non-finite
    solution."""
    if len(cams_and_obs) < 2:
        return None
    point = triangulate_pixels(
        [cam for cam, _ in cams_and_obs],
        [obs.centroid_px for _, obs in cams_and_obs],
        [obs.weight for _, obs in cams_and_obs],
    )
    return point if np.all(np.isfinite(point)) else None


def centroid_loo_residual(cams_and_obs: Sequence[CamObs]) -> list[dict[str, Any]]:
    """The box-centre method's leave-one-camera-out score for the scoreboard: for each view
    with at least two others, the point from the others projected into it (undistorted px)
    against its axis midpoint (its centroid without an axis): ``distance_px`` to that target
    and ``perpendicular_px`` to the axis line (None without an axis)."""
    out: list[dict[str, Any]] = []
    for i, (cam, obs) in enumerate(cams_and_obs):
        others = [pair for j, pair in enumerate(cams_and_obs) if j != i]
        point = centroid_point(others) if len(others) >= 2 else None
        record: dict[str, Any] = {
            "view": obs.view,
            "fitted": point is not None,
            "distance_px": None,
            "perpendicular_px": None,
        }
        if point is not None:
            projected = _linear_project(cam, point)[0]
            target = cam.undistort(obs.axis_midpoint_px).reshape(2)
            record["distance_px"] = float(np.linalg.norm(projected - target))
            if obs.endpoints_px is not None and obs.axis_length_px >= MIN_AXIS_LENGTH_PX:
                ends = cam.undistort(obs.endpoints_px)
                unit = (ends[1] - ends[0]) / np.linalg.norm(ends[1] - ends[0])
                normal = np.array([-unit[1], unit[0]])
                record["perpendicular_px"] = float(abs((projected - ends[0]) @ normal))
        out.append(record)
    return out
