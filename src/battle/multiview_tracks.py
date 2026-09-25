"""Multi-view 3D object tracker for FineBio (`p3-tracker`, the core): the tracked entity is a
3D object with a persistent id; every camera's box or mask is an observation of it.

Per frame:

* **Observations** (`FineBioObservation` rows -> `Obs`): `point_px` (mask centroid, else the
  box centre), class, detector and SAM3 scores, per-view slot, pose validity (the fpv only
  observes when its shipped pose is valid). Source rule `auto`: per view, frame and class the
  SAM3 rows when present, else the detector rows; `detector` / `sam3` force one source. An
  observation is *class-confirmed* when it is a detector row, carries a detector score, or a
  detector row of its class sits within the association gate in the same view and frame.
* **Predict.** Stationary prior: the position stays, the scalar uncertainty grows by
  `process_noise_cm` per frame (a constant-position Kalman model; velocity is not modelled
  until a test shows it is needed).
* **Update.** Every live, localised track (`observed` / `single_view`) is projected into every
  view with a valid pose; same-class observations within the gate (association gate for the
  fixed views, hand-off gate for the fpv, both inflated by the projected uncertainty) are
  candidates; the nearest wins, conflicts between tracks resolved by distance; with >= 2 views
  the track is re-triangulated (weighted DLT: fixed views 1, fpv `fpv_weight`), views over
  the gate dropped once, and the position blended with the prior by the scalar Kalman gain.
  A view whose SAM3 slot differs from the slot this track had there before is a **slot
  disagreement**, kept in the row (`slot_disagreement_views`) and folded into the confidence,
  never silently resolved.
* **Occlusion.** Support 0 -> `coasting`, uncertainty grows by `coast_growth_cm` per frame;
  after `coast_timeout_frames` -> `lost` (a `lost` track never returns; a new track of the
  class may be born).
* **Birth.** Unassigned observations of one class in the fixed views: every view pair is
  triangulated, pair cost = the larger of the two reprojection residuals, gated by the
  association gate, Hungarian assignment per view pair, accepted pairs merged into cliques
  (one observation per view, every pair accepted), clique re-triangulated with the worst
  member dropped while it exceeds the gate. The fpv joins a clique when its nearest unassigned
  same-class observation is within the hand-off gate of the clique point. A track is born with
  >= `birth_min_fixed_views` fixed views, or >= `birth_fixed_views_with_fpv` fixed views plus
  the fpv with a valid pose.
* **Re-acquisition.** A birth candidate that lies within the inflated gate of a `coasting`
  track of its class in >= `reacquire_min_views` views is a return candidate. Exactly one
  candidate for exactly one coasting track, class-confirmed in >= `reacquire_min_views`
  views -> the id resumes (`reacquired`, latency in frames). Unconfirmed -> a new id is born
  with `possibly_same_as` naming the coasting track. Two candidates for one track, or one
  candidate for two tracks -> nobody resumes: every candidate is born as a new id with
  `possibly_same_as` and an `ambiguous` event; the ambiguity is counted.
* **Hand-off re-seed.** For a live `observed` track, a view with a valid pose that has had no
  associated observation for `handoff_after_frames` frames (and every such interval after)
  while the projection is inside the image emits `handoff_reseed` with the reprojected box
  (the view's last extent, else `default_reseed_box_px`, centred on the projection; the
  worker's `track_reproject` correction consumes it), or `detector_reseed` with the detector
  box when one of the class sits within the hand-off gate there instead.

Gates come from lane B's rig output (`gates` block: `association_px`, `handoff_px`,
`birth_min_fixed_views`, `birth_fixed_views_with_fpv`) or the CLI, defaulting to P03's
preflight values (30 px association, 55 px hand-off at 1920). Confidence is a heuristic rank
of support, residual and slot agreement, to be replaced by `p5-confidence`; the identity
metrics against SAM3 per-view slots are a proxy, labelled as such. Units: board centimetres,
z into the bench; raw pixels; raw frame indices. Nothing here is ground truth.

**Extensions** (`p3-tracker-ext`, Sep 25; every one opt-in and off by default, so the core
above reproduces byte for byte). Each was added on the occlusion inventory of arms (a)-(c):

* `--motion-model`: a per-track velocity (smoothed finite difference of the filtered
  position over observed frames). A track whose speed exceeds `mover_speed_cm_per_frame`
  is a *mover*: its prediction adds `velocity * dt`, its process noise adds
  `mover_noise_factor * speed * dt` (so the gate opens with the speed), a single view updates
  it laterally along the observation ray at the predicted depth, and it coasts along a damped
  velocity. Bench objects below the threshold keep the stationary prior unchanged. Evidence:
  the in-hand pipette fragmented into 47 / 80 / 30 ids on arms (a)/(b)/(c) before any
  occlusion.
* `--containers`: container volumes = the container class's bench footprint (the bench cells
  whose projection lies inside the class's median top-scoring detector box in every fixed
  view that sees it stably, a visual hull on the bench plane; the rig's static point when given
  must fall inside) extruded from the bench to a height per class (`--container-heights-cm`).
  A track whose support drops to 0 with its position inside a volume enters `contained`
  (event), keeps the container's position (stationary here), does not time out, and resumes
  under the core's confirmation rule when a candidate reappears inside the volume or the gate
  (`reacquired`, `from_state: contained`). Evidence: 125-138 support-0 episodes per arm inside
  a container box, four fifths of them micro tubes in the micro-tube rack, 12-16 in the
  centrifuge.
* `--group-tracks`: for the identical-instance classes (`group_classes`), the observations
  inside one container footprint are one *group track* per (class, container) with a
  `group_size` (the median per-view count), and any group-class track absorbs the same-class
  observations inside its own association gate instead of leaving them to be born as
  near-duplicate ids; a group-class candidate born outside every footprint within
  `group_split_radius_cm` of a group is a *split* (`split_from`, `group_split` event).
  Evidence: 145-171 group-candidate episodes per arm, about half ending ambiguous.
* `--held`: a track whose support drops to 0 (and is not contained) with its projection inside
  a hand box of one hand class in >= `held_min_views` views enters `held` (event), follows
  that hand's live track at a fixed offset, times out as coasting does, and resumes under the
  core rule. Evidence: 5-19 episodes per arm inside a hand box in every projecting view
  (89-166 by the two-view test, most of them occlusion by a hand rather than carrying).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field

from .finebio_cameras import Camera, cameras_from_config, read_camera_config
from .finebio_slice import (
    FpvCameraSource,
    depth_cm,
    parse_frames,
    resolve_fpv_source,
    triangulate_pixels,
)
from .multiview_schemas import (
    FINEBIO_FPV_VIEW,
    FineBioObservation,
    Track3D,
    TrackEvent,
    read_jsonl,
    write_jsonl,
)
from .schemas import VersionedModel

SAM3_SOURCES = ("sam3_decode", "sam3_video")
# `held` and `contained` only occur with the extensions on; with them off the core's state
# machine is unchanged.
SUPPORT0_STATES = ("coasting", "held", "contained")
LIVE_STATES = ("observed", "single_view", *SUPPORT0_STATES)
LOCALISED_STATES = ("observed", "single_view")
HUNGARIAN_FORBIDDEN = 1e9
# Identical-instance classes (the preflight: per-instance identity is below the fixed
# cameras' resolution), the default `group_classes` of the group-track extension.
GROUP_CLASSES = (
    "micro_tube",
    "50ml_tube",
    "15ml_tube",
    "8_tube_stripes",
    "blue_tip",
    "yellow_tip",
    "red_tip",
    "8_channel_tip",
)
HAND_CLASSES = ("left_hand", "right_hand")
# Heights of the container volumes above the bench (cm), per class; the plan's examples for
# the rack and the centrifuge, the others read off the rig's half-heights x 2 rounded up.
DEFAULT_CONTAINER_HEIGHTS_CM = (
    ("micro_tube_rack", 6.0),
    ("centrifuge", 12.0),
    ("magnetic_rack", 6.0),
    ("50ml_tube_rack", 12.0),
    ("15ml_tube_rack", 12.0),
    ("8_tube_stripes_rack", 6.0),
    ("vortex_mixer", 8.0),
    ("pcr_machine", 10.0),
    ("trash_can", 25.0),
)
DEFAULT_CONTAINER_HEIGHT_CM = 10.0


# --------------------------------------------------------------------------- parameters


@dataclass(frozen=True)
class Gates:
    """Geometric gates in raw pixels at 1920 wide; P03's preflight values by default."""

    association_px: float = 30.0
    handoff_px: float = 55.0
    birth_min_fixed_views: int = 3
    birth_fixed_views_with_fpv: int = 2

    @classmethod
    def from_rig(cls, rig: dict[str, Any]) -> Gates:
        """The `gates` block of a rig output (or the block itself); missing keys keep the
        defaults."""
        block = rig.get("gates", rig)
        kwargs = {k: block[k] for k in cls.__dataclass_fields__ if k in block}
        return cls(**kwargs)


@dataclass(frozen=True)
class TrackerParams:
    gates: Gates = Gates()
    observation_source: str = "auto"  # auto | detector | sam3
    min_detector_score: float = 0.3
    coast_timeout_frames: int = 30
    handoff_after_frames: int = 5
    # A (track, view) that stays without an observation re-emits its re-seed at this cadence.
    handoff_repeat_frames: int = 30
    reacquire_min_views: int = 2
    # Live same-class tracks closer than this are marked possibly_same_as on both rows.
    duplicate_distance_cm: float = 3.0
    process_noise_cm: float = 2.0
    coast_growth_cm: float = 1.0
    base_uncertainty_cm: float = 1.0
    max_uncertainty_cm: float = 30.0
    fpv_weight: float = 0.5
    default_reseed_box_px: tuple[float, float] = (120.0, 80.0)
    abstain_below_confidence: float = 0.3
    # -- extensions (p3-tracker-ext), every one off by default -----------------------------
    motion_model: bool = False
    # A track is a mover once its smoothed speed exceeds this on `mover_confirm_frames`
    # consecutive >= 2-view updates (a one-frame jitter or a neighbour swap is not motion);
    # half of it switches it off.
    mover_speed_cm_per_frame: float = 0.5
    mover_confirm_frames: int = 3
    velocity_smoothing: float = 0.3
    mover_noise_factor: float = 1.0
    mover_coast_damping: float = 0.8
    containers: tuple[str, ...] = ()
    container_heights_cm: tuple[tuple[str, float], ...] = DEFAULT_CONTAINER_HEIGHTS_CM
    default_container_height_cm: float = DEFAULT_CONTAINER_HEIGHT_CM
    # 0 = a contained track never times out (the plan's rule); > 0 bounds the ghosts.
    contained_timeout_frames: int = 0
    group_tracks: bool = False
    group_classes: tuple[str, ...] = GROUP_CLASSES
    group_split_radius_cm: float = 25.0
    held: bool = False
    held_min_views: int = 2
    hand_classes: tuple[str, ...] = HAND_CLASSES

    def as_dict(self) -> dict[str, Any]:
        out = {k: getattr(self, k) for k in CORE_PARAM_KEYS}
        out["gates"] = dict(self.gates.__dict__)
        out["extensions"] = {
            k: getattr(self, k)
            for k in self.__dataclass_fields__
            if k not in CORE_PARAM_KEYS and k != "gates"
        }
        out["extensions"]["container_heights_cm"] = dict(self.container_heights_cm)
        return out

    @property
    def any_extension(self) -> bool:
        return bool(self.motion_model or self.containers or self.group_tracks or self.held)

    def container_height(self, cls: str) -> float:
        return dict(self.container_heights_cm).get(cls, self.default_container_height_cm)


CORE_PARAM_KEYS = (
    "observation_source",
    "min_detector_score",
    "coast_timeout_frames",
    "handoff_after_frames",
    "handoff_repeat_frames",
    "reacquire_min_views",
    "duplicate_distance_cm",
    "process_noise_cm",
    "coast_growth_cm",
    "base_uncertainty_cm",
    "max_uncertainty_cm",
    "fpv_weight",
    "default_reseed_box_px",
    "abstain_below_confidence",
)


# --------------------------------------------------------------------------- observations


@dataclass
class Obs:
    view: str
    point: np.ndarray
    object_class: str
    slot: str
    source: str
    detector_score: float | None
    sam3_score: float | None
    box: tuple[float, float, float, float] | None
    confirmed: bool
    index: int = -1

    @property
    def extent(self) -> tuple[float, float] | None:
        if self.box is None:
            return None
        return (self.box[2] - self.box[0], self.box[3] - self.box[1])


@dataclass
class FrameObservations:
    frame: int
    tracked: dict[str, list[Obs]]  # per view, the rows the tracker associates
    detector: dict[str, list[Obs]]  # per view, every detector row (confirmation, re-seed)


def _obs(row: FineBioObservation) -> Obs:
    box = row.mask_bbox_px if row.mask_bbox_px is not None else row.box_xyxy_px
    return Obs(
        view=row.view,
        point=np.array(row.point_px, dtype=np.float64),
        object_class=row.object_class,
        slot=row.slot,
        source=row.source,
        detector_score=row.detector_score,
        sam3_score=row.sam3_object_score,
        box=box,
        confirmed=row.source == "detector",
    )


def select_observations(
    frame: int, rows: Iterable[FineBioObservation], params: TrackerParams
) -> FrameObservations:
    """Apply the source rule and class confirmation to one frame's rows (all views)."""
    detector: dict[str, list[Obs]] = defaultdict(list)
    sam3: dict[tuple[str, str], list[Obs]] = defaultdict(list)
    det_by_class: dict[tuple[str, str], list[Obs]] = defaultdict(list)
    for row in rows:
        if row.view == FINEBIO_FPV_VIEW and not row.pose_valid:
            continue
        obs = _obs(row)
        if row.source == "detector":
            if (row.detector_score or 0.0) < params.min_detector_score:
                continue
            detector[row.view].append(obs)
            det_by_class[(row.view, row.object_class)].append(obs)
        elif row.source in SAM3_SOURCES:
            sam3[(row.view, row.object_class)].append(obs)
    tracked: dict[str, list[Obs]] = defaultdict(list)
    if params.observation_source in ("auto", "sam3"):
        for (view, cls), items in sam3.items():
            for obs in items:
                obs.confirmed = (
                    obs.detector_score is not None
                    and obs.detector_score >= params.min_detector_score
                ) or any(
                    np.linalg.norm(d.point - obs.point) <= params.gates.association_px
                    for d in det_by_class.get((view, cls), ())
                )
                tracked[view].append(obs)
    if params.observation_source in ("auto", "detector"):
        for (view, cls), items in det_by_class.items():
            if params.observation_source == "auto" and (view, cls) in sam3:
                continue
            tracked[view].extend(items)
    for view, items in tracked.items():
        for i, obs in enumerate(items):
            obs.index = i
    return FrameObservations(frame=frame, tracked=dict(tracked), detector=dict(detector))


# --------------------------------------------------------------------------- assignment


def hungarian(cost: np.ndarray) -> list[tuple[int, int]]:
    """Minimum-cost assignment on a rectangular matrix (Kuhn-Munkres, shortest augmenting
    paths, O(n^2 m)); returns (row, col) pairs, one per row of the smaller side. Forbidden
    cells carry `HUNGARIAN_FORBIDDEN`; the caller drops pairs above its gate. `scipy` is not
    in this environment, and the matrices here are a handful of instances per class."""
    cost = np.asarray(cost, dtype=np.float64)
    if cost.size == 0:
        return []
    n, m = cost.shape
    transposed = n > m
    if transposed:
        cost = cost.T
        n, m = m, n
    inf = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    pairs = [(p[j] - 1, j - 1) for j in range(1, m + 1) if p[j] != 0]
    if transposed:
        pairs = [(c, r) for r, c in pairs]
    return sorted(pairs)


# --------------------------------------------------------------------------- tracks


class ResidualRow(VersionedModel):
    """One (track, view) association at one frame (a row of residuals.jsonl)."""

    frame_index: int = Field(ge=0)
    track_id: str = Field(min_length=1)
    view: str = Field(min_length=1)
    residual_px: float = Field(ge=0)
    gate_px: float = Field(gt=0)
    slot: str = Field(min_length=1)
    source: str = Field(min_length=1)
    confirmed: bool


@dataclass
class Track:
    track_id: str
    object_class: str
    position: np.ndarray
    uncertainty_cm: float
    state: str
    born_frame: int
    last_observed_frame: int
    frames_unobserved: int = 0
    support_views: tuple[str, ...] = ()
    residuals: dict[str, float] = field(default_factory=dict)
    per_view_last_seen: dict[str, int] = field(default_factory=dict)
    per_view_extent: dict[str, tuple[float, float]] = field(default_factory=dict)
    per_view_slot: dict[str, str] = field(default_factory=dict)
    support_slots: dict[str, str] = field(default_factory=dict)
    slot_disagreement_views: tuple[str, ...] = ()
    slot_disagreements: int = 0
    possibly_same_as: tuple[str, ...] = ()
    last_reseed_frame: dict[str, int] = field(default_factory=dict)
    lost_frame: int | None = None
    reacquisitions: int = 0
    # -- extensions (unused, and never written, with them off)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mover: bool = False
    mover_frames: int = 0
    last_observed_position: np.ndarray | None = None
    container_id: str | None = None
    held_by: str | None = None
    held_offset: np.ndarray | None = None
    group_size: int | None = None
    group_container: str | None = None
    split_from: str | None = None


def pixels_per_cm(cam: Camera, point: np.ndarray) -> float:
    depth = max(depth_cm(cam, point), 1.0)
    return float(cam.K[0, 0]) / depth


def gate_px(track: Track, cam: Camera, params: TrackerParams) -> float:
    """Association gate: the rig's gate (hand-off gate for the fpv) plus the track's
    uncertainty projected into the view."""
    base = params.gates.handoff_px if cam.name == FINEBIO_FPV_VIEW else params.gates.association_px
    return base + track.uncertainty_cm * pixels_per_cm(cam, track.position)


def reseed_gate_px(track: Track, cam: Camera, params: TrackerParams) -> float:
    """Where a detector box may stand in for a missing observation: the hand-off gate plus
    the projected uncertainty."""
    return params.gates.handoff_px + track.uncertainty_cm * pixels_per_cm(cam, track.position)


def project(cam: Camera, point: np.ndarray) -> np.ndarray | None:
    if depth_cm(cam, point) <= 0:
        return None
    return cam.project(point)[0]


def _in_image(pixel: np.ndarray, cam: Camera) -> bool:
    return bool(0 <= pixel[0] < cam.size[0] and 0 <= pixel[1] < cam.size[1])


def ray_point(cam: Camera, pixel: np.ndarray, near: np.ndarray) -> np.ndarray:
    """The point of the pixel's back-projected ray closest to `near` (a one-view lateral
    update: the depth is kept from the prediction, the lateral position from the view)."""
    undistorted = cam.undistort(np.asarray(pixel, dtype=np.float64)).reshape(2)
    direction_cam = np.linalg.solve(cam.K, np.array([undistorted[0], undistorted[1], 1.0]))
    direction = cam.R.T @ direction_cam
    direction = direction / np.linalg.norm(direction)
    centre = cam.centre
    along = max(0.0, float(np.dot(near - centre, direction)))
    return centre + along * direction


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain on a handful of 2D points (counter-clockwise, no repeats)."""
    pts = sorted({(float(x), float(y)) for x, y in np.asarray(points, dtype=np.float64)})
    if len(pts) <= 2:
        return np.asarray(pts, dtype=np.float64).reshape(-1, 2)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def inside_convex_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Point-in-convex-polygon by the sign of the edge cross products (either orientation)."""
    if len(polygon) < 3:
        return False
    signs = []
    for i in range(len(polygon)):
        a, b = polygon[i], polygon[(i + 1) % len(polygon)]
        signs.append((b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0]))
    return all(s >= -1e-9 for s in signs) or all(s <= 1e-9 for s in signs)


# --------------------------------------------------------------------------- containers


@dataclass
class ContainerVolume:
    """A container's bench footprint (a boolean grid of bench cells) extruded to a height.

    Built by `build_container_volumes` from the class's median top-scoring detector box per
    fixed view: a bench cell belongs to the footprint when its projection lies inside that box
    in every stable view (a visual hull on the bench plane). World z points into the bench, so
    a point is inside when `0 - tolerance <= -z <= height` and its (x, y) cell is set."""

    container_id: str
    object_class: str
    height_cm: float
    x_edges: np.ndarray
    y_edges: np.ndarray
    cells: np.ndarray  # bool, shape (len(y_edges) - 1, len(x_edges) - 1)
    bounds_cm: tuple[float, float, float, float]  # xmin, ymin, xmax, ymax
    centre_cm: tuple[float, float]
    views: tuple[str, ...]
    boxes_px: dict[str, tuple[float, float, float, float]]
    rig_point_cm: tuple[float, float, float] | None = None
    rig_point_inside: bool | None = None
    basis: str = "visual_hull_on_bench"
    below_bench_tolerance_cm: float = 1.5

    @property
    def radius_cm(self) -> float:
        x0, y0, x1, y1 = self.bounds_cm
        return float(np.hypot(x1 - x0, y1 - y0) / 2)

    @property
    def centre_3d(self) -> np.ndarray:
        return np.array([self.centre_cm[0], self.centre_cm[1], -self.height_cm / 2])

    def contains(self, point: np.ndarray) -> bool:
        height = -float(point[2])
        if height < -self.below_bench_tolerance_cm or height > self.height_cm:
            return False
        ix = int(np.searchsorted(self.x_edges, point[0], side="right")) - 1
        iy = int(np.searchsorted(self.y_edges, point[1], side="right")) - 1
        if ix < 0 or iy < 0 or ix >= self.cells.shape[1] or iy >= self.cells.shape[0]:
            return False
        return bool(self.cells[iy, ix])

    def corners(self) -> np.ndarray:
        x0, y0, x1, y1 = self.bounds_cm
        return np.array(
            [[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (0.0, -self.height_cm)]
        )

    def polygon_px(self, cam: Camera) -> np.ndarray | None:
        """The convex hull of the volume's projected corners, or None when any corner is
        behind the camera."""
        corners = self.corners()
        if any(depth_cm(cam, c) <= 0 for c in corners):
            return None
        return convex_hull(cam.project(corners))

    def pixel_inside(self, cam: Camera, pixel: np.ndarray) -> bool:
        polygon = self.polygon_px(cam)
        return polygon is not None and inside_convex_polygon(pixel, polygon)

    def summary(self) -> dict[str, Any]:
        return {
            "container_id": self.container_id,
            "object_class": self.object_class,
            "height_cm": self.height_cm,
            "bounds_cm": [round(float(v), 1) for v in self.bounds_cm],
            "centre_cm": [round(float(v), 1) for v in self.centre_cm],
            "footprint_area_cm2": int(self.cells.sum()),
            "views": list(self.views),
            "boxes_px": {v: [round(float(x), 1) for x in b] for v, b in self.boxes_px.items()},
            "rig_point_cm": self.rig_point_cm,
            "rig_point_inside": self.rig_point_inside,
            "basis": self.basis,
        }


def _largest_component(cells: np.ndarray, seed: tuple[int, int] | None = None) -> np.ndarray:
    """Keep one 4-connected component of a boolean grid: the one holding `seed` when it is
    set, else the largest."""
    ny, nx = cells.shape
    labels = np.zeros_like(cells, dtype=np.int32)
    sizes: dict[int, int] = {}
    current = 0
    for iy, ix in zip(*np.nonzero(cells)):
        if labels[iy, ix]:
            continue
        current += 1
        stack = [(int(iy), int(ix))]
        labels[iy, ix] = current
        count = 0
        while stack:
            y, x = stack.pop()
            count += 1
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                y2, x2 = y + dy, x + dx
                if 0 <= y2 < ny and 0 <= x2 < nx and cells[y2, x2] and not labels[y2, x2]:
                    labels[y2, x2] = current
                    stack.append((y2, x2))
        sizes[current] = count
    if not sizes:
        return cells
    if seed is not None and 0 <= seed[0] < ny and 0 <= seed[1] < nx and labels[seed] > 0:
        keep = int(labels[seed])
    else:
        keep = max(sizes, key=sizes.get)
    return labels == keep


def build_container_volumes(
    rows: Iterable[FineBioObservation],
    fixed_cams: dict[str, Camera],
    classes: Sequence[str],
    params: TrackerParams,
    rig_static: dict[str, Sequence[float]] | None = None,
    *,
    cell_cm: float = 1.0,
    grid_half_extent_cm: float = 150.0,
    min_views: int = 3,
) -> tuple[list[ContainerVolume], dict[str, Any]]:
    """Container volumes from the detector rows: per class and fixed view the median of the
    top-scoring (`<class>#0`) boxes over the run (a view is *stable* when the box centre's
    inter-quartile spread is under half the box size); the footprint is the set of bench cells
    projecting inside every stable view's box (>= `min_views` views), reduced to the component
    that holds the rig's static point when one is given (else the largest); height per class
    from the params. Classes without enough stable views are skipped, with the reason."""
    boxes: dict[tuple[str, str], list[tuple[float, float, float, float]]] = defaultdict(list)
    wanted = set(classes)
    for row in rows:
        if (
            row.source != "detector"
            or row.object_class not in wanted
            or row.view not in fixed_cams
            or row.box_xyxy_px is None
            or not row.slot.endswith("#0")
            or (row.detector_score or 0.0) < params.min_detector_score
        ):
            continue
        boxes[(row.object_class, row.view)].append(tuple(float(x) for x in row.box_xyxy_px))
    half = grid_half_extent_cm
    n = int(round(2 * half / cell_cm))
    x_edges = np.linspace(-half, half, n + 1)
    y_edges = np.linspace(-half, half, n + 1)
    xs = (x_edges[:-1] + x_edges[1:]) / 2
    ys = (y_edges[:-1] + y_edges[1:]) / 2
    gx, gy = np.meshgrid(xs, ys)
    bench = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)
    projected = {}
    for view, cam in fixed_cams.items():
        pix = cam.project(bench)
        depth = (cam.R @ bench.T).T[:, 2] + cam.tvec[2]
        projected[view] = (pix, depth > 0)
    volumes: list[ContainerVolume] = []
    report: dict[str, Any] = {"built": [], "skipped": {}}
    for cls in classes:
        stable: dict[str, tuple[float, float, float, float]] = {}
        for view in fixed_cams:
            arr = np.asarray(boxes.get((cls, view), ()), dtype=np.float64)
            if len(arr) < 10:
                continue
            median = np.median(arr, axis=0)
            centres = np.stack([(arr[:, 0] + arr[:, 2]) / 2, (arr[:, 1] + arr[:, 3]) / 2], 1)
            iqr = np.percentile(centres, 75, axis=0) - np.percentile(centres, 25, axis=0)
            size = np.array([median[2] - median[0], median[3] - median[1]])
            if np.all(iqr < 0.5 * size):
                stable[view] = tuple(float(v) for v in median)
        if len(stable) < min_views:
            report["skipped"][cls] = f"{len(stable)} stable fixed views, need {min_views}"
            continue
        inside = np.ones(len(bench), dtype=bool)
        for view, box in stable.items():
            pix, in_front = projected[view]
            inside &= in_front
            inside &= (pix[:, 0] >= box[0]) & (pix[:, 0] <= box[2])
            inside &= (pix[:, 1] >= box[1]) & (pix[:, 1] <= box[3])
        cells = inside.reshape(gy.shape)
        rig_point = None
        seed = None
        if rig_static and cls in rig_static:
            rig_point = tuple(float(v) for v in rig_static[cls][:3])
            seed = (
                int(np.searchsorted(y_edges, rig_point[1], side="right")) - 1,
                int(np.searchsorted(x_edges, rig_point[0], side="right")) - 1,
            )
        if not cells.any():
            report["skipped"][cls] = "empty footprint (the stable boxes do not intersect)"
            continue
        cells = _largest_component(cells, seed)
        iy, ix = np.nonzero(cells)
        bounds = (
            float(x_edges[ix.min()]),
            float(y_edges[iy.min()]),
            float(x_edges[ix.max() + 1]),
            float(y_edges[iy.max() + 1]),
        )
        centre = (float(xs[ix].mean()), float(ys[iy].mean()))
        rig_inside = (
            None
            if rig_point is None
            else volume_contains_xy(cells, x_edges, y_edges, rig_point[0], rig_point[1])
        )
        volume = ContainerVolume(
            container_id=cls,
            object_class=cls,
            height_cm=params.container_height(cls),
            x_edges=x_edges,
            y_edges=y_edges,
            cells=cells,
            bounds_cm=bounds,
            centre_cm=centre,
            views=tuple(sorted(stable)),
            boxes_px=stable,
            rig_point_cm=rig_point,
            rig_point_inside=rig_inside,
        )
        volumes.append(volume)
        report["built"].append(volume.summary())
    report["cell_cm"] = cell_cm
    report["min_views"] = min_views
    return volumes, report


def volume_contains_xy(
    cells: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray, x: float, y: float
) -> bool:
    ix = int(np.searchsorted(x_edges, x, side="right")) - 1
    iy = int(np.searchsorted(y_edges, y, side="right")) - 1
    if ix < 0 or iy < 0 or ix >= cells.shape[1] or iy >= cells.shape[0]:
        return False
    return bool(cells[iy, ix])


# --------------------------------------------------------------------------- birth


@dataclass
class Candidate:
    object_class: str
    point: np.ndarray
    members: dict[str, Obs]
    residuals: dict[str, float]
    fpv_rule: bool = False
    group_size: int | None = None


def _pair_cost(cam_a: Camera, obs_a: Obs, cam_b: Camera, obs_b: Obs) -> float:
    point = triangulate_pixels([cam_a, cam_b], [obs_a.point, obs_b.point])
    if not np.all(np.isfinite(point)) or depth_cm(cam_a, point) <= 0 or depth_cm(cam_b, point) <= 0:
        return HUNGARIAN_FORBIDDEN
    ra = np.linalg.norm(cam_a.project(point)[0] - obs_a.point)
    rb = np.linalg.norm(cam_b.project(point)[0] - obs_b.point)
    return float(max(ra, rb))


def _triangulate_members(
    members: dict[str, Obs], cams: dict[str, Camera], params: TrackerParams
) -> tuple[np.ndarray, dict[str, float]]:
    views = list(members)
    weights = [params.fpv_weight if v == FINEBIO_FPV_VIEW else 1.0 for v in views]
    point = triangulate_pixels(
        [cams[v] for v in views], [members[v].point for v in views], weights=weights
    )
    residuals = {}
    for v in views:
        pixel = project(cams[v], point)
        residuals[v] = (
            float("inf") if pixel is None else float(np.linalg.norm(pixel - members[v].point))
        )
    return point, residuals


def _prune_clique(
    members: dict[str, Obs], cams: dict[str, Camera], params: TrackerParams
) -> tuple[np.ndarray, dict[str, Obs], dict[str, float]]:
    """Re-triangulate, dropping the worst member while it exceeds its gate and >= 2 remain."""
    members = dict(members)
    while True:
        point, residuals = _triangulate_members(members, cams, params)
        over = {
            v: r
            for v, r in residuals.items()
            if r
            > (params.gates.handoff_px if v == FINEBIO_FPV_VIEW else params.gates.association_px)
        }
        if not over or len(members) <= 2:
            if over and len(members) <= 2:
                return point, {}, residuals
            return point, members, residuals
        worst = max(over, key=over.get)
        del members[worst]


def birth_candidates(
    unassigned: dict[str, list[Obs]],
    cams: dict[str, Camera],
    fixed_views: Sequence[str],
    params: TrackerParams,
) -> list[Candidate]:
    """Same-class pairwise triangulation across the fixed views, Hungarian per view pair,
    cliques, then the fpv attached where it fits. Every returned candidate satisfies the
    birth rule."""
    classes = sorted({o.object_class for v in fixed_views for o in unassigned.get(v, ())})
    candidates: list[Candidate] = []
    for cls in classes:
        per_view = {
            v: [o for o in unassigned.get(v, ()) if o.object_class == cls] for v in fixed_views
        }
        per_view = {v: items for v, items in per_view.items() if items}
        if len(per_view) < min(
            params.gates.birth_min_fixed_views, params.gates.birth_fixed_views_with_fpv
        ):
            continue
        edges: dict[tuple[str, int, str, int], float] = {}
        for u, v in combinations(sorted(per_view), 2):
            cost = np.full((len(per_view[u]), len(per_view[v])), HUNGARIAN_FORBIDDEN)
            for i, a in enumerate(per_view[u]):
                for j, b in enumerate(per_view[v]):
                    cost[i, j] = _pair_cost(cams[u], a, cams[v], b)
            for i, j in hungarian(cost):
                if cost[i, j] <= params.gates.association_px:
                    edges[(u, i, v, j)] = float(cost[i, j])
        used: set[tuple[str, int]] = set()
        for (u, i, v, j), _cost in sorted(edges.items(), key=lambda kv: kv[1]):
            if (u, i) in used or (v, j) in used:
                continue
            clique = {u: i, v: j}
            for w in sorted(per_view):
                if w in clique:
                    continue
                best, best_cost = None, float("inf")
                for k in range(len(per_view[w])):
                    if (w, k) in used:
                        continue
                    total = 0.0
                    for cv, ci in clique.items():
                        key = (cv, ci, w, k) if cv < w else (w, k, cv, ci)
                        if key not in edges:
                            total = float("inf")
                            break
                        total += edges[key]
                    if total < best_cost:
                        best, best_cost = k, total
                if best is not None:
                    clique[w] = best
            members = {w: per_view[w][k] for w, k in clique.items()}
            point, kept, residuals = _prune_clique(members, cams, params)
            if len(kept) < 2:
                continue
            for w in kept:
                used.add((w, clique[w]))
            candidates.append(Candidate(cls, point, kept, {w: residuals[w] for w in kept}))
    fpv_cam = cams.get(FINEBIO_FPV_VIEW)
    fpv_unassigned = list(unassigned.get(FINEBIO_FPV_VIEW, ())) if fpv_cam is not None else []
    fpv_taken: set[int] = set()
    born: list[Candidate] = []
    for cand in sorted(candidates, key=lambda c: float(np.median(list(c.residuals.values())))):
        n_fixed = len(cand.members)
        if fpv_cam is not None:
            pixel = project(fpv_cam, cand.point)
            best, best_d = None, params.gates.handoff_px
            if pixel is not None:
                for o in fpv_unassigned:
                    if o.object_class != cand.object_class or o.index in fpv_taken:
                        continue
                    d = float(np.linalg.norm(o.point - pixel))
                    if d <= best_d:
                        best, best_d = o, d
            if best is not None:
                members = {**cand.members, FINEBIO_FPV_VIEW: best}
                point, residuals = _triangulate_members(members, cams, params)
                if residuals[FINEBIO_FPV_VIEW] <= params.gates.handoff_px and all(
                    residuals[v] <= params.gates.association_px for v in cand.members
                ):
                    cand = Candidate(cand.object_class, point, members, residuals)
                    fpv_taken.add(best.index)
        has_fpv = FINEBIO_FPV_VIEW in cand.members
        if n_fixed >= params.gates.birth_min_fixed_views:
            born.append(cand)
        elif has_fpv and n_fixed >= params.gates.birth_fixed_views_with_fpv:
            cand.fpv_rule = True
            born.append(cand)
    return born


# --------------------------------------------------------------------------- the tracker


@dataclass
class TrackerOutput:
    rows: list[Track3D]
    events: list[TrackEvent]
    residuals: list[ResidualRow]
    metrics: dict[str, Any]


class MultiviewTracker:
    def __init__(
        self,
        fixed_cams: dict[str, Camera],
        fpv_source: FpvCameraSource,
        params: TrackerParams | None = None,
        volumes: Sequence[ContainerVolume] = (),
    ) -> None:
        self.fixed_cams = dict(fixed_cams)
        self.fixed_views = tuple(fixed_cams)
        self.fpv_source = fpv_source
        self.params = params or TrackerParams()
        self.volumes = list(volumes)
        self.volume_by_id = {v.container_id: v for v in self.volumes}
        self.tracks: dict[str, Track] = {}
        self.rows: list[Track3D] = []
        self.events: list[TrackEvent] = []
        self.residual_rows: list[ResidualRow] = []
        # (view, SAM3 slot) -> [(frame, track id)]; detector slots are score ranks, carry no
        # identity, and are not recorded.
        self.slot_history: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
        self.sam3_slots: set[tuple[str, str]] = set()
        self.duplicate_pair_frames = 0
        self.ambiguities = 0
        self.reacquisition_latencies: list[int] = []
        self._next_id = 0
        self._previous_frame: int | None = None
        self.class_births: dict[str, int] = defaultdict(int)
        self.max_simultaneous: dict[str, int] = defaultdict(int)
        # -- extension counters (all zero with the extensions off)
        self.mover_track_ids: set[str] = set()
        self.mover_frames = 0
        self.single_view_ray_updates = 0
        self.reacquired_from: dict[str, int] = defaultdict(int)
        self.contained_by_container: dict[str, int] = defaultdict(int)
        self.held_by_hand: dict[str, int] = defaultdict(int)
        self.held_fell_back_to_coasting = 0
        self.group_track_ids: set[str] = set()
        self.footprint_group_ids: set[str] = set()
        self.splits = 0
        self.group_joins = 0
        self.candidates_merged_into_groups = 0
        self.absorbed_observations = 0
        self.max_group_size: dict[str, int] = defaultdict(int)

    # -- helpers

    def _new_id(self, cls: str) -> str:
        self._next_id += 1
        return f"{cls}-{self._next_id:03d}"

    def _event(self, frame: int, track_id: str, kind: str, **payload: Any) -> None:
        self.events.append(
            TrackEvent(frame_index=frame, track_id=track_id, kind=kind, payload=payload)
        )

    def live_tracks(self) -> list[Track]:
        return [t for t in self.tracks.values() if t.state in LIVE_STATES]

    def _cams_for(self, frame: int) -> dict[str, Camera]:
        cams = dict(self.fixed_cams)
        fpv = self.fpv_source(frame)
        if fpv is not None:
            cams[FINEBIO_FPV_VIEW] = fpv
        return cams

    def _is_group_class(self, cls: str) -> bool:
        return self.params.group_tracks and cls in self.params.group_classes

    def _containing_volume(
        self, point: np.ndarray, object_class: str | None = None
    ) -> ContainerVolume | None:
        """The first volume holding `point`; a container is never contained in itself."""
        for volume in self.volumes:
            if volume.object_class != object_class and volume.contains(point):
                return volume
        return None

    # -- the frame step

    def step(self, frame: int, rows: Iterable[FineBioObservation]) -> None:
        params = self.params
        dt = 1 if self._previous_frame is None else max(1, frame - self._previous_frame)
        self._previous_frame = frame
        cams = self._cams_for(frame)
        obs = select_observations(frame, rows, params)

        for t in self.live_tracks():
            self._predict(t, dt)

        assigned = self._associate(cams, obs)
        taken: set[tuple[str, int]] = set()
        already_support0 = [t for t in self.live_tracks() if t.state in SUPPORT0_STATES]
        if params.group_tracks and self.volumes:
            # Observations already assigned to an individual track are not the group's.
            taken |= {(v, o.index) for (_tid, v), o in assigned.items()}
            self._footprint_groups(frame, cams, obs, taken)
        for t in self.live_tracks():
            if t.state not in LOCALISED_STATES or t.group_container is not None:
                continue
            mine = {v: o for (tid, v), o in assigned.items() if tid == t.track_id}
            self._update(t, mine, cams, frame, dt, obs)
            for v, o in mine.items():
                taken.add((v, o.index))
        if params.group_tracks:
            self._gate_absorption(frame, cams, obs, taken)
        for t in already_support0:
            if t.state in SUPPORT0_STATES:
                self._coast(t, frame, dt)
        self._handoff_reseeds(frame, cams, obs)

        unassigned = {
            v: [o for o in items if (v, o.index) not in taken] for v, items in obs.tracked.items()
        }
        candidates = birth_candidates(unassigned, cams, self.fixed_views, params)
        if params.group_tracks:
            candidates = self._merge_group_candidates(candidates, cams)
        self._births_and_reacquisitions(frame, candidates, cams)

        duplicates = self._near_duplicates()
        live_by_class: dict[str, int] = defaultdict(int)
        for t in self.tracks.values():
            if t.state in LIVE_STATES or t.lost_frame == frame:
                self.rows.append(self._row(t, frame, cams, duplicates.get(t.track_id, ())))
            if t.state in LIVE_STATES:
                live_by_class[t.object_class] += 1
        for cls, n in live_by_class.items():
            self.max_simultaneous[cls] = max(self.max_simultaneous[cls], n)

    def _predict(self, t: Track, dt: int) -> None:
        """Stationary prior: the position stays, the uncertainty grows. With the motion model,
        a mover advances along its velocity with noise that grows with its speed; a contained
        track is bounded by its container and does not grow."""
        params = self.params
        if t.state == "contained":
            return
        noise = params.process_noise_cm * dt
        if params.motion_model and t.mover and t.state in (*LOCALISED_STATES, "coasting"):
            t.position = t.position + t.velocity * dt
            speed = float(np.linalg.norm(t.velocity))
            noise = float(np.hypot(noise, params.mover_noise_factor * speed * dt))
            if t.state == "coasting":
                t.velocity = t.velocity * params.mover_coast_damping**dt
            self.mover_frames += 1
        t.uncertainty_cm = min(params.max_uncertainty_cm, float(np.hypot(t.uncertainty_cm, noise)))

    def _associate(
        self, cams: dict[str, Camera], obs: FrameObservations
    ) -> dict[tuple[str, str], Obs]:
        pairs: list[tuple[float, str, str, Obs]] = []
        for t in self.live_tracks():
            if t.state not in LOCALISED_STATES or t.group_container is not None:
                continue
            for v, cam in cams.items():
                pixel = project(cam, t.position)
                if pixel is None:
                    continue
                gate = gate_px(t, cam, self.params)
                for o in obs.tracked.get(v, ()):
                    if o.object_class != t.object_class:
                        continue
                    d = float(np.linalg.norm(o.point - pixel))
                    if d <= gate:
                        pairs.append((d, t.track_id, v, o))
        assigned: dict[tuple[str, str], Obs] = {}
        used_obs: set[tuple[str, int]] = set()
        for d, tid, v, o in sorted(pairs, key=lambda p: p[0]):
            if (tid, v) in assigned or (v, o.index) in used_obs:
                continue
            assigned[(tid, v)] = o
            used_obs.add((v, o.index))
        return assigned

    def _update(
        self,
        t: Track,
        mine: dict[str, Obs],
        cams: dict[str, Camera],
        frame: int,
        dt: int,
        obs: FrameObservations | None = None,
    ) -> None:
        params = self.params
        disagreement = []
        for v, o in mine.items():
            previous = t.per_view_slot.get(v)
            if previous is not None and o.source in SAM3_SOURCES and previous != o.slot:
                disagreement.append(v)
        if disagreement:
            t.slot_disagreements += len(disagreement)
        t.slot_disagreement_views = tuple(sorted(disagreement))
        if len(mine) >= 2:
            point, kept, residuals = _prune_clique(mine, cams, params)
            if len(kept) >= 2:
                mine = kept
                depth = np.median([depth_cm(cams[v], point) for v in kept])
                f = np.median([cams[v].K[0, 0] for v in kept])
                meas_cm = max(
                    params.base_uncertainty_cm,
                    float(np.median(list(residuals.values()))) * depth / f,
                )
                prior_var = t.uncertainty_cm**2
                gain = prior_var / (prior_var + meas_cm**2)
                t.position = t.position + gain * (point - t.position)
                t.uncertainty_cm = float(np.sqrt((1 - gain) * prior_var))
                t.state = "observed"
                if params.motion_model:
                    self._update_velocity(t, frame)
                t.last_observed_frame = frame
                t.frames_unobserved = 0
                t.residuals = {v: round(residuals[v], 2) for v in kept}
            else:
                mine = {}
        if len(mine) == 1:
            (v, o) = next(iter(mine.items()))
            pixel = project(cams[v], t.position)
            t.residuals = (
                {v: round(float(np.linalg.norm(pixel - o.point)), 2)} if pixel is not None else {}
            )
            if params.motion_model and t.mover:
                # A mover seen in one view: lateral update along the ray at the predicted depth
                # (the velocity is left alone: measured on arm (b), updating it here cost more
                # ambiguities than it saved).
                t.position = ray_point(cams[v], o.point, t.position)
                self.single_view_ray_updates += 1
            t.state = "single_view"
            t.frames_unobserved = 0
            t.last_observed_frame = frame
        elif not mine:
            self._start_coasting(t, frame, dt, cams, obs)
            return
        t.support_views = tuple(sorted(mine))
        t.support_slots = {v: o.slot for v, o in mine.items()}
        for v, o in mine.items():
            self._record_support(t, v, o, cams[v], frame)

    def _update_velocity(self, t: Track, frame: int) -> None:
        """Smoothed finite difference of the filtered position over >= 2-view updates; the
        mover flag switches on above `mover_speed_cm_per_frame` and off below half of it."""
        params = self.params
        if t.last_observed_position is not None and frame > t.last_observed_frame:
            v_new = (t.position - t.last_observed_position) / (frame - t.last_observed_frame)
            alpha = params.velocity_smoothing
            t.velocity = (1.0 - alpha) * t.velocity + alpha * v_new
        t.last_observed_position = t.position.copy()
        speed = float(np.linalg.norm(t.velocity))
        # Symmetric confirmation: on after `mover_confirm_frames` updates above the threshold,
        # off after as many below half of it (a turning point is one or two slow frames, not
        # a stop).
        if t.mover:
            if speed < 0.5 * params.mover_speed_cm_per_frame:
                t.mover_frames += 1
                if t.mover_frames >= params.mover_confirm_frames:
                    t.mover = False
                    t.mover_frames = 0
                    t.velocity = np.zeros(3)
            else:
                t.mover_frames = 0
        elif speed > params.mover_speed_cm_per_frame:
            t.mover_frames += 1
            if t.mover_frames >= params.mover_confirm_frames:
                t.mover = True
                t.mover_frames = 0
                self.mover_track_ids.add(t.track_id)
        else:
            t.mover_frames = 0

    def _record_support(self, t: Track, view: str, o: Obs, cam: Camera, frame: int) -> None:
        t.per_view_last_seen[view] = frame
        t.per_view_slot[view] = o.slot
        if o.extent is not None:
            t.per_view_extent[view] = o.extent
        if o.source in SAM3_SOURCES:
            self.slot_history[(view, o.slot)].append((frame, t.track_id))
            self.sam3_slots.add((view, o.slot))
        self.residual_rows.append(
            ResidualRow(
                frame_index=frame,
                track_id=t.track_id,
                view=view,
                residual_px=t.residuals.get(view, 0.0),
                gate_px=gate_px(t, cam, self.params),
                slot=o.slot,
                source=o.source,
                confirmed=o.confirmed,
            )
        )

    # -- support 0: coasting, contained, held

    def _start_coasting(
        self,
        t: Track,
        frame: int,
        dt: int,
        cams: dict[str, Camera] | None = None,
        obs: FrameObservations | None = None,
    ) -> None:
        params = self.params
        if t.state not in SUPPORT0_STATES:
            volume = (
                self._containing_volume(t.position, t.object_class) if params.containers else None
            )
            hand = None
            if volume is None and params.held and cams is not None and obs is not None:
                hand = self._hand_holding(t, cams, obs)
            if volume is not None and self._is_group_class(t.object_class):
                group = self._footprint_group_for(t.object_class, volume.container_id)
                if group is not None and group.track_id != t.track_id:
                    self._join_group(t, group, frame, volume)
                    return
            if volume is not None:
                t.state = "contained"
                t.container_id = volume.container_id
                t.uncertainty_cm = min(
                    params.max_uncertainty_cm, max(t.uncertainty_cm, volume.radius_cm)
                )
                self.contained_by_container[volume.container_id] += 1
                self._event(
                    frame,
                    t.track_id,
                    "contained",
                    container_id=volume.container_id,
                    last_observed_frame=t.last_observed_frame,
                    position_cm=[round(float(x), 2) for x in t.position],
                    group_size=t.group_size,
                )
            elif hand is not None:
                hand_class, hand_track, views = hand
                t.state = "held"
                t.held_by = hand_track.track_id if hand_track is not None else None
                t.held_offset = t.position - hand_track.position if hand_track is not None else None
                self.held_by_hand[hand_class] += 1
                self._event(
                    frame,
                    t.track_id,
                    "held",
                    hand_class=hand_class,
                    held_by=t.held_by,
                    views=sorted(views),
                    last_observed_frame=t.last_observed_frame,
                )
            else:
                self._event(
                    frame, t.track_id, "coasting", last_observed_frame=t.last_observed_frame
                )
                t.state = "coasting"
        t.support_views = ()
        t.support_slots = {}
        t.residuals = {}
        self._coast(t, frame, dt)

    def _hand_holding(
        self, t: Track, cams: dict[str, Camera], obs: FrameObservations
    ) -> tuple[str, Track | None, list[str]] | None:
        """The hand class whose detector box holds the track's projection in >= `held_min_views`
        views, with the nearest live localised track of that class (None when the hands are
        not tracked)."""
        params = self.params
        views_by_hand: dict[str, list[str]] = defaultdict(list)
        for v, cam in cams.items():
            pixel = project(cam, t.position)
            if pixel is None:
                continue
            for d in obs.detector.get(v, ()):
                if d.object_class in params.hand_classes and d.box is not None:
                    box = d.box
                    if box[0] <= pixel[0] <= box[2] and box[1] <= pixel[1] <= box[3]:
                        views_by_hand[d.object_class].append(v)
        if not views_by_hand:
            return None
        hand_class, views = max(views_by_hand.items(), key=lambda kv: (len(kv[1]), kv[0]))
        if len(set(views)) < params.held_min_views:
            return None
        hands = [
            h
            for h in self.live_tracks()
            if h.object_class == hand_class and h.state in LOCALISED_STATES
        ]
        nearest = (
            min(hands, key=lambda h: float(np.linalg.norm(h.position - t.position)))
            if hands
            else None
        )
        return hand_class, nearest, sorted(set(views))

    def _coast(self, t: Track, frame: int, dt: int) -> None:
        params = self.params
        t.frames_unobserved = frame - t.last_observed_frame
        if t.state == "contained":
            volume = self.volume_by_id.get(t.container_id or "")
            if volume is not None:
                t.uncertainty_cm = min(
                    params.max_uncertainty_cm, max(t.uncertainty_cm, volume.radius_cm)
                )
            if (
                params.contained_timeout_frames
                and t.frames_unobserved > params.contained_timeout_frames
            ):
                self._lose(t, frame, from_state="contained")
            return
        if t.state == "held":
            hand = self.tracks.get(t.held_by or "")
            if hand is not None and hand.state in LOCALISED_STATES and t.held_offset is not None:
                t.position = hand.position + t.held_offset
            elif t.held_by is not None:
                # The hand track is gone or not localised: back to plain coasting.
                t.state = "coasting"
                t.held_by = None
                t.held_offset = None
                self.held_fell_back_to_coasting += 1
        t.uncertainty_cm = min(
            params.max_uncertainty_cm, t.uncertainty_cm + params.coast_growth_cm * dt
        )
        if t.frames_unobserved > params.coast_timeout_frames:
            self._lose(t, frame, from_state=t.state)

    def _lose(self, t: Track, frame: int, *, from_state: str) -> None:
        extra = {} if from_state == "coasting" else {"from_state": from_state}
        t.state = "lost"
        t.lost_frame = frame
        self._event(
            frame,
            t.track_id,
            "lost",
            frames_unobserved=t.frames_unobserved,
            last_observed_frame=t.last_observed_frame,
            **extra,
        )

    # -- group tracks

    def _footprint_group_for(self, cls: str, container_id: str) -> Track | None:
        for t in self.tracks.values():
            if (
                t.object_class == cls
                and t.group_container == container_id
                and t.state in LIVE_STATES
            ):
                return t
        return None

    def _join_group(self, t: Track, group: Track, frame: int, volume: ContainerVolume) -> None:
        """An individual of a group class whose support drops inside a footprint that already
        has a group track ends here and its identity joins the group (the plan's rule: no
        per-instance identity inside a rack)."""
        t.state = "lost"
        t.lost_frame = frame
        t.support_views = ()
        t.support_slots = {}
        t.residuals = {}
        t.frames_unobserved = frame - t.last_observed_frame
        self.group_joins += 1
        self._event(
            frame,
            t.track_id,
            "group_joined",
            group_track=group.track_id,
            container_id=volume.container_id,
            last_observed_frame=t.last_observed_frame,
        )

    def _footprint_groups(
        self,
        frame: int,
        cams: dict[str, Camera],
        obs: FrameObservations,
        taken: set[tuple[str, int]],
    ) -> None:
        """One group track per (group class, container footprint): every tracked observation
        of the class whose pixel lies inside the projected volume in a view belongs to it."""
        params = self.params
        for volume in self.volumes:
            polygons = {v: volume.polygon_px(cam) for v, cam in cams.items()}
            classes = sorted(
                {
                    o.object_class
                    for items in obs.tracked.values()
                    for o in items
                    if self._is_group_class(o.object_class)
                }
            )
            for cls in classes:
                inside: dict[str, list[Obs]] = {}
                for v, items in obs.tracked.items():
                    polygon = polygons.get(v)
                    if polygon is None:
                        continue
                    hits = [
                        o
                        for o in items
                        if o.object_class == cls
                        and (v, o.index) not in taken
                        and inside_convex_polygon(o.point, polygon)
                    ]
                    if hits:
                        inside[v] = hits
                group = self._footprint_group_for(cls, volume.container_id)
                if group is None:
                    if not inside:
                        continue
                    fixed = [v for v in inside if v != FINEBIO_FPV_VIEW]
                    enough = len(fixed) >= params.gates.birth_min_fixed_views or (
                        FINEBIO_FPV_VIEW in inside
                        and len(fixed) >= params.gates.birth_fixed_views_with_fpv
                    )
                    size = int(round(float(np.median([len(h) for h in inside.values()]))))
                    if not enough or size < 2:
                        continue
                    group = self._birth_group(frame, cls, volume, inside, cams)
                else:
                    self._group_update(group, volume, inside, cams, frame)
                if group is not None:
                    for v, hits in inside.items():
                        for o in hits:
                            taken.add((v, o.index))
                            self.absorbed_observations += 1

    def _group_means(
        self, inside: dict[str, list[Obs]], cams: dict[str, Camera]
    ) -> tuple[np.ndarray | None, dict[str, Obs], dict[str, float]]:
        """Triangulate the per-view mean pixel of the inside observations (pruned over the
        gate); None when fewer than two views agree."""
        means: dict[str, Obs] = {}
        for v, hits in inside.items():
            point = np.mean([o.point for o in hits], axis=0)
            first = hits[0]
            means[v] = Obs(
                view=v,
                point=point,
                object_class=first.object_class,
                slot=first.slot,
                source=first.source,
                detector_score=first.detector_score,
                sam3_score=first.sam3_score,
                box=first.box,
                confirmed=any(o.confirmed for o in hits),
                index=first.index,
            )
        if len(means) < 2:
            return None, {}, {}
        point, kept, residuals = _prune_clique(means, cams, self.params)
        if len(kept) < 2:
            return None, {}, {}
        return point, kept, {v: residuals[v] for v in kept}

    def _birth_group(
        self,
        frame: int,
        cls: str,
        volume: ContainerVolume,
        inside: dict[str, list[Obs]],
        cams: dict[str, Camera],
    ) -> Track:
        point, kept, residuals = self._group_means(inside, cams)
        basis = "triangulated_mean"
        if point is None or not volume.contains(point):
            point, basis = volume.centre_3d.copy(), "volume_centre"
            residuals = {}
        size = int(round(float(np.median([len(h) for h in inside.values()]))))
        t = Track(
            track_id=self._new_id(cls),
            object_class=cls,
            position=np.asarray(point, dtype=np.float64).copy(),
            uncertainty_cm=max(self.params.base_uncertainty_cm, volume.radius_cm / 2),
            state="observed",
            born_frame=frame,
            last_observed_frame=frame,
            group_size=size,
            group_container=volume.container_id,
        )
        t.support_views = tuple(sorted(inside))
        t.support_slots = {v: hits[0].slot for v, hits in inside.items()}
        t.residuals = {v: round(r, 2) for v, r in residuals.items()}
        for v, hits in inside.items():
            for o in hits:
                self._record_support(t, v, o, cams[v], frame)
        self.tracks[t.track_id] = t
        self.class_births[cls] += 1
        self.group_track_ids.add(t.track_id)
        self.footprint_group_ids.add(t.track_id)
        self.max_group_size[cls] = max(self.max_group_size[cls], size)
        self._event(
            frame,
            t.track_id,
            "birth",
            views=sorted(inside),
            residual_px={v: round(r, 2) for v, r in residuals.items()},
            point_cm=[round(float(x), 2) for x in t.position],
            fpv_rule=False,
            group_size=size,
            container_id=volume.container_id,
            members_per_view={v: len(h) for v, h in inside.items()},
            position_basis=basis,
        )
        self._event(
            frame,
            t.track_id,
            "group_formed",
            container_id=volume.container_id,
            group_size=size,
            members_per_view={v: len(h) for v, h in inside.items()},
        )
        return t

    def _group_update(
        self,
        t: Track,
        volume: ContainerVolume,
        inside: dict[str, list[Obs]],
        cams: dict[str, Camera],
        frame: int,
    ) -> None:
        params = self.params
        if not inside:
            if t.state in LOCALISED_STATES:
                self._start_coasting(t, frame, 1)
            return
        confirmed_views = sum(1 for hits in inside.values() if any(o.confirmed for o in hits))
        if t.state in SUPPORT0_STATES:
            needed = params.reacquire_min_views
            if len(inside) < needed or confirmed_views < needed:
                return
            latency = frame - t.last_observed_frame
            self.reacquisition_latencies.append(latency)
            self.reacquired_from[t.state] += 1
            self._event(
                frame,
                t.track_id,
                "reacquired",
                latency_frames=latency,
                views=sorted(inside),
                residual_px={},
                confirmed_views=confirmed_views,
                from_state=t.state,
                group_size=int(round(float(np.median([len(h) for h in inside.values()])))),
            )
            t.reacquisitions += 1
            t.container_id = None
            t.held_by = None
        point, kept, residuals = self._group_means(inside, cams)
        if point is not None and volume.contains(point):
            depth = np.median([depth_cm(cams[v], point) for v in kept])
            f = np.median([cams[v].K[0, 0] for v in kept])
            meas_cm = max(
                params.base_uncertainty_cm, float(np.median(list(residuals.values()))) * depth / f
            )
            prior_var = t.uncertainty_cm**2
            gain = prior_var / (prior_var + meas_cm**2)
            t.position = t.position + gain * (point - t.position)
            t.uncertainty_cm = max(
                params.base_uncertainty_cm, float(np.sqrt((1 - gain) * prior_var))
            )
            t.residuals = {v: round(r, 2) for v, r in residuals.items()}
        else:
            t.residuals = {}
        t.state = "observed" if len(inside) >= 2 else "single_view"
        t.last_observed_frame = frame
        t.frames_unobserved = 0
        t.group_size = int(round(float(np.median([len(h) for h in inside.values()]))))
        self.max_group_size[t.object_class] = max(self.max_group_size[t.object_class], t.group_size)
        t.support_views = tuple(sorted(inside))
        t.support_slots = {v: hits[0].slot for v, hits in inside.items()}
        t.slot_disagreement_views = ()
        for v, hits in inside.items():
            for o in hits:
                self._record_support(t, v, o, cams[v], frame)

    def _gate_absorption(
        self,
        frame: int,
        cams: dict[str, Camera],
        obs: FrameObservations,
        taken: set[tuple[str, int]],
    ) -> None:
        """A localised group-class track absorbs the unassigned same-class observations inside
        its own gate per view (one group id with a member count, not near-duplicate births)."""
        for t in self.live_tracks():
            if (
                t.state not in LOCALISED_STATES
                or t.group_container is not None
                or not self._is_group_class(t.object_class)
            ):
                continue
            counts: dict[str, int] = {}
            for v in t.support_views:
                cam = cams.get(v)
                if cam is None:
                    continue
                pixel = project(cam, t.position)
                if pixel is None:
                    continue
                gate = gate_px(t, cam, self.params)
                absorbed = 0
                for o in obs.tracked.get(v, ()):
                    if o.object_class != t.object_class or (v, o.index) in taken:
                        continue
                    if float(np.linalg.norm(o.point - pixel)) <= gate:
                        taken.add((v, o.index))
                        absorbed += 1
                        self.absorbed_observations += 1
                        self._record_support(t, v, o, cam, frame)
                counts[v] = absorbed
            extra = int(round(float(np.median(list(counts.values()))))) if counts else 0
            if extra >= 1:
                t.group_size = 1 + extra
                self.group_track_ids.add(t.track_id)
                self.max_group_size[t.object_class] = max(
                    self.max_group_size[t.object_class], t.group_size
                )
            elif t.group_container is None:
                t.group_size = None

    def _merge_group_candidates(
        self, candidates: list[Candidate], cams: dict[str, Camera]
    ) -> list[Candidate]:
        """Birth candidates of a group class within `duplicate_distance_cm` of each other are
        one candidate with a member count; one within that distance of a live localised
        same-class track is absorbed into it rather than born as a near duplicate."""
        params = self.params
        out: list[Candidate] = []
        used = [False] * len(candidates)
        for i, cand in enumerate(candidates):
            if used[i]:
                continue
            used[i] = True
            if not self._is_group_class(cand.object_class):
                out.append(cand)
                continue
            near_track = None
            for t in self.live_tracks():
                if (
                    t.object_class == cand.object_class
                    and t.state in LOCALISED_STATES
                    and np.linalg.norm(t.position - cand.point) <= params.duplicate_distance_cm
                ):
                    near_track = t
                    break
            if near_track is not None:
                near_track.group_size = (near_track.group_size or 1) + 1
                self.group_track_ids.add(near_track.track_id)
                self.candidates_merged_into_groups += 1
                self.max_group_size[cand.object_class] = max(
                    self.max_group_size[cand.object_class], near_track.group_size
                )
                continue
            merged = [cand]
            for j in range(i + 1, len(candidates)):
                other = candidates[j]
                if (
                    not used[j]
                    and other.object_class == cand.object_class
                    and np.linalg.norm(other.point - cand.point) <= params.duplicate_distance_cm
                ):
                    used[j] = True
                    merged.append(other)
            if len(merged) > 1:
                point = np.mean([c.point for c in merged], axis=0)
                cand = Candidate(
                    cand.object_class,
                    point,
                    dict(cand.members),
                    dict(cand.residuals),
                    cand.fpv_rule,
                    group_size=len(merged),
                )
                self.candidates_merged_into_groups += len(merged) - 1
            out.append(cand)
        return out

    def _split_source(self, cand: Candidate) -> Track | None:
        """The nearest live group track of the candidate's class within the split radius,
        when the candidate lies outside every container footprint."""
        params = self.params
        if not self._is_group_class(cand.object_class):
            return None
        if self._containing_volume(cand.point, cand.object_class) is not None:
            return None
        groups = [
            t
            for t in self.live_tracks()
            if t.object_class == cand.object_class
            and (t.group_size or 0) >= 2
            and np.linalg.norm(t.position - cand.point) <= params.group_split_radius_cm
        ]
        if not groups:
            return None
        return min(groups, key=lambda t: float(np.linalg.norm(t.position - cand.point)))

    # -- hand-off re-seeds

    def _handoff_reseeds(self, frame: int, cams: dict[str, Camera], obs: FrameObservations) -> None:
        params = self.params
        K = params.handoff_after_frames
        for t in self.live_tracks():
            if t.state != "observed":
                continue
            for v, cam in cams.items():
                last_seen = t.per_view_last_seen.get(v, t.born_frame)
                missing = frame - last_seen
                since_last = frame - t.last_reseed_frame.get(v, -(10**9))
                if missing < K or since_last < params.handoff_repeat_frames:
                    continue
                pixel = project(cam, t.position)
                if pixel is None or not _in_image(pixel, cam):
                    continue
                t.last_reseed_frame[v] = frame
                detector = None
                search = reseed_gate_px(t, cam, params)
                for d in obs.detector.get(v, ()):
                    if d.object_class != t.object_class:
                        continue
                    dist = float(np.linalg.norm(d.point - pixel))
                    if dist <= search and (detector is None or dist < detector[0]):
                        detector = (dist, d)
                if detector is not None:
                    d = detector[1]
                    self._event(
                        frame,
                        t.track_id,
                        "detector_reseed",
                        view=v,
                        box_xyxy_px=[float(x) for x in d.box],
                        detector_score=d.detector_score,
                        slot=d.slot,
                        point_px=[float(x) for x in pixel],
                        residual_px=round(detector[0], 2),
                        frames_missing=missing,
                        provenance="detector_reseed",
                    )
                    continue
                w, h = t.per_view_extent.get(v, params.default_reseed_box_px)
                box = [
                    float(np.clip(pixel[0] - w / 2, 0, cam.size[0])),
                    float(np.clip(pixel[1] - h / 2, 0, cam.size[1])),
                    float(np.clip(pixel[0] + w / 2, 0, cam.size[0])),
                    float(np.clip(pixel[1] + h / 2, 0, cam.size[1])),
                ]
                self._event(
                    frame,
                    t.track_id,
                    "handoff_reseed",
                    view=v,
                    box_xyxy_px=box,
                    point_px=[float(x) for x in pixel],
                    box_from="last_extent" if v in t.per_view_extent else "default",
                    frames_missing=missing,
                    uncertainty_cm=round(t.uncertainty_cm, 2),
                    gate_px=round(gate_px(t, cam, params), 1),
                    provenance="track_reproject",
                )

    # -- births and re-acquisitions

    def _candidate_hits(self, t: Track, cand: Candidate, cams: dict[str, Camera]) -> int:
        """Views in which the candidate's observation lies inside the track's gate; for a
        contained track a candidate inside the container volume counts in every view."""
        if t.state == "contained":
            volume = self.volume_by_id.get(t.container_id or "")
            if volume is not None and volume.contains(cand.point):
                return len(cand.members)
        hits = 0
        for v, o in cand.members.items():
            pixel = project(cams[v], t.position)
            if pixel is not None and np.linalg.norm(pixel - o.point) <= gate_px(
                t, cams[v], self.params
            ):
                hits += 1
        return hits

    def _births_and_reacquisitions(
        self, frame: int, candidates: list[Candidate], cams: dict[str, Camera]
    ) -> None:
        params = self.params
        waiting = [t for t in self.tracks.values() if t.state in SUPPORT0_STATES]
        inside: dict[int, list[str]] = defaultdict(list)  # candidate index -> waiting ids
        for ci, cand in enumerate(candidates):
            for t in waiting:
                if t.object_class != cand.object_class:
                    continue
                if self._candidate_hits(t, cand, cams) >= params.reacquire_min_views:
                    inside[ci].append(t.track_id)
        per_track: dict[str, list[int]] = defaultdict(list)
        for ci, ids in inside.items():
            for tid in ids:
                per_track[tid].append(ci)
        for ci, cand in enumerate(candidates):
            ids = inside.get(ci, [])
            if not ids:
                self._birth(frame, cand, cams)
                continue
            ambiguous = len(ids) > 1 or any(len(per_track[tid]) > 1 for tid in ids)
            confirmed_views = sum(o.confirmed for o in cand.members.values())
            if ambiguous:
                new = self._birth(frame, cand, cams, possibly_same_as=tuple(sorted(ids)))
                self._event(
                    frame,
                    new.track_id,
                    "ambiguous",
                    coasting_tracks=sorted(ids),
                    candidates_for_those_tracks=sorted({c for tid in ids for c in per_track[tid]}),
                    confirmed_views=confirmed_views,
                )
                continue
            (tid,) = ids
            if confirmed_views < params.reacquire_min_views:
                self._birth(
                    frame,
                    cand,
                    cams,
                    possibly_same_as=(tid,),
                    unconfirmed_reacquisition_of=tid,
                    confirmed_views=confirmed_views,
                )
                continue
            self._resume(frame, self.tracks[tid], cand, cams)
        ambiguous_tracks = {tid for tid, cis in per_track.items() if len(cis) > 1}
        ambiguous_tracks |= {tid for ci, ids in inside.items() if len(ids) > 1 for tid in ids}
        self.ambiguities += len(ambiguous_tracks)

    def _birth(
        self,
        frame: int,
        cand: Candidate,
        cams: dict[str, Camera],
        possibly_same_as: tuple[str, ...] = (),
        **payload: Any,
    ) -> Track:
        t = Track(
            track_id=self._new_id(cand.object_class),
            object_class=cand.object_class,
            position=cand.point.copy(),
            uncertainty_cm=self.params.base_uncertainty_cm,
            state="observed",
            born_frame=frame,
            last_observed_frame=frame,
            possibly_same_as=possibly_same_as,
        )
        extra: dict[str, Any] = {}
        if cand.group_size is not None and cand.group_size >= 2:
            t.group_size = cand.group_size
            self.group_track_ids.add(t.track_id)
            self.max_group_size[t.object_class] = max(
                self.max_group_size[t.object_class], cand.group_size
            )
            extra["group_size"] = cand.group_size
        source = self._split_source(cand) if self.params.group_tracks else None
        if source is not None:
            t.split_from = source.track_id
            self.splits += 1
            extra["split_from"] = source.track_id
        if self.params.motion_model:
            t.last_observed_position = t.position.copy()
        self._attach(t, cand, cams, frame)
        self.tracks[t.track_id] = t
        self.class_births[cand.object_class] += 1
        self._event(
            frame,
            t.track_id,
            "birth",
            views=sorted(cand.members),
            residual_px={v: round(r, 2) for v, r in cand.residuals.items()},
            point_cm=[round(float(x), 2) for x in cand.point],
            fpv_rule=cand.fpv_rule,
            **({"possibly_same_as": list(possibly_same_as)} if possibly_same_as else {}),
            **payload,
            **extra,
        )
        if source is not None:
            self._event(
                frame,
                t.track_id,
                "group_split",
                split_from=source.track_id,
                group_size_before=source.group_size,
                distance_cm=round(float(np.linalg.norm(source.position - cand.point)), 2),
            )
        return t

    def _resume(self, frame: int, t: Track, cand: Candidate, cams: dict[str, Camera]) -> None:
        latency = frame - t.last_observed_frame
        from_state = t.state
        t.position = cand.point.copy()
        t.uncertainty_cm = self.params.base_uncertainty_cm
        t.state = "observed"
        t.last_observed_frame = frame
        t.frames_unobserved = 0
        t.reacquisitions += 1
        extra: dict[str, Any] = {}
        if from_state != "coasting":
            extra["from_state"] = from_state
            self.reacquired_from[from_state] += 1
        t.container_id = None
        t.held_by = None
        t.held_offset = None
        if self.params.motion_model:
            t.velocity = np.zeros(3)
            t.mover = False
            t.last_observed_position = t.position.copy()
        if cand.group_size is not None and cand.group_size >= 2:
            t.group_size = cand.group_size
            extra["group_size"] = cand.group_size
        self._attach(t, cand, cams, frame)
        self.reacquisition_latencies.append(latency)
        self._event(
            frame,
            t.track_id,
            "reacquired",
            latency_frames=latency,
            views=sorted(cand.members),
            residual_px={v: round(r, 2) for v, r in cand.residuals.items()},
            confirmed_views=sum(o.confirmed for o in cand.members.values()),
            **extra,
        )

    def _attach(self, t: Track, cand: Candidate, cams: dict[str, Camera], frame: int) -> None:
        t.support_views = tuple(sorted(cand.members))
        t.support_slots = {v: o.slot for v, o in cand.members.items()}
        t.residuals = {v: round(r, 2) for v, r in cand.residuals.items()}
        t.slot_disagreement_views = ()
        for v, o in cand.members.items():
            self._record_support(t, v, o, cams[v], frame)

    # -- rows and metrics

    def _confidence(self, t: Track, cams: dict[str, Camera]) -> tuple[float, bool]:
        params = self.params
        if t.state == "lost":
            return 0.0, True
        if t.state in SUPPORT0_STATES:
            decay = max(0.0, 1.0 - t.frames_unobserved / max(1, params.coast_timeout_frames))
            return round(0.5 * decay, 3), True
        support = len(t.support_views) / max(2, len(cams))
        residual = float(np.median(list(t.residuals.values()))) if t.residuals else 0.0
        residual_term = float(np.clip(1.0 - residual / params.gates.handoff_px, 0.0, 1.0))
        slot_term = 0.5 if t.slot_disagreement_views else 1.0
        confidence = round(float(np.clip(support, 0, 1)) * residual_term * slot_term, 3)
        abstain = (
            t.state != "observed"
            or bool(t.slot_disagreement_views)
            or confidence < params.abstain_below_confidence
        )
        return confidence, abstain

    def _near_duplicates(self) -> dict[str, tuple[str, ...]]:
        """Live same-class tracks within `duplicate_distance_cm` of each other: two ids for
        what may be one object, shown on both rows and counted, never merged by guessing."""
        near: dict[str, set[str]] = defaultdict(set)
        live = [t for t in self.live_tracks() if t.state in LOCALISED_STATES]
        for a, b in combinations(live, 2):
            if a.object_class != b.object_class:
                continue
            if np.linalg.norm(a.position - b.position) <= self.params.duplicate_distance_cm:
                near[a.track_id].add(b.track_id)
                near[b.track_id].add(a.track_id)
                self.duplicate_pair_frames += 1
        return {tid: tuple(sorted(ids)) for tid, ids in near.items()}

    def _row(
        self, t: Track, frame: int, cams: dict[str, Camera], duplicates: tuple[str, ...] = ()
    ) -> Track3D:
        confidence, abstain = self._confidence(t, cams)
        possibly = tuple(sorted(set(t.possibly_same_as) | set(duplicates)))
        return Track3D(
            frame_index=frame,
            track_id=t.track_id,
            object_class=t.object_class,
            position_cm=tuple(round(float(x), 3) for x in t.position),
            uncertainty_cm=round(float(t.uncertainty_cm), 3),
            support_views=t.support_views,
            state=t.state,
            confidence=confidence,
            abstain=abstain or bool(duplicates),
            possibly_same_as=possibly,
            residual_px=dict(t.residuals),
            support_slots=dict(t.support_slots),
            slot_disagreement_views=t.slot_disagreement_views,
            frames_unobserved=t.frames_unobserved,
            group_size=t.group_size,
            split_from=t.split_from,
            container_id=t.container_id if t.state == "contained" else None,
            held_by=t.held_by if t.state == "held" else None,
        )

    def identity_metrics(self, reference_labels: dict[str, str] | None = None) -> dict[str, Any]:
        """Id switches against a reference labelling (`"view/slot" -> identity`) when given,
        else against the SAM3 per-view slots as a proxy (each SAM3 slot its own identity);
        fragmentation, re-acquisition latency, ambiguities, births and losses."""
        sequences: dict[str, list[tuple[int, str]]] = defaultdict(list)
        used_reference = reference_labels is not None
        for (view, slot), history in self.slot_history.items():
            key = f"{view}/{slot}"
            if used_reference:
                identity = reference_labels.get(key)
                if identity is None:
                    continue
            elif (view, slot) in self.sam3_slots:
                identity = key
            else:
                continue
            sequences[identity].extend(history)
        switches, per_identity = 0, {}
        for identity, history in sequences.items():
            ordered = [tid for _, tid in sorted(history)]
            count = sum(1 for a, b in zip(ordered, ordered[1:]) if a != b)
            per_identity[identity] = {"frames": len(ordered), "id_switches": count}
            switches += count
        per_class = {}
        for cls, births in self.class_births.items():
            per_class[cls] = {
                "tracks_born": births,
                "max_simultaneous": self.max_simultaneous.get(cls, 0),
                "fragmentation": max(0, births - self.max_simultaneous.get(cls, 0)),
                "tracks_lost": sum(
                    1 for t in self.tracks.values() if t.object_class == cls and t.state == "lost"
                ),
            }
        latencies = self.reacquisition_latencies
        return {
            "tracks_born": len(self.tracks),
            "tracks_lost": sum(1 for t in self.tracks.values() if t.state == "lost"),
            "tracks_live_at_end": sum(1 for t in self.tracks.values() if t.state in LIVE_STATES),
            "reacquisitions": len(latencies),
            "reacquisition_latency_frames": {
                "median": float(np.median(latencies)) if latencies else None,
                "max": max(latencies) if latencies else None,
            },
            "ambiguities": self.ambiguities,
            "duplicate_pair_frames": self.duplicate_pair_frames,
            "fragmentation": sum(c["fragmentation"] for c in per_class.values()),
            "slot_disagreements": sum(t.slot_disagreements for t in self.tracks.values()),
            "id_switches": switches,
            "id_switch_reference": "provided_labels" if used_reference else "sam3_slots_proxy",
            "id_switch_note": (
                "against the labels given"
                if used_reference
                else "proxy: each SAM3 per-view slot is treated as one identity; detector slots "
                "are score ranks, not identities, and are excluded"
            ),
            "per_identity": per_identity,
            "per_class": per_class,
            "events": dict(sorted(self._event_counts().items())),
            "params": self.params.as_dict(),
            "extensions": self.extension_metrics(),
        }

    def extension_metrics(self) -> dict[str, Any]:
        """What each extension did (zeros with them off)."""
        by_class_movers: dict[str, int] = defaultdict(int)
        for tid in self.mover_track_ids:
            by_class_movers[self.tracks[tid].object_class] += 1
        contained_live = [t.track_id for t in self.tracks.values() if t.state == "contained"]
        return {
            "enabled": {
                "motion_model": self.params.motion_model,
                "containers": bool(self.params.containers),
                "group_tracks": self.params.group_tracks,
                "held": self.params.held,
            },
            "motion_model": {
                "mover_tracks": len(self.mover_track_ids),
                "mover_tracks_by_class": dict(sorted(by_class_movers.items())),
                "mover_frames": self.mover_frames,
                "single_view_ray_updates": self.single_view_ray_updates,
            },
            "contained": {
                "episodes": sum(self.contained_by_container.values()),
                "by_container": dict(sorted(self.contained_by_container.items())),
                "reacquired": self.reacquired_from.get("contained", 0),
                "live_at_end": len(contained_live),
                "volumes": [v.summary() for v in self.volumes],
            },
            "held": {
                "episodes": sum(self.held_by_hand.values()),
                "by_hand": dict(sorted(self.held_by_hand.items())),
                "reacquired": self.reacquired_from.get("held", 0),
                "fell_back_to_coasting": self.held_fell_back_to_coasting,
            },
            "group_tracks": {
                "group_tracks_formed": len(self.group_track_ids),
                "footprint_groups": len(self.footprint_group_ids),
                "splits": self.splits,
                "individuals_joined_a_group": self.group_joins,
                "candidates_merged": self.candidates_merged_into_groups,
                "observations_absorbed": self.absorbed_observations,
                "max_group_size_by_class": dict(sorted(self.max_group_size.items())),
            },
        }

    def _event_counts(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for e in self.events:
            counts[e.kind] += 1
        return dict(counts)


def run_tracker(
    rows: Iterable[FineBioObservation],
    fixed_cams: dict[str, Camera],
    fpv_source: FpvCameraSource,
    frames: Sequence[int] | None = None,
    params: TrackerParams | None = None,
    reference_labels: dict[str, str] | None = None,
    rig_static: dict[str, Sequence[float]] | None = None,
) -> TrackerOutput:
    """Run the tracker over `frames` (default: every frame with a row). With
    `params.containers` set, the container volumes are built from the detector rows first
    (`build_container_volumes`; `rig_static` = class -> point_cm from the rig, optional) and
    their report lands in `metrics["extensions"]["contained"]`."""
    params = params or TrackerParams()
    by_frame: dict[int, list[FineBioObservation]] = defaultdict(list)
    for row in rows:
        by_frame[row.frame_index].append(row)
    frame_list = list(frames) if frames is not None else sorted(by_frame)
    volumes: list[ContainerVolume] = []
    volume_report: dict[str, Any] | None = None
    if params.containers:
        volumes, volume_report = build_container_volumes(
            (r for f in frame_list for r in by_frame.get(f, ())),
            fixed_cams,
            params.containers,
            params,
            rig_static,
        )
    tracker = MultiviewTracker(fixed_cams, fpv_source, params, volumes)
    for frame in frame_list:
        tracker.step(frame, by_frame.get(frame, []))
    metrics = tracker.identity_metrics(reference_labels)
    if volume_report is not None:
        metrics["extensions"]["contained"]["volume_report"] = volume_report
    return TrackerOutput(
        rows=tracker.rows,
        events=tracker.events,
        residuals=tracker.residual_rows,
        metrics=metrics,
    )


def parse_container_classes(spec: str) -> tuple[str, ...]:
    """`--containers`: a comma-separated class list, or the path of a clip config whose
    `containers` list is used."""
    path = Path(spec)
    if path.suffix == ".json" and path.is_file():
        doc = json.loads(path.read_text(encoding="utf-8"))
        return tuple(str(c) for c in doc.get("containers", ()))
    return tuple(c.strip() for c in spec.split(",") if c.strip())


def parse_heights(spec: str | None) -> tuple[tuple[str, float], ...]:
    """`--container-heights-cm class=cm,...` over the defaults."""
    heights = dict(DEFAULT_CONTAINER_HEIGHTS_CM)
    if spec:
        for item in spec.split(","):
            if not item.strip():
                continue
            cls, _, value = item.partition("=")
            heights[cls.strip()] = float(value)
    return tuple(sorted(heights.items()))


def rig_static_points(rig: dict[str, Any]) -> dict[str, Sequence[float]]:
    """class -> point_cm from a rig output's `static` list (empty when absent)."""
    return {
        str(s["class"]): tuple(float(v) for v in s["point_cm"])
        for s in rig.get("static", ())
        if "class" in s and "point_cm" in s
    }


# --------------------------------------------------------------------------- cli


def residual_summary(output: TrackerOutput) -> dict[str, Any]:
    per: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in output.residuals:
        cls = r.track_id.rsplit("-", 1)[0]
        per[(cls, r.view)].append(r.residual_px)
    out: dict[str, Any] = defaultdict(dict)
    for (cls, view), values in sorted(per.items()):
        out[cls][view] = {
            "n": len(values),
            "median": round(float(np.median(values)), 2),
            "p90": round(float(np.percentile(values, 90)), 2),
        }
    return dict(out)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--fixtures", type=Path, default=None, help="fixture directory")
    parser.add_argument("--observations", type=Path, default=None)
    parser.add_argument("--cameras", type=Path, default=None)
    parser.add_argument("--fpv-poses", type=Path, default=None)
    parser.add_argument(
        "--gates",
        type=Path,
        default=None,
        help="rig output JSON with a `gates` block (association_px, handoff_px, "
        "birth_min_fixed_views, birth_fixed_views_with_fpv); CLI flags override it",
    )
    parser.add_argument("--association-px", type=float, default=None)
    parser.add_argument("--handoff-px", type=float, default=None)
    parser.add_argument("--birth-min-fixed-views", type=int, default=None)
    parser.add_argument("--birth-fixed-views-with-fpv", type=int, default=None)
    parser.add_argument("--coast-timeout", type=int, default=TrackerParams.coast_timeout_frames)
    parser.add_argument("--handoff-after", type=int, default=TrackerParams.handoff_after_frames)
    parser.add_argument(
        "--reacquire-min-views", type=int, default=TrackerParams.reacquire_min_views
    )
    parser.add_argument("--min-score", type=float, default=TrackerParams.min_detector_score)
    parser.add_argument("--process-noise-cm", type=float, default=TrackerParams.process_noise_cm)
    parser.add_argument("--coast-growth-cm", type=float, default=TrackerParams.coast_growth_cm)
    parser.add_argument("--fpv-weight", type=float, default=TrackerParams.fpv_weight)
    parser.add_argument("--source", choices=("auto", "detector", "sam3"), default="auto")
    parser.add_argument("--frames", default="all", help="start:count, a-b, list, all, consecutive")
    parser.add_argument(
        "--reference-labels",
        type=Path,
        default=None,
        help='JSON {"view/slot": identity} for id switches; default: SAM3 slots as a proxy',
    )
    parser.add_argument("--output", type=Path, required=True)
    ext = parser.add_argument_group(
        "extensions (p3-tracker-ext; all off by default, the core reproduces byte for byte)"
    )
    ext.add_argument(
        "--motion-model",
        action="store_true",
        help="constant-velocity state for tracks whose speed exceeds --mover-speed",
    )
    ext.add_argument(
        "--mover-speed",
        type=float,
        default=TrackerParams.mover_speed_cm_per_frame,
        help="cm per frame above which a track is a mover (off below half of it)",
    )
    ext.add_argument("--mover-confirm-frames", type=int, default=TrackerParams.mover_confirm_frames)
    ext.add_argument("--mover-noise-factor", type=float, default=TrackerParams.mover_noise_factor)
    ext.add_argument(
        "--containers",
        default=None,
        help="comma-separated container classes, or a clip config JSON with a `containers` "
        "list; volumes are built from the detector rows (+ the rig's static points via --gates)",
    )
    ext.add_argument(
        "--container-heights-cm",
        default=None,
        help="class=cm,... over the defaults (rack 6, centrifuge 12, ...)",
    )
    ext.add_argument(
        "--contained-timeout",
        type=int,
        default=TrackerParams.contained_timeout_frames,
        help="frames before a contained track is lost; 0 = never (the plan's rule)",
    )
    ext.add_argument(
        "--group-tracks",
        action="store_true",
        help="identical-instance group tracks (one id per class and footprint or gate)",
    )
    ext.add_argument(
        "--group-classes",
        default=",".join(GROUP_CLASSES),
        help="comma-separated classes tracked as groups",
    )
    ext.add_argument(
        "--group-split-radius-cm",
        type=float,
        default=TrackerParams.group_split_radius_cm,
    )
    ext.add_argument(
        "--held",
        action="store_true",
        help="support 0 inside a hand box in >= --held-min-views views -> held, follows the hand",
    )
    ext.add_argument("--held-min-views", type=int, default=TrackerParams.held_min_views)
    args = parser.parse_args(argv)

    if args.fixtures is not None:
        observations = args.fixtures / "observations.jsonl"
        cameras = args.fixtures / "cameras.json"
        fpv_json = args.fixtures / "fpv_poses.json"
    else:
        if args.observations is None or args.cameras is None:
            raise SystemExit("give --fixtures <dir> or both --observations and --cameras")
        observations, cameras, fpv_json = args.observations, args.cameras, args.fpv_poses
    config = read_camera_config(cameras)
    rows = list(read_jsonl(observations, FineBioObservation))
    fpv_source = resolve_fpv_source(config, fpv_json)
    gates = Gates()
    rig_static: dict[str, Sequence[float]] = {}
    if args.gates is not None:
        rig = json.loads(args.gates.read_text(encoding="utf-8"))
        gates = Gates.from_rig(rig)
        rig_static = rig_static_points(rig)
    overrides = {
        "association_px": args.association_px,
        "handoff_px": args.handoff_px,
        "birth_min_fixed_views": args.birth_min_fixed_views,
        "birth_fixed_views_with_fpv": args.birth_fixed_views_with_fpv,
    }
    gates = Gates(**{**gates.__dict__, **{k: v for k, v in overrides.items() if v is not None}})
    params = TrackerParams(
        gates=gates,
        observation_source=args.source,
        min_detector_score=args.min_score,
        coast_timeout_frames=args.coast_timeout,
        handoff_after_frames=args.handoff_after,
        reacquire_min_views=args.reacquire_min_views,
        process_noise_cm=args.process_noise_cm,
        coast_growth_cm=args.coast_growth_cm,
        fpv_weight=args.fpv_weight,
        motion_model=args.motion_model,
        mover_speed_cm_per_frame=args.mover_speed,
        mover_confirm_frames=args.mover_confirm_frames,
        mover_noise_factor=args.mover_noise_factor,
        containers=parse_container_classes(args.containers) if args.containers else (),
        container_heights_cm=parse_heights(args.container_heights_cm),
        contained_timeout_frames=args.contained_timeout,
        group_tracks=args.group_tracks,
        group_classes=tuple(c for c in args.group_classes.split(",") if c),
        group_split_radius_cm=args.group_split_radius_cm,
        held=args.held,
        held_min_views=args.held_min_views,
    )
    frames = parse_frames(args.frames, {r.frame_index for r in rows})
    labels = (
        json.loads(args.reference_labels.read_text(encoding="utf-8"))
        if args.reference_labels is not None
        else None
    )
    output = run_tracker(
        rows,
        cameras_from_config(config),
        fpv_source,
        frames,
        params,
        labels,
        rig_static=rig_static or None,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output.rows, args.output / "tracks.jsonl")
    write_jsonl(output.events, args.output / "events.jsonl")
    write_jsonl(output.residuals, args.output / "residuals.jsonl")
    metrics = {
        **output.metrics,
        "trial": config.trial,
        "frames": {
            "count": len(frames),
            "first": frames[0] if frames else None,
            "last": frames[-1] if frames else None,
        },
        "gates_source": str(args.gates) if args.gates is not None else "cli_defaults",
        "residual_px_by_class_and_view": residual_summary(output),
    }
    (args.output / "identity_metrics.json").write_text(
        json.dumps(metrics, indent=1) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: metrics[k]
                for k in (
                    "tracks_born",
                    "tracks_lost",
                    "tracks_live_at_end",
                    "reacquisitions",
                    "ambiguities",
                    "fragmentation",
                    "id_switches",
                    "events",
                )
            },
            indent=1,
        ),
        file=sys.stdout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
