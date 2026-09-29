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
* `--line-classes pipette` (`p2-tracker-lines`, Sep 28): the four pipette classes are one
  geometric class of 3D line segments fitted from the per-view mask axes
  (`multiview_lines`), the observed class kept as a per-track vote (`observed_class`) and the
  colour vote as a hook (`colour_identity`). Birth pairs views on line plausibility and fits
  the clique's line; association gates each view on the perpendicular distance of its axis to
  the projected predicted line and the multi-view fit on `line_distance`; the endpoints come
  from the soft length prior (`configs/finebio/pipettes.json`); a mask wider than the track's
  running width, longer than the prior or too rough is dropped as merged; with
  `--motion-model` both endpoints carry a velocity; with `--held` the hand holds the butt.
  Evidence: 61 blue-pipette ids on trial 1 from box centres that are different places on a
  23 cm shaft in T1, T4 and the head camera.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
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
from .multiview_lines import (
    END_TOLERANCE_CM,
    AxisObs,
    Line3D,
    axis_residual,
    fit_line,
    is_elongated,
    line_distance,
    line_endpoints,
    loo_residual,
    plane_from_axis,
    point_residual,
    ray_from_point,
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
# p2-tracker-lines: the pipette classes `--line-classes pipette` expands to, the length prior
# the stand slice wrote, and the per-frame process noise on a line's direction (high: a held
# pipette swings between the tube and the plate, so the measured direction is trusted over the
# predicted one; a measured direction is worth about `LINE_DIRECTION_MEAS_DEG`).
LINE_CLASS_SHORTHANDS: dict[str, tuple[str, ...]] = {
    "pipette": ("blue_pipette", "yellow_pipette", "red_pipette", "8_channel_pipette"),
}
DEFAULT_LINE_PRIOR_PATH = "configs/finebio/pipettes.json"
LINE_DIRECTION_NOISE_DEG = 20.0
LINE_DIRECTION_MEAS_DEG = 2.0
# A hand resolves tip and butt when it is within this of one end and that end is nearer than
# the other by the margin.
LINE_HAND_REACH_CM = 15.0
LINE_HAND_MARGIN_CM = 3.0
# A pipette seen by a compact view alone gives a ray, not an extent; a view's extent shorter
# than this is not used to decide which end the prior reconstructs.
LINE_MIN_VIEW_EXTENT_CM = 1.0


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
    # -- p2-tracker-lines (Sep 28): the classes tracked as one geometric class of 3D line
    # segments (empty = off, no code path changes), the name of that class, the length prior
    # file, the line gates (angle between the predicted and the fitted line; perpendicular
    # distance at the midpoint, inflated by the track's uncertainty as `gate_px` is), the
    # smallest plane-pair angle a fit accepts, and the merged-mask rejections: a view whose mask
    # width exceeds the factor times the track's running median width in that view, whose own
    # extent on the line exceeds the factor times the prior, or whose skeleton residual exceeds
    # the maximum, is dropped from that frame's fit.
    line_classes: tuple[str, ...] = ()
    line_geometric_class: str = "pipette"
    line_prior_path: str = DEFAULT_LINE_PRIOR_PATH
    line_gate_deg: float = 15.0
    line_gate_cm: float = 6.0
    line_min_pair_angle_deg: float = 5.0
    line_merged_width_factor: float = 1.6
    line_merged_extent_factor: float = 1.2
    line_max_axis_residual_px: float = 25.0

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
        return bool(
            self.motion_model
            or self.containers
            or self.group_tracks
            or self.held
            or self.line_classes
        )

    def container_height(self, cls: str) -> float:
        return dict(self.container_heights_cm).get(cls, self.default_container_height_cm)

    def is_line_class(self, cls: str) -> bool:
        return bool(self.line_classes) and cls in self.line_classes


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


# --------------------------------------------------------------------------- line prior


@dataclass(frozen=True)
class LinePrior:
    """The soft length prior of the line classes (`configs/finebio/pipettes.json`, written by
    the stand slice): the shared length, the spread that serves as its tolerance, and the
    per-class medians where the stand measured a class (the median over the trials that did).
    `length_for(cls)` is the class median when the class is known, else the shared value."""

    length_cm: float
    spread_cm: float
    per_class: dict[str, float] = field(default_factory=dict)
    source: str = "default"

    def length_for(self, cls: str | None) -> float:
        if cls is not None and cls in self.per_class:
            return self.per_class[cls]
        return self.length_cm

    def as_dict(self) -> dict[str, Any]:
        return {
            "length_cm": self.length_cm,
            "spread_cm": self.spread_cm,
            "per_class": dict(sorted(self.per_class.items())),
            "source": self.source,
        }


def load_line_prior(path: str | Path) -> LinePrior:
    """Read the pipettes config; a relative path that is not found from the working directory
    is tried from the repository root, so tests and the CLI resolve the default alike."""
    candidate = Path(path)
    if not candidate.is_file() and not candidate.is_absolute():
        from_root = Path(__file__).resolve().parents[2] / candidate
        if from_root.is_file():
            candidate = from_root
    doc = json.loads(candidate.read_text(encoding="utf-8"))
    per_trial = (doc.get("provenance") or {}).get("per_trial") or {}
    medians: dict[str, list[float]] = defaultdict(list)
    for trial in per_trial.values():
        for cls, value in (trial.get("per_pipette_median_cm") or {}).items():
            medians[cls].append(float(value))
    return LinePrior(
        length_cm=float(doc["length_cm"]),
        spread_cm=float(doc.get("length_spread_cm", 0.0)),
        per_class={cls: float(np.median(v)) for cls, v in medians.items()},
        source=str(path),
    )


def parse_line_classes(spec: str | None) -> tuple[str, ...]:
    """`--line-classes`: a comma-separated class list; `pipette` expands to the four pipette
    classes (`LINE_CLASS_SHORTHANDS`)."""
    if not spec:
        return ()
    out: list[str] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        out.extend(LINE_CLASS_SHORTHANDS.get(item, (item,)))
    return tuple(dict.fromkeys(out))


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
    # -- p2-tracker-lines: the mask axis fields as the row carries them (None without a mask
    # axis), and the row's own class when `object_class` was rewritten to the geometric class.
    axis_px: tuple[tuple[float, float], tuple[float, float]] | None = None
    elongation: float | None = None
    width_px: float | None = None
    axis_residual_px: float | None = None
    colour_class: str | None = None

    @property
    def extent(self) -> tuple[float, float] | None:
        if self.box is None:
            return None
        return (self.box[2] - self.box[0], self.box[3] - self.box[1])

    def axis_obs(self, weight: float = 1.0) -> AxisObs:
        """The `multiview_lines` record of this observation: the mask axis as the plane
        constraint when elongated, the centroid's ray otherwise."""
        return AxisObs(
            self.view,
            None if self.axis_px is None else np.asarray(self.axis_px, dtype=np.float64),
            self.point,
            self.elongation,
            weight,
        )


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
        axis_px=row.mask_axis_px,
        elongation=row.mask_elongation,
        width_px=row.mask_width_px,
        axis_residual_px=row.mask_axis_residual_px,
    )


def _to_geometric_class(items: Iterable[Obs], params: TrackerParams) -> None:
    """Line classes become the geometric class; the row's own class is kept as the colour
    attribute. Applied after the source rule, which still runs per original class."""
    for o in items:
        if params.is_line_class(o.object_class) and o.colour_class is None:
            o.colour_class = o.object_class
            o.object_class = params.line_geometric_class


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
    if params.line_classes:
        for items in tracked.values():
            _to_geometric_class(items, params)
        for items in detector.values():
            _to_geometric_class(items, params)
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
    # -- p2-tracker-lines (None / empty unless the track is a line track): unit direction from
    # endpoints_cm[0] to endpoints_cm[1]; the endpoints (2x3, cm; `position` is their
    # midpoint); their velocities (2x3) under the motion model; the direction's scalar
    # uncertainty (deg); the accumulated per-view observed classes; the colour vote's
    # histogram (p2-colour-vote fills it); which end is the tip once a hand resolved it; the
    # running median mask width per view (merged-mask check); this frame's line residuals and
    # dropped views; whether this frame's update was a line fit or the point fallback.
    direction: np.ndarray | None = None
    endpoints_cm: np.ndarray | None = None
    endpoint_velocity: np.ndarray | None = None
    direction_uncertainty_deg: float = 0.0
    class_votes: Counter = field(default_factory=Counter)
    colour_hist: dict[str, float] | None = None
    tip_is_endpoint_0: bool | None = None
    last_observed_endpoints: np.ndarray | None = None
    last_endpoints_frame: int | None = None
    width_history: dict[str, list[float]] = field(default_factory=dict)
    line_residuals: dict[str, float] = field(default_factory=dict)
    merged_views: tuple[str, ...] = ()
    line_this_frame: bool | None = None

    @property
    def is_line(self) -> bool:
        return self.direction is not None and self.endpoints_cm is not None

    def line(self) -> Line3D:
        assert self.direction is not None and self.endpoints_cm is not None
        return Line3D(point=self.position, direction=self.direction, endpoints=self.endpoints_cm)

    @property
    def observed_class(self) -> str | None:
        """The plurality of the observed classes (ties broken alphabetically)."""
        if not self.class_votes:
            return None
        top = max(self.class_votes.values())
        return min(cls for cls, n in self.class_votes.items() if n == top)

    def set_endpoints(self, endpoints: np.ndarray) -> None:
        """Endpoints define the line: position at the midpoint, direction along them."""
        ends = np.asarray(endpoints, dtype=np.float64).reshape(2, 3)
        self.endpoints_cm = ends
        self.position = ends.mean(axis=0)
        span = ends[1] - ends[0]
        norm = float(np.linalg.norm(span))
        if norm > 1e-9:
            self.direction = span / norm

    def tip_and_butt(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self.endpoints_cm is None or self.tip_is_endpoint_0 is None:
            return None
        tip = 0 if self.tip_is_endpoint_0 else 1
        return self.endpoints_cm[tip], self.endpoints_cm[1 - tip]


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


def _pixel_in_box(pixel: np.ndarray, box: tuple[float, float, float, float]) -> bool:
    return bool(box[0] <= pixel[0] <= box[2] and box[1] <= pixel[1] <= box[3])


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
    # -- p2-tracker-lines: the fitted line and its prior-completed endpoints (None on a point
    # candidate, including a line class whose fit was degenerate), the views dropped as merged,
    # the visible length before the prior and whether the prior reconstructed an end.
    line: Line3D | None = None
    endpoints: np.ndarray | None = None
    merged_views: tuple[str, ...] = ()
    visible_length_cm: float | None = None
    extended: bool = False


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


# --------------------------------------------------------------------------- lines
#
# p2-tracker-lines (Sep 28). A line class observation is a plane (elongated mask axis) or a
# ray (compact mask, detector box) through its camera; `multiview_lines.fit_line` turns two or
# more of them into a 3D line. Two planes always meet in a line, so a two-view fit carries no
# residual: pairs are gated on plausibility (the two views' extents on the line overlap or fit
# within one pipette length; a ray meets the plane along the other view's axis), and the
# residual check runs once a clique has three or more views, as the point birth's clique
# prune does. Endpoints come from `line_endpoints` with the soft prior: a visible extent
# shorter than the class prior by more than the spread is completed to the prior at the tip
# end when the track knows its tip (the tip, the white cone away from the hand, is
# systematically the shorter end of a SAM3 mask), else from the end more views reach; within
# the spread it is left as seen. The coloured part of a pipette is the plunger button at the
# butt, where the hand is: the end nearer a hand is the butt, the tip the other end, and
# `endpoints_cm[0]` is the tip in the output once known. A view is dropped from a frame's fit
# as *merged* when its mask width exceeds the track's running median width in that view by the
# factor, its own extent on the line exceeds the factor times the prior, or its skeleton
# residual exceeds the maximum.


@dataclass
class LineFit:
    """One frame's line fit over a track's (or clique's) members: the line, the endpoints
    after the soft prior, the members kept, their perpendicular residuals (undistorted px)
    against the line, the views dropped as merged, the visible length before the prior and
    whether the prior reconstructed an end."""

    line: Line3D
    endpoints: np.ndarray
    members: dict[str, Obs]
    residuals: dict[str, float]
    merged_views: tuple[str, ...]
    visible_length_cm: float
    extended: bool
    plane_views: tuple[str, ...]
    # True when the direction came from the prediction (one plane plus rays), not the views.
    aided: bool = False


def _view_weight(view: str, params: TrackerParams) -> float:
    return params.fpv_weight if view == FINEBIO_FPV_VIEW else 1.0


def _view_gate_px(view: str, params: TrackerParams) -> float:
    return params.gates.handoff_px if view == FINEBIO_FPV_VIEW else params.gates.association_px


def _cams_and_obs(
    members: dict[str, Obs], cams: dict[str, Camera], params: TrackerParams
) -> list[tuple[Camera, AxisObs]]:
    return [(cams[v], o.axis_obs(_view_weight(v, params))) for v, o in members.items()]


def _line_view_residual(cam: Camera, o: Obs, line: Line3D) -> float:
    """One observation against a line in its view: the mean distance of the undistorted axis
    endpoints to the projected line (an elongated mask), else the centroid's distance (px;
    NaN when the line does not project)."""
    axis = o.axis_obs()
    if is_elongated(axis):
        assert axis.endpoints_px is not None
        return axis_residual(cam, axis.endpoints_px, line)[0]
    return point_residual(cam, o.point, line)


def _along_line_interval(cam: Camera, o: Obs, line: Line3D) -> tuple[float, float]:
    """The interval of the line parameter the observation covers: its axis endpoints' rays'
    closest points (an elongated mask), else the centroid ray's."""
    axis = o.axis_obs()
    pixels = axis.endpoints_px if is_elongated(axis) else o.point.reshape(1, 2)
    assert pixels is not None
    params = [line.parameter_closest_to_ray(*ray_from_point(cam, p)) for p in pixels]
    return min(params), max(params)


def _interval_gap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, max(a[0], b[0]) - min(a[1], b[1]))


def _line_pair_cost(
    cam_a: Camera,
    a: Obs,
    cam_b: Camera,
    b: Obs,
    params: TrackerParams,
    prior_length_cm: float,
) -> float:
    """Plausibility of two line-class observations in two views being one object, in cm
    (`HUNGARIAN_FORBIDDEN` when implausible). Two planes: their line must lie in front of
    both cameras and the two extents on it must overlap within `line_gate_cm` and fit within
    `line_merged_extent_factor` times the prior; a plane and a ray: the ray meets the plane
    along the axis (within the gate plus the length the axis view does not see); two rays:
    the centroid pair residual under the association gate. A colour mismatch adds one cm so
    same-colour pairs win ties, no more."""
    axis_a, axis_b = (
        a.axis_obs(_view_weight(a.view, params)),
        b.axis_obs(_view_weight(b.view, params)),
    )
    plane_a, plane_b = is_elongated(axis_a), is_elongated(axis_b)
    colour_penalty = 1.0 if a.colour_class != b.colour_class else 0.0
    if plane_a and plane_b:
        line = fit_line(
            [(cam_a, axis_a), (cam_b, axis_b)], min_pair_angle_deg=params.line_min_pair_angle_deg
        )
        if line is None:
            return _ray_pair_cost_cm(cam_a, a, cam_b, b, params) + colour_penalty
        if depth_cm(cam_a, line.midpoint) <= 0 or depth_cm(cam_b, line.midpoint) <= 0:
            return HUNGARIAN_FORBIDDEN
        ia, ib = _along_line_interval(cam_a, a, line), _along_line_interval(cam_b, b, line)
        gap = _interval_gap(ia, ib)
        union = max(ia[1], ib[1]) - min(ia[0], ib[0])
        if gap > params.line_gate_cm or union > params.line_merged_extent_factor * prior_length_cm:
            return HUNGARIAN_FORBIDDEN
        return gap + max(0.0, union - prior_length_cm) + colour_penalty
    if plane_a or plane_b:
        cam_p, p, cam_r, r = (cam_a, a, cam_b, b) if plane_a else (cam_b, b, cam_a, a)
        cost = _plane_ray_cost_cm(cam_p, p, cam_r, r, params, prior_length_cm)
        return cost if cost >= HUNGARIAN_FORBIDDEN else cost + colour_penalty
    cost = _ray_pair_cost_cm(cam_a, a, cam_b, b, params)
    return cost if cost >= HUNGARIAN_FORBIDDEN else cost + colour_penalty


def _ray_pair_cost_cm(cam_a: Camera, a: Obs, cam_b: Camera, b: Obs, params: TrackerParams) -> float:
    """The point birth's pair cost on the centroids, gated as there and converted to cm at the
    triangulated point."""
    px = _pair_cost(cam_a, a, cam_b, b)
    if px > min(_view_gate_px(a.view, params), _view_gate_px(b.view, params)):
        return HUNGARIAN_FORBIDDEN
    point = triangulate_pixels([cam_a, cam_b], [a.point, b.point])
    scale = 0.5 * (pixels_per_cm(cam_a, point) + pixels_per_cm(cam_b, point))
    return px / max(scale, 1e-6)


def _plane_ray_cost_cm(
    cam_p: Camera, p: Obs, cam_r: Camera, r: Obs, params: TrackerParams, prior_length_cm: float
) -> float:
    """A compact view's centroid ray against an axis view: the ray's point on the axis plane
    must project onto the axis segment, allowing the length the axis view may not see."""
    assert p.axis_px is not None
    try:
        normal, offset = plane_from_axis(cam_p, np.asarray(p.axis_px, dtype=np.float64))
    except ValueError:
        return HUNGARIAN_FORBIDDEN
    origin, direction = ray_from_point(cam_r, r.point)
    denominator = float(normal @ direction)
    if abs(denominator) < math.sin(math.radians(params.line_min_pair_angle_deg)):
        return HUNGARIAN_FORBIDDEN
    t = (offset - float(normal @ origin)) / denominator
    if t <= 0:
        return HUNGARIAN_FORBIDDEN
    point = origin + t * direction
    if depth_cm(cam_p, point) <= 0:
        return HUNGARIAN_FORBIDDEN
    pixel = project(cam_p, point)
    if pixel is None:
        return HUNGARIAN_FORBIDDEN
    ends = np.asarray(p.axis_px, dtype=np.float64)
    segment = ends[1] - ends[0]
    length_px = float(np.linalg.norm(segment))
    unit = segment / max(length_px, 1e-9)
    along = float((pixel - ends[0]) @ unit)
    across = abs(float((pixel - ends[0]) @ np.array([-unit[1], unit[0]])))
    scale = pixels_per_cm(cam_p, point)
    if across / scale > params.line_gate_cm:
        return HUNGARIAN_FORBIDDEN
    overshoot_cm = max(0.0, -along, along - length_px) / scale
    slack = max(0.0, prior_length_cm - length_px / scale)
    if overshoot_cm > slack + params.line_gate_cm:
        return HUNGARIAN_FORBIDDEN
    return overshoot_cm + across / scale


def complete_extent(
    line: Line3D,
    cams_and_obs: Sequence[tuple[Camera, AxisObs]],
    params: TrackerParams,
    prior_length_cm: float,
    prior_spread_cm: float,
    tip_hint: np.ndarray | None = None,
) -> tuple[np.ndarray, float, bool] | None:
    """The soft prior on a fitted line's visible extent: ``(endpoints, visible length,
    extended)``. The visible extent is the axis views' (`line_endpoints`) widened by the
    compact views' centroid rays (their closest points on the line are on the shaft too).
    Within `prior_spread_cm` of the prior it is left as seen; shorter, it is completed to the
    prior at the tip end (the visible end nearer `tip_hint`, the track's predicted tip) unless
    every view reaches that end and not the other, in which case the other end is the one the
    masks missed; without a hint, from the end more views reach, both ends by half when tied.
    None when no view is elongated."""
    extent = line_endpoints(line, cams_and_obs)
    if extent is None:
        return None
    lo, hi = line.parameter(extent.visible[0]), line.parameter(extent.visible[1])
    support = list(extent.support)
    n_views = len(extent.per_view)
    for cam, obs in cams_and_obs:
        if is_elongated(obs):
            continue
        parameter = line.parameter_closest_to_ray(*ray_from_point(cam, obs.centroid_px))
        n_views += 1
        if parameter < lo:
            lo, support[0] = parameter, 1
        elif parameter <= lo + END_TOLERANCE_CM:
            support[0] += 1
        if parameter > hi:
            hi, support[1] = parameter, 1
        elif parameter >= hi - END_TOLERANCE_CM:
            support[1] += 1
    visible = hi - lo
    if visible >= prior_length_cm - prior_spread_cm:
        return np.stack([line.point_at(lo), line.point_at(hi)]), visible, False
    missing = prior_length_cm - visible
    extend_end: int | None = None
    if tip_hint is not None:
        ends = np.stack([line.point_at(lo), line.point_at(hi)])
        tip_end = int(
            np.argmin(np.linalg.norm(ends - np.asarray(tip_hint, dtype=np.float64), axis=1))
        )
        tip_reached_by_all = support[tip_end] >= n_views and support[1 - tip_end] < n_views
        extend_end = 1 - tip_end if tip_reached_by_all else tip_end
    elif support[1] > support[0]:
        extend_end = 0
    elif support[0] > support[1]:
        extend_end = 1
    if extend_end == 0:
        lo = hi - prior_length_cm
    elif extend_end == 1:
        hi = lo + prior_length_cm
    else:
        lo, hi = lo - missing / 2, hi + missing / 2
    endpoints = np.stack([line.point_at(lo), line.point_at(hi)])
    return endpoints, visible, missing > END_TOLERANCE_CM


def fit_line_members(
    members: dict[str, Obs],
    cams: dict[str, Camera],
    params: TrackerParams,
    prior_length_cm: float,
    prior_spread_cm: float,
    width_medians: dict[str, float] | None = None,
    predicted: Line3D | None = None,
    uncertainty_cm: float = 0.0,
    tip_hint: np.ndarray | None = None,
) -> tuple[LineFit | None, tuple[str, ...]]:
    """The line fit over a set of members (one per view), with the merged-mask rejections,
    the residual prune (a member over its gate is dropped while three or more remain) and,
    against a `predicted` line, the line gate (angle and perpendicular offset at the
    midpoint, the latter inflated by `uncertainty_cm`; the member whose view disagrees most
    with the prediction is dropped and the fit repeated). Returns ``(fit, merged views)``;
    the fit is None when no line can be fitted from the members that survive (fewer than
    two, no plane, degenerate, a two-member fit over its gate), the merged views are
    reported either way."""
    members = dict(members)
    merged: list[str] = []
    for v, o in list(members.items()):
        too_rough = (
            o.axis_residual_px is not None and o.axis_residual_px > params.line_max_axis_residual_px
        )
        too_wide = (
            width_medians is not None
            and v in width_medians
            and o.width_px is not None
            and o.width_px > params.line_merged_width_factor * width_medians[v]
        )
        if too_rough or too_wide:
            merged.append(v)
            del members[v]
    while len(members) >= 2:
        cams_and_obs = _cams_and_obs(members, cams, params)
        line = fit_line(cams_and_obs, min_pair_angle_deg=params.line_min_pair_angle_deg)
        if line is None or any(depth_cm(cams[v], line.midpoint) <= 0 for v in members):
            return None, tuple(sorted(merged))
        residuals = {
            v: (r.perpendicular_px if np.isfinite(r.perpendicular_px) else float("inf"))
            for v, r in line.residuals.items()
        }
        extent = line_endpoints(line, cams_and_obs)
        assert extent is not None
        too_long = [
            v
            for v, (lo, hi) in extent.per_view.items()
            if hi - lo > params.line_merged_extent_factor * prior_length_cm
        ]
        if too_long:
            for v in too_long:
                merged.append(v)
                del members[v]
            continue
        if extent.visible_length_cm > params.line_merged_extent_factor * prior_length_cm:
            # The views' extents together span more than one pipette: two objects, or one view
            # on the wrong one. With three or more, the odd interval goes; with two, no fit.
            if len(extent.per_view) < 3:
                return None, tuple(sorted(merged))
            centres = {v: 0.5 * (lo + hi) for v, (lo, hi) in extent.per_view.items()}
            median = float(np.median(list(centres.values())))
            del members[max(centres, key=lambda v: abs(centres[v] - median))]
            continue
        if predicted is not None:
            angle, perpendicular, _ = line_distance(predicted, line)
            if angle > params.line_gate_deg or perpendicular > params.line_gate_cm + uncertainty_cm:
                worst = max(
                    members, key=lambda v: _line_view_residual(cams[v], members[v], predicted)
                )
                del members[worst]
                continue
        over = {v: r for v, r in residuals.items() if r > _view_gate_px(v, params)}
        if over and len(members) >= 3:
            del members[max(over, key=over.get)]
            continue
        if over:
            return None, tuple(sorted(merged))
        completed = complete_extent(
            line, cams_and_obs, params, prior_length_cm, prior_spread_cm, tip_hint
        )
        assert completed is not None
        endpoints, visible, extended = completed
        fit = LineFit(
            line=line,
            endpoints=endpoints,
            members=members,
            residuals=residuals,
            merged_views=tuple(sorted(merged)),
            visible_length_cm=visible,
            extended=extended,
            plane_views=line.plane_views,
        )
        return fit, fit.merged_views
    return None, tuple(sorted(merged))


def decisive_class(votes: Counter) -> str | None:
    """The plurality class when it holds more than half of the votes, else None."""
    if not votes:
        return None
    cls, top = max(sorted(votes.items()), key=lambda kv: kv[1])
    return cls if top * 2 > sum(votes.values()) else None


def line_birth_candidates(
    unassigned: dict[str, list[Obs]],
    cams: dict[str, Camera],
    params: TrackerParams,
    prior: LinePrior,
) -> list[Candidate]:
    """Birth candidates of the geometric class: pairwise plausibility (`_line_pair_cost`)
    across every view with a pose, Hungarian per view pair, cliques as the point birth forms
    them, then `fit_line_members` per clique with the class prior when the members' colour
    vote is decisive; a clique whose fit is degenerate falls back to the point birth's
    triangulation of its centroids. The birth rule (fixed views, or fixed views plus the fpv)
    is the core's."""
    cls = params.line_geometric_class
    per_view = {
        v: [o for o in unassigned.get(v, ()) if o.object_class == cls] for v in sorted(cams)
    }
    per_view = {v: items for v, items in per_view.items() if items}
    born: list[Candidate] = []
    if len(per_view) < min(
        params.gates.birth_min_fixed_views, params.gates.birth_fixed_views_with_fpv
    ):
        return born
    edges: dict[tuple[str, int, str, int], float] = {}
    for u, v in combinations(sorted(per_view), 2):
        cost = np.full((len(per_view[u]), len(per_view[v])), HUNGARIAN_FORBIDDEN)
        for i, a in enumerate(per_view[u]):
            for j, b in enumerate(per_view[v]):
                cost[i, j] = _line_pair_cost(cams[u], a, cams[v], b, params, prior.length_cm)
        for i, j in hungarian(cost):
            if cost[i, j] < HUNGARIAN_FORBIDDEN:
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
        votes = Counter(o.colour_class for o in members.values() if o.colour_class)
        fit, _merged = fit_line_members(
            members, cams, params, prior.length_for(decisive_class(votes)), prior.spread_cm
        )
        if fit is not None and len(fit.members) >= 2:
            kept = fit.members
            cand = Candidate(
                cls,
                fit.endpoints.mean(axis=0),
                kept,
                {w: round(fit.residuals[w], 2) for w in kept},
                line=fit.line,
                endpoints=fit.endpoints,
                merged_views=fit.merged_views,
                visible_length_cm=fit.visible_length_cm,
                extended=fit.extended,
            )
        else:
            point, kept, residuals = _prune_clique(members, cams, params)
            if len(kept) < 2:
                continue
            cand = Candidate(cls, point, kept, {w: residuals[w] for w in kept})
        for w in kept:
            used.add((w, clique[w]))
        n_fixed = sum(1 for w in kept if w != FINEBIO_FPV_VIEW)
        has_fpv = FINEBIO_FPV_VIEW in kept
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
        line_prior: LinePrior | None = None,
    ) -> None:
        self.fixed_cams = dict(fixed_cams)
        self.fixed_views = tuple(fixed_cams)
        self.fpv_source = fpv_source
        self.params = params or TrackerParams()
        self.line_prior = line_prior
        if self.params.line_classes and self.line_prior is None:
            self.line_prior = load_line_prior(self.params.line_prior_path)
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
        # -- p2-tracker-lines counters (all zero / empty with the extension off)
        self.line_births = 0
        self.line_point_births = 0
        self.line_frames = 0
        self.line_aided_frames = 0
        self.line_point_frames = 0
        self.line_single_view_frames = 0
        self.line_degenerate_fits = 0
        self.line_loo_px: list[float] = []
        self.line_loo_deg: list[float] = []
        self.line_lengths: list[tuple[str | None, float, float]] = []
        self.line_merged_by_view: dict[str, int] = defaultdict(int)
        self.line_extended_frames = 0
        self.line_track_frames = 0
        self.line_tip_resolved_frames = 0
        self.line_tip_resolutions: dict[str, int] = defaultdict(int)

    # -- helpers

    def _is_geometric(self, t: Track) -> bool:
        return bool(self.params.line_classes) and t.object_class == self.params.line_geometric_class

    def _prior_length(self, t: Track) -> float:
        assert self.line_prior is not None
        return self.line_prior.length_for(decisive_class(t.class_votes))

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
        if params.line_classes:
            geometric = params.line_geometric_class
            lines = {
                v: [o for o in items if o.object_class == geometric]
                for v, items in unassigned.items()
            }
            unassigned = {
                v: [o for o in items if o.object_class != geometric]
                for v, items in unassigned.items()
            }
            assert self.line_prior is not None
            line_candidates = line_birth_candidates(lines, cams, params, self.line_prior)
        else:
            line_candidates = []
        candidates = birth_candidates(unassigned, cams, self.fixed_views, params)
        if params.group_tracks:
            candidates = self._merge_group_candidates(candidates, cams)
        self._births_and_reacquisitions(frame, candidates + line_candidates, cams)

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
            if t.is_line:
                # Constant velocity on both endpoints: the segment translates and turns; its
                # length is a property of the object and is kept.
                assert t.endpoints_cm is not None
                endpoint_velocity = (
                    t.endpoint_velocity
                    if t.endpoint_velocity is not None
                    else np.stack([t.velocity, t.velocity])
                )
                length = float(np.linalg.norm(t.endpoints_cm[1] - t.endpoints_cm[0]))
                moved = t.endpoints_cm + endpoint_velocity * dt
                span = moved[1] - moved[0]
                norm = float(np.linalg.norm(span))
                if norm > 1e-9:
                    midpoint = moved.mean(axis=0)
                    moved = np.stack(
                        [midpoint - span / norm * length / 2, midpoint + span / norm * length / 2]
                    )
                t.set_endpoints(moved)
                if t.state == "coasting":
                    t.endpoint_velocity = endpoint_velocity * params.mover_coast_damping**dt
            else:
                t.position = t.position + t.velocity * dt
            speed = float(np.linalg.norm(t.velocity))
            noise = float(np.hypot(noise, params.mover_noise_factor * speed * dt))
            if t.state == "coasting":
                t.velocity = t.velocity * params.mover_coast_damping**dt
            self.mover_frames += 1
        if t.is_line:
            t.direction_uncertainty_deg = float(
                np.hypot(t.direction_uncertainty_deg, LINE_DIRECTION_NOISE_DEG * dt)
            )
        t.uncertainty_cm = min(params.max_uncertainty_cm, float(np.hypot(t.uncertainty_cm, noise)))

    def _associate(
        self, cams: dict[str, Camera], obs: FrameObservations
    ) -> dict[tuple[str, str], Obs]:
        pairs: list[tuple[float, str, str, Obs]] = []
        for t in self.live_tracks():
            if t.state not in LOCALISED_STATES or t.group_container is not None or t.is_line:
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
        if self.params.line_classes:
            assigned.update(self._associate_lines(cams, obs, used_obs))
        return assigned

    def _line_view_cost(self, t: Track, cam: Camera, o: Obs) -> float:
        """A line track against one observation in one view: the perpendicular distance of
        the observed axis (or centroid) to the projected predicted line, gated by `gate_px`,
        plus the along-line check that the observation's extent on the line overlaps the
        track's within the line gate and the uncertainty (`HUNGARIAN_FORBIDDEN` outside). An
        observation whose class differs from the track's decisive plurality costs half the
        gate more, so two pipettes side by side in the stand keep their own masks when both
        are within reach; the colour never bars an association."""
        line = t.line()
        residual = _line_view_residual(cam, o, line)
        gate = gate_px(t, cam, self.params)
        if not np.isfinite(residual) or residual > gate:
            return HUNGARIAN_FORBIDDEN
        assert t.endpoints_cm is not None
        mine = (line.parameter(t.endpoints_cm[0]), line.parameter(t.endpoints_cm[1]))
        gap = _interval_gap((min(mine), max(mine)), _along_line_interval(cam, o, line))
        if gap > self.params.line_gate_cm + t.uncertainty_cm:
            return HUNGARIAN_FORBIDDEN
        plurality = decisive_class(t.class_votes)
        if plurality is not None and o.colour_class is not None and o.colour_class != plurality:
            residual += 0.5 * gate
        return residual

    def _associate_lines(
        self, cams: dict[str, Camera], obs: FrameObservations, used_obs: set[tuple[str, int]]
    ) -> dict[tuple[str, str], Obs]:
        """Per view, Hungarian assignment of the localised line tracks to the geometric
        class's observations on `_line_view_cost`."""
        tracks = [
            t
            for t in self.live_tracks()
            if t.state in LOCALISED_STATES and t.group_container is None and t.is_line
        ]
        assigned: dict[tuple[str, str], Obs] = {}
        if not tracks:
            return assigned
        cls = self.params.line_geometric_class
        for v, cam in cams.items():
            items = [
                o
                for o in obs.tracked.get(v, ())
                if o.object_class == cls and (v, o.index) not in used_obs
            ]
            if not items:
                continue
            cost = np.full((len(tracks), len(items)), HUNGARIAN_FORBIDDEN)
            for i, t in enumerate(tracks):
                if depth_cm(cam, t.position) <= 0:
                    continue
                for j, o in enumerate(items):
                    cost[i, j] = self._line_view_cost(t, cam, o)
            for i, j in hungarian(cost):
                if cost[i, j] < HUNGARIAN_FORBIDDEN:
                    assigned[(tracks[i].track_id, v)] = items[j]
                    used_obs.add((v, items[j].index))
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
        if self._is_geometric(t):
            self._update_line(t, mine, cams, frame, dt, obs)
            return
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

    # -- line tracks (p2-tracker-lines)

    def _width_medians(self, t: Track, cams: dict[str, Camera]) -> dict[str, float]:
        """Running median mask width per view once five samples exist (the merged check), in
        pixels at the track's current depth in that view (the history is kept in cm so a
        pipette carried towards a camera does not look merged)."""
        return {
            v: float(np.median(widths)) * pixels_per_cm(cams[v], t.position)
            for v, widths in t.width_history.items()
            if len(widths) >= 5 and v in cams
        }

    def _record_line_support(self, t: Track, mine: dict[str, Obs], cams: dict[str, Camera]) -> None:
        for v, o in mine.items():
            if o.colour_class is not None:
                t.class_votes[o.colour_class] += 1
            if o.width_px is not None and o.axis_px is not None and v in cams:
                history = t.width_history.setdefault(v, [])
                history.append(float(o.width_px) / pixels_per_cm(cams[v], t.position))
                del history[:-60]

    def _update_line(
        self,
        t: Track,
        mine: dict[str, Obs],
        cams: dict[str, Camera],
        frame: int,
        dt: int,
        obs: FrameObservations | None,
    ) -> None:
        """The geometric class's update: with two or more views the line fit
        (`fit_line_members` with the merged checks and the line gate against the prediction),
        falling back to the point path when no line can be fitted; with one view a lateral
        update as the core's, on the axis plane for a line track; class votes, mask widths and
        the tip / butt resolution afterwards."""
        params = self.params
        assert self.line_prior is not None
        t.line_residuals = {}
        t.merged_views = ()
        t.line_this_frame = None
        if len(mine) >= 2:
            resolved = t.tip_and_butt()
            fit, merged = fit_line_members(
                mine,
                cams,
                params,
                self._prior_length(t),
                self.line_prior.spread_cm,
                width_medians=self._width_medians(t, cams),
                predicted=t.line() if t.is_line else None,
                uncertainty_cm=t.uncertainty_cm,
                tip_hint=None if resolved is None else resolved[0],
            )
            for v in merged:
                self.line_merged_by_view[v] += 1
            t.merged_views = merged
            remaining = {v: o for v, o in mine.items() if v not in merged} or mine
            if fit is None and t.is_line:
                fit = self._aided_line_fit(t, remaining, cams)
                if fit is not None:
                    self.line_aided_frames += 1
            if fit is not None:
                self._apply_line_fit(t, fit, cams, frame)
                mine = fit.members
            else:
                self.line_degenerate_fits += 1
                point, kept, residuals = _prune_clique(remaining, cams, params)
                if len(kept) >= 2:
                    self._apply_point_fit(t, point, kept, residuals, cams, frame)
                    mine = kept
                    t.line_this_frame = False
                    self.line_point_frames += 1
                else:
                    best = min(
                        remaining,
                        key=lambda v: self._single_view_residual(t, cams[v], remaining[v]),
                    )
                    mine = {best: remaining[best]}
        if len(mine) == 1:
            (v, o) = next(iter(mine.items()))
            residual = self._single_view_residual(t, cams[v], o)
            t.residuals = {v: round(residual, 2)} if np.isfinite(residual) else {}
            if params.motion_model and t.mover:
                self._lateral_line_update(t, cams[v], o)
                self.single_view_ray_updates += 1
            t.state = "single_view"
            t.frames_unobserved = 0
            t.last_observed_frame = frame
            self.line_single_view_frames += 1
        elif not mine:
            self._start_coasting(t, frame, dt, cams, obs)
            return
        t.support_views = tuple(sorted(mine))
        t.support_slots = {v: o.slot for v, o in mine.items()}
        for v, o in mine.items():
            self._record_support(t, v, o, cams[v], frame)
        self._record_line_support(t, mine, cams)
        if t.is_line and obs is not None:
            self._resolve_tip(t, cams, obs)

    def _single_view_residual(self, t: Track, cam: Camera, o: Obs) -> float:
        if t.is_line:
            return _line_view_residual(cam, o, t.line())
        pixel = project(cam, t.position)
        return float("inf") if pixel is None else float(np.linalg.norm(pixel - o.point))

    def _apply_point_fit(
        self,
        t: Track,
        point: np.ndarray,
        kept: dict[str, Obs],
        residuals: dict[str, float],
        cams: dict[str, Camera],
        frame: int,
    ) -> None:
        """The core's >= 2-view position update; a line track's endpoints move with it."""
        params = self.params
        depth = np.median([depth_cm(cams[v], point) for v in kept])
        f = np.median([cams[v].K[0, 0] for v in kept])
        meas_cm = max(
            params.base_uncertainty_cm, float(np.median(list(residuals.values()))) * depth / f
        )
        prior_var = t.uncertainty_cm**2
        gain = prior_var / (prior_var + meas_cm**2)
        new_position = t.position + gain * (point - t.position)
        if t.is_line:
            # Centroids of different visible portions carry no along-shaft information: only
            # the component across the line moves the segment.
            assert t.endpoints_cm is not None and t.direction is not None
            delta = new_position - t.position
            delta = delta - float(delta @ t.direction) * t.direction
            t.set_endpoints(t.endpoints_cm + delta)
        else:
            t.position = new_position
        t.uncertainty_cm = float(np.sqrt((1 - gain) * prior_var))
        t.state = "observed"
        if params.motion_model:
            self._update_velocity(t, frame)
        t.last_observed_frame = frame
        t.frames_unobserved = 0
        t.residuals = {v: round(residuals[v], 2) for v in kept}

    def _apply_line_fit(self, t: Track, fit: LineFit, cams: dict[str, Camera], frame: int) -> None:
        """Blend the fitted line into the track: the midpoint and the length by the scalar
        Kalman gain the core uses for a position, the direction by its own gain (its process
        noise is `LINE_DIRECTION_NOISE_DEG` per frame, a measurement is worth
        `LINE_DIRECTION_MEAS_DEG`, so it follows the measurement while pipetting); a point
        track of the geometric class is promoted to a line track."""
        params = self.params
        measured = fit.endpoints
        direction = fit.line.direction
        kept = fit.members
        depth = np.median([depth_cm(cams[v], fit.line.midpoint) for v in kept])
        f = np.median([cams[v].K[0, 0] for v in kept])
        finite = [r for r in fit.residuals.values() if np.isfinite(r)]
        meas_cm = max(
            params.base_uncertainty_cm, float(np.median(finite)) * depth / f if finite else 0.0
        )
        prior_var = t.uncertainty_cm**2
        gain = prior_var / (prior_var + meas_cm**2)
        if t.is_line:
            assert t.direction is not None and t.endpoints_cm is not None
            if float(direction @ t.direction) < 0:
                direction, measured = -direction, measured[::-1]
            midpoint = t.position + gain * (measured.mean(axis=0) - t.position)
            if fit.aided:
                # The direction is the prediction's, corrected onto the plane: nothing was
                # measured about it in the plane, so its uncertainty stays.
                blended = direction
            else:
                var_dir = t.direction_uncertainty_deg**2
                gain_dir = var_dir / (var_dir + LINE_DIRECTION_MEAS_DEG**2)
                blended = (1.0 - gain_dir) * t.direction + gain_dir * direction
                blended /= np.linalg.norm(blended)
                t.direction_uncertainty_deg = float(np.sqrt((1 - gain_dir) * var_dir))
            length_p = float(np.linalg.norm(t.endpoints_cm[1] - t.endpoints_cm[0]))
            length_m = float(np.linalg.norm(measured[1] - measured[0]))
            length = length_p + gain * (length_m - length_p)
            ends = np.stack([midpoint - blended * length / 2, midpoint + blended * length / 2])
        else:
            midpoint = t.position + gain * (measured.mean(axis=0) - t.position)
            ends = measured - measured.mean(axis=0) + midpoint
            t.direction_uncertainty_deg = LINE_DIRECTION_MEAS_DEG
        t.set_endpoints(ends)
        t.uncertainty_cm = float(np.sqrt((1 - gain) * prior_var))
        t.state = "observed"
        if params.motion_model:
            self._update_velocity(t, frame, endpoints=not fit.aided)
        t.last_observed_frame = frame
        t.frames_unobserved = 0
        t.residuals = {v: round(fit.residuals[v], 2) for v in kept}
        t.line_residuals = dict(t.residuals)
        t.line_this_frame = True
        self.line_frames += 1
        self._record_line_metrics(t, fit, cams, self._prior_length(t))

    def _record_line_metrics(
        self, t: Track, fit: LineFit, cams: dict[str, Camera], prior_length: float
    ) -> None:
        self.line_lengths.append(
            (
                decisive_class(t.class_votes),
                fit.visible_length_cm,
                fit.visible_length_cm - prior_length,
            )
        )
        if fit.extended:
            self.line_extended_frames += 1
        if len(fit.plane_views) >= 3:
            for record in loo_residual(
                _cams_and_obs(fit.members, cams, self.params),
                min_pair_angle_deg=self.params.line_min_pair_angle_deg,
            ):
                if record["fitted"] and np.isfinite(record["perpendicular_px"]):
                    self.line_loo_px.append(float(record["perpendicular_px"]))
                    if record["angle_deg"] is not None and np.isfinite(record["angle_deg"]):
                        self.line_loo_deg.append(float(record["angle_deg"]))

    def _lateral_line_update(self, t: Track, cam: Camera, o: Obs) -> None:
        """A moving line track seen in one view: an elongated mask moves each endpoint onto
        the plane through the camera and the observed axis (the perpendicular correction, the
        length kept, the depth along the view kept); a compact one moves the midpoint along
        the centroid's ray as the core does for a point."""
        assert t.endpoints_cm is not None
        axis = o.axis_obs()
        if is_elongated(axis):
            assert axis.endpoints_px is not None
            try:
                normal, offset = plane_from_axis(cam, axis.endpoints_px)
            except ValueError:
                return
            feet = t.endpoints_cm - ((t.endpoints_cm @ normal) - offset)[:, None] * normal
            length = float(np.linalg.norm(t.endpoints_cm[1] - t.endpoints_cm[0]))
            span = feet[1] - feet[0]
            norm = float(np.linalg.norm(span))
            if norm < 1e-9:
                return
            midpoint = feet.mean(axis=0)
            ends = np.stack(
                [midpoint - span / norm * length / 2, midpoint + span / norm * length / 2]
            )
        else:
            ends = t.endpoints_cm + (ray_point(cam, o.point, t.position) - t.position)
        t.set_endpoints(ends)

    def _aided_line_fit(
        self, t: Track, members: dict[str, Obs], cams: dict[str, Camera]
    ) -> LineFit | None:
        """The fit `fit_line` cannot make on its own: one plane (or planes meeting under the
        pair angle) plus rays to one spot of the shaft. The predicted direction projected
        onto the plane gives the direction; the line runs through the mean of the ray-plane
        intersections (the rays' points on the shaft), else through the foot of the predicted
        midpoint. Members over their view gate against that line are left out; the extent
        and the prior follow as for a full fit. None without a plane in front of its camera."""
        params = self.params
        assert t.direction is not None and self.line_prior is not None
        planes: list[tuple[str, np.ndarray, float]] = []
        rays: list[tuple[str, np.ndarray, np.ndarray]] = []
        for v, o in members.items():
            axis = o.axis_obs()
            if is_elongated(axis):
                assert axis.endpoints_px is not None
                try:
                    normal, offset = plane_from_axis(cams[v], axis.endpoints_px)
                except ValueError:
                    continue
                planes.append((v, normal, offset))
            else:
                rays.append((v, *ray_from_point(cams[v], o.point)))
        if not planes:
            return None
        _, normal, offset = planes[0]
        direction = t.direction - float(t.direction @ normal) * normal
        if np.linalg.norm(direction) < 1e-6:
            return None
        direction /= np.linalg.norm(direction)
        foot = t.position - (float(normal @ t.position) - offset) * normal
        across = np.cross(normal, direction)
        hits = []
        min_sine = math.sin(math.radians(params.line_min_pair_angle_deg))
        for _, origin, ray_direction in rays:
            denominator = float(normal @ ray_direction)
            if abs(denominator) < min_sine:
                continue
            s = (offset - float(normal @ origin)) / denominator
            if s > 0:
                hits.append(origin + s * ray_direction)
        if hits:
            foot = foot + float(np.mean([(h - foot) @ across for h in hits])) * across
        line = Line3D(point=foot, direction=direction)
        kept = {}
        residuals = {}
        for v, o in members.items():
            residual = _line_view_residual(cams[v], o, line)
            if np.isfinite(residual) and residual <= _view_gate_px(v, params):
                kept[v] = o
                residuals[v] = residual
        if not any(is_elongated(o.axis_obs()) for o in kept.values()):
            return None
        cams_and_obs = _cams_and_obs(kept, cams, params)
        resolved = t.tip_and_butt()
        completed = complete_extent(
            line,
            cams_and_obs,
            params,
            self._prior_length(t),
            self.line_prior.spread_cm,
            None if resolved is None else resolved[0],
        )
        if completed is None:
            return None
        endpoints, visible, extended = completed
        return LineFit(
            line=line,
            endpoints=endpoints,
            members=kept,
            residuals=residuals,
            merged_views=(),
            visible_length_cm=visible,
            extended=extended,
            plane_views=tuple(v for v, _, _ in planes if v in kept),
            aided=True,
        )

    def _resolve_tip(self, t: Track, cams: dict[str, Camera], obs: FrameObservations) -> None:
        """The end nearer a hand is the butt (the plunger button, the coloured part), the
        other end the tip: the nearest live hand track within `LINE_HAND_REACH_CM` of one end
        and `LINE_HAND_MARGIN_CM` nearer to it than to the other decides; without a hand
        track, an end whose projection lies inside a hand box in >= `held_min_views` views
        while the other end's does not. Otherwise the ordering is kept by direction
        continuity (the update orients the measured direction along the predicted one)."""
        assert t.endpoints_cm is not None
        params = self.params
        hands = [
            h
            for h in self.live_tracks()
            if h.object_class in params.hand_classes and h.state in LOCALISED_STATES
        ]
        for hand in sorted(
            hands, key=lambda h: float(np.linalg.norm(t.endpoints_cm - h.position, axis=1).min())
        ):
            d0, d1 = np.linalg.norm(t.endpoints_cm - hand.position, axis=1)
            if min(d0, d1) <= LINE_HAND_REACH_CM and abs(d0 - d1) >= LINE_HAND_MARGIN_CM:
                self._set_tip(t, tip_is_endpoint_0=bool(d0 > d1), basis="hand_track")
                return
        counts = [0, 0]
        for v, cam in cams.items():
            pixels = [project(cam, e) for e in t.endpoints_cm]
            for d in obs.detector.get(v, ()):
                if d.object_class not in params.hand_classes or d.box is None:
                    continue
                for k, pixel in enumerate(pixels):
                    if pixel is not None and _pixel_in_box(pixel, d.box):
                        counts[k] += 1
                        break
        if max(counts) >= params.held_min_views and counts[0] != counts[1]:
            self._set_tip(t, tip_is_endpoint_0=counts[0] < counts[1], basis="hand_box")

    def _set_tip(self, t: Track, *, tip_is_endpoint_0: bool, basis: str) -> None:
        if t.tip_is_endpoint_0 != tip_is_endpoint_0:
            self.line_tip_resolutions[basis] += 1
        t.tip_is_endpoint_0 = tip_is_endpoint_0

    def _update_velocity(self, t: Track, frame: int, *, endpoints: bool = True) -> None:
        """Smoothed finite difference of the filtered position over >= 2-view updates; the
        mover flag switches on above `mover_speed_cm_per_frame` and off below half of it. A
        line track's endpoint velocities follow the same rule when `endpoints` (a full fit);
        after a prediction-aided fit they are left as they were, so a direction nobody
        measured cannot feed its own velocity."""
        params = self.params
        if t.last_observed_position is not None and frame > t.last_observed_frame:
            v_new = (t.position - t.last_observed_position) / (frame - t.last_observed_frame)
            alpha = params.velocity_smoothing
            t.velocity = (1.0 - alpha) * t.velocity + alpha * v_new
        if (
            t.is_line
            and endpoints
            and t.last_observed_endpoints is not None
            and t.last_endpoints_frame is not None
            and frame > t.last_endpoints_frame
        ):
            assert t.endpoints_cm is not None
            previous = t.endpoint_velocity if t.endpoint_velocity is not None else np.zeros((2, 3))
            step = (t.endpoints_cm - t.last_observed_endpoints) / (frame - t.last_endpoints_frame)
            alpha = params.velocity_smoothing
            t.endpoint_velocity = (1.0 - alpha) * previous + alpha * step
        t.last_observed_position = t.position.copy()
        if t.is_line and (endpoints or t.last_observed_endpoints is None):
            assert t.endpoints_cm is not None
            t.last_observed_endpoints = t.endpoints_cm.copy()
            t.last_endpoints_frame = frame
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
                    t.endpoint_velocity = None
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
                if hand_track is None:
                    t.held_offset = None
                elif t.is_line:
                    # The hand holds the butt: the offset is measured from the hand to it
                    # (the end nearer the hand, resolved now if it was not).
                    anchor = self._butt_anchor(t, hand_track.position, resolve=True)
                    t.held_offset = anchor - hand_track.position
                else:
                    t.held_offset = t.position - hand_track.position
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
        # A line track is inside the hand box when its midpoint or either end is.
        points = [t.position, *t.endpoints_cm] if t.is_line else [t.position]
        for v, cam in cams.items():
            pixels = [p for p in (project(cam, x) for x in points) if p is not None]
            if not pixels:
                continue
            for d in obs.detector.get(v, ()):
                if d.object_class in params.hand_classes and d.box is not None:
                    if any(_pixel_in_box(pixel, d.box) for pixel in pixels):
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
                if t.is_line:
                    # The butt follows the hand; the segment keeps its direction and length.
                    assert t.endpoints_cm is not None
                    anchor = self._butt_anchor(t, hand.position)
                    t.set_endpoints(t.endpoints_cm + (hand.position + t.held_offset - anchor))
                else:
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

    def _butt_anchor(
        self, t: Track, hand_position: np.ndarray, *, resolve: bool = False
    ) -> np.ndarray:
        """The butt end of a line track (with `resolve`, tip / butt are first decided from the
        hand's position when it is nearer one end by the margin); the midpoint when nothing
        is resolved."""
        assert t.endpoints_cm is not None
        if resolve:
            d0, d1 = np.linalg.norm(t.endpoints_cm - hand_position, axis=1)
            if abs(d0 - d1) >= LINE_HAND_MARGIN_CM:
                self._set_tip(t, tip_is_endpoint_0=bool(d0 > d1), basis="hand_track")
        ends = t.tip_and_butt()
        return t.position.copy() if ends is None else ends[1].copy()

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
        if t.is_line:
            for v, o in cand.members.items():
                if depth_cm(cams[v], t.position) > 0 and (
                    self._line_view_cost(t, cams[v], o) < HUNGARIAN_FORBIDDEN
                ):
                    hits += 1
            return hits
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
        if self._is_geometric(t):
            if cand.line is not None:
                self._apply_candidate_line(t, cand, cams)
                self.line_births += 1
                extra["line"] = True
                extra["merged_views"] = list(cand.merged_views)
                extra["visible_length_cm"] = round(float(cand.visible_length_cm or 0.0), 2)
                extra["extended_by_prior"] = cand.extended
            else:
                self.line_point_births += 1
                extra["line"] = False
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
        if t.is_line and cand.line is None:
            # A point candidate returns a line track: the segment moves with its midpoint.
            assert t.endpoints_cm is not None
            t.set_endpoints(t.endpoints_cm + (cand.point - t.position))
        else:
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
            t.endpoint_velocity = None
        if cand.line is not None:
            self._apply_candidate_line(t, cand, cams)
            extra["line"] = True
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
        if self._is_geometric(t):
            self._record_line_support(t, cand.members, cams)

    def _apply_candidate_line(self, t: Track, cand: Candidate, cams: dict[str, Camera]) -> None:
        """A line candidate's geometry onto a (new or returning) track."""
        assert cand.line is not None and cand.endpoints is not None
        t.set_endpoints(cand.endpoints)
        t.direction_uncertainty_deg = LINE_DIRECTION_MEAS_DEG
        t.line_residuals = {v: round(r, 2) for v, r in cand.residuals.items()}
        t.merged_views = cand.merged_views
        t.line_this_frame = True
        t.endpoint_velocity = None
        t.last_observed_endpoints = t.endpoints_cm.copy() if t.endpoints_cm is not None else None
        t.last_endpoints_frame = t.last_observed_frame
        for v in cand.merged_views:
            self.line_merged_by_view[v] += 1
        self.line_frames += 1
        votes = Counter(o.colour_class for o in cand.members.values() if o.colour_class)
        fit = LineFit(
            line=cand.line,
            endpoints=cand.endpoints,
            members=cand.members,
            residuals=cand.residuals,
            merged_views=cand.merged_views,
            visible_length_cm=float(cand.visible_length_cm or 0.0),
            extended=cand.extended,
            plane_views=cand.line.plane_views,
        )
        assert self.line_prior is not None
        prior_length = self.line_prior.length_for(decisive_class(t.class_votes + votes))
        self._record_line_metrics(t, fit, cams, prior_length)

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
            if a.is_line and b.is_line:
                # Pipettes in the stand sit 2.9 cm apart: two lines are one object only when
                # nearly coincident, half the point distance and within the angle gate.
                angle, perpendicular, _ = line_distance(a.line(), b.line())
                duplicate = (
                    perpendicular <= 0.5 * self.params.duplicate_distance_cm
                    and angle <= self.params.line_gate_deg
                )
            else:
                duplicate = bool(
                    np.linalg.norm(a.position - b.position) <= self.params.duplicate_distance_cm
                )
            if duplicate:
                near[a.track_id].add(b.track_id)
                near[b.track_id].add(a.track_id)
                self.duplicate_pair_frames += 1
        return {tid: tuple(sorted(ids)) for tid, ids in near.items()}

    def _row(
        self, t: Track, frame: int, cams: dict[str, Camera], duplicates: tuple[str, ...] = ()
    ) -> Track3D:
        confidence, abstain = self._confidence(t, cams)
        possibly = tuple(sorted(set(t.possibly_same_as) | set(duplicates)))
        extra = self._line_row_fields(t) if self._is_geometric(t) else {}
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
            **extra,
        )

    def _line_row_fields(self, t: Track) -> dict[str, Any]:
        """The p2-tracker-lines fields: endpoints tip first when the tip is known, the
        direction from the first endpoint to the second, the class plurality, this frame's
        line residuals and merged views. `colour_identity` / `colour_confidence` are written
        only when `colour_hist` holds a histogram (the plurality bin and its share); the
        tracker leaves it empty and `battle-finebio-colour annotate` fills the fields on the
        rows afterwards from the plunger button at the butt end."""
        fields: dict[str, Any] = {"observed_class": t.observed_class}
        if t.is_line:
            assert t.endpoints_cm is not None
            resolved = t.tip_and_butt()
            ends = np.stack(resolved) if resolved is not None else t.endpoints_cm
            span = ends[1] - ends[0]
            norm = float(np.linalg.norm(span))
            direction = span / norm if norm > 1e-9 else np.asarray(t.direction)
            fields["direction"] = tuple(round(float(x), 4) for x in direction)
            fields["endpoints_cm"] = tuple(tuple(round(float(x), 3) for x in e) for e in ends)
            fields["tip_resolved"] = resolved is not None
            self.line_track_frames += 1
            if resolved is not None:
                self.line_tip_resolved_frames += 1
            if t.state in LOCALISED_STATES:
                fields["line_residual_px"] = dict(t.line_residuals) or None
                fields["merged_views"] = t.merged_views or None
        if t.colour_hist:
            identity = max(sorted(t.colour_hist), key=t.colour_hist.__getitem__)
            total = sum(t.colour_hist.values())
            fields["colour_identity"] = identity
            fields["colour_confidence"] = (
                round(float(t.colour_hist[identity] / total), 3) if total > 0 else None
            )
        return fields

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
        if self.params.line_classes:
            per_class.update(self._observed_class_metrics())
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

    def _observed_class_metrics(self) -> dict[str, dict[str, Any]]:
        """The geometric class's tracks counted per original class by their final plurality
        `observed_class`, so the per-pipette id numbers stay comparable with the point
        tracker's; `max_simultaneous` counts the live rows per frame and plurality class."""
        geometric = self.params.line_geometric_class
        tracks = [t for t in self.tracks.values() if t.object_class == geometric]
        out: dict[str, dict[str, Any]] = {}
        simultaneous: dict[tuple[int, str], int] = defaultdict(int)
        for row in self.rows:
            if row.object_class == geometric and row.state != "lost" and row.observed_class:
                simultaneous[(row.frame_index, row.observed_class)] += 1
        max_simultaneous: dict[str, int] = defaultdict(int)
        for (_frame, cls), n in simultaneous.items():
            max_simultaneous[cls] = max(max_simultaneous[cls], n)
        for cls in sorted(self.params.line_classes):
            mine = [t for t in tracks if t.observed_class == cls]
            if not mine and cls not in max_simultaneous:
                continue
            out[cls] = {
                "tracks_born": len(mine),
                "max_simultaneous": max_simultaneous.get(cls, 0),
                "fragmentation": max(0, len(mine) - max_simultaneous.get(cls, 0)),
                "tracks_lost": sum(1 for t in mine if t.state == "lost"),
                "counted_by": f"plurality observed_class of the {geometric} tracks",
            }
        return out

    def _line_metrics(self) -> dict[str, Any]:
        params = self.params
        if not params.line_classes:
            return {"enabled": False}
        geometric = params.line_geometric_class
        tracks = [t for t in self.tracks.values() if t.object_class == geometric]

        def summary(values: Sequence[float]) -> dict[str, Any]:
            arr = np.asarray(values, dtype=np.float64)
            if arr.size == 0:
                return {"n": 0, "median": None, "p90": None}
            return {
                "n": int(arr.size),
                "median": round(float(np.median(arr)), 3),
                "p90": round(float(np.percentile(arr, 90)), 3),
            }

        visible = [length for _, length, _ in self.line_lengths]
        deviation = [abs(d) for _, _, d in self.line_lengths]
        by_class_length: dict[str, list[float]] = defaultdict(list)
        for cls, length, _ in self.line_lengths:
            by_class_length[cls or "undecided"].append(length)
        agree = sum(max(t.class_votes.values()) for t in tracks if t.class_votes)
        total = sum(sum(t.class_votes.values()) for t in tracks)
        multi_view_frames = self.line_frames + self.line_point_frames
        return {
            "enabled": True,
            "geometric_class": geometric,
            "classes": list(params.line_classes),
            "prior": self.line_prior.as_dict() if self.line_prior is not None else None,
            "tracks": len(tracks),
            "line_tracks": sum(1 for t in tracks if t.is_line),
            "line_births": self.line_births,
            "point_births": self.line_point_births,
            "frames": {
                "line": self.line_frames,
                "line_prediction_aided": self.line_aided_frames,
                "point_fallback": self.line_point_frames,
                "line_fraction": (
                    round(self.line_frames / multi_view_frames, 4) if multi_view_frames else None
                ),
                "single_view": self.line_single_view_frames,
                "degenerate_fits": self.line_degenerate_fits,
                "extended_by_prior": self.line_extended_frames,
            },
            "loo_residual_px": summary(self.line_loo_px),
            "loo_angle_deg": summary(self.line_loo_deg),
            "length_cm": {
                "visible": summary(visible),
                "visible_spread_p10_p90": (
                    round(float(np.percentile(visible, 90) - np.percentile(visible, 10)), 3)
                    if visible
                    else None
                ),
                "deviation_from_prior": summary(deviation),
                "by_class_median": {
                    cls: round(float(np.median(v)), 3) for cls, v in sorted(by_class_length.items())
                },
            },
            "merged_views_by_view": dict(sorted(self.line_merged_by_view.items())),
            "tip_resolved_fraction": (
                round(self.line_tip_resolved_frames / self.line_track_frames, 4)
                if self.line_track_frames
                else None
            ),
            "tip_resolutions_by_basis": dict(sorted(self.line_tip_resolutions.items())),
            "class_agreement": round(agree / total, 4) if total else None,
            "class_votes_by_track": {
                t.track_id: dict(sorted(t.class_votes.items())) for t in tracks if t.class_votes
            },
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
            "lines": self._line_metrics(),
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
    line_prior: LinePrior | None = None,
) -> TrackerOutput:
    """Run the tracker over `frames` (default: every frame with a row). With
    `params.containers` set, the container volumes are built from the detector rows first
    (`build_container_volumes`; `rig_static` = class -> point_cm from the rig, optional) and
    their report lands in `metrics["extensions"]["contained"]`. With `params.line_classes`
    set, the length prior is `line_prior` or read from `params.line_prior_path`."""
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
    tracker = MultiviewTracker(fixed_cams, fpv_source, params, volumes, line_prior=line_prior)
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
    ext.add_argument(
        "--line-classes",
        default=None,
        help="comma-separated classes tracked as 3D line segments of one geometric class "
        "(p2-tracker-lines); `pipette` expands to the four pipette classes",
    )
    ext.add_argument(
        "--line-geometric-class",
        default=TrackerParams.line_geometric_class,
        help="the class name the line classes are tracked under",
    )
    ext.add_argument(
        "--line-prior",
        default=TrackerParams.line_prior_path,
        help="pipettes config with length_cm / length_spread_cm (the soft length prior)",
    )
    ext.add_argument("--line-gate-deg", type=float, default=TrackerParams.line_gate_deg)
    ext.add_argument(
        "--line-gate-cm",
        type=float,
        default=TrackerParams.line_gate_cm,
        help="perpendicular distance at the midpoint between the predicted and fitted line, "
        "plus the track's uncertainty",
    )
    ext.add_argument(
        "--line-min-pair-angle-deg", type=float, default=TrackerParams.line_min_pair_angle_deg
    )
    ext.add_argument(
        "--line-merged-width-factor", type=float, default=TrackerParams.line_merged_width_factor
    )
    ext.add_argument(
        "--line-merged-extent-factor",
        type=float,
        default=TrackerParams.line_merged_extent_factor,
    )
    ext.add_argument(
        "--line-max-axis-residual-px",
        type=float,
        default=TrackerParams.line_max_axis_residual_px,
    )
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
        line_classes=parse_line_classes(args.line_classes),
        line_geometric_class=args.line_geometric_class,
        line_prior_path=args.line_prior,
        line_gate_deg=args.line_gate_deg,
        line_gate_cm=args.line_gate_cm,
        line_min_pair_angle_deg=args.line_min_pair_angle_deg,
        line_merged_width_factor=args.line_merged_width_factor,
        line_merged_extent_factor=args.line_merged_extent_factor,
        line_max_axis_residual_px=args.line_max_axis_residual_px,
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
