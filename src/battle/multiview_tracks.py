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
LIVE_STATES = ("observed", "single_view", "coasting")
LOCALISED_STATES = ("observed", "single_view")
HUNGARIAN_FORBIDDEN = 1e9


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

    def as_dict(self) -> dict[str, Any]:
        out = {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "gates"}
        out["gates"] = dict(self.gates.__dict__)
        return out


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


# --------------------------------------------------------------------------- birth


@dataclass
class Candidate:
    object_class: str
    point: np.ndarray
    members: dict[str, Obs]
    residuals: dict[str, float]
    fpv_rule: bool = False


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
    ) -> None:
        self.fixed_cams = dict(fixed_cams)
        self.fixed_views = tuple(fixed_cams)
        self.fpv_source = fpv_source
        self.params = params or TrackerParams()
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

    # -- the frame step

    def step(self, frame: int, rows: Iterable[FineBioObservation]) -> None:
        params = self.params
        dt = 1 if self._previous_frame is None else max(1, frame - self._previous_frame)
        self._previous_frame = frame
        cams = self._cams_for(frame)
        obs = select_observations(frame, rows, params)

        for t in self.live_tracks():
            t.uncertainty_cm = min(
                params.max_uncertainty_cm,
                float(np.hypot(t.uncertainty_cm, params.process_noise_cm * dt)),
            )

        assigned = self._associate(cams, obs)
        taken: set[tuple[str, int]] = set()
        already_coasting = [t for t in self.live_tracks() if t.state == "coasting"]
        for t in self.live_tracks():
            if t.state not in LOCALISED_STATES:
                continue
            mine = {v: o for (tid, v), o in assigned.items() if tid == t.track_id}
            self._update(t, mine, cams, frame, dt)
            for v, o in mine.items():
                taken.add((v, o.index))
        for t in already_coasting:
            self._coast(t, frame, dt)
        self._handoff_reseeds(frame, cams, obs)

        unassigned = {
            v: [o for o in items if (v, o.index) not in taken] for v, items in obs.tracked.items()
        }
        candidates = birth_candidates(unassigned, cams, self.fixed_views, params)
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

    def _associate(
        self, cams: dict[str, Camera], obs: FrameObservations
    ) -> dict[tuple[str, str], Obs]:
        pairs: list[tuple[float, str, str, Obs]] = []
        for t in self.live_tracks():
            if t.state not in LOCALISED_STATES:
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
        self, t: Track, mine: dict[str, Obs], cams: dict[str, Camera], frame: int, dt: int
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
            t.state = "single_view"
            t.frames_unobserved = 0
            t.last_observed_frame = frame
        elif not mine:
            self._start_coasting(t, frame, dt)
            return
        t.support_views = tuple(sorted(mine))
        t.support_slots = {v: o.slot for v, o in mine.items()}
        for v, o in mine.items():
            self._record_support(t, v, o, cams[v], frame)

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

    def _start_coasting(self, t: Track, frame: int, dt: int) -> None:
        if t.state != "coasting":
            self._event(frame, t.track_id, "coasting", last_observed_frame=t.last_observed_frame)
        t.state = "coasting"
        t.support_views = ()
        t.support_slots = {}
        t.residuals = {}
        self._coast(t, frame, dt)

    def _coast(self, t: Track, frame: int, dt: int) -> None:
        params = self.params
        t.frames_unobserved = frame - t.last_observed_frame
        t.uncertainty_cm = min(
            params.max_uncertainty_cm, t.uncertainty_cm + params.coast_growth_cm * dt
        )
        if t.frames_unobserved > params.coast_timeout_frames:
            t.state = "lost"
            t.lost_frame = frame
            self._event(
                frame,
                t.track_id,
                "lost",
                frames_unobserved=t.frames_unobserved,
                last_observed_frame=t.last_observed_frame,
            )

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

    def _births_and_reacquisitions(
        self, frame: int, candidates: list[Candidate], cams: dict[str, Camera]
    ) -> None:
        params = self.params
        coasting = [t for t in self.tracks.values() if t.state == "coasting"]
        inside: dict[int, list[str]] = defaultdict(list)  # candidate index -> coasting ids
        for ci, cand in enumerate(candidates):
            for t in coasting:
                if t.object_class != cand.object_class:
                    continue
                hits = 0
                for v, o in cand.members.items():
                    pixel = project(cams[v], t.position)
                    if pixel is not None and np.linalg.norm(pixel - o.point) <= gate_px(
                        t, cams[v], params
                    ):
                        hits += 1
                if hits >= params.reacquire_min_views:
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
        )
        return t

    def _resume(self, frame: int, t: Track, cand: Candidate, cams: dict[str, Camera]) -> None:
        latency = frame - t.last_observed_frame
        t.position = cand.point.copy()
        t.uncertainty_cm = self.params.base_uncertainty_cm
        t.state = "observed"
        t.last_observed_frame = frame
        t.frames_unobserved = 0
        t.reacquisitions += 1
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
        if t.state == "coasting":
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
) -> TrackerOutput:
    by_frame: dict[int, list[FineBioObservation]] = defaultdict(list)
    for row in rows:
        by_frame[row.frame_index].append(row)
    frame_list = list(frames) if frames is not None else sorted(by_frame)
    tracker = MultiviewTracker(fixed_cams, fpv_source, params)
    for frame in frame_list:
        tracker.step(frame, by_frame.get(frame, []))
    return TrackerOutput(
        rows=tracker.rows,
        events=tracker.events,
        residuals=tracker.residual_rows,
        metrics=tracker.identity_metrics(reference_labels),
    )


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
    if args.gates is not None:
        gates = Gates.from_rig(json.loads(args.gates.read_text(encoding="utf-8")))
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
    )
    frames = parse_frames(args.frames, {r.frame_index for r in rows})
    labels = (
        json.loads(args.reference_labels.read_text(encoding="utf-8"))
        if args.reference_labels is not None
        else None
    )
    output = run_tracker(rows, cameras_from_config(config), fpv_source, frames, params, labels)
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
