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
  Sep 29, after the first full run (10-22% of line rows longer than twice a pipette, 6% of
  midpoint steps over 10 cm): every runaway extent began in a prediction-aided fit whose
  in-plane direction nobody checked. The aided fit now rejects a plane whose own extent on
  the aided line exceeds the merged factor times the prior and drops rays far along the line;
  a written extent is clamped to that length (`extent_clamped`); two-plane updates must
  overlap along the line as births must; a view looking along the shaft (under
  `line_min_view_angle_deg`) gives no extent and no single-view update; a midpoint step over
  `line_max_step_cm` a frame is cut to it and the row demoted to prediction-aided
  (`line_update`); the held offset is taken only from a frame where the hand was within reach
  of the butt (`butt_to_hand_cm` on the rows); and an observation whose class differs from a
  track's decisive plurality is refused while a same-class candidate exists, a track whose
  votes flip to a second colour splitting into a new id (`class_split`).
  Sep 29, disposable tips: a `*_tip` detector box on the end of a pipette's mask axis is
  attached to that observation (`attach_tip_box`), extends the view's extent to the tip,
  marks the tip end (the strongest tip / butt basis), casts one class vote and drives a
  two-state length prior (`LinePrior.expected_length`: bare body, or body plus tip) with a
  hysteresis on the track's `tip_attached` state. A fourth tip / butt basis reads the mask's
  end widths (`mask_end_widths_px`): the 8-channel pipette's wide end is the manifold that
  carries the tips, a single-channel pipette's wide end is its grip (the butt); it decides
  for an unresolved track or one a hand box named, and reports its votes.
  Sep 29, v4 (after `finebio_tipseg`): the rows carry the long-thin-tail tip side
  (`tip_side`) and the mask's terminal centroid at each end (`body_end_px`). The tip side is
  a fifth tip / butt basis (`tail`), ranked under the tip box and the hand track and over
  the hand box and the widths, for every class but the 8-channel (its manifold is its wide
  end; the tail rule is unvalidated there and the class rule stays). The terminal centroid
  is the end observation for the extent along the line (the plane and the residual still
  read the axis). `tip_attached` no longer follows the tip boxes: it reads the track's
  measured butt-to-tip length over the last line frames against the class's two length
  modes (`length_modes_cm` in the prior; bare-only classes are never "attached"), with a
  hysteresis band around the midpoint; the tip boxes corroborate and are reported as such.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
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
    attach_tip_box,
    axis_residual,
    clamp_interval,
    fit_line,
    is_elongated,
    line_distance,
    line_endpoints,
    loo_residual,
    plane_from_axis,
    point_residual,
    ray_from_point,
    view_angle_to_line_deg,
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
# Sep 29 (disposable tips): the detector's tip classes and the pipette class each one fits.
# A tip box that sits on the end of a pipette's mask axis in a view is attached to that
# observation (`multiview_lines.attach_tip_box`): it extends the view's extent to the far edge
# of the box, marks that end as the tip, casts one class vote for its pipette class, and
# feeds the track's two-state length prior (bare body, or body plus tip). Tip boxes in a rack,
# in the trash or loose are never attached and the line tracker ignores them.
LINE_TIP_CLASSES: tuple[tuple[str, str], ...] = (
    ("blue_tip", "blue_pipette"),
    ("yellow_tip", "yellow_pipette"),
    ("red_tip", "red_pipette"),
    ("8_channel_tip", "8_channel_pipette"),
)
TIP_BASIS = "tip_box"
# Sep 29, tip / butt by the width profile (the fourth basis). Per view, the two end widths of
# the mask (`mask_end_widths_px`) name the wide end when their ratio reaches the minimum; the
# class rule says which end that is: an 8-channel pipette's wide end is the manifold that
# carries the tips (the tip end), a single-channel pipette's wide end is the grip and plunger
# (the butt). Votes accumulate over a window of line frames and decide once the margin is
# met; the basis overrides an unresolved track or one decided by a hand box, never a tip box
# or a hand track.
WIDTH_BASIS = "width"
LINE_WIDTH_RATIO_MIN = 1.25
LINE_WIDTH_VOTE_WINDOW = 30
LINE_WIDTH_VOTE_MARGIN = 5
LINE_WIDE_TIP_CLASSES: tuple[str, ...] = ("8_channel_pipette",)
# Sep 29 v4, the fifth basis: the row's `tip_side` (the axis end with the longer thin tail,
# `finebio_observations.tip_side_from_tails`) is matched to the track endpoint its axis end
# projects nearest and cast as a vote; the net vote over a window of line frames decides at
# a margin. It is skipped for the wide-tip classes (the 8-channel), whose tail rule is
# unvalidated, and it never overrides a tip box while the state is on or a hand track.
TAIL_BASIS = "tail"
LINE_TAIL_VOTE_WINDOW = 30
LINE_TAIL_VOTE_MARGIN = 3
# Sep 29 v4, the tip state from the 3D length: the median of the visible butt-to-tip length
# over the last `LINE_TIP_LENGTH_WINDOW` measured line frames (at least the minimum) is
# compared with the class's two modes; the state flips on when the median clears the
# midpoint by the hysteresis band and off when it falls under it by the band; a median more
# than the truncated margin under the bare mode is a mask the hand cut short and decides
# nothing.
LINE_TIP_LENGTH_WINDOW = 15
LINE_TIP_LENGTH_MIN_FRAMES = 5
LINE_TIP_LENGTH_HYSTERESIS_CM = 1.0
LINE_TIP_LENGTH_TRUNCATED_CM = 3.0


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
    # -- Sep 29, geometry corrections after the first run. A view whose ray to the segment is
    # within this angle of the line looks along the shaft: its extent on the line and a
    # single-view update from it are unconstrained and are not used. A midpoint step over
    # this many cm a frame (15 cm at 30 fps is 4.5 m/s) is cut to it and the row demoted to
    # prediction-aided. The class veto: a track whose plurality class holds at least the
    # share of its votes over at least the frames refuses an observation of another class
    # while a same-class candidate exists in that view; a track whose votes over the last
    # window of frames give a second colour at least the split share continues under a new
    # id.
    line_min_view_angle_deg: float = 10.0
    line_max_step_cm: float = 15.0
    line_class_veto_share: float = 0.75
    line_class_veto_frames: int = 30
    line_class_split_share: float = 0.4
    line_class_split_window: int = 60
    # -- Sep 29, disposable tips (on whenever line classes are on). `*_tip` detector boxes are
    # attached to a line observation in the same view when the box centre lies within the
    # association gate of the axis line and beyond the body end; the track's `tip_attached`
    # state turns on when at least `line_tip_on_frames` of the last `line_tip_on_window` line
    # frames carried an attached tip in some view and off when none of the last
    # `line_tip_off_window` did.
    line_tip_boxes: bool = True
    line_tip_classes: tuple[tuple[str, str], ...] = LINE_TIP_CLASSES
    line_tip_on_frames: int = 5
    line_tip_on_window: int = 8
    line_tip_off_window: int = 15
    # The width basis (module constants above): the classes whose wide end is the tip.
    line_wide_tip_classes: tuple[str, ...] = LINE_WIDE_TIP_CLASSES
    line_width_ratio_min: float = LINE_WIDTH_RATIO_MIN
    line_width_vote_window: int = LINE_WIDTH_VOTE_WINDOW
    line_width_vote_margin: int = LINE_WIDTH_VOTE_MARGIN
    # Off (the brief's order): tip box, hand track, hand box, then the widths as a
    # tie-breaker. On: a decisive width vote outranks the hand (the variant run reports it).
    line_width_over_hand: bool = False
    # -- Sep 29 v4. The tail basis (the row's `tip_side`; module constants above) and the
    # 3D-length tip state; either off reproduces the v3 behaviour for that piece.
    line_tail_basis: bool = True
    line_tail_vote_window: int = LINE_TAIL_VOTE_WINDOW
    line_tail_vote_margin: int = LINE_TAIL_VOTE_MARGIN
    # Off (the brief's ranking): the tail sits under the hand track. On (a reported variant,
    # as `line_width_over_hand` was): a decisive tail vote outranks the hand track, which
    # then only measures the butt offset.
    line_tail_over_hand: bool = False
    line_tip_length_rule: bool = True
    # With the length rule: a birth pair and a fit may span the body plus its tip whether or
    # not a tip box attached (a mask may cover a tip on any frame); off, only with a box (v3).
    line_birth_tip_slack: bool = True
    line_tip_length_window: int = LINE_TIP_LENGTH_WINDOW
    line_tip_length_min_frames: int = LINE_TIP_LENGTH_MIN_FRAMES
    line_tip_length_hysteresis_cm: float = LINE_TIP_LENGTH_HYSTERESIS_CM
    line_tip_length_truncated_cm: float = LINE_TIP_LENGTH_TRUNCATED_CM

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

    @property
    def tip_boxes_on(self) -> bool:
        return bool(self.line_classes) and self.line_tip_boxes

    def pipette_class_of_tip(self, tip_class: str) -> str | None:
        return dict(self.line_tip_classes).get(tip_class)

    def tip_class_of_pipette(self, pipette_class: str | None) -> str | None:
        for tip, pipette in self.line_tip_classes:
            if pipette == pipette_class:
                return tip
        return None


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
    `length_for(cls)` is the class median when the class is known, else the shared value.

    Sep 29 (disposable tips): the prior is two-state. `bare` holds the bare-body length per
    pipette class (`bare_length_cm` in the config) and `tip` the tip length per tip class
    (`tip_length_cm`); `expected_length(cls, tip_class, attached)` is the bare length, plus
    the tip when a tip is attached. A config without the two-state keys gives the old
    single-state prior (bare = `length_for`, tip = 0), so earlier files still load.

    Sep 29 v4: `modes` holds, per pipette class, the two measured body-plus-tip length modes
    (`length_modes_cm` in the config: `bare` and `with_tip`, the latter None for a class
    whose with-tip length was never measured). `modes_for(cls)` returns them, and for a
    config without the block derives them from the two-state keys (bare, bare plus tip) so
    a test prior works; with the block present a class it does not name is bare-only. The
    tip state reads the modes; the completion prior still reads `expected_length`."""

    length_cm: float
    spread_cm: float
    per_class: dict[str, float] = field(default_factory=dict)
    source: str = "default"
    bare: dict[str, float] = field(default_factory=dict)
    tip: dict[str, float] = field(default_factory=dict)
    tip_of_class: dict[str, str] = field(default_factory=lambda: dict(LINE_TIP_CLASSES))
    modes: dict[str, tuple[float, float | None]] | None = None

    def length_for(self, cls: str | None) -> float:
        if cls is not None and cls in self.per_class:
            return self.per_class[cls]
        return self.length_cm

    def bare_for(self, cls: str | None) -> float:
        if cls is not None and cls in self.bare:
            return self.bare[cls]
        return self.length_for(cls)

    def tip_for(self, tip_class: str | None) -> float:
        """The tip length of a tip class; a class the config does not name gets the median
        of the named ones; no tip lengths at all means 0 (the single-state prior)."""
        if tip_class is not None and tip_class in self.tip:
            return self.tip[tip_class]
        if self.tip:
            return float(np.median(list(self.tip.values())))
        return 0.0

    def tip_class_for(self, cls: str | None) -> str | None:
        for tip, pipette in self.tip_of_class.items():
            if pipette == cls:
                return tip
        return None

    def expected_length(
        self, cls: str | None, tip_class: str | None = None, *, attached: bool = False
    ) -> float:
        if not attached:
            return self.bare_for(cls)
        return self.bare_for(cls) + self.tip_for(tip_class or self.tip_class_for(cls))

    @property
    def two_state(self) -> bool:
        return bool(self.bare) and bool(self.tip)

    def modes_for(self, cls: str | None) -> tuple[float, float | None]:
        """(bare mode, with-tip mode or None) of a class for the 3D-length tip state."""
        if self.modes is not None:
            if cls is not None and cls in self.modes:
                return self.modes[cls]
            return self.bare_for(cls), None
        bare = self.bare_for(cls)
        tip = self.tip_for(self.tip_class_for(cls)) if self.two_state else 0.0
        return bare, (bare + tip if tip > 0 else None)

    def as_dict(self) -> dict[str, Any]:
        return {
            "length_cm": self.length_cm,
            "spread_cm": self.spread_cm,
            "per_class": dict(sorted(self.per_class.items())),
            "bare_length_cm": dict(sorted(self.bare.items())),
            "tip_length_cm": dict(sorted(self.tip.items())),
            "two_state": self.two_state,
            "length_modes_cm": (
                None
                if self.modes is None
                else {
                    cls: {"bare": bare, "with_tip": with_tip}
                    for cls, (bare, with_tip) in sorted(self.modes.items())
                }
            ),
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
    bare = {
        str(cls): float(v) for cls, v in (doc.get("bare_length_cm") or {}).items() if v is not None
    }
    tip = {
        str(cls): float(v) for cls, v in (doc.get("tip_length_cm") or {}).items() if v is not None
    }
    modes: dict[str, tuple[float, float | None]] | None = None
    if doc.get("length_modes_cm") is not None:
        modes = {}
        for cls, block in doc["length_modes_cm"].items():
            if not isinstance(block, Mapping) or block.get("bare") is None:
                continue
            with_tip = block.get("with_tip")
            modes[str(cls)] = (float(block["bare"]), None if with_tip is None else float(with_tip))
    return LinePrior(
        length_cm=float(doc["length_cm"]),
        spread_cm=float(doc.get("length_spread_cm", 0.0)),
        per_class={cls: float(np.median(v)) for cls, v in medians.items()},
        source=str(path),
        bare=bare,
        tip=tip,
        modes=modes,
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
    # Sep 29: the mask's width at each axis end (`mask_end_widths_px`), the width basis.
    end_widths_px: tuple[float, float] | None = None
    # Sep 29 v4: the row's long-thin-tail tip side (an index into `axis_px`) and the mask's
    # terminal centroid at each end, in the order of `axis_px` (the end observations).
    tip_side: int | None = None
    body_ends_px: tuple[tuple[float, float], tuple[float, float]] | None = None
    # -- Sep 29, disposable tips (set on a copy by `attach_tip`): the attached `*_tip` detector
    # box, its class, the index of the axis end it extends (`axis_px` is then the extended
    # axis and `body_axis_px` the mask's own), how it attached (`axis`: on this observation's
    # own axis; `track`: on the track's projected line, for a compact or box-only observation,
    # no geometry changed) and the box centre, the pixel the tip / butt resolution reads.
    tip_box: tuple[float, float, float, float] | None = None
    tip_class: str | None = None
    tip_end: int | None = None
    tip_mode: str | None = None
    tip_pixel: np.ndarray | None = None
    body_axis_px: tuple[tuple[float, float], tuple[float, float]] | None = None

    @property
    def tip_attached(self) -> bool:
        return self.tip_box is not None

    @property
    def extent(self) -> tuple[float, float] | None:
        if self.box is None:
            return None
        return (self.box[2] - self.box[0], self.box[3] - self.box[1])

    def axis_obs(self, weight: float = 1.0) -> AxisObs:
        """The `multiview_lines` record of this observation: the mask axis as the plane
        constraint when elongated, the centroid's ray otherwise. Sep 29 v4: the terminal
        centroids are the end observations for the extent along the line when the row
        carries them (the tip box's far edge still stands in on the end it extended)."""
        ends_px = None
        if self.axis_px is not None and self.body_ends_px is not None:
            ends = np.asarray(self.body_ends_px, dtype=np.float64).reshape(2, 2).copy()
            if self.tip_mode == "axis" and self.tip_end is not None:
                ends[self.tip_end] = np.asarray(self.axis_px[self.tip_end], dtype=np.float64)
            ends_px = ends
        return AxisObs(
            self.view,
            None if self.axis_px is None else np.asarray(self.axis_px, dtype=np.float64),
            self.point,
            self.elongation,
            weight,
            ends_px=ends_px,
        )


@dataclass
class FrameObservations:
    frame: int
    tracked: dict[str, list[Obs]]  # per view, the rows the tracker associates
    detector: dict[str, list[Obs]]  # per view, every detector row (confirmation, re-seed)
    # Sep 29: per view, the `*_tip` detector boxes the line tracker may attach (empty with
    # the line classes off).
    tips: dict[str, list[Obs]] = field(default_factory=dict)


def attach_tip(obs: Obs, tip: Obs, attachment: Any, mode: str) -> Obs:
    """A copy of a line observation with a tip box attached: in `axis` mode the axis is the
    attachment's extended one (the extent then reaches the tip), in `track` mode nothing
    geometric changes (the observation has no axis of its own)."""
    out = replace(obs)
    out.tip_box = tip.box
    out.tip_class = tip.object_class
    out.tip_end = int(attachment.end)
    out.tip_mode = mode
    assert tip.box is not None
    out.tip_pixel = np.array(
        [0.5 * (tip.box[0] + tip.box[2]), 0.5 * (tip.box[1] + tip.box[3])], dtype=np.float64
    )
    if mode == "axis":
        out.body_axis_px = obs.axis_px
        out.axis_px = tuple(tuple(float(x) for x in e) for e in attachment.axis_px)  # type: ignore[assignment]
    return out


def attach_tips_in_view(
    items: Sequence[Obs],
    tips: Sequence[Obs],
    gate_px: float,
    *,
    side_of: Callable[[Obs], int | None] | None = None,
) -> dict[int, Obs]:
    """Attach the view's tip boxes to its elongated line observations, nearest axis first,
    one tip per observation and one observation per tip. `side_of(obs)` names the axis end
    a tip may attach to (the tip side when the track knows it; None for either). Returns
    ``{index into items: the observation copy with the tip}``."""
    pairs: list[tuple[float, int, int, Any]] = []
    for i, o in enumerate(items):
        if o.axis_px is None or not is_elongated(o.axis_obs()):
            continue
        side = side_of(o) if side_of is not None else None
        axis = np.asarray(o.axis_px, dtype=np.float64)
        for j, tip in enumerate(tips):
            if tip.box is None:
                continue
            attachment = attach_tip_box(axis, tip.box, gate_px, side=side)
            if attachment is not None:
                pairs.append((attachment.across_px, i, j, attachment))
    out: dict[int, Obs] = {}
    used_tips: set[int] = set()
    for _across, i, j, attachment in sorted(pairs, key=lambda p: (p[0], p[1], p[2])):
        if i in out or j in used_tips:
            continue
        out[i] = attach_tip(items[i], tips[j], attachment, "axis")
        used_tips.add(j)
    return out


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
        end_widths_px=row.mask_end_widths_px,
        tip_side=row.tip_side,
        body_ends_px=row.body_end_px,
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
    tips: dict[str, list[Obs]] = {}
    if params.line_classes:
        for items in tracked.values():
            _to_geometric_class(items, params)
        for items in detector.values():
            _to_geometric_class(items, params)
        if params.tip_boxes_on:
            tip_classes = {tip for tip, _ in params.line_tip_classes}
            for view, items in detector.items():
                mine = [o for o in items if o.object_class in tip_classes and o.box is not None]
                if mine:
                    tips[view] = mine
    return FrameObservations(frame=frame, tracked=dict(tracked), detector=dict(detector), tips=tips)


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
    # -- Sep 29 defects: what moved the segment this frame (`Track3D.line_update`), whether
    # the extent was clamped, the frames that cast a class vote and the last window of
    # per-frame votes (the class split), and the butt-to-hand offset measured on a localised
    # frame with the hand within reach (the only offset a hold may follow).
    line_update: str | None = None
    extent_clamped: bool = False
    class_vote_frames: int = 0
    recent_class_votes: list[Counter] = field(default_factory=list)
    butt_offset: np.ndarray | None = None
    butt_offset_hand: str | None = None
    butt_offset_frame: int | None = None
    # -- Sep 29, disposable tips: per line frame whether some view carried an attached tip
    # box (the hysteresis window), the tip classes those frames saw, the two-state flag
    # (None until the windows decide), the tip class the state follows, the views that
    # attached a tip this frame and the basis of the tip / butt resolution.
    tip_history: list[bool] = field(default_factory=list)
    tip_class_history: list[str | None] = field(default_factory=list)
    tip_ever: bool = False
    tip_attached: bool | None = None
    tip_class: str | None = None
    tip_views_this_frame: tuple[str, ...] = ()
    tip_basis: str | None = None
    # The width basis: per line frame the net vote (+1 endpoint 0 is the tip, -1 endpoint 1).
    width_votes: list[int] = field(default_factory=list)
    # -- Sep 29 v4: the tail basis votes (as the width votes), the visible butt-to-tip lengths
    # of the last measured line fits (the 3D-length tip state) and the median the state
    # read this frame (None while too few frames measured).
    tail_votes: list[int] = field(default_factory=list)
    length_history: list[float] = field(default_factory=list)
    tip_rule_length_cm: float | None = None

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
    clamped: bool = False


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
    # True when the extent was cut to the merged factor times the prior (Sep 29), and when no
    # view could measure an extent and the prediction's was kept (not a visible length).
    clamped: bool = False
    extent_from_prediction: bool = False


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
    """The interval of the line parameter the observation covers: its end observations'
    rays' closest points (an elongated mask; v4: the terminal centroids when the row has
    them), else the centroid ray's."""
    axis = o.axis_obs()
    pixels = axis.extent_px if is_elongated(axis) else o.point.reshape(1, 2)
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


@dataclass
class CompletedExtent:
    """`complete_extent` result: the endpoints after the clamp and the soft prior, the
    visible length before either, whether the prior reconstructed an end, whether the clamp
    cut the extent, the compact views whose ray point lies off this shaft (not this pipette
    on this line), the views that gave an extent, and whether the extent is the prediction's
    because no view could measure one."""

    endpoints: np.ndarray
    visible_length_cm: float
    extended: bool
    clamped: bool
    far_rays: tuple[str, ...]
    extent_views: tuple[str, ...]
    from_prediction: bool


def _sees_broadside(cam: Camera, line: Line3D, params: TrackerParams) -> bool:
    return view_angle_to_line_deg(cam, line) >= params.line_min_view_angle_deg


def complete_extent(
    line: Line3D,
    cams_and_obs: Sequence[tuple[Camera, AxisObs]],
    params: TrackerParams,
    prior_length_cm: float,
    prior_spread_cm: float,
    tip_hint: np.ndarray | None = None,
    fallback_extent: np.ndarray | None = None,
    ceiling_length_cm: float | None = None,
) -> CompletedExtent | None:
    """The soft prior on a fitted line's visible extent. The visible extent is the axis
    views' (`line_endpoints`) widened by the compact views' centroid rays (their closest
    points on the line are on the shaft too). Within `prior_spread_cm` of the prior it is left
    as seen; shorter, it is completed to the prior at the tip end (the visible end nearer
    `tip_hint`, the track's predicted tip) unless every view reaches that end and not the
    other, in which case the other end is the one the masks missed; without a hint, from the
    end more views reach, both ends by half when tied.

    Sep 29 corrections: a view whose ray runs within `line_min_view_angle_deg` of the line
    looks along the shaft and gives no extent (its rays meet the line at a grazing angle); a
    ray point beyond the axis views' interval by more than the length they may not see plus
    `line_gate_cm` is another object's and is reported in `far_rays`, not used; an extent
    over `line_merged_extent_factor` times the prior is cut to that length from the
    better-supported end (`clamped`). When no view can give an extent, `fallback_extent`
    (the track's predicted endpoints) is projected onto the line (`from_prediction`); None
    without one. Sep 29 v4: `ceiling_length_cm` (the body plus its tip; the prior when not
    given) is the length the far-ray slack and the clamp read, so a mask that covers a tip
    is not cut or refused while the state still says bare."""
    ceiling = (
        prior_length_cm if ceiling_length_cm is None else max(ceiling_length_cm, prior_length_cm)
    )
    extent_pairs = [
        (cam, obs)
        for cam, obs in cams_and_obs
        if is_elongated(obs) and _sees_broadside(cam, line, params)
    ]
    extent = line_endpoints(line, extent_pairs) if extent_pairs else None
    from_prediction = False
    if extent is not None:
        lo, hi = line.parameter(extent.visible[0]), line.parameter(extent.visible[1])
        support = list(extent.support)
        n_views = len(extent.per_view)
        extent_views = list(extent.per_view)
    elif fallback_extent is not None:
        lo, hi = sorted(line.parameter(e) for e in np.asarray(fallback_extent).reshape(2, 3))
        support, n_views, extent_views, from_prediction = [0, 0], 0, [], True
    else:
        return None
    slack = max(0.0, ceiling - (hi - lo))
    far_rays: list[str] = []
    for cam, obs in cams_and_obs:
        if is_elongated(obs) or not _sees_broadside(cam, line, params):
            continue
        parameter = line.parameter_closest_to_ray(*ray_from_point(cam, obs.centroid_px))
        if max(0.0, lo - parameter, parameter - hi) > slack + params.line_gate_cm:
            far_rays.append(obs.view)
            continue
        n_views += 1
        extent_views.append(obs.view)
        if parameter < lo:
            lo, support[0] = parameter, 1
        elif parameter <= lo + END_TOLERANCE_CM:
            support[0] += 1
        if parameter > hi:
            hi, support[1] = parameter, 1
        elif parameter >= hi - END_TOLERANCE_CM:
            support[1] += 1
    visible = hi - lo
    lo, hi, clamped = clamp_interval(
        lo, hi, (support[0], support[1]), params.line_merged_extent_factor * ceiling
    )

    def result(lo: float, hi: float, extended: bool) -> CompletedExtent:
        return CompletedExtent(
            endpoints=np.stack([line.point_at(lo), line.point_at(hi)]),
            visible_length_cm=visible,
            extended=extended,
            clamped=clamped,
            far_rays=tuple(sorted(far_rays)),
            extent_views=tuple(sorted(extent_views)),
            from_prediction=from_prediction,
        )

    if visible >= prior_length_cm - prior_spread_cm:
        return result(lo, hi, False)
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
    return result(lo, hi, missing > END_TOLERANCE_CM)


def _ends_corroborated(view: str, per_view: dict[str, tuple[float, float]]) -> bool:
    """Sep 29 v4: a view whose extent on the line runs past the state's prior (up to the
    body-plus-tip ceiling) is kept only when other views reach within `END_TOLERANCE_CM` of
    both its ends. A mask that covers the tip in two views passes (the tipseg lengths came
    from exactly such frames); a lone view whose rays meet the line at a grazing angle and
    run off (test B's head camera) does not, and is dropped as merged as before."""
    lo, hi = per_view[view]
    others = [interval for v, interval in per_view.items() if v != view]
    return any(abs(o_lo - lo) <= END_TOLERANCE_CM for o_lo, _ in others) and any(
        abs(o_hi - hi) <= END_TOLERANCE_CM for _, o_hi in others
    )


def _odd_interval(per_view: dict[str, tuple[float, float]]) -> str:
    """The view whose interval centre is farthest from the median centre."""
    centres = {v: 0.5 * (lo + hi) for v, (lo, hi) in per_view.items()}
    median = float(np.median(list(centres.values())))
    return max(centres, key=lambda v: abs(centres[v] - median))


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
    direction_uncertainty_deg: float = 0.0,
    ceiling_length_cm: float | None = None,
) -> tuple[LineFit | None, tuple[str, ...]]:
    """The line fit over a set of members (one per view), with the merged-mask rejections,
    the residual prune (a member over its gate is dropped while three or more remain) and,
    against a `predicted` line, the line gate (angle and perpendicular offset at the
    midpoint, inflated by `direction_uncertainty_deg` and `uncertainty_cm` as the pixel gate
    is; the member whose view disagrees most with the prediction is dropped and the fit
    repeated). Sep 29: the axis views' intervals along the line must overlap within
    `line_gate_cm`, as `_line_pair_cost` asks of a birth pair (with three or more the odd
    interval's view is dropped, with two there is no fit); a view looking along the line
    gives no interval; a compact view whose ray point lies off the shaft is dropped. Returns
    ``(fit, merged views)``; the fit is None when no line can be fitted from the members
    that survive (fewer than two, no plane, degenerate, a two-member fit over its gate), the
    merged views are reported either way. Sep 29 v4: the merged-extent gates read
    `ceiling_length_cm` (the body plus its tip) when given, the completion `prior_length_cm`."""
    members = dict(members)
    merged: list[str] = []
    ceiling = (
        prior_length_cm if ceiling_length_cm is None else max(ceiling_length_cm, prior_length_cm)
    )
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
        extent_pairs = [
            (cam, obs) for cam, obs in cams_and_obs if _sees_broadside(cam, line, params)
        ]
        extent = line_endpoints(line, extent_pairs) if extent_pairs else None
        if extent is not None:
            too_long = [
                v
                for v, (lo, hi) in extent.per_view.items()
                if hi - lo > params.line_merged_extent_factor * prior_length_cm
                and not (
                    hi - lo <= params.line_merged_extent_factor * ceiling
                    and _ends_corroborated(v, extent.per_view)
                )
            ]
            if too_long:
                for v in too_long:
                    merged.append(v)
                    del members[v]
                continue
            if len(extent.per_view) >= 2:
                # The axis views must see one shaft: their intervals overlap within the gate.
                gaps = {
                    v: max(
                        _interval_gap(extent.per_view[v], extent.per_view[w])
                        for w in extent.per_view
                        if w != v
                    )
                    for v in extent.per_view
                }
                if max(gaps.values()) > params.line_gate_cm:
                    if len(extent.per_view) < 3:
                        return None, tuple(sorted(merged))
                    del members[_odd_interval(extent.per_view)]
                    continue
            if extent.visible_length_cm > params.line_merged_extent_factor * ceiling:
                # The views' extents together span more than one pipette: two objects, or one
                # view on the wrong one. With three or more, the odd interval goes; with two,
                # no fit.
                if len(extent.per_view) < 3:
                    return None, tuple(sorted(merged))
                del members[_odd_interval(extent.per_view)]
                continue
        if predicted is not None:
            angle, perpendicular, _ = line_distance(predicted, line)
            if (
                angle > params.line_gate_deg + direction_uncertainty_deg
                or perpendicular > params.line_gate_cm + uncertainty_cm
            ):
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
            line,
            cams_and_obs,
            params,
            prior_length_cm,
            prior_spread_cm,
            tip_hint,
            fallback_extent=None if predicted is None else predicted.endpoints,
            ceiling_length_cm=ceiling,
        )
        if completed is None:
            return None, tuple(sorted(merged))
        if completed.far_rays:
            # A compact view whose ray point lies off the shaft sees another object.
            for v in completed.far_rays:
                del members[v]
            continue
        fit = LineFit(
            line=line,
            endpoints=completed.endpoints,
            members=members,
            residuals=residuals,
            merged_views=tuple(sorted(merged)),
            visible_length_cm=completed.visible_length_cm,
            extended=completed.extended,
            plane_views=line.plane_views,
            clamped=completed.clamped,
            extent_from_prediction=completed.from_prediction,
        )
        return fit, fit.merged_views
    return None, tuple(sorted(merged))


def may_carry_tip(prior: LinePrior, cls: str | None) -> bool:
    """Sep 29 v4: whether a class's masks may honestly run past its bare body, that is,
    whether the prior holds a with-tip length mode for it (an undecided class may)."""
    return cls is None or prior.modes_for(cls)[1] is not None


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
    tips: dict[str, list[Obs]] | None = None,
) -> list[Candidate]:
    """Birth candidates of the geometric class: pairwise plausibility (`_line_pair_cost`)
    across every view with a pose, Hungarian per view pair, cliques as the point birth forms
    them, then `fit_line_members` per clique with the class prior when the members' colour
    vote is decisive; a clique whose fit is degenerate falls back to the point birth's
    triangulation of its centroids. The birth rule (fixed views, or fixed views plus the fpv)
    is the core's. Sep 29: `tips` (per view, the `*_tip` detector boxes) are attached to the
    elongated observations first, either side, so a birth sees the tip's extent and the prior
    it expects is the two-state one (bare plus tip when a member carries a tip). Sep 29 v4
    (the length rule on): a mask may cover a tip on any frame, so the pair gate and the
    merged-extent gates allow the body plus its tip whether or not a box attached; the
    completion still expects the bare body unless a box says otherwise."""
    cls = params.line_geometric_class
    per_view = {
        v: [o for o in unassigned.get(v, ()) if o.object_class == cls] for v in sorted(cams)
    }
    per_view = {v: items for v, items in per_view.items() if items}
    if tips and params.tip_boxes_on:
        for v, items in per_view.items():
            attached = attach_tips_in_view(items, tips.get(v, ()), params.gates.association_px)
            per_view[v] = [attached.get(i, o) for i, o in enumerate(items)]
    tip_slack = max([prior.tip_for(tip) for tip, _ in params.line_tip_classes] or [0.0])
    slack_always = params.line_tip_length_rule and params.line_birth_tip_slack and prior.two_state
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
                pair_may = slack_always and (
                    may_carry_tip(prior, a.colour_class) or may_carry_tip(prior, b.colour_class)
                )
                pair_prior = prior.length_cm + (
                    tip_slack if (pair_may or a.tip_attached or b.tip_attached) else 0.0
                )
                cost[i, j] = _line_pair_cost(cams[u], a, cams[v], b, params, pair_prior)
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
        tip_classes = Counter(o.tip_class for o in members.values() if o.tip_class)
        tip_class = tip_classes.most_common(1)[0][0] if tip_classes else None
        birth_class = decisive_class(votes)
        fit, _merged = fit_line_members(
            members,
            cams,
            params,
            prior.expected_length(birth_class, tip_class, attached=bool(tip_classes)),
            prior.spread_cm,
            ceiling_length_cm=(
                prior.expected_length(birth_class, tip_class, attached=True)
                if slack_always and may_carry_tip(prior, birth_class)
                else None
            ),
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
                clamped=fit.clamped,
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
        # -- Sep 29 defect corrections (all zero with the extension off)
        self.line_extent_clamped = 0
        self.line_step_capped = 0
        self.line_single_view_vetoed = 0
        self.line_aided_rejected = 0
        self.line_aided_from_prediction = 0
        self.line_far_rays_dropped = 0
        self.line_held_refused = 0
        self.line_held_offset_remeasured = 0
        self.line_class_vetoes = 0
        self.line_class_splits = 0
        # -- Sep 29 disposable tips (all zero / empty with the extension off)
        self.line_tip_line_frames = 0
        self.line_tip_attached_frames = 0
        self.line_tip_attached_by_view: dict[str, int] = defaultdict(int)
        self.line_tip_attached_by_mode: dict[str, int] = defaultdict(int)
        self.line_tip_attached_by_class: dict[str, int] = defaultdict(int)
        self.line_tip_state_on = 0
        self.line_tip_state_off = 0
        self.line_tip_votes = 0
        self.line_tip_side_refused = 0
        self.line_width_votes_cast = 0
        self.line_tip_basis_frames: dict[str, int] = defaultdict(int)
        # -- Sep 29 v4: the tail basis and the 3D-length tip state
        self.line_tail_votes_cast = 0
        self.line_tail_votes_skipped_class = 0
        self.line_tip_length_undecided = 0
        self.line_tip_length_truncated = 0
        self.line_tip_length_no_mode = 0
        self.line_tip_box_corroboration: dict[str, int] = defaultdict(int)

    # -- helpers

    def _is_geometric(self, t: Track) -> bool:
        return bool(self.params.line_classes) and t.object_class == self.params.line_geometric_class

    def _prior_length(self, t: Track) -> float:
        """The length the track expects this frame: the class's bare body, plus its tip when
        the track's `tip_attached` state is on (v4: the state alone; under the v3 box rule
        also when a view attached a tip box this frame)."""
        assert self.line_prior is not None
        cls = decisive_class(t.class_votes)
        attached = bool(t.tip_attached)
        if not self.params.line_tip_length_rule:
            attached = attached or bool(t.tip_views_this_frame)
        return self.line_prior.expected_length(cls, t.tip_class, attached=attached)

    def _ceiling_length(self, t: Track) -> float:
        """Sep 29 v4: the longest a single mask of this track may honestly be, the body plus
        its tip. The merged-extent gates and the clamp read it (a mask that covers a tip is
        not two masks) whatever the state says; the completion reads `_prior_length`."""
        assert self.line_prior is not None
        prior = self._prior_length(t)
        if not self.params.line_tip_length_rule:
            return prior
        cls = decisive_class(t.class_votes)
        if not may_carry_tip(self.line_prior, cls):
            # A bare-only class (no with-tip mode): nothing longer than its body is honest,
            # and the 8-channel's merged masks must still be dropped.
            return prior
        return max(prior, self.line_prior.expected_length(cls, t.tip_class, attached=True))

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
            line_candidates = line_birth_candidates(
                lines, cams, params, self.line_prior, tips=obs.tips
            )
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
                self._class_veto(t, items, cost[i])
            for i, j in hungarian(cost):
                if cost[i, j] < HUNGARIAN_FORBIDDEN:
                    assigned[(tracks[i].track_id, v)] = items[j]
                    used_obs.add((v, items[j].index))
        return assigned

    def _decisive_class(self, t: Track) -> str | None:
        """The class veto's plurality: the top class when it holds at least
        `line_class_veto_share` of the votes cast over at least `line_class_veto_frames`
        frames; None before that (the colour never bars an association on a young track)."""
        params = self.params
        if t.class_vote_frames < params.line_class_veto_frames or not t.class_votes:
            return None
        cls, top = max(sorted(t.class_votes.items()), key=lambda kv: kv[1])
        total = sum(t.class_votes.values())
        return cls if top >= params.line_class_veto_share * total else None

    def _class_veto(self, t: Track, items: Sequence[Obs], costs: np.ndarray) -> None:
        """Sep 29: an observation whose class differs from the track's decisive plurality is
        refused while a same-class observation is within reach in that view (T2's blue slot
        joined the red pipette's track on 6 of 7 red disagreements; the geometric class
        alone permitted it). Without a same-class candidate the colour still never bars an
        association: the detector's class is least reliable while pipetting."""
        decisive = self._decisive_class(t)
        if decisive is None:
            return
        same = [
            j
            for j, o in enumerate(items)
            if o.colour_class == decisive and costs[j] < HUNGARIAN_FORBIDDEN
        ]
        if not same:
            return
        vetoed = 0
        for j, o in enumerate(items):
            if (
                o.colour_class is not None
                and o.colour_class != decisive
                and costs[j] < HUNGARIAN_FORBIDDEN
            ):
                costs[j] = HUNGARIAN_FORBIDDEN
                vetoed += 1
        self.line_class_vetoes += vetoed

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
        votes: Counter = Counter()
        for v, o in mine.items():
            if o.colour_class is not None:
                t.class_votes[o.colour_class] += 1
                votes[o.colour_class] += 1
            # Sep 29: an attached tip is a second, larger colour patch; its class casts one
            # vote for its pipette class, as a detector class does.
            if o.tip_attached and o.tip_class is not None:
                pipette_class = self.params.pipette_class_of_tip(o.tip_class)
                if pipette_class is not None:
                    t.class_votes[pipette_class] += 1
                    votes[pipette_class] += 1
                    self.line_tip_votes += 1
            if o.width_px is not None and o.axis_px is not None and v in cams:
                history = t.width_history.setdefault(v, [])
                history.append(float(o.width_px) / pixels_per_cm(cams[v], t.position))
                del history[:-60]
        if votes:
            t.class_vote_frames += 1
            t.recent_class_votes.append(votes)
            del t.recent_class_votes[: -self.params.line_class_split_window]

    def _class_flip(self, t: Track) -> str | None:
        """The second colour that took at least `line_class_split_share` of the votes over the
        last `line_class_split_window` voting frames of a track with a decisive plurality of
        another class; None otherwise (or while the window is not full)."""
        params = self.params
        decisive = self._decisive_class(t)
        if decisive is None or len(t.recent_class_votes) < params.line_class_split_window:
            return None
        recent: Counter = sum(t.recent_class_votes, Counter())
        total = sum(recent.values())
        for cls, n in sorted(recent.items(), key=lambda kv: (-kv[1], kv[0])):
            if cls != decisive and n >= params.line_class_split_share * total:
                return cls
        return None

    def _class_split(self, t: Track, flipped_to: str, frame: int) -> Track:
        """Sep 29: the track's recent votes belong to another pipette. The geometry continues
        under a new id whose votes are the recent window's; the old id ends here (`lost`
        with `split_to`), its plurality intact."""
        recent: Counter = sum(t.recent_class_votes, Counter())
        new = Track(
            track_id=self._new_id(t.object_class),
            object_class=t.object_class,
            position=t.position.copy(),
            uncertainty_cm=t.uncertainty_cm,
            state=t.state,
            born_frame=frame,
            last_observed_frame=t.last_observed_frame,
            support_views=t.support_views,
            residuals=dict(t.residuals),
            per_view_last_seen=dict(t.per_view_last_seen),
            per_view_extent=dict(t.per_view_extent),
            per_view_slot=dict(t.per_view_slot),
            support_slots=dict(t.support_slots),
            velocity=t.velocity.copy(),
            mover=t.mover,
            mover_frames=t.mover_frames,
            last_observed_position=(
                None if t.last_observed_position is None else t.last_observed_position.copy()
            ),
            direction=None if t.direction is None else t.direction.copy(),
            endpoints_cm=None if t.endpoints_cm is None else t.endpoints_cm.copy(),
            endpoint_velocity=(None if t.endpoint_velocity is None else t.endpoint_velocity.copy()),
            direction_uncertainty_deg=t.direction_uncertainty_deg,
            class_votes=Counter(recent),
            tip_is_endpoint_0=t.tip_is_endpoint_0,
            last_observed_endpoints=(
                None if t.last_observed_endpoints is None else t.last_observed_endpoints.copy()
            ),
            last_endpoints_frame=t.last_endpoints_frame,
            width_history={v: list(h) for v, h in t.width_history.items()},
            line_residuals=dict(t.line_residuals),
            merged_views=t.merged_views,
            line_this_frame=t.line_this_frame,
            line_update=t.line_update,
            extent_clamped=t.extent_clamped,
            class_vote_frames=len(t.recent_class_votes),
            recent_class_votes=[Counter(c) for c in t.recent_class_votes],
            butt_offset=None if t.butt_offset is None else t.butt_offset.copy(),
            butt_offset_hand=t.butt_offset_hand,
            butt_offset_frame=t.butt_offset_frame,
            tip_history=list(t.tip_history),
            tip_class_history=list(t.tip_class_history),
            tip_ever=t.tip_ever,
            tip_attached=t.tip_attached,
            tip_class=t.tip_class,
            tip_views_this_frame=t.tip_views_this_frame,
            tip_basis=t.tip_basis,
            width_votes=list(t.width_votes),
            tail_votes=list(t.tail_votes),
            length_history=list(t.length_history),
            tip_rule_length_cm=t.tip_rule_length_cm,
        )
        if t.mover:
            self.mover_track_ids.add(new.track_id)
        self.tracks[new.track_id] = new
        self.class_births[new.object_class] += 1
        self.line_class_splits += 1
        plurality_from = t.observed_class
        t.state = "lost"
        t.lost_frame = frame
        self._event(
            frame,
            new.track_id,
            "class_split",
            split_from=t.track_id,
            plurality_from=plurality_from,
            plurality_to=flipped_to,
            recent_votes=dict(sorted(recent.items())),
            views=sorted(t.support_views),
        )
        self._event(
            frame,
            t.track_id,
            "lost",
            frames_unobserved=0,
            last_observed_frame=t.last_observed_frame,
            split_to=new.track_id,
        )
        return new

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
        t.line_update = None
        t.extent_clamped = False
        t.tip_views_this_frame = ()
        if obs is not None and params.tip_boxes_on and mine:
            mine = self._attach_tips(t, mine, cams, obs)
            t.tip_views_this_frame = tuple(sorted(v for v, o in mine.items() if o.tip_attached))
        # The prediction this frame's update moves away from: the step cap measures against it.
        predicted = t.position.copy()
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
                direction_uncertainty_deg=t.direction_uncertainty_deg,
                ceiling_length_cm=self._ceiling_length(t),
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
                self._apply_line_fit(t, fit, cams, frame, predicted=predicted, dt=dt)
                mine = fit.members
            else:
                self.line_degenerate_fits += 1
                point, kept, residuals = _prune_clique(remaining, cams, params)
                if len(kept) >= 2:
                    self._apply_point_fit(
                        t, point, kept, residuals, cams, frame, predicted=predicted, dt=dt
                    )
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
            t.line_update = "predicted"
            if params.motion_model and t.mover:
                if t.is_line:
                    if self._lateral_line_update(t, cams[v], o):
                        t.line_update = "lateral"
                        self.single_view_ray_updates += 1
                    else:
                        self.line_single_view_vetoed += 1
                else:
                    # A geometric-class track still tracked as a point (born from a
                    # degenerate fit): the core's lateral update along the ray.
                    t.position = ray_point(cams[v], o.point, t.position)
                    t.line_update = "lateral"
                    self.single_view_ray_updates += 1
                if t.line_update == "lateral" and self._cap_step(t, predicted, dt):
                    t.line_update = "capped"
            t.state = "single_view"
            t.frames_unobserved = 0
            t.last_observed_frame = frame
            self.line_single_view_frames += 1
        elif not mine:
            self._start_coasting(t, frame, dt, cams, obs)
            return
        t.support_views = tuple(sorted(mine))
        t.support_slots = {v: o.slot for v, o in mine.items()}
        # A tip attached in a view the update dropped is not a tip on this track this frame.
        t.tip_views_this_frame = tuple(sorted(v for v, o in mine.items() if o.tip_attached))
        for v, o in mine.items():
            self._record_support(t, v, o, cams[v], frame)
        self._record_line_support(t, mine, cams)
        if t.is_line:
            self._update_tip_state(t, mine)
        if t.is_line and obs is not None:
            self._resolve_tip(t, cams, obs, frame, mine)
        flipped = self._class_flip(t)
        if flipped is not None:
            self._class_split(t, flipped, frame)

    # -- disposable tips (Sep 29)

    def _tip_side_in_view(self, t: Track, cam: Camera, axis_px: np.ndarray) -> int | None:
        """The index of the axis end nearer the track's projected tip, when the track's tip
        was decided by a tip box and the state is on (the tip side is then known); None
        otherwise (a hand's guess may be overruled by the box, either side)."""
        if t.tip_basis != TIP_BASIS or not t.tip_attached:
            return None
        ends = t.tip_and_butt()
        if ends is None:
            return None
        tip_pixel = project(cam, ends[0])
        if tip_pixel is None:
            return None
        distances = np.linalg.norm(np.asarray(axis_px, dtype=np.float64) - tip_pixel, axis=1)
        return int(np.argmin(distances))

    def _projected_body_axis(self, t: Track, cam: Camera) -> tuple[np.ndarray, int | None] | None:
        """The track's bare body projected into a view as a 2D axis ``(2x2 px, tip end
        index or None)``: the tip end pulled back by the tip length when the track's state
        already includes the tip, so a tip box beyond the body end is beyond the body."""
        assert t.endpoints_cm is not None and t.direction is not None and self.line_prior
        ends = t.endpoints_cm.copy()
        tip_index: int | None = None
        if t.tip_is_endpoint_0 is not None:
            tip_index = 0 if t.tip_is_endpoint_0 else 1
            # The written segment already includes the tip once a frame attached one (the
            # state may still be undecided): the body is the bare length from the butt.
            bare = self.line_prior.bare_for(decisive_class(t.class_votes))
            span = ends[tip_index] - ends[1 - tip_index]
            norm = float(np.linalg.norm(span))
            if norm > bare + 1e-6:
                ends[tip_index] = ends[1 - tip_index] + span / norm * bare
        pixels = [project(cam, e) for e in ends]
        if any(p is None for p in pixels):
            return None
        return np.stack(pixels), tip_index  # type: ignore[arg-type]

    def _attach_tips(
        self, t: Track, mine: dict[str, Obs], cams: dict[str, Camera], obs: FrameObservations
    ) -> dict[str, Obs]:
        """Attach the frame's `*_tip` detector boxes to this track's observations, per view:
        on the observation's own axis when it has one (`axis` mode, the extent then reaches
        the tip), else on the track's projected body when the track is a line (`track` mode,
        state and vote only). A box on the butt side is refused once the tip side is known
        from an earlier box."""
        gate = self.params.gates.association_px
        out = dict(mine)
        for v, o in mine.items():
            tips = obs.tips.get(v)
            if not tips or v not in cams:
                continue
            cam = cams[v]
            if o.axis_px is not None and is_elongated(o.axis_obs()):
                axis = np.asarray(o.axis_px, dtype=np.float64)
                side = self._tip_side_in_view(t, cam, axis) if t.is_line else None
                attached = attach_tips_in_view([o], tips, gate, side_of=lambda _o: side)
                if 0 in attached:
                    out[v] = attached[0]
                elif side is not None and attach_tips_in_view([o], tips, gate):
                    self.line_tip_side_refused += 1
                continue
            if not t.is_line:
                continue
            projected = self._projected_body_axis(t, cam)
            if projected is None:
                continue
            axis, tip_index = projected
            side = tip_index if (t.tip_basis == TIP_BASIS and t.tip_attached) else None
            best: tuple[float, Obs, Any] | None = None
            for tip in tips:
                assert tip.box is not None
                attachment = attach_tip_box(axis, tip.box, gate, side=side)
                if attachment is None:
                    if side is not None and attach_tip_box(axis, tip.box, gate) is not None:
                        self.line_tip_side_refused += 1
                    continue
                if best is None or attachment.across_px < best[0]:
                    best = (attachment.across_px, tip, attachment)
            if best is not None:
                out[v] = attach_tip(o, best[1], best[2], "track")
        return out

    def _update_tip_state(self, t: Track, mine: dict[str, Obs]) -> None:
        """The two-state flag. Sep 29 v4 (`line_tip_length_rule`): the median visible
        butt-to-tip length over the track's last measured line fits against the class's two
        length modes (`_tip_state_from_length`), the attached boxes recorded as
        corroboration. The v3 rule otherwise: on after `line_tip_on_frames` of the last
        `line_tip_on_window` line frames carried an attached tip in some view, off after
        `line_tip_off_window` frames without one; None before either window decides. The tip
        class follows the plurality of the recent attached classes, else the pipette's own
        tip class while the state is on."""
        params = self.params
        attached = [o for o in mine.values() if o.tip_attached]
        self.line_tip_line_frames += 1
        if attached:
            self.line_tip_attached_frames += 1
            for v, o in mine.items():
                if o.tip_attached:
                    self.line_tip_attached_by_view[v] += 1
                    self.line_tip_attached_by_mode[o.tip_mode or "axis"] += 1
                    self.line_tip_attached_by_class[o.tip_class or "?"] += 1
        classes = Counter(o.tip_class for o in attached if o.tip_class)
        t.tip_ever = t.tip_ever or bool(attached)
        t.tip_history.append(bool(attached))
        t.tip_class_history.append(classes.most_common(1)[0][0] if classes else None)
        keep = max(params.line_tip_on_window, params.line_tip_off_window)
        del t.tip_history[:-keep]
        del t.tip_class_history[:-keep]
        if params.line_tip_length_rule:
            self._tip_state_from_length(t)
            state = {True: "on", False: "off", None: "undecided"}[t.tip_attached]
            self.line_tip_box_corroboration[f"{state}_{'with' if attached else 'without'}_box"] += 1
        else:
            recent = t.tip_history[-params.line_tip_on_window :]
            if sum(recent) >= params.line_tip_on_frames:
                if t.tip_attached is not True:
                    self.line_tip_state_on += 1
                t.tip_attached = True
            elif len(t.tip_history) >= params.line_tip_off_window and not any(
                t.tip_history[-params.line_tip_off_window :]
            ):
                if t.tip_attached is True:
                    self.line_tip_state_off += 1
                t.tip_attached = False
        recent_classes = Counter(
            c for c in t.tip_class_history[-params.line_tip_on_window :] if c is not None
        )
        if recent_classes:
            t.tip_class = recent_classes.most_common(1)[0][0]
        elif t.tip_attached is True and params.line_tip_length_rule:
            assert self.line_prior is not None
            t.tip_class = t.tip_class or self.line_prior.tip_class_for(
                decisive_class(t.class_votes)
            )
        elif t.tip_attached is False:
            t.tip_class = None

    def _tip_state_from_length(self, t: Track) -> None:
        """Sep 29 v4, the tipseg finding as the tracker's rule: SAM3's body mask covers the
        tip when one is on, so the pipette's butt-to-tip length has two modes (the blue
        pipette: 22.3 cm bare, 29.5 with a tip) and the per-view boxes are not needed. The
        median of the last `line_tip_length_window` measured visible lengths (at least
        `line_tip_length_min_frames`) is read against the class's modes: over the midpoint
        by the hysteresis band the state is on, under it by the band it is off, in the band
        it keeps; a median more than `line_tip_length_truncated_cm` under the bare mode is a
        mask the hand cut short and decides nothing; a class with no with-tip mode is bare
        when the median sits within that margin of its bare mode and undecided otherwise."""
        params = self.params
        assert self.line_prior is not None
        recent = t.length_history[-params.line_tip_length_window :]
        if len(recent) < params.line_tip_length_min_frames:
            t.tip_rule_length_cm = None
            self.line_tip_length_undecided += 1
            return
        median = float(np.median(recent))
        t.tip_rule_length_cm = round(median, 2)
        bare, with_tip = self.line_prior.modes_for(decisive_class(t.class_votes))
        decision: bool | None = None
        if median < bare - params.line_tip_length_truncated_cm:
            self.line_tip_length_truncated += 1
        elif with_tip is None:
            if median <= bare + params.line_tip_length_truncated_cm:
                decision = False
            else:
                self.line_tip_length_no_mode += 1
        else:
            midpoint = 0.5 * (bare + with_tip)
            band = params.line_tip_length_hysteresis_cm
            if median >= midpoint + band:
                decision = True
            elif median <= midpoint - band:
                decision = False
        if decision is None or decision == t.tip_attached:
            return
        if decision:
            self.line_tip_state_on += 1
        elif t.tip_attached is True:
            self.line_tip_state_off += 1
        t.tip_attached = decision

    def _resolve_tip_from_boxes(
        self, t: Track, cams: dict[str, Camera], mine: dict[str, Obs]
    ) -> bool:
        """The attached tip marks the tip end: every view with an attached box votes for the
        track end whose projection lies nearer the box centre; the majority decides. Returns
        whether a decision was made."""
        assert t.endpoints_cm is not None
        votes = [0, 0]
        for v, o in mine.items():
            if not o.tip_attached or o.tip_pixel is None or v not in cams:
                continue
            pixels = [project(cams[v], e) for e in t.endpoints_cm]
            if any(p is None for p in pixels):
                continue
            distances = [float(np.linalg.norm(p - o.tip_pixel)) for p in pixels]  # type: ignore[operator]
            votes[int(np.argmin(distances))] += 1
        if votes[0] == votes[1]:
            return False
        self._set_tip(t, tip_is_endpoint_0=votes[0] > votes[1], basis=TIP_BASIS)
        return True

    def _cap_step(self, t: Track, predicted: np.ndarray, dt: int) -> bool:
        """Sep 29: a midpoint step over `line_max_step_cm` a frame is not a pipette moving, it
        is a bad update; the step is cut to the cap along its own direction and the caller
        demotes the row to prediction-aided (no uncertainty shrink, no velocity fed)."""
        limit = self.params.line_max_step_cm * dt
        step = t.position - predicted
        norm = float(np.linalg.norm(step))
        if norm <= limit:
            return False
        pull = step * (limit / norm - 1.0)
        if t.is_line:
            assert t.endpoints_cm is not None
            t.set_endpoints(t.endpoints_cm + pull)
        else:
            t.position = t.position + pull
        self.line_step_capped += 1
        return True

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
        *,
        predicted: np.ndarray | None = None,
        dt: int = 1,
    ) -> None:
        """The core's >= 2-view position update; a line track's endpoints move with it. With
        `predicted` the step from it is capped (`_cap_step`) and the row demoted."""
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
        capped = predicted is not None and self._cap_step(t, predicted, dt)
        t.line_update = "capped" if capped else "point"
        if not capped:
            t.uncertainty_cm = float(np.sqrt((1 - gain) * prior_var))
        t.state = "observed"
        if params.motion_model and not capped:
            self._update_velocity(t, frame)
        t.last_observed_frame = frame
        t.frames_unobserved = 0
        t.residuals = {v: round(residuals[v], 2) for v in kept}

    def _apply_line_fit(
        self,
        t: Track,
        fit: LineFit,
        cams: dict[str, Camera],
        frame: int,
        *,
        predicted: np.ndarray | None = None,
        dt: int = 1,
    ) -> None:
        """Blend the fitted line into the track: the midpoint and the length by the scalar
        Kalman gain the core uses for a position, the direction by its own gain (its process
        noise is `LINE_DIRECTION_NOISE_DEG` per frame, a measurement is worth
        `LINE_DIRECTION_MEAS_DEG`, so it follows the measurement while pipetting); a point
        track of the geometric class is promoted to a line track. With `predicted` the
        midpoint step from it is capped (`_cap_step`) and the row demoted to
        prediction-aided: the uncertainty is not shrunk and no velocity is fed."""
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
        capped = predicted is not None and self._cap_step(t, predicted, dt)
        t.line_update = "capped" if capped else ("aided" if fit.aided else "fit")
        t.extent_clamped = fit.clamped
        if fit.clamped:
            self.line_extent_clamped += 1
        if not capped:
            t.uncertainty_cm = float(np.sqrt((1 - gain) * prior_var))
        t.state = "observed"
        if params.motion_model and not capped:
            self._update_velocity(t, frame, endpoints=not fit.aided)
        t.last_observed_frame = frame
        t.frames_unobserved = 0
        t.residuals = {v: round(fit.residuals[v], 2) for v in kept}
        t.line_residuals = dict(t.residuals)
        t.line_this_frame = True
        self.line_frames += 1
        self._record_length(t, fit)
        self._record_line_metrics(t, fit, cams, self._prior_length(t))

    def _record_length(self, t: Track, fit: LineFit) -> None:
        """Sep 29 v4: the visible butt-to-tip length of a measured fit joins the track's
        length history (the 3D-length tip state reads its median). A prediction-aided fit
        (its direction is the prediction's), an extent the prediction supplied and a clamped
        one are not measurements of the pipette."""
        if fit.aided or fit.extent_from_prediction or fit.clamped or fit.visible_length_cm <= 0:
            return
        t.length_history.append(float(fit.visible_length_cm))
        del t.length_history[: -self.params.line_tip_length_window]

    def _record_line_metrics(
        self, t: Track, fit: LineFit, cams: dict[str, Camera], prior_length: float
    ) -> None:
        if not fit.extent_from_prediction:
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

    def _lateral_line_update(self, t: Track, cam: Camera, o: Obs) -> bool:
        """A moving line track seen in one view: an elongated mask moves each endpoint onto
        the plane through the camera and the observed axis (the perpendicular correction, the
        length kept, the depth along the view kept); a compact one moves the midpoint along
        the centroid's ray as the core does for a point. Sep 29: no update (False) when the
        view looks along the predicted line within `line_min_view_angle_deg` (the plane then
        constrains nothing along the shaft and the projection slides the ends away) or the
        predicted direction is within that angle of the plane's normal (the projection
        collapses the segment and its direction is noise); the frame keeps the prediction."""
        assert t.endpoints_cm is not None and t.direction is not None
        if not _sees_broadside(cam, t.line(), self.params):
            return False
        axis = o.axis_obs()
        if is_elongated(axis):
            assert axis.endpoints_px is not None
            try:
                normal, offset = plane_from_axis(cam, axis.endpoints_px)
            except ValueError:
                return False
            if abs(float(normal @ t.direction)) > math.cos(
                math.radians(self.params.line_min_view_angle_deg)
            ):
                return False
            feet = t.endpoints_cm - ((t.endpoints_cm @ normal) - offset)[:, None] * normal
            length = float(np.linalg.norm(t.endpoints_cm[1] - t.endpoints_cm[0]))
            span = feet[1] - feet[0]
            norm = float(np.linalg.norm(span))
            if norm < 1e-9:
                return False
            midpoint = feet.mean(axis=0)
            ends = np.stack(
                [midpoint - span / norm * length / 2, midpoint + span / norm * length / 2]
            )
        else:
            ends = t.endpoints_cm + (ray_point(cam, o.point, t.position) - t.position)
        t.set_endpoints(ends)
        return True

    def _aided_line_fit(
        self, t: Track, members: dict[str, Obs], cams: dict[str, Camera]
    ) -> LineFit | None:
        """The fit `fit_line` cannot make on its own: one plane (or planes meeting under the
        pair angle) plus rays to one spot of the shaft. The predicted direction projected
        onto the plane gives the direction; the line runs through the mean of the ray-plane
        intersections (the rays' points on the shaft), else through the foot of the predicted
        midpoint. Members over their view gate against that line are left out; the extent
        and the prior follow as for a full fit. None without a plane in front of its camera.

        Sep 29: every runaway extent of the first run began here. The line lies in the
        plane, so the plane view's residual is zero whatever the in-plane direction; the
        only check on that direction is the plane view's own extent on the aided line, which
        runs off when the predicted direction disagrees with the observed axis (found 70-80
        deg off on the worst rows). A plane view whose extent exceeds
        `line_merged_extent_factor` times the prior rejects the fit, and so does an aided
        line running within `line_min_view_angle_deg` of a plane view's ray (an elongated
        mask is a shaft seen broadside); a ray far along the line is dropped
        (`complete_extent`); fewer than two members left is no fit (a lone plane is the
        single-view update's case, bounded); the extent is clamped."""
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
        prior_length = self._prior_length(t)
        ceiling = self._ceiling_length(t)
        for v, o in kept.items():
            if not is_elongated(o.axis_obs()):
                continue
            # An elongated mask is a shaft seen broadside; an aided line running along that
            # view's ray contradicts it (the worst rows: 4-8 deg), as does an extent on the
            # aided line longer than a pipette. The extent check keeps the state's prior, not
            # the with-tip ceiling: the extent here is the sign of a wrong in-plane direction,
            # and loosening it by a tip length lets wrong directions through (test B).
            if not _sees_broadside(cams[v], line, params):
                self.line_aided_rejected += 1
                return None
            lo, hi = _along_line_interval(cams[v], o, line)
            if hi - lo > params.line_merged_extent_factor * prior_length:
                self.line_aided_rejected += 1
                return None
        resolved = t.tip_and_butt()
        completed = complete_extent(
            line,
            _cams_and_obs(kept, cams, params),
            params,
            prior_length,
            self.line_prior.spread_cm,
            None if resolved is None else resolved[0],
            fallback_extent=t.endpoints_cm,
            ceiling_length_cm=ceiling,
        )
        if completed is None:
            return None
        for v in completed.far_rays:
            del kept[v]
            del residuals[v]
            self.line_far_rays_dropped += 1
        if len(kept) < 2 or not any(is_elongated(o.axis_obs()) for o in kept.values()):
            return None
        if completed.from_prediction:
            self.line_aided_from_prediction += 1
        return LineFit(
            line=line,
            endpoints=completed.endpoints,
            members=kept,
            residuals=residuals,
            merged_views=(),
            visible_length_cm=completed.visible_length_cm,
            extended=completed.extended,
            plane_views=tuple(v for v, _, _ in planes if v in kept),
            aided=True,
            clamped=completed.clamped,
            extent_from_prediction=completed.from_prediction,
        )

    def _resolve_tip(
        self,
        t: Track,
        cams: dict[str, Camera],
        obs: FrameObservations,
        frame: int,
        mine: dict[str, Obs] | None = None,
    ) -> None:
        """Which end is the tip, by five bases in order of strength. Sep 29: an attached
        `*_tip` box marks the tip end (`_resolve_tip_from_boxes`), and while the track's tip
        state is on that decision stands. Then the hand: the end nearer a hand is the butt
        (the plunger button, the coloured part), the other end the tip: the nearest live hand
        track within `LINE_HAND_REACH_CM` of one end and `LINE_HAND_MARGIN_CM` nearer to it
        than to the other decides. Sep 29 v4: then the rows' long-thin-tail side
        (`_resolve_tip_from_tail`). Then, without a hand track, an end whose projection lies
        inside a hand box in >= `held_min_views` views while the other end's does not; then
        the widths. Otherwise the ordering is kept by direction continuity (the update
        orients the measured direction along the predicted one). A hand track within reach
        of the butt leaves the butt-to-hand offset it measured (`butt_offset`), the only
        offset a later hold may follow."""
        assert t.endpoints_cm is not None
        params = self.params
        # The box basis follows the hysteresis state, not one frame's box: a loose tip
        # passing a resting pipette's end must not name its tip (trial 1, frames 754, 3528).
        if mine and t.tip_attached is True:
            self._resolve_tip_from_boxes(t, cams, mine)
        if mine:
            self._accumulate_width_votes(t, cams, mine)
            if params.line_tail_basis:
                self._accumulate_tail_votes(t, cams, mine)
        box_holds = t.tip_attached is True and t.tip_basis == TIP_BASIS
        width_total = sum(t.width_votes)
        width_decisive = abs(width_total) >= params.line_width_vote_margin
        width_first = params.line_width_over_hand and width_decisive and not box_holds
        if width_first:
            self._set_tip(t, tip_is_endpoint_0=width_total > 0, basis=WIDTH_BASIS)
        tail_total = sum(t.tail_votes)
        tail_first = (
            params.line_tail_basis
            and params.line_tail_over_hand
            and abs(tail_total) >= params.line_tail_vote_margin
            and not box_holds
        )
        if tail_first:
            self._set_tip(t, tip_is_endpoint_0=tail_total > 0, basis=TAIL_BASIS)
        stands = box_holds or width_first or tail_first
        hands = [
            h
            for h in self.live_tracks()
            if h.object_class in params.hand_classes and h.state in LOCALISED_STATES
        ]
        for hand in sorted(
            hands, key=lambda h: float(np.linalg.norm(t.endpoints_cm - h.position, axis=1).min())
        ):
            d0, d1 = np.linalg.norm(t.endpoints_cm - hand.position, axis=1)
            if stands:
                resolved = t.tip_and_butt()
                assert resolved is not None
                butt = resolved[1]
                if float(np.linalg.norm(butt - hand.position)) <= LINE_HAND_REACH_CM:
                    t.butt_offset = butt - hand.position
                    t.butt_offset_hand = hand.track_id
                    t.butt_offset_frame = frame
                    return
                continue
            if min(d0, d1) <= LINE_HAND_REACH_CM and abs(d0 - d1) >= LINE_HAND_MARGIN_CM:
                self._set_tip(t, tip_is_endpoint_0=bool(d0 > d1), basis="hand_track")
                butt = t.endpoints_cm[1 if d0 > d1 else 0]
                t.butt_offset = butt - hand.position
                t.butt_offset_hand = hand.track_id
                t.butt_offset_frame = frame
                return
        if stands:
            return
        if params.line_tail_basis and self._resolve_tip_from_tail(t):
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
            return
        self._resolve_tip_from_width(t)

    def _accumulate_tail_votes(
        self, t: Track, cams: dict[str, Camera], mine: dict[str, Obs]
    ) -> None:
        """Sep 29 v4, the fifth basis: each view whose row carries a `tip_side` (the axis
        end with the longer thin tail) names that axis end; it is matched to the track
        endpoint whose projection lies nearer it (the other end must match the other
        endpoint) and cast as a vote for the tip end. The wide-tip classes (the 8-channel,
        whose manifold is its wide end) cast none: the tail rule is unvalidated on them and
        the class rule of the width basis stands. The frame's net vote joins the window."""
        assert t.endpoints_cm is not None
        params = self.params
        cls = decisive_class(t.class_votes)
        net = 0
        for v, o in mine.items():
            if o.tip_side is None or v not in cams:
                continue
            obs_cls = cls or o.colour_class
            if obs_cls in params.line_wide_tip_classes:
                self.line_tail_votes_skipped_class += 1
                continue
            axis = o.body_axis_px if o.body_axis_px is not None else o.axis_px
            if axis is None:
                continue
            ends = np.asarray(axis, dtype=np.float64)
            pixels = [project(cams[v], e) for e in t.endpoints_cm]
            if any(p is None for p in pixels):
                continue
            tip_track_end = int(
                np.argmin([np.linalg.norm(p - ends[o.tip_side]) for p in pixels])  # type: ignore[operator]
            )
            butt_track_end = int(
                np.argmin([np.linalg.norm(p - ends[1 - o.tip_side]) for p in pixels])  # type: ignore[operator]
            )
            if tip_track_end == butt_track_end:
                continue
            net += 1 if tip_track_end == 0 else -1
            self.line_tail_votes_cast += 1
        t.tail_votes.append(net)
        del t.tail_votes[: -params.line_tail_vote_window]

    def _resolve_tip_from_tail(self, t: Track) -> bool:
        """The tail basis decides when the window's net vote reaches the margin and nothing
        stronger stands: a tip box while the state is on, or a hand track that named the
        ends, keeps its say (the brief's ranking); a track that is unresolved or was named by
        a hand box or the widths follows the tail. Returns whether the tail basis holds the
        decision (decided now, or already), so the weaker bases stand down."""
        if t.tip_is_endpoint_0 is not None:
            if t.tip_basis == TIP_BASIS and t.tip_attached is True:
                return False
            if t.tip_basis == "hand_track" and not self.params.line_tail_over_hand:
                return False
        total = sum(t.tail_votes)
        if abs(total) < self.params.line_tail_vote_margin:
            return t.tip_basis == TAIL_BASIS and t.tip_is_endpoint_0 is not None
        self._set_tip(t, tip_is_endpoint_0=total > 0, basis=TAIL_BASIS)
        return True

    def _accumulate_width_votes(
        self, t: Track, cams: dict[str, Camera], mine: dict[str, Obs]
    ) -> None:
        """Sep 29, the fourth basis: each view whose mask end widths differ by the ratio
        names its wide axis end; that end is matched to the track endpoint whose projection
        lies nearer it (the other end must match the other endpoint), and the class rule
        turns it into a vote for the tip end: the wide end for the classes in
        `line_wide_tip_classes` (the 8-channel manifold), the narrow end otherwise (the
        single-channel grip is the butt). The frame's net vote joins the window."""
        assert t.endpoints_cm is not None
        params = self.params
        cls = decisive_class(t.class_votes)
        net = 0
        for v, o in mine.items():
            if o.end_widths_px is None or v not in cams:
                continue
            w0, w1 = o.end_widths_px
            if min(w0, w1) <= 0 or max(w0, w1) / min(w0, w1) < params.line_width_ratio_min:
                continue
            axis = o.body_axis_px if o.body_axis_px is not None else o.axis_px
            if axis is None:
                continue
            ends = np.asarray(axis, dtype=np.float64)
            pixels = [project(cams[v], e) for e in t.endpoints_cm]
            if any(p is None for p in pixels):
                continue
            wide_axis_end = 0 if w0 > w1 else 1
            wide_track_end = int(
                np.argmin([np.linalg.norm(p - ends[wide_axis_end]) for p in pixels])  # type: ignore[operator]
            )
            narrow_track_end = int(
                np.argmin([np.linalg.norm(p - ends[1 - wide_axis_end]) for p in pixels])  # type: ignore[operator]
            )
            if wide_track_end == narrow_track_end:
                continue
            obs_cls = cls or o.colour_class
            if obs_cls is None:
                continue
            tip_end = (
                wide_track_end if obs_cls in params.line_wide_tip_classes else 1 - wide_track_end
            )
            net += 1 if tip_end == 0 else -1
            self.line_width_votes_cast += 1
        t.width_votes.append(net)
        del t.width_votes[: -params.line_width_vote_window]

    def _resolve_tip_from_width(self, t: Track) -> None:
        """The width basis decides when the window's net vote reaches the margin and nothing
        stronger stands: a tip box while the tip state is on, or a hand track that once named
        the ends (unless `line_width_over_hand`), keeps its say; an unresolved track, one named
        by a hand box alone, or one whose tip box decision lapsed with the state, follows the
        widths."""
        if t.tip_is_endpoint_0 is not None:
            if t.tip_basis == TIP_BASIS and t.tip_attached is True:
                return
            if t.tip_basis == "hand_track" and not self.params.line_width_over_hand:
                return
            if t.tip_basis == TAIL_BASIS:
                return
        total = sum(t.width_votes)
        if abs(total) < self.params.line_width_vote_margin:
            return
        self._set_tip(t, tip_is_endpoint_0=total > 0, basis=WIDTH_BASIS)

    def _set_tip(self, t: Track, *, tip_is_endpoint_0: bool, basis: str) -> None:
        if t.tip_is_endpoint_0 != tip_is_endpoint_0:
            self.line_tip_resolutions[basis] += 1
        t.tip_is_endpoint_0 = tip_is_endpoint_0
        t.tip_basis = basis

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
            line_offset: np.ndarray | None = None
            if hand is not None and hand[1] is not None and t.is_line:
                line_offset = self._held_offset(t, hand[1], frame)
                if line_offset is None:
                    # The hand was never within reach of the butt: not a hold, a coast.
                    self.line_held_refused += 1
                    hand = None
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
                    # The hand holds the butt: the offset measured on a localised frame with
                    # the hand within reach of it (`_held_offset`).
                    assert line_offset is not None
                    t.held_offset = line_offset
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

    def _held_offset(self, t: Track, hand_track: Track, frame: int) -> np.ndarray | None:
        """Sep 29: the offset a hold follows. The first run measured it from the coasted
        prediction at the moment support dropped, whatever the hand's distance (butt to hand
        12.7 / 19.9 cm at the median while held). Now: the butt-to-hand offset this hand
        measured on a localised frame within the coast timeout (`_resolve_tip`), else one
        measured now when the butt lies within `LINE_HAND_REACH_CM` of the hand; None when
        the hand was never near the butt, and the track coasts instead."""
        params = self.params
        if (
            t.butt_offset is not None
            and t.butt_offset_hand == hand_track.track_id
            and t.butt_offset_frame is not None
            and frame - t.butt_offset_frame <= params.coast_timeout_frames
        ):
            self.line_held_offset_remeasured += 1
            return t.butt_offset.copy()
        anchor = self._butt_anchor(t, hand_track.position, resolve=True)
        offset = anchor - hand_track.position
        if float(np.linalg.norm(offset)) <= LINE_HAND_REACH_CM:
            return offset
        return None

    def _hand_distances(self, t: Track) -> tuple[float, float] | None:
        """``(butt, tip)`` distances (cm) to the holding hand track while held, else to the
        nearest localised hand track when it is within twice `LINE_HAND_REACH_CM` of an end;
        None without a resolved tip or a hand. Written on the row so a wrong tip / butt
        assignment shows as a tip nearer the hand than the butt."""
        ends = t.tip_and_butt()
        if ends is None:
            return None
        tip, butt = ends
        hand: Track | None = None
        if t.state == "held" and t.held_by is not None:
            hand = self.tracks.get(t.held_by)
            if hand is not None and hand.state not in LOCALISED_STATES:
                hand = None
        if hand is None:
            hands = [
                h
                for h in self.live_tracks()
                if h.object_class in self.params.hand_classes and h.state in LOCALISED_STATES
            ]
            if not hands:
                return None
            hand = min(hands, key=lambda h: float(np.linalg.norm(butt - h.position)))
        d_butt = float(np.linalg.norm(butt - hand.position))
        d_tip = float(np.linalg.norm(tip - hand.position))
        if t.state != "held" and min(d_butt, d_tip) > 2.0 * LINE_HAND_REACH_CM:
            return None
        return d_butt, d_tip

    def _butt_anchor(
        self, t: Track, hand_position: np.ndarray, *, resolve: bool = False
    ) -> np.ndarray:
        """The butt end of a line track (with `resolve`, tip / butt are first decided from the
        hand's position when it is nearer one end by the margin); the midpoint when nothing
        is resolved."""
        assert t.endpoints_cm is not None
        box_holds = t.tip_basis == TIP_BASIS and bool(t.tip_attached)
        if resolve and not box_holds:
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
            t.line_update = "point"
            t.extent_clamped = False
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
            if t.direction is not None and float(cand.line.direction @ t.direction) < 0:
                # Direction continuity across the gap: the candidate's endpoint order is
                # arbitrary, the track's tip / butt flag refers to its own order.
                assert cand.endpoints is not None
                flipped_line = replace(
                    cand.line,
                    direction=-cand.line.direction,
                    endpoints=(
                        None if cand.line.endpoints is None else cand.line.endpoints[::-1].copy()
                    ),
                )
                cand = replace(cand, line=flipped_line, endpoints=cand.endpoints[::-1].copy())
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
            t.tip_views_this_frame = tuple(
                sorted(v for v, o in cand.members.items() if o.tip_attached)
            )
            self._record_line_support(t, cand.members, cams)
            if t.is_line:
                self._update_tip_state(t, cand.members)
                if t.tip_attached is True:
                    self._resolve_tip_from_boxes(t, cams, cand.members)

    def _apply_candidate_line(self, t: Track, cand: Candidate, cams: dict[str, Camera]) -> None:
        """A line candidate's geometry onto a (new or returning) track."""
        assert cand.line is not None and cand.endpoints is not None
        t.set_endpoints(cand.endpoints)
        t.direction_uncertainty_deg = LINE_DIRECTION_MEAS_DEG
        t.line_residuals = {v: round(r, 2) for v, r in cand.residuals.items()}
        t.merged_views = cand.merged_views
        t.line_this_frame = True
        t.line_update = "fit"
        t.extent_clamped = cand.clamped
        if cand.clamped:
            self.line_extent_clamped += 1
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
            clamped=cand.clamped,
        )
        self._record_length(t, fit)
        assert self.line_prior is not None
        tip_classes = Counter(o.tip_class for o in cand.members.values() if o.tip_class)
        prior_length = self.line_prior.expected_length(
            decisive_class(t.class_votes + votes),
            tip_classes.most_common(1)[0][0] if tip_classes else t.tip_class,
            attached=bool(tip_classes) or bool(t.tip_attached),
        )
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
            self.line_tip_basis_frames[
                (t.tip_basis or "unresolved") if resolved is not None else "unresolved"
            ] += 1
            if t.state in LOCALISED_STATES:
                fields["line_residual_px"] = dict(t.line_residuals) or None
                fields["merged_views"] = t.merged_views or None
                fields["line_update"] = t.line_update
                fields["extent_clamped"] = t.extent_clamped
                fields["tip_attached_views"] = t.tip_views_this_frame or None
            if self.params.tip_boxes_on:
                fields["tip_attached"] = t.tip_attached
                fields["tip_class"] = t.tip_class
                fields["tip_basis"] = t.tip_basis if resolved is not None else None
            if self.params.line_tip_length_rule:
                fields["tip_rule_length_cm"] = t.tip_rule_length_cm
            distances = self._hand_distances(t)
            if distances is not None:
                fields["butt_to_hand_cm"] = round(distances[0], 2)
                fields["tip_to_hand_cm"] = round(distances[1], 2)
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
                # Sep 29 corrections: how often each one acted.
                "extent_clamped": self.line_extent_clamped,
                "step_capped": self.line_step_capped,
                "single_view_vetoed": self.line_single_view_vetoed,
                "aided_rejected": self.line_aided_rejected,
                "aided_extent_from_prediction": self.line_aided_from_prediction,
                "far_rays_dropped": self.line_far_rays_dropped,
            },
            "held": {
                "refused_hand_far_from_butt": self.line_held_refused,
                "offset_from_a_localised_frame": self.line_held_offset_remeasured,
            },
            "class_veto": {
                "observations_refused": self.line_class_vetoes,
                "splits": self.line_class_splits,
                "rule": (
                    f"plurality >= {params.line_class_veto_share} of the votes over >= "
                    f"{params.line_class_veto_frames} frames refuses another class while a "
                    f"same-class candidate exists; a second colour at >= "
                    f"{params.line_class_split_share} of the last "
                    f"{params.line_class_split_window} voting frames splits the id"
                ),
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
            # Sep 29: line rows by the basis that named their tip (the five bases in order of
            # strength: tip_box, hand_track, tail, hand_box, width; `unresolved` otherwise).
            "tip_basis_frames": dict(sorted(self.line_tip_basis_frames.items())),
            "tail_basis": {
                "enabled": params.line_tail_basis,
                "rule": (
                    "per view, the row's tip_side (the axis end with the longer thin tail) is "
                    "matched to the track endpoint it projects nearest and cast as a vote; the "
                    f"net vote over the last {params.line_tail_vote_window} line frames decides "
                    f"at a margin of {params.line_tail_vote_margin}, under a tip box while the "
                    "state is on and under a hand track, over a hand box and the widths; no "
                    f"vote for {list(params.line_wide_tip_classes)} (unvalidated there)"
                ),
                "votes_cast": self.line_tail_votes_cast,
                "votes_skipped_wide_tip_class": self.line_tail_votes_skipped_class,
            },
            "width_basis": {
                "rule": (
                    "per view, mask end widths whose ratio reaches "
                    f"{params.line_width_ratio_min} name the wide end; the wide end is the tip "
                    f"for {list(params.line_wide_tip_classes)} (the manifold), the butt for the "
                    "other line classes (the grip); the net vote over the last "
                    f"{params.line_width_vote_window} line frames decides at a margin of "
                    f"{params.line_width_vote_margin}, over an unresolved track or a hand-box "
                    "decision only"
                ),
                "votes_cast": self.line_width_votes_cast,
            },
            "class_agreement": round(agree / total, 4) if total else None,
            "class_votes_by_track": {
                t.track_id: dict(sorted(t.class_votes.items())) for t in tracks if t.class_votes
            },
            "tips": self._tip_metrics(tracks),
        }

    def _tip_metrics(self, tracks: Sequence[Track]) -> dict[str, Any]:
        """Sep 29, disposable tips: how often a tip box attached (per line frame, view, mode
        and class), the state transitions, the tip votes and the tracks that ever carried a
        tip."""
        params = self.params
        ever = [t for t in tracks if t.tip_ever]
        if params.line_tip_length_rule:
            rule = (
                "the median visible butt-to-tip length over the last "
                f"{params.line_tip_length_window} measured line fits (at least "
                f"{params.line_tip_length_min_frames}) against the class's two length modes: "
                f"on over the midpoint by {params.line_tip_length_hysteresis_cm} cm, off under "
                f"it by as much, no decision more than {params.line_tip_length_truncated_cm} cm "
                "under the bare mode (a truncated mask) or on a class without a with-tip mode "
                "(never on); a *_tip detector box within "
                f"{params.gates.association_px:.1f} px of the axis line and adjacent to an end "
                "still extends that view's extent and is reported as corroboration"
            )
        else:
            rule = (
                f"a *_tip detector box within {params.gates.association_px:.1f} px of the axis "
                "line, beyond the body end and adjacent to it, extends that view's extent; the "
                f"state turns on with >= {params.line_tip_on_frames} of the last "
                f"{params.line_tip_on_window} line frames attached and off after "
                f"{params.line_tip_off_window} without"
            )
        modes = None
        if self.line_prior is not None and params.line_classes:
            modes = {
                cls: {"bare": bare, "with_tip": with_tip}
                for cls in params.line_classes
                for bare, with_tip in [self.line_prior.modes_for(cls)]
            }
        return {
            "enabled": params.tip_boxes_on,
            "tip_classes": dict(params.line_tip_classes),
            "rule": rule,
            "length_rule": {
                "enabled": params.line_tip_length_rule,
                "modes_cm": modes,
                "window": params.line_tip_length_window,
                "min_frames": params.line_tip_length_min_frames,
                "hysteresis_cm": params.line_tip_length_hysteresis_cm,
                "truncated_cm": params.line_tip_length_truncated_cm,
                "line_frames_without_a_median": self.line_tip_length_undecided,
                "line_frames_truncated": self.line_tip_length_truncated,
                "line_frames_over_bare_without_a_mode": self.line_tip_length_no_mode,
                # The tip boxes as corroboration: line frames by state and whether a box
                # attached in some view that frame.
                "box_corroboration": dict(sorted(self.line_tip_box_corroboration.items())),
            },
            "line_frames": self.line_tip_line_frames,
            "line_frames_with_attached_tip": self.line_tip_attached_frames,
            "attached_fraction": (
                round(self.line_tip_attached_frames / self.line_tip_line_frames, 4)
                if self.line_tip_line_frames
                else None
            ),
            "attached_by_view": dict(sorted(self.line_tip_attached_by_view.items())),
            "attached_by_mode": dict(sorted(self.line_tip_attached_by_mode.items())),
            "attached_by_tip_class": dict(sorted(self.line_tip_attached_by_class.items())),
            "side_refused": self.line_tip_side_refused,
            "state_transitions": {"on": self.line_tip_state_on, "off": self.line_tip_state_off},
            "tip_class_votes": self.line_tip_votes,
            "tracks_ever_attached": len(ever),
            "tracks_attached_at_end": sum(1 for t in tracks if t.tip_attached),
            "tip_class_by_track": {
                t.track_id: t.tip_class for t in tracks if t.tip_class is not None
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
    ext.add_argument(
        "--no-line-tip-boxes",
        action="store_true",
        help="Sep 29: do not attach *_tip detector boxes (the v2 behaviour)",
    )
    ext.add_argument(
        "--line-width-over-hand",
        action="store_true",
        help="Sep 29 variant: a decisive mask-width vote names the tip before the hand does",
    )
    ext.add_argument(
        "--no-line-tail-basis",
        action="store_true",
        help="Sep 29 v4: do not read the rows' tip_side as a tip / butt basis (the v3 bases)",
    )
    ext.add_argument(
        "--no-line-tip-length-rule",
        action="store_true",
        help="Sep 29 v4: read tip_attached from the tip boxes' hysteresis (v3) instead of "
        "the 3D butt-to-tip length against the class's two modes",
    )
    ext.add_argument(
        "--line-tail-over-hand",
        action="store_true",
        help="Sep 29 v4 variant: a decisive tail vote names the tip before the hand track does",
    )
    ext.add_argument(
        "--no-line-birth-tip-slack",
        action="store_true",
        help="Sep 29 v4 ablation: a birth pair spans the body plus its tip only with a tip "
        "box attached (v3), not always",
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
        line_tip_boxes=not args.no_line_tip_boxes,
        line_width_over_hand=args.line_width_over_hand,
        line_tail_basis=not args.no_line_tail_basis,
        line_tail_over_hand=args.line_tail_over_hand,
        line_tip_length_rule=not args.no_line_tip_length_rule,
        line_birth_tip_slack=not args.no_line_birth_tip_slack,
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
