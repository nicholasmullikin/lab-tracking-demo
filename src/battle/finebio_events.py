"""`battle-finebio-events`: object-object events from the 3D tracks, with hysteresis (`p5-events`).

Three relations are read off one arm's `Track3D` rows, the trial's rig and its clip config; all
of it is **model output** (3D tracks from detector boxes and SAM3 masks, volumes from the rig's
static triangulations and the detector's box sizes), labelled so on every row and report:

* **contained**: a track's position enters a container volume. A container's footprint is a
  square around the rig's static point for that class (`rig.json["static"]`), whose half-extent
  is the median over the fixed views of half the class's detector-box width converted to
  centimetres at the point's depth; the footprint is extruded from the bench (with a small
  tolerance below it) up to a per-class height (`CONTAINER_HEIGHT_CM`, else twice the rig's
  half-height plus a margin). The centrifuge's lid state (closed intervals from
  `configs/finebio/trials.json`, the T5 rotor heuristic) is a second per-frame series, so "a
  tube inside while the lid is closed" is one look at the strip.
* **held**: a track's position within `held_radius_cm` of a hand track's position (the tracker
  tracks the hands as probes; the rig's per-frame hand triangulations are the fallback).
* **proximity**: a pipette's tip inside the plate volume (the plate track's median position,
  footprint from its boxes, extruded `plate_above_cm` above its top). The tip is the
  triangulation of (mask centroid x, mask bbox bottom) over the side views where the pipette
  has a mask; when fewer than two side views have one, the track point stands in with a wider
  margin (`tip_fallback_margin_cm`), recorded per frame.

Every relation runs through the same hysteresis: a signed distance to the volume (negative
inside) must stay at or under `enter_cm` for `dwell_frames` consecutive frames to start an
episode and at or over `exit_cm` for `dwell_frames` frames to end one; a track that ends
(lost or window end) ends its episodes with that reason. `events.jsonl` holds the start / end
rows (`ObjectEvent`, a `TrackEvent` whose kind set adds `proximity`), `episodes.jsonl` one row
per episode, `events_strip.jsonl` the active relations per frame with the lid state, and
`events.md` the counts, durations and the centrifuge cycles vs `contained` cross-table.

Sep 29, behind `--tip-events` (disposable tips): the line tracker's rows carry `tip_attached`
(a two-state flag with hysteresis from the detector's `*_tip` boxes on the pipette's end) and
the resolved tip end. **tip_picked** is the flag turning on with the tip end inside a
`*_tip_rack` volume in the preceding frames; **tip_ejected** is the flag turning off with the
tip end within a margin of the `trash_can` volume around the last frame that still carried an
attached tip box. The rack and trash volumes are built from the rig's static points the way
the other containers are (`container_volumes`); a rack the rig has no point for is triangulated
from its median detector box centres across the fixed views (the way the plate volume takes
its track's median), and the report says which basis each volume has. No lid attachment
events (no observable). Nothing under `runs/` is committed.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field

from .finebio_arms import ClipWindow, load_clip
from .finebio_cameras import Camera, cameras_from_config, read_camera_config
from .finebio_confidence import resolve_tracks_dir
from .finebio_slice import depth_cm, reprojection_residuals, triangulate_pixels
from .multiview_schemas import FINEBIO_FPV_VIEW, Track3D, TrackEvent, read_jsonl, write_jsonl
from .schemas import VersionedModel

SCHEMA = "battle-finebio-events/1"
MODEL_OUTPUT = (
    "model output: 3D tracks from detector boxes and SAM3 masks, volumes from the rig's static "
    "triangulations and the detector's box sizes, lid state from a pixel heuristic; no human "
    "label"
)
EVENT_KINDS: tuple[str, ...] = ("contained", "held", "proximity")
# Per-class container heights in cm above the bench; anything else uses twice the rig's
# half-height plus a margin. Parameters, not measurements (the rig gives half-heights only).
CONTAINER_HEIGHT_CM: dict[str, float] = {
    "centrifuge": 22.0,
    "vortex_mixer": 12.0,
    "pcr_machine": 14.0,
    "micro_tube_rack": 8.0,
    "50ml_tube_rack": 12.0,
    "15ml_tube_rack": 12.0,
    "8_tube_stripes_rack": 6.0,
    "magnetic_rack": 8.0,
    "trash_can": 22.0,
}
DEFAULT_HEIGHT_MARGIN_CM = 3.0
BENCH_TOLERANCE_CM = 2.0
PLATE_CLASS = "cell_culture_plate"
PIPETTE_SUFFIX = "_pipette"
CONTAINABLE_CLASSES: tuple[str, ...] = (
    "micro_tube",
    "50ml_tube",
    "15ml_tube",
    "8_tube_stripes",
    PLATE_CLASS,
)
HAND_STATES = ("observed", "single_view")
# Side views for the pipette tip: the top-down camera and the head camera see no "bottom".
TOP_DOWN_VIEW = "T5"
DEFAULT_TRIALS = Path("configs/finebio/trials.json")


@dataclass(frozen=True)
class EventParams:
    enter_cm: float = 0.0
    exit_cm: float = 3.0
    dwell_frames: int = 5
    held_radius_cm: float = 12.0
    held_exit_margin_cm: float = 5.0
    proximity_margin_cm: float = 3.0
    plate_above_cm: float = 6.0
    tip_fallback_margin_cm: float = 8.0
    footprint_scale: float = 1.0
    proximity_targets: tuple[str, ...] = (PLATE_CLASS,)
    # `contained` is judged for the classes a container can hold (their `_group` slots too);
    # a pipette passing over the centrifuge is the hand's business (`held`), not containment.
    containable_classes: tuple[str, ...] = CONTAINABLE_CLASSES
    # `held` starts only when the object has moved over the dwell window (a tube resting in a
    # rack under a hand is occluded, not carried: the arms' occlusion inventory).
    held_min_motion_cm: float = 2.0

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


class ObjectEvent(TrackEvent):
    """A `TrackEvent` whose kind set adds `proximity` (the schema's `TrackEventKind` has `held`
    and `contained`; `proximity` is this module's addition, one line for the schema owner) and,
    Sep 29 behind `--tip-events`, `tip_picked` / `tip_ejected`."""

    kind: Literal["held", "contained", "proximity", "tip_picked", "tip_ejected"]  # type: ignore[assignment]


class EventStripRow(VersionedModel):
    """The active relations at one frame."""

    frame_index: int = Field(ge=0)
    lid_closed: bool | None = None
    contained: list[dict[str, Any]] = Field(default_factory=list)
    held: list[dict[str, Any]] = Field(default_factory=list)
    proximity: list[dict[str, Any]] = Field(default_factory=list)
    model_output: bool = True


# --------------------------------------------------------------------------- volumes


@dataclass(frozen=True)
class Volume:
    """An axis-aligned box in board centimetres: `z_top` is the highest point (most negative z,
    z pointing into the bench) and `z_bottom` the lowest (at or just below the bench)."""

    name: str
    object_class: str
    centre_xy: tuple[float, float]
    half_x_cm: float
    half_y_cm: float
    z_top: float
    z_bottom: float
    provenance: dict[str, Any] = field(default_factory=dict)

    def signed_distance(self, point: Sequence[float]) -> float:
        dx = abs(point[0] - self.centre_xy[0]) - self.half_x_cm
        dy = abs(point[1] - self.centre_xy[1]) - self.half_y_cm
        dz = max(self.z_top - point[2], point[2] - self.z_bottom)
        return float(max(dx, dy, dz))

    def to_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "object_class": self.object_class,
            "centre_xy_cm": [round(v, 2) for v in self.centre_xy],
            "half_x_cm": round(self.half_x_cm, 2),
            "half_y_cm": round(self.half_y_cm, 2),
            "z_top_cm": round(self.z_top, 2),
            "z_bottom_cm": round(self.z_bottom, 2),
            "provenance": self.provenance,
        }


def median_box_sizes(
    observations: Path, classes: Iterable[str], *, min_score: float = 0.3
) -> dict[tuple[str, str], tuple[float, float]]:
    """(view, class) -> median (width, height) px of the top-ranked detector box per frame."""
    wanted = set(classes)
    sizes: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    with Path(observations).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("source") != "detector" or row["object_class"] not in wanted:
                continue
            if not row["slot"].endswith("#0") or (row.get("detector_score") or 0) < min_score:
                continue
            box = row["box_xyxy_px"]
            sizes[(row["view"], row["object_class"])].append((box[2] - box[0], box[3] - box[1]))
    return {
        key: (float(np.median([w for w, _ in v])), float(np.median([h for _, h in v])))
        for key, v in sizes.items()
    }


def footprint_half_extent_cm(
    point: np.ndarray,
    cams: dict[str, Camera],
    box_sizes: dict[tuple[str, str], tuple[float, float]],
    object_class: str,
    views: Iterable[str],
) -> tuple[float | None, dict[str, float]]:
    """Half the detector-box width in cm at the point's depth, per view, and their median."""
    per_view: dict[str, float] = {}
    for view in views:
        cam = cams.get(view)
        size = box_sizes.get((view, object_class))
        if cam is None or size is None:
            continue
        depth = depth_cm(cam, point)
        if depth <= 0:
            continue
        per_view[view] = round(0.5 * size[0] * depth / float(cam.K[0, 0]), 2)
    if not per_view:
        return None, per_view
    return float(np.median(list(per_view.values()))), per_view


def container_volumes(
    rig: dict[str, Any],
    containers: Iterable[str],
    cams: dict[str, Camera],
    box_sizes: dict[tuple[str, str], tuple[float, float]],
    *,
    heights: dict[str, float] | None = None,
    scale: float = 1.0,
) -> list[Volume]:
    heights = {**CONTAINER_HEIGHT_CM, **(heights or {})}
    wanted = set(containers)
    volumes = []
    for static in rig.get("static", ()):
        cls = static["class"]
        if cls not in wanted:
            continue
        point = np.asarray(static["point_cm"], dtype=np.float64)
        half, per_view = footprint_half_extent_cm(
            point, cams, box_sizes, cls, static.get("views", cams.keys())
        )
        if half is None:
            continue
        rig_height = float(static.get("height_cm", -point[2]))
        height = heights.get(cls, max(2.0 * rig_height + DEFAULT_HEIGHT_MARGIN_CM, 6.0))
        volumes.append(
            Volume(
                name=cls,
                object_class=cls,
                centre_xy=(float(point[0]), float(point[1])),
                half_x_cm=half * scale,
                half_y_cm=half * scale,
                z_top=-height,
                z_bottom=BENCH_TOLERANCE_CM,
                provenance={
                    "centre": "rig static point",
                    "rig_height_cm": round(rig_height, 2),
                    "height_cm": height,
                    "height_source": "CONTAINER_HEIGHT_CM" if cls in heights else "2x rig + margin",
                    "half_extent_per_view_cm": per_view,
                    "footprint_scale": scale,
                },
            )
        )
    return volumes


def plate_volume(
    rows: Sequence[Track3D],
    cams: dict[str, Camera],
    box_sizes: dict[tuple[str, str], tuple[float, float]],
    *,
    above_cm: float,
    plate_class: str = PLATE_CLASS,
    scale: float = 1.0,
) -> Volume | None:
    """The plate as a target volume: its track's median position, its footprint from its
    boxes, extruded from the bench to `above_cm` above the plate's top."""
    points = [r.position_cm for r in rows if r.object_class == plate_class and r.state != "lost"]
    if not points:
        return None
    centre = np.median(np.asarray(points, dtype=np.float64), axis=0)
    half, per_view = footprint_half_extent_cm(centre, cams, box_sizes, plate_class, cams.keys())
    if half is None:
        return None
    top = min(float(centre[2]), 0.0) - above_cm
    return Volume(
        name=plate_class,
        object_class=plate_class,
        centre_xy=(float(centre[0]), float(centre[1])),
        half_x_cm=half * scale,
        half_y_cm=half * scale,
        z_top=top,
        z_bottom=BENCH_TOLERANCE_CM,
        provenance={
            "centre": "median of the plate track",
            "frames": len(points),
            "above_cm": above_cm,
            "half_extent_per_view_cm": per_view,
            "footprint_scale": scale,
        },
    )


TIP_RACK_SUFFIX = "_tip_rack"
TRASH_CLASS = "trash_can"
TIP_EVENT_KINDS: tuple[str, ...] = ("tip_picked", "tip_ejected")


def median_box_centres(
    observations: Path, classes: Iterable[str], *, min_score: float = 0.3, min_rows: int = 10
) -> dict[tuple[str, str], tuple[float, float]]:
    """(view, class) -> median centre px of the top-ranked detector box, over views with at
    least `min_rows` rows (the plate volume's method applied to a class without a track)."""
    wanted = set(classes)
    centres: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    with Path(observations).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("source") != "detector" or row["object_class"] not in wanted:
                continue
            if not row["slot"].endswith("#0") or (row.get("detector_score") or 0) < min_score:
                continue
            box = row["box_xyxy_px"]
            centres[(row["view"], row["object_class"])].append(
                ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            )
    return {
        key: (float(np.median([x for x, _ in v])), float(np.median([y for _, y in v])))
        for key, v in centres.items()
        if len(v) >= min_rows
    }


def box_volumes(
    classes: Iterable[str],
    cams: dict[str, Camera],
    box_centres: dict[tuple[str, str], tuple[float, float]],
    box_sizes: dict[tuple[str, str], tuple[float, float]],
    *,
    heights: dict[str, float] | None = None,
    gate_px: float = 30.0,
    min_views: int = 2,
) -> tuple[list[Volume], dict[str, str]]:
    """Volumes for classes the rig has no static point for: the median detector box centres
    of the fixed views triangulated (at least `min_views`, every reprojection within
    `gate_px`), the footprint from the box sizes as the plate volume takes it. Returns the
    volumes and, per class not built, the reason."""
    heights = {**CONTAINER_HEIGHT_CM, **(heights or {})}
    volumes: list[Volume] = []
    skipped: dict[str, str] = {}
    for cls in classes:
        pixels = {
            view: np.asarray(box_centres[(view, cls)], dtype=np.float64)
            for view in cams
            if (view, cls) in box_centres
        }
        if len(pixels) < min_views:
            skipped[cls] = f"{len(pixels)} fixed views with a stable box, need {min_views}"
            continue
        views = list(pixels)
        point = triangulate_pixels([cams[v] for v in views], [pixels[v] for v in views])
        if not np.all(np.isfinite(point)):
            skipped[cls] = "the box centres do not triangulate"
            continue
        residuals = reprojection_residuals(cams, pixels, point)
        if residuals and max(residuals.values()) > gate_px:
            skipped[cls] = (
                f"box centres disagree: reprojection {max(residuals.values()):.1f} px over "
                f"the {gate_px:.0f} px gate"
            )
            continue
        half, per_view = footprint_half_extent_cm(point, cams, box_sizes, cls, views)
        if half is None:
            skipped[cls] = "no box size to give a footprint"
            continue
        height = heights.get(cls, max(2.0 * float(-point[2]) + DEFAULT_HEIGHT_MARGIN_CM, 6.0))
        volumes.append(
            Volume(
                name=cls,
                object_class=cls,
                centre_xy=(float(point[0]), float(point[1])),
                half_x_cm=half,
                half_y_cm=half,
                z_top=-height,
                z_bottom=BENCH_TOLERANCE_CM,
                provenance={
                    "centre": "median detector box centres triangulated (no rig static point)",
                    "views": views,
                    "reprojection_px": {v: round(r, 2) for v, r in residuals.items()},
                    "height_cm": height,
                    "height_source": "CONTAINER_HEIGHT_CM"
                    if cls in heights
                    else "2x point + margin",
                    "half_extent_per_view_cm": per_view,
                },
            )
        )
    return volumes, skipped


def tip_end_of(row: Track3D) -> np.ndarray | None:
    """The resolved tip end of a line row, else None."""
    if row.endpoints_cm is None or not row.tip_resolved:
        return None
    return np.asarray(row.endpoints_cm[0], dtype=np.float64)


def _either_end_inside(row: Track3D, volume: Volume, margin: float) -> bool:
    """Either end of the segment (the tip when resolved, both when not) within `margin` of
    the volume; the track point when the row is not a line."""
    if row.endpoints_cm is None:
        return volume.signed_distance(row.position_cm) <= margin
    tip = tip_end_of(row)
    ends = [tip] if tip is not None else [np.asarray(e, dtype=np.float64) for e in row.endpoints_cm]
    return any(volume.signed_distance(e) <= margin for e in ends)


def tip_events(
    rows: Sequence[Track3D],
    *,
    racks: Sequence[Volume],
    trash: Sequence[Volume],
    lookback_frames: int = 15,
    margin_cm: float = 5.0,
) -> tuple[list[ObjectEvent], dict[str, Any]]:
    """Sep 29: per line track, the `tip_attached` transitions. Off -> on (or undecided -> on)
    with the tip end (either end while unresolved) inside a rack volume on one of the
    `lookback_frames` frames up to the flip is `tip_picked`; on -> off with the tip end within
    `margin_cm` of the trash volume within `lookback_frames` frames of the last frame that
    still attached a tip box (the hysteresis turns the flag off later) is `tip_ejected`. Flips
    that met neither volume are counted as such, with the nearest volume, and not emitted.
    Sep 29 v4: the state comes from the 3D length and a box may be long gone or never have
    attached, so the eject window is centred on the last attached box only when that box
    lies within `lookback_frames` of the flip, else on the flip itself."""
    by_track: dict[str, list[Track3D]] = defaultdict(list)
    for r in rows:
        if r.tip_attached is not None or r.tip_attached_views:
            by_track[r.track_id].append(r)
    events: list[ObjectEvent] = []
    counts: Counter = Counter()
    unmatched: list[dict[str, Any]] = []
    for track_id, trows in sorted(by_track.items()):
        trows.sort(key=lambda r: r.frame_index)
        previous: bool | None = None
        for i, row in enumerate(trows):
            state = row.tip_attached
            if state is None or state == previous or (state is False and previous is None):
                # Undecided, unchanged, or the first decision being "no tip": not a flip.
                previous = state if state is not None else previous
                continue
            window = [r for r in trows[max(0, i - lookback_frames) : i + 1]]
            if state is True:
                hits = [
                    (v.name, r.frame_index)
                    for r in window
                    for v in racks
                    if _either_end_inside(r, v, 0.0)
                ]
                kind = "tip_picked"
                targets = racks
            else:
                last_attached = next(
                    (r for r in reversed(trows[: i + 1]) if r.tip_attached_views), None
                )
                centre = row.frame_index
                if (
                    last_attached is not None
                    and row.frame_index - last_attached.frame_index <= lookback_frames
                ):
                    centre = last_attached.frame_index
                window = [
                    r
                    for r in trows
                    if centre - lookback_frames <= r.frame_index <= centre + lookback_frames
                ]
                hits = [
                    (v.name, r.frame_index)
                    for r in window
                    for v in trash
                    if _either_end_inside(r, v, margin_cm)
                ]
                kind = "tip_ejected"
                targets = trash
            previous_state = previous
            previous = state
            if hits:
                target = Counter(name for name, _ in hits).most_common(1)[0][0]
                counts[kind] += 1
                events.append(
                    ObjectEvent(
                        frame_index=row.frame_index,
                        track_id=track_id,
                        kind=kind,  # type: ignore[arg-type]
                        payload={
                            "object_class": row.observed_class or row.object_class,
                            "tip_class": row.tip_class,
                            "target": target,
                            "from_state": previous_state,
                            "frames_in_volume": len({f for _, f in hits}),
                            "first_frame_in_volume": min(f for _, f in hits),
                            "tracker_state": row.state,
                            "source": "geometry",
                            "model_output": True,
                        },
                    )
                )
            else:
                nearest = None
                tip = tip_end_of(row)
                if tip is not None and targets:
                    nearest = min(
                        ((v.name, v.signed_distance(tip)) for v in targets), key=lambda x: x[1]
                    )
                counts[f"{kind}_unmatched"] += 1
                unmatched.append(
                    {
                        "track_id": track_id,
                        "frame_index": row.frame_index,
                        "kind": kind,
                        "from_state": previous_state,
                        "nearest_volume": None if nearest is None else nearest[0],
                        "nearest_distance_cm": None if nearest is None else round(nearest[1], 1),
                    }
                )
    summary = {
        "rule": (
            f"tip_picked: tip_attached turns on with the tip end inside a *_tip_rack volume on "
            f"one of the {lookback_frames} frames up to the flip; tip_ejected: it turns off with "
            f"the tip end within {margin_cm} cm of the trash_can volume within {lookback_frames} "
            "frames of the last attached tip box"
        ),
        "counts": {k: counts.get(k, 0) for k in ("tip_picked", "tip_ejected")},
        "unmatched_flips": {
            k: counts.get(f"{k}_unmatched", 0) for k in ("tip_picked", "tip_ejected")
        },
        "events": [
            {
                "kind": e.kind,
                "track_id": e.track_id,
                "frame_index": e.frame_index,
                "target": e.payload["target"],
                "object_class": e.payload["object_class"],
                "tip_class": e.payload["tip_class"],
                "from_state": e.payload["from_state"],
            }
            for e in events
        ],
        "unmatched": unmatched,
        "racks": [v.to_record() for v in racks],
        "trash": [v.to_record() for v in trash],
        "tracks_with_a_tip_state": len(by_track),
    }
    return events, summary


# --------------------------------------------------------------------------- lid state


def lid_closed_intervals(trials: dict[str, Any], trial: str) -> list[tuple[int, int]]:
    for entry in trials.get("trials", ()):
        if entry.get("trial") == trial:
            return [
                (int(a), int(b))
                for a, b in entry.get("centrifuge", {}).get("lid_closed_intervals_all", ())
            ]
    return []


def lid_closed_at(intervals: Sequence[tuple[int, int]], frame: int) -> bool:
    return any(a <= frame < b for a, b in intervals)


def cycles_in_window(
    intervals: Sequence[tuple[int, int]], start: int, end_exclusive: int
) -> list[tuple[int, int]]:
    """Closed intervals that begin inside the window (a lid closed at the window start is the
    initial state, not a cycle)."""
    return [(a, b) for a, b in intervals if start < a < end_exclusive]


# --------------------------------------------------------------------------- hysteresis


@dataclass
class Hysteresis:
    """Enter when the signed distance stays <= `enter` for `dwell` frames, exit when it stays
    >= `exit` for `dwell` frames; frames between the two thresholds keep the current state."""

    enter: float
    exit: float
    dwell: int
    inside: bool = False
    run: int = 0
    run_start: int | None = None
    start_frame: int | None = None

    def feed(self, frame: int, distance: float) -> tuple[str, int] | None:
        """Returns ("start", first frame) or ("end", last inside frame) on a transition."""
        if not self.inside:
            if distance <= self.enter:
                if self.run == 0:
                    self.run_start = frame
                self.run += 1
                if self.run >= self.dwell:
                    self.inside, self.run = True, 0
                    self.start_frame = frame if self.run_start is None else self.run_start
                    return ("start", self.start_frame)
            else:
                self.run = 0
            return None
        if distance >= self.exit:
            if self.run == 0:
                self.run_start = frame
            self.run += 1
            if self.run >= self.dwell:
                self.inside, self.run = False, 0
                first_outside = frame if self.run_start is None else self.run_start
                return ("end", first_outside - 1)
        else:
            self.run = 0
        return None


@dataclass
class Episode:
    kind: str
    track_id: str
    object_class: str
    target: str
    target_class: str
    start_frame: int
    end_frame: int | None = None
    end_reason: str | None = None
    frames_lid_closed: int = 0
    frames: int = 0
    min_distance_cm: float | None = None
    tip_from_masks_frames: int = 0
    tracker_state_frames: dict[str, int] = field(default_factory=Counter)

    @property
    def duration_frames(self) -> int | None:
        return None if self.end_frame is None else self.end_frame - self.start_frame + 1

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["duration_frames"] = self.duration_frames
        record["tracker_state_frames"] = dict(self.tracker_state_frames)
        record["model_output"] = True
        return record


# --------------------------------------------------------------------------- per-frame inputs


def by_frame(rows: Iterable[Track3D]) -> dict[int, list[Track3D]]:
    out: dict[int, list[Track3D]] = defaultdict(list)
    for r in rows:
        out[r.frame_index].append(r)
    return dict(sorted(out.items()))


def hand_positions(
    rows: Sequence[Track3D], probes: Sequence[str]
) -> dict[int, list[tuple[str, str, np.ndarray]]]:
    """frame -> [(hand track id, class, position)] from the tracker's hand tracks."""
    out: dict[int, list[tuple[str, str, np.ndarray]]] = defaultdict(list)
    for r in rows:
        if r.object_class in probes and r.state in HAND_STATES:
            out[r.frame_index].append((r.track_id, r.object_class, np.asarray(r.position_cm)))
    return out


def rig_hand_positions(rig: dict[str, Any]) -> dict[int, list[tuple[str, str, np.ndarray]]]:
    out: dict[int, list[tuple[str, str, np.ndarray]]] = defaultdict(list)
    for hand, entries in (rig.get("hands") or {}).items():
        for e in entries:
            out[int(e["frame"])].append(
                (f"rig/{hand}", hand, np.asarray(e["point_cm"], dtype=np.float64))
            )
    return out


TipKey = tuple[str, int, str]


def load_tip_pixels(
    observations: Path, wanted: set[TipKey], *, exclude_views: Sequence[str] = ()
) -> dict[TipKey, tuple[float, float]]:
    """(view, frame, slot) -> (mask centroid x, mask bbox bottom) for the wanted keys."""
    out: dict[TipKey, tuple[float, float]] = {}
    excluded = set(exclude_views)
    with Path(observations).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["view"] in excluded:
                continue
            key = (row["view"], int(row["frame_index"]), row["slot"])
            if key not in wanted or row.get("mask_bbox_px") is None:
                continue
            bbox = row["mask_bbox_px"]
            centroid = row.get("mask_centroid_px") or [(bbox[0] + bbox[2]) / 2, 0.0]
            out[key] = (float(centroid[0]), float(bbox[3]))
    return out


def pipette_tip(
    row: Track3D,
    tips: dict[TipKey, tuple[float, float]],
    cams: dict[str, Camera],
    *,
    gate_px: float,
) -> tuple[np.ndarray, bool]:
    """The tip triangulated from >= 2 side views' (centroid x, bbox bottom), accepted when every
    reprojection residual is within the gate; else the track point (flagged)."""
    pixels = {}
    for view, slot in row.support_slots.items():
        pixel = tips.get((view, row.frame_index, slot))
        if pixel is not None and view in cams:
            pixels[view] = np.asarray(pixel, dtype=np.float64)
    if len(pixels) >= 2:
        views = list(pixels)
        point = triangulate_pixels([cams[v] for v in views], [pixels[v] for v in views])
        if np.all(np.isfinite(point)):
            residuals = reprojection_residuals(cams, pixels, point)
            if residuals and max(residuals.values()) <= gate_px:
                return point, True
    return np.asarray(row.position_cm, dtype=np.float64), False


# --------------------------------------------------------------------------- the pass


def base_class(object_class: str) -> str:
    suffix = "_group"
    return object_class[: -len(suffix)] if object_class.endswith(suffix) else object_class


def eligible_object(row: Track3D, targets: Sequence[str], excluded: set[str]) -> bool:
    return base_class(row.object_class) in targets and row.object_class not in excluded


def moved_over_window(history: Sequence[np.ndarray], min_cm: float) -> bool:
    """Has the track's position moved at least `min_cm` between the oldest and the newest of
    the last `dwell` positions (the co-motion a carried object shows and a resting one does not)."""
    if min_cm <= 0:
        return True
    if len(history) < 2:
        return False
    return float(np.linalg.norm(history[-1] - history[0])) >= min_cm


def detect_events(
    rows: Sequence[Track3D],
    *,
    containers: Sequence[Volume],
    hands: dict[int, list[tuple[str, str, np.ndarray]]],
    proximity_targets: Sequence[Volume],
    tips: dict[TipKey, tuple[float, float]],
    side_cams: dict[str, Camera],
    lid_intervals: Sequence[tuple[int, int]],
    frames: Sequence[int],
    targets: Sequence[str],
    params: EventParams,
    gate_px: float = 30.0,
) -> tuple[list[Episode], list[EventStripRow], list[ObjectEvent]]:
    """One pass over the frames: per eligible track row the signed distances to every container,
    to the nearest hand of each hand class, and (pipettes) of the tip to every proximity target,
    each fed through its own hysteresis; the strip is the open episodes per frame."""
    excluded = {v.object_class for v in containers}
    containable = set(params.containable_classes)
    frame_rows = by_frame(rows)
    detector = _Detector(params)
    strip: list[EventStripRow] = []
    tip_flag: dict[tuple[str, int], bool] = {}
    history: dict[str, list[np.ndarray]] = defaultdict(list)
    for frame in frames:
        lid = lid_closed_at(lid_intervals, frame) if lid_intervals else None
        present = frame_rows.get(frame, [])
        live = {r.track_id for r in present if r.state != "lost"}
        lost = {r.track_id for r in present if r.state == "lost"}
        hands_here = hands.get(frame, [])
        for row in present:
            if row.state == "lost" or not eligible_object(row, targets, excluded):
                continue
            point = np.asarray(row.position_cm, dtype=np.float64)
            past = history[row.track_id]
            past.append(point)
            del past[: -params.dwell_frames]
            if base_class(row.object_class) in containable:
                for volume in containers:
                    detector.feed(
                        ("contained", row.track_id, volume.name),
                        row,
                        frame,
                        volume.signed_distance(point),
                        volume.object_class,
                        lid,
                    )
            hand_classes = {h[1] for h in hands_here} | {
                key[2] for key in detector.open if key[0] == "held" and key[1] == row.track_id
            }
            moving = moved_over_window(past, params.held_min_motion_cm)
            for hand_class in sorted(hand_classes):
                same = [h for h in hands_here if h[1] == hand_class]
                key = ("held", row.track_id, hand_class)
                if same:
                    nearest = min(same, key=lambda h: float(np.linalg.norm(point - h[2])))
                    distance = float(np.linalg.norm(point - nearest[2])) - params.held_radius_cm
                    extra = {"hand_track": nearest[0]}
                    if key not in detector.open and not moving:
                        distance = float("inf")
                else:
                    distance, extra = float("inf"), {}
                detector.feed(key, row, frame, distance, hand_class, lid, extra)
            if row.object_class.endswith(PIPETTE_SUFFIX) and proximity_targets:
                tip, from_masks = pipette_tip(row, tips, side_cams, gate_px=gate_px)
                tip_flag[(row.track_id, frame)] = from_masks
                margin = params.proximity_margin_cm
                if not from_masks:
                    margin += params.tip_fallback_margin_cm
                for volume in proximity_targets:
                    detector.feed(
                        ("proximity", row.track_id, volume.name),
                        row,
                        frame,
                        volume.signed_distance(tip) - margin,
                        volume.object_class,
                        lid,
                        {"tip_from_masks": from_masks},
                    )
        detector.close_gone(frame, live, lost)
        strip.append(
            EventStripRow(
                frame_index=frame,
                lid_closed=lid,
                contained=[
                    {"track_id": ep.track_id, "object_class": ep.object_class, "target": ep.target}
                    for ep in detector.open.values()
                    if ep.kind == "contained"
                ],
                held=[
                    {"track_id": ep.track_id, "object_class": ep.object_class, "hand": ep.target}
                    for ep in detector.open.values()
                    if ep.kind == "held"
                ],
                proximity=[
                    {
                        "track_id": ep.track_id,
                        "object_class": ep.object_class,
                        "target": ep.target,
                        "tip_from_masks": tip_flag.get((ep.track_id, frame)),
                    }
                    for ep in detector.open.values()
                    if ep.kind == "proximity"
                ],
            )
        )
    if frames:
        for key in list(detector.open):
            detector.close(key, frames[-1], "window_end")
    episodes = sorted(detector.episodes, key=lambda e: (e.start_frame, e.kind, e.track_id))
    events = sorted(
        detector.events,
        key=lambda e: (e.frame_index, e.track_id, e.payload.get("phase") == "end"),
    )
    return episodes, strip, events


class _Detector:
    """The hysteresis states, the open episodes and the emitted rows of one pass."""

    def __init__(self, params: EventParams) -> None:
        self.params = params
        self.states: dict[tuple[str, str, str], Hysteresis] = {}
        self.open: dict[tuple[str, str, str], Episode] = {}
        self.episodes: list[Episode] = []
        self.events: list[ObjectEvent] = []
        self.last_live: dict[str, int] = {}
        # Per key, the frames of the current entry (or exit) run: an episode starts at the first
        # frame of its entry run, so those frames' statistics join it once the dwell is met;
        # the frames of an exit run leave with it once the episode ends.
        self.pending: dict[tuple[str, str, str], list[tuple[Any, ...]]] = defaultdict(list)

    @staticmethod
    def _apply(ep: Episode, stats: tuple[Any, ...]) -> None:
        lid, state, distance, extra = stats
        ep.frames += 1
        ep.frames_lid_closed += int(bool(lid))
        ep.tracker_state_frames[state] += 1
        if np.isfinite(distance):
            d = round(distance, 2)
            ep.min_distance_cm = d if ep.min_distance_cm is None else min(ep.min_distance_cm, d)
        if extra.get("tip_from_masks"):
            ep.tip_from_masks_frames += 1

    def _hysteresis(self, key: tuple[str, str, str]) -> Hysteresis:
        if key not in self.states:
            p = self.params
            if key[0] == "held":
                self.states[key] = Hysteresis(0.0, p.held_exit_margin_cm, p.dwell_frames)
            else:
                self.states[key] = Hysteresis(p.enter_cm, p.exit_cm, p.dwell_frames)
        return self.states[key]

    def feed(
        self,
        key: tuple[str, str, str],
        row: Track3D,
        frame: int,
        distance: float,
        target_class: str,
        lid: bool | None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        extra = extra or {}
        self.last_live[row.track_id] = frame
        hysteresis = self._hysteresis(key)
        was_inside = hysteresis.inside
        transition = hysteresis.feed(frame, distance)
        stats = (lid, row.state, distance, extra)
        pending = self.pending[key]
        ep = self.open.get(key)
        if transition is not None and transition[0] == "start" and ep is None:
            ep = Episode(
                kind=key[0],
                track_id=row.track_id,
                object_class=row.object_class,
                target=key[2],
                target_class=target_class,
                start_frame=transition[1],
            )
            self.open[key] = ep
            self.episodes.append(ep)
            self.events.append(
                ObjectEvent(
                    frame_index=transition[1],
                    track_id=row.track_id,
                    kind=key[0],  # type: ignore[arg-type]
                    payload={
                        "phase": "start",
                        "object_class": row.object_class,
                        "target": key[2],
                        "target_class": target_class,
                        "distance_cm": round(distance, 2) if np.isfinite(distance) else None,
                        "lid_closed": lid,
                        "tracker_state": row.state,
                        "source": "geometry",
                        "model_output": True,
                        **extra,
                    },
                )
            )
            for past in pending:
                self._apply(ep, past)
            self._apply(ep, stats)
            pending.clear()
            return
        if not was_inside:
            if hysteresis.run > 0:
                pending.append(stats)
            else:
                pending.clear()
            return
        if transition is not None and transition[0] == "end":
            pending.clear()
            if key in self.open:
                self.close(key, transition[1], "exit")
            return
        if hysteresis.run > 0:
            pending.append(stats)
        elif ep is not None:
            for past in pending:
                self._apply(ep, past)
            pending.clear()
            self._apply(ep, stats)

    def close(self, key: tuple[str, str, str], end_frame: int, reason: str) -> None:
        ep = self.open.pop(key)
        ep.end_frame, ep.end_reason = max(end_frame, ep.start_frame), reason
        self.events.append(
            ObjectEvent(
                frame_index=ep.end_frame,
                track_id=ep.track_id,
                kind=ep.kind,  # type: ignore[arg-type]
                payload={
                    "phase": "end",
                    "object_class": ep.object_class,
                    "target": ep.target,
                    "target_class": ep.target_class,
                    "start_frame": ep.start_frame,
                    "duration_frames": ep.duration_frames,
                    "end_reason": reason,
                    "frames_lid_closed": ep.frames_lid_closed,
                    "source": "geometry",
                    "model_output": True,
                },
            )
        )
        state = self.states[key]
        self.states[key] = Hysteresis(state.enter, state.exit, state.dwell)
        self.pending[key].clear()

    def close_gone(self, frame: int, live: set[str], lost: set[str]) -> None:
        """Episodes of tracks without a live row this frame end at the track's last live frame."""
        for key in list(self.open):
            track_id = key[1]
            if track_id in live:
                continue
            reason = "track_lost" if track_id in lost else "track_gone"
            self.close(key, self.last_live.get(track_id, frame - 1), reason)


# --------------------------------------------------------------------------- summary


def _dist(values: Sequence[float]) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "median": None, "p90": None, "max": None}
    return {
        "n": int(arr.size),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
    }


def cycle_cross_table(
    episodes: Sequence[Episode],
    rows: Sequence[Track3D],
    cycles: Sequence[tuple[int, int]],
    *,
    container: str = "centrifuge",
    margin_frames: int = 90,
    successor_radius_cm: float = 10.0,
) -> list[dict[str, Any]]:
    """Per lid-closed cycle: the `contained` episodes in the container that overlap it (with
    loading before, ending while closed, surviving), and the same-class births within
    `margin_frames` after the lid opens near an episode's last position (the successor ids the
    coast timeout produces when the closure outlasts it)."""
    births: dict[str, Track3D] = {}
    last_rows: dict[str, Track3D] = {}
    for r in rows:
        if r.track_id not in births:
            births[r.track_id] = r
        last_rows[r.track_id] = r
    table = []
    for a, b in cycles:
        overlapping = [
            ep
            for ep in episodes
            if ep.kind == "contained"
            and ep.target == container
            and ep.end_frame is not None
            and ep.start_frame < b + margin_frames
            and ep.end_frame >= a - margin_frames
        ]
        entries = []
        for ep in overlapping:
            assert ep.end_frame is not None
            last = last_rows.get(ep.track_id)
            if ep.start_frame >= b:
                phase = "after_opening"
            elif ep.end_frame < a:
                phase = "before_closure"
            elif ep.end_frame >= b:
                phase = "through_the_cycle"
            else:
                phase = "ended_while_closed"
            successors = []
            # Successor ids matter for an episode the closure outlasted: the same-class births
            # near its last position once the lid re-opens.
            if last is not None and ep.end_frame < b:
                for tid, birth in births.items():
                    if (
                        tid != ep.track_id
                        and birth.object_class == ep.object_class
                        and b <= birth.frame_index < b + margin_frames
                        and float(np.linalg.norm(np.subtract(birth.position_cm, last.position_cm)))
                        <= successor_radius_cm
                    ):
                        successors.append(
                            {
                                "track_id": tid,
                                "frame_index": birth.frame_index,
                                "possibly_same_as": list(birth.possibly_same_as),
                            }
                        )
            entries.append(
                {
                    "track_id": ep.track_id,
                    "object_class": ep.object_class,
                    "start_frame": ep.start_frame,
                    "end_frame": ep.end_frame,
                    "end_reason": ep.end_reason,
                    "phase": phase,
                    "entered_before_closure": ep.start_frame < a,
                    "inside_while_closed_frames": max(
                        0, min(ep.end_frame, b - 1) - max(ep.start_frame, a) + 1
                    ),
                    "ended_while_closed": phase == "ended_while_closed",
                    "same_id_after_opening": phase == "through_the_cycle",
                    "successors_after_opening": successors,
                }
            )
        table.append(
            {
                "cycle": [a, b],
                "closed_frames": b - a,
                "contained_episodes": entries,
                "episodes": len(entries),
                "entered_before_closure": sum(e["entered_before_closure"] for e in entries),
                "ended_while_closed": sum(e["ended_while_closed"] for e in entries),
                "same_id_after_opening": sum(e["same_id_after_opening"] for e in entries),
                "started_after_opening": sum(e["phase"] == "after_opening" for e in entries),
                "with_successor_after_opening": sum(
                    bool(e["successors_after_opening"]) for e in entries
                ),
            }
        )
    return table


def summarise(
    episodes: Sequence[Episode],
    strip: Sequence[EventStripRow],
    *,
    arm: str,
    params: EventParams,
    volumes: Sequence[Volume],
    proximity_targets: Sequence[Volume],
    cycles: Sequence[tuple[int, int]],
    cross: Sequence[dict[str, Any]],
    tracks_dir: Path,
    hand_source: str,
) -> dict[str, Any]:
    by_kind: dict[str, list[Episode]] = defaultdict(list)
    for ep in episodes:
        by_kind[ep.kind].append(ep)
    kinds = {}
    for kind in EVENT_KINDS:
        eps = by_kind.get(kind, [])
        durations = [ep.duration_frames for ep in eps if ep.duration_frames is not None]
        kinds[kind] = {
            "episodes": len(eps),
            "tracks": len({ep.track_id for ep in eps}),
            "by_class": dict(Counter(ep.object_class for ep in eps).most_common()),
            "by_target": dict(Counter(ep.target for ep in eps).most_common()),
            "end_reasons": dict(Counter(ep.end_reason for ep in eps).most_common()),
            "duration_frames": _dist(durations),
            "frames_active": sum(len(getattr(row, kind)) for row in strip),
            "frames_with_any": sum(1 for row in strip if getattr(row, kind)),
        }
        if kind == "proximity":
            kinds[kind]["tip_from_masks_frames"] = sum(ep.tip_from_masks_frames for ep in eps)
    lid_frames = sum(1 for row in strip if row.lid_closed)
    contained_closed = sum(len(row.contained) for row in strip if row.lid_closed and row.contained)
    return {
        "schema": SCHEMA,
        "arm": arm,
        "tracks_dir": tracks_dir.as_posix(),
        "label": MODEL_OUTPUT,
        "params": params.to_record(),
        "hand_source": hand_source,
        "frames": len(strip),
        "lid_closed_frames": lid_frames,
        "contained_track_frames_while_lid_closed": contained_closed,
        "cycles_in_window": [list(c) for c in cycles],
        "containers": [v.to_record() for v in volumes],
        "proximity_targets": [v.to_record() for v in proximity_targets],
        "kinds": kinds,
        "centrifuge_cycles_vs_contained": list(cross),
    }


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def summary_markdown(summary: dict[str, Any]) -> str:
    p = summary["params"]
    lines = [
        f"# Events, arm ({summary['arm']})",
        "",
        f"**{summary['label']}.** Tracks from `{summary['tracks_dir']}`; "
        f"{summary['frames']} frames; hands from {summary['hand_source']}.",
        "",
        f"Hysteresis: enter at signed distance <= {p['enter_cm']} cm, exit at >= {p['exit_cm']} "
        f"cm, dwell {p['dwell_frames']} frames; `contained` judged for "
        + ", ".join(p["containable_classes"])
        + f" (and their groups); `held` within {p['held_radius_cm']} cm of a hand track and only "
        f"once the object has moved >= {p['held_min_motion_cm']} cm over the dwell window, exit at "
        f"+{p['held_exit_margin_cm']} cm; `proximity` margin {p['proximity_margin_cm']} cm "
        f"(+{p['tip_fallback_margin_cm']} when the tip is the track point), plate volume "
        f"{p['plate_above_cm']} cm above the plate; footprint scale {p['footprint_scale']}.",
        "",
        "## Volumes",
        "",
        "| volume | centre x, y cm | half-extent cm | z top .. bottom cm | height source |",
        "|---|---|---|---|---|",
    ]
    for v in [*summary["containers"], *summary["proximity_targets"]]:
        prov = v["provenance"]
        source = prov.get("height_source", prov.get("centre"))
        lines.append(
            f"| {v['name']} | {v['centre_xy_cm'][0]}, {v['centre_xy_cm'][1]} | {v['half_x_cm']} | "
            f"{v['z_top_cm']} .. {v['z_bottom_cm']} | {source} |"
        )
    lines += [
        "",
        "## Counts",
        "",
        "| kind | episodes | tracks | frames with any | duration median / p90 / max | "
        "end reasons | by target |",
        "|---|---|---|---|---|---|---|",
    ]
    for kind, k in summary["kinds"].items():
        d = k["duration_frames"]
        lines.append(
            f"| {kind} | {k['episodes']} | {k['tracks']} | {k['frames_with_any']} | "
            f"{_fmt(d['median'])} / {_fmt(d['p90'])} / {_fmt(d['max'])} | "
            + ", ".join(f"{a} {b}" for a, b in k["end_reasons"].items())
            + " | "
            + ", ".join(f"{a} {b}" for a, b in list(k["by_target"].items())[:6])
            + " |"
        )
    lines += ["", "By class: "]
    for kind, k in summary["kinds"].items():
        lines.append(
            f"- `{kind}`: " + (", ".join(f"{a} {b}" for a, b in k["by_class"].items()) or "none")
        )
    prox = summary["kinds"]["proximity"]
    lines += [
        "",
        f"Lid closed on {summary['lid_closed_frames']} frames; contained track-frames while the "
        f"lid is closed: {summary['contained_track_frames_while_lid_closed']}. Proximity tips "
        f"from masks on {prox.get('tip_from_masks_frames', 0)} episode frames.",
        "",
        "## Centrifuge cycles vs `contained` episodes",
        "",
        "| cycle (raw, closed) | frames | episodes overlapping +/-90 | entered before closure | "
        "ended while closed | same id through the cycle | started after opening | "
        "ended before opening with a successor id |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in summary["centrifuge_cycles_vs_contained"]:
        lines.append(
            f"| [{c['cycle'][0]}, {c['cycle'][1]}) | {c['closed_frames']} | {c['episodes']} | "
            f"{c['entered_before_closure']} | {c['ended_while_closed']} | "
            f"{c['same_id_after_opening']} | {c['started_after_opening']} | "
            f"{c['with_successor_after_opening']} |"
        )
    for c in summary["centrifuge_cycles_vs_contained"]:
        if not c["contained_episodes"]:
            continue
        lines += ["", f"Cycle [{c['cycle'][0]}, {c['cycle'][1]}):", ""]
        for e in c["contained_episodes"]:
            succ = ", ".join(
                f"{s['track_id']} at {s['frame_index']}"
                + (
                    f" (possibly_same_as {', '.join(s['possibly_same_as'])})"
                    if s["possibly_same_as"]
                    else ""
                )
                for s in e["successors_after_opening"]
            )
            lines.append(
                f"- `{e['track_id']}` ({e['object_class']}) contained "
                f"{e['start_frame']}..{e['end_frame']} ({e['end_reason']}), "
                f"{e['phase'].replace('_', ' ')}; inside while closed "
                f"{e['inside_while_closed_frames']} frames"
                + (f"; successors after opening: {succ}" if succ else "")
            )
    tips = summary.get("tip_events")
    if tips:
        lines += [
            "",
            "## Disposable tips (`--tip-events`)",
            "",
            f"Rule: {tips['rule']}.",
            "",
            "| kind | events | flips that met no volume | measured against |",
            "|---|---|---|---|",
            f"| tip_picked | {tips['counts']['tip_picked']} | "
            f"{tips['unmatched_flips']['tip_picked']} | the tracker's tip state and the rack "
            "volumes |",
            f"| tip_ejected | {tips['counts']['tip_ejected']} | "
            f"{tips['unmatched_flips']['tip_ejected']} | the tracker's tip state and the trash "
            "volume |",
            "",
            "Volumes from the rig's static points: "
            + (", ".join(tips["volumes_from_rig"]) or "none")
            + "; from median detector boxes: "
            + (", ".join(tips["volumes_from_boxes"]) or "none")
            + "; skipped: "
            + (", ".join(f"{k} ({v})" for k, v in tips["volumes_skipped"].items()) or "none")
            + f". Tracks with a tip state: {tips['tracks_with_a_tip_state']}.",
            "",
        ]
        if tips["events"]:
            lines += ["Events:", ""]
            for e in tips["events"]:
                lines.append(
                    f"- {e['kind']} `{e['track_id']}` ({e['object_class']}, {e['tip_class']}) at "
                    f"frame {e['frame_index']} in {e['target']} (from {e['from_state']})"
                )
            lines.append("")
    lines += [
        "",
        "A `contained` episode that ends `track_lost` while the lid is closed is the core "
        "tracker's coast timeout (30 frames) running out before the lid re-opens; the successor "
        "ids listed are the same-class births near the last position within 90 frames of the "
        "opening. With the tracker extensions' `contained` state the same id should persist; "
        "re-run with `--tracks-dir tracks-ext`.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- cli


def run(
    arm_dir: Path,
    clip_path: Path,
    rig_path: Path,
    output: Path,
    *,
    tracks_dir: str | None = None,
    trials_path: Path = DEFAULT_TRIALS,
    params: EventParams = EventParams(),
    arm: str | None = None,
    observations: Path | None = None,
    tip_events_on: bool = False,
    tip_lookback_frames: int = 15,
    tip_margin_cm: float = 5.0,
) -> dict[str, Any]:
    arm_dir = Path(arm_dir)
    clip: ClipWindow = load_clip(clip_path)
    rig = json.loads(Path(rig_path).read_text(encoding="utf-8"))
    config = read_camera_config(clip.camera_config)
    cams = cameras_from_config(config)
    tracks_path = resolve_tracks_dir(arm_dir, tracks_dir)
    rows = list(read_jsonl(tracks_path / "tracks.jsonl", Track3D))
    clip_doc = json.loads(Path(clip_path).read_text(encoding="utf-8"))
    targets = tuple(clip_doc.get("targets", ()))
    observations = (
        Path(observations) if observations is not None else arm_dir / "observations.jsonl"
    )
    if observations.is_dir():
        observations = observations / "observations.jsonl"
    rack_classes = sorted(
        {
            str(s["class"])
            for s in rig.get("static", ())
            if str(s["class"]).endswith(TIP_RACK_SUFFIX)
        }
        | {c for c in targets if c.endswith(TIP_RACK_SUFFIX)}
    )
    tip_volume_classes = [*rack_classes, TRASH_CLASS] if tip_events_on else []
    box_sizes = median_box_sizes(observations, [*clip.containers, PLATE_CLASS, *tip_volume_classes])
    volumes = container_volumes(rig, clip.containers, cams, box_sizes, scale=params.footprint_scale)
    proximity_targets: list[Volume] = []
    for name in params.proximity_targets:
        if name == PLATE_CLASS:
            plate = plate_volume(
                rows, cams, box_sizes, above_cm=params.plate_above_cm, scale=params.footprint_scale
            )
            if plate is not None:
                proximity_targets.append(plate)
        else:
            proximity_targets.extend(v for v in volumes if v.name == name)
    hands = hand_positions(rows, clip.probes)
    hand_source = "the arm's hand tracks"
    if not hands:
        hands = rig_hand_positions(rig)
        hand_source = "the rig's per-frame hand triangulations (no hand track in the arm)"
    side_cams = {v: c for v, c in cams.items() if v not in (TOP_DOWN_VIEW, FINEBIO_FPV_VIEW)}
    tip_keys = {
        (view, r.frame_index, slot)
        for r in rows
        if r.object_class.endswith(PIPETTE_SUFFIX)
        for view, slot in r.support_slots.items()
        if view in side_cams
    }
    tips = load_tip_pixels(observations, tip_keys, exclude_views=(TOP_DOWN_VIEW, FINEBIO_FPV_VIEW))
    trials = (
        json.loads(Path(trials_path).read_text(encoding="utf-8"))
        if Path(trials_path).is_file()
        else {}
    )
    intervals = lid_closed_intervals(trials, clip.trial)
    cycles = cycles_in_window(intervals, clip.start_frame, clip.end_frame_exclusive)
    gate = float((rig.get("gates") or {}).get("association_px", 30.0))
    frames = list(clip.frames)
    episodes, strip, events = detect_events(
        rows,
        containers=volumes,
        hands=hands,
        proximity_targets=proximity_targets,
        tips=tips,
        side_cams=side_cams,
        lid_intervals=intervals,
        frames=frames,
        targets=targets,
        params=params,
        gate_px=gate,
    )
    cross = cycle_cross_table(episodes, rows, cycles)
    tip_summary: dict[str, Any] | None = None
    if tip_events_on:
        from_rig = container_volumes(
            rig, tip_volume_classes, cams, box_sizes, scale=params.footprint_scale
        )
        have = {v.name for v in from_rig}
        missing = [c for c in tip_volume_classes if c not in have]
        from_boxes, skipped = box_volumes(
            missing, cams, median_box_centres(observations, missing), box_sizes, gate_px=gate
        )
        volumes_for_tips = [*from_rig, *from_boxes]
        racks = [v for v in volumes_for_tips if v.name.endswith(TIP_RACK_SUFFIX)]
        trash = [v for v in volumes_for_tips if v.name == TRASH_CLASS]
        tip_rows, tip_summary = tip_events(
            rows,
            racks=racks,
            trash=trash,
            lookback_frames=tip_lookback_frames,
            margin_cm=tip_margin_cm,
        )
        tip_summary["volumes_from_rig"] = sorted(have)
        tip_summary["volumes_from_boxes"] = [v.name for v in from_boxes]
        tip_summary["volumes_skipped"] = skipped
        events = sorted(
            [*events, *tip_rows],
            key=lambda e: (e.frame_index, e.track_id, e.payload.get("phase") == "end"),
        )
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(events, output / "events.jsonl")
    with (output / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for ep in episodes:
            handle.write(json.dumps(ep.to_record()) + "\n")
    write_jsonl(strip, output / "events_strip.jsonl", compact=True)
    summary = summarise(
        episodes,
        strip,
        arm=arm or arm_dir.name.split("-", 1)[0],
        params=params,
        volumes=volumes,
        proximity_targets=proximity_targets,
        cycles=cycles,
        cross=cross,
        tracks_dir=tracks_path,
        hand_source=hand_source,
    )
    summary["arm_dir"] = arm_dir.as_posix()
    summary["tip_pixels_loaded"] = len(tips)
    summary["tip_events"] = tip_summary
    (output / "events_summary.json").write_text(
        json.dumps(summary, indent=1) + "\n", encoding="utf-8"
    )
    (output / "events.md").write_text(summary_markdown(summary), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--arm-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="FineBio clip config JSON")
    parser.add_argument("--rig", type=Path, required=True, help="rig.json of the window")
    parser.add_argument("--trials", type=Path, default=DEFAULT_TRIALS)
    parser.add_argument("--tracks-dir", default=None)
    parser.add_argument("--arm", default=None)
    parser.add_argument("--output", type=Path, required=True)
    d = EventParams()
    parser.add_argument("--enter-cm", type=float, default=d.enter_cm)
    parser.add_argument("--exit-cm", type=float, default=d.exit_cm)
    parser.add_argument("--dwell-frames", type=int, default=d.dwell_frames)
    parser.add_argument("--held-radius-cm", type=float, default=d.held_radius_cm)
    parser.add_argument("--held-exit-margin-cm", type=float, default=d.held_exit_margin_cm)
    parser.add_argument("--proximity-margin-cm", type=float, default=d.proximity_margin_cm)
    parser.add_argument("--plate-above-cm", type=float, default=d.plate_above_cm)
    parser.add_argument("--tip-fallback-margin-cm", type=float, default=d.tip_fallback_margin_cm)
    parser.add_argument("--footprint-scale", type=float, default=d.footprint_scale)
    parser.add_argument(
        "--proximity-targets",
        default=",".join(d.proximity_targets),
        help="comma list of target volumes for the pipette tip (the plate, or container names)",
    )
    parser.add_argument(
        "--containable-classes",
        default=",".join(d.containable_classes),
        help="comma list of classes `contained` is judged for (their _group slots included)",
    )
    parser.add_argument("--held-min-motion-cm", type=float, default=d.held_min_motion_cm)
    parser.add_argument(
        "--observations",
        type=Path,
        default=None,
        help="observations.jsonl (or its directory); default <arm-dir>/observations.jsonl",
    )
    parser.add_argument(
        "--tip-events",
        action="store_true",
        help="Sep 29: tip_picked / tip_ejected from the line tracker's tip_attached state and "
        "the *_tip_rack / trash_can volumes",
    )
    parser.add_argument("--tip-lookback-frames", type=int, default=15)
    parser.add_argument("--tip-margin-cm", type=float, default=5.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    params = EventParams(
        enter_cm=args.enter_cm,
        exit_cm=args.exit_cm,
        dwell_frames=args.dwell_frames,
        held_radius_cm=args.held_radius_cm,
        held_exit_margin_cm=args.held_exit_margin_cm,
        proximity_margin_cm=args.proximity_margin_cm,
        plate_above_cm=args.plate_above_cm,
        tip_fallback_margin_cm=args.tip_fallback_margin_cm,
        footprint_scale=args.footprint_scale,
        proximity_targets=tuple(t for t in args.proximity_targets.split(",") if t),
        containable_classes=tuple(t for t in args.containable_classes.split(",") if t),
        held_min_motion_cm=args.held_min_motion_cm,
    )
    summary = run(
        args.arm_dir,
        args.config,
        args.rig,
        args.output,
        tracks_dir=args.tracks_dir,
        trials_path=args.trials,
        params=params,
        arm=args.arm,
        observations=args.observations,
        tip_events_on=args.tip_events,
        tip_lookback_frames=args.tip_lookback_frames,
        tip_margin_cm=args.tip_margin_cm,
    )
    kinds = summary["kinds"]
    tips = summary.get("tip_events") or {}
    print(
        f"arm {summary['arm']}: "
        + ", ".join(f"{k} {v['episodes']}" for k, v in kinds.items())
        + f" episodes; lid closed {summary['lid_closed_frames']} frames"
        + (f"; tips {tips['counts']}" if tips else "")
        + f" -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
