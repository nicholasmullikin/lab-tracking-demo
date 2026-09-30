"""`battle-finebio-tips`: tip clicks on about 30 frames, triangulated to 3D tip anchors.

The soft gate of `p3-tip-anchors`: I click the very end of each pipette's white tip cone in
two or three views on about 30 frames, the clicks are triangulated to a 3D tip, and every
tracker's tip is measured against it in centimetres. About 20 minutes of clicking::

    battle-finebio-tips prepare --trial P03_03_01 --observations <dir or jsonl> \\
        --clip-config configs/clips/finebio_P03_03_01_600-4200.json --frames 30 \\
        --output runs/finebio-tips-<trial>-<date>/           # CPU, reads ~180 proxy frames
    battle-finebio-tips serve --workspace runs/finebio-tips-<trial>-<date>/ \\
        [--record decisions.json] [--port 8767] [--tailscale | --bind 127.0.0.1]
    battle-finebio-tips score --workspace <workspace> --tracks <tracks.jsonl> [--tracks ...] \\
        --clip-config <clip config> --output <dir>
    battle-finebio-tips export --workspace <workspace> --record <decisions.json> \\
        --output docs/qa/<trial>-tip-clicks.human-record.json

* ``prepare`` chooses the frames from the detector rows, stratified over the window and over
  three pipette states: **rest** (no pipette box moved within half a second), **held** (a
  pipette box centre moved more than 2 cm/frame equivalent, the box's long side standing for
  the shared pipette length) and **low** (an active pipette's box bottom in the lowest fifth
  of its active frames in T4 or T5, or, when line tracks exist, a resolved tip under 5 cm
  above the bench). For every chosen frame and pipette it picks the two or three views where
  the SAM3 mask is largest and the tip end is inside the frame (the fpv only on valid-pose
  frames), writes a reduced whole frame and a 400 px zoom around the suggested tip end (the
  axis end farther from any hand box, else the lower end), and lists every cell in
  ``cells.json``.
* ``serve`` is a small page per frame: the view crops side by side, a faint marker at the
  suggested tip that a click replaces, and the keys below. Every change is written at once
  to the record (temporary file, then rename). The suggestion is never saved: a cell's
  ``tip_px`` comes from a click or stays null.
* ``score`` triangulates each frame's clicks per pipette (undistort + DLT, per-click
  reprojection residual; frames with a single clicked view are dropped, and so are frames
  whose clicks still disagree by more than the residual gate after one view is dropped),
  then for every tracks file finds that pipette's track on that frame (by
  ``support_slots``, else the nearest endpoint) and reports the tip error: the distance from
  ``endpoints_cm[0]`` when ``tip_resolved``, from the nearer endpoint when not (flagged), and
  from ``position_cm`` for a point track (a midpoint, so expected large, and named as such).
  A line track's error is also split along its axis (signed: positive when the anchor lies
  beyond the track's tip end, as it does when a disposable tip is attached and the mask stops
  at the body's cone) and across it, by state and by pipette class, with the fraction of
  anchors more than 2 cm beyond the end. Pipette-only IDF1 and the colour vote's agreement
  come from the identity keys (``1`` blue, ``2`` yellow, ``3`` red, ``4`` 8-channel) I press
  per pipette per frame, with the detector's class of the clicked cell as a named fallback;
  ``colour_identity`` is read from a ``tracks_colour.jsonl`` beside the tracks when present.
* ``export`` writes the no-pixel human record under ``docs/qa/``: clicks in full-frame
  pixels, hidden flags, identities, the decisions file's SHA-256 and, with ``--protocol``,
  a note on how I chose between a click and ``h`` in that session (what ``hidden`` means).

Claim boundary: the anchors are one person's clicks on the proxies' pixels at about 30
frames of one trial; the triangulation inherits the rig's residual (a few px). They rank
trackers against each other and read the pre-registered rule (median tip error under 2 cm);
they are not ground truth. FineBio is non-commercial research data: frames and crops stay
under ``runs/`` and never enter the repository.

Keys on a frame page: click sets the tip (and moves on), ``h`` hidden, ``x`` clear, ``u``
undo, ``j``/``k`` next/previous cell, ``]``/``[`` next/previous frame, ``1``-``4`` the
plunger colour seen (blue, yellow, red, 8-channel), ``0`` no identity, ``o`` whole frames.

Sep 29 (the re-click, after `finebio_tipseg` found half the first clicks on the plunger):
``prepare --no-marker`` draws no suggested marker (the crop shows the whole pipette and a
small arrow at the crop's edge, labelled "tracker's guess", points at the end the rows'
long-thin-tail rule names), ``--single-channel-only`` leaves the 8-channel out, and
``--event-frames 2180,2330 --event-quota 10 --event-class blue_pipette`` weights the frame
choice to a pipette's tip timeline. Keys ``t`` / ``b`` say a disposable tip IS attached /
the pipette is bare, per pipette per frame (``tip_attached_label`` on every view cell), and
``score`` reads them against the tracker's ``tip_attached`` and reports the 3D tip error on
the consistently labelled frames alone.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import cv2
import numpy as np

from . import fs_common
from .finebio_anchors import idf1
from .finebio_anchors_web import (
    SERVED_SUFFIXES,
    STYLE,
    _e,
    _html,
    _json,
    _script_json,
    default_author,
    resolve_bind,
    utc_now,
)
from .finebio_cameras import (
    Camera,
    cameras_from_config,
    fpv_camera_from_config,
    fpv_poses,
    read_camera_config,
)
from .finebio_slice import triangulate_pixels
from .muggled_calibration_web import DEFAULT_HOST, reachable_urls

SCHEMA = "battle-finebio-tips/1"
ANCHOR_KIND = "human_tip_click"
PIPETTE_CLASSES: tuple[str, ...] = (
    "blue_pipette",
    "yellow_pipette",
    "red_pipette",
    "8_channel_pipette",
)
IDENTITY_KEYS: dict[str, str] = {
    "1": "blue_pipette",
    "2": "yellow_pipette",
    "3": "red_pipette",
    "4": "8_channel_pipette",
}
IDENTITY_SHORT: dict[str, str] = {
    "blue_pipette": "blue",
    "yellow_pipette": "yellow",
    "red_pipette": "red",
    "8_channel_pipette": "8-channel",
}
COLOUR_TO_CLASS: dict[str, str] = {short: cls for cls, short in IDENTITY_SHORT.items()}
LINE_CLASS = "pipette"
HAND_CLASSES: tuple[str, ...] = ("left_hand", "right_hand")
STATES: tuple[str, ...] = ("rest", "held", "low")
# The order the states are served in when the frames are split (the thin pools first).
DEFAULT_STATE_ORDER: tuple[str, ...] = ("low", "held", "rest")
# Sep 29: a frame chosen for its place in a pipette's tip timeline (`--event-frames`).
EVENT_STATE = "event"
ALL_STATES: tuple[str, ...] = (*STATES, EVENT_STATE)
SINGLE_CHANNEL_CLASSES: tuple[str, ...] = ("blue_pipette", "yellow_pipette", "red_pipette")
TIP_LABEL_KEYS: dict[str, bool] = {"t": True, "b": False}
DEFAULT_EVENT_RADIUS_FRAMES = 60
# The whole-pipette crop (no marker): the mask box padded by this fraction of its long side,
# at least this many pixels, shown at most this wide.
PIPETTE_CROP_PAD = 0.3
PIPETTE_CROP_MIN_PAD_PX = 40
PIPETTE_CROP_MAX_PX = 720
FPV_VIEW = "fpv"
SAM3_SOURCES = ("sam3_decode", "sam3_video")
VIEW_ORDER: tuple[str, ...] = ("fpv", "T1", "T2", "T3", "T4", "T5")
DEFAULT_FRAMES = 30
DEFAULT_SPACING_FRAMES = 20
DEFAULT_MOVE_CM_PER_FRAME = 2.0
DEFAULT_LENGTH_CM = 23.0
DEFAULT_ACTIVE_WINDOW_FRAMES = 15
DEFAULT_LOW_FRACTION = 0.2
DEFAULT_LOW_TIP_HEIGHT_CM = 5.0
DEFAULT_LOW_VIEWS: tuple[str, ...] = ("T4", "T5")
DEFAULT_VIEWS_PER_CELL = 3
MIN_VIEWS = 2
ZOOM_SIZE = 400
ZOOM_NARROW_WIDTH_PX = 20.0
OVERVIEW_WIDTH = 960
EDGE_MARGIN_PX = 6.0
RESIDUAL_GATE_PX = 30.0
MATCH_GATE_CM = 15.0
THRESHOLD_MEDIAN_CM = 2.0
ATTACHED_ALONG_CM = 2.0
COLOUR_TRACKS_NAME = "tracks_colour.jsonl"
HISTORY_LIMIT = 500
DEFAULT_PORT = 8767
CELLS_NAME = "cells.json"
TEMPLATE_NAME = "decisions.template.json"
RECORD_NAME = "decisions.json"
DEFAULT_PIPETTES_CONFIG = Path("configs/finebio/pipettes.json")
CLASS_COLOURS_BGR: dict[str, tuple[int, int, int]] = {
    "blue_pipette": (255, 120, 0),
    "yellow_pipette": (0, 220, 255),
    "red_pipette": (60, 60, 255),
    "8_channel_pipette": (255, 0, 200),
}
CLAIM_BOUNDARY = (
    "Tip anchors are one person's clicks at the end of the white tip cone in two or three "
    "views on about 30 frames of one trial, triangulated through the rig; they inherit the "
    "rig's residual of a few pixels and the click's own scatter. They rank trackers against "
    "each other and read the pre-registered rule (median tip error under 2 cm); they are not "
    "ground truth and support no accuracy claim."
)
LICENCE_NOTE = (
    "FineBio is licensed for non-commercial research; frames and crops derived from it stay "
    "under runs/ and are never committed or redistributed. cells.json (frames, pixels of "
    "detector boxes and mask axes) and the human record (clicks, flags, identities) carry no "
    "image."
)

Box = tuple[float, float, float, float]
Point = tuple[float, float]
CellKey = tuple[int, str, int]


# --------------------------------------------------------------------------------------------
# inputs


def load_clip(path: Path) -> dict[str, Any]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("config_kind") != "finebio_clip_config":
        raise ValueError(f"{path} is not a finebio_clip_config")
    return doc


def observations_file(path: Path) -> Path:
    path = Path(path)
    return path / "observations.jsonl" if path.is_dir() else path


def read_rows(path: Path, classes: Iterable[str]) -> list[dict[str, Any]]:
    """The rows of `classes` as dicts; a substring filter before the JSON parse keeps an
    800k-row file to a couple of seconds."""
    keys = {f'"object_class":"{c}"' for c in classes}
    wanted = set(classes)
    rows: list[dict[str, Any]] = []
    with observations_file(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            if '"object_class":' in line and not any(k in line for k in keys):
                continue
            row = json.loads(line)
            if row.get("object_class") in wanted:
                rows.append(row)
    return rows


def pipette_length_cm(pipettes_config: Path | None) -> float:
    if pipettes_config is None or not Path(pipettes_config).is_file():
        return DEFAULT_LENGTH_CM
    doc = json.loads(Path(pipettes_config).read_text(encoding="utf-8"))
    value = doc.get("length_cm")
    return float(value) if value else DEFAULT_LENGTH_CM


# --------------------------------------------------------------------------------------------
# per-frame tables from the observation rows


@dataclass
class SelectionParams:
    frames: int = DEFAULT_FRAMES
    spacing_frames: int = DEFAULT_SPACING_FRAMES
    move_cm_per_frame: float = DEFAULT_MOVE_CM_PER_FRAME
    length_cm: float = DEFAULT_LENGTH_CM
    active_window_frames: int = DEFAULT_ACTIVE_WINDOW_FRAMES
    low_fraction: float = DEFAULT_LOW_FRACTION
    low_tip_height_cm: float = DEFAULT_LOW_TIP_HEIGHT_CM
    low_views: tuple[str, ...] = DEFAULT_LOW_VIEWS
    views_per_cell: int = DEFAULT_VIEWS_PER_CELL
    min_views: int = MIN_VIEWS
    edge_margin_px: float = EDGE_MARGIN_PX
    zoom_size: int = ZOOM_SIZE
    overview_width: int = OVERVIEW_WIDTH
    # On held and low frames only the active pipettes become cells: the ones at rest in the
    # stand are the same anchor the rest frames already give, and the budget is 20 minutes.
    active_only_when_moving: bool = True
    # A move counts only with a hand box on the pipette's box in some view.
    require_hand: bool = True
    # Line-track tip heights beyond this are triangulation failures, not heights.
    max_track_height_cm: float = 100.0
    # -- Sep 29, the re-click. `marker` False: no suggested tip, the crop shows the whole
    # pipette (both axis ends inside the frame) with an arrow at its edge for the rows'
    # tail-rule side. `single_channel_only` leaves the 8-channel out. `event_frames` (raw)
    # take `event_quota` of the frames, spread within `event_radius` of each, on frames where
    # `event_class` has a cell; the rest is split over `states` in that order.
    marker: bool = True
    single_channel_only: bool = False
    event_frames: tuple[int, ...] = ()
    event_radius: int = DEFAULT_EVENT_RADIUS_FRAMES
    event_class: str | None = None
    event_quota: int = 0
    states: tuple[str, ...] = DEFAULT_STATE_ORDER

    @property
    def classes(self) -> tuple[str, ...]:
        return SINGLE_CHANNEL_CLASSES if self.single_channel_only else PIPETTE_CLASSES

    def as_dict(self) -> dict[str, Any]:
        return {
            "frames": self.frames,
            "spacing_frames": self.spacing_frames,
            "move_cm_per_frame": self.move_cm_per_frame,
            "length_cm": self.length_cm,
            "active_window_frames": self.active_window_frames,
            "low_fraction": self.low_fraction,
            "low_tip_height_cm": self.low_tip_height_cm,
            "low_views": list(self.low_views),
            "views_per_cell": self.views_per_cell,
            "min_views": self.min_views,
            "edge_margin_px": self.edge_margin_px,
            "zoom_size": self.zoom_size,
            "overview_width": self.overview_width,
            "active_only_when_moving": self.active_only_when_moving,
            "require_hand": self.require_hand,
            "max_track_height_cm": self.max_track_height_cm,
            "marker": self.marker,
            "single_channel_only": self.single_channel_only,
            "classes": list(self.classes),
            "event_frames": list(self.event_frames),
            "event_radius": self.event_radius,
            "event_class": self.event_class,
            "event_quota": self.event_quota,
            "states": list(self.states),
        }


@dataclass
class Tables:
    """What the frame choice reads: the top detector box per class and view, the SAM3 row
    with a mask axis per (frame, view, class), the hand boxes and the fpv pose flag."""

    window: tuple[int, int]
    fixed_views: tuple[str, ...]
    top_boxes: dict[tuple[str, str], dict[int, Box]] = field(default_factory=dict)
    axis_rows: dict[tuple[int, str, str], dict[str, Any]] = field(default_factory=dict)
    hands: dict[tuple[int, str], list[Box]] = field(default_factory=dict)
    fpv_valid: dict[int, bool] = field(default_factory=dict)

    @property
    def views(self) -> tuple[str, ...]:
        seen = {view for _, view in self.top_boxes} | {view for _, view, _ in self.axis_rows}
        return tuple(v for v in VIEW_ORDER if v in seen) + tuple(
            sorted(v for v in seen if v not in VIEW_ORDER)
        )


def build_tables(
    rows: Iterable[Mapping[str, Any]], *, window: tuple[int, int], fixed_views: Sequence[str]
) -> Tables:
    tables = Tables(window=(int(window[0]), int(window[1])), fixed_views=tuple(fixed_views))
    for row in rows:
        frame = int(row["frame_index"])
        if not window[0] <= frame < window[1]:
            continue
        view = str(row["view"])
        cls = str(row["object_class"])
        if view == FPV_VIEW and row.get("pose_valid") is not None:
            tables.fpv_valid[frame] = bool(row["pose_valid"]) or tables.fpv_valid.get(frame, False)
        source = row.get("source")
        box = row.get("box_xyxy_px")
        if cls in HAND_CLASSES:
            if source == "detector" and box is not None:
                tables.hands.setdefault((frame, view), []).append(tuple(map(float, box)))
            continue
        if cls not in PIPETTE_CLASSES:
            continue
        if source == "detector":
            if box is not None and str(row.get("slot", "")).endswith("#0"):
                tables.top_boxes.setdefault((cls, view), {})[frame] = tuple(map(float, box))
        elif source in SAM3_SOURCES and row.get("mask_axis_px"):
            key = (frame, view, cls)
            current = tables.axis_rows.get(key)
            if current is None or (row.get("mask_area_px") or 0) > (
                current.get("mask_area_px") or 0
            ):
                tables.axis_rows[key] = dict(row)
    return tables


# --------------------------------------------------------------------------------------------
# pipette states per frame


def _centre(box: Box) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def move_threshold_px(box: Box, params: SelectionParams) -> float:
    """The pixel step that stands for `move_cm_per_frame`: the box's long side is taken as
    the shared pipette length (`length_cm`)."""
    long_side = max(box[2] - box[0], box[3] - box[1])
    return params.move_cm_per_frame / params.length_cm * long_side


def _overlaps(a: Box, b: Box) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def hand_on_box(tables: Tables, cls: str, frame: int) -> bool:
    """A hand box overlaps the class's top box in some view on this frame."""
    for view in tables.views:
        box = tables.top_boxes.get((cls, view), {}).get(frame)
        if box is None:
            continue
        if any(_overlaps(box, hand) for hand in tables.hands.get((frame, view), [])):
            return True
    return False


def moving_frames(tables: Tables, cls: str, params: SelectionParams) -> dict[int, list[str]]:
    """frame -> the fixed views in which the class's top box centre stepped more than the
    threshold since the previous frame; kept when at least two views agree (or every view
    that saw the box on both frames, when fewer than two did) and, with `require_hand`, a
    hand box touches the pipette's box in some view (a detector box that jumps between two
    pipettes at rest is not a move)."""
    moves: dict[int, list[str]] = defaultdict(list)
    pairs: Counter[int] = Counter()
    start, end = tables.window
    for view in tables.fixed_views:
        boxes = tables.top_boxes.get((cls, view), {})
        for frame in range(start + 1, end):
            before, now = boxes.get(frame - 1), boxes.get(frame)
            if before is None or now is None:
                continue
            pairs[frame] += 1
            step = math.dist(_centre(before), _centre(now))
            if step > move_threshold_px(now, params):
                moves[frame].append(view)
    return {
        f: views
        for f, views in moves.items()
        if len(views) >= min(2, pairs[f])
        and (not params.require_hand or hand_on_box(tables, cls, f))
    }


def expand_frames(frames: Iterable[int], radius: int, window: tuple[int, int]) -> set[int]:
    out: set[int] = set()
    for frame in frames:
        out.update(range(max(window[0], frame - radius), min(window[1], frame + radius + 1)))
    return out


def low_frames(
    tables: Tables,
    cls: str,
    active: set[int],
    params: SelectionParams,
    tip_heights: Mapping[tuple[int, str], float] | None = None,
) -> dict[int, str]:
    """frame -> why the class's tip counts as low: a resolved line-track tip under
    `low_tip_height_cm` when tracks are given, else a box bottom in `low_views` above the
    (1 - low_fraction) quantile of the class's active frames in that view."""
    out: dict[int, str] = {}
    if tip_heights is not None:
        for frame in active:
            height = tip_heights.get((frame, cls))
            if height is None or abs(height) > params.max_track_height_cm:
                continue
            if height < params.low_tip_height_cm:
                out[frame] = f"line-track tip {height:.1f} cm above the bench"
        return out
    for view in params.low_views:
        boxes = tables.top_boxes.get((cls, view), {})
        bottoms = {f: boxes[f][3] for f in active if f in boxes}
        if len(bottoms) < 5:
            continue
        cut = float(np.quantile(list(bottoms.values()), 1.0 - params.low_fraction))
        for frame, bottom in bottoms.items():
            if bottom > cut:
                out.setdefault(
                    frame,
                    f"box bottom {bottom:.0f} px in {view}, the lowest "
                    f"{params.low_fraction:.0%} of its active frames",
                )
    return out


@dataclass
class FrameState:
    raw_frame: int
    state: str  # rest | held | low | transition
    reason: str
    moving: dict[str, list[str]]
    active: list[str]
    low: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw_frame": self.raw_frame,
            "state": self.state,
            "reason": self.reason,
            "moving": self.moving,
            "active": self.active,
            "low": self.low,
        }


def classify_frames(
    tables: Tables,
    params: SelectionParams,
    tip_heights: Mapping[tuple[int, str], float] | None = None,
) -> dict[int, FrameState]:
    """Every frame of the window with one state: `low` beats `held` beats `rest`; a frame
    where a pipette is active but neither moving nor low is a `transition` and is not
    chosen."""
    start, end = tables.window
    moving = {cls: moving_frames(tables, cls, params) for cls in PIPETTE_CLASSES}
    active = {
        cls: expand_frames(moving[cls], params.active_window_frames, tables.window)
        for cls in PIPETTE_CLASSES
    }
    low = {
        cls: low_frames(tables, cls, active[cls], params, tip_heights) for cls in PIPETTE_CLASSES
    }
    out: dict[int, FrameState] = {}
    for frame in range(start, end):
        low_here = {cls: low[cls][frame] for cls in PIPETTE_CLASSES if frame in low[cls]}
        moving_here = {cls: moving[cls][frame] for cls in PIPETTE_CLASSES if frame in moving[cls]}
        active_here = [cls for cls in PIPETTE_CLASSES if frame in active[cls]]
        if low_here:
            cls, why = next(iter(low_here.items()))
            state, reason = "low", f"{cls}: {why}"
        elif moving_here:
            cls, views = next(iter(moving_here.items()))
            state = "held"
            reason = (
                f"{cls} box centre moved more than {params.move_cm_per_frame:g} cm/frame "
                f"equivalent in {', '.join(views)}"
            )
        elif not active_here:
            state = "rest"
            reason = f"no pipette box moved within {params.active_window_frames} frames"
        else:
            state = "transition"
            reason = f"{', '.join(active_here)} active but not moving or low on this frame"
        out[frame] = FrameState(frame, state, reason, moving_here, active_here, low_here)
    return out


def tip_heights_from_tracks(path: Path | None, window: tuple[int, int]) -> dict | None:
    """(frame, class) -> tip height above the bench (cm, `-z`) from a line tracks file's
    resolved tips; None when no file is given or none exists."""
    if path is None or not Path(path).is_file():
        return None
    heights: dict[tuple[int, str], float] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if '"endpoints_cm"' not in line or '"tip_resolved":true' not in line:
                continue
            row = json.loads(line)
            frame = int(row["frame_index"])
            if not window[0] <= frame < window[1] or not row.get("tip_resolved"):
                continue
            endpoints = row.get("endpoints_cm")
            if not endpoints:
                continue
            height = -float(endpoints[0][2])
            for cls in (
                row.get("colour_identity"),
                row.get("observed_class"),
                row.get("object_class"),
            ):
                if cls in PIPETTE_CLASSES:
                    heights[(frame, cls)] = min(height, heights.get((frame, cls), math.inf))
                    break
    return heights


# --------------------------------------------------------------------------------------------
# cells: which views, the suggested tip


def view_rank(view: str) -> tuple[int, str]:
    return (VIEW_ORDER.index(view) if view in VIEW_ORDER else len(VIEW_ORDER), view)


def point_box_distance(point: Point, box: Box) -> float:
    dx = max(box[0] - point[0], 0.0, point[0] - box[2])
    dy = max(box[1] - point[1], 0.0, point[1] - box[3])
    return math.hypot(dx, dy)


def suggested_tip(axis: Sequence[Sequence[float]], hands: Sequence[Box]) -> tuple[Point, str]:
    """The axis end that is the tip: the end farther from the nearest hand box when a hand
    is within half the axis length of an end (the hand holds the body by the plunger); when
    both ends touch a hand box, the end farther from the nearest hand-box centre (the
    plunger sits in the palm, the shaft leaves it); else the lower end in the image (a
    pipette at rest stands tip down)."""
    p0 = (float(axis[0][0]), float(axis[0][1]))
    p1 = (float(axis[1][0]), float(axis[1][1]))
    if hands:
        d0 = min(point_box_distance(p0, h) for h in hands)
        d1 = min(point_box_distance(p1, h) for h in hands)
        near = min(d0, d1) <= 0.5 * math.dist(p0, p1)
        if near and abs(d0 - d1) > 2.0:
            return (p1 if d1 > d0 else p0), "axis end farther from the nearest hand box"
        if near:
            c0 = min(math.dist(p0, _centre(h)) for h in hands)
            c1 = min(math.dist(p1, _centre(h)) for h in hands)
            if abs(c0 - c1) > 1e-6:
                return (p1 if c1 > c0 else p0), "axis end farther from the hand-box centre"
    return (p0 if p0[1] >= p1[1] else p1), "lower axis end in the image (no hand box near)"


def inside_image(point: Point, image_wh: Sequence[int], margin: float) -> bool:
    return margin <= point[0] <= image_wh[0] - margin and margin <= point[1] <= image_wh[1] - margin


def cell_classes(state: FrameState | None, params: SelectionParams) -> tuple[str, ...]:
    """The pipettes that become cells on a frame: every class (the single-channel ones with
    `single_channel_only`), or on a held or low frame only the active ones
    (`active_only_when_moving`); the event class always, on a frame near an event."""
    classes = params.classes
    if state is None or not params.active_only_when_moving or state.state not in ("held", "low"):
        return classes
    active = [c for c in classes if c in state.active or c in state.low or c in state.moving]
    return tuple(active) or classes


def tail_guess(row: Mapping[str, Any]) -> tuple[Point, str] | None:
    """Sep 29: the tracker's guess at the tip end from the row's long-thin-tail side
    (`tip_side`), at the mask's terminal centroid (`body_end_px`) when the row carries it,
    else the axis end; None when the tail rule did not decide."""
    side = row.get("tip_side")
    if side is None:
        return None
    ends = row.get("body_end_px") or row["mask_axis_px"]
    point = ends[int(side)]
    return (float(point[0]), float(point[1])), "tail"


def frame_cells(
    frame: int,
    tables: Tables,
    view_sizes: Mapping[str, Sequence[int]],
    params: SelectionParams,
    classes: Sequence[str] = PIPETTE_CLASSES,
) -> list[dict[str, Any]]:
    """The cells of one frame: per pipette class in `classes` with a mask axis in at least
    `min_views` views (fpv on valid-pose frames only, the tip end inside the frame; with no
    marker, both axis ends inside it, since the whole pipette has to show), the
    `views_per_cell` views with the largest mask. Without a marker a cell carries no
    `suggested_tip_px`; the rows' tail-rule end is kept apart as `guess_tip_px`."""
    cells: list[dict[str, Any]] = []
    for slot, cls in enumerate(PIPETTE_CLASSES):
        if cls not in classes or cls not in params.classes:
            continue
        candidates: list[dict[str, Any]] = []
        for view in tables.views:
            if view == FPV_VIEW and not tables.fpv_valid.get(frame, False):
                continue
            row = tables.axis_rows.get((frame, view, cls))
            if row is None or view not in view_sizes:
                continue
            hands = tables.hands.get((frame, view), [])
            axis = [list(map(float, p)) for p in row["mask_axis_px"]]
            guess = tail_guess(row)
            if params.marker:
                tip, rule = suggested_tip(row["mask_axis_px"], hands)
                if not inside_image(tip, view_sizes[view], params.edge_margin_px):
                    continue
                suggested: list[float] | None = [round(tip[0], 1), round(tip[1], 1)]
            else:
                if not all(
                    inside_image((p[0], p[1]), view_sizes[view], params.edge_margin_px)
                    for p in axis
                ):
                    continue
                suggested, rule = None, None
            candidates.append(
                {
                    "raw_frame": frame,
                    "view": view,
                    "slot": slot,
                    "class": cls,
                    "label": str(row["slot"]),
                    "image_size": [int(view_sizes[view][0]), int(view_sizes[view][1])],
                    "mask_area_px": int(row.get("mask_area_px") or 0),
                    "mask_width_px": row.get("mask_width_px"),
                    "mask_bbox_px": row.get("mask_bbox_px"),
                    "axis_px": axis,
                    "suggested_tip_px": suggested,
                    "tip_rule": rule,
                    "guess_tip_px": None
                    if guess is None
                    else [round(guess[0][0], 1), round(guess[0][1], 1)],
                    "guess_rule": None if guess is None else guess[1],
                    "tip_side": row.get("tip_side"),
                    "hand_boxes": len(hands),
                }
            )
        candidates.sort(key=lambda c: (-c["mask_area_px"], view_rank(c["view"])))
        chosen = candidates[: params.views_per_cell]
        if len(chosen) >= params.min_views:
            cells.extend(sorted(chosen, key=lambda c: view_rank(c["view"])))
    return cells


# --------------------------------------------------------------------------------------------
# frame choice


def pick_spread(pool: Sequence[int], count: int, taken: Sequence[int], spacing: int) -> list[int]:
    """`count` frames from `pool` at even quantiles, each the nearest pool frame at least
    `spacing` from every frame already taken; deterministic."""
    pool = sorted(pool)
    picked: list[int] = []
    if not pool or count <= 0:
        return picked
    if count == 1:
        targets = [pool[len(pool) // 2]]
    else:
        targets = [pool[int(i * (len(pool) - 1) / (count - 1) + 0.5)] for i in range(count)]
    for target in targets:
        free = [f for f in pool if all(abs(f - t) >= spacing for t in (*taken, *picked))]
        if free:
            picked.append(min(free, key=lambda f: (abs(f - target), f)))
    return picked


def select_frames(
    states: Mapping[int, FrameState],
    eligible: set[int],
    params: SelectionParams,
    event_pool: set[int] | None = None,
) -> list[dict[str, Any]]:
    """About `params.frames` eligible frames: first `event_quota` of them spread within
    `event_radius` of each of `params.event_frames` over `event_pool` (the eligible frames
    where the event class has a cell; the state is then `event`), then the rest split evenly
    over `params.states` in that order (`low`, `held`, `rest` by default, the shortfall of a
    thin pool topped up from the others), spread over the window with `spacing_frames`
    between any two."""
    order = tuple(s for s in params.states if s in STATES) or DEFAULT_STATE_ORDER
    pools = {state: sorted(f for f in eligible if states[f].state == state) for state in order}
    chosen: dict[int, str] = {}
    reasons: dict[int, str] = {}
    events = tuple(params.event_frames)
    if events and params.event_quota > 0:
        pool = sorted(event_pool if event_pool is not None else eligible)
        base, extra = divmod(params.event_quota, len(events))
        for i, centre in enumerate(events):
            near = [f for f in pool if abs(f - centre) <= params.event_radius]
            picks = pick_spread(
                near, base + (1 if i < extra else 0), list(chosen), params.spacing_frames
            )
            for frame in picks:
                chosen[frame] = EVENT_STATE
                reasons[frame] = (
                    f"{frame - centre:+d} frames from {centre} ({params.event_class or 'event'} "
                    f"tip timeline); {states[frame].state}: {states[frame].reason}"
                )
    remaining = params.frames - len(chosen)
    base, extra = divmod(max(remaining, 0), len(order))
    quota = {state: base + (1 if i < extra else 0) for i, state in enumerate(order)}
    for state in order:
        pool = [f for f in pools[state] if f not in chosen]
        for frame in pick_spread(pool, quota[state], list(chosen), params.spacing_frames):
            chosen[frame] = state
    shortfall = params.frames - len(chosen)
    for state in reversed(order):
        if shortfall <= 0:
            break
        pool = [f for f in pools[state] if f not in chosen]
        for frame in pick_spread(pool, shortfall, list(chosen), params.spacing_frames):
            chosen[frame] = state
        shortfall = params.frames - len(chosen)
    return [
        {"raw_frame": frame, "state": state, "reason": reasons.get(frame, states[frame].reason)}
        for frame, state in sorted(chosen.items())
    ]


# --------------------------------------------------------------------------------------------
# crops and the click mapping


def zoom_factor(mask_width_px: float | None) -> int:
    return 2 if mask_width_px is not None and float(mask_width_px) < ZOOM_NARROW_WIDTH_PX else 1


def zoom_geometry(
    tip: Sequence[float], *, size: int, zoom: int, image_wh: Sequence[int]
) -> dict[str, Any]:
    """A `size` px crop around `tip` at `zoom` (source side = size / zoom), clamped inside
    the image: ``offset`` (source px), ``scale`` and ``size`` (crop px)."""
    width, height = int(image_wh[0]), int(image_wh[1])
    side = max(1, int(round(size / zoom)))
    w, h = min(side, width), min(side, height)
    x0 = int(round(min(max(tip[0] - w / 2, 0), width - w)))
    y0 = int(round(min(max(tip[1] - h / 2, 0), height - h)))
    return {
        "offset": [x0, y0],
        "scale": float(zoom),
        "source_size": [w, h],
        "size": [w * zoom, h * zoom],
    }


def pipette_geometry(
    bbox: Sequence[float], *, image_wh: Sequence[int], max_px: int = PIPETTE_CROP_MAX_PX
) -> dict[str, Any]:
    """Sep 29 (no marker): a square crop around the whole mask box, padded by
    `PIPETTE_CROP_PAD` of its long side (at least `PIPETTE_CROP_MIN_PAD_PX`), clamped inside
    the image, shown at `max_px` at most (a small pipette is scaled up to 2x, a large one
    down); the same ``offset`` / ``scale`` / ``size`` shape as `zoom_geometry`."""
    width, height = int(image_wh[0]), int(image_wh[1])
    x0, y0, x1, y1 = (float(v) for v in bbox)
    long_side = max(x1 - x0, y1 - y0)
    pad = max(PIPETTE_CROP_PAD * long_side, PIPETTE_CROP_MIN_PAD_PX)
    side = int(round(min(long_side + 2 * pad, max(width, height))))
    w, h = min(side, width), min(side, height)
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    ox = int(round(min(max(cx - w / 2, 0), width - w)))
    oy = int(round(min(max(cy - h / 2, 0), height - h)))
    scale = min(2.0, max_px / max(w, h))
    scale = float(np.floor(scale * 4) / 4) if scale < 1.0 else min(2.0, float(int(scale)))
    scale = max(scale, 0.25)
    return {
        "offset": [ox, oy],
        "scale": scale,
        "source_size": [w, h],
        "size": [int(round(w * scale)), int(round(h * scale))],
    }


def overview_geometry(image_wh: Sequence[int], width: int) -> dict[str, Any]:
    scale = min(1.0, width / int(image_wh[0]))
    return {
        "offset": [0, 0],
        "scale": scale,
        "source_size": [int(image_wh[0]), int(image_wh[1])],
        "size": [int(round(image_wh[0] * scale)), int(round(image_wh[1] * scale))],
    }


def full_to_crop(point: Sequence[float], crop: Mapping[str, Any]) -> Point:
    return (
        (float(point[0]) - crop["offset"][0]) * crop["scale"],
        (float(point[1]) - crop["offset"][1]) * crop["scale"],
    )


def crop_to_full(point: Sequence[float], crop: Mapping[str, Any]) -> Point:
    return (
        float(point[0]) / crop["scale"] + crop["offset"][0],
        float(point[1]) / crop["scale"] + crop["offset"][1],
    )


def crop_file(raw_frame: int, view: str, slot: int | None) -> str:
    if slot is None:
        return f"crops/f{raw_frame:06d}_{view}_full.jpg"
    return f"crops/f{raw_frame:06d}_{view}_s{slot:02d}_zoom.jpg"


class ProxyFrames:
    """``(view, raw frame) -> BGR image`` over the window proxies, one capture per view, a
    seek per request (the frames are few and far apart)."""

    def __init__(self, proxies: Mapping[str, Path], frame_index_offset: int) -> None:
        self.proxies = {v: Path(p) for v, p in proxies.items()}
        self.offset = int(frame_index_offset)
        self._caps: dict[str, cv2.VideoCapture] = {}
        self.reads = 0

    def __call__(self, view: str, raw_frame: int) -> np.ndarray | None:
        cap = self._caps.get(view)
        if cap is None:
            path = self.proxies.get(view)
            if path is None or not path.is_file():
                return None
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                return None
            self._caps[view] = cap
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(raw_frame) - self.offset)
        ok, image = cap.read()
        self.reads += 1
        return image if ok else None

    def close(self) -> None:
        for cap in self._caps.values():
            cap.release()
        self._caps.clear()


def render_crops(
    cells: Sequence[dict[str, Any]],
    frames: Callable[[str, int], np.ndarray | None],
    *,
    output: Path,
    params: SelectionParams,
) -> None:
    """Per (frame, view) one reduced whole frame with every cell's mask box drawn thin, per
    cell one clean zoom around the suggested tip; the geometry is written into each cell
    under ``crops`` so a click maps back to full-frame pixels."""
    by_frame_view: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        by_frame_view[(int(cell["raw_frame"]), str(cell["view"]))].append(cell)
    (output / "crops").mkdir(parents=True, exist_ok=True)
    for (raw, view), group in sorted(by_frame_view.items()):
        image = frames(view, raw)
        if image is None:
            raise RuntimeError(f"{view}: could not read raw frame {raw}")
        height, width = image.shape[:2]
        overview = overview_geometry((width, height), params.overview_width)
        drawn = image.copy()
        for cell in group:
            box = cell.get("mask_bbox_px")
            colour = CLASS_COLOURS_BGR.get(cell["class"], (200, 200, 200))
            if box is not None:
                cv2.rectangle(
                    drawn, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), colour, 2
                )
                cv2.putText(
                    drawn,
                    f"{cell['slot']} {IDENTITY_SHORT.get(cell['class'], cell['class'])}",
                    (int(box[0]), max(24, int(box[1]) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    colour,
                    2,
                    cv2.LINE_AA,
                )
        small = cv2.resize(drawn, tuple(overview["size"]), interpolation=cv2.INTER_AREA)
        full_uri = crop_file(raw, view, None)
        cv2.imwrite(str(output / full_uri), small, [cv2.IMWRITE_JPEG_QUALITY, 80])
        for cell in group:
            if params.marker and cell.get("suggested_tip_px") is not None:
                zoom = zoom_factor(cell.get("mask_width_px"))
                geometry = zoom_geometry(
                    cell["suggested_tip_px"],
                    size=params.zoom_size,
                    zoom=zoom,
                    image_wh=(width, height),
                )
            else:
                # No marker: the whole pipette, so the plunger button and the far end show.
                bbox = cell.get("mask_bbox_px") or [
                    min(p[0] for p in cell["axis_px"]),
                    min(p[1] for p in cell["axis_px"]),
                    max(p[0] for p in cell["axis_px"]),
                    max(p[1] for p in cell["axis_px"]),
                ]
                geometry = pipette_geometry(bbox, image_wh=(width, height))
            x0, y0 = geometry["offset"]
            w, h = geometry["source_size"]
            crop = image[y0 : y0 + h, x0 : x0 + w]
            if tuple(geometry["size"]) != (w, h):
                interpolation = cv2.INTER_CUBIC if geometry["scale"] > 1 else cv2.INTER_AREA
                crop = cv2.resize(crop, tuple(geometry["size"]), interpolation=interpolation)
            zoom_uri = crop_file(raw, view, int(cell["slot"]))
            cv2.imwrite(str(output / zoom_uri), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            cell["crops"] = {
                "zoom": {"uri": zoom_uri, **geometry},
                "full": {"uri": full_uri, **overview},
            }


# --------------------------------------------------------------------------------------------
# the workspace documents


def cell_key(cell: Mapping[str, Any]) -> CellKey:
    return (int(cell["raw_frame"]), str(cell["view"]), int(cell["slot"]))


def frame_summary(
    frames: Sequence[dict[str, Any]], cells: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """The chosen frames with, per pipette slot, the views that became cells."""
    by_frame: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    for cell in cells:
        entry = by_frame[int(cell["raw_frame"])].setdefault(
            int(cell["slot"]), {"slot": int(cell["slot"]), "class": cell["class"], "views": []}
        )
        entry["views"].append(cell["view"])
    out = []
    for entry in frames:
        slots = by_frame.get(int(entry["raw_frame"]), {})
        out.append(
            {
                **entry,
                "slots": [slots[s] for s in sorted(slots)],
                "cells": sum(len(s["views"]) for s in slots.values()),
            }
        )
    return out


def decisions_template(workspace: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA}/decisions",
        "workspace": workspace.get("workspace_dir"),
        "trial": workspace["trial"],
        "author": None,
        "reviewed_at": None,
        "how": (
            "One row per (frame, view, pipette). tip_px: the full-frame pixel of the very end of "
            "the pipette, the end away from the coloured plunger button, where a disposable "
            "tip's point would be, set by a click in battle-finebio-tips serve (no suggestion "
            "is ever copied here); hidden: true when that end is not visible in this view; "
            "instance_identity: the pipette colour seen on the plunger (keys 1-4), the same "
            "value on every view of that pipette on that frame; tip_attached_label: true when "
            "a disposable tip IS attached (key t), false when the pipette is bare (key b), null "
            "when not said, the same value on every view of that pipette on that frame; note: "
            "free text."
        ),
        "identity_values": [*PIPETTE_CLASSES, None],
        "tip_attached_label_values": [True, False, None],
        "cells": [
            {
                "raw_frame": cell["raw_frame"],
                "proxy_frame": cell["proxy_frame"],
                "view": cell["view"],
                "slot": cell["slot"],
                "class": cell["class"],
                "state": cell["state"],
                "tip_px": None,
                "clicked_in": None,
                "hidden": False,
                "instance_identity": None,
                "tip_attached_label": None,
                "note": "",
            }
            for cell in workspace["cells"]
        ],
        "claim_boundary": CLAIM_BOUNDARY,
        "licence": LICENCE_NOTE,
    }


def workspace_readme(workspace: Mapping[str, Any]) -> str:
    counts = workspace["counts"]
    by_state = counts.get("frames_by_state", {})
    states = ", ".join(f"{by_state.get(s, 0)} {s}" for s in ALL_STATES if by_state.get(s))
    selection = workspace.get("selection", {})
    marker = selection.get("marker", True)
    minutes = "20" if marker else "15"
    lines = [
        f"# Tip clicks, {workspace['trial']} (p3-tip-anchors, soft gate)",
        "",
        f"**{counts['frames']} frames, {counts['cells']} cells, about {minutes} minutes of "
        f"clicking.** The frames are {states}. Each pipette appears in two or three views per "
        "frame, so the clicks triangulate to one 3D tip per pipette per frame. Nothing here "
        "is a label until `decisions.json` exists.",
        "",
        "## What to do",
        "",
        "Start the pages and open the URL it prints:",
        "",
        "```bash",
        "uv run battle-finebio-tips serve --workspace <this directory> --tailscale",
        "```",
        "",
    ]
    if marker:
        lines += [
            "On every cell, click the very end of the white tip cone. The faint dashed marker "
            "is the detector's guess and is never saved; your click replaces it and moves to "
            "the next cell.",
            "",
            "- If the tip is hidden in this view, press `h`.",
        ]
    else:
        lines += [
            "On every cell, click the very end of the pipette: the end away from the coloured "
            "plunger button, where a disposable tip's point would be. If a tip is on, click "
            "the point of the tip; if the pipette is bare, click the end of the cone. There is "
            "no suggested marker. The small arrow at the edge of the crop, labelled "
            '"tracker\'s guess", points at the end the tracker would call the tip; it is a '
            "guess, not a hint to accept, and is never saved. The click moves to the next cell.",
            "",
            "- If that end is hidden in this view, press `h`.",
            "- Once per pipette per frame, press `t` if a disposable tip IS attached and `b` "
            "if the pipette is bare. It is saved on every view of that pipette on that frame; "
            "`-` clears it.",
        ]
    lines += [
        "- If the crop shows two pipettes, pick the one whose plunger colour matches the class "
        "in the caption.",
        "- If two cells on a frame show the same pipette (the detector gave one mask two "
        "classes), click its end in both and press the colour you see on both. The identity "
        "then records which class was wrong.",
        "- Press `1`, `2`, `3` or `4` for the plunger colour you see (blue, yellow, red, "
        "8-channel). It is saved on every view of that pipette on that frame.",
        "- `u` undoes the last save, `x` clears a cell, `o` shows the whole frames, `j` and `k` "
        "move between cells, `]` and `[` between frames.",
        "",
        "Then score whichever tracks exist:",
    ]
    lines += [
        "",
        "```bash",
        "uv run battle-finebio-tips score --workspace <this directory> \\",
        "  --tracks <run>/tracks-lines/tracks.jsonl --tracks <run>/tracks-ext/tracks.jsonl \\",
        f"  --clip-config {workspace['clip_config']['uri']} --output <this directory>/scoreboard",
        "uv run battle-finebio-tips export --workspace <this directory> \\",
        "  --record <this directory>/decisions.json \\",
        f"  --output docs/qa/finebio-{workspace['trial']}-tip-clicks.human-record.json",
        "```",
        "",
        "## Files",
        "",
        "- `cells.json`: the chosen frames with their state and reason, and every cell with its "
        "views, mask axis, "
        + (
            "suggested tip"
            if marker
            else "the tracker's guess (kept apart, never shown as a marker)"
        )
        + " and crop geometry.",
        "- `crops/f<raw>_<view>_full.jpg`: the whole frame at reduced size with the pipette boxes.",
        "- `crops/f<raw>_<view>_s<slot>_zoom.jpg`: "
        + (
            "400 px around the suggested tip, clean pixels."
            if marker
            else "the whole pipette with a margin, clean pixels."
        ),
        "- `decisions.template.json`: the empty record; `serve` creates `decisions.json` from it.",
        "",
        "## Claim boundary",
        "",
        CLAIM_BOUNDARY,
        "",
        LICENCE_NOTE,
        "",
        f"Evidence: `{workspace.get('observations', {}).get('uri', '')}`, plan todo "
        "`p3-tip-anchors`.",
        "",
    ]
    return "\n".join(lines)


def load_workspace(path: Path) -> dict[str, Any]:
    doc = json.loads((Path(path) / CELLS_NAME).read_text(encoding="utf-8"))
    if doc.get("schema") != f"{SCHEMA}/workspace":
        raise ValueError(f"{path} is not a battle-finebio-tips workspace")
    return doc


def build_workspace(
    *,
    clip: Mapping[str, Any],
    clip_path: Path,
    observations: Path,
    tables: Tables,
    params: SelectionParams,
    output: Path,
    frames: Callable[[str, int], np.ndarray | None],
    repository_root: Path,
    tracks: Path | None = None,
) -> dict[str, Any]:
    """Classify, choose, cut and write: `cells.json`, the crops, the template and the README."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    offset = int(clip.get("frame_index_offset", clip["window"]["start_frame"]))
    view_sizes = {v: (int(w), int(h)) for v, (w, h) in clip["view_sizes"].items()}
    tip_heights = tip_heights_from_tracks(tracks, tables.window)
    states = classify_frames(tables, params, tip_heights)
    cells_by_frame = {
        f: frame_cells(f, tables, view_sizes, params, cell_classes(states.get(f), params))
        for f in range(*tables.window)
    }
    eligible = {f for f, cells in cells_by_frame.items() if cells}
    event_pool: set[int] | None = None
    if params.event_frames:
        # Near an event the event class is a cell whatever the frame's state says: the frame
        # is chosen for that pipette.
        event_cells = {
            f: frame_cells(f, tables, view_sizes, params, (params.event_class,))
            for centre in params.event_frames
            for f in range(
                max(tables.window[0], centre - params.event_radius),
                min(tables.window[1], centre + params.event_radius + 1),
            )
            if params.event_class is not None
        }
        event_pool = {f for f, cells in event_cells.items() if cells}
        for f in event_pool:
            others = [c for c in cells_by_frame[f] if c["class"] != params.event_class]
            cells_by_frame[f] = sorted(others + event_cells[f], key=lambda c: c["slot"])
        eligible |= event_pool
    chosen = select_frames(states, eligible, params, event_pool)
    cells: list[dict[str, Any]] = []
    for entry in chosen:
        for cell in cells_by_frame[entry["raw_frame"]]:
            cells.append(
                {**cell, "proxy_frame": entry["raw_frame"] - offset, "state": entry["state"]}
            )
    render_crops(cells, frames, output=output, params=params)
    for entry in chosen:
        entry["proxy_frame"] = entry["raw_frame"] - offset
    state_counts = Counter(s.state for s in states.values())
    workspace = {
        "schema": f"{SCHEMA}/workspace",
        "anchor_kind": ANCHOR_KIND,
        "trial": clip["trial"],
        "workspace_dir": str(output.resolve()),
        "clip_config": {
            "uri": fs_common.relative_uri(Path(clip_path), repository_root),
            "sha256": fs_common.sha256_file(Path(clip_path)),
        },
        "observations": {
            "uri": fs_common.relative_uri(observations_file(observations), repository_root),
            "sha256": fs_common.sha256_file(observations_file(observations)),
        },
        "tracks_for_low_state": (
            fs_common.relative_uri(Path(tracks), repository_root)
            if tip_heights is not None and tracks is not None
            else None
        ),
        "window": {
            "start": tables.window[0],
            "end": tables.window[1],
            "frame_index_offset": offset,
        },
        "frame_convention": (
            "raw = shipped video frame index; proxy frame = raw - frame_index_offset; every "
            "pixel is a raw-video pixel of the view"
        ),
        "view_sizes": {v: list(s) for v, s in view_sizes.items()},
        "identity_keys": IDENTITY_KEYS,
        "tip_label_keys": TIP_LABEL_KEYS,
        "selection": {
            **params.as_dict(),
            "states": {
                "rest": "no pipette box moved within active_window_frames",
                "held": (
                    "a pipette's top detector box centre stepped more than move_cm_per_frame / "
                    "length_cm x the box's long side since the previous frame, in two fixed "
                    "views, with a hand box on the pipette's box in some view (require_hand)"
                ),
                "low": (
                    "an active pipette's box bottom above the (1 - low_fraction) quantile of its "
                    "active frames in low_views; with line tracks, a resolved tip under "
                    "low_tip_height_cm (heights beyond max_track_height_cm ignored); active = "
                    "moved within active_window_frames"
                ),
                "low_from_tracks": tip_heights is not None,
            },
            "views": (
                "per pipette the views_per_cell views with the largest SAM3 mask whose suggested "
                "tip end is edge_margin_px inside the frame; fpv on valid-pose frames only; a "
                "pipette needs min_views views to be a cell; on held and low frames only the "
                "active pipettes are cells when active_only_when_moving"
            ),
            "suggested_tip": (
                (
                    "the mask-axis end farther from the nearest hand box on that frame and view, "
                    "else the lower end in the image; a suggestion only, never a decision"
                )
                if params.marker
                else (
                    "none: no marker is drawn; guess_tip_px is the rows' long-thin-tail end "
                    "(tip_side, at the terminal centroid), shown as an arrow at the crop's edge "
                    "labelled tracker's guess, never saved"
                )
            ),
            "window_states": dict(state_counts),
            "eligible_frames": len(eligible),
            "eligible_by_state": {
                s: sum(1 for f in eligible if states[f].state == s) for s in (*STATES, "transition")
            },
            "event_pool_frames": None if event_pool is None else len(event_pool),
        },
        "frames": frame_summary(chosen, cells),
        "cells": cells,
        "counts": {
            "frames": len(chosen),
            "frames_by_state": dict(Counter(e["state"] for e in chosen)),
            "cells": len(cells),
            "cells_by_state": dict(Counter(c["state"] for c in cells)),
            "cells_by_view": dict(Counter(c["view"] for c in cells)),
            "cells_by_class": dict(Counter(c["class"] for c in cells)),
            "pipette_frames": len({(c["raw_frame"], c["slot"]) for c in cells}),
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "licence": LICENCE_NOTE,
    }
    fs_common.write_json(output / CELLS_NAME, workspace)
    fs_common.write_json(output / TEMPLATE_NAME, decisions_template(workspace))
    (output / "README.md").write_text(workspace_readme(workspace), encoding="utf-8")
    return workspace


def run_prepare(args: argparse.Namespace) -> dict[str, Any]:
    root = Path.cwd().resolve()
    clip = load_clip(args.clip_config)
    if args.trial and clip["trial"] != args.trial:
        raise ValueError(f"--trial {args.trial} but the clip config is for {clip['trial']}")
    window = (int(clip["window"]["start_frame"]), int(clip["window"]["end_frame_exclusive"]))
    event_frames = tuple(int(f) for f in (args.event_frames or "").split(",") if f.strip())
    states = (
        tuple(s.strip() for s in (args.states or "").split(",") if s.strip()) or DEFAULT_STATE_ORDER
    )
    unknown = [s for s in states if s not in STATES]
    if unknown:
        raise ValueError(f"--states names {unknown}; choose from {list(STATES)}")
    if event_frames and args.event_class not in PIPETTE_CLASSES:
        raise ValueError(f"--event-frames needs --event-class in {list(PIPETTE_CLASSES)}")
    params = SelectionParams(
        frames=args.frames,
        spacing_frames=args.spacing,
        move_cm_per_frame=args.move_cm_per_frame,
        length_cm=pipette_length_cm(args.pipettes),
        views_per_cell=args.views_per_cell,
        active_only_when_moving=not args.all_pipettes,
        marker=not args.no_marker,
        single_channel_only=args.single_channel_only,
        event_frames=event_frames,
        event_radius=args.event_radius,
        event_class=args.event_class if event_frames else None,
        event_quota=args.event_quota if event_frames else 0,
        states=states,
    )
    rows = read_rows(args.observations, (*PIPETTE_CLASSES, *HAND_CLASSES))
    tables = build_tables(rows, window=window, fixed_views=clip["fixed_views"])
    offset = int(clip.get("frame_index_offset", window[0]))
    proxies = {v: root / uri for v, uri in clip["proxies"].items()}
    frames = ProxyFrames(proxies, offset)
    try:
        workspace = build_workspace(
            clip=clip,
            clip_path=Path(args.clip_config),
            observations=Path(args.observations),
            tables=tables,
            params=params,
            output=Path(args.output),
            frames=frames,
            repository_root=root,
            tracks=args.tracks,
        )
    finally:
        frames.close()
    for entry in workspace["frames"]:
        views = "; ".join(
            f"{IDENTITY_SHORT.get(s['class'], s['class'])}: {','.join(s['views'])}"
            for s in entry["slots"]
        )
        print(
            f"  raw {entry['raw_frame']} (proxy {entry['proxy_frame']}) {entry['state']:5s} "
            f"{entry['cells']:2d} cells  {views}"
        )
    print(json.dumps(workspace["counts"]))
    print(f"{frames.reads} proxy frames read -> {args.output}")
    return workspace


# --------------------------------------------------------------------------------------------
# the record


def validate_identity(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if value in IDENTITY_KEYS:
        return IDENTITY_KEYS[str(value)]
    if value in PIPETTE_CLASSES:
        return str(value)
    raise ValueError(
        f"instance_identity must be one of {list(PIPETTE_CLASSES)} or null; got {value!r}"
    )


def validate_tip_label(value: Any) -> bool | None:
    """`t` / true -> True (a tip is attached), `b` / false -> False (bare), null / "" -> None."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value in TIP_LABEL_KEYS:
        return TIP_LABEL_KEYS[value]
    raise ValueError("tip_attached_label must be true (t), false (b) or null")


def validate_tip(value: Any, image_size: Sequence[int]) -> list[float]:
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError, IndexError):
        raise ValueError("tip_px must be [x, y]") from None
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError("tip_px must be finite")
    if not (0 <= x <= image_size[0] and 0 <= y <= image_size[1]):
        raise ValueError(f"tip_px {[x, y]} is outside the {image_size[0]}x{image_size[1]} frame")
    return [round(x, 1), round(y, 1)]


def decided(cell: Mapping[str, Any]) -> bool:
    return cell.get("tip_px") is not None or bool(cell.get("hidden"))


class TipRecord:
    """Owner of the decisions file: created from the template when missing, every change
    validated against the workspace's cells and written whole and atomically, with an undo
    history for the session."""

    def __init__(
        self,
        *,
        workspace_dir: Path,
        workspace: Mapping[str, Any],
        path: Path,
        author: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.lock = threading.RLock()
        self.workspace_cells = {cell_key(c): c for c in workspace["cells"]}
        template = decisions_template(workspace)
        template_cells = {cell_key(c): c for c in template["cells"]}
        if self.path.is_file():
            self.doc = load_record(self.path)
            created = False
        else:
            self.doc = template
            created = True
        self.doc.setdefault("cells", [])
        present = {cell_key(c) for c in self.doc["cells"]}
        added = 0
        for key, cell in template_cells.items():
            if key not in present:
                self.doc["cells"].append(dict(cell))
                added += 1
        self.cells: dict[CellKey, dict[str, Any]] = {cell_key(c): c for c in self.doc["cells"]}
        self.history: list[list[tuple[CellKey, dict[str, Any]]]] = []
        author_changed = bool(author) and self.doc.get("author") != author
        if author_changed:
            self.doc["author"] = author
        if created or added or author_changed:
            self._save()

    def _save(self) -> None:
        self.doc["updated_at"] = utc_now()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fs_common.write_json(self.path, self.doc, atomic=True)

    def cell(self, key: CellKey) -> dict[str, Any]:
        try:
            return self.cells[key]
        except KeyError:
            raise KeyError(f"no cell for frame {key[0]} {key[1]} slot {key[2]}") from None

    def _remember(self, keys: Iterable[CellKey]) -> None:
        self.history.append([(k, dict(self.cells[k])) for k in keys])
        del self.history[:-HISTORY_LIMIT]

    def update(self, key: CellKey, changes: Mapping[str, Any]) -> dict[str, Any]:
        """Apply `tip_px` (a full-frame pixel or null), `click` (crop, x, y mapped through the
        cell's crop geometry), `hidden`, `clear`, `note`; a tip clears hidden and hidden
        clears the tip."""
        allowed = {"tip_px", "click", "hidden", "clear", "note", "clicked_in"}
        unknown = sorted(set(changes) - allowed)
        if unknown:
            raise ValueError(f"unknown cell fields {unknown}; editable: {sorted(allowed)}")
        if not changes:
            raise ValueError("nothing to change")
        with self.lock:
            cell = self.cell(key)
            workspace_cell = self.workspace_cells[key]
            # Validate on a copy: a refused change leaves the cell, the file and the undo
            # history exactly as they were.
            updated = dict(cell)
            if changes.get("clear"):
                updated["tip_px"] = None
                updated["clicked_in"] = None
                updated["hidden"] = False
            if "click" in changes:
                click = changes["click"]
                crops = workspace_cell.get("crops", {})
                if not isinstance(click, Mapping) or click.get("crop") not in crops:
                    raise ValueError(f"click needs crop in {sorted(crops)}, x and y")
                full = crop_to_full((click["x"], click["y"]), crops[click["crop"]])
                updated["tip_px"] = validate_tip(full, workspace_cell["image_size"])
                updated["clicked_in"] = str(click["crop"])
                updated["hidden"] = False
            if "tip_px" in changes:
                if changes["tip_px"] is None:
                    updated["tip_px"] = None
                    updated["clicked_in"] = None
                else:
                    updated["tip_px"] = validate_tip(
                        changes["tip_px"], workspace_cell["image_size"]
                    )
                    updated["clicked_in"] = str(changes.get("clicked_in") or "full")
                    updated["hidden"] = False
            if "hidden" in changes:
                if not isinstance(changes["hidden"], bool):
                    raise ValueError("hidden must be true or false")
                updated["hidden"] = changes["hidden"]
                if updated["hidden"]:
                    updated["tip_px"] = None
                    updated["clicked_in"] = None
            if "note" in changes:
                note = changes["note"]
                if note is not None and not isinstance(note, str):
                    raise ValueError("note must be a string")
                updated["note"] = " ".join((note or "").split())
            self._remember([key])
            cell.update(updated)
            self._save()
            return dict(cell)

    def set_identity(self, raw_frame: int, slot: int, value: Any) -> list[dict[str, Any]]:
        """The plunger colour seen, on every view of that pipette on that frame."""
        identity = validate_identity(value)
        return self._set_on_slot(raw_frame, slot, "instance_identity", identity)

    def set_tip_label(self, raw_frame: int, slot: int, value: Any) -> list[dict[str, Any]]:
        """Sep 29: whether a disposable tip is attached (key `t`, true), the pipette is bare
        (`b`, false) or nothing is said (null), on every view of that pipette on that frame."""
        return self._set_on_slot(raw_frame, slot, "tip_attached_label", validate_tip_label(value))

    def _set_on_slot(
        self, raw_frame: int, slot: int, name: str, value: Any
    ) -> list[dict[str, Any]]:
        with self.lock:
            keys = [k for k in self.cells if k[0] == int(raw_frame) and k[2] == int(slot)]
            if not keys:
                raise KeyError(f"no cells for frame {raw_frame} slot {slot}")
            self._remember(keys)
            for key in keys:
                self.cells[key][name] = value
            self._save()
            return [dict(self.cells[k]) for k in keys]

    def undo(self) -> list[dict[str, Any]]:
        with self.lock:
            if not self.history:
                raise ValueError("nothing to undo")
            restored = []
            for key, previous in self.history.pop():
                self.cells[key].clear()
                self.cells[key].update(previous)
                restored.append(dict(self.cells[key]))
            self._save()
            return restored

    def set_author(self, author: Any) -> str | None:
        with self.lock:
            text = " ".join(str(author or "").split())
            self.doc["author"] = text or None
            self._save()
            return self.doc["author"]

    def progress(self, keys: Iterable[CellKey] | None = None) -> dict[str, int]:
        with self.lock:
            cells = (
                list(self.cells.values())
                if keys is None
                else [self.cells[k] for k in keys if k in self.cells]
            )
            return {
                "decided": sum(1 for c in cells if decided(c)),
                "clicked": sum(1 for c in cells if c.get("tip_px") is not None),
                "hidden": sum(1 for c in cells if c.get("hidden")),
                "tip_labelled": sum(1 for c in cells if c.get("tip_attached_label") is not None),
                "total": len(cells),
            }


def load_record(path: Path | None) -> dict[str, Any]:
    if path is None or not Path(path).is_file():
        return {"schema": f"{SCHEMA}/decisions", "cells": [], "author": None, "reviewed_at": None}
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("schema") != f"{SCHEMA}/decisions":
        raise ValueError(f"{path} is not a battle-finebio-tips decisions file")
    return doc


# --------------------------------------------------------------------------------------------
# the pages


def page_url(raw_frame: int) -> str:
    return f"/frame/{int(raw_frame)}"


def markers_for(cell: Mapping[str, Any], point: Sequence[float] | None) -> dict[str, Any]:
    """A full-frame point mapped into each crop of the cell (None outside the crop)."""
    out: dict[str, Any] = {}
    for name, crop in (cell.get("crops") or {}).items():
        if point is None:
            out[name] = None
            continue
        x, y = full_to_crop(point, crop)
        inside = 0 <= x <= crop["size"][0] and 0 <= y <= crop["size"][1]
        out[name] = [round(x, 1), round(y, 1)] if inside else None
    return out


class TipsApp:
    """One tips workspace, its record and the pages over them."""

    def __init__(
        self, *, workspace_dir: Path, workspace: dict[str, Any], record: TipRecord
    ) -> None:
        self.workspace_dir = Path(workspace_dir).resolve()
        self.workspace = workspace
        self.record = record
        self.frames = [int(f["raw_frame"]) for f in workspace["frames"]]
        self.frame_info = {int(f["raw_frame"]): f for f in workspace["frames"]}
        self.page_index = {raw: i for i, raw in enumerate(self.frames)}
        self.page_cells: dict[int, list[dict[str, Any]]] = {raw: [] for raw in self.frames}
        for cell in workspace["cells"]:
            self.page_cells[int(cell["raw_frame"])].append(cell)
        for cells in self.page_cells.values():
            cells.sort(key=lambda c: (int(c["slot"]), view_rank(c["view"])))

    @classmethod
    def open(
        cls, *, workspace_dir: Path, record_path: Path | None = None, author: str | None = None
    ) -> TipsApp:
        workspace_dir = Path(workspace_dir)
        workspace = load_workspace(workspace_dir)
        record_path = Path(record_path) if record_path else workspace_dir / RECORD_NAME
        record = TipRecord(
            workspace_dir=workspace_dir, workspace=workspace, path=record_path, author=author
        )
        return cls(workspace_dir=workspace_dir, workspace=workspace, record=record)

    # ---- files

    def resolve_file(self, relative: str) -> Path | None:
        target = (self.workspace_dir / unquote(relative).lstrip("/")).resolve()
        if self.workspace_dir not in target.parents:
            return None
        if not target.is_file() or target.suffix.lower() not in SERVED_SUFFIXES:
            return None
        return target

    # ---- state

    def page_progress(self, raw: int) -> dict[str, int]:
        return self.record.progress(cell_key(c) for c in self.page_cells[raw])

    @property
    def marker(self) -> bool:
        return bool(self.workspace.get("selection", {}).get("marker", True))

    def _cell_payload(self, record_cell: Mapping[str, Any]) -> dict[str, Any]:
        workspace_cell = self.record.workspace_cells[cell_key(record_cell)]
        suggested = workspace_cell.get("suggested_tip_px") if self.marker else None
        return {
            "cell": dict(record_cell),
            "markers": markers_for(workspace_cell, record_cell.get("tip_px")),
            "suggested": markers_for(workspace_cell, suggested),
        }

    def state(self) -> dict[str, Any]:
        return {
            "workspace": str(self.workspace_dir),
            "record": str(self.record.path),
            "author": self.record.doc.get("author"),
            "updated_at": self.record.doc.get("updated_at"),
            "progress": self.record.progress(),
            "frames": [
                {
                    "raw_frame": raw,
                    "state": self.frame_info[raw]["state"],
                    "url": page_url(raw),
                    "progress": self.page_progress(raw),
                }
                for raw in self.frames
            ],
            "undo_depth": len(self.record.history),
        }

    def _key(self, body: Mapping[str, Any]) -> CellKey:
        try:
            return (int(body["raw_frame"]), str(body["view"]), int(body["slot"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"cell address needs raw_frame, view and slot ({error})") from None

    def update_cell(self, body: Mapping[str, Any]) -> dict[str, Any]:
        key = self._key(body)
        changes = {k: v for k, v in body.items() if k not in ("raw_frame", "view", "slot")}
        cell = self.record.update(key, changes)
        return {
            **self._cell_payload(cell),
            "page": self.page_progress(key[0]),
            "progress": self.record.progress(),
            "updated_at": self.record.doc.get("updated_at"),
        }

    def set_identity(self, body: Mapping[str, Any]) -> dict[str, Any]:
        try:
            raw, slot = int(body["raw_frame"]), int(body["slot"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"identity needs raw_frame and slot ({error})") from None
        cells = self.record.set_identity(raw, slot, body.get("instance_identity"))
        return {
            "cells": [self._cell_payload(c) for c in cells],
            "instance_identity": cells[0]["instance_identity"],
            "progress": self.record.progress(),
            "updated_at": self.record.doc.get("updated_at"),
        }

    def set_tip_label(self, body: Mapping[str, Any]) -> dict[str, Any]:
        try:
            raw, slot = int(body["raw_frame"]), int(body["slot"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"tip label needs raw_frame and slot ({error})") from None
        cells = self.record.set_tip_label(raw, slot, body.get("tip_attached_label"))
        return {
            "cells": [self._cell_payload(c) for c in cells],
            "tip_attached_label": cells[0]["tip_attached_label"],
            "progress": self.record.progress(),
            "updated_at": self.record.doc.get("updated_at"),
        }

    def undo(self) -> dict[str, Any]:
        restored = self.record.undo()
        frames = sorted({int(c["raw_frame"]) for c in restored})
        return {
            "cells": [self._cell_payload(c) for c in restored],
            "pages": {str(raw): self.page_progress(raw) for raw in frames},
            "progress": self.record.progress(),
            "undo_depth": len(self.record.history),
            "updated_at": self.record.doc.get("updated_at"),
        }

    def set_author(self, body: Mapping[str, Any]) -> dict[str, Any]:
        author = self.record.set_author(body.get("author"))
        return {"author": author, "updated_at": self.record.doc.get("updated_at")}

    # ---- HTML

    def index_html(self) -> str:
        progress = self.record.progress()
        rows = []
        for i, raw in enumerate(self.frames, start=1):
            info = self.frame_info[raw]
            page = self.page_progress(raw)
            done = " done" if page["total"] and page["decided"] == page["total"] else ""
            pipettes = ", ".join(
                f"{IDENTITY_SHORT.get(s['class'], s['class'])} ({', '.join(s['views'])})"
                for s in info["slots"]
            )
            rows.append(
                f'<tr class="page{done}"><td>{i}</td>'
                f'<td><a href="{page_url(raw)}">f{raw:06d}</a></td>'
                f"<td>{int(info['proxy_frame'])}</td><td>{_e(info['state'])}</td>"
                f"<td>{_e(pipettes)}</td>"
                f'<td class="progress">{page["decided"]} / {page["total"]}</td></tr>'
            )
        if self.marker:
            instructions = (
                "Click the very end of the white tip cone in every crop. The dashed marker is "
                "the detector's guess and is never saved; your click replaces it. Press "
                "<kbd>h</kbd> when the tip is hidden in that view, <kbd>t</kbd> / <kbd>b</kbd> "
                "for a disposable tip on / a bare pipette, and <kbd>1</kbd>-<kbd>4</kbd> for the "
                "plunger colour you see."
            )
        else:
            instructions = (
                "Click the very end of the pipette in every crop: the end away from the "
                "coloured plunger button, where a disposable tip's point would be (the point "
                "of the tip when one is on, the end of the cone when bare). There is no "
                "suggested marker; the small arrow at the crop's edge is the tracker's guess "
                "and is never saved. Press <kbd>h</kbd> when that end is hidden in that view, "
                "<kbd>t</kbd> once per pipette per frame when a disposable tip IS attached and "
                "<kbd>b</kbd> when it is bare, and <kbd>1</kbd>-<kbd>4</kbd> for the plunger "
                "colour you see."
            )
        return INDEX_TEMPLATE.format(
            style=STYLE + TIPS_STYLE,
            script=SCRIPT,
            trial=_e(self.workspace.get("trial", "")),
            decided=progress["decided"],
            total=progress["total"],
            record=_e(self.record.path),
            author=_e(self.record.doc.get("author") or ""),
            updated=_e(self.record.doc.get("updated_at") or ""),
            rows="".join(rows),
            instructions=instructions,
            keys=KEYS_HTML,
            first_url=page_url(self.frames[0]) if self.frames else "/",
            page_data=_script_json(
                {"kind": "index", "next": page_url(self.frames[0]) if self.frames else None}
            ),
        )

    def frame_html(self, raw: int) -> str:
        raw = int(raw)
        if raw not in self.page_index:
            raise KeyError(f"no frame {raw} in this workspace")
        position = self.page_index[raw]
        previous = page_url(self.frames[position - 1]) if position > 0 else None
        following = page_url(self.frames[position + 1]) if position + 1 < len(self.frames) else None
        info = self.frame_info[raw]
        cells = self.page_cells[raw]
        by_slot: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for cell in cells:
            by_slot[int(cell["slot"])].append(cell)
        with self.record.lock:
            sections = "".join(
                self._slot_html(raw, slot, group) for slot, group in sorted(by_slot.items())
            )
            page_progress = self.page_progress(raw)
            progress = self.record.progress()
        data = {
            "kind": "frame",
            "raw_frame": raw,
            "prev": previous,
            "next": following or "/",
            "position": position + 1,
            "pages": len(self.frames),
            "cells": [{"view": c["view"], "slot": int(c["slot"])} for c in cells],
        }
        return FRAME_TEMPLATE.format(
            style=STYLE + TIPS_STYLE,
            script=SCRIPT,
            trial=_e(self.workspace.get("trial", "")),
            raw=raw,
            proxy=int(info["proxy_frame"]),
            state=_e(info["state"]),
            reason=_e(info["reason"]),
            position=position + 1,
            pages=len(self.frames),
            prev_link=f'<a href="{previous}" title="[">&lsaquo; prev</a>'
            if previous
            else '<span class="muted">&lsaquo; prev</span>',
            next_link=f'<a href="{following}" title="]">next &rsaquo;</a>'
            if following
            else '<a href="/" title="]">index &rsaquo;</a>',
            page_decided=page_progress["decided"],
            page_total=page_progress["total"],
            decided=progress["decided"],
            total=progress["total"],
            slots=sections,
            keys=KEYS_HTML,
            page_data=_script_json(data),
        )

    def _slot_html(self, raw: int, slot: int, group: Sequence[dict[str, Any]]) -> str:
        cls = group[0]["class"]
        identity = self.record.cells[cell_key(group[0])].get("instance_identity")
        buttons = []
        for key, name in IDENTITY_KEYS.items():
            active = " active" if identity == name else ""
            buttons.append(
                f'<button data-identity="{name}" class="{active.strip()}" title="{key}">'
                f"{IDENTITY_SHORT[name]} <kbd>{key}</kbd></button>"
            )
        buttons.append(
            f'<button data-identity="" class="{"active" if identity is None else ""}" title="0">'
            "none <kbd>0</kbd></button>"
        )
        label = self.record.cells[cell_key(group[0])].get("tip_attached_label")
        tip_buttons = "".join(
            f'<button data-tip-label="{value}" class="{"active" if label is state else ""}" '
            f'title="{key}">{text} <kbd>{key}</kbd></button>'
            for value, state, key, text in (
                ("true", True, "t", "tip on"),
                ("false", False, "b", "bare"),
                ("", None, "-", "unsaid"),
            )
        )
        colour = CLASS_COLOURS_BGR.get(cls, (200, 200, 200))
        swatch = f"rgb({colour[2]},{colour[1]},{colour[0]})"
        cells = "".join(self._cell_html(cell) for cell in group)
        return (
            f'<section class="slot" data-slot="{slot}" data-raw="{raw}">'
            f'<div class="slothead"><span class="swatch" style="background:{swatch}"></span>'
            f"<b>{_e(IDENTITY_SHORT.get(cls, cls))}</b> "
            f'<span class="cls">{_e(cls)} &middot; slot {slot} &middot; {len(group)} views</span>'
            f'<span class="tiplabel">disposable tip: {tip_buttons}</span>'
            f'<span class="identity">plunger colour seen: {"".join(buttons)}</span></div>'
            f'<div class="cells">{cells}</div></section>'
        )

    @staticmethod
    def _guess_html(cell: Mapping[str, Any], name: str, crop: Mapping[str, Any]) -> str:
        """Sep 29 (no marker): a small arrow at the crop's edge pointing from the crop's
        centre towards the rows' tail-rule end, labelled "tracker's guess"; nothing when the
        rule did not decide or the guess lies outside the crop."""
        guess = cell.get("guess_tip_px")
        if guess is None:
            return ""
        point = markers_for(cell, guess).get(name)
        if point is None:
            return ""
        w, h = float(crop["size"][0]), float(crop["size"][1])
        dx, dy = point[0] - w / 2, point[1] - h / 2
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            return ""
        # Where the ray from the centre through the guess leaves the crop, inset a little.
        scale = min(
            (w / 2 - 14) / abs(dx) if abs(dx) > 1e-6 else math.inf,
            (h / 2 - 14) / abs(dy) if abs(dy) > 1e-6 else math.inf,
        )
        x, y = w / 2 + dx * scale, h / 2 + dy * scale
        angle = math.degrees(math.atan2(dy, dx))
        return (
            f'<div class="guess" style="left:{100 * x / w:.2f}%;top:{100 * y / h:.2f}%;'
            f'transform:translate(-50%,-50%) rotate({angle:.1f}deg)" '
            f"title=\"tracker's guess ({_e(cell.get('guess_rule') or '')}): the end it would "
            'call the tip. A guess, never saved.">&#10148;</div>'
            f'<div class="guesslabel" style="left:{100 * x / w:.2f}%;top:{100 * y / h:.2f}%">'
            "tracker's guess</div>"
        )

    def _cell_html(self, cell: dict[str, Any]) -> str:
        record_cell = self.record.cells[cell_key(cell)]
        tip = record_cell.get("tip_px")
        hidden = bool(record_cell.get("hidden"))
        marker = self.marker
        classes = "cell"
        if tip is not None or hidden:
            classes += " decided"
        if hidden:
            classes += " hidden"
        undecided = "suggested" if marker else "undecided"
        state_text = "hidden" if hidden else ("tip set" if tip is not None else undecided)
        pics = []
        for name in ("zoom", "full"):
            crop = cell["crops"][name]
            suggested = cell.get("suggested_tip_px") if marker else None
            point = tip if tip is not None else (None if hidden else suggested)
            marker_class = "marker set" if tip is not None else "marker suggested"
            position = markers_for(cell, point).get(name) if point is not None else None
            style = (
                f"left:{100 * position[0] / crop['size'][0]:.2f}%;"
                f"top:{100 * position[1] / crop['size'][1]:.2f}%"
                if position is not None
                else "display:none"
            )
            guess = "" if marker or name != "zoom" else self._guess_html(cell, name, crop)
            pics.append(
                f'<div class="pic {name}" data-w="{crop["size"][0]}" data-h="{crop["size"][1]}">'
                f'<img src="/files/{_e(crop["uri"])}" data-crop="{name}" '
                f'width="{crop["size"][0]}" height="{crop["size"][1]}" '
                f'alt="{name} crop {_e(cell["view"])}">'
                f'<div class="{marker_class}" style="{style}"></div>{guess}</div>'
            )
        return (
            f'<figure class="{classes}" id="cell-{_e(cell["view"])}-{int(cell["slot"])}" '
            f'data-view="{_e(cell["view"])}" data-slot="{int(cell["slot"])}">'
            f"<figcaption><b>{_e(cell['view'])}</b> &middot; mask {int(cell['mask_area_px'])} px "
            f"&middot; zoom {cell['crops']['zoom']['scale']:g}x &middot; "
            f'<span class="state">{state_text}</span></figcaption>'
            f"{''.join(pics)}"
            '<div class="actions"><button data-action="hidden" title="h">hidden <kbd>h</kbd>'
            '</button><button data-action="clear" title="x">clear <kbd>x</kbd></button></div>'
            "</figure>"
        )


TIPS_STYLE = """
.slot{border:1px solid #333;border-radius:6px;padding:8px;margin:10px 0;background:#181818}
.slothead{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:6px}
.slothead .identity{display:flex;gap:4px;align-items:center;flex-wrap:wrap;color:#aaa;
  font-size:12px;margin-left:auto}
.cells{display:flex;gap:10px;flex-wrap:wrap;align-items:flex-start}
.cell{margin:0;border:2px solid #333;border-left:6px solid #555;border-radius:4px;
  background:#000;display:flex;flex-direction:column}
.cell.decided{border-left-color:#3c9}
.cell.hidden{border-left-color:#c63}
.cell.current{border-color:#fc3}
.cell.current{border-left-color:#fc3}
.cell figcaption{font-size:11px;padding:2px 4px;background:#222}
.cell .state{color:#fc3;font-weight:bold}
.cell .actions{display:flex;gap:4px;padding:3px}
.pic{position:relative;line-height:0}
.pic img{display:block;max-width:100%;height:auto;cursor:crosshair}
.pic.full{display:none;max-width:640px}
body.show-full .pic.full{display:block}
.marker{position:absolute;width:16px;height:16px;margin:-8px 0 0 -8px;border:2px solid #fc3;
  border-radius:50%;pointer-events:none;box-shadow:0 0 0 1px #000}
.marker::after{content:'';position:absolute;left:5px;top:5px;width:2px;height:2px;
  background:#fff;box-shadow:0 0 0 1px #000}
.marker.suggested{border-style:dashed;border-color:#9cf;opacity:.6}
.marker.set{border-color:#fc3;background:rgba(255,204,51,.2)}
.swatch{display:inline-block;width:12px;height:12px;border-radius:2px;vertical-align:middle}
.slothead .tiplabel{display:flex;gap:4px;align-items:center;color:#aaa;font-size:12px}
.slothead .tiplabel button.active,.slothead .identity button.active{outline:2px solid #fc3}
.guess{position:absolute;color:#9cf;font-size:18px;line-height:1;pointer-events:none;
  text-shadow:0 0 2px #000,0 0 2px #000;opacity:.85}
.guesslabel{position:absolute;color:#9cf;font-size:10px;pointer-events:none;
  transform:translate(-50%,12px);white-space:nowrap;text-shadow:0 0 2px #000,0 0 2px #000}
"""

KEYS_HTML = (
    '<div class="keys"><b>click</b> sets the end and moves on &nbsp; <kbd>h</kbd> hidden &nbsp; '
    "<kbd>x</kbd> clear &nbsp; <kbd>u</kbd> undo &nbsp; <kbd>j</kbd>/<kbd>k</kbd> cell &nbsp; "
    "<kbd>]</kbd>/<kbd>[</kbd> frame &nbsp; <kbd>t</kbd> a disposable tip IS attached "
    "<kbd>b</kbd> bare <kbd>-</kbd> unsaid (per pipette per frame, saved on every view) &nbsp; "
    "<kbd>1</kbd> blue <kbd>2</kbd> yellow <kbd>3</kbd> red "
    "<kbd>4</kbd> 8-channel <kbd>0</kbd> none (plunger colour seen, saved on every view of the "
    "pipette) &nbsp; <kbd>o</kbd> whole frames</div>"
)

INDEX_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Tip clicks {trial}: {decided} / {total}</title>
<style>{style}</style></head>
<body>
<header><h1>Tip clicks {trial}</h1>
<span class="progress"><b id="global-progress">{decided} / {total}</b> cells decided</span>
<a href="{first_url}">start &rsaquo;</a>
<span id="status" class="muted"></span></header>
<main>
<p class="muted">record <code>{record}</code> &middot; author
<input id="author" value="{author}" placeholder="your name" autocomplete="off">
&middot; last write <span id="updated">{updated}</span></p>
<table><thead><tr><th>#</th><th>frame</th><th>proxy</th><th>state</th><th>pipettes (views)</th>
<th>decided</th></tr></thead><tbody>{rows}</tbody></table>
{keys}
<p class="muted">{instructions}</p>
</main>
<script id="page-data" type="application/json">{page_data}</script>
<script>{script}</script>
</body></html>
"""

FRAME_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>f{raw:06d} tips: {page_decided} / {page_total}</title>
<style>{style}</style></head>
<body>
<header><a href="/">index</a> {prev_link}
<h1>f{raw:06d} <span class="muted">(proxy {proxy}; {state}; frame {position} of {pages})</span></h1>
{next_link}
<span class="progress">frame <b id="page-progress">{page_decided} / {page_total}</b> &middot;
all <b id="global-progress">{decided} / {total}</b> cells</span>
<span id="status" class="muted"></span></header>
<main>
<p class="muted">{reason}</p>
<div id="slots">{slots}</div>
{keys}
</main>
<script id="page-data" type="application/json">{page_data}</script>
<script>{script}</script>
</body></html>
"""

SCRIPT = r"""
(function () {
  'use strict';
  var data = JSON.parse(document.getElementById('page-data').textContent);
  var statusEl = document.getElementById('status');
  function status(text, kind) {
    if (!statusEl) return;
    statusEl.textContent = text;
    statusEl.className = kind || 'muted';
  }
  async function postJson(url, payload) {
    var res = await fetch(url, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload || {})
    });
    var body = null;
    try { body = await res.json(); } catch (e) { body = {error: res.statusText}; }
    if (!res.ok) throw new Error(body.error || res.statusText);
    return body;
  }
  var author = document.getElementById('author');
  if (author) {
    author.addEventListener('change', function () {
      postJson('/api/author', {author: author.value}).then(function (body) {
        document.getElementById('updated').textContent = body.updated_at || '';
        status('author saved', 'ok');
      }).catch(function (e) { status(e.message, 'error'); });
    });
  }
  if (data.kind === 'index') {
    document.addEventListener('keydown', function (e) {
      var t = e.target;
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA')) {
        if (e.key === 'Enter' || e.key === 'Escape') { t.blur(); e.preventDefault(); }
        return;
      }
      if (e.key === ']' && data.next) { location.href = data.next; e.preventDefault(); }
    });
    return;
  }

  var cells = Array.prototype.slice.call(document.querySelectorAll('.cell'));
  if (!cells.length) return;
  var current = -1;
  function setCurrent(i, scroll) {
    if (i < 0 || i >= cells.length) return false;
    if (current >= 0) cells[current].classList.remove('current');
    current = i;
    cells[current].classList.add('current');
    if (scroll !== false) cells[current].scrollIntoView({block: 'nearest', behavior: 'smooth'});
    return true;
  }
  var start = cells.findIndex(function (c) { return !c.classList.contains('decided'); });
  setCurrent(start < 0 ? 0 : start, false);
  function advance() {
    if (!setCurrent(current + 1)) status('last cell on this frame; ] goes to the next', 'muted');
  }
  function placeMarker(pic, point, cls) {
    var marker = pic.querySelector('.marker');
    if (!point) { marker.style.display = 'none'; return; }
    marker.className = 'marker ' + cls;
    marker.style.display = 'block';
    marker.style.left = (100 * point[0] / Number(pic.dataset.w)).toFixed(2) + '%';
    marker.style.top = (100 * point[1] / Number(pic.dataset.h)).toFixed(2) + '%';
  }
  function findCell(rec) {
    if (rec.raw_frame !== data.raw_frame) return null;
    return document.getElementById('cell-' + rec.view + '-' + rec.slot);
  }
  function applyCell(payload) {
    var rec = payload.cell;
    var cell = findCell(rec);
    if (!cell) return;
    var hasTip = rec.tip_px !== null && rec.tip_px !== undefined;
    cell.classList.toggle('decided', hasTip || rec.hidden);
    cell.classList.toggle('hidden', !!rec.hidden);
    cell.querySelector('.state').textContent =
      rec.hidden ? 'hidden' : (hasTip ? 'tip set' : 'suggested');
    ['zoom', 'full'].forEach(function (name) {
      var pic = cell.querySelector('.pic.' + name);
      if (hasTip) placeMarker(pic, payload.markers[name], 'set');
      else if (rec.hidden) placeMarker(pic, null);
      else placeMarker(pic, payload.suggested[name], 'suggested');
    });
    var slot = cell.closest('.slot');
    slot.querySelectorAll('button[data-identity]').forEach(function (b) {
      b.classList.toggle('active',
        (b.dataset.identity || null) === (rec.instance_identity || null));
    });
    var label = rec.tip_attached_label === true ? 'true'
      : rec.tip_attached_label === false ? 'false' : '';
    slot.querySelectorAll('button[data-tip-label]').forEach(function (b) {
      b.classList.toggle('active', b.dataset.tipLabel === label);
    });
  }
  function applyProgress(body) {
    if (body.page) {
      document.getElementById('page-progress').textContent =
        body.page.decided + ' / ' + body.page.total;
    }
    if (body.pages && body.pages[String(data.raw_frame)]) {
      var p = body.pages[String(data.raw_frame)];
      document.getElementById('page-progress').textContent = p.decided + ' / ' + p.total;
    }
    if (body.progress) {
      document.getElementById('global-progress').textContent =
        body.progress.decided + ' / ' + body.progress.total;
    }
    status('saved ' + (body.updated_at || ''), 'ok');
  }
  function post(cell, changes, moveOn) {
    var payload = {
      raw_frame: data.raw_frame, view: cell.dataset.view, slot: Number(cell.dataset.slot)
    };
    Object.keys(changes).forEach(function (k) { payload[k] = changes[k]; });
    return postJson('/api/cell', payload).then(function (body) {
      applyCell(body);
      applyProgress(body);
      if (moveOn) advance();
      return body;
    }).catch(function (e) { status(e.message, 'error'); return null; });
  }
  function identity(cell, value) {
    return postJson('/api/identity', {
      raw_frame: data.raw_frame, slot: Number(cell.dataset.slot), instance_identity: value
    }).then(function (body) {
      body.cells.forEach(applyCell);
      applyProgress(body);
      status('identity ' + (body.instance_identity || 'none') + ' saved on ' + body.cells.length +
        ' views', 'ok');
    }).catch(function (e) { status(e.message, 'error'); });
  }
  function tipLabel(cell, value) {
    return postJson('/api/tip-label', {
      raw_frame: data.raw_frame, slot: Number(cell.dataset.slot), tip_attached_label: value
    }).then(function (body) {
      body.cells.forEach(applyCell);
      applyProgress(body);
      var said = body.tip_attached_label === true ? 'tip attached'
        : body.tip_attached_label === false ? 'bare' : 'unsaid';
      status('disposable tip: ' + said + ' saved on ' + body.cells.length + ' views', 'ok');
    }).catch(function (e) { status(e.message, 'error'); });
  }
  function undo() {
    return postJson('/api/undo', {}).then(function (body) {
      body.cells.forEach(applyCell);
      applyProgress(body);
      var here = body.cells.filter(function (p) { return p.cell.raw_frame === data.raw_frame; });
      status('undone (' + body.cells.length + ' cells' + (here.length ? '' : ', on another frame') +
        '; ' + body.undo_depth + ' more)', 'ok');
      if (here.length) {
        var i = cells.indexOf(findCell(here[0].cell));
        if (i >= 0) setCurrent(i);
      }
    }).catch(function (e) { status(e.message, 'error'); });
  }
  function toggleFull() { document.body.classList.toggle('show-full'); }

  document.addEventListener('keydown', function (e) {
    var t = e.target;
    if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) {
      if (e.key === 'Enter' || e.key === 'Escape') { t.blur(); e.preventDefault(); }
      return;
    }
    if (e.altKey || e.ctrlKey || e.metaKey) return;
    var cell = cells[current];
    switch (e.key) {
      case 'h': post(cell, {hidden: true}, true); break;
      case 'x': post(cell, {clear: true}, false); break;
      case 'u': undo(); break;
      case 'j': case 'ArrowDown': case 'ArrowRight': setCurrent(current + 1); break;
      case 'k': case 'ArrowUp': case 'ArrowLeft': setCurrent(current - 1); break;
      case ']': if (data.next) location.href = data.next; break;
      case '[': if (data.prev) location.href = data.prev; break;
      case '1': case '2': case '3': case '4': identity(cell, e.key); break;
      case '0': identity(cell, null); break;
      case 't': tipLabel(cell, true); break;
      case 'b': tipLabel(cell, false); break;
      case '-': tipLabel(cell, null); break;
      case 'o': toggleFull(); break;
      default: return;
    }
    e.preventDefault();
  });

  cells.forEach(function (cell, i) {
    cell.addEventListener('mousedown', function () { setCurrent(i, false); });
    cell.querySelectorAll('.pic img').forEach(function (img) {
      img.addEventListener('click', function (e) {
        var pic = img.parentNode;
        var rect = img.getBoundingClientRect();
        var w = img.naturalWidth || Number(pic.dataset.w);
        var h = img.naturalHeight || Number(pic.dataset.h);
        var x = (e.clientX - rect.left) * w / rect.width;
        var y = (e.clientY - rect.top) * h / rect.height;
        setCurrent(i, false);
        post(cell, {click: {crop: img.dataset.crop, x: x, y: y}}, true);
      });
    });
    cell.querySelectorAll('button[data-action]').forEach(function (b) {
      b.addEventListener('click', function () {
        setCurrent(i, false);
        if (b.dataset.action === 'hidden') post(cell, {hidden: true}, true);
        else post(cell, {clear: true}, false);
      });
    });
  });
  document.querySelectorAll('.slot').forEach(function (slot) {
    slot.querySelectorAll('button[data-identity]').forEach(function (b) {
      b.addEventListener('click', function () {
        var first = slot.querySelector('.cell');
        identity(first, b.dataset.identity || null);
      });
    });
    slot.querySelectorAll('button[data-tip-label]').forEach(function (b) {
      b.addEventListener('click', function () {
        var first = slot.querySelector('.cell');
        var v = b.dataset.tipLabel;
        tipLabel(first, v === 'true' ? true : v === 'false' ? false : null);
      });
    });
  });
})();
"""


def make_tips_handler(app: TipsApp) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args: object) -> None:
            return

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            value = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(value, dict):
                raise ValueError("JSON body must be an object")
            return value

        def _file(self, relative: str) -> None:
            target = app.resolve_file(relative)
            if target is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            content = target.read_bytes()
            ctype = SERVED_SUFFIXES[target.suffix.lower()]
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(content)))
            self.send_header(
                "Cache-Control", "max-age=3600" if ctype.startswith("image/") else "no-store"
            )
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            try:
                if path == "/":
                    _html(self, app.index_html())
                elif path.startswith("/frame/"):
                    tail = path.removeprefix("/frame/").strip("/")
                    if not tail.isdigit():
                        self.send_error(HTTPStatus.NOT_FOUND)
                        return
                    _html(self, app.frame_html(int(tail)))
                elif path == "/api/state":
                    _json(self, HTTPStatus.OK, app.state())
                elif path.startswith("/files/"):
                    self._file(path.removeprefix("/files/"))
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
            except KeyError as error:
                _json(self, HTTPStatus.NOT_FOUND, {"error": str(error)})

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            try:
                body = self._body()
                if path == "/api/cell":
                    _json(self, HTTPStatus.OK, app.update_cell(body))
                elif path == "/api/identity":
                    _json(self, HTTPStatus.OK, app.set_identity(body))
                elif path == "/api/tip-label":
                    _json(self, HTTPStatus.OK, app.set_tip_label(body))
                elif path == "/api/undo":
                    _json(self, HTTPStatus.OK, app.undo())
                elif path == "/api/author":
                    _json(self, HTTPStatus.OK, app.set_author(body))
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
            except KeyError as error:
                _json(self, HTTPStatus.NOT_FOUND, {"error": str(error)})
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                _json(self, HTTPStatus.BAD_REQUEST, {"error": str(error)})
            except OSError as error:
                _json(
                    self,
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": f"could not write the record: {error}"},
                )

    return Handler


def run_serve(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    try:
        host, dns_name = resolve_bind(args.bind, args.tailscale)
        app = TipsApp.open(
            workspace_dir=args.workspace, record_path=args.record, author=args.author
        )
        server = ThreadingHTTPServer((host, args.port), make_tips_handler(app))
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    urls = reachable_urls(host, server.server_port, dns_name)
    progress = app.record.progress()
    print(f"Tip workspace: {urls[0]}", flush=True)
    for url in urls[1:]:
        print(f"Also reachable at: {url}", flush=True)
    if host != DEFAULT_HOST:
        print(
            "Bound beyond loopback: no authentication; any client that reaches this address "
            "can edit the record.",
            flush=True,
        )
    print(
        f"Record: {app.record.path} ({progress['decided']} / {progress['total']} cells decided; "
        f"{len(app.frames)} frames)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


# --------------------------------------------------------------------------------------------
# scoring: clicks -> 3D tips -> tip error per tracks file


def cameras_from_clip(clip: Mapping[str, Any], root: Path) -> Callable[[int], dict[str, Camera]]:
    """``at(raw frame) -> {view: Camera}``: the window camera config's fixed views plus the
    fpv where its shipped pose is valid (absent when the pose file is not on this machine)."""
    config = read_camera_config(root / clip["window_camera_config"])
    fixed = cameras_from_config(config)
    try:
        poses = fpv_poses(config.trial)
    except FileNotFoundError:
        poses = None

    def at(frame: int) -> dict[str, Camera]:
        cams = dict(fixed)
        if poses is not None:
            fpv = fpv_camera_from_config(config, int(frame), poses)
            if fpv is not None:
                cams[FPV_VIEW] = fpv
        return cams

    return at


@dataclass
class TipAnchor:
    raw_frame: int
    slot: int
    cls: str
    state: str
    identity: str | None
    labels: dict[str, str]
    pixels: dict[str, tuple[float, float]]
    tip_cm: np.ndarray | None = None
    residual_px: dict[str, float] = field(default_factory=dict)
    dropped_view: str | None = None
    hidden_views: list[str] = field(default_factory=list)
    # Sep 29: the `t` / `b` label (None when not said) and whether every view cell of the
    # pipette-frame carries that same value (the label is written on all of them, so a
    # disagreement means a hand-edited record).
    tip_label: bool | None = None
    tip_label_consistent: bool = False
    tip_label_inconsistent: bool = False

    @property
    def views(self) -> list[str]:
        return sorted(self.pixels, key=view_rank)

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw_frame": self.raw_frame,
            "slot": self.slot,
            "class": self.cls,
            "state": self.state,
            "instance_identity": self.identity,
            "tip_attached_label": self.tip_label if self.tip_label_consistent else None,
            "tip_label_consistent": self.tip_label_consistent,
            "views": self.views,
            "n_views": len(self.residual_px) if self.residual_px else len(self.pixels),
            "hidden_views": self.hidden_views,
            "tip_cm": None if self.tip_cm is None else [round(float(v), 3) for v in self.tip_cm],
            "tip_height_cm": None if self.tip_cm is None else round(-float(self.tip_cm[2]), 2),
            "residual_px": {v: round(r, 2) for v, r in self.residual_px.items()},
            "residual_max_px": round(max(self.residual_px.values()), 2)
            if self.residual_px
            else None,
            "dropped_view": self.dropped_view,
        }


def _triangulate(
    cams: Mapping[str, Camera], pixels: Mapping[str, tuple[float, float]], views: Sequence[str]
) -> tuple[np.ndarray, dict[str, float]]:
    point = triangulate_pixels(
        [cams[v] for v in views], [np.array(pixels[v], dtype=np.float64) for v in views]
    )
    residuals = {
        v: float(np.linalg.norm(cams[v].project(point)[0] - np.array(pixels[v]))) for v in views
    }
    return point, residuals


def triangulate_clicks(
    workspace: Mapping[str, Any],
    record: Mapping[str, Any],
    cameras_at: Callable[[int], Mapping[str, Camera]],
    *,
    residual_gate_px: float = RESIDUAL_GATE_PX,
) -> tuple[list[TipAnchor], list[dict[str, Any]]]:
    """One anchor per (frame, pipette) with at least two clicked views that have a camera
    (undistort + DLT); with three or more views the worst view is dropped once when its
    residual exceeds `residual_gate_px`. A (frame, pipette) with fewer clicked views, or
    whose remaining clicks still disagree by more than the gate (two views cannot say which
    click is wrong), keeps its identity for IDF1 but has no 3D tip and is listed under
    `dropped` with its reason."""
    decisions = {cell_key(c): c for c in record.get("cells", [])}
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for cell in workspace["cells"]:
        groups[(int(cell["raw_frame"]), int(cell["slot"]))].append(cell)
    anchors: list[TipAnchor] = []
    dropped: list[dict[str, Any]] = []
    for (raw, slot), cells in sorted(groups.items()):
        cams = cameras_at(raw)
        pixels: dict[str, tuple[float, float]] = {}
        labels = {str(c["view"]): str(c["label"]) for c in cells}
        hidden: list[str] = []
        identity = None
        no_camera: list[str] = []
        tip_labels: list[bool | None] = []
        for cell in cells:
            entry = decisions.get(cell_key(cell), {})
            identity = identity or entry.get("instance_identity")
            tip_labels.append(entry.get("tip_attached_label"))
            if entry.get("hidden"):
                hidden.append(str(cell["view"]))
            tip = entry.get("tip_px")
            if tip is None:
                continue
            if cell["view"] not in cams:
                no_camera.append(str(cell["view"]))
                continue
            pixels[str(cell["view"])] = (float(tip[0]), float(tip[1]))
        anchor = TipAnchor(
            raw, slot, cells[0]["class"], cells[0]["state"], identity, labels, pixels
        )
        anchor.hidden_views = hidden
        said = {label for label in tip_labels if label is not None}
        if len(said) == 1 and all(label is not None for label in tip_labels):
            anchor.tip_label = said.pop()
            anchor.tip_label_consistent = True
        elif said:
            anchor.tip_label_inconsistent = True
        if len(pixels) < 2:
            reason = (
                "no click"
                if not pixels and not no_camera
                else f"{len(pixels)} clicked view with a camera"
                + (f" (no camera for {', '.join(no_camera)})" if no_camera else "")
            )
            dropped.append({"raw_frame": raw, "slot": slot, "class": anchor.cls, "reason": reason})
            if identity:
                anchors.append(anchor)
            continue
        views = anchor.views
        point, residuals = _triangulate(cams, pixels, views)
        if len(views) >= 3 and max(residuals.values()) > residual_gate_px:
            worst = max(residuals, key=residuals.get)
            kept = [v for v in views if v != worst]
            point, residuals = _triangulate(cams, pixels, kept)
            anchor.dropped_view = worst
        anchor.residual_px = residuals
        if max(residuals.values()) > residual_gate_px:
            dropped.append(
                {
                    "raw_frame": raw,
                    "slot": slot,
                    "class": anchor.cls,
                    "reason": (
                        f"clicks disagree: residual {max(residuals.values()):.0f} px over the "
                        f"{residual_gate_px:g} px gate on {', '.join(residuals)}"
                    ),
                    "views": list(residuals),
                    "residual_px": {v: round(r, 2) for v, r in residuals.items()},
                    "dropped_view": anchor.dropped_view,
                }
            )
            if identity:
                anchors.append(anchor)
            continue
        anchor.tip_cm = point
        anchors.append(anchor)
    return anchors, dropped


def parse_tracks(items: Sequence[str]) -> dict[str, Path]:
    """``name=path`` or a path; an unnamed ``.../tracks.jsonl`` is named by its directory."""
    out: dict[str, Path] = {}
    for item in items:
        name, separator, value = item.partition("=")
        if separator and not name.strip().startswith(("/", ".")) and "/" not in name:
            path = Path(value.strip())
            label = name.strip()
        else:
            path = Path(item.strip())
            label = path.parent.name if path.name == "tracks.jsonl" else path.stem
        if label in out:
            raise ValueError(f"two tracks files named {label!r}; name them with name=path")
        out[label] = path
    if not out:
        raise ValueError("--tracks names at least one tracks.jsonl")
    return out


def is_pipette_track(row: Mapping[str, Any]) -> bool:
    return row.get("object_class") in PIPETTE_CLASSES or row.get("object_class") == LINE_CLASS


def read_track_rows(path: Path, frames: Iterable[int]) -> dict[int, list[dict[str, Any]]]:
    """frame -> the pipette track rows on `frames` (line tracks of the `pipette` class and
    point tracks of the four detector classes)."""
    wanted = {int(f) for f in frames}
    # A cheap text prefilter before json.loads; the tracker writes compact JSON and the
    # colour annotator writes it with a space after the colon, so both spellings pass.
    keys = {f'"frame_index":{s}{f},' for f in wanted for s in ("", " ")}
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            if '"frame_index":' in line and not any(k in line for k in keys):
                continue
            row = json.loads(line)
            if int(row["frame_index"]) in wanted and is_pipette_track(row):
                out[int(row["frame_index"])].append(row)
    return out


def track_points(row: Mapping[str, Any]) -> list[np.ndarray]:
    endpoints = row.get("endpoints_cm")
    if endpoints:
        return [np.asarray(p, dtype=np.float64) for p in endpoints]
    return [np.asarray(row["position_cm"], dtype=np.float64)]


def match_track(
    anchor: TipAnchor, rows: Sequence[Mapping[str, Any]], *, gate_cm: float = MATCH_GATE_CM
) -> tuple[dict[str, Any] | None, str]:
    """The track of this pipette on this frame: the row whose `support_slots` share the most
    (view, slot label) pairs with the anchor's cells, else the row with the endpoint (or
    position) nearest the anchor's tip within `gate_cm`. A support match on a single view
    yields to a nearer track when its own endpoint is outside the gate: the detector's
    classes are confused, so one agreeing label is weak evidence."""

    def distance(row: Mapping[str, Any]) -> float | None:
        if anchor.tip_cm is None:
            return None
        return min(float(np.linalg.norm(p - anchor.tip_cm)) for p in track_points(row))

    best, best_hits = None, 0
    for row in rows:
        support = row.get("support_slots") or {}
        hits = sum(1 for view, label in anchor.labels.items() if support.get(view) == label)
        if hits > best_hits:
            best, best_hits = row, hits
    nearest, nearest_cm = None, gate_cm
    for row in rows:
        far = distance(row)
        if far is not None and far <= nearest_cm:
            nearest, nearest_cm = row, far
    if best is not None:
        best_cm = distance(best)
        if best_hits >= 2 or best_cm is None or best_cm <= gate_cm:
            return dict(best), f"support_slots ({best_hits} views)"
        if nearest is not None:
            return dict(nearest), (
                f"nearest endpoint ({nearest_cm:.1f} cm; the 1-view support match "
                f"{best.get('track_id')} was {best_cm:.1f} cm away)"
            )
        return dict(best), f"support_slots (1 views, {best_cm:.1f} cm away)"
    if anchor.tip_cm is None:
        return None, "no support match and no 3D tip"
    if nearest is None:
        return None, f"no pipette track within {gate_cm:g} cm"
    return dict(nearest), f"nearest endpoint ({nearest_cm:.1f} cm)"


def along_across(
    anchor_cm: np.ndarray, tip_end: np.ndarray, other_end: np.ndarray
) -> tuple[float, float] | None:
    """Split (anchor - tip_end) along the segment's axis: `along` is signed along the
    outward direction from `other_end` through `tip_end`, positive when the anchor lies
    beyond the track's tip end (the mask stopped short of the anchor, as it does when a
    disposable tip is attached and SAM3 stops at the body's cone), negative when the track
    overshoots; `across` is the perpendicular distance. None for a degenerate segment."""
    axis = np.asarray(tip_end, dtype=np.float64) - np.asarray(other_end, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-9:
        return None
    axis /= norm
    offset = np.asarray(anchor_cm, dtype=np.float64) - np.asarray(tip_end, dtype=np.float64)
    along = float(offset @ axis)
    across = float(np.linalg.norm(offset - along * axis))
    return along, across


def along_band(along: float | None, *, attached_cm: float = ATTACHED_ALONG_CM) -> str | None:
    """`beyond`: anchor more than `attached_cm` past the track's tip end (a tip is probably
    attached); `within`: inside +-`attached_cm`; `behind`: the track overshoots by more."""
    if along is None:
        return None
    if along > attached_cm:
        return "beyond"
    if along < -attached_cm:
        return "behind"
    return "within"


def tip_error(anchor: TipAnchor, row: Mapping[str, Any]) -> dict[str, Any]:
    """Distance from the anchor's tip to the track's tip: `endpoints_cm[0]` when
    `tip_resolved`, the nearer endpoint when not (flagged), and `position_cm` for a point
    track (a midpoint, named as such). A line track's error is also split into `along_cm`
    (signed, along the track's axis, positive = anchor beyond the tip end) and `across_cm`
    (perpendicular); a point track with a `direction` gets the perpendicular distance to
    its axis (`axis_distance_cm`) and no signed split, since a midpoint has no tip end."""
    assert anchor.tip_cm is not None
    endpoints = row.get("endpoints_cm")
    out: dict[str, Any] = {
        "track_id": row.get("track_id"),
        "track_class": row.get("object_class"),
        "observed_class": row.get("observed_class"),
        "colour_identity": row.get("colour_identity"),
        "colour_confidence": row.get("colour_confidence"),
        "track_kind": "line" if endpoints else "point",
        "tip_resolved": row.get("tip_resolved"),
        "along_cm": None,
        "across_cm": None,
        "along_band": None,
        "other_end_nearer": None,
        "axis_distance_cm": None,
    }
    if endpoints:
        ends = [np.asarray(p, dtype=np.float64) for p in endpoints]
        distances = [float(np.linalg.norm(p - anchor.tip_cm)) for p in ends]
        if row.get("tip_resolved"):
            tip_index = 0
            out["measure"] = "tip_endpoint"
            # The anchor sits nearer the track's butt than its named tip: either the track
            # named the wrong end or the clicks sit at the butt. The number says which end,
            # not whose fault.
            out["other_end_nearer"] = bool(distances[1] < distances[0])
        else:
            tip_index = int(np.argmin(distances))
            out["measure"] = "nearer_endpoint_unresolved"
        out["tip_error_cm"] = round(distances[tip_index], 3)
        out["other_end_cm"] = round(distances[1 - tip_index], 3)
        split = along_across(anchor.tip_cm, ends[tip_index], ends[1 - tip_index])
        if split is not None:
            out["along_cm"] = round(split[0], 3)
            out["across_cm"] = round(split[1], 3)
            out["along_band"] = along_band(split[0])
        return out
    position = np.asarray(row["position_cm"], dtype=np.float64)
    out["tip_error_cm"] = round(float(np.linalg.norm(position - anchor.tip_cm)), 3)
    out["measure"] = "point_to_midpoint"
    direction = row.get("direction")
    if direction:
        d = np.asarray(direction, dtype=np.float64)
        norm = float(np.linalg.norm(d))
        if norm > 0:
            offset = anchor.tip_cm - position
            along = float(offset @ (d / norm))
            out["axis_distance_cm"] = round(float(np.linalg.norm(offset - along * d / norm)), 3)
    return out


def _percentiles(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "median_cm": None, "p90_cm": None, "max_cm": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "median_cm": round(float(np.median(arr)), 3),
        "p90_cm": round(float(np.percentile(arr, 90)), 3),
        "max_cm": round(float(arr.max()), 3),
    }


ERROR_MEASURES: tuple[str, ...] = ("raw", "along", "across")


def _error_summary(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """`raw` (the tip error), `along` (signed) and `across` percentiles over the matched
    cells that carry them, plus the along-axis bands and the attached fraction."""
    raw = [c["tip_error_cm"] for c in entries if c.get("tip_error_cm") is not None]
    along = [c["along_cm"] for c in entries if c.get("along_cm") is not None]
    across = [c["across_cm"] for c in entries if c.get("across_cm") is not None]
    bands = Counter(c["along_band"] for c in entries if c.get("along_band"))
    split = sum(bands.values())
    # The flat n / median_cm / p90_cm / max_cm are the raw error, so `by_state[s]["median_cm"]`
    # reads as before; the split sits beside them.
    return {
        **_percentiles(raw),
        "raw": _percentiles(raw),
        "along": _percentiles(along),
        "across": _percentiles(across),
        "along_bands": {band: bands.get(band, 0) for band in ("beyond", "within", "behind")},
        "attached_fraction": round(bands.get("beyond", 0) / split, 3) if split else None,
        "other_end_nearer": sum(1 for c in entries if c.get("other_end_nearer")),
    }


def read_colour_identities(
    path: Path, frames: Iterable[int]
) -> dict[tuple[int, str], dict[str, Any]]:
    """(frame, track_id) -> the colour vote's fields on the pipette rows of a
    `tracks_colour.jsonl` (written by `battle-finebio-colour annotate` beside the tracks)."""
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for frame, rows in read_track_rows(path, frames).items():
        for row in rows:
            if row.get("colour_identity") is None:
                continue
            out[(frame, str(row.get("track_id")))] = {
                # The vote names a colour (`yellow`); the identity keys name the class.
                "colour_identity": COLOUR_TO_CLASS.get(
                    str(row["colour_identity"]), str(row["colour_identity"])
                ),
                "colour_confidence": row.get("colour_confidence"),
            }
    return out


def colour_tracks_beside(path: Path) -> Path | None:
    candidate = Path(path).parent / COLOUR_TRACKS_NAME
    return candidate if candidate.is_file() and candidate != Path(path) else None


def _agreement(row: Mapping[str, Any], identity: str, into: Mapping[str, Counter]) -> None:
    for field_name, key in (
        ("colour_identity", "colour_identity"),
        ("observed_class", "observed_class"),
        ("track_class", "object_class"),
    ):
        value = row.get(key)
        if value is None or value == LINE_CLASS:
            into[field_name]["absent"] += 1
        else:
            into[field_name]["agree" if value == identity else "differ"] += 1


def score_tracks_file(
    name: str,
    path: Path,
    anchors: Sequence[TipAnchor],
    *,
    repository_root: Path,
    gate_cm: float = MATCH_GATE_CM,
    threshold_cm: float = THRESHOLD_MEDIAN_CM,
    colour_path: Path | None = None,
) -> dict[str, Any]:
    """Score one tracks file against the anchors. `colour_path` (default: a
    `tracks_colour.jsonl` beside the tracks, when one exists) supplies `colour_identity`
    for the matched rows. Identity is read two ways: from the keys I pressed
    (`pipette_idf1`, the designed measure, null when no key was pressed) and from the
    detector class of the clicked cells (`pipette_idf1_by_cell_class`, the fallback: it
    inherits the detector's class confusions and is named as such)."""
    frames = sorted({a.raw_frame for a in anchors})
    oriented = Path(path).with_name("tracks_oriented.jsonl")
    if path.name != "tracks_oriented.jsonl" and oriented.is_file():
        path = oriented
    rows_by_frame = read_track_rows(path, frames)
    if colour_path is None:
        colour_path = colour_tracks_beside(Path(path))
    colours = read_colour_identities(colour_path, frames) if colour_path else {}
    cells: list[dict[str, Any]] = []
    by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    measures: Counter[str] = Counter()
    identity_pairs: list[tuple[str, str | None]] = []
    class_pairs: list[tuple[str, str | None]] = []
    agreement = {
        "colour_identity": Counter(),
        "observed_class": Counter(),
        "track_class": Counter(),
    }
    class_agreement = {
        "colour_identity": Counter(),
        "observed_class": Counter(),
        "track_class": Counter(),
    }
    # Sep 29: my `t` / `b` label against the tracker's `tip_attached` on the matched row.
    label_agreement: Counter[str] = Counter()
    for anchor in anchors:
        row, match_by = match_track(
            anchor, rows_by_frame.get(anchor.raw_frame, []), gate_cm=gate_cm
        )
        entry: dict[str, Any] = {**anchor.as_dict(), "match": match_by, "track_id": None}
        entry["tracker_tip_attached"] = None
        entry["tip_label_agreement"] = None
        if row is not None:
            colour = colours.get((anchor.raw_frame, str(row.get("track_id"))))
            if colour and row.get("colour_identity") is None:
                row = {**row, **colour}
            entry["track_id"] = row.get("track_id")
            entry["tracker_tip_attached"] = row.get("tip_attached")
            if anchor.tip_label_consistent:
                tracker = row.get("tip_attached")
                verdict = (
                    "tracker_undecided"
                    if tracker is None
                    else "agree"
                    if bool(tracker) == anchor.tip_label
                    else "differ"
                )
                entry["tip_label_agreement"] = verdict
                label_agreement[verdict] += 1
                label_agreement[f"{'t' if anchor.tip_label else 'b'}_{verdict}"] += 1
            if anchor.tip_cm is not None:
                entry.update(tip_error(anchor, row))
                by_state[anchor.state].append(entry)
                by_class[anchor.cls].append(entry)
                measures[entry["measure"]] += 1
                _agreement(row, anchor.cls, class_agreement)
                if anchor.tip_label_consistent:
                    by_label["labelled"].append(entry)
                    by_label["t" if anchor.tip_label else "b"].append(entry)
            else:
                entry["measure"] = None
            if anchor.identity:
                _agreement(row, anchor.identity, agreement)
        elif anchor.tip_label_consistent:
            label_agreement["no_track"] += 1
        if anchor.identity:
            identity_pairs.append((anchor.identity, entry["track_id"]))
        if anchor.tip_cm is not None:
            class_pairs.append((anchor.cls, entry["track_id"]))
        cells.append(entry)
    with_tip = sum(1 for a in anchors if a.tip_cm is not None)
    matched = [c for c in cells if c.get("tip_error_cm") is not None]
    summary = _error_summary(matched)
    across = summary["across"]["median_cm"]
    labelled_total = sum(label_agreement[k] for k in ("agree", "differ", "tracker_undecided"))
    return {
        "tracks": name,
        "path": fs_common.relative_uri(Path(path), repository_root),
        "colour_tracks": None
        if colour_path is None
        else fs_common.relative_uri(Path(colour_path), repository_root),
        "anchors": with_tip,
        "matched": len(matched),
        "unmatched": with_tip - len(matched),
        "tip_error": summary["raw"],
        "along": summary["along"],
        "across": summary["across"],
        "along_bands": summary["along_bands"],
        "attached_fraction": summary["attached_fraction"],
        "attached_along_cm": ATTACHED_ALONG_CM,
        "other_end_nearer": summary["other_end_nearer"],
        "by_state": {state: _error_summary(by_state.get(state, [])) for state in ALL_STATES},
        "by_class": {cls: _error_summary(by_class.get(cls, [])) for cls in PIPETTE_CLASSES},
        # Sep 29: the anchors whose pipette-frame I labelled t or b on every view.
        "by_tip_label": {
            key: _error_summary(by_label.get(key, [])) for key in ("labelled", "t", "b")
        },
        "tip_label": {
            "labelled_anchors_with_a_track": labelled_total,
            "agree": label_agreement.get("agree", 0),
            "differ": label_agreement.get("differ", 0),
            "tracker_undecided": label_agreement.get("tracker_undecided", 0),
            "no_track": label_agreement.get("no_track", 0),
            "accuracy": (
                round(
                    label_agreement.get("agree", 0)
                    / (label_agreement.get("agree", 0) + label_agreement.get("differ", 0)),
                    3,
                )
                if label_agreement.get("agree", 0) + label_agreement.get("differ", 0)
                else None
            ),
            "by_label": {
                label: {
                    verdict: label_agreement.get(f"{label}_{verdict}", 0)
                    for verdict in ("agree", "differ", "tracker_undecided")
                }
                for label in ("t", "b")
            },
        },
        "measures": dict(measures),
        "tips_unresolved": measures.get("nearer_endpoint_unresolved", 0),
        "point_tracks": measures.get("point_to_midpoint", 0),
        "threshold_median_cm": threshold_cm,
        "median_under_threshold": (
            bool(summary["raw"]["median_cm"] < threshold_cm) if matched else None
        ),
        "median_across_under_threshold": (None if across is None else bool(across < threshold_cm)),
        "identity": {
            "pipette_idf1": idf1(identity_pairs),
            "agreement": {k: dict(v) for k, v in agreement.items()},
            "pipette_idf1_by_cell_class": idf1(class_pairs),
            "agreement_by_cell_class": {k: dict(v) for k, v in class_agreement.items()},
        },
        "cells": cells,
    }


def score_workspace(
    *,
    workspace_dir: Path,
    record_path: Path | None,
    tracks: Mapping[str, Path],
    cameras_at: Callable[[int], Mapping[str, Camera]],
    repository_root: Path,
    gate_cm: float = MATCH_GATE_CM,
    threshold_cm: float = THRESHOLD_MEDIAN_CM,
) -> dict[str, Any]:
    workspace = load_workspace(workspace_dir)
    record = load_record(record_path)
    anchors, dropped = triangulate_clicks(workspace, record, cameras_at)
    with_tip = [a for a in anchors if a.tip_cm is not None]
    residuals = [r for a in with_tip for r in a.residual_px.values()]
    residual_max = [max(a.residual_px.values()) for a in with_tip]
    cells = record.get("cells", [])
    dropped_kinds = Counter(
        "clicks_disagree"
        if d["reason"].startswith("clicks disagree")
        else "no_click"
        if d["reason"] == "no click"
        else "single_view"
        for d in dropped
    )
    return {
        "schema": f"{SCHEMA}/scoreboard",
        "anchor_kind": ANCHOR_KIND,
        "workspace": fs_common.relative_uri(Path(workspace_dir), repository_root),
        "record": None
        if record_path is None or not Path(record_path).is_file()
        else {
            "uri": fs_common.relative_uri(Path(record_path), repository_root),
            "sha256": fs_common.sha256_file(Path(record_path)),
            "author": record.get("author"),
            "updated_at": record.get("updated_at"),
        },
        "record_summary": {
            "cells": len(cells),
            "clicked": sum(1 for c in cells if c.get("tip_px") is not None),
            "hidden": sum(1 for c in cells if c.get("hidden")),
            "with_identity": sum(1 for c in cells if c.get("instance_identity")),
            "with_tip_label": sum(1 for c in cells if c.get("tip_attached_label") is not None),
        },
        "anchors": {
            "count": len(with_tip),
            "by_state": dict(Counter(a.state for a in with_tip)),
            "by_class": dict(Counter(a.cls for a in with_tip)),
            "by_tip_label": dict(
                Counter(
                    ("t" if a.tip_label else "b") if a.tip_label_consistent else "unsaid"
                    for a in with_tip
                )
            ),
            "pipette_frames_labelled": sum(1 for a in anchors if a.tip_label_consistent),
            "pipette_frames_labelled_inconsistently": sum(
                1 for a in anchors if a.tip_label_inconsistent
            ),
            "views_per_anchor": dict(Counter(len(a.views) for a in with_tip)),
            "reprojection_residual_px": {
                "median": round(float(np.median(residuals)), 2) if residuals else None,
                "p90": round(float(np.percentile(residuals, 90)), 2) if residuals else None,
                "max": round(float(max(residuals)), 2) if residuals else None,
            },
            "residual_max_per_anchor_px": {
                "median": round(float(np.median(residual_max)), 2) if residual_max else None,
                "p90": round(float(np.percentile(residual_max, 90)), 2) if residual_max else None,
                "max": round(float(max(residual_max)), 2) if residual_max else None,
            },
            "views_dropped_by_residual_gate": sum(1 for a in with_tip if a.dropped_view),
            "residual_gate_px": RESIDUAL_GATE_PX,
            "identity_only": len(anchors) - len(with_tip),
            "dropped_by_kind": {
                kind: dropped_kinds.get(kind, 0)
                for kind in ("single_view", "no_click", "clicks_disagree")
            },
            "dropped": dropped,
            "rows": [a.as_dict() for a in with_tip],
        },
        "threshold": {
            "rule": "pre-registered: median tip error under 2 cm on the clicked anchors",
            "median_cm": threshold_cm,
        },
        "arms": [
            score_tracks_file(
                name,
                path,
                anchors,
                repository_root=repository_root,
                gate_cm=gate_cm,
                threshold_cm=threshold_cm,
            )
            for name, path in tracks.items()
            if Path(path).is_file()
        ],
        "missing_tracks": [
            {"tracks": name, "path": str(path)}
            for name, path in tracks.items()
            if not Path(path).is_file()
        ],
        "generated_at": fs_common.run_timestamp(),
        "claim_boundary": CLAIM_BOUNDARY,
        "licence": LICENCE_NOTE,
    }


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _split_cell(summary: Mapping[str, Any]) -> str:
    """`raw / along / across (n)` medians of one error summary, `-` where absent."""
    raw = summary["raw"] if "raw" in summary else summary["tip_error"]
    return (
        f"{_fmt(raw['median_cm'])} / {_fmt(summary['along']['median_cm'])} / "
        f"{_fmt(summary['across']['median_cm'])} ({raw['n']})"
    )


def table_states(report: Mapping[str, Any]) -> tuple[str, ...]:
    """The state columns of the markdown: the states any arm has anchors in, in `ALL_STATES`
    order (the event state shows only for a workspace with event frames), else `STATES`."""
    present = tuple(
        state
        for state in ALL_STATES
        if any((arm["by_state"].get(state) or {}).get("n") for arm in report["arms"])
    )
    return present or STATES


def scoreboard_markdown(report: dict[str, Any]) -> str:
    anchors = report["anchors"]
    summary = report["record_summary"]
    residual = anchors["reprojection_residual_px"]
    kinds = anchors.get("dropped_by_kind", {})
    states = table_states(report)
    lines = [
        f"# Tip scoreboard: {anchors['count']} anchors from {summary['clicked']} clicks",
        "",
        f"Anchors by state: {json.dumps(anchors['by_state'])}; by class: "
        f"{json.dumps(anchors['by_class'])}; views per anchor: "
        f"{json.dumps(anchors['views_per_anchor'])}. Reprojection residual of the kept clicks "
        f"median {_fmt(residual['median'])} px, p90 {_fmt(residual['p90'])} px, max "
        f"{_fmt(residual['max'])} px; {anchors['views_dropped_by_residual_gate']} views dropped "
        f"by the {anchors['residual_gate_px']:g} px gate. Pipette-frames without an anchor: "
        f"{kinds.get('single_view', 0)} with one clicked view, {kinds.get('no_click', 0)} with "
        f"none, {kinds.get('clicks_disagree', 0)} whose clicks disagree beyond the gate. "
        f"{summary['hidden']} cells marked h (see the note below), {summary['with_identity']} "
        f"with an identity key, {summary.get('with_tip_label', 0)} with a t / b tip label "
        f"(anchors by label: {json.dumps(anchors.get('by_tip_label', {}))}).",
        "",
    ]
    if anchors["count"] == 0:
        lines.append(
            "No anchor yet: click two or three views of a pipette on a frame (battle-finebio-tips "
            "serve), then re-run."
        )
        lines.append("")
    header = [
        "tracks",
        "anchors matched",
        "tip error median cm",
        "p90 cm",
        "along median cm",
        "across median cm",
        "attached (along > 2 cm)",
        *(f"{state} median (n)" for state in states),
        "measure",
        "median < 2 cm raw",
        "median < 2 cm across",
        "pipette IDF1 keys",
        "IDF1 by cell class",
        "colour vote agrees / differs / absent (vs keys; vs cell class)",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for arm in report["arms"]:
        by_state = arm["by_state"]
        measures = ", ".join(f"{k} {v}" for k, v in sorted(arm["measures"].items())) or "-"
        keyed = arm["identity"]["agreement"]["colour_identity"]
        colour = arm["identity"]["agreement_by_cell_class"]["colour_identity"]
        bands = arm["along_bands"]
        split_n = sum(bands.values())
        lines.append(
            "| "
            + " | ".join(
                [
                    arm["tracks"],
                    f"{arm['matched']} / {arm['anchors']}",
                    _fmt(arm["tip_error"]["median_cm"]),
                    _fmt(arm["tip_error"]["p90_cm"]),
                    _fmt(arm["along"]["median_cm"]),
                    _fmt(arm["across"]["median_cm"]),
                    f"{bands['beyond']} / {split_n}" if split_n else "-",
                    *(
                        f"{_fmt(by_state[state]['median_cm'])} ({by_state[state]['n']})"
                        for state in states
                    ),
                    measures,
                    _fmt(arm["median_under_threshold"]),
                    _fmt(arm["median_across_under_threshold"]),
                    _fmt(arm["identity"]["pipette_idf1"]["idf1"], 3),
                    _fmt(arm["identity"]["pipette_idf1_by_cell_class"]["idf1"], 3),
                    f"{keyed.get('agree', 0)} / {keyed.get('differ', 0)} / "
                    f"{keyed.get('absent', 0)}; {colour.get('agree', 0)} / "
                    f"{colour.get('differ', 0)} / {colour.get('absent', 0)}",
                ]
            )
            + " |"
        )
    lines.append("")
    for missing in report.get("missing_tracks", []):
        lines.append(f"Tracks file not found: `{missing['path']}` ({missing['tracks']}).")
    if report.get("missing_tracks"):
        lines.append("")
    if any(arm["along"]["n"] for arm in report["arms"]):
        lines.append("## Tip error split: raw / along / across medians in cm (n)")
        lines.append("")
        lines.append(
            "| tracks | all | p90 raw / along / across | "
            + " | ".join(states)
            + " | "
            + " | ".join(IDENTITY_SHORT[c] for c in PIPETTE_CLASSES)
            + " | beyond / within / behind 2 cm | anchor nearer other end |"
        )
        lines.append("|" + "---|" * (3 + len(states) + len(PIPETTE_CLASSES) + 2))
        for arm in report["arms"]:
            if not arm["along"]["n"]:
                continue
            bands = arm["along_bands"]
            lines.append(
                f"| {arm['tracks']} | {_split_cell(arm)} | "
                f"{_fmt(arm['tip_error']['p90_cm'])} / {_fmt(arm['along']['p90_cm'])} / "
                f"{_fmt(arm['across']['p90_cm'])} | "
                + " | ".join(_split_cell(arm["by_state"][s]) for s in states)
                + " | "
                + " | ".join(_split_cell(arm["by_class"][c]) for c in PIPETTE_CLASSES)
                + f" | {bands['beyond']} / {bands['within']} / {bands['behind']} "
                f"| {arm['other_end_nearer']} |"
            )
        lines.append("")
    if any(
        (arm.get("tip_label") or {}).get("labelled_anchors_with_a_track") for arm in report["arms"]
    ):
        lines.append("## Disposable tip: my t / b label against the tracker's tip_attached")
        lines.append("")
        lines.append(
            "| tracks | labelled pipette-frames with a track | agree | differ | tracker undecided "
            "| accuracy (agree / decided) | t: agree / differ / undecided | b: agree / differ / "
            "undecided | tip error median cm on labelled anchors (n) | on t anchors (n) | on b "
            "anchors (n) |"
        )
        lines.append("|" + "---|" * 11)
        for arm in report["arms"]:
            label = arm.get("tip_label") or {}
            by = label.get("by_label") or {}
            errors = arm.get("by_tip_label") or {}

            def err(key: str) -> str:
                block = errors.get(key) or {}
                return f"{_fmt(block.get('median_cm'))} ({block.get('n', 0)})"

            lines.append(
                f"| {arm['tracks']} | {label.get('labelled_anchors_with_a_track', 0)} | "
                f"{label.get('agree', 0)} | {label.get('differ', 0)} | "
                f"{label.get('tracker_undecided', 0)} | {_fmt(label.get('accuracy'), 3)} | "
                + " / ".join(
                    str(by.get("t", {}).get(k, 0)) for k in ("agree", "differ", "tracker_undecided")
                )
                + " | "
                + " / ".join(
                    str(by.get("b", {}).get(k, 0)) for k in ("agree", "differ", "tracker_undecided")
                )
                + f" | {err('labelled')} | {err('t')} | {err('b')} |"
            )
        lines.append("")
    if anchors.get("dropped"):
        lines.append("## Pipette-frames without an anchor")
        lines.append("")
        lines.append("| frame | pipette | reason |")
        lines.append("|---|---|---|")
        for item in anchors["dropped"]:
            pipette = IDENTITY_SHORT.get(item["class"], item["class"])
            lines.append(f"| {item['raw_frame']} | {pipette} | {item['reason']} |")
        lines.append("")
    for arm in report["arms"]:
        lines.append(f"## {arm['tracks']}: per pipette-frame")
        lines.append("")
        lines.append(
            "| frame | state | pipette | views | residual max px | track | match | tip error cm "
            "| along cm | across cm | measure | axis cm |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for cell in arm["cells"]:
            if cell["tip_cm"] is None:
                continue
            pipette = IDENTITY_SHORT.get(cell["class"], cell["class"])
            measure = cell.get("measure") or "-"
            if cell.get("other_end_nearer"):
                measure += " (other end nearer)"
            lines.append(
                f"| {cell['raw_frame']} | {cell['state']} | {pipette} "
                f"| {','.join(cell['views'])} | {_fmt(cell['residual_max_px'])} | "
                f"{cell['track_id'] or '-'} | {cell['match']} | "
                f"{_fmt(cell.get('tip_error_cm'))} | {_fmt(cell.get('along_cm'))} | "
                f"{_fmt(cell.get('across_cm'))} | {measure} | "
                f"{_fmt(cell.get('axis_distance_cm'))} |"
            )
        lines.append("")
    lines.append(
        "tip error = distance in cm from the triangulated click to the track's tip: "
        "`endpoints_cm[0]` when `tip_resolved` (tip_endpoint), the nearer endpoint when the "
        "track has not resolved which end is the tip (nearer_endpoint_unresolved), and "
        "`position_cm` for a point track (point_to_midpoint: the box-centre tracker's point is "
        "the middle of the shaft, so this distance is expected near half a pipette length and is "
        "not a tip error). along cm = the signed part of that error along the track's axis, "
        "positive when the anchor lies beyond the track's tip end (the mask stopped short: a "
        "disposable tip is attached and SAM3 stops at the body's cone, so a steady 3-7 cm here is "
        "the tip's length, not a geometry error); across cm = the perpendicular part; attached = "
        f"anchors with along > {ATTACHED_ALONG_CM:g} cm over anchors with a split. across is the "
        "error with the whole along-axis part removed, so it is the generous reading; raw is the "
        "pre-registered one. other end nearer = a resolved track whose other end is nearer the "
        "anchor than its named tip (along is then about minus one pipette length): the track "
        "named the wrong end, or the clicks sit at the butt because I accepted a suggested marker "
        "that sat on the plunger end, and the number does not say which. marked h = cells with "
        "no click: the tip not visible, or, when I click only to accept a suggested marker that "
        "sits on the tip, the marker not on the tip. It is not a count of hidden tips. "
        "axis cm = for a point track with a direction, the perpendicular "
        "distance from the anchor to that axis. pipette IDF1 keys = identity F1 between the "
        "plunger colour I named and the track ids over the named pipette-frames; by cell class = "
        "the same with the detector's class of the clicked cell standing in for my key, which "
        "inherits the detector's class confusions. t / b label = my key per pipette per frame "
        "(a disposable tip IS attached / bare), read against the tracker's tip_attached on the "
        "matched row; a pipette-frame counts only when every view cell carries the same label; "
        "the labelled tip error is the same raw error over those anchors alone, split by the "
        "label. " + report["claim_boundary"]
    )
    lines.append("")
    return "\n".join(lines)


def run_score(args: argparse.Namespace) -> dict[str, Any]:
    root = Path.cwd().resolve()
    clip = load_clip(args.clip_config)
    tracks = parse_tracks(args.tracks)
    record_path = Path(args.record) if args.record else Path(args.workspace) / RECORD_NAME
    report = score_workspace(
        workspace_dir=Path(args.workspace),
        record_path=record_path,
        tracks=tracks,
        cameras_at=cameras_from_clip(clip, root),
        repository_root=root,
        gate_cm=args.match_gate_cm,
    )
    output = Path(args.output) if args.output else Path(args.workspace) / "scoreboard"
    output.mkdir(parents=True, exist_ok=True)
    fs_common.write_json(output / "tip_scoreboard.json", report)
    table = scoreboard_markdown(report)
    (output / "tip_scoreboard.md").write_text(table, encoding="utf-8")
    print(table, end="")
    print(f"Report: {output / 'tip_scoreboard.json'}")
    return report


# --------------------------------------------------------------------------------------------
# export: the committed human record (no pixels)


def export_record(
    *,
    workspace_dir: Path,
    record_path: Path,
    output: Path,
    repository_root: Path,
    protocol: str | None = None,
) -> dict[str, Any]:
    workspace = load_workspace(workspace_dir)
    if not Path(record_path).is_file():
        raise FileNotFoundError(f"no decisions file at {record_path}; nothing to export")
    record = load_record(record_path)
    decisions = {cell_key(c): c for c in record.get("cells", [])}
    entries = []
    for cell in workspace["cells"]:
        entry = decisions.get(cell_key(cell), {})
        entries.append(
            {
                "raw_frame": cell["raw_frame"],
                "proxy_frame": cell["proxy_frame"],
                "view": cell["view"],
                "slot": cell["slot"],
                "class": cell["class"],
                "label": cell["label"],
                "state": cell["state"],
                "tip_px": entry.get("tip_px"),
                "clicked_in": entry.get("clicked_in"),
                "hidden": bool(entry.get("hidden", False)),
                "instance_identity": entry.get("instance_identity"),
                "tip_attached_label": entry.get("tip_attached_label"),
                "note": entry.get("note") or "",
            }
        )
    counts = {
        "cells": len(entries),
        "clicked": sum(1 for e in entries if e["tip_px"] is not None),
        "hidden": sum(1 for e in entries if e["hidden"]),
        "undecided": sum(1 for e in entries if e["tip_px"] is None and not e["hidden"]),
        "with_identity": sum(1 for e in entries if e["instance_identity"]),
        "with_tip_label": sum(1 for e in entries if e["tip_attached_label"] is not None),
        "frames": len({e["raw_frame"] for e in entries}),
        "frames_by_state": dict(Counter(f["state"] for f in workspace["frames"])),
    }
    doc = {
        "schema": f"{SCHEMA}/human-record",
        "anchor_kind": ANCHOR_KIND,
        "author": record.get("author"),
        "reviewed_at": record.get("reviewed_at"),
        "trial": workspace["trial"],
        "workspace": fs_common.relative_uri(Path(workspace_dir), repository_root),
        "cells_json": {
            "uri": fs_common.relative_uri(Path(workspace_dir) / CELLS_NAME, repository_root),
            "sha256": fs_common.sha256_file(Path(workspace_dir) / CELLS_NAME),
        },
        "clip_config": workspace["clip_config"],
        "observations": workspace["observations"],
        "decisions": {
            "uri": fs_common.relative_uri(Path(record_path), repository_root),
            "sha256": fs_common.sha256_file(Path(record_path)),
        },
        "frame_convention": workspace["frame_convention"],
        "protocol": protocol,
        "counts": counts,
        "clicks": entries,
        "claim_boundary": CLAIM_BOUNDARY,
        "licence": LICENCE_NOTE,
        "notes": (
            "Written by battle-finebio-tips export. tip_px is the clicked full-frame pixel (raw "
            "video pixels of the view), never the suggested tip; author and reviewed_at come from "
            "the decisions file and stay null until filled in. protocol, when given, is how I "
            "chose between a click and h in this session, and so what hidden means here. Crops "
            "stay under runs/."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    fs_common.write_json(output, doc)
    return doc


# --------------------------------------------------------------------------------------------
# CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Tip clicks on about 30 frames: choose frames and cut crops, serve the click pages, "
            "triangulate the clicks and score trackers' tips in cm, export the human record."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="Choose frames, cut crops, write cells.json (CPU).")
    prep.add_argument("--trial", default=None, help="checked against the clip config")
    prep.add_argument(
        "--observations",
        type=Path,
        required=True,
        help="observations.jsonl (or its directory) with mask axes on the SAM3 rows",
    )
    prep.add_argument("--clip-config", type=Path, required=True)
    prep.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument(
        "--tracks",
        type=Path,
        default=None,
        help="line tracks.jsonl; when it exists the low state reads resolved tip heights",
    )
    prep.add_argument("--pipettes", type=Path, default=DEFAULT_PIPETTES_CONFIG)
    prep.add_argument("--spacing", type=int, default=DEFAULT_SPACING_FRAMES)
    prep.add_argument("--move-cm-per-frame", type=float, default=DEFAULT_MOVE_CM_PER_FRAME)
    prep.add_argument("--views-per-cell", type=int, default=DEFAULT_VIEWS_PER_CELL)
    prep.add_argument(
        "--all-pipettes",
        action="store_true",
        help="cells for every pipette on held and low frames too (default: the active ones)",
    )
    prep.add_argument(
        "--no-marker",
        action="store_true",
        help="Sep 29: no suggested marker; the crop shows the whole pipette and an arrow at "
        "its edge, labelled tracker's guess, points at the rows' long-thin-tail end",
    )
    prep.add_argument(
        "--single-channel-only",
        action="store_true",
        help="Sep 29: cells for the blue, yellow and red pipettes only (no 8-channel)",
    )
    prep.add_argument(
        "--event-frames",
        default=None,
        help="Sep 29: comma-separated raw frames of a pipette's tip events; --event-quota "
        "frames are chosen within --event-radius of them, on frames where --event-class "
        "has a cell",
    )
    prep.add_argument("--event-class", default=None, help="the pipette of --event-frames")
    prep.add_argument("--event-quota", type=int, default=10)
    prep.add_argument("--event-radius", type=int, default=DEFAULT_EVENT_RADIUS_FRAMES)
    prep.add_argument(
        "--states",
        default=None,
        help="Sep 29: comma-separated states that share the frames left after the event "
        "quota, in order (default low,held,rest)",
    )

    serve = sub.add_parser("serve", help="Serve the click pages (headless; loopback or tailnet).")
    serve.add_argument("--workspace", type=Path, required=True)
    serve.add_argument(
        "--record",
        type=Path,
        default=None,
        help=f"decisions file (default <workspace>/{RECORD_NAME}; created from the template)",
    )
    serve.add_argument("--author", default=default_author(), help="written into the record")
    serve.add_argument("--port", type=int, default=DEFAULT_PORT, help="0 picks a free port")
    where = serve.add_mutually_exclusive_group()
    where.add_argument(
        "--bind", default=DEFAULT_HOST, help=f"address to bind (default {DEFAULT_HOST})"
    )
    where.add_argument(
        "--tailscale",
        action="store_true",
        help="bind this machine's Tailscale IPv4 (`tailscale ip -4`) for the tailnet",
    )

    score = sub.add_parser("score", help="Triangulate the clicks and score tracks files' tips.")
    score.add_argument("--workspace", type=Path, required=True)
    score.add_argument(
        "--record", type=Path, default=None, help=f"default <workspace>/{RECORD_NAME}"
    )
    score.add_argument(
        "--tracks",
        action="append",
        default=[],
        required=True,
        metavar="[NAME=]TRACKS.JSONL",
        help="a tracks.jsonl to score; repeatable",
    )
    score.add_argument("--clip-config", type=Path, required=True)
    score.add_argument("--match-gate-cm", type=float, default=MATCH_GATE_CM)
    score.add_argument("--output", type=Path, default=None, help="default <workspace>/scoreboard")

    export = sub.add_parser("export", help="Write the committed human record (no pixels).")
    export.add_argument("--workspace", type=Path, required=True)
    export.add_argument("--record", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument(
        "--protocol",
        default=None,
        help=(
            "one paragraph on how I chose between a click and h in this session (what hidden "
            "means); written as the record's `protocol` field"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "serve":
        return run_serve(args, parser)
    try:
        if args.command == "prepare":
            run_prepare(args)
        elif args.command == "score":
            run_score(args)
        elif args.command == "export":
            doc = export_record(
                workspace_dir=Path(args.workspace),
                record_path=Path(args.record),
                output=Path(args.output),
                repository_root=Path.cwd().resolve(),
                protocol=args.protocol,
            )
            print(f"Human record: {args.output} ({json.dumps(doc['counts'])})")
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
