"""`battle-finebio-tipseg`: where the disposable tip ends, five ways, on trial 1's pipettes.

The line tracker needs one observation per view that is the end of the pipette *with* its
tip, and SAM3's body mask (arm b) usually stops at the cone. This lane tries five methods on
the same rows and compares them label-free and against the 138 clicked tip anchors:

* A. Detector tip boxes: a `*_tip` box whose centre lies within the gate of the axis line and
  beyond the body end (`multiview_lines.attach_tip_box`). Tip end = far edge, junction = near
  edge. Costs nothing, but the detector rarely labels tips.
* B. Width profile of the existing mask along its axis: the across-axis width every 2 px; a
  sustained fall under a fraction of the body width is the junction, and a tail beyond it
  means SAM3's mask already includes (some of) the tip.
* C. Intensity walk in the proxy frame: from the mask's thin end outward along the axis, a
  bright, low-saturation run against a local background sampled 15 px either side of the
  axis; the run ends where the brightness falls back to background.
* D. SAM3 box prompts on the tip in the same encoded image: the tip box from A, or a slim box
  6 cm-equivalent beyond the body end (`D`), and the union box minus the body mask (`D2`).
  GPU, the arms' worker in box-decode mode, one encode per frame and view.
* E. Fusion: per row, the methods that see a tip vote along the axis (agreement within 10 px,
  confidence-weighted); per frame and pipette, the fused ends are triangulated across views
  and the 3D tip length read against the body.

Every method returns, per pipette row, `tip_present` (True / False / None = abstain),
`tip_end_px`, `junction_px`, `tip_length_px`, `confidence`, and `end_px`, the method's end of
the pipette (the tip end when it sees one, else the body end). Frames are raw indices, pixels
raw video pixels. Three steps: `prepare` (working set, A, B, C, the box streams for D), `sam3`
(the worker, GPU) and `score` (D, E, the scoreboard, sheets and README). Nothing under
`runs/` is committed (FineBio licence).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from . import fs_common
from .contact_sheet import render_grid
from .finebio_cameras import Camera
from .finebio_observations import (
    filter_mask_components,
    read_mask,
    terminal_centroids,
    tip_side_from_tails,
)
from .finebio_slice import triangulate_pixels
from .finebio_tips import (
    HAND_CLASSES,
    PIPETTE_CLASSES,
    ProxyFrames,
    cameras_from_clip,
    load_clip,
    point_box_distance,
)
from .multiview_lines import attach_tip_box

SCHEMA = "battle-finebio-tipseg/1"
METHODS: tuple[str, ...] = ("A", "B", "C", "D", "D2", "E")
METHOD_NAMES: dict[str, str] = {
    "A": "detector tip box",
    "B": "mask width profile",
    "C": "intensity walk",
    "D": "SAM3 tip prompt",
    "D2": "SAM3 union minus body",
    "E": "fusion",
}
SINGLE_CHANNEL: tuple[str, ...] = ("blue_pipette", "yellow_pipette", "red_pipette")
MULTI_CHANNEL = "8_channel_pipette"
TIP_CLASS: dict[str, str] = {
    "blue_pipette": "blue_tip",
    "yellow_pipette": "yellow_tip",
    "red_pipette": "red_tip",
    "8_channel_pipette": "8_channel_tip",
}
TIP_CLASSES: tuple[str, ...] = tuple(TIP_CLASS.values())
RACK_CLASSES: tuple[str, ...] = (
    "blue_tip_rack",
    "yellow_tip_rack",
    "red_tip_rack",
    "8_channel_tip_rack",
)
TRASH_CLASS = "trash_can"
EVENT_CLASSES: tuple[str, ...] = (*RACK_CLASSES, TRASH_CLASS)
FPV_VIEW = "fpv"
FIXED_VIEWS: tuple[str, ...] = ("T1", "T2", "T3", "T4", "T5")
SAM3_SOURCES = ("sam3_decode", "sam3_video")
DEFAULT_BODY_LENGTH_CM = 23.21
TIP_LENGTH_CM = 5.0
TIP_LENGTH_RANGE_CM = (3.0, 7.0)
# Method A: the box centre within this of the axis line (the rig's association gate order).
BOX_GATE_PX = 30.0
# Method B: profile step, the junction fraction of the body width, the shortest tail that
# counts as a tip, and the outer band of the axis whose width names the thin end.
PROFILE_STEP_PX = 2.0
JUNCTION_FRACTION = 0.4
MIN_TAIL_CM = 1.0
END_BAND = 0.15
TAIL_SIDE_RATIO = 1.5
# Method C: walk up to WALK_FACTOR x the expected tip; background at BACKGROUND_OFFSET_PX either
# side of the axis; a step is on the tip when it is CONTRAST_MIN brighter than the background and
# below SATURATION_MAX (OpenCV 0-255 HSV); the run ends after WALK_GAP_PX steps off it.
WALK_FACTOR = 1.5
BACKGROUND_OFFSET_PX = 15
ON_AXIS_HALF_WIDTH_PX = 2
CONTRAST_MIN = 25.0
SATURATION_MAX = 110.0
WALK_GAP_PX = 4
MIN_RUN_CM = 1.0
# Method D: the slim box beyond the body end when no detector tip box attached.
SLIM_BOX_CM = 6.0
SLIM_BOX_MIN_HALF_WIDTH_PX = 8.0
TIP_BOX_PAD_PX = 4.0
MIN_TIP_MASK_PX = 30
MIN_TIP_EXTENT_PX = 8.0
# A SAM3 mask beyond the body end counts as a tip only when it is thin (across width at most
# this many cm-equivalent), elongated (extent over width) and starts at the body end (within
# this many cm of it): a box on empty bench comes back as a blob the size of the box.
TIP_MAX_WIDTH_CM = 1.2
TIP_MIN_ELONGATION = 3.0
TIP_ATTACH_GAP_CM = 0.75
# Two-state: a frame's body-plus-tip length this far over the body prior carries a tip.
TIP_ATTACHED_MARGIN_CM = 2.0
# Method E: two methods agree when their ends sit within this along the axis; a lone method
# needs this confidence to carry a row; the anchors' reprojection gate.
AGREE_PX = 10.0
LONE_CONFIDENCE = 0.6
RESIDUAL_GATE_PX = 30.0
# The working set: every click frame plus this stride over the window.
DEFAULT_STRIDE = 10
# The timeline: a rack or trash box counts as near the tip end within this.
NEAR_BOX_PX = 80.0
MASK_BAND_FACTOR = 1.5
SHEET_CELLS = 20
SHEET_CROP_PX = 280
METHOD_COLOURS_BGR: dict[str, tuple[int, int, int]] = {
    "A": (60, 60, 255),
    "B": (80, 220, 80),
    "C": (0, 230, 255),
    "D": (255, 200, 0),
    "D2": (255, 120, 120),
    "E": (255, 0, 230),
}
CLICK_COLOUR_BGR = (255, 255, 255)
CLAIM_BOUNDARY = (
    "Every number here is one method against another on SAM3's body masks and the detector's "
    "boxes, or a method against my 138 clicks on the tip end (one person, 30 frames, two or "
    "three views each, triangulated through the rig). Nothing is ground truth; the clicks "
    "accepted a suggested marker and the 8-channel clicks often sat on the plunger."
)
LICENCE_NOTE = (
    "FineBio is licensed for non-commercial research; frames, crops and masks derived from "
    "it stay under runs/ and are never committed or redistributed."
)

Point = tuple[float, float]
Box = tuple[float, float, float, float]


# --------------------------------------------------------------------------------------------
# records


@dataclass
class PipetteRow:
    """One SAM3 pipette row with an axis: what every method starts from."""

    view: str
    frame: int
    slot: str
    cls: str
    axis: list[list[float]]
    width_px: float
    elongation: float | None
    mask_bbox: list[float]
    centroid: list[float]
    box: list[float] | None
    detector_score: float | None
    sam3_score: float | None
    mask_uri: str | None
    pose_valid: bool = True
    # Filled by `prepare`: the axis end index the tip sits at, how it was chosen, the pixel
    # scale of this view at the pipette's depth and the end widths of the mask profile.
    tip_side: int | None = None
    tip_side_rule: str | None = None
    px_per_cm: float | None = None
    end_widths_px: list[float] | None = None
    # The body end every method starts from: the centroid of the mask's last pixels along
    # the axis on the tip side (the axis endpoint itself can sit off a thin tip sideways).
    body_end_px: list[float] | None = None

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.view, self.frame, self.slot)


@dataclass
class TipEstimate:
    """One method's answer on one row (module docstring)."""

    method: str
    view: str
    frame: int
    slot: str
    cls: str
    tip_present: bool | None
    tip_end_px: list[float] | None
    junction_px: list[float] | None
    tip_length_px: float | None
    confidence: float
    end_px: list[float]
    side: int | None
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.view, self.frame, self.slot)


@dataclass
class AxisFrame:
    """The row's axis as a 1D coordinate: `tip_end` is the tip-side axis end, `unit` points
    outward from the body through it, `along` is signed (positive beyond the body end)."""

    tip_end: np.ndarray
    other_end: np.ndarray
    unit: np.ndarray
    normal: np.ndarray
    length: float

    @classmethod
    def from_axis(
        cls,
        axis: Sequence[Sequence[float]],
        side: int,
        body_end: Sequence[float] | None = None,
    ) -> AxisFrame:
        """The direction is the axis's; the origin is `body_end` when given (the mask's
        terminal centroid), else the axis endpoint on `side`."""
        ends = np.asarray(axis, dtype=np.float64).reshape(2, 2)
        tip_end, other = ends[side], ends[1 - side]
        segment = tip_end - other
        length = float(np.linalg.norm(segment))
        unit = segment / length if length > 1e-9 else np.array([0.0, 1.0])
        if body_end is not None:
            tip_end = np.asarray(body_end, dtype=np.float64).reshape(2)
        return cls(tip_end, other, unit, np.array([-unit[1], unit[0]]), length)

    def along(self, point: Sequence[float]) -> float:
        return float((np.asarray(point, dtype=np.float64) - self.tip_end) @ self.unit)

    def across(self, point: Sequence[float]) -> float:
        return float((np.asarray(point, dtype=np.float64) - self.tip_end) @ self.normal)

    def point(self, t: float, a: float = 0.0) -> np.ndarray:
        return self.tip_end + t * self.unit + a * self.normal


def _pt(point: np.ndarray | Sequence[float]) -> list[float]:
    return [round(float(v), 1) for v in np.asarray(point, dtype=np.float64).reshape(2)]


def _row_axis(row: PipetteRow) -> AxisFrame:
    side = row.tip_side if row.tip_side is not None else 1
    return AxisFrame.from_axis(row.axis, side, row.body_end_px)


def terminal_centroid(
    mask: np.ndarray, axis: Sequence[Sequence[float]], side: int, depth_px: float = 3.0
) -> list[float] | None:
    """The centroid of the mask pixels within `depth_px` of the far extreme along the axis
    on `side`: the body end on the tip itself rather than on the fitted axis line (v4: the
    observation layer's `terminal_centroids`, one end of it)."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    both = terminal_centroids(np.column_stack([xs + 0.5, ys + 0.5]), axis, depth_px)
    return None if both is None else list(both[side])


def _body_end_estimate(
    method: str, row: PipetteRow, present: bool | None, **detail: Any
) -> TipEstimate:
    """A method's row when it sees no tip: the end is the body's thin end."""
    frame = _row_axis(row)
    return TipEstimate(
        method,
        row.view,
        row.frame,
        row.slot,
        row.cls,
        present,
        None,
        None,
        None,
        0.0,
        _pt(frame.tip_end),
        row.tip_side,
        detail,
    )


# --------------------------------------------------------------------------------------------
# inputs: the working set


def click_frames(workspace: Mapping[str, Any]) -> list[int]:
    return sorted({int(f["raw_frame"]) for f in workspace["frames"]})


def working_frames(
    click: Iterable[int], window: tuple[int, int], stride: int = DEFAULT_STRIDE
) -> list[int]:
    """Every click frame plus a stride sweep of the window."""
    start, end = window
    return sorted(set(click) | set(range(start, end, stride)))


def _frame_of_line(line: str) -> int | None:
    marker = '"frame_index":'
    at = line.find(marker)
    if at < 0:
        return None
    start = at + len(marker)
    stop = line.find(",", start)
    try:
        return int(line[start : stop if stop > 0 else None])
    except ValueError:
        return None


@dataclass
class FrameTables:
    """The rows of the working frames, by (view, frame): SAM3 pipette rows with an axis,
    the `*_tip` boxes, the hand boxes and the rack / trash boxes."""

    pipettes: dict[tuple[str, int], list[dict[str, Any]]]
    tips: dict[tuple[str, int], list[dict[str, Any]]]
    hands: dict[tuple[str, int], list[dict[str, Any]]]
    events: dict[tuple[str, int], list[dict[str, Any]]]
    pipettes_without_axis: int = 0


def read_frame_tables(path: Path, frames: Iterable[int]) -> FrameTables:
    """One pass over the observation file, a text prefilter on the frame before the parse."""
    wanted = {int(f) for f in frames}
    tables = FrameTables(defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list))
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            frame = _frame_of_line(line)
            if frame is None or frame not in wanted:
                continue
            row = json.loads(line)
            cls = row.get("object_class")
            key = (str(row["view"]), int(row["frame_index"]))
            source = row.get("source")
            if source in SAM3_SOURCES:
                if cls not in PIPETTE_CLASSES:
                    continue
                if row.get("mask_axis_px") is None:
                    tables.pipettes_without_axis += 1
                    continue
                tables.pipettes[key].append(row)
            elif source == "detector" and row.get("box_xyxy_px") is not None:
                if cls in TIP_CLASSES:
                    tables.tips[key].append(row)
                elif cls in HAND_CLASSES:
                    tables.hands[key].append(row)
                elif cls in EVENT_CLASSES:
                    tables.events[key].append(row)
    return tables


def mask_uri_for(row: Mapping[str, Any], worker_runs: Mapping[str, str], offset: int) -> str | None:
    """The worker's mask PNG for a SAM3 row, from its provenance (`object_id` ``sam3-NN``)
    and the view's worker run, as `finebio_observations.worker_to_observations` names it."""
    run = worker_runs.get(str(row["view"]))
    object_id = str((row.get("provenance") or {}).get("object_id", ""))
    tail = object_id.rsplit("-", 1)[-1]
    if run is None or not tail.isdigit():
        return None
    analysis = int(row["frame_index"]) - offset
    return f"{run}/masks/{analysis:06d}_{int(tail):02d}.png"


def pipette_rows(
    tables: FrameTables, worker_runs: Mapping[str, str], offset: int
) -> list[PipetteRow]:
    rows: list[PipetteRow] = []
    for (view, frame), items in sorted(tables.pipettes.items()):
        for row in items:
            if view == FPV_VIEW and not row.get("pose_valid", True):
                continue
            rows.append(
                PipetteRow(
                    view=view,
                    frame=frame,
                    slot=str(row["slot"]),
                    cls=str(row["object_class"]),
                    axis=[[float(v) for v in e] for e in row["mask_axis_px"]],
                    width_px=float(row.get("mask_width_px") or 0.0),
                    elongation=row.get("mask_elongation"),
                    mask_bbox=[float(v) for v in row["mask_bbox_px"]],
                    centroid=[float(v) for v in (row.get("mask_centroid_px") or (0.0, 0.0))],
                    box=None if row.get("box_xyxy_px") is None else list(row["box_xyxy_px"]),
                    detector_score=row.get("detector_score"),
                    sam3_score=row.get("sam3_object_score"),
                    mask_uri=mask_uri_for(row, worker_runs, offset),
                    pose_valid=bool(row.get("pose_valid", True)),
                )
            )
    return rows


# --------------------------------------------------------------------------------------------
# scale: pixels per centimetre at the pipette's depth


def focal_px(cam: Camera) -> float:
    return float(0.5 * (cam.K[0, 0] + cam.K[1, 1]))


def depth_cm(cam: Camera, point: np.ndarray) -> float:
    return float((cam.R @ np.asarray(point, dtype=np.float64).reshape(3) + cam.tvec)[2])


def frame_scales(
    rows: Sequence[PipetteRow], cams: Mapping[str, Camera]
) -> dict[tuple[str, int, str], tuple[float, str]]:
    """(view, frame, class) -> (px per cm, source). With two or more fixed views of the class
    on the frame the centroids are triangulated and the depth is the point's; else the depth
    to the bench origin stands in (`bench_origin`)."""
    by_frame_class: dict[tuple[int, str], list[PipetteRow]] = defaultdict(list)
    for row in rows:
        by_frame_class[(row.frame, row.cls)].append(row)
    out: dict[tuple[str, int, str], tuple[float, str]] = {}
    origin = np.zeros(3)
    for (frame, cls), group in by_frame_class.items():
        fixed = [r for r in group if r.view in FIXED_VIEWS and r.view in cams]
        point: np.ndarray | None = None
        if len(fixed) >= 2:
            point = triangulate_pixels(
                [cams[r.view] for r in fixed],
                [np.asarray(r.centroid, dtype=np.float64) for r in fixed],
            )
            if not np.all(np.isfinite(point)) or np.linalg.norm(point) > 300.0:
                point = None
        for row in group:
            cam = cams.get(row.view)
            if cam is None:
                continue
            target, source = (
                (point, "triangulated_centroids") if point is not None else (origin, "bench_origin")
            )
            depth = depth_cm(cam, target)
            if depth <= 10.0:
                depth, source = depth_cm(cam, origin), "bench_origin"
            out[(row.view, frame, cls)] = (focal_px(cam) / depth, source)
    return out


# --------------------------------------------------------------------------------------------
# method B: the width profile, and the thin end


@dataclass
class WidthProfile:
    """Across-axis width every `step` px along the axis from end 0 to end 1 of the row's
    `axis`; `t` is the bin centre's distance from end 0."""

    t: np.ndarray
    width: np.ndarray
    step: float

    @property
    def body_width(self) -> float:
        return float(np.percentile(self.width, 75)) if self.width.size else 0.0

    def end_widths(self, band: float = END_BAND) -> tuple[float, float]:
        n = max(1, int(round(band * self.width.size)))
        return float(np.median(self.width[:n])), float(np.median(self.width[-n:]))


def width_profile(
    mask: np.ndarray, axis: Sequence[Sequence[float]], step: float = PROFILE_STEP_PX
) -> WidthProfile | None:
    ends = np.asarray(axis, dtype=np.float64).reshape(2, 2)
    segment = ends[1] - ends[0]
    length = float(np.linalg.norm(segment))
    ys, xs = np.nonzero(mask)
    if xs.size == 0 or length < step:
        return None
    unit = segment / length
    normal = np.array([-unit[1], unit[0]])
    rel = np.column_stack([xs + 0.5, ys + 0.5]) - ends[0]
    along, across = rel @ unit, rel @ normal
    count = max(1, int(math.ceil(length / step)))
    index = np.clip(np.floor(along / step).astype(int), 0, count - 1)
    low = np.full(count, np.inf)
    high = np.full(count, -np.inf)
    filled = np.zeros(count, dtype=np.int64)
    np.minimum.at(low, index, across)
    np.maximum.at(high, index, across)
    np.add.at(filled, index, 1)
    keep = filled >= 2
    if not keep.any():
        return None
    t = (np.arange(count) + 0.5) * step
    return WidthProfile(t[keep], high[keep] - low[keep] + 1.0, step)


@dataclass
class Junction:
    present: bool
    tail_px: float
    junction_t: float
    body_width: float
    tail_width: float
    # The median width over the outermost `terminal_px` of the tail: a tip narrows to a
    # point, a bare cone or shaft does not.
    terminal_width: float


def junction_from_profile(
    profile: WidthProfile,
    side: int,
    *,
    fraction: float = JUNCTION_FRACTION,
    min_tail_px: float,
    terminal_px: float = 10.0,
) -> Junction:
    """From the tip-side end inward: bins under `fraction` of the body width are the tail; the
    tail ends at the first two consecutive bins at or over it. A tail at least `min_tail_px`
    long is a tip inside the mask, and `junction_t` its distance from the tip-side end."""
    body = profile.body_width
    # Walk the bins from the tip-side end inward.
    widths = profile.width[::-1] if side == 1 else profile.width
    limit = fraction * body
    tail = 0
    for i, width in enumerate(widths):
        if width >= limit and (i + 1 >= len(widths) or widths[i + 1] >= limit):
            break
        tail = i + 1
    terminal_bins = max(1, int(math.ceil(terminal_px / profile.step)))
    terminal = float(np.median(widths[:terminal_bins])) if widths.size else 0.0
    if tail == 0:
        return Junction(False, 0.0, 0.0, body, float(widths[0]) if widths.size else 0.0, terminal)
    tail_px = tail * profile.step
    tail_width = float(np.median(widths[:tail]))
    return Junction(tail_px >= min_tail_px, tail_px, tail_px, body, tail_width, terminal)


def choose_tip_side(
    row: PipetteRow,
    profile: WidthProfile | None,
    hands: Sequence[Mapping[str, Any]],
    px_per_cm: float = 10.0,
) -> tuple[int, str]:
    """Which axis end is the tip. The end with the longer thin tail (bins under the junction
    fraction of the body width, walked in from each end) when the tails differ by more than
    1 cm and one is at least 1.5x the other: the shaft and cone make a long thin run, the
    plunger stem a short one, so the thinner end *band* alone picks the plunger when the
    pipette rests grip-up (`tail`; v4: the rule itself lives in
    `finebio_observations.tip_side_from_tails` and every row carries it). Else the end
    farther from the nearest hand box (`hand`), else the lower end in the image (`lower`)."""
    ends = np.asarray(row.axis, dtype=np.float64)
    if profile is not None and profile.body_width > 0:
        tails = [junction_from_profile(profile, side, min_tail_px=0.0).tail_px for side in (0, 1)]
        side = tip_side_from_tails(tails, px_per_cm, ratio=TAIL_SIDE_RATIO)
        if side is not None:
            return side, "tail"
    if hands:
        boxes = [tuple(float(v) for v in h["box_xyxy_px"]) for h in hands]
        far = [min(point_box_distance(tuple(e), b) for b in boxes) for e in ends]
        if abs(far[0] - far[1]) > 1e-6:
            return (0 if far[0] > far[1] else 1), "hand"
    return (0 if ends[0][1] > ends[1][1] else 1), "lower"


def width_profile_estimate(
    row: PipetteRow, profile: WidthProfile | None, min_tail_px: float
) -> TipEstimate:
    """Method B on one row (the row's `tip_side` is set)."""
    if profile is None or row.tip_side is None:
        return _body_end_estimate("B", row, None, reason="no profile")
    junction = junction_from_profile(
        profile, row.tip_side, min_tail_px=min_tail_px, terminal_px=min_tail_px
    )
    frame = _row_axis(row)
    if not junction.present:
        return _body_end_estimate(
            "B",
            row,
            False,
            body_width_px=round(junction.body_width, 1),
            tail_px=round(junction.tail_px, 1),
            terminal_width_px=round(junction.terminal_width, 1),
        )
    confidence = min(1.0, 0.5 + 0.5 * (junction.tail_px - min_tail_px) / max(min_tail_px, 1.0))
    return TipEstimate(
        "B",
        row.view,
        row.frame,
        row.slot,
        row.cls,
        True,
        _pt(frame.tip_end),
        _pt(frame.point(-junction.junction_t)),
        round(junction.tail_px, 1),
        round(confidence, 3),
        _pt(frame.tip_end),
        row.tip_side,
        {
            "body_width_px": round(junction.body_width, 1),
            "tail_width_px": round(junction.tail_width, 1),
            "terminal_width_px": round(junction.terminal_width, 1),
        },
    )


# --------------------------------------------------------------------------------------------
# method A: detector tip boxes


def detector_tip_estimate(
    row: PipetteRow, tips: Sequence[Mapping[str, Any]], gate_px: float = BOX_GATE_PX
) -> TipEstimate:
    """Method A: the attached `*_tip` box nearest the axis line; the tip end is the box's far
    edge along the axis, the junction its near edge. Without a box the method abstains."""
    axis = np.asarray(row.axis, dtype=np.float64)
    best: tuple[Any, Mapping[str, Any]] | None = None
    for tip in tips:
        attachment = attach_tip_box(axis, tip["box_xyxy_px"], gate_px)
        if attachment is None:
            continue
        if best is None or attachment.across_px < best[0].across_px:
            best = (attachment, tip)
    if best is None:
        return _body_end_estimate("A", row, None, reason="no tip box attached")
    attachment, tip = best
    side = attachment.end
    frame = AxisFrame.from_axis(row.axis, side)
    tip_end = attachment.axis_px[side]
    junction = frame.point(max(attachment.gap_px, 0.0))
    score = float(tip.get("detector_score") or 0.0)
    matches = tip.get("object_class") == TIP_CLASS.get(row.cls)
    return TipEstimate(
        "A",
        row.view,
        row.frame,
        row.slot,
        row.cls,
        True,
        _pt(tip_end),
        _pt(junction),
        round(frame.along(tip_end) - max(attachment.gap_px, 0.0), 1),
        round(score * (1.0 if matches else 0.7), 3),
        _pt(tip_end),
        side,
        {
            "tip_class": tip.get("object_class"),
            "class_matches": bool(matches),
            "across_px": round(attachment.across_px, 1),
            "gap_px": round(attachment.gap_px, 1),
            "side_conflict": row.tip_side is not None and side != row.tip_side,
            "box": [float(v) for v in tip["box_xyxy_px"]],
        },
    )


# --------------------------------------------------------------------------------------------
# method C: the intensity walk


@dataclass
class WalkResult:
    present: bool
    run_px: float
    end_t: float
    mean_contrast: float
    stop_reason: str
    profile_contrast: list[float] = field(default_factory=list)


def _sample(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Nearest-pixel samples of `image` (H x W x C or H x W) at float (x, y) points, clipped
    to the frame."""
    h, w = image.shape[:2]
    xs = np.clip(np.rint(points[..., 0]).astype(int), 0, w - 1)
    ys = np.clip(np.rint(points[..., 1]).astype(int), 0, h - 1)
    return image[ys, xs]


def intensity_walk(
    hsv: np.ndarray,
    frame: AxisFrame,
    *,
    max_t: float,
    min_run_px: float,
    start_t: float = 0.0,
    background_offset: float = BACKGROUND_OFFSET_PX,
    contrast_min: float = CONTRAST_MIN,
    saturation_max: float = SATURATION_MAX,
    gap_px: int = WALK_GAP_PX,
) -> WalkResult:
    """Walk outward from `start_t` along the axis in 1 px steps. At each step the on-axis
    value is the brightest of the pixels within `ON_AXIS_HALF_WIDTH_PX` across the axis and
    the background the mean value `background_offset` px either side; the step is on the tip
    when the contrast exceeds `contrast_min` and the bright pixel's saturation is under
    `saturation_max`. The run ends after `gap_px` steps off the tip, at the frame edge or at
    `max_t`; the end is refined to the steepest fall of the on-axis value near the stop."""
    steps = int(max(0.0, max_t - start_t))
    if steps < 2:
        return WalkResult(False, 0.0, start_t, 0.0, "no room")
    t = start_t + np.arange(steps, dtype=np.float64)
    offsets = np.arange(-ON_AXIS_HALF_WIDTH_PX, ON_AXIS_HALF_WIDTH_PX + 1, dtype=np.float64)
    on_points = (
        frame.tip_end + t[:, None, None] * frame.unit + offsets[None, :, None] * frame.normal
    )
    on = _sample(hsv, on_points).astype(np.float64)  # steps x offsets x 3
    value = on[..., 2]
    brightest = value.argmax(axis=1)
    on_value = value[np.arange(steps), brightest]
    on_saturation = on[np.arange(steps), brightest, 1]
    sides = np.array(
        [-background_offset, -background_offset + 1, background_offset - 1, background_offset]
    )
    bg_points = frame.tip_end + t[:, None, None] * frame.unit + sides[None, :, None] * frame.normal
    background = _sample(hsv, bg_points).astype(np.float64)[..., 2].mean(axis=1)
    contrast = on_value - background
    on_tip = (contrast > contrast_min) & (on_saturation < saturation_max)
    h, w = hsv.shape[:2]
    centre = frame.tip_end + t[:, None] * frame.unit
    inside = (centre[:, 0] >= 0) & (centre[:, 0] < w) & (centre[:, 1] >= 0) & (centre[:, 1] < h)
    last_on = -1
    off = 0
    reason = "max_t"
    # The first steps may still have the cone under the background samples: a run gets this
    # long to begin before the walk gives up.
    grace = max(gap_px + 2, int(min_run_px))
    for i in range(steps):
        if not inside[i]:
            reason = "frame edge"
            break
        if on_tip[i]:
            last_on = i
            off = 0
            continue
        off += 1
        if last_on >= 0 and off >= gap_px:
            reason = "background"
            break
        if last_on < 0 and i >= grace:
            reason = "no run"
            break
    if last_on < 0:
        return WalkResult(False, 0.0, start_t, 0.0, reason, [round(c, 1) for c in contrast[:8]])
    end = last_on
    # Refine: the steepest fall of the on-axis value within 3 px of the stop.
    lo, hi = max(1, end - 3), min(steps - 1, end + 3)
    if hi > lo:
        gradient = np.diff(on_value[lo - 1 : hi + 1])
        end = lo - 1 + int(np.argmin(gradient))
    run = float(end + 1)
    mean_contrast = float(contrast[: end + 1].mean())
    return WalkResult(
        run >= min_run_px,
        run,
        start_t + end + 0.5,
        mean_contrast,
        reason,
        [round(c, 1) for c in contrast[: end + 2]],
    )


def intensity_estimate(
    row: PipetteRow, hsv: np.ndarray, *, walk_px: float, min_run_px: float
) -> TipEstimate:
    """Method C on one row: the walk from the mask's thin end."""
    if row.tip_side is None:
        return _body_end_estimate("C", row, None, reason="no tip side")
    frame = _row_axis(row)
    walk = intensity_walk(hsv, frame, max_t=walk_px, min_run_px=min_run_px)
    detail = {
        "run_px": round(walk.run_px, 1),
        "mean_contrast": round(walk.mean_contrast, 1),
        "stop": walk.stop_reason,
    }
    if not walk.present:
        return _body_end_estimate("C", row, False, **detail)
    confidence = min(1.0, walk.mean_contrast / 60.0) * (
        1.0 if walk.run_px >= 2 * min_run_px else 0.6
    )
    end = frame.point(walk.end_t)
    return TipEstimate(
        "C",
        row.view,
        row.frame,
        row.slot,
        row.cls,
        True,
        _pt(end),
        _pt(frame.tip_end),
        round(walk.run_px, 1),
        round(confidence, 3),
        _pt(end),
        row.tip_side,
        detail,
    )


# --------------------------------------------------------------------------------------------
# method D: SAM3 box prompts


def clamp_box(box: Sequence[float], size: Sequence[int]) -> Box | None:
    w, h = size
    x0, y0, x1, y1 = (float(v) for v in box)
    x0, x1 = max(0.0, x0), min(float(w), x1)
    y0, y1 = max(0.0, y0), min(float(h), y1)
    if x1 - x0 < 2.0 or y1 - y0 < 2.0:
        return None
    return (round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1))


def slim_tip_box(row: PipetteRow, length_px: float) -> Box:
    """The axis-aligned box around the segment from the body's thin end `length_px` outward,
    padded across by half the body width (at least `SLIM_BOX_MIN_HALF_WIDTH_PX`)."""
    frame = _row_axis(row)
    half = max(0.5 * row.width_px, SLIM_BOX_MIN_HALF_WIDTH_PX)
    corners = np.array([frame.point(t, a) for t in (-half, length_px) for a in (-half, half)])
    return (
        float(corners[:, 0].min()),
        float(corners[:, 1].min()),
        float(corners[:, 0].max()),
        float(corners[:, 1].max()),
    )


def union_box(a: Sequence[float], b: Sequence[float]) -> Box:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def prompt_boxes(
    row: PipetteRow, a_estimate: TipEstimate | None, size: Sequence[int], slim_px: float
) -> dict[str, Box] | None:
    """The two D prompts of a row: `tip` (A's box padded, else the slim box) and `union`."""
    if row.tip_side is None:
        return None
    tip: Box | None
    if a_estimate is not None and a_estimate.tip_present and a_estimate.detail.get("box"):
        x0, y0, x1, y1 = a_estimate.detail["box"]
        tip = (x0 - TIP_BOX_PAD_PX, y0 - TIP_BOX_PAD_PX, x1 + TIP_BOX_PAD_PX, y1 + TIP_BOX_PAD_PX)
    else:
        tip = slim_tip_box(row, slim_px)
    tip = clamp_box(tip, size)
    if tip is None:
        return None
    union = clamp_box(union_box(row.mask_bbox, tip), size)
    if union is None:
        return None
    return {"tip": tip, "union": union}


def write_box_streams(
    rows: Sequence[PipetteRow],
    prompts: Mapping[tuple[str, int, str], Mapping[str, Box]],
    *,
    output: Path,
    offset: int,
) -> dict[str, dict[str, Any]]:
    """One box stream per view for the arms' worker (`muggled_worker.parse_box_stream`):
    slots are `<slot label>|tip` and `<slot label>|union`, contiguous per view; frame indices
    are analysis indices from the view's first prompted proxy frame (`start_frame`)."""
    output.mkdir(parents=True, exist_ok=True)
    by_view: dict[str, dict[int, list[tuple[str, Box]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        boxes = prompts.get(row.key)
        if not boxes:
            continue
        for kind in ("tip", "union"):
            by_view[row.view][row.frame - offset].append((f"{row.slot}|{kind}", boxes[kind]))
    streams: dict[str, dict[str, Any]] = {}
    for view, frames in sorted(by_view.items()):
        labels = sorted({label for boxes in frames.values() for label, _ in boxes})
        slot_of = {label: i for i, label in enumerate(labels)}
        start = min(frames)
        path = output / f"{view}.jsonl"
        count = 0
        with path.open("w", encoding="utf-8") as handle:
            for proxy_frame in sorted(frames):
                boxes = [
                    {
                        "slot": slot_of[label],
                        "label": label,
                        "box_xyxy_px": [float(v) for v in box],
                        "score": None,
                        "source": "finebio_tipseg",
                    }
                    for label, box in sorted(frames[proxy_frame], key=lambda item: slot_of[item[0]])
                ]
                handle.write(
                    json.dumps({"frame_index": proxy_frame - start, "boxes": boxes}) + "\n"
                )
                count += len(boxes)
        streams[view] = {
            "path": str(path),
            "start_frame": start,
            "max_frames": max(frames) - start + 1,
            "labels": labels,
            "frames": len(frames),
            "prompts": count,
        }
    return streams


def run_sam3_views(
    streams: Mapping[str, Mapping[str, Any]],
    proxies: Mapping[str, str],
    *,
    run_root: Path,
    views: Sequence[str] | None = None,
    allow_gpu_neighbour: Sequence[int] = (),
    max_side_length: int = 1280,
) -> dict[str, Any]:
    """The arms' worker in box-decode mode, one run per view, on the tip box streams."""
    from .muggled_arms import parse_args as arms_parse_args
    from .muggled_arms import run_arm

    results: dict[str, Any] = {}
    for view, stream in sorted(streams.items()):
        if views and view not in views:
            continue
        run_id = f"tipseg-{view.lower()}"
        if (run_root / run_id).exists():
            results[view] = {"run_directory": str(run_root / run_id), "state": "existing"}
            continue
        argv = [
            "box-decode",
            "--video",
            str(proxies[view]),
            "--view-id",
            view,
            "--run-root",
            str(run_root),
            "--run-id",
            run_id,
            "--start-frame",
            str(int(stream["start_frame"])),
            "--max-frames",
            str(int(stream["max_frames"])),
            "--max-side-length",
            str(max_side_length),
            "--gpu-guard-profile",
            f"sam3_{max_side_length}",
            "--box-stream",
            str(stream["path"]),
        ]
        for pid in allow_gpu_neighbour:
            argv.extend(["--allow-gpu-neighbour", str(int(pid))])
        started = perf_counter()
        run_directory = run_arm(arms_parse_args(argv))
        manifest = json.loads((run_directory / "manifest.json").read_text(encoding="utf-8"))
        status = manifest["method_statuses"][0]
        results[view] = {
            "run_directory": str(run_directory),
            "state": status["state"],
            "blocker": status.get("blocker"),
            "elapsed_seconds": round(perf_counter() - started, 1),
            "worker_elapsed_seconds": manifest["measurements"]["elapsed_seconds"],
            "gpu_peak_vram_bytes": manifest["measurements"].get("gpu_peak_vram_bytes"),
            "observation_rows": manifest.get("observation_rows"),
            "mask_artifact_count": manifest.get("mask_artifact_count"),
        }
        print(f"{view}: {status['state']} in {results[view]['elapsed_seconds']} s", flush=True)
    return results


def _axis_band_extent(
    mask: np.ndarray, frame: AxisFrame, band_px: float
) -> tuple[float, float, int, float] | None:
    """(min t, max t, pixels, median across width) of the mask pixels within `band_px` of the
    axis line and beyond the body end (t > -band_px / 2); None when too few."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    rel = np.column_stack([xs + 0.5, ys + 0.5]) - frame.tip_end
    along, across = rel @ frame.unit, rel @ frame.normal
    keep = (np.abs(across) <= band_px) & (along > -0.5 * band_px)
    if int(keep.sum()) < MIN_TIP_MASK_PX:
        return None
    along_k = along[keep]
    return (
        float(np.percentile(along_k, 1)),
        float(np.percentile(along_k, 99)),
        int(keep.sum()),
        float(np.median(np.abs(across[keep])) * 2.0),
    )


def tip_mask_verdict(
    extent: tuple[float, float, int, float] | None, px_per_cm: float
) -> str | None:
    """Why a SAM3 mask beyond the body end is not a tip, or None when it passes: too few
    pixels, too short, too wide for a tip, not elongated along the axis, or not attached to
    the body end."""
    if extent is None:
        return "too few pixels beyond the body end"
    t_min, t_max, _pixels, across_width = extent
    length = t_max - max(t_min, 0.0)
    if length < MIN_TIP_EXTENT_PX:
        return "too short"
    if across_width > TIP_MAX_WIDTH_CM * px_per_cm:
        return "too wide for a tip"
    if length < TIP_MIN_ELONGATION * max(across_width, 1.0):
        return "not elongated"
    if t_min > TIP_ATTACH_GAP_CM * px_per_cm:
        return "not attached to the body end"
    return None


def measure_sam3_masks(
    run_directory: Path,
    rows_by_key: Mapping[tuple[str, int, str], PipetteRow],
    *,
    view: str,
    raw_start: int,
    body_masks: Callable[[PipetteRow], np.ndarray | None],
) -> list[TipEstimate]:
    """Methods D and D2 from one view's worker run: the `|tip` mask's extent along the axis
    beyond the body end, and the `|union` mask minus the body mask."""
    from .observations import load_observations

    frames = load_observations(run_directory / "observations.jsonl")
    out: list[TipEstimate] = []
    for frame in frames:
        raw = raw_start + frame.analysis_frame_index
        by_label: dict[str, Any] = {obj.label: obj for obj in frame.objects}
        slots = {label.rsplit("|", 1)[0] for label in by_label}
        for slot in sorted(slots):
            row = rows_by_key.get((view, raw, slot))
            if row is None or row.tip_side is None:
                continue
            axis = _row_axis(row)
            band = max(MASK_BAND_FACTOR * row.width_px, 12.0)
            for method, label in (("D", f"{slot}|tip"), ("D2", f"{slot}|union")):
                obj = by_label.get(label)
                if obj is None or obj.mask is None:
                    out.append(_body_end_estimate(method, row, None, reason="no mask"))
                    continue
                mask = read_mask(run_directory / obj.mask.uri)
                if mask is None:
                    out.append(_body_end_estimate(method, row, None, reason="mask unreadable"))
                    continue
                mask, _, _ = filter_mask_components(mask)
                if method == "D2":
                    body = body_masks(row)
                    if body is None:
                        out.append(_body_end_estimate(method, row, None, reason="no body mask"))
                        continue
                    mask = mask & ~body
                extent = _axis_band_extent(mask, axis, band)
                iou = float(obj.confidence)
                px_per_cm = row.px_per_cm or 8.0
                verdict = tip_mask_verdict(extent, px_per_cm)
                if verdict is not None:
                    out.append(
                        _body_end_estimate(
                            method,
                            row,
                            False,
                            decoder_iou=round(iou, 3),
                            reason=verdict,
                            beyond_px=None if extent is None else round(extent[1], 1),
                            across_width_px=None if extent is None else round(extent[3], 1),
                        )
                    )
                    continue
                assert extent is not None
                t_min, t_max, pixels, across_width = extent
                junction_t = max(t_min, 0.0)
                thin = across_width <= max(row.width_px, 6.0)
                confidence = min(1.0, iou) * (1.0 if thin else 0.5)
                out.append(
                    TipEstimate(
                        method,
                        row.view,
                        row.frame,
                        row.slot,
                        row.cls,
                        True,
                        _pt(axis.point(t_max)),
                        _pt(axis.point(junction_t)),
                        round(t_max - junction_t, 1),
                        round(confidence, 3),
                        _pt(axis.point(t_max)),
                        row.tip_side,
                        {
                            "decoder_iou": round(iou, 3),
                            "pixels_beyond": pixels,
                            "across_width_px": round(across_width, 1),
                            "thin": bool(thin),
                            "t_min": round(t_min, 1),
                        },
                    )
                )
    return out


# --------------------------------------------------------------------------------------------
# method E: fusion, then the views


def fuse_row(
    row: PipetteRow,
    estimates: Mapping[str, TipEstimate],
    *,
    agree_px: float = AGREE_PX,
    lone_confidence: float = LONE_CONFIDENCE,
) -> TipEstimate:
    """The per-row fusion, in two parts.

    The end: methods that see something beyond the mask end (A, C, D, D2) vote with their
    end's position along the row's axis; the largest cluster within `agree_px` wins when it
    has two members or one member with confidence at least `lone_confidence`, and the fused
    end is the cluster's confidence-weighted median. Without such a cluster the end is the
    mask's own end (B's end, the terminal centroid), which the clicks say is the tip end
    whenever SAM3's mask covers the tip.

    The presence: True with a winning cluster (a thin extension beyond the mask end is a
    tip), None when only B sees a thin tail inside the mask (shaft or tip, undecidable in one
    view: the 3D two-state length settles it), False when B sees no thin tail and nothing
    extends the mask.
    """
    frame = _row_axis(row)
    beyond = [
        (frame.along(e.end_px), max(e.confidence, 1e-3), name, e)
        for name, e in estimates.items()
        if e.tip_present and name in ("A", "C", "D", "D2")
    ]
    noes = [name for name, e in estimates.items() if e.tip_present is False and name != "E"]
    b = estimates.get("B")
    detail: dict[str, Any] = {"votes": [v[2] for v in beyond], "noes": noes}
    best: list[tuple[float, float, str, TipEstimate]] = []
    if beyond:
        for t, _, _, _ in beyond:
            cluster = [v for v in beyond if abs(v[0] - t) <= agree_px]
            if sum(v[1] for v in cluster) > sum(v[1] for v in best):
                best = cluster
        if len(best) < 2 and best[0][1] < lone_confidence:
            best = []
            detail["reason"] = "one weak vote beyond the mask end"
    if not best:
        tail = b is not None and b.tip_present is True
        present: bool | None = None if tail else (False if b is not None else None)
        detail["basis"] = "mask end; B tail" if tail else "mask end; no tail"
        out = _body_end_estimate("E", row, present, **detail)
        if tail:
            assert b is not None
            out.junction_px = b.junction_px
            out.tip_length_px = b.tip_length_px
            out.confidence = round(0.5 * b.confidence, 3)
        return out
    total = sum(v[1] for v in beyond)
    weight = sum(v[1] for v in best)
    ts = np.array([v[0] for v in best])
    ws = np.array([v[1] for v in best])
    order = np.argsort(ts)
    cumulative = np.cumsum(ws[order])
    t_fused = float(ts[order][int(np.searchsorted(cumulative, 0.5 * cumulative[-1]))])
    confidence = weight / total * (1.0 if len(best) >= 2 else 0.5)
    end = frame.point(t_fused)
    heaviest = max(best, key=lambda v: v[1])[3]
    junction = heaviest.junction_px or _pt(frame.tip_end)
    detail.update({"agreeing": [v[2] for v in best], "basis": "beyond the mask end"})
    return TipEstimate(
        "E",
        row.view,
        row.frame,
        row.slot,
        row.cls,
        True,
        _pt(end),
        junction,
        round(t_fused - frame.along(junction), 1),
        round(confidence, 3),
        _pt(end),
        row.tip_side,
        detail,
    )


@dataclass
class Tip3D:
    """A method's end triangulated across the views of one frame and pipette, with its
    junction (the body end: the method's own junction where it sees a tip, else the mask
    end) and the butt from the same views: the 3D tip length and body-plus-tip length.
    `tip_votes` counts the used views where the method saw a tip; `with_tip` reads the total
    length against the body prior (the two-state rule) when the butt triangulated."""

    method: str
    frame: int
    cls: str
    views: list[str]
    tip_cm: list[float]
    body_end_cm: list[float] | None
    butt_cm: list[float] | None
    tip_length_cm: float | None
    body_length_cm: float | None
    total_length_cm: float | None
    residual_px: dict[str, float]
    dropped_view: str | None
    tip_present_votes: dict[str, bool | None]
    tip_votes: int = 0
    with_tip: bool | None = None


def _triangulate(
    cams: Mapping[str, Camera], pixels: Mapping[str, Sequence[float]]
) -> tuple[np.ndarray, dict[str, float]]:
    views = sorted(pixels)
    point = triangulate_pixels(
        [cams[v] for v in views], [np.asarray(pixels[v], dtype=np.float64) for v in views]
    )
    residuals = {
        v: float(np.linalg.norm(cams[v].project(point)[0] - np.asarray(pixels[v]))) for v in views
    }
    return point, residuals


def _gated_triangulation(
    cams: Mapping[str, Camera],
    pixels: Mapping[str, Sequence[float]],
    gate_px: float = RESIDUAL_GATE_PX,
) -> tuple[np.ndarray, dict[str, float], str | None] | None:
    """Triangulate; with three or more views drop the worst once when it exceeds the gate;
    None when the remaining views still disagree by more than the gate."""
    if len(pixels) < 2:
        return None
    point, residuals = _triangulate(cams, pixels)
    dropped = None
    if len(pixels) >= 3 and max(residuals.values()) > gate_px:
        dropped = max(residuals, key=residuals.get)
        point, residuals = _triangulate(cams, {v: p for v, p in pixels.items() if v != dropped})
    if max(residuals.values()) > gate_px:
        return None
    return point, residuals, dropped


def triangulate_method(
    method: str,
    estimates: Sequence[TipEstimate],
    rows_by_key: Mapping[tuple[str, int, str], PipetteRow],
    cameras_at: Callable[[int], Mapping[str, Camera]],
    *,
    body_length_cm: float = DEFAULT_BODY_LENGTH_CM,
    margin_cm: float = TIP_ATTACHED_MARGIN_CM,
) -> list[Tip3D]:
    """Per (frame, class): the method's `end_px` across the views where it answered (a tip
    or no tip; abstentions carry no end), its junction and the butt across the same views.
    A method that only answers with a box or a mask (A, D, D2) is triangulated on the frames
    where it saw a tip in two views."""
    by_frame_class: dict[tuple[int, str], list[TipEstimate]] = defaultdict(list)
    for e in estimates:
        if e.method == method:
            by_frame_class[(e.frame, e.cls)].append(e)
    out: list[Tip3D] = []
    for (frame, cls), group in sorted(by_frame_class.items()):
        cams = cameras_at(frame)
        answered = [e for e in group if e.tip_present is not None and e.view in cams]
        if method == "E":
            answered = [e for e in group if e.view in cams]
        if len(answered) < 2:
            continue
        pixels = {e.view: e.end_px for e in answered}
        tip = _gated_triangulation(cams, pixels)
        if tip is None:
            continue
        point, residuals, dropped = tip
        used = [v for v in pixels if v != dropped]
        body_pixels: dict[str, list[float]] = {}
        butt_pixels: dict[str, list[float]] = {}
        votes = 0
        for e in answered:
            if e.view not in used:
                continue
            row = rows_by_key[e.key]
            axis = _row_axis(row)
            junction = e.junction_px if e.tip_present and e.junction_px else _pt(axis.tip_end)
            body_pixels[e.view] = junction
            butt_pixels[e.view] = _pt(axis.other_end)
            votes += int(bool(e.tip_present))
        body = _gated_triangulation(cams, body_pixels)
        butt = _gated_triangulation(cams, butt_pixels)
        tip_len = None if body is None else float(np.linalg.norm(point - body[0]))
        body_len = (
            None if body is None or butt is None else float(np.linalg.norm(body[0] - butt[0]))
        )
        total = None if butt is None else float(np.linalg.norm(point - butt[0]))
        out.append(
            Tip3D(
                method,
                frame,
                cls,
                used,
                [round(float(v), 3) for v in point],
                None if body is None else [round(float(v), 3) for v in body[0]],
                None if butt is None else [round(float(v), 3) for v in butt[0]],
                None if tip_len is None else round(tip_len, 2),
                None if body_len is None else round(body_len, 2),
                None if total is None else round(total, 2),
                {v: round(r, 2) for v, r in residuals.items()},
                dropped,
                {e.view: e.tip_present for e in group},
                votes,
                None if total is None else bool(total >= body_length_cm + margin_cm),
            )
        )
    return out


# --------------------------------------------------------------------------------------------
# scoring


def _pct(values: Sequence[float], digits: int = 1) -> dict[str, Any]:
    if not values:
        return {"n": 0, "median": None, "p10": None, "p90": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "median": round(float(np.median(arr)), digits),
        "p10": round(float(np.percentile(arr, 10)), digits),
        "p90": round(float(np.percentile(arr, 90)), digits),
    }


def coverage_table(
    estimates: Sequence[TipEstimate], rows: Sequence[PipetteRow]
) -> dict[str, dict[str, Any]]:
    """Score 1: per method, rows with a tip found, abstentions, and the tip length px per class."""
    n_rows = len(rows)
    out: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        mine = [e for e in estimates if e.method == method]
        found = [e for e in mine if e.tip_present]
        by_class = {
            cls: _pct([e.tip_length_px for e in found if e.cls == cls and e.tip_length_px])
            for cls in PIPETTE_CLASSES
        }
        out[method] = {
            "rows": n_rows,
            "answered": len([e for e in mine if e.tip_present is not None]),
            "tip_found": len(found),
            "tip_absent": len([e for e in mine if e.tip_present is False]),
            "abstained": n_rows - len([e for e in mine if e.tip_present is not None]),
            "found_fraction": round(len(found) / n_rows, 3) if n_rows else None,
            "found_fraction_single_channel": _fraction(
                [e for e in found if e.cls in SINGLE_CHANNEL],
                [r for r in rows if r.cls in SINGLE_CHANNEL],
            ),
            "found_fraction_8_channel": _fraction(
                [e for e in found if e.cls == MULTI_CHANNEL],
                [r for r in rows if r.cls == MULTI_CHANNEL],
            ),
            "tip_length_px_by_class": by_class,
        }
    return out


def _fraction(part: Sequence[Any], whole: Sequence[Any]) -> float | None:
    return round(len(part) / len(whole), 3) if whole else None


def consistency_table(tips3d: Sequence[Tip3D]) -> dict[str, dict[str, Any]]:
    """Score 1, 3D: per method the triangulated tip length (median, p10-p90) and the
    reprojection residual over frames with at least two views."""
    out: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        mine = [t for t in tips3d if t.method == method]
        single = [t for t in mine if t.cls in SINGLE_CHANNEL]
        voted = [t for t in single if t.tip_votes > 0 and t.tip_length_cm is not None]
        out[method] = {
            "frames_with_3d_end": len(mine),
            "frames_with_tip_vote": len([t for t in mine if t.tip_votes > 0]),
            "tip_length_cm_single_channel": _pct([t.tip_length_cm for t in voted], 2),
            "tip_length_cm_8_channel": _pct(
                [
                    t.tip_length_cm
                    for t in mine
                    if t.cls == MULTI_CHANNEL and t.tip_votes > 0 and t.tip_length_cm
                ],
                2,
            ),
            "tip_length_in_3_7_cm_fraction": _fraction(
                [
                    t
                    for t in voted
                    if TIP_LENGTH_RANGE_CM[0] <= t.tip_length_cm <= TIP_LENGTH_RANGE_CM[1]
                ],
                voted,
            ),
            "total_length_cm_single_channel": _pct(
                [t.total_length_cm for t in single if t.total_length_cm is not None], 2
            ),
            "with_tip_fraction_single_channel": _fraction(
                [t for t in single if t.with_tip], [t for t in single if t.with_tip is not None]
            ),
            "reprojection_residual_px": _pct([max(t.residual_px.values()) for t in mine], 2),
        }
    return out


def agreement_table(
    estimates: Sequence[TipEstimate],
    rows_by_key: Mapping[tuple[str, int, str], PipetteRow],
    agree_px: float = AGREE_PX,
) -> dict[str, dict[str, Any]]:
    """Score 2: for every pair of methods, over the rows where both see a tip, the fraction
    whose ends sit within `agree_px` along the axis, and the median gap."""
    by_key: dict[tuple[str, int, str], dict[str, TipEstimate]] = defaultdict(dict)
    for e in estimates:
        by_key[e.key][e.method] = e
    out: dict[str, dict[str, Any]] = {}
    for i, a in enumerate(METHODS):
        for b in METHODS[i + 1 :]:
            gaps = []
            both_present = 0
            presence_agree = 0
            compared = 0
            for key, methods in by_key.items():
                ea, eb = methods.get(a), methods.get(b)
                if ea is None or eb is None or ea.tip_present is None or eb.tip_present is None:
                    continue
                compared += 1
                if ea.tip_present == eb.tip_present:
                    presence_agree += 1
                if ea.tip_present and eb.tip_present:
                    both_present += 1
                    frame = _row_axis(rows_by_key[key])
                    gaps.append(abs(frame.along(ea.end_px) - frame.along(eb.end_px)))
            out[f"{a}-{b}"] = {
                "rows_both_answered": compared,
                "presence_agreement": round(presence_agree / compared, 3) if compared else None,
                "rows_both_found": both_present,
                "end_within_gate": round(sum(1 for g in gaps if g <= agree_px) / len(gaps), 3)
                if gaps
                else None,
                "end_gap_px": _pct(gaps),
            }
    return out


@dataclass
class ClickCell:
    frame: int
    view: str
    slot_index: int
    cls: str
    label: str
    state: str
    tip_px: list[float] | None
    hidden: bool
    suggested_px: list[float]


def click_cells(workspace: Mapping[str, Any], record: Mapping[str, Any]) -> list[ClickCell]:
    decisions = {
        (int(c["raw_frame"]), str(c["view"]), int(c["slot"])): c for c in record.get("cells", [])
    }
    out = []
    for cell in workspace["cells"]:
        key = (int(cell["raw_frame"]), str(cell["view"]), int(cell["slot"]))
        decision = decisions.get(key, {})
        tip = decision.get("tip_px")
        out.append(
            ClickCell(
                key[0],
                key[1],
                key[2],
                str(cell["class"]),
                str(cell["label"]),
                str(cell.get("state", "")),
                None if tip is None else [float(v) for v in tip],
                bool(decision.get("hidden")),
                [float(v) for v in cell["suggested_tip_px"]],
            )
        )
    return out


def click_end(cell: ClickCell, row: PipetteRow) -> int | None:
    """The axis end the click sits nearer to, or None without a click."""
    if cell.tip_px is None:
        return None
    ends = np.asarray(row.axis, dtype=np.float64)
    distances = np.linalg.norm(ends - np.asarray(cell.tip_px), axis=1)
    return int(np.argmin(distances))


def split_cells(
    cells: Sequence[ClickCell], rows_by_key: Mapping[tuple[str, int, str], PipetteRow]
) -> dict[str, list[ClickCell]]:
    """The clicked cells with a row, by where the click sits: `tip_end` (the row's tip side,
    the end with the long thin tail), `other_end` (the plunger side: the marker rule put the
    marker there and the protocol accepted it) and `hidden`."""
    out: dict[str, list[ClickCell]] = {"tip_end": [], "other_end": [], "hidden": []}
    for cell in cells:
        row = rows_by_key.get((cell.view, cell.frame, cell.label))
        if row is None:
            continue
        if cell.hidden:
            out["hidden"].append(cell)
            continue
        end = click_end(cell, row)
        if end is None:
            continue
        out["tip_end" if end == row.tip_side else "other_end"].append(cell)
    return out


def _click_errors(
    method: str,
    cells: Sequence[ClickCell],
    by_key: Mapping[tuple[str, int, str], Mapping[str, TipEstimate]],
    rows_by_key: Mapping[tuple[str, int, str], PipetteRow],
    agree_px: float,
) -> dict[str, Any]:
    along: list[float] = []
    across: list[float] = []
    found = 0
    for cell in cells:
        key = (cell.view, cell.frame, cell.label)
        estimate = by_key.get(key, {}).get(method)
        if estimate is None or cell.tip_px is None:
            continue
        frame = _row_axis(rows_by_key[key])
        if estimate.tip_present:
            found += 1
        offset = np.asarray(cell.tip_px) - np.asarray(estimate.end_px)
        along.append(float(offset @ frame.unit))
        across.append(abs(float(offset @ frame.normal)))
    abs_along = [abs(a) for a in along]
    return {
        "cells": len(along),
        "tip_found": found,
        "along_px_signed": _pct(along),
        "abs_along_px": _pct(abs_along),
        "across_px": _pct(across),
        "within_gate_fraction": _fraction([a for a in abs_along if a <= agree_px], abs_along),
        "click_beyond_end_fraction": _fraction([a for a in along if a > agree_px], along),
    }


def anchor_table(
    estimates: Sequence[TipEstimate],
    rows_by_key: Mapping[tuple[str, int, str], PipetteRow],
    cells: Sequence[ClickCell],
    tips3d: Sequence[Tip3D],
    anchors3d: Mapping[tuple[int, str], np.ndarray],
    *,
    agree_px: float = AGREE_PX,
) -> dict[str, Any]:
    """Score 3: per method, against the clicked cells (the click is the tip end in that view):
    along- and across-axis error px of the method's `end_px` and the fraction within the
    gate, over the cells whose click sits at the row's tip end (`tip_end`) and over all
    clicked cells (`all`); the 3D error of the method's triangulated tip and of its butt
    against the triangulated anchors; and on the hidden cells whether the method put its end
    somewhere else than the suggested marker."""
    by_key: dict[tuple[str, int, str], dict[str, TipEstimate]] = defaultdict(dict)
    for e in estimates:
        by_key[e.key][e.method] = e
    groups = split_cells(cells, rows_by_key)
    out: dict[str, Any] = {
        "cells": {name: len(group) for name, group in groups.items()},
        "other_end_by_class": dict(Counter(c.cls for c in groups["other_end"])),
        "other_end_by_state": dict(Counter(c.state for c in groups["other_end"])),
        "methods": {},
    }
    for method in METHODS:
        hidden_moved = 0
        hidden_no_tip = 0
        for cell in groups["hidden"]:
            key = (cell.view, cell.frame, cell.label)
            estimate = by_key.get(key, {}).get(method)
            if estimate is None:
                continue
            moved = float(np.linalg.norm(np.asarray(estimate.end_px) - cell.suggested_px))
            if moved > agree_px:
                hidden_moved += 1
            if estimate.tip_present is not True:
                hidden_no_tip += 1
        tip_3d: list[float] = []
        butt_3d: list[float] = []
        for t in tips3d:
            if t.method != method:
                continue
            anchor = anchors3d.get((t.frame, t.cls))
            if anchor is None:
                continue
            tip_3d.append(float(np.linalg.norm(np.asarray(t.tip_cm) - anchor)))
            if t.butt_cm is not None:
                butt_3d.append(float(np.linalg.norm(np.asarray(t.butt_cm) - anchor)))
        out["methods"][method] = {
            "tip_end": _click_errors(method, groups["tip_end"], by_key, rows_by_key, agree_px),
            "all": _click_errors(
                method, groups["tip_end"] + groups["other_end"], by_key, rows_by_key, agree_px
            ),
            "error_3d_cm": _pct(tip_3d, 2),
            "butt_error_3d_cm": _pct(butt_3d, 2),
            "anchor_nearer_butt": sum(1 for a, b in zip(tip_3d, butt_3d) if b < a),
            "hidden_cells_with_row": len(groups["hidden"]),
            "hidden_end_moved_from_marker": hidden_moved,
            "hidden_no_tip_claimed": hidden_no_tip,
        }
    return out


def two_state_table(tips3d: Sequence[Tip3D], method: str = "E") -> dict[str, Any]:
    """Score 4: per class, the body-plus-tip 3D length from the fused tips, split into two
    clusters by the best 1D two-means cut; bimodal when the cluster medians sit more than
    twice the pooled spread apart and each cluster holds a tenth of the frames."""
    out: dict[str, Any] = {}
    for cls in PIPETTE_CLASSES:
        mine = [
            t
            for t in tips3d
            if t.method == method and t.cls == cls and t.total_length_cm is not None
        ]
        values = sorted(t.total_length_cm for t in mine)
        block: dict[str, Any] = {
            "frames": len(values),
            "total_length_cm": _pct(values, 2),
            "with_tip_frames": sum(1 for t in mine if t.with_tip),
            "with_tip_fraction": _fraction([t for t in mine if t.with_tip], mine),
            "total_length_cm_with_tip": _pct([t.total_length_cm for t in mine if t.with_tip], 2),
            "total_length_cm_without_tip": _pct(
                [t.total_length_cm for t in mine if t.with_tip is False], 2
            ),
        }
        if len(values) >= 6:
            arr = np.asarray(values)
            best = None
            for cut in range(2, len(arr) - 1):
                lo, hi = arr[:cut], arr[cut:]
                cost = float(((lo - lo.mean()) ** 2).sum() + ((hi - hi.mean()) ** 2).sum())
                if best is None or cost < best[0]:
                    best = (cost, cut)
            assert best is not None
            lo, hi = arr[: best[1]], arr[best[1] :]
            pooled = (
                float(np.sqrt(0.5 * (lo.var() + hi.var()))) if len(lo) > 1 and len(hi) > 1 else 0.0
            )
            separation = float(np.median(hi) - np.median(lo))
            block.update(
                {
                    "mode_low_cm": round(float(np.median(lo)), 2),
                    "mode_high_cm": round(float(np.median(hi)), 2),
                    "n_low": int(lo.size),
                    "n_high": int(hi.size),
                    "separation_cm": round(separation, 2),
                    "pooled_spread_cm": round(pooled, 2),
                    "bimodal": bool(
                        separation > 2.0 * max(pooled, 0.5)
                        and min(lo.size, hi.size) >= 0.1 * arr.size
                    ),
                }
            )
        out[cls] = block
    return out


def blue_timeline(
    estimates: Sequence[TipEstimate],
    rows_by_key: Mapping[tuple[str, int, str], PipetteRow],
    tips3d: Sequence[Tip3D],
    tables: FrameTables,
    frames: Sequence[int],
    *,
    cls: str = "blue_pipette",
    method: str = "E",
    near_px: float = NEAR_BOX_PX,
    body_length_cm: float = DEFAULT_BODY_LENGTH_CM,
    margin_cm: float = TIP_ATTACHED_MARGIN_CM,
) -> dict[str, Any]:
    """Score 5: per working frame the verdict on the blue pipette from the 3D total length
    (fused end to butt): a tip when it exceeds the body prior by the margin, none when it
    sits within the margin of the prior, undecided when shorter (a mask the hand truncated
    cannot say) or when the frame does not triangulate. The largest projected 2D length
    over the views is kept beside it for information only (a glove merged into the mask
    inflates it). Steps are the changes in the verdict after one-frame flickers are dropped;
    each step records whether a tip rack or trash box sat near the tip end in some view."""
    by_frame: dict[int, list[TipEstimate]] = defaultdict(list)
    for e in estimates:
        if e.method == method and e.cls == cls:
            by_frame[e.frame].append(e)
    three_d = {t.frame: t for t in tips3d if t.method == method and t.cls == cls}
    series = []
    for frame in frames:
        group = by_frame.get(frame, [])
        answered = [e for e in group if e.tip_present is not None]
        yes = sum(1 for e in answered if e.tip_present)
        verdict: bool | None = None
        basis = None
        t3 = three_d.get(frame)
        projected = []
        for e in group:
            row = rows_by_key[e.key]
            if row.px_per_cm:
                axis = _row_axis(row)
                length_px = float(np.linalg.norm(np.asarray(e.end_px) - axis.other_end))
                projected.append(length_px / row.px_per_cm)
        if t3 is not None and t3.total_length_cm is not None:
            if t3.total_length_cm >= body_length_cm + margin_cm:
                verdict, basis = True, "3d_length"
            elif t3.total_length_cm >= body_length_cm - margin_cm:
                verdict, basis = False, "3d_length"
            else:
                basis = "3d_truncated"
        near = []
        for e in group:
            for box in tables.events.get((e.view, frame), []):
                if point_box_distance(tuple(e.end_px), tuple(box["box_xyxy_px"])) <= near_px:
                    near.append(f"{e.view}:{box['object_class']}")
        series.append(
            {
                "frame": frame,
                "views": len(group),
                "answered": len(answered),
                "tip_votes": yes,
                "tip_present": verdict,
                "basis": basis,
                "projected_length_cm_max": round(max(projected), 2) if projected else None,
                "total_length_cm": None if t3 is None else t3.total_length_cm,
                "tip_length_cm": None if t3 is None else t3.tip_length_cm,
                "near_boxes": sorted(set(near)),
            }
        )
    decided = [s for s in series if s["tip_present"] is not None]
    # A decided frame whose verdict differs from both decided neighbours is a flicker, not a
    # state; flickers are dropped one at a time until none is left, then the steps are read.
    kept = list(decided)
    flickers = 0
    while True:
        index = next(
            (
                i
                for i in range(1, len(kept) - 1)
                if kept[i]["tip_present"] != kept[i - 1]["tip_present"]
                and kept[i]["tip_present"] != kept[i + 1]["tip_present"]
            ),
            None,
        )
        if index is None:
            break
        del kept[index]
        flickers += 1
    steps = []
    for before, after in zip(kept, kept[1:]):
        if before["tip_present"] == after["tip_present"]:
            continue
        steps.append(
            {
                "frame": after["frame"],
                "previous_frame": before["frame"],
                "kind": "pick_up" if after["tip_present"] else "eject",
                "near_boxes": sorted(set(before["near_boxes"]) | set(after["near_boxes"])),
            }
        )
    return {
        "class": cls,
        "method": method,
        "frames": len(series),
        "frames_with_verdict": len(decided),
        "frames_by_basis": dict(Counter(s["basis"] for s in series if s["basis"])),
        "frames_tip_present": sum(1 for s in decided if s["tip_present"]),
        "frames_projected_over_prior": sum(
            1
            for s in series
            if s["projected_length_cm_max"] is not None
            and s["projected_length_cm_max"] >= body_length_cm + margin_cm
        ),
        "steps": steps,
        "steps_with_rack_or_trash_near": sum(1 for s in steps if s["near_boxes"]),
        "one_frame_flickers": flickers,
        "series": series,
    }


# --------------------------------------------------------------------------------------------
# contact sheets


def draw_cell(
    image: np.ndarray,
    row: PipetteRow,
    estimates: Mapping[str, TipEstimate],
    click: Sequence[float] | None,
    *,
    size: int = SHEET_CROP_PX,
) -> np.ndarray:
    """A crop around the tip side of the axis with each method's end drawn (a filled circle
    on the end, in the method's colour) and the click as a white cross."""
    frame = _row_axis(row)
    ends = [np.asarray(e.end_px) for e in estimates.values()] + [frame.tip_end]
    if click is not None:
        ends.append(np.asarray(click, dtype=np.float64))
    centre = np.mean(np.stack(ends), axis=0)
    span = max(float(np.max(np.linalg.norm(np.stack(ends) - centre, axis=1))) * 2.4, 120.0)
    half = int(round(span / 2))
    h, w = image.shape[:2]
    x0, y0 = int(round(centre[0] - half)), int(round(centre[1] - half))
    x0, y0 = max(0, min(x0, w - 2 * half)), max(0, min(y0, h - 2 * half))
    crop = image[y0 : y0 + 2 * half, x0 : x0 + 2 * half].copy()
    scale = size / crop.shape[1]
    crop = cv2.resize(
        crop, (size, int(round(crop.shape[0] * scale))), interpolation=cv2.INTER_LINEAR
    )

    def to_crop(p: Sequence[float]) -> tuple[int, int]:
        return int(round((p[0] - x0) * scale)), int(round((p[1] - y0) * scale))

    cv2.line(
        crop, to_crop(frame.other_end), to_crop(frame.tip_end), (200, 200, 200), 1, cv2.LINE_AA
    )
    for method in METHODS:
        e = estimates.get(method)
        if e is None or e.tip_present is None:
            continue
        colour = METHOD_COLOURS_BGR[method]
        p = to_crop(e.end_px)
        if e.tip_present:
            cv2.circle(crop, p, 5, colour, -1, cv2.LINE_AA)
        else:
            cv2.circle(crop, p, 5, colour, 1, cv2.LINE_AA)
    if click is not None:
        cx, cy = to_crop(click)
        cv2.drawMarker(crop, (cx, cy), CLICK_COLOUR_BGR, cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
    return crop


def render_sheets(
    rows_by_key: Mapping[tuple[str, int, str], PipetteRow],
    estimates: Sequence[TipEstimate],
    cells: Sequence[ClickCell],
    frames: Callable[[str, int], np.ndarray | None],
    *,
    output: Path,
    per_sheet: int = SHEET_CELLS,
) -> dict[str, str]:
    """Three sheets, `per_sheet` cells each spread over classes and states: clicks at the
    row's tip end, clicks at the other end, hidden cells. Filled circle: the method sees a tip
    there; ring: the method's body end with no tip; cross: my click."""
    by_key: dict[tuple[str, int, str], dict[str, TipEstimate]] = defaultdict(dict)
    for e in estimates:
        by_key[e.key][e.method] = e
    legend = "  ".join(f"{m}={METHOD_NAMES[m]}" for m in METHODS)
    out: dict[str, str] = {}
    for name, with_row in split_cells(cells, rows_by_key).items():
        if not with_row:
            continue
        # Spread: round-robin over (class, state) groups in frame order.
        groups: dict[tuple[str, str], list[ClickCell]] = defaultdict(list)
        for c in sorted(with_row, key=lambda c: (c.frame, c.view)):
            groups[(c.cls, c.state)].append(c)
        picked: list[ClickCell] = []
        while len(picked) < per_sheet and any(groups.values()):
            for key in sorted(groups):
                if groups[key] and len(picked) < per_sheet:
                    picked.append(groups[key].pop(0))
        tiles, labels = [], []
        for c in picked:
            image = frames(c.view, c.frame)
            if image is None:
                continue
            key = (c.view, c.frame, c.label)
            tiles.append(draw_cell(image, rows_by_key[key], by_key.get(key, {}), c.tip_px))
            e = by_key.get(key, {}).get("E")
            verdict = (
                "?"
                if e is None or e.tip_present is None
                else ("tip" if e.tip_present else "no tip")
            )
            labels.append(
                f"f{c.frame} {c.view} {c.cls.replace('_pipette', '')} {c.state} E:{verdict}"
            )
        if not tiles:
            continue
        sheet = render_grid(
            tiles, columns=5, labels=labels, title=f"{name} cells; {legend}; cross = click", gap=4
        )
        path = output / f"contact_sheet_{name}.png"
        cv2.imwrite(str(path), sheet)
        out[name] = path.name
    return out


# --------------------------------------------------------------------------------------------
# io


def write_estimates(estimates: Iterable[TipEstimate], path: Path) -> int:
    count = 0
    with Path(path).open("w", encoding="utf-8") as handle:
        for e in sorted(estimates, key=lambda e: (e.frame, e.view, e.slot, e.method)):
            handle.write(json.dumps(asdict(e), separators=(",", ":")) + "\n")
            count += 1
    return count


def read_estimates(path: Path) -> list[TipEstimate]:
    out = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                out.append(TipEstimate(**json.loads(line)))
    return out


def write_rows(rows: Iterable[PipetteRow], path: Path) -> int:
    count = 0
    with Path(path).open("w", encoding="utf-8") as handle:
        for r in rows:
            handle.write(json.dumps(asdict(r), separators=(",", ":")) + "\n")
            count += 1
    return count


def read_rows(path: Path) -> list[PipetteRow]:
    out = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                out.append(PipetteRow(**json.loads(line)))
    return out


def iter_proxy_frames(path: Path, wanted: Iterable[int]) -> Iterator[tuple[int, np.ndarray]]:
    """Sequential read of a proxy, yielding only the wanted proxy frame indices (a grab
    without retrieve for the rest)."""
    todo = sorted({int(f) for f in wanted})
    if not todo:
        return
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {path}")
    last = todo[-1]
    wanted_set = set(todo)
    index = 0
    try:
        while index <= last:
            if index in wanted_set:
                ok, frame = cap.read()
                if not ok:
                    break
                yield index, frame
            elif not cap.grab():
                break
            index += 1
    finally:
        cap.release()


def _pipettes_length_cm(path: Path | None) -> float:
    if path is None or not Path(path).is_file():
        return DEFAULT_BODY_LENGTH_CM
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return float(doc.get("length_cm") or DEFAULT_BODY_LENGTH_CM)


# --------------------------------------------------------------------------------------------
# steps


def run_prepare(args: argparse.Namespace) -> int:
    root = Path.cwd()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    clip = load_clip(Path(args.clip_config))
    offset = int(clip["frame_index_offset"])
    window = (int(clip["window"]["start_frame"]), int(clip["window"]["end_frame_exclusive"]))
    workspace = json.loads((Path(args.tips_workspace) / "cells.json").read_text(encoding="utf-8"))
    clicks = click_frames(workspace)
    frames = working_frames(clicks, window, args.stride)
    observations = Path(args.observations)
    observations = observations / "observations.jsonl" if observations.is_dir() else observations
    summary = json.loads(
        (observations.parent / "observations_summary.json").read_text(encoding="utf-8")
    )
    worker_runs = summary["worker_runs"]
    print(f"reading {observations} for {len(frames)} frames", flush=True)
    tables = read_frame_tables(observations, frames)
    rows = pipette_rows(tables, worker_runs, offset)
    cameras_at = cameras_from_clip(clip, root)
    fixed_cams = {v: c for v, c in cameras_at(frames[0]).items() if v in FIXED_VIEWS}
    scales = frame_scales(rows, fixed_cams)
    # The fpv scale needs its per-frame pose.
    for row in rows:
        if row.view == FPV_VIEW and (row.view, row.frame, row.cls) not in scales:
            cam = cameras_at(row.frame).get(FPV_VIEW)
            if cam is not None:
                scales[(row.view, row.frame, row.cls)] = (
                    focal_px(cam) / max(depth_cm(cam, np.zeros(3)), 10.0),
                    "bench_origin_fpv",
                )
    body_length_cm = _pipettes_length_cm(args.pipettes_config)
    print(f"{len(rows)} pipette rows with an axis on {len(frames)} frames", flush=True)

    estimates: list[TipEstimate] = []
    a_by_key: dict[tuple[str, int, str], TipEstimate] = {}
    rows_by_view: dict[str, list[PipetteRow]] = defaultdict(list)
    missing_masks = 0
    for row in rows:
        mask = read_mask(root / row.mask_uri) if row.mask_uri else None
        if mask is None:
            missing_masks += 1
            profile = None
        else:
            mask, _, _ = filter_mask_components(mask)
            profile = width_profile(mask, row.axis)
        scale = scales.get((row.view, row.frame, row.cls))
        row.px_per_cm = None if scale is None else round(scale[0], 3)
        px_per_cm = row.px_per_cm or 8.0
        hands = tables.hands.get((row.view, row.frame), [])
        row.tip_side, row.tip_side_rule = choose_tip_side(row, profile, hands, px_per_cm)
        if profile is not None:
            row.end_widths_px = [round(v, 1) for v in profile.end_widths()]
        if mask is not None:
            row.body_end_px = terminal_centroid(mask, row.axis, row.tip_side)
        a = detector_tip_estimate(row, tables.tips.get((row.view, row.frame), []))
        estimates.append(a)
        a_by_key[row.key] = a
        estimates.append(width_profile_estimate(row, profile, MIN_TAIL_CM * px_per_cm))
        rows_by_view[row.view].append(row)

    # Method C needs the frames: one sequential pass per view.
    if not args.no_frames:
        for view, view_rows in sorted(rows_by_view.items()):
            proxy = root / clip["proxies"][view]
            by_proxy_frame: dict[int, list[PipetteRow]] = defaultdict(list)
            for row in view_rows:
                by_proxy_frame[row.frame - offset].append(row)
            print(
                f"{view}: walking {len(view_rows)} rows on {len(by_proxy_frame)} frames", flush=True
            )
            for proxy_frame, image in iter_proxy_frames(proxy, by_proxy_frame):
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                for row in by_proxy_frame[proxy_frame]:
                    px_per_cm = row.px_per_cm or 8.0
                    estimates.append(
                        intensity_estimate(
                            row,
                            hsv,
                            walk_px=WALK_FACTOR * TIP_LENGTH_CM * px_per_cm,
                            min_run_px=MIN_RUN_CM * px_per_cm,
                        )
                    )

    # D's prompts on the click frames only (the GPU budget).
    click_set = set(clicks)
    prompts: dict[tuple[str, int, str], dict[str, Box]] = {}
    sizes = clip["view_sizes"]
    for row in rows:
        if row.frame not in click_set:
            continue
        boxes = prompt_boxes(
            row, a_by_key.get(row.key), sizes[row.view], SLIM_BOX_CM * (row.px_per_cm or 8.0)
        )
        if boxes:
            prompts[row.key] = boxes
    streams = write_box_streams(
        rows, prompts, output=output / "sam3" / "box_streams", offset=offset
    )

    write_rows(rows, output / "rows.jsonl")
    write_estimates(estimates, output / "estimates_abc.jsonl")
    fs_common.write_json(
        output / "working_set.json",
        {
            "schema": SCHEMA,
            "clip_config": str(args.clip_config),
            "observations": str(observations),
            "observations_sha256": fs_common.sha256_file(observations),
            "tips_workspace": str(args.tips_workspace),
            "window": list(window),
            "stride": args.stride,
            "click_frames": clicks,
            "frames": frames,
            "rows": len(rows),
            "rows_by_view": Counter(r.view for r in rows),
            "rows_by_class": Counter(r.cls for r in rows),
            "pipette_rows_without_axis_skipped": tables.pipettes_without_axis,
            "masks_missing": missing_masks,
            "tip_side_rules": Counter(r.tip_side_rule or "none" for r in rows),
            "scale_sources": Counter(s[1] for s in scales.values()),
            "px_per_cm_by_view": {
                v: _pct([r.px_per_cm for r in rows if r.view == v and r.px_per_cm], 2)
                for v in sorted(rows_by_view)
            },
            "body_length_cm": body_length_cm,
            "tip_length_cm_expected": TIP_LENGTH_CM,
            "sam3_streams": streams,
            "sam3_prompts": sum(s["prompts"] for s in streams.values()),
            "settings": {
                "box_gate_px": BOX_GATE_PX,
                "profile_step_px": PROFILE_STEP_PX,
                "junction_fraction": JUNCTION_FRACTION,
                "min_tail_cm": MIN_TAIL_CM,
                "walk_factor": WALK_FACTOR,
                "background_offset_px": BACKGROUND_OFFSET_PX,
                "contrast_min": CONTRAST_MIN,
                "saturation_max": SATURATION_MAX,
                "walk_gap_px": WALK_GAP_PX,
                "min_run_cm": MIN_RUN_CM,
                "slim_box_cm": SLIM_BOX_CM,
                "agree_px": AGREE_PX,
            },
            "elapsed_seconds": round(perf_counter() - started, 1),
            "claim_boundary": CLAIM_BOUNDARY,
            "licence": LICENCE_NOTE,
        },
    )
    print(
        f"{len(estimates)} estimates, {sum(s['prompts'] for s in streams.values())} SAM3 prompts "
        f"in {perf_counter() - started:.1f} s -> {output}",
        flush=True,
    )
    return 0


def run_sam3(args: argparse.Namespace) -> int:
    output = Path(args.output)
    working = json.loads((output / "working_set.json").read_text(encoding="utf-8"))
    clip = load_clip(Path(working["clip_config"]))
    proxies = {v: str(Path.cwd() / p) for v, p in clip["proxies"].items()}
    started = perf_counter()
    results = run_sam3_views(
        working["sam3_streams"],
        proxies,
        run_root=output / "sam3",
        views=args.view or None,
        allow_gpu_neighbour=args.allow_gpu_neighbour,
    )
    record = {
        "schema": SCHEMA,
        "runs": results,
        "elapsed_seconds": round(perf_counter() - started, 1),
        "gpu_seconds": round(
            sum(float(r.get("worker_elapsed_seconds") or 0.0) for r in results.values()), 1
        ),
    }
    fs_common.write_json(output / "sam3" / "runs.json", record)
    failed = [v for v, r in results.items() if r.get("state") not in ("succeeded", "existing")]
    print(json.dumps({k: v for k, v in record.items() if k != "runs"}), flush=True)
    return 1 if failed else 0


def run_score(args: argparse.Namespace) -> int:
    root = Path.cwd()
    output = Path(args.output)
    started = perf_counter()
    working = json.loads((output / "working_set.json").read_text(encoding="utf-8"))
    clip = load_clip(Path(working["clip_config"]))
    offset = int(clip["frame_index_offset"])
    rows = read_rows(output / "rows.jsonl")
    rows_by_key = {r.key: r for r in rows}
    estimates = [
        e for e in read_estimates(output / "estimates_abc.jsonl") if e.method in ("A", "B", "C")
    ]
    cameras_at = cameras_from_clip(clip, root)

    # D from the worker runs, when they exist.
    runs_path = output / "sam3" / "runs.json"
    sam3_runs = json.loads(runs_path.read_text(encoding="utf-8")) if runs_path.is_file() else None
    body_cache: dict[tuple[str, int, str], np.ndarray | None] = {}

    def body_mask(row: PipetteRow) -> np.ndarray | None:
        if row.key not in body_cache:
            mask = read_mask(root / row.mask_uri) if row.mask_uri else None
            body_cache[row.key] = None if mask is None else filter_mask_components(mask)[0]
        return body_cache[row.key]

    d_rows = 0
    if sam3_runs is not None:
        for view, result in sorted(sam3_runs["runs"].items()):
            run_directory = Path(result["run_directory"])
            if not (run_directory / "observations.jsonl").is_file():
                continue
            stream = working["sam3_streams"][view]
            measured = measure_sam3_masks(
                run_directory,
                rows_by_key,
                view=view,
                raw_start=offset + int(stream["start_frame"]),
                body_masks=body_mask,
            )
            estimates.extend(measured)
            d_rows += len(measured)
        body_cache.clear()

    by_key: dict[tuple[str, int, str], dict[str, TipEstimate]] = defaultdict(dict)
    for e in estimates:
        by_key[e.key][e.method] = e
    for row in rows:
        estimates.append(fuse_row(row, by_key[row.key]))
    write_estimates(estimates, output / "estimates.jsonl")

    tips3d: list[Tip3D] = []
    body_length_cm = float(working.get("body_length_cm") or DEFAULT_BODY_LENGTH_CM)
    for method in METHODS:
        tips3d.extend(
            triangulate_method(
                method, estimates, rows_by_key, cameras_at, body_length_cm=body_length_cm
            )
        )
    with (output / "tips3d.jsonl").open("w", encoding="utf-8") as handle:
        for t in tips3d:
            handle.write(json.dumps(asdict(t), separators=(",", ":")) + "\n")

    workspace = json.loads((Path(args.tips_workspace) / "cells.json").read_text(encoding="utf-8"))
    record = json.loads((Path(args.tips_workspace) / "decisions.json").read_text(encoding="utf-8"))
    cells = click_cells(workspace, record)
    scoreboard_path = Path(
        args.tip_scoreboard or (Path(args.tips_workspace) / "scoreboard" / "tip_scoreboard.json")
    )
    anchors3d: dict[tuple[int, str], np.ndarray] = {}
    if scoreboard_path.is_file():
        board = json.loads(scoreboard_path.read_text(encoding="utf-8"))
        for anchor in board["anchors"].get("rows", []):
            if anchor.get("tip_cm") is not None:
                anchors3d[(int(anchor["raw_frame"]), str(anchor["class"]))] = np.asarray(
                    anchor["tip_cm"], dtype=np.float64
                )

    observations = Path(working["observations"])
    tables = read_frame_tables(observations, working["frames"])
    single_rows = [r for r in rows if r.cls in SINGLE_CHANNEL]
    scoreboard = {
        "schema": SCHEMA,
        "working_set": {
            k: working[k]
            for k in ("frames", "click_frames", "rows", "rows_by_view", "rows_by_class", "stride")
        },
        "sam3": None
        if sam3_runs is None
        else {k: v for k, v in sam3_runs.items() if k != "runs"} | {"d_estimates": d_rows},
        "coverage": coverage_table(estimates, rows),
        "coverage_single_channel": coverage_table(
            [e for e in estimates if e.cls in SINGLE_CHANNEL], single_rows
        ),
        "consistency_3d": consistency_table(tips3d),
        "agreement": agreement_table(estimates, rows_by_key),
        "agreement_single_channel": agreement_table(
            [e for e in estimates if e.cls in SINGLE_CHANNEL], rows_by_key
        ),
        "anchors": anchor_table(estimates, rows_by_key, cells, tips3d, anchors3d),
        "anchors_single_channel": anchor_table(
            estimates, rows_by_key, [c for c in cells if c.cls in SINGLE_CHANNEL], tips3d, anchors3d
        ),
        "anchors_3d_available": len(anchors3d),
        "two_state": two_state_table(tips3d),
        "two_state_by_method": {m: two_state_table(tips3d, m) for m in ("B", "C", "D")},
        "timeline_blue": None,
        "elapsed_seconds": None,
        "claim_boundary": CLAIM_BOUNDARY,
        "licence": LICENCE_NOTE,
    }
    timeline = blue_timeline(
        estimates,
        rows_by_key,
        tips3d,
        tables,
        working["frames"],
        body_length_cm=body_length_cm,
    )
    fs_common.write_json(output / "timeline_blue.json", timeline)
    scoreboard["timeline_blue"] = {k: v for k, v in timeline.items() if k != "series"}

    frames = ProxyFrames({v: root / p for v, p in clip["proxies"].items()}, offset)
    sheets = render_sheets(rows_by_key, estimates, cells, frames, output=output)
    frames.close()
    scoreboard["contact_sheets"] = sheets
    scoreboard["elapsed_seconds"] = round(perf_counter() - started, 1)
    fs_common.write_json(output / "scoreboard.json", scoreboard)
    (output / "scoreboard.md").write_text(
        scoreboard_markdown(scoreboard, working), encoding="utf-8"
    )
    elapsed = scoreboard["elapsed_seconds"]
    print(f"scored {len(estimates)} estimates, {len(tips3d)} 3D tips in {elapsed} s")
    return 0


# --------------------------------------------------------------------------------------------
# the scoreboard page


def _f(value: Any, digits: int = 1) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _pct_cell(block: Mapping[str, Any], digits: int = 1) -> str:
    if not block or block.get("n") in (None, 0):
        return "- (0)"
    spread = f"{_f(block['p10'], digits)}–{_f(block['p90'], digits)}"
    return f"{_f(block['median'], digits)} ({spread}) ({block['n']})"


def _table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out.extend("| " + " | ".join(str(c) for c in row) + " |" for row in rows)
    return out


def scoreboard_markdown(board: Mapping[str, Any], working: Mapping[str, Any]) -> str:
    """The tables of `scoreboard.json` as one page; the README's verdict is written by hand."""
    cov = board["coverage_single_channel"]
    cov_all = board["coverage"]
    con = board["consistency_3d"]
    anc = board["anchors_single_channel"]
    anc_all = board["anchors"]
    agr = board["agreement_single_channel"]
    two = board["two_state"]
    tl = board["timeline_blue"]
    lines = [
        "# Tip against body, P03_03_01: the tables",
        "",
        f"{working['rows']} pipette rows with an axis on {len(working['frames'])} frames "
        f"(the 30 click frames and a stride-{working['stride']} sweep of the window). "
        "Single-channel pipettes unless a table says otherwise. Cells are median (p10–p90) (n).",
        "",
        "## Coverage",
        "",
        *_table(
            (
                "method",
                "rows",
                "tip found",
                "no tip",
                "abstained",
                "found fraction",
                "tip length px blue",
                "yellow",
                "red",
            ),
            (
                (
                    m,
                    cov[m]["rows"],
                    cov[m]["tip_found"],
                    cov[m]["tip_absent"],
                    cov[m]["abstained"],
                    _f(cov[m]["found_fraction"], 3),
                    _pct_cell(cov[m]["tip_length_px_by_class"]["blue_pipette"]),
                    _pct_cell(cov[m]["tip_length_px_by_class"]["yellow_pipette"]),
                    _pct_cell(cov[m]["tip_length_px_by_class"]["red_pipette"]),
                )
                for m in METHODS
            ),
        ),
        "",
        "8-channel rows with a tip found, per method: "
        + ", ".join(f"{m} {_f(cov_all[m]['found_fraction_8_channel'], 3)}" for m in METHODS)
        + ".",
        "",
        "## Cross-view consistency, 3D",
        "",
        *_table(
            (
                "method",
                "frames with a 3D end",
                "with a tip vote",
                "tip length cm (voted frames)",
                "in 3–7 cm",
                "tip length cm, 8-channel",
                "total length cm",
                "with tip by length",
                "reprojection residual px",
            ),
            (
                (
                    m,
                    con[m]["frames_with_3d_end"],
                    con[m]["frames_with_tip_vote"],
                    _pct_cell(con[m]["tip_length_cm_single_channel"], 2),
                    _f(con[m]["tip_length_in_3_7_cm_fraction"], 2),
                    _pct_cell(con[m]["tip_length_cm_8_channel"], 2),
                    _pct_cell(con[m]["total_length_cm_single_channel"], 2),
                    _f(con[m]["with_tip_fraction_single_channel"], 2),
                    _pct_cell(con[m]["reprojection_residual_px"], 1),
                )
                for m in METHODS
            ),
        ),
        "",
        "## Pairwise agreement",
        "",
        *_table(
            (
                "pair",
                "both answered",
                "presence agreement",
                "both found",
                "ends within 10 px",
                "end gap px",
            ),
            (
                (
                    pair,
                    a["rows_both_answered"],
                    _f(a["presence_agreement"], 2),
                    a["rows_both_found"],
                    _f(a["end_within_gate"], 2),
                    _pct_cell(a["end_gap_px"]),
                )
                for pair, a in agr.items()
            ),
        ),
        "",
        "## Against the clicks",
        "",
        f"Clicked cells with a row: {anc['cells']['tip_end']} with the click at the row's tip "
        f"end, {anc['cells']['other_end']} at the other end (by class "
        f"{anc['other_end_by_class']}, by state {anc['other_end_by_state']}), "
        f"{anc['cells']['hidden']} hidden. The error columns use the tip-end cells; the "
        "all-cells column includes the other-end clicks.",
        "",
        *_table(
            (
                "method",
                "cells",
                "tip found there",
                "along px signed",
                "abs along px",
                "across px",
                "within 10 px",
                "abs along px, all cells",
                "3D tip error cm",
                "3D butt error cm",
                "anchors nearer the butt",
                "hidden cells",
                "end moved off marker",
                "no tip claimed",
            ),
            (
                (
                    m,
                    anc["methods"][m]["tip_end"]["cells"],
                    anc["methods"][m]["tip_end"]["tip_found"],
                    _pct_cell(anc["methods"][m]["tip_end"]["along_px_signed"]),
                    _pct_cell(anc["methods"][m]["tip_end"]["abs_along_px"]),
                    _pct_cell(anc["methods"][m]["tip_end"]["across_px"]),
                    _f(anc["methods"][m]["tip_end"]["within_gate_fraction"], 2),
                    _pct_cell(anc["methods"][m]["all"]["abs_along_px"]),
                    _pct_cell(anc["methods"][m]["error_3d_cm"], 2),
                    _pct_cell(anc["methods"][m]["butt_error_3d_cm"], 2),
                    anc["methods"][m]["anchor_nearer_butt"],
                    anc["methods"][m]["hidden_cells_with_row"],
                    anc["methods"][m]["hidden_end_moved_from_marker"],
                    anc["methods"][m]["hidden_no_tip_claimed"],
                )
                for m in METHODS
            ),
        ),
        "",
        "All classes, abs along px at the tip-end cells: "
        + ", ".join(
            f"{m} {_pct_cell(anc_all['methods'][m]['tip_end']['abs_along_px'])}" for m in METHODS
        )
        + ".",
        "",
        "## Two-state length (fused tip end to butt, 3D)",
        "",
        *_table(
            (
                "class",
                "frames",
                "total length cm",
                "with tip (prior + 2 cm)",
                "length with tip",
                "length without",
                "low mode cm (n)",
                "high mode cm (n)",
                "separation cm",
                "pooled spread cm",
                "bimodal",
            ),
            (
                (
                    cls,
                    t["frames"],
                    _pct_cell(t["total_length_cm"], 2),
                    f"{t['with_tip_frames']} ({_f(t['with_tip_fraction'], 2)})",
                    _pct_cell(t["total_length_cm_with_tip"], 2),
                    _pct_cell(t["total_length_cm_without_tip"], 2),
                    f"{_f(t.get('mode_low_cm'), 2)} ({t.get('n_low', '-')})",
                    f"{_f(t.get('mode_high_cm'), 2)} ({t.get('n_high', '-')})",
                    _f(t.get("separation_cm"), 2),
                    _f(t.get("pooled_spread_cm"), 2),
                    t.get("bimodal", "-"),
                )
                for cls, t in two.items()
            ),
        ),
        "",
        "## Blue pipette timeline",
        "",
        f"{tl['frames_with_verdict']} of {tl['frames']} working frames have a verdict "
        f"(by basis {tl['frames_by_basis']}), "
        f"{tl['frames_tip_present']} with a tip; {len(tl['steps'])} steps, "
        f"{tl['steps_with_rack_or_trash_near']} with a tip rack or trash box within "
        f"{NEAR_BOX_PX:g} px of the tip end; {tl['one_frame_flickers']} one-frame flickers.",
        "",
        *_table(
            ("frame", "from", "kind", "rack or trash near"),
            (
                (s["frame"], s["previous_frame"], s["kind"], ", ".join(s["near_boxes"]) or "-")
                for s in tl["steps"]
            ),
        ),
        "",
        CLAIM_BOUNDARY,
        "",
        LICENCE_NOTE,
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="step", required=True)
    prepare = sub.add_parser(
        "prepare", help="working set, methods A, B, C and the SAM3 box streams"
    )
    prepare.add_argument("--clip-config", type=Path, required=True)
    prepare.add_argument(
        "--observations",
        type=Path,
        required=True,
        help="observations.jsonl (or its directory) with detector and SAM3 axis rows",
    )
    prepare.add_argument(
        "--tips-workspace", type=Path, required=True, help="the tip clicks workspace"
    )
    prepare.add_argument(
        "--pipettes-config", type=Path, default=Path("configs/finebio/pipettes.json")
    )
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    prepare.add_argument("--no-frames", action="store_true", help="skip method C (no video read)")
    prepare.set_defaults(func=run_prepare)
    sam3 = sub.add_parser("sam3", help="the SAM3 worker on the tip box streams (GPU)")
    sam3.add_argument("--output", type=Path, required=True)
    sam3.add_argument("--view", action="append", default=[])
    sam3.add_argument("--allow-gpu-neighbour", type=int, action="append", default=[])
    sam3.set_defaults(func=run_sam3)
    score = sub.add_parser("score", help="methods D and E, the scoreboard, sheets and README")
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--tips-workspace", type=Path, required=True)
    score.add_argument("--tip-scoreboard", type=Path, default=None)
    score.set_defaults(func=run_score)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
