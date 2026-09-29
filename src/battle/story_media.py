"""Story media: the GIFs and stills of `docs/story.md`, composited from what the repo owns.

Rerun's CLI offers one screenshot at load and no time cursor, so every animation here is
built directly from the sources on disk: a proxy or a run's `input_*.mp4`, the run's
`observations.jsonl` (mask PNGs and normalised boxes per frame), the FineBio tracker's
`tracks.jsonl` rows, the rig's static points and camera poses.  `configs/story/media.json`
lists one entry per GIF or still; `battle-story-media render` writes `media/story/<day>-<id>.gif`
(or `.png`) plus `media/story/manifest.json`, which repeats every entry's sources, frame
range and any fallback taken.

Kinds:

* `overlay`: one video tile with mask tints, outlines and boxes from its observations.
* `grid`: several such tiles side by side (methods, views, arms), `columns` per row.
* `world`: the FineBio layout, camera tiles above a top-down bench panel drawn from
  `Track3D` rows, the rig's static objects, the container volumes and the camera centres;
  masks and points share one colour per track id.  `options.line_tracks` names a second
  tracks file whose rows carry `endpoints_cm` (the pipettes-as-lines extension, Sep 28):
  each segment is drawn on the panel and, through the fixed cameras in `options.cameras`,
  projected into its tile, with the resolved tip as a white dot.
* `still`: one or more images tiled into a PNG.
* `cells`: crops around boxes on chosen frames (a human gate's accept / reject tiles).
* `rerun_still`: `rerun --headless --screenshot-to` on a recording and a blueprint.

A missing source never drops a day: the entry falls back to its `fallback` image (the day's
contact sheet), or to a placeholder naming the failure, and the manifest says so.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SCHEMA = "battle-story-media/1"
KINDS = ("overlay", "grid", "world", "still", "cells", "rerun_still")
ANIMATED_KINDS = ("overlay", "grid", "world")
DEFAULT_FPS = 10
DEFAULT_WIDTH = 640
DEFAULT_MAX_BYTES = 3_000_000
DEFAULT_TARGET_BYTES = 1_500_000
CAPTION_HEIGHT = 22
LABEL_HEIGHT = 16
FONT = cv2.FONT_HERSHEY_SIMPLEX
DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")

# Distinct tints (BGR), indexed by a stable hash of the label or track id.
PALETTE: tuple[tuple[int, int, int], ...] = (
    (80, 200, 255),  # amber
    (255, 160, 60),  # sky
    (90, 230, 120),  # green
    (230, 90, 230),  # magenta
    (60, 90, 255),  # red
    (255, 240, 90),  # cyan
    (140, 110, 255),  # salmon
    (255, 120, 200),  # violet
    (60, 220, 220),  # yellow
    (200, 200, 200),  # grey
)
MUTED = (120, 120, 120)
ACCEPT = (90, 200, 90)
REJECT = (60, 60, 230)
TIP = (255, 255, 255)


# --------------------------------------------------------------------------- manifest


@dataclass(frozen=True)
class FrameRange:
    """`start .. stop` (exclusive) on the source clock, every `stride`-th frame."""

    start: int
    stop: int
    stride: int = 1

    def indices(self) -> list[int]:
        return list(range(self.start, self.stop, self.stride))

    def to_record(self) -> dict[str, int]:
        return {"start": self.start, "stop": self.stop, "stride": self.stride}


@dataclass
class MediaEntry:
    id: str
    day: str
    kind: str
    caption: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    frames: FrameRange | None = None
    fps: int = DEFAULT_FPS
    width: int = DEFAULT_WIDTH
    options: dict[str, Any] = field(default_factory=dict)
    fallback: str | None = None
    max_bytes: int = DEFAULT_MAX_BYTES
    target_bytes: int = DEFAULT_TARGET_BYTES
    struggle: str = ""

    @property
    def extension(self) -> str:
        return "gif" if self.kind in ANIMATED_KINDS else "png"

    @property
    def output_name(self) -> str:
        return f"{self.day}-{self.id}.{self.extension}"


def _frame_range(raw: Any, entry_id: str) -> FrameRange:
    if not isinstance(raw, dict):
        raise ValueError(f"{entry_id}: frames must be an object with start/stop/stride")
    start, stop = int(raw["start"]), int(raw["stop"])
    stride = int(raw.get("stride", 1))
    if start < 0 or stop <= start or stride < 1:
        raise ValueError(f"{entry_id}: bad frame range {raw}")
    return FrameRange(start, stop, stride)


def load_manifest(path: Path) -> list[MediaEntry]:
    """Parse `configs/story/media.json`; every entry validated, ids unique, days dated."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema must be {SCHEMA!r}, got {doc.get('schema')!r}")
    defaults = doc.get("defaults") or {}
    entries: list[MediaEntry] = []
    seen: set[str] = set()
    for raw in doc.get("entries") or []:
        entry_id = str(raw.get("id") or "")
        if not ID_RE.match(entry_id):
            raise ValueError(f"bad entry id {entry_id!r} (lowercase slug expected)")
        day = str(raw.get("day") or "")
        if not DAY_RE.match(day):
            raise ValueError(f"{entry_id}: day must be YYYY-MM-DD, got {day!r}")
        kind = str(raw.get("kind") or "")
        if kind not in KINDS:
            raise ValueError(f"{entry_id}: kind {kind!r} not one of {KINDS}")
        caption = str(raw.get("caption") or "").strip()
        if not caption or "\n" in caption:
            raise ValueError(f"{entry_id}: caption must be one non-empty line")
        key = f"{day}-{entry_id}"
        if key in seen:
            raise ValueError(f"duplicate entry {key}")
        seen.add(key)
        frames = _frame_range(raw["frames"], entry_id) if kind in ANIMATED_KINDS else None
        entries.append(
            MediaEntry(
                id=entry_id,
                day=day,
                kind=kind,
                caption=caption,
                sources=list(raw.get("sources") or []),
                frames=frames,
                fps=int(raw.get("fps", defaults.get("fps", DEFAULT_FPS))),
                width=int(raw.get("width", defaults.get("width", DEFAULT_WIDTH))),
                options=dict(raw.get("options") or {}),
                fallback=raw.get("fallback"),
                max_bytes=int(raw.get("max_bytes", defaults.get("max_bytes", DEFAULT_MAX_BYTES))),
                target_bytes=int(
                    raw.get("target_bytes", defaults.get("target_bytes", DEFAULT_TARGET_BYTES))
                ),
                struggle=str(raw.get("struggle") or ""),
            )
        )
    if not entries:
        raise ValueError(f"{path}: no entries")
    return entries


# --------------------------------------------------------------------------- drawing


def colour_for(key: str) -> tuple[int, int, int]:
    """A palette tint chosen by a stable hash of `key` (same id, same colour, every run)."""
    digest = 0
    for ch in key:
        digest = (digest * 131 + ord(ch)) % 1_000_003
    return PALETTE[digest % len(PALETTE)]


def put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.4,
    colour: tuple[int, int, int] = (245, 245, 245),
    thickness: int = 1,
    shadow: bool = True,
) -> None:
    if shadow:
        cv2.putText(image, text, origin, FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, origin, FONT, scale, colour, thickness, cv2.LINE_AA)


def tint_mask(
    image: np.ndarray,
    mask: np.ndarray,
    colour: tuple[int, int, int],
    *,
    alpha: float = 0.45,
    outline: int = 1,
) -> None:
    """Blend `colour` over `mask` (bool, image-sized) in place and draw its contour."""
    if mask.shape[:2] != image.shape[:2]:
        mask = (
            cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=0) > 0
        )
    if not mask.any():
        return
    region = image[mask].astype(np.float32)
    image[mask] = (region * (1 - alpha) + np.array(colour, dtype=np.float32) * alpha).astype(
        np.uint8
    )
    if outline > 0:
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(image, contours, -1, colour, outline, cv2.LINE_AA)


def fit_crop(crop: Sequence[float], width: int, height: int, aspect: float) -> tuple[int, ...]:
    """Grow a normalised `[x0, y0, x1, y1]` window to `aspect` (w / h) inside the image."""
    x0, y0, x1, y1 = (float(v) for v in crop)
    cw, ch = max(x1 - x0, 1e-3) * width, max(y1 - y0, 1e-3) * height
    if cw / ch < aspect:
        cw = ch * aspect
    else:
        ch = cw / aspect
    cx, cy = (x0 + x1) / 2 * width, (y0 + y1) / 2 * height
    cw, ch = min(cw, width), min(ch, height)
    left = int(round(min(max(cx - cw / 2, 0), width - cw)))
    top = int(round(min(max(cy - ch / 2, 0), height - ch)))
    return left, top, left + int(round(cw)), top + int(round(ch))


def scale_to_width(image: np.ndarray, width: int) -> np.ndarray:
    if image.shape[1] == width:
        return image
    height = max(1, int(round(image.shape[0] * width / image.shape[1])))
    interpolation = cv2.INTER_AREA if width < image.shape[1] else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=interpolation)


def letterbox(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Fit `image` inside `size` (w, h) on a dark background, aspect kept."""
    w, h = size
    scale = min(w / image.shape[1], h / image.shape[0])
    tw, th = max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))
    resized = cv2.resize(image, (tw, th), interpolation=cv2.INTER_AREA)
    canvas = np.full((h, w, 3), 18, dtype=np.uint8)
    x, y = (w - tw) // 2, (h - th) // 2
    canvas[y : y + th, x : x + tw] = resized
    return canvas


def tile_grid(tiles: Sequence[np.ndarray], columns: int, *, gap: int = 2) -> np.ndarray:
    """Equal-size tiles in rows of `columns`; a short last row is padded with background."""
    if not tiles:
        raise ValueError("no tiles")
    if columns < 1:
        raise ValueError("columns must be >= 1")
    h, w = tiles[0].shape[:2]
    for tile in tiles:
        if tile.shape[:2] != (h, w):
            raise ValueError("tiles must share one size")
    rows = (len(tiles) + columns - 1) // columns
    canvas = np.full(
        (rows * h + (rows - 1) * gap, columns * w + (columns - 1) * gap, 3), 18, np.uint8
    )
    for index, tile in enumerate(tiles):
        r, c = divmod(index, columns)
        y, x = r * (h + gap), c * (w + gap)
        canvas[y : y + h, x : x + w] = tile
    return canvas


def caption_bar(
    width: int, text: str, *, height: int = CAPTION_HEIGHT, reserve_right: int = 0
) -> np.ndarray:
    """The caption under a frame; a caption too wide for `width - reserve_right` wraps onto
    a second line rather than running under the frame label."""
    scale = 0.42
    limit = max(width - reserve_right - 12, 40)
    lines: list[str] = []
    line = ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if cv2.getTextSize(trial, FONT, scale, 1)[0][0] > limit and line:
            lines.append(line)
            line = word
        else:
            line = trial
    lines.append(line)
    lines = lines[:2]
    bar = np.full((height * len(lines), width, 3), 24, dtype=np.uint8)
    for i, row in enumerate(lines):
        put_text(bar, row, (6, height * (i + 1) - 7), scale=scale, shadow=False)
    return bar


def label_tile(tile: np.ndarray, text: str) -> None:
    """Burn a small label into the tile's top-left corner."""
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.38, 1)
    cv2.rectangle(tile, (0, 0), (tw + 8, th + 8), (0, 0, 0), -1)
    put_text(tile, text, (4, th + 4), scale=0.38, shadow=False)


# --------------------------------------------------------------------------- observations


@dataclass(frozen=True)
class FrameObject:
    label: str
    object_id: str
    box: tuple[float, float, float, float] | None  # normalised x, y, w, h
    mask_uri: str | None


def read_observations(path: Path) -> dict[int, list[FrameObject]]:
    """`analysis_frame_index` -> objects, tolerant of the worker and exporter layouts (a
    `mask` object with `uri`, or a bare `mask_uri`); duplicates of one object per frame keep
    the first row with a mask."""
    out: dict[int, list[FrameObject]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            frame = int(row["analysis_frame_index"])
            objects = out.setdefault(frame, [])
            seen = {(o.object_id, o.label): i for i, o in enumerate(objects)}
            for obj in row.get("objects") or ():
                mask = obj.get("mask")
                uri = mask.get("uri") if isinstance(mask, dict) else obj.get("mask_uri")
                box = obj.get("box")
                norm = (
                    (float(box["x"]), float(box["y"]), float(box["width"]), float(box["height"]))
                    if isinstance(box, dict)
                    else None
                )
                item = FrameObject(
                    str(obj.get("label", "")), str(obj.get("object_id", "")), norm, uri
                )
                key = (item.object_id, item.label)
                if key in seen:
                    if objects[seen[key]].mask_uri is None and uri is not None:
                        objects[seen[key]] = item
                    continue
                seen[key] = len(objects)
                objects.append(item)
    return out


def read_mask(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None if image is None else image > 0


def read_frames(video: Path, indices: Sequence[int]) -> dict[int, np.ndarray]:
    """Frames at `indices` (BGR): one seek to the first, then sequential reads."""
    wanted = sorted(set(int(i) for i in indices))
    out: dict[int, np.ndarray] = {}
    if not wanted:
        return out
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video {video}")
        index = wanted[0]
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        remaining = set(wanted)
        last = wanted[-1]
        while remaining and index <= last:
            ok, frame = cap.read()
            if not ok:
                break
            if index in remaining:
                out[index] = frame
                remaining.discard(index)
            index += 1
    finally:
        cap.release()
    return out


# --------------------------------------------------------------------------- tiles


@dataclass
class TileSource:
    """One video tile: the clip, its observations and how to colour them."""

    label: str
    video: Path
    observations: dict[int, list[FrameObject]]
    run: Path
    frame_offset: int = 0
    crop: Any = None
    view: str | None = None
    labels: tuple[str, ...] | None = None
    show_boxes: bool = True
    show_masks: bool = True
    alpha: float = 0.45

    def objects_at(self, frame: int) -> list[FrameObject]:
        rows = self.observations.get(frame + self.frame_offset, [])
        if self.labels is not None:
            rows = [r for r in rows if label_matches(r.label, self.labels)]
        return rows


def label_matches(label: str, patterns: Sequence[str]) -> bool:
    """Exact match, or a prefix match for a pattern ending in `*` (`blue_pipette#*`)."""
    return any(label.startswith(p[:-1]) if p.endswith("*") else label == p for p in patterns)


def auto_crop(source: TileSource, frames: Iterable[int], margin: float = 0.12) -> list[float]:
    """The union of the source's boxes over `frames`, padded, as a normalised window."""
    xs0, ys0, xs1, ys1 = [], [], [], []
    for frame in frames:
        for obj in source.objects_at(frame):
            if obj.box is None:
                continue
            x, y, w, h = obj.box
            xs0.append(x)
            ys0.append(y)
            xs1.append(x + w)
            ys1.append(y + h)
    if not xs0:
        return [0.0, 0.0, 1.0, 1.0]
    x0, y0, x1, y1 = min(xs0), min(ys0), max(xs1), max(ys1)
    mx, my = (x1 - x0) * margin + 0.02, (y1 - y0) * margin + 0.02
    return [max(0.0, x0 - mx), max(0.0, y0 - my), min(1.0, x1 + mx), min(1.0, y1 + my)]


@dataclass(frozen=True)
class TileTransform:
    """How a full-frame pixel lands on a composited tile: the crop window's origin, the
    scale and the letterbox padding (`composite_tile` and any later drawing share it)."""

    x0: int
    y0: int
    x1: int
    y1: int
    scale: float
    inner: tuple[int, int]
    pad: tuple[int, int]

    def to_tile(self, u: float, v: float) -> tuple[int, int]:
        return (
            int(round((u - self.x0) * self.scale)) + self.pad[0],
            int(round((v - self.y0) * self.scale)) + self.pad[1],
        )


def tile_transform(
    frame_shape: Sequence[int], size: tuple[int, int], crop: Sequence[float] | None
) -> TileTransform:
    height, width = int(frame_shape[0]), int(frame_shape[1])
    if crop is None:
        # No crop asked for: the whole frame, letterboxed (an ego 4:3 clip keeps its edges).
        x0, y0, x1, y1 = 0, 0, width, height
    else:
        x0, y0, x1, y1 = fit_crop(crop, width, height, size[0] / size[1])
    scale = min(size[0] / max(x1 - x0, 1), size[1] / max(y1 - y0, 1))
    inner = (max(1, int(round((x1 - x0) * scale))), max(1, int(round((y1 - y0) * scale))))
    pad = ((size[0] - inner[0]) // 2, (size[1] - inner[1]) // 2)
    return TileTransform(x0, y0, x1, y1, scale, inner, pad)


def composite_tile(
    frame: np.ndarray,
    objects: Sequence[FrameObject],
    *,
    size: tuple[int, int],
    run: Path,
    crop: Sequence[float] | None = None,
    colour_of: Any = None,
    show_boxes: bool = True,
    show_masks: bool = True,
    alpha: float = 0.45,
    tag_of: Any = None,
) -> np.ndarray:
    """Masks tinted and outlined, boxes drawn, then cropped and scaled to `size` (w, h).

    `colour_of(obj)` returns a BGR tint or None to skip the object; `tag_of(obj)` an optional
    short text drawn at the box's top-left.
    """
    height, width = frame.shape[:2]
    transform = tile_transform(frame.shape, size, crop)
    x0, y0, x1, y1 = transform.x0, transform.y0, transform.x1, transform.y1
    scale, inner, pad = transform.scale, transform.inner, transform.pad
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    view = np.full((size[1], size[0], 3), 18, dtype=np.uint8)
    view[pad[1] : pad[1] + inner[1], pad[0] : pad[0] + inner[0]] = cv2.resize(
        frame[y0:y1, x0:x1], inner, interpolation=interpolation
    )
    for obj in objects:
        colour = colour_of(obj) if colour_of else colour_for(obj.label)
        if colour is None:
            continue
        if show_masks and obj.mask_uri:
            mask = read_mask(run / obj.mask_uri)
            if mask is not None:
                if mask.shape[:2] != (height, width):
                    mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=0) > 0
                small = cv2.resize(mask[y0:y1, x0:x1].astype(np.uint8), inner, interpolation=0)
                placed = np.zeros((size[1], size[0]), dtype=bool)
                placed[pad[1] : pad[1] + inner[1], pad[0] : pad[0] + inner[0]] = small > 0
                tint_mask(view, placed, colour, alpha=alpha)
        if obj.box is not None and (show_boxes or not obj.mask_uri):
            bx, by, bw, bh = obj.box
            p0 = (
                int((bx * width - x0) * scale) + pad[0],
                int((by * height - y0) * scale) + pad[1],
            )
            p1 = (
                int(((bx + bw) * width - x0) * scale) + pad[0],
                int(((by + bh) * height - y0) * scale) + pad[1],
            )
            cv2.rectangle(view, p0, p1, colour, 1)
            if tag_of is not None:
                tag = tag_of(obj)
                if tag:
                    put_text(view, tag, (p0[0] + 2, max(p0[1] - 3, 10)), scale=0.34, colour=colour)
    return view


# --------------------------------------------------------------------------- tracks / world


@dataclass(frozen=True)
class TrackRow:
    frame_index: int
    track_id: str
    object_class: str
    position: tuple[float, float, float]
    state: str
    support_slots: dict[str, str]
    # The line extension's fields, None on a point track: the segment's two endpoints (cm),
    # whether `endpoints[0]` is the tip, and the plurality of the per-view detector classes.
    endpoints: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
    tip_resolved: bool | None = None
    observed_class: str | None = None

    @property
    def shown_class(self) -> str:
        return self.observed_class or self.object_class

    @property
    def tip(self) -> tuple[float, float, float] | None:
        return self.endpoints[0] if self.endpoints is not None and self.tip_resolved else None


def read_tracks(path: Path, frames: Iterable[int]) -> dict[int, list[TrackRow]]:
    """`Track3D` rows at `frames` (raw frame index), keyed by frame; the frame index is
    sliced out of the line before the JSON parse so a 60 MB file costs a second."""
    wanted = set(int(f) for f in frames)
    out: dict[int, list[TrackRow]] = {f: [] for f in wanted}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            marker = line.find('"frame_index":')
            if marker < 0:
                continue
            tail = line[marker + 14 : marker + 26]
            digits = tail.lstrip().split(",", 1)[0].split("}", 1)[0].strip()
            if not digits.isdigit() or int(digits) not in wanted:
                continue
            row = json.loads(line)
            ends = row.get("endpoints_cm")
            out[int(row["frame_index"])].append(
                TrackRow(
                    int(row["frame_index"]),
                    str(row["track_id"]),
                    str(row["object_class"]),
                    tuple(float(v) for v in row["position_cm"]),  # type: ignore[arg-type]
                    str(row["state"]),
                    dict(row.get("support_slots") or {}),
                    endpoints=(
                        (tuple(float(v) for v in ends[0]), tuple(float(v) for v in ends[1]))  # type: ignore[arg-type]
                        if ends
                        else None
                    ),
                    tip_resolved=row.get("tip_resolved"),
                    observed_class=row.get("observed_class"),
                )
            )
    return out


def camera_centres(config: dict[str, Any]) -> dict[str, tuple[float, float, float]]:
    """Fixed camera centres (cm) from a FineBio camera config's rvec / tvec."""
    out = {}
    for view, cam in (config.get("fixed") or {}).items():
        rvec = np.asarray(cam["rvec"], dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(cam["tvec"], dtype=np.float64).reshape(3, 1)
        rot = cv2.Rodrigues(rvec)[0]
        centre = (-rot.T @ tvec).ravel()
        out[view] = (float(centre[0]), float(centre[1]), float(centre[2]))
    return out


@dataclass
class WorldScene:
    """Everything the top-down panel draws besides the per-frame track rows."""

    static: list[dict[str, Any]] = field(default_factory=list)  # class, point_cm
    containers: list[dict[str, Any]] = field(default_factory=list)  # name, centre_xy_cm, half
    cameras: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    fpv_by_frame: dict[int, tuple[float, float, float]] = field(default_factory=dict)
    lid_intervals: list[tuple[int, int]] = field(default_factory=list)
    extent: tuple[float, float, float, float] | None = None  # x0, y0, x1, y1 in cm

    def lid_closed(self, raw_frame: int) -> bool:
        return any(a <= raw_frame < b for a, b in self.lid_intervals)


def scene_extent(
    scene: WorldScene, rows: Iterable[TrackRow], pad: float = 8.0
) -> tuple[float, ...]:
    xs, ys = [], []
    for item in scene.static:
        xs.append(float(item["point_cm"][0]))
        ys.append(float(item["point_cm"][1]))
    for row in rows:
        xs.append(row.position[0])
        ys.append(row.position[1])
    for centre in scene.cameras.values():
        xs.append(centre[0])
        ys.append(centre[1])
    if not xs:
        return (-50.0, -40.0, 50.0, 60.0)
    return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)


def render_world_panel(
    rows: Sequence[TrackRow],
    scene: WorldScene,
    *,
    size: tuple[int, int],
    raw_frame: int,
    highlight: Sequence[str] = (),
    title: str = "",
    colour_of: Any = None,
) -> np.ndarray:
    """Top-down bench (x right, y down, cm): static objects grey, containers boxed, cameras
    as labelled triangles, one dot per track coloured by id, highlighted ids labelled."""
    w, h = size
    panel = np.full((h, w, 3), 30, dtype=np.uint8)
    x0, y0, x1, y1 = scene.extent or scene_extent(scene, rows)
    scale = min((w - 20) / max(x1 - x0, 1e-3), (h - 20) / max(y1 - y0, 1e-3))
    ox = (w - (x1 - x0) * scale) / 2
    oy = (h - (y1 - y0) * scale) / 2

    def to_px(x: float, y: float) -> tuple[int, int]:
        return int(round(ox + (x - x0) * scale)), int(round(oy + (y - y0) * scale))

    for gx in range(int(np.floor(x0 / 10)) * 10, int(np.ceil(x1 / 10)) * 10 + 1, 10):
        cv2.line(panel, to_px(gx, y0), to_px(gx, y1), (42, 42, 42), 1)
    for gy in range(int(np.floor(y0 / 10)) * 10, int(np.ceil(y1 / 10)) * 10 + 1, 10):
        cv2.line(panel, to_px(x0, gy), to_px(x1, gy), (42, 42, 42), 1)
    for item in scene.static:
        px, py = to_px(float(item["point_cm"][0]), float(item["point_cm"][1]))
        cv2.rectangle(panel, (px - 4, py - 4), (px + 4, py + 4), (110, 110, 110), -1)
        put_text(panel, str(item["class"]), (px + 6, py + 4), scale=0.3, colour=(170, 170, 170))
    for box in scene.containers:
        cx, cy = box["centre_xy_cm"]
        hx, hy = float(box.get("half_x_cm", 5)), float(box.get("half_y_cm", 5))
        closed = box.get("name") == "centrifuge" and scene.lid_closed(raw_frame)
        colour = (60, 60, 220) if closed else (160, 160, 90)
        cv2.rectangle(
            panel, to_px(cx - hx, cy - hy), to_px(cx + hx, cy + hy), colour, 2 if closed else 1
        )
        if closed:
            px, py = to_px(cx - hx, cy - hy)
            put_text(panel, "lid closed", (px, max(py - 4, 10)), scale=0.36, colour=(80, 80, 255))
    for name, centre in scene.cameras.items():
        # A camera outside the drawn extent sits clamped on the border, its label marked.
        inside = x0 <= centre[0] <= x1 and y0 <= centre[1] <= y1
        px, py = to_px(min(max(centre[0], x0), x1), min(max(centre[1], y0), y1))
        px, py = min(max(px, 8), w - 8), min(max(py, 8), h - 8)
        pts = np.array([[px, py - 6], [px - 5, py + 4], [px + 5, py + 4]], np.int32)
        cv2.fillPoly(panel, [pts], (200, 170, 90))
        label = name if inside else f"{name} (off panel)"
        put_text(panel, label, (px + 6, py + 4), scale=0.32, colour=(220, 200, 140))
    fpv = scene.fpv_by_frame.get(raw_frame)
    if fpv is not None:
        px, py = to_px(fpv[0], fpv[1])
        cv2.circle(panel, (px, py), 5, (200, 170, 90), 1)
        put_text(panel, "fpv", (px + 6, py + 4), scale=0.32, colour=(220, 200, 140))
    for row in rows:
        if row.state == "lost":
            continue
        colour = colour_of(row.track_id) if colour_of else colour_for(row.track_id)
        px, py = to_px(row.position[0], row.position[1])
        filled = row.state in ("observed", "contained", "held")
        if row.endpoints is not None:
            # A line track: the segment in the id's colour and its tip as a white dot; no
            # midpoint dot, so it reads apart from the point tracker's dots.
            a = to_px(row.endpoints[0][0], row.endpoints[0][1])
            b = to_px(row.endpoints[1][0], row.endpoints[1][1])
            cv2.line(panel, a, b, colour, 2 if filled else 1, cv2.LINE_AA)
            if row.tip is not None:
                cv2.circle(panel, a, 4, TIP, -1)
                cv2.circle(panel, a, 4, colour, 1)
        else:
            cv2.circle(panel, (px, py), 4, colour, -1 if filled else 1)
        if row.track_id in highlight:
            cv2.circle(panel, (px, py), 9, colour, 2)
            put_text(
                panel, f"{row.track_id} [{row.state}]", (px + 12, py - 8), scale=0.38, colour=colour
            )
    if title:
        put_text(panel, title, (8, h - 8), scale=0.36, colour=(200, 200, 200))
    return panel


# --------------------------------------------------------------------------- gif encode


def encode_gif(
    frame_dir: Path,
    output: Path,
    fps: int,
    *,
    max_colors: int = 128,
    dither: str = "bayer:bayer_scale=5",
) -> None:
    """Two-pass palettegen / paletteuse over `frame_dir/%05d.png`."""
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("ffmpeg is required to write GIFs")
    filters = (
        f"split[a][b];[a]palettegen=max_colors={max_colors}:stats_mode=diff[p];"
        f"[b][p]paletteuse=dither={dither}:diff_mode=rectangle"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-i",
        str(frame_dir / "%05d.png"),
        "-lavfi",
        filters,
        "-loop",
        "0",
        str(output),
    ]
    subprocess.run(cmd, check=True)


def write_gif(
    frames: Sequence[np.ndarray],
    output: Path,
    fps: int,
    *,
    max_bytes: int,
    target_bytes: int | None = None,
    min_width: int = 480,
) -> dict[str, Any]:
    """Encode, then shrink the palette and the width until the file fits.

    `target_bytes` (default `DEFAULT_TARGET_BYTES`) is the size aimed for while the width
    stays at or above `min_width`; `max_bytes` is the hard cap that keeps shrinking below it.
    """
    if not frames:
        raise ValueError("no frames to encode")
    target = DEFAULT_TARGET_BYTES if target_bytes is None else target_bytes
    attempts: list[dict[str, Any]] = []
    ladder = [(128, 1.0), (64, 1.0), (64, 0.85), (48, 0.75), (32, 0.65), (32, 0.55)]
    width = frames[0].shape[1]
    for colours, scale in ladder:
        scaled = int(width * scale)
        if attempts and scaled < min_width and attempts[-1]["size_bytes"] <= max_bytes:
            break
        with tempfile.TemporaryDirectory(prefix="story-media-") as tmp:
            tmp_dir = Path(tmp)
            for index, frame in enumerate(frames):
                image = frame if scale == 1.0 else scale_to_width(frame, scaled)
                cv2.imwrite(str(tmp_dir / f"{index:05d}.png"), image)
            encode_gif(tmp_dir, output, fps, max_colors=colours)
        size = output.stat().st_size
        attempts.append(
            {"colours": colours, "scale": round(scale, 2), "width": scaled, "size_bytes": size}
        )
        if size <= target:
            break
    return {
        "attempts": attempts,
        "size_bytes": output.stat().st_size,
        "target_bytes": target,
        "max_bytes": max_bytes,
    }


# --------------------------------------------------------------------------- rendering


class MissingSource(FileNotFoundError):
    """A source the entry needs is not on disk (the Sep 24 prune or a path typo)."""


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _require(root: Path, value: str | Path, what: str) -> Path:
    path = _resolve(root, value)
    if not path.exists():
        raise MissingSource(f"{what} missing: {value}")
    return path


def load_tile_source(raw: dict[str, Any], root: Path) -> TileSource:
    video = _require(root, raw["video"], "video")
    observations: dict[int, list[FrameObject]] = {}
    run = video.parent
    if raw.get("observations"):
        obs_path = _require(root, raw["observations"], "observations")
        observations = read_observations(obs_path)
        run = obs_path.parent
    if raw.get("run"):
        run = _require(root, raw["run"], "run directory")
    labels = raw.get("labels")
    return TileSource(
        label=str(raw.get("label") or video.stem),
        video=video,
        observations=observations,
        run=run,
        frame_offset=int(raw.get("frame_offset", 0)),
        crop=raw.get("crop"),
        view=raw.get("view"),
        labels=tuple(labels) if labels else None,
        show_boxes=bool(raw.get("show_boxes", True)),
        show_masks=bool(raw.get("show_masks", True)),
        alpha=float(raw.get("alpha", 0.45)),
    )


def _tile_size(width: int, columns: int, aspect: float, gap: int = 2) -> tuple[int, int]:
    tile_w = (width - gap * (columns - 1)) // columns
    return tile_w, max(1, int(round(tile_w / aspect)))


def render_tiles(
    entry: MediaEntry,
    root: Path,
    *,
    colour_of: Any = None,
    tag_of: Any = None,
    per_frame_hook: Any = None,
    per_tile_hook: Any = None,
    legend_extra: Sequence[tuple[str, tuple[int, int, int]]] = (),
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """The per-frame tile grid of an overlay / grid / world entry (before any panel).

    `per_tile_hook(source, frame, tile, transform)` draws on one composited tile in place,
    with the `TileTransform` that maps full-frame pixels onto it; `legend_extra` adds rows
    to the legend strip under the tiles.
    """
    assert entry.frames is not None
    sources = [load_tile_source(raw, root) for raw in entry.sources]
    if not sources:
        raise MissingSource("no sources")
    columns = int(entry.options.get("columns", min(len(sources), 3)))
    aspect = float(entry.options.get("tile_aspect", 16 / 9))
    size = _tile_size(entry.width, columns, aspect)
    indices = entry.frames.indices()
    crops: list[list[float] | None] = []
    for source in sources:
        if source.crop == "auto":
            crops.append(auto_crop(source, indices))
        else:
            crops.append(list(source.crop) if source.crop else None)
    # `shared_crop`: every tile uses one window, the union of the auto crops or the crop of
    # the source at that index (the arm that holds, so the arm that drifts leaves the frame).
    shared = entry.options.get("shared_crop")
    if shared is not None:
        if isinstance(shared, bool):
            found = [c for c in crops if c is not None]
            window = (
                [
                    min(c[0] for c in found),
                    min(c[1] for c in found),
                    max(c[2] for c in found),
                    max(c[3] for c in found),
                ]
                if found
                else None
            )
        else:
            window = crops[int(shared)]
        crops = [window for _ in crops]
    decoded = [read_frames(s.video, [i + s.frame_offset for i in indices]) for s in sources]
    legend: list[tuple[str, tuple[int, int, int]]] = [
        ("mask colour = 3D track id", PALETTE[1]),
        ("grey = no track claims the mask", MUTED),
    ]
    if colour_of is None:
        # One tint per label, assigned in sorted order so two labels never share a colour
        # and the legend under the tiles reads the same in every entry.
        labels = sorted({o.label for s in sources for f in indices for o in s.objects_at(f)})
        by_label = {label: PALETTE[i % len(PALETTE)] for i, label in enumerate(labels)}
        legend = [(label, by_label[label]) for label in labels]

        def colour_of(obj: FrameObject, view: str | None, frame: int) -> tuple[int, int, int]:
            return by_label.get(obj.label, MUTED)

    frames: list[np.ndarray] = []
    for frame in indices:
        tiles = []
        for s_index, source in enumerate(sources):
            image = decoded[s_index].get(frame + source.frame_offset)
            if image is None:
                tile = np.full((size[1], size[0], 3), 18, dtype=np.uint8)
                put_text(tile, "no frame", (6, size[1] // 2), scale=0.4)
            else:
                objects = source.objects_at(frame)
                tile = composite_tile(
                    image,
                    objects,
                    size=size,
                    run=source.run,
                    crop=crops[s_index],
                    colour_of=(lambda o, v=source.view, f=frame: colour_of(o, v, f))
                    if colour_of
                    else None,
                    show_boxes=source.show_boxes,
                    show_masks=source.show_masks,
                    alpha=source.alpha,
                    tag_of=(lambda o, v=source.view, f=frame: tag_of(o, v, f)) if tag_of else None,
                )
                if per_tile_hook is not None:
                    per_tile_hook(
                        source, frame, tile, tile_transform(image.shape, size, crops[s_index])
                    )
            label_tile(tile, source.label)
            tiles.append(tile)
        grid = tile_grid(tiles, columns)
        if per_frame_hook is not None:
            grid = per_frame_hook(frame, grid)
        frames.append(grid)
    records = [
        {
            "label": s.label,
            "video": str(_relative(root, s.video)),
            "observations": str(_relative(root, Path(raw["observations"])))
            if raw.get("observations")
            else None,
            "run": str(_relative(root, s.run)),
            "frame_offset": s.frame_offset,
            "crop": crops[i] if crops[i] is not None else None,
            "view": s.view,
        }
        for i, (s, raw) in enumerate(zip(sources, entry.sources, strict=True))
    ]
    legend = [*legend, *legend_extra]
    if legend:
        strip = legend_bar(frames[0].shape[1], legend) if frames else None
        if strip is not None:
            frames = [np.vstack([f, strip]) for f in frames]
    return frames, records


def legend_bar(
    width: int, items: Sequence[tuple[str, tuple[int, int, int]]], *, height: int = LABEL_HEIGHT
) -> np.ndarray:
    """Coloured squares and their labels on one dark strip (label -> tint of the tiles)."""
    bar = np.full((height, width, 3), 24, dtype=np.uint8)
    x = 6
    for label, colour in items:
        cv2.rectangle(bar, (x, 4), (x + 9, height - 5), colour, -1)
        put_text(bar, label, (x + 13, height - 5), scale=0.34, shadow=False)
        x += 13 + cv2.getTextSize(label, FONT, 0.34, 1)[0][0] + 12
        if x > width - 40:
            break
    return bar


def _relative(root: Path, path: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return path


def _finish_frames(
    entry: MediaEntry, frames: list[np.ndarray], label_of_frame: Any = None
) -> list[np.ndarray]:
    """The caption bar under every frame, with the per-frame label (frame number, id count)
    right-aligned on the same bar."""
    out = []
    burn = bool(entry.options.get("burn_caption", True))
    for index, frame in enumerate(frames):
        image = frame
        text = label_of_frame(index) if label_of_frame is not None else ""
        if burn:
            (tw, th), _ = cv2.getTextSize(text, FONT, 0.36, 1) if text else ((0, 0), 0)
            bar = caption_bar(image.shape[1], entry.caption, reserve_right=tw + 16 if text else 0)
            if text:
                put_text(bar, text, (image.shape[1] - tw - 8, CAPTION_HEIGHT - 7), scale=0.36)
            image = np.vstack([image, bar])
        elif text:
            (tw, th), _ = cv2.getTextSize(text, FONT, 0.36, 1)
            put_text(image, text, (image.shape[1] - tw - 8, th + 6), scale=0.36)
        out.append(image)
    return out


def load_world_scene(entry: MediaEntry, root: Path, raw_frames: Iterable[int]) -> WorldScene:
    opts = entry.options
    scene = WorldScene()
    if opts.get("rig"):
        rig = json.loads(_require(root, opts["rig"], "rig").read_text(encoding="utf-8"))
        scene.static = [
            {"class": s["class"], "point_cm": s["point_cm"]} for s in rig.get("static", ())
        ]
        wanted = set(raw_frames)
        for item in rig.get("fpv", ()):
            if int(item["frame"]) in wanted and item.get("fpv_centre_cm"):
                scene.fpv_by_frame[int(item["frame"])] = tuple(item["fpv_centre_cm"])  # type: ignore[assignment]
    if opts.get("cameras"):
        config = json.loads(
            _require(root, opts["cameras"], "camera config").read_text(encoding="utf-8")
        )
        scene.cameras = camera_centres(config)
    if opts.get("events_summary"):
        summary = json.loads(
            _require(root, opts["events_summary"], "events summary").read_text(encoding="utf-8")
        )
        scene.containers = [
            {
                "name": c["name"],
                "centre_xy_cm": c["centre_xy_cm"],
                "half_x_cm": c["half_x_cm"],
                "half_y_cm": c["half_y_cm"],
            }
            for c in summary.get("containers", ())
        ]
        scene.lid_intervals = [(int(a), int(b)) for a, b in summary.get("cycles_in_window", ())]
    if opts.get("extent"):
        scene.extent = tuple(float(v) for v in opts["extent"])  # type: ignore[assignment]
    return scene


def _track_colouring(
    tracks: dict[int, list[TrackRow]], offset: int, highlight: Sequence[str]
) -> tuple[Any, Any]:
    """Colour a view's slot by the track that claims it this frame; unclaimed slots muted."""

    def claim(view: str | None, frame: int, label: str) -> TrackRow | None:
        if view is None:
            return None
        for row in tracks.get(frame + offset, ()):
            if row.support_slots.get(view) == label:
                return row
        return None

    def colour_of(obj: FrameObject, view: str | None, frame: int) -> tuple[int, int, int] | None:
        row = claim(view, frame, obj.label)
        return colour_for(row.track_id) if row else MUTED

    def tag_of(obj: FrameObject, view: str | None, frame: int) -> str | None:
        row = claim(view, frame, obj.label)
        return row.track_id if row and row.track_id in highlight else None

    return colour_of, tag_of


def _id_counter(
    tracks: dict[int, list[TrackRow]], frames: Sequence[int], offset: int, object_class: str
) -> list[int]:
    """Distinct live track ids of `object_class` seen up to each frame of the clip."""
    seen: set[str] = set()
    counts = []
    for frame in frames:
        for row in tracks.get(frame + offset, ()):
            if row.object_class == object_class and row.state != "lost":
                seen.add(row.track_id)
        counts.append(len(seen))
    return counts


def _project(cam: Any, point: Sequence[float]) -> tuple[float, float] | None:
    """A world point through a fixed camera (`finebio_cameras.Camera`), None behind it."""
    p = np.asarray(point, dtype=np.float64)
    if float((cam.R @ p + cam.tvec.reshape(3))[2]) <= 0:
        return None
    u, v = cam.project(p)[0]
    return float(u), float(v)


def _line_id_counter(
    tracks: dict[int, list[TrackRow]], frames: Sequence[int], offset: int, shown_class: str
) -> list[int]:
    """Distinct live line-track ids whose observed class is `shown_class`, up to each frame."""
    seen: set[str] = set()
    counts = []
    for frame in frames:
        for row in tracks.get(frame + offset, ()):
            if row.endpoints is not None and row.shown_class == shown_class and row.state != "lost":
                seen.add(row.track_id)
        counts.append(len(seen))
    return counts


def render_world(entry: MediaEntry, root: Path) -> tuple[list[np.ndarray], dict[str, Any]]:
    assert entry.frames is not None
    opts = entry.options
    offset = int(opts.get("tracks_frame_offset", 0))
    indices = entry.frames.indices()
    raw_frames = [f + offset for f in indices]
    tracks = read_tracks(_require(root, opts["tracks"], "tracks"), raw_frames)
    scene = load_world_scene(entry, root, raw_frames)
    highlight = tuple(opts.get("highlight") or ())
    label_classes = tuple(opts.get("label_classes") or ())
    colour_of, tag_of = _track_colouring(tracks, offset, highlight)
    panel_h = int(opts.get("panel_height", 260))
    classes = opts.get("panel_classes")
    # A second tracks file of line rows (segments with a tip), drawn on the panel and, through
    # the fixed cameras, into the tiles; the point rows keep the mask colouring and the count.
    line_tracks: dict[int, list[TrackRow]] = {}
    cams: dict[str, Any] = {}
    if opts.get("line_tracks"):
        line_tracks = read_tracks(_require(root, opts["line_tracks"], "line tracks"), raw_frames)
        if opts.get("cameras"):
            from .finebio_cameras import cameras_from_config, read_camera_config

            cams = cameras_from_config(
                read_camera_config(_require(root, opts["cameras"], "camera config"))
            )

    def keep(row: TrackRow) -> bool:
        return not classes or row.shown_class in classes or row.track_id in highlight

    def line_rows_at(raw: int) -> list[TrackRow]:
        return [r for r in line_tracks.get(raw, ()) if r.endpoints is not None and keep(r)]

    all_rows = [r for rows in tracks.values() for r in rows if keep(r)]
    all_rows += [r for raw in raw_frames for r in line_rows_at(raw)]
    if scene.extent is None:
        scene.extent = scene_extent(scene, all_rows)  # type: ignore[assignment]

    def hook(frame: int, grid: np.ndarray) -> np.ndarray:
        raw = frame + offset
        rows = [r for r in tracks.get(raw, []) if keep(r)] + line_rows_at(raw)
        labelled = tuple(highlight) + tuple(
            r.track_id for r in rows if r.shown_class in label_classes
        )
        panel = render_world_panel(
            rows,
            scene,
            size=(grid.shape[1], panel_h),
            raw_frame=raw,
            highlight=labelled,
            title=f"top-down bench, cm | raw frame {raw} ({raw / 30:.1f} s)",
        )
        return np.vstack([grid, panel])

    def tile_hook(source: TileSource, frame: int, tile: np.ndarray, transform: TileTransform):
        cam = cams.get(source.view or "")
        if cam is None:
            return
        for row in line_rows_at(frame + offset):
            if row.state == "lost":
                continue
            ends = [_project(cam, p) for p in row.endpoints or ()]
            if any(p is None for p in ends):
                continue
            a, b = (transform.to_tile(*p) for p in ends)  # type: ignore[misc]
            colour = colour_for(row.track_id)
            cv2.line(tile, a, b, colour, 2, cv2.LINE_AA)
            if row.tip is not None:
                cv2.circle(tile, a, 5, TIP, -1, cv2.LINE_AA)
                cv2.circle(tile, a, 5, colour, 1, cv2.LINE_AA)

    frames, records = render_tiles(
        entry,
        root,
        colour_of=colour_of,
        tag_of=tag_of,
        per_frame_hook=hook,
        per_tile_hook=tile_hook if cams else None,
        legend_extra=(
            [("segment = the pipette tracked as a 3D line, white dot = its tip", TIP)]
            if line_tracks
            else []
        ),
    )
    detail = {
        "tiles": records,
        "tracks": opts["tracks"],
        "tracks_frame_offset": offset,
        "line_tracks": opts.get("line_tracks"),
        "rig": opts.get("rig"),
        "cameras": opts.get("cameras"),
        "events_summary": opts.get("events_summary"),
        "highlight": list(highlight),
    }
    label = None
    counter_class = opts.get("count_ids_of_class")
    if counter_class:
        counts = _id_counter(tracks, indices, offset, counter_class)
        detail["distinct_ids_in_clip"] = {counter_class: counts[-1] if counts else 0}
        if line_tracks:
            line_counts = _line_id_counter(line_tracks, indices, offset, counter_class)
            detail["distinct_line_ids_in_clip"] = {
                counter_class: line_counts[-1] if line_counts else 0
            }
            label = lambda i: f"{counter_class} ids: points {counts[i]}, lines {line_counts[i]}"  # noqa: E731
        else:
            label = lambda i: f"{counter_class} ids so far: {counts[i]}"  # noqa: E731
    return _finish_frames(entry, frames, label), detail


def render_overlay_or_grid(
    entry: MediaEntry, root: Path
) -> tuple[list[np.ndarray], dict[str, Any]]:
    assert entry.frames is not None
    opts = entry.options
    colour_of = tag_of = None
    detail: dict[str, Any] = {}
    counter_class = opts.get("count_ids_of_class")
    tracks: dict[int, list[TrackRow]] = {}
    offset = int(opts.get("tracks_frame_offset", 0))
    if opts.get("tracks"):
        raw_frames = [f + offset for f in entry.frames.indices()]
        tracks = read_tracks(_require(root, opts["tracks"], "tracks"), raw_frames)
        colour_of, tag_of = _track_colouring(tracks, offset, tuple(opts.get("highlight") or ()))
        detail["tracks"] = opts["tracks"]
        detail["tracks_frame_offset"] = offset
    elif opts.get("colour_by") == "object":
        colour_of = lambda o, v, f: colour_for(o.object_id)  # noqa: E731
    ids_at: list[int] = []
    if counter_class:
        ids_at = _id_counter(tracks, entry.frames.indices(), offset, counter_class)
        detail["distinct_ids_in_clip"] = {counter_class: ids_at[-1] if ids_at else 0}
    frames, records = render_tiles(entry, root, colour_of=colour_of, tag_of=tag_of)
    detail["tiles"] = records
    indices = entry.frames.indices()
    if counter_class:
        label = lambda i: f"{counter_class} ids so far: {ids_at[i]}"  # noqa: E731
    elif opts.get("frame_label", True):
        label = lambda i: f"frame {indices[i]}"  # noqa: E731
    else:
        label = None
    return _finish_frames(entry, frames, label), detail


def render_still(entry: MediaEntry, root: Path) -> tuple[np.ndarray, dict[str, Any]]:
    tiles = []
    records = []
    columns = int(entry.options.get("columns", len(entry.sources) or 1))
    for raw in entry.sources:
        path = _require(root, raw["image"], "image")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise MissingSource(f"unreadable image: {raw['image']}")
        if raw.get("crop"):
            h, w = image.shape[:2]
            cx0, cy0, cx1, cy1 = (float(v) for v in raw["crop"])
            image = image[int(cy0 * h) : int(cy1 * h), int(cx0 * w) : int(cx1 * w)]
            if image.size == 0:
                raise ValueError(f"empty crop {raw['crop']} on {raw['image']}")
        tiles.append((str(raw.get("label") or ""), image))
        records.append({"image": raw["image"], "label": raw.get("label"), "crop": raw.get("crop")})
    tile_w = (entry.width - 2 * (columns - 1)) // columns
    aspect = float(entry.options.get("tile_aspect", 0) or 0)
    if aspect <= 0:
        aspect = float(np.median([img.shape[1] / img.shape[0] for _, img in tiles]))
    size = (tile_w, max(1, int(round(tile_w / aspect))))
    rendered = []
    for label, image in tiles:
        tile = letterbox(image, size)
        if label:
            label_tile(tile, label)
        rendered.append(tile)
    grid = tile_grid(rendered, columns)
    if entry.options.get("burn_caption", True):
        grid = np.vstack([grid, caption_bar(grid.shape[1], entry.caption)])
    return grid, {"images": records}


def render_cells(entry: MediaEntry, root: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Crops around boxes: green border accepted, red rejected, the mask tinted when given."""
    opts = entry.options
    columns = int(opts.get("columns", 3))
    cell_w = (entry.width - 2 * (columns - 1)) // columns
    cell_h = int(opts.get("cell_height", cell_w))
    margin = float(opts.get("margin", 0.6))
    by_video: dict[str, list[int]] = {}
    for raw in entry.sources:
        by_video.setdefault(raw["video"], []).append(int(raw["frame"]))
    decoded = {
        video: read_frames(_require(root, video, "video"), frames)
        for video, frames in by_video.items()
    }
    tiles = []
    records = []
    for raw in entry.sources:
        frame = decoded[raw["video"]].get(int(raw["frame"]))
        if frame is None:
            raise MissingSource(f"frame {raw['frame']} of {raw['video']} unreadable")
        h, w = frame.shape[:2]
        bx0, by0, bx1, by1 = (float(v) for v in raw["box"])
        image = frame.copy()
        mask_note = None
        if raw.get("mask"):
            mask_path = _resolve(root, raw["mask"])
            mask = read_mask(mask_path) if mask_path.exists() else None
            if mask is None:
                mask_note = f"mask missing: {raw['mask']}"
            else:
                verdict_colour = REJECT if raw.get("verdict") == "reject" else ACCEPT
                tint_mask(image, mask, verdict_colour, alpha=0.4, outline=2)
        cv2.rectangle(image, (int(bx0), int(by0)), (int(bx1), int(by1)), (255, 255, 255), 2)
        mx, my = (bx1 - bx0) * margin + 8, (by1 - by0) * margin + 8
        crop = [(bx0 - mx) / w, (by0 - my) / h, (bx1 + mx) / w, (by1 + my) / h]
        x0, y0, x1, y1 = fit_crop(crop, w, h, cell_w / cell_h)
        tile = cv2.resize(image[y0:y1, x0:x1], (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        colour = REJECT if raw.get("verdict") == "reject" else ACCEPT
        cv2.rectangle(tile, (0, 0), (cell_w - 1, cell_h - 1), colour, 4)
        label_tile(tile, str(raw.get("label") or ""))
        note = str(raw.get("note") or "")
        if note:
            bar_h = 30
            bar = np.full((bar_h, cell_w, 3), 24, dtype=np.uint8)
            words, lines, line = note.split(), [], ""
            for word in words:
                trial = f"{line} {word}".strip()
                if cv2.getTextSize(trial, FONT, 0.33, 1)[0][0] > cell_w - 8 and line:
                    lines.append(line)
                    line = word
                else:
                    line = trial
            lines.append(line)
            for i, text in enumerate(lines[:2]):
                put_text(bar, text, (4, 12 + 13 * i), scale=0.33, shadow=False)
            tile = np.vstack([tile, bar])
        tiles.append(tile)
        record = {
            k: raw.get(k) for k in ("video", "frame", "box", "mask", "label", "verdict", "note")
        }
        if mask_note:
            record["mask_missing"] = mask_note
        records.append(record)
    max_h = max(t.shape[0] for t in tiles)
    tiles = [
        np.vstack([t, np.full((max_h - t.shape[0], t.shape[1], 3), 18, np.uint8)])
        if t.shape[0] < max_h
        else t
        for t in tiles
    ]
    grid = tile_grid(tiles, columns)
    if opts.get("burn_caption", True):
        grid = np.vstack([grid, caption_bar(grid.shape[1], entry.caption)])
    return grid, {"cells": records}


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def render_rerun_still(entry: MediaEntry, root: Path, output: Path) -> dict[str, Any]:
    """A viewer screenshot of `<rrd>` under `<rbl>`.

    `rerun --headless --screenshot-to` shoots before the recording has loaded (an empty
    layout), so the viewer is started headless, given `settle_seconds` to load, and asked
    for the screenshot through the SDK's `ViewerClient.save_screenshot`.  There is no time
    cursor control: the still shows the viewer's own choice of time (the last frame).
    """
    if not entry.sources:
        raise MissingSource("rerun_still needs a source with rrd (and blueprint)")
    raw = entry.sources[0]
    rrd = _require(root, raw["rrd"], "recording")
    blueprint = _require(root, raw["blueprint"], "blueprint") if raw.get("blueprint") else None
    window = str(entry.options.get("window_size", "1600x900"))
    settle = float(entry.options.get("settle_seconds", 30))
    port = _free_port()
    cmd = ["uv", "run", "rerun", "--headless", "--window-size", window, "--port", str(port)]
    cmd.append(str(rrd))
    if blueprint is not None:
        cmd.append(str(blueprint))
    output.unlink(missing_ok=True)
    process = subprocess.Popen(
        cmd, cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True
    )
    try:
        deadline = time.monotonic() + settle
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise MissingSource(f"rerun headless viewer exited early ({process.returncode})")
            time.sleep(0.5)
        from rerun.experimental import ViewerClient

        client = ViewerClient(f"rerun+http://127.0.0.1:{port}/proxy")
        client.save_screenshot(str(output))
        for _ in range(60):
            if output.exists() and output.stat().st_size > 0:
                break
            time.sleep(0.5)
        time.sleep(1.0)
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
    image = cv2.imread(str(output), cv2.IMREAD_COLOR) if output.exists() else None
    if image is None or float(image.std()) < 2.0:
        output.unlink(missing_ok=True)
        raise MissingSource("rerun headless screenshot missing or blank")
    if image.shape[1] != entry.width:
        image = scale_to_width(image, entry.width)
    if entry.options.get("burn_caption", True):
        image = np.vstack([image, caption_bar(image.shape[1], entry.caption)])
    cv2.imwrite(str(output), image)
    return {
        "rrd": raw["rrd"],
        "blueprint": raw.get("blueprint"),
        "command": " ".join(cmd),
        "screenshot": "rerun.experimental.ViewerClient.save_screenshot after "
        f"{settle:.0f} s settle; time cursor = viewer default (last frame)",
    }


def placeholder_image(width: int, text: str) -> np.ndarray:
    image = np.full((width * 9 // 16, width, 3), 28, dtype=np.uint8)
    put_text(image, "no media rendered", (12, 30), scale=0.6)
    for i, line in enumerate(_wrap(text, 70)[:6]):
        put_text(image, line, (12, 60 + 18 * i), scale=0.4, colour=(200, 200, 200))
    return image


def _wrap(text: str, limit: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        if len(line) + len(word) + 1 > limit and line:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        lines.append(line)
    return lines


def render_fallback(entry: MediaEntry, root: Path, output_dir: Path, reason: str) -> dict[str, Any]:
    """The day's contact sheet as a PNG (or a placeholder), with the reason recorded."""
    output = output_dir / f"{entry.day}-{entry.id}.png"
    used: str | None = None
    if entry.fallback:
        path = _resolve(root, entry.fallback)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR) if path.exists() else None
        if image is not None:
            image = scale_to_width(image, entry.width)
            if entry.options.get("burn_caption", True):
                image = np.vstack([image, caption_bar(image.shape[1], entry.caption)])
            used = entry.fallback
        else:
            reason = f"{reason}; fallback image missing too: {entry.fallback}"
            image = placeholder_image(entry.width, reason)
    else:
        reason = f"{reason}; no fallback image configured"
        image = placeholder_image(entry.width, reason)
    cv2.imwrite(str(output), image)
    return {
        "output": output,
        "fallback": {"reason": reason, "used": used, "placeholder": used is None},
        "detail": {},
    }


def render_entry(entry: MediaEntry, root: Path, output_dir: Path) -> dict[str, Any]:
    """Render one entry; the returned record is the manifest row (fallbacks included)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / entry.output_name
    detail: dict[str, Any] = {}
    fallback: dict[str, Any] | None = None
    encode: dict[str, Any] = {}
    try:
        if entry.kind in ("overlay", "grid"):
            frames, detail = render_overlay_or_grid(entry, root)
            encode = write_gif(
                frames,
                output,
                entry.fps,
                max_bytes=entry.max_bytes,
                target_bytes=entry.target_bytes,
            )
        elif entry.kind == "world":
            frames, detail = render_world(entry, root)
            encode = write_gif(
                frames,
                output,
                entry.fps,
                max_bytes=entry.max_bytes,
                target_bytes=entry.target_bytes,
            )
        elif entry.kind == "still":
            image, detail = render_still(entry, root)
            cv2.imwrite(str(output), image)
        elif entry.kind == "cells":
            image, detail = render_cells(entry, root)
            cv2.imwrite(str(output), image)
        elif entry.kind == "rerun_still":
            detail = render_rerun_still(entry, root, output)
        else:  # pragma: no cover - load_manifest refuses unknown kinds
            raise ValueError(entry.kind)
    except (
        MissingSource,
        FileNotFoundError,
        OSError,
        ValueError,
        KeyError,
        subprocess.CalledProcessError,
    ) as error:
        # A missing or unreadable source, or a config referencing a field the run lacks:
        # the day still gets its contact sheet and the manifest names the cause.
        reason = f"{type(error).__name__}: {error}"
        if output.suffix == ".gif" and output.exists():
            output.unlink()
        result = render_fallback(entry, root, output_dir, reason)
        output = result["output"]
        fallback = result["fallback"]
    size = output.stat().st_size if output.exists() else 0
    record: dict[str, Any] = {
        "id": entry.id,
        "day": entry.day,
        "kind": entry.kind,
        "caption": entry.caption,
        "struggle": entry.struggle,
        "output": str(_relative(root, output)),
        "size_bytes": size,
        "sources": entry.sources,
        "frames": entry.frames.to_record() if entry.frames else None,
        "fps": entry.fps if entry.kind in ANIMATED_KINDS else None,
        "duration_seconds": round(len(entry.frames.indices()) / entry.fps, 2)
        if entry.frames and fallback is None
        else None,
        "width": entry.width,
        "options": entry.options,
        "fallback": fallback,
        "detail": detail,
        "encode": encode or None,
    }
    return record


# --------------------------------------------------------------------------- manifest out


def write_output_manifest(
    path: Path, records: Sequence[dict[str, Any]], *, config: Path, root: Path, merge: bool
) -> dict[str, Any]:
    existing: list[dict[str, Any]] = []
    if merge and path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8")).get("entries", [])
        except (OSError, ValueError):
            existing = []
    by_key = {(e["day"], e["id"]): e for e in existing}
    for record in records:
        by_key[(record["day"], record["id"])] = record
    entries = [by_key[k] for k in sorted(by_key)]
    days = sorted({e["day"] for e in entries})
    coverage = {
        day: {
            "images": [e["output"] for e in entries if e["day"] == day],
            "fallbacks": [e["id"] for e in entries if e["day"] == day and e.get("fallback")],
            "placeholders": [
                e["id"]
                for e in entries
                if e["day"] == day and e.get("fallback") and e["fallback"].get("placeholder")
            ],
        }
        for day in days
    }
    doc = {
        "schema": f"{SCHEMA}/output",
        "config": str(_relative(root, config)),
        "renderer": "battle-story-media render",
        "entries": entries,
        "coverage": coverage,
    }
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return doc


def _report(doc: dict[str, Any], stream: Any) -> None:
    for entry in doc["entries"]:
        size = entry["size_bytes"] / 1e6
        line = f"{entry['output']}  {size:.2f} MB"
        if entry.get("duration_seconds"):
            line += f"  {entry['duration_seconds']} s @ {entry['fps']} fps"
        if entry.get("fallback"):
            line += f"  FALLBACK: {entry['fallback']['reason']}"
        print(line, file=stream)
    print("coverage:", file=stream)
    for day, info in doc["coverage"].items():
        note = ""
        if info["placeholders"]:
            note = f"  placeholders: {', '.join(info['placeholders'])}"
        elif info["fallbacks"]:
            note = f"  fallbacks: {', '.join(info['fallbacks'])}"
        print(f"  {day}: {len(info['images'])} image(s){note}", file=stream)


# --------------------------------------------------------------------------- cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="battle-story-media", description="Render the docs/story.md GIFs and stills."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    render = sub.add_parser("render", help="Render every entry (or --only one) and the manifest.")
    render.add_argument("--manifest", type=Path, default=Path("configs/story/media.json"))
    render.add_argument("--output", type=Path, default=Path("media/story"))
    render.add_argument("--only", action="append", default=[], help="Entry id (repeatable).")
    render.add_argument(
        "--repository-root",
        type=Path,
        default=Path.cwd(),
        help="Checkout the runs/ and data/ paths are relative to (default: cwd).",
    )
    check = sub.add_parser("check", help="Parse the manifest and list its entries.")
    check.add_argument("--manifest", type=Path, default=Path("configs/story/media.json"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    entries = load_manifest(args.manifest)
    if args.command == "check":
        for entry in entries:
            print(f"{entry.day}  {entry.kind:<12} {entry.output_name}  {entry.caption}")
        return 0
    root = args.repository_root.resolve()
    output_dir = args.output if args.output.is_absolute() else root / args.output
    selected = entries
    if args.only:
        wanted = set(args.only)
        selected = [e for e in entries if e.id in wanted]
        missing = wanted - {e.id for e in selected}
        if missing:
            print(f"unknown entry id(s): {', '.join(sorted(missing))}", file=sys.stderr)
            return 2
    records = []
    for entry in selected:
        print(f"rendering {entry.output_name} ...", file=sys.stderr, flush=True)
        records.append(render_entry(entry, root, output_dir))
    doc = write_output_manifest(
        output_dir / "manifest.json",
        records,
        config=args.manifest,
        root=root,
        merge=bool(args.only),
    )
    _report(doc, sys.stdout)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
