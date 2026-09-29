"""`battle-finebio-colour`: the pipette's colour ring as an accumulated track attribute
(plan `p2-colour-vote`, Sep 28).

The three single-channel pipettes (`blue_pipette`, `yellow_pipette`, `red_pipette`) are one
body with a colour-coded ring, so the detector's class is an appearance vote on a small
feature, wrong whenever that feature is hidden or a few pixels wide. This module reads the
ring's hue from the proxy frame and accumulates it per track, so identity becomes a
histogram with an entropy rather than a per-frame label.

**Where the ring is.** The plan assumed the ring sits at the tip. On the P03 and P20 proxies
the colour is the **plunger button** at the top of the pipette (a yellow disc, a magenta-red
ring, a blue button, seen by looking at the frames), while the tip end is a white cone; a
blue grip band runs round every pipette's body a third of the way down.
`ColourSettings.ring_end` records this as ``"plunger"`` and every function that answers
"which end carries the ring" (`sample_from_observation`, `sample_track_ends`) returns both the
ring index and the tip index derived from it, so the tracker's held coupling (hand at the
plunger end, ring next to the thumb) and tip events read the right end.

* `ring_hue_sample` samples a disc at one end of a mask axis or projected line, inset along
  the axis so it lands on the button and not the background, converts it to HSV, keeps the
  saturated and bright pixels, and votes them into three hue bins. The bins, the saturation
  and value gates, the disc geometry and the visibility gate live in `ColourSettings`; the
  defaults are the values calibrated on P03_03_01 (`battle-finebio-colour sample`) and the
  same numbers are written to ``configs/finebio/pipettes.json`` under ``colour`` with
  provenance, which `load_colour_settings` reads back.
* `ColourVote` accumulates samples per track: colour weights (patch size times ring
  fraction), `identity()` (colour, top share), `entropy()` in bits, `agreement(class)` as the
  running fraction of samples whose top colour is the detector class's colour, and the
  running vote on which line end carries the ring.
* `sample_track_end` / `sample_track_ends` project a 3D end and a point 2 cm inward through
  every camera of the frame, rank the views by expected ring size (the nearest camera with
  the end inside its frame; the fpv only when its pose is valid, which the caller expresses
  by including it in `cams`) and sample the first that sees a ring. `sample_from_observation`
  is the fallback for the point tracker's rows: both ends of ``mask_axis_px`` are tried and
  the saturated one wins.
* `annotate_tracks` reads tracks as dicts, samples every k-th frame of every pipette track,
  and writes additive fields (``colour_identity``, ``colour_confidence``, ``colour_entropy``,
  ``colour_samples``, ``colour_agreement``, ``colour_ring_end``, ``colour_tip_end``,
  ``colour_ring_end_agreement``); nothing is imported from `multiview_schemas`.

Claim boundary: agreement is measured against the detector's class, which is itself the
appearance vote this module replaces; a disagreement is a disagreement, not an error count.
Pixels come from the crf-18 proxies, whose chroma is subsampled. FineBio is non-commercial
research data; frames, masks and videos derived from it stay under `runs/` and `data/`.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .finebio_cameras import (
    Camera,
    cameras_from_config,
    fpv_camera_from_config,
    fpv_poses,
    read_camera_config,
)

SCHEMA = "battle-finebio-colour/1"
COLOURS: tuple[str, ...] = ("blue", "yellow", "red")
CLASS_COLOUR: dict[str, str] = {
    "blue_pipette": "blue",
    "yellow_pipette": "yellow",
    "red_pipette": "red",
}
SINGLE_CHANNEL_PIPETTES: tuple[str, ...] = tuple(CLASS_COLOUR)
# Track classes the annotate path votes on: the three detector classes and the extension's
# one geometric class.
ANNOTATE_CLASSES: tuple[str, ...] = (*SINGLE_CHANNEL_PIPETTES, "pipette")
HUE_BIN_COUNT = 36  # 5-degree bins over OpenCV's 0..179 hue for the calibration histograms
HUE_BIN_DEG = 180 / HUE_BIN_COUNT
WIDTH_BINS_PX: tuple[tuple[str, float, float], ...] = (
    ("<25", 0.0, 25.0),
    ("25-40", 25.0, 40.0),
    (">=40", 40.0, float("inf")),
)
CLAIM_BOUNDARY = (
    "Agreement is measured against the detector's class, itself an appearance vote on the "
    "same ring; a disagreement is not an error count. Hues are read from crf-18 proxies "
    "with subsampled chroma. The ring is the plunger button, not the tip: 'tip' indices are "
    "the other end of the axis under that finding."
)
LICENCE_NOTE = (
    "FineBio is licensed for non-commercial research; frames, masks and videos derived from it "
    "stay under runs/ or data/ and are never committed or redistributed"
)

HueBins = dict[str, tuple[tuple[float, float], ...]]

# Calibrated on P03_03_01 (runs/finebio-lines-P03_03_01-20260928/observations-b, 200 rows
# per class, seed 0): the saturated pixels at the button sit at hue 20-30 (yellow, peak
# 27.5), 105-120 (blue, peak 112.5) and 165-175 (the magenta-red button, peak 167.5); skin
# is 5-20 and lands outside. The bins are each class's mode region widened by 5 degrees
# (`derive_hue_bins`); `configs/finebio/pipettes.json` carries the same numbers with
# provenance.
DEFAULT_HUE_BINS: HueBins = {
    "blue": ((100.0, 125.0),),
    "yellow": ((15.0, 35.0),),
    "red": ((160.0, 180.0),),
}


@dataclass(frozen=True)
class ColourSettings:
    """Everything `ring_hue_sample` needs, serialisable as the ``colour`` block of
    ``configs/finebio/pipettes.json``. Hues are OpenCV's 0..179; intervals are ``[lo, hi)``.
    """

    hue_bins: HueBins = field(default_factory=lambda: dict(DEFAULT_HUE_BINS))
    # HSV gates: below either, a pixel is body, glare or bench and does not vote.
    min_saturation: int = 80
    min_value: int = 50
    # A disc narrower than this cannot show a ring (the plan's ~12 px).
    min_patch_px: float = 12.0
    # The ring is visible when at least this fraction of the disc is saturated *and* in a
    # hue bin: the button covers about a tenth of the disc (P03 median 0.10, p10 0.044) and
    # the other end sits at zero to its p90; the gate is half the p10.
    min_ring_fraction: float = 0.022
    min_ring_pixels: int = 6
    # The disc: centred `disc_inset_factor * width` inward from the end along the axis,
    # radius `disc_radius_factor * width`.
    disc_inset_factor: float = 0.5
    disc_radius_factor: float = 0.6
    # Which end of the pipette carries the ring: "plunger" (the finding) or "tip".
    ring_end: str = "plunger"
    # For projecting a 3D end: the body diameter that sets the expected ring size per view,
    # and the step inward that gives the 2D axis direction.
    pipette_diameter_cm: float = 2.5
    inward_step_cm: float = 2.0

    def colour_of_hue(self, hue: float) -> str | None:
        for colour, intervals in self.hue_bins.items():
            for lo, hi in intervals:
                if lo <= hue < hi:
                    return colour
        return None

    def hue_colour_index(self, hues: np.ndarray) -> np.ndarray:
        """Per pixel the index into `COLOURS`, -1 outside every bin."""
        out = np.full(hues.shape, -1, dtype=np.int64)
        for k, colour in enumerate(COLOURS):
            for lo, hi in self.hue_bins.get(colour, ()):
                out[(hues >= lo) & (hues < hi) & (out < 0)] = k
        return out

    def tip_index_for_ring(self, ring_index: int) -> int:
        return ring_index if self.ring_end == "tip" else 1 - ring_index

    def as_dict(self) -> dict[str, Any]:
        return {
            "hue_bins_opencv": {
                c: [[float(lo), float(hi)] for lo, hi in intervals if hi > lo]
                for c, intervals in self.hue_bins.items()
            },
            "min_saturation": self.min_saturation,
            "min_value": self.min_value,
            "min_patch_px": self.min_patch_px,
            "min_ring_fraction": self.min_ring_fraction,
            "min_ring_pixels": self.min_ring_pixels,
            "disc_inset_factor": self.disc_inset_factor,
            "disc_radius_factor": self.disc_radius_factor,
            "ring_end": self.ring_end,
            "pipette_diameter_cm": self.pipette_diameter_cm,
            "inward_step_cm": self.inward_step_cm,
        }

    @classmethod
    def from_dict(cls, block: Mapping[str, Any]) -> ColourSettings:
        bins = block.get("hue_bins_opencv")
        kwargs: dict[str, Any] = {}
        if bins:
            kwargs["hue_bins"] = {
                c: tuple((float(lo), float(hi)) for lo, hi in intervals)
                for c, intervals in bins.items()
            }
        for key in (
            "min_saturation",
            "min_value",
            "min_patch_px",
            "min_ring_fraction",
            "min_ring_pixels",
            "disc_inset_factor",
            "disc_radius_factor",
            "ring_end",
            "pipette_diameter_cm",
            "inward_step_cm",
        ):
            if key in block:
                kwargs[key] = block[key]
        return cls(**kwargs)


DEFAULT_SETTINGS = ColourSettings()


def load_colour_settings(pipettes_config: Path | None) -> ColourSettings:
    """The ``colour`` block of a pipettes config, or the module defaults when the file or the
    block is absent."""
    if pipettes_config is None or not Path(pipettes_config).is_file():
        return DEFAULT_SETTINGS
    doc = json.loads(Path(pipettes_config).read_text(encoding="utf-8"))
    block = doc.get("colour")
    return DEFAULT_SETTINGS if not block else ColourSettings.from_dict(block)


# --------------------------------------------------------------------------- the disc


@dataclass(frozen=True)
class DiscPatch:
    centre_px: np.ndarray
    radius_px: float
    hsv: np.ndarray  # (n, 3) uint8 pixels inside the disc and the frame

    @property
    def patch_px(self) -> float:
        return 2.0 * self.radius_px

    @property
    def pixels(self) -> int:
        return int(self.hsv.shape[0])


def disc_pixels(
    image_bgr: np.ndarray,
    end_px: Sequence[float] | np.ndarray,
    axis_dir_px: Sequence[float] | np.ndarray,
    width_px: float,
    settings: ColourSettings = DEFAULT_SETTINGS,
) -> DiscPatch | None:
    """HSV pixels of the sampling disc at `end_px`: centred `disc_inset_factor * width`
    inward along `axis_dir_px` (pointing from the end into the body), radius
    `disc_radius_factor * width`. None when the disc is under `min_patch_px` across, when
    the direction is degenerate, or when less than half of it lies inside the frame."""
    end = np.asarray(end_px, dtype=np.float64).reshape(2)
    direction = np.asarray(axis_dir_px, dtype=np.float64).reshape(2)
    norm = float(np.linalg.norm(direction))
    if norm <= 0 or not np.isfinite(norm) or width_px <= 0:
        return None
    direction = direction / norm
    radius = settings.disc_radius_factor * float(width_px)
    if 2.0 * radius < settings.min_patch_px:
        return None
    centre = end + settings.disc_inset_factor * float(width_px) * direction
    height, width = image_bgr.shape[:2]
    x0 = max(0, int(math.floor(centre[0] - radius)))
    y0 = max(0, int(math.floor(centre[1] - radius)))
    x1 = min(width, int(math.ceil(centre[0] + radius)) + 1)
    y1 = min(height, int(math.ceil(centre[1] + radius)) + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    inside = (xx - centre[0]) ** 2 + (yy - centre[1]) ** 2 <= radius * radius
    count = int(inside.sum())
    if count < 0.5 * math.pi * radius * radius or count == 0:
        return None
    bgr = image_bgr[y0:y1, x0:x1][inside].reshape(-1, 1, 3)
    hsv = cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    return DiscPatch(centre, radius, hsv)


def ring_hue_sample(
    image_bgr: np.ndarray,
    end_px: Sequence[float] | np.ndarray,
    axis_dir_px: Sequence[float] | np.ndarray,
    width_px: float,
    settings: ColourSettings = DEFAULT_SETTINGS,
) -> dict[str, Any] | None:
    """The colour vote of one end.

    Samples the disc (`disc_pixels`), keeps pixels with saturation and value at or above the
    gates, bins their hues into the three colours, and returns::

        {"colour", "confidence" (top share), "hist" {colour: share of in-bin pixels},
         "saturated_fraction" (all saturated pixels / disc), "ring_fraction" (in-bin
         saturated / disc), "other_fraction" (saturated but outside every bin / saturated),
         "patch_px" (disc diameter), "pixels", "ring_pixels", "hue_median" (of the top
         colour's pixels), "weight" (patch_px * ring_fraction), "centre_px"}

    None when the disc is too small or mostly outside the frame, or when the in-bin
    saturated pixels are under `min_ring_fraction` of the disc or under `min_ring_pixels`
    (the ring is hidden, or a hand or the bench is there instead).
    """
    patch = disc_pixels(image_bgr, end_px, axis_dir_px, width_px, settings)
    if patch is None:
        return None
    hues = patch.hsv[:, 0].astype(np.float64)
    saturated = (patch.hsv[:, 1] >= settings.min_saturation) & (
        patch.hsv[:, 2] >= settings.min_value
    )
    colour_index = settings.hue_colour_index(hues)
    in_bin = saturated & (colour_index >= 0)
    pixels = patch.pixels
    ring_pixels = int(in_bin.sum())
    ring_fraction = ring_pixels / pixels
    if ring_pixels < settings.min_ring_pixels or ring_fraction < settings.min_ring_fraction:
        return None
    counts = np.bincount(colour_index[in_bin], minlength=len(COLOURS)).astype(np.float64)
    shares = counts / counts.sum()
    top = int(np.argmax(shares))
    saturated_count = int(saturated.sum())
    return {
        "colour": COLOURS[top],
        "confidence": float(shares[top]),
        "hist": {c: float(shares[k]) for k, c in enumerate(COLOURS)},
        "saturated_fraction": saturated_count / pixels,
        "ring_fraction": ring_fraction,
        "other_fraction": float((saturated & ~in_bin).sum() / max(saturated_count, 1)),
        "patch_px": patch.patch_px,
        "pixels": pixels,
        "ring_pixels": ring_pixels,
        "hue_median": float(np.median(hues[in_bin & (colour_index == top)])),
        "weight": patch.patch_px * ring_fraction,
        "centre_px": [float(patch.centre_px[0]), float(patch.centre_px[1])],
    }


# --------------------------------------------------------------------------- the vote


@dataclass
class ColourVote:
    """Per-track accumulator of `ring_hue_sample` results.

    Colour weights are ``weight * hist[colour]`` summed over samples (weight = disc diameter
    times ring fraction, so a big, clearly visible button counts more than a far, marginal
    one). `identity()` is the top colour and its share of the weight; `entropy()` the Shannon
    entropy of the shares in bits (0 certain, log2(3) = 1.585 uniform); `agreement(class)`
    the fraction of samples whose own top colour is the class's colour; `ring_end()` the
    majority vote over the line ends that carried the ring (when the caller passes
    ``end_index``) and its share.
    """

    weights: dict[str, float] = field(default_factory=lambda: {c: 0.0 for c in COLOURS})
    samples: int = 0
    top_counts: Counter[str] = field(default_factory=Counter)
    end_votes: Counter[int] = field(default_factory=Counter)
    ambiguous: int = 0
    views: Counter[str] = field(default_factory=Counter)

    def add(self, sample: Mapping[str, Any], *, end_index: int | None = None) -> None:
        weight = float(sample["weight"])
        for colour, share in sample["hist"].items():
            self.weights[colour] = self.weights.get(colour, 0.0) + weight * float(share)
        self.samples += 1
        self.top_counts[str(sample["colour"])] += 1
        if end_index is not None:
            self.end_votes[int(end_index)] += 1
        if sample.get("ambiguous"):
            self.ambiguous += 1
        if sample.get("view"):
            self.views[str(sample["view"])] += 1

    @property
    def total_weight(self) -> float:
        return float(sum(self.weights.values()))

    def shares(self) -> dict[str, float]:
        total = self.total_weight
        if total <= 0:
            return {c: 0.0 for c in self.weights}
        return {c: w / total for c, w in self.weights.items()}

    def identity(self) -> tuple[str | None, float]:
        if self.total_weight <= 0:
            return None, 0.0
        colour, share = max(self.shares().items(), key=lambda kv: kv[1])
        return colour, float(share)

    def entropy(self) -> float:
        return max(0.0, float(-sum(p * math.log2(p) for p in self.shares().values() if p > 0)))

    def agreement(self, detector_class: str | None) -> float | None:
        colour = CLASS_COLOUR.get(detector_class or "", detector_class)
        if colour not in COLOURS or self.samples == 0:
            return None
        return self.top_counts.get(colour, 0) / self.samples

    def ring_end(self) -> tuple[int | None, float]:
        votes = sum(self.end_votes.values())
        if votes == 0:
            return None, 0.0
        end, count = self.end_votes.most_common(1)[0]
        return int(end), count / votes

    def as_dict(self) -> dict[str, Any]:
        colour, confidence = self.identity()
        end, end_share = self.ring_end()
        return {
            "identity": colour,
            "confidence": round(confidence, 4),
            "entropy_bits": round(self.entropy(), 4),
            "samples": self.samples,
            "weights": {c: round(w, 3) for c, w in self.weights.items()},
            "top_counts": dict(self.top_counts),
            "ring_end": end,
            "ring_end_agreement": round(end_share, 4),
            "ambiguous_samples": self.ambiguous,
            "views": dict(self.views),
        }


# --------------------------------------------------------------------------- observations


def _axis_ends(row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    axis = row.get("mask_axis_px")
    if not axis:
        return None
    p0 = np.asarray(axis[0], dtype=np.float64).reshape(2)
    p1 = np.asarray(axis[1], dtype=np.float64).reshape(2)
    d = p1 - p0
    norm = float(np.linalg.norm(d))
    if norm <= 0:
        return None
    return p0, p1, d / norm


def _pick_end(
    samples: Sequence[dict[str, Any] | None], settings: ColourSettings
) -> dict[str, Any] | None:
    """The end whose sample carries the ring: the only one with a sample, else the heavier of
    the two (flagged ambiguous)."""
    found = [(i, s) for i, s in enumerate(samples) if s is not None]
    if not found:
        return None
    index, sample = max(found, key=lambda item: item[1]["weight"])
    out = dict(sample)
    out["ring_index"] = index
    out["tip_index"] = settings.tip_index_for_ring(index)
    out["ambiguous"] = len(found) > 1
    out["end_samples"] = list(samples)
    return out


def sample_from_observation(
    image_bgr: np.ndarray,
    row: Mapping[str, Any],
    settings: ColourSettings = DEFAULT_SETTINGS,
) -> dict[str, Any] | None:
    """The colour vote of one observation row (``mask_axis_px``, ``mask_width_px``) in its
    own frame: both axis ends are sampled and the one with the ring wins. Adds
    ``ring_index`` (0 or 1 into ``mask_axis_px``), ``tip_index`` (the other end under the
    plunger finding), ``ambiguous`` (both ends showed a ring) and ``end_samples``. None when
    the row has no axis or neither end shows a ring."""
    ends = _axis_ends(row)
    width = row.get("mask_width_px")
    if ends is None or not width:
        return None
    p0, p1, direction = ends
    samples = [
        ring_hue_sample(image_bgr, p0, direction, float(width), settings),
        ring_hue_sample(image_bgr, p1, -direction, float(width), settings),
    ]
    return _pick_end(samples, settings)


# --------------------------------------------------------------------------- 3D ends


class FrameLookup:
    """`frames.get(view)` over a ``provider(view, frame_index)`` callable, reading lazily and
    at most once per view."""

    def __init__(self, provider: Callable[[str, int], np.ndarray | None], frame_index: int):
        self._provider = provider
        self._frame = int(frame_index)
        self._cache: dict[str, np.ndarray | None] = {}

    def get(self, view: str) -> np.ndarray | None:
        if view not in self._cache:
            self._cache[view] = self._provider(view, self._frame)
        return self._cache[view]


def _depth_cm(cam: Camera, point: np.ndarray) -> float:
    return float((cam.R @ point + cam.tvec.reshape(3))[2])


def rank_views(
    cams: Mapping[str, Camera],
    end_cm: np.ndarray,
    width_px_by_view: Mapping[str, float] | None,
    settings: ColourSettings,
) -> list[tuple[str, np.ndarray, float]]:
    """Views that see `end_cm` in front of the camera and inside the frame (with the disc's
    margin), as ``(view, end_px, expected_width_px)`` in decreasing expected ring size. The
    expected width is the caller's per-view mask width when given, else the body diameter
    projected at the end's depth."""
    ranked: list[tuple[str, np.ndarray, float]] = []
    for view, cam in cams.items():
        depth = _depth_cm(cam, end_cm)
        if not np.isfinite(depth) or depth <= 1.0:
            continue
        px = cam.project(end_cm.reshape(1, 3))[0]
        if not np.all(np.isfinite(px)):
            continue
        width = None
        if width_px_by_view is not None:
            width = width_px_by_view.get(view)
        if width is None:
            width = float(cam.K[0, 0]) * settings.pipette_diameter_cm / depth
        margin = settings.disc_radius_factor * float(width)
        w, h = cam.size
        if not (margin <= px[0] <= w - margin and margin <= px[1] <= h - margin):
            continue
        ranked.append((view, px, float(width)))
    ranked.sort(key=lambda item: -item[2])
    return ranked


def sample_track_end(
    cams: Mapping[str, Camera],
    frames: Any,
    end_cm: Sequence[float] | np.ndarray,
    direction_cm: Sequence[float] | np.ndarray,
    width_px_by_view: Mapping[str, float] | None = None,
    *,
    settings: ColourSettings = DEFAULT_SETTINGS,
) -> dict[str, Any] | None:
    """Sample the ring at a 3D end of a line track.

    `direction_cm` points from the end into the body. Views are ranked by expected ring size
    (`rank_views`); for each in turn the end and the point `inward_step_cm` along the
    direction are projected to give the 2D axis, and `ring_hue_sample` runs on
    ``frames.get(view)`` (a dict of images or a `FrameLookup`). The first view that shows a
    ring wins; the result gains ``view``, ``end_px``, ``expected_width_px`` and
    ``views_tried``. The fpv counts only when the caller put its per-frame camera in `cams`.
    """
    end = np.asarray(end_cm, dtype=np.float64).reshape(3)
    direction = np.asarray(direction_cm, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(direction))
    if norm <= 0:
        return None
    direction = direction / norm
    tried: list[str] = []
    for view, end_px, width in rank_views(cams, end, width_px_by_view, settings):
        image = frames.get(view)
        if image is None:
            continue
        tried.append(view)
        inward = cams[view].project((end + settings.inward_step_cm * direction).reshape(1, 3))[0]
        axis = inward - end_px
        if not np.all(np.isfinite(axis)) or float(np.linalg.norm(axis)) <= 0:
            continue
        sample = ring_hue_sample(image, end_px, axis, width, settings)
        if sample is not None:
            sample["view"] = view
            sample["end_px"] = [float(end_px[0]), float(end_px[1])]
            sample["expected_width_px"] = width
            sample["views_tried"] = list(tried)
            return sample
    return None


# The plan's name for the same call: it samples whichever end the caller passes.
sample_track_tip = sample_track_end


def sample_track_ends(
    cams: Mapping[str, Camera],
    frames: Any,
    endpoints_cm: Sequence[Sequence[float]] | np.ndarray,
    width_px_by_view: Mapping[str, float] | None = None,
    *,
    settings: ColourSettings = DEFAULT_SETTINGS,
) -> dict[str, Any] | None:
    """Both ends of a line track through `sample_track_end`; the one with the ring wins and
    the result carries ``ring_end`` / ``ring_index`` (index into `endpoints_cm`),
    ``tip_index`` and ``ambiguous``."""
    ends = np.asarray(endpoints_cm, dtype=np.float64).reshape(2, 3)
    samples = [
        sample_track_end(
            cams, frames, ends[0], ends[1] - ends[0], width_px_by_view, settings=settings
        ),
        sample_track_end(
            cams, frames, ends[1], ends[0] - ends[1], width_px_by_view, settings=settings
        ),
    ]
    picked = _pick_end(samples, settings)
    if picked is not None:
        picked["ring_end"] = picked["ring_index"]
    return picked


# --------------------------------------------------------------------------- frame sources


class ProxyFrameSource:
    """``(view, raw_frame) -> BGR image`` over the window proxies (proxy frame = raw frame -
    `frame_index_offset`). One capture per view; a request within `seek_gap` frames ahead of
    the last decoded frame reads forward sequentially, anything else seeks."""

    def __init__(
        self, proxies: Mapping[str, Path], frame_index_offset: int, *, seek_gap: int = 90
    ) -> None:
        self.proxies = {v: Path(p) for v, p in proxies.items()}
        self.offset = int(frame_index_offset)
        self.seek_gap = int(seek_gap)
        self._caps: dict[str, cv2.VideoCapture] = {}
        self._next: dict[str, int] = {}
        self.reads = 0
        self.seeks = 0

    def _capture(self, view: str) -> cv2.VideoCapture | None:
        if view not in self._caps:
            path = self.proxies.get(view)
            if path is None or not path.is_file():
                return None
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                return None
            self._caps[view] = cap
            self._next[view] = 0
        return self._caps[view]

    def __call__(self, view: str, frame_index: int) -> np.ndarray | None:
        return self.frame(view, frame_index)

    def frame(self, view: str, frame_index: int) -> np.ndarray | None:
        cap = self._capture(view)
        if cap is None:
            return None
        proxy = int(frame_index) - self.offset
        if proxy < 0:
            return None
        position = self._next[view]
        if proxy < position or proxy > position + self.seek_gap:
            cap.set(cv2.CAP_PROP_POS_FRAMES, proxy)
            position = proxy
            self.seeks += 1
        image = None
        while position <= proxy:
            ok, image = cap.read()
            self.reads += 1
            if not ok:
                self._next[view] = position
                return None
            position += 1
        self._next[view] = position
        return image

    def close(self) -> None:
        for cap in self._caps.values():
            cap.release()
        self._caps.clear()


@dataclass
class RigCameras:
    """The fixed cameras plus the fpv at a raw frame when its shipped pose is valid."""

    fixed: dict[str, Camera]
    config: Any = None
    poses: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def at(self, frame_index: int) -> dict[str, Camera]:
        cams = dict(self.fixed)
        if self.config is not None and self.poses is not None:
            fpv = fpv_camera_from_config(self.config, int(frame_index), self.poses)
            if fpv is not None:
                cams["fpv"] = fpv
        return cams


def load_rig(camera_config: Path, *, with_fpv: bool = True) -> RigCameras:
    config = read_camera_config(Path(camera_config))
    poses = None
    if with_fpv:
        try:
            poses = fpv_poses(config.trial)
        except FileNotFoundError:
            poses = None
    return RigCameras(cameras_from_config(config), config, poses)


def load_clip(clip_config: Path, root: Path) -> dict[str, Any]:
    clip = json.loads(Path(clip_config).read_text(encoding="utf-8"))
    clip["_proxies"] = {v: root / uri for v, uri in clip["proxies"].items()}
    clip["_offset"] = int(clip.get("frame_index_offset", clip["window"]["start_frame"]))
    clip["_camera_config"] = root / clip["window_camera_config"]
    return clip


def observations_path(path: Path) -> Path:
    path = Path(path)
    return path / "observations.jsonl" if path.is_dir() else path


def load_axis_rows(
    path: Path, classes: Iterable[str] = SINGLE_CHANNEL_PIPETTES
) -> list[dict[str, Any]]:
    """The SAM3 rows of `classes` that carry a mask axis, as dicts (a substring filter before
    the JSON parse keeps this to seconds on 800k-row files)."""
    wanted = set(classes)
    rows: list[dict[str, Any]] = []
    with observations_path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if '"mask_axis_px"' not in line:
                continue
            row = json.loads(line)
            if row.get("object_class") in wanted and row.get("mask_axis_px"):
                rows.append(row)
    return rows


def index_by_slot(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, int, str], dict[str, Any]]:
    """``(view, frame_index, slot) -> row`` (the highest SAM3 score when duplicated)."""
    out: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["view"]), int(row["frame_index"]), str(row["slot"]))
        current = out.get(key)
        if current is None or (row.get("sam3_object_score") or 0.0) > (
            current.get("sam3_object_score") or 0.0
        ):
            out[key] = dict(row)
    return out


def sample_from_support(
    frames: Any,
    row: Mapping[str, Any],
    observations: Mapping[tuple[str, int, str], Mapping[str, Any]],
    settings: ColourSettings = DEFAULT_SETTINGS,
) -> dict[str, Any] | None:
    """Point-tracker fallback: the track row's ``support_slots`` (view -> slot) name the
    observation rows of this frame; the widest mask with an axis is sampled first."""
    frame = int(row["frame_index"])
    candidates = []
    for view, slot in (row.get("support_slots") or {}).items():
        obs = observations.get((str(view), frame, str(slot)))
        if obs is not None and obs.get("mask_axis_px"):
            candidates.append(obs)
    candidates.sort(key=lambda o: -(o.get("mask_width_px") or 0.0))
    tried: list[str] = []
    for obs in candidates:
        image = frames.get(str(obs["view"]))
        if image is None:
            continue
        tried.append(str(obs["view"]))
        sample = sample_from_observation(image, obs, settings)
        if sample is not None:
            sample["view"] = str(obs["view"])
            sample["views_tried"] = list(tried)
            return sample
    return None


# --------------------------------------------------------------------------- annotate


COLOUR_FIELDS = (
    "colour_identity",
    "colour_confidence",
    "colour_entropy",
    "colour_samples",
    "colour_agreement",
    "colour_ring_end",
    "colour_tip_end",
    "colour_ring_end_agreement",
)


def annotate_tracks(
    rows: Sequence[dict[str, Any]],
    *,
    cameras_for_frame: Callable[[int], Mapping[str, Camera]] | None,
    frame_provider: Callable[[str, int], np.ndarray | None],
    observations: Mapping[tuple[str, int, str], Mapping[str, Any]] | None = None,
    every: int = 3,
    classes: Iterable[str] = ANNOTATE_CLASSES,
    settings: ColourSettings = DEFAULT_SETTINGS,
    width_px_by_view: Callable[[Mapping[str, Any]], Mapping[str, float] | None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Colour-annotate track rows (dicts) in place and return them with a per-track summary.

    Every track of `classes` is sampled on every `every`-th frame of its life: a row with
    ``endpoints_cm`` goes through `sample_track_ends` with the frame's cameras; a row without
    them falls back to `sample_from_support` over `observations`. All rows of the track then
    receive ``colour_identity``, ``colour_confidence`` (top share), ``colour_entropy``
    (bits), ``colour_samples``, ``colour_agreement`` (with the row's ``object_class``),
    ``colour_ring_end`` / ``colour_tip_end`` (index into ``endpoints_cm``, line tracks only)
    and ``colour_ring_end_agreement``. `width_px_by_view` may return the per-view mask
    widths of a row to size the disc; otherwise the projected body diameter is used.
    """
    every = max(1, int(every))
    wanted = set(classes)
    by_track: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("object_class") in wanted and row.get("track_id"):
            by_track[str(row["track_id"])].append(row)
    summary: dict[str, Any] = {}
    for track_id, trows in by_track.items():
        trows.sort(key=lambda r: int(r["frame_index"]))
        first = int(trows[0]["frame_index"])
        vote = ColourVote()
        sampled_frames = 0
        line_track = False
        for row in trows:
            frame = int(row["frame_index"])
            if (frame - first) % every:
                continue
            frames = FrameLookup(frame_provider, frame)
            sample = None
            if row.get("endpoints_cm") is not None and cameras_for_frame is not None:
                line_track = True
                widths = width_px_by_view(row) if width_px_by_view else None
                sample = sample_track_ends(
                    cameras_for_frame(frame), frames, row["endpoints_cm"], widths, settings=settings
                )
                if sample is not None:
                    vote.add(sample, end_index=sample["ring_end"])
            elif observations is not None and row.get("support_slots"):
                sample = sample_from_support(frames, row, observations, settings)
                if sample is not None:
                    vote.add(sample)
            sampled_frames += 1
        colour, confidence = vote.identity()
        end, end_share = vote.ring_end()
        for row in trows:
            row["colour_identity"] = colour
            row["colour_confidence"] = round(confidence, 4)
            row["colour_entropy"] = round(vote.entropy(), 4)
            row["colour_samples"] = vote.samples
            agreement = vote.agreement(row.get("object_class"))
            row["colour_agreement"] = None if agreement is None else round(agreement, 4)
            row["colour_ring_end"] = end if line_track else None
            row["colour_tip_end"] = (
                settings.tip_index_for_ring(end) if line_track and end is not None else None
            )
            row["colour_ring_end_agreement"] = round(end_share, 4) if line_track else None
        summary[track_id] = {
            **vote.as_dict(),
            "object_class": trows[0].get("object_class"),
            "rows": len(trows),
            "frames_sampled": sampled_frames,
            "line_track": line_track,
            "agreement_with_class": vote.agreement(trows[0].get("object_class")),
        }
    return list(rows), summary


# --------------------------------------------------------------------------- calibration


def select_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    rows_per_class: int,
    seed: int = 0,
    classes: Sequence[str] = SINGLE_CHANNEL_PIPETTES,
) -> list[dict[str, Any]]:
    """About `rows_per_class` rows per class, spread evenly over the views that carry the
    class (a view short of its quota hands the rest to the others)."""
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for cls in classes:
        by_view: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            if r.get("object_class") == cls:
                by_view[str(r["view"])].append(dict(r))
        views = sorted(by_view)
        if not views:
            continue
        quota = math.ceil(rows_per_class / len(views))
        chosen: list[dict[str, Any]] = []
        leftovers: list[dict[str, Any]] = []
        for view in views:
            pool = by_view[view]
            rng.shuffle(pool)
            chosen.extend(pool[:quota])
            leftovers.extend(pool[quota:])
        if len(chosen) < rows_per_class:
            rng.shuffle(leftovers)
            chosen.extend(leftovers[: rows_per_class - len(chosen)])
        out.extend(chosen[:rows_per_class])
    return out


def _hue_hist(hues: np.ndarray) -> list[int]:
    if hues.size == 0:
        return [0] * HUE_BIN_COUNT
    return np.bincount((hues / HUE_BIN_DEG).astype(int) % HUE_BIN_COUNT, minlength=HUE_BIN_COUNT)[
        :HUE_BIN_COUNT
    ].tolist()


def _percentiles(values: np.ndarray, qs: Sequence[int] = (10, 50, 90)) -> dict[str, float | None]:
    if values.size == 0:
        return {f"p{q}": None for q in qs}
    return {f"p{q}": round(float(np.percentile(values, q)), 1) for q in qs}


GEOMETRY_SWEEP: tuple[tuple[float, float], ...] = ((0.5, 0.6), (0.25, 0.5), (0.25, 0.6), (0.5, 0.5))


def measure_row(
    image_bgr: np.ndarray,
    row: Mapping[str, Any],
    settings: ColourSettings,
    *,
    sweep: Sequence[tuple[float, float]] = GEOMETRY_SWEEP,
) -> dict[str, Any] | None:
    """Both ends of one row for the calibration: the raw hue histogram of the saturated
    pixels (no hue bins involved), saturation and value percentiles of the disc, the vote
    (`ring_hue_sample`) under `settings`, and the ring fraction under each disc geometry of
    `sweep` (inset, radius factors)."""
    ends = _axis_ends(row)
    width = row.get("mask_width_px")
    if ends is None or not width:
        return None
    p0, p1, direction = ends
    own = (settings.disc_inset_factor, settings.disc_radius_factor)
    geometries = tuple(dict.fromkeys([own, *sweep]))
    record: dict[str, Any] = {
        "view": row["view"],
        "frame_index": int(row["frame_index"]),
        "slot": row.get("slot"),
        "object_class": row["object_class"],
        "width_px": float(width),
        "axis_length_px": float(np.linalg.norm(p1 - p0)),
        "ends": [],
    }
    for end, inward in ((p0, direction), (p1, -direction)):
        patch = disc_pixels(image_bgr, end, inward, float(width), settings)
        entry: dict[str, Any] = {"patch_px": None, "pixels": 0}
        if patch is not None:
            hues = patch.hsv[:, 0].astype(np.float64)
            sat = patch.hsv[:, 1].astype(np.float64)
            val = patch.hsv[:, 2].astype(np.float64)
            saturated = (sat >= settings.min_saturation) & (val >= settings.min_value)
            entry.update(
                {
                    "patch_px": round(patch.patch_px, 1),
                    "pixels": patch.pixels,
                    "saturated_pixels": int(saturated.sum()),
                    "saturated_fraction": round(float(saturated.mean()), 4),
                    "hue_hist": _hue_hist(hues[saturated]),
                    "saturation": _percentiles(sat),
                    "value": _percentiles(val),
                    "saturated_saturation": _percentiles(sat[saturated]),
                    "saturated_value": _percentiles(val[saturated]),
                }
            )
        sample = ring_hue_sample(image_bgr, end, inward, float(width), settings)
        entry["sample"] = (
            None
            if sample is None
            else {
                k: sample[k]
                for k in (
                    "colour",
                    "confidence",
                    "hist",
                    "ring_fraction",
                    "other_fraction",
                    "weight",
                )
            }
        )
        entry["sweep"] = {}
        for inset, radius in geometries:
            variant = replace(settings, disc_inset_factor=inset, disc_radius_factor=radius)
            s = ring_hue_sample(image_bgr, end, inward, float(width), variant)
            entry["sweep"][f"{inset}/{radius}"] = (
                None if s is None else round(s["ring_fraction"], 4)
            )
        record["ends"].append(entry)
    found = [i for i, e in enumerate(record["ends"]) if e["sample"] is not None]
    record["ring_ends_found"] = len(found)
    picked = _pick_end([e["sample"] for e in record["ends"]], settings)
    record["ring_index"] = None if picked is None else picked["ring_index"]
    record["colour"] = None if picked is None else picked["colour"]
    record["agrees"] = (
        None if picked is None else picked["colour"] == CLASS_COLOUR.get(str(row["object_class"]))
    )
    saturated_counts = [e.get("saturated_pixels", 0) for e in record["ends"]]
    record["hue_source_end"] = (
        int(np.argmax(saturated_counts))
        if max(saturated_counts) >= settings.min_ring_pixels
        else None
    )
    return record


def measure_rows(
    rows: Sequence[Mapping[str, Any]],
    frame_provider: Callable[[str, int], np.ndarray | None],
    settings: ColourSettings,
    *,
    sweep: Sequence[tuple[float, float]] = GEOMETRY_SWEEP,
) -> list[dict[str, Any]]:
    """`measure_row` over rows ordered by view and frame (sequential reads where possible)."""
    ordered = sorted(rows, key=lambda r: (str(r["view"]), int(r["frame_index"])))
    out: list[dict[str, Any]] = []
    for row in ordered:
        image = frame_provider(str(row["view"]), int(row["frame_index"]))
        if image is None:
            continue
        record = measure_row(image, row, settings, sweep=sweep)
        if record is not None:
            out.append(record)
    return out


def class_hue_histograms(
    measurements: Sequence[Mapping[str, Any]],
) -> dict[str, list[float]]:
    """Per detector class the mean over rows of each row's normalised saturated-pixel hue
    histogram at its `hue_source_end` (the end with more saturated pixels), so one large
    patch cannot dominate."""
    sums: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(HUE_BIN_COUNT))
    counts: Counter[str] = Counter()
    for m in measurements:
        end = m.get("hue_source_end")
        if end is None:
            continue
        hist = np.asarray(m["ends"][end].get("hue_hist") or [0] * HUE_BIN_COUNT, dtype=np.float64)
        if hist.sum() <= 0:
            continue
        cls = str(m["object_class"])
        sums[cls] += hist / hist.sum()
        counts[cls] += 1
    return {cls: (sums[cls] / counts[cls]).round(5).tolist() for cls in sums if counts[cls]}


def _wrap_intervals(lo: float, hi: float) -> tuple[tuple[float, float], ...]:
    """An unwrapped hue interval (lo may be negative, hi may exceed 180) as [lo, hi) pieces
    inside [0, 180)."""
    if hi - lo >= 180:
        return ((0.0, 180.0),)
    lo_m, hi_m = lo % 180, hi % 180
    if lo_m < hi_m:
        return ((lo_m, hi_m),)
    pieces = []
    if lo_m < 180:
        pieces.append((lo_m, 180.0))
    if hi_m > 0:
        pieces.append((0.0, hi_m))
    return tuple(pieces)


def derive_hue_bins(
    class_hists: Mapping[str, Sequence[float]],
    *,
    margin_deg: float = 5.0,
    floor: float = 0.1,
    max_half_width_bins: int = 6,
) -> tuple[HueBins, dict[str, Any]]:
    """Hue bins from the per-class histograms: each colour's bin is the contiguous region
    round its class's peak where the density stays at or above `floor` times the peak
    (circular, at most `max_half_width_bins` each way), widened by `margin_deg`; overlapping
    bins are cut at the midpoint of their overlap. Returns the bins and the per-colour mode
    record (peak hue, region, widened interval, share of the histogram inside the bin)."""
    unwrapped: dict[str, tuple[float, float]] = {}
    modes: dict[str, Any] = {}
    for cls, colour in CLASS_COLOUR.items():
        hist = np.asarray(class_hists.get(cls, []), dtype=np.float64)
        if hist.size != HUE_BIN_COUNT or hist.sum() <= 0:
            continue
        peak = int(np.argmax(hist))
        lo_bin, hi_bin = peak, peak
        for step in range(1, max_half_width_bins + 1):
            if (
                hist[(peak - step) % HUE_BIN_COUNT] >= floor * hist[peak]
                and lo_bin == peak - step + 1
            ):
                lo_bin = peak - step
        for step in range(1, max_half_width_bins + 1):
            if (
                hist[(peak + step) % HUE_BIN_COUNT] >= floor * hist[peak]
                and hi_bin == peak + step - 1
            ):
                hi_bin = peak + step
        lo = lo_bin * HUE_BIN_DEG - margin_deg
        hi = (hi_bin + 1) * HUE_BIN_DEG + margin_deg
        unwrapped[colour] = (lo, hi)
        inside = sum(hist[b % HUE_BIN_COUNT] for b in range(lo_bin, hi_bin + 1)) / hist.sum()
        modes[colour] = {
            "class": cls,
            "peak_hue": round((peak + 0.5) * HUE_BIN_DEG, 1),
            "region_deg": [round(lo_bin * HUE_BIN_DEG, 1), round((hi_bin + 1) * HUE_BIN_DEG, 1)],
            "share_in_region": round(float(inside), 4),
        }
    colours = list(unwrapped)
    for i, a in enumerate(colours):
        for b in colours[i + 1 :]:
            a_lo, a_hi = unwrapped[a]
            for shift in (-180.0, 0.0, 180.0):
                b_lo, b_hi = unwrapped[b][0] + shift, unwrapped[b][1] + shift
                if a_lo < b_hi and b_lo < a_hi:
                    mid = (max(a_lo, b_lo) + min(a_hi, b_hi)) / 2
                    if a_lo <= b_lo:
                        unwrapped[a] = (a_lo, mid)
                        unwrapped[b] = (mid - shift, unwrapped[b][1])
                    else:
                        unwrapped[b] = (unwrapped[b][0], mid - shift)
                        unwrapped[a] = (mid, a_hi)
                    a_lo, a_hi = unwrapped[a]
    bins: HueBins = {}
    for colour in COLOURS:
        if colour in unwrapped:
            lo, hi = unwrapped[colour]
            bins[colour] = _wrap_intervals(round(lo, 1), round(hi, 1))
            modes[colour]["bin_deg"] = [list(p) for p in bins[colour]]
    return bins, modes


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else round(numerator / denominator, 4)


def _width_bin(width: float) -> str:
    for name, lo, hi in WIDTH_BINS_PX:
        if lo <= width < hi:
            return name
    return WIDTH_BINS_PX[-1][0]


def _group_rates(measurements: Sequence[Mapping[str, Any]], key: Callable[[Mapping], str]):
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for m in measurements:
        groups[key(m)].append(m)
    out: dict[str, Any] = {}
    for name, ms in sorted(groups.items()):
        n = len(ms)
        one = sum(1 for m in ms if m["ring_ends_found"] == 1)
        both = sum(1 for m in ms if m["ring_ends_found"] == 2)
        none = sum(1 for m in ms if m["ring_ends_found"] == 0)
        found = [m for m in ms if m["colour"] is not None]
        agree = sum(1 for m in found if m["agrees"])
        out[name] = {
            "rows": n,
            "ring_found_one_end": _rate(one, n),
            "ring_found_both_ends": _rate(both, n),
            "ring_found_none": _rate(none, n),
            "agreement_with_detector": _rate(agree, len(found)),
            "colours": dict(Counter(str(m["colour"]) for m in found)),
        }
    return out


def _fraction_thresholds(
    measurements: Sequence[Mapping[str, Any]], settings: ColourSettings
) -> dict[str, Any]:
    """Distribution of the in-bin (ring) fraction at the fuller end and at the emptier end of
    each row under the default geometry, and the visibility gate they suggest: half the
    fuller end's p10 (the emptier end sits at zero to its p90, so the gate only trades
    recall on far rings against both-ends ambiguity), floored at 0.02."""
    hi_vals, lo_vals = [], []
    key = f"{settings.disc_inset_factor}/{settings.disc_radius_factor}"
    for m in measurements:
        fr = [e["sweep"].get(key) or 0.0 for e in m["ends"]]
        if len(fr) != 2:
            continue
        hi_vals.append(max(fr))
        lo_vals.append(min(fr))
    hi = np.asarray(hi_vals)
    lo = np.asarray(lo_vals)
    if hi.size == 0:
        return {"suggested_min_ring_fraction": settings.min_ring_fraction}
    hi_med = float(np.median(hi))
    hi_p10 = float(np.percentile(hi, 10))
    lo_p90 = float(np.percentile(lo, 90))
    suggested = max(0.02, 0.5 * hi_p10)
    return {
        "fuller_end_ring_fraction": {
            "p10": round(hi_p10, 4),
            "p50": round(hi_med, 4),
            "p90": round(float(np.percentile(hi, 90)), 4),
        },
        "emptier_end_ring_fraction": {
            "p50": round(float(np.median(lo)), 4),
            "p90": round(lo_p90, 4),
            "p99": round(float(np.percentile(lo, 99)), 4),
        },
        "suggested_min_ring_fraction": round(suggested, 3),
        "rule": "half the fuller end's p10 ring fraction, floored at 0.02",
    }


def _geometry_sweep(measurements: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per disc geometry: the median ring fraction at the fuller end, the p90 at the emptier
    end, and how often exactly one end passes the default gate."""
    out: dict[str, Any] = {}
    keys = sorted({k for m in measurements for e in m["ends"] for k in e["sweep"]})
    for key in keys:
        hi, lo, one = [], [], 0
        for m in measurements:
            fr = [e["sweep"].get(key) for e in m["ends"]]
            if len(fr) != 2:
                continue
            present = [f for f in fr if f is not None]
            one += len(present) == 1
            vals = [f or 0.0 for f in fr]
            hi.append(max(vals))
            lo.append(min(vals))
        if hi:
            out[key] = {
                "fuller_end_median": round(float(np.median(hi)), 4),
                "emptier_end_p90": round(float(np.percentile(lo, 90)), 4),
                "exactly_one_end_rate": round(one / len(hi), 4),
            }
    return out


def _saturation_value_summary(measurements: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Saturation and value percentiles of the saturated pixels at the ring end and of every
    pixel at the other end, per class (the gates' justification)."""
    out: dict[str, Any] = {}
    by_class: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for m in measurements:
        by_class[str(m["object_class"])].append(m)
    for cls, ms in sorted(by_class.items()):
        ring_s, ring_v, other_s, other_v = [], [], [], []
        for m in ms:
            if m["ring_index"] is None:
                continue
            ring = m["ends"][m["ring_index"]]
            other = m["ends"][1 - m["ring_index"]]
            if ring.get("saturated_saturation", {}).get("p50") is not None:
                ring_s.append(ring["saturated_saturation"]["p50"])
                ring_v.append(ring["saturated_value"]["p50"])
            if other.get("saturation", {}).get("p90") is not None:
                other_s.append(other["saturation"]["p90"])
                other_v.append(other["value"]["p50"])
        out[cls] = {
            "ring_pixels_saturation_p50": _percentiles(np.asarray(ring_s)),
            "ring_pixels_value_p50": _percentiles(np.asarray(ring_v)),
            "other_end_saturation_p90": _percentiles(np.asarray(other_s)),
            "other_end_value_p50": _percentiles(np.asarray(other_v)),
        }
    return out


def calibration_report(
    measurements: Sequence[Mapping[str, Any]],
    settings: ColourSettings,
    *,
    trial: str,
    edges_source: str,
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """The `colour_calibration.json` document: per-class hue distributions, the bins the data
    would derive, the bins used, ring-found and agreement rates per class, view, view group
    and width bin, the saturation and value evidence, the geometry sweep and the proposed
    ``colour`` block for the pipettes config."""
    hists = class_hue_histograms(measurements)
    derived_bins, modes = derive_hue_bins(hists)
    thresholds = _fraction_thresholds(measurements, settings)
    per_class_hue = {}
    for cls, hist in hists.items():
        arr = np.asarray(hist)
        centres = (np.arange(HUE_BIN_COUNT) + 0.5) * HUE_BIN_DEG
        top = np.argsort(arr)[::-1][:4]
        per_class_hue[cls] = {
            "rows": sum(
                1
                for m in measurements
                if m["object_class"] == cls and m["hue_source_end"] is not None
            ),
            "top_bins": [
                {"hue": float(centres[i]), "share": round(float(arr[i]), 4)}
                for i in top
                if arr[i] > 0
            ],
            "hist_5deg": hist,
        }
    proposed_settings = replace(
        settings,
        hue_bins=derived_bins if derived_bins else settings.hue_bins,
        min_ring_fraction=float(
            thresholds.get("suggested_min_ring_fraction", settings.min_ring_fraction)
        ),
    )
    by_class = _group_rates(measurements, lambda m: str(m["object_class"]))

    def view_group(m: Mapping[str, Any]) -> str:
        return "fpv" if m["view"] == "fpv" else "fixed"

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "trial": trial,
        "inputs": dict(inputs),
        "rows_measured": len(measurements),
        "rows_per_class": dict(Counter(str(m["object_class"]) for m in measurements)),
        "settings_used": settings.as_dict(),
        "edges_source": edges_source,
        "hue_by_class": per_class_hue,
        "derived_hue_bins": {c: [list(p) for p in v] for c, v in derived_bins.items()},
        "derived_modes": modes,
        "ring_fraction_thresholds": thresholds,
        "by_class": by_class,
        "by_view": _group_rates(measurements, lambda m: str(m["view"])),
        "by_view_group": _group_rates(measurements, view_group),
        "by_class_and_view": _group_rates(
            measurements, lambda m: f"{m['object_class']}/{m['view']}"
        ),
        "by_width_px": _group_rates(measurements, lambda m: _width_bin(float(m["width_px"]))),
        "by_class_and_width_px": _group_rates(
            measurements, lambda m: f"{m['object_class']}/{_width_bin(float(m['width_px']))}"
        ),
        "saturation_value": _saturation_value_summary(measurements),
        "geometry_sweep": _geometry_sweep(measurements),
        "proposed_colour_block": {
            **proposed_settings.as_dict(),
            "provenance": {
                "trial": trial,
                "inputs": dict(inputs),
                "rows_measured": len(measurements),
                "method": (
                    "battle-finebio-colour sample: rows with a mask axis sampled per class and "
                    "view, both axis ends read from the proxy frame (disc inset "
                    f"{settings.disc_inset_factor} x width, radius {settings.disc_radius_factor} x "
                    "width, HSV gates); hue bins are each class's mode region of the saturated "
                    "pixels at the fuller end widened by 5 degrees (`derive_hue_bins`); the "
                    "visibility gate follows `ring_fraction_thresholds.rule`"
                ),
                "derived_modes": modes,
                "ring_found_by_class": {c: v["ring_found_one_end"] for c, v in by_class.items()},
                "agreement_by_class": {
                    c: v["agreement_with_detector"] for c, v in by_class.items()
                },
                "ring_end_finding": (
                    "the coloured ring is the plunger button at the top of the pipette, not the "
                    "tip; tip indices are the other axis end"
                ),
                "claim_boundary": CLAIM_BOUNDARY,
            },
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "licence_note": LICENCE_NOTE,
    }
    return report


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _sv_cell(block: Mapping[str, Any]) -> str:
    return f"{_fmt(block['p50'], 0)} ({_fmt(block['p10'], 0)}-{_fmt(block['p90'], 0)})"


def _rates_table(title: str, rates: Mapping[str, Any]) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| group | rows | ring at one end | both ends | none | agreement with detector | colours |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in rates.items():
        lines.append(
            f"| {name} | {r['rows']} | {_fmt(r['ring_found_one_end'])} | "
            f"{_fmt(r['ring_found_both_ends'])} | {_fmt(r['ring_found_none'])} | "
            f"{_fmt(r['agreement_with_detector'])} | {r['colours']} |"
        )
    lines.append("")
    return lines


def report_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        f"# Pipette colour ring, {report['trial']}",
        "",
        f"{report['rows_measured']} SAM3 rows with a mask axis ({report['rows_per_class']}), both "
        f"axis ends sampled from the proxies; hue bins from **{report['edges_source']}**. The "
        "ring is the plunger button at the top of the pipette; the other end is the tip.",
        "",
        "## Hue at the fuller end, per detector class (OpenCV hue, 5-degree bins)",
        "",
        "| class | rows | top bins (hue: share) |",
        "|---|---|---|",
    ]
    for cls, h in report["hue_by_class"].items():
        tops = ", ".join(f"{b['hue']:.0f}: {b['share']:.3f}" for b in h["top_bins"])
        lines.append(f"| {cls} | {h['rows']} | {tops} |")
    lines += [
        "",
        "## Hue bins",
        "",
        "| colour | derived here | used in this report |",
        "|---|---|---|",
    ]
    used = report["settings_used"]["hue_bins_opencv"]
    for colour in COLOURS:
        derived = report["derived_hue_bins"].get(colour, "-")
        lines.append(f"| {colour} | {derived} | {used.get(colour, '-')} |")
    th = report["ring_fraction_thresholds"]
    lines += [
        "",
        "## Visibility gate (in-bin saturated fraction of the disc)",
        "",
        f"Fuller end: {th.get('fuller_end_ring_fraction')}; emptier end: "
        f"{th.get('emptier_end_ring_fraction')}; suggested min_ring_fraction "
        f"**{th.get('suggested_min_ring_fraction')}** ({th.get('rule', '')}); used "
        f"{report['settings_used']['min_ring_fraction']}.",
        "",
    ]
    lines += _rates_table("Per class", report["by_class"])
    lines += _rates_table("Per view", report["by_view"])
    lines += _rates_table("fpv vs fixed", report["by_view_group"])
    lines += _rates_table("Per class and view", report["by_class_and_view"])
    lines += _rates_table("Per mask width (px, a distance proxy)", report["by_width_px"])
    lines += _rates_table("Per class and mask width", report["by_class_and_width_px"])
    lines += [
        "## Saturation and value (0-255) behind the gates",
        "",
        "| class | ring pixels S p50 (p10-p90 over rows) | ring pixels V p50 | other end S p90 | "
        "other end V p50 |",
        "|---|---|---|---|---|",
    ]
    for cls, sv in report["saturation_value"].items():
        lines.append(
            f"| {cls} | {_sv_cell(sv['ring_pixels_saturation_p50'])} | "
            f"{_sv_cell(sv['ring_pixels_value_p50'])} | "
            f"{_sv_cell(sv['other_end_saturation_p90'])} | "
            f"{_sv_cell(sv['other_end_value_p50'])} |"
        )
    lines += [
        "",
        "## Disc geometry sweep (inset/radius as factors of the mask width)",
        "",
        "| geometry | fuller end median ring fraction | emptier end p90 | exactly one end |",
        "|---|---|---|---|",
    ]
    for key, g in report["geometry_sweep"].items():
        lines.append(
            f"| {key} | {_fmt(g['fuller_end_median'], 4)} | {_fmt(g['emptier_end_p90'], 4)} | "
            f"{_fmt(g['exactly_one_end_rate'])} |"
        )
    lines += ["", report["claim_boundary"], "", report["licence_note"], ""]
    return "\n".join(lines)


def merge_colour_block(pipettes_path: Path, block: Mapping[str, Any]) -> dict[str, Any]:
    """Read the pipettes config, set its ``colour`` block, keep everything else, write it
    back (indent 1, trailing newline) and return the document."""
    path = Path(pipettes_path)
    doc = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    doc["colour"] = dict(block)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return doc


# --------------------------------------------------------------------------- cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    smp = sub.add_parser("sample", help="calibrate the ring colour on observation rows")
    smp.add_argument("--observations", type=Path, required=True, help="dir or observations.jsonl")
    smp.add_argument("--clip-config", type=Path, required=True)
    smp.add_argument("--rows-per-class", type=int, default=200)
    smp.add_argument("--seed", type=int, default=0)
    smp.add_argument(
        "--pipettes",
        type=Path,
        default=None,
        help="pipettes.json whose colour block supplies the hue bins (else derived from the data)",
    )
    smp.add_argument("--output", type=Path, required=True)
    smp.add_argument("--root", type=Path, default=Path.cwd())
    cfg = sub.add_parser("config", help="write the colour block into pipettes.json")
    cfg.add_argument("--calibration", type=Path, required=True, help="colour_calibration.json")
    cfg.add_argument("--pipettes", type=Path, required=True)
    ann = sub.add_parser("annotate", help="colour identity per track")
    ann.add_argument("--tracks", type=Path, required=True)
    ann.add_argument("--observations", type=Path, required=True, help="dir or observations.jsonl")
    ann.add_argument("--clip-config", type=Path, required=True)
    ann.add_argument("--output", type=Path, required=True, help="tracks_colour.jsonl")
    ann.add_argument("--pipettes", type=Path, default=Path("configs/finebio/pipettes.json"))
    ann.add_argument("--every", type=int, default=3)
    ann.add_argument(
        "--summary", type=Path, default=None, help="per-track JSON (default beside output)"
    )
    ann.add_argument("--root", type=Path, default=Path.cwd())
    return parser


def run_sample(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    clip = load_clip(args.clip_config, root)
    settings = DEFAULT_SETTINGS
    edges_source = "data"
    if args.pipettes is not None:
        loaded = load_colour_settings(args.pipettes)
        if loaded is not DEFAULT_SETTINGS:
            settings, edges_source = loaded, f"config {args.pipettes}"
    rows = load_axis_rows(args.observations)
    chosen = select_rows(rows, rows_per_class=args.rows_per_class, seed=args.seed)
    frames = ProxyFrameSource(clip["_proxies"], clip["_offset"])
    try:
        if edges_source == "data":
            # First pass with the defaults to read the hue distribution, then the vote with
            # the bins the data gives.
            first = measure_rows(chosen, frames, settings, sweep=())
            bins, _ = derive_hue_bins(class_hue_histograms(first))
            if bins:
                settings = replace(settings, hue_bins=bins)
            frames.close()
            frames = ProxyFrameSource(clip["_proxies"], clip["_offset"])
        measurements = measure_rows(chosen, frames, settings)
    finally:
        frames.close()
    report = calibration_report(
        measurements,
        settings,
        trial=str(clip["trial"]),
        edges_source=edges_source,
        inputs={
            "observations": str(args.observations),
            "clip_config": str(args.clip_config),
            "rows_per_class": args.rows_per_class,
            "seed": args.seed,
            "pipettes": None if args.pipettes is None else str(args.pipettes),
        },
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "colour_calibration.json").write_text(
        json.dumps(report, indent=1) + "\n", encoding="utf-8"
    )
    with (args.output / "measurements.jsonl").open("w", encoding="utf-8") as handle:
        for m in measurements:
            handle.write(json.dumps(m) + "\n")
    markdown = report_markdown(report)
    (args.output / "colour_calibration.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


def run_config(args: argparse.Namespace) -> int:
    report = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    doc = merge_colour_block(args.pipettes, report["proposed_colour_block"])
    print(json.dumps(doc["colour"], indent=1))
    return 0


def run_annotate(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    clip = load_clip(args.clip_config, root)
    settings = load_colour_settings(args.pipettes)
    rows = [
        json.loads(line)
        for line in Path(args.tracks).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    observations = index_by_slot(load_axis_rows(args.observations))
    rig = load_rig(clip["_camera_config"])
    frames = ProxyFrameSource(clip["_proxies"], clip["_offset"])
    try:
        annotated, summary = annotate_tracks(
            rows,
            cameras_for_frame=rig.at,
            frame_provider=frames,
            observations=observations,
            every=args.every,
            settings=settings,
        )
    finally:
        frames.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in annotated:
            handle.write(json.dumps(row) + "\n")
    summary_path = args.summary or args.output.with_name(args.output.stem + "_summary.json")
    summary_doc = {
        "schema": SCHEMA,
        "tracks": summary,
        "inputs": {
            "tracks": str(args.tracks),
            "observations": str(args.observations),
            "clip_config": str(args.clip_config),
            "pipettes": str(args.pipettes),
            "every": args.every,
        },
        "settings": settings.as_dict(),
        "frame_reads": frames.reads,
        "frame_seeks": frames.seeks,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    summary_path.write_text(json.dumps(summary_doc, indent=1) + "\n", encoding="utf-8")
    for track_id, s in summary.items():
        print(
            f"{track_id}: {s['identity']} {s['confidence']:.2f} entropy {s['entropy_bits']:.2f} "
            f"samples {s['samples']} agreement {s['agreement_with_class']} ring end {s['ring_end']}"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "sample":
        return run_sample(args)
    if args.command == "config":
        return run_config(args)
    if args.command == "annotate":
        return run_annotate(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
