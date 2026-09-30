"""Tip / butt as a weighted vote (Sep 29, part 1).

Five cues each name an endpoint and a confidence in [0, 1]. A per-camera cue contributes
once per frame, at the length-weighted best camera, so three copies of one wrong cue
cannot outvote one other cue. The frame's log-odds are the sum. Online, an episode
accumulator decays with a 60-frame half-life and flips only past a margin. A post-pass
writes back the sign at the peak of a decay-free sum over the episode.

The log-odds scale is `LOG_ODDS_SCALE` = 7. A cue contributes that times its confidence
toward its end. An attached tip box is confidence 0.9, so one box on a full-length shaft
contributes 6.3. That is above the resolve threshold (1.1) and, from an accumulator
parked at the opposite flip margin (-3), still clears +3 after one frame of decay, so
one tip box resolves on its own and can flip.

`BENCH_UP` is the bench normal from `finebio_stand` (z into the bench). Gravity is silent
at 45 degrees from horizontal and below: a pipette resting flat must not be decided by it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .finebio_stand import BENCH_UP

# One cue's signed log-odds is LOG_ODDS_SCALE * confidence (positive: endpoint 0 is the tip).
# 7 * 0.9 = 6.3, which clears both thresholds below. See the module docstring.
LOG_ODDS_SCALE = 7.0
RESOLVE_LOG_ODDS = 1.1
FLIP_MARGIN_LOG_ODDS = 3.0
HALF_LIFE_FRAMES = 60
DECAY_PER_FRAME = 0.5 ** (1.0 / HALF_LIFE_FRAMES)

# A 400 px shaft votes at full strength; a 30 px stub votes at 30/400.
LENGTH_FULL_PX = 400.0
# Colour size factor: mask width over this, clamped with the hue share into [0, 1].
COLOUR_SIZE_FULL_PX = 25.0
TIP_BOX_CONFIDENCE = 0.9
# A fragmented mask (skeleton residual over this) casts no taper vote. Same gate as the
# tracker's merged-mask residual.
SKELETON_RESIDUAL_MAX_PX = 25.0
# Width ratio at which the taper cue starts, and the ratio that saturates its confidence.
# The minimum matches the tracker's LINE_WIDTH_RATIO_MIN.
WIDTH_RATIO_MIN = 1.25
WIDTH_RATIO_FULL = 2.5
# A named long-thin tail, already gated on the row. Not used for the wide-tip classes.
TAIL_CONFIDENCE = 0.75

GRAVITY_SILENT_DEG = 45.0
GRAVITY_FULL_DEG = 75.0

# The hand cue's reach and margin are the tracker's LINE_HAND_REACH_CM and
# LINE_HAND_MARGIN_CM. Confidence is 1 with the hand on the end and 0 at the reach.
HAND_REACH_CM = 15.0
HAND_MARGIN_CM = 3.0

CUE_COLOUR = "colour"
CUE_TAPER = "taper"
CUE_HAND = "hand"
CUE_GRAVITY = "gravity"
CUE_TIP_BOX = "tip_box"
CUE_NAMES = (CUE_COLOUR, CUE_TAPER, CUE_HAND, CUE_GRAVITY, CUE_TIP_BOX)


@dataclass(frozen=True)
class CueVote:
    """One cue's one contribution this frame. `tip_end` is 0 or 1 in track-endpoint order."""

    cue: str
    tip_end: int
    confidence: float


@dataclass(frozen=True)
class OrientRecord:
    """One line row's vote, in track-endpoint order (positive: end 0 is the tip)."""

    track_id: str
    frame_index: int
    episode: int
    contribution: float
    endpoint0_is_track_end0: bool


@dataclass
class EpisodeAccumulator:
    """Online log-odds with decay, plus the decay-free sum of the current retrofit episode.

    `resume` keeps `log_odds` and `sign` (the prior across a gap) and starts a new
    decay-free episode. The flip margin still applies to the carried value.
    """

    log_odds: float = 0.0
    sign: int = 0
    episode: int = 0
    decay_free: float = 0.0
    peak_abs: float = -1.0
    peak_sign: int = 0

    def add(self, contribution: float, *, elapsed_frames: int = 1) -> None:
        self.log_odds = self.log_odds * DECAY_PER_FRAME**elapsed_frames + float(contribution)
        if self.sign > 0:
            if self.log_odds < -FLIP_MARGIN_LOG_ODDS:
                self.sign = -1
        elif self.sign < 0:
            if self.log_odds > FLIP_MARGIN_LOG_ODDS:
                self.sign = 1
        elif self.log_odds > RESOLVE_LOG_ODDS:
            self.sign = 1
        elif self.log_odds < -RESOLVE_LOG_ODDS:
            self.sign = -1
        self.decay_free += float(contribution)
        if abs(self.decay_free) > self.peak_abs:
            self.peak_abs = abs(self.decay_free)
            self.peak_sign = 1 if self.decay_free > 0 else -1 if self.decay_free < 0 else 0

    def resume(self) -> None:
        self.episode += 1
        self.decay_free = 0.0
        self.peak_abs = -1.0
        self.peak_sign = 0

    @property
    def resolved(self) -> bool:
        return abs(self.log_odds) > RESOLVE_LOG_ODDS

    @property
    def confidence(self) -> float:
        """Sigmoid of the online accumulator: near 1 when endpoint 0 is the tip, near 0
        when endpoint 1 is."""
        odds = max(-40.0, min(40.0, self.log_odds))
        return float(1.0 / (1.0 + math.exp(-odds)))


def length_factor(length_px: float) -> float:
    if length_px <= 0:
        return 0.0
    return min(1.0, float(length_px) / LENGTH_FULL_PX)


def colour_confidence(hue_share: float, width_px: float) -> float:
    """The sampler's hue-bin share times a size factor, clamped to [0, 1]."""
    size = min(1.0, max(0.0, float(width_px)) / COLOUR_SIZE_FULL_PX)
    return float(np.clip(float(hue_share) * size, 0.0, 1.0))


def best_camera(
    candidates: list[tuple[int, float, float]] | tuple[tuple[int, float, float], ...],
) -> tuple[int, float] | None:
    """`(tip_end, confidence, length_px)` per camera. The camera with the largest
    confidence times length factor wins, and that product is the cue's confidence.
    Cameras are not summed."""
    best_end = 0
    best_weight = 0.0
    found = False
    for tip_end, confidence, length_px in candidates:
        weight = float(np.clip(confidence, 0.0, 1.0)) * length_factor(length_px)
        if weight > best_weight:
            best_weight = weight
            best_end = int(tip_end)
            found = True
    if not found or best_weight <= 0.0:
        return None
    return best_end, best_weight


def width_confidence(ratio: float) -> float:
    """0 at the 1.25 width-ratio gate, 1 at 2.5, clamped."""
    if ratio < WIDTH_RATIO_MIN:
        return 0.0
    span = WIDTH_RATIO_FULL - WIDTH_RATIO_MIN
    return float(np.clip((ratio - WIDTH_RATIO_MIN) / span, 0.0, 1.0))


def taper_camera(
    *,
    end_widths: tuple[float, float] | None,
    tip_side: int | None,
    wide_is_tip: bool,
    axis_residual_px: float | None,
    axis_to_track: tuple[int, int] | None,
    residual_max_px: float = SKELETON_RESIDUAL_MAX_PX,
) -> tuple[int, float] | None:
    """One camera's taper vote in track-endpoint order, or None.

    The wide end is the tip for `wide_is_tip` classes (the 8-channel manifold) and the
    butt otherwise. The long-thin tail is added only when `wide_is_tip` is false: the
    tail rule is unvalidated on the 8-channel, and the width already names its tip.
    Width and tail must agree. A skeleton residual over the gate abstains.
    """
    if axis_to_track is None:
        return None
    if axis_residual_px is not None and axis_residual_px > residual_max_px:
        return None
    width_vote: tuple[int, float] | None = None
    if end_widths is not None:
        w0, w1 = float(end_widths[0]), float(end_widths[1])
        small = min(w0, w1)
        if small > 0:
            ratio = max(w0, w1) / small
            confidence = width_confidence(ratio)
            if confidence > 0:
                wide_axis = 0 if w0 > w1 else 1
                tip_axis = wide_axis if wide_is_tip else 1 - wide_axis
                width_vote = (axis_to_track[tip_axis], confidence)
    tail_vote: tuple[int, float] | None = None
    if not wide_is_tip and tip_side is not None:
        tail_vote = (axis_to_track[int(tip_side)], TAIL_CONFIDENCE)
    if width_vote is not None and tail_vote is not None:
        if width_vote[0] != tail_vote[0]:
            return None
        return width_vote[0], max(width_vote[1], tail_vote[1])
    return width_vote if width_vote is not None else tail_vote


def hand_cue(
    endpoints: np.ndarray,
    hand_positions: list[np.ndarray] | tuple[np.ndarray, ...],
    *,
    both_ends_in_hand_box: bool,
    reach_cm: float = HAND_REACH_CM,
    margin_cm: float = HAND_MARGIN_CM,
) -> CueVote | None:
    """The hand end is the butt. Confidence falls from 1 at the end to 0 at `reach_cm`.

    The nearer end must be inside the reach and at least `margin_cm` nearer than the
    other. Both ends inside a hand box abstain, whatever the hand track says.
    """
    if both_ends_in_hand_box:
        return None
    best: tuple[float, int] | None = None
    ends = np.asarray(endpoints, dtype=np.float64).reshape(2, 3)
    for hand in hand_positions:
        distances = np.linalg.norm(ends - np.asarray(hand, dtype=np.float64).reshape(3), axis=1)
        d0, d1 = float(distances[0]), float(distances[1])
        near = min(d0, d1)
        if near > reach_cm or abs(d0 - d1) < margin_cm:
            continue
        confidence = (reach_cm - near) / reach_cm
        if confidence <= 0:
            continue
        tip_end = 0 if d0 > d1 else 1
        if best is None or confidence > best[0]:
            best = (confidence, tip_end)
    if best is None:
        return None
    return CueVote(CUE_HAND, best[1], float(np.clip(best[0], 0.0, 1.0)))


def _elevation_deg(direction: np.ndarray) -> float:
    vertical = abs(float(np.dot(direction, BENCH_UP)))
    vertical = min(1.0, max(0.0, vertical))
    return math.degrees(math.asin(vertical))


def gravity_cue(endpoints: np.ndarray) -> CueVote | None:
    """The lower end is the tip once the line is above 45 degrees from horizontal.

    Confidence is 0 at 45 degrees and 1 at 75. At 45 and below the cue is silent:
    pipettes rest flat, and gravity must not decide those frames. Up is `BENCH_UP`.
    """
    ends = np.asarray(endpoints, dtype=np.float64).reshape(2, 3)
    span = ends[1] - ends[0]
    norm = float(np.linalg.norm(span))
    if norm < 1e-9:
        return None
    angle = _elevation_deg(span / norm)
    if angle <= GRAVITY_SILENT_DEG:
        return None
    confidence = (angle - GRAVITY_SILENT_DEG) / (GRAVITY_FULL_DEG - GRAVITY_SILENT_DEG)
    confidence = float(np.clip(confidence, 0.0, 1.0))
    if confidence <= 0:
        return None
    heights = ends @ BENCH_UP
    return CueVote(CUE_GRAVITY, int(np.argmin(heights)), confidence)


def tip_box_cue(
    cameras: list[tuple[int, float]] | tuple[tuple[int, float], ...],
) -> CueVote | None:
    """An attached tip box marks its end at confidence 0.9, then the length-weighted best camera."""
    picked = best_camera([(end, TIP_BOX_CONFIDENCE, length) for end, length in cameras])
    if picked is None:
        return None
    return CueVote(CUE_TIP_BOX, picked[0], picked[1])


def colour_cue(
    cameras: list[tuple[int, float, float]] | tuple[tuple[int, float, float], ...],
) -> CueVote | None:
    """`(track tip end, confidence, length_px)` per view. The caller omits ambiguous views."""
    picked = best_camera(cameras)
    if picked is None:
        return None
    return CueVote(CUE_COLOUR, picked[0], picked[1])


def taper_cue(
    cameras: list[tuple[int, float, float]] | tuple[tuple[int, float, float], ...],
) -> CueVote | None:
    picked = best_camera(cameras)
    if picked is None:
        return None
    return CueVote(CUE_TAPER, picked[0], picked[1])


def frame_log_odds(votes: list[CueVote] | tuple[CueVote, ...]) -> tuple[float, dict[str, float]]:
    """One contribution per cue. Positive log-odds: endpoint 0 is the tip.

    A repeated cue name keeps the last vote. Callers pass one vote per cue.
    """
    per: dict[str, float] = {}
    for vote in votes:
        sign = 1.0 if int(vote.tip_end) == 0 else -1.0
        per[vote.cue] = LOG_ODDS_SCALE * float(np.clip(vote.confidence, 0.0, 1.0)) * sign
    return float(sum(per.values())), per


def peak_signs(
    records: list[OrientRecord] | tuple[OrientRecord, ...],
) -> dict[tuple[str, int], int]:
    """`(track_id, episode) -> sign` at the frame of maximum |decay-free sum|.

    Positive means track endpoint 0 is the tip. The sign is 0 when the peak never
    clears the resolve threshold, so a weak episode is left as the online pass wrote it.
    Ties keep the earlier frame.
    """
    grouped: dict[tuple[str, int], list[OrientRecord]] = {}
    for record in records:
        grouped.setdefault((record.track_id, record.episode), []).append(record)
    signs: dict[tuple[str, int], int] = {}
    for key, items in grouped.items():
        total = 0.0
        peak_abs = -1.0
        peak_sign = 0
        for record in sorted(items, key=lambda item: item.frame_index):
            total += record.contribution
            if abs(total) > peak_abs:
                peak_abs = abs(total)
                peak_sign = 1 if total > 0 else -1 if total < 0 else 0
        signs[key] = peak_sign if peak_abs > RESOLVE_LOG_ODDS else 0
    return signs


def plunger_end_from_sample(
    sample: Mapping[str, object] | None, width_px: float
) -> dict[str, object] | None:
    """One view's plunger sample, or None when neither end showed a ring.

    `ambiguous` (both ends showed a ring) abstains: `tip_end` is null and the confidence
    is 0. Otherwise the confidence is the hue-bin share times the size factor, clamped.
    """
    if sample is None:
        return None
    if sample.get("ambiguous"):
        return {"ambiguous": True, "tip_end": None, "confidence": 0.0}
    confidence = colour_confidence(float(sample["confidence"]), width_px)  # type: ignore[arg-type]
    if confidence <= 0:
        return None
    return {
        "ambiguous": False,
        "tip_end": int(sample["tip_index"]),  # type: ignore[arg-type]
        "confidence": round(confidence, 4),
    }


def load_plunger_ends(path: Path) -> dict[tuple[str, int, str], tuple[int | None, float, bool]]:
    """``(view, frame, slot) -> (tip axis index or None, confidence, ambiguous)``."""
    out: dict[tuple[str, int, str], tuple[int | None, float, bool]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            tip = row.get("tip_end")
            out[(str(row["view"]), int(row["frame_index"]), str(row["slot"]))] = (
                None if tip is None else int(tip),
                float(row.get("confidence") or 0.0),
                bool(row.get("ambiguous")),
            )
    return out


def cue_agreement(votes: dict[str, float] | None, retrofit_sign: int | None) -> dict[str, str]:
    """Per cue, `agree` or `differ` with the retrofit sign, in track-endpoint coordinates.

    A missing cue or a sign of 0 is omitted. Positive votes and a positive sign both
    mean track endpoint 0 is the tip.
    """
    if not votes or retrofit_sign not in (1, -1):
        return {}
    out: dict[str, str] = {}
    for cue, value in votes.items():
        if not value:
            continue
        same = (float(value) > 0) == (int(retrofit_sign) > 0)
        out[cue] = "agree" if same else "differ"
    return out
