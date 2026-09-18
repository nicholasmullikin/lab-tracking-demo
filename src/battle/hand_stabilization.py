"""Deterministically stabilize WiLoR hand observations without hiding raw gaps.

This is a rendering/review postprocessor, not a hand tracker or a source of labels.  It
uses short-lived spatial lanes solely to smooth a hand's rigid wrist/palm motion and
never exposes those lanes as persistent identities.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .schemas import FrameObservations, NormalizedBox, NormalizedPoint, PerFrameHand

DEFAULT_FRAME_COUNT = 600
WILOR_MIN_CONFIDENCE = 0.55
# A partially visible hand (a few fingers) often keeps a real WiLoR detection just under the
# primary gate. Such detections are accepted only while they continue a lane that was
# accepted at full confidence within MAX_FALLBACK_GAP frames, for at most
# MAX_CONTINUATION_STREAK consecutive frames, and they are tagged distinctly.
WILOR_CONTINUATION_MIN_CONFIDENCE = 0.35
MAX_CONTINUATION_STREAK = 5
MEDIAPIPE_MIN_CONFIDENCE = 0.85
MAX_FALLBACK_GAP = 5
MAX_LANE_DISTANCE = 0.16
MIN_HAND_AREA = 0.0004
MAX_HAND_AREA = 0.22
DEDUP_WRIST_DISTANCE = 0.035


@dataclass(frozen=True)
class HandProvenance:
    """One explicit postprocessing disposition for a rendered hand."""

    analysis_frame_index: int
    output_hand_id: str
    source: str
    state: str
    reason: str
    raw_hand_id: str | None


@dataclass(frozen=True)
class StabilizationResult:
    observations: tuple[FrameObservations, ...]
    provenance: tuple[HandProvenance, ...]
    metrics: dict[str, float | int]


def _wrist(hand: PerFrameHand) -> np.ndarray:
    return np.asarray((hand.landmarks[0].x, hand.landmarks[0].y), dtype=np.float64)


def _palm(hand: PerFrameHand) -> np.ndarray:
    points = np.asarray([(point.x, point.y) for point in hand.landmarks[:5]], dtype=np.float64)
    return points.mean(axis=0)


def _box_area(hand: PerFrameHand) -> float:
    return hand.box.width * hand.box.height


def _valid_shape(hand: PerFrameHand) -> bool:
    area = _box_area(hand)
    if not MIN_HAND_AREA <= area <= MAX_HAND_AREA:
        return False
    points = np.asarray([(point.x, point.y) for point in hand.landmarks], dtype=np.float64)
    return bool(
        np.isfinite(points).all() and np.ptp(points[:, 0]) > 0.003 and np.ptp(points[:, 1]) > 0.003
    )


def _deduplicate(hands: tuple[PerFrameHand, ...]) -> list[PerFrameHand]:
    """Keep the highest-confidence wrist-near candidate with deterministic tie breaks."""

    kept: list[PerFrameHand] = []
    for hand in sorted(hands, key=lambda item: (-item.confidence, item.hand_id)):
        if hand.confidence < WILOR_MIN_CONFIDENCE or not _valid_shape(hand):
            continue
        if any(
            float(np.linalg.norm(_wrist(hand) - _wrist(other))) < DEDUP_WRIST_DISTANCE
            for other in kept
        ):
            continue
        kept.append(hand)
    return kept


def _continuation_candidates(
    hands: tuple[PerFrameHand, ...],
    kept: list[PerFrameHand],
    active: dict[str, tuple[PerFrameHand, np.ndarray, int]],
    streaks: dict[str, int],
    frame: int,
    parts: FrameObservations | None,
) -> dict[str, str]:
    """Return {raw hand id: lane} for sub-gate WiLoR detections that continue a live lane.

    Nothing here invents a position: every accepted hand is a real WiLoR detection on this
    frame.  It only relaxes the confidence gate to WILOR_CONTINUATION_MIN_CONFIDENCE when the
    wrist stays within MAX_LANE_DISTANCE of a lane accepted within MAX_FALLBACK_GAP frames and
    that lane has not already run MAX_CONTINUATION_STREAK consecutive relaxed frames.
    """

    accepted: dict[str, str] = {}
    taken: list[PerFrameHand] = list(kept)
    for hand in sorted(hands, key=lambda item: (-item.confidence, item.hand_id)):
        if not WILOR_CONTINUATION_MIN_CONFIDENCE <= hand.confidence < WILOR_MIN_CONFIDENCE:
            continue
        if not _valid_shape(hand) or not _workspace_supported(hand, parts):
            continue
        if any(
            float(np.linalg.norm(_wrist(hand) - _wrist(other))) < DEDUP_WRIST_DISTANCE
            for other in taken
        ):
            continue
        nearest = sorted(
            (
                (float(np.linalg.norm(_wrist(hand) - _wrist(previous))), lane)
                for lane, (previous, _, seen) in active.items()
                if frame - seen <= MAX_FALLBACK_GAP
                and streaks.get(lane, 0) < MAX_CONTINUATION_STREAK
                and lane not in accepted.values()
            ),
            key=lambda item: (item[0], item[1]),
        )
        if not nearest or nearest[0][0] > MAX_LANE_DISTANCE:
            continue
        accepted[hand.hand_id] = nearest[0][1]
        taken.append(hand)
    return accepted


def _workspace_distance(point: np.ndarray, parts: FrameObservations | None) -> float | None:
    """Distance to the evidence-defined union of current SAM3 part boxes."""

    if parts is None or not parts.objects:
        return None
    distances = []
    for item in parts.objects:
        x0, y0 = item.box.x, item.box.y
        x1, y1 = x0 + item.box.width, y0 + item.box.height
        dx = max(x0 - point[0], 0.0, point[0] - x1)
        dy = max(y0 - point[1], 0.0, point[1] - y1)
        distances.append(float(np.hypot(dx, dy)))
    return min(distances) if distances else None


def _workspace_supported(hand: PerFrameHand, parts: FrameObservations | None) -> bool:
    """Reject only candidates unsupported by the current part workspace evidence.

    The allowance scales with the observed part-box union rather than using a fixed image
    strip.  A hand overlapping/near the named interaction parts is retained; when no
    usable part evidence exists, the gate deliberately does not invent a rejection.
    """

    distance = _workspace_distance(_wrist(hand), parts)
    if distance is None:
        return True
    areas = [item.box.width * item.box.height for item in parts.objects]
    support_radius = max(0.08, min(0.25, float(np.sqrt(sum(areas))) * 0.7))
    return distance <= support_radius


def _one_euro(
    previous: np.ndarray, current: np.ndarray, velocity: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """One-Euro update at the fixed 30 Hz review clock."""

    dt = 1 / 30
    d_cutoff = 1.0
    min_cutoff = 1.5
    beta = 0.007
    derivative = (current - previous) / dt
    alpha_d = 1.0 / (1.0 + (1.0 / (2 * np.pi * d_cutoff)) / dt)
    filtered_velocity = alpha_d * derivative + (1 - alpha_d) * velocity
    cutoff = min_cutoff + beta * float(np.linalg.norm(filtered_velocity))
    alpha = 1.0 / (1.0 + (1.0 / (2 * np.pi * cutoff)) / dt)
    return alpha * current + (1 - alpha) * previous, filtered_velocity


def _smoothed_hand(
    hand: PerFrameHand, prior: PerFrameHand, velocity: np.ndarray
) -> tuple[PerFrameHand, np.ndarray]:
    """Smooth only the wrist/palm rigid translation; preserve relative fingers."""

    prior_wrist, wrist = _wrist(prior), _wrist(hand)
    prior_palm, palm = _palm(prior), _palm(hand)
    filtered_wrist, filtered_velocity = _one_euro(prior_wrist, wrist, velocity)
    filtered_palm, _ = _one_euro(prior_palm, palm, velocity)
    raw_origin = (wrist + palm) / 2
    filtered_origin = (filtered_wrist + filtered_palm) / 2
    translation = filtered_origin - raw_origin
    landmarks = tuple(
        NormalizedPoint(
            x=float(np.clip(point.x + translation[0], 0, 1)),
            y=float(np.clip(point.y + translation[1], 0, 1)),
        )
        for point in hand.landmarks
    )
    xs, ys = [point.x for point in landmarks], [point.y for point in landmarks]
    box = NormalizedBox(
        x=max(0.0, min(xs)),
        y=max(0.0, min(ys)),
        width=max(0.0001, min(1.0, max(xs)) - max(0.0, min(xs))),
        height=max(0.0001, min(1.0, max(ys)) - max(0.0, min(ys))),
    )
    return hand.model_copy(update={"landmarks": landmarks, "box": box}), filtered_velocity


def stabilize(
    wilor: dict[int, FrameObservations],
    mediapipe: dict[int, FrameObservations],
    parts: dict[int, FrameObservations],
    *,
    frame_count: int = DEFAULT_FRAME_COUNT,
) -> StabilizationResult:
    """Apply WiLoR-primary filtering, limited MP fallback, and short-lane smoothing."""

    expected = set(range(frame_count))
    for name, rows in (("WiLoR", wilor), ("MediaPipe", mediapipe), ("parts", parts)):
        if not expected.issubset(rows):
            raise ValueError(f"{name} rows must contain every frame [0,{frame_count})")
    output: list[FrameObservations] = []
    provenance: list[HandProvenance] = []
    active: dict[str, tuple[PerFrameHand, np.ndarray, int]] = {}
    continuation_streaks: dict[str, int] = {}
    fallback_frames = raw_frames = smoothed_frames = missing_frames = 0
    continuation_instances = 0
    for frame in range(frame_count):
        raw = wilor[frame]
        candidates = _deduplicate(raw.hands)
        # A far candidate is only evidence of a camera-rig phantom when another hand on
        # that frame anchors the interaction workspace. A lone incoming/leaving hand is
        # retained instead of being rejected by an unexplained image strip.
        supported = [hand for hand in candidates if _workspace_supported(hand, parts.get(frame))]
        if len(candidates) > 2 and len(supported) >= 2:
            candidates = supported
        continuation = _continuation_candidates(
            raw.hands, candidates, active, continuation_streaks, frame, parts.get(frame)
        )
        candidates = [*candidates, *(hand for hand in raw.hands if hand.hand_id in continuation)]
        source = "wilor"
        if not candidates:
            forward = next(
                (
                    index
                    for index in range(frame + 1, min(frame_count, frame + MAX_FALLBACK_GAP + 1))
                    if _deduplicate(wilor[index].hands)
                ),
                None,
            )
            mp = [
                hand
                for hand in mediapipe[frame].hands
                if hand.confidence >= MEDIAPIPE_MIN_CONFIDENCE
                and _valid_shape(hand)
                and _workspace_supported(hand, parts.get(frame))
            ]
            if forward is not None and mp:
                candidates, source = mp, "mediapipe_fallback"
            else:
                missing_frames += 1
        rendered: list[PerFrameHand] = []
        available = dict(active)
        ordered = sorted(candidates, key=lambda item: (item.landmarks[0].x, item.hand_id))
        for index, hand in enumerate(ordered):
            nearest = sorted(
                (
                    (float(np.linalg.norm(_wrist(hand) - _wrist(previous))), lane)
                    for lane, (previous, _, seen) in available.items()
                    if frame - seen <= MAX_FALLBACK_GAP
                ),
                key=lambda item: (item[0], item[1]),
            )
            continued_lane = continuation.get(hand.hand_id)
            if continued_lane is not None and continued_lane in available:
                lane = continued_lane
            else:
                lane = (
                    nearest[0][1]
                    if nearest and nearest[0][0] <= MAX_LANE_DISTANCE
                    else f"lane-{frame}-{index}"
                )
            previous = active.get(lane)
            reason = "accepted_after_gates"
            if previous is not None and source == "wilor":
                value, velocity = _smoothed_hand(hand, previous[0], previous[1])
                state = "smoothed"
                smoothed_frames += 1
            else:
                value = hand
                velocity = np.zeros(2)
                state = "raw" if source == "wilor" else "fallback"
            if continued_lane is not None:
                state = "low_confidence_continuation"
                reason = "wilor_confidence_below_gate_within_active_lane"
                continuation_streaks[lane] = continuation_streaks.get(lane, 0) + 1
                continuation_instances += 1
            else:
                continuation_streaks[lane] = 0
            rendered.append(value.model_copy(update={"hand_id": f"stabilized-{lane}"}))
            active[lane] = (hand, velocity, frame)
            available.pop(lane, None)
            provenance.append(
                HandProvenance(
                    frame,
                    f"stabilized-{lane}",
                    source,
                    state,
                    reason,
                    hand.hand_id,
                )
            )
        if source == "wilor" and candidates:
            raw_frames += 1
        if source == "mediapipe_fallback":
            fallback_frames += 1
        if not rendered:
            provenance.append(
                HandProvenance(frame, "", "missing", "missing", "no_supported_hand", None)
            )
        output.append(raw.model_copy(update={"hands": tuple(rendered)}))
    metrics: dict[str, float | int] = {
        "raw_wilor_hand_frames": sum(bool(wilor[frame].hands) for frame in range(frame_count)),
        "stabilized_hand_frames": sum(bool(item.hands) for item in output),
        "wilor_primary_frames": raw_frames,
        "mediapipe_fallback_frames": fallback_frames,
        "missing_frames": missing_frames,
        "smoothed_hand_instances": smoothed_frames,
        "low_confidence_continuation_instances": continuation_instances,
        "wilor_min_confidence": WILOR_MIN_CONFIDENCE,
        "wilor_continuation_min_confidence": WILOR_CONTINUATION_MIN_CONFIDENCE,
        "max_continuation_streak_frames": MAX_CONTINUATION_STREAK,
        "mediapipe_min_confidence": MEDIAPIPE_MIN_CONFIDENCE,
        "max_fallback_gap_frames": MAX_FALLBACK_GAP,
    }
    return StabilizationResult(tuple(output), tuple(provenance), metrics)


def write_result(result: StabilizationResult, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "observations.jsonl").write_text(
        "".join(item.model_dump_json() + "\n" for item in result.observations), encoding="utf-8"
    )
    (directory / "hand_provenance.json").write_text(
        json.dumps([item.__dict__ for item in result.provenance], indent=2) + "\n", encoding="utf-8"
    )
    (directory / "stabilization_metrics.json").write_text(
        json.dumps(result.metrics, indent=2) + "\n", encoding="utf-8"
    )


def wrist_jitter(observations: tuple[FrameObservations, ...]) -> float:
    """Median nearest-wrist frame delta in normalized image coordinates."""

    deltas: list[float] = []
    prior: list[np.ndarray] = []
    for observation in observations:
        current = [_wrist(hand) for hand in observation.hands]
        for point in current:
            if prior:
                deltas.append(min(float(np.linalg.norm(point - candidate)) for candidate in prior))
        prior = current
    return float(np.median(deltas)) if deltas else 0.0


def enrich_metrics(
    result: StabilizationResult,
    wilor: dict[int, FrameObservations],
    *,
    frame_count: int = DEFAULT_FRAME_COUNT,
) -> dict[str, float | int]:
    """Add transparent output-stability diagnostics, never pose accuracy."""

    metrics = dict(result.metrics)
    raw = tuple(wilor[index] for index in range(frame_count))
    stabilized = result.observations
    raw_jitter = wrist_jitter(raw)
    stabilized_jitter = wrist_jitter(stabilized)
    raw_instances = sum(len(item.hands) for item in raw)
    output_instances = sum(len(item.hands) for item in stabilized)
    metrics.update(
        {
            "raw_wrist_jitter_normalized": raw_jitter,
            "stabilized_wrist_jitter_normalized": stabilized_jitter,
            "wrist_jitter_reduction_fraction": (
                (raw_jitter - stabilized_jitter) / raw_jitter if raw_jitter else 0.0
            ),
            "raw_hand_instances": raw_instances,
            "stabilized_hand_instances": output_instances,
            "filtered_or_missing_raw_instances": max(0, raw_instances - output_instances),
            "metric_scope": "output-stability diagnostics, not pose accuracy",
        }
    )
    return metrics
