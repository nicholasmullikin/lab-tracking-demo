from __future__ import annotations

from battle.hand_stabilization import MAX_FALLBACK_GAP, stabilize, wrist_jitter
from battle.schemas import (
    FrameObservations,
    HandSide,
    NormalizedBox,
    NormalizedPoint,
    PerFrameHand,
)


def _hand(x: float, *, confidence: float = 0.9) -> PerFrameHand:
    return PerFrameHand(
        hand_id="raw",
        side=HandSide.LEFT,
        confidence=confidence,
        landmarks=tuple(
            NormalizedPoint(x=x + index * 0.001, y=0.5 + (index % 5) * 0.004) for index in range(21)
        ),
        box=NormalizedBox(x=x, y=0.45, width=0.08, height=0.1),
        model_side=HandSide.LEFT,
        model_handedness_confidence=confidence,
    )


def _rows(hands: tuple[PerFrameHand, ...], frame_count: int = 600) -> dict[int, FrameObservations]:
    return {
        frame: FrameObservations(
            view_id="static-c10379",
            analysis_frame_index=frame,
            source_seconds=294 + frame / 30,
            hands=hands,
        )
        for frame in range(frame_count)
    }


def test_stabilization_prefers_wilor_and_rejects_low_confidence_duplicates() -> None:
    wilor = _rows((_hand(0.5), _hand(0.505, confidence=0.4)))
    mediapipe = _rows((_hand(0.6, confidence=0.99),))

    result = stabilize(wilor, mediapipe, _rows(()))

    assert len(result.observations[0].hands) == 1
    assert result.provenance[0].source == "wilor"
    assert result.metrics["mediapipe_fallback_frames"] == 0


def test_mediapipe_fallback_requires_short_future_wilor_gap() -> None:
    wilor = _rows(())
    wilor[3] = wilor[3].model_copy(update={"hands": (_hand(0.5),)})
    mediapipe = _rows((_hand(0.6, confidence=0.99),))

    result = stabilize(wilor, mediapipe, _rows(()))

    assert result.provenance[0].state == "fallback"
    assert result.provenance[10].state == "missing"
    assert MAX_FALLBACK_GAP == 5


def test_sub_gate_wilor_detection_continues_a_live_lane_for_a_bounded_streak() -> None:
    wilor = _rows(())
    for frame in range(3):
        wilor[frame] = wilor[frame].model_copy(update={"hands": (_hand(0.5),)})
    for frame in range(3, 12):
        wilor[frame] = wilor[frame].model_copy(
            update={"hands": (_hand(0.5 + 0.002 * frame, confidence=0.45),)}
        )
    # A lone sub-gate detection far from any lane never starts a hand on its own.
    wilor[300] = wilor[300].model_copy(update={"hands": (_hand(0.2, confidence=0.45),)})

    result = stabilize(wilor, _rows(()), _rows(()))
    by_frame = {item.analysis_frame_index: item for item in result.provenance}

    assert by_frame[2].state == "smoothed"
    assert all(by_frame[frame].state == "low_confidence_continuation" for frame in range(3, 8))
    assert all(
        by_frame[frame].reason == "wilor_confidence_below_gate_within_active_lane"
        for frame in range(3, 8)
    )
    assert by_frame[8].state == "missing"  # streak cap of five relaxed frames
    assert by_frame[300].state == "missing"
    assert result.metrics["low_confidence_continuation_instances"] == 5
    assert result.metrics["wilor_continuation_min_confidence"] == 0.35
    assert result.observations[3].hands[0].hand_id == result.observations[2].hands[0].hand_id


def test_wrist_jitter_is_zero_without_consecutive_hands() -> None:
    observations = tuple(_rows(()).values())
    assert wrist_jitter(observations) == 0.0


def test_stabilization_preserves_every_first_minute_row_and_provenance() -> None:
    frame_count = 1800
    wilor = _rows((), frame_count)
    wilor[1200] = wilor[1200].model_copy(update={"hands": (_hand(0.5),)})
    result = stabilize(
        wilor, _rows((_hand(0.6),), frame_count), _rows((), frame_count), frame_count=frame_count
    )

    assert len(result.observations) == frame_count
    assert result.observations[1200].hands
    assert {item.analysis_frame_index for item in result.provenance} == set(range(frame_count))
    assert result.metrics["missing_frames"] == frame_count - 6
    assert result.metrics["mediapipe_fallback_frames"] == 5
