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


def _rows(hands: tuple[PerFrameHand, ...]) -> dict[int, FrameObservations]:
    return {
        frame: FrameObservations(
            view_id="static-c10379",
            analysis_frame_index=frame,
            source_seconds=294 + frame / 30,
            hands=hands,
        )
        for frame in range(600)
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


def test_wrist_jitter_is_zero_without_consecutive_hands() -> None:
    observations = tuple(_rows(()).values())
    assert wrist_jitter(observations) == 0.0
