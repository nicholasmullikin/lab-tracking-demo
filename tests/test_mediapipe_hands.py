from __future__ import annotations

from battle.mediapipe_hands import (
    HandTrackAssigner,
    _landmark_box,
    anatomical_side,
    canonical_hand_ids,
    fuse_hand_candidates,
    parse_roi,
    remap_landmarks,
)
from battle.schemas import HandSide, NormalizedPoint


def test_unmirrored_camera_swaps_mediapipe_selfie_handedness() -> None:
    assert anatomical_side(HandSide.LEFT, input_mirrored=False) is HandSide.RIGHT
    assert anatomical_side(HandSide.RIGHT, input_mirrored=False) is HandSide.LEFT
    assert anatomical_side(HandSide.LEFT, input_mirrored=True) is HandSide.LEFT
    assert anatomical_side(HandSide.UNKNOWN, input_mirrored=False) is HandSide.UNKNOWN


def test_track_assigner_keeps_nearby_wrist_id_and_temporally_votes_side() -> None:
    assigner = HandTrackAssigner()

    first = assigner.assign(frame_index=0, wrists=((0.2, 0.3),), sides=(HandSide.LEFT,))
    second = assigner.assign(frame_index=1, wrists=((0.21, 0.31),), sides=(HandSide.RIGHT,))
    third = assigner.assign(frame_index=2, wrists=((0.22, 0.32),), sides=(HandSide.LEFT,))

    assert first == (("hand-1", HandSide.LEFT),)
    assert second == (("hand-1", HandSide.RIGHT),)
    assert third == (("hand-1", HandSide.LEFT),)


def test_landmark_box_stays_valid_at_normalized_image_edge() -> None:
    landmarks = tuple(NormalizedPoint(x=1.0, y=1.0) for _ in range(21))

    box = _landmark_box(landmarks)

    assert box.x + box.width <= 1.0
    assert box.y + box.height <= 1.0
    assert box.width > 0
    assert box.height > 0


def test_public_hand_ids_are_explicitly_frame_local() -> None:
    assignments = (
        ("internal-4", HandSide.RIGHT),
        ("internal-9", HandSide.LEFT),
        ("internal-10", HandSide.LEFT),
    )

    assert canonical_hand_ids(assignments) == (
        ("hand-detection-1", HandSide.RIGHT),
        ("hand-detection-2", HandSide.LEFT),
        ("hand-detection-3", HandSide.LEFT),
    )


def test_roi_landmarks_map_back_to_full_frame_coordinates() -> None:
    roi = parse_roi("0.5,0.25,0.5,0.5")
    landmarks = tuple(NormalizedPoint(x=0.2, y=0.4) for _ in range(21))

    remapped = remap_landmarks(landmarks, roi)

    assert remapped[0] == NormalizedPoint(x=0.6, y=0.45)


def test_two_pass_fusion_keeps_best_duplicate_and_distinct_second_hand() -> None:
    def hand(x: float, y: float) -> tuple[NormalizedPoint, ...]:
        return tuple(
            NormalizedPoint(x=x + index * 0.001, y=y + index * 0.001) for index in range(21)
        )

    fused = fuse_hand_candidates(
        ((hand(0.60, 0.50), HandSide.LEFT, 0.8),),
        (
            (hand(0.61, 0.51), HandSide.LEFT, 0.95),
            (hand(0.82, 0.60), HandSide.RIGHT, 0.7),
        ),
    )

    assert len(fused) == 2
    assert [candidate[2] for candidate in fused] == [0.8, 0.7]

    full_primary = fuse_hand_candidates(
        (
            (hand(0.50, 0.40), HandSide.LEFT, 0.8),
            (hand(0.70, 0.40), HandSide.RIGHT, 0.8),
        ),
        ((hand(0.90, 0.70), HandSide.RIGHT, 0.99),),
    )

    assert len(full_primary) == 2

    deduplicated_primary = fuse_hand_candidates(
        (
            (hand(0.50, 0.40), HandSide.LEFT, 0.8),
            (hand(0.51, 0.41), HandSide.LEFT, 0.7),
        ),
        ((hand(0.80, 0.60), HandSide.RIGHT, 0.75),),
    )

    assert [candidate[2] for candidate in deduplicated_primary] == [0.8, 0.75]
