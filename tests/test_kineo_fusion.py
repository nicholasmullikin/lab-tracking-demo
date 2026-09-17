from __future__ import annotations

from battle.kineo_fusion import FRAME_COUNT, fuse_person_boxes, verify_source_alignment
from battle.schemas import FrameObservations, NormalizedBox


def _rows(
    with_boxes: set[int], *, shift: float = 0.0, frame_count: int = FRAME_COUNT
) -> dict[int, FrameObservations]:
    return {
        frame: FrameObservations(
            view_id="static-c10379",
            analysis_frame_index=frame,
            source_seconds=294 + frame / 30,
            objects=()
            if frame not in with_boxes
            else (
                {
                    "object_id": f"person-{frame}",
                    "label": "person",
                    "confidence": 0.9,
                    "box": NormalizedBox(x=0.2 + shift, y=0.1, width=0.3, height=0.7),
                },
            ),
        )
        for frame in range(frame_count)
    }


def test_fusion_prefers_native_uses_consistent_boxmot_and_limits_residual_gaps() -> None:
    native = _rows(set(range(FRAME_COUNT)) - set(range(100, 104)) - set(range(200, 206)))
    boxmot = _rows(set(range(FRAME_COUNT)) - set(range(200, 206)))

    fused = fuse_person_boxes(native, boxmot)

    assert all(item.source == "detected_native" for item in fused.provenance[:100])
    assert [item.source for item in fused.provenance[100:104]] == [
        "boxmot_fallback",
        "boxmot_fallback",
        "boxmot_fallback",
        "boxmot_fallback",
    ]
    assert all(item.source == "missing" for item in fused.provenance[200:206])
    assert fused.counts["boxmot_fallback"] == 4
    assert fused.counts["missing"] == 6


def test_fusion_interpolates_at_hard_limit_but_rejects_incompatible_boxmot() -> None:
    native = _rows(set(range(FRAME_COUNT)) - set(range(100, 105)) - set(range(300, 307)))
    boxmot = _rows(set(range(FRAME_COUNT)) - set(range(100, 105)) - set(range(301, 307)), shift=0.5)

    fused = fuse_person_boxes(native, boxmot)

    assert all(item.source == "interpolated" for item in fused.provenance[100:105])
    assert fused.provenance[100].residual_gap_length == 5
    assert all(item.source == "missing" for item in fused.provenance[300:307])
    assert fused.provenance[300].boxmot_box is not None
    assert fused.provenance[300].source_mapping_verified is False


def test_source_alignment_rejects_different_clock_mapping() -> None:
    from battle.fixtures import synthetic_run_manifest

    native = synthetic_run_manifest()
    static_clip = native.clip.model_copy(update={"views": ("static-c10379",)})
    native = native.model_copy(update={"clip": static_clip})
    boxmot = native.model_copy(
        update={
            "clip": static_clip.model_copy(
                update={
                    "timing": static_clip.timing.model_copy(
                        update={
                            "mappings": tuple(
                                mapping.model_copy(
                                    update={
                                        "source_offset_seconds": mapping.source_offset_seconds + 1
                                    }
                                )
                                for mapping in static_clip.timing.mappings
                            )
                        }
                    )
                }
            )
        }
    )

    try:
        verify_source_alignment(native, boxmot)
    except ValueError as error:
        assert "clock mappings" in str(error)
    else:
        raise AssertionError("misaligned source mappings must be rejected")


def test_first_minute_fusion_retains_all_rows_and_bounded_interpolation() -> None:
    frame_count = 1800
    native = _rows(
        set(range(frame_count)) - set(range(1584, 1589)),
        frame_count=frame_count,
    )
    boxmot = _rows(
        set(range(frame_count)) - set(range(1584, 1589)),
        frame_count=frame_count,
    )

    fused = fuse_person_boxes(native, boxmot, frame_count=frame_count)

    assert len(fused.provenance) == frame_count
    assert all(item.source == "interpolated" for item in fused.provenance[1584:1589])
    assert all(item.residual_gap_length == 5 for item in fused.provenance[1584:1589])
