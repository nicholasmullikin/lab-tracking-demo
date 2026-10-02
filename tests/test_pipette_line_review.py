"""Frame/selection adapters and conservative direction for the offline comparison."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from battle.finebio_orientation import CUE_COLOUR, CUE_TAPER, CueVote
from battle.pipette_line_review import (
    CONDITION,
    cameras_at_raw_frame,
    choose_mask,
    clip_image_segment,
    comparison_summary,
    conservative_direction,
    draw_overlay,
    endpoint_correspondence,
)


def test_extreme_projection_is_clipped_without_moving_endpoint_labels_to_the_border():
    projected = np.array([[-1e12, 50], [1e12, 50]])
    np.testing.assert_allclose(
        clip_image_segment(projected, (100, 100)), [[0, 50], [99, 50]], atol=0.01
    )
    original = np.zeros((100, 100, 3), np.uint8)
    result = draw_overlay(
        original, np.zeros((100, 100), bool), {"axis": {"axis_px": None}}, projected, 1
    )
    np.testing.assert_array_equal(result[50, 50], [60, 255, 60])
    assert not np.any(np.all(result == 255, axis=2))  # No false W0/W1 markers at clip boundaries.
    np.testing.assert_array_equal(projected, [[-1e12, 50], [1e12, 50]])


def test_off_image_and_nonfinite_segments_do_not_become_false_overlays():
    assert clip_image_segment([[1e12, 1e12], [2e12, 2e12]], (100, 100)) is None
    assert clip_image_segment([[None, 20], [10, 20]], (100, 100)) is None


def test_substitutions_use_correct_selection_policy_and_preserve_other_views():
    labels = {
        "frames": {
            "0": {
                "views": {
                    v: {
                        "mask_uri": f"0/final/{v}.png",
                        "selection": {"candidate_index": 2},
                        "quality": "needs-review",
                        "review_note": "earlier",
                    }
                    for v in ("T1", "T2")
                }
            }
        }
    }
    ratings = {
        "cases": {
            "f000-T1": {
                "conditions": {
                    CONDITION: {"model_top_candidate_index": 0, "selected_candidate_index": 1}
                }
            }
        }
    }
    base = Path("snapshot")
    old, provenance = choose_mask(base, labels, ratings, 0, "T1", "earlier-selected")
    assert old == base / "every100-review/0/final/T1.png"
    assert not provenance["substituted"]
    top, _ = choose_mask(base, labels, ratings, 0, "T1", "recipe-model-top")
    best, _ = choose_mask(base, labels, ratings, 0, "T1", "recipe-visual-best")
    assert top.name.endswith("candidate-00.png")
    assert best.name.endswith("candidate-01.png")
    unchanged, provenance = choose_mask(base, labels, ratings, 0, "T2", "recipe-model-top")
    assert unchanged == base / "every100-review/0/final/T2.png"
    assert not provenance["substituted"]


@pytest.mark.parametrize("frame", [0, 100, 600])
def test_raw_frame_is_passed_without_clip_offset(frame):
    seen = []
    fixed = SimpleNamespace(size=(1920, 1080))

    def at(index):
        seen.append(index)
        return {"T1": fixed}  # Missing FPV pose does not invent a camera.

    cams = cameras_at_raw_frame(
        SimpleNamespace(at=at), frame, {"T1": (1920, 1080), "fpv": (1920, 1440)}
    )
    assert seen == [frame]
    assert "fpv" not in cams


def test_camera_resolution_mismatch_is_rejected():
    rig = SimpleNamespace(at=lambda _: {"T1": SimpleNamespace(size=(960, 540))})
    with pytest.raises(ValueError, match="camera"):
        cameras_at_raw_frame(rig, 0, {"T1": (1920, 1080)})


def test_endpoint_mapping_tracks_reversed_world_order_and_abstains_on_ambiguity():
    local = np.array([[10, 20], [200, 60]])
    assert endpoint_correspondence(local, local + [3, 2]) == (0, 1)
    assert endpoint_correspondence(local, local[::-1] + [3, 2]) == (1, 0)
    assert endpoint_correspondence(local, np.array([[0, 0], [1, 1]])) is None
    assert (
        endpoint_correspondence(np.array([[-10, 0], [10, 0]]), np.array([[0, -10], [0, 10]]))
        is None
    )
    assert endpoint_correspondence(local, np.full((2, 2), np.nan)) is None


def test_direction_abstains_on_conflicting_cues_even_if_one_is_stronger():
    result = conservative_direction([CueVote(CUE_COLOUR, 0, 1), CueVote(CUE_TAPER, 1, 0.2)])
    assert result["tip_end"] is None
    assert result["reason"] == "colour_taper_disagreement"


def test_direction_requires_evidence_and_threshold_but_accepts_agreement():
    assert conservative_direction([])["tip_end"] is None
    assert conservative_direction([CueVote(CUE_COLOUR, 1, 0.1)])["tip_end"] is None
    result = conservative_direction([CueVote(CUE_COLOUR, 1, 0.5), CueVote(CUE_TAPER, 1, 0.4)])
    assert result["tip_end"] == 1
    assert result["reason"] is None


def test_paired_metrics_keep_failures_in_coverage_and_exclude_unmatched_residuals():
    def frame(rows, fitted=True):
        return {
            "loo": rows,
            "line": {} if fitted else None,
            "direction": {"tip_end": None},
            "views": {"T1": {"axis": {"axis_px": None}}},
        }

    def row(view, value):
        return {
            "view": view,
            "fitted": value is not None,
            "perpendicular_px": value,
            "angle_deg": 2 if value is not None else None,
        }

    report = {
        "variants": {
            "earlier-selected": {
                "0": frame([row("T1", 2), row("T2", 8)]),
                "100": frame([row("T1", 20)]),
            },
            "recipe-model-top": {
                "0": frame([row("T1", 1), row("T3", 0.1)]),
                "100": frame([row("T1", None)], fitted=False),
            },
        }
    }
    result = comparison_summary(report)[1]
    assert result["fits"] == 1
    assert result["paired_cells"] == 1
    assert result["loo_cells"] == 2
    assert result["paired_median_px"] == 1
    assert result["median_paired_change_px"] == -1
    assert result["improved_gt_0p1_px"] == 1
