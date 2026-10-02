"""Paired coverage and sign-safe motion calculations for the full-video comparison."""

import numpy as np
import pytest

from battle.pipette_video_metrics import angle_degrees, motion, paired_residuals


def geometry(point, direction=(1, 0, 0), tip=1):
    direction = np.array(direction, dtype=float)
    point = np.array(point, dtype=float)
    return {
        "line": {
            "point": point.tolist(),
            "direction": direction.tolist(),
            "endpoints": [list(point - direction), list(point + direction)],
        },
        "direction": {"tip_end": tip},
    }


def test_svd_sign_change_is_not_axis_motion_or_a_physical_direction_flip():
    assert angle_degrees(np.array([1, 0, 0]), np.array([-1, 0, 0]), unoriented=True) == 0
    result = motion(geometry([0, 0, 0], (-1, 0, 0), tip=0), geometry([0, 0, 0]), 1)
    assert result == {"midpoint_cm_s": 0, "axis_deg_s": 0, "directed_deg_s": 0}


def test_actual_reversed_arrow_is_reported_and_elapsed_time_is_used():
    result = motion(geometry([0, 3, 4], tip=0), geometry([0, 0, 0]), 2)
    assert result["midpoint_cm_s"] == 2.5
    assert result["axis_deg_s"] == 0
    assert result["directed_deg_s"] == 90


def test_unresolved_direction_and_missing_fit_are_not_imputed():
    result = motion(geometry([0, 0, 0], tip=None), geometry([0, 0, 0]), 1)
    assert result["directed_deg_s"] is None
    absent = {"line": None, "direction": {"tip_end": None}}
    assert all(value is None for value in motion(absent, geometry([0, 0, 0]), 1).values())


def test_per_second_motion_uses_native_elapsed_time():
    elapsed = 30 / (30000 / 1001)
    result = motion(geometry([2, 0, 0]), geometry([0, 0, 0]), elapsed)
    assert result["midpoint_cm_s"] == pytest.approx(2 / 1.001)


def test_display_does_not_invent_a_frame_at_the_duration_boundary():
    from battle.pipette_video_review import display_frames

    fps = 30000 / 1001
    assert display_frames(30, fps) == {0: 0}
    assert display_frames(31, fps) == {0: 0, 30: 1}
    assert len(display_frames(600, fps)) == 20
    full = display_frames(8492, fps)
    assert len(full) == 284
    assert max(full) == 8482


def test_correction_flags_cover_the_lagged_window_and_ignore_seed_as_a_step():
    from battle.pipette_video_review import correction_counts

    fps = 30000 / 1001
    schedule = {"T1": [0], "T2": [0, 100]}
    initial = correction_counts(0, fps, 8492, schedule)
    assert initial["correction_camera_events"] == 2
    assert initial["native_frames_with_correction"] == 1
    assert initial["step_native_intervals_crossing_correction"] == 0
    assert initial["1s_native_intervals_crossing_correction"] == 0
    during = correction_counts(3, fps, 8492, schedule)
    after = correction_counts(4, fps, 8492, schedule)
    assert during["correction_camera_events"] == 1
    assert during["1s_native_intervals_crossing_correction"] == 20
    assert after["correction_camera_events"] == 0
    assert after["1s_native_intervals_crossing_correction"] == 10
    assert correction_counts(5, fps, 8492, schedule)["1s_native_intervals_crossing_correction"] == 0


def test_residual_comparison_matches_views_and_preserves_unavailable_cells():
    def row(view, value):
        return {"view": view, "fitted": value is not None, "perpendicular_px": value}

    baseline = {"loo": [row("T1", 5), row("T2", None), row("T3", 1)]}
    revised = {"loo": [row("T1", 3), row("T2", 0.1), row("T4", 0.1)]}
    result = paired_residuals(baseline, revised)
    assert result == {"views": ["T1"], "baseline_px": [5], "revised_px": [3], "delta_px": [-2]}


def test_aggregate_preserves_partial_final_second_and_missing_measurements(tmp_path, monkeypatch):
    import json

    from battle import pipette_video_metrics as metrics
    from battle.pipette_line_review import VARIANTS

    fps = 30000 / 1001
    view = {"frame_count": 31, "size_wh": [1920, 1080], "aliases": {v: "g0" for v in VARIANTS}}
    (tmp_path / "config.json").write_text(json.dumps({"native_fps": fps, "views": {"T1": view}}))
    (tmp_path / "observations").mkdir()
    observation = {"axis": {"axis_px": None}, "active": False, "centroid_px": None}
    (tmp_path / "observations/T1.jsonl").write_text(
        "".join(
            json.dumps({"raw_frame": f, "groups": {"g0": observation}}) + "\n" for f in range(31)
        )
    )
    monkeypatch.setattr(metrics, "load_rig", lambda _: None)
    monkeypatch.setattr(metrics, "cameras_at_raw_frame", lambda *_: {})
    metrics.aggregate(tmp_path)
    rows = json.loads((tmp_path / "per-second.json").read_text())
    baseline = [x for x in rows if x["variant"] == VARIANTS[0]]
    assert [x["native_frames"] for x in baseline] == [30, 1]
    assert [x["second"] for x in baseline] == [0, 1]
    assert all(x["median_loo_px"] is None for x in rows)
    assert all(x["1s_midpoint_cm_s_median"] is None for x in rows)
    assert all(x["1s_midpoint_cm_s_samples"] == 0 for x in rows)
    assert baseline[0]["fit_fraction_change_from_previous_second"] is None
    assert baseline[1]["fit_fraction_change_from_previous_second"] == 0
    assert len((tmp_path / "geometry.jsonl").read_text().splitlines()) == 31
    assert len((tmp_path / "per-second.csv").read_text().splitlines()) == 7
