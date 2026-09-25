"""The thin slice (p0-slice) on the preflight fixtures: geometry helpers, per-frame
triangulation, the summary numbers against the preflight, and the CLI end to end."""

from __future__ import annotations

import json

import numpy as np
import pytest
from finebio_fixtures import FIXTURE_DIR, load_preflight_fixtures

from battle.finebio_slice import (
    HANDOFF_CLASS,
    STATIC_CLASSES,
    SlicePoint,
    SliceSettings,
    consecutive_frames,
    leave_one_out_residuals,
    main,
    parse_frames,
    reprojection_residuals,
    run_slice,
    summarise,
    summary_markdown,
    top_per_class,
    triangulate_pixels,
)
from battle.multiview_geometry import dlt_triangulate
from battle.multiview_schemas import read_jsonl

WINDOW = list(range(1798, 1858))


@pytest.fixture(scope="module")
def fixtures():
    return load_preflight_fixtures()


@pytest.fixture(scope="module")
def slice_result(fixtures):
    return run_slice(fixtures.observations, fixtures.fixed_cameras(), fixtures.fpv_camera, WINDOW)


def test_weighted_triangulation_matches_dlt_at_equal_weights(fixtures) -> None:
    cams = fixtures.fixed_cameras()
    point = np.array([12.0, -8.0, -4.0])
    views = ["T1", "T2", "T3", "T4", "T5"]
    pixels = [cams[v].project(point)[0] for v in views]
    plain = triangulate_pixels([cams[v] for v in views], pixels)
    weighted = triangulate_pixels([cams[v] for v in views], pixels, weights=[1.0] * 5)
    reference = dlt_triangulate(
        np.stack([cams[v].undistort(p).reshape(2) for v, p in zip(views, pixels)])[:, None, :],
        np.stack([cams[v].projection for v in views]),
    )[0]
    assert np.allclose(plain, point, atol=1e-6)
    assert np.allclose(weighted, reference, atol=1e-6)
    residuals = reprojection_residuals(cams, dict(zip(views, pixels)), plain)
    assert max(residuals.values()) < 1e-4
    loo = leave_one_out_residuals(cams, dict(zip(views, pixels)), views)
    assert set(loo) == set(views) and max(loo.values()) < 1e-4


def test_frame_parsing() -> None:
    available = [1, 2, 3, 10, 30, 31]
    assert consecutive_frames(available) == [1, 2, 3, 30, 31]
    assert parse_frames(None, available) == [1, 2, 3, 30, 31]
    assert parse_frames("all", available) == available
    assert parse_frames("2:3", available) == [2, 3]
    assert parse_frames("3-30", available) == [3, 10, 30]
    assert parse_frames("1,31,99", available) == [1, 31]


def test_top_per_class_takes_the_top_box_and_the_sam3_row(fixtures) -> None:
    rows = fixtures.rows(view="T2", frame_index=1800)
    detector = top_per_class(rows, source_set="detector", min_detector_score=0.3)
    plate = detector["cell_culture_plate"]
    assert plate.source == "detector"
    assert plate.detector_score == max(
        r.detector_score for r in rows if r.object_class == "cell_culture_plate"
    )
    sam3 = top_per_class(rows, source_set="sam3", min_detector_score=0.3)
    assert sam3["cell_culture_plate"].source == "sam3_video"
    assert set(sam3) == {"cell_culture_plate", "blue_pipette", "centrifuge", "50ml_tube"}
    assert not top_per_class(rows, source_set="detector", min_detector_score=1.01)


def test_plate_triangulates_on_every_window_frame_with_the_preflight_residuals(
    slice_result, fixtures
) -> None:
    plate = [
        p
        for p in slice_result.points
        if p.object_class == HANDOFF_CLASS and p.source_set == "detector"
    ]
    assert [p.frame_index for p in plate] == WINDOW
    assert all(len(p.views_used) >= 3 and not p.fpv_used for p in plate)
    reference = fixtures.rig_reference["loo_moving"][HANDOFF_CLASS]
    for view in ("T1", "T2", "T3", "T4", "T5"):
        loo = [p.loo_px[view] for p in plate if view in p.loo_px]
        assert np.median(loo) == pytest.approx(reference[view]["median_px"], abs=1.5), view
    handoff = [p.fpv_residual_px for p in plate if p.fpv_residual_px is not None]
    assert len(handoff) == 60
    assert np.median(handoff) == pytest.approx(
        fixtures.rig_reference["fpv"]["plate_fixed_to_fpv_px"]["median"], abs=1.0
    )
    assert all(p.fpv_inside_box for p in plate if p.fpv_inside_box is not None)
    assert np.median([p.height_cm for p in plate]) == pytest.approx(3.25, abs=0.3)


def test_static_classes_land_on_the_preflight_points(slice_result, fixtures) -> None:
    summary = summarise(slice_result, fixtures.rig_reference)
    comparison = summary["static_vs_preflight"]["per_class"]
    assert set(comparison) == set(STATIC_CLASSES)
    for cls, entry in comparison.items():
        assert entry["frames_triangulated"] == 60, cls
        assert entry["median_point_to_reference_cm"] < 0.05, cls
        assert entry["per_frame_distance_cm"]["median"] < 0.1, cls
    assert summary["static_vs_preflight"]["median_point_to_reference_cm"]["p90"] < 0.05
    detector = summary["classes"]["detector"]
    # The top-box-per-class rule collapses several tubes onto different instances per view.
    assert detector["50ml_tube"]["one_object_under_top_box_rule"] is False
    assert detector["micro_tube"]["one_object_under_top_box_rule"] is False
    assert detector[HANDOFF_CLASS]["one_object_under_top_box_rule"] is True
    assert detector["left_hand"]["frames_triangulated"] == 60
    markdown = summary_markdown(summary, "P03_01_01")
    assert "| cell_culture_plate | 60 |" in markdown and "one object" in markdown


def test_sam3_set_uses_mask_centroids_and_the_fpv_birth_rule(slice_result) -> None:
    sam3 = slice_result.by_class("sam3")
    plate = sam3[HANDOFF_CLASS]
    assert len(plate) == 60
    assert plate[0].frame_index == 1798 and len(plate[0].views_used) == 5  # sam3_decode frame
    assert all(set(p.views_used) == {"T2", "T4", "T5"} for p in plate[1:])
    pipette = sam3["blue_pipette"]
    # Seeded in T2 and T5 only: two fixed views plus the fpv when its pose is valid.
    assert all(p.fpv_used and set(p.views_used) == {"T2", "T5", "fpv"} for p in pipette[1:])
    assert all(p.fpv_residual_px is None for p in pipette[1:])


def test_settings_change_the_birth_rule(fixtures) -> None:
    strict = run_slice(
        fixtures.observations,
        fixtures.fixed_cameras(),
        fixtures.fpv_camera,
        [1800],
        SliceSettings(min_fixed_views=5, fixed_views_with_fpv=5, source_sets=("detector",)),
    )
    assert all(len(p.views_used) == 5 for p in strict.points)
    loose = run_slice(
        fixtures.observations,
        fixtures.fixed_cameras(),
        fixtures.fpv_camera,
        [1800],
        SliceSettings(min_fixed_views=3, source_sets=("detector",)),
    )
    assert len(loose.points) > len(strict.points)


def test_cli_writes_points_summary_and_recording(tmp_path) -> None:
    out = tmp_path / "slice"
    assert (
        main(
            [
                "--fixtures",
                str(FIXTURE_DIR),
                "--output",
                str(out),
                "--frames",
                "1798:3",
                "--image-every",
                "1",
            ]
        )
        == 0
    )
    points = list(read_jsonl(out / "points3d.jsonl", SlicePoint))
    assert {p.frame_index for p in points} == {1798, 1799, 1800}
    summary = json.loads((out / "summary.json").read_text())
    assert summary["trial"] == "P03_01_01" and summary["points_written"] == len(points)
    assert "static_vs_preflight" in summary
    text = (out / "summary.md").read_text()
    assert "What the floor demo still needs" in text
    assert (out / "slice.rrd").stat().st_size > 10_000
