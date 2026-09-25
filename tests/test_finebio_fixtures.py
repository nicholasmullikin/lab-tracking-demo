"""The committed FineBio preflight fixtures load, and reproduce the preflight's rig numbers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from finebio_fixtures import FIXTURE_DIR, load_preflight_fixtures

from battle.finebio_cameras import Camera
from battle.multiview_geometry import dlt_triangulate

ROOT = Path(__file__).resolve().parents[1]
FIXED_VIEWS = ("T1", "T2", "T3", "T4", "T5")
FIXTURE_BYTES_CAP = 5 * 1024 * 1024


@pytest.fixture(scope="module")
def fixtures():
    return load_preflight_fixtures()


def _centre(box) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


def _triangulate(cams: list[Camera], pixels: list[np.ndarray]) -> np.ndarray:
    pts = np.stack([c.undistort(p).reshape(2) for c, p in zip(cams, pixels)])[:, None, :]
    return dlt_triangulate(pts, np.stack([c.projection for c in cams]))[0]


def _top(rows, cls: str, min_score: float):
    cands = [r for r in rows if r.object_class == cls and (r.detector_score or 0) >= min_score]
    return max(cands, key=lambda r: r.detector_score) if cands else None


def test_fixture_directory_is_small_and_numeric_only() -> None:
    files = sorted(p.name for p in FIXTURE_DIR.iterdir())
    assert files == [
        "README.md",
        "cameras.json",
        "fpv_poses.json",
        "observations.jsonl",
        "rig_reference.json",
    ]
    assert sum(p.stat().st_size for p in FIXTURE_DIR.iterdir()) < FIXTURE_BYTES_CAP
    assert (FIXTURE_DIR / "cameras.json").read_bytes() == (
        ROOT / "configs/finebio/cameras/P03_01_01.json"
    ).read_bytes()


def test_observations_match_the_reference_counts_and_frames(fixtures) -> None:
    reference = fixtures.rig_reference["observations"]
    assert len(fixtures.observations) == reference["rows"] == 20209
    by_source = {
        source: len(fixtures.rows(source=source))
        for source in ("detector", "sam3_decode", "sam3_video")
    }
    assert by_source == reference["by_source"]
    assert by_source["detector"] == 15695 and by_source["sam3_video"] == 4462
    assert {r.view for r in fixtures.observations} == {"fpv", *FIXED_VIEWS}
    detector_frames = {r.frame_index for r in fixtures.rows(source="detector")}
    assert sorted(detector_frames) == fixtures.rig_reference["frames"]["all"]
    assert len(detector_frames) == 78 and len(fixtures.consecutive_frames) == 61
    video_frames = {r.frame_index for r in fixtures.rows(source="sam3_video")}
    assert min(video_frames) == 1799 and max(video_frames) == 2097
    assert all((r.detector_score or 0) >= 0.3 for r in fixtures.rows(source="detector"))
    # The T2 50ml tube lost 14 frames behind the arm: those rows are absent, not empty.
    tube = fixtures.rows(view="T2", source="sam3_video", object_class="50ml_tube")
    assert {r.frame_index for r in tube}.isdisjoint(range(1912, 1926)) and len(tube) == 285
    assert fixtures.rig_reference["observations"]["sam3"]["track"]["T2/50ml_tube"][
        "lost_frames_omitted"
    ] == list(range(1912, 1926))


def test_sam3_rows_carry_a_mask_centroid_at_the_bbox_centre(fixtures) -> None:
    for row in fixtures.rows(source="sam3_video")[:200]:
        assert row.mask_bbox_px is not None and row.mask_centroid_px is not None
        assert row.mask_centroid_px == pytest.approx(_centre(row.mask_bbox_px), abs=0.06)
        assert row.point_px == row.mask_centroid_px
        assert row.mask_area_px > 0 and row.sam3_object_score is not None
        assert row.provenance["encoder_side"] == 1280
    frame = fixtures.rows(view="fpv", frame_index=1799, source="sam3_video")
    assert {r.object_class for r in frame} == {
        "cell_culture_plate",
        "blue_pipette",
        "centrifuge",
        "50ml_tube",
    }
    plate = next(r for r in frame if r.object_class == "cell_culture_plate")
    assert plate.box_xyxy_px is not None and plate.detector_score == pytest.approx(0.4453, 1e-3)
    assert plate.provenance["detector_box_iou"] == pytest.approx(0.946, abs=1e-3)


def test_fpv_pose_validity_is_recorded_per_frame(fixtures) -> None:
    invalid = {r.frame_index for r in fixtures.rows(view="fpv") if not r.pose_valid}
    valid = {r.frame_index for r in fixtures.rows(view="fpv") if r.pose_valid}
    assert invalid.isdisjoint(valid)
    assert all(fixtures.fpv_camera(f) is None for f in invalid)
    assert all(fixtures.fpv_camera(f) is not None for f in valid)
    cam = fixtures.fpv_camera(1798)
    assert cam is not None
    assert np.allclose(cam.centre, [5.96, 36.31, -35.54], atol=0.01)  # rig.json fpv_centre_cm
    assert fixtures.fpv_camera(0) is None


def test_static_objects_triangulate_to_the_preflight_points(fixtures) -> None:
    cams = fixtures.fixed_cameras()
    frames = fixtures.rig_reference["frames"]["spaced"]
    by_frame = fixtures.by_frame(source="detector")
    reproduced = 0
    for static in fixtures.rig_reference["static"]:
        cls, views = static["class"], static["views"]
        per_view = {}
        for view in views:
            centres = [
                _centre(_top(by_frame[f].get(view, ()), cls, 0.5).box_xyxy_px)
                for f in frames
                if _top(by_frame[f].get(view, ()), cls, 0.5)
            ]
            assert len(centres) >= max(3, len(frames) // 3), (cls, view)
            per_view[view] = np.median(np.stack(centres), axis=0)
        point = _triangulate([cams[v] for v in views], [per_view[v] for v in views])
        assert np.allclose(point, static["point_cm"], atol=0.05), cls
        assert -point[2] == pytest.approx(static["height_cm"], abs=0.05)
        for view in views:
            others = [u for u in views if u != view]
            loo = _triangulate([cams[u] for u in others], [per_view[u] for u in others])
            residual = float(np.linalg.norm(cams[view].project(loo)[0] - per_view[view]))
            assert residual == pytest.approx(static["loo_px"][view], abs=0.5), (cls, view)
        reproduced += 1
    assert reproduced == len(fixtures.rig_reference["static"]) >= 11
    summary = fixtures.rig_reference["static_loo_px"]
    assert summary["median"] == pytest.approx(10.4, abs=0.1)
    assert summary["p90"] == pytest.approx(25.3, abs=0.1)


def test_plate_hand_off_into_the_fpv_reproduces_the_preflight(fixtures) -> None:
    cams = fixtures.fixed_cameras()
    by_frame = fixtures.by_frame(source="detector")
    residuals, inside = [], []
    for frame in fixtures.consecutive_frames:
        fpv = fixtures.fpv_camera(frame)
        fpv_box = _top(by_frame[frame].get("fpv", ()), "cell_culture_plate", 0.4)
        views = [
            v for v in FIXED_VIEWS if _top(by_frame[frame].get(v, ()), "cell_culture_plate", 0.4)
        ]
        if fpv is None or fpv_box is None or len(views) < 3:
            continue
        point = _triangulate(
            [cams[v] for v in views],
            [
                _centre(_top(by_frame[frame][v], "cell_culture_plate", 0.4).box_xyxy_px)
                for v in views
            ],
        )
        projected = fpv.project(point)[0]
        residuals.append(float(np.linalg.norm(projected - _centre(fpv_box.box_xyxy_px))))
        x0, y0, x1, y1 = fpv_box.box_xyxy_px
        inside.append(x0 <= projected[0] <= x1 and y0 <= projected[1] <= y1)
    reference = fixtures.rig_reference["fpv"]["plate_fixed_to_fpv_px"]
    assert len(residuals) == reference["n"] == 45
    assert np.median(residuals) == pytest.approx(reference["median"], abs=0.5)
    assert np.percentile(residuals, 90) == pytest.approx(reference["p90"], abs=0.5)
    assert np.mean(inside) == reference["inside_box_fraction"] == 1.0


def test_reference_carries_the_preflight_headline_numbers(fixtures) -> None:
    reference = fixtures.rig_reference
    assert reference["day"]["chosen"] == "221013" and reference["camera_permutation_ok"]
    assert {v: d["camera_id"] for v, d in reference["views"].items()} == {
        "T1": 1,
        "T2": 2,
        "T3": 3,
        "T4": 4,
        "T5": 6,
    }
    assert reference["views"]["T5"]["shipped_median_corner_rms_px"] == pytest.approx(93.7, 0.01)
    assert reference["views"]["T5"]["pnp_vs_shipped_cm"] == pytest.approx(6.43, abs=0.01)
    assert reference["views"]["T5"]["provenance"] == "marker_pnp"
    assert reference["hands"]["left_hand"]["frames_with_3_views"] == 61
    assert reference["hands"]["right_hand"]["frames_with_3_views"] == 14
    assert reference["clock"]["T3"]["right_hand"]["best_offset"] == 1
    assert reference["clock"]["T5"]["right_hand"]["best_offset"] == -1
    plate = reference["loo_moving"]["cell_culture_plate"]
    assert all(plate[v]["inside_fraction"] >= 0.98 for v in FIXED_VIEWS)
    assert reference["fpv_pose"]["corner_rms_px"]["median"] == pytest.approx(0.93, abs=0.01)
    assert reference["fpv_pose"]["frames_checked"] == 196
