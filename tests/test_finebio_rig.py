"""``battle-finebio-rig`` (p1-rig): the gate formulas on synthetic inputs, the rig run on the
committed preflight fixtures (default tier), and the regression on the preflight detections
with the camera-6 negative control (`real_data`)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
from conftest import require_artifact
from finebio_fixtures import load_preflight_fixtures

from battle import finebio_rig as fr
from battle.finebio_cameras import POSES, read_camera_config

ROOT = Path(__file__).resolve().parents[1]
P03_CONFIG = ROOT / "configs/finebio/cameras/P03_01_01.json"
PREFLIGHT_DETECTIONS = ROOT / "runs/preflight-finebio-20260924/detections"


def _static_rows(loo_values: list[float]) -> list[dict]:
    return [
        {"class": f"c{i}", "loo_px": {"T1": v}, "residual_px": {}} for i, v in enumerate(loo_values)
    ]


def test_frame_helpers() -> None:
    assert fr.parse_frames("3,1-2,10:16:5") == [1, 2, 3, 10, 15]
    assert fr.consecutive_frames([1, 2, 3, 10, 20, 21]) == [1, 2, 3, 20, 21]
    assert fr.subsample(list(range(100)), 0) == list(range(100))
    picked = fr.subsample(list(range(100)), 5)
    assert picked[0] == 0 and picked[-1] == 99 and len(picked) == 5
    assert fr.frames_in_detections(
        {"T1": {1: [], 2: []}, "T2": {2: []}, "T3": {2: []}}, ("T1", "T2", "T3")
    ) == [2]
    assert (
        fr.top_box([{"class": "a", "score": 0.4}, {"class": "a", "score": 0.9}], "a", 0.5)["score"]
        == 0.9
    )
    assert fr.top_box([{"class": "a", "score": 0.4}], "a", 0.5) is None
    assert np.allclose(fr.centre([0, 0, 10, 20]), [5, 10])
    assert fr.inside([0, 0, 10, 20], np.array([5.0, 10.0])) and not fr.inside(
        [0, 0, 10, 20], np.array([11.0, 1.0])
    )


def test_gates_are_the_documented_formulas_with_floor_and_cap() -> None:
    clock = {"T1": {"x": {"best_offset": 2, "informative": True}}, "T2": {}}
    by_view = {"T1": [10.0] * 80, "T2": [10.0] * 10 + [40.0] * 10, "fpv": [100.0] * 5}

    gates = fr.compute_gates(_static_rows([8.0, 10.0, 12.0]), by_view, clock)

    assert gates["inputs"]["static_loo_median_px"] == 10.0
    assert gates["association_px"] == pytest.approx(30.0)
    # T2's p90 (40) sets the hand-off, not the pooled p90; the fpv has too few residuals.
    assert set(gates["inputs"]["moving_loo_p90_px_by_view"]) == {"T1", "T2"}
    assert gates["handoff_px"] == pytest.approx(40.0)
    assert gates["inputs"]["moving_loo_pooled_p90_px"] < 40.0
    assert gates["clock_offset_frames"]["T1"] == {
        "offset_frames": 2,
        "uncertainty_frames": 1,
        "significant": True,
        "informative_scans": 1,
        "informative_best_offsets": [2],
        "basis": "median best offset over informative scans",
    }
    assert gates["clock_offset_frames"]["T2"]["offset_frames"] == 0
    assert not gates["clock_offset_frames"]["T2"]["significant"]
    assert gates["birth_min_fixed_views"] == 3 and gates["birth_fixed_views_with_fpv"] == 2
    assert "clamp(3 * static_loo_median_px, floor, cap)" == gates["formula"]["association_px"]

    low = fr.compute_gates(_static_rows([2.0]), {"T1": [3.0] * 20}, {})
    assert low["association_px"] == 15.0 and low["handoff_px"] == 15.0
    high = fr.compute_gates(_static_rows([50.0]), {"T1": [300.0] * 20}, {})
    assert high["association_px"] == 80.0 and high["handoff_px"] == 80.0
    half = fr.compute_gates(_static_rows([50.0]), {"T1": [300.0] * 20}, {}, image_width_px=960)
    assert half["floor_px"] == 7.5 and half["cap_px"] == 40.0 and half["handoff_px"] == 40.0
    empty = fr.compute_gates([], {}, {})
    assert empty["association_px"] is None and empty["handoff_px"] is None


@pytest.fixture(scope="module")
def fixture_rig() -> tuple[dict, dict]:
    fixtures = load_preflight_fixtures()
    dets: fr.Detections = defaultdict(dict)
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in fixtures.rows(source="detector"):
        grouped[(row.view, row.frame_index)].append(
            {
                "class": row.object_class,
                "score": row.detector_score,
                "box_xyxy_px": list(row.box_xyxy_px),
            }
        )
    for (view, frame), boxes in grouped.items():
        dets[view][frame] = boxes
    frames = list(fixtures.rig_reference["frames"]["all"])
    report = fr.run_rig(
        fixtures.cameras, dict(dets), frames, fixtures.fpv_camera, frames_source="fixtures"
    )
    return report, fixtures.rig_reference


def _assert_reproduces_reference(report: dict, reference: dict) -> None:
    assert report["day"] == reference["day"]["chosen"]
    assert report["camera_ids"] == {v: r["camera_id"] for v, r in reference["views"].items()}
    assert report["camera_provenance"] == {
        v: r["provenance"] for v, r in reference["views"].items()
    }
    by_class = {row["class"]: row for row in report["static"]}
    assert list(by_class) == [row["class"] for row in reference["static"]]
    for ref in reference["static"]:
        row = by_class[ref["class"]]
        assert row["views"] == ref["views"], ref["class"]
        assert np.allclose(row["point_cm"], ref["point_cm"], atol=0.05), ref["class"]
        for view, value in ref["loo_px"].items():
            assert row["loo_px"][view] == pytest.approx(value, abs=0.5), (ref["class"], view)
        for view, value in ref["residual_px"].items():
            assert row["residual_px"][view] == pytest.approx(value, abs=0.5), (ref["class"], view)
    loo = report["static_loo_px"]
    assert loo["n"] == reference["static_loo_px"]["n"]
    assert loo["median"] == pytest.approx(reference["static_loo_px"]["median"], abs=0.5)
    assert loo["p90"] == pytest.approx(reference["static_loo_px"]["p90"], abs=0.5)
    for cls, ref in reference["hands"].items():
        got = report["hands_summary"][cls]
        assert got["frames_with_3_views"] == ref["frames_with_3_views"]
        assert got["residual_px"]["n"] == ref["residual_px"]["n"]
        assert got["residual_px"]["median"] == pytest.approx(ref["residual_px"]["median"], abs=0.5)
        assert got["height_cm_median"] == pytest.approx(ref["height_cm_median"], abs=0.05)
    for view, scans in reference["clock"].items():
        for cls, ref in scans.items():
            assert report["clock"][view][cls]["best_offset"] == ref["best_offset"], (view, cls)
            assert report["clock"][view][cls]["residual_px_at_best"] == pytest.approx(
                ref["residual_px_at_best"], abs=0.5
            )
    for cls, views in reference["loo_moving"].items():
        for view, ref in views.items():
            got = report["loo_moving"][cls][view]
            assert got["frames"] == ref["frames"], (cls, view)
            assert got["median_px"] == pytest.approx(ref["median_px"], abs=0.5), (cls, view)
            assert got["inside_fraction"] == pytest.approx(ref["inside_fraction"])
    plate = report["fpv_summary"]["plate_fixed_to_fpv_px"]
    ref_plate = reference["fpv"]["plate_fixed_to_fpv_px"]
    assert plate["n"] == ref_plate["n"]
    assert plate["median"] == pytest.approx(ref_plate["median"], abs=0.5)
    assert plate["p90"] == pytest.approx(ref_plate["p90"], abs=0.5)
    assert plate["inside_box_fraction"] == ref_plate["inside_box_fraction"]
    left = report["fpv_summary"]["left_hand_fixed_to_fpv_px"]
    assert left["n"] == reference["fpv"]["left_hand_fixed_to_fpv_px"]["n"]
    assert left["median"] == pytest.approx(
        reference["fpv"]["left_hand_fixed_to_fpv_px"]["median"], abs=0.5
    )


def test_rig_on_the_committed_fixtures_reproduces_the_preflight(fixture_rig) -> None:
    report, reference = fixture_rig

    _assert_reproduces_reference(report, reference)
    assert report["frames"]["count"] == 78 and report["frames"]["consecutive"] == 61
    # Heights are half the objects' physical heights: units are centimetres, z into the bench.
    heights = {row["class"]: row["height_cm"] for row in report["static"]}
    assert 3.5 < heights["centrifuge"] < 4.5 and 9 < heights["trash_can"] < 10.5
    assert all(-0.5 < h < 11 for h in heights.values())


def test_p03_gates_come_out_of_the_formulas_at_the_preflight_values(fixture_rig) -> None:
    report, _ = fixture_rig
    gates = report["gates"]

    assert gates["floor_px"] == 15.0 and gates["cap_px"] == 80.0
    assert gates["inputs"]["static_loo_median_px"] == pytest.approx(10.37, abs=0.1)
    assert gates["association_px"] == pytest.approx(3 * gates["inputs"]["static_loo_median_px"])
    assert 30 <= gates["association_px"] <= 32
    by_view = gates["inputs"]["moving_loo_p90_px_by_view"]
    assert set(by_view) == {"T1", "T2", "T3", "T4", "T5", "fpv"}
    assert max(by_view, key=by_view.get) == "fpv"
    assert gates["handoff_px"] == pytest.approx(max(by_view.values()))
    assert 50 <= gates["handoff_px"] <= 60
    assert 40 <= gates["inputs"]["moving_loo_pooled_p90_px"] <= 50
    offsets = {v: c["offset_frames"] for v, c in gates["clock_offset_frames"].items()}
    assert all(abs(o) <= 1 for o in offsets.values()) and offsets["T4"] == 1 and offsets["T5"] == -1
    assert not any(c["significant"] for c in gates["clock_offset_frames"].values())
    text = fr.rig_markdown(report)
    assert f"= **{gates['association_px']:.1f} px**" in text and "= **51.9 px**" in text
    assert "## Gates" in text and "per-view p90: T1" in text


@pytest.mark.real_data
def test_rig_on_the_preflight_detections_with_the_camera_6_negative_control() -> None:
    detections = require_artifact(PREFLIGHT_DETECTIONS)
    require_artifact(POSES / "first_person_camera_poses/P03_01_01.npz")
    require_artifact(POSES / "third_person_camera_poses/221013")
    config = read_camera_config(P03_CONFIG)
    dets = fr.load_detections(detections, (*fr.FINEBIO_FIXED_VIEWS, "fpv"))
    frames = sorted(json.loads((detections / "frames.json").read_text())["frames"])
    reference = json.loads(
        (ROOT / "tests/fixtures/finebio_preflight/rig_reference.json").read_text()
    )

    report = fr.run_rig(
        config,
        dets,
        frames,
        fr.fpv_source_from_data(config, "P03_01_01"),
        with_negative_control=True,
    )

    _assert_reproduces_reference(report, reference)
    control = report["negative_control"]
    assert list(control) == ["T5"]
    t5 = control["T5"]
    assert t5["marker_rms_px"]["marker_pnp"] < 1 and t5["marker_rms_px"]["shipped"] > 90
    assert t5["centre_cm"]["distance"] == pytest.approx(6.43, abs=0.05)
    assert t5["static_loo_px"]["median"]["marker_pnp"] < 10
    assert t5["static_loo_px"]["median"]["shipped"] > 80
    assert all(v > 80 for v in t5["static_loo_px"]["shipped"].values())
    assert t5["hands_px"]["marker_pnp"]["left_hand"]["median"] < 10
    assert t5["hands_px"]["shipped"]["left_hand"]["median"] > 40
    text = fr.rig_markdown(report)
    assert "## Negative control" in text and "0.71 vs 93.7 px" in text
