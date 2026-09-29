"""`battle-finebio-tipseg` on rendered masks and frames, synthetic cameras, no data tree."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from battle import finebio_tipseg as ts
from battle.finebio_cameras import Camera
from battle.muggled_worker import parse_box_stream

# --------------------------------------------------------------------------------------------
# a rendered pipette: a wide body along +x, a thin tip on the right, a short plunger stem on
# the left; the axis runs from the stem end (x=20) to the tip end (x=290)

BODY = (60, 230, 80, 120)  # x0, x1, y0, y1
TIP = (230, 290, 96, 104)  # 8 px wide, 60 px long
STEM = (20, 60, 94, 106)  # 12 px wide, 40 px long
AXIS = [[20.0, 100.0], [290.0, 100.0]]


def _mask(*, tip: bool = True, stem: bool = True) -> np.ndarray:
    mask = np.zeros((200, 320), dtype=bool)
    x0, x1, y0, y1 = BODY
    mask[y0:y1, x0:x1] = True
    if tip:
        x0, x1, y0, y1 = TIP
        mask[y0:y1, x0:x1] = True
    if stem:
        x0, x1, y0, y1 = STEM
        mask[y0:y1, x0:x1] = True
    return mask


def _row(**overrides) -> ts.PipetteRow:
    fields = dict(
        view="T4",
        frame=700,
        slot="blue_pipette#0",
        cls="blue_pipette",
        axis=AXIS,
        width_px=40.0,
        elongation=6.0,
        mask_bbox=[20.0, 80.0, 290.0, 120.0],
        centroid=[150.0, 100.0],
        box=None,
        detector_score=0.8,
        sam3_score=0.9,
        mask_uri=None,
        tip_side=1,
        tip_side_rule="tail",
        px_per_cm=10.0,
        body_end_px=[289.5, 100.0],
    )
    fields.update(overrides)
    return ts.PipetteRow(**fields)


def _estimate(method: str, row: ts.PipetteRow, present, end_px, confidence=0.8, junction=None):
    return ts.TipEstimate(
        method,
        row.view,
        row.frame,
        row.slot,
        row.cls,
        present,
        list(end_px) if present else None,
        junction,
        None,
        confidence,
        list(end_px),
        row.tip_side,
    )


# --------------------------------------------------------------------------------------------
# method B: the width profile and the junction


def test_width_profile_junction_finds_the_tip_on_a_rendered_body_plus_tip() -> None:
    profile = ts.width_profile(_mask(), AXIS)
    assert profile is not None
    assert profile.body_width == pytest.approx(40.0, abs=1.0)
    junction = ts.junction_from_profile(profile, 1, min_tail_px=20.0)
    assert junction.present
    # The tail is the 60 px tip; the 2 px bins land within one bin of it.
    assert junction.tail_px == pytest.approx(60.0, abs=2.0)
    assert junction.tail_width == pytest.approx(8.0, abs=1.0)
    assert junction.terminal_width == pytest.approx(8.0, abs=1.0)
    # From the stem end the tail is the 40 px stem (12 px wide, under 40% of 40).
    stem = ts.junction_from_profile(profile, 0, min_tail_px=20.0)
    assert stem.tail_px == pytest.approx(40.0, abs=2.0)


def test_width_profile_reports_no_tail_on_a_bare_body() -> None:
    profile = ts.width_profile(_mask(tip=False, stem=False), [[60.0, 100.0], [230.0, 100.0]])
    assert profile is not None
    junction = ts.junction_from_profile(profile, 1, min_tail_px=10.0)
    assert not junction.present
    assert junction.tail_px == 0.0


def test_width_profile_estimate_places_the_junction_inside_the_mask() -> None:
    row = _row()
    profile = ts.width_profile(_mask(), AXIS)
    estimate = ts.width_profile_estimate(row, profile, min_tail_px=20.0)
    assert estimate.method == "B" and estimate.tip_present is True
    assert estimate.end_px == [289.5, 100.0]
    assert estimate.junction_px is not None
    assert estimate.junction_px[0] == pytest.approx(230.0, abs=3.0)
    assert estimate.tip_length_px == pytest.approx(60.0, abs=2.0)


def test_tip_side_prefers_the_long_thin_tail_over_the_thin_stem() -> None:
    row = _row(tip_side=None)
    profile = ts.width_profile(_mask(), AXIS)
    side, rule = ts.choose_tip_side(row, profile, hands=[], px_per_cm=10.0)
    assert (side, rule) == (1, "tail")
    # With no tail difference the hand box decides: a hand on the right puts the tip left.
    bare = ts.width_profile(_mask(tip=False, stem=False), [[60.0, 100.0], [230.0, 100.0]])
    hand = [{"box_xyxy_px": [200.0, 60.0, 260.0, 140.0]}]
    side, rule = ts.choose_tip_side(
        _row(axis=[[60.0, 100.0], [230.0, 100.0]]), bare, hand, px_per_cm=10.0
    )
    assert (side, rule) == (0, "hand")
    side, rule = ts.choose_tip_side(
        _row(axis=[[60.0, 100.0], [230.0, 100.0]]), bare, [], px_per_cm=10.0
    )
    assert rule == "lower"


def test_terminal_centroid_sits_on_a_tip_offset_from_the_axis_line() -> None:
    mask = _mask(tip=False)
    # The tip runs 12 px above the axis line.
    mask[84:92, 230:290] = True
    centroid = ts.terminal_centroid(mask, AXIS, 1)
    assert centroid is not None
    assert centroid[0] == pytest.approx(288.5, abs=1.0)
    assert centroid[1] == pytest.approx(88.0, abs=1.0)


# --------------------------------------------------------------------------------------------
# method C: the intensity walk on a rendered frame


def _frame(*, tip: bool, saturated: bool = False) -> np.ndarray:
    bgr = np.full((200, 320, 3), 30, dtype=np.uint8)
    x0, x1, y0, y1 = BODY
    bgr[y0:y1, x0:x1] = 235
    if tip:
        x0, x1, y0, y1 = TIP
        bgr[y0:y1, x0:x1] = (40, 40, 220) if saturated else 225
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)


def test_intensity_walk_finds_the_bright_thin_run_and_stops_at_the_bench() -> None:
    frame = ts.AxisFrame.from_axis(AXIS, 1, body_end=[229.5, 100.0])
    walk = ts.intensity_walk(_frame(tip=True), frame, max_t=100.0, min_run_px=10.0)
    assert walk.present
    assert walk.run_px == pytest.approx(60.0, abs=3.0)
    assert walk.stop_reason == "background"
    assert walk.mean_contrast > 150.0


def test_intensity_walk_reports_no_run_without_a_tip_or_on_a_saturated_object() -> None:
    frame = ts.AxisFrame.from_axis(AXIS, 1, body_end=[229.5, 100.0])
    bare = ts.intensity_walk(_frame(tip=False), frame, max_t=100.0, min_run_px=10.0)
    assert not bare.present and bare.run_px == 0.0
    red = ts.intensity_walk(_frame(tip=True, saturated=True), frame, max_t=100.0, min_run_px=10.0)
    assert not red.present


def test_intensity_estimate_extends_the_row_end_by_the_run() -> None:
    row = _row(body_end_px=[229.5, 100.0])
    estimate = ts.intensity_estimate(row, _frame(tip=True), walk_px=100.0, min_run_px=10.0)
    assert estimate.method == "C" and estimate.tip_present is True
    assert estimate.end_px[0] == pytest.approx(289.5, abs=3.0)
    assert estimate.junction_px == [229.5, 100.0]


# --------------------------------------------------------------------------------------------
# method A: the box attach rule


def test_detector_tip_box_attaches_beyond_the_axis_end_within_the_gate() -> None:
    row = _row()
    box = {
        "object_class": "blue_tip",
        "detector_score": 0.7,
        "box_xyxy_px": [285.0, 90.0, 345.0, 110.0],
    }
    estimate = ts.detector_tip_estimate(row, [box], gate_px=30.0)
    assert estimate.tip_present is True and estimate.side == 1
    assert estimate.end_px == [345.0, 100.0]
    assert estimate.junction_px == [290.0, 100.0]
    assert estimate.tip_length_px == pytest.approx(55.0, abs=0.1)
    assert estimate.confidence == pytest.approx(0.7)
    assert estimate.detail["class_matches"] is True and estimate.detail["box"] == box["box_xyxy_px"]


def test_detector_tip_box_off_the_axis_or_inside_the_body_abstains() -> None:
    row = _row()
    off_axis = {
        "object_class": "blue_tip",
        "detector_score": 0.7,
        "box_xyxy_px": [285.0, 140.0, 345.0, 180.0],
    }
    inside = {
        "object_class": "blue_tip",
        "detector_score": 0.7,
        "box_xyxy_px": [120.0, 90.0, 180.0, 110.0],
    }
    for box in (off_axis, inside):
        estimate = ts.detector_tip_estimate(row, [box], gate_px=30.0)
        assert estimate.tip_present is None
        assert estimate.end_px == [289.5, 100.0]
    other = {
        "object_class": "yellow_tip",
        "detector_score": 0.5,
        "box_xyxy_px": [285.0, 90.0, 345.0, 110.0],
    }
    estimate = ts.detector_tip_estimate(row, [other], gate_px=30.0)
    assert estimate.confidence == pytest.approx(0.35)


# --------------------------------------------------------------------------------------------
# method D: the tip-mask verdict and the box streams


def test_tip_mask_verdict_rejects_blobs_and_detached_masks() -> None:
    assert ts.tip_mask_verdict(None, 10.0) == "too few pixels beyond the body end"
    assert ts.tip_mask_verdict((0.0, 5.0, 100, 4.0), 10.0) == "too short"
    assert ts.tip_mask_verdict((0.0, 60.0, 900, 30.0), 10.0) == "too wide for a tip"
    assert ts.tip_mask_verdict((0.0, 20.0, 200, 10.0), 10.0) == "not elongated"
    assert ts.tip_mask_verdict((20.0, 80.0, 400, 8.0), 10.0) == "not attached to the body end"
    assert ts.tip_mask_verdict((-3.0, 60.0, 400, 8.0), 10.0) is None


def test_box_streams_round_trip_through_the_worker_parser(tmp_path: Path) -> None:
    rows = [_row(frame=700), _row(frame=730, slot="yellow_pipette#0", cls="yellow_pipette")]
    prompts = {
        rows[0].key: {"tip": (280.0, 90.0, 350.0, 110.0), "union": (20.0, 80.0, 350.0, 120.0)},
        rows[1].key: {"tip": (280.0, 90.0, 350.0, 110.0), "union": (20.0, 80.0, 350.0, 120.0)},
    }
    streams = ts.write_box_streams(rows, prompts, output=tmp_path, offset=600)
    stream = streams["T4"]
    assert stream["start_frame"] == 100 and stream["max_frames"] == 31 and stream["prompts"] == 4
    frames, labels = parse_box_stream(Path(stream["path"]).read_text())
    assert sorted(frames) == [0, 30]
    assert set(labels) == {
        "blue_pipette#0|tip",
        "blue_pipette#0|union",
        "yellow_pipette#0|tip",
        "yellow_pipette#0|union",
    }
    assert frames[0][0]["box_xyxy_px"] == [280.0, 90.0, 350.0, 110.0]


def test_slim_tip_box_extends_from_the_body_end_along_the_axis() -> None:
    box = ts.slim_tip_box(_row(), 60.0)
    assert box[0] == pytest.approx(269.5) and box[2] == pytest.approx(349.5)
    assert box[1] == pytest.approx(80.0) and box[3] == pytest.approx(120.0)
    assert ts.clamp_box((-5.0, 0.0, 10.0, 1.0), (100, 100)) is None


# --------------------------------------------------------------------------------------------
# method E: the fusion


def test_fusion_takes_the_agreeing_cluster_beyond_the_mask_end() -> None:
    row = _row()
    estimates = {
        "A": _estimate("A", row, None, [289.5, 100.0], 0.0),
        "B": _estimate("B", row, True, [289.5, 100.0], 1.0, junction=[230.0, 100.0]),
        "C": _estimate("C", row, True, [340.0, 100.0], 0.8, junction=[289.5, 100.0]),
        "D": _estimate("D", row, True, [346.0, 100.0], 0.6, junction=[289.5, 100.0]),
        "D2": _estimate("D2", row, True, [400.0, 100.0], 0.9, junction=[289.5, 100.0]),
    }
    fused = ts.fuse_row(row, estimates)
    assert fused.tip_present is True
    assert fused.detail["agreeing"] == ["C", "D"]
    assert fused.end_px == [340.0, 100.0]
    assert fused.junction_px == [289.5, 100.0]
    assert fused.tip_length_px == pytest.approx(50.5, abs=0.1)
    assert fused.confidence == pytest.approx(1.4 / 2.3, abs=1e-3)


def test_fusion_abstains_on_a_tail_alone_and_says_no_without_one() -> None:
    row = _row()
    tail_only = {
        "B": _estimate("B", row, True, [289.5, 100.0], 1.0, junction=[230.0, 100.0]),
        "C": _estimate("C", row, False, [289.5, 100.0], 0.0),
    }
    fused = ts.fuse_row(row, tail_only)
    assert fused.tip_present is None and fused.end_px == [289.5, 100.0]
    assert fused.junction_px == [230.0, 100.0]
    no_tail = {
        "B": _estimate("B", row, False, [289.5, 100.0], 0.5),
        "C": _estimate("C", row, False, [289.5, 100.0], 0.0),
    }
    assert ts.fuse_row(row, no_tail).tip_present is False
    weak = {
        "B": _estimate("B", row, False, [289.5, 100.0], 0.5),
        "C": _estimate("C", row, True, [320.0, 100.0], 0.3),
    }
    fused = ts.fuse_row(row, weak)
    assert fused.tip_present is False and fused.end_px == [289.5, 100.0]
    assert fused.detail["reason"] == "one weak vote beyond the mask end"


# --------------------------------------------------------------------------------------------
# the views: two synthetic cameras and a 3D segment


def _camera(name: str, rvec, tvec) -> Camera:
    K = np.array([[900.0, 0.0, 960.0], [0.0, 900.0, 540.0], [0.0, 0.0, 1.0]])
    return Camera(
        name, K, np.zeros(5), np.array(rvec, dtype=float), np.array(tvec, dtype=float), (1920, 1080)
    )


def test_triangulate_method_recovers_the_tip_length_and_the_two_state_flag() -> None:
    cams = {
        "T4": _camera("T4", (0.0, 0.0, 0.0), (0.0, 0.0, 150.0)),
        "T5": _camera("T5", (0.0, 0.6, 0.0), (-40.0, 0.0, 150.0)),
    }
    butt, junction, tip = (
        np.array([-10.0, 0.0, 0.0]),
        np.array([13.0, 0.0, 0.0]),
        np.array([19.0, 0.0, 0.0]),
    )
    rows, estimates = {}, []
    for view, cam in cams.items():
        p_butt, p_junction, p_tip = (cam.project(p)[0] for p in (butt, junction, tip))
        row = _row(
            view=view,
            axis=[p_butt.tolist(), p_junction.tolist()],
            tip_side=1,
            body_end_px=p_junction.tolist(),
        )
        rows[row.key] = row
        estimates.append(
            _estimate("C", row, True, p_tip.tolist(), 0.8, junction=p_junction.tolist())
        )
    tips3d = ts.triangulate_method("C", estimates, rows, lambda frame: cams, body_length_cm=23.0)
    assert len(tips3d) == 1
    t = tips3d[0]
    assert t.tip_length_cm == pytest.approx(6.0, abs=0.05)
    assert t.body_length_cm == pytest.approx(23.0, abs=0.05)
    assert t.total_length_cm == pytest.approx(29.0, abs=0.05)
    assert t.with_tip is True and t.tip_votes == 2
    assert max(t.residual_px.values()) < 0.1


# --------------------------------------------------------------------------------------------
# the scorer arithmetic


def test_percentiles_fraction_and_agreement() -> None:
    assert ts._pct([]) == {"n": 0, "median": None, "p10": None, "p90": None}
    assert ts._pct([1.0, 2.0, 3.0]) == {"n": 3, "median": 2.0, "p10": 1.2, "p90": 2.8}
    assert ts._fraction([1], [1, 2, 3, 4]) == 0.25 and ts._fraction([], []) is None
    row = _row()
    rows = {row.key: row}
    estimates = [
        _estimate("B", row, True, [289.5, 100.0], 1.0),
        _estimate("C", row, True, [295.0, 100.0], 0.8),
        _estimate("D", row, False, [289.5, 100.0], 0.0),
    ]
    table = ts.agreement_table(estimates, rows)
    assert table["B-C"]["rows_both_found"] == 1 and table["B-C"]["end_within_gate"] == 1.0
    assert table["B-C"]["end_gap_px"]["median"] == 5.5
    assert table["B-D"]["presence_agreement"] == 0.0 and table["B-D"]["rows_both_found"] == 0
    coverage = ts.coverage_table(estimates, [row])
    assert coverage["B"]["tip_found"] == 1 and coverage["D"]["tip_absent"] == 1
    assert coverage["A"]["abstained"] == 1 and coverage["B"]["found_fraction"] == 1.0


def _tip3d(cls: str, total: float, frame: int, with_tip: bool | None = None) -> ts.Tip3D:
    return ts.Tip3D(
        "E",
        frame,
        cls,
        ["T4", "T5"],
        [0, 0, 0],
        None,
        None,
        None,
        None,
        total,
        {"T4": 1.0},
        None,
        {},
        0,
        with_tip,
    )


def test_two_state_table_splits_two_modes() -> None:
    values = [22.0, 22.3, 21.9, 22.1, 22.2, 21.8, 29.0, 29.4, 28.8, 29.1]
    tips = [_tip3d("blue_pipette", v, i, v > 25.0) for i, v in enumerate(values)]
    block = ts.two_state_table(tips)["blue_pipette"]
    assert block["frames"] == 10 and block["with_tip_frames"] == 4
    assert block["mode_low_cm"] == pytest.approx(22.05, abs=0.01)
    assert block["mode_high_cm"] == pytest.approx(29.05, abs=0.01)
    assert block["n_low"] == 6 and block["n_high"] == 4 and block["bimodal"] is True
    flat = ts.two_state_table([_tip3d("red_pipette", 24.0 + 0.1 * i, i, False) for i in range(8)])
    assert flat["red_pipette"]["bimodal"] is False


def test_blue_timeline_reads_steps_from_the_3d_length_and_drops_flickers() -> None:
    frames = [600, 610, 620, 630, 640, 650, 660]
    totals = {600: 22.0, 610: 22.1, 620: 29.0, 630: 22.0, 640: 29.1, 650: 29.2, 660: 15.0}
    tips = [_tip3d("blue_pipette", totals[f], f) for f in frames]
    rows, estimates = {}, []
    for f in frames:
        row = _row(frame=f)
        rows[row.key] = row
        estimates.append(_estimate("E", row, None, [289.5, 100.0], 0.0))
    tables = ts.FrameTables(
        {},
        {},
        {},
        {
            ("T4", 640): [
                {"object_class": "blue_tip_rack", "box_xyxy_px": [300.0, 90.0, 360.0, 110.0]}
            ]
        },
    )
    timeline = ts.blue_timeline(estimates, rows, tips, tables, frames, body_length_cm=23.0)
    assert timeline["frames_with_verdict"] == 6 and timeline["frames_by_basis"]["3d_truncated"] == 1
    assert timeline["one_frame_flickers"] == 1
    assert [(s["frame"], s["kind"]) for s in timeline["steps"]] == [(640, "pick_up")]
    assert timeline["steps"][0]["near_boxes"] == ["T4:blue_tip_rack"]


def test_working_frames_and_frame_prefilter() -> None:
    assert ts.working_frames([605, 1234], (600, 660), 20) == [600, 605, 620, 640, 1234]
    assert ts._frame_of_line('{"view":"T4","frame_index":1234,"slot":"x"}') == 1234
    assert ts._frame_of_line('{"view":"T4"}') is None


def test_estimates_and_rows_round_trip(tmp_path: Path) -> None:
    row = _row()
    estimate = _estimate("C", row, True, [340.0, 100.0], 0.8, junction=[289.5, 100.0])
    ts.write_rows([row], tmp_path / "rows.jsonl")
    ts.write_estimates([estimate], tmp_path / "estimates.jsonl")
    assert ts.read_rows(tmp_path / "rows.jsonl") == [row]
    assert ts.read_estimates(tmp_path / "estimates.jsonl") == [estimate]
    first = json.loads((tmp_path / "estimates.jsonl").read_text().splitlines()[0])
    assert set(first) >= {
        "method",
        "tip_present",
        "tip_end_px",
        "junction_px",
        "tip_length_px",
        "confidence",
        "end_px",
        "side",
    }
