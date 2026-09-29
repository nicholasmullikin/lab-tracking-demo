"""`battle-finebio-stand` (p0-stand-slice): static frames from detector boxes, 3D segments
from per-view mask axes on the real P03 rig fixture with synthetic pipettes, the rest gates,
the report and the pipettes config (no data/, no GPU)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from finebio_fixtures import FIXTURE_DIR, load_preflight_fixtures

from battle.finebio_stand import (
    BENCH_UP,
    StandSettings,
    main,
    match_endpoints,
    pipettes_config,
    segment_from_views,
    stand_report,
    stand_window,
    static_frames,
    stationary_runs,
)
from battle.multiview_schemas import FineBioObservation

FRAMES = range(1800, 1840)


@pytest.fixture(scope="module")
def fixtures():
    return load_preflight_fixtures()


def _det(view: str, frame: int, cls: str, box) -> FineBioObservation:
    return FineBioObservation(
        view=view,
        frame_index=frame,
        slot=f"{cls}#0",
        object_class=cls,
        detector_score=0.9,
        box_xyxy_px=tuple(float(x) for x in box),
        pose_valid=True,
        source="detector",
    )


def test_stationary_runs_need_small_steps_and_a_minimum_length():
    centres = {f: (100.0 + (f % 2), 50.0) for f in range(0, 40)}  # jitter of 1 px
    centres.update({f: (100.0 + 5.0 * f, 50.0) for f in range(40, 60)})  # moving
    centres.update({f: (300.0, 50.0) for f in range(70, 90)})  # a gap, then 20 still frames
    runs = stationary_runs(centres, max_step_px=3.0, min_run_frames=30)
    assert runs == [(0, 39)]
    assert stationary_runs(centres, max_step_px=3.0, min_run_frames=20) == [(0, 39), (70, 89)]
    assert stationary_runs({}, max_step_px=3.0, min_run_frames=1) == []


def test_static_frames_count_fixed_views_and_the_window_prefers_more_pipettes():
    settings = StandSettings(min_run_frames=10, classes=("blue_pipette", "red_pipette"))
    rows = []
    for f in range(600, 660):
        for view in ("T1", "T2", "T3"):
            rows.append(_det(view, f, "blue_pipette", (10, 10, 50, 90)))
        # The red is still in one fixed view only, and its fpv box does not count.
        rows.append(_det("T4", f, "red_pipette", (200, 10, 240, 90)))
        rows.append(_det("fpv", f, "red_pipette", (200, 10, 240, 90)))
        rows.append(_det("T5", f, "red_pipette", (200 + 4 * f, 10, 240 + 4 * f, 90)))
    static = static_frames(rows, settings)
    assert set(static) == {"blue_pipette"}
    assert static["blue_pipette"][600] == ("T1", "T2", "T3") and len(static["blue_pipette"]) == 60
    resting = {"blue_pipette": range(600, 660), "red_pipette": range(620, 635)}
    window = stand_window(
        resting, StandSettings(min_run_frames=10, shared_classes=settings.classes)
    )
    assert (window.start, window.end, window.static_count) == (620, 634, 2)
    assert window.classes == ("blue_pipette", "red_pipette")
    # Too short a joint run: the longest single-pipette run wins.
    window = stand_window(
        resting, StandSettings(min_run_frames=20, shared_classes=settings.classes)
    )
    assert (window.start, window.end, window.static_count) == (600, 659, 1)
    assert stand_window({}, settings) is None


# ------------------------------------------------------------------ synthetic pipettes


def _project(cam, point: np.ndarray) -> np.ndarray:
    return cam.project(np.asarray(point, dtype=np.float64))[0]


def _rows_for_segment(
    cams,
    cls: str,
    end_top: np.ndarray,
    end_bottom: np.ndarray,
    frames,
    *,
    seed: int,
    views=None,
    bad_view: str | None = None,
) -> list[FineBioObservation]:
    """Detector and SAM3 rows of one segment in every fixed view: the axis is the projected
    ends (in a per-view shuffled order), the box their bounds; `bad_view` gets one end pushed
    60 px along the axis, a mask that runs long."""
    rng = np.random.default_rng(seed)
    rows = []
    for view, cam in cams.items():
        if views is not None and view not in views:
            continue
        top, bottom = _project(cam, end_top), _project(cam, end_bottom)
        if view == bad_view:
            direction = (bottom - top) / np.linalg.norm(bottom - top)
            bottom = bottom + 60.0 * direction
        ends = [top, bottom] if rng.random() < 0.5 else [bottom, top]
        x0, y0 = np.minimum(top, bottom) - 8
        x1, y1 = np.maximum(top, bottom) + 8
        box = (round(float(x0), 1), round(float(y0), 1), round(float(x1), 1), round(float(y1), 1))
        centroid = tuple(round(float(v), 1) for v in (top + bottom) / 2)
        for f in frames:
            rows.append(_det(view, f, cls, box))
            rows.append(
                FineBioObservation(
                    view=view,
                    frame_index=f,
                    slot=f"{cls}#0",
                    object_class=cls,
                    detector_score=0.9,
                    box_xyxy_px=box,
                    mask_bbox_px=box,
                    mask_centroid_px=centroid,
                    mask_area_px=2000,
                    mask_axis_px=tuple(tuple(round(float(v), 1) for v in p) for p in ends),
                    mask_elongation=8.0,
                    mask_width_px=20.0,
                    mask_axis_residual_px=1.0,
                    sam3_object_score=0.9,
                    pose_valid=True,
                    source="sam3_decode",
                    provenance={"axis_method": "ransac_skeleton"},
                )
            )
    return rows


def _standing(x: float, y: float, length: float = 25.0, tilt_deg: float = 0.0):
    """A pipette standing on the bench at (x, y): bottom on the plane, top `length` up the
    bench normal, leaning `tilt_deg` towards +x."""
    bottom = np.array([x, y, 0.0])
    lean = np.array([np.sin(np.radians(tilt_deg)), 0.0, 0.0])
    up = BENCH_UP * np.cos(np.radians(tilt_deg)) + lean
    return bottom + length * up, bottom


def test_match_endpoints_labels_the_higher_end_a_in_every_view(fixtures):
    cams = fixtures.fixed_cameras()
    top, bottom = _standing(20.0, 10.0)
    axes = {}
    for i, (view, cam) in enumerate(cams.items()):
        t, b = _project(cam, top), _project(cam, bottom)
        axes[view] = (t, b) if i % 2 else (b, t)
    ends_a, ends_b, info = match_endpoints(cams, axes, (top + bottom) / 2)
    for view, cam in cams.items():
        assert np.allclose(ends_a[view], _project(cam, top), atol=1e-6)
        assert np.allclose(ends_b[view], _project(cam, bottom), atol=1e-6)
    assert info["up_order_disagreements"] == [] and info["anchor_view"] in cams


def test_vertical_segments_on_the_rig_fixture_recover_length_angle_and_spacing(fixtures):
    """Three 25 cm pipettes standing 4 cm apart, a leaning fourth, a fifth held in the air and
    a sixth with one view's mask running long: the resting ones come back within the rig's
    noise, the held one fails the height gate and the long mask's view is dropped."""
    cams = fixtures.fixed_cameras()
    rows = []
    stand = {}
    for i, cls in enumerate(("blue_pipette", "yellow_pipette", "red_pipette")):
        top, bottom = _standing(10.0 + 4.0 * i, 20.0)
        stand[cls] = (top, bottom)
        rows += _rows_for_segment(cams, cls, top, bottom, FRAMES, seed=i)
    top, bottom = _standing(40.0, 20.0, length=22.0, tilt_deg=20.0)
    rows += _rows_for_segment(cams, "8_channel_pipette", top, bottom, FRAMES, seed=7)
    settings = StandSettings(min_run_frames=20)
    report, segments = stand_report(rows, cams, trial="fixture", settings=settings)
    window = report["stand_window"]
    assert (window["start_frame"], window["end_frame"], window["static_pipettes"]) == (
        1800,
        1839,
        3,
    )
    assert window["classes"] == ["blue_pipette", "yellow_pipette", "red_pipette"]
    for cls, (top, bottom) in stand.items():
        p = report["per_pipette"][cls]
        assert p["frames_with_segment"] == len(FRAMES) and p["frames_failed_rest_gate"] == 0
        assert abs(p["length_cm"]["median"] - 25.0) < 0.3
        assert p["length_cm"]["p90"] - p["length_cm"]["p10"] < 0.2
        assert p["angle_to_bench_normal_deg"]["median"] < 1.0
        assert abs(p["end_a_height_cm"]["median"] - 25.0) < 0.3
        assert abs(p["end_b_height_cm"]["median"]) < 0.3
        assert p["end_a_drift_window"]["max_cm"] < 0.1 and p["end_b_drift_window"]["max_cm"] < 0.1
        assert p["view_sets"] == {"T1,T2,T3,T4,T5": len(FRAMES)}
        for view in cams:
            assert p["residual_a_px_per_view"][view]["median"] < 2.0
            assert p["loo_b_px_per_view"][view]["median"] < 3.0
    leaning = report["per_pipette"]["8_channel_pipette"]
    assert abs(leaning["length_cm"]["median"] - 22.0) < 0.3
    assert abs(leaning["angle_to_bench_normal_deg"]["median"] - 20.0) < 1.0
    pairs = {tuple(p["pair"]): p for p in report["pairwise"]}
    assert abs(pairs[("blue_pipette", "yellow_pipette")]["spacing_cm"]["median"] - 4.0) < 0.2
    assert abs(pairs[("blue_pipette", "red_pipette")]["spacing_cm"]["median"] - 8.0) < 0.2
    assert pairs[("yellow_pipette", "red_pipette")]["angle_deg"]["median"] < 1.0
    assert abs(pairs[("red_pipette", "8_channel_pipette")]["angle_deg"]["median"] - 20.0) < 1.0
    shared = report["shared_length"]
    assert abs(shared["length_cm"] - 25.0) < 0.3 and shared["spread_cm"] < 0.2
    assert shared["pipettes"] == 3 and shared["excluded_too_few_resting_frames"] == []
    assert all(seg.views_dropped == () for seg in segments["blue_pipette"])


def test_rest_gates_exclude_a_held_pipette_and_drop_a_long_mask_view(fixtures):
    cams = fixtures.fixed_cameras()
    rows = []
    top, bottom = _standing(10.0, 20.0)
    rows += _rows_for_segment(cams, "red_pipette", top, bottom, FRAMES, seed=1)
    # Held still 30 cm up: static by its boxes, out by the height gate.
    lift = np.array([0.0, 0.0, -30.0])
    rows += _rows_for_segment(cams, "blue_pipette", top + lift, bottom + lift, FRAMES, seed=2)
    # T2's mask runs 60 px long at one end: the leave-one-out rule drops T2.
    top_y, bottom_y = _standing(18.0, 20.0)
    rows += _rows_for_segment(
        cams, "yellow_pipette", top_y, bottom_y, FRAMES, seed=3, bad_view="T2"
    )
    settings = StandSettings(min_run_frames=20)
    report, segments = stand_report(rows, cams, trial="fixture", settings=settings)
    blue = report["per_pipette"]["blue_pipette"]
    assert blue["frames_static_with_axes"] == len(FRAMES)
    assert blue["frames_failed_height_gate"] == len(FRAMES) and blue["frames_with_segment"] == 0
    assert blue["frames_failed_residual_gate"] == 0
    yellow = report["per_pipette"]["yellow_pipette"]
    assert yellow["frames_with_segment"] == len(FRAMES)
    assert yellow["views_dropped"] == {"T2": len(FRAMES)}
    assert yellow["view_sets"] == {"T1,T3,T4,T5": len(FRAMES)}
    assert abs(yellow["length_cm"]["median"] - 25.0) < 0.3
    window = report["stand_window"]
    assert window["classes"] == ["yellow_pipette", "red_pipette"] and window["static_pipettes"] == 2
    shared = report["shared_length"]
    assert shared["pipettes"] == 2 and shared["excluded_too_few_resting_frames"] == ["blue_pipette"]
    # Two views cannot vote a view out: the segment keeps both and fails the gate instead.
    per_view = {
        r.view: r
        for r in rows
        if r.object_class == "yellow_pipette"
        and r.source == "sam3_decode"
        and r.frame_index == 1800
        and r.view in ("T2", "T4")
    }
    seg = segment_from_views(
        1800, "yellow_pipette", cams, per_view, min_views=2, rest_residual_px=30.0
    )
    assert seg is not None and seg.views_dropped == ()
    assert seg.max_residual_px > 30.0 or abs(seg.length_cm - 25.0) > 1.0


def test_pipettes_config_keeps_trial_one_and_writes_a_disagreement_note():
    def report(trial: str, lengths: dict[str, float], spacing: float):
        values = list(lengths.values())
        return {
            "trial": trial,
            "stand_window": {"start_frame": 600, "end_frame": 900},
            "shared_length": {
                "length_cm": float(np.median(values)),
                "spread_cm": round(max(values) - min(values), 2),
                "per_pipette_median_cm": lengths,
            },
            "pairwise": [
                {"pair": ["yellow_pipette", "red_pipette"], "spacing_cm": {"median": spacing}},
                {"pair": ["yellow_pipette", "8_channel_pipette"], "spacing_cm": {"median": 20.0}},
            ],
        }

    one = report("P03_03_01", {"yellow_pipette": 22.0, "red_pipette": 24.5}, 2.9)
    two = report("P20_03_01", {"blue_pipette": 28.0, "red_pipette": 29.0}, 3.1)
    doc = pipettes_config([one, two], run_dirs=["runs/one/stand", "runs/two/stand"])
    assert doc["config_kind"] == "finebio_pipettes"
    assert doc["length_cm"] == 23.25 and doc["length_spread_cm"] == 2.5
    assert doc["stand_spacing_cm"] == 2.9  # single-channel pairs only
    assert doc["elongation_threshold"] == 2.5
    assert doc["merged_width_factor"] == 1.6 and doc["merged_extent_factor"] == 1.2
    prov = doc["provenance"]
    assert prov["trial"] == "P03_03_01" and prov["frames"] == [600, 900]
    assert prov["run_dir"] == "runs/one/stand"
    assert prov["per_trial"]["P20_03_01"]["primary"] is False
    assert len(prov["notes"]) == 1 and "kept" in prov["notes"][0]
    close = report("P20_03_01", {"blue_pipette": 22.5, "red_pipette": 24.0}, 3.1)
    agree = pipettes_config([one, close], run_dirs=["a", "b"])
    assert agree["length_cm"] == 23.25 and "within the spread" in agree["provenance"]["notes"][0]


def test_cli_report_and_config_write_their_files(tmp_path: Path, fixtures):
    from battle.finebio_observations import write_observations

    cams = fixtures.fixed_cameras()
    rows = []
    for i, cls in enumerate(("yellow_pipette", "red_pipette")):
        top, bottom = _standing(10.0 + 3.0 * i, 20.0)
        rows += _rows_for_segment(cams, cls, top, bottom, FRAMES, seed=i)
    obs = tmp_path / "observations.jsonl"
    write_observations(rows, obs)
    out = tmp_path / "stand"
    argv = [
        "report",
        "--observations",
        str(obs),
        "--cameras",
        str(FIXTURE_DIR / "cameras.json"),
        "--output",
        str(out),
        "--min-run-frames",
        "20",
        "--frames",
        "1800-1839",
    ]
    assert main(argv) == 0
    report = json.loads((out / "stand_report.json").read_text())
    assert report["trial"] == "P03_01_01" and report["stand_window"]["frames"] == 40
    assert abs(report["shared_length"]["length_cm"] - 25.0) < 0.3
    assert (out / "segments.jsonl").read_text().count("\n") == 80
    md = (out / "stand_report.md").read_text()
    assert md.startswith("# Pipettes at rest, P03_01_01") and "## Shared length" in md
    config = tmp_path / "pipettes.json"
    assert (
        main(["config", "--report", str(out / "stand_report.json"), "--output", str(config)]) == 0
    )
    doc = json.loads(config.read_text())
    assert abs(doc["length_cm"] - 25.0) < 0.3 and abs(doc["stand_spacing_cm"] - 3.0) < 0.2
    assert doc["provenance"]["run_dir"] == str(out)


def _tip_box(view: str, frame: int, cam, body_end, tip_end, cls: str) -> FineBioObservation:
    pixels = cam.project(np.stack([body_end, tip_end]))
    x0, y0 = pixels.min(axis=0) - 6.0
    x1, y1 = pixels.max(axis=0) + 6.0
    return _det(view, frame, cls, (x0, y0, x1, y1))


def test_two_state_rest_lengths_split_by_the_attached_tip_and_write_the_config(
    tmp_path: Path, fixtures
):
    """Sep 29, disposable tips. A yellow pipette rests bare for 40 frames (the bare mode),
    then moves for 30 frames carrying a 5 cm tip that two fixed views see as a `yellow_tip`
    box on the body's end: the all-frames population measures the tip as the extended minus
    the bare length, and `tips-config` writes `bare_length_cm` and `tip_length_cm` beside the
    old keys, naming the assumed classes. Without tip boxes (the v2 rows) the report has one
    state and no tip estimate."""
    from battle.finebio_observations import write_observations
    from battle.finebio_stand import tip_length_report, two_state_config

    cams = fixtures.fixed_cameras()
    bare, tip = 22.0, 5.0
    top, bottom = _standing(10.0, 20.0, length=bare)
    rest = list(range(1800, 1840))
    rows = _rows_for_segment(cams, "yellow_pipette", top, bottom, rest, seed=1)
    # Then carried flat above the bench, 1 cm a frame along y, the tip on the +x end.
    moving = list(range(1840, 1870))
    for f in moving:
        butt = np.array([-5.0, -20.0 + (f - 1840), -12.0])
        body_end = butt + np.array([bare, 0.0, 0.0])
        tip_end = body_end + np.array([tip, 0.0, 0.0])
        rows += _rows_for_segment(cams, "yellow_pipette", butt, body_end, [f], seed=f)
        for view in ("T1", "T4"):
            rows.append(_tip_box(view, f, cams[view], body_end, tip_end, "yellow_tip"))
    settings = StandSettings(min_run_frames=20, fixed_views=tuple(cams))
    report, records = tip_length_report(
        rows, cams, trial="P03_01_01", settings=settings, gate_px=30.0, min_attached_views=2
    )
    block = report["per_class"]["yellow_pipette"]
    assert block["bare_body_length_cm"]["n"] == 40
    assert abs(block["bare_body_length_cm"]["median"] - bare) < 0.3
    assert block["frames_with_tip_attached"]["ge_2_views"] == 0  # no tip at rest
    all_frames = block["all_frames_with_tip"]
    assert all_frames["frames"] == 30 and all_frames["views"] == {"T1": 30, "T4": 30}
    # The extension runs to the far edge of the detector box (6 px of slack here), so the
    # measured tip is the true one plus that slack in cm: biased long, as documented.
    assert tip <= all_frames["difference_cm"]["median"] < tip + 1.2
    assert bare + tip <= all_frames["extended_length_cm"]["median"] < bare + tip + 1.2
    assert tip - 0.3 <= block["tip_length_estimate_cm"] < tip + 1.2
    assert block["tip_class"] == "yellow_tip"
    assert block["tip_length_basis"].startswith("all frames attached in >= 2 views: the extended")
    assert len(records) == 40 and all(r["attached_views"] == [] for r in records)
    # The v2 rows: no tip box anywhere, one state.
    plain = [r for r in rows if r.object_class != "yellow_tip"]
    one_state, _ = tip_length_report(
        plain, cams, trial="P03_01_01", settings=settings, gate_px=30.0, min_attached_views=2
    )
    assert one_state["per_class"]["yellow_pipette"]["tip_length_estimate_cm"] is None
    assert one_state["per_class"]["yellow_pipette"]["all_frames_with_tip"]["frames"] == 0
    # The config: bare from the rest mode, the yellow tip measured, the others assumed.
    config = {
        "length_cm": 23.0,
        "length_spread_cm": 2.0,
        "provenance": {"per_trial": {"X": {"per_pipette_median_cm": {"red_pipette": 24.5}}}},
        "colour": {"kept": True},
    }
    doc = two_state_config(config, [report], report_paths=["tips.json"])
    assert doc["colour"] == {"kept": True} and doc["length_cm"] == 23.0
    assert abs(doc["bare_length_cm"]["yellow_pipette"] - bare) < 0.3
    assert doc["bare_length_cm"]["red_pipette"] == 24.5  # the single-state median, named
    assert "single-state" in doc["two_state_provenance"]["bare_basis"]["red_pipette"]
    assert doc["bare_length_cm"]["blue_pipette"] == 23.0  # the shared length, named
    assert tip - 0.3 <= doc["tip_length_cm"]["yellow_tip"] < tip + 1.2
    assert doc["tip_length_cm"]["blue_tip"] == doc["tip_length_cm"]["yellow_tip"]
    assert doc["two_state_provenance"]["tip_basis"]["blue_tip"].startswith("assumed")
    # And through the CLI, writing beside the old keys.
    obs = tmp_path / "observations.jsonl"
    write_observations(rows, obs)
    out = tmp_path / "tips"
    assert (
        main(
            [
                "tips",
                "--observations",
                str(obs),
                "--cameras",
                str(FIXTURE_DIR / "cameras.json"),
                "--output",
                str(out),
                "--min-run-frames",
                "20",
            ]
        )
        == 0
    )
    written = json.loads((out / "tip_lengths.json").read_text())
    assert tip - 0.3 <= written["per_class"]["yellow_pipette"]["tip_length_estimate_cm"] < tip + 1.2
    assert (out / "tip_lengths.md").read_text().startswith("# Two-state rest lengths")
    cfg = tmp_path / "pipettes.json"
    cfg.write_text(json.dumps(config))
    assert main(["tips-config", "--config", str(cfg), "--tips", str(out / "tip_lengths.json")]) == 0
    updated = json.loads(cfg.read_text())
    assert set(updated) >= {"bare_length_cm", "tip_length_cm", "two_state_provenance", "colour"}
