"""`battle-finebio-arms lines-scoreboard` and `lines-negative-controls` (p3-metrics) on
synthetic line tracks over the preflight fixture cameras (no data/, no GPU)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from finebio_fixtures import FIXTURE_DIR, load_preflight_fixtures

from battle.finebio_arms import main
from battle.finebio_lines_scoreboard import (
    GEOMETRIC_CLASS,
    REST_MIN_FRAMES,
    ambiguities,
    build_scoreboard,
    colour_summary,
    held_pipette_class,
    ids_summary,
    loo_and_length,
    pipette_observations,
    plausibility,
    resolve_tracks_dir,
    rest_geometry,
    rows_by_track,
    run_negative_controls,
    scoreboard_markdown,
    shipped_camera_config,
    stationary_runs,
    write_control_observations,
)
from battle.multiview_schemas import FineBioObservation, Track3D, write_jsonl
from battle.multiview_tracks import LinePrior

FIXED = ("T1", "T2", "T3", "T4", "T5")
FRAMES = list(range(1798, 1798 + REST_MIN_FRAMES + 6))
PRIOR = LinePrior(length_cm=23.0, spread_cm=2.0, per_class={"blue_pipette": 22.0})
CLIP_CONFIG = Path("configs/clips/finebio_P03_03_01_600-4200.json")


@pytest.fixture(scope="module")
def fixtures():
    return load_preflight_fixtures()


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _segment(mid, direction, length):
    mid, direction = np.asarray(mid, dtype=float), _unit(direction)
    return mid - direction * length / 2, mid + direction * length / 2


def _obs_row(cam, view, frame, a, b, cls, slot, rng, *, compact=False) -> FineBioObservation:
    ends = cam.project(np.stack([a, b])) + rng.normal(0.0, 0.3, (2, 2))
    centroid = cam.project(0.5 * (a + b))[0]
    x0, y0 = ends.min(axis=0) - 10
    x1, y1 = ends.max(axis=0) + 10
    box = (float(x0), float(y0), float(x1), float(y1))
    return FineBioObservation(
        view=view,
        frame_index=frame,
        slot=slot,
        object_class=cls,
        detector_score=0.8,
        box_xyxy_px=box,
        mask_bbox_px=box,
        mask_centroid_px=(float(centroid[0]), float(centroid[1])),
        mask_area_px=4000,
        mask_axis_px=None
        if compact
        else ((float(ends[0, 0]), float(ends[0, 1])), (float(ends[1, 0]), float(ends[1, 1]))),
        mask_elongation=1.2 if compact else 6.0,
        mask_width_px=20.0,
        mask_axis_residual_px=2.0,
        sam3_object_score=0.9,
        pose_valid=True,
        source="sam3_decode",
    )


def _line_row(track_id, frame, a, b, cls, views, *, state="observed", **extra) -> Track3D:
    mid = 0.5 * (a + b)
    direction = _unit(b - a)
    return Track3D(
        frame_index=frame,
        track_id=track_id,
        object_class=GEOMETRIC_CLASS,
        position_cm=tuple(float(x) for x in mid),
        uncertainty_cm=1.0,
        support_views=tuple(views),
        state=state,
        confidence=0.8,
        abstain=False,
        support_slots={v: f"{cls}#0" for v in views},
        direction=tuple(float(x) for x in direction),
        endpoints_cm=(tuple(float(x) for x in a), tuple(float(x) for x in b)),
        tip_resolved=True,
        observed_class=cls,
        line_residual_px={v: 1.5 for v in views} if state in ("observed", "single_view") else None,
        **extra,
    )


def _point_row(track_id, frame, cls, point) -> Track3D:
    return Track3D(
        frame_index=frame,
        track_id=track_id,
        object_class=cls,
        position_cm=tuple(float(x) for x in point),
        uncertainty_cm=1.0,
        support_views=("T1", "T4"),
        state="observed",
        confidence=0.8,
        abstain=False,
    )


def _metrics(per_class, *, lines=None, gates=None) -> dict:
    return {
        "tracks_born": sum(v["tracks_born"] for v in per_class.values()),
        "ambiguities": 1,
        "duplicate_pair_frames": 4,
        "id_switches": 3,
        "id_switch_reference": "sam3_slots_proxy",
        "per_class": per_class,
        "params": {
            "fpv_weight": 0.5,
            "gates": gates or {"association_px": 30.0, "handoff_px": 27.0},
            "extensions": {"line_min_pair_angle_deg": 5.0},
        },
        "extensions": {"lines": lines or {"enabled": False}},
    }


@pytest.fixture(scope="module")
def synthetic(fixtures, tmp_path_factory):
    """Two line tracks: a resting blue pipette (every frame, five planes) and a red pipette
    1 cm beside it, parallel, sliding 1 cm a frame along its axis (an ambiguity outside the
    rest frames), held by the right hand on its last frames; observations for both; a point
    tracker baseline with five blue ids and a point run with two; colour rows that disagree
    on the blue track."""
    root = tmp_path_factory.mktemp("lines")
    cams = fixtures.fixed_cameras()
    rng = np.random.default_rng(7)
    length = PRIOR.length_for("blue_pipette")
    direction = _unit([1.0, 0.2, 0.0])
    a1, b1 = _segment([0.0, -5.0, -1.0], direction, length)
    across = _unit(np.cross(direction, [0.0, 0.0, 1.0]))
    observations, lines_rows, ext_rows, base_rows, colour_rows = [], [], [], [], []
    for k, frame in enumerate(FRAMES):
        for v in FIXED:
            observations.append(
                _obs_row(cams[v], v, frame, a1, b1, "blue_pipette", "blue_pipette#0", rng)
            )
        lines_rows.append(_line_row("pipette-001", frame, a1, b1, "blue_pipette", FIXED))
        a2, b2 = a1 + across * 1.0 + direction * k, b1 + across * 1.0 + direction * k
        views2 = ("T1", "T3", "T4")
        for v in views2:
            observations.append(
                _obs_row(cams[v], v, frame, a2, b2, "red_pipette", "red_pipette#0", rng)
            )
        held = k >= len(FRAMES) - 4
        lines_rows.append(
            _line_row(
                "pipette-002",
                frame,
                a2,
                b2,
                "red_pipette",
                views2,
                state="held" if held else "observed",
                held_by="right_hand-001" if held else None,
            )
        )
        lines_rows.append(_point_row("right_hand-001", frame, "right_hand", b2 + across * 2.0))
        # Point runs: the baseline splits the blue pipette into five ids, the new run into two.
        base_rows.append(_point_row(f"blue_pipette-{k // 7 + 1:03d}", frame, "blue_pipette", a1))
        ext_rows.append(_point_row(f"blue_pipette-{k // 18 + 1:03d}", frame, "blue_pipette", a1))
        base_rows.append(_point_row("red_pipette-009", frame, "red_pipette", a2))
        ext_rows.append(_point_row("red_pipette-009", frame, "red_pipette", a2))
    for row in lines_rows:
        if row.object_class != GEOMETRIC_CLASS:
            continue
        colour_rows.append(
            {
                **row.model_dump(exclude_none=True),
                "colour_identity": "yellow" if row.track_id == "pipette-001" else "red",
                "colour_confidence": 0.7,
                "colour_entropy": 0.9,
                "colour_samples": 12,
            }
        )
    obs_dir = root / "observations-b"
    obs_dir.mkdir()
    write_jsonl(observations, obs_dir / "observations.jsonl")
    lines_dir = root / "arm-b-lines" / "tracks-lines"
    ext_dir = root / "arm-b-ext" / "tracks-ext"
    base_dir = root / "baseline" / "tracks-ext"
    for d in (lines_dir, ext_dir, base_dir):
        d.mkdir(parents=True)
    write_jsonl(lines_rows, lines_dir / "tracks.jsonl")
    write_jsonl(ext_rows, ext_dir / "tracks.jsonl")
    write_jsonl(base_rows, base_dir / "tracks.jsonl")
    n = len(FRAMES)
    lines_metrics = _metrics(
        {
            "pipette": {"tracks_born": 2, "max_simultaneous": 2, "fragmentation": 0},
            "blue_pipette": {
                "tracks_born": 1,
                "max_simultaneous": 1,
                "fragmentation": 0,
                "counted_by": "plurality observed_class of the pipette tracks",
            },
            "red_pipette": {
                "tracks_born": 1,
                "max_simultaneous": 1,
                "fragmentation": 0,
                "counted_by": "plurality observed_class of the pipette tracks",
            },
        },
        lines={
            "enabled": True,
            "frames": {
                "line": 2 * n,
                "line_prediction_aided": 3,
                "point_fallback": 2,
                "single_view": 1,
                "degenerate_fits": 2,
                "extended_by_prior": 4,
            },
            "loo_residual_px": {"n": 5 * n, "median": 1.4, "p90": 3.0},
            "loo_angle_deg": {"n": 5 * n, "median": 0.3, "p90": 1.0},
            "length_cm": {"visible": {"n": 2 * n, "median": 22.5, "p90": 23.5}},
            "merged_views_by_view": {"T2": 3},
            "tip_resolved_fraction": 0.9,
            "tip_resolutions_by_basis": {"hand_track": 2},
            "line_births": 2,
            "point_births": 0,
            "class_agreement": 0.95,
        },
    )
    (lines_dir / "identity_metrics.json").write_text(json.dumps(lines_metrics))
    (ext_dir / "identity_metrics.json").write_text(
        json.dumps(
            _metrics(
                {
                    "blue_pipette": {"tracks_born": 2, "fragmentation": 1},
                    "red_pipette": {"tracks_born": 1, "fragmentation": 0},
                }
            )
        )
    )
    (base_dir / "identity_metrics.json").write_text(
        json.dumps(
            _metrics(
                {
                    "blue_pipette": {"tracks_born": 5, "fragmentation": 4},
                    "red_pipette": {"tracks_born": 1, "fragmentation": 0},
                }
            )
        )
    )
    with (lines_dir / "tracks_colour.jsonl").open("w") as handle:
        for row in colour_rows:
            handle.write(json.dumps(row) + "\n")
    angle = float(np.degrees(np.arccos(abs(direction @ np.array([0.0, 0.0, -1.0])))))
    stand = {
        "settings": {"max_step_px": 3.0},
        "per_pipette": {
            "blue_pipette": {
                "angle_to_bench_normal_deg": {"median": round(angle, 1)},
                "end_a_drift_window": {"p90_cm": 0.1},
                "end_b_drift_window": {"p90_cm": 0.2},
                "frames_with_segment": 100,
            }
        },
    }
    (root / "stand_report.json").write_text(json.dumps(stand))
    (root / "rig.json").write_text(
        json.dumps({"gates": {"inputs": {"static_loo_median_px": 10.0}, "association_px": 30.0}})
    )
    return {
        "root": root,
        "cams": cams,
        "lines_dir": lines_dir,
        "ext_dir": ext_dir,
        "base_dir": base_dir,
        "obs_dir": obs_dir,
        "lines_rows": lines_rows,
        "lines_metrics": lines_metrics,
        "stand": stand,
        "segment": (a1, b1),
        "colour_rows": colour_rows,
    }


def test_observations_index_prefers_sam3_rows_and_keeps_pipette_classes_only(synthetic):
    obs = pipette_observations(synthetic["obs_dir"])
    assert all(k[2].endswith("#0") for k in obs)
    assert len(obs) == len(FRAMES) * (5 + 3)
    assert obs[("T1", FRAMES[0], "blue_pipette#0")].source == "sam3_decode"
    assert (
        resolve_tracks_dir(synthetic["lines_dir"].parent, "tracks-lines") == synthetic["lines_dir"]
    )
    with pytest.raises(FileNotFoundError):
        resolve_tracks_dir(synthetic["root"], "tracks-lines")


def test_ids_lifetimes_and_births_per_minute(synthetic):
    lines_rows = synthetic["lines_rows"]
    by_track = rows_by_track(lines_rows, lambda r: r.object_class == GEOMETRIC_CLASS)
    ids = ids_summary(synthetic["lines_metrics"], by_track, geometric=True, minutes=2.0)
    assert (
        ids["by_class"]["blue_pipette"]["ids"] == 1 and ids["by_class"]["red_pipette"]["ids"] == 1
    )
    assert ids["by_class"]["blue_pipette"]["lifetime_frames"]["median"] == len(FRAMES)
    assert ids["by_class"]["blue_pipette"]["births_per_minute"] == 0.5
    assert ids["geometric"]["ids"] == 2 and ids["counted_by"].startswith("plurality")
    base = json.loads((synthetic["base_dir"] / "identity_metrics.json").read_text())
    assert held_pipette_class(base) == "blue_pipette"


def test_loo_line_beats_box_centre_and_length_sits_on_the_prior(synthetic, fixtures):
    obs = pipette_observations(synthetic["obs_dir"])
    out = loo_and_length(
        synthetic["lines_rows"], obs, synthetic["cams"], lambda f: None, PRIOR, fpv_weight=0.5
    )
    # Every line-fit frame of both tracks has >= 3 planes; the red track's four held rows
    # carry no line fit and are left out.
    fit_rows = 2 * len(FRAMES) - 4
    cells = 5 * len(FRAMES) + 3 * (len(FRAMES) - 4)
    assert out["frames_with_3_planes"] == fit_rows and out["line_fit_rows"] == fit_rows
    assert out["line_fit_rows_with_missing_observations"] == 0
    line = out["line_recomputed"]["perpendicular_px"]
    box = out["box_centre"]
    # A held-out fit whose remaining planes meet under the pair angle is not a cell.
    assert 0.9 * cells <= line["n"] <= cells and line["median"] < 2.0
    # The centroids are the shafts' midpoints seen whole from every view, so the box-centre
    # point lands on the shaft: small perpendicular residual, small midpoint distance too.
    assert box["perpendicular_to_axis_px"]["n"] == cells
    assert box["perpendicular_to_axis_px"]["median"] < 5.0
    assert box["distance_to_axis_midpoint_px"]["median"] < 10.0
    assert set(box["by_view_distance_median_px"]) == set(FIXED)
    lengths = out["length_by_class"]
    assert lengths["blue_pipette"]["visible_cm"]["median"] == pytest.approx(22.0, abs=0.5)
    assert lengths["blue_pipette"]["prior_cm"] == 22.0
    assert abs(lengths["blue_pipette"]["deviation_from_prior_cm"]["median"]) < 0.5
    assert lengths["red_pipette"]["prior_cm"] == 23.0


def test_rest_runs_angle_and_drift_against_the_stand_and_the_rig(synthetic):
    rows = synthetic["lines_rows"]
    by_track = rows_by_track(rows, lambda r: r.object_class == GEOMETRIC_CLASS)
    assert len(stationary_runs(by_track["pipette-001"])) == 1
    assert stationary_runs(by_track["pipette-002"]) == []  # slides 1 cm a frame
    block, rest_frames = rest_geometry(
        rows,
        synthetic["stand"],
        fixed_cams=synthetic["cams"],
        rig_static_loo_px=10.0,
        prior=PRIOR,
    )
    assert set(rest_frames) == {"pipette-001"} and len(rest_frames["pipette-001"]) == len(FRAMES)
    blue = block["by_class"]["blue_pipette"]
    assert blue["runs"] == 1 and blue["frames"] == len(FRAMES)
    assert blue["end_drift_p90_cm"] == 0.0
    assert blue["angle_difference_deg"] < 0.2
    assert blue["within_rig_noise"] is True and block["within_rig_noise_all_classes"] is True
    assert 0.3 < block["rig_static_noise_cm"] < 3.0


def test_plausibility_and_ambiguities(synthetic):
    rows = synthetic["lines_rows"]
    p = plausibility(rows, PRIOR)
    # The blue pipette rests, the red slides 1 cm a frame: medians in between, no jumps.
    assert p["midpoint_speed_cm_per_frame"]["max"] == pytest.approx(1.0, abs=0.01)
    assert p["rows_over_twice_the_prior"] == 0 and p["midpoint_steps_over_10cm"] == 0
    assert p["row_segment_length_cm"]["median"] == pytest.approx(22.0, abs=0.01)
    assert p["tip_speed_cm_per_frame"]["max"] == pytest.approx(1.0, abs=0.01)
    assert p["tip_steps_over_10cm_fraction"] == 0.0
    assert p["held_rows"] == 4 and p["held_rows_with_resolved_butt"] == 4
    assert p["butt_to_hand_cm_while_held"]["median"] == pytest.approx(2.0, abs=0.01)
    _, rest_frames = rest_geometry(rows, None)
    amb = ambiguities(rows, rest_frames, synthetic["lines_metrics"])
    assert amb["near_duplicate_pairs"] == 1
    assert amb["pairs"][0]["tracks"] == ["pipette-001", "pipette-002"]
    assert amb["pairs"][0]["frames"] == len(FRAMES) and amb["pair_frames_at_rest_excluded"] == 0
    assert amb["frames_with_an_ambiguity"] == len(FRAMES)
    assert amb["id_switches"] == 3 and "colour-cross" in amb["id_switch_note"]
    # Both at rest: the same pair is excluded.
    both_rest = {tid: set(FRAMES) for tid in ("pipette-001", "pipette-002")}
    assert ambiguities(rows, both_rest, synthetic["lines_metrics"])["near_duplicate_pairs"] == 0


def test_colour_disagreement_names_the_track_and_its_slots(synthetic):
    c = colour_summary(synthetic["colour_rows"], synthetic["lines_rows"])
    assert c["available"] and c["tracks"] == 2 and c["agree"] == 1 and c["disagree"] == 1
    (d,) = c["disagreements"]
    assert d["track_id"] == "pipette-001" and d["colour_identity"] == "yellow"
    assert d["observed_class"] == "blue_pipette"
    assert any(s.startswith("T1/blue_pipette#0 x") for s in d["slots"])
    assert colour_summary(None, synthetic["lines_rows"])["available"] is False


def test_scoreboard_cli_writes_json_and_markdown_with_the_rule(synthetic, tmp_path):
    root = synthetic["root"]
    out = tmp_path / "board"
    controls = {
        "frames": [600, 899],
        "runs": {
            "normal": {
                "loo_residual_px": {"n": 10, "median": 5.0, "p90": 9.0},
                "ratio_to_normal": 1.0,
                "cells_fraction_of_normal": 1.0,
                "rises_clearly": None,
                "description": "the run as is",
            },
            "shipped-T5": {
                "loo_residual_px": {"n": 4, "median": 20.0, "p90": 40.0},
                "ratio_to_normal": 4.0,
                "cells_fraction_of_normal": 0.4,
                "rises_clearly": True,
                "description": "shipped pose",
            },
        },
        "verdict": "Both controls raise the LOO median by at least 1.5x.",
    }
    (tmp_path / "nc.json").write_text(json.dumps(controls))
    rc = main(
        [
            "lines-scoreboard",
            "--lines-dir",
            str(root / "arm-b-lines"),
            "--ext-dir",
            str(root / "arm-b-ext"),
            "--baseline-ext-dir",
            str(root / "baseline" / "tracks-ext"),
            "--observations",
            str(synthetic["obs_dir"]),
            "--clip-config",
            str(_clip_for(root)),
            "--rig",
            str(root / "rig.json"),
            "--prior",
            str(_prior_for(root)),
            "--stand-report",
            str(root / "stand_report.json"),
            "--negative-controls",
            str(tmp_path / "nc.json"),
            "--fpv-poses",
            str(FIXTURE_DIR / "fpv_poses.json"),
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    board = json.loads((out / "lines_scoreboard.json").read_text())
    text = (out / "lines_scoreboard.md").read_text()
    rule = board["rule"]["this_trial"]
    assert rule["held_pipette_class"] == "blue_pipette"
    assert rule["ids_baseline"] == 5 and rule["ids_lines"] == 1 and rule["ids_fall_at_least_5x"]
    assert rule["tip_error_under_2cm"] == "pending human anchors"
    assert rule["flat_rest_within_rig_noise"] is True
    assert board["ids"]["point_reproduces_baseline"]["blue_pipette"] is False
    assert board["ids"]["point_reproduces_baseline"]["red_pipette"] is True
    assert board["negative_controls"][0]["runs"]["shipped-T5"]["rises_clearly"] is True
    assert board["colour"]["disagree"] == 1
    for heading in (
        "## Ids per pipette class",
        "## Leave-one-camera-out residual",
        "## Length",
        "## Flat-rest geometry",
        "## Plausibility",
        "## Ambiguities",
        "## Colour vote",
        "## Negative controls",
        "## The pre-registered rule",
    ):
        assert heading in text
    assert "measured against" in text and "pending human anchors" in text
    # The other trial's board gives the both-trials read-out.
    other = json.loads(json.dumps(board))
    other["rule"]["this_trial"]["ids_fall_at_least_5x"] = False
    (tmp_path / "other.json").write_text(json.dumps(other))
    second = build_scoreboard(
        lines_tracks_dir=synthetic["lines_dir"],
        ext_tracks_dir=synthetic["ext_dir"],
        baseline_tracks_dir=synthetic["base_dir"],
        observations=synthetic["obs_dir"],
        camera_config=FIXTURE_DIR / "cameras.json",
        window=(FRAMES[0], FRAMES[-1] + 1),
        rig=json.loads((root / "rig.json").read_text()),
        prior=PRIOR,
        trial="P03_01_01",
        other_trial=other,
        fpv_poses=FIXTURE_DIR / "fpv_poses.json",
    )
    both = second["rule"]["both_trials"]
    assert both["ids_fall_at_least_5x"] is False and both["tip_error_under_2cm"].startswith(
        "pending"
    )
    assert "Both trials" in scoreboard_markdown(second)


def _clip_for(root: Path) -> Path:
    path = root / "clip.json"
    if not path.exists():
        path.write_text(
            json.dumps(
                {
                    "config_kind": "finebio_clip_config",
                    "clip_id": "fixture-window",
                    "trial": "P03_01_01",
                    "window": {"start_frame": FRAMES[0], "end_frame_exclusive": FRAMES[-1] + 1},
                    "views": ["T1", "T2", "T3", "T4", "T5", "fpv"],
                    "fixed_views": ["T1", "T2", "T3", "T4", "T5"],
                    "fpv_view": "fpv",
                    "camera_config": str(FIXTURE_DIR / "cameras.json"),
                    "containers": ["centrifuge"],
                    "probes": ["left_hand", "right_hand"],
                }
            )
        )
    return path


def _prior_for(root: Path) -> Path:
    path = root / "pipettes.json"
    if not path.exists():
        path.write_text(
            json.dumps(
                {
                    "length_cm": 23.0,
                    "length_spread_cm": 2.0,
                    "provenance": {
                        "per_trial": {"x": {"per_pipette_median_cm": {"blue_pipette": 22.0}}}
                    },
                }
            )
        )
    return path


def test_negative_controls_shift_and_shipped_pose_and_verdict(synthetic, tmp_path):
    root = synthetic["root"]
    out = tmp_path / "controls"
    window = (FRAMES[2], FRAMES[6])
    window_path, shift_path, counts = write_control_observations(
        synthetic["obs_dir"], out, window, "T3"
    )
    rows = [json.loads(line) for line in window_path.read_text().splitlines()]
    shifted = [json.loads(line) for line in shift_path.read_text().splitlines()]
    assert {r["frame_index"] for r in rows} == set(range(window[0], window[1] + 1))
    assert counts["window_rows"] == len(rows) == 5 * 8
    t3_before = sorted(r["frame_index"] for r in rows if r["view"] == "T3")
    t3_after = sorted(r["frame_index"] for r in shifted if r["view"] == "T3")
    assert t3_after == t3_before  # the same frame set, each row one frame later
    original = {(r["view"], r["frame_index"], r["slot"]): r for r in rows}
    for r in shifted:
        if r["view"] == "T3":
            src = original.get(("T3", r["frame_index"] - 1, r["slot"]))
            if src is not None:
                assert src["mask_axis_px"] == r["mask_axis_px"]
    others = [r for r in shifted if r["view"] != "T3"]
    assert len(others) == 5 * 6
    # The shipped-pose config replaces one view's pose and says so.
    config = shipped_camera_config(
        FIXTURE_DIR / "cameras.json",
        "T5",
        out / "cameras-shipped-T5.json",
        shipped_pose=([0.1, 0.2, 0.3], [1.0, 2.0, 3.0]),
    )
    doc = json.loads(config.read_text())
    assert doc["fixed"]["T5"]["provenance"] == "shipped"
    assert doc["fixed"]["T5"]["rvec"] == [0.1, 0.2, 0.3] and doc["fixed"]["T5"]["tvec"] == [1, 2, 3]
    assert (
        doc["fixed"]["T4"] == json.loads((FIXTURE_DIR / "cameras.json").read_text())["fixed"]["T4"]
    )
    assert "negative_control" in doc["provenance"]

    calls = []

    def fake_tracker(obs, cams, gates, output, extra, fpv_poses):
        calls.append((Path(obs).name, Path(cams).name, list(extra)))
        median = 5.0
        if "shipped" in Path(cams).name:
            median = 30.0
        elif "shift" in Path(obs).name:
            median = 6.0
        return {
            "per_class": {"pipette": {"tracks_born": 2}},
            "extensions": {
                "lines": {
                    "loo_residual_px": {"n": 20, "median": median, "p90": 2 * median},
                    "loo_angle_deg": {"n": 20, "median": 0.5, "p90": 1.0},
                    "frames": {"line": 10},
                }
            },
        }

    report = run_negative_controls(
        observations=synthetic["obs_dir"],
        camera_config=FIXTURE_DIR / "cameras.json",
        gates=root / "rig.json",
        tracker_flags=["--line-classes", "pipette"],
        output=out,
        frames=window,
        shipped_view="T5",
        shift_views=("T3", "T4"),
        shipped_pose=([0.1, 0.2, 0.3], [1.0, 2.0, 3.0]),
        run_tracker=fake_tracker,
    )
    assert [c[0] for c in calls] == [
        "observations-window.jsonl",
        "observations-window.jsonl",
        "observations-shift-T3.jsonl",
        "observations-shift-T4.jsonl",
    ]
    assert calls[1][1] == "cameras-shipped-T5.json" and all(
        c[2] == ["--line-classes", "pipette"] for c in calls
    )
    runs = report["runs"]
    assert runs["shipped-T5"]["ratio_to_normal"] == 6.0 and runs["shipped-T5"]["rises_clearly"]
    assert runs["shift-T3"]["ratio_to_normal"] == 1.2 and runs["shift-T3"]["rises_clearly"] is False
    assert runs["normal"]["rises_clearly"] is None
    assert "shift-T3" in report["verdict"] and "does not raise" in report["verdict"]
    assert (out / "negative_controls.json").is_file()
