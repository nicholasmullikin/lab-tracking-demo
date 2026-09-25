"""Object-object events (p5-events) on synthetic tracks: the hysteresis and dwell, container
volumes from a rig point and box sizes, `contained` with the lid cycles cross-table, `held`
with the co-motion gate, the pipette tip from masks, and the CLI end to end."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from finebio_fixtures import FIXTURE_DIR

from battle.finebio_cameras import cameras_from_config, read_camera_config
from battle.finebio_events import (
    EventParams,
    Hysteresis,
    ObjectEvent,
    Volume,
    container_volumes,
    cycle_cross_table,
    cycles_in_window,
    detect_events,
    lid_closed_at,
    main,
    moved_over_window,
    pipette_tip,
    plate_volume,
)
from battle.multiview_schemas import Track3D, read_jsonl

TARGETS = ("micro_tube", "50ml_tube", "blue_pipette", "cell_culture_plate")


def _row(
    frame: int,
    track_id: str,
    cls: str,
    position: tuple[float, float, float],
    *,
    state: str = "observed",
    slots: dict[str, str] | None = None,
) -> Track3D:
    views = tuple(slots) if slots else ("T1", "T2", "T3")
    return Track3D(
        frame_index=frame,
        track_id=track_id,
        object_class=cls,
        position_cm=position,
        uncertainty_cm=1.0,
        support_views=views,
        support_slots=slots or {v: f"{cls}#0" for v in views},
        state=state,
        confidence=0.5,
        abstain=False,
    )


def _volume(name: str, centre=(20.0, -20.0), half=10.0, height=20.0) -> Volume:
    return Volume(name, name, centre, half, half, -height, 2.0)


def test_hysteresis_needs_the_dwell_and_ignores_flicker():
    h = Hysteresis(enter=0.0, exit=3.0, dwell=3)
    # Two frames inside then out again: no start.
    assert h.feed(0, -1.0) is None and h.feed(1, -1.0) is None and h.feed(2, 5.0) is None
    assert not h.inside
    # Three consecutive frames inside start the episode at the first of them.
    assert h.feed(3, -1.0) is None and h.feed(4, -0.5) is None
    assert h.feed(5, -0.2) == ("start", 3)
    # Between the thresholds nothing changes; a single frame past exit does not end it.
    assert h.feed(6, 1.5) is None and h.feed(7, 4.0) is None and h.feed(8, 1.0) is None
    assert h.inside
    # Three frames at or over the exit threshold end it at the frame before the run began.
    assert h.feed(9, 3.0) is None and h.feed(10, 3.5) is None
    assert h.feed(11, 9.0) == ("end", 8)
    assert not h.inside


def test_volume_signed_distance_and_lid_helpers():
    v = _volume("centrifuge")
    assert v.signed_distance((20.0, -20.0, -5.0)) < 0  # inside, 5 cm above the bench
    assert v.signed_distance((20.0, -20.0, -25.0)) > 0  # above the volume
    assert np.isclose(v.signed_distance((33.0, -20.0, -5.0)), 3.0)  # 3 cm outside in x
    assert np.isclose(v.signed_distance((20.0, -20.0, 4.0)), 2.0)  # 2 cm below the tolerance
    intervals = [(0, 1099), (1176, 1228), (3224, 3311)]
    assert lid_closed_at(intervals, 1200) and not lid_closed_at(intervals, 1228)
    assert cycles_in_window(intervals, 600, 4200) == [(1176, 1228), (3224, 3311)]
    assert cycles_in_window(intervals, 0, 1300) == [(1176, 1228)]  # closed at start: no cycle


def test_contained_episode_ends_when_the_track_is_lost_and_the_cross_table_names_successors():
    frames = list(range(600, 700))
    rows: list[Track3D] = []
    # A tube walks into the centrifuge at 620, sits inside, coasts from 660 and is lost at 670.
    for f in frames:
        if f < 620:
            rows.append(_row(f, "micro_tube-001", "micro_tube", (40.0, -20.0, -5.0)))
        elif f < 660:
            rows.append(_row(f, "micro_tube-001", "micro_tube", (20.0, -20.0, -6.0)))
        elif f < 670:
            rows.append(
                _row(f, "micro_tube-001", "micro_tube", (20.0, -20.0, -6.0), state="coasting")
            )
        elif f == 670:
            rows.append(_row(f, "micro_tube-001", "micro_tube", (20.0, -20.0, -6.0), state="lost"))
    # The lid closes at 655 and opens at 690; a successor is born inside at 692.
    for f in range(692, 700):
        rows.append(_row(f, "micro_tube-002", "micro_tube", (20.5, -20.0, -6.0)))
    # A pipette over the centrifuge is not containable.
    for f in range(630, 650):
        rows.append(_row(f, "blue_pipette-001", "blue_pipette", (20.0, -20.0, -10.0)))
    params = EventParams(dwell_frames=3)
    episodes, strip, events = detect_events(
        rows,
        containers=[_volume("centrifuge")],
        hands={},
        proximity_targets=[],
        tips={},
        side_cams={},
        lid_intervals=[(655, 690)],
        frames=frames,
        targets=TARGETS,
        params=params,
    )
    contained = [e for e in episodes if e.kind == "contained"]
    assert [(e.track_id, e.start_frame, e.end_frame, e.end_reason) for e in contained] == [
        ("micro_tube-001", 620, 669, "track_lost"),
        ("micro_tube-002", 692, 699, "window_end"),
    ]
    first = contained[0]
    assert first.frames_lid_closed == 669 - 655 + 1
    assert first.tracker_state_frames == {"observed": 40, "coasting": 10}
    assert not any(e.track_id == "blue_pipette-001" for e in episodes)
    starts = [e for e in events if e.payload["phase"] == "start"]
    assert starts[0].frame_index == 620 and starts[0].payload["lid_closed"] is False
    assert all(e.payload["model_output"] for e in events)
    assert all(isinstance(e, ObjectEvent) for e in events)
    # The strip: contained while the lid is closed, nothing once the track is lost.
    by_frame = {s.frame_index: s for s in strip}
    assert by_frame[660].lid_closed and by_frame[660].contained[0]["track_id"] == "micro_tube-001"
    assert by_frame[675].contained == [] and by_frame[695].contained[0]["track_id"] == (
        "micro_tube-002"
    )
    table = cycle_cross_table(episodes, rows, [(655, 690)])
    assert table[0]["episodes"] == 2 and table[0]["ended_while_closed"] == 1
    entry = table[0]["contained_episodes"][0]
    assert entry["phase"] == "ended_while_closed" and entry["inside_while_closed_frames"] == 15
    assert [s["track_id"] for s in entry["successors_after_opening"]] == ["micro_tube-002"]
    assert table[0]["contained_episodes"][1]["phase"] == "after_opening"


def test_held_needs_proximity_and_co_motion_then_exits_on_distance():
    frames = list(range(0, 60))
    rows: list[Track3D] = []
    hands = {}
    for f in frames:
        # The hand rests over a rack for 20 frames, then carries the tube away, then leaves it.
        if f < 20:
            hand = np.array([0.0, 0.0, -12.0])
            tube = (0.0, 0.0, -4.0)  # 8 cm under the hand, not moving: occluded, not held
        elif f < 40:
            hand = np.array([1.5 * (f - 20), 0.0, -15.0])
            tube = (1.5 * (f - 20), 0.0, -8.0)  # moves with the hand, 7 cm below it
        else:
            hand = np.array([80.0, 40.0, -15.0])
            tube = (30.0, 0.0, -4.0)  # put down; the hand goes away
        hands[f] = [("right_hand-001", "right_hand", hand)]
        rows.append(_row(f, "50ml_tube-001", "50ml_tube", tube))
    params = EventParams(dwell_frames=3, held_radius_cm=12.0, held_exit_margin_cm=5.0)
    episodes, strip, _ = detect_events(
        rows,
        containers=[],
        hands=hands,
        proximity_targets=[],
        tips={},
        side_cams={},
        lid_intervals=[],
        frames=frames,
        targets=TARGETS,
        params=params,
    )
    held = [e for e in episodes if e.kind == "held"]
    assert len(held) == 1
    ep = held[0]
    assert ep.target == "right_hand" and 20 <= ep.start_frame <= 23
    assert ep.end_frame == 39 and ep.end_reason == "exit"
    assert not any(s.held for s in strip if s.frame_index < 20)
    # Without the motion gate the resting tube under the hand would count as held.
    loose = EventParams(dwell_frames=3, held_min_motion_cm=0.0)
    episodes_loose, _, _ = detect_events(
        rows,
        containers=[],
        hands=hands,
        proximity_targets=[],
        tips={},
        side_cams={},
        lid_intervals=[],
        frames=frames,
        targets=TARGETS,
        params=loose,
    )
    assert [e.start_frame for e in episodes_loose if e.kind == "held"][0] == 0
    assert moved_over_window([np.zeros(3), np.array([3.0, 0.0, 0.0])], 2.0)
    assert not moved_over_window([np.zeros(3)], 2.0)


def test_pipette_tip_from_two_side_views_else_the_track_point():
    config = read_camera_config(FIXTURE_DIR / "cameras.json")
    cams = cameras_from_config(config)
    side = {v: c for v, c in cams.items() if v != "T5"}
    tip = np.array([-30.0, 0.0, -2.0])
    pixels = {v: tuple(side[v].project(tip)[0]) for v in ("T1", "T2", "T3")}
    row = _row(700, "blue_pipette-001", "blue_pipette", (-30.0, 0.0, -14.0))
    tips = {(v, 700, "blue_pipette#0"): pixels[v] for v in pixels}
    point, from_masks = pipette_tip(row, tips, side, gate_px=30.0)
    assert from_masks and np.allclose(point, tip, atol=0.2)
    # One view only: the track point stands in.
    point, from_masks = pipette_tip(
        row, {("T1", 700, "blue_pipette#0"): pixels["T1"]}, side, gate_px=30.0
    )
    assert not from_masks and np.allclose(point, row.position_cm)
    # Inconsistent pixels (one view's bottom is somewhere else) fail the gate.
    bad = dict(tips)
    bad[("T2", 700, "blue_pipette#0")] = (pixels["T2"][0] + 300.0, pixels["T2"][1])
    point, from_masks = pipette_tip(row, bad, side, gate_px=30.0)
    assert not from_masks

    # Proximity fires when the tip enters the plate volume; the strip records the tip source.
    plate = Volume("cell_culture_plate", "cell_culture_plate", (-30.0, 0.0), 6.0, 6.0, -8.0, 2.0)
    frames = list(range(700, 720))
    rows = [_row(f, "blue_pipette-001", "blue_pipette", (-30.0, 0.0, -14.0)) for f in frames]
    tips_all = {(v, f, "blue_pipette#0"): pixels[v] for f in frames if f >= 705 for v in pixels}
    episodes, strip, _ = detect_events(
        rows,
        containers=[],
        hands={},
        proximity_targets=[plate],
        tips=tips_all,
        side_cams=side,
        lid_intervals=[],
        frames=frames,
        targets=TARGETS,
        params=EventParams(dwell_frames=3, tip_fallback_margin_cm=0.0),
    )
    prox = [e for e in episodes if e.kind == "proximity"]
    assert len(prox) == 1 and prox[0].start_frame == 705 and prox[0].target == "cell_culture_plate"
    assert prox[0].tip_from_masks_frames == 15
    assert {s.frame_index: s for s in strip}[710].proximity[0]["tip_from_masks"] is True


def test_container_and_plate_volumes_from_rig_point_and_box_sizes():
    config = read_camera_config(FIXTURE_DIR / "cameras.json")
    cams = cameras_from_config(config)
    rig = {
        "static": [
            {
                "class": "centrifuge",
                "views": ["T1", "T2", "T3"],
                "point_cm": [20.0, -20.0, -8.0],
                "height_cm": 8.0,
            },
            {
                "class": "unknown_rack",
                "views": ["T1"],
                "point_cm": [0.0, 0.0, -1.0],
                "height_cm": 1.0,
            },
        ]
    }
    # A 200 px wide box at ~1 m depth on a ~1000 px focal length is about 20 cm wide.
    sizes = {("T1", "centrifuge"): (200.0, 200.0), ("T2", "centrifuge"): (200.0, 300.0)}
    volumes = container_volumes(rig, ["centrifuge", "unknown_rack"], cams, sizes)
    assert [v.name for v in volumes] == ["centrifuge"]  # no boxes for the rack: no volume
    vol = volumes[0]
    assert 5.0 < vol.half_x_cm < 20.0 and vol.z_top == -22.0 and vol.z_bottom == 2.0
    assert vol.provenance["height_source"] == "CONTAINER_HEIGHT_CM"
    assert set(vol.provenance["half_extent_per_view_cm"]) == {"T1", "T2"}
    scaled = container_volumes(rig, ["centrifuge"], cams, sizes, scale=2.0)[0]
    assert np.isclose(scaled.half_x_cm, 2.0 * vol.half_x_cm)
    rows = [_row(f, "plate-001", "cell_culture_plate", (-30.0, 0.0, -1.0)) for f in range(5)]
    plate = plate_volume(rows, cams, {("T3", "cell_culture_plate"): (150.0, 100.0)}, above_cm=6.0)
    assert plate is not None and plate.centre_xy == (-30.0, 0.0) and plate.z_top == -7.0
    assert plate_volume([], cams, {}, above_cm=6.0) is None


def test_cli_end_to_end_on_a_synthetic_arm(tmp_path: Path):
    config = read_camera_config(FIXTURE_DIR / "cameras.json")
    cams = cameras_from_config(config)
    arm = tmp_path / "b-box-decode-arm"
    (arm / "tracks").mkdir(parents=True)
    frames = list(range(600, 700))
    rows = []
    for f in frames:
        rows.append(_row(f, "micro_tube-001", "micro_tube", (20.0, -20.0, -6.0)))
        rows.append(_row(f, "right_hand-001", "right_hand", (60.0, 30.0, -20.0)))
    with (arm / "tracks" / "tracks.jsonl").open("w") as handle:
        for r in rows:
            handle.write(r.model_dump_json(exclude_none=True) + "\n")
    with (arm / "observations.jsonl").open("w") as handle:
        for view, cam in cams.items():
            centre = cam.project(np.array([20.0, -20.0, -8.0]))[0]
            for f in frames[:10]:
                handle.write(
                    json.dumps(
                        {
                            "view": view,
                            "frame_index": f,
                            "slot": "centrifuge#0",
                            "object_class": "centrifuge",
                            "detector_score": 0.9,
                            "box_xyxy_px": [
                                float(centre[0] - 100),
                                float(centre[1] - 100),
                                float(centre[0] + 100),
                                float(centre[1] + 100),
                            ],
                            "pose_valid": True,
                            "source": "detector",
                        }
                    )
                    + "\n"
                )
    rig = {
        "static": [
            {
                "class": "centrifuge",
                "views": list(cams),
                "point_cm": [20.0, -20.0, -8.0],
                "height_cm": 8.0,
            }
        ],
        "gates": {"association_px": 30.0},
    }
    (tmp_path / "rig.json").write_text(json.dumps(rig))
    clip = {
        "config_kind": "finebio_clip_config",
        "clip_id": "synthetic",
        "trial": "P03_01_01",
        "views": [*config.fixed, "fpv"],
        "fixed_views": list(config.fixed),
        "fpv_view": "fpv",
        "window": {"start_frame": 600, "end_frame_exclusive": 700, "frame_count": 100},
        "window_camera_config": str(FIXTURE_DIR / "cameras.json"),
        "targets": list(TARGETS),
        "containers": ["centrifuge"],
        "probes": ["left_hand", "right_hand"],
    }
    (tmp_path / "clip.json").write_text(json.dumps(clip))
    trials = {
        "trials": [
            {
                "trial": "P03_01_01",
                "centrifuge": {"lid_closed_intervals_all": [[0, 100], [650, 680]]},
            }
        ]
    }
    (tmp_path / "trials.json").write_text(json.dumps(trials))
    output = tmp_path / "events"
    assert (
        main(
            [
                "--arm-dir",
                str(arm),
                "--config",
                str(tmp_path / "clip.json"),
                "--rig",
                str(tmp_path / "rig.json"),
                "--trials",
                str(tmp_path / "trials.json"),
                "--output",
                str(output),
                "--dwell-frames",
                "3",
            ]
        )
        == 0
    )
    events = list(read_jsonl(output / "events.jsonl", ObjectEvent))
    assert [e.kind for e in events] == ["contained", "contained"]
    assert events[0].payload["target"] == "centrifuge" and events[1].payload["end_reason"] == (
        "window_end"
    )
    summary = json.loads((output / "events_summary.json").read_text())
    assert summary["cycles_in_window"] == [[650, 680]]
    assert summary["kinds"]["contained"]["episodes"] == 1
    assert summary["kinds"]["held"]["episodes"] == 0
    assert summary["lid_closed_frames"] == 30
    assert summary["centrifuge_cycles_vs_contained"][0]["same_id_after_opening"] == 1
    assert summary["label"].startswith("model output")
    strip = [json.loads(line) for line in (output / "events_strip.jsonl").read_text().splitlines()]
    assert len(strip) == 100 and strip[60]["lid_closed"] is True
    assert (output / "events.md").read_text().startswith("# Events, arm (b)")
    assert (output / "episodes.jsonl").read_text().count("\n") == 1
