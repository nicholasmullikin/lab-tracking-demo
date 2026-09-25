"""The review recording builder (p5-viewer) on a synthetic arm: the entity layout (the seven
cross-checks, states as entity paths, masks per slot, the negative control), the three presets
validated against the recording, the storyboard, and the small helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from finebio_fixtures import FIXTURE_DIR
from PIL import Image

from battle import finebio_confidence, finebio_events
from battle.finebio_cameras import cameras_from_config, read_camera_config
from battle.finebio_viewer import (
    ArmData,
    StoryItem,
    TrackRow,
    _track_label,
    build,
    derived_dir_names,
    load_arm,
    parse_arm_dirs,
    pick_confidence_drop,
    read_track_rows,
    storyboard_markdown,
)
from battle.multiview_schemas import Track3D

TARGETS = ["micro_tube", "50ml_tube", "blue_pipette", "cell_culture_plate"]
FRAMES = list(range(600, 700))


def _row(frame: int, track_id: str, cls: str, position, *, state="observed", slots=None) -> Track3D:
    views = tuple(slots) if slots else ("T1", "T2", "T3")
    return Track3D(
        frame_index=frame,
        track_id=track_id,
        object_class=cls,
        position_cm=position,
        uncertainty_cm=1.0,
        support_views=views,
        support_slots=slots or {v: f"{cls}#0" for v in views},
        residual_px={v: 4.0 for v in views},
        state=state,
        confidence=0.5,
        abstain=False,
    )


def _synthetic_arm(root: Path, label: str = "b") -> Path:
    """One arm directory: a tube that enters the centrifuge and is lost while the lid is closed,
    a successor, a plate, a hand; detector rows for the volumes; one SAM3 mask on T1."""
    config = read_camera_config(FIXTURE_DIR / "cameras.json")
    cams = cameras_from_config(config)
    arm = root / f"{label}-synthetic-arm"
    (arm / "tracks").mkdir(parents=True)
    rows: list[Track3D] = []
    for f in FRAMES:
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
        if f >= 692:
            rows.append(_row(f, "micro_tube-002", "micro_tube", (20.5, -20.0, -6.0)))
        rows.append(_row(f, "cell_culture_plate-001", "cell_culture_plate", (-30.0, 0.0, -1.0)))
        rows.append(_row(f, "right_hand-001", "right_hand", (60.0, 30.0, -20.0)))
    with (arm / "tracks" / "tracks.jsonl").open("w") as handle:
        for r in rows:
            handle.write(r.model_dump_json(exclude_none=True) + "\n")
    with (arm / "tracks" / "residuals.jsonl").open("w") as handle:
        for r in rows:
            for v in r.support_views:
                handle.write(
                    json.dumps(
                        {
                            "frame_index": r.frame_index,
                            "track_id": r.track_id,
                            "view": v,
                            "residual_px": 4.0,
                            "gate_px": 30.0,
                            "slot": r.support_slots[v],
                            "source": "detector",
                            "confirmed": True,
                        }
                    )
                    + "\n"
                )
    (arm / "tracks" / "events.jsonl").write_text("")
    (arm / "tracks" / "identity_metrics.json").write_text(
        json.dumps({"tracks_born": 4, "tracks_lost": 1, "per_class": {}})
    )
    worker = arm.parent / f"{label}-worker" / "T1" / "run"
    (worker / "masks").mkdir(parents=True)
    mask = np.zeros((48, 64), dtype=np.uint8)
    mask[10:30, 20:50] = 255
    for f in FRAMES:
        Image.fromarray(mask, mode="L").save(worker / "masks" / f"{f - 600:06d}_03.png")
    (arm / "observations_summary.json").write_text(json.dumps({"worker_runs": {"T1": str(worker)}}))
    with (arm / "observations.jsonl").open("w") as handle:
        for view, cam in cams.items():
            centre = cam.project(np.array([20.0, -20.0, -8.0]))[0]
            plate = cam.project(np.array([-30.0, 0.0, -1.0]))[0]
            for f in FRAMES:
                for cls, c, size in (
                    ("centrifuge", centre, 100.0),
                    ("cell_culture_plate", plate, 60.0),
                    ("micro_tube", centre, 12.0),
                ):
                    handle.write(
                        json.dumps(
                            {
                                "view": view,
                                "frame_index": f,
                                "slot": f"{cls}#0",
                                "object_class": cls,
                                "detector_score": 0.9,
                                "box_xyxy_px": [
                                    float(c[0] - size),
                                    float(c[1] - size),
                                    float(c[0] + size),
                                    float(c[1] + size),
                                ],
                                "pose_valid": True,
                                "source": "detector",
                            }
                        )
                        + "\n"
                    )
                if view == "T1":
                    handle.write(
                        json.dumps(
                            {
                                "view": "T1",
                                "frame_index": f,
                                "slot": "cell_culture_plate#0",
                                "object_class": "cell_culture_plate",
                                "detector_score": 0.9,
                                "box_xyxy_px": [
                                    float(plate[0] - 60),
                                    float(plate[1] - 60),
                                    float(plate[0] + 60),
                                    float(plate[1] + 60),
                                ],
                                "mask_bbox_px": [20.0, 10.0, 50.0, 30.0],
                                "mask_centroid_px": [35.0, 20.0],
                                "mask_area_px": 600,
                                "sam3_object_score": 0.95,
                                "pose_valid": True,
                                "source": "sam3_decode",
                                "provenance": {
                                    "object_id": "sam3-03",
                                    "detector_box_iou": 0.9,
                                    "worker_label": "cell_culture_plate#0",
                                },
                            }
                        )
                        + "\n"
                    )
    (arm / "measures.md").write_text("# measures\n")
    (arm / "occlusion_inventory.md").write_text("# inventory\n")
    return arm


def _configs(root: Path, cams) -> tuple[Path, Path, Path]:
    rig = {
        "static": [
            {
                "class": "centrifuge",
                "views": list(cams),
                "point_cm": [20.0, -20.0, -8.0],
                "height_cm": 8.0,
                "loo_px": {v: 5.0 for v in cams},
            }
        ],
        "gates": {
            "association_px": 30.0,
            "handoff_px": 50.0,
            "floor_px": 15.0,
            "cap_px": 80.0,
            "image_width_px": 1920,
            "birth_min_fixed_views": 3,
            "birth_fixed_views_with_fpv": 2,
            "formula": {},
            "inputs": {},
            "clock_offset_frames": {},
        },
        "hands": {
            "left_hand": [
                {"frame": f, "point_cm": [0.0, 20.0, -3.0], "views": ["T1", "T2", "T3"]}
                for f in FRAMES[:50]
            ]
        },
        "fpv": [
            {"frame": f, "plate_fixed_to_fpv": {"residual_px": 10.0, "inside_box": True}}
            for f in FRAMES
        ],
        "clock": {
            "T1": {
                "left_hand": {
                    "best_offset": 0,
                    "residual_px_at_best": 5.0,
                    "residual_px_at_0": 5.0,
                    "informative": False,
                }
            }
        },
        "negative_control": {
            "T5": {
                "camera_id": 6,
                "marker_rms_px": {"marker_pnp": 0.7, "shipped": 93.7},
                "centre_cm": {"marker_pnp": [0, 0, -90], "shipped": [6, 0, -90], "distance": 6.4},
                "static_loo_px": {
                    "marker_pnp": {"centrifuge": 5.0},
                    "shipped": {"centrifuge": 95.0},
                    "median": {"marker_pnp": 5.0, "shipped": 95.0},
                },
                "hands_px": {
                    "marker_pnp": {"left_hand": {"median": 8.0}},
                    "shipped": {"left_hand": {"median": 50.0}},
                },
            }
        },
    }
    rig_path = root / "rig.json"
    rig_path.write_text(json.dumps(rig))
    (root / "rig.md").write_text("# rig\n")
    clip = {
        "config_kind": "finebio_clip_config",
        "clip_id": "synthetic",
        "trial": "P03_01_01",
        "views": [*cams, "fpv"],
        "fixed_views": list(cams),
        "fpv_view": "fpv",
        "window": {"start_frame": 600, "end_frame_exclusive": 700, "frame_count": 100},
        "window_camera_config": str(FIXTURE_DIR / "cameras.json"),
        "view_sizes": {v: [1920, 1080] for v in cams} | {"fpv": [1920, 1440]},
        "targets": TARGETS,
        "containers": ["centrifuge"],
        "probes": ["left_hand", "right_hand"],
        "annotated_frames_in_window": {"six_view": [650]},
    }
    clip_path = root / "clip.json"
    clip_path.write_text(json.dumps(clip))
    trials_path = root / "trials.json"
    trials_path.write_text(
        json.dumps(
            {
                "trials": [
                    {
                        "trial": "P03_01_01",
                        "centrifuge": {"lid_closed_intervals_all": [[0, 100], [650, 680]]},
                    }
                ]
            }
        )
    )
    return rig_path, clip_path, trials_path


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("viewer")
    config = read_camera_config(FIXTURE_DIR / "cameras.json")
    cams = cameras_from_config(config)
    arm = _synthetic_arm(root)
    rig_path, clip_path, trials_path = _configs(root, cams)
    finebio_confidence.run(arm, arm / "confidence", arm="b")
    finebio_events.run(
        arm,
        clip_path,
        rig_path,
        arm / "events",
        trials_path=trials_path,
        params=finebio_events.EventParams(dwell_frames=3),
        arm="b",
    )
    return {"root": root, "arm": arm, "rig": rig_path, "clip": clip_path, "trials": trials_path}


def test_helpers():
    assert parse_arm_dirs("a=/x, b=/y") == {"a": Path("/x"), "b": Path("/y")}
    assert derived_dir_names("tracks") == ("confidence", "events")
    assert derived_dir_names("tracks-ext") == ("confidence-ext", "events-ext")
    with pytest.raises(ValueError):
        parse_arm_dirs("nolabel")
    row = TrackRow(
        600,
        "t-1",
        "micro_tube",
        (0.0, 0.0, 0.0),
        1.0,
        ("T1",),
        {},
        "contained",
        {},
        (),
        container_id="centrifuge",
        group_size=3,
    )
    assert _track_label(row, True) == "t-1 [contained] in centrifuge x3 !"
    assert (
        _track_label(TrackRow(600, "t-2", "x", (0, 0, 0), 1.0, (), {}, "observed", {}, ()), False)
        == "t-2"
    )
    text = storyboard_markdown(
        [StoryItem(1, "s", "title", 1234, "note", ("world/x",), {})], offset=600, trial="P"
    )
    assert "| 1 | s | title | 1234 | 634 |" in text and "`world/x`" in text


def test_light_rows_and_arm_loading(synthetic):
    rows = read_track_rows(synthetic["arm"] / "tracks" / "tracks.jsonl")
    assert rows[0].track_id == "micro_tube-001" and rows[0].position == (40.0, -20.0, -5.0)
    arm = load_arm("b", synthetic["arm"], None)
    assert arm.tracks_dir.name == "tracks"
    assert ("micro_tube-001", 630) in arm.confidence
    assert arm.strip[660]["lid_closed"] is True and arm.strip[660]["contained"]
    assert arm.worker_runs["T1"].is_dir()
    assert arm.events_summary["kinds"]["contained"]["episodes"] == 2


def test_confidence_drop_picker_on_synthetic_series():
    rows = [
        TrackRow(
            600 + k,
            "t-1",
            "50ml_tube",
            (0.0, 0.0, 0.0),
            1.0,
            ("T2",),
            {"T2": "50ml_tube#0"},
            "observed",
            {},
            (),
        )
        for k in range(60)
    ]
    confidence = {("t-1", 600 + k): (0.8 if k < 30 else 0.2, k >= 30) for k in range(60)}
    signals = {
        ("T2", 600 + k, "50ml_tube#0"): {"detector_box_iou": 0.9 if k < 30 else 0.1}
        for k in range(60)
    }
    arm = ArmData(
        label="c",
        arm_dir=Path("."),
        tracks_dir=Path("."),
        rows=rows,
        by_frame={},
        confidence=confidence,
        confidence_md=None,
        strip={},
        episodes=[],
        events_summary=None,
        events_md=None,
        measures_md=None,
        inventory_md=None,
        identity=None,
        worker_runs={},
        events_by_frame={},
    )
    items = pick_confidence_drop(arm, signals, preferred_slots=(("T2", "50ml_tube#0"),))
    assert len(items) == 1
    assert items[0].frame == 630 and items[0].numbers["slot"] == "50ml_tube#0"
    assert items[0].numbers["confidence_before"] > items[0].numbers["confidence_after"]
    assert items[0].numbers["detector_box_iou_after"] < 0.2
    # No IoU evidence: no story (a drop alone is not shown to be a failure).
    assert pick_confidence_drop(arm, {}) == []


def test_build_writes_recording_presets_and_storyboard(synthetic, tmp_path: Path):
    output = tmp_path / "review"
    index = build(
        clip_path=synthetic["clip"],
        arm_dirs={"b": synthetic["arm"]},
        rig_path=synthetic["rig"],
        seeds_dir=None,
        output=output,
        mask_arm="b",
        secondary_mask_arm=None,
        mask_every=4,
        trials_path=synthetic["trials"],
        preset_dir=tmp_path / "presets",
        skip_video=True,
        fpv_poses=FIXTURE_DIR / "fpv_poses.json",
    )
    assert (output / "review.rrd").is_file()
    assert index["rerun_cli"]["verify"]["returncode"] == 0
    paths = set((output / "entity_paths.txt").read_text().splitlines())
    # The seven cross-checks and the negative control as standing entities.
    for path in (
        "/world/T1/markers_projected",
        "/world/T1/fpv_camera_centre",
        "/world/static_objects",
        "/world/T1/static_reprojected",
        "/world/left_hand",
        "/world/T1/left_hand_triangulated",
        "/checks/clock_scan",
        "/checks/loo/cell_culture_plate/T1",
        "/checks/handoff/cell_culture_plate",
        "/checks/handoff/cell_culture_plate_inside",
        "/checks/negative_control",
        "/world/T5/markers_projected_shipped",
        "/world/T5_shipped",
    ):
        if path.endswith("markers_projected") or path.endswith("markers_projected_shipped"):
            continue  # marker corners need the shipped calibration archive under data/
        if path == "/world/T5_shipped":
            continue  # the shipped camera-6 pose also lives under data/
        assert path in paths, path
    # States are entity paths; masks per slot under the arm; the volumes; the series.
    for path in (
        "/world/tracks/b/observed",
        "/world/tracks/b/coasting",
        "/world/tracks/b/trails",
        "/world/T1/masks/b/cell_culture_plate-0",
        "/world/T1/tracks/b",
        "/world/T1/detector",
        "/world/containers/centrifuge",
        "/world/containers/centrifuge_lid",
        "/events/lid_closed",
        "/events/b/contained",
        "/events/b/active",
        "/events/b/log",
        "/confidence/b/micro_tube",
        "/confidence/b/abstain_fraction",
        "/checks/residuals/b/T1",
        "/checks/occlusion/b/coasting_tracks",
        "/checks/measures/b",
        "/checks/identity/b",
        "/storyboard/guide",
        "/storyboard/marks",
        "/storyboard/current",
    ):
        assert path in paths, path
    assert "/world/T1/video" not in paths and "/world/T1/video_asset" not in paths
    # Masks every 4th frame plus the storyboard frames.
    assert index["recording"]["masks_logged"] >= 25
    # Presets validated against the entity tree, and copied for committing.
    assert index["presets"]["ok"], index["presets"]
    assert {p["preset"] for p in index["presets"]["presets"]} == {
        "world.rbl",
        "cameras.rbl",
        "evidence.rbl",
    }
    for name in ("world", "cameras", "evidence"):
        assert (output / f"{name}.rbl").is_file()
        assert (tmp_path / "presets" / f"finebio_{name}.rbl").is_file()
    # The storyboard found the centrifuge story from the events cross-table.
    story = json.loads((output / "storyboard.json").read_text())
    titles = [it["title"] for it in story]
    assert "a tube goes into the centrifuge" in titles and "the lid closes" in titles
    entry = next(it for it in story if it["title"] == "the lid closes")
    assert entry["raw_frame"] == 650 and entry["proxy_frame"] == 50
    successor = next(
        it for it in story if it["story"] == "centrifuge" and "successor" in it["title"]
    )
    assert successor["raw_frame"] == 692 and successor["numbers"]["track_id"] == "micro_tube-001"
    assert (output / "storyboard.md").read_text().startswith("# Storyboard")
    assert index["cycles_in_window"] == [[650, 680]]
    assert index["containers"][0]["name"] == "centrifuge"
    assert index["mask_arm"] == "b" and index["secondary_mask_arm"] is None
