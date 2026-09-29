"""Synthetic checks of the story-media renderer: manifest parsing, overlay compositing on a
four-frame fixture, grid tiling, the top-down world panel from two fixture tracks, and the
fallback path.  No real data; the fixture clip is written with ffmpeg."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest
from conftest import require_executable

from battle import story_media as sm

MANIFEST_DOC = {
    "schema": sm.SCHEMA,
    "defaults": {"fps": 8, "width": 320},
    "entries": [
        {
            "id": "one",
            "day": "2026-09-08",
            "kind": "overlay",
            "caption": "A caption",
            "sources": [{"video": "v.mp4", "observations": "o.jsonl"}],
            "frames": {"start": 0, "stop": 4, "stride": 1},
        },
        {
            "id": "two",
            "day": "2026-09-09",
            "kind": "still",
            "caption": "Another",
            "sources": [{"image": "a.png"}],
            "fps": 12,
        },
    ],
}


def _write_manifest(tmp_path: Path, doc: dict) -> Path:
    path = tmp_path / "media.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_load_manifest_applies_defaults_and_derives_output_names(tmp_path: Path) -> None:
    entries = sm.load_manifest(_write_manifest(tmp_path, MANIFEST_DOC))
    assert [e.output_name for e in entries] == ["2026-09-08-one.gif", "2026-09-09-two.png"]
    assert entries[0].fps == 8 and entries[0].width == 320
    assert entries[0].frames == sm.FrameRange(0, 4, 1)
    assert entries[1].fps == 12 and entries[1].frames is None
    assert entries[0].max_bytes == sm.DEFAULT_MAX_BYTES


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"id": "Bad Id"}, "bad entry id"),
        ({"day": "Sep 8"}, "day must be"),
        ({"kind": "movie"}, "kind"),
        ({"caption": ""}, "caption"),
        ({"frames": {"start": 5, "stop": 2}}, "bad frame range"),
    ],
)
def test_load_manifest_refuses_bad_entries(tmp_path: Path, patch: dict, message: str) -> None:
    doc = json.loads(json.dumps(MANIFEST_DOC))
    doc["entries"][0].update(patch)
    with pytest.raises(ValueError, match=message):
        sm.load_manifest(_write_manifest(tmp_path, doc))


def test_load_manifest_refuses_duplicate_and_wrong_schema(tmp_path: Path) -> None:
    doc = json.loads(json.dumps(MANIFEST_DOC))
    doc["entries"].append(dict(doc["entries"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        sm.load_manifest(_write_manifest(tmp_path, doc))
    doc = json.loads(json.dumps(MANIFEST_DOC))
    doc["schema"] = "other"
    with pytest.raises(ValueError, match="schema"):
        sm.load_manifest(_write_manifest(tmp_path, doc))


# --------------------------------------------------------------------------- fixture clip


def _fixture_run(root: Path, *, frames: int = 4, size: tuple[int, int] = (64, 36)) -> Path:
    """A run directory: a flat-colour clip whose frame k is grey level 40 + 40k, one object
    per frame with a moving box and a mask PNG under masks/."""
    require_executable("ffmpeg", "the story-media fixture clip")
    run = root / "run"
    (run / "masks").mkdir(parents=True)
    w, h = size
    frame_dir = root / "frames"
    frame_dir.mkdir()
    for k in range(frames):
        cv2.imwrite(str(frame_dir / f"{k:05d}.png"), np.full((h, w, 3), 40 + 40 * k, np.uint8))
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            "30",
            "-i",
            str(frame_dir / "%05d.png"),
            "-c:v",
            "libx264",
            "-qp",
            "0",
            "-pix_fmt",
            "yuv444p",
            str(run / "input.mp4"),
        ],
        check=True,
    )
    with (run / "observations.jsonl").open("w", encoding="utf-8") as handle:
        for k in range(frames):
            mask = np.zeros((h, w), np.uint8)
            x0 = 8 + 8 * k
            mask[10:26, x0 : x0 + 16] = 255
            cv2.imwrite(str(run / "masks" / f"{k:06d}_00.png"), mask)
            row = {
                "view_id": "fixture",
                "analysis_frame_index": k,
                "source_seconds": k / 30,
                "objects": [
                    {
                        "object_id": "sam3-00",
                        "label": "part",
                        "confidence": 1.0,
                        "box": {"x": x0 / w, "y": 10 / h, "width": 16 / w, "height": 16 / h},
                        "mask": {"uri": f"masks/{k:06d}_00.png", "storage": "x", "format": "png"},
                    },
                    {
                        "object_id": "sam3-01",
                        "label": "other",
                        "confidence": 1.0,
                        "box": {"x": 0.05, "y": 0.6, "width": 0.1, "height": 0.2},
                        "mask": None,
                    },
                ],
            }
            handle.write(json.dumps(row) + "\n")
    return run


def test_read_observations_and_frames_round_trip(tmp_path: Path) -> None:
    run = _fixture_run(tmp_path)
    observations = sm.read_observations(run / "observations.jsonl")
    assert sorted(observations) == [0, 1, 2, 3]
    part = observations[2][0]
    assert part.label == "part" and part.mask_uri == "masks/000002_00.png"
    assert observations[2][1].mask_uri is None
    frames = sm.read_frames(run / "input.mp4", [1, 3])
    assert sorted(frames) == [1, 3]
    assert abs(int(frames[3][:, :, 1].mean()) - 160) <= 3


def test_composite_tile_tints_the_mask_and_leaves_the_rest(tmp_path: Path) -> None:
    run = _fixture_run(tmp_path)
    observations = sm.read_observations(run / "observations.jsonl")
    frame = sm.read_frames(run / "input.mp4", [0])[0]
    tile = sm.composite_tile(
        frame,
        observations[0],
        size=(64, 36),
        run=run,
        colour_of=lambda o: (0, 0, 255) if o.label == "part" else None,
        show_boxes=False,
    )
    assert tile.shape == (36, 64, 3)
    inside = tile[18, 16].astype(int)  # centre of the frame-0 mask
    outside = tile[30, 50].astype(int)
    assert inside[2] > inside[0] + 40, "the mask pixel should be tinted red"
    assert abs(int(outside[0]) - 40) <= 4 and abs(int(outside[2]) - 40) <= 4


def test_auto_crop_covers_every_box_in_the_clip(tmp_path: Path) -> None:
    run = _fixture_run(tmp_path)
    source = sm.load_tile_source(
        {"video": str(run / "input.mp4"), "observations": str(run / "observations.jsonl")},
        tmp_path,
    )
    x0, y0, x1, y1 = sm.auto_crop(source, [0, 1, 2, 3])
    assert x0 <= 8 / 64 and x1 >= (8 + 24 + 16) / 64
    assert y0 <= 10 / 36 and y1 >= 0.8


def test_tile_grid_places_tiles_and_pads_a_short_row() -> None:
    tiles = [np.full((4, 6, 3), v, np.uint8) for v in (1, 2, 3)]
    grid = sm.tile_grid(tiles, columns=2, gap=0)
    assert grid.shape == (8, 12, 3)
    assert grid[0, 0, 0] == 1 and grid[0, 6, 0] == 2 and grid[4, 0, 0] == 3
    assert grid[4, 6, 0] == 18  # background fill under the missing fourth tile
    with pytest.raises(ValueError, match="share one size"):
        sm.tile_grid([tiles[0], np.zeros((5, 6, 3), np.uint8)], columns=2)
    with pytest.raises(ValueError, match="no tiles"):
        sm.tile_grid([], columns=1)


def test_overlay_entry_renders_a_gif_of_the_four_frames(tmp_path: Path) -> None:
    _fixture_run(tmp_path)
    doc = {
        "schema": sm.SCHEMA,
        "entries": [
            {
                "id": "fixture",
                "day": "2026-09-08",
                "kind": "overlay",
                "caption": "Fixture",
                "sources": [{"video": "run/input.mp4", "observations": "run/observations.jsonl"}],
                "frames": {"start": 0, "stop": 4, "stride": 1},
                "width": 128,
                "fps": 4,
            }
        ],
    }
    entry = sm.load_manifest(_write_manifest(tmp_path, doc))[0]
    record = sm.render_entry(entry, tmp_path, tmp_path / "out")
    output = tmp_path / "out" / "2026-09-08-fixture.gif"
    assert output.is_file() and record["fallback"] is None
    assert record["duration_seconds"] == 1.0 and record["size_bytes"] == output.stat().st_size
    assert record["detail"]["tiles"][0]["video"] == "run/input.mp4"
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames,width",
            "-of",
            "csv=p=0",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    width, count = probe.stdout.strip().split(",")
    assert int(width) == 128 and int(count) == 4


def test_world_panel_draws_two_tracks_and_the_closed_lid(tmp_path: Path) -> None:
    tracks = tmp_path / "tracks.jsonl"
    rows = [
        {
            "frame_index": 600,
            "track_id": "micro_tube-001",
            "object_class": "micro_tube",
            "position_cm": [20.0, -20.0, -5.0],
            "state": "contained",
            "support_slots": {"T1": "micro_tube#0"},
        },
        {
            "frame_index": 600,
            "track_id": "50ml_tube-002",
            "object_class": "50ml_tube",
            "position_cm": [-30.0, 10.0, -3.0],
            "state": "coasting",
            "support_slots": {},
        },
        {
            "frame_index": 601,
            "track_id": "micro_tube-001",
            "object_class": "micro_tube",
            "position_cm": [20.0, -20.0, -5.0],
            "state": "observed",
            "support_slots": {},
        },
    ]
    tracks.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    by_frame = sm.read_tracks(tracks, [600])
    assert sorted(by_frame) == [600] and len(by_frame[600]) == 2
    scene = sm.WorldScene(
        static=[{"class": "centrifuge", "point_cm": [20.0, -20.0, -8.0]}],
        containers=[
            {"name": "centrifuge", "centre_xy_cm": [20.0, -20.0], "half_x_cm": 10, "half_y_cm": 10}
        ],
        cameras={"T1": (-40.0, 70.0, -80.0)},
        lid_intervals=[(590, 610)],
        extent=(-50.0, -40.0, 50.0, 40.0),
    )
    panel = sm.render_world_panel(
        by_frame[600], scene, size=(320, 160), raw_frame=600, highlight=["micro_tube-001"]
    )
    assert panel.shape == (160, 320, 3)
    # The contained tube is drawn as a filled dot in its track colour at (20, -20).
    px = int(
        round((320 - 100 * min(300 / 100, 140 / 80)) / 2 + (20 + 50) * min(300 / 100, 140 / 80))
    )
    py = int(
        round((160 - 80 * min(300 / 100, 140 / 80)) / 2 + (-20 + 40) * min(300 / 100, 140 / 80))
    )
    assert tuple(int(v) for v in panel[py, px]) == sm.colour_for("micro_tube-001")
    # The lid is closed at raw 600: the centrifuge box is drawn red and labelled.
    assert scene.lid_closed(600) and not scene.lid_closed(620)
    red = (panel[:, :, 2].astype(int) - panel[:, :, 1].astype(int)) > 100
    assert red.any()
    # A coasting track is a hollow ring: its centre pixel stays background.
    cx = int(round((320 - 100 * 1.75) / 2 + (-30 + 50) * 1.75))
    cy = int(round((160 - 80 * 1.75) / 2 + (10 + 40) * 1.75))
    assert tuple(int(v) for v in panel[cy, cx]) != sm.colour_for("50ml_tube-002")


def test_world_panel_draws_a_line_track_as_a_segment_with_its_tip(tmp_path: Path) -> None:
    tracks = tmp_path / "tracks.jsonl"
    rows = [
        {
            "frame_index": 700,
            "track_id": "pipette-007",
            "object_class": "pipette",
            "observed_class": "blue_pipette",
            "position_cm": [0.0, 0.0, -5.0],
            "state": "held",
            "support_slots": {},
            "endpoints_cm": [[-20.0, 0.0, -5.0], [20.0, 0.0, -5.0]],
            "tip_resolved": True,
        },
        {
            "frame_index": 700,
            "track_id": "pipette-008",
            "object_class": "pipette",
            "position_cm": [0.0, 20.0, -5.0],
            "state": "observed",
            "support_slots": {},
            "endpoints_cm": [[-10.0, 20.0, -5.0], [10.0, 20.0, -5.0]],
            "tip_resolved": False,
        },
    ]
    tracks.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    by_frame = sm.read_tracks(tracks, [700])
    held, bare = by_frame[700]
    assert held.endpoints == ((-20.0, 0.0, -5.0), (20.0, 0.0, -5.0))
    assert held.tip == (-20.0, 0.0, -5.0) and held.shown_class == "blue_pipette"
    assert bare.tip is None and bare.shown_class == "pipette"
    scene = sm.WorldScene(extent=(-50.0, -40.0, 50.0, 40.0))
    panel = sm.render_world_panel(by_frame[700], scene, size=(320, 160), raw_frame=700)
    scale = min(300 / 100, 140 / 80)
    ox, oy = (320 - 100 * scale) / 2, (160 - 80 * scale) / 2
    tip = (int(round(oy + 40 * scale)), int(round(ox + 30 * scale)))
    mid = (int(round(oy + 40 * scale)), int(round(ox + 50 * scale)))
    assert tuple(int(v) for v in panel[tip]) == sm.TIP
    colour = np.array(sm.colour_for("pipette-007"), dtype=int)
    assert np.abs(panel[mid].astype(int) - colour).max() < 60, (
        "the segment is drawn in the id colour"
    )
    # The unresolved track has a segment and no white tip at its end.
    end = (int(round(oy + 60 * scale)), int(round(ox + 40 * scale)))
    assert tuple(int(v) for v in panel[end]) != sm.TIP
    counts = sm._line_id_counter(by_frame, [700], 0, "blue_pipette")
    assert counts == [1]


def test_tile_transform_maps_frame_pixels_onto_the_tile_and_projects_points() -> None:
    from battle.finebio_cameras import Camera

    transform = sm.tile_transform((360, 640), (320, 180), None)
    assert transform.scale == 0.5 and transform.pad == (0, 0)
    assert transform.to_tile(640, 360) == (320, 180)
    cropped = sm.tile_transform((360, 640), (320, 180), [0.25, 0.25, 0.75, 0.75])
    assert cropped.to_tile(cropped.x0, cropped.y0) == cropped.pad
    cam = Camera(
        "T1",
        np.array([[100.0, 0.0, 320.0], [0.0, 100.0, 180.0], [0.0, 0.0, 1.0]]),
        np.zeros(5),
        np.zeros(3),
        np.array([0.0, 0.0, 50.0]),
        (640, 360),
    )
    assert sm._project(cam, (0.0, 0.0, 0.0)) == pytest.approx((320.0, 180.0))
    assert sm._project(cam, (10.0, 0.0, 50.0)) == pytest.approx((330.0, 180.0))
    assert sm._project(cam, (0.0, 0.0, -60.0)) is None


def test_camera_centres_invert_the_extrinsics() -> None:
    config = {"fixed": {"T1": {"rvec": [0.0, 0.0, 0.0], "tvec": [-10.0, 20.0, 80.0]}}}
    assert sm.camera_centres(config)["T1"] == pytest.approx((10.0, -20.0, -80.0))


def test_missing_source_falls_back_to_the_contact_sheet_and_is_reported(tmp_path: Path) -> None:
    sheet = tmp_path / "sheet.png"
    cv2.imwrite(str(sheet), np.full((30, 90, 3), 200, np.uint8))
    doc = {
        "schema": sm.SCHEMA,
        "entries": [
            {
                "id": "gone",
                "day": "2026-09-08",
                "kind": "overlay",
                "caption": "Gone",
                "sources": [{"video": "missing/input.mp4", "observations": "missing/o.jsonl"}],
                "frames": {"start": 0, "stop": 4, "stride": 1},
                "width": 180,
                "fallback": "sheet.png",
            },
            {
                "id": "nothing",
                "day": "2026-09-09",
                "kind": "still",
                "caption": "Nothing",
                "sources": [{"image": "missing.png"}],
                "width": 180,
            },
        ],
    }
    entries = sm.load_manifest(_write_manifest(tmp_path, doc))
    records = [sm.render_entry(e, tmp_path, tmp_path / "out") for e in entries]
    first, second = records
    assert first["output"] == "out/2026-09-08-gone.png" and first["fallback"]["used"] == "sheet.png"
    assert (
        "video missing" in first["fallback"]["reason"] and first["fallback"]["placeholder"] is False
    )
    image = cv2.imread(str(tmp_path / first["output"]))
    assert image.shape[1] == 180 and not (tmp_path / "out" / "2026-09-08-gone.gif").exists()
    assert second["fallback"]["placeholder"] is True
    assert "no fallback image configured" in second["fallback"]["reason"]
    assert (tmp_path / second["output"]).is_file()
    doc_out = sm.write_output_manifest(
        tmp_path / "out" / "manifest.json",
        records,
        config=tmp_path / "media.json",
        root=tmp_path,
        merge=False,
    )
    assert doc_out["coverage"]["2026-09-08"]["fallbacks"] == ["gone"]
    assert doc_out["coverage"]["2026-09-09"]["placeholders"] == ["nothing"]
    written = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert [e["id"] for e in written["entries"]] == ["gone", "nothing"]


def test_output_manifest_merges_a_single_entry_rerender(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    base = [
        {"id": "a", "day": "2026-09-08", "output": "a.gif", "size_bytes": 1, "fallback": None},
        {"id": "b", "day": "2026-09-09", "output": "b.gif", "size_bytes": 1, "fallback": None},
    ]
    sm.write_output_manifest(path, base, config=tmp_path / "m.json", root=tmp_path, merge=False)
    update = [
        {"id": "b", "day": "2026-09-09", "output": "b.gif", "size_bytes": 2, "fallback": None}
    ]
    doc = sm.write_output_manifest(
        path, update, config=tmp_path / "m.json", root=tmp_path, merge=True
    )
    assert [(e["id"], e["size_bytes"]) for e in doc["entries"]] == [("a", 1), ("b", 2)]


def test_label_matches_exact_and_prefix() -> None:
    assert sm.label_matches("blue_pipette#3", ["blue_pipette#*"])
    assert sm.label_matches("chassis", ["chassis", "cabin"])
    assert not sm.label_matches("chassis_x", ["chassis"])


def test_caption_bar_wraps_a_long_caption_onto_a_second_line() -> None:
    short = sm.caption_bar(300, "short")
    long = sm.caption_bar(300, "a caption that is far too long to fit on one line of a narrow bar")
    assert short.shape[0] == sm.CAPTION_HEIGHT and long.shape[0] == 2 * sm.CAPTION_HEIGHT
