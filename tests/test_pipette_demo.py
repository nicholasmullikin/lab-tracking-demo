"""Demo exports preserve clocks, display coordinates and abstentions."""

import json
import tarfile

import cv2
import numpy as np
import pytest
import rerun as rr

from battle import pipette_demo as demo


def test_display_transform_uses_rounded_dimensions():
    size = (1921, 1080)
    shown = demo.display_size(size, 960)
    np.testing.assert_allclose(demo.transform([[0, 0], size], size, shown), [[0, 0], shown])
    assert demo.display_size((320, 180), 960) == (320, 180)


def test_samples_keep_native_clock_and_both_endpoints():
    fps = 30000 / 1001
    samples = sorted(demo.sampled_frames(8492, fps))
    assert samples[0] == 0 and samples[-1] == 8491
    assert samples[2] == round(fps)
    assert set(np.diff(samples[:-1])) <= {14, 15}


@pytest.mark.parametrize("content", ["", '{"raw_frame": 0}', '{"raw_frame": 1}\n'])
def test_incomplete_or_noncontiguous_inputs_are_rejected(tmp_path, content):
    path = tmp_path / "geometry.jsonl"
    path.write_text(content)
    with pytest.raises(ValueError):
        demo.prefix(path, 1, "raw_frame")


def test_abstention_clears_previous_arrow_and_missing_fit_clears_shaft(monkeypatch):
    calls = []
    monkeypatch.setattr(rr, "log", lambda path, value, **kwargs: calls.append((path, value)))
    line = {"endpoints": [[0, 0, 0], [10, 0, 0]]}
    demo.log_world({"line": line}, 1)
    demo.log_world({"line": line}, None)
    demo.log_world({"line": None}, None)
    final_calls = calls[-4:]
    assert {path for path, value in final_calls if isinstance(value, rr.Clear)} == {
        "world/blue",
        "world/blue/shaft",
        "world/blue/dispensing",
        "world/blue/end",
    }
    arrows = [(path, value) for path, value in calls if isinstance(value, rr.Arrows3D)]
    assert len(arrows) == 1
    assert all(isinstance(value, rr.Clear) for path, value in calls if path == "world/blue")


def test_missing_camera_observations_clear_each_cached_leaf(monkeypatch):
    calls = []
    monkeypatch.setattr(rr, "log", lambda path, value, **kwargs: calls.append((path, value)))
    rows = {label: {"mask": None, "axis": {"axis_px": None}} for label in demo.LABELS}
    demo.log_camera(np.zeros((108, 192, 3), np.uint8), rows, "fpv", None)
    cleared = {path for path, value in calls if isinstance(value, rr.Clear)}
    for label in demo.LABELS:
        assert f"cameras/fpv/{label}/mask" in cleared
        assert f"cameras/fpv/{label}/axis" in cleared
    assert "cameras/fpv/projection/dispensing" in cleared
    masks = [
        value
        for path, value in calls
        if path.endswith("/mask") and isinstance(value, rr.EncodedImage)
    ]
    assert len(masks) == 4
    for value in masks:
        pixels = cv2.imdecode(
            np.array(value.blob.as_arrow_array().to_pylist()[0], np.uint8), cv2.IMREAD_UNCHANGED
        )
        assert not pixels[..., 3].any()


def test_mask_png_preserves_label_rgb_colour(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(rr, "log", lambda path, value, **kwargs: calls.append((path, value)))
    path = tmp_path / "mask.png"
    cv2.imwrite(str(path), np.full((10, 20), 255, np.uint8))
    rows = {label: {"mask": str(path), "axis": {"axis_px": None}} for label in demo.LABELS}
    demo.log_camera(np.zeros((10, 20, 3), np.uint8), rows, "fpv", None)
    masks = [
        value
        for path, value in calls
        if path.endswith("/mask") and isinstance(value, rr.EncodedImage)
    ]
    for value, colour in zip(masks, demo.COLOURS, strict=True):
        pixels = cv2.imdecode(
            np.array(value.blob.as_arrow_array().to_pylist()[0], np.uint8), cv2.IMREAD_UNCHANGED
        )
        assert tuple(pixels[0, 0]) == (*colour[::-1], 80)
    labels = [
        value for path, value in calls if path.endswith("/label") and isinstance(value, rr.Points2D)
    ]
    assert len(labels) == 4  # Partial masks keep their identities even without a fitted shaft.
    assert [value.labels.as_arrow_array().to_pylist()[0] for value in labels] == [
        demo.DISPLAY_LABELS[label] for label in demo.LABELS
    ]
    assert not any(isinstance(value, rr.LineStrips2D) for path, value in calls)


def test_stage_review_must_match_requested_frames(tmp_path):
    (tmp_path / "baseline").mkdir()
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "reviewed_seeds": True,
                "labels": list(demo.LABELS),
            }
        )
    )
    (tmp_path / "baseline/review-300.json").write_text('{"stage_frames": 299}')
    with pytest.raises(ValueError, match="Review does not match"):
        demo.validate(tmp_path, "baseline", 300)


def test_saved_bundle_is_self_contained_and_checksums_cover_embedded_data(tmp_path):
    from rerun.experimental import RrdReader

    source = tmp_path / "source"
    (source / "baseline").mkdir(parents=True)
    video = source / "fixture.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 36))
    assert writer.isOpened()
    for level in (40, 100):
        writer.write(np.full((36, 64, 3), level, np.uint8))
    writer.release()
    mask = source / "mask.png"
    cv2.imwrite(str(mask), np.full((36, 64), 255, np.uint8))
    config = {
        "reviewed_seeds": True,
        "labels": list(demo.LABELS),
        "native_fps": 30,
        "views": {
            view: {"video": str(video), "frame_count": 2, "size_wh": [64, 36]}
            for view in demo.VIEWS
        },
    }
    (source / "config.json").write_text(json.dumps(config))
    geometry = []
    for frame in range(2):
        views = {
            view: {
                label: {"mask": str(mask), "axis": {"axis_px": [[12, 20], [40, 20]]}}
                for label in demo.LABELS
            }
            for view in demo.VIEWS
        }
        blue = {
            "views": {view: views[view][demo.LABELS[0]] for view in demo.VIEWS},
            "line": {"endpoints": [[0, 0, 0], [10, 0, 0]]} if frame == 0 else None,
            "direction": {"tip_end": 1 if frame == 0 else None, "contributions": {}},
            "loo": [],
        }
        geometry.append({"raw_frame": frame, "views": views, "blue": blue})
    (source / "baseline/geometry.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in geometry)
    )
    for view in demo.VIEWS:
        folder = source / "baseline" / view
        folder.mkdir()
        objects = [
            {"label": label, "object_id": f"sam3-{i:02d}"} for i, label in enumerate(demo.LABELS)
        ]
        (folder / "observations.jsonl").write_text(
            "".join(
                json.dumps({"analysis_frame_index": frame, "objects": objects}) + "\n"
                for frame in range(2)
            )
        )
    (source / "baseline/review-2.json").write_text('{"stage_frames": 2, "frames": {}}')
    (source / "baseline/review-1.json").write_text(
        '{"stage_frames": 1, "frames": {"0": {"identity": "inspected at earlier stage"}}}'
    )
    (source / "baseline/review-1-mask-hashes.json").write_text(
        json.dumps({str(mask): demo.digest(mask)})
    )
    output = tmp_path / "bundle"
    demo.build(source, "baseline", 2, output)
    review = json.loads((output / "review.json").read_text())
    assert review["frames"]["0"]["identity"] == "inspected at earlier stage"
    assert review["frame_review_sources"]["0"] == "review-1.json"
    cv2.imwrite(str(mask), np.zeros((36, 64), np.uint8))
    with pytest.raises(ValueError, match="Reviewed mask changed"):
        demo.build(source, "baseline", 2, tmp_path / "changed-mask-bundle")
    (output / "viewer-review.json").write_text('{"verified": true}')
    demo.package_bundle(output)
    # Source dependencies become unavailable; the recordings still contain their pixels.
    source.rename(tmp_path / "unavailable-source")
    reader = RrdReader(output / "overview.rrd")
    schema = str(reader.store().schema())
    assert "EncodedImage:blob" in schema and "Image:buffer" not in schema
    assert "Arrows3D:vectors" in schema
    assert "raw_frame" in schema and "seconds" in schema
    assert list(reader.blueprints())
    with tarfile.open(output.with_suffix(".tar.gz")) as archive:
        assert "bundle/overview.rrd" in archive.getnames()
        assert "bundle/SHA256SUMS" in archive.getnames()
        assert "bundle/viewer-review.json" in archive.getnames()
    for line in (output / "SHA256SUMS").read_text().splitlines():
        expected, filename = line.split("  ", 1)
        assert demo.digest(output / filename) == expected
