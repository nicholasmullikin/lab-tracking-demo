"""Projection streams, merge quality gates and the steep single-view fit on the rig."""

import json

import numpy as np
import pytest
from finebio_fixtures import load_preflight_fixtures

from battle.finebio_reprojection import (
    generate_requests,
    gravity_segment,
    merge_rows,
    perpendicular_residual,
    projected_box,
    quality_rejection,
)
from battle.muggled_worker import parse_box_stream
from battle.multiview_schemas import FineBioObservation, Track3D
from battle.multiview_tracks import LinePrior, _obs


@pytest.fixture
def cameras():
    return load_preflight_fixtures().fixed_cameras()


def track(frame=1798, cls="blue_pipette"):
    return Track3D(
        frame_index=frame,
        track_id="pipette-008",
        object_class="pipette",
        observed_class=cls,
        position_cm=(5, 2, -1),
        uncertainty_cm=1,
        support_views=("T1", "T4"),
        state="held",
        confidence=0.8,
        abstain=False,
        endpoints_cm=((-5, 0, -1), (15, 4, -1)),
        direction=(1, 0.2, 0),
        tip_resolved=True,
        tip_confidence=0.99,
        orientation_retrofit_sign=1,
    )


def obs(cam, view="T2", frame=1798, offset=0):
    axis = cam.project(np.asarray(track().endpoints_cm)) + [0, offset]
    lo, hi = axis.min(axis=0) - 5, axis.max(axis=0) + 5
    return FineBioObservation(
        view=view,
        frame_index=frame,
        slot="blue_pipette#pipette-008",
        object_class="blue_pipette",
        mask_axis_px=axis.tolist(),
        mask_elongation=8,
        mask_width_px=10,
        mask_axis_residual_px=2,
        mask_area_px=300,
        mask_bbox_px=(*lo, *hi),
        mask_centroid_px=tuple(axis.mean(axis=0)),
        pose_valid=True,
        source="sam3_decode",
    )


def test_stream_projects_to_missing_view_with_dense_slots_and_raw_frame_offset(cameras, tmp_path):
    report = generate_requests(
        [track()],
        [obs(cameras[v], v) for v in ("T1", "T4")],
        lambda frame: cameras,
        views=("T1", "T2", "T4", "fpv"),
        start_frame=1798,
        gate_px=30,
        prior=LinePrior(23, 2),
        output=tmp_path,
    )
    assert report["per_view"]["T2"]["requests"] == 1
    assert report["per_view"]["T1"]["requests"] == 0
    assert report["per_view"]["T4"]["requests"] == 0
    assert report["per_view"]["fpv"]["requests"] == 0
    assert report["held_fraction_touched"] == 1
    frames, labels = parse_box_stream((tmp_path / "T2.jsonl").read_text())
    assert labels == ("blue_pipette#pipette-008",) and frames[0][0]["slot"] == 0
    box = frames[0][0]["box_xyxy_px"]
    pixels = cameras["T2"].project(np.asarray(track().endpoints_cm))
    assert np.all(pixels >= box[:2]) and np.all(pixels <= box[2:])


def test_extension_requires_confidence_and_does_not_double_add_the_tip(cameras, tmp_path):
    prior = LinePrior(20, 2, bare={"blue_pipette": 20}, tip={"blue_tip": 6})
    row = track().model_copy(update={"tip_attached": True, "tip_class": "blue_tip"})
    for name, probability in [("confident", 0.99), ("weak", 0.6), ("opposite", 0.0)]:
        candidate = row.model_copy(update={"tip_confidence": probability})
        if name == "opposite":
            candidate = candidate.model_copy(update={"orientation_retrofit_sign": -1})
        generate_requests(
            [candidate],
            [],
            lambda frame: {"T2": cameras["T2"]},
            views=("T2",),
            start_frame=1798,
            gate_px=30,
            prior=prior,
            output=tmp_path / name,
        )
    a = json.loads((tmp_path / "confident/requests.jsonl").read_text())
    b = json.loads((tmp_path / "weak/requests.jsonl").read_text())
    opposite = json.loads((tmp_path / "opposite/requests.jsonl").read_text())
    assert opposite["projected_axis_px"] == a["projected_axis_px"]
    assert np.linalg.norm(np.diff(a["projected_axis_px"], axis=0)) > np.linalg.norm(
        np.diff(b["projected_axis_px"], axis=0)
    )
    ends = np.asarray(row.endpoints_cm)
    full = np.stack(
        [ends[1] + (ends[0] - ends[1]) * 26 / np.linalg.norm(ends[0] - ends[1]), ends[1]]
    )
    generate_requests(
        [row.model_copy(update={"endpoints_cm": full.tolist()})],
        [],
        lambda frame: {"T2": cameras["T2"]},
        views=("T2",),
        start_frame=1798,
        gate_px=30,
        prior=prior,
        output=tmp_path / "already_full",
    )
    full_request = json.loads((tmp_path / "already_full/requests.jsonl").read_text())
    assert np.allclose(full_request["projected_axis_px"], cameras["T2"].project(full))


def test_shift_control_is_60_pixels_perpendicular_to_the_projected_shaft(cameras):
    cam = cameras["T2"]
    normal, pixels = projected_box(cam, np.asarray(track().endpoints_cm), 10)
    shifted, _ = projected_box(cam, np.asarray(track().endpoints_cm), 10, shift_px=60)
    a, b = np.asarray(normal), np.asarray(shifted)
    assert np.linalg.norm(b[:2] - a[:2]) == pytest.approx(60, abs=0.002)
    assert abs(np.dot(b[:2] - a[:2], np.diff(pixels, axis=0)[0])) < 1


def test_merge_keeps_original_bytes_and_flags_an_elongated_disagreeing_mask(cameras, tmp_path):
    cam = cameras["T2"]
    row = obs(cam)
    box, pixels = projected_box(cam, np.asarray(track().endpoints_cm), 30)
    request = {
        "view": "T2",
        "raw_frame": 1798,
        "label": row.slot,
        "track_id": "pipette-008",
        "box_xyxy_px": box,
        "projected_axis_px": pixels,
    }
    source = tmp_path / "old"
    source.mkdir()
    detector = row.model_copy(
        update={"source": "detector", "mask_axis_px": None, "box_xyxy_px": row.mask_bbox_px}
    )
    original = ("  " + detector.model_dump_json(exclude_none=True) + "  \n").encode()
    (source / "observations.jsonl").write_bytes(original)
    (source / "plunger_ends.jsonl").write_text('{"cached":true}\n')
    displaced = row.model_copy(
        update={"mask_axis_px": tuple(tuple(np.asarray(p) + [0, 100]) for p in row.mask_axis_px)}
    )
    result = merge_rows(
        source, [displaced], [request], lambda f: cameras, output=tmp_path / "merged", gate_px=30
    )
    assert result["counts"]["accepted"] == 1 and result["counts"]["disagrees"] == 1
    data = (tmp_path / "merged/observations.jsonl").read_bytes()
    assert data.startswith(original)
    added = FineBioObservation.model_validate_json(data[len(original) :])
    assert added.provenance["prompt_source"] == "reprojection"
    assert added.provenance["reprojection_disagrees"] is True
    assert _obs(added).axis_obs().weight == 0.25
    assert (tmp_path / "merged/plunger_ends.jsonl").read_bytes() == (
        source / "plunger_ends.jsonl"
    ).read_bytes()
    # An existing SAM3 row for this class prevents any appended replacement.
    (source / "observations.jsonl").write_text(row.model_dump_json(exclude_none=True) + "\n")
    result = merge_rows(
        source, [row], [request], lambda f: cameras, output=tmp_path / "duplicate", gate_px=30
    )
    assert result["counts"]["existing_class_row"] == 1
    assert (tmp_path / "duplicate/observations.jsonl").read_bytes() == (
        source / "observations.jsonl"
    ).read_bytes()


def test_fill_fraction_control_is_rejected_even_with_a_good_axis(cameras):
    row = obs(cameras["T2"])
    box = (0, 0, 100, 100)
    assert (
        quality_rejection(row.model_copy(update={"mask_area_px": 10000}), box)[0]
        == "fills_prompt_box"
    )
    assert (
        quality_rejection(row.model_copy(update={"mask_axis_px": None}), box)[0] == "not_elongated"
    )
    assert (
        quality_rejection(row.model_copy(update={"mask_axis_residual_px": 26}), box)[0]
        == "axis_residual"
    )
    assert quality_rejection(row, box)[0] is None


def test_vertical_single_view_segment_is_recovered_within_one_centimetre(cameras):
    cam = cameras["T2"]
    truth = np.array([[5, 2, -25], [5, 2, -2]], dtype=float)
    predicted = truth + [2, 1, 0]
    axis = cam.project(truth)
    fitted = gravity_segment(cam, axis, predicted, 23)
    assert fitted is not None
    assert np.max(np.linalg.norm(fitted - truth, axis=1)) < 1
    pinned = gravity_segment(cam, axis, predicted, 23, hand_point=truth[0])
    assert pinned is not None and np.max(np.linalg.norm(pinned - truth, axis=1)) < 1
    shallow = np.array([[-5, 2, -2], [18, 2, -2]], dtype=float)
    assert gravity_segment(cam, cam.project(shallow), shallow, 23) is None
    assert perpendicular_residual(cam, axis, cam.project(fitted)) < 1


def test_coverage_uses_frozen_detector_and_held_frames_and_distinct_cameras(cameras):
    from battle.finebio_lines_scoreboard import reprojection_coverage

    row = obs(cameras["T2"])
    detector = row.model_copy(
        update={"source": "detector", "detector_score": 0.9, "box_xyxy_px": row.mask_bbox_px}
    )
    missing = detector.model_copy(update={"frame_index": 1799})
    before = [detector, missing, row, row.model_copy(update={"slot": "duplicate"})]
    disagree = row.model_copy(update={"view": "T1", "provenance": {"reprojection_disagrees": True}})
    # A new frame and a compact mask cannot change the frozen denominator or axis count.
    outside = row.model_copy(update={"frame_index": 1800})
    compact = row.model_copy(update={"frame_index": 1799, "mask_elongation": 1})
    baseline = track().model_copy(update={"support_slots": {"T2": row.slot}})
    updated = baseline.model_copy(update={"support_slots": {"T2": row.slot, "T1": row.slot}})
    report = reprojection_coverage(
        before,
        [*before, disagree, outside, compact],
        [baseline],
        "blue_pipette",
        (1798, 1801),
        after_lines=[updated],
    )
    assert report["before"]["detector_frames"]["frames"] == 2
    assert report["before"]["detector_frames"]["two_or_more_fraction"] == 0
    assert report["after"]["detector_frames"]["two_or_more_fraction"] == 0.5
    assert report["after"]["detector_frames"]["two_or_more_without_disagreement_fraction"] == 0
    assert report["after"]["baseline_held_frames"]["frames"] == 1
    assert report["after"]["baseline_held_frames"]["two_or_more_fraction"] == 1
    assert report["before"]["baseline_held_supported"]["two_or_more_fraction"] == 0
    assert report["after"]["baseline_held_supported"]["two_or_more_fraction"] == 1
    assert report["after"]["baseline_held_supported"]["two_or_more_without_disagreement"] == 0


def test_tracker_single_view_prior_hook_keeps_a_shallow_line_out(cameras):
    from battle.multiview_tracks import MultiviewTracker, Track, TrackerParams

    params = TrackerParams(line_classes=("blue_pipette",), line_single_view_prior=True)
    tracker = MultiviewTracker(cameras, lambda frame: None, params, line_prior=LinePrior(23, 2))
    truth = np.array([[5, 2, -25], [5, 2, -2]], dtype=float)
    row = obs(cameras["T2"]).model_copy(update={"mask_axis_px": cameras["T2"].project(truth)})
    t = Track("pipette-008", "pipette", truth.mean(axis=0) + [2, 1, 0], 1, "observed", 0, 0)
    t.set_endpoints(truth + [2, 1, 0])
    assert tracker._gravity_line_update(t, cameras["T2"], _obs(row))
    assert np.max(np.linalg.norm(t.endpoints_cm - truth, axis=1)) < 1
    horizontal = np.array([[-5, 2, -2], [18, 2, -2]], dtype=float)
    t.set_endpoints(horizontal)
    assert not tracker._gravity_line_update(t, cameras["T2"], _obs(row))
    assert np.allclose(t.endpoints_cm, horizontal)


def test_box_filling_mask_has_a_good_axis_but_is_rejected_before_merge(cameras):
    from battle.finebio_observations import mask_axis_measurements, mask_measurements

    mask = np.zeros((500, 500), dtype=np.uint8)
    mask[100:400, 100:120] = 1
    bbox, centroid, area = mask_measurements(mask)
    axis = mask_axis_measurements(mask)
    assert axis.elongation > 10 and axis.residual_px < 1
    row = obs(cameras["T2"]).model_copy(
        update={
            "mask_area_px": area,
            "mask_bbox_px": bbox,
            "mask_centroid_px": centroid,
            "mask_axis_px": axis.axis_px,
            "mask_elongation": axis.elongation,
            "mask_axis_residual_px": axis.residual_px,
        }
    )
    assert quality_rejection(row, (100, 100, 120, 400)) == ("fills_prompt_box", 1)


def test_tiny_behind_camera_and_mostly_outside_segments_have_no_prompt(cameras):
    cam = cameras["T2"]
    point = np.array([5, 2, -10], dtype=float)
    assert projected_box(cam, np.stack([point, point + [0.01, 0, 0]]), 10) is None
    behind = cam.centre + cam.R.T @ np.array([0, 0, -10.0])
    assert projected_box(cam, np.stack([behind, behind + [10, 0, 0]]), 10) is None
    outside = cam.centre + cam.R.T @ np.array([1000, 0, 100.0])
    assert projected_box(cam, np.stack([outside, outside + [20, 0, 0]]), 10) is None


def test_cli_control_filters_raw_frames_and_records_its_inputs(cameras, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from battle import finebio_arms

    clip = SimpleNamespace(
        camera_config=tmp_path / "camera.json",
        views=("T1", "T2", "fpv"),
        fpv_view="fpv",
        start_frame=1798,
    )
    monkeypatch.setattr(finebio_arms, "load_clip", lambda path: clip)
    monkeypatch.setattr(finebio_arms, "read_camera_config", lambda path: {})
    monkeypatch.setattr(finebio_arms, "cameras_from_config", lambda config: cameras)
    monkeypatch.setattr(finebio_arms, "resolve_fpv_source", lambda config, path: lambda frame: None)
    tracks = tmp_path / "tracks.jsonl"
    tracks.write_text(track().model_dump_json() + "\n" + track(1799).model_dump_json() + "\n")
    observations = tmp_path / "observations.jsonl"
    observations.write_text(obs(cameras["T1"], "T1").model_dump_json() + "\n")
    rig = tmp_path / "rig.json"
    rig.write_text("{}")
    output = tmp_path / "requests"
    assert (
        finebio_arms.main(
            [
                "reproject-prompts",
                "--tracks",
                str(tracks),
                "--observations-dir",
                str(observations),
                "--clip-config",
                "clip.json",
                "--rig",
                str(rig),
                "--output",
                str(output),
                "--frames",
                "1798",
                "--shift-px",
                "60",
            ]
        )
        == 0
    )
    report = json.loads((output / "requests_summary.json").read_text())
    assert report["selected_raw_frames"] == [1798] and report["shift_px"] == 60
    assert report["inputs"]["tracks"] == str(tracks)
    frames, labels = parse_box_stream((output / "T2.jsonl").read_text())
    assert set(frames) == {0} and labels == ("blue_pipette#pipette-008",)


def test_multiple_track_ids_get_dense_slots_with_stable_colour_labels(cameras, tmp_path):
    other = track(1799, "yellow_pipette").model_copy(update={"track_id": "pipette-019"})
    generate_requests(
        [other, track()],
        [],
        lambda frame: {"T2": cameras["T2"]},
        views=("T2",),
        start_frame=1798,
        gate_px=30,
        prior=LinePrior(23, 2),
        output=tmp_path,
    )
    frames, labels = parse_box_stream((tmp_path / "T2.jsonl").read_text())
    assert labels == ("blue_pipette#pipette-008", "yellow_pipette#pipette-019")
    assert frames[0][0]["slot"] == 0 and frames[1][0]["slot"] == 1


def test_held_touch_report_counts_only_the_requests_kept_after_class_dedup(cameras, tmp_path):
    resting = track().model_copy(update={"state": "observed"})
    held = track().model_copy(update={"track_id": "pipette-019"})
    report = generate_requests(
        [resting, held],
        [],
        lambda frame: {"T2": cameras["T2"]},
        views=("T2",),
        start_frame=1798,
        gate_px=30,
        prior=LinePrior(23, 2),
        output=tmp_path,
    )
    assert report["requests"] == 1 and report["skipped"]["duplicate_class_prompt"] == 1
    assert report["held_track_frames"] == 1 and report["held_track_frames_touched"] == 0
    assert report["held_fraction_touched"] == 0


def test_decode_reuses_only_matching_boxes_and_resolves_each_runs_own_labels(
    cameras, tmp_path, monkeypatch
):
    from battle import finebio_reprojection

    initial = obs(cameras["T2"]).model_copy(update={"box_xyxy_px": (100, 100, 200, 200)})
    corrected = initial.model_copy(update={"box_xyxy_px": (100, 100, 200, 250)})
    corrected_raw = np.array([100.035, 100.026, 200.048, 250.007])
    root = tmp_path / "workers/T2"
    for name in ("initial", "corrective"):
        run = root / name
        run.mkdir(parents=True)
        (run / "manifest.json").write_text(
            json.dumps({"method_statuses": [{"state": "succeeded"}]})
        )
        raw = corrected_raw if name == "corrective" else np.array(initial.box_xyxy_px)
        normalized = raw / np.tile(cameras["T2"].size, 2)
        (run / "observations.jsonl").write_text(
            json.dumps(
                {
                    "analysis_frame_index": 0,
                    "objects": [
                        {
                            "label": initial.slot,
                            "mask": {"uri": "masks/result.png"},
                            "prompt_box": {
                                "x": normalized[0],
                                "y": normalized[1],
                                "width": normalized[2] - normalized[0],
                                "height": normalized[3] - normalized[1],
                            },
                        }
                    ],
                }
            )
            + "\n"
        )

    def decode(run, view, start_frame, **kwargs):
        assert "slot_labels" not in kwargs
        return [corrected if run.name == "corrective" else initial]

    monkeypatch.setattr(finebio_reprojection, "worker_to_observations", decode)
    requests = tmp_path / "requests"
    requests.mkdir()
    request = {
        "view": "T2",
        "raw_frame": 1798,
        "label": initial.slot,
        "box_xyxy_px": corrected_raw.tolist(),
    }
    (requests / "requests.jsonl").write_text(json.dumps(request) + "\n")
    rows, _ = finebio_reprojection.decode_rows(
        tmp_path / "workers", requests, 1798, lambda frame: cameras
    )
    assert len(rows) == 1 and rows[0].box_xyxy_px == corrected.box_xyxy_px
    assert rows[0].provenance["reprojection_worker_run"].endswith("corrective")
    assert rows[0].provenance["reprojection_worker_mask_uri"] == "masks/result.png"
