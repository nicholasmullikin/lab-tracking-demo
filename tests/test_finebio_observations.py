"""The two FineBio observation adapters (detector JSONL, SAM3 worker run) and the pose fill."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from conftest import require_artifact

from battle.finebio_observations import (
    AXIS_ELONGATION_THRESHOLD,
    NO_AXIS,
    VIEWS,
    class_of_slot,
    detection_row_to_observations,
    detections_to_observations,
    fill_pose_valid,
    filter_mask_components,
    mask_axis_measurements,
    mask_measurements,
    pose_validity_from_fixture,
    worker_to_observations,
    write_observations,
)
from battle.multiview_schemas import FineBioObservation, read_jsonl

FIXTURES = Path("tests/fixtures/finebio_preflight")
PREFLIGHT_DETECTIONS = Path("runs/preflight-finebio-20260924/detections")
SMOKE_ROOT = Path("runs/p3-worker-gpu-smoke-20260924")


def _record(view: str, frame: int, detections: list[tuple[str, float, list[float]]], **extra):
    return {
        "view": view,
        "frame_index": frame,
        "image_hw": [1080, 1920],
        "detections": [
            {"class": cls, "class_id": 0, "score": score, "box_xyxy_px": box}
            for cls, score, box in detections
        ],
        "interpolated": False,
        **extra,
    }


def test_detection_row_ranks_same_class_by_score_and_keeps_interpolation():
    record = _record(
        "T1",
        700,
        [
            ("micro_tube", 0.55, [10, 10, 30, 40]),
            ("micro_tube", 0.91, [50, 10, 70, 40]),
            ("pcr_machine", 0.2, [100, 100, 300, 300]),
            ("centrifuge", 0.84123456, [500.04, 0, 700.26, 200]),
        ],
        interpolated=True,
        source_frames=[695, 705],
    )
    rows = detection_row_to_observations(record, min_score=0.3)
    assert [r.slot for r in rows] == ["micro_tube#0", "centrifuge#0", "micro_tube#1"]
    assert rows[0].box_xyxy_px == (50.0, 10.0, 70.0, 40.0)
    assert rows[1].detector_score == 0.8412 and rows[1].box_xyxy_px == (500.0, 0.0, 700.3, 200.0)
    assert all(r.source == "detector" and r.pose_valid for r in rows)
    assert all(r.provenance == {"interpolated": True, "source_frames": [695, 705]} for r in rows)
    plain = detection_row_to_observations(_record("T1", 701, [("pen", 0.5, [0, 0, 5, 5])]))
    assert plain[0].provenance == {}


def test_detections_to_observations_needs_a_pose_lookup_for_the_fpv(tmp_path: Path):
    for view in ("fpv", "T1"):
        (tmp_path / f"{view}.jsonl").write_text(
            "\n".join(
                json.dumps(_record(view, frame, [("cell_culture_plate", 0.6, [0, 0, 10, 10])]))
                for frame in (600, 601)
            )
            + "\n"
        )
    with pytest.raises(ValueError, match="pose validity"):
        detections_to_observations(tmp_path, ("fpv", "T1"))
    rows = detections_to_observations(tmp_path, ("fpv", "T1"), pose_valid=lambda f: f == 600)
    by = {(r.view, r.frame_index): r.pose_valid for r in rows}
    assert by == {("fpv", 600): True, ("fpv", 601): False, ("T1", 600): True, ("T1", 601): True}
    only = detections_to_observations(tmp_path, ("T1",), frames=[601])
    assert [(r.view, r.frame_index) for r in only] == [("T1", 601)]


def test_mask_measurements_use_pixel_centres():
    mask = np.zeros((20, 30), dtype=bool)
    mask[10, 10] = True
    assert mask_measurements(mask) == ((10.0, 10.0, 11.0, 11.0), (10.5, 10.5), 1)
    mask[10:14, 10:16] = True
    bbox, centroid, area = mask_measurements(mask)
    assert bbox == (10.0, 10.0, 16.0, 14.0) and area == 24
    assert centroid == (13.0, 12.0)
    assert mask_measurements(np.zeros((4, 4), dtype=bool)) is None


# ------------------------------------------------------------------ mask axis (Sep 28)


def _rect(angle_deg: float, length: float, width: float, centre=(300.0, 200.0), shape=(400, 600)):
    import cv2

    mask = np.zeros(shape, np.uint8)
    box = cv2.boxPoints(((centre[0], centre[1]), (length, width), angle_deg))
    cv2.fillPoly(mask, [np.round(box).astype(np.int32)], 255)
    return mask > 0


def _blob(radius: int, centre=(300, 200), shape=(400, 600)):
    import cv2

    mask = np.zeros(shape, np.uint8)
    cv2.circle(mask, centre, radius, 255, -1)
    return mask > 0


def _axis_angle_deg(axis) -> float:
    (x0, y0), (x1, y1) = axis
    return float(np.degrees(np.arctan2(y1 - y0, x1 - x0)) % 180.0)


def _angle_gap(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _tiny_mask():
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:12, 10:12] = True
    return mask


# The old fields on deterministic synthetic masks, recorded with the Sep 25 code before the
# axis was added: bbox, centroid and area must stay byte for byte what they were.
OLD_MEASUREMENTS = {
    "rect30": ((190.0, 130.0, 411.0, 271.0), (300.5, 200.5), 5829),
    "rect_vert": ((292.0, 100.0, 309.0, 301.0), (300.5, 200.5), 3417),
    "blob": ((260.0, 160.0, 341.0, 241.0), (300.5, 200.5), 5025),
    "rect_blob": ((180.0, 188.0, 421.0, 261.0), (317.6, 210.2), 8419),
    "tiny": ((10.0, 10.0, 12.0, 12.0), (11.0, 11.0), 4),
    "empty": None,
}


def _golden_masks():
    return {
        "rect30": _rect(30, 240, 24),
        "rect_vert": _rect(90, 200, 16),
        "blob": _blob(40),
        "rect_blob": _rect(0, 240, 24) | _blob(30, (360, 230)),
        "tiny": _tiny_mask(),
        "empty": np.zeros((40, 40), dtype=bool),
    }


def test_old_mask_measurements_are_unchanged_by_the_axis():
    for name, mask in _golden_masks().items():
        kept, dropped, _ = filter_mask_components(mask)
        assert dropped == 0
        assert mask_measurements(kept) == OLD_MEASUREMENTS[name], name


def test_axis_of_a_rotated_rectangle_matches_its_angle_length_and_width():
    for angle, length, width in ((30, 240, 24), (-30, 240, 24), (0, 240, 24), (90, 200, 16)):
        axis = mask_axis_measurements(_rect(angle, length, width))
        assert axis.method == "ransac_skeleton" and axis.reason is None
        assert axis.axis_px is not None
        assert _angle_gap(_axis_angle_deg(axis.axis_px), angle % 180) < 1.0
        # Endpoints: the rectangle's centre +/- half the length along its axis.
        u = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
        expected = sorted(
            [tuple(np.array([300.0, 200.0]) + s * length / 2 * u) for s in (-1, 1)],
            key=lambda p: (p[1], p[0]),
        )
        for got, want in zip(axis.axis_px, expected):
            assert np.hypot(got[0] - want[0], got[1] - want[1]) < 2.0, (angle, got, want)
        # A rasterised rectangle is one pixel longer and wider than its nominal size.
        assert abs(axis.width_px - (width + 1)) <= 0.1 * width
        assert abs(axis.elongation - (length + 1) / (width + 1)) < 0.5
        assert axis.residual_px is not None and axis.residual_px < 1.0
        # Top first: the endpoints are ordered by image y then x.
        assert (axis.axis_px[0][1], axis.axis_px[0][0]) <= (axis.axis_px[1][1], axis.axis_px[1][0])


def test_end_widths_name_the_wide_end_of_a_shaft_with_a_head():
    """Sep 29 (tip / butt by the width profile): a 240 x 16 px shaft with a 40 x 56 px head
    on one end. The end widths follow the axis endpoint order (top first) and the head's end
    reads about three times the shaft's; a plain rectangle reads the same width at both
    ends, and a compact blob has none."""
    for angle in (0, 30, 90):
        u = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
        shaft = _rect(angle, 240, 16)
        head_centre = np.array([300.0, 200.0]) + 100.0 * u
        head = _rect(angle, 40, 56, centre=tuple(head_centre))
        axis = mask_axis_measurements(shaft | head)
        assert axis.axis_px is not None and axis.end_widths_px is not None
        ends = np.asarray(axis.axis_px)
        head_end = int(np.argmin(np.linalg.norm(ends - head_centre, axis=1)))
        wide, narrow = axis.end_widths_px[head_end], axis.end_widths_px[1 - head_end]
        assert wide > 2.5 * narrow, (angle, axis.end_widths_px)
        assert abs(narrow - 17.0) <= 3.0, (angle, axis.end_widths_px)
    plain = mask_axis_measurements(_rect(30, 240, 24))
    assert plain.end_widths_px is not None
    assert abs(plain.end_widths_px[0] - plain.end_widths_px[1]) <= 2.0
    assert mask_axis_measurements(_blob(40)).end_widths_px is None


def test_tip_side_is_the_end_with_the_longer_thin_tail_and_body_ends_sit_on_the_mask():
    """Sep 29 v4 (the tipseg tail rule in the observation layer): a pipette-shaped mask, a
    36 px wide grip with a 130 px thin shaft on one side and a 24 px thin plunger stem on
    the other, names the shaft's end as `tip_side` whatever the axis order or angle; the
    thinner end *band* alone would tie them. The terminal centroid at each end sits on the
    mask's own end. A plain rectangle and a shaft with one head decide nothing or the
    shaft's end respectively, and a compact blob has neither."""
    from battle.finebio_observations import tip_side_from_tails

    for angle in (0, 30, 90, -60):
        u = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
        centre = np.array([300.0, 200.0])
        grip = _rect(angle, 60, 36, centre=tuple(centre))
        shaft = _rect(angle, 130, 12, centre=tuple(centre + 95.0 * u))
        stem = _rect(angle, 24, 12, centre=tuple(centre - 42.0 * u))
        axis = mask_axis_measurements(grip | shaft | stem)
        assert axis.axis_px is not None and axis.tip_side is not None, angle
        ends = np.asarray(axis.axis_px)
        shaft_end = centre + 160.0 * u
        assert axis.tip_side == int(np.argmin(np.linalg.norm(ends - shaft_end, axis=1))), angle
        assert axis.tails_px is not None
        assert axis.tails_px[axis.tip_side] > 3 * axis.tails_px[1 - axis.tip_side]
        assert axis.body_ends_px is not None
        for end in (0, 1):
            assert np.hypot(*(np.asarray(axis.body_ends_px[end]) - ends[end])) < 4.0, (angle, end)
        assert axis.provenance()["tails_px"] == list(axis.tails_px)
    plain = mask_axis_measurements(_rect(30, 240, 24))
    assert plain.tip_side is None and plain.body_ends_px is not None
    # A body (100 x 56) with a thin shaft (140 x 16) on one side: the shaft's free end.
    head = mask_axis_measurements(_rect(0, 140, 16, centre=(230.0, 200.0)) | _rect(0, 100, 56))
    assert head.axis_px is not None and head.tip_side is not None
    assert head.axis_px[head.tip_side][0] < 200.0
    # A shaft that is most of the mask (240 x 16 with a 40 x 56 head): the body width is the
    # shaft's, nothing is a tail, no decision.
    mostly_shaft = mask_axis_measurements(
        _rect(0, 240, 16) | _rect(0, 40, 56, centre=(400.0, 200.0))
    )
    assert mostly_shaft.tip_side is None and mostly_shaft.tails_px == (0.0, 0.0)
    blob = mask_axis_measurements(_blob(40))
    assert blob.tip_side is None and blob.body_ends_px is None and blob.tails_px is None
    # The rule itself: the longer tail wins by more than the floor and at least 1.5x.
    assert tip_side_from_tails((30.0, 4.0), 10.0) == 0
    assert tip_side_from_tails((4.0, 30.0), 10.0) == 1
    assert tip_side_from_tails((30.0, 22.0), 10.0) is None  # under 1.5x
    assert tip_side_from_tails((14.0, 6.0), 10.0) is None  # under the floor
    assert tip_side_from_tails((0.0, 0.0), 10.0) is None


def test_round_blob_has_no_axis_but_an_elongation_and_a_width():
    axis = mask_axis_measurements(_blob(40))
    assert axis.axis_px is None and axis.residual_px is None
    assert axis.method == "pca" and axis.reason == "compact"
    assert abs(axis.elongation - 1.0) < 0.05
    assert 60.0 <= axis.width_px <= 81.0
    assert axis.provenance() == {"axis_method": "pca", "axis_reason": "compact"}


def test_ransac_axis_ignores_a_blob_stuck_on_one_side():
    plain = mask_axis_measurements(_rect(0, 240, 24))
    assert plain.residual_px < 0.5
    # A 30 px blob whose centre sits 30-40 px to one side of the rectangle's axis.
    for angle, blob_centre in ((0, (360, 230)), (20, (330, 170)), (60, (288, 253))):
        merged = mask_axis_measurements(_rect(angle, 240, 24) | _blob(30, blob_centre))
        assert merged.method == "ransac_skeleton" and merged.axis_px is not None
        assert _angle_gap(_axis_angle_deg(merged.axis_px), angle) < 3.0, angle
        # The rectangle's ends, not the blob's, bound the axis; the branch raises the residual.
        for got, want in zip(merged.axis_px, mask_axis_measurements(_rect(angle, 240, 24)).axis_px):
            assert np.hypot(got[0] - want[0], got[1] - want[1]) < 3.0
        assert merged.residual_px > 2.0, (angle, merged)
        assert abs(merged.width_px - plain.width_px) < 2.0


def test_empty_and_tiny_masks_have_no_axis_with_a_reason():
    for mask in (np.zeros((40, 40), dtype=bool), _tiny_mask()):
        axis = mask_axis_measurements(mask)
        assert axis == NO_AXIS
        assert axis.method == "none" and axis.reason == "too_few_pixels"
        assert axis.elongation is None and axis.width_px is None
    assert mask_axis_measurements(_rect(45, 150, 1)).method == "ransac_skeleton"
    assert AXIS_ELONGATION_THRESHOLD == 2.5


def test_large_masks_are_stride_sampled_without_moving_the_axis():
    mask = _rect(15, 900, 90, centre=(960.0, 540.0), shape=(1080, 1920))
    full = mask_axis_measurements(mask, max_points=10**7)
    sampled = mask_axis_measurements(mask, max_points=20_000)
    assert sampled.method == full.method == "ransac_skeleton"
    assert _angle_gap(_axis_angle_deg(sampled.axis_px), 15) < 0.5
    assert abs(sampled.elongation - full.elongation) < 0.05
    assert abs(sampled.width_px - full.width_px) < 2.0
    for got, want in zip(sampled.axis_px, full.axis_px):
        assert np.hypot(got[0] - want[0], got[1] - want[1]) < 2.0


def _write_worker_run(root: Path, *, video_mode: bool) -> None:
    import cv2

    (root / "masks").mkdir(parents=True)
    width, height = 64, 48
    lines = []
    for k in range(2):
        objects = []
        for slot, label, (x0, y0, x1, y1) in (
            (0, "cell_culture_plate", (8, 8, 24, 20)),
            (1, "micro_tube", (40 + k, 30, 50 + k, 44)),
        ):
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[y0:y1, x0:x1] = 255
            uri = f"masks/{k:06d}_{slot:02d}.png"
            cv2.imwrite(str(root / uri), mask)
            obj = {
                "object_id": f"sam3-{slot:02d}",
                "label": label,
                "confidence": 0.9,
                "box": {"x": x0 / width, "y": y0 / height, "width": 0.2, "height": 0.2},
                "mask": {"uri": uri, "storage": "external_artifact", "format": "png"},
                "object_score": 9.5 if video_mode else 0.9,
                "iou_prediction": 0.9,
            }
            if not video_mode:
                obj.update(
                    prompt_box={"x": 0.1, "y": 0.15, "width": 0.25, "height": 0.25},
                    source="sam3_decode",
                    prompt_source="finebio_dino",
                    prompt_score=0.42,
                )
            objects.append(obj)
        lines.append(
            json.dumps(
                {
                    "view_id": "T2",
                    "analysis_frame_index": k,
                    "source_seconds": k / 30.0,
                    "objects": objects,
                    "hands": [],
                }
            )
        )
    (root / "observations.jsonl").write_text("\n".join(lines) + "\n")


def test_worker_to_observations_box_decode_rows(tmp_path: Path):
    _write_worker_run(tmp_path, video_mode=False)
    rows = worker_to_observations(tmp_path, "T2", 600, ["cell_culture_plate#0", "micro_tube#3"])
    assert [(r.frame_index, r.slot) for r in rows] == [
        (600, "cell_culture_plate#0"),
        (600, "micro_tube#3"),
        (601, "cell_culture_plate#0"),
        (601, "micro_tube#3"),
    ]
    plate = rows[0]
    assert plate.source == "sam3_decode" and plate.object_class == "cell_culture_plate"
    assert plate.mask_bbox_px == (8.0, 8.0, 24.0, 20.0)
    assert plate.mask_centroid_px == (16.0, 14.0) and plate.mask_area_px == 16 * 12
    assert plate.point_px == (16.0, 14.0)
    # The box that prompted the decode is the detector box, in pixels of the 64x48 mask.
    assert plate.box_xyxy_px == (6.4, 7.2, 22.4, 19.2) and plate.detector_score == 0.42
    assert plate.sam3_object_score == 0.9 and plate.provenance["decoder_iou_pred"] == 0.9
    assert plate.provenance["prompt_source"] == "finebio_dino"
    tube = rows[3]
    assert tube.mask_bbox_px == (41.0, 30.0, 51.0, 44.0) and tube.mask_centroid_px == (46.0, 37.0)


def test_worker_rows_carry_the_mask_axis_fields(tmp_path: Path):
    """Sep 28 (p0-axis-observations): a compact mask gets elongation and width and no axis, an
    elongated one the axis too; the old fields are what they were on the same run."""
    import cv2

    _write_worker_run(tmp_path, video_mode=False)
    # Overwrite the tube mask of frame 1 with a 40 x 4 px bar at 0 degrees.
    mask = np.zeros((48, 64), dtype=np.uint8)
    mask[20:24, 10:50] = 255
    cv2.imwrite(str(tmp_path / "masks/000001_01.png"), mask)
    rows = worker_to_observations(tmp_path, "T2", 600, ["cell_culture_plate#0", "micro_tube#3"])
    plate = rows[0]
    assert plate.mask_bbox_px == (8.0, 8.0, 24.0, 20.0)
    assert plate.mask_centroid_px == (16.0, 14.0) and plate.mask_area_px == 16 * 12
    assert plate.mask_axis_px is None and plate.mask_axis_residual_px is None
    assert plate.mask_elongation == pytest.approx(16 / 12, abs=0.01)
    assert plate.mask_width_px == 12.0
    assert plate.provenance["axis_method"] == "pca"
    assert plate.provenance["axis_reason"] == "compact"
    bar = rows[3]
    assert bar.slot == "micro_tube#3" and bar.frame_index == 601
    assert bar.mask_bbox_px == (10.0, 20.0, 50.0, 24.0) and bar.mask_area_px == 160
    # The thinned line of an even-width bar sits half a pixel off its centre (22.0).
    assert bar.mask_axis_px == ((10.0, 21.5), (50.0, 21.5))
    assert bar.mask_elongation == 10.0 and bar.mask_width_px == 4.0
    assert bar.mask_axis_residual_px == 0.0
    assert bar.provenance["axis_method"] == "ransac_skeleton"
    assert "axis_reason" not in bar.provenance
    # The compact JSONL omits None fields, so a row without an axis reads back as before.
    out = tmp_path / "obs.jsonl"
    write_observations(rows, out)
    back = list(read_jsonl(out, FineBioObservation))
    assert back == sorted(rows, key=lambda r: (r.frame_index, r.slot))
    assert "mask_axis_px" not in out.read_text().splitlines()[0]


def test_worker_to_observations_video_rows_match_detector_boxes(tmp_path: Path):
    _write_worker_run(tmp_path, video_mode=True)
    detector = [
        FineBioObservation(
            view="T2",
            frame_index=601,
            slot="micro_tube#0",
            object_class="micro_tube",
            detector_score=0.77,
            box_xyxy_px=(40.0, 29.0, 52.0, 45.0),
            pose_valid=True,
            source="detector",
        ),
        FineBioObservation(
            view="T2",
            frame_index=601,
            slot="micro_tube#1",
            object_class="micro_tube",
            detector_score=0.5,
            box_xyxy_px=(0.0, 0.0, 5.0, 5.0),
            pose_valid=True,
            source="detector",
        ),
    ]
    rows = worker_to_observations(tmp_path, "T2", 600, detector_rows=detector)
    assert all(r.source == "sam3_video" for r in rows)
    # No slot labels given and plain class labels: the slot index names the slot.
    assert [r.slot for r in rows[:2]] == ["cell_culture_plate#0", "micro_tube#1"]
    assert rows[1].box_xyxy_px is None and rows[1].detector_score is None
    tube_601 = rows[3]
    assert tube_601.box_xyxy_px == (40.0, 29.0, 52.0, 45.0) and tube_601.detector_score == 0.77
    assert tube_601.provenance["detector_slot"] == "micro_tube#0"
    assert 0.6 < tube_601.provenance["detector_box_iou"] < 1.0
    assert tube_601.sam3_object_score == 9.5


def test_worker_rows_drop_speckle_components_like_the_worker_box(tmp_path: Path):
    """Sep 25 (p4-arms): SAM3.1 video-memory masks carry a few isolated positive pixels far
    from the object; the bbox, centroid and area come from the components the worker's own
    box rule keeps, and the dropped pixels are on the record."""
    import cv2

    _write_worker_run(tmp_path, video_mode=True)
    path = tmp_path / "masks/000000_00.png"
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    mask[2, 60] = 255  # one stray pixel far from the 16x12 plate
    mask[45, 1] = 255
    mask[46, 1] = 255  # a two-pixel speckle
    cv2.imwrite(str(path), mask)
    rows = worker_to_observations(tmp_path, "T2", 600)
    plate = rows[0]
    assert plate.mask_bbox_px == (8.0, 8.0, 24.0, 20.0)
    assert plate.mask_centroid_px == (16.0, 14.0) and plate.mask_area_px == 16 * 12
    assert plate.provenance["mask_speckle_pixels_dropped"] == 3
    assert plate.provenance["mask_components"] == 3
    clean = rows[1]
    assert "mask_speckle_pixels_dropped" not in clean.provenance
    kept, dropped, components = filter_mask_components(mask > 0)
    assert dropped == 3 and components == 3 and int(kept.sum()) == 16 * 12
    # A component at least 20% of the largest is part of the object, not a speckle.
    two = np.zeros((20, 40), dtype=bool)
    two[2:12, 2:12] = True
    two[5:10, 30:36] = True
    kept, dropped, components = filter_mask_components(two)
    assert dropped == 0 and components == 2 and kept.sum() == two.sum()
    assert filter_mask_components(np.zeros((4, 4), dtype=bool))[1:] == (0, 0)


def test_worker_rows_fallback_to_the_box_when_the_mask_is_missing(tmp_path: Path):
    _write_worker_run(tmp_path, video_mode=True)
    (tmp_path / "masks/000001_00.png").unlink()
    rows = worker_to_observations(tmp_path, "T2", 600)
    missing = rows[2]
    assert missing.frame_index == 601 and missing.slot == "cell_culture_plate#0"
    assert missing.mask_centroid_px is None and missing.mask_area_px is None
    assert missing.mask_bbox_px == (8.0, 8.0, 20.8, 17.6)
    assert missing.provenance["mask"] == "absent"
    assert missing.point_px == (14.4, 12.8)


def test_pose_fill_from_the_fixture_poses_and_slot_class():
    validity = pose_validity_from_fixture(FIXTURES / "fpv_poses.json")
    rows = [
        FineBioObservation(
            view=view,
            frame_index=frame,
            slot="pen#0",
            object_class="pen",
            box_xyxy_px=(0.0, 0.0, 1.0, 1.0),
            pose_valid=False,
            source="detector",
        )
        for view, frame in (("fpv", 1798), ("fpv", 1), ("T1", 1))
    ]
    filled = fill_pose_valid(rows, validity)
    assert [r.pose_valid for r in filled] == [True, False, False]
    assert class_of_slot("micro_tube_group#2") == "micro_tube_group"
    assert class_of_slot("centrifuge") == "centrifuge"


def test_write_observations_orders_rows(tmp_path: Path):
    rows = [
        FineBioObservation(
            view=view,
            frame_index=frame,
            slot=slot,
            object_class="pen",
            box_xyxy_px=(0.0, 0.0, 1.0, 1.0),
            pose_valid=True,
            source="detector",
        )
        for view, frame, slot in (("T2", 5, "pen#1"), ("fpv", 5, "pen#0"), ("T1", 4, "pen#0"))
    ]
    out = tmp_path / "obs.jsonl"
    assert write_observations(rows, out) == 3
    back = list(read_jsonl(out, FineBioObservation))
    assert [(r.frame_index, r.view) for r in back] == [(4, "T1"), (5, "fpv"), (5, "T2")]


@pytest.mark.real_data
def test_preflight_detections_rebuild_the_fixture_rows():
    from battle.finebio_observations import fpv_pose_validity

    require_artifact(PREFLIGHT_DETECTIONS / "T5.jsonl")
    require_artifact("data/raw/finebio/misc/finebio_camera_poses")
    rows = detections_to_observations(
        PREFLIGHT_DETECTIONS, VIEWS, 0.3, pose_valid=fpv_pose_validity("P03_01_01")
    )
    fixture = [
        r
        for r in read_jsonl(FIXTURES / "observations.jsonl", FineBioObservation)
        if r.source == "detector"
    ]
    key = lambda r: (r.frame_index, r.view, r.slot)  # noqa: E731
    assert sorted(rows, key=key) == sorted(fixture, key=key)


@pytest.mark.real_data
def test_worker_smoke_runs_convert_to_rows():
    from battle.finebio_observations import fpv_pose_validity

    require_artifact(SMOKE_ROOT)
    require_artifact("data/raw/finebio/misc/finebio_camera_poses")
    validity = fpv_pose_validity("P03_01_01")
    detector = detections_to_observations(
        require_artifact(PREFLIGHT_DETECTIONS), ("fpv",), 0.3, pose_valid=validity
    )
    for mode in ("box_decode", "video_memory"):
        # pytest tmp_path names: `test_box_decode_arm_masks_all_0`, `..._arm_takes_bo0`.
        runs = sorted(SMOKE_ROOT.glob(f"test_{mode}_arm_*0/runs/*"))
        if not runs:
            pytest.skip(f"no {mode} smoke run under {SMOKE_ROOT}")
        rows = worker_to_observations(
            runs[-1], "fpv", 1798, pose_valid=validity, detector_rows=detector
        )
        assert rows and {r.frame_index for r in rows} <= set(range(1798, 1818))
        for row in rows:
            assert row.mask_bbox_px is not None and row.mask_centroid_px is not None
            x0, y0, x1, y1 = row.mask_bbox_px
            assert x0 <= row.mask_centroid_px[0] <= x1 and y0 <= row.mask_centroid_px[1] <= y1
            assert row.mask_area_px is not None and row.mask_area_px > 0
        sources = {r.source for r in rows}
        if mode == "box_decode":
            assert sources == {"sam3_decode"}
            assert all(r.box_xyxy_px is not None and r.detector_score for r in rows)
        else:
            assert "sam3_video" in sources
        plate = [
            r for r in rows if r.slot.startswith("cell_culture_plate") and r.frame_index == 1798
        ]
        assert len(plate) == 1 and plate[0].provenance["detector_box_iou"] > 0.9
