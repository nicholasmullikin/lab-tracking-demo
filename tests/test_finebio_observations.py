"""The two FineBio observation adapters (detector JSONL, SAM3 worker run) and the pose fill."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from conftest import require_artifact

from battle.finebio_observations import (
    VIEWS,
    class_of_slot,
    detection_row_to_observations,
    detections_to_observations,
    fill_pose_valid,
    filter_mask_components,
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
