"""Round trips and validation rules for the FineBio 3D-tracking contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from battle.multiview_schemas import (
    FINEBIO_WORLD_UNITS,
    FineBioCameraConfig,
    FineBioFixedCamera,
    FineBioFpvCamera,
    FineBioObservation,
    Track3D,
    TrackEvent,
    read_jsonl,
    write_jsonl,
)

IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _fixed(view: str, camera_id: int, provenance: str = "shipped") -> FineBioFixedCamera:
    return FineBioFixedCamera(
        view=view,
        camera_id=camera_id,
        provenance=provenance,
        K=((900.0, 0.0, 960.0), (0.0, 900.0, 540.0), (0.0, 0.0, 1.0)),
        distortion=(0.1, -0.2, 0.0, 0.0, 0.05),
        rvec=(0.1, 0.2, 0.3),
        tvec=(1.0, 2.0, 90.0),
        image_size=(1920, 1080),
        marker_fit_residual_px=6.3,
        shipped_marker_residual_px=6.3,
    )


def _config() -> FineBioCameraConfig:
    return FineBioCameraConfig(
        trial="P03_01_01",
        recording_day="221013",
        fixed={"T1": _fixed("T1", 1), "T5": _fixed("T5", 6, "marker_pnp")},
        fpv=FineBioFpvCamera(
            K=IDENTITY,
            distortion=(0.0, 0.0, 0.0, 0.0, 0.0),
            image_size=(1920, 1440),
            pose_source="misc/finebio_camera_poses/first_person_camera_poses/P03_01_01.npz",
            pose_frame_count=5032,
            valid_pose_fraction=0.972,
            marker_residual_gate_px=20.0,
            velocity_gate_cm_per_frame=5.0,
        ),
        provenance={"mapping": "runs/preflight-finebio-20260924/mapping/mapping.json"},
    )


def test_detector_observation_round_trips_and_points_at_the_box_centre() -> None:
    row = FineBioObservation(
        view="T1",
        frame_index=1798,
        slot="cell_culture_plate#0",
        object_class="cell_culture_plate",
        detector_score=0.81,
        box_xyxy_px=(100.0, 200.0, 300.0, 260.0),
        pose_valid=True,
        source="detector",
        provenance={"detector": "finebio_dino"},
    )

    again = FineBioObservation.model_validate_json(row.model_dump_json())

    assert again == row
    assert again.point_px == (200.0, 230.0)
    assert again.mask_bbox_px is None


def test_sam3_observation_prefers_the_mask_centroid() -> None:
    row = FineBioObservation(
        view="fpv",
        frame_index=1799,
        slot="blue_pipette#0",
        object_class="blue_pipette",
        detector_score=0.69,
        box_xyxy_px=(972.9, 92.4, 1357.9, 663.4),
        mask_bbox_px=(976.0, 107.0, 1361.0, 672.0),
        mask_centroid_px=(1168.5, 389.5),
        mask_area_px=70705,
        sam3_object_score=9.875,
        pose_valid=True,
        source="sam3_video",
        provenance={"encoder_side": 1280, "detector_box_iou": 0.9357},
    )

    assert row.point_px == (1168.5, 389.5)
    assert FineBioObservation.model_validate(json.loads(row.model_dump_json())) == row


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"source": "detector"}, "carries a detector box, a mask bbox, or both"),
        ({"source": "detector", "mask_bbox_px": (0.0, 0.0, 5.0, 5.0)}, "detector observation"),
        ({"source": "sam3_video", "box_xyxy_px": (0.0, 0.0, 5.0, 5.0)}, "SAM3 observation"),
        ({"source": "detector", "box_xyxy_px": (5.0, 0.0, 5.0, 5.0)}, "positive width"),
    ],
)
def test_observation_rejects_rows_that_are_not_observations(fields: dict, message: str) -> None:
    base = {
        "view": "T2",
        "frame_index": 0,
        "slot": "centrifuge#0",
        "object_class": "centrifuge",
        "pose_valid": True,
    }
    with pytest.raises(ValidationError, match=message):
        FineBioObservation(**base, **fields)


def test_observation_forbids_undeclared_fields_and_is_frozen() -> None:
    row = FineBioObservation(
        view="T1",
        frame_index=1,
        slot="pen#0",
        object_class="pen",
        box_xyxy_px=(0.0, 0.0, 1.0, 1.0),
        pose_valid=True,
        source="detector",
    )
    with pytest.raises(ValidationError):
        FineBioObservation.model_validate({**row.model_dump(), "extra": 1})
    with pytest.raises(ValidationError):
        row.frame_index = 2  # type: ignore[misc]


def test_camera_config_round_trips_with_exact_floats() -> None:
    config = _config()

    text = config.model_dump_json(indent=1)
    again = FineBioCameraConfig.model_validate_json(text)

    assert again == config
    assert again.units == FINEBIO_WORLD_UNITS
    assert again.frame_index_offset == 0
    assert again.fixed["T5"].provenance == "marker_pnp"
    assert again.fpv.pose_frame_count == 5032
    assert again.config_kind == "finebio_camera_config"


def test_camera_config_rejects_a_view_stored_under_another_key() -> None:
    with pytest.raises(ValidationError, match="stored under key"):
        FineBioCameraConfig(
            trial="P03_01_01",
            recording_day="221013",
            fixed={"T2": _fixed("T1", 1)},
            fpv=_config().fpv,
        )
    with pytest.raises(ValidationError):
        FineBioCameraConfig(
            trial="P03_01_01", recording_day="2022-10-13", fixed={}, fpv=_config().fpv
        )


def test_tracks_and_events_round_trip_through_jsonl(tmp_path: Path) -> None:
    tracks = [
        Track3D(
            frame_index=1798,
            track_id="t0001",
            object_class="cell_culture_plate",
            position_cm=(-6.15, 23.0, -3.2),
            uncertainty_cm=1.5,
            support_views=("T1", "T3", "T4", "T5"),
            state="observed",
            confidence=0.8,
            abstain=False,
        ),
        Track3D(
            frame_index=1799,
            track_id="t0001",
            object_class="cell_culture_plate",
            position_cm=(-6.1, 23.1, -3.2),
            uncertainty_cm=4.0,
            state="coasting",
            confidence=0.3,
            abstain=True,
            possibly_same_as=("t0002",),
        ),
    ]
    events = [
        TrackEvent(frame_index=1798, track_id="t0001", kind="birth", payload={"views": 4}),
        TrackEvent(frame_index=1799, track_id="t0001", kind="coasting"),
        TrackEvent(
            frame_index=1805,
            track_id="t0001",
            kind="handoff_reseed",
            payload={"view": "T2", "box_xyxy_px": [1.0, 2.0, 3.0, 4.0]},
        ),
    ]

    assert write_jsonl(tracks, tmp_path / "tracks.jsonl") == 2
    assert write_jsonl(events, tmp_path / "events.jsonl") == 3
    assert list(read_jsonl(tmp_path / "tracks.jsonl", Track3D)) == tracks
    assert list(read_jsonl(tmp_path / "events.jsonl", TrackEvent)) == events
    first_line = (tmp_path / "tracks.jsonl").read_text().splitlines()[0]
    assert "\n" not in first_line and json.loads(first_line)["state"] == "observed"


@pytest.mark.parametrize("kind", ["nap", "Birth", ""])
def test_event_kind_and_track_state_are_closed_vocabularies(kind: str) -> None:
    with pytest.raises(ValidationError):
        TrackEvent(frame_index=0, track_id="t", kind=kind)
    with pytest.raises(ValidationError):
        Track3D(
            frame_index=0,
            track_id="t",
            object_class="c",
            position_cm=(0.0, 0.0, 0.0),
            uncertainty_cm=0.0,
            state="tracked",
            confidence=0.5,
            abstain=False,
        )
