"""FineBio camera library: projection maths, marker matching, and the committed P03 config."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from conftest import require_artifact

from battle import finebio_cameras as fc
from battle.multiview_schemas import FineBioCameraConfig

ROOT = Path(__file__).resolve().parents[1]
P03_CONFIG = ROOT / "configs/finebio/cameras/P03_01_01.json"
PREFLIGHT = ROOT / "runs/preflight-finebio-20260924"


def _camera(rvec=(0.0, 0.0, 0.0), tvec=(0.0, 0.0, 100.0)) -> fc.Camera:
    K = np.array([[900.0, 0.0, 960.0], [0.0, 900.0, 540.0], [0.0, 0.0, 1.0]])
    return fc.Camera("t", K, np.zeros(5), np.array(rvec), np.array(tvec), (1920, 1080))


def test_camera_centre_projection_and_undistort_agree_with_the_pinhole() -> None:
    cam = _camera(rvec=(0.0, 0.1, 0.0), tvec=(1.0, 2.0, 100.0))

    centre = cam.centre
    assert np.allclose(cam.R @ centre + cam.tvec, 0.0)
    point = np.array([3.0, -2.0, 5.0])
    homogeneous = cam.projection @ np.append(point, 1.0)
    assert np.allclose(cam.project(point)[0], homogeneous[:2] / homogeneous[2])
    pixels = np.array([[100.0, 200.0], [1500.0, 900.0]])
    assert np.allclose(cam.undistort(pixels), pixels)


def test_match_markers_picks_the_nearest_marker_and_the_best_corner_order() -> None:
    square = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    projected = np.stack([square, square + 500.0])
    detected = {7: np.roll(square + 500.0, 2, axis=0) + 0.5}

    (marker_id, index, rms, rolled), *rest = fc.match_markers(detected, projected)

    assert not rest and marker_id == 7 and index == 1
    assert rms == pytest.approx(np.sqrt(0.5))
    assert np.allclose(rolled, np.roll(square + 500.0, 2, axis=0))


def test_committed_p03_config_rebuilds_the_preflight_rig() -> None:
    config = fc.read_camera_config(P03_CONFIG)

    assert config.trial == "P03_01_01" and config.recording_day == "221013"
    assert {v: c.camera_id for v, c in config.fixed.items()} == {
        "T1": 1,
        "T2": 2,
        "T3": 3,
        "T4": 4,
        "T5": 6,
    }
    assert {v: c.provenance for v, c in config.fixed.items()} == {
        "T1": "shipped",
        "T2": "shipped",
        "T3": "shipped",
        "T4": "shipped",
        "T5": "marker_pnp",
    }
    assert config.fixed["T5"].shipped_marker_residual_px > 90
    assert config.fixed["T5"].marker_fit_residual_px < 1.0
    assert all(c.marker_fit_residual_px < 10 for c in config.fixed.values())
    assert config.fpv.image_size == (1920, 1440) and config.fpv.pose_frame_count == 5032
    assert config.frame_index_offset == 0

    cameras = fc.cameras_from_config(config)
    # The preflight's rig.json: T5 (camera 6) marker-PnP centre and the fixed views' heights.
    assert np.allclose(cameras["T5"].centre, [-2.87, 5.27, -90.58], atol=0.01)
    assert all(-95 < cam.centre[2] < -50 for cam in cameras.values())
    assert cameras["T1"].K[0, 0] == pytest.approx(config.fixed["T1"].K[0][0])
    assert cameras["T1"].size == (1920, 1080)
    assert FineBioCameraConfig.model_validate_json(P03_CONFIG.read_text()) == config
    assert P03_CONFIG.read_text().endswith("}\n")


def test_fpv_camera_from_config_respects_pose_validity() -> None:
    config = fc.read_camera_config(P03_CONFIG)
    rets = np.array([True, False])
    rots = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    trans = np.array([[0.0, 0.0, 40.0], [0.0, 0.0, 40.0]])

    cam = fc.fpv_camera_from_config(config, 0, (rets, rots, trans))
    assert cam is not None and cam.size == (1920, 1440)
    assert np.allclose(cam.centre, [0.0, 0.0, -40.0])
    assert fc.fpv_camera_from_config(config, 1, (rets, rots, trans)) is None
    assert fc.fpv_camera_from_config(config, 2, (rets, rots, trans)) is None


def test_write_camera_config_round_trips_exactly(tmp_path: Path) -> None:
    config = fc.read_camera_config(P03_CONFIG)

    written = fc.write_camera_config(config, tmp_path / "cameras" / "P03.json")

    assert written.read_bytes() == P03_CONFIG.read_bytes()


@pytest.mark.real_data
def test_camera_config_from_mapping_reproduces_the_committed_config() -> None:
    mapping = json.loads(require_artifact(PREFLIGHT / "mapping/mapping.json").read_text())
    require_artifact(fc.POSES / "intrinsic_parameters")
    committed = fc.read_camera_config(P03_CONFIG)

    rebuilt = fc.camera_config_from_mapping(mapping, "P03_01_01")

    assert rebuilt.model_dump(exclude={"provenance"}) == committed.model_dump(
        exclude={"provenance"}
    )
    cams, provenance = fc.rig_cameras(mapping)
    for view, cam in fc.cameras_from_config(committed).items():
        assert provenance[view] == committed.fixed[view].provenance
        for attr in ("K", "dist", "rvec", "tvec"):
            assert np.array_equal(getattr(cam, attr), getattr(cams[view], attr)), (view, attr)


@pytest.mark.real_data
def test_detect_markers_on_a_raw_fixed_frame_fits_the_committed_pose() -> None:
    from battle.finebio_frames import read_frame, video_path

    video = require_artifact(video_path("P03_01_01", "T4"))
    require_artifact(fc.POSES / "third_person_camera_poses/221013")
    config = fc.read_camera_config(P03_CONFIG)
    frame = read_frame(video, int(round(60 * 30000 / 1001)))

    detected = fc.detect_markers(frame)
    assert detected, "the T4 frame at 60 s carries ArUco markers in the preflight"
    rms = fc.marker_corner_rms(fc.cameras_from_config(config)["T4"], "221013", frame)
    assert rms is not None and rms < 10
    assert cv2.aruco.DICT_6X6_50 == fc.ARUCO_DICT
