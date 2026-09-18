"""CameraRig: projection, rays, DLT triangulation with filtering, epipolar distance, table plane."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from conftest import require_artifact

from battle import multiview_geometry as mvg
from battle.assembly101_pose_schemas import Assembly101CameraModel, Assembly101ClockRule


def _look_at(position: np.ndarray, target: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    """Camera-to-world pose looking from `position` at `target`, OpenCV axes (y down)."""
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up_hint)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = down
    pose[:3, 2] = forward
    pose[:3, 3] = position
    return pose


def _camera(view: str, pose: np.ndarray | None, *, ego: bool = False) -> Assembly101CameraModel:
    if ego:
        return Assembly101CameraModel(
            view_key=f"{view[4:]}:mono10bit",
            raw_image_size=(636, 480),
            intrinsic_matrix=((190.0, 0.0, 318.0), (0.0, 190.0, 240.0), (0.0, 0.0, 1.0)),
            distortion=(0.4, 0.05, 0.0, 0.0, 0.0, 0.5, 0.1, 0.01),
            distortion_model="rational",
            extrinsics_kind="per_frame_ego",
            provenance="estimated_from_dataset_landmark_projection",
            fit_rms_pixels=0.0,
            fit_point_count=1,
        )
    assert pose is not None
    return Assembly101CameraModel(
        view_key=f"{view}:rgb",
        raw_image_size=(1920, 1080),
        intrinsic_matrix=((1250.0, 0.0, 955.0), (0.0, 1250.0, 530.0), (0.0, 0.0, 1.0)),
        distortion=(-0.13, 0.12, 1e-4, 2e-4, -0.02),
        camera_to_world=tuple(tuple(float(v) for v in row) for row in pose),
        provenance="estimated_from_dataset_landmark_projection",
        fit_rms_pixels=0.0,
        fit_point_count=1,
    )


def _rule(view_key: str, offset: int) -> Assembly101ClockRule:
    return Assembly101ClockRule(
        view_key=view_key,
        proxy_start_raw_frame=17640,
        raw_frames_per_proxy_frame=2,
        pose_offset_frames=offset,
        pose_fps=60,
        analysis_fps=30,
        offset_uncertainty_frames=1,
        offset_evidence="fixture",
    )


# World: table around the origin, "up" is -Y (so gravity/down is +Y), cameras above the table.
UP = np.array([0.0, -1.0, 0.0])


@pytest.fixture
def rig() -> mvg.CameraRig:
    target = np.array([0.0, 0.0, 0.0])
    poses = {
        "C10001": _look_at(np.array([600.0, -700.0, -900.0]), target, UP),
        "C10002": _look_at(np.array([-800.0, -650.0, -700.0]), target, UP),
        "C10003": _look_at(np.array([100.0, -900.0, 800.0]), target, UP),
    }
    cameras = {view: _camera(view, pose) for view, pose in poses.items()}
    cameras["HMC_00000001"] = _camera("HMC_00000001", None, ego=True)
    ego_poses = {
        17640 + k: {
            "00000001:mono10bit": _look_at(
                np.array([50.0 + 5.0 * k, -500.0, -350.0 - 3.0 * k]), target, UP
            )
        }
        for k in range(0, 40)
    }
    rules = {
        "C10001": _rule("C10001:rgb", 5),
        "C10002": _rule("C10002:rgb", 9),
        "HMC_00000001": _rule("00000001:mono10bit", 0),
    }
    return mvg.CameraRig(cameras, rules, ego_poses)


def _points(seed: int = 0, count: int = 40) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform([-200, -120, -200], [200, 0, 200], size=(count, 3))


def test_rig_bookkeeping_and_clock_rules(rig: mvg.CameraRig) -> None:
    assert rig.static_views == ("C10001", "C10002", "C10003")
    assert rig.ego_views == ("HMC_00000001",)
    assert rig.pose_frame("C10001", 10) == 17640 + 20 + 5
    assert rig.pose_frame("C10002", 10) == 17640 + 20 + 9
    with pytest.raises(KeyError, match="no measured clock rule"):
        rig.pose_frame("C10003", 0)
    with pytest.raises(KeyError, match="not in the rig"):
        rig.camera("C99999")
    with pytest.raises(ValueError, match="pose_frame"):
        rig.camera_to_world("HMC_00000001")
    with pytest.raises(KeyError, match="no ego pose"):
        rig.camera_to_world("HMC_00000001", 99)
    with pytest.raises(ValueError, match="view_key"):
        mvg.CameraRig({"C10002": rig.camera("C10001")}, {})


def test_project_matches_opencv_and_rays_pass_through_the_points(rig: mvg.CameraRig) -> None:
    points = _points()
    for view, frame in (("C10001", None), ("HMC_00000001", 17645)):
        camera = rig.camera(view)
        extrinsic = rig.world_to_camera(view, frame)
        expected, _ = cv2.projectPoints(
            points.reshape(-1, 1, 3),
            cv2.Rodrigues(extrinsic[:3, :3])[0],
            extrinsic[:3, 3],
            np.asarray(camera.intrinsic_matrix),
            np.asarray(camera.distortion),
        )
        pixels = rig.project(view, points, frame)
        assert np.allclose(pixels, expected.reshape(-1, 2))
        assert (rig.depth(view, points, frame) > 0).all()
        for point, pixel in zip(points[:5], pixels[:5], strict=True):
            ray = rig.ray(view, pixel, frame)
            assert np.allclose(ray.origin, rig.camera_position(view, frame))
            offset = point - ray.origin
            distance = np.linalg.norm(offset - (offset @ ray.direction) * ray.direction)
            assert distance < 0.05  # mm: ray through the pixel passes through the point


def test_ego_pose_changes_per_frame(rig: mvg.CameraRig) -> None:
    point = np.array([[120.0, -30.0, 60.0]])
    early = rig.project("HMC_00000001", point, 17640)[0]
    late = rig.project("HMC_00000001", point, 17679)[0]
    assert np.linalg.norm(early - late) > 1.0


def test_triangulation_recovers_points_and_drops_the_corrupted_view(rig: mvg.CameraRig) -> None:
    points = _points(1)
    observations = {view: rig.project(view, points) for view in rig.static_views}
    observations["HMC_00000001"] = rig.project("HMC_00000001", points, 17650)
    result = rig.triangulate(observations, 17650)
    assert np.nanmax(np.linalg.norm(result.points - points, axis=1)) < 0.01
    assert result.used.all()
    # Two views are enough; one corrupted third view is identified and removed.
    corrupted = {
        "C10001": observations["C10001"],
        "C10002": observations["C10002"],
        "C10003": observations["C10003"] + np.array([120.0, -80.0]),
    }
    result = rig.triangulate(corrupted, reproj_filter_px=30.0)
    assert np.nanmax(np.linalg.norm(result.points - points, axis=1)) < 0.01
    assert not result.used[:, 2].any() and result.used[:, :2].all()
    assert (result.view_counts == 2).all()
    # Without the filter the corrupted view pulls the estimate away.
    unfiltered = rig.triangulate(corrupted, reproj_filter_px=None)
    assert np.nanmedian(np.linalg.norm(unfiltered.points - points, axis=1)) > 1.0
    # Missing observations (NaN) per point are handled; a lone view yields NaN.
    partial = {view: obs.copy() for view, obs in observations.items() if view != "HMC_00000001"}
    partial["C10001"][0] = np.nan
    partial["C10002"][0] = np.nan
    partial["C10003"][1] = np.nan
    result = rig.triangulate(partial)
    assert np.isnan(result.points[0]).all() and result.view_counts[0] == 0
    assert np.linalg.norm(result.points[1] - points[1]) < 0.01 and result.view_counts[1] == 2


def test_dlt_handles_nan_patterns_directly() -> None:
    shifted = np.array([[1, 0, 0, -100.0], [0, 1, 0, 0], [0, 0, 1, 0]])
    projections = np.stack([np.eye(4)[:3], shifted])
    world = np.array([[10.0, 5.0, 500.0], [-20.0, 30.0, 800.0]])
    normalised = np.stack(
        [
            (world[:, :2] / world[:, 2:]),
            ((world + np.array([-100.0, 0.0, 0.0]))[:, :2] / world[:, 2:]),
        ]
    )
    recovered = mvg.dlt_triangulate(normalised, projections)
    assert np.allclose(recovered, world, atol=1e-6)
    normalised[1, 1] = np.nan
    recovered = mvg.dlt_triangulate(normalised, projections)
    assert np.allclose(recovered[0], world[0]) and np.isnan(recovered[1]).all()


def test_epipolar_distance_is_zero_for_a_consistent_pair(rig: mvg.CameraRig) -> None:
    point = np.array([[35.0, -40.0, 20.0]])
    a = rig.project("C10001", point)[0]
    b = rig.project("C10002", point)[0]
    assert rig.epipolar_distance("C10001", a, "C10002", b) < 0.05
    assert rig.epipolar_distance("C10001", a, "C10002", b + np.array([0.0, 40.0])) > 10.0
    ego = rig.project("HMC_00000001", point, 17660)[0]
    assert rig.epipolar_distance("C10001", a, "HMC_00000001", ego, 17660) < 0.05


def test_table_plane_uses_camera_derived_down_and_recovers_a_tilted_table(
    rig: mvg.CameraRig,
) -> None:
    # The fixture cameras all have image-down roughly along +Y, so 'down' is derived, not
    # assumed; the resting fingertips lie on a plane slightly tilted from the axes.
    rng = np.random.default_rng(5)
    normal = np.array([0.05, -1.0, 0.03])
    normal /= np.linalg.norm(normal)
    landmarks: dict[str, dict[str, list[list[float]]]] = {}
    confidences: dict[str, dict[str, float]] = {}
    for frame in range(17640, 17640 + 300):
        hands = {}
        for hand in ("0", "1"):
            base = rng.uniform([-200, 0, -200], [200, 0, 200])
            joints = base + rng.normal(0.0, 3.0, size=(21, 3))
            # Fingertips (0-4) rest on the plane in 30 % of frames; otherwise the hand hovers.
            if rng.random() < 0.3:
                joints[:5] -= np.outer(joints[:5] @ normal, normal)
                joints[:5] += rng.normal(0.0, 1.0, size=(5, 1)) * normal
            else:
                joints -= rng.uniform(40, 250) * normal
            hands[hand] = joints.tolist()
        landmarks[str(frame)] = hands
        confidences[str(frame)] = {"0": 0.95, "1": 0.9 if frame % 7 else 0.2}
    plane = rig.fit_table_plane(
        landmarks, confidences, pose_frames=range(17640, 17640 + 300), lowest_fraction=0.05
    )
    fitted = np.asarray(plane.normal)
    assert fitted @ (-np.asarray(plane.down_vector)) > 0  # normal points up
    angle = np.degrees(np.arccos(abs(fitted @ normal)))
    assert angle < 5.0
    assert plane.residual_rms_mm < 5.0
    assert plane.fingertips_below_plane_by_20mm_fraction < 0.02
    assert plane.down_vector_source == "mean_static_camera_y_axis"
    # Ray/plane intersection returns a point on the plane in front of the camera.
    ray = rig.ray("C10001", rig.project("C10001", np.array([[0.0, 0.0, 0.0]]))[0])
    hit = plane.intersect_ray(ray)
    assert hit is not None and abs(plane.signed_distance(hit[None])[0]) < 1e-6


# --- real data -------------------------------------------------------------------------


@pytest.mark.real_data
def test_rig_loads_all_twelve_views_and_reproduces_the_dataset_projection() -> None:
    from battle.assembly101_camera_fit import PoseMembers
    from battle.assembly101_clock_offset import SHIPPED_2D_WINDOW, npz_key

    require_artifact(SHIPPED_2D_WINDOW)
    root = Path.cwd()
    rig = mvg.CameraRig.load(root, pose_frames=range(17640, 17640 + 200))
    assert set(rig.views) == set(mvg.ALL_VIEWS)
    assert rig.pose_frame("C10379", 0) == 17649 and rig.pose_frame("HMC_21110305", 0) == 17640
    members = PoseMembers(root)
    with np.load(root / SHIPPED_2D_WINDOW) as archive:
        shipped = {view: np.asarray(archive[npz_key(view)]) for view in ("C10395", "HMC_21110305")}
    frame = 17700
    world = np.asarray(members.landmarks3d[str(frame)]["1"])
    static = rig.project("C10395", world)
    assert np.abs(static - shipped["C10395"][frame - 17640, 1]).max() < 0.01
    ego = rig.project("HMC_21110305", world, frame)
    assert np.abs(ego - shipped["HMC_21110305"][frame - 17640, 1]).max() < 3.0
    observations = {view: rig.project(view, world) for view in ("C10379", "C10115", "C10404")}
    result = rig.triangulate(observations)
    assert np.linalg.norm(result.points - world, axis=1).max() < 0.01


@pytest.mark.real_data
def test_rig_check_report_shows_sub_millimetre_static_agreement() -> None:
    report = mvg.RigCheckReport.model_validate_json(
        require_artifact(mvg.CHECK_OUTPUT_ROOT / "report.json").read_text()
    )
    for check in report.projection:
        limit = 0.01 if not check.view.startswith("HMC_") else 6.0
        assert check.rms_pixels < limit, check
    all_static = next(c for c in report.triangulation if len(c.views) == 8)
    assert all_static.p95_error_mm < 0.01
    assert report.table_plane.normal_vs_down_deg < 10.0
    assert report.table_plane.fingertips_below_plane_by_20mm_fraction < 0.01
