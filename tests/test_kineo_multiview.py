"""battle-kineo-multiview: YAML generation, camera conventions, alignment, PKL parsing."""

from __future__ import annotations

import enum
import importlib.util
import json
import pickle
import sys
import types
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import yaml
from conftest import require_artifact
from test_multiview_geometry import UP, _camera, _look_at, _rule

from battle import kineo_multiview as km
from battle import multiview_geometry as mvg
from battle.overnight_queue import QueueSpec

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUN_DIRECTORIES = {arm: REPOSITORY_ROOT / km.RUN_ROOT_TEMPLATE.format(arm=arm) for arm in km.ARMS}


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    k = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], dtype=float
    )
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


@pytest.fixture
def static_rig() -> mvg.CameraRig:
    target = np.array([0.0, 0.0, 0.0])
    positions = {
        "C10001": [600.0, -700.0, -900.0],
        "C10002": [-800.0, -650.0, -700.0],
        "C10003": [100.0, -900.0, 800.0],
        "C10004": [-300.0, -800.0, 900.0],
    }
    cameras = {
        view: _camera(view, _look_at(np.array(position), target, UP))
        for view, position in positions.items()
    }
    rules = {
        "C10001": _rule("C10001:rgb", 5),
        "C10002": _rule("C10002:rgb", 6),
        "C10003": _rule("C10003:rgb", 7),
        "C10004": _rule("C10004:rgb", 9),
    }
    return mvg.CameraRig(cameras, rules)


# -- geometry ------------------------------------------------------------------------------------


def test_scale_intrinsics_scales_focal_and_principal_point_only() -> None:
    k = np.array([[1250.0, 0.0, 955.0], [0.0, 1240.0, 530.0], [0.0, 0.0, 1.0]])
    scaled = km.scale_intrinsics(k, 2 / 3)
    assert np.allclose(scaled[0], [1250 * 2 / 3, 0.0, 955 * 2 / 3])
    assert np.allclose(scaled[1], [0.0, 1240 * 2 / 3, 530 * 2 / 3])
    assert np.allclose(scaled[2], [0.0, 0.0, 1.0])


def test_world_to_camera_inverts_the_dataset_pose() -> None:
    pose = _look_at(np.array([600.0, -700.0, -900.0]), np.zeros(3), UP)
    rotation, translation = km.world_to_camera(pose)
    point = np.array([120.0, -40.0, 75.0])
    expected = (np.linalg.inv(pose) @ np.append(point, 1.0))[:3]
    assert np.allclose(rotation @ point + translation, expected)
    assert np.allclose(km.camera_center(rotation, translation), pose[:3, 3])
    assert np.allclose(rotation @ rotation.T, np.eye(3))


def test_umeyama_recovers_a_known_similarity() -> None:
    rng = np.random.default_rng(3)
    source = rng.uniform(-1, 1, size=(8, 3))
    rotation = _rotation(np.array([0.3, -1.0, 0.5]), 0.9)
    scale, translation = 1000.0, np.array([120.0, -300.0, 45.0])
    target = km.apply_similarity(source, scale, rotation, translation)
    s, r, t = km.umeyama(source, target)
    assert abs(s - scale) < 1e-6
    assert np.allclose(r, rotation, atol=1e-9)
    assert np.allclose(t, translation, atol=1e-6)
    assert km.rotation_angle_deg(r, rotation) < 1e-6
    s_fixed, r_fixed, _ = km.umeyama(source, target, with_scale=False)
    assert s_fixed == 1.0 and np.allclose(r_fixed, rotation, atol=1e-9)
    with pytest.raises(ValueError, match="at least three"):
        km.umeyama(source[:2], target[:2])


def test_rotation_angle_is_geodesic() -> None:
    r = _rotation(np.array([0.0, 0.0, 1.0]), np.radians(12.5))
    assert abs(km.rotation_angle_deg(np.eye(3), r) - 12.5) < 1e-9


# -- clock alignment -----------------------------------------------------------------------------


def test_view_start_frames_shift_earlier_started_views_forward(static_rig: mvg.CameraRig) -> None:
    reference, starts = km.view_start_frames(static_rig.clock_rules, static_rig.static_views)
    assert reference == "C10004"  # latest-starting camera (+9)
    assert {view: s.start_frame for view, s in starts.items()} == {
        "C10001": 2,  # (9-5)/2 = 2
        "C10002": 2,  # (9-6)/2 = 1.5 -> 2
        "C10003": 1,  # (9-7)/2 = 1
        "C10004": 0,
    }
    assert starts["C10002"].residual_pose_frames == -1.0
    assert abs(starts["C10002"].residual_seconds + 1 / 60) < 1e-12
    assert starts["C10001"].residual_pose_frames == 0.0
    # Trimmed frame 0 of every view shows the reference's pose frame to within one pose frame.
    for view, start in starts.items():
        shown = static_rig.pose_frame(view, start.start_frame)
        assert abs(shown - static_rig.pose_frame(reference, 0)) <= 1


# -- GT annotations ------------------------------------------------------------------------------


def test_gt_annotations_use_proxy_scale_metres_and_world_to_camera(
    static_rig: mvg.CameraRig, tmp_path: Path
) -> None:
    views = ("C10001", "C10002", "C10003")
    intrinsics, extrinsics = km.build_gt_annotations(
        static_rig, views, image_scale=2 / 3, translation_scale=1e-3
    )
    assert [row["view_id"] for row in intrinsics["annotations"]] == list(views)
    row = intrinsics["annotations"][0]
    assert row["resolution_hw"] == [720, 1280]
    assert row["distortion_model"] == "brown_conrady"
    assert len(row["distortion_coefficients"]) == 5
    assert np.allclose(np.array(row["K"])[0, 0], 1250.0 * 2 / 3)
    ext = extrinsics["annotations"][1]
    rotation, translation = km.world_to_camera(static_rig.camera_to_world("C10002"))
    assert np.allclose(ext["R"], rotation)
    assert np.allclose(ext["t"], translation / 1000.0)
    # Round trip through the restricted unpickler used on Kineo's own exports.
    path = tmp_path / "camera_extrinsics.pkl"
    path.write_bytes(pickle.dumps(extrinsics, protocol=4))
    parsed = km.parse_extrinsics(km.load_kineo_pkl(path))
    assert set(parsed) == set(views)
    assert np.allclose(
        km.camera_center(*parsed["C10002"]) * 1000, static_rig.camera_position("C10002")
    )


def test_gt_annotations_refuse_ego_cameras() -> None:
    rig = mvg.CameraRig({"HMC_00000001": _camera("HMC_00000001", None, ego=True)}, {})
    with pytest.raises(ValueError, match="ego camera"):
        km.build_gt_annotations(rig, ("HMC_00000001",), image_scale=1.0, translation_scale=1.0)


# -- YAML generation -----------------------------------------------------------------------------


def _stock_stage(order: int, extra: dict | None = None) -> dict:
    stage = {
        "_target_": f"kineo.pipeline.stages.stage_{order}.Stage{order}",
        "name": f"Stage {order}",
        "order": order,
        "runtime_cfg": {"_target_": f"kineo.pipeline.stages.stage_{order}.Config{order}"},
    }
    if extra:
        stage["runtime_cfg"].update(extra)
    return stage


@pytest.fixture
def stock_config() -> dict:
    """Shape of Kineo's `nlf_single_person_sam2.yaml` with every stage the generator touches."""
    stages = {
        "sam2_semiauto_bbox_detection_rtmlib": _stock_stage(5),
        "moge_intrinsics_estimation": _stock_stage(10),
        "nlf_smpl_keypoints_detection": _stock_stage(20),
        "global_time_resampling": _stock_stage(30, {"target_fps": 50}),
        "keypoints_pairs_sampling": _stock_stage(40),
        "sfm_camera_extrinsics_initialization": _stock_stage(50),
        "bundle_adjustment_sampling": _stock_stage(55),
        "bundle_adjustment_1": _stock_stage(60, {"shared_intrinsics": "${shared_intrinsics}"}),
        "bundle_adjustment_2": _stock_stage(70, {"shared_intrinsics": "${shared_intrinsics}"}),
        "bundle_adjustment_3": _stock_stage(80, {"shared_intrinsics": "${shared_intrinsics}"}),
        "mvs_triangulation": _stock_stage(90),
        "smpl_global_scale_estimation": _stock_stage(100),
        "global_scale_application": _stock_stage(110),
        "scene_reorientation": _stock_stage(120),
        "bundle_adjustment_history_rerun_export": _stock_stage(
            125, {"output_path_template": "./outputs/rerun/{sequence_name}_ba_history.rrd"}
        ),
        "smpl_fitting": _stock_stage(130),
        "background_subtraction": _stock_stage(135),
        "moge_scene_reconstruction": _stock_stage(136),
        "export_bvh": _stock_stage(140, {"output_path_template": "x/{sequence_name}.bvh"}),
        "rerun_export": _stock_stage(
            160,
            {
                "output_path_template": "x/{sequence_name}.rrd",
                "log_pred_smpl": True,
                "log_pred_smpl_skeleton_2d": True,
                "log_world_reconstruction": True,
                "log_gt_cameras": False,
            },
        ),
        "annotations_export": _stock_stage(
            170, {"output_path_template": "x/{annotation_key}.pkl", "not_found_error": True}
        ),
    }
    return {
        "output_root_dir": "./outputs/x",
        "cache_root_dir": "./cache/x",
        "use_cache": False,
        "batch_size": 32,
        "use_half_precision": True,
        "shared_intrinsics": True,
        "sam2_bbox_detection_frame_step": 5,
        "smplx_joints_indices": list(range(55)),
        "pipeline": {"seed": 19, "stages": stages},
    }


def test_generate_selfcal_config_replaces_sam2_and_keeps_the_calibration_stages(
    stock_config: dict,
) -> None:
    config = km.generate_kineo_config(
        stock_config, arm="selfcal", output_root_dir="/run/out", cache_root_dir="/run/cache"
    )
    stages = config["pipeline"]["stages"]
    assert "sam2_semiauto_bbox_detection_rtmlib" not in stages
    assert stages["rtmlib_bbox_detection"]["runtime_cfg"]["best_bbox_only"] is True
    assert stages["rtmlib_bbox_detection"]["order"] == 5
    assert config["shared_intrinsics"] is False
    assert config["use_cache"] is False
    assert config["smplx_joints_indices"] == list(range(55))
    assert stages["global_time_resampling"]["runtime_cfg"]["target_fps"] == 30
    for name in km.KNOWN_REMOVED[1:]:
        assert name in stages, name
    assert "transfer_gt_annotations" not in stages
    rerun = stages["rerun_export"]["runtime_cfg"]
    assert rerun["log_gt_cameras"] is True and rerun["log_pred_smpl"] is True
    assert rerun["output_path_template"] == "${output_root_dir}/{sequence_name}.rrd"
    assert stages["annotations_export"]["runtime_cfg"]["output_path_template"].startswith(
        "${output_root_dir}/annotations/"
    )
    assert stages["annotations_export"]["runtime_cfg"]["not_found_error"] is False
    assert config["output_root_dir"] == "/run/out"
    # stage order is preserved and the YAML round-trips
    orders = [stage["order"] for stage in stages.values()]
    assert orders == sorted(orders)
    text = km.dump_yaml(config)
    assert yaml.safe_load(text) == config
    assert km.stage_names(config)[0] == "rtmlib_bbox_detection"
    # the stock dict was not mutated
    assert "sam2_semiauto_bbox_detection_rtmlib" in stock_config["pipeline"]["stages"]


def test_generate_known_config_transfers_cameras_and_drops_estimation(stock_config: dict) -> None:
    config = km.generate_kineo_config(
        stock_config, arm="known", output_root_dir="/run/out", cache_root_dir="/run/cache"
    )
    stages = config["pipeline"]["stages"]
    assert km.stage_names(config) == (
        "transfer_gt_annotations",
        "rtmlib_bbox_detection",
        "nlf_smpl_keypoints_detection",
        "global_time_resampling",
        "mvs_triangulation",
        "rerun_export",
        "annotations_export",
    )
    transfer = stages["transfer_gt_annotations"]
    assert transfer["order"] == 0
    assert transfer["runtime_cfg"]["annotations_keys"] == ["camera_intrinsics", "camera_extrinsics"]
    assert transfer["runtime_cfg"]["raise_error_if_not_found"] is True
    rerun = stages["rerun_export"]["runtime_cfg"]
    assert rerun["log_pred_smpl"] is False and rerun["log_world_reconstruction"] is False
    assert "smplx_joints_indices" not in config


def test_generate_config_refuses_a_stock_file_missing_a_stage(stock_config: dict) -> None:
    del stock_config["pipeline"]["stages"]["bundle_adjustment_2"]
    with pytest.raises(KeyError, match="bundle_adjustment_2"):
        km.generate_kineo_config(stock_config, arm="known", output_root_dir="o", cache_root_dir="c")


# -- Kineo PKL parsing ---------------------------------------------------------------------------


class _CameraDistortionModel(enum.Enum):
    """Pickles by reference to Kineo's module path, exactly like Kineo's own exports."""

    BROWN_CONRADY = "brown_conrady"
    OPENCV_FISHEYE = "opencv_fisheye"


_CameraDistortionModel.__module__ = "kineo.annotations.camera_intrinsics"
_CameraDistortionModel.__qualname__ = "CameraDistortionModel"


@contextmanager
def _fake_kineo_modules() -> Iterator[None]:
    """Let pickle resolve the enum's Kineo module path without Kineo installed."""
    names = ("kineo", "kineo.annotations", "kineo.annotations.camera_intrinsics")
    if names[0] in sys.modules:
        yield
        return
    modules = {name: types.ModuleType(name) for name in names}
    modules[names[-1]].CameraDistortionModel = _CameraDistortionModel  # type: ignore[attr-defined]
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name in names:
            sys.modules.pop(name, None)


def _synthetic_kineo_exports(
    directory: Path,
    rig: mvg.CameraRig,
    views: tuple[str, ...],
    *,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
    wrists_mm: np.ndarray,
) -> None:
    """Write `camera_*.pkl`, `keypoints_3d.pkl`, `stage_timings.pkl` the way Kineo's
    `AnnotationsExportStage` does (`Annotations.to_dict()` -> pickle), for a fake Kineo world
    related to the dataset world by `X_mm = scale R X_kineo + translation`."""
    extrinsics = []
    intrinsics = []
    for view in views:
        r_c, t_c = km.world_to_camera(rig.camera_to_world(view))
        extrinsics.append(
            {
                "view_id": view,
                "frame_idx": 0,
                "R": (r_c @ rotation).tolist(),
                "t": ((r_c @ translation + t_c) / scale).tolist(),
            }
        )
        intrinsics.append(
            {
                "view_id": view,
                "frame_idx": 0,
                "K": km.scale_intrinsics(
                    np.asarray(rig.camera(view).intrinsic_matrix), 2 / 3
                ).tolist(),
                "distortion_coefficients": list(rig.camera(view).distortion[:5]),
                "distortion_model": _CameraDistortionModel.BROWN_CONRADY,
                "resolution_hw": (720, 1280),
            }
        )
    names = [f"joint_{i}" for i in range(55)] + [f"vertex_{i}" for i in range(4)]
    names[20], names[21] = "left_wrist", "right_wrist"
    frame_count = wrists_mm.shape[0]
    rows = []
    for frame in range(frame_count):
        xyz = np.zeros((59, 3))
        scores = np.zeros(59)
        for s in range(2):
            wrist = wrists_mm[frame, s]
            if np.isfinite(wrist).all():
                xyz[20 + s] = (rotation.T @ (wrist - translation)) / scale
                scores[20 + s] = 0.9
        xyz[0] = (rotation.T @ (np.array([0.0, -100.0, 0.0]) - translation)) / scale
        scores[0] = 1.0
        rows.append(
            {
                "frame_idx": frame,
                "subject_id": "subject_0",
                "xyz": xyz.tolist(),
                "annotated": [True] * 59,
                "scores": scores.tolist(),
                "format": "nlf_smplx",
            }
        )
    keypoints = {
        "metadata": {
            "version": "1",
            "formats": [
                {
                    "name": "nlf_smplx",
                    "n_keypoints": 59,
                    "keypoints_names": names,
                    "keypoints_connectivity": [(0, 20), (0, 21), (20, 57)],
                }
            ],
        },
        "annotations": rows,
    }
    timings = {
        "annotations": [
            {"stage_name": "Rtmlib Bbox Detection", "stage_idx": 0, "duration_seconds": 10.0},
            {"stage_name": "MVS Triangulation", "stage_idx": 1, "duration_seconds": 5.0},
            {"stage_name": "Annotations Export", "stage_idx": 2, "duration_seconds": 1.0},
        ]
    }
    directory.mkdir(parents=True, exist_ok=True)
    with _fake_kineo_modules():
        _write_pickles(directory, extrinsics, intrinsics, keypoints, timings)


def _write_pickles(directory: Path, extrinsics, intrinsics, keypoints, timings) -> None:
    for name, payload in (
        ("camera_extrinsics", {"metadata": {"version": "1"}, "annotations": extrinsics}),
        ("camera_intrinsics", {"metadata": {"version": "1"}, "annotations": intrinsics}),
        ("keypoints_3d", keypoints),
        ("stage_timings", timings),
        (
            "global_scale",
            {"metadata": {"version": "1"}, "annotations": [{"frame_idx": 0, "scale": 0.98}]},
        ),
    ):
        (directory / f"{name}.pkl").write_bytes(pickle.dumps(payload, protocol=4))


def test_parse_kineo_exports_from_synthetic_pkls(static_rig: mvg.CameraRig, tmp_path: Path) -> None:
    views = static_rig.static_views
    wrists = np.full((6, 2, 3), np.nan)
    wrists[:, 0] = [10.0, -20.0, 30.0]
    wrists[2:, 1] = [-40.0, -10.0, 60.0]
    _synthetic_kineo_exports(
        tmp_path,
        static_rig,
        views,
        scale=1000.0,
        rotation=np.eye(3),
        translation=np.zeros(3),
        wrists_mm=wrists,
    )
    extrinsics = km.parse_extrinsics(km.load_kineo_pkl(tmp_path / "camera_extrinsics.pkl"))
    assert set(extrinsics) == set(views)
    intrinsics = km.parse_intrinsics(km.load_kineo_pkl(tmp_path / "camera_intrinsics.pkl"))
    assert intrinsics["C10001"]["distortion_model"] == "brown_conrady"
    assert intrinsics["C10001"]["resolution_hw"] == (720, 1280)
    body = km.parse_keypoints_3d(km.load_kineo_pkl(tmp_path / "keypoints_3d.pkl"))
    assert body.names[20] == "left_wrist" and len(body.names) == 55
    assert body.xyz.shape == (6, 55, 3)
    assert body.edges == ((0, 20), (0, 21))  # the vertex edge is dropped
    assert body.subject_ids == ("subject_0",)
    assert np.allclose(body.xyz[0, 20] * 1000, wrists[0, 0])
    assert body.scores[0, 21] == 0.0
    timings = km.parse_stage_timings(km.load_kineo_pkl(tmp_path / "stage_timings.pkl"))
    assert km.time_to_first_3d_output(timings) == 15.0
    assert km.time_to_first_3d_output(timings[:1]) is None


def test_restricted_unpickler_refuses_other_classes(tmp_path: Path) -> None:
    path = tmp_path / "bad.pkl"
    path.write_bytes(pickle.dumps({"annotations": [Path("x")]}))
    with pytest.raises(pickle.UnpicklingError, match="refusing"):
        km.load_kineo_pkl(path)
    with pytest.raises(ValueError, match="more than one"):
        km.parse_extrinsics(
            {
                "annotations": [
                    {"view_id": "A", "R": np.eye(3).tolist(), "t": [0, 0, 0]},
                    {"view_id": "A", "R": np.eye(3).tolist(), "t": [0, 0, 1]},
                ]
            }
        )


# -- alignment and wrists ------------------------------------------------------------------------


def test_align_cameras_selfcal_recovers_scale_rotation_and_zero_errors(
    static_rig: mvg.CameraRig, tmp_path: Path
) -> None:
    views = static_rig.static_views
    rotation = _rotation(np.array([1.0, 0.2, -0.4]), 1.3)
    translation = np.array([250.0, -80.0, 400.0])
    _synthetic_kineo_exports(
        tmp_path,
        static_rig,
        views,
        scale=1000.0,
        rotation=rotation,
        translation=translation,
        wrists_mm=np.full((1, 2, 3), np.nan),
    )
    kineo = km.parse_extrinsics(km.load_kineo_pkl(tmp_path / "camera_extrinsics.pkl"))
    alignment, cameras = km.align_cameras(kineo, static_rig, views, arm="selfcal")
    assert abs(alignment.scale_mm_per_kineo_unit - 1000.0) < 1e-6
    assert np.allclose(np.asarray(alignment.rotation), rotation, atol=1e-9)
    assert alignment.camera_center_rms_mm < 1e-6
    assert all(c.rotation_error_deg < 1e-6 and c.translation_error_mm < 1e-6 for c in cameras)
    # A perturbed camera shows up in its own row after alignment.
    r, t = kineo["C10003"]
    kineo["C10003"] = (r, t + np.array([0.0, 0.0, 0.05]))
    alignment, cameras = km.align_cameras(kineo, static_rig, views, arm="selfcal")
    by_view = {c.view: c for c in cameras}
    assert by_view["C10003"].translation_error_mm > by_view["C10001"].translation_error_mm
    assert alignment.camera_center_rms_mm > 1.0


def test_align_cameras_known_uses_fixed_metres(static_rig: mvg.CameraRig, tmp_path: Path) -> None:
    views = static_rig.static_views
    _synthetic_kineo_exports(
        tmp_path,
        static_rig,
        views,
        scale=1000.0,
        rotation=np.eye(3),
        translation=np.zeros(3),
        wrists_mm=np.full((1, 2, 3), np.nan),
    )
    kineo = km.parse_extrinsics(km.load_kineo_pkl(tmp_path / "camera_extrinsics.pkl"))
    alignment, cameras = km.align_cameras(kineo, static_rig, views, arm="known")
    assert alignment.method == "fixed_metres_to_millimetres_identity"
    assert alignment.scale_mm_per_kineo_unit == 1000.0
    assert all(c.translation_error_mm < 1e-6 and c.rotation_error_deg < 1e-6 for c in cameras)


def test_compare_wrists_reports_distance_and_side_swaps() -> None:
    frames = 10
    dataset = np.zeros((frames, 2, 3))
    dataset[:, 0] = [0.0, 0.0, 0.0]
    dataset[:, 1] = [200.0, 0.0, 0.0]
    dataset[7:, 1] = np.nan  # right hand absent for three frames
    kineo = dataset.copy()
    kineo[:, 0, 2] += 12.0  # left 12 mm off
    kineo[:3, 1] = dataset[:3, 0]  # right hand sits on the dataset's left wrist for 3 frames
    left, right = km.compare_wrists(kineo, dataset)
    assert left.frames_compared == 10 and abs(left.median_mm - 12.0) < 1e-9
    assert left.swapped_side_closer_fraction == 0.0
    assert right.frames_compared == 7
    assert abs(right.swapped_side_closer_fraction - 3 / 7) < 1e-9
    assert right.p90_mm is not None and right.p90_mm > 100.0
    _, empty = km.compare_wrists(np.full((2, 2, 3), np.nan), dataset[:2])
    assert empty.frames_compared == 0 and empty.median_mm is None


# -- queue job and runner --------------------------------------------------------------------------


def test_queue_job_validates_against_the_overnight_queue_spec(tmp_path: Path) -> None:
    trims = {view: tmp_path / f"{view}.mp4" for view in ("C10379", "C10395")}
    argv = km.runner_argv(
        repository_root=REPOSITORY_ROOT,
        run_directory=tmp_path,
        config_path=tmp_path / "kineo_known.yaml",
        sequence_name="seq",
        gt_directory=tmp_path / "gt_annotations",
        runtime_json=tmp_path / "kineo_runtime.json",
        views=("C10379", "C10395"),
        trims=trims,
    )
    job = km.queue_job(
        name="kineo-multiview-known", argv=argv, kineo_root=Path("/k"), timeout_s=1800
    )
    spec = QueueSpec.model_validate(job)
    assert spec.jobs[0].command[:3] == ["pixi", "run", "python"]
    assert spec.jobs[0].command[3].endswith("scripts/kineo_multiview_runner.py")
    assert spec.jobs[0].cwd == "/k" and spec.jobs[0].timeout_s == 1800
    assert spec.jobs[0].env == {"CUDA_VISIBLE_DEVICES": "0"}
    assert argv[-2:] == [f"C10379={trims['C10379']}", f"C10395={trims['C10395']}"]
    assert km.TIMEOUT_S == {"selfcal": 3600, "known": 1800}


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "kineo_multiview_runner", REPOSITORY_ROOT / km.RUNNER_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_runner_parses_view_specs_and_shares_the_model_free_list() -> None:
    runner = _load_runner()
    assert runner.parse_views(["C10379=/a.mp4", "C10395=/b.mp4"]) == [
        ("C10379", "/a.mp4"),
        ("C10395", "/b.mp4"),
    ]
    with pytest.raises(SystemExit):
        runner.parse_views(["/a.mp4"])
    assert set(runner.MODEL_FREE_STAGES) == set(km.MODEL_FREE_STAGES)


def test_read_queue_record_picks_the_last_job_end(tmp_path: Path) -> None:
    log = tmp_path / "queue.log"
    log.write_text(
        "\n".join(
            [
                json.dumps({"event": "job_start", "job": "kineo-multiview-known"}),
                json.dumps(
                    {
                        "time": "t1",
                        "event": "job_end",
                        "name": "kineo-multiview-known",
                        "state": "failed",
                        "exit_code": 1,
                        "duration_s": 3.0,
                        "stdout_log": "a.log",
                    }
                ),
                "not json",
                json.dumps(
                    {
                        "time": "t2",
                        "event": "job_end",
                        "name": "kineo-multiview-known",
                        "state": "succeeded",
                        "exit_code": 0,
                        "duration_s": 900.5,
                        "stdout_log": "b.log",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    record = km.read_queue_record(log, "kineo-multiview-known")
    assert record is not None and record.state == "succeeded" and record.duration_s == 900.5
    assert km.read_queue_record(log, "other") is None
    assert km.read_queue_record(tmp_path / "missing.log", "x") is None


# -- real data -----------------------------------------------------------------------------------


@pytest.mark.real_data
@pytest.mark.parametrize("arm", km.ARMS)
def test_prepared_run_directory_is_complete(arm: str) -> None:
    run_directory = require_artifact(RUN_DIRECTORIES[arm])
    prepared = km.load_prepare(run_directory)
    assert prepared.arm == arm and prepared.views == km.STATIC_VIEWS
    assert prepared.reference_view == "C10379" and prepared.frame_count == 1800
    starts = {trim.view: trim.start.start_frame for trim in prepared.trims}
    assert starts == {
        "C10095": 2,
        "C10115": 2,
        "C10118": 2,
        "C10119": 1,
        "C10379": 0,
        "C10390": 1,
        "C10395": 2,
        "C10404": 2,
    }
    for trim in prepared.trims:
        assert trim.verified_frame_count == 1800
        assert (REPOSITORY_ROOT / trim.trimmed.uri).is_file()
        assert abs(trim.start.residual_seconds) <= 1 / 60 + 1e-12
    config = yaml.safe_load((REPOSITORY_ROOT / prepared.kineo_config.uri).read_text())
    assert config["shared_intrinsics"] is False
    assert "sam2_semiauto_bbox_detection_rtmlib" not in config["pipeline"]["stages"]
    if arm == "known":
        assert prepared.kineo_stages[0] == "transfer_gt_annotations"
        assert "sfm_camera_extrinsics_initialization" not in prepared.kineo_stages
    else:
        assert "sfm_camera_extrinsics_initialization" in prepared.kineo_stages
        assert "smpl_fitting" in prepared.kineo_stages
    assert prepared.kineo_git.head == km.DEFAULT_KINEO_REVISION
    job_path = REPOSITORY_ROOT / prepared.queue_job
    spec = QueueSpec.model_validate_json(job_path.read_text(encoding="utf-8"))
    assert spec.jobs[0].name == f"kineo-multiview-{arm}"
    assert spec.jobs[0].timeout_s == km.TIMEOUT_S[arm]
    validation = json.loads((run_directory / "validation.json").read_text(encoding="utf-8"))
    assert validation["exit_code"] == 0 and validation["report"]["ok"] is True
    assert validation["report"]["cuda_visible"] is False
    assert set(validation["report"]["video_frame_counts"].values()) == {1800}
    gt = km.parse_extrinsics(
        km.load_kineo_pkl(run_directory / "gt_annotations/camera_extrinsics.pkl")
    )
    rig = mvg.CameraRig.load(REPOSITORY_ROOT, views=km.STATIC_VIEWS)
    for view in km.STATIC_VIEWS:
        assert np.allclose(km.camera_center(*gt[view]) * 1000, rig.camera_position(view), atol=1e-6)


@pytest.mark.real_data
def test_evaluate_and_rerun_on_a_synthetic_kineo_export(tmp_path: Path) -> None:
    """A fake Kineo export built from the dataset's own wrists must evaluate to ~0 disagreement."""
    from battle import assembly101_reference as a101
    from battle.kineo_multiview_review import build_recording

    source = require_artifact(RUN_DIRECTORIES["selfcal"])
    require_artifact(REPOSITORY_ROOT / km.DATASET_REFERENCE_RUN / "hands.jsonl")
    run_directory = tmp_path / "kineo-multiview-selfcal-synthetic"
    run_directory.mkdir()
    prepared = km.load_prepare(source)
    output_root = run_directory / "kineo_outputs"
    prepared = prepared.model_copy(
        update={"kineo_output_root": str(output_root), "kineo_cache_root": str(run_directory)}
    )
    (run_directory / "prepare.json").write_text(prepared.model_dump_json(indent=2))
    rig = mvg.CameraRig.load(REPOSITORY_ROOT, views=km.STATIC_VIEWS)
    dataset = a101.load_reference(km.DATASET_REFERENCE_RUN, REPOSITORY_ROOT, frame_count=1800)
    wrists = km.dataset_wrists(dataset.frames, 1800, min_confidence=0.5)
    rotation = _rotation(np.array([0.1, 1.0, 0.3]), 0.7)
    translation = np.array([300.0, 50.0, -120.0])
    _synthetic_kineo_exports(
        output_root / "annotations" / prepared.sequence_name,
        rig,
        km.STATIC_VIEWS,
        scale=1000.0,
        rotation=rotation,
        translation=translation,
        wrists_mm=wrists,
    )
    manifest_path = km.evaluate(REPOSITORY_ROOT, run_directory, queue_log=tmp_path / "no-queue.log")
    result = km.load_evaluation(manifest_path.parent)
    assert result.arm == "selfcal" and result.views == km.STATIC_VIEWS
    assert abs(result.alignment.scale_mm_per_kineo_unit - 1000.0) < 1e-3
    # The dataset rotations are orthonormal only to JSON precision; 1e-3 deg is far below anything
    # a real Kineo estimate could reach.
    assert result.camera_rotation_error_deg_median < 1e-3
    assert result.camera_translation_error_mm_median < 1e-3
    assert result.kineo_global_scale == 0.98
    for wrist in result.wrists:
        assert wrist.frames_compared > 1500
        assert wrist.median_mm is not None and wrist.median_mm < 1e-6
    assert result.measures.coverage_frames_with_body_3d == 1800
    assert result.measures.time_to_first_3d_output_seconds == 15.0
    assert result.measures.runtime_seconds is None and result.queue_record is None
    assert (run_directory / "body_aligned_mm.npz").is_file()
    rrd = build_recording(REPOSITORY_ROOT, run_directory)
    assert rrd.is_file() and rrd.stat().st_size > 100_000
