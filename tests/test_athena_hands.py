"""battle-athena-hands: joint mapping, clock alignment, side matching, synthetic round trip."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from conftest import require_artifact
from test_multiview_geometry import UP, _camera, _look_at, _rule

from battle import athena_hands as ah
from battle import multiview_geometry as mvg
from battle.assembly101_pose_schemas import ASSEMBLY101_JOINT_NAMES

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_joint_mapping_covers_twenty_common_joints_and_drops_palm_and_thumb_cmc() -> None:
    dataset_indices = [d for _, d, _ in ah.COMMON_JOINTS]
    method_indices = [m for _, _, m in ah.COMMON_JOINTS]
    assert len(ah.COMMON_JOINTS) == 20
    assert sorted(dataset_indices) == list(range(20))  # dataset 20 (palm) dropped
    assert sorted(method_indices) == [i for i in range(21) if i != 1]  # MediaPipe thumb_cmc
    assert all(ASSEMBLY101_JOINT_NAMES[d] == name for name, d, _ in ah.COMMON_JOINTS)
    assert ah.COMMON_JOINT_NAMES[ah.WRIST] == "wrist"
    assert [ah.COMMON_JOINT_NAMES[i] for i in ah.FINGERTIPS] == [
        "thumb_tip",
        "index_tip",
        "middle_tip",
        "ring_tip",
        "pinky_tip",
    ]
    assert all(0 <= a < 20 and 0 <= b < 20 for a, b in ah.COMMON_EDGES)


def test_nearest_analysis_frame_reports_the_half_frame_residual() -> None:
    reference = _rule("C10379:rgb", 9)
    pose_frame = reference.pose_frame(100)  # 17849
    frame, residual = ah.nearest_analysis_frame(_rule("C10095:rgb", 5), pose_frame)
    assert (frame, residual) == (102, 0)
    frame, residual = ah.nearest_analysis_frame(_rule("C10115:rgb", 6), pose_frame)
    assert frame == 102 and residual == 1  # exact shift is +1.5 frames; rounds up
    frame, residual = ah.nearest_analysis_frame(_rule("00000001:mono10bit", 0), pose_frame)
    assert frame == 105 and residual == 1
    assert ah.nearest_analysis_frame(_rule("C10095:rgb", 5), 17640)[0] == 0


def test_side_matching_is_handedness_agnostic_and_thresholded() -> None:
    left = np.zeros((21, 2))
    left[0] = (100.0, 100.0)
    right = np.zeros((21, 2))
    right[0] = (900.0, 120.0)
    wrists = {"left": np.array([905.0, 118.0]), "right": np.array([98.0, 104.0])}
    assignment = ah.match_hands_to_dataset([left, right], wrists, threshold_px=50.0)
    assert {side: index for side, (index, _) in assignment.items()} == {"left": 1, "right": 0}
    far = {"left": np.array([500.0, 500.0])}
    assert ah.match_hands_to_dataset([left, right], far, threshold_px=50.0) == {}
    one = ah.match_hands_to_dataset([left], wrists, threshold_px=50.0)
    assert set(one) == {"right"} and one["right"][0] == 0
    assert ah.match_hands_to_dataset([], wrists, threshold_px=50.0) == {}


def test_savgol_preserves_cubics_and_undistorted_pixels_apply_the_intrinsics() -> None:
    x = np.arange(40, dtype=np.float64)
    cubic = 0.01 * x**3 - 0.5 * x**2 + 3.0 * x - 7.0
    assert np.allclose(ah.savgol(cubic, 5, 3), cubic, atol=1e-8)
    intrinsics = np.array([[[1000.0, 0.0, 960.0], [0.0, 1100.0, 540.0], [0.0, 0.0, 1.0]]])
    undist = np.array([[[0.1, -0.2], [np.nan, np.nan]]])
    pixels = ah._undistorted_pixels(undist, intrinsics)
    assert np.allclose(pixels[0, 0], [1060.0, 320.0])
    assert np.isnan(pixels[0, 1]).all()


class _FakeMembers:
    """The two pose-archive members `triangulate_hands` reads, keyed like the dataset."""

    def __init__(self, landmarks: dict[int, dict[str, np.ndarray]], confidence: float = 0.95):
        self.landmarks3d = {
            str(k): {side: hand.tolist() for side, hand in hands.items()}
            for k, hands in landmarks.items()
        }
        self.confidences = {
            str(k): {side: confidence for side in hands} for k, hands in landmarks.items()
        }


def _synthetic_scene(frame_count: int = 24):
    target = np.array([0.0, 0.0, 0.0])
    poses = {
        "C10001": _look_at(np.array([600.0, -700.0, -900.0]), target, UP),
        "C10002": _look_at(np.array([-800.0, -650.0, -700.0]), target, UP),
        "C10003": _look_at(np.array([100.0, -900.0, 800.0]), target, UP),
    }
    cameras = {view: _camera(view, pose) for view, pose in poses.items()}
    rules = {
        "C10001": _rule("C10001:rgb", 5),
        "C10002": _rule("C10002:rgb", 9),
        "C10003": _rule("C10003:rgb", 6),
    }
    rig = mvg.CameraRig(cameras, rules)
    rng = np.random.default_rng(3)
    base = rng.uniform([-60, -40, -60], [60, 0, 60], size=(21, 3))
    landmarks: dict[int, dict[str, np.ndarray]] = {}
    for k in range(17640, 17640 + 2 * frame_count + 40):
        t = (k - 17640) * 0.5
        landmarks[k] = {
            "0": base + np.array([-150.0 + 2.0 * t, -20.0, 30.0 * np.sin(t / 10)]),
            "1": base + np.array([150.0 - 1.5 * t, -25.0, -40.0 * np.cos(t / 12)]),
        }
    return rig, landmarks


def _view_hands_from_truth(
    rig: mvg.CameraRig,
    view: str,
    landmarks: dict[int, dict[str, np.ndarray]],
    *,
    frames: int,
    noise_px: float,
    swap: bool,
) -> ah.ViewHands:
    rule = rig.clock_rule(view)
    hands = ah.ViewHands(view=view, run_directory=Path(f"runs/fake-{view}"), source="mediapipe")
    rng = np.random.default_rng(hash(view) % 1000)
    for frame in range(frames + 5):  # shifted views are read up to 2.5 frames past the window
        k = rule.pose_frame(frame)
        detections = []
        for side in ("0", "1"):
            world = landmarks[k][side]
            pixels = np.full((21, 2), np.nan)
            projected = rig.project(view, world[ah.DATASET_INDICES], None)
            pixels[ah.METHOD_INDICES] = projected + rng.normal(0.0, noise_px, size=projected.shape)
            pixels[1] = pixels[0] + (5.0, 5.0)  # MediaPipe thumb_cmc, never used
            detections.append(pixels)
        if swap:
            detections.reverse()
        hands.frames[frame] = detections
        hands.confidences[frame] = [0.9, 0.9]
        hands.camera_relative_3d[frame] = [None, None]
    hands.frame_count = frames + 5
    return hands


def test_synthetic_hands_round_trip_through_the_rig_triangulator() -> None:
    frames = 24
    rig, landmarks = _synthetic_scene(frames)
    members = _FakeMembers(landmarks)
    views = ("C10001", "C10002", "C10003")
    hands = {
        view: _view_hands_from_truth(
            rig, view, landmarks, frames=frames, noise_px=0.3, swap=(view == "C10003")
        )
        for view in views
    }
    result, alignments = ah.triangulate_hands(
        REPOSITORY_ROOT,
        views=views,
        hands=hands,
        rig=rig,
        members=members,
        reference_view="C10002",
        frame_count=frames,
        triangulator="rig",
    )
    assert result.points.shape == (frames, 2, 20, 3)
    solved = ~np.isnan(result.points[..., 0])
    assert solved.all()
    # C10002 carries the reference clock (+9): C10001 (+5) is two frames later with no residual,
    # C10003 (+6) is 1.5 frames later and rounds to a +1 pose-frame residual.
    assert alignments["C10001"].analysis_frame_shift == 2.0
    assert alignments["C10001"].residual_pose_frames_used == (0,)
    assert alignments["C10003"].residual_pose_frames_used == (1,)
    # Detections were re-synthesised at each view's own frame, so the reference-clock truth
    # differs only by the half-frame residual motion (< 2 mm here); the swapped view still
    # matched by proximity.
    error = np.linalg.norm(result.points - result.dataset, axis=-1)
    assert np.median(error) < 2.0 and error.max() < 6.0
    assert alignments["C10003"].detections_unmatched == 0
    assert result.used.sum(axis=-1).min() == 3
    assert np.nanmax(result.reprojection_px) < 3.0
    smooth_error = np.linalg.norm(result.smoothed - result.dataset, axis=-1)
    assert np.median(smooth_error) < 2.0


@pytest.mark.skipif(not ah.ATHENA_PYTHON.is_file(), reason="ATHENA venv not installed")
def test_synthetic_hands_round_trip_through_athena_worker() -> None:
    frames = 12
    rig, landmarks = _synthetic_scene(frames)
    members = _FakeMembers(landmarks)
    views = ("C10001", "C10002", "C10003")
    hands = {
        view: _view_hands_from_truth(rig, view, landmarks, frames=frames, noise_px=0.3, swap=False)
        for view in views
    }
    # Corrupt one view's left wrist at its frame 6 (= reference frame 4 under its +2 shift):
    # the filter must drop that view and keep the other two.
    hands["C10001"].frames[6][0][0] += (400.0, -300.0)
    result, _ = ah.triangulate_hands(
        REPOSITORY_ROOT,
        views=views,
        hands=hands,
        rig=rig,
        members=members,
        reference_view="C10002",
        frame_count=frames,
        triangulator="athena",
    )
    error = np.linalg.norm(result.points - result.dataset, axis=-1)
    assert np.median(error) < 2.0
    corrupted = result.used[4, 0, ah.WRIST]
    assert corrupted.tolist() == [False, True, True]
    assert error[4, 0, ah.WRIST] < 6.0
    assert result.smoothed.shape == result.points.shape


@pytest.mark.real_data
def test_wilor_run_loads_with_camera_translation_and_single_view_is_refused(tmp_path: Path) -> None:
    run = Path("runs/wilor-hands-static-60s-overnight-v2")
    require_artifact(REPOSITORY_ROOT / run / "observations.jsonl")
    hands = ah.load_view_hands(REPOSITORY_ROOT, "C10379", run, "wilor")
    assert hands.frame_count == 1800 and hands.frames_with_hands > 1500
    assert hands.camera_translation_attached
    first = next(
        joints for joints in hands.camera_relative_3d.values() if joints and joints[0] is not None
    )
    assert first[0].shape == (21, 3) and first[0][0, 2] > 1.0  # camera-frame depth attached
    with pytest.raises(ValueError, match="at least two views"):
        ah.build_run(
            REPOSITORY_ROOT,
            views=("C10379",),
            hand_source="wilor",
            output_root=tmp_path / "x",
            run_overrides={"C10379": run.as_posix()},
        )
    with pytest.raises(ValueError, match="not C10404"):
        ah.load_view_hands(REPOSITORY_ROOT, "C10404", run, "wilor")


@pytest.mark.real_data
@pytest.mark.parametrize(
    "run",
    ["runs/athena-hands-first-minute-mediapipe", "runs/athena-hands-first-minute-mediapipe-ego"],
)
def test_built_athena_runs_report_sane_cross_source_disagreement(run: str) -> None:
    root = REPOSITORY_ROOT / run
    require_artifact(root / "manifest.json")
    manifest = ah.load_manifest(root)
    assert manifest.triangulator == "athena" and manifest.athena_revision is not None
    assert manifest.frame_count == 1800 and len(manifest.views) >= 8
    assert ("HMC_21110305" in manifest.views) == run.endswith("-ego")
    for hand in manifest.hands:
        groups = {g.group: g for g in hand.disagreement_raw}
        assert hand.frames_with_solution >= 1500
        assert hand.mean_contributing_views is not None and hand.mean_contributing_views >= 3.0
        assert 5.0 < groups["wrist"].median_mm < 40.0
        assert groups["fingertips"].median_mm < 60.0
        assert all(
            v.rms_px_used is None or v.rms_px_used <= 30.0 for v in hand.per_view_reprojection
        )
        steps = {s.series: s for s in hand.steadiness}
        assert steps["triangulated_wrist_raw"].median_step < 6.0
    rows = [json.loads(line) for line in (root / "hands.jsonl").read_text().splitlines()]
    assert len(rows) == 1800 and sum(bool(r["hands"]) for r in rows) >= 1500
    assert any(
        boundary.startswith("Triangulated hands versus") for boundary in manifest.claim_boundaries
    )
    assert (root / "hands.rrd").is_file()
