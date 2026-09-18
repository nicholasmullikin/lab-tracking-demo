"""battle-egoexo-correspondence: clock mapping, mask I/O, hull projection, evaluate, typing."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from conftest import require_artifact
from PIL import Image
from test_multiview_geometry import UP, _camera, _look_at, _rule

from battle import egoexo_correspondence as eec
from battle import multiview_geometry as mvg
from battle.overnight_queue import QueueSpec
from battle.schemas import FrameObservations

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXO_WH = (1280, 720)
EGO_WH = (954, 720)
FIXTURE_FRAMES = (0, 150)


# -- clock mapping and small helpers ---------------------------------------------------------------


def test_keyframes_are_twelve_every_150() -> None:
    assert eec.keyframes() == tuple(range(0, 1800, 150))
    assert len(eec.keyframes()) == 12


def test_mapped_analysis_frame_rounds_the_half_frame_down() -> None:
    exo = _rule("C10379:rgb", 9)
    ego = _rule("21110305:mono10bit", 0)
    for p in (0, 150, 1650, 7):
        target, source_pose, target_pose, residual = eec.mapped_analysis_frame(exo, ego, p)
        assert target == p + 4
        assert source_pose == 17649 + 2 * p
        assert target_pose == 17648 + 2 * p
        assert residual == -1
    same = eec.mapped_analysis_frame(exo, exo, 30)
    assert same == (30, 17709, 17709, 0)
    # An even offset difference maps exactly.
    six = _rule("C10115:rgb", 6)
    assert eec.mapped_analysis_frame(six, ego, 10)[:1] == (13,)
    late = eec.mapped_analysis_frame(ego, exo, 10)
    # ego frame 10 (pose 17660) is shown by exo frame 5.5 -> 5 (pose 17659), one pose frame early
    assert late == (5, 17660, 17659, -1)


def test_mapped_analysis_frame_refuses_frames_before_the_target_proxy() -> None:
    exo = _rule("C10379:rgb", 0)
    ego = _rule("21110305:mono10bit", 9)
    with pytest.raises(ValueError, match="precedes"):
        eec.mapped_analysis_frame(exo, ego, 0)


def test_iou_edge_cases() -> None:
    a = np.zeros((4, 4), dtype=bool)
    b = np.zeros((4, 4), dtype=bool)
    assert eec.iou(a, b) is None
    a[:2] = True
    assert eec.iou(a, b) == 0.0
    assert eec.iou(a, a) == 1.0
    b[1:3] = True
    assert eec.iou(a, b) == pytest.approx(4 / 12)
    with pytest.raises(ValueError, match="shapes differ"):
        eec.iou(a, np.zeros((3, 4), dtype=bool))


def test_mask_png_round_trip(tmp_path: Path) -> None:
    mask = np.zeros((9, 13), dtype=bool)
    mask[2:5, 3:9] = True
    path = tmp_path / "nested" / "m.png"
    eec.write_mask_png(path, mask)
    with Image.open(path) as image:
        assert image.mode == "L" and image.size == (13, 9)
    assert np.array_equal(eec.read_mask_png(path), mask)


def test_hand_mask_lookup_uses_object_labels() -> None:
    observation = FrameObservations.model_validate(
        {
            "view_id": "ego-hmc21110305",
            "analysis_frame_index": 4,
            "source_seconds": 294.13,
            "objects": [
                {
                    "object_id": "h-left",
                    "label": "left_hand",
                    "confidence": 1.0,
                    "box": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2},
                    "mask": {
                        "uri": "masks/000004_left.png",
                        "storage": "external_artifact",
                        "format": "png",
                    },
                },
                {
                    "object_id": "c",
                    "label": "chassis",
                    "confidence": 1.0,
                    "box": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2},
                    "mask": {
                        "uri": "masks/000004_00.png",
                        "storage": "external_artifact",
                        "format": "png",
                    },
                },
            ],
        }
    )
    assert eec.hand_mask_uri_for(observation, "left_hand") == "masks/000004_left.png"
    assert eec.hand_mask_uri_for(observation, "right_hand") is None
    assert eec.hand_mask_uri_for(None, "left_hand") is None
    assert eec.mask_uri_for(observation, "chassis") == "masks/000004_00.png"
    assert eec.mask_uri_for(observation, "cabin") is None


# -- hull projection ---------------------------------------------------------------------------


def test_voxel_corners_and_rasterize_boxes() -> None:
    corners = eec.voxel_corners_mm(np.array([[0, 0, 0], [2, 0, 0]]), (10.0, 20.0, 30.0), 5.0)
    assert corners.shape == (2, 8, 3)
    assert corners[0].min(axis=0).tolist() == [10.0, 20.0, 30.0]
    assert corners[0].max(axis=0).tolist() == [15.0, 25.0, 35.0]
    assert corners[1, :, 0].min() == 20.0

    boxes = np.array(
        [
            [[1.2, 1.5]] * 4 + [[3.7, 2.9]] * 4,  # fills x 1..4, y 1..3
            [[np.nan, 0.0]] * 8,  # dropped
            [[-5.0, -5.0]] * 4 + [[0.5, 0.5]] * 4,  # clipped to the top-left pixel
            [[50.0, 50.0]] * 8,  # fully outside
        ]
    )
    mask = eec.rasterize_boxes(boxes, (8, 6))
    assert mask.shape == (6, 8)
    expected = np.zeros((6, 8), dtype=bool)
    expected[1:3, 1:4] = True
    expected[0, 0] = True
    assert np.array_equal(mask, expected)
    assert not eec.rasterize_boxes(np.zeros((0, 8, 2)), (8, 6)).any()


def _fixture_rig() -> mvg.CameraRig:
    target = np.array([0.0, 0.0, 0.0])
    cameras = {
        eec.EXO_CAMERA: _camera(
            eec.EXO_CAMERA, _look_at(np.array([600.0, -700.0, -900.0]), target, UP)
        ),
        eec.EGO_CAMERA: _camera(eec.EGO_CAMERA, None, ego=True),
    }
    ego_key = f"{eec.EGO_CAMERA[4:]}:mono10bit"
    ego_poses = {
        17640 + k: {ego_key: _look_at(np.array([50.0 + k, -500.0, -350.0]), target, UP)}
        for k in range(0, 400)
    }
    rules = {
        eec.EXO_CAMERA: _rule(f"{eec.EXO_CAMERA}:rgb", 9),
        eec.EGO_CAMERA: _rule(ego_key, 0),
    }
    return mvg.CameraRig(cameras, rules, ego_poses)


def test_project_hull_to_view_lands_where_the_rig_projects() -> None:
    rig = _fixture_rig()
    origin = (-100.0, -100.0, -100.0)
    indices = np.array([[20, 20, 20], [21, 20, 20], [20, 21, 20]])
    mask = eec.project_hull_to_view(
        rig,
        camera=eec.EGO_CAMERA,
        pose_frame=17648,
        voxel_indices=indices,
        grid_origin_mm=origin,
        voxel_size_mm=5.0,
        image_wh=EGO_WH,
    )
    assert mask.shape == (EGO_WH[1], EGO_WH[0])
    assert mask.any()
    centres = np.asarray(origin) + (indices + 0.5) * 5.0
    raw = rig.project(eec.EGO_CAMERA, centres, pose_frame=17648)
    scaled = raw * np.array([EGO_WH[0] / 636, EGO_WH[1] / 480])
    ys, xs = np.nonzero(mask)
    assert abs(xs.mean() - scaled[:, 0].mean()) < 6
    assert abs(ys.mean() - scaled[:, 1].mean()) < 6
    # Voxels behind the ego camera (it sits at y=-500 looking down) are dropped.
    behind = eec.project_hull_to_view(
        rig,
        camera=eec.EGO_CAMERA,
        pose_frame=17648,
        voxel_indices=np.array([[20, 20, 20]]),
        grid_origin_mm=(-100.0, -900.0, -100.0),
        voxel_size_mm=5.0,
        image_wh=EGO_WH,
    )
    assert not behind.any()


# -- prepare / evaluate / rerun on a fixture repository ------------------------------------------


def _box_mask(shape_hw: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape_hw, dtype=bool)
    y0, y1, x0, x1 = box
    mask[y0:y1, x0:x1] = True
    return mask


def _write_run(
    root: Path,
    run: Path,
    *,
    view_id: str,
    shape_hw: tuple[int, int],
    frames: dict[int, dict[str, tuple[int, int, int, int] | None]],
) -> None:
    run_dir = root / run
    (run_dir / "masks").mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({"fixture": run.name}), encoding="utf-8")
    with (run_dir / "observations.jsonl").open("w", encoding="utf-8") as handle:
        for frame, parts in frames.items():
            objects = []
            for label, box in parts.items():
                if box is None:
                    continue
                uri = f"masks/{frame:06d}_{label}.png"
                eec.write_mask_png(run_dir / uri, _box_mask(shape_hw, box))
                objects.append(
                    {
                        "object_id": f"fx-{label}",
                        "label": label,
                        "confidence": 1.0,
                        "box": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2},
                        "mask": {"uri": uri, "storage": "external_artifact", "format": "png"},
                    }
                )
            handle.write(
                json.dumps(
                    {
                        "view_id": view_id,
                        "analysis_frame_index": frame,
                        "source_seconds": 294.0 + frame / 30,
                        "objects": objects,
                    }
                )
                + "\n"
            )


def _fake_frame(video: Path, index: int) -> np.ndarray:
    width, height = EGO_WH if "ego" in video.name else EXO_WH
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = (index * 7) % 256
    rgb[..., 1] = 40 if "ego" in video.name else 200
    return rgb


@pytest.fixture
def fixture_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "configs/assembly101").mkdir(parents=True)
    shutil.copy(
        REPOSITORY_ROOT / "configs/assembly101/clock_rules.json",
        root / "configs/assembly101/clock_rules.json",
    )
    (root / "scripts").mkdir()
    for script in ("lm_eec_driver.py", "install_lm_eec.sh"):
        shutil.copy(REPOSITORY_ROOT / "scripts" / script, root / "scripts" / script)
    (root / "data").mkdir()
    (root / "data/exo.mp4").write_bytes(b"exo")
    (root / "data/ego.mp4").write_bytes(b"ego")
    exo_hw = (EXO_WH[1], EXO_WH[0])
    ego_hw = (EGO_WH[1], EGO_WH[0])
    _write_run(
        root,
        Path("runs/exo-fixture"),
        view_id="static-c10379",
        shape_hw=exo_hw,
        frames={
            0: {
                "chassis": (100, 200, 300, 500),
                "cabin": (400, 500, 600, 700),
                "rear_body": (10, 20, 10, 20),
                "interior": (50, 60, 50, 60),
            },
            150: {
                "chassis": (120, 220, 320, 520),
                "cabin": None,
                "rear_body": (10, 20, 10, 20),
                "interior": (50, 60, 50, 60),
            },
        },
    )
    _write_run(
        root,
        Path("runs/ego-fixture"),
        view_id="ego-hmc21110305",
        shape_hw=ego_hw,
        frames={
            4: {
                "chassis": (100, 200, 100, 300),
                "cabin": (300, 400, 500, 600),
                "rear_body": (10, 20, 10, 20),
                "interior": None,
            },
            154: {
                "chassis": (100, 200, 100, 300),
                "cabin": (300, 400, 500, 600),
                "rear_body": None,
                "interior": None,
            },
        },
    )
    hull = root / "runs/hull-fixture"
    hull.mkdir(parents=True)
    np.savez(
        hull / "hull_voxels_1fps.npz",
        **{
            "chassis/000000": np.array([[20, 20, 20], [21, 20, 20]], dtype=np.int16),
            "cabin/000000": np.zeros((0, 3), dtype=np.int16),
            "chassis/000150": np.array([[20, 20, 20]], dtype=np.int16),
        },
    )
    (hull / "manifest.json").write_text(
        json.dumps({"grid_origin_mm": [-100.0, -100.0, -100.0], "voxel_size_mm": 5.0}),
        encoding="utf-8",
    )
    (root / "ckpt").mkdir()
    (root / "ckpt/exo_ego.pt").write_bytes(b"exo->ego")
    (root / "ckpt/ego_exo.pt").write_bytes(b"ego->exo")
    return root


def _prepare_fixture(root: Path) -> eec.PairsManifest:
    return eec.prepare(
        root,
        run_directory=Path("runs/t7-fixture"),
        exo_query_run=Path("runs/exo-fixture"),
        ego_reference_run=Path("runs/ego-fixture"),
        hull_run=Path("runs/hull-fixture"),
        exo_proxy=Path("data/exo.mp4"),
        ego_proxy=Path("data/ego.mp4"),
        frames=FIXTURE_FRAMES,
        frame_reader=_fake_frame,
        rig=_fixture_rig(),
        queue_job_path=Path("runs/queue/jobs_t7.json"),
        checkpoints={
            "exo_to_ego": root / "ckpt/exo_ego.pt",
            "ego_to_exo": root / "ckpt/ego_exo.pt",
        },
    )


def test_prepare_writes_pairs_masks_and_queue_job(fixture_repo: Path) -> None:
    manifest = _prepare_fixture(fixture_repo)
    run_dir = fixture_repo / "runs/t7-fixture"
    reloaded = eec.load_pairs(run_dir)
    assert reloaded == manifest
    assert [pair.key for pair in manifest.keyframes] == ["000000", "000150"]
    for pair in manifest.keyframes:
        assert pair.ego_analysis_frame == pair.exo_analysis_frame + 4
        assert pair.residual_pose_frames == -1
        with Image.open(run_dir / pair.exo_frame_uri) as image:
            assert image.size == EXO_WH and image.format == "JPEG"
        with Image.open(run_dir / pair.ego_frame_uri) as image:
            assert image.size == EGO_WH
        assert Path(pair.exo_frame_uri).stem == Path(pair.ego_frame_uri).stem == pair.key
    first, second = manifest.keyframes
    assert first.exo_query_masks["chassis"] == "query_masks/static-c10379/chassis/000000.png"
    assert second.exo_query_masks["cabin"] is None
    assert first.ego_reference_masks["interior"] is None
    assert second.ego_reference_masks["rear_body"] is None
    assert first.hull_voxels_available == {
        "chassis": True,
        "interior": False,
        "rear_body": False,
        "cabin": False,
    }
    assert second.hull_voxels_available["chassis"] is True
    mask = eec.read_mask_png(run_dir / first.exo_query_masks["chassis"])
    assert mask.shape == (EXO_WH[1], EXO_WH[0]) and mask[150, 400] and not mask[0, 0]
    assert manifest.exo.proxy_size_wh == EXO_WH and manifest.ego.proxy_size_wh == EGO_WH
    assert manifest.exo.raw_size_wh == (1920, 1080) and manifest.ego.raw_size_wh == (636, 480)
    assert manifest.exo.pose_offset_frames == 9 and manifest.ego.pose_offset_frames == 0
    assert "rounded down" in manifest.frame_mapping_rule
    exo_to_ego, ego_to_exo = manifest.directions
    assert exo_to_ego.skipped_reason is None and exo_to_ego.objects == eec.PARTS
    assert exo_to_ego.checkpoint_sha256 is not None
    assert (
        ego_to_exo.skipped_reason is not None and "no ego hand masks" in ego_to_exo.skipped_reason
    )
    assert manifest.sources["hull_manifest"] is not None
    assert manifest.model.pinned_commit == eec.LM_EEC_PINNED_COMMIT
    assert manifest.model.mode == "sequence"
    assert len(manifest.claim_boundaries) >= 5

    spec = QueueSpec.model_validate_json(
        (fixture_repo / manifest.queue_job).read_text(encoding="utf-8")
    )
    job = spec.jobs[0]
    assert job.name == eec.QUEUE_JOB_NAME and job.timeout_s == 1800
    assert job.interpreter == [str(eec.LM_EEC_INTERPRETER)]
    assert job.cwd == str(eec.LM_EEC_ROOT)
    assert job.env["CUDA_VISIBLE_DEVICES"] == "0"
    assert job.argv[0].endswith("scripts/lm_eec_driver.py")
    assert "--device" in job.argv and job.argv[job.argv.index("--device") + 1] == "cuda"
    assert str(run_dir) in job.argv


def test_prepare_without_hull_records_no_availability(fixture_repo: Path) -> None:
    manifest = eec.prepare(
        fixture_repo,
        run_directory=Path("runs/t7-nohull"),
        exo_query_run=Path("runs/exo-fixture"),
        ego_reference_run=Path("runs/ego-fixture"),
        hull_run=None,
        exo_proxy=Path("data/exo.mp4"),
        ego_proxy=Path("data/ego.mp4"),
        frames=(0,),
        frame_reader=_fake_frame,
        rig=_fixture_rig(),
        queue_job_path=Path("runs/queue/jobs_nohull.json"),
        checkpoints={
            "exo_to_ego": fixture_repo / "ckpt/exo_ego.pt",
            "ego_to_exo": fixture_repo / "missing.pt",
        },
        hash_checkpoints=False,
    )
    assert manifest.sources["hull_manifest"] is None
    assert all(not any(pair.hull_voxels_available.values()) for pair in manifest.keyframes)
    assert manifest.directions[0].checkpoint_sha256 is None


def _write_predictions(run_dir: Path, pairs: eec.PairsManifest, *, cpu_smoke: bool = True) -> None:
    rows = []
    for pair in pairs.keyframes:
        for part in eec.PARTS:
            query = pair.exo_query_masks[part]
            if query is None:
                continue
            reference = pair.ego_reference_masks[part]
            if part == "chassis" and reference is not None:
                mask = eec.read_mask_png(run_dir / reference)  # perfect agreement
            elif part == "cabin":
                mask = np.zeros((EGO_WH[1], EGO_WH[0]), dtype=bool)  # empty prediction
            else:
                mask = _box_mask((EGO_WH[1], EGO_WH[0]), (600, 650, 600, 650))  # disjoint
            uri = f"predictions/exo_to_ego/{part}/{pair.key}.png"
            eec.write_mask_png(run_dir / uri, mask)
            rows.append(
                {
                    "direction": "exo_to_ego",
                    "object": part,
                    "key": pair.key,
                    "mask_uri": uri,
                    "mask_area_px": int(mask.sum()),
                    "empty": not bool(mask.any()),
                    "iou_prediction": 0.7,
                    "object_score_logit": 2.0,
                    "object_score": 0.88,
                    "max_logit": 4.0,
                    "seconds": 0.4,
                }
            )
    payload = {
        "manifest_kind": "egoexo_correspondence_predictions",
        "run_id": pairs.run_id,
        "predicted_at_utc": "2026-09-18T07:00:00+00:00",
        "pairs_sha256": "0" * 64,
        "checkpoints": {"exo_to_ego": "fixture"},
        "predictions": rows,
        "skipped": [f"ego_to_exo: {pairs.directions[1].skipped_reason}"],
        "runtime": {
            "device": "cpu",
            "device_name": None,
            "torch_version": "2.7.1+cu128",
            "cuda_version": "12.8",
            "mode": "sequence",
            "cpu_smoke": cpu_smoke,
            "lm_eec_head": eec.LM_EEC_PINNED_COMMIT,
            "model_load_seconds": {"exo_to_ego": 1.9},
            "time_to_first_output_seconds": 3.3,
            "total_seconds": 23.8,
            "gpu_peak_vram_reserved_bytes": None,
            "gpu_peak_vram_allocated_bytes": None,
        },
    }
    (run_dir / "predictions.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")


def test_evaluate_and_rerun_on_fixture(fixture_repo: Path) -> None:
    pairs = _prepare_fixture(fixture_repo)
    run_dir = fixture_repo / "runs/t7-fixture"
    _write_predictions(run_dir, pairs)
    (fixture_repo / "runs/queue").mkdir(exist_ok=True)
    queue_log = fixture_repo / "runs/queue/queue.log"
    queue_log.write_text(
        json.dumps(
            {
                "event": "job_end",
                "name": eec.QUEUE_JOB_NAME,
                "state": "succeeded",
                "exit_code": 0,
                "duration_s": 71.5,
                "stdout_log": "logs/x.log",
                "time": "2026-09-18T07:01:00+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = eec.evaluate(
        fixture_repo,
        run_directory=Path("runs/t7-fixture"),
        hull_run=Path("runs/hull-fixture"),
        rig=_fixture_rig(),
        queue_log=Path("runs/queue/queue.log"),
    )
    assert eec.load_evaluation(run_dir) == manifest
    by_key = {(c.object, c.key): c for c in manifest.comparisons}
    assert len(manifest.comparisons) == 7  # 8 query slots minus the missing cabin at 150
    chassis0 = by_key[("chassis", "000000")]
    assert chassis0.iou_vs_reference_mask == 1.0
    assert chassis0.hull_projection_uri is not None
    assert chassis0.iou_vs_hull_projection is not None
    assert chassis0.iou_reference_vs_hull is not None
    assert (run_dir / chassis0.hull_projection_uri).is_file()
    cabin0 = by_key[("cabin", "000000")]
    assert cabin0.prediction_empty and cabin0.iou_vs_reference_mask == 0.0
    assert cabin0.hull_projection_uri is None
    assert any("no voxels" in note for note in cabin0.notes)
    rear150 = by_key[("rear_body", "000150")]
    assert rear150.reference_uri is None and rear150.iou_vs_reference_mask is None
    interior0 = by_key[("interior", "000000")]
    assert interior0.iou_vs_reference_mask is None  # no ego interior reference at frame 4

    summaries = {s.object: s for s in manifest.summaries}
    assert summaries["chassis"].vs_reference_mask.median == 1.0
    assert summaries["chassis"].vs_reference_mask.at_or_above_0_5 == 2
    assert summaries["cabin"].predicted_nonempty == 0
    assert summaries["rear_body"].vs_reference_mask.count == 1
    assert summaries["rear_body"].vs_reference_mask.median == 0.0
    assert summaries["interior"].vs_reference_mask.count == 0

    m = manifest.measures
    assert (m.coverage_pairs_predicted_nonempty, m.coverage_pairs_total) == (6, 7)
    # frame 0: chassis, cabin, rear_body carry ego references; frame 150: chassis only
    assert m.coverage_pairs_with_reference_mask == 4
    assert m.coverage_pairs_with_hull_projection == 2
    assert m.runtime_seconds == 71.5 and m.driver_seconds == 23.8
    assert m.time_to_first_output_seconds == 3.3
    assert m.gpu_peak_vram_bytes is None and m.id_resets is None
    assert any("CPU SMOKE" in note for note in m.notes)
    assert manifest.queue_record is not None and manifest.queue_record.state == "succeeded"
    assert len(manifest.skipped) == 1  # the direction skip is not repeated
    assert manifest.hull_source is not None

    output = eec.build_recording(fixture_repo, run_directory=Path("runs/t7-fixture"))
    assert output.is_file() and output.stat().st_size > 1000


def test_evaluate_without_hull_marks_every_pair(fixture_repo: Path) -> None:
    pairs = _prepare_fixture(fixture_repo)
    run_dir = fixture_repo / "runs/t7-fixture"
    _write_predictions(run_dir, pairs, cpu_smoke=False)
    manifest = eec.evaluate(
        fixture_repo, run_directory=Path("runs/t7-fixture"), hull_run=None, rig=_fixture_rig()
    )
    assert manifest.hull_source is None
    assert all(c.iou_vs_hull_projection is None for c in manifest.comparisons)
    assert all(any("hull run not present" in n for n in c.notes) for c in manifest.comparisons)
    assert manifest.measures.runtime_seconds is None and manifest.queue_record is None
    assert not any("CPU SMOKE" in note for note in manifest.measures.notes)


# -- typing and argv -----------------------------------------------------------------------------


def test_predictions_file_typing_rejects_bad_scores() -> None:
    base = {
        "manifest_kind": "egoexo_correspondence_predictions",
        "run_id": "x",
        "predicted_at_utc": "t",
        "pairs_sha256": "a" * 64,
        "checkpoints": {},
        "predictions": [],
        "skipped": [],
        "runtime": {
            "device": "cuda",
            "device_name": "gpu",
            "torch_version": "2.7.1",
            "cuda_version": "12.8",
            "mode": "independent",
            "cpu_smoke": False,
            "lm_eec_head": None,
            "model_load_seconds": {},
            "time_to_first_output_seconds": None,
            "total_seconds": 1.0,
            "gpu_peak_vram_reserved_bytes": 2_000_000_000,
            "gpu_peak_vram_allocated_bytes": 1_500_000_000,
        },
    }
    assert eec.PredictionsFile.model_validate(base).runtime.mode == "independent"
    bad = dict(base)
    bad["predictions"] = [
        {
            "direction": "exo_to_ego",
            "object": "chassis",
            "key": "000000",
            "mask_uri": "p.png",
            "mask_area_px": 1,
            "empty": False,
            "iou_prediction": 0.5,
            "object_score_logit": 1.0,
            "object_score": 1.5,
            "max_logit": 1.0,
            "seconds": 0.1,
        }
    ]
    with pytest.raises(ValueError):
        eec.PredictionsFile.model_validate(bad)
    with pytest.raises(ValueError):
        eec.KeyframePair.model_validate(
            {
                "key": "12",
                "exo_analysis_frame": 0,
                "ego_analysis_frame": 4,
                "exo_pose_frame": 1,
                "ego_pose_frame": 1,
                "residual_pose_frames": 0,
                "exo_frame_uri": "a",
                "ego_frame_uri": "b",
                "exo_query_masks": {},
                "ego_reference_masks": {},
                "hull_voxels_available": {},
            }
        )


def test_driver_argv_and_queue_job_validate() -> None:
    argv = eec.driver_argv(
        repository_root=REPOSITORY_ROOT,
        run_directory=REPOSITORY_ROOT / "runs/x",
        mode="independent",
    )
    assert argv[0] == str(REPOSITORY_ROOT / eec.DRIVER_SCRIPT)
    assert argv[argv.index("--mode") + 1] == "independent"
    assert "--cpu-smoke" not in argv
    smoke = eec.driver_argv(
        repository_root=REPOSITORY_ROOT,
        run_directory=REPOSITORY_ROOT / "runs/x",
        mode="sequence",
        device="cpu",
        cpu_smoke=True,
    )
    assert smoke[-1] == "--cpu-smoke" and smoke[smoke.index("--device") + 1] == "cpu"
    spec = QueueSpec.model_validate(eec.queue_job(argv=argv))
    assert spec.jobs[0].command[0] == str(eec.LM_EEC_INTERPRETER)
    assert spec.jobs[0].timeout_s == eec.QUEUE_TIMEOUT_S == 1800


def test_summarize_handles_missing_values() -> None:
    empty = eec.summarize([None, None])
    assert empty.count == 0 and empty.median is None
    summary = eec.summarize([0.2, None, 0.8, 0.5])
    assert summary.count == 3 and summary.median == 0.5 and summary.at_or_above_0_5 == 2
    assert summary.min == 0.2 and summary.max == 0.8


# -- real data -----------------------------------------------------------------------------------


@pytest.mark.real_data
def test_prepared_run_directory_is_consistent() -> None:
    run_dir = require_artifact(REPOSITORY_ROOT / eec.RUN_ROOT)
    pairs = eec.load_pairs(run_dir)
    assert len(pairs.keyframes) == 12
    assert [p.exo_analysis_frame for p in pairs.keyframes] == list(eec.keyframes())
    for pair in pairs.keyframes:
        assert pair.ego_analysis_frame == pair.exo_analysis_frame + 4
        assert pair.residual_pose_frames == -1
        assert (run_dir / pair.exo_frame_uri).is_file() and (run_dir / pair.ego_frame_uri).is_file()
        for uri in (*pair.exo_query_masks.values(), *pair.ego_reference_masks.values()):
            if uri is not None:
                assert (run_dir / uri).is_file()
    query_count = sum(
        uri is not None for p in pairs.keyframes for uri in p.exo_query_masks.values()
    )
    assert query_count >= 40
    assert pairs.exo.proxy_size_wh == EXO_WH and pairs.ego.proxy_size_wh == EGO_WH
    assert pairs.directions[0].checkpoint_sha256 is not None
    assert pairs.directions[1].skipped_reason is not None
    spec = QueueSpec.model_validate_json(
        (REPOSITORY_ROOT / pairs.queue_job).read_text(encoding="utf-8")
    )
    assert spec.jobs[0].name == pairs.queue_job_name
    if (run_dir / "predictions.json").is_file():
        predictions = eec.load_predictions(run_dir)
        assert predictions.run_id == pairs.run_id


@pytest.mark.real_data
def test_lm_eec_checkout_is_pinned_and_checkpoint_present() -> None:
    root = require_artifact(eec.LM_EEC_ROOT)
    require_artifact(eec.LM_EEC_INTERPRETER)
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    assert head == eec.LM_EEC_PINNED_COMMIT
    checkpoint = require_artifact(eec.LM_EEC_CHECKPOINTS["exo_to_ego"])
    assert checkpoint.stat().st_size > 900_000_000
