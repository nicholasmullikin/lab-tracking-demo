"""Exemplar seeding (recording 2): seed-frame rule, grid, consistency, gate; the multiview
profile's agent-correction schedule and non-default recording range in muggled_smoke."""

from __future__ import annotations

import hashlib
import json
import types
from pathlib import Path

import cv2
import numpy as np
import pytest
from test_multiview_geometry import UP, _camera, _look_at, _rule

from battle import exemplar_seed as es
from battle import muggled_smoke as ms
from battle import multiview_geometry as mvg
from battle.multiview_geometry import TablePlane
from battle.schemas import ArtifactFingerprint, MultiviewAgentCorrectionSchedule

TARGET_VIEW = "C10379"
SHAPE = (720, 1280)


@pytest.fixture
def rig() -> mvg.CameraRig:
    target = np.array([0.0, 0.0, 0.0])
    poses = {
        TARGET_VIEW: _look_at(np.array([600.0, -700.0, -900.0]), target, UP),
        "C10001": _look_at(np.array([-800.0, -650.0, -700.0]), target, UP),
        "C10002": _look_at(np.array([100.0, -900.0, 800.0]), target, UP),
        "C10003": _look_at(np.array([-300.0, -800.0, 700.0]), target, UP),
    }
    cameras = {view: _camera(view, pose) for view, pose in poses.items()}
    return mvg.CameraRig(cameras, {view: _rule(f"{view}:rgb", 5) for view in poses}, {})


def _plane() -> TablePlane:
    # Table at y = 0 with "up" = -y (the synthetic cameras sit at negative y).
    return TablePlane(
        normal=(0.0, -1.0, 0.0),
        offset_mm=0.0,
        point_on_plane_mm=(0.0, 0.0, 0.0),
        down_vector=(0.0, 1.0, 0.0),
        down_vector_source="fixture",
        down_vector_alternatives_deg={},
        normal_vs_down_deg=0.0,
        fingertip_count=10,
        fitted_point_count=3,
        lowest_fraction=0.05,
        residual_rms_mm=0.0,
        residual_p95_mm=0.0,
        fingertips_below_plane_by_20mm_fraction=0.0,
        claim_boundary="fixture",
    )


class _Members:
    """Just enough of PoseMembers for the seed-frame rule: two hands, one joint each."""

    def __init__(self, heights_by_pose: dict[int, float | None]) -> None:
        self.landmarks3d = {
            str(pose): {"0": [[0.0, -h, 0.0]], "1": [[0.0, -h, 0.0]]}
            for pose, h in heights_by_pose.items()
            if h is not None
        }
        self.confidences = {str(pose): {"0": 1.0, "1": 1.0} for pose in self.landmarks3d}


def test_seed_frame_is_where_the_hands_stay_highest_over_the_window(rig: mvg.CameraRig) -> None:
    rule = rig.clock_rule(TARGET_VIEW)
    heights = {rule.pose_frame(f): 20.0 for f in range(300, 320)}
    heights.update({rule.pose_frame(f): 90.0 for f in range(305, 308)})  # 3-frame spike
    heights.update({rule.pose_frame(f): 70.0 for f in range(312, 319)})  # sustained
    choice = es.choose_seed_frame(
        rig, _Members(heights), _plane(), reference_view=TARGET_VIEW, search=(300, 320), window=5
    )
    assert choice.seed_frame == 312 and choice.min_joint_height_mm == 70.0
    assert "table plane" in choice.rule


def _disc(centre: np.ndarray, radius: float) -> np.ndarray:
    grid_y, grid_x = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]]
    return (grid_x + 0.5 - centre[0]) ** 2 + (grid_y + 0.5 - centre[1]) ** 2 <= radius**2


def _candidate(
    rig: mvg.CameraRig,
    view: str,
    world: np.ndarray,
    radius_mm: float,
    sim: dict[str, float],
    tmp: Path,
) -> es.Candidate:
    from battle.multiview_seed_transfer import proxy_focal_px, proxy_to_raw_scale

    depth = float(rig.depth(view, world.reshape(1, 3))[0])
    centre = rig.project(view, world.reshape(1, 3))[0] / proxy_to_raw_scale(view)
    radius_px = radius_mm * proxy_focal_px(rig, view) / depth
    mask = _disc(centre, radius_px)
    path = tmp / f"{view}_{abs(hash(tuple(world.round(1)))) % 10**6}_{radius_mm:.0f}.png"
    cv2.imwrite(str(path), mask.astype(np.uint8) * 255)
    return es.Candidate(
        view=view,
        prompt_id=f"t000383-b{abs(hash(path.name)) % 90 + 1:02d}",
        candidate_index=0,
        mask_path=path,
        mask=mask,
        decoder_iou=0.8,
        area_px=int(mask.sum()),
        centroid_proxy=centre,
        centroid_raw=centre * proxy_to_raw_scale(view),
        depth_mm=depth,
        radius_mm=radius_mm,
        similarity=sim,
    )


def test_consistency_needs_three_views_and_rejects_off_size_and_neighbours(
    rig: mvg.CameraRig, tmp_path: Path
) -> None:
    views = list(rig.views)
    truth = np.array([40.0, -30.0, -20.0])
    decoy = np.array([-160.0, -30.0, 90.0])
    per_view = {}
    for view in views:
        per_view[view] = [
            _candidate(rig, view, truth, 45.0, {"cabin": 0.9, "chassis": 0.3}, tmp_path),
            _candidate(rig, view, decoy, 45.0, {"cabin": 0.6, "chassis": 0.8}, tmp_path),
        ]
    # One view's cabin candidate is far too large: it must not count as a supporter.
    per_view[views[3]][0].radius_mm = 200.0
    result = es.find_consistent_part(rig, _plane(), "cabin", per_view)
    assert result.reached and len(result.views_used) >= 3
    assert set(views[:3]) <= set(result.views_used)
    assert per_view[views[3]][0].key not in result.chosen.values()
    assert result.world_point_mm is not None
    assert np.linalg.norm(np.asarray(result.world_point_mm) - truth) < 5.0
    assert all(e <= es.CONSISTENCY_MAX_RAW_PX for e in result.reprojection_px.values())
    assert all(m > 0 for m in result.margin.values())

    # The chassis must land elsewhere (neighbourhood exclusion), on the decoy.
    used = set(result.chosen.values())
    chassis = es.find_consistent_part(
        rig,
        _plane(),
        "chassis",
        per_view,
        excluded_points=[np.asarray(result.world_point_mm)],
        used_keys=used,
    )
    assert chassis.reached
    assert np.linalg.norm(np.asarray(chassis.world_point_mm) - decoy) < 5.0

    # Two views only: blocked, not guessed.
    two = {v: per_view[v] for v in views[:2]}
    blocked = es.find_consistent_part(rig, _plane(), "cabin", two)
    assert not blocked.reached and "views" in blocked.reason


def test_the_gate_uses_only_the_parts_that_passed_on_recording_1() -> None:
    assert es.PASSING_PARTS == ("rear_body", "cabin")
    assert all(es.REC1_HELD_OUT_IOU[p] < es.SEED_GATE_IOU for p in ("chassis", "interior"))


def test_table_grid_covers_the_hand_region_on_the_plane() -> None:
    class Recording:
        window_start_raw_frame = 1000
        window_raw_frame_count = 100

    heights = {1000 + i: 30.0 for i in range(0, 100, 10)}
    members = _Members(heights)
    # Spread the joints out on the plane so the rectangle is non-degenerate.
    for i, key in enumerate(sorted(members.landmarks3d)):
        members.landmarks3d[key] = {"0": [[20.0 * i, -30.0, 0.0]], "1": [[0.0, -30.0, 15.0 * i]]}
    grid, source = es.table_grid(
        _plane(), members, Recording(), spacing_mm=50.0, margin_mm=100.0, confidence_floor=0.5
    )
    assert grid.shape[1] == 3 and grid.shape[0] > 20
    assert np.allclose(_plane().signed_distance(grid), 0.0, atol=1e-6)
    assert "hand joints" in source


def test_anchor_frame_selection_spaces_ranked_frames_and_seeds_the_random_draw() -> None:
    suspicion = {f: 0.0 for f in range(300, 2100)}
    for f in (500, 510, 520, 900, 1500, 1501):
        suspicion[f] = 0.9
    suspicion[1200] = 0.5
    ranked, randoms = es.select_anchor_frames(
        suspicion, frame_range=(383, 2100), top=3, random_count=2, min_spacing=60, seed=1
    )
    assert ranked == [500, 900, 1500]  # 510/520/1501 are within 60 frames of a chosen frame
    assert len(randoms) == 2 and all(383 <= f < 2100 for f in randoms)
    assert all(abs(f - r) >= 20 for f in randoms for r in ranked)
    again = es.select_anchor_frames(
        suspicion, frame_range=(383, 2100), top=3, random_count=2, min_spacing=60, seed=1
    )
    assert again == (ranked, randoms)


# -- muggled_smoke: multiview profile on another recording, agent corrections ----------------


def test_multiview_range_permits_c10379_and_max_frames_on_another_recording() -> None:
    with pytest.raises(ValueError):
        ms.require_four_part_multiview_range("static-c10379", 0, 1800)
    frame_range = ms.require_four_part_multiview_range(
        "static-c10379", 0, 1717, recording="nusar_9061", proxy_frame_count=1717
    )
    assert (frame_range.start_frame, frame_range.end_frame_exclusive) == (0, 1717)
    with pytest.raises(ValueError):
        ms.require_four_part_multiview_range(
            "static-c10379", 0, 1800, recording="nusar_9061", proxy_frame_count=1717
        )
    with pytest.raises(ValueError):
        ms.require_four_part_multiview_range("ego-hmc21179183", 0, 100, recording="nusar_9061")
    args = types.SimpleNamespace(
        four_part_multiview_first_minute=True, recording="nusar_9061", max_frames=1717
    )
    assert ms.selected_frame_budget(args, 30.0) == 1717
    args.recording = "nusar_9033"
    assert ms.selected_frame_budget(args, 30.0) == ms.FOUR_PART_MULTIVIEW_FRAMES
    assert ms.is_default_recording(None) and ms.is_default_recording("nusar_9033")
    assert not ms.is_default_recording("nusar_9061")


def _fp(path: Path, root: Path) -> ArtifactFingerprint:
    return ArtifactFingerprint(
        uri=str(path.relative_to(root)),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source="measured",
    )


def test_agent_correction_schedule_rides_in_the_worker_payload(tmp_path: Path) -> None:
    root = tmp_path
    config = root / "clip.json"
    config.write_text("{}")
    seeds_path = root / "seed_manifest.json"
    seeds_path.write_text("{}")
    mask = root / "masks" / "t000500-b01_candidate-00.png"
    mask.parent.mkdir()
    cv2.imwrite(str(mask), np.zeros((4, 4), dtype=np.uint8) + 255)
    proxy_sha = "ab" * 32
    proxy = types.SimpleNamespace(
        view_id="static-c10379", proxy_uri="proxy.mp4", checksum_sha256=proxy_sha
    )
    schedule = MultiviewAgentCorrectionSchedule(
        manifest_kind="multiview_agent_correction_schedule",
        view_id="static-c10379",
        clip_config=_fp(config, root),
        proxy=ArtifactFingerprint(uri="proxy.mp4", sha256=proxy_sha, source="measured"),
        seed_manifest=_fp(seeds_path, root),
        plan=_fp(config, root),
        corrections=(
            {
                "target": "cabin",
                "multiplex_slot": 1,
                "analysis_frame_index": 500,
                "candidate_id": "t000500-b01",
                "candidate_index": 0,
                "mask": _fp(mask, root),
                "iteration": 1,
            },
        ),
        claim_boundaries=("fixture",),
    )
    schedule_path = root / "agent.json"
    schedule_path.write_text(schedule.model_dump_json())
    seed_payload = {
        "seeds": [
            {"target": "rear_body", "multiplex_slot": 0, "frame_index": 0, "selected_by": "agent"},
            {"target": "cabin", "multiplex_slot": 1, "frame_index": 0, "selected_by": "agent"},
        ]
    }
    payload, audit = ms._load_agent_correction_schedule(
        schedule_path=schedule_path,
        seed_manifest_path=seeds_path,
        seed_payload=seed_payload,
        repository_root=root,
        config_path=config,
        proxy=proxy,
        max_frame_exclusive=1717,
        correction_memory_semantics="append_prompt_memory_and_reset_frame_memory",
    )
    assert payload["seeds"] == seed_payload["seeds"]
    assert payload["memory_semantics"] == "append_prompt_memory_and_reset_frame_memory"
    (correction,) = payload["corrections"]
    assert (correction["target"], correction["multiplex_slot"], correction["frame_index"]) == (
        "cabin",
        1,
        500,
    )
    assert correction["selected_by"] == "agent" and correction["object_id"] == "sam3-01"
    assert audit["correction_frames"] == (500,)
    # A correction on a slot the seeds did not fill, or past the budget, is refused.
    bad = schedule.model_copy(
        update={"corrections": (schedule.corrections[0].model_copy(update={"target": "chassis"}),)}
    )
    schedule_path.write_text(bad.model_dump_json())
    with pytest.raises(ValueError, match="did not seed"):
        ms._load_agent_correction_schedule(
            schedule_path=schedule_path,
            seed_manifest_path=seeds_path,
            seed_payload=seed_payload,
            repository_root=root,
            config_path=config,
            proxy=proxy,
            max_frame_exclusive=1717,
            correction_memory_semantics="append_prompt_memory_and_reset_frame_memory",
        )
    schedule_path.write_text(schedule.model_dump_json())
    with pytest.raises(ValueError, match="budget"):
        ms._load_agent_correction_schedule(
            schedule_path=schedule_path,
            seed_manifest_path=seeds_path,
            seed_payload=seed_payload,
            repository_root=root,
            config_path=config,
            proxy=proxy,
            max_frame_exclusive=400,
            correction_memory_semantics="append_prompt_memory_and_reset_frame_memory",
        )
    # Seed manifest drift is refused too.
    seeds_path.write_text(json.dumps({"changed": True}))
    with pytest.raises(ValueError, match="different geometric seed manifest"):
        ms._load_agent_correction_schedule(
            schedule_path=schedule_path,
            seed_manifest_path=seeds_path,
            seed_payload=seed_payload,
            repository_root=root,
            config_path=config,
            proxy=proxy,
            max_frame_exclusive=1717,
            correction_memory_semantics="append_prompt_memory_and_reset_frame_memory",
        )
