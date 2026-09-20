"""Multiview re-prompt loop: onsets, prompt geometry, acceptance, schedules, commands, cap."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from test_multiview_geometry import UP, _camera, _look_at, _rule

from battle import multiview_geometry as mvg
from battle import multiview_reprompt as mr
from battle.muggled_calibration import build_manifest, finalize_correction_schedule
from battle.muggled_smoke import _load_multi_keyframe_correction_schedule
from battle.multiview_seed_transfer import project_to_proxy, proxy_focal_px
from battle.schemas import (
    ArtifactFingerprint,
    G2PreprocessingManifest,
    MuggledSAMCalibrationCandidate,
    MuggledSAMMultiKeyframeCorrectionSchedule,
    MultiviewAgentCorrectionSchedule,
    MultiviewConsensusProvenanceFile,
    MultiviewContradictionOnset,
    MultiviewRepromptPlan,
    RepromptCandidateScore,
)

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_g2.json"
TARGET_CONFIG = ROOT / "configs/muggledsam_static_four_part_reassembly_focused_manual_seed.json"
POLICY_V4 = (
    ROOT / "configs/muggledsam_static_four_part_reassembly_focused_correction_policy_v4.json"
)
TARGETS = ("chassis", "interior", "rear_body", "cabin")
SHAPE = (720, 1280)
TARGET_VIEW = "C10379"  # the synthetic rig names its target camera after the real one


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
    rules = {view: _rule(f"{view}:rgb", 5) for view in poses}
    return mvg.CameraRig(cameras, rules, {})


def _fingerprint(path: Path) -> ArtifactFingerprint:
    return ArtifactFingerprint(
        uri=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), source="measured"
    )


def _disc(centre: np.ndarray, radius: float, shape: tuple[int, int] = SHAPE) -> np.ndarray:
    grid_y, grid_x = np.mgrid[0 : shape[0], 0 : shape[1]]
    return (grid_x + 0.5 - centre[0]) ** 2 + (grid_y + 0.5 - centre[1]) ** 2 <= radius**2


# -- onset detection -----------------------------------------------------------------------------


def _consensus_fixture(frames: int = 400) -> dict[str, Any]:
    error = np.zeros(frames)
    error[100:120] = 60.0  # 20-frame contradiction
    error[125:140] = 50.0  # 5-frame gap, merged under the 15-frame rule
    error[200:206] = 90.0  # 6 frames: an episode, but a merged run shorter than 10 is dropped
    error[300:330] = 70.0  # a run where only two static views agree
    exists = np.ones(frames, dtype=bool)
    observed = np.ones(frames, dtype=bool)
    used = error <= 40.0
    used_counts = np.where(used, 4, 3)
    static_used = np.where(used, 3, 3).astype(np.int64)
    static_used[300:330] = 2
    return {
        "errors": {"chassis": {TARGET_VIEW: error}},
        "consensus_exists": {"chassis": exists},
        "dropped": {"chassis": {TARGET_VIEW: observed & ~used}},
        "used_counts": {"chassis": used_counts},
        "static_used_counts": {"chassis": static_used},
    }


def test_onsets_merge_gaps_drop_short_runs_and_gate_on_static_majority() -> None:
    seeds = mr.detect_onsets(
        target_view=TARGET_VIEW, config=mr.DEFAULT_DETECTOR, **_consensus_fixture()
    )
    by_frame = {seed.onset_frame: seed for seed in seeds}
    assert sorted(by_frame) == [100, 300]
    merged = by_frame[100]
    assert merged.end_frame_exclusive == 140
    assert merged.source_episodes == ((100, 120), (125, 140))
    assert merged.target_error_px == 60.0 and merged.max_target_error_px == 60.0
    assert merged.gate_reason is None and merged.static_views_used == 3
    gated = by_frame[300]
    assert gated.gate_reason is not None and "2 static views" in gated.gate_reason


def test_a_detector_file_overrides_only_the_keys_it_carries(tmp_path: Path) -> None:
    path = tmp_path / "detector.json"
    path.write_text(json.dumps({"threshold_px": 55.5, "min_agreeing_static_views": 2, "other": 1}))
    config = mr.load_detector_config(path, tmp_path)
    assert (config.threshold_px, config.min_agreeing_static_views) == (55.5, 2)
    assert config.merge_gap_frames == mr.DEFAULT_DETECTOR.merge_gap_frames
    assert config.source == "file" and config.source_fingerprint is not None
    assert config.source_fingerprint.uri == "detector.json"
    seeds = mr.detect_onsets(target_view=TARGET_VIEW, config=config, **_consensus_fixture())
    # With a 55.5 px threshold the 50 px run no longer joins; the 60 px run alone is 20 frames,
    # and a two-static-view majority now passes the gate at 300.
    assert [(s.onset_frame, s.end_frame_exclusive) for s in seeds if s.gate_reason is None] == [
        (100, 120),
        (300, 330),
    ]
    path.write_text(json.dumps({"unrelated": 1}))
    with pytest.raises(ValueError, match="none of the keys"):
        mr.load_detector_config(path, tmp_path)


def test_merge_with_gaps_is_strict_on_the_gap() -> None:
    assert mr.merge_with_gaps([(0, 10), (20, 30)], 10) == ((0, 10), (20, 30))
    assert mr.merge_with_gaps([(0, 10), (20, 30)], 11) == ((0, 30),)
    assert mr.merge_with_gaps([(20, 30), (0, 10), (5, 12)], 0) == ((0, 12), (20, 30))


# -- prompt geometry -----------------------------------------------------------------------------


def test_sphere_projects_to_boxes_with_negatives_at_the_other_parts(rig: mvg.CameraRig) -> None:
    centre = np.array([40.0, -30.0, -20.0])
    radius_mm = 40.0
    masks: dict[str, np.ndarray] = {}
    for view in ("C10001", "C10002", "C10003"):
        centroid = project_to_proxy(rig, view, centre)[0]
        depth = float(rig.depth(view, centre.reshape(1, 3))[0])
        masks[view] = _disc(centroid, radius_mm * proxy_focal_px(rig, view) / depth)
    geometry = mr.onset_geometry(
        rig,
        target_view=TARGET_VIEW,
        point_world_mm=centre,
        masks_by_view=masks,
        pose_frames=dict.fromkeys(masks),
    )
    assert geometry is not None
    assert abs(geometry.radius_mm - radius_mm) < 0.1 * radius_mm
    half = radius_mm * proxy_focal_px(rig, TARGET_VIEW) / geometry.depth_mm
    assert abs(geometry.expected_area_px - np.pi * half**2) < 0.15 * np.pi * half**2
    expected_centroid = project_to_proxy(rig, TARGET_VIEW, centre)[0]
    assert np.allclose(geometry.centroid_proxy_px, expected_centroid)

    near = project_to_proxy(rig, TARGET_VIEW, centre + np.array([radius_mm * 1.2, 0.0, 0.0]))[0]
    far = np.array([5.0, 5.0])
    prompts = mr.build_prompts(
        target="chassis",
        frame=296,
        centroid_proxy_px=expected_centroid,
        half_size_px=half,
        shape=SHAPE,
        other_centroids={"rear_body": near, "cabin": far, "interior": np.array([np.nan, np.nan])},
    )
    assert [p.prompt_id for p in prompts] == ["t000296-b01", "t000296-b02"]
    assert [p.variant for p in prompts] == [
        "consensus_sphere_box_margin_0.25",
        "consensus_sphere_box_margin_0.60",
    ]
    for prompt in prompts:
        box = prompt.pixel_box
        assert box.x1 <= expected_centroid[0] <= box.x2 and box.y1 <= expected_centroid[1] <= box.y2
        assert prompt.background_point_sources == ("rear_body_consensus_centroid",)
        (point,) = prompt.background_points
        assert abs(point.x - near[0]) <= 1 and abs(point.y - near[1]) <= 1
    wide, narrow = prompts[1].pixel_box, prompts[0].pixel_box
    assert wide.x2 - wide.x1 > narrow.x2 - narrow.x1

    with_hands = mr.build_prompts(
        target="chassis",
        frame=296,
        centroid_proxy_px=expected_centroid,
        half_size_px=half,
        shape=SHAPE,
        other_centroids={},
        hand_joints={"left_hand": np.stack([expected_centroid + 3.0, far])},
    )
    assert with_hands[0].background_point_sources == ("left_hand_joint_00",)


def test_geometry_is_none_without_a_contributing_view(rig: mvg.CameraRig) -> None:
    assert (
        mr.onset_geometry(
            rig,
            target_view=TARGET_VIEW,
            point_world_mm=np.zeros(3),
            masks_by_view={"C10001": np.zeros(SHAPE, dtype=bool)},
            pose_frames={"C10001": None},
        )
        is None
    )


# -- acceptance ---------------------------------------------------------------------------------


def _onset(
    rig: mvg.CameraRig, target: str, frame: int, centre: np.ndarray, radius_mm: float
) -> tuple[MultiviewContradictionOnset, np.ndarray, float]:
    centroid = project_to_proxy(rig, TARGET_VIEW, centre)[0]
    depth = float(rig.depth(TARGET_VIEW, centre.reshape(1, 3))[0])
    half = radius_mm * proxy_focal_px(rig, TARGET_VIEW) / depth
    prompts = mr.build_prompts(
        target=target,
        frame=frame,
        centroid_proxy_px=centroid,
        half_size_px=half,
        shape=SHAPE,
        other_centroids={},
    )
    onset = MultiviewContradictionOnset(
        target=target,
        onset_frame=frame,
        interval_end_frame_exclusive=frame + 20,
        source_episodes=((frame, frame + 20),),
        target_error_px=60.0,
        max_target_error_px=80.0,
        consensus_world_mm=tuple(float(v) for v in centre),
        views_used=("C10001", "C10002", "C10003"),
        static_views_used=3,
        radius_mm=radius_mm,
        depth_mm=depth,
        projected_centroid_proxy_px=(float(centroid[0]), float(centroid[1])),
        expected_area_px=float(np.pi * half**2),
        prompts=prompts,
        status="planned",
    )
    return onset, centroid, half


def _score(prompt_id: str, index: int, iou: float, path: Path) -> RepromptCandidateScore:
    return RepromptCandidateScore(
        prompt_id=prompt_id,
        candidate_index=index,
        mask=_fingerprint(path),
        decoder_iou_estimate=iou,
        mask_area_px=0,
        passed=False,
    )


def test_acceptance_rule_uses_area_band_and_centroid_ray(
    rig: mvg.CameraRig, tmp_path: Path
) -> None:
    centre = np.array([40.0, -30.0, -20.0])
    onset, centroid, half = _onset(rig, "chassis", 500, centre, 40.0)
    stub = tmp_path / "m.png"
    stub.write_bytes(b"x")
    good = _disc(centroid, half)
    small = _disc(centroid, half / 4)
    far = _disc(centroid + np.array([6 * half, 0.0]), half)
    decision = mr.score_candidates(
        rig,
        target_view=TARGET_VIEW,
        onset=onset,
        candidates=[
            (_score("t000500-b01", 0, 0.7, stub), good),
            (_score("t000500-b01", 1, 0.9, stub), small),
            (_score("t000500-b02", 0, 0.95, stub), far),
        ],
    )
    assert decision.decision == "accepted" and decision.accepted is not None
    assert (decision.accepted.prompt_id, decision.accepted.candidate_index) == ("t000500-b01", 0)
    assert decision.accepted.centroid_ray_distance_radii is not None
    assert decision.accepted.centroid_ray_distance_radii < 0.5
    assert 0.8 < (decision.accepted.area_ratio_vs_expected or 0) < 1.2
    by_index = {(c.prompt_id, c.candidate_index): c for c in decision.candidates}
    assert not by_index[("t000500-b01", 1)].passed
    assert any("area ratio" in n for n in by_index[("t000500-b01", 1)].notes)
    assert not by_index[("t000500-b02", 0)].passed
    assert any("centroid ray" in n for n in by_index[("t000500-b02", 0)].notes)

    rejected = mr.score_candidates(
        rig,
        target_view=TARGET_VIEW,
        onset=onset,
        candidates=[
            (_score("t000500-b01", 0, 0.9, stub), small),
            (_score("t000500-b01", 1, 0.9, stub), np.zeros(SHAPE, dtype=bool)),
        ],
    )
    assert rejected.decision == "rejected" and rejected.accepted is None
    assert "no candidate passed" in rejected.reason


# -- decode with a stub decoder, schedule derivation, fingerprint rebinding --------------------


class _StubDecoder:
    """Writes real PNG masks the way the isolated worker does, from a per-prompt mask plan."""

    def __init__(self, results_directory: Path, masks_for: Any) -> None:
        self.results_directory = results_directory
        self.masks_for = masks_for
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.closed = False

    def request(
        self, command: str, payload: dict[str, Any], timeout: float = 120
    ) -> dict[str, Any]:
        self.requests.append((command, payload))
        if command == "frame_preview":
            return {"frame_index": payload["frame_index"]}
        decoded = []
        for prompt in payload["prompts"]:
            candidate_id = prompt["candidate_id"]
            (self.results_directory / "masks").mkdir(parents=True, exist_ok=True)
            candidates = []
            masks = self.masks_for(prompt)
            for index, (mask, iou) in enumerate(masks):
                path = (
                    self.results_directory / "masks" / f"{candidate_id}_candidate-{index:02d}.png"
                )
                assert cv2.imwrite(str(path), mask.astype(np.uint8) * 255)
                candidates.append(
                    {
                        "candidate_index": index,
                        "iou_score": iou,
                        "mask_uri": f"results/masks/{path.name}",
                        "is_deterministic_best": index == len(masks) - 1,
                    }
                )
            decoded.append(
                {
                    "box_id": prompt["box_id"],
                    "candidate_id": candidate_id,
                    "decoder_result": {
                        "api": "muggledsam_sam3_interactive",
                        "candidate_count": len(candidates),
                        "deterministic_best_candidate_index": len(candidates) - 1,
                        "candidates": candidates,
                        "overlay_uri": f"results/{candidate_id}_overlay.png",
                    },
                }
            )
        return {"decoded": decoded}

    def close(self) -> None:
        self.closed = True


def _candidate_payload(candidate_id: str, target: str, frame_index: int) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "intended_target": target,
        "frame": {
            "analysis_frame_index": frame_index,
            "proxy_seconds": frame_index / 30,
            "analysis_seconds": frame_index / 30,
            "source_seconds": 294.0 + frame_index / 30,
        },
        "pixel_box": {"x1": 10, "y1": 10, "x2": 50, "y2": 50},
        "normalized_box": {"x": 10 / 1280, "y": 10 / 720, "width": 40 / 1280, "height": 40 / 720},
        "decoder_result": {
            "api": "muggledsam_sam3_interactive",
            "candidate_count": 1,
            "deterministic_best_candidate_index": 0,
            "candidates": [
                {
                    "candidate_index": 0,
                    "iou_score": 0.9,
                    "mask_uri": f"results/masks/{candidate_id}_candidate-00.png",
                    "is_deterministic_best": True,
                }
            ],
            "overlay_uri": f"results/{candidate_id}_overlay.png",
        },
        "human_selected_candidate_index": 0,
        "human_accepted": True,
        "selected_for_finalization": frame_index == 0,
        "selected_for_correction": frame_index != 0,
    }


def _source_calibration(tmp_path: Path) -> tuple[Path, Path]:
    """A finalized human calibration: four frame-0 seeds, chassis corrections at 900 and 2700."""
    source = tmp_path / "runs" / "source-calibration"
    (source / "results" / "masks").mkdir(parents=True)
    candidates = [_candidate_payload(f"t000000-b{i + 1:02d}", t, 0) for i, t in enumerate(TARGETS)]
    candidates.append(_candidate_payload("t000900-b01", "chassis", 900))
    candidates.append(_candidate_payload("t002700-b01", "chassis", 2700))
    for payload in candidates:
        mask = source / payload["decoder_result"]["candidates"][0]["mask_uri"]
        mask.write_bytes(payload["candidate_id"].encode())
    schedule_path = source / "multi_keyframe_correction_schedule.json"
    manifest = build_manifest(
        repository_root=ROOT,
        config_path=CONFIG,
        timestamps=(0.0, 30.0, 90.0),
        result_directory=source / "results",
        calibration_id="source-calibration",
        view_id="static-c10379",
    ).model_copy(
        update={
            "candidates": tuple(
                MuggledSAMCalibrationCandidate.model_validate(item) for item in candidates
            ),
            "final_correction_schedule_uri": str(schedule_path),
            "plan_revision": 1,
        }
    )
    manifest_path = source / "calibration_manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n")
    finalize_correction_schedule(
        manifest_path=manifest_path,
        candidate_ids=tuple(c["candidate_id"] for c in candidates),
        schedule_path=schedule_path,
        correction_policy_path=POLICY_V4,
        manual_seed_target_config_path=TARGET_CONFIG,
        repository_root=ROOT,
    )
    return manifest_path, schedule_path


def _plan(
    rig: mvg.CameraRig,
    tmp_path: Path,
    *,
    onsets: tuple[MultiviewContradictionOnset, ...],
    schedule_capable: bool = True,
    checkpoint_frames: tuple[int, ...] = (),
    iteration: int = 1,
) -> tuple[Path, dict[str, Any]]:
    dummy = tmp_path / "consensus" / "manifest.json"
    dummy.parent.mkdir(parents=True, exist_ok=True)
    dummy.write_text("{}")
    points = dummy.with_name("consensus_points.npz")
    points.write_bytes(b"npz")
    extras: dict[str, Any] = {}
    if schedule_capable:
        manifest_path, schedule_path = _source_calibration(tmp_path)
        extras = {
            "source_run_profile": "four_part_static_focused",
            "source_schedule": _fingerprint(schedule_path),
            "source_calibration_manifest": _fingerprint(manifest_path),
            "source_correction_frames": (900,),
        }
    else:
        seed_manifest = tmp_path / "seeds" / "seed_manifest.json"
        seed_manifest.parent.mkdir(parents=True, exist_ok=True)
        # Only the accepted-part order matters to the re-prompt loop (slot assignment); the
        # tracker validates the full manifest itself.
        seed_manifest.write_text(
            json.dumps(
                {
                    "parts": [
                        {"target": "chassis", "status": "accepted", "accepted": {"x": 1}},
                        {"target": "interior", "status": "blocked", "accepted": None},
                        {"target": "rear_body", "status": "accepted", "accepted": {"x": 1}},
                        {"target": "cabin", "status": "accepted", "accepted": {"x": 1}},
                    ]
                }
            )
        )
        extras = {
            "source_run_profile": "four_part_multiview",
            "source_geometric_seed_manifest": _fingerprint(seed_manifest),
        }
    plan = MultiviewRepromptPlan(
        manifest_kind="multiview_reprompt_plan",
        target_view=TARGET_VIEW,
        target_view_id="static-c10379",
        iteration=iteration,
        consensus_root_uri=str(dummy.parent),
        consensus_manifest=_fingerprint(dummy),
        consensus_points=_fingerprint(points),
        consensus_includes_target_view=True,
        consensus_reference_run_uri="runs/pm-append",
        reference_dependency_note="fixture",
        source_run_uri="runs/source-run",
        source_run_manifest=_fingerprint(dummy),
        source_checkpoint_frames=checkpoint_frames,
        clip_config=ArtifactFingerprint(
            uri="configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_g2.json",
            sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
            source="measured",
        ),
        proxy=_fingerprint(dummy),
        proxy_dimensions=(1280, 720),
        proxy_to_raw_scale=1.5,
        frame_count=1800,
        detector=mr.DEFAULT_DETECTOR,
        box_margins=(0.25, 0.60),
        onsets=onsets,
        claim_boundaries=mr.CLAIM_BOUNDARIES,
        **extras,
    )
    iteration_dir = tmp_path / "reprompt" / TARGET_VIEW / f"iter{iteration}"
    iteration_dir.mkdir(parents=True)
    path = iteration_dir / mr.PLAN_NAME
    path.write_text(plan.model_dump_json(indent=2) + "\n")
    return path, extras


def _masks_for_factory(good_by_frame: dict[int, tuple[np.ndarray, float]]) -> Any:
    def masks_for(prompt: dict[str, Any]) -> list[tuple[np.ndarray, float]]:
        frame = int(prompt["frame_index"])
        centroid, half = good_by_frame[frame]
        if prompt["intended_target"] == "rear_body":
            # Every rear_body candidate misses: too small, or far from the consensus point.
            return [(_disc(centroid, half / 5), 0.9), (_disc(centroid + 8 * half, half), 0.95)]
        return [
            (_disc(centroid, half), 0.7),
            (_disc(centroid, half / 5), 0.8),
            (_disc(centroid + np.array([6 * half, 0.0]), half), 0.95),
        ]

    return masks_for


def test_decode_writes_decisions_and_derives_rebound_schedules(
    rig: mvg.CameraRig, tmp_path: Path
) -> None:
    chassis, chassis_centroid, chassis_half = _onset(
        rig, "chassis", 500, np.array([40.0, -30.0, -20.0]), 40.0
    )
    rear, rear_centroid, rear_half = _onset(
        rig, "rear_body", 700, np.array([-60.0, -20.0, 30.0]), 30.0
    )
    plan_path, extras = _plan(rig, tmp_path, onsets=(chassis, rear))
    decoder = _StubDecoder(
        plan_path.parent / mr.CALIBRATION_DIR_NAME / "results",
        _masks_for_factory(
            {500: (chassis_centroid, chassis_half), 700: (rear_centroid, rear_half)}
        ),
    )

    decisions_path = mr.decode_plan(plan_path, repository_root=ROOT, decoder=decoder, rig=rig)

    # One warm worker: one preview and one batch per onset frame, closed by the caller only.
    assert [c for c, _ in decoder.requests] == ["frame_preview", "batch_decode"] * 2
    assert decoder.closed is False
    decisions = mr.load_decisions(decisions_path)
    assert (decisions.accepted_count, decisions.rejected_count) == (1, 1)
    accepted, rejected = decisions.decisions
    assert (accepted.target, accepted.onset_frame, accepted.decision) == (
        "chassis",
        500,
        "accepted",
    )
    assert accepted.accepted is not None
    assert accepted.accepted.calibration_candidate_id == "t000500-b01"
    assert accepted.accepted.candidate_index == 0  # geometry, not the decoder's 0.95, decided
    assert len(accepted.candidates) == 6  # two prompts x three candidates, all kept
    assert (rejected.target, rejected.decision) == ("rear_body", "rejected")
    assert decisions.source_corrections_dropped_out_of_range == (2700,)
    assert decisions.schedule_blocked_reason is None
    assert set(decisions.schedules) == {"human-plus-consensus", "consensus-only"}

    # Both schedules bind to the bytes of the derived manifest on disk (fingerprint rebinding).
    calibration_dir = plan_path.parent / mr.CALIBRATION_DIR_NAME
    manifest_path = calibration_dir / "calibration_manifest.json"
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert decisions.derived_calibration_manifest is not None
    assert decisions.derived_calibration_manifest.sha256 == manifest_sha
    manifest = json.loads(manifest_path.read_text())
    assert (
        manifest["derived_from_calibration"]["sha256"]
        == extras["source_calibration_manifest"].sha256
    )
    new = {c["candidate_id"]: c for c in manifest["candidates"] if c["selected_by"] == "agent"}
    assert set(new) == {"t000500-b01", "t000500-b02", "t000700-b01", "t000700-b02"}
    assert new["t000500-b01"]["human_accepted"] is True
    assert new["t000500-b01"]["selected_for_correction"] is True
    assert new["t000500-b01"]["human_selected_candidate_index"] == 0
    assert all(not new[k]["human_accepted"] for k in ("t000500-b02", "t000700-b01", "t000700-b02"))

    schedules = {
        arm: MuggledSAMMultiKeyframeCorrectionSchedule.model_validate_json(
            Path(item.uri).read_text()
        )
        for arm, item in decisions.schedules.items()
    }
    for schedule in schedules.values():
        assert schedule.calibration_manifest_fingerprint.sha256 == manifest_sha
        assert {
            c.multiplex_slot for c in schedule.corrections if c.frame.analysis_frame_index == 0
        } == {0, 1, 2, 3}
        assert all(c.frame.analysis_frame_index < 1800 for c in schedule.corrections)
    later = {
        arm: [
            (c.frame.analysis_frame_index, c.target_id, c.selected_by)
            for c in s.corrections
            if c.frame.analysis_frame_index
        ]
        for arm, s in schedules.items()
    }
    assert later["human-plus-consensus"] == [(500, "chassis", "agent"), (900, "chassis", "human")]
    assert later["consensus-only"] == [(500, "chassis", "agent")]
    new_mask = next(
        c for c in schedules["consensus-only"].corrections if c.frame.analysis_frame_index == 500
    )
    mask_path = Path(new_mask.calibration_mask_fingerprint.uri)
    assert (
        mask_path.is_file()
        and hashlib.sha256(mask_path.read_bytes()).hexdigest()
        == new_mask.calibration_mask_fingerprint.sha256
    )
    assert mask_path.parent == calibration_dir / "results" / "masks"

    # The schedules pass the tracker's own integrity loader, the one battle-muggled-smoke runs.
    config = G2PreprocessingManifest.model_validate_json(CONFIG.read_text())
    proxy = next(p for p in config.proxies if p.view_id == "static-c10379")
    for arm, item in decisions.schedules.items():
        payload, metadata = _load_multi_keyframe_correction_schedule(
            schedule_path=Path(item.uri),
            repository_root=ROOT,
            config_path=CONFIG,
            proxy=proxy,
            analysis_fps=30.0,
            max_frame_exclusive=1800,
        )
        assert [s["target"] for s in payload["seeds"]] == list(TARGETS)
        assert metadata.agent_selected_correction_frame_indices == (500,)
        assert metadata.scheduled_correction_frame_indices == (
            (500, 900) if arm == "human-plus-consensus" else (500,)
        )

    provenance = MultiviewConsensusProvenanceFile.model_validate_json(
        (calibration_dir / mr.PROVENANCE_NAME).read_text()
    )
    (record,) = provenance.corrections
    assert (record.candidate_id, record.selected_by, record.provenance) == (
        "t000500-b01",
        "agent",
        "multiview_consensus",
    )
    assert (record.iteration, record.onset_frame, record.target_error_px) == (1, 500, 60.0)
    assert record.views_used == ("C10001", "C10002", "C10003")
    assert len(record.rejected_alternatives) == 5
    assert record.consensus_reference_run_uri == "runs/pm-append"
    log = [
        json.loads(line)
        for line in (calibration_dir / "agent_acceptances.jsonl").read_text().splitlines()
    ]
    assert [(item["candidate_id"], item["provenance"]) for item in log] == [
        ("t000500-b01", "multiview_consensus")
    ]


def test_policy_keyframe_budget_rejects_a_passing_candidate(
    rig: mvg.CameraRig, tmp_path: Path
) -> None:
    chassis, centroid, half = _onset(rig, "chassis", 500, np.array([40.0, -30.0, -20.0]), 40.0)
    plan_path, _ = _plan(rig, tmp_path, onsets=(chassis,))
    policy = json.loads(POLICY_V4.read_text())
    policy.update({"policy_version": "1", "maximum_later_correction_keyframes_per_target": 1})
    policy_path = tmp_path / "policy_one.json"
    policy_path.write_text(json.dumps(policy))
    decoder = _StubDecoder(
        plan_path.parent / mr.CALIBRATION_DIR_NAME / "results",
        _masks_for_factory({500: (centroid, half)}),
    )
    decisions = mr.load_decisions(
        mr.decode_plan(
            plan_path, repository_root=ROOT, decoder=decoder, rig=rig, correction_policy=policy_path
        )
    )
    (decision,) = decisions.decisions
    assert decision.decision == "rejected" and decision.reason.startswith("policy_keyframe_limit")
    assert any(c.passed for c in decision.candidates)
    assert set(decisions.schedules) == {"human-plus-consensus", "consensus-only"}


def test_decode_of_a_geometry_seeded_source_writes_an_agent_schedule(
    rig: mvg.CameraRig, tmp_path: Path
) -> None:
    """A multiview (geometry-seeded) run has no human calibration; its accepted corrections go
    into a MultiviewAgentCorrectionSchedule bound to the seed manifest, consensus-only arm only."""
    chassis, centroid, half = _onset(rig, "chassis", 500, np.array([40.0, -30.0, -20.0]), 40.0)
    plan_path, extras = _plan(rig, tmp_path, onsets=(chassis,), schedule_capable=False)
    decoder = _StubDecoder(
        plan_path.parent / mr.CALIBRATION_DIR_NAME / "results",
        _masks_for_factory({500: (centroid, half)}),
    )
    decisions = mr.load_decisions(
        mr.decode_plan(plan_path, repository_root=ROOT, decoder=decoder, rig=rig)
    )
    assert decisions.accepted_count == 1 and set(decisions.schedules) == {"consensus-only"}
    assert decisions.schedule_blocked_reason is None
    schedule_path = ROOT / decisions.schedules["consensus-only"].uri
    schedule = MultiviewAgentCorrectionSchedule.model_validate_json(schedule_path.read_text())
    assert schedule.seed_manifest == extras["source_geometric_seed_manifest"]
    assert [c.analysis_frame_index for c in schedule.corrections] == [500]
    correction = schedule.corrections[0]
    # chassis is the first accepted part of the fixture manifest -> slot 0.
    assert (correction.target, correction.multiplex_slot, correction.iteration) == ("chassis", 0, 1)
    assert correction.selected_by == "agent" and correction.provenance == "multiview_consensus"
    assert (
        hashlib.sha256((ROOT / correction.mask.uri).read_bytes()).hexdigest()
        == correction.mask.sha256
    )
    assert (plan_path.parent / mr.CALIBRATION_DIR_NAME / mr.PROVENANCE_NAME).is_file()

    commands = mr.run_commands(plan_path.parent, repository_root=ROOT)
    by_arm = {c.arm: c for c in commands}
    assert by_arm["human-plus-consensus"].mode == "blocked"
    assert "no human corrections" in by_arm["human-plus-consensus"].note
    argv = list(by_arm["consensus-only"].argv)
    assert by_arm["consensus-only"].mode == "full_run"
    assert "--four-part-multiview-first-minute" in argv
    assert argv[argv.index("--agent-correction-schedule") + 1].endswith(mr.AGENT_SCHEDULE_NAME)
    assert "--multi-keyframe-correction-schedule" not in argv
    assert "--recording" not in argv


def test_decode_carries_a_previous_agent_schedule_forward(
    rig: mvg.CameraRig, tmp_path: Path
) -> None:
    chassis, centroid, half = _onset(rig, "chassis", 500, np.array([40.0, -30.0, -20.0]), 40.0)
    plan_path, _ = _plan(rig, tmp_path, onsets=(chassis,), schedule_capable=False)
    decoder = _StubDecoder(
        plan_path.parent / mr.CALIBRATION_DIR_NAME / "results",
        _masks_for_factory({500: (centroid, half)}),
    )
    first = mr.load_decisions(
        mr.decode_plan(plan_path, repository_root=ROOT, decoder=decoder, rig=rig)
    )
    # Iteration 2: the source run applied the iteration-1 agent schedule; a new onset at 900.
    chassis2, centroid2, half2 = _onset(rig, "chassis", 900, np.array([40.0, -30.0, -20.0]), 40.0)
    plan2_path, _ = _plan(
        rig, tmp_path / "second", onsets=(chassis2,), schedule_capable=False, iteration=2
    )
    plan2 = mr.load_plan(plan2_path)
    plan2 = plan2.model_copy(update={"source_agent_schedule": first.schedules["consensus-only"]})
    plan2_path.write_text(plan2.model_dump_json(indent=2) + "\n")
    decoder2 = _StubDecoder(
        plan2_path.parent / mr.CALIBRATION_DIR_NAME / "results",
        _masks_for_factory({900: (centroid2, half2)}),
    )
    second = mr.load_decisions(
        mr.decode_plan(plan2_path, repository_root=ROOT, decoder=decoder2, rig=rig)
    )
    schedule = MultiviewAgentCorrectionSchedule.model_validate_json(
        (ROOT / second.schedules["consensus-only"].uri).read_text()
    )
    assert [(c.analysis_frame_index, c.iteration) for c in schedule.corrections] == [
        (500, 1),
        (900, 2),
    ]


# -- run commands --------------------------------------------------------------------------------


def test_resume_frame_respects_the_arm_prefix() -> None:
    kwargs = dict(checkpoint_frames=(300, 327, 900, 950), source_correction_frames=(327, 900))
    # human-plus-consensus keeps every source correction: nearest checkpoint at or before onset.
    assert (
        mr.choose_resume_frame(
            earliest_onset=1000, arm_correction_frames=(327, 900, 1000), **kwargs
        )
        == 950
    )
    # consensus-only lacks 327 and 900: a checkpoint at 327 (state before stepping 327) is fine,
    # anything later would carry a human correction in its prefix.
    assert (
        mr.choose_resume_frame(earliest_onset=1000, arm_correction_frames=(1000,), **kwargs) == 327
    )
    assert (
        mr.choose_resume_frame(earliest_onset=296, arm_correction_frames=(296,), **kwargs) is None
    )
    assert mr.choose_resume_frame(earliest_onset=300, arm_correction_frames=(300,), **kwargs) == 300


def test_run_emits_both_arms_with_full_runs_and_resume_variants(
    rig: mvg.CameraRig, tmp_path: Path
) -> None:
    chassis, centroid, half = _onset(rig, "chassis", 500, np.array([40.0, -30.0, -20.0]), 40.0)
    plan_path, _ = _plan(rig, tmp_path, onsets=(chassis,), checkpoint_frames=(300, 900))
    decoder = _StubDecoder(
        plan_path.parent / mr.CALIBRATION_DIR_NAME / "results",
        _masks_for_factory({500: (centroid, half)}),
    )
    mr.decode_plan(plan_path, repository_root=ROOT, decoder=decoder, rig=rig)

    commands = mr.run_commands(plan_path.parent, repository_root=ROOT)
    by_arm = {c.arm: c for c in commands}
    assert set(by_arm) == {"consensus-only", "human-plus-consensus"}
    for arm, command in by_arm.items():
        assert command.mode == "full_run"
        argv = list(command.argv)
        assert argv[:3] == [
            "battle-muggled-smoke",
            "--config",
            "configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_g2.json",
        ]
        assert "--four-part-static-focused" in argv and "--max-frames" in argv
        assert argv[argv.index("--max-side-length") + 1] == "1280"
        assert argv[argv.index("--prompt-memory-semantics") + 1] == "append"
        assert argv[argv.index("--checkpoint-every") + 1] == "300"
        assert argv[argv.index("--multi-keyframe-correction-schedule") + 1] == command.schedule.uri
        assert argv[argv.index("--run-root") + 1].endswith(f"arms/{arm}")
        assert "--resume-run" not in argv and "--drop-correction-frame" not in argv
        # Checkpoint 300 precedes onset 500 for both arms; the resume variant is kept aside.
        assert command.resume_argv is not None
        assert list(command.resume_argv)[-4:] == [
            "--resume-run",
            "runs/source-run",
            "--resume-at",
            "300",
        ]
        assert command.consensus_correction_frames == (500,)
    assert by_arm["consensus-only"].dropped_human_correction_frames == (900,)
    assert by_arm["consensus-only"].later_correction_frames == (500,)
    assert by_arm["human-plus-consensus"].dropped_human_correction_frames == ()
    assert by_arm["human-plus-consensus"].later_correction_frames == (500, 900)
    assert Path(by_arm["consensus-only"].schedule.uri).name.endswith("consensus-only.json")
    script = (plan_path.parent / mr.COMMANDS_SCRIPT_NAME).read_text()
    assert script.count("uv run battle-muggled-smoke") >= 2
    written = json.loads((plan_path.parent / mr.COMMANDS_NAME).read_text())
    assert [item["arm"] for item in written] == ["consensus-only", "human-plus-consensus"]

    resumed = mr.run_commands(
        plan_path.parent, repository_root=ROOT, prefer_resume=True, arms=("human-plus-consensus",)
    )
    (command,) = resumed
    assert command.mode == "checkpoint_resumed" and command.resume_at == 300
    assert list(command.argv)[-2:] == ["--resume-at", "300"]


# -- iteration cap -------------------------------------------------------------------------------


def test_iterations_are_capped_at_three(rig: mvg.CameraRig, tmp_path: Path) -> None:
    chassis, *_ = _onset(rig, "chassis", 500, np.array([40.0, -30.0, -20.0]), 40.0)
    assert mr.next_iteration(None, None) == 1
    assert mr.next_iteration(None, 3) == 3
    with pytest.raises(ValueError, match="capped at 3"):
        mr.next_iteration(None, 4)
    plan_path, _ = _plan(rig, tmp_path, onsets=(chassis,), schedule_capable=False, iteration=3)
    previous = mr.load_plan(plan_path)
    with pytest.raises(ValueError, match="capped at 3"):
        mr.next_iteration(previous, None)
    with pytest.raises(ValueError, match="disagrees"):
        mr.next_iteration(previous.model_copy(update={"iteration": 1}), 3)
    assert mr.next_iteration(previous.model_copy(update={"iteration": 1}), None) == 2
    with pytest.raises(ValueError):
        MultiviewRepromptPlan.model_validate({**previous.model_dump(mode="json"), "iteration": 4})


# -- real data -----------------------------------------------------------------------------------


@pytest.mark.real_data
def test_plan_on_the_pm_append_consensus_matches_the_ensemble_v2_intervals(tmp_path: Path) -> None:
    from conftest import require_artifact

    require_artifact(ROOT / mr.DEFAULT_CONSENSUS_ROOT / "consensus_points.npz")
    require_artifact(ROOT / mr.DEFAULT_SOURCE_RUN / "manifest.json")
    path = mr.plan_reprompt(ROOT, output_root=tmp_path / "reprompt")
    plan = mr.load_plan(path)
    assert plan.consensus_includes_target_view is True
    assert plan.source_checkpoint_frames == (327, 900, 1172, 1235)
    onsets = {(o.target, o.onset_frame): o for o in plan.onsets}
    # The same runs the ensemble v2 fallback rule selected (merge < 15, drop < 10).
    assert {k for k, o in onsets.items() if o.status == "planned"} == {
        ("chassis", 296),
        ("chassis", 475),
        ("chassis", 1049),
    }
    assert onsets[("chassis", 1049)].interval_end_frame_exclusive == 1085
    assert onsets[("rear_body", 1762)].status == "blocked"
    for onset in plan.planned_onsets:
        assert len(onset.prompts) == 2 and onset.static_views_used >= 3
        assert onset.radius_mm is not None and 30 < onset.radius_mm < 80
