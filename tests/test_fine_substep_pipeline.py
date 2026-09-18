from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import require_artifact

from battle.fine_substep_contract import load_contract
from battle.fine_substep_pipeline import (
    build_crop_manifest,
    build_crop_sample,
    contact_bonus,
    evaluate_against_agent_labels,
    fuse_scores,
    load_observations,
    monotonic_substep_dp,
    phase_condition_scores,
    sample_frame_indices,
    temporal_delta_features,
)
from battle.schemas import FrameObservations


def _empty_obs(frame_index: int) -> FrameObservations:
    return FrameObservations.model_validate(
        {
            "view_id": "static-c10379",
            "analysis_frame_index": frame_index,
            "source_seconds": 294.0 + frame_index / 30,
            "objects": [],
            "hands": [],
        }
    )


def test_sample_frame_indices_default_three_fps() -> None:
    frames = sample_frame_indices(sample_fps=3.0, frame_count=600)
    assert frames[0] == 0
    assert frames[1] == 10
    assert len(frames) == 60


@pytest.mark.real_data
def test_build_crop_sample_prefers_wilor_over_mediapipe() -> None:
    wilor = FrameObservations.model_validate_json(
        require_artifact("runs/wilor-hands-static-20s-audited-source-state/observations.jsonl")
        .read_text()
        .splitlines()[0]
    )
    mediapipe = FrameObservations.model_validate_json(
        require_artifact(
            "runs/mediapipe-hands-static-20s-fused-dedup-th035-20260916t0428z/observations.jsonl"
        )
        .read_text()
        .splitlines()[0]
    )
    sample = build_crop_sample(0, wilor=wilor, mediapipe=mediapipe, parts=_empty_obs(0))
    assert sample.hand_cue_source == "wilor"
    assert sample.mediapipe_fallback_used is False


@pytest.mark.real_data
def test_build_crop_sample_falls_back_to_mediapipe() -> None:
    mediapipe = FrameObservations.model_validate_json(
        require_artifact(
            "runs/mediapipe-hands-static-20s-fused-dedup-th035-20260916t0428z/observations.jsonl"
        )
        .read_text()
        .splitlines()[0]
    )
    sample = build_crop_sample(0, wilor=_empty_obs(0), mediapipe=mediapipe, parts=_empty_obs(0))
    assert sample.hand_cue_source == "mediapipe"
    assert sample.mediapipe_fallback_used is True


def test_monotonic_dp_is_non_decreasing() -> None:
    scores = np.array(
        [
            [0.9, 0.1, 0.1],
            [0.8, 0.7, 0.1],
            [0.2, 0.85, 0.2],
            [0.1, 0.2, 0.95],
        ]
    )
    path, _cost = monotonic_substep_dp(scores, 3)
    assert path == sorted(path)


def test_temporal_delta_and_fusion() -> None:
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    motion = temporal_delta_features(embeddings)
    assert motion[0] == 0.0
    assert motion[1] > 0.0
    clip = np.ones((3, 2), dtype=np.float32) * 0.5
    fused = fuse_scores(
        clip,
        motion=motion,
        contact=np.array([0.0, 1.0, 0.5]),
        weights={"motion": 0.1, "contact": 0.2},
    )
    assert fused[1, 0] > fused[0, 0]


def test_contact_bonus_missing_is_zero() -> None:
    assert contact_bonus(None) == 0.0
    assert contact_bonus(0.0) == 1.0


def test_phase_conditioning_is_soft_and_only_gates_tool_language() -> None:
    scores = np.zeros((2, 11), dtype=np.float32)
    conditioned = phase_condition_scores(
        scores, sample_frames=[300, 400], tool_cues=np.asarray([0.0, 0.02])
    )

    assert conditioned[0, 6] == np.float32(-0.12)
    assert conditioned[0, 5] == 0
    assert conditioned[1, 6] == np.float32(0.08)


def test_evaluate_against_agent_labels_reports_checkpoints() -> None:
    contract = load_contract(
        Path("configs/fine_substeps/assembly101_focused_static_first_20s_agent_labels.json")
    )
    sample_frames = sample_frame_indices(sample_fps=3.0)
    substep_indices = list(range(len(sample_frames)))
    substep_indices = [min(index, len(contract.substeps) - 1) for index in substep_indices]
    report = evaluate_against_agent_labels(
        contract,
        sample_frames=sample_frames,
        substep_indices=substep_indices,
        alignment_name="test",
    )
    assert report["alignment_name"] == "test"
    assert len(report["checkpoints"]) == len(contract.checkpoints)


@pytest.mark.real_data
def test_build_crop_manifest_from_real_runs() -> None:
    repo = Path(".")
    wilor = load_observations(
        require_artifact(
            repo / "runs/wilor-hands-static-20s-audited-source-state/observations.jsonl"
        )
    )
    mediapipe = load_observations(
        require_artifact(
            repo
            / "runs/mediapipe-hands-static-20s-fused-dedup-th035-20260916t0428z/observations.jsonl"
        )
    )
    parts = load_observations(
        require_artifact(
            repo
            / "runs"
            / "muggledsam-sam3-four-part-static-focused-reassembly-static-c10379-20260916t023700z"
            / "observations.jsonl"
        )
    )
    samples = build_crop_manifest([0, 30, 300], wilor=wilor, mediapipe=mediapipe, parts=parts)
    assert len(samples) == 3
    assert all(sample.workspace_box.x1 > sample.workspace_box.x0 for sample in samples)
