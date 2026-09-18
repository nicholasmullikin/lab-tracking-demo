from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from battle.muggled_agent_correction import (
    accept_agent_candidate,
    decode_candidates,
    derive_calibration,
    finalize_agent_schedule,
    parse_prompt,
)
from battle.muggled_calibration import build_manifest
from battle.schemas import (
    MuggledSAMCalibrationCandidate,
    MuggledSAMMultiKeyframeCorrection,
    MuggledSAMMultiKeyframeCorrectionPolicy,
    MuggledSAMMultiKeyframeCorrectionSchedule,
)

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_g2.json"
TARGET_CONFIG = ROOT / "configs/muggledsam_static_four_part_reassembly_focused_manual_seed.json"
POLICY_V3 = (
    ROOT / "configs/muggledsam_static_four_part_reassembly_focused_correction_policy_v3.json"
)
TARGETS = ("chassis", "interior", "rear_body", "cabin")


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


def test_candidate_selected_by_defaults_to_human_and_agents_cannot_seed_frame_zero() -> None:
    human = MuggledSAMCalibrationCandidate.model_validate(
        _candidate_payload("t000000-b01", "chassis", 0)
    )
    assert human.selected_by == "human"

    agent_later = MuggledSAMCalibrationCandidate.model_validate(
        {**_candidate_payload("t001172-b01", "chassis", 1172), "selected_by": "agent"}
    )
    assert agent_later.selected_by == "agent"
    with pytest.raises(ValueError, match="never seeds"):
        MuggledSAMCalibrationCandidate.model_validate(
            {**_candidate_payload("t000000-b02", "chassis", 0), "selected_by": "agent"}
        )
    with pytest.raises(ValueError, match="human-selected"):
        MuggledSAMMultiKeyframeCorrection.model_validate(
            {
                "candidate_id": "t000000-b02",
                "human_selected_candidate_index": 0,
                "selected_by": "agent",
                "target_id": "chassis",
                "object_id": "sam3-00",
                "multiplex_slot": 0,
                "frame": _candidate_payload("t000000-b02", "chassis", 0)["frame"],
                "calibration_mask_fingerprint": {
                    "uri": "runs/x/results/masks/a.png",
                    "sha256": "0" * 64,
                    "source": "measured",
                },
            }
        )


def test_policy_v3_allows_six_keyframes_but_v2_stays_at_five() -> None:
    policy = MuggledSAMMultiKeyframeCorrectionPolicy.model_validate_json(POLICY_V3.read_text())
    assert (policy.policy_version, policy.maximum_later_correction_keyframes_per_target) == ("3", 6)
    with pytest.raises(ValueError, match="v2 permits at most five"):
        MuggledSAMMultiKeyframeCorrectionPolicy.model_validate(
            {**policy.model_dump(mode="json"), "policy_version": "2"}
        )


def test_prompt_parser_reads_box_and_points() -> None:
    target, box, fg, bg = parse_prompt("interior=800,400,900,480;fg=850,440;bg=700,300")
    assert (target, box.x1, box.y2) == ("interior", 800, 480)
    assert [(p.x, p.y) for p in fg] == [(850, 440)]
    assert [(p.x, p.y) for p in bg] == [(700, 300)]
    with pytest.raises(ValueError, match="target=x1,y1,x2,y2"):
        parse_prompt("chassis=1,2,3")


class _FakeDecoder:
    def __init__(self, results_directory: Path) -> None:
        self.results_directory = results_directory
        self.requests: list[tuple[str, dict[str, Any]]] = []

    def request(
        self, command: str, payload: dict[str, Any], timeout: float = 120
    ) -> dict[str, Any]:
        self.requests.append((command, payload))
        if command == "frame_preview":
            return {"frame_index": payload["frame_index"], "image_uri": "results/frames/x.jpg"}
        decoded = []
        for prompt in payload["prompts"]:
            candidate_id = prompt["candidate_id"]
            (self.results_directory / "masks").mkdir(parents=True, exist_ok=True)
            candidates = []
            for index in range(2):
                mask = (
                    self.results_directory / "masks" / f"{candidate_id}_candidate-{index:02d}.png"
                )
                mask.write_bytes(f"{candidate_id}:{index}".encode())
                candidates.append(
                    {
                        "candidate_index": index,
                        "iou_score": 0.5 + index / 10,
                        "mask_uri": f"results/masks/{mask.name}",
                        "is_deterministic_best": index == 1,
                    }
                )
            decoded.append(
                {
                    "box_id": prompt["box_id"],
                    "candidate_id": candidate_id,
                    "decoder_result": {
                        "api": "muggledsam_sam3_interactive",
                        "candidate_count": 2,
                        "deterministic_best_candidate_index": 1,
                        "candidates": candidates,
                        "overlay_uri": f"results/{candidate_id}_overlay.png",
                    },
                }
            )
        return {"decoded": decoded}

    def close(self) -> None:
        pass


def _finalized_source(tmp_path: Path) -> Path:
    source = tmp_path / "runs" / "source-calibration"
    (source / "results" / "masks").mkdir(parents=True)
    candidates = [_candidate_payload(f"t000000-b{i + 1:02d}", t, 0) for i, t in enumerate(TARGETS)]
    candidates.append(_candidate_payload("t000900-b01", "chassis", 900))
    for payload in candidates:
        mask = source / payload["decoder_result"]["candidates"][0]["mask_uri"]
        mask.write_bytes(payload["candidate_id"].encode())
    manifest = build_manifest(
        repository_root=ROOT,
        config_path=CONFIG,
        timestamps=(0.0, 30.0),
        result_directory=source / "results",
        calibration_id="source-calibration",
        view_id="static-c10379",
    ).model_copy(
        update={
            "candidates": tuple(
                MuggledSAMCalibrationCandidate.model_validate(item) for item in candidates
            ),
            "final_correction_schedule_uri": "runs/source-calibration/schedule.json",
            "plan_revision": 1,
        }
    )
    (source / "calibration_manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
    return source


def test_agent_correction_flow_keeps_provenance_explicit(tmp_path: Path) -> None:
    source = _finalized_source(tmp_path)
    derived_dir = tmp_path / "runs" / "agent-calibration"

    with pytest.raises(ValueError, match="finalized"):
        decode_candidates(
            calibration_dir=source,
            frame_index=1172,
            prompts=("chassis=1,1,5,5",),
            repository_root=ROOT,
            decoder=_FakeDecoder(source / "results"),
        )
    manifest_path = derive_calibration(
        source_dir=source, output_dir=derived_dir, repository_root=ROOT
    )
    derived = json.loads(manifest_path.read_text())
    assert derived["final_correction_schedule_uri"] is None
    assert derived["derived_from_calibration"]["uri"].endswith(
        "runs/source-calibration/calibration_manifest.json"
    )
    assert (derived_dir / "results/masks/t000900-b01_candidate-00.png").is_file()

    with pytest.raises(ValueError, match="later-frame only"):
        decode_candidates(
            calibration_dir=derived_dir,
            frame_index=0,
            prompts=("chassis=1,1,5,5",),
            repository_root=ROOT,
            decoder=_FakeDecoder(derived_dir / "results"),
        )
    ids = decode_candidates(
        calibration_dir=derived_dir,
        frame_index=1172,
        prompts=("chassis=900,440,1000,560", "interior=840,450,900,520;fg=870,480"),
        repository_root=ROOT,
        decoder=_FakeDecoder(derived_dir / "results"),
    )
    assert ids == ("t001172-b01", "t001172-b02")
    manifest = json.loads(manifest_path.read_text())
    decoded = {item["candidate_id"]: item for item in manifest["candidates"]}
    assert decoded["t001172-b01"]["selected_by"] == "agent"
    assert decoded["t001172-b01"]["human_accepted"] is False
    assert decoded["t001172-b02"]["pixel_fg_points"] == [
        {"schema_version": "1.0", "x": 870, "y": 480}
    ]
    assert 1172 / 30 in manifest["requested_proxy_timestamps_seconds"]

    with pytest.raises(ValueError, match="rationale"):
        accept_agent_candidate(
            calibration_dir=derived_dir,
            candidate_id="t001172-b01",
            candidate_index=1,
            rationale=" ",
        )
    with pytest.raises(ValueError, match="agent-decoded"):
        accept_agent_candidate(
            calibration_dir=derived_dir,
            candidate_id="t000900-b01",
            candidate_index=0,
            rationale="not allowed",
        )
    for candidate_id in ids:
        accepted = accept_agent_candidate(
            calibration_dir=derived_dir,
            candidate_id=candidate_id,
            candidate_index=1,
            rationale="visual review of the candidate sheet",
        )
        assert (accepted.selected_by, accepted.selected_for_correction) == ("agent", True)
    log = [
        json.loads(line)
        for line in (derived_dir / "agent_acceptances.jsonl").read_text().splitlines()
    ]
    assert [item["provenance"] for item in log] == ["agent_authored_visual_review"] * 2

    schedule_path = finalize_agent_schedule(
        calibration_dir=derived_dir,
        correction_policy_path=POLICY_V3,
        manual_seed_target_config_path=TARGET_CONFIG,
        repository_root=ROOT,
    )
    schedule = MuggledSAMMultiKeyframeCorrectionSchedule.model_validate_json(
        schedule_path.read_text()
    )
    by_frame = {
        (item.frame.analysis_frame_index, item.target_id): item.selected_by
        for item in schedule.corrections
    }
    assert by_frame[(0, "chassis")] == "human"
    assert by_frame[(900, "chassis")] == "human"
    assert by_frame[(1172, "chassis")] == "agent"
    assert by_frame[(1172, "interior")] == "agent"
    final_manifest = json.loads(manifest_path.read_text())
    assert final_manifest["final_correction_schedule_uri"].endswith(
        "runs/agent-calibration/multi_keyframe_correction_schedule.json"
    )
    # The written manifest bytes are exactly what the schedule fingerprint hashed.
    import hashlib

    assert (
        schedule.calibration_manifest_fingerprint.sha256
        == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError, match="already has"):
        finalize_agent_schedule(
            calibration_dir=derived_dir,
            correction_policy_path=POLICY_V3,
            manual_seed_target_config_path=TARGET_CONFIG,
            repository_root=ROOT,
        )
