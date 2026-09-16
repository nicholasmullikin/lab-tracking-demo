from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from battle.fixtures import synthetic_timing
from battle.human_qa import evenly_spaced_frame_indices
from battle.schemas import FixedTimestampHumanQARecord, FrameRange


def _fingerprint(uri: str) -> dict[str, str]:
    return {"uri": uri, "sha256": "0" * 64, "source": "measured"}


def _checkpoint(
    *,
    role: str,
    source_seconds: float,
    analysis_frame_index: int,
    disposition: str = "pending",
) -> dict[str, object]:
    checkpoint: dict[str, object] = {
        "role": role,
        "clock": "source",
        "source_seconds": source_seconds,
        "source_frame_index": round(source_seconds * 60),
        "analysis_frame_index": analysis_frame_index,
        "evidence": [
            {
                "uri": "runs/example/human_qa/two-checkpoints.png",
                "sha256": "1" * 64,
                "artifact_kind": "two_checkpoint_contact_sheet",
            }
        ],
        "disposition": disposition,
    }
    if disposition != "pending":
        checkpoint["reviewed_by"] = "human-reviewer"
        checkpoint["reviewed_at"] = "2026-09-13T21:00:00-04:00"
    return checkpoint


def _record_payload(
    *,
    easy_disposition: str = "pending",
    hard_disposition: str = "pending",
    overall_status: str = "pending",
) -> dict[str, object]:
    return {
        "manifest_kind": "fixed_timestamp_human_qa",
        "qa_protocol": "assembly101_easy_hard_source_timestamps_v1",
        "selection_rule": "one_easy_manipulation_and_one_hard_or_occluded_manipulation",
        "run_id": "completed-run",
        "run_profile": "full_duration",
        "method_id": "example-method",
        "clip_id": "synthetic-clip",
        "view_id": "static-01",
        "run_manifest_fingerprint": _fingerprint("runs/completed-run/manifest.json"),
        "config_fingerprint": _fingerprint("configs/clips/example.json"),
        "source_video_fingerprint": {
            **_fingerprint("data/raw/example.mp4"),
            "source": "approved_config",
        },
        "timing": synthetic_timing().model_dump(mode="json"),
        "run_analysis_frame_range": {"start_frame": 0, "end_frame_exclusive": 300},
        "checkpoints": [
            _checkpoint(
                role="easy_manipulation",
                source_seconds=1.0,
                analysis_frame_index=30,
                disposition=easy_disposition,
            ),
            _checkpoint(
                role="hard_or_occluded_manipulation",
                source_seconds=2.0,
                analysis_frame_index=60,
                disposition=hard_disposition,
            ),
        ],
        "overall_status": overall_status,
        "ground_truth_accuracy_claim": False,
    }


def test_pending_human_qa_record_is_valid_and_unattributed() -> None:
    record = FixedTimestampHumanQARecord.model_validate(_record_payload())

    assert record.overall_status.value == "pending"
    assert all(checkpoint.reviewed_by is None for checkpoint in record.checkpoints)
    assert record.ground_truth_accuracy_claim is False


def test_candidate_grid_includes_evenly_spaced_completed_run_endpoints() -> None:
    indices = evenly_spaced_frame_indices(
        FrameRange(start_frame=0, end_frame_exclusive=5400), 12
    )

    assert indices == (0, 491, 982, 1472, 1963, 2454, 2945, 3436, 3927, 4417, 4908, 5399)


def test_human_completed_record_requires_and_retains_attribution() -> None:
    record = FixedTimestampHumanQARecord.model_validate(
        _record_payload(
            easy_disposition="pass",
            hard_disposition="flag",
            overall_status="flag",
        )
    )

    assert record.overall_status.value == "flag"
    assert all(checkpoint.reviewed_by == "human-reviewer" for checkpoint in record.checkpoints)


@pytest.mark.parametrize("invalid_case", ["count", "clock", "value", "evidence"])
def test_human_qa_rejects_wrong_checkpoint_contract(invalid_case: str) -> None:
    payload = _record_payload()
    checkpoints = payload["checkpoints"]
    assert isinstance(checkpoints, list)
    if invalid_case == "count":
        payload["checkpoints"] = checkpoints[:1]
    elif invalid_case == "clock":
        checkpoints[0]["clock"] = "analysis"
    elif invalid_case == "value":
        checkpoints[0]["source_seconds"] = 1.1
    else:
        checkpoints[0]["evidence"] = []

    with pytest.raises(ValidationError):
        FixedTimestampHumanQARecord.model_validate(payload)


def test_human_qa_rejects_duplicate_checkpoints() -> None:
    payload = _record_payload()
    checkpoints = payload["checkpoints"]
    assert isinstance(checkpoints, list)
    checkpoints[1]["source_seconds"] = 1.0
    checkpoints[1]["source_frame_index"] = 60
    checkpoints[1]["analysis_frame_index"] = 30

    with pytest.raises(ValidationError, match="distinct source timestamps"):
        FixedTimestampHumanQARecord.model_validate(payload)


def test_non_pending_human_qa_requires_reviewer_attribution() -> None:
    payload = _record_payload(easy_disposition="pass")
    checkpoints = payload["checkpoints"]
    assert isinstance(checkpoints, list)
    checkpoints[0].pop("reviewed_by")

    with pytest.raises(ValidationError, match="reviewer identity and time"):
        FixedTimestampHumanQARecord.model_validate(payload)


@pytest.mark.parametrize(
    ("easy", "hard", "overall"),
    [
        ("pass", "pass", "pass"),
        ("pass", "pending", "pending"),
        ("pending", "flag", "flag"),
        ("pass", "fail", "fail"),
    ],
)
def test_human_qa_aggregate_status_is_conservative(
    easy: str, hard: str, overall: str
) -> None:
    payload = _record_payload(
        easy_disposition=easy,
        hard_disposition=hard,
        overall_status=overall,
    )
    record = FixedTimestampHumanQARecord.model_validate(payload)
    wrong = deepcopy(payload)
    wrong["overall_status"] = "pass" if overall != "pass" else "pending"

    assert record.overall_status.value == overall
    with pytest.raises(ValidationError, match="conservatively derived"):
        FixedTimestampHumanQARecord.model_validate(wrong)
