from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from battle.fixtures import synthetic_run_manifest, synthetic_timing
from battle.metrics import calculate_success_measure
from battle.schemas import (
    ClockName,
    G2PreprocessingManifest,
    MethodState,
    MuggledSAMMultiKeyframeCorrectionPolicy,
    NormalizedBox,
    RunManifest,
    SmokeRunMetadata,
    TrackerSlotDiagnostic,
)


def test_fixture_has_the_planned_four_clocks() -> None:
    timing = synthetic_timing()

    assert timing.clocks.fps_for(ClockName.SOURCE) == 60
    assert timing.clocks.fps_for(ClockName.ANALYSIS) == 30
    assert timing.clocks.fps_for(ClockName.ANNOTATION) == 30
    assert timing.clocks.fps_for(ClockName.POSE) == 60


def test_clock_mapping_converts_analysis_and_pose_frames() -> None:
    timing = synthetic_timing()

    assert timing.source_seconds_for_frame(ClockName.ANALYSIS, 30) == pytest.approx(1.0)
    assert timing.source_seconds_for_frame(ClockName.POSE, 60) == pytest.approx(1.0)


def test_normalized_box_cannot_extend_past_frame() -> None:
    with pytest.raises(ValueError, match="inside"):
        NormalizedBox(x=0.9, y=0.2, width=0.2, height=0.3)


def test_observations_must_be_monotonic_per_view() -> None:
    fixture = synthetic_run_manifest()

    with pytest.raises(ValueError, match="monotonically"):
        RunManifest(
            run_id=fixture.run_id,
            clip=fixture.clip,
            coverage=fixture.coverage,
            chunk_policy=fixture.chunk_policy,
            method_statuses=fixture.method_statuses,
            observations=tuple(reversed(fixture.observations)),
        )


def test_fixture_success_measure_is_contract_coverage_not_accuracy() -> None:
    fixture = synthetic_run_manifest()
    measure = calculate_success_measure(fixture.coverage, fixture.method_statuses)

    assert measure.coverage_ratio == 1.0
    assert measure.successful_method_ratio == 0.5
    assert measure.combined_ratio == 0.75
    assert fixture.method_statuses[1].state is MethodState.NOT_RUN


def test_approved_assembly101_g2_manifest_has_consistent_proxy_clocks() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "clips" / "assembly101_nusar_9033_g2.json"
    manifest = G2PreprocessingManifest.model_validate_json(config_path.read_text())

    assert manifest.g1.state == "approved"
    assert manifest.g2.state == "approved"
    assert manifest.raw_frame_range.frame_count == 10_800
    assert manifest.analysis_frame_range.frame_count == 5_400
    assert [(proxy.dimensions.width, proxy.dimensions.height) for proxy in manifest.proxies] == [
        (1280, 720),
        (954, 720),
    ]
    assert not manifest.annotations_or_poses_downloaded_by_g2
    assert not manifest.annotations_or_poses_used_by_g2


def test_focused_four_part_manifest_maps_old_frame_3120_to_new_frame_zero() -> None:
    config_path = (
        Path(__file__).parents[1]
        / "configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_g2.json"
    )
    manifest = G2PreprocessingManifest.model_validate_json(config_path.read_text())

    assert manifest.source_interval.start_seconds == 294.0
    assert manifest.raw_frame_range.start_frame == 17_640
    assert manifest.analysis_frame_range.start_frame == 8_820
    assert manifest.proxy_frame_range.start_frame == 0
    assert manifest.proxy_frame_range.frame_count == 2_781
    assert manifest.proxies[0].frame_count == 2_781
    assert manifest.proxies[0].view_id == "static-c10379"


def test_focused_ego_manifest_matches_the_static_reassembly_interval() -> None:
    config_path = (
        Path(__file__).parents[1]
        / "configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_ego_g2.json"
    )
    manifest = G2PreprocessingManifest.model_validate_json(config_path.read_text())

    assert manifest.source_interval.start_seconds == 294.0
    assert manifest.source_interval.end_seconds == 386.7
    assert manifest.proxy_frame_range.frame_count == 2_781
    assert manifest.proxies[0].dimensions.width == 954
    assert manifest.proxies[0].view_id == "ego-hmc21110305"


def test_second_correction_policy_version_allows_five_later_keyframes() -> None:
    policy_path = (
        Path(__file__).parents[1]
        / "configs/muggledsam_static_four_part_reassembly_focused_correction_policy_v2.json"
    )
    policy = MuggledSAMMultiKeyframeCorrectionPolicy.model_validate_json(policy_path.read_text())

    assert policy.policy_version == "2"
    assert policy.maximum_later_correction_keyframes_per_target == 5
    invalid = policy.model_dump()
    invalid["maximum_later_correction_keyframes_per_target"] = 6
    with pytest.raises(ValidationError):
        MuggledSAMMultiKeyframeCorrectionPolicy.model_validate(invalid)
    invalid["policy_version"] = "1"
    invalid["maximum_later_correction_keyframes_per_target"] = 5
    with pytest.raises(ValidationError, match="v1 permits at most three"):
        MuggledSAMMultiKeyframeCorrectionPolicy.model_validate(invalid)


def test_object_score_is_not_clamped_to_a_probability_range() -> None:
    """The presence logit is unbounded; clamping it would erase the lost-object signal."""
    diagnostic = TrackerSlotDiagnostic(
        object_id="sam3-02",
        label="yellow_toy_top",
        multiplex_slot=2,
        object_score=-5.44,
        iou_prediction=0.0,
        active=False,
    )

    assert diagnostic.object_score == -5.44
    assert diagnostic.active is False
    assert diagnostic.corrected is False


def test_iou_prediction_stays_within_its_sigmoid_bounds() -> None:
    with pytest.raises(ValidationError):
        TrackerSlotDiagnostic(
            object_id="sam3-02",
            label="yellow_toy_top",
            multiplex_slot=2,
            object_score=11.0,
            iou_prediction=1.4,
            active=True,
        )


def test_slot_diagnostics_round_trip_with_and_without_the_policy_fields() -> None:
    legacy = TrackerSlotDiagnostic.model_validate(
        {
            "object_id": "sam3-00",
            "label": "chassis",
            "multiplex_slot": 0,
            "object_score": 3.2,
            "iou_prediction": 0.9,
            "active": True,
            "corrected": False,
        }
    )
    policed = TrackerSlotDiagnostic.model_validate(
        {
            **legacy.model_dump(mode="json"),
            "contested_fraction": 0.25,
            "memory_written": False,
            "memory_gate_reason": "contested",
        }
    )

    assert legacy.contested_fraction is None and legacy.memory_gate_reason is None
    assert policed.memory_written is False and policed.contested_fraction == 0.25
    assert TrackerSlotDiagnostic.model_validate_json(policed.model_dump_json()) == policed
    with pytest.raises(ValidationError):
        TrackerSlotDiagnostic.model_validate(
            {**legacy.model_dump(mode="json"), "memory_gate_reason": "because"}
        )


def test_tracker_memory_policy_defaults_are_off_and_bands_are_ordered() -> None:
    from battle.schemas import TrackerMemoryPolicy

    default = TrackerMemoryPolicy()
    assert default.is_default and default.run_id_suffix() == ""
    arm = TrackerMemoryPolicy(slot_exclusivity="argmax", memory_gate="on")
    assert not arm.is_default and arm.run_id_suffix() == "xargmax-gon"
    assert TrackerMemoryPolicy(memory_gate="on").run_id_suffix() == "gon"
    assert "--slot-exclusivity" in arm.worker_arguments()
    assert arm.worker_arguments()[arm.worker_arguments().index("--gate-area-band") + 1] == "0.5,2.0"
    with pytest.raises(ValidationError, match="0 < low <= high"):
        TrackerMemoryPolicy(gate_area_band=(2.0, 0.5))
    with pytest.raises(ValidationError):
        TrackerMemoryPolicy(slot_exclusivity="max")


def _four_part_focused_payload(*, frame_count: int, requested_seconds: float) -> dict[str, object]:
    payload = _smoke_metadata_payload(analysis_fps=30, frame_count=frame_count)
    payload.update(
        {
            "requested_seconds": requested_seconds,
            "view_id": "static-c10379",
            "concepts": ["chassis", "interior", "rear_body", "cabin"],
            "multi_keyframe_corrections": {
                "schedule_fingerprint": payload["source_fingerprint"],
                "correction_policy_fingerprint": payload["source_fingerprint"],
                "correction_memory_semantics": "replace_prompt_memory_and_reset_frame_memory",
            },
            "rescope_reason": (
                "old proxy frame 3120 starts with four separated parts before reassembly"
            ),
        }
    )
    return payload


def test_focused_metadata_accepts_the_first_minute_bound_only_as_sixty_seconds() -> None:
    from battle.schemas import FourPartFocusedRunMetadata

    full = FourPartFocusedRunMetadata.model_validate(
        _four_part_focused_payload(frame_count=2781, requested_seconds=92.7)
    )
    bounded = FourPartFocusedRunMetadata.model_validate(
        _four_part_focused_payload(frame_count=1800, requested_seconds=60.0)
    )

    assert full.requested_analysis_frame_range.frame_count == 2781
    assert bounded.requested_analysis_frame_range.frame_count == 1800
    assert bounded.multi_keyframe_corrections.dropped_correction_frame_indices == ()
    assert bounded.multi_keyframe_corrections.frame_zero_seeds_only is False
    with pytest.raises(ValidationError, match="first-minute bound"):
        FourPartFocusedRunMetadata.model_validate(
            _four_part_focused_payload(frame_count=1800, requested_seconds=92.7)
        )
    with pytest.raises(ValidationError):
        FourPartFocusedRunMetadata.model_validate(
            _four_part_focused_payload(frame_count=1500, requested_seconds=60.0)
        )


def _smoke_metadata_payload(*, analysis_fps: object, frame_count: int) -> dict[str, object]:
    fingerprint = {
        "uri": "artifact.bin",
        "sha256": "0" * 64,
        "source": "measured",
    }
    return {
        "requested_analysis_frame_range": {
            "start_frame": 0,
            "end_frame_exclusive": frame_count,
        },
        "requested_seconds": 10.0,
        "concepts": ["hand"],
        "source_fingerprint": fingerprint,
        "proxy_fingerprint": fingerprint,
        "config_fingerprint": fingerprint,
        "adapter": {
            "name": "test-adapter",
            "version": "1.0",
            "implementation_basis": "schema test",
            "external_source_uri": "https://example.com/adapter",
        },
        "continuity": {
            "max_prompt_memory_entries": 1,
            "max_frame_memory_entries": 4,
            "detected_object_limit": 1,
        },
        "runtime_settings": {"analysis_fps": analysis_fps},
        "measurements": {"elapsed_seconds": 0.0},
        "mask_artifact_count": 0,
    }


@pytest.mark.parametrize(("analysis_fps", "frame_count"), [(30, 300), (60, 600)])
def test_smoke_frame_count_matches_analysis_clock(analysis_fps: int, frame_count: int) -> None:
    smoke = SmokeRunMetadata.model_validate(
        _smoke_metadata_payload(analysis_fps=analysis_fps, frame_count=frame_count)
    )

    assert smoke.requested_analysis_frame_range.frame_count == frame_count


@pytest.mark.parametrize(("analysis_fps", "frame_count"), [(30, 600), (60, 300)])
def test_smoke_frame_count_rejects_mismatch_in_either_direction(
    analysis_fps: int, frame_count: int
) -> None:
    with pytest.raises(ValidationError, match="requested_seconds \\* analysis_fps"):
        SmokeRunMetadata.model_validate(
            _smoke_metadata_payload(analysis_fps=analysis_fps, frame_count=frame_count)
        )


@pytest.mark.parametrize("analysis_fps", [True, "30", None, 24])
def test_smoke_analysis_clock_must_be_supported_numeric_value(analysis_fps: object) -> None:
    with pytest.raises(ValidationError, match="analysis_fps must be numeric"):
        SmokeRunMetadata.model_validate(
            _smoke_metadata_payload(analysis_fps=analysis_fps, frame_count=300)
        )
