from __future__ import annotations

from pathlib import Path

import pytest

from battle.fixtures import synthetic_run_manifest, synthetic_timing
from battle.metrics import calculate_success_measure
from battle.schemas import (
    ClockName,
    G2PreprocessingManifest,
    MethodState,
    NormalizedBox,
    RunManifest,
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
