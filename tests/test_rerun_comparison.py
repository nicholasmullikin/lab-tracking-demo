from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from battle.fixtures import synthetic_run_manifest
from battle.rerun_comparison import (
    FIRST_MINUTE_FRAME_COUNT,
    _camera_name,
    _create_first_minute_video,
    _first_minute_manifest,
    _merge_static_hands,
    measured_static_clock_shift,
)
from battle.schemas import (
    AdapterMetadata,
    ArtifactFingerprint,
    FrameRange,
    MediaPipeHandsRunMetadata,
    RuntimeMeasurements,
)


def _manifest_with_frame_count(frame_count: int):
    manifest = synthetic_run_manifest()
    template = manifest.observations[0]
    observations = tuple(
        template.model_copy(
            update={
                "analysis_frame_index": frame_index,
                "source_seconds": frame_index / 30,
                "objects": (),
                "hands": (),
            }
        )
        for frame_index in range(frame_count)
    )
    return manifest.model_copy(update={"observations": observations})


def test_first_minute_manifest_has_a_consistent_sixty_second_contract() -> None:
    bounded = _first_minute_manifest(_manifest_with_frame_count(FIRST_MINUTE_FRAME_COUNT))

    assert len(bounded.observations) == 1800
    assert bounded.clip.source_duration_seconds == 60.0
    assert bounded.coverage.source_duration_seconds == 60.0
    assert bounded.coverage.ratio == 1.0

    with pytest.raises(ValueError, match=r"\[0, 1800\)"):
        _first_minute_manifest(_manifest_with_frame_count(1799))


def test_first_minute_video_is_cfr_and_reuses_only_a_valid_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "bounded.mp4"
    source.write_bytes(b"source")
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        Path(command[-1]).write_bytes(b"bounded")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("battle.rerun_comparison.subprocess.run", fake_run)
    assert _create_first_minute_video(source, output) == output
    assert "-fps_mode" in commands[0]
    assert "cfr" in commands[0]

    output.touch()
    monkeypatch.setattr(
        "battle.rerun_comparison._video_frame_count",
        lambda _path: FIRST_MINUTE_FRAME_COUNT,
    )
    assert _create_first_minute_video(source, output) == output
    assert len(commands) == 1


def test_static_comparison_manifest_can_attach_aligned_hands() -> None:
    static = _first_minute_manifest(_manifest_with_frame_count(FIRST_MINUTE_FRAME_COUNT))
    template_hand = synthetic_run_manifest().observations[0].hands
    hand_observations = tuple(
        observation.model_copy(update={"hands": template_hand})
        for observation in static.observations
    )
    fingerprint = ArtifactFingerprint(uri="input", sha256="a" * 64, source="measured")
    metadata = MediaPipeHandsRunMetadata(
        requested_analysis_frame_range=FrameRange(
            start_frame=0, end_frame_exclusive=FIRST_MINUTE_FRAME_COUNT
        ),
        requested_seconds=60.0,
        source_fingerprint=fingerprint,
        proxy_fingerprint=fingerprint,
        config_fingerprint=fingerprint,
        model_fingerprint=fingerprint,
        adapter=AdapterMetadata(
            name="test",
            version="1",
            implementation_basis="fixture",
            external_source_uri="https://example.com",
        ),
        runtime_settings={"analysis_fps": 30},
        measurements=RuntimeMeasurements(elapsed_seconds=1),
        observations_uri="observations.jsonl",
    )
    hands = static.model_copy(
        update={"observations": hand_observations, "mediapipe_hands": metadata}
    )

    merged = _merge_static_hands(static, hands)

    assert all(len(observation.hands) == 1 for observation in merged.observations)
    assert len(merged.observations) == FIRST_MINUTE_FRAME_COUNT


def test_view_ids_map_onto_camera_names() -> None:
    assert _camera_name("static-c10379") == "C10379"
    assert _camera_name("ego-hmc21110305") == "HMC_21110305"
    with pytest.raises(ValueError):
        _camera_name("mystery-view")


def test_measured_static_clock_shift_comes_from_the_tracked_clock_rules() -> None:
    shift, note = measured_static_clock_shift(
        Path.cwd(), static_view_id="static-c10379", ego_view_id="ego-hmc21110305"
    )
    assert shift == pytest.approx(9 / 60)
    assert "+9 pose frames" in note and "+4.5 analysis frames" in note
    shift_e4, _ = measured_static_clock_shift(
        Path.cwd(), static_view_id="static-c10379", ego_view_id="ego-hmc21179183"
    )
    assert shift_e4 == pytest.approx(0.150)
