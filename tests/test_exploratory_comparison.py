from __future__ import annotations

from pathlib import Path

import pytest

from battle.exploratory_comparison import (
    LoadedMethod,
    MethodSpec,
    _log_method_frame,
    _method_coverage,
    output_paths,
    validate_artifact_fingerprint,
    validate_observation_timestamps,
)
from battle.fixtures import synthetic_run_manifest
from battle.schemas import ArtifactFingerprint


def test_changed_proxy_fingerprint_is_rejected(tmp_path: Path) -> None:
    proxy = tmp_path / "proxy.mp4"
    proxy.write_bytes(b"synthetic proxy")

    with pytest.raises(ValueError, match="proxy fingerprint mismatch"):
        validate_artifact_fingerprint(
            ArtifactFingerprint(uri="proxy.mp4", sha256="a" * 64, source="measured"),
            tmp_path,
            label="proxy",
        )


def test_mismatched_source_timestamp_is_rejected() -> None:
    manifest = synthetic_run_manifest()
    observation = manifest.observations[0].model_copy(update={"source_seconds": 99.0})

    with pytest.raises(ValueError, match="source timestamp mismatch"):
        validate_observation_timestamps(manifest, {0: observation}, frame_count=600)


def test_partial_coverage_does_not_fill_missing_observations() -> None:
    observation = synthetic_run_manifest().observations[0]

    assert _method_coverage({0: observation}, 0) == (1.0, 1.0)
    assert _method_coverage({0: observation}, 1) == (0.0, 0.0)


def test_missing_method_frame_clears_render_entities(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = synthetic_run_manifest()
    spec = MethodSpec(
        method_id="mediapipe",
        display_name="fixture",
        run_directory=Path("unused"),
        state="succeeded",
        coordinate_semantics=("normalized image coordinates",),
        comparability_limits=("fixture only",),
    )
    loaded = LoadedMethod(spec, tmp_path, manifest, {}, object())  # type: ignore[arg-type]
    logged: list[tuple[str, object]] = []
    monkeypatch.setattr(
        "battle.exploratory_comparison.rr.log", lambda path, value: logged.append((path, value))
    )

    _log_method_frame(loaded, 0, dimensions=(16, 8))

    assert logged[0][0].endswith("/metrics/processed")
    assert logged[1][0].endswith("/metrics/output_present")
    assert logged[2][0].endswith("/render")
    assert logged[2][1].__class__.__name__ == "Clear"


def test_output_paths_stay_under_ignored_runs_directory(tmp_path: Path) -> None:
    recording, index = output_paths(tmp_path)

    assert (
        recording
        == tmp_path / "runs/exploratory-first-20s-comparison/exploratory_first_20s_comparison.rrd"
    )
    assert index.parent == recording.parent
