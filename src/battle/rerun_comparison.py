"""Build an inference-free synchronized ego/static Rerun comparison from existing runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from .exporter import export_synchronized_comparison
from .schemas import EncodedAssetInput, G2PreprocessingManifest, RunManifest, VideoProxy

DEFAULT_EGO_RUN = Path(
    "runs/muggledsam-sam3-full-ego-manual-seed-multiplexed-ego-hmc21179183-20260909t224835z"
)
DEFAULT_STATIC_RUN = Path("runs/muggledsam-sam3-g3-full-static-c10379-20260909t030710z")
DEFAULT_OUTPUT_NAME = "ego_manual_seed_vs_static_g3_synchronized_comparison.rrd"
APPROVED_FRAME_COUNT = 5400
APPROVED_DURATION_SECONDS = 180.0


def _load_manifest(run_directory: Path) -> RunManifest:
    manifest_path = run_directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"run manifest is unavailable: {manifest_path}")
    return RunManifest.model_validate_json(manifest_path.read_text())


def _load_proxy(
    *,
    repository_root: Path,
    metadata_config_uri: str,
    view_id: str,
    expected_proxy_uri: str,
    expected_proxy_sha256: str,
) -> VideoProxy:
    config_path = (repository_root / metadata_config_uri).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"approved proxy configuration is unavailable: {config_path}")
    config = G2PreprocessingManifest.model_validate_json(config_path.read_text())
    proxy = next((item for item in config.proxies if item.view_id == view_id), None)
    if proxy is None:
        raise ValueError(f"view {view_id!r} is absent from {config_path}")
    if proxy.proxy_uri != expected_proxy_uri or proxy.checksum_sha256 != expected_proxy_sha256:
        raise ValueError(f"approved proxy fingerprint does not match run metadata for {view_id}")
    return proxy


def _require_approved_pair(ego: RunManifest, static: RunManifest) -> None:
    """Verify both persisted outputs describe the same approved source interval."""
    if ego.full_ego_manual_seed is None:
        raise ValueError("ego run is not a full manual-seed baseline")
    if static.g3_candidate is None:
        raise ValueError("static run is not the approved full static G3 candidate")
    ego_metadata = ego.full_ego_manual_seed
    static_metadata = static.g3_candidate
    if (
        ego_metadata.requested_analysis_frame_range.frame_count != APPROVED_FRAME_COUNT
        or static_metadata.requested_analysis_frame_range.frame_count != APPROVED_FRAME_COUNT
        or ego_metadata.requested_seconds != APPROVED_DURATION_SECONDS
        or static_metadata.requested_seconds != APPROVED_DURATION_SECONDS
    ):
        raise ValueError("comparison requires two approved 180-second / 5,400-frame runs")
    if (
        ego.clip.timing.model_dump(mode="json") != static.clip.timing.model_dump(mode="json")
        or ego.clip.source_duration_seconds != static.clip.source_duration_seconds
    ):
        raise ValueError("comparison runs do not share the same timing contract")


def build_comparison(
    *,
    repository_root: Path,
    ego_run_directory: Path,
    static_run_directory: Path,
    output_path: Path | None = None,
) -> Path:
    """Repackage persisted SAM3 outputs and videos; this function never invokes inference."""
    ego_run_directory = ego_run_directory.resolve()
    static_run_directory = static_run_directory.resolve()
    ego = _load_manifest(ego_run_directory)
    static = _load_manifest(static_run_directory)
    _require_approved_pair(ego, static)
    assert ego.full_ego_manual_seed is not None
    assert static.g3_candidate is not None
    ego_metadata = ego.full_ego_manual_seed
    static_metadata = static.g3_candidate
    ego_proxy = _load_proxy(
        repository_root=repository_root,
        metadata_config_uri=ego_metadata.config_fingerprint.uri,
        view_id=ego_metadata.view_id,
        expected_proxy_uri=ego_metadata.proxy_fingerprint.uri,
        expected_proxy_sha256=ego_metadata.proxy_fingerprint.sha256,
    )
    static_proxy = _load_proxy(
        repository_root=repository_root,
        metadata_config_uri=static_metadata.config_fingerprint.uri,
        view_id=static_metadata.view_id,
        expected_proxy_uri=static_metadata.proxy_fingerprint.uri,
        expected_proxy_sha256=static_metadata.proxy_fingerprint.sha256,
    )
    if (
        ego_proxy.frame_count != APPROVED_FRAME_COUNT
        or static_proxy.frame_count != APPROVED_FRAME_COUNT
        or ego_proxy.fps != 30
        or static_proxy.fps != 30
    ):
        raise ValueError("comparison proxies must be 30-fps, 5,400-frame assets")
    ego_video = ego_run_directory / "input_5400f.mp4"
    static_video = static_run_directory / "input_5400f.mp4"
    if output_path is None:
        output_path = ego_run_directory / DEFAULT_OUTPUT_NAME
    return export_synchronized_comparison(
        ego,
        static,
        output_path,
        ego_video_path=ego_video,
        ego_video_dimensions=(ego_proxy.dimensions.width, ego_proxy.dimensions.height),
        ego_asset_reference=EncodedAssetInput(
            uri=ego_proxy.proxy_uri,
            media_type="video/mp4",
            checksum_sha256=ego_proxy.checksum_sha256,
        ),
        ego_mask_artifact_root=ego_run_directory,
        static_video_path=static_video,
        static_video_dimensions=(static_proxy.dimensions.width, static_proxy.dimensions.height),
        static_asset_reference=EncodedAssetInput(
            uri=static_proxy.proxy_uri,
            media_type="video/mp4",
            checksum_sha256=static_proxy.checksum_sha256,
        ),
        static_mask_artifact_root=static_run_directory,
        static_label="existing G3 SAM3 candidate outputs (not manual-seed)",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an inference-free, analysis-time-synchronized ego/static RRD."
    )
    parser.add_argument("--ego-run", type=Path, default=DEFAULT_EGO_RUN)
    parser.add_argument("--static-run", type=Path, default=DEFAULT_STATIC_RUN)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = build_comparison(
        repository_root=Path.cwd().resolve(),
        ego_run_directory=args.ego_run,
        static_run_directory=args.static_run,
        output_path=args.output,
    )
    print(f"Wrote synchronized ego/static comparison: {output}")


if __name__ == "__main__":
    main()
