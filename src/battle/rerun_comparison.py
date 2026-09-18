"""Build an inference-free synchronized ego/static Rerun comparison from existing runs."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from . import digest_cache, media_probe
from .exporter import export_synchronized_comparison
from .schemas import (
    EncodedAssetInput,
    FullDurationCoverage,
    G2PreprocessingManifest,
    RunManifest,
    TimeInterval,
    VideoProxy,
)

DEFAULT_EGO_RUN = Path(
    "runs/muggledsam-sam3-full-ego-manual-seed-multiplexed-ego-hmc21179183-20260909t224835z"
)
DEFAULT_STATIC_RUN = Path("runs/muggledsam-sam3-g3-full-static-c10379-20260909t030710z")
DEFAULT_OUTPUT_NAME = "ego_manual_seed_vs_static_g3_synchronized_comparison.rrd"
APPROVED_FRAME_COUNT = 5400
APPROVED_DURATION_SECONDS = 180.0
FOCUSED_FRAME_COUNT = 2781
FIRST_MINUTE_FRAME_COUNT = 1800
FIRST_MINUTE_SECONDS = 60.0
DEFAULT_FOCUSED_EGO_RUN = Path(
    "runs/muggledsam-sam3-four-part-ego-focused-reassembly-ego-hmc21110305-20260916t031515z"
)
DEFAULT_FOCUSED_STATIC_RUN = Path(
    "runs/muggledsam-sam3-four-part-static-focused-reassembly-static-c10379-20260916t023700z"
)
DEFAULT_MEDIAPIPE_HANDS_RUN = Path(
    "runs/mediapipe-hands-static-60s-fused-dedup-th035-20260916t0430z"
)
FOCUSED_OUTPUT_NAME = "four_part_focused_first_minute_ego_static_comparison.rrd"


def _load_manifest(run_directory: Path) -> RunManifest:
    manifest_path = run_directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"run manifest is unavailable: {manifest_path}")
    return RunManifest.model_validate_json(manifest_path.read_text())


def _sha256_file(path: Path) -> str:
    return digest_cache.sha256_file(path)


def _relative_uri(path: Path, repository_root: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root).as_posix()
    except ValueError:
        return path.resolve().as_posix()


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


def _require_focused_pair(ego: RunManifest, static: RunManifest) -> None:
    """Verify the selected focused runs cover aligned ego/static source clocks."""
    if ego.four_part_focused is None or static.four_part_focused is None:
        raise ValueError("focused comparison requires two focused four-part runs")
    if (
        ego.four_part_focused.view_id != "ego-hmc21110305"
        or static.four_part_focused.view_id != "static-c10379"
    ):
        raise ValueError("focused comparison requires HMC_21110305 ego and C10379 static views")
    if (
        ego.four_part_focused.requested_analysis_frame_range.frame_count != FOCUSED_FRAME_COUNT
        or static.four_part_focused.requested_analysis_frame_range.frame_count
        != FOCUSED_FRAME_COUNT
        or ego.four_part_focused.requested_seconds != 92.7
        or static.four_part_focused.requested_seconds != 92.7
        or ego.clip.source_duration_seconds != static.clip.source_duration_seconds
        or ego.clip.source_duration_seconds != 92.7
    ):
        raise ValueError("focused comparison requires two aligned 92.7-second source runs")
    ego_clock = ego.clip.timing.model_dump(mode="json")
    static_clock = static.clip.timing.model_dump(mode="json")
    if ego_clock != static_clock:
        raise ValueError("focused comparison runs do not share the same source clock")


def _first_minute_manifest(manifest: RunManifest) -> RunManifest:
    observations = tuple(
        observation
        for observation in manifest.observations
        if observation.analysis_frame_index < FIRST_MINUTE_FRAME_COUNT
    )
    if len(observations) != FIRST_MINUTE_FRAME_COUNT:
        raise ValueError("focused run does not contain every frame in [0, 1800)")
    return RunManifest(
        run_id=f"{manifest.run_id}-first-minute",
        clip=manifest.clip.model_copy(update={"source_duration_seconds": FIRST_MINUTE_SECONDS}),
        coverage=FullDurationCoverage(
            source_duration_seconds=FIRST_MINUTE_SECONDS,
            covered_intervals=(TimeInterval(start_seconds=0.0, end_seconds=FIRST_MINUTE_SECONDS),),
        ),
        chunk_policy=manifest.chunk_policy,
        method_statuses=manifest.method_statuses,
        observations=observations,
    )


def _merge_static_hands(static: RunManifest, hands: RunManifest) -> RunManifest:
    """Attach an aligned MediaPipe hand layer to static SAM3 observations."""
    if hands.mediapipe_hands is None:
        raise ValueError("hand comparison input is not a MediaPipe Hands run")
    metadata = hands.mediapipe_hands
    if (
        metadata.requested_analysis_frame_range.frame_count != FIRST_MINUTE_FRAME_COUNT
        or metadata.requested_seconds != FIRST_MINUTE_SECONDS
        or len(hands.observations) != FIRST_MINUTE_FRAME_COUNT
    ):
        raise ValueError("MediaPipe hand comparison requires exactly 60 seconds / 1,800 frames")
    if static.four_part_focused is not None:
        static_proxy = static.four_part_focused.proxy_fingerprint
        if metadata.proxy_fingerprint != static_proxy:
            raise ValueError("MediaPipe and static SAM3 runs do not share the same proxy")
    if hands.clip.timing.model_dump(mode="json") != static.clip.timing.model_dump(mode="json"):
        raise ValueError("MediaPipe and static SAM3 runs do not share the same timing contract")

    observations = []
    for static_observation, hand_observation in zip(
        static.observations, hands.observations, strict=True
    ):
        if (
            hand_observation.view_id != static_observation.view_id
            or hand_observation.analysis_frame_index != static_observation.analysis_frame_index
            or abs(hand_observation.source_seconds - static_observation.source_seconds) > 1e-6
        ):
            raise ValueError("MediaPipe and static SAM3 observations are not frame-aligned")
        observations.append(
            static_observation.model_copy(update={"hands": hand_observation.hands})
        )
    return static.model_copy(
        update={
            "run_id": f"{static.run_id}--{hands.run_id}",
            "method_statuses": static.method_statuses + hands.method_statuses,
            "observations": tuple(observations),
        }
    )


def _create_first_minute_video(source: Path, output: Path) -> Path:
    if (
        output.is_file()
        and output.stat().st_mtime_ns >= source.stat().st_mtime_ns
        and _video_frame_count(output) == FIRST_MINUTE_FRAME_COUNT
    ):
        return output
    if output.exists():
        output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            f"trim=end_frame={FIRST_MINUTE_FRAME_COUNT},setpts=PTS-STARTPTS",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-fps_mode",
            "cfr",
            "-movflags",
            "+faststart",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or not output.is_file():
        raise RuntimeError(f"could not create first-minute video: {completed.stderr.strip()}")
    return output


def _video_frame_count(path: Path) -> int:
    return media_probe.video_frame_count(path)


def build_focused_first_minute_comparison(
    *,
    repository_root: Path,
    ego_run_directory: Path,
    static_run_directory: Path,
    hands_run_directory: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    """Build the selected first-minute two-view comparison without inference."""
    ego_run_directory = ego_run_directory.resolve()
    static_run_directory = static_run_directory.resolve()
    ego_full = _load_manifest(ego_run_directory)
    static_full = _load_manifest(static_run_directory)
    hands = _load_manifest(hands_run_directory.resolve()) if hands_run_directory else None
    _require_focused_pair(ego_full, static_full)
    assert ego_full.four_part_focused is not None
    assert static_full.four_part_focused is not None
    ego_metadata = ego_full.four_part_focused
    static_metadata = static_full.four_part_focused
    if (
        hands is not None
        and hands.mediapipe_hands is not None
        and hands.mediapipe_hands.proxy_fingerprint != static_metadata.proxy_fingerprint
    ):
        raise ValueError("MediaPipe and static SAM3 runs do not share the same proxy")
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
    if output_path is None:
        output_path = (
            repository_root / "runs/four-part-focused-first-minute-comparison" / FOCUSED_OUTPUT_NAME
        )
    output_path = output_path.resolve()
    ego_video = _create_first_minute_video(
        ego_run_directory / "input_2781f.mp4",
        output_path.parent / "ego_input_1800f.mp4",
    )
    static_video = _create_first_minute_video(
        static_run_directory / "input_2781f.mp4",
        output_path.parent / "static_input_1800f.mp4",
    )
    ego_minute = _first_minute_manifest(ego_full)
    static_minute = _first_minute_manifest(static_full)
    if hands is not None:
        static_minute = _merge_static_hands(static_minute, hands)
    return export_synchronized_comparison(
        ego_minute,
        static_minute,
        output_path,
        ego_video_path=ego_video,
        ego_video_dimensions=(ego_proxy.dimensions.width, ego_proxy.dimensions.height),
        ego_asset_reference=EncodedAssetInput(
            uri=_relative_uri(ego_video, repository_root),
            media_type="video/mp4",
            checksum_sha256=_sha256_file(ego_video),
        ),
        ego_mask_artifact_root=ego_run_directory,
        static_video_path=static_video,
        static_video_dimensions=(static_proxy.dimensions.width, static_proxy.dimensions.height),
        static_asset_reference=EncodedAssetInput(
            uri=_relative_uri(static_video, repository_root),
            media_type="video/mp4",
            checksum_sha256=_sha256_file(static_video),
        ),
        static_mask_artifact_root=static_run_directory,
        static_label=(
            "focused static four-part outputs + MediaPipe Hands; first 60 seconds"
            if hands is not None
            else "focused static four-part outputs; first 60 seconds"
        ),
        ego_label="focused monochrome ego four-part outputs; first 60 seconds",
        ego_description=(
            "First 1,800 frames derived from the fingerprinted focused ego run input."
        ),
        static_description=(
            "First 1,800 frames derived from the fingerprinted focused static run input."
        ),
    )


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
        static_label=(
            "aligned hybrid G3 outputs (three text targets, one reviewed-mask target)"
            if static_metadata.hybrid_initialization is not None
            else "historical G3 SAM3 zero-shot candidate outputs"
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an inference-free, analysis-time-synchronized ego/static RRD."
    )
    parser.add_argument("--ego-run", type=Path)
    parser.add_argument("--static-run", type=Path)
    parser.add_argument("--hands-run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--focused-first-minute",
        action="store_true",
        help="Compare the selected first 1,800 frames of aligned focused four-part runs.",
    )
    args = parser.parse_args()
    if args.focused_first_minute:
        output = build_focused_first_minute_comparison(
            repository_root=Path.cwd().resolve(),
            ego_run_directory=args.ego_run or DEFAULT_FOCUSED_EGO_RUN,
            static_run_directory=args.static_run or DEFAULT_FOCUSED_STATIC_RUN,
            hands_run_directory=args.hands_run or DEFAULT_MEDIAPIPE_HANDS_RUN,
            output_path=args.output,
        )
    else:
        output = build_comparison(
            repository_root=Path.cwd().resolve(),
            ego_run_directory=args.ego_run or DEFAULT_EGO_RUN,
            static_run_directory=args.static_run or DEFAULT_STATIC_RUN,
            output_path=args.output,
        )
    print(f"Wrote synchronized ego/static comparison: {output}")


if __name__ == "__main__":
    main()
