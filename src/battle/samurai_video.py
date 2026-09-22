"""Run bounded SAMURAI SAM2 video propagation on the approved static proxy."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .cli_common import add_output_root, add_repository_root
from .fs_common import relative_uri, run_timestamp
from .observations import rebuild_tracker_observations as _load_observations
from .schemas import (
    AdapterMetadata,
    ArtifactFingerprint,
    ClockName,
    FrameObservations,
    G2PreprocessingManifest,
    MethodState,
    NormalizedBox,
    RunManifest,
    SamuraiVideoRunMetadata,
    VideoProxy,
)
from .video_driver import (
    assemble_run_manifest,
    bounded_video,
    common_metadata_fields,
    export_manifest,
    load_worker_observations,
    run_external_worker,
    tracker_contact_sheet,
    verify_inputs,
    worker_measurements,
    worker_runtime_settings,
    worker_status,
)

DEFAULT_CONFIG = Path("configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_g2.json")
DEFAULT_VIEW_ID = "static-c10379"
DEFAULT_SECONDS = 10.0
MIN_SECONDS = 1.0
MAX_SECONDS = 20.0
SAMURAI_PYTHON = Path("/home/nick/.pyenv/versions/samurai/bin/python")
SAMURAI_ROOT = Path("/home/nick/src/samurai")
SAMURAI_REVISION = "76ba195984892b0d1e3db5d9c90bb62175680a"
SAM2_CHECKPOINT = SAMURAI_ROOT / "sam2" / "checkpoints" / "sam2.1_hiera_tiny.pt"
SAM2_CHECKPOINT_SHA256 = "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69"
SAM2_CONFIG = "configs/samurai/sam2.1_hiera_t.yaml"
INIT_BBOX_XYWH = (881, 446, 152, 129)
METHOD_NAME = "samurai_sam2_video_smoke"
RERUN_NAME = "samurai.rrd"


def _normalize_box(
    x1: float, y1: float, x2: float, y2: float, width: int, height: int
) -> NormalizedBox:
    left = min(max(x1 / width, 0.0), 1.0)
    top = min(max(y1 / height, 0.0), 1.0)
    right = min(max(x2 / width, 0.0), 1.0)
    bottom = min(max(y2 / height, 0.0), 1.0)
    box_width = min(max(right - left, 1e-4), 1.0 - left)
    box_height = min(max(bottom - top, 1e-4), 1.0 - top)
    return NormalizedBox(x=left, y=top, width=box_width, height=box_height)


def _verify_inputs(
    *,
    repository_root: Path,
    config_path: Path,
    view_id: str,
    seconds: float,
) -> tuple[G2PreprocessingManifest, VideoProxy, Path, int]:
    return verify_inputs(
        repository_root=repository_root,
        config_path=config_path,
        view_id=view_id,
        seconds=seconds,
        min_seconds=MIN_SECONDS,
        max_seconds=MAX_SECONDS,
        checkpoints=((SAM2_CHECKPOINT, SAM2_CHECKPOINT_SHA256, "SAM2 checkpoint"),),
        min_frames=30,
    )


def _run_worker(
    *,
    worker_path: Path,
    run_directory: Path,
    proxy_path: Path,
    view_id: str,
    source_offset_seconds: float,
    max_frames: int,
    analysis_fps: float,
) -> dict[str, object]:
    argv = [
        str(worker_path.resolve()),
        "--run-directory",
        str(run_directory.resolve()),
        "--video",
        str(proxy_path.resolve()),
        "--view-id",
        view_id,
        "--source-offset-seconds",
        str(source_offset_seconds),
        "--max-frames",
        str(max_frames),
        "--analysis-fps",
        str(analysis_fps),
        "--sam2-checkpoint",
        str(SAM2_CHECKPOINT.resolve()),
        "--samurai-root",
        str(SAMURAI_ROOT.resolve()),
    ]
    return run_external_worker(
        SAMURAI_PYTHON, argv, run_directory=run_directory, cwd=SAMURAI_ROOT, record_command=True
    )


def _build_manifest(
    *,
    run_id: str,
    repository_root: Path,
    run_directory: Path,
    config: G2PreprocessingManifest,
    config_path: Path,
    proxy: VideoProxy,
    view_id: str,
    seconds: float,
    requested_frames: int,
    worker_result: Mapping[str, Any],
    observations: tuple[FrameObservations, ...],
    observations_path: Path,
    qa_path: Path,
) -> tuple[RunManifest, MethodState]:
    rerun_path = run_directory / RERUN_NAME
    metadata = SamuraiVideoRunMetadata(
        **common_metadata_fields(
            proxy=proxy,
            config_path=config_path,
            repository_root=repository_root,
            requested_frames=requested_frames,
            seconds=seconds,
            observations_path=observations_path,
            rerun_path=rerun_path,
            qa_path=qa_path,
        ),
        sam2_checkpoint_fingerprint=ArtifactFingerprint(
            uri=str(SAM2_CHECKPOINT),
            sha256=SAM2_CHECKPOINT_SHA256,
            source="measured",
        ),
        adapter=AdapterMetadata(
            name=METHOD_NAME,
            version="0.1.0",
            implementation_basis=(
                "SAMURAI SAM2.1 tiny video predictor with samurai_mode config and "
                "deterministic frame-0 hand bbox seed"
            ),
            external_source_uri=str(SAMURAI_ROOT),
            external_revision=SAMURAI_REVISION,
        ),
        runtime_settings=worker_runtime_settings(
            worker_result,
            analysis_fps=proxy.fps,
            extra_keys=("frames_with_masks", "unique_mask_hashes"),
        ),
        measurements=worker_measurements(
            worker_result, known_unavailable_measures=("ground-truth mask quality",)
        ),
        native_masks_uri=relative_uri(run_directory / "native" / "masks", repository_root),
        sam2_model_config=SAM2_CONFIG,
        init_bbox_xywh=INIT_BBOX_XYWH,
        initialization_note=(
            "Frame 0 is initialized from the approved focused static hand box "
            f"{INIT_BBOX_XYWH} (xywh); SAMURAI config sets samurai_mode=true."
        ),
    )
    state, blocker = worker_status(worker_result)
    manifest = assemble_run_manifest(
        run_id=run_id,
        config=config,
        seconds=seconds,
        view_id=view_id,
        repository_root=repository_root,
        observations=observations,
        observations_path=observations_path,
        rerun_path=rerun_path,
        method_name=METHOD_NAME,
        stage="objects",
        export_method_name="rerun-samurai-export",
        export_measured_on="normalized mask observations; bounded input video logged once",
        state=state,
        blocker=blocker,
        samurai_video=metadata,
    )
    return manifest, state


def run(args: argparse.Namespace) -> Path:
    repository_root = args.repository_root.resolve()
    config_path = (repository_root / args.config).resolve()
    config, proxy, proxy_path, requested_frames = _verify_inputs(
        repository_root=repository_root,
        config_path=config_path,
        view_id=args.view,
        seconds=args.seconds,
    )
    run_id = args.run_id or f"{METHOD_NAME}-{args.seconds:g}s-{run_timestamp()}"
    run_directory = (repository_root / args.output_root / run_id).resolve()
    run_directory.mkdir(parents=True, exist_ok=False)
    worker_path = Path(__file__).with_name("samurai_video_worker.py")
    worker_result = _run_worker(
        worker_path=worker_path,
        run_directory=run_directory,
        proxy_path=proxy_path,
        view_id=args.view,
        source_offset_seconds=config.proxy_timing.source_seconds_for_frame(ClockName.ANALYSIS, 0),
        max_frames=requested_frames,
        analysis_fps=float(proxy.fps),
    )
    observations_path, observations = load_worker_observations(
        run_directory, worker_result, requested_frames, loader=_load_observations, exact=True
    )
    video_path = bounded_video(proxy_path, run_directory / "input.mp4", requested_frames)
    qa_path = tracker_contact_sheet(
        video_path=video_path,
        observations=observations,
        output_path=run_directory / "contact_sheet.png",
    )
    manifest, state = _build_manifest(
        run_id=run_id,
        repository_root=repository_root,
        run_directory=run_directory,
        config=config,
        config_path=config_path,
        proxy=proxy,
        view_id=args.view,
        seconds=args.seconds,
        requested_frames=requested_frames,
        worker_result=worker_result,
        observations=observations,
        observations_path=observations_path,
        qa_path=qa_path,
    )
    manifest_path = run_directory / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n")
    if state is MethodState.SUCCEEDED:
        export_manifest(
            manifest,
            run_directory / RERUN_NAME,
            video_path=video_path,
            proxy=proxy,
            mask_artifact_root=run_directory,
        )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_repository_root(parser)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--view", default=DEFAULT_VIEW_ID)
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    add_output_root(parser, Path("runs"))
    parser.add_argument("--run-id")
    args = parser.parse_args()
    print(run(args))


if __name__ == "__main__":
    main()
