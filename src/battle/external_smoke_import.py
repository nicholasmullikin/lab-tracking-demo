"""Import evidence-only upstream smokes into Battle's normalized/Rerun contract.

These importers perform no model inference.  They preserve upstream output semantics and
write a separate run rather than relabelling an external smoke as an integrated method.

JSON and PKL inputs are read only from caller-controlled local artifact paths under the
repository root. Parsing is structural (schema/shape checks, not a hardened untrusted-data
sandbox) and is intended for trusted local evidence produced by the documented upstream
commands, not for arbitrary third-party uploads.
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from .digest_cache import sha256_file
from .exporter import export_run
from .fs_common import relative_uri
from .schemas import (
    AdapterMetadata,
    ArtifactFingerprint,
    ChunkContinuityPolicy,
    ClockName,
    EncodedAssetInput,
    ExternalPartialRunMetadata,
    FrameObservations,
    FullDurationCoverage,
    G2PreprocessingManifest,
    ImageLandmark2D,
    MaskReference,
    MaskStorage,
    MethodState,
    MethodStatus,
    NormalizedBox,
    PerFrameNlfBody2D,
    PerFrameObject,
    RunManifest,
    RuntimeMeasurements,
    TimeInterval,
)

DEFAULT_CONFIG = Path("configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_g2.json")
DEFAULT_SOURCE_OFFSET_SECONDS = 294.0
DEFAULT_DAM4SAM_NATIVE = Path("runs/dam4sam-static-20s-smoke-20260916t0520z/masks")
DEFAULT_KINEO_NATIVE = Path(
    "runs/kineo/infer_nlf_headless_only/offline_demo/annotations/assembly101_focused_static_20s"
)
DEFAULT_GROUNDED_SAM2_NATIVE = Path(
    "runs/grounded-sam2-static-frame0-smoke-20260916t0450z/grounded_sam2_hf_model_demo_results.json"
)
DEFAULT_SMOKE_VIDEO = Path("data/derived/assembly101/smoke_frames/focused_static_20s.mp4")
DEFAULT_SMOKE_IMAGE = Path("data/derived/assembly101/smoke_frames/focused_static_frame0.jpg")
NLF_BODY_JOINT_COUNT = 55


def _fingerprint(path: Path, repository_root: Path) -> ArtifactFingerprint:
    return ArtifactFingerprint(
        uri=relative_uri(path, repository_root), sha256=sha256_file(path), source="measured"
    )


def _write_observations(path: Path, observations: tuple[FrameObservations, ...]) -> None:
    path.write_text(
        "".join(observation.model_dump_json() + "\n" for observation in observations),
        encoding="utf-8",
    )


def _single_frame_video(image_path: Path, output_path: Path) -> Path:
    """Encode the audited source image once for a spatially aligned, inference-free RRD."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-loop",
            "1",
            "-i",
            str(image_path),
            "-frames:v",
            "1",
            "-r",
            "30",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ],
        check=True,
    )
    return output_path


def _clip_for_duration(config: G2PreprocessingManifest, seconds: float) -> object:
    return config.clip.model_copy(update={"source_duration_seconds": seconds})


def _write_native_index(
    native_paths: list[Path], output_path: Path, repository_root: Path
) -> ArtifactFingerprint:
    payload = [
        {
            "uri": relative_uri(path, repository_root),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in native_paths
    ]
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return _fingerprint(output_path, repository_root)


def _bounded_clip(config: G2PreprocessingManifest, frame_count: int) -> tuple[object, float]:
    analysis_fps = config.proxy_timing.clocks.fps_for(ClockName.ANALYSIS)
    return _clip_for_duration(config, frame_count / analysis_fps), analysis_fps


def _write_imported_run(
    *,
    repository_root: Path,
    run_directory: Path,
    run_id: str,
    config: G2PreprocessingManifest,
    observations: tuple[FrameObservations, ...],
    metadata: ExternalPartialRunMetadata,
    output_name: str,
    source_video: Path | None,
    video_dimensions: tuple[int, int] | None,
    mask_artifact_root: Path | None = None,
) -> Path:
    duration = (
        metadata.decoded_frame_count / config.proxy_timing.clocks.fps_for(ClockName.ANALYSIS)
        if metadata.decoded_frame_count
        else 1 / config.proxy_timing.clocks.fps_for(ClockName.ANALYSIS)
    )
    manifest = RunManifest(
        run_id=run_id,
        clip=_clip_for_duration(config, duration),
        coverage=FullDurationCoverage(
            source_duration_seconds=duration,
            covered_intervals=(TimeInterval(start_seconds=0, end_seconds=duration),),
        ),
        chunk_policy=ChunkContinuityPolicy(
            overlap_seconds=0, max_allowed_gap_seconds=0, preserve_track_ids=False
        ),
        method_statuses=(
            MethodStatus(
                method_name=metadata.adapter.name,
                stage="objects" if observations and observations[0].objects else "pose",
                state=MethodState.SUCCEEDED,
                artifact_uri=metadata.normalized_artifact_uri,
                measured_on=metadata.classification,
            ),
            MethodStatus(
                method_name="rerun-external-smoke-import",
                stage="export",
                state=MethodState.SUCCEEDED,
                artifact_uri=metadata.rerun_artifact_uri,
                measured_on="inference-free import of preserved native artifacts",
            ),
        ),
        observations=observations,
        external_partial=metadata,
    )
    manifest_path = run_directory / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    export_run(
        manifest,
        run_directory / output_name,
        video_path=source_video,
        video_dimensions=video_dimensions,
        asset_reference=EncodedAssetInput(
            uri=relative_uri(source_video, repository_root),
            media_type="video/mp4",
            checksum_sha256=sha256_file(source_video),
        )
        if source_video is not None
        else None,
        mask_artifact_root=mask_artifact_root,
    )
    return manifest_path


def _normalize_box(
    x1: float, y1: float, x2: float, y2: float, width: int, height: int
) -> NormalizedBox:
    left = min(max(x1 / width, 0), 1 - 1e-4)
    top = min(max(y1 / height, 0), 1 - 1e-4)
    right = min(max(x2 / width, left + 1e-4), 1)
    bottom = min(max(y2 / height, top + 1e-4), 1)
    return NormalizedBox(x=left, y=top, width=right - left, height=bottom - top)


def import_grounded_sam2(args: argparse.Namespace) -> Path:
    root = args.repository_root.resolve()
    config = G2PreprocessingManifest.model_validate_json((root / args.config).read_text())
    native_path = (root / args.native).resolve()
    image_path = (root / args.image).resolve()
    payload = json.loads(native_path.read_text(encoding="utf-8"))
    annotations = payload["annotations"]
    if len(annotations) != 1:
        raise ValueError(f"expected one Grounded-SAM-2 annotation, found {len(annotations)}")
    annotation = annotations[0]
    width, height = int(payload["img_width"]), int(payload["img_height"])
    box = _normalize_box(*annotation["bbox"], width, height)
    observation = FrameObservations(
        view_id="static-c10379",
        analysis_frame_index=0,
        source_seconds=DEFAULT_SOURCE_OFFSET_SECONDS,
        objects=(
            PerFrameObject(
                object_id="grounded-sam2-0",
                label=str(annotation["class_name"]),
                confidence=float(annotation["score"]),
                box=box,
            ),
        ),
    )
    run_id = args.run_id or "grounded-sam2-static-frame0-normalized"
    run_directory = (root / args.output_root / run_id).resolve()
    run_directory.mkdir(parents=True, exist_ok=False)
    normalized_path = run_directory / "observations.jsonl"
    _write_observations(normalized_path, (observation,))
    video_path = _single_frame_video(image_path, run_directory / "input_frame.mp4")
    metadata = ExternalPartialRunMetadata(
        classification="single_frame_smoke",
        requested_input_fingerprint=_fingerprint(image_path, root),
        native_artifact_fingerprints=(_fingerprint(native_path, root),),
        adapter=AdapterMetadata(
            name="grounded-sam2-hf-image-import",
            version="0.1.0",
            implementation_basis=(
                "imported Grounding DINO Tiny + SAM2 image-predictor JSON; no video propagation"
            ),
            external_source_uri="/home/nick/src/Grounded-SAM-2",
            external_revision="b7a9c29f196edff0eb54dbe14588d7ae5e3dde28",
        ),
        reproduced_command=(
            "python grounded_sam2_hf_model_demo.py --grounding-model "
            "IDEA-Research/grounding-dino-tiny --text-prompt 'hand.' "
            "--img-path <focused_static_frame0.jpg> --sam2-checkpoint <sam2.1_hiera_tiny.pt>"
        ),
        decoded_frame_count=1,
        frames_with_normalized_output=1,
        source_offset_seconds=DEFAULT_SOURCE_OFFSET_SECONDS,
        normalized_artifact_uri=relative_uri(normalized_path, root),
        rerun_artifact_uri=relative_uri(run_directory / "grounded_sam2.rrd", root),
        limitations=(
            "The Hugging Face detector revision was not pinned in the native smoke.",
            "The imported RRD records the detector box only; the native COCO RLE remains in JSON.",
            "This is image segmentation, not a video-tracking integration.",
        ),
    )
    return _write_imported_run(
        repository_root=root,
        run_directory=run_directory,
        run_id=run_id,
        config=config,
        observations=(observation,),
        metadata=metadata,
        output_name="grounded_sam2.rrd",
        source_video=video_path,
        video_dimensions=(width, height),
    )


def import_dam4sam(args: argparse.Namespace) -> Path:
    root = args.repository_root.resolve()
    config = G2PreprocessingManifest.model_validate_json((root / args.config).read_text())
    masks_root = (root / args.native).resolve()
    source_video = (root / args.video).resolve()
    masks = sorted(masks_root.glob("*.png"))
    if not masks:
        raise FileNotFoundError(f"no PNG masks under {masks_root}")
    dimensions: tuple[int, int] | None = None
    observations: list[FrameObservations] = []
    for index, mask_path in enumerate(masks):
        with Image.open(mask_path) as image:
            mask = np.asarray(image.convert("L")) > 0
        height, width = mask.shape
        dimensions = (width, height)
        ys, xs = np.where(mask)
        objects: tuple[PerFrameObject, ...] = ()
        if len(xs):
            objects = (
                PerFrameObject(
                    object_id="dam4sam-seeded-hand",
                    label="seeded_hand",
                    confidence=1.0,
                    box=_normalize_box(
                        xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, width, height
                    ),
                    mask=MaskReference(
                        uri=f"masks/{mask_path.name}",
                        storage=MaskStorage.EXTERNAL_ARTIFACT,
                        format="png",
                    ),
                ),
            )
        observations.append(
            FrameObservations(
                view_id="static-c10379",
                analysis_frame_index=index,
                source_seconds=DEFAULT_SOURCE_OFFSET_SECONDS + index / 30,
                objects=objects,
            )
        )
    run_id = args.run_id or "dam4sam-static-20s-normalized"
    run_directory = (root / args.output_root / run_id).resolve()
    run_directory.mkdir(parents=True, exist_ok=False)
    normalized_path = run_directory / "observations.jsonl"
    _write_observations(normalized_path, tuple(observations))
    native_index = _write_native_index(masks, run_directory / "native_masks_index.json", root)
    metadata = ExternalPartialRunMetadata(
        classification="external_partial",
        requested_input_fingerprint=_fingerprint(source_video, root),
        native_artifact_fingerprints=(native_index,),
        adapter=AdapterMetadata(
            name="dam4sam-mask-import",
            version="0.1.0",
            implementation_basis="imported headless DAM4SAM bbox-init mask PNGs",
            external_source_uri="/home/nick/src/DAM4SAM",
            external_revision="9c954504b39ebca4c412f207be0787c26bfac85a",
        ),
        reproduced_command=(
            "python <headless DAM4SAM wrapper> --frames <602 decoded JPG frames> "
            "--bbox 881,446,152,129 --output-dir <masks>"
        ),
        decoded_frame_count=len(masks),
        frames_with_normalized_output=sum(
            bool(observation.objects) for observation in observations
        ),
        source_offset_seconds=DEFAULT_SOURCE_OFFSET_SECONDS,
        measurements=RuntimeMeasurements(elapsed_seconds=53.6),
        normalized_artifact_uri=relative_uri(normalized_path, root),
        rerun_artifact_uri=relative_uri(run_directory / "dam4sam.rrd", root),
        limitations=(
            "The upstream interactive runner was bypassed by an unversioned headless wrapper.",
            (
                "Native logs show SAM2 post-processing was skipped because its extension "
                "was ABI-incompatible."
            ),
            "The output is one seeded object; it does not establish DAM4SAM distractor semantics.",
        ),
    )
    return _write_imported_run(
        repository_root=root,
        run_directory=run_directory,
        run_id=run_id,
        config=config,
        observations=tuple(observations),
        metadata=metadata,
        output_name="dam4sam.rrd",
        source_video=source_video,
        video_dimensions=dimensions,
        mask_artifact_root=masks_root.parent,
    )


def _nlf_body_landmarks(
    keypoint_row: dict[str, object], joint_names: list[str], width: int, height: int
) -> PerFrameNlfBody2D:
    xy = keypoint_row["xy"][:NLF_BODY_JOINT_COUNT]
    scores = keypoint_row["scores"][:NLF_BODY_JOINT_COUNT]
    landmarks = tuple(
        ImageLandmark2D(
            name=joint_names[index],
            x=min(max(float(point[0]) / width, 0.0), 1.0),
            y=min(max(float(point[1]) / height, 0.0), 1.0),
            confidence=float(scores[index]),
        )
        for index, point in enumerate(xy)
    )
    return PerFrameNlfBody2D(
        subject_id=str(keypoint_row["subject_id"]),
        landmarks=landmarks,
    )


def import_kineo(args: argparse.Namespace) -> Path:
    root = args.repository_root.resolve()
    config = G2PreprocessingManifest.model_validate_json((root / args.config).read_text())
    native_root = (root / args.native).resolve()
    source_video = (root / args.video).resolve()
    bboxes_path = native_root / "bboxes_2d.pkl"
    keypoints_path = native_root / "keypoints_2d.pkl"
    timings_path = native_root / "stage_timings.pkl"
    intrinsics_path = native_root / "camera_intrinsics.pkl"
    bbox_payload = pickle.loads(bboxes_path.read_bytes())
    keypoint_payload = pickle.loads(keypoints_path.read_bytes())
    timings = pickle.loads(timings_path.read_bytes())["annotations"]
    bboxes = bbox_payload["annotations"]
    keypoints = keypoint_payload["annotations"]
    joint_names = keypoint_payload["metadata"]["formats"][0]["keypoints_names"][
        :NLF_BODY_JOINT_COUNT
    ]
    if len(bboxes) != len(keypoints):
        raise ValueError("Kineo bbox/keypoint annotation counts differ")
    by_frame_objects: dict[int, list[PerFrameObject]] = {}
    by_frame_body: dict[int, list[PerFrameNlfBody2D]] = {}
    for bbox, keypoint in zip(bboxes, keypoints, strict=True):
        frame_idx = int(bbox["frame_idx"])
        x1, y1, x2, y2 = bbox["xyxy"]
        by_frame_objects.setdefault(frame_idx, []).append(
            PerFrameObject(
                object_id=f"kineo-{bbox['subject_id']}",
                label="person_nlf_bbox",
                confidence=float(bbox["score"]),
                box=_normalize_box(x1, y1, x2, y2, 1280, 720),
            )
        )
        by_frame_body.setdefault(frame_idx, []).append(
            _nlf_body_landmarks(keypoint, joint_names, 1280, 720)
        )
    frame_count = max(by_frame_objects) + 1 if by_frame_objects else 0
    observations = tuple(
        FrameObservations(
            view_id="static-c10379",
            analysis_frame_index=index,
            source_seconds=DEFAULT_SOURCE_OFFSET_SECONDS + index / 30,
            objects=tuple(by_frame_objects.get(index, ())),
            nlf_body_2d=tuple(by_frame_body.get(index, ())),
        )
        for index in range(frame_count)
    )
    run_id = args.run_id or "kineo-nlf-static-20s-normalized"
    run_directory = (root / args.output_root / run_id).resolve()
    run_directory.mkdir(parents=True, exist_ok=False)
    normalized_path = run_directory / "observations.jsonl"
    _write_observations(normalized_path, observations)
    native_paths = [bboxes_path, keypoints_path, timings_path]
    if intrinsics_path.is_file():
        native_paths.append(intrinsics_path)
    native_index = _write_native_index(native_paths, run_directory / "native_pkls_index.json", root)
    stage_seconds = sum(float(item["duration_seconds"]) for item in timings)
    metadata = ExternalPartialRunMetadata(
        classification="kineo_nlf_only_partial",
        requested_input_fingerprint=_fingerprint(source_video, root),
        native_artifact_fingerprints=(native_index,),
        adapter=AdapterMetadata(
            name="kineo-nlf-pkl-import",
            version="0.2.0",
            implementation_basis=(
                "imported Kineo NLF 2D bbox and first 55 SMPL-X body joints in image pixels"
            ),
            external_source_uri="/home/nick/src/kineo",
            external_revision="03b36e31c79bd40bc8bb1ce4c9c08907140952dd",
        ),
        reproduced_command=(
            "uv run battle-kineo-nlf --seconds 20 "
            "--kineo-config configs/kineo_nlf_headless_only.yaml"
        ),
        decoded_frame_count=frame_count,
        frames_with_normalized_output=len(by_frame_objects),
        source_offset_seconds=DEFAULT_SOURCE_OFFSET_SECONDS,
        measurements=RuntimeMeasurements(elapsed_seconds=stage_seconds),
        normalized_artifact_uri=relative_uri(normalized_path, root),
        rerun_artifact_uri=relative_uri(run_directory / "kineo_nlf_partial.rrd", root),
        limitations=(
            f"{len(by_frame_objects)} of {frame_count} frames contain person bbox/body joints.",
            "Body joints are NLF SMPL-X image pixels, not Battle hand landmarks or metric 3D.",
            "MoGe intrinsics are estimated and retained only in native PKL; no world metric pose.",
            "No SfM, BVH, or multi-view Kineo pipeline output was produced.",
        ),
    )
    return _write_imported_run(
        repository_root=root,
        run_directory=run_directory,
        run_id=run_id,
        config=config,
        observations=observations,
        metadata=metadata,
        output_name="kineo_nlf_partial.rrd",
        source_video=source_video,
        video_dimensions=(1280, 720),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("grounded-sam2", "dam4sam", "kineo"))
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--run-id")
    parser.add_argument("--native", type=Path)
    parser.add_argument("--video", type=Path, default=DEFAULT_SMOKE_VIDEO)
    parser.add_argument("--image", type=Path, default=DEFAULT_SMOKE_IMAGE)
    args = parser.parse_args()
    defaults = {
        "grounded-sam2": DEFAULT_GROUNDED_SAM2_NATIVE,
        "dam4sam": DEFAULT_DAM4SAM_NATIVE,
        "kineo": DEFAULT_KINEO_NATIVE,
    }
    args.native = args.native or defaults[args.kind]
    handler = {
        "grounded-sam2": import_grounded_sam2,
        "dam4sam": import_dam4sam,
        "kineo": import_kineo,
    }[args.kind]
    print(handler(args))


if __name__ == "__main__":
    main()
