"""Compose bounded exploratory outputs into one source-aligned Rerun recording.

This module only reads normalized artifacts already produced by method-specific runners.  It
does not import, initialize, or invoke any model package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from PIL import Image

from .build_phases import PhaseTimer
from .digest_cache import sha256_file
from .exporter import HAND_CONNECTIONS, HAND_LANDMARK_NAMES, _rgba_mask_png
from .schemas import (
    ArtifactFingerprint,
    ClockName,
    ExploratoryComparisonIndexManifest,
    ExploratoryComparisonMethod,
    ExploratoryTimelineAlignment,
    FrameObservations,
    RunManifest,
    TimeInterval,
)

COMPARISON_ID = "exploratory-first-20s-comparison"
FRAME_COUNT = 600
ANALYSIS_FPS = 30
OUTPUT_ROOT = Path("runs/exploratory-first-20s-comparison")
OUTPUT_NAME = "exploratory_first_20s_comparison.rrd"
INDEX_NAME = "exploratory_comparison_index.json"
METHOD_COLORS: dict[str, tuple[int, int, int]] = {
    "mediapipe": (65, 169, 245),
    "wilor": (255, 152, 67),
    "boxmot": (185, 110, 255),
    "grounding_dino_sam2": (255, 90, 90),
    "samurai": (70, 210, 165),
    "dam4sam": (255, 215, 70),
    "kineo_nlf": (140, 220, 155),
}


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    display_name: str
    run_directory: Path
    state: str
    coordinate_semantics: tuple[str, ...]
    comparability_limits: tuple[str, ...]
    default_visible: bool = False
    alignment_artifact: str | None = None


DEFAULT_METHODS = (
    MethodSpec(
        "mediapipe",
        "MediaPipe Hands",
        Path("runs/mediapipe-hands-static-20s-fused-dedup-th035-20260916t0428z"),
        "succeeded",
        ("normalized 2D image coordinates; unmirrored static RGB",),
        ("Frame-local detections; no hand identity equivalence or accuracy claim.",),
        True,
    ),
    MethodSpec(
        "wilor",
        "WiLoR Hands",
        Path("runs/wilor-hands-static-20s-audited-source-state"),
        "external_partial",
        (
            "normalized 2D image coordinates",
            "separate camera-relative non-metric 3D; never world-aligned",
        ),
        (
            "The upstream checkout was dirty; base revision does not fully reproduce it.",
            "Frame-local hand IDs and non-metric 3D must not be compared across methods.",
        ),
    ),
    MethodSpec(
        "boxmot",
        "BoxMOT over YOLO detections",
        Path("runs/boxmot-yolo-static-20s-20260916t0445z"),
        "succeeded",
        ("normalized 2D image coordinates",),
        ("Track IDs are detector-conditioned COCO-person associations only.",),
        True,
    ),
    MethodSpec(
        "grounding_dino_sam2",
        "Grounding-DINO + SAM2",
        Path("runs/transformers_grounding_dino_plus_sam2_video_smoke-10s-20260916t0518z"),
        "smoke_only",
        ("normalized 2D image coordinates; propagated native masks",),
        ("Only frames 0–299 / first 10 seconds were run.",),
    ),
    MethodSpec(
        "samurai",
        "SAMURAI",
        Path("runs/samurai_sam2_video_smoke-20s-20260916t052348z"),
        "integrated_smoke",
        ("normalized 2D image coordinates; propagated native masks",),
        ("One deterministic frame-zero hand-box seed; not distractor-scene validation.",),
    ),
    MethodSpec(
        "dam4sam",
        "DAM4SAM",
        Path("runs/dam4sam_video_smoke-20s-20260916t052447z"),
        "integrated_smoke",
        ("normalized 2D image coordinates; propagated native masks",),
        ("One deterministic frame-zero hand-box seed; not distractor-scene validation.",),
    ),
    MethodSpec(
        "kineo_nlf",
        "Kineo NLF-only partial",
        Path("runs/kineo-nlf-headless-20s-20260916t0540z"),
        "external_partial",
        ("NLF normalized top-left image coordinates for person boxes and body joints",),
        (
            "NLF body joints are not hand landmarks.",
            "No SfM, BVH, multi-view output, or verified metric/world 3D is shown.",
        ),
    ),
    MethodSpec(
        "drop_dtw",
        "CLIP + Drop-DTW",
        Path("runs/drop-dtw-static-20s-pinned-openclip-rerun"),
        "integrated_smoke",
        ("temporal alignment only; no spatial geometry",),
        (
            "Assembly101 coarse GT transcript is weak supervision, not a prediction target.",
            "Intervals and cost are not temporal-action accuracy claims.",
        ),
        alignment_artifact="alignment.json",
    ),
)
ATHENA_FIXTURE = Path("runs/athena-fixture-triangulation-smoke")


@dataclass(frozen=True)
class LoadedMethod:
    spec: MethodSpec
    run_directory: Path
    manifest: RunManifest
    observations: dict[int, FrameObservations]
    index_method: ExploratoryComparisonMethod


def _relative_uri(path: Path, repository_root: Path) -> str:
    return path.resolve().relative_to(repository_root.resolve()).as_posix()


def _file_fingerprint(path: Path, repository_root: Path) -> ArtifactFingerprint:
    if not path.is_file():
        raise FileNotFoundError(f"comparison input is unavailable: {path}")
    return ArtifactFingerprint(
        uri=_relative_uri(path, repository_root),
        sha256=sha256_file(path),
        source="measured",
    )


def _mask_tree_fingerprint(
    path: Path,
    repository_root: Path,
    referenced_uris: tuple[str, ...],
    *,
    verify: bool = False,
) -> ArtifactFingerprint:
    """Fingerprint the masks this comparison reads, not every file beside them.

    Earlier revisions hashed the whole `native/masks` tree, which charged the build for
    caches and contact sheets that no observation references. The digest now covers the
    referenced URIs in sorted order, so it still changes whenever a drawn mask changes.
    """
    if not path.is_dir():
        raise FileNotFoundError(f"referenced mask directory is unavailable: {path}")
    digest = hashlib.sha256()
    run_directory = path.parent.parent
    for uri in sorted(set(referenced_uris)):
        child = (run_directory / uri).resolve()
        if not child.is_file():
            raise FileNotFoundError(f"referenced mask is unavailable: {child}")
        digest.update(_relative_uri(child, repository_root).encode())
        digest.update(sha256_file(child, verify=verify).encode())
    return ArtifactFingerprint(
        uri=_relative_uri(path, repository_root),
        sha256=digest.hexdigest(),
        source="measured",
    )


def validate_artifact_fingerprint(
    fingerprint: ArtifactFingerprint, repository_root: Path, *, label: str, verify: bool = False
) -> Path:
    """Refuse a changed local source, proxy, config, or artifact before composition.

    `verify` re-reads the bytes instead of trusting a digest cached against the file's
    size and modification time.
    """
    if "://" in fingerprint.uri:
        return Path(fingerprint.uri)
    path = (repository_root / fingerprint.uri).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is unavailable: {path}")
    actual = sha256_file(path, verify=verify)
    if actual != fingerprint.sha256:
        raise ValueError(
            f"{label} fingerprint mismatch for {fingerprint.uri}: "
            f"expected {fingerprint.sha256}, got {actual}"
        )
    return path


def _metadata(manifest: RunManifest) -> Any | None:
    for name in (
        "mediapipe_hands",
        "wilor_hands",
        "boxmot",
        "drop_dtw",
        "grounding_dino_sam2_video",
        "samurai_video",
        "dam4sam_video",
        "external_partial",
    ):
        metadata = getattr(manifest, name)
        if metadata is not None:
            return metadata
    return None


def _metadata_fingerprints(metadata: Any) -> tuple[ArtifactFingerprint, ...]:
    """Return local source/proxy/config inputs the compositor actually validates.

    Checkpoint and detector fingerprints document inference provenance, but this builder never
    opens those weights. Requiring their local cache would make an otherwise inference-free
    composition irreproducible on a review-only checkout.
    """
    names = (
        "source_fingerprint",
        "proxy_fingerprint",
        "config_fingerprint",
        "requested_input_fingerprint",
    )
    return tuple(
        value
        for name in names
        if isinstance((value := getattr(metadata, name, None)), ArtifactFingerprint)
    )


def _declared_inference_fingerprints(metadata: Any) -> tuple[ArtifactFingerprint, ...]:
    """Preserve every fingerprint declared by the source manifest without loading model caches."""
    names = (
        "source_fingerprint",
        "proxy_fingerprint",
        "config_fingerprint",
        "requested_input_fingerprint",
        "transcript_fingerprint",
        "openclip_checkpoint_fingerprint",
        "detector_fingerprint",
        "checkpoint_fingerprint",
        "grounding_model_fingerprint",
        "sam2_checkpoint_fingerprint",
        "dam4sam_config_fingerprint",
    )
    return tuple(
        value
        for name in names
        if isinstance((value := getattr(metadata, name, None)), ArtifactFingerprint)
    )


def _observation_artifact(metadata: Any, run_directory: Path, repository_root: Path) -> Path | None:
    for name in ("observations_uri", "normalized_artifact_uri", "alignment_uri"):
        if (uri := getattr(metadata, name, None)) and "://" not in uri:
            path = (repository_root / uri).resolve()
            if path.is_file():
                return path
    fallback = run_directory / "observations.jsonl"
    return fallback if fallback.is_file() else None


def validate_observation_timestamps(
    manifest: RunManifest, observations: dict[int, FrameObservations], *, frame_count: int
) -> None:
    """Check each retained observation maps exactly to the declared source-time clock."""
    if manifest.clip.timing.clocks.fps_for(ClockName.ANALYSIS) != ANALYSIS_FPS:
        raise ValueError("exploratory comparison requires a 30-fps analysis clock")
    for frame_index, observation in observations.items():
        if not 0 <= frame_index < frame_count:
            raise ValueError(f"observation frame {frame_index} lies outside the bounded comparison")
        expected = manifest.clip.timing.source_seconds_for_frame(ClockName.ANALYSIS, frame_index)
        if abs(observation.source_seconds - expected) > 1e-6:
            raise ValueError(
                f"source timestamp mismatch at analysis frame {frame_index}: "
                f"expected {expected}, got {observation.source_seconds}"
            )


def _load_observations(manifest: RunManifest, *, frame_count: int) -> dict[int, FrameObservations]:
    observations: dict[int, FrameObservations] = {}
    for observation in manifest.observations:
        if observation.analysis_frame_index >= frame_count:
            continue
        if observation.analysis_frame_index in observations:
            raise ValueError("comparison input contains duplicate analysis frame observations")
        observations[observation.analysis_frame_index] = observation
    validate_observation_timestamps(manifest, observations, frame_count=frame_count)
    return observations


def _method_coverage(
    observations: dict[int, FrameObservations], frame_index: int
) -> tuple[float, float]:
    """Expose processing coverage separately from output presence."""
    observation = observations.get(frame_index)
    if observation is None:
        return 0.0, 0.0
    has_output = bool(observation.objects or observation.hands or observation.nlf_body_2d)
    return 1.0, float(has_output)


def _load_method(spec: MethodSpec, *, repository_root: Path, frame_count: int) -> LoadedMethod:
    run_directory = (repository_root / spec.run_directory).resolve()
    manifest_path = run_directory / "manifest.json"
    manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    metadata = _metadata(manifest)
    if metadata is None:
        raise ValueError(f"{spec.method_id} run has no recognized method metadata")
    if manifest.clip.views != ("static-c10379",):
        raise ValueError(f"{spec.method_id} does not target the approved static RGB view")
    metadata_fingerprints = _metadata_fingerprints(metadata)
    for fingerprint in metadata_fingerprints:
        if fingerprint.uri.startswith("hf://"):
            continue
        validate_artifact_fingerprint(
            fingerprint, repository_root, label=f"{spec.method_id} declared input"
        )
    observations = _load_observations(manifest, frame_count=frame_count)
    input_artifacts = [_file_fingerprint(manifest_path, repository_root)]
    for fingerprint in metadata_fingerprints:
        if fingerprint not in input_artifacts:
            input_artifacts.append(fingerprint)
    if (artifact := _observation_artifact(metadata, run_directory, repository_root)) is not None:
        input_artifacts.append(_file_fingerprint(artifact, repository_root))
    if spec.alignment_artifact:
        input_artifacts.append(
            _file_fingerprint(run_directory / spec.alignment_artifact, repository_root)
        )
    referenced_masks = tuple(
        item.mask.uri
        for observation in observations.values()
        for item in observation.objects
        if item.mask
    )
    if referenced_masks:
        input_artifacts.append(
            _mask_tree_fingerprint(
                run_directory / "native" / "masks", repository_root, referenced_masks
            )
        )
    source_offset = manifest.clip.timing.source_seconds_for_frame(ClockName.ANALYSIS, 0)
    method = ExploratoryComparisonMethod(
        method_id=spec.method_id,
        display_name=spec.display_name,
        state=spec.state,
        run_directory_uri=_relative_uri(run_directory, repository_root),
        input_manifest=input_artifacts[0],
        input_artifacts=tuple(input_artifacts[1:] or input_artifacts),
        inference_input_fingerprints=_declared_inference_fingerprints(metadata),
        view_id="static-c10379",
        analysis_fps=ANALYSIS_FPS,
        source_offset_seconds=source_offset,
        coverage=manifest.coverage,
        coordinate_semantics=spec.coordinate_semantics,
        comparability_limits=spec.comparability_limits,
        timeline_alignment=ExploratoryTimelineAlignment.SOURCE_ALIGNED,
        default_visible=spec.default_visible,
    )
    return LoadedMethod(spec, run_directory, manifest, observations, method)


def _video_info(video_path: Path) -> tuple[int, int, tuple[int, int]]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(completed.stdout)["streams"][0]
    numerator, denominator = stream["avg_frame_rate"].split("/")
    fps = round(int(numerator) / int(denominator))
    return int(stream["nb_read_frames"]), fps, (int(stream["width"]), int(stream["height"]))


def _validate_shared_contract(methods: list[LoadedMethod]) -> ArtifactFingerprint:
    first = methods[0].manifest
    for loaded in methods:
        if loaded.manifest.clip.clip_id != first.clip.clip_id:
            raise ValueError("all comparison inputs must name the same clip")
        if loaded.manifest.clip.timing.model_dump(mode="json") != first.clip.timing.model_dump(
            mode="json"
        ):
            raise ValueError("comparison inputs must share the same source clock mapping")
    metadata = _metadata(first)
    assert metadata is not None
    source = getattr(metadata, "source_fingerprint", None)
    if not isinstance(source, ArtifactFingerprint):
        raise ValueError("first comparison input lacks a declared source fingerprint")
    for loaded in methods:
        candidate = getattr(_metadata(loaded.manifest), "source_fingerprint", None)
        if isinstance(candidate, ArtifactFingerprint) and candidate != source:
            raise ValueError("comparison inputs do not share the same raw source fingerprint")
    return source


def _load_athena_metadata(
    repository_root: Path,
) -> ExploratoryComparisonMethod:
    fixture = (repository_root / ATHENA_FIXTURE).resolve()
    manifest = fixture / "external_partial.json"
    artifact = fixture / "athena_fixture.rrd"
    return ExploratoryComparisonMethod(
        method_id="athena",
        display_name="ATHENA multi-view hand triangulation",
        state="blocked",
        run_directory_uri=_relative_uri(fixture, repository_root),
        input_manifest=_file_fingerprint(manifest, repository_root),
        input_artifacts=(_file_fingerprint(artifact, repository_root),),
        coordinate_semantics=("synthetic fixture coordinate system only",),
        comparability_limits=(
            "Assembly101 lacks the required per-camera intrinsics.",
            "Fixture is intentionally not overlaid on the real static RGB timeline.",
        ),
        timeline_alignment=ExploratoryTimelineAlignment.METADATA_ONLY,
    )


def _box_data(
    observation: FrameObservations, dimensions: tuple[int, int]
) -> tuple[list[list[float]], list[list[float]], list[str]]:
    width, height = dimensions
    return (
        [[item.box.x * width, item.box.y * height] for item in observation.objects],
        [[item.box.width * width, item.box.height * height] for item in observation.objects],
        [f"{item.label} ({item.object_id}) {item.confidence:.2f}" for item in observation.objects],
    )


def _log_hands(
    root: str,
    observation: FrameObservations,
    *,
    dimensions: tuple[int, int],
    color: tuple[int, int, int],
    include_3d: bool,
    include_2d: bool = True,
) -> None:
    """Log hand overlays; 2D pixel archetypes and 3D camera-relative pose stay separable.

    A root that only feeds a `Spatial3DView` must pass `include_2d=False`: 2D archetypes
    under a 3D view root need a pinhole ancestor and otherwise surface as viewer errors.
    """
    if not include_2d and not include_3d:
        raise ValueError("hand logging must include at least one of 2D or 3D")
    hands_root = f"{root}/hands"
    if not observation.hands:
        if include_2d:
            rr.log(hands_root, rr.Clear(recursive=True))
        if include_3d:
            rr.log(f"{root}/camera_relative_3d", rr.Clear(recursive=True))
        return
    width, height = dimensions
    positions: list[list[float]] = []
    labels: list[str] = []
    strips: list[list[list[float]]] = []
    three_d: list[list[float]] = []
    three_d_labels: list[str] = []
    three_d_strips: list[list[list[float]]] = []
    for hand in observation.hands:
        hand_positions = [[point.x * width, point.y * height] for point in hand.landmarks]
        positions.extend(hand_positions)
        labels.extend(f"{hand.hand_id}: {name}" for name in HAND_LANDMARK_NAMES)
        strips.extend([[hand_positions[a], hand_positions[b]] for a, b in HAND_CONNECTIONS])
        if include_3d and hand.joints_3d_camera_relative:
            pose = [[point.x, point.y, point.z] for point in hand.joints_3d_camera_relative]
            three_d.extend(pose)
            three_d_labels.extend(f"{hand.hand_id}: {name}" for name in HAND_LANDMARK_NAMES)
            three_d_strips.extend([[pose[a], pose[b]] for a, b in HAND_CONNECTIONS])
    if include_2d:
        rr.log(
            f"{hands_root}/landmarks",
            rr.Points2D(positions, labels=labels, colors=[color] * len(positions), radii=2.5),
        )
        rr.log(
            f"{hands_root}/skeletons",
            rr.LineStrips2D(strips, colors=[color] * len(strips), radii=1.5),
        )
        mins = [[hand.box.x * width, hand.box.y * height] for hand in observation.hands]
        sizes = [[hand.box.width * width, hand.box.height * height] for hand in observation.hands]
        rr.log(
            f"{hands_root}/boxes",
            rr.Boxes2D(
                mins=mins,
                sizes=sizes,
                labels=[
                    f"{hand.hand_id}: {hand.side} ({hand.confidence:.2f})"
                    for hand in observation.hands
                ],
                colors=[color] * len(mins),
            ),
        )
    if include_3d:
        three_d_root = f"{root}/camera_relative_3d"
        if three_d:
            rr.log(
                f"{three_d_root}/joints",
                rr.Points3D(
                    three_d, labels=three_d_labels, colors=[color] * len(three_d), radii=0.002
                ),
            )
            rr.log(
                f"{three_d_root}/skeletons",
                rr.LineStrips3D(three_d_strips, colors=[color] * len(three_d_strips), radii=0.001),
            )
        else:
            rr.log(three_d_root, rr.Clear(recursive=True))


def _log_masks(
    root: str,
    observation: FrameObservations,
    *,
    run_directory: Path,
    dimensions: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    mask_root = f"{root}/masks"
    rr.log(mask_root, rr.Clear(recursive=True))
    width, height = dimensions
    for item in observation.objects:
        if item.mask is None:
            continue
        mask_path = (run_directory / item.mask.uri).resolve()
        if not mask_path.is_relative_to(run_directory) or not mask_path.is_file():
            raise FileNotFoundError(f"referenced comparison mask is unavailable: {item.mask.uri}")
        with Image.open(mask_path) as image:
            binary = np.asarray(image.convert("L"), dtype=np.uint8) > 0
        if binary.shape != (height, width):
            raise ValueError(f"mask dimensions do not match the shared video: {mask_path}")
        rr.log(
            f"{mask_root}/{item.object_id}",
            rr.EncodedImage(
                contents=_rgba_mask_png(binary, color),
                media_type="image/png",
                opacity=0.35,
                draw_order=1.0,
            ),
        )


def _log_nlf_body(
    root: str,
    observation: FrameObservations,
    *,
    dimensions: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    body_root = f"{root}/nlf_body_2d"
    if not observation.nlf_body_2d:
        rr.log(body_root, rr.Clear(recursive=True))
        return
    width, height = dimensions
    for body in observation.nlf_body_2d:
        points = [[item.x * width, item.y * height] for item in body.landmarks]
        rr.log(
            f"{body_root}/{body.subject_id}/joints",
            rr.Points2D(
                points,
                labels=[f"{body.subject_id}: {item.name}" for item in body.landmarks],
                colors=[color] * len(points),
                radii=1.5,
            ),
        )


def _log_method_frame(
    loaded: LoadedMethod, frame_index: int, *, dimensions: tuple[int, int]
) -> None:
    root = f"world/{loaded.manifest.clip.clip_id}/methods/{loaded.spec.method_id}"
    observation = loaded.observations.get(frame_index)
    processed, observed = _method_coverage(loaded.observations, frame_index)
    rr.log(f"{root}/metrics/processed", rr.Scalars([processed]))
    rr.log(f"{root}/metrics/output_present", rr.Scalars([observed]))
    if observation is None:
        rr.log(f"{root}/render", rr.Clear(recursive=True))
        return
    color = METHOD_COLORS.get(loaded.spec.method_id, (220, 220, 220))
    if observation.objects:
        mins, sizes, labels = _box_data(observation, dimensions)
        rr.log(
            f"{root}/render/boxes",
            rr.Boxes2D(mins=mins, sizes=sizes, labels=labels, colors=[color] * len(mins)),
        )
        _log_masks(
            f"{root}/render",
            observation,
            run_directory=loaded.run_directory,
            dimensions=dimensions,
            color=color,
        )
        rr.log(
            f"{root}/metrics/mean_confidence",
            rr.Scalars(
                [sum(item.confidence for item in observation.objects) / len(observation.objects)]
            ),
        )
    else:
        rr.log(f"{root}/render/boxes", rr.Clear(recursive=True))
        rr.log(f"{root}/render/masks", rr.Clear(recursive=True))
    _log_hands(
        f"{root}/render",
        observation,
        dimensions=dimensions,
        color=color,
        include_3d=loaded.spec.method_id == "wilor",
    )
    _log_nlf_body(f"{root}/render", observation, dimensions=dimensions, color=color)
    if observation.hands:
        rr.log(
            f"{root}/metrics/mean_confidence",
            rr.Scalars(
                [sum(item.confidence for item in observation.hands) / len(observation.hands)]
            ),
        )


def _comparison_blueprint(root: str, dimensions: tuple[int, int]) -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial2DView(
                    origin=root,
                    name="Static RGB with selected exploratory overlays",
                    contents=(
                        "$origin/source/video",
                        "$origin/methods/mediapipe/render/hands/**",
                        "$origin/methods/boxmot/render/boxes",
                    ),
                    visual_bounds=rrb.VisualBounds2D(
                        x_range=[0, dimensions[0]], y_range=[0, dimensions[1]]
                    ),
                ),
                rrb.Vertical(
                    rrb.TimeSeriesView(
                        origin=f"{root}/methods",
                        name="Method coverage and output presence",
                        contents="$origin/**/metrics/{processed,output_present}",
                    ),
                    rrb.TimeSeriesView(
                        origin=f"{root}/methods",
                        name="Available method confidences and Drop-DTW cost",
                        contents=(
                            "$origin/**/metrics/mean_confidence",
                            "$origin/drop_dtw/temporal/alignment_cost",
                        ),
                    ),
                ),
                column_shares=[3, 1],
            ),
            rrb.Horizontal(
                rrb.TextDocumentView(
                    origin=f"{root}/metadata/review_notes", name="Scope and limits"
                ),
                rrb.TextDocumentView(
                    origin=f"{root}/methods/drop_dtw/temporal/alignment",
                    name="Drop-DTW weak supervision",
                ),
                rrb.Spatial3DView(
                    origin=f"{root}/methods/wilor/render/camera_relative_3d",
                    name="WiLoR camera-relative non-metric 3D",
                    contents="$origin/**",
                ),
                column_shares=[2, 2, 2],
            ),
            row_shares=[4, 1],
        ),
        rrb.TimePanel(
            timeline="analysis_time",
            fps=ANALYSIS_FPS,
            time_selection=rr.encodings.AbsoluteTimeRange(0, 0),
        ),
        auto_layout=False,
        auto_views=False,
    )


def _drop_dtw_text(path: Path) -> tuple[str, float]:
    alignment = json.loads(path.read_text(encoding="utf-8"))
    intervals = alignment["intervals"]
    text = "\n".join(
        [
            "# CLIP + Drop-DTW (weak supervision)",
            alignment["weak_supervision_note"],
            f"Alignment cost: **{alignment['alignment_cost']:.6f}**",
            "",
            "Matched intervals:",
            *[
                f"- {item['action']}: {item['matched_seconds']} "
                f"(analysis frames {item['matched_analysis_frames']})"
                for item in intervals
            ],
        ]
    )
    return text, float(alignment["alignment_cost"])


def output_paths(repository_root: Path, output_root: Path = OUTPUT_ROOT) -> tuple[Path, Path]:
    root = (repository_root / output_root).resolve()
    return root / OUTPUT_NAME, root / INDEX_NAME


def build_exploratory_comparison(
    *,
    repository_root: Path,
    output_root: Path = OUTPUT_ROOT,
    methods: tuple[MethodSpec, ...] = DEFAULT_METHODS,
    frame_count: int = FRAME_COUNT,
    overwrite: bool = True,
    timer: PhaseTimer | None = None,
) -> Path:
    """Validate and compose the fixed bounded exploration without re-running inference."""
    timer = timer or PhaseTimer("exploratory comparison", enabled=False)
    if frame_count != FRAME_COUNT:
        raise ValueError("the final exploratory deliverable is fixed to 600 frames / 20 seconds")
    timer.start("validate")
    repository_root = repository_root.resolve()
    loaded = [
        _load_method(spec, repository_root=repository_root, frame_count=frame_count)
        for spec in methods
    ]
    source_fingerprint = _validate_shared_contract(loaded)
    for loaded_method in loaded:
        validate_artifact_fingerprint(
            source_fingerprint, repository_root, label="shared raw source"
        )
        proxy = loaded_method.manifest.clip.asset
        proxy_fingerprint = ArtifactFingerprint(
            uri=proxy.uri,
            sha256=proxy.checksum_sha256 or "",
            source="approved_config",
        )
        validate_artifact_fingerprint(
            proxy_fingerprint, repository_root, label=f"{loaded_method.spec.method_id} shared proxy"
        )
    video_path = loaded[0].run_directory / "input.mp4"
    bounded_video = _file_fingerprint(video_path, repository_root)
    actual_frames, fps, dimensions = _video_info(video_path)
    if actual_frames != frame_count or fps != ANALYSIS_FPS:
        raise ValueError(
            f"shared embedded video must be {frame_count} frames at {ANALYSIS_FPS} fps; "
            f"got {actual_frames} frames at {fps} fps"
        )
    timer.stop("validate")
    timer.start("export")
    rrd_path, index_path = output_paths(repository_root, output_root)
    if not overwrite and rrd_path.exists():
        raise FileExistsError(f"{rrd_path} already exists; pass --overwrite to replace it")
    rrd_path.parent.mkdir(parents=True, exist_ok=True)
    athena = _load_athena_metadata(repository_root)
    index = ExploratoryComparisonIndexManifest(
        manifest_kind="exploratory_first_20s_comparison",
        comparison_id=COMPARISON_ID,
        clip_id=loaded[0].manifest.clip.clip_id,
        source_video=source_fingerprint,
        bounded_video=bounded_video,
        bounded_video_frame_count=actual_frames,
        bounded_video_fps=fps,
        source_interval=TimeInterval(
            start_seconds=loaded[0].manifest.clip.timing.source_seconds_for_frame(
                ClockName.ANALYSIS, 0
            ),
            end_seconds=loaded[0].manifest.clip.timing.source_seconds_for_frame(
                ClockName.ANALYSIS, frame_count
            ),
        ),
        methods=tuple(item.index_method for item in loaded) + (athena,),
        build_notes=(
            "The 20-second bounded RGB asset is embedded exactly once and all 2D layers "
            "reference it.",
            "Methods remain separate entity roots; no cross-method ID or identity "
            "equivalence is asserted.",
            "Missing or out-of-coverage frames explicitly clear render entities.",
            "ATHENA is metadata-only: its synthetic fixture is not placed on the real "
            "source timeline.",
        ),
    )
    index_path.write_text(index.model_dump_json(indent=2) + "\n", encoding="utf-8")
    root = f"world/{index.clip_id}"
    rr.init("battle-exploratory-comparison", recording_id=COMPARISON_ID)
    rr.save(rrd_path)
    rr.log(f"{root}/source/video_asset", rr.AssetVideo(path=video_path), static=True)
    rr.log(
        f"{root}/metadata/comparison_index",
        rr.TextDocument(index.model_dump_json(indent=2), media_type="application/json"),
        static=True,
    )
    rr.log(
        f"{root}/metadata/review_notes",
        rr.TextDocument(
            "\n".join(
                [
                    "# Bounded exploratory comparison",
                    "All spatial layers use source-aligned static RGB frames 0–599 "
                    "(294.000–313.967 s).",
                    "Only MediaPipe hands and BoxMOT boxes are initially visible to avoid "
                    "mask/pose clutter.",
                    "Use the entity tree to toggle each independent method root.",
                    "ATHENA is blocked for real data; fixture URI: "
                    "runs/athena-fixture-triangulation-smoke/athena_fixture.rrd",
                ]
            ),
            media_type="text/markdown",
        ),
        static=True,
    )
    drop = next(item for item in loaded if item.spec.method_id == "drop_dtw")
    drop_text, drop_cost = _drop_dtw_text(drop.run_directory / "alignment.json")
    rr.log(
        f"{root}/methods/drop_dtw/temporal/alignment",
        rr.TextDocument(drop_text, media_type="text/markdown"),
        static=True,
    )
    for frame_index in range(frame_count):
        analysis_seconds = frame_index / ANALYSIS_FPS
        source_seconds = index.source_interval.start_seconds + analysis_seconds
        rr.set_time("analysis_frame", sequence=frame_index)
        rr.set_time("analysis_time", duration=analysis_seconds)
        rr.set_time("source_time", duration=source_seconds)
        rr.log(
            f"{root}/source/video",
            rr.VideoFrameReference(
                seconds=analysis_seconds, video_reference=f"{root}/source/video_asset"
            ),
        )
        rr.log(
            f"{root}/frame_counter",
            rr.TextDocument(
                f"analysis frame **{frame_index}** / {frame_count - 1}\n\n"
                f"source time {source_seconds:.3f} s",
                media_type="text/markdown",
            ),
        )
        for loaded_method in loaded:
            if loaded_method.spec.method_id != "drop_dtw":
                _log_method_frame(loaded_method, frame_index, dimensions=dimensions)
        if frame_index == 0:
            rr.log(f"{root}/methods/drop_dtw/temporal/alignment_cost", rr.Scalars([drop_cost]))
    rr.send_blueprint(_comparison_blueprint(root, dimensions))
    rr.disconnect()
    final_index = index.model_copy(
        update={"output_rrd": _file_fingerprint(rrd_path, repository_root)}
    )
    index_path.write_text(final_index.model_dump_json(indent=2) + "\n", encoding="utf-8")
    timer.stop("export")
    timer.print_report()
    return rrd_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Refuse to replace an existing recording in --output-root.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the per-phase timing report.",
    )
    args = parser.parse_args()
    output = build_exploratory_comparison(
        repository_root=args.repository_root,
        output_root=args.output_root,
        overwrite=not args.no_overwrite,
        timer=PhaseTimer("exploratory comparison", enabled=not args.quiet),
    )
    print(f"Wrote exploratory comparison: {output}")


if __name__ == "__main__":
    main()
