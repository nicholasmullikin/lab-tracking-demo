"""Export only synthetic, normalized fixture observations to a Rerun recording."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from PIL import Image

from .fixtures import synthetic_run_manifest
from .schemas import ClockName, EncodedAssetInput, FrameObservations, RunManifest

MASK_FPS = 5
SEGMENTATION_OPACITY = 0.45
FRAME_COUNTER_PATH = "frame_counter"
DIAGNOSTICS_PATH = "tracker_diagnostics"
BACKGROUND_ANNOTATION: tuple[int, str, tuple[int, int, int, int]] = (
    0,
    "background",
    (0, 0, 0, 0),
)
OBJECT_COLORS = (
    (255, 85, 85),
    (255, 210, 65),
    (80, 180, 255),
    (180, 115, 255),
    (70, 210, 165),
)
ObjectAnnotation = tuple[int, str, tuple[int, int, int]]
AnnotationsByView = dict[str, dict[str, ObjectAnnotation]]


def _object_annotations(manifest: RunManifest) -> AnnotationsByView:
    """Assign stable, distinct segmentation classes to objects in every view."""
    annotations: AnnotationsByView = {}
    for observation in manifest.observations:
        view_annotations = annotations.setdefault(observation.view_id, {})
        for object_ in observation.objects:
            if object_.object_id not in view_annotations:
                class_id = len(view_annotations) + 1
                view_annotations[object_.object_id] = (
                    class_id,
                    object_.label,
                    OBJECT_COLORS[(class_id - 1) % len(OBJECT_COLORS)],
                )
    return annotations


def _annotation_context(view_annotations: dict[str, ObjectAnnotation]) -> rr.AnnotationContext:
    """Map unlabeled pixels to transparent and object classes to their display colors."""
    return rr.AnnotationContext(
        [
            rr.AnnotationInfo(*BACKGROUND_ANNOTATION),
            *[
                rr.AnnotationInfo(id=class_id, label=label, color=color)
                for class_id, label, color in view_annotations.values()
            ],
        ]
    )


def _segmentation_image(
    observation: FrameObservations,
    *,
    mask_artifact_root: Path,
    annotations: dict[str, ObjectAnnotation],
    video_dimensions: tuple[int, int] | None,
) -> np.ndarray | None:
    """Compose this frame's external binary masks into one Rerun segmentation image."""
    mask_root = mask_artifact_root.resolve()
    segmentation: np.ndarray | None = None
    for object_ in observation.objects:
        if object_.mask is None:
            continue
        mask_path = (mask_root / object_.mask.uri).resolve()
        if not mask_path.is_relative_to(mask_root):
            raise ValueError(f"mask reference escapes mask artifact root: {object_.mask.uri}")
        if not mask_path.is_file():
            raise FileNotFoundError(f"referenced mask is unavailable: {mask_path}")
        with Image.open(mask_path) as mask_file:
            binary_mask = np.asarray(mask_file.convert("L"), dtype=np.uint8) > 0
        if segmentation is None:
            height, width = binary_mask.shape
            if video_dimensions is not None and (width, height) != video_dimensions:
                raise ValueError(
                    f"mask dimensions {(width, height)} do not match video dimensions "
                    f"{video_dimensions}: {mask_path}"
                )
            segmentation = np.zeros(binary_mask.shape, dtype=np.uint16)
        elif binary_mask.shape != segmentation.shape:
            raise ValueError(f"mask dimensions differ within one frame: {mask_path}")
        segmentation[binary_mask] = annotations[object_.object_id][0]
    return segmentation


def _log_frame_counter(
    root: str,
    *,
    analysis_frame_index: int,
    analysis_seconds: float,
    total_frames: int,
    analysis_fps: int,
) -> None:
    """Log the active analysis frame so the viewer shows a frame count without scrubbing."""
    last_index = max(total_frames - 1, 0)
    rr.log(
        f"{root}/{FRAME_COUNTER_PATH}",
        rr.TextDocument(
            f"analysis frame **{analysis_frame_index}** / {last_index}"
            f" ({total_frames} frames)\n\n"
            f"{analysis_seconds:.3f} s at {analysis_fps} fps",
            media_type="text/markdown",
        ),
    )


def _diagnostics_view_pair(
    view_root: str, *, label_prefix: str = ""
) -> tuple[rrb.TimeSeriesView, rrb.TimeSeriesView]:
    """Build the raw tracker traces that explain why a slot was kept or dropped."""
    return (
        rrb.TimeSeriesView(
            origin=f"{view_root}/{DIAGNOSTICS_PATH}/object_score",
            contents="$origin/**",
            name=f"{label_prefix}Object score (lost at or below 0)",
        ),
        rrb.TimeSeriesView(
            origin=f"{view_root}/{DIAGNOSTICS_PATH}/iou_prediction",
            contents="$origin/**",
            name=f"{label_prefix}Predicted mask IoU (self-estimate)",
        ),
    )


def _log_tracker_diagnostics(
    observation: FrameObservations,
    *,
    view_root: str,
    annotations: dict[str, ObjectAnnotation],
) -> None:
    """Log each multiplex slot's raw score and IoU, including frames where it was dropped."""
    if not observation.tracker_diagnostics:
        return
    root = f"{view_root}/{DIAGNOSTICS_PATH}"
    rr.log(f"{root}/object_score/lost_threshold", rr.Scalars([0.0]))
    for diagnostic in observation.tracker_diagnostics:
        color = annotations.get(diagnostic.object_id, (0, "", (200, 200, 200)))[2]
        series = f"{diagnostic.label} ({diagnostic.object_id})"
        rr.log(
            f"{root}/object_score/{diagnostic.object_id}",
            rr.Scalars([diagnostic.object_score]),
            rr.SeriesLines(colors=[color], names=[series]),
        )
        if diagnostic.iou_prediction is not None:
            rr.log(
                f"{root}/iou_prediction/{diagnostic.object_id}",
                rr.Scalars([diagnostic.iou_prediction]),
                rr.SeriesLines(colors=[color], names=[series]),
            )


def _frame_counter_view(root: str) -> rrb.TextDocumentView:
    """Build the compact readout that reports the active analysis frame."""
    return rrb.TextDocumentView(
        origin=f"{root}/{FRAME_COUNTER_PATH}",
        name="Frame count",
    )


def pinned_blueprint(
    clip_id: str,
    view_id: str = "ego-01",
    *,
    analysis_fps: int = 30,
    video_dimensions: tuple[int, int] | None = None,
) -> rrb.Blueprint:
    """Build a deterministic layout for the exported observation view."""
    root = f"world/{clip_id}"
    view_root = f"{root}/views/{view_id}"
    visual_bounds = (
        rrb.VisualBounds2D(
            x_range=[0, video_dimensions[0]],
            y_range=[0, video_dimensions[1]],
        )
        if video_dimensions is not None
        else None
    )
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial2DView(
                    origin=view_root,
                    contents=["$origin/**", f"- $origin/{DIAGNOSTICS_PATH}/**"],
                    name=f"{view_id} video and detections",
                    visual_bounds=visual_bounds,
                ),
                rrb.Vertical(*_diagnostics_view_pair(view_root)),
                column_shares=[2, 1],
            ),
            rrb.Horizontal(
                rrb.TimeSeriesView(
                    origin=f"{root}/quality",
                    contents="$origin/**",
                    name="Coverage",
                ),
                _frame_counter_view(root),
                column_shares=[3, 1],
            ),
            row_shares=[4, 1],
        ),
        rrb.TimePanel(
            timeline="analysis_time",
            fps=analysis_fps,
            time_selection=rr.encodings.AbsoluteTimeRange(0, 0),
        ),
        auto_layout=False,
        auto_views=False,
    )


def pinned_comparison_blueprint(
    clip_id: str,
    *,
    ego_view_id: str,
    static_view_id: str,
    analysis_fps: int,
    ego_video_dimensions: tuple[int, int],
    static_video_dimensions: tuple[int, int],
    ego_label: str = "ego manual-seed SAM3 baseline",
    static_label: str = "static existing SAM3 output",
) -> rrb.Blueprint:
    """Build a pinned side-by-side layout sharing the analysis-time timeline."""
    root = f"world/{clip_id}/synchronized_ego_static_comparison"

    def spatial_view(
        view_id: str, label: str, dimensions: tuple[int, int]
    ) -> rrb.Spatial2DView:
        return rrb.Spatial2DView(
            origin=f"{root}/views/{view_id}",
            contents=["$origin/**", f"- $origin/{DIAGNOSTICS_PATH}/**"],
            name=f"{view_id}: {label}",
            visual_bounds=rrb.VisualBounds2D(
                x_range=[0, dimensions[0]],
                y_range=[0, dimensions[1]],
            ),
        )

    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                spatial_view(ego_view_id, ego_label, ego_video_dimensions),
                spatial_view(static_view_id, static_label, static_video_dimensions),
                column_shares=[1, 1],
            ),
            rrb.Horizontal(
                *_diagnostics_view_pair(f"{root}/views/{ego_view_id}", label_prefix="ego "),
                column_shares=[1, 1],
            ),
            rrb.Horizontal(
                rrb.TimeSeriesView(
                    origin=f"{root}/quality",
                    contents="$origin/**",
                    name="Output coverage",
                ),
                _frame_counter_view(root),
                column_shares=[3, 1],
            ),
            row_shares=[4, 2, 1],
        ),
        rrb.TimePanel(
            timeline="analysis_time",
            fps=analysis_fps,
            time_selection=rr.encodings.AbsoluteTimeRange(0, 0),
        ),
        auto_layout=False,
        auto_views=False,
    )


def _log_source_asset(
    *,
    view_root: str,
    asset_reference: EncodedAssetInput,
    video_path: Path,
    description: str,
) -> str:
    """Log one bounded asset and the provenance that identifies it."""
    video_asset_path = f"{view_root}/video_asset"
    rr.log(
        f"{view_root}/asset_reference",
        rr.TextDocument(
            json.dumps(
                {
                    "asset": asset_reference.model_dump(mode="json"),
                    "embedded_video": str(video_path),
                    "description": description,
                },
                indent=2,
            ),
            media_type="application/json",
        ),
        static=True,
    )
    rr.log(video_asset_path, rr.AssetVideo(path=video_path), static=True)
    return video_asset_path


def _log_observation(
    observation: FrameObservations,
    *,
    view_root: str,
    video_asset_path: str,
    video_dimensions: tuple[int, int],
    annotations: dict[str, ObjectAnnotation],
    mask_artifact_root: Path | None,
    analysis_seconds: float,
    mask_frame_period: int,
) -> None:
    """Log one view at an already-selected synchronized timestamp."""
    rr.log(
        f"{view_root}/video",
        rr.VideoFrameReference(seconds=analysis_seconds, video_reference=video_asset_path),
    )
    if observation.objects:
        width, height = video_dimensions
        rr.log(
            f"{view_root}/objects",
            rr.Boxes2D(
                mins=[
                    [object_.box.x * width, object_.box.y * height]
                    for object_ in observation.objects
                ],
                sizes=[
                    [object_.box.width * width, object_.box.height * height]
                    for object_ in observation.objects
                ],
                labels=[
                    f"{object_.label} ({object_.object_id})" for object_ in observation.objects
                ],
                colors=[annotations[object_.object_id][2] for object_ in observation.objects],
            ),
        )
    mask_references = [
        object_.mask.model_dump(mode="json")
        for object_ in observation.objects
        if object_.mask and observation.analysis_frame_index % mask_frame_period == 0
    ]
    if mask_references:
        rr.log(
            f"{view_root}/mask_references",
            rr.TextDocument(json.dumps(mask_references), media_type="application/json"),
        )
    if mask_artifact_root is None or observation.analysis_frame_index % mask_frame_period != 0:
        return
    segmentation = _segmentation_image(
        observation,
        mask_artifact_root=mask_artifact_root,
        annotations=annotations,
        video_dimensions=video_dimensions,
    )
    if segmentation is not None:
        rr.log(
            f"{view_root}/segmentation",
            rr.SegmentationImage(
                segmentation,
                opacity=SEGMENTATION_OPACITY,
                draw_order=1.0,
            ),
        )


def _require_synchronized_observations(
    ego_observations: tuple[FrameObservations, ...],
    static_observations: tuple[FrameObservations, ...],
    *,
    analysis_fps: int,
) -> None:
    """Reject comparison exports whose frames do not share an analysis-time base."""
    if not ego_observations or len(ego_observations) != len(static_observations):
        raise ValueError("comparison views must contain the same non-zero frame count")
    for expected_frame, (ego, static) in enumerate(zip(ego_observations, static_observations)):
        if (
            ego.analysis_frame_index != expected_frame
            or static.analysis_frame_index != expected_frame
        ):
            raise ValueError("comparison observations must be contiguous from analysis frame zero")
        if abs(ego.source_seconds - static.source_seconds) > 1e-6:
            raise ValueError("comparison observations do not share source timestamps")
        expected_seconds = expected_frame / analysis_fps
        if abs(ego.source_seconds - ego_observations[0].source_seconds - expected_seconds) > 1e-6:
            raise ValueError("ego observations do not match the declared analysis FPS")


def export_synchronized_comparison(
    ego_manifest: RunManifest,
    static_manifest: RunManifest,
    output_path: Path,
    *,
    ego_video_path: Path,
    ego_video_dimensions: tuple[int, int],
    ego_asset_reference: EncodedAssetInput,
    ego_mask_artifact_root: Path | None,
    static_video_path: Path,
    static_video_dimensions: tuple[int, int],
    static_asset_reference: EncodedAssetInput,
    static_mask_artifact_root: Path | None,
    static_label: str,
) -> Path:
    """Export two existing views to one analysis-time-synchronized RRD without inference."""
    analysis_fps = ego_manifest.clip.timing.clocks.fps_for(ClockName.ANALYSIS)
    static_analysis_fps = static_manifest.clip.timing.clocks.fps_for(ClockName.ANALYSIS)
    if analysis_fps != static_analysis_fps:
        raise ValueError("comparison views must use the same analysis FPS")
    _require_synchronized_observations(
        ego_manifest.observations, static_manifest.observations, analysis_fps=analysis_fps
    )
    for video_path in (ego_video_path, static_video_path):
        if not video_path.is_file():
            raise FileNotFoundError(f"comparison video is unavailable: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    root = f"world/{ego_manifest.clip.clip_id}/synchronized_ego_static_comparison"
    ego_view_id = ego_manifest.observations[0].view_id
    static_view_id = static_manifest.observations[0].view_id
    if ego_view_id == static_view_id:
        raise ValueError("comparison views must have distinct view IDs")
    blueprint = pinned_comparison_blueprint(
        ego_manifest.clip.clip_id,
        ego_view_id=ego_view_id,
        static_view_id=static_view_id,
        analysis_fps=analysis_fps,
        ego_video_dimensions=ego_video_dimensions,
        static_video_dimensions=static_video_dimensions,
        static_label=static_label,
    )
    rr.init("battle-synchronized-ego-static-comparison")
    rr.save(output_path)
    rr.log(
        f"{root}/comparison_metadata",
        rr.TextDocument(
            json.dumps(
                {
                    "timeline": "analysis_time",
                    "analysis_fps": analysis_fps,
                    "source_time_mapping": (
                        "source_seconds = "
                        f"{ego_manifest.observations[0].source_seconds:.6f} + "
                        f"analysis_frame / {analysis_fps}"
                    ),
                    "ego": {
                        "view_id": ego_view_id,
                        "label": "manual-seed multiplexed SAM3 baseline",
                    },
                    "static": {"view_id": static_view_id, "label": static_label},
                },
                indent=2,
            ),
            media_type="application/json",
        ),
        static=True,
    )
    ego_root = f"{root}/views/{ego_view_id}"
    static_root = f"{root}/views/{static_view_id}"
    ego_asset = _log_source_asset(
        view_root=ego_root,
        asset_reference=ego_asset_reference,
        video_path=ego_video_path,
        description="Existing bounded ego manual-seed baseline video.",
    )
    static_asset = _log_source_asset(
        view_root=static_root,
        asset_reference=static_asset_reference,
        video_path=static_video_path,
        description="Existing bounded static-camera video paired to the same analysis timeline.",
    )
    ego_annotations = _object_annotations(ego_manifest)[ego_view_id]
    static_annotations = _object_annotations(static_manifest)[static_view_id]
    rr.log(ego_root, _annotation_context(ego_annotations), static=True)
    rr.log(static_root, _annotation_context(static_annotations), static=True)

    mask_frame_period = analysis_fps // MASK_FPS
    total_frames = len(ego_manifest.observations)
    for ego_observation, static_observation in zip(
        ego_manifest.observations, static_manifest.observations
    ):
        analysis_seconds = ego_observation.analysis_frame_index / analysis_fps
        rr.set_time("analysis_frame", sequence=ego_observation.analysis_frame_index)
        rr.set_time("analysis_time", duration=analysis_seconds)
        rr.set_time("source_time", duration=ego_observation.source_seconds)
        _log_frame_counter(
            root,
            analysis_frame_index=ego_observation.analysis_frame_index,
            analysis_seconds=analysis_seconds,
            total_frames=total_frames,
            analysis_fps=analysis_fps,
        )
        _log_observation(
            ego_observation,
            view_root=ego_root,
            video_asset_path=ego_asset,
            video_dimensions=ego_video_dimensions,
            annotations=ego_annotations,
            mask_artifact_root=ego_mask_artifact_root,
            analysis_seconds=analysis_seconds,
            mask_frame_period=mask_frame_period,
        )
        _log_observation(
            static_observation,
            view_root=static_root,
            video_asset_path=static_asset,
            video_dimensions=static_video_dimensions,
            annotations=static_annotations,
            mask_artifact_root=static_mask_artifact_root,
            analysis_seconds=analysis_seconds,
            mask_frame_period=mask_frame_period,
        )
        _log_tracker_diagnostics(
            ego_observation, view_root=ego_root, annotations=ego_annotations
        )
        _log_tracker_diagnostics(
            static_observation, view_root=static_root, annotations=static_annotations
        )
    rr.log(f"{root}/quality/ego_coverage", rr.Scalars([ego_manifest.coverage.ratio]), static=True)
    rr.log(
        f"{root}/quality/static_coverage", rr.Scalars([static_manifest.coverage.ratio]), static=True
    )
    rr.send_blueprint(blueprint)
    rr.disconnect()
    return output_path


def export_run(
    manifest: RunManifest,
    output_path: Path,
    *,
    video_path: Path | None = None,
    video_dimensions: tuple[int, int] | None = None,
    asset_reference: EncodedAssetInput | None = None,
    mask_artifact_root: Path | None = None,
) -> Path:
    """Write normalized observations and optionally embed referenced masks without inference."""
    if (video_path is None) != (video_dimensions is None):
        raise ValueError("video_path and video_dimensions must be supplied together")
    if video_dimensions is not None and (video_dimensions[0] <= 0 or video_dimensions[1] <= 0):
        raise ValueError("video dimensions must be positive")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    clip = manifest.clip
    root = f"world/{clip.clip_id}"
    analysis_fps = clip.timing.clocks.fps_for(ClockName.ANALYSIS)
    active_view_id = manifest.observations[0].view_id if manifest.observations else clip.views[0]
    active_view_root = f"{root}/views/{active_view_id}"
    video_asset_path = f"{active_view_root}/video_asset"
    blueprint = pinned_blueprint(
        clip.clip_id,
        active_view_id,
        analysis_fps=analysis_fps,
        video_dimensions=video_dimensions,
    )
    rr.init("battle-session-1")
    rr.save(output_path)

    rr.log(
        f"{root}/source/asset_reference",
        rr.TextDocument(
            json.dumps((asset_reference or clip.asset).model_dump(mode="json"), indent=2),
            media_type="application/json",
        ),
        static=True,
    )
    rr.log(
        f"{root}/source/asset_policy",
        rr.TextDocument(
            (
                "The bounded input video is embedded once as an AssetVideo and linked to every "
                "analysis frame."
                if video_path is not None
                else (
                    "Encoded asset contract only. No video payload is read or embedded in this RRD."
                )
            ),
            media_type="text/plain",
        ),
        static=True,
    )
    if video_path is not None:
        rr.log(video_asset_path, rr.AssetVideo(path=video_path), static=True)

    annotations_by_view = _object_annotations(manifest)
    for view_id, view_annotations in annotations_by_view.items():
        rr.log(
            f"{root}/views/{view_id}",
            _annotation_context(view_annotations),
            static=True,
        )

    for clock in clip.timing.clocks.clocks:
        rr.log(
            f"{root}/timing/{clock.name}",
            rr.TextDocument(
                json.dumps({"fps": clock.fps, "clock": clock.name}, indent=2),
                media_type="application/json",
            ),
            static=True,
        )

    mask_frame_period = analysis_fps // MASK_FPS
    blueprint_sent = False
    total_frames = len(manifest.observations)
    for observation in manifest.observations:
        rr.set_time("analysis_frame", sequence=observation.analysis_frame_index)
        analysis_seconds = observation.analysis_frame_index / analysis_fps
        rr.set_time("analysis_time", duration=analysis_seconds)
        _log_frame_counter(
            root,
            analysis_frame_index=observation.analysis_frame_index,
            analysis_seconds=analysis_seconds,
            total_frames=total_frames,
            analysis_fps=analysis_fps,
        )
        view_root = f"{root}/views/{observation.view_id}"
        annotations = annotations_by_view[observation.view_id]

        if video_path is not None:
            rr.log(
                f"{view_root}/video",
                rr.VideoFrameReference(
                    seconds=analysis_seconds,
                    video_reference=video_asset_path,
                ),
            )
        if observation.objects:
            width, height = video_dimensions or (1, 1)
            rr.log(
                f"{view_root}/objects",
                rr.Boxes2D(
                    mins=[
                        [object_.box.x * width, object_.box.y * height]
                        for object_ in observation.objects
                    ],
                    sizes=[
                        [object_.box.width * width, object_.box.height * height]
                        for object_ in observation.objects
                    ],
                    labels=[
                        f"{object_.label} ({object_.object_id})" for object_ in observation.objects
                    ],
                    colors=[annotations[object_.object_id][2] for object_ in observation.objects],
                ),
            )
        mask_references = [
            object_.mask.model_dump(mode="json")
            for object_ in observation.objects
            if object_.mask and observation.analysis_frame_index % mask_frame_period == 0
        ]
        if mask_references:
            rr.log(
                f"{view_root}/mask_references",
                rr.TextDocument(
                    json.dumps(mask_references),
                    media_type="application/json",
                ),
            )
        if (
            mask_artifact_root is not None
            and observation.analysis_frame_index % mask_frame_period == 0
        ):
            segmentation = _segmentation_image(
                observation,
                mask_artifact_root=mask_artifact_root,
                annotations=annotations,
                video_dimensions=video_dimensions,
            )
            if segmentation is not None:
                rr.log(
                    f"{view_root}/segmentation",
                    rr.SegmentationImage(
                        segmentation,
                        opacity=SEGMENTATION_OPACITY,
                        draw_order=1.0,
                    ),
                )

        for hand in observation.hands:
            width, height = video_dimensions or (1, 1)
            rr.log(
                f"{view_root}/hands",
                rr.Points2D(
                    positions=[
                        [landmark.x * width, landmark.y * height] for landmark in hand.landmarks
                    ],
                    labels=[hand.hand_id] * len(hand.landmarks),
                    colors=[[255, 170, 70]],
                ),
            )
        _log_tracker_diagnostics(observation, view_root=view_root, annotations=annotations)
        if not blueprint_sent:
            rr.send_blueprint(blueprint)
            blueprint_sent = True

    rr.log(
        f"{root}/quality/coverage",
        rr.Scalars([manifest.coverage.ratio]),
        static=True,
    )
    if not blueprint_sent:
        rr.send_blueprint(blueprint)
    rr.disconnect()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the synthetic Battle fixture to Rerun.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/synthetic_fixture.rrd"),
        help="Path for the generated RRD.",
    )
    args = parser.parse_args()
    output = export_run(synthetic_run_manifest(), args.output)
    print(f"Wrote synthetic Rerun fixture: {output}")


if __name__ == "__main__":
    main()
