"""Export only synthetic, normalized fixture observations to a Rerun recording."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from . import mask_cache
from .fixtures import synthetic_run_manifest
from .schemas import ClockName, EncodedAssetInput, FrameObservations, RunManifest

# Per-object masks are logged on every analysis frame as RGBA PNG `EncodedImage`s. The store
# keeps only the compressed bytes, so full-rate masks cost tens of megabytes rather than the
# hundreds a raster `SegmentationImage` would, and PNG alpha gives a true cut-out overlay.
# The class-labelled `SegmentationImage` is retained as a sparse record at this rate. It is
# not drawn in the default view, because masks sit above it and would swallow its hover
# anyway; it stays in the recording for anyone who wants class labels at those frames.
SEGMENTATION_FPS = 1
MASK_OPACITY = 0.45
SEGMENTATION_OPACITY = 0.45
MASKS_PATH = "masks"
SEGMENTATION_PATH = "segmentation"
FRAME_COUNTER_PATH = "frame_counter"
DIAGNOSTICS_PATH = "tracker_diagnostics"
HAND_METRICS_PATH = "hand_metrics"
HAND_LANDMARK_NAMES = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)
HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)
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


def _load_binary_masks(
    observation: FrameObservations,
    *,
    mask_artifact_root: Path,
    video_dimensions: tuple[int, int] | None,
) -> list[tuple[str, np.ndarray]]:
    """Read this frame's external binary masks, checking each stays inside the artifact root."""
    mask_root = mask_artifact_root.resolve()
    cache = mask_cache.cache_for(mask_root)
    masks: list[tuple[str, np.ndarray]] = []
    expected_shape: tuple[int, int] | None = None
    for object_ in observation.objects:
        if object_.mask is None:
            continue
        mask_path = (mask_root / object_.mask.uri).resolve()
        binary_mask = cache.mask(object_.mask.uri)
        if expected_shape is None:
            height, width = binary_mask.shape
            if video_dimensions is not None and (width, height) != video_dimensions:
                raise ValueError(
                    f"mask dimensions {(width, height)} do not match video dimensions "
                    f"{video_dimensions}: {mask_path}"
                )
            expected_shape = binary_mask.shape
        elif binary_mask.shape != expected_shape:
            raise ValueError(f"mask dimensions differ within one frame: {mask_path}")
        masks.append((object_.object_id, binary_mask))
    return masks


def _segmentation_image(
    masks: list[tuple[str, np.ndarray]],
    *,
    annotations: dict[str, ObjectAnnotation],
) -> np.ndarray | None:
    """Compose one frame's binary masks into a single class-id image."""
    segmentation: np.ndarray | None = None
    for object_id, binary_mask in masks:
        if segmentation is None:
            segmentation = np.zeros(binary_mask.shape, dtype=np.uint16)
        segmentation[binary_mask] = annotations[object_id][0]
    return segmentation


def _rgba_mask_png(binary_mask: np.ndarray, color: tuple[int, int, int]) -> bytes:
    """Encode a cut-out: the object colour where the mask is set, transparent elsewhere."""
    return mask_cache.encode_rgba_mask_png(binary_mask, color)


def _log_masks(
    observation: FrameObservations,
    *,
    view_root: str,
    annotations: dict[str, ObjectAnnotation],
    mask_artifact_root: Path | None,
    video_dimensions: tuple[int, int] | None,
    segmentation_frame_period: int,
) -> None:
    """Log every object's mask this frame, plus the sparse class-labelled segmentation."""
    mask_references = [
        object_.mask.model_dump(mode="json") for object_ in observation.objects if object_.mask
    ]
    if not mask_references:
        return
    rr.log(
        f"{view_root}/mask_references",
        rr.TextDocument(json.dumps(mask_references), media_type="application/json"),
    )
    if mask_artifact_root is None:
        return
    cache = mask_cache.cache_for(mask_artifact_root.resolve())
    masks = _load_binary_masks(
        observation, mask_artifact_root=mask_artifact_root, video_dimensions=video_dimensions
    )
    references = {
        object_.object_id: object_.mask.uri for object_ in observation.objects if object_.mask
    }
    for object_id, binary_mask in masks:
        rr.log(
            f"{view_root}/{MASKS_PATH}/{object_id}",
            rr.EncodedImage(
                contents=cache.rgba_png(references[object_id], annotations[object_id][2]),
                media_type="image/png",
                opacity=MASK_OPACITY,
                draw_order=1.0,
            ),
        )
    if observation.analysis_frame_index % segmentation_frame_period != 0:
        return
    segmentation = _segmentation_image(masks, annotations=annotations)
    if segmentation is not None:
        rr.log(
            f"{view_root}/{SEGMENTATION_PATH}",
            rr.SegmentationImage(segmentation, opacity=SEGMENTATION_OPACITY, draw_order=0.5),
        )


def _segmentation_frame_period(analysis_fps: int) -> int:
    return max(1, analysis_fps // SEGMENTATION_FPS)


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


def _log_nlf_body_2d(
    observation: FrameObservations,
    *,
    view_root: str,
    video_dimensions: tuple[int, int] | None,
) -> None:
    width, height = video_dimensions or (1, 1)
    root = f"{view_root}/nlf_body_2d"
    if not observation.nlf_body_2d:
        rr.log(root, rr.Clear(recursive=True))
        return
    for pose in observation.nlf_body_2d:
        positions = [[landmark.x * width, landmark.y * height] for landmark in pose.landmarks]
        labels = [f"{pose.subject_id}:{landmark.name}" for landmark in pose.landmarks]
        rr.log(
            f"{root}/{pose.subject_id}/landmarks",
            rr.Points2D(
                positions,
                labels=labels,
                colors=[(120, 220, 160)] * len(positions),
                radii=2.5,
            ),
        )


def _log_hands(
    observation: FrameObservations,
    *,
    view_root: str,
    video_dimensions: tuple[int, int] | None,
) -> None:
    """Log all hands as one replaceable overlay plus per-track confidence traces."""
    width, height = video_dimensions or (1, 1)
    hands_root = f"{view_root}/hands"
    metrics_root = f"{view_root}/{HAND_METRICS_PATH}"
    rr.log(f"{metrics_root}/count", rr.Scalars([len(observation.hands)]))
    if not observation.hands:
        rr.log(hands_root, rr.Clear(recursive=True))
        return
    rr.log(
        f"{metrics_root}/mean_handedness_confidence",
        rr.Scalars([sum(hand.confidence for hand in observation.hands) / len(observation.hands)]),
    )

    colors = {
        "left": (80, 180, 255),
        "right": (255, 170, 70),
        "unknown": (210, 210, 210),
    }
    landmark_positions = []
    landmark_labels = []
    landmark_colors = []
    strips = []
    strip_colors = []
    joints_3d_positions = []
    joints_3d_labels = []
    joints_3d_colors = []
    joints_3d_strips = []
    joints_3d_strip_colors = []
    for hand in observation.hands:
        color = colors[str(hand.side)]
        positions = [[landmark.x * width, landmark.y * height] for landmark in hand.landmarks]
        landmark_positions.extend(positions)
        landmark_labels.extend(f"{hand.hand_id}: {name}" for name in HAND_LANDMARK_NAMES)
        landmark_colors.extend([color] * len(positions))
        strips.extend([[positions[start], positions[end]] for start, end in HAND_CONNECTIONS])
        strip_colors.extend([color] * len(HAND_CONNECTIONS))
        if hand.joints_3d_camera_relative:
            joint_positions = [
                [joint.x, joint.y, joint.z] for joint in hand.joints_3d_camera_relative
            ]
            joints_3d_positions.extend(joint_positions)
            joints_3d_labels.extend(f"{hand.hand_id}: {name}" for name in HAND_LANDMARK_NAMES)
            joints_3d_colors.extend([color] * len(joint_positions))
            joints_3d_strips.extend(
                [[joint_positions[start], joint_positions[end]] for start, end in HAND_CONNECTIONS]
            )
            joints_3d_strip_colors.extend([color] * len(HAND_CONNECTIONS))
    rr.log(
        f"{hands_root}/landmarks",
        rr.Points2D(
            landmark_positions,
            labels=landmark_labels,
            colors=landmark_colors,
            radii=3.0,
            draw_order=3.0,
        ),
    )
    rr.log(
        f"{hands_root}/skeletons",
        rr.LineStrips2D(strips, colors=strip_colors, radii=2.0, draw_order=2.5),
    )
    rr.log(
        f"{hands_root}/boxes",
        rr.Boxes2D(
            mins=[[hand.box.x * width, hand.box.y * height] for hand in observation.hands],
            sizes=[
                [hand.box.width * width, hand.box.height * height] for hand in observation.hands
            ],
            labels=[
                f"{hand.hand_id}: {hand.side} ({hand.confidence:.2f})" for hand in observation.hands
            ],
            colors=[colors[str(hand.side)] for hand in observation.hands],
            draw_order=2.0,
        ),
    )
    if joints_3d_positions:
        rr.log(
            f"{hands_root}/joints_3d_camera_relative",
            rr.Points3D(
                joints_3d_positions,
                labels=joints_3d_labels,
                colors=joints_3d_colors,
                radii=0.002,
            ),
        )
        rr.log(
            f"{hands_root}/skeletons_3d_camera_relative",
            rr.LineStrips3D(joints_3d_strips, colors=joints_3d_strip_colors, radii=0.001),
        )
    else:
        rr.log(f"{hands_root}/joints_3d_camera_relative", rr.Clear(recursive=True))
        rr.log(f"{hands_root}/skeletons_3d_camera_relative", rr.Clear(recursive=True))


def _spatial_view_contents() -> list[str]:
    """Show video, boxes, and per-object masks; leave traces and the sparse segmentation off.

    The segmentation is excluded rather than removed so it can be toggled on from the
    blueprint panel when class labels are wanted at its keyframes.
    """
    return [
        "$origin/**",
        f"- $origin/{DIAGNOSTICS_PATH}/**",
        f"- $origin/{SEGMENTATION_PATH}",
    ]


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
    has_hands: bool = False,
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
                    contents=_spatial_view_contents(),
                    name=f"{view_id} video and detections",
                    visual_bounds=visual_bounds,
                ),
                rrb.Vertical(
                    *_diagnostics_view_pair(view_root),
                    *(
                        (
                            rrb.TimeSeriesView(
                                origin=f"{view_root}/{HAND_METRICS_PATH}",
                                contents="$origin/**",
                                name="Hand detections",
                            ),
                        )
                        if has_hands
                        else ()
                    ),
                ),
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
    has_static_hands: bool = False,
) -> rrb.Blueprint:
    """Build a pinned side-by-side layout sharing the analysis-time timeline."""
    root = f"world/{clip_id}/synchronized_ego_static_comparison"

    def spatial_view(view_id: str, label: str, dimensions: tuple[int, int]) -> rrb.Spatial2DView:
        return rrb.Spatial2DView(
            origin=f"{root}/views/{view_id}",
            contents=_spatial_view_contents(),
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
                *(
                    (
                        rrb.TimeSeriesView(
                            origin=f"{root}/views/{static_view_id}/{HAND_METRICS_PATH}",
                            contents="$origin/**",
                            name="static MediaPipe hand detections",
                        ),
                    )
                    if has_static_hands
                    else ()
                ),
                column_shares=[1, 1, 1] if has_static_hands else [1, 1],
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
    segmentation_frame_period: int,
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
    _log_masks(
        observation,
        view_root=view_root,
        annotations=annotations,
        mask_artifact_root=mask_artifact_root,
        video_dimensions=video_dimensions,
        segmentation_frame_period=segmentation_frame_period,
    )
    _log_hands(
        observation,
        view_root=view_root,
        video_dimensions=video_dimensions,
    )
    _log_nlf_body_2d(
        observation,
        view_root=view_root,
        video_dimensions=video_dimensions,
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
    ego_label: str = "manual-seed multiplexed SAM3 baseline",
    ego_description: str = "Existing bounded ego manual-seed baseline video.",
    static_description: str = (
        "Existing bounded static-camera video paired to the same analysis timeline."
    ),
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
        ego_label=ego_label,
        static_label=static_label,
        has_static_hands=any(observation.hands for observation in static_manifest.observations),
    )
    rr.init(
        f"battle-synchronized-ego-static-comparison-{ego_manifest.clip.clip_id}",
        recording_id=f"{ego_manifest.run_id}--{static_manifest.run_id}",
    )
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
                        "label": ego_label,
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
        description=ego_description,
    )
    static_asset = _log_source_asset(
        view_root=static_root,
        asset_reference=static_asset_reference,
        video_path=static_video_path,
        description=static_description,
    )
    ego_annotations = _object_annotations(ego_manifest)[ego_view_id]
    static_annotations = _object_annotations(static_manifest)[static_view_id]
    rr.log(ego_root, _annotation_context(ego_annotations), static=True)
    rr.log(static_root, _annotation_context(static_annotations), static=True)

    segmentation_frame_period = _segmentation_frame_period(analysis_fps)
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
            segmentation_frame_period=segmentation_frame_period,
        )
        _log_observation(
            static_observation,
            view_root=static_root,
            video_asset_path=static_asset,
            video_dimensions=static_video_dimensions,
            annotations=static_annotations,
            mask_artifact_root=static_mask_artifact_root,
            analysis_seconds=analysis_seconds,
            segmentation_frame_period=segmentation_frame_period,
        )
        _log_tracker_diagnostics(ego_observation, view_root=ego_root, annotations=ego_annotations)
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
        has_hands=any(observation.hands for observation in manifest.observations),
    )
    # Rerun keys a blueprint by application id, and every view in ours is anchored under a
    # clip-specific entity root. Recordings of different clips must therefore not share an
    # application id: loading them together would let one clip's blueprint win and point
    # every view at entity paths the other recordings do not contain, showing empty views.
    #
    # The recording id is the run id rather than a random value. Random ids are only unique
    # per process, so exporting several runs from one process gave them the same id and the
    # viewer merged them into a single recording. The run id is unique by construction and
    # also makes the recording identifiable in the viewer.
    rr.init(f"battle-{clip.clip_id}", recording_id=manifest.run_id)
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

    segmentation_frame_period = _segmentation_frame_period(analysis_fps)
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
        _log_masks(
            observation,
            view_root=view_root,
            annotations=annotations,
            mask_artifact_root=mask_artifact_root,
            video_dimensions=video_dimensions,
            segmentation_frame_period=segmentation_frame_period,
        )

        _log_hands(
            observation,
            view_root=view_root,
            video_dimensions=video_dimensions,
        )
        _log_nlf_body_2d(
            observation,
            view_root=view_root,
            video_dimensions=video_dimensions,
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
