from __future__ import annotations

import io
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from battle.exporter import (
    _annotation_context,
    _rgba_mask_png,
    export_run,
    export_synchronized_comparison,
    pinned_blueprint,
    pinned_comparison_blueprint,
)
from battle.fixtures import synthetic_run_manifest
from battle.schemas import ClockName, TrackerSlotDiagnostic


def test_annotation_context_serializes_a_transparent_unlabeled_background() -> None:
    context = _annotation_context({})

    background = context.context.as_arrow_array().to_pylist()[0][0]["class_description"]["info"]

    assert background == {"id": 0, "label": "background", "color": 0}


def test_exporter_writes_synthetic_rrd_without_media_input(tmp_path) -> None:
    manifest = synthetic_run_manifest()
    output = export_run(manifest, tmp_path / "fixture.rrd")

    assert output.exists()
    assert output.stat().st_size > 0


def test_blueprint_is_pinned_for_fixture_clip() -> None:
    blueprint = pinned_blueprint(synthetic_run_manifest().clip.clip_id)

    assert blueprint is not None


def test_comparison_blueprint_is_pinned_for_both_views() -> None:
    blueprint = pinned_comparison_blueprint(
        synthetic_run_manifest().clip.clip_id,
        ego_view_id="ego-01",
        static_view_id="static-01",
        analysis_fps=30,
        ego_video_dimensions=(16, 8),
        static_video_dimensions=(20, 10),
    )

    assert blueprint is not None


def test_video_export_embeds_time_aligned_frames_in_the_observation_view(tmp_path: Path) -> None:
    """Exercise the RRD structure that a viewer needs to show an overlay."""
    video_path = tmp_path / "input.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=16x8:r=30",
            "-frames:v",
            "3",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ],
        check=True,
    )
    manifest = synthetic_run_manifest()
    clip = manifest.clip.model_copy(update={"views": ("ego-e4",)})
    observations = tuple(
        observation.model_copy(update={"view_id": "ego-e4"})
        for observation in manifest.observations
    )
    manifest = manifest.model_copy(update={"clip": clip, "observations": observations})
    stripe = np.zeros((8, 16), dtype=np.uint8)
    stripe[:, 2:4] = 255
    for observation in observations:
        for object_ in observation.objects:
            assert object_.mask is not None
            mask_path = tmp_path / object_.mask.uri
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(stripe).save(mask_path)
    output = export_run(
        manifest,
        tmp_path / "video.rrd",
        video_path=video_path,
        video_dimensions=(16, 8),
        mask_artifact_root=tmp_path,
    )

    printed = subprocess.run(
        [str(Path(sys.executable).with_name("rerun")), "rrd", "print", "-vvv", str(output)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    assert "/views/ego-e4/video" in printed
    assert "VideoFrameReference:timestamp" in printed
    assert "/views/ego-e4/masks/" in printed
    assert "EncodedImage:blob" in printed
    assert "/views/ego-e4/segmentation" in printed
    assert "SegmentationImage:buffer" in printed
    assert "AnnotationContext:context" in printed
    assert "synthetic screwdriver" in printed
    assert "/hands/landmarks" in printed
    assert "/hands/skeletons" in printed
    assert "/hands/boxes" in printed
    assert "/hand_metrics/count" in printed
    assert "ego-e4 video and detections" in printed
    assert "make_active: true" in printed
    assert "/frame_counter" in printed
    assert "Frame count" in printed
    assert "analysis frame" in printed


def test_comparison_export_embeds_synchronized_ego_and_static_views(tmp_path: Path) -> None:
    video_path = tmp_path / "input.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=16x8:r=30",
            "-frames:v",
            "3",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ],
        check=True,
    )
    ego_manifest = synthetic_run_manifest()
    static_manifest = ego_manifest.model_copy(
        update={
            "observations": tuple(
                observation.model_copy(update={"view_id": "static-01"})
                for observation in ego_manifest.observations
            )
        }
    )
    output = export_synchronized_comparison(
        ego_manifest,
        static_manifest,
        tmp_path / "comparison.rrd",
        ego_video_path=video_path,
        ego_video_dimensions=(16, 8),
        ego_asset_reference=ego_manifest.clip.asset,
        ego_mask_artifact_root=None,
        static_video_path=video_path,
        static_video_dimensions=(16, 8),
        static_asset_reference=ego_manifest.clip.asset,
        static_mask_artifact_root=None,
        static_label="existing static model outputs",
        ego_label="selected ego model outputs",
    )

    printed = subprocess.run(
        [str(Path(sys.executable).with_name("rerun")), "rrd", "print", "-vvv", str(output)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    assert "/views/ego-01/video" in printed
    assert "/views/static-01/video" in printed
    assert "ego-01: selected ego model outputs" in printed
    assert "static-01: existing static model outputs" in printed
    assert "analysis_time" in printed
    assert "/views/static-01/hands/landmarks" in printed
    assert "/views/static-01/hands/skeletons" in printed
    assert "/views/static-01/hands/boxes" in printed
    assert "/views/static-01/hand_metrics/count" in printed
    assert "static MediaPipe hand detections" in printed
    assert "/frame_counter" in printed
    assert "Frame count" in printed


def test_tracker_diagnostics_are_logged_even_when_a_slot_is_dropped(tmp_path: Path) -> None:
    """The trace must stay gapless across the frames where tracking actually fails."""
    manifest = synthetic_run_manifest()
    observations = []
    for index, observation in enumerate(manifest.observations):
        lost = index == 1
        observations.append(
            observation.model_copy(
                update={
                    "objects": () if lost else observation.objects,
                    "tracker_diagnostics": (
                        TrackerSlotDiagnostic(
                            object_id="tool-1",
                            label="synthetic screwdriver",
                            multiplex_slot=0,
                            object_score=-4.5 if lost else 11.25,
                            iou_prediction=0.0 if lost else 0.93,
                            active=not lost,
                        ),
                    ),
                }
            )
        )
    manifest = manifest.model_copy(update={"observations": tuple(observations)})

    output = export_run(manifest, tmp_path / "diagnostics.rrd")
    printed = subprocess.run(
        [str(Path(sys.executable).with_name("rerun")), "rrd", "print", "-vvv", str(output)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    assert "tracker_diagnostics/object_score/tool-1" in printed
    assert "tracker_diagnostics/iou_prediction/tool-1" in printed
    assert "tracker_diagnostics/object_score/lost_threshold" in printed
    assert "Object score (lost at or below 0)" in printed
    assert "Predicted mask IoU (self-estimate)" in printed


def test_frame_counter_reports_every_analysis_frame_index(tmp_path: Path) -> None:
    """The readout must name the active frame and the clip's frame total."""
    manifest = synthetic_run_manifest()
    output = export_run(manifest, tmp_path / "counter.rrd")

    printed = subprocess.run(
        [str(Path(sys.executable).with_name("rerun")), "rrd", "print", "-vvv", str(output)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    total = len(manifest.observations)
    for observation in manifest.observations:
        assert f"analysis frame **{observation.analysis_frame_index}** / {total - 1}" in printed
    assert f"({total} frames)" in printed
    assert "text/markdown" in printed


def test_rgba_mask_png_is_a_transparent_cutout_in_the_object_color() -> None:
    """Alpha is what lets the mask overlay the video without veiling the background."""
    binary_mask = np.zeros((4, 6), dtype=bool)
    binary_mask[1:3, 2:5] = True

    with Image.open(io.BytesIO(_rgba_mask_png(binary_mask, (80, 180, 255)))) as decoded:
        assert decoded.mode == "RGBA"
        rgba = np.asarray(decoded)

    assert rgba.shape == (4, 6, 4)
    assert tuple(rgba[2, 3]) == (80, 180, 255, 255)
    assert tuple(rgba[0, 0]) == (0, 0, 0, 0)
    assert int((rgba[..., 3] > 0).sum()) == int(binary_mask.sum())


_CHUNK_HEADER = re.compile(r"^Chunk\(\S+\) with (\d+) rows? \([^)]*\) - (/\S+) - ", re.MULTILINE)


def _row_counts_by_entity(rrd_path: Path) -> dict[str, int]:
    """Sum logged rows per entity from the chunk headers `rrd print` emits."""
    printed = subprocess.run(
        [str(Path(sys.executable).with_name("rerun")), "rrd", "print", str(rrd_path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    counts: dict[str, int] = {}
    for rows, entity in _CHUNK_HEADER.findall(printed):
        counts[entity] = counts.get(entity, 0) + int(rows)
    return counts


def test_masks_are_logged_every_frame_and_segmentation_once_per_second(
    tmp_path: Path,
) -> None:
    """Per-object masks carry the overlay at full rate; the class image is a sparse record."""
    manifest = synthetic_run_manifest()
    frame_indices = [o.analysis_frame_index for o in manifest.observations]
    assert len(frame_indices) > 1, "fixture must span several frames for a cadence test"
    object_ids: set[str] = set()
    for observation in manifest.observations:
        for object_ in observation.objects:
            assert object_.mask is not None
            object_ids.add(object_.object_id)
            mask_path = tmp_path / object_.mask.uri
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.full((8, 16), 255, dtype=np.uint8)).save(mask_path)

    output = export_run(manifest, tmp_path / "cadence.rrd", mask_artifact_root=tmp_path)
    counts = _row_counts_by_entity(output)
    view_root = f"/world/{manifest.clip.clip_id}/views/{manifest.observations[0].view_id}"

    for object_id in object_ids:
        assert counts[f"{view_root}/masks/{object_id}"] == len(frame_indices)
    analysis_fps = manifest.clip.timing.clocks.fps_for(ClockName.ANALYSIS)
    expected_segmentation_rows = sum(1 for i in frame_indices if i % analysis_fps == 0)
    assert 0 < expected_segmentation_rows < len(frame_indices)
    assert counts[f"{view_root}/segmentation"] == expected_segmentation_rows


def _store_id(rrd_path: Path) -> tuple[str, str]:
    """Return the recording's (application id, recording id)."""
    printed = subprocess.run(
        [str(Path(sys.executable).with_name("rerun")), "rrd", "print", str(rrd_path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    recording = printed.split("StoreId(", 1)[1]
    quoted = recording.split('"')
    return quoted[1], quoted[3]


def _application_id(rrd_path: Path) -> str:
    return _store_id(rrd_path)[0]


def test_runs_of_the_same_clip_exported_in_one_process_stay_separate_recordings(
    tmp_path: Path,
) -> None:
    """Random recording ids are only unique per process.

    Re-exporting several runs from one script gave them one id, and the viewer merged them
    into a single recording with duplicated rows. Deriving the id from the run id keeps
    them apart regardless of how many are exported together.
    """
    first = synthetic_run_manifest()
    second = first.model_copy(update={"run_id": f"{first.run_id}-again"})

    first_app, first_recording = _store_id(export_run(first, tmp_path / "first.rrd"))
    second_app, second_recording = _store_id(export_run(second, tmp_path / "second.rrd"))

    assert first_app == second_app, "same clip must share a blueprint"
    assert first_recording == first.run_id
    assert second_recording == second.run_id
    assert first_recording != second_recording


def test_recordings_of_different_clips_do_not_share_an_application_id(tmp_path: Path) -> None:
    """Blueprints are keyed by application id and anchored to a clip-specific entity root.

    Sharing an id across clips lets one clip's blueprint activate for all of them, so views
    resolve against entity paths the other recordings never logged and render empty. That
    only shows up when several recordings are opened together, so it is asserted here.
    """
    first = synthetic_run_manifest()
    renamed_clip = first.clip.model_copy(update={"clip_id": f"{first.clip.clip_id}-at-60fps"})
    second = first.model_copy(update={"clip": renamed_clip})

    first_id = _application_id(export_run(first, tmp_path / "first.rrd"))
    second_id = _application_id(export_run(second, tmp_path / "second.rrd"))

    assert first.clip.clip_id in first_id
    assert second.clip.clip_id in second_id
    assert first_id != second_id
