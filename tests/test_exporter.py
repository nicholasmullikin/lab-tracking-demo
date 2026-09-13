from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from battle.exporter import (
    _annotation_context,
    export_run,
    export_synchronized_comparison,
    pinned_blueprint,
    pinned_comparison_blueprint,
)
from battle.fixtures import synthetic_run_manifest


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
    mask_reference = observations[0].objects[0].mask
    assert mask_reference is not None
    mask_path = tmp_path / mask_reference.uri
    mask_path.parent.mkdir(parents=True)
    Image.fromarray(
        np.array(
            [
                [0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )
    ).save(mask_path)
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
    assert "/views/ego-e4/segmentation" in printed
    assert "SegmentationImage:buffer" in printed
    assert "AnnotationContext:context" in printed
    assert "synthetic screwdriver" in printed
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
    )

    printed = subprocess.run(
        [str(Path(sys.executable).with_name("rerun")), "rrd", "print", "-vvv", str(output)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    assert "/views/ego-01/video" in printed
    assert "/views/static-01/video" in printed
    assert "ego-01: ego manual-seed SAM3 baseline" in printed
    assert "static-01: existing static model outputs" in printed
    assert "analysis_time" in printed
    assert "/frame_counter" in printed
    assert "Frame count" in printed


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
