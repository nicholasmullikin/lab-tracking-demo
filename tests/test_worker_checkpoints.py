from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from battle.muggled_worker import (
    _frame_list,
    checkpoint_frames,
    stream_identity,
)


def _args(tmp_path: Path, **overrides: object) -> argparse.Namespace:
    video = tmp_path / "input.mp4"
    video.write_bytes(b"proxy")
    model = tmp_path / "sam3.pt"
    model.write_bytes(b"weights")
    defaults = {
        "video": str(video),
        "model": str(model),
        "view_id": "static-c10379",
        "source_offset_seconds": 294.0,
        "analysis_fps": 30.0,
        "max_side_length": 720,
        "max_frame_memory": 4,
        "preprocessing": "original_bgr",
        "lower_percentile": 1.0,
        "upper_percentile": 99.0,
        "clahe_clip_limit": 2.0,
        "clahe_tile_grid_size": 8,
        "prompt_mode": "manual_seed_multiplexed_keyframes",
        "text_targets_json": None,
        "hybrid_initialization_json": None,
        "manual_box_json": None,
        "manual_seeds_json": None,
        "multi_keyframe_schedule_json": '{"seeds": []}',
    }
    return argparse.Namespace(**{**defaults, **overrides})


def test_correction_keyframes_are_always_checkpointed() -> None:
    frames = checkpoint_frames(
        every=0, correction_frames=(327, 1172), extra_frames=(), max_frames=1800
    )

    assert frames == {327, 1172}


def test_periodic_and_extra_frames_join_the_correction_keyframes() -> None:
    frames = checkpoint_frames(
        every=600, correction_frames=(327,), extra_frames=(1000,), max_frames=1800
    )

    assert frames == {327, 600, 1000, 1200}


def test_frames_outside_the_run_are_dropped() -> None:
    frames = checkpoint_frames(
        every=0, correction_frames=(0, 300, 1800, 5000), extra_frames=(-5,), max_frames=600
    )

    assert frames == {300}


def test_a_changed_stream_input_changes_its_identity(tmp_path: Path) -> None:
    baseline = stream_identity(_args(tmp_path), ("chassis",))

    assert stream_identity(_args(tmp_path), ("chassis",)) == baseline
    assert stream_identity(_args(tmp_path), ("interior",)) != baseline
    assert stream_identity(_args(tmp_path, max_side_length=504), ("chassis",)) != baseline
    assert stream_identity(_args(tmp_path, max_frame_memory=8), ("chassis",)) != baseline
    assert (
        stream_identity(_args(tmp_path, preprocessing="gray_p01_p99_clahe"), ("chassis",))
        != baseline
    )
    rescheduled = _args(tmp_path, multi_keyframe_schedule_json='{"seeds": [1]}')
    assert stream_identity(rescheduled, ("chassis",)) != baseline


def test_a_rewritten_video_changes_the_stream_identity(tmp_path: Path) -> None:
    args = _args(tmp_path)
    baseline = stream_identity(args, ("chassis",))

    Path(args.video).write_bytes(b"a different proxy")

    assert stream_identity(args, ("chassis",)) != baseline


def test_checkpoint_frame_lists_are_parsed_and_bounded() -> None:
    assert _frame_list("327, 1172,327") == (327, 1172)

    with pytest.raises(argparse.ArgumentTypeError, match="at least 1"):
        _frame_list("0,5")
