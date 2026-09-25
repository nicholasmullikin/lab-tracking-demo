"""Pure-function checks for the SAM3 worker's FineBio modes (Sep 24, plan `p3-worker`).

The worker is loaded by file path so the same tests run under the battle interpreter (no
torch; the tensor cases skip) and under the MuggledSAM interpreter::

    /home/nick/.pyenv/versions/muggled_sam/bin/python -m pytest tests/test_worker_finebio_modes.py

Covered: the per-frame box stream (parsing and validation), the correction schedule with
`prompt_box` entries and the two provenance kinds, per-slot start-frame bookkeeping, the tau
memory-write hook beside the Sep 18 gate, and what joins the stream identity.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

WORKER_PATH = Path(__file__).resolve().parents[1] / "src" / "battle" / "muggled_worker.py"
TODAY = "replace_prompt_memory_and_reset_frame_memory"
APPEND = "append_prompt_memory_and_reset_frame_memory"


def _load_worker():
    spec = importlib.util.spec_from_file_location("battle_muggled_worker_finebio", WORKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


worker = _load_worker()


def _stream(*records: dict) -> str:
    return "\n".join(json.dumps(record) for record in records) + "\n"


def _box(slot: int, label: str, **overrides: object) -> dict:
    return {
        "slot": slot,
        "label": label,
        "box_xyxy_px": [10, 20, 110, 220],
        "score": 0.5,
        "source": "finebio_dino",
        **overrides,
    }


# --- box stream --------------------------------------------------------------------------


def test_box_stream_parses_pixel_and_normalised_boxes_and_keeps_slot_labels() -> None:
    text = _stream(
        {"frame_index": 0, "boxes": [_box(0, "plate"), _box(1, "tube", score=None)]},
        {"frame_index": 2, "boxes": []},
        {
            "frame_index": 1,
            "boxes": [
                {
                    "slot": 1,
                    "label": "tube",
                    "box_xyxy_norm": [0.1, 0.2, 0.3, 0.4],
                    "source": "track_reproject",
                }
            ],
        },
    )

    frames, labels = worker.parse_box_stream("\n" + text + "\n\n")

    assert labels == ("plate", "tube")
    assert sorted(frames) == [0, 1, 2]
    assert frames[2] == []
    assert frames[0][0]["box_xyxy_px"] == [10.0, 20.0, 110.0, 220.0]
    assert frames[0][0]["box_xyxy_norm"] is None
    assert frames[0][1]["score"] is None and frames[0][1]["source"] == "finebio_dino"
    assert frames[1][0]["box_xyxy_norm"] == [0.1, 0.2, 0.3, 0.4]
    assert frames[1][0]["box_xyxy_px"] is None and frames[1][0]["score"] is None


def test_box_stream_slots_are_sorted_within_a_frame_and_an_empty_stream_has_no_labels() -> None:
    frames, labels = worker.parse_box_stream(
        _stream({"frame_index": 3, "boxes": [_box(1, "b"), _box(0, "a")]})
    )
    assert [box["slot"] for box in frames[3]] == [0, 1]
    assert labels == ("a", "b")
    assert worker.parse_box_stream("") == ({}, ())


@pytest.mark.parametrize(
    ("text", "message"),
    (
        ("{not json", "line 1 is not JSON"),
        (_stream([1, 2]), "must be a JSON object"),
        (_stream({"frame_index": -1, "boxes": []}), "frame_index must be an int >= 0"),
        (_stream({"frame_index": True, "boxes": []}), "frame_index must be an int >= 0"),
        (_stream({"frame_index": 0, "boxes": {}}), "boxes must be a list"),
        (
            _stream({"frame_index": 0, "boxes": []}, {"frame_index": 0, "boxes": []}),
            "frame 0 repeats",
        ),
        (_stream({"frame_index": 0, "boxes": [_box(0, "a"), _box(0, "a")]}), "slot 0 repeats"),
        (
            _stream(
                {"frame_index": 0, "boxes": [_box(0, "a")]},
                {"frame_index": 1, "boxes": [_box(0, "b")]},
            ),
            "labelled 'b' but was 'a'",
        ),
        (_stream({"frame_index": 0, "boxes": [_box(1, "a")]}), "contiguous from zero"),
        (_stream({"frame_index": 0, "boxes": [_box(0, "")]}), "label must be a non-empty"),
        (_stream({"frame_index": 0, "boxes": [_box(-1, "a")]}), "slot must be an int >= 0"),
        (
            _stream({"frame_index": 0, "boxes": [_box(0, "a", box_xyxy_norm=[0, 0, 1, 1])]}),
            "exactly one of box_xyxy_px / box_xyxy_norm",
        ),
        (
            _stream({"frame_index": 0, "boxes": [{"slot": 0, "label": "a"}]}),
            "exactly one of box_xyxy_px / box_xyxy_norm",
        ),
        (
            _stream({"frame_index": 0, "boxes": [_box(0, "a", box_xyxy_px=[10, 20, 5, 220])]}),
            "x1 > x0",
        ),
        (
            _stream({"frame_index": 0, "boxes": [_box(0, "a", box_xyxy_px=[10, 20, 30])]}),
            "four numbers",
        ),
        (
            _stream(
                {
                    "frame_index": 0,
                    "boxes": [{"slot": 0, "label": "a", "box_xyxy_norm": [0.0, 0.0, 1.5, 1.0]}],
                }
            ),
            "inside the unit square",
        ),
        (_stream({"frame_index": 0, "boxes": [_box(0, "a", score="high")]}), "score must be"),
        (_stream({"frame_index": 0, "boxes": [_box(0, "a", source="")]}), "source must be"),
    ),
)
def test_box_stream_validation_names_the_line_and_the_fault(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        worker.parse_box_stream(text)


def test_box_stream_rejects_non_finite_coordinates() -> None:
    text = '{"frame_index": 0, "boxes": [{"slot": 0, "label": "a", "box_xyxy_px": [0, 0, NaN, 1]}]}'
    with pytest.raises(ValueError, match="finite"):
        worker.parse_box_stream(text)


# --- box prompts -> normalised boxes ------------------------------------------------------


def test_prompt_box_xyxy_normalises_pixels_and_clamps_to_the_frame() -> None:
    shape = (100, 200)
    assert worker.prompt_box_xyxy({"box_xyxy_px": [20, 10, 220, 60]}, shape) == (
        0.1,
        0.1,
        1.0,
        0.6,
    )
    assert worker.prompt_box_xyxy({"prompt_box_xyxy_px": [-5, 0, 100, 50]}, shape) == (
        0.0,
        0.0,
        0.5,
        0.5,
    )
    assert worker.prompt_box_xyxy(
        {"prompt_box": {"x": 0.25, "y": 0.5, "width": 0.5, "height": 0.25}}, shape
    ) == (0.25, 0.5, 0.75, 0.75)
    assert worker.prompt_box_xyxy({"box_xyxy_norm": [0.1, 0.2, 0.3, 0.4]}, shape) == (
        0.1,
        0.2,
        0.3,
        0.4,
    )
    with pytest.raises(ValueError, match="outside the frame"):
        worker.prompt_box_xyxy({"box_xyxy_px": [250, 10, 300, 60]}, shape)
    with pytest.raises(ValueError, match="no box prompt"):
        worker.prompt_box_xyxy({"mask_path": "m.png"}, shape)


def test_normalised_box_record_never_overshoots_the_unit_square() -> None:
    record = worker.normalised_box_record((0.1, 0.7, 1.0, 1.0))
    assert record["x"] + record["width"] <= 1.0
    assert record["y"] + record["height"] <= 1.0
    assert math.isclose(record["width"], 0.9, rel_tol=1e-12)
    for x0 in (0.1, 0.3, 0.7, 0.9, 0.123456789):
        record = worker.normalised_box_record((x0, 0.0, 1.0, 1.0))
        assert record["x"] + record["width"] <= 1.0


# --- schedule: prompt_box corrections, provenance kinds, per-slot start frames ------------


def _schedule(**overrides: object) -> dict:
    payload = {
        "memory_semantics": TODAY,
        "seeds": [
            {"target": "plate", "initial_multiplex_slot": 0},
            {"target": "tube", "initial_multiplex_slot": 1},
        ],
        "corrections": [],
    }
    payload.update(overrides)
    return payload


def test_existing_mask_schedules_load_exactly_as_before() -> None:
    payload = _schedule(
        corrections=[
            {
                "frame_index": 900,
                "multiplex_slot": 0,
                "target": "plate",
                "mask_path": "/m.png",
                "mask_sha256": "0" * 64,
                "selected_by": "agent",
            }
        ]
    )
    grouped = worker._corrections_by_frame(payload, ("plate", "tube"))
    assert grouped == {900: payload["corrections"]}
    assert worker.slot_start_frames(payload) == {}


def test_box_corrections_need_one_of_the_two_provenance_kinds() -> None:
    box = {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4}
    for kind in ("detector_reseed", "track_reproject"):
        payload = _schedule(
            corrections=[
                {
                    "frame_index": 12,
                    "multiplex_slot": 1,
                    "target": "tube",
                    "prompt_box": box,
                    "selected_by": kind,
                }
            ]
        )
        assert list(worker._corrections_by_frame(payload, ("plate", "tube"))) == [12]
    for kind in ("human", "agent", None, "detector"):
        payload = _schedule(
            corrections=[
                {
                    "frame_index": 12,
                    "multiplex_slot": 1,
                    "target": "tube",
                    "prompt_box": box,
                    "selected_by": kind,
                }
            ]
        )
        with pytest.raises(ValueError, match="selected_by one of detector_reseed"):
            worker._corrections_by_frame(payload, ("plate", "tube"))


def test_a_pixel_box_correction_is_accepted_and_mixed_prompts_are_refused() -> None:
    pixel = {
        "frame_index": 3,
        "multiplex_slot": 0,
        "target": "plate",
        "prompt_box_xyxy_px": [10, 20, 110, 220],
        "selected_by": "track_reproject",
    }
    assert list(worker._corrections_by_frame(_schedule(corrections=[pixel]), ("plate", "tube")))
    with pytest.raises(ValueError, match="not both"):
        worker._corrections_by_frame(
            _schedule(
                corrections=[{**pixel, "prompt_box": {"x": 0, "y": 0, "width": 0.5, "height": 0.5}}]
            ),
            ("plate", "tube"),
        )
    with pytest.raises(ValueError, match="mask or a box, not both"):
        worker._corrections_by_frame(
            _schedule(corrections=[{**pixel, "mask_path": "/m.png", "mask_sha256": "0" * 64}]),
            ("plate", "tube"),
        )


@pytest.mark.parametrize(
    "box",
    (
        {"x": 0.1, "y": 0.2, "width": 0.3},
        {"x": 0.1, "y": 0.2, "width": 0.0, "height": 0.4},
        {"x": 0.9, "y": 0.2, "width": 0.3, "height": 0.4},
        {"x": -0.1, "y": 0.2, "width": 0.3, "height": 0.4},
        {"x": "a", "y": 0.2, "width": 0.3, "height": 0.4},
        [0.1, 0.2, 0.3, 0.4],
    ),
)
def test_malformed_prompt_boxes_are_refused(box: object) -> None:
    payload = _schedule(
        corrections=[
            {
                "frame_index": 12,
                "multiplex_slot": 1,
                "target": "tube",
                "prompt_box": box,
                "selected_by": "detector_reseed",
            }
        ]
    )
    with pytest.raises(ValueError, match="prompt_box"):
        worker._corrections_by_frame(payload, ("plate", "tube"))


def test_a_mid_stream_seed_is_applied_at_its_start_frame_through_the_correction_path() -> None:
    payload = _schedule(
        memory_semantics=APPEND,
        seeds=[
            {"target": "plate", "initial_multiplex_slot": 0},
            {
                "target": "tube",
                "initial_multiplex_slot": 1,
                "start_frame": 45,
                "prompt_box_xyxy_px": [1, 2, 3, 4],
                "selected_by": "detector_reseed",
            },
        ],
        corrections=[
            {
                "frame_index": 45,
                "multiplex_slot": 0,
                "target": "plate",
                "prompt_box": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2},
                "selected_by": "track_reproject",
            }
        ],
    )

    grouped = worker._corrections_by_frame(payload, ("plate", "tube"), memory_semantics=APPEND)

    assert worker.slot_start_frames(payload) == {1: 45}
    assert list(grouped) == [45]
    assert [entry["multiplex_slot"] for entry in grouped[45]] == [0, 1]
    seed_entry = grouped[45][1]
    assert seed_entry["seed_start"] is True
    assert seed_entry["frame_index"] == 45 and seed_entry["selected_by"] == "detector_reseed"
    assert seed_entry["prompt_box_xyxy_px"] == [1, 2, 3, 4]


def test_a_mid_stream_seed_defaults_its_provenance_and_needs_a_prompt() -> None:
    seeds = [
        {"target": "plate", "initial_multiplex_slot": 0},
        {"target": "tube", "initial_multiplex_slot": 1, "start_frame": 5},
    ]
    with pytest.raises(ValueError, match="needs a prompt_box or a verified mask"):
        worker._corrections_by_frame(_schedule(seeds=seeds), ("plate", "tube"))
    seeds[1].update({"mask_path": "/m.png", "mask_sha256": "0" * 64})
    grouped = worker._corrections_by_frame(_schedule(seeds=seeds), ("plate", "tube"))
    assert grouped[5][0]["selected_by"] == "detector_reseed"
    with pytest.raises(ValueError, match="must not be negative"):
        worker.slot_start_frames({"seeds": [{"start_frame": -1}]})


def test_a_start_frame_cannot_collide_with_a_correction_of_the_same_slot() -> None:
    payload = _schedule(
        seeds=[
            {"target": "plate", "initial_multiplex_slot": 0},
            {
                "target": "tube",
                "initial_multiplex_slot": 1,
                "start_frame": 5,
                "prompt_box_xyxy_px": [1, 2, 3, 4],
            },
        ],
        corrections=[
            {
                "frame_index": 5,
                "multiplex_slot": 1,
                "target": "tube",
                "prompt_box_xyxy_px": [1, 2, 3, 4],
                "selected_by": "track_reproject",
            }
        ],
    )
    with pytest.raises(ValueError, match="cannot also carry a later correction"):
        worker._corrections_by_frame(payload, ("plate", "tube"))


def test_has_box_prompt_reads_both_box_forms() -> None:
    assert worker._has_box_prompt({"prompt_box": {"x": 0, "y": 0, "width": 1, "height": 1}})
    assert worker._has_box_prompt({"prompt_box_xyxy_px": [0, 0, 1, 1]})
    assert not worker._has_box_prompt({"mask_path": "m.png"})


# --- the tau hook -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("score", "tau", "expected"),
    (
        (0.2, None, True),
        (-3.0, None, True),
        (0.5, 0.5, True),
        (0.49, 0.5, False),
        (12.0, 0.5, True),
        (0.1, 0.0, True),
        (-0.1, 0.0, False),
    ),
)
def test_memory_write_allowed_is_the_score_test_alone(
    score: float, tau: float | None, expected: bool
) -> None:
    assert worker.memory_write_allowed(score, tau) is expected


def test_tau_joins_the_policy_record_only_when_set() -> None:
    default = worker.memory_policy_from_args(argparse.Namespace())
    assert "memory_write_min_score" not in default
    assert worker.memory_policy_is_default(default)
    with_tau = worker.memory_policy_from_args(argparse.Namespace(memory_write_min_score=0.5))
    assert with_tau["memory_write_min_score"] == 0.5
    assert with_tau["memory_gate"] == "off"
    assert not worker.memory_policy_is_default(with_tau)


def test_tau_gates_on_score_alone_with_the_sep18_gate_off() -> None:
    torch = pytest.importorskip("torch")
    policy = worker.memory_policy_from_args(argparse.Namespace(memory_write_min_score=0.5))
    scores = torch.tensor([2.0, 0.5, 0.49, -1.0])
    # IoU 0.1 and a large area would fail the Sep 18 gate; the tau hook ignores them.
    ious = torch.tensor([[0.1], [0.1], [0.9], [0.9]])

    gated, written, reasons, history = worker.memory_gate(
        scores, ious, [0.9, 0.9, 0.0, 0.0], [10_000, 10, 10, 10], [], policy
    )

    assert written == [True, True, False, False]
    assert reasons == ["ok", "ok", "low_object_score", "low_object_score"]
    assert gated.tolist() == [2.0, 0.5, -1.0, -1.0]
    assert scores.tolist() == pytest.approx([2.0, 0.5, 0.49, -1.0]), "raw scores are untouched"
    assert history == [[], [], [], []], "the area history is only kept for the Sep 18 gate"


def test_tau_and_the_sep18_gate_compose_with_tau_judged_first() -> None:
    torch = pytest.importorskip("torch")
    policy = worker.memory_policy_from_args(
        argparse.Namespace(memory_write_min_score=1.0, memory_gate="on")
    )
    scores = torch.tensor([0.8, 2.0, 2.0])
    ious = torch.tensor([[0.9], [0.2], [0.9]])

    _, written, reasons, _ = worker.memory_gate(
        scores, ious, [0.0, 0.0, 0.0], [10, 10, 10], [], policy
    )

    assert reasons == ["low_object_score", "low_iou", "warmup"]
    assert written == [False, False, True]


def test_gate_off_and_no_tau_is_the_identity() -> None:
    torch = pytest.importorskip("torch")
    scores = torch.tensor([0.1, -0.2])
    gated, written, reasons, _ = worker.memory_gate(
        scores, None, [0.0, 0.0], [1, 1], [], worker.memory_policy_from_args(argparse.Namespace())
    )
    assert gated is scores and written == [True, True] and reasons == ["ok", "ok"]


def test_unseeded_slots_are_memorised_as_absent_and_say_so() -> None:
    torch = pytest.importorskip("torch")
    scores = torch.tensor([3.0, 4.0, 5.0])

    gated, diagnostics = worker._mark_unseeded(scores, [False, True, False], [], 3)

    assert gated.tolist() == [3.0, -1.0, 5.0]
    assert scores.tolist() == [3.0, 4.0, 5.0]
    assert diagnostics[1] == {"memory_written": False, "memory_gate_reason": "unseeded"}
    assert diagnostics[0] == {"memory_written": True, "memory_gate_reason": "ok"}
    existing = [{"contested_fraction": 0.0, "memory_written": True, "memory_gate_reason": "ok"}] * 3
    _, merged = worker._mark_unseeded(scores, [True, False, False], existing, 3)
    assert merged[0]["contested_fraction"] == 0.0 and merged[0]["memory_gate_reason"] == "unseeded"
    assert merged[1]["memory_gate_reason"] == "ok"


def test_the_seed_batch_switches_to_logits_only_when_a_slot_is_empty() -> None:
    numpy = pytest.importorskip("numpy")
    full = numpy.array([[[True, False], [False, True]], [[False, True], [True, False]]])
    assert worker._prompt_mask_batch(full) is full
    torch = pytest.importorskip("torch")
    with_empty = numpy.array([[[True, False], [False, False]], [[False, False], [False, False]]])
    logits = worker._prompt_mask_batch(with_empty)
    assert isinstance(logits, torch.Tensor) and logits.shape == (2, 2, 2)
    assert logits[0, 0, 0] == 1024.0 and logits[1].max() == -1024.0


# --- stream identity and CLI --------------------------------------------------------------


def _args(tmp_path: Path, **overrides: object) -> argparse.Namespace:
    video = tmp_path / "input.mp4"
    video.write_bytes(b"proxy")
    model = tmp_path / "sam3.pt"
    model.write_bytes(b"weights")
    defaults = {
        "video": str(video),
        "model": str(model),
        "view_id": "fpv",
        "source_offset_seconds": 60.0,
        "analysis_fps": 30.0,
        "max_side_length": 1280,
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


def test_tau_start_frame_and_the_box_stream_hash_join_the_stream_identity(tmp_path: Path) -> None:
    baseline = worker.stream_identity(_args(tmp_path), ("plate",))
    assert worker.stream_identity(_args(tmp_path, start_frame=0), ("plate",)) == baseline
    assert worker.stream_identity(_args(tmp_path, memory_write_min_score=None), ("plate",)) == (
        baseline
    )
    assert worker.stream_identity(_args(tmp_path, start_frame=1798), ("plate",)) != baseline
    assert worker.stream_identity(_args(tmp_path, memory_write_min_score=0.5), ("plate",)) != (
        baseline
    )
    stream = tmp_path / "boxes.jsonl"
    stream.write_text('{"frame_index": 0, "boxes": []}\n')
    with_stream = worker.stream_identity(
        _args(tmp_path, prompt_mode="box_stream", box_stream=str(stream)), ("plate",)
    )
    assert with_stream != baseline
    stream.write_text('{"frame_index": 1, "boxes": []}\n')
    assert (
        worker.stream_identity(
            _args(tmp_path, prompt_mode="box_stream", box_stream=str(stream)), ("plate",)
        )
        != with_stream
    ), "a changed box stream is a different stream"
    assert worker.stream_identity(
        _args(
            tmp_path,
            prompt_mode="box_stream",
            box_stream=str(stream),
            other_slots_as_negatives=True,
        ),
        ("plate",),
    ) != worker.stream_identity(
        _args(tmp_path, prompt_mode="box_stream", box_stream=str(stream)), ("plate",)
    )


def _cli(*extra: str) -> list[str]:
    return [
        "--run-directory",
        "/tmp/run",
        "--video",
        "/tmp/v.mp4",
        "--view-id",
        "fpv",
        "--source-offset-seconds",
        "0",
        "--model",
        "/tmp/m.pt",
        *extra,
    ]


def test_box_stream_mode_needs_its_file_and_refuses_a_checkpoint_resume() -> None:
    args = worker.parse_args(_cli("--prompt-mode", "box_stream", "--box-stream", "/tmp/b.jsonl"))
    assert args.box_stream == "/tmp/b.jsonl" and args.start_frame == 0
    assert args.memory_write_min_score is None
    with pytest.raises(SystemExit):
        worker.parse_args(_cli("--prompt-mode", "box_stream"))
    with pytest.raises(SystemExit):
        worker.parse_args(_cli("--box-stream", "/tmp/b.jsonl"))
    with pytest.raises(SystemExit):
        worker.parse_args(
            _cli(
                "--prompt-mode",
                "box_stream",
                "--box-stream",
                "/tmp/b.jsonl",
                "--resume-from-checkpoint",
                "/tmp/c.pt",
            )
        )
    with pytest.raises(SystemExit):
        worker.parse_args(_cli("--start-frame", "-1"))
    assert worker.parse_args(_cli("--memory-write-min-score", "0.5")).memory_write_min_score == 0.5


def test_box_stream_run_is_blocked_without_its_inputs_and_records_the_condition(
    tmp_path: Path,
) -> None:
    stream = tmp_path / "boxes.jsonl"
    stream.write_text(
        _stream({"frame_index": 0, "boxes": [_box(0, "plate")]}, {"frame_index": 7, "boxes": []})
    )
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    args = worker.parse_args(
        [
            "--run-directory",
            str(run_directory),
            "--video",
            str(tmp_path / "missing.mp4"),
            "--view-id",
            "fpv",
            "--source-offset-seconds",
            "0",
            "--model",
            str(tmp_path / "missing.pt"),
            "--prompt-mode",
            "box_stream",
            "--box-stream",
            str(stream),
            "--checkpoint-every",
            "50",
            "--gpu-guard",
            "strict",
        ]
    )

    assert worker.run_box_stream(args) == 2

    result = json.loads((run_directory / "worker_result.json").read_text())
    assert result["state"] == "blocked" and "does not exist" in result["reason"]
    settings = result["runtime_settings"]
    assert settings["mode"] == "box_stream"
    assert settings["concepts"] == ["plate"]
    assert settings["box_stream_prompted_frames"] == 1
    assert settings["box_stream_prompted_boxes"] == 1
    assert settings["other_slots_as_negatives"] is False
    assert settings["row_source"] == "sam3_decode"
    assert settings["ignored_checkpoint_flags"] == {"checkpoint_every": 50, "checkpoint_at": []}
    assert settings["checkpoints"].startswith("none")
    assert len(settings["box_stream_sha256"]) == 64


def test_box_stream_run_refuses_concepts_that_disagree_with_the_stream(tmp_path: Path) -> None:
    stream = tmp_path / "boxes.jsonl"
    stream.write_text(_stream({"frame_index": 0, "boxes": [_box(0, "plate")]}))
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    args = worker.parse_args(
        _cli(
            "--prompt-mode",
            "box_stream",
            "--box-stream",
            str(stream),
            "--concepts-json",
            '["tube"]',
        )
    )
    args.run_directory = str(run_directory)
    with pytest.raises(ValueError, match="does not match the box stream"):
        worker.run_box_stream(args)


def test_runtime_settings_describe_box_prompt_schedules_and_the_tau_hook() -> None:
    schedule = _schedule(
        memory_semantics=APPEND,
        seeds=[
            {"target": "plate", "initial_multiplex_slot": 0, "prompt_box_xyxy_px": [1, 2, 3, 4]},
            {
                "target": "tube",
                "initial_multiplex_slot": 1,
                "start_frame": 5,
                "prompt_box_xyxy_px": [1, 2, 3, 4],
                "selected_by": "detector_reseed",
            },
        ],
        corrections=[
            {
                "frame_index": 10,
                "multiplex_slot": 0,
                "target": "plate",
                "prompt_box": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2},
                "selected_by": "track_reproject",
            }
        ],
    )
    args = argparse.Namespace(
        prompt_mode="manual_seed_multiplexed_keyframes",
        preprocessing="original_bgr",
        manual_box_json=None,
        manual_seeds_json=None,
        hybrid_initialization_json=None,
        multi_keyframe_schedule_json=json.dumps(schedule),
        text_targets_json=None,
        max_frame_memory=4,
        prompt_memory_semantics="append",
        memory_write_min_score=0.5,
        start_frame=1798,
    )

    settings = worker._runtime_settings(args, ("plate", "tube"))

    record = settings["multi_keyframe_correction_schedule"]
    assert record["box_prompt_correction_frames"] == [10]
    assert record["correction_selected_by_kinds"] == ["detector_reseed", "track_reproject"]
    assert record["slot_start_frames"] == {"1": 5}
    assert record["box_prompt_seed_slots"] == [0, 1]
    assert settings["tracker_memory_policy"]["memory_write_min_score"] == 0.5
    assert settings["start_frame"] == 1798
    assert "box_prompt_api" in settings and "slot_start_frame_semantics" in settings
    plain = worker._runtime_settings(
        argparse.Namespace(
            **{**vars(args), "multi_keyframe_schedule_json": json.dumps(_schedule())}
        ),
        ("plate", "tube"),
    )
    assert "box_prompt_correction_frames" not in plain["multi_keyframe_correction_schedule"]
