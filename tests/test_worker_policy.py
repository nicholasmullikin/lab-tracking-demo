"""Pure-function checks for the worker's slot exclusivity and score-gated memory policy.

The worker is loaded by file path so the same tests run under the battle interpreter (which
has no torch and skips the tensor cases) and under the MuggledSAM interpreter:

    /home/nick/.pyenv/versions/muggled_sam/bin/python -m pytest tests/test_worker_policy.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

WORKER_PATH = Path(__file__).resolve().parents[1] / "src" / "battle" / "muggled_worker.py"


def _load_worker():
    spec = importlib.util.spec_from_file_location("battle_muggled_worker_under_test", WORKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


worker = _load_worker()


def _policy(**overrides: object) -> dict[str, object]:
    return worker.memory_policy_from_args(argparse.Namespace(**overrides))


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


def test_the_default_policy_is_off_everywhere() -> None:
    policy = _policy()

    assert worker.memory_policy_is_default(policy)
    assert policy == {
        "slot_exclusivity": "off",
        "memory_gate": "off",
        "exclusivity_loser_logit": -8.0,
        "gate_min_object_score": 0.0,
        "gate_min_iou": 0.5,
        "gate_max_contested_fraction": 0.2,
        "gate_area_band": [0.5, 2.0],
        "gate_area_history_frames": 30,
    }
    assert not worker.memory_policy_is_default(_policy(slot_exclusivity="argmax"))
    assert not worker.memory_policy_is_default(_policy(memory_gate="on"))


def test_only_a_non_default_policy_changes_the_stream_identity(tmp_path: Path) -> None:
    baseline = worker.stream_identity(_args(tmp_path), ("chassis",))

    assert worker.stream_identity(_args(tmp_path, memory_gate="off"), ("chassis",)) == baseline
    assert worker.stream_identity(_args(tmp_path, memory_gate="on"), ("chassis",)) != baseline
    assert (
        worker.stream_identity(_args(tmp_path, slot_exclusivity="argmax"), ("chassis",)) != baseline
    )


def test_area_band_parsing_rejects_inverted_or_empty_bands() -> None:
    assert worker._area_band("0.5, 2") == (0.5, 2.0)
    with pytest.raises(argparse.ArgumentTypeError):
        worker._area_band("2,0.5")
    with pytest.raises(argparse.ArgumentTypeError):
        worker._area_band("0,1")


def test_gate_off_writes_every_slot_and_returns_the_scores_unchanged() -> None:
    torch = pytest.importorskip("torch")
    scores = torch.tensor([1.0, -0.2, 0.3])

    gated, written, reasons, history = worker.memory_gate(
        scores, None, [0.9, 0.0, 0.0], [10, 0, 5], [], _policy()
    )

    assert gated is scores
    assert written == [True, True, True]
    assert reasons == ["ok", "ok", "ok"]
    assert history == [[], [], []]


def _logits() -> tuple[object, object]:
    torch = pytest.importorskip("torch")
    masks = torch.full((3, 1, 2, 3), -4.0)
    # Slot 0 owns column 0, slot 1 owns column 1; column 2 is contested between them and the
    # higher logit differs per row.  Slot 2 claims everything but is absent (score <= 0).
    masks[0, 0, :, 0] = 3.0
    masks[1, 0, :, 1] = 3.0
    masks[0, 0, 0, 2] = 2.0
    masks[1, 0, 0, 2] = 1.0
    masks[0, 0, 1, 2] = 0.5
    masks[1, 0, 1, 2] = 2.5
    masks[2] = 5.0
    scores = torch.tensor([1.0, 1.0, -0.5])
    return masks, scores


def test_argmax_exclusivity_gives_contested_pixels_to_the_higher_logit() -> None:
    masks, scores = _logits()

    resolved, contested = worker.resolve_slot_exclusivity(masks, scores, mode="argmax")

    assert resolved[0, 0, 0, 2] == 2.0 and resolved[1, 0, 0, 2] == -8.0
    assert resolved[1, 0, 1, 2] == 2.5 and resolved[0, 0, 1, 2] == -8.0
    # Uncontested pixels keep their logits exactly.
    assert bool((resolved[0, 0, :, 0] == 3.0).all()) and bool((resolved[1, 0, :, 1] == 3.0).all())
    assert bool((resolved[0, 0, :, 1] == -4.0).all())
    # The absent slot neither competes nor is rewritten.
    assert bool((resolved[2] == 5.0).all())
    # Each present slot had 4 positive pixels, 2 of them contested.
    assert contested == pytest.approx([0.5, 0.5, 0.0])
    assert masks[1, 0, 0, 2] == 1.0, "the input tensor is left untouched"


def test_exclusivity_off_is_the_identity_but_still_measures_contest() -> None:
    masks, scores = _logits()

    resolved, contested = worker.resolve_slot_exclusivity(masks, scores, mode="off")

    assert resolved is masks
    assert contested == pytest.approx([0.5, 0.5, 0.0])


def test_exclusivity_keeps_losers_already_below_the_bound_and_handles_bfloat16() -> None:
    torch = pytest.importorskip("torch")
    masks = torch.full((2, 1, 1, 2), 1.0, dtype=torch.bfloat16)
    masks[0, 0, 0, 1] = 4.0
    masks[1, 0, 0, 0] = 4.0
    scores = torch.tensor([0.5, 0.5], dtype=torch.bfloat16)

    resolved, contested = worker.resolve_slot_exclusivity(masks, scores, mode="argmax")

    assert resolved.dtype == torch.bfloat16
    assert resolved[0, 0, 0, 0] == -8.0 and resolved[1, 0, 0, 0] == 4.0
    assert resolved[0, 0, 0, 1] == 4.0 and resolved[1, 0, 0, 1] == -8.0
    assert contested == [1.0, 1.0]


def test_exclusivity_rejects_unknown_modes_and_empty_batches() -> None:
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="unknown slot exclusivity mode"):
        worker.resolve_slot_exclusivity(torch.zeros((1, 1, 1, 1)), torch.ones(1), mode="max")
    empty, contested = worker.resolve_slot_exclusivity(
        torch.zeros((0, 1, 1, 1)), torch.zeros(0), mode="argmax"
    )
    assert empty.shape[0] == 0 and contested == []


def test_gate_reasons_follow_the_threshold_order() -> None:
    torch = pytest.importorskip("torch")
    policy = _policy(memory_gate="on", gate_area_history_frames=2)
    scores = torch.tensor([-0.1, 2.0, 2.0, 2.0, 2.0])
    ious = torch.tensor([[0.9], [0.2], [0.9], [0.9], [0.9]])
    history = [[100.0, 100.0], [100.0, 100.0], [100.0, 100.0], [100.0, 100.0], [100.0]]

    gated, written, reasons, updated = worker.memory_gate(
        scores, ious, [0.0, 0.0, 0.5, 0.0, 0.0], [100, 100, 100, 500, 100], history, policy
    )

    assert reasons == ["low_object_score", "low_iou", "contested", "area_jump", "warmup"]
    assert written == [False, False, False, False, True]
    assert gated.tolist() == [-1.0, -1.0, -1.0, -1.0, 2.0]
    assert scores.tolist() == pytest.approx([-0.1, 2.0, 2.0, 2.0, 2.0]), (
        "raw scores are left for diagnostics"
    )
    # Untrusted frames do not enter the area history; the warm-up slot's does.
    assert updated[:4] == history[:4]
    assert updated[4] == [100.0, 100.0]


def test_gate_area_band_uses_the_rolling_median_of_trusted_frames() -> None:
    torch = pytest.importorskip("torch")
    policy = _policy(memory_gate="on", gate_area_history_frames=3)
    scores = torch.tensor([1.0])
    history: list[list[float]] = []
    trace = []
    for area in (100, 100, 100, 150, 250, 40, 100):
        _, written, reasons, history = worker.memory_gate(
            scores, None, [0.0], [area], history, policy
        )
        trace.append((reasons[0], written[0], list(history[0])))

    assert [item[0] for item in trace] == [
        "warmup",
        "warmup",
        "warmup",
        "ok",
        "area_jump",
        "area_jump",
        "ok",
    ]
    assert trace[3][2] == [100.0, 100.0, 150.0]
    # Rejected areas never enter the history, so a later normal frame is judged against
    # the trusted median and accepted.
    assert trace[6][2] == [100.0, 150.0, 100.0]


def test_gate_pads_history_for_slots_it_has_not_seen() -> None:
    torch = pytest.importorskip("torch")
    _, written, reasons, history = worker.memory_gate(
        torch.tensor([1.0, 1.0]), None, [0.0, 0.0], [10, 20], [[5.0]], _policy(memory_gate="on")
    )

    assert written == [True, True] and reasons == ["warmup", "warmup"]
    assert history == [[5.0, 10.0], [20.0]]


def test_slot_areas_count_positive_logits_per_slot() -> None:
    torch = pytest.importorskip("torch")
    masks = torch.tensor([[[[1.0, -1.0], [2.0, 0.0]]], [[[-1.0, -1.0], [-1.0, -1.0]]]])

    assert worker._slot_areas(masks) == [2, 0]
