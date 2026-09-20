"""The battle-owned DAM4SAM wrapper, driven by a stub predictor (no torch, no weights)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import require_artifact
from PIL import Image

from battle import dam4sam_streaming as streaming
from battle.dam4sam_video import SAM2_CHECKPOINT_SHA256 as TINY_PIN_IN_SMOKE_MODULE

DAM4SAM_ROOT = Path("/home/nick/src/DAM4SAM")
FRAME_SHAPE = (12, 16)


class StubPredictor:
    """Records `add_new_mask` calls and answers with logits that are positive inside a box."""

    image_size = 1024

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.images_present_at_call: list[bool] = []

    def add_new_mask(self, *, inference_state: dict[str, Any], frame_idx: int, obj_id: int, mask):
        self.calls.append({"frame_idx": frame_idx, "obj_id": obj_id, "mask": np.asarray(mask)})
        self.images_present_at_call.append(frame_idx in inference_state["images"])
        logits = np.full((1, 1, *FRAME_SHAPE), -5.0, dtype=np.float32)
        logits[0, 0, 2:6, 3:9] = 4.0
        return frame_idx, [obj_id], logits


class StubBase:
    """Stands in for the upstream `DAM4SAMTracker`: state after `initialize` + `track` calls."""

    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - never reached
        raise AssertionError("the mixin must not call the upstream constructor")

    def initialize(self, image, init_mask):
        self.frame_index = 0
        self.object_sizes: list[int] = []
        self.last_added = -1
        self.img_width, self.img_height = image.size
        self.inference_state = {"images": {}, "obj_id_to_idx": {0: 0}}
        return {"pred_mask": np.asarray(init_mask, dtype=np.uint8)}

    def track(self, image):
        self.frame_index += 1
        self.object_sizes.append(7)
        return {"pred_mask": np.zeros(FRAME_SHAPE, dtype=np.uint8)}

    def _prepare_image(self, image):
        return ("prepared", image.size)


def _tracker(predictor: StubPredictor, **kwargs) -> Any:
    tracker_class = streaming.shared_predictor_tracker_class(base=StubBase)
    return tracker_class(predictor, **kwargs)


def _frame() -> Image.Image:
    return Image.new("RGB", (FRAME_SHAPE[1], FRAME_SHAPE[0]))


def _seed() -> np.ndarray:
    seed = np.zeros(FRAME_SHAPE, dtype=bool)
    seed[1:4, 1:4] = True
    return seed


def test_four_trackers_share_one_predictor_but_not_their_inference_state() -> None:
    predictor = StubPredictor()
    trackers = {name: _tracker(predictor) for name in ("chassis", "interior", "rear_body", "cabin")}
    for tracker in trackers.values():
        tracker.initialize(_frame(), _seed())

    assert all(tracker.predictor is predictor for tracker in trackers.values())
    states = [id(tracker.inference_state) for tracker in trackers.values()]
    assert len(set(states)) == 4
    assert all(tracker.input_image_size == 1024 for tracker in trackers.values())
    assert _tracker(predictor, input_image_size=1536).input_image_size == 1536
    assert type(trackers["chassis"]).__name__ == "SharedPredictorDAM4SAMTracker"
    assert issubclass(type(trackers["chassis"]), StubBase)


def test_correct_conditions_the_current_frame_without_moving_the_stream() -> None:
    predictor = StubPredictor()
    tracker = _tracker(predictor)
    tracker.initialize(_frame(), _seed())
    for _ in range(5):
        tracker.track(_frame())
    assert tracker.frame_index == 5 and tracker.object_sizes == [7] * 5
    correction = np.zeros(FRAME_SHAPE, dtype=bool)
    correction[2:6, 3:9] = True

    result = tracker.correct(_frame(), correction)

    assert tracker.frame_index == 5
    assert len(predictor.calls) == 1
    assert predictor.calls[0]["frame_idx"] == 5
    assert predictor.calls[0]["obj_id"] == 0
    assert np.array_equal(predictor.calls[0]["mask"], correction)
    assert predictor.images_present_at_call == [True]
    assert tracker.inference_state["images"] == {}
    assert result["pred_mask"].dtype == np.uint8
    assert np.array_equal(result["pred_mask"].astype(bool), correction)
    assert tracker.object_sizes == [7, 7, 7, 7, 24], "the frame's size is replaced, not doubled"
    assert tracker.last_added == -1
    assert tracker.correction_frames == [5]


def test_correct_can_count_as_a_drm_addition_only_when_asked() -> None:
    tracker = _tracker(StubPredictor(), add_correction_to_drm=True)
    tracker.initialize(_frame(), _seed())
    tracker.track(_frame())
    tracker.correct(_frame(), _seed())
    assert tracker.last_added == 1


def test_correct_appends_a_size_when_none_was_recorded_for_the_frame() -> None:
    tracker = _tracker(StubPredictor())
    tracker.initialize(_frame(), _seed())
    tracker.correct(_frame(), _seed())
    assert tracker.object_sizes == [24]
    assert tracker.frame_index == 0


def test_correct_resizes_a_foreign_mask_and_refuses_an_empty_one() -> None:
    predictor = StubPredictor()
    tracker = _tracker(predictor)
    tracker.initialize(_frame(), _seed())
    tracker.track(_frame())
    big = np.zeros((FRAME_SHAPE[0] * 2, FRAME_SHAPE[1] * 2), dtype=bool)
    big[4:12, 6:18] = True
    tracker.correct(_frame(), big)
    assert predictor.calls[-1]["mask"].shape == FRAME_SHAPE
    assert predictor.calls[-1]["mask"][2:6, 3:9].all()
    with pytest.raises(ValueError, match="empty correction mask"):
        tracker.correct(_frame(), np.zeros(FRAME_SHAPE, dtype=bool))
    with pytest.raises(RuntimeError, match="initialize"):
        _tracker(predictor).correct(_frame(), _seed())


def test_logits_threshold_matches_dam4sam() -> None:
    logits = np.array([[[[-1.0, 0.0], [0.5, 3.0]]]], dtype=np.float32)
    assert streaming.logits_to_mask(logits).tolist() == [[0, 0], [1, 1]]
    assert streaming.logits_to_mask(logits).dtype == np.uint8


def test_input_size_config_copy_replaces_exactly_the_image_size_line(tmp_path: Path) -> None:
    source = tmp_path / "sam21pp_hiera_l.yaml"
    source.write_text(
        "# @package _global_\nmodel:\n  num_maskmem: 7\n  image_size: 1024\n"
        "  sigmoid_scale_for_mem_enc: 20.0\n  max_cond_frames_in_attn: 4\n",
        encoding="utf-8",
    )
    copied = streaming.write_input_size_config(source, 1536, tmp_path / "run" / "sam2_config")

    text = copied.read_text(encoding="utf-8")
    assert copied.name == "sam21pp_hiera_l_image1536.yaml"
    assert "  image_size: 1536\n" in text and "1024" not in text
    assert text.replace("1536", "1024") == source.read_text(encoding="utf-8")
    provenance = streaming.config_provenance(copied, source="measured")
    assert provenance["sha256"] == hashlib.sha256(copied.read_bytes()).hexdigest()
    assert provenance["uri"] == str(copied.resolve())
    with pytest.raises(ValueError, match="exactly one image_size"):
        streaming.replace_image_size("model:\n  size: 1\n", 1536)
    with pytest.raises(ValueError, match="exactly one image_size"):
        streaming.replace_image_size("image_size: 1024\nimage_size: 512\n", 1536)
    with pytest.raises(ValueError, match="not one of"):
        streaming.write_input_size_config(source, 1500, tmp_path)


def test_checkpoint_pins_and_model_table() -> None:
    assert streaming.SAM2_MODELS["tiny"]["sha256"] == TINY_PIN_IN_SMOKE_MODULE
    assert streaming.SAM2_MODELS["large"]["sha256"] != streaming.SAM2_MODELS["tiny"]["sha256"]
    assert len(streaming.SAM2_LARGE_CHECKPOINT_SHA256) == 64
    spec = streaming.resolve_sam2_model(DAM4SAM_ROOT, "large")
    assert spec.checkpoint == DAM4SAM_ROOT / "checkpoints/sam2.1_hiera_large.pt"
    assert spec.config_path == DAM4SAM_ROOT / "sam2/sam21pp_hiera_l.yaml"
    assert spec.tracker_name == "sam21pp-L"
    with pytest.raises(ValueError, match="unknown SAM2 model"):
        streaming.resolve_sam2_model(DAM4SAM_ROOT, "base")


@pytest.mark.real_data
@pytest.mark.slow
@pytest.mark.parametrize("model", ["tiny", "large"])
def test_pinned_checkpoint_sha_matches_the_checkout(model: str) -> None:
    spec = streaming.resolve_sam2_model(DAM4SAM_ROOT, model)
    require_artifact(spec.checkpoint)
    assert streaming.verify_checkpoint(spec) == spec.checkpoint_sha256


@pytest.mark.real_data
def test_checkout_yaml_has_exactly_one_image_size_line_at_1024() -> None:
    for model in ("tiny", "large"):
        spec = streaming.resolve_sam2_model(DAM4SAM_ROOT, model)
        require_artifact(spec.config_path)
        text = spec.config_path.read_text(encoding="utf-8")
        assert streaming.IMAGE_SIZE_LINE.findall(text)[0][1] == "1024"
        assert "image_size: 1536" in streaming.replace_image_size(text, 1536)


def test_vram_extrapolation_is_linear_in_allocated_memory_with_a_peak_offset() -> None:
    probes = [
        {
            "frames_processed": 30,
            "memory_allocated_bytes": 1_000,
            "max_memory_allocated_bytes": 5_000,
        },
        {
            "frames_processed": 300,
            "memory_allocated_bytes": 2_350,
            "max_memory_allocated_bytes": 6_000,
        },
    ]
    projection = streaming.extrapolate_vram(probes, to_frames=1800, limit_bytes=13_000)

    assert projection["slope_bytes_per_frame"] == pytest.approx(5.0)
    assert projection["from_frames"] == (30, 300)
    assert projection["projected_allocated_bytes"] == 2_350 + 5 * 1500
    assert projection["projected_peak_bytes"] == 6_000 + 5 * 1500
    assert projection["within_limit"] is False
    assert streaming.extrapolate_vram(probes, to_frames=1800, limit_bytes=14_000)["within_limit"]
    already_full = streaming.extrapolate_vram(probes, to_frames=300, limit_bytes=1)
    assert already_full["projected_peak_bytes"] == 6_000
    shrinking = [dict(probes[0]), {**probes[1], "memory_allocated_bytes": 500}]
    assert (
        streaming.extrapolate_vram(shrinking, to_frames=1800, limit_bytes=1)["projected_peak_bytes"]
        == 6_000
    )
    with pytest.raises(ValueError, match="two probes"):
        streaming.extrapolate_vram(probes[:1], to_frames=1800, limit_bytes=1)
    with pytest.raises(ValueError, match="distinct"):
        streaming.extrapolate_vram([probes[0], dict(probes[0])], to_frames=1800, limit_bytes=1)


def test_probe_frames_parse_and_merge_as_per_frame_maximum() -> None:
    assert streaming.parse_probe_frames("30,300") == (30, 300)
    assert streaming.parse_probe_frames("300, 30,30") == (30, 300)
    with pytest.raises(ValueError):
        streaming.parse_probe_frames("0,30")
    merged = streaming.merge_probes(
        [
            [
                {
                    "frames_processed": 30,
                    "memory_allocated_bytes": 10,
                    "max_memory_allocated_bytes": 20,
                }
            ],
            [
                {
                    "frames_processed": 30,
                    "memory_allocated_bytes": 15,
                    "max_memory_allocated_bytes": 12,
                },
                {
                    "frames_processed": 300,
                    "memory_allocated_bytes": 1,
                    "max_memory_allocated_bytes": 2,
                },
            ],
        ]
    )
    assert merged == [
        {"frames_processed": 30, "memory_allocated_bytes": 15, "max_memory_allocated_bytes": 20},
        {"frames_processed": 300, "memory_allocated_bytes": 1, "max_memory_allocated_bytes": 2},
    ]


def test_correction_masks_load_like_the_sam3_worker_and_resize_only_when_needed(
    tmp_path: Path,
) -> None:
    mask = np.zeros(FRAME_SHAPE, dtype=np.uint8)
    mask[2:6, 3:9] = 255
    path = tmp_path / "t000327-b02_candidate-00.png"
    Image.fromarray(mask).save(path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()

    same, resized = streaming.read_correction_mask(path, sha, FRAME_SHAPE)
    assert not resized and same.dtype == bool and same.sum() == 24
    bigger, resized = streaming.read_correction_mask(
        path, sha, (FRAME_SHAPE[0] * 3, FRAME_SHAPE[1] * 3)
    )
    assert resized and bigger.shape == (36, 48) and bigger.sum() == 24 * 9
    with pytest.raises(ValueError, match="SHA-256 changed"):
        streaming.read_correction_mask(path, "0" * 64, FRAME_SHAPE)
    empty = tmp_path / "empty.png"
    Image.fromarray(np.zeros(FRAME_SHAPE, dtype=np.uint8)).save(empty)
    with pytest.raises(ValueError, match="empty"):
        streaming.read_correction_mask(
            empty, hashlib.sha256(empty.read_bytes()).hexdigest(), FRAME_SHAPE
        )


def test_corrections_index_by_frame_and_target() -> None:
    grouped = streaming.corrections_by_frame(
        [
            {"frame_index": 327, "target": "chassis", "mask_path": "a"},
            {"frame_index": 327, "target": "cabin", "mask_path": "b"},
            {"frame_index": 900, "target": "chassis", "mask_path": "c"},
        ]
    )
    assert sorted(grouped) == [327, 900]
    assert grouped[327]["cabin"]["mask_path"] == "b"
    with pytest.raises(ValueError, match="duplicate"):
        streaming.corrections_by_frame(
            [{"frame_index": 1, "target": "chassis"}, {"frame_index": 1, "target": "chassis"}]
        )
    with pytest.raises(ValueError, match="positive"):
        streaming.corrections_by_frame([{"frame_index": 0, "target": "chassis"}])
