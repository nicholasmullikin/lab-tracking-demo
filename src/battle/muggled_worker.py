"""Dependency-light SAM3 worker run by the separately managed MuggledSAM interpreter.

This module deliberately imports only the standard library until after preflight.  The
project's parent process validates its JSON output against Battle's Pydantic contracts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import traceback
from collections import deque
from pathlib import Path
from time import perf_counter
from typing import Any

CONCEPTS = ("hand", "yellow toy body", "toy wheel")
# Every frame by default: the exporter logs masks as compressed PNGs, so full-rate masks are
# affordable in the viewer, and a binary PNG per object per frame is small on disk.
MASK_PERIOD_FRAMES = 1
MAX_FRAMES = 300
MAX_FRAME_MEMORY = 4
MAX_PROMPT_MEMORY = 1
MAX_SIDE_LENGTH = 504
ANALYSIS_FPS = 30.0
DETECTION_THRESHOLD = 0.40
# Connected components smaller than this fraction of the largest one are treated as mask
# speckle and excluded from the reported box; the saved mask PNG is left untouched.
BOX_COMPONENT_KEEP_FRACTION = 0.20

# object index, concept, mask logits, reported confidence, raw presence logit, predicted IoU.
# The last two are tracker diagnostics and are absent for prompt and detector initialization.
TrackedMask = tuple[int, str, Any, float, float | None, float | None]


def _prepare_frame(frame: Any, args: argparse.Namespace) -> Any:
    """Apply only the explicitly configured decoded-image representation."""
    import cv2
    import numpy as np

    if args.preprocessing == "original_bgr":
        return frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    low, high = np.percentile(gray, (args.lower_percentile, args.upper_percentile))
    if high <= low:
        normalized = np.zeros_like(gray)
    else:
        normalized = np.clip((gray.astype(np.float32) - low) * (255.0 / (high - low)), 0, 255)
        normalized = normalized.astype(np.uint8)
    clahe = cv2.createCLAHE(
        clipLimit=args.clahe_clip_limit,
        tileGridSize=(args.clahe_tile_grid_size, args.clahe_tile_grid_size),
    )
    return cv2.cvtColor(clahe.apply(normalized), cv2.COLOR_GRAY2BGR)


def _runtime_settings(args: argparse.Namespace, concepts: tuple[str, ...]) -> dict[str, Any]:
    """Persist the condition definition beside the worker measurements."""
    settings: dict[str, Any] = {
        "device": "cuda:0",
        "dtype": "bfloat16",
        "max_frames": MAX_FRAMES,
        "max_side_length": MAX_SIDE_LENGTH,
        "detection_threshold": DETECTION_THRESHOLD,
        "mask_period_frames": MASK_PERIOD_FRAMES,
        "box_derivation": (
            "union of the mask's 8-connected components with area at least "
            f"{BOX_COMPONENT_KEEP_FRACTION:g} of the largest; smaller components are ignored"
        ),
        "box_component_keep_fraction": BOX_COMPONENT_KEEP_FRACTION,
        "max_prompt_memory_entries": MAX_PROMPT_MEMORY,
        "max_frame_memory_entries": MAX_FRAME_MEMORY,
        "chunking": "none; one continuous tracker stream",
        "intentional_id_resets": False,
        "concepts": list(concepts),
        "prompt_mode": args.prompt_mode,
        "preprocessing": args.preprocessing,
        "analysis_fps": ANALYSIS_FPS,
        "max_frame_memory": MAX_FRAME_MEMORY,
        "frame_memory_span_seconds": MAX_FRAME_MEMORY / ANALYSIS_FPS,
        "tracker_diagnostics_semantics": (
            "object_score is the tracker's raw unbounded presence logit, recorded for every "
            "multiplex slot on every tracked frame including frames where the slot was dropped "
            "at or below zero; iou_prediction is the tracker's own mask-quality estimate. "
            "Both are diagnostic traces and neither is measured against ground truth."
        ),
        "lost_object_score_threshold": 0.0,
    }
    if args.preprocessing == "gray_p01_p99_clahe":
        settings.update(
            {
                "lower_percentile": args.lower_percentile,
                "upper_percentile": args.upper_percentile,
                "clahe_clip_limit": args.clahe_clip_limit,
                "clahe_tile_grid_size": args.clahe_tile_grid_size,
            }
        )
    if args.manual_box_json:
        settings["manual_box_seed"] = json.loads(args.manual_box_json)
        settings["initial_confidence_semantics"] = (
            "1.0 is a manual-prompt initialization sentinel; the direct tracking "
            "prompt API does not expose a detector confidence"
        )
    if args.manual_seeds_json:
        settings["manual_seed_multiplex"] = json.loads(args.manual_seeds_json)
        settings["initialization_api"] = "encode_prompt_memory_from_mask"
        settings["initial_confidence_semantics"] = (
            "1.0 is a human-selected-mask initialization sentinel; it is not a "
            "detector confidence or an accuracy score"
        )
    if args.hybrid_initialization_json:
        hybrid = json.loads(args.hybrid_initialization_json)
        settings["hybrid_initialization"] = {
            "targets": hybrid["targets"],
            "text_targets": hybrid["text_targets"],
            "manual_seeds": [
                {key: value for key, value in seed.items() if key != "mask_path"}
                for seed in hybrid["manual_seeds"]
            ],
        }
        settings["initialization_api"] = "encode_prompt_memory_from_mask"
        settings["initial_confidence_semantics"] = (
            "text targets retain detector confidence; 1.0 marks the human-reviewed "
            "mask and is not a detector confidence or accuracy score"
        )
    if args.multi_keyframe_schedule_json:
        schedule = json.loads(args.multi_keyframe_schedule_json)
        settings["multi_keyframe_correction_schedule"] = {
            "initial_seed_count": len(schedule["seeds"]),
            "later_correction_count": len(schedule["corrections"]),
            "correction_frames": sorted(
                {int(correction["frame_index"]) for correction in schedule["corrections"]}
            ),
            "memory_semantics": schedule["memory_semantics"],
        }
        settings["initialization_api"] = "encode_prompt_memory_from_mask"
        settings["correction_api"] = "encode_prompt_memory_from_mask"
        settings["correction_memory_semantics"] = schedule["memory_semantics"]
        settings["initial_confidence_semantics"] = (
            "1.0 marks a human-selected initialization/correction mask; it is not a "
            "detector confidence or an accuracy score"
        )
    if args.text_targets_json:
        settings["text_prompt_mapping"] = json.loads(args.text_targets_json)
    return settings


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _is_numpy_mask(mask: Any) -> bool:
    """Identify OpenCV/NumPy masks without relying on NumPy implementation details."""
    import numpy as np

    return isinstance(mask, np.ndarray)


def _gpu_processes() -> list[dict[str, str]]:
    """Return visible compute processes without failing if nvidia-smi is unavailable."""
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader",
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return []
    if completed.returncode != 0:
        return []
    processes: list[dict[str, str]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", maxsplit=2)]
        if len(fields) == 3 and fields[0] != "No running processes found":
            processes.append({"pid": fields[0], "process_name": fields[1], "memory": fields[2]})
    return processes


def _looks_like_model_process(process: dict[str, str]) -> bool:
    """Flag GPU processes that could be running a model, tolerating the Rerun viewer.

    The viewer is a renderer, not a model, but it lives under a Python venv path
    (``.../site-packages/rerun_sdk/rerun_cli/rerun``) and is launched by a ``python``
    wrapper, so the name tokens alone would refuse to start beside an open recording.
    It still appears in ``gpu_processes_before_initialization`` for the record; per-process
    peak VRAM is unaffected, though wall-clock timing may see GPU contention.
    """
    name = process["process_name"].lower()
    if _is_rerun_viewer(name):
        return False
    return any(token in name for token in ("python", "torch", "ollama", "llama", "vllm"))


def _is_rerun_viewer(name: str) -> bool:
    executable = name.rsplit("/", 1)[-1]
    return executable == "rerun" or "/rerun_sdk/rerun_cli/" in name


def _result(
    *,
    state: str,
    reason: str | None,
    frames_processed: int,
    elapsed_seconds: float,
    ttfu_seconds: float | None,
    gpu_peak_vram_bytes: int | None,
    masks_written: int,
    gpu_processes: list[dict[str, str]],
    unavailable: list[str],
    runtime_settings: dict[str, Any],
) -> dict[str, Any]:
    return {
        "state": state,
        "reason": reason,
        "frames_processed": frames_processed,
        "elapsed_seconds": elapsed_seconds,
        "time_to_first_usable_output_seconds": ttfu_seconds,
        "gpu_peak_vram_bytes": gpu_peak_vram_bytes,
        "masks_written": masks_written,
        "gpu_processes_before_initialization": gpu_processes,
        "known_unavailable_measures": unavailable,
        "runtime_settings": runtime_settings,
    }


def _binary_mask(mask_logits: Any, frame_shape: tuple[int, int]) -> Any:
    """Threshold one object's logits (or a NumPy mask) into a boolean mask at frame size."""
    import cv2
    import torch.nn.functional as functional

    if _is_numpy_mask(mask_logits):
        mask = mask_logits.squeeze()
        if mask.shape != frame_shape:
            mask = cv2.resize(
                mask, (frame_shape[1], frame_shape[0]), interpolation=cv2.INTER_NEAREST
            )
        return mask > 0
    resized = functional.interpolate(
        mask_logits, size=frame_shape, mode="bilinear", align_corners=False
    )
    return (resized > 0.0).squeeze().cpu().numpy()


def _save_mask(mask: Any, path: Path) -> bool:
    import cv2

    return bool(cv2.imwrite(str(path), mask.astype("uint8") * 255))


def _box_from_mask(mask: Any) -> dict[str, float] | None:
    """Box the dominant connected component(s) rather than every positive pixel.

    A min/max box over all positive pixels lets a few stray speckles far from the object
    balloon the box while the mask itself still reads correctly.  Only components with at
    least ``BOX_COMPONENT_KEEP_FRACTION`` of the largest component's area contribute, so
    an object legitimately split by occlusion keeps a box over both parts.
    """
    import cv2

    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask.astype("uint8"), connectivity=8
    )
    if count < 2:
        return None
    return _box_from_component_stats(stats[1:], mask.shape)


# Column order of ``cv2.connectedComponentsWithStats`` statistics (``cv2.CC_STAT_*``).
_STAT_LEFT, _STAT_TOP, _STAT_WIDTH, _STAT_HEIGHT, _STAT_AREA = range(5)


def _box_from_component_stats(components: Any, frame_shape: tuple[int, int]) -> dict[str, float]:
    """Union the pixel boxes of every foreground component large enough to keep."""
    areas = components[:, _STAT_AREA]
    kept = components[areas >= BOX_COMPONENT_KEEP_FRACTION * areas.max()]
    height, width = frame_shape
    x1 = kept[:, _STAT_LEFT].min() / width
    y1 = kept[:, _STAT_TOP].min() / height
    x2 = (kept[:, _STAT_LEFT] + kept[:, _STAT_WIDTH]).max() / width
    y2 = (kept[:, _STAT_TOP] + kept[:, _STAT_HEIGHT]).max() / height
    return {"x": float(x1), "y": float(y1), "width": float(x2 - x1), "height": float(y2 - y1)}


def _scalar(values: Any, position: int) -> float | None:
    """Read one slot from a tracker diagnostic tensor without assuming its trailing shape."""
    if values is None:
        return None
    try:
        selected = values[position]
    except (IndexError, KeyError, TypeError):
        return None
    reshaped = selected.reshape(-1) if hasattr(selected, "reshape") else selected
    try:
        return float(reshaped[0] if hasattr(reshaped, "__len__") and len(reshaped) else reshaped)
    except (TypeError, ValueError):
        return None


def _observation(
    *,
    view_id: str,
    frame_index: int,
    source_offset_seconds: float,
    masks: list[TrackedMask],
    frame_shape: tuple[int, int],
    masks_directory: Path,
    run_directory: Path,
    diagnostics: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], int]:
    objects = []
    mask_count = 0
    for object_index, concept, mask_logits, confidence, object_score, iou_prediction in masks:
        mask = _binary_mask(mask_logits, frame_shape)
        box = _box_from_mask(mask)
        if box is None:
            continue
        mask_reference = None
        if frame_index % MASK_PERIOD_FRAMES == 0:
            filename = f"{frame_index:06d}_{object_index:02d}.png"
            mask_path = masks_directory / filename
            if _save_mask(mask, mask_path):
                mask_count += 1
                mask_reference = {
                    "uri": mask_path.relative_to(run_directory).as_posix(),
                    "storage": "external_artifact",
                    "format": "png",
                }
        objects.append(
            {
                "object_id": f"sam3-{object_index:02d}",
                "label": concept,
                "confidence": max(0.0, min(1.0, confidence)),
                "box": box,
                "mask": mask_reference,
                "object_score": object_score,
                "iou_prediction": iou_prediction,
            }
        )
    return (
        {
            "schema_version": "1.0",
            "view_id": view_id,
            "analysis_frame_index": frame_index,
            "source_seconds": source_offset_seconds + frame_index / ANALYSIS_FPS,
            "objects": objects,
            "hands": [],
            "tracker_diagnostics": diagnostics or [],
        },
        mask_count,
    )


def _validate_manual_seed_slots(
    seed_records: list[dict[str, Any]], concepts: tuple[str, ...]
) -> None:
    """Require one ordered selected mask for every configured multiplex target."""
    if not seed_records or len(concepts) != len(seed_records):
        raise ValueError(
            "manual multiplex initialization requires one mask for each configured target"
        )
    for slot, seed in enumerate(seed_records):
        if seed.get("target") != concepts[slot] or seed.get("initial_multiplex_slot") != slot:
            raise ValueError("manual seed target-to-multiplex-slot association is invalid")


def _validate_text_targets(text_targets: list[dict[str, Any]], concepts: tuple[str, ...]) -> None:
    """Require one distinct detector prompt for each ordered output label."""
    labels = [target.get("output_label") for target in text_targets]
    prompts = [target.get("text_prompt") for target in text_targets]
    if tuple(labels) != concepts:
        raise ValueError("text-target labels must match the ordered output concepts")
    if any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
        raise ValueError("text-target prompts must be non-empty strings")
    if len(set(prompts)) != len(prompts):
        raise ValueError("text-target prompts must be distinct")


def _validate_hybrid_initialization(hybrid: dict[str, Any], concepts: tuple[str, ...]) -> None:
    """Require the explicit canonical 3-text/1-reviewed-mask worker contract."""
    expected = (
        ("left_hand", 0, "text_prompt"),
        ("right_hand", 1, "text_prompt"),
        ("yellow_toy_top", 2, "text_prompt"),
        ("black_toy_top_base", 3, "human_reviewed_mask"),
    )
    targets = tuple(
        (
            target.get("output_label"),
            target.get("initial_multiplex_slot"),
            target.get("initialization_source"),
        )
        for target in hybrid.get("targets", [])
    )
    if concepts != tuple(item[0] for item in expected) or targets != expected:
        raise ValueError("hybrid worker targets must preserve canonical labels, slots, and sources")
    text_targets = list(hybrid.get("text_targets", []))
    expected_text = tuple(item[0] for item in expected if item[2] == "text_prompt")
    _validate_text_targets(text_targets, expected_text)
    manual_seeds = list(hybrid.get("manual_seeds", []))
    expected_manual = [item for item in expected if item[2] == "human_reviewed_mask"]
    if len(manual_seeds) != len(expected_manual):
        raise ValueError("hybrid worker requires exactly one reviewed manual mask")
    for seed, (label, slot, _) in zip(manual_seeds, expected_manual, strict=True):
        if seed.get("target") != label or seed.get("initial_multiplex_slot") != slot:
            raise ValueError("hybrid reviewed mask target-to-slot association is invalid")
    if {target["output_label"] for target in text_targets} & {
        seed["target"] for seed in manual_seeds
    }:
        raise ValueError("hybrid text and manual target labels must be disjoint")


def _ordered_hybrid_initial_masks(
    targets: list[dict[str, Any]],
    text_masks: dict[str, tuple[Any, float]],
    manual_masks: dict[str, Any],
) -> list[TrackedMask]:
    """Order present text detections and required manual masks by canonical slot."""
    ordered: list[TrackedMask] = []
    for target in targets:
        label = str(target["output_label"])
        slot = int(target["initial_multiplex_slot"])
        if target["initialization_source"] == "text_prompt":
            detected = text_masks.get(label)
            if detected is not None:
                mask, confidence = detected
                ordered.append((slot, label, mask, confidence, None, None))
        else:
            if label not in manual_masks:
                raise ValueError(f"hybrid reviewed mask is unavailable for {label}")
            ordered.append((slot, label, manual_masks[label], 1.0, None, None))
    return ordered


def _corrections_by_frame(
    schedule: dict[str, Any], concepts: tuple[str, ...]
) -> dict[int, list[dict[str, Any]]]:
    """Validate worker payload slots before grouping optional later corrections."""
    if schedule.get("memory_semantics") != "replace_prompt_memory_and_reset_frame_memory":
        raise ValueError("unsupported correction memory semantics")
    _validate_manual_seed_slots(list(schedule.get("seeds", [])), concepts)
    grouped: dict[int, list[dict[str, Any]]] = {}
    seen: set[tuple[int, int]] = set()
    for correction in schedule.get("corrections", []):
        frame_index, slot = int(correction["frame_index"]), int(correction["multiplex_slot"])
        if frame_index <= 0 or slot < 0 or slot >= len(concepts):
            raise ValueError("correction frames must be positive and slots must name a target")
        if correction.get("target") != concepts[slot] or (frame_index, slot) in seen:
            raise ValueError("correction schedule has an ambiguous frame/slot assignment")
        seen.add((frame_index, slot))
        grouped.setdefault(frame_index, []).append(correction)
    return grouped


def _read_verified_mask(path: str, expected_sha256: str, frame_shape: tuple[int, int]) -> Any:
    """Load exactly one non-empty, source-sized calibration mask after hashing it."""
    import cv2

    mask_path = Path(path)
    if hashlib.sha256(mask_path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError(f"selected calibration mask SHA-256 changed: {mask_path}")
    selected_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if selected_mask is None or selected_mask.shape != frame_shape:
        raise ValueError(f"selected calibration mask has wrong shape: {mask_path}")
    binary_mask = selected_mask > 0
    if not bool(binary_mask.any()):
        raise ValueError(f"selected calibration mask is empty: {mask_path}")
    return binary_mask


def _replace_prompt_memory_for_correction(
    *,
    predicted_source_masks: Any,
    correction_masks_by_slot: dict[int, Any],
    prompt_memories: deque[Any],
    frame_memories: deque[Any],
    encoded_frame: Any,
    encode_prompt_memory_from_mask: Any,
) -> Any:
    """Replace the one prompt entry and automatic history using a full multiplex mask batch."""
    if predicted_source_masks.shape[0] == 0:
        raise ValueError("cannot apply a correction without multiplex predictions")
    rebased_masks = predicted_source_masks.copy()
    for slot, mask in correction_masks_by_slot.items():
        if slot < 0 or slot >= rebased_masks.shape[0] or mask.shape != rebased_masks.shape[1:]:
            raise ValueError("correction mask does not match the multiplex source-mask shape")
        rebased_masks[slot] = mask
    replacement_memory = encode_prompt_memory_from_mask(encoded_frame, rebased_masks)
    prompt_memories.clear()
    prompt_memories.append(replacement_memory)
    frame_memories.clear()
    return rebased_masks


def _source_binary_masks(mask_logits: Any, frame_shape: tuple[int, int]) -> Any:
    """Map all multiplex logits onto source pixels before rebasing prompt memory."""
    import torch.nn.functional as functional

    return (
        functional.interpolate(mask_logits, size=frame_shape, mode="bilinear", align_corners=False)
        .gt(0)
        .squeeze(1)
        .cpu()
        .numpy()
    )


def run(args: argparse.Namespace) -> int:
    run_directory = Path(args.run_directory)
    output_path = run_directory / "observations.jsonl"
    result_path = run_directory / "worker_result.json"
    masks_directory = run_directory / "masks"
    video_path = Path(args.video)
    model_path = Path(args.model)
    concepts = tuple(json.loads(args.concepts_json))
    text_targets = (
        json.loads(args.text_targets_json)
        if args.text_targets_json
        else [{"output_label": concept, "text_prompt": concept} for concept in concepts]
    )
    manual_box = json.loads(args.manual_box_json) if args.manual_box_json else None
    manual_seeds = json.loads(args.manual_seeds_json) if args.manual_seeds_json else None
    hybrid_initialization = (
        json.loads(args.hybrid_initialization_json) if args.hybrid_initialization_json else None
    )
    multi_keyframe_schedule = (
        json.loads(args.multi_keyframe_schedule_json) if args.multi_keyframe_schedule_json else None
    )
    if hybrid_initialization is not None and any(
        value is not None
        for value in (
            args.text_targets_json,
            manual_box,
            manual_seeds,
            multi_keyframe_schedule,
        )
    ):
        raise ValueError("hybrid initialization cannot be combined with another prompt payload")
    runtime_settings = _runtime_settings(args, concepts)
    start = perf_counter()
    gpu_processes = _gpu_processes()
    concurrent_models = [process for process in gpu_processes if _looks_like_model_process(process)]

    if not video_path.is_file():
        _write_json(
            result_path,
            _result(
                state="blocked",
                reason=f"approved proxy does not exist: {video_path}",
                frames_processed=0,
                elapsed_seconds=perf_counter() - start,
                ttfu_seconds=None,
                gpu_peak_vram_bytes=None,
                masks_written=0,
                gpu_processes=gpu_processes,
                unavailable=["time_to_first_usable_output_seconds", "gpu_peak_vram_bytes"],
                runtime_settings=runtime_settings,
            ),
        )
        return 2
    if not model_path.is_file():
        _write_json(
            result_path,
            _result(
                state="blocked",
                reason=f"SAM3 checkpoint is unavailable at configured path: {model_path}",
                frames_processed=0,
                elapsed_seconds=perf_counter() - start,
                ttfu_seconds=None,
                gpu_peak_vram_bytes=None,
                masks_written=0,
                gpu_processes=gpu_processes,
                unavailable=["time_to_first_usable_output_seconds", "gpu_peak_vram_bytes"],
                runtime_settings=runtime_settings,
            ),
        )
        return 2
    if concurrent_models:
        _write_json(
            result_path,
            _result(
                state="blocked",
                reason=f"concurrent GPU model process(es) detected: {concurrent_models}",
                frames_processed=0,
                elapsed_seconds=perf_counter() - start,
                ttfu_seconds=None,
                gpu_peak_vram_bytes=None,
                masks_written=0,
                gpu_processes=gpu_processes,
                unavailable=["time_to_first_usable_output_seconds", "gpu_peak_vram_bytes"],
                runtime_settings=runtime_settings,
            ),
        )
        return 2

    try:
        import cv2
        import numpy as np
        import torch
        from muggled_sam.make_sam import make_sam_from_state_dict

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; smoke policy does not permit CPU inference")
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        capture = cv2.VideoCapture(str(video_path))
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        ok, first_frame = capture.read()
        if not ok:
            raise RuntimeError(f"could not decode first proxy frame: {video_path}")
        first_model_frame = _prepare_frame(first_frame, args)
        condition_writer = None
        if args.condition_input_video:
            condition_input_path = Path(args.condition_input_video)
            condition_writer = cv2.VideoWriter(
                str(condition_input_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                ANALYSIS_FPS,
                (first_model_frame.shape[1], first_model_frame.shape[0]),
            )
            if not condition_writer.isOpened():
                raise RuntimeError(f"could not open condition input video: {condition_input_path}")
            condition_writer.write(first_model_frame)

        core = make_sam_from_state_dict(model_path)
        core.to(device="cuda:0", dtype=torch.bfloat16)
        tracking = core.get_tracking_context()
        if not hasattr(tracking, "step_video_masking_multiplex"):
            raise TypeError("configured checkpoint does not expose SAM3.1 multiplex video tracking")

        initial_masks: list[TrackedMask] = []
        initial_memory: Any = None
        correction_schedule: dict[int, list[dict[str, Any]]] = {}
        if hybrid_initialization is not None:
            _validate_hybrid_initialization(hybrid_initialization, concepts)
            detector = core.get_detector_context()
            detector_encoded = detector.encode_image(first_model_frame, MAX_SIDE_LENGTH, True)
            text_masks: dict[str, tuple[Any, float]] = {}
            for target in hybrid_initialization["text_targets"]:
                exemplars = detector.encode_exemplars(detector_encoded, text=target["text_prompt"])
                masks, _, scores, _ = detector.generate_detections(
                    detector_encoded,
                    exemplars,
                    detection_filter_threshold=DETECTION_THRESHOLD,
                )
                if masks.shape[1] == 0:
                    continue
                best_index = int(scores[0].argmax())
                source_mask = _source_binary_masks(masks[:, [best_index]], first_frame.shape[:2])[0]
                text_masks[target["output_label"]] = (
                    source_mask,
                    float(scores[0, best_index]),
                )
            manual_masks = {
                seed["target"]: _read_verified_mask(
                    str(seed["mask_path"]),
                    str(seed["mask_sha256"]),
                    first_frame.shape[:2],
                )
                for seed in hybrid_initialization["manual_seeds"]
            }
            initial_masks = _ordered_hybrid_initial_masks(
                hybrid_initialization["targets"], text_masks, manual_masks
            )
            tracking_encoded = tracking.encode_image(first_model_frame, MAX_SIDE_LENGTH, True)
            initial_memory = tracking.encode_prompt_memory_from_mask(
                tracking_encoded,
                np.stack([mask for _, _, mask, *_ in initial_masks], axis=0),
            )
        elif manual_seeds is not None or multi_keyframe_schedule is not None:
            if manual_seeds is not None and multi_keyframe_schedule is not None:
                raise ValueError("manual seeds and a multi-keyframe schedule cannot be combined")
            seed_records = (
                manual_seeds.get("seeds", [])
                if manual_seeds is not None
                else multi_keyframe_schedule.get("seeds", [])
            )
            _validate_manual_seed_slots(seed_records, concepts)
            correction_schedule = (
                _corrections_by_frame(multi_keyframe_schedule, concepts)
                if multi_keyframe_schedule is not None
                else {}
            )
            source_masks = []
            for slot, seed in enumerate(seed_records):
                binary_mask = _read_verified_mask(
                    str(seed["mask_path"]), str(seed["mask_sha256"]), first_frame.shape[:2]
                )
                source_masks.append(binary_mask)
                initial_masks.append((slot, concepts[slot], binary_mask, 1.0, None, None))
            tracking_encoded = tracking.encode_image(first_model_frame, MAX_SIDE_LENGTH, True)
            initial_memory = tracking.encode_prompt_memory_from_mask(
                tracking_encoded, np.stack(source_masks, axis=0)
            )
        elif manual_box is not None:
            normalized_box = manual_box["normalized_box"]
            box = [
                [
                    (normalized_box["x"], normalized_box["y"]),
                    (
                        normalized_box["x"] + normalized_box["width"],
                        normalized_box["y"] + normalized_box["height"],
                    ),
                ]
            ]
            tracking_encoded = tracking.encode_image(first_model_frame, MAX_SIDE_LENGTH, True)
            initial_mask, initial_memory = tracking.encode_prompt_memory(
                tracking_encoded, box, [], []
            )
            initial_masks.append((0, concepts[0], initial_mask, 1.0, None, None))
        else:
            _validate_text_targets(text_targets, concepts)
            detector = core.get_detector_context()
            detector_encoded = detector.encode_image(first_model_frame, MAX_SIDE_LENGTH, True)
            for concept_index, target in enumerate(text_targets):
                concept = target["output_label"]
                exemplars = detector.encode_exemplars(detector_encoded, text=target["text_prompt"])
                masks, _, scores, _ = detector.generate_detections(
                    detector_encoded, exemplars, detection_filter_threshold=DETECTION_THRESHOLD
                )
                if masks.shape[1] == 0:
                    continue
                best_index = int(scores[0].argmax())
                initial_masks.append(
                    (
                        concept_index,
                        concept,
                        masks[:, [best_index]],
                        float(scores[0, best_index]),
                        None,
                        None,
                    )
                )
            if initial_masks:
                initial_tensor = torch.cat([mask for _, _, mask, *_ in initial_masks], dim=1)
                tracking_encoded = tracking.encode_image(first_model_frame, MAX_SIDE_LENGTH, True)
                initial_memory = tracking.encode_prompt_memory_from_mask(
                    tracking_encoded, initial_tensor
                )

        if hybrid_initialization is not None:
            initialized_labels = {label for _, label, *_ in initial_masks}
            runtime_settings["hybrid_initialization_outcomes"] = [
                {
                    "output_label": target["output_label"],
                    "initial_multiplex_slot": target["initial_multiplex_slot"],
                    "initialization_source": target["initialization_source"],
                    "initialized_at_frame_zero": target["output_label"] in initialized_labels,
                }
                for target in hybrid_initialization["targets"]
            ]

        masks_directory.mkdir(exist_ok=True)
        processed = 0
        masks_written = 0
        first_usable: float | None = None
        with output_path.open("w") as observations:
            initial_observation, written = _observation(
                view_id=args.view_id,
                frame_index=0,
                source_offset_seconds=args.source_offset_seconds,
                masks=initial_masks,
                frame_shape=first_frame.shape[:2],
                masks_directory=masks_directory,
                run_directory=run_directory,
            )
            observations.write(json.dumps(initial_observation, sort_keys=True) + "\n")
            processed = 1
            masks_written += written
            if initial_observation["objects"]:
                first_usable = perf_counter() - start

            if initial_masks:
                prompt_memories = deque([initial_memory], maxlen=MAX_PROMPT_MEMORY)
                frame_memories = deque([], maxlen=MAX_FRAME_MEMORY)
                for frame_index in range(1, MAX_FRAMES):
                    ok, frame = capture.read()
                    if not ok:
                        break
                    model_frame = _prepare_frame(frame, args)
                    if condition_writer is not None:
                        condition_writer.write(model_frame)
                    encoded = tracking.encode_image(model_frame, MAX_SIDE_LENGTH, True)
                    masks, ious, pointers, scores = tracking.step_video_masking_multiplex(
                        encoded,
                        prompt_memories,
                        frame_memories,
                        num_multiplex_objects=len(initial_masks),
                    )
                    active = scores > 0
                    correction_masks = {
                        int(correction["multiplex_slot"]): _read_verified_mask(
                            str(correction["mask_path"]),
                            str(correction["mask_sha256"]),
                            frame.shape[:2],
                        )
                        for correction in correction_schedule.get(frame_index, [])
                    }
                    if correction_masks:
                        source_masks = _replace_prompt_memory_for_correction(
                            predicted_source_masks=_source_binary_masks(masks, frame.shape[:2]),
                            correction_masks_by_slot=correction_masks,
                            prompt_memories=prompt_memories,
                            frame_memories=frame_memories,
                            encoded_frame=encoded,
                            encode_prompt_memory_from_mask=tracking.encode_prompt_memory_from_mask,
                        )
                    elif bool(active.any()):
                        frame_memories.append(
                            tracking.encode_frame_memory(encoded, masks, pointers, scores)
                        )
                    tracked = [
                        (
                            index,
                            concept,
                            (
                                source_masks[position : position + 1]
                                if correction_masks
                                else masks[position : position + 1]
                            ),
                            1.0 if position in correction_masks else float(scores[position]),
                            float(scores[position]),
                            _scalar(ious, position),
                        )
                        for position, (index, concept, *_) in enumerate(initial_masks)
                        if bool(active[position]) or position in correction_masks
                    ]
                    diagnostics = [
                        {
                            "schema_version": "1.0",
                            "object_id": f"sam3-{index:02d}",
                            "label": concept,
                            "multiplex_slot": index,
                            "object_score": float(scores[position]),
                            "iou_prediction": _scalar(ious, position),
                            "active": bool(active[position]),
                            "corrected": position in correction_masks,
                        }
                        for position, (index, concept, *_) in enumerate(initial_masks)
                    ]
                    observation, written = _observation(
                        view_id=args.view_id,
                        frame_index=frame_index,
                        source_offset_seconds=args.source_offset_seconds,
                        masks=tracked,
                        frame_shape=frame.shape[:2],
                        masks_directory=masks_directory,
                        run_directory=run_directory,
                        diagnostics=diagnostics,
                    )
                    observations.write(json.dumps(observation, sort_keys=True) + "\n")
                    processed += 1
                    masks_written += written
                    if first_usable is None and observation["objects"]:
                        first_usable = perf_counter() - start
            else:
                # Preserve the per-frame contract even when no initial concept produces a mask.
                for frame_index in range(1, MAX_FRAMES):
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if condition_writer is not None:
                        condition_writer.write(_prepare_frame(frame, args))
                    observations.write(
                        json.dumps(
                            {
                                "schema_version": "1.0",
                                "view_id": args.view_id,
                                "analysis_frame_index": frame_index,
                                "source_seconds": (
                                    args.source_offset_seconds + frame_index / ANALYSIS_FPS
                                ),
                                "objects": [],
                                "hands": [],
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    processed += 1
        if condition_writer is not None:
            condition_writer.release()
        capture.release()
        torch.cuda.synchronize(0)
        elapsed = perf_counter() - start
        _write_json(
            result_path,
            _result(
                state="succeeded",
                reason=None,
                frames_processed=processed,
                elapsed_seconds=elapsed,
                ttfu_seconds=first_usable,
                gpu_peak_vram_bytes=int(torch.cuda.max_memory_allocated(0)),
                masks_written=masks_written,
                gpu_processes=gpu_processes,
                unavailable=[]
                if first_usable is not None
                else ["time_to_first_usable_output_seconds"],
                runtime_settings=runtime_settings,
            ),
        )
        return 0
    except Exception as error:
        if "condition_writer" in locals() and condition_writer is not None:
            condition_writer.release()
        _write_json(
            result_path,
            _result(
                state="failed",
                reason=f"{type(error).__name__}: {error}",
                frames_processed=0,
                elapsed_seconds=perf_counter() - start,
                ttfu_seconds=None,
                gpu_peak_vram_bytes=None,
                masks_written=0,
                gpu_processes=gpu_processes,
                unavailable=["time_to_first_usable_output_seconds", "gpu_peak_vram_bytes"],
                runtime_settings=runtime_settings,
            ),
        )
        traceback.print_exc()
        return 1


def main() -> None:
    global MAX_FRAMES, MAX_SIDE_LENGTH, MAX_FRAME_MEMORY, ANALYSIS_FPS, MASK_PERIOD_FRAMES

    parser = argparse.ArgumentParser(
        description="Run one fixed 300-frame MuggledSAM/SAM3 smoke worker."
    )
    parser.add_argument("--run-directory", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--view-id", required=True)
    parser.add_argument("--source-offset-seconds", type=float, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-frames", type=int, choices=(300, 600, 1800, 5400, 5901), default=300)
    parser.add_argument("--concepts-json", default=json.dumps(CONCEPTS))
    parser.add_argument(
        "--text-targets-json",
        help="Ordered output_label/text_prompt mappings for text detection.",
    )
    parser.add_argument(
        "--hybrid-initialization-json",
        help="Explicit combined text/reviewed-mask target and slot contract.",
    )
    parser.add_argument(
        "--preprocessing",
        choices=("original_bgr", "gray_p01_p99_clahe"),
        default="original_bgr",
    )
    parser.add_argument("--lower-percentile", type=float, default=1.0)
    parser.add_argument("--upper-percentile", type=float, default=99.0)
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0)
    parser.add_argument("--clahe-tile-grid-size", type=int, default=8)
    parser.add_argument(
        "--prompt-mode",
        choices=(
            "text_detection",
            "manual_box",
            "manual_seed_multiplexed",
            "manual_seed_multiplexed_keyframes",
            "hybrid_text_and_manual_mask",
        ),
        default="text_detection",
    )
    parser.add_argument("--manual-box-json")
    parser.add_argument("--manual-seeds-json")
    parser.add_argument("--multi-keyframe-schedule-json")
    parser.add_argument("--condition-input-video")
    parser.add_argument(
        "--max-side-length",
        type=int,
        default=MAX_SIDE_LENGTH,
        help="Longest encoded input side. Raising it costs VRAM and time per frame.",
    )
    parser.add_argument(
        "--max-frame-memory",
        type=int,
        default=MAX_FRAME_MEMORY,
        help="Frame-memory entries. Its span in seconds is this count divided by the frame rate.",
    )
    parser.add_argument(
        "--analysis-fps",
        type=float,
        default=ANALYSIS_FPS,
        help="Proxy frame rate, used to map analysis frame indices back onto source seconds.",
    )
    parser.add_argument(
        "--mask-period-frames",
        type=int,
        default=MASK_PERIOD_FRAMES,
        help="Write mask PNGs every N analysis frames; 1 writes one per object on every frame.",
    )
    args = parser.parse_args()
    if args.mask_period_frames < 1:
        parser.error("--mask-period-frames must be at least 1")
    MAX_FRAMES = args.max_frames
    MAX_SIDE_LENGTH = args.max_side_length
    MAX_FRAME_MEMORY = args.max_frame_memory
    ANALYSIS_FPS = args.analysis_fps
    MASK_PERIOD_FRAMES = args.mask_period_frames
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
