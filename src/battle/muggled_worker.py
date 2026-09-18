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
# Bumped whenever the saved tracker state stops being loadable by this worker.
CHECKPOINT_FORMAT = "muggledsam-sam3-multiplex-tracker-state/1"
DETECTION_THRESHOLD = 0.40
# Connected components smaller than this fraction of the largest one are treated as mask
# speckle and excluded from the reported box; the saved mask PNG is left untouched.
BOX_COMPONENT_KEEP_FRACTION = 0.20
# Tracker memory policy defaults.  Every default is "off" so that a run without the flags is
# byte-identical to the runs made before the policy existed.
SLOT_EXCLUSIVITY_MODES = ("off", "argmax")
MEMORY_GATE_MODES = ("off", "on")
EXCLUSIVITY_LOSER_LOGIT = -8.0
GATE_MIN_OBJECT_SCORE = 0.0
GATE_MIN_IOU = 0.5
GATE_MAX_CONTESTED_FRACTION = 0.2
GATE_AREA_BAND = (0.5, 2.0)
GATE_AREA_HISTORY_FRAMES = 30
# The score the memory encoder sees for a gated slot: below zero it adds `no_object_embed`
# for that multiplex entry only, so the frame is memorised as "absent" for that slot.
GATED_OBJECT_SCORE = -1.0
GATE_REASONS = ("ok", "warmup", "low_object_score", "low_iou", "contested", "area_jump")

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
        "tracker_memory_policy": memory_policy_from_args(args),
        "tracker_memory_policy_semantics": (
            "slot_exclusivity argmax gives every pixel predicted positive by more than one "
            "present slot to the slot with the larger logit and pushes the others to at most "
            "exclusivity_loser_logit before memory encoding and output; memory_gate on hands "
            "the memory encoder a score of -1 (its no-object embedding) for a slot whose raw "
            "score, predicted IoU, contested fraction or area (against the rolling median of "
            "its trusted frames) fails the thresholds, so that frame is memorised as absent "
            "for that slot only. Both off reproduces the unpoliced tracker exactly. Neither "
            "is an accuracy claim."
        ),
    }
    if getattr(args, "resume_from_checkpoint", None):
        settings["resumed_from_checkpoint"] = args.resume_from_checkpoint
        settings["chunking"] = (
            "none; one tracker stream resumed from a saved state rather than stepped from frame 0"
        )
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
        manual_seeds = json.loads(args.manual_seeds_json)
        settings["manual_seed_multiplex"] = manual_seeds
        settings["initialization_api"] = "encode_prompt_memory_from_mask"
        agent_seeded = any(
            seed.get("selected_by") == "agent" for seed in manual_seeds.get("seeds", [])
        )
        settings["initial_confidence_semantics"] = (
            (
                "1.0 is an agent-selected-mask initialization sentinel (provenance "
                f"{manual_seeds.get('seed_provenance', 'agent')}); it is not a human review, "
                "a detector confidence or an accuracy score"
            )
            if agent_seeded
            else (
                "1.0 is a human-selected-mask initialization sentinel; it is not a "
                "detector confidence or an accuracy score"
            )
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
        agent_frames = sorted(
            {
                int(correction["frame_index"])
                for correction in schedule["corrections"]
                if correction.get("selected_by") == "agent"
            }
        )
        settings["multi_keyframe_correction_schedule"] = {
            "initial_seed_count": len(schedule["seeds"]),
            "later_correction_count": len(schedule["corrections"]),
            "correction_frames": sorted(
                {int(correction["frame_index"]) for correction in schedule["corrections"]}
            ),
            "agent_selected_correction_frames": agent_frames,
            "memory_semantics": schedule["memory_semantics"],
        }
        settings["initialization_api"] = "encode_prompt_memory_from_mask"
        settings["correction_api"] = "encode_prompt_memory_from_mask"
        settings["correction_memory_semantics"] = schedule["memory_semantics"]
        settings["initial_confidence_semantics"] = (
            "1.0 marks a human-selected initialization/correction mask"
            + (
                f" (agent-selected correction masks at frames {agent_frames})"
                if agent_frames
                else ""
            )
            + "; it is not a detector confidence or an accuracy score"
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


def memory_policy_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Read the tracker memory policy as one plain record, so it can be logged and hashed."""
    band = getattr(args, "gate_area_band", GATE_AREA_BAND)
    return {
        "slot_exclusivity": getattr(args, "slot_exclusivity", "off"),
        "memory_gate": getattr(args, "memory_gate", "off"),
        "exclusivity_loser_logit": float(
            getattr(args, "exclusivity_loser_logit", EXCLUSIVITY_LOSER_LOGIT)
        ),
        "gate_min_object_score": float(
            getattr(args, "gate_min_object_score", GATE_MIN_OBJECT_SCORE)
        ),
        "gate_min_iou": float(getattr(args, "gate_min_iou", GATE_MIN_IOU)),
        "gate_max_contested_fraction": float(
            getattr(args, "gate_max_contested_fraction", GATE_MAX_CONTESTED_FRACTION)
        ),
        "gate_area_band": [float(band[0]), float(band[1])],
        "gate_area_history_frames": int(
            getattr(args, "gate_area_history_frames", GATE_AREA_HISTORY_FRAMES)
        ),
    }


def memory_policy_is_default(policy: dict[str, Any]) -> bool:
    return policy["slot_exclusivity"] == "off" and policy["memory_gate"] == "off"


def resolve_slot_exclusivity(
    masks_m1hw: Any,
    scores_m: Any,
    *,
    mode: str,
    loser_logit: float = EXCLUSIVITY_LOSER_LOGIT,
) -> tuple[Any, list[float]]:
    """Give each contested pixel to one multiplex slot before anything downstream sees it.

    A pixel is contested when more than one *present* slot (raw score > 0) predicts it
    positive.  In ``argmax`` mode the slot with the larger logit keeps it and every other
    present slot has that pixel pushed to at most ``loser_logit``; in ``off`` mode the logits
    are returned untouched.  Both modes report, per slot, the fraction of its positive pixels
    that were contested before resolution (0 when it had none or was absent).
    """
    import torch

    if mode not in SLOT_EXCLUSIVITY_MODES:
        raise ValueError(f"unknown slot exclusivity mode {mode!r}")
    slot_count = int(masks_m1hw.shape[0])
    if slot_count == 0:
        return masks_m1hw, []
    present = (scores_m.reshape(-1) > 0).reshape(slot_count, 1, 1, 1)
    positive = (masks_m1hw > 0) & present
    claims = positive.sum(dim=0, keepdim=True)
    contested = claims > 1
    positive_counts = positive.flatten(1).sum(dim=1)
    contested_counts = (positive & contested).flatten(1).sum(dim=1)
    fractions = torch.where(
        positive_counts > 0,
        contested_counts.to(torch.float32) / positive_counts.clamp(min=1).to(torch.float32),
        torch.zeros_like(positive_counts, dtype=torch.float32),
    )
    contested_fraction = [float(value) for value in fractions.tolist()]
    if mode == "off" or not bool(contested.any()):
        return masks_m1hw, contested_fraction
    competing_logits = masks_m1hw.to(torch.float32).masked_fill(~present, float("-inf"))
    winner = competing_logits.argmax(dim=0, keepdim=True)
    slot_index = torch.arange(slot_count, device=masks_m1hw.device).reshape(slot_count, 1, 1, 1)
    loser = positive & contested & (slot_index != winner)
    resolved = torch.where(loser, masks_m1hw.clamp(max=loser_logit), masks_m1hw)
    return resolved, contested_fraction


def memory_gate(
    scores_m: Any,
    ious_m: Any,
    contested_fraction: list[float],
    areas: list[int],
    area_history: list[list[float]],
    policy: dict[str, Any],
) -> tuple[Any, list[bool], list[str], list[list[float]]]:
    """Decide, per slot, whether this frame's mask may be memorised for that slot.

    Returns the scores to hand to ``encode_frame_memory`` (``GATED_OBJECT_SCORE`` for an
    untrusted slot, the raw score otherwise), a ``written`` flag and a reason per slot, and the
    updated per-slot history of trusted areas.  The area band is judged against the rolling
    median of the last ``gate_area_history_frames`` trusted areas and is never applied until
    that history is full (``warmup``).  With the gate off every slot is written with reason
    ``ok`` and the scores are returned as they came.
    """
    slot_count = int(scores_m.reshape(-1).shape[0])
    history = [list(entries) for entries in area_history]
    while len(history) < slot_count:
        history.append([])
    if policy["memory_gate"] == "off":
        return scores_m, [True] * slot_count, ["ok"] * slot_count, history
    import torch

    raw_scores = [float(value) for value in scores_m.reshape(-1).tolist()]
    raw_ious = [_scalar(ious_m, position) for position in range(slot_count)]
    band_low, band_high = policy["gate_area_band"]
    history_frames = int(policy["gate_area_history_frames"])
    written: list[bool] = []
    reasons: list[str] = []
    for slot in range(slot_count):
        area = float(areas[slot]) if slot < len(areas) else 0.0
        fraction = contested_fraction[slot] if slot < len(contested_fraction) else 0.0
        iou = raw_ious[slot]
        if raw_scores[slot] <= policy["gate_min_object_score"]:
            reason = "low_object_score"
        elif iou is not None and iou < policy["gate_min_iou"]:
            reason = "low_iou"
        elif fraction > policy["gate_max_contested_fraction"]:
            reason = "contested"
        elif len(history[slot]) < history_frames:
            reason = "warmup"
        else:
            median = float(sorted(history[slot])[len(history[slot]) // 2])
            ratio = area / median if median > 0 else float("inf")
            reason = "ok" if band_low <= ratio <= band_high else "area_jump"
        trusted = reason in ("ok", "warmup")
        written.append(trusted)
        reasons.append(reason)
        if trusted:
            history[slot] = (history[slot] + [area])[-history_frames:]
    gated = scores_m.clone()
    flat = gated.reshape(-1)
    for slot, trusted in enumerate(written):
        if not trusted:
            flat[slot] = torch.tensor(GATED_OBJECT_SCORE, dtype=flat.dtype, device=flat.device)
    return gated, written, reasons, history


def _slot_areas(masks_m1hw: Any) -> list[int]:
    """Positive-logit pixel count per slot on the model's mask grid."""
    return [int(value) for value in (masks_m1hw > 0).flatten(1).sum(dim=1).tolist()]


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
        resumed: dict[str, Any] | None = None
        if args.resume_from_checkpoint:
            resumed = torch.load(
                Path(args.resume_from_checkpoint), map_location="cuda:0", weights_only=False
            )
            if resumed.get("format") != CHECKPOINT_FORMAT:
                raise ValueError(
                    f"checkpoint format {resumed.get('format')!r} is not {CHECKPOINT_FORMAT!r}"
                )
            if resumed["stream_identity"] != stream_identity(args, concepts):
                raise ValueError(
                    "checkpoint was produced by a different stream: the video, model, encoder "
                    "settings, preprocessing, or initialization does not match this run"
                )
            if resumed["max_frame_memory"] != MAX_FRAME_MEMORY:
                raise ValueError(
                    f"checkpoint holds {resumed['max_frame_memory']} frame-memory entries; "
                    f"this run is configured for {MAX_FRAME_MEMORY}"
                )
        start_frame = 1 if resumed is None else int(resumed["frame_index"])
        if start_frame >= MAX_FRAMES:
            raise ValueError(
                f"checkpoint resumes at frame {start_frame}, at or past this run's {MAX_FRAMES}"
            )
        mode = "w" if resumed is None else "a"
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
        identity = stream_identity(args, concepts)
        policy = memory_policy_from_args(args)
        policy_default = memory_policy_is_default(policy)
        gate_area_history: list[list[float]] = (
            [list(entries) for entries in resumed.get("gate_area_history", [])]
            if resumed is not None
            else []
        )
        while len(gate_area_history) < len(initial_masks):
            gate_area_history.append([])
        wanted_checkpoints: frozenset[int] = frozenset()
        if not args.no_checkpoints:
            wanted_checkpoints = checkpoint_frames(
                every=args.checkpoint_every,
                correction_frames=tuple(correction_schedule),
                extra_frames=tuple(args.checkpoint_at),
                max_frames=MAX_FRAMES,
            )
        checkpoint_directory = run_directory / "native" / "checkpoints"

        def save_checkpoint(frame_index: int, prompt_memories: Any, frame_memories: Any) -> None:
            """Persist the state that is ready to process `frame_index`.

            Naming a checkpoint by the frame it has not yet stepped is what lets a rerun
            restart exactly at a correction keyframe and apply a new mask there.
            """
            checkpoint_directory.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "format": CHECKPOINT_FORMAT,
                    "frame_index": frame_index,
                    "stream_identity": identity,
                    "slots": [(index, concept) for index, concept, *_ in initial_masks],
                    "prompt_memories": list(prompt_memories),
                    "frame_memories": list(frame_memories),
                    "max_prompt_memory": MAX_PROMPT_MEMORY,
                    "max_frame_memory": MAX_FRAME_MEMORY,
                    "tracker_memory_policy": policy,
                    "gate_area_history": [list(entries) for entries in gate_area_history],
                },
                checkpoint_directory / f"f{frame_index:06d}.pt",
            )

        with output_path.open(mode) as observations:
            if resumed is None:
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
                if resumed is None:
                    prompt_memories = deque([initial_memory], maxlen=MAX_PROMPT_MEMORY)
                    frame_memories = deque([], maxlen=MAX_FRAME_MEMORY)
                else:
                    prompt_memories = deque(resumed["prompt_memories"], maxlen=MAX_PROMPT_MEMORY)
                    frame_memories = deque(resumed["frame_memories"], maxlen=MAX_FRAME_MEMORY)
                    # Decode, without encoding, up to the resumed frame: seeking a
                    # long-GOP proxy by index is not frame-exact in every backend.
                    for _ in range(1, start_frame):
                        if not capture.read()[0]:
                            raise RuntimeError(
                                f"proxy ended before the resumed frame {start_frame}"
                            )
                for frame_index in range(start_frame, MAX_FRAMES):
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if frame_index in wanted_checkpoints:
                        save_checkpoint(frame_index, prompt_memories, frame_memories)
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
                    memory_scores = scores
                    policy_diagnostics: list[dict[str, Any]] = []
                    if not policy_default:
                        # Exclusivity runs before the memory encoder and before the output so
                        # neither ever sees a pixel claimed by two present slots.
                        masks, contested_fraction = resolve_slot_exclusivity(
                            masks,
                            scores,
                            mode=policy["slot_exclusivity"],
                            loser_logit=policy["exclusivity_loser_logit"],
                        )
                        if correction_masks:
                            # A correction frame replaces the prompt memory outright; the
                            # gate has nothing to decide and the corrected slots start a
                            # fresh area history from their reviewed mask.
                            written = [True] * len(initial_masks)
                            reasons = ["corrected"] * len(initial_masks)
                            for slot in correction_masks:
                                gate_area_history[slot] = []
                        else:
                            memory_scores, written, reasons, gate_area_history = memory_gate(
                                scores,
                                ious,
                                contested_fraction,
                                _slot_areas(masks),
                                gate_area_history,
                                policy,
                            )
                        policy_diagnostics = [
                            {
                                "contested_fraction": contested_fraction[position],
                                "memory_written": written[position],
                                "memory_gate_reason": reasons[position],
                            }
                            for position in range(len(initial_masks))
                        ]
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
                            tracking.encode_frame_memory(encoded, masks, pointers, memory_scores)
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
                            **(policy_diagnostics[position] if policy_diagnostics else {}),
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
                for frame_index in range(start_frame, MAX_FRAMES):
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


def _positive_frame_count(value: str) -> int:
    frame_count = int(value)
    if frame_count <= 0:
        raise argparse.ArgumentTypeError("frame count must be positive")
    return frame_count


def _frame_list(value: str) -> tuple[int, ...]:
    frames = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    if any(frame < 1 for frame in frames):
        raise argparse.ArgumentTypeError("checkpoint frames must be at least 1")
    return frames


def stream_identity(args: argparse.Namespace, concepts: tuple[str, ...]) -> str:
    """Hash everything that decides what this stream produces frame by frame.

    A checkpoint may only be resumed by a run whose video, model, encoder settings,
    preprocessing, and initialization would have produced that exact tracker state.
    """
    video_path = Path(args.video)
    model_path = Path(args.model)
    payload = {
        "video": str(video_path.resolve()),
        "video_size_bytes": video_path.stat().st_size if video_path.is_file() else None,
        "model": str(model_path.resolve()),
        "model_size_bytes": model_path.stat().st_size if model_path.is_file() else None,
        "view_id": args.view_id,
        "source_offset_seconds": args.source_offset_seconds,
        "analysis_fps": args.analysis_fps,
        "max_side_length": args.max_side_length,
        "max_frame_memory": args.max_frame_memory,
        "preprocessing": args.preprocessing,
        "lower_percentile": args.lower_percentile,
        "upper_percentile": args.upper_percentile,
        "clahe_clip_limit": args.clahe_clip_limit,
        "clahe_tile_grid_size": args.clahe_tile_grid_size,
        "prompt_mode": args.prompt_mode,
        "concepts": list(concepts),
        "text_targets_json": args.text_targets_json,
        "hybrid_initialization_json": args.hybrid_initialization_json,
        "manual_box_json": args.manual_box_json,
        "manual_seeds_json": args.manual_seeds_json,
        "multi_keyframe_schedule_json": args.multi_keyframe_schedule_json,
    }
    # Only a non-default policy joins the identity, so checkpoints written before the policy
    # existed stay resumable by a run that does not use it.
    policy = memory_policy_from_args(args)
    if not memory_policy_is_default(policy):
        payload["tracker_memory_policy"] = policy
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _area_band(value: str) -> tuple[float, float]:
    parts = [float(item) for item in value.split(",") if item.strip()]
    if len(parts) != 2 or parts[0] <= 0 or parts[1] < parts[0]:
        raise argparse.ArgumentTypeError("area band must be 'low,high' with 0 < low <= high")
    return parts[0], parts[1]


def checkpoint_frames(
    *,
    every: int,
    correction_frames: tuple[int, ...],
    extra_frames: tuple[int, ...],
    max_frames: int,
) -> frozenset[int]:
    """Choose the frames whose tracker state is worth keeping.

    Correction keyframes are always included: they are where a reviewer re-seeds a
    target, so they are exactly the frames a later run wants to restart from.
    """
    frames = {frame for frame in correction_frames if 0 < frame < max_frames}
    frames.update(frame for frame in extra_frames if 0 < frame < max_frames)
    if every > 0:
        frames.update(range(every, max_frames, every))
    return frozenset(frames)


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
    parser.add_argument("--max-frames", type=_positive_frame_count, default=300)
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
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Save tracker state every N frames; 0 saves only the correction keyframes.",
    )
    parser.add_argument(
        "--checkpoint-at",
        type=_frame_list,
        default=(),
        help="Additional comma-separated frames whose tracker state should be saved.",
    )
    parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="Save no tracker state, not even at correction keyframes.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        help="Restore tracker state from this checkpoint and continue from its frame.",
    )
    parser.add_argument(
        "--slot-exclusivity",
        choices=SLOT_EXCLUSIVITY_MODES,
        default="off",
        help="argmax: a pixel positive in several present slots goes to the highest logit.",
    )
    parser.add_argument(
        "--exclusivity-loser-logit",
        type=float,
        default=EXCLUSIVITY_LOSER_LOGIT,
        help="Upper bound applied to a losing slot's logit on a contested pixel.",
    )
    parser.add_argument(
        "--memory-gate",
        choices=MEMORY_GATE_MODES,
        default="off",
        help="on: memorise a slot as absent on frames where its prediction is not trusted.",
    )
    parser.add_argument("--gate-min-object-score", type=float, default=GATE_MIN_OBJECT_SCORE)
    parser.add_argument("--gate-min-iou", type=float, default=GATE_MIN_IOU)
    parser.add_argument(
        "--gate-max-contested-fraction", type=float, default=GATE_MAX_CONTESTED_FRACTION
    )
    parser.add_argument(
        "--gate-area-band",
        type=_area_band,
        default=GATE_AREA_BAND,
        help="Accepted mask area as 'low,high' multiples of the slot's rolling median area.",
    )
    parser.add_argument(
        "--gate-area-history-frames",
        type=int,
        default=GATE_AREA_HISTORY_FRAMES,
        help="Trusted frames behind the rolling median; the band is not applied before then.",
    )
    args = parser.parse_args()
    if args.gate_area_history_frames < 1:
        parser.error("--gate-area-history-frames must be at least 1")
    if args.mask_period_frames < 1:
        parser.error("--mask-period-frames must be at least 1")
    if args.checkpoint_every < 0:
        parser.error("--checkpoint-every must not be negative")
    MAX_FRAMES = args.max_frames
    MAX_SIDE_LENGTH = args.max_side_length
    MAX_FRAME_MEMORY = args.max_frame_memory
    ANALYSIS_FPS = args.analysis_fps
    MASK_PERIOD_FRAMES = args.mask_period_frames
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
