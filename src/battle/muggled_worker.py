"""Dependency-light SAM3 worker run by the separately managed MuggledSAM interpreter.

This module deliberately imports only the standard library until after preflight.  The
project's parent process validates its JSON output against Battle's Pydantic contracts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import traceback
from collections import deque
from collections.abc import Iterable
from pathlib import Path
from time import perf_counter
from typing import Any

try:
    from . import fs_common, gpu_guard
except ImportError:
    # Run as a script by the MuggledSAM interpreter (or loaded by file path): the guard and
    # helper modules sit beside this file and are imported as top-level modules, like the
    # DAM4SAM workers do.
    _HERE = str(Path(__file__).resolve().parent)
    if _HERE not in sys.path:
        sys.path.append(_HERE)
    import fs_common  # type: ignore[no-redef]
    import gpu_guard  # type: ignore[no-redef]

CONCEPTS = ("hand", "yellow toy body", "toy wheel")
# Every frame by default: the exporter logs masks as compressed PNGs, so full-rate masks are
# affordable in the viewer, and a binary PNG per object per frame is small on disk.
MASK_PERIOD_FRAMES = 1
MAX_FRAMES = 300
MAX_FRAME_MEMORY = 4
MAX_PROMPT_MEMORY = 1
# Correction memory semantics.  Every default reproduces the runs made before the flags
# existed: one replaced prompt entry, frame memory cleared at a correction, frame memories
# handed to the fusion model oldest-first (so `is_recent_first` is False).
PROMPT_MEMORY_SEMANTICS = ("replace", "append")
DEFAULT_PROMPT_MEMORY_SEMANTICS = "replace"
APPEND_PROMPT_MEMORY_DEFAULT_ENTRIES = 32
KEEP_FRAME_MEMORY_AT_CORRECTION = False
IS_RECENT_FIRST = False
# MuggledSAM's SAM3 memory fusion has learned position offsets for six frame-memory entries
# (`memory_image_fusion_model.py`: `max_memory_history = 6`); older deltas are clamped onto
# the sixth, so a larger bank runs but with a position encoding the model never trained on.
TRAINED_MAX_FRAME_MEMORY = 6
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
# FineBio arms (Sep 24).  `box_stream` is the memory-free per-frame decode: every row it
# writes carries this `source`, and its `object_score` is the image decoder's IoU prediction
# for the chosen candidate, not a tracker presence logit.  A correction (or a mid-stream seed)
# may be a box instead of a mask; these are the provenance kinds a box prompt may carry.
BOX_STREAM_SOURCE = "sam3_decode"
BOX_PROMPT_SELECTED_BY = ("detector_reseed", "track_reproject")
# `--memory-write-min-score`: off by default, so a run without the flag is byte-identical to
# the runs made before it existed.  The Sep 18 `memory_gate on` bundles four tests (score,
# predicted IoU, contested fraction, area band); this is the score test alone.
MEMORY_WRITE_MIN_SCORE: float | None = None

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
    memory = memory_settings_from_args(args)
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
        "max_prompt_memory_entries": memory["max_prompt_memory"],
        "max_frame_memory_entries": memory["max_frame_memory"],
        "prompt_memory_semantics": memory["prompt_memory_semantics"],
        "keep_frame_memory_at_correction": memory["keep_frame_memory_at_correction"],
        "is_recent_first": memory["is_recent_first"],
        "is_recent_first_semantics": (
            "the worker appends frame memories oldest-to-newest; is_recent_first is the "
            "ordering the fusion model is told to assume, so True reverses the temporal "
            "position encoding of the frame memories and object pointers (the oldest entry "
            "is encoded as the most recent). It does not reorder the memory bank."
        ),
        "frame_memory_position_encoding": frame_memory_position_encoding(
            memory["max_frame_memory"]
        ),
        "correction_memory_semantics": correction_memory_semantics(
            prompt_memory_semantics=memory["prompt_memory_semantics"],
            keep_frame_memory_at_correction=memory["keep_frame_memory_at_correction"],
        ),
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
        "gpu_guard_mode": getattr(args, "gpu_guard", gpu_guard.DEFAULT_GUARD_MODE),
        "tracker_memory_policy": memory_policy_from_args(args),
        "tracker_memory_policy_semantics": (
            "slot_exclusivity argmax gives every pixel predicted positive by more than one "
            "present slot to the slot with the larger logit and pushes the others to at most "
            "exclusivity_loser_logit before memory encoding and output; memory_gate on hands "
            "the memory encoder a score of -1 (its no-object embedding) for a slot whose raw "
            "score, predicted IoU, contested fraction or area (against the rolling median of "
            "its trusted frames) fails the thresholds, so that frame is memorised as absent "
            "for that slot only. memory_write_min_score (tau) is the score test alone: a "
            "present slot whose raw score is below tau is memorised as absent for that frame "
            "and still reported in the observations. Both off and tau unset reproduces the "
            "unpoliced tracker exactly. Neither is an accuracy claim."
        ),
    }
    if getattr(args, "start_frame", 0):
        settings["start_frame"] = int(args.start_frame)
        settings["start_frame_semantics"] = (
            "analysis frame 0 is this source frame; the worker decodes and discards the "
            "earlier frames rather than seeking, so the index is exact on every backend"
        )
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
        box_prompt_frames = sorted(
            {
                int(entry["frame_index"])
                for entry in schedule["corrections"]
                if _has_box_prompt(entry)
            }
        )
        start_frames = slot_start_frames(schedule)
        if (
            box_prompt_frames
            or start_frames
            or any(_has_box_prompt(seed) for seed in schedule["seeds"])
        ):
            settings["multi_keyframe_correction_schedule"].update(
                {
                    "box_prompt_correction_frames": box_prompt_frames,
                    "correction_selected_by_kinds": sorted(
                        {
                            str(entry.get("selected_by", "human"))
                            for entry in schedule["corrections"]
                        }
                        | {
                            str(schedule["seeds"][slot].get("selected_by") or "detector_reseed")
                            for slot in start_frames
                        }
                    ),
                    "slot_start_frames": {
                        str(slot): frame for slot, frame in sorted(start_frames.items())
                    },
                    "box_prompt_seed_slots": [
                        int(seed["initial_multiplex_slot"])
                        for seed in schedule["seeds"]
                        if _has_box_prompt(seed)
                    ],
                }
            )
            settings["box_prompt_api"] = (
                "interactive encode_prompts([box], [], []) + generate_masks on the tracking "
                "encoder's image tokens; top-IoU candidate, logits > 0, rebased into the "
                "multiplex batch and installed with encode_prompt_memory_from_mask"
            )
            settings["slot_start_frame_semantics"] = (
                "the multiplex object count is fixed when the first prompt memory is encoded, "
                "so a slot with start_frame > 0 is allocated at frame 0 with an empty mask, "
                "forced absent (no object row, memory score -1, reason unseeded) until its "
                "start frame, where its seed prompt is applied through the correction path"
            )
        settings["initialization_api"] = "encode_prompt_memory_from_mask"
        settings["correction_api"] = "encode_prompt_memory_from_mask"
        # `correction_memory_semantics` is derived from the flags above; `_corrections_by_frame`
        # refuses a schedule payload that declares anything else.
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
    """The `--gpu-guard strict` rule (Sep 21 logic, kept verbatim in `gpu_guard`).

    Flags GPU processes whose nvidia-smi name could be running a model, tolerating the Rerun
    viewer.  Every process still appears in ``gpu_processes_before_initialization`` for the
    record; per-process peak VRAM is unaffected, though wall-clock timing may see contention.
    """
    return gpu_guard.looks_like_model_process(process)


def _is_rerun_viewer(name: str) -> bool:
    return gpu_guard.is_rerun_viewer(name)


def partition_gpu_neighbours(
    gpu_processes: list[dict[str, str]], allowed_pids: Iterable[int]
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Strict-mode split of the model-like GPU processes into (blocking, allowed-by-operator).

    The guard refuses to start beside another model process so that VRAM peaks and timings
    stay per-process.  An operator may name specific PIDs (never process names) that are
    allowed to share the GPU, e.g. a human's calibration workspace worker that must not be
    stopped; every allowed neighbour is recorded in the run's settings as provenance.
    """
    return gpu_guard.partition_neighbours_strict(gpu_processes, allowed_pids)


def _default_gpu_guard_profile(args: argparse.Namespace) -> str:
    return gpu_guard.sam3_profile_for_side_length(int(args.max_side_length))


def gpu_guard_decision(
    args: argparse.Namespace, gpu_processes: list[dict[str, str]]
) -> tuple[str | None, dict[str, Any] | None]:
    """(refusal reason or None, `runtime_settings.gpu_guard` record or None) for this worker.

    ``strict`` reproduces the Sep 21 behaviour exactly on the `gpu_processes` already read
    for ``gpu_processes_before_initialization``: nvidia-smi names, model-like tokens, a record
    only when the operator named neighbours.  ``vram`` queries the card itself, classifies
    every neighbour by its /proc cmdline and checks the headroom against this profile's
    expected peak; its record is always written so a later reader can see what shared the
    card.
    """
    mode = getattr(args, "gpu_guard", gpu_guard.DEFAULT_GUARD_MODE)
    if mode == "strict":
        concurrent_models, tolerated_neighbours = partition_gpu_neighbours(
            gpu_processes, args.allow_gpu_neighbour
        )
        record: dict[str, Any] | None = None
        if args.allow_gpu_neighbour:
            record = {
                "allowed_neighbour_pids": [int(pid) for pid in args.allow_gpu_neighbour],
                "tolerated_neighbours": tolerated_neighbours,
                "semantics": (
                    "the operator allowed these GPU processes (by PID) to run beside this "
                    "worker; gpu_peak_vram_bytes stays this process's own allocation but "
                    "wall-clock timing may see contention; the guard still refuses any other "
                    "model process"
                ),
            }
        if concurrent_models:
            return f"concurrent GPU model process(es) detected: {concurrent_models}", record
        return None, record
    decision = gpu_guard.evaluate(
        mode=mode,
        allowed_pids=args.allow_gpu_neighbour,
        profile=getattr(args, "gpu_guard_profile", None) or _default_gpu_guard_profile(args),
        expected_peak_vram_bytes=getattr(args, "expected_peak_vram_bytes", None),
        own_pid=os.getpid(),
    )
    return decision.reason, decision.as_provenance()


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
    extras: dict[int, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], int]:
    """One observations.jsonl row; `extras` adds per-object fields keyed by object index."""
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
                **((extras or {}).get(object_index, {})),
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
    schedule: dict[str, Any],
    concepts: tuple[str, ...],
    *,
    memory_semantics: str = "replace_prompt_memory_and_reset_frame_memory",
) -> dict[int, list[dict[str, Any]]]:
    """Validate worker payload slots before grouping optional later corrections.

    The payload must declare the semantics this worker is configured to run, so a driver
    cannot hand a replace-semantics schedule to an append-semantics worker unnoticed.
    """
    if schedule.get("memory_semantics") != memory_semantics:
        raise ValueError(
            f"unsupported correction memory semantics {schedule.get('memory_semantics')!r}; "
            f"this worker is configured for {memory_semantics!r}"
        )
    seeds = list(schedule.get("seeds", []))
    _validate_manual_seed_slots(seeds, concepts)
    grouped: dict[int, list[dict[str, Any]]] = {}
    seen: set[tuple[int, int]] = set()
    for correction in schedule.get("corrections", []):
        frame_index, slot = int(correction["frame_index"]), int(correction["multiplex_slot"])
        if frame_index <= 0 or slot < 0 or slot >= len(concepts):
            raise ValueError("correction frames must be positive and slots must name a target")
        if correction.get("target") != concepts[slot] or (frame_index, slot) in seen:
            raise ValueError("correction schedule has an ambiguous frame/slot assignment")
        _validate_prompt_entry(correction, later_correction=True)
        seen.add((frame_index, slot))
        grouped.setdefault(frame_index, []).append(correction)
    # A slot that starts mid-stream applies its own seed prompt through the correction path
    # at its start frame; the slot itself exists from frame 0 (see `slot_start_frames`).
    for slot, frame_index in slot_start_frames(schedule).items():
        if (frame_index, slot) in seen:
            raise ValueError("a slot's start frame cannot also carry a later correction")
        seed = seeds[slot]
        _validate_prompt_entry(seed, later_correction=False)
        if seed.get("prompt_box") is None and seed.get("prompt_box_xyxy_px") is None:
            if not seed.get("mask_path") or not seed.get("mask_sha256"):
                raise ValueError("a mid-stream seed needs a prompt_box or a verified mask")
        seen.add((frame_index, slot))
        grouped.setdefault(frame_index, []).append(
            {
                **seed,
                "frame_index": frame_index,
                "multiplex_slot": slot,
                "selected_by": str(seed.get("selected_by") or BOX_PROMPT_SELECTED_BY[0]),
                "seed_start": True,
            }
        )
    return grouped


def slot_start_frames(schedule: dict[str, Any]) -> dict[int, int]:
    """{slot: start frame} for every seed that starts after frame 0; empty for today's payloads."""
    starts: dict[int, int] = {}
    for slot, seed in enumerate(schedule.get("seeds", [])):
        start = int(seed.get("start_frame", 0) or 0)
        if start < 0:
            raise ValueError("a seed's start_frame must not be negative")
        if start:
            starts[slot] = start
    return starts


def _validate_prompt_entry(entry: dict[str, Any], *, later_correction: bool) -> None:
    """A seed or correction carries a mask, a box, or (legacy payloads) neither; never both.

    A box that re-prompts a slot after frame 0 must say where it came from: the detector
    (`detector_reseed`) or the 3D track's reprojection (`track_reproject`).
    """
    has_mask = entry.get("mask_path") is not None or entry.get("mask_sha256") is not None
    box = entry.get("prompt_box")
    box_px = entry.get("prompt_box_xyxy_px")
    if box is not None and box_px is not None:
        raise ValueError("a prompt entry carries prompt_box or prompt_box_xyxy_px, not both")
    if box is None and box_px is None:
        return
    if has_mask:
        raise ValueError("a prompt entry carries a mask or a box, not both")
    if box is not None:
        if not isinstance(box, dict) or set(box) != {"x", "y", "width", "height"}:
            raise ValueError("prompt_box must be a normalised {x, y, width, height} box")
        values = [box[key] for key in ("x", "y", "width", "height")]
        if not all(_is_finite_number(value) for value in values):
            raise ValueError("prompt_box values must be finite numbers")
        x, y, width, height = (float(value) for value in values)
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and width > 0 and height > 0):
            raise ValueError("prompt_box must lie inside the unit square with positive size")
        if x + width > 1.0 + 1e-6 or y + height > 1.0 + 1e-6:
            raise ValueError("prompt_box must lie inside the unit square with positive size")
    else:
        _xyxy(box_px, normalised=False)
    if later_correction and entry.get("selected_by") not in BOX_PROMPT_SELECTED_BY:
        raise ValueError(
            "a box correction must be selected_by one of "
            + ", ".join(BOX_PROMPT_SELECTED_BY)
            + f"; got {entry.get('selected_by')!r}"
        )


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value == value
        and value not in (float("inf"), float("-inf"))
    )


def _xyxy(values: Any, *, normalised: bool) -> tuple[float, float, float, float]:
    """Four finite numbers, x1 > x0 and y1 > y0; inside the unit square when normalised."""
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError("a box must be four numbers [x0, y0, x1, y1]")
    if not all(_is_finite_number(value) for value in values):
        raise ValueError("box coordinates must be finite numbers")
    x0, y0, x1, y1 = (float(value) for value in values)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("a box needs x1 > x0 and y1 > y0")
    if normalised and not all(0.0 <= value <= 1.0 for value in (x0, y0, x1, y1)):
        raise ValueError("a normalised box must lie inside the unit square")
    return x0, y0, x1, y1


def prompt_box_xyxy(
    entry: dict[str, Any], frame_shape: tuple[int, int]
) -> tuple[float, float, float, float]:
    """The normalised (x0, y0, x1, y1) of a seed/correction/box-stream entry, clamped to [0, 1]."""
    height, width = frame_shape
    if entry.get("prompt_box") is not None:
        box = entry["prompt_box"]
        raw = (
            float(box["x"]),
            float(box["y"]),
            float(box["x"]) + float(box["width"]),
            float(box["y"]) + float(box["height"]),
        )
    elif entry.get("prompt_box_xyxy_px") is not None:
        x0, y0, x1, y1 = _xyxy(entry["prompt_box_xyxy_px"], normalised=False)
        raw = (x0 / width, y0 / height, x1 / width, y1 / height)
    elif entry.get("box_xyxy_norm") is not None:
        raw = _xyxy(entry["box_xyxy_norm"], normalised=True)
    elif entry.get("box_xyxy_px") is not None:
        x0, y0, x1, y1 = _xyxy(entry["box_xyxy_px"], normalised=False)
        raw = (x0 / width, y0 / height, x1 / width, y1 / height)
    else:
        raise ValueError("entry carries no box prompt")
    x0, y0, x1, y1 = (min(max(value, 0.0), 1.0) for value in raw)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("box prompt lies outside the frame")
    return x0, y0, x1, y1


def normalised_box_record(xyxy: tuple[float, float, float, float]) -> dict[str, float]:
    """The repo's `{x, y, width, height}` form of a normalised (x0, y0, x1, y1) box.

    `NormalizedBox` requires ``x + width <= 1`` exactly, so the extent is nudged down by an
    ulp where floating-point rounding would put it a hair past the edge.
    """
    x0, y0, x1, y1 = xyxy
    width, height = x1 - x0, y1 - y0
    while x0 + width > 1.0:
        width = math.nextafter(width, 0.0)
    while y0 + height > 1.0:
        height = math.nextafter(height, 0.0)
    return {"x": x0, "y": y0, "width": width, "height": height}


def parse_box_stream(text: str) -> tuple[dict[int, list[dict[str, Any]]], tuple[str, ...]]:
    """Validate a per-frame box stream into ``{frame_index: [box prompt, ...]}`` plus slot labels.

    One JSON record per line::

        {"frame_index": 12, "boxes": [{"slot": 0, "label": "cell_culture_plate",
          "box_xyxy_px": [857.7, 462.8, 1216.6, 714.4], "score": 0.42, "source": "finebio_dino"}]}

    A box carries exactly one of ``box_xyxy_px`` (source pixels) or ``box_xyxy_norm`` (unit
    square); ``score`` and ``source`` are optional provenance copied into the observations.  A
    frame absent from the stream, or listed with no boxes, produces an empty observation row.
    Slots are contiguous from zero over the whole stream and a slot always carries one label,
    so the labels double as the run's ordered concepts.
    """
    frames: dict[int, list[dict[str, Any]]] = {}
    labels: dict[int, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"box stream line {line_number} is not JSON: {error}") from None
        if not isinstance(record, dict):
            raise ValueError(f"box stream line {line_number} must be a JSON object")
        frame_index = record.get("frame_index")
        if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
            raise ValueError(f"box stream line {line_number}: frame_index must be an int >= 0")
        if frame_index in frames:
            raise ValueError(f"box stream line {line_number}: frame {frame_index} repeats")
        boxes = record.get("boxes", [])
        if not isinstance(boxes, list):
            raise ValueError(f"box stream line {line_number}: boxes must be a list")
        parsed: list[dict[str, Any]] = []
        slots_seen: set[int] = set()
        for box in boxes:
            prompt = _box_stream_prompt(box, line_number)
            slot = prompt["slot"]
            if slot in slots_seen:
                raise ValueError(f"box stream line {line_number}: slot {slot} repeats")
            slots_seen.add(slot)
            if labels.setdefault(slot, prompt["label"]) != prompt["label"]:
                raise ValueError(
                    f"box stream line {line_number}: slot {slot} is labelled "
                    f"{prompt['label']!r} but was {labels[slot]!r} earlier"
                )
            parsed.append(prompt)
        parsed.sort(key=lambda item: item["slot"])
        frames[frame_index] = parsed
    if labels and sorted(labels) != list(range(len(labels))):
        raise ValueError("box stream slots must be contiguous from zero")
    return frames, tuple(labels[slot] for slot in range(len(labels)))


def _box_stream_prompt(box: Any, line_number: int) -> dict[str, Any]:
    if not isinstance(box, dict):
        raise ValueError(f"box stream line {line_number}: every box must be a JSON object")
    slot = box.get("slot")
    if isinstance(slot, bool) or not isinstance(slot, int) or slot < 0:
        raise ValueError(f"box stream line {line_number}: slot must be an int >= 0")
    label = box.get("label")
    if not isinstance(label, str) or not label.strip():
        raise ValueError(f"box stream line {line_number}: label must be a non-empty string")
    pixels, unit = box.get("box_xyxy_px"), box.get("box_xyxy_norm")
    if (pixels is None) == (unit is None):
        raise ValueError(
            f"box stream line {line_number}: exactly one of box_xyxy_px / box_xyxy_norm"
        )
    try:
        values = _xyxy(pixels if pixels is not None else unit, normalised=unit is not None)
    except ValueError as error:
        raise ValueError(f"box stream line {line_number}: {error}") from None
    score = box.get("score")
    if score is not None and not _is_finite_number(score):
        raise ValueError(f"box stream line {line_number}: score must be a number or null")
    source = box.get("source", "unknown")
    if not isinstance(source, str) or not source.strip():
        raise ValueError(f"box stream line {line_number}: source must be a non-empty string")
    return {
        "slot": slot,
        "label": label,
        "box_xyxy_px": list(values) if pixels is not None else None,
        "box_xyxy_norm": list(values) if unit is not None else None,
        "score": float(score) if score is not None else None,
        "source": source,
    }


def decode_box_prompts(
    interact: Any,
    encoded_image: Any,
    boxes_xyxy_norm: list[tuple[float, float, float, float]],
    *,
    other_boxes_as_negatives: bool = False,
) -> tuple[Any, list[float], list[int]]:
    """One batched image-decoder pass over N box prompts; the top-IoU candidate of each.

    MuggledSAM's `encode_prompts` batches prompts along B (`BxNx2x2` boxes, `BxKx2` points)
    and the mask decoder expands one image encoding over that batch, so N boxes cost one
    decoder call.  With ``other_boxes_as_negatives`` every prompt also carries the other
    boxes' centres as background points (the same K for every prompt, as batching needs).
    Returns the chosen logits as ``Bx1xHxW`` on the model grid, the predicted IoU per box and
    the chosen candidate index per box.
    """
    import torch

    if not boxes_xyxy_norm:
        raise ValueError("decode_box_prompts needs at least one box")
    boxes = torch.tensor([[[[x0, y0], [x1, y1]]] for x0, y0, x1, y1 in boxes_xyxy_norm])
    background: Any = []
    if other_boxes_as_negatives and len(boxes_xyxy_norm) > 1:
        centres = [((x0 + x1) / 2, (y0 + y1) / 2) for x0, y0, x1, y1 in boxes_xyxy_norm]
        background = torch.tensor(
            [
                [centre for other, centre in enumerate(centres) if other != index]
                for index in range(len(centres))
            ]
        )
    prompts = interact.encode_prompts(boxes, [], background)
    masks, ious = interact.generate_masks(encoded_image, prompts)
    best = ious.argmax(dim=1)
    rows = torch.arange(masks.shape[0], device=masks.device)
    chosen = masks[rows, best].unsqueeze(1)
    return (
        chosen,
        [float(value) for value in ious[rows, best].float().tolist()],
        [int(value) for value in best.tolist()],
    )


def memory_write_allowed(score: float, tau: float | None) -> bool:
    """The tau hook: may this slot's mask enter frame memory on this frame?

    ``tau`` unset writes every slot (the pre-flag behaviour).  Otherwise a slot is written
    only when its raw presence score is at least ``tau``; below it the frame is memorised as
    absent for that slot while the observation row still reports the mask.
    """
    return tau is None or score >= tau


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


def correction_memory_semantics(
    *, prompt_memory_semantics: str, keep_frame_memory_at_correction: bool
) -> str:
    """Name what a correction does to the two memory banks, as recorded in every manifest."""
    if prompt_memory_semantics not in PROMPT_MEMORY_SEMANTICS:
        raise ValueError(f"unknown prompt memory semantics {prompt_memory_semantics!r}")
    frame_action = "keep" if keep_frame_memory_at_correction else "reset"
    return f"{prompt_memory_semantics}_prompt_memory_and_{frame_action}_frame_memory"


def default_max_prompt_memory(prompt_memory_semantics: str) -> int:
    """One replaced entry today; the demo's 32-entry bank when corrections are appended."""
    if prompt_memory_semantics not in PROMPT_MEMORY_SEMANTICS:
        raise ValueError(f"unknown prompt memory semantics {prompt_memory_semantics!r}")
    return (
        APPEND_PROMPT_MEMORY_DEFAULT_ENTRIES
        if prompt_memory_semantics == "append"
        else MAX_PROMPT_MEMORY
    )


def frame_memory_position_encoding(max_frame_memory: int) -> str:
    """Say whether every frame-memory entry gets a position offset the model was trained with."""
    if max_frame_memory > TRAINED_MAX_FRAME_MEMORY:
        return "clamped_beyond_6"
    return "within_trained_range"


def memory_settings_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Read the memory-bank condition as one plain record, so it can be logged and compared.

    Missing attributes fall back to the pre-flag behaviour, which is also how a checkpoint
    written before these settings existed is interpreted on resume.
    """
    semantics = getattr(args, "prompt_memory_semantics", DEFAULT_PROMPT_MEMORY_SEMANTICS)
    if semantics not in PROMPT_MEMORY_SEMANTICS:
        raise ValueError(f"unknown prompt memory semantics {semantics!r}")
    max_prompt_memory = getattr(args, "max_prompt_memory", None)
    if max_prompt_memory is None:
        max_prompt_memory = default_max_prompt_memory(semantics)
    if int(max_prompt_memory) < 1:
        raise ValueError("max prompt memory must be at least 1")
    if semantics == "replace" and int(max_prompt_memory) != MAX_PROMPT_MEMORY:
        raise ValueError(
            "replace semantics keeps exactly one prompt memory entry; "
            f"got max prompt memory {max_prompt_memory}"
        )
    return {
        "prompt_memory_semantics": semantics,
        "max_prompt_memory": int(max_prompt_memory),
        "max_frame_memory": int(getattr(args, "max_frame_memory", MAX_FRAME_MEMORY)),
        "keep_frame_memory_at_correction": bool(
            getattr(args, "keep_frame_memory_at_correction", KEEP_FRAME_MEMORY_AT_CORRECTION)
        ),
        "is_recent_first": bool(getattr(args, "recent_first", IS_RECENT_FIRST)),
    }


def memory_settings_are_default(settings: dict[str, Any]) -> bool:
    """True when the run is byte-identical to one made before the memory flags existed."""
    return (
        settings["prompt_memory_semantics"] == DEFAULT_PROMPT_MEMORY_SEMANTICS
        and settings["max_prompt_memory"] == MAX_PROMPT_MEMORY
        and settings["keep_frame_memory_at_correction"] is KEEP_FRAME_MEMORY_AT_CORRECTION
        and settings["is_recent_first"] is IS_RECENT_FIRST
    )


# Checkpoint keys added with the memory flags, and how a checkpoint written without them
# is read: every one of these means "the behaviour before the flag existed".
LEGACY_CHECKPOINT_MEMORY_SETTINGS: dict[str, Any] = {
    "prompt_memory_semantics": DEFAULT_PROMPT_MEMORY_SEMANTICS,
    "max_prompt_memory": MAX_PROMPT_MEMORY,
    "keep_frame_memory_at_correction": KEEP_FRAME_MEMORY_AT_CORRECTION,
    "is_recent_first": IS_RECENT_FIRST,
}


def check_checkpoint_memory_settings(checkpoint: dict[str, Any], settings: dict[str, Any]) -> None:
    """Refuse to resume a checkpoint into a run with a different memory-bank condition.

    A resumed stream must continue exactly as it would have run unbroken, so every memory
    setting has to match.  `max_frame_memory` has always been stored; the newer keys are
    read with their pre-flag defaults so older checkpoints still resume into default runs.
    """
    if checkpoint["max_frame_memory"] != settings["max_frame_memory"]:
        raise ValueError(
            f"checkpoint holds {checkpoint['max_frame_memory']} frame-memory entries; "
            f"this run is configured for {settings['max_frame_memory']}"
        )
    for key, legacy_value in LEGACY_CHECKPOINT_MEMORY_SETTINGS.items():
        stored = checkpoint.get(key, legacy_value)
        if stored != settings[key]:
            raise ValueError(
                f"checkpoint was written with {key}={stored!r}; "
                f"this run is configured for {settings[key]!r}"
            )


def build_memory_banks(
    *,
    prompt_memories: list[Any],
    frame_memories: list[Any],
    max_prompt_memory: int,
    max_frame_memory: int,
) -> tuple[deque[Any], deque[Any]]:
    """Bounded prompt and frame banks; the oldest entry falls off when either is full."""
    return (
        deque(prompt_memories, maxlen=max_prompt_memory),
        deque(frame_memories, maxlen=max_frame_memory),
    )


def apply_correction_to_memory(
    *,
    prompt_memories: deque[Any],
    frame_memories: deque[Any],
    correction_memory: Any,
    prompt_memory_semantics: str = DEFAULT_PROMPT_MEMORY_SEMANTICS,
    keep_frame_memory_at_correction: bool = KEEP_FRAME_MEMORY_AT_CORRECTION,
) -> None:
    """Install one freshly encoded correction prompt memory under the configured semantics.

    ``replace`` drops every earlier prompt entry (the seed included) so the corrected frame
    is the only prompt; ``append`` keeps the seed and every earlier correction as a bank and
    lets the deque bound evict the oldest.  Frame memory is cleared unless asked to keep it.
    """
    if prompt_memory_semantics not in PROMPT_MEMORY_SEMANTICS:
        raise ValueError(f"unknown prompt memory semantics {prompt_memory_semantics!r}")
    if prompt_memory_semantics == "replace":
        prompt_memories.clear()
    prompt_memories.append(correction_memory)
    if not keep_frame_memory_at_correction:
        frame_memories.clear()


def _replace_prompt_memory_for_correction(
    *,
    predicted_source_masks: Any,
    correction_masks_by_slot: dict[int, Any],
    prompt_memories: deque[Any],
    frame_memories: deque[Any],
    encoded_frame: Any,
    encode_prompt_memory_from_mask: Any,
    prompt_memory_semantics: str = DEFAULT_PROMPT_MEMORY_SEMANTICS,
    keep_frame_memory_at_correction: bool = KEEP_FRAME_MEMORY_AT_CORRECTION,
) -> Any:
    """Encode a corrected full multiplex mask batch as prompt memory and install it."""
    if predicted_source_masks.shape[0] == 0:
        raise ValueError("cannot apply a correction without multiplex predictions")
    rebased_masks = predicted_source_masks.copy()
    for slot, mask in correction_masks_by_slot.items():
        if slot < 0 or slot >= rebased_masks.shape[0] or mask.shape != rebased_masks.shape[1:]:
            raise ValueError("correction mask does not match the multiplex source-mask shape")
        rebased_masks[slot] = mask
    correction_memory = encode_prompt_memory_from_mask(encoded_frame, rebased_masks)
    apply_correction_to_memory(
        prompt_memories=prompt_memories,
        frame_memories=frame_memories,
        correction_memory=correction_memory,
        prompt_memory_semantics=prompt_memory_semantics,
        keep_frame_memory_at_correction=keep_frame_memory_at_correction,
    )
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


def _open_capture(video_path: Path, start_frame: int) -> Any:
    """Open the video positioned so that the next `read()` returns `start_frame`.

    Frames before it are decoded and discarded rather than sought: seeking a long-GOP file
    by index is not frame-exact in every backend, and the FineBio proxies index the shipped
    per-frame pose by raw frame number.
    """
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    for skipped in range(start_frame):
        if not capture.grab():
            raise RuntimeError(f"video ended at frame {skipped}, before start frame {start_frame}")
    return capture


def _has_box_prompt(entry: dict[str, Any]) -> bool:
    return entry.get("prompt_box") is not None or entry.get("prompt_box_xyxy_px") is not None


def _prompt_mask_batch(masks_bhw: Any) -> Any:
    """The batch handed to `encode_prompt_memory_from_mask` for the frame-0 seeds.

    MuggledSAM rescales a NumPy mask by its own min and max, which is a division by zero
    when a slot is empty (a mid-stream start).  In that case the batch is handed over as a
    tensor of the same +/-1024 logits the NumPy path would have produced, per pixel; when
    every slot has pixels the NumPy path is kept so existing runs stay byte-identical.
    """
    if bool(masks_bhw.reshape(masks_bhw.shape[0], -1).any(axis=1).all()):
        return masks_bhw
    import torch

    return torch.tensor(masks_bhw, dtype=torch.float32) * 2048.0 - 1024.0


def _correction_masks_for_frame(
    corrections: list[dict[str, Any]],
    *,
    frame_shape: tuple[int, int],
    interact: Any,
    encoded_frame: Any,
) -> tuple[dict[int, Any], dict[int, dict[str, Any]]]:
    """Source-size masks per corrected slot, from a verified mask or a decoded box prompt.

    A box is decoded on the tracker's own image tokens (top-IoU candidate, logits > 0); one
    that decodes to nothing is skipped and recorded rather than installed as an absent prompt.
    The second mapping carries each slot's provenance for the observation row (`object_extras`)
    and the diagnostics (`selected_by`, `prompt_box`, decoder IoU), plus the confidence the
    row reports: 1.0 for a reviewed mask, the decoder's IoU prediction for a box.
    """
    masks: dict[int, Any] = {}
    records: dict[int, dict[str, Any]] = {}
    for correction in corrections:
        slot = int(correction["multiplex_slot"])
        selected_by = str(correction.get("selected_by") or "human")
        if _has_box_prompt(correction):
            xyxy = prompt_box_xyxy(correction, frame_shape)
            decoded, decoder_ious, candidates = decode_box_prompts(interact, encoded_frame, [xyxy])
            mask = _binary_mask(decoded[0:1], frame_shape)
            box_record = normalised_box_record(xyxy)
            if not bool(mask.any()):
                records[slot] = {
                    "confidence": 0.0,
                    "object_extras": {},
                    "diagnostics": {
                        "selected_by": selected_by,
                        "prompt_box": box_record,
                        "prompt_decoder_iou": float(decoder_ious[0]),
                        "correction_skipped": "decoded_empty",
                    },
                }
                continue
            masks[slot] = mask
            records[slot] = {
                "confidence": max(0.0, min(1.0, float(decoder_ious[0]))),
                "object_extras": {"prompt_box": box_record, "source": BOX_STREAM_SOURCE},
                "diagnostics": {
                    "selected_by": selected_by,
                    "prompt_box": box_record,
                    "prompt_decoder_iou": float(decoder_ious[0]),
                    "decoder_candidate_index": int(candidates[0]),
                },
            }
        else:
            masks[slot] = _read_verified_mask(
                str(correction["mask_path"]), str(correction["mask_sha256"]), frame_shape
            )
            records[slot] = {
                "confidence": 1.0,
                "object_extras": {},
                "diagnostics": (
                    {"selected_by": selected_by} if selected_by not in ("human", "agent") else {}
                ),
            }
        if correction.get("seed_start"):
            records[slot]["diagnostics"]["seed_start"] = True
    return masks, records


def _mark_unseeded(
    memory_scores: Any,
    unseeded: list[bool],
    policy_diagnostics: list[dict[str, Any]],
    slot_count: int,
) -> tuple[Any, list[dict[str, Any]]]:
    """Hand the memory encoder score -1 for every slot that has not started yet.

    The diagnostics say so per slot (`memory_written` False, reason `unseeded`), created if
    the default policy had not produced any.
    """
    import torch

    gated = memory_scores.clone()
    flat = gated.reshape(-1)
    diagnostics = list(policy_diagnostics) or [
        {"memory_written": True, "memory_gate_reason": "ok"} for _ in range(slot_count)
    ]
    for position, flag in enumerate(unseeded):
        if flag:
            flat[position] = torch.tensor(GATED_OBJECT_SCORE, dtype=flat.dtype, device=flat.device)
            diagnostics[position] = {
                **diagnostics[position],
                "memory_written": False,
                "memory_gate_reason": "unseeded",
            }
    return gated, diagnostics


def memory_policy_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Read the tracker memory policy as one plain record, so it can be logged and hashed."""
    band = getattr(args, "gate_area_band", GATE_AREA_BAND)
    policy = {
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
    # The tau hook joins the record only when set, so a policy record written before the
    # flag existed compares equal to today's default and old checkpoints keep their identity.
    tau = getattr(args, "memory_write_min_score", MEMORY_WRITE_MIN_SCORE)
    if tau is not None:
        policy["memory_write_min_score"] = float(tau)
    return policy


def memory_policy_is_default(policy: dict[str, Any]) -> bool:
    return (
        policy["slot_exclusivity"] == "off"
        and policy["memory_gate"] == "off"
        and policy.get("memory_write_min_score") is None
    )


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

    The tau hook (``memory_write_min_score``) is judged first and independently of the gate
    mode: with the Sep 18 gate off it is the only test, so a FineBio arm can skip memory
    writes on score alone without the IoU, contest and area-band tests that starved slots in
    the ablation.
    """
    slot_count = int(scores_m.reshape(-1).shape[0])
    history = [list(entries) for entries in area_history]
    while len(history) < slot_count:
        history.append([])
    tau = policy.get("memory_write_min_score")
    gate_on = policy["memory_gate"] != "off"
    if not gate_on and tau is None:
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
        if not memory_write_allowed(raw_scores[slot], tau):
            reason = "low_object_score"
        elif not gate_on:
            reason = "ok"
        elif raw_scores[slot] <= policy["gate_min_object_score"]:
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
        if trusted and gate_on:
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
    memory_settings = memory_settings_from_args(args)
    effective_memory_semantics = correction_memory_semantics(
        prompt_memory_semantics=memory_settings["prompt_memory_semantics"],
        keep_frame_memory_at_correction=memory_settings["keep_frame_memory_at_correction"],
    )
    runtime_settings = _runtime_settings(args, concepts)
    start = perf_counter()
    gpu_processes = _gpu_processes()
    guard_refusal, guard_record = gpu_guard_decision(args, gpu_processes)
    if guard_record is not None:
        runtime_settings["gpu_guard"] = guard_record

    if not video_path.is_file():
        fs_common.write_json(
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
            sort_keys=True,
        )
        return 2
    if not model_path.is_file():
        fs_common.write_json(
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
            sort_keys=True,
        )
        return 2
    if guard_refusal is not None:
        fs_common.write_json(
            result_path,
            _result(
                state="blocked",
                reason=guard_refusal,
                frames_processed=0,
                elapsed_seconds=perf_counter() - start,
                ttfu_seconds=None,
                gpu_peak_vram_bytes=None,
                masks_written=0,
                gpu_processes=gpu_processes,
                unavailable=["time_to_first_usable_output_seconds", "gpu_peak_vram_bytes"],
                runtime_settings=runtime_settings,
            ),
            sort_keys=True,
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
        capture = _open_capture(video_path, int(getattr(args, "start_frame", 0) or 0))
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
        # The interactive context shares every module with the tracker (no extra VRAM); its
        # decoder reads the same image tokens the tracker encodes, so a box prompt at any
        # frame costs one decoder call and no second image encode.
        interact = core.get_interactive_context()

        initial_masks: list[TrackedMask] = []
        initial_memory: Any = None
        correction_schedule: dict[int, list[dict[str, Any]]] = {}
        slot_starts: dict[int, int] = {}
        resumed: dict[str, Any] | None = None
        if args.resume_from_checkpoint:
            resumed = torch.load(
                Path(args.resume_from_checkpoint), map_location="cuda:0", weights_only=False
            )
            if resumed.get("format") != CHECKPOINT_FORMAT:
                raise ValueError(
                    f"checkpoint format {resumed.get('format')!r} is not {CHECKPOINT_FORMAT!r}"
                )
            # Memory settings are compared first so a mismatch names the setting instead
            # of failing as an anonymous identity difference.
            check_checkpoint_memory_settings(resumed, memory_settings)
            if resumed["stream_identity"] != stream_identity(args, concepts):
                raise ValueError(
                    "checkpoint was produced by a different stream: the video, model, encoder "
                    "settings, preprocessing, or initialization does not match this run"
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
                _corrections_by_frame(
                    multi_keyframe_schedule,
                    concepts,
                    memory_semantics=effective_memory_semantics,
                )
                if multi_keyframe_schedule is not None
                else {}
            )
            slot_starts = (
                slot_start_frames(multi_keyframe_schedule)
                if multi_keyframe_schedule is not None
                else {}
            )
            tracking_encoded = tracking.encode_image(first_model_frame, MAX_SIDE_LENGTH, True)
            source_masks = []
            for slot, seed in enumerate(seed_records):
                if slot in slot_starts:
                    # Allocated now, seeded at its start frame through the correction path.
                    binary_mask = np.zeros(first_frame.shape[:2], dtype=bool)
                    confidence = 0.0
                elif _has_box_prompt(seed):
                    decoded, decoder_ious, _ = decode_box_prompts(
                        interact, tracking_encoded, [prompt_box_xyxy(seed, first_frame.shape[:2])]
                    )
                    binary_mask = _binary_mask(decoded[0:1], first_frame.shape[:2])
                    if not bool(binary_mask.any()):
                        raise ValueError(f"box seed for slot {slot} decoded to an empty mask")
                    confidence = float(decoder_ious[0])
                else:
                    binary_mask = _read_verified_mask(
                        str(seed["mask_path"]), str(seed["mask_sha256"]), first_frame.shape[:2]
                    )
                    confidence = 1.0
                source_masks.append(binary_mask)
                initial_masks.append((slot, concepts[slot], binary_mask, confidence, None, None))
            initial_memory = tracking.encode_prompt_memory_from_mask(
                tracking_encoded, _prompt_mask_batch(np.stack(source_masks, axis=0))
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
        runtime_settings["stream_identity"] = identity
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
                    "max_prompt_memory": memory_settings["max_prompt_memory"],
                    "max_frame_memory": memory_settings["max_frame_memory"],
                    "prompt_memory_semantics": memory_settings["prompt_memory_semantics"],
                    "keep_frame_memory_at_correction": memory_settings[
                        "keep_frame_memory_at_correction"
                    ],
                    "is_recent_first": memory_settings["is_recent_first"],
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
                prompt_memories, frame_memories = build_memory_banks(
                    prompt_memories=(
                        [initial_memory] if resumed is None else resumed["prompt_memories"]
                    ),
                    frame_memories=[] if resumed is None else resumed["frame_memories"],
                    max_prompt_memory=memory_settings["max_prompt_memory"],
                    max_frame_memory=memory_settings["max_frame_memory"],
                )
                if resumed is not None:
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
                        is_recent_first=memory_settings["is_recent_first"],
                        num_multiplex_objects=len(initial_masks),
                    )
                    active = scores > 0
                    correction_masks, correction_records = _correction_masks_for_frame(
                        correction_schedule.get(frame_index, []),
                        frame_shape=frame.shape[:2],
                        interact=interact,
                        encoded_frame=encoded,
                    )
                    # Slots that start later are forced absent until their start frame: no
                    # object row, and the memory encoder sees them as absent (score -1).
                    unseeded = [
                        frame_index < slot_starts.get(index, 0) and position not in correction_masks
                        for position, (index, *_) in enumerate(initial_masks)
                    ]
                    if any(unseeded):
                        active = active.clone()
                        for position, flag in enumerate(unseeded):
                            if flag:
                                active.reshape(-1)[position] = False
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
                    if any(unseeded):
                        memory_scores, policy_diagnostics = _mark_unseeded(
                            memory_scores, unseeded, policy_diagnostics, len(initial_masks)
                        )
                    if correction_masks:
                        source_masks = _replace_prompt_memory_for_correction(
                            predicted_source_masks=_source_binary_masks(masks, frame.shape[:2]),
                            correction_masks_by_slot=correction_masks,
                            prompt_memories=prompt_memories,
                            frame_memories=frame_memories,
                            encoded_frame=encoded,
                            encode_prompt_memory_from_mask=tracking.encode_prompt_memory_from_mask,
                            prompt_memory_semantics=memory_settings["prompt_memory_semantics"],
                            keep_frame_memory_at_correction=memory_settings[
                                "keep_frame_memory_at_correction"
                            ],
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
                            (
                                correction_records[position]["confidence"]
                                if position in correction_masks
                                else float(scores[position])
                            ),
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
                            **(correction_records.get(position, {}).get("diagnostics", {})),
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
                        extras={
                            initial_masks[position][0]: record["object_extras"]
                            for position, record in correction_records.items()
                            if record["object_extras"]
                        },
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
        fs_common.write_json(
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
            sort_keys=True,
        )
        return 0
    except Exception as error:
        if "condition_writer" in locals() and condition_writer is not None:
            condition_writer.release()
        fs_common.write_json(
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
            sort_keys=True,
        )
        traceback.print_exc()
        return 1


def _box_stream_runtime_settings(
    args: argparse.Namespace,
    concepts: tuple[str, ...],
    *,
    box_stream_path: Path,
    box_stream_sha256: str | None,
    prompted_frames: int,
    prompted_boxes: int,
) -> dict[str, Any]:
    """The condition record of a memory-free box-decode run, beside the worker measurements."""
    settings: dict[str, Any] = {
        "device": "cuda:0",
        "dtype": "bfloat16",
        "mode": "box_stream",
        "max_frames": MAX_FRAMES,
        "max_side_length": MAX_SIDE_LENGTH,
        "use_square_sizing": True,
        "mask_period_frames": MASK_PERIOD_FRAMES,
        "box_derivation": (
            "union of the mask's 8-connected components with area at least "
            f"{BOX_COMPONENT_KEEP_FRACTION:g} of the largest; smaller components are ignored"
        ),
        "box_component_keep_fraction": BOX_COMPONENT_KEEP_FRACTION,
        "concepts": list(concepts),
        "prompt_mode": args.prompt_mode,
        "preprocessing": args.preprocessing,
        "analysis_fps": ANALYSIS_FPS,
        "box_stream": str(box_stream_path),
        "box_stream_sha256": box_stream_sha256,
        "box_stream_prompted_frames": prompted_frames,
        "box_stream_prompted_boxes": prompted_boxes,
        "other_slots_as_negatives": bool(args.other_slots_as_negatives),
        "decode_api": (
            "interactive encode_image once per prompted frame; encode_prompts(BxNx2x2 boxes"
            + (
                ", other boxes' centres as background points"
                if args.other_slots_as_negatives
                else ""
            )
            + ") + generate_masks in one batched decoder call; top-IoU candidate per box; "
            "logits > 0 bilinearly resized to source pixels"
        ),
        "video_memory": "none; every frame is decoded from its own boxes and identity is the slot",
        "chunking": "none",
        "intentional_id_resets": False,
        "object_score_semantics": (
            "object_score and iou_prediction are the image decoder's IoU prediction for the "
            "chosen candidate (0..1); confidence is the same value. There is no tracker "
            "presence logit in this mode and nothing is measured against ground truth."
        ),
        "row_source": BOX_STREAM_SOURCE,
        "checkpoints": "none; box_stream mode keeps no tracker state",
        "gpu_guard_mode": getattr(args, "gpu_guard", gpu_guard.DEFAULT_GUARD_MODE),
    }
    if getattr(args, "start_frame", 0):
        settings["start_frame"] = int(args.start_frame)
    if args.preprocessing == "gray_p01_p99_clahe":
        settings.update(
            {
                "lower_percentile": args.lower_percentile,
                "upper_percentile": args.upper_percentile,
                "clahe_clip_limit": args.clahe_clip_limit,
                "clahe_tile_grid_size": args.clahe_tile_grid_size,
            }
        )
    return settings


def _write_blocked(
    result_path: Path,
    *,
    reason: str,
    start: float,
    gpu_processes: list[dict[str, str]],
    runtime_settings: dict[str, Any],
) -> int:
    fs_common.write_json(
        result_path,
        _result(
            state="blocked",
            reason=reason,
            frames_processed=0,
            elapsed_seconds=perf_counter() - start,
            ttfu_seconds=None,
            gpu_peak_vram_bytes=None,
            masks_written=0,
            gpu_processes=gpu_processes,
            unavailable=["time_to_first_usable_output_seconds", "gpu_peak_vram_bytes"],
            runtime_settings=runtime_settings,
        ),
        sort_keys=True,
    )
    return 2


def _timing_summary(values_ms: list[float]) -> dict[str, float | int | None]:
    if not values_ms:
        return {"n": 0, "median": None, "mean": None, "p90": None, "max": None}
    ordered = sorted(values_ms)
    return {
        "n": len(ordered),
        "median": ordered[len(ordered) // 2],
        "mean": sum(ordered) / len(ordered),
        "p90": ordered[min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))],
        "max": ordered[-1],
    }


def run_box_stream(args: argparse.Namespace) -> int:
    """Memory-free per-frame box decode: boxes in, masks out, no tracker state.

    Every frame that has boxes is encoded once with the interactive context and every box is
    decoded from that encoding in one batched decoder call; a frame without boxes writes an
    empty row.  Output is the same `observations.jsonl` + mask PNG layout as the tracker,
    with `prompt_box` and `source: sam3_decode` on every object row.
    """
    run_directory = Path(args.run_directory)
    output_path = run_directory / "observations.jsonl"
    result_path = run_directory / "worker_result.json"
    masks_directory = run_directory / "masks"
    video_path = Path(args.video)
    model_path = Path(args.model)
    box_stream_path = Path(args.box_stream)
    start = perf_counter()
    gpu_processes = _gpu_processes()
    guard_refusal, guard_record = gpu_guard_decision(args, gpu_processes)

    stream_frames: dict[int, list[dict[str, Any]]] = {}
    concepts: tuple[str, ...] = ()
    box_stream_sha256: str | None = None
    if box_stream_path.is_file():
        box_stream_sha256 = fs_common.sha256_file(box_stream_path)
        stream_frames, stream_labels = parse_box_stream(box_stream_path.read_text())
        requested = tuple(json.loads(args.concepts_json)) if args.concepts_json else ()
        if requested and requested != CONCEPTS and requested != stream_labels:
            raise ValueError(
                f"--concepts-json {list(requested)} does not match the box stream's slot "
                f"labels {list(stream_labels)}"
            )
        concepts = stream_labels
    prompted = {frame: boxes for frame, boxes in stream_frames.items() if boxes}
    runtime_settings = _box_stream_runtime_settings(
        args,
        concepts,
        box_stream_path=box_stream_path,
        box_stream_sha256=box_stream_sha256,
        prompted_frames=len([frame for frame in prompted if frame < MAX_FRAMES]),
        prompted_boxes=sum(len(boxes) for frame, boxes in prompted.items() if frame < MAX_FRAMES),
    )
    if guard_record is not None:
        runtime_settings["gpu_guard"] = guard_record
    if args.checkpoint_every or args.checkpoint_at:
        print(
            "box_stream mode keeps no tracker state; --checkpoint-every/--checkpoint-at are "
            "ignored (resume by re-running from --start-frame)",
            flush=True,
        )
        runtime_settings["ignored_checkpoint_flags"] = {
            "checkpoint_every": args.checkpoint_every,
            "checkpoint_at": list(args.checkpoint_at),
        }

    blocked = None
    if not video_path.is_file():
        blocked = f"approved proxy does not exist: {video_path}"
    elif not model_path.is_file():
        blocked = f"SAM3 checkpoint is unavailable at configured path: {model_path}"
    elif not box_stream_path.is_file():
        blocked = f"box stream does not exist: {box_stream_path}"
    elif guard_refusal is not None:
        blocked = guard_refusal
    if blocked is not None:
        return _write_blocked(
            result_path,
            reason=blocked,
            start=start,
            gpu_processes=gpu_processes,
            runtime_settings=runtime_settings,
        )

    runtime_settings["stream_identity"] = stream_identity(args, concepts)
    try:
        import torch
        from muggled_sam.make_sam import make_sam_from_state_dict

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; smoke policy does not permit CPU inference")
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        capture = _open_capture(video_path, int(args.start_frame))
        core = make_sam_from_state_dict(model_path)
        core.to(device="cuda:0", dtype=torch.bfloat16)
        interact = core.get_interactive_context()
        masks_directory.mkdir(exist_ok=True)
        processed = 0
        masks_written = 0
        first_usable: float | None = None
        encode_ms: list[float] = []
        decode_ms: list[float] = []
        frame_ms: list[float] = []
        box_ms: list[float] = []
        decoder_ious: list[float] = []
        with output_path.open("w") as observations:
            for frame_index in range(MAX_FRAMES):
                ok, frame = capture.read()
                if not ok:
                    break
                model_frame = _prepare_frame(frame, args)
                boxes = stream_frames.get(frame_index, [])
                frame_shape = frame.shape[:2]
                tracked: list[TrackedMask] = []
                extras: dict[int, dict[str, Any]] = {}
                diagnostics: list[dict[str, Any]] = []
                if boxes:
                    step_start = perf_counter()
                    encoded = interact.encode_image(model_frame, MAX_SIDE_LENGTH, True)
                    torch.cuda.synchronize(0)
                    encoded_at = perf_counter()
                    xyxy = [prompt_box_xyxy(box, frame_shape) for box in boxes]
                    logits, ious, candidates = decode_box_prompts(
                        interact,
                        encoded,
                        xyxy,
                        other_boxes_as_negatives=bool(args.other_slots_as_negatives),
                    )
                    torch.cuda.synchronize(0)
                    decoded_at = perf_counter()
                    encode_ms.append((encoded_at - step_start) * 1000.0)
                    decode_ms.append((decoded_at - encoded_at) * 1000.0)
                    frame_ms.append((decoded_at - step_start) * 1000.0)
                    box_ms.append((decoded_at - encoded_at) * 1000.0 / len(boxes))
                    for position, box in enumerate(boxes):
                        slot = int(box["slot"])
                        iou = float(ious[position])
                        decoder_ious.append(iou)
                        box_record = normalised_box_record(xyxy[position])
                        tracked.append(
                            (slot, box["label"], logits[position : position + 1], iou, iou, iou)
                        )
                        extras[slot] = {
                            "prompt_box": box_record,
                            "source": BOX_STREAM_SOURCE,
                            "prompt_source": box["source"],
                            "prompt_score": box["score"],
                        }
                        diagnostics.append(
                            {
                                "schema_version": "1.0",
                                "object_id": f"sam3-{slot:02d}",
                                "label": box["label"],
                                "multiplex_slot": slot,
                                "object_score": iou,
                                "iou_prediction": iou,
                                "active": bool((logits[position] > 0).any()),
                                "corrected": False,
                                "prompt_box": box_record,
                                "prompt_source": box["source"],
                                "prompt_score": box["score"],
                                "prompt_decoder_iou": iou,
                                "decoder_candidate_index": int(candidates[position]),
                            }
                        )
                observation, written = _observation(
                    view_id=args.view_id,
                    frame_index=frame_index,
                    source_offset_seconds=args.source_offset_seconds,
                    masks=tracked,
                    frame_shape=frame_shape,
                    masks_directory=masks_directory,
                    run_directory=run_directory,
                    diagnostics=diagnostics,
                    extras=extras,
                )
                observations.write(json.dumps(observation, sort_keys=True) + "\n")
                processed += 1
                masks_written += written
                if first_usable is None and observation["objects"]:
                    first_usable = perf_counter() - start
        capture.release()
        torch.cuda.synchronize(0)
        runtime_settings["box_decode_timing_ms"] = {
            "per_prompted_frame": _timing_summary(frame_ms),
            "image_encode": _timing_summary(encode_ms),
            "decode_all_boxes": _timing_summary(decode_ms),
            "per_box": _timing_summary(box_ms),
            "semantics": (
                "wall-clock per prompted frame after cuda.synchronize: one image encode plus "
                "one batched decoder call over the frame's boxes; per_box divides the decode "
                "time by the box count. Frames without boxes are not timed."
            ),
        }
        runtime_settings["decoder_iou_prediction"] = _timing_summary(decoder_ious)
        fs_common.write_json(
            result_path,
            _result(
                state="succeeded",
                reason=None,
                frames_processed=processed,
                elapsed_seconds=perf_counter() - start,
                ttfu_seconds=first_usable,
                gpu_peak_vram_bytes=int(torch.cuda.max_memory_allocated(0)),
                masks_written=masks_written,
                gpu_processes=gpu_processes,
                unavailable=[]
                if first_usable is not None
                else ["time_to_first_usable_output_seconds"],
                runtime_settings=runtime_settings,
            ),
            sort_keys=True,
        )
        return 0
    except Exception as error:
        fs_common.write_json(
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
            sort_keys=True,
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
    # Likewise for the memory-bank flags: only a non-default condition joins the identity.
    memory = memory_settings_from_args(args)
    if not memory_settings_are_default(memory):
        payload["memory_settings"] = {
            key: value for key, value in memory.items() if key != "max_frame_memory"
        }
    # FineBio additions, each present only when used so earlier identities are unchanged.
    start_frame = int(getattr(args, "start_frame", 0) or 0)
    if start_frame:
        payload["start_frame"] = start_frame
    box_stream = getattr(args, "box_stream", None)
    if box_stream:
        stream_path = Path(box_stream)
        payload["box_stream_sha256"] = (
            fs_common.sha256_file(stream_path) if stream_path.is_file() else None
        )
        if getattr(args, "other_slots_as_negatives", False):
            payload["other_slots_as_negatives"] = True
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


def build_parser() -> argparse.ArgumentParser:
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
            "box_stream",
        ),
        default="text_detection",
    )
    parser.add_argument("--manual-box-json")
    parser.add_argument("--manual-seeds-json")
    parser.add_argument("--multi-keyframe-schedule-json")
    parser.add_argument("--condition-input-video")
    box_group = parser.add_argument_group(
        "box stream (memory-free per-frame decode)",
        "`--prompt-mode box_stream`: identity comes from outside; every frame's boxes are "
        "decoded by the image decoder with no video memory.",
    )
    box_group.add_argument(
        "--box-stream",
        help=(
            "JSONL of per-frame box prompts: {frame_index, boxes: [{slot, label, box_xyxy_px "
            "| box_xyxy_norm, score, source}]}; its SHA-256 joins the stream identity."
        ),
    )
    box_group.add_argument(
        "--other-slots-as-negatives",
        action="store_true",
        help="Give every box prompt the other boxes' centres as background points.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Source frame that becomes analysis frame 0; earlier frames are decoded and dropped.",
    )
    parser.add_argument(
        "--memory-write-min-score",
        type=float,
        default=MEMORY_WRITE_MIN_SCORE,
        metavar="TAU",
        help=(
            "Video-memory mode: a slot whose raw object score is below TAU on a frame is "
            "memorised as absent for that frame (its mask is still reported). Unset writes "
            "every present slot, as before. Joins the memory policy record and the stream "
            "identity."
        ),
    )
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
    memory_group = parser.add_argument_group(
        "correction memory semantics",
        "Opt-in memory-bank condition; the defaults reproduce the earlier runs byte for byte.",
    )
    memory_group.add_argument(
        "--prompt-memory-semantics",
        choices=PROMPT_MEMORY_SEMANTICS,
        default=DEFAULT_PROMPT_MEMORY_SEMANTICS,
        help=(
            "replace: a correction becomes the only prompt memory (today). append: the "
            "frame-0 seed and every correction stay in a bounded prompt bank."
        ),
    )
    memory_group.add_argument(
        "--max-prompt-memory",
        type=int,
        default=None,
        help=(
            "Prompt-memory entries; defaults to 1 for replace and "
            f"{APPEND_PROMPT_MEMORY_DEFAULT_ENTRIES} for append. Replace requires exactly 1."
        ),
    )
    memory_group.add_argument(
        "--keep-frame-memory-at-correction",
        action="store_true",
        help="Do not clear the frame-memory bank when a correction is applied.",
    )
    memory_group.add_argument(
        "--recent-first",
        action="store_true",
        help=(
            "Tell the memory fusion model that index 0 of the frame bank is the most recent "
            "entry. The worker stores oldest-first, so this reverses the temporal position "
            "encoding; recorded as is_recent_first."
        ),
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
    parser.add_argument(
        "--allow-gpu-neighbour",
        type=int,
        action="append",
        default=[],
        metavar="PID",
        help=(
            "PID of a GPU model process the guard may tolerate (repeatable); recorded under "
            "runtime_settings.gpu_guard. Any other model process still blocks the run."
        ),
    )
    parser.add_argument(
        "--gpu-guard",
        choices=gpu_guard.GUARD_MODES,
        default=gpu_guard.DEFAULT_GUARD_MODE,
        help=(
            "strict: refuse beside any model-like process by nvidia-smi name (Sep 21 logic); "
            "vram: classify neighbours by /proc cmdline, refuse another Battle worker, an "
            "unknown neighbour above 2 GiB, or headroom below 1.5 x the expected peak."
        ),
    )
    parser.add_argument(
        "--gpu-guard-profile",
        choices=tuple(gpu_guard.EXPECTED_PEAK_BYTES),
        default=None,
        help=(
            "Expected-peak profile for the vram guard; default from --max-side-length "
            "(sam3_1280 up to 1280, sam3_1080p up to 1920, unknown above)."
        ),
    )
    parser.add_argument(
        "--expected-peak-vram-bytes",
        type=int,
        default=None,
        help="Override the profile's expected peak VRAM (bytes) for the vram guard.",
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
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and cross-validate the worker command line without starting a run."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.gate_area_history_frames < 1:
        parser.error("--gate-area-history-frames must be at least 1")
    if args.mask_period_frames < 1:
        parser.error("--mask-period-frames must be at least 1")
    if args.checkpoint_every < 0:
        parser.error("--checkpoint-every must not be negative")
    if args.max_frame_memory < 1:
        parser.error("--max-frame-memory must be at least 1")
    if args.start_frame < 0:
        parser.error("--start-frame must not be negative")
    if (args.prompt_mode == "box_stream") != (args.box_stream is not None):
        parser.error("--prompt-mode box_stream and --box-stream go together")
    if args.prompt_mode == "box_stream" and args.resume_from_checkpoint:
        parser.error("box_stream mode keeps no tracker state, so there is nothing to resume")
    try:
        memory_settings_from_args(args)
    except ValueError as error:
        parser.error(str(error))
    return args


def main() -> None:
    global MAX_FRAMES, MAX_SIDE_LENGTH, MAX_FRAME_MEMORY, ANALYSIS_FPS, MASK_PERIOD_FRAMES

    args = parse_args()
    MAX_FRAMES = args.max_frames
    MAX_SIDE_LENGTH = args.max_side_length
    MAX_FRAME_MEMORY = args.max_frame_memory
    ANALYSIS_FPS = args.analysis_fps
    MASK_PERIOD_FRAMES = args.mask_period_frames
    if args.prompt_mode == "box_stream":
        raise SystemExit(run_box_stream(args))
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
