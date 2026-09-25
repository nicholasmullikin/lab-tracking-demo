"""`battle-finebio-anchors`: human gate 2 of the FineBio 3D-tracking plan (`p6-anchors`).

The human labels later; this module owns the frame choice, the labelling workspace and the
scoreboard, so that scoring runs whenever labels exist and re-runs as more land::

    battle-finebio-anchors select --clip <clip config> --arm-b <arm (b) dir> \\
        --seeds <seeds.json> --fixed-view auto --output configs/qa/<trial>_review_anchors.json
    battle-finebio-anchors workspace --config <anchor config> --arm-b <arm (b) dir> \\
        --output runs/finebio-anchors-<trial>-<date>/                  # GPU, a few minutes
    battle-finebio-anchors score --workspace <workspace> --record <decisions.json> \\
        --arms a=<dir>,b=<dir>,c=<dir>,d=<dir> --output <workspace>/scoreboard
    battle-finebio-anchors export --workspace <workspace> --record <decisions.json> \\
        --output docs/qa/<trial>-review-anchors.human-record.json

* ``select``: the six-view annotated frame in all six views; N frames chosen by
  detector-vs-mask **disagreement** in arm (b) (the lowest ``provenance.detector_box_iou`` per
  frame pooled over the non-group slots of the fpv and one fixed view, lowest first, with a
  minimum spacing between chosen frames and a minimum count inside the centrifuge cycles'
  neighbourhoods); M **random** frames with a fixed seed. The fixed view is the one with the
  most individually tracked (non-group) slots whose plate slot has a mask on most frames.
  Writes the typed anchor config (frames raw + proxy, views, slots per view with labels, a
  reason per frame, the seed and every rule parameter).
* ``workspace``: for every (frame, view, slot) the SAM3 image-decoder **candidates** the human
  chooses among rather than draws: candidate 0 is arm (b)'s mask (the tight detector box),
  the alternates are the 0.15-margin box, the box plus a positive point, and the tight box's
  second-ranked decoder output; decoded on the GPU under the MuggledSAM interpreter the way
  `battle-detector-seed decode` does (same guard, encoder side 1280, one image encode per
  frame and view). Static per-frame candidate sheets (rows = slots, columns = context and
  the candidates), a full-frame overview with slot numbers, ``workspace.json`` and
  ``decisions.template.json`` with a ``decision`` (candidate index, ``"box"``, ``"hidden"``,
  ``"none_fits"`` or null) and an ``instance_identity`` per cell. The Assembly101 calibration
  web workspace was not generalised: it is bound to the Assembly101 clip configs and the
  four-part target policy, so gate 2 follows the gate-1 pattern (sheets + a decisions JSON).
* ``score``: per arm and cell, mask IoU against the accepted candidate (box IoU against the
  accepted candidate's bbox or the detector box for box-level accepts, and for the boxes-only
  arm), false positives on hidden cells, missing masks; identity against the human's
  ``instance_identity`` (cross-camera IDF1 on the six-view frame and pooled over every
  labelled cell, identities split across tracks, tracks merging identities). Runs on an empty
  or partial record and says how many cells are labelled.
* ``export``: the committed human-record skeleton under ``docs/qa/`` (states, SHA-256 of the
  accepted candidate masks, identities; no mask leaves ``runs/``).

Claim boundary: anchors are one person's choice among decoder masks on a handful of frames of
one trial, not ground truth; they rank arms against each other. FineBio is non-commercial
research data: frames, masks and sheets stay under ``runs/`` and are never committed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from . import fs_common, gpu_guard
    from .finebio_detect import box_iou
except ImportError:
    # `decode-worker` runs under the MuggledSAM interpreter, where the battle package is not
    # installed: the stdlib-only helpers beside this file are imported as top-level modules.
    _HERE = str(Path(__file__).resolve().parent)
    if _HERE not in sys.path:
        sys.path.append(_HERE)
    import fs_common  # type: ignore[no-redef]
    import gpu_guard  # type: ignore[no-redef]
    from finebio_detect import box_iou  # type: ignore[no-redef]

SCHEMA = "battle-finebio-anchors/1"
CONFIG_KIND = "finebio_review_anchor_config"
ANCHOR_KIND = "human_review_anchor"
VIEWS: tuple[str, ...] = ("fpv", "T1", "T2", "T3", "T4", "T5")
FPV_VIEW = "fpv"
GROUP_SUFFIX = "_group"
PLATE_CLASS = "cell_culture_plate"
SAM3_SOURCES = ("sam3_decode", "sam3_video")
# Bench objects the protocol holds once: their identity is the class and is pre-filled in the
# decisions template; pipettes are not (the detector confuses their colours) and tubes are not.
SINGLETON_IDENTITY_CLASSES: tuple[str, ...] = (
    "centrifuge",
    "vortex_mixer",
    "pcr_machine",
    PLATE_CLASS,
    "trash_can",
)
DECISION_WORDS = ("box", "hidden", "none_fits")
CANDIDATE_KINDS = ("arm_b_tight", "margin", "box_point", "tight_rank2")
DEFAULT_SPACING = 90
DEFAULT_DISAGREEMENT_FRAMES = 12
DEFAULT_RANDOM_FRAMES = 5
DEFAULT_MIN_IN_CYCLES = 3
DEFAULT_CYCLE_PAD = 90
DEFAULT_SEED = 20260925
DEFAULT_MARGIN = 0.15
DEFAULT_PLATE_VISIBLE_FRACTION = 0.9
DUPLICATE_IOU = 0.97
MATCH_IOU_DETECTOR = 0.1
MUGGLED_SAM_SOURCE = Path("/home/nick/src/muggled_sam")
MUGGLED_SAM_PYTHON = Path("/home/nick/.pyenv/versions/muggled_sam/bin/python")
DEFAULT_MODEL = MUGGLED_SAM_SOURCE / "model_weights" / "sam3.1_multiplex.pt"
ENCODER_SIDE = 1280
TILE = 256
CLAIM_BOUNDARY = (
    "Anchors are one person's choice among SAM3 image-decoder candidate masks (or a hidden or "
    "box-level mark) on a handful of frames of one trial: the mask boundary is the decoder's, "
    "the choice is the human's. They are review evidence for ranking tracking arms against "
    "each other, not a dataset, not ground truth, and support no accuracy claim. The FineBio "
    "detector that proposed every box was trained on this lab's objects and cameras."
)
LICENCE_NOTE = (
    "FineBio is licensed for non-commercial research; frames, masks and sheets derived from "
    "it stay under runs/ and are never committed or redistributed. The anchor config (frame "
    "numbers, slot labels) and the human record (states, hashes, identities) carry no pixels."
)

Box = tuple[float, float, float, float]


# --------------------------------------------------------------------------------------------
# inputs


def load_clip_doc(path: Path) -> dict[str, Any]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("config_kind") != "finebio_clip_config":
        raise ValueError(f"{path} is not a finebio_clip_config")
    return doc


def load_seed_slots(seeds_path: Path) -> dict[str, list[dict[str, Any]]]:
    """view -> slots (slot, label, class, role, rule, start_frame) from a seeds.json."""
    doc = json.loads(Path(seeds_path).read_text(encoding="utf-8"))
    out: dict[str, list[dict[str, Any]]] = {}
    for view, selected in doc["views"].items():
        out[view] = [
            {
                "slot": int(s["slot"]),
                "label": str(s["label"]),
                "class": str(s["class"]),
                "role": s.get("role"),
                "rule": s.get("rule"),
                "start_frame": int(s.get("start_frame", doc["window"]["start"])),
                "status": s.get("status"),
            }
            for s in sorted(selected["slots"], key=lambda s: int(s["slot"]))
        ]
    return out


def read_rows(
    observations: Path,
    *,
    frames: Iterable[int] | None = None,
    views: Iterable[str] | None = None,
    sam3_only: bool = False,
) -> list[dict[str, Any]]:
    """`FineBioObservation` rows as dicts, pre-filtered on the raw text so an 800k-row arm
    file is read in a second or two."""
    wanted_frames = None if frames is None else {int(f) for f in frames}
    wanted_views = None if views is None else {str(v) for v in views}
    frame_keys = None if wanted_frames is None else {f'"frame_index":{f},' for f in wanted_frames}
    view_keys = None if wanted_views is None else {f'"view":"{v}"' for v in wanted_views}
    rows = []
    with Path(observations).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            # The text pre-filter only applies to compactly written rows (the arms' files);
            # anything else is parsed and filtered on the values.
            compact = '"frame_index":' in line and '"view":' in line
            if compact:
                if frame_keys is not None and not any(k in line for k in frame_keys):
                    continue
                if view_keys is not None and not any(k in line for k in view_keys):
                    continue
                if sam3_only and '"source":"sam3' not in line:
                    continue
            row = json.loads(line)
            if wanted_frames is not None and int(row["frame_index"]) not in wanted_frames:
                continue
            if wanted_views is not None and str(row["view"]) not in wanted_views:
                continue
            if sam3_only and row.get("source") not in SAM3_SOURCES:
                continue
            rows.append(row)
    return rows


def arm_worker_runs(arm_dir: Path, repository_root: Path) -> dict[str, Path]:
    """view -> SAM3 worker run directory (masks/), from the arm's measures.json; empty for the
    boxes-only arm."""
    measures = Path(arm_dir) / "measures.json"
    if not measures.is_file():
        return {}
    doc = json.loads(measures.read_text(encoding="utf-8"))
    runs = (doc.get("observations") or {}).get("worker_runs") or {}
    out = {}
    for view, uri in runs.items():
        path = Path(uri)
        out[view] = path if path.is_absolute() else (repository_root / path)
    return out


def worker_mask_path(worker_run: Path, row: dict[str, Any], frame_offset: int) -> Path | None:
    """`masks/<analysis frame:06d>_<slot index:02d>.png` of a SAM3 row (`object_id` sam3-NN)."""
    object_id = str((row.get("provenance") or {}).get("object_id") or "")
    tail = object_id.rsplit("-", 1)[-1]
    if not tail.isdigit():
        return None
    analysis = int(row["frame_index"]) - frame_offset
    return Path(worker_run) / "masks" / f"{analysis:06d}_{int(tail):02d}.png"


def read_mask(path: Path) -> np.ndarray | None:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None if image is None else image > 0


def clean_mask(mask: np.ndarray | None) -> np.ndarray | None:
    """The arms' speckle rule (components >= 20% of the largest), so a candidate copied from
    arm (b) and an arm mask read at scoring time are measured the same way."""
    if mask is None or not mask.any():
        return None
    from .finebio_observations import filter_mask_components

    cleaned, _dropped, _components = filter_mask_components(mask)
    return cleaned


def mask_bbox(mask: np.ndarray) -> Box | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int(np.logical_or(a, b).sum())
    return int(np.logical_and(a, b).sum()) / union if union else 0.0


def expand_box(box: Box, margin: float, image_hw: tuple[int, int]) -> Box:
    width, height = box[2] - box[0], box[3] - box[1]
    return (
        max(0.0, box[0] - margin * width),
        max(0.0, box[1] - margin * height),
        min(float(image_hw[1]), box[2] + margin * width),
        min(float(image_hw[0]), box[3] + margin * height),
    )


def _inside(point: Sequence[float], box: Sequence[float]) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


# --------------------------------------------------------------------------------------------
# frame selection


@dataclass
class SelectionParams:
    spacing_frames: int = DEFAULT_SPACING
    disagreement_frames: int = DEFAULT_DISAGREEMENT_FRAMES
    random_frames: int = DEFAULT_RANDOM_FRAMES
    min_in_cycles: int = DEFAULT_MIN_IN_CYCLES
    cycle_pad_frames: int = DEFAULT_CYCLE_PAD
    seed: int = DEFAULT_SEED
    plate_visible_fraction: float = DEFAULT_PLATE_VISIBLE_FRACTION


def is_group(label: str) -> bool:
    return label.rsplit("#", 1)[0].endswith(GROUP_SUFFIX)


def disagreement_scores(
    rows: Iterable[dict[str, Any]], views: Sequence[str]
) -> dict[int, dict[str, Any]]:
    """frame -> the lowest `detector_box_iou` over the non-group SAM3 rows of `views`, with the
    slot that produced it and the number of rows pooled; frames need a row in every view."""
    per_frame: dict[int, dict[str, Any]] = {}
    seen_views: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        if row.get("view") not in views or row.get("source") not in SAM3_SOURCES:
            continue
        frame = int(row["frame_index"])
        seen_views[frame].add(row["view"])
        if is_group(row["slot"]):
            continue
        iou = (row.get("provenance") or {}).get("detector_box_iou")
        if iou is None:
            continue
        entry = per_frame.setdefault(frame, {"min_iou": 2.0, "slot": None, "rows": 0})
        entry["rows"] += 1
        if float(iou) < entry["min_iou"]:
            entry["min_iou"] = round(float(iou), 4)
            entry["slot"] = f"{row['view']}/{row['slot']}"
    return {
        f: e for f, e in per_frame.items() if seen_views[f] >= set(views) and e["slot"] is not None
    }


def neighbourhoods(
    cycles: Sequence[Sequence[int]], pad: int, window: tuple[int, int]
) -> list[tuple[int, int]]:
    return [(max(window[0], int(lo) - pad), min(window[1], int(hi) + pad)) for lo, hi in cycles]


def _in_any(frame: int, intervals: Sequence[tuple[int, int]]) -> bool:
    return any(lo <= frame < hi for lo, hi in intervals)


def select_frames(
    scores: dict[int, dict[str, Any]],
    *,
    window: tuple[int, int],
    anchor_frames: Sequence[int],
    cycles: Sequence[Sequence[int]],
    params: SelectionParams,
    eligible: set[int] | None = None,
) -> list[dict[str, Any]]:
    """The anchor frames, then the disagreement frames (lowest pooled min IoU first, the
    cycle neighbourhoods served first until `min_in_cycles`), then the random frames; every
    chosen frame at least `spacing_frames` from every other."""
    chosen: list[dict[str, Any]] = []
    hoods = neighbourhoods(cycles, params.cycle_pad_frames, window)

    def far_enough(frame: int) -> bool:
        return all(abs(frame - c["raw_frame"]) >= params.spacing_frames for c in chosen)

    for frame in anchor_frames:
        chosen.append(
            {
                "raw_frame": int(frame),
                "origin": "six_view_annotated",
                "reason": "the frame the FineBio authors annotated in every camera",
                "in_cycle_neighbourhood": _in_any(int(frame), hoods),
                "min_detector_box_iou": scores.get(int(frame)),
            }
        )
    ranked = sorted(scores.items(), key=lambda kv: (kv[1]["min_iou"], kv[0]))
    candidates = [
        (f, e)
        for f, e in ranked
        if window[0] <= f < window[1] and (eligible is None or f in eligible)
    ]
    taken = 0
    in_cycles = 0

    def take(frame: int, entry: dict[str, Any], phase: str) -> None:
        nonlocal taken, in_cycles
        inside = _in_any(frame, hoods)
        chosen.append(
            {
                "raw_frame": frame,
                "origin": "disagreement",
                "reason": (
                    f"arm (b) lowest detector_box_iou {entry['min_iou']:.3f} at {entry['slot']} "
                    f"({entry['rows']} slot rows pooled; {phase})"
                ),
                "in_cycle_neighbourhood": inside,
                "min_detector_box_iou": entry,
            }
        )
        taken += 1
        in_cycles += int(inside)

    for frame, entry in candidates:
        if in_cycles >= params.min_in_cycles or taken >= params.disagreement_frames:
            break
        if _in_any(frame, hoods) and far_enough(frame):
            take(frame, entry, "centrifuge-cycle neighbourhood first")
    for frame, entry in candidates:
        if taken >= params.disagreement_frames:
            break
        if far_enough(frame):
            take(frame, entry, "window-wide")
    rng = np.random.default_rng(params.seed)
    pool = sorted(
        f for f in (eligible if eligible is not None else scores) if window[0] <= f < window[1]
    )
    randoms = 0
    attempts = 0
    while randoms < params.random_frames and pool and attempts < 10_000:
        attempts += 1
        frame = int(pool[int(rng.integers(0, len(pool)))])
        if far_enough(frame):
            chosen.append(
                {
                    "raw_frame": frame,
                    "origin": "random",
                    "reason": (
                        f"uniform over the eligible frames, seed {params.seed}, draw {attempts}"
                    ),
                    "in_cycle_neighbourhood": _in_any(frame, hoods),
                    "min_detector_box_iou": scores.get(frame),
                }
            )
            randoms += 1
    return sorted(chosen, key=lambda c: c["raw_frame"])


def fixed_view_table(
    rows: Iterable[dict[str, Any]],
    slots: dict[str, list[dict[str, Any]]],
    fixed_views: Sequence[str],
    window: tuple[int, int],
) -> dict[str, dict[str, Any]]:
    """Per fixed view: non-group and group slots, frames with a plate mask and the plate's
    det-box IoU median, the pooled det-box IoU median (arm (b))."""
    plate_frames: dict[str, set[int]] = defaultdict(set)
    plate_ious: dict[str, list[float]] = defaultdict(list)
    pooled: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        view = row.get("view")
        if view not in fixed_views or row.get("source") not in SAM3_SOURCES:
            continue
        if is_group(row["slot"]):
            continue
        iou = (row.get("provenance") or {}).get("detector_box_iou")
        if iou is not None:
            pooled[view].append(float(iou))
        if row.get("object_class") == PLATE_CLASS and row.get("mask_area_px"):
            plate_frames[view].add(int(row["frame_index"]))
            if iou is not None:
                plate_ious[view].append(float(iou))
    length = window[1] - window[0]
    table = {}
    for view in fixed_views:
        view_slots = slots.get(view, [])
        table[view] = {
            "non_group_slots": sum(1 for s in view_slots if not is_group(s["label"])),
            "group_slots": sum(1 for s in view_slots if is_group(s["label"])),
            "plate_slot": any(s["class"] == PLATE_CLASS for s in view_slots),
            "plate_frames_with_mask": len(plate_frames[view]),
            "plate_visible_fraction": round(len(plate_frames[view]) / length, 4)
            if length
            else None,
            "plate_detector_box_iou_median": (
                round(float(np.median(plate_ious[view])), 4) if plate_ious[view] else None
            ),
            "detector_box_iou_median": (
                round(float(np.median(pooled[view])), 4) if pooled[view] else None
            ),
        }
    return table


def choose_fixed_view(
    table: dict[str, dict[str, Any]], plate_visible_fraction: float
) -> tuple[str, str]:
    """The fixed view with the most individually tracked slots among those whose plate slot
    has a mask on at least `plate_visible_fraction` of the window; ties by det-box IoU."""
    ok = {
        v: t
        for v, t in table.items()
        if t["plate_slot"] and (t["plate_visible_fraction"] or 0.0) >= plate_visible_fraction
    }
    if not ok:
        raise ValueError("no fixed view has a plate slot with a mask on most frames")
    view = max(
        ok, key=lambda v: (ok[v]["non_group_slots"], ok[v]["detector_box_iou_median"] or 0.0)
    )
    others = ", ".join(
        f"{v} {t['non_group_slots']}+{t['group_slots']} group"
        for v, t in table.items()
        if v != view
    )
    reason = (
        f"{view} has the most individually tracked slots ({table[view]['non_group_slots']} "
        f"non-group, {table[view]['group_slots']} group; others: {others}) and its plate slot has "
        f"a mask on {table[view]['plate_visible_fraction']:.1%} of the window (det-box IoU median "
        f"{table[view]['plate_detector_box_iou_median']}); group slots have no single detector "
        "box and are outside the disagreement pool"
    )
    return view, reason


def build_config(
    *,
    clip_path: Path,
    clip: dict[str, Any],
    seeds_path: Path,
    arm_b_dir: Path,
    slots: dict[str, list[dict[str, Any]]],
    fixed_view: str,
    fixed_view_reason: str,
    fixed_view_table_: dict[str, dict[str, Any]],
    frames: list[dict[str, Any]],
    params: SelectionParams,
    eligible_count: int,
    repository_root: Path,
) -> dict[str, Any]:
    start, end = int(clip["window"]["start_frame"]), int(clip["window"]["end_frame_exclusive"])
    offset = int(clip.get("frame_index_offset", start))
    six = list(clip["views"])
    pair = [clip.get("fpv_view", FPV_VIEW), fixed_view]
    cycles = [list(map(int, c)) for c in clip.get("centrifuge_cycles_in_window", [])]
    six_view_frames = [
        int(f) for f in clip.get("annotated_frames_in_window", {}).get("six_view", [])
    ]
    entries = []
    for entry in frames:
        raw = int(entry["raw_frame"])
        views = six if entry["origin"] == "six_view_annotated" else pair
        entries.append(
            {
                "raw_frame": raw,
                "proxy_frame": raw - offset,
                "views": views,
                "origin": entry["origin"],
                "reason": entry["reason"],
                "in_cycle_neighbourhood": bool(entry["in_cycle_neighbourhood"]),
                "arm_b_min_detector_box_iou": entry.get("min_detector_box_iou"),
            }
        )
    return {
        "schema": SCHEMA,
        "config_kind": CONFIG_KIND,
        "config_id": f"finebio-{clip['trial']}-review-anchors",
        "anchor_kind": ANCHOR_KIND,
        "clip_config": fs_common.relative_uri(Path(clip_path), repository_root),
        "trial": clip["trial"],
        "window": {"start": start, "end": end, "frame_index_offset": offset},
        "frame_convention": (
            "raw = shipped video frame index; proxy frame = raw - frame_index_offset (the arms' "
            "analysis frame); masks under a worker run are named by the proxy frame"
        ),
        "views": {"six_view": six, "pair": pair},
        "fixed_view": fixed_view,
        "fixed_view_reason": fixed_view_reason,
        "fixed_view_candidates": fixed_view_table_,
        "selection": {
            "arm_b": {
                "uri": fs_common.relative_uri(Path(arm_b_dir), repository_root),
                "observations_sha256": fs_common.sha256_file(
                    Path(arm_b_dir) / "observations.jsonl"
                ),
            },
            "seeds": {
                "uri": fs_common.relative_uri(Path(seeds_path), repository_root),
                "sha256": fs_common.sha256_file(Path(seeds_path)),
            },
            "six_view_annotated_frames": six_view_frames,
            "disagreement": {
                "count": params.disagreement_frames,
                "rule": (
                    "per frame, the minimum of arm (b)'s provenance.detector_box_iou over the "
                    "non-group SAM3 rows of the pair views; frames ranked lowest first; a frame is "
                    "taken when it is >= spacing_frames from every frame already chosen (the "
                    "six-view frame included); the centrifuge-cycle neighbourhoods are served "
                    "first until min_in_cycles is met"
                ),
                "spacing_frames": params.spacing_frames,
                "min_in_cycles": params.min_in_cycles,
                "centrifuge_cycles": cycles,
                "cycle_pad_frames": params.cycle_pad_frames,
                "cycle_neighbourhoods": [
                    list(h) for h in neighbourhoods(cycles, params.cycle_pad_frames, (start, end))
                ],
            },
            "random": {
                "count": params.random_frames,
                "seed": params.seed,
                "rule": (
                    "numpy default_rng(seed) uniform over the eligible frames, rejection on spacing"
                ),
            },
            "eligible_frames": eligible_count,
            "eligibility": (
                "a frame inside the window with at least one SAM3 row in each pair view in arm (b) "
                "and a valid fpv pose"
            ),
        },
        "slots": slots,
        "frames": entries,
        "counts": {
            "frames": len(entries),
            "by_origin": dict(Counter(e["origin"] for e in entries)),
            "in_cycle_neighbourhood": sum(1 for e in entries if e["in_cycle_neighbourhood"]),
            "frame_views": sum(len(e["views"]) for e in entries),
            "cells": sum(len(slots.get(v, [])) for e in entries for v in e["views"]),
        },
        "provenance": {"author": None, "reviewed_at": None, "tool": "battle-finebio-anchors"},
        "claim_boundary": CLAIM_BOUNDARY,
        "licence": LICENCE_NOTE,
    }


def load_config(path: Path) -> dict[str, Any]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("config_kind") != CONFIG_KIND:
        raise ValueError(f"{path} is not a {CONFIG_KIND}")
    frames = [int(f["raw_frame"]) for f in doc["frames"]]
    if frames != sorted(set(frames)):
        raise ValueError("anchor frames must be increasing and unique")
    offset = int(doc["window"]["frame_index_offset"])
    for entry in doc["frames"]:
        if int(entry["proxy_frame"]) != int(entry["raw_frame"]) - offset:
            raise ValueError(f"frame {entry['raw_frame']}: proxy_frame != raw - offset")
        for view in entry["views"]:
            if view not in doc["slots"]:
                raise ValueError(f"frame {entry['raw_frame']}: view {view} has no slot list")
    return doc


def run_select(args: argparse.Namespace) -> dict[str, Any]:
    root = Path.cwd().resolve()
    clip = load_clip_doc(args.clip)
    slots = load_seed_slots(args.seeds)
    arm_b = Path(args.arm_b)
    params = SelectionParams(
        spacing_frames=args.spacing,
        disagreement_frames=args.disagreement_frames,
        random_frames=args.random_frames,
        min_in_cycles=args.min_in_cycles,
        cycle_pad_frames=args.cycle_pad,
        seed=args.seed,
        plate_visible_fraction=args.plate_visible_fraction,
    )
    start, end = int(clip["window"]["start_frame"]), int(clip["window"]["end_frame_exclusive"])
    rows = read_rows(arm_b / "observations.jsonl", sam3_only=True)
    table = fixed_view_table(rows, slots, clip["fixed_views"], (start, end))
    if args.fixed_view == "auto":
        fixed_view, reason = choose_fixed_view(table, params.plate_visible_fraction)
    else:
        fixed_view = args.fixed_view
        reason = (
            args.fixed_view_reason or choose_fixed_view({fixed_view: table[fixed_view]}, 0.0)[1]
        )
    fpv = clip.get("fpv_view", FPV_VIEW)
    pair = (fpv, fixed_view)
    scores = disagreement_scores(rows, pair)
    # fpv pose validity: SAM3 rows carry pose_valid; a frame with an invalid pose is not eligible.
    invalid = {
        int(r["frame_index"]) for r in rows if r.get("view") == fpv and r.get("pose_valid") is False
    }
    eligible = {f for f in scores if f not in invalid}
    six_view_frames = [
        int(f) for f in clip.get("annotated_frames_in_window", {}).get("six_view", [])
    ]
    frames = select_frames(
        scores,
        window=(start, end),
        anchor_frames=six_view_frames,
        cycles=clip.get("centrifuge_cycles_in_window", []),
        params=params,
        eligible=eligible,
    )
    config = build_config(
        clip_path=Path(args.clip),
        clip=clip,
        seeds_path=Path(args.seeds),
        arm_b_dir=arm_b,
        slots=slots,
        fixed_view=fixed_view,
        fixed_view_reason=reason,
        fixed_view_table_=table,
        frames=frames,
        params=params,
        eligible_count=len(eligible),
        repository_root=root,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fs_common.write_json(output, config)
    print(f"fixed view: {fixed_view}")
    for entry in config["frames"]:
        print(
            f"  raw {entry['raw_frame']} (proxy {entry['proxy_frame']}) {entry['origin']:20s} "
            f"views {','.join(entry['views'])}: {entry['reason']}"
        )
    print(f"{config['counts']} -> {output}")
    return config


# --------------------------------------------------------------------------------------------
# workspace: candidate decode requests, the worker, assembly and sheets


def arm_b_cells(
    config: dict[str, Any], arm_b_dir: Path, repository_root: Path
) -> tuple[dict[tuple[int, str, str], dict[str, Any]], dict[str, Path]]:
    """(raw frame, view, label) -> arm (b)'s SAM3 row on the anchor frames, and the worker
    run per view (where the masks are)."""
    frames = [int(f["raw_frame"]) for f in config["frames"]]
    views = sorted({v for f in config["frames"] for v in f["views"]})
    rows = read_rows(
        Path(arm_b_dir) / "observations.jsonl", frames=frames, views=views, sam3_only=True
    )
    by_cell = {(int(r["frame_index"]), str(r["view"]), str(r["slot"])): r for r in rows}
    return by_cell, arm_worker_runs(Path(arm_b_dir), repository_root)


def build_requests(
    config: dict[str, Any],
    cells: dict[tuple[int, str, str], dict[str, Any]],
    *,
    proxies: dict[str, Path],
    view_sizes: dict[str, tuple[int, int]],
    margin: float = DEFAULT_MARGIN,
) -> dict[str, Any]:
    """Per view and anchor frame, the slots with an arm (b) box on that frame: the reference
    box, the positive point (arm (b)'s mask centroid when it lies inside the box, else the box
    centre) and the three alternate prompts the worker decodes."""
    offset = int(config["window"]["frame_index_offset"])
    views: dict[str, Any] = {}
    for entry in config["frames"]:
        raw = int(entry["raw_frame"])
        for view in entry["views"]:
            width, height = view_sizes[view]
            spec = views.setdefault(
                view,
                {"video": str(proxies[view]), "image_hw": [height, width], "frames": []},
            )
            slot_specs = []
            for slot in config["slots"][view]:
                row = cells.get((raw, view, slot["label"]))
                if row is None or row.get("box_xyxy_px") is None:
                    continue
                box = tuple(float(v) for v in row["box_xyxy_px"])
                centroid = row.get("mask_centroid_px")
                if centroid is not None and _inside(centroid, box):
                    point, point_source = (
                        [float(centroid[0]), float(centroid[1])],
                        "arm_b_mask_centroid",
                    )
                else:
                    point, point_source = (
                        [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2],
                        "box_centre",
                    )
                slot_specs.append(
                    {
                        "slot": int(slot["slot"]),
                        "label": slot["label"],
                        "class": slot["class"],
                        "box": [round(v, 1) for v in box],
                        "margin_box": [
                            round(v, 1) for v in expand_box(box, margin, (height, width))
                        ],
                        "point": [round(v, 1) for v in point],
                        "point_source": point_source,
                    }
                )
            spec["frames"].append(
                {"raw_frame": raw, "proxy_frame": raw - offset, "slots": slot_specs}
            )
    return {
        "schema": SCHEMA,
        "trial": config["trial"],
        "encoder_side": ENCODER_SIDE,
        "margin": margin,
        "candidate_kinds": list(CANDIDATE_KINDS),
        "views": views,
    }


def mask_metrics(mask: np.ndarray, box: Sequence[float]) -> dict[str, Any]:
    bbox = mask_bbox(mask)
    if bbox is None:
        return {"mask_area_px": 0, "mask_bbox_px": None, "mask_bbox_iou_vs_box": 0.0}
    return {
        "mask_area_px": int(mask.sum()),
        "mask_bbox_px": [int(v) for v in bbox],
        "mask_bbox_iou_vs_box": round(box_iou(list(bbox), list(box)), 4),
    }


def candidate_file(view: str, raw_frame: int, slot: int, index: int) -> str:
    return f"candidates/{view}/f{raw_frame:06d}_s{slot:02d}_c{index}.png"


def decode_worker_main(args: argparse.Namespace) -> int:
    """The GPU half: one image encode per anchor frame and view, three decodes per slot (the
    0.15-margin box, the box plus a positive point, the tight box's second-ranked output),
    every mask written as PNG. Runs under the MuggledSAM interpreter; imports nothing from
    the battle package."""
    requests = json.loads(Path(args.requests).read_text(encoding="utf-8"))
    out_dir = Path(args.results).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    decision = gpu_guard.evaluate(
        mode=args.gpu_guard,
        allowed_pids=args.allow_gpu_neighbour,
        profile=args.gpu_guard_profile,
        expected_peak_vram_bytes=args.expected_peak_vram_bytes,
        own_pid=os.getpid(),
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "state": "failed",
        "reason": None,
        "gpu_guard": decision.as_provenance(),
        "interpreter": sys.executable,
        "model": str(args.model),
        "encoder_side": int(requests["encoder_side"]),
        "margin": float(requests["margin"]),
        "views": {},
        "image_encodes": 0,
        "decodes": 0,
        "elapsed_seconds": None,
        "model_load_seconds": None,
        "gpu_peak_vram_bytes": None,
    }

    def finish(state: str, reason: str | None) -> int:
        result["state"] = state
        result["reason"] = reason
        result["elapsed_seconds"] = time.perf_counter() - started
        fs_common.write_json(Path(args.results), result, sort_keys=True)
        if reason:
            print(f"{state}: {reason}", file=sys.stderr, flush=True)
        return 0 if state == "succeeded" else 2

    if not decision.accepted:
        return finish("blocked", f"GPU guard refused: {decision.reason}")
    try:
        import cv2
        import torch

        if str(MUGGLED_SAM_SOURCE) not in sys.path:
            sys.path.insert(0, str(MUGGLED_SAM_SOURCE))
        from muggled_sam.make_sam import make_sam_from_state_dict

        load_start = time.perf_counter()
        core = make_sam_from_state_dict(str(args.model))
        core.to(device="cuda:0", dtype=torch.bfloat16)
        interact = core.get_interactive_context()
        torch.cuda.reset_peak_memory_stats()
        result["model_load_seconds"] = time.perf_counter() - load_start
        side = int(requests["encoder_side"])

        def decode(encoded, height, width, norm_box, points):
            with torch.inference_mode():
                prompts = interact.encode_prompts([norm_box], points, [])
                masks, ious = interact.generate_masks(encoded, prompts)
                order = torch.argsort(ious[0], descending=True).tolist()
                logits = torch.nn.functional.interpolate(
                    masks[:, order[:2]].float(),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
                pair = logits.gt(0).squeeze(0).cpu().numpy()
            result["decodes"] += 1
            scores = [round(float(ious[0, k]), 4) for k in order]
            return pair, scores, order

        for view, spec in requests["views"].items():
            view_result: dict[str, Any] = {"frames": {}, "frames_encoded": []}
            result["views"][view] = view_result
            capture = cv2.VideoCapture(str(spec["video"]))
            if not capture.isOpened():
                raise RuntimeError(f"could not open {spec['video']}")
            (out_dir / "candidates" / view).mkdir(parents=True, exist_ok=True)
            for frame_spec in spec["frames"]:
                raw, proxy = int(frame_spec["raw_frame"]), int(frame_spec["proxy_frame"])
                if not frame_spec["slots"]:
                    view_result["frames"][str(raw)] = {}
                    continue
                capture.set(cv2.CAP_PROP_POS_FRAMES, proxy)
                ok, image = capture.read()
                if not ok:
                    raise RuntimeError(f"{view}: could not read proxy frame {proxy} (raw {raw})")
                height, width = image.shape[:2]
                with torch.inference_mode():
                    encoded = interact.encode_image(image, side, True)
                result["image_encodes"] += 1
                view_result["frames_encoded"].append(raw)
                frame_result: dict[str, Any] = {}
                for slot in frame_spec["slots"]:
                    box = slot["box"]
                    tight = [(box[0] / width, box[1] / height), (box[2] / width, box[3] / height)]
                    margin_box = slot["margin_box"]
                    margin = [
                        (margin_box[0] / width, margin_box[1] / height),
                        (margin_box[2] / width, margin_box[3] / height),
                    ]
                    point = [(slot["point"][0] / width, slot["point"][1] / height)]
                    candidates = []
                    margin_pair, margin_scores, _ = decode(encoded, height, width, margin, [])
                    point_pair, point_scores, _ = decode(encoded, height, width, tight, point)
                    tight_pair, tight_scores, _ = decode(encoded, height, width, tight, [])
                    for index, kind, mask, score, prompt in (
                        (1, "margin", margin_pair[0], margin_scores[0], margin_box),
                        (2, "box_point", point_pair[0], point_scores[0], box),
                        (3, "tight_rank2", tight_pair[1], tight_scores[1], box),
                    ):
                        uri = candidate_file(view, raw, int(slot["slot"]), index)
                        cv2.imwrite(str(out_dir / uri), mask.astype(np.uint8) * 255)
                        candidates.append(
                            {
                                "index": index,
                                "kind": kind,
                                "mask_uri": uri,
                                "prompt_box": [round(float(v), 1) for v in prompt],
                                "prompt_point": slot["point"] if kind == "box_point" else None,
                                "decoder_iou_pred": score,
                                **mask_metrics(mask, box),
                            }
                        )
                    frame_result[str(slot["slot"])] = {
                        "label": slot["label"],
                        "candidates": candidates,
                        "tight_candidate_ious": tight_scores,
                    }
                view_result["frames"][str(raw)] = frame_result
            capture.release()
            print(
                f"{view}: {len(view_result['frames_encoded'])} frames encoded, "
                f"{result['decodes']} decodes so far",
                flush=True,
            )
        torch.cuda.synchronize()
        result["gpu_peak_vram_bytes"] = int(torch.cuda.max_memory_reserved())
        return finish("succeeded", None)
    except Exception as error:  # noqa: BLE001 - recorded, not swallowed
        traceback.print_exc()
        return finish("failed", f"{type(error).__name__}: {error}")


def decode_worker_command(args: argparse.Namespace, *, requests: Path, results: Path) -> list[str]:
    command = [
        str(args.external_python),
        str(Path(__file__).resolve()),
        "decode-worker",
        "--requests",
        str(requests.resolve()),
        "--results",
        str(results.resolve()),
        "--model",
        str(Path(args.model).resolve()),
        "--gpu-guard",
        args.gpu_guard,
        "--gpu-guard-profile",
        args.gpu_guard_profile,
    ]
    if args.expected_peak_vram_bytes is not None:
        command += ["--expected-peak-vram-bytes", str(args.expected_peak_vram_bytes)]
    for pid in args.allow_gpu_neighbour:
        command += ["--allow-gpu-neighbour", str(int(pid))]
    return command


def worker_environment() -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0") or "0"
    paths = [str(MUGGLED_SAM_SOURCE)]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def assemble_cells(
    config: dict[str, Any],
    requests: dict[str, Any],
    results: dict[str, Any] | None,
    arm_b: dict[tuple[int, str, str], dict[str, Any]],
    worker_runs: dict[str, Path],
    *,
    output: Path,
) -> list[dict[str, Any]]:
    """Every (frame, view, slot) cell of the config with its candidate list: candidate 0 is
    arm (b)'s mask copied (speckle-filtered) into the workspace, the alternates come from the
    worker's results when it ran; alternates within `DUPLICATE_IOU` of an earlier candidate
    are marked `duplicate_of` so the human can skip them."""
    import cv2

    offset = int(config["window"]["frame_index_offset"])
    request_slots = {
        (view, int(f["raw_frame"]), int(s["slot"])): s
        for view, spec in requests["views"].items()
        for f in spec["frames"]
        for s in f["slots"]
    }
    cells: list[dict[str, Any]] = []
    for entry in config["frames"]:
        raw = int(entry["raw_frame"])
        for view in entry["views"]:
            for slot in config["slots"][view]:
                key = (view, raw, int(slot["slot"]))
                cell: dict[str, Any] = {
                    "raw_frame": raw,
                    "proxy_frame": raw - offset,
                    "view": view,
                    "slot": int(slot["slot"]),
                    "label": slot["label"],
                    "class": slot["class"],
                    "role": slot.get("role"),
                    "origin": entry["origin"],
                    "in_cycle_neighbourhood": entry["in_cycle_neighbourhood"],
                    "reference_box": None,
                    "arm_b_row": False,
                    "candidates": [],
                    "sheet": f"sheets/f{raw:06d}_{view}.jpg",
                }
                spec = request_slots.get(key)
                if spec is None:
                    cell["note"] = (
                        "no detector box for this slot on this frame in arm (b): only hidden / "
                        "none_fits apply"
                    )
                    cells.append(cell)
                    continue
                cell["reference_box"] = spec["box"]
                cell["point"] = spec["point"]
                cell["point_source"] = spec["point_source"]
                row = arm_b[(raw, view, slot["label"])]
                cell["arm_b_row"] = True
                cell["arm_b_detector_score"] = row.get("detector_score")
                cell["arm_b_detector_box_iou"] = (row.get("provenance") or {}).get(
                    "detector_box_iou"
                )
                masks: list[np.ndarray | None] = []
                run = worker_runs.get(view)
                source = worker_mask_path(run, row, offset) if run is not None else None
                mask0 = (
                    clean_mask(read_mask(source))
                    if source is not None and source.is_file()
                    else None
                )
                if mask0 is not None:
                    uri = candidate_file(view, raw, int(slot["slot"]), 0)
                    (output / uri).parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(output / uri), mask0.astype(np.uint8) * 255)
                    cell["candidates"].append(
                        {
                            "index": 0,
                            "kind": CANDIDATE_KINDS[0],
                            "mask_uri": uri,
                            "prompt_box": spec["box"],
                            "prompt_point": None,
                            "decoder_iou_pred": (row.get("provenance") or {}).get(
                                "decoder_iou_pred"
                            ),
                            "source_mask": str(source),
                            "speckle_filtered": True,
                            **mask_metrics(mask0, spec["box"]),
                            "duplicate_of": None,
                        }
                    )
                    masks.append(mask0)
                else:
                    cell["arm_b_mask_missing"] = str(source)
                decoded = (
                    ((results or {}).get("views", {}).get(view, {}).get("frames", {}) or {})
                    .get(str(raw), {})
                    .get(str(slot["slot"]))
                )
                for candidate in (decoded or {}).get("candidates", []):
                    mask = read_mask(output / candidate["mask_uri"])
                    duplicate = None
                    if mask is not None and mask.any():
                        for earlier, other in zip(cell["candidates"], masks, strict=True):
                            if other is not None and mask_iou(mask, other) >= DUPLICATE_IOU:
                                duplicate = earlier["index"]
                                break
                    cell["candidates"].append({**candidate, "duplicate_of": duplicate})
                    masks.append(mask if mask is not None and mask.any() else None)
                if decoded is not None:
                    cell["tight_candidate_ious"] = decoded.get("tight_candidate_ious")
                cells.append(cell)
    return cells


def _family_colour(object_class: str) -> tuple[int, int, int]:
    from .detector_seed import FAMILY_COLOURS, class_family

    return FAMILY_COLOURS[class_family(object_class)]


def render_sheets(
    config: dict[str, Any],
    cells: Sequence[dict[str, Any]],
    *,
    proxies: dict[str, Path],
    output: Path,
    tile: int = TILE,
) -> dict[str, str]:
    """Per (frame, view): `sheets/f<raw>_<view>.jpg` (rows = slots; columns = the context crop
    with the reference box, then the candidates with their kind, decoder IoU and bbox IoU) and
    `sheets/f<raw>_<view>_overview.jpg` (the frame with every slot's box and number)."""
    import cv2

    from .detector_seed import draw_seed_tile, grid

    sheets_dir = output / "sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    by_frame_view: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        by_frame_view[(cell["raw_frame"], cell["view"])].append(cell)
    columns = 1 + len(CANDIDATE_KINDS)
    written: dict[str, str] = {}
    captures: dict[str, Any] = {}
    try:
        for (raw, view), group in sorted(by_frame_view.items()):
            proxy_frame = group[0]["proxy_frame"]
            capture = captures.get(view)
            if capture is None:
                capture = cv2.VideoCapture(str(proxies[view]))
                captures[view] = capture
            capture.set(cv2.CAP_PROP_POS_FRAMES, proxy_frame)
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"{view}: could not read proxy frame {proxy_frame}")
            tiles: list[np.ndarray] = []
            blank = np.zeros((tile + 18 * 3 + 8, tile, 3), dtype=np.uint8)
            overview = image.copy()
            for cell in sorted(group, key=lambda c: c["slot"]):
                colour = _family_colour(cell["class"])
                if cell["reference_box"] is None:
                    text = np.zeros_like(blank)
                    for i, line in enumerate(
                        (
                            f"{cell['label']}  slot {cell['slot']}",
                            "no box on this frame",
                            "hidden / none_fits only",
                        )
                    ):
                        cv2.putText(
                            text,
                            line,
                            (4, 24 + 20 * i),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (200, 200, 200),
                            1,
                            cv2.LINE_AA,
                        )
                    tiles.append(text)
                    tiles.extend([blank.copy() for _ in range(columns - 1)])
                    continue
                box = tuple(float(v) for v in cell["reference_box"])
                cv2.rectangle(
                    overview, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), colour, 3
                )
                cv2.putText(
                    overview,
                    f"{cell['slot']} {cell['label']}",
                    (int(box[0]), max(20, int(box[1]) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    colour,
                    2,
                    cv2.LINE_AA,
                )
                tiles.append(
                    draw_seed_tile(
                        image,
                        box=box,
                        mask=None,
                        colour=colour,
                        lines=[
                            f"{cell['label']}  slot {cell['slot']}",
                            f"raw f{raw}  {view}  det {cell.get('arm_b_detector_score') or 0:.2f}",
                            "reference box (arm b prompt)",
                        ],
                        tile=tile,
                    )
                )
                by_index = {c["index"]: c for c in cell["candidates"]}
                for index in range(len(CANDIDATE_KINDS)):
                    candidate = by_index.get(index)
                    if candidate is None:
                        tiles.append(blank.copy())
                        continue
                    mask = read_mask(output / candidate["mask_uri"])
                    dup = candidate.get("duplicate_of")
                    dup_text = f"  = c{dup}" if dup is not None else ""
                    tiles.append(
                        draw_seed_tile(
                            image,
                            box=box,
                            mask=mask,
                            colour=colour if dup is None else (128, 128, 128),
                            lines=[
                                f"c{index} {candidate['kind']}{dup_text}",
                                f"dec {candidate.get('decoder_iou_pred') or 0:.2f}  bbox IoU "
                                f"{candidate.get('mask_bbox_iou_vs_box') or 0:.2f}",
                                f"area {candidate.get('mask_area_px', 0)} px",
                            ],
                            tile=tile,
                        )
                    )
            title = (
                f"{config['trial']} raw f{raw} (proxy {proxy_frame}) {view}: {len(group)} slots; "
                "pick one candidate index per row, or hidden / box / none_fits"
            )
            sheet = grid(tiles, columns, title)
            path = sheets_dir / f"f{raw:06d}_{view}.jpg"
            cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
            scale = 1280 / overview.shape[1]
            small = cv2.resize(overview, (1280, int(overview.shape[0] * scale)))
            cv2.putText(
                small,
                f"{config['trial']} raw f{raw} {view}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
            )
            overview_path = sheets_dir / f"f{raw:06d}_{view}_overview.jpg"
            cv2.imwrite(str(overview_path), small, [cv2.IMWRITE_JPEG_QUALITY, 85])
            written[f"f{raw:06d}_{view}"] = str(path.relative_to(output))
    finally:
        for capture in captures.values():
            capture.release()
    return written


def decisions_template(
    config: dict[str, Any], cells: Sequence[dict[str, Any]], *, workspace: Path
) -> dict[str, Any]:
    entries = []
    for cell in cells:
        identity = cell["class"] if cell["class"] in SINGLETON_IDENTITY_CLASSES else None
        entries.append(
            {
                "raw_frame": cell["raw_frame"],
                "proxy_frame": cell["proxy_frame"],
                "view": cell["view"],
                "slot": cell["slot"],
                "label": cell["label"],
                "class": cell["class"],
                "origin": cell["origin"],
                "candidates": [c["index"] for c in cell["candidates"]],
                "sheet": cell["sheet"],
                "decision": None,
                "instance_identity": identity,
                "note": "",
            }
        )
    return {
        "schema": f"{SCHEMA}/decisions",
        "workspace": str(workspace.resolve()),
        "config": config["config_id"],
        "trial": config["trial"],
        "author": None,
        "reviewed_at": None,
        "how": (
            "Open sheets/f<raw>_<view>.jpg (and the _overview.jpg beside it). Per row set "
            "decision to the index of the candidate whose mask is the object the label names "
            "(0 = arm (b)'s own mask), or 'box' (the reference box is the object and no candidate "
            "mask fits), 'hidden' (the object is not visible in this view on this frame; any "
            "arm mask there is a false positive), 'none_fits' (visible, but no candidate and no "
            "box is acceptable), or leave null (unlabelled, skipped). instance_identity: the same "
            "short name for the same physical object across views and frames (tube_A, ...); "
            "pre-filled for the bench's singletons; leave null when you cannot tell. Nothing is "
            "drawn. Then: battle-finebio-anchors score --workspace <this dir> --record <this "
            "file> --arms a=...,b=...,c=...,d=..."
        ),
        "decision_values": ["<candidate index>", *DECISION_WORDS, None],
        "cells": entries,
        "claim_boundary": CLAIM_BOUNDARY,
        "licence": LICENCE_NOTE,
    }


def workspace_readme(
    config: dict[str, Any], cells: Sequence[dict[str, Any]], results: dict[str, Any] | None
) -> str:
    n_frames = len(config["frames"])
    frame_views = sum(len(f["views"]) for f in config["frames"])
    with_candidates = sum(1 for c in cells if c["candidates"])
    lines = [
        f"# Human review anchors, {config['trial']} (gate 2, soft)",
        "",
        f"{n_frames} frames, {frame_views} frame-views, {len(cells)} cells ({with_candidates} with "
        "decoder candidates). Nothing here is a label until `decisions.json` exists; the pipeline "
        "never waits for it.",
        "",
        "- `sheets/f<raw>_<view>.jpg`: one row per slot, columns = context crop with the reference "
        "box, then candidates c0 (arm (b)'s mask from the tight detector box), c1 (0.15-margin "
        "box), c2 (box + positive point), c3 (tight box, decoder's second-ranked output); `= cK` "
        "marks a duplicate of an earlier candidate.",
        "- `sheets/f<raw>_<view>_overview.jpg`: the whole frame with every slot's box and number.",
        "- `candidates/<view>/f<raw>_s<slot>_c<k>.png`: the candidate masks (proxy resolution).",
        "- `workspace.json`: every cell with its reference box and candidates; `requests.json` / "
        "`results.json`: what the decoder was asked and what it answered (GPU guard record "
        "inside).",
        "- `decisions.template.json`: copy to `decisions.json` and fill `decision` and "
        "`instance_identity` per cell (see `how` inside).",
        "",
        "Score whenever labels exist:",
        "",
        "```bash",
        "uv run battle-finebio-anchors score --workspace <this directory> \\",
        "  --record <this directory>/decisions.json \\",
        "  --arms a=<arms run>/a-boxes-only,b=<arms run>/b-box-decode-arm,\\",
        "c=<arms run>/c-video-memory-arm,d=<arms run>/d-video-memory-arm \\",
        "  --output <this directory>/scoreboard",
        "# <arms run> = runs/finebio-arms-<trial>-<date>",
        "```",
        "",
        f"Decode: {results.get('state') if results else 'not run'}"
        + (
            f", {results.get('image_encodes')} image encodes, {results.get('decodes')} decodes, "
            f"{(results.get('elapsed_seconds') or 0):.0f} s, peak VRAM "
            f"{(results.get('gpu_peak_vram_bytes') or 0) / 2**30:.2f} GiB"
            if results
            else ""
        ),
        "",
        config["claim_boundary"],
        "",
        config["licence"],
        "",
    ]
    return "\n".join(lines)


def run_workspace(args: argparse.Namespace) -> dict[str, Any]:
    root = Path.cwd().resolve()
    config_path = Path(args.config)
    config = load_config(config_path)
    clip = load_clip_doc(root / config["clip_config"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    proxies = {view: root / uri for view, uri in clip["proxies"].items()}
    view_sizes = {view: (int(w), int(h)) for view, (w, h) in clip["view_sizes"].items()}
    arm_b = Path(args.arm_b)
    cells_b, worker_runs = arm_b_cells(config, arm_b, root)
    requests = build_requests(
        config, cells_b, proxies=proxies, view_sizes=view_sizes, margin=args.margin
    )
    requests_path = output / "requests.json"
    fs_common.write_json(requests_path, requests)
    results_path = output / "results.json"
    results: dict[str, Any] | None = None
    if args.skip_decode:
        results = {"state": "skipped", "reason": "--skip-decode"}
        fs_common.write_json(results_path, results)
    else:
        interpreter = Path(args.external_python)
        if not interpreter.is_file():
            results = {
                "state": "blocked",
                "reason": f"MuggledSAM interpreter missing: {interpreter}",
            }
            fs_common.write_json(results_path, results)
        else:
            command = decode_worker_command(args, requests=requests_path, results=results_path)
            fs_common.write_json(output / "worker_command.json", command)
            started = time.perf_counter()
            completed = subprocess.run(
                command, capture_output=True, text=True, env=worker_environment(), check=False
            )
            (output / "worker.log").write_text(completed.stdout + "\n" + completed.stderr)
            if results_path.is_file():
                results = json.loads(results_path.read_text(encoding="utf-8"))
            else:
                results = {
                    "state": "failed",
                    "reason": f"worker exited {completed.returncode} without results.json",
                }
            results["wall_seconds"] = time.perf_counter() - started
            fs_common.write_json(results_path, results, sort_keys=True)
            print(f"decode worker {results['state']} in {results['wall_seconds']:.0f}s", flush=True)
    decoded = results if results and results.get("state") == "succeeded" else None
    cells = assemble_cells(config, requests, decoded, cells_b, worker_runs, output=output)
    sheets = render_sheets(config, cells, proxies=proxies, output=output)
    workspace = {
        "schema": f"{SCHEMA}/workspace",
        "config": {
            "uri": fs_common.relative_uri(config_path, root),
            "sha256": fs_common.sha256_file(config_path),
        },
        "arm_b": fs_common.relative_uri(arm_b, root),
        "worker_runs": {v: fs_common.relative_uri(p, root) for v, p in worker_runs.items()},
        "trial": config["trial"],
        "frame_index_offset": config["window"]["frame_index_offset"],
        "decode": {
            k: results.get(k)
            for k in (
                "state",
                "reason",
                "image_encodes",
                "decodes",
                "elapsed_seconds",
                "wall_seconds",
                "gpu_peak_vram_bytes",
            )
        }
        if results
        else None,
        "cells": cells,
        "sheets": sheets,
        "counts": {
            "cells": len(cells),
            "cells_with_candidates": sum(1 for c in cells if c["candidates"]),
            "candidates": sum(len(c["candidates"]) for c in cells),
            "duplicates": sum(
                1 for c in cells for k in c["candidates"] if k.get("duplicate_of") is not None
            ),
            "cells_without_box": sum(1 for c in cells if c["reference_box"] is None),
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "licence": LICENCE_NOTE,
    }
    fs_common.write_json(output / "workspace.json", workspace)
    fs_common.write_json(
        output / "decisions.template.json", decisions_template(config, cells, workspace=output)
    )
    (output / "README.md").write_text(workspace_readme(config, cells, results), encoding="utf-8")
    print(json.dumps(workspace["counts"]))
    if results and results.get("state") not in ("succeeded", "skipped"):
        print(f"decode {results.get('state')}: {results.get('reason')}", file=sys.stderr)
    return workspace


def load_workspace(path: Path) -> dict[str, Any]:
    doc = json.loads((Path(path) / "workspace.json").read_text(encoding="utf-8"))
    if doc.get("schema") != f"{SCHEMA}/workspace":
        raise ValueError(f"{path} is not a battle-finebio-anchors workspace")
    return doc


# --------------------------------------------------------------------------------------------
# the human record and the scoreboard


def load_record(path: Path | None) -> dict[str, Any]:
    """The human's decisions file; a missing path is an empty record (nothing labelled)."""
    if path is None or not Path(path).is_file():
        return {"schema": f"{SCHEMA}/decisions", "cells": [], "author": None, "reviewed_at": None}
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("schema") != f"{SCHEMA}/decisions":
        raise ValueError(f"{path} is not a battle-finebio-anchors decisions file")
    return doc


def cell_key(cell: dict[str, Any]) -> tuple[int, str, int]:
    return (int(cell["raw_frame"]), str(cell["view"]), int(cell["slot"]))


def anchor_state(decision: Any) -> str:
    if decision is None:
        return "unlabeled"
    if isinstance(decision, bool):
        raise ValueError(f"decision must be a candidate index or one of {DECISION_WORDS}")
    if isinstance(decision, int):
        return "mask"
    if decision in DECISION_WORDS:
        return str(decision)
    raise ValueError(f"unknown decision {decision!r}")


@dataclass
class Anchor:
    cell: dict[str, Any]
    state: str  # mask | box | hidden | none_fits | unlabeled
    mask: np.ndarray | None = None
    bbox: Box | None = None
    identity: str | None = None
    candidate_index: int | None = None


def anchors_from_record(
    workspace: dict[str, Any], record: dict[str, Any], *, workspace_dir: Path
) -> list[Anchor]:
    decisions = {cell_key(c): c for c in record.get("cells", [])}
    anchors = []
    for cell in workspace["cells"]:
        entry = decisions.get(cell_key(cell), {})
        state = anchor_state(entry.get("decision"))
        anchor = Anchor(cell=cell, state=state, identity=entry.get("instance_identity") or None)
        if state == "mask":
            index = int(entry["decision"])
            candidate = next((c for c in cell["candidates"] if c["index"] == index), None)
            if candidate is None:
                raise ValueError(
                    f"frame {cell['raw_frame']} {cell['view']} slot {cell['slot']}: candidate "
                    f"{index} does not exist"
                )
            mask = read_mask(workspace_dir / candidate["mask_uri"])
            if mask is None or not mask.any():
                raise FileNotFoundError(f"candidate mask unavailable: {candidate['mask_uri']}")
            anchor.mask = mask
            anchor.bbox = mask_bbox(mask)
            anchor.candidate_index = index
        elif state == "box":
            if cell.get("reference_box") is None:
                raise ValueError(
                    f"frame {cell['raw_frame']} {cell['view']} slot {cell['slot']}: 'box' needs a "
                    "reference box"
                )
            anchor.bbox = tuple(float(v) for v in cell["reference_box"])
        anchors.append(anchor)
    return anchors


def match_arm_row(rows: Sequence[dict[str, Any]], anchor: Anchor) -> dict[str, Any] | None:
    """The arm's row for a cell: its SAM3 row with the cell's slot label, else the same-class
    detector row whose box overlaps the anchor's reference (the boxes-only arm, or an arm
    without a mask there)."""
    cell = anchor.cell
    for row in rows:
        if row.get("source") in SAM3_SOURCES and (
            row.get("slot") == cell["label"]
            or (row.get("provenance") or {}).get("worker_label") == cell["label"]
        ):
            return row
    reference = anchor.bbox or cell.get("reference_box")
    if reference is None:
        return None
    best, best_iou = None, MATCH_IOU_DETECTOR
    for row in rows:
        if row.get("source") != "detector" or row.get("object_class") != cell["class"]:
            continue
        if row.get("box_xyxy_px") is None:
            continue
        iou = box_iou(list(row["box_xyxy_px"]), list(reference))
        if iou >= best_iou:
            best, best_iou = row, iou
    return best


def score_cell(
    anchor: Anchor, row: dict[str, Any] | None, arm_mask: np.ndarray | None
) -> dict[str, Any]:
    """One cell of one arm; pure so the tests pin every branch."""
    cell = anchor.cell
    base: dict[str, Any] = {
        "raw_frame": cell["raw_frame"],
        "view": cell["view"],
        "slot": cell["slot"],
        "label": cell["label"],
        "class": cell["class"],
        "origin": cell["origin"],
        "in_cycle_neighbourhood": cell.get("in_cycle_neighbourhood", False),
        "anchor_state": anchor.state,
        "arm_slot": None if row is None else row.get("slot"),
        "arm_source": None if row is None else row.get("source"),
        "mask_iou": None,
        "box_iou": None,
        "arm_area": None,
        "anchor_area": None,
    }
    arm_present = arm_mask is not None and bool(arm_mask.any())
    arm_bbox: Box | None = mask_bbox(arm_mask) if arm_present else None
    if arm_bbox is None and row is not None:
        box = (
            row.get("mask_bbox_px") if row.get("source") in SAM3_SOURCES else row.get("box_xyxy_px")
        )
        arm_bbox = tuple(float(v) for v in box) if box is not None else None
    if anchor.state == "unlabeled":
        return {**base, "outcome": "unlabeled_skipped"}
    if anchor.state == "none_fits":
        return {**base, "outcome": "none_fits_skipped"}
    if anchor.state == "hidden":
        if arm_present:
            return {**base, "outcome": "hidden_false_positive", "arm_area": int(arm_mask.sum())}
        if row is not None and row.get("source") in SAM3_SOURCES:
            return {**base, "outcome": "hidden_false_positive", "arm_area": 0}
        return {**base, "outcome": "hidden_correct"}
    if arm_bbox is None:
        return {
            **base,
            "outcome": "run_mask_missing",
            "mask_iou": 0.0 if anchor.mask is not None else None,
            "box_iou": 0.0,
        }
    assert anchor.bbox is not None
    out = {
        **base,
        "outcome": "scored",
        "box_iou": round(box_iou(list(arm_bbox), list(anchor.bbox)), 4),
    }
    if anchor.mask is not None:
        out["anchor_area"] = int(anchor.mask.sum())
        if arm_present:
            assert arm_mask is not None
            if arm_mask.shape != anchor.mask.shape:
                raise ValueError("arm mask and anchor mask differ in size")
            out["mask_iou"] = round(mask_iou(arm_mask, anchor.mask), 4)
            out["arm_area"] = int(arm_mask.sum())
        else:
            out["mask_iou"] = None if row is not None and row.get("source") == "detector" else 0.0
    return out


def tracks_at(arm_dir: Path, frames: Iterable[int]) -> dict[int, dict[tuple[str, str], str]]:
    """frame -> (view, slot) -> track id from the arm's tracks.jsonl `support_slots`."""
    path = Path(arm_dir) / "tracks" / "tracks.jsonl"
    out: dict[int, dict[tuple[str, str], str]] = defaultdict(dict)
    if not path.is_file():
        return out
    wanted = {int(f) for f in frames}
    keys = {f'"frame_index":{f},' for f in wanted}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            if '"frame_index":' in line and not any(k in line for k in keys):
                continue
            row = json.loads(line)
            if int(row["frame_index"]) not in wanted:
                continue
            for view, slot in (row.get("support_slots") or {}).items():
                out[int(row["frame_index"])][(str(view), str(slot))] = str(row["track_id"])
    return out


def idf1(pairs: Sequence[tuple[str, str | None]]) -> dict[str, Any]:
    """Identity F1 over labelled cells: `pairs` = (human identity, arm track id or None per
    cell). IDTP is the maximum-weight one-to-one matching between identities and track ids
    (weight = cells in common); IDF1 = 2 IDTP / (cells with an identity + cells with a track)."""
    from .multiview_tracks import hungarian

    human_cells = len(pairs)
    weights: dict[tuple[str, str], int] = Counter()
    track_cells = 0
    for identity, track in pairs:
        if track is not None:
            weights[(identity, track)] += 1
            track_cells += 1
    identities = sorted({i for i, _ in weights})
    tracks = sorted({t for _, t in weights})
    idtp = 0
    if identities and tracks:
        cost = np.zeros((len(identities), len(tracks)))
        for (identity, track), w in weights.items():
            cost[identities.index(identity), tracks.index(track)] = -w
        for i, j in hungarian(cost):
            idtp += int(-cost[i, j])
    denominator = human_cells + track_cells
    by_identity: dict[str, set[str]] = defaultdict(set)
    by_track: dict[str, set[str]] = defaultdict(set)
    for identity, track in pairs:
        if track is not None:
            by_identity[identity].add(track)
            by_track[track].add(identity)
    return {
        "cells": human_cells,
        "cells_with_track": track_cells,
        "identities": len({i for i, _ in pairs}),
        "tracks": len(tracks),
        "idtp": idtp,
        "idfp": track_cells - idtp,
        "idfn": human_cells - idtp,
        "idf1": round(2 * idtp / denominator, 4) if denominator else None,
        "identities_split_across_tracks": sum(1 for t in by_identity.values() if len(t) > 1),
        "tracks_merging_identities": sum(1 for i in by_track.values() if len(i) > 1),
        "cells_without_track": human_cells - track_cells,
    }


def _mean(values: Iterable[float | None]) -> float | None:
    items = [float(v) for v in values if v is not None]
    return round(float(np.mean(items)), 4) if items else None


def _frac_ge(values: Iterable[float | None], threshold: float) -> float | None:
    items = [float(v) for v in values if v is not None]
    return round(sum(1 for v in items if v >= threshold) / len(items), 4) if items else None


def score_arm(
    name: str,
    arm_dir: Path,
    anchors: Sequence[Anchor],
    *,
    frame_offset: int,
    six_view_frames: Sequence[int],
    repository_root: Path,
) -> dict[str, Any]:
    frames = sorted({a.cell["raw_frame"] for a in anchors})
    views = sorted({a.cell["view"] for a in anchors})
    rows = read_rows(Path(arm_dir) / "observations.jsonl", frames=frames, views=views)
    by_frame_view: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_frame_view[(int(row["frame_index"]), str(row["view"]))].append(row)
    worker_runs = arm_worker_runs(Path(arm_dir), repository_root)
    tracks = tracks_at(Path(arm_dir), frames)
    cells = []
    identity_pairs: list[tuple[int, str, str | None]] = []
    for anchor in anchors:
        cell = anchor.cell
        candidates = by_frame_view.get((cell["raw_frame"], cell["view"]), [])
        row = match_arm_row(candidates, anchor)
        arm_mask = None
        if row is not None and row.get("source") in SAM3_SOURCES:
            run = worker_runs.get(cell["view"])
            path = worker_mask_path(run, row, frame_offset) if run is not None else None
            if path is not None and path.is_file():
                arm_mask = clean_mask(read_mask(path))
        scored = score_cell(anchor, row, arm_mask)
        track = None
        if row is not None:
            track = tracks.get(cell["raw_frame"], {}).get((cell["view"], str(row.get("slot"))))
        scored["track_id"] = track
        cells.append(scored)
        if anchor.identity and anchor.state in ("mask", "box"):
            identity_pairs.append((cell["raw_frame"], anchor.identity, track))
    scored_cells = [c for c in cells if c["outcome"] == "scored"]
    counts = Counter(c["outcome"] for c in cells)

    def group(key: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for value in sorted({str(c[key]) for c in scored_cells}):
            subset = [c for c in scored_cells if str(c[key]) == value]
            out[value] = {
                "cells": len(subset),
                "mask_iou_mean": _mean(c["mask_iou"] for c in subset),
                "box_iou_mean": _mean(c["box_iou"] for c in subset),
            }
        return out

    six = [p for p in identity_pairs if p[0] in set(six_view_frames)]
    return {
        "arm": name,
        "arm_dir": fs_common.relative_uri(Path(arm_dir), repository_root),
        "has_masks": bool(worker_runs),
        "cells": cells,
        "counts": dict(counts),
        "cells_scored": len(scored_cells),
        "mask_iou_mean": _mean(c["mask_iou"] for c in scored_cells),
        "mask_iou_frac_ge_0_5": _frac_ge((c["mask_iou"] for c in scored_cells), 0.5),
        "box_iou_mean": _mean(c["box_iou"] for c in scored_cells),
        "run_mask_missing": counts.get("run_mask_missing", 0),
        "hidden_false_positives": counts.get("hidden_false_positive", 0),
        "hidden_false_positive_area": sum(
            c.get("arm_area") or 0 for c in cells if c["outcome"] == "hidden_false_positive"
        ),
        "by_class": group("class"),
        "by_view": group("view"),
        "by_origin": group("origin"),
        "in_cycle_neighbourhood": {
            "cells": sum(1 for c in scored_cells if c["in_cycle_neighbourhood"]),
            "mask_iou_mean": _mean(
                c["mask_iou"] for c in scored_cells if c["in_cycle_neighbourhood"]
            ),
            "box_iou_mean": _mean(
                c["box_iou"] for c in scored_cells if c["in_cycle_neighbourhood"]
            ),
        },
        "identity": {
            "six_view_frame": idf1([(i, t) for _, i, t in six]),
            "all_labelled": idf1([(i, t) for _, i, t in identity_pairs]),
        },
    }


def parse_arms(text: str) -> dict[str, Path]:
    arms: dict[str, Path] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        name, separator, path = item.partition("=")
        if not separator:
            raise ValueError(f"--arms entries are name=path; got {item!r}")
        arms[name.strip()] = Path(path.strip())
    if not arms:
        raise ValueError("--arms names at least one arm")
    return arms


def record_summary(anchors: Sequence[Anchor]) -> dict[str, Any]:
    states = Counter(a.state for a in anchors)
    labelled = sum(v for k, v in states.items() if k != "unlabeled")
    return {
        "cells": len(anchors),
        "labelled": labelled,
        "by_state": dict(states),
        "with_identity": sum(1 for a in anchors if a.identity and a.state in ("mask", "box")),
        "identities": sorted(
            {a.identity for a in anchors if a.identity and a.state in ("mask", "box")}
        ),
    }


def score_arms(
    *,
    workspace_dir: Path,
    record_path: Path | None,
    arms: dict[str, Path],
    repository_root: Path,
) -> dict[str, Any]:
    workspace = load_workspace(workspace_dir)
    config = load_config(repository_root / workspace["config"]["uri"])
    record = load_record(record_path)
    anchors = anchors_from_record(workspace, record, workspace_dir=workspace_dir)
    summary = record_summary(anchors)
    six = config["selection"]["six_view_annotated_frames"]
    offset = int(config["window"]["frame_index_offset"])
    scored = [
        score_arm(
            name,
            path,
            anchors,
            frame_offset=offset,
            six_view_frames=six,
            repository_root=repository_root,
        )
        for name, path in arms.items()
    ]
    return {
        "schema": f"{SCHEMA}/scoreboard",
        "anchor_kind": ANCHOR_KIND,
        "workspace": fs_common.relative_uri(workspace_dir, repository_root),
        "config": workspace["config"],
        "record": None
        if record_path is None or not Path(record_path).is_file()
        else {
            "uri": fs_common.relative_uri(Path(record_path), repository_root),
            "sha256": fs_common.sha256_file(Path(record_path)),
            "author": record.get("author"),
            "reviewed_at": record.get("reviewed_at"),
        },
        "record_summary": summary,
        "arms": scored,
        "generated_at": fs_common.run_timestamp(),
        "claim_boundary": CLAIM_BOUNDARY,
        "licence": LICENCE_NOTE,
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def scoreboard_markdown(report: dict[str, Any]) -> str:
    summary = report["record_summary"]
    lines = [
        f"# Anchor scoreboard: {summary['labelled']} / {summary['cells']} cells labelled",
        "",
        f"States: {json.dumps(summary['by_state'])}; identity given on {summary['with_identity']} "
        f"labelled cells ({len(summary['identities'])} identities).",
        "",
    ]
    if summary["labelled"] == 0:
        lines.append(
            "No cell is labelled yet: every arm scores 0 cells. Copy `decisions.template.json` to "
            "`decisions.json`, fill it, re-run."
        )
        lines.append("")
    header = [
        "arm",
        "cells scored",
        "mask IoU mean",
        "mask IoU >= 0.5",
        "box IoU mean",
        "missing",
        "hidden FP (px)",
        "in-cycle mask IoU",
        "IDF1 six-view frame",
        "IDF1 all labelled",
        "splits / merges / no track",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for arm in report["arms"]:
        six = arm["identity"]["six_view_frame"]
        every = arm["identity"]["all_labelled"]
        lines.append(
            "| "
            + " | ".join(
                [
                    arm["arm"],
                    str(arm["cells_scored"]),
                    _fmt(arm["mask_iou_mean"]),
                    _fmt(arm["mask_iou_frac_ge_0_5"]),
                    _fmt(arm["box_iou_mean"]),
                    str(arm["run_mask_missing"]),
                    f"{arm['hidden_false_positives']} ({arm['hidden_false_positive_area']})",
                    _fmt(arm["in_cycle_neighbourhood"]["mask_iou_mean"]),
                    _fmt(six["idf1"]),
                    _fmt(every["idf1"]),
                    f"{every['identities_split_across_tracks']} / "
                    f"{every['tracks_merging_identities']} / {every['cells_without_track']}",
                ]
            )
            + " |"
        )
    lines.append("")
    classes = sorted({c for arm in report["arms"] for c in arm["by_class"]})
    if classes:
        lines.append("Per class, mask IoU mean (box IoU mean for the boxes-only arm):")
        lines.append("")
        lines.append("| arm | " + " | ".join(classes) + " |")
        lines.append("|" + "---|" * (len(classes) + 1))
        for arm in report["arms"]:
            cells = []
            for cls in classes:
                entry = arm["by_class"].get(cls)
                if entry is None:
                    cells.append("-")
                else:
                    cells.append(
                        _fmt(entry["mask_iou_mean"] if arm["has_masks"] else entry["box_iou_mean"])
                    )
            lines.append(f"| {arm['arm']} | " + " | ".join(cells) + " |")
        lines.append("")
    lines.append(
        "mask IoU = arm mask vs the accepted candidate mask (arms without masks: -); box IoU = the "
        "arm's mask bbox or detector box vs the accepted candidate's bbox (or the reference box "
        "for 'box' accepts); missing = labelled cell without any arm row; hidden FP = an arm mask "
        "on a cell marked hidden; IDF1 = identity F1 between the human's instance_identity and "
        "the tracker's ids over the labelled cells (maximum one-to-one matching), on the six-view "
        "frame and pooled over every labelled frame; splits = identities seen under more than one "
        "track "
        "id, merges = track ids covering more than one identity. " + report["claim_boundary"]
    )
    lines.append("")
    return "\n".join(lines)


def run_score(args: argparse.Namespace) -> dict[str, Any]:
    root = Path.cwd().resolve()
    arms = parse_arms(args.arms)
    report = score_arms(
        workspace_dir=Path(args.workspace),
        record_path=Path(args.record) if args.record else None,
        arms=arms,
        repository_root=root,
    )
    output = Path(args.output) if args.output else Path(args.workspace) / "scoreboard"
    output.mkdir(parents=True, exist_ok=True)
    fs_common.write_json(output / "anchor_scoreboard.json", report)
    table = scoreboard_markdown(report)
    (output / "anchor_scoreboard.md").write_text(table, encoding="utf-8")
    print(table, end="")
    print(f"Report: {output / 'anchor_scoreboard.json'}")
    return report


# --------------------------------------------------------------------------------------------
# export: the committed human record (no pixels)


def export_record(
    *, workspace_dir: Path, record_path: Path, output: Path, repository_root: Path
) -> dict[str, Any]:
    workspace = load_workspace(workspace_dir)
    record = load_record(record_path)
    anchors = anchors_from_record(workspace, record, workspace_dir=workspace_dir)
    entries = []
    for anchor in anchors:
        cell = anchor.cell
        sha = None
        if anchor.state == "mask":
            candidate = next(c for c in cell["candidates"] if c["index"] == anchor.candidate_index)
            sha = fs_common.sha256_file(workspace_dir / candidate["mask_uri"])
        entries.append(
            {
                "raw_frame": cell["raw_frame"],
                "view": cell["view"],
                "slot": cell["slot"],
                "label": cell["label"],
                "class": cell["class"],
                "origin": cell["origin"],
                "state": anchor.state,
                "candidate_index": anchor.candidate_index,
                "candidate_kind": (
                    next(
                        c["kind"]
                        for c in cell["candidates"]
                        if c["index"] == anchor.candidate_index
                    )
                    if anchor.candidate_index is not None
                    else None
                ),
                "mask_sha256": sha,
                "area_pixels": int(anchor.mask.sum()) if anchor.mask is not None else None,
                "instance_identity": anchor.identity,
            }
        )
    doc = {
        "schema": f"{SCHEMA}/human-record",
        "anchor_kind": ANCHOR_KIND,
        "author": record.get("author"),
        "reviewed_at": record.get("reviewed_at"),
        "config": workspace["config"],
        "workspace": fs_common.relative_uri(workspace_dir, repository_root),
        "decisions": {
            "uri": fs_common.relative_uri(Path(record_path), repository_root),
            "sha256": fs_common.sha256_file(Path(record_path)),
        },
        "counts": record_summary(anchors),
        "anchors": entries,
        "claim_boundary": CLAIM_BOUNDARY,
        "licence": LICENCE_NOTE,
        "notes": (
            "Skeleton written by battle-finebio-anchors export; author and reviewed_at come from "
            "the decisions file and stay null until the human fills them in. Masks stay under "
            "runs/."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    fs_common.write_json(output, doc)
    return doc


# --------------------------------------------------------------------------------------------
# CLI


def _add_gpu_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--external-python", type=Path, default=MUGGLED_SAM_PYTHON)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--allow-gpu-neighbour", type=int, action="append", default=[], metavar="PID"
    )
    parser.add_argument(
        "--gpu-guard", choices=gpu_guard.GUARD_MODES, default=gpu_guard.DEFAULT_GUARD_MODE
    )
    parser.add_argument(
        "--gpu-guard-profile", choices=tuple(gpu_guard.EXPECTED_PEAK_BYTES), default="sam3_1280"
    )
    parser.add_argument("--expected-peak-vram-bytes", type=int, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Human review anchors for a FineBio trial: frame choice, candidate workspace, "
            "scoreboard."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sel = sub.add_parser("select", help="Choose the anchor frames and write the anchor config.")
    sel.add_argument(
        "--clip", type=Path, required=True, help="configs/clips/finebio_<trial>_<window>.json"
    )
    sel.add_argument(
        "--arm-b", type=Path, required=True, help="arm (b) directory with observations.jsonl"
    )
    sel.add_argument("--seeds", type=Path, required=True, help="the seeds.json the arms ran from")
    sel.add_argument("--fixed-view", default="auto", help="fixed view beside the fpv, or 'auto'")
    sel.add_argument("--fixed-view-reason", default=None)
    sel.add_argument("--output", type=Path, required=True)
    sel.add_argument("--spacing", type=int, default=DEFAULT_SPACING)
    sel.add_argument("--disagreement-frames", type=int, default=DEFAULT_DISAGREEMENT_FRAMES)
    sel.add_argument("--random-frames", type=int, default=DEFAULT_RANDOM_FRAMES)
    sel.add_argument("--min-in-cycles", type=int, default=DEFAULT_MIN_IN_CYCLES)
    sel.add_argument("--cycle-pad", type=int, default=DEFAULT_CYCLE_PAD)
    sel.add_argument("--seed", type=int, default=DEFAULT_SEED)
    sel.add_argument("--plate-visible-fraction", type=float, default=DEFAULT_PLATE_VISIBLE_FRACTION)

    ws = sub.add_parser(
        "workspace", help="Decode the candidates (GPU) and write sheets + template."
    )
    ws.add_argument("--config", type=Path, required=True)
    ws.add_argument("--arm-b", type=Path, required=True)
    ws.add_argument("--output", type=Path, required=True)
    ws.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    ws.add_argument("--skip-decode", action="store_true", help="candidate 0 only, no GPU")
    _add_gpu_arguments(ws)

    worker = sub.add_parser("decode-worker", help="internal: runs under the MuggledSAM interpreter")
    worker.add_argument("--requests", type=Path, required=True)
    worker.add_argument("--results", type=Path, required=True)
    worker.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    worker.add_argument(
        "--allow-gpu-neighbour", type=int, action="append", default=[], metavar="PID"
    )
    worker.add_argument(
        "--gpu-guard", choices=gpu_guard.GUARD_MODES, default=gpu_guard.DEFAULT_GUARD_MODE
    )
    worker.add_argument(
        "--gpu-guard-profile", choices=tuple(gpu_guard.EXPECTED_PEAK_BYTES), default="sam3_1280"
    )
    worker.add_argument("--expected-peak-vram-bytes", type=int, default=None)

    sc = sub.add_parser(
        "score", help="Score arms against the human record (empty or partial is fine)."
    )
    sc.add_argument("--workspace", type=Path, required=True)
    sc.add_argument(
        "--record", type=Path, default=None, help="decisions.json (missing = nothing labelled)"
    )
    sc.add_argument("--arms", required=True, help="name=path,name=path,... arm directories")
    sc.add_argument("--output", type=Path, default=None, help="default <workspace>/scoreboard")

    ex = sub.add_parser("export", help="Write the committed human-record skeleton (no pixels).")
    ex.add_argument("--workspace", type=Path, required=True)
    ex.add_argument("--record", type=Path, required=True)
    ex.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "decode-worker":
        return decode_worker_main(args)
    try:
        if args.command == "select":
            run_select(args)
        elif args.command == "workspace":
            workspace = run_workspace(args)
            decode = workspace.get("decode") or {}
            if decode.get("state") not in ("succeeded", "skipped"):
                return 3
        elif args.command == "score":
            run_score(args)
        elif args.command == "export":
            doc = export_record(
                workspace_dir=Path(args.workspace),
                record_path=Path(args.record),
                output=Path(args.output),
                repository_root=Path.cwd().resolve(),
            )
            print(f"Human record: {args.output} ({json.dumps(doc['counts']['by_state'])})")
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
