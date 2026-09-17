"""CPU-side crop construction, score fusion, and monotonic alignment for fine substeps."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .fine_substep_contract import (
    ANALYSIS_FPS,
    FRAME_COUNT,
    FineSubstepAgentLabelContract,
    FineSubstepDefinition,
    substep_for_frame,
)
from .schemas import FrameObservations, NormalizedBox

HandCueSource = Literal["wilor", "mediapipe", "missing"]
PART_TARGETS = ("chassis", "interior", "rear_body", "cabin")
DIMENSIONS = (1280, 720)
CROP_MARGIN = 0.08
MIN_CROP_SIZE = 0.18


@dataclass(frozen=True)
class PixelBox:
    x0: int
    y0: int
    x1: int
    y1: int

    def as_dict(self) -> dict[str, int]:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}


@dataclass(frozen=True)
class CropSample:
    analysis_frame_index: int
    source_seconds: float
    hand_cue_source: HandCueSource
    workspace_box: PixelBox
    wilor_available: bool
    mediapipe_fallback_used: bool
    part_boxes_present: tuple[str, ...]
    contact_proximity: float | None
    contact_state: Literal["observed", "missing_hand", "missing_mask"]


def _pixel_box(box: NormalizedBox, dimensions: tuple[int, int]) -> PixelBox:
    width, height = dimensions
    x0 = max(0, min(width - 1, round(box.x * (width - 1))))
    y0 = max(0, min(height - 1, round(box.y * (height - 1))))
    x1 = max(
        x0 + 1,
        min(width, round((box.x + box.width) * (width - 1)) + 1),
    )
    y1 = max(
        y0 + 1,
        min(height, round((box.y + box.height) * (height - 1)) + 1),
    )
    return PixelBox(x0=x0, y0=y0, x1=x1, y1=y1)


def _union_boxes(boxes: list[PixelBox], dimensions: tuple[int, int]) -> PixelBox | None:
    if not boxes:
        return None
    width, height = dimensions
    x0 = max(0, min(box.x0 for box in boxes))
    y0 = max(0, min(box.y0 for box in boxes))
    x1 = min(width, max(box.x1 for box in boxes))
    y1 = min(height, max(box.y1 for box in boxes))
    if x1 <= x0 or y1 <= y0:
        return None
    return PixelBox(x0=x0, y0=y0, x1=x1, y1=y1)


def _expand_box(box: PixelBox, dimensions: tuple[int, int]) -> PixelBox:
    width, height = dimensions
    box_w = box.x1 - box.x0
    box_h = box.y1 - box.y0
    min_w = max(int(width * MIN_CROP_SIZE), box_w)
    min_h = max(int(height * MIN_CROP_SIZE), box_h)
    cx = (box.x0 + box.x1) / 2
    cy = (box.y0 + box.y1) / 2
    half_w = max(box_w / 2 * (1 + CROP_MARGIN), min_w / 2)
    half_h = max(box_h / 2 * (1 + CROP_MARGIN), min_h / 2)
    x0 = max(0, int(round(cx - half_w)))
    y0 = max(0, int(round(cy - half_h)))
    x1 = min(width, int(round(cx + half_w)))
    y1 = min(height, int(round(cy + half_h)))
    return PixelBox(x0=x0, y0=y0, x1=x1, y1=y1)


def _hands_from_observation(observation: FrameObservations) -> tuple[object, ...]:
    return observation.hands


def _part_boxes(
    observation: FrameObservations,
    *,
    active_parts: tuple[str, ...],
    dimensions: tuple[int, int],
) -> list[tuple[str, PixelBox]]:
    boxes: list[tuple[str, PixelBox]] = []
    for obj in observation.objects:
        if obj.label in active_parts:
            boxes.append((obj.label, _pixel_box(obj.box, dimensions)))
    return boxes


def _wrist_boxes(hands: tuple[object, ...], dimensions: tuple[int, int]) -> list[PixelBox]:
    boxes: list[PixelBox] = []
    for hand in hands:
        boxes.append(_pixel_box(hand.box, dimensions))
    return boxes


def _contact_proximity(
    hands: tuple[object, ...],
    observation: FrameObservations,
    dimensions: tuple[int, int],
) -> tuple[float | None, Literal["observed", "missing_hand", "missing_mask"]]:
    if not hands:
        return None, "missing_hand"
    active = _part_boxes(observation, active_parts=("chassis", "interior"), dimensions=dimensions)
    if not active:
        return None, "missing_mask"
    width, height = dimensions
    min_distance = float("inf")
    for hand in hands:
        wrist = hand.landmarks[0]
        px = min(width - 1, max(0, round(wrist.x * (width - 1))))
        py = min(height - 1, max(0, round(wrist.y * (height - 1))))
        for _, box in active:
            dx = max(box.x0 - px, 0, px - (box.x1 - 1))
            dy = max(box.y0 - py, 0, py - (box.y1 - 1))
            min_distance = min(min_distance, float((dx * dx + dy * dy) ** 0.5))
    if min_distance == float("inf"):
        return None, "missing_mask"
    return min_distance, "observed"


def build_crop_sample(
    frame_index: int,
    *,
    wilor: FrameObservations | None,
    mediapipe: FrameObservations | None,
    parts: FrameObservations | None,
    dimensions: tuple[int, int] = DIMENSIONS,
) -> CropSample:
    source_seconds = frame_index / ANALYSIS_FPS + 294.0
    wilor_hands = _hands_from_observation(wilor) if wilor is not None else ()
    mp_hands = _hands_from_observation(mediapipe) if mediapipe is not None else ()
    hand_source: HandCueSource
    hands: tuple[object, ...]
    fallback = False
    if wilor_hands:
        hands = wilor_hands
        hand_source = "wilor"
    elif mp_hands:
        hands = mp_hands
        hand_source = "mediapipe"
        fallback = True
    else:
        hands = ()
        hand_source = "missing"

    part_boxes = (
        _part_boxes(parts, active_parts=PART_TARGETS, dimensions=dimensions)
        if parts is not None
        else []
    )
    boxes = _wrist_boxes(hands, dimensions) + [box for _, box in part_boxes]
    union = _union_boxes(boxes, dimensions)
    if union is None:
        union = PixelBox(x0=640, y0=300, x1=1180, y1=680)
    workspace = _expand_box(union, dimensions)
    proximity, contact_state = (
        _contact_proximity(hands, parts, dimensions)
        if parts is not None
        else (None, "missing_mask")
    )
    return CropSample(
        analysis_frame_index=frame_index,
        source_seconds=source_seconds,
        hand_cue_source=hand_source,
        workspace_box=workspace,
        wilor_available=wilor is not None and bool(wilor_hands),
        mediapipe_fallback_used=fallback,
        part_boxes_present=tuple(label for label, _ in part_boxes),
        contact_proximity=proximity,
        contact_state=contact_state,
    )


def sample_frame_indices(*, sample_fps: float, frame_count: int = FRAME_COUNT) -> list[int]:
    stride = max(int(round(ANALYSIS_FPS / sample_fps)), 1)
    return list(range(0, frame_count, stride))


def load_observations(path: Path) -> dict[int, FrameObservations]:
    observations: dict[int, FrameObservations] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obs = FrameObservations.model_validate_json(line)
        observations[obs.analysis_frame_index] = obs
    return observations


def build_crop_manifest(
    frame_indices: list[int],
    *,
    wilor: dict[int, FrameObservations],
    mediapipe: dict[int, FrameObservations],
    parts: dict[int, FrameObservations],
) -> list[CropSample]:
    return [
        build_crop_sample(
            frame_index,
            wilor=wilor.get(frame_index),
            mediapipe=mediapipe.get(frame_index),
            parts=parts.get(frame_index),
        )
        for frame_index in frame_indices
    ]


def temporal_delta_features(embeddings: np.ndarray) -> np.ndarray:
    """Return per-sample motion magnitude in embedding space (first row is zero)."""
    if embeddings.shape[0] == 0:
        return embeddings[:, :0]
    deltas = np.zeros(embeddings.shape[0], dtype=np.float32)
    if embeddings.shape[0] > 1:
        diffs = embeddings[1:] - embeddings[:-1]
        deltas[1:] = np.linalg.norm(diffs, axis=1)
    max_delta = float(deltas.max()) if deltas.size else 0.0
    if max_delta <= 1e-8:
        return deltas
    return deltas / max_delta


def fuse_scores(
    clip_scores: np.ndarray,
    *,
    motion: np.ndarray,
    contact: np.ndarray,
    weights: dict[str, float],
) -> np.ndarray:
    """Combine CLIP max-prompt scores with weak motion/contact terms."""
    fused = clip_scores.copy()
    motion_term = motion[:, None] * weights.get("motion", 0.0)
    contact_term = contact[:, None] * weights.get("contact", 0.0)
    return fused + motion_term + contact_term


def contact_bonus(contact_proximity: float | None, *, threshold_px: float = 24.0) -> float:
    if contact_proximity is None:
        return 0.0
    if contact_proximity <= threshold_px:
        return 1.0 - (contact_proximity / threshold_px)
    return 0.0


def monotonic_substep_dp(
    scores: np.ndarray,
    substep_count: int,
    *,
    stay_penalty: float = 0.02,
    advance_bonus: float = 0.0,
) -> tuple[list[int], float]:
    """Assign each sample to a monotonically non-decreasing substep index."""
    sample_count = scores.shape[0]
    if sample_count == 0:
        return [], 0.0
    dp = np.full((sample_count, substep_count), -np.inf, dtype=np.float64)
    back = np.full((sample_count, substep_count), -1, dtype=np.int32)
    dp[0, 0] = scores[0, 0]
    for step in range(1, substep_count):
        if scores[0, step] > dp[0, step - 1] - stay_penalty + advance_bonus:
            dp[0, step] = scores[0, step]
            back[0, step] = step - 1
    for sample in range(1, sample_count):
        for step in range(substep_count):
            candidates = [(dp[sample - 1, step] - stay_penalty, step)]
            if step > 0:
                candidates.append((dp[sample - 1, step - 1] + advance_bonus, step - 1))
            best_prev, prev_step = max(candidates, key=lambda item: item[0])
            dp[sample, step] = best_prev + scores[sample, step]
            back[sample, step] = prev_step
    last_step = int(np.argmax(dp[-1]))
    total = float(dp[-1, last_step])
    path = [last_step]
    for sample in range(sample_count - 1, 0, -1):
        last_step = int(back[sample, last_step])
        path.append(last_step)
    path.reverse()
    return path, total


def labels_to_intervals(
    contract: FineSubstepAgentLabelContract,
    sample_frames: list[int],
    substep_indices: list[int],
) -> list[dict[str, object]]:
    substeps = contract.substeps
    intervals: list[dict[str, object]] = []
    for step_index, substep in enumerate(substeps):
        matched = [
            sample_frames[sample_index]
            for sample_index, assigned in enumerate(substep_indices)
            if assigned == step_index
        ]
        intervals.append(
            {
                "substep_id": substep.substep_id,
                "label": substep.label,
                "agent_start_frame": substep.start_frame,
                "agent_end_frame_exclusive": substep.end_frame_exclusive,
                "matched_analysis_frames": matched,
                "matched_seconds": [round(frame / ANALYSIS_FPS, 3) for frame in matched],
            }
        )
    return intervals


def predicted_substep_at_frame(
    contract: FineSubstepAgentLabelContract,
    sample_frames: list[int],
    substep_indices: list[int],
    frame_index: int,
) -> FineSubstepDefinition:
    eligible = [
        (sample_frames[index], substep_indices[index])
        for index in range(len(sample_frames))
        if sample_frames[index] <= frame_index
    ]
    if not eligible:
        return contract.substeps[substep_indices[0]]
    _, step_index = eligible[-1]
    return contract.substeps[step_index]


def boundary_frames_from_assignment(
    contract: FineSubstepAgentLabelContract,
    sample_frames: list[int],
    substep_indices: list[int],
) -> list[int]:
    boundaries = [contract.substeps[0].start_frame]
    for sample_index in range(1, len(sample_frames)):
        if substep_indices[sample_index] != substep_indices[sample_index - 1]:
            boundaries.append(sample_frames[sample_index])
    boundaries.append(contract.substeps[-1].end_frame_exclusive)
    return boundaries


def evaluate_against_agent_labels(
    contract: FineSubstepAgentLabelContract,
    *,
    sample_frames: list[int],
    substep_indices: list[int],
    alignment_name: str,
) -> dict[str, object]:
    predicted_boundaries = boundary_frames_from_assignment(contract, sample_frames, substep_indices)
    agent_boundaries = [substep.start_frame for substep in contract.substeps[1:]] + [FRAME_COUNT]
    offsets = [
        predicted - agent
        for predicted, agent in zip(predicted_boundaries, agent_boundaries, strict=False)
    ]
    checkpoint_rows = []
    for checkpoint in contract.checkpoints:
        agent = substep_for_frame(contract, checkpoint.analysis_frame_index)
        predicted = predicted_substep_at_frame(
            contract, sample_frames, substep_indices, checkpoint.analysis_frame_index
        )
        checkpoint_rows.append(
            {
                "analysis_frame_index": checkpoint.analysis_frame_index,
                "source_seconds": checkpoint.source_seconds,
                "note": checkpoint.note,
                "agent_substep_id": agent.substep_id,
                "agent_label": agent.label,
                "predicted_substep_id": predicted.substep_id,
                "predicted_label": predicted.label,
                "match": agent.substep_id == predicted.substep_id,
            }
        )
    recovered = sum(1 for row in checkpoint_rows if row["match"])
    collapsed = len({substep_indices[index] for index in range(len(substep_indices))}) < len(
        contract.substeps
    ) // 2
    return {
        "alignment_name": alignment_name,
        "predicted_boundaries": predicted_boundaries,
        "agent_boundaries": agent_boundaries,
        "boundary_offsets_frames": offsets,
        "mean_abs_boundary_offset_frames": float(np.mean(np.abs(offsets))) if offsets else 0.0,
        "recovered_substep_count": len(
            {substep_indices[index] for index in range(len(substep_indices))}
        ),
        "target_substep_count": len(contract.substeps),
        "collapsed": collapsed,
        "checkpoint_accuracy": recovered / len(checkpoint_rows) if checkpoint_rows else 0.0,
        "checkpoints": checkpoint_rows,
    }


def write_crop_manifest(path: Path, samples: list[CropSample]) -> None:
    payload = [
        {
            "analysis_frame_index": sample.analysis_frame_index,
            "source_seconds": sample.source_seconds,
            "hand_cue_source": sample.hand_cue_source,
            "workspace_box": sample.workspace_box.as_dict(),
            "wilor_available": sample.wilor_available,
            "mediapipe_fallback_used": sample.mediapipe_fallback_used,
            "part_boxes_present": list(sample.part_boxes_present),
            "contact_proximity": sample.contact_proximity,
            "contact_state": sample.contact_state,
        }
        for sample in samples
    ]
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
