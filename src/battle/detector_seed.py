"""`battle-detector-seed`: per-view SAM3 slots from FineBio detector boxes (plan `p2-seeds`).

The detector (`battle-finebio-detect`) gives class and box per view and frame; this tool
decides **which objects get a SAM3 slot in which view**, where each slot starts, and whether
the SAM3 image decoder accepts the detector box as a seed. Nothing here uses ground truth or a
human; the human's part is gate 1 (`sheets`, `docs/labeling-sessions-2026-09-25-finebio.md`),
applied afterwards as a filter (`apply-decisions`).

Steps (one run directory, each step reads the previous one's files)::

    battle-detector-seed instances --detections <dir> --views fpv,T1,T2,T3,T4,T5 \\
        --start 600 --end 4200 --output runs/<run>
    battle-detector-seed select --output runs/<run>
    battle-detector-seed decode --output runs/<run> --trial P03_03_01        # GPU, ~1 min
    battle-detector-seed sheets --output runs/<run> --trial P03_03_01
    battle-detector-seed apply-decisions --output runs/<run> --decisions runs/<run>/decisions.json

* ``instances``: per view, same-class boxes on consecutive detected frames are associated by
  IoU >= 0.3 (greedy one-to-one, a gap of up to ``--max-gap`` frames allowed) into instance
  tracklets with per-frame boxes and scores; rows the detector's CPU fallback interpolated
  are carried but flagged. Hands are tracked too, as probes, never as slots.
* ``select``: the plan's seed rule. A slot opens for a persistent instance (>= N frames
  detected at score >= 0.3) that **moves** (smoothed box-centre displacement over a window,
  ego-motion compensated by the median displacement of the other instances, > 20 px at 1920
  at least once) or **sits inside a hand box** (centre inside `left_hand` / `right_hand` on
  > 10% of its frames); the **named containers** (centrifuge, vortex mixer, PCR machine, the
  racks) open once as static volumes (`role: container`); identical-instance classes with
  >= 3 members in one rack become one **group** slot per rack unless a member moves, in which
  case that member is its own slot; everything else stays `detector_only`. A per-view cap
  keeps moving / in-hand objects first (by persistence, then movement), then containers, then
  groups. Every slot has a start frame (the first detected frame of a dense run) and a label
  ``<class>#<k>``; the slot set is enumerated over the whole window because the video-memory
  worker needs the full multiplex set at frame 0.
* ``decode``: for each slot's seed frame the tight detector box is the only prompt to the
  SAM3 image decoder (MuggledSAM `sam3.1_multiplex.pt`, encoder side 1280, one image encode
  per distinct seed frame per view; other slots' centres as negatives optional); the
  decoder's top-IoU candidate is **accepted by mask-bbox IoU vs the box >= 0.6** (the plate's
  fill ratio is 0.5 when correct, so fill is not the test); a rejected tight box is retried
  with a 0.15 margin, then on the next persistent frames (3 attempts), else the slot is
  ``unseeded`` with its reason. Writes ``seeds.json`` (provenance `selected_by: detector`,
  `provenance: auto`, class, score, frame, box, rule, candidate scores, decode IoU) and the
  worker-ready files: a mode-(i) box stream per view (every frame from each selected slot's
  start, boxes from the tracklets) and a mode-(ii) schedule per view (accepted seeds with
  ``prompt_box_xyxy_px``, ``start_frame``, ``initial_multiplex_slot``), analysis frame 0 =
  the window start.
* ``sheets``: per-view contact sheets of the accepted seeds at their seed frames (mask,
  box, label, rule; class-family colours) and ``decisions.template.json`` for gate 1.

Detector scores are the detector's own; IoU between a SAM3 mask's bounding box and the
detector box is agreement between two models, not accuracy. FineBio is non-commercial
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
from collections import defaultdict
from dataclasses import asdict, dataclass, field
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

SCHEMA = "battle-detector-seed/1"
VIEWS: tuple[str, ...] = ("fpv", "T1", "T2", "T3", "T4", "T5")
FPV_VIEW = "fpv"
HAND_CLASSES: tuple[str, ...] = ("left_hand", "right_hand")
# Static volumes the tracker needs (plan: centrifuge, racks, vortex, pcr_machine).
CONTAINER_CLASSES: tuple[str, ...] = (
    "centrifuge",
    "vortex_mixer",
    "pcr_machine",
    "magnetic_rack",
    "micro_tube_rack",
    "50ml_tube_rack",
    "15ml_tube_rack",
    "8_tube_stripes_rack",
    "blue_tip_rack",
    "yellow_tip_rack",
    "red_tip_rack",
    "8_channel_tip_rack",
)
# Racks that hold identical instances; members that never move are one group slot per rack.
RACK_CLASSES: tuple[str, ...] = (
    "micro_tube_rack",
    "50ml_tube_rack",
    "15ml_tube_rack",
    "8_tube_stripes_rack",
    "blue_tip_rack",
    "yellow_tip_rack",
    "red_tip_rack",
    "8_channel_tip_rack",
)
GROUP_CLASSES: tuple[str, ...] = (
    "micro_tube",
    "50ml_tube",
    "15ml_tube",
    "8_tube_stripes",
    "blue_tip",
    "yellow_tip",
    "red_tip",
    "8_channel_tip",
    "tube_with_spin_column",
    "spin_column",
    "tube_without_lid",
)
RULES = ("moves", "in_hand", "container", "group")
ROLES = ("object", "container", "group")
REFERENCE_WIDTH = 1920
DETECTOR_SOURCE = "finebio_dino"
MUGGLED_SAM_SOURCE = Path("/home/nick/src/muggled_sam")
MUGGLED_SAM_PYTHON = Path("/home/nick/.pyenv/versions/muggled_sam/bin/python")
DEFAULT_MODEL = MUGGLED_SAM_SOURCE / "model_weights" / "sam3.1_multiplex.pt"
DEFAULT_RAW_ROOT = Path("data/raw/finebio")
ENCODER_SIDE = 1280
LICENCE_NOTE = (
    "FineBio is licensed for non-commercial research; frames, masks and sheets derived from "
    "it stay under runs/ and are never committed or redistributed"
)
CLAIM_BOUNDARY = (
    "Slots are opened by rules over FineBio DINO boxes (the detector's own scores); seed "
    "acceptance is the IoU between a SAM3 mask's bounding box and the detector box, i.e. "
    "agreement between two models. No human and no ground truth decided anything here; the "
    "human's accept/reject (gate 1) is applied afterwards as a filter."
)

Box = tuple[float, float, float, float]


# --------------------------------------------------------------------------------------------
# parameters


@dataclass
class SeedParams:
    assoc_iou: float = 0.3
    assoc_min_score: float = 0.2
    max_gap: int = 5
    min_score: float = 0.3
    min_persistence: int = 15
    move_px: float = 20.0
    move_px_fpv: float = 40.0
    move_window: int = 30
    smooth_frames: int = 5
    hand_fraction: float = 0.10
    rack_fraction: float = 0.5
    group_min_members: int = 3
    duplicate_containment: float = 0.7
    duplicate_fraction: float = 0.5
    same_box_iou: float = 0.8
    fpv_corroboration: bool = True
    # Displacement is measured against the scene's own motion only where the camera moves.
    scene_motion_views: tuple[str, ...] = (FPV_VIEW,)
    slot_cap: int = 10
    landmark_classes: tuple[str, ...] = ("centrifuge", "vortex_mixer", "pcr_machine")
    seed_attempts: int = 3
    accept_iou: float = 0.6
    decode_margin: float = 0.15
    other_instances_as_negatives: bool = False
    container_classes: tuple[str, ...] = CONTAINER_CLASSES
    rack_classes: tuple[str, ...] = RACK_CLASSES
    group_classes: tuple[str, ...] = GROUP_CLASSES
    hand_classes: tuple[str, ...] = HAND_CLASSES

    def move_threshold_px(self, view: str, image_width: int) -> float:
        base = self.move_px_fpv if view == FPV_VIEW else self.move_px
        return base * image_width / REFERENCE_WIDTH


# --------------------------------------------------------------------------------------------
# instances (IoU association)


@dataclass
class Instance:
    instance: str
    view: str
    object_class: str
    frames: list[int] = field(default_factory=list)
    boxes: list[Box] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    interpolated: list[bool] = field(default_factory=list)

    @property
    def first_frame(self) -> int:
        return self.frames[0]

    @property
    def last_frame(self) -> int:
        return self.frames[-1]

    def box_at(self) -> dict[int, Box]:
        return dict(zip(self.frames, self.boxes))

    def to_record(self) -> dict[str, Any]:
        return {
            "instance": self.instance,
            "view": self.view,
            "class": self.object_class,
            "frames": self.frames,
            "boxes": [[round(v, 1) for v in box] for box in self.boxes],
            "scores": [round(s, 4) for s in self.scores],
            "interpolated": self.interpolated,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Instance:
        return cls(
            instance=record["instance"],
            view=record["view"],
            object_class=record["class"],
            frames=[int(f) for f in record["frames"]],
            boxes=[tuple(float(v) for v in box) for box in record["boxes"]],
            scores=[float(s) for s in record["scores"]],
            interpolated=[bool(v) for v in record["interpolated"]],
        )


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every row of `a` (n x 4) and every row of `b` (m x 4), xyxy."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]))
    ix0 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy0 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix1 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def greedy_pairs(ious: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """One-to-one (row, column) pairs by descending IoU, IoU >= threshold."""
    pairs: list[tuple[int, int]] = []
    if ious.size == 0:
        return pairs
    order = np.argsort(-ious, axis=None, kind="stable")
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    for flat in order:
        row, col = divmod(int(flat), ious.shape[1])
        if ious[row, col] < threshold:
            break
        if row in used_rows or col in used_cols:
            continue
        used_rows.add(row)
        used_cols.add(col)
        pairs.append((row, col))
    return pairs


def read_detection_records(path: Path, start: int | None, end: int | None) -> list[dict]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            frame = int(record["frame_index"])
            if (start is not None and frame < start) or (end is not None and frame >= end):
                continue
            records.append(record)
    records.sort(key=lambda r: int(r["frame_index"]))
    return records


def associate_instances(
    view: str, records: list[dict[str, Any]], params: SeedParams
) -> list[Instance]:
    """Same-class IoU association of one view's per-frame detections into tracklets.

    A detection at ``score >= assoc_min_score`` joins the same-class tracklet whose last box
    overlaps it most (IoU >= ``assoc_iou``, greedy one-to-one); a tracklet unmatched for more
    than ``max_gap`` frames is closed. Every class is tracked, hands included (probes).
    """
    active: dict[str, list[Instance]] = defaultdict(list)
    done: list[Instance] = []
    counter: dict[str, int] = defaultdict(int)
    for record in records:
        frame = int(record["frame_index"])
        interpolated = bool(record.get("interpolated", False))
        by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for det in record["detections"]:
            if float(det["score"]) >= params.assoc_min_score:
                by_class[str(det["class"])].append(det)
        for cls, tracklets in list(active.items()):
            alive = [t for t in tracklets if frame - t.last_frame <= params.max_gap + 1]
            done.extend(t for t in tracklets if frame - t.last_frame > params.max_gap + 1)
            active[cls] = alive
        for cls, dets in by_class.items():
            alive = active[cls]
            boxes = np.array([d["box_xyxy_px"] for d in dets], dtype=float)
            last = np.array([t.boxes[-1] for t in alive], dtype=float).reshape(-1, 4)
            pairs = greedy_pairs(iou_matrix(last, boxes), params.assoc_iou)
            matched_dets: set[int] = set()
            for row, col in pairs:
                det = dets[col]
                alive[row].frames.append(frame)
                alive[row].boxes.append(tuple(float(v) for v in det["box_xyxy_px"]))
                alive[row].scores.append(float(det["score"]))
                alive[row].interpolated.append(interpolated)
                matched_dets.add(col)
            for col, det in enumerate(dets):
                if col in matched_dets:
                    continue
                index = counter[cls]
                counter[cls] += 1
                alive.append(
                    Instance(
                        instance=f"{view}/{cls}/{index}",
                        view=view,
                        object_class=cls,
                        frames=[frame],
                        boxes=[tuple(float(v) for v in det["box_xyxy_px"])],
                        scores=[float(det["score"])],
                        interpolated=[interpolated],
                    )
                )
    for tracklets in active.values():
        done.extend(tracklets)
    done.sort(key=lambda t: (t.first_frame, t.object_class, t.instance))
    return done


def lifetimes_summary(instances: list[Instance]) -> dict[str, dict[str, Any]]:
    per_class: dict[str, list[int]] = defaultdict(list)
    for inst in instances:
        per_class[inst.object_class].append(inst.last_frame - inst.first_frame + 1)
    out = {}
    for cls, spans in sorted(per_class.items()):
        arr = np.asarray(spans)
        out[cls] = {
            "instances": int(arr.size),
            "lifetime_frames": {
                "min": int(arr.min()),
                "median": float(np.median(arr)),
                "max": int(arr.max()),
            },
        }
    return out


def write_instances(
    output: Path,
    view: str,
    instances: list[Instance],
    *,
    image_hw: tuple[int, int],
    frames: list[int],
) -> Path:
    directory = output / "instances"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{view}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "meta": True,
                    "view": view,
                    "image_hw": list(image_hw),
                    "frame_first": frames[0] if frames else None,
                    "frame_last": frames[-1] if frames else None,
                    "frame_count": len(frames),
                    "instances": len(instances),
                }
            )
            + "\n"
        )
        for inst in instances:
            handle.write(json.dumps(inst.to_record(), separators=(",", ":")) + "\n")
    return path


def read_instances(output: Path, view: str) -> tuple[dict[str, Any], list[Instance]]:
    path = Path(output) / "instances" / f"{view}.jsonl"
    meta: dict[str, Any] = {}
    instances: list[Instance] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("meta"):
                meta = record
            else:
                instances.append(Instance.from_record(record))
    return meta, instances


def run_instances(args: argparse.Namespace) -> dict[str, Any]:
    params = params_from_args(args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    views = [v for v in str(args.views).split(",") if v]
    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "step": "instances",
        "detections": str(Path(args.detections).resolve()),
        "views": {},
        "window": {"start": args.start, "end": args.end},
        "params": asdict(params),
    }
    manifest_path = Path(args.detections) / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary["detector"] = {
            "state": manifest.get("state"),
            "coverage": manifest.get("detection_coverage"),
            "model": (manifest.get("model") or {}).get("name") or manifest.get("model"),
            "trial": manifest.get("trial"),
        }
    started = time.perf_counter()
    for view in views:
        records = read_detection_records(
            Path(args.detections) / f"{view}.jsonl", args.start, args.end
        )
        if not records:
            raise ValueError(f"{view}: no detection rows in the window")
        instances = associate_instances(view, records, params)
        frames = [int(r["frame_index"]) for r in records]
        image_hw = tuple(int(v) for v in records[0]["image_hw"])
        write_instances(output, view, instances, image_hw=image_hw, frames=frames)
        summary["views"][view] = {
            "image_hw": list(image_hw),
            "frames": len(frames),
            "frame_first": frames[0],
            "frame_last": frames[-1],
            "interpolated_frames": sum(bool(r.get("interpolated")) for r in records),
            "instances": len(instances),
            "per_class": lifetimes_summary(instances),
        }
        print(f"{view}: {len(frames)} frames, {len(instances)} instances", flush=True)
    summary["elapsed_seconds"] = time.perf_counter() - started
    fs_common.write_json(output / "instances" / "summary.json", summary)
    return summary


# --------------------------------------------------------------------------------------------
# select (the seed rule)


def centre(box: Box) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def inside(point: tuple[float, float], box: Box) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def box_clipped(box: Box, image_hw: tuple[int, int], margin: float) -> bool:
    """A box touching the frame border is cut by it: its centre no longer follows the object."""
    return (
        box[0] <= margin
        or box[1] <= margin
        or box[2] >= image_hw[1] - margin
        or box[3] >= image_hw[0] - margin
    )


class SceneMotion:
    """The motion of the whole scene between two frames of one view, from the other boxes.

    In the head camera every box moves with the head; a moving object is one that moves
    **against the scene**. The scene's motion between consecutive detected frames is an affine
    map fitted by least squares to the centres of the non-hand instances present on both and
    not cut by the frame border (the bench is close to a plane, so an affine map absorbs head
    translation, rotation and zoom; parallax of tall objects remains and sets the fpv
    threshold), with one trimming pass that drops the worst quarter of the residuals so the
    moving objects do not pull the fit; fewer than six usable centres fall back to the median
    translation, fewer than three to the identity. The map between two frames a window apart
    is fitted directly when at least ten centres are shared, else composed from the
    per-step maps (many points and a small motion per step). In a fixed view every map is the
    identity to a fraction of a pixel.
    """

    MIN_AFFINE_POINTS = 6
    MIN_TRANSLATION_POINTS = 3
    MIN_DIRECT_POINTS = 10
    TRIM_FRACTION = 0.25
    INLIER_PX = 5.0
    BORDER_MARGIN_PX = 4.0

    def __init__(
        self,
        instances: list[Instance],
        params: SeedParams,
        image_hw: tuple[int, int],
        *,
        enabled: bool = True,
    ) -> None:
        self.image_hw = image_hw
        self.enabled = enabled
        self.centres: dict[int, dict[str, np.ndarray]] = defaultdict(dict)
        for inst in instances:
            if not enabled or inst.object_class in params.hand_classes:
                continue
            for frame, box in zip(inst.frames, inst.boxes):
                if box_clipped(box, image_hw, self.BORDER_MARGIN_PX):
                    continue
                self.centres[frame][inst.instance] = np.array(centre(box), dtype=float)
        self.frames = sorted(self.centres)
        self._index = {frame: i for i, frame in enumerate(self.frames)}
        self._cumulative: list[np.ndarray] | None = None
        self._cache: dict[tuple[int, int], np.ndarray] = {}

    @staticmethod
    def _lstsq(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        design = np.hstack([source, np.ones((source.shape[0], 1))])
        solution, *_ = np.linalg.lstsq(design, target, rcond=None)
        return solution  # 3 x 2: [x, y, 1] @ solution = [x', y']

    @classmethod
    def _fit(cls, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Robust affine fit as a 3x3 homogeneous matrix (rows act on column vectors)."""
        count = source.shape[0]
        if count >= cls.MIN_AFFINE_POINTS:
            solution = cls._lstsq(source, target)
            residuals = np.linalg.norm(
                np.hstack([source, np.ones((count, 1))]) @ solution - target, axis=1
            )
            keep = residuals <= np.quantile(residuals, 1.0 - cls.TRIM_FRACTION)
            if keep.sum() >= cls.MIN_AFFINE_POINTS:
                solution = cls._lstsq(source[keep], target[keep])
                residuals = np.linalg.norm(
                    np.hstack([source, np.ones((count, 1))]) @ solution - target, axis=1
                )
                # One inlier pass: static points agree to a few pixels, movers do not.
                inliers = residuals <= max(cls.INLIER_PX, 2.0 * float(np.median(residuals)))
                if inliers.sum() >= cls.MIN_AFFINE_POINTS:
                    solution = cls._lstsq(source[inliers], target[inliers])
        elif count >= cls.MIN_TRANSLATION_POINTS:
            solution = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
            solution[2] = np.median(target - source, axis=0)
        else:
            return np.eye(3)
        matrix = np.eye(3)
        matrix[:2, :] = solution.T
        return matrix

    def _shared(self, frame_i: int, frame_j: int) -> tuple[np.ndarray, np.ndarray]:
        shared = sorted(self.centres[frame_i].keys() & self.centres[frame_j].keys())
        source = np.array([self.centres[frame_i][k] for k in shared]).reshape(-1, 2)
        target = np.array([self.centres[frame_j][k] for k in shared]).reshape(-1, 2)
        return source, target

    def _cumulative_maps(self) -> list[np.ndarray]:
        if self._cumulative is None:
            maps = [np.eye(3)]
            for previous, frame in zip(self.frames, self.frames[1:]):
                step = self._fit(*self._shared(previous, frame))
                maps.append(step @ maps[-1])
            self._cumulative = maps
        return self._cumulative

    def matrix(self, frame_i: int, frame_j: int) -> np.ndarray:
        """The 3x3 homogeneous map carrying frame_i scene points onto frame_j."""
        if not self.enabled:
            return np.eye(3)
        key = (frame_i, frame_j)
        if key in self._cache:
            return self._cache[key]
        source, target = self._shared(frame_i, frame_j)
        if source.shape[0] >= self.MIN_DIRECT_POINTS:
            matrix = self._fit(source, target)
        elif frame_i in self._index and frame_j in self._index:
            maps = self._cumulative_maps()
            matrix = maps[self._index[frame_j]] @ np.linalg.inv(maps[self._index[frame_i]])
        else:
            matrix = self._fit(source, target)
        self._cache[key] = matrix
        return matrix

    def residual(
        self, frame_i: int, frame_j: int, point_i: np.ndarray, point_j: np.ndarray
    ) -> float:
        """How far `point_j` lies from where the scene's motion would have carried `point_i`."""
        predicted = self.matrix(frame_i, frame_j) @ np.append(point_i, 1.0)
        return float(np.linalg.norm(point_j - predicted[:2]))


def smoothed_centres(boxes: list[Box], smooth_frames: int) -> np.ndarray:
    raw = np.array([centre(b) for b in boxes], dtype=float).reshape(-1, 2)
    if raw.shape[0] <= 2 or smooth_frames <= 1:
        return raw
    half = smooth_frames // 2
    out = np.empty_like(raw)
    for i in range(raw.shape[0]):
        # Symmetric window, shrunk at the ends so a moving centre is not biased there.
        reach = min(half, i, raw.shape[0] - 1 - i)
        out[i] = np.median(raw[i - reach : i + reach + 1], axis=0)
    return out


@dataclass
class InstanceStats:
    instance: str
    object_class: str
    detected_frames: list[int]
    start_frame: int | None
    max_move_px: float
    max_move_raw_px: float
    move_frame: int | None
    in_hand_fraction: float
    rack_instance: str | None
    rack_fraction: float

    @property
    def persistent(self) -> bool:
        return self.start_frame is not None


def dense_start_frame(detected: list[int], min_persistence: int) -> int | None:
    """First detected frame with >= `min_persistence` detections in the next 2x frames."""
    if len(detected) < min_persistence:
        return None
    arr = np.asarray(detected)
    window = 2 * min_persistence
    for i, frame in enumerate(detected):
        count = int(np.searchsorted(arr, frame + window, side="left") - i)
        if count >= min_persistence:
            return int(frame)
    return None


def instance_stats(
    inst: Instance,
    *,
    params: SeedParams,
    hands_by_frame: dict[int, list[Box]],
    racks: list[Instance],
    motion: SceneMotion,
) -> InstanceStats:
    detected = [
        f
        for f, s, interp in zip(inst.frames, inst.scores, inst.interpolated)
        if s >= params.min_score and not interp
    ]
    start = dense_start_frame(detected, params.min_persistence)
    det_set = set(detected)
    keep = [i for i, f in enumerate(inst.frames) if f in det_set]
    frames = [inst.frames[i] for i in keep]
    boxes = [inst.boxes[i] for i in keep]
    max_move = max_raw = 0.0
    move_frame: int | None = None
    if len(frames) >= 2:
        smooth = smoothed_centres(boxes, params.smooth_frames)
        raw = np.array([centre(b) for b in boxes])
        arr = np.asarray(frames)
        clipped = [box_clipped(b, motion.image_hw, SceneMotion.BORDER_MARGIN_PX) for b in boxes]
        for i in range(len(frames)):
            j = int(np.searchsorted(arr, frames[i] + params.move_window, side="right") - 1)
            if j <= i or clipped[i] or clipped[j]:
                continue
            move = motion.residual(frames[i], frames[j], smooth[i], smooth[j])
            max_raw = max(max_raw, float(np.linalg.norm(raw[j] - raw[i])))
            if move > max_move:
                max_move, move_frame = move, frames[i]
    in_hand = 0
    for frame, box in zip(frames, boxes):
        c = centre(box)
        if any(inside(c, hand) for hand in hands_by_frame.get(frame, ())):
            in_hand += 1
    rack_instance, rack_fraction = None, 0.0
    if inst.object_class in params.group_classes and frames:
        counts: dict[str, int] = defaultdict(int)
        rack_boxes = {r.instance: r.box_at() for r in racks}
        for frame, box in zip(frames, boxes):
            c = centre(box)
            for rack_id, at in rack_boxes.items():
                rack_box = at.get(frame)
                if rack_box is not None and inside(c, rack_box):
                    counts[rack_id] += 1
        if counts:
            rack_instance, hits = max(counts.items(), key=lambda kv: kv[1])
            rack_fraction = hits / len(frames)
            if rack_fraction < params.rack_fraction:
                rack_instance = None
    return InstanceStats(
        instance=inst.instance,
        object_class=inst.object_class,
        detected_frames=detected,
        start_frame=start,
        max_move_px=max_move,
        max_move_raw_px=max_raw,
        move_frame=move_frame,
        in_hand_fraction=in_hand / len(frames) if frames else 0.0,
        rack_instance=rack_instance,
        rack_fraction=rack_fraction,
    )


def container_priority(object_class: str, params: SeedParams) -> int:
    """Machines first (the plan's containers and landmarks), then the racks, in list order."""
    return (
        params.container_classes.index(object_class)
        if object_class in params.container_classes
        else len(params.container_classes)
    )


def containment(inner: Box, outer: Box) -> float:
    """Fraction of `inner` covered by `outer`."""
    ix0, iy0 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix1, iy1 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area = max(0.0, inner[2] - inner[0]) * max(0.0, inner[3] - inner[1])
    return inter / area if area > 0 else 0.0


def suppress_nested_duplicates(
    instances: list[Instance], stats: dict[str, InstanceStats], params: SeedParams
) -> dict[str, dict[str, Any]]:
    """Same-class instances whose box sits inside another's on most shared frames are the
    detector's duplicate of one object (the plate and the plate-with-lid box); the one with
    fewer detected frames is dropped and recorded."""
    suppressed: dict[str, dict[str, Any]] = {}
    by_class: dict[str, list[Instance]] = defaultdict(list)
    for inst in instances:
        by_class[inst.object_class].append(inst)
    for members in by_class.values():
        if len(members) < 2:
            continue
        ordered = sorted(members, key=lambda i: -len(stats[i.instance].detected_frames))
        for a_index, keeper in enumerate(ordered):
            if keeper.instance in suppressed:
                continue
            keeper_at = keeper.box_at()
            for other in ordered[a_index + 1 :]:
                if other.instance in suppressed:
                    continue
                other_at = other.box_at()
                shared = sorted(keeper_at.keys() & other_at.keys())
                if len(shared) < max(5, 0.5 * min(len(keeper.frames), len(other.frames))):
                    continue
                nested = [
                    max(
                        containment(other_at[f], keeper_at[f]),
                        containment(keeper_at[f], other_at[f]),
                    )
                    for f in shared
                ]
                fraction = float(np.mean(np.asarray(nested) >= params.duplicate_containment))
                if fraction >= params.duplicate_fraction:
                    suppressed[other.instance] = {
                        "duplicate_of": keeper.instance,
                        "class": other.object_class,
                        "reason": "nested_same_class",
                        "shared_frames": len(shared),
                        "nested_fraction": round(fraction, 3),
                        "detected_frames": len(stats[other.instance].detected_frames),
                    }
    # The same box under two class names (a pipette read as blue and as yellow): one object,
    # the class with the higher mean score keeps it.
    remaining = [i for i in instances if i.instance not in suppressed]
    remaining.sort(key=lambda i: -float(np.mean(i.scores)))
    for a_index, keeper in enumerate(remaining):
        if keeper.instance in suppressed:
            continue
        keeper_at = keeper.box_at()
        for other in remaining[a_index + 1 :]:
            if other.instance in suppressed or other.object_class == keeper.object_class:
                continue
            other_at = other.box_at()
            shared = sorted(keeper_at.keys() & other_at.keys())
            if len(shared) < max(5, 0.5 * min(len(keeper.frames), len(other.frames))):
                continue
            same = [
                box_iou(list(keeper_at[f]), list(other_at[f])) >= params.same_box_iou
                for f in shared
            ]
            fraction = float(np.mean(same))
            if fraction >= params.duplicate_fraction:
                suppressed[other.instance] = {
                    "duplicate_of": keeper.instance,
                    "class": other.object_class,
                    "reason": "same_box_other_class",
                    "shared_frames": len(shared),
                    "nested_fraction": round(fraction, 3),
                    "detected_frames": len(stats[other.instance].detected_frames),
                    "mean_score": round(float(np.mean(other.scores)), 4),
                    "keeper_mean_score": round(float(np.mean(keeper.scores)), 4),
                }
    return suppressed


@dataclass
class Slot:
    view: str
    slot: int
    label: str
    object_class: str
    role: str
    rule: str
    instance: str
    members: list[str]
    start_frame: int
    first_frame: int
    last_frame: int
    detected_frames: int
    max_move_px: float
    move_frame: int | None
    in_hand_fraction: float
    seed_candidates: list[dict[str, Any]]

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["class"] = record.pop("object_class")
        record["max_move_px"] = round(self.max_move_px, 1)
        record["in_hand_fraction"] = round(self.in_hand_fraction, 3)
        return record


def union_box(boxes: list[Box]) -> Box:
    arr = np.asarray(boxes, dtype=float).reshape(-1, 4)
    return (
        float(arr[:, 0].min()),
        float(arr[:, 1].min()),
        float(arr[:, 2].max()),
        float(arr[:, 3].max()),
    )


def group_boxes(members: list[Instance]) -> tuple[list[int], list[Box], list[float]]:
    """Per frame, the union of the members' boxes and their best score."""
    per_frame: dict[int, list[tuple[Box, float]]] = defaultdict(list)
    for member in members:
        for frame, box, score in zip(member.frames, member.boxes, member.scores):
            per_frame[frame].append((box, score))
    frames = sorted(per_frame)
    boxes = [union_box([b for b, _ in per_frame[f]]) for f in frames]
    scores = [max(s for _, s in per_frame[f]) for f in frames]
    return frames, boxes, scores


def seed_candidates(
    frames: list[int],
    boxes: list[Box],
    scores: list[float],
    detected: list[int],
    start_frame: int,
    params: SeedParams,
) -> list[dict[str, Any]]:
    """The seed frames to try: the start frame, then every `min_persistence`-th detected
    frame after it, up to `seed_attempts` frames."""
    box_at = dict(zip(frames, boxes))
    score_at = dict(zip(frames, scores))
    after = [f for f in detected if f >= start_frame]
    picks = after[:: max(1, params.min_persistence)][: params.seed_attempts]
    return [
        {
            "frame": int(f),
            "box": [round(v, 1) for v in box_at[f]],
            "score": round(score_at[f], 4),
        }
        for f in picks
        if f in box_at
    ]


def select_view_slots(
    view: str,
    instances: list[Instance],
    *,
    image_hw: tuple[int, int],
    params: SeedParams,
    corroborated_classes: set[str] | None = None,
) -> dict[str, Any]:
    """Apply the seed rule to one view's instances; returns slots, capped and detector_only.

    `corroborated_classes` (the fpv): the classes that opened a `moves` / `in_hand` slot in
    some fixed view; a dynamic slot in the head camera needs its class in that set.
    """
    hands_by_frame: dict[int, list[Box]] = defaultdict(list)
    for inst in instances:
        if inst.object_class in params.hand_classes:
            for frame, box, score in zip(inst.frames, inst.boxes, inst.scores):
                if score >= params.min_score:
                    hands_by_frame[frame].append(box)
    racks = [i for i in instances if i.object_class in params.rack_classes]
    motion = SceneMotion(instances, params, image_hw, enabled=view in params.scene_motion_views)
    threshold = params.move_threshold_px(view, image_hw[1])
    by_id = {i.instance: i for i in instances}
    stats: dict[str, InstanceStats] = {}
    for inst in instances:
        if inst.object_class in params.hand_classes:
            continue
        stats[inst.instance] = instance_stats(
            inst, params=params, hands_by_frame=hands_by_frame, racks=racks, motion=motion
        )
    suppressed = suppress_nested_duplicates(
        [i for i in instances if i.instance in stats and stats[i.instance].persistent],
        stats,
        params,
    )

    dynamic: list[tuple[Instance, InstanceStats, str]] = []
    containers: list[tuple[Instance, InstanceStats]] = []
    group_members: dict[str, list[tuple[Instance, InstanceStats]]] = defaultdict(list)
    detector_only: dict[str, list[dict[str, Any]]] = defaultdict(list)
    uncorroborated: list[dict[str, Any]] = []
    not_persistent = 0

    def detector_only_row(inst: Instance, st: InstanceStats, reason: str) -> dict[str, Any]:
        return {
            "instance": inst.instance,
            "detected_frames": len(st.detected_frames),
            "max_move_px": round(st.max_move_px, 1),
            "in_hand_fraction": round(st.in_hand_fraction, 3),
            "reason": reason,
        }

    for inst in instances:
        st = stats.get(inst.instance)
        if st is None:
            continue
        if not st.persistent:
            not_persistent += 1
            continue
        if inst.instance in suppressed:
            continue
        moves = st.max_move_px > threshold
        in_hand = st.in_hand_fraction > params.hand_fraction
        rule = "moves" if moves else "in_hand" if in_hand else None
        if rule is not None and corroborated_classes is not None:
            # The head camera moves with the head and its hand boxes cover half the bench: a
            # dynamic slot there needs a fixed camera to have seen the class move or be held.
            if inst.object_class not in corroborated_classes:
                uncorroborated.append(
                    {**detector_only_row(inst, st, f"{rule} in the fpv only"), "rule": rule}
                )
                rule = None
                moves = False
        if st.rack_instance is not None and not moves:
            group_members[st.rack_instance].append((inst, st))
            continue
        if rule is not None:
            dynamic.append((inst, st, rule))
        elif inst.object_class in params.container_classes:
            containers.append((inst, st))
        else:
            detector_only[inst.object_class].append(
                detector_only_row(inst, st, "static, not in a hand, not a named container")
            )
    groups: list[tuple[str, list[tuple[Instance, InstanceStats]]]] = []
    for rack_id, members in group_members.items():
        if len(members) >= params.group_min_members:
            groups.append((rack_id, members))
        else:
            for inst, st in members:
                held = st.in_hand_fraction > params.hand_fraction and (
                    corroborated_classes is None or inst.object_class in corroborated_classes
                )
                if held:
                    dynamic.append((inst, st, "in_hand"))
                else:
                    detector_only[inst.object_class].append(
                        detector_only_row(
                            inst,
                            st,
                            f"in rack {rack_id} with fewer than "
                            f"{params.group_min_members} members, static",
                        )
                    )
    dynamic.sort(key=lambda t: (-len(t[1].detected_frames), -t[1].max_move_px, t[0].instance))
    containers.sort(
        key=lambda t: (
            container_priority(t[0].object_class, params),
            -len(t[1].detected_frames),
            t[0].instance,
        )
    )
    groups.sort(key=lambda g: (-sum(len(st.detected_frames) for _, st in g[1]), g[0]))
    # A landmark keeps its place under the cap whatever rule opened it (a centrifuge whose
    # lid is worked "moves"; the rule stays on the record, the role is container).
    landmarks = [
        (inst, st, "container")
        for inst, st in containers
        if inst.object_class in params.landmark_classes
    ]
    landmarks += [t for t in dynamic if t[0].object_class in params.landmark_classes]
    landmarks.sort(key=lambda t: (container_priority(t[0].object_class, params), t[0].instance))
    dynamic = [t for t in dynamic if t[0].object_class not in params.landmark_classes]
    racks_only = [t for t in containers if t[0].object_class not in params.landmark_classes]

    ordered: list[Slot] = []
    class_counter: dict[str, int] = defaultdict(int)

    def label_for(cls: str) -> str:
        k = class_counter[cls]
        class_counter[cls] += 1
        return f"{cls}#{k}"

    def instance_slot(inst: Instance, st: InstanceStats, rule: str) -> Slot:
        role = "container" if inst.object_class in params.container_classes else "object"
        start = int(st.start_frame)  # type: ignore[arg-type]
        return Slot(
            view=view,
            slot=len(ordered),
            label=label_for(inst.object_class),
            object_class=inst.object_class,
            role=role,
            rule=rule,
            instance=inst.instance,
            members=[],
            start_frame=start,
            first_frame=inst.first_frame,
            last_frame=inst.last_frame,
            detected_frames=len(st.detected_frames),
            max_move_px=st.max_move_px,
            move_frame=st.move_frame,
            in_hand_fraction=st.in_hand_fraction,
            seed_candidates=seed_candidates(
                inst.frames, inst.boxes, inst.scores, st.detected_frames, start, params
            ),
        )

    # Order under the cap: the landmark containers (the plan's centrifuge, vortex, PCR
    # machine), the moving / held objects by persistence then movement, the racks, the groups.
    for inst, st, rule in landmarks:
        ordered.append(instance_slot(inst, st, rule))
    for inst, st, rule in dynamic:
        ordered.append(instance_slot(inst, st, rule))
    for inst, st in racks_only:
        ordered.append(instance_slot(inst, st, "container"))
    for rack_id, members in groups:
        cls = members[0][0].object_class
        g_frames, g_boxes, g_scores = group_boxes([m for m, _ in members])
        detected = sorted({f for _, st in members for f in st.detected_frames})
        start = min(int(st.start_frame) for _, st in members)  # type: ignore[arg-type]
        ordered.append(
            Slot(
                view=view,
                slot=len(ordered),
                label=label_for(f"{cls}_group"),
                object_class=f"{cls}_group",
                role="group",
                rule="group",
                instance=rack_id,
                members=[m.instance for m, _ in members],
                start_frame=start,
                first_frame=g_frames[0],
                last_frame=g_frames[-1],
                detected_frames=len(detected),
                max_move_px=max(st.max_move_px for _, st in members),
                move_frame=None,
                in_hand_fraction=max(st.in_hand_fraction for _, st in members),
                seed_candidates=seed_candidates(
                    g_frames, g_boxes, g_scores, detected, start, params
                ),
            )
        )
    kept, capped = ordered[: params.slot_cap], ordered[params.slot_cap :]
    for i, slot in enumerate(kept):
        slot.slot = i
    return {
        "view": view,
        "image_hw": list(image_hw),
        "move_threshold_px": round(threshold, 1),
        "slots": [s.to_record() for s in kept],
        "capped": [{**s.to_record(), "slot": None} for s in capped],
        "suppressed": [
            {**info, "instance": instance} for instance, info in sorted(suppressed.items())
        ],
        "uncorroborated": uncorroborated,
        "corroborated_classes": (
            None if corroborated_classes is None else sorted(corroborated_classes)
        ),
        "detector_only": {cls: rows for cls, rows in sorted(detector_only.items())},
        "not_persistent_instances": not_persistent,
        "hand_instances": sum(1 for i in instances if i.object_class in params.hand_classes),
        "slots_by_rule": {rule: sum(1 for s in kept if s.rule == rule) for rule in RULES},
        "_instances": by_id,
    }


def run_select(args: argparse.Namespace) -> dict[str, Any]:
    params = params_from_args(args)
    output = Path(args.output)
    summary = json.loads((output / "instances" / "summary.json").read_text(encoding="utf-8"))
    window_start = args.window_start
    if window_start is None:
        window_start = min(v["frame_first"] for v in summary["views"].values())
    window_end = max(v["frame_last"] for v in summary["views"].values()) + 1
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "step": "select",
        "window": {"start": int(window_start), "end": int(window_end)},
        "params": asdict(params),
        "detector": summary.get("detector"),
        "views": {},
    }
    # Fixed views first: the classes they see move or held corroborate the fpv's rules.
    views = sorted(summary["views"], key=lambda v: v == FPV_VIEW)
    dynamic_classes: set[str] = set()
    for view in views:
        meta, instances = read_instances(output, view)
        selected = select_view_slots(
            view,
            instances,
            image_hw=(int(meta["image_hw"][0]), int(meta["image_hw"][1])),
            params=params,
            corroborated_classes=(
                dynamic_classes if view == FPV_VIEW and params.fpv_corroboration else None
            ),
        )
        if view != FPV_VIEW:
            dynamic_classes |= {
                s["class"] for s in selected["slots"] if s["rule"] in ("moves", "in_hand")
            }
        selected.pop("_instances")
        result["views"][view] = selected
        print(
            f"{view}: {len(selected['slots'])} slots {selected['slots_by_rule']}, "
            f"{len(selected['capped'])} capped, "
            f"{sum(len(v) for v in selected['detector_only'].values())} detector_only",
            flush=True,
        )
    fs_common.write_json(output / "slots.json", result)
    return result


# --------------------------------------------------------------------------------------------
# decode (SAM3 image decoder, MuggledSAM interpreter)


def expand_box(box: Box, margin: float, image_hw: tuple[int, int]) -> Box:
    width, height = box[2] - box[0], box[3] - box[1]
    return (
        max(0.0, box[0] - margin * width),
        max(0.0, box[1] - margin * height),
        min(float(image_hw[1]), box[2] + margin * width),
        min(float(image_hw[0]), box[3] + margin * height),
    )


def build_decode_requests(
    slots_doc: dict[str, Any],
    *,
    output: Path,
    trial: str,
    videos: dict[str, Path],
    params: SeedParams,
) -> dict[str, Any]:
    """Per view and slot, the seed attempts the worker tries in order: for every candidate
    frame the tight box, then the same box with a margin; other slots' centres on that frame
    as negatives when asked for."""
    views: dict[str, Any] = {}
    for view, selected in slots_doc["views"].items():
        instance_boxes: dict[str, dict[int, Box]] = {}
        if params.other_instances_as_negatives:
            _meta, instances = read_instances(output, view)
            instance_boxes = {i.instance: i.box_at() for i in instances}
        image_hw = (int(selected["image_hw"][0]), int(selected["image_hw"][1]))
        entries = []
        for slot in selected["slots"]:
            attempts = []
            for candidate in slot["seed_candidates"]:
                box = tuple(float(v) for v in candidate["box"])
                negatives: list[list[float]] = []
                if params.other_instances_as_negatives:
                    for other in selected["slots"]:
                        if other["label"] == slot["label"]:
                            continue
                        for member in other["members"] or [other["instance"]]:
                            at = instance_boxes.get(member, {}).get(candidate["frame"])
                            if at is not None:
                                negatives.append([round(v, 1) for v in centre(at)])
                for kind, prompt in (
                    ("tight", box),
                    ("margin", expand_box(box, params.decode_margin, image_hw)),
                ):
                    attempts.append(
                        {
                            "frame": int(candidate["frame"]),
                            "kind": kind,
                            "box": [round(v, 1) for v in box],
                            "prompt_box": [round(v, 1) for v in prompt],
                            "detector_score": candidate["score"],
                            "negatives": negatives,
                        }
                    )
            entries.append(
                {
                    "slot": slot["slot"],
                    "label": slot["label"],
                    "class": slot["class"],
                    "rule": slot["rule"],
                    "attempts": attempts,
                }
            )
        views[view] = {
            "video": str(videos[view]),
            "image_hw": list(image_hw),
            "slots": entries,
        }
    return {
        "schema": SCHEMA,
        "trial": trial,
        "encoder_side": ENCODER_SIDE,
        "accept_iou": params.accept_iou,
        "max_attempt_frames": params.seed_attempts,
        "views": views,
    }


def mask_metrics(mask: np.ndarray, box: Box) -> dict[str, Any]:
    """Area, bbox, mask-bbox IoU vs the box and fill ratio (fill is reported, not tested)."""
    ys, xs = np.nonzero(mask)
    area = int(xs.size)
    x0, y0 = max(int(round(box[0])), 0), max(int(round(box[1])), 0)
    x1, y1 = min(int(round(box[2])), mask.shape[1]), min(int(round(box[3])), mask.shape[0])
    inside_count = int(mask[y0:y1, x0:x1].sum()) if x1 > x0 and y1 > y0 else 0
    box_area = max((x1 - x0) * (y1 - y0), 1)
    if area == 0:
        return {
            "mask_area_px": 0,
            "mask_bbox_px": None,
            "mask_bbox_iou_vs_box": 0.0,
            "fill_ratio": 0.0,
        }
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    return {
        "mask_area_px": area,
        "mask_bbox_px": bbox,
        "mask_bbox_iou_vs_box": round(box_iou([float(v) for v in bbox], list(box)), 4),
        "fill_ratio": round(inside_count / box_area, 4),
    }


def decode_worker_main(args: argparse.Namespace) -> int:
    """The GPU half: one image encode per distinct seed frame per view, the image decoder on
    every pending prompt, acceptance by mask-bbox IoU, masks written as PNG. Runs under the
    MuggledSAM interpreter; imports nothing from the battle package."""
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
        "state": "failed",
        "reason": None,
        "gpu_guard": decision.as_provenance(),
        "interpreter": sys.executable,
        "model": str(args.model),
        "encoder_side": int(requests["encoder_side"]),
        "accept_iou": float(requests["accept_iou"]),
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
        accept_iou = float(requests["accept_iou"])

        for view, spec in requests["views"].items():
            view_result: dict[str, Any] = {"slots": {}, "frames_encoded": []}
            result["views"][view] = view_result
            capture = cv2.VideoCapture(str(spec["video"]))
            if not capture.isOpened():
                raise RuntimeError(f"could not open {spec['video']}")
            pending: dict[str, int] = {s["label"]: 0 for s in spec["slots"] if s["attempts"]}
            slots_by_label = {s["label"]: s for s in spec["slots"]}
            records: dict[str, dict[str, Any]] = {
                s["label"]: {
                    "slot": s["slot"],
                    "label": s["label"],
                    "class": s["class"],
                    "rule": s["rule"],
                    "status": "unseeded",
                    "reason": "no seed candidate frame" if not s["attempts"] else None,
                    "attempts": [],
                    "seed": None,
                }
                for s in spec["slots"]
            }
            mask_dir = out_dir / "masks" / view
            mask_dir.mkdir(parents=True, exist_ok=True)
            while pending:
                frame_index = min(
                    slots_by_label[label]["attempts"][i]["frame"] for label, i in pending.items()
                )
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, image = capture.read()
                if not ok:
                    raise RuntimeError(f"{view}: could not read raw frame {frame_index}")
                height, width = image.shape[:2]
                with torch.inference_mode():
                    encoded = interact.encode_image(image, side, True)
                result["image_encodes"] += 1
                view_result["frames_encoded"].append(frame_index)
                for label in list(pending):
                    spec_slot = slots_by_label[label]
                    index = pending[label]
                    while (
                        index < len(spec_slot["attempts"])
                        and spec_slot["attempts"][index]["frame"] == frame_index
                    ):
                        attempt = spec_slot["attempts"][index]
                        prompt = attempt["prompt_box"]
                        norm_box = [
                            (prompt[0] / width, prompt[1] / height),
                            (prompt[2] / width, prompt[3] / height),
                        ]
                        negatives = [(x / width, y / height) for x, y in attempt["negatives"]]
                        with torch.inference_mode():
                            prompts = interact.encode_prompts([norm_box], [], negatives)
                            masks, ious = interact.generate_masks(encoded, prompts)
                            best = int(ious[0].argmax())
                            logits = torch.nn.functional.interpolate(
                                masks[:, [best]].float(),
                                size=(height, width),
                                mode="bilinear",
                                align_corners=False,
                            )
                            mask = logits.gt(0).squeeze().cpu().numpy()
                        result["decodes"] += 1
                        box = tuple(float(v) for v in attempt["box"])
                        metrics = mask_metrics(mask, box)
                        record = {
                            **attempt,
                            "candidate_ious": [round(float(v), 4) for v in ious[0].tolist()],
                            "chosen_candidate": best,
                            "decoder_iou_pred": round(float(ious[0, best]), 4),
                            **metrics,
                            "accepted": metrics["mask_bbox_iou_vs_box"] >= accept_iou,
                        }
                        records[label]["attempts"].append(record)
                        index += 1
                        if record["accepted"]:
                            mask_path = (
                                mask_dir / f"{label.replace('#', '_')}_{frame_index:06d}.png"
                            )
                            cv2.imwrite(str(mask_path), (mask.astype(np.uint8) * 255))
                            records[label]["status"] = "accepted"
                            records[label]["reason"] = None
                            records[label]["seed"] = {
                                **record,
                                "mask_path": str(mask_path.relative_to(out_dir)),
                                "image_hw": [height, width],
                            }
                            break
                    if records[label]["status"] == "accepted":
                        del pending[label]
                    elif index >= len(spec_slot["attempts"]):
                        del pending[label]
                        best_iou = max(
                            (a["mask_bbox_iou_vs_box"] for a in records[label]["attempts"]),
                            default=0.0,
                        )
                        frames_tried = sorted({a["frame"] for a in records[label]["attempts"]})
                        records[label]["reason"] = (
                            f"mask-bbox IoU vs box {best_iou:.2f} < {accept_iou} on "
                            f"{len(records[label]['attempts'])} decodes over frames {frames_tried}"
                        )
                    else:
                        pending[label] = index
            capture.release()
            view_result["slots"] = records
            accepted = sum(1 for r in records.values() if r["status"] == "accepted")
            print(
                f"{view}: {accepted}/{len(records)} seeds accepted, "
                f"{len(view_result['frames_encoded'])} frames encoded",
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


def run_decode(args: argparse.Namespace) -> dict[str, Any]:
    params = params_from_args(args)
    output = Path(args.output)
    slots_doc = json.loads((output / "slots.json").read_text(encoding="utf-8"))
    videos = resolve_videos(args, list(slots_doc["views"]))
    decode_dir = output / "decode"
    decode_dir.mkdir(parents=True, exist_ok=True)
    requests = build_decode_requests(
        slots_doc, output=output, trial=args.trial, videos=videos, params=params
    )
    requests_path = decode_dir / "requests.json"
    fs_common.write_json(requests_path, requests)
    results_path = decode_dir / "results.json"
    interpreter = Path(args.external_python)
    if not interpreter.is_file():
        result = {"state": "blocked", "reason": f"MuggledSAM interpreter missing: {interpreter}"}
        fs_common.write_json(results_path, result)
    else:
        command = decode_worker_command(args, requests=requests_path, results=results_path)
        fs_common.write_json(decode_dir / "worker_command.json", command)
        started = time.perf_counter()
        completed = subprocess.run(
            command, capture_output=True, text=True, env=worker_environment(), check=False
        )
        (decode_dir / "worker.log").write_text(completed.stdout + "\n" + completed.stderr)
        if results_path.is_file():
            result = json.loads(results_path.read_text(encoding="utf-8"))
        else:
            result = {
                "state": "failed",
                "reason": f"worker exited {completed.returncode} without results.json",
            }
        result["wall_seconds"] = time.perf_counter() - started
        print(f"decode worker {result['state']} in {result['wall_seconds']:.0f}s", flush=True)
    seeds = write_seeds_and_worker_files(
        output, slots_doc, result, trial=args.trial, params=params, videos=videos
    )
    if result.get("state") != "succeeded":
        print(f"decode {result.get('state')}: {result.get('reason')}", file=sys.stderr)
    return seeds


def resolve_videos(args: argparse.Namespace, views: list[str]) -> dict[str, Path]:
    raw_root = Path(args.raw_root)
    videos = {}
    for view in views:
        if view == FPV_VIEW:
            path = raw_root / "finebio_videos_fpv_test/finebio_videos" / f"{args.trial}.mp4"
        else:
            path = raw_root / "finebio_videos_tpv_test/finebio_videos" / f"{args.trial}_{view}.mp4"
        videos[view] = path.resolve()
    return videos


# --------------------------------------------------------------------------------------------
# seeds.json and the worker-ready files


def write_box_stream(
    path: Path,
    slots: list[dict[str, Any]],
    instances: dict[str, Instance],
    *,
    window_start: int,
    window_end: int,
) -> dict[str, Any]:
    """Mode-(i) box stream: per analysis frame the boxes of every selected slot from its
    tracklet (start frame onwards); slots are the seeds.json slot indices, labels the slot
    labels, so the worker's concepts are the slot labels."""
    per_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    boxes_per_slot: dict[str, int] = {}
    for slot in slots:
        if slot["members"]:
            frames, boxes, scores = group_boxes([instances[m] for m in slot["members"]])
            flags = [False] * len(frames)
            source = f"{DETECTOR_SOURCE}_group"
        else:
            inst = instances[slot["instance"]]
            frames, boxes, scores, flags = inst.frames, inst.boxes, inst.scores, inst.interpolated
            source = DETECTOR_SOURCE
        count = 0
        for frame, box, score, interpolated in zip(frames, boxes, scores, flags):
            if frame < slot["start_frame"] or frame < window_start or frame >= window_end:
                continue
            per_frame[frame - window_start].append(
                {
                    "slot": slot["slot"],
                    "label": slot["label"],
                    "box_xyxy_px": [round(v, 1) for v in box],
                    "score": round(score, 4),
                    "source": f"{source}_interpolated" if interpolated else source,
                }
            )
            count += 1
        boxes_per_slot[slot["label"]] = count
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for frame in sorted(per_frame):
            boxes = sorted(per_frame[frame], key=lambda b: b["slot"])
            handle.write(
                json.dumps({"frame_index": frame, "boxes": boxes}, separators=(",", ":")) + "\n"
            )
    return {
        "path": str(path),
        "frames_with_boxes": len(per_frame),
        "boxes_per_slot": boxes_per_slot,
        "concepts": [s["label"] for s in slots],
    }


def write_schedule(
    path: Path, accepted: list[dict[str, Any]], *, window_start: int
) -> dict[str, Any]:
    """Mode-(ii) schedule: the accepted seeds as box prompts, contiguous multiplex slots,
    `start_frame` in analysis frames when the seed frame is after the window start. The
    worker's vocabulary applies: a frame-0 seed is `detector`, a later one `detector_reseed`."""
    seeds = []
    for index, seed in enumerate(accepted):
        start = int(seed["seed"]["frame"]) - window_start
        entry: dict[str, Any] = {
            "target": seed["label"],
            "initial_multiplex_slot": index,
            "prompt_box_xyxy_px": [float(v) for v in seed["seed"]["box"]],
            "selected_by": "detector" if start == 0 else "detector_reseed",
        }
        if start > 0:
            entry["start_frame"] = start
        seeds.append(entry)
    payload = {"seeds": seeds, "corrections": []}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return {
        "path": str(path),
        "slots": len(seeds),
        "concepts": [s["target"] for s in seeds],
        "start_frames": {s["target"]: s.get("start_frame", 0) for s in seeds},
    }


def validate_worker_files(box_stream: Path, schedule: Path | None) -> dict[str, Any]:
    """Run the SAM3 worker's own parsers over the files (the battle env has them)."""
    from .muggled_worker import _corrections_by_frame, parse_box_stream, slot_start_frames

    frames, concepts = parse_box_stream(box_stream.read_text(encoding="utf-8"))
    report: dict[str, Any] = {
        "box_stream": {"frames": len(frames), "concepts": list(concepts)},
    }
    if schedule is not None:
        payload = json.loads(schedule.read_text(encoding="utf-8"))
        payload = {**payload, "memory_semantics": "append_prompt_memory_and_reset_frame_memory"}
        targets = tuple(s["target"] for s in payload["seeds"])
        grouped = _corrections_by_frame(
            payload, targets, memory_semantics=payload["memory_semantics"]
        )
        report["schedule"] = {
            "slots": len(targets),
            "start_frames": slot_start_frames(payload),
            "correction_frames": sorted(grouped),
        }
    return report


def write_seeds_and_worker_files(
    output: Path,
    slots_doc: dict[str, Any],
    decode_result: dict[str, Any],
    *,
    trial: str,
    params: SeedParams,
    videos: dict[str, Path],
) -> dict[str, Any]:
    window_start = int(slots_doc["window"]["start"])
    window_end = int(slots_doc["window"]["end"])
    seeds_doc: dict[str, Any] = {
        "schema": SCHEMA,
        "step": "decode",
        "trial": trial,
        "window": {"start": window_start, "end": window_end, "analysis_frame_0": window_start},
        "provenance": "auto",
        "selected_by": "detector",
        "decode": {
            k: decode_result.get(k)
            for k in (
                "state",
                "reason",
                "encoder_side",
                "accept_iou",
                "image_encodes",
                "decodes",
                "model_load_seconds",
                "elapsed_seconds",
                "wall_seconds",
                "gpu_peak_vram_bytes",
                "gpu_guard",
            )
        },
        "params": asdict(params),
        "detector": slots_doc.get("detector"),
        "views": {},
        "claim_boundary": CLAIM_BOUNDARY,
        "licence_note": LICENCE_NOTE,
    }
    decoded_views = decode_result.get("views", {}) if isinstance(decode_result, dict) else {}
    for view, selected in slots_doc["views"].items():
        _meta, instances = read_instances(output, view)
        by_id = {i.instance: i for i in instances}
        decoded = decoded_views.get(view, {}).get("slots", {})
        slot_records = []
        for slot in selected["slots"]:
            record = {
                **slot,
                "selected_by": "detector",
                "provenance": "auto",
                "video": str(videos.get(view, "")),
            }
            dec = decoded.get(slot["label"])
            if dec is None:
                record["status"] = "unseeded"
                record["reason"] = decode_result.get("reason") or "decode did not run"
                record["attempts"] = []
                record["seed"] = None
            else:
                record["status"] = dec["status"]
                record["reason"] = dec["reason"]
                record["attempts"] = dec["attempts"]
                record["seed"] = dec["seed"]
                if dec["seed"] is not None:
                    record["seed"]["mask_path"] = str(Path("decode") / dec["seed"]["mask_path"])
            slot_records.append(record)
        accepted = [r for r in slot_records if r["status"] == "accepted"]
        for index, record in enumerate(accepted):
            record["schedule_slot"] = index
        stream = write_box_stream(
            output / "box_streams" / f"{view}.jsonl",
            selected["slots"],
            by_id,
            window_start=window_start,
            window_end=window_end,
        )
        schedule = (
            write_schedule(
                output / "schedules" / f"{view}.json", accepted, window_start=window_start
            )
            if accepted
            else None
        )
        seeds_doc["views"][view] = {
            "image_hw": selected["image_hw"],
            "move_threshold_px": selected["move_threshold_px"],
            "slots": slot_records,
            "capped": selected["capped"],
            "detector_only": selected["detector_only"],
            "slots_by_rule": selected["slots_by_rule"],
            "accepted": len(accepted),
            "unseeded": sum(1 for r in slot_records if r["status"] == "unseeded"),
            "decode_ious": {
                r["label"]: r["seed"]["mask_bbox_iou_vs_box"] for r in accepted if r["seed"]
            },
            "box_stream": stream,
            "schedule": schedule,
            "frames_encoded": decoded_views.get(view, {}).get("frames_encoded", []),
        }
    fs_common.write_json(output / "seeds.json", seeds_doc)
    (output / "seeds.md").write_text(seeds_markdown(seeds_doc), encoding="utf-8")
    return seeds_doc


def seeds_markdown(seeds_doc: dict[str, Any]) -> str:
    lines = [
        f"# Detector seeds, {seeds_doc['trial']}, window "
        f"[{seeds_doc['window']['start']}, {seeds_doc['window']['end']})",
        "",
        f"Decode {seeds_doc['decode'].get('state')}: "
        f"{seeds_doc['decode'].get('image_encodes')} image encodes, "
        f"{seeds_doc['decode'].get('decodes')} decodes, accept IoU >= "
        f"{seeds_doc['decode'].get('accept_iou')}, peak VRAM "
        f"{(seeds_doc['decode'].get('gpu_peak_vram_bytes') or 0) / 2**30:.2f} GiB.",
        "",
        "| view | slot | label | role | rule | start | detected | move px | in hand | "
        "status | seed frame | det score | decode IoU pred | mask-bbox IoU |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for view, selected in seeds_doc["views"].items():
        for r in selected["slots"]:
            seed = r.get("seed") or {}
            lines.append(
                f"| {view} | {r['slot']} | {r['label']} | {r['role']} | {r['rule']} | "
                f"{r['start_frame']} | {r['detected_frames']} | {r['max_move_px']} | "
                f"{r['in_hand_fraction']:.2f} | {r['status']} | {seed.get('frame', '')} | "
                f"{seed.get('detector_score', '')} | {seed.get('decoder_iou_pred', '')} | "
                f"{seed.get('mask_bbox_iou_vs_box', '')} |"
            )
    lines += ["", "## Per view", ""]
    for view, selected in seeds_doc["views"].items():
        det_only = {cls: len(rows) for cls, rows in selected["detector_only"].items()}
        lines.append(
            f"- **{view}**: {len(selected['slots'])} slots by rule {selected['slots_by_rule']}; "
            f"accepted {selected['accepted']}, unseeded {selected['unseeded']}; "
            f"capped {len(selected['capped'])}; detector_only {det_only}; "
            f"frames encoded {selected['frames_encoded']}."
        )
        for r in selected["slots"]:
            if r["status"] != "accepted":
                lines.append(f"  - unseeded {r['label']} ({r['rule']}): {r['reason']}")
    lines += ["", seeds_doc["claim_boundary"], "", seeds_doc["licence_note"], ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# sheets (gate 1)

FAMILY_COLOURS: dict[str, tuple[int, int, int]] = {
    "pipette": (0, 140, 255),
    "tip": (0, 200, 255),
    "plate": (0, 255, 0),
    "tube": (255, 0, 255),
    "rack": (0, 255, 255),
    "machine": (255, 200, 0),
    "group": (255, 255, 255),
    "other": (200, 200, 200),
}


def class_family(object_class: str) -> str:
    if object_class.endswith("_group"):
        return "group"
    if "pipette" in object_class:
        return "pipette"
    if object_class.endswith("_rack"):
        return "rack"
    if "tip" in object_class:
        return "tip"
    if "plate" in object_class:
        return "plate"
    if "tube" in object_class or "spin_column" in object_class or "stripes" in object_class:
        return "tube"
    if object_class in ("centrifuge", "vortex_mixer", "pcr_machine", "magnetic_rack"):
        return "machine"
    return "other"


def crop_window(box: Box, image_hw: tuple[int, int], *, min_side: int = 320) -> Box:
    """A square-ish crop around the box with context: 2.5x the box, at least `min_side`."""
    cx, cy = centre(box)
    side = max(2.5 * max(box[2] - box[0], box[3] - box[1]), float(min_side))
    half = side / 2
    x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
    x1, y1 = int(min(image_hw[1], cx + half)), int(min(image_hw[0], cy + half))
    return (float(x0), float(y0), float(x1), float(y1))


def draw_seed_tile(
    frame: np.ndarray,
    *,
    box: Box,
    mask: np.ndarray | None,
    colour: tuple[int, int, int],
    lines: list[str],
    tile: int = 320,
) -> np.ndarray:
    import cv2

    image_hw = (frame.shape[0], frame.shape[1])
    x0, y0, x1, y1 = (int(v) for v in crop_window(box, image_hw, min_side=tile))
    crop = frame[y0:y1, x0:x1].copy()
    if mask is not None:
        sub = mask[y0:y1, x0:x1]
        layer = crop.copy()
        layer[sub] = colour
        crop = cv2.addWeighted(crop, 0.55, layer, 0.45, 0)
        contours, _ = cv2.findContours(
            sub.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(crop, contours, -1, colour, 2)
    bx0, by0 = int(box[0]) - x0, int(box[1]) - y0
    bx1, by1 = int(box[2]) - x0, int(box[3]) - y0
    cv2.rectangle(crop, (bx0, by0), (bx1, by1), colour, 2)
    scale = tile / max(crop.shape[0], crop.shape[1])
    resized = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)))
    canvas = np.zeros((tile + 18 * len(lines) + 8, tile, 3), dtype=np.uint8)
    oy = (tile - resized.shape[0]) // 2
    ox = (tile - resized.shape[1]) // 2
    canvas[oy : oy + resized.shape[0], ox : ox + resized.shape[1]] = resized
    for i, text in enumerate(lines):
        cv2.putText(
            canvas,
            text,
            (4, tile + 14 + 18 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def grid(tiles: list[np.ndarray], columns: int, title: str) -> np.ndarray:
    import cv2

    if not tiles:
        canvas = np.zeros((60, 900, 3), dtype=np.uint8)
        cv2.putText(canvas, title, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        return canvas
    height = max(t.shape[0] for t in tiles)
    width = max(t.shape[1] for t in tiles)
    rows = (len(tiles) + columns - 1) // columns
    canvas = np.zeros((50 + rows * height, columns * width, 3), dtype=np.uint8)
    cv2.putText(canvas, title, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, columns)
        y, x = 50 + r * height, c * width
        canvas[y : y + tile.shape[0], x : x + tile.shape[1]] = tile
    return canvas


def run_sheets(args: argparse.Namespace) -> dict[str, Any]:
    import cv2

    output = Path(args.output)
    seeds_doc = json.loads((output / "seeds.json").read_text(encoding="utf-8"))
    videos = resolve_videos(args, list(seeds_doc["views"]))
    sheets_dir = output / "sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"schema": SCHEMA, "step": "sheets", "views": {}}
    decisions = []
    for view, selected in seeds_doc["views"].items():
        capture = cv2.VideoCapture(str(videos[view]))
        frames_cache: dict[int, np.ndarray] = {}

        def frame_at(index: int) -> np.ndarray | None:
            if index not in frames_cache:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, image = capture.read()
                frames_cache[index] = image if ok else None
            return frames_cache[index]

        accepted_tiles, other_tiles = [], []
        for record in selected["slots"]:
            colour = FAMILY_COLOURS[class_family(record["class"])]
            seed = record.get("seed")
            decisions.append(
                {
                    "view": view,
                    "slot": record["slot"],
                    "label": record["label"],
                    "class": record["class"],
                    "rule": record["rule"],
                    "role": record["role"],
                    "status": record["status"],
                    "seed_frame": None if seed is None else seed["frame"],
                    "decision": None,
                    "note": "",
                }
            )
            if seed is not None:
                image = frame_at(int(seed["frame"]))
                if image is None:
                    continue
                mask_path = output / seed["mask_path"]
                mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                mask = None if mask_img is None else mask_img > 0
                accepted_tiles.append(
                    draw_seed_tile(
                        image,
                        box=tuple(float(v) for v in seed["box"]),
                        mask=mask,
                        colour=colour,
                        lines=[
                            f"{record['label']}  slot {record['slot']}",
                            f"{record['rule']} / {record['role']}  f{seed['frame']}",
                            f"det {seed['detector_score']:.2f}  dec {seed['decoder_iou_pred']:.2f}"
                            f"  bbox IoU {seed['mask_bbox_iou_vs_box']:.2f} {seed['kind']}",
                        ],
                    )
                )
            else:
                candidates = record.get("seed_candidates") or []
                if not candidates:
                    continue
                cand = candidates[0]
                image = frame_at(int(cand["frame"]))
                if image is None:
                    continue
                best = max(
                    (a.get("mask_bbox_iou_vs_box", 0.0) for a in record.get("attempts", [])),
                    default=0.0,
                )
                other_tiles.append(
                    draw_seed_tile(
                        image,
                        box=tuple(float(v) for v in cand["box"]),
                        mask=None,
                        colour=(0, 0, 255),
                        lines=[
                            f"{record['label']}  UNSEEDED",
                            f"{record['rule']} / {record['role']}  f{cand['frame']}",
                            f"det {cand['score']:.2f}  best bbox IoU {best:.2f}",
                        ],
                    )
                )
        capture.release()
        title = (
            f"{seeds_doc['trial']} {view}: {len(accepted_tiles)} accepted seeds "
            f"(auto), {selected['unseeded']} unseeded; gate 1 = accept/reject per slot"
        )
        sheet = grid(accepted_tiles, args.columns, title)
        path = sheets_dir / f"{view}.jpg"
        cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        entry = {"sheet": str(path), "accepted_tiles": len(accepted_tiles)}
        if other_tiles:
            other = grid(other_tiles, args.columns, f"{seeds_doc['trial']} {view}: unseeded slots")
            other_path = sheets_dir / f"{view}_unseeded.jpg"
            cv2.imwrite(str(other_path), other, [cv2.IMWRITE_JPEG_QUALITY, 85])
            entry["unseeded_sheet"] = str(other_path)
        report["views"][view] = entry
        print(f"{view}: sheet with {len(accepted_tiles)} accepted tiles", flush=True)
    template = {
        "schema": f"{SCHEMA}/decisions",
        "run": str(output.resolve()),
        "trial": seeds_doc["trial"],
        "seeds_sha256": fs_common.sha256_file(output / "seeds.json"),
        "how": (
            "Look at sheets/<view>.jpg; for each slot set decision to 'accept' or 'reject' "
            "(null = keep the auto-accepted seed). Then: battle-detector-seed apply-decisions "
            "--output <run> --decisions <this file>. Only rejected slots are removed from the "
            "filtered box streams and schedules; nothing is drawn or edited by hand."
        ),
        "default": "accept (provenance auto)",
        "decisions": decisions,
    }
    fs_common.write_json(output / "decisions.template.json", template)
    report["decisions_template"] = str(output / "decisions.template.json")
    fs_common.write_json(sheets_dir / "sheets.json", report)
    return report


# --------------------------------------------------------------------------------------------
# apply-decisions (the gate-1 filter)


def apply_decisions(output: Path, decisions_path: Path) -> dict[str, Any]:
    seeds_doc = json.loads((output / "seeds.json").read_text(encoding="utf-8"))
    decisions_doc = json.loads(Path(decisions_path).read_text(encoding="utf-8"))
    by_key = {
        (d["view"], d["label"]): (d.get("decision"), d.get("note", ""))
        for d in decisions_doc["decisions"]
    }
    window_start = int(seeds_doc["window"]["start"])
    window_end = int(seeds_doc["window"]["end"])
    filtered_dir = output / "filtered"
    filtered: dict[str, Any] = {
        **{k: v for k, v in seeds_doc.items() if k != "views"},
        "step": "apply-decisions",
        "provenance": "human_filtered",
        "decisions_file": str(Path(decisions_path).resolve()),
        "decisions_sha256": fs_common.sha256_file(Path(decisions_path)),
        "views": {},
    }
    for view, selected in seeds_doc["views"].items():
        _meta, instances = read_instances(output, view)
        by_id = {i.instance: i for i in instances}
        kept, rejected = [], []
        for record in selected["slots"]:
            decision, note = by_key.get((view, record["label"]), (None, ""))
            entry = {**record, "decision": decision, "note": note}
            if decision == "reject":
                rejected.append(entry)
            else:
                entry["provenance"] = "human_accepted" if decision == "accept" else "auto"
                kept.append(entry)
        for index, record in enumerate(kept):
            record["slot"] = index
        accepted = [r for r in kept if r["status"] == "accepted"]
        for index, record in enumerate(accepted):
            record["schedule_slot"] = index
        stream = write_box_stream(
            filtered_dir / "box_streams" / f"{view}.jsonl",
            kept,
            by_id,
            window_start=window_start,
            window_end=window_end,
        )
        schedule = (
            write_schedule(
                filtered_dir / "schedules" / f"{view}.json", accepted, window_start=window_start
            )
            if accepted
            else None
        )
        filtered["views"][view] = {
            **{k: v for k, v in selected.items() if k not in ("slots", "box_stream", "schedule")},
            "slots": kept,
            "rejected": rejected,
            "accepted": len(accepted),
            "box_stream": stream,
            "schedule": schedule,
        }
        print(f"{view}: kept {len(kept)}, rejected {len(rejected)}", flush=True)
    fs_common.write_json(filtered_dir / "seeds.json", filtered)
    return filtered


# --------------------------------------------------------------------------------------------
# CLI


def params_from_args(args: argparse.Namespace) -> SeedParams:
    params = SeedParams()
    for name in asdict(params):
        if hasattr(args, name) and getattr(args, name) is not None:
            value = getattr(args, name)
            if name.endswith("_classes") and isinstance(value, str):
                value = tuple(v for v in value.split(",") if v)
            setattr(params, name, value)
    # A later step reuses the parameters an earlier step recorded unless overridden.
    return params


def _load_recorded_params(output: Path, args: argparse.Namespace) -> None:
    for name in ("slots.json", "instances/summary.json"):
        path = output / name
        if path.is_file():
            recorded = json.loads(path.read_text(encoding="utf-8")).get("params", {})
            for key, value in recorded.items():
                if getattr(args, key, None) is None and key in SeedParams.__dataclass_fields__:
                    setattr(args, key, tuple(value) if isinstance(value, list) else value)
            return


def _add_param_arguments(parser: argparse.ArgumentParser, names: tuple[str, ...]) -> None:
    defaults = SeedParams()
    helps = {
        "assoc_iou": "IoU for same-class association across frames",
        "assoc_min_score": "detections below this score are not associated at all",
        "max_gap": "frames a tracklet may be unmatched before it closes",
        "min_score": "a frame counts as detected at this score (the plan's 0.3)",
        "min_persistence": "detected frames (score >= min-score) a slot needs",
        "move_px": "box-centre displacement over the window that opens a slot, px at 1920",
        "move_px_fpv": "the same for the head camera (ego-compensated displacement)",
        "move_window": "frames of the displacement window",
        "smooth_frames": "median filter over detected frames before measuring displacement",
        "hand_fraction": "fraction of frames with the centre inside a hand box that opens a slot",
        "rack_fraction": "fraction of frames inside one rack box that makes an instance a member",
        "group_min_members": "members in one rack that form a group slot",
        "duplicate_containment": "box containment that marks two same-class boxes as nested",
        "duplicate_fraction": "fraction of shared frames nested that suppresses the smaller one",
        "same_box_iou": "IoU at which two boxes of different classes are the same box",
        "fpv_corroboration": "fpv dynamic slots need the class to move or be held in a fixed view",
        "slot_cap": "slots per view",
        "seed_attempts": "seed frames tried per slot",
        "accept_iou": "mask-bbox IoU vs the detector box that accepts a seed",
        "decode_margin": "box margin (fraction) of the second prompt when the tight box fails",
        "container_classes": "comma-separated container classes (static volumes)",
        "landmark_classes": "comma-separated containers ranked before the moving objects",
        "rack_classes": "comma-separated rack classes (group hosts)",
        "group_classes": "comma-separated identical-instance classes",
    }
    for name in names:
        default = getattr(defaults, name)
        kind = str if name.endswith("_classes") else type(default)
        parser.add_argument(
            f"--{name.replace('_', '-')}",
            type=kind,
            default=None,
            help=f"{helps.get(name, name)} (default {default})",
        )


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


def _add_video_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--trial", required=True, help="FineBio trial id, e.g. P03_03_01")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Per-view SAM3 slots and seeds from FineBio detector boxes."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    inst = sub.add_parser("instances", help="IoU association of detections into tracklets.")
    inst.add_argument("--detections", type=Path, required=True)
    inst.add_argument("--views", default=",".join(VIEWS))
    inst.add_argument("--start", type=int, default=None, help="first raw frame (inclusive)")
    inst.add_argument("--end", type=int, default=None, help="last raw frame (exclusive)")
    inst.add_argument("--output", type=Path, required=True)
    _add_param_arguments(inst, ("assoc_iou", "assoc_min_score", "max_gap"))

    sel = sub.add_parser("select", help="The seed rule over the instances.")
    sel.add_argument("--output", type=Path, required=True)
    sel.add_argument(
        "--window-start", type=int, default=None, help="analysis frame 0 (default: first frame)"
    )
    _add_param_arguments(
        sel,
        (
            "min_score",
            "min_persistence",
            "move_px",
            "move_px_fpv",
            "move_window",
            "smooth_frames",
            "hand_fraction",
            "rack_fraction",
            "group_min_members",
            "duplicate_containment",
            "duplicate_fraction",
            "same_box_iou",
            "slot_cap",
            "seed_attempts",
            "container_classes",
            "landmark_classes",
            "rack_classes",
            "group_classes",
        ),
    )

    dec = sub.add_parser("decode", help="SAM3 image-decoder seeds (GPU) + worker files.")
    dec.add_argument("--output", type=Path, required=True)
    _add_video_arguments(dec)
    _add_param_arguments(dec, ("accept_iou", "decode_margin"))
    dec.add_argument(
        "--other-instances-as-negatives",
        dest="other_instances_as_negatives",
        action="store_const",
        const=True,
        default=None,
    )
    _add_gpu_arguments(dec)

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

    sheets = sub.add_parser("sheets", help="Gate-1 contact sheets and the decisions template.")
    sheets.add_argument("--output", type=Path, required=True)
    sheets.add_argument("--columns", type=int, default=4)
    _add_video_arguments(sheets)

    apply = sub.add_parser("apply-decisions", help="Filter the worker files by the decisions.")
    apply.add_argument("--output", type=Path, required=True)
    apply.add_argument("--decisions", type=Path, required=True)

    run = sub.add_parser("run", help="instances, select, decode and sheets in one go.")
    run.add_argument("--detections", type=Path, required=True)
    run.add_argument("--views", default=",".join(VIEWS))
    run.add_argument("--start", type=int, default=None)
    run.add_argument("--end", type=int, default=None)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--window-start", type=int, default=None)
    run.add_argument("--columns", type=int, default=4)
    run.add_argument("--skip-decode", action="store_true", help="stop after select")
    _add_video_arguments(run)
    _add_param_arguments(
        run, tuple(n for n in asdict(SeedParams()) if n != "other_instances_as_negatives")
    )
    run.add_argument(
        "--other-instances-as-negatives",
        dest="other_instances_as_negatives",
        action="store_const",
        const=True,
        default=None,
    )
    _add_gpu_arguments(run)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "decode-worker":
        return decode_worker_main(args)
    output = Path(args.output)
    if args.command in ("select", "decode", "sheets"):
        _load_recorded_params(output, args)
    if args.command == "instances":
        run_instances(args)
    elif args.command == "select":
        run_select(args)
    elif args.command == "decode":
        seeds = run_decode(args)
        if seeds["decode"].get("state") != "succeeded":
            return 3
    elif args.command == "sheets":
        run_sheets(args)
    elif args.command == "apply-decisions":
        apply_decisions(output, Path(args.decisions))
    elif args.command == "run":
        run_instances(args)
        run_select(args)
        if args.skip_decode:
            return 0
        seeds = run_decode(args)
        run_sheets(args)
        if seeds["decode"].get("state") != "succeeded":
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
