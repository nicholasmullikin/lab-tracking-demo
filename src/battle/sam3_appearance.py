"""SAM3 appearance signals under the MuggledSAM interpreter (Track A/B of the exemplar plan).

Two memory-free components of the SAM3 model the tracker already loads, applied offline to a
tracked run:

- **backbone embeddings**: the 1024-channel ViT token map of every frame, bilinearly
  upsampled to the 4x grid and mask-pooled under each part's tracked mask (L2-normalised), and
  the same pooling under human reference masks on their frames;
- **visual-exemplar detections**: exemplar tokens cut from human masks (boxes on the
  reference frames, `include_coordinate_encodings=False` so image content and not position is
  what transfers), in two variants (positives only; positives plus negative boxes at the other
  parts' reference masks on the same frame) and from one or more reference sets (same view;
  cross view), run through `generate_detections` on every frame; per row the tracked mask's
  best-overlap IoU with a detection, the top detection's centroid distance, its score and the
  presence score.

Reference leakage is bookkept, not assumed away: when the frame being scored is itself a
reference frame, that frame's exemplars are left out and the row records which reference
frames were used, so a scorer can score the cell leave-reference-out.

The module runs as a script under `/home/nick/.pyenv/versions/muggled_sam/bin/python` with
`muggled_sam` on `PYTHONPATH` (the same way `muggled_calibration_worker.py` runs): standard
library, numpy, torch and cv2 only; no pydantic. `gpu_guard` is imported as a sibling module.
Every output is validated by the Battle side (`battle.exemplar_pool`,
`battle.detector_scorecard_v2`).

Commands:

- `pass`: offline pass over a run (`native/embeddings.npz`, `detections.jsonl`, kept masks
  at named frames, `manifest.json`).
- `serve-jsonl`: a `batch_decode` server with the calibration worker's protocol whose
  candidates are exemplar detections gated to the prompt box (`--candidate-source
  exemplar_detector`), the image decoder (`image_decoder`, via the calibration worker's own
  function), or both merged in one process.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    from . import gpu_guard
except ImportError:
    _HERE = str(Path(__file__).resolve().parent)
    if _HERE not in sys.path:
        sys.path.append(_HERE)
    import gpu_guard  # type: ignore[no-redef]

REFERENCE_SCHEMA = "battle-sam3-appearance-references/1"
MANIFEST_SCHEMA = "battle-sam3-appearance-pass/1"
DETECTIONS_SCHEMA = "battle-sam3-appearance-detections/1"
VARIANTS = ("pos", "posneg")
DEFAULT_TOP_K = 10
DEFAULT_MAX_SIDE_LENGTH = 1280
EMBEDDING_CHANNELS = 1024
EMPTY_MASK_LIMITATION = (
    "no exemplar detection centroid fell inside the prompt box; one empty mask is returned so "
    "the acceptance rule rejects the onset"
)
EXEMPLAR_API = "muggledsam_sam3_exemplar_detector"
IMAGE_API = "muggledsam_sam3_interactive"
CANDIDATE_SOURCES = ("image_decoder", "exemplar_detector", "both")


# ------------------------------------------------------------------------------ references


@dataclass(frozen=True)
class Reference:
    frame: int
    target: str
    mask_path: str
    role: str = "positive"
    provenance: str = ""
    label: str | None = None


@dataclass(frozen=True)
class ReferenceSet:
    """Human masks on one view's proxy: positives per (frame, target) and optional distractors."""

    name: str
    view: str
    proxy: str
    references: tuple[Reference, ...]
    distractors: tuple[Reference, ...] = ()

    @property
    def frames(self) -> tuple[int, ...]:
        return tuple(
            sorted({r.frame for r in self.references} | {d.frame for d in self.distractors})
        )

    def positives_at(self, frame: int) -> tuple[Reference, ...]:
        return tuple(r for r in self.references if r.frame == frame)

    def targets(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(r.target for r in self.references))


def load_reference_spec(path: Path) -> tuple[str, tuple[ReferenceSet, ...]]:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    if spec.get("schema") != REFERENCE_SCHEMA:
        raise ValueError(f"{path}: expected schema {REFERENCE_SCHEMA}, got {spec.get('schema')!r}")
    sets = []
    for item in spec["sets"]:
        sets.append(
            ReferenceSet(
                name=str(item["name"]),
                view=str(item["view"]),
                proxy=str(item["proxy"]),
                references=tuple(
                    Reference(
                        frame=int(r["frame"]),
                        target=str(r["target"]),
                        mask_path=str(r["mask_path"]),
                        role=str(r.get("role", "positive")),
                        provenance=str(r.get("provenance", "")),
                    )
                    for r in item["references"]
                ),
                distractors=tuple(
                    Reference(
                        frame=int(d["frame"]),
                        target=str(d["target"]),
                        mask_path=str(d["mask_path"]),
                        role="distractor",
                        provenance=str(d.get("provenance", "")),
                        label=str(d.get("label", "distractor")),
                    )
                    for d in item.get("distractors", ())
                ),
            )
        )
    names = [s.name for s in sets]
    if len(set(names)) != len(names):
        raise ValueError("reference set names must be unique")
    return str(spec["view"]), tuple(sets)


# ------------------------------------------------------------------------------ mask helpers


def read_mask(path: str | Path) -> np.ndarray:
    """Bool mask from a PNG: alpha where RGBA (the run's cut-outs), else any non-zero value."""
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"mask is unavailable: {path}")
    if image.ndim == 3 and image.shape[2] == 4:
        return image[..., 3] > 0
    if image.ndim == 3:
        return image.max(axis=2) > 0
    return image > 0


def write_mask(path: Path, mask: np.ndarray) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), (mask.astype(np.uint8) * 255)):
        raise RuntimeError(f"could not write mask: {path}")


def mask_box_px(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Inclusive-exclusive pixel box (x1, y1, x2, y2) of a mask, None when empty."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def normalized_box(box_px: tuple[int, int, int, int], width: int, height: int) -> list[list[float]]:
    x1, y1, x2, y2 = box_px
    return [[x1 / width, y1 / height], [x2 / width, y2 / height]]


def mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def mask_iou(a: np.ndarray, b: np.ndarray) -> float | None:
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return None
    return float(np.logical_and(a, b).sum() / union)


def pool_weights(mask: np.ndarray, grid_hw: tuple[int, int]) -> np.ndarray:
    """Area-resampled mask weights on a (squashed) token grid; sums to the mask's coverage."""
    import cv2

    return cv2.resize(
        mask.astype(np.float32), (grid_hw[1], grid_hw[0]), interpolation=cv2.INTER_AREA
    )


def pooled_embedding(feature_map: np.ndarray, weights: np.ndarray) -> np.ndarray | None:
    """Weighted mean of a [C, h, w] map under [h, w] weights, L2-normalised; None if no weight."""
    total = float(weights.sum())
    if total <= 1e-6:
        return None
    vector = np.tensordot(feature_map, weights, axes=([1, 2], [0, 1])) / total
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        return None
    return (vector / norm).astype(np.float32)


def point_in_box(
    point: tuple[float, float], box: dict[str, int] | tuple[int, int, int, int]
) -> bool:
    if isinstance(box, dict):
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
    else:
        x1, y1, x2, y2 = box
    return x1 <= point[0] < x2 and y1 <= point[1] < y2


# ------------------------------------------------------------------------- run observations


def read_run_masks(run_directory: Path, frame_count: int) -> dict[int, dict[str, str]]:
    """Frame -> label -> mask path (absolute) from `observations.jsonl` (frames < frame_count)."""
    found: dict[int, dict[str, str]] = {}
    with (run_directory / "observations.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            frame = int(row["analysis_frame_index"])
            if frame >= frame_count:
                continue
            entry = found.setdefault(frame, {})
            for item in row.get("objects", ()):
                mask = item.get("mask") or {}
                if item.get("label") and mask.get("uri"):
                    entry[str(item["label"])] = str(run_directory / str(mask["uri"]))
    return found


# --------------------------------------------------------------------------------- backend


@dataclass
class Detections:
    """Top-K detections of one exemplar set on one frame, masks at frame resolution."""

    scores: np.ndarray  # [K]
    boxes_norm: np.ndarray  # [K, 2, 2]
    masks: np.ndarray  # [K, H, W] bool
    presence: float


class Sam3Backend:
    """The real model. Constructed only under the MuggledSAM interpreter."""

    def __init__(self, model_path: Path, *, device: str, max_side_length: int) -> None:
        import torch
        from muggled_sam.make_sam import make_sam_from_state_dict

        if not model_path.is_file():
            raise FileNotFoundError(f"SAM3 checkpoint is unavailable: {model_path}")
        if "cuda" in device and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable for the SAM3 appearance pass")
        self.torch = torch
        self.device = device
        self.dtype = torch.bfloat16 if "cuda" in device else torch.float32
        core = make_sam_from_state_dict(str(model_path))
        core.to(device=device, dtype=self.dtype)
        self.core = core
        self.detector = core.get_detector_context()
        self.interactive = core.get_interactive_context()
        self.max_side_length = max_side_length

    def encode(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        """Raw 1024-channel tokens plus the three projections (the detector's is the third)."""
        torch = self.torch
        encoder = self.detector.image_encoder
        with torch.inference_mode():
            prepared = encoder.prepare_image(frame_bgr, self.max_side_length, True)
            tokens = encoder(prepared)
            projections = self.detector.image_projection(tokens)
        return {"tokens": tokens, "encoded": projections, "frame_hw": frame_bgr.shape[:2]}

    def feature_map_4x(self, encoding: dict[str, Any]) -> np.ndarray:
        """[1024, 4h, 4w] float32 numpy map, bilinear from the token grid."""
        torch = self.torch
        tokens = encoding["tokens"]
        h, w = tokens.shape[-2:]
        with torch.inference_mode():
            up = torch.nn.functional.interpolate(
                tokens.float(), size=(4 * h, 4 * w), mode="bilinear", align_corners=False
            )
        return up[0].cpu().numpy()

    def exemplar_tokens(
        self,
        encoding: dict[str, Any],
        boxes_norm: list[list[list[float]]],
        negative_boxes_norm: list[list[list[float]]] | None,
    ) -> Any:
        with self.torch.inference_mode():
            return self.detector.encode_exemplars(
                encoding["encoded"],
                box_xy1xy2_norm_list=[[tuple(b[0]), tuple(b[1])] for b in boxes_norm],
                negative_boxes_list=(
                    [[tuple(b[0]), tuple(b[1])] for b in negative_boxes_norm]
                    if negative_boxes_norm
                    else None
                ),
                include_coordinate_encodings=False,
            )

    def concat_tokens(self, token_sets: list[Any]) -> Any:
        return self.torch.cat(token_sets, dim=1)

    def detect(
        self, encoding: dict[str, Any], token_sets: list[Any], top_k: int
    ) -> list[Detections]:
        """Batched `generate_detections` with the segmentation head run on the top-K tokens only."""
        torch = self.torch
        detector = self.detector
        low, x2, x4 = encoding["encoded"][-1]
        frame_h, frame_w = encoding["frame_hw"]
        batch = len(token_sets)
        if batch == 0:
            return []
        width = max(int(t.shape[1]) for t in token_sets)
        channels = int(token_sets[0].shape[2])
        with torch.inference_mode():
            padded = torch.zeros((batch, width, channels), device=low.device, dtype=low.dtype)
            padding = torch.ones((batch, width), device=low.device, dtype=torch.bool)
            for index, tokens in enumerate(token_sets):
                count = int(tokens.shape[1])
                if count == 0:
                    raise ValueError("an exemplar set has no tokens")
                padded[index, :count] = tokens[0].to(low.dtype)
                padding[index, :count] = False
            low_b = low.expand(batch, -1, -1, -1)
            x2_b = x2.expand(batch, -1, -1, -1)
            x4_b = x4.expand(batch, -1, -1, -1)
            fused = detector.image_exemplar_fusion(low_b, padded, padding)
            det_tokens, boxes, scores, presence = detector.exemplar_detector(fused, padded, padding)
            k = min(top_k, int(scores.shape[1]))
            top_scores, top_index = scores.float().topk(k, dim=1)
            top_tokens = det_tokens.gather(
                1, top_index.unsqueeze(-1).expand(-1, -1, det_tokens.shape[2])
            )
            top_boxes = boxes.gather(1, top_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2, 2))
            masks, _ = detector.exemplar_segmentation(
                top_tokens, fused, x2_b, x4_b, padded, padding
            )
            masks_full = torch.nn.functional.interpolate(
                masks.float(), size=(frame_h, frame_w), mode="bilinear", align_corners=False
            ).gt(0)
            presence_flat = presence.float().reshape(batch)
            scores_np = top_scores.cpu().numpy()
            boxes_np = top_boxes.float().cpu().numpy()
            masks_np = masks_full.cpu().numpy()
            presence_np = presence_flat.cpu().numpy()
        return [
            Detections(
                scores=scores_np[i],
                boxes_norm=boxes_np[i],
                masks=masks_np[i],
                presence=float(presence_np[i]),
            )
            for i in range(batch)
        ]

    def peak_vram_bytes(self) -> int | None:
        torch = self.torch
        if "cuda" in self.device and torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated(0))
        return None


# ----------------------------------------------------------------------------- exemplar bank


@dataclass
class ExemplarBank:
    """Encoded reference frames, per-(set, frame, target, variant) tokens, reference embeddings."""

    backend: Any
    sets: tuple[ReferenceSet, ...]
    frame_sizes: dict[str, tuple[int, int]] = field(default_factory=dict)
    tokens: dict[tuple[str, int, str, str], Any] = field(default_factory=dict)
    reference_embeddings: list[tuple[str, int, str, str, np.ndarray]] = field(default_factory=list)
    reference_areas: dict[tuple[str, int, str], int] = field(default_factory=dict)
    _cache: dict[tuple[str, str, str, int | None], Any] = field(default_factory=dict)

    def build(self) -> None:
        import cv2

        for reference_set in self.sets:
            capture = cv2.VideoCapture(reference_set.proxy)
            if not capture.isOpened():
                raise RuntimeError(f"could not open reference proxy: {reference_set.proxy}")
            try:
                for frame_index in reference_set.frames:
                    frame = read_frame(capture, frame_index)
                    height, width = frame.shape[:2]
                    self.frame_sizes[reference_set.name] = (width, height)
                    encoding = self.backend.encode(frame)
                    feature_map = None
                    positives = reference_set.positives_at(frame_index)
                    masks = {r.target: read_mask(r.mask_path) for r in positives}
                    boxes = {t: mask_box_px(m) for t, m in masks.items()}
                    for reference in positives:
                        mask = masks[reference.target]
                        if mask.shape != (height, width):
                            raise ValueError(
                                f"{reference_set.name} f{frame_index} {reference.target}: mask "
                                f"{mask.shape[::-1]} does not match the proxy {width}x{height}"
                            )
                        box = boxes[reference.target]
                        if box is None:
                            continue
                        self.reference_areas[
                            (reference_set.name, frame_index, reference.target)
                        ] = int(mask.sum())
                        if feature_map is None:
                            feature_map = self.backend.feature_map_4x(encoding)
                        vector = pooled_embedding(
                            feature_map, pool_weights(mask, feature_map.shape[1:])
                        )
                        if vector is not None:
                            self.reference_embeddings.append(
                                (
                                    reference_set.name,
                                    frame_index,
                                    reference.target,
                                    "positive",
                                    vector,
                                )
                            )
                        positive_boxes = [normalized_box(box, width, height)]
                        negatives = [
                            normalized_box(other, width, height)
                            for other_target, other in boxes.items()
                            if other_target != reference.target and other is not None
                        ]
                        self.tokens[(reference_set.name, frame_index, reference.target, "pos")] = (
                            self.backend.exemplar_tokens(encoding, positive_boxes, None)
                        )
                        self.tokens[
                            (reference_set.name, frame_index, reference.target, "posneg")
                        ] = self.backend.exemplar_tokens(
                            encoding, positive_boxes, negatives or None
                        )
                    for distractor in reference_set.distractors:
                        if distractor.frame != frame_index:
                            continue
                        mask = read_mask(distractor.mask_path)
                        if feature_map is None:
                            feature_map = self.backend.feature_map_4x(encoding)
                        vector = pooled_embedding(
                            feature_map, pool_weights(mask, feature_map.shape[1:])
                        )
                        if vector is not None:
                            self.reference_embeddings.append(
                                (
                                    reference_set.name,
                                    frame_index,
                                    distractor.target,
                                    f"distractor:{distractor.label}",
                                    vector,
                                )
                            )
            finally:
                capture.release()

    def reference_frames(self, set_name: str, target: str, variant: str) -> tuple[int, ...]:
        return tuple(
            sorted(
                f for (s, f, t, v) in self.tokens if s == set_name and t == target and v == variant
            )
        )

    def tokens_for(
        self, set_name: str, target: str, variant: str, exclude_frame: int | None
    ) -> tuple[Any | None, tuple[int, ...]]:
        """Concatenated exemplar tokens over the set's reference frames, minus `exclude_frame`."""
        all_frames = self.reference_frames(set_name, target, variant)
        frames = tuple(f for f in all_frames if f != exclude_frame)
        if not frames:
            return None, ()
        key = (set_name, target, variant, exclude_frame if exclude_frame in all_frames else None)
        if key not in self._cache:
            self._cache[key] = self.backend.concat_tokens(
                [self.tokens[(set_name, f, target, variant)] for f in frames]
            )
        return self._cache[key], frames


def read_frame(capture: Any, frame_index: int) -> np.ndarray:
    capture.set(1, frame_index)  # cv2.CAP_PROP_POS_FRAMES
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"could not decode proxy frame {frame_index}")
    return frame


# --------------------------------------------------------------------------------- the pass


def detection_row(
    *,
    frame: int,
    target: str,
    set_name: str,
    reference_view: str,
    variant: str,
    reference_frames: tuple[int, ...],
    excluded_frame: int | None,
    tracked: np.ndarray | None,
    detections: Detections,
    width: int,
    height: int,
) -> dict[str, Any]:
    """One detections.jsonl row: how the tracked mask relates to the exemplar detections."""
    tracked_area = int(tracked.sum()) if tracked is not None else 0
    tracked_centroid = mask_centroid(tracked) if tracked is not None and tracked_area else None
    scores = detections.scores
    order = int(np.argmax(scores)) if scores.size else None
    top_mask = detections.masks[order] if order is not None else None
    top_centroid = mask_centroid(top_mask) if top_mask is not None else None
    if top_centroid is None and order is not None:
        box = detections.boxes_norm[order]
        top_centroid = (
            float((box[0][0] + box[1][0]) / 2 * width),
            float((box[0][1] + box[1][1]) / 2 * height),
        )
    distance = (
        float(
            np.hypot(top_centroid[0] - tracked_centroid[0], top_centroid[1] - tracked_centroid[1])
        )
        if top_centroid is not None and tracked_centroid is not None
        else None
    )
    best_iou: float | None = None
    best_index: int | None = None
    top_iou: float | None = None
    if tracked is not None and tracked_area:
        for index in range(scores.size):
            iou = mask_iou(tracked, detections.masks[index])
            if iou is None:
                continue
            if index == order:
                top_iou = iou
            if best_iou is None or iou > best_iou:
                best_iou, best_index = iou, index
    return {
        "frame": frame,
        "target": target,
        "set": set_name,
        "reference_view": reference_view,
        "variant": variant,
        "reference_frames_used": list(reference_frames),
        "excluded_reference_frame": excluded_frame,
        "tracked_area_px": tracked_area,
        "tracked_centroid_px": list(tracked_centroid) if tracked_centroid else None,
        "presence": float(detections.presence),
        "detections": int(scores.size),
        "detections_above_0.5": int((scores >= 0.5).sum()),
        "top_score": float(scores[order]) if order is not None else None,
        "top_area_px": int(top_mask.sum()) if top_mask is not None else None,
        "top_centroid_px": list(top_centroid) if top_centroid else None,
        "top_centroid_distance_px": distance,
        "top_iou_tracked": top_iou,
        "best_overlap_iou": best_iou,
        "best_overlap_index": best_index,
        "best_overlap_score": float(scores[best_index]) if best_index is not None else None,
    }


def guard_or_refuse(args: argparse.Namespace, *, own_pid: int) -> dict[str, Any]:
    profile = gpu_guard.sam3_profile_for_side_length(int(args.max_side_length))
    decision = gpu_guard.evaluate(
        mode=args.gpu_guard,
        allowed_pids=[int(p) for p in (args.allow_gpu_neighbour or [])],
        profile=profile,
        own_pid=own_pid,
    )
    provenance = decision.as_provenance()
    if not decision.accepted and "cuda" in args.device:
        raise SystemExit(
            json.dumps({"state": "blocked", "reason": decision.reason, "gpu_guard": provenance})
        )
    return provenance


def run_pass(args: argparse.Namespace, *, backend: Any | None = None) -> Path:
    """Offline pass: embeddings + detections for every frame and part of one tracked run."""
    import cv2

    started = time.monotonic()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "native").mkdir(exist_ok=True)
    target_view, sets = load_reference_spec(Path(args.references))
    targets = tuple(args.target) if args.target else ("chassis", "interior", "rear_body", "cabin")
    guard = None
    if backend is None:
        guard = guard_or_refuse(args, own_pid=os.getpid())
        backend = Sam3Backend(
            Path(args.model), device=args.device, max_side_length=int(args.max_side_length)
        )
    run_directory = Path(args.run).resolve()
    frame_count = int(args.frame_count)
    run_masks = read_run_masks(run_directory, frame_count)
    frames = sorted({int(f) for f in args.frames}) if args.frames else list(range(frame_count))
    keep_frames = {int(f) for f in (args.keep_frame or [])}
    bank = ExemplarBank(backend=backend, sets=sets)
    bank.build()
    same_view = next((s for s in sets if s.view == target_view), None)
    same_view_reference_frames = set(same_view.frames) if same_view is not None else set()

    capture = cv2.VideoCapture(str(args.proxy))
    if not capture.isOpened():
        raise RuntimeError(f"could not open proxy: {args.proxy}")
    channels = (
        int(bank.reference_embeddings[0][4].shape[0])
        if bank.reference_embeddings
        else EMBEDDING_CHANNELS
    )
    embeddings = np.zeros((len(frames), len(targets), channels), dtype=np.float16)
    has_mask = np.zeros((len(frames), len(targets)), dtype=bool)
    areas = np.zeros((len(frames), len(targets)), dtype=np.int32)
    rows_written = 0
    kept_written = 0
    top_k = int(args.top_k)
    detections_path = output / "detections.jsonl"
    kept_path = output / "detections_kept.jsonl"
    plan: list[tuple[str, str, str]] = [
        (s.name, target, variant) for s in sets for target in targets for variant in VARIANTS
    ]
    try:
        with (
            detections_path.open("w", encoding="utf-8") as det_handle,
            kept_path.open("w", encoding="utf-8") as kept_handle,
        ):
            for position, frame_index in enumerate(frames):
                frame = read_frame(capture, frame_index)
                height, width = frame.shape[:2]
                encoding = backend.encode(frame)
                tracked: dict[str, np.ndarray | None] = {}
                for target in targets:
                    path = run_masks.get(frame_index, {}).get(target)
                    tracked[target] = read_mask(path) if path is not None else None
                feature_map = backend.feature_map_4x(encoding)
                for column, target in enumerate(targets):
                    mask = tracked[target]
                    if mask is None or not mask.any():
                        continue
                    areas[position, column] = int(mask.sum())
                    vector = pooled_embedding(
                        feature_map, pool_weights(mask, feature_map.shape[1:])
                    )
                    if vector is not None:
                        embeddings[position, column] = vector.astype(np.float16)
                        has_mask[position, column] = True
                exclude = frame_index if frame_index in same_view_reference_frames else None
                token_sets = []
                meta = []
                for set_name, target, variant in plan:
                    tokens, used = bank.tokens_for(
                        set_name,
                        target,
                        variant,
                        exclude
                        if next(s for s in sets if s.name == set_name).view == target_view
                        else None,
                    )
                    if tokens is None:
                        det_handle.write(
                            json.dumps(
                                {
                                    "frame": frame_index,
                                    "target": target,
                                    "set": set_name,
                                    "reference_view": next(
                                        s for s in sets if s.name == set_name
                                    ).view,
                                    "variant": variant,
                                    "reference_frames_used": [],
                                    "excluded_reference_frame": exclude,
                                    "unavailable": (
                                        "no reference frame remains after leave-reference-out"
                                    ),
                                }
                            )
                            + "\n"
                        )
                        rows_written += 1
                        continue
                    token_sets.append(tokens)
                    meta.append((set_name, target, variant, used))
                results = backend.detect(encoding, token_sets, top_k) if token_sets else []
                for (set_name, target, variant, used), detections in zip(
                    meta, results, strict=True
                ):
                    reference_view = next(s for s in sets if s.name == set_name).view
                    row = detection_row(
                        frame=frame_index,
                        target=target,
                        set_name=set_name,
                        reference_view=reference_view,
                        variant=variant,
                        reference_frames=used,
                        excluded_frame=exclude if reference_view == target_view else None,
                        tracked=tracked[target],
                        detections=detections,
                        width=width,
                        height=height,
                    )
                    det_handle.write(json.dumps(row) + "\n")
                    rows_written += 1
                    if frame_index in keep_frames:
                        kept = []
                        for index in range(int(detections.scores.size)):
                            mask = detections.masks[index]
                            name = (
                                f"f{frame_index:06d}_{target}_{set_name}_{variant}_{index:02d}.png"
                            )
                            write_mask(output / "detections" / name, mask)
                            box = mask_box_px(mask)
                            kept.append(
                                {
                                    "index": index,
                                    "score": float(detections.scores[index]),
                                    "area_px": int(mask.sum()),
                                    "box_px": list(box) if box else None,
                                    "centroid_px": list(mask_centroid(mask) or ()) or None,
                                    "uri": f"detections/{name}",
                                }
                            )
                        kept_handle.write(
                            json.dumps(
                                {
                                    "frame": frame_index,
                                    "target": target,
                                    "set": set_name,
                                    "reference_view": reference_view,
                                    "variant": variant,
                                    "reference_frames_used": list(used),
                                    "excluded_reference_frame": row["excluded_reference_frame"],
                                    "presence": row["presence"],
                                    "masks": kept,
                                }
                            )
                            + "\n"
                        )
                        kept_written += 1
                if position % 100 == 0:
                    print(
                        f"frame {frame_index} ({position + 1}/{len(frames)}) "
                        f"{time.monotonic() - started:.0f}s",
                        file=sys.stderr,
                        flush=True,
                    )
    finally:
        capture.release()
    reference_keys = [f"{s}|{f}|{t}|{role}" for (s, f, t, role, _) in bank.reference_embeddings]
    np.savez_compressed(
        output / "native" / "embeddings.npz",
        frames=np.asarray(frames, dtype=np.int32),
        targets=np.asarray(targets),
        embeddings=embeddings,
        has_mask=has_mask,
        areas=areas,
        reference_keys=np.asarray(reference_keys),
        reference_sets=np.asarray([s for (s, _, _, _, _) in bank.reference_embeddings]),
        reference_frames=np.asarray(
            [f for (_, f, _, _, _) in bank.reference_embeddings], dtype=np.int32
        ),
        reference_targets=np.asarray([t for (_, _, t, _, _) in bank.reference_embeddings]),
        reference_roles=np.asarray([r for (_, _, _, r, _) in bank.reference_embeddings]),
        reference_embeddings=(
            np.stack([v for (_, _, _, _, v) in bank.reference_embeddings]).astype(np.float16)
            if bank.reference_embeddings
            else np.zeros((0, channels), dtype=np.float16)
        ),
    )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "detections_schema": DETECTIONS_SCHEMA,
        "target_view": target_view,
        "run_directory": str(run_directory),
        "proxy": str(args.proxy),
        "references": str(Path(args.references).resolve()),
        "reference_sets": [
            {
                "name": s.name,
                "view": s.view,
                "proxy": s.proxy,
                "frames": list(s.frames),
                "references": [
                    {
                        "frame": r.frame,
                        "target": r.target,
                        "mask_path": r.mask_path,
                        "provenance": r.provenance,
                    }
                    for r in s.references
                ],
                "distractors": [
                    {
                        "frame": d.frame,
                        "target": d.target,
                        "mask_path": d.mask_path,
                        "label": d.label,
                    }
                    for d in s.distractors
                ],
            }
            for s in sets
        ],
        "targets": list(targets),
        "variants": list(VARIANTS),
        "frames": {
            "count": len(frames),
            "first": frames[0] if frames else None,
            "last": frames[-1] if frames else None,
        },
        "keep_frames": sorted(keep_frames),
        "top_k": top_k,
        "max_side_length": int(args.max_side_length),
        "include_coordinate_encodings": False,
        "pooling": (
            "1024-channel ViT tokens bilinearly upsampled to the 4x grid, area-resampled mask "
            "weights, weighted mean, L2-normalised"
        ),
        "detection_rows": rows_written,
        "kept_rows": kept_written,
        "elapsed_seconds": time.monotonic() - started,
        "gpu_peak_vram_bytes": backend.peak_vram_bytes()
        if hasattr(backend, "peak_vram_bytes")
        else None,
        "gpu_guard": guard,
        "model": str(args.model) if args.model else None,
        "device": args.device,
        "claim_boundary": (
            "Reference masks are human review evidence on a handful of frames, not ground truth; "
            "rows at a reference frame used the other reference frames only "
            "(reference_frames_used, excluded_reference_frame)."
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


# ---------------------------------------------------------------------------- serve (Track B)


def gate_detections(detections: Detections, prompt_box: dict[str, int]) -> list[int]:
    """Indices (score order) of detections whose mask centroid lies inside the prompt box."""
    kept = []
    for index in np.argsort(-detections.scores):
        centroid = mask_centroid(detections.masks[index])
        if centroid is not None and point_in_box(centroid, prompt_box):
            kept.append(int(index))
    return kept


def exemplar_decoder_result(
    *,
    detections: Detections,
    prompt_box: dict[str, int],
    candidate_id: str,
    results_directory: Path,
    frame: np.ndarray | None,
    reference_frames: tuple[int, ...],
    set_name: str,
    variant: str,
    write_overlay: bool = True,
) -> dict[str, Any]:
    """A `MuggledSAMImageDecoderResult`-shaped record whose candidates are gated detections."""
    height, width = detections.masks.shape[1:]
    kept = gate_detections(detections, prompt_box)
    records = []
    masks_dir = results_directory / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    limitations = [
        "Candidates are SAM3 visual-exemplar detections (encode_exemplars from human reference "
        f"masks, set {set_name!r}, variant {variant!r}, reference frames {list(reference_frames)}, "
        "include_coordinate_encodings=False) whose centroid lies inside the prompt box; "
        "iou_score is the detection score scaled by the presence score, not an IoU estimate.",
        f"presence_score={detections.presence:.4f}; "
        f"detections_returned={int(detections.scores.size)}; gated_inside_box={len(kept)}",
    ]
    if not kept:
        empty = np.zeros((height, width), dtype=bool)
        path = masks_dir / f"{candidate_id}_exemplar-00.png"
        write_mask(path, empty)
        records.append(
            {
                "candidate_index": 0,
                "iou_score": 0.0,
                "predicted_box": None,
                "mask_uri": f"results/masks/{path.name}",
                "review_uri": None,
                "is_deterministic_best": True,
            }
        )
        limitations.append(EMPTY_MASK_LIMITATION)
    for position, index in enumerate(kept):
        mask = detections.masks[index]
        path = masks_dir / f"{candidate_id}_exemplar-{position:02d}.png"
        write_mask(path, mask)
        box = mask_box_px(mask)
        records.append(
            {
                "candidate_index": position,
                "iou_score": float(detections.scores[index]),
                "predicted_box": (
                    {
                        "x": box[0] / width,
                        "y": box[1] / height,
                        "width": (box[2] - box[0]) / width,
                        "height": (box[3] - box[1]) / height,
                    }
                    if box
                    else None
                ),
                "mask_uri": f"results/masks/{path.name}",
                "review_uri": None,
                "is_deterministic_best": position == 0,
            }
        )
    overlay_uri = f"results/{candidate_id}_overlay.png"
    if write_overlay and frame is not None:
        import cv2

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (int(prompt_box["x1"]), int(prompt_box["y1"])),
            (int(prompt_box["x2"]), int(prompt_box["y2"])),
            (255, 255, 255),
            1,
        )
        for position, index in enumerate(kept):
            colour = (40, 200, 40) if position == 0 else (60, 60, 230)
            contours, _ = cv2.findContours(
                detections.masks[index].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, colour, 1)
        cv2.imwrite(str(results_directory / f"{candidate_id}_overlay.png"), overlay)
    return {
        "api": EXEMPLAR_API,
        "candidate_count": len(records),
        "deterministic_best_candidate_index": 0,
        "candidates": records,
        "overlay_uri": overlay_uri,
        "stability_score_available": False,
        "limitations": limitations,
    }


def merge_decoder_results(image: dict[str, Any], exemplar: dict[str, Any]) -> dict[str, Any]:
    """One result with the image decoder's candidates first, then the exemplar detections."""
    candidates = []
    for record in image["candidates"]:
        candidates.append(
            {**record, "candidate_index": len(candidates), "is_deterministic_best": False}
        )
    exemplar_start = len(candidates)
    for record in exemplar["candidates"]:
        candidates.append(
            {**record, "candidate_index": len(candidates), "is_deterministic_best": False}
        )
    best = max(range(len(candidates)), key=lambda i: (candidates[i]["iou_score"], -i))
    candidates[best]["is_deterministic_best"] = True
    return {
        "api": EXEMPLAR_API,
        "candidate_count": len(candidates),
        "deterministic_best_candidate_index": best,
        "candidates": candidates,
        "overlay_uri": exemplar["overlay_uri"],
        "stability_score_available": False,
        "limitations": [
            f"candidates 0..{exemplar_start - 1} are image-decoder masks (iou_score = decoder IoU "
            f"estimate); candidates {exemplar_start}.. are exemplar detections (iou_score = "
            "detection score); the two scores are not on one scale",
            *image.get("limitations", []),
            *exemplar.get("limitations", []),
        ],
    }


class ServeRuntime:
    """`--serve-jsonl`: the calibration worker protocol with exemplar-detection candidates."""

    def __init__(self, args: argparse.Namespace, *, backend: Any | None = None) -> None:
        import cv2

        self.cv2 = cv2
        self.args = args
        self.results_directory = Path(args.results_directory)
        (self.results_directory / "masks").mkdir(parents=True, exist_ok=True)
        (self.results_directory / "frames").mkdir(parents=True, exist_ok=True)
        self.capture = cv2.VideoCapture(str(args.proxy))
        if not self.capture.isOpened():
            raise RuntimeError(f"could not open proxy: {args.proxy}")
        self.guard = None
        if backend is None:
            self.guard = guard_or_refuse(args, own_pid=os.getpid())
            backend = Sam3Backend(
                Path(args.model), device=args.device, max_side_length=int(args.max_side_length)
            )
        self.backend = backend
        self.target_view, self.sets = load_reference_spec(Path(args.references))
        self.bank = ExemplarBank(backend=backend, sets=self.sets)
        self.bank.build()
        self.set_name = args.exemplar_set
        if self.set_name not in {s.name for s in self.sets}:
            raise ValueError(f"unknown exemplar set {self.set_name!r}")
        self.variant = args.exemplar_variant
        self.top_k = int(args.top_k)
        self.candidate_source = args.candidate_source
        self.frames: dict[int, np.ndarray] = {}
        self.encodings: dict[int, Any] = {}
        self.image_encodings: dict[int, Any] = {}
        self.detections: dict[tuple[int, str], tuple[Detections, tuple[int, ...]]] = {}

    def close(self) -> None:
        self.capture.release()

    def frame(self, frame_index: int) -> np.ndarray:
        if frame_index not in self.frames:
            self.frames[frame_index] = read_frame(self.capture, frame_index)
        return self.frames[frame_index]

    def encoding(self, frame_index: int) -> Any:
        if frame_index not in self.encodings:
            self.encodings[frame_index] = self.backend.encode(self.frame(frame_index))
        return self.encodings[frame_index]

    def detections_for(self, frame_index: int, target: str) -> tuple[Detections, tuple[int, ...]]:
        key = (frame_index, target)
        if key not in self.detections:
            set_view = next(s for s in self.sets if s.name == self.set_name).view
            exclude = frame_index if set_view == self.target_view else None
            tokens, used = self.bank.tokens_for(self.set_name, target, self.variant, exclude)
            if tokens is None:
                raise ValueError(
                    f"no exemplar reference for {target} usable at frame {frame_index}"
                )
            result = self.backend.detect(self.encoding(frame_index), [tokens], self.top_k)[0]
            self.detections[key] = (result, used)
        return self.detections[key]

    def image_decoder_result(self, prompt: dict[str, Any]) -> dict[str, Any]:
        """The calibration worker's own box decode, imported as a sibling module."""
        try:
            from . import muggled_calibration_worker as calibration_worker
        except ImportError:
            import muggled_calibration_worker as calibration_worker  # type: ignore[no-redef]
        frame_index = int(prompt["frame_index"])
        frame = self.frame(frame_index)
        if frame_index not in self.image_encodings:
            self.image_encodings[frame_index] = self.backend.interactive.encode_image(
                frame, None, True
            )
        return calibration_worker._decode_rectangle(
            interactive_model=self.backend.interactive,
            encoded_image=self.image_encodings[frame_index],
            frame=frame,
            prompt_box=prompt["pixel_box"],
            target_label=str(prompt["intended_target"]),
            candidate_id=prompt["candidate_id"],
            results_directory=self.results_directory,
            boxes=prompt.get("boxes"),
            fg_points=prompt.get("fg_points"),
            bg_points=prompt.get("bg_points"),
            show_review=False,
        )

    def handle(self, command: str, payload: dict[str, Any]) -> dict[str, Any]:
        if command == "health":
            return {"ready": True, "candidate_source": self.candidate_source}
        if command == "frame_preview":
            frame_index = int(payload["frame_index"])
            frame = self.frame(frame_index)
            path = self.results_directory / "frames" / f"frame-{frame_index:06d}.jpg"
            if not self.cv2.imwrite(str(path), frame):
                raise RuntimeError(f"could not write frame preview: {path}")
            return {"frame_index": frame_index, "image_uri": f"results/frames/{path.name}"}
        if command == "batch_decode":
            decoded = []
            for prompt in payload["prompts"]:
                frame_index = int(prompt["frame_index"])
                target = str(prompt["intended_target"])
                candidate_id = str(prompt["candidate_id"])
                exemplar_result = None
                if self.candidate_source in ("exemplar_detector", "both"):
                    detections, used = self.detections_for(frame_index, target)
                    exemplar_result = exemplar_decoder_result(
                        detections=detections,
                        prompt_box=prompt["pixel_box"],
                        candidate_id=candidate_id,
                        results_directory=self.results_directory,
                        frame=self.frame(frame_index),
                        reference_frames=used,
                        set_name=self.set_name,
                        variant=self.variant,
                    )
                if self.candidate_source == "exemplar_detector":
                    result = exemplar_result
                elif self.candidate_source == "image_decoder":
                    result = self.image_decoder_result(prompt)
                else:
                    assert exemplar_result is not None
                    result = merge_decoder_results(
                        self.image_decoder_result(prompt), exemplar_result
                    )
                decoded.append(
                    {
                        "box_id": prompt["box_id"],
                        "candidate_id": candidate_id,
                        "decoder_result": result,
                    }
                )
            return {"decoded": decoded}
        raise ValueError(f"unknown protocol command: {command}")


def serve_jsonl(args: argparse.Namespace, *, backend: Any | None = None) -> int:
    runtime: ServeRuntime | None = None
    try:
        with contextlib.redirect_stdout(sys.stderr):
            runtime = ServeRuntime(args, backend=backend)
        for line in sys.stdin:
            request_id = ""
            try:
                request = json.loads(line)
                request_id = str(request["request_id"])
                command = str(request["command"])
                if command == "shutdown":
                    print(
                        json.dumps(
                            {"request_id": request_id, "ok": True, "result": {"stopped": True}}
                        ),
                        flush=True,
                    )
                    break
                result = runtime.handle(command, dict(request.get("payload", {})))
                response = {"request_id": request_id, "ok": True, "result": result}
            except (KeyError, TypeError, ValueError, RuntimeError, OSError) as error:
                response = {"request_id": request_id, "ok": False, "error": str(error)}
            print(json.dumps(response), flush=True)
        return 0
    finally:
        if runtime is not None:
            runtime.close()


# ------------------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    def common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--proxy", required=True, help="the target view's proxy video")
        sub.add_argument(
            "--references",
            required=True,
            help="reference spec JSON (battle-exemplar-pool references)",
        )
        sub.add_argument(
            "--model", default="/home/nick/src/muggled_sam/model_weights/sam3.1_multiplex.pt"
        )
        sub.add_argument("--device", default="cuda:0")
        sub.add_argument("--max-side-length", type=int, default=DEFAULT_MAX_SIDE_LENGTH)
        sub.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
        sub.add_argument(
            "--gpu-guard", choices=gpu_guard.GUARD_MODES, default=gpu_guard.DEFAULT_GUARD_MODE
        )
        sub.add_argument("--allow-gpu-neighbour", type=int, action="append", default=[])

    run = commands.add_parser("pass", help="offline embeddings + detections over a tracked run")
    common(run)
    run.add_argument(
        "--run", required=True, help="tracked run directory (observations.jsonl, masks/)"
    )
    run.add_argument("--output", required=True)
    run.add_argument("--frame-count", type=int, default=1800)
    run.add_argument(
        "--frames", type=int, nargs="*", default=None, help="explicit frames (default: all)"
    )
    run.add_argument(
        "--keep-frame",
        type=int,
        action="append",
        default=None,
        help="frames whose top-K masks are written",
    )
    run.add_argument("--target", action="append", default=None)

    serve = commands.add_parser("serve-jsonl", help="batch_decode server with exemplar candidates")
    common(serve)
    serve.add_argument("--results-directory", required=True)
    serve.add_argument("--candidate-source", choices=CANDIDATE_SOURCES, default="exemplar_detector")
    serve.add_argument("--exemplar-set", default="same_view")
    serve.add_argument("--exemplar-variant", choices=VARIANTS, default="posneg")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "pass":
        output = run_pass(args)
        print(json.dumps({"state": "ok", "output": str(output)}))
        return 0
    return serve_jsonl(args)


if __name__ == "__main__":
    sys.exit(main())
