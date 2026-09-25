"""Adapters from the two FineBio producers to `FineBioObservation` rows (plan `p2-seeds`).

The tracker (`battle-multiview-tracks`) and the slice consume one row shape, in raw pixels
and raw frame indices.  Two tools produce the evidence in their own formats:

* `battle-finebio-detect` writes one JSONL per view with raw ``frame_index`` and
  ``detections[{class, class_id, score, box_xyxy_px}]`` (plus ``interpolated`` and
  ``source_frames`` on rows the CPU fallback filled in).  :func:`detections_to_observations`
  turns every box at or above a score into a ``detector`` row with ``slot = <class>#<rank>``
  (rank = same-class score rank in that frame, not an identity) and keeps ``interpolated`` in
  the provenance, so a reader can tell a detector box from a linear guess.
* The SAM3 worker (`battle-muggled-arms`, both modes) writes ``observations.jsonl`` in analysis
  frames (frame 0 = ``--start-frame``) with normalised boxes and mask PNGs.
  :func:`worker_to_observations` reads the masks back, computes the **area centroid** (the
  preflight fixtures carry the bbox centre; the arms carry the real one), the mask bbox and
  area, maps analysis frame ``k`` to raw frame ``start_frame + k``, keeps the decoder's IoU
  or the tracker's presence logit as ``sam3_object_score`` and names the source
  (``sam3_decode`` for the memory-free decode, ``sam3_video`` for the video-memory tracker).
  A box-decode row carries the detector box that prompted it; a video row is matched to a
  same-class detector row of the same frame by IoU when detector rows are supplied.

The fpv observes only with a valid shipped pose: :func:`fpv_pose_validity` (from the trial's
pose file) or :func:`pose_validity_from_fixture` (from ``fpv_poses.json``) give the lookup,
:func:`fill_pose_valid` applies it.  Nothing here reads a frame of video; masks are read only
to measure them and are never copied (FineBio, non-commercial research).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .finebio_detect import box_iou
from .multiview_schemas import (
    FINEBIO_FPV_VIEW,
    BoxXYXY,
    FineBioObservation,
    ObservationSource,
    write_jsonl,
)
from .observations import load_observations
from .schemas import FrameObservations, PerFrameObject

VIEWS: tuple[str, ...] = ("fpv", "T1", "T2", "T3", "T4", "T5")
DEFAULT_MIN_SCORE = 0.3
DEFAULT_MATCH_IOU = 0.3
PoseValidity = Callable[[int], bool]


def _box(values: Sequence[float]) -> BoxXYXY:
    x0, y0, x1, y1 = (round(float(v), 1) for v in values)
    return (x0, y0, x1, y1)


def _score(value: Any) -> float | None:
    if value is None:
        return None
    return round(min(max(float(value), 0.0), 1.0), 4)


def slot_label(object_class: str, rank: int) -> str:
    return f"{object_class}#{rank}"


def class_of_slot(slot: str) -> str:
    """``<class>#<k>`` -> ``<class>``; a label without ``#`` is its own class."""
    return slot.rsplit("#", 1)[0] if "#" in slot else slot


# -------------------------------------------------------------------------------- pose validity


def fpv_pose_validity(trial: str) -> PoseValidity:
    """Validity of the shipped per-frame fpv pose, from the trial's pose file under data/."""
    from .finebio_cameras import fpv_poses

    rets = fpv_poses(trial)[0]

    def valid(frame: int) -> bool:
        return 0 <= frame < len(rets) and bool(rets[frame])

    return valid


def pose_validity_from_fixture(path: Path) -> PoseValidity:
    """Validity from a fixtures-shaped ``fpv_poses.json`` (``frames: {"<frame>": {valid}}``);
    a frame the file does not list is invalid."""
    frames = json.loads(Path(path).read_text(encoding="utf-8"))["frames"]
    valid_frames = {int(frame) for frame, pose in frames.items() if pose.get("valid")}
    return lambda frame: frame in valid_frames


def fill_pose_valid(
    rows: Iterable[FineBioObservation], validity: PoseValidity
) -> list[FineBioObservation]:
    """Copy of `rows` with every fpv row's ``pose_valid`` set from `validity`."""
    out = []
    for row in rows:
        if row.view == FINEBIO_FPV_VIEW:
            row = row.model_copy(update={"pose_valid": bool(validity(row.frame_index))})
        out.append(row)
    return out


# ------------------------------------------------------------------------- detector -> rows


def read_detection_rows(path: Path) -> list[dict[str, Any]]:
    """The per-frame records of one `battle-finebio-detect` view file, in file order."""
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def detection_row_to_observations(
    record: Mapping[str, Any],
    *,
    min_score: float = DEFAULT_MIN_SCORE,
    pose_valid: bool = True,
) -> list[FineBioObservation]:
    """Every box of one per-frame detector record at ``score >= min_score`` as a detector row."""
    view = str(record["view"])
    frame = int(record["frame_index"])
    interpolated = bool(record.get("interpolated", False))
    provenance: dict[str, Any] = {}
    if interpolated:
        provenance = {"interpolated": True}
        if record.get("source_frames") is not None:
            provenance["source_frames"] = [int(f) for f in record["source_frames"]]
    kept = [d for d in record["detections"] if float(d["score"]) >= min_score]
    kept.sort(key=lambda d: -float(d["score"]))
    rank: dict[str, int] = {}
    rows = []
    for det in kept:
        cls = str(det["class"])
        k = rank.get(cls, 0)
        rank[cls] = k + 1
        rows.append(
            FineBioObservation(
                view=view,
                frame_index=frame,
                slot=slot_label(cls, k),
                object_class=cls,
                detector_score=_score(det["score"]),
                box_xyxy_px=_box(det["box_xyxy_px"]),
                pose_valid=pose_valid,
                source="detector",
                provenance=dict(provenance),
            )
        )
    return rows


def detections_to_observations(
    detections_dir: Path,
    views: Sequence[str] = VIEWS,
    min_score: float = DEFAULT_MIN_SCORE,
    *,
    pose_valid: PoseValidity | None = None,
    frames: Collection[int] | None = None,
) -> list[FineBioObservation]:
    """`battle-finebio-detect` output -> detector rows for `views` (files ``<view>.jsonl``).

    ``pose_valid`` is required when the fpv is among the views (the fpv observes only with a
    valid shipped pose; see :func:`fpv_pose_validity`).  ``frames`` restricts the raw frames.
    Boxes are rounded to 0.1 px and scores to four decimals, as the preflight fixtures are.
    """
    if FINEBIO_FPV_VIEW in views and pose_valid is None:
        raise ValueError("fpv rows need a pose validity lookup (fpv_pose_validity(trial))")
    detections_dir = Path(detections_dir)
    rows: list[FineBioObservation] = []
    wanted = None if frames is None else {int(f) for f in frames}
    for view in views:
        for record in read_detection_rows(detections_dir / f"{view}.jsonl"):
            frame = int(record["frame_index"])
            if wanted is not None and frame not in wanted:
                continue
            valid = bool(pose_valid(frame)) if view == FINEBIO_FPV_VIEW and pose_valid else True
            rows.extend(
                detection_row_to_observations(record, min_score=min_score, pose_valid=valid)
            )
    return rows


# --------------------------------------------------------------------------- worker -> rows


def mask_measurements(mask: np.ndarray) -> tuple[BoxXYXY, tuple[float, float], int] | None:
    """(mask bbox, area centroid, area) of a boolean mask, None when it is empty.

    The bbox is ``[x_min, y_min, x_max + 1, y_max + 1]`` and coordinates use pixel centres
    (``+ 0.5``), so a one-pixel mask at (10, 10) has bbox (10, 10, 11, 11) and centroid
    (10.5, 10.5), the same convention as a box centre.
    """
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    bbox = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
    centroid = (round(float(xs.mean()) + 0.5, 1), round(float(ys.mean()) + 0.5, 1))
    return bbox, centroid, int(xs.size)


def read_mask(path: Path) -> np.ndarray | None:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None if image is None else image > 0


def _slot_index(object_id: str) -> int | None:
    tail = object_id.rsplit("-", 1)[-1]
    return int(tail) if tail.isdigit() else None


def _resolve_slot_label(
    obj: PerFrameObject, slot_labels: Sequence[str] | Mapping[int, str] | None
) -> str:
    index = _slot_index(obj.object_id)
    if slot_labels is not None and index is not None:
        if isinstance(slot_labels, Mapping):
            if index in slot_labels:
                return str(slot_labels[index])
        elif index < len(slot_labels):
            return str(slot_labels[index])
    if "#" in obj.label:
        return obj.label
    return slot_label(obj.label, index if index is not None else 0)


def _normalised_to_px(box: Any, width: int, height: int) -> BoxXYXY:
    return _box(
        (box.x * width, box.y * height, (box.x + box.width) * width, (box.y + box.height) * height)
    )


def worker_to_observations(
    run_dir: Path,
    view: str,
    start_frame: int,
    slot_labels: Sequence[str] | Mapping[int, str] | None = None,
    *,
    pose_valid: PoseValidity | None = None,
    detector_rows: Iterable[FineBioObservation] | None = None,
    match_iou: float = DEFAULT_MATCH_IOU,
    image_size: tuple[int, int] | None = None,
) -> list[FineBioObservation]:
    """A SAM3 worker run (``observations.jsonl`` + ``masks/``) -> SAM3 rows in raw frames.

    ``slot_labels`` maps the worker's multiplex slot index (``object_id`` ``sam3-NN``) to the
    ``<class>#<k>`` slot label; without it a label that already carries ``#`` is used as is
    and a plain class label becomes ``<class>#<slot index>``.  ``detector_rows`` (same view,
    raw frames) attach the same-class detector box with the best IoU >= ``match_iou`` to
    video rows; box-decode rows carry the box that prompted them.  ``image_size`` (w, h) is
    read from the first mask when not given.
    """
    run_dir = Path(run_dir)
    frames: tuple[FrameObservations, ...] = load_observations(run_dir / "observations.jsonl")
    det_index: dict[tuple[int, str], list[FineBioObservation]] = {}
    for det in detector_rows or ():
        if det.view == view and det.source == "detector" and det.box_xyxy_px is not None:
            det_index.setdefault((det.frame_index, det.object_class), []).append(det)
    size = image_size
    rows: list[FineBioObservation] = []
    for frame in frames:
        raw_frame = start_frame + frame.analysis_frame_index
        valid = bool(pose_valid(raw_frame)) if view == FINEBIO_FPV_VIEW and pose_valid else True
        for obj in frame.objects:
            mask = read_mask(run_dir / obj.mask.uri) if obj.mask is not None else None
            if mask is not None and size is None:
                size = (int(mask.shape[1]), int(mask.shape[0]))
            if size is None:
                raise ValueError(
                    f"{run_dir}: no mask to read the image size from; pass image_size=(w, h)"
                )
            width, height = size
            slot = _resolve_slot_label(obj, slot_labels)
            cls = class_of_slot(slot)
            source: ObservationSource = (
                "sam3_decode" if obj.source == "sam3_decode" else "sam3_video"
            )
            provenance: dict[str, Any] = {"worker_label": obj.label, "object_id": obj.object_id}
            measured = mask_measurements(mask) if mask is not None else None
            if measured is None:
                mask_bbox = _normalised_to_px(obj.box, width, height)
                centroid = None
                area = None
                provenance["mask"] = "absent" if mask is None else "empty"
            else:
                mask_bbox, centroid, area = measured
            if obj.iou_prediction is not None:
                provenance["decoder_iou_pred"] = round(float(obj.iou_prediction), 4)
            if source == "sam3_decode":
                provenance["decoder_iou_pred"] = round(float(obj.confidence), 4)
            box: BoxXYXY | None = None
            det_score: float | None = None
            if obj.prompt_box is not None:
                box = _normalised_to_px(obj.prompt_box, width, height)
                det_score = _score(obj.prompt_score)
                provenance["prompt_source"] = obj.prompt_source
            elif det_index:
                best, best_iou = None, 0.0
                for det in det_index.get((raw_frame, cls), ()):
                    assert det.box_xyxy_px is not None
                    iou = box_iou(list(mask_bbox), list(det.box_xyxy_px))
                    if iou > best_iou:
                        best, best_iou = det, iou
                if best is not None and best_iou >= match_iou:
                    box, det_score = best.box_xyxy_px, best.detector_score
                    provenance["detector_box_iou"] = round(best_iou, 4)
                    provenance["detector_slot"] = best.slot
            if box is not None:
                provenance.setdefault(
                    "detector_box_iou", round(box_iou(list(mask_bbox), list(box)), 4)
                )
            rows.append(
                FineBioObservation(
                    view=view,
                    frame_index=raw_frame,
                    slot=slot,
                    object_class=cls,
                    detector_score=det_score,
                    box_xyxy_px=box,
                    mask_bbox_px=mask_bbox,
                    mask_centroid_px=centroid,
                    mask_area_px=area,
                    sam3_object_score=(
                        None if obj.object_score is None else round(float(obj.object_score), 4)
                    ),
                    pose_valid=valid,
                    source=source,
                    provenance=provenance,
                )
            )
    return rows


def write_observations(rows: Iterable[FineBioObservation], path: Path) -> int:
    """Rows sorted by frame, view, source and slot, written compactly; returns the count."""
    ordered = sorted(
        rows,
        key=lambda r: (
            r.frame_index,
            VIEWS.index(r.view) if r.view in VIEWS else len(VIEWS),
            r.source,
            r.slot,
        ),
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return write_jsonl(ordered, Path(path), compact=True)
