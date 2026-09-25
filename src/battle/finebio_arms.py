"""`battle-finebio-arms`: the tracking arms on a FineBio trial window (plan `p4-arms`).

One arm = one observation set for the 3D tracker (`battle-multiview-tracks`) on a trial
window, plus the label-free measures and the occlusion inventory the plan asks for:

* **(a) boxes only**: the detector rows of every view (`detections_to_observations`).
* **(b) per-frame box decode**: the SAM3 worker's `box-decode` run per view
  (`worker_to_observations`, area centroids, detector box attached) beside the detector rows.
* **(c) video memory seeded once** and **(d) (c) + re-seeds**: the worker's `video-memory`
  runs per view, same adapter, video rows matched to the same-class detector box by IoU.

The tracker is the one CLI (`multiview_tracks.main`), fed the arm's `observations.jsonl`, the
window camera config and the rig's gates. From its `tracks.jsonl`, `events.jsonl`,
`residuals.jsonl` and `identity_metrics.json` this module derives:

* **measures.json / .md**: per view and slot the frames with a mask, the mask-bbox IoU against
  the best same-class DINO box of that view and frame (median, p10, fraction >= 0.5; the same
  reference for every arm, so (b), whose prompt *is* a detector box, and (c), whose mask is
  free, are compared on one footing; group slots have no single detector box and are excluded
  from the IoU pool), the SAM3 object score (decoder IoU in (b), the tracker's presence logit in
  (c)/(d): not comparable across arms), the mask area series stability (median relative step,
  fraction of jumps > 0.5), the fraction of slot rows the tracker associated, the cross-view
  residuals per view, and the identity metrics.
* **occlusion_inventory.jsonl / .md**: every support-0 episode of every track (a run of
  `coasting` rows): start, length, outcome (re-acquired with latency / lost / open at the
  window end), the last position, whether the position projected into >= 2 views lay inside a
  hand box or inside a container's detector box at the first coasting frame, whether the
  detector still had the class within the gate there (an association miss rather than an
  occlusion), and the successor ids born with `possibly_same_as`. The summary counts the
  episodes that would need the `held`, `contained` and identical-instance group extensions
  (`p3-tracker-ext`).
* **reseed-schedule**: the (d) correction schedule from (c)'s `detector_reseed` /
  `handoff_reseed` events, one box correction per slot per K frames, in the worker's own
  payload format.
* **sanity**: the first-view check of a worker run against the preflight numbers (ms per
  frame or step, mask-bbox IoU vs the prompt or detector box on the first N frames).

Claim boundary: the FineBio detector was trained on FineBio's own objects and cameras; every
"IoU vs detector box" here is agreement between two models, not accuracy; identity metrics
against SAM3 per-view slots are a proxy; no human anchor exists yet. Nothing under `runs/` or
`data/` is committed (FineBio licence).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import multiview_tracks
from .finebio_cameras import Camera, cameras_from_config, read_camera_config
from .finebio_detect import box_iou
from .finebio_observations import (
    detections_to_observations,
    fpv_pose_validity,
    pose_validity_from_fixture,
    worker_to_observations,
    write_observations,
)
from .finebio_slice import resolve_fpv_source
from .multiview_schemas import (
    FINEBIO_FPV_VIEW,
    FineBioObservation,
    Track3D,
    TrackEvent,
    read_jsonl,
)
from .multiview_tracks import SAM3_SOURCES, ResidualRow, project

SCHEMA = "battle-finebio-arms/1"
ARMS = ("a", "b", "c", "d")
ARM_NAMES = {
    "a": "boxes-only control (detector + 3D tracker on box centres, no SAM3)",
    "b": "per-frame box decode, no video memory (SAM3 image decoder on the detector box)",
    "c": "SAM3.1 video memory seeded once (detector box seeds, pm-append, no tau)",
    "d": "(c) + detector re-seed + hand-off re-seed from (c)'s tracker events",
}
WORKER_MODE = {"b": "box_decode", "c": "video_memory", "d": "video_memory"}
HAND_CLASSES = ("left_hand", "right_hand")
# Identical-instance classes: per-instance identity is below the fixed cameras' resolution
# (preflight), so an occlusion episode of one of these is a candidate for a group track.
IDENTICAL_INSTANCE_CLASSES = (
    "micro_tube",
    "50ml_tube",
    "15ml_tube",
    "8_tube_stripes",
    "blue_tip",
    "yellow_tip",
    "red_tip",
    "8_channel_tip",
)
GROUP_SUFFIX = "_group"
DEFAULT_MIN_SCORE = 0.3
DEFAULT_RESEED_K = 30
DECISION_MARGIN = 0.02
# The preflight's worker numbers the first view of an arm is checked against.
SANITY_REFERENCE_MS = {"b": 152.0, "c": 220.0, "d": 220.0}
SANITY_MIN_IOU = 0.8
CLAIM_BOUNDARY = (
    "The FineBio DINO detector was trained on FineBio's own objects and on frames from these "
    "cameras; mask-bbox IoU against its boxes is agreement between two models, not accuracy. "
    "Identity metrics against SAM3 per-view slots are a proxy. No human anchor exists yet."
)
LICENCE_NOTE = (
    "FineBio is licensed for non-commercial research; frames, masks and videos derived from it "
    "stay under runs/ or data/ and are never committed or redistributed"
)

Box = tuple[float, float, float, float]


# --------------------------------------------------------------------------- clip config


@dataclass(frozen=True)
class ClipWindow:
    trial: str
    views: tuple[str, ...]
    fixed_views: tuple[str, ...]
    fpv_view: str
    start_frame: int
    end_frame_exclusive: int
    camera_config: Path
    containers: tuple[str, ...]
    probes: tuple[str, ...]
    proxies: dict[str, Path]
    clip_id: str

    @property
    def frames(self) -> range:
        return range(self.start_frame, self.end_frame_exclusive)


def load_clip(path: Path) -> ClipWindow:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("config_kind") != "finebio_clip_config":
        raise ValueError(f"{path} is not a finebio_clip_config")
    window = doc["window"]
    return ClipWindow(
        trial=str(doc["trial"]),
        views=tuple(doc["views"]),
        fixed_views=tuple(doc["fixed_views"]),
        fpv_view=str(doc.get("fpv_view", FINEBIO_FPV_VIEW)),
        start_frame=int(window["start_frame"]),
        end_frame_exclusive=int(window["end_frame_exclusive"]),
        camera_config=Path(doc.get("window_camera_config") or doc["camera_config"]),
        containers=tuple(doc.get("containers", ())),
        probes=tuple(doc.get("probes", HAND_CLASSES)),
        proxies={view: Path(uri) for view, uri in doc.get("proxies", {}).items()},
        clip_id=str(doc.get("clip_id", path.stem)),
    )


# --------------------------------------------------------------------------- worker runs


def find_worker_run(root: Path) -> Path:
    """The one succeeded `battle-muggled-arms` run directory under `root` (the newest when
    several succeeded); a run directory itself is accepted."""
    root = Path(root)
    if (root / "manifest.json").is_file() and (root / "observations.jsonl").is_file():
        return root
    candidates = []
    for manifest in sorted(root.glob("*/manifest.json")):
        doc = json.loads(manifest.read_text(encoding="utf-8"))
        statuses = doc.get("method_statuses") or []
        state = statuses[0].get("state") if statuses else None
        if state == "succeeded" and (manifest.parent / "observations.jsonl").is_file():
            candidates.append(manifest.parent)
    if not candidates:
        raise FileNotFoundError(f"no succeeded worker run under {root}")
    return candidates[-1]


def worker_timing(run_dir: Path) -> dict[str, Any]:
    """Wall-clock and VRAM of a worker run from its `worker_result.json` / manifest."""
    result = json.loads((Path(run_dir) / "worker_result.json").read_text(encoding="utf-8"))
    settings = result.get("runtime_settings", {})
    frames = int(result.get("frames_processed") or 0)
    elapsed = float(result.get("elapsed_seconds") or 0.0)
    ttfu = result.get("time_to_first_usable_output_seconds")
    out: dict[str, Any] = {
        "state": result.get("state"),
        "frames_processed": frames,
        "elapsed_seconds": round(elapsed, 1),
        "time_to_first_usable_output_seconds": ttfu,
        "ms_per_frame_elapsed": round(1000.0 * elapsed / frames, 1) if frames else None,
        "gpu_peak_vram_bytes": result.get("gpu_peak_vram_bytes"),
        "gpu_peak_vram_gib": (
            round(result["gpu_peak_vram_bytes"] / 2**30, 2)
            if result.get("gpu_peak_vram_bytes")
            else None
        ),
        "masks_written": result.get("masks_written"),
        "mode": settings.get("mode") or settings.get("prompt_mode"),
        "concepts": settings.get("concepts"),
    }
    if ttfu is not None and frames > 1:
        # Steady-state step time: the first usable output includes the model load.
        out["ms_per_frame_steady"] = round(1000.0 * (elapsed - float(ttfu)) / (frames - 1), 1)
    timing = settings.get("box_decode_timing_ms")
    if timing:
        out["box_decode_ms_per_prompted_frame_median"] = timing["per_prompted_frame"].get("median")
        out["box_decode_ms_image_encode_median"] = timing["image_encode"].get("median")
    return out


# --------------------------------------------------------------------------- observations


def _detector_rows(
    detections_dir: Path, views: Sequence[str], min_score: float, validity
) -> list[FineBioObservation]:
    return detections_to_observations(
        Path(detections_dir), tuple(views), min_score, pose_valid=validity
    )


def pose_validity(trial: str, fpv_poses: Path | None):
    """The fpv pose validity lookup: the fixtures' `fpv_poses.json` when given, else the
    trial's shipped pose file under data/."""
    if fpv_poses is not None:
        return pose_validity_from_fixture(Path(fpv_poses))
    return fpv_pose_validity(trial)


def _worker_rows_for_view(
    task: tuple[str, str, str, int, str, float, str | None],
) -> list[dict[str, Any]]:
    """Process-pool worker: one view's SAM3 rows (as dicts, for pickling)."""
    run_dir, view, detections_dir, start_frame, trial, min_score, fpv_poses = task
    validity = (
        pose_validity(trial, Path(fpv_poses) if fpv_poses else None)
        if view == FINEBIO_FPV_VIEW
        else None
    )
    detector = _detector_rows(Path(detections_dir), (view,), min_score, validity)
    rows = worker_to_observations(
        Path(run_dir),
        view,
        int(start_frame),
        pose_valid=validity,
        detector_rows=detector,
    )
    return [r.model_dump(exclude_none=True) for r in rows]


def build_observations(
    clip: ClipWindow,
    detections_dir: Path,
    arm: str,
    *,
    worker_root: Path | None = None,
    min_score: float = DEFAULT_MIN_SCORE,
    jobs: int = 6,
    fpv_poses: Path | None = None,
) -> tuple[list[FineBioObservation], dict[str, Any]]:
    """The arm's observation rows: every view's detector rows, plus the worker's SAM3 rows per
    view for arms (b)/(c)/(d) (`worker_root/<view>/<run>`)."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    validity = pose_validity(clip.trial, fpv_poses)
    detector = _detector_rows(detections_dir, clip.views, min_score, validity)
    summary: dict[str, Any] = {
        "arm": arm,
        "arm_name": ARM_NAMES[arm],
        "detector_rows": len(detector),
        "detector_rows_by_view": dict(Counter(r.view for r in detector)),
        "min_detector_score": min_score,
        "worker_runs": {},
        "sam3_rows_by_view": {},
        "sam3_rows_without_mask_by_view": {},
    }
    rows = list(detector)
    if arm == "a":
        return rows, summary
    if worker_root is None:
        raise ValueError(f"arm {arm} needs --worker-root with one run per view")
    tasks = []
    for view in clip.views:
        run_dir = find_worker_run(Path(worker_root) / view)
        summary["worker_runs"][view] = str(run_dir)
        tasks.append(
            (
                str(run_dir),
                view,
                str(detections_dir),
                clip.start_frame,
                clip.trial,
                min_score,
                str(fpv_poses) if fpv_poses is not None else None,
            )
        )
    if jobs > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=min(jobs, len(tasks))) as pool:
            per_view = list(pool.map(_worker_rows_for_view, tasks))
    else:
        per_view = [_worker_rows_for_view(task) for task in tasks]
    for task, dumped in zip(tasks, per_view):
        view = task[1]
        sam3 = [FineBioObservation.model_validate(d) for d in dumped]
        summary["sam3_rows_by_view"][view] = len(sam3)
        summary["sam3_rows_without_mask_by_view"][view] = sum(
            1 for r in sam3 if r.mask_area_px is None
        )
        rows.extend(sam3)
    return rows, summary


# --------------------------------------------------------------------------- tracker


def run_tracker_cli(
    observations: Path,
    cameras: Path,
    gates: Path | None,
    output: Path,
    extra: Sequence[str] = (),
    fpv_poses: Path | None = None,
) -> dict[str, Any]:
    argv = ["--observations", str(observations), "--cameras", str(cameras), "--output", str(output)]
    if gates is not None:
        argv += ["--gates", str(gates)]
    if fpv_poses is not None:
        argv += ["--fpv-poses", str(fpv_poses)]
    argv += list(extra)
    multiview_tracks.main(argv)
    return json.loads((Path(output) / "identity_metrics.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- measures


def _percentiles(values: Sequence[float]) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "median": None, "p10": None, "p90": None, "mean": None}
    return {
        "n": int(arr.size),
        "median": round(float(np.median(arr)), 4),
        "p10": round(float(np.percentile(arr, 10)), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
        "mean": round(float(arr.mean()), 4),
    }


def iou_summary(values: Sequence[float], threshold: float = 0.5) -> dict[str, Any]:
    out = _percentiles(values)
    arr = np.asarray(values, dtype=np.float64)
    out["fraction_ge_0p5"] = round(float(np.mean(arr >= threshold)), 4) if arr.size else None
    return out


def area_stability(areas: Sequence[tuple[int, int]]) -> dict[str, Any]:
    """Relative area step between consecutive frames of one slot: |a_t - a_s| / max(a_t, a_s)
    over frame pairs one frame apart; the median and the fraction of jumps over 0.5."""
    ordered = sorted(areas)
    steps = []
    for (f0, a0), (f1, a1) in zip(ordered, ordered[1:]):
        if f1 - f0 != 1 or max(a0, a1) <= 0:
            continue
        steps.append(abs(a1 - a0) / max(a0, a1))
    arr = np.asarray(steps, dtype=np.float64)
    return {
        "steps": int(arr.size),
        "relative_step_median": round(float(np.median(arr)), 4) if arr.size else None,
        "relative_step_p90": round(float(np.percentile(arr, 90)), 4) if arr.size else None,
        "jump_fraction_gt_0p5": round(float(np.mean(arr > 0.5)), 4) if arr.size else None,
        "area_median_px": int(np.median([a for _, a in ordered])) if ordered else None,
    }


def detector_index(
    rows: Iterable[FineBioObservation], min_score: float = DEFAULT_MIN_SCORE
) -> dict[tuple[str, int, str], list[FineBioObservation]]:
    """(view, frame, class) -> detector rows at score >= min_score."""
    index: dict[tuple[str, int, str], list[FineBioObservation]] = defaultdict(list)
    for r in rows:
        if r.source == "detector" and (r.detector_score or 0.0) >= min_score:
            index[(r.view, r.frame_index, r.object_class)].append(r)
    return index


def best_detector_iou(
    mask_bbox: Box, candidates: Sequence[FineBioObservation]
) -> tuple[float | None, FineBioObservation | None]:
    best, best_iou = None, None
    for det in candidates:
        assert det.box_xyxy_px is not None
        iou = box_iou(list(mask_bbox), list(det.box_xyxy_px))
        if best_iou is None or iou > best_iou:
            best, best_iou = det, iou
    return best_iou, best


def slot_measures(
    rows: Sequence[FineBioObservation],
    det_index: dict[tuple[str, int, str], list[FineBioObservation]],
    residuals: Sequence[ResidualRow],
    *,
    window: range,
    slot_start: dict[tuple[str, str], int] | None = None,
) -> dict[str, Any]:
    """Per (view, slot) label-free measures over the SAM3 rows; pooled per view and overall."""
    associated: set[tuple[str, int, str]] = {
        (r.view, r.frame_index, r.slot) for r in residuals if r.source in SAM3_SOURCES
    }
    per_slot: dict[tuple[str, str], dict[str, Any]] = {}
    ious: dict[tuple[str, str], list[float]] = defaultdict(list)
    scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    areas: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    frames_with_mask: dict[tuple[str, str], set[int]] = defaultdict(set)
    frames_no_mask_row: dict[tuple[str, str], int] = Counter()
    frames_mask_no_reference: dict[tuple[str, str], int] = Counter()
    frames_associated: dict[tuple[str, str], int] = Counter()
    classes: dict[tuple[str, str], str] = {}
    for r in rows:
        if r.source not in SAM3_SOURCES:
            continue
        key = (r.view, r.slot)
        classes[key] = r.object_class
        if r.mask_area_px is None:
            frames_no_mask_row[key] += 1
            continue
        frames_with_mask[key].add(r.frame_index)
        areas[key].append((r.frame_index, int(r.mask_area_px)))
        if r.sam3_object_score is not None:
            scores[key].append(float(r.sam3_object_score))
        if (r.view, r.frame_index, r.slot) in associated:
            frames_associated[key] += 1
        if r.object_class.endswith(GROUP_SUFFIX):
            continue
        assert r.mask_bbox_px is not None
        iou, _ = best_detector_iou(
            r.mask_bbox_px, det_index.get((r.view, r.frame_index, r.object_class), ())
        )
        if iou is None:
            frames_mask_no_reference[key] += 1
        else:
            ious[key].append(iou)
    for key in sorted(classes):
        view, slot = key
        cls = classes[key]
        with_mask = frames_with_mask[key]
        start = (slot_start or {}).get(key)
        expected = len([f for f in window if f >= start]) if start is not None else len(window)
        class_detected = 0
        class_detected_no_mask = 0
        if not cls.endswith(GROUP_SUFFIX):
            for f in window:
                if det_index.get((view, f, cls)):
                    class_detected += 1
                    if f not in with_mask:
                        class_detected_no_mask += 1
        per_slot[key] = {
            "view": view,
            "slot": slot,
            "class": cls,
            "group": cls.endswith(GROUP_SUFFIX),
            "frames_with_mask": len(with_mask),
            "frames_in_window_from_start": expected,
            "mask_fraction_of_window_from_start": (
                round(len(with_mask) / expected, 4) if expected else None
            ),
            "rows_without_mask": frames_no_mask_row[key],
            "frames_class_detected": class_detected,
            "frames_class_detected_without_mask": class_detected_no_mask,
            "frames_mask_without_detector_reference": frames_mask_no_reference[key],
            "detector_iou": iou_summary(ious[key]),
            "sam3_object_score": _percentiles(scores[key]),
            "area": area_stability(areas[key]),
            "frames_associated_by_tracker": frames_associated[key],
            "associated_fraction": (
                round(frames_associated[key] / len(with_mask), 4) if with_mask else None
            ),
        }
    per_view: dict[str, Any] = {}
    for view in sorted({k[0] for k in classes}):
        pooled = [v for (vw, _), vals in ious.items() if vw == view for v in vals]
        masks = sum(len(frames_with_mask[k]) for k in classes if k[0] == view)
        per_view[view] = {
            "slots": sum(1 for k in classes if k[0] == view),
            "frames_with_mask_total": masks,
            "detector_iou_pooled": iou_summary(pooled),
            "associated_fraction": (
                round(sum(frames_associated[k] for k in classes if k[0] == view) / masks, 4)
                if masks
                else None
            ),
        }
    everything = [v for vals in ious.values() for v in vals]
    return {
        "per_slot": [per_slot[k] for k in sorted(per_slot)],
        "per_view": per_view,
        "pooled": {
            "detector_iou": iou_summary(everything),
            "slots": len(classes),
            "frames_with_mask_total": sum(len(s) for s in frames_with_mask.values()),
            "rows_without_mask_total": sum(frames_no_mask_row.values()),
        },
        "reference": (
            "mask-bbox IoU against the best same-class FineBio DINO box (score >= 0.3) of the "
            "same view and frame; group slots (<class>_group) have no single detector box and "
            "are excluded from the IoU pool; rows whose class is not detected in that view and "
            "frame are counted, not scored"
        ),
    }


def residual_measures(residuals: Sequence[ResidualRow]) -> dict[str, Any]:
    per_view: dict[str, list[float]] = defaultdict(list)
    per_source: dict[str, list[float]] = defaultdict(list)
    for r in residuals:
        per_view[r.view].append(r.residual_px)
        per_source["sam3" if r.source in SAM3_SOURCES else "detector"].append(r.residual_px)
    return {
        "per_view": {v: _percentiles(vals) for v, vals in sorted(per_view.items())},
        "per_source": {s: _percentiles(vals) for s, vals in sorted(per_source.items())},
        "pooled": _percentiles([v for vals in per_view.values() for v in vals]),
    }


IDENTITY_KEYS = (
    "tracks_born",
    "tracks_lost",
    "tracks_live_at_end",
    "reacquisitions",
    "reacquisition_latency_frames",
    "ambiguities",
    "duplicate_pair_frames",
    "fragmentation",
    "slot_disagreements",
    "id_switches",
    "id_switch_reference",
    "events",
)


def identity_summary(
    metrics: dict[str, Any], probes: Sequence[str] = HAND_CLASSES
) -> dict[str, Any]:
    """The tracker's identity metrics, plus the births / losses / fragmentation of the object
    classes alone (the tracker also tracks the hands, which this plan treats as probes)."""
    out = {k: metrics.get(k) for k in IDENTITY_KEYS}
    per_class = metrics.get("per_class") or {}
    objects = {cls: v for cls, v in per_class.items() if cls not in probes}
    out["objects_only"] = {
        "tracks_born": sum(v.get("tracks_born", 0) for v in objects.values()),
        "tracks_lost": sum(v.get("tracks_lost", 0) for v in objects.values()),
        "fragmentation": sum(v.get("fragmentation", 0) for v in objects.values()),
        "classes": len(objects),
    }
    out["probes_excluded"] = list(probes)
    return out


# --------------------------------------------------------------------------- inventory


@dataclass
class Episode:
    track_id: str
    object_class: str
    start_frame: int  # first coasting frame
    last_observed_frame: int
    end_frame: int  # last coasting frame of the episode
    length_frames: int
    outcome: str  # reacquired | lost | open_at_window_end
    reacquired_frame: int | None
    reacquisition_latency_frames: int | None
    last_position_cm: tuple[float, float, float]
    last_support_views: tuple[str, ...]
    last_support_slots: dict[str, str]
    projected_views: dict[str, tuple[float, float]] = field(default_factory=dict)
    in_hand_views: dict[str, str] = field(default_factory=dict)
    in_container_views: dict[str, list[str]] = field(default_factory=dict)
    detector_within_gate_views: dict[str, float] = field(default_factory=dict)
    inferred_state: str = "unexplained"
    identical_instance_class: bool = False
    group_slot: bool = False
    probe_class: bool = False
    lost_at_timeout: bool = False
    held_in_all_projected_views: bool = False
    successor_tracks: list[dict[str, Any]] = field(default_factory=list)
    ambiguous_events: int = 0

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def _inside(point: Sequence[float], box: Box) -> bool:
    return bool(box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3])


def episodes_from_tracks(rows: Sequence[Track3D]) -> list[Episode]:
    """Support-0 episodes: maximal runs of `coasting` rows per track."""
    by_track: dict[str, list[Track3D]] = defaultdict(list)
    for r in rows:
        by_track[r.track_id].append(r)
    episodes: list[Episode] = []
    for track_id, track_rows in by_track.items():
        track_rows.sort(key=lambda r: r.frame_index)
        i = 0
        while i < len(track_rows):
            if track_rows[i].state != "coasting":
                i += 1
                continue
            j = i
            while j + 1 < len(track_rows) and track_rows[j + 1].state == "coasting":
                j += 1
            first, last = track_rows[i], track_rows[j]
            prev = track_rows[i - 1] if i > 0 else None
            nxt = track_rows[j + 1] if j + 1 < len(track_rows) else None
            if nxt is None:
                outcome = "open_at_window_end"
            elif nxt.state == "lost":
                outcome = "lost"
            else:
                outcome = "reacquired"
            last_observed = (
                prev.frame_index
                if prev is not None
                else first.frame_index - first.frames_unobserved
            )
            source_row = prev if prev is not None else first
            episodes.append(
                Episode(
                    track_id=track_id,
                    object_class=first.object_class,
                    start_frame=first.frame_index,
                    last_observed_frame=last_observed,
                    end_frame=last.frame_index if outcome != "lost" else nxt.frame_index,
                    length_frames=(
                        (nxt.frame_index if outcome == "reacquired" else last.frame_index)
                        - last_observed
                    ),
                    outcome=outcome,
                    reacquired_frame=nxt.frame_index if outcome == "reacquired" else None,
                    reacquisition_latency_frames=(
                        nxt.frame_index - last_observed if outcome == "reacquired" else None
                    ),
                    last_position_cm=tuple(float(x) for x in source_row.position_cm),
                    last_support_views=tuple(source_row.support_views),
                    last_support_slots=dict(source_row.support_slots),
                    identical_instance_class=first.object_class in IDENTICAL_INSTANCE_CLASSES,
                    group_slot=first.object_class.endswith(GROUP_SUFFIX),
                    lost_at_timeout=outcome == "lost",
                )
            )
            i = j + 1
    return episodes


def annotate_episodes(
    episodes: list[Episode],
    *,
    det_index: dict[tuple[str, int, str], list[FineBioObservation]],
    events: Sequence[TrackEvent],
    fixed_cams: dict[str, Camera],
    fpv_source,
    containers: Sequence[str],
    probes: Sequence[str],
    gate_px: float,
    min_views: int = 2,
) -> None:
    """Where the object was when its support dropped: the last position projected into every
    view with a valid pose at the first coasting frame, tested against the hand boxes, the
    container boxes and the same-class detector boxes there."""
    successors: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ambiguous: dict[str, int] = Counter()
    for e in events:
        if e.kind == "birth":
            for tid in e.payload.get("possibly_same_as", []):
                successors[tid].append(
                    {
                        "track_id": e.track_id,
                        "frame_index": e.frame_index,
                        "unconfirmed_reacquisition": e.payload.get("unconfirmed_reacquisition_of")
                        == tid,
                    }
                )
        elif e.kind == "ambiguous":
            for tid in e.payload.get("coasting_tracks", []):
                ambiguous[tid] += 1
    for ep in episodes:
        ep.probe_class = ep.object_class in probes
        ep.identical_instance_class = ep.object_class in IDENTICAL_INSTANCE_CLASSES
        ep.group_slot = ep.object_class.endswith(GROUP_SUFFIX)
        cams = dict(fixed_cams)
        fpv = fpv_source(ep.start_frame)
        if fpv is not None:
            cams[FINEBIO_FPV_VIEW] = fpv
        point = np.asarray(ep.last_position_cm, dtype=np.float64)
        for view, cam in cams.items():
            pixel = project(cam, point)
            if pixel is None or not (0 <= pixel[0] < cam.size[0] and 0 <= pixel[1] < cam.size[1]):
                continue
            ep.projected_views[view] = (round(float(pixel[0]), 1), round(float(pixel[1]), 1))
            for hand in probes:
                for det in det_index.get((view, ep.start_frame, hand), ()):
                    assert det.box_xyxy_px is not None
                    if _inside(pixel, det.box_xyxy_px):
                        ep.in_hand_views[view] = hand
                        break
                if view in ep.in_hand_views:
                    break
            inside = []
            for container in containers:
                if container == ep.object_class:
                    continue
                for det in det_index.get((view, ep.start_frame, container), ()):
                    assert det.box_xyxy_px is not None
                    if _inside(pixel, det.box_xyxy_px):
                        inside.append(container)
                        break
            if inside:
                ep.in_container_views[view] = inside
            best = None
            for det in det_index.get((view, ep.start_frame, ep.object_class), ()):
                assert det.box_xyxy_px is not None
                centre = (
                    (det.box_xyxy_px[0] + det.box_xyxy_px[2]) / 2,
                    (det.box_xyxy_px[1] + det.box_xyxy_px[3]) / 2,
                )
                d = float(np.hypot(centre[0] - pixel[0], centre[1] - pixel[1]))
                if d <= gate_px and (best is None or d < best):
                    best = d
            if best is not None:
                ep.detector_within_gate_views[view] = round(best, 1)
        held = len(ep.in_hand_views) >= min_views
        contained = len(ep.in_container_views) >= min_views
        visible = len(ep.detector_within_gate_views) >= min_views
        ep.held_in_all_projected_views = held and len(ep.in_hand_views) == len(ep.projected_views)
        if held and contained:
            ep.inferred_state = "held+contained"
        elif held:
            ep.inferred_state = "held"
        elif contained:
            ep.inferred_state = "contained"
        elif visible:
            ep.inferred_state = "detector_visible_association_miss"
        else:
            ep.inferred_state = "unexplained"
        ep.successor_tracks = successors.get(ep.track_id, [])
        ep.ambiguous_events = ambiguous.get(ep.track_id, 0)


def inventory_summary(episodes: Sequence[Episode], *, coast_timeout: int) -> dict[str, Any]:
    """Counts over the object episodes; hand (probe) tracks are counted apart, since hands are
    occluders and probes in this plan, not tracked objects."""
    probe_episodes = [ep for ep in episodes if ep.probe_class]
    episodes = [ep for ep in episodes if not ep.probe_class]
    by_state = Counter(ep.inferred_state for ep in episodes)
    by_outcome = Counter(ep.outcome for ep in episodes)
    by_class = Counter(ep.object_class for ep in episodes)
    lengths = [ep.length_frames for ep in episodes]
    reacquired_lengths = [ep.length_frames for ep in episodes if ep.outcome == "reacquired"]
    identical = [ep for ep in episodes if ep.identical_instance_class or ep.group_slot]
    identical_ambiguous = [
        ep for ep in identical if ep.ambiguous_events or any(s for s in ep.successor_tracks)
    ]
    contained_class: Counter[str] = Counter()
    for ep in episodes:
        if ep.inferred_state in ("contained", "held+contained"):
            votes = Counter(c for classes in ep.in_container_views.values() for c in classes)
            contained_class[votes.most_common(1)[0][0]] += 1
    return {
        "episodes": len(episodes),
        "tracks_with_episodes": len({ep.track_id for ep in episodes}),
        "probe_episodes_excluded": len(probe_episodes),
        "by_inferred_state": dict(sorted(by_state.items())),
        "by_outcome": dict(sorted(by_outcome.items())),
        "by_class": dict(by_class.most_common()),
        "length_frames": _percentiles(lengths),
        "length_frames_reacquired_only": _percentiles(reacquired_lengths),
        "episodes_lost_at_timeout": sum(1 for ep in episodes if ep.lost_at_timeout),
        "coast_timeout_frames": coast_timeout,
        "extension_needs": {
            "held": by_state.get("held", 0) + by_state.get("held+contained", 0),
            "held_in_all_projected_views": sum(
                1 for ep in episodes if ep.held_in_all_projected_views
            ),
            "contained": by_state.get("contained", 0) + by_state.get("held+contained", 0),
            "contained_by_container_class": dict(contained_class.most_common()),
            "group_tracks_identical_instance_episodes": len(identical),
            "group_tracks_with_ambiguity_or_possibly_same_as": len(identical_ambiguous),
            "detector_visible_association_miss": by_state.get(
                "detector_visible_association_miss", 0
            ),
            "unexplained": by_state.get("unexplained", 0),
        },
        "reacquired_latency_frames": _percentiles(
            [ep.reacquisition_latency_frames for ep in episodes if ep.outcome == "reacquired"]
        ),
        "semantics": (
            "an episode is a maximal run of coasting rows of one track id (hand tracks are "
            "counted apart as probes); a lost episode's length is the coast timeout, not the "
            "occlusion's duration; held = the last 3D position projected inside a hand box in "
            ">= 2 views at the first coasting frame (hand boxes are large in the fixed views, so "
            "this is an upper bound; held_in_all_projected_views is the stricter count); "
            "contained = inside a container's detector box in >= 2 views; "
            "detector_visible_association_miss = a same-class detector box within the "
            "association gate of the projection in >= 2 views, i.e. the detector still saw the "
            "object there and the tracker did not associate it (usually another track took the "
            "box); group candidates are episodes of identical-instance classes or group slots"
        ),
    }


def inventory_markdown(summary: dict[str, Any], episodes: Sequence[Episode], *, arm: str) -> str:
    episodes = [ep for ep in episodes if not ep.probe_class]
    lines = [
        f"# Occlusion inventory, arm ({arm})",
        "",
        f"{summary['episodes']} support-0 episodes on {summary['tracks_with_episodes']} object "
        f"tracks ({summary['probe_episodes_excluded']} hand-track episodes counted apart); "
        f"{summary['episodes_lost_at_timeout']} lost at the {summary['coast_timeout_frames']}-"
        f"frame coast timeout; re-acquired episodes last median "
        f"{summary['length_frames_reacquired_only']['median']} frames, p90 "
        f"{summary['length_frames_reacquired_only']['p90']}.",
        "",
        "| inferred state | episodes |",
        "|---|---|",
    ]
    for state, n in summary["by_inferred_state"].items():
        lines.append(f"| {state} | {n} |")
    lines += ["", "| outcome | episodes |", "|---|---|"]
    for outcome, n in summary["by_outcome"].items():
        lines.append(f"| {outcome} | {n} |")
    needs = summary["extension_needs"]
    lines += [
        "",
        "## What `p3-tracker-ext` would need",
        "",
        f"- `held`: {needs['held']} episodes (last position inside a hand box in >= 2 views; "
        f"{needs['held_in_all_projected_views']} in every view that projects)",
        f"- `contained`: {needs['contained']} episodes "
        f"(by container class: {needs['contained_by_container_class']})",
        f"- group tracks: {needs['group_tracks_identical_instance_episodes']} episodes of "
        f"identical-instance classes or group slots, of which "
        f"{needs['group_tracks_with_ambiguity_or_possibly_same_as']} ended in an ambiguity or a "
        "`possibly_same_as` successor",
        f"- association misses (detector still within the gate in >= 2 views): "
        f"{needs['detector_visible_association_miss']}; unexplained: {needs['unexplained']}",
        "",
        "## Episodes by class",
        "",
        "| class | episodes |",
        "|---|---|",
    ]
    for cls, n in summary["by_class"].items():
        lines.append(f"| {cls} | {n} |")
    longest = sorted(episodes, key=lambda ep: -ep.length_frames)[:25]
    lines += [
        "",
        "## Longest 25 episodes",
        "",
        "| track | class | start | length | outcome | state | hand views | container views | "
        "detector in gate | successors |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for ep in longest:
        lines.append(
            f"| {ep.track_id} | {ep.object_class} | {ep.start_frame} | {ep.length_frames} | "
            f"{ep.outcome} | {ep.inferred_state} | {','.join(sorted(ep.in_hand_views)) or '-'} | "
            f"{','.join(sorted(ep.in_container_views)) or '-'} | "
            f"{','.join(sorted(ep.detector_within_gate_views)) or '-'} | "
            f"{len(ep.successor_tracks)} |"
        )
    lines += ["", summary["semantics"], "", CLAIM_BOUNDARY, ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- reseed schedule


def build_reseed_schedule(
    events: Sequence[TrackEvent],
    residuals: Sequence[ResidualRow],
    schedule: dict[str, Any],
    *,
    view: str,
    frame_offset: int,
    window_frames: int,
    k: int = DEFAULT_RESEED_K,
    image_size: tuple[int, int] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Arm (d)'s schedule for one view: (c)'s seeds plus one box correction per slot per `k`
    frames from the tracker's `detector_reseed` / `handoff_reseed` events in that view. The
    slot is the SAM3 slot the track last had in the view (from `residuals.jsonl`); an event
    for a track the view never masked has no slot to correct and is counted, not invented."""
    seeds = sorted(schedule["seeds"], key=lambda s: int(s["initial_multiplex_slot"]))
    slot_of = {str(s["target"]): int(s["initial_multiplex_slot"]) for s in seeds}
    start_of = {str(s["target"]): int(s.get("start_frame", 0) or 0) for s in seeds}
    history: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for r in residuals:
        if r.view == view and r.source in SAM3_SOURCES and r.slot in slot_of:
            history[r.track_id].append((r.frame_index, r.slot))
    for h in history.values():
        h.sort()
    corrections: list[dict[str, Any]] = []
    last_by_slot: dict[str, int] = {}
    skipped: Counter[str] = Counter()
    considered = 0
    for e in sorted(events, key=lambda e: e.frame_index):
        if e.kind not in ("detector_reseed", "handoff_reseed") or e.payload.get("view") != view:
            continue
        considered += 1
        frame = e.frame_index - frame_offset
        if frame < 1 or frame >= window_frames:
            skipped["outside_window"] += 1
            continue
        slots = [slot for f, slot in history.get(e.track_id, ()) if f < e.frame_index]
        if not slots:
            skipped["no_sam3_slot_for_track_in_view"] += 1
            continue
        label = slots[-1]
        if frame <= start_of[label]:
            skipped["before_slot_start"] += 1
            continue
        if label in last_by_slot and frame - last_by_slot[label] < k:
            skipped["within_k_of_previous"] += 1
            continue
        box = [float(x) for x in e.payload["box_xyxy_px"]]
        if image_size is not None:
            box = [
                min(max(box[0], 0.0), image_size[0]),
                min(max(box[1], 0.0), image_size[1]),
                min(max(box[2], 0.0), image_size[0]),
                min(max(box[3], 0.0), image_size[1]),
            ]
        if box[2] - box[0] < 2 or box[3] - box[1] < 2:
            skipped["degenerate_box"] += 1
            continue
        last_by_slot[label] = frame
        corrections.append(
            {
                "frame_index": frame,
                "multiplex_slot": slot_of[label],
                "target": label,
                "prompt_box_xyxy_px": [round(x, 1) for x in box],
                "selected_by": (
                    "detector_reseed" if e.kind == "detector_reseed" else "track_reproject"
                ),
                "tracker_event": {
                    "track_id": e.track_id,
                    "raw_frame_index": e.frame_index,
                    "frames_missing": e.payload.get("frames_missing"),
                },
            }
        )
    corrections.sort(key=lambda c: (c["frame_index"], c["multiplex_slot"]))
    payload = {"seeds": seeds, "corrections": corrections}
    report = {
        "view": view,
        "events_considered": considered,
        "corrections": len(corrections),
        "by_kind": dict(Counter(c["selected_by"] for c in corrections)),
        "by_slot": dict(Counter(c["target"] for c in corrections)),
        "skipped": dict(skipped),
        "k_frames": k,
    }
    return payload, report


# --------------------------------------------------------------------------- sanity


def sanity_check(
    run_dir: Path,
    *,
    arm: str,
    view: str,
    frames: int,
    frame_offset: int,
    detections_dir: Path | None = None,
    trial: str | None = None,
    reference_ms: float | None = None,
    min_iou: float = SANITY_MIN_IOU,
    min_score: float = DEFAULT_MIN_SCORE,
    fpv_poses: Path | None = None,
) -> dict[str, Any]:
    """First-view check of a worker run: ms per frame against the preflight reference (within
    2x) and the mask-bbox IoU against the prompt box (box-decode) or the best same-class
    detector box (video memory) over the first `frames` analysis frames (median >= 0.8)."""
    run_dir = Path(run_dir)
    timing = worker_timing(run_dir)
    validity = None
    if view == FINEBIO_FPV_VIEW and (trial or fpv_poses is not None):
        validity = pose_validity(trial or "", fpv_poses)
    detector: list[FineBioObservation] = []
    if detections_dir is not None:
        wanted = range(frame_offset, frame_offset + frames)
        detector = detections_to_observations(
            Path(detections_dir), (view,), min_score, pose_valid=validity, frames=wanted
        )
    rows = worker_to_observations(
        run_dir, view, frame_offset, pose_valid=validity, detector_rows=detector
    )
    rows = [r for r in rows if r.frame_index < frame_offset + frames]
    det_index = detector_index(detector, min_score)
    ious: list[float] = []
    per_slot: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r.mask_bbox_px is None or r.mask_area_px is None:
            continue
        if r.box_xyxy_px is not None and r.source == "sam3_decode":
            iou = box_iou(list(r.mask_bbox_px), list(r.box_xyxy_px))
        else:
            iou, _ = best_detector_iou(
                r.mask_bbox_px, det_index.get((r.view, r.frame_index, r.object_class), ())
            )
            if iou is None:
                continue
        ious.append(iou)
        per_slot[r.slot].append(iou)
    reference = reference_ms if reference_ms is not None else SANITY_REFERENCE_MS.get(arm)
    ms = timing.get("box_decode_ms_per_prompted_frame_median") or timing.get("ms_per_frame_steady")
    summary = iou_summary(ious)
    ms_ok = ms is not None and reference is not None and ms <= 2 * reference
    iou_ok = summary["median"] is not None and summary["median"] >= min_iou
    return {
        "run_dir": str(run_dir),
        "arm": arm,
        "view": view,
        "frames_checked": frames,
        "rows_checked": len(rows),
        "masks_measured": len(ious),
        "timing": timing,
        "ms_per_frame_used": ms,
        "reference_ms": reference,
        "ms_within_2x": ms_ok,
        "iou_reference": (
            "prompt box" if arm == "b" else "best same-class detector box (score >= 0.3)"
        ),
        "mask_bbox_iou": summary,
        "per_slot_median": {
            slot: round(float(np.median(v)), 4) for slot, v in sorted(per_slot.items())
        },
        "iou_median_ge_threshold": iou_ok,
        "min_iou": min_iou,
        "passed": bool(ms_ok and iou_ok),
    }


# --------------------------------------------------------------------------- seeds annotation


def mark_plan_slots(
    seeds_dir: Path, classes: Sequence[str], *, rule: str, note: str
) -> dict[str, Any]:
    """Record that slots of `classes` were added by the plan's shortlist, not by the seed rule:
    `rule` on the slot records of `seeds.json` / `slots.json` (the original kept as
    `rule_from_seed_tool`) and a note in `seeds.md`."""
    seeds_dir = Path(seeds_dir)
    changed: dict[str, list[str]] = defaultdict(list)
    for name in ("seeds.json", "slots.json"):
        path = seeds_dir / name
        if not path.is_file():
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        for view, selected in doc.get("views", {}).items():
            for slot in selected.get("slots", []):
                if slot.get("class") in classes and slot.get("rule") != rule:
                    slot["rule_from_seed_tool"] = slot["rule"]
                    slot["rule"] = rule
                    slot["plan_driven"] = note
                    changed[name].append(f"{view}/{slot['label']}")
        doc.setdefault("plan_driven_slots", {})[rule] = {"classes": list(classes), "note": note}
        path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    md = seeds_dir / "seeds.md"
    if md.is_file():
        lines = md.read_text(encoding="utf-8").splitlines()
        out = []
        for line in lines:
            cells = line.split("|")
            if len(cells) > 6 and any(cells[3].strip().startswith(c + "#") for c in classes):
                cells[5] = f" {rule} "
                line = "|".join(cells)
            out.append(line)
        out += [
            "",
            f"## Plan-driven slots (`rule: {rule}`)",
            "",
            note,
            "",
        ]
        md.write_text("\n".join(out) + "\n", encoding="utf-8")
    return {"rule": rule, "classes": list(classes), "changed": dict(changed)}


# --------------------------------------------------------------------------- run an arm


def run_arm(
    *,
    clip: ClipWindow,
    detections_dir: Path,
    arm: str,
    output: Path,
    gates: Path | None,
    worker_root: Path | None = None,
    seeds_dir: Path | None = None,
    min_score: float = DEFAULT_MIN_SCORE,
    tracker_extra: Sequence[str] = (),
    jobs: int = 6,
    fpv_poses: Path | None = None,
    reuse_observations: bool = False,
) -> dict[str, Any]:
    """observations -> tracker -> measures + inventory under `output`. With
    `reuse_observations` an existing `observations.jsonl` (and its summary) is read back instead
    of rebuilt, so the tracker and the measures can be re-run without re-reading the masks."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    obs_path = output / "observations.jsonl"
    summary_path = output / "observations_summary.json"
    if reuse_observations and obs_path.is_file():
        rows = list(read_jsonl(obs_path, FineBioObservation))
        obs_summary = (
            json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
        )
        obs_summary["reused_observations"] = True
    else:
        rows, obs_summary = build_observations(
            clip,
            detections_dir,
            arm,
            worker_root=worker_root,
            min_score=min_score,
            jobs=jobs,
            fpv_poses=fpv_poses,
        )
        obs_summary["rows_written"] = write_observations(rows, obs_path)
    summary_path.write_text(json.dumps(obs_summary, indent=1) + "\n", encoding="utf-8")
    tracks_dir = output / "tracks"
    metrics = run_tracker_cli(
        obs_path, clip.camera_config, gates, tracks_dir, tracker_extra, fpv_poses=fpv_poses
    )
    return finish_arm(
        clip=clip,
        arm=arm,
        output=output,
        rows=rows,
        metrics=metrics,
        obs_summary=obs_summary,
        seeds_dir=seeds_dir,
        worker_root=worker_root,
        min_score=min_score,
        fpv_poses=fpv_poses,
    )


def _slot_starts(seeds_dir: Path | None, arm: str, frame_offset: int) -> dict[tuple[str, str], int]:
    """Raw start frame per (view, slot label): the box stream's first frame in (b), the
    schedule's `start_frame` in (c)/(d)."""
    if seeds_dir is None:
        return {}
    seeds_dir = Path(seeds_dir)
    starts: dict[tuple[str, str], int] = {}
    if arm == "b":
        for path in sorted((seeds_dir / "box_streams").glob("*.jsonl")):
            view = path.stem
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    for box in record["boxes"]:
                        key = (view, str(box["label"]))
                        frame = int(record["frame_index"]) + frame_offset
                        if key not in starts or frame < starts[key]:
                            starts[key] = frame
    else:
        for path in sorted((seeds_dir / "schedules").glob("*.json")):
            view = path.stem
            for seed in json.loads(path.read_text(encoding="utf-8"))["seeds"]:
                starts[(view, str(seed["target"]))] = (
                    int(seed.get("start_frame", 0) or 0) + frame_offset
                )
    return starts


def finish_arm(
    *,
    clip: ClipWindow,
    arm: str,
    output: Path,
    rows: Sequence[FineBioObservation],
    metrics: dict[str, Any],
    obs_summary: dict[str, Any],
    seeds_dir: Path | None,
    worker_root: Path | None,
    min_score: float,
    fpv_poses: Path | None = None,
) -> dict[str, Any]:
    output = Path(output)
    tracks_dir = output / "tracks"
    tracks = list(read_jsonl(tracks_dir / "tracks.jsonl", Track3D))
    events = list(read_jsonl(tracks_dir / "events.jsonl", TrackEvent))
    residuals = list(read_jsonl(tracks_dir / "residuals.jsonl", ResidualRow))
    det_index = detector_index(rows, min_score)
    config = read_camera_config(clip.camera_config)
    fixed_cams = cameras_from_config(config)
    fpv_source = resolve_fpv_source(config, fpv_poses)
    params = metrics.get("params", {})
    gates = params.get("gates", {})

    measures: dict[str, Any] = {
        "schema": SCHEMA,
        "arm": arm,
        "arm_name": ARM_NAMES[arm],
        "clip_id": clip.clip_id,
        "trial": clip.trial,
        "window": [clip.start_frame, clip.end_frame_exclusive],
        "observations": obs_summary,
        "identity": identity_summary(metrics, clip.probes),
        "residual_px": residual_measures(residuals),
        "residual_px_by_class_and_view": metrics.get("residual_px_by_class_and_view"),
        "gates": gates,
        "tracker_params": {k: v for k, v in params.items() if k != "gates"},
        "claim_boundary": CLAIM_BOUNDARY,
        "licence_note": LICENCE_NOTE,
    }
    if arm != "a":
        measures["slots"] = slot_measures(
            rows,
            det_index,
            residuals,
            window=clip.frames,
            slot_start=_slot_starts(seeds_dir, arm, clip.start_frame),
        )
        if worker_root is not None:
            measures["worker_timing"] = {
                view: worker_timing(find_worker_run(Path(worker_root) / view))
                for view in clip.views
            }
    episodes = episodes_from_tracks(tracks)
    annotate_episodes(
        episodes,
        det_index=det_index,
        events=events,
        fixed_cams=fixed_cams,
        fpv_source=fpv_source,
        containers=clip.containers,
        probes=clip.probes,
        gate_px=float(gates.get("association_px", 30.0)),
    )
    inv_summary = inventory_summary(
        episodes, coast_timeout=int(params.get("coast_timeout_frames", 30))
    )
    measures["occlusion_inventory"] = inv_summary
    with (output / "occlusion_inventory.jsonl").open("w", encoding="utf-8") as handle:
        for ep in sorted(episodes, key=lambda ep: (ep.start_frame, ep.track_id)):
            handle.write(json.dumps(ep.to_record()) + "\n")
    (output / "occlusion_inventory.md").write_text(
        inventory_markdown(inv_summary, episodes, arm=arm), encoding="utf-8"
    )
    (output / "measures.json").write_text(json.dumps(measures, indent=1) + "\n", encoding="utf-8")
    (output / "measures.md").write_text(measures_markdown(measures), encoding="utf-8")
    return measures


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def measures_markdown(m: dict[str, Any]) -> str:
    lines = [
        f"# Arm ({m['arm']}): {m['arm_name']}",
        "",
        f"Trial {m['trial']}, raw frames [{m['window'][0]}, {m['window'][1]}); gates "
        f"association {_fmt(m['gates'].get('association_px'), 1)} px, hand-off "
        f"{_fmt(m['gates'].get('handoff_px'), 1)} px.",
        "",
        "## Identity metrics (tracker)",
        "",
        "| measure | value |",
        "|---|---|",
    ]
    for key in IDENTITY_KEYS:
        if key == "events":
            continue
        lines.append(f"| {key} | {m['identity'].get(key)} |")
    lines.append(f"| events | {m['identity'].get('events')} |")
    lines.append(f"| objects only (hands excluded) | {m['identity'].get('objects_only')} |")
    res = m["residual_px"]
    lines += [
        "",
        "## Cross-view residual (px, per associated observation)",
        "",
        "| view | n | median | p90 |",
        "|---|---|---|---|",
    ]
    for view, stats in res["per_view"].items():
        lines.append(
            f"| {view} | {stats['n']} | {_fmt(stats['median'], 1)} | {_fmt(stats['p90'], 1)} |"
        )
    lines.append(
        f"| pooled | {res['pooled']['n']} | {_fmt(res['pooled']['median'], 1)} | "
        f"{_fmt(res['pooled']['p90'], 1)} |"
    )
    if "slots" in m:
        slots = m["slots"]
        lines += [
            "",
            "## Label-free measures per view (SAM3 rows)",
            "",
            "| view | slots | frames with mask | det-box IoU median | p10 | frac >= 0.5 | n | "
            "associated by tracker |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for view, stats in slots["per_view"].items():
            iou = stats["detector_iou_pooled"]
            lines.append(
                f"| {view} | {stats['slots']} | {stats['frames_with_mask_total']} | "
                f"{_fmt(iou['median'])} | {_fmt(iou['p10'])} | {_fmt(iou['fraction_ge_0p5'])} | "
                f"{iou['n']} | {_fmt(stats['associated_fraction'])} |"
            )
        pooled = slots["pooled"]["detector_iou"]
        lines.append(
            f"| **pooled** | {slots['pooled']['slots']} | "
            f"{slots['pooled']['frames_with_mask_total']} | {_fmt(pooled['median'])} | "
            f"{_fmt(pooled['p10'])} | {_fmt(pooled['fraction_ge_0p5'])} | {pooled['n']} | - |"
        )
        lines += [
            "",
            "## Per slot",
            "",
            "| view | slot | frames with mask / from start | class detected, no mask | "
            "det-box IoU median | p10 | frac >= 0.5 | SAM3 score median | area step median | "
            "jumps > 0.5 | associated |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for s in slots["per_slot"]:
            iou = s["detector_iou"]
            lines.append(
                f"| {s['view']} | {s['slot']} | {s['frames_with_mask']} / "
                f"{s['frames_in_window_from_start']} | "
                f"{s['frames_class_detected_without_mask'] if not s['group'] else '-'} | "
                f"{_fmt(iou['median'])} | {_fmt(iou['p10'])} | {_fmt(iou['fraction_ge_0p5'])} | "
                f"{_fmt(s['sam3_object_score']['median'])} | "
                f"{_fmt(s['area']['relative_step_median'])} | "
                f"{_fmt(s['area']['jump_fraction_gt_0p5'])} | {_fmt(s['associated_fraction'])} |"
            )
        lines += ["", slots["reference"]]
    if "worker_timing" in m:
        lines += [
            "",
            "## Worker timing",
            "",
            "| view | frames | elapsed s | ms/frame (elapsed) | ms/frame (steady) | "
            "ms/prompted frame (box decode) | peak VRAM GiB |",
            "|---|---|---|---|---|---|---|",
        ]
        for view, t in m["worker_timing"].items():
            lines.append(
                f"| {view} | {t['frames_processed']} | {t['elapsed_seconds']} | "
                f"{_fmt(t.get('ms_per_frame_elapsed'), 1)} | "
                f"{_fmt(t.get('ms_per_frame_steady'), 1)} | "
                f"{_fmt(t.get('box_decode_ms_per_prompted_frame_median'), 1)} | "
                f"{_fmt(t.get('gpu_peak_vram_gib'), 2)} |"
            )
    inv = m["occlusion_inventory"]
    lines += [
        "",
        "## Occlusion inventory (summary; details in occlusion_inventory.md)",
        "",
        f"{inv['episodes']} episodes on {inv['tracks_with_episodes']} tracks; by state "
        f"{inv['by_inferred_state']}; by outcome {inv['by_outcome']}; extension needs "
        f"{inv['extension_needs']}.",
        "",
        m["claim_boundary"],
        "",
        m["licence_note"],
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- scoreboard


def pooled_decision(
    measures_c: dict[str, Any], measures_b: dict[str, Any], *, margin: float = DECISION_MARGIN
) -> dict[str, Any]:
    """The plan's rule for arm (d): (c) must beat (b) by `margin` on the pooled detector-box
    IoU, or clearly on identity (fewer id switches, fragmentation and ambiguities, none worse)."""
    iou_c = measures_c["slots"]["pooled"]["detector_iou"]["median"]
    iou_b = measures_b["slots"]["pooled"]["detector_iou"]["median"]
    frac_c = measures_c["slots"]["pooled"]["detector_iou"]["fraction_ge_0p5"]
    frac_b = measures_b["slots"]["pooled"]["detector_iou"]["fraction_ge_0p5"]
    ident_c, ident_b = measures_c["identity"], measures_b["identity"]
    keys = ("id_switches", "fragmentation", "ambiguities")
    better = {k: (ident_c.get(k) or 0) < (ident_b.get(k) or 0) for k in keys}
    worse = {k: (ident_c.get(k) or 0) > (ident_b.get(k) or 0) for k in keys}
    identity_clearly_better = all(better.values()) and not any(worse.values())
    iou_delta = None if iou_c is None or iou_b is None else round(iou_c - iou_b, 4)
    beats_on_iou = iou_delta is not None and iou_delta >= margin
    return {
        "pooled_detector_iou_median": {"b": iou_b, "c": iou_c, "delta_c_minus_b": iou_delta},
        "pooled_fraction_ge_0p5": {"b": frac_b, "c": frac_c},
        "identity": {
            "b": {k: ident_b.get(k) for k in keys},
            "c": {k: ident_c.get(k) for k in keys},
            "c_better": better,
            "c_worse": worse,
        },
        "margin": margin,
        "c_beats_b_on_iou": beats_on_iou,
        "c_clearly_better_on_identity": identity_clearly_better,
        "run_arm_d": bool(beats_on_iou or identity_clearly_better),
        "rule": (
            "arm (d) runs only if (c) beats (b) by >= margin on the pooled detector-box IoU "
            "median, or is better on all of id switches, fragmentation and ambiguities and "
            "worse on none"
        ),
    }


def scoreboard(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Arms x measures, per view and pooled, from each arm's measures.json."""
    views = sorted({v for m in arms.values() for v in m["residual_px"]["per_view"]})
    table: dict[str, Any] = {"arms": sorted(arms), "views": views, "rows": {}}
    for arm, m in sorted(arms.items()):
        row: dict[str, Any] = {
            "identity": m["identity"],
            "residual_pooled": m["residual_px"]["pooled"],
        }
        row["residual_per_view"] = {
            v: m["residual_px"]["per_view"].get(v, {}).get("median") for v in views
        }
        if "slots" in m:
            row["det_iou_pooled"] = m["slots"]["pooled"]["detector_iou"]
            row["det_iou_per_view"] = {
                v: m["slots"]["per_view"].get(v, {}).get("detector_iou_pooled", {}).get("median")
                for v in views
            }
            row["frac_ge_0p5_per_view"] = {
                v: m["slots"]["per_view"]
                .get(v, {})
                .get("detector_iou_pooled", {})
                .get("fraction_ge_0p5")
                for v in views
            }
            row["frames_with_mask_per_view"] = {
                v: m["slots"]["per_view"].get(v, {}).get("frames_with_mask_total") for v in views
            }
        if "worker_timing" in m:
            row["worker_elapsed_seconds"] = {
                v: m["worker_timing"].get(v, {}).get("elapsed_seconds") for v in views
            }
            row["worker_peak_vram_gib"] = {
                v: m["worker_timing"].get(v, {}).get("gpu_peak_vram_gib") for v in views
            }
        row["occlusion_inventory"] = m["occlusion_inventory"]["extension_needs"]
        row["occlusion_episodes"] = m["occlusion_inventory"]["episodes"]
        table["rows"][arm] = row
    return table


def scoreboard_markdown(board: dict[str, Any]) -> str:
    views = board["views"]
    lines = [
        "| arm | det-box IoU pooled median | p10 | frac >= 0.5 | n | "
        + " | ".join(f"{v} IoU" for v in views)
        + " |",
        "|---|---|---|---|---|" + "---|" * len(views),
    ]
    for arm, row in board["rows"].items():
        iou = row.get("det_iou_pooled")
        if iou is None:
            lines.append(f"| ({arm}) | - | - | - | - |" + " - |" * len(views))
            continue
        lines.append(
            f"| ({arm}) | {_fmt(iou['median'])} | {_fmt(iou['p10'])} | "
            f"{_fmt(iou['fraction_ge_0p5'])} | {iou['n']} | "
            + " | ".join(_fmt(row["det_iou_per_view"].get(v)) for v in views)
            + " |"
        )
    lines += [
        "",
        "| arm | frac >= 0.5 pooled | " + " | ".join(f"{v} frac >= 0.5" for v in views) + " | "
        "frames with mask pooled | " + " | ".join(f"{v} masks" for v in views) + " |",
        "|---|---|" + "---|" * len(views) + "---|" + "---|" * len(views),
    ]
    for arm, row in board["rows"].items():
        iou = row.get("det_iou_pooled")
        if iou is None:
            continue
        lines.append(
            f"| ({arm}) | {_fmt(iou['fraction_ge_0p5'])} | "
            + " | ".join(_fmt(row["frac_ge_0p5_per_view"].get(v)) for v in views)
            + f" | {sum(x or 0 for x in row['frames_with_mask_per_view'].values())} | "
            + " | ".join(str(row["frames_with_mask_per_view"].get(v)) for v in views)
            + " |"
        )
    lines += [
        "",
        "| arm | tracks born | lost | reacquired | latency median | ambiguities | fragmentation | "
        "id switches (SAM3-slot proxy) | slot disagreements | residual pooled median px | "
        + " | ".join(f"{v} px" for v in views)
        + " |",
        "|---|---|---|---|---|---|---|---|---|---|" + "---|" * len(views),
    ]
    for arm, row in board["rows"].items():
        ident = row["identity"]
        lat = (ident.get("reacquisition_latency_frames") or {}).get("median")
        lines.append(
            f"| ({arm}) | {ident.get('tracks_born')} | {ident.get('tracks_lost')} | "
            f"{ident.get('reacquisitions')} | {_fmt(lat, 1)} | {ident.get('ambiguities')} | "
            f"{ident.get('fragmentation')} | {ident.get('id_switches')} | "
            f"{ident.get('slot_disagreements')} | {_fmt(row['residual_pooled']['median'], 1)} | "
            + " | ".join(_fmt(row["residual_per_view"].get(v), 1) for v in views)
            + " |"
        )
    lines += [
        "",
        "| arm | episodes | held | contained | group candidates | association misses | "
        "unexplained |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm, row in board["rows"].items():
        needs = row["occlusion_inventory"]
        lines.append(
            f"| ({arm}) | {row['occlusion_episodes']} | {needs['held']} | {needs['contained']} | "
            f"{needs['group_tracks_identical_instance_episodes']} | "
            f"{needs['detector_visible_association_miss']} | {needs['unexplained']} |"
        )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- cli


def _add_clip_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--clip-config", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True, help="DINO JSONL directory")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="observations -> tracker -> measures -> inventory")
    _add_clip_arguments(run)
    run.add_argument("--arm", choices=ARMS, required=True)
    run.add_argument("--worker-root", type=Path, default=None, help="<root>/<view>/<run>")
    run.add_argument("--seeds", type=Path, default=None, help="seed run dir (slot starts)")
    run.add_argument("--gates", type=Path, default=None, help="rig.json with a gates block")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--jobs", type=int, default=6)
    run.add_argument(
        "--tracker-arg", action="append", default=[], help="extra battle-multiview-tracks flag"
    )
    run.add_argument(
        "--fpv-poses", type=Path, default=None, help="fixtures-shaped fpv_poses.json (tests)"
    )
    run.add_argument(
        "--reuse-observations",
        action="store_true",
        help="read <output>/observations.jsonl instead of rebuilding it",
    )

    sanity = sub.add_parser("sanity", help="first-view check of a worker run")
    sanity.add_argument("--run", type=Path, required=True, help="worker run dir or its root")
    sanity.add_argument("--arm", choices=("b", "c", "d"), required=True)
    sanity.add_argument("--view", required=True)
    sanity.add_argument("--frames", type=int, default=100)
    sanity.add_argument("--clip-config", type=Path, required=True)
    sanity.add_argument("--detections", type=Path, default=None)
    sanity.add_argument("--reference-ms", type=float, default=None)
    sanity.add_argument("--output", type=Path, default=None)

    reseed = sub.add_parser("reseed-schedule", help="arm (d) schedules from arm (c)'s events")
    reseed.add_argument("--arm-dir", type=Path, required=True, help="arm (c) output directory")
    reseed.add_argument("--seeds", type=Path, required=True, help="seed run dir (schedules/)")
    reseed.add_argument("--clip-config", type=Path, required=True)
    reseed.add_argument("--output", type=Path, required=True)
    reseed.add_argument("--k", type=int, default=DEFAULT_RESEED_K)

    mark = sub.add_parser("mark-plan-slots", help="record plan-driven slots in a seed run")
    mark.add_argument("--seeds", type=Path, required=True)
    mark.add_argument("--classes", required=True, help="comma-separated classes")
    mark.add_argument("--rule", default="landmark_plan_shortlist")
    mark.add_argument("--note", required=True)

    board = sub.add_parser("scoreboard", help="arms x measures from measures.json files")
    board.add_argument("--arm", action="append", required=True, help="<letter>=<arm dir>")
    board.add_argument("--output", type=Path, required=True)

    decide = sub.add_parser("decide", help="the (c)-vs-(b) rule for arm (d)")
    decide.add_argument("--b", type=Path, required=True)
    decide.add_argument("--c", type=Path, required=True)
    decide.add_argument("--output", type=Path, required=True)
    return parser


def _load_measures(arm_dir: Path) -> dict[str, Any]:
    return json.loads((Path(arm_dir) / "measures.json").read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        clip = load_clip(args.clip_config)
        measures = run_arm(
            clip=clip,
            detections_dir=args.detections,
            arm=args.arm,
            output=args.output,
            gates=args.gates,
            worker_root=args.worker_root,
            seeds_dir=args.seeds,
            min_score=args.min_score,
            tracker_extra=args.tracker_arg,
            jobs=args.jobs,
            fpv_poses=args.fpv_poses,
            reuse_observations=args.reuse_observations,
        )
        print(
            json.dumps(
                {
                    "identity": measures["identity"],
                    "inventory": measures["occlusion_inventory"]["extension_needs"],
                },
                indent=1,
            )
        )
        return 0
    if args.command == "sanity":
        clip = load_clip(args.clip_config)
        result = sanity_check(
            find_worker_run(args.run),
            arm=args.arm,
            view=args.view,
            frames=args.frames,
            frame_offset=clip.start_frame,
            detections_dir=args.detections,
            trial=clip.trial,
            reference_ms=args.reference_ms,
        )
        text = json.dumps(result, indent=1) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
        print(text)
        return 0 if result["passed"] else 2
    if args.command == "reseed-schedule":
        clip = load_clip(args.clip_config)
        tracks_dir = Path(args.arm_dir) / "tracks"
        events = list(read_jsonl(tracks_dir / "events.jsonl", TrackEvent))
        residuals = list(read_jsonl(tracks_dir / "residuals.jsonl", ResidualRow))
        config = read_camera_config(clip.camera_config)
        sizes = {v: tuple(c.image_size) for v, c in config.fixed.items()}
        sizes[clip.fpv_view] = tuple(config.fpv.image_size)
        args.output.mkdir(parents=True, exist_ok=True)
        reports = {}
        for view in clip.views:
            schedule = json.loads(
                (Path(args.seeds) / "schedules" / f"{view}.json").read_text(encoding="utf-8")
            )
            payload, report = build_reseed_schedule(
                events,
                residuals,
                schedule,
                view=view,
                frame_offset=clip.start_frame,
                window_frames=clip.end_frame_exclusive - clip.start_frame,
                k=args.k,
                image_size=sizes.get(view),
            )
            (args.output / f"{view}.json").write_text(
                json.dumps(payload, indent=1) + "\n", encoding="utf-8"
            )
            reports[view] = report
        (args.output / "reseed_report.json").write_text(
            json.dumps(reports, indent=1) + "\n", encoding="utf-8"
        )
        print(json.dumps(reports, indent=1))
        return 0
    if args.command == "mark-plan-slots":
        report = mark_plan_slots(
            args.seeds,
            tuple(c for c in args.classes.split(",") if c),
            rule=args.rule,
            note=args.note,
        )
        print(json.dumps(report, indent=1))
        return 0
    if args.command == "scoreboard":
        arms = {}
        for spec in args.arm:
            letter, _, path = spec.partition("=")
            arms[letter] = _load_measures(Path(path))
        board = scoreboard(arms)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "scoreboard.json").write_text(
            json.dumps(board, indent=1) + "\n", encoding="utf-8"
        )
        (args.output / "scoreboard.md").write_text(scoreboard_markdown(board), encoding="utf-8")
        print(scoreboard_markdown(board))
        return 0
    if args.command == "decide":
        decision = pooled_decision(_load_measures(args.c), _load_measures(args.b))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(decision, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(decision, indent=1))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
