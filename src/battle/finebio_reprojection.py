"""A second, memory-free SAM3 pass prompted by existing 3D pipette segments.

Streams use proxy frame indices and dense integer slots; the manifest keeps raw frames,
colour classes and 3D track ids. Existing observations are copied as bytes on merge.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .finebio_cameras import Camera
from .finebio_observations import AXIS_ELONGATION_THRESHOLD, worker_to_observations
from .multiview_schemas import FineBioObservation, Track3D
from .multiview_tracks import SAM3_SOURCES, LinePrior

PIPETTE_CLASSES = ("blue_pipette", "yellow_pipette", "red_pipette", "8_channel_pipette")
MAX_FILL_FRACTION = 0.8
MAX_AXIS_RESIDUAL_PX = 25.0
GRAVITY_MIN_CONFIDENCE = 0.8


def pipette_rows(path: Path) -> list[FineBioObservation]:
    source = path / "observations.jsonl" if path.is_dir() else path
    rows = []
    with source.open() as handle:
        for line in handle:
            if "pipette" in line:
                row = FineBioObservation.model_validate_json(line)
                if row.object_class in PIPETTE_CLASSES:
                    rows.append(row)
    return rows


def line_rows(path: Path) -> list[Track3D]:
    if path.is_dir():
        if (path / "tracks-lines").is_dir():
            path = path / "tracks-lines"
        path = path / "tracks.jsonl"
    oriented = path.with_name("tracks_oriented.jsonl")
    if oriented.is_file():
        path = oriented
    with path.open() as handle:
        return [Track3D.model_validate_json(line) for line in handle if '"endpoints_cm"' in line]


def segment_distance(points: np.ndarray, segment: np.ndarray) -> np.ndarray:
    delta = segment[1] - segment[0]
    squared = float(delta @ delta)
    if squared < 1e-9:
        return np.linalg.norm(points - segment[0], axis=1)
    t = np.clip((points - segment[0]) @ delta / squared, 0, 1)
    return np.linalg.norm(points - segment[0] - t[:, None] * delta, axis=1)


def perpendicular_residual(cam: Camera, axis: Sequence, segment: Sequence) -> float:
    ends = cam.undistort(np.asarray(segment, dtype=float))
    points = cam.undistort(np.asarray(axis, dtype=float))
    delta = ends[1] - ends[0]
    length = float(np.linalg.norm(delta))
    if length < 1e-9:
        return float("inf")
    normal = np.array([-delta[1], delta[0]]) / length
    return float(np.mean(np.abs((points - ends[0]) @ normal)))


def projected_box(cam: Camera, ends: np.ndarray, width_px: float, shift_px: float = 0):
    depth = (cam.R @ ends.T).T[:, 2] + cam.tvec[2]
    if np.any(depth <= 1) or not np.all(np.isfinite(ends)):
        return None
    pixels = cam.project(ends)
    delta = pixels[1] - pixels[0]
    length = float(np.linalg.norm(delta))
    if not np.all(np.isfinite(pixels)) or length < 40:
        return None
    along = delta / length
    across = np.array([-along[1], along[0]])
    shifted = pixels + across * shift_px
    corners = np.array(
        [
            point + sign * across * 1.5 * width_px
            for point in (shifted[0] - delta * 0.1, shifted[1] + delta * 0.1)
            for sign in (-1, 1)
        ]
    )
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    clipped_lo, clipped_hi = np.maximum(lo, 0), np.minimum(hi, cam.size)
    area = float(np.prod(hi - lo))
    if np.any(clipped_hi <= clipped_lo) or float(np.prod(clipped_hi - clipped_lo)) < 0.8 * area:
        return None
    return [round(float(x), 3) for x in (*clipped_lo, *clipped_hi)], pixels.tolist()


def generate_requests(
    tracks: Sequence[Track3D],
    observations: Sequence[FineBioObservation],
    cameras_at: Callable[[int], dict[str, Camera]],
    *,
    views: Sequence[str],
    start_frame: int,
    gate_px: float,
    prior: LinePrior,
    output: Path,
    shift_px: float = 0,
) -> dict[str, Any]:
    widths = defaultdict(list)
    present = defaultdict(list)
    for row in observations:
        if row.source in SAM3_SOURCES and row.mask_area_px:
            present[(row.view, row.frame_index, row.object_class)].append(row)
            if row.mask_width_px:
                widths[(row.view, row.object_class)].append(row.mask_width_px)
    medians = {key: float(np.median(items)) for key, items in widths.items()}
    candidates = []
    skipped = Counter()
    held_frames = set()
    cam_cache: dict[int, dict[str, Camera]] = {}
    for track in tracks:
        cls = track.observed_class or track.object_class
        if cls not in PIPETTE_CLASSES or track.endpoints_cm is None or track.state == "lost":
            continue
        frame = track.frame_index
        held_key = (track.track_id, frame)
        if track.state == "held":
            held_frames.add(held_key)
        ends = np.asarray(track.endpoints_cm, dtype=float).copy()
        length = float(np.linalg.norm(ends[1] - ends[0]))
        if length < 1e-6:
            continue
        confidence = 0.5 if track.tip_confidence is None else float(track.tip_confidence)
        if track.orientation_retrofit_sign == -1:
            confidence = 1 - confidence
        elif track.orientation_retrofit_sign != 1:
            confidence = max(confidence, 1 - confidence)
        if track.tip_attached and track.tip_resolved and confidence >= 0.9:
            target = prior.expected_length(cls, track.tip_class, attached=True)
            if target > length:
                ends[0] += (ends[0] - ends[1]) / length * (target - length)
        if frame not in cam_cache:
            cam_cache[frame] = cameras_at(frame)
        for view in views:
            cam = cam_cache[frame].get(view)
            if cam is None:
                skipped["invalid_pose"] += 1
                continue
            result = projected_box(cam, ends, medians.get((view, cls), 15.0), shift_px)
            if result is None:
                skipped["short_or_outside"] += 1
                continue
            box, pixels = result
            # Missing means no existing mask near this segment, not just no detector box.
            near = False
            for row in present.get((view, frame, cls), ()):
                point = row.mask_centroid_px or row.point_px
                if float(segment_distance(np.asarray([point]), np.asarray(pixels))[0]) <= gate_px:
                    near = True
                    break
            if near:
                skipped["existing_mask"] += 1
                continue
            candidates.append(
                {
                    "view": view,
                    "raw_frame": frame,
                    "track_id": track.track_id,
                    "object_class": cls,
                    "label": f"{cls}#{track.track_id}",
                    "box_xyxy_px": box,
                    "projected_axis_px": pixels,
                    "state": track.state,
                }
            )
    output.mkdir(parents=True, exist_ok=True)
    # Only one prompt for a view/class/frame: the merge cannot add duplicate class rows.
    requests = {}
    for item in sorted(
        candidates, key=lambda row: (row["view"], row["raw_frame"], row["track_id"])
    ):
        key = (item["view"], item["raw_frame"], item["object_class"])
        if key in requests:
            skipped["duplicate_class_prompt"] += 1
        else:
            requests[key] = item
    touched_held = {
        (row["track_id"], row["raw_frame"]) for row in requests.values() if row["state"] == "held"
    }
    per_view = {}
    for view in views:
        items = [row for row in requests.values() if row["view"] == view]
        labels = sorted({row["label"] for row in items})
        slots = {label: i for i, label in enumerate(labels)}
        frames = defaultdict(list)
        for item in items:
            item["slot"] = slots[item["label"]]
            frames[item["raw_frame"] - start_frame].append(
                {
                    "slot": item["slot"],
                    "label": item["label"],
                    "box_xyxy_px": item["box_xyxy_px"],
                    "source": "reprojection",
                }
            )
        with (output / f"{view}.jsonl").open("w") as handle:
            for frame, boxes in sorted(frames.items()):
                handle.write(json.dumps({"frame_index": frame, "boxes": boxes}) + "\n")
        per_view[view] = {
            "requests": len(items),
            "prompted_frames": len(frames),
            "slots": len(labels),
        }
    with (output / "requests.jsonl").open("w") as handle:
        for item in requests.values():
            handle.write(json.dumps(item) + "\n")
    report = {
        "schema": "battle-finebio-reprojection/1",
        "start_frame": start_frame,
        "gate_px": gate_px,
        "shift_px": shift_px,
        "per_view": per_view,
        "requests": len(requests),
        "skipped": dict(skipped),
        "held_track_frames": len(held_frames),
        "held_track_frames_touched": len(touched_held),
        "held_fraction_touched": len(touched_held) / len(held_frames) if held_frames else None,
        "held_coverage_basis": "selected requests after class/frame deduplication",
        "median_width_px": {f"{v}/{c}": x for (v, c), x in medians.items()},
    }
    (output / "requests_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def quality_rejection(
    row: FineBioObservation, box: Sequence[float], *, fill_max: float = MAX_FILL_FRACTION
):
    area = (box[2] - box[0]) * (box[3] - box[1])
    fill = float(row.mask_area_px or 0) / max(1, area)
    # Check filling first so the explicit fill-fraction control has its own rejection count.
    if fill >= fill_max:
        return "fills_prompt_box", fill
    if row.mask_axis_px is None or (row.mask_elongation or 0) < AXIS_ELONGATION_THRESHOLD:
        return "not_elongated", fill
    if row.mask_axis_residual_px is None or row.mask_axis_residual_px > MAX_AXIS_RESIDUAL_PX:
        return "axis_residual", fill
    if not row.mask_area_px:
        return "empty", fill
    return None, fill


def merge_rows(
    source: Path,
    additions: Sequence[FineBioObservation],
    requests: Sequence[dict],
    cameras_at: Callable[[int], dict[str, Camera]],
    *,
    output: Path,
    gate_px: float,
) -> dict[str, Any]:
    source = source / "observations.jsonl" if source.is_dir() else source
    if output.resolve() == source.parent.resolve():
        raise ValueError("merge output must differ from the source observations directory")
    existing = {
        (row.view, row.frame_index, row.object_class)
        for row in pipette_rows(source)
        if row.source in SAM3_SOURCES
    }
    index = {(row["view"], row["raw_frame"], row["label"]): row for row in requests}
    accepted, audits = [], []
    counters = Counter()
    for row in additions:
        key = (row.view, row.frame_index, row.slot)
        request = index.get(key)
        if request is None:
            counters["unknown_request"] += 1
            continue
        reject, fill = quality_rejection(row, request["box_xyxy_px"])
        class_key = (row.view, row.frame_index, row.object_class)
        if reject is None and class_key in existing:
            reject = "existing_class_row"
        cam = cameras_at(row.frame_index).get(row.view)
        residual = None
        if reject is None and cam is None:
            reject = "invalid_pose"
        if reject is None:
            residual = perpendicular_residual(cam, row.mask_axis_px, request["projected_axis_px"])
            provenance = {
                **row.provenance,
                "prompt_source": "reprojection",
                "prompt_track_id": request["track_id"],
                "prompt_fill_fraction": round(fill, 4),
                "prompt_line_residual_px": round(residual, 3),
                "reprojection_disagrees": residual > gate_px,
            }
            accepted.append(row.model_copy(update={"provenance": provenance}))
            existing.add(class_key)
            counters["accepted"] += 1
            counters["disagrees"] += int(residual > gate_px)
        else:
            counters[reject] += 1
        audits.append(
            {
                "view": row.view,
                "raw_frame": row.frame_index,
                "slot": row.slot,
                "accepted": reject is None,
                "rejection": reject,
                "fill_fraction": round(fill, 4),
                "residual_px": residual,
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    target = output / "observations.jsonl"
    shutil.copyfile(source, target)
    with target.open("ab") as handle:
        if target.stat().st_size:
            with source.open("rb") as old:
                old.seek(-1, 2)
                if old.read(1) != b"\n":
                    handle.write(b"\n")
        for row in accepted:
            handle.write(row.model_dump_json(exclude_none=True).encode() + b"\n")
    # Sample only these new masks when extending the cached plunger-colour sidecar.
    with (output / "reprojection_observations.jsonl").open("w") as handle:
        for row in accepted:
            handle.write(row.model_dump_json(exclude_none=True) + "\n")
    with (output / "decoded_reprojection.jsonl").open("w") as handle:
        for row in additions:
            handle.write(row.model_dump_json(exclude_none=True) + "\n")
    sidecar = source.with_name("plunger_ends.jsonl")
    if sidecar.is_file():
        shutil.copyfile(sidecar, output / sidecar.name)
    with (output / "merge_audit.jsonl").open("w") as handle:
        for row in audits:
            handle.write(json.dumps(row) + "\n")
    report = {
        "schema": "battle-finebio-reprojection-merge/1",
        "source": str(source),
        "old_rows_byte_identical": True,
        "returned_masks": len(additions),
        "requests": len(requests),
        "counts": dict(counters),
        "quality_gates": {
            "fill_fraction_max": MAX_FILL_FRACTION,
            "axis_residual_max_px": MAX_AXIS_RESIDUAL_PX,
            "elongation_min": AXIS_ELONGATION_THRESHOLD,
        },
        "gate_px": gate_px,
        "output": str(output),
    }
    (output / "merge_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    summary_path = source.with_name("observations_summary.json")
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    summary["reprojection"] = report
    (output / "observations_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return report


def decode_rows(worker_root: Path, requests_dir: Path, start_frame: int, cameras_at):
    from .finebio_arms import find_worker_run

    requests = [
        json.loads(line) for line in (requests_dir / "requests.jsonl").read_text().splitlines()
    ]
    rows = []
    index = {(row["view"], row["raw_frame"], row["label"]): row for row in requests}
    seen = set()
    for view in sorted({row["view"] for row in requests}):
        root = worker_root / view
        runs = []
        for candidate in sorted(root.iterdir()):
            manifest = candidate / "manifest.json"
            if not manifest.is_file() or not (candidate / "observations.jsonl").is_file():
                continue
            statuses = json.loads(manifest.read_text()).get("method_statuses", [])
            if statuses and all(status["state"] == "succeeded" for status in statuses):
                runs.append(candidate)
        if not runs:
            runs = [find_worker_run(root)]
        for run in runs:
            masks = {}
            prompt_boxes = {}
            for frame in map(json.loads, (run / "observations.jsonl").read_text().splitlines()):
                for obj in frame["objects"]:
                    key = (frame["analysis_frame_index"] + start_frame, obj["label"])
                    if obj.get("mask"):
                        masks[key] = obj["mask"]["uri"]
                    box = obj.get("prompt_box")
                    if box is not None:
                        prompt_boxes[key] = np.array(
                            [box["x"], box["y"], box["x"] + box["width"], box["y"] + box["height"]]
                        )
            # Box streams already carry the stable colour#track label. Integer slots may
            # differ in a corrective sparse run, so resolve its own labels, not another run's.
            decoded = worker_to_observations(
                run, view, start_frame, pose_valid=lambda frame: "fpv" in cameras_at(frame)
            )
            for row in decoded:
                key = (view, row.frame_index, row.slot)
                request = index.get(key)
                prompt = prompt_boxes.get((row.frame_index, row.slot))
                cam = cameras_at(row.frame_index).get(view)
                if request is None or key in seen or prompt is None or cam is None:
                    continue
                # A changed prompt is decoded again. Reuse a cached result only for the
                # exact same raw box (2 milli-pixel tolerance for serialization). The
                # adapter rounds display boxes to tenths, so do not compare those.
                if not np.allclose(
                    prompt * np.tile(cam.size, 2), request["box_xyxy_px"], atol=0.002, rtol=0
                ):
                    continue
                seen.add(key)
                rows.append(
                    row.model_copy(
                        update={
                            "provenance": {
                                **row.provenance,
                                "reprojection_worker_run": str(run),
                                "reprojection_worker_mask_uri": masks.get(
                                    (row.frame_index, row.slot)
                                ),
                            }
                        }
                    )
                )
    return rows, requests


def gravity_segment(
    cam: Camera,
    axis_px: np.ndarray,
    predicted_ends: np.ndarray,
    length_cm: float,
    hand_point: np.ndarray | None = None,
    *,
    min_view_angle_deg: float = 10,
) -> np.ndarray | None:
    """A steep single-view fit with a soft gravity direction and optional butt point.

    A shallow prediction abstains. Without a hand, the two endpoint rays plus the known
    length set depth; with a hand, its point projected onto the axis plane pins the butt.
    The result is kept in the prediction's continuous endpoint order.
    """
    from .finebio_orientation import gravity_cue
    from .finebio_stand import BENCH_UP
    from .multiview_lines import Line3D, plane_from_axis, ray_from_point, view_angle_to_line_deg

    cue = gravity_cue(predicted_ends)
    if cue is None or cue.confidence < GRAVITY_MIN_CONFIDENCE:
        return None
    try:
        normal, offset = plane_from_axis(cam, axis_px)
    except ValueError:
        return None
    prediction = predicted_ends[1] - predicted_ends[0]
    prediction /= np.linalg.norm(prediction)
    gravity = BENCH_UP - float(BENCH_UP @ normal) * normal
    if np.linalg.norm(gravity) < 1e-6:
        return None
    gravity /= np.linalg.norm(gravity)
    if float(gravity @ prediction) < 0:
        gravity = -gravity
    direction = cue.confidence * gravity + (1 - cue.confidence) * prediction
    direction -= float(direction @ normal) * normal
    direction /= np.linalg.norm(direction)
    if (
        view_angle_to_line_deg(cam, Line3D(predicted_ends.mean(axis=0), direction))
        < min_view_angle_deg
    ):
        return None
    # Match raw axis order to continuous track endpoints; the mask has no tip ordering.
    projected = cam.project(predicted_ends)
    if np.linalg.norm(axis_px[0] - projected[0]) > np.linalg.norm(axis_px[1] - projected[0]):
        axis_px = axis_px[::-1]
    if hand_point is not None:
        # The lower end is the tip on this steep fit; the hand anchors the upper butt.
        butt = np.asarray(hand_point, dtype=float).copy()
        butt -= (float(butt @ normal) - offset) * normal
        # Use the in-plane direction, retaining whether endpoint 0 or 1 is the butt.
        if float(direction @ BENCH_UP) > 0:
            ends = np.stack([butt - direction * length_cm, butt])
        else:
            ends = np.stack([butt, butt + direction * length_cm])
    else:
        origin0, ray0 = ray_from_point(cam, axis_px[0])
        origin1, ray1 = ray_from_point(cam, axis_px[1])
        distances, *_ = np.linalg.lstsq(
            np.column_stack([-ray0, ray1]), direction * length_cm, rcond=None
        )
        if np.any(distances <= 1):
            return None
        ends = np.stack([origin0 + ray0 * distances[0], origin1 + ray1 * distances[1]])
        midpoint = ends.mean(axis=0)
        ends = np.stack(
            [midpoint - direction * length_cm / 2, midpoint + direction * length_cm / 2]
        )
    if not np.all(np.isfinite(ends)):
        return None
    if np.any((cam.R @ ends.T).T[:, 2] + cam.tvec[2] <= 1):
        return None
    return ends
