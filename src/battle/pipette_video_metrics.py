"""Native-rate observations and per-second paired metrics for the full pipette video."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .finebio_colour import load_colour_settings, load_rig, sample_from_observation
from .finebio_observations import mask_axis_measurements
from .finebio_orientation import colour_cue, plunger_end_from_sample, taper_camera, taper_cue
from .multiview_lines import AxisObs, fit_line, loo_residual
from .pipette_line_review import (
    VARIANTS,
    cameras_at_raw_frame,
    conservative_direction,
    endpoint_correspondence,
    json_value,
)


def observe_view(root: Path, view: str) -> None:
    """Record failures so a background pipeline cannot wait forever for completion."""
    directory = root / "observations"
    directory.mkdir(exist_ok=True)
    failure = directory / f"{view}-failed.json"
    failure.unlink(missing_ok=True)
    try:
        _observe_view(root, view)
    except Exception as error:
        failure.write_text(json.dumps({"view": view, "error": repr(error)}) + "\n")
        raise


def _observe_view(root: Path, view: str) -> None:
    """Follow a worker's append-only native frames; geometry never runs on GPU."""
    config = json.loads((root / "config.json").read_text())
    record = config["views"][view]
    native = root / "native" / view
    directory = root / "observations"
    directory.mkdir(exist_ok=True)
    destination = directory / f"{view}.jsonl"
    while not (native / "frames.jsonl").exists():
        if (native / "failed.json").exists():
            raise RuntimeError(f"Inference failed before producing frames for {view}")
        time.sleep(0.5)
    capture = cv2.VideoCapture(record["video"])
    settings = load_colour_settings(Path("configs/finebio/pipettes.json"))
    with (native / "frames.jsonl").open() as source, destination.open("w") as output:
        for frame in range(record["frame_count"]):
            while True:
                offset = source.tell()
                line = source.readline()
                if line.endswith("\n"):
                    raw = json.loads(line)
                    break
                source.seek(offset)
                if (native / "failed.json").exists():
                    raise RuntimeError(f"Inference failed for {view}")
                if (native / "complete.json").exists():
                    raise RuntimeError(f"Missing frame {frame} for completed {view}")
                time.sleep(0.2)
            if raw["raw_frame"] != frame:
                raise ValueError(f"{view}: expected raw {frame}, got {raw['raw_frame']}")
            ok, image = capture.read()
            if not ok:
                raise ValueError(f"{view}: source ended at raw {frame}")
            groups = {}
            for group, prediction in raw["groups"].items():
                pixels = cv2.imread(prediction["mask"], 0)
                if pixels is None or pixels.shape != image.shape[:2]:
                    raise ValueError(f"Bad native mask at {view}/{frame}/{group}")
                mask = pixels > 0
                axis = mask_axis_measurements(mask)
                ys, xs = np.nonzero(mask)
                centroid = None if not len(xs) else [float(xs.mean() + 0.5), float(ys.mean() + 0.5)]
                sample = sample_from_observation(
                    image, {"mask_axis_px": axis.axis_px, "mask_width_px": axis.width_px}, settings
                )
                groups[group] = {
                    **prediction,
                    "axis": asdict(axis),
                    "centroid_px": centroid,
                    "colour": plunger_end_from_sample(sample, axis.width_px or 0),
                }
            output.write(
                json.dumps(json_value({"raw_frame": frame, "groups": groups}), allow_nan=False)
                + "\n"
            )
            if (frame + 1) % 30 == 0:
                output.flush()
                (directory / f"{view}-progress.json").write_text(json.dumps({"frames": frame + 1}))
    capture.release()
    (directory / f"{view}-complete.json").write_text(json.dumps({"frames": record["frame_count"]}))


def frame_geometry(rows: dict, cams: dict) -> dict:
    pairs = []
    for view, row in rows.items():
        row["camera_available"] = view in cams
        if view in cams and row["centroid_px"] is not None:
            axis = row["axis"]
            pairs.append(
                (
                    cams[view],
                    AxisObs(
                        view,
                        axis["axis_px"],
                        row["centroid_px"],
                        axis["elongation"],
                        ends_px=axis["body_ends_px"],
                    ),
                )
            )
    line = fit_line(pairs)
    colours, tapers = [], []
    for view, row in rows.items():
        row["projected_world_endpoints_px"] = None
        if line is None or line.endpoints is None or view not in cams:
            continue
        cam = cams[view]
        if np.any((line.endpoints @ cam.R.T + cam.tvec.reshape(3))[:, 2] <= 0):
            continue
        projected = cam.project(line.endpoints)
        row["projected_world_endpoints_px"] = projected
        axis = row["axis"]
        if axis["axis_px"] is None:
            continue
        local = np.array(axis["axis_px"])
        mapping = endpoint_correspondence(local, projected)
        row["axis_to_world_endpoint"] = mapping
        if mapping is None:
            continue
        length = float(np.linalg.norm(local[1] - local[0]))
        colour = row["colour"]
        if colour and colour["tip_end"] is not None:
            colours.append((mapping[colour["tip_end"]], colour["confidence"], length))
        taper = taper_camera(
            end_widths=axis["end_widths_px"],
            tip_side=axis["tip_side"],
            wide_is_tip=False,
            axis_residual_px=axis["residual_px"],
            axis_to_track=mapping,
        )
        if taper:
            tapers.append((taper[0], taper[1], length))
    direction = conservative_direction([colour_cue(colours), taper_cue(tapers)])
    return json_value(
        {
            "views": rows,
            "line": None if line is None else asdict(line),
            "direction": direction,
            "loo": loo_residual(pairs),
        }
    )


def angle_degrees(a: np.ndarray, b: np.ndarray, *, unoriented: bool) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    cosine = float(a @ b) / float(np.linalg.norm(a) * np.linalg.norm(b))
    if unoriented:
        cosine = abs(cosine)
    return float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))


def motion(current: dict, previous: dict, elapsed: float) -> dict:
    if current["line"] is None or previous["line"] is None:
        return {"midpoint_cm_s": None, "axis_deg_s": None, "directed_deg_s": None}
    a, b = current["line"], previous["line"]
    result = {
        "midpoint_cm_s": float(np.linalg.norm(np.array(a["point"]) - b["point"]) / elapsed),
        "axis_deg_s": angle_degrees(a["direction"], b["direction"], unoriented=True) / elapsed,
        "directed_deg_s": None,
    }
    ia, ib = current["direction"]["tip_end"], previous["direction"]["tip_end"]
    if ia is not None and ib is not None and a["endpoints"] and b["endpoints"]:
        da = np.array(a["endpoints"])[ia] - np.array(a["endpoints"])[1 - ia]
        db = np.array(b["endpoints"])[ib] - np.array(b["endpoints"])[1 - ib]
        result["directed_deg_s"] = angle_degrees(da, db, unoriented=False) / elapsed
    return result


def paired_residuals(baseline: dict, revised: dict) -> dict:
    def cells(data):
        return {
            row["view"]: row
            for row in data["loo"]
            if row["fitted"] and row["perpendicular_px"] is not None
        }

    old, new = cells(baseline), cells(revised)
    keys = sorted(old.keys() & new.keys())
    return {
        "views": keys,
        "baseline_px": [old[v]["perpendicular_px"] for v in keys],
        "revised_px": [new[v]["perpendicular_px"] for v in keys],
        "delta_px": [new[v]["perpendicular_px"] - old[v]["perpendicular_px"] for v in keys],
    }


def aggregate(root: Path) -> None:
    import csv

    config = json.loads((root / "config.json").read_text())
    n = min(view["frame_count"] for view in config["views"].values())
    fps = config["native_fps"]
    rig = load_rig(Path("configs/finebio/cameras/P03_03_01.json"))
    sizes = {v: x["size_wh"] for v, x in config["views"].items()}
    streams = {v: (root / "observations" / f"{v}.jsonl").open() for v in config["views"]}
    buckets = {}
    previous = None
    per_second_reference = {}
    with (root / "geometry.jsonl").open("w") as output:
        for frame in range(n):
            obs = {v: json.loads(stream.readline()) for v, stream in streams.items()}
            if any(x["raw_frame"] != frame for x in obs.values()):
                raise ValueError(f"Camera synchronization mismatch at {frame}")
            cams = cameras_at_raw_frame(rig, frame, sizes)
            variants = {}
            for variant in VARIANTS:
                rows = {
                    v: dict(x["groups"][config["views"][v]["aliases"][variant]])
                    for v, x in obs.items()
                }
                variants[variant] = frame_geometry(rows, cams)
            second = int(frame / fps)
            bucket = buckets.setdefault(
                second,
                {
                    v: {
                        "frames": 0,
                        "fits": 0,
                        "directed": 0,
                        "active_views": [],
                        "axis_views": [],
                        "loo": [],
                        "paired_old": [],
                        "paired_new": [],
                        "paired_delta": [],
                        "length": [],
                        "motion_1s": [],
                        "step_motion": [],
                        "view_active": {v: 0 for v in config["views"]},
                        "view_axis": {v: 0 for v in config["views"]},
                    }
                    for v in VARIANTS
                },
            )
            for variant, data in variants.items():
                record = bucket[variant]
                record["frames"] += 1
                record["fits"] += int(data["line"] is not None)
                record["directed"] += int(data["direction"]["tip_end"] is not None)
                record["active_views"].append(sum(row["active"] for row in data["views"].values()))
                record["axis_views"].append(
                    sum(row["axis"]["axis_px"] is not None for row in data["views"].values())
                )
                for view, observation in data["views"].items():
                    record["view_active"][view] += int(observation["active"])
                    record["view_axis"][view] += int(observation["axis"]["axis_px"] is not None)
                record["loo"].extend(
                    row["perpendicular_px"]
                    for row in data["loo"]
                    if row["fitted"] and row["perpendicular_px"] is not None
                )
                if data["line"] and data["line"]["endpoints"]:
                    record["length"].append(
                        float(np.linalg.norm(np.diff(np.array(data["line"]["endpoints"]), axis=0)))
                    )
                paired = paired_residuals(variants[VARIANTS[0]], data)
                record["paired_old"].extend(paired["baseline_px"])
                record["paired_new"].extend(paired["revised_px"])
                record["paired_delta"].extend(paired["delta_px"])
                if previous:
                    record["step_motion"].append(motion(data, previous[variant], 1 / fps))
                ref = per_second_reference.get(frame - round(fps))
                if ref:
                    record["motion_1s"].append(motion(data, ref[variant], round(fps) / fps))
            per_second_reference[frame] = variants
            per_second_reference.pop(frame - round(fps) - 1, None)
            previous = variants
            output.write(
                json.dumps(
                    {"raw_frame": frame, "seconds": frame / fps, "variants": variants},
                    allow_nan=False,
                )
                + "\n"
            )
            if frame % 300 == 0:
                output.flush()
                print(f"Geometry {frame}/{n}", flush=True)
    for stream in streams.values():
        stream.close()
    rows = []

    def median(values):
        return float(np.median(values)) if values else None

    last_rows = {}
    for second, variants in buckets.items():
        for variant, data in variants.items():
            row = {
                "second": second,
                "variant": variant,
                "native_frames": data["frames"],
                "fit_frames": data["fits"],
                "direction_frames": data["directed"],
                "fit_fraction": data["fits"] / data["frames"],
                "direction_fraction": data["directed"] / data["frames"],
                "mean_active_views": float(np.mean(data["active_views"])),
                "mean_axis_views": float(np.mean(data["axis_views"])),
                "median_loo_px": median(data["loo"]),
                "loo_cells": len(data["loo"]),
                "paired_cells": len(data["paired_delta"]),
                "paired_baseline_median_px": median(data["paired_old"]),
                "paired_revised_median_px": median(data["paired_new"]),
                "median_paired_delta_px": median(data["paired_delta"]),
                "median_visible_length_cm": median(data["length"]),
            }
            for view in config["views"]:
                row[f"{view}_active_fraction"] = data["view_active"][view] / data["frames"]
                row[f"{view}_axis_fraction"] = data["view_axis"][view] / data["frames"]
            for interval, values in [("1s", data["motion_1s"]), ("step", data["step_motion"])]:
                for key in ["midpoint_cm_s", "axis_deg_s", "directed_deg_s"]:
                    valid = [x[key] for x in values if x[key] is not None]
                    row[f"{interval}_{key}_median"] = median(valid)
                    row[f"{interval}_{key}_samples"] = len(valid)
                    row[f"{interval}_{key}_p90"] = (
                        float(np.percentile(valid, 90)) if valid else None
                    )
            before = last_rows.get(variant)
            for key in ("median_loo_px", "fit_fraction", "direction_fraction"):
                row[f"{key}_change_from_previous_second"] = (
                    row[key] - before[key]
                    if before and row[key] is not None and before[key] is not None
                    else None
                )
            last_rows[variant] = row
            rows.append(row)
    (root / "per-second.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    with (root / "per-second.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("runs/finebio-pipette-improvement-20260930/full-video-line-comparison"),
    )
    parser.add_argument("--stage", choices=["observe", "aggregate"], required=True)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    if args.stage == "observe":
        config = json.loads((args.root / "config.json").read_text())
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(observe_view, args.root, view) for view in config["views"]]
            for future in futures:
                future.result()
    else:
        aggregate(args.root)


if __name__ == "__main__":
    main()
