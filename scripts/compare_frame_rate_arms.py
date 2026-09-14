#!/usr/bin/env python
"""Compare frame-rate arms on the shared wall-clock timeline rather than frame index.

Arms sample at different rates, so frame indices are not comparable and raw frame counts
flatter whichever arm has more of them. Everything here is keyed on source seconds, and
coverage is reported as a fraction of each arm's own frames.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

TARGETS = ("left_hand", "right_hand", "yellow_toy_top", "black_toy_top_base")
# A box this large is the stray-pixel inflation defect rather than a plausible target.
INFLATED_NORMALIZED_AREA = 0.20


def _load_observations(run_directory: Path) -> list[dict[str, Any]]:
    path = run_directory / "observations.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _target_summary(observations: list[dict[str, Any]], target: str) -> dict[str, Any]:
    seconds_present: list[float] = []
    seconds_absent: list[float] = []
    areas: list[float] = []
    scores: list[float] = []
    ious: list[float] = []

    for observation in observations:
        seconds = observation["source_seconds"]
        match = next((o for o in observation["objects"] if o.get("label") == target), None)
        if match is None:
            seconds_absent.append(seconds)
            continue
        seconds_present.append(seconds)
        areas.append(match["box"]["width"] * match["box"]["height"])
        if match.get("object_score") is not None:
            scores.append(match["object_score"])
        if match.get("iou_prediction") is not None:
            ious.append(match["iou_prediction"])

    first_drop = None
    if seconds_present and seconds_absent:
        after_first_emission = [s for s in seconds_absent if s > seconds_present[0]]
        first_drop = min(after_first_emission) if after_first_emission else None

    inflated = [
        seconds
        for seconds, area in zip(seconds_present, areas, strict=True)
        if area >= INFLATED_NORMALIZED_AREA
    ]
    return {
        "emitted_frames": len(seconds_present),
        "emitted_fraction": round(len(seconds_present) / len(observations), 4),
        "first_dropout_source_seconds": first_drop,
        "inflated_box_frames": len(inflated),
        "first_inflated_source_seconds": min(inflated) if inflated else None,
        "median_normalized_box_area": round(statistics.median(areas), 5) if areas else None,
        "max_normalized_box_area": round(max(areas), 5) if areas else None,
        "median_iou_prediction": round(statistics.median(ious), 4) if ious else None,
        "min_iou_prediction": round(min(ious), 4) if ious else None,
        "min_object_score": round(min(scores), 3) if scores else None,
    }


def summarize_arm(run_directory: Path) -> dict[str, Any]:
    manifest = json.loads((run_directory / "manifest.json").read_text())
    smoke = manifest["smoke"]
    settings = smoke["runtime_settings"]
    observations = _load_observations(run_directory)
    measurements = smoke["measurements"]
    return {
        "run_id": run_directory.name,
        "analysis_fps": settings.get("analysis_fps"),
        "max_side_length": settings.get("max_side_length"),
        "max_frame_memory": settings.get("max_frame_memory"),
        "frame_memory_span_seconds": settings.get("frame_memory_span_seconds"),
        "frames_processed": len(observations),
        "elapsed_seconds": measurements.get("elapsed_seconds"),
        "time_to_first_usable_output_seconds": measurements.get(
            "time_to_first_usable_output_seconds"
        ),
        "compute_seconds_per_second_of_video": (
            round(measurements["elapsed_seconds"] / 10.0, 3)
            if measurements.get("elapsed_seconds")
            else None
        ),
        "gpu_peak_vram_gib": (
            round(measurements["gpu_peak_vram_bytes"] / 2**30, 3)
            if measurements.get("gpu_peak_vram_bytes")
            else None
        ),
        "per_target": {
            target: _target_summary(observations, target) for target in TARGETS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="Run directories to compare.")
    parser.add_argument("--output", type=Path, help="Write the JSON report here as well.")
    args = parser.parse_args()

    report = {
        "comparison_scope": (
            "output/continuity measures over the same ten seconds of source video at "
            "different analysis rates; not ground-truth accuracy"
        ),
        "arms": [summarize_arm(run) for run in args.runs],
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
