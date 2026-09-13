"""Summarize completed, fixed-budget SAM3 ego-viewpoint screen runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from battle.schemas import RunManifest

REQUIRED_VIEWS = (
    "ego-hmc21176875",
    "ego-hmc21176623",
    "ego-hmc21110305",
    "ego-hmc21179183",
)


def _missing_ranges(present: set[int], frame_count: int) -> list[dict[str, int]]:
    ranges: list[dict[str, int]] = []
    start: int | None = None
    for frame_index in range(frame_count):
        if frame_index not in present and start is None:
            start = frame_index
        elif frame_index in present and start is not None:
            ranges.append({"start_frame": start, "end_frame_exclusive": frame_index})
            start = None
    if start is not None:
        ranges.append({"start_frame": start, "end_frame_exclusive": frame_count})
    return ranges


def _view_report(run_directory: Path, expected_view: str) -> dict[str, Any]:
    manifest = RunManifest.model_validate_json((run_directory / "manifest.json").read_text())
    if manifest.run_id != run_directory.name:
        raise ValueError(f"manifest ID does not match run directory: {run_directory}")
    if manifest.smoke is None:
        raise ValueError(f"screen input is not a smoke run: {run_directory}")
    if manifest.smoke.requested_analysis_frame_range.frame_count != 300:
        raise ValueError(f"screen input is not 300 frames: {run_directory}")
    if not manifest.observations or manifest.observations[0].view_id != expected_view:
        raise ValueError(f"screen run does not identify expected view {expected_view}")

    frame_count = len(manifest.observations)
    labels = list(manifest.smoke.concepts)
    initial = {item.label: item.object_id for item in manifest.observations[0].objects}
    label_frames = {label: set() for label in labels}
    label_ids = {label: set() for label in labels}
    for observation in manifest.observations:
        for item in observation.objects:
            if item.label in label_frames:
                label_frames[item.label].add(observation.analysis_frame_index)
                label_ids[item.label].add(item.object_id)

    worker = json.loads((run_directory / "worker_result.json").read_text())
    return {
        "run_directory": run_directory.as_posix(),
        "run_id": manifest.run_id,
        "state": worker["state"],
        "frames_processed": worker["frames_processed"],
        "requested_frames": 300,
        "coverage_seconds": frame_count / 30.0,
        "requested_seconds": 10.0,
        "elapsed_seconds": worker["elapsed_seconds"],
        "time_to_first_usable_output_seconds": worker["time_to_first_usable_output_seconds"],
        "gpu_peak_vram_bytes": worker["gpu_peak_vram_bytes"],
        "mask_artifact_count": worker["masks_written"],
        "initial_concept_results": {
            label: {"object_id": initial.get(label), "present": label in initial}
            for label in labels
        },
        "per_concept_continuity": {
            label: {
                "frames_emitted": len(label_frames[label]),
                "coverage_ratio": len(label_frames[label]) / frame_count,
                "emission_gaps": _missing_ranges(label_frames[label], frame_count),
                "ids_introduced_after_initialization": sorted(label_ids[label] - {initial[label]})
                if label in initial
                else sorted(label_ids[label]),
            }
            for label in labels
        },
        "intentional_id_resets": manifest.smoke.continuity.intentional_id_resets,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="append", required=True, metavar="VIEW=RUN_DIRECTORY", help="One per view."
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        run_directories = {
            entry.split("=", maxsplit=1)[0]: Path(entry.split("=", maxsplit=1)[1]).resolve()
            for entry in args.run
        }
    except IndexError:
        parser.error("--run must use VIEW=RUN_DIRECTORY")
    if tuple(run_directories) != REQUIRED_VIEWS:
        parser.error(f"runs must be supplied once in this order: {REQUIRED_VIEWS}")

    report = {
        "report_kind": "assembly101_ego_viewpoint_screen",
        "scope": (
            "fixed original-decoded-image, three-text-concept, continuous-memory 300-frame "
            "screen; the e3 entry reuses its preserved original zero-shot baseline"
        ),
        "review_timestamps_seconds": [0.0, 5.0, 299 / 30],
        "ground_truth_available": False,
        "views": {
            view_id: _view_report(run_directories[view_id], view_id) for view_id in REQUIRED_VIEWS
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Wrote ego viewpoint screen report: {args.output}")


if __name__ == "__main__":
    main()
