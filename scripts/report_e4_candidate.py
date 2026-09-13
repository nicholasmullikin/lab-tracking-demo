"""Write factual continuity measures for the approved e4-only 60-second candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from battle.schemas import RunManifest


def missing_ranges(present: set[int], frame_count: int) -> list[dict[str, int]]:
    """Return half-open missing observation ranges."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    run_directory = args.run_directory.resolve()
    manifest = RunManifest.model_validate_json((run_directory / "manifest.json").read_text())
    candidate = manifest.e4_candidate
    if candidate is None or candidate.view_id != "ego-hmc21179183":
        parser.error("run manifest is not the approved e4-only candidate")
    frame_count = len(manifest.observations)
    if frame_count != 1800:
        parser.error(f"candidate observations must contain 1,800 frames; found {frame_count}")

    initial = {item.label: item.object_id for item in manifest.observations[0].objects}
    emitted = {concept: set() for concept in candidate.concepts}
    observed_ids = {concept: set() for concept in candidate.concepts}
    for observation in manifest.observations:
        for item in observation.objects:
            if item.label in emitted:
                emitted[item.label].add(observation.analysis_frame_index)
                observed_ids[item.label].add(item.object_id)
    report = {
        "report_kind": "assembly101_e4_60_second_candidate",
        "run_id": manifest.run_id,
        "scope": (
            "e4-only monochrome candidate; original decoded input, fixed three text concepts, "
            "and bounded continuous memory; no accuracy or general ego claim"
        ),
        "candidate_metadata": candidate.model_dump(mode="json"),
        "coverage_seconds": manifest.coverage.covered_seconds,
        "coverage_ratio_of_proxy": manifest.coverage.ratio,
        "initial_concept_results": {
            concept: {"object_id": initial.get(concept), "present": concept in initial}
            for concept in candidate.concepts
        },
        "per_track_continuity": {
            concept: {
                "frames_emitted": len(emitted[concept]),
                "coverage_ratio": len(emitted[concept]) / frame_count,
                "emission_gaps": missing_ranges(emitted[concept], frame_count),
                "ids_introduced_after_initialization": sorted(
                    observed_ids[concept] - {initial[concept]}
                )
                if concept in initial
                else sorted(observed_ids[concept]),
            }
            for concept in candidate.concepts
        },
        "intentional_id_resets": candidate.continuity.intentional_id_resets,
        "method_statuses": [status.model_dump(mode="json") for status in manifest.method_statuses],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Wrote e4 candidate report: {args.output}")


if __name__ == "__main__":
    main()
