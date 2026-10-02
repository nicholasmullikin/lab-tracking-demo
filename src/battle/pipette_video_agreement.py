"""Agreement with approximate reviewed SAM drafts that were not used as video prompts.

These sparse first-20-second references are not independent ground-truth masks.
All seeded/corrected camera frames and needs-review drafts are excluded from scores.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .pipette_line_review import VARIANTS


def evaluate(root: Path, labels_path: Path) -> dict:
    config = json.loads((root / "config.json").read_text())
    labels = json.loads(labels_path.read_text())
    cells = []
    scores = {variant: {} for variant in VARIANTS}
    for frame_string, frame_data in labels["frames"].items():
        frame = int(frame_string)
        for view, reference in frame_data["views"].items():
            camera = config["views"][view]
            if frame >= camera["frame_count"]:
                continue
            prompted = any(frame_string in g["corrections"] for g in camera["groups"].values())
            exclusion = (
                "video_prompt_frame"
                if prompted
                else (
                    None
                    if reference["quality"] == "usable-approximate-draft"
                    else "needs_review_draft"
                )
            )
            path = labels_path.parent / reference["mask_uri"]
            pixels = cv2.imread(str(path), 0)
            if pixels is None or pixels.shape != tuple(reversed(camera["size_wh"])):
                raise ValueError(f"Bad draft reference: {path}")
            reference_mask = pixels > 0
            if not reference_mask.any():
                exclusion = exclusion or "empty_draft_reference"
            for variant in VARIANTS:
                prediction_path = (
                    root
                    / "native"
                    / view
                    / "masks"
                    / camera["aliases"][variant]
                    / f"{frame:06d}.png"
                )
                predicted = cv2.imread(str(prediction_path), 0)
                if predicted is None or predicted.shape != pixels.shape:
                    raise ValueError(f"Missing or incorrectly sized prediction: {prediction_path}")
                mask = predicted > 0
                union = int(np.logical_or(reference_mask, mask).sum())
                intersection = int(np.logical_and(reference_mask, mask).sum())
                iou = None if exclusion else intersection / union
                cell = {
                    "raw_frame": frame,
                    "seconds": frame / config["native_fps"],
                    "view": view,
                    "variant": variant,
                    "reference_quality": reference["quality"],
                    "excluded_reason": exclusion,
                    "draft_reference_iou": iou,
                    "prediction_empty": not bool(mask.any()),
                    "reference_mask": str(path.resolve()),
                    "reference_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "prediction_mask": str(prediction_path.resolve()),
                }
                cells.append(cell)
                if iou is not None:
                    scores[variant][(frame, view)] = cell
    summaries = []
    baseline = scores[VARIANTS[0]]
    for variant, measurements in scores.items():
        matched = sorted(baseline.keys() & measurements.keys())
        values = [measurements[k]["draft_reference_iou"] for k in matched]
        changes = [
            measurements[k]["draft_reference_iou"] - baseline[k]["draft_reference_iou"]
            for k in matched
        ]
        summaries.append(
            {
                "variant": variant,
                "matched_draft_cells": len(matched),
                "mean_draft_reference_iou": float(np.mean(values)) if values else None,
                "median_draft_reference_iou": float(np.median(values)) if values else None,
                "mean_paired_iou_change": float(np.mean(changes)) if changes else None,
                "median_paired_iou_change": float(np.median(changes)) if changes else None,
                "zero_overlap_cells": sum(x == 0 for x in values),
                "empty_prediction_cells": sum(measurements[k]["prediction_empty"] for k in matched),
            }
        )
    report = {
        "ground_truth": False,
        "reference_type": "reviewed approximate SAM drafts",
        "scope": (
            "Reference camera frames not used as tracker seeds/corrections; usable drafts only. "
            "Sparse raw 0-600 references do not assess the rest of the video."
        ),
        "labels_path": str(labels_path.resolve()),
        "summary": summaries,
        "cells": cells,
    }
    (root / "draft-agreement.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (root / "draft-agreement.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(cells[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(cells)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("runs/finebio-pipette-improvement-20260930/full-video-line-comparison"),
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("runs/finebio-pipette-improvement-20260930/every100-review/labels.json"),
    )
    args = parser.parse_args()
    report = evaluate(args.root, args.labels)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
