from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from battle.segmentation_disagreement import (
    compare_runs,
    disagreement_episodes,
    load_mask_index,
    mask_iou,
    per_frame_target_iou,
)

SIZE = (24, 32)


def _write_run(root: Path, masks: dict[int, dict[str, tuple[int, int, int, int] | None]]) -> Path:
    """Write observations.jsonl plus PNG masks; ``None`` records the label without a mask."""
    root.mkdir(parents=True)
    lines = []
    for frame in sorted(masks):
        objects = []
        for label, box in masks[frame].items():
            if box is None:
                continue
            canvas = np.zeros(SIZE, dtype=np.uint8)
            x0, y0, x1, y1 = box
            canvas[y0:y1, x0:x1] = 255
            uri = f"masks/{label}/{frame:05d}.png"
            (root / uri).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(canvas).save(root / uri)
            objects.append(
                {
                    "object_id": f"fixture-{label}",
                    "label": label,
                    "confidence": 1.0,
                    "box": {"x": 0.0, "y": 0.0, "width": 0.1, "height": 0.1},
                    "mask": {"uri": uri, "storage": "native_artifact", "format": "png"},
                }
            )
        lines.append(
            json.dumps(
                {
                    "view_id": "static-c10379",
                    "analysis_frame_index": frame,
                    "source_seconds": 294.0 + frame / 30,
                    "objects": objects,
                }
            )
        )
    (root / "observations.jsonl").write_text("\n".join(lines) + "\n")
    return root


def test_mask_iou_treats_one_sided_absence_as_full_disagreement() -> None:
    a = np.zeros(SIZE, bool)
    a[0:4, 0:4] = True
    b = np.zeros(SIZE, bool)
    b[0:4, 0:2] = True
    assert mask_iou(a, b) == (0.5, "both_present")
    assert mask_iou(a, None) == (0.0, "missing_in_b")
    assert mask_iou(None, b) == (0.0, "missing_in_a")
    assert mask_iou(None, None) == (None, "both_missing")


def test_episodes_need_five_consecutive_low_frames_and_skip_both_missing(tmp_path: Path) -> None:
    same = (0, 0, 8, 8)
    shifted = (6, 0, 14, 8)  # IoU 2/14 with `same`
    frames = range(12)
    run_a = _write_run(
        tmp_path / "a",
        {f: {"chassis": same, "interior": same if f != 9 else None} for f in frames},
    )
    run_b = _write_run(
        tmp_path / "b",
        {
            f: {
                # frames 2..7 disagree (6 frames): one episode.
                "chassis": shifted if 2 <= f <= 7 else same,
                # frames 3..6 disagree (4 frames) then frame 9 is missing on both sides:
                # 4 < 5 so no episode; frame 9 must neither break nor extend anything.
                "interior": shifted if 3 <= f <= 6 else (None if f == 9 else same),
            }
            for f in frames
        },
    )
    rows = per_frame_target_iou(
        load_mask_index(run_a),
        load_mask_index(run_b),
        frames=frames,
        targets=("chassis", "interior"),
    )
    by_key = {(row.frame, row.target): row for row in rows}
    assert by_key[(0, "chassis")].iou == 1.0
    assert abs(by_key[(3, "chassis")].iou - 2 / 14) < 1e-9
    assert by_key[(9, "interior")].status == "both_missing"
    episodes = disagreement_episodes(rows, threshold=0.5, minimum_frames=5)
    assert [(e.target, e.start_frame, e.end_frame_exclusive, e.frame_count) for e in episodes] == [
        ("chassis", 2, 8, 6)
    ]
    assert episodes[0].missing_frames_a == episodes[0].missing_frames_b == 0

    report = compare_runs(
        run_a=run_a,
        run_b=run_b,
        start_frame=0,
        end_frame_exclusive=12,
        targets=("chassis", "interior"),
    )
    assert report["ground_truth_accuracy_claim"] is False
    assert "not ground-truth accuracy" in str(report["claim"])
    assert report["summary"]["chassis"]["frames_below_0_5"] == 6
    assert report["summary"]["interior"]["both_missing"] == 1
    assert len(report["rows"]) == 24


def test_missing_on_one_side_extends_an_episode(tmp_path: Path) -> None:
    same = (0, 0, 8, 8)
    run_a = _write_run(tmp_path / "a", {f: {"cabin": same} for f in range(8)})
    run_b = _write_run(tmp_path / "b", {f: {"cabin": same if f < 3 else None} for f in range(8)})
    rows = per_frame_target_iou(
        load_mask_index(run_a), load_mask_index(run_b), frames=range(8), targets=("cabin",)
    )
    episodes = disagreement_episodes(rows)
    assert len(episodes) == 1
    assert (episodes[0].start_frame, episodes[0].end_frame_exclusive) == (3, 8)
    assert episodes[0].missing_frames_b == 5
    assert episodes[0].mean_iou == 0.0
