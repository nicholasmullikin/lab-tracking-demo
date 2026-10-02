"""Draft-mask agreement must exclude prompt frames and unreviewed references."""

import json

import cv2
import numpy as np

from battle.pipette_line_review import VARIANTS
from battle.pipette_video_agreement import evaluate


def test_seed_and_needs_review_drafts_cannot_inflate_reference_agreement(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    refs = tmp_path / "references"
    refs.mkdir()
    target = np.zeros((4, 4), np.uint8)
    target[:2, :2] = 255
    wrong = np.zeros_like(target)
    wrong[2:, 2:] = 255
    frames = {}
    for frame in (0, 100, 200):
        path = refs / f"{frame}.png"
        assert cv2.imwrite(str(path), target)
        frames[str(frame)] = {
            "views": {
                "T1": {
                    "mask_uri": path.name,
                    "quality": "needs-review" if frame == 200 else "usable-approximate-draft",
                }
            }
        }
        for index in range(3):
            folder = root / "native/T1/masks" / f"g{index}"
            folder.mkdir(parents=True, exist_ok=True)
            mask = (
                target
                if frame != 100 or index == 1
                else (wrong if index == 0 else np.zeros_like(target))
            )
            assert cv2.imwrite(str(folder / f"{frame:06d}.png"), mask)
    config = {
        "native_fps": 30000 / 1001,
        "views": {
            "T1": {
                "frame_count": 301,
                "size_wh": [4, 4],
                "aliases": dict(zip(VARIANTS, ("g0", "g1", "g2"), strict=True)),
                "groups": {f"g{i}": {"corrections": {"0": {}}} for i in range(3)},
            }
        },
    }
    (root / "config.json").write_text(json.dumps(config))
    labels = refs / "labels.json"
    labels.write_text(json.dumps({"frames": frames}))
    report = evaluate(root, labels)
    baseline, top, best = report["summary"]
    assert report["ground_truth"] is False
    assert all(x["matched_draft_cells"] == 1 for x in report["summary"])
    assert baseline["mean_draft_reference_iou"] == 0
    assert top["mean_draft_reference_iou"] == 1
    assert top["mean_paired_iou_change"] == 1
    assert best["empty_prediction_cells"] == 1
    assert all(x["draft_reference_iou"] is None for x in report["cells"] if x["raw_frame"] != 100)
