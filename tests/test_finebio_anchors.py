"""`battle-finebio-anchors` (p6-anchors): frame selection, the candidate workspace without a
GPU, the scoreboard on empty / partial / synthetic records, identity F1, the export skeleton
and the committed trial-1 anchor config."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from battle import finebio_anchors as fa

COMMITTED_CONFIG = Path("configs/qa/finebio_P03_03_01_review_anchors.json")
WIDTH, HEIGHT = 64, 48


# ------------------------------------------------------------------------------ selection


def _sam3_row(view, frame, slot, iou, *, pose_valid=True, cls=None, area=100):
    return {
        "view": view,
        "frame_index": frame,
        "slot": slot,
        "object_class": cls or slot.rsplit("#", 1)[0],
        "source": "sam3_decode",
        "pose_valid": pose_valid,
        "mask_area_px": area,
        "box_xyxy_px": [10.0, 10.0, 30.0, 30.0],
        "mask_bbox_px": [10.0, 10.0, 30.0, 30.0],
        "mask_centroid_px": [20.0, 20.0],
        "provenance": {"detector_box_iou": iou, "object_id": "sam3-00"},
    }


def test_disagreement_scores_pool_non_group_slots_of_both_views():
    rows = [
        _sam3_row("fpv", 700, "plate#0", 0.9),
        _sam3_row("T4", 700, "plate#0", 0.4),
        _sam3_row("T4", 700, "micro_tube_group#0", 0.1),  # group: outside the pool
        _sam3_row("fpv", 701, "plate#0", 0.2),  # T4 missing on 701: not eligible
    ]
    scores = fa.disagreement_scores(rows, ("fpv", "T4"))
    assert set(scores) == {700}
    assert scores[700] == {"min_iou": 0.4, "slot": "T4/plate#0", "rows": 2}


def test_select_frames_spacing_cycles_and_seeded_randoms():
    window = (600, 4200)
    scores = {f: {"min_iou": 0.9, "slot": "fpv/plate#0", "rows": 3} for f in range(600, 4200)}
    # Lowest disagreement clustered around 2000 (only one may be taken under the spacing) and
    # two frames inside the cycle neighbourhood.
    for f in (2000, 2010, 2020, 2050):
        scores[f]["min_iou"] = 0.05
    scores[1200]["min_iou"] = 0.3
    scores[1300]["min_iou"] = 0.35
    params = fa.SelectionParams(
        spacing_frames=90,
        disagreement_frames=4,
        random_frames=2,
        min_in_cycles=2,
        cycle_pad_frames=90,
        seed=7,
    )
    chosen = fa.select_frames(
        scores,
        window=window,
        anchor_frames=[916],
        cycles=[[1176, 1228]],
        params=params,
        eligible=set(scores),
    )
    frames = [c["raw_frame"] for c in chosen]
    assert frames == sorted(frames) and len(set(frames)) == len(frames)
    for a in frames:
        for b in frames:
            assert a == b or abs(a - b) >= 90
    origins = [c["origin"] for c in chosen]
    assert origins.count("six_view_annotated") == 1 and 916 in frames
    assert origins.count("disagreement") == 4 and origins.count("random") == 2
    disagreement = [c for c in chosen if c["origin"] == "disagreement"]
    assert sum(c["in_cycle_neighbourhood"] for c in disagreement) >= 2
    assert {1200, 1300} <= {c["raw_frame"] for c in disagreement}
    assert sum(1 for c in disagreement if 2000 <= c["raw_frame"] <= 2050) == 1
    again = fa.select_frames(
        scores,
        window=window,
        anchor_frames=[916],
        cycles=[[1176, 1228]],
        params=params,
        eligible=set(scores),
    )
    assert [c["raw_frame"] for c in again] == frames
    other = fa.select_frames(
        scores,
        window=window,
        anchor_frames=[916],
        cycles=[[1176, 1228]],
        params=fa.SelectionParams(**{**params.__dict__, "seed": 8}),
        eligible=set(scores),
    )
    assert [c["raw_frame"] for c in other if c["origin"] == "random"] != [
        c["raw_frame"] for c in chosen if c["origin"] == "random"
    ]


def test_fixed_view_choice_prefers_the_most_tracked_slots_with_the_plate_visible():
    slots = {
        "T4": [
            {"slot": 0, "label": "cell_culture_plate#0", "class": "cell_culture_plate"},
            {"slot": 1, "label": "blue_pipette#0", "class": "blue_pipette"},
            {"slot": 2, "label": "50ml_tube#0", "class": "50ml_tube"},
        ],
        "T5": [
            {"slot": 0, "label": "cell_culture_plate#0", "class": "cell_culture_plate"},
            {"slot": 1, "label": "blue_pipette#0", "class": "blue_pipette"},
            {"slot": 2, "label": "micro_tube_group#0", "class": "micro_tube_group"},
        ],
        "T1": [
            {"slot": 0, "label": "cell_culture_plate#0", "class": "cell_culture_plate"},
            {"slot": 1, "label": "a#0", "class": "a"},
            {"slot": 2, "label": "b#0", "class": "b"},
            {"slot": 3, "label": "c#0", "class": "c"},
        ],
    }
    rows = []
    for f in range(600, 700):
        rows.append(_sam3_row("T4", f, "cell_culture_plate#0", 0.9))
        rows.append(_sam3_row("T5", f, "cell_culture_plate#0", 0.9))
        if f < 620:  # T1's plate has a mask on a fifth of the window only
            rows.append(_sam3_row("T1", f, "cell_culture_plate#0", 0.9))
    table = fa.fixed_view_table(rows, slots, ("T1", "T4", "T5"), (600, 700))
    assert table["T4"]["non_group_slots"] == 3 and table["T5"]["non_group_slots"] == 2
    assert table["T1"]["plate_visible_fraction"] == 0.2
    view, reason = fa.choose_fixed_view(table, 0.9)
    assert view == "T4" and "most individually tracked slots" in reason


def test_committed_trial1_config_follows_the_plan():
    config = fa.load_config(COMMITTED_CONFIG)
    assert config["trial"] == "P03_03_01" and config["fixed_view"] == "T4"
    assert config["views"]["pair"] == ["fpv", "T4"]
    frames = config["frames"]
    assert config["counts"]["by_origin"] == {
        "six_view_annotated": 1,
        "disagreement": 12,
        "random": 5,
    }
    six = [f for f in frames if f["origin"] == "six_view_annotated"]
    assert six[0]["raw_frame"] == 916 and six[0]["views"] == ["T1", "T2", "T3", "T4", "T5", "fpv"]
    raws = [f["raw_frame"] for f in frames]
    for a in raws:
        for b in raws:
            assert a == b or abs(a - b) >= config["selection"]["disagreement"]["spacing_frames"]
    assert all(f["proxy_frame"] == f["raw_frame"] - 600 for f in frames)
    assert all(600 <= f["raw_frame"] < 4200 for f in frames)
    hoods = config["selection"]["disagreement"]["cycle_neighbourhoods"]
    assert hoods == [[1086, 1318], [3134, 3401]]
    in_cycles = [f for f in frames if f["origin"] == "disagreement" and f["in_cycle_neighbourhood"]]
    assert len(in_cycles) >= 3
    for f in frames:
        if f["origin"] != "six_view_annotated":
            assert f["views"] == ["fpv", "T4"]
        if f["origin"] == "disagreement":
            assert f["arm_b_min_detector_box_iou"]["min_iou"] < 0.5
    assert all(len(config["slots"][v]) == 11 for v in config["views"]["six_view"])
    assert config["selection"]["random"]["seed"] == 20260925
    assert "not ground truth" in config["claim_boundary"]


# ------------------------------------------------------------------------------ workspace


def _png(path: Path, mask: np.ndarray) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask.astype(np.uint8) * 255)


def _rect(x0, y0, x1, y1) -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _write_video(path: Path, frames: int) -> None:
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (WIDTH, HEIGHT))
    for i in range(frames):
        image = np.full((HEIGHT, WIDTH, 3), 40 + (i % 50), dtype=np.uint8)
        writer.write(image)
    writer.release()


OFFSET = 900
FRAMES = (916, 1000)  # proxy 16 and 100
SLOTS = {
    "fpv": [
        {"slot": 0, "label": "centrifuge#0", "class": "centrifuge", "role": "container"},
        {"slot": 1, "label": "50ml_tube#0", "class": "50ml_tube", "role": "object"},
    ],
    "T4": [
        {"slot": 0, "label": "centrifuge#0", "class": "centrifuge", "role": "container"},
        {"slot": 1, "label": "50ml_tube#0", "class": "50ml_tube", "role": "object"},
    ],
}
BOXES = {"centrifuge#0": (4.0, 4.0, 24.0, 28.0), "50ml_tube#0": (36.0, 10.0, 52.0, 40.0)}
MASKS = {"centrifuge#0": _rect(5, 5, 23, 27), "50ml_tube#0": _rect(37, 11, 51, 39)}


def _write_arm(
    root: Path,
    *,
    sam3: bool,
    masks: dict[str, np.ndarray] | None,
    tracks: dict[tuple[int, str, str], str],
    skip: set[tuple[int, str, str]] = frozenset(),
) -> Path:
    """A synthetic arm directory: observations.jsonl (detector rows, plus SAM3 rows with masks
    under a worker run when `sam3`), measures.json naming the worker runs, tracks.jsonl with
    support_slots giving each (frame, view, slot) its track id."""
    root.mkdir(parents=True)
    rows = []
    worker_runs = {}
    for view in SLOTS:
        if sam3:
            run = root / "worker" / view
            (run / "masks").mkdir(parents=True)
            worker_runs[view] = str(run)
        for frame in FRAMES:
            for spec in SLOTS[view]:
                label = spec["label"]
                box = BOXES[label]
                rows.append(
                    {
                        "view": view,
                        "frame_index": frame,
                        "slot": f"{spec['class']}#0",
                        "object_class": spec["class"],
                        "detector_score": 0.8,
                        "box_xyxy_px": list(box),
                        "pose_valid": True,
                        "source": "detector",
                    }
                )
                if not sam3 or (frame, view, label) in skip:
                    continue
                mask = (masks or MASKS)[label]
                index = spec["slot"]
                _png(run / "masks" / f"{frame - OFFSET:06d}_{index:02d}.png", mask)
                ys, xs = np.nonzero(mask)
                rows.append(
                    {
                        "view": view,
                        "frame_index": frame,
                        "slot": label,
                        "object_class": spec["class"],
                        "detector_score": 0.8,
                        "box_xyxy_px": list(box),
                        "mask_bbox_px": [
                            float(xs.min()),
                            float(ys.min()),
                            float(xs.max() + 1),
                            float(ys.max() + 1),
                        ],
                        "mask_centroid_px": [float(xs.mean()) + 0.5, float(ys.mean()) + 0.5],
                        "mask_area_px": int(mask.sum()),
                        "sam3_object_score": 0.9,
                        "pose_valid": True,
                        "source": "sam3_decode",
                        "provenance": {
                            "worker_label": label,
                            "object_id": f"sam3-{index:02d}",
                            "decoder_iou_pred": 0.9,
                            "detector_box_iou": 0.85,
                        },
                    }
                )
    with (root / "observations.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    (root / "measures.json").write_text(
        json.dumps({"observations": {"worker_runs": worker_runs}}), encoding="utf-8"
    )
    (root / "tracks").mkdir()
    by_frame_track: dict[tuple[int, str], dict[str, str]] = {}
    for (frame, view, slot), track in tracks.items():
        by_frame_track.setdefault((frame, track), {})[view] = slot
    with (root / "tracks" / "tracks.jsonl").open("w", encoding="utf-8") as handle:
        for (frame, track), support in by_frame_track.items():
            handle.write(
                json.dumps(
                    {"frame_index": frame, "track_id": track, "support_slots": support},
                    separators=(",", ":"),
                )
                + "\n"
            )
    return root


def _config(root: Path, clip_path: Path) -> dict:
    return {
        "schema": fa.SCHEMA,
        "config_kind": fa.CONFIG_KIND,
        "config_id": "synthetic-review-anchors",
        "anchor_kind": fa.ANCHOR_KIND,
        "clip_config": str(clip_path.relative_to(root)),
        "trial": "SYN",
        "window": {"start": OFFSET, "end": OFFSET + 200, "frame_index_offset": OFFSET},
        "views": {"six_view": ["fpv", "T4"], "pair": ["fpv", "T4"]},
        "fixed_view": "T4",
        "selection": {"six_view_annotated_frames": [916]},
        "slots": SLOTS,
        "frames": [
            {
                "raw_frame": 916,
                "proxy_frame": 16,
                "views": ["fpv", "T4"],
                "origin": "six_view_annotated",
                "reason": "test",
                "in_cycle_neighbourhood": False,
            },
            {
                "raw_frame": 1000,
                "proxy_frame": 100,
                "views": ["fpv", "T4"],
                "origin": "disagreement",
                "reason": "test",
                "in_cycle_neighbourhood": True,
            },
        ],
        "claim_boundary": fa.CLAIM_BOUNDARY,
        "licence": fa.LICENCE_NOTE,
    }


@pytest.fixture
def synthetic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A repository root with a clip config, two tiny proxies, arm (b) with masks and a
    workspace built through the CLI without a GPU (`--skip-decode`)."""
    root = tmp_path
    monkeypatch.chdir(root)
    proxies = {}
    for view in SLOTS:
        video = root / "data" / f"{view}.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        _write_video(video, 101)
        proxies[view] = str(video.relative_to(root))
    clip_path = root / "configs" / "clip.json"
    clip_path.parent.mkdir(parents=True)
    clip_path.write_text(
        json.dumps(
            {
                "config_kind": "finebio_clip_config",
                "trial": "SYN",
                "views": ["fpv", "T4"],
                "fixed_views": ["T4"],
                "fpv_view": "fpv",
                "view_sizes": {"fpv": [WIDTH, HEIGHT], "T4": [WIDTH, HEIGHT]},
                "window": {"start_frame": OFFSET, "end_frame_exclusive": OFFSET + 200},
                "frame_index_offset": OFFSET,
                "proxies": proxies,
            }
        ),
        encoding="utf-8",
    )
    config_path = root / "configs" / "anchors.json"
    config_path.write_text(json.dumps(_config(root, clip_path)), encoding="utf-8")
    # Arm (b): every cell has a mask, one track per object across both views.
    tracks_b = {}
    for frame in FRAMES:
        for view in SLOTS:
            tracks_b[(frame, view, "centrifuge#0")] = "centrifuge-001"
            tracks_b[(frame, view, "50ml_tube#0")] = "50ml_tube-001"
    arm_b = _write_arm(root / "arms" / "b", sam3=True, masks=None, tracks=tracks_b)
    # Arm (c): shifted masks (IoU < 1), no mask for the fpv tube on 1000, the tube split into
    # two track ids across the views.
    shifted = {"centrifuge#0": _rect(7, 5, 25, 27), "50ml_tube#0": _rect(37, 15, 51, 43)}
    tracks_c = dict(tracks_b)
    tracks_c[(916, "T4", "50ml_tube#0")] = "50ml_tube-002"
    arm_c = _write_arm(
        root / "arms" / "c",
        sam3=True,
        masks=shifted,
        tracks=tracks_c,
        skip={(1000, "fpv", "50ml_tube#0")},
    )
    # Arm (a): detector rows only; the tracker's slots are the detector's <class>#0.
    tracks_a = {
        (f, v, f"{s['class']}#0"): t
        for (f, v, label), t in tracks_b.items()
        for s in SLOTS[v]
        if s["label"] == label
    }
    arm_a = _write_arm(root / "arms" / "a", sam3=False, masks=None, tracks=tracks_a)
    workspace = root / "runs" / "anchors"
    code = fa.main(
        [
            "workspace",
            "--config",
            str(config_path),
            "--arm-b",
            str(arm_b),
            "--output",
            str(workspace),
            "--skip-decode",
        ]
    )
    assert code == 0
    return {
        "root": root,
        "config": config_path,
        "workspace": workspace,
        "arms": {"a": arm_a, "b": arm_b, "c": arm_c},
    }


def test_workspace_without_a_gpu_has_candidate_zero_sheets_and_the_template(synthetic):
    workspace = synthetic["workspace"]
    doc = fa.load_workspace(workspace)
    assert doc["counts"] == {
        "cells": 8,
        "cells_with_candidates": 8,
        "candidates": 8,
        "duplicates": 0,
        "cells_without_box": 0,
    }
    assert doc["decode"]["state"] == "skipped"
    cell = next(c for c in doc["cells"] if c["view"] == "T4" and c["label"] == "50ml_tube#0")
    assert cell["reference_box"] == [36.0, 10.0, 52.0, 40.0]
    assert cell["point_source"] == "arm_b_mask_centroid"
    (candidate,) = cell["candidates"]
    assert candidate["kind"] == "arm_b_tight" and candidate["duplicate_of"] is None
    assert candidate["mask_area_px"] == int(MASKS["50ml_tube#0"].sum())
    copied = fa.read_mask(workspace / candidate["mask_uri"])
    assert np.array_equal(copied, MASKS["50ml_tube#0"])
    for frame in FRAMES:
        for view in SLOTS:
            assert (workspace / "sheets" / f"f{frame:06d}_{view}.jpg").is_file()
            assert (workspace / "sheets" / f"f{frame:06d}_{view}_overview.jpg").is_file()
    template = json.loads((workspace / "decisions.template.json").read_text())
    assert template["schema"] == f"{fa.SCHEMA}/decisions"
    assert len(template["cells"]) == 8
    assert all(c["decision"] is None for c in template["cells"])
    identities = {c["label"]: c["instance_identity"] for c in template["cells"]}
    assert identities == {"centrifuge#0": "centrifuge", "50ml_tube#0": None}
    requests = json.loads((workspace / "requests.json").read_text())
    slot = requests["views"]["T4"]["frames"][0]["slots"][1]
    assert slot["margin_box"] == [33.6, 5.5, 54.4, 44.5]
    assert (workspace / "README.md").read_text().startswith("# Human review anchors")


def test_build_requests_falls_back_to_the_box_centre_and_skips_missing_boxes():
    config = {
        "window": {"frame_index_offset": 0},
        "frames": [{"raw_frame": 5, "views": ["T4"], "origin": "random"}],
        "slots": {
            "T4": [
                {"slot": 0, "label": "a#0", "class": "a"},
                {"slot": 1, "label": "b#0", "class": "b"},
            ]
        },
        "trial": "t",
    }
    cells = {
        (5, "T4", "a#0"): {"box_xyxy_px": [0, 0, 10, 10], "mask_centroid_px": [50, 50]},
    }
    requests = fa.build_requests(
        config, cells, proxies={"T4": Path("v.mp4")}, view_sizes={"T4": (100, 80)}
    )
    (frame,) = requests["views"]["T4"]["frames"]
    (slot,) = frame["slots"]
    assert slot["label"] == "a#0" and slot["point"] == [5.0, 5.0]
    assert slot["point_source"] == "box_centre"
    assert requests["views"]["T4"]["image_hw"] == [80, 100]


# ------------------------------------------------------------------------------ scoring


def _anchor(state, mask=None, bbox=None, cell=None, identity=None):
    cell = cell or {
        "raw_frame": 916,
        "view": "T4",
        "slot": 1,
        "label": "50ml_tube#0",
        "class": "50ml_tube",
        "origin": "six_view_annotated",
        "in_cycle_neighbourhood": False,
        "reference_box": [36.0, 10.0, 52.0, 40.0],
        "candidates": [],
    }
    return fa.Anchor(cell=cell, state=state, mask=mask, bbox=bbox, identity=identity)


def test_score_cell_covers_every_branch():
    human = _rect(37, 11, 51, 39)
    sam3_row = {"source": "sam3_decode", "slot": "50ml_tube#0", "mask_bbox_px": [37, 11, 51, 39]}
    det_row = {"source": "detector", "slot": "50ml_tube#0", "box_xyxy_px": [36.0, 10.0, 52.0, 40.0]}
    perfect = fa.score_cell(_anchor("mask", human, fa.mask_bbox(human)), sam3_row, human)
    assert (
        perfect["outcome"] == "scored" and perfect["mask_iou"] == 1.0 and perfect["box_iou"] == 1.0
    )
    half = fa.score_cell(
        _anchor("mask", human, fa.mask_bbox(human)), sam3_row, _rect(37, 25, 51, 39)
    )
    assert (
        half["mask_iou"] == 0.5 and half["arm_area"] == 14 * 14 and half["anchor_area"] == 14 * 28
    )
    boxes_only = fa.score_cell(_anchor("mask", human, fa.mask_bbox(human)), det_row, None)
    assert boxes_only["outcome"] == "scored" and boxes_only["mask_iou"] is None
    assert 0.6 < boxes_only["box_iou"] < 1.0
    missing = fa.score_cell(_anchor("mask", human, fa.mask_bbox(human)), None, None)
    assert missing["outcome"] == "run_mask_missing" and missing["mask_iou"] == 0.0
    box_level = fa.score_cell(_anchor("box", bbox=(36.0, 10.0, 52.0, 40.0)), det_row, None)
    assert box_level["outcome"] == "scored" and box_level["box_iou"] == 1.0
    assert box_level["mask_iou"] is None
    assert fa.score_cell(_anchor("hidden"), None, None)["outcome"] == "hidden_correct"
    assert fa.score_cell(_anchor("hidden"), det_row, None)["outcome"] == "hidden_correct"
    fp = fa.score_cell(_anchor("hidden"), sam3_row, human)
    assert fp["outcome"] == "hidden_false_positive" and fp["arm_area"] == int(human.sum())
    assert fa.score_cell(_anchor("unlabeled"), sam3_row, human)["outcome"] == "unlabeled_skipped"
    assert fa.score_cell(_anchor("none_fits"), sam3_row, human)["outcome"] == "none_fits_skipped"


def test_match_arm_row_by_label_then_by_class_and_overlap():
    anchor = _anchor("box", bbox=(36.0, 10.0, 52.0, 40.0))
    rows = [
        {
            "source": "detector",
            "slot": "50ml_tube#1",
            "object_class": "50ml_tube",
            "box_xyxy_px": [0, 0, 5, 5],
        },
        {
            "source": "detector",
            "slot": "50ml_tube#0",
            "object_class": "50ml_tube",
            "box_xyxy_px": [35, 10, 52, 40],
        },
        {
            "source": "detector",
            "slot": "centrifuge#0",
            "object_class": "centrifuge",
            "box_xyxy_px": [36, 10, 52, 40],
        },
    ]
    assert fa.match_arm_row(rows, anchor)["slot"] == "50ml_tube#0"
    sam3 = {"source": "sam3_video", "slot": "x", "provenance": {"worker_label": "50ml_tube#0"}}
    assert fa.match_arm_row([*rows, sam3], anchor) is sam3
    assert fa.match_arm_row(rows[:1], anchor) is None


def test_idf1_perfect_split_merge_and_missing():
    perfect = fa.idf1([("tube_A", "t1"), ("tube_A", "t1"), ("tube_B", "t2")])
    assert perfect["idf1"] == 1.0 and perfect["idtp"] == 3
    split = fa.idf1([("tube_A", "t1"), ("tube_A", "t2"), ("tube_B", "t3")])
    assert split["idtp"] == 2 and split["idf1"] == pytest.approx(4 / 6, abs=1e-3)
    assert split["identities_split_across_tracks"] == 1 and split["tracks_merging_identities"] == 0
    merge = fa.idf1([("tube_A", "t1"), ("tube_B", "t1")])
    assert merge["idtp"] == 1 and merge["tracks_merging_identities"] == 1
    missing = fa.idf1([("tube_A", "t1"), ("tube_A", None)])
    assert missing["cells_without_track"] == 1 and missing["idf1"] == pytest.approx(2 / 3, abs=1e-3)
    assert fa.idf1([])["idf1"] is None


def test_scoreboard_on_an_empty_record_says_nothing_is_labelled(synthetic, capsys):
    root, workspace, arms = synthetic["root"], synthetic["workspace"], synthetic["arms"]
    report = fa.score_arms(
        workspace_dir=workspace, record_path=None, arms=arms, repository_root=root
    )
    assert report["record_summary"] == {
        "cells": 8,
        "labelled": 0,
        "by_state": {"unlabeled": 8},
        "with_identity": 0,
        "identities": [],
    }
    assert all(arm["cells_scored"] == 0 for arm in report["arms"])
    table = fa.scoreboard_markdown(report)
    assert table.startswith("# Anchor scoreboard: 0 / 8 cells labelled")
    assert "No cell is labelled yet" in table
    code = fa.main(
        [
            "score",
            "--workspace",
            str(workspace),
            "--arms",
            ",".join(f"{k}={v}" for k, v in arms.items()),
            "--output",
            str(root / "runs" / "score-empty"),
        ]
    )
    assert code == 0
    assert (root / "runs" / "score-empty" / "anchor_scoreboard.md").is_file()
    assert "0 / 8 cells labelled" in capsys.readouterr().out


def test_scoreboard_on_a_partial_record_scores_masks_boxes_hidden_and_identity(synthetic):
    root, workspace, arms = synthetic["root"], synthetic["workspace"], synthetic["arms"]
    template = json.loads((workspace / "decisions.template.json").read_text())
    for cell in template["cells"]:
        if cell["raw_frame"] == 916:
            cell["decision"] = 0
            if cell["label"] == "50ml_tube#0":
                cell["instance_identity"] = "tube_A"
        elif cell["view"] == "T4" and cell["label"] == "50ml_tube#0":
            cell["decision"] = "box"
            cell["instance_identity"] = "tube_A"
        elif cell["view"] == "fpv" and cell["label"] == "50ml_tube#0":
            cell["decision"] = "hidden"
    template["author"] = "test"
    record = root / "runs" / "decisions.json"
    record.write_text(json.dumps(template), encoding="utf-8")
    report = fa.score_arms(
        workspace_dir=workspace, record_path=record, arms=arms, repository_root=root
    )
    summary = report["record_summary"]
    assert summary["labelled"] == 6 and summary["by_state"] == {
        "mask": 4,
        "box": 1,
        "hidden": 1,
        "unlabeled": 2,
    }
    # The centrifuge cells carry the pre-filled singleton identity.
    assert summary["with_identity"] == 5 and summary["identities"] == ["centrifuge", "tube_A"]
    by_name = {arm["arm"]: arm for arm in report["arms"]}
    b, c, a = by_name["b"], by_name["c"], by_name["a"]
    # Arm (b) reproduces its own accepted masks: the scorer's built-in check.
    assert b["cells_scored"] == 5 and b["mask_iou_mean"] == 1.0
    # The 'box' cell compares (b)'s mask bbox with the reference box, not with its own mask.
    assert 0.95 < b["box_iou_mean"] < 1.0
    assert b["hidden_false_positives"] == 1  # its mask on the cell marked hidden
    assert b["identity"]["six_view_frame"]["idf1"] == 1.0
    assert b["identity"]["all_labelled"]["idf1"] == 1.0
    # Arm (c): shifted masks score below 1, the tube split across two ids on the six-view
    # frame, its fpv tube on 1000 is marked hidden and it has no mask there.
    assert c["cells_scored"] == 5 and 0.5 < c["mask_iou_mean"] < 1.0
    assert c["hidden_false_positives"] == 0
    assert c["identity"]["six_view_frame"]["identities_split_across_tracks"] == 1
    assert c["identity"]["six_view_frame"]["idf1"] < 1.0
    tube_916 = next(
        cell
        for cell in c["cells"]
        if cell["raw_frame"] == 916 and cell["view"] == "T4" and cell["slot"] == 1
    )
    assert tube_916["track_id"] == "50ml_tube-002" and tube_916["mask_iou"] == pytest.approx(
        0.75, abs=0.01
    )
    # Arm (a): boxes only, scored by box IoU, masks not applicable, identity from detector slots.
    assert a["has_masks"] is False and a["mask_iou_mean"] is None
    assert a["cells_scored"] == 5 and a["box_iou_mean"] > 0.8
    assert a["hidden_false_positives"] == 0
    assert a["identity"]["all_labelled"]["idf1"] == 1.0
    assert a["by_class"]["50ml_tube"]["box_iou_mean"] > 0.8
    table = fa.scoreboard_markdown(report)
    assert "6 / 8 cells labelled" in table and "| b | 5 | 1.000 | 1.000 | 0.9" in table
    assert "| 0 | 1 (" in table  # (b): no missing cell, one hidden false positive
    exported = fa.export_record(
        workspace_dir=workspace,
        record_path=record,
        output=root / "docs" / "qa" / "record.json",
        repository_root=root,
    )
    assert exported["author"] == "test" and exported["counts"]["labelled"] == 6
    states = {(e["raw_frame"], e["view"], e["slot"]): e for e in exported["anchors"]}
    labelled = states[(916, "T4", 1)]
    assert labelled["state"] == "mask" and labelled["candidate_kind"] == "arm_b_tight"
    assert labelled["mask_sha256"] and labelled["instance_identity"] == "tube_A"
    assert states[(1000, "fpv", 1)]["state"] == "hidden"
    assert (
        states[(1000, "T4", 1)]["state"] == "box" and states[(1000, "T4", 1)]["mask_sha256"] is None
    )
    text = (root / "docs" / "qa" / "record.json").read_text()
    assert "candidates/" not in text  # no mask leaves runs/


def test_record_validation_and_arm_parsing(synthetic):
    root, workspace = synthetic["root"], synthetic["workspace"]
    template = json.loads((workspace / "decisions.template.json").read_text())
    template["cells"][0]["decision"] = 7
    bad = root / "bad.json"
    bad.write_text(json.dumps(template), encoding="utf-8")
    with pytest.raises(ValueError, match="candidate 7 does not exist"):
        fa.score_arms(workspace_dir=workspace, record_path=bad, arms={}, repository_root=root)
    template["cells"][0]["decision"] = "maybe"
    bad.write_text(json.dumps(template), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown decision"):
        fa.score_arms(workspace_dir=workspace, record_path=bad, arms={}, repository_root=root)
    assert fa.parse_arms("a=/x, b=/y") == {"a": Path("/x"), "b": Path("/y")}
    with pytest.raises(ValueError):
        fa.parse_arms("nopath")


def test_decode_worker_command_and_environment(tmp_path: Path):
    import argparse

    args = argparse.Namespace(
        external_python=Path("/usr/bin/python3"),
        model=tmp_path / "m.pt",
        gpu_guard="vram",
        gpu_guard_profile="sam3_1280",
        expected_peak_vram_bytes=None,
        allow_gpu_neighbour=[4242],
    )
    command = fa.decode_worker_command(
        args, requests=tmp_path / "r.json", results=tmp_path / "s.json"
    )
    assert command[0] == "/usr/bin/python3" and command[2] == "decode-worker"
    assert command[-2:] == ["--allow-gpu-neighbour", "4242"]
    env = fa.worker_environment()
    assert env["CUDA_VISIBLE_DEVICES"] and str(fa.MUGGLED_SAM_SOURCE) in env["PYTHONPATH"]


@pytest.mark.real_data
def test_committed_config_is_what_the_rule_selects_on_the_arm_b_run(tmp_path: Path):
    """Re-running `select` on the retained arm (b) observations reproduces the committed
    frames (same rule, same seed)."""
    arm_b = Path("runs/finebio-arms-P03_03_01-20260925/b-box-decode-arm")
    seeds = Path("runs/finebio-seeds-P03_03_01-20260925/with-plate/seeds.json")
    if not (arm_b / "observations.jsonl").is_file() or not seeds.is_file():
        pytest.skip("trial-1 arm (b) run is not on this machine")
    output = tmp_path / "anchors.json"
    code = fa.main(
        [
            "select",
            "--clip",
            "configs/clips/finebio_P03_03_01_600-4200.json",
            "--arm-b",
            str(arm_b),
            "--seeds",
            str(seeds),
            "--output",
            str(output),
        ]
    )
    assert code == 0
    fresh = fa.load_config(output)
    committed = fa.load_config(COMMITTED_CONFIG)
    assert [(f["raw_frame"], f["origin"]) for f in fresh["frames"]] == [
        (f["raw_frame"], f["origin"]) for f in committed["frames"]
    ]
    assert fresh["fixed_view"] == committed["fixed_view"]
