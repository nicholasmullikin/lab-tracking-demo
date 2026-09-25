"""`battle-detector-seed`: association, the seed rules, cap, start frames, the worker files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from conftest import require_artifact

from battle import detector_seed as ds
from battle.muggled_worker import _corrections_by_frame, parse_box_stream, slot_start_frames

PREFLIGHT_DETECTIONS = Path("runs/preflight-finebio-20260924/detections")
HW = (1080, 1920)


def _instance(
    view: str,
    cls: str,
    index: int,
    frames: list[int],
    boxes: list[tuple[float, float, float, float]],
    scores: list[float] | float = 0.8,
) -> ds.Instance:
    if isinstance(scores, float):
        scores = [scores] * len(frames)
    return ds.Instance(
        instance=f"{view}/{cls}/{index}",
        view=view,
        object_class=cls,
        frames=list(frames),
        boxes=[tuple(float(v) for v in b) for b in boxes],
        scores=list(scores),
        interpolated=[False] * len(frames),
    )


def _static(x: float, y: float, w: float, h: float, n: int, jitter: float = 0.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        dx, dy = rng.uniform(-jitter, jitter, 2) if jitter else (0.0, 0.0)
        out.append((x + dx, y + dy, x + w + dx, y + h + dy))
    return out


def _moving(x: float, y: float, w: float, h: float, n: int, dx_total: float):
    return [
        (x + dx_total * i / (n - 1), y, x + w + dx_total * i / (n - 1), y + h) for i in range(n)
    ]


def _record(view, frame, dets, interpolated=False):
    return {
        "view": view,
        "frame_index": frame,
        "image_hw": list(HW),
        "detections": [
            {"class": c, "class_id": 0, "score": s, "box_xyxy_px": list(b)} for c, s, b in dets
        ],
        "interpolated": interpolated,
    }


# ------------------------------------------------------------------------------- association


def test_iou_matrix_and_greedy_pairs():
    a = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=float)
    b = np.array([[1, 1, 11, 11], [20, 20, 30, 30], [100, 100, 110, 110]], dtype=float)
    ious = ds.iou_matrix(a, b)
    assert ious.shape == (2, 3)
    assert ious[1, 1] == 1.0 and ious[0, 2] == 0.0 and 0.6 < ious[0, 0] < 0.7
    assert ds.greedy_pairs(ious, 0.3) == [(1, 1), (0, 0)]
    assert ds.greedy_pairs(np.zeros((0, 3)), 0.3) == []


def test_associate_instances_links_same_class_across_a_gap_and_splits_on_class():
    params = ds.SeedParams(max_gap=3)
    records = []
    for frame in range(30):
        dets = [("pen", 0.9, (500, 500, 540, 520))]
        # The plate drifts 2 px per frame and is missing on frames 10..12 (a gap of 3).
        if frame not in (10, 11, 12):
            dets.append(("cell_culture_plate", 0.6, (100 + 2 * frame, 100, 200 + 2 * frame, 160)))
        # A second plate far away, same class, and a low-score box that is never associated.
        dets.append(("cell_culture_plate", 0.5, (1500, 800, 1600, 860)))
        dets.append(("cell_culture_plate", 0.1, (900, 900, 950, 950)))
        records.append(_record("T1", 600 + frame, dets))
    instances = ds.associate_instances("T1", records, params)
    by_class = {}
    for inst in instances:
        by_class.setdefault(inst.object_class, []).append(inst)
    assert len(by_class["pen"]) == 1 and len(by_class["pen"][0].frames) == 30
    plates = sorted(by_class["cell_culture_plate"], key=lambda i: i.boxes[0][0])
    assert len(plates) == 2
    assert plates[0].frames == [600 + f for f in range(30) if f not in (10, 11, 12)]
    assert plates[1].frames == list(range(600, 630))
    assert plates[0].instance.startswith("T1/cell_culture_plate/")
    # A gap longer than max_gap starts a new instance.
    records = [
        _record("T1", f, [("pen", 0.9, (0, 0, 10, 10))]) for f in [*range(5), *range(10, 15)]
    ]
    instances = ds.associate_instances("T1", records, params)
    assert [len(i.frames) for i in instances] == [5, 5]
    summary = ds.lifetimes_summary(instances)
    assert summary["pen"]["instances"] == 2 and summary["pen"]["lifetime_frames"]["max"] == 5


def test_dense_start_frame_and_seed_candidates():
    assert ds.dense_start_frame([1, 2, 3], 15) is None
    detected = [0, 1, 50, 51, *range(100, 130)]
    assert ds.dense_start_frame(detected, 15) == 100
    frames = list(range(100, 130))
    boxes = [(0.0, 0.0, 10.0, 10.0)] * 30
    picks = ds.seed_candidates(frames, boxes, [0.5] * 30, frames, 100, ds.SeedParams())
    assert [p["frame"] for p in picks] == [100, 115]
    picks = ds.seed_candidates(
        frames, boxes, [0.5] * 30, frames, 100, ds.SeedParams(min_persistence=5)
    )
    assert [p["frame"] for p in picks] == [100, 105, 110]


# ------------------------------------------------------------------------------- the rules


def _bench(view: str = "T2", plate_dx: float = 60.0) -> list[ds.Instance]:
    n = 60
    frames = list(range(600, 600 + n))
    instances = [
        _instance(
            view, "cell_culture_plate", 0, frames, _moving(800, 500, 180, 90, n, plate_dx), 0.5
        ),
        _instance(view, "pcr_machine", 0, frames, _static(300, 700, 150, 120, n, jitter=3.0)),
        _instance(view, "centrifuge", 0, frames, _static(1200, 200, 200, 190, n, jitter=1.0)),
        _instance(view, "pen", 0, frames, _static(1500, 900, 60, 20, n, jitter=1.0)),
        _instance(view, "trash_can", 0, frames, _static(100, 100, 120, 200, n)),
        _instance(view, "blue_pipette", 0, frames, _static(1000, 300, 60, 300, n)),
        _instance(view, "micro_tube_rack", 0, frames, _static(500, 300, 200, 100, n)),
        _instance(view, "50ml_tube", 0, frames[:10], _static(1700, 600, 40, 120, 10)),
    ]
    # Hand over the pipette on 30% of the frames, over the trash can on 5%.
    hand_boxes = []
    for i in range(n):
        if i < 18:
            hand_boxes.append((950.0, 250.0, 1100.0, 700.0))
        elif i < 21:
            hand_boxes.append((80.0, 80.0, 260.0, 320.0))
        else:
            hand_boxes.append((1700.0, 1000.0, 1800.0, 1070.0))
    instances.append(_instance(view, "right_hand", 0, frames, hand_boxes, 0.9))
    # Four static tubes in the rack and one that walks out of it.
    for k in range(4):
        instances.append(
            _instance(view, "micro_tube", k, frames, _static(520 + 40 * k, 330, 20, 40, n))
        )
    instances.append(_instance(view, "micro_tube", 4, frames, _moving(680, 330, 20, 40, n, 400.0)))
    return instances


def test_seed_rules_move_in_hand_container_group_and_detector_only():
    params = ds.SeedParams(slot_cap=20)
    selected = ds.select_view_slots("T2", _bench(), image_hw=HW, params=params)
    by_label = {s["label"]: s for s in selected["slots"]}
    assert by_label["cell_culture_plate#0"]["rule"] == "moves"
    assert by_label["cell_culture_plate#0"]["max_move_px"] > 20
    assert by_label["blue_pipette#0"]["rule"] == "in_hand"
    assert 0.25 < by_label["blue_pipette#0"]["in_hand_fraction"] < 0.35
    assert by_label["pcr_machine#0"]["rule"] == "container"
    assert by_label["pcr_machine#0"]["role"] == "container"
    assert by_label["centrifuge#0"]["rule"] == "container"
    assert by_label["micro_tube_rack#0"]["rule"] == "container"
    group = by_label["micro_tube_group#0"]
    assert group["rule"] == "group" and group["role"] == "group"
    assert sorted(group["members"]) == [f"T2/micro_tube/{k}" for k in range(4)]
    assert group["instance"] == "T2/micro_tube_rack/0"
    walker = by_label["micro_tube#0"]
    assert walker["rule"] == "moves" and walker["instance"] == "T2/micro_tube/4"
    assert "pen" in selected["detector_only"] and "trash_can" in selected["detector_only"]
    assert selected["not_persistent_instances"] == 1  # the 10-frame 50ml tube
    assert selected["hand_instances"] == 1
    assert all(s["start_frame"] == 600 for s in selected["slots"])
    assert [s["slot"] for s in selected["slots"]] == list(range(len(selected["slots"])))
    # Landmarks, the movers, the group, the held object, the rack.
    rules = [s["rule"] for s in selected["slots"]]
    assert rules == [
        "container",
        "container",
        "moves",
        "moves",
        "group",
        "in_hand",
        "container",
    ]
    assert [s["label"] for s in selected["slots"]][:2] == ["centrifuge#0", "pcr_machine#0"]
    assert [s["label"] for s in selected["slots"]][-1] == "micro_tube_rack#0"


def test_start_frame_is_the_first_dense_detected_frame():
    n = 80
    frames = list(range(600, 600 + n))
    # Detected at 0.3+ only from frame 640 on (score 0.25 before): the slot starts at 640.
    scores = [0.25] * 40 + [0.6] * 40
    plate = _instance(
        "T1", "cell_culture_plate", 0, frames, _moving(800, 500, 180, 90, n, 80), scores
    )
    selected = ds.select_view_slots("T1", [plate], image_hw=HW, params=ds.SeedParams())
    slot = selected["slots"][0]
    assert slot["start_frame"] == 640 and slot["detected_frames"] == 40
    assert [c["frame"] for c in slot["seed_candidates"]] == [640, 655, 670]


def test_scene_motion_cancels_head_motion_and_keeps_the_mover():
    rng = np.random.default_rng(1)
    n = 40
    frames = list(range(n))
    statics = rng.uniform(200, 1500, size=(12, 2))
    instances = []
    for k, (x, y) in enumerate(statics):
        boxes = []
        for i in range(n):
            # The head drifts 3 px/frame and zooms 0.2 % per frame about the image centre.
            scale = 1 + 0.002 * i
            cx = 960 + (x - 960) * scale + 3 * i
            cy = 540 + (y - 540) * scale + 1.5 * i
            boxes.append((cx - 30, cy - 20, cx + 30, cy + 20))
        instances.append(_instance("fpv", "pen", k, frames, boxes))
    mover = []
    for i in range(n):
        scale = 1 + 0.002 * i
        cx = 960 + (700 - 960) * scale + 3 * i + 4 * i  # 4 px/frame against the scene
        cy = 540 + (400 - 540) * scale + 1.5 * i
        mover.append((cx - 40, cy - 40, cx + 40, cy + 40))
    instances.append(_instance("fpv", "cell_culture_plate", 0, frames, mover))
    params = ds.SeedParams()
    motion = ds.SceneMotion(instances, params, (1440, 1920))
    hands: dict[int, list] = {}
    static_moves = [
        ds.instance_stats(
            inst, params=params, hands_by_frame=hands, racks=[], motion=motion
        ).max_move_px
        for inst in instances[:12]
    ]
    plate = ds.instance_stats(
        instances[-1], params=params, hands_by_frame=hands, racks=[], motion=motion
    )
    assert max(static_moves) < 2.0
    # 4 px/frame against the scene; the medians of the two 20-frame halves sit 20 frames apart.
    assert 60 < plate.max_move_px < 100
    assert plate.max_move_raw_px > plate.max_move_px
    # A box cut by the frame border never counts as a move.
    clipped = _instance(
        "fpv", "pen", 99, frames, [(0.0, 100.0, 60.0, 140.0 + 5 * i) for i in frames]
    )
    assert ds.box_clipped(clipped.boxes[0], (1440, 1920), 4.0)


def test_fpv_dynamic_slots_need_a_fixed_view_to_corroborate_the_class():
    # 120 px over 60 frames: 60 px per 30-frame window, over the fpv's 40 px threshold.
    instances = _bench("fpv", plate_dx=120.0)
    hw = (1440, 1920)
    without = ds.select_view_slots(
        "fpv", instances, image_hw=hw, params=ds.SeedParams(), corroborated_classes=set()
    )
    assert all(s["rule"] not in ("moves", "in_hand") for s in without["slots"])
    assert {u["instance"] for u in without["uncorroborated"]} >= {
        "fpv/cell_culture_plate/0",
        "fpv/blue_pipette/0",
    }
    with_plate = ds.select_view_slots(
        "fpv",
        instances,
        image_hw=hw,
        params=ds.SeedParams(),
        corroborated_classes={"cell_culture_plate"},
    )
    rules = {s["label"]: s["rule"] for s in with_plate["slots"]}
    assert rules["cell_culture_plate#0"] == "moves"
    assert "blue_pipette#0" not in rules
    # The tube that walked out of the rack is not corroborated either: no slot, recorded.
    group = next(s for s in with_plate["slots"] if s["rule"] == "group")
    assert len(group["members"]) == 4
    assert "fpv/micro_tube/4" in {u["instance"] for u in with_plate["uncorroborated"]}


def test_slot_cap_keeps_landmarks_then_persistence_then_movement():
    n = 60
    frames = list(range(600, 600 + n))
    instances = [
        _instance("T1", "centrifuge", 0, frames, _static(1200, 200, 200, 190, n)),
        _instance("T1", "cell_culture_plate", 0, frames, _moving(800, 500, 180, 90, n, 60)),
        _instance("T1", "blue_pipette", 0, frames[:40], _moving(100, 100, 60, 300, 40, 200)),
        _instance("T1", "red_pipette", 0, frames[:40], _moving(400, 100, 60, 300, 40, 90)),
        _instance("T1", "50ml_tube_rack", 0, frames, _static(500, 800, 200, 100, n)),
        _instance("T1", "blue_tip_rack", 0, frames, _static(900, 800, 200, 100, n)),
    ]
    selected = ds.select_view_slots("T1", instances, image_hw=HW, params=ds.SeedParams(slot_cap=4))
    assert [s["label"] for s in selected["slots"]] == [
        "centrifuge#0",
        "cell_culture_plate#0",
        "blue_pipette#0",
        "red_pipette#0",
    ]
    assert [s["label"] for s in selected["capped"]] == ["50ml_tube_rack#0", "blue_tip_rack#0"]
    assert all(s["slot"] is None for s in selected["capped"])


def test_nested_and_same_box_duplicates_are_suppressed():
    n = 40
    frames = list(range(n))
    big = _instance("T1", "cell_culture_plate", 0, frames, _static(700, 380, 150, 140, n), 0.5)
    small = _instance(
        "T1", "cell_culture_plate", 1, frames[:30], _static(710, 450, 100, 60, 30), 0.3
    )
    apart = _instance("T1", "cell_culture_plate", 2, frames, _static(1500, 380, 150, 140, n), 0.5)
    blue = _instance("T1", "blue_pipette", 0, frames, _static(100, 100, 60, 300, n), 0.6)
    yellow = _instance("T1", "yellow_pipette", 0, frames, _static(102, 101, 60, 300, n), 0.4)
    params = ds.SeedParams()
    stats = {
        i.instance: ds.instance_stats(
            i,
            params=params,
            hands_by_frame={},
            racks=[],
            motion=ds.SceneMotion([big, small, apart, blue, yellow], params, HW),
        )
        for i in (big, small, apart, blue, yellow)
    }
    suppressed = ds.suppress_nested_duplicates([big, small, apart, blue, yellow], stats, params)
    assert set(suppressed) == {small.instance, yellow.instance}
    assert suppressed[small.instance]["duplicate_of"] == big.instance
    assert suppressed[small.instance]["reason"] == "nested_same_class"
    assert suppressed[yellow.instance]["duplicate_of"] == blue.instance
    assert suppressed[yellow.instance]["reason"] == "same_box_other_class"


# ------------------------------------------------------------------------- worker files


def _slots_and_instances(tmp_path: Path):
    n = 60
    frames = list(range(600, 600 + n))
    instances = [
        _instance("T1", "centrifuge", 0, frames, _static(1200, 200, 200, 190, n)),
        _instance("T1", "cell_culture_plate", 0, frames, _moving(800, 500, 180, 90, n, 60)),
        _instance("T1", "blue_pipette", 0, frames[20:], _moving(100, 100, 60, 300, 40, 200)),
    ]
    for k in range(3):
        instances.append(
            _instance("T1", "micro_tube", k, frames, _static(520 + 40 * k, 330, 20, 40, n))
        )
    instances.append(_instance("T1", "micro_tube_rack", 0, frames, _static(500, 300, 200, 100, n)))
    selected = ds.select_view_slots("T1", instances, image_hw=HW, params=ds.SeedParams())
    ds.write_instances(tmp_path, "T1", instances, image_hw=HW, frames=frames)
    return selected, {i.instance: i for i in instances}


def test_box_stream_and_schedule_pass_the_worker_parsers(tmp_path: Path):
    selected, by_id = _slots_and_instances(tmp_path)
    labels = [s["label"] for s in selected["slots"]]
    assert labels == [
        "centrifuge#0",
        "cell_culture_plate#0",
        "blue_pipette#0",
        "micro_tube_group#0",
        "micro_tube_rack#0",
    ]
    stream_path = tmp_path / "box_streams" / "T1.jsonl"
    report = ds.write_box_stream(
        stream_path, selected["slots"], by_id, window_start=600, window_end=660
    )
    frames, concepts = parse_box_stream(stream_path.read_text())
    assert concepts == tuple(labels) and len(frames) == 60
    assert report["frames_with_boxes"] == 60
    # The pipette starts at analysis frame 20; the group box is the union of its members.
    assert [b["label"] for b in frames[0]] == [x for x in labels if x != "blue_pipette#0"]
    assert [b["label"] for b in frames[20]] == labels
    group = next(b for b in frames[0] if b["label"] == "micro_tube_group#0")
    assert group["box_xyxy_px"] == [520.0, 330.0, 620.0, 370.0]
    assert group["source"] == "finebio_dino_group"
    accepted = []
    for record in selected["slots"]:
        if record["label"] == "micro_tube_rack#0":
            continue  # unseeded in this scenario
        cand = record["seed_candidates"][0]
        accepted.append(
            {
                **record,
                "status": "accepted",
                "seed": {"frame": cand["frame"], "box": cand["box"], "kind": "tight"},
            }
        )
    schedule_path = tmp_path / "schedules" / "T1.json"
    report = ds.write_schedule(schedule_path, accepted, window_start=600)
    payload = json.loads(schedule_path.read_text())
    targets = tuple(s["target"] for s in payload["seeds"])
    assert targets == (
        "centrifuge#0",
        "cell_culture_plate#0",
        "blue_pipette#0",
        "micro_tube_group#0",
    )
    assert [s["initial_multiplex_slot"] for s in payload["seeds"]] == [0, 1, 2, 3]
    pipette = payload["seeds"][2]
    assert pipette["start_frame"] == 20 and pipette["selected_by"] == "detector_reseed"
    assert "start_frame" not in payload["seeds"][0]
    assert payload["seeds"][0]["selected_by"] == "detector"
    semantics = "append_prompt_memory_and_reset_frame_memory"
    grouped = _corrections_by_frame(
        {**payload, "memory_semantics": semantics}, targets, memory_semantics=semantics
    )
    assert sorted(grouped) == [20] and grouped[20][0]["seed_start"] is True
    assert slot_start_frames(payload) == {2: 20}
    assert report["start_frames"]["blue_pipette#0"] == 20
    validated = ds.validate_worker_files(stream_path, schedule_path)
    assert validated["schedule"]["slots"] == 4


def test_decode_requests_metrics_and_margins():
    box = (100.0, 100.0, 200.0, 150.0)
    assert ds.expand_box(box, 0.15, HW) == (85.0, 92.5, 215.0, 157.5)
    assert ds.expand_box((0.0, 0.0, 100.0, 100.0), 0.5, HW)[:2] == (0.0, 0.0)
    mask = np.zeros(HW, dtype=bool)
    mask[110:140, 120:180] = True
    metrics = ds.mask_metrics(mask, box)
    assert metrics["mask_bbox_px"] == [120, 110, 180, 140] and metrics["mask_area_px"] == 1800
    assert metrics["fill_ratio"] == pytest.approx(1800 / 5000)
    assert 0.3 < metrics["mask_bbox_iou_vs_box"] < 0.4
    assert ds.mask_metrics(np.zeros(HW, dtype=bool), box)["mask_bbox_iou_vs_box"] == 0.0
    slots_doc = {
        "views": {
            "T1": {
                "image_hw": list(HW),
                "slots": [
                    {
                        "slot": 0,
                        "label": "centrifuge#0",
                        "class": "centrifuge",
                        "rule": "container",
                        "instance": "T1/centrifuge/0",
                        "members": [],
                        "seed_candidates": [
                            {"frame": 600, "box": [1200, 200, 1400, 390], "score": 0.9},
                            {"frame": 615, "box": [1200, 200, 1400, 390], "score": 0.9},
                        ],
                    },
                    {
                        "slot": 1,
                        "label": "pen#0",
                        "class": "pen",
                        "rule": "moves",
                        "instance": "T1/pen/0",
                        "members": [],
                        "seed_candidates": [{"frame": 600, "box": [10, 10, 50, 30], "score": 0.5}],
                    },
                ],
            }
        }
    }
    requests = ds.build_decode_requests(
        slots_doc,
        output=Path("/nonexistent"),
        trial="P03_01_01",
        videos={"T1": Path("/v/T1.mp4")},
        params=ds.SeedParams(),
    )
    attempts = requests["views"]["T1"]["slots"][0]["attempts"]
    assert [(a["frame"], a["kind"]) for a in attempts] == [
        (600, "tight"),
        (600, "margin"),
        (615, "tight"),
        (615, "margin"),
    ]
    assert attempts[1]["prompt_box"] == [1170.0, 171.5, 1430.0, 418.5]
    assert attempts[0]["negatives"] == []
    assert requests["accept_iou"] == 0.6 and requests["encoder_side"] == 1280


def test_cli_end_to_end_on_synthetic_detections_without_a_gpu(tmp_path: Path):
    detections = tmp_path / "detections"
    detections.mkdir()
    n = 60
    for view, hw in (("T1", HW), ("fpv", (1440, 1920))):
        lines = []
        for i in range(n):
            frame = 600 + i
            dets = [
                ("centrifuge", 0.9, (1200, 200, 1400, 390)),
                ("cell_culture_plate", 0.6, (800 + i, 500, 980 + i, 590)),
                ("pen", 0.8, (1500, 900, 1560, 920)),
                ("left_hand", 0.9, (700 + i, 450, 900 + i, 650)),
            ]
            if view == "fpv":
                dets.append(("trash_can", 0.7, (100 + 2 * i, 100, 220 + 2 * i, 300)))
            lines.append(json.dumps(_record(view, frame, dets)))
        (detections / f"{view}.jsonl").write_text("\n".join(lines) + "\n")
    output = tmp_path / "run"
    assert (
        ds.main(
            [
                "instances",
                "--detections",
                str(detections),
                "--views",
                "T1,fpv",
                "--output",
                str(output),
                "--start",
                "600",
                "--end",
                "660",
            ]
        )
        == 0
    )
    summary = json.loads((output / "instances/summary.json").read_text())
    assert summary["views"]["T1"]["instances"] == 4 and summary["views"]["fpv"]["instances"] == 5
    assert ds.main(["select", "--output", str(output)]) == 0
    slots = json.loads((output / "slots.json").read_text())
    assert slots["window"] == {"start": 600, "end": 660}
    t1 = {s["label"]: s["rule"] for s in slots["views"]["T1"]["slots"]}
    assert t1 == {"centrifuge#0": "container", "cell_culture_plate#0": "moves"}
    fpv = {s["label"]: s["rule"] for s in slots["views"]["fpv"]["slots"]}
    # The plate is dynamic in T1, so the fpv may open it (here under the hand: 30 px per
    # window is under the fpv's 40 px); the trash can moves in the fpv only and stays out.
    assert fpv == {"centrifuge#0": "container", "cell_culture_plate#0": "in_hand"}
    assert [u["instance"] for u in slots["views"]["fpv"]["uncorroborated"]] == ["fpv/trash_can/0"]
    # Without the MuggledSAM interpreter the decode is blocked; the worker files still exist.
    code = ds.main(
        [
            "decode",
            "--output",
            str(output),
            "--trial",
            "P03_01_01",
            "--raw-root",
            str(tmp_path),
            "--external-python",
            str(tmp_path / "missing-python"),
        ]
    )
    assert code == 3
    seeds = json.loads((output / "seeds.json").read_text())
    assert seeds["decode"]["state"] == "blocked"
    assert seeds["provenance"] == "auto" and seeds["selected_by"] == "detector"
    for view in ("T1", "fpv"):
        assert all(s["status"] == "unseeded" for s in seeds["views"][view]["slots"])
        assert (output / "box_streams" / f"{view}.jsonl").is_file()
        assert seeds["views"][view]["schedule"] is None
        frames, concepts = parse_box_stream((output / "box_streams" / f"{view}.jsonl").read_text())
        assert len(frames) == 60 and concepts == ("centrifuge#0", "cell_culture_plate#0")
        assert min(frames) == 0 and max(frames) == 59
    # A decisions file rejects one slot; the filtered stream renumbers the rest.
    decisions = {
        "decisions": [{"view": "T1", "label": "centrifuge#0", "decision": "reject", "note": "x"}]
    }
    (tmp_path / "decisions.json").write_text(json.dumps(decisions))
    filtered = ds.apply_decisions(output, tmp_path / "decisions.json")
    assert filtered["provenance"] == "human_filtered"
    assert [s["label"] for s in filtered["views"]["T1"]["slots"]] == ["cell_culture_plate#0"]
    assert filtered["views"]["T1"]["rejected"][0]["label"] == "centrifuge#0"
    frames, concepts = parse_box_stream((output / "filtered/box_streams/T1.jsonl").read_text())
    assert concepts == ("cell_culture_plate#0",) and frames[0][0]["slot"] == 0


def test_decode_worker_command_and_environment(tmp_path: Path):
    args = ds.build_parser().parse_args(
        [
            "decode",
            "--output",
            str(tmp_path),
            "--trial",
            "P03_03_01",
            "--allow-gpu-neighbour",
            "4242",
            "--external-python",
            "/py",
        ]
    )
    command = ds.decode_worker_command(
        args, requests=tmp_path / "requests.json", results=tmp_path / "results.json"
    )
    assert command[:3] == ["/py", str(Path(ds.__file__).resolve()), "decode-worker"]
    assert "--allow-gpu-neighbour" in command and "4242" in command
    assert command[command.index("--gpu-guard-profile") + 1] == "sam3_1280"
    env = ds.worker_environment()
    assert env["CUDA_VISIBLE_DEVICES"] in ("0", env.get("CUDA_VISIBLE_DEVICES"))
    assert str(ds.MUGGLED_SAM_SOURCE) in env["PYTHONPATH"]
    assert ds.class_family("blue_pipette") == "pipette"
    assert ds.class_family("micro_tube_group") == "group"
    assert ds.class_family("8_tube_stripes_rack") == "rack"
    assert ds.class_family("centrifuge") == "machine"


# ------------------------------------------------------------------------------ real data


@pytest.mark.real_data
def test_preflight_detections_select_the_plate_and_the_held_pipette(tmp_path: Path):
    """On P03_01_01 (78 frames) the plate and the held pipette open slots by the rule, the
    PCR machine opens as a container, never as a moving object."""
    detections = require_artifact(PREFLIGHT_DETECTIONS)
    output = tmp_path / "seeds"
    assert ds.main(["instances", "--detections", str(detections), "--output", str(output)]) == 0
    assert ds.main(["select", "--output", str(output)]) == 0
    slots = json.loads((output / "slots.json").read_text())
    plate_views, pipette_views = set(), set()
    for view in ("T1", "T2", "T3", "T4", "T5"):
        selected = slots["views"][view]
        for slot in selected["slots"]:
            if slot["class"] == "pcr_machine":
                assert slot["rule"] == "container" and slot["max_move_px"] < 20
            if slot["class"] == "cell_culture_plate" and slot["rule"] in ("moves", "in_hand"):
                plate_views.add(view)
            if slot["class"].endswith("pipette") and slot["rule"] in ("moves", "in_hand"):
                pipette_views.add(view)
        assert selected["slots"][0]["label"] == "centrifuge#0"
        assert not any(
            row["instance"].split("/")[1] == "pcr_machine"
            for rows in selected["detector_only"].values()
            for row in rows
        )
    assert plate_views >= {"T1", "T2", "T3", "T4"}
    assert pipette_views >= {"T1", "T2", "T3", "T5"}
    fpv = slots["views"]["fpv"]
    assert "cell_culture_plate" in fpv["corroborated_classes"]
    assert any(s["class"] == "cell_culture_plate" for s in fpv["slots"])
    assert not any(s["class"] == "pcr_machine" and s["rule"] == "moves" for s in fpv["slots"])
