"""`battle-finebio-arms` (p4-arms): the occlusion inventory, the label-free measures, the (d)
schedule builder, the sanity check, the plan-slot annotation and the arm-(a) pipeline on the
preflight fixtures (no data/, no GPU)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
from finebio_fixtures import FIXTURE_DIR, load_preflight_fixtures

from battle.finebio_arms import (
    Episode,
    annotate_episodes,
    area_stability,
    build_reseed_schedule,
    detector_index,
    episodes_from_tracks,
    find_worker_run,
    inventory_summary,
    load_clip,
    main,
    mark_plan_slots,
    pooled_decision,
    sanity_check,
    scoreboard,
    scoreboard_markdown,
    slot_measures,
)
from battle.multiview_schemas import FineBioObservation, Track3D, TrackEvent, read_jsonl
from battle.multiview_tracks import ResidualRow

CLIP_CONFIG = Path("configs/clips/finebio_P03_03_01_600-4200.json")


@pytest.fixture(scope="module")
def fixtures():
    return load_preflight_fixtures()


def _track_row(frame: int, tid: str, state: str, cls: str = "micro_tube", **kw) -> Track3D:
    defaults = dict(
        position_cm=(10.0, 5.0, -1.0),
        uncertainty_cm=1.0,
        support_views=("T1", "T2", "T3") if state == "observed" else (),
        confidence=0.8 if state == "observed" else 0.2,
        abstain=state != "observed",
        support_slots={"T1": "micro_tube#0"} if state == "observed" else {},
    )
    defaults.update(kw)
    return Track3D(frame_index=frame, track_id=tid, object_class=cls, state=state, **defaults)


def test_episodes_from_tracks_reacquired_lost_and_open():
    rows = [
        _track_row(10, "micro_tube-001", "observed"),
        _track_row(11, "micro_tube-001", "coasting", frames_unobserved=1),
        _track_row(12, "micro_tube-001", "coasting", frames_unobserved=2),
        _track_row(13, "micro_tube-001", "coasting", frames_unobserved=3),
        _track_row(14, "micro_tube-001", "observed"),
        _track_row(15, "micro_tube-001", "coasting", frames_unobserved=1),
        _track_row(16, "micro_tube-001", "lost", frames_unobserved=2),
        _track_row(10, "centrifuge-002", "observed", cls="centrifuge"),
        _track_row(11, "centrifuge-002", "coasting", cls="centrifuge", frames_unobserved=1),
        _track_row(12, "centrifuge-002", "coasting", cls="centrifuge", frames_unobserved=2),
        _track_row(20, "left_hand-003", "observed", cls="left_hand"),
        _track_row(21, "left_hand-003", "coasting", cls="left_hand", frames_unobserved=1),
    ]
    episodes = sorted(episodes_from_tracks(rows), key=lambda e: (e.track_id, e.start_frame))
    assert [(e.track_id, e.start_frame, e.outcome) for e in episodes] == [
        ("centrifuge-002", 11, "open_at_window_end"),
        ("left_hand-003", 21, "open_at_window_end"),
        ("micro_tube-001", 11, "reacquired"),
        ("micro_tube-001", 15, "lost"),
    ]
    first = episodes[2]
    assert first.last_observed_frame == 10 and first.reacquired_frame == 14
    assert first.length_frames == 4 and first.reacquisition_latency_frames == 4
    assert first.identical_instance_class and not first.group_slot and not first.lost_at_timeout
    assert first.tracker_handled is None and first.tracker_states == {"coasting": 3}
    second = episodes[3]
    assert second.lost_at_timeout and second.length_frames == 1 and second.end_frame == 16
    assert not episodes[0].identical_instance_class
    # With the tracker extensions on, contained / held rows are support-0 rows of the same
    # episode and the tracker's handling is recorded.
    ext_rows = [
        _track_row(10, "micro_tube-005", "observed"),
        _track_row(
            11, "micro_tube-005", "contained", frames_unobserved=1, container_id="centrifuge"
        ),
        _track_row(
            12, "micro_tube-005", "contained", frames_unobserved=2, container_id="centrifuge"
        ),
        _track_row(13, "micro_tube-005", "observed"),
        _track_row(20, "50ml_tube-006", "observed", cls="50ml_tube"),
        _track_row(
            21,
            "50ml_tube-006",
            "held",
            cls="50ml_tube",
            frames_unobserved=1,
            held_by="left_hand-001",
        ),
        _track_row(22, "50ml_tube-006", "coasting", cls="50ml_tube", frames_unobserved=2),
        _track_row(23, "50ml_tube-006", "lost", cls="50ml_tube", frames_unobserved=3),
    ]
    held, contained = sorted(episodes_from_tracks(ext_rows), key=lambda e: e.track_id)
    assert contained.tracker_handled == "contained" and contained.container_id == "centrifuge"
    assert contained.outcome == "reacquired" and contained.tracker_states == {"contained": 2}
    assert held.tracker_handled == "held" and held.held_by == "left_hand-001"
    assert held.outcome == "lost" and held.tracker_states == {"held": 1, "coasting": 1}
    summary = inventory_summary([contained, held] + episodes, coast_timeout=30)
    assert summary["tracker_handled"]["by_state"] == {"coasting": 4, "contained": 1, "held": 1}
    assert summary["tracker_handled"]["contained_by_container"] == {"centrifuge": 1}
    assert summary["tracker_handled"]["contained_reacquired"] == 1


def _det(view: str, frame: int, cls: str, box, score: float = 0.9) -> FineBioObservation:
    return FineBioObservation(
        view=view,
        frame_index=frame,
        slot=f"{cls}#0",
        object_class=cls,
        detector_score=score,
        box_xyxy_px=tuple(float(v) for v in box),
        pose_valid=True,
        source="detector",
    )


def _around(pixel, half=(40.0, 30.0)):
    return (pixel[0] - half[0], pixel[1] - half[1], pixel[0] + half[0], pixel[1] + half[1])


def test_annotate_episodes_infers_held_contained_and_association_miss(fixtures):
    cams = fixtures.fixed_cameras()
    point = np.array([10.0, 5.0, -1.0])
    frame = 1800
    pixels = {v: cams[v].project(point)[0] for v in cams}
    rows = []
    # Episode 1: a hand box around the projection in T1 and T2 -> held.
    rows += [_det(v, frame, "left_hand", _around(pixels[v], (60, 60))) for v in ("T1", "T2")]
    # Episode 2 (another point): the micro-tube rack around it in T3, T4, T5 -> contained.
    point2 = np.array([-8.0, 12.0, -0.5])
    pixels2 = {v: cams[v].project(point2)[0] for v in cams}
    rows += [
        _det(v, frame, "micro_tube_rack", _around(pixels2[v], (80, 50))) for v in ("T3", "T4", "T5")
    ]
    # Episode 3 (a third point): a same-class detector box within the gate in T1, T3 -> miss.
    point3 = np.array([0.0, -10.0, -2.0])
    pixels3 = {v: cams[v].project(point3)[0] for v in cams}
    rows += [_det(v, frame, "50ml_tube", _around(pixels3[v] + 5.0, (15, 25))) for v in ("T1", "T3")]
    index = detector_index(rows)
    episodes = [
        Episode(
            "micro_tube-001",
            "micro_tube",
            frame,
            frame - 1,
            frame + 5,
            6,
            "lost",
            None,
            None,
            tuple(point),
            ("T1", "T2", "T3"),
            {},
        ),
        Episode(
            "micro_tube-002",
            "micro_tube",
            frame,
            frame - 1,
            frame + 5,
            6,
            "lost",
            None,
            None,
            tuple(point2),
            ("T3", "T4", "T5"),
            {},
        ),
        Episode(
            "50ml_tube-003",
            "50ml_tube",
            frame,
            frame - 1,
            frame + 5,
            6,
            "lost",
            None,
            None,
            tuple(point3),
            ("T1", "T3"),
            {},
        ),
        Episode(
            "left_hand-004",
            "left_hand",
            frame,
            frame - 1,
            frame + 5,
            6,
            "lost",
            None,
            None,
            tuple(point),
            ("T1", "T2"),
            {},
        ),
    ]
    events = [
        TrackEvent(
            frame_index=frame + 8,
            track_id="micro_tube-009",
            kind="birth",
            payload={"possibly_same_as": ["micro_tube-001"], "views": ["T1", "T2", "T3"]},
        ),
        TrackEvent(
            frame_index=frame + 8,
            track_id="micro_tube-009",
            kind="ambiguous",
            payload={"coasting_tracks": ["micro_tube-001"]},
        ),
    ]
    annotate_episodes(
        episodes,
        det_index=index,
        events=events,
        fixed_cams=cams,
        fpv_source=lambda f: None,
        containers=("centrifuge", "micro_tube_rack"),
        probes=("left_hand", "right_hand"),
        gate_px=30.0,
    )
    held, contained, miss, hand = episodes
    assert held.inferred_state == "held" and set(held.in_hand_views) == {"T1", "T2"}
    assert len(held.projected_views) == 5 and not held.held_in_all_projected_views
    assert held.successor_tracks[0]["track_id"] == "micro_tube-009" and held.ambiguous_events == 1
    assert contained.inferred_state == "contained"
    assert contained.in_container_views == {v: ["micro_tube_rack"] for v in ("T3", "T4", "T5")}
    assert miss.inferred_state == "detector_visible_association_miss"
    assert set(miss.detector_within_gate_views) == {"T1", "T3"}
    assert hand.probe_class and hand.inferred_state == "held"
    summary = inventory_summary(episodes, coast_timeout=30)
    assert summary["episodes"] == 3 and summary["probe_episodes_excluded"] == 1
    assert summary["extension_needs"]["held"] == 1
    assert summary["extension_needs"]["contained"] == 1
    assert summary["extension_needs"]["contained_by_container_class"] == {"micro_tube_rack": 1}
    assert summary["extension_needs"]["group_tracks_identical_instance_episodes"] == 3
    assert summary["extension_needs"]["group_tracks_with_ambiguity_or_possibly_same_as"] == 1
    assert summary["extension_needs"]["detector_visible_association_miss"] == 1
    assert summary["by_outcome"] == {"lost": 3}


def _sam3(view, frame, slot, cls, mask_bbox, *, area, score=0.9, source="sam3_video", box=None):
    return FineBioObservation(
        view=view,
        frame_index=frame,
        slot=slot,
        object_class=cls,
        mask_bbox_px=mask_bbox,
        mask_centroid_px=((mask_bbox[0] + mask_bbox[2]) / 2, (mask_bbox[1] + mask_bbox[3]) / 2),
        mask_area_px=area,
        sam3_object_score=score,
        box_xyxy_px=box,
        pose_valid=True,
        source=source,
    )


def test_slot_measures_reference_is_the_best_same_class_detector_box():
    frames = range(600, 606)
    rows = []
    for f in frames:
        rows.append(_det("T1", f, "centrifuge", (100, 100, 300, 300)))
        rows.append(_det("T1", f, "centrifuge", (700, 700, 720, 720), score=0.4))
        rows.append(_det("T1", f, "micro_tube", (10, 10, 30, 40)))
        # The slot's mask matches the strong box exactly on 5 frames and the wrong box on one.
        bbox = (100.0, 100.0, 300.0, 300.0) if f != 605 else (700.0, 700.0, 720.0, 720.0)
        rows.append(
            _sam3("T1", f, "centrifuge#0", "centrifuge", bbox, area=40000 if f != 605 else 400)
        )
        # A group slot: never scored against a box; area jumps once.
        rows.append(
            _sam3(
                "T1", f, "micro_tube_group#0", "micro_tube_group", (0.0, 0.0, 50.0, 50.0), area=2500
            )
        )
    # A slot with a mask but no same-class detector box in that view/frame.
    rows.append(
        _sam3("T1", 600, "trash_can#0", "trash_can", (500.0, 500.0, 600.0, 600.0), area=9000)
    )
    # A row without a mask (fallback box) is counted, not measured.
    rows.append(
        FineBioObservation(
            view="T1",
            frame_index=601,
            slot="trash_can#0",
            object_class="trash_can",
            mask_bbox_px=(500.0, 500.0, 600.0, 600.0),
            pose_valid=True,
            source="sam3_video",
            provenance={"mask": "absent"},
        )
    )
    residuals = [
        ResidualRow(
            frame_index=f,
            track_id="centrifuge-001",
            view="T1",
            residual_px=2.0,
            gate_px=30.0,
            slot="centrifuge#0",
            source="sam3_video",
            confirmed=True,
        )
        for f in range(600, 603)
    ]
    measures = slot_measures(
        rows,
        detector_index(rows),
        residuals,
        window=frames,
        slot_start={("T1", "micro_tube_group#0"): 603},
    )
    by_slot = {(s["view"], s["slot"]): s for s in measures["per_slot"]}
    centrifuge = by_slot[("T1", "centrifuge#0")]
    assert centrifuge["frames_with_mask"] == 6 and centrifuge["detector_iou"]["n"] == 6
    assert centrifuge["detector_iou"]["median"] == 1.0
    assert centrifuge["detector_iou"]["fraction_ge_0p5"] == 1.0  # the wrong box still matches
    assert (
        centrifuge["frames_associated_by_tracker"] == 3 and centrifuge["associated_fraction"] == 0.5
    )
    assert centrifuge["area"]["steps"] == 5 and centrifuge["area"]["jump_fraction_gt_0p5"] == 0.2
    group = by_slot[("T1", "micro_tube_group#0")]
    assert group["group"] and group["detector_iou"]["n"] == 0
    assert group["frames_in_window_from_start"] == 3 and group["frames_with_mask"] == 6
    trash = by_slot[("T1", "trash_can#0")]
    assert trash["frames_mask_without_detector_reference"] == 1 and trash["rows_without_mask"] == 1
    assert trash["detector_iou"]["n"] == 0
    assert measures["pooled"]["detector_iou"]["n"] == 6
    assert measures["per_view"]["T1"]["slots"] == 3
    assert measures["pooled"]["rows_without_mask_total"] == 1


def test_area_stability_uses_consecutive_frames_only():
    series = [(1, 100), (2, 100), (3, 10), (5, 10), (6, 100)]
    out = area_stability(series)
    assert out["steps"] == 3 and out["jump_fraction_gt_0p5"] == pytest.approx(2 / 3, abs=1e-4)
    assert out["relative_step_median"] == 0.9 and out["area_median_px"] == 100
    assert area_stability([])["steps"] == 0


def test_reseed_schedule_one_correction_per_slot_per_k_frames():
    schedule = {
        "seeds": [
            {
                "target": "centrifuge#0",
                "initial_multiplex_slot": 0,
                "prompt_box_xyxy_px": [0, 0, 10, 10],
            },
            {
                "target": "50ml_tube#0",
                "initial_multiplex_slot": 1,
                "prompt_box_xyxy_px": [0, 0, 10, 10],
                "start_frame": 100,
                "selected_by": "detector_reseed",
            },
        ],
        "corrections": [],
    }
    residuals = [
        ResidualRow(
            frame_index=650,
            track_id="centrifuge-001",
            view="T2",
            residual_px=1.0,
            gate_px=30.0,
            slot="centrifuge#0",
            source="sam3_video",
            confirmed=True,
        ),
        ResidualRow(
            frame_index=650,
            track_id="centrifuge-001",
            view="T1",
            residual_px=1.0,
            gate_px=30.0,
            slot="centrifuge#0",
            source="sam3_video",
            confirmed=True,
        ),
        ResidualRow(
            frame_index=640,
            track_id="50ml_tube-002",
            view="T2",
            residual_px=1.0,
            gate_px=30.0,
            slot="50ml_tube#0",
            source="sam3_video",
            confirmed=True,
        ),
        # A detector-source association is not a SAM3 slot.
        ResidualRow(
            frame_index=640,
            track_id="pen-003",
            view="T2",
            residual_px=1.0,
            gate_px=30.0,
            slot="pen#0",
            source="detector",
            confirmed=True,
        ),
    ]

    def event(frame, tid, kind, view="T2", box=(10.0, 20.0, 110.0, 80.0)):
        return TrackEvent(
            frame_index=frame,
            track_id=tid,
            kind=kind,
            payload={"view": view, "box_xyxy_px": list(box), "frames_missing": 5},
        )

    events = [
        event(660, "centrifuge-001", "handoff_reseed"),
        event(670, "centrifuge-001", "detector_reseed"),  # within K of 660 -> skipped
        event(700, "centrifuge-001", "detector_reseed"),
        event(660, "centrifuge-001", "handoff_reseed", view="T1"),  # other view
        event(690, "50ml_tube-002", "handoff_reseed"),  # analysis 90 <= start 100 -> skipped
        event(720, "50ml_tube-002", "handoff_reseed", box=(-5.0, 0.0, 3000.0, 50.0)),
        event(660, "pen-003", "handoff_reseed"),  # no SAM3 slot in T2
        event(4300, "centrifuge-001", "handoff_reseed"),  # outside the window
        event(600, "centrifuge-001", "handoff_reseed"),  # analysis frame 0
        TrackEvent(frame_index=680, track_id="centrifuge-001", kind="coasting", payload={}),
    ]
    payload, report = build_reseed_schedule(
        events,
        residuals,
        schedule,
        view="T2",
        frame_offset=600,
        window_frames=3600,
        k=30,
        image_size=(1920, 1080),
    )
    assert [(c["frame_index"], c["target"], c["selected_by"]) for c in payload["corrections"]] == [
        (60, "centrifuge#0", "track_reproject"),
        (100, "centrifuge#0", "detector_reseed"),
        (120, "50ml_tube#0", "track_reproject"),
    ]
    assert payload["corrections"][2]["prompt_box_xyxy_px"] == [0.0, 0.0, 1920.0, 50.0]
    assert payload["corrections"][0]["multiplex_slot"] == 0
    assert payload["corrections"][2]["multiplex_slot"] == 1
    assert payload["seeds"] == schedule["seeds"]
    assert report["corrections"] == 3 and report["events_considered"] == 8
    assert report["skipped"] == {
        "within_k_of_previous": 1,
        "before_slot_start": 1,
        "no_sam3_slot_for_track_in_view": 1,
        "outside_window": 2,
    }
    # The worker's own parser accepts the payload.
    from battle.muggled_worker import _corrections_by_frame

    grouped = _corrections_by_frame(
        {**payload, "memory_semantics": "append_prompt_memory_and_reset_frame_memory"},
        ("centrifuge#0", "50ml_tube#0"),
        memory_semantics="append_prompt_memory_and_reset_frame_memory",
    )
    assert sorted(grouped) == [60, 100, 120]


def _write_worker_run(root: Path, *, frames: int, ms_per_frame: float, iou_good: bool) -> Path:
    import cv2

    run = root / "muggledsam-arm-box-decode-t1-20260925t000000z-r1280"
    (run / "masks").mkdir(parents=True)
    width, height = 200, 100
    lines = []
    for k in range(frames):
        mask = np.zeros((height, width), dtype=np.uint8)
        x0, y0, x1, y1 = (20, 20, 60, 60) if iou_good else (20, 20, 30, 30)
        mask[y0:y1, x0:x1] = 255
        uri = f"masks/{k:06d}_00.png"
        cv2.imwrite(str(run / uri), mask)
        lines.append(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "view_id": "T1",
                    "analysis_frame_index": k,
                    "source_seconds": k / 29.97,
                    "objects": [
                        {
                            "object_id": "sam3-00",
                            "label": "centrifuge#0",
                            "confidence": 0.9,
                            "box": {"x": 0.1, "y": 0.2, "width": 0.2, "height": 0.4},
                            "mask": {"uri": uri, "storage": "external_artifact", "format": "png"},
                            "object_score": 0.9,
                            "iou_prediction": 0.9,
                            "prompt_box": {"x": 0.1, "y": 0.2, "width": 0.2, "height": 0.4},
                            "source": "sam3_decode",
                            "prompt_source": "finebio_dino",
                            "prompt_score": 0.8,
                        }
                    ],
                    "hands": [],
                }
            )
        )
    (run / "observations.jsonl").write_text("\n".join(lines) + "\n")
    (run / "worker_result.json").write_text(
        json.dumps(
            {
                "state": "succeeded",
                "frames_processed": frames,
                "elapsed_seconds": frames * ms_per_frame / 1000 + 4.0,
                "time_to_first_usable_output_seconds": 4.0 + ms_per_frame / 1000,
                "gpu_peak_vram_bytes": 2_300_000_000,
                "masks_written": frames,
                "runtime_settings": {
                    "mode": "box_stream",
                    "concepts": ["centrifuge#0"],
                    "box_decode_timing_ms": {
                        "per_prompted_frame": {"median": ms_per_frame, "n": frames},
                        "image_encode": {"median": ms_per_frame - 2, "n": frames},
                    },
                },
            }
        )
    )
    (run / "manifest.json").write_text(
        json.dumps({"method_statuses": [{"state": "succeeded"}], "run_id": run.name})
    )
    return run


def test_sanity_check_passes_and_fails_on_timing_and_iou(tmp_path: Path):
    good = _write_worker_run(tmp_path / "good", frames=5, ms_per_frame=150.0, iou_good=True)
    result = sanity_check(good, arm="b", view="T1", frames=5, frame_offset=600)
    assert result["passed"] and result["ms_within_2x"] and result["iou_median_ge_threshold"]
    assert result["mask_bbox_iou"]["median"] == 1.0 and result["masks_measured"] == 5
    assert result["timing"]["ms_per_frame_elapsed"] > 150
    assert find_worker_run(tmp_path / "good") == good
    slow = _write_worker_run(tmp_path / "slow", frames=5, ms_per_frame=400.0, iou_good=True)
    assert not sanity_check(slow, arm="b", view="T1", frames=5, frame_offset=600)["passed"]
    off = _write_worker_run(tmp_path / "off", frames=5, ms_per_frame=150.0, iou_good=True)
    bad = sanity_check(off, arm="b", view="T1", frames=5, frame_offset=600, reference_ms=50.0)
    assert not bad["passed"] and bad["iou_median_ge_threshold"]
    wrong = _write_worker_run(tmp_path / "wrong", frames=5, ms_per_frame=150.0, iou_good=False)
    assert not sanity_check(wrong, arm="b", view="T1", frames=5, frame_offset=600)["passed"]
    with pytest.raises(FileNotFoundError):
        find_worker_run(tmp_path / "missing")


def test_mark_plan_slots_rewrites_rule_and_seeds_md(tmp_path: Path):
    slots = {
        "views": {
            "T1": {
                "slots": [
                    {"label": "centrifuge#0", "class": "centrifuge", "rule": "moves"},
                    {
                        "label": "cell_culture_plate#0",
                        "class": "cell_culture_plate",
                        "rule": "container",
                    },
                ]
            }
        }
    }
    (tmp_path / "seeds.json").write_text(json.dumps(slots))
    (tmp_path / "slots.json").write_text(json.dumps(slots))
    (tmp_path / "seeds.md").write_text(
        "| view | slot | label | role | rule | start |\n|---|---|---|---|---|---|\n"
        "| T1 | 0 | centrifuge#0 | container | moves | 600 |\n"
        "| T1 | 1 | cell_culture_plate#0 | container | container | 600 |\n"
    )
    report = mark_plan_slots(
        tmp_path, ("cell_culture_plate",), rule="landmark_plan_shortlist", note="plan-driven"
    )
    assert report["changed"] == {
        "seeds.json": ["T1/cell_culture_plate#0"],
        "slots.json": ["T1/cell_culture_plate#0"],
    }
    seeds = json.loads((tmp_path / "seeds.json").read_text())
    plate = seeds["views"]["T1"]["slots"][1]
    assert (
        plate["rule"] == "landmark_plan_shortlist" and plate["rule_from_seed_tool"] == "container"
    )
    assert seeds["views"]["T1"]["slots"][0]["rule"] == "moves"
    assert seeds["plan_driven_slots"]["landmark_plan_shortlist"]["note"] == "plan-driven"
    md = (tmp_path / "seeds.md").read_text()
    assert "| T1 | 1 | cell_culture_plate#0 | container | landmark_plan_shortlist | 600 |" in md
    assert "| T1 | 0 | centrifuge#0 | container | moves | 600 |" in md
    assert "## Plan-driven slots" in md
    # Idempotent: a second call changes nothing.
    assert (
        mark_plan_slots(
            tmp_path, ("cell_culture_plate",), rule="landmark_plan_shortlist", note="plan-driven"
        )["changed"]
        == {}
    )


def test_filter_observations_drops_rejected_slots_by_view_and_label(tmp_path: Path):
    from battle.finebio_observations import write_observations

    rows = []
    for f in (600, 601):
        rows.append(_det("T1", f, "centrifuge", (100, 100, 300, 300)))
        rows.append(_det("T3", f, "blue_pipette", (10, 10, 60, 90)))
        rows.append(
            _sam3("T1", f, "centrifuge#0", "centrifuge", (100.0, 100.0, 300.0, 300.0), area=1)
        )
        rows.append(_sam3("T1", f, "blue_pipette#0", "blue_pipette", (0.0, 0.0, 9.0, 9.0), area=1))
        # The same label kept in another view: the match is per (view, label), not per label.
        rows.append(
            _sam3("T3", f, "blue_pipette#0", "blue_pipette", (10.0, 10.0, 60.0, 90.0), area=1)
        )
    rows.append(
        FineBioObservation(
            view="T3",
            frame_index=602,
            slot="blue_pipette#0",
            object_class="blue_pipette",
            mask_bbox_px=(10.0, 10.0, 60.0, 90.0),
            pose_valid=True,
            source="sam3_video",
            provenance={"mask": "absent"},
        )
    )
    src = tmp_path / "sep25"
    src.mkdir()
    write_observations(rows, src / "observations.jsonl")
    (src / "observations_summary.json").write_text(
        json.dumps({"arm": "b", "worker_runs": {"T1": "w/T1", "T3": "w/T3"}, "rows_written": 11})
    )
    seeds = tmp_path / "filtered"
    seeds.mkdir()
    (seeds / "seeds.json").write_text(
        json.dumps(
            {
                "step": "apply-decisions",
                "provenance": "human_filtered",
                "decisions_sha256": "abc",
                "views": {
                    "T1": {
                        "slots": [{"label": "centrifuge#0"}, {"label": "vortex_mixer#0"}],
                        "rejected": [{"label": "blue_pipette#0"}],
                    },
                    "T3": {"slots": [{"label": "blue_pipette#0"}], "rejected": []},
                },
            }
        )
    )
    out = tmp_path / "filtered-arm"
    assert (
        main(
            [
                "filter-observations",
                "--observations",
                str(src / "observations.jsonl"),
                "--seeds",
                str(seeds),
                "--output",
                str(out),
            ]
        )
        == 0
    )
    kept = list(read_jsonl(out / "observations.jsonl", FineBioObservation))
    assert len(kept) == 9 and not any(r.view == "T1" and r.slot == "blue_pipette#0" for r in kept)
    assert sum(r.source == "detector" for r in kept) == 4
    assert sum(r.view == "T3" and r.source == "sam3_video" for r in kept) == 3
    summary = json.loads((out / "observations_summary.json").read_text())
    assert summary["worker_runs"] == {"T1": "w/T1", "T3": "w/T3"}  # the source's, carried
    assert summary["rows_written"] == 9
    assert summary["sam3_rows_by_view"] == {"T1": 2, "T3": 3}
    assert summary["sam3_rows_without_mask_by_view"] == {"T1": 0, "T3": 1}
    sf = summary["seed_filter"]
    assert sf["dropped_slots"] == [{"view": "T1", "slot": "blue_pipette#0", "rows": 2}]
    assert sf["rows_dropped"] == 2 and sf["detector_rows"] == 4
    assert sf["seeds_provenance"] == "human_filtered" and sf["decisions_sha256"] == "abc"
    assert sf["kept_slots_not_in_source"] == {"T1": ["vortex_mixer#0"]}
    # The source is untouched.
    assert len(list(read_jsonl(src / "observations.jsonl", FineBioObservation))) == 11


def _write_axis_worker_run(root: Path, view: str, frames: int) -> None:
    """A box-decode worker run with one compact plate and one 40 x 4 px bar per frame."""
    import cv2

    (root / "masks").mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps({"method_statuses": [{"state": "succeeded"}]}))
    lines = []
    for k in range(frames):
        objects = []
        for slot, label, (x0, y0, x1, y1) in (
            (0, "cell_culture_plate#0", (8, 8, 24, 20)),
            (1, "blue_pipette#0", (10 + k, 30, 50 + k, 34)),
        ):
            mask = np.zeros((48, 64), dtype=np.uint8)
            mask[y0:y1, x0:x1] = 255
            uri = f"masks/{k:06d}_{slot:02d}.png"
            cv2.imwrite(str(root / uri), mask)
            objects.append(
                {
                    "object_id": f"sam3-{slot:02d}",
                    "label": label,
                    "confidence": 0.9,
                    "box": {"x": x0 / 64, "y": y0 / 48, "width": (x1 - x0) / 64, "height": 0.1},
                    "mask": {"uri": uri, "storage": "external_artifact", "format": "png"},
                    "object_score": 0.9,
                    "iou_prediction": 0.9,
                    "prompt_box": {"x": 0.1, "y": 0.15, "width": 0.25, "height": 0.25},
                    "source": "sam3_decode",
                    "prompt_source": "finebio_dino",
                    "prompt_score": 0.42,
                }
            )
        lines.append(
            json.dumps(
                {
                    "view_id": view,
                    "analysis_frame_index": k,
                    "source_seconds": k / 30.0,
                    "objects": objects,
                    "hands": [],
                }
            )
        )
    (root / "observations.jsonl").write_text("\n".join(lines) + "\n")


AXIS_FIELDS = ("mask_axis_px", "mask_elongation", "mask_width_px", "mask_axis_residual_px")
AXIS_PROVENANCE = ("axis_method", "axis_reason")


def _strip_axis(line: str) -> str:
    """A Sep 28 row as the Sep 25 adapter would have written it."""
    row = json.loads(line)
    for key in AXIS_FIELDS:
        row.pop(key, None)
    for key in AXIS_PROVENANCE:
        row.get("provenance", {}).pop(key, None)
    return FineBioObservation.model_validate(row).model_dump_json(
        exclude_none=True, exclude_defaults=True
    )


def test_remeasure_adds_the_axis_fields_and_keeps_every_other_field(tmp_path: Path, fixtures):
    """Sep 28 (p0-axis-observations): `remeasure` re-reads the masks behind an arm's rows and
    adds the axis fields; stripping them from its output gives the input byte for byte, and the
    fixture's detector rows pass through untouched."""
    from battle.finebio_observations import worker_to_observations, write_observations

    worker_root = tmp_path / "b-box-decode"
    for view in ("T1", "T2"):
        _write_axis_worker_run(worker_root / view / f"run-{view.lower()}", view, frames=3)
    rows = [r for r in fixtures.observations if r.source == "detector" and r.frame_index <= 1802]
    for view in ("T1", "T2"):
        rows.extend(worker_to_observations(worker_root / view / f"run-{view.lower()}", view, 1800))
    with_axis = tmp_path / "with-axis.jsonl"
    write_observations(rows, with_axis)
    sep25 = tmp_path / "sep25"
    sep25.mkdir()
    (sep25 / "observations.jsonl").write_text(
        "".join(_strip_axis(line) + "\n" for line in with_axis.read_text().splitlines())
    )
    (sep25 / "observations_summary.json").write_text(json.dumps({"arm": "b", "rows_written": 1}))
    assert "mask_axis_px" not in (sep25 / "observations.jsonl").read_text()
    out = tmp_path / "observations-b"
    assert (
        main(
            [
                "remeasure",
                "--observations",
                str(sep25 / "observations.jsonl"),
                "--worker-root",
                str(worker_root),
                "--output",
                str(out),
                "--jobs",
                "1",
            ]
        )
        == 0
    )
    source_lines = (sep25 / "observations.jsonl").read_text().splitlines()
    out_lines = (out / "observations.jsonl").read_text().splitlines()
    assert len(out_lines) == len(source_lines) == len(rows)
    # Every other field byte-identical, in the source order; the detector rows untouched.
    assert [_strip_axis(line) for line in out_lines] == source_lines
    assert [line for line in out_lines if '"source":"detector"' in line] == [
        line for line in source_lines if '"source":"detector"' in line
    ]
    # And the added fields are what the adapter computes from the same masks.
    assert out_lines == with_axis.read_text().splitlines()
    back = list(read_jsonl(out / "observations.jsonl", FineBioObservation))
    bars = [r for r in back if r.object_class == "blue_pipette" and r.source == "sam3_decode"]
    assert len(bars) == 6 and all(r.mask_axis_px is not None for r in bars)
    assert bars[0].mask_axis_px == ((10.0, 31.5), (50.0, 31.5))
    assert bars[0].mask_elongation == 10.0 and bars[0].mask_width_px == 4.0
    assert all(r.provenance["axis_method"] == "ransac_skeleton" for r in bars)
    plates = [
        r for r in back if r.object_class == "cell_culture_plate" and r.source == "sam3_decode"
    ]
    assert len(plates) == 6
    assert all(r.mask_axis_px is None and r.provenance["axis_reason"] == "compact" for r in plates)
    summary = json.loads((out / "remeasure_summary.json").read_text())
    assert summary["start_frame_by_view"] == {"T1": 1800, "T2": 1800}
    assert summary["sam3_rows"] == 12 and summary["detector_rows_copied"] == len(rows) - 12
    assert summary["old_fields_mismatch_by_view"] == {}
    assert summary["masks_missing_by_view"] == {} and summary["masks_unresolved_by_view"] == {}
    assert summary["reserialisation_mismatches"] == 0
    assert summary["per_class"]["blue_pipette"]["rows_with_axis"] == 6
    assert summary["per_class"]["blue_pipette"]["methods"] == {"ransac_skeleton": 6}
    assert summary["per_class"]["cell_culture_plate"]["axis_fraction"] == 0.0
    assert summary["per_class_view"]["blue_pipette/T1"]["rows"] == 3
    obs_summary = json.loads((out / "observations_summary.json").read_text())
    assert obs_summary["arm"] == "b" and obs_summary["rows_written"] == len(rows)
    assert obs_summary["remeasure"]["worker_runs"]["T1"].endswith("run-t1")
    assert (out / "remeasure_summary.md").read_text().startswith("# Mask axes")
    # The source is untouched.
    assert (sep25 / "observations.jsonl").read_text().splitlines() == source_lines


def _measures(iou_median: float, frac: float, **identity):
    ident = {"id_switches": 3, "fragmentation": 10, "ambiguities": 5}
    ident.update(identity)
    return {
        "slots": {"pooled": {"detector_iou": {"median": iou_median, "fraction_ge_0p5": frac}}},
        "identity": ident,
    }


def test_pooled_decision_rule():
    b = _measures(0.80, 0.95)
    assert not pooled_decision(_measures(0.81, 0.95), b)["run_arm_d"]
    yes = pooled_decision(_measures(0.82, 0.96), b)
    assert yes["run_arm_d"] and yes["c_beats_b_on_iou"]
    assert yes["pooled_detector_iou_median"]["delta_c_minus_b"] == pytest.approx(0.02)
    identity = pooled_decision(
        _measures(0.80, 0.95, id_switches=1, fragmentation=5, ambiguities=2), b
    )
    assert identity["run_arm_d"] and identity["c_clearly_better_on_identity"]
    mixed = pooled_decision(_measures(0.80, 0.95, id_switches=1, fragmentation=5, ambiguities=7), b)
    assert not mixed["run_arm_d"]


def test_load_clip_reads_the_committed_trial_config():
    clip = load_clip(CLIP_CONFIG)
    assert (
        clip.trial == "P03_03_01" and clip.start_frame == 600 and clip.end_frame_exclusive == 4200
    )
    assert clip.views == ("T1", "T2", "T3", "T4", "T5", "fpv") and clip.fpv_view == "fpv"
    assert "centrifuge" in clip.containers and clip.probes == ("left_hand", "right_hand")
    assert clip.camera_config == Path("configs/finebio/cameras/P03_03_01_600-4200.json")
    assert len(clip.frames) == 3600


def _fixture_detections(fixtures, root: Path, frames) -> None:
    """The fixture detector rows as battle-finebio-detect JSONL, per view."""
    per_view: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for row in fixtures.observations:
        if row.source != "detector" or row.frame_index not in frames:
            continue
        per_view[row.view][row.frame_index].append(
            {
                "class": row.object_class,
                "class_id": 0,
                "score": row.detector_score,
                "box_xyxy_px": list(row.box_xyxy_px),
            }
        )
    for view, by_frame in per_view.items():
        hw = [1440, 1920] if view == "fpv" else [1080, 1920]
        with (root / f"{view}.jsonl").open("w") as handle:
            for frame in sorted(by_frame):
                handle.write(
                    json.dumps(
                        {
                            "view": view,
                            "frame_index": frame,
                            "image_hw": hw,
                            "detections": by_frame[frame],
                            "interpolated": False,
                        }
                    )
                    + "\n"
                )


def test_arm_a_pipeline_on_the_preflight_fixtures(tmp_path: Path, fixtures):
    frames = set(range(1798, 1828))
    detections = tmp_path / "dino"
    detections.mkdir()
    _fixture_detections(fixtures, detections, frames)
    clip_path = tmp_path / "clip.json"
    clip_path.write_text(
        json.dumps(
            {
                "config_kind": "finebio_clip_config",
                "clip_id": "fixture-window",
                "trial": "P03_01_01",
                "window": {"start_frame": 1798, "end_frame_exclusive": 1828},
                "views": ["T1", "T2", "T3", "T4", "T5", "fpv"],
                "fixed_views": ["T1", "T2", "T3", "T4", "T5"],
                "fpv_view": "fpv",
                "camera_config": str(FIXTURE_DIR / "cameras.json"),
                "containers": ["centrifuge", "micro_tube_rack", "vortex_mixer"],
                "probes": ["left_hand", "right_hand"],
            }
        )
    )
    gates = tmp_path / "rig.json"
    gates.write_text(json.dumps({"gates": {"association_px": 31.0, "handoff_px": 52.0}}))
    output = tmp_path / "a"
    rc = main(
        [
            "run",
            "--clip-config",
            str(clip_path),
            "--detections",
            str(detections),
            "--arm",
            "a",
            "--gates",
            str(gates),
            "--output",
            str(output),
            "--fpv-poses",
            str(FIXTURE_DIR / "fpv_poses.json"),
        ]
    )
    assert rc == 0
    for name in (
        "observations.jsonl",
        "observations_summary.json",
        "tracks/tracks.jsonl",
        "tracks/events.jsonl",
        "tracks/identity_metrics.json",
        "measures.json",
        "measures.md",
        "occlusion_inventory.jsonl",
        "occlusion_inventory.md",
    ):
        assert (output / name).is_file(), name
    measures = json.loads((output / "measures.json").read_text())
    assert measures["arm"] == "a" and measures["gates"]["association_px"] == 31.0
    assert measures["identity"]["tracks_born"] > 10
    assert "objects_only" in measures["identity"]
    assert measures["residual_px"]["pooled"]["n"] > 0
    assert "slots" not in measures  # no SAM3 rows in arm (a)
    # The claim boundary on the record points at where the human-anchored numbers live and
    # does not claim they are absent (gate 2 was held on Sep 27).
    assert "battle-finebio-anchors" in measures["claim_boundary"]
    assert "no human anchor exists" not in measures["claim_boundary"].lower()
    inventory = measures["occlusion_inventory"]
    assert set(inventory["extension_needs"]) >= {"held", "contained", "unexplained"}
    summary = json.loads((output / "observations_summary.json").read_text())
    assert summary["detector_rows_by_view"]["T1"] > 0 and summary["rows_written"] > 0
    # Re-running from the written observations reproduces the identity metrics.
    rc = main(
        [
            "run",
            "--clip-config",
            str(clip_path),
            "--detections",
            str(detections),
            "--arm",
            "a",
            "--gates",
            str(gates),
            "--output",
            str(output),
            "--fpv-poses",
            str(FIXTURE_DIR / "fpv_poses.json"),
            "--reuse-observations",
        ]
    )
    assert rc == 0
    again = json.loads((output / "measures.json").read_text())
    assert again["identity"] == measures["identity"]
    assert again["observations"]["reused_observations"] is True
    board = scoreboard({"a": measures})
    text = scoreboard_markdown(board)
    assert (
        "| (a) |" in text and board["rows"]["a"]["det_iou_pooled"] is None
        if "det_iou_pooled" in board["rows"]["a"]
        else True
    )
    assert "Tracker extensions" not in text
    # `--ext` runs the tracker with the extensions into tracks-ext/ beside tracks/, with the
    # measures and the inventory suffixed; the core files are not touched.
    core_bytes = {
        name: (output / "tracks" / name).read_bytes()
        for name in ("tracks.jsonl", "events.jsonl", "residuals.jsonl")
    }
    core_measures = (output / "measures.json").read_bytes()
    rc = main(
        [
            "run",
            "--clip-config",
            str(clip_path),
            "--detections",
            str(detections),
            "--arm",
            "a",
            "--gates",
            str(gates),
            "--output",
            str(output),
            "--fpv-poses",
            str(FIXTURE_DIR / "fpv_poses.json"),
            "--reuse-observations",
            "--ext",
        ]
    )
    assert rc == 0
    for name in (
        "tracks-ext/tracks.jsonl",
        "tracks-ext/events.jsonl",
        "tracks-ext/residuals.jsonl",
        "tracks-ext/identity_metrics.json",
        "measures-ext.json",
        "measures-ext.md",
        "occlusion_inventory-ext.jsonl",
        "occlusion_inventory-ext.md",
    ):
        assert (output / name).is_file(), name
    for name, data in core_bytes.items():
        assert (output / "tracks" / name).read_bytes() == data
    assert (output / "measures.json").read_bytes() == core_measures
    ext = json.loads((output / "measures-ext.json").read_text())
    assert ext["tracker_extensions"] is True and ext["tracks_dir"] == "tracks-ext"
    enabled = ext["identity"]["extensions"]["enabled"]
    assert enabled == {"motion_model": True, "containers": True, "group_tracks": True, "held": True}
    assert ext["identity"]["per_class"]["cell_culture_plate"]["tracks_born"] == 1
    assert "tracker_handled" in ext["occlusion_inventory"]
    assert ext["tracker_params"]["extensions"]["containers"] == [
        "centrifuge",
        "micro_tube_rack",
        "vortex_mixer",
    ]
    assert "## Tracker extensions" in (output / "measures-ext.md").read_text()
    board = scoreboard({"a": measures, "a-ext": ext})
    text = scoreboard_markdown(board)
    assert "| (a-ext) |" in text and "Tracker extensions" in text
    assert board["rows"]["a-ext"]["extensions"]["enabled"]["held"] is True
    rc = main(
        [
            "scoreboard",
            "--arm",
            f"a={output}",
            "--arm",
            f"a-ext={output / 'measures-ext.json'}",
            "--output",
            str(tmp_path / "board"),
        ]
    )
    assert rc == 0 and "| (a-ext) |" in (tmp_path / "board" / "scoreboard.md").read_text()
    # The arm (b) worker root is required.
    with pytest.raises(ValueError, match="worker-root"):
        from battle.finebio_arms import build_observations

        build_observations(
            load_clip(clip_path), detections, "b", fpv_poses=FIXTURE_DIR / "fpv_poses.json"
        )
