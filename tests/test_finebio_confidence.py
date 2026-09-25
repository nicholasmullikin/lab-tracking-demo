"""Per-track confidence (p5-confidence) on synthetic rows: the rank combination, the abstain
rule (support, missing signal, bottom decile), the DDETR agreement column and the CLI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from battle.finebio_confidence import (
    SIGNALS,
    TrackConfidence,
    combine,
    ddetr_agreement,
    infer_arm_label,
    load_observation_signals,
    main,
    raw_signals,
    resolve_tracks_dir,
    score_arm,
    summarise,
    summary_markdown,
)
from battle.multiview_schemas import Track3D, read_jsonl


def _track(
    frame: int,
    track_id: str = "plate-001",
    *,
    views: tuple[str, ...] = ("T1", "T2", "T3"),
    residual: dict[str, float] | None = None,
    state: str = "observed",
    cls: str = "cell_culture_plate",
) -> Track3D:
    return Track3D(
        frame_index=frame,
        track_id=track_id,
        object_class=cls,
        position_cm=(0.0, 0.0, -1.0),
        uncertainty_cm=1.0,
        support_views=views,
        support_slots={v: f"{cls}#0" for v in views},
        residual_px=residual or {v: 5.0 for v in views},
        state=state,
        confidence=0.5,
        abstain=False,
    )


def _signal(iou: float | None, score: float | None, det: float | None, box=None) -> dict:
    return {
        "object_class": "cell_culture_plate",
        "detector_box_iou": iou,
        "sam3_object_score": score,
        "detector_score": det,
        "box": box or [10.0, 10.0, 50.0, 50.0],
        "source": "sam3_decode",
    }


def test_raw_signals_aggregate_over_support_views():
    row = _track(600, residual={"T1": 4.0, "T2": 6.0, "T3": 20.0})
    signals = {
        ("T1", 600, "cell_culture_plate#0"): _signal(0.9, 0.8, 0.7),
        ("T2", 600, "cell_culture_plate#0"): _signal(0.7, 0.6, 0.5),
        ("T3", 600, "cell_culture_plate#0"): _signal(None, None, 0.9),  # a detector row
    }
    values, counts, agree = raw_signals(row, signals, {}, {})
    assert values["detector_box_iou"] == 0.8 and counts["detector_box_iou"] == 2
    assert values["sam3_object_score"] == 0.7
    assert np.isclose(values["detector_score"], 0.7) and counts["detector_score"] == 3
    assert values["residual_px"] == 6.0  # the median over the three views
    assert values["support"] == 3.0
    assert agree is None
    # residuals.jsonl wins over the row's own residual_px when present.
    values, _, agree = raw_signals(
        row,
        signals,
        {("plate-001", "T1", 600): 40.0, ("plate-001", "T2", 600): 40.0},
        {("T1", 600, "cell_culture_plate#0"): True, ("T2", 600, "cell_culture_plate#0"): False},
    )
    assert values["residual_px"] == 40.0
    assert agree == 0.5


def test_combine_ranks_within_arm_and_applies_the_abstain_rule():
    rows = [_track(600 + k, f"t-{k:03d}") for k in range(20)]
    raw = []
    for k in range(20):
        raw.append(
            {
                "detector_box_iou": 0.5 + 0.02 * k,
                "residual_px": 20.0 - k,  # lower is better, so it improves with k as well
                "sam3_object_score": 0.4 + 0.03 * k,
                "detector_score": 0.3 + 0.03 * k,
                "support": 3.0,
            }
        )
    ranks, confidence, abstain, reasons = combine(rows, raw)
    assert confidence[0] is not None and confidence[-1] is not None
    assert confidence[-1] > confidence[0]
    # Ranks are percentiles in [0, 1]; the negated residual ranks the smallest residual highest.
    assert ranks[-1]["residual_px"] == 1.0 and ranks[0]["residual_px"] == 0.0
    assert all(0.0 <= r[name] <= 1.0 for r in ranks for name in SIGNALS)
    # Bottom decile of 20 rows: the two lowest combined values abstain, nothing else does.
    assert abstain[:2] == [True, True] and reasons[0] == ("bottom_decile",)
    assert not any(abstain[2:])

    # Support below two abstains even at the top of the ranking.
    rows[-1] = _track(619, "t-019", views=("T1",))
    raw[-1]["support"] = 1.0
    _, _, abstain, reasons = combine(rows, raw)
    assert abstain[-1] and "support_lt_2" in reasons[-1]

    # A missing signal abstains and is named; the confidence is still the mean of the rest.
    raw[10]["sam3_object_score"] = None
    ranks, confidence, abstain, reasons = combine(rows, raw)
    assert abstain[10] and reasons[10] == ("missing:sam3_object_score",)
    assert ranks[10]["sam3_object_score"] is None and confidence[10] is not None


def test_score_arm_boxes_only_rows_abstain_by_construction():
    rows = [_track(600 + k, "rack-001", cls="micro_tube_rack") for k in range(5)]
    signals = {
        (v, 600 + k, "micro_tube_rack#0"): _signal(None, None, 0.9)
        for k in range(5)
        for v in ("T1", "T2", "T3")
    }
    for s in signals.values():
        s["source"] = "detector"
    scored = score_arm(rows, signals, {}, {})
    assert len(scored) == 5
    assert all(r.abstain for r in scored)
    assert all("missing:detector_box_iou" in r.abstain_reasons for r in scored)
    assert all("missing:sam3_object_score" in r.abstain_reasons for r in scored)
    assert all(r.confidence is not None for r in scored)  # ranks of the defined signals
    summary = summarise(scored, arm="a", tracks_dir=Path("tracks"), ddetr_dir=None)
    assert summary["abstain_fraction"] == 1.0
    assert summary["rows_with_all_signals"] == 0
    assert "micro_tube_rack" in summary["per_class"]
    text = summary_markdown(summary)
    assert "Abstain rule" in text and "No Deformable DETR pass" in text


def test_ddetr_agreement_streams_the_view_files(tmp_path: Path):
    ddetr = tmp_path / "ddetr"
    ddetr.mkdir()
    rows = [
        {
            "frame_index": 600,
            "detections": [
                {"class": "cell_culture_plate", "score": 0.9, "box_xyxy_px": [10, 10, 52, 48]},
                {"class": "centrifuge", "score": 0.9, "box_xyxy_px": [10, 10, 50, 50]},
            ],
        },
        {
            "frame_index": 601,
            "detections": [
                {"class": "cell_culture_plate", "score": 0.2, "box_xyxy_px": [10, 10, 50, 50]},
            ],
        },
    ]
    (ddetr / "T1.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    signals = {
        ("T1", 600, "cell_culture_plate#0"): _signal(0.9, 0.9, 0.9),
        ("T1", 601, "cell_culture_plate#0"): _signal(0.9, 0.9, 0.9),  # partner below 0.3
        ("T1", 600, "centrifuge#0"): {**_signal(0.9, 0.9, 0.9), "object_class": "centrifuge"},
        ("T2", 600, "cell_culture_plate#0"): _signal(0.9, 0.9, 0.9),  # no T2 file
    }
    agreement = ddetr_agreement(ddetr, signals)
    assert agreement[("T1", 600, "cell_culture_plate#0")] is True
    assert agreement[("T1", 601, "cell_culture_plate#0")] is False
    assert agreement[("T1", 600, "centrifuge#0")] is True
    assert ("T2", 600, "cell_culture_plate#0") not in agreement
    assert ddetr_agreement(None, signals) == {}


def _write_arm(arm_dir: Path, *, tracks_name: str = "tracks") -> None:
    tracks_dir = arm_dir / tracks_name
    tracks_dir.mkdir(parents=True)
    rows = []
    for k in range(12):
        views = ("T1", "T2", "T3") if k < 10 else ("T1",)
        rows.append(_track(600 + k, "plate-001", views=views, residual={v: 5.0 + k for v in views}))
    rows.append(_track(600, "rack-001", cls="micro_tube_rack"))
    with (tracks_dir / "tracks.jsonl").open("w") as handle:
        for r in rows:
            handle.write(r.model_dump_json(exclude_none=True) + "\n")
    with (tracks_dir / "residuals.jsonl").open("w") as handle:
        for r in rows:
            for v in r.support_views:
                handle.write(
                    json.dumps(
                        {
                            "frame_index": r.frame_index,
                            "track_id": r.track_id,
                            "view": v,
                            "residual_px": 3.0,
                            "gate_px": 30.0,
                            "slot": r.support_slots[v],
                            "source": "sam3_decode",
                            "confirmed": True,
                        }
                    )
                    + "\n"
                )
    with (arm_dir / "observations.jsonl").open("w") as handle:
        for r in rows:
            for v in r.support_views:
                obs = {
                    "view": v,
                    "frame_index": r.frame_index,
                    "slot": r.support_slots[v],
                    "object_class": r.object_class,
                    "detector_score": 0.8,
                    "box_xyxy_px": [10.0, 10.0, 50.0, 50.0],
                    "pose_valid": True,
                    "source": "detector",
                }
                if r.object_class == "cell_culture_plate":
                    obs.update(
                        mask_bbox_px=[11.0, 11.0, 49.0, 49.0],
                        sam3_object_score=0.9,
                        source="sam3_decode",
                        provenance={"detector_box_iou": 0.9, "object_id": "sam3-03"},
                    )
                handle.write(json.dumps(obs) + "\n")
            # A row the tracks never used must be skipped by the loader.
            handle.write(
                json.dumps(
                    {
                        "view": "T5",
                        "frame_index": r.frame_index,
                        "slot": "pen#0",
                        "object_class": "pen",
                        "detector_score": 0.4,
                        "box_xyxy_px": [1.0, 1.0, 5.0, 5.0],
                        "pose_valid": True,
                        "source": "detector",
                    }
                )
                + "\n"
            )


def test_cli_end_to_end_prefers_tracks_ext_and_writes_the_three_files(tmp_path: Path):
    arm_dir = tmp_path / "b-box-decode-arm"
    _write_arm(arm_dir, tracks_name="tracks-ext")
    assert resolve_tracks_dir(arm_dir, None) == arm_dir / "tracks-ext"
    assert resolve_tracks_dir(arm_dir, "tracks-ext") == arm_dir / "tracks-ext"
    assert infer_arm_label(arm_dir) == "b"
    output = tmp_path / "confidence"
    assert (
        main(
            [
                "--arm-dir",
                str(arm_dir),
                "--detections",
                str(tmp_path / "dino"),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    rows = list(read_jsonl(output / "confidence.jsonl", TrackConfidence))
    assert len(rows) == 13
    plate = [r for r in rows if r.track_id == "plate-001"]
    assert all(r.signals["detector_box_iou"] == 0.9 for r in plate)
    assert all(r.signals["residual_px"] == 3.0 for r in plate)  # residuals.jsonl, not the row
    assert sum(r.abstain for r in plate if r.support == 3) <= 2  # the bottom decile at most
    assert all(r.abstain and "support_lt_2" in r.abstain_reasons for r in plate if r.support == 1)
    rack = next(r for r in rows if r.track_id == "rack-001")
    assert rack.abstain and "missing:sam3_object_score" in rack.abstain_reasons
    assert all(r.ddetr_agreement is None for r in rows)  # no ddetr directory beside dino
    summary = json.loads((output / "confidence_summary.json").read_text())
    assert summary["tracks_dir"].endswith("tracks-ext")
    assert summary["rows"] == 13 and summary["arm"] == "b"
    assert (output / "confidence.md").read_text().startswith("# Confidence, arm (b)")


def test_load_observation_signals_keeps_only_wanted_keys(tmp_path: Path):
    arm_dir = tmp_path / "arm"
    _write_arm(arm_dir)
    wanted = {("T1", 600, "cell_culture_plate#0"), ("T5", 600, "pen#0")}
    signals = load_observation_signals(arm_dir / "observations.jsonl", wanted)
    assert set(signals) == wanted
    assert signals[("T1", 600, "cell_culture_plate#0")]["detector_box_iou"] == 0.9
    assert signals[("T5", 600, "pen#0")]["sam3_object_score"] is None
    assert np.isclose(signals[("T5", 600, "pen#0")]["detector_score"], 0.4)
