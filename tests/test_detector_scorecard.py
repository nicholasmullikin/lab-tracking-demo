from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from battle import detector_scorecard as ds
from battle.schemas import (
    DetectorConfidenceRow,
    DetectorOperatingPoint,
    DetectorScore,
    DetectorScorecardManifest,
    DetectorTruthCell,
    LeaveOneFrameOutFold,
    LeaveOneFrameOutResult,
    ProposedAnchorFrame,
    ProposedAnchorFrames,
)

WINDOWS = {"279-408": (279, 408), "573-722": (573, 722), "1020-1172": (1020, 1172)}
ANCHOR_FRAMES = (300, 370, 400, 600, 650, 700, 900, 1050, 1100, 1150, 1200, 1500, 1700)


# ------------------------------------------------------------------------- failure classes


def test_failure_class_assignment_from_windows() -> None:
    classes = {f: ds.failure_class_for_frame(f, WINDOWS) for f in ANCHOR_FRAMES}
    assert classes == {
        300: "occlusion_leak",
        370: "occlusion_leak",
        400: "occlusion_leak",
        600: "rotation_swap",
        650: "rotation_swap",
        700: "rotation_swap",
        900: "outside",
        1050: "rotation_swap",
        1100: "rotation_swap",
        1150: "rotation_swap",
        1200: "distractor",
        1500: "distractor",
        1700: "distractor",
    }
    # Half-open windows: 408 is outside, 407 inside; 1235 has no anchor and is `outside`.
    assert ds.failure_class_for_frame(408, WINDOWS) == "outside"
    assert ds.failure_class_for_frame(407, WINDOWS) == "occlusion_leak"
    assert ds.failure_class_for_frame(1235, WINDOWS) == "outside"
    # Explicit distractor frames override a window they happen to lie in.
    assert ds.failure_class_for_frame(300, WINDOWS, distractor_frames=(300,)) == "distractor"


def test_cell_failed_rule() -> None:
    assert ds.cell_failed("scored", 0.49, 100) is True
    assert ds.cell_failed("scored", 0.5, 100) is False
    assert ds.cell_failed("run_mask_missing", 0.0, 0) is True
    assert ds.cell_failed("hidden_false_positive", None, 1265) is True
    assert ds.cell_failed("hidden_false_positive", None, 300) is False
    assert ds.cell_failed("hidden_correct", None, 0) is False
    assert ds.cell_failed("unlabeled_skipped", None, 0) is None


# ------------------------------------------------------------------------- rank statistics


def test_average_ranks_handles_ties() -> None:
    assert ds.average_ranks(np.array([0.2, 0.1, 0.2, 0.9])).tolist() == [2.5, 1.0, 2.5, 4.0]


def test_auroc_on_a_toy_set() -> None:
    truth = np.array([True, True, False, False])
    assert ds.auroc(np.array([0.9, 0.8, 0.2, 0.1]), truth) == 1.0
    assert ds.auroc(np.array([0.1, 0.2, 0.8, 0.9]), truth) == 0.0
    # One positive above both negatives, one below one of them: 3 of 4 pairs ordered.
    assert ds.auroc(np.array([0.9, 0.15, 0.2, 0.1]), truth) == pytest.approx(0.75)
    # Ties count half.
    assert ds.auroc(np.array([0.5, 0.5, 0.5, 0.5]), truth) == pytest.approx(0.5)
    # Without both classes there is no AUROC.
    assert ds.auroc(np.array([0.9, 0.8]), np.array([True, True])) is None


def test_auroc_ignores_nan_cells() -> None:
    truth = np.array([True, True, False, False, True])
    scores = np.array([0.9, 0.8, 0.2, 0.1, np.nan])
    assert ds.auroc(scores, truth) == 1.0


def test_threshold_selection_and_counts() -> None:
    truth = np.array([True, True, True, False, False, False, False, False])
    scores = np.array([0.9, 0.8, 0.3, 0.7, 0.2, 0.1, 0.05, np.nan])
    # F1-max: threshold 0.8 gives TP 2 FP 0 FN 1 (F1 0.8); 0.3 gives TP 3 FP 1 (F1 0.857).
    threshold = ds.best_f1_threshold(scores, truth)
    assert threshold == pytest.approx(0.3)
    point = ds.confusion_at(scores, truth, threshold)
    assert (point.tp, point.fp, point.fn, point.tn) == (3, 1, 0, 3)
    assert point.precision == pytest.approx(0.75)
    assert point.recall == 1.0
    # Recall >= 0.8 needs all three positives, so the highest such threshold is 0.3 too;
    # recall >= 0.6 is satisfied by two positives at the higher threshold 0.8.
    assert ds.recall_floor_threshold(scores, truth, floor=0.8) == pytest.approx(0.3)
    assert ds.recall_floor_threshold(scores, truth, floor=0.6) == pytest.approx(0.8)
    # No positives: no threshold, and the score reports undefined operating points.
    assert ds.best_f1_threshold(scores, np.zeros(8, dtype=bool)) is None
    score = ds.score_detector("toy", "all", scores, truth)
    assert (score.cells, score.positives, score.negatives, score.undefined) == (7, 3, 4, 1)
    assert score.auroc == pytest.approx(11 / 12)
    assert score.best_f1 is not None and score.best_f1.threshold == pytest.approx(0.3)


def test_best_f1_ties_prefer_the_higher_threshold() -> None:
    truth = np.array([True, False, False])
    scores = np.array([0.9, 0.5, 0.1])
    # Thresholds 0.9 and any value in (0.5, 0.9] give the same F1; the reported one is 0.9.
    assert ds.best_f1_threshold(scores, truth) == pytest.approx(0.9)


def test_normalized_ranks_and_rank_average_with_nan() -> None:
    values = np.array([[3.0, np.nan], [1.0, 2.0]])
    normalized = ds.normalized_ranks(values)
    assert np.isnan(normalized[0, 1])
    assert normalized[1, 0] == 0.0 and normalized[0, 0] == 1.0 and normalized[1, 1] == 0.5
    assert ds.normalized_ranks(np.array([np.nan, 4.0]))[1] == 0.5
    combined = ds.rank_average({"a": normalized, "b": np.full((2, 2), np.nan)}, ("a", "b"))
    assert np.isnan(combined[0, 1]) and combined[0, 0] == 1.0
    with pytest.raises(ValueError):
        ds.rank_average({"a": normalized}, ())


def test_select_top_detectors_skips_undefined_and_breaks_ties_by_name() -> None:
    aurocs = {"d": 0.7, "a": None, "c": 0.9, "b": 0.7, "e": 0.1}
    assert ds.select_top_detectors(aurocs, k=3) == ("c", "b", "d")


def test_leave_one_frame_out_selects_on_the_other_frames() -> None:
    frames = np.repeat(np.array([10, 20, 30, 40]), 2)
    truth = np.array([True, False, True, False, True, False, True, False])
    perfect = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4])
    noise = np.array([0.5, 0.4, 0.1, 0.9, 0.3, 0.2, 0.8, 0.7])
    # `only_frame_10` is perfect on frame 10 and useless elsewhere: it must not be selected
    # when frame 10 is held out, so the fold's chosen set reveals the leave-out logic.
    only_10 = np.array([0.9, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    raw = {"perfect": perfect, "noise": noise, "only_10": only_10}
    normalized = {name: ds.normalized_ranks(values) for name, values in raw.items()}
    result = ds.leave_one_frame_out(
        frames=frames, truth=truth, raw=raw, normalized=normalized, k=1, recall_floor=0.8
    )
    assert [fold.held_out_frame for fold in result.folds] == [10, 20, 30, 40]
    assert all(fold.selected_detectors == ("perfect",) for fold in result.folds)
    assert result.pooled_auroc == 1.0
    assert result.best_f1 is not None
    # Both thresholds land on the lowest training positive (F1-max by its tie-break, the recall
    # floor because 2 of 3 training positives is below 0.8), so the fold holding out frame 40,
    # whose positive scores 0.6 below every training positive, misses it: 3 TP, 1 FN. That is
    # the leave-out logic doing its job; in-sample both would report 4/0/0/4.
    for point in (result.best_f1, result.recall_floor):
        assert point is not None
        assert (point.tp, point.fp, point.fn, point.tn) == (3, 0, 1, 4)
        assert point.threshold is None


# --------------------------------------------------------------------------- feature series


def test_area_jump_and_area_vs_seed_series() -> None:
    areas = np.array([100.0, 100.0, 100.0, 200.0, 0.0])
    jump = ds.area_jump_series(areas, window=15)
    assert np.isnan(jump[0])
    assert jump[1] == 0.0 and jump[3] == pytest.approx(1.0) and jump[4] == pytest.approx(1.0)
    seed = ds.area_vs_seed_series(areas)
    assert seed[0] == 0.0
    assert seed[3] == pytest.approx(abs(np.log(201 / 101)))
    # An empty mask is finite (the +1 floor), not -inf.
    assert np.isfinite(seed[4])
    # A collapse from a zero median is finite too.
    assert np.isfinite(ds.area_jump_series(np.array([0.0, 0.0, 50.0]))[2])


def test_overlap_fraction_and_iou_edge_cases() -> None:
    a = np.zeros((4, 4), dtype=bool)
    a[:2, :2] = True
    b = np.zeros((4, 4), dtype=bool)
    b[1:3, 1:3] = True
    empty = np.zeros((4, 4), dtype=bool)
    assert ds.overlap_fraction(a, [b]) == pytest.approx(0.25)
    assert ds.overlap_fraction(a, [None, empty]) == 0.0
    assert np.isnan(ds.overlap_fraction(empty, [a]))
    assert ds._iou(a, b) == pytest.approx(1 / 7)
    assert ds._iou(a, empty) == 0.0 and ds._iou(a, None) == 0.0
    assert np.isnan(ds._iou(empty, None))


def test_consensus_and_hull_series_from_artifacts(tmp_path: Path) -> None:
    targets = ("chassis", "interior")
    consensus = tmp_path / "consensus"
    consensus.mkdir()
    rows = []
    for frame in range(6):
        rows.append(
            {
                "analysis_frame_index": frame,
                "parts": {
                    "chassis": {
                        "consensus_world_mm": [0, 0, 0],
                        "view_error_px": {"C10379": float(frame * 10)},
                    },
                    "interior": {"consensus_world_mm": None, "views_used": []},
                },
            }
        )
    (consensus / "per_frame.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    (consensus / "manifest.json").write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "view": "C10379",
                        "target": "chassis",
                        "start_frame": 3,
                        "end_frame_exclusive": 9,
                        "contradicts_majority": True,
                    },
                    {
                        "view": "C10379",
                        "target": "chassis",
                        "start_frame": 0,
                        "end_frame_exclusive": 2,
                        "contradicts_majority": False,
                    },
                    {
                        "view": "C10095",
                        "target": "chassis",
                        "start_frame": 0,
                        "end_frame_exclusive": 6,
                        "contradicts_majority": True,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    error, contradicted = ds.consensus_error_series(
        consensus, view="C10379", targets=targets, frame_count=6
    )
    assert error[:, 0].tolist() == [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    assert np.isnan(error[:, 1]).all() and np.isnan(contradicted[:, 1]).all()
    assert contradicted[:, 0].tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]

    hull = tmp_path / "hull"
    hull.mkdir()
    np.savez(
        hull / "hull_series.npz",
        **{
            "iou/chassis/C10379": np.array([0.8, 0.6, np.nan, 0.2, 0.9, 0.9]),
            "iou/interior/C10379": np.full(6, np.nan),
        },
    )
    (hull / "manifest.json").write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "view": "C10379",
                        "target": "chassis",
                        "start_frame": 2,
                        "end_frame_exclusive": 4,
                        "contradicts_majority": True,
                        "kind": "hull_disagreement",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    disagreement, flag = ds.hull_disagreement_series(
        hull, view="C10379", targets=targets, frame_count=6
    )
    assert disagreement[0, 0] == pytest.approx(0.2) and np.isnan(disagreement[2, 0])
    assert np.isnan(flag[2, 0]) and flag[3, 0] == 1.0 and flag[4, 0] == 0.0
    assert np.isnan(disagreement[:, 1]).all()


def test_propose_anchor_frames_excludes_anchors_and_spaces_proposals() -> None:
    targets = ("chassis", "interior")
    frame_count = 400
    values = {name: np.full((frame_count, len(targets)), np.nan) for name in ds.DETECTORS}
    rng = np.random.default_rng(0)
    values["area_jump"] = rng.random((frame_count, len(targets)))
    # Adjacent spikes at 200/210/220 and one far away at 60; the anchor at 300 shields 280-320.
    values["area_jump"][200, 0] = 9.0
    values["area_jump"][210, 0] = 8.0
    values["area_jump"][220, 1] = 7.0
    values["area_jump"][60, 1] = 6.0
    values["area_jump"][300, 0] = 5.0
    values["area_jump"][310, 0] = 5.0
    series = ds.DetectorSeries(targets=targets, frame_count=frame_count, values=values)
    combined = ds.normalized_ranks(values["area_jump"])
    manifest = _toy_manifest(top_detectors=("area_jump",))
    scorecard = ds.Scorecard(manifest=manifest, rows=(), series=series, combined_series=combined)
    proposals = ds.propose_anchor_frames(
        run_name="toy",
        scorecard=scorecard,
        scorecard_uri="runs/toy/scorecard.json",
        existing_anchor_frames=(300,),
        seed=7,
        ranked_count=3,
        random_count=2,
    )
    ranked = [f.analysis_frame_index for f in proposals.frames if f.selection == "detector_ranked"]
    random = [f.analysis_frame_index for f in proposals.frames if f.selection == "random"]
    assert ranked == [200, 220, 60]
    assert proposals.frames[0].target == "chassis" and proposals.frames[1].target == "interior"
    assert proposals.min_spacing_frames == 20 and proposals.random_seed == 7
    assert len(random) == 2
    for frame in [*ranked, *random]:
        assert frame % 10 == 0 and abs(frame - 300) > 20
    for frame in random:
        assert all(abs(frame - other) >= 20 for other in ranked)
    # Same seed, same draw.
    again = ds.propose_anchor_frames(
        run_name="toy",
        scorecard=scorecard,
        scorecard_uri="runs/toy/scorecard.json",
        existing_anchor_frames=(300,),
        seed=7,
        ranked_count=3,
        random_count=2,
    )
    assert again.frames == proposals.frames


def _toy_manifest(*, top_detectors: tuple[str, ...]) -> DetectorScorecardManifest:
    cell = DetectorTruthCell(
        analysis_frame_index=300,
        target="chassis",
        failure_class="occlusion_leak",
        anchor_state="labeled",
        outcome="scored",
        iou=0.9,
        run_area=100,
        failed=False,
    )
    return DetectorScorecardManifest(
        manifest_kind="detector_scorecard",
        run_name="toy",
        run_directory="runs/toy",
        run_observations={
            "uri": "runs/toy/observations.jsonl",
            "sha256": "a" * 64,
            "source": "measured",
        },
        anchor_set={
            "uri": "runs/anchors/anchors/anchor_masks.json",
            "sha256": "b" * 64,
            "source": "measured",
        },
        comparators={},
        view="C10379",
        frame_count=400,
        targets=("chassis", "interior"),
        anchor_frames=(300,),
        truth_rule=ds.TRUTH_RULE,
        failure_class_frames={"occlusion_leak": (300,)},
        class_counts={"all": {"cells": 1, "failed": 0, "undefined": 0}},
        truth_cells=(cell,),
        detectors=ds.DETECTORS,
        detector_definitions=ds.DETECTOR_DEFINITIONS,
        scores=(),
        top_detectors=top_detectors,
        combined=(),
        leave_one_frame_out=LeaveOneFrameOutResult(folds=()),
        confidence_uri="runs/toy/confidence.jsonl",
        generated_at="2026-09-20T12:00:00Z",
        claim_boundary=ds.CLAIM_BOUNDARY,
    )


# ---------------------------------------------------------------------------------- schemas


def test_schema_round_trip() -> None:
    point = DetectorOperatingPoint(
        threshold=0.3, tp=3, fp=1, fn=0, tn=3, precision=0.75, recall=1.0, f1=0.857
    )
    score = DetectorScore(
        detector="area_jump",
        subset="all",
        cells=7,
        positives=3,
        negatives=4,
        undefined=1,
        auroc=0.9,
        best_f1=point,
        recall_floor=point,
    )
    cell = DetectorTruthCell(
        analysis_frame_index=1700,
        target="rear_body",
        failure_class="distractor",
        anchor_state="hidden",
        outcome="hidden_false_positive",
        run_area=1265,
        failed=True,
    )
    manifest = DetectorScorecardManifest(
        manifest_kind="detector_scorecard",
        run_name="toy",
        run_directory="runs/toy",
        run_observations={
            "uri": "runs/toy/observations.jsonl",
            "sha256": "a" * 64,
            "source": "measured",
        },
        anchor_set={
            "uri": "runs/anchors/anchors/anchor_masks.json",
            "sha256": "b" * 64,
            "source": "measured",
        },
        comparators={"dam4sam_large": "runs/dam"},
        view="C10379",
        frame_count=1800,
        targets=("chassis", "rear_body"),
        anchor_frames=ANCHOR_FRAMES,
        truth_rule=ds.TRUTH_RULE,
        failure_class_frames={"distractor": (1200, 1500, 1700)},
        class_counts={"all": {"cells": 1, "failed": 1, "undefined": 0}},
        truth_cells=(cell,),
        detectors=ds.DETECTORS,
        detector_definitions=ds.DETECTOR_DEFINITIONS,
        scores=(score,),
        top_detectors=("area_jump",),
        combined=(score.model_copy(update={"detector": ds.COMBINED_NAME}),),
        leave_one_frame_out=LeaveOneFrameOutResult(
            folds=(
                LeaveOneFrameOutFold(
                    held_out_frame=300, selected_detectors=("area_jump",), best_f1_threshold=0.4
                ),
            ),
            pooled_auroc=0.8,
            best_f1=DetectorOperatingPoint(
                tp=1, fp=0, fn=0, tn=1, precision=1.0, recall=1.0, f1=1.0
            ),
        ),
        abstain_confidence_threshold=0.6,
        confidence_uri="runs/detector-scorecard/toy/confidence.jsonl",
        generated_at="2026-09-20T12:00:00Z",
        claim_boundary=ds.CLAIM_BOUNDARY,
    )
    restored = DetectorScorecardManifest.model_validate_json(manifest.model_dump_json())
    assert restored == manifest
    assert restored.leave_one_frame_out.best_f1 is not None
    assert restored.leave_one_frame_out.best_f1.threshold is None

    row = DetectorConfidenceRow(
        analysis_frame_index=300,
        target="chassis",
        detectors={"area_jump": 0.1, "consensus_error_px": None},
        combined_suspicion=0.7,
        confidence=0.3,
        abstain=True,
        is_anchor_frame=True,
        anchor_truth_failed=True,
    )
    assert DetectorConfidenceRow.model_validate_json(row.model_dump_json()) == row
    proposals = ProposedAnchorFrames(
        manifest_kind="proposed_anchor_frames",
        run_name="toy",
        scorecard_uri="runs/detector-scorecard/toy/scorecard.json",
        frame_step=10,
        exclusion_radius_frames=20,
        existing_anchor_frames=ANCHOR_FRAMES,
        random_seed=20260920,
        frames=(
            ProposedAnchorFrame(
                analysis_frame_index=480,
                selection="detector_ranked",
                rank=1,
                combined_suspicion=0.95,
                target="chassis",
                reason="chassis: combined suspicion 0.95",
            ),
            ProposedAnchorFrame(analysis_frame_index=40, selection="random", reason="random"),
        ),
        claim_boundary=ds.CLAIM_BOUNDARY,
    )
    assert ProposedAnchorFrames.model_validate_json(proposals.model_dump_json()) == proposals
    with pytest.raises(ValueError):
        DetectorConfidenceRow(
            analysis_frame_index=0, target="chassis", detectors={}, confidence=1.5, abstain=False
        )
