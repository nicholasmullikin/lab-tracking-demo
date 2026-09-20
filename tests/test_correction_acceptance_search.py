"""Default-tier tests for the correction acceptance-rule search (synthetic, no data)."""

from __future__ import annotations

import numpy as np

from battle.correction_acceptance_search import (
    MIN_FIT_ACCEPTANCE,
    POOLS,
    AcceptanceRuleSpec,
    CandidateRecord,
    Cell,
    Outcome,
    _variant_prompts,
    apply_rule,
    candidate_passes,
    current_rule,
    hand_hull_mask,
    leave_frames_out,
    meets_bar,
    named_rules,
    objective,
    outcome_for,
    overlap_fraction,
    rule_grid,
    subtract_hand,
    summarize,
)
from battle.schemas import MultiviewRepromptPrompt, PixelBox, PixelPoint


def _candidate(
    key: str,
    *,
    area: int = 5000,
    score: float = 0.8,
    ratio: float | None = 1.0,
    seed_log: float | None = 0.0,
    ray_radii: float | None = 0.3,
    hand: float = 0.0,
    wilor: float = 0.0,
    other: float = 0.0,
    truth_iou: float = 0.7,
    variant: str = "consensus_sphere_box_margin_0.25",
    source: str = "decoded",
    parent: str | None = None,
) -> CandidateRecord:
    return CandidateRecord(
        key=key,
        prompt_id="t000300-b01",
        candidate_index=0,
        variant=variant,
        source=source,
        parent_key=parent,
        area_px=area,
        decoder_score=score,
        area_ratio_vs_expected=ratio,
        log_ratio_vs_seed=seed_log,
        ray_mm=None if ray_radii is None else ray_radii * 50.0,
        ray_radii=ray_radii,
        hand_overlap_dataset=hand,
        hand_overlap_wilor=wilor,
        other_part_overlap=other,
        iou_vs_truth=truth_iou,
    )


def test_hand_hull_and_subtraction() -> None:
    shape = (60, 80)
    hands = {"left_hand": np.array([[10.0, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]])}
    hull = hand_hull_mask(hands, shape, dilation_px=2)
    assert hull[20, 20] and hull[8, 20] and not hull[50, 70]
    assert not hand_hull_mask({"h": np.array([[1.0, 1.0], [2.0, 2.0]])}, shape, 2).any()
    mask = np.zeros(shape, dtype=bool)
    mask[15:25, 15:60] = True  # a bar crossing the hand hull on its left end
    mask[40:45, 70:75] = True  # a small island far away
    cut = subtract_hand(mask, hull)
    assert cut is not None
    assert not np.logical_and(cut, hull).any()
    assert cut[20, 50] and not cut[42, 72], "largest component kept, island dropped"
    assert subtract_hand(mask, np.zeros(shape, dtype=bool)) is None
    inside = np.zeros(shape, dtype=bool)
    inside[12:28, 12:28] = True
    assert subtract_hand(inside, hull) is None, "a mask wholly inside the hull leaves nothing"
    assert overlap_fraction(mask, hull) > 0.1
    assert overlap_fraction(np.zeros(shape, dtype=bool), hull) == 0.0


def test_candidate_passes_each_filter() -> None:
    good = _candidate("g")
    assert candidate_passes(current_rule(), good) == (True, [])
    assert not candidate_passes(current_rule(), _candidate("a", ratio=3.5))[0]
    assert not candidate_passes(current_rule(), _candidate("r", ray_radii=2.0))[0]
    assert not candidate_passes(current_rule(), _candidate("e", area=0))[0]
    assert not candidate_passes(current_rule(), _candidate("n", ray_radii=None))[0]
    tight = AcceptanceRuleSpec(name="b", band_low=0.7, band_high=1.3)
    assert not candidate_passes(tight, _candidate("w", ratio=1.6))[0]
    seed = AcceptanceRuleSpec(name="c", band_kind="seed_log_ratio", max_abs_log_ratio_vs_seed=0.4)
    assert candidate_passes(seed, _candidate("s", ratio=9.0, seed_log=0.3))[0]
    assert not candidate_passes(seed, _candidate("s", seed_log=-0.5))[0]
    assert not candidate_passes(seed, _candidate("s", seed_log=None))[0]
    score = AcceptanceRuleSpec(name="d", min_decoder_score=0.7)
    assert not candidate_passes(score, _candidate("l", score=0.6))[0]
    hand = AcceptanceRuleSpec(name="e", hand_rule="dataset")
    assert not candidate_passes(hand, _candidate("h", hand=0.5))[0]
    assert candidate_passes(hand, _candidate("h", hand=0.1, wilor=0.9))[0]
    wilor = AcceptanceRuleSpec(name="e2", hand_rule="wilor")
    assert not candidate_passes(wilor, _candidate("w", wilor=0.3))[0]
    other = AcceptanceRuleSpec(name="f", reject_other_part_overlap=True)
    assert not candidate_passes(other, _candidate("o", other=0.4))[0]
    assert "pool=base" in other.describe() and "other-part" in other.describe()


def test_apply_rule_ranks_by_ray_then_score_and_respects_pools() -> None:
    near_low_score = _candidate("near", ray_radii=0.2, score=0.5, truth_iou=0.9)
    far_high_score = _candidate("far", ray_radii=0.9, score=0.99, truth_iou=0.3)
    assert apply_rule(current_rule(), [far_high_score, near_low_score]) is near_low_score
    tie_a = _candidate("a", ray_radii=0.2, score=0.6)
    tie_b = _candidate("b", ray_radii=0.2, score=0.7)
    assert apply_rule(current_rule(), [tie_a, tie_b]) is tie_b
    by_score = AcceptanceRuleSpec(name="g", ranking="score_then_ray")
    assert apply_rule(by_score, [far_high_score, near_low_score]) is far_high_score
    assert "decoder score then" in by_score.describe()
    assert apply_rule(current_rule(), [_candidate("x", ratio=5.0)]) is None
    variant = _candidate(
        "v", ray_radii=0.1, variant="consensus_sphere_box_margin_0.25|hand_negatives_dataset"
    )
    cut = _candidate(
        "c", ray_radii=0.05, source="hand_subtracted_dataset", parent="t000300-b01:0:base"
    )
    cut_of_variant = _candidate(
        "cv",
        ray_radii=0.01,
        source="hand_subtracted_dataset",
        parent="t000300-b01:0:hand_negatives_dataset",
    )
    pool = [near_low_score, variant, cut, cut_of_variant]
    assert apply_rule(current_rule("base"), pool) is near_low_score
    assert apply_rule(current_rule("base+hand_negatives_dataset"), pool) is variant
    assert apply_rule(current_rule("base+hand_subtracted_dataset"), pool) is cut
    assert apply_rule(current_rule("base+hand_negatives_wilor"), pool) is near_low_score
    assert apply_rule(current_rule("all"), pool) is cut_of_variant


def test_metrics_objective_and_bar() -> None:
    outcomes = [
        Outcome(300, "chassis", True, 0.8, "a"),
        Outcome(370, "chassis", True, 0.2, "b"),
        Outcome(400, "chassis", False, None, None),
        Outcome(600, "chassis", True, 0.6, "c"),
    ]
    m = summarize(outcomes)
    assert m.cells == 4 and m.accepted == 3 and m.harm == 1
    assert abs((m.acceptance_rate or 0) - 0.75) < 1e-9
    assert abs((m.mean_iou_accepted or 0) - (0.8 + 0.2 + 0.6) / 3) < 1e-9
    assert abs((m.harm_rate or 0) - 1 / 3) < 1e-9
    assert abs(m.utility - ((0.4) + (-0.2) + 0.0 + 0.2) / 4) < 1e-9
    assert not meets_bar(m)
    clean = summarize(
        [Outcome(300, "chassis", True, 0.7, "a"), Outcome(370, "chassis", False, None, None)]
    )
    assert meets_bar(clean)
    assert objective(clean)[0] == 1.0
    sparse = summarize(
        [Outcome(300, "chassis", True, 0.95, "a")]
        + [Outcome(f, "chassis", False, None, None) for f in range(9)]
    )
    assert (sparse.acceptance_rate or 0) < MIN_FIT_ACCEPTANCE
    assert objective(sparse)[0] == 0.0, "a rule that barely accepts anything does not qualify"
    assert objective(clean) > objective(sparse)
    empty = summarize([])
    assert empty.mean_iou_accepted is None and empty.acceptance_rate is None


def test_leave_frames_out_picks_the_fit_winner_and_scores_held_frames() -> None:
    anchors = [300, 370, 400, 600, 650]
    always_fit = [0]
    # Rule "good" accepts a 0.8 mask on every frame; rule "greedy" accepts everything with 0.3.
    cells = [
        Cell(
            f,
            "chassis",
            "positive",
            5000,
            [
                _candidate("g", ray_radii=0.2, truth_iou=0.8),
                _candidate("h", ray_radii=0.9, truth_iou=0.3, hand=0.5),
            ],
        )
        for f in [0, *anchors]
    ]
    grid = [
        AcceptanceRuleSpec(name="good", hand_rule="dataset"),
        AcceptanceRuleSpec(name="greedy", band_low=0.0, band_high=10.0, max_ray_radii=5.0),
    ]
    outcomes = {r.name: [outcome_for(r, c) for c in cells] for r in grid}
    assert all(o.accepted and o.iou == 0.8 for o in outcomes["good"])
    result = leave_frames_out(
        grid, cells, anchor_frames=anchors, always_fit=always_fit, outcomes=outcomes
    )
    assert set(result.winners_per_split.values()) == {"good"}
    assert result.held_out.cells == len(anchors) * 3
    assert abs((result.held_out.mean_iou_accepted or 0) - 0.8) < 1e-9
    assert result.held_out.harm == 0
    assert abs(result.held_out_per_frame["300"]["good"] - 0.8) < 1e-9


def test_grid_and_named_rules_are_the_documented_sizes() -> None:
    grid = rule_grid()
    assert len(grid) == len(POOLS) * 6 * 4 * 3 * 2 * 2
    assert len({r.name for r in grid}) == len(grid)
    named = named_rules()
    assert [r.name for r in named][:1] == ["a:current"]
    assert sum(r.name.startswith("g:") for r in named) == 1
    assert sum(r.name.startswith("b:") for r in named) == 3
    assert sum(r.name.startswith("c:") for r in named) == 2
    assert sum(r.name.startswith("d:") for r in named) == 3
    assert sum(r.name.startswith("e:") for r in named) == 2
    assert sum(r.name.startswith("f:") for r in named) == 1
    assert sum(r.name.startswith("a:current@") for r in named) == len(POOLS) - 1


def test_variant_prompts_keep_only_boxes_that_gained_a_negative() -> None:
    box = PixelBox(x1=10, y1=10, x2=50, y2=50)
    base = [
        MultiviewRepromptPrompt(
            prompt_id="t000300-b01",
            target="chassis",
            variant="consensus_sphere_box_margin_0.25",
            pixel_box=box,
            background_points=(PixelPoint(x=20, y=20),),
            background_point_sources=("cabin_consensus_centroid",),
        )
    ]
    same = [base[0].model_copy(update={"prompt_id": "t000300-b02"})]
    assert _variant_prompts(base, same, "hand_negatives_dataset") == []
    more = [
        base[0].model_copy(
            update={
                "prompt_id": "t000300-b03",
                "background_points": (PixelPoint(x=20, y=20), PixelPoint(x=30, y=30)),
                "background_point_sources": ("cabin_consensus_centroid", "left_hand_joint_04"),
            }
        )
    ]
    kept = _variant_prompts(base, more, "hand_negatives_dataset")
    assert len(kept) == 1 and kept[0].variant.endswith("|hand_negatives_dataset")
