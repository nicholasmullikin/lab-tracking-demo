from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from battle.ensemble_reference import (
    POLICY_PATH,
    EnsembleDecider,
    cluster_intervals,
    fallback_intervals_from_contradictions,
    load_policy,
    summarize_target,
    validity_intervals_from_sidecar,
)
from battle.ensemble_reference_schemas import (
    PROVENANCE_CODES,
    EnsembleFallbackRules,
    EnsembleFrameProvenance,
    EnsembleLabeledInterval,
    EnsembleProvenanceSidecar,
    EnsembleReferencePolicy,
    EnsembleSanityBounds,
    EnsembleSourceRunPolicy,
    EnsembleTargetPolicy,
)
from battle.schemas import ArtifactFingerprint, FrameObservations, NormalizedBox, PerFrameObject

HEIGHT, WIDTH = 72, 128


def _mask(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _interval(start: int, end: int, rationale: str = "fixture") -> EnsembleLabeledInterval:
    return EnsembleLabeledInterval(start_frame=start, end_frame_exclusive=end, rationale=rationale)


def _policy(
    *,
    fallback: tuple[EnsembleLabeledInterval, ...] = (_interval(40, 80),),
    hidden: tuple[EnsembleLabeledInterval, ...] = (),
    ineligible: tuple[EnsembleLabeledInterval, ...] = (),
    rules: EnsembleFallbackRules | None = None,
    sanity: EnsembleSanityBounds | None = None,
) -> EnsembleReferencePolicy:
    def target(target_id: str, **overrides) -> EnsembleTargetPolicy:
        return EnsembleTargetPolicy(
            target_id=target_id,
            rules=rules or EnsembleFallbackRules(),
            sanity=sanity or EnsembleSanityBounds(),
            **overrides,
        )

    return EnsembleReferencePolicy(
        policy_id="fixture",
        provenance_tag="agent_authored_assumption",
        frame_count=1800,
        primary=EnsembleSourceRunPolicy(label="sam3_corrected", run_directory="runs/a"),
        fallback=EnsembleSourceRunPolicy(label="dam4sam_fallback", run_directory="runs/b"),
        rolling_median_window_frames=30,
        rolling_median_min_samples=5,
        contact_eligible_through_frame=100,
        late_ineligibility_rationale="fixture late boundary",
        targets=(
            target(
                "chassis",
                fallback_intervals=fallback,
                not_contact_eligible_intervals=ineligible,
            ),
            target("interior", hidden_intervals=hidden),
            target("rear_body"),
            target("cabin"),
        ),
        claim_boundaries=("fixture",),
    )


def _run(
    decider: EnsembleDecider,
    frames: range,
    primary: dict[int, np.ndarray | None],
    fallback: dict[int, np.ndarray | None],
    *,
    other: dict[str, np.ndarray | None] | None = None,
    target: str = "chassis",
):
    other = other or {"interior": None, "rear_body": None, "cabin": None}
    return [
        decider.decide(frame, target, primary.get(frame), fallback.get(frame), other)
        for frame in frames
    ]


def test_checked_in_policy_is_typed_and_agent_authored() -> None:
    policy = load_policy(POLICY_PATH)

    assert policy.provenance_tag == "agent_authored_assumption"
    assert policy.no_blend is True
    chassis, interior = policy.target("chassis"), policy.target("interior")
    assert [(i.start_frame, i.end_frame_exclusive) for i in chassis.fallback_intervals] == [
        (1055, 1172)
    ]
    assert [(i.start_frame, i.end_frame_exclusive) for i in interior.hidden_intervals] == [
        (1024, 1172)
    ]
    assert all(item.human_confirmation_pending for item in interior.hidden_intervals)
    assert interior.fallback_intervals == ()
    assert policy.contact_eligible_through_frame == 1200


def test_v2_policy_chooses_arms_by_anchors_and_intervals_label_free() -> None:
    policy = load_policy(Path("configs/ensemble_reference/first_minute_v2.json"))

    assert policy.primary.arm == "pm-append"
    assert policy.primary.run_directory.startswith("runs/sam3-memory-arms-20260919/arms/pm-append/")
    assert policy.fallback.arm == "dam4sam-large-1024-sched-60s"
    assert all(target.hidden_intervals == () for target in policy.targets), "hidden label withdrawn"
    intervals = {
        target.target_id: tuple(
            (i.start_frame, i.end_frame_exclusive) for i in target.fallback_intervals
        )
        for target in policy.targets
    }
    assert intervals == {
        "chassis": ((296, 313), (475, 515), (1049, 1085)),
        "interior": (),
        "rear_body": ((1762, 1794),),
        "cabin": (),
    }
    selection = policy.selection_provenance
    assert selection is not None
    assert selection.arm_selection.method == "human_review_anchor_iou"
    assert selection.interval_selection.method == "multiview_consensus_contradiction"
    raw = selection.interval_selection.raw_contradiction_intervals
    for target in policy.targets:
        derived = fallback_intervals_from_contradictions(
            raw[target.target_id],
            merge_gap_below_frames=selection.interval_selection.merge_gap_below_frames,
            min_interval_frames=selection.interval_selection.min_interval_frames,
        )
        assert derived == intervals[target.target_id], "the config restates the rule's output"
        assert all(
            i.provenance == "multiview_consensus_contradiction" for i in target.fallback_intervals
        )
    assert "selection-biased" in " ".join(policy.claim_boundaries)
    distractor = policy.target("rear_body").not_contact_eligible_intervals
    assert [(i.start_frame, i.end_frame_exclusive, i.failure_case) for i in distractor] == [
        (1660, 1800, "distractor_confusion")
    ]
    assert distractor[0].provenance == "primary_mask_trajectory"
    assert policy.target("rear_body").in_intervals(1700, distractor)


def test_v1_policy_still_loads_without_selection_provenance() -> None:
    policy = load_policy(POLICY_PATH)

    assert policy.selection_provenance is None
    assert all(
        interval.provenance == "agent_authored_visual_review" and interval.failure_case is None
        for target in policy.targets
        for interval in (
            *target.fallback_intervals,
            *target.hidden_intervals,
            *target.not_contact_eligible_intervals,
        )
    )


def test_fallback_intervals_from_contradictions_merges_then_drops() -> None:
    raw = ((296, 313), (475, 494), (508, 515), (1049, 1057), (1058, 1085), (1662, 1667))

    assert fallback_intervals_from_contradictions(raw) == (
        (296, 313),
        (475, 515),
        (1049, 1085),
    )
    # Merge happens before the length filter: two short neighbours survive together.
    assert fallback_intervals_from_contradictions(((10, 15), (20, 26))) == ((10, 26),)
    # A gap of exactly the threshold is not merged; unsorted input is fine.
    assert fallback_intervals_from_contradictions(
        ((40, 60), (0, 25)), merge_gap_below_frames=15
    ) == (
        (0, 25),
        (40, 60),
    )
    assert fallback_intervals_from_contradictions(()) == ()
    with pytest.raises(ValueError, match="inverted"):
        fallback_intervals_from_contradictions(((5, 5),))


def test_selection_provenance_requires_consensus_derived_fallback_intervals() -> None:
    v2 = load_policy(Path("configs/ensemble_reference/first_minute_v2.json"))
    reviewed = _interval(100, 120, "drawn by eye")
    chassis = v2.target("chassis").model_copy(update={"fallback_intervals": (reviewed,)})
    with pytest.raises(ValueError, match="cross-view consensus"):
        EnsembleReferencePolicy.model_validate(
            v2.model_copy(update={"targets": (chassis, *v2.targets[1:])}).model_dump()
        )


def test_policy_rejects_overlapping_hidden_and_fallback_intervals() -> None:
    with pytest.raises(ValueError, match="must not overlap"):
        EnsembleTargetPolicy(
            target_id="chassis",
            fallback_intervals=(_interval(10, 20),),
            hidden_intervals=(_interval(15, 30),),
        )
    with pytest.raises(ValueError, match="contract order"):
        _policy().model_copy(update={"targets": _policy().targets[::-1]}).model_validate(
            _policy().model_copy(update={"targets": _policy().targets[::-1]}).model_dump()
        )


def test_primary_stands_when_no_rule_fires_even_inside_fallback_interval() -> None:
    policy = _policy()
    decider = EnsembleDecider(policy)
    steady = _mask(10, 10, 50, 40)
    different = _mask(60, 10, 100, 40)
    decisions = _run(
        decider, range(0, 60), {f: steady for f in range(60)}, {f: different for f in range(60)}
    )

    assert {d.record.provenance for d in decisions} == {"sam3_corrected"}
    assert all(d.record.sane and d.record.contact_eligible for d in decisions)
    assert all(d.substitution is None for d in decisions)
    assert decisions[50].record.inside_fallback_interval is True
    assert decisions[50].record.rules_fired == ()


def test_area_rule_substitutes_a_sane_fallback_only_inside_the_interval() -> None:
    policy = _policy()
    decider = EnsembleDecider(policy)
    full = _mask(10, 10, 50, 40)
    collapsed = _mask(10, 10, 16, 16)
    # Healthy primary except a short collapse outside the interval (20-24) and a long one
    # inside it (45-79); the fallback stays full throughout.
    primary = {f: (collapsed if 20 <= f < 25 or 45 <= f < 80 else full) for f in range(90)}
    fallback = {f: full for f in range(90)}
    decisions = _run(decider, range(0, 90), primary, fallback)

    outside = [d for d in decisions if 20 <= d.record.analysis_frame_index < 25]
    assert {d.record.provenance for d in outside} == {"sam3_corrected"}
    assert all("area_below_rolling_median_fraction" in d.record.rules_fired for d in outside)
    assert all(d.record.sane for d in outside), "outside intervals firings are diagnostic only"
    untouched = [d for d in decisions if 40 <= d.record.analysis_frame_index < 45]
    assert {d.record.provenance for d in untouched} == {"sam3_corrected"}
    inside = [d for d in decisions if 45 <= d.record.analysis_frame_index < 80]
    assert {d.record.provenance for d in inside} == {"dam4sam_fallback"}
    assert all(
        d.substitution is not None and d.substitution.outcome == "substituted" for d in inside
    )
    assert all(d.record.contact_eligible for d in inside)
    after = [d for d in decisions if d.record.analysis_frame_index >= 80]
    assert {d.record.provenance for d in after} == {"sam3_corrected"}


def test_continuity_allowance_grows_per_frame_but_is_capped() -> None:
    bounds = EnsembleSanityBounds(
        max_centroid_jump_pixels=30.0,
        centroid_jump_pixels_per_frame=5.0,
        max_continuity_allowance_pixels=90.0,
    )

    assert bounds.continuity_allowance(0) == 30.0
    assert bounds.continuity_allowance(4) == 50.0
    assert bounds.continuity_allowance(1000) == 90.0
    with pytest.raises(ValueError, match="continuity cap"):
        EnsembleSanityBounds(max_centroid_jump_pixels=50.0, max_continuity_allowance_pixels=40.0)


def test_fallback_failing_sanity_is_declined_and_primary_flagged_unsane() -> None:
    # No per-frame growth so the far fallback cannot become continuous by waiting.
    policy = _policy(sanity=EnsembleSanityBounds(centroid_jump_pixels_per_frame=0.0))
    decider = EnsembleDecider(policy)
    full = _mask(10, 10, 50, 40)
    collapsed = _mask(10, 10, 16, 16)
    tiny_fallback = _mask(10, 10, 15, 15)
    far_fallback = _mask(80, 40, 120, 70)
    primary = {f: (full if f < 40 else collapsed) for f in range(60)}
    fallback = {f: (tiny_fallback if f < 50 else far_fallback) for f in range(60)}
    decisions = _run(decider, range(0, 60), primary, fallback)

    declined_small = [d for d in decisions if 40 <= d.record.analysis_frame_index < 50]
    assert {d.record.provenance for d in declined_small} == {"sam3_corrected"}
    assert all(not d.record.sane and not d.record.contact_eligible for d in declined_small)
    assert all(d.substitution.outcome == "declined" for d in declined_small)
    assert "outside" in declined_small[0].substitution.rationale
    declined_far = [d for d in decisions if d.record.analysis_frame_index >= 50]
    assert all(d.substitution.outcome == "declined" for d in declined_far)
    assert "discontinuous" in declined_far[0].substitution.rationale


def test_discontinuity_rule_catches_a_primary_that_jumps_to_another_object() -> None:
    policy = _policy()
    decider = EnsembleDecider(policy)
    body = _mask(10, 10, 50, 40)
    jumped = _mask(80, 40, 120, 70)  # same area, different object
    primary = {f: (body if f < 45 else jumped) for f in range(60)}
    fallback = {f: body for f in range(60)}
    decisions = _run(decider, range(0, 60), primary, fallback)

    inside = [d for d in decisions if d.record.analysis_frame_index >= 45]
    assert {d.record.provenance for d in inside} == {"dam4sam_fallback"}
    assert all(d.record.rules_fired == ("discontinuous_with_last_accepted",) for d in inside)


def test_other_target_overlap_rule_and_primary_missing_rule() -> None:
    policy = _policy()
    decider = EnsembleDecider(policy)
    body = _mask(10, 10, 50, 40)
    primary = {f: (None if f in (50, 51) else body) for f in range(60)}
    fallback = {f: body for f in range(60)}
    decisions = _run(
        decider,
        range(0, 60),
        primary,
        fallback,
        other={"interior": body, "rear_body": None, "cabin": None},
    )

    fired = {d.record.analysis_frame_index: d.record.rules_fired for d in decisions}
    assert "other_target_iou_above_threshold" in fired[45]
    assert fired[50] == ("primary_missing",)
    assert decisions[50].record.provenance == "dam4sam_fallback"
    assert decisions[45].record.provenance == "dam4sam_fallback"


def test_hidden_interval_yields_explicit_empty_mask_that_is_never_eligible() -> None:
    policy = _policy(hidden=(_interval(30, 50, "block hidden"),))
    decider = EnsembleDecider(policy)
    block = _mask(10, 10, 20, 20)
    decisions = _run(
        decider,
        range(0, 60),
        {f: block for f in range(60)},
        {f: block for f in range(60)},
        target="interior",
    )

    hidden = [d for d in decisions if 30 <= d.record.analysis_frame_index < 50]
    assert {d.record.provenance for d in hidden} == {"hidden_agent_label"}
    assert all(d.mask is None and d.record.area_pixels == 0 for d in hidden)
    assert all(not d.record.sane and not d.record.contact_eligible for d in hidden)
    assert all(d.record.primary_area_pixels == 100 for d in hidden), "the primary mask is recorded"
    visible = [d for d in decisions if not 30 <= d.record.analysis_frame_index < 50]
    assert {d.record.provenance for d in visible} == {"sam3_corrected"}
    with pytest.raises(ValueError, match="empty, unsane and ineligible"):
        EnsembleFrameProvenance(
            analysis_frame_index=0,
            target_id="interior",
            provenance="hidden_agent_label",
            area_pixels=5,
            inside_fallback_interval=False,
            sane=False,
            contact_eligible=False,
        )


def test_masks_are_copied_whole_never_blended() -> None:
    policy = _policy()
    decider = EnsembleDecider(policy)
    full = _mask(10, 10, 50, 40)
    collapsed = _mask(10, 10, 16, 16)
    shifted = _mask(12, 12, 52, 42)
    primary = {f: (full if f < 40 else collapsed) for f in range(60)}
    fallback = {f: shifted for f in range(60)}
    decisions = _run(decider, range(0, 60), primary, fallback)

    for decision in decisions:
        frame = decision.record.analysis_frame_index
        assert decision.mask is not None
        source = (
            primary[frame] if decision.record.provenance == "sam3_corrected" else fallback[frame]
        )
        assert np.array_equal(decision.mask, source), "each mask is exactly one source mask"
        assert decision.mask.dtype == bool
    assert {d.record.provenance for d in decisions if d.record.analysis_frame_index >= 40} == {
        "dam4sam_fallback"
    }
    with pytest.raises(ValueError, match="only allowed inside fallback intervals"):
        EnsembleFrameProvenance(
            analysis_frame_index=0,
            target_id="chassis",
            provenance="dam4sam_fallback",
            area_pixels=5,
            inside_fallback_interval=False,
            sane=True,
            contact_eligible=True,
        )


def test_policy_ineligible_interval_and_late_boundary_remove_eligibility() -> None:
    policy = _policy(ineligible=(_interval(10, 20, "leak onset"),))
    decider = EnsembleDecider(policy)
    body = _mask(10, 10, 50, 40)
    decisions = _run(
        decider, range(0, 120), {f: body for f in range(120)}, {f: body for f in range(120)}
    )

    eligible = [d.record.analysis_frame_index for d in decisions if d.record.contact_eligible]
    assert cluster_intervals(eligible) == ((0, 10), (20, 100))
    assert all(d.record.sane for d in decisions), "ineligibility is a label, not an unsane mask"


def test_summary_and_validity_intervals_carry_policy_rationales() -> None:
    policy = _policy(
        hidden=(_interval(30, 50, "block hidden"),), ineligible=(_interval(10, 20, "leak onset"),)
    )
    decider = EnsembleDecider(policy)
    body = _mask(10, 10, 50, 40)
    collapsed = _mask(10, 10, 16, 16)
    records = []
    declined = []
    for frame in range(1800):
        for target in ("chassis", "interior", "rear_body", "cabin"):
            primary = collapsed if (target == "chassis" and 45 <= frame < 60) else body
            fallback = body if frame != 47 else None
            decision = decider.decide(
                frame,
                target,
                primary,
                fallback,
                {"interior": None, "rear_body": None, "cabin": None},
            )
            records.append(decision.record.model_copy(update={"mask_uri": "masks/x.png"}))
            if decision.substitution is not None and decision.substitution.outcome == "declined":
                declined.append(decision.substitution)
    summaries = tuple(
        summarize_target(t, records, declined)
        for t in ("chassis", "interior", "rear_body", "cabin")
    )
    fingerprint = ArtifactFingerprint(uri="x", sha256="0" * 64, source="measured")
    sidecar = EnsembleProvenanceSidecar(
        manifest_kind="ensemble_reference_provenance_v1",
        policy=fingerprint,
        primary_manifest=fingerprint,
        fallback_manifest=fingerprint,
        frame_count=1800,
        claim_boundaries=("fixture",),
        summaries=summaries,
        declined=tuple(declined),
        frames=tuple(records),
    )
    chassis = sidecar.summaries[0]
    assert chassis.substituted_intervals == ((45, 47), (48, 60))
    assert chassis.declined_intervals == ((47, 48),)
    assert chassis.contact_eligible_intervals == ((0, 10), (20, 47), (48, 100))
    assert sidecar.summaries[1].hidden_intervals == ((30, 50),)
    assert sum(chassis.provenance_counts.values()) == 1800

    intervals = validity_intervals_from_sidecar(sidecar, policy)
    chassis_rows = [
        (i.start_frame, i.end_frame_exclusive, i.state)
        for i in intervals
        if i.target_id == "chassis"
    ]
    assert chassis_rows[:2] == [(0, 10, "contact_eligible"), (10, 20, "not_contact_eligible")]
    assert (47, 48, "not_contact_eligible") in chassis_rows
    assert chassis_rows[-1] == (100, 1800, "not_contact_eligible")
    by_key = {(i.target_id, i.start_frame): i for i in intervals}
    assert by_key[("chassis", 10)].rationale == "leak onset"
    assert by_key[("interior", 30)].rationale == "block hidden"
    assert by_key[("chassis", 100)].rationale == "fixture late boundary"
    assert "failed the sanity" in by_key[("chassis", 47)].rationale
    assert all(i.provenance == "ensemble_reference_policy" for i in intervals)


def test_provenance_layer_logs_series_overlay_and_eligibility_per_part(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from battle.interaction_review_v4 import (
        _log_reference_provenance_frame,
        _reference_provenance_static,
    )

    logged: list[tuple[str, str, object]] = []

    def record(path, value, **kwargs):
        logged.append((path, value.__class__.__name__, kwargs.get("static", False)))

    monkeypatch.setattr("battle.interaction_review_v4.rr.log", record)
    root = "world/clip/interaction_review_v4"
    observation = FrameObservations(
        view_id="static-c10379",
        analysis_frame_index=1100,
        source_seconds=294 + 1100 / 30,
        objects=(
            PerFrameObject(
                object_id="ensemble-chassis",
                label="chassis",
                confidence=1.0,
                box=NormalizedBox(x=0.6, y=0.5, width=0.1, height=0.1),
            ),
        ),
    )
    provenance = {
        (1100, "chassis"): "dam4sam_fallback",
        (1100, "interior"): "hidden_agent_label",
        (1100, "rear_body"): "sam3_corrected",
        (1100, "cabin"): "sam3_corrected",
    }
    eligibility = {"chassis": ((1062, 1200),), "rear_body": ((0, 1200),), "cabin": ((0, 1200),)}
    _reference_provenance_static(root)
    _log_reference_provenance_frame(root, 1100, observation, provenance, eligibility, (1280, 720))

    statics = {path for path, kind, static in logged if static and kind == "SeriesLines"}
    assert statics == {f"{root}/diagnostics/reference_provenance/{p}" for p in provenance_parts()}
    series = {
        path: kind
        for path, kind, static in logged
        if not static and "reference_provenance/" in path
    }
    assert series == {
        f"{root}/diagnostics/reference_provenance/{p}": "Scalars" for p in provenance_parts()
    }
    overlays = {
        path.rsplit("/", 1)[1]: kind
        for path, kind, _ in logged
        if "reference_provenance_overlay" in path
    }
    assert overlays == {
        "chassis": "Boxes2D",
        "interior": "Clear",
        "rear_body": "Clear",
        "cabin": "Clear",
    }
    eligible = {
        path.rsplit("/", 1)[1]: kind
        for path, kind, _ in logged
        if "segmentation_contact_eligible/" in path
    }
    assert set(eligible) == set(provenance_parts())
    assert PROVENANCE_CODES == {
        "missing": 0,
        "sam3_corrected": 1,
        "dam4sam_fallback": 2,
        "hidden_agent_label": 3,
    }


def provenance_parts() -> tuple[str, ...]:
    return ("chassis", "interior", "rear_body", "cabin")


@pytest.mark.real_data
def test_built_ensemble_run_matches_its_policy_and_sidecar() -> None:
    """Integration check against the ignored, built ensemble run (skipped when absent)."""
    from conftest import require_artifact

    from battle.ensemble_reference import OUTPUT_ROOT, PROVENANCE_NAME, load_sidecar
    from battle.schemas import RunManifest

    run = Path(OUTPUT_ROOT)
    require_artifact(run / PROVENANCE_NAME)
    sidecar = load_sidecar(run)
    manifest = RunManifest.model_validate_json((run / "manifest.json").read_text())
    assert manifest.ensemble_reference is not None
    assert manifest.ensemble_reference.provenance_counts == {
        s.target_id: s.provenance_counts for s in sidecar.summaries
    }
    by_target = {s.target_id: s for s in sidecar.summaries}
    assert by_target["interior"].hidden_intervals == ((1024, 1172),)
    assert by_target["interior"].provenance_counts["dam4sam_fallback"] == 0
    assert all(1055 <= s < e <= 1172 for s, e in by_target["chassis"].substituted_intervals)
    assert by_target["rear_body"].provenance_counts["sam3_corrected"] == 1800
    assert by_target["cabin"].provenance_counts["sam3_corrected"] == 1800
    check = json.loads((run / "segmentation_episode_check.json").read_text())
    assert check["after"]["episodes_in_check_window"] == []
    assert check["new_after_episodes"] == []
    # Hidden frames are explicit empty masks on disk and absent from the observation rows.
    hidden_row = next(
        r for r in sidecar.frames if r.target_id == "interior" and r.analysis_frame_index == 1100
    )
    from PIL import Image

    assert hidden_row.mask_uri is not None
    with Image.open(run / hidden_row.mask_uri) as image:
        assert not np.asarray(image).any()
    observation = manifest.observations[1100]
    assert "interior" not in {item.label for item in observation.objects}
    assert {item.label for item in observation.objects} == {"chassis", "rear_body", "cabin"}
