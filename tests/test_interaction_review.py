from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from battle.fixtures import synthetic_run_manifest
from battle.interaction_review import (
    CONTACT_END_FRAMES,
    CONTACT_START_FRAMES,
    DEFAULT_SOURCES,
    FIRST_20S_STATIC_TEXT_PANELS,
    _blueprint,
    _contacts,
    _drop_dtw_bookmarks,
    _log_diagnostics_frame,
    _log_navigation_frame,
    build_interaction_review,
    cluster_segmentation_triggers,
    coarse_gt_for_frame,
    contact_eligible_frame,
    contact_measurement,
    debounce_contact,
    deterministic_pinned_moments,
    nearest_wrist_matches,
    output_paths,
    validate_review_observations,
)
from battle.interaction_review_v4 import (
    COARSE_GT,
    FINE_LABELS,
    FIRST_MINUTE_REFERENCE_SEGMENTATION,
    SEGMENTATION_CONTACT_ELIGIBLE_INTERVALS,
    SEGMENTATION_CONTACT_ELIGIBLE_THROUGH,
    STATIC_TEXT_PANELS,
    _log_static_documents,
    _prepare_output_root,
)
from battle.interaction_review_v4 import (
    FRAME_COUNT as FIRST_MINUTE_FRAME_COUNT,
)
from battle.interaction_review_v4 import (
    OUTPUT_NAME as V4_OUTPUT_NAME,
)
from battle.interaction_review_v4 import (
    OUTPUT_ROOT as V4_OUTPUT_ROOT,
)
from battle.schemas import (
    FrameObservations,
    HumanFeedbackReviewRecord,
    InteractionContactDiagnostic,
    InteractionContactEvent,
    InteractionHandDisagreement,
    OvernightReviewRecord,
    SegmentationReviewTrigger,
)


def _hand(x: float, y: float, hand_id: str = "hand") -> SimpleNamespace:
    return SimpleNamespace(
        hand_id=hand_id,
        side="left",
        landmarks=tuple(SimpleNamespace(x=x, y=y) for _ in range(21)),
    )


def test_contact_distance_reports_exact_inside_and_source_pixel_distance() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[4:6, 4:6] = True

    palm, fingertip, minimum, inside = contact_measurement(_hand(0.5, 0.5), mask, (10, 10))
    assert (palm, fingertip, minimum, inside) == (0.0, 0.0, 0.0, True)

    _, _, minimum, inside = contact_measurement(_hand(0.0, 0.0), mask, (10, 10))
    assert minimum > 0
    assert inside is False


def test_contact_hysteresis_and_missing_values_do_not_persist() -> None:
    values = [False, True, True, False, False, False, None, True]
    result = debounce_contact(values)

    assert result[2] is True  # two observed candidate frames start contact
    assert result[5] is False  # three observed false frames end contact
    assert result[6] is None
    assert result[7] is False
    assert CONTACT_START_FRAMES == 2
    assert CONTACT_END_FRAMES == 3


def test_nearest_wrist_assignment_is_same_frame_spatial_only() -> None:
    mediapipe = (_hand(0.1, 0.1, "mp-near"), _hand(0.9, 0.9, "mp-far"))
    wilor = (_hand(0.11, 0.1, "wi-near"),)

    assert nearest_wrist_matches(mediapipe, wilor, (100, 100)) == [(0, 0), (1, None)]


def test_deterministic_pins_include_required_and_derived_bookmarks() -> None:
    disagreements = (
        InteractionHandDisagreement(
            analysis_frame_index=10,
            assignment_state="matched",
            mediapipe_hand_id="mp",
            wilor_hand_id="wi",
            mean_landmark_distance_pixels=42,
            max_landmark_distance_pixels=80,
            handedness_disagrees=False,
        ),
        InteractionHandDisagreement(
            analysis_frame_index=20, assignment_state="mediapipe_only", mediapipe_hand_id="mp"
        ),
        InteractionHandDisagreement(
            analysis_frame_index=500, assignment_state="wilor_only", wilor_hand_id="wi"
        ),
    )
    contacts = (
        InteractionContactDiagnostic(
            analysis_frame_index=11,
            hand_source_id="spatial-lane-1",
            part_id="cabin",
            observation_state="observed",
            palm_distance_pixels=0,
            fingertip_distance_pixels=0,
            minimum_distance_pixels=0,
            inside_mask=True,
            raw_contact_candidate=True,
            debounced_contact_candidate=True,
        ),
    )
    events = (
        InteractionContactEvent(
            analysis_frame_index=11,
            hand_source_id="spatial-lane-1",
            part_id="cabin",
            event_type="contact_candidate_start",
        ),
    )
    kineo = SimpleNamespace(
        observations={
            frame: FrameObservations(
                view_id="static-c10379", analysis_frame_index=frame, source_seconds=294 + frame / 30
            )
            for frame in range(600)
        }
    )
    pins = deterministic_pinned_moments(disagreements, contacts, events, kineo)
    categories = {pin.analysis_frame_index: pin.categories for pin in pins}

    assert {0, 300, 599, 10, 20, 11, 500}.issubset(categories)
    assert "high_hand_disagreement" in categories[10]
    assert "kineo_gap" in categories[0]
    assert all(pin.disposition == "pending" for pin in pins)


def test_mismatched_source_timestamp_is_rejected() -> None:
    manifest = synthetic_run_manifest()
    observation = manifest.observations[0].model_copy(update={"source_seconds": 99.0})

    with pytest.raises(ValueError, match="source timestamp mismatch"):
        validate_review_observations(manifest, {0: observation})


def test_missing_diagnostics_clear_rrd_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    logged: list[tuple[str, object]] = []
    monkeypatch.setattr(
        "battle.interaction_review.rr.log",
        lambda path, value, **_: logged.append((path, value)),
    )
    contact = InteractionContactDiagnostic(
        analysis_frame_index=5,
        hand_source_id="spatial-lane-1",
        part_id="cabin",
        observation_state="missing_hand",
    )
    _log_diagnostics_frame("world/fixture", 5, (contact,), ())

    cleared = {path for path, value in logged if value.__class__.__name__ == "Clear"}
    assert (
        "world/fixture/diagnostics/contact/spatial-lane-1/cabin/minimum_distance_pixels" in cleared
    )
    assert "world/fixture/diagnostics/hand_disagreement/mean_pixels" in cleared


def test_invalid_mask_diagnostics_carry_no_stale_contact_candidate() -> None:
    invalid = InteractionContactDiagnostic(
        analysis_frame_index=1584,
        hand_source_id="spatial-lane-1",
        part_id="cabin",
        observation_state="invalid_mask",
    )

    assert invalid.raw_contact_candidate is None
    with pytest.raises(ValueError, match="cannot retain stale"):
        InteractionContactDiagnostic.model_validate(
            invalid.model_copy(update={"minimum_distance_pixels": 1}).model_dump()
        )


def test_late_invalid_masks_clear_every_previously_seen_lane() -> None:
    rows = {
        frame: SimpleNamespace(hands=(_hand(0.5, 0.5),) if frame == 0 else (), objects=())
        for frame in range(1800)
    }
    source = SimpleNamespace(observations=rows, run_directory=Path("."))

    diagnostics, _ = _contacts(
        source,
        source,
        (1280, 720),
        frame_count=1800,
        segmentation_contact_eligible_through=1200,
    )

    assert [
        item.observation_state for item in diagnostics if item.analysis_frame_index == 1200
    ] == ["invalid_mask"] * 4


def test_v4_declares_first_minute_alignment_and_late_contact_cutoff() -> None:
    assert FIRST_MINUTE_FRAME_COUNT == 1800
    assert SEGMENTATION_CONTACT_ELIGIBLE_THROUGH == 1200
    assert SEGMENTATION_CONTACT_ELIGIBLE_INTERVALS == ((0, 1020), (1172, 1200))
    assert FIRST_MINUTE_REFERENCE_SEGMENTATION.name.endswith("20260918t001210z")


def test_contact_eligibility_intervals_yield_invalid_mask_inside_the_swap_gap() -> None:
    rows = {
        frame: SimpleNamespace(hands=(_hand(0.5, 0.5),) if frame == 0 else (), objects=())
        for frame in range(1800)
    }
    source = SimpleNamespace(observations=rows, run_directory=Path("."))

    diagnostics, _ = _contacts(
        source,
        source,
        (1280, 720),
        frame_count=1800,
        segmentation_contact_eligible_intervals=((0, 1020), (1172, 1200)),
    )
    states = {}
    for item in diagnostics:
        states.setdefault(item.analysis_frame_index, set()).add(item.observation_state)

    assert states[1019] == {"missing_hand"}
    assert states[1020] == {"invalid_mask"}
    assert states[1171] == {"invalid_mask"}
    assert states[1172] == {"missing_hand"}
    assert states[1200] == {"invalid_mask"}
    assert contact_eligible_frame(1172, ((0, 1020), (1172, 1200))) is True
    assert contact_eligible_frame(1100, ((0, 1020), (1172, 1200))) is False
    with pytest.raises(ValueError, match="not both"):
        _contacts(
            source,
            source,
            (1280, 720),
            frame_count=1800,
            segmentation_contact_eligible_through=1200,
            segmentation_contact_eligible_intervals=((0, 1020),),
        )


def _blueprint_views(blueprint) -> list:
    from rerun.blueprint.api import View

    def walk(node):
        if isinstance(node, View):
            yield node
            return
        for child in node.contents:
            yield from walk(child)

    return list(walk(blueprint.root_container))


def _referenced_entities(blueprint) -> set[str]:
    """Entity paths a blueprint's text and time-series views expect to find."""
    referenced: set[str] = set()
    for view in _blueprint_views(blueprint):
        kind = type(view).__name__
        if kind == "TextDocumentView":
            referenced.add(str(view.origin))
        elif kind == "TimeSeriesView" and isinstance(view.contents, tuple):
            for item in view.contents:
                referenced.add(str(item).replace("$origin", str(view.origin)))
    return referenced


def _logged_paths(monkeypatch: pytest.MonkeyPatch, log_calls) -> set[str]:
    logged: set[str] = set()

    def record(path, value, **_):
        if value.__class__.__name__ != "Clear":
            logged.add(path)

    monkeypatch.setattr("battle.interaction_review.rr.log", record)
    monkeypatch.setattr("battle.interaction_review_v4.rr.log", record)
    log_calls()
    return logged


def test_v4_blueprint_text_and_navigation_panels_have_logged_entities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The user saw empty guide/GT/substep panels: every referenced entity must be logged."""
    from battle.fine_substep_contract import load_contract, substep_for_frame

    root = "world/clip/interaction_review_v4"
    contract = load_contract(FINE_LABELS)

    def log_everything() -> None:
        _log_static_documents(root, guide="# guide", fine_contract=contract)
        for frame in (0, 599, 600, 1799):
            _log_navigation_frame(
                root,
                frame,
                source_seconds=294 + frame / 30,
                substep=substep_for_frame(contract, frame) if frame < 600 else None,
                coarse_gt=coarse_gt_for_frame(COARSE_GT, frame),
            )

    logged = _logged_paths(monkeypatch, log_everything)
    referenced = _referenced_entities(
        _blueprint(root, (1280, 720), static_text_panels=STATIC_TEXT_PANELS)
    )

    assert f"{root}/metadata/review_notes" in referenced
    assert f"{root}/metadata/navigation/current" in referenced
    assert f"{root}/metadata/navigation/agent_substep_index" in referenced
    assert f"{root}/metadata/navigation/coarse_gt_index" in referenced
    assert f"{root}/metadata/coarse_gt" in referenced
    assert referenced <= logged, referenced - logged


def test_v3_blueprint_references_only_paths_the_20s_builder_logs() -> None:
    root = "world/clip/interaction_review"
    referenced = _referenced_entities(
        _blueprint(root, (1280, 720), static_text_panels=FIRST_20S_STATIC_TEXT_PANELS)
    )
    builder_source = Path("src/battle/interaction_review.py").read_text()

    assert f"{root}/metadata/review_notes" in referenced
    assert f"{root}/metadata/drop_dtw" in referenced
    assert f"{root}/metadata/agent_substeps" in referenced
    for suffix in ("REVIEW_NOTES", "metadata/drop_dtw", "metadata/agent_substeps"):
        assert suffix in builder_source


def test_navigation_frame_clears_substep_index_outside_agent_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logged: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "battle.interaction_review.rr.log",
        lambda path, value, **_: logged.append((path, value.__class__.__name__)),
    )
    _log_navigation_frame(
        "world/x",
        1500,
        source_seconds=344.0,
        substep=None,
        coarse_gt=coarse_gt_for_frame(COARSE_GT, 1500),
    )

    assert ("world/x/metadata/navigation/agent_substep_index", "Clear") in logged
    assert ("world/x/metadata/navigation/coarse_gt_index", "Scalars") in logged
    assert ("world/x/metadata/navigation/current", "TextDocument") in logged
    assert coarse_gt_for_frame(COARSE_GT, 1500) == (3, ("screw chassis", 1281, 1800))
    assert coarse_gt_for_frame(COARSE_GT, 1800) is None


def test_v4_output_root_accepts_empty_directory_and_explains_non_empty(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    _prepare_output_root(empty, overwrite=False)
    assert empty.is_dir()

    (empty / "review_guide.md").write_text("old")
    with pytest.raises(FileExistsError, match="--overwrite"):
        _prepare_output_root(empty, overwrite=False)
    _prepare_output_root(empty, overwrite=True)
    assert not (empty / "review_guide.md").exists()

    (empty / "unrelated.txt").write_text("keep")
    with pytest.raises(FileExistsError, match="unexpected entry"):
        _prepare_output_root(empty, overwrite=True)
    assert (empty / "unrelated.txt").exists()


_CHUNK_HEADER = re.compile(r"^Chunk\(\S+\) with (\d+) rows? \([^)]*\) - (/\S+) - ", re.MULTILINE)


def _rrd_row_counts(rrd_path: Path) -> tuple[dict[str, int], str]:
    printed = subprocess.run(
        [str(Path(sys.executable).with_name("rerun")), "rrd", "print", str(rrd_path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    counts: dict[str, int] = {}
    for rows, entity in _CHUNK_HEADER.findall(printed):
        counts[entity] = counts.get(entity, 0) + int(rows)
    return counts, printed


def test_exported_v4_rrd_carries_navigation_entities_and_blueprint_views() -> None:
    """Integration check against the ignored, rebuilt package (skipped when absent)."""
    rrd_path = Path(V4_OUTPUT_ROOT) / V4_OUTPUT_NAME
    if not rrd_path.is_file():
        pytest.skip(f"{rrd_path} is not built in this checkout")
    counts, printed = _rrd_row_counts(rrd_path)
    roots = {path.split("/metadata/")[0] for path in counts if "/metadata/navigation/" in path}
    assert len(roots) == 1, roots
    root = roots.pop()

    assert counts[f"{root}/metadata/navigation/current"] == FIRST_MINUTE_FRAME_COUNT
    assert counts[f"{root}/metadata/navigation/coarse_gt_index"] >= FIRST_MINUTE_FRAME_COUNT
    assert counts[f"{root}/metadata/navigation/agent_substep_index"] >= 600
    for relative, _ in STATIC_TEXT_PANELS:
        assert counts[f"{root}/metadata/{relative.split('/', 1)[1]}"] >= 1
    assert counts[f"{root}/metadata/review_notes"] >= 1
    expected_views = _blueprint_views(
        _blueprint(root.lstrip("/"), (1280, 720), static_text_panels=STATIC_TEXT_PANELS)
    )
    assert printed.count("ViewBlueprint:display_name") == len(expected_views)


def test_drop_dtw_bookmarks_use_only_declared_matched_frames(tmp_path) -> None:
    alignment = tmp_path / "alignment.json"
    alignment.write_text(
        '{"intervals":[{"action":"attach interior","matched_analysis_frames":[0,30]}]}'
    )

    assert _drop_dtw_bookmarks(alignment) == {0: "attach interior", 30: "attach interior"}


def test_segmentation_triggers_cluster_without_discarding_raw_rows() -> None:
    triggers = (
        SegmentationReviewTrigger(
            analysis_frame_index=10, target_id="cabin", trigger_type="area_ratio_gt_2", value=3
        ),
        SegmentationReviewTrigger(
            analysis_frame_index=14,
            target_id="cabin",
            trigger_type="centroid_jump_gt_25px",
            value=30,
        ),
        SegmentationReviewTrigger(
            analysis_frame_index=30, target_id="cabin", trigger_type="area_ratio_gt_2", value=3
        ),
    )

    episodes = cluster_segmentation_triggers(triggers, max_frame_gap=6)

    assert [(item.start_frame, item.end_frame, item.trigger_count) for item in episodes] == [
        (10, 14, 2),
        (30, 30, 1),
    ]
    assert len(triggers) == 3


def test_output_paths_stay_in_ignored_run_directory(tmp_path) -> None:
    paths = output_paths(tmp_path)

    assert paths[0] == tmp_path / "runs/interaction-review-first-20s/interaction_review.rrd"
    assert all(path.parent == paths[0].parent for path in paths)


def test_v3_defaults_use_corrected_sam3_and_fused_kineo_context() -> None:
    import inspect

    assert DEFAULT_SOURCES["kineo"].run_directory.name == "kineo-nlf-fused-20s-overnight-v3"
    assert (
        inspect.signature(build_interaction_review)
        .parameters["reference_segmentation_method"]
        .default
        == "baseline_sam3"
    )


def test_human_feedback_record_stays_human_authored_and_verbatim() -> None:
    record = HumanFeedbackReviewRecord.model_validate(
        json.loads(
            Path("docs/qa/interaction-review-first-minute-v4.human-feedback.json").read_text()
        )
    )

    assert (record.author_type, record.provenance_tag) == ("human", "human_feedback_report")
    assert record.human_decisions_pending is True
    assert any("1100 - 1200" in item.verbatim for item in record.feedback)
    assert all(
        action.disposition == "agent_authored_visual_review" for action in record.agent_actions
    )
    with pytest.raises(ValueError, match="unknown agent actions"):
        HumanFeedbackReviewRecord.model_validate(
            {
                **record.model_dump(mode="json"),
                "feedback": [
                    {"subject": "x", "verbatim": "y", "agent_action_subjects": ["not an action"]}
                ],
            }
        )


def test_overnight_agent_review_keeps_human_feedback_distinct() -> None:
    for filename in (
        "overnight-interaction-review-v2.agent-review.json",
        "overnight-interaction-review-v3.agent-review.json",
        "interaction-review-first-minute-v4.agent-review.json",
        "interaction-review-first-minute-v4r2.agent-review.json",
        "interaction-review-first-minute-v4r3.agent-review.json",
    ):
        record = OvernightReviewRecord.model_validate(
            json.loads((Path("docs/qa") / filename).read_text())
        )

        assert record.provenance_tag == "agent_authored_visual_review"
        assert record.human_decisions_pending is True
        assert all(
            "WiLoR" not in item.finding or item.disposition for item in record.agent_findings
        )
