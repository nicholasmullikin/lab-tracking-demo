from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from battle.fixtures import synthetic_run_manifest
from battle.interaction_review import (
    CONTACT_END_FRAMES,
    CONTACT_START_FRAMES,
    _drop_dtw_bookmarks,
    _log_diagnostics_frame,
    contact_measurement,
    debounce_contact,
    deterministic_pinned_moments,
    nearest_wrist_matches,
    output_paths,
    validate_review_observations,
)
from battle.schemas import (
    FrameObservations,
    InteractionContactDiagnostic,
    InteractionContactEvent,
    InteractionHandDisagreement,
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


def test_drop_dtw_bookmarks_use_only_declared_matched_frames(tmp_path) -> None:
    alignment = tmp_path / "alignment.json"
    alignment.write_text(
        '{"intervals":[{"action":"attach interior","matched_analysis_frames":[0,30]}]}'
    )

    assert _drop_dtw_bookmarks(alignment) == {0: "attach interior", 30: "attach interior"}


def test_output_paths_stay_in_ignored_run_directory(tmp_path) -> None:
    paths = output_paths(tmp_path)

    assert paths[0] == tmp_path / "runs/interaction-review-first-20s/interaction_review.rrd"
    assert all(path.parent == paths[0].parent for path in paths)
