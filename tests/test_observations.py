"""`battle.observations`: the schema-validating reader, the tracker rebuild and the label lookup."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from battle import observations
from battle.schemas import FrameObservations, MaskReference, NormalizedBox, PerFrameObject

RECORD = {
    "view_id": "static-c10379",
    "analysis_frame_index": 3,
    "source_seconds": 294.1,
    "objects": [
        {
            "object_id": "dam4sam-chassis",
            "label": "chassis",
            "confidence": 1.0,
            "box": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4},
            "mask": {
                "uri": "native/masks/chassis/00003.png",
                "storage": "native_artifact",
                "format": "png",
            },
        },
        {
            "object_id": "dam4sam-cabin",
            "label": "cabin",
            "confidence": 0.5,
            "box": {"x": 0.5, "y": 0.5, "width": 0.1, "height": 0.1},
        },
    ],
}


def _write(path: Path, *rows: dict) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows) + "\n", encoding="utf-8")
    return path


def test_load_observations_validates_and_names_the_bad_line(tmp_path: Path) -> None:
    path = _write(tmp_path / "observations.jsonl", RECORD, {**RECORD, "analysis_frame_index": 4})
    loaded = observations.load_observations(path)
    assert [o.analysis_frame_index for o in loaded] == [3, 4]
    assert loaded[0].objects[0].mask is not None
    assert observations.observations_by_frame(path)[4].objects[1].label == "cabin"

    bad = _write(tmp_path / "bad.jsonl", RECORD, {**RECORD, "unknown_key": 1})
    with pytest.raises(ValueError, match=rf"invalid observation at {bad}:2"):
        observations.load_observations(bad)


def test_rebuild_tracker_observations_takes_known_fields_and_ignores_the_rest(
    tmp_path: Path,
) -> None:
    row = json.loads(json.dumps(RECORD))
    row["worker_note"] = "ignored"  # unknown frame-level key
    row["objects"][0]["tracker_state"] = {"ignored": True}  # unknown object-level key
    row["analysis_frame_index"] = "3"  # coerced like the drivers did
    path = _write(tmp_path / "observations.jsonl", row)

    (frame,) = observations.rebuild_tracker_observations(path)

    assert frame == FrameObservations(
        view_id="static-c10379",
        analysis_frame_index=3,
        source_seconds=294.1,
        objects=(
            PerFrameObject(
                object_id="dam4sam-chassis",
                label="chassis",
                confidence=1.0,
                box=NormalizedBox(x=0.1, y=0.2, width=0.3, height=0.4),
                mask=MaskReference(
                    uri="native/masks/chassis/00003.png", storage="native_artifact", format="png"
                ),
            ),
            PerFrameObject(
                object_id="dam4sam-cabin",
                label="cabin",
                confidence=0.5,
                box=NormalizedBox(x=0.5, y=0.5, width=0.1, height=0.1),
            ),
        ),
    )
    with pytest.raises(ValueError):
        observations.load_observations(path)  # the strict reader rejects the unknown keys


def test_rebuild_tracker_observations_builds_hands_only_through_the_builder(tmp_path: Path) -> None:
    row = {**RECORD, "objects": [], "hands": [{"hand_id": "h1"}]}
    path = _write(tmp_path / "observations.jsonl", row)

    (frame,) = observations.rebuild_tracker_observations(path)
    assert frame.hands == () and frame.objects == ()

    seen: list[dict] = []

    def builder(hand):  # noqa: ANN001 - test double
        seen.append(dict(hand))
        raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        observations.rebuild_tracker_observations(path, hand_builder=builder)
    assert seen == [{"hand_id": "h1"}]


def test_object_for_label_prefers_the_first_labelled_object_with_a_mask() -> None:
    frame = FrameObservations.model_validate(RECORD)
    assert observations.object_for_label(frame, "chassis").object_id == "dam4sam-chassis"
    assert observations.object_for_label(frame, "cabin") is None  # no mask
    assert observations.object_for_label(frame, "cabin", require_mask=False).object_id == (
        "dam4sam-cabin"
    )
    assert observations.object_for_label(frame, "interior", require_mask=False) is None
