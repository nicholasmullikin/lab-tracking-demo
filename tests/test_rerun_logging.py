"""The shared Rerun idioms issue exactly the calls the builders wrote by hand."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import rerun as rr
import rerun.blueprint as rrb

from battle import rerun_logging


def _batches(archetype: Any) -> list[tuple[str, list[Any]]]:
    return [
        (str(batch.component_descriptor()), batch.as_arrow_array().to_pylist())
        for batch in archetype.as_component_batches()
    ]


def _capture(monkeypatch) -> list[tuple[str, Any, dict[str, Any]]]:
    calls: list[tuple[str, Any, dict[str, Any]]] = []
    monkeypatch.setattr(
        rerun_logging.rr, "log", lambda path, value, **kw: calls.append((str(path), value, kw))
    )
    return calls


def test_init_and_save_writes_the_recording_under_its_ids(tmp_path: Path) -> None:
    from rerun.experimental import RrdReader

    target = tmp_path / "nested" / "out.rrd"
    target.parent.mkdir()
    returned = rerun_logging.init_and_save("battle-test-app", target, recording_id="rec-42")
    rr.log("world/text", rr.TextDocument("hello"), static=True)
    rr.disconnect()
    assert returned == target
    assert target.is_file()
    stores = list(RrdReader(target).recordings())
    assert len(stores) == 1
    assert stores[0].recording_id == "rec-42"
    assert stores[0].application_id == "battle-test-app"


def test_init_and_save_accepts_a_string_path_and_a_default_blueprint(tmp_path: Path) -> None:
    from rerun.experimental import RrdReader

    blueprint = rrb.Blueprint(rrb.TextDocumentView(origin="world/text", name="Text"))
    target = tmp_path / "out.rrd"
    rerun_logging.init_and_save("battle-test-app", str(target), default_blueprint=blueprint)
    rr.log("world/text", rr.TextDocument("hello"), static=True)
    rr.disconnect()
    reader = RrdReader(target)
    assert len(list(reader.recordings())) == 1
    assert len(list(reader.blueprints())) == 1


def test_log_rgba_masks_logs_each_cut_out_under_the_root_with_the_same_archetype(
    monkeypatch,
) -> None:
    calls = _capture(monkeypatch)
    png_a, png_b = b"\x89PNG-a", b"\x89PNG-b"
    rerun_logging.log_rgba_masks(
        "world/view/masks", [("chassis", png_a), ("cabin", png_b)], opacity=0.35
    )
    assert [path for path, _, _ in calls] == ["world/view/masks/chassis", "world/view/masks/cabin"]
    expected = rr.EncodedImage(contents=png_a, media_type="image/png", opacity=0.35, draw_order=1.0)
    assert _batches(calls[0][1]) == _batches(expected)
    assert calls[0][2] == {}
    rerun_logging.log_rgba_mask("world/x", png_b, opacity=0.5, draw_order=3.0)
    expected = rr.EncodedImage(contents=png_b, media_type="image/png", opacity=0.5, draw_order=3.0)
    assert _batches(calls[-1][1]) == _batches(expected)


def test_log_rgba_masks_consumes_a_lazy_iterable_in_order(monkeypatch) -> None:
    calls = _capture(monkeypatch)
    seen: list[str] = []

    def cut_outs():
        for name in ("a", "b"):
            seen.append(f"decode {name}")
            yield name, name.encode()
            seen.append(f"after {name}")

    rerun_logging.log_rgba_masks("root", cut_outs(), opacity=0.4)
    assert [path for path, _, _ in calls] == ["root/a", "root/b"]
    # The mask is logged between its decode and the next one's, as the per-item loops did.
    assert seen == ["decode a", "after a", "decode b", "after b"]


def _item(x: float, y: float, w: float, h: float, label: str, object_id: str) -> Any:
    return SimpleNamespace(
        box=SimpleNamespace(x=x, y=y, width=w, height=h), label=label, object_id=object_id
    )


def test_log_boxes_from_observation_matches_the_hand_written_boxes2d(monkeypatch) -> None:
    calls = _capture(monkeypatch)
    items = [_item(0.1, 0.2, 0.3, 0.4, "chassis", "o1"), _item(0.5, 0.25, 0.25, 0.5, "cabin", "o2")]
    width, height = 1280, 720
    colors = [(70, 130, 255), (255, 95, 100)]
    rerun_logging.log_boxes_from_observation(
        "world/view/objects",
        items,
        (width, height),
        labels=[f"{item.label} ({item.object_id})" for item in items],
        colors=colors,
        draw_order=2.0,
    )
    expected = rr.Boxes2D(
        mins=[[item.box.x * width, item.box.y * height] for item in items],
        sizes=[[item.box.width * width, item.box.height * height] for item in items],
        labels=[f"{item.label} ({item.object_id})" for item in items],
        colors=colors,
        draw_order=2.0,
    )
    assert len(calls) == 1
    assert calls[0][0] == "world/view/objects"
    assert _batches(calls[0][1]) == _batches(expected)
    # Without extra keywords no draw order is sent, as in the callers that never set one.
    rerun_logging.log_boxes_from_observation(
        "p", items, (16, 8), labels=["a", "b"], colors=[(1, 2, 3)] * 2
    )
    assert _batches(calls[1][1]) == _batches(
        rr.Boxes2D(
            mins=[[1.6, 1.6], [8.0, 2.0]],
            sizes=[[4.8, 3.2], [4.0, 4.0]],
            labels=["a", "b"],
            colors=[(1, 2, 3)] * 2,
        )
    )


def test_box_corners_scales_normalised_boxes_to_pixels() -> None:
    mins, sizes = rerun_logging.box_corners([_item(0.5, 0.5, 0.25, 0.5, "x", "y")], (100, 50))
    assert mins == [[50.0, 25.0]]
    assert sizes == [[25.0, 25.0]]


def _view_fields(view: rrb.TimeSeriesView) -> tuple[str, Any, Any]:
    return (str(view.origin), view.name, view.contents)


def test_time_series_view_defaults_to_the_subtree_and_keeps_explicit_contents() -> None:
    default = rerun_logging.time_series_view("world/x/diagnostics", "Contact")
    reference = rrb.TimeSeriesView(origin="world/x/diagnostics", name="Contact")
    assert _view_fields(default) == _view_fields(reference)
    assert default.contents == "$origin/**"
    explicit = rerun_logging.time_series_view(
        "world/x/anchors", "Anchors", ("$origin/anchor_frame", "$origin/failed_cells")
    )
    assert explicit.contents == ("$origin/anchor_frame", "$origin/failed_cells")


def test_time_series_stack_builds_one_view_per_entry_under_the_root() -> None:
    stack = rerun_logging.time_series_stack(
        "world/x",
        (
            ("growth", "Growth ratios"),
            ("hands", "Hands"),
            ("anchors", "Anchors", ("$origin/anchor_frame",)),
        ),
    )
    assert [_view_fields(view) for view in stack] == [
        ("world/x/growth", "Growth ratios", "$origin/**"),
        ("world/x/hands", "Hands", "$origin/**"),
        ("world/x/anchors", "Anchors", ("$origin/anchor_frame",)),
    ]
    assert all(isinstance(view, rrb.TimeSeriesView) for view in stack)
