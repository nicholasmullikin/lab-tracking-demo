"""Rerun logging helpers shared by the review, comparison and export builders.

The blueprints stay with their products (they *are* the products); what moves here is the
handful of `rr.*` idioms every builder repeated: the `rr.init` + `rr.save` opening, the
RGBA cut-out mask logged as an `EncodedImage`, the `Boxes2D` scaled from an observation's
normalised boxes, and the `TimeSeriesView(origin, name, "$origin/**")` panel.  Each helper
issues exactly the calls its callers issued, with the same archetype fields, so a recording
built through them digests identically (`scripts/dedup_equivalence.py` checks that).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol

import rerun as rr
import rerun.blueprint as rrb

EVERYTHING = "$origin/**"


def init_and_save(
    application_id: str,
    path: str | Path,
    *,
    recording_id: str | None = None,
    default_blueprint: rrb.BlueprintLike | None = None,
) -> Path:
    """Start a recording and stream it to `path`; returns the path.

    `rr.init(application_id, recording_id=...)` (never spawning a viewer) followed by
    `rr.save(path, default_blueprint=...)`, the opening every builder wrote by hand.  The
    caller keeps creating parent directories where it did so before.
    """
    rr.init(application_id, recording_id=recording_id, spawn=False)
    rr.save(str(path), default_blueprint=default_blueprint)
    return Path(path)


def log_rgba_mask(entity: str, png: bytes, *, opacity: float, draw_order: float = 1.0) -> None:
    """Log one RGBA PNG cut-out (the object colour where set, transparent elsewhere)."""
    rr.log(
        entity,
        rr.EncodedImage(
            contents=png, media_type="image/png", opacity=opacity, draw_order=draw_order
        ),
    )


def log_rgba_masks(
    root: str, masks: Iterable[tuple[str, bytes]], *, opacity: float, draw_order: float = 1.0
) -> None:
    """Log `(name, png)` cut-outs under `root/<name>`; the caller decides the names."""
    for name, png in masks:
        log_rgba_mask(f"{root}/{name}", png, opacity=opacity, draw_order=draw_order)


class _NormalisedBox(Protocol):
    x: float
    y: float
    width: float
    height: float


class _Boxed(Protocol):
    @property
    def box(self) -> _NormalisedBox: ...


def box_corners(
    items: Iterable[_Boxed], dimensions: tuple[int, int]
) -> tuple[list[list[float]], list[list[float]]]:
    """`(mins, sizes)` in pixels for items carrying a normalised `box`."""
    width, height = dimensions
    items = list(items)
    mins = [[item.box.x * width, item.box.y * height] for item in items]
    sizes = [[item.box.width * width, item.box.height * height] for item in items]
    return mins, sizes


def log_boxes_from_observation(
    entity: str,
    items: Iterable[_Boxed],
    dimensions: tuple[int, int],
    *,
    labels: Iterable[str],
    colors: Iterable[Any],
    **boxes: Any,
) -> None:
    """Log the items' normalised boxes as one `Boxes2D` scaled to `dimensions`.

    `items` is an observation's `objects` or `hands` (anything with a normalised `box`);
    `labels` and `colors` are per item and stay with the caller because every builder
    formats them differently.  Further keyword arguments (`draw_order`) go to `Boxes2D`.
    """
    mins, sizes = box_corners(items, dimensions)
    rr.log(
        entity,
        rr.Boxes2D(mins=mins, sizes=sizes, labels=list(labels), colors=list(colors), **boxes),
    )


def time_series_view(
    origin: str, name: str, contents: str | Sequence[str] = EVERYTHING
) -> rrb.TimeSeriesView:
    """`rrb.TimeSeriesView(origin=..., name=..., contents=...)`; contents default to the subtree."""
    return rrb.TimeSeriesView(origin=origin, name=name, contents=contents)


TimeSeriesEntry = tuple[str, str] | tuple[str, str, str | Sequence[str]]


def time_series_stack(
    root: str, entries: Iterable[TimeSeriesEntry]
) -> tuple[rrb.TimeSeriesView, ...]:
    """One `TimeSeriesView` per `(relative_origin, name[, contents])` under `root`.

    The caller lays them out (`rrb.Vertical(*stack)`, a tab list, interleaved with other
    views); this only removes the repeated constructor.
    """
    views = []
    for entry in entries:
        relative, name = entry[0], entry[1]
        contents = entry[2] if len(entry) > 2 else EVERYTHING
        views.append(time_series_view(f"{root}/{relative}", name, contents))
    return tuple(views)
