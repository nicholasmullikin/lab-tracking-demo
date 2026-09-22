"""Readers for a run's `observations.jsonl`, one `FrameObservations` per line.

Two ways to read the file, kept apart because they accept different inputs:

* :func:`load_observations` validates every line against the schema
  (`FrameObservations.model_validate_json`), so an unknown key is an error and every declared
  field (hands, tracker diagnostics, object scores) is kept.  This is the reader for files the
  Battle exporters wrote.  :func:`observations_by_frame` is the same keyed by frame.
* :func:`rebuild_tracker_observations` is the tolerant rebuild the external-worker drivers
  used (DAM4SAM, SAMURAI, Grounding DINO + SAM2, BoxMOT, WiLoR): the frame fields and, per
  object, `object_id` / `label` / `confidence` / `box` / `mask` are taken and everything else
  a worker may have written is ignored; hands are rebuilt through `hand_builder` when a
  driver supplies one (WiLoR).

:func:`object_for_label` is the repeated `next(...)` lookup for a part's object on a frame.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .schemas import FrameObservations, MaskReference, NormalizedBox, PerFrameHand, PerFrameObject


def load_observations(path: Path) -> tuple[FrameObservations, ...]:
    """Validate the file's records one line at a time; an invalid line names `path:line`."""
    observations: list[FrameObservations] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if line.strip():
                try:
                    observations.append(FrameObservations.model_validate_json(line))
                except ValueError as error:
                    raise ValueError(
                        f"invalid observation at {path}:{line_number}: {error}"
                    ) from error
    return tuple(observations)


def observations_by_frame(path: Path) -> dict[int, FrameObservations]:
    """`load_observations` keyed by `analysis_frame_index` (a later duplicate frame wins)."""
    return {
        observation.analysis_frame_index: observation for observation in load_observations(path)
    }


def rebuild_tracker_observations(
    path: Path,
    *,
    hand_builder: Callable[[Mapping[str, Any]], PerFrameHand] | None = None,
) -> tuple[FrameObservations, ...]:
    """Rebuild worker records from their known fields, ignoring anything else they carry.

    Blank lines are skipped.  Objects keep `object_id`, `label`, `confidence`, `box` and an
    optional `mask`; `hands` are rebuilt only when `hand_builder` is given.
    """
    observations: list[FrameObservations] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        objects = tuple(
            PerFrameObject(
                object_id=obj["object_id"],
                label=obj["label"],
                confidence=float(obj["confidence"]),
                box=NormalizedBox(**obj["box"]),
                mask=MaskReference(**obj["mask"]) if obj.get("mask") else None,
            )
            for obj in payload.get("objects", ())
        )
        hands = (
            tuple(hand_builder(hand) for hand in payload.get("hands", ()))
            if hand_builder is not None
            else ()
        )
        observations.append(
            FrameObservations(
                view_id=payload["view_id"],
                analysis_frame_index=int(payload["analysis_frame_index"]),
                source_seconds=float(payload["source_seconds"]),
                objects=objects,
                hands=hands,
            )
        )
    return tuple(observations)


def object_for_label(
    frame: FrameObservations, label: str, *, require_mask: bool = True
) -> PerFrameObject | None:
    """First object on the frame with this label (and, by default, a mask), else None."""
    return next(
        (
            item
            for item in frame.objects
            if item.label == label and (not require_mask or item.mask is not None)
        ),
        None,
    )
