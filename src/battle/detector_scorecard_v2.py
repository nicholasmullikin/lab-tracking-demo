"""Detector scorecard v2: several views and runs, plus the SAM3 appearance detectors.

Extends `battle.detector_scorecard` (imported, not copied) with

- **records**: one tracked run per view, each scored against that view's human review anchors
  (C10379 13 frames, C10119 26, e4 26); the ten existing detectors are computed by the v1
  `compute_series` and the new ones read from a `sam3_appearance.py pass` directory;
- **new detectors** (per frame per part, higher = more suspicious): `emb_self` = 1 - max
  cosine(current mask-pooled embedding, own-part same-view references); `emb_swap` = max other-part
  reference cosine - max own cosine; `emb_distractor` = screwdriver-reference cosine - max own
  cosine; `det_iou` = 1 - best-overlap IoU between the tracked mask and a same-view exemplar
  detection (positives + negatives); `det_top_iou` = 1 - IoU with the top detection; `det_presence`
  = 1 - presence score; `det_disagree` = 1 when the top detection's centroid is > 40 px from the
  tracked mask; `det_top_score` = 1 - top detection score; `_pos` (positives-only exemplars) and
  `_xv` (cross-view exemplars, C10379 masks applied on the other cameras) variants;
- **leave-reference-out**: at a frame that supplied references, the embedding detectors use the
  other reference frames only, and the detection rows already carry `reference_frames_used`;
- **failure classes**: occlusion_leak / rotation_swap / outside from the view's mapped windows,
  distractor for the late anchor frames, and `out_of_frame` for the e4 hidden cells (the head
  camera looks away; the C10379 1700 and C10119 401 hidden cells keep their frame's class);
- **named tests**: the detector values and ranks at (C10119 consensus-only, rear_body after 1533),
  (pm-append, 1050 chassis) and the 23 e4 hidden cells.

Truth is review evidence on 13-26 frames per camera, not ground truth; every table carries counts.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import detector_scorecard as v1
from .fs_common import relative_uri
from .review_anchors import load_mask_set, resolve_run_directory, score_run
from .schemas import DetectorConfidenceRow, DetectorScore, fingerprint

EMBEDDING_DETECTORS: tuple[str, ...] = (
    "emb_self",
    "emb_swap",
    "emb_distractor",
    "emb_self_xv",
    "emb_swap_xv",
)
DETECTION_DETECTORS: tuple[str, ...] = (
    "det_iou",
    "det_top_iou",
    "det_presence",
    "det_disagree",
    "det_top_score",
    "det_iou_pos",
    "det_presence_pos",
    "det_iou_xv",
    "det_top_iou_xv",
    "det_presence_xv",
    "det_disagree_xv",
)
NEW_DETECTORS: tuple[str, ...] = EMBEDDING_DETECTORS + DETECTION_DETECTORS
ALL_DETECTORS: tuple[str, ...] = v1.DETECTORS + NEW_DETECTORS
DISAGREE_PX = 40.0
OUT_OF_FRAME_CLASS = "out_of_frame"
DISTRACTOR_FROM_FRAME = 1195
NEW_DEFINITIONS: dict[str, str] = {
    "emb_self": (
        "1 - max cosine(mask-pooled 1024-d embedding, own-part same-view references); "
        "leave-reference-out."
    ),
    "emb_swap": "max other-part reference cosine - max own-part reference cosine (same view).",
    "emb_distractor": (
        "screwdriver-reference cosine (pm-append rear_body at 1690) - max own-part cosine."
    ),
    "emb_self_xv": "emb_self against another camera's references (cross-view).",
    "emb_swap_xv": "emb_swap against another camera's references (cross-view).",
    "det_iou": (
        "1 - best IoU between the tracked mask and any top-K same-view exemplar detection "
        "(positives + negatives); NaN without a tracked mask."
    ),
    "det_top_iou": "1 - IoU between the tracked mask and the top-scored detection.",
    "det_presence": "1 - SAM3 presence score for the part's same-view exemplars on this frame.",
    "det_disagree": (
        f"1 when the top detection's centroid is > {DISAGREE_PX:g} px from the tracked mask's "
        "centroid, else 0."
    ),
    "det_top_score": "1 - top detection score.",
    "det_iou_pos": "det_iou with positives-only exemplars.",
    "det_presence_pos": "det_presence with positives-only exemplars.",
    "det_iou_xv": "det_iou with cross-view exemplars (C10379 human masks on this camera).",
    "det_top_iou_xv": "det_top_iou with cross-view exemplars.",
    "det_presence_xv": "det_presence with cross-view exemplars.",
    "det_disagree_xv": "det_disagree with cross-view exemplars.",
}
CLAIM_BOUNDARY = (
    "Human review anchors on 13 (C10379), 26 (C10119) and 26 (e4) frames: review evidence that "
    "ranks detectors against each other, not ground truth, not a dataset, no accuracy claim. "
    "Reference cells are scored leave-reference-out. Counts accompany every rate."
)


# ------------------------------------------------------------------------------ pass data


@dataclass
class PassData:
    directory: Path
    frames: np.ndarray
    targets: tuple[str, ...]
    embeddings: np.ndarray  # [F, T, C] float32
    has_mask: np.ndarray
    references: list[tuple[str, int, str, str, np.ndarray]]  # set, frame, target, role, vector
    rows: dict[tuple[int, str, str, str], dict[str, Any]] = field(default_factory=dict)
    sets: dict[str, str] = field(default_factory=dict)  # set name -> view

    def positives(self, set_name: str) -> list[tuple[int, str, np.ndarray]]:
        return [
            (f, t, v)
            for (s, f, t, role, v) in self.references
            if s == set_name and role == "positive"
        ]

    def distractors(self) -> list[tuple[str, int, str, str, np.ndarray]]:
        return [r for r in self.references if r[3].startswith("distractor")]


def load_pass(directory: Path) -> PassData:
    directory = directory.resolve()
    with np.load(directory / "native" / "embeddings.npz") as archive:
        refs = [
            (str(s), int(f), str(t), str(r), np.asarray(v, dtype=np.float32))
            for s, f, t, r, v in zip(
                archive["reference_sets"],
                archive["reference_frames"],
                archive["reference_targets"],
                archive["reference_roles"],
                archive["reference_embeddings"],
                strict=True,
            )
        ]
        data = PassData(
            directory=directory,
            frames=np.asarray(archive["frames"], dtype=int),
            targets=tuple(str(t) for t in archive["targets"]),
            embeddings=np.asarray(archive["embeddings"], dtype=np.float32),
            has_mask=np.asarray(archive["has_mask"], dtype=bool),
            references=refs,
        )
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    data.sets = {s["name"]: s["view"] for s in manifest["reference_sets"]}
    with (directory / "detections.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            data.rows[
                (int(row["frame"]), str(row["target"]), str(row["set"]), str(row["variant"]))
            ] = row
    return data


def _max_cosine(vector: np.ndarray, references: Sequence[np.ndarray]) -> float:
    if not references:
        return float("nan")
    return float(max(float(vector @ r) for r in references))


def embedding_detectors(
    data: PassData,
    *,
    targets: Sequence[str],
    frame_count: int,
    cross_references: Sequence[tuple[int, str, np.ndarray]] = (),
    distractors: Sequence[np.ndarray] = (),
) -> dict[str, np.ndarray]:
    """emb_self / emb_swap / emb_distractor (same view, leave-reference-out) and _xv variants."""
    shape = (frame_count, len(targets))
    out = {name: np.full(shape, np.nan) for name in EMBEDDING_DETECTORS}
    same = data.positives("same_view")
    frame_index = {int(f): i for i, f in enumerate(data.frames)}
    for frame in range(frame_count):
        position = frame_index.get(frame)
        if position is None:
            continue
        for column, target in enumerate(targets):
            if (
                target not in data.targets
                or not data.has_mask[position, data.targets.index(target)]
            ):
                continue
            vector = data.embeddings[position, data.targets.index(target)]
            own = [v for (f, t, v) in same if t == target and f != frame]
            others = [v for (f, t, v) in same if t != target and f != frame]
            own_cos = _max_cosine(vector, own)
            other_cos = _max_cosine(vector, others)
            if math.isfinite(own_cos):
                out["emb_self"][frame, column] = 1.0 - own_cos
                if math.isfinite(other_cos):
                    out["emb_swap"][frame, column] = other_cos - own_cos
                if distractors:
                    out["emb_distractor"][frame, column] = (
                        _max_cosine(vector, distractors) - own_cos
                    )
            if cross_references:
                own_xv = _max_cosine(vector, [v for (_, t, v) in cross_references if t == target])
                other_xv = _max_cosine(vector, [v for (_, t, v) in cross_references if t != target])
                if math.isfinite(own_xv):
                    out["emb_self_xv"][frame, column] = 1.0 - own_xv
                    if math.isfinite(other_xv):
                        out["emb_swap_xv"][frame, column] = other_xv - own_xv
    return out


def detection_detectors(
    data: PassData, *, targets: Sequence[str], frame_count: int
) -> dict[str, np.ndarray]:
    shape = (frame_count, len(targets))
    out = {name: np.full(shape, np.nan) for name in DETECTION_DETECTORS}
    variants = {
        "": ("same_view", "posneg"),
        "_pos": ("same_view", "pos"),
        "_xv": ("cross_view", "posneg"),
    }
    for (frame, target, set_name, variant), row in data.rows.items():
        if frame >= frame_count or target not in targets or row.get("unavailable"):
            continue
        column = list(targets).index(target)
        for suffix, (want_set, want_variant) in variants.items():
            if (set_name, variant) != (want_set, want_variant):
                continue
            presence = row.get("presence")
            if presence is not None and f"det_presence{suffix}" in out:
                out[f"det_presence{suffix}"][frame, column] = 1.0 - float(presence)
            if row.get("best_overlap_iou") is not None and f"det_iou{suffix}" in out:
                out[f"det_iou{suffix}"][frame, column] = 1.0 - float(row["best_overlap_iou"])
            if row.get("top_iou_tracked") is not None and f"det_top_iou{suffix}" in out:
                out[f"det_top_iou{suffix}"][frame, column] = 1.0 - float(row["top_iou_tracked"])
            if row.get("top_centroid_distance_px") is not None and f"det_disagree{suffix}" in out:
                out[f"det_disagree{suffix}"][frame, column] = float(
                    float(row["top_centroid_distance_px"]) > DISAGREE_PX
                )
            if row.get("top_score") is not None and f"det_top_score{suffix}" in out:
                out[f"det_top_score{suffix}"][frame, column] = 1.0 - float(row["top_score"])
    return out


# --------------------------------------------------------------------------------- records


@dataclass
class RecordSpec:
    name: str
    view: str
    view_id: str
    run: Path
    pass_dir: Path
    anchors: Path
    dam4sam_large: Path | None = None
    dam4sam_tiny: Path | None = None
    consensus: Path | None = None
    hull: Path | None = None
    cross_reference_passes: tuple[Path, ...] = ()
    hidden_class: str | None = None  # e.g. out_of_frame for e4
    frame_count: int = 1800
    # Drop cells where the run has no mask at all (a 3-part run has no interior slot): those
    # failures are structural, not appearance failures, and every mask-based detector is NaN there.
    exclude_missing: bool = False


@dataclass
class RecordScore:
    spec: RecordSpec
    targets: tuple[str, ...]
    cells: list[dict[str, Any]]
    values: dict[str, np.ndarray]  # detector -> [F, T]
    reference_frames: tuple[int, ...]
    scores: list[DetectorScore]
    top: tuple[str, ...]
    combined: list[DetectorScore]
    loo: Any
    combined_series: np.ndarray
    abstain_confidence: float | None
    anchor_mean_iou: float | None
    class_counts: dict[str, dict[str, int]]


def load_specs(
    path: Path, repository_root: Path, *, exclude_missing: bool = False
) -> list[RecordSpec]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if exclude_missing:
        raw["exclude_missing"] = True
    specs = []
    for item in raw["records"]:

        def p(key: str) -> Path | None:
            value = item.get(key)
            return (repository_root / value).resolve() if value else None

        specs.append(
            RecordSpec(
                name=item["name"],
                view=item["view"],
                view_id=item["view_id"],
                run=resolve_run_directory(str(repository_root / item["run"]))[1],
                pass_dir=p("pass"),  # type: ignore[arg-type]
                anchors=p("anchors"),  # type: ignore[arg-type]
                dam4sam_large=(
                    resolve_run_directory(str(repository_root / item["dam4sam_large"]))[1]
                    if item.get("dam4sam_large")
                    else None
                ),
                dam4sam_tiny=(
                    resolve_run_directory(str(repository_root / item["dam4sam_tiny"]))[1]
                    if item.get("dam4sam_tiny")
                    else None
                ),
                consensus=p("consensus"),
                hull=p("hull"),
                cross_reference_passes=tuple(
                    (repository_root / c).resolve() for c in item.get("cross_reference_passes", ())
                ),
                hidden_class=item.get("hidden_class"),
                frame_count=int(item.get("frame_count", 1800)),
                exclude_missing=bool(
                    item.get("exclude_missing", raw.get("exclude_missing", False))
                ),
            )
        )
    return specs


def classify_cell(
    frame: int,
    *,
    anchor_state: str,
    windows: Mapping[str, tuple[int, int]],
    hidden_class: str | None,
) -> str:
    if anchor_state == "hidden" and hidden_class:
        return hidden_class
    for name, (low, high) in windows.items():
        if low <= frame < high:
            return v1.DEFAULT_CLASS_BY_WINDOW.get(name, name)
    return v1.DISTRACTOR_CLASS if frame >= DISTRACTOR_FROM_FRAME else v1.OUTSIDE_CLASS


def build_record(spec: RecordSpec, *, repository_root: Path) -> RecordScore:
    mask_set, anchors_root = load_mask_set(spec.anchors, view_id=spec.view_id)
    targets = tuple(mask_set.targets)
    anchor_score = score_run(
        run_name=spec.name, run_directory=spec.run, mask_set=mask_set, anchors_root=anchors_root
    )
    series = v1.compute_series(
        run_directory=spec.run,
        targets=targets,
        frame_count=spec.frame_count,
        view=spec.view,
        dam4sam_large=spec.dam4sam_large,
        dam4sam_tiny=spec.dam4sam_tiny,
        consensus_dir=spec.consensus,
        hull_dir=spec.hull,
    )
    values: dict[str, np.ndarray] = dict(series.values)
    data = load_pass(spec.pass_dir)
    cross: list[tuple[int, str, np.ndarray]] = []
    distractors = [v for (_, _, _, _, v) in data.distractors()]
    for other in spec.cross_reference_passes:
        other_data = load_pass(other)
        cross.extend(other_data.positives("same_view"))
        if not distractors:
            distractors = [v for (_, _, _, _, v) in other_data.distractors()]
    if not cross and "cross_view" in data.sets:
        cross = data.positives("cross_view")
    values.update(
        embedding_detectors(
            data,
            targets=targets,
            frame_count=spec.frame_count,
            cross_references=cross,
            distractors=distractors,
        )
    )
    values.update(detection_detectors(data, targets=targets, frame_count=spec.frame_count))
    reference_frames = tuple(sorted({f for (f, _, _) in data.positives("same_view")}))

    cells = []
    missing_excluded = 0
    for cell in anchor_score.cells:
        failed = v1.cell_failed(cell.outcome, cell.iou, cell.run_area)
        if failed is None:
            continue
        if spec.exclude_missing and cell.outcome == "run_mask_missing":
            missing_excluded += 1
            continue
        frame = cell.analysis_frame_index
        cells.append(
            {
                "record": spec.name,
                "view": spec.view,
                "frame": frame,
                "target": cell.target,
                "anchor_state": cell.anchor_state,
                "outcome": cell.outcome,
                "iou": cell.iou,
                "run_area": cell.run_area,
                "failed": bool(failed),
                "failure_class": classify_cell(
                    frame,
                    anchor_state=cell.anchor_state,
                    windows=mask_set.windows,
                    hidden_class=spec.hidden_class,
                ),
                "reference_frame": frame in reference_frames,
            }
        )
    frames = np.array([c["frame"] for c in cells])
    cell_targets = [c["target"] for c in cells]
    truth = np.array([c["failed"] for c in cells])
    classes = np.array([c["failure_class"] for c in cells])
    columns = np.array([targets.index(t) for t in cell_targets])
    normalized_full = {name: v1.normalized_ranks(values[name]) for name in ALL_DETECTORS}
    raw = {name: values[name][frames, columns] for name in ALL_DETECTORS}
    normalized = {name: normalized_full[name][frames, columns] for name in ALL_DETECTORS}
    for name in ALL_DETECTORS:
        for index, cell in enumerate(cells):
            value = float(raw[name][index])
            cell.setdefault("detectors", {})[name] = value if math.isfinite(value) else None

    scores: list[DetectorScore] = []
    overall_auroc: dict[str, float | None] = {}
    for name in ALL_DETECTORS:
        overall = v1.score_detector(name, v1.ALL_SUBSET, raw[name], truth)
        overall_auroc[name] = overall.auroc
        scores.append(overall)
        scores.extend(v1._class_scores(name, raw[name], truth, classes, overall, v1.RECALL_FLOOR))
    top = v1.select_top_detectors(overall_auroc)
    combined_cells = v1.rank_average(normalized, top) if top else np.full(truth.shape, np.nan)
    combined_overall = v1.score_detector(v1.COMBINED_NAME, v1.ALL_SUBSET, combined_cells, truth)
    combined = [combined_overall]
    combined.extend(
        v1._class_scores(
            v1.COMBINED_NAME, combined_cells, truth, classes, combined_overall, v1.RECALL_FLOOR
        )
    )
    loo = v1.leave_one_frame_out(frames=frames, truth=truth, raw=raw, normalized=normalized)
    combined_series = (
        v1.rank_average(normalized_full, top)
        if top
        else np.full((spec.frame_count, len(targets)), np.nan)
    )
    abstain = (
        1.0 - combined_overall.recall_floor.threshold
        if combined_overall.recall_floor is not None
        and combined_overall.recall_floor.threshold is not None
        else None
    )
    counts: dict[str, dict[str, int]] = {}
    for cell in cells:
        for key in (cell["failure_class"], v1.ALL_SUBSET):
            bucket = counts.setdefault(key, {"cells": 0, "failed": 0})
            bucket["cells"] += 1
            bucket["failed"] += int(cell["failed"])
    counts[v1.ALL_SUBSET]["missing_excluded"] = missing_excluded
    return RecordScore(
        spec=spec,
        targets=targets,
        cells=cells,
        values=values,
        reference_frames=reference_frames,
        scores=scores,
        top=top,
        combined=combined,
        loo=loo,
        combined_series=combined_series,
        abstain_confidence=abstain,
        anchor_mean_iou=anchor_score.mean_iou,
        class_counts=counts,
    )


# ------------------------------------------------------------------------- pooled scoring


def pooled_scores(
    records: Sequence[RecordScore],
) -> tuple[list[DetectorScore], dict[str, list[DetectorScore]]]:
    """Every cell of every record together, raw values per detector (they share units)."""
    truth = np.array([c["failed"] for r in records for c in r.cells])
    classes = np.array([c["failure_class"] for r in records for c in r.cells])
    overall: list[DetectorScore] = []
    per_class: dict[str, list[DetectorScore]] = {}
    for name in ALL_DETECTORS:
        raw = np.array(
            [
                c["detectors"].get(name) if c["detectors"].get(name) is not None else np.nan
                for r in records
                for c in r.cells
            ],
            dtype=float,
        )
        score = v1.score_detector(name, v1.ALL_SUBSET, raw, truth)
        overall.append(score)
        for label in dict.fromkeys(classes.tolist()):
            keep = classes == label
            per_class.setdefault(label, []).append(
                v1.score_detector(
                    name,
                    label,
                    raw[keep],
                    truth[keep],
                    overall_best_f1=score.best_f1.threshold if score.best_f1 else None,
                    overall_recall_floor=score.recall_floor.threshold
                    if score.recall_floor
                    else None,
                )
            )
    return overall, per_class


def leave_one_record_out(records: Sequence[RecordScore], k: int = v1.TOP_K) -> list[dict[str, Any]]:
    """Choose the top-k detectors and the R>=0.8 threshold on the other records, score this one.

    Detectors are rank-normalised within each record (the confidence series' units); the
    threshold chosen on the training records is applied to the held-out record's combined
    rank-average.
    """
    out = []
    for held in records:
        train = [r for r in records if r is not held]
        if not train:
            continue
        truth_train = np.array([c["failed"] for r in train for c in r.cells])
        aurocs: dict[str, float | None] = {}
        normalized_train: dict[str, np.ndarray] = {}
        for name in ALL_DETECTORS:
            pieces = []
            raw_pieces = []
            for r in train:
                frames = np.array([c["frame"] for c in r.cells])
                columns = np.array([r.targets.index(c["target"]) for c in r.cells])
                pieces.append(v1.normalized_ranks(r.values[name])[frames, columns])
                raw_pieces.append(r.values[name][frames, columns])
            normalized_train[name] = np.concatenate(pieces)
            aurocs[name] = v1.auroc(np.concatenate(raw_pieces), truth_train)
        chosen = v1.select_top_detectors(aurocs, k=k)
        if not chosen:
            continue
        combined_train = v1.rank_average(normalized_train, chosen)
        threshold = v1.recall_floor_threshold(combined_train, truth_train)
        frames = np.array([c["frame"] for c in held.cells])
        columns = np.array([held.targets.index(c["target"]) for c in held.cells])
        normalized_held = {
            name: v1.normalized_ranks(held.values[name])[frames, columns] for name in chosen
        }
        combined_held = v1.rank_average(normalized_held, chosen)
        truth_held = np.array([c["failed"] for c in held.cells])
        point = (
            v1.confusion_at(combined_held, truth_held, threshold) if threshold is not None else None
        )
        out.append(
            {
                "held_out_record": held.spec.name,
                "selected_detectors": list(chosen),
                "threshold": threshold,
                "pooled_auroc": v1.auroc(combined_held, truth_held),
                "recall_floor": point.model_dump(mode="json") if point is not None else None,
            }
        )
    return out


# ----------------------------------------------------------------------------- named tests


def named_tests(
    records: Sequence[RecordScore], requests: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Detector values and their percentile rank among the record's cells at named cells."""
    out = []
    by_name = {r.spec.name: r for r in records}
    for request in requests:
        record = by_name.get(str(request["record"]))
        if record is None:
            out.append({**request, "status": "record not scored"})
            continue
        wanted = [
            c
            for c in record.cells
            if c["target"] == request["target"]
            and (request.get("frame") is None or c["frame"] == request["frame"])
            and (request.get("frames") is None or c["frame"] in set(request["frames"]))
            and (
                request.get("anchor_state") is None or c["anchor_state"] == request["anchor_state"]
            )
        ]
        for cell in wanted:
            ranks = {}
            for name in ALL_DETECTORS:
                value = cell["detectors"].get(name)
                if value is None:
                    ranks[name] = None
                    continue
                others = np.array(
                    [
                        c["detectors"][name]
                        for c in record.cells
                        if c["detectors"].get(name) is not None
                    ],
                    dtype=float,
                )
                # Fraction of the record's cells this cell is at least as suspicious as.
                ranks[name] = float(np.mean(others <= value))
            flagged = sorted(
                (name for name, rank in ranks.items() if rank is not None and rank >= 0.9),
                key=lambda n: -(ranks[n] or 0),
            )
            out.append(
                {
                    "test": request.get("label", ""),
                    "record": record.spec.name,
                    "frame": cell["frame"],
                    "target": cell["target"],
                    "anchor_state": cell["anchor_state"],
                    "failed": cell["failed"],
                    "iou": cell["iou"],
                    "run_area": cell["run_area"],
                    "values": cell["detectors"],
                    "percentile_rank": ranks,
                    "flagged_top_decile": flagged,
                }
            )
    return out


def visibility_scores(records: Sequence[RecordScore]) -> list[dict[str, Any]]:
    """Presence as a *visibility* detector: hidden cells (truth = the human said the part is not
    there) against labelled cells, per record, for the same-view and cross-view presence.

    This is a different truth from the scorecard's (which asks whether the tracker failed): a
    hidden cell with no tracked mask is a correct tracker outcome but still a "not here" cell.
    """
    out = []
    for record in records:
        hidden = np.array([c["anchor_state"] == "hidden" for c in record.cells])
        if not hidden.any() or hidden.all():
            continue
        for name in ("det_presence", "det_presence_pos", "det_presence_xv", "det_top_score"):
            values = np.array(
                [
                    c["detectors"].get(name) if c["detectors"].get(name) is not None else np.nan
                    for c in record.cells
                ],
                dtype=float,
            )
            score = v1.score_detector(name, "hidden_vs_visible", values, hidden)
            finite = np.isfinite(values)
            out.append(
                {
                    "record": record.spec.name,
                    "detector": name,
                    "hidden_cells": int((hidden & finite).sum()),
                    "visible_cells": int((~hidden & finite).sum()),
                    "auroc": score.auroc,
                    "mean_presence_hidden": (
                        float(1.0 - np.nanmean(values[hidden])) if (hidden & finite).any() else None
                    ),
                    "mean_presence_visible": (
                        float(1.0 - np.nanmean(values[~hidden]))
                        if (~hidden & finite).any()
                        else None
                    ),
                    "recall_floor": (
                        score.recall_floor.model_dump(mode="json") if score.recall_floor else None
                    ),
                }
            )
    return out


# ---------------------------------------------------------------------------------- report


def _table(scores: Sequence[DetectorScore], title: str) -> list[str]:
    return v1._score_table(scores, title=title)


def _summary_table(records: Sequence[RecordScore]) -> list[str]:
    header = ["detector", *[f"{r.spec.name} AUROC (n fail/ok/undef)" for r in records]]
    lines = [
        "### Overall AUROC per record",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for name in ALL_DETECTORS:
        row = [f"`{name}`"]
        for r in records:
            score = next(s for s in r.scores if s.detector == name and s.subset == v1.ALL_SUBSET)
            row.append(
                f"{v1._fmt(score.auroc)} ({score.positives}/{score.negatives}/{score.undefined})"
            )
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def markdown_report(
    records: Sequence[RecordScore],
    pooled: tuple[list[DetectorScore], dict[str, list[DetectorScore]]],
    loro: Sequence[Mapping[str, Any]],
    tests: Sequence[Mapping[str, Any]],
    visibility: Sequence[Mapping[str, Any]] = (),
) -> str:
    lines = [
        "# Detector scorecard v2: SAM3 appearance detectors on three cameras",
        "",
        f"**{CLAIM_BOUNDARY}**",
        "",
        "## Records",
        "",
        "| record | view | anchor frames | cells | failed | reference frames "
        "(leave-reference-out) | anchor mean IoU | top-3 | in-sample R>=0.8 P/R (TP/FP/FN/TN) | "
        "LOFO R>=0.8 P/R (TP/FP/FN/TN) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        c = next(s for s in r.combined if s.subset == v1.ALL_SUBSET)
        point = c.recall_floor
        loo = r.loo.recall_floor
        lines.append(
            f"| {r.spec.name} | {r.spec.view} | {len(set(x['frame'] for x in r.cells))} | "
            f"{r.class_counts['all']['cells']} | {r.class_counts['all']['failed']}"
            + (
                f" ({r.class_counts['all']['missing_excluded']} no-mask cells excluded)"
                if r.class_counts["all"].get("missing_excluded")
                else ""
            )
            + " | "
            f"{', '.join(str(f) for f in r.reference_frames)} | {v1._fmt(r.anchor_mean_iou)} | "
            f"{', '.join(r.top)} | "
            + (
                f"{v1._fmt(point.precision)} / {v1._fmt(point.recall)} "
                f"({point.tp}/{point.fp}/{point.fn}/{point.tn})"
                if point
                else "-"
            )
            + " | "
            + (
                f"{v1._fmt(loo.precision)} / {v1._fmt(loo.recall)} "
                f"({loo.tp}/{loo.fp}/{loo.fn}/{loo.tn})"
                if loo
                else "-"
            )
            + " |"
        )
    lines += ["", "## New detectors", ""]
    for name in NEW_DETECTORS:
        lines.append(f"- `{name}`: {NEW_DEFINITIONS[name]}")
    lines += ["", "## Overall AUROC per record", ""]
    lines += _summary_table(records)[2:]
    overall, per_class = pooled
    n_cells = sum(len(r.cells) for r in records)
    n_failed = sum(r.class_counts["all"]["failed"] for r in records)
    lines += [
        "## Pooled over every record",
        "",
        f"{n_cells} cells ({n_failed} failed) from {len(records)} records; raw detector values "
        "pooled.",
        "",
    ]
    lines += _table(overall, "Pooled, overall")
    for label, scores in per_class.items():
        failed = sum(
            1 for r in records for c in r.cells if c["failure_class"] == label and c["failed"]
        )
        total = sum(1 for r in records for c in r.cells if c["failure_class"] == label)
        lines += _table(
            scores,
            f"Pooled, class `{label}` ({total} cells, {failed} failed; thresholds inside the "
            "class)",
        )
    lines += [
        "## Leave-one-record-out (top-3 and the R>=0.8 threshold chosen on the other records)",
        "",
        "| held-out record | selected detectors | threshold | pooled AUROC | P | R | TP/FP/FN/TN |",
        "|---|---|---|---|---|---|---|",
    ]
    for item in loro:
        point = item.get("recall_floor") or {}
        lines.append(
            f"| {item['held_out_record']} | {', '.join(item['selected_detectors'])} | "
            f"{v1._fmt(item['threshold'])} | {v1._fmt(item['pooled_auroc'])} | "
            f"{v1._fmt(point.get('precision'))} | {v1._fmt(point.get('recall'))} | "
            f"{point.get('tp', '-')}/{point.get('fp', '-')}/{point.get('fn', '-')}/"
            f"{point.get('tn', '-')} |"
        )
    lines += ["", "## Per-record tables", ""]
    for r in records:
        lines += [f"### `{r.spec.name}` ({r.spec.view})", ""]
        lines += ["| class | cells | failed |", "|---|---|---|"]
        for label, bucket in r.class_counts.items():
            lines.append(f"| {label} | {bucket['cells']} | {bucket['failed']} |")
        lines.append("")
        failed_cells = [c for c in r.cells if c["failed"]]
        lines.append(
            "Failed cells: "
            + (
                ", ".join(
                    f"{c['frame']} {c['target']} ("
                    + (
                        v1._fmt(c["iou"], 2)
                        if c["iou"] is not None
                        else f"{c['run_area']} px hidden"
                    )
                    + ")"
                    for c in failed_cells
                )
                or "none"
            )
            + "."
        )
        lines.append("")
        lines += _table(
            [s for s in r.scores if s.subset == v1.ALL_SUBSET], f"{r.spec.name}: overall"
        )
        for label in [k for k in r.class_counts if k != v1.ALL_SUBSET]:
            subset = [s for s in r.scores if s.subset == label]
            bucket = r.class_counts[label]
            if bucket["failed"] == 0 or bucket["failed"] == bucket["cells"]:
                lines.append(
                    f"Class `{label}`: {bucket['cells']} cells, {bucket['failed']} failed; "
                    "AUROC undefined."
                )
                lines.append("")
                continue
            lines += _table(
                subset,
                f"{r.spec.name}: class `{label}` ({bucket['cells']} cells, "
                f"{bucket['failed']} failed)",
            )
        lines += v1._score_table(
            r.combined, title=f"{r.spec.name}: combined (in-sample)", show_subset=True
        )
        loo = r.loo
        lines += [
            f"Leave-one-frame-out: pooled AUROC {v1._fmt(loo.pooled_auroc)}; R>=0.8 P/R "
            + (
                f"{v1._fmt(loo.recall_floor.precision)} / {v1._fmt(loo.recall_floor.recall)} "
                f"({loo.recall_floor.tp}/{loo.recall_floor.fp}/{loo.recall_floor.fn}/{loo.recall_floor.tn})"
                if loo.recall_floor
                else "-"
            )
            + "; F1-max P/R "
            + (
                f"{v1._fmt(loo.best_f1.precision)} / {v1._fmt(loo.best_f1.recall)} "
                f"({loo.best_f1.tp}/{loo.best_f1.fp}/{loo.best_f1.fn}/{loo.best_f1.tn})"
                if loo.best_f1
                else "-"
            )
            + ".",
            "",
        ]
    lines += ["## Named tests", ""]
    lines += [
        "| test | record | frame | target | state | failed | IoU / px | flagged in the top "
        "decile of the record's cells | det_iou | det_presence | det_disagree | emb_self | "
        "emb_swap | emb_distractor | sam3_score |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for t in tests:
        if t.get("status"):
            lines.append(
                f"| {t.get('label', '')} | {t['record']} | - | {t.get('target', '')} | - | - | - | "
                f"{t['status']} | | | | | | | |"
            )
            continue
        detail = v1._fmt(t["iou"], 2) if t["iou"] is not None else f"{t['run_area']} px"
        values = t["values"]
        lines.append(
            f"| {t['test']} | {t['record']} | {t['frame']} | {t['target']} | {t['anchor_state']} | "
            f"{'yes' if t['failed'] else 'no'} | {detail} | "
            f"{', '.join(t['flagged_top_decile']) or 'nothing'} | "
            + " | ".join(
                v1._fmt(values.get(n), 3)
                for n in (
                    "det_iou",
                    "det_presence",
                    "det_disagree",
                    "emb_self",
                    "emb_swap",
                    "emb_distractor",
                    "sam3_score",
                )
            )
            + " |"
        )
    lines += [
        "",
        "## Presence as a visibility detector (hidden vs labelled cells; a different truth)",
        "",
        "| record | detector | hidden / visible cells | AUROC | mean presence hidden | "
        "mean presence visible | R>=0.8 P / R (TP/FP/FN/TN) |",
        "|---|---|---|---|---|---|---|",
    ]
    for item in visibility:
        point = item.get("recall_floor") or {}
        lines.append(
            f"| {item['record']} | `{item['detector']}` | {item['hidden_cells']} / "
            f"{item['visible_cells']} | {v1._fmt(item['auroc'])} | "
            f"{v1._fmt(item['mean_presence_hidden'])} | {v1._fmt(item['mean_presence_visible'])} | "
            f"{v1._fmt(point.get('precision'))} / {v1._fmt(point.get('recall'))} "
            f"({point.get('tp', '-')}/{point.get('fp', '-')}/{point.get('fn', '-')}/"
            f"{point.get('tn', '-')}) |"
        )
    lines.append("")
    return "\n".join(lines)


def confidence_rows(record: RecordScore) -> list[DetectorConfidenceRow]:
    truth = {(c["frame"], c["target"]): c["failed"] for c in record.cells}
    anchor_frames = {c["frame"] for c in record.cells}
    rows = []
    for frame in range(record.spec.frame_count):
        for column, target in enumerate(record.targets):
            combined = float(record.combined_series[frame, column])
            defined = math.isfinite(combined)
            confidence = 1.0 - combined if defined else None
            rows.append(
                DetectorConfidenceRow(
                    analysis_frame_index=frame,
                    target=target,
                    detectors={
                        name: (
                            float(record.values[name][frame, column])
                            if math.isfinite(float(record.values[name][frame, column]))
                            else None
                        )
                        for name in ALL_DETECTORS
                    },
                    combined_suspicion=v1._clip01(combined) if defined else None,
                    confidence=v1._clip01(confidence) if confidence is not None else None,
                    abstain=(
                        True
                        if confidence is None
                        else (
                            record.abstain_confidence is not None
                            and confidence <= record.abstain_confidence
                        )
                    ),
                    is_anchor_frame=frame in anchor_frames,
                    anchor_truth_failed=truth.get((frame, target)),
                )
            )
    return rows


def write_outputs(
    *,
    records: Sequence[RecordScore],
    output_dir: Path,
    repository_root: Path,
    spec_path: Path,
    tests: Sequence[Mapping[str, Any]],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    pooled = pooled_scores(records)
    loro = leave_one_record_out(records)
    visibility = visibility_scores(records)
    report = markdown_report(records, pooled, loro, tests, visibility)
    (output_dir / "scorecard_v2.md").write_text(report, encoding="utf-8")
    payload = {
        "manifest_kind": "detector_scorecard_v2",
        "spec": fingerprint(spec_path, repository_root).model_dump(mode="json"),
        "detectors": list(ALL_DETECTORS),
        "definitions": {**v1.DETECTOR_DEFINITIONS, **NEW_DEFINITIONS},
        "records": [
            {
                "name": r.spec.name,
                "view": r.spec.view,
                "run": relative_uri(r.spec.run, repository_root),
                "pass": relative_uri(r.spec.pass_dir, repository_root),
                "anchors": relative_uri(r.spec.anchors, repository_root),
                "reference_frames": list(r.reference_frames),
                "anchor_mean_iou": r.anchor_mean_iou,
                "class_counts": r.class_counts,
                "cells": r.cells,
                "scores": [s.model_dump(mode="json") for s in r.scores],
                "top_detectors": list(r.top),
                "combined": [s.model_dump(mode="json") for s in r.combined],
                "leave_one_frame_out": r.loo.model_dump(mode="json"),
                "abstain_confidence_threshold": r.abstain_confidence,
                "confidence_uri": relative_uri(
                    output_dir / f"confidence_v2_{r.spec.name}.jsonl", repository_root
                ),
            }
            for r in records
        ],
        "pooled": {
            "overall": [s.model_dump(mode="json") for s in pooled[0]],
            "per_class": {k: [s.model_dump(mode="json") for s in v] for k, v in pooled[1].items()},
        },
        "leave_one_record_out": loro,
        "named_tests": list(tests),
        "presence_as_visibility": visibility,
        "generated_at": datetime.now(UTC).isoformat(),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / "scorecard_v2.json").write_text(
        json.dumps(payload, indent=1) + "\n", encoding="utf-8"
    )
    for r in records:
        rows = confidence_rows(r)
        # Two copies: a flat file per record, and the `RUN_DIR/confidence.jsonl` layout that
        # `battle-build-interaction-review-v4 --confidence RUN_DIR` reads.
        alias = output_dir / "confidence_v2" / r.spec.name
        alias.mkdir(parents=True, exist_ok=True)
        for path in (output_dir / f"confidence_v2_{r.spec.name}.jsonl", alias / "confidence.jsonl"):
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(row.model_dump_json() + "\n")
    return output_dir / "scorecard_v2.md"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--spec", type=Path, required=True, help="records JSON (see module docstring)"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--exclude-missing",
        action="store_true",
        help="drop cells where the run has no mask (structural failures of a 3-part run)",
    )
    args = parser.parse_args()
    root = Path.cwd().resolve()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))

    records = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for record_spec in load_specs(args.spec, root, exclude_missing=args.exclude_missing):
            print(f"scoring {record_spec.name}", flush=True)
            records.append(build_record(record_spec, repository_root=root))
        tests = named_tests(records, spec.get("named_tests", ()))
        path = write_outputs(
            records=records,
            output_dir=args.output.resolve(),
            repository_root=root,
            spec_path=args.spec.resolve(),
            tests=tests,
        )
    print(f"Scorecard: {path}")


if __name__ == "__main__":
    main()
