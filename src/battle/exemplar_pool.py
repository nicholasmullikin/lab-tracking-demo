"""Exemplar detections as correction candidates, and the reference specs that feed them.

Battle-side companion of `sam3_appearance.py` (which runs the SAM3 model under the MuggledSAM
interpreter). This module owns:

- `references`: the reference spec (human masks per frame and part, one set per view) that the
  GPU pass and the exemplar decoder read: C10379 from the human frame-0 seeds and the human
  correction frames (327 / 900 / 1235; the agent-selected 1172 correction is left out and
  said so), C10119 and e4 from the run's frame-0 seeds plus the human anchors on their
  earliest interior frame (41 / 224), a screwdriver distractor cut from the pm-append
  rear_body mask at a non-anchor frame, and a `cross_view` set copied from another spec;
- `ExemplarDecoderClient`: a `WorkerClient` look-alike that starts `sam3_appearance.py
  serve-jsonl` so `battle-multiview-reprompt decode --candidate-source exemplar_detector|both`
  and the acceptance-search harness get exemplar detections (gated to the prompt box, ranked
  by detection score) through the same `batch_decode` protocol as the image decoder;
- `compare`: the pool comparison on truth cells (oracle ceiling, held-out IoU, acceptance and
  harm under the tool's filters, leave-frames-out) between the exemplar pool read from a pass's
  kept detections and the image-decoder pool of the acceptance search;
- `zero-shot`: recording-1 exemplars applied to recording 2, IoU against the human masks
  accepted there so far and the presence score at the hidden cells.

Every IoU here is against one person's choice of SAM3 decoder masks on a handful of frames:
review evidence that ranks pools against each other, not ground truth.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import mask_cache
from .fs_common import relative_uri
from .review_anchors import load_mask_set, resolve_run_directory, view_paths
from .sam3_appearance import (
    CANDIDATE_SOURCES,
    REFERENCE_SCHEMA,
    VARIANTS,
    mask_iou,
    read_run_masks,
)
from .schemas import fingerprint

CORRECTIONS_CALIBRATION = Path(
    "runs/muggledsam-sam3-four-part-focused-corrections-agent-swap-20260918t000947z/calibration_manifest.json"
)
PM_APPEND_RUN = Path("runs/sam3-memory-arms-20260919/arms/pm-append")
DISTRACTOR_FRAME = 1690
DISTRACTOR_TARGET = "rear_body"
FIRST_MINUTE = 1800
TARGETS = ("chassis", "interior", "rear_body", "cabin")
VIEW_IDS = {"C10379": "static-c10379", "C10119": "static-c10119", "HMC_21179183": "ego-hmc21179183"}
HUMAN_REFERENCE_FRAME = {"C10119": 41, "HMC_21179183": 224}
HARM_IOU = 0.4
BAR_IOU = 0.6
BAR_HARM = 0.15
CLAIM_BOUNDARY = (
    "Reference masks and truth masks are one person's choice of SAM3 decoder masks on a handful "
    "of frames per camera: review evidence that ranks pools and detectors against each other, not "
    "ground truth, not a dataset, no accuracy claim. Counts accompany every rate."
)


# ------------------------------------------------------------------------------ references


def _fingerprint(path: Path, repository_root: Path) -> dict[str, Any]:
    return fingerprint(path, repository_root).model_dump(mode="json")


def proxy_for_run(run_directory: Path) -> Path:
    source = json.loads((run_directory / "input_1800f.mp4.source.json").read_text(encoding="utf-8"))
    return Path(source["source"])


def _reference(
    frame: int, target: str, mask_path: Path, provenance: str, repository_root: Path
) -> dict[str, Any]:
    mask_path = mask_path.resolve()
    if not mask_path.is_file():
        raise FileNotFoundError(f"reference mask is unavailable: {mask_path}")
    return {
        "frame": frame,
        "target": target,
        "mask_path": str(mask_path),
        "role": "positive",
        "provenance": provenance,
        "fingerprint": _fingerprint(mask_path, repository_root),
    }


def c10379_human_references(repository_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Human-accepted seeds and corrections of the C10379 calibration inside the first minute."""
    manifest_path = repository_root / CORRECTIONS_CALIBRATION
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    workspace = manifest_path.parent
    references: list[dict[str, Any]] = []
    left_out: list[str] = []
    for candidate in manifest["candidates"]:
        frame = int(candidate["frame"]["analysis_frame_index"])
        if not candidate.get("human_accepted") or candidate.get("rejected"):
            continue
        if frame >= FIRST_MINUTE:
            left_out.append(f"{frame} {candidate['intended_target']}: outside the first minute")
            continue
        if candidate.get("selected_by") != "human":
            left_out.append(
                f"{frame} {candidate['intended_target']}: "
                f"selected_by={candidate.get('selected_by')}, "
                "not a human mask"
            )
            continue
        index = int(candidate["human_selected_candidate_index"])
        option = next(
            o
            for o in candidate["decoder_result"]["candidates"]
            if int(o["candidate_index"]) == index
        )
        provenance = "human_seed_frame0" if frame == 0 else "human_correction"
        references.append(
            _reference(
                frame,
                str(candidate["intended_target"]),
                workspace / option["mask_uri"],
                provenance,
                repository_root,
            )
        )
    references.sort(key=lambda r: (r["frame"], TARGETS.index(r["target"])))
    return references, left_out


def run_seed_references(
    run_directory: Path, repository_root: Path, *, targets: Sequence[str] = TARGETS
) -> list[dict[str, Any]]:
    """The run's own frame-0 masks (its seeds) as references, provenance from its manifest."""
    masks = read_run_masks(run_directory, 1).get(0, {})
    manifest = json.loads((run_directory / "manifest.json").read_text(encoding="utf-8"))
    provenance = seed_provenance_by_target(manifest)
    out = []
    for target in targets:
        path = masks.get(target)
        if path is None:
            continue
        label = provenance.get(target, "unknown")
        if not is_human_seed_provenance(label):
            # Geometric or agent seeds are not human masks; the human anchors supply the part.
            continue
        out.append(_reference(0, target, Path(path), f"run_seed:{label}", repository_root))
    return out


HUMAN_SEED_PROVENANCES = frozenset(
    {"agent_proposed_human_accepted", "human", "human_seed", "manual"}
)


def is_human_seed_provenance(label: str) -> bool:
    """Only masks a human chose or accepted count as references (a geometric seed built *from*
    human masks on other cameras is still an agent mask on this one)."""
    return label in HUMAN_SEED_PROVENANCES or label.startswith("human_")


def seed_provenance_by_target(manifest: Mapping[str, Any]) -> dict[str, str]:
    """Per-part seed provenance from a multiview run manifest (`seeds[].provenance`)."""
    out: dict[str, str] = {}
    for key in ("four_part_multiview", "four_part_focused", "four_part_full"):
        block = manifest.get(key)
        if not isinstance(block, dict):
            continue
        for seed in block.get("seeds") or ():
            if isinstance(seed, dict) and seed.get("target"):
                out[str(seed["target"])] = str(
                    seed.get("provenance") or block.get("seed_provenance") or ""
                )
        if out:
            break
    return out


def anchor_references(
    view_id: str, frame: int, repository_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Every labelled human anchor on one frame of a view's anchor set."""
    _, workspace, _ = view_paths(view_id)
    mask_set, root = load_mask_set(repository_root / workspace, view_id=view_id)
    out = []
    for anchor in mask_set.anchors:
        if (
            anchor.analysis_frame_index != frame
            or anchor.state != "labeled"
            or anchor.mask_uri is None
        ):
            continue
        out.append(
            _reference(
                frame,
                anchor.target,
                root / anchor.mask_uri,
                f"human_anchor:{view_id}",
                repository_root,
            )
        )
    return out, _fingerprint(root / "anchors" / "anchor_masks.json", repository_root)


def distractor_reference(repository_root: Path) -> dict[str, Any]:
    """The pm-append rear_body mask at 1690: the tracker on the screwdriver, no anchor there."""
    _, run_directory = resolve_run_directory(str(repository_root / PM_APPEND_RUN))
    masks = read_run_masks(run_directory, DISTRACTOR_FRAME + 1).get(DISTRACTOR_FRAME, {})
    path = masks.get(DISTRACTOR_TARGET)
    if path is None:
        raise FileNotFoundError(f"pm-append has no {DISTRACTOR_TARGET} mask at {DISTRACTOR_FRAME}")
    return {
        "frame": DISTRACTOR_FRAME,
        "target": DISTRACTOR_TARGET,
        "mask_path": str(Path(path).resolve()),
        "label": "screwdriver",
        "provenance": (
            "tracked pm-append rear_body mask on the screwdriver at a non-anchor frame 10 frames "
            "before the human's 1700 hidden mark (distractor_confusion)"
        ),
        "fingerprint": _fingerprint(Path(path), repository_root),
    }


def build_reference_spec(
    *,
    view: str,
    run_directory: Path,
    repository_root: Path,
    cross_view_spec: Path | None = None,
    include_distractor: bool = True,
) -> dict[str, Any]:
    """Reference spec for one view: `same_view` set (human masks) plus an optional `cross_view`."""
    run_directory = run_directory.resolve()
    proxy = proxy_for_run(run_directory)
    notes: list[str] = []
    sources: dict[str, Any] = {}
    if view == "C10379":
        references, left_out = c10379_human_references(repository_root)
        notes.extend(f"left out: {item}" for item in left_out)
        sources["corrections_calibration"] = _fingerprint(
            repository_root / CORRECTIONS_CALIBRATION, repository_root
        )
    else:
        view_id = VIEW_IDS[view]
        seeds = run_seed_references(run_directory, repository_root)
        anchors, fingerprint = anchor_references(
            view_id, HUMAN_REFERENCE_FRAME[view], repository_root
        )
        anchor_targets = {a["target"] for a in anchors}
        # Where the human labelled a part on the reference frame, the run's own seed for that part
        # is kept too (both are references; the anchor frame is scored leave-reference-out).
        references = seeds + anchors
        notes.append(
            f"human anchors on frame {HUMAN_REFERENCE_FRAME[view]}: {sorted(anchor_targets)}; run "
            f"seeds at frame 0: {sorted(s['target'] for s in seeds)}"
        )
        sources["anchor_set"] = fingerprint
    same_view = {
        "name": "same_view",
        "view": view,
        "proxy": str(proxy),
        "references": references,
        "distractors": [distractor_reference(repository_root)]
        if include_distractor and view == "C10379"
        else [],
    }
    sets = [same_view]
    if cross_view_spec is not None:
        other = json.loads(cross_view_spec.read_text(encoding="utf-8"))
        other_same = next(s for s in other["sets"] if s["name"] == "same_view")
        if other_same["view"] == view:
            raise ValueError("a cross-view spec must come from another view")
        sets.append({**other_same, "name": "cross_view", "distractors": []})
        sources["cross_view_spec"] = _fingerprint(cross_view_spec, repository_root)
    return {
        "schema": REFERENCE_SCHEMA,
        "view": view,
        "run_directory": relative_uri(run_directory, repository_root),
        "sets": sets,
        "sources": sources,
        "notes": notes,
        "generated_at": datetime.now(UTC).isoformat(),
        "claim_boundary": CLAIM_BOUNDARY,
    }


# ------------------------------------------------------------------------- decoder client


class ExemplarDecoderClient:
    """Starts `sam3_appearance.py serve-jsonl`; same interface as `WorkerClient`."""

    def __init__(
        self,
        *,
        proxy_path: Path,
        references: Path,
        results_directory: Path,
        stderr_path: Path,
        candidate_source: str = "exemplar_detector",
        exemplar_set: str = "same_view",
        exemplar_variant: str = "posneg",
        top_k: int = 10,
        external_python: Path | None = None,
        model: Path | None = None,
        device: str = "cuda:0",
        allow_gpu_neighbours: Iterable[int] = (),
    ) -> None:
        from .muggled_calibration_web import WorkerClient
        from .muggled_smoke import DEFAULT_MODEL, MUGGLED_SAM_PYTHON, MUGGLED_SAM_SOURCE

        if candidate_source not in CANDIDATE_SOURCES:
            raise ValueError(f"unknown candidate source {candidate_source!r}")
        if exemplar_variant not in VARIANTS:
            raise ValueError(f"unknown exemplar variant {exemplar_variant!r}")
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = "0"
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(MUGGLED_SAM_SOURCE), environment["PYTHONPATH"]]
            if environment.get("PYTHONPATH")
            else [str(MUGGLED_SAM_SOURCE)]
        )
        argv = [
            str(external_python or MUGGLED_SAM_PYTHON),
            str(Path(__file__).with_name("sam3_appearance.py")),
            "serve-jsonl",
            "--proxy",
            str(proxy_path),
            "--references",
            str(references),
            "--model",
            str((model or DEFAULT_MODEL).resolve()),
            "--results-directory",
            str(results_directory),
            "--device",
            device,
            "--candidate-source",
            candidate_source,
            "--exemplar-set",
            exemplar_set,
            "--exemplar-variant",
            exemplar_variant,
            "--top-k",
            str(int(top_k)),
        ]
        for pid in allow_gpu_neighbours:
            argv += ["--allow-gpu-neighbour", str(int(pid))]
        self.argv = argv
        self.candidate_source = candidate_source
        self._client = WorkerClient(argv, environment=environment, stderr_path=stderr_path)

    def request(
        self, command: str, payload: dict[str, Any], *, timeout: float = 600
    ) -> dict[str, Any]:
        return self._client.request(command, payload, timeout=timeout)

    def close(self) -> None:
        self._client.close()


def record_candidate_source(
    iteration_dir: Path,
    *,
    candidate_source: str,
    references: Path | None,
    exemplar_set: str,
    exemplar_variant: str,
    repository_root: Path,
) -> Path:
    """Sidecar beside a decode: which candidate source produced the decisions.

    The decision and provenance schemas are human contracts hashed into schedules, so the
    source is recorded here rather than on them; `decoder_iou_estimate` on the decisions is the
    detection score whenever the source is not the image decoder.
    """
    payload = {
        "candidate_source": candidate_source,
        "exemplar_references": (
            _fingerprint(references, repository_root) if references is not None else None
        ),
        "exemplar_set": exemplar_set,
        "exemplar_variant": exemplar_variant,
        "score_semantics": (
            "decoder IoU estimate"
            if candidate_source == "image_decoder"
            else "SAM3 detection score scaled by the presence score (candidates with api "
            "muggledsam_sam3_exemplar_detector); image-decoder candidates keep their IoU "
            "estimate under `both`"
        ),
        "recorded_at": datetime.now(UTC).isoformat(),
    }
    path = iteration_dir / "candidate_source.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def decoder_for_source(
    candidate_source: str,
    *,
    proxy_path: Path,
    results_directory: Path,
    stderr_path: Path,
    references: Path | None,
    external_python: Path | None = None,
    model: Path | None = None,
    device: str = "cuda:0",
    exemplar_set: str = "same_view",
    exemplar_variant: str = "posneg",
    allow_gpu_neighbours: Iterable[int] = (),
) -> Any:
    """The decoder object `multiview_reprompt.decode_plan` should use for a candidate source.

    `image_decoder` returns None so the caller keeps its own calibration worker (byte-identical
    control); the other two start the exemplar server, which holds the image decoder too when
    `both` is asked for, so one GPU process serves the decode either way.
    """
    if candidate_source not in CANDIDATE_SOURCES:
        raise ValueError(
            f"unknown candidate source {candidate_source!r}; one of {CANDIDATE_SOURCES}"
        )
    if candidate_source == "image_decoder":
        return None
    if references is None:
        raise ValueError(f"--candidate-source {candidate_source} needs --exemplar-references")
    return ExemplarDecoderClient(
        proxy_path=proxy_path,
        references=references,
        results_directory=results_directory,
        stderr_path=stderr_path,
        candidate_source=candidate_source,
        exemplar_set=exemplar_set,
        exemplar_variant=exemplar_variant,
        external_python=external_python,
        model=model,
        device=device,
        allow_gpu_neighbours=allow_gpu_neighbours,
    )


# ------------------------------------------------------------------------ pool comparison


@dataclass(frozen=True)
class PoolCandidate:
    """One candidate mask for one truth cell with everything the acceptance filters need."""

    pool: str
    key: str
    score: float
    area_px: int
    centroid_px: tuple[float, float] | None
    iou_vs_truth: float
    inside_box: bool | None = None


@dataclass(frozen=True)
class PoolCell:
    frame: int
    part: str
    truth_area_px: int
    expected_area_px: float | None
    projected_radius_px: float | None
    centroid_proxy_px: tuple[float, float] | None
    candidates: tuple[PoolCandidate, ...]
    reference_frames_used: tuple[int, ...] = ()


def _load_kept_rows(pass_dir: Path) -> list[dict[str, Any]]:
    rows = []
    with (pass_dir / "detections_kept.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def exemplar_candidates_for_cell(
    pass_dir: Path,
    rows: Sequence[dict[str, Any]],
    *,
    frame: int,
    part: str,
    set_name: str,
    variant: str,
    truth_mask: np.ndarray,
    gate_box: tuple[float, float, float, float] | None,
) -> tuple[list[PoolCandidate], tuple[int, ...]]:
    """Kept detections of one (frame, part, set, variant) as candidates against a truth mask."""
    out: list[PoolCandidate] = []
    used: tuple[int, ...] = ()
    for row in rows:
        if (row["frame"], row["target"], row["set"], row["variant"]) != (
            frame,
            part,
            set_name,
            variant,
        ):
            continue
        used = tuple(int(f) for f in row["reference_frames_used"])
        for item in row["masks"]:
            mask = mask_cache.decode_mask_png(pass_dir / item["uri"])
            iou = mask_iou(mask, truth_mask)
            centroid = tuple(item["centroid_px"]) if item.get("centroid_px") else None
            inside = None
            if gate_box is not None:
                inside = bool(
                    centroid is not None
                    and gate_box[0] <= centroid[0] < gate_box[2]
                    and gate_box[1] <= centroid[1] < gate_box[3]
                )
            out.append(
                PoolCandidate(
                    pool=f"exemplar:{set_name}:{variant}",
                    key=f"{set_name}:{variant}:{item['index']}",
                    score=float(item["score"]),
                    area_px=int(item["area_px"]),
                    centroid_px=centroid,  # type: ignore[arg-type]
                    iou_vs_truth=float(iou if iou is not None else 0.0),
                    inside_box=inside,
                )
            )
    return out, used


def accept_from_pool(
    candidates: Sequence[PoolCandidate],
    *,
    expected_area_px: float | None,
    centroid_proxy_px: tuple[float, float] | None,
    projected_radius_px: float | None,
    area_band: tuple[float, float] = (0.3, 3.0),
    max_radii: float = 1.5,
    min_score: float = 0.0,
    require_inside_box: bool = True,
) -> PoolCandidate | None:
    """The tool's filters in the image plane, then the highest score wins.

    Area within `area_band` x the consensus expected area; centroid within `max_radii` x the
    projected radius of the consensus centroid (the planar stand-in for the ray rule, which
    needs the rig); optionally inside the gate box; then rank by score.
    """
    passing = []
    for c in candidates:
        if c.area_px == 0:
            continue
        if require_inside_box and c.inside_box is False:
            continue
        if c.score < min_score:
            continue
        if expected_area_px:
            ratio = c.area_px / expected_area_px
            if not area_band[0] <= ratio <= area_band[1]:
                continue
        if centroid_proxy_px is not None and projected_radius_px and c.centroid_px is not None:
            distance = float(
                np.hypot(
                    c.centroid_px[0] - centroid_proxy_px[0], c.centroid_px[1] - centroid_proxy_px[1]
                )
            )
            if distance > max_radii * projected_radius_px:
                continue
        passing.append(c)
    if not passing:
        return None
    return max(passing, key=lambda c: (c.score, -c.area_px))


def pool_metrics(
    cells: Sequence[PoolCell], pool_prefix: str, *, min_score: float = 0.0
) -> dict[str, Any]:
    """Oracle, accepted mean IoU, acceptance and harm for one pool over a set of cells."""
    oracle = []
    accepted_ious = []
    harm = 0
    accepted = 0
    for cell in cells:
        pool = [c for c in cell.candidates if c.pool.startswith(pool_prefix)]
        oracle.append(max((c.iou_vs_truth for c in pool), default=0.0))
        choice = accept_from_pool(
            pool,
            expected_area_px=cell.expected_area_px,
            centroid_proxy_px=cell.centroid_proxy_px,
            projected_radius_px=cell.projected_radius_px,
            min_score=min_score,
        )
        if choice is not None:
            accepted += 1
            accepted_ious.append(choice.iou_vs_truth)
            if choice.iou_vs_truth < HARM_IOU:
                harm += 1
    n = len(cells)
    return {
        "cells": n,
        "oracle_mean_iou": float(np.mean(oracle)) if oracle else None,
        "oracle_cells_at_least_0.6": int(sum(1 for o in oracle if o >= BAR_IOU)),
        "accepted": accepted,
        "acceptance_rate": accepted / n if n else None,
        "accepted_mean_iou": float(np.mean(accepted_ious)) if accepted_ious else None,
        "harm": harm,
        "harm_rate": harm / accepted if accepted else None,
        "harm_rate_over_cells": harm / n if n else None,
        "min_score": min_score,
    }


def leave_frames_out_pool(
    cells: Sequence[PoolCell],
    pool_prefix: str,
    *,
    frames: Sequence[int],
    score_floors: Sequence[float] = (0.0, 0.2, 0.4, 0.6),
    held_out: int = 3,
) -> dict[str, Any]:
    """Pick the score floor on all but `held_out` frames (rotated), score the held-out cells.

    Fit objective: highest accepted mean IoU among floors with fit harm rate <= 0.15 and fit
    acceptance >= 0.25 (the acceptance search's rule); the pooled held-out cells give the
    reported held-out IoU / acceptance / harm.
    """
    frames = sorted(set(frames))
    if len(frames) <= held_out:
        return {"folds": 0, "note": "too few frames for leave-frames-out"}
    held_ious: list[float] = []
    held_cells = 0
    held_accepted = 0
    held_harm = 0
    chosen: list[float] = []
    for start in range(0, len(frames), held_out):
        test_frames = set(frames[start : start + held_out])
        fit = [c for c in cells if c.frame not in test_frames]
        test = [c for c in cells if c.frame in test_frames]
        if not fit or not test:
            continue
        best_floor = None
        best_value = -1.0
        for floor in score_floors:
            m = pool_metrics(fit, pool_prefix, min_score=floor)
            if m["accepted"] == 0 or (m["acceptance_rate"] or 0) < 0.25:
                continue
            if (m["harm_rate"] or 0) > BAR_HARM:
                continue
            if (m["accepted_mean_iou"] or 0) > best_value:
                best_value = m["accepted_mean_iou"] or 0
                best_floor = floor
        if best_floor is None:
            best_floor = 0.0
        chosen.append(best_floor)
        held_cells += len(test)
        for cell in test:
            pool = [c for c in cell.candidates if c.pool.startswith(pool_prefix)]
            choice = accept_from_pool(
                pool,
                expected_area_px=cell.expected_area_px,
                centroid_proxy_px=cell.centroid_proxy_px,
                projected_radius_px=cell.projected_radius_px,
                min_score=best_floor,
            )
            if choice is not None:
                held_accepted += 1
                held_ious.append(choice.iou_vs_truth)
                if choice.iou_vs_truth < HARM_IOU:
                    held_harm += 1
    return {
        "folds": len(chosen),
        "held_out_cells": held_cells,
        "held_out_accepted": held_accepted,
        "held_out_mean_iou": float(np.mean(held_ious)) if held_ious else None,
        "held_out_acceptance_rate": held_accepted / held_cells if held_cells else None,
        "held_out_harm": held_harm,
        "held_out_harm_rate": held_harm / held_accepted if held_accepted else None,
        "chosen_floors": chosen,
    }


def meets_bar(metrics: Mapping[str, Any]) -> bool:
    iou = metrics.get("held_out_mean_iou")
    harm = metrics.get("held_out_harm_rate")
    return iou is not None and iou >= BAR_IOU and (harm is None or harm <= BAR_HARM)


# --------------------------------------------------------------------------- truth plans


def build_truth_plan(
    repository_root: Path,
    *,
    view: str,
    view_id: str,
    anchors: Path,
    consensus_root: Path,
    output_dir: Path,
    extra_positives: Sequence[tuple[int, str, Path]] = (),
    seed_run: Path | None = None,
) -> Path:
    """An `AcceptanceSearchPlan` for any view: every human anchor cell as a re-prompt onset.

    The acceptance-search module builds this for C10379 from the seed-search truth set; this
    is the same construction (others-only consensus sphere -> two boxes with margins
    0.25 / 0.60 and other-part negatives) for a view's anchor mask set, so C10119 gets the same
    harness. No hand variants: the pools compared here are the image decoder's base pool and
    the exemplar pool.
    """
    from .assembly101_clock_offset import is_ego
    from .correction_acceptance_search import (
        BOX_MARGINS,
        CLAIM_BOUNDARIES,
        AcceptanceSearchPlan,
        CellGeometry,
        FramePlan,
        TruthCell,
    )
    from .multiview_consensus import load_consensus, load_view_run
    from .multiview_geometry import CameraRig
    from .multiview_reprompt import build_prompts, onset_geometry
    from .multiview_seed_transfer import proxy_focal_px
    from .seed_search import proxy_for_view

    repository_root = repository_root.resolve()
    mask_set, anchors_root = load_mask_set(anchors, view_id=view_id)
    proxy, (width, height) = proxy_for_view(repository_root, view)
    shape = (height, width)
    rig = CameraRig.load(repository_root)
    consensus_dir = (repository_root / consensus_root).resolve()
    consensus = load_consensus(consensus_dir / "manifest.json")
    if any(s.view == view for s in consensus.sources):
        raise ValueError(f"the consensus {consensus_dir.name} includes {view}; use an excl build")
    with np.load(consensus_dir / "consensus_points.npz") as archive:
        views = tuple(str(v) for v in archive["views"])
        points = {t: np.asarray(archive[f"consensus/{t}"])[:FIRST_MINUTE] for t in TARGETS}
        used = {
            t: {v: np.asarray(archive[f"used/{t}/{v}"], dtype=bool)[:FIRST_MINUTE] for v in views}
            for t in TARGETS
        }
    runs = {
        s.view: load_view_run(
            repository_root,
            repository_root / s.run_directory_uri,
            view=s.view,
            frame_count=FIRST_MINUTE,
        )
        for s in consensus.sources
    }
    seed_area: dict[str, int | None] = {}
    if seed_run is not None:
        masks0 = read_run_masks(seed_run, 1).get(0, {})
        for part in TARGETS:
            path = masks0.get(part)
            seed_area[part] = int(mask_cache.decode_mask_png(Path(path)).sum()) if path else None
    focal = proxy_focal_px(rig, view)

    cells: dict[tuple[int, str], dict[str, Any]] = {}
    for anchor in mask_set.anchors:
        if anchor.state == "labeled" and anchor.mask_uri is not None:
            cells[(anchor.analysis_frame_index, anchor.target)] = {
                "role": "positive",
                "source": "anchors",
                "mask": anchors_root / anchor.mask_uri,
                "area": anchor.area_pixels,
            }
        elif anchor.state == "hidden":
            cells[(anchor.analysis_frame_index, anchor.target)] = {
                "role": "hidden",
                "source": "anchors",
                "mask": None,
                "area": None,
            }
    for frame, part, mask_path in extra_positives:
        cells.setdefault(
            (frame, part),
            {
                "role": "positive",
                "source": "corrections",
                "mask": mask_path,
                "area": int(mask_cache.decode_mask_png(mask_path).sum()),
            },
        )
    frames = sorted({frame for frame, _ in cells})
    frame_plans = []
    prompt_count = 0
    unprompted = 0
    for frame in frames:
        geometry: dict[str, CellGeometry] = {}
        for part in TARGETS:
            point = points[part][frame]
            if np.isnan(point).any():
                continue
            masks = {}
            poses = {}
            for other_view in views:
                if not used[part][other_view][frame]:
                    continue
                mask = runs[other_view].mask(frame, part)
                if mask is not None:
                    masks[other_view] = mask
                    poses[other_view] = (
                        rig.pose_frame(other_view, frame) if is_ego(other_view) else None
                    )
            g = onset_geometry(
                rig,
                target_view=view,
                point_world_mm=point,
                masks_by_view=masks,
                pose_frames=poses,
                target_pose_frame=rig.pose_frame(view, frame) if is_ego(view) else None,
            )
            if g is None or np.isnan(g.centroid_proxy_px).any():
                continue
            geometry[part] = CellGeometry(
                part=part,
                consensus_world_mm=tuple(float(v) for v in point),
                radius_mm=g.radius_mm,
                depth_mm=g.depth_mm,
                expected_area_px=g.expected_area_px,
                centroid_proxy_px=(float(g.centroid_proxy_px[0]), float(g.centroid_proxy_px[1])),
                projected_radius_px=float(g.radius_mm * focal / g.depth_mm),
                views_used=tuple(masks),
            )
        counter = 1
        frame_cells = []
        for part in TARGETS:
            entry = cells.get((frame, part))
            if entry is None:
                continue
            common = dict(
                frame=frame,
                part=part,
                role=entry["role"],
                source=entry["source"],
                mask=(
                    fingerprint(entry["mask"], repository_root)
                    if entry["mask"] is not None
                    else None
                ),
                truth_area_px=entry["area"],
                seed_area_px=seed_area.get(part),
            )
            g = geometry.get(part)
            if g is None:
                frame_cells.append(
                    TruthCell(
                        **common,
                        unprompted_reason=(
                            f"no consensus for {part} at this frame in {consensus_dir.name}"
                        ),
                    )
                )
                unprompted += 1
                continue
            centroid = np.asarray(g.centroid_proxy_px)
            others = {o: np.asarray(geometry[o].centroid_proxy_px) for o in geometry if o != part}
            prompts = build_prompts(
                target=part,
                frame=frame,
                centroid_proxy_px=centroid,
                half_size_px=g.projected_radius_px,
                shape=shape,
                other_centroids=others,
                hand_joints=None,
                margins=BOX_MARGINS,
                counter_start=counter,
            )
            counter += len(prompts)
            prompt_count += len(prompts)
            frame_cells.append(TruthCell(**common, prompts=tuple(prompts)))
        frame_plans.append(
            FramePlan(
                frame=frame,
                geometry=geometry,
                hands_dataset_px={},
                hands_wilor_px={},
                cells=tuple(frame_cells),
            )
        )
    anchors_json = anchors_root / "anchors" / "anchor_masks.json"
    plan = AcceptanceSearchPlan(
        manifest_kind="correction_acceptance_search_plan",
        target_view=view,
        proxy=proxy,
        proxy_dimensions=(width, height),
        consensus_root_uri=relative_uri(consensus_dir, repository_root),
        consensus_manifest=fingerprint(consensus_dir / "manifest.json", repository_root),
        consensus_points=fingerprint(consensus_dir / "consensus_points.npz", repository_root),
        truth_set=fingerprint(anchors_json, repository_root),
        wilor_observations=fingerprint(anchors_json, repository_root),
        seed_run_uri=relative_uri(seed_run, repository_root) if seed_run else "",
        box_margins=tuple(float(m) for m in BOX_MARGINS),
        hand_hull_dilation_px=0,
        frames=tuple(frame_plans),
        prompt_count=prompt_count,
        cell_count=len(cells),
        unprompted_cells=unprompted,
        anchor_frames=tuple(mask_set.frames),
        always_fit_frames=tuple(sorted({f for (f, _, _) in extra_positives})),
        generated_at=datetime.now(UTC),
        claim_boundaries=(
            *CLAIM_BOUNDARIES,
            f"Truth cells here are the {view} human review anchors ({mask_set.counts}); "
            "wilor_observations points at the anchor set because this plan carries no hand "
            "variants.",
        ),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "plan.json"
    path.write_text(plan.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def c10379_correction_positives(repository_root: Path) -> list[tuple[int, str, Path]]:
    """The human correction masks at 0 / 327 / 900 / 1235 (the acceptance search's extra truth)."""
    references, _ = c10379_human_references(repository_root)
    return [(r["frame"], r["target"], Path(r["mask_path"])) for r in references]


def decode_control(
    repository_root: Path, plan_path: Path, *, output_dir: Path, device: str = "cuda:0"
) -> Path:
    """The image-decoder control pool for a truth plan (GPU; the calibration worker)."""
    from .correction_acceptance_search import load_plan, run_decode

    plan = load_plan(plan_path)
    return run_decode(repository_root, plan, output_dir=output_dir, device=device)


# ---------------------------------------------------------------------------- pool compare


def _gate_box(prompts: Sequence[Any]) -> tuple[float, float, float, float] | None:
    boxes = [p.pixel_box for p in prompts]
    if not boxes:
        return None
    return (
        float(min(b.x1 for b in boxes)),
        float(min(b.y1 for b in boxes)),
        float(max(b.x2 for b in boxes)),
        float(max(b.y2 for b in boxes)),
    )


def load_pool_cells(
    repository_root: Path,
    *,
    plan_path: Path,
    decode_dir: Path | None,
    pass_dir: Path,
    exemplar_sets: Sequence[str] = ("same_view", "cross_view"),
) -> list[PoolCell]:
    """Every prompted positive truth cell with its decoder-pool and exemplar-pool candidates."""
    from .correction_acceptance_search import load_plan

    plan = load_plan(plan_path)
    decoded = (
        json.loads((decode_dir / "decode_result.json").read_text(encoding="utf-8"))["decoded"]
        if decode_dir is not None
        else {}
    )
    rows = _load_kept_rows(pass_dir)
    cells: list[PoolCell] = []
    for frame_plan in plan.frames:
        for cell in frame_plan.cells:
            if cell.role != "positive" or cell.mask is None or not cell.prompts:
                continue
            truth = mask_cache.decode_mask_png(repository_root / cell.mask.uri)
            g = frame_plan.geometry[cell.part]
            gate = _gate_box(cell.prompts)
            candidates: list[PoolCandidate] = []
            for prompt in cell.prompts:
                result = decoded.get(prompt.prompt_id)
                if result is None:
                    continue
                for option in result["candidates"]:
                    mask = mask_cache.decode_mask_png(decode_dir / option["mask_uri"])  # type: ignore[operator]
                    centroid = None
                    ys, xs = np.nonzero(mask)
                    if ys.size:
                        centroid = (float(xs.mean()), float(ys.mean()))
                    iou_value = mask_iou(mask, truth)
                    candidates.append(
                        PoolCandidate(
                            pool="decoder:base",
                            key=f"{prompt.prompt_id}:{option['candidate_index']}",
                            score=float(option["iou_score"]),
                            area_px=int(mask.sum()),
                            centroid_px=centroid,
                            iou_vs_truth=float(iou_value if iou_value is not None else 0.0),
                            inside_box=(
                                gate is not None
                                and centroid is not None
                                and gate[0] <= centroid[0] < gate[2]
                                and gate[1] <= centroid[1] < gate[3]
                            ),
                        )
                    )
            used_frames: tuple[int, ...] = ()
            for set_name in exemplar_sets:
                for variant in VARIANTS:
                    more, used = exemplar_candidates_for_cell(
                        pass_dir,
                        rows,
                        frame=frame_plan.frame,
                        part=cell.part,
                        set_name=set_name,
                        variant=variant,
                        truth_mask=truth,
                        gate_box=gate,
                    )
                    candidates.extend(more)
                    if set_name == "same_view" and used:
                        used_frames = used
            cells.append(
                PoolCell(
                    frame=frame_plan.frame,
                    part=cell.part,
                    truth_area_px=int(cell.truth_area_px or truth.sum()),
                    expected_area_px=g.expected_area_px,
                    projected_radius_px=g.projected_radius_px,
                    centroid_proxy_px=g.centroid_proxy_px,
                    candidates=tuple(candidates),
                    reference_frames_used=used_frames,
                )
            )
    return cells


def compare_pools(
    cells: Sequence[PoolCell],
    *,
    pools: Sequence[str],
    anchor_frames: Sequence[int],
) -> dict[str, Any]:
    """Per part and pool: oracle, current-rule metrics, and leave-frames-out with a score floor."""
    parts = tuple(dict.fromkeys(c.part for c in cells))
    report: dict[str, Any] = {"parts": {}, "pools": list(pools), "cells": len(cells)}
    for part in parts:
        part_cells = [c for c in cells if c.part == part]
        frames = sorted({c.frame for c in part_cells if c.frame in set(anchor_frames)})
        entry: dict[str, Any] = {"cells": len(part_cells), "pools": {}}
        for pool in pools:
            in_sample = pool_metrics(part_cells, pool)
            held = leave_frames_out_pool(part_cells, pool, frames=frames)
            entry["pools"][pool] = {
                "in_sample": in_sample,
                "leave_frames_out": held,
                "meets_bar": meets_bar(held),
                "per_cell": [
                    {
                        "frame": c.frame,
                        "oracle": max(
                            (x.iou_vs_truth for x in c.candidates if x.pool.startswith(pool)),
                            default=0.0,
                        ),
                        "accepted_iou": (
                            (lambda ch: ch.iou_vs_truth if ch else None)(
                                accept_from_pool(
                                    [x for x in c.candidates if x.pool.startswith(pool)],
                                    expected_area_px=c.expected_area_px,
                                    centroid_proxy_px=c.centroid_proxy_px,
                                    projected_radius_px=c.projected_radius_px,
                                )
                            )
                        ),
                        "candidates": sum(1 for x in c.candidates if x.pool.startswith(pool)),
                    }
                    for c in part_cells
                ],
            }
        report["parts"][part] = entry
    return report


def compare_table(report: Mapping[str, Any], *, title: str) -> str:
    lines = [
        f"### {title}",
        "",
        "| part | pool | cells | oracle mean IoU | cells with a >= 0.6 candidate | accepted | "
        "accepted mean IoU | harm (accepted < 0.4) | held-out mean IoU | held-out acceptance | "
        "held-out harm | bar (>= 0.6, harm <= 0.15) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def f(value: Any, digits: int = 3) -> str:
        return "-" if value is None else f"{value:.{digits}f}"

    for part, entry in report["parts"].items():
        for pool, block in entry["pools"].items():
            s = block["in_sample"]
            h = block["leave_frames_out"]
            lines.append(
                f"| {part} | `{pool}` | {s['cells']} | {f(s['oracle_mean_iou'])} | "
                f"{s['oracle_cells_at_least_0.6']} / {s['cells']} | "
                f"{s['accepted']} / {s['cells']} | "
                f"{f(s['accepted_mean_iou'])} | {s['harm']} / {s['accepted']} "
                f"({f(s['harm_rate'], 2)}) | {f(h.get('held_out_mean_iou'))} | "
                f"{f(h.get('held_out_acceptance_rate'), 2)} | "
                f"{h.get('held_out_harm', '-')} / {h.get('held_out_accepted', '-')} "
                f"({f(h.get('held_out_harm_rate'), 2)}) | {'yes' if block['meets_bar'] else 'no'} |"
            )
    lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------------------ zero-shot


def rec2_reference_spec(
    rec1_spec: Path, *, target_view: str, repository_root: Path
) -> dict[str, Any]:
    """Recording 1's C10379 human masks as the only (cross-view) reference set for recording 2."""
    other = json.loads(rec1_spec.read_text(encoding="utf-8"))
    same = next(s for s in other["sets"] if s["name"] == "same_view")
    return {
        "schema": REFERENCE_SCHEMA,
        "view": target_view,
        "sets": [{**same, "name": "cross_view", "distractors": []}],
        "sources": {"rec1_spec": _fingerprint(rec1_spec, repository_root)},
        "notes": [
            "zero-shot: recording-1 exemplars applied to recording 2; no recording-2 mask is a "
            "reference, so nothing here is leave-reference-out"
        ],
        "generated_at": datetime.now(UTC).isoformat(),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def human_cells_from_calibration(
    manifest_path: Path,
) -> tuple[list[dict[str, Any]], list[tuple[int, str]]]:
    """Accepted human masks (frame, target, mask path) and hidden marks of a calibration workspace."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    workspace = manifest_path.parent
    accepted = []
    for candidate in manifest["candidates"]:
        if not candidate.get("human_accepted") or candidate.get("rejected"):
            continue
        if candidate.get("selected_by") != "human":
            continue
        index = int(candidate["human_selected_candidate_index"])
        option = next(
            o
            for o in candidate["decoder_result"]["candidates"]
            if int(o["candidate_index"]) == index
        )
        accepted.append(
            {
                "frame": int(candidate["frame"]["analysis_frame_index"]),
                "target": str(candidate["intended_target"]),
                "mask_path": str((workspace / option["mask_uri"]).resolve()),
                "candidate_id": candidate["candidate_id"],
            }
        )
    hidden = sorted(
        (int(h["frame"]["analysis_frame_index"]), str(h["intended_target"]))
        for h in manifest.get("hidden_targets", ())
    )
    return accepted, hidden


def score_zero_shot(
    *,
    pass_dir: Path,
    calibration_manifest: Path,
    set_name: str = "cross_view",
    repository_root: Path,
) -> dict[str, Any]:
    """IoU of the exemplar detections against recording 2's human masks; presence at hidden cells."""
    accepted, hidden = human_cells_from_calibration(calibration_manifest)
    rows = _load_kept_rows(pass_dir)
    detections = {}
    with (pass_dir / "detections.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                detections[(row["frame"], row["target"], row["set"], row["variant"])] = row
    cells = []
    for cell in accepted:
        truth = mask_cache.decode_mask_png(Path(cell["mask_path"]))
        entry: dict[str, Any] = {
            **cell,
            "human_area_px": int(truth.sum()),
            "variants": {},
        }
        for variant in VARIANTS:
            candidates, used = exemplar_candidates_for_cell(
                pass_dir,
                rows,
                frame=cell["frame"],
                part=cell["target"],
                set_name=set_name,
                variant=variant,
                truth_mask=truth,
                gate_box=None,
            )
            row = detections.get((cell["frame"], cell["target"], set_name, variant), {})
            top = max(candidates, key=lambda c: c.score, default=None)
            best = max(candidates, key=lambda c: c.iou_vs_truth, default=None)
            entry["variants"][variant] = {
                "presence": row.get("presence"),
                "top_score": top.score if top else None,
                "top_iou_vs_human": top.iou_vs_truth if top else None,
                "top_area_px": top.area_px if top else None,
                "best_iou_vs_human": best.iou_vs_truth if best else None,
                "best_score": best.score if best else None,
                "candidates": len(candidates),
                "reference_frames_used": list(used),
            }
        cells.append(entry)
    hidden_rows = []
    for frame, target in hidden:
        for variant in VARIANTS:
            row = detections.get((frame, target, set_name, variant))
            if row is None:
                continue
            hidden_rows.append(
                {
                    "frame": frame,
                    "target": target,
                    "variant": variant,
                    "presence": row.get("presence"),
                    "top_score": row.get("top_score"),
                    "top_area_px": row.get("top_area_px"),
                }
            )
    summary: dict[str, Any] = {}
    for variant in VARIANTS:
        top_ious = [
            c["variants"][variant]["top_iou_vs_human"]
            for c in cells
            if c["variants"][variant]["top_iou_vs_human"] is not None
        ]
        best_ious = [
            c["variants"][variant]["best_iou_vs_human"]
            for c in cells
            if c["variants"][variant]["best_iou_vs_human"] is not None
        ]
        presence_visible = [
            c["variants"][variant]["presence"]
            for c in cells
            if c["variants"][variant]["presence"] is not None
        ]
        presence_hidden = [
            h["presence"]
            for h in hidden_rows
            if h["variant"] == variant and h["presence"] is not None
        ]
        summary[variant] = {
            "cells": len(cells),
            "top_mean_iou": float(np.mean(top_ious)) if top_ious else None,
            "best_mean_iou": float(np.mean(best_ious)) if best_ious else None,
            "top_at_least_0.5": int(sum(1 for v in top_ious if v >= 0.5)),
            "presence_visible_mean": float(np.mean(presence_visible)) if presence_visible else None,
            "presence_hidden_mean": float(np.mean(presence_hidden)) if presence_hidden else None,
            "presence_hidden_cells": len(presence_hidden),
        }
    return {
        "manifest_kind": "exemplar_zero_shot",
        "pass": relative_uri(pass_dir, repository_root),
        "calibration_manifest": _fingerprint(calibration_manifest, repository_root),
        "set": set_name,
        "accepted_cells": cells,
        "hidden_cells": hidden_rows,
        "summary": summary,
        "generated_at": datetime.now(UTC).isoformat(),
        "claim_boundary": (
            f"{len(cells)} human-accepted masks and {len(hidden)} hidden marks on recording 2 so "
            "far (a calibration in progress): review evidence, not ground truth; the exemplars are "
            "recording 1's human masks, so this is zero-shot on the new recording."
        ),
    }


def zero_shot_table(report: Mapping[str, Any]) -> str:
    lines = [
        "| frame | part | human px | variant | presence | top score | top IoU vs human | top px | best IoU in top-K |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    def f(v: Any, d: int = 3) -> str:
        return "-" if v is None else f"{v:.{d}f}"

    for cell in report["accepted_cells"]:
        for variant, v in cell["variants"].items():
            lines.append(
                f"| {cell['frame']} | {cell['target']} | {cell['human_area_px']} | {variant} | "
                f"{f(v['presence'])} | {f(v['top_score'])} | {f(v['top_iou_vs_human'])} | "
                f"{v['top_area_px'] if v['top_area_px'] is not None else '-'} | {f(v['best_iou_vs_human'])} |"
            )
    lines += [
        "",
        "| hidden cell | variant | presence | top score | top px |",
        "|---|---|---|---|---|",
    ]
    for h in report["hidden_cells"]:
        lines.append(
            f"| {h['frame']} {h['target']} | {h['variant']} | {f(h['presence'])} | {f(h['top_score'])} | "
            f"{h['top_area_px'] if h['top_area_px'] is not None else '-'} |"
        )
    lines += [
        "",
        "| variant | cells | mean top IoU | mean best IoU | top >= 0.5 | presence visible | presence hidden (n) |",
        "|---|---|---|---|---|---|---|",
    ]
    for variant, s in report["summary"].items():
        lines.append(
            f"| {variant} | {s['cells']} | {f(s['top_mean_iou'])} | {f(s['best_mean_iou'])} | "
            f"{s['top_at_least_0.5']} / {s['cells']} | {f(s['presence_visible_mean'])} | "
            f"{f(s['presence_hidden_mean'])} ({s['presence_hidden_cells']}) |"
        )
    lines.append("")
    lines.append(report["claim_boundary"])
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------- CLI


def _references_main(args: argparse.Namespace, root: Path) -> None:
    _, run_directory = resolve_run_directory(str(args.run))
    spec = build_reference_spec(
        view=args.view,
        run_directory=run_directory,
        repository_root=root,
        cross_view_spec=args.cross_view_from.resolve() if args.cross_view_from else None,
        include_distractor=not args.no_distractor,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    counts = {s["name"]: len(s["references"]) for s in spec["sets"]}
    print(f"Reference spec: {args.output} ({counts}; notes: {spec['notes']})")


def _spawn_pass_argv(args: argparse.Namespace) -> list[str]:
    from .muggled_smoke import MUGGLED_SAM_PYTHON

    argv = [
        str(args.external_python or MUGGLED_SAM_PYTHON),
        str(Path(__file__).with_name("sam3_appearance.py")),
        "pass",
        "--proxy",
        str(args.proxy),
        "--references",
        str(args.references),
        "--run",
        str(args.run),
        "--output",
        str(args.output),
        "--frame-count",
        str(args.frame_count),
        "--max-side-length",
        str(args.max_side_length),
        "--top-k",
        str(args.top_k),
    ]
    for frame in args.keep_frame or []:
        argv += ["--keep-frame", str(frame)]
    if args.frames:
        argv += ["--frames", *[str(f) for f in args.frames]]
    for pid in args.allow_gpu_neighbour or []:
        argv += ["--allow-gpu-neighbour", str(pid)]
    return argv


def _pass_main(args: argparse.Namespace, root: Path) -> None:
    """Run the GPU pass under the MuggledSAM interpreter (the queue calls this)."""
    from .muggled_smoke import MUGGLED_SAM_SOURCE

    _, run_directory = resolve_run_directory(str(args.run))
    args.run = run_directory
    if args.proxy is None:
        args.proxy = proxy_for_run(run_directory)
    argv = _spawn_pass_argv(args)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(MUGGLED_SAM_SOURCE), environment["PYTHONPATH"]]
        if environment.get("PYTHONPATH")
        else [str(MUGGLED_SAM_SOURCE)]
    )
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / "pass_command.json").write_text(
        json.dumps({"argv": argv, "cwd": str(root)}, indent=2) + "\n", encoding="utf-8"
    )
    with (Path(args.output) / "pass.stderr.log").open("a", encoding="utf-8") as stderr:
        completed = subprocess.run(argv, cwd=root, env=environment, stderr=stderr, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    references = commands.add_parser("references", help="write the reference spec for one view")
    references.add_argument("--view", choices=sorted(VIEW_IDS), required=True)
    references.add_argument("--run", required=True, help="the tracked run (frame-0 seeds, proxy)")
    references.add_argument("--output", type=Path, required=True)
    references.add_argument(
        "--cross-view-from", type=Path, default=None, help="another view's spec"
    )
    references.add_argument("--no-distractor", action="store_true")

    run_pass = commands.add_parser(
        "pass", help="run sam3_appearance.py pass under the MuggledSAM interpreter"
    )
    run_pass.add_argument("--run", required=True)
    run_pass.add_argument("--references", type=Path, required=True)
    run_pass.add_argument("--output", type=Path, required=True)
    run_pass.add_argument("--proxy", type=Path, default=None)
    run_pass.add_argument("--frame-count", type=int, default=FIRST_MINUTE)
    run_pass.add_argument("--frames", type=int, nargs="*", default=None)
    run_pass.add_argument("--keep-frame", type=int, action="append", default=None)
    run_pass.add_argument("--max-side-length", type=int, default=1280)
    run_pass.add_argument("--top-k", type=int, default=10)
    run_pass.add_argument("--external-python", type=Path, default=None)
    run_pass.add_argument("--allow-gpu-neighbour", type=int, action="append", default=None)

    truth_plan = commands.add_parser(
        "truth-plan", help="every human anchor cell of a view as a re-prompt onset (CPU)"
    )
    truth_plan.add_argument("--view", choices=sorted(VIEW_IDS), required=True)
    truth_plan.add_argument(
        "--anchors", type=Path, default=None, help="anchor workspace (default: the view's)"
    )
    truth_plan.add_argument(
        "--consensus-root", type=Path, required=True, help="a consensus built without the view"
    )
    truth_plan.add_argument(
        "--seed-run", default=None, help="run whose frame-0 masks give seed areas"
    )
    truth_plan.add_argument("--output-dir", type=Path, required=True)
    truth_plan.add_argument(
        "--with-c10379-corrections",
        action="store_true",
        help="add the human correction masks at 0/327/900/1235 as truth cells (C10379 only)",
    )

    decode = commands.add_parser(
        "decode-control", help="image-decoder control pool for a truth plan (GPU)"
    )
    decode.add_argument("--plan", type=Path, required=True)
    decode.add_argument("--output-dir", type=Path, required=True)
    decode.add_argument("--device", default="cuda:0")
    decode.add_argument("--allow-gpu-neighbour", type=int, action="append", default=None)

    compare = commands.add_parser(
        "compare", help="exemplar pool vs decoder pool on the truth cells (CPU)"
    )
    compare.add_argument("--plan", type=Path, required=True)
    compare.add_argument(
        "--decode-dir", type=Path, default=None, help="decode-control output (the control pool)"
    )
    compare.add_argument(
        "--pass",
        dest="pass_dir",
        type=Path,
        required=True,
        help="sam3_appearance pass with kept masks",
    )
    compare.add_argument(
        "--output", type=Path, required=True, help="report JSON (a .md is written beside it)"
    )
    compare.add_argument("--title", default="Pool comparison")

    rec2 = commands.add_parser(
        "rec2-references", help="recording-1 masks as the reference set for recording 2"
    )
    rec2.add_argument("--rec1-spec", type=Path, required=True)
    rec2.add_argument("--target-view", default="C10379-rec2")
    rec2.add_argument("--output", type=Path, required=True)

    zero = commands.add_parser(
        "zero-shot", help="score a zero-shot pass against recording 2's human masks"
    )
    zero.add_argument("--pass", dest="pass_dir", type=Path, required=True)
    zero.add_argument("--calibration-manifest", type=Path, required=True)
    zero.add_argument("--set", default="cross_view")
    zero.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    root = Path.cwd().resolve()
    try:
        if args.command == "references":
            _references_main(args, root)
        elif args.command == "rec2-references":
            spec = rec2_reference_spec(
                args.rec1_spec.resolve(), target_view=args.target_view, repository_root=root
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
            print(
                f"Reference spec: {args.output} ({len(spec['sets'][0]['references'])} references)"
            )
        elif args.command == "zero-shot":
            report = score_zero_shot(
                pass_dir=args.pass_dir.resolve(),
                calibration_manifest=args.calibration_manifest.resolve(),
                set_name=args.set,
                repository_root=root,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
            table = zero_shot_table(report)
            args.output.with_suffix(".md").write_text(table, encoding="utf-8")
            print(table)
        elif args.command == "pass":
            _pass_main(args, root)
        elif args.command == "truth-plan":
            anchors = args.anchors or (root / view_paths(VIEW_IDS[args.view])[1])
            seed_run = resolve_run_directory(str(args.seed_run))[1] if args.seed_run else None
            path = build_truth_plan(
                root,
                view=args.view,
                view_id=VIEW_IDS[args.view],
                anchors=anchors,
                consensus_root=args.consensus_root,
                output_dir=args.output_dir,
                extra_positives=(
                    c10379_correction_positives(root) if args.with_c10379_corrections else ()
                ),
                seed_run=seed_run,
            )
            from .correction_acceptance_search import load_plan

            plan = load_plan(path)
            print(
                f"Truth plan: {path} ({plan.cell_count} cells, {plan.unprompted_cells} unprompted, "
                f"{plan.prompt_count} prompts on {len(plan.frames)} frames)"
            )
        elif args.command == "decode-control":
            path = decode_control(root, args.plan, output_dir=args.output_dir, device=args.device)
            print(f"Decode: {path}")
        elif args.command == "compare":
            from .correction_acceptance_search import load_plan

            plan = load_plan(args.plan)
            cells = load_pool_cells(
                root, plan_path=args.plan, decode_dir=args.decode_dir, pass_dir=args.pass_dir
            )
            pools = ["exemplar:same_view:posneg", "exemplar:same_view:pos"]
            if any(
                c.pool.startswith("exemplar:cross_view") for cell in cells for c in cell.candidates
            ):
                pools += ["exemplar:cross_view:posneg", "exemplar:cross_view:pos"]
            if args.decode_dir is not None:
                pools.insert(0, "decoder:base")
            report = compare_pools(cells, pools=pools, anchor_frames=plan.anchor_frames)
            report["plan"] = fingerprint(args.plan, root).model_dump(mode="json")
            report["pass"] = relative_uri(args.pass_dir, root)
            report["decode_dir"] = relative_uri(args.decode_dir, root) if args.decode_dir else None
            report["claim_boundary"] = CLAIM_BOUNDARY
            report["generated_at"] = datetime.now(UTC).isoformat()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
            table = compare_table(report, title=args.title)
            args.output.with_suffix(".md").write_text(table, encoding="utf-8")
            print(table)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
