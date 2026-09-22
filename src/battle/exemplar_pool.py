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

    args = parser.parse_args()
    root = Path.cwd().resolve()
    try:
        if args.command == "references":
            _references_main(args, root)
        elif args.command == "pass":
            _pass_main(args, root)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
