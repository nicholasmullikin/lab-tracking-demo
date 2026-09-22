"""Seeds after the Sep 21 labelling sessions: human-accepted proposals and the interior.

Two things the human's Sep 21 decisions allow, both written as per-view seed manifests under
one new root (`runs/multiview-seeds-human-accepted-20260921/<VIEW>/seed_manifest.json`) so
the B3 seed manifests stay what they were:

1. `apply`: every seed-search proposal the human accepted in the decisions file becomes that
   view's seed for that part (provenance `agent_proposed_human_accepted`: the mask is the
   agent decoder's, the choice is the human's; the decision file's fingerprint and note are
   recorded on the seed). An undecided cell keeps the existing B3 seed. Each decision is also
   copied into the cell's `proposal.json` (`human_decision`) and the acceptance rate per view
   and part is reported.
2. `interior-plan` / `interior-decode` / `interior-accept`: the interior centroid is
   triangulated from two human masks at the pose instant of C10379 frame 0 (the C10379 frame-0
   calibration seed and the C10119 anchor at frame 41, the earliest human interior mask on a
   second camera; the interior is hand-held and still there) and projected as the seed-search
   winner prompt (margin 0.25, other parts' centroids as negatives, one box, decoder top score)
   into every other view at its frame 0. A candidate is accepted when its centroid
   triangulates with the human observations and the other views' picks (>= 3 views inside the
   30 raw px filter) and its area lies in [0.3, 3.0] x the two human masks' area carried by
   focal/depth; provenance `geometric_from_two_human_views`, held `until the human confirms`
   through the proposal sheet written beside it.

Nothing here is ground truth: the human decisions are one person's choice among agent
decoder candidates on one frame per cell; the interior seeds are agent-selected. CC BY-NC 4.0.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from pydantic import Field

from .assembly101_clock_offset import is_ego
from .digest_cache import sha256_file
from .four_part_contract import TARGETS
from .multiview_consensus import relative_uri
from .multiview_geometry import CameraRig
from .multiview_schemas import (
    HUMAN_ACCEPTED_SEED_PROVENANCE,
    MIXED_SEED_PROVENANCE,
    TWO_HUMAN_VIEWS_SEED_PROVENANCE,
    MultiviewSeedTransferManifest,
    SeedCandidate,
    SeedPrompt,
    SeedTransferPart,
)
from .multiview_seed_transfer import (
    inside_box,
    iou,
    load_manifest,
    load_mask,
    mask_centroid,
    project_to_proxy,
    proxy_focal_px,
    proxy_to_raw_scale,
    square_box,
    write_manifest,
)
from .schemas import ArtifactFingerprint, PixelPoint, VersionedModel
from .schemas import fingerprint as _fingerprint
from .seed_search import (
    E4_VIEW,
    REFERENCE_VIEW,
    TRANSFER_VIEWS,
    make_decoder,
    pm_append_run,
    proxy_for_view,
)

OUTPUT_ROOT = Path("runs/multiview-seeds-human-accepted-20260921")
B3_SEEDS_ROOT = Path("runs/seed-search-20260920/seeds")
DEFAULT_DECISIONS = Path("configs/qa/seed_proposal_decisions_2026-09-21.json")
C10119_VIEW = "C10119"
C10119_ANCHOR_WORKSPACE = Path("runs/human-review-anchors-first-minute-static-c10119")
C10119_ANCHOR_RECORD = Path("docs/qa/first-minute-review-anchors-static-c10119.human-record.json")
C10379_ANCHOR_WORKSPACE = Path("runs/human-review-anchors-first-minute")
E4_ANCHOR_WORKSPACE = Path("runs/human-review-anchors-first-minute-ego-hmc21179183")
C10119_INTERIOR_FRAME = 41
C10119_INTERIOR_CHECK_FRAME = 81
E4_INTERIOR_FRAME = 224
IN_HAND_CHECK = {REFERENCE_VIEW: 300, C10119_VIEW: 301, E4_VIEW: 304}
STATIC_MOTION_MAX_PX = 5.0
BOX_MARGIN = 0.25
AREA_BAND = (0.3, 3.0)
# The interior is a black part: a candidate whose median luminance exceeds this multiple of the
# human masks' median (31 on both C10379 f0 and C10119 f41) is the hand holding it (the frame-0
# decode measured hand candidates at 95-124 against 20-40 for the part), not the interior.
LUMINANCE_RATIO_MAX = 2.0
CONSISTENCY_MIN_VIEWS = 3
CONSISTENCY_MAX_RAW_PX = 30.0
PROPOSALS_ROOT = Path("runs/labeling-sessions-20260921/interior_proposals")
SHEETS_ROOT = Path("runs/labeling-sessions-20260921/interior_proposal_sheets")
TEMPLATE_PATH = Path("configs/qa/interior_seed_decisions.template.json")
SHEET_CONSENSUS_ROOT = Path("runs/multiview-part-consensus-first-minute-r1280-pm-append-seeded")
CLAIM_BOUNDARIES = (
    "A human decision accepts one agent decoder candidate on one frame; it is review evidence "
    "for seeding, not a dataset and not ground truth.",
    "Interior seeds on views other than C10379 and C10119 are agent-selected from geometry "
    "(two human masks triangulated) and a multi-view consistency rule; they are held until the "
    "human confirms them through the interior proposal sheet.",
    "Cross-view consistency measures agreement between estimates, not accuracy.",
    "Assembly101 is CC BY-NC 4.0; attribution applies to every derived artifact.",
)


def b3_manifest_path(repository_root: Path, view: str) -> Path:
    return repository_root / B3_SEEDS_ROOT / view / "seed_manifest.json"


def output_manifest_path(repository_root: Path, output_root: Path, view: str) -> Path:
    return repository_root / output_root / view / "seed_manifest.json"


# -- 1. human decisions -----------------------------------------------------------------------


class DecisionOutcome(VersionedModel):
    view: str
    part: str
    analysis_frame_index: int
    decision: Literal["accept", "reject", "unsure", "undecided"]
    accepted_candidate: int | None
    strategy: str | None = None
    mask_sha256: str | None = None
    area_px: int | None = None
    note: str = ""
    seed_action: Literal["seeded", "kept_existing", "recorded_only", "interior_pending"]


class DecisionsReport(VersionedModel):
    manifest_kind: Literal["seed_proposal_decisions_applied"]
    decisions_file: ArtifactFingerprint
    author: str | None
    reviewed_at: str | None
    outcomes: tuple[DecisionOutcome, ...]
    acceptance_by_part: dict[str, dict[str, int]]
    acceptance_by_view: dict[str, dict[str, int]]
    manifests_written: dict[str, str]
    generated_at: datetime
    claim_boundaries: tuple[str, ...]


def _decision_word(cell: dict[str, Any]) -> str:
    return cell.get("decision") or "undecided"


def acceptance_counts(cells: list[dict[str, Any]], key: str) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for cell in cells:
        bucket = counts.setdefault(
            str(cell[key]), {"accept": 0, "reject": 0, "unsure": 0, "undecided": 0, "cells": 0}
        )
        bucket[_decision_word(cell)] += 1
        bucket["cells"] += 1
    return counts


def human_accepted_part(
    base: SeedTransferPart,
    cell: dict[str, Any],
    *,
    repository_root: Path,
    decisions: ArtifactFingerprint,
) -> SeedTransferPart:
    """The B3 part record with the human-accepted proposal candidate as its seed."""
    proposal_dir = repository_root / Path(cell["proposal"]).parent
    record = json.loads((proposal_dir / "proposal.json").read_text(encoding="utf-8"))
    index = int(cell["accepted_candidate"])
    candidate = record["candidates"][index]
    declared = cell["candidates"][index]
    if declared["mask_sha256"] != candidate["sha256"]:
        raise ValueError(f"{cell['view']} {cell['part']}: decisions file and proposal disagree")
    mask_path = proposal_dir / candidate["mask_uri"]
    if sha256_file(mask_path) != candidate["sha256"]:
        raise ValueError(f"proposal mask changed on disk: {mask_path}")
    mask = load_mask(mask_path)
    area = int(mask.sum())
    expected = base.expected_area_px
    previous = base.accepted
    accepted = SeedCandidate(
        prompt_id=previous.prompt_id if previous is not None else "t000000-b00",
        candidate_index=index,
        mask=_fingerprint(mask_path, repository_root),
        decoder_iou_estimate=float(candidate["decoder_iou_estimate"]),
        mask_area_px=area,
        area_ratio_vs_expected=(area / expected) if expected else None,
        sanity_pass=True,
        acceptance_basis="human_accepted_proposal",
        sanity_notes=(
            f"human accepted proposal candidate {index} ({candidate['strategy']}) in "
            f"{decisions.uri} (sha256 {decisions.sha256[:12]})",
            f"human note: {cell.get('note') or '(none)'}",
            (
                f"replaces the B3 seed {previous.mask.sha256[:12]} "
                f"({previous.acceptance_basis}); IoU with it "
                f"{iou(mask, load_mask(repository_root / previous.mask.uri)):.2f}"
                if previous is not None
                else "the part had no B3 seed"
            ),
        ),
        iou_vs_sep18_seed=previous.iou_vs_sep18_seed if previous is not None else None,
    )
    return base.model_copy(
        update={
            "status": "accepted",
            "blocked_reason": None,
            "provenance": HUMAN_ACCEPTED_SEED_PROVENANCE,
            "prompts": base.prompts,
            "candidates": (),
            "accepted": accepted,
        }
    )


def apply_decisions(
    repository_root: Path,
    *,
    decisions_path: Path,
    output_root: Path = OUTPUT_ROOT,
) -> DecisionsReport:
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    if decisions.get("manifest_kind") != "seed_proposal_decisions":
        raise ValueError(f"{decisions_path} is not a seed_proposal_decisions file")
    fingerprint = _fingerprint(decisions_path, repository_root)
    cells: list[dict[str, Any]] = list(decisions["cells"])
    outcomes: list[DecisionOutcome] = []
    written: dict[str, str] = {}
    by_view: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        by_view.setdefault(cell["view"], []).append(cell)
        proposal_path = repository_root / cell["proposal"]
        record = json.loads(proposal_path.read_text(encoding="utf-8"))
        record["human_decision"] = {
            "decision": cell.get("decision"),
            "accepted_candidate": cell.get("accepted_candidate"),
            "note": cell.get("note", ""),
            "author": decisions.get("author"),
            "at": decisions.get("reviewed_at"),
            "decisions_file": fingerprint.model_dump(mode="json"),
        }
        proposal_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    for view in TRANSFER_VIEWS:
        base = load_manifest(b3_manifest_path(repository_root, view))
        view_cells = {cell["part"]: cell for cell in by_view.get(view, [])}
        parts: list[SeedTransferPart] = []
        for part in base.parts:
            cell = view_cells.get(part.target)
            if cell is None:
                parts.append(part)
                continue
            word = _decision_word(cell)
            candidate = (
                cell["candidates"][int(cell["accepted_candidate"])]
                if word == "accept" and cell.get("accepted_candidate") is not None
                else None
            )
            common = dict(
                view=view,
                part=part.target,
                analysis_frame_index=int(cell["analysis_frame_index"]),
                decision=word,
                accepted_candidate=cell.get("accepted_candidate"),
                strategy=candidate["strategy"] if candidate else None,
                mask_sha256=candidate["mask_sha256"] if candidate else None,
                area_px=candidate["area_px"] if candidate else None,
                note=cell.get("note") or "",
            )
            if part.target == "interior":
                # Interior cells were proposals at the rest frame 427-430, not frame 0; the
                # interior is seeded by `interior-accept` at frame 0 and the decision is kept
                # as evidence there.
                parts.append(part)
                outcomes.append(DecisionOutcome(seed_action="interior_pending", **common))
            elif word == "accept" and candidate is not None:
                parts.append(
                    human_accepted_part(
                        part, cell, repository_root=repository_root, decisions=fingerprint
                    )
                )
                outcomes.append(DecisionOutcome(seed_action="seeded", **common))
            else:
                note = (
                    f"human decision on Sep 21 was '{word}' ({decisions_path.name}); the B3 seed "
                    "is kept unchanged"
                )
                if part.accepted is not None:
                    parts.append(
                        part.model_copy(
                            update={
                                "accepted": part.accepted.model_copy(
                                    update={
                                        "sanity_notes": (*part.accepted.sanity_notes, note),
                                    }
                                )
                            }
                        )
                    )
                else:
                    parts.append(part)
                outcomes.append(DecisionOutcome(seed_action="kept_existing", **common))
        accepted_count = sum(p.status == "accepted" for p in parts)
        summary = ", ".join(f"{p.target}={p.provenance if p.status == 'accepted' else 'blocked'}"
                            for p in parts)  # fmt: skip
        manifest = base.model_copy(
            update={
                "transfer_method": (
                    base.transfer_method + " | Sep 21: the human's seed-proposal decisions "
                    f"({fingerprint.uri}) replace the B3 seed wherever a proposal was accepted "
                    "(provenance agent_proposed_human_accepted); undecided cells keep the B3 "
                    "seed; the interior is seeded separately from two human masks."
                ),
                "parts": tuple(parts),
                "provenance": MIXED_SEED_PROVENANCE,
                "run_decision": "run" if accepted_count >= base.rules.min_parts_to_run else "skip",
                "run_decision_reason": (
                    f"{accepted_count} of {len(TARGETS)} parts seeded ({summary})"
                ),
                "claim_boundaries": (*base.claim_boundaries, *CLAIM_BOUNDARIES),
            }
        )
        path = output_manifest_path(repository_root, output_root, view)
        write_manifest(path, manifest)
        written[view] = relative_uri(path, repository_root)
    report = DecisionsReport(
        manifest_kind="seed_proposal_decisions_applied",
        decisions_file=fingerprint,
        author=decisions.get("author"),
        reviewed_at=decisions.get("reviewed_at"),
        outcomes=tuple(outcomes),
        acceptance_by_part=acceptance_counts(cells, "part"),
        acceptance_by_view=acceptance_counts(cells, "view"),
        manifests_written=written,
        generated_at=datetime.now(UTC),
        claim_boundaries=CLAIM_BOUNDARIES,
    )
    root = repository_root / output_root
    root.mkdir(parents=True, exist_ok=True)
    (root / "decisions_report.json").write_text(
        report.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (root / "decisions_report.md").write_text(decisions_table(report), encoding="utf-8")
    return report


def decisions_table(report: DecisionsReport) -> str:
    lines = [
        "| part | cells | accept | reject | unsure | undecided |",
        "|---|---|---|---|---|---|",
    ]
    for part, c in report.acceptance_by_part.items():
        lines.append(
            f"| {part} | {c['cells']} | {c['accept']} | {c['reject']} | {c['unsure']} | "
            f"{c['undecided']} |"
        )
    lines += [
        "",
        "| view | part | frame | decision | candidate | strategy | area px | seed action | note |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for o in report.outcomes:
        lines.append(
            f"| {o.view} | {o.part} | {o.analysis_frame_index} | {o.decision} | "
            f"{'-' if o.accepted_candidate is None else o.accepted_candidate} | "
            f"{o.strategy or '-'} | {o.area_px or '-'} | {o.seed_action} | {o.note} |"
        )
    lines.append("")
    lines.append(
        f"Decisions file `{report.decisions_file.uri}` "
        f"(sha256 {report.decisions_file.sha256[:12]}), author {report.author}, reviewed "
        f"{report.reviewed_at}. Manifests: "
        + ", ".join(f"`{p}`" for p in report.manifests_written.values())
        + "."
    )
    return "\n".join(lines) + "\n"


# -- 2. interior from two human views ---------------------------------------------------------


class HumanMaskObservation(VersionedModel):
    view: str
    analysis_frame_index: int
    pose_frame_index: int
    mask: ArtifactFingerprint
    centroid_proxy_px: tuple[float, float]
    area_px: int
    source: str
    median_luminance: float | None = None


class InteriorTriangulation(VersionedModel):
    label: str
    observations: tuple[HumanMaskObservation, ...]
    centre_world_mm: tuple[float, float, float]
    reprojection_raw_px: dict[str, float]
    radius_mm_by_view: dict[str, float]
    radius_mm: float
    height_above_table_mm: float | None = None
    note: str | None = None


class InteriorViewPlan(VersionedModel):
    view: str
    view_id: str
    analysis_frame_index: int
    pose_frame_index: int
    is_ego: bool
    proxy: ArtifactFingerprint
    proxy_dimensions: tuple[int, int]
    projected_centre_px: tuple[float, float] | None
    half_size_px: float | None
    depth_mm: float | None
    expected_area_px: float | None
    expected_area_by_human_view: dict[str, float] = Field(default_factory=dict)
    prompt: SeedPrompt | None = None
    negatives_from: tuple[str, ...] = ()
    blocked_reason: str | None = None
    note: str | None = None


class InteriorSeedPlan(VersionedModel):
    manifest_kind: Literal["interior_seed_plan_two_human_views"]
    primary: InteriorTriangulation
    checks: tuple[InteriorTriangulation, ...]
    static_check: dict[str, Any]
    strategy: str
    views: tuple[InteriorViewPlan, ...]
    generated_at: datetime
    claim_boundaries: tuple[str, ...]


def anchor_mask_record(
    repository_root: Path, workspace: Path, frame: int, part: str
) -> tuple[Path, dict[str, Any]]:
    mask_set = json.loads(
        (repository_root / workspace / "anchors" / "anchor_masks.json").read_text(encoding="utf-8")
    )
    for entry in mask_set["anchors"]:
        if entry["analysis_frame_index"] == frame and entry["target"] == part:
            if entry.get("state") != "labeled" or not entry.get("mask_uri"):
                raise ValueError(f"{workspace}: {part} at {frame} is {entry.get('state')}")
            if entry.get("selected_by", "human") != "human":
                raise ValueError(f"{workspace}: {part} at {frame} was not human-selected")
            path = repository_root / workspace / entry["mask_uri"]
            if sha256_file(path) != entry["mask_sha256"]:
                raise ValueError(f"anchor mask changed on disk: {path}")
            return path, entry
    raise KeyError(f"{workspace}: no {part} anchor at frame {frame}")


def reference_frame0_interior(repository_root: Path) -> Path:
    """The human C10379 frame-0 interior seed of the pm-append reference run."""
    settings = json.loads(
        (pm_append_run(repository_root) / "runtime_settings.json").read_text(encoding="utf-8")
    )
    seed = next(
        s
        for s in settings["multi_keyframe_correction_schedule"]["seeds"]
        if s["target"] == "interior"
    )
    if seed["frame_index"] != 0 or seed.get("selected_by", "human") != "human":
        raise ValueError("the reference interior seed must be the human frame-0 mask")
    path = Path(seed["mask_path"])
    if sha256_file(path) != seed["mask_sha256"]:
        raise ValueError(f"reference seed mask changed: {path}")
    return path


def read_proxy_frame(repository_root: Path, view: str, frame: int) -> np.ndarray | None:
    proxy, _ = proxy_for_view(repository_root, view)
    capture = cv2.VideoCapture(str(repository_root / proxy.uri))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = capture.read()
    finally:
        capture.release()
    return image if ok else None


def median_luminance(frame_bgr: np.ndarray | None, mask: np.ndarray) -> float | None:
    if frame_bgr is None or frame_bgr.shape[:2] != mask.shape or not mask.any():
        return None
    grey = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.median(grey[mask]))


def _observation(
    rig: CameraRig,
    repository_root: Path,
    view: str,
    frame: int,
    mask_path: Path,
    *,
    source: str,
) -> tuple[HumanMaskObservation, np.ndarray]:
    mask = load_mask(mask_path)
    centroid = mask_centroid(mask)
    return (
        HumanMaskObservation(
            view=view,
            analysis_frame_index=frame,
            pose_frame_index=rig.pose_frame(view, frame),
            mask=_fingerprint(mask_path, repository_root),
            centroid_proxy_px=(float(centroid[0]), float(centroid[1])),
            area_px=int(mask.sum()),
            source=source,
            median_luminance=median_luminance(read_proxy_frame(repository_root, view, frame), mask),
        ),
        mask,
    )


def triangulate_observations(
    rig: CameraRig,
    observations: list[tuple[HumanMaskObservation, np.ndarray]],
    *,
    label: str,
    plane: Any | None = None,
    note: str | None = None,
) -> InteriorTriangulation:
    points = {
        o.view: (np.asarray(o.centroid_proxy_px) * proxy_to_raw_scale(o.view)).reshape(1, 2)
        for o, _ in observations
    }
    poses = {o.view: o.pose_frame_index for o, _ in observations if is_ego(o.view)}
    result = rig.triangulate(points, pose_frame=poses, reproj_filter_px=None)
    point = result.points[0]
    if np.isnan(point).any():
        raise ValueError(f"{label}: the observations do not triangulate")
    radii: dict[str, float] = {}
    for o, mask in observations:
        depth = float(rig.depth(o.view, point.reshape(1, 3), poses.get(o.view))[0])
        radii[o.view] = float(np.sqrt(mask.sum() / np.pi) * depth / proxy_focal_px(rig, o.view))
    return InteriorTriangulation(
        label=label,
        observations=tuple(o for o, _ in observations),
        centre_world_mm=tuple(float(v) for v in point),
        reprojection_raw_px={
            v: float(e) for v, e in zip(result.views, result.reprojection_px[0], strict=True)
        },
        radius_mm_by_view=radii,
        radius_mm=float(np.median(list(radii.values()))),
        height_above_table_mm=(
            float(plane.signed_distance(point.reshape(1, 3))[0]) if plane is not None else None
        ),
        note=note,
    )


def _other_part_centroids(
    manifest: MultiviewSeedTransferManifest, rig: CameraRig, view: str, pose: int | None
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for part in manifest.parts:
        if part.target == "interior" or not part.source_points_world_mm:
            continue
        point = np.asarray(part.source_points_world_mm[0]).reshape(1, 3)
        projected = project_to_proxy(rig, view, point, pose)[0]
        if not np.isnan(projected).any():
            out[part.target] = projected
    return out


def plan_interior(
    repository_root: Path,
    *,
    output_root: Path = OUTPUT_ROOT,
    views: tuple[str, ...] = TRANSFER_VIEWS,
) -> InteriorSeedPlan:
    from .assembly101_camera_fit import PoseMembers
    from .multiview_consensus import load_view_run
    from .multiview_seed_transfer import fit_first_minute_plane

    rig = CameraRig.load(repository_root)
    plane = fit_first_minute_plane(rig, PoseMembers(repository_root))
    reference_mask = reference_frame0_interior(repository_root)
    c10119_41, _ = anchor_mask_record(
        repository_root, C10119_ANCHOR_WORKSPACE, C10119_INTERIOR_FRAME, "interior"
    )
    c10119_81, _ = anchor_mask_record(
        repository_root, C10119_ANCHOR_WORKSPACE, C10119_INTERIOR_CHECK_FRAME, "interior"
    )
    ref_obs = _observation(
        rig, repository_root, REFERENCE_VIEW, 0, reference_mask, source="human frame-0 seed"
    )
    c41_obs = _observation(
        rig,
        repository_root,
        C10119_VIEW,
        C10119_INTERIOR_FRAME,
        c10119_41,
        source="human review anchor",
    )
    c81_obs = _observation(
        rig,
        repository_root,
        C10119_VIEW,
        C10119_INTERIOR_CHECK_FRAME,
        c10119_81,
        source="human review anchor",
    )
    motion = float(
        np.linalg.norm(
            np.asarray(c41_obs[0].centroid_proxy_px) - np.asarray(c81_obs[0].centroid_proxy_px)
        )
    )
    primary = triangulate_observations(
        rig,
        [ref_obs, c41_obs],
        label="C10379 human f0 x C10119 human f41 (pose frames 17649 / 17729)",
        plane=plane,
        note=(
            "the two masks are 40 analysis frames apart; the interior is hand-held and held "
            "still (C10119 f41 -> f81 centroid motion below) so the pair is used as the frame-0 "
            "sphere, as the task specified"
        ),
    )
    checks: list[InteriorTriangulation] = []
    # Same pose instant: the C10379 pm-append run's own interior mask at frame 40 (tracker
    # output, human-seeded at 0) against the human C10119 f41 mask.
    reference_run = load_view_run(
        repository_root, pm_append_run(repository_root), view=REFERENCE_VIEW
    )
    run_mask_40 = reference_run.mask(40, "interior")
    if run_mask_40 is not None and run_mask_40.any():
        centroid = mask_centroid(run_mask_40)
        run_obs = (
            HumanMaskObservation(
                view=REFERENCE_VIEW,
                analysis_frame_index=40,
                pose_frame_index=rig.pose_frame(REFERENCE_VIEW, 40),
                mask=ArtifactFingerprint(
                    uri="pm-append run interior mask at frame 40 (tracker output)",
                    sha256="0" * 64,
                    source="measured",
                ),
                centroid_proxy_px=(float(centroid[0]), float(centroid[1])),
                area_px=int(run_mask_40.sum()),
                source="C10379 pm-append run mask (not human)",
            ),
            run_mask_40,
        )
        checks.append(
            triangulate_observations(
                rig,
                [run_obs, c41_obs],
                label="same instant: C10379 run f40 x C10119 human f41",
                plane=plane,
                note=(
                    "C10379 interior centroid moved "
                    f"{np.linalg.norm(centroid - np.asarray(ref_obs[0].centroid_proxy_px)):.1f} "
                    "proxy px between frames 0 and 40 (hand drift)"
                ),
            )
        )
    # In hand, three views at one pose instant (anchor 300 / 301 / 304).
    in_hand: list[tuple[HumanMaskObservation, np.ndarray]] = []
    for view, frame, workspace in (
        (REFERENCE_VIEW, IN_HAND_CHECK[REFERENCE_VIEW], C10379_ANCHOR_WORKSPACE),
        (C10119_VIEW, IN_HAND_CHECK[C10119_VIEW], C10119_ANCHOR_WORKSPACE),
        (E4_VIEW, IN_HAND_CHECK[E4_VIEW], E4_ANCHOR_WORKSPACE),
    ):
        path, _ = anchor_mask_record(repository_root, workspace, frame, "interior")
        in_hand.append(
            _observation(rig, repository_root, view, frame, path, source="human review anchor")
        )
    checks.append(
        triangulate_observations(
            rig, in_hand, label="in hand: C10379 300 x C10119 301 x e4 304", plane=plane
        )
    )
    two_static = triangulate_observations(
        rig, in_hand[:2], label="in hand: C10379 300 x C10119 301", plane=plane
    )
    e4_projected = project_to_proxy(
        rig,
        E4_VIEW,
        np.asarray(two_static.centre_world_mm).reshape(1, 3),
        in_hand[2][0].pose_frame_index,
    )[0]
    e4_human = np.asarray(in_hand[2][0].centroid_proxy_px)
    two_static = two_static.model_copy(
        update={
            "note": (
                "projected into e4 at 304: "
                f"({e4_projected[0]:.1f}, {e4_projected[1]:.1f}) vs the human e4 centroid "
                f"({e4_human[0]:.1f}, {e4_human[1]:.1f}), "
                f"{np.linalg.norm(e4_projected - e4_human):.1f} px"
            )
        }
    )
    checks.append(two_static)

    centre = np.asarray(primary.centre_world_mm).reshape(1, 3)
    human_obs = {REFERENCE_VIEW: ref_obs, C10119_VIEW: c41_obs}
    plans: list[InteriorViewPlan] = []
    for view in views:
        proxy, (width, height) = proxy_for_view(repository_root, view)
        shape = (height, width)
        pose = rig.pose_frame(view, 0) if is_ego(view) else None
        depth = float(rig.depth(view, centre, pose)[0])
        projected = project_to_proxy(rig, view, centre, pose)[0]
        base = load_manifest(output_manifest_path(repository_root, output_root, view))
        common: dict[str, Any] = dict(
            view=view,
            view_id=base.view_id,
            analysis_frame_index=0,
            pose_frame_index=rig.pose_frame(view, 0),
            is_ego=is_ego(view),
            proxy=proxy,
            proxy_dimensions=(width, height),
        )
        if (
            depth <= 0
            or np.isnan(projected).any()
            or not (0 <= projected[0] < width and 0 <= projected[1] < height)
        ):
            note = None
            if view == E4_VIEW:
                pose4 = rig.pose_frame(view, 4)
                depth4 = float(rig.depth(view, centre, pose4)[0])
                note = (
                    f"e4 frame 4 (the C10379 frame-0 pose instant): depth {depth4:.0f} mm, "
                    "also behind the camera; the human left the interior unlabelled at e4 44 "
                    f"and 84 and first labelled it at {E4_INTERIOR_FRAME}, which is the seed a "
                    "per-slot start frame would take (not built)"
                )
            plans.append(
                InteriorViewPlan(
                    projected_centre_px=None,
                    half_size_px=None,
                    depth_mm=depth,
                    expected_area_px=None,
                    blocked_reason=(
                        f"interior centre projects outside the view at frame 0 (depth {depth:.0f} "
                        "mm)"
                    ),
                    note=note,
                    **common,
                )
            )
            continue
        half = primary.radius_mm * proxy_focal_px(rig, view) / depth
        expected_by_view: dict[str, float] = {}
        for human_view, (obs, _) in human_obs.items():
            z_h = float(rig.depth(human_view, centre, None)[0])
            scale = (
                proxy_focal_px(rig, view) * z_h / (proxy_focal_px(rig, human_view) * depth)
            ) ** 2
            expected_by_view[human_view] = float(obs.area_px * scale)
        expected = float(np.mean(list(expected_by_view.values())))
        box = square_box(projected, half, BOX_MARGIN, shape)
        if box is None:
            plans.append(
                InteriorViewPlan(
                    projected_centre_px=(float(projected[0]), float(projected[1])),
                    half_size_px=half,
                    depth_mm=depth,
                    expected_area_px=expected,
                    expected_area_by_human_view=expected_by_view,
                    blocked_reason="projected footprint is degenerate in this view",
                    **common,
                )
            )
            continue
        others = _other_part_centroids(base, rig, view, pose)
        negatives = tuple(
            PixelPoint(x=int(round(p[0])), y=int(round(p[1])))
            for p in others.values()
            if inside_box(p, box, slack=0.0) and 0 <= p[0] < width and 0 <= p[1] < height
        )
        plans.append(
            InteriorViewPlan(
                projected_centre_px=(float(projected[0]), float(projected[1])),
                half_size_px=half,
                depth_mm=depth,
                expected_area_px=expected,
                expected_area_by_human_view=expected_by_view,
                prompt=SeedPrompt(
                    prompt_id="t000000-b01",
                    target="interior",
                    variant="two_human_views_sphere_m0.25_other_centroids_one_highest_score",
                    pixel_box=box,
                    background_points=negatives,
                ),
                negatives_from=tuple(
                    name
                    for name, p in others.items()
                    if inside_box(p, box, slack=0.0) and 0 <= p[0] < width and 0 <= p[1] < height
                ),
                **common,
            )
        )
    plan = InteriorSeedPlan(
        manifest_kind="interior_seed_plan_two_human_views",
        primary=primary,
        checks=tuple(checks),
        static_check={
            "c10119_f41_to_f81_centroid_motion_px": motion,
            "threshold_px": STATIC_MOTION_MAX_PX,
            "passed": motion < STATIC_MOTION_MAX_PX,
            "f41_area_px": c41_obs[0].area_px,
            "f81_area_px": c81_obs[0].area_px,
        },
        strategy="m0.25|other_centroids|one|highest_score (the C10379 seed-search interior winner)",
        views=tuple(plans),
        generated_at=datetime.now(UTC),
        claim_boundaries=CLAIM_BOUNDARIES,
    )
    path = repository_root / output_root / "interior" / "interior_plan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(plan.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return plan


def load_interior_plan(path: Path) -> InteriorSeedPlan:
    return InteriorSeedPlan.model_validate_json(path.read_text(encoding="utf-8"))


def decode_interior(
    repository_root: Path,
    plan: InteriorSeedPlan,
    *,
    output_root: Path = OUTPUT_ROOT,
    device: str = "cuda:0",
) -> Path:
    """One warm image decoder per view (each has its own proxy); frame 0 only."""
    summary: dict[str, Any] = {}
    for item in plan.views:
        if item.prompt is None:
            summary[item.view] = {"prompt_count": 0, "decoded": {}, "reason": item.blocked_reason}
            continue
        view_dir = repository_root / output_root / "interior" / item.view
        decoder = make_decoder(repository_root, item.proxy, view_dir / "results", device=device)
        width, height = item.proxy_dimensions
        box = item.prompt.pixel_box
        request = {
            "box_id": f"p{item.prompt.prompt_id[1:]}",
            "candidate_id": item.prompt.prompt_id,
            "frame_index": item.analysis_frame_index,
            "pixel_box": box.model_dump(mode="json"),
            "intended_target": "interior",
            "boxes": [[[box.x1 / width, box.y1 / height], [box.x2 / width, box.y2 / height]]],
            "fg_points": [],
            "bg_points": [[p.x / width, p.y / height] for p in item.prompt.background_points],
        }
        started = time.monotonic()
        try:
            decoder.request(
                "frame_preview", {"frame_index": item.analysis_frame_index}, timeout=600
            )
            response = decoder.request("batch_decode", {"prompts": [request]}, timeout=900)
        finally:
            decoder.close()
        decoded = {d["candidate_id"]: d["decoder_result"] for d in response["decoded"]}
        summary[item.view] = {
            "prompt_count": 1,
            "elapsed_seconds": time.monotonic() - started,
            "decoded": decoded,
        }
        print(f"{item.view}: decoded in {summary[item.view]['elapsed_seconds']:.1f} s", flush=True)
    output = repository_root / output_root / "interior" / "decode_result.json"
    output.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return output


class InteriorCandidate(VersionedModel):
    candidate_index: int
    mask: ArtifactFingerprint
    decoder_iou_estimate: float
    area_px: int
    area_ratio_vs_expected: float | None
    in_area_band: bool
    centroid_proxy_px: tuple[float, float]
    median_luminance: float | None = None
    luminance_ratio_vs_human: float | None = None
    in_luminance_band: bool = True

    @property
    def eligible(self) -> bool:
        return self.in_area_band and self.in_luminance_band


class InteriorViewOutcome(VersionedModel):
    view: str
    decision: Literal["accepted", "not_accepted", "blocked"]
    reason: str
    pick: int | None = None
    candidates: tuple[InteriorCandidate, ...] = ()
    consistency_views_used: tuple[str, ...] = ()
    consistency_reprojection_raw_px: float | None = None
    iou_vs_human_same_view: float | None = None
    centroid_distance_vs_human_same_view_px: float | None = None
    disagreement_top3: float | None = None
    proposal_dir: str | None = None


class InteriorReport(VersionedModel):
    manifest_kind: Literal["interior_seed_report_two_human_views"]
    plan: ArtifactFingerprint
    decode_result: ArtifactFingerprint
    joint_triangulation: dict[str, Any]
    outcomes: tuple[InteriorViewOutcome, ...]
    manifests_updated: dict[str, str]
    proposals_root: str
    template: str | None
    generated_at: datetime
    claim_boundaries: tuple[str, ...]


def _load_candidates(
    repository_root: Path,
    item: InteriorViewPlan,
    decoded: dict[str, Any],
    view_dir: Path,
    *,
    human_luminance: float | None,
) -> list[tuple[InteriorCandidate, np.ndarray]]:
    assert item.prompt is not None
    result = decoded.get(item.prompt.prompt_id)
    if result is None:
        return []
    frame_path = view_dir / "results" / "frames" / f"frame-{item.analysis_frame_index:06d}.jpg"
    frame = cv2.imread(str(frame_path)) if frame_path.is_file() else None
    out: list[tuple[InteriorCandidate, np.ndarray]] = []
    for candidate in result["candidates"]:
        mask_path = view_dir / candidate["mask_uri"]
        mask = load_mask(mask_path)
        area = int(mask.sum())
        if area == 0:
            continue
        ratio = area / item.expected_area_px if item.expected_area_px else None
        centroid = mask_centroid(mask)
        luminance = median_luminance(frame, mask)
        luminance_ratio = (
            luminance / human_luminance if luminance is not None and human_luminance else None
        )
        out.append(
            (
                InteriorCandidate(
                    candidate_index=int(candidate["candidate_index"]),
                    mask=_fingerprint(mask_path, repository_root),
                    decoder_iou_estimate=float(candidate["iou_score"]),
                    area_px=area,
                    area_ratio_vs_expected=ratio,
                    in_area_band=ratio is not None and AREA_BAND[0] <= ratio <= AREA_BAND[1],
                    centroid_proxy_px=(float(centroid[0]), float(centroid[1])),
                    median_luminance=luminance,
                    luminance_ratio_vs_human=luminance_ratio,
                    in_luminance_band=(
                        luminance_ratio is None or luminance_ratio <= LUMINANCE_RATIO_MAX
                    ),
                ),
                mask,
            )
        )
    out.sort(key=lambda c: -c[0].decoder_iou_estimate)
    return out


def accept_interior(
    repository_root: Path,
    *,
    plan: InteriorSeedPlan,
    plan_path: Path,
    decode_path: Path,
    output_root: Path = OUTPUT_ROOT,
    proposals_root: Path = PROPOSALS_ROOT,
) -> InteriorReport:
    rig = CameraRig.load(repository_root)
    decoded_all = json.loads(decode_path.read_text(encoding="utf-8"))
    interior_dir = repository_root / output_root / "interior"
    candidates_by_view: dict[str, list[tuple[InteriorCandidate, np.ndarray]]] = {}
    picks: dict[str, tuple[InteriorCandidate, np.ndarray]] = {}
    human_luminances = [
        o.median_luminance for o in plan.primary.observations if o.median_luminance is not None
    ]
    human_luminance = float(np.median(human_luminances)) if human_luminances else None
    for item in plan.views:
        if item.prompt is None:
            continue
        decoded = decoded_all.get(item.view, {}).get("decoded", {})
        candidates = _load_candidates(
            repository_root,
            item,
            decoded,
            interior_dir / item.view,
            human_luminance=human_luminance,
        )
        candidates_by_view[item.view] = candidates
        eligible = [c for c in candidates if c[0].eligible]
        if eligible:
            picks[item.view] = eligible[0]
    # Joint consistency: the two human observations plus every view's pick, one point.
    points: dict[str, np.ndarray] = {}
    poses: dict[str, int] = {}
    for obs in plan.primary.observations:
        points[obs.view] = (
            np.asarray(obs.centroid_proxy_px) * proxy_to_raw_scale(obs.view)
        ).reshape(1, 2)
    human_views = set(points)
    for view, (pick, _) in picks.items():
        # A picked candidate on a human-observed view replaces that view's human observation in
        # the triangulation (one observation per camera); the human mask is compared directly.
        points[view] = (np.asarray(pick.centroid_proxy_px) * proxy_to_raw_scale(view)).reshape(1, 2)
        if is_ego(view):
            poses[view] = rig.pose_frame(view, 0)
    result = rig.triangulate(points, pose_frame=poses, reproj_filter_px=CONSISTENCY_MAX_RAW_PX)
    used = tuple(v for v, u in zip(result.views, result.used[0], strict=True) if u)
    errors = {
        v: (float(e) if np.isfinite(e) else None)
        for v, e in zip(result.views, result.reprojection_px[0], strict=True)
    }
    joint = {
        "views_offered": list(result.views),
        "views_used": list(used),
        "point_world_mm": (
            None if np.isnan(result.points[0]).any() else [float(v) for v in result.points[0]]
        ),
        "reprojection_raw_px": errors,
        "filter_raw_px": CONSISTENCY_MAX_RAW_PX,
        "human_observations_replaced_by_picks": sorted(human_views & set(picks)),
        "human_median_luminance": human_luminance,
        "luminance_ratio_max": LUMINANCE_RATIO_MAX,
        "distance_from_primary_mm": (
            None
            if np.isnan(result.points[0]).any()
            else float(np.linalg.norm(result.points[0] - np.asarray(plan.primary.centre_world_mm)))
        ),
    }
    human_masks: dict[str, tuple[np.ndarray, int]] = {}
    for obs in plan.primary.observations:
        human_masks[obs.view] = (
            load_mask(repository_root / obs.mask.uri),
            obs.analysis_frame_index,
        )

    outcomes: list[InteriorViewOutcome] = []
    updated: dict[str, str] = {}
    (repository_root / proposals_root).mkdir(parents=True, exist_ok=True)
    for item in plan.views:
        manifest_path = output_manifest_path(repository_root, output_root, item.view)
        manifest = load_manifest(manifest_path)
        candidates = candidates_by_view.get(item.view, [])
        records = tuple(c for c, _ in candidates)
        if item.prompt is None:
            outcome = InteriorViewOutcome(
                view=item.view,
                decision="blocked",
                reason=(item.blocked_reason or "no prompt")
                + (f"; {item.note}" if item.note else ""),
            )
        elif item.view not in picks:
            outcome = InteriorViewOutcome(
                view=item.view,
                decision="not_accepted",
                reason=(
                    f"no candidate inside the area band {AREA_BAND} x expected "
                    f"{item.expected_area_px:.0f} px with median luminance <= "
                    f"{LUMINANCE_RATIO_MAX:.1f} x the human masks' "
                    f"({human_luminance:.0f}); candidate luminances "
                    + ", ".join(
                        f"{c.median_luminance:.0f}"
                        for c in records
                        if c.median_luminance is not None
                    )
                    + " (the hand holding the interior)"
                    if item.expected_area_px and human_luminance is not None
                    else "no expected area"
                ),
                candidates=records,
            )
        else:
            pick, pick_mask = picks[item.view]
            error = errors.get(item.view)
            ok = item.view in used and len(used) >= CONSISTENCY_MIN_VIEWS
            reason = (
                f"{len(used)} views agree, reprojection {error:.1f} raw px"
                if ok and error is not None
                else (
                    f"candidate centroid dropped by the {CONSISTENCY_MAX_RAW_PX:.0f} px filter "
                    f"(reprojection {error:.0f} raw px)"
                    if item.view not in used and error is not None
                    else f"only {len(used)} consistent views (< {CONSISTENCY_MIN_VIEWS})"
                )
            )
            if pick.median_luminance is not None and human_luminance is not None:
                reason += (
                    f"; median luminance {pick.median_luminance:.0f} vs the human masks' "
                    f"{human_luminance:.0f}"
                )
            iou_h = dist_h = None
            if item.view in human_masks:
                human_mask, human_frame = human_masks[item.view]
                iou_h = iou(pick_mask, human_mask)
                dist_h = float(
                    np.linalg.norm(np.asarray(pick.centroid_proxy_px) - mask_centroid(human_mask))
                )
                reason += (
                    f"; vs the human {item.view} mask at frame {human_frame}: IoU {iou_h:.2f}, "
                    f"centroid {dist_h:.1f} px"
                )
            top3 = [m for _, m in candidates[:3]]
            pairs = [iou(a, b) for a, b in itertools.combinations(top3, 2)]
            outcome = InteriorViewOutcome(
                view=item.view,
                decision="accepted" if ok else "not_accepted",
                reason=reason,
                pick=pick.candidate_index,
                candidates=records,
                consistency_views_used=used if ok else tuple(v for v in used),
                consistency_reprojection_raw_px=error,
                iou_vs_human_same_view=iou_h,
                centroid_distance_vs_human_same_view_px=dist_h,
                disagreement_top3=float(1.0 - np.mean(pairs)) if pairs else None,
            )
        # Proposal cell for the human (top-3 by decoder score), whatever the geometric decision.
        if candidates:
            outcome = outcome.model_copy(
                update={
                    "proposal_dir": write_interior_proposal(
                        repository_root,
                        proposals_root,
                        item,
                        candidates[:3],
                        outcome,
                        frame_path=interior_dir
                        / item.view
                        / "results"
                        / "frames"
                        / "frame-000000.jpg",
                    )
                }
            )
        outcomes.append(outcome)
        # Seed manifest update.
        parts = []
        for part in manifest.parts:
            if part.target != "interior":
                parts.append(part)
                continue
            if outcome.decision == "accepted" and item.view in picks:
                pick, _ = picks[item.view]
                notes = (
                    outcome.reason,
                    "sphere from two human masks: "
                    f"{plan.primary.label}; radius {plan.primary.radius_mm:.1f} mm",
                    "geometric_from_two_human_views until the human confirms "
                    f"(interior proposal sheet, {proposals_root.as_posix()})",
                )
                parts.append(
                    part.model_copy(
                        update={
                            "status": "accepted",
                            "blocked_reason": None,
                            "provenance": TWO_HUMAN_VIEWS_SEED_PROVENANCE,
                            "source_points_world_mm": (plan.primary.centre_world_mm,),
                            "triangulation_reprojection_px": plan.primary.reprojection_raw_px,
                            "height_above_table_mm": plan.primary.height_above_table_mm,
                            "radius_mm": plan.primary.radius_mm,
                            "projected_centroid_proxy_px": item.projected_centre_px,
                            "depth_mm": item.depth_mm,
                            "expected_area_px": item.expected_area_px,
                            "prompts": (item.prompt,),
                            "candidates": (),
                            "accepted": SeedCandidate(
                                prompt_id=item.prompt.prompt_id,
                                candidate_index=pick.candidate_index,
                                mask=pick.mask,
                                decoder_iou_estimate=pick.decoder_iou_estimate,
                                mask_area_px=pick.area_px,
                                area_ratio_vs_expected=pick.area_ratio_vs_expected,
                                sanity_pass=True,
                                acceptance_basis="multiview_consistency",
                                sanity_notes=notes,
                                consistency_views_used=outcome.consistency_views_used,
                                consistency_reprojection_px=outcome.consistency_reprojection_raw_px,
                            ),
                        }
                    )
                )
            else:
                parts.append(
                    part.model_copy(
                        update={
                            "status": "blocked",
                            "blocked_reason": f"interior ({outcome.decision}): {outcome.reason}",
                            "prompts": (item.prompt,) if item.prompt is not None else (),
                            "candidates": (),
                            "accepted": None,
                        }
                    )
                )
        accepted_count = sum(p.status == "accepted" for p in parts)
        summary = ", ".join(
            f"{p.target}={p.provenance if p.status == 'accepted' else 'blocked'}" for p in parts
        )
        write_manifest(
            manifest_path,
            manifest.model_copy(
                update={
                    "parts": tuple(parts),
                    "run_decision": (
                        "run" if accepted_count >= manifest.rules.min_parts_to_run else "skip"
                    ),
                    "run_decision_reason": (
                        f"{accepted_count} of {len(TARGETS)} parts seeded ({summary})"
                    ),
                }
            ),
        )
        updated[item.view] = relative_uri(manifest_path, repository_root)
    template = render_interior_sheets(repository_root, proposals_root)
    report = InteriorReport(
        manifest_kind="interior_seed_report_two_human_views",
        plan=_fingerprint(plan_path, repository_root),
        decode_result=_fingerprint(decode_path, repository_root),
        joint_triangulation=joint,
        outcomes=tuple(outcomes),
        manifests_updated=updated,
        proposals_root=proposals_root.as_posix(),
        template=template,
        generated_at=datetime.now(UTC),
        claim_boundaries=CLAIM_BOUNDARIES,
    )
    (interior_dir / "interior_report.json").write_text(
        report.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (interior_dir / "interior_report.md").write_text(interior_table(plan, report), encoding="utf-8")
    return report


def write_interior_proposal(
    repository_root: Path,
    proposals_root: Path,
    item: InteriorViewPlan,
    candidates: list[tuple[InteriorCandidate, np.ndarray]],
    outcome: InteriorViewOutcome,
    *,
    frame_path: Path,
) -> str:
    proposal_dir = repository_root / proposals_root / item.view / "interior_f000000"
    proposal_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    frame = cv2.imread(str(frame_path)) if frame_path.is_file() else None
    palette = [(0, 200, 255), (255, 120, 0), (0, 255, 120)]
    for rank, (candidate, mask) in enumerate(candidates):
        target = proposal_dir / f"candidate_{rank:02d}.png"
        target.write_bytes((repository_root / candidate.mask.uri).read_bytes())
        entries.append(
            {
                "strategy": (
                    f"decoder rank {rank} (score {candidate.decoder_iou_estimate:.2f}, "
                    f"area x{candidate.area_ratio_vs_expected:.2f} expected, luminance "
                    f"{candidate.median_luminance:.0f})"
                    if candidate.area_ratio_vs_expected is not None
                    and candidate.median_luminance is not None
                    else f"decoder rank {rank}"
                ),
                "mask_uri": target.name,
                "sha256": candidate.mask.sha256,
                "area_px": candidate.area_px,
                "decoder_iou_estimate": candidate.decoder_iou_estimate,
                "is_accepted_seed": outcome.decision == "accepted"
                and outcome.pick == candidate.candidate_index,
            }
        )
        if frame is not None:
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(frame, contours, -1, palette[rank % 3], 2)
    if frame is not None and item.prompt is not None:
        box = item.prompt.pixel_box
        cv2.rectangle(frame, (box.x1, box.y1), (box.x2, box.y2), (255, 255, 255), 1)
        cv2.imwrite(str(proposal_dir / "overlay.png"), frame)
    (proposal_dir / "proposal.json").write_text(
        json.dumps(
            {
                "view": item.view,
                "view_id": item.view_id,
                "part": "interior",
                "analysis_frame_index": 0,
                "disagreement_1_minus_mean_pairwise_iou": outcome.disagreement_top3 or 0.0,
                "decision_for_b3": (
                    "accepted_seed" if outcome.decision == "accepted" else "not_accepted"
                ),
                "reason": outcome.reason,
                "candidates": entries,
                "human_decision": None,
                "instructions": (
                    "Frame 0 of this view; the interior is in the subject's hand. Accept the one "
                    "candidate that is the interior and only the interior, or reject all. The "
                    "white box is the prompt from the two human masks (C10379 frame 0, C10119 "
                    "frame 41); candidates are agent decoder outputs ranked by the decoder's own "
                    "score. The run already uses the geometrically accepted candidate "
                    "(is_accepted_seed) with provenance geometric_from_two_human_views until you "
                    "confirm."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return relative_uri(proposal_dir, repository_root)


def render_interior_sheets(repository_root: Path, proposals_root: Path) -> str | None:
    """Contact sheets and the decisions template through `battle-seed-proposal-sheets`."""
    command = [
        sys.executable,
        "-m",
        "battle.seed_proposal_sheets",
        "--repository-root",
        str(repository_root),
        "--proposals-root",
        str(proposals_root),
        "--consensus-root",
        str(SHEET_CONSENSUS_ROOT),
        "--output-root",
        str(SHEETS_ROOT),
        "--template",
        str(TEMPLATE_PATH),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        print(f"proposal sheets failed: {completed.stderr[-2000:]}", file=sys.stderr)
        return None
    print(completed.stdout, end="")
    return TEMPLATE_PATH.as_posix()


def _opt(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def interior_table(plan: InteriorSeedPlan, report: InteriorReport) -> str:
    lines = [
        f"Primary sphere: {plan.primary.label}: centre "
        f"{tuple(round(v, 1) for v in plan.primary.centre_world_mm)} mm, radius "
        f"{plan.primary.radius_mm:.1f} mm (per view "
        + ", ".join(f"{v} {r:.1f}" for v, r in plan.primary.radius_mm_by_view.items())
        + "), reprojection "
        + ", ".join(f"{v} {e:.1f} raw px" for v, e in plan.primary.reprojection_raw_px.items())
        + (
            f", height above the fitted plane {plan.primary.height_above_table_mm:.0f} mm"
            if plan.primary.height_above_table_mm is not None
            else ""
        )
        + ".",
        "",
        f"Static check: C10119 f41 -> f81 centroid motion "
        f"{plan.static_check['c10119_f41_to_f81_centroid_motion_px']:.1f} px "
        f"(< {plan.static_check['threshold_px']:.0f}: {plan.static_check['passed']}).",
        "",
    ]
    for check in plan.checks:
        lines.append(
            f"- {check.label}: centre {tuple(round(v, 1) for v in check.centre_world_mm)} mm, "
            "reprojection "
            + ", ".join(f"{v} {e:.1f}" for v, e in check.reprojection_raw_px.items())
            + " raw px"
            + (
                f", height {check.height_above_table_mm:.0f} mm"
                if check.height_above_table_mm is not None
                else ""
            )
            + (f"; {check.note}" if check.note else "")
        )
    joint = report.joint_triangulation
    lines += [
        "",
        f"Joint triangulation (filter {joint['filter_raw_px']:.0f} raw px): used "
        f"{joint['views_used']} of {joint['views_offered']}; distance from the primary centre "
        f"{joint['distance_from_primary_mm']:.1f} mm."
        if joint.get("distance_from_primary_mm") is not None
        else "Joint triangulation failed.",
        "",
        "| view | decision | pick | decoder score | area px | area ratio | luminance | "
        "views used | reproj raw px | vs human same view (IoU / px) | candidates eligible | "
        "reason |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for o in report.outcomes:
        pick = next((c for c in o.candidates if c.candidate_index == o.pick), None)
        lines.append(
            f"| {o.view} | {o.decision} | {'-' if o.pick is None else o.pick} | "
            f"{'-' if pick is None else f'{pick.decoder_iou_estimate:.2f}'} | "
            f"{'-' if pick is None else pick.area_px} | "
            f"{_opt(None if pick is None else pick.area_ratio_vs_expected)} | "
            f"{_opt(None if pick is None else pick.median_luminance, 0)} | "
            f"{len(o.consistency_views_used) if o.consistency_views_used else '-'} | "
            f"{_opt(o.consistency_reprojection_raw_px, 1)} | "
            f"{_opt(o.iou_vs_human_same_view)} / "
            f"{_opt(o.centroid_distance_vs_human_same_view_px, 1)} | "
            f"{sum(c.eligible for c in o.candidates)} / {len(o.candidates)} | {o.reason} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


# -- CLI --------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    apply = commands.add_parser("apply", help="write the human decisions into seed manifests (CPU)")
    apply.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    commands.add_parser("interior-plan", help="triangulate the interior from two human masks (CPU)")
    decode = commands.add_parser("interior-decode", help="decode the interior prompts (GPU)")
    decode.add_argument("--device", default="cuda:0")
    commands.add_parser("interior-accept", help="accept, write seeds, proposals and sheets (CPU)")
    args = parser.parse_args()
    root = args.repository_root.resolve()
    plan_path = root / args.output_root / "interior" / "interior_plan.json"
    decode_path = root / args.output_root / "interior" / "decode_result.json"
    if args.command == "apply":
        report = apply_decisions(
            root, decisions_path=root / args.decisions, output_root=args.output_root
        )
        print(decisions_table(report))
    elif args.command == "interior-plan":
        plan = plan_interior(root, output_root=args.output_root)
        print(
            f"primary: {plan.primary.label}: centre {plan.primary.centre_world_mm}, radius "
            f"{plan.primary.radius_mm:.1f} mm, reprojection {plan.primary.reprojection_raw_px}"
        )
        print(f"static check: {plan.static_check}")
        for check in plan.checks:
            print(f"check: {check.label}: {check.reprojection_raw_px} {check.note or ''}")
        for item in plan.views:
            print(
                f"{item.view}: "
                + (
                    f"box {item.prompt.pixel_box.model_dump(mode='json')} negatives "
                    f"{item.negatives_from} expected {item.expected_area_px:.0f} px"
                    if item.prompt is not None
                    else f"BLOCKED {item.blocked_reason}"
                )
            )
        print(f"-> {plan_path}")
    elif args.command == "interior-decode":
        output = decode_interior(
            root, load_interior_plan(plan_path), output_root=args.output_root, device=args.device
        )
        print(f"-> {output}")
    else:
        plan = load_interior_plan(plan_path)
        report = accept_interior(
            root,
            plan=plan,
            plan_path=plan_path,
            decode_path=decode_path,
            output_root=args.output_root,
        )
        print(interior_table(plan, report))


if __name__ == "__main__":
    main()
