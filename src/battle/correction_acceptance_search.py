"""Acceptance-rule search for consensus re-prompt corrections, on frames with human truth.

The B4 re-prompt loop located the chassis correctly from the other cameras and then accepted
the wrong pixels: every accepted correction on C10379 was a chassis+hand blob of 15-23k px
where the human's chassis is about 5k px, because the seed-transfer rule (area within
[0.3, 3.0] x the sphere model's expected area, centroid ray within 1.5 radii) cannot tell the
part from the hand holding it.  This module treats every C10379 frame that has a human mask
(the 13 anchor frames and the human correction frames) as if it were a re-prompt onset for
each part with a mask there, builds the geometric prompt from the *others-only* consensus
at that frame exactly as `battle-multiview-reprompt plan` does, decodes the same prompt set
plus hand-aware variants with one warm image decoder, and then scores acceptance rules on the
candidate pools against the human mask.

Four commands under `runs/correction-acceptance-search-20260920/`:

- `plan` (CPU): truth cells x geometric prompts (base = the tool's two boxes with negatives at
  the other parts' consensus centroids; variants add negatives at the dataset hand joints or
  at the stabilized WiLoR landmarks that fall inside the box).
- `decode` (GPU): one warm worker, `batch_decode` per frame.
- `score` (CPU): candidate pools (decoded masks plus hand-subtracted versions), the rule grid
  (area band x decoder-score floor x hand-overlap rule x other-part rule), per-rule metrics,
  leave-frames-out selection over the 13 anchor frames, one chosen rule per part, and the
  `v2_rule.json` the re-prompt tool can apply behind `--acceptance-rule v2`.
- `sheet` (CPU): human | current-rule pick | chosen-rule pick at six frames.

Metric: IoU of the accepted candidate against the human mask; an abstention on a frame where
a human mask exists is neutral (the tracker keeps its own mask), a wrong acceptance (IoU < 0.4)
is the harmful case.  Every number here is against one person's choice of decoder mask on 16
frames of one view: it ranks rules against each other and is not accuracy.  CC BY-NC 4.0.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from pydantic import Field

from . import mask_cache
from .assembly101_camera_fit import PoseMembers
from .assembly101_clock_offset import is_ego
from .four_part_contract import TARGETS
from .mask_ops import overlap_fraction
from .multiview_consensus import load_consensus, load_view_run, relative_uri
from .multiview_geometry import CameraRig
from .multiview_reprompt import build_prompts, onset_geometry
from .multiview_seed_transfer import (
    BOX_MARGINS,
    REFERENCE_VIEW,
    dataset_hands,
    iou,
    mask_centroid,
    project_to_proxy,
    proxy_focal_px,
    ray_point_distance,
)
from .schemas import ArtifactFingerprint, MultiviewRepromptPrompt, VersionedModel, fingerprint
from .seed_search import (
    OUTPUT_ROOT as SEED_SEARCH_ROOT,
)
from .seed_search import (
    SeedTruthSet,
    load_truth_set,
    make_decoder,
    pm_append_run,
    proxy_for_view,
)

OUTPUT_ROOT = Path("runs/correction-acceptance-search-20260920")
CONSENSUS_ROOT = Path("runs/multiview-part-consensus-first-minute-r1280-others-only")
TRUTH_SET = SEED_SEARCH_ROOT / "truth_set.json"
WILOR_RUN = Path("runs/wilor-hands-stabilized-60s-v5")
FIRST_MINUTE = 1800
# Radius (proxy px) the convex hull of a hand's projected joints is grown by before it is used
# to subtract the hand from a candidate or to measure a candidate's overlap with the hand.
HAND_HULL_DILATION_PX = 12
HARM_IOU = 0.4
BAR_IOU = 0.6
BAR_HARM_RATE = 0.15
# A rule that abstains on almost everything trivially has a high mean IoU over what it does
# accept; the fitting objective asks a rule to accept at least this share of the fit cells.
MIN_FIT_ACCEPTANCE = 0.25
HELD_OUT_PER_SPLIT = 3
MAX_RAY_RADII = 1.5
CURRENT_BAND = (0.3, 3.0)
TIGHT_BANDS = ((0.5, 2.0), (0.6, 1.5), (0.7, 1.3))
SEED_LOG_LIMITS = (0.4, 0.7)
SCORE_FLOORS = (0.5, 0.7, 0.9)
MAX_HAND_OVERLAP = 0.2
MAX_OTHER_OVERLAP = 0.3
SHEET_FRAMES = (300, 327, 600, 1050, 1235, 1500)
POOLS = (
    "base",
    "base+hand_negatives_dataset",
    "base+hand_negatives_wilor",
    "base+hand_subtracted_dataset",
    "base+hand_subtracted_wilor",
    "all",
)
CLAIM_BOUNDARIES = (
    "Every IoU is against one person's choice of SAM3 image-decoder mask on 16 frames of one "
    "view (C10379): it ranks acceptance rules against each other and is not accuracy, not a "
    "dataset and not ground truth.",
    "Geometric prompts come from the others-only consensus (seven statics + e4 seeded by "
    "geometric transfer of the C10379 human frame-0 masks) and never from the human mask being "
    "scored; the interior has no consensus and is not searched.",
    "An abstention where a human mask exists is neutral (the tracker keeps its own mask); an "
    "accepted candidate with IoU < 0.4 is the harmful case. CC BY-NC 4.0 applies.",
)
NEGATIVE_VARIANTS = ("hand_negatives_dataset", "hand_negatives_wilor")
SUBTRACT_SOURCES = ("hand_subtracted_dataset", "hand_subtracted_wilor")

PART_COLOURS = {
    "chassis": (60, 60, 230),
    "interior": (230, 200, 40),
    "rear_body": (60, 200, 60),
    "cabin": (230, 120, 40),
}


# ------------------------------------------------------------------------------------ plan


class CellGeometry(VersionedModel):
    """The consensus sphere of one part at one frame, projected into the target view."""

    part: str
    consensus_world_mm: tuple[float, float, float]
    radius_mm: float
    depth_mm: float
    expected_area_px: float
    centroid_proxy_px: tuple[float, float]
    projected_radius_px: float
    views_used: tuple[str, ...]


class TruthCell(VersionedModel):
    frame: int = Field(ge=0)
    part: str
    role: Literal["positive", "hidden"]
    source: str
    mask: ArtifactFingerprint | None = None
    truth_area_px: int | None = None
    seed_area_px: int | None = None
    prompts: tuple[MultiviewRepromptPrompt, ...] = ()
    unprompted_reason: str | None = None


class FramePlan(VersionedModel):
    frame: int = Field(ge=0)
    geometry: dict[str, CellGeometry]
    hands_dataset_px: dict[str, tuple[tuple[float, float], ...]]
    hands_wilor_px: dict[str, tuple[tuple[float, float], ...]]
    cells: tuple[TruthCell, ...]


class AcceptanceSearchPlan(VersionedModel):
    manifest_kind: Literal["correction_acceptance_search_plan"]
    target_view: str
    proxy: ArtifactFingerprint
    proxy_dimensions: tuple[int, int]
    consensus_root_uri: str
    consensus_manifest: ArtifactFingerprint
    consensus_points: ArtifactFingerprint
    truth_set: ArtifactFingerprint
    wilor_observations: ArtifactFingerprint
    seed_run_uri: str
    box_margins: tuple[float, ...]
    hand_hull_dilation_px: int
    frames: tuple[FramePlan, ...]
    prompt_count: int
    cell_count: int
    unprompted_cells: int
    anchor_frames: tuple[int, ...]
    always_fit_frames: tuple[int, ...]
    generated_at: datetime
    claim_boundaries: tuple[str, ...]

    @property
    def prompts(self) -> list[tuple[int, MultiviewRepromptPrompt]]:
        return [(f.frame, p) for f in self.frames for c in f.cells for p in c.prompts]


def truth_cells(truth: SeedTruthSet) -> dict[tuple[int, str], Any]:
    """One human mask per (frame, part) on C10379; the anchor mask wins a duplicate at 900."""
    cells: dict[tuple[int, str], Any] = {}
    for entry in truth.entries:
        if entry.view != REFERENCE_VIEW or entry.role not in ("positive", "hidden"):
            continue
        key = (entry.analysis_frame_index, entry.part)
        current = cells.get(key)
        if current is None or (current.source != "anchors" and entry.source == "anchors"):
            cells[key] = entry
    return cells


def load_wilor_landmarks(path: Path, shape: tuple[int, int]) -> dict[int, dict[str, np.ndarray]]:
    """`frame -> hand -> (21, 2)` proxy px from a stabilized WiLoR `observations.jsonl`."""
    height, width = shape
    out: dict[int, dict[str, np.ndarray]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            hands: dict[str, np.ndarray] = {}
            for hand in row.get("hands") or []:
                points = np.array(
                    [[p["x"] * width, p["y"] * height] for p in hand["landmarks"]], dtype=float
                )
                if points.size:
                    hands[f"{hand['side']}_hand"] = points
            out[int(row["analysis_frame_index"])] = hands
    return out


def hand_hull_mask(
    hands: Mapping[str, np.ndarray], shape: tuple[int, int], dilation_px: int
) -> np.ndarray:
    """Union of the dilated convex hulls of each hand's projected joints (bool, proxy px)."""
    height, width = shape
    hull = np.zeros((height, width), dtype=np.uint8)
    for points in hands.values():
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        pts = pts[~np.isnan(pts).any(axis=1)]
        if pts.shape[0] < 3:
            continue
        convex = cv2.convexHull(np.round(pts).astype(np.int32))
        cv2.fillConvexPoly(hull, convex, 1)
    if dilation_px > 0 and hull.any():
        size = 2 * dilation_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        hull = cv2.dilate(hull, kernel)
    return hull.astype(bool)


def subtract_hand(mask: np.ndarray, hull: np.ndarray) -> np.ndarray | None:
    """`mask` minus the hand hull, largest connected component kept; None when nothing changes."""
    if not hull.any() or not np.logical_and(mask, hull).any():
        return None
    remainder = np.logical_and(mask, ~hull).astype(np.uint8)
    if not remainder.any():
        return None
    count, labels, stats, _ = cv2.connectedComponentsWithStats(remainder, connectivity=8)
    if count <= 1:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def _points_px(hands: Mapping[str, np.ndarray]) -> dict[str, tuple[tuple[float, float], ...]]:
    out = {}
    for name, points in hands.items():
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        pts = pts[~np.isnan(pts).any(axis=1)]
        out[name] = tuple((float(x), float(y)) for x, y in pts)
    return out


def _hands_from_px(
    hands: Mapping[str, Sequence[tuple[float, float]]],
) -> dict[str, np.ndarray]:
    return {name: np.asarray(points, dtype=float).reshape(-1, 2) for name, points in hands.items()}


def _variant_prompts(
    base: Sequence[MultiviewRepromptPrompt],
    variant: Sequence[MultiviewRepromptPrompt],
    label: str,
) -> list[MultiviewRepromptPrompt]:
    """Variant prompts that add at least one negative over the base prompt of the same box."""
    by_box = {(p.pixel_box.x1, p.pixel_box.y1, p.pixel_box.x2, p.pixel_box.y2): p for p in base}
    out = []
    for prompt in variant:
        key = (prompt.pixel_box.x1, prompt.pixel_box.y1, prompt.pixel_box.x2, prompt.pixel_box.y2)
        twin = by_box.get(key)
        if twin is not None and {(p.x, p.y) for p in prompt.background_points} == {
            (p.x, p.y) for p in twin.background_points
        }:
            continue
        out.append(prompt.model_copy(update={"variant": f"{prompt.variant}|{label}"}))
    return out


def plan_search(
    repository_root: Path,
    *,
    output_dir: Path,
    consensus_root: Path = CONSENSUS_ROOT,
    truth_path: Path = TRUTH_SET,
    wilor_run: Path = WILOR_RUN,
    hand_hull_dilation_px: int = HAND_HULL_DILATION_PX,
) -> Path:
    """Write `plan.json`: every truth cell with its others-only geometry and prompt set."""
    repository_root = repository_root.resolve()
    truth = load_truth_set(repository_root / truth_path)
    cells = truth_cells(truth)
    frames = sorted({frame for frame, _ in cells})
    proxy, (width, height) = proxy_for_view(repository_root, REFERENCE_VIEW)
    shape = (height, width)
    rig = CameraRig.load(repository_root)
    members = PoseMembers(repository_root)

    consensus_dir = repository_root / consensus_root
    consensus = load_consensus(consensus_dir / "manifest.json")
    if consensus.reference_view != REFERENCE_VIEW or any(
        s.view == REFERENCE_VIEW for s in consensus.sources
    ):
        raise ValueError("the consensus must be an others-only build for C10379")
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
    seed_run_dir = pm_append_run(repository_root)
    seed_run = load_view_run(repository_root, seed_run_dir, view=REFERENCE_VIEW, frame_count=1)
    seed_area = {}
    for part in TARGETS:
        mask = seed_run.mask(0, part)
        seed_area[part] = int(mask.sum()) if mask is not None else None
    wilor_path = repository_root / wilor_run / "observations.jsonl"
    wilor = load_wilor_landmarks(wilor_path, shape)
    focal = proxy_focal_px(rig, REFERENCE_VIEW)

    frame_plans: list[FramePlan] = []
    prompt_count = 0
    unprompted = 0
    for frame in frames:
        geometry: dict[str, CellGeometry] = {}
        for part in TARGETS:
            point = points[part][frame]
            if np.isnan(point).any():
                continue
            masks: dict[str, np.ndarray] = {}
            poses: dict[str, int | None] = {}
            for view in views:
                if not used[part][view][frame]:
                    continue
                mask = runs[view].mask(frame, part)
                if mask is not None:
                    masks[view] = mask
                    poses[view] = rig.pose_frame(view, frame) if is_ego(view) else None
            g = onset_geometry(
                rig,
                target_view=REFERENCE_VIEW,
                point_world_mm=point,
                masks_by_view=masks,
                pose_frames=poses,
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
        pose_frame = rig.pose_frame(REFERENCE_VIEW, frame)
        hands_dataset = {
            name: project_to_proxy(rig, REFERENCE_VIEW, joints)
            for name, joints in dataset_hands(members, pose_frame).items()
        }
        hands_wilor = wilor.get(frame, {})
        counter = itertools.count(1)
        frame_cells: list[TruthCell] = []
        for part in TARGETS:
            entry = cells.get((frame, part))
            if entry is None:
                continue
            common = dict(
                frame=frame,
                part=part,
                role=entry.role,
                source=entry.source,
                mask=entry.mask,
                truth_area_px=entry.area_px,
                seed_area_px=seed_area.get(part),
            )
            g = geometry.get(part)
            if g is None:
                frame_cells.append(
                    TruthCell(
                        **common,
                        unprompted_reason=(
                            "no others-only consensus for this part at this frame"
                            + (" (no other view tracks the interior)" if part == "interior" else "")
                        ),
                    )
                )
                unprompted += 1
                continue
            centroid = np.asarray(g.centroid_proxy_px)
            others = {o: np.asarray(geometry[o].centroid_proxy_px) for o in geometry if o != part}

            def make(hands: Mapping[str, np.ndarray] | None) -> tuple[MultiviewRepromptPrompt, ...]:
                start = next(counter)
                prompts = build_prompts(
                    target=part,
                    frame=frame,
                    centroid_proxy_px=centroid,
                    half_size_px=g.projected_radius_px,
                    shape=shape,
                    other_centroids=others,
                    hand_joints=hands,
                    margins=BOX_MARGINS,
                    counter_start=start,
                )
                for _ in range(max(len(prompts) - 1, 0)):
                    next(counter)
                return prompts

            base = make(None)
            prompts = list(base)
            prompts += _variant_prompts(base, make(hands_dataset), "hand_negatives_dataset")
            prompts += _variant_prompts(base, make(hands_wilor), "hand_negatives_wilor")
            prompt_count += len(prompts)
            frame_cells.append(TruthCell(**common, prompts=tuple(prompts)))
        frame_plans.append(
            FramePlan(
                frame=frame,
                geometry=geometry,
                hands_dataset_px=_points_px(hands_dataset),
                hands_wilor_px=_points_px(hands_wilor),
                cells=tuple(frame_cells),
            )
        )
    anchor_frames = tuple(
        sorted(
            {
                e.analysis_frame_index
                for e in truth.entries
                if e.source == "anchors" and e.view == REFERENCE_VIEW
            }
        )
    )
    always_fit = tuple(
        sorted(
            {
                e.analysis_frame_index
                for e in truth.entries
                if e.source == "corrections" and e.view == REFERENCE_VIEW and e.role == "positive"
            }
        )
    )
    plan = AcceptanceSearchPlan(
        manifest_kind="correction_acceptance_search_plan",
        target_view=REFERENCE_VIEW,
        proxy=proxy,
        proxy_dimensions=(width, height),
        consensus_root_uri=relative_uri(consensus_dir, repository_root),
        consensus_manifest=fingerprint(consensus_dir / "manifest.json", repository_root),
        consensus_points=fingerprint(consensus_dir / "consensus_points.npz", repository_root),
        truth_set=fingerprint(repository_root / truth_path, repository_root),
        wilor_observations=fingerprint(wilor_path, repository_root),
        seed_run_uri=relative_uri(seed_run_dir, repository_root),
        box_margins=tuple(float(m) for m in BOX_MARGINS),
        hand_hull_dilation_px=hand_hull_dilation_px,
        frames=tuple(frame_plans),
        prompt_count=prompt_count,
        cell_count=len(cells),
        unprompted_cells=unprompted,
        anchor_frames=anchor_frames,
        always_fit_frames=always_fit,
        generated_at=datetime.now(UTC),
        claim_boundaries=CLAIM_BOUNDARIES,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "plan.json"
    path.write_text(plan.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def load_plan(path: Path) -> AcceptanceSearchPlan:
    return AcceptanceSearchPlan.model_validate_json(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------- decode


def decode_request(frame: int, prompt: MultiviewRepromptPrompt, shape: tuple[int, int]) -> dict:
    height, width = shape
    box = prompt.pixel_box
    return {
        "box_id": f"p{prompt.prompt_id[1:]}",
        "candidate_id": prompt.prompt_id,
        "frame_index": frame,
        "pixel_box": box.model_dump(mode="json"),
        "intended_target": prompt.target,
        "boxes": [[[box.x1 / width, box.y1 / height], [box.x2 / width, box.y2 / height]]],
        "fg_points": [],
        "bg_points": [[p.x / width, p.y / height] for p in prompt.background_points],
    }


def run_decode(
    repository_root: Path,
    plan: AcceptanceSearchPlan,
    *,
    output_dir: Path,
    device: str = "cuda:0",
    decoder: Any | None = None,
) -> Path:
    """Decode every planned prompt (one warm worker) into `output_dir/results/`."""
    results_dir = output_dir / "results"
    owns = decoder is None
    if decoder is None:
        decoder = make_decoder(repository_root, plan.proxy, results_dir, device=device)
    started = time.monotonic()
    shape = (plan.proxy_dimensions[1], plan.proxy_dimensions[0])
    decoded: dict[str, dict[str, Any]] = {}
    try:
        for frame_plan in plan.frames:
            requests = [
                decode_request(frame_plan.frame, p, shape)
                for cell in frame_plan.cells
                for p in cell.prompts
            ]
            if not requests:
                continue
            decoder.request("frame_preview", {"frame_index": frame_plan.frame}, timeout=600)
            response = decoder.request("batch_decode", {"prompts": requests}, timeout=1800)
            for item in response["decoded"]:
                decoded[item["candidate_id"]] = item["decoder_result"]
            print(f"frame {frame_plan.frame}: {len(requests)} prompts decoded", flush=True)
    finally:
        if owns:
            decoder.close()
    payload = {
        "target_view": plan.target_view,
        "prompt_count": len(decoded),
        "elapsed_seconds": time.monotonic() - started,
        "decoded": decoded,
    }
    output = output_dir / "decode_result.json"
    output.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return output


# ----------------------------------------------------------------------------------- rules


class AcceptanceRuleSpec(VersionedModel):
    """One acceptance rule: which candidates are eligible, which filters, how to rank."""

    name: str
    pool: str = "base"
    band_kind: Literal["expected_area", "seed_log_ratio"] = "expected_area"
    band_low: float = CURRENT_BAND[0]
    band_high: float = CURRENT_BAND[1]
    max_abs_log_ratio_vs_seed: float | None = None
    max_ray_radii: float = MAX_RAY_RADII
    min_decoder_score: float | None = None
    hand_rule: Literal["dataset", "wilor"] | None = None
    max_hand_overlap: float = MAX_HAND_OVERLAP
    reject_other_part_overlap: bool = False
    max_other_part_overlap: float = MAX_OTHER_OVERLAP
    # `ray_then_score` is the tool's ranking (closest centroid ray wins); `score_then_ray`
    # takes the decoder's own top IoU estimate first and breaks ties by the ray.
    ranking: Literal["ray_then_score", "score_then_ray"] = "ray_then_score"

    def describe(self) -> str:
        parts = [f"pool={self.pool}"]
        parts.append(
            "ranked by centroid ray then decoder score"
            if self.ranking == "ray_then_score"
            else "ranked by decoder score then centroid ray"
        )
        if self.band_kind == "expected_area":
            parts.append(f"area in [{self.band_low}, {self.band_high}] x expected")
        else:
            parts.append(f"|log(area/seed area)| < {self.max_abs_log_ratio_vs_seed}")
        parts.append(f"ray <= {self.max_ray_radii} radii")
        if self.min_decoder_score is not None:
            parts.append(f"decoder score >= {self.min_decoder_score}")
        if self.hand_rule is not None:
            parts.append(f"{self.hand_rule} hand-hull overlap <= {self.max_hand_overlap}")
        if self.reject_other_part_overlap:
            parts.append(f"other-part disc overlap <= {self.max_other_part_overlap}")
        return "; ".join(parts)


def current_rule(pool: str = "base") -> AcceptanceRuleSpec:
    return AcceptanceRuleSpec(
        name="a:current" if pool == "base" else f"a:current@{pool}", pool=pool
    )


def named_rules() -> list[AcceptanceRuleSpec]:
    """The rules the task names, (a) to (f), plus the current rule on each candidate pool."""
    rules = [current_rule()]
    for low, high in TIGHT_BANDS:
        rules.append(AcceptanceRuleSpec(name=f"b:band[{low},{high}]", band_low=low, band_high=high))
    for limit in SEED_LOG_LIMITS:
        rules.append(
            AcceptanceRuleSpec(
                name=f"c:seed_log<{limit}",
                band_kind="seed_log_ratio",
                max_abs_log_ratio_vs_seed=limit,
            )
        )
    for floor in SCORE_FLOORS:
        rules.append(AcceptanceRuleSpec(name=f"d:score>={floor}", min_decoder_score=floor))
    rules.append(AcceptanceRuleSpec(name="e:hand_overlap_dataset<=0.2", hand_rule="dataset"))
    rules.append(AcceptanceRuleSpec(name="e:hand_overlap_wilor<=0.2", hand_rule="wilor"))
    rules.append(
        AcceptanceRuleSpec(name="f:other_part_overlap<=0.3", reject_other_part_overlap=True)
    )
    rules.append(AcceptanceRuleSpec(name="g:rank_by_decoder_score", ranking="score_then_ray"))
    for pool in POOLS[1:]:
        rules.append(current_rule(pool))
    return rules


def rule_grid() -> list[AcceptanceRuleSpec]:
    """Every combination of pool x band x score floor x hand rule x other-part rule."""
    bands: list[tuple[str, dict[str, Any]]] = [(f"band[{CURRENT_BAND[0]},{CURRENT_BAND[1]}]", {})]
    for low, high in TIGHT_BANDS:
        bands.append((f"band[{low},{high}]", {"band_low": low, "band_high": high}))
    for limit in SEED_LOG_LIMITS:
        bands.append(
            (
                f"seed_log<{limit}",
                {"band_kind": "seed_log_ratio", "max_abs_log_ratio_vs_seed": limit},
            )
        )
    grid: list[AcceptanceRuleSpec] = []
    for pool in POOLS:
        for band_name, band in bands:
            for floor in (None, *SCORE_FLOORS):
                for hand in (None, "dataset", "wilor"):
                    for other in (False, True):
                        for ranking in ("ray_then_score", "score_then_ray"):
                            name = "+".join(
                                [
                                    pool,
                                    band_name,
                                    *([f"score>={floor}"] if floor is not None else []),
                                    *([f"hand_{hand}<=0.2"] if hand is not None else []),
                                    *(["other<=0.3"] if other else []),
                                    *(["rank=score"] if ranking == "score_then_ray" else []),
                                ]
                            )
                            grid.append(
                                AcceptanceRuleSpec(
                                    name=name,
                                    pool=pool,
                                    min_decoder_score=floor,
                                    hand_rule=hand,
                                    reject_other_part_overlap=other,
                                    ranking=ranking,
                                    **band,
                                )
                            )
    return grid


@dataclass
class CandidateRecord:
    """One candidate mask with every measurement an acceptance rule may consult."""

    key: str
    prompt_id: str
    candidate_index: int
    variant: str
    source: str  # decoded | hand_subtracted_dataset | hand_subtracted_wilor
    parent_key: str | None
    area_px: int
    decoder_score: float
    area_ratio_vs_expected: float | None
    log_ratio_vs_seed: float | None
    ray_mm: float | None
    ray_radii: float | None
    hand_overlap_dataset: float
    hand_overlap_wilor: float
    other_part_overlap: float
    iou_vs_truth: float
    mask: np.ndarray | None = field(default=None, repr=False)

    @property
    def in_base(self) -> bool:
        return self.source == "decoded" and "|" not in self.variant

    def in_pool(self, pool: str) -> bool:
        if pool == "all":
            return True
        if self.in_base:
            return True
        if pool == "base":
            return False
        extra = pool.split("+", 1)[1]
        if extra in NEGATIVE_VARIANTS:
            return self.source == "decoded" and self.variant.endswith("|" + extra)
        if extra in SUBTRACT_SOURCES:
            return self.source == extra and (self.parent_key or "").endswith(":base")
        raise ValueError(pool)


def candidate_passes(rule: AcceptanceRuleSpec, c: CandidateRecord) -> tuple[bool, list[str]]:
    notes: list[str] = []
    ok = True
    if c.area_px == 0:
        return False, ["empty mask"]
    if rule.band_kind == "expected_area":
        if c.area_ratio_vs_expected is None:
            ok, notes = False, notes + ["no expected area"]
        elif not rule.band_low <= c.area_ratio_vs_expected <= rule.band_high:
            ok = False
            notes.append(
                f"area ratio {c.area_ratio_vs_expected:.2f} outside "
                f"[{rule.band_low}, {rule.band_high}]"
            )
    else:
        limit = rule.max_abs_log_ratio_vs_seed or 0.0
        if c.log_ratio_vs_seed is None:
            ok, notes = False, notes + ["no seed area"]
        elif abs(c.log_ratio_vs_seed) >= limit:
            ok = False
            notes.append(f"|log(area/seed)| {abs(c.log_ratio_vs_seed):.2f} >= {limit}")
    if c.ray_radii is None:
        ok = False
        notes.append("centroid ray points away from the consensus point or no radius")
    elif c.ray_radii > rule.max_ray_radii:
        ok = False
        notes.append(f"centroid ray {c.ray_radii:.2f} radii > {rule.max_ray_radii}")
    if rule.min_decoder_score is not None and c.decoder_score < rule.min_decoder_score:
        ok = False
        notes.append(f"decoder score {c.decoder_score:.2f} < {rule.min_decoder_score}")
    if rule.hand_rule is not None:
        overlap = c.hand_overlap_dataset if rule.hand_rule == "dataset" else c.hand_overlap_wilor
        if overlap > rule.max_hand_overlap:
            ok = False
            notes.append(f"{rule.hand_rule} hand overlap {overlap:.2f} > {rule.max_hand_overlap}")
    if rule.reject_other_part_overlap and c.other_part_overlap > rule.max_other_part_overlap:
        ok = False
        notes.append(
            f"other-part overlap {c.other_part_overlap:.2f} > {rule.max_other_part_overlap}"
        )
    return ok, notes


def apply_rule(
    rule: AcceptanceRuleSpec, candidates: Sequence[CandidateRecord]
) -> CandidateRecord | None:
    """The accepted candidate under `rule` (see `AcceptanceRuleSpec.ranking`)."""
    passing = [c for c in candidates if c.in_pool(rule.pool) and candidate_passes(rule, c)[0]]
    if not passing:
        return None
    if rule.ranking == "score_then_ray":
        return min(
            passing,
            key=lambda c: (-c.decoder_score, c.ray_mm if c.ray_mm is not None else np.inf),
        )
    return min(
        passing, key=lambda c: (c.ray_mm if c.ray_mm is not None else np.inf, -c.decoder_score)
    )


# --------------------------------------------------------------------------------- scoring


@dataclass
class Cell:
    frame: int
    part: str
    role: str
    truth_area: int | None
    candidates: list[CandidateRecord]


def load_cells(
    repository_root: Path, plan: AcceptanceSearchPlan, decode_dir: Path, *, rig: CameraRig
) -> list[Cell]:
    """Every prompted truth cell with its measured candidate pool (decoded + hand-subtracted)."""
    decoded = json.loads((decode_dir / "decode_result.json").read_text(encoding="utf-8"))["decoded"]
    shape = (plan.proxy_dimensions[1], plan.proxy_dimensions[0])
    cells: list[Cell] = []
    for frame_plan in plan.frames:
        hull_dataset = hand_hull_mask(
            _hands_from_px(frame_plan.hands_dataset_px), shape, plan.hand_hull_dilation_px
        )
        hull_wilor = hand_hull_mask(
            _hands_from_px(frame_plan.hands_wilor_px), shape, plan.hand_hull_dilation_px
        )
        for cell in frame_plan.cells:
            if not cell.prompts:
                continue
            geometry = frame_plan.geometry[cell.part]
            point = np.asarray(geometry.consensus_world_mm)
            other_discs = np.zeros(shape, dtype=np.uint8)
            for other, g in frame_plan.geometry.items():
                if other == cell.part:
                    continue
                cv2.circle(
                    other_discs,
                    (int(round(g.centroid_proxy_px[0])), int(round(g.centroid_proxy_px[1]))),
                    max(1, int(round(g.projected_radius_px))),
                    1,
                    -1,
                )
            other_discs_b = other_discs.astype(bool)
            truth_mask = (
                mask_cache.decode_mask_png(repository_root / cell.mask.uri)
                if cell.mask is not None
                else None
            )

            def measure(
                key: str,
                prompt: MultiviewRepromptPrompt,
                index: int,
                source: str,
                parent: str | None,
                mask: np.ndarray,
                score: float,
            ) -> CandidateRecord:
                area = int(np.count_nonzero(mask))
                ray_mm = ray_radii = None
                if area:
                    d = ray_point_distance(rig, REFERENCE_VIEW, mask_centroid(mask), point, None)
                    if np.isfinite(d):
                        ray_mm = float(d)
                        ray_radii = ray_mm / geometry.radius_mm if geometry.radius_mm > 0 else None
                return CandidateRecord(
                    key=key,
                    prompt_id=prompt.prompt_id,
                    candidate_index=index,
                    variant=prompt.variant,
                    source=source,
                    parent_key=parent,
                    area_px=area,
                    decoder_score=score,
                    area_ratio_vs_expected=(
                        area / geometry.expected_area_px if geometry.expected_area_px else None
                    ),
                    log_ratio_vs_seed=(
                        float(np.log((area + 1) / (cell.seed_area_px + 1)))
                        if cell.seed_area_px
                        else None
                    ),
                    ray_mm=ray_mm,
                    ray_radii=ray_radii,
                    hand_overlap_dataset=overlap_fraction(mask, hull_dataset),
                    hand_overlap_wilor=overlap_fraction(mask, hull_wilor),
                    other_part_overlap=overlap_fraction(mask, other_discs_b),
                    iou_vs_truth=(iou(mask, truth_mask) if truth_mask is not None else 0.0),
                    mask=mask,
                )

            records: list[CandidateRecord] = []
            for prompt in cell.prompts:
                result = decoded.get(prompt.prompt_id)
                if result is None:
                    continue
                pool_tag = "base" if "|" not in prompt.variant else prompt.variant.split("|", 1)[1]
                for option in result["candidates"]:
                    mask = mask_cache.decode_mask_png(decode_dir / option["mask_uri"])
                    index = int(option["candidate_index"])
                    key = f"{prompt.prompt_id}:{index}:{pool_tag}"
                    score = float(option["iou_score"])
                    records.append(measure(key, prompt, index, "decoded", None, mask, score))
                    for hull, source in (
                        (hull_dataset, SUBTRACT_SOURCES[0]),
                        (hull_wilor, SUBTRACT_SOURCES[1]),
                    ):
                        cut = subtract_hand(mask, hull)
                        if cut is not None:
                            records.append(
                                measure(f"{key}:{source}", prompt, index, source, key, cut, score)
                            )
            cells.append(Cell(frame_plan.frame, cell.part, cell.role, cell.truth_area_px, records))
    return cells


@dataclass(frozen=True)
class Outcome:
    frame: int
    part: str
    accepted: bool
    iou: float | None
    candidate_key: str | None

    @property
    def harm(self) -> bool:
        return self.accepted and (self.iou or 0.0) < HARM_IOU


def outcome_for(rule: AcceptanceRuleSpec, cell: Cell) -> Outcome:
    chosen = apply_rule(rule, cell.candidates)
    if chosen is None:
        return Outcome(cell.frame, cell.part, False, None, None)
    return Outcome(cell.frame, cell.part, True, chosen.iou_vs_truth, chosen.key)


class RuleMetrics(VersionedModel):
    cells: int
    accepted: int
    acceptance_rate: float | None
    mean_iou_accepted: float | None
    harm: int
    harm_rate: float | None
    # mean over cells of (IoU - 0.4) where accepted, 0 where abstained: positive when the
    # accepted corrections are on balance above the harm line.
    utility: float


def summarize(outcomes: Iterable[Outcome]) -> RuleMetrics:
    items = list(outcomes)
    accepted = [o for o in items if o.accepted]
    harm = sum(1 for o in accepted if o.harm)
    return RuleMetrics(
        cells=len(items),
        accepted=len(accepted),
        acceptance_rate=len(accepted) / len(items) if items else None,
        mean_iou_accepted=(float(np.mean([o.iou or 0.0 for o in accepted])) if accepted else None),
        harm=harm,
        harm_rate=harm / len(accepted) if accepted else None,
        utility=(
            float(np.mean([((o.iou or 0.0) - HARM_IOU) if o.accepted else 0.0 for o in items]))
            if items
            else 0.0
        ),
    )


def objective(metrics: RuleMetrics) -> tuple[float, float, float]:
    """Higher is better: bar-qualifying rules first, then mean accepted IoU, then acceptance."""
    qualifies = (
        metrics.accepted > 0
        and (metrics.harm_rate or 0.0) <= BAR_HARM_RATE
        and (metrics.acceptance_rate or 0.0) >= MIN_FIT_ACCEPTANCE
    )
    if qualifies:
        return (1.0, metrics.mean_iou_accepted or 0.0, metrics.acceptance_rate or 0.0)
    return (0.0, metrics.utility, metrics.acceptance_rate or 0.0)


def meets_bar(metrics: RuleMetrics) -> bool:
    return (
        metrics.accepted > 0
        and (metrics.mean_iou_accepted or 0.0) >= BAR_IOU
        and (metrics.harm_rate or 0.0) <= BAR_HARM_RATE
    )


class RuleRow(VersionedModel):
    rule: str
    description: str
    all_cells: RuleMetrics
    anchor_cells: RuleMetrics
    per_frame_iou: dict[str, float | None]


class LeaveFramesOut(VersionedModel):
    held_out: RuleMetrics
    winners_per_split: dict[str, str]
    winner_counts: dict[str, int]
    held_out_per_frame: dict[str, dict[str, float | None]]


class OracleSummary(VersionedModel):
    """The best candidate in a pool per cell: the ceiling any acceptance rule can reach."""

    pool: str
    mean_best_iou_all_cells: float
    mean_best_iou_anchor_cells: float
    cells_with_best_iou_ge_bar: int
    best_iou_per_frame: dict[str, float]


def oracle_summary(cells: Sequence[Cell], anchor_frames: Sequence[int], pool: str) -> OracleSummary:
    anchors = set(anchor_frames)
    best: dict[int, float] = {}
    for cell in cells:
        pool_candidates = [c for c in cell.candidates if c.in_pool(pool)]
        best[cell.frame] = max((c.iou_vs_truth for c in pool_candidates), default=0.0)
    anchor_values = [v for f, v in best.items() if f in anchors]
    return OracleSummary(
        pool=pool,
        mean_best_iou_all_cells=float(np.mean(list(best.values()))) if best else 0.0,
        mean_best_iou_anchor_cells=float(np.mean(anchor_values)) if anchor_values else 0.0,
        cells_with_best_iou_ge_bar=sum(1 for v in best.values() if v >= BAR_IOU),
        best_iou_per_frame={str(f): round(v, 3) for f, v in sorted(best.items())},
    )


class PartReport(VersionedModel):
    part: str
    cells: int
    anchor_cells: int
    oracle: tuple[OracleSummary, ...] = ()
    named_rules: tuple[RuleRow, ...]
    grid_size: int
    chosen_rule: AcceptanceRuleSpec
    chosen_all_cells: RuleMetrics
    chosen_anchor_cells: RuleMetrics
    chosen_per_frame: dict[str, float | None]
    leave_frames_out: LeaveFramesOut
    current_held_out: RuleMetrics
    top_grid_rules: tuple[RuleRow, ...]
    meets_bar_held_out: bool


class SearchReport(VersionedModel):
    manifest_kind: Literal["correction_acceptance_search_report"]
    plan: ArtifactFingerprint
    decode_result: ArtifactFingerprint
    truth_set: ArtifactFingerprint
    cells_scored: int
    unprompted_cells: int
    candidates_total: int
    grid_size: int
    parts: dict[str, PartReport]
    bar: dict[str, float]
    objective_note: str
    generated_at: datetime
    claim_boundaries: tuple[str, ...]


def _rows(
    rules: Sequence[AcceptanceRuleSpec], cells: Sequence[Cell], anchor_frames: Sequence[int]
) -> list[RuleRow]:
    anchors = set(anchor_frames)
    rows = []
    for rule in rules:
        outcomes = [outcome_for(rule, c) for c in cells]
        rows.append(
            RuleRow(
                rule=rule.name,
                description=rule.describe(),
                all_cells=summarize(outcomes),
                anchor_cells=summarize(o for o in outcomes if o.frame in anchors),
                per_frame_iou={str(o.frame): (o.iou if o.accepted else None) for o in outcomes},
            )
        )
    return rows


def leave_frames_out(
    grid: Sequence[AcceptanceRuleSpec],
    cells: Sequence[Cell],
    *,
    anchor_frames: Sequence[int],
    always_fit: Sequence[int],
    outcomes: Mapping[str, list[Outcome]],
) -> LeaveFramesOut:
    """Fit on all but 3 anchor frames, score the fit winner on those 3, rotate over 13 splits."""
    anchors = [f for f in anchor_frames if any(c.frame == f for c in cells)]
    held_outcomes: list[Outcome] = []
    winners: dict[str, str] = {}
    counts: dict[str, int] = {}
    per_frame: dict[str, dict[str, list[float | None]]] = {}
    for i in range(len(anchors)):
        held = {anchors[(i + k) % len(anchors)] for k in range(HELD_OUT_PER_SPLIT)}
        held -= set(always_fit)
        best_name, best_value = None, None
        for rule in grid:
            fit = [o for o in outcomes[rule.name] if o.frame not in held]
            value = objective(summarize(fit))
            if best_value is None or value > best_value:
                best_name, best_value = rule.name, value
        assert best_name is not None
        label = ",".join(str(h) for h in sorted(held))
        winners[label] = best_name
        counts[best_name] = counts.get(best_name, 0) + 1
        for o in outcomes[best_name]:
            if o.frame in held:
                held_outcomes.append(o)
                per_frame.setdefault(str(o.frame), {}).setdefault(best_name, []).append(
                    o.iou if o.accepted else None
                )
    return LeaveFramesOut(
        held_out=summarize(held_outcomes),
        winners_per_split=winners,
        winner_counts=dict(sorted(counts.items(), key=lambda t: -t[1])),
        held_out_per_frame={
            f: {
                n: (
                    None
                    if all(v is None for v in vs)
                    else float(np.mean([v for v in vs if v is not None]))
                )
                for n, vs in d.items()
            }
            for f, d in sorted(per_frame.items(), key=lambda t: int(t[0]))
        },
    )


def score_search(
    repository_root: Path, *, plan_path: Path, decode_dir: Path, output_dir: Path
) -> SearchReport:
    repository_root = repository_root.resolve()
    plan = load_plan(plan_path)
    rig = CameraRig.load(repository_root)
    cells = load_cells(repository_root, plan, decode_dir, rig=rig)
    grid = rule_grid()
    named = named_rules()
    anchors = set(plan.anchor_frames)
    parts: dict[str, PartReport] = {}
    for part in TARGETS:
        part_cells = [c for c in cells if c.part == part]
        if not part_cells:
            continue
        outcomes = {rule.name: [outcome_for(rule, c) for c in part_cells] for rule in grid}
        by_name = {rule.name: rule for rule in grid}
        ranked = sorted(grid, key=lambda r: objective(summarize(outcomes[r.name])), reverse=True)
        chosen = ranked[0]
        lofo = leave_frames_out(
            grid,
            part_cells,
            anchor_frames=plan.anchor_frames,
            always_fit=plan.always_fit_frames,
            outcomes=outcomes,
        )
        current_outcomes = [outcome_for(current_rule(), c) for c in part_cells]
        chosen_outcomes = outcomes[chosen.name]
        parts[part] = PartReport(
            part=part,
            cells=len(part_cells),
            anchor_cells=sum(1 for c in part_cells if c.frame in anchors),
            oracle=tuple(
                oracle_summary(part_cells, plan.anchor_frames, pool) for pool in ("base", "all")
            ),
            named_rules=tuple(_rows(named, part_cells, plan.anchor_frames)),
            grid_size=len(grid),
            chosen_rule=chosen,
            chosen_all_cells=summarize(chosen_outcomes),
            chosen_anchor_cells=summarize(o for o in chosen_outcomes if o.frame in anchors),
            chosen_per_frame={
                str(o.frame): (o.iou if o.accepted else None) for o in chosen_outcomes
            },
            leave_frames_out=lofo,
            current_held_out=summarize(o for o in current_outcomes if o.frame in anchors),
            top_grid_rules=tuple(
                _rows([by_name[r.name] for r in ranked[:10]], part_cells, plan.anchor_frames)
            ),
            meets_bar_held_out=meets_bar(lofo.held_out),
        )
    report = SearchReport(
        manifest_kind="correction_acceptance_search_report",
        plan=fingerprint(plan_path, repository_root),
        decode_result=fingerprint(decode_dir / "decode_result.json", repository_root),
        truth_set=plan.truth_set,
        cells_scored=len(cells),
        unprompted_cells=plan.unprompted_cells,
        candidates_total=sum(len(c.candidates) for c in cells),
        grid_size=len(grid),
        parts=parts,
        bar={"mean_iou_accepted": BAR_IOU, "harm_rate": BAR_HARM_RATE, "harm_iou": HARM_IOU},
        objective_note=(
            "A split's winner is the grid rule with the highest (qualifies, mean accepted IoU, "
            "acceptance rate) on the fitting cells, where qualifies = harm rate <= "
            f"{BAR_HARM_RATE} and acceptance rate >= {MIN_FIT_ACCEPTANCE}; when no rule "
            "qualifies the utility "
            f"mean(accepted ? IoU - {HARM_IOU} : 0) decides. The chosen rule per part is the "
            "winner on all cells; its held-out numbers are the leave-frames-out procedure's, not "
            "the winner's own."
        ),
        generated_at=datetime.now(UTC),
        claim_boundaries=CLAIM_BOUNDARIES,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "search_report.json").write_text(
        report.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "search_table.md").write_text(search_table(report), encoding="utf-8")
    v2 = {
        "manifest_kind": "correction_acceptance_rule_v2",
        "report": fingerprint(output_dir / "search_report.json", repository_root).model_dump(
            mode="json"
        ),
        "rules": {
            part: {
                **p.chosen_rule.model_dump(mode="json"),
                "held_out": p.leave_frames_out.held_out.model_dump(mode="json"),
                "meets_bar_held_out": p.meets_bar_held_out,
            }
            for part, p in parts.items()
        },
        "bar": report.bar,
        "claim_boundaries": list(CLAIM_BOUNDARIES),
    }
    (output_dir / "v2_rule.json").write_text(
        json.dumps(v2, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _fmt(m: RuleMetrics) -> str:
    iou_text = "-" if m.mean_iou_accepted is None else f"{m.mean_iou_accepted:.3f}"
    acc = "-" if m.acceptance_rate is None else f"{m.acceptance_rate:.2f}"
    harm = "-" if m.harm_rate is None else f"{m.harm_rate:.2f}"
    return f"{iou_text} | {acc} ({m.accepted}/{m.cells}) | {harm} ({m.harm}/{m.accepted})"


def search_table(report: SearchReport) -> str:
    lines = [
        "Columns per cell set: mean IoU over accepted | acceptance rate (accepted/cells) | "
        f"harm rate (accepted with IoU < {HARM_IOU} / accepted). Abstention is neutral.",
        "",
    ]
    for part, p in report.parts.items():
        lines.append(f"## {part} ({p.cells} cells, {p.anchor_cells} on anchor frames)")
        lines.append("")
        for o in p.oracle:
            lines.append(
                f"Oracle ceiling, pool `{o.pool}` (best candidate per cell): mean IoU "
                f"{o.mean_best_iou_all_cells:.3f} all cells / {o.mean_best_iou_anchor_cells:.3f} "
                f"anchor cells; {o.cells_with_best_iou_ge_bar}/{p.cells} cells have any candidate "
                f">= {BAR_IOU}."
            )
        lines.append("")
        lines.append(
            "| rule | all cells: IoU \\| accept \\| harm | anchor cells: IoU \\| accept \\| harm |"
        )
        lines.append("|---|---|---|")
        for row in p.named_rules:
            lines.append(f"| `{row.rule}` | {_fmt(row.all_cells)} | {_fmt(row.anchor_cells)} |")
        lines.append("")
        lines.append(
            f"Chosen (winner on all cells over {p.grid_size} grid rules): `{p.chosen_rule.name}` "
            f"= {p.chosen_rule.describe()}; all cells {_fmt(p.chosen_all_cells)}; anchor cells "
            f"{_fmt(p.chosen_anchor_cells)}."
        )
        lines.append(
            f"Leave-frames-out held-out (fit winner per split scored on its 3 held frames): "
            f"{_fmt(p.leave_frames_out.held_out)}; current rule on the same anchor cells "
            f"{_fmt(p.current_held_out)}; bar (IoU >= {BAR_IOU}, harm <= {BAR_HARM_RATE}) "
            f"{'MET' if p.meets_bar_held_out else 'not met'}."
        )
        lines.append(
            "Split winners: "
            + ", ".join(f"`{n}` x{k}" for n, k in p.leave_frames_out.winner_counts.items())
        )
        lines.append("")
        lines.append("Top grid rules on all cells:")
        lines.append("")
        lines.append("| rule | all cells | anchor cells |")
        lines.append("|---|---|---|")
        for row in p.top_grid_rules:
            lines.append(f"| `{row.rule}` | {_fmt(row.all_cells)} | {_fmt(row.anchor_cells)} |")
        lines.append("")
    lines.append(report.objective_note)
    lines.append("")
    lines.append(" ".join(report.claim_boundaries))
    return "\n".join(lines) + "\n"


def load_report(path: Path) -> SearchReport:
    return SearchReport.model_validate_json(path.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------------- sheet


def render_sheet(
    repository_root: Path,
    *,
    plan: AcceptanceSearchPlan,
    report: SearchReport,
    decode_dir: Path,
    output: Path,
    frames: Sequence[int] = SHEET_FRAMES,
    tile_width: int = 520,
) -> Path:
    """Rows = frames; columns = human masks | current-rule pick | chosen-rule pick per part."""
    rig = CameraRig.load(repository_root)
    cells = load_cells(repository_root, plan, decode_dir, rig=rig)
    truth_entries = truth_cells(load_truth_set(repository_root / plan.truth_set.uri))
    rows = []
    for frame in frames:
        frame_plan = next((f for f in plan.frames if f.frame == frame), None)
        image_path = decode_dir / "results" / "frames" / f"frame-{frame:06d}.jpg"
        if frame_plan is None or not image_path.is_file():
            continue
        image = cv2.imread(str(image_path))
        human = image.copy()
        current = image.copy()
        chosen = image.copy()
        xs: list[int] = []
        ys: list[int] = []
        captions_current: list[str] = []
        captions_chosen: list[str] = []
        for part in TARGETS:
            colour = PART_COLOURS[part]
            entry = truth_entries.get((frame, part))
            if entry is not None and entry.mask is not None:
                truth_mask = mask_cache.decode_mask_png(repository_root / entry.mask.uri)
                yy, xx = np.nonzero(truth_mask)
                if xx.size:
                    xs += [int(xx.min()), int(xx.max())]
                    ys += [int(yy.min()), int(yy.max())]
                _draw(human, truth_mask, colour)
            cell = next((c for c in cells if c.frame == frame and c.part == part), None)
            if cell is None:
                continue
            for p in next(c for c in frame_plan.cells if c.part == part).prompts:
                xs += [p.pixel_box.x1, p.pixel_box.x2]
                ys += [p.pixel_box.y1, p.pixel_box.y2]
            for tile, rule, captions in (
                (current, current_rule(), captions_current),
                (chosen, report.parts[part].chosen_rule, captions_chosen),
            ):
                pick = apply_rule(rule, cell.candidates)
                if pick is None:
                    captions.append(f"{part[:2]} abstain")
                    continue
                assert pick.mask is not None
                _draw(tile, pick.mask, colour)
                tag = "FP" if cell.role == "hidden" else f"{pick.iou_vs_truth:.2f}"
                captions.append(f"{part[:2]} {tag}")
        if not xs:
            continue
        pad = 40
        x0, x1 = max(0, min(xs) - pad), min(image.shape[1], max(xs) + pad)
        y0, y1 = max(0, min(ys) - pad), min(image.shape[0], max(ys) + pad)

        def tile(img: np.ndarray, caption: str) -> np.ndarray:
            crop = img[y0:y1, x0:x1]
            scale = tile_width / crop.shape[1]
            out = cv2.resize(crop, (tile_width, int(crop.shape[0] * scale)))
            for colour, thickness in (((0, 0, 0), 3), ((235, 235, 235), 1)):
                cv2.putText(
                    out,
                    caption,
                    (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    colour,
                    thickness,
                    cv2.LINE_AA,
                )
            return out

        rows.append(
            np.hstack(
                [
                    tile(human, f"f{frame} human"),
                    tile(current, f"f{frame} current rule  " + " ".join(captions_current)),
                    tile(chosen, f"f{frame} chosen rule  " + " ".join(captions_chosen)),
                ]
            )
        )
    if not rows:
        raise ValueError("no frames to render")
    width = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0))) for r in rows]
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), np.vstack(rows))
    return output


def _draw(image: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int]) -> None:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(image, contours, -1, colour, 2)


# ------------------------------------------------------------------------------------- CLI


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="truth cells x others-only geometric prompts (CPU)")
    plan.add_argument("--consensus-root", type=Path, default=CONSENSUS_ROOT)
    plan.add_argument("--truth-set", type=Path, default=TRUTH_SET)
    plan.add_argument("--wilor-run", type=Path, default=WILOR_RUN)
    decode = commands.add_parser("decode", help="one warm image decoder over the plan (GPU)")
    decode.add_argument("--device", default="cuda:0")
    commands.add_parser("score", help="rule grid, metrics, leave-frames-out, v2_rule.json (CPU)")
    sheet = commands.add_parser("sheet", help="human | current pick | chosen pick contact sheet")
    sheet.add_argument("--frame", type=int, action="append", default=None)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    output_root = root / args.output_root
    decode_dir = output_root / "decode"
    if args.command == "plan":
        path = plan_search(
            root,
            output_dir=output_root,
            consensus_root=args.consensus_root,
            truth_path=args.truth_set,
            wilor_run=args.wilor_run,
        )
        loaded = load_plan(path)
        print(
            f"{loaded.cell_count} truth cells, {loaded.cell_count - loaded.unprompted_cells} "
            f"prompted ({loaded.unprompted_cells} without consensus), {loaded.prompt_count} "
            f"prompts over {len(loaded.frames)} frames -> {path}"
        )
    elif args.command == "decode":
        loaded = load_plan(output_root / "plan.json")
        path = run_decode(root, loaded, output_dir=decode_dir, device=args.device)
        print(f"-> {path}")
    elif args.command == "score":
        report = score_search(
            root, plan_path=output_root / "plan.json", decode_dir=decode_dir, output_dir=output_root
        )
        for part, p in report.parts.items():
            print(
                f"{part}: chosen `{p.chosen_rule.name}` all {_fmt(p.chosen_all_cells)}; "
                f"held-out {_fmt(p.leave_frames_out.held_out)}; bar "
                f"{'MET' if p.meets_bar_held_out else 'not met'}"
            )
        print(f"-> {output_root / 'search_report.json'}")
    else:
        loaded = load_plan(output_root / "plan.json")
        report = load_report(output_root / "search_report.json")
        path = render_sheet(
            root,
            plan=loaded,
            report=report,
            decode_dir=decode_dir,
            output=output_root / "acceptance_contact_sheet.png",
            frames=tuple(args.frame) if args.frame else SHEET_FRAMES,
        )
        print(f"-> {path}")


if __name__ == "__main__":
    main()
