"""Seeding strategy search on the C10379 human masks, then transfer to the other views.

Three commands, all under `runs/seed-search-20260920/`:

- `battle-seed-truth-set`: every human-chosen SAM3 mask on C10379 with its frame and part
  (the 52 anchor cells, the human correction masks at frames 0 / 327 / 900 / 1235, the
  frame-0 seeds), the decoder alternatives the human rejected or did not choose at those
  frames as labelled negatives, and the Sep 16 e3 (HMC_21110305) human frame-0 masks.
  Agent-selected masks (frame 1172) are listed as excluded, never as truth.
- `battle-seed-search plan|decode|score`: for each truth frame and part a geometric prompt
  built WITHOUT the human mask being scored (the part's centroid triangulated from the other
  views' 1280 runs at the same pose frame, projected into C10379 as a sphere; for the
  interior, which those runs do not track, the run's own previous-frame mask, labelled
  `self_prior`), decoded through the isolated MuggledSAM image decoder over a strategy grid
  (box margin x negative placement x one/two boxes x candidate pick), scored by IoU against
  the human mask with leave-frames-out rotation over the 13 anchor frames.
- `battle-seed-search transfer`: the winning strategy per part applied on the seven other
  statics and e4 at frame 0 (and the interior at its first table-rest frame), accepted by
  multi-view consistency, written as seed manifests the multiview run profile accepts, plus
  ranked accept/reject proposals for the human.

Every IoU here is against one person's choice of decoder mask on a handful of frames of one
view: it ranks strategies against each other and is not accuracy. Seeds written for other
views are agent-selected (`selected_by: agent`). CC BY-NC 4.0 applies to the frames.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
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
from .anchor_frames_for_view import mapped_frame
from .assembly101_camera_fit import PoseMembers
from .assembly101_clock_offset import is_ego
from .assembly101_pose_schemas import ASSEMBLY101_HAND_SIDES
from .four_part_contract import TARGETS
from .muggled_smoke import relative_uri, sha256_file
from .multiview_consensus import ViewRun, load_view_run, mask_centroid_raw
from .multiview_geometry import CameraRig
from .multiview_seed_transfer import (
    REFERENCE_VIEW,
    SECONDARY_RUN,
    SECONDARY_VIEW,
    inside_box,
    iou,
    project_to_proxy,
    proxy_focal_px,
    proxy_to_raw_scale,
    square_box,
    view_id_for,
)
from .schemas import ArtifactFingerprint, PixelBox, PixelPoint, VersionedModel

OUTPUT_ROOT = Path("runs/seed-search-20260920")
ANCHORS_WORKSPACE = Path("runs/human-review-anchors-first-minute")
CORRECTIONS_CALIBRATION = Path(
    "runs/muggledsam-sam3-four-part-focused-corrections-agent-swap-20260918t000947z"
    "/calibration_manifest.json"
)
E3_CALIBRATION = Path(
    "runs/muggledsam-sam3-ego-four-part-focused-calibration-20260916t024915z"
    "/calibration_manifest.json"
)
PM_APPEND_ARM = Path("runs/sam3-memory-arms-20260919/arms/pm-append")
VIEW_RUNS_ROOT = Path("runs/sam3-views-r1280-20260919/views")
OLD_SEED_ROOT = Path("runs/multiview-seed-transfer-20260918")
FIRST_MINUTE = 1800
CORRECTION_FRAMES = (0, 327, 900, 1172, 1235)
OTHER_VIEWS = ("C10095", "C10115", "C10118", "C10119", "C10390", "C10395", "C10404")
E4_VIEW = "HMC_21179183"
TRANSFER_VIEWS = (*OTHER_VIEWS, E4_VIEW)
MARGINS = (0.15, 0.25, 0.4, 0.6)
NEGATIVE_SETS = ("none", "other_centroids", "chassis_rim", "cabin_edge", "hand_joints")
BOX_MODES = ("one", "two")
PICKS = ("highest_score", "largest", "smallest", "exemplar")
TWO_BOX_SECOND_MARGIN = 0.6
RIM_POINTS = 4
MAX_HAND_POINTS = 8
HAND_CONFIDENCE_FLOOR = 0.5
HELD_OUT_PER_SPLIT = 3
TRANSFER_PASS_IOU = 0.6
CONSISTENCY_MIN_VIEWS = 3
CONSISTENCY_MAX_RAW_PX = 30.0
AREA_BAND = (0.3, 3.0)
CLAIM_BOUNDARIES = (
    "Every IoU is against one person's choice of SAM3 image-decoder mask on a handful of "
    "frames of one view (C10379): it ranks seeding strategies against each other and is not "
    "accuracy, not a dataset and not ground truth.",
    "Geometric prompts are built from other views' tracker masks and the calibrated rig, "
    "never from the human mask being scored; the interior arm on C10379 uses the run's own "
    "previous-frame mask (`self_prior`) because no other view tracks the interior.",
    "Seeds written for the other views are agent-selected; acceptance is cross-view "
    "consistency of one tracker's masks, not correctness. CC BY-NC 4.0 applies.",
)

TruthRole = Literal["positive", "hidden", "negative_rejected", "negative_unchosen"]
TruthSource = Literal["anchors", "corrections", "e3_calibration"]


# ------------------------------------------------------------------------------- truth set


class TruthMask(VersionedModel):
    view: str
    view_id: str
    analysis_frame_index: int = Field(ge=0)
    pose_frame_index: int = Field(ge=0)
    part: str
    role: TruthRole
    source: TruthSource
    selected_by: Literal["human"]
    mask: ArtifactFingerprint | None = None
    area_px: int | None = Field(default=None, ge=0)
    source_candidate_id: str
    candidate_index: int | None = Field(default=None, ge=0)
    decoder_iou_estimate: float | None = None
    # For negatives: the candidate id of the positive they were alternatives to (or None
    # when the whole prompt was rejected).
    positive_candidate_id: str | None = None
    note: str | None = None


class ExcludedMask(VersionedModel):
    view: str
    analysis_frame_index: int
    part: str
    selected_by: str
    source_candidate_id: str
    reason: str


class SeedTruthSet(VersionedModel):
    manifest_kind: Literal["seed_search_truth_set"]
    sources: dict[str, ArtifactFingerprint]
    entries: tuple[TruthMask, ...]
    excluded: tuple[ExcludedMask, ...]
    counts: dict[str, int]
    counts_by_part: dict[str, dict[str, int]]
    frames_c10379: tuple[int, ...]
    generated_at: datetime
    claim_boundaries: tuple[str, ...]

    def positives(self, view: str = REFERENCE_VIEW) -> list[TruthMask]:
        return [e for e in self.entries if e.role == "positive" and e.view == view]

    def positive(self, view: str, frame: int, part: str) -> TruthMask | None:
        for entry in self.entries:
            if (
                entry.role == "positive"
                and entry.view == view
                and entry.analysis_frame_index == frame
                and entry.part == part
            ):
                return entry
        return None


def _fingerprint(path: Path, repository_root: Path) -> ArtifactFingerprint:
    return ArtifactFingerprint(
        uri=relative_uri(path.resolve(), repository_root.resolve()),
        sha256=sha256_file(path),
        source="measured",
    )


def _mask_area(path: Path) -> int:
    return int(mask_cache.decode_mask_png(path).sum())


def _collect_calibration(
    manifest_path: Path,
    *,
    repository_root: Path,
    rig: CameraRig,
    view: str,
    source: TruthSource,
    frame_filter: Iterable[int] | None,
) -> tuple[list[TruthMask], list[ExcludedMask]]:
    """Human-chosen masks (positives), their unchosen alternatives and rejected prompts."""
    payload = json.loads((repository_root / manifest_path).read_text(encoding="utf-8"))
    workspace = (repository_root / manifest_path).parent
    wanted = None if frame_filter is None else set(frame_filter)
    entries: list[TruthMask] = []
    excluded: list[ExcludedMask] = []
    view_id = payload["view_id"]
    for candidate in payload["candidates"]:
        frame = int(candidate["frame"]["analysis_frame_index"])
        part = candidate["intended_target"]
        if wanted is not None and frame not in wanted:
            excluded.append(
                ExcludedMask(
                    view=view,
                    analysis_frame_index=frame,
                    part=part,
                    selected_by=candidate.get("selected_by") or "human",
                    source_candidate_id=candidate["candidate_id"],
                    reason="frame outside the first minute truth frames",
                )
            )
            continue
        selected_by = candidate.get("selected_by") or "human"
        if selected_by != "human":
            excluded.append(
                ExcludedMask(
                    view=view,
                    analysis_frame_index=frame,
                    part=part,
                    selected_by=selected_by,
                    source_candidate_id=candidate["candidate_id"],
                    reason="agent-selected mask; not human truth",
                )
            )
            continue
        pose = rig.pose_frame(view, frame)
        chosen = candidate.get("human_selected_candidate_index")
        accepted = bool(candidate.get("human_accepted")) and not candidate.get("rejected")
        for option in candidate["decoder_result"]["candidates"]:
            mask_path = workspace / option["mask_uri"]
            index = int(option["candidate_index"])
            if accepted and index == chosen:
                role: TruthRole = "positive"
                positive_id = None
            elif candidate.get("rejected"):
                role = "negative_rejected"
                positive_id = None
            elif accepted:
                role = "negative_unchosen"
                positive_id = candidate["candidate_id"]
            else:
                # Neither accepted nor rejected: an abandoned prompt; not evidence either way.
                continue
            entries.append(
                TruthMask(
                    view=view,
                    view_id=view_id,
                    analysis_frame_index=frame,
                    pose_frame_index=pose,
                    part=part,
                    role=role,
                    source=source,
                    selected_by="human",
                    mask=_fingerprint(mask_path, repository_root),
                    area_px=_mask_area(mask_path),
                    source_candidate_id=candidate["candidate_id"],
                    candidate_index=index,
                    decoder_iou_estimate=float(option["iou_score"]),
                    positive_candidate_id=positive_id,
                )
            )
    return entries, excluded


def _hidden_marks(
    manifest_path: Path, *, repository_root: Path, rig: CameraRig, view: str
) -> list[TruthMask]:
    payload = json.loads((repository_root / manifest_path).read_text(encoding="utf-8"))
    out = []
    for mark in payload.get("hidden_targets", []):
        frame = int(mark["frame"]["analysis_frame_index"])
        out.append(
            TruthMask(
                view=view,
                view_id=payload["view_id"],
                analysis_frame_index=frame,
                pose_frame_index=rig.pose_frame(view, frame),
                part=mark["intended_target"],
                role="hidden",
                source="anchors",
                selected_by="human",
                source_candidate_id=f"hidden-{frame:06d}-{mark['intended_target']}",
                note="human marked the part not visible; any mask here is a false positive",
            )
        )
    return out


def build_truth_set(repository_root: Path) -> SeedTruthSet:
    rig = CameraRig.load(repository_root)
    anchors_manifest = ANCHORS_WORKSPACE / "calibration_manifest.json"
    entries: list[TruthMask] = []
    excluded: list[ExcludedMask] = []
    a, x = _collect_calibration(
        anchors_manifest,
        repository_root=repository_root,
        rig=rig,
        view=REFERENCE_VIEW,
        source="anchors",
        frame_filter=None,
    )
    entries += a + _hidden_marks(
        anchors_manifest, repository_root=repository_root, rig=rig, view=REFERENCE_VIEW
    )
    excluded += x
    c, x = _collect_calibration(
        CORRECTIONS_CALIBRATION,
        repository_root=repository_root,
        rig=rig,
        view=REFERENCE_VIEW,
        source="corrections",
        frame_filter=CORRECTION_FRAMES,
    )
    entries += c
    excluded += x
    e, x = _collect_calibration(
        E3_CALIBRATION,
        repository_root=repository_root,
        rig=rig,
        view=SECONDARY_VIEW,
        source="e3_calibration",
        frame_filter=(0,),
    )
    entries += e
    excluded += x
    # The exported anchor masks must be the same bytes as the chosen workspace candidates.
    exported = json.loads(
        (repository_root / ANCHORS_WORKSPACE / "anchors/anchor_masks.json").read_text()
    )
    exported_sha = {
        (item["analysis_frame_index"], item["target"]): item["mask_sha256"]
        for item in exported["anchors"]
        if item["state"] == "labeled"
    }
    for entry in entries:
        if entry.source == "anchors" and entry.role == "positive":
            key = (entry.analysis_frame_index, entry.part)
            if exported_sha.get(key) != (entry.mask.sha256 if entry.mask else None):
                raise ValueError(f"anchor export disagrees with the workspace at {key}")
    counts: dict[str, int] = {}
    by_part: dict[str, dict[str, int]] = {}
    for entry in entries:
        key = f"{entry.role}:{entry.source}"
        counts[key] = counts.get(key, 0) + 1
        by_part.setdefault(entry.part, {})
        by_part[entry.part][entry.role] = by_part[entry.part].get(entry.role, 0) + 1
    counts["positive_total"] = sum(1 for e in entries if e.role == "positive")
    counts["positive_c10379"] = sum(
        1 for e in entries if e.role == "positive" and e.view == REFERENCE_VIEW
    )
    counts["hidden"] = sum(1 for e in entries if e.role == "hidden")
    counts["negative_rejected"] = sum(1 for e in entries if e.role == "negative_rejected")
    counts["negative_unchosen"] = sum(1 for e in entries if e.role == "negative_unchosen")
    counts["excluded"] = len(excluded)
    frames = tuple(
        sorted(
            {
                e.analysis_frame_index
                for e in entries
                if e.view == REFERENCE_VIEW and e.role in ("positive", "hidden")
            }
        )
    )
    return SeedTruthSet(
        manifest_kind="seed_search_truth_set",
        sources={
            "anchors_calibration": _fingerprint(
                repository_root / anchors_manifest, repository_root
            ),
            "anchors_export": _fingerprint(
                repository_root / ANCHORS_WORKSPACE / "anchors/anchor_masks.json", repository_root
            ),
            "corrections_calibration": _fingerprint(
                repository_root / CORRECTIONS_CALIBRATION, repository_root
            ),
            "e3_calibration": _fingerprint(repository_root / E3_CALIBRATION, repository_root),
        },
        entries=tuple(entries),
        excluded=tuple(excluded),
        counts=counts,
        counts_by_part=by_part,
        frames_c10379=frames,
        generated_at=datetime.now(UTC),
        claim_boundaries=CLAIM_BOUNDARIES,
    )


def load_truth_set(path: Path) -> SeedTruthSet:
    return SeedTruthSet.model_validate_json(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------------ geometric prompts


def pm_append_run(repository_root: Path) -> Path:
    runs = sorted(
        d for d in (repository_root / PM_APPEND_ARM).iterdir() if (d / "manifest.json").is_file()
    )
    if len(runs) != 1:
        raise FileNotFoundError(f"{PM_APPEND_ARM} must hold exactly one run")
    return runs[0]


def view_run_directory(repository_root: Path, view: str) -> Path:
    root = repository_root / VIEW_RUNS_ROOT / view
    runs = sorted(d for d in root.iterdir() if (d / "manifest.json").is_file())
    if len(runs) != 1:
        raise FileNotFoundError(f"{root} must hold exactly one run")
    return runs[0]


class Sphere(VersionedModel):
    """A part's triangulated centre and radius, with the views that produced it."""

    part: str
    centre_world_mm: tuple[float, float, float] | None
    radius_mm: float | None
    views_used: tuple[str, ...]
    reprojection_px: dict[str, float]
    source: Literal["other_views", "self_prior", "reference_plus_e3", "unavailable"]
    # `self_prior` only: the 2D centre / half size in the target view (proxy px).
    prior_centre_px: tuple[float, float] | None = None
    prior_half_size_px: float | None = None
    note: str | None = None


def mask_boundary_points(mask: np.ndarray, count: int) -> np.ndarray:
    """`count` points spread along the largest external contour (proxy px), or empty."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros((0, 2))
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    if contour.shape[0] < count:
        return contour + 0.5
    picks = np.linspace(0, contour.shape[0], count, endpoint=False).astype(int)
    return contour[picks] + 0.5


@dataclass
class GeometryContext:
    """Rig, poses and the tracker runs the prompts are built from (loaded once)."""

    repository_root: Path
    rig: CameraRig
    members: PoseMembers
    runs: dict[str, ViewRun] = field(default_factory=dict)
    frame_shift: dict[str, int] = field(default_factory=dict)

    @classmethod
    def load(
        cls, repository_root: Path, *, views: Iterable[str] = TRANSFER_VIEWS
    ) -> GeometryContext:
        rig = CameraRig.load(repository_root)
        context = cls(repository_root, rig, PoseMembers(repository_root))
        context.runs[REFERENCE_VIEW] = load_view_run(
            repository_root, pm_append_run(repository_root), view=REFERENCE_VIEW
        )
        for view in views:
            context.runs[view] = load_view_run(
                repository_root, view_run_directory(repository_root, view), view=view
            )
        context.runs[SECONDARY_VIEW] = load_view_run(
            repository_root, repository_root / SECONDARY_RUN, view=SECONDARY_VIEW
        )
        source_rule = rig.clock_rule(REFERENCE_VIEW)
        for view in context.runs:
            context.frame_shift[view] = mapped_frame(source_rule, rig.clock_rule(view), 0)[0]
        return context

    def frame_in(self, view: str, reference_frame: int) -> int:
        """The analysis frame of `view` showing C10379's `reference_frame` (half frames down)."""
        return reference_frame + self.frame_shift[view]

    def pose(self, view: str, frame: int) -> int | None:
        return self.rig.pose_frame(view, frame) if is_ego(view) else None

    def mask(self, view: str, reference_frame: int, part: str) -> np.ndarray | None:
        run = self.runs.get(view)
        if run is None:
            return None
        frame = self.frame_in(view, reference_frame)
        if frame < 0 or frame >= FIRST_MINUTE:
            return None
        return run.mask(frame, part)

    def sphere_from_views(self, part: str, reference_frame: int, views: Sequence[str]) -> Sphere:
        """Triangulate the part centroid from `views`' tracker masks at the same pose frame."""
        points: dict[str, np.ndarray] = {}
        poses: dict[str, int] = {}
        areas: dict[str, float] = {}
        for view in views:
            mask = self.mask(view, reference_frame, part)
            if mask is None or not mask.any():
                continue
            scale = proxy_to_raw_scale(view)
            centroid = mask_centroid_raw(mask, scale)
            if centroid is None:
                continue
            points[view] = centroid.reshape(1, 2)
            areas[view] = float(mask.sum())
            if is_ego(view):
                poses[view] = self.rig.pose_frame(view, self.frame_in(view, reference_frame))
        if len(points) < 2:
            return Sphere(
                part=part,
                centre_world_mm=None,
                radius_mm=None,
                views_used=(),
                reprojection_px={},
                source="unavailable",
                note=f"only {len(points)} view(s) carry a {part} mask at this frame",
            )
        result = self.rig.triangulate(
            points, pose_frame=poses, reproj_filter_px=CONSISTENCY_MAX_RAW_PX
        )
        point = result.points[0]
        if np.isnan(point).any():
            return Sphere(
                part=part,
                centre_world_mm=None,
                radius_mm=None,
                views_used=(),
                reprojection_px={},
                source="unavailable",
                note="triangulation left fewer than two consistent views",
            )
        used = tuple(v for v, u in zip(result.views, result.used[0], strict=True) if u)
        radii = []
        for view in used:
            depth = float(self.rig.depth(view, point.reshape(1, 3), poses.get(view))[0])
            if depth <= 0:
                continue
            radii.append(np.sqrt(areas[view] / np.pi) * depth / proxy_focal_px(self.rig, view))
        return Sphere(
            part=part,
            centre_world_mm=tuple(float(v) for v in point),
            radius_mm=float(np.median(radii)) if radii else None,
            views_used=used,
            reprojection_px={
                v: float(e)
                for v, e in zip(result.views, result.reprojection_px[0], strict=True)
                if np.isfinite(e)
            },
            source="other_views",
        )

    def self_prior(self, view: str, reference_frame: int, part: str) -> Sphere:
        """The target run's own mask one frame earlier (or later at frame 0), 2D only."""
        for delta in (-1, 1, -2, 2):
            frame = reference_frame + delta
            if frame < 0:
                continue
            mask = self.mask(view, frame, part)
            if mask is not None and mask.any():
                ys, xs = np.nonzero(mask)
                centre = (float(xs.mean() + 0.5), float(ys.mean() + 0.5))
                return Sphere(
                    part=part,
                    centre_world_mm=None,
                    radius_mm=None,
                    views_used=(view,),
                    reprojection_px={},
                    source="self_prior",
                    prior_centre_px=centre,
                    prior_half_size_px=float(np.sqrt(mask.sum() / np.pi)),
                    note=f"{view} run mask at reference frame {frame} (delta {delta:+d})",
                )
        return Sphere(
            part=part,
            centre_world_mm=None,
            radius_mm=None,
            views_used=(),
            reprojection_px={},
            source="unavailable",
            note="no neighbouring run mask",
        )

    def projected_centre(
        self, sphere: Sphere, view: str, reference_frame: int
    ) -> tuple[np.ndarray, float] | None:
        """(centre proxy px, half size proxy px) of a sphere in `view`, or None if unseen."""
        if sphere.source == "self_prior":
            assert sphere.prior_centre_px is not None and sphere.prior_half_size_px is not None
            return np.asarray(sphere.prior_centre_px), sphere.prior_half_size_px
        if sphere.centre_world_mm is None or sphere.radius_mm is None:
            return None
        pose = self.pose(view, self.frame_in(view, reference_frame))
        point = np.asarray(sphere.centre_world_mm).reshape(1, 3)
        depth = float(self.rig.depth(view, point, pose)[0])
        centre = project_to_proxy(self.rig, view, point, pose)[0]
        if depth <= 0 or np.isnan(centre).any():
            return None
        return centre, sphere.radius_mm * proxy_focal_px(self.rig, view) / depth

    def hand_joints_px(self, view: str, reference_frame: int) -> np.ndarray:
        pose_frame = self.rig.pose_frame(view, self.frame_in(view, reference_frame))
        key = str(pose_frame)
        frame = self.members.landmarks3d.get(key)
        if frame is None:
            return np.zeros((0, 2))
        joints = []
        for index in ASSEMBLY101_HAND_SIDES:
            if float(self.members.confidences[key][str(index)]) < HAND_CONFIDENCE_FLOOR:
                continue
            joints.append(np.asarray(frame[str(index)], dtype=np.float64))
        if not joints:
            return np.zeros((0, 2))
        projected = project_to_proxy(
            self.rig,
            view,
            np.concatenate(joints),
            self.pose(view, self.frame_in(view, reference_frame)),
        )
        return projected[~np.isnan(projected).any(axis=1)]


class PromptSpec(VersionedModel):
    """One decodable prompt: a box and background points, with its strategy coordinates."""

    prompt_id: str
    view: str
    reference_frame: int = Field(ge=0)
    target_frame: int = Field(ge=0)
    part: str
    margin: float
    negatives: str
    pixel_box: PixelBox
    background_points: tuple[PixelPoint, ...] = ()
    sphere_source: str
    note: str | None = None


def _points_inside(points: np.ndarray, box: PixelBox, shape: tuple[int, int]) -> list[PixelPoint]:
    out = []
    for p in points:
        if inside_box(p, box, slack=0.0) and 0 <= p[0] < shape[1] and 0 <= p[1] < shape[0]:
            out.append(PixelPoint(x=int(round(p[0])), y=int(round(p[1]))))
    return out


def build_prompts(
    context: GeometryContext,
    *,
    view: str,
    reference_frame: int,
    part: str,
    sphere: Sphere,
    other_spheres: Mapping[str, Sphere],
    shape: tuple[int, int],
    counter: itertools.count,
    margins: Sequence[float] = MARGINS,
    negative_sets: Sequence[str] = NEGATIVE_SETS,
) -> list[PromptSpec]:
    """The unique prompts of the grid for one (view, frame, part); empty when unseen."""
    projected = context.projected_centre(sphere, view, reference_frame)
    if projected is None:
        return []
    centre, half = projected
    target_frame = context.frame_in(view, reference_frame)
    others = {
        name: context.projected_centre(s, view, reference_frame)
        for name, s in other_spheres.items()
        if name != part
    }
    other_points = np.array([c for c, _ in others.values() if c is not None]).reshape(-1, 2)
    rim_sources = {"chassis_rim": "chassis", "cabin_edge": "cabin"}
    prompts: list[PromptSpec] = []
    seen: set[tuple[Any, ...]] = set()
    for margin in margins:
        box = square_box(centre, half, margin, shape)
        if box is None:
            continue
        for negatives in negative_sets:
            note = None
            if negatives == "none":
                points: list[PixelPoint] = []
            elif negatives == "other_centroids":
                points = _points_inside(other_points, box, shape)
            elif negatives in rim_sources:
                source_part = rim_sources[negatives]
                if source_part == part:
                    continue
                mask = context.mask(view, reference_frame, source_part)
                if mask is None:
                    continue
                points = _points_inside(mask_boundary_points(mask, RIM_POINTS), box, shape)
                note = f"{negatives} from the {view} run's own {source_part} mask"
            elif negatives == "hand_joints":
                joints = context.hand_joints_px(view, reference_frame)
                if joints.shape[0]:
                    order = np.argsort(np.linalg.norm(joints - centre, axis=1))
                    joints = joints[order]
                points = _points_inside(joints, box, shape)[:MAX_HAND_POINTS]
            else:
                raise ValueError(negatives)
            key = (box.x1, box.y1, box.x2, box.y2, tuple((p.x, p.y) for p in points))
            if key in seen:
                # A negative set that adds no point inside this box is the `none` prompt.
                continue
            seen.add(key)
            prompts.append(
                PromptSpec(
                    prompt_id=f"t{target_frame:06d}-b{next(counter):02d}",
                    view=view,
                    reference_frame=reference_frame,
                    target_frame=target_frame,
                    part=part,
                    margin=margin,
                    negatives=negatives,
                    pixel_box=box,
                    background_points=tuple(points),
                    sphere_source=sphere.source,
                    note=note,
                )
            )
    return prompts


def decode_request(prompt: PromptSpec, shape: tuple[int, int]) -> dict[str, Any]:
    height, width = shape
    box = prompt.pixel_box
    return {
        "box_id": f"p{prompt.prompt_id[1:]}",
        "candidate_id": prompt.prompt_id,
        "frame_index": prompt.target_frame,
        "pixel_box": box.model_dump(mode="json"),
        "intended_target": prompt.part,
        "boxes": [[[box.x1 / width, box.y1 / height], [box.x2 / width, box.y2 / height]]],
        "fg_points": [],
        "bg_points": [[p.x / width, p.y / height] for p in prompt.background_points],
    }


# ------------------------------------------------------------------------------- planning


class FramePlan(VersionedModel):
    view: str
    reference_frame: int
    target_frame: int
    spheres: dict[str, Sphere]
    prompts: tuple[PromptSpec, ...]


class SearchPlan(VersionedModel):
    manifest_kind: Literal["seed_search_plan"]
    view: str
    view_id: str
    proxy: ArtifactFingerprint
    proxy_dimensions: tuple[int, int]
    truth_set: ArtifactFingerprint
    frames: tuple[FramePlan, ...]
    prompt_count: int
    grid: dict[str, Any]
    claim_boundaries: tuple[str, ...]


def proxy_for_view(repository_root: Path, view: str) -> tuple[ArtifactFingerprint, tuple[int, int]]:
    from .anchor_frames_for_view import clip_config_for
    from .schemas import G2PreprocessingManifest

    config_path = (
        Path("configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_g2.json")
        if view == REFERENCE_VIEW
        else clip_config_for(view)
    )
    config = G2PreprocessingManifest.model_validate_json(
        (repository_root / config_path).read_text(encoding="utf-8")
    )
    proxy = next(p for p in config.proxies if p.view_id == view_id_for(view))
    return (
        ArtifactFingerprint(
            uri=proxy.proxy_uri, sha256=proxy.checksum_sha256, source="approved_config"
        ),
        (proxy.dimensions.width, proxy.dimensions.height),
    )


def plan_search(
    repository_root: Path, truth: SeedTruthSet, truth_path: Path, *, context: GeometryContext
) -> SearchPlan:
    """Prompts for every C10379 truth frame from the other views' masks (never the human's)."""
    proxy, (width, height) = proxy_for_view(repository_root, REFERENCE_VIEW)
    shape = (height, width)
    frames: list[FramePlan] = []
    total = 0
    for frame in truth.frames_c10379:
        counter = itertools.count(1)
        spheres: dict[str, Sphere] = {}
        for part in TARGETS:
            sphere = context.sphere_from_views(part, frame, TRANSFER_VIEWS)
            if sphere.source == "unavailable":
                sphere = context.self_prior(REFERENCE_VIEW, frame, part)
            spheres[part] = sphere
        prompts: list[PromptSpec] = []
        for part in TARGETS:
            has_truth = any(
                e.analysis_frame_index == frame
                and e.part == part
                and e.view == REFERENCE_VIEW
                and e.role in ("positive", "hidden")
                for e in truth.entries
            )
            if not has_truth:
                continue
            prompts += build_prompts(
                context,
                view=REFERENCE_VIEW,
                reference_frame=frame,
                part=part,
                sphere=spheres[part],
                other_spheres=spheres,
                shape=shape,
                counter=counter,
            )
        total += len(prompts)
        frames.append(
            FramePlan(
                view=REFERENCE_VIEW,
                reference_frame=frame,
                target_frame=frame,
                spheres=spheres,
                prompts=tuple(prompts),
            )
        )
    return SearchPlan(
        manifest_kind="seed_search_plan",
        view=REFERENCE_VIEW,
        view_id=view_id_for(REFERENCE_VIEW),
        proxy=proxy,
        proxy_dimensions=(width, height),
        truth_set=_fingerprint(truth_path, repository_root),
        frames=tuple(frames),
        prompt_count=total,
        grid={
            "margins": list(MARGINS),
            "negative_sets": list(NEGATIVE_SETS),
            "box_modes": list(BOX_MODES),
            "second_box_margin": TWO_BOX_SECOND_MARGIN,
            "picks": list(PICKS),
        },
        claim_boundaries=CLAIM_BOUNDARIES,
    )


# -------------------------------------------------------------------------------- decoding


def make_decoder(
    repository_root: Path, proxy: ArtifactFingerprint, results_dir: Path, *, device: str
):
    from .muggled_calibration_web import WorkerClient
    from .muggled_smoke import DEFAULT_MODEL, MUGGLED_SAM_PYTHON, MUGGLED_SAM_SOURCE

    proxy_path = (repository_root / proxy.uri).resolve()
    if sha256_file(proxy_path) != proxy.sha256:
        raise ValueError(f"proxy checksum changed: {proxy_path}")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(MUGGLED_SAM_SOURCE), environment["PYTHONPATH"]]
        if environment.get("PYTHONPATH")
        else [str(MUGGLED_SAM_SOURCE)]
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    return WorkerClient(
        [
            str(MUGGLED_SAM_PYTHON),
            str(Path(__file__).with_name("muggled_calibration_worker.py")),
            "--serve-jsonl",
            "--proxy",
            str(proxy_path),
            "--model",
            str(DEFAULT_MODEL.resolve()),
            "--results-directory",
            str(results_dir),
            "--device",
            device,
        ],
        environment=environment,
        stderr_path=results_dir.parent / "decode_worker.stderr.log",
    )


def decode_prompts(
    decoder: Any,
    prompts: Sequence[PromptSpec],
    shape: tuple[int, int],
    *,
    batch: int = 40,
    log: Any = None,
) -> dict[str, dict[str, Any]]:
    """`prompt_id -> decoder_result` for every prompt; frames are encoded once by the worker."""
    decoded: dict[str, dict[str, Any]] = {}
    ordered = sorted(prompts, key=lambda p: p.target_frame)
    for frame in sorted({p.target_frame for p in ordered}):
        decoder.request("frame_preview", {"frame_index": frame}, timeout=600)
    for start in range(0, len(ordered), batch):
        chunk = ordered[start : start + batch]
        response = decoder.request(
            "batch_decode", {"prompts": [decode_request(p, shape) for p in chunk]}, timeout=1800
        )
        for item in response["decoded"]:
            decoded[item["candidate_id"]] = item["decoder_result"]
        if log is not None:
            log(f"decoded {min(start + batch, len(ordered))}/{len(ordered)} prompts")
    return decoded


def run_decode(
    repository_root: Path, plan: SearchPlan, *, output_dir: Path, device: str = "cuda:0"
) -> Path:
    results_dir = output_dir / "results"
    decoder = make_decoder(repository_root, plan.proxy, results_dir, device=device)
    started = time.monotonic()
    shape = (plan.proxy_dimensions[1], plan.proxy_dimensions[0])
    try:
        prompts = [p for frame in plan.frames for p in frame.prompts]
        decoded = decode_prompts(decoder, prompts, shape, log=lambda m: print(m, flush=True))
    finally:
        decoder.close()
    payload = {
        "view": plan.view,
        "prompt_count": len(decoded),
        "elapsed_seconds": time.monotonic() - started,
        "decoded": decoded,
    }
    output = output_dir / "decode_result.json"
    output.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return output


# ------------------------------------------------------------------------------- exemplars


DINOV2_SCRIPT = r"""
import json, sys
import numpy as np, torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
spec = json.load(open(sys.argv[1]))
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
model = AutoModel.from_pretrained("facebook/dinov2-small").eval()
out = {}
with torch.no_grad():
    items = spec["crops"]
    for start in range(0, len(items), 32):
        chunk = items[start:start + 32]
        images = []
        for item in chunk:
            image = Image.open(item["image"]).convert("RGB")
            x0, y0, x1, y1 = item["box"]
            images.append(image.crop((x0, y0, x1, y1)).resize((224, 224)))
        inputs = processor(images=images, return_tensors="pt")
        features = model(**inputs).pooler_output
        features = torch.nn.functional.normalize(features, dim=-1).cpu().numpy()
        for item, vector in zip(chunk, features):
            out[item["key"]] = vector.tolist()
np.save(sys.argv[2], out, allow_pickle=True)
print(len(out))
"""


def _crop_box(mask: np.ndarray, pad: float = 0.15) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    w, h = x1 - x0, y1 - y0
    side = max(w, h) * (1 + 2 * pad)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hgt, wid = mask.shape
    bx0, by0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    bx1, by1 = int(min(wid, cx + side / 2)), int(min(hgt, cy + side / 2))
    if bx1 - bx0 < 4 or by1 - by0 < 4:
        return None
    return bx0, by0, bx1, by1


def compute_embeddings(
    crops: list[dict[str, Any]], *, work_dir: Path, python: Path
) -> dict[str, np.ndarray] | None:
    """DINOv2-small pooled embeddings on CPU in the MuggledSAM interpreter; None if unavailable."""
    if not crops:
        return {}
    work_dir.mkdir(parents=True, exist_ok=True)
    spec = work_dir / "dinov2_crops.json"
    spec.write_text(json.dumps({"crops": crops}), encoding="utf-8")
    script = work_dir / "dinov2_embed.py"
    script.write_text(DINOV2_SCRIPT, encoding="utf-8")
    output = work_dir / "dinov2_embeddings.npy"
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    completed = subprocess.run(
        [str(python), str(script), str(spec), str(output)],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    (work_dir / "dinov2_embed.log").write_text(
        completed.stdout + completed.stderr, encoding="utf-8"
    )
    if completed.returncode != 0:
        return None
    loaded = np.load(output, allow_pickle=True).item()
    return {k: np.asarray(v, dtype=np.float64) for k, v in loaded.items()}


# -------------------------------------------------------------------------------- scoring


class StrategyKey(VersionedModel):
    margin: float
    negatives: str
    boxes: str
    pick: str

    @property
    def name(self) -> str:
        return f"m{self.margin:.2f}|{self.negatives}|{self.boxes}|{self.pick}"


def strategy_grid(picks: Sequence[str] = PICKS) -> list[StrategyKey]:
    return [
        StrategyKey(margin=m, negatives=n, boxes=b, pick=p)
        for m in MARGINS
        for n in NEGATIVE_SETS
        for b in BOX_MODES
        for p in picks
    ]


@dataclass
class Candidate:
    prompt: PromptSpec
    index: int
    mask_uri: str
    score: float
    area: int
    sha256: str
    mask: np.ndarray


def load_candidates(
    plan: SearchPlan, decoded: Mapping[str, Mapping[str, Any]], results_root: Path
) -> dict[str, list[Candidate]]:
    """`prompt_id -> candidates` with masks loaded (results dir holds the worker's PNGs)."""
    out: dict[str, list[Candidate]] = {}
    for frame in plan.frames:
        for prompt in frame.prompts:
            result = decoded.get(prompt.prompt_id)
            if result is None:
                continue
            items = []
            for option in result["candidates"]:
                path = results_root / option["mask_uri"]
                mask = mask_cache.decode_mask_png(path)
                items.append(
                    Candidate(
                        prompt=prompt,
                        index=int(option["candidate_index"]),
                        mask_uri=option["mask_uri"],
                        score=float(option["iou_score"]),
                        area=int(mask.sum()),
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        mask=mask,
                    )
                )
            out[prompt.prompt_id] = items
    return out


def pool_for_strategy(
    prompts: Sequence[PromptSpec],
    candidates: Mapping[str, list[Candidate]],
    key: StrategyKey,
) -> list[Candidate]:
    """Candidates a strategy chooses among: its margin box (and the 0.60 box when `two`)."""

    def matching(margin: float) -> list[PromptSpec]:
        exact = [
            p for p in prompts if abs(p.margin - margin) < 1e-9 and p.negatives == key.negatives
        ]
        if exact:
            return exact
        # A negative set that added no point collapsed into `none`: fall back to it.
        return [p for p in prompts if abs(p.margin - margin) < 1e-9 and p.negatives == "none"]

    chosen = matching(key.margin)
    if key.boxes == "two" and abs(key.margin - TWO_BOX_SECOND_MARGIN) > 1e-9:
        chosen = chosen + matching(TWO_BOX_SECOND_MARGIN)
    pool: list[Candidate] = []
    for prompt in chosen:
        pool.extend(candidates.get(prompt.prompt_id, []))
    return [c for c in pool if c.area > 0]


def pick_candidate(
    pool: Sequence[Candidate],
    pick: str,
    *,
    embeddings: Mapping[str, np.ndarray] | None = None,
    exemplars: Sequence[np.ndarray] | None = None,
) -> Candidate | None:
    if not pool:
        return None
    if pick == "highest_score":
        return max(pool, key=lambda c: (c.score, -c.index))
    if pick == "largest":
        return max(pool, key=lambda c: (c.area, c.score))
    if pick == "smallest":
        return min(pool, key=lambda c: (c.area, -c.score))
    if pick == "exemplar":
        if not embeddings or not exemplars:
            return None
        library = np.stack(exemplars)

        def similarity(c: Candidate) -> float:
            vector = embeddings.get(f"cand:{c.prompt.prompt_id}:{c.index}")
            return float((library @ vector).max()) if vector is not None else -1.0

        return max(pool, key=lambda c: (similarity(c), c.score))
    raise ValueError(pick)


class CellResult(VersionedModel):
    reference_frame: int
    part: str
    truth_role: TruthRole
    truth_area: int | None
    sphere_source: str
    # strategy name -> IoU (or false-positive area on a hidden cell)
    iou: dict[str, float]
    exact_match: dict[str, bool]
    chosen_area: dict[str, int]
    chosen_candidate: dict[str, str]
    prompt_count: int


class PartSummary(VersionedModel):
    part: str
    frames_scored: int
    winner: str
    winner_mean_iou_all_frames: float
    runner_up: str
    runner_up_gap: float
    held_out_iou_mean: float | None
    held_out_iou_min: float | None
    held_out_per_frame: dict[str, float]
    winner_per_frame_iou: dict[str, float]
    exact_match_fraction: float
    winners_per_split: dict[str, str]
    passes_transfer_gate: bool
    best_by_pick: dict[str, float]
    best_by_negatives: dict[str, float]
    best_by_margin: dict[str, float]
    best_by_boxes: dict[str, float]
    hidden_false_positive_px: dict[str, int] = Field(default_factory=dict)


class SearchReport(VersionedModel):
    manifest_kind: Literal["seed_search_report"]
    plan: ArtifactFingerprint
    decode_result: ArtifactFingerprint
    truth_set: ArtifactFingerprint
    exemplar_arm: Literal["run", "not_run"]
    exemplar_note: str | None
    strategies: tuple[str, ...]
    anchor_frames: tuple[int, ...]
    always_fit_frames: tuple[int, ...]
    cells: tuple[CellResult, ...]
    parts: dict[str, PartSummary]
    transfer_gate_iou: float
    generated_at: datetime
    claim_boundaries: tuple[str, ...]


def score_search(
    repository_root: Path,
    *,
    plan: SearchPlan,
    plan_path: Path,
    decode_path: Path,
    truth: SeedTruthSet,
    truth_path: Path,
    output_dir: Path,
    exemplar_python: Path | None,
) -> SearchReport:
    decoded = json.loads(decode_path.read_text(encoding="utf-8"))["decoded"]
    results_root = output_dir / "results"
    candidates = load_candidates(plan, decoded, results_root)
    truth_masks: dict[tuple[int, str], tuple[TruthMask, np.ndarray | None]] = {}
    for entry in truth.entries:
        if entry.view == REFERENCE_VIEW and entry.role in ("positive", "hidden"):
            mask = (
                mask_cache.decode_mask_png(repository_root / entry.mask.uri)
                if entry.mask is not None
                else None
            )
            truth_masks[(entry.analysis_frame_index, entry.part)] = (entry, mask)

    # Exemplar arm: DINOv2-small crops of every candidate and every human mask, CPU.
    exemplar_note = None
    embeddings: dict[str, np.ndarray] | None = None
    frames_dir = results_root / "frames"
    crops: list[dict[str, Any]] = []
    for prompt_id, items in candidates.items():
        for c in items:
            box = _crop_box(c.mask)
            image = frames_dir / f"frame-{c.prompt.target_frame:06d}.jpg"
            if box is not None and image.is_file():
                crops.append(
                    {"key": f"cand:{prompt_id}:{c.index}", "image": str(image), "box": box}
                )
    for (frame, part), (entry, mask) in truth_masks.items():
        if mask is None:
            continue
        box = _crop_box(mask)
        image = frames_dir / f"frame-{frame:06d}.jpg"
        if box is not None and image.is_file():
            crops.append({"key": f"truth:{frame}:{part}", "image": str(image), "box": box})
    if exemplar_python is None:
        exemplar_note = "exemplar arm skipped: no interpreter with transformers given"
    else:
        embeddings = compute_embeddings(
            crops, work_dir=output_dir / "exemplar", python=exemplar_python
        )
        if embeddings is None:
            exemplar_note = (
                "exemplar arm not run: DINOv2-small could not be loaded in the given "
                "interpreter (see exemplar/dinov2_embed.log)"
            )
    picks = list(PICKS) if embeddings else [p for p in PICKS if p != "exemplar"]
    strategies = strategy_grid(picks)
    names = tuple(s.name for s in strategies)

    cells: list[CellResult] = []
    for frame_plan in plan.frames:
        frame = frame_plan.reference_frame
        for part in TARGETS:
            key = (frame, part)
            if key not in truth_masks:
                continue
            entry, truth_mask = truth_masks[key]
            prompts = [p for p in frame_plan.prompts if p.part == part]
            exemplars = None
            if embeddings:
                exemplars = [
                    embeddings[f"truth:{f}:{p}"]
                    for (f, p) in truth_masks
                    if p == part and f != frame and f"truth:{f}:{p}" in embeddings
                ]
            ious: dict[str, float] = {}
            exact: dict[str, bool] = {}
            areas: dict[str, int] = {}
            chosen_ids: dict[str, str] = {}
            for strategy in strategies:
                pool = pool_for_strategy(prompts, candidates, strategy)
                chosen = pick_candidate(
                    pool, strategy.pick, embeddings=embeddings, exemplars=exemplars
                )
                if chosen is None:
                    ious[strategy.name] = 0.0
                    exact[strategy.name] = False
                    areas[strategy.name] = 0
                    chosen_ids[strategy.name] = ""
                    continue
                if entry.role == "hidden":
                    ious[strategy.name] = float(chosen.area)
                    exact[strategy.name] = False
                else:
                    assert truth_mask is not None
                    ious[strategy.name] = iou(chosen.mask, truth_mask)
                    exact[strategy.name] = (
                        bool(entry.mask is not None and chosen.sha256 == entry.mask.sha256)
                        or ious[strategy.name] >= 0.999
                    )
                areas[strategy.name] = chosen.area
                chosen_ids[strategy.name] = f"{chosen.prompt.prompt_id}:{chosen.index}"
            cells.append(
                CellResult(
                    reference_frame=frame,
                    part=part,
                    truth_role=entry.role,
                    truth_area=entry.area_px,
                    sphere_source=frame_plan.spheres[part].source,
                    iou=ious,
                    exact_match=exact,
                    chosen_area=areas,
                    chosen_candidate=chosen_ids,
                    prompt_count=len(prompts),
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
    parts = {
        part: summarize_part(
            part, [c for c in cells if c.part == part], names, anchor_frames, always_fit
        )
        for part in TARGETS
    }
    return SearchReport(
        manifest_kind="seed_search_report",
        plan=_fingerprint(plan_path, repository_root),
        decode_result=_fingerprint(decode_path, repository_root),
        truth_set=_fingerprint(truth_path, repository_root),
        exemplar_arm="run" if embeddings else "not_run",
        exemplar_note=exemplar_note,
        strategies=names,
        anchor_frames=anchor_frames,
        always_fit_frames=always_fit,
        cells=tuple(cells),
        parts=parts,
        transfer_gate_iou=TRANSFER_PASS_IOU,
        generated_at=datetime.now(UTC),
        claim_boundaries=CLAIM_BOUNDARIES,
    )


def _mean_over(cells: Sequence[CellResult], frames: Iterable[int], name: str) -> float | None:
    values = [
        c.iou[name]
        for c in cells
        if c.reference_frame in set(frames) and c.truth_role == "positive" and name in c.iou
    ]
    return float(np.mean(values)) if values else None


def summarize_part(
    part: str,
    cells: Sequence[CellResult],
    names: Sequence[str],
    anchor_frames: Sequence[int],
    always_fit: Sequence[int],
) -> PartSummary:
    scored = [c for c in cells if c.truth_role == "positive"]
    hidden = [c for c in cells if c.truth_role == "hidden"]
    all_frames = [c.reference_frame for c in scored]

    def winner_on(frames: Iterable[int]) -> tuple[str, float, str, float]:
        means = {n: _mean_over(scored, frames, n) for n in names}
        ranked = sorted(
            ((v, n) for n, v in means.items() if v is not None), key=lambda t: (-t[0], t[1])
        )
        if not ranked:
            return names[0], 0.0, names[0], 0.0
        best_v, best_n = ranked[0]
        second_v, second_n = ranked[1] if len(ranked) > 1 else ranked[0]
        return best_n, best_v, second_n, best_v - second_v

    winner, winner_mean, runner_up, gap = winner_on(all_frames)
    anchors = [f for f in anchor_frames if f in all_frames]
    held_out: dict[int, list[float]] = {}
    winners_per_split: dict[str, str] = {}
    for i in range(len(anchors)):
        held = {anchors[(i + k) % len(anchors)] for k in range(HELD_OUT_PER_SPLIT)}
        fit = [f for f in all_frames if f not in held or f in always_fit]
        fit_winner, _, _, _ = winner_on(fit)
        winners_per_split[",".join(str(h) for h in sorted(held))] = fit_winner
        for cell in scored:
            if cell.reference_frame in held and cell.reference_frame not in always_fit:
                held_out.setdefault(cell.reference_frame, []).append(cell.iou[fit_winner])
    per_frame_held = {str(f): float(np.mean(v)) for f, v in sorted(held_out.items())}
    held_values = list(per_frame_held.values())
    exact = [c.exact_match[winner] for c in scored]

    def best_by(attribute: int) -> dict[str, float]:
        out: dict[str, float] = {}
        for name in names:
            level = name.split("|")[attribute]
            value = _mean_over(scored, all_frames, name)
            if value is not None:
                out[level] = max(out.get(level, 0.0), value)
        return out

    return PartSummary(
        part=part,
        frames_scored=len(scored),
        winner=winner,
        winner_mean_iou_all_frames=winner_mean,
        runner_up=runner_up,
        runner_up_gap=gap,
        held_out_iou_mean=float(np.mean(held_values)) if held_values else None,
        held_out_iou_min=float(np.min(held_values)) if held_values else None,
        held_out_per_frame=per_frame_held,
        winner_per_frame_iou={str(c.reference_frame): c.iou[winner] for c in scored},
        exact_match_fraction=float(np.mean(exact)) if exact else 0.0,
        winners_per_split=winners_per_split,
        passes_transfer_gate=bool(held_values) and float(np.mean(held_values)) >= TRANSFER_PASS_IOU,
        best_by_pick=best_by(3),
        best_by_negatives=best_by(1),
        best_by_margin=best_by(0),
        best_by_boxes=best_by(2),
        hidden_false_positive_px={str(c.reference_frame): int(c.iou[winner]) for c in hidden},
    )


def search_table(report: SearchReport) -> str:
    lines = [
        "| part | frames | winner (margin / negatives / boxes / pick) | mean IoU all frames | "
        "runner-up | gap | held-out mean | held-out min | exact-match fraction | gate >= 0.6 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for part, s in report.parts.items():
        lines.append(
            f"| {part} | {s.frames_scored} | `{s.winner}` | {s.winner_mean_iou_all_frames:.3f} | "
            f"`{s.runner_up}` | {s.runner_up_gap:.3f} | "
            f"{'-' if s.held_out_iou_mean is None else f'{s.held_out_iou_mean:.3f}'} | "
            f"{'-' if s.held_out_iou_min is None else f'{s.held_out_iou_min:.3f}'} | "
            f"{s.exact_match_fraction:.2f} | {'pass' if s.passes_transfer_gate else 'FAIL'} |"
        )
    lines.append("")
    lines.append(
        "Per-frame IoU of the winner (C10379 frame: IoU); `held-out` = IoU of each split's "
        "fitting-set winner on the frames it did not see:"
    )
    lines.append("")
    for part, s in report.parts.items():
        cells = ", ".join(f"{f} {v:.2f}" for f, v in s.winner_per_frame_iou.items())
        held = ", ".join(f"{f} {v:.2f}" for f, v in s.held_out_per_frame.items())
        lines.append(f"- **{part}** winner: {cells}")
        lines.append(f"  held-out: {held}")
        if s.hidden_false_positive_px:
            lines.append(f"  hidden-cell false-positive px: {s.hidden_false_positive_px}")
    lines.append("")
    lines.append("Best mean IoU over all frames by one grid axis (the other axes free):")
    lines.append("")
    lines.append("| part | pick | negatives | margin | boxes |")
    lines.append("|---|---|---|---|---|")
    for part, s in report.parts.items():

        def fmt(d: dict[str, float]) -> str:
            return ", ".join(f"{k} {v:.3f}" for k, v in sorted(d.items(), key=lambda t: -t[1]))

        lines.append(
            f"| {part} | {fmt(s.best_by_pick)} | {fmt(s.best_by_negatives)} | "
            f"{fmt(s.best_by_margin)} | {fmt(s.best_by_boxes)} |"
        )
    lines.append("")
    lines.append(
        f"Exemplar arm: {report.exemplar_arm}"
        + (f" ({report.exemplar_note})" if report.exemplar_note else "")
        + f". Anchor frames {list(report.anchor_frames)}; always in the fitting set "
        f"{list(report.always_fit_frames)}; {HELD_OUT_PER_SPLIT} anchor frames held out per "
        f"split, rotated over {len(report.anchor_frames)} splits. "
        + " ".join(report.claim_boundaries)
    )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------- CLI


def _root(args: argparse.Namespace) -> Path:
    return Path(args.repository_root).resolve()


def truth_set_main() -> None:
    parser = argparse.ArgumentParser(description="Assemble the C10379 (+ e3) human mask truth set.")
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT / "truth_set.json")
    args = parser.parse_args()
    root = _root(args)
    truth = build_truth_set(root)
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(truth.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(json.dumps(truth.counts, indent=1))
    print(json.dumps(truth.counts_by_part, indent=1))
    print(f"frames (C10379): {list(truth.frames_c10379)}")
    print(f"excluded: {len(truth.excluded)} -> {output}")


def search_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan", help="build the prompt grid on C10379 (CPU)")
    decode = commands.add_parser("decode", help="decode the planned prompts (GPU)")
    decode.add_argument("--device", default="cuda:0")
    score = commands.add_parser("score", help="score strategies against the truth set (CPU)")
    score.add_argument(
        "--exemplar-python",
        type=Path,
        default=None,
        help="interpreter with torch+transformers for the DINOv2-small exemplar arm (CPU)",
    )
    args = parser.parse_args()
    root = _root(args)
    output_root = root / args.output_root
    search_dir = output_root / "search"
    truth_path = output_root / "truth_set.json"
    plan_path = search_dir / "plan.json"
    if args.command == "plan":
        truth = load_truth_set(truth_path)
        context = GeometryContext.load(root)
        plan = plan_search(root, truth, truth_path, context=context)
        search_dir.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(plan.model_dump_json(indent=1) + "\n", encoding="utf-8")
        for frame in plan.frames:
            sources = {p: s.source for p, s in frame.spheres.items()}
            print(f"frame {frame.reference_frame}: {len(frame.prompts)} prompts, spheres {sources}")
        print(f"{plan.prompt_count} prompts -> {plan_path}")
    elif args.command == "decode":
        plan = SearchPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
        output = run_decode(root, plan, output_dir=search_dir, device=args.device)
        print(f"decoded -> {output}")
    else:
        plan = SearchPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
        truth = load_truth_set(truth_path)
        report = score_search(
            root,
            plan=plan,
            plan_path=plan_path,
            decode_path=search_dir / "decode_result.json",
            truth=truth,
            truth_path=truth_path,
            output_dir=search_dir,
            exemplar_python=args.exemplar_python,
        )
        (search_dir / "search_report.json").write_text(
            report.model_dump_json(indent=1) + "\n", encoding="utf-8"
        )
        table = search_table(report)
        (search_dir / "search_table.md").write_text(table, encoding="utf-8")
        print(table)


if __name__ == "__main__":
    sys.exit(search_main())
