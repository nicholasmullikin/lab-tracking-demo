"""Per-part visual hulls from the calibrated static views' SAM3 masks (space carving).

A 5 mm voxel grid covers the table working volume (dataset hand positions plus a margin,
cut at the fitted table plane).  Every static camera is constant, so each voxel's projection
into each view (distortion applied) is computed once; per frame and part the grid is carved
by the views that have a mask for that part and are not inside an active cross-view
disagreement episode.  The surviving voxels give a hull count, centroid and bounding box per
frame, sparse voxel indices at 1 fps for the viewer, and a `hull_projection` mask per view
that is compared with the view's own SAM3 mask (IoU, area ratio, "mask larger than hull").
Runs of disagreement become `hull_disagreement` episodes; nothing is substituted.

The hull is the intersection of silhouettes, so it is an upper bound on the object volume and
inherits every mask's errors; it is geometry-only comparison evidence, not accuracy.  Learned
multi-view fusion (MVDet-style BEV training) and homography-to-BEV identity handoff are
recorded as not run: the first would be training, the second solves a multi-person problem
this single-person recording does not pose.  CC BY-NC 4.0 applies to the dataset assets.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from pathlib import Path

import cv2
import numpy as np

from .assembly101_camera_fit import PoseMembers
from .assembly101_clock_offset import is_ego
from .digest_cache import sha256_file
from .four_part_contract import TARGETS
from .multiview_consensus import (
    MANIFEST_NAME as CONSENSUS_MANIFEST_NAME,
)
from .multiview_consensus import OUTPUT_ROOT as CONSENSUS_ROOT
from .multiview_consensus import (
    ViewRun,
    episodes_from_errors,
    load_consensus,
    load_view_run,
    merge_intervals,
    relative_uri,
)
from .multiview_geometry import CameraRig, TablePlane
from .multiview_schemas import (
    MULTIVIEW_CLAIM_BOUNDARIES,
    ConsensusEpisodeRules,
    DisagreementEpisode,
    HullPartSummary,
    HullViewComparison,
    ProposedValidityInterval,
    VisualHullManifest,
)
from .multiview_seed_transfer import (
    REFERENCE_VIEW,
    fit_first_minute_plane,
    proxy_to_raw_scale,
)
from .schemas import ArtifactFingerprint

OUTPUT_ROOT = Path("runs/multiview-visual-hull-first-minute")
VOXEL_MM = 5.0
VOLUME_MARGIN_MM = 150.0
VOXEL_FPS = 1
HULL_RULES = ConsensusEpisodeRules(
    reprojection_filter_px=30.0,
    min_views=2,
    disagreement_threshold_px=0.5,
    min_episode_frames=5,
    description=(
        "disagreement_threshold_px is a flag threshold here (a frame is flagged 1.0 or 0.0). "
        "A view disagrees with the hull on a frame when the IoU between its SAM3 mask and the "
        "hull projection is below 0.3 and at least 0.15 below that frame's median IoU over the "
        "views, or its mask area exceeds max(2.0, 1.5 x the frame's median ratio) times the hull "
        "projection area (a hull carved from imperfect silhouettes is over-carved, so only a "
        "view that disagrees clearly more than the others is flagged); >= 5 consecutive such "
        "frames form a hull_disagreement episode. Views inside an active "
        "multiview_disagreement episode for the part do not carve that frame."
    ),
)
IOU_FLOOR = 0.3
AREA_RATIO_CEILING = 2.0
NOT_RUN = {
    "mvdet_style_learned_fusion": (
        "not_run: MVDet-style multi-view BEV fusion needs scene-specific training data "
        "(Wildtrack/MultiviewX-style ground-plane labels) and would be training a model, which "
        "the plan forbids; the carved hull is the training-free equivalent."
    ),
    "homography_to_bev_id_handoff": (
        "not_run: homography-to-BEV identity handoff is a multi-person tracking device; this "
        "recording has one person and four rigid parts, so there is no identity to hand off."
    ),
}


def _fingerprint(path: Path, repository_root: Path) -> ArtifactFingerprint:
    return ArtifactFingerprint(
        uri=relative_uri(path, repository_root), sha256=sha256_file(path), source="measured"
    )


class VoxelGrid:
    """Axis-aligned world-mm grid with per-static-view projected pixel indices."""

    def __init__(
        self,
        rig: CameraRig,
        plane: TablePlane,
        members: PoseMembers,
        *,
        pose_frames: range,
        voxel_mm: float = VOXEL_MM,
        margin_mm: float = VOLUME_MARGIN_MM,
    ) -> None:
        joints: list[np.ndarray] = []
        for frame in pose_frames:
            key = str(frame)
            landmarks = members.landmarks3d.get(key)
            if landmarks is None:
                continue
            for hand in ("0", "1"):
                if float(members.confidences[key][hand]) >= 0.5:
                    joints.append(np.asarray(landmarks[hand], dtype=np.float64))
        cloud = np.concatenate(joints)
        low = cloud.min(axis=0) - margin_mm
        high = cloud.max(axis=0) + margin_mm
        self.voxel_mm = voxel_mm
        self.origin = np.floor(low / voxel_mm) * voxel_mm
        self.shape = tuple(int(v) for v in np.ceil((high - self.origin) / voxel_mm))
        indices = np.stack(
            np.meshgrid(*[np.arange(n) for n in self.shape], indexing="ij"), axis=-1
        ).reshape(-1, 3)
        centres = self.origin + (indices + 0.5) * voxel_mm
        above = plane.signed_distance(centres) >= -voxel_mm
        self.indices = indices[above].astype(np.int16)
        self.centres = centres[above]
        self.volume_bounds_source = (
            f"dataset hand joints (confidence >= 0.5) over pose frames "
            f"[{pose_frames.start}, {pose_frames.stop}) plus {margin_mm:.0f} mm, cut at the "
            "table plane"
        )
        self.pixels: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for view in rig.static_views:
            self.pixels[view] = self._project(rig, view, self.centres, None)

    @staticmethod
    def _project(
        rig: CameraRig, view: str, centres: np.ndarray, pose_frame: int | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Proxy-pixel column/row per voxel; -1 where the voxel is outside or behind."""
        width, height = rig.image_size(view)
        scale = proxy_to_raw_scale(view)
        proxy_w, proxy_h = int(round(width / scale)), int(round(height / scale))
        depth = rig.depth(view, centres, pose_frame)
        raw = rig.project(view, centres, pose_frame)
        proxy = raw / scale
        finite = np.isfinite(proxy).all(axis=1) & (depth > 0)
        col = np.full(centres.shape[0], -1, dtype=np.int32)
        row = np.full(centres.shape[0], -1, dtype=np.int32)
        clipped = np.clip(np.nan_to_num(proxy, nan=-1.0), -1.0, 1.0e6)
        c = np.floor(clipped[:, 0]).astype(np.int32)
        r = np.floor(clipped[:, 1]).astype(np.int32)
        inside = finite & (c >= 0) & (c < proxy_w) & (r >= 0) & (r < proxy_h)
        col[inside] = c[inside]
        row[inside] = r[inside]
        return col, row

    def carve(
        self, masks: Mapping[str, np.ndarray], extra: Mapping[str, tuple[np.ndarray, np.ndarray]]
    ) -> np.ndarray:
        """Boolean occupancy over `self.centres` after carving with every given view mask."""
        alive = np.ones(self.centres.shape[0], dtype=bool)
        for view, mask in masks.items():
            col, row = self.pixels[view] if view in self.pixels else extra[view]
            candidates = np.nonzero(alive)[0]
            c, r = col[candidates], row[candidates]
            visible = (c >= 0) & (r >= 0)
            keep = np.zeros(candidates.size, dtype=bool)
            keep[visible] = mask[r[visible], c[visible]]
            alive[candidates[~keep]] = False
            if not alive.any():
                break
        return alive


def hull_projection_mask(
    rig: CameraRig,
    view: str,
    centres: np.ndarray,
    shape: tuple[int, int],
    voxel_mm: float,
    pose_frame: int | None = None,
) -> np.ndarray:
    """Rasterise occupied voxel centres into `view` and close the gaps between voxels."""
    height, width = shape
    mask = np.zeros(shape, dtype=np.uint8)
    if centres.shape[0] == 0:
        return mask.astype(bool)
    scale = proxy_to_raw_scale(view)
    depth = rig.depth(view, centres, pose_frame)
    proxy = rig.project(view, centres, pose_frame) / scale
    finite = np.isfinite(proxy).all(axis=1) & (depth > 0)
    c = np.floor(np.clip(proxy[finite, 0], -1, width)).astype(np.int32)
    r = np.floor(np.clip(proxy[finite, 1], -1, height)).astype(np.int32)
    inside = (c >= 0) & (c < width) & (r >= 0) & (r < height)
    mask[r[inside], c[inside]] = 1
    if depth[finite].size:
        # One voxel spans roughly f * voxel / depth proxy pixels; dilate by that footprint.
        focal = float(rig.camera(view).intrinsic_matrix[0][0]) / scale
        footprint = int(np.clip(np.ceil(focal * voxel_mm / np.median(depth[finite])), 1, 15))
        kernel = np.ones((2 * footprint + 1, 2 * footprint + 1), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, np.ones((footprint + 1, footprint + 1), dtype=np.uint8))
    return mask.astype(bool)


def active_episode_frames(
    episodes: tuple[DisagreementEpisode, ...], frame_count: int
) -> dict[tuple[str, str], np.ndarray]:
    flags: dict[tuple[str, str], np.ndarray] = {}
    for episode in episodes:
        key = (episode.view, episode.target)
        array = flags.setdefault(key, np.zeros(frame_count, dtype=bool))
        array[episode.start_frame : episode.end_frame_exclusive] = True
    return flags


def build_visual_hull(
    repository_root: Path,
    *,
    output_root: Path = OUTPUT_ROOT,
    consensus_root: Path = CONSENSUS_ROOT,
    voxel_mm: float = VOXEL_MM,
    frame_count: int | None = None,
    ego_views: tuple[str, ...] = (),
) -> VisualHullManifest:
    started = time.monotonic()
    repository_root = repository_root.resolve()
    consensus_path = repository_root / consensus_root / CONSENSUS_MANIFEST_NAME
    consensus = load_consensus(consensus_path)
    frame_count = frame_count or consensus.frame_count
    rig = CameraRig.load(repository_root)
    members = PoseMembers(repository_root)
    plane = fit_first_minute_plane(rig, members)
    start_pose = rig.pose_frame(REFERENCE_VIEW, 0)
    grid = VoxelGrid(
        rig,
        plane,
        members,
        pose_frames=range(start_pose, start_pose + 2 * frame_count),
        voxel_mm=voxel_mm,
    )
    runs: dict[str, ViewRun] = {
        source.view: load_view_run(
            repository_root,
            Path(source.run_directory_uri),
            view=source.view,
            frame_count=frame_count,
        )
        for source in consensus.sources
        if not source.is_ego or source.view in ego_views
    }
    static_views = tuple(v for v in runs if not is_ego(v))
    flags = active_episode_frames(consensus.episodes, frame_count)
    with np.load(repository_root / consensus_root / "consensus_points.npz") as archive:
        ego_gate = {
            view: archive[f"observed/{TARGETS[0]}/{view}"]  # observed only where the gate passed
            for view in ego_views
            if f"observed/{TARGETS[0]}/{view}" in archive
        }

    root = repository_root / output_root
    projection_root = root / "hull_projection_masks"
    root.mkdir(parents=True, exist_ok=True)
    per_frame_path = root / "per_frame.jsonl"
    voxels_1fps: dict[str, np.ndarray] = {}
    counts = {t: np.full(frame_count, np.nan) for t in TARGETS}
    views_used = {t: np.zeros(frame_count, dtype=np.int64) for t in TARGETS}
    iou = {t: {v: np.full(frame_count, np.nan) for v in runs} for t in TARGETS}
    area_ratio = {t: {v: np.full(frame_count, np.nan) for v in runs} for t in TARGETS}
    with per_frame_path.open("w", encoding="utf-8") as handle:
        for frame in range(frame_count):
            record: dict = {"analysis_frame_index": frame, "parts": {}}
            ego_pixels: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for target in TARGETS:
                masks: dict[str, np.ndarray] = {}
                extra: dict[str, tuple[np.ndarray, np.ndarray]] = {}
                excluded: list[str] = []
                for view, run in runs.items():
                    if target not in run.targets:
                        continue
                    mask = run.mask(frame, target)
                    if mask is None or not mask.any():
                        continue
                    flag = flags.get((view, target))
                    if flag is not None and flag[frame]:
                        excluded.append(view)
                        continue
                    if is_ego(view):
                        gate = ego_gate.get(view)
                        if gate is None or not gate[frame]:
                            continue
                        if view not in ego_pixels:
                            pose = rig.pose_frame(view, frame)
                            ego_pixels[view] = grid._project(rig, view, grid.centres, pose)
                        extra[view] = ego_pixels[view]
                    masks[view] = mask
                if len(masks) < HULL_RULES.min_views:
                    record["parts"][target] = {
                        "voxel_count": 0,
                        "views_used": [],
                        "excluded": excluded,
                    }
                    continue
                alive = grid.carve(masks, extra)
                occupied = grid.centres[alive]
                count = int(alive.sum())
                counts[target][frame] = count
                views_used[target][frame] = len(masks)
                part_record: dict = {
                    "voxel_count": count,
                    "views_used": sorted(masks),
                    "excluded": excluded,
                    "centroid_mm": [round(float(v), 1) for v in occupied.mean(axis=0)]
                    if count
                    else None,
                    "bbox_min_mm": [round(float(v), 1) for v in occupied.min(axis=0)]
                    if count
                    else None,
                    "bbox_max_mm": [round(float(v), 1) for v in occupied.max(axis=0)]
                    if count
                    else None,
                    "height_above_table_mm": (
                        round(
                            float(plane.signed_distance(occupied.mean(axis=0, keepdims=True))[0]), 1
                        )
                        if count
                        else None
                    ),
                    "view_iou": {},
                    "view_area_ratio_mask_over_hull": {},
                }
                if frame % (30 // VOXEL_FPS) == 0:
                    voxels_1fps[f"{target}/{frame:06d}"] = grid.indices[alive]
                if count:
                    for view, run in runs.items():
                        mask = run.mask(frame, target) if target in run.targets else None
                        if mask is None:
                            continue
                        pose = rig.pose_frame(view, frame) if is_ego(view) else None
                        projection = hull_projection_mask(
                            rig, view, occupied, mask.shape, voxel_mm, pose
                        )
                        union = np.logical_or(projection, mask).sum()
                        value = (
                            float(np.logical_and(projection, mask).sum() / union) if union else 0.0
                        )
                        hull_area = int(projection.sum())
                        ratio = float(mask.sum() / hull_area) if hull_area else float("inf")
                        iou[target][view][frame] = value
                        area_ratio[target][view][frame] = ratio
                        part_record["view_iou"][view] = round(value, 3)
                        part_record["view_area_ratio_mask_over_hull"][view] = (
                            round(ratio, 3) if np.isfinite(ratio) else None
                        )
                        if frame % (30 // VOXEL_FPS) == 0:
                            out = projection_root / view
                            out.mkdir(parents=True, exist_ok=True)
                            rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
                            rgba[projection] = (255, 255, 255, 255)
                            cv2.imwrite(str(out / f"{frame:06d}_{target}.png"), rgba)
                record["parts"][target] = part_record
            handle.write(json.dumps(record) + "\n")

    # Disagreement episodes: IoU floor or mask far larger than the hull projection.
    episodes: list[DisagreementEpisode] = []
    proposals: list[ProposedValidityInterval] = []
    comparisons: list[HullViewComparison] = []
    summaries: list[HullPartSummary] = []
    for target in TARGETS:
        exists = np.isfinite(counts[target]) & (counts[target] > 0)
        errors = {}
        # A hull carved from imperfect silhouettes is over-carved, so every mask tends to be
        # larger than its projection; a view is flagged only when it disagrees clearly more
        # than the other views do on the same frame.
        iou_stack = np.stack([iou[target][v] for v in runs])
        ratio_stack = np.stack([area_ratio[target][v] for v in runs])
        with np.errstate(invalid="ignore", all="ignore"):
            median_iou = np.nanmedian(iou_stack, axis=0)
            median_ratio = np.nanmedian(np.where(np.isfinite(ratio_stack), ratio_stack, np.nan), 0)
        for view in runs:
            with np.errstate(invalid="ignore"):
                flagged = (
                    (iou[target][view] < IOU_FLOOR) & (iou[target][view] < median_iou - 0.15)
                ) | (
                    area_ratio[target][view]
                    > np.maximum(AREA_RATIO_CEILING, 1.5 * np.nan_to_num(median_ratio, nan=1.0))
                )
            error = np.where(np.isfinite(iou[target][view]), flagged.astype(float), np.nan)
            errors[view] = error
        dropped = {view: np.zeros(frame_count, dtype=bool) for view in runs}
        target_episodes = episodes_from_errors(
            errors,
            exists,
            dropped,
            views_used[target],
            target=target,
            threshold_px=0.5,
            min_frames=HULL_RULES.min_episode_frames,
        )
        target_episodes = tuple(
            e.model_copy(
                update={
                    "kind": "hull_disagreement",
                    "contradicts_majority": bool(
                        e.view == REFERENCE_VIEW
                        and np.all(views_used[target][e.start_frame : e.end_frame_exclusive] >= 3)
                    ),
                    "max_reprojection_error_px": float(
                        np.nanmax(1.0 - iou[target][e.view][e.start_frame : e.end_frame_exclusive])
                    ),
                    "mean_reprojection_error_px": float(
                        np.nanmean(1.0 - iou[target][e.view][e.start_frame : e.end_frame_exclusive])
                    ),
                }
            )
            for e in target_episodes
        )
        episodes.extend(target_episodes)
        for start, end in merge_intervals(
            (e.start_frame, e.end_frame_exclusive)
            for e in target_episodes
            if e.view == REFERENCE_VIEW and e.contradicts_majority
        ):
            proposals.append(
                ProposedValidityInterval(
                    target=target,
                    start_frame=start,
                    end_frame_exclusive=end,
                    proposed_state="not_contact_eligible",
                    trigger="hull_disagreement",
                    rationale=(
                        f"C10379 {target} mask disagrees with the hull carved by >= 3 other "
                        f"views (IoU < {IOU_FLOOR} or area > {AREA_RATIO_CEILING}x) for "
                        f"{end - start} consecutive frames; proposal only, not applied"
                    ),
                )
            )
        for view in runs:
            valid = np.isfinite(iou[target][view])
            comparisons.append(
                HullViewComparison(
                    view=view,
                    target=target,
                    frames_compared=int(valid.sum()),
                    median_iou=float(np.median(iou[target][view][valid])) if valid.any() else None,
                    p10_iou=float(np.quantile(iou[target][view][valid], 0.1))
                    if valid.any()
                    else None,
                    median_area_ratio_mask_over_hull=(
                        float(
                            np.median(
                                area_ratio[target][view][
                                    valid & np.isfinite(area_ratio[target][view])
                                ]
                            )
                        )
                        if (valid & np.isfinite(area_ratio[target][view])).any()
                        else None
                    ),
                    mask_larger_than_hull_fraction=(
                        float(np.mean(area_ratio[target][view][valid] > 1.25))
                        if valid.any()
                        else None
                    ),
                )
            )
        summaries.append(
            HullPartSummary(
                target=target,
                frames_with_hull=int(exists.sum()),
                median_voxel_count=float(np.median(counts[target][exists]))
                if exists.any()
                else 0.0,
                p10_voxel_count=float(np.quantile(counts[target][exists], 0.1))
                if exists.any()
                else 0.0,
                p90_voxel_count=float(np.quantile(counts[target][exists], 0.9))
                if exists.any()
                else 0.0,
                median_views_used=float(np.median(views_used[target][exists]))
                if exists.any()
                else 0.0,
                episodes=len(target_episodes),
            )
        )

    voxels_path = root / "hull_voxels_1fps.npz"
    np.savez_compressed(voxels_path, **voxels_1fps)
    np.savez_compressed(
        root / "hull_series.npz",
        **{f"voxel_count/{t}": counts[t] for t in TARGETS},
        **{f"views_used/{t}": views_used[t] for t in TARGETS},
        **{f"iou/{t}/{v}": iou[t][v] for t in TARGETS for v in runs},
        **{f"area_ratio/{t}/{v}": area_ratio[t][v] for t in TARGETS for v in runs},
        views=np.array(list(runs)),
    )
    manifest = VisualHullManifest(
        manifest_kind="multiview_visual_hull",
        frame_count=frame_count,
        analysis_fps=30,
        voxel_size_mm=voxel_mm,
        grid_origin_mm=tuple(float(v) for v in grid.origin),
        grid_shape=grid.shape,  # type: ignore[arg-type]
        grid_axes="world_xyz",
        table_plane=plane,
        volume_bounds_source=grid.volume_bounds_source,
        views_used=tuple(runs),
        consensus_manifest=_fingerprint(consensus_path, repository_root),
        sources=tuple(s for s in consensus.sources if s.view in runs),
        per_frame_uri=relative_uri(per_frame_path, repository_root),
        voxels_npz_uri=relative_uri(voxels_path, repository_root),
        voxels_npz_format=(
            "npz; key '<target>/<analysis_frame:06d>' -> (N, 3) int16 voxel indices (i, j, k) on "
            "world axes; centre_mm = grid_origin_mm + (index + 0.5) * voxel_size_mm; frames every "
            f"{30 // VOXEL_FPS} analysis frames"
        ),
        voxels_fps=VOXEL_FPS,
        hull_projection_masks_uri=relative_uri(projection_root, repository_root),
        part_summaries=tuple(summaries),
        view_comparisons=tuple(comparisons),
        episodes=tuple(episodes),
        proposed_validity_intervals=tuple(proposals),
        disagreement_rules=HULL_RULES,
        runtime_seconds=time.monotonic() - started,
        not_run=NOT_RUN,
        claim_boundaries=(
            *MULTIVIEW_CLAIM_BOUNDARIES,
            "A visual hull is the intersection of the views' silhouette cones: an upper bound on "
            "the part's volume that inherits every mask error and every camera-model error; it "
            "is comparison evidence, not a reconstruction claim.",
            f"Static views only carve; {len(static_views)} static views were available.",
        ),
    )
    (root / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--consensus-root", type=Path, default=CONSENSUS_ROOT)
    parser.add_argument("--voxel-mm", type=float, default=VOXEL_MM)
    parser.add_argument("--frame-count", type=int, default=None)
    parser.add_argument("--ego-view", action="append", default=[])
    args = parser.parse_args()
    manifest = build_visual_hull(
        args.repository_root,
        output_root=args.output_root,
        consensus_root=args.consensus_root,
        voxel_mm=args.voxel_mm,
        frame_count=args.frame_count,
        ego_views=tuple(args.ego_view),
    )
    print(
        f"grid {manifest.grid_shape} @ {manifest.voxel_size_mm:g} mm from "
        f"{np.round(manifest.grid_origin_mm, 0).tolist()}; views {', '.join(manifest.views_used)}"
    )
    for summary in manifest.part_summaries:
        print(
            f"{summary.target}: hull on {summary.frames_with_hull} frames, median "
            f"{summary.median_voxel_count:.0f} voxels (p10 {summary.p10_voxel_count:.0f}, p90 "
            f"{summary.p90_voxel_count:.0f}), median {summary.median_views_used:.0f} views, "
            f"{summary.episodes} hull_disagreement episodes"
        )
    for comparison in manifest.view_comparisons:
        if comparison.median_iou is not None:
            ratio = comparison.median_area_ratio_mask_over_hull
            print(
                f"  {comparison.target:9s} {comparison.view}: median IoU "
                f"{comparison.median_iou:.2f}, p10 {comparison.p10_iou:.2f}, mask/hull area "
                f"{ratio if ratio is None else round(ratio, 2)}, mask larger than hull "
                f"{comparison.mask_larger_than_hull_fraction:.2f}"
            )
    for proposal in manifest.proposed_validity_intervals:
        print(
            f"  proposed {proposal.target} [{proposal.start_frame},{proposal.end_frame_exclusive})"
        )
    print(f"{manifest.runtime_seconds:.0f} s -> {args.output_root}")


if __name__ == "__main__":
    main()
