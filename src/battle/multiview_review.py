"""Review surfaces for the multi-view pass.

`battle-build-multiview-static-comparison` writes one Rerun recording with every static
proxy as its own 2D view (part masks as RGBA cut-outs, the consensus point reprojected as a
marker with its pixel error), the world-millimetre 3D view (consensus centroids, dataset
hands, all camera frusta, optional hull voxels) and a time panel of per-view reprojection
error.  `MultiviewLayer` is the same data reduced to what the v4 interaction review logs as
its `assembly101_multiview` layer.

Everything drawn here is cross-view disagreement between runs of one tracker seeded by an
agent from geometry (C10379 excepted); it is not accuracy and it changes no reference mask.
CC BY-NC 4.0 attribution applies to the dataset assets.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from . import assembly101_reference as a101
from . import mask_cache
from .assembly101_pose_schemas import ASSEMBLY101_EDGES, Assembly101HandFrame
from .cli_common import add_output_root, add_repository_root
from .four_part_contract import TARGETS
from .multiview_consensus import (
    MANIFEST_NAME,
    load_consensus,
    load_view_run,
)
from .multiview_consensus import (
    OUTPUT_ROOT as CONSENSUS_ROOT,
)
from .multiview_geometry import CameraRig
from .multiview_schemas import DisagreementEpisode, MultiviewConsensusManifest
from .multiview_seed_transfer import REFERENCE_VIEW, proxy_to_raw_scale
from .observations import object_for_label
from .rerun_logging import init_and_save, log_rgba_mask, time_series_stack, time_series_view

OUTPUT_ROOT = Path("runs/multiview-static-comparison-first-minute")
RECORDING_NAME = "multiview_static_comparison.rrd"
HULL_ROOT = Path("runs/multiview-visual-hull-first-minute")
PART_COLORS = mask_cache.REVIEW_COLORS
HAND_COLORS = {"left": (255, 235, 130), "right": (140, 255, 235)}
CAMERA_PLANE_MM = 250.0
WORLD_3D = "world_mm_3d"
ENTITY_ROOT = "world/assembly101_multiview_first_minute"
APPLICATION_ID = "battle-multiview-static-comparison"
RECORDING_ID = "multiview_static_comparison"
# Human anchor masks (C10379 only) drawn as outlines on that tile at the anchor frames, and the
# seed-search accept/reject proposals drawn as outlined candidates on each view's tile at the
# proposal frame.  Both are cleared on the following frame.
ANCHOR_MASKS = Path("runs/human-review-anchors-first-minute/anchors/anchor_masks.json")
PROPOSALS_ROOT = Path("runs/seed-search-20260920/proposals")
PROPOSAL_LABEL = "proposal: accept/reject pending"
PROPOSAL_COLORS = ((255, 255, 255), (255, 120, 40), (80, 220, 255), (200, 120, 255))
ANCHOR_SERIES = "metadata/anchors"


class MultiviewLayer:
    """Consensus arrays and episodes loaded once for a review build."""

    def __init__(self, repository_root: Path, root: Path = CONSENSUS_ROOT) -> None:
        self.repository_root = repository_root.resolve()
        self.root = self.repository_root / root
        self.manifest: MultiviewConsensusManifest = load_consensus(self.root / MANIFEST_NAME)
        with np.load(self.root / "consensus_points.npz") as archive:
            self.views = tuple(str(v) for v in archive["views"])
            self.consensus = {t: archive[f"consensus/{t}"] for t in TARGETS}
            self.used_counts = {t: archive[f"used_counts/{t}"] for t in TARGETS}
            self.error = {t: {v: archive[f"error/{t}/{v}"] for v in self.views} for t in TARGETS}
        self.episodes_by_frame: dict[int, list[DisagreementEpisode]] = {}
        for episode in self.manifest.episodes:
            for frame in range(episode.start_frame, episode.end_frame_exclusive):
                self.episodes_by_frame.setdefault(frame, []).append(episode)

    def consensus_point(self, frame: int, target: str) -> np.ndarray | None:
        point = self.consensus[target][frame]
        return None if np.isnan(point).any() else point

    def active_episodes(self, frame: int) -> list[DisagreementEpisode]:
        return self.episodes_by_frame.get(frame, [])

    def navigation_lines(self, frame: int) -> tuple[str, ...]:
        lines = []
        for target in TARGETS:
            point = self.consensus_point(frame, target)
            if point is None:
                lines.append(f"- multiview `{target}`: no consensus (fewer than 2 views)")
                continue
            reference_error = self.error[target].get(REFERENCE_VIEW, np.full(1, np.nan))[frame]
            lines.append(
                f"- multiview `{target}`: {int(self.used_counts[target][frame])} views agree; "
                f"C10379 error {reference_error:.0f} raw px"
                if np.isfinite(reference_error)
                else f"- multiview `{target}`: {int(self.used_counts[target][frame])} views agree; "
                "C10379 has no mask"
            )
        for episode in self.active_episodes(frame):
            lines.append(
                f"- `{episode.kind}` episode: {episode.view} {episode.target} "
                f"[{episode.start_frame},{episode.end_frame_exclusive}) "
                + ("**contradicts the majority**" if episode.contradicts_majority else "outlier")
            )
        return tuple(lines)

    def episodes_markdown(self) -> str:
        lines = [
            "# Cross-view disagreement (not accuracy)",
            "",
            f"Views: {', '.join(self.views)}. Rules: {self.manifest.rules.description}",
            "",
            "## C10379 contradicted by the majority",
        ]
        if not self.manifest.reference_contradiction_intervals:
            lines.append("- none")
        for target, start, end in self.manifest.reference_contradiction_intervals:
            lines.append(f"- `{target}` frames [{start},{end}) -> proposed `not_contact_eligible`")
        lines += ["", "## All episodes"]
        for episode in self.manifest.episodes:
            lines.append(
                f"- {episode.view} `{episode.target}` [{episode.start_frame},"
                f"{episode.end_frame_exclusive}) max {episode.max_reprojection_error_px:.0f} px"
                + (" (contradicts majority)" if episode.contradicts_majority else "")
            )
        lines += ["", "## Claim boundaries", *[f"- {b}" for b in self.manifest.claim_boundaries]]
        return "\n".join(lines)


# -- shared logging ----------------------------------------------------------------------------


def log_camera_frusta(root: str, rig: CameraRig, views: tuple[str, ...]) -> None:
    for view in views:
        if rig.camera(view).is_ego:
            continue
        pose = rig.camera_to_world(view)
        path = f"{root}/camera/{view}"
        rr.log(path, rr.Transform3D(translation=pose[:3, 3], mat3x3=pose[:3, :3]), static=True)
        rr.log(
            path,
            rr.Pinhole(
                image_from_camera=np.asarray(rig.camera(view).intrinsic_matrix, dtype=np.float64),
                resolution=list(rig.image_size(view)),
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=CAMERA_PLANE_MM,
            ),
            static=True,
        )


def log_consensus_3d(root: str, layer: MultiviewLayer, frame: int) -> None:
    points, colors, labels = [], [], []
    for target in TARGETS:
        point = layer.consensus_point(frame, target)
        if point is None:
            continue
        points.append(point.tolist())
        colors.append(PART_COLORS[target])
        labels.append(f"consensus {target} ({int(layer.used_counts[target][frame])} views)")
    path = f"{root}/multiview_consensus"
    if points:
        rr.log(path, rr.Points3D(points, colors=colors, labels=labels, radii=12.0))
    else:
        rr.log(path, rr.Clear(recursive=True))


def log_dataset_hands_3d(root: str, frame: Assembly101HandFrame, threshold: float) -> None:
    drawn = [hand for hand in frame.hands if hand.confidence >= threshold]
    path = f"{root}/hands"
    if not drawn:
        rr.log(path, rr.Clear(recursive=True))
        return
    points, strips, colors_p, colors_s = [], [], [], []
    for hand in drawn:
        color = HAND_COLORS[hand.side]
        world = [[p.x, p.y, p.z] for p in hand.joints_world_mm]
        points.extend(world)
        colors_p.extend([color] * len(world))
        strips.extend([[world[a], world[b]] for a, b in ASSEMBLY101_EDGES])
        colors_s.extend([color] * len(ASSEMBLY101_EDGES))
    rr.log(f"{path}/joints", rr.Points3D(points, colors=colors_p, radii=4.0))
    rr.log(f"{path}/skeletons", rr.LineStrips3D(strips, colors=colors_s, radii=2.0))


def log_error_series_static(root: str, layer: MultiviewLayer) -> None:
    for target in TARGETS:
        for view in layer.views:
            rr.log(
                f"{root}/{target}/{view}",
                rr.SeriesLines(names=f"{view} {target} error (raw px)"),
                static=True,
            )
        rr.log(
            f"{root}/{target}/views_used",
            rr.SeriesLines(names=f"{target} views in consensus", colors=[PART_COLORS[target]]),
            static=True,
        )


def log_error_series_frame(root: str, layer: MultiviewLayer, frame: int) -> None:
    for target in TARGETS:
        for view in layer.views:
            value = layer.error[target][view][frame]
            path = f"{root}/{target}/{view}"
            if np.isfinite(value):
                rr.log(path, rr.Scalars([float(value)]))
            else:
                rr.log(path, rr.Clear(recursive=False))
        rr.log(f"{root}/{target}/views_used", rr.Scalars([int(layer.used_counts[target][frame])]))


# -- v4 interaction review layer ---------------------------------------------------------------

V4_TEXT_PANEL = ("metadata/multiview_disagreement", "Cross-view disagreement (multiview)")
V4_DIAGNOSTICS = "diagnostics/multiview"
V4_3D_ROOT = "contexts/assembly101_world_mm_3d"


def v4_coverage(layer: MultiviewLayer) -> dict[str, int]:
    coverage = {
        f"multiview_consensus_frames_{summary.target}": summary.frames_with_consensus
        for summary in layer.manifest.summaries
    }
    coverage["multiview_disagreement_episodes"] = len(layer.manifest.episodes)
    coverage["multiview_reference_contradicted_frames"] = sum(
        end - start for _, start, end in layer.manifest.reference_contradiction_intervals
    )
    coverage["multiview_views"] = len(layer.views)
    return coverage


def v4_log_static(entity: str, layer: MultiviewLayer) -> None:
    """Disagreement document plus the per-part series names for the multiview panel."""
    rr.log(
        f"{entity}/{V4_TEXT_PANEL[0]}",
        rr.TextDocument(layer.episodes_markdown(), media_type="text/markdown"),
        static=True,
    )
    for part in TARGETS:
        rr.log(
            f"{entity}/{V4_DIAGNOSTICS}/{part}/views_used",
            rr.SeriesLines(names=f"{part}: views in consensus", colors=[PART_COLORS[part]]),
            static=True,
        )
        rr.log(
            f"{entity}/{V4_DIAGNOSTICS}/{part}/c10379_error_px",
            rr.SeriesLines(
                names=f"{part}: C10379 error vs consensus (raw px)", colors=[PART_COLORS[part]]
            ),
            static=True,
        )


def _scalar_or_clear(path: str, value: float | None) -> None:
    if value is None:
        rr.log(path, rr.Clear(recursive=False))
    else:
        rr.log(path, rr.Scalars([value]))


def v4_log_frame(entity: str, layer: MultiviewLayer, frame: int) -> None:
    """Consensus centroids in the existing world-mm 3D view and the C10379 agreement series."""
    for part in TARGETS:
        used = int(layer.used_counts[part][frame])
        _scalar_or_clear(
            f"{entity}/{V4_DIAGNOSTICS}/{part}/views_used", float(used) if used else None
        )
        error = layer.error[part].get(REFERENCE_VIEW, np.full(frame + 1, np.nan))[frame]
        _scalar_or_clear(
            f"{entity}/{V4_DIAGNOSTICS}/{part}/c10379_error_px",
            float(error) if np.isfinite(error) else None,
        )
    log_consensus_3d(f"{entity}/{V4_3D_ROOT}", layer, frame)


# -- hull overlay (Track 5) --------------------------------------------------------------------


class HullOverlay:
    """Sparse hull voxels (1 fps) and per-view hull projections when the hull run exists."""

    def __init__(self, repository_root: Path, root: Path = HULL_ROOT) -> None:
        from .multiview_schemas import VisualHullManifest

        self.root = (repository_root / root).resolve()
        self.manifest = VisualHullManifest.model_validate_json(
            (self.root / "manifest.json").read_text(encoding="utf-8")
        )
        self.voxels = np.load(self.root / "hull_voxels_1fps.npz")
        self.projection_root = repository_root / self.manifest.hull_projection_masks_uri

    def voxel_centres(self, frame: int, target: str) -> np.ndarray | None:
        key = f"{target}/{frame:06d}"
        if key not in self.voxels:
            return None
        indices = self.voxels[key]
        if indices.size == 0:
            return None
        origin = np.asarray(self.manifest.grid_origin_mm)
        return origin + (indices.astype(np.float64) + 0.5) * self.manifest.voxel_size_mm

    def projection_png(self, view: str, frame: int, target: str) -> Path | None:
        path = self.projection_root / view / f"{frame:06d}_{target}.png"
        return path if path.is_file() else None


def log_hull_3d(root: str, hull: HullOverlay, frame: int) -> None:
    second = (frame // 30) * 30
    for target in TARGETS:
        path = f"{root}/hull/{target}"
        centres = hull.voxel_centres(second, target)
        if centres is None:
            rr.log(path, rr.Clear(recursive=True))
            continue
        rr.log(
            path,
            rr.Points3D(
                centres,
                colors=[PART_COLORS[target]] * len(centres),
                radii=hull.manifest.voxel_size_mm * 0.45,
            ),
        )


# -- human anchor outlines and seed-search proposals -----------------------------------------


def _mask_outlines(mask: np.ndarray) -> list[list[list[float]]]:
    import cv2

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    strips: list[list[list[float]]] = []
    for contour in contours:
        points = contour.reshape(-1, 2).astype(float).tolist()
        if len(points) >= 3:
            strips.append([*points, points[0]])
    return strips


def load_anchor_outlines(
    repository_root: Path, mask_set: Path = ANCHOR_MASKS
) -> tuple[str, dict[int, dict[str, Path]]]:
    """(view, {frame: {part: mask png}}) from an exported human anchor mask set."""
    path = repository_root / mask_set
    exported = json.loads(path.read_text(encoding="utf-8"))
    view = str(exported["view_id"]).split("-", 1)[1].upper()
    by_frame: dict[int, dict[str, Path]] = {}
    for anchor in exported["anchors"]:
        frame = int(anchor["analysis_frame_index"])
        by_frame.setdefault(frame, {})
        if anchor.get("mask_uri"):
            by_frame[frame][str(anchor["target"])] = path.parent.parent / anchor["mask_uri"]
    return view, by_frame


class ProposalCell:
    """One `proposal.json` of `battle-seed-search`: candidates for one view, part and frame."""

    def __init__(self, directory: Path) -> None:
        record = json.loads((directory / "proposal.json").read_text(encoding="utf-8"))
        self.directory = directory
        self.view = str(record["view"])
        self.part = str(record["part"])
        self.frame = int(record["analysis_frame_index"])
        self.disagreement = float(record.get("disagreement_1_minus_mean_pairwise_iou", 0.0))
        self.decision = record.get("human_decision")
        self.candidates = tuple(
            (str(item["strategy"]), directory / str(item["mask_uri"]))
            for item in record["candidates"]
        )


def load_proposals(
    repository_root: Path, root: Path = PROPOSALS_ROOT
) -> dict[str, list[ProposalCell]]:
    """Proposals by view, each list sorted by frame (empty when the root is absent)."""
    base = repository_root / root
    if not base.is_dir():
        return {}
    by_view: dict[str, list[ProposalCell]] = {}
    for proposal in sorted(base.glob("*/*/proposal.json")):
        cell = ProposalCell(proposal.parent)
        by_view.setdefault(cell.view, []).append(cell)
    for cells in by_view.values():
        cells.sort(key=lambda cell: (cell.frame, cell.part))
    return by_view


def log_anchor_marks_static(root: str, frames: tuple[int, ...], view: str) -> None:
    rr.log(
        f"{root}/{ANCHOR_SERIES}/anchor_frame",
        rr.SeriesPoints(
            names=f"human anchor frame ({len(frames)} frames, {view})",
            colors=[(255, 255, 255)],
            markers="diamond",
            marker_sizes=6.0,
        ),
        static=True,
    )


def log_anchor_outlines_frame(
    root: str, view_root: str, frame: int, anchors: dict[int, dict[str, Path]]
) -> None:
    """Outlines of the human masks on the labelled view at its anchor frames only."""
    if frame in anchors:
        rr.log(f"{root}/{ANCHOR_SERIES}/anchor_frame", rr.Scalars([1.0]))
        for part, mask_path in anchors[frame].items():
            strips = _mask_outlines(mask_cache.decode_mask_png(mask_path))
            if not strips:
                continue
            rr.log(
                f"{view_root}/human_anchor_outlines/{part}",
                rr.LineStrips2D(
                    strips,
                    colors=[PART_COLORS[part]] * len(strips),
                    radii=1.5,
                    labels=[f"human anchor {part} f{frame}"],
                    draw_order=3.0,
                ),
            )
    elif (frame - 1) in anchors:
        rr.log(f"{root}/{ANCHOR_SERIES}/anchor_frame", rr.Clear(recursive=False))
        rr.log(f"{view_root}/human_anchor_outlines", rr.Clear(recursive=True))


def log_proposals_frame(view_root: str, frame: int, cells: list[ProposalCell]) -> None:
    """Each candidate of a proposal cell as an outline in its own colour at the cell's frame."""
    for cell in cells:
        path = f"{view_root}/proposals/{cell.part}"
        if cell.frame == frame:
            strips: list[list[list[float]]] = []
            colors: list[tuple[int, int, int]] = []
            for index, (_, mask_path) in enumerate(cell.candidates):
                outlines = _mask_outlines(mask_cache.decode_mask_png(mask_path))
                strips.extend(outlines)
                colors.extend([PROPOSAL_COLORS[index % len(PROPOSAL_COLORS)]] * len(outlines))
            if not strips:
                continue
            status = (
                PROPOSAL_LABEL if cell.decision is None else f"proposal: decided {cell.decision}"
            )
            strategies = "; ".join(
                f"c{index:02d} {strategy}" for index, (strategy, _) in enumerate(cell.candidates)
            )
            rr.log(
                path,
                rr.LineStrips2D(
                    strips,
                    colors=colors,
                    radii=1.5,
                    labels=[f"{status} - {cell.part} f{cell.frame} ({strategies})"],
                    draw_order=3.0,
                ),
            )
        elif cell.frame == frame - 1:
            rr.log(path, rr.Clear(recursive=False))


# -- the eight-view recording ----------------------------------------------------------------


def _project_marker(rig: CameraRig, view: str, frame: int, point: np.ndarray) -> np.ndarray | None:
    """Raw-pixel projection of a world point; an ego view uses its pose at the view's frame
    and yields None where the dataset has no pose or the point is behind the camera."""
    if not rig.camera(view).is_ego:
        return rig.project(view, point.reshape(1, 3))[0]
    try:
        pose_frame = rig.pose_frame(view, frame)
        if rig.depth(view, point.reshape(1, 3), pose_frame)[0] <= 0:
            return None
        return rig.project(view, point.reshape(1, 3), pose_frame)[0]
    except KeyError:
        return None


def build_static_comparison(
    repository_root: Path,
    *,
    output_root: Path = OUTPUT_ROOT,
    consensus_root: Path = CONSENSUS_ROOT,
    hull_root: Path | None = HULL_ROOT,
    mask_every: int = 1,
    anchor_masks: Path | None = ANCHOR_MASKS,
    proposals_root: Path | None = PROPOSALS_ROOT,
    application_id: str = APPLICATION_ID,
    recording_id: str = RECORDING_ID,
    embed_blueprint: bool = True,
) -> Path:
    """Write the multi-view comparison recording.

    `anchor_masks` draws the exported human anchor masks as outlines on the labelled view at
    the anchor frames; `proposals_root` draws the seed-search proposal candidates on each view
    at their frames.  `application_id` / `recording_id` may be set to another package's so
    `rerun rrd merge` can put both into one store; `embed_blueprint=False` then leaves the
    layout to that package's `.rbl` presets.
    """
    from .muggled_smoke import _create_bounded_rerun_video

    started = time.monotonic()
    repository_root = repository_root.resolve()
    layer = MultiviewLayer(repository_root, consensus_root)
    anchor_view, anchors = (
        load_anchor_outlines(repository_root, anchor_masks)
        if anchor_masks is not None and (repository_root / anchor_masks).is_file()
        else (None, {})
    )
    proposals = load_proposals(repository_root, proposals_root) if proposals_root else {}
    rig = CameraRig.load(repository_root)
    frame_count = layer.manifest.frame_count
    runs = {
        source.view: load_view_run(
            repository_root,
            Path(source.run_directory_uri),
            view=source.view,
            frame_count=frame_count,
        )
        for source in layer.manifest.sources
    }
    dataset = a101.load_reference(a101.OUTPUT_ROOT, repository_root, frame_count=frame_count)
    hull = (
        HullOverlay(repository_root, hull_root)
        if hull_root is not None and (repository_root / hull_root / "manifest.json").is_file()
        else None
    )
    root_dir = repository_root / output_root
    root_dir.mkdir(parents=True, exist_ok=True)
    videos: dict[str, Path] = {}
    for view, run in runs.items():
        bounded = run.run_directory / f"input_{frame_count}f.mp4"
        if not bounded.is_file():
            proxy_uri = (
                run.manifest.four_part_focused.proxy_fingerprint.uri
                if run.manifest.four_part_focused is not None
                else run.manifest.four_part_multiview.proxy_fingerprint.uri  # type: ignore[union-attr]
            )
            bounded = _create_bounded_rerun_video(
                proxy_path=repository_root / proxy_uri,
                output_path=root_dir / f"{view}_input_{frame_count}f.mp4",
                run_directory=root_dir,
                frame_count=frame_count,
            )
        videos[view] = bounded
    rrd_path = root_dir / RECORDING_NAME
    init_and_save(application_id, rrd_path, recording_id=recording_id)
    entity = ENTITY_ROOT
    if anchors and anchor_view in runs:
        log_anchor_marks_static(entity, tuple(sorted(anchors)), anchor_view)
    for view, path in videos.items():
        rr.log(f"{entity}/views/{view}/video_asset", rr.AssetVideo(path=path), static=True)
    rr.log(
        f"{entity}/metadata/consensus_manifest",
        rr.TextDocument(layer.manifest.model_dump_json(indent=2), media_type="application/json"),
        static=True,
    )
    rr.log(
        f"{entity}/metadata/disagreement",
        rr.TextDocument(layer.episodes_markdown(), media_type="text/markdown"),
        static=True,
    )
    log_camera_frusta(f"{entity}/{WORLD_3D}", rig, tuple(runs))
    log_error_series_static(f"{entity}/diagnostics/multiview", layer)
    for frame in range(frame_count):
        seconds = frame / 30.0
        rr.set_time("analysis_frame", sequence=frame)
        rr.set_time("analysis_time", duration=seconds)
        for view, run in runs.items():
            view_root = f"{entity}/views/{view}"
            rr.log(
                f"{view_root}/video",
                rr.VideoFrameReference(seconds=seconds, video_reference=f"{view_root}/video_asset"),
            )
            scale = proxy_to_raw_scale(view)
            if frame % mask_every == 0:
                observation = run.observations.get(frame)
                for target in TARGETS:
                    item = (
                        object_for_label(observation, target) if observation is not None else None
                    )
                    path = f"{view_root}/masks/{target}"
                    if item is None or item.mask is None:
                        rr.log(path, rr.Clear(recursive=False))
                        continue
                    log_rgba_mask(
                        path, run.cache.rgba_png(item.mask.uri, PART_COLORS[target]), opacity=0.4
                    )
                    if hull is not None:
                        projection = hull.projection_png(view, frame, target)
                        hull_path = f"{view_root}/hull_projection/{target}"
                        if projection is None:
                            rr.log(hull_path, rr.Clear(recursive=False))
                        else:
                            rr.log(
                                hull_path,
                                rr.EncodedImage(
                                    path=projection,
                                    media_type="image/png",
                                    opacity=0.35,
                                    draw_order=2.0,
                                ),
                            )
            markers, colors, labels = [], [], []
            for target in TARGETS:
                point = layer.consensus_point(frame, target)
                if point is None:
                    continue
                pixel = _project_marker(rig, view, frame, point)
                if pixel is None:
                    continue
                pixel = pixel / scale
                error = layer.error[target][view][frame]
                markers.append(pixel.tolist())
                colors.append(PART_COLORS[target])
                labels.append(
                    f"{target} consensus"
                    + (f" ({error:.0f} px off)" if np.isfinite(error) and error > 0 else "")
                )
            marker_path = f"{view_root}/consensus_markers"
            if markers:
                rr.log(marker_path, rr.Points2D(markers, colors=colors, labels=labels, radii=6.0))
            else:
                rr.log(marker_path, rr.Clear(recursive=False))
            if view == anchor_view and anchors:
                log_anchor_outlines_frame(entity, view_root, frame, anchors)
            if view in proposals:
                log_proposals_frame(view_root, frame, proposals[view])
        log_consensus_3d(f"{entity}/{WORLD_3D}", layer, frame)
        log_dataset_hands_3d(
            f"{entity}/{WORLD_3D}",
            dataset.frames[frame],
            dataset.manifest.draw_confidence_threshold,
        )
        if hull is not None:
            log_hull_3d(f"{entity}/{WORLD_3D}", hull, frame)
        log_error_series_frame(f"{entity}/diagnostics/multiview", layer, frame)
    if embed_blueprint:
        rr.send_blueprint(
            static_comparison_blueprint(
                entity,
                tuple(runs),
                hull is not None,
                anchor_view=anchor_view if anchors else None,
                proposal_views=tuple(view for view in runs if view in proposals),
            )
        )
    rr.disconnect()
    index = {
        "manifest_kind": "multiview_static_comparison",
        "recording": rrd_path.relative_to(repository_root).as_posix(),
        "application_id": application_id,
        "recording_id": recording_id,
        "embedded_blueprint": embed_blueprint,
        "consensus_root": consensus_root.as_posix(),
        "hull_root": hull_root.as_posix() if hull is not None and hull_root is not None else None,
        "consensus_manifest": layer.manifest.per_frame_fingerprint.model_dump(mode="json")
        if layer.manifest.per_frame_fingerprint
        else None,
        "views": list(runs),
        "hull_overlay": hull is not None,
        "mask_every": mask_every,
        "anchor_outlines": {
            "view": anchor_view,
            "frames": sorted(anchors),
            "mask_set": anchor_masks.as_posix() if anchor_masks is not None else None,
        }
        if anchors
        else None,
        "proposals": {
            view: [
                {
                    "part": cell.part,
                    "frame": cell.frame,
                    "candidates": len(cell.candidates),
                    "human_decision": cell.decision,
                }
                for cell in cells
            ]
            for view, cells in proposals.items()
            if view in runs
        },
        "runtime_seconds": time.monotonic() - started,
        "recording_bytes": rrd_path.stat().st_size,
        "claim_boundaries": [
            *layer.manifest.claim_boundaries,
            "Human anchor outlines exist on the labelled view only (C10379); every other view's "
            "seeds are agent-authored and unreviewed.",
            "Seed-search proposals are agent decoder candidates awaiting human accept/reject; "
            "drawing them changes no seed and no run.",
        ],
    }
    (root_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return rrd_path


def view_tile(
    entity: str,
    view: str,
    *,
    hull: bool,
    anchor_view: str | None = None,
    proposal_views: tuple[str, ...] = (),
) -> rrb.Spatial2DView:
    """One camera tile: video, part masks, consensus markers, hull projection, and the human
    anchor outlines / seed proposals where they exist for that view."""
    contents = [
        "$origin/video",
        "$origin/masks/**",
        "$origin/consensus_markers",
        *(["$origin/hull_projection/**"] if hull else []),
        *(["$origin/human_anchor_outlines/**"] if view == anchor_view else []),
        *(["$origin/proposals/**"] if view in proposal_views else []),
    ]
    if view == REFERENCE_VIEW:
        seeds = " (human seeds + corrections" + (
            ", anchor outlines)" if view == anchor_view else ")"
        )
    else:
        seeds = " (agent seeds, unreviewed" + (", proposals)" if view in proposal_views else ")")
    return rrb.Spatial2DView(
        origin=f"{entity}/views/{view}", name=f"{view}{seeds}", contents=contents
    )


def static_comparison_blueprint(
    entity: str,
    views: tuple[str, ...],
    hull: bool,
    *,
    anchor_view: str | None = None,
    proposal_views: tuple[str, ...] = (),
) -> rrb.Blueprint:
    grid = rrb.Grid(
        *[
            view_tile(
                entity, view, hull=hull, anchor_view=anchor_view, proposal_views=proposal_views
            )
            for view in views
        ],
        grid_columns=4,
    )
    series_tabs = list(
        time_series_stack(
            f"{entity}/diagnostics/multiview",
            [(target, f"{target}: per-view error (raw px)") for target in TARGETS],
        )
    )
    if anchor_view is not None:
        series_tabs.append(
            time_series_view(
                f"{entity}/{ANCHOR_SERIES}",
                f"human anchor frames ({anchor_view})",
                "$origin/anchor_frame",
            )
        )
    return rrb.Blueprint(
        rrb.Vertical(
            grid,
            rrb.Horizontal(
                rrb.Spatial3DView(
                    origin=f"{entity}/{WORLD_3D}",
                    name="World mm: consensus centroids, dataset hands, cameras"
                    + (", hull voxels (1 fps)" if hull else ""),
                    contents="$origin/**",
                ),
                rrb.Tabs(*series_tabs),
                rrb.TextDocumentView(
                    origin=f"{entity}/metadata/disagreement", name="Disagreement episodes"
                ),
                column_shares=[3, 3, 2],
            ),
            row_shares=[3, 2],
        ),
        rrb.TimePanel(timeline="analysis_time", fps=30),
        auto_layout=False,
        auto_views=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_repository_root(parser)
    add_output_root(parser, OUTPUT_ROOT)
    parser.add_argument("--consensus-root", type=Path, default=CONSENSUS_ROOT)
    parser.add_argument("--hull-root", type=Path, default=HULL_ROOT)
    parser.add_argument("--no-hull", action="store_true")
    parser.add_argument("--mask-every", type=int, default=1)
    parser.add_argument(
        "--anchor-masks",
        type=Path,
        default=ANCHOR_MASKS,
        help="Exported human anchor mask set drawn as outlines on its view at the anchor frames.",
    )
    parser.add_argument("--no-anchor-outlines", action="store_true")
    parser.add_argument(
        "--proposals-root",
        type=Path,
        default=PROPOSALS_ROOT,
        help="battle-seed-search proposals drawn as outlined candidates on each view's tile.",
    )
    parser.add_argument("--no-proposals", action="store_true")
    parser.add_argument("--application-id", default=APPLICATION_ID)
    parser.add_argument(
        "--recording-id",
        default=RECORDING_ID,
        help="Set to the interaction-review package's id so `rerun rrd merge` yields one store.",
    )
    parser.add_argument(
        "--no-blueprint",
        action="store_true",
        help="Do not embed a layout (use when merging into a package that carries .rbl presets).",
    )
    args = parser.parse_args()
    path = build_static_comparison(
        args.repository_root,
        output_root=args.output_root,
        consensus_root=args.consensus_root,
        hull_root=None if args.no_hull else args.hull_root,
        mask_every=args.mask_every,
        anchor_masks=None if args.no_anchor_outlines else args.anchor_masks,
        proposals_root=None if args.no_proposals else args.proposals_root,
        application_id=args.application_id,
        recording_id=args.recording_id,
        embed_blueprint=not args.no_blueprint,
    )
    print(f"{path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
