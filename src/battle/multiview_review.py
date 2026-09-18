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

OUTPUT_ROOT = Path("runs/multiview-static-comparison-first-minute")
RECORDING_NAME = "multiview_static_comparison.rrd"
HULL_ROOT = Path("runs/multiview-visual-hull-first-minute")
PART_COLORS = mask_cache.REVIEW_COLORS
HAND_COLORS = {"left": (255, 235, 130), "right": (140, 255, 235)}
CAMERA_PLANE_MM = 250.0
WORLD_3D = "world_mm_3d"


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


# -- the eight-view recording ----------------------------------------------------------------


def build_static_comparison(
    repository_root: Path,
    *,
    output_root: Path = OUTPUT_ROOT,
    consensus_root: Path = CONSENSUS_ROOT,
    hull_root: Path | None = HULL_ROOT,
    mask_every: int = 1,
) -> Path:
    from .muggled_smoke import _create_bounded_rerun_video

    started = time.monotonic()
    repository_root = repository_root.resolve()
    layer = MultiviewLayer(repository_root, consensus_root)
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
    rr.init("battle-multiview-static-comparison", recording_id="multiview_static_comparison")
    rr.save(rrd_path)
    entity = "world/assembly101_multiview_first_minute"
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
                        next((o for o in observation.objects if o.label == target and o.mask), None)
                        if observation is not None
                        else None
                    )
                    path = f"{view_root}/masks/{target}"
                    if item is None or item.mask is None:
                        rr.log(path, rr.Clear(recursive=False))
                        continue
                    rr.log(
                        path,
                        rr.EncodedImage(
                            contents=run.cache.rgba_png(item.mask.uri, PART_COLORS[target]),
                            media_type="image/png",
                            opacity=0.4,
                            draw_order=1.0,
                        ),
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
                pixel = rig.project(view, point.reshape(1, 3))[0] / scale
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
        log_consensus_3d(f"{entity}/{WORLD_3D}", layer, frame)
        log_dataset_hands_3d(
            f"{entity}/{WORLD_3D}",
            dataset.frames[frame],
            dataset.manifest.draw_confidence_threshold,
        )
        if hull is not None:
            log_hull_3d(f"{entity}/{WORLD_3D}", hull, frame)
        log_error_series_frame(f"{entity}/diagnostics/multiview", layer, frame)
    rr.send_blueprint(_static_comparison_blueprint(entity, tuple(runs), hull is not None))
    rr.disconnect()
    index = {
        "manifest_kind": "multiview_static_comparison",
        "recording": rrd_path.relative_to(repository_root).as_posix(),
        "consensus_manifest": layer.manifest.per_frame_fingerprint.model_dump(mode="json")
        if layer.manifest.per_frame_fingerprint
        else None,
        "views": list(runs),
        "hull_overlay": hull is not None,
        "mask_every": mask_every,
        "runtime_seconds": time.monotonic() - started,
        "claim_boundaries": list(layer.manifest.claim_boundaries),
    }
    (root_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return rrd_path


def _static_comparison_blueprint(entity: str, views: tuple[str, ...], hull: bool) -> rrb.Blueprint:
    def view_2d(view: str) -> rrb.Spatial2DView:
        contents = [
            "$origin/video",
            "$origin/masks/**",
            "$origin/consensus_markers",
            *(["$origin/hull_projection/**"] if hull else []),
        ]
        return rrb.Spatial2DView(
            origin=f"{entity}/views/{view}",
            name=f"{view}" + (" (human seeds)" if view == REFERENCE_VIEW else " (agent seeds)"),
            contents=contents,
        )

    grid = rrb.Grid(*[view_2d(view) for view in views], grid_columns=4)
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
                rrb.Tabs(
                    *[
                        rrb.TimeSeriesView(
                            origin=f"{entity}/diagnostics/multiview/{target}",
                            name=f"{target}: per-view error (raw px)",
                            contents="$origin/**",
                        )
                        for target in TARGETS
                    ]
                ),
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
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--consensus-root", type=Path, default=CONSENSUS_ROOT)
    parser.add_argument("--hull-root", type=Path, default=HULL_ROOT)
    parser.add_argument("--no-hull", action="store_true")
    parser.add_argument("--mask-every", type=int, default=1)
    args = parser.parse_args()
    path = build_static_comparison(
        args.repository_root,
        output_root=args.output_root,
        consensus_root=args.consensus_root,
        hull_root=None if args.no_hull else args.hull_root,
        mask_every=args.mask_every,
    )
    print(f"{path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
