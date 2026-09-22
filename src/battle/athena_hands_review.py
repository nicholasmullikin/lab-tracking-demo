"""Rerun logging for `battle-athena-hands` runs: triangulated hands beside the dataset hands.

Two entry points.  `build_recording` writes a standalone recording with the dataset hands, the
triangulated (raw and smoothed) hands and every contributing camera's frustum in one world-mm
3D view, plus time series of contributing view count, wrist disagreement and per-view
reprojection RMS.  `log_static` / `log_frame` are the hooks the v4 review calls under its
`athena_hands` layer so the same entities land inside the existing world-mm view.

Everything logged here is cross-source comparison evidence: triangulated positions come from a
DLT on MediaPipe/WiLoR detections through fitted intrinsics; the dataset hands are the
dataset's own tracker output.  Neither is ground truth for the other.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from . import assembly101_reference as a101
from .assembly101_pose_schemas import ASSEMBLY101_EDGES, Assembly101HandFrame
from .athena_hands import (
    COMMON_EDGES,
    COMMON_JOINT_NAMES,
    DATASET_REFERENCE_RUN,
    DEFAULT_OUTPUT_ROOT,
    SIDES,
    WRIST,
    AthenaHandsManifest,
    load_manifest,
)
from .cli_common import add_repository_root
from .multiview_geometry import CameraRig
from .rerun_logging import init_and_save, time_series_view

THREE_D_ROOT = "contexts/assembly101_world_mm_3d"
# Entity label per arm: the v4 review logs the MediaPipe arm under `athena_hands` and the
# three-view WiLoR arm beside it under `athena_hands_wilor`, so both sit in one 3D view.
DEFAULT_LABEL = "athena_hands"
WILOR_LABEL = "athena_hands_wilor"
WILOR_OUTPUT_ROOT = Path("runs/athena-hands-first-minute-wilor")
HANDS_ENTITY = f"{THREE_D_ROOT}/{DEFAULT_LABEL}"
CAMERAS_ENTITY = f"{THREE_D_ROOT}/camera"
DIAGNOSTICS = f"diagnostics/assembly101/{DEFAULT_LABEL}"
DATASET_ENTITY = f"{THREE_D_ROOT}/hands"
HAND_COLORS: dict[str, tuple[int, int, int]] = {
    "left": (255, 140, 60),
    "right": (90, 170, 255),
}
SMOOTHED_COLORS: dict[str, tuple[int, int, int]] = {
    "left": (200, 90, 30),
    "right": (40, 110, 200),
}
WILOR_HAND_COLORS: dict[str, tuple[int, int, int]] = {
    "left": (255, 80, 200),
    "right": (80, 230, 120),
}
WILOR_SMOOTHED_COLORS: dict[str, tuple[int, int, int]] = {
    "left": (190, 40, 150),
    "right": (30, 170, 80),
}
ARM_COLORS: dict[str, tuple[dict[str, tuple[int, int, int]], dict[str, tuple[int, int, int]]]] = {
    DEFAULT_LABEL: (HAND_COLORS, SMOOTHED_COLORS),
    WILOR_LABEL: (WILOR_HAND_COLORS, WILOR_SMOOTHED_COLORS),
}


def hands_entity(label: str) -> str:
    return f"{THREE_D_ROOT}/{label}"


def diagnostics_entity(label: str) -> str:
    return f"diagnostics/assembly101/{label}"


DATASET_COLORS: dict[str, tuple[int, int, int]] = {
    "left": (255, 235, 130),
    "right": (140, 255, 235),
}
CAMERA_PLANE_MM = 300.0


@dataclass
class LoadedAthenaHands:
    manifest: AthenaHandsManifest
    run_directory: Path
    frames: np.ndarray
    points: np.ndarray  # (F, 2, J, 3)
    smoothed: np.ndarray
    used: np.ndarray  # (F, 2, J, V)
    reprojection_px: np.ndarray  # (F, 2, J, V)
    dataset: np.ndarray  # (F, 2, J, 3)
    dataset_confidence: np.ndarray  # (F, 2)
    views: tuple[str, ...]


def load_run(run_directory: Path) -> LoadedAthenaHands:
    manifest = load_manifest(run_directory)
    with np.load(run_directory / "hands.npz") as archive:
        return LoadedAthenaHands(
            manifest=manifest,
            run_directory=run_directory,
            frames=np.asarray(archive["frames"]),
            points=np.asarray(archive["points"]),
            smoothed=np.asarray(archive["smoothed"]),
            used=np.asarray(archive["used"]),
            reprojection_px=np.asarray(archive["reprojection_px"]),
            dataset=np.asarray(archive["dataset"]),
            dataset_confidence=np.asarray(archive["dataset_confidence"]),
            views=tuple(str(v) for v in archive["views"]),
        )


def static_rig(repository_root: Path, run: LoadedAthenaHands) -> CameraRig:
    """Rig restricted to the run's static cameras (frusta only; no ego poses are read)."""
    return CameraRig.load(
        repository_root, views=tuple(view for view in run.views if not view.startswith("HMC_"))
    )


def load_run_if_present(run_directory: Path) -> LoadedAthenaHands | None:
    if (
        not (run_directory / "manifest.json").is_file()
        or not (run_directory / "hands.npz").is_file()
    ):
        return None
    return load_run(run_directory)


# -- hooks ---------------------------------------------------------------------------------------


def log_static(
    entity: str, run: LoadedAthenaHands, rig: CameraRig | None, *, label: str = DEFAULT_LABEL
) -> None:
    """Frusta for every contributing static camera and the diagnostics series names."""
    hand_colors, _ = ARM_COLORS.get(label, (HAND_COLORS, SMOOTHED_COLORS))
    diagnostics = diagnostics_entity(label)
    source = run.manifest.hand_source
    if rig is not None:
        for view in run.views:
            if rig.camera(view).is_ego:
                continue
            pose = rig.camera_to_world(view)
            camera_root = f"{entity}/{CAMERAS_ENTITY}/{view}"
            rr.log(
                camera_root,
                rr.Transform3D(translation=pose[:3, 3], mat3x3=pose[:3, :3]),
                static=True,
            )
            rr.log(
                camera_root,
                rr.Pinhole(
                    image_from_camera=rig._intrinsics[view],
                    resolution=list(rig.image_size(view)),
                    camera_xyz=rr.ViewCoordinates.RDF,
                    image_plane_distance=CAMERA_PLANE_MM,
                ),
                static=True,
            )
    for side in SIDES:
        rr.log(
            f"{entity}/{diagnostics}/contributing_views/{side}",
            rr.SeriesLines(
                names=f"{side} wrist ({source}): contributing views", colors=[hand_colors[side]]
            ),
            static=True,
        )
        rr.log(
            f"{entity}/{diagnostics}/wrist_disagreement_mm/{side}",
            rr.SeriesLines(
                names=f"{side} wrist ({source}): triangulated vs dataset (mm)",
                colors=[hand_colors[side]],
            ),
            static=True,
        )
    for index, view in enumerate(run.views):
        shade = 60 + (index * 23) % 160
        rr.log(
            f"{entity}/{diagnostics}/reprojection_rms_px/{view}",
            rr.SeriesLines(
                names=f"{view} reprojection RMS ({source}, px, used points)",
                colors=[(shade, 200 - shade // 2, 255 - shade)],
            ),
            static=True,
        )
    rr.log(
        f"{entity}/{hands_entity(label)}/manifest",
        rr.TextDocument(run.manifest.model_dump_json(indent=2), media_type="application/json"),
        static=True,
    )


def _scalar_or_clear(path: str, value: float | None) -> None:
    if value is None or not np.isfinite(value):
        rr.log(path, rr.Clear(recursive=False))
    else:
        rr.log(path, rr.Scalars([float(value)]))


def log_frame(
    entity: str, run: LoadedAthenaHands, frame: int, *, label: str = DEFAULT_LABEL
) -> None:
    """Triangulated hands (raw solid, smoothed thin) and the per-frame series."""
    hands_root = f"{entity}/{hands_entity(label)}"
    diagnostics = diagnostics_entity(label)
    hand_colors, smoothed_colors = ARM_COLORS.get(label, (HAND_COLORS, SMOOTHED_COLORS))
    source = run.manifest.hand_source
    points: list[list[float]] = []
    labels: list[str] = []
    point_colors: list[tuple[int, int, int]] = []
    strips: list[list[list[float]]] = []
    strip_colors: list[tuple[int, int, int]] = []
    smooth_strips: list[list[list[float]]] = []
    smooth_colors: list[tuple[int, int, int]] = []
    for s, side in enumerate(SIDES):
        raw = run.points[frame, s]
        solved = ~np.isnan(raw[:, 0])
        wrist_views = int(run.used[frame, s, WRIST].sum()) if solved[WRIST] else None
        _scalar_or_clear(f"{entity}/{diagnostics}/contributing_views/{side}", wrist_views)
        disagreement = None
        if solved[WRIST] and run.dataset_confidence[frame, s] >= 0.5:
            disagreement = float(np.linalg.norm(raw[WRIST] - run.dataset[frame, s, WRIST]))
        _scalar_or_clear(f"{entity}/{diagnostics}/wrist_disagreement_mm/{side}", disagreement)
        if not solved.any():
            continue
        for j in np.where(solved)[0]:
            points.append(raw[j].tolist())
            labels.append(f"triangulated {source} {side}: {COMMON_JOINT_NAMES[j]}")
            point_colors.append(hand_colors[side])
        for a, b in COMMON_EDGES:
            if solved[a] and solved[b]:
                strips.append([raw[a].tolist(), raw[b].tolist()])
                strip_colors.append(hand_colors[side])
        smooth = run.smoothed[frame, s]
        smooth_ok = ~np.isnan(smooth[:, 0])
        for a, b in COMMON_EDGES:
            if smooth_ok[a] and smooth_ok[b]:
                smooth_strips.append([smooth[a].tolist(), smooth[b].tolist()])
                smooth_colors.append(smoothed_colors[side])
    for v, view in enumerate(run.views):
        used = run.used[frame, :, :, v]
        errors = run.reprojection_px[frame, :, :, v][used]
        _scalar_or_clear(
            f"{entity}/{diagnostics}/reprojection_rms_px/{view}",
            float(np.sqrt(np.mean(errors**2))) if errors.size else None,
        )
    if not points:
        rr.log(hands_root, rr.Clear(recursive=True))
        return
    rr.log(
        f"{hands_root}/joints", rr.Points3D(points, labels=labels, colors=point_colors, radii=4.0)
    )
    rr.log(f"{hands_root}/skeletons", rr.LineStrips3D(strips, colors=strip_colors, radii=2.0))
    if smooth_strips:
        rr.log(
            f"{hands_root}/skeletons_smoothed",
            rr.LineStrips3D(smooth_strips, colors=smooth_colors, radii=0.8),
        )
    else:
        rr.log(f"{hands_root}/skeletons_smoothed", rr.Clear(recursive=False))


def log_dataset_frame(entity: str, frame: Assembly101HandFrame, *, draw_threshold: float) -> None:
    """Dataset hands in world mm (same drawing as the v4 dataset layer, 3D only)."""
    root = f"{entity}/{DATASET_ENTITY}"
    drawn = [hand for hand in frame.hands if hand.confidence >= draw_threshold]
    if not drawn:
        rr.log(root, rr.Clear(recursive=True))
        return
    points: list[list[float]] = []
    strips: list[list[list[float]]] = []
    colors: list[tuple[int, int, int]] = []
    strip_colors: list[tuple[int, int, int]] = []
    for hand in drawn:
        world = [[p.x, p.y, p.z] for p in hand.joints_world_mm]
        points.extend(world)
        colors.extend([DATASET_COLORS[hand.side]] * len(world))
        strips.extend([[world[a], world[b]] for a, b in ASSEMBLY101_EDGES])
        strip_colors.extend([DATASET_COLORS[hand.side]] * len(ASSEMBLY101_EDGES))
    rr.log(f"{root}/joints", rr.Points3D(points, colors=colors, radii=3.0))
    rr.log(f"{root}/skeletons", rr.LineStrips3D(strips, colors=strip_colors, radii=1.5))


# -- standalone recording ----------------------------------------------------------------------


def blueprint(root: str, run: LoadedAthenaHands) -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin=f"{root}/{THREE_D_ROOT}",
                name=(
                    "World mm: dataset hands (yellow/mint) vs triangulated "
                    f"{run.manifest.hand_source} hands (orange/blue; thin = smoothed), "
                    f"{len(run.views)} cameras"
                ),
                contents="$origin/**",
            ),
            rrb.Vertical(
                time_series_view(
                    f"{root}/{DIAGNOSTICS}",
                    "Contributing views (wrist) and wrist disagreement vs dataset (mm)",
                    (
                        "$origin/contributing_views/**",
                        "$origin/wrist_disagreement_mm/**",
                    ),
                ),
                time_series_view(
                    f"{root}/{DIAGNOSTICS}/reprojection_rms_px",
                    "Per-view reprojection RMS of used points (px)",
                ),
                rrb.TextDocumentView(origin=f"{root}/{HANDS_ENTITY}/manifest", name="Manifest"),
            ),
            column_shares=[3, 2],
        ),
        collapse_panels=True,
    )


def build_recording(
    repository_root: Path,
    *,
    run_directory: Path,
    output: Path | None = None,
    dataset_reference: Path = DATASET_REFERENCE_RUN,
) -> Path:
    repository_root = repository_root.resolve()
    run = load_run(repository_root / run_directory)
    output = output or (repository_root / run_directory / "hands.rrd")
    dataset = a101.load_reference(
        dataset_reference, repository_root, frame_count=run.manifest.frame_count, verify=False
    )
    rig = CameraRig.load(repository_root, views=run.views)
    root = "athena_hands"
    init_and_save(run.manifest.run_id, output, default_blueprint=blueprint(root, run))
    log_static(root, run, rig)
    for frame in range(run.manifest.frame_count):
        rr.set_time("analysis_frame", sequence=frame)
        rr.set_time("analysis_time", duration=frame / run.manifest.analysis_fps)
        log_dataset_frame(
            root, dataset.frames[frame], draw_threshold=dataset.manifest.draw_confidence_threshold
        )
        log_frame(root, run, frame)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_repository_root(parser)
    parser.add_argument("--run", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    print(build_recording(args.repository_root, run_directory=args.run, output=args.output))


if __name__ == "__main__":
    main()
