"""Inference-free Rerun recording for a `battle-kineo-multiview` evaluation.

One world-millimetre 3D view holds the dataset cameras (from the tracked camera estimates, as
`interaction_review_v4._log_assembly101_static` draws them), Kineo's cameras after the
evaluation's similarity alignment, Kineo's SMPL-X body joints per frame (aligned) and the
dataset hands; a time panel carries the per-side wrist disagreement.  Everything is read from
`manifest.json`, `body_aligned_mm.npz` and the dataset reference window; nothing is inferred.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from . import assembly101_reference as a101
from .athena_hands_review import CAMERA_PLANE_MM, THREE_D_ROOT, log_dataset_frame
from .kineo_multiview import (
    DATASET_REFERENCE_RUN,
    KINEO_WRIST_NAMES,
    SIDES,
    KineoMultiviewManifest,
    load_evaluation,
    load_prepare,
)
from .multiview_geometry import CameraRig
from .rerun_logging import init_and_save, time_series_view

ROOT = "kineo_multiview"
BODY_ENTITY = f"{THREE_D_ROOT}/kineo_body"
DATASET_CAMERAS_ENTITY = f"{THREE_D_ROOT}/camera"
KINEO_CAMERAS_ENTITY = f"{THREE_D_ROOT}/kineo_camera"
DIAGNOSTICS = "diagnostics/assembly101/kineo_multiview"
BODY_COLOR = (120, 220, 120)
WRIST_COLORS: dict[str, tuple[int, int, int]] = {"left": (255, 140, 60), "right": (90, 170, 255)}
KINEO_CAMERA_COLOR = (255, 80, 200)


@dataclass
class LoadedKineoBody:
    manifest: KineoMultiviewManifest
    xyz: np.ndarray  # (F, J, 3) aligned mm, NaN where Kineo produced nothing
    scores: np.ndarray
    names: tuple[str, ...]
    edges: np.ndarray  # (E, 2)
    dataset_wrists: np.ndarray  # (F, 2, 3)
    camera_centers: np.ndarray  # (V, 3) aligned mm
    camera_rotations: np.ndarray  # (V, 3, 3) camera-to-world after alignment
    intrinsics: np.ndarray  # (V, 3, 3) Kineo K for the proxy
    resolution_hw: np.ndarray  # (V, 2)
    views: tuple[str, ...]


def load_body(run_directory: Path) -> LoadedKineoBody:
    manifest = load_evaluation(run_directory)
    with np.load(run_directory / "body_aligned_mm.npz") as archive:
        return LoadedKineoBody(
            manifest=manifest,
            xyz=np.asarray(archive["xyz"]),
            scores=np.asarray(archive["scores"]),
            names=tuple(str(n) for n in archive["names"]),
            edges=np.asarray(archive["edges"]).reshape(-1, 2),
            dataset_wrists=np.asarray(archive["dataset_wrists"]),
            camera_centers=np.asarray(archive["kineo_camera_centers_aligned_mm"]),
            camera_rotations=np.asarray(archive["kineo_camera_rotations_c2w_aligned"]),
            intrinsics=np.asarray(archive["kineo_intrinsics"]),
            resolution_hw=np.asarray(archive["kineo_resolution_hw"]),
            views=tuple(str(v) for v in archive["views"]),
        )


def _log_pinhole(entity: str, pose_c2w: np.ndarray, intrinsics: np.ndarray, resolution_wh) -> None:
    rr.log(
        entity,
        rr.Transform3D(translation=pose_c2w[:3, 3], mat3x3=pose_c2w[:3, :3]),
        static=True,
    )
    rr.log(
        entity,
        rr.Pinhole(
            image_from_camera=np.asarray(intrinsics, dtype=np.float64),
            resolution=[int(resolution_wh[0]), int(resolution_wh[1])],
            camera_xyz=rr.ViewCoordinates.RDF,
            image_plane_distance=CAMERA_PLANE_MM,
        ),
        static=True,
    )


def log_static(root: str, body: LoadedKineoBody, rig: CameraRig) -> None:
    for index, view in enumerate(body.views):
        _log_pinhole(
            f"{root}/{DATASET_CAMERAS_ENTITY}/{view}",
            rig.camera_to_world(view),
            rig._intrinsics[view],
            rig.image_size(view),
        )
        pose = np.eye(4)
        pose[:3, :3] = body.camera_rotations[index]
        pose[:3, 3] = body.camera_centers[index]
        height, width = body.resolution_hw[index]
        _log_pinhole(
            f"{root}/{KINEO_CAMERAS_ENTITY}/{view}", pose, body.intrinsics[index], (width, height)
        )
        rr.log(
            f"{root}/{KINEO_CAMERAS_ENTITY}/{view}/label",
            rr.Points3D(
                [body.camera_centers[index]],
                labels=[f"kineo {view}"],
                colors=[KINEO_CAMERA_COLOR],
                radii=8.0,
            ),
            static=True,
        )
    for side in SIDES:
        rr.log(
            f"{root}/{DIAGNOSTICS}/wrist_disagreement_mm/{side}",
            rr.SeriesLines(
                names=f"{side} wrist: Kineo body vs dataset hand (mm)", colors=[WRIST_COLORS[side]]
            ),
            static=True,
        )
    rr.log(
        f"{root}/{BODY_ENTITY}/manifest",
        rr.TextDocument(body.manifest.model_dump_json(indent=2), media_type="application/json"),
        static=True,
    )


def log_frame(root: str, body: LoadedKineoBody, frame: int) -> None:
    joints = body.xyz[frame]
    present = np.isfinite(joints[:, 0])
    for side in SIDES:
        s = SIDES.index(side)
        wrist = joints[body.names.index(KINEO_WRIST_NAMES[side])]
        dataset = body.dataset_wrists[frame, s]
        path = f"{root}/{DIAGNOSTICS}/wrist_disagreement_mm/{side}"
        if np.isfinite(wrist).all() and np.isfinite(dataset).all():
            rr.log(path, rr.Scalars([float(np.linalg.norm(wrist - dataset))]))
        else:
            rr.log(path, rr.Clear(recursive=False))
    entity = f"{root}/{BODY_ENTITY}"
    if not present.any():
        rr.log(entity, rr.Clear(recursive=True))
        return
    rr.log(
        f"{entity}/joints",
        rr.Points3D(
            joints[present],
            labels=[f"kineo {name}" for name, ok in zip(body.names, present, strict=True) if ok],
            colors=[BODY_COLOR] * int(present.sum()),
            radii=6.0,
        ),
    )
    strips = [
        [joints[a].tolist(), joints[b].tolist()] for a, b in body.edges if present[a] and present[b]
    ]
    if strips:
        rr.log(f"{entity}/skeleton", rr.LineStrips3D(strips, colors=[BODY_COLOR], radii=3.0))
    else:
        rr.log(f"{entity}/skeleton", rr.Clear(recursive=False))


def blueprint(root: str, body: LoadedKineoBody) -> rrb.Blueprint:
    arm = body.manifest.arm
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin=f"{root}/{THREE_D_ROOT}",
                name=(
                    f"World mm ({arm}): dataset cameras vs Kineo cameras (magenta labels), "
                    "Kineo SMPL-X body (green), dataset hands (yellow/mint)"
                ),
                contents="$origin/**",
            ),
            rrb.Vertical(
                time_series_view(
                    f"{root}/{DIAGNOSTICS}/wrist_disagreement_mm",
                    "Kineo body wrist vs dataset hand wrist (mm)",
                ),
                rrb.TextDocumentView(origin=f"{root}/{BODY_ENTITY}/manifest", name="Manifest"),
            ),
            column_shares=[3, 2],
        ),
        collapse_panels=True,
    )


def build_recording(
    repository_root: Path,
    run_directory: Path,
    *,
    output: Path | None = None,
    dataset_reference: Path = DATASET_REFERENCE_RUN,
) -> Path:
    repository_root = repository_root.resolve()
    run_directory = (repository_root / run_directory).resolve()
    body = load_body(run_directory)
    prepared = load_prepare(run_directory)
    output = output or (run_directory / "comparison.rrd")
    dataset = a101.load_reference(
        dataset_reference, repository_root, frame_count=prepared.frame_count, verify=False
    )
    rig = CameraRig.load(repository_root, views=body.views)
    init_and_save(prepared.run_id, output, default_blueprint=blueprint(ROOT, body))
    log_static(ROOT, body, rig)
    for frame in range(prepared.frame_count):
        rr.set_time("analysis_frame", sequence=frame)
        rr.set_time("analysis_time", duration=frame / prepared.analysis_fps)
        log_dataset_frame(
            ROOT, dataset.frames[frame], draw_threshold=dataset.manifest.draw_confidence_threshold
        )
        log_frame(ROOT, body, frame)
    return output
