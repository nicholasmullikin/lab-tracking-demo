"""Projection-only visibility audit of the head-mounted cameras (Track 6).

For each ego view the frame-0..1799 per-frame poses project the Track 5 hull voxels (1 fps;
the consensus centroid plus a sphere when no hull exists) and the dataset hand joints into
the sensor.  The report says, per part and per hand, what fraction of the minute the object
lies inside the frame.  A SAM3 run is warranted only when a part is visible for more than
half of the minute; otherwise the view is recorded as `not_run` with the measured reason.

Visibility here is geometric (inside the image rectangle, in front of the camera); it says
nothing about occlusion by the hands or about what a tracker would do.  CC BY-NC 4.0 applies.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .assembly101_camera_fit import PoseMembers
from .assembly101_fetch_view import EGO_VIEWS
from .assembly101_pose_schemas import ASSEMBLY101_HAND_SIDES
from .four_part_contract import TARGETS
from .multiview_consensus import OUTPUT_ROOT as CONSENSUS_ROOT
from .multiview_geometry import CameraRig
from .multiview_visual_hull import OUTPUT_ROOT as HULL_ROOT

OUTPUT_ROOT = Path("runs/multiview-ego-visibility-audit")
VISIBLE_FRACTION = 0.5
RUN_THRESHOLD = 0.5
FALLBACK_RADIUS_MM = 45.0


def _sphere_samples(centre: np.ndarray, radius: float) -> np.ndarray:
    offsets = np.array(
        [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=np.float64
    )
    offsets = offsets[np.linalg.norm(offsets, axis=1) > 0]
    offsets /= np.linalg.norm(offsets, axis=1, keepdims=True)
    return np.concatenate([centre.reshape(1, 3), centre + radius * offsets])


def inside_fraction(rig: CameraRig, view: str, points: np.ndarray, pose_frame: int) -> float | None:
    if points.shape[0] == 0 or pose_frame not in rig.ego_poses:
        return None
    width, height = rig.image_size(view)
    depth = rig.depth(view, points, pose_frame)
    raw = rig.project(view, points, pose_frame)
    finite = np.isfinite(raw).all(axis=1) & (depth > 0)
    inside = (
        finite & (raw[:, 0] >= 0) & (raw[:, 0] < width) & (raw[:, 1] >= 0) & (raw[:, 1] < height)
    )
    return float(inside.mean())


def audit(
    repository_root: Path,
    *,
    views: tuple[str, ...],
    output_root: Path = OUTPUT_ROOT,
    hull_root: Path = HULL_ROOT,
    consensus_root: Path = CONSENSUS_ROOT,
    frame_count: int = 1800,
) -> dict:
    started = time.monotonic()
    repository_root = repository_root.resolve()
    rig = CameraRig.load(repository_root)
    members = PoseMembers(repository_root)
    hull_manifest_path = repository_root / hull_root / "manifest.json"
    hull_source: str
    part_points: dict[str, dict[int, np.ndarray]] = {t: {} for t in TARGETS}
    if hull_manifest_path.is_file():
        manifest = json.loads(hull_manifest_path.read_text(encoding="utf-8"))
        origin = np.asarray(manifest["grid_origin_mm"])
        voxel = float(manifest["voxel_size_mm"])
        with np.load(repository_root / manifest["voxels_npz_uri"]) as archive:
            for key in archive.files:
                target, frame = key.split("/")
                indices = archive[key]
                if indices.size:
                    part_points[target][int(frame)] = (
                        origin + (indices.astype(np.float64) + 0.5) * voxel
                    )
        hull_source = (
            f"visual hull voxels at {manifest['voxels_fps']} fps ({manifest['voxels_npz_uri']})"
        )
    else:
        with np.load(repository_root / consensus_root / "consensus_points.npz") as archive:
            for target in TARGETS:
                points = archive[f"consensus/{target}"]
                for frame in range(0, frame_count, 30):
                    if not np.isnan(points[frame]).any():
                        part_points[target][frame] = _sphere_samples(
                            points[frame], FALLBACK_RADIUS_MM
                        )
        hull_source = (
            f"consensus centroid + {FALLBACK_RADIUS_MM:.0f} mm sphere samples at 1 fps "
            "(no hull run)"
        )
    report: dict = {
        "manifest_kind": "multiview_ego_visibility_audit",
        "part_geometry_source": hull_source,
        "visible_frame_rule": (
            f"a part is visible on a frame when > {VISIBLE_FRACTION:.0%} of its hull points "
            "project inside the sensor in front of the camera; a view warrants a SAM3 run when a "
            "part is "
            f"visible on > {RUN_THRESHOLD:.0%} of the sampled frames"
        ),
        "views": {},
        "claim_boundaries": [
            "Visibility is geometric (inside the image rectangle with the dataset's per-frame ego "
            "pose and the fitted rational intrinsics); it ignores occlusion by hands and says "
            "nothing about tracker behaviour.",
            "e1/e2 intrinsics are 4-5 px estimates fitted on few hand observations; their "
            "visibility fractions are coarse.",
            "CC BY-NC 4.0 applies to the dataset assets.",
        ],
    }
    for view in views:
        entry: dict = {"parts": {}, "hands": {}, "pose_frames_missing": 0}
        for target in TARGETS:
            fractions = []
            for frame, points in sorted(part_points[target].items()):
                pose = rig.pose_frame(view, frame)
                value = inside_fraction(rig, view, points, pose)
                if value is None:
                    entry["pose_frames_missing"] += 1
                    continue
                fractions.append(value)
            array = np.asarray(fractions)
            entry["parts"][target] = {
                "sampled_frames": int(array.size),
                "mean_inside_fraction": float(array.mean()) if array.size else None,
                "visible_frame_fraction": (
                    float(np.mean(array > VISIBLE_FRACTION)) if array.size else None
                ),
            }
        for index, side in ASSEMBLY101_HAND_SIDES.items():
            fractions = []
            for frame in range(frame_count):
                pose = rig.pose_frame(view, frame)
                key = str(pose)
                if (
                    key not in members.landmarks3d
                    or float(members.confidences[key][str(index)]) < 0.5
                ):
                    continue
                value = inside_fraction(
                    rig,
                    view,
                    np.asarray(members.landmarks3d[key][str(index)], dtype=np.float64),
                    pose,
                )
                if value is not None:
                    fractions.append(value)
            array = np.asarray(fractions)
            entry["hands"][side] = {
                "frames_with_confident_pose": int(array.size),
                "mean_joints_inside_fraction": float(array.mean()) if array.size else None,
                "frames_with_wrist_region_inside_fraction": (
                    float(np.mean(array > VISIBLE_FRACTION)) if array.size else None
                ),
            }
        best = max(
            (
                (item["visible_frame_fraction"] or 0.0, target)
                for target, item in entry["parts"].items()
            ),
            default=(0.0, None),
        )
        entry["best_part"] = {"target": best[1], "visible_frame_fraction": best[0]}
        entry["decision"] = (
            "run_sam3_warranted"
            if best[0] > RUN_THRESHOLD
            else (
                f"not_run: best part {best[1]} visible on {best[0]:.0%} of sampled frames (<= 50 %)"
            )
        )
        report["views"][view] = entry
    report["runtime_seconds"] = time.monotonic() - started
    root = repository_root / output_root
    root.mkdir(parents=True, exist_ok=True)
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--view", action="append", default=[], help=f"ego view(s); default all of {EGO_VIEWS}"
    )
    args = parser.parse_args()
    report = audit(
        args.repository_root, views=tuple(args.view) or EGO_VIEWS, output_root=args.output_root
    )
    print(f"part geometry: {report['part_geometry_source']}")
    for view, entry in report["views"].items():
        parts = ", ".join(
            f"{t} {item['visible_frame_fraction']:.0%}"
            if item["visible_frame_fraction"] is not None
            else f"{t} n/a"
            for t, item in entry["parts"].items()
        )
        hands = ", ".join(
            f"{side} {item['mean_joints_inside_fraction']:.0%}"
            if item["mean_joints_inside_fraction"] is not None
            else f"{side} n/a"
            for side, item in entry["hands"].items()
        )
        print(
            f"{view}: parts visible [{parts}]; hand joints inside [{hands}] -> {entry['decision']}"
        )


if __name__ == "__main__":
    main()
