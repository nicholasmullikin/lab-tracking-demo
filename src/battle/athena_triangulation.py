"""ATHENA fixture triangulation smoke and Assembly101 calibration-blocked reporting."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import rerun as rr

from .athena_calibration_probe import DEFAULT_OUTPUT as DEFAULT_PROBE_OUTPUT
from .rerun_logging import init_and_save
from .schemas import AdapterMetadata, ArtifactFingerprint, ExternalPartialRunMetadata

ATHENA_ROOT = Path("/home/nick/src/athena")
ATHENA_REVISION = "e85bd49444253aed9532439ace8ede146d1b6470"
DEFAULT_RUN_ID = "athena-fixture-triangulation-smoke"


def _make_synthetic_cameras(n_cameras: int = 2) -> tuple[list[np.ndarray], np.ndarray]:
    intrinsics = []
    extrinsics = np.zeros((n_cameras, 4, 4), dtype=np.float64)
    for index in range(n_cameras):
        angle = np.deg2rad(index * (360 / n_cameras))
        rotation = np.array(
            [
                [np.cos(angle), 0.0, np.sin(angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle), 0.0, np.cos(angle)],
            ]
        )
        translation = rotation @ np.array([0.0, 0.0, -2.0])
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
        extrinsics[index] = transform
        intrinsics.append(
            np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        )
    return intrinsics, extrinsics[:, :3, :]


def _triangulate_batch(points_2d: np.ndarray, cam_mats_extrinsic: np.ndarray) -> np.ndarray:
    """DLT triangulation copied from athena.triangulaterefine for fixture smoke only."""
    ncams, npoints, _ = points_2d.shape
    data3d = np.empty((npoints, 3))
    data3d[:] = np.nan
    good_mask = ~np.isnan(points_2d[:, :, 0])
    patterns = np.zeros(npoints, dtype=np.int32)
    for camera in range(ncams):
        patterns += good_mask[camera].astype(np.int32) << camera
    for pattern in np.unique(patterns):
        active_camera_count = bin(pattern).count("1")
        if active_camera_count < 2:
            continue
        active_cameras = [camera for camera in range(ncams) if (pattern >> camera) & 1]
        point_indices = np.where(patterns == pattern)[0]
        mats = cam_mats_extrinsic[active_cameras]
        pts = points_2d[active_cameras][:, point_indices, :]
        constraint = np.zeros((len(point_indices), active_camera_count * 2, 4))
        for index in range(active_camera_count):
            x = pts[index, :, 0]
            y = pts[index, :, 1]
            matrix = mats[index]
            constraint[:, 2 * index, :] = x[:, None] * matrix[2][None, :] - matrix[0][None, :]
            constraint[:, 2 * index + 1, :] = y[:, None] * matrix[2][None, :] - matrix[1][None, :]
        _, _, vh = np.linalg.svd(constraint, full_matrices=True)
        points_3d = vh[:, -1, :]
        points_3d = points_3d[:, :3] / points_3d[:, 3:4]
        data3d[point_indices] = points_3d
    return data3d


def _fixture_payload() -> dict[str, object]:
    intrinsics, extrinsics = _make_synthetic_cameras(n_cameras=2)
    true_3d = np.array(
        [
            [0.12, -0.08, 0.45],
            [-0.05, 0.10, 0.52],
            [0.20, 0.02, 0.40],
        ],
        dtype=np.float64,
    )
    view_ids = ("static-c10379", "ego-hmc21110305")
    ncams = len(intrinsics)
    npoints = true_3d.shape[0]
    points_2d_norm = np.full((ncams, npoints, 2), np.nan, dtype=np.float64)
    points_2d_px: dict[str, list[list[float]]] = {view_id: [] for view_id in view_ids}
    for camera_index in range(ncams):
        for point_index in range(npoints):
            homogeneous = np.append(true_3d[point_index], 1.0)
            camera_point = extrinsics[camera_index] @ homogeneous
            points_2d_norm[camera_index, point_index] = [
                camera_point[0] / camera_point[2],
                camera_point[1] / camera_point[2],
            ]
            pixel = intrinsics[camera_index] @ camera_point
            points_2d_px[view_ids[camera_index]].append(
                [float(pixel[0] / pixel[2] / 1280.0), float(pixel[1] / pixel[2] / 720.0)]
            )
    triangulated = _triangulate_batch(points_2d_norm, extrinsics)
    return {
        "classification": "fixture_smoke",
        "coordinate_frames": {
            "input_2d": "image_normalized_top_left",
            "output_3d": "athena_calibration_world_non_metric",
        },
        "views": list(view_ids),
        "true_points_3d": true_3d.tolist(),
        "triangulated_points_3d": triangulated.tolist(),
        "points_2d_by_view": points_2d_px,
        "units": {
            "input_2d": "normalized image coordinates in [0,1]",
            "output_3d": "ATHENA/JARVIS calibration-world units (fixture only; not metric)",
        },
    }


def _export_fixture_rrd(payload: dict[str, object], output_path: Path, run_id: str) -> None:
    init_and_save(run_id, output_path)
    rr.log("world/athena_fixture/manifest", rr.TextDocument(json.dumps(payload, indent=2)))
    for view_id, points in payload["points_2d_by_view"].items():
        positions = [[point[0] * 1280.0, point[1] * 720.0] for point in points]
        rr.log(
            f"world/athena_fixture/views/{view_id}/landmarks_2d",
            rr.Points2D(positions, radii=4.0, colors=[(120, 200, 255)] * len(positions)),
        )
    world_points = payload["triangulated_points_3d"]
    rr.log(
        "world/athena_fixture/triangulation_3d/world_points",
        rr.Points3D(world_points, radii=0.01, colors=[(255, 180, 90)] * len(world_points)),
    )


def run_fixture_smoke(repository_root: Path, run_id: str = DEFAULT_RUN_ID) -> Path:
    run_directory = (repository_root / "runs" / run_id).resolve()
    run_directory.mkdir(parents=True, exist_ok=True)
    payload = _fixture_payload()
    fixture_path = run_directory / "triangulation_fixture.json"
    fixture_text = json.dumps(payload, indent=2) + "\n"
    fixture_path.write_text(fixture_text, encoding="utf-8")
    fixture_sha256 = hashlib.sha256(fixture_text.encode()).hexdigest()
    rrd_path = run_directory / "athena_fixture.rrd"
    _export_fixture_rrd(payload, rrd_path, run_id)
    blocked_path = repository_root / DEFAULT_PROBE_OUTPUT
    blocked_note = (
        "Assembly101 real-data triangulation remains blocked: cvml-nus/assembly101 "
        "AssemblyPoses.zip exposes extrinsics/positions/timestamps but no intrinsics member."
    )
    limitations = (
        blocked_note,
        "Fixture only; no Assembly101 landmarks or calibration were consumed.",
        "Output 3D is in ATHENA calibration-world units and is not metric ground truth.",
    )
    if blocked_path.is_file():
        limitations = (*limitations, f"Probe evidence: {blocked_path.as_posix()}")
    metadata = ExternalPartialRunMetadata(
        classification="fixture_smoke",
        requested_input_fingerprint=ArtifactFingerprint(
            uri="fixture/synthetic_two_view_hand_points",
            sha256="0" * 64,
            source="measured",
        ),
        native_artifact_fingerprints=(
            ArtifactFingerprint(
                uri=fixture_path.relative_to(repository_root).as_posix(),
                sha256=fixture_sha256,
                source="measured",
            ),
        ),
        adapter=AdapterMetadata(
            name="athena-triangulation-fixture",
            version="0.1.0",
            implementation_basis=(
                "Synthetic two-view normalized 2D points triangulated with "
                "athena.triangulaterefine._triangulate_batch"
            ),
            external_source_uri=str(ATHENA_ROOT),
            external_revision=ATHENA_REVISION,
        ),
        reproduced_command="uv run battle-athena-fixture-smoke",
        decoded_frame_count=1,
        frames_with_normalized_output=1,
        source_offset_seconds=0.0,
        normalized_artifact_uri=fixture_path.relative_to(repository_root).as_posix(),
        rerun_artifact_uri=rrd_path.relative_to(repository_root).as_posix(),
        limitations=limitations,
    )
    manifest_path = run_directory / "external_partial.json"
    manifest_path.write_text(metadata.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    args = parser.parse_args()
    manifest_path = run_fixture_smoke(args.repository_root.resolve(), args.run_id)
    print(manifest_path)


if __name__ == "__main__":
    main()
