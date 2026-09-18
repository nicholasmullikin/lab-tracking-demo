"""Run ATHENA's triangulation and smoothing inside ATHENA's own virtualenv.

Invoked by `battle.athena_hands` as `<athena>/.venv/bin/python
scripts/athena_triangulate_worker.py inputs.npz outputs.npz`.

`inputs.npz` carries `batch_count` and, per batch `i`, `undist_i` (V, N, 2) undistorted
normalised coordinates with NaN gaps, `px_i` (V, N, 2) matching pixel coordinates of the
undistorted image, `extrinsics_i` (V, 3, 4) world-to-camera and `intrinsics_i` (V, 3, 3);
plus `reproj_threshold`, `fps` and `smooth_shape` (frames, landmarks, 3).  Every batch is
triangulated with `_triangulate_with_filtering`; the concatenated points, reshaped to
`smooth_shape`, are smoothed with `_smooth3d`.  Only numpy crosses the process boundary.
"""

from __future__ import annotations

import sys

import numpy as np
from athena.triangulaterefine import _smooth3d, _triangulate_with_filtering


def main(argv: list[str]) -> int:
    inputs_path, outputs_path = argv[1], argv[2]
    with np.load(inputs_path) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive}
    batch_count = int(payload["batch_count"])
    threshold = float(payload["reproj_threshold"])
    fps = float(payload["fps"])
    outputs: dict[str, np.ndarray] = {}
    pieces = []
    for index in range(batch_count):
        undist = payload[f"undist_{index}"]
        if np.isnan(undist[..., 0]).all():
            points = np.full((undist.shape[1], 3), np.nan)
        else:
            points = _triangulate_with_filtering(
                undist,
                payload[f"px_{index}"],
                payload[f"extrinsics_{index}"],
                list(payload[f"intrinsics_{index}"]),
                reproj_threshold=threshold,
                min_cams=2,
                max_iterations=3,
            )
        outputs[f"points3d_{index}"] = points
        pieces.append(points)
    frames, landmarks, _ = (int(v) for v in payload["smooth_shape"])
    stacked = np.concatenate(pieces).reshape(frames, landmarks, 3)
    outputs["smoothed"] = _smooth3d(stacked, fps, frequency_cutoff=20, polyorder=3)
    np.savez(outputs_path, **outputs)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
