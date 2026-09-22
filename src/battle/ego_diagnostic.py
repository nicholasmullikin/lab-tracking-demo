"""Reproducible image characterization and factual smoke-run comparison for ego SAM3 tests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from . import fs_common
except ImportError:
    # Run as a script by the MuggledSAM interpreter (`battle-muggled-smoke` launches it by
    # path): the stdlib-only helper module sits beside this file.
    _HERE = str(Path(__file__).resolve().parent)
    if _HERE not in sys.path:
        sys.path.append(_HERE)
    import fs_common  # type: ignore[no-redef]

FPS = 30
FRAME_INDICES = (0, 150, 299)
PANEL_WIDTH = 360
ROW_LABEL_WIDTH = 210
HEADER_HEIGHT = 58
COLORS_BGR = ((80, 200, 80), (60, 190, 255), (220, 120, 255))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fs_common.write_json(path, payload, sort_keys=True)


def _histogram_percentile(histogram: Any, percentile: float) -> int:
    """Return the integer nearest-rank percentile from a uint8 histogram."""
    import numpy as np

    rank = max(1, int(np.ceil(percentile / 100 * int(histogram.sum()))))
    return int(np.searchsorted(np.cumsum(histogram), rank))


def characterize_decoded_video(video_path: Path, max_frames: int = 300) -> dict[str, Any]:
    """Measure decoded BGR channels and luminance without loading every frame at once."""
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_square_sum = np.zeros(3, dtype=np.float64)
    channel_cross_sum = np.zeros((3, 3), dtype=np.float64)
    equal_count = {"b_g": 0, "b_r": 0, "g_r": 0}
    absolute_difference_sum = {"b_g": 0.0, "b_r": 0.0, "g_r": 0.0}
    histogram = np.zeros(256, dtype=np.int64)
    frame_p1: list[float] = []
    frame_p99: list[float] = []
    frame_count = 0
    shape: list[int] | None = None
    try:
        while frame_count < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            shape = list(frame.shape)
            flat = frame.reshape(-1, 3).astype(np.float64)
            channel_sum += flat.sum(axis=0)
            channel_square_sum += np.square(flat).sum(axis=0)
            channel_cross_sum += flat.T @ flat
            equal_count["b_g"] += int(np.count_nonzero(frame[..., 0] == frame[..., 1]))
            equal_count["b_r"] += int(np.count_nonzero(frame[..., 0] == frame[..., 2]))
            equal_count["g_r"] += int(np.count_nonzero(frame[..., 1] == frame[..., 2]))
            absolute_difference_sum["b_g"] += float(np.abs(flat[:, 0] - flat[:, 1]).sum())
            absolute_difference_sum["b_r"] += float(np.abs(flat[:, 0] - flat[:, 2]).sum())
            absolute_difference_sum["g_r"] += float(np.abs(flat[:, 1] - flat[:, 2]).sum())
            luminance = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            histogram += np.bincount(luminance.ravel(), minlength=256)
            frame_p1.append(float(np.percentile(luminance, 1)))
            frame_p99.append(float(np.percentile(luminance, 99)))
            frame_count += 1
    finally:
        capture.release()
    if frame_count == 0 or shape is None:
        raise RuntimeError(f"video contains no decodable frames: {video_path}")
    pixel_count = int(histogram.sum())
    channel_mean = channel_sum / pixel_count
    channel_variance = channel_square_sum / pixel_count - np.square(channel_mean)
    channel_stddev = np.sqrt(np.maximum(channel_variance, 0))

    def correlation(first: int, second: int) -> float:
        covariance = channel_cross_sum[first, second] / pixel_count - (
            channel_mean[first] * channel_mean[second]
        )
        return float(
            np.clip(covariance / (channel_stddev[first] * channel_stddev[second]), -1.0, 1.0)
        )

    percentiles = {
        f"p{percentile:g}": _histogram_percentile(histogram, percentile)
        for percentile in (0, 1, 5, 50, 95, 99, 100)
    }
    luminance_values = np.arange(256, dtype=np.float64)
    luminance_sum = float((histogram * luminance_values).sum())
    luminance_square_sum = float((histogram * np.square(luminance_values)).sum())
    return {
        "schema_version": "1.0",
        "measurement": "decoded_bgr_first_300_analysis_frames",
        "video": str(video_path),
        "frames_decoded": frame_count,
        "frame_shape_hwc": shape,
        "pixel_count": pixel_count,
        "channel_equality_fraction": {
            key: value / pixel_count for key, value in equal_count.items()
        },
        "channel_mean_absolute_difference": {
            key: value / pixel_count for key, value in absolute_difference_sum.items()
        },
        "channel_pearson_correlation": {
            "b_g": correlation(0, 1),
            "b_r": correlation(0, 2),
            "g_r": correlation(1, 2),
        },
        "luminance": {
            "definition": "OpenCV BGR-to-gray uint8",
            "percentile_method": "nearest-rank from aggregate uint8 histogram",
            "percentiles": percentiles,
            "mean": luminance_sum / pixel_count,
            "stddev": float(
                np.sqrt(luminance_square_sum / pixel_count - luminance_sum**2 / pixel_count**2)
            ),
            "value_0_fraction": float(histogram[0] / pixel_count),
            "value_255_fraction": float(histogram[255] / pixel_count),
            "per_frame_p1_range": [min(frame_p1), max(frame_p1)],
            "per_frame_p99_range": [min(frame_p99), max(frame_p99)],
        },
    }


def _load_observations(run_directory: Path) -> list[dict[str, Any]]:
    with (run_directory / "observations.jsonl").open() as file:
        return [json.loads(line) for line in file if line.strip()]


def _missing_runs(observations: list[dict[str, Any]]) -> list[dict[str, int]]:
    runs: list[dict[str, int]] = []
    first: int | None = None
    for observation in observations:
        frame_index = int(observation["analysis_frame_index"])
        if observation["objects"]:
            if first is not None:
                runs.append({"start_frame": first, "end_frame_exclusive": frame_index})
                first = None
        elif first is None:
            first = frame_index
    if first is not None:
        runs.append(
            {
                "start_frame": first,
                "end_frame_exclusive": int(observations[-1]["analysis_frame_index"]) + 1,
            }
        )
    return runs


def summarize_run(run_directory: Path, name: str) -> dict[str, Any]:
    """Report emitted-output coverage, not unlabelled target accuracy."""
    observations = _load_observations(run_directory)
    worker = json.loads((run_directory / "worker_result.json").read_text())
    initial_objects = observations[0]["objects"] if observations else []
    emitted_frames = sum(bool(observation["objects"]) for observation in observations)
    missing_runs = _missing_runs(observations) if observations else []
    all_ids = {
        object_["object_id"] for observation in observations for object_ in observation["objects"]
    }
    initial_ids = {object_["object_id"] for object_ in initial_objects}
    prompt_mode = worker["runtime_settings"].get("prompt_mode", "text_detection")
    return {
        "name": name,
        "run_directory": str(run_directory),
        "state": worker["state"],
        "frames_processed": worker["frames_processed"],
        "initial_output_count": len(initial_objects),
        "initial_detector_detection_count": (
            len(initial_objects) if prompt_mode == "text_detection" else None
        ),
        "initialization_note": (
            "manual prompt; detector count is not applicable"
            if prompt_mode == "manual_box"
            else "text detector output after configured threshold"
        ),
        "emitted_frame_count": emitted_frames,
        "emission_coverage_fraction": emitted_frames / len(observations) if observations else 0.0,
        "emission_gap_count": len(missing_runs),
        "longest_emission_gap_frames": max(
            (gap["end_frame_exclusive"] - gap["start_frame"] for gap in missing_runs),
            default=0,
        ),
        "emission_gaps": missing_runs,
        "unique_track_ids": sorted(all_ids),
        "id_restarts_after_initialization": len(all_ids - initial_ids),
        "elapsed_seconds": worker["elapsed_seconds"],
        "time_to_first_usable_output_seconds": worker["time_to_first_usable_output_seconds"],
        "gpu_peak_vram_bytes": worker["gpu_peak_vram_bytes"],
        "prompt_mode": prompt_mode,
        "preprocessing": worker["runtime_settings"].get("preprocessing", "original_bgr"),
    }


def _transform_for_display(frame: Any, runtime: dict[str, Any]) -> Any:
    import cv2
    import numpy as np

    preprocessing = runtime.get("preprocessing", {})
    if preprocessing.get("mode", "original_bgr") == "original_bgr":
        return frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    low, high = np.percentile(
        gray, (preprocessing["lower_percentile"], preprocessing["upper_percentile"])
    )
    normalized = (
        np.zeros_like(gray)
        if high <= low
        else np.clip((gray.astype(np.float32) - low) * 255 / (high - low), 0, 255).astype(np.uint8)
    )
    clahe = cv2.createCLAHE(
        clipLimit=preprocessing["clahe_clip_limit"],
        tileGridSize=(preprocessing["clahe_tile_grid_size"],) * 2,
    )
    return cv2.cvtColor(clahe.apply(normalized), cv2.COLOR_GRAY2BGR)


def _render_panel(
    frame: Any, observation: dict[str, Any], run_directory: Path, frame_index: int
) -> Any:
    import cv2

    height, width = frame.shape[:2]
    for index, object_ in enumerate(observation["objects"]):
        color = COLORS_BGR[index % len(COLORS_BGR)]
        mask = object_.get("mask")
        if mask and (mask_path := run_directory / mask["uri"]).is_file():
            decoded_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if decoded_mask is not None:
                overlay = frame.copy()
                overlay[decoded_mask > 0] = color
                frame = cv2.addWeighted(frame, 0.60, overlay, 0.40, 0)
        box = object_["box"]
        x1, y1 = round(box["x"] * width), round(box["y"] * height)
        x2 = round((box["x"] + box["width"]) * width)
        y2 = round((box["y"] + box["height"]) * height)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            object_["label"],
            (x1, max(17, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )
    panel_height = round(height * PANEL_WIDTH / width)
    panel = cv2.resize(frame, (PANEL_WIDTH, panel_height))
    cv2.rectangle(panel, (0, 0), (PANEL_WIDTH, 25), (20, 20, 20), -1)
    cv2.putText(
        panel,
        f"t={frame_index / FPS:.3f}s | frame {frame_index}",
        (7, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return panel


def render_comparison(
    repository_root: Path,
    runs: list[tuple[str, Path]],
    output_path: Path,
) -> None:
    """Render exactly three conditions at the fixed review timestamps."""
    import cv2
    import numpy as np

    rendered_rows = []
    for name, run_directory in runs:
        runtime = json.loads((run_directory / "runtime_settings.json").read_text())
        observations = {
            int(observation["analysis_frame_index"]): observation
            for observation in _load_observations(run_directory)
        }
        proxy_path = repository_root / runtime["proxy"]
        capture = cv2.VideoCapture(str(proxy_path))
        if not capture.isOpened():
            raise RuntimeError(f"could not open proxy: {proxy_path}")
        try:
            panels = []
            for frame_index in FRAME_INDICES:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"could not decode proxy frame {frame_index}")
                panels.append(
                    _render_panel(
                        _transform_for_display(frame, runtime),
                        observations[frame_index],
                        run_directory,
                        frame_index,
                    )
                )
        finally:
            capture.release()
        label = np.full((panels[0].shape[0], ROW_LABEL_WIDTH, 3), 235, dtype=np.uint8)
        cv2.putText(label, name, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2)
        cv2.putText(
            label,
            f"{runtime.get('prompt_mode', 'text')} | {runtime.get('concepts', [])}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (20, 20, 20),
            1,
        )
        rendered_rows.append(np.hstack((label, *panels)))
    header = np.full((HEADER_HEIGHT, rendered_rows[0].shape[1], 3), 35, dtype=np.uint8)
    cv2.putText(
        header,
        "EGO SAM3 10-SECOND PROXY — ZERO-SHOT BASELINE VS EXPLICIT CONDITIONS",
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        header,
        "Displays the representation supplied to each condition; overlays are recorded outputs.",
        (12, 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (225, 225, 225),
        1,
        cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), np.vstack((header, *rendered_rows))):
        raise RuntimeError(f"could not write comparison sheet: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    characterize = subparsers.add_parser("characterize")
    characterize.add_argument("--video", type=Path, required=True)
    characterize.add_argument("--output", type=Path, required=True)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--repository-root", type=Path, required=True)
    compare.add_argument("--baseline-run", type=Path, required=True)
    compare.add_argument(
        "--condition-run",
        nargs=2,
        action="append",
        metavar=("NAME", "RUN"),
        required=True,
    )
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--metrics-output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "characterize":
        _write_json(args.output, characterize_decoded_video(args.video))
        return
    runs = [("ZERO-SHOT BASELINE", args.baseline_run)]
    runs.extend((name, Path(run)) for name, run in args.condition_run)
    render_comparison(args.repository_root, runs, args.output)
    _write_json(
        args.metrics_output,
        {
            "schema_version": "1.0",
            "measurement_scope": "factual 300-frame proxy output measures; not accuracy",
            "runs": [summarize_run(run_directory, name) for name, run_directory in runs],
        },
    )


if __name__ == "__main__":
    main()
