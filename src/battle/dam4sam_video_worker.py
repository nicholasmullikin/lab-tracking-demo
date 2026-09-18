"""DAM4SAM headless bbox-init tracking worker for Battle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from time import perf_counter
from typing import Any

INIT_BBOX_XYWH = (881, 446, 152, 129)
TRACKER_NAME = "sam21pp-T"
SAM2_CONFIG = "sam21pp_hiera_t.yaml"


def _normalize_box(
    x1: float, y1: float, x2: float, y2: float, width: int, height: int
) -> dict[str, float]:
    left = min(max(x1 / width, 0.0), 1.0)
    top = min(max(y1 / height, 0.0), 1.0)
    right = min(max(x2 / width, 0.0), 1.0)
    bottom = min(max(y2 / height, 0.0), 1.0)
    box_width = min(max(right - left, 1e-4), 1.0 - left)
    box_height = min(max(bottom - top, 1e-4), 1.0 - top)
    return {"x": left, "y": top, "width": box_width, "height": box_height}


def _write_result(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _extract_frames(video_path: Path, frames_dir: Path, max_frames: int) -> list[Path]:
    import cv2

    frames_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    frame_paths: list[Path] = []
    while len(frame_paths) < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        frame_path = frames_dir / f"{len(frame_paths):05d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        frame_paths.append(frame_path)
    capture.release()
    if len(frame_paths) != max_frames:
        raise RuntimeError(
            f"video produced {len(frame_paths)} frames; expected exactly {max_frames}"
        )
    return frame_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--view-id", required=True)
    parser.add_argument("--source-offset-seconds", type=float, required=True)
    parser.add_argument("--max-frames", type=int, required=True)
    parser.add_argument("--analysis-fps", type=float, default=30.0)
    parser.add_argument("--dam4sam-root", type=Path, required=True)
    args = parser.parse_args()

    run_directory = args.run_directory
    run_directory.mkdir(parents=True, exist_ok=True)
    native_root = run_directory / "native"
    frames_dir = native_root / "frames"
    masks_dir = native_root / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    observations_path = run_directory / "observations.jsonl"
    result_path = run_directory / "worker_result.json"

    started = perf_counter()
    first_output_seconds: float | None = None
    peak_vram_bytes = 0
    frames_processed = 0
    frames_with_masks = 0
    unique_mask_hashes: set[str] = set()
    drm_additions = 0
    state = "succeeded"
    reason = ""

    try:
        import cv2
        import numpy as np
        import torch
        from PIL import Image

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for DAM4SAM video smoke runs")

        dam4sam_root = args.dam4sam_root.resolve()
        sys.path.insert(0, str(dam4sam_root))
        os.chdir(dam4sam_root)
        from dam4sam_tracker import DAM4SAMTracker

        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        frame_paths = _extract_frames(args.video, frames_dir, args.max_frames)
        init_box = list(INIT_BBOX_XYWH)
        init_seed = {
            "bbox_xywh": init_box,
            "seed_provenance": "approved focused static proxy frame-0 hand box",
            "initialization_mode": "headless_bbox_to_mask",
            "vot_reference": (
                "Official VOT wrapper initializes from mask prompts on frame 0; "
                "this smoke uses bbox->estimate_mask_from_box->add_new_mask instead."
            ),
        }
        (native_root / "frame0_seed.json").write_text(json.dumps(init_seed, indent=2) + "\n")

        tracker = DAM4SAMTracker(tracker_name=TRACKER_NAME)
        last_added_before = -1

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            with observations_path.open("w", encoding="utf-8") as observations_file:
                for frame_idx, frame_path in enumerate(frame_paths):
                    image = Image.open(frame_path)
                    width, height = image.size
                    if frame_idx == 0:
                        outputs = tracker.initialize(image, None, bbox=init_box)
                        last_added_before = tracker.last_added
                    else:
                        outputs = tracker.track(image)
                    if first_output_seconds is None:
                        first_output_seconds = perf_counter() - started
                    if tracker.last_added != last_added_before:
                        drm_additions += 1
                        last_added_before = tracker.last_added

                    pred_mask = np.asarray(outputs["pred_mask"], dtype=bool)
                    if not pred_mask.any():
                        raise RuntimeError(f"empty mask on analysis frame {frame_idx}")
                    mask_png = pred_mask.astype(np.uint8) * 255
                    mask_name = f"{frame_idx:05d}.png"
                    cv2.imwrite(str(masks_dir / mask_name), mask_png)
                    unique_mask_hashes.add(hashlib.sha256(mask_png.tobytes()).hexdigest())
                    ys, xs = np.where(pred_mask)
                    box = _normalize_box(
                        float(xs.min()),
                        float(ys.min()),
                        float(xs.max() + 1),
                        float(ys.max() + 1),
                        width,
                        height,
                    )
                    payload = {
                        "view_id": args.view_id,
                        "analysis_frame_index": frame_idx,
                        "source_seconds": args.source_offset_seconds
                        + frame_idx / args.analysis_fps,
                        "objects": [
                            {
                                "object_id": "dam4sam-seeded-hand",
                                "label": "seeded_hand",
                                "confidence": 1.0,
                                "box": box,
                                "mask": {
                                    "uri": f"native/masks/{mask_name}",
                                    "storage": "native_artifact",
                                    "format": "png",
                                },
                            }
                        ],
                    }
                    observations_file.write(json.dumps(payload) + "\n")
                    frames_processed += 1
                    frames_with_masks += 1

        peak_vram_bytes = int(torch.cuda.max_memory_allocated())
        if len(unique_mask_hashes) < 2:
            raise RuntimeError("mask outputs did not vary across frames")

    except Exception as exc:  # noqa: BLE001 - worker must persist failure state
        state = "failed"
        reason = f"{type(exc).__name__}: {exc}"
        (run_directory / "worker_traceback.log").write_text(traceback.format_exc())

    elapsed_seconds = perf_counter() - started
    _write_result(
        result_path,
        {
            "state": state,
            "reason": reason,
            "frames_processed": frames_processed,
            "frames_with_masks": frames_with_masks,
            "unique_mask_hashes": len(unique_mask_hashes),
            "drm_memory_additions": drm_additions,
            "elapsed_seconds": elapsed_seconds,
            "time_to_first_usable_output_seconds": first_output_seconds,
            "gpu_peak_vram_bytes": peak_vram_bytes or None,
            "runtime_settings": {
                "tracker_name": TRACKER_NAME,
                "sam2_model_config": SAM2_CONFIG,
                "init_bbox_xywh": list(INIT_BBOX_XYWH),
                "analysis_fps": args.analysis_fps,
                "max_frames": args.max_frames,
                "dam4sam_config": "dam4sam_config.yaml",
            },
        },
    )


if __name__ == "__main__":
    main()
