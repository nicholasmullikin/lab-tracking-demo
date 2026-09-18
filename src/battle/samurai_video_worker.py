"""SAMURAI SAM2 video propagation worker for Battle."""

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

SAM2_CONFIG = "configs/samurai/sam2.1_hiera_t.yaml"
INIT_BBOX_XYWH = (881, 446, 152, 129)


def _bbox_xyxy(xywh: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x, y, width, height = xywh
    return (x, y, x + width, y + height)


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


def _extract_frames(video_path: Path, frames_dir: Path, max_frames: int) -> int:
    import cv2

    frames_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    written = 0
    while written < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        cv2.imwrite(str(frames_dir / f"{written:05d}.jpg"), frame)
        written += 1
    capture.release()
    if written != max_frames:
        raise RuntimeError(
            f"video produced {written} frames; expected exactly {max_frames} for bounded smoke"
        )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--view-id", required=True)
    parser.add_argument("--source-offset-seconds", type=float, required=True)
    parser.add_argument("--max-frames", type=int, required=True)
    parser.add_argument("--analysis-fps", type=float, default=30.0)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--samurai-root", type=Path, required=True)
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
    state = "succeeded"
    reason = ""

    try:
        import cv2
        import numpy as np
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for SAMURAI video smoke runs")

        samurai_root = args.samurai_root.resolve()
        sys.path.insert(0, str(samurai_root / "sam2"))
        os.chdir(samurai_root)
        from sam2.build_sam import build_sam2_video_predictor

        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        frame_count = _extract_frames(args.video, frames_dir, args.max_frames)
        init_box = _bbox_xyxy(INIT_BBOX_XYWH)
        init_seed = {
            "bbox_xywh": list(INIT_BBOX_XYWH),
            "bbox_xyxy": list(init_box),
            "seed_provenance": "approved focused static proxy frame-0 hand box",
        }
        (native_root / "frame0_seed.json").write_text(json.dumps(init_seed, indent=2) + "\n")

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            predictor = build_sam2_video_predictor(
                SAM2_CONFIG, str(args.sam2_checkpoint.resolve()), device="cuda:0"
            )
            inference_state = predictor.init_state(
                str(frames_dir.resolve()),
                offload_video_to_cpu=True,
            )
            _, _, _ = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=0,
                box=np.array(init_box, dtype=np.float32),
            )
            if first_output_seconds is None:
                first_output_seconds = perf_counter() - started

            video_segments: dict[int, dict[int, np.ndarray]] = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                inference_state
            ):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[index] > 0.0).cpu().numpy()
                    for index, out_obj_id in enumerate(out_obj_ids)
                }

        peak_vram_bytes = int(torch.cuda.max_memory_allocated())

        with observations_path.open("w", encoding="utf-8") as observations_file:
            for frame_idx in range(frame_count):
                frame_path = frames_dir / f"{frame_idx:05d}.jpg"
                image = cv2.imread(str(frame_path))
                if image is None:
                    raise RuntimeError(f"could not read decoded frame {frame_idx}")
                height, width = image.shape[:2]
                segments = video_segments.get(frame_idx, {})
                objects: list[dict[str, Any]] = []
                for obj_id, mask in segments.items():
                    if mask.ndim == 3:
                        mask = mask.squeeze(0)
                    mask_bool = np.asarray(mask, dtype=bool)
                    if not mask_bool.any():
                        continue
                    mask_png = mask_bool.astype(np.uint8) * 255
                    mask_name = f"{frame_idx:05d}.png"
                    cv2.imwrite(str(masks_dir / mask_name), mask_png)
                    unique_mask_hashes.add(hashlib.sha256(mask_png.tobytes()).hexdigest())
                    ys, xs = np.where(mask_bool)
                    box = _normalize_box(
                        float(xs.min()),
                        float(ys.min()),
                        float(xs.max() + 1),
                        float(ys.max() + 1),
                        width,
                        height,
                    )
                    objects.append(
                        {
                            "object_id": f"samurai-{obj_id}",
                            "label": "seeded_hand",
                            "confidence": 1.0,
                            "box": box,
                            "mask": {
                                "uri": f"native/masks/{mask_name}",
                                "storage": "native_artifact",
                                "format": "png",
                            },
                        }
                    )
                    frames_with_masks += 1
                payload = {
                    "view_id": args.view_id,
                    "analysis_frame_index": frame_idx,
                    "source_seconds": args.source_offset_seconds + frame_idx / args.analysis_fps,
                    "objects": objects,
                }
                observations_file.write(json.dumps(payload) + "\n")
                frames_processed += 1

        if frames_with_masks != frame_count:
            raise RuntimeError(
                f"expected masks on all {frame_count} frames; got {frames_with_masks}"
            )
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
            "elapsed_seconds": elapsed_seconds,
            "time_to_first_usable_output_seconds": first_output_seconds,
            "gpu_peak_vram_bytes": peak_vram_bytes or None,
            "runtime_settings": {
                "sam2_config": SAM2_CONFIG,
                "samurai_mode": True,
                "sam2_checkpoint": str(args.sam2_checkpoint),
                "init_bbox_xywh": list(INIT_BBOX_XYWH),
                "analysis_fps": args.analysis_fps,
                "max_frames": args.max_frames,
            },
        },
    )


if __name__ == "__main__":
    main()
