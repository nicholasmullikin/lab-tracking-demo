"""HF Grounding DINO frame-0 detection plus SAM2 video propagation for Battle."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from time import perf_counter
from typing import Any

GROUNDING_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
GROUNDING_MODEL_REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
TEXT_PROMPT = "hand."
DETECTION_THRESHOLD = 0.4
TEXT_THRESHOLD = 0.3
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"


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
    if written == 0:
        raise RuntimeError(f"video produced zero frames: {video_path}")
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
    parser.add_argument("--grounded-sam2-root", type=Path, required=True)
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
        import hashlib

        import cv2
        import numpy as np
        import torch
        from PIL import Image
        from sam2.build_sam import build_sam2_video_predictor
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Grounding DINO + SAM2 video smoke runs")

        os.chdir(args.grounded_sam2_root)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        frame_count = _extract_frames(args.video, frames_dir, args.max_frames)
        frame_names = sorted(frames_dir.glob("*.jpg"))
        if len(frame_names) != frame_count:
            raise RuntimeError("decoded frame count mismatch after extraction")

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            processor = AutoProcessor.from_pretrained(
                GROUNDING_MODEL_ID, revision=GROUNDING_MODEL_REVISION
            )
            grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                GROUNDING_MODEL_ID, revision=GROUNDING_MODEL_REVISION
            ).to("cuda")
            video_predictor = build_sam2_video_predictor(
                SAM2_CONFIG, str(args.sam2_checkpoint.resolve())
            )
            inference_state = video_predictor.init_state(video_path=str(frames_dir))

            init_image = Image.open(frame_names[0])
            inputs = processor(images=init_image, text=TEXT_PROMPT, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = grounding_model(**inputs)
            detection = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=DETECTION_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
                target_sizes=[init_image.size[::-1]],
            )[0]
            if len(detection["boxes"]) == 0:
                raise RuntimeError(
                    f"Grounding DINO found no objects for prompt {TEXT_PROMPT!r} on frame 0"
                )
            best_index = int(detection["scores"].argmax().item())
            init_box = detection["boxes"][best_index].cpu().numpy()
            init_score = float(detection["scores"][best_index].item())
            init_label = str(detection["labels"][best_index])
            frame0_detection = {
                "class_name": init_label,
                "bbox_xyxy": init_box.tolist(),
                "score": init_score,
                "text_prompt": TEXT_PROMPT,
                "grounding_model_id": GROUNDING_MODEL_ID,
                "grounding_model_revision": GROUNDING_MODEL_REVISION,
            }
            (native_root / "frame0_detection.json").write_text(
                json.dumps(frame0_detection, indent=2) + "\n"
            )

            _, _, _ = video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                box=init_box,
            )
            if first_output_seconds is None:
                first_output_seconds = perf_counter() - started

            video_segments: dict[int, dict[int, np.ndarray]] = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(
                inference_state
            ):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

        peak_vram_bytes = int(torch.cuda.max_memory_allocated())

        with observations_path.open("w", encoding="utf-8") as observations_file:
            for frame_idx in range(frame_count):
                frame_path = frame_names[frame_idx]
                with Image.open(frame_path) as image:
                    width, height = image.size
                segments = video_segments.get(frame_idx, {})
                objects: list[dict[str, Any]] = []
                for obj_id, mask in segments.items():
                    if mask.ndim == 3:
                        mask = mask.squeeze(0)
                    mask_bool = np.asarray(mask, dtype=bool)
                    mask_png = mask_bool.astype(np.uint8) * 255
                    mask_name = f"{frame_idx:05d}.png"
                    cv2.imwrite(str(masks_dir / mask_name), mask_png)
                    unique_mask_hashes.add(hashlib.sha256(mask_png.tobytes()).hexdigest())
                    ys, xs = np.where(mask_bool)
                    if len(xs):
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
                                "object_id": f"grounding-sam2-{obj_id}",
                                "label": init_label,
                                "confidence": init_score,
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
                "grounding_model_id": GROUNDING_MODEL_ID,
                "grounding_model_revision": GROUNDING_MODEL_REVISION,
                "text_prompt": TEXT_PROMPT,
                "detection_threshold": DETECTION_THRESHOLD,
                "text_threshold": TEXT_THRESHOLD,
                "sam2_config": SAM2_CONFIG,
                "sam2_checkpoint": str(args.sam2_checkpoint),
                "analysis_fps": args.analysis_fps,
                "max_frames": args.max_frames,
            },
        },
    )


if __name__ == "__main__":
    main()
