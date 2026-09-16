"""Per-frame YOLO detection plus BoxMOT association for the wilor interpreter env."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from time import perf_counter
from typing import Any

YOLO_CLASS_NAMES = {
    0: "person",
}


def _normalize_box(
    x1: float, y1: float, x2: float, y2: float, width: int, height: int
) -> dict[str, float]:
    x = min(max(x1 / width, 0.0), 1.0)
    y = min(max(y1 / height, 0.0), 1.0)
    x2_norm = min(max(x2 / width, 0.0), 1.0)
    y2_norm = min(max(y2 / height, 0.0), 1.0)
    box_width = min(max(x2_norm - x, 1e-4), 1.0 - x)
    box_height = min(max(y2_norm - y, 1e-4), 1.0 - y)
    return {"x": x, "y": y, "width": box_width, "height": box_height}


def _write_result(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--view-id", required=True)
    parser.add_argument("--source-offset-seconds", type=float, required=True)
    parser.add_argument("--detector-weights", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, required=True)
    parser.add_argument("--analysis-fps", type=float, default=30.0)
    parser.add_argument("--detector-confidence", type=float, default=0.25)
    parser.add_argument("--detector-classes", default="0")
    args = parser.parse_args()

    run_directory = args.run_directory
    run_directory.mkdir(parents=True, exist_ok=True)
    observations_path = run_directory / "observations.jsonl"
    class_ids = [int(part) for part in args.detector_classes.split(",") if part.strip()]

    started = perf_counter()
    first_output_seconds: float | None = None
    peak_vram_bytes = 0
    frames_processed = 0
    frames_with_tracks = 0
    total_tracks = 0
    state = "succeeded"
    reason = ""

    try:
        import cv2
        import numpy as np
        import torch
        from boxmot import BotSort
        from ultralytics import YOLO

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for BoxMOT smoke runs on this adapter")

        device = torch.device("cuda:0")
        detector = YOLO(str(args.detector_weights))
        tracker = BotSort(
            reid_weights=None,
            device=0,
            half=False,
            frame_rate=int(args.analysis_fps),
        )

        capture = cv2.VideoCapture(str(args.video))
        if not capture.isOpened():
            raise RuntimeError(f"could not open video: {args.video}")

        with observations_path.open("w", encoding="utf-8") as observations_file:
            while frames_processed < args.max_frames:
                ok, frame = capture.read()
                if not ok:
                    break

                result = detector(
                    frame,
                    conf=args.detector_confidence,
                    classes=class_ids or None,
                    verbose=False,
                )[0]
                detections: list[list[float]] = []
                if result.boxes is not None and len(result.boxes):
                    for box in result.boxes:
                        xyxy = box.xyxy.cpu().numpy().squeeze()
                        if xyxy.ndim == 0:
                            continue
                        confidence = float(box.conf.cpu().numpy().squeeze())
                        class_id = int(box.cls.cpu().numpy().squeeze())
                        detections.append(
                            [
                                float(xyxy[0]),
                                float(xyxy[1]),
                                float(xyxy[2]),
                                float(xyxy[3]),
                                confidence,
                                float(class_id),
                            ]
                        )
                detection_array = (
                    np.asarray(detections, dtype=np.float32)
                    if detections
                    else np.empty((0, 6), dtype=np.float32)
                )
                tracks = tracker.update(detection_array, frame)
                frame_height, frame_width = frame.shape[:2]
                objects: list[dict[str, Any]] = []
                if tracks is not None and len(tracks):
                    for track in tracks:
                        x1, y1, x2, y2, track_id, confidence, class_id, *_rest = track
                        class_index = int(class_id)
                        label = YOLO_CLASS_NAMES.get(class_index, f"class-{class_index}")
                        objects.append(
                            {
                                "object_id": f"boxmot-{int(track_id)}",
                                "label": label,
                                "confidence": min(max(float(confidence), 0.0), 1.0),
                                "box": _normalize_box(
                                    float(x1),
                                    float(y1),
                                    float(x2),
                                    float(y2),
                                    frame_width,
                                    frame_height,
                                ),
                            }
                        )
                if objects:
                    frames_with_tracks += 1
                    total_tracks += len(objects)
                    if first_output_seconds is None:
                        first_output_seconds = perf_counter() - started

                observation = {
                    "view_id": args.view_id,
                    "analysis_frame_index": frames_processed,
                    "source_seconds": args.source_offset_seconds
                    + frames_processed / args.analysis_fps,
                    "objects": objects,
                }
                observations_file.write(json.dumps(observation) + "\n")
                frames_processed += 1
                peak_vram_bytes = max(
                    peak_vram_bytes, int(torch.cuda.max_memory_allocated(device))
                )

        capture.release()
    except Exception as error:  # noqa: BLE001 - worker must persist failure state
        state = "failed"
        reason = f"{type(error).__name__}: {error}"
        (run_directory / "worker.traceback.log").write_text(traceback.format_exc())

    elapsed_seconds = perf_counter() - started
    _write_result(
        run_directory / "worker_result.json",
        {
            "state": state,
            "reason": reason,
            "frames_processed": frames_processed,
            "frames_with_tracks": frames_with_tracks,
            "total_track_observations": total_tracks,
            "elapsed_seconds": elapsed_seconds,
            "time_to_first_usable_output_seconds": first_output_seconds,
            "gpu_peak_vram_bytes": peak_vram_bytes or None,
            "runtime_settings": {
                "device": "cuda:0",
                "tracker": "boxmot.BotSort",
                "detector": "ultralytics.YOLO per-frame without built-in tracking",
                "detector_weights": str(args.detector_weights),
                "detector_confidence": args.detector_confidence,
                "detector_classes": class_ids,
                "analysis_fps": args.analysis_fps,
                "association_policy": (
                    "independent per-frame detections fed to BoxMOT; no MuggledSAM IDs"
                ),
            },
        },
    )


if __name__ == "__main__":
    main()
