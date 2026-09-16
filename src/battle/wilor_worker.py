"""Frame-wise WiLoR inference worker for the separately managed wilor interpreter.

This module imports only the standard library until model code is needed. The parent
Battle process validates its JSON output against Pydantic contracts.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from time import perf_counter
from typing import Any

BATCH_SIZE = 1
DEFAULT_DETECTOR_CONFIDENCE = 0.3
DEFAULT_RESCALE_FACTOR = 2.0


def _project_full_img(points: Any, cam_trans: Any, focal_length: float, img_res: Any) -> Any:
    import numpy as np

    camera_center = [float(img_res[0]) / 2.0, float(img_res[1]) / 2.0]
    intrinsic = np.eye(3, dtype=np.float64)
    intrinsic[0, 0] = focal_length
    intrinsic[1, 1] = focal_length
    intrinsic[0, 2] = camera_center[0]
    intrinsic[1, 2] = camera_center[1]
    transformed = np.asarray(points, dtype=np.float64) + np.asarray(cam_trans, dtype=np.float64)
    transformed = transformed / transformed[..., -1:]
    projected = (intrinsic @ transformed.T).T
    return projected[..., :-1]


def _normalize_landmarks(
    keypoints_xy: Any, frame_width: int, frame_height: int
) -> list[dict[str, float]]:
    landmarks: list[dict[str, float]] = []
    for x_value, y_value in keypoints_xy:
        x_norm = min(max(float(x_value) / frame_width, 0.0), 1.0)
        y_norm = min(max(float(y_value) / frame_height, 0.0), 1.0)
        landmarks.append({"x": x_norm, "y": y_norm})
    return landmarks


def _landmark_box(landmarks: list[dict[str, float]]) -> dict[str, float]:
    xs = [point["x"] for point in landmarks]
    ys = [point["y"] for point in landmarks]
    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)
    width = min(max(x_max - x_min, 1e-4), 1.0 - x_min)
    height = min(max(y_max - y_min, 1e-4), 1.0 - y_min)
    return {"x": x_min, "y": y_min, "width": width, "height": height}


def _side(is_right: float) -> str:
    return "right" if float(is_right) >= 0.5 else "left"


def _write_result(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--view-id", required=True)
    parser.add_argument("--source-offset-seconds", type=float, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-config", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, required=True)
    parser.add_argument("--analysis-fps", type=float, default=30.0)
    parser.add_argument("--detector-confidence", type=float, default=DEFAULT_DETECTOR_CONFIDENCE)
    parser.add_argument("--rescale-factor", type=float, default=DEFAULT_RESCALE_FACTOR)
    parser.add_argument("--save-native-evidence", action="store_true")
    args = parser.parse_args()

    run_directory = args.run_directory
    run_directory.mkdir(parents=True, exist_ok=True)
    observations_path = run_directory / "observations.jsonl"
    native_root = run_directory / "native_evidence"
    if args.save_native_evidence:
        native_root.mkdir(parents=True, exist_ok=True)

    started = perf_counter()
    first_output_seconds: float | None = None
    peak_vram_bytes = 0
    frames_processed = 0
    state = "succeeded"
    reason = ""

    try:
        import cv2
        import numpy as np
        import torch
        from ultralytics import YOLO
        from wilor.datasets.vitdet_dataset import ViTDetDataset
        from wilor.models import load_wilor
        from wilor.utils import recursive_to
        from wilor.utils.renderer import cam_crop_to_full

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for WiLoR smoke runs on this adapter")

        device = torch.device("cuda:0")
        model, model_cfg = load_wilor(
            checkpoint_path=str(args.checkpoint),
            cfg_path=str(args.checkpoint_config),
        )
        detector = YOLO(str(args.detector))
        model = model.to(device).eval()
        detector = detector.to(device)

        capture = cv2.VideoCapture(str(args.video))
        if not capture.isOpened():
            raise RuntimeError(f"could not open video: {args.video}")

        with observations_path.open("w", encoding="utf-8") as observations_file:
            while frames_processed < args.max_frames:
                ok, frame = capture.read()
                if not ok:
                    break

                detections = detector(frame, conf=args.detector_confidence, verbose=False)[0]
                boxes: list[list[float]] = []
                rights: list[float] = []
                detector_confidences: list[float] = []
                for detection in detections:
                    box_tensor = detection.boxes.data.cpu().detach().squeeze()
                    if box_tensor.ndim == 0:
                        continue
                    box_values = box_tensor.numpy()
                    boxes.append(box_values[:4].tolist())
                    rights.append(float(detection.boxes.cls.cpu().detach().squeeze().item()))
                    detector_confidences.append(
                        float(detection.boxes.conf.cpu().detach().squeeze().item())
                    )

                hands: list[dict[str, Any]] = []
                native_hands: list[dict[str, Any]] = []
                frame_height, frame_width = frame.shape[:2]

                if boxes:
                    dataset = ViTDetDataset(
                        model_cfg,
                        frame,
                        np.stack(boxes),
                        np.stack(rights),
                        rescale_factor=args.rescale_factor,
                    )
                    loader = torch.utils.data.DataLoader(
                        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
                    )
                    for hand_index, batch in enumerate(loader):
                        batch = recursive_to(batch, device)
                        with torch.no_grad():
                            output = model(batch)

                        multiplier = 2 * batch["right"] - 1
                        pred_cam = output["pred_cam"]
                        pred_cam[:, 1] = multiplier * pred_cam[:, 1]
                        box_center = batch["box_center"].float()
                        box_size = batch["box_size"].float()
                        img_size = batch["img_size"].float()
                        scaled_focal_length = (
                            model_cfg.EXTRA.FOCAL_LENGTH
                            / model_cfg.MODEL.IMAGE_SIZE
                            * img_size.max()
                        )
                        pred_cam_t_full = (
                            cam_crop_to_full(
                                pred_cam, box_center, box_size, img_size, scaled_focal_length
                            )
                            .detach()
                            .cpu()
                            .numpy()
                        )

                        for slot in range(batch["img"].shape[0]):
                            joints = output["pred_keypoints_3d"][slot].detach().cpu().numpy()
                            is_right = float(batch["right"][slot].cpu().numpy())
                            joints[:, 0] = (2 * is_right - 1) * joints[:, 0]
                            cam_t = pred_cam_t_full[slot]
                            keypoints_2d = _project_full_img(
                                joints,
                                cam_t,
                                float(scaled_focal_length),
                                img_size[slot].cpu().numpy(),
                            )
                            if hasattr(keypoints_2d, "detach"):
                                keypoints_2d = keypoints_2d.detach().cpu().numpy()
                            landmarks = _normalize_landmarks(
                                keypoints_2d, frame_width, frame_height
                            )
                            side = _side(is_right)
                            confidence = min(
                                max(detector_confidences[hand_index + slot], 0.0), 1.0
                            )
                            hand_id = f"hand-detection-{len(hands) + 1}"
                            joints_payload = [
                                {"x": float(point[0]), "y": float(point[1]), "z": float(point[2])}
                                for point in joints
                            ]
                            hands.append(
                                {
                                    "hand_id": hand_id,
                                    "side": side,
                                    "confidence": confidence,
                                    "landmarks": landmarks,
                                    "box": _landmark_box(landmarks),
                                    "model_side": side,
                                    "model_handedness_confidence": confidence,
                                    "joints_3d_camera_relative": joints_payload,
                                }
                            )
                            native_hands.append(
                                {
                                    "hand_id": hand_id,
                                    "detector_box_xyxy": boxes[hand_index + slot],
                                    "detector_confidence": confidence,
                                    "is_right": is_right,
                                    "pred_cam_t_full": cam_t.tolist(),
                                    "scaled_focal_length": float(scaled_focal_length),
                                    "pred_keypoints_3d": joints.tolist(),
                                    "pred_keypoints_2d_projected": keypoints_2d.tolist(),
                                }
                            )

                observation = {
                    "view_id": args.view_id,
                    "analysis_frame_index": frames_processed,
                    "source_seconds": args.source_offset_seconds
                    + frames_processed / args.analysis_fps,
                    "hands": hands,
                }
                observations_file.write(json.dumps(observation) + "\n")

                if args.save_native_evidence and native_hands:
                    native_path = native_root / f"frame_{frames_processed:06d}.json"
                    native_path.write_text(
                        json.dumps(
                            {
                                "analysis_frame_index": frames_processed,
                                "frame_size": [frame_width, frame_height],
                                "hands": native_hands,
                            },
                            indent=2,
                        )
                        + "\n"
                    )

                frames_processed += 1
                if first_output_seconds is None and hands:
                    first_output_seconds = perf_counter() - started
                if torch.cuda.is_available():
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
            "elapsed_seconds": elapsed_seconds,
            "time_to_first_usable_output_seconds": first_output_seconds,
            "gpu_peak_vram_bytes": peak_vram_bytes or None,
            "runtime_settings": {
                "device": "cuda:0",
                "batch_size": BATCH_SIZE,
                "detector_confidence": args.detector_confidence,
                "rescale_factor": args.rescale_factor,
                "analysis_fps": args.analysis_fps,
                "joint_frame_semantics": (
                    "camera_relative_non_metric; WiLoR pred_keypoints_3d after handedness "
                    "reflection and pred_cam_t_full projection"
                ),
                "mesh_export": False,
                "public_hand_id_policy": "frame-local detector order; no persistent identity claim",
            },
        },
    )


if __name__ == "__main__":
    main()
