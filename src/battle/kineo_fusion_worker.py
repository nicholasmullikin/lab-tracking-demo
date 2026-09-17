"""Run Kineo's NLF model on a caller-supplied one-person box stream."""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from kineo.pipeline.stages.nlf.model_wrapper import NLFModelWrapper
from kineo.pipeline.stages.nlf.smpl_keypoints_detection import _batch_infer_keypoints


def _load_frames(capture: cv2.VideoCapture, start: int, count: int) -> torch.Tensor:
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames: list[np.ndarray] = []
    for _ in range(count):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"could not decode input frame {start + len(frames)}")
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    array = np.ascontiguousarray(np.stack(frames))
    return torch.from_numpy(array).permute(0, 3, 1, 2).to("cuda")


def run(args: argparse.Namespace) -> None:
    bbox_payload = pickle.loads(args.bboxes.read_bytes())
    template = pickle.loads(args.template_keypoints.read_bytes())
    annotations = sorted(bbox_payload["annotations"], key=lambda item: int(item["frame_idx"]))
    if not annotations:
        raise ValueError("no fused NLF input boxes")
    if len({int(item["frame_idx"]) for item in annotations}) != len(annotations):
        raise ValueError("fused NLF input must have one box per frame")
    model_path = "/home/nick/src/kineo/checkpoints/nlf_l_multi_0.3.2.torchscript"
    model = NLFModelWrapper(model_path).eval().to("cuda")
    capture = cv2.VideoCapture(str(args.video))
    output: list[dict[str, object]] = []
    started = time.perf_counter()
    try:
        start = 0
        while start < len(annotations):
            end = min(start + args.batch_size, len(annotations))
            for index in range(start + 1, end):
                contiguous = int(annotations[index]["frame_idx"]) == (
                    int(annotations[index - 1]["frame_idx"]) + 1
                )
                if not contiguous:
                    end = index
                    break
            batch = annotations[start:end]
            frames = _load_frames(capture, int(batch[0]["frame_idx"]), len(batch))
            boxes = [
                torch.tensor(
                    [[
                        item["xyxy"][0],
                        item["xyxy"][1],
                        item["xyxy"][2] - item["xyxy"][0],
                        item["xyxy"][3] - item["xyxy"][1],
                        item["score"],
                    ]],
                    dtype=torch.float32,
                    device="cuda",
                )
                for item in batch
            ]
            result = _batch_infer_keypoints(
                frames_rgb=frames,
                model=model,
                bboxes_xywhs=boxes,
                model_name="smplx",
                use_half_precision=True,
            )
            for index, item in enumerate(batch):
                xy = torch.cat(
                    [result["joints2d"][index][0], result["vertices2d"][index][0]], dim=0
                ).cpu().tolist()
                scores = torch.cat(
                    [
                        result["joints_confidences"][index][0],
                        result["vertices_confidences"][index][0],
                    ],
                    dim=0,
                ).cpu().tolist()
                output.append(
                    {
                        "view_id": item["view_id"],
                        "frame_idx": item["frame_idx"],
                        "subject_id": item["subject_id"],
                        "xy": xy,
                        "scores": scores,
                        "annotated": [True] * len(scores),
                        "format": template["annotations"][0]["format"],
                    }
                )
            del frames, boxes, result
            torch.cuda.empty_cache()
            start = end
    finally:
        capture.release()
    args.output.write_bytes(
        pickle.dumps({"metadata": template["metadata"], "annotations": output})
    )
    print(f"nlf_elapsed_seconds={time.perf_counter() - started:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--bboxes", type=Path, required=True)
    parser.add_argument("--template-keypoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
