"""Shared-image-encoder, independent-memory SAM3 propagation for a paired video study.

Executed as a file by the model interpreter; does not import the Battle package.
Each comparison state has its own one-object prompt/frame banks. Variants with identical
verified correction schedules reuse the same result, rather than duplicating inference.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import deque
from pathlib import Path
from time import perf_counter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--view", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    view = config["views"][args.view]
    sys.path.insert(0, config["muggled_checkout"])
    spec = importlib.util.spec_from_file_location(
        "worker", Path(__file__).with_name("muggled_worker.py")
    )
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    import cv2
    import numpy as np
    import torch
    from muggled_sam.make_sam import make_sam_from_state_dict

    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "complete.json").unlink(missing_ok=True)
    limit = min(args.max_frames or view["frame_count"], view["frame_count"])
    capture = cv2.VideoCapture(view["video"])
    if not capture.isOpened():
        raise RuntimeError("Could not open source video")
    core = make_sam_from_state_dict(Path(config["checkpoint"]))
    core.to(device="cuda", dtype=torch.bfloat16)
    tracker = core.get_tracking_context()
    state_path = args.out / "state.pt"
    states = {}
    start = 0
    masks = {}
    for group, record in view["groups"].items():
        masks[group] = {}
        for frame, entry in record["corrections"].items():
            masks[group][int(frame)] = worker._read_verified_mask(
                entry["path"], entry["sha256"], tuple(reversed(view["size_wh"]))
            )
    if args.resume and state_path.is_file():
        saved = torch.load(state_path, map_location="cuda", weights_only=False)
        if saved["view_config"] != view or saved["max_side_length"] != config["max_side_length"]:
            raise ValueError("Resume state does not match this input/configuration")
        states, start = saved["states"], saved["next_frame"]
        for _ in range(start):
            if not capture.grab():
                raise RuntimeError("Video ended before resume frame")
        # Remove records written after the last saved native state.
        records = args.out / "frames.jsonl"
        kept = [
            line
            for line in records.read_text().splitlines()
            if json.loads(line)["raw_frame"] < start
        ]
        records.write_text("\n".join(kept) + ("\n" if kept else ""))
    elapsed_start = perf_counter()

    def checkpoint(next_frame: int) -> None:
        tmp = args.out / "state.tmp.pt"
        torch.save(
            {
                "next_frame": next_frame,
                "states": states,
                "view_config": view,
                "max_side_length": config["max_side_length"],
            },
            tmp,
        )
        os.replace(tmp, state_path)

    with torch.inference_mode(), (args.out / "frames.jsonl").open("a" if start else "w") as output:
        for frame_index in range(start, limit):
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"Video ended early at {frame_index}, expected {limit}")
            encoded = tracker.encode_image(image, config["max_side_length"], True)
            rows = {}
            for group in view["groups"]:
                corrected = frame_index in masks[group]
                if frame_index == 0:
                    states[group] = {"prompt": deque(maxlen=1), "frame": deque(maxlen=4)}
                    logits = None
                    score, iou, active = 1.0, None, True
                else:
                    bank = states[group]
                    logits, ious, pointers, scores = tracker.step_video_masking_multiplex(
                        encoded,
                        bank["prompt"],
                        bank["frame"],
                        is_recent_first=False,
                        num_multiplex_objects=1,
                    )
                    score, iou = float(scores.reshape(-1)[0]), float(ious.reshape(-1)[0])
                    active = score > 0
                if corrected:
                    binary = masks[group][frame_index]
                    bank = states[group]
                    bank["prompt"].clear()
                    bank["frame"].clear()
                    bank["prompt"].append(
                        tracker.encode_prompt_memory_from_mask(
                            encoded, worker._prompt_mask_batch(binary[None])
                        )
                    )
                    active = True
                elif active:
                    bank["frame"].append(
                        tracker.encode_frame_memory(encoded, logits, pointers, scores)
                    )
                    binary = worker._binary_mask(logits[0:1], image.shape[:2])
                else:
                    binary = np.zeros(image.shape[:2], dtype=bool)
                folder = args.out / "masks" / group
                folder.mkdir(parents=True, exist_ok=True)
                mask_path = folder / f"{frame_index:06d}.png"
                if not cv2.imwrite(
                    str(mask_path), binary.astype(np.uint8) * 255, [cv2.IMWRITE_PNG_COMPRESSION, 1]
                ):
                    raise RuntimeError(f"Could not save {mask_path}")
                rows[group] = {
                    "active": bool(active),
                    "object_score": score,
                    "estimated_iou": iou,
                    "corrected": corrected,
                    "mask": str(mask_path.resolve()),
                    "pixels": int(binary.sum()),
                }
            output.write(json.dumps({"raw_frame": frame_index, "groups": rows}) + "\n")
            if (frame_index + 1) % 30 == 0 or frame_index + 1 == limit:
                output.flush()
                elapsed = perf_counter() - elapsed_start
                status = {
                    "view": args.view,
                    "next_frame": frame_index + 1,
                    "expected": view["frame_count"],
                    "run_until_frame": limit,
                    "elapsed_seconds": elapsed,
                    "inference_fps": (frame_index + 1 - start) / elapsed,
                    "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                }
                (args.out / "progress.json").write_text(json.dumps(status, indent=2) + "\n")
                print(json.dumps(status), flush=True)
            if (frame_index + 1) % 300 == 0 or frame_index + 1 == limit:
                output.flush()
                checkpoint(frame_index + 1)
    capture.release()
    marker = "complete.json" if limit == view["frame_count"] else "partial-complete.json"
    (args.out / marker).write_text(json.dumps({"frames": limit, "view": args.view}) + "\n")


if __name__ == "__main__":
    main()
