#!/usr/bin/env python3
"""FineBio preflight, detector pass: DINO boxes on a set of raw frames per view.

Runs in the detector venv (CPU torch):

    CUDA_VISIBLE_DEVICES="" /home/nick/src/finebio-detector/.venv/bin/python \
        scripts/finebio_preflight_detect.py --trial P03_01_01 \
        --consecutive 1798:60 --spaced 1798:2398:30 \
        --output runs/preflight-finebio-20260924/detections

Writes one JSONL per view (``<view>.jsonl``) with ``frame_index``, ``image_hw`` and every
detection at score >= 0.05 as ``class``, ``score``, ``box_xyxy_px`` in raw-video pixels
(fixed cameras 1920x1080, fpv 1920x1440). Frames are read from the raw FineBio videos, not a
proxy, so pixel coordinates line up with the shipped calibration after the resolution rescale.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

RAW = Path("/home/nick/src/battle/data/raw/finebio")
DET = Path("/home/nick/src/finebio-detector")
VIEWS = ("fpv", "T1", "T2", "T3", "T4", "T5")


def video_path(trial: str, view: str) -> Path:
    if view == "fpv":
        return RAW / "finebio_videos_fpv_test/finebio_videos" / f"{trial}.mp4"
    return RAW / "finebio_videos_tpv_test/finebio_videos" / f"{trial}_{view}.mp4"


def parse_frames(consecutive: str | None, spaced: str | None) -> list[int]:
    frames: set[int] = set()
    if consecutive:
        start, count = (int(v) for v in consecutive.split(":"))
        frames.update(range(start, start + count))
    if spaced:
        start, stop, step = (int(v) for v in spaced.split(":"))
        frames.update(range(start, stop, step))
    return sorted(frames)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", default="P03_01_01")
    parser.add_argument("--views", default=",".join(VIEWS))
    parser.add_argument("--consecutive", default=None, help="start:count")
    parser.add_argument("--spaced", default=None, help="start:stop:step")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--record-threshold", type=float, default=0.05)
    args = parser.parse_args()

    import cv2
    import torch
    from mmdet.apis import inference_detector, init_detector

    frames = parse_frames(args.consecutive, args.spaced)
    if not frames:
        print("no frames requested", file=sys.stderr)
        return 2
    args.output.mkdir(parents=True, exist_ok=True)
    model = init_detector(
        str(DET / "mmdetection/configs/dino/dino-4scale_r50_8xb2-12e_finebio.py"),
        str(DET / "checkpoints/dino.pth"),
        device="cpu",
    )
    classes = tuple(model.dataset_meta["classes"])
    started = perf_counter()
    for view in args.views.split(","):
        path = video_path(args.trial, view)
        capture = cv2.VideoCapture(str(path))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frames[0])
        wanted = set(frames)
        index = frames[0]
        out_path = args.output / f"{view}.jsonl"
        done = 0
        with out_path.open("w", encoding="utf-8") as out:
            while wanted:
                ok, frame = capture.read()
                if not ok:
                    break
                if index in wanted:
                    wanted.discard(index)
                    with torch.inference_mode():
                        result = inference_detector(model, frame)
                    inst = result.pred_instances
                    scores = inst.scores.cpu().numpy()
                    labels = inst.labels.cpu().numpy()
                    boxes = inst.bboxes.cpu().numpy()
                    keep = scores >= args.record_threshold
                    order = scores[keep].argsort()[::-1]
                    dets = [
                        {
                            "class": classes[int(labels[keep][i])],
                            "score": float(scores[keep][i]),
                            "box_xyxy_px": [float(v) for v in boxes[keep][i]],
                        }
                        for i in order
                    ]
                    out.write(
                        json.dumps(
                            {
                                "view": view,
                                "frame_index": index,
                                "image_hw": [int(frame.shape[0]), int(frame.shape[1])],
                                "detections": dets,
                            }
                        )
                        + "\n"
                    )
                    done += 1
                    if done % 10 == 0:
                        print(
                            f"{view}: {done}/{len(frames)} frames, {perf_counter() - started:.0f}s",
                            flush=True,
                        )
                index += 1
        capture.release()
        if wanted:
            print(f"{view}: video ended before frames {sorted(wanted)[:5]}", file=sys.stderr)
        print(f"{view}: done ({done} frames) -> {out_path}", flush=True)
    (args.output / "frames.json").write_text(
        json.dumps({"trial": args.trial, "frames": frames, "views": args.views.split(",")})
    )
    print(f"PREFLIGHT_DETECT_DONE {perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
