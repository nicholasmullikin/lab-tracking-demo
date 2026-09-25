#!/usr/bin/env python3
"""FineBio preflight, SAM3 checks: box-prompted masks per view, and a short box-seeded track.

MuggledSAM interpreter, GPU:

    CUDA_VISIBLE_DEVICES=0 /home/nick/.pyenv/versions/muggled_sam/bin/python \
        scripts/finebio_preflight_sam3.py --trial P03_01_01 --frame 1798 \
        --detections runs/preflight-finebio-20260924/detections \
        --output runs/preflight-finebio-20260924/sam3 --track-views fpv,T5,T2 --track-frames 300

Check 1 (``decode``): on one raw frame per view, the FineBio DINO box of each target class is
the only prompt to the SAM3 image decoder (encoder max side 1280 with square sizing, as the
Battle runs; 1920 as well for the fixed cameras). Per (view, class, size): decoder IoU
prediction, fill ratio (mask inside the box / box area), spill (mask outside the box / mask),
IoU between the mask's bounding box and the prompt box. Overlays per view and size.

Check 2 (``track``): the same boxes at the start frame seed the SAM3.1 multiplex tracker
(prompt memory 1, frame memory 4, oldest first, no corrections) for ``--track-frames`` frames
at 1280; per frame per object the object score and mask area; where the detector pass has a
box on the same frame, the IoU between the mask's bounding box and the same-class detector
box. Overlays every 75 frames.

Nothing here is compared with ground truth; the detector boxes are the reference and were
produced by a model trained on FineBio.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/home/nick/src/muggled_sam")
import torch  # noqa: E402
from muggled_sam.make_sam import make_sam_from_state_dict  # noqa: E402

RAW = Path("/home/nick/src/battle/data/raw/finebio")
MODEL = "/home/nick/src/muggled_sam/model_weights/sam3.1_multiplex.pt"
DECODE_CLASSES = (
    "cell_culture_plate",
    "blue_pipette",
    "centrifuge",
    "50ml_tube",
    "micro_tube_rack",
    "blue_tip_rack",
    "8_tube_stripes_rack",
    "micro_tube",
    "vortex_mixer",
)
TRACK_CLASSES = ("cell_culture_plate", "blue_pipette", "centrifuge", "50ml_tube")
COLOURS = {
    "cell_culture_plate": (0, 255, 0),
    "blue_pipette": (255, 128, 0),
    "centrifuge": (0, 200, 255),
    "50ml_tube": (255, 0, 255),
    "micro_tube_rack": (255, 255, 0),
    "blue_tip_rack": (128, 0, 255),
    "8_tube_stripes_rack": (0, 128, 255),
    "micro_tube": (255, 255, 255),
    "vortex_mixer": (0, 0, 255),
}


def video_path(trial: str, view: str) -> Path:
    if view == "fpv":
        return RAW / "finebio_videos_fpv_test/finebio_videos" / f"{trial}.mp4"
    return RAW / "finebio_videos_tpv_test/finebio_videos" / f"{trial}_{view}.mp4"


def load_detections(root: Path, view: str) -> dict[int, list[dict]]:
    out = {}
    for line in (root / f"{view}.jsonl").read_text().splitlines():
        rec = json.loads(line)
        out[int(rec["frame_index"])] = rec["detections"]
    return out


def top_box(dets: list[dict], cls: str, min_score: float) -> dict | None:
    cands = [d for d in dets if d["class"] == cls and d["score"] >= min_score]
    return max(cands, key=lambda d: d["score"]) if cands else None


def norm_box(box: list[float], w: int, h: int) -> list[tuple[float, float]]:
    return [(box[0] / w, box[1] / h), (box[2] / w, box[3] / h)]


def to_binary(mask_logits: torch.Tensor, hw: tuple[int, int]) -> np.ndarray:
    m = torch.nn.functional.interpolate(
        mask_logits.float(), size=hw, mode="bilinear", align_corners=False
    )
    return m.gt(0).squeeze().cpu().numpy()


def mask_metrics(mask: np.ndarray, box: list[float]) -> dict:
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, mask.shape[1]), min(y1, mask.shape[0])
    area = int(mask.sum())
    inside = int(mask[y0:y1, x0:x1].sum())
    box_area = max((x1 - x0) * (y1 - y0), 1)
    ys, xs = np.nonzero(mask)
    if len(xs):
        mb = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
        ix0, iy0 = max(mb[0], x0), max(mb[1], y0)
        ix1, iy1 = min(mb[2], x1), min(mb[3], y1)
        inter = max(ix1 - ix0, 0) * max(iy1 - iy0, 0)
        union = (mb[2] - mb[0]) * (mb[3] - mb[1]) + box_area - inter
        bbox_iou = inter / union if union else 0.0
    else:
        mb, bbox_iou = None, 0.0
    return {
        "mask_area_px": area,
        "fill_ratio": inside / box_area,
        "spill_ratio": (area - inside) / area if area else 0.0,
        "mask_bbox_px": None if mb is None else [int(v) for v in mb],
        "mask_bbox_iou_vs_prompt": float(bbox_iou),
    }


def overlay(
    img: np.ndarray, masks: list[tuple[str, np.ndarray, list[float] | None, str]]
) -> np.ndarray:
    vis = img.copy()
    for cls, mask, box, text in masks:
        col = COLOURS.get(cls, (200, 200, 200))
        if mask is not None and mask.any():
            layer = vis.copy()
            layer[mask] = col
            vis = cv2.addWeighted(vis, 0.55, layer, 0.45, 0)
            cnts, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(vis, cnts, -1, col, 2)
        if box is not None:
            x0, y0, x1, y1 = (int(v) for v in box)
            cv2.rectangle(vis, (x0, y0), (x1, y1), col, 2)
            cv2.putText(vis, text, (x0, max(14, y0 - 6)), 0, 0.55, col, 2, cv2.LINE_AA)
    return vis


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--trial", default="P03_01_01")
    parser.add_argument("--frame", type=int, default=1798)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decode-views", default="fpv,T1,T2,T3,T4,T5")
    parser.add_argument("--track-views", default="fpv,T5,T2")
    parser.add_argument("--track-frames", type=int, default=300)
    parser.add_argument("--min-score", type=float, default=0.3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    core = make_sam_from_state_dict(MODEL)
    core.to(device="cuda:0", dtype=torch.bfloat16)
    interact = core.get_interactive_context()
    tracking = core.get_tracking_context()
    load_s = time.perf_counter() - t0
    report: dict = {
        "trial": args.trial,
        "frame": args.frame,
        "model_load_s": load_s,
        "decode": [],
        "track": {},
    }

    # ---------------------------------------------------------------- check 1: box-prompted decode
    for view in args.decode_views.split(","):
        dets = load_detections(args.detections, view)
        if args.frame not in dets:
            continue
        cap = cv2.VideoCapture(str(video_path(args.trial, view)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ok, img = cap.read()
        cap.release()
        if not ok:
            continue
        h, w = img.shape[:2]
        sizes = (1280,) if view == "fpv" else (1280, 1920)
        for side in sizes:
            with torch.inference_mode():
                enc = interact.encode_image(img, side, True)
            drawn = []
            for cls in DECODE_CLASSES:
                b = top_box(dets[args.frame], cls, args.min_score)
                if not b:
                    continue
                with torch.inference_mode():
                    prompts = interact.encode_prompts([norm_box(b["box_xyxy_px"], w, h)], [], [])
                    masks, ious = interact.generate_masks(enc, prompts)
                best = int(ious[0].argmax())
                mask = to_binary(masks[:, [best]], (h, w))
                metrics = mask_metrics(mask, b["box_xyxy_px"])
                row = {
                    "view": view,
                    "encoder_side": side,
                    "class": cls,
                    "detector_score": b["score"],
                    "box_px": b["box_xyxy_px"],
                    "box_wh_px": [
                        b["box_xyxy_px"][2] - b["box_xyxy_px"][0],
                        b["box_xyxy_px"][3] - b["box_xyxy_px"][1],
                    ],
                    "decoder_iou_pred": float(ious[0, best]),
                    **metrics,
                }
                report["decode"].append(row)
                drawn.append(
                    (
                        cls,
                        mask,
                        b["box_xyxy_px"],
                        f"{cls} fill {metrics['fill_ratio']:.2f} iou {row['decoder_iou_pred']:.2f}",
                    )
                )
            vis = overlay(img, drawn)
            cv2.putText(
                vis,
                f"{args.trial} {view} f{args.frame} box-prompted SAM3 decode, encoder {side}",
                (10, 30),
                0,
                0.9,
                (255, 255, 0),
                2,
            )
            cv2.imwrite(
                str(args.output / f"decode_{view}_{side}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 85]
            )
        print(
            f"decode {view}: {len([r for r in report['decode'] if r['view'] == view])} rows",
            flush=True,
        )

    # ---------------------------------------------------------------- check 2: box-seeded track
    for view in args.track_views.split(","):
        dets = load_detections(args.detections, view)
        cap = cv2.VideoCapture(str(video_path(args.trial, view)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ok, first = cap.read()
        if not ok:
            continue
        h, w = first.shape[:2]
        seeds = [(cls, top_box(dets[args.frame], cls, args.min_score)) for cls in TRACK_CLASSES]
        seeds = [(cls, b) for cls, b in seeds if b]
        if not seeds:
            continue
        with torch.inference_mode():
            enc = interact.encode_image(first, 1280, True)
            seed_masks = []
            for cls, b in seeds:
                prompts = interact.encode_prompts([norm_box(b["box_xyxy_px"], w, h)], [], [])
                masks, ious = interact.generate_masks(enc, prompts)
                seed_masks.append(to_binary(masks[:, [int(ious[0].argmax())]], (h, w)))
            tenc = tracking.encode_image(first, 1280, True)
            prompt_mem = tracking.encode_prompt_memory_from_mask(tenc, np.stack(seed_masks, axis=0))
        prompt_mems = deque([prompt_mem], maxlen=1)
        frame_mems: deque = deque([], maxlen=4)
        n = len(seeds)
        series = {cls: [] for cls, _ in seeds}
        overlays = {0, 75, 150, 225, args.track_frames - 1}
        vis0 = overlay(
            first,
            [
                (cls, m, b["box_xyxy_px"], f"seed {cls} {b['score']:.2f}")
                for (cls, b), m in zip(seeds, seed_masks)
            ],
        )
        cv2.putText(vis0, f"{view} f{args.frame} seeds", (10, 30), 0, 0.9, (255, 255, 0), 2)
        cv2.imwrite(
            str(args.output / f"track_{view}_f{args.frame:05d}.jpg"),
            vis0,
            [cv2.IMWRITE_JPEG_QUALITY, 80],
        )
        step_times = []
        for i in range(1, args.track_frames):
            ok, frame = cap.read()
            if not ok:
                break
            f = args.frame + i
            t1 = time.perf_counter()
            with torch.inference_mode():
                tenc = tracking.encode_image(frame, 1280, True)
                masks_m, ious_m, ptrs_m, scores_m = tracking.step_video_masking_multiplex(
                    tenc, prompt_mems, frame_mems, num_multiplex_objects=n
                )
                ok_obj = scores_m > 0
                if bool(ok_obj.any()):
                    frame_mems.append(tracking.encode_frame_memory(tenc, masks_m, ptrs_m, scores_m))
            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - t1)
            drawn = []
            for k, (cls, _) in enumerate(seeds):
                score = float(scores_m[k])
                if score > 0:
                    mask = to_binary(masks_m[[k]], (h, w))
                    ys, xs = np.nonzero(mask)
                    mb = (
                        [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
                        if len(xs)
                        else None
                    )
                else:
                    mask, mb = None, None
                row = {
                    "frame": f,
                    "object_score": score,
                    "area_frac": (float(mask.sum()) / (h * w)) if mask is not None else 0.0,
                    "mask_bbox_px": mb,
                }
                det_box = top_box(dets.get(f, []), cls, args.min_score) if f in dets else None
                if det_box and mb:
                    x0, y0, x1, y1 = det_box["box_xyxy_px"]
                    ix0, iy0, ix1, iy1 = (
                        max(mb[0], x0),
                        max(mb[1], y0),
                        min(mb[2], x1),
                        min(mb[3], y1),
                    )
                    inter = max(ix1 - ix0, 0) * max(iy1 - iy0, 0)
                    union = (mb[2] - mb[0]) * (mb[3] - mb[1]) + (x1 - x0) * (y1 - y0) - inter
                    row["detector_box_iou"] = inter / union if union else 0.0
                    row["detector_score"] = det_box["score"]
                elif det_box:
                    row["detector_box_iou"] = 0.0
                    row["detector_score"] = det_box["score"]
                series[cls].append(row)
                drawn.append(
                    (
                        cls,
                        mask,
                        det_box["box_xyxy_px"] if det_box else None,
                        f"{cls} s={score:.1f}"
                        + (
                            f" iou={row['detector_box_iou']:.2f}"
                            if "detector_box_iou" in row
                            else ""
                        ),
                    )
                )
            if i in overlays:
                vis = overlay(frame, drawn)
                cv2.putText(
                    vis,
                    f"{view} f{f} (+{i}) SAM3.1 track at 1280; boxes = DINO",
                    (10, 30),
                    0,
                    0.9,
                    (255, 255, 0),
                    2,
                )
                cv2.imwrite(
                    str(args.output / f"track_{view}_f{f:05d}.jpg"),
                    vis,
                    [cv2.IMWRITE_JPEG_QUALITY, 80],
                )
        cap.release()
        summary = {}
        for cls, rows in series.items():
            lost = [r["frame"] for r in rows if r["object_score"] <= 0]
            ious = [r["detector_box_iou"] for r in rows if "detector_box_iou" in r]
            summary[cls] = {
                "frames": len(rows),
                "frames_lost": len(lost),
                "lost_frames_first_last": [lost[0], lost[-1]] if lost else None,
                "object_score_median": float(np.median([r["object_score"] for r in rows])),
                "area_frac_min_med_max": [
                    float(min(r["area_frac"] for r in rows)),
                    float(np.median([r["area_frac"] for r in rows])),
                    float(max(r["area_frac"] for r in rows)),
                ],
                "detector_box_iou": {
                    "n": len(ious),
                    "median": float(np.median(ious)),
                    "p10": float(np.percentile(ious, 10)),
                    "frac_over_0.5": float(np.mean(np.array(ious) > 0.5)),
                }
                if ious
                else None,
            }
        report["track"][view] = {
            "seeds": [
                {"class": cls, "score": b["score"], "box_px": b["box_xyxy_px"]} for cls, b in seeds
            ],
            "step_ms_median": float(np.median(step_times) * 1000) if step_times else None,
            "summary": summary,
            "series": series,
        }
        print(
            f"track {view}: {n} objects, {len(step_times)} steps, "
            f"{np.median(step_times) * 1000:.0f} ms/step",
            flush=True,
        )
        for cls, s in summary.items():
            print(
                f"   {cls}: lost {s['frames_lost']}/{s['frames']}, "
                f"score med {s['object_score_median']:.1f}, det-box IoU "
                + (
                    f"med {s['detector_box_iou']['median']:.2f} "
                    f"(n={s['detector_box_iou']['n']}, "
                    f">0.5 on {s['detector_box_iou']['frac_over_0.5']:.2f})"
                    if s["detector_box_iou"]
                    else "n/a"
                ),
                flush=True,
            )

    report["peak_vram_gib"] = torch.cuda.max_memory_allocated() / 2**30
    (args.output / "sam3_preflight.json").write_text(json.dumps(report, indent=1))
    # decode table
    lines = [
        "| view | side | class | det score | box w x h | decoder IoU | fill | spill | bbox IoU |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in report["decode"]:
        lines.append(
            f"| {r['view']} | {r['encoder_side']} | {r['class']} | {r['detector_score']:.2f} | "
            f"{r['box_wh_px'][0]:.0f}x{r['box_wh_px'][1]:.0f} | {r['decoder_iou_pred']:.2f} | "
            f"{r['fill_ratio']:.2f} | {r['spill_ratio']:.2f} | "
            f"{r['mask_bbox_iou_vs_prompt']:.2f} |"
        )
    (args.output / "decode_table.md").write_text("\n".join(lines) + "\n")
    print(
        f"PREFLIGHT_SAM3_DONE {time.perf_counter() - t0:.0f}s "
        f"peak VRAM {report['peak_vram_gib']:.2f} GiB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
