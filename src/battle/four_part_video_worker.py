"""External-environment worker for real four-target mask propagation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path
from time import perf_counter
from typing import Any

TARGETS = ("chassis", "interior", "rear_body", "cabin")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _extract_frames(video: Path, directory: Path, frame_count: int) -> list[Path]:
    import cv2

    directory.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    frames: list[Path] = []
    while len(frames) < frame_count:
        ok, frame = capture.read()
        if not ok:
            break
        path = directory / f"{len(frames):05d}.jpg"
        cv2.imwrite(str(path), frame)
        frames.append(path)
    capture.release()
    if len(frames) != frame_count:
        raise RuntimeError(f"decoded {len(frames)} frames, expected {frame_count}")
    return frames


def _box(mask: Any, width: int, height: int) -> dict[str, float]:
    import numpy as np

    ys, xs = np.where(mask)
    if not len(xs):
        raise ValueError("cannot box an empty mask")
    return {
        "x": float(xs.min() / width),
        "y": float(ys.min() / height),
        "width": float((xs.max() + 1 - xs.min()) / width),
        "height": float((ys.max() + 1 - ys.min()) / height),
    }


def _read_seed_masks(contract: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    output: dict[str, Any] = {}
    for seed in contract["reviewed_frame_zero_seeds"]:
        path = Path(seed["mask_path"])
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != seed["mask_sha256"]:
            raise ValueError(f"reviewed seed changed: {seed['target_id']}")
        with Image.open(path) as image:
            mask = np.asarray(image.convert("L"), dtype=np.uint8) > 0
        if not mask.any():
            raise ValueError(f"reviewed seed is empty: {seed['target_id']}")
        output[seed["target_id"]] = mask
    if tuple(output) != TARGETS:
        raise ValueError("worker contract target order is invalid")
    return output


def _write_observations(
    *,
    run_directory: Path,
    frames: list[Path],
    source_offset_seconds: float,
    analysis_fps: float,
    masks_by_frame: dict[int, dict[str, Any]],
    confidences: dict[str, float],
    method: str,
) -> tuple[dict[str, int], dict[str, int]]:
    import cv2
    import numpy as np

    mask_root = run_directory / "native" / "masks"
    coverage = {target: 0 for target in TARGETS}
    variations: dict[str, set[str]] = {target: set() for target in TARGETS}
    with (run_directory / "observations.jsonl").open("w", encoding="utf-8") as output:
        for index, frame_path in enumerate(frames):
            image = cv2.imread(str(frame_path))
            assert image is not None
            height, width = image.shape[:2]
            objects: list[dict[str, Any]] = []
            for target in TARGETS:
                mask = masks_by_frame.get(index, {}).get(target)
                if mask is None:
                    continue
                mask = np.asarray(mask, dtype=bool).squeeze()
                if mask.shape != (height, width) or not mask.any():
                    continue
                target_root = mask_root / target
                target_root.mkdir(parents=True, exist_ok=True)
                mask_name = f"{index:05d}.png"
                bytes_ = mask.astype(np.uint8) * 255
                cv2.imwrite(str(target_root / mask_name), bytes_)
                coverage[target] += 1
                variations[target].add(hashlib.sha256(bytes_.tobytes()).hexdigest())
                objects.append(
                    {
                        "object_id": f"{method}-{target}",
                        "label": target,
                        "confidence": confidences.get(target, 1.0),
                        "box": _box(mask, width, height),
                        "mask": {
                            "uri": f"native/masks/{target}/{mask_name}",
                            "storage": "native_artifact",
                            "format": "png",
                        },
                    }
                )
            output.write(
                json.dumps(
                    {
                        "view_id": "static-c10379",
                        "analysis_frame_index": index,
                        "source_seconds": source_offset_seconds + index / analysis_fps,
                        "objects": objects,
                    }
                )
                + "\n"
            )
    return coverage, {target: len(values) for target, values in variations.items()}


def _sam2_propagate(
    *,
    frames: list[Path],
    sam2_root: Path,
    sam2_config: str,
    checkpoint: Path,
    initial_masks: dict[str, Any] | None,
    initial_boxes: dict[str, Any] | None,
) -> dict[int, dict[str, Any]]:
    import numpy as np

    sys.path.insert(0, str(sam2_root))
    from sam2.build_sam import build_sam2_video_predictor

    predictor = build_sam2_video_predictor(sam2_config, str(checkpoint), device="cuda:0")
    state = predictor.init_state(video_path=str(frames[0].parent), offload_video_to_cpu=True)
    selected = initial_masks if initial_masks is not None else initial_boxes or {}
    for object_id, target in enumerate(TARGETS):
        if target not in selected:
            continue
        if initial_masks is not None:
            predictor.add_new_mask(state, frame_idx=0, obj_id=object_id, mask=initial_masks[target])
        else:
            predictor.add_new_points_or_box(
                state,
                frame_idx=0,
                obj_id=object_id,
                box=np.asarray(initial_boxes[target], dtype=np.float32),
            )
    output: dict[int, dict[str, Any]] = {}
    for frame_index, object_ids, logits in predictor.propagate_in_video(state):
        output[frame_index] = {
            TARGETS[int(object_id)]: (logits[position] > 0).cpu().numpy()
            for position, object_id in enumerate(object_ids)
        }
    return output


def _grounding_boxes(
    frame_path: Path, prompts: dict[str, list[str]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    model_id = "IDEA-Research/grounding-dino-tiny"
    revision = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, revision=revision).to(
        "cuda"
    )
    image = Image.open(frame_path)
    boxes: dict[str, Any] = {}
    report: dict[str, Any] = {}
    for target in TARGETS:
        attempts: list[dict[str, Any]] = []
        best: tuple[float, Any, str] | None = None
        for prompt in prompts[target]:
            inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                outputs = model(**inputs)
            detection = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=0.4,
                text_threshold=0.3,
                target_sizes=[image.size[::-1]],
            )[0]
            scores = detection["scores"].detach().cpu().tolist()
            detected_boxes = detection["boxes"].detach().cpu().tolist()
            attempts.append({"prompt": prompt, "scores": scores, "boxes_xyxy": detected_boxes})
            if scores:
                index = max(range(len(scores)), key=scores.__getitem__)
                candidate = (float(scores[index]), detected_boxes[index], prompt)
                if best is None or candidate[0] > best[0]:
                    best = candidate
        if best is None:
            report[target] = {"status": "failed", "attempts": attempts}
        else:
            boxes[target] = best[1]
            report[target] = {
                "status": "succeeded",
                "selected_prompt": best[2],
                "selected_score": best[0],
                "selected_box_xyxy": best[1],
                "attempts": attempts,
            }
    return boxes, report


def _dam4sam_propagate(
    *, frames: list[Path], dam4sam_root: Path, initial_masks: dict[str, Any]
) -> tuple[dict[int, dict[str, Any]], dict[str, int]]:
    import numpy as np
    from PIL import Image

    sys.path.insert(0, str(dam4sam_root))
    from dam4sam_tracker import DAM4SAMTracker

    trackers = {target: DAM4SAMTracker(tracker_name="sam21pp-T") for target in TARGETS}
    output: dict[int, dict[str, Any]] = {}
    drm_additions = {target: 0 for target in TARGETS}
    prior_additions = {target: -1 for target in TARGETS}
    for frame_index, frame_path in enumerate(frames):
        with Image.open(frame_path) as image:
            masks: dict[str, Any] = {}
            for target in TARGETS:
                result = (
                    trackers[target].initialize(image, initial_masks[target])
                    if frame_index == 0
                    else trackers[target].track(image)
                )
                if trackers[target].last_added != prior_additions[target]:
                    drm_additions[target] += 1
                    prior_additions[target] = trackers[target].last_added
                mask = np.asarray(result["pred_mask"], dtype=bool)
                if mask.any():
                    masks[target] = mask
            output[frame_index] = masks
    return output, drm_additions


def _samurai_independent_streams(
    *,
    frames: list[Path],
    sam2_root: Path,
    sam2_config: str,
    checkpoint: Path,
    seeds: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    """SAMURAI's mode is not multi-object safe; release one real predictor per target."""
    import torch

    output: dict[int, dict[str, Any]] = {frame: {} for frame in range(len(frames))}
    for target in TARGETS:
        stream = _sam2_propagate(
            frames=frames,
            sam2_root=sam2_root,
            sam2_config=sam2_config,
            checkpoint=checkpoint,
            initial_masks={target: seeds[target]},
            initial_boxes=None,
        )
        for frame, masks in stream.items():
            output[frame].update(masks)
        torch.cuda.empty_cache()
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=(
            "grounding_dino_sam2_open_vocabulary",
            "reviewed_seed_sam2_control",
            "samurai",
            "dam4sam",
        ),
        required=True,
    )
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--source-offset-seconds", type=float, required=True)
    parser.add_argument("--analysis-fps", type=float, default=30.0)
    parser.add_argument("--frame-count", type=int, default=600)
    parser.add_argument("--sam2-root", type=Path, required=True)
    parser.add_argument("--sam2-config", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    started = perf_counter()
    result: dict[str, Any] = {"state": "failed", "method": args.method, "initialization": {}}
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        contract = json.loads(args.contract.read_text(encoding="utf-8"))
        if tuple(contract["targets"]) != TARGETS or args.frame_count not in (600, 1800):
            raise ValueError("worker only supports the 600- or 1800-frame ordered-target contract")
        frames = _extract_frames(
            args.video, args.run_directory / "native" / "frames", args.frame_count
        )
        seeds = _read_seed_masks(contract)
        confidences = {target: 1.0 for target in TARGETS}
        extra: dict[str, Any] = {}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            if args.method == "grounding_dino_sam2_open_vocabulary":
                boxes, initialization = _grounding_boxes(
                    frames[0], contract["open_vocabulary_prompts"]
                )
                extra["initialization"] = initialization
                confidences = {
                    target: float(initialization[target].get("selected_score", 0.0))
                    for target in TARGETS
                }
                masks = (
                    _sam2_propagate(
                        frames=frames,
                        sam2_root=args.sam2_root,
                        sam2_config=args.sam2_config,
                        checkpoint=args.checkpoint,
                        initial_masks=None,
                        initial_boxes=boxes,
                    )
                    if boxes
                    else {}
                )
            elif args.method == "dam4sam":
                masks, additions = _dam4sam_propagate(
                    frames=frames, dam4sam_root=args.sam2_root, initial_masks=seeds
                )
                extra["initialization"] = {target: {"status": "succeeded"} for target in TARGETS}
                extra["drm_memory_additions"] = additions
            elif args.method == "samurai":
                masks = _samurai_independent_streams(
                    frames=frames,
                    sam2_root=args.sam2_root,
                    sam2_config=args.sam2_config,
                    checkpoint=args.checkpoint,
                    seeds=seeds,
                )
                extra["initialization"] = {target: {"status": "succeeded"} for target in TARGETS}
            else:
                masks = _sam2_propagate(
                    frames=frames,
                    sam2_root=args.sam2_root,
                    sam2_config=args.sam2_config,
                    checkpoint=args.checkpoint,
                    initial_masks=seeds,
                    initial_boxes=None,
                )
                extra["initialization"] = {target: {"status": "succeeded"} for target in TARGETS}
        coverage, variation = _write_observations(
            run_directory=args.run_directory,
            frames=frames,
            source_offset_seconds=args.source_offset_seconds,
            analysis_fps=args.analysis_fps,
            masks_by_frame=masks,
            confidences=confidences,
            method=args.method,
        )
        result.update(
            {
                "state": "succeeded",
                "frames_processed": len(frames),
                "per_target_mask_coverage": coverage,
                "per_target_unique_mask_hashes": variation,
                "gpu_peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
                **extra,
            }
        )
    except Exception as exc:  # noqa: BLE001
        result["reason"] = f"{type(exc).__name__}: {exc}"
        (args.run_directory / "worker_traceback.log").write_text(traceback.format_exc())
    result["elapsed_seconds"] = perf_counter() - started
    _write_json(args.run_directory / "worker_result.json", result)


if __name__ == "__main__":
    main()
