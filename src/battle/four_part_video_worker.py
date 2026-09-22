"""External-environment worker for real four-target mask propagation.

Runs under the method's own interpreter (samurai or grounded_sam2 pyenv), so it imports
nothing from the battle package; `dam4sam_streaming` sits beside it and is imported as a
top-level module.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import traceback
from pathlib import Path
from time import perf_counter
from typing import Any

TARGETS = ("chassis", "interior", "rear_body", "cabin")
DEFAULT_VIEW_ID = "static-c10379"
DEFAULT_SAM2_MODEL = "tiny"
DEFAULT_INPUT_SIZE = 1024
DEFAULT_VRAM_PROBE_FRAMES = "30,300"
SUPPORTED_FRAME_COUNTS = (300, 600, 1800)


def _sibling(name: str) -> Any:
    # The script's own directory is already sys.path[0] when run as a worker; this only
    # matters when the module is imported some other way.
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.append(here)
    return importlib.import_module(name)


def _streaming_module() -> Any:
    return _sibling("dam4sam_streaming")


def _frame_shape(frame_path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(frame_path) as image:
        return image.height, image.width


def _load_schedule(
    path: Path | None,
) -> tuple[dict[int, dict[str, dict[str, Any]]], dict[str, Any]]:
    """Read the driver-resolved schedule: later corrections keyed by frame and target."""
    if path is None:
        return {}, {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _streaming_module().corrections_by_frame(list(payload.get("corrections", []))), payload


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


def _read_seed_masks(
    contract: dict[str, Any], frame_shape: tuple[int, int] | None = None
) -> tuple[dict[str, Any], bool]:
    """Reviewed frame-zero masks; resized nearest-neighbour only when the proxy differs."""
    import numpy as np
    from PIL import Image

    output: dict[str, Any] = {}
    resized = False
    for seed in contract["reviewed_frame_zero_seeds"]:
        path = Path(seed["mask_path"])
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != seed["mask_sha256"]:
            raise ValueError(f"reviewed seed changed: {seed['target_id']}")
        with Image.open(path) as image:
            mask = np.asarray(image.convert("L"), dtype=np.uint8) > 0
        if not mask.any():
            raise ValueError(f"reviewed seed is empty: {seed['target_id']}")
        if frame_shape is not None and mask.shape != tuple(frame_shape):
            mask = _streaming_module().fit_mask(mask, frame_shape)
            resized = True
        output[seed["target_id"]] = mask
    if tuple(output) != TARGETS:
        raise ValueError("worker contract target order is invalid")
    return output, resized


def _write_observations(
    *,
    run_directory: Path,
    frames: list[Path],
    source_offset_seconds: float,
    analysis_fps: float,
    masks_by_frame: dict[int, dict[str, Any]],
    confidences: dict[str, float],
    method: str,
    view_id: str = DEFAULT_VIEW_ID,
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
                        "view_id": view_id,
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
    corrections: dict[int, dict[str, dict[str, Any]]] | None = None,
    probe_frames: tuple[int, ...] = (),
    probes: list[dict[str, int]] | None = None,
    correction_report: dict[str, Any] | None = None,
) -> dict[int, dict[str, Any]]:
    """Offline SAM2 propagation over the extracted frames.

    Scheduled corrections are added as conditioning frames *before* `propagate_in_video`,
    at their scheduled indices: the offline API consolidates every prompt in its preflight
    and refuses new prompts mid-propagation, so this is the only order it allows.  The
    driver records that difference (`correction_timing`) in the manifest.
    """
    import numpy as np

    sys.path.insert(0, str(sam2_root))
    from sam2.build_sam import build_sam2_video_predictor

    streaming = _streaming_module()
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
    frame_shape = _frame_shape(frames[0])
    for frame_index in sorted(corrections or {}):
        for target, entry in (corrections or {})[frame_index].items():
            if target not in selected:
                continue
            mask, resized = streaming.read_correction_mask(
                entry["mask_path"], entry["mask_sha256"], frame_shape
            )
            if resized and correction_report is not None:
                correction_report["correction_masks_resized_to_frame"] = True
            predictor.add_new_mask(
                state, frame_idx=frame_index, obj_id=TARGETS.index(target), mask=mask
            )
            if correction_report is not None:
                correction_report.setdefault("corrections_applied", []).append(
                    {"frame_index": frame_index, "target": target}
                )
    output: dict[int, dict[str, Any]] = {}
    for frame_index, object_ids, logits in predictor.propagate_in_video(state):
        output[frame_index] = {
            TARGETS[int(object_id)]: (logits[position] > 0).cpu().numpy()
            for position, object_id in enumerate(object_ids)
        }
        if probes is not None and frame_index + 1 in probe_frames:
            probes.append(streaming.cuda_memory_probe(frame_index + 1))
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
    *,
    frames: list[Path],
    dam4sam_root: Path,
    run_directory: Path,
    initial_masks: dict[str, Any],
    sam2_model: str = DEFAULT_SAM2_MODEL,
    input_size: int = DEFAULT_INPUT_SIZE,
    corrections: dict[int, dict[str, dict[str, Any]]] | None = None,
    add_correction_to_drm: bool = False,
    probe_frames: tuple[int, ...] = (),
) -> tuple[dict[int, dict[str, Any]], dict[str, int], dict[str, Any]]:
    """Four DAM4SAM DRM streams on one shared SAM2 predictor, corrected mid-stream."""
    import numpy as np
    from PIL import Image

    sys.path.insert(0, str(dam4sam_root))
    streaming = _streaming_module()
    predictor, provenance = streaming.build_shared_predictor(
        dam4sam_root, sam2_model, input_size, run_directory / "sam2_config"
    )
    tracker_class = streaming.shared_predictor_tracker_class()
    trackers = {
        target: tracker_class(
            predictor, input_image_size=input_size, add_correction_to_drm=add_correction_to_drm
        )
        for target in TARGETS
    }
    corrections = corrections or {}
    output: dict[int, dict[str, Any]] = {}
    drm_additions = {target: 0 for target in TARGETS}
    prior_additions = {target: -1 for target in TARGETS}
    probes: list[dict[str, int]] = []
    applied: list[dict[str, Any]] = []
    masks_resized = False
    for frame_index, frame_path in enumerate(frames):
        with Image.open(frame_path) as image:
            frame_shape = (image.height, image.width)
            masks: dict[str, Any] = {}
            for target in TARGETS:
                tracker = trackers[target]
                result = (
                    tracker.initialize(image, initial_masks[target])
                    if frame_index == 0
                    else tracker.track(image)
                )
                entry = corrections.get(frame_index, {}).get(target)
                if entry is not None:
                    correction_mask, resized = streaming.read_correction_mask(
                        entry["mask_path"], entry["mask_sha256"], frame_shape
                    )
                    masks_resized = masks_resized or resized
                    result = tracker.correct(image, correction_mask)
                    applied.append({"frame_index": frame_index, "target": target})
                if tracker.last_added != prior_additions[target]:
                    drm_additions[target] += 1
                    prior_additions[target] = tracker.last_added
                mask = np.asarray(result["pred_mask"], dtype=bool)
                if mask.any():
                    masks[target] = mask
            output[frame_index] = masks
        if frame_index + 1 in probe_frames:
            probes.append(streaming.cuda_memory_probe(frame_index + 1))
    extra = {
        "sam2": {**provenance, "add_correction_to_drm": bool(add_correction_to_drm)},
        "vram_probes": probes,
        "corrections_applied": applied,
        "correction_masks_resized_to_frame": masks_resized,
    }
    return output, drm_additions, extra


def _samurai_independent_streams(
    *,
    frames: list[Path],
    sam2_root: Path,
    sam2_config: str,
    checkpoint: Path,
    seeds: dict[str, Any],
    corrections: dict[int, dict[str, dict[str, Any]]] | None = None,
    probe_frames: tuple[int, ...] = (),
    correction_report: dict[str, Any] | None = None,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, int]]]:
    """SAMURAI's mode is not multi-object safe; release one real predictor per target.

    VRAM probes are taken in every per-target stream and merged as the per-frame maximum.
    """
    import torch

    output: dict[int, dict[str, Any]] = {frame: {} for frame in range(len(frames))}
    probe_lists: list[list[dict[str, int]]] = []
    for target in TARGETS:
        probes: list[dict[str, int]] = []
        stream = _sam2_propagate(
            frames=frames,
            sam2_root=sam2_root,
            sam2_config=sam2_config,
            checkpoint=checkpoint,
            initial_masks={target: seeds[target]},
            initial_boxes=None,
            corrections=corrections,
            probe_frames=probe_frames,
            probes=probes,
            correction_report=correction_report,
        )
        probe_lists.append(probes)
        for frame, masks in stream.items():
            output[frame].update(masks)
        torch.cuda.empty_cache()
    return output, _streaming_module().merge_probes(probe_lists)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--view-id", default=DEFAULT_VIEW_ID)
    parser.add_argument(
        "--sam2-model",
        choices=("tiny", "large"),
        default=DEFAULT_SAM2_MODEL,
        help="DAM4SAM only: which SAM2.1 Hiera checkpoint + yaml pair the shared predictor loads.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=DEFAULT_INPUT_SIZE,
        help="DAM4SAM only: tracker input_image_size and the yaml image_size (1024 or 1536).",
    )
    parser.add_argument(
        "--multi-keyframe-correction-schedule",
        type=Path,
        default=None,
        help="Driver-resolved schedule JSON; later corrections are applied via add_new_mask.",
    )
    parser.add_argument(
        "--add-correction-to-drm",
        action="store_true",
        help="DAM4SAM only: let a correction frame count as a DRM addition (last_added).",
    )
    parser.add_argument(
        "--vram-probe-frames",
        default=DEFAULT_VRAM_PROBE_FRAMES,
        help="Frames-processed counts at which torch.cuda allocator counters are recorded.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    started = perf_counter()
    result: dict[str, Any] = {"state": "failed", "method": args.method, "initialization": {}}
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        contract = json.loads(args.contract.read_text(encoding="utf-8"))
        if tuple(contract["targets"]) != TARGETS or args.frame_count not in SUPPORTED_FRAME_COUNTS:
            raise ValueError(
                "worker only supports the 300- (smoke), 600- or 1800-frame ordered-target contract"
            )
        if args.method != "dam4sam" and (
            args.sam2_model != DEFAULT_SAM2_MODEL or args.input_size != DEFAULT_INPUT_SIZE
        ):
            raise ValueError(
                "--sam2-model/--input-size are DAM4SAM knobs; the offline SAM2 arms take their "
                "checkpoint from --checkpoint/--sam2-config and image_size from that yaml"
            )
        probe_frames = _streaming_module().parse_probe_frames(args.vram_probe_frames)
        corrections, schedule_payload = _load_schedule(args.multi_keyframe_correction_schedule)
        frames = _extract_frames(
            args.video, args.run_directory / "native" / "frames", args.frame_count
        )
        frame_shape = _frame_shape(frames[0])
        seeds, seeds_resized = _read_seed_masks(contract, frame_shape)
        confidences = {target: 1.0 for target in TARGETS}
        extra: dict[str, Any] = {
            "seed_masks_resized_to_frame": seeds_resized,
            "vram_probe_frames": list(probe_frames),
            "correction_masks_resized_to_frame": False,
            "corrections_applied": [],
        }
        if schedule_payload:
            extra["correction_schedule"] = {
                key: value for key, value in schedule_payload.items() if key != "corrections"
            }
        probes: list[dict[str, int]] = []
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
                        corrections=corrections,
                        probe_frames=probe_frames,
                        probes=probes,
                        correction_report=extra,
                    )
                    if boxes
                    else {}
                )
            elif args.method == "dam4sam":
                masks, additions, dam4sam_extra = _dam4sam_propagate(
                    frames=frames,
                    dam4sam_root=args.sam2_root,
                    run_directory=args.run_directory,
                    initial_masks=seeds,
                    sam2_model=args.sam2_model,
                    input_size=args.input_size,
                    corrections=corrections,
                    add_correction_to_drm=args.add_correction_to_drm,
                    probe_frames=probe_frames,
                )
                extra["initialization"] = {target: {"status": "succeeded"} for target in TARGETS}
                extra["drm_memory_additions"] = additions
                probes = dam4sam_extra.pop("vram_probes")
                extra.update(dam4sam_extra)
            elif args.method == "samurai":
                masks, probes = _samurai_independent_streams(
                    frames=frames,
                    sam2_root=args.sam2_root,
                    sam2_config=args.sam2_config,
                    checkpoint=args.checkpoint,
                    seeds=seeds,
                    corrections=corrections,
                    probe_frames=probe_frames,
                    correction_report=extra,
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
                    corrections=corrections,
                    probe_frames=probe_frames,
                    probes=probes,
                    correction_report=extra,
                )
                extra["initialization"] = {target: {"status": "succeeded"} for target in TARGETS}
        extra["vram_probes"] = probes
        coverage, variation = _write_observations(
            run_directory=args.run_directory,
            frames=frames,
            source_offset_seconds=args.source_offset_seconds,
            analysis_fps=args.analysis_fps,
            masks_by_frame=masks,
            confidences=confidences,
            method=args.method,
            view_id=args.view_id,
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
    _sibling("fs_common").write_json(args.run_directory / "worker_result.json", result)


if __name__ == "__main__":
    main()
