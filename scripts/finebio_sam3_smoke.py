"""SAM3 zero-shot text-prompt smoke on one FineBio first-person clip, visualised in Rerun.

Two phases, two interpreters, because the MuggledSAM environment has no Rerun and the
Battle environment has no SAM3:

    /home/nick/.pyenv/versions/muggled_sam/bin/python scripts/finebio_sam3_smoke.py track ...
    uv run python scripts/finebio_sam3_smoke.py export ...

``track`` copies the SAM3 calls of ``src/battle/muggled_worker.py`` (text detection at frame 0,
top-scored detection per prompt above the worker's threshold, then one continuous multiplex
tracker stream with the worker's default memory banks) and writes ``masks/``,
``observations.jsonl`` and ``track_result.json``.  ``export`` reads those and writes the RRD
(video asset, per-frame VideoFrameReference, per-object RGBA EncodedImage masks, labelled
boxes, per-object area and raw object-score series, pinned blueprint), a six-frame contact
sheet, and ``manifest.json``.

This is a toolchain smoke on wet-lab footage, not an experiment. Nothing here is measured
against ground truth and no accuracy is claimed.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import traceback
from collections import deque
from pathlib import Path
from time import perf_counter
from typing import Any

try:
    from battle import fs_common, gpu_guard
except ImportError:
    # Under the MuggledSAM interpreter the battle package is not installed: import the
    # stdlib-only helper modules by path, as the workers do.
    sys.path.append(str(Path(__file__).resolve().parents[1] / "src" / "battle"))
    import fs_common  # type: ignore[no-redef]
    import gpu_guard  # type: ignore[no-redef]

# Worker constants, restated so the run is comparable with the Assembly101 text-prompt runs.
DETECTION_THRESHOLD = 0.40
MAX_FRAME_MEMORY = 4
MAX_PROMPT_MEMORY = 1
IS_RECENT_FIRST = False
BOX_COMPONENT_KEEP_FRACTION = 0.20
DEFAULT_MODEL = Path("/home/nick/src/muggled_sam/model_weights/sam3.1_multiplex.pt")
MUGGLED_SAM_SOURCE = Path("/home/nick/src/muggled_sam")
# Processes whose presence means another tracker owns the GPU; matched against `pgrep -af`.
TRACKER_PROCESS_PATTERN = "muggled_worker|battle-muggled-smoke|four_part_video_worker"
OBJECT_COLORS = (
    (255, 85, 85),
    (255, 210, 65),
    (80, 180, 255),
    (180, 115, 255),
    (70, 210, 165),
)
CLAIM_BOUNDARY = (
    "Toolchain smoke on one 20-second FineBio first-person clip. Frame-0 detections are the "
    "SAM3 detector's top-scored box/mask per text prompt above the threshold; every later "
    "frame is the tracker's own output with no correction. Object scores and predicted IoU "
    "are the model's self-reports. Nothing is compared with ground truth; masks were not "
    "reviewed; no accuracy, coverage or generalisation claim is made."
)


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


# --------------------------------------------------------------------------------------------
# GPU guard (the worker's strict rule through `battle.gpu_guard`: refuse beside any model-like
# GPU process not allowed by PID, and additionally refuse while a Battle tracker process
# exists at all).


def tracker_processes() -> list[str]:
    completed = subprocess.run(
        ["pgrep", "-af", TRACKER_PROCESS_PATTERN], check=False, capture_output=True, text=True
    )
    own = str(Path(__file__).name)
    return [
        line
        for line in completed.stdout.splitlines()
        if own not in line and "pgrep" not in line and line.strip()
    ]


def guard_gpu(
    allowed_pids: list[int],
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[str]]:
    """Return (blocking GPU processes, tolerated GPU processes, foreign tracker processes)."""
    blocking, tolerated = gpu_guard.partition_neighbours_strict(
        gpu_guard.legacy_gpu_processes(), allowed_pids
    )
    return blocking, tolerated, tracker_processes()


# --------------------------------------------------------------------------------------------
# track phase (MuggledSAM interpreter).  The SAM3 calls below are a deliberate copy of
# `src/battle/muggled_worker.py`'s detector / multiplex-tracker sequence, kept separate on
# purpose: this is a dual-interpreter smoke of the upstream API, not a Battle adapter.


def binary_mask(mask_logits: Any, frame_shape: tuple[int, int]) -> Any:
    import torch.nn.functional as functional

    resized = functional.interpolate(
        mask_logits, size=frame_shape, mode="bilinear", align_corners=False
    )
    return (resized > 0.0).squeeze().cpu().numpy()


def box_from_mask(mask: Any) -> dict[str, float] | None:
    """Box the dominant connected component(s), as the worker does; speckle is ignored."""
    import cv2

    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask.astype("uint8"), connectivity=8
    )
    if count < 2:
        return None
    components = stats[1:]
    areas = components[:, cv2.CC_STAT_AREA]
    kept = components[areas >= BOX_COMPONENT_KEEP_FRACTION * areas.max()]
    height, width = mask.shape
    x1 = kept[:, cv2.CC_STAT_LEFT].min() / width
    y1 = kept[:, cv2.CC_STAT_TOP].min() / height
    x2 = (kept[:, cv2.CC_STAT_LEFT] + kept[:, cv2.CC_STAT_WIDTH]).max() / width
    y2 = (kept[:, cv2.CC_STAT_TOP] + kept[:, cv2.CC_STAT_HEIGHT]).max() / height
    return {"x": float(x1), "y": float(y1), "width": float(x2 - x1), "height": float(y2 - y1)}


def scalar(values: Any, position: int) -> float | None:
    if values is None:
        return None
    try:
        selected = values[position]
    except (IndexError, KeyError, TypeError):
        return None
    reshaped = selected.reshape(-1) if hasattr(selected, "reshape") else selected
    try:
        return float(reshaped[0] if hasattr(reshaped, "__len__") and len(reshaped) else reshaped)
    except (TypeError, ValueError):
        return None


def run_track(args: argparse.Namespace) -> int:
    run_directory = Path(args.run_directory)
    run_directory.mkdir(parents=True, exist_ok=True)
    masks_directory = run_directory / "masks"
    result_path = run_directory / "track_result.json"
    video_path = Path(args.video)
    model_path = Path(args.model)
    prompts = list(args.prompt)
    if len(set(prompts)) != len(prompts) or not prompts:
        raise SystemExit("prompts must be non-empty and distinct")
    start = perf_counter()
    seen_gpu = gpu_guard.legacy_gpu_processes()
    blocking, tolerated, foreign_trackers = guard_gpu(args.allow_gpu_neighbour)
    settings = {
        "device": "cuda:0",
        "dtype": "bfloat16",
        "max_frames": args.max_frames,
        "max_side_length": args.max_side_length,
        "use_square_sizing": True,
        "detection_threshold": DETECTION_THRESHOLD,
        "max_prompt_memory_entries": MAX_PROMPT_MEMORY,
        "max_frame_memory_entries": MAX_FRAME_MEMORY,
        "is_recent_first": IS_RECENT_FIRST,
        "prompt_memory_semantics": "replace",
        "tracker_memory_policy": {"slot_exclusivity": "off", "memory_gate": "off"},
        "chunking": "none; one continuous tracker stream, no corrections",
        "box_derivation": (
            "union of the mask's 8-connected components with area at least "
            f"{BOX_COMPONENT_KEEP_FRACTION:g} of the largest"
        ),
        "prompts": prompts,
        "analysis_fps": args.analysis_fps,
        "source_offset_seconds": args.source_offset_seconds,
        "gpu_guard": {
            "allowed_neighbour_pids": list(args.allow_gpu_neighbour),
            "tolerated_neighbours": tolerated,
            "gpu_processes_before_initialization": seen_gpu,
        },
    }

    def fail(state: str, reason: str) -> int:
        fs_common.write_json(
            result_path,
            {
                "state": state,
                "reason": reason,
                "elapsed_seconds": perf_counter() - start,
                "runtime_settings": settings,
            },
            sort_keys=True,
        )
        print(f"{state}: {reason}", file=sys.stderr)
        return 2

    if not video_path.is_file():
        return fail("blocked", f"proxy does not exist: {video_path}")
    if not model_path.is_file():
        return fail("blocked", f"SAM3 checkpoint is unavailable: {model_path}")
    if blocking:
        return fail("blocked", f"concurrent GPU model process(es): {blocking}")
    if foreign_trackers:
        return fail("blocked", f"Battle tracker process(es) running: {foreign_trackers}")

    try:
        import cv2
        import torch

        # The MuggledSAM package is not installed into its interpreter; Battle's adapter puts
        # the checkout on PYTHONPATH, so this script does the same for itself.
        if str(MUGGLED_SAM_SOURCE) not in sys.path:
            sys.path.insert(0, str(MUGGLED_SAM_SOURCE))
        from muggled_sam.make_sam import make_sam_from_state_dict

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        capture = cv2.VideoCapture(str(video_path))
        ok, first_frame = capture.read()
        if not ok:
            raise RuntimeError(f"could not decode first frame: {video_path}")
        frame_shape = first_frame.shape[:2]

        load_start = perf_counter()
        core = make_sam_from_state_dict(model_path)
        core.to(device="cuda:0", dtype=torch.bfloat16)
        tracking = core.get_tracking_context()
        if not hasattr(tracking, "step_video_masking_multiplex"):
            raise TypeError("checkpoint does not expose SAM3.1 multiplex video tracking")
        detector = core.get_detector_context()
        model_load_seconds = perf_counter() - load_start

        # Frame 0: one detector pass per prompt, top-scored detection kept, absence recorded.
        detector_encoded = detector.encode_image(first_frame, args.max_side_length, True)
        detections: list[dict[str, Any]] = []
        initial: list[tuple[int, str, Any, float]] = []
        for slot, prompt in enumerate(prompts):
            exemplars = detector.encode_exemplars(detector_encoded, text=prompt)
            masks, boxes, scores, presence = detector.generate_detections(
                detector_encoded, exemplars, detection_filter_threshold=DETECTION_THRESHOLD
            )
            record: dict[str, Any] = {
                "slot": slot,
                "prompt": prompt,
                "label": slugify(prompt),
                "object_id": f"sam3-{slot:02d}",
                "presence_score": float(presence.reshape(-1)[0]),
                "detections_above_threshold": int(masks.shape[1]),
                "all_scores_above_threshold": [float(s) for s in scores[0].tolist()],
            }
            if masks.shape[1] == 0:
                record.update({"detected": False, "score": None, "box_xy1xy2_norm": None})
            else:
                best = int(scores[0].argmax())
                box = boxes[0, best].to(torch.float32).cpu().numpy().reshape(-1).tolist()
                record.update(
                    {
                        "detected": True,
                        "score": float(scores[0, best]),
                        "box_xy1xy2_norm": [float(v) for v in box],
                    }
                )
                initial.append((slot, record["label"], masks[:, [best]], float(scores[0, best])))
            detections.append(record)
        settings["frame_zero_detections"] = detections
        detection_seconds = perf_counter() - load_start - model_load_seconds

        masks_directory.mkdir(exist_ok=True)
        masks_written = 0
        processed = 0

        def observe(
            frame_index: int,
            tracked: list[tuple[int, str, Any, float, float | None, float | None]],
            diagnostics: list[dict[str, Any]],
        ) -> dict[str, Any]:
            nonlocal masks_written
            objects = []
            for slot, label, logits, confidence, object_score, iou in tracked:
                mask = binary_mask(logits, frame_shape)
                box = box_from_mask(mask)
                if box is None:
                    continue
                filename = f"{frame_index:06d}_{slot:02d}.png"
                mask_uri = None
                if cv2.imwrite(str(masks_directory / filename), mask.astype("uint8") * 255):
                    masks_written += 1
                    mask_uri = f"masks/{filename}"
                objects.append(
                    {
                        "object_id": f"sam3-{slot:02d}",
                        "label": label,
                        "confidence": max(0.0, min(1.0, confidence)),
                        "box": box,
                        "mask_uri": mask_uri,
                        "mask_area_fraction": float(mask.mean()),
                        "object_score": object_score,
                        "iou_prediction": iou,
                    }
                )
            return {
                "analysis_frame_index": frame_index,
                "source_seconds": args.source_offset_seconds + frame_index / args.analysis_fps,
                "objects": objects,
                "tracker_diagnostics": diagnostics,
            }

        with (run_directory / "observations.jsonl").open("w", encoding="utf-8") as out:
            out.write(
                json.dumps(
                    observe(
                        0,
                        [(slot, label, m, c, None, None) for slot, label, m, c in initial],
                        [],
                    ),
                    sort_keys=True,
                )
                + "\n"
            )
            processed = 1
            if initial:
                tracking_encoded = tracking.encode_image(first_frame, args.max_side_length, True)
                initial_memory = tracking.encode_prompt_memory_from_mask(
                    tracking_encoded, torch.cat([m for _, _, m, _ in initial], dim=1)
                )
                prompt_memories: deque[Any] = deque([initial_memory], maxlen=MAX_PROMPT_MEMORY)
                frame_memories: deque[Any] = deque(maxlen=MAX_FRAME_MEMORY)
            for frame_index in range(1, args.max_frames):
                ok, frame = capture.read()
                if not ok:
                    break
                if not initial:
                    out.write(json.dumps(observe(frame_index, [], []), sort_keys=True) + "\n")
                    processed += 1
                    continue
                encoded = tracking.encode_image(frame, args.max_side_length, True)
                masks, ious, pointers, scores = tracking.step_video_masking_multiplex(
                    encoded,
                    prompt_memories,
                    frame_memories,
                    is_recent_first=IS_RECENT_FIRST,
                    num_multiplex_objects=len(initial),
                )
                active = scores > 0
                if bool(active.any()):
                    frame_memories.append(
                        tracking.encode_frame_memory(encoded, masks, pointers, scores)
                    )
                tracked = [
                    (
                        slot,
                        label,
                        masks[position : position + 1],
                        float(scores[position]),
                        float(scores[position]),
                        scalar(ious, position),
                    )
                    for position, (slot, label, _, _) in enumerate(initial)
                    if bool(active[position])
                ]
                diagnostics = [
                    {
                        "object_id": f"sam3-{slot:02d}",
                        "label": label,
                        "multiplex_slot": slot,
                        "object_score": float(scores[position]),
                        "iou_prediction": scalar(ious, position),
                        "active": bool(active[position]),
                    }
                    for position, (slot, label, _, _) in enumerate(initial)
                ]
                out.write(json.dumps(observe(frame_index, tracked, diagnostics), sort_keys=True))
                out.write("\n")
                processed += 1
                if frame_index % 100 == 0:
                    print(f"frame {frame_index} {perf_counter() - start:.1f}s", flush=True)
        capture.release()
        torch.cuda.synchronize(0)
        fs_common.write_json(
            result_path,
            {
                "state": "succeeded",
                "reason": None,
                "frames_processed": processed,
                "masks_written": masks_written,
                "elapsed_seconds": perf_counter() - start,
                "model_load_seconds": model_load_seconds,
                "frame_zero_detection_seconds": detection_seconds,
                "gpu_peak_vram_bytes": int(torch.cuda.max_memory_allocated(0)),
                "gpu_peak_vram_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
                "torch_version": torch.__version__,
                "cuda_device_name": torch.cuda.get_device_name(0),
                "python": sys.version.split()[0],
                "interpreter": sys.executable,
                "runtime_settings": settings,
            },
            sort_keys=True,
        )
        return 0
    except Exception as error:  # noqa: BLE001 - recorded, not swallowed
        traceback.print_exc()
        return fail("failed", f"{type(error).__name__}: {error}")


# --------------------------------------------------------------------------------------------
# export phase (Battle interpreter: rerun, cv2, numpy, PIL)


git_revision = fs_common.git_revision


def load_observations(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def rgba_png(mask: Any, color: tuple[int, int, int]) -> bytes:
    import io

    import numpy as np
    from PIL import Image

    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[mask] = (*color, 255)
    buffer = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG", compress_level=1)
    return buffer.getvalue()


def read_mask(path: Path) -> Any:
    import cv2

    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return mask > 0


def export_rrd(
    *,
    run_directory: Path,
    observations: list[dict[str, Any]],
    detections: list[dict[str, Any]],
    video_path: Path,
    dimensions: tuple[int, int],
    analysis_fps: float,
    clip_id: str,
    view_id: str,
    run_id: str,
    asset_reference: dict[str, Any],
) -> Path:
    import rerun as rr
    import rerun.blueprint as rrb

    output_path = run_directory / "recording.rrd"
    root = f"world/{clip_id}"
    view_root = f"{root}/views/{view_id}"
    video_asset_path = f"{view_root}/video_asset"
    colors = {d["object_id"]: OBJECT_COLORS[d["slot"] % len(OBJECT_COLORS)] for d in detections}
    labels = {d["object_id"]: d["label"] for d in detections}
    width, height = dimensions

    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial2DView(
                    origin=view_root,
                    contents=[
                        "$origin/**",
                        "- $origin/tracker_diagnostics/**",
                        "- $origin/area/**",
                    ],
                    name=f"{view_id} video, boxes and masks",
                    visual_bounds=rrb.VisualBounds2D(x_range=[0, width], y_range=[0, height]),
                ),
                rrb.Vertical(
                    rrb.TimeSeriesView(
                        origin=f"{view_root}/area",
                        contents="$origin/**",
                        name="Mask area (fraction of frame)",
                    ),
                    rrb.TimeSeriesView(
                        origin=f"{view_root}/tracker_diagnostics/object_score",
                        contents="$origin/**",
                        name="Object score (lost at or below 0)",
                    ),
                    rrb.TimeSeriesView(
                        origin=f"{view_root}/tracker_diagnostics/iou_prediction",
                        contents="$origin/**",
                        name="Predicted mask IoU (self-estimate)",
                    ),
                ),
                column_shares=[2, 1],
            ),
            rrb.Horizontal(
                rrb.TextDocumentView(origin=f"{root}/claim_boundary", name="Claim boundary"),
                rrb.TextDocumentView(origin=f"{root}/frame_counter", name="Frame count"),
                column_shares=[3, 1],
            ),
            row_shares=[4, 1],
        ),
        rrb.TimePanel(
            timeline="analysis_time",
            fps=int(analysis_fps),
            time_selection=rr.encodings.AbsoluteTimeRange(0, 0),
        ),
        auto_layout=False,
        auto_views=False,
    )

    rr.init(f"battle-{clip_id}", recording_id=run_id)
    rr.save(output_path)
    rr.log(
        f"{root}/source/asset_reference",
        rr.TextDocument(json.dumps(asset_reference, indent=2), media_type="application/json"),
        static=True,
    )
    rr.log(
        f"{root}/source/asset_policy",
        rr.TextDocument(
            "The 600-frame proxy is embedded once as an AssetVideo and linked to every analysis "
            "frame. FineBio licence: non-commercial research; do not redistribute this recording.",
            media_type="text/plain",
        ),
        static=True,
    )
    rr.log(
        f"{root}/claim_boundary",
        rr.TextDocument(CLAIM_BOUNDARY, media_type="text/plain"),
        static=True,
    )
    rr.log(
        f"{root}/frame_zero_detections",
        rr.TextDocument(json.dumps(detections, indent=2), media_type="application/json"),
        static=True,
    )
    rr.log(video_asset_path, rr.AssetVideo(path=video_path), static=True)
    rr.log(
        view_root,
        rr.AnnotationContext(
            [
                rr.AnnotationInfo(id=0, label="background", color=(0, 0, 0, 0)),
                *[
                    rr.AnnotationInfo(
                        id=d["slot"] + 1, label=d["label"], color=colors[d["object_id"]]
                    )
                    for d in detections
                ],
            ]
        ),
        static=True,
    )
    for object_id, label in labels.items():
        series = f"{label} ({object_id})"
        rr.log(
            f"{view_root}/area/{object_id}",
            rr.SeriesLines(colors=[colors[object_id]], names=[series]),
            static=True,
        )
        rr.log(
            f"{view_root}/tracker_diagnostics/object_score/{object_id}",
            rr.SeriesLines(colors=[colors[object_id]], names=[series]),
            static=True,
        )
        rr.log(
            f"{view_root}/tracker_diagnostics/iou_prediction/{object_id}",
            rr.SeriesLines(colors=[colors[object_id]], names=[series]),
            static=True,
        )
    rr.log(
        f"{view_root}/tracker_diagnostics/object_score/lost_threshold",
        rr.SeriesLines(colors=[(160, 160, 160)], names=["lost threshold"]),
        static=True,
    )

    total = len(observations)
    blueprint_sent = False
    for observation in observations:
        index = int(observation["analysis_frame_index"])
        seconds = index / analysis_fps
        rr.set_time("analysis_frame", sequence=index)
        rr.set_time("analysis_time", duration=seconds)
        rr.set_time("source_time", duration=float(observation["source_seconds"]))
        rr.log(
            f"{root}/frame_counter",
            rr.TextDocument(
                f"analysis frame **{index}** / {total - 1} ({total} frames)\n\n"
                f"{seconds:.3f} s at {analysis_fps:g} fps; "
                f"source {observation['source_seconds']:.3f} s",
                media_type="text/markdown",
            ),
        )
        rr.log(
            f"{view_root}/video",
            rr.VideoFrameReference(seconds=seconds, video_reference=video_asset_path),
        )
        objects = observation["objects"]
        present = {o["object_id"] for o in objects}
        if objects:
            rr.log(
                f"{view_root}/objects",
                rr.Boxes2D(
                    mins=[[o["box"]["x"] * width, o["box"]["y"] * height] for o in objects],
                    sizes=[
                        [o["box"]["width"] * width, o["box"]["height"] * height] for o in objects
                    ],
                    labels=[f"{o['label']} ({o['object_id']})" for o in objects],
                    colors=[colors[o["object_id"]] for o in objects],
                ),
            )
        else:
            rr.log(f"{view_root}/objects", rr.Clear(recursive=False))
        for object_ in objects:
            if object_["mask_uri"]:
                rr.log(
                    f"{view_root}/masks/{object_['object_id']}",
                    rr.EncodedImage(
                        contents=rgba_png(
                            read_mask(run_directory / object_["mask_uri"]),
                            colors[object_["object_id"]],
                        ),
                        media_type="image/png",
                        opacity=0.45,
                        draw_order=1.0,
                    ),
                )
            rr.log(
                f"{view_root}/area/{object_['object_id']}",
                rr.Scalars([object_["mask_area_fraction"]]),
            )
        for object_id in labels:
            if object_id not in present:
                rr.log(f"{view_root}/masks/{object_id}", rr.Clear(recursive=False))
                rr.log(f"{view_root}/area/{object_id}", rr.Scalars([0.0]))
        diagnostics = observation.get("tracker_diagnostics") or []
        if diagnostics:
            rr.log(
                f"{view_root}/tracker_diagnostics/object_score/lost_threshold", rr.Scalars([0.0])
            )
        for diagnostic in diagnostics:
            rr.log(
                f"{view_root}/tracker_diagnostics/object_score/{diagnostic['object_id']}",
                rr.Scalars([diagnostic["object_score"]]),
            )
            if diagnostic.get("iou_prediction") is not None:
                rr.log(
                    f"{view_root}/tracker_diagnostics/iou_prediction/{diagnostic['object_id']}",
                    rr.Scalars([diagnostic["iou_prediction"]]),
                )
        if not blueprint_sent:
            rr.send_blueprint(blueprint)
            blueprint_sent = True
    if not blueprint_sent:
        rr.send_blueprint(blueprint)
    rr.disconnect()
    return output_path


def render_contact_sheet(
    *,
    run_directory: Path,
    video_path: Path,
    observations: list[dict[str, Any]],
    detections: list[dict[str, Any]],
    frame_indices: list[int],
    columns: int = 3,
) -> Path:
    import cv2
    import numpy as np

    colors = {d["object_id"]: OBJECT_COLORS[d["slot"] % len(OBJECT_COLORS)] for d in detections}
    by_index = {int(o["analysis_frame_index"]): o for o in observations}
    capture = cv2.VideoCapture(str(video_path))
    frames: dict[int, Any] = {}
    wanted = set(frame_indices)
    index = 0
    while wanted:
        ok, frame = capture.read()
        if not ok:
            break
        if index in wanted:
            frames[index] = frame
            wanted.discard(index)
        index += 1
    capture.release()
    tiles = []
    for frame_index in frame_indices:
        frame = frames.get(frame_index)
        if frame is None:
            continue
        tile = frame.copy()
        observation = by_index.get(frame_index, {"objects": []})
        height, width = tile.shape[:2]
        for object_ in observation["objects"]:
            color_rgb = colors[object_["object_id"]]
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
            if object_["mask_uri"]:
                mask = read_mask(run_directory / object_["mask_uri"])
                overlay = tile.copy()
                overlay[mask] = color_bgr
                tile = cv2.addWeighted(overlay, 0.45, tile, 0.55, 0.0)
            box = object_["box"]
            x1, y1 = int(box["x"] * width), int(box["y"] * height)
            x2, y2 = (
                int((box["x"] + box["width"]) * width),
                int((box["y"] + box["height"]) * height),
            )
            cv2.rectangle(tile, (x1, y1), (x2, y2), color_bgr, 3)
            score = object_.get("object_score")
            caption = object_["label"] + ("" if score is None else f" s={score:.2f}")
            cv2.putText(
                tile,
                caption,
                (x1 + 4, max(y1 - 8, 24)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                tile,
                caption,
                (x1 + 4, max(y1 - 8, 24)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color_bgr,
                2,
                cv2.LINE_AA,
            )
        header = (
            f"frame {frame_index}  source {observation.get('source_seconds', 0.0):.2f} s  "
            f"objects {len(observation['objects'])}"
        )
        cv2.rectangle(tile, (0, 0), (width, 44), (0, 0, 0), -1)
        cv2.putText(
            tile, header, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA
        )
        tiles.append(cv2.resize(tile, (width // 2, height // 2), interpolation=cv2.INTER_AREA))
    if not tiles:
        raise RuntimeError("no frames decoded for the contact sheet")
    blank = np.zeros_like(tiles[0])
    while len(tiles) % columns:
        tiles.append(blank)
    rows = [np.hstack(tiles[i : i + columns]) for i in range(0, len(tiles), columns)]
    sheet = np.vstack(rows)
    legend = np.zeros((40, sheet.shape[1], 3), dtype=np.uint8)
    x = 10
    for detection in detections:
        color_rgb = colors[detection["object_id"]]
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        text = f"{detection['prompt']}: " + (
            f"frame-0 score {detection['score']:.2f}"
            if detection["detected"]
            else "no detection at frame 0"
        )
        cv2.putText(legend, text, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2, cv2.LINE_AA)
        x += 24 + int(cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0])
    output_path = run_directory / "contact_sheet.png"
    cv2.imwrite(str(output_path), np.vstack([sheet, legend]))
    return output_path


def run_export(args: argparse.Namespace) -> int:
    import cv2

    run_directory = Path(args.run_directory)
    track_result = json.loads((run_directory / "track_result.json").read_text(encoding="utf-8"))
    if track_result.get("state") != "succeeded":
        raise SystemExit(f"track phase did not succeed: {track_result.get('reason')}")
    settings = track_result["runtime_settings"]
    detections = settings["frame_zero_detections"]
    observations = load_observations(run_directory / "observations.jsonl")
    video_path = Path(args.video)
    source_path = Path(args.source_video) if args.source_video else None
    capture = cv2.VideoCapture(str(video_path))
    dimensions = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    proxy_fps = float(capture.get(cv2.CAP_PROP_FPS))
    proxy_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    analysis_fps = float(settings["analysis_fps"])
    if proxy_frames != len(observations):
        raise SystemExit(
            f"proxy has {proxy_frames} frames but {len(observations)} observations were written; "
            "the AssetVideo must cover exactly the tracked frames"
        )

    asset_reference = {
        "proxy": {
            "uri": str(video_path),
            "sha256": fs_common.sha256_file(video_path),
            "dimensions": dimensions,
            "fps": proxy_fps,
            "frame_count": proxy_frames,
        },
        "source": (
            {
                "uri": str(source_path),
                "sha256": fs_common.sha256_file(source_path),
                "interval_seconds": [
                    args.source_offset_seconds,
                    args.source_offset_seconds + proxy_frames / analysis_fps,
                ],
            }
            if source_path is not None
            else None
        ),
        "source_name": "FineBio (Yagi et al., IJCV 2025)",
        "source_license": "FineBio licence agreement, non-commercial research",
    }
    rrd_path = export_rrd(
        run_directory=run_directory,
        observations=observations,
        detections=detections,
        video_path=video_path,
        dimensions=dimensions,
        analysis_fps=analysis_fps,
        clip_id=args.clip_id,
        view_id=args.view_id,
        run_id=args.run_id or run_directory.name,
        asset_reference=asset_reference,
    )
    last = len(observations) - 1
    frame_indices = [round(i * last / 5) for i in range(6)]
    sheet_path = render_contact_sheet(
        run_directory=run_directory,
        video_path=video_path,
        observations=observations,
        detections=detections,
        frame_indices=frame_indices,
    )

    per_object: dict[str, dict[str, Any]] = {}
    for detection in detections:
        object_id = detection["object_id"]
        active_frames = [
            o["analysis_frame_index"]
            for o in observations
            if any(x["object_id"] == object_id for x in o["objects"])
        ]
        per_object[object_id] = {
            "prompt": detection["prompt"],
            "label": detection["label"],
            "frame_zero_detected": detection["detected"],
            "frame_zero_score": detection["score"],
            "frames_with_output": len(active_frames),
            "first_frame_without_output": next(
                (i for i in range(len(observations)) if i not in set(active_frames)), None
            )
            if detection["detected"]
            else None,
        }
    model_path = Path(args.model)
    manifest = {
        "run_id": args.run_id or run_directory.name,
        "kind": "finebio_sam3_zero_shot_smoke",
        "claim_boundary": CLAIM_BOUNDARY,
        "inputs": asset_reference,
        "prompts": detections,
        "per_object_summary": per_object,
        "model": {
            "weights": str(model_path),
            "weights_sha256": fs_common.sha256_file(model_path),
            "weights_size_bytes": model_path.stat().st_size,
            "muggled_sam": {"path": str(MUGGLED_SAM_SOURCE), **git_revision(MUGGLED_SAM_SOURCE)},
            "torch_version": track_result.get("torch_version"),
            "cuda_device_name": track_result.get("cuda_device_name"),
            "interpreter": track_result.get("interpreter"),
        },
        "runtime": {
            "elapsed_seconds": track_result["elapsed_seconds"],
            "model_load_seconds": track_result["model_load_seconds"],
            "frame_zero_detection_seconds": track_result["frame_zero_detection_seconds"],
            "gpu_peak_vram_bytes": track_result["gpu_peak_vram_bytes"],
            "gpu_peak_vram_reserved_bytes": track_result["gpu_peak_vram_reserved_bytes"],
            "frames_processed": track_result["frames_processed"],
            "masks_written": track_result["masks_written"],
            "settings": {k: v for k, v in settings.items() if k != "frame_zero_detections"},
        },
        "battle": git_revision(Path(__file__).resolve().parents[1]),
        "host": platform.node(),
        "outputs": {
            "recording_rrd": rrd_path.name,
            "contact_sheet": sheet_path.name,
            "contact_sheet_frames": frame_indices,
            "observations": "observations.jsonl",
            "masks_directory": "masks/",
        },
    }
    fs_common.write_json(run_directory / "manifest.json", manifest, sort_keys=True)
    print(
        json.dumps(
            {"rrd": str(rrd_path), "contact_sheet": str(sheet_path), "per_object": per_object},
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="phase", required=True)
    track = sub.add_parser("track", help="SAM3 detection + tracking (MuggledSAM interpreter)")
    track.add_argument("--run-directory", required=True)
    track.add_argument("--video", required=True, help="600-frame 30 fps proxy")
    track.add_argument("--model", default=str(DEFAULT_MODEL))
    track.add_argument("--prompt", action="append", required=True, help="text prompt (repeatable)")
    track.add_argument("--max-frames", type=int, default=600)
    track.add_argument("--max-side-length", type=int, default=1280)
    track.add_argument("--analysis-fps", type=float, default=30.0)
    track.add_argument("--source-offset-seconds", type=float, default=0.0)
    track.add_argument(
        "--allow-gpu-neighbour", type=int, action="append", default=[], metavar="PID"
    )
    track.set_defaults(func=run_track)
    export = sub.add_parser("export", help="RRD, contact sheet and manifest (Battle interpreter)")
    export.add_argument("--run-directory", required=True)
    export.add_argument("--video", required=True)
    export.add_argument("--source-video")
    export.add_argument("--source-offset-seconds", type=float, default=0.0)
    export.add_argument("--model", default=str(DEFAULT_MODEL))
    export.add_argument("--clip-id", default="finebio-p03-01-01-pipetting-smoke")
    export.add_argument("--view-id", default="fpv-p03-01-01")
    export.add_argument("--run-id")
    export.set_defaults(func=run_export)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
