"""FineBio's shipped object detector on the SAM3 smoke clip, visualised in Rerun beside SAM3.

Superseded Sep 24 by ``battle-finebio-detect`` (``src/battle/finebio_detect.py``: raw frames,
six views, CPU or CUDA venv, strides + interpolation); kept because the Sep 21 record cites it.

Runs the authors' MMDetection DINO (or Deformable DETR) checkpoint, fine-tuned on FineBio's
35 wet-lab classes, over the 600-frame proxy that ``scripts/finebio_sam3_smoke.py`` tracked,
and writes one RRD carrying the video, the detector's labelled boxes and the SAM3 smoke's
masks and boxes so the two methods sit side by side.

Two phases, two interpreters, because the detector environment (torch 2.1 CPU, mmcv 2.1,
mmdet 3.3; see ``scripts/install_finebio_detector.sh``) has no Rerun and the Battle
environment has no MMDetection:

    CUDA_VISIBLE_DEVICES="" /home/nick/src/finebio-detector/.venv/bin/python \
        scripts/finebio_dino_detect.py detect ...
    uv run python scripts/finebio_dino_detect.py export ...

``detect`` uses ``mmdet.apis.init_detector`` / ``inference_detector`` on the frames selected
by ``--frame-stride`` and ``--include-frame`` and writes every detection with score >= 0.05 to
``detections.jsonl`` (boxes in proxy pixels) plus ``detect_result.json``. It runs on the CPU by
default; ``--device cuda:0`` is refused unless no Battle tracker process exists, no other model
process holds the GPU and the newest queue log ends with ``queue_end``. ``export`` reads those
files and the SAM3 smoke's ``observations.jsonl`` + ``masks/`` and writes ``recording.rrd``, a
six-frame contact sheet, ``class_counts.md`` and ``manifest.json``.

This is a qualitative first run on one clip. The FineBio COCO annotations are not on this
machine, so nothing is scored; no accuracy claim is made for either method.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

try:
    from battle import fs_common, gpu_guard
except ImportError:
    # Under the MMDetection interpreter the battle package is not installed: import the
    # stdlib-only helper modules by path, as the workers do.
    sys.path.append(str(Path(__file__).resolve().parents[1] / "src" / "battle"))
    import fs_common  # type: ignore[no-redef]
    import gpu_guard  # type: ignore[no-redef]

DETECTOR_DIR = Path("/home/nick/src/finebio-detector")
MODELS: dict[str, dict[str, str]] = {
    "dino": {
        "config": "mmdetection/configs/dino/dino-4scale_r50_8xb2-12e_finebio.py",
        "weights": "checkpoints/dino.pth",
        "description": "DINO 4-scale R50, 12 epochs, FineBio fine-tune (authors' checkpoint)",
    },
    "deformable-detr": {
        "config": (
            "mmdetection/configs/deformable_detr/"
            "deformable-detr-refine-twostage_r50_16xb2-50e_finebio.py"
        ),
        "weights": "checkpoints/deformable-detr.pth",
        "description": (
            "Deformable DETR two-stage + refine R50, 50 epochs, FineBio fine-tune "
            "(authors' checkpoint)"
        ),
    },
}
# The 35 FineBio object classes in checkpoint order (object_detection configs, aistairc/FineBio).
FINEBIO_CLASSES = (
    "left_hand",
    "right_hand",
    "blue_pipette",
    "yellow_pipette",
    "red_pipette",
    "8_channel_pipette",
    "blue_tip",
    "yellow_tip",
    "red_tip",
    "8_channel_tip",
    "blue_tip_rack",
    "yellow_tip_rack",
    "red_tip_rack",
    "8_channel_tip_rack",
    "50ml_tube",
    "15ml_tube",
    "micro_tube",
    "8_tube_stripes",
    "8_tube_stripes_lid",
    "50ml_tube_rack",
    "15ml_tube_rack",
    "micro_tube_rack",
    "8_tube_stripes_rack",
    "8_tube_stripes_rack_lid",
    "cell_culture_plate",
    "cell_culture_plate_lid",
    "trash_can",
    "centrifuge",
    "vortex_mixer",
    "magnetic_rack",
    "pcr_machine",
    "tube_with_spin_column",
    "spin_column",
    "tube_without_lid",
    "pen",
)
# Colour per class family (RGB), so 35 classes stay readable on a contact sheet.
FAMILY_COLORS: dict[str, tuple[int, int, int]] = {
    "hand": (255, 200, 40),
    "pipette": (255, 80, 80),
    "tip": (255, 130, 220),
    "tip_rack": (170, 100, 255),
    "tube": (60, 220, 220),
    "tube_rack": (70, 140, 255),
    "plate": (80, 230, 100),
    "instrument": (235, 235, 235),
    "other": (160, 160, 160),
}
RECORD_THRESHOLD = 0.05
DISPLAY_THRESHOLD = 0.30
SAM3_OBJECT_COLORS = (
    (255, 85, 85),
    (255, 210, 65),
    (80, 180, 255),
    (180, 115, 255),
    (70, 210, 165),
)
# Processes whose presence means another tracker owns the GPU; matched against `pgrep -af`.
TRACKER_PROCESS_PATTERN = "muggled_worker|battle-muggled-smoke|four_part_video_worker"
CLAIM_BOUNDARY = (
    "Qualitative first run of FineBio's shipped MMDetection detector (the authors' fine-tuned "
    "checkpoint, 35 classes) on one 20-second FineBio first-person clip, shown beside the "
    "Sep 21 SAM3 zero-shot smoke on the same frames. Scores are the detector's own softmax "
    "outputs; the FineBio annotations are not on this machine, so nothing is scored against "
    "ground truth. Every statement about this run is a description of what the boxes look "
    "like on these frames, not an accuracy, coverage or generalisation claim for either method."
)


def class_family(name: str) -> str:
    if name.endswith("_hand"):
        return "hand"
    if name.endswith("_pipette"):
        return "pipette"
    if name.endswith("_tip"):
        return "tip"
    if name.endswith("_tip_rack"):
        return "tip_rack"
    if name in {"cell_culture_plate", "cell_culture_plate_lid"}:
        return "plate"
    if name in {"centrifuge", "vortex_mixer", "pcr_machine"}:
        return "instrument"
    if name.endswith("_rack") or name.endswith("_rack_lid"):
        return "tube_rack"
    if "tube" in name or "spin_column" in name:
        return "tube"
    return "other"


def class_color(name: str) -> tuple[int, int, int]:
    return FAMILY_COLORS[class_family(name)]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def output_names(model_name: str) -> tuple[str, str]:
    suffix = "" if model_name == "dino" else f".{model_name}"
    return f"detections{suffix}.jsonl", f"detect_result{suffix}.json"


# --------------------------------------------------------------------------------------------
# GPU coordination (the SAM3 smoke's rule through `battle.gpu_guard`: refuse the GPU beside any
# model-like GPU process not allowed by PID, and refuse while a Battle tracker process exists
# at all; plus the other session's queue log must have ended).


def tracker_processes() -> list[str]:
    completed = subprocess.run(
        ["pgrep", "-af", TRACKER_PROCESS_PATTERN], check=False, capture_output=True, text=True
    )
    own = str(Path(__file__).name)
    return [
        line[:200]
        for line in completed.stdout.splitlines()
        if own not in line and "pgrep" not in line and line.strip()
    ]


def newest_queue_log(repo_root: Path) -> Path | None:
    logs = sorted(repo_root.glob("runs/*/queue.log"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def queue_state(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {"path": None if path is None else str(path), "last_event": None}
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    last: dict[str, Any] = {}
    if lines:
        try:
            last = json.loads(lines[-1])
        except json.JSONDecodeError:
            last = {"raw": lines[-1][:200]}
    return {
        "path": str(path),
        "last_event": last.get("event"),
        "last_job": last.get("job") or last.get("name"),
        "last_time": last.get("time"),
    }


def gpu_coordination(allowed_pids: list[int], queue_log: Path | None) -> dict[str, Any]:
    seen = gpu_guard.legacy_gpu_processes()
    blocking, tolerated = gpu_guard.partition_neighbours_strict(seen, allowed_pids)
    queue = queue_state(queue_log)
    trackers = tracker_processes()
    return {
        "checked_at": subprocess.run(
            ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], check=False, capture_output=True, text=True
        ).stdout.strip(),
        "gpu_processes": seen,
        "blocking_gpu_processes": blocking,
        "tolerated_gpu_processes": tolerated,
        "tracker_processes": trackers,
        "queue_log": queue,
        "gpu_free_for_us": not blocking and not trackers and queue["last_event"] == "queue_end",
    }


# --------------------------------------------------------------------------------------------
# detect phase (detector interpreter: torch CPU, mmcv, mmdet)


def selected_frames(stride: int, include: list[int], max_frames: int) -> list[int]:
    frames = set(range(0, max_frames, stride)) if stride > 0 else set()
    frames.update(f for f in include if 0 <= f < max_frames)
    return sorted(frames)


def run_detect(args: argparse.Namespace) -> int:
    run_directory = Path(args.run_directory)
    run_directory.mkdir(parents=True, exist_ok=True)
    detections_name, result_name = output_names(args.model)
    result_path = run_directory / result_name
    video_path = Path(args.video)
    config_path = DETECTOR_DIR / MODELS[args.model]["config"]
    weights_path = DETECTOR_DIR / MODELS[args.model]["weights"]
    frames = selected_frames(args.frame_stride, args.include_frame, args.max_frames)
    start = perf_counter()
    repo_root = Path(__file__).resolve().parents[1]
    queue_log = Path(args.queue_log) if args.queue_log else newest_queue_log(repo_root)
    coordination = gpu_coordination(args.allow_gpu_neighbour, queue_log)
    settings = {
        "model": args.model,
        "model_description": MODELS[args.model]["description"],
        "config": str(config_path),
        "weights": str(weights_path),
        "device": args.device,
        "torch_threads": args.threads,
        "frame_stride": args.frame_stride,
        "included_frames": sorted(set(args.include_frame)),
        "max_frames": args.max_frames,
        "selected_frame_count": len(frames),
        "record_threshold": args.record_threshold,
        "analysis_fps": args.analysis_fps,
        "source_offset_seconds": args.source_offset_seconds,
        "test_pipeline": None,
        "gpu_coordination": coordination,
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
    if not config_path.is_file() or not weights_path.is_file():
        return fail(
            "blocked",
            f"detector files missing ({config_path}, {weights_path}); "
            "run scripts/install_finebio_detector.sh",
        )
    if args.device != "cpu" and not coordination["gpu_free_for_us"]:
        return fail(
            "blocked",
            "GPU requested but not free: "
            f"trackers={coordination['tracker_processes']} "
            f"blocking={coordination['blocking_gpu_processes']} "
            f"queue={coordination['queue_log']}",
        )

    try:
        import cv2
        import mmcv
        import mmdet
        import mmengine
        import torch
        from mmdet.apis import inference_detector, init_detector

        if args.threads:
            torch.set_num_threads(args.threads)
        settings["torch_threads"] = torch.get_num_threads()
        if args.device == "cpu" and torch.cuda.is_available():
            raise RuntimeError("CPU run requested but CUDA is visible; set CUDA_VISIBLE_DEVICES=''")

        load_start = perf_counter()
        model = init_detector(str(config_path), str(weights_path), device=args.device)
        model_load_seconds = perf_counter() - load_start
        classes = tuple(model.dataset_meta["classes"])
        if classes != FINEBIO_CLASSES:
            raise RuntimeError(f"checkpoint classes differ from the FineBio list: {classes}")
        pipeline = model.cfg.test_dataloader.dataset.pipeline
        settings["test_pipeline"] = [
            {k: (list(v) if isinstance(v, tuple) else v) for k, v in step.items()}
            for step in pipeline
        ]
        settings["num_queries"] = int(getattr(model, "num_queries", 0) or 0)
        settings["max_per_img"] = int(model.cfg.model.test_cfg.get("max_per_img", 0))

        capture = cv2.VideoCapture(str(video_path))
        wanted = set(frames)
        per_frame_seconds: list[float] = []
        processed: list[int] = []
        detection_total = 0
        index = 0
        with (run_directory / detections_name).open("w", encoding="utf-8") as out:
            while wanted:
                ok, frame = capture.read()
                if not ok:
                    break
                if index in wanted:
                    wanted.discard(index)
                    infer_start = perf_counter()
                    with torch.inference_mode():
                        result = inference_detector(model, frame)
                    infer_seconds = perf_counter() - infer_start
                    instances = result.pred_instances
                    scores = instances.scores.cpu().numpy()
                    labels = instances.labels.cpu().numpy()
                    boxes = instances.bboxes.cpu().numpy()
                    keep = scores >= args.record_threshold
                    order = scores[keep].argsort()[::-1]
                    detections = [
                        {
                            "class_id": int(labels[keep][i]),
                            "class": classes[int(labels[keep][i])],
                            "score": float(scores[keep][i]),
                            "box_xyxy_px": [float(v) for v in boxes[keep][i]],
                        }
                        for i in order
                    ]
                    detection_total += len(detections)
                    out.write(
                        json.dumps(
                            {
                                "analysis_frame_index": index,
                                "source_seconds": args.source_offset_seconds
                                + index / args.analysis_fps,
                                "inference_seconds": infer_seconds,
                                "image_hw": [int(frame.shape[0]), int(frame.shape[1])],
                                "detections": detections,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    per_frame_seconds.append(infer_seconds)
                    processed.append(index)
                    if len(processed) % 10 == 0:
                        print(
                            f"frame {index} ({len(processed)}/{len(frames)}) "
                            f"{infer_seconds:.2f}s/frame, {perf_counter() - start:.1f}s total",
                            flush=True,
                        )
                index += 1
        capture.release()
        if wanted:
            raise RuntimeError(f"proxy ended before frames {sorted(wanted)[:5]}...")
        fs_common.write_json(
            result_path,
            {
                "state": "succeeded",
                "reason": None,
                "frames_processed": processed,
                "frames_processed_count": len(processed),
                "detections_recorded": detection_total,
                "elapsed_seconds": perf_counter() - start,
                "model_load_seconds": model_load_seconds,
                "per_frame_seconds": per_frame_seconds,
                "per_frame_seconds_summary": {
                    "min": min(per_frame_seconds),
                    "median": sorted(per_frame_seconds)[len(per_frame_seconds) // 2],
                    "mean": sum(per_frame_seconds) / len(per_frame_seconds),
                    "max": max(per_frame_seconds),
                },
                "versions": {
                    "python": sys.version.split()[0],
                    "interpreter": sys.executable,
                    "torch": torch.__version__,
                    "mmcv": mmcv.__version__,
                    "mmengine": mmengine.__version__,
                    "mmdet": mmdet.__version__,
                    "cuda_available": bool(torch.cuda.is_available()),
                },
                "classes": list(classes),
                "detections_file": detections_name,
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


def load_detector_runs(run_directory: Path) -> dict[str, dict[str, Any]]:
    """Return {model_name: {"result": ..., "frames": {index: record}}} for every finished model."""
    runs: dict[str, dict[str, Any]] = {}
    for model_name in MODELS:
        detections_name, result_name = output_names(model_name)
        result_path = run_directory / result_name
        if not result_path.is_file():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("state") != "succeeded":
            print(f"skipping {model_name}: {result.get('state')} {result.get('reason')}")
            continue
        records = load_jsonl(run_directory / detections_name)
        runs[model_name] = {
            "result": result,
            "frames": {int(r["analysis_frame_index"]): r for r in records},
        }
    return runs


def class_count_table(
    frames: dict[int, dict[str, Any]], threshold: float
) -> dict[str, dict[str, Any]]:
    frames_with: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    scores: defaultdict[str, list[float]] = defaultdict(list)
    for record in frames.values():
        seen: set[str] = set()
        for det in record["detections"]:
            if det["score"] < threshold:
                continue
            totals[det["class"]] += 1
            scores[det["class"]].append(det["score"])
            seen.add(det["class"])
        for name in seen:
            frames_with[name] += 1
    table = {}
    for name in FINEBIO_CLASSES:
        values = scores.get(name, [])
        table[name] = {
            "family": class_family(name),
            "frames_with_detection": frames_with.get(name, 0),
            "total_detections": totals.get(name, 0),
            "max_score": max(values) if values else None,
            "mean_score": sum(values) / len(values) if values else None,
        }
    return table


def class_count_markdown(
    tables: dict[str, dict[str, dict[str, Any]]], frame_counts: dict[str, int], threshold: float
) -> str:
    models = list(tables)
    header = "| class | family |" + "".join(
        f" {m}: frames with det (of {frame_counts[m]}) | {m}: total dets | {m}: max score |"
        for m in models
    )
    rule = "| --- | --- |" + " --- | --- | --- |" * len(models)
    lines = [
        f"Per-class detections at score >= {threshold:.2f} over the processed frames.",
        "",
        header,
        rule,
    ]
    for name in FINEBIO_CLASSES:
        if all(tables[m][name]["total_detections"] == 0 for m in models):
            continue
        cells = []
        for m in models:
            row = tables[m][name]
            best = "-" if row["max_score"] is None else f"{row['max_score']:.2f}"
            cells.append(f" {row['frames_with_detection']} | {row['total_detections']} | {best} |")
        lines.append(f"| `{name}` | {class_family(name)} |" + "".join(cells))
    absent = [
        name
        for name in FINEBIO_CLASSES
        if all(tables[m][name]["total_detections"] == 0 for m in models)
    ]
    lines += ["", f"Never detected at >= {threshold:.2f}: " + ", ".join(f"`{n}`" for n in absent)]
    return "\n".join(lines) + "\n"


def export_rrd(
    *,
    run_directory: Path,
    detector_runs: dict[str, dict[str, Any]],
    sam3_run_directory: Path,
    sam3_observations: list[dict[str, Any]],
    sam3_detections: list[dict[str, Any]],
    video_path: Path,
    dimensions: tuple[int, int],
    frame_count: int,
    analysis_fps: float,
    source_offset_seconds: float,
    display_threshold: float,
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
    width, height = dimensions
    sam3_colors = {
        d["object_id"]: SAM3_OBJECT_COLORS[d["slot"] % len(SAM3_OBJECT_COLORS)]
        for d in sam3_detections
    }
    sam3_by_index = {int(o["analysis_frame_index"]): o for o in sam3_observations}
    # Classes that appear at least once above the display threshold get a count series.
    series_classes: dict[str, list[str]] = {}
    for model_name, run in detector_runs.items():
        present = {
            det["class"]
            for record in run["frames"].values()
            for det in record["detections"]
            if det["score"] >= display_threshold
        }
        series_classes[model_name] = [c for c in FINEBIO_CLASSES if c in present]

    def spatial(name: str, contents: list[str]) -> Any:
        return rrb.Spatial2DView(
            origin=view_root,
            contents=contents,
            name=name,
            visual_bounds=rrb.VisualBounds2D(x_range=[0, width], y_range=[0, height]),
        )

    method_views = [
        spatial(
            f"FineBio {model_name} (supervised, 35 classes), score >= {display_threshold:.2f}",
            ["$origin/video", "$origin/video_asset", f"$origin/detector/{model_name}/boxes"],
        )
        for model_name in detector_runs
    ]
    method_views.append(
        spatial(
            "SAM3 zero-shot text prompts (Sep 21 smoke): masks and boxes",
            [
                "$origin/video",
                "$origin/video_asset",
                "$origin/sam3/objects",
                "$origin/sam3/masks/**",
            ],
        )
    )
    count_views = [
        rrb.TimeSeriesView(
            origin=f"{view_root}/detector/{model_name}/counts",
            contents="$origin/**",
            name=f"{model_name}: detections per class, score >= {display_threshold:.2f}",
        )
        for model_name in detector_runs
    ]
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(*method_views),
            rrb.Horizontal(
                rrb.Vertical(*count_views),
                rrb.Vertical(
                    rrb.TextDocumentView(origin=f"{root}/claim_boundary", name="Claim boundary"),
                    rrb.TextDocumentView(origin=f"{root}/frame_counter", name="Frame count"),
                ),
                column_shares=[3, 1],
            ),
            row_shares=[3, 1],
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
    rr.log(video_asset_path, rr.AssetVideo(path=video_path), static=True)
    rr.log(
        f"{view_root}/detector",
        rr.AnnotationContext(
            [
                rr.AnnotationInfo(id=0, label="background", color=(0, 0, 0, 0)),
                *[
                    rr.AnnotationInfo(id=i + 1, label=name, color=class_color(name))
                    for i, name in enumerate(FINEBIO_CLASSES)
                ],
            ]
        ),
        static=True,
    )
    rr.log(
        f"{view_root}/sam3",
        rr.AnnotationContext(
            [
                rr.AnnotationInfo(id=0, label="background", color=(0, 0, 0, 0)),
                *[
                    rr.AnnotationInfo(
                        id=d["slot"] + 1, label=d["label"], color=sam3_colors[d["object_id"]]
                    )
                    for d in sam3_detections
                ],
            ]
        ),
        static=True,
    )
    rr.log(
        f"{view_root}/sam3/frame_zero_detections",
        rr.TextDocument(json.dumps(sam3_detections, indent=2), media_type="application/json"),
        static=True,
    )
    for model_name, run in detector_runs.items():
        rr.log(
            f"{view_root}/detector/{model_name}/result",
            rr.TextDocument(
                json.dumps(
                    {k: v for k, v in run["result"].items() if k != "per_frame_seconds"},
                    indent=2,
                ),
                media_type="application/json",
            ),
            static=True,
        )
        rr.log(
            f"{view_root}/detector/{model_name}/counts/all_classes",
            rr.SeriesLines(colors=[(255, 255, 255)], names=["all classes"], widths=[2.0]),
            static=True,
        )
        for name in series_classes[model_name]:
            rr.log(
                f"{view_root}/detector/{model_name}/counts/{name}",
                rr.SeriesLines(colors=[class_color(name)], names=[name]),
                static=True,
            )

    blueprint_sent = False
    for index in range(frame_count):
        seconds = index / analysis_fps
        source_seconds = source_offset_seconds + seconds
        rr.set_time("analysis_frame", sequence=index)
        rr.set_time("analysis_time", duration=seconds)
        rr.set_time("source_time", duration=source_seconds)
        processed_here = [m for m, run in detector_runs.items() if index in run["frames"]]
        rr.log(
            f"{root}/frame_counter",
            rr.TextDocument(
                f"analysis frame **{index}** / {frame_count - 1} ({frame_count} frames)\n\n"
                f"{seconds:.3f} s at {analysis_fps:g} fps; source {source_seconds:.3f} s\n\n"
                f"detector frames here: {', '.join(processed_here) or 'none (boxes held)'}",
                media_type="text/markdown",
            ),
        )
        rr.log(
            f"{view_root}/video",
            rr.VideoFrameReference(seconds=seconds, video_reference=video_asset_path),
        )
        for model_name, run in detector_runs.items():
            record = run["frames"].get(index)
            if record is None:
                continue
            shown = [d for d in record["detections"] if d["score"] >= display_threshold]
            boxes_path = f"{view_root}/detector/{model_name}/boxes"
            if shown:
                rr.log(
                    boxes_path,
                    rr.Boxes2D(
                        mins=[[d["box_xyxy_px"][0], d["box_xyxy_px"][1]] for d in shown],
                        sizes=[
                            [
                                d["box_xyxy_px"][2] - d["box_xyxy_px"][0],
                                d["box_xyxy_px"][3] - d["box_xyxy_px"][1],
                            ]
                            for d in shown
                        ],
                        labels=[f"{d['class']} {d['score']:.2f}" for d in shown],
                        class_ids=[d["class_id"] + 1 for d in shown],
                        colors=[class_color(d["class"]) for d in shown],
                    ),
                )
            else:
                rr.log(boxes_path, rr.Clear(recursive=False))
            counts = Counter(d["class"] for d in shown)
            rr.log(
                f"{view_root}/detector/{model_name}/counts/all_classes", rr.Scalars([len(shown)])
            )
            for name in series_classes[model_name]:
                rr.log(
                    f"{view_root}/detector/{model_name}/counts/{name}",
                    rr.Scalars([counts.get(name, 0)]),
                )
        observation = sam3_by_index.get(index)
        if observation is not None:
            objects = observation["objects"]
            if objects:
                rr.log(
                    f"{view_root}/sam3/objects",
                    rr.Boxes2D(
                        mins=[[o["box"]["x"] * width, o["box"]["y"] * height] for o in objects],
                        sizes=[
                            [o["box"]["width"] * width, o["box"]["height"] * height]
                            for o in objects
                        ],
                        labels=[f"{o['label']} ({o['object_id']})" for o in objects],
                        colors=[sam3_colors[o["object_id"]] for o in objects],
                    ),
                )
            else:
                rr.log(f"{view_root}/sam3/objects", rr.Clear(recursive=False))
            present = {o["object_id"] for o in objects}
            for object_ in objects:
                if object_["mask_uri"]:
                    rr.log(
                        f"{view_root}/sam3/masks/{object_['object_id']}",
                        rr.EncodedImage(
                            contents=rgba_png(
                                read_mask(sam3_run_directory / object_["mask_uri"]),
                                sam3_colors[object_["object_id"]],
                            ),
                            media_type="image/png",
                            opacity=0.45,
                            draw_order=1.0,
                        ),
                    )
            for object_id in sam3_colors:
                if object_id not in present:
                    rr.log(f"{view_root}/sam3/masks/{object_id}", rr.Clear(recursive=False))
        if not blueprint_sent:
            rr.send_blueprint(blueprint)
            blueprint_sent = True
    if not blueprint_sent:
        rr.send_blueprint(blueprint)
    rr.disconnect()
    return output_path


def render_contact_sheet(
    *,
    output_path: Path,
    video_path: Path,
    frames: dict[int, dict[str, Any]],
    frame_indices: list[int],
    display_threshold: float,
    title: str,
    columns: int = 3,
) -> Path:
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(str(video_path))
    images: dict[int, Any] = {}
    wanted = set(frame_indices)
    index = 0
    while wanted:
        ok, frame = capture.read()
        if not ok:
            break
        if index in wanted:
            images[index] = frame
            wanted.discard(index)
        index += 1
    capture.release()
    tiles = []
    for frame_index in frame_indices:
        frame = images.get(frame_index)
        if frame is None:
            continue
        tile = frame.copy()
        record = frames.get(frame_index)
        shown = (
            []
            if record is None
            else [d for d in record["detections"] if d["score"] >= display_threshold]
        )
        height, width = tile.shape[:2]
        # Draw low scores first so the strongest boxes and captions end up on top.
        for det in sorted(shown, key=lambda d: d["score"]):
            rgb = class_color(det["class"])
            bgr = (rgb[2], rgb[1], rgb[0])
            x1, y1, x2, y2 = (int(round(v)) for v in det["box_xyxy_px"])
            cv2.rectangle(tile, (x1, y1), (x2, y2), bgr, 3)
            caption = f"{det['class']} {det['score']:.2f}"
            origin = (x1 + 4, max(y1 - 8, 24) if y1 > 60 else min(y2 + 26, height - 8))
            cv2.putText(
                tile, caption, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA
            )
            cv2.putText(tile, caption, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.75, bgr, 2, cv2.LINE_AA)
        source_seconds = 0.0 if record is None else record["source_seconds"]
        status = (
            "not processed" if record is None else f"boxes >= {display_threshold:.2f}: {len(shown)}"
        )
        header = f"frame {frame_index}  source {source_seconds:.2f} s  {status}"
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
    legend = np.zeros((80, sheet.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        legend, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
    )
    x = 10
    for family, rgb in FAMILY_COLORS.items():
        bgr = (rgb[2], rgb[1], rgb[0])
        cv2.putText(legend, family, (x, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr, 2, cv2.LINE_AA)
        x += 30 + int(cv2.getTextSize(family, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0])
    cv2.imwrite(str(output_path), np.vstack([sheet, legend]))
    return output_path


def run_export(args: argparse.Namespace) -> int:
    import cv2

    run_directory = Path(args.run_directory)
    sam3_run_directory = Path(args.sam3_run_directory)
    detector_runs = load_detector_runs(run_directory)
    if "dino" not in detector_runs:
        raise SystemExit("no successful DINO detect phase in the run directory")
    sam3_track = json.loads((sam3_run_directory / "track_result.json").read_text(encoding="utf-8"))
    if sam3_track.get("state") != "succeeded":
        raise SystemExit(f"SAM3 smoke did not succeed: {sam3_track.get('reason')}")
    sam3_detections = sam3_track["runtime_settings"]["frame_zero_detections"]
    sam3_observations = load_jsonl(sam3_run_directory / "observations.jsonl")
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
    dino_settings = detector_runs["dino"]["result"]["runtime_settings"]
    analysis_fps = float(dino_settings["analysis_fps"])
    source_offset_seconds = float(dino_settings["source_offset_seconds"])
    if proxy_frames != len(sam3_observations):
        raise SystemExit(
            f"proxy has {proxy_frames} frames but the SAM3 smoke wrote {len(sam3_observations)}"
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
                    source_offset_seconds,
                    source_offset_seconds + proxy_frames / analysis_fps,
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
        detector_runs=detector_runs,
        sam3_run_directory=sam3_run_directory,
        sam3_observations=sam3_observations,
        sam3_detections=sam3_detections,
        video_path=video_path,
        dimensions=dimensions,
        frame_count=proxy_frames,
        analysis_fps=analysis_fps,
        source_offset_seconds=source_offset_seconds,
        display_threshold=args.display_threshold,
        clip_id=args.clip_id,
        view_id=args.view_id,
        run_id=args.run_id or run_directory.name,
        asset_reference=asset_reference,
    )
    last = proxy_frames - 1
    frame_indices = [round(i * last / 5) for i in range(6)]
    sheets = {}
    tables = {}
    for model_name, run in detector_runs.items():
        suffix = "" if model_name == "dino" else f".{model_name}"
        sheets[model_name] = render_contact_sheet(
            output_path=run_directory / f"contact_sheet{suffix}.png",
            video_path=video_path,
            frames=run["frames"],
            frame_indices=frame_indices,
            display_threshold=args.display_threshold,
            title=(
                f"FineBio {model_name} on P03_01_01 60-80 s, boxes with score >= "
                f"{args.display_threshold:.2f}; colour = class family"
            ),
        ).name
        tables[model_name] = class_count_table(run["frames"], args.display_threshold)
    frame_counts = {m: len(run["frames"]) for m, run in detector_runs.items()}
    (run_directory / "class_counts.md").write_text(
        class_count_markdown(tables, frame_counts, args.display_threshold), encoding="utf-8"
    )

    models_manifest = {}
    for model_name, run in detector_runs.items():
        result = run["result"]
        settings = result["runtime_settings"]
        config_path = Path(settings["config"])
        weights_path = Path(settings["weights"])
        models_manifest[model_name] = {
            "description": settings["model_description"],
            "config": str(config_path),
            "config_sha256": fs_common.sha256_file(config_path),
            "weights": str(weights_path),
            "weights_sha256": fs_common.sha256_file(weights_path),
            "weights_size_bytes": weights_path.stat().st_size,
            "versions": result["versions"],
            "device": settings["device"],
            "torch_threads": settings["torch_threads"],
            "test_pipeline": settings["test_pipeline"],
            "frames_processed": result["frames_processed"],
            "frames_processed_count": result["frames_processed_count"],
            "frame_stride": settings["frame_stride"],
            "included_frames": settings["included_frames"],
            "record_threshold": settings["record_threshold"],
            "elapsed_seconds": result["elapsed_seconds"],
            "model_load_seconds": result["model_load_seconds"],
            "per_frame_seconds_summary": result["per_frame_seconds_summary"],
            "detections_recorded": result["detections_recorded"],
            "gpu_coordination": settings["gpu_coordination"],
            "class_counts_at_display_threshold": tables[model_name],
            "detections_file": result["detections_file"],
        }
    manifest = {
        "run_id": args.run_id or run_directory.name,
        "kind": "finebio_shipped_detector_first_run",
        "claim_boundary": CLAIM_BOUNDARY,
        "inputs": asset_reference,
        "display_threshold": args.display_threshold,
        "models": models_manifest,
        "mmdetection": {
            "path": str(DETECTOR_DIR / "mmdetection"),
            **git_revision(DETECTOR_DIR / "mmdetection"),
        },
        "sam3_smoke": {
            "run_directory": str(sam3_run_directory),
            "manifest_sha256": fs_common.sha256_file(sam3_run_directory / "manifest.json"),
            "observations_sha256": fs_common.sha256_file(sam3_run_directory / "observations.jsonl"),
            "frame_zero_detections": sam3_detections,
        },
        "battle": git_revision(Path(__file__).resolve().parents[1]),
        "host": platform.node(),
        "outputs": {
            "recording_rrd": rrd_path.name,
            "contact_sheets": sheets,
            "contact_sheet_frames": frame_indices,
            "class_counts": "class_counts.md",
        },
    }
    fs_common.write_json(run_directory / "manifest.json", manifest, sort_keys=True)
    print(
        json.dumps(
            {
                "rrd": str(rrd_path),
                "contact_sheets": sheets,
                "frames_processed": frame_counts,
                "classes_detected": {
                    m: [c for c, row in t.items() if row["total_detections"]]
                    for m, t in tables.items()
                },
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="phase", required=True)
    detect = sub.add_parser("detect", help="MMDetection inference (detector interpreter)")
    detect.add_argument("--run-directory", required=True)
    detect.add_argument("--video", required=True, help="600-frame 30 fps proxy")
    detect.add_argument("--model", choices=sorted(MODELS), default="dino")
    detect.add_argument("--device", default="cpu", help="cpu (default) or cuda:0")
    detect.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    detect.add_argument("--frame-stride", type=int, default=10, help="0 = only --include-frame")
    detect.add_argument("--include-frame", type=int, action="append", default=[])
    detect.add_argument("--max-frames", type=int, default=600)
    detect.add_argument("--record-threshold", type=float, default=RECORD_THRESHOLD)
    detect.add_argument("--analysis-fps", type=float, default=30.0)
    detect.add_argument("--source-offset-seconds", type=float, default=0.0)
    detect.add_argument(
        "--allow-gpu-neighbour", type=int, action="append", default=[], metavar="PID"
    )
    detect.add_argument("--queue-log", help="other session's queue.log (default: newest)")
    detect.set_defaults(func=run_detect)
    export = sub.add_parser("export", help="RRD, contact sheet, counts, manifest (Battle)")
    export.add_argument("--run-directory", required=True)
    export.add_argument("--video", required=True)
    export.add_argument("--source-video")
    export.add_argument("--sam3-run-directory", default="runs/finebio-sam3-smoke-20260921")
    export.add_argument("--display-threshold", type=float, default=DISPLAY_THRESHOLD)
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
