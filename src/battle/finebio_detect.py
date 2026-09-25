"""`battle-finebio-detect`: FineBio's shipped detector on raw FineBio frames, one JSONL per view.

Supersedes ``scripts/finebio_dino_detect.py`` (Sep 21, one proxy, CPU, Rerun export) and
``scripts/finebio_preflight_detect.py`` (Sep 24, the preflight's raw-frame pass); both stay
in place because records cite them.

Frames are read from the **raw** FineBio videos by raw frame index (fixed cameras 1920x1080,
first-person 1920x1440, 30000/1001 fps, identical frame counts per trial), so a box's pixel
coordinates line up with the shipped calibration after the resolution rescale and the plan's
frame-index contract (proxy frame k == raw frame start+k) never has to be undone.

Two interpreters, as the Sep 21 script: the CLI runs in the Battle venv (``uv run``), the
MMDetection inference in the detector venv under ``/home/nick/src/finebio-detector`` built by
``scripts/install_finebio_detector.sh`` (``.venv`` = CPU torch 2.1, ``.venv-cuda`` = the CUDA
build); ``--device`` selects the interpreter, and the CLI spawns it on this file's ``worker``
subcommand, which imports only the standard library, ``fs_common`` and ``gpu_guard`` by path::

    uv run battle-finebio-detect run --trial P03_03_01 --views fpv,T1,T2,T3,T4,T5 \\
        --start 600 --count 3600 --model dino --device cuda --output runs/<run>/detections
    uv run battle-finebio-detect interpolate --directory runs/<run>/detections

Outputs in ``--output``: ``<view>.jsonl`` (one row per raw frame: ``view``, ``frame_index``,
``image_hw``, ``detections`` with ``class``, ``class_id``, ``score``, ``box_xyxy_px`` at score
>= 0.05, ``interpolated`` and, on filled rows, ``source_frames``), ``frames.json``,
``worker_result.json`` (the detector interpreter's own record) and ``manifest.json``.

Strides (``--stride-fpv``, ``--stride-fixed``) are the plan's CPU fallback: detected frames are
written as they are and ``interpolate`` fills the skipped frames by linear interpolation of box
coordinates and scores between same-class detections associated by IoU >= 0.3 on the two
neighbouring detected frames; a class absent on either side is never invented. The manifest's
``detection_coverage`` says which happened: ``full``, ``strided``, ``strided_interpolated`` or
``explicit`` (an irregular ``--frames`` list such as the preflight set).

GPU coordination is ``battle.gpu_guard`` in the worker process (the one that will hold the
card): the ``vram`` mode by default with the ``unknown`` profile (4 GiB expected peak, so 6 GiB
of headroom is required), a benign neighbour such as a game or a compositor never blocks,
another Battle worker does unless its PID is named with ``--allow-gpu-neighbour``. The
decision is recorded in the manifest. Rerun export is not part of this tool: the viewer lane
logs detector boxes from the JSONL.

FineBio licence: non-commercial research; nothing under ``data/`` or ``runs/`` is committed.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

try:
    from . import digest_cache, fs_common, gpu_guard
except ImportError:
    # Under the detector interpreter the battle package is not installed: import the
    # stdlib-only helper modules by path, as the workers do.
    sys.path.append(str(Path(__file__).resolve().parent))
    import fs_common  # type: ignore[no-redef]
    import gpu_guard  # type: ignore[no-redef]

    digest_cache = None  # type: ignore[assignment]

MANIFEST_SCHEMA = "battle-finebio-detect/1"
MANIFEST_KIND = "finebio_detections"
DEFAULT_DETECTOR_DIR = Path("/home/nick/src/finebio-detector")
DEFAULT_RAW_ROOT = Path("data/raw/finebio")
VIEWS: tuple[str, ...] = ("fpv", "T1", "T2", "T3", "T4", "T5")
FIXED_VIEWS: tuple[str, ...] = VIEWS[1:]
DEVICES = ("cpu", "cuda")
INTERPRETERS = {"cpu": ".venv/bin/python", "cuda": ".venv-cuda/bin/python"}
RECORD_THRESHOLD = 0.05
INTERPOLATION_IOU = 0.3
COVERAGES = ("full", "strided", "strided_interpolated", "explicit")
STATES = ("succeeded", "failed", "blocked")
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
FINEBIO_CLASSES: tuple[str, ...] = (
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
CLAIM_BOUNDARY = (
    "FineBio's shipped MMDetection detector (the authors' fine-tuned checkpoint, 35 classes) "
    "run on raw FineBio frames. Scores are the detector's own outputs; the FineBio annotations "
    "are treated as unavailable, so nothing here is scored against ground truth. The detector "
    "was trained on FineBio's own objects and cameras, so its quality on these frames is an "
    "upper bound for a new bench. Interpolated rows are linear guesses between two detections "
    "and are flagged as such."
)


# --------------------------------------------------------------------------------------------
# paths and frame selection


def video_path(raw_root: Path, trial: str, view: str) -> Path:
    """Raw FineBio video of `trial` for `view` (`fpv` or `T1..T5`) under `raw_root`."""
    if view == "fpv":
        return raw_root / "finebio_videos_fpv_test" / "finebio_videos" / f"{trial}.mp4"
    return raw_root / "finebio_videos_tpv_test" / "finebio_videos" / f"{trial}_{view}.mp4"


def parse_views(text: str) -> list[str]:
    """`fpv,T1,T5` -> `['fpv', 'T1', 'T5']`, in the canonical order, duplicates dropped."""
    wanted = [token.strip() for token in text.split(",") if token.strip()]
    unknown = sorted(set(wanted) - set(VIEWS))
    if unknown:
        raise ValueError(f"unknown view(s) {unknown}; known: {', '.join(VIEWS)}")
    if not wanted:
        raise ValueError("no views requested")
    return [view for view in VIEWS if view in wanted]


def parse_frame_spec(text: str) -> list[int]:
    """Raw frame indices from `N`, `A-B` (inclusive) and `A:B:S` (`range(A, B, S)`) tokens.

    `1798-1802,1830:2400:30` is the shape of the preflight set; the result is sorted and
    de-duplicated.
    """
    frames: set[int] = set()
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            parts = [int(v) for v in token.split(":")]
            if len(parts) != 3 or parts[2] <= 0:
                raise ValueError(f"expected A:B:S with S > 0, got {token!r}")
            frames.update(range(parts[0], parts[1], parts[2]))
        elif "-" in token:
            first, last = (int(v) for v in token.split("-", 1))
            if last < first:
                raise ValueError(f"range {token!r} runs backwards")
            frames.update(range(first, last + 1))
        else:
            frames.add(int(token))
    if any(frame < 0 for frame in frames):
        raise ValueError("frame indices must be >= 0")
    return sorted(frames)


def strided_frames(start: int, count: int, stride: int) -> list[int]:
    """`start, start+stride, ...` inside `[start, start+count)`, always including the last frame.

    The last frame is kept so interpolation can cover the whole window: filled rows exist
    only between two detected frames.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    frames = list(range(start, start + count, stride))
    last = start + count - 1
    if frames[-1] != last:
        frames.append(last)
    return frames


def stride_for_view(view: str, stride_fpv: int, stride_fixed: int) -> int:
    return stride_fpv if view == "fpv" else stride_fixed


def frames_per_view(
    views: list[str],
    *,
    start: int | None,
    count: int | None,
    explicit: list[int] | None,
    stride_fpv: int,
    stride_fixed: int,
) -> tuple[dict[str, list[int]], str]:
    """Detected frames per view and the coverage they imply (`full`, `strided`, `explicit`)."""
    if explicit is not None:
        if not explicit:
            raise ValueError("--frames selected no frames")
        return {view: list(explicit) for view in views}, "explicit"
    if start is None or count is None:
        raise ValueError("either --frames or both --start and --count are required")
    per_view = {
        view: strided_frames(start, count, stride_for_view(view, stride_fpv, stride_fixed))
        for view in views
    }
    strided = any(stride_for_view(view, stride_fpv, stride_fixed) > 1 for view in views)
    return per_view, ("strided" if strided else "full")


# --------------------------------------------------------------------------------------------
# JSONL rows


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write `rows` to `<path>.tmp` and rename: a half-written file never replaces a whole one."""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def detection_row(
    view: str, frame_index: int, image_hw: tuple[int, int], detections: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "view": view,
        "frame_index": int(frame_index),
        "image_hw": [int(image_hw[0]), int(image_hw[1])],
        "detections": detections,
        "interpolated": False,
    }


# --------------------------------------------------------------------------------------------
# interpolation (the plan's CPU fallback)


def box_iou(a: list[float], b: list[float]) -> float:
    """IoU of two `[x1, y1, x2, y2]` boxes; 0 for degenerate boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def associate_detections(
    left: list[dict[str, Any]], right: list[dict[str, Any]], iou_threshold: float
) -> list[tuple[int, int, float]]:
    """Greedy one-to-one same-class pairing `(left index, right index, IoU)` by descending IoU.

    Only pairs with the same `class` and IoU >= `iou_threshold` are candidates; a detection
    is used once. A class present on one side only, or whose boxes moved further than the
    threshold allows, gets no pair, so nothing is interpolated for it.
    """
    candidates: list[tuple[float, int, int]] = []
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            if a["class"] != b["class"]:
                continue
            iou = box_iou(a["box_xyxy_px"], b["box_xyxy_px"])
            if iou >= iou_threshold:
                candidates.append((iou, i, j))
    candidates.sort(key=lambda c: (-c[0], c[1], c[2]))
    used_left: set[int] = set()
    used_right: set[int] = set()
    pairs: list[tuple[int, int, float]] = []
    for iou, i, j in candidates:
        if i in used_left or j in used_right:
            continue
        used_left.add(i)
        used_right.add(j)
        pairs.append((i, j, iou))
    return pairs


def interpolate_between(
    left_row: dict[str, Any], right_row: dict[str, Any], iou_threshold: float
) -> list[dict[str, Any]]:
    """Filled rows for every frame strictly between two detected rows of one view."""
    a, b = int(left_row["frame_index"]), int(right_row["frame_index"])
    if b - a < 2:
        return []
    pairs = associate_detections(left_row["detections"], right_row["detections"], iou_threshold)
    rows = []
    for frame in range(a + 1, b):
        t = (frame - a) / (b - a)
        detections = []
        for i, j, _iou in pairs:
            da, db = left_row["detections"][i], right_row["detections"][j]
            detections.append(
                {
                    "class": da["class"],
                    "class_id": da.get("class_id", db.get("class_id")),
                    "score": (1.0 - t) * float(da["score"]) + t * float(db["score"]),
                    "box_xyxy_px": [
                        (1.0 - t) * float(u) + t * float(v)
                        for u, v in zip(da["box_xyxy_px"], db["box_xyxy_px"], strict=True)
                    ],
                }
            )
        detections.sort(key=lambda d: -d["score"])
        rows.append(
            {
                "view": left_row["view"],
                "frame_index": frame,
                "image_hw": list(left_row["image_hw"]),
                "detections": detections,
                "interpolated": True,
                "source_frames": [a, b],
            }
        )
    return rows


def interpolate_rows(
    rows: list[dict[str, Any]], iou_threshold: float = INTERPOLATION_IOU
) -> tuple[list[dict[str, Any]], int]:
    """Detected rows plus filled rows between each consecutive pair; `(rows, filled count)`.

    Earlier filled rows are dropped first, so the operation is idempotent.
    """
    detected = sorted(
        (row for row in rows if not row.get("interpolated", False)),
        key=lambda row: int(row["frame_index"]),
    )
    output: list[dict[str, Any]] = []
    filled = 0
    for left, right in zip(detected, detected[1:], strict=False):
        output.append(left)
        between = interpolate_between(left, right, iou_threshold)
        filled += len(between)
        output.extend(between)
    if detected:
        output.append(detected[-1])
    return output, filled


# --------------------------------------------------------------------------------------------
# manifest


@dataclass
class DetectManifest:
    """`manifest.json` of one detector run; stdlib only so both interpreters can build it."""

    state: str
    trial: str
    views: list[str]
    model: dict[str, Any]
    settings: dict[str, Any]
    frames: dict[str, Any]
    detection_coverage: str
    reason: str | None = None
    run_id: str = ""
    versions: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, Any] = field(default_factory=dict)
    per_frame_seconds: dict[str, list[float]] = field(default_factory=dict)
    gpu: dict[str, Any] = field(default_factory=dict)
    interpolation: dict[str, Any] | None = None
    inputs: dict[str, Any] = field(default_factory=dict)
    battle: dict[str, Any] = field(default_factory=dict)
    mmdetection: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    host: str = ""
    created_at: str = ""
    schema: str = MANIFEST_SCHEMA
    kind: str = MANIFEST_KIND
    claim_boundary: str = CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        if self.schema != MANIFEST_SCHEMA:
            raise ValueError(f"not a {MANIFEST_SCHEMA} manifest: {self.schema!r}")
        if self.state not in STATES:
            raise ValueError(f"state must be one of {STATES}, got {self.state!r}")
        if self.detection_coverage not in COVERAGES:
            raise ValueError(
                f"detection_coverage must be one of {COVERAGES}, got {self.detection_coverage!r}"
            )
        if self.model.get("name") not in MODELS:
            raise ValueError(f"model.name must be one of {sorted(MODELS)}")
        unknown = sorted(set(self.views) - set(VIEWS))
        if unknown:
            raise ValueError(f"unknown views in manifest: {unknown}")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DetectManifest:
        known = {f for f in cls.__dataclass_fields__}
        unexpected = sorted(set(payload) - known)
        if unexpected:
            raise ValueError(f"unexpected manifest keys: {unexpected}")
        return cls(**payload)


def read_manifest(directory: Path) -> DetectManifest:
    return DetectManifest.from_dict(
        json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    )


def write_manifest(directory: Path, manifest: DetectManifest) -> Path:
    path = directory / "manifest.json"
    fs_common.write_json(path, manifest.as_dict(), sort_keys=True)
    return path


def sha256(path: Path) -> str:
    if digest_cache is not None:
        return digest_cache.sha256_file(path)
    return fs_common.sha256_file(path)


def utc_now() -> str:
    # `datetime.UTC` is 3.11+ and the detector interpreters are 3.10, hence `time.gmtime`.
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def summarize(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": ordered[len(ordered) // 2],
        "mean": sum(ordered) / len(ordered),
        "max": ordered[-1],
        "count": len(ordered),
    }


# --------------------------------------------------------------------------------------------
# worker (detector interpreter: torch, mmcv, mmdet, cv2)


def gpu_guard_decision(args: argparse.Namespace) -> gpu_guard.GuardDecision:
    """The worker's own guard decision, with itself as `own_pid` (it will hold the card)."""
    return gpu_guard.evaluate(
        mode=args.gpu_guard,
        allowed_pids=args.allow_gpu_neighbour,
        profile=args.gpu_guard_profile,
        expected_peak_vram_bytes=args.expected_peak_vram_bytes,
        own_pid=os.getpid(),
    )


def read_frames_file(path: Path) -> dict[str, list[int]]:
    """`frames_per_view` of a `frames.json` written by :func:`write_frames_json`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        view: [int(frame) for frame in frames]
        for view, frames in payload["frames_per_view"].items()
    }


def run_worker(args: argparse.Namespace) -> int:
    """MMDetection inference over the requested raw frames.

    Writes `<view>.jsonl` per view and `worker_result.json`; exit 0 only when every requested
    frame was detected.
    """
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "worker_result.json"
    detector_dir = Path(args.detector_dir)
    config_path = detector_dir / MODELS[args.model]["config"]
    weights_path = detector_dir / MODELS[args.model]["weights"]
    per_view_frames = read_frames_file(Path(args.frames_file))
    videos = {view: Path(path) for view, path in json.loads(args.videos_json).items()}
    start = perf_counter()
    guard = gpu_guard_decision(args)
    record: dict[str, Any] = {
        "state": "failed",
        "reason": None,
        "model": args.model,
        "device": args.device,
        "interpreter": sys.executable,
        "config": str(config_path),
        "weights": str(weights_path),
        "record_threshold": args.record_threshold,
        "torch_threads": args.threads,
        "tf32": None,
        "gpu_guard": guard.as_provenance(),
        "videos": {view: str(path) for view, path in videos.items()},
        "video_info": {},
        "frames_requested": {view: len(frames) for view, frames in per_view_frames.items()},
        "frames_processed": {},
        "per_frame_seconds": {},
        "detections_recorded": {},
        "model_load_seconds": None,
        "elapsed_seconds": None,
        "versions": {},
        "gpu_peak_vram_bytes": None,
        "gpu_peak_allocated_bytes": None,
        "test_pipeline": None,
        "num_queries": None,
        "max_per_img": None,
        "classes": None,
    }

    def finish(state: str, reason: str | None) -> int:
        record["state"] = state
        record["reason"] = reason
        record["elapsed_seconds"] = perf_counter() - start
        fs_common.write_json(result_path, record, sort_keys=True)
        if reason:
            print(f"{state}: {reason}", file=sys.stderr)
        return 0 if state == "succeeded" else 2

    missing = [str(path) for path in videos.values() if not path.is_file()]
    if missing:
        return finish("blocked", f"raw video(s) missing: {missing}")
    if not config_path.is_file() or not weights_path.is_file():
        return finish(
            "blocked",
            f"detector files missing ({config_path}, {weights_path}); "
            "run scripts/install_finebio_detector.sh",
        )
    if args.device != "cpu" and not guard.accepted:
        return finish("blocked", f"GPU guard refused: {guard.reason}")

    try:
        # torch >= 2.6 loads checkpoints weights-only by default; the authors' checkpoints
        # carry mmengine HistoryBuffers (numpy arrays pickled through `getattr`) in their
        # message hub, which the safe unpickler refuses. The file is the one whose sha256 the
        # manifest records, so this process alone reads it in full, as torch 2.1 always did.
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        import cv2
        import mmcv
        import mmdet
        import mmengine
        import torch
        from mmdet.apis import inference_detector, init_detector

        if args.threads:
            torch.set_num_threads(args.threads)
        record["torch_threads"] = torch.get_num_threads()
        record["versions"] = {
            "python": sys.version.split()[0],
            "interpreter": sys.executable,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "mmcv": mmcv.__version__,
            "mmengine": mmengine.__version__,
            "mmdet": mmdet.__version__,
            "opencv": cv2.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "torch_load_weights_only": False,
        }
        if args.device == "cpu" and torch.cuda.is_available():
            raise RuntimeError("CPU run requested but CUDA is visible; set CUDA_VISIBLE_DEVICES=''")
        if args.device != "cpu" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        device = "cuda:0" if args.device == "cuda" else args.device
        if device.startswith("cuda"):
            # TF32 convolutions (cuDNN's default) move scores by up to ~0.03 against the CPU
            # reference; strict fp32 reproduces it to 0.01 px / 0.0004 at ~10% more time.
            torch.backends.cudnn.allow_tf32 = bool(args.tf32)
            torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
            torch.cuda.reset_peak_memory_stats()
        record["tf32"] = bool(args.tf32) if device.startswith("cuda") else None

        load_start = perf_counter()
        model = init_detector(str(config_path), str(weights_path), device=device)
        record["model_load_seconds"] = perf_counter() - load_start
        classes = tuple(model.dataset_meta["classes"])
        if classes != FINEBIO_CLASSES:
            raise RuntimeError(f"checkpoint classes differ from the FineBio list: {classes}")
        record["classes"] = list(classes)
        pipeline = model.cfg.test_dataloader.dataset.pipeline
        record["test_pipeline"] = [
            {k: (list(v) if isinstance(v, tuple) else v) for k, v in step.items()}
            for step in pipeline
        ]
        record["num_queries"] = int(getattr(model, "num_queries", 0) or 0)
        record["max_per_img"] = int(model.cfg.model.test_cfg.get("max_per_img", 0))

        for view, frames in per_view_frames.items():
            path = videos[view]
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                raise RuntimeError(f"could not open {path}")
            record["video_info"][view] = {
                "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
                "fps": float(capture.get(cv2.CAP_PROP_FPS)),
                "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            }
            # Seek once to the first wanted frame, then read sequentially (the preflight's
            # access pattern, so the decoded pixels and the boxes are the same).
            capture.set(cv2.CAP_PROP_POS_FRAMES, frames[0])
            wanted = set(frames)
            index = frames[0]
            seconds: list[float] = []
            processed: list[int] = []
            recorded = 0
            rows: list[dict[str, Any]] = []
            view_start = perf_counter()
            while wanted:
                if index not in wanted:
                    if not capture.grab():
                        break
                    index += 1
                    continue
                ok, frame = capture.read()
                if not ok:
                    break
                wanted.discard(index)
                infer_start = perf_counter()
                with torch.inference_mode():
                    result = inference_detector(model, frame)
                instances = result.pred_instances
                scores = instances.scores.cpu().numpy()
                labels = instances.labels.cpu().numpy()
                boxes = instances.bboxes.cpu().numpy()
                infer_seconds = perf_counter() - infer_start
                keep = scores >= args.record_threshold
                order = scores[keep].argsort(kind="stable")[::-1]
                detections = [
                    {
                        "class": classes[int(labels[keep][i])],
                        "class_id": int(labels[keep][i]),
                        "score": float(scores[keep][i]),
                        "box_xyxy_px": [float(v) for v in boxes[keep][i]],
                    }
                    for i in order
                ]
                recorded += len(detections)
                rows.append(
                    detection_row(view, index, (frame.shape[0], frame.shape[1]), detections)
                )
                seconds.append(infer_seconds)
                processed.append(index)
                if len(processed) % 50 == 0 or len(processed) == len(frames):
                    print(
                        f"{view}: {len(processed)}/{len(frames)} frames, "
                        f"{1000 * infer_seconds:.0f} ms/frame, "
                        f"{perf_counter() - view_start:.0f}s this view, "
                        f"{perf_counter() - start:.0f}s total",
                        flush=True,
                    )
                index += 1
            capture.release()
            write_jsonl(output / f"{view}.jsonl", rows)
            record["frames_processed"][view] = processed
            record["per_frame_seconds"][view] = seconds
            record["detections_recorded"][view] = recorded
            if wanted:
                raise RuntimeError(f"{view}: video ended before frames {sorted(wanted)[:5]}")
        if device.startswith("cuda"):
            record["gpu_peak_vram_bytes"] = int(torch.cuda.max_memory_reserved())
            record["gpu_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        return finish("succeeded", None)
    except Exception as error:  # noqa: BLE001 - recorded, not swallowed
        traceback.print_exc()
        return finish("failed", f"{type(error).__name__}: {error}")


# --------------------------------------------------------------------------------------------
# driver (Battle interpreter)


def interpreter_for(detector_dir: Path, device: str, override: str | None) -> Path:
    if override:
        return Path(override)
    return detector_dir / INTERPRETERS[device]


def worker_command(
    args: argparse.Namespace,
    *,
    interpreter: Path,
    output: Path,
    per_view_frames: dict[str, list[int]],
    videos: dict[str, Path],
) -> list[str]:
    """The detector-interpreter command line for one run (this file, `worker` subcommand)."""
    command = [
        str(interpreter),
        str(Path(__file__).resolve()),
        "worker",
        "--model",
        args.model,
        "--device",
        args.device,
        "--detector-dir",
        str(Path(args.detector_dir).resolve()),
        "--output",
        str(output.resolve()),
        "--record-threshold",
        str(args.record_threshold),
        "--threads",
        str(args.threads),
        "--gpu-guard",
        args.gpu_guard,
        "--gpu-guard-profile",
        args.gpu_guard_profile,
        "--frames-file",
        str((output / "frames.json").resolve()),
        "--videos-json",
        json.dumps({view: str(path.resolve()) for view, path in videos.items()}),
    ]
    if args.expected_peak_vram_bytes is not None:
        command += ["--expected-peak-vram-bytes", str(args.expected_peak_vram_bytes)]
    if args.tf32:
        command.append("--tf32")
    for pid in args.allow_gpu_neighbour:
        command += ["--allow-gpu-neighbour", str(pid)]
    return command


def worker_environment(device: str) -> dict[str, str]:
    """The worker's environment: CUDA hidden for a CPU run, device 0 otherwise."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "" if device == "cpu" else env.get("CUDA_VISIBLE_DEVICES", "0")
    return env


def _timing(record: dict[str, Any]) -> dict[str, Any]:
    per_view = {}
    all_seconds: list[float] = []
    for view, seconds in record.get("per_frame_seconds", {}).items():
        all_seconds.extend(seconds)
        per_view[view] = {
            "frames": len(seconds),
            "inference_seconds": sum(seconds),
            "mean_ms_per_frame": (1000.0 * sum(seconds) / len(seconds)) if seconds else None,
        }
    summary = summarize(all_seconds)
    return {
        "elapsed_seconds": record.get("elapsed_seconds"),
        "model_load_seconds": record.get("model_load_seconds"),
        "per_frame_seconds_summary": summary,
        "mean_ms_per_frame": None if summary is None else 1000.0 * summary["mean"],
        "per_view": per_view,
    }


def build_manifest(
    *,
    args: argparse.Namespace,
    output: Path,
    views: list[str],
    per_view_frames: dict[str, list[int]],
    coverage: str,
    videos: dict[str, Path],
    interpreter: Path,
    worker_record: dict[str, Any] | None,
    state: str,
    reason: str | None,
    repository_root: Path,
) -> DetectManifest:
    detector_dir = Path(args.detector_dir)
    config_path = detector_dir / MODELS[args.model]["config"]
    weights_path = detector_dir / MODELS[args.model]["weights"]
    record = worker_record or {}
    model: dict[str, Any] = {
        "name": args.model,
        "description": MODELS[args.model]["description"],
        "config": str(config_path),
        "weights": str(weights_path),
        "config_sha256": sha256(config_path) if config_path.is_file() else None,
        "weights_sha256": sha256(weights_path) if weights_path.is_file() else None,
        "weights_size_bytes": weights_path.stat().st_size if weights_path.is_file() else None,
        "classes": record.get("classes"),
        "num_queries": record.get("num_queries"),
        "max_per_img": record.get("max_per_img"),
        "test_pipeline": record.get("test_pipeline"),
    }
    notes: list[str] = []
    video_info = record.get("video_info", {})
    counts = {info["frame_count"] for info in video_info.values()}
    if len(counts) > 1:
        notes.append(f"raw videos report different frame counts: {video_info}")
    inputs = {
        view: {
            "uri": fs_common.relative_uri(path, repository_root),
            "sha256": sha256(path) if (path.is_file() and state == "succeeded") else None,
            **video_info.get(view, {}),
        }
        for view, path in videos.items()
    }
    window = (
        None
        if args.frames is not None
        else {"start": args.start, "count": args.count, "end_exclusive": args.start + args.count}
    )
    frames = {
        "window": window,
        "explicit": args.frames,
        "stride_fpv": args.stride_fpv,
        "stride_fixed": args.stride_fixed,
        "per_view": {
            view: {
                "stride": stride_for_view(view, args.stride_fpv, args.stride_fixed),
                "detected_count": len(frames_),
                "first": frames_[0],
                "last": frames_[-1],
            }
            for view, frames_ in per_view_frames.items()
        },
        "file": "frames.json",
    }
    return DetectManifest(
        state=state,
        reason=reason,
        run_id=args.run_id or output.resolve().parent.name,
        trial=args.trial,
        views=views,
        model=model,
        settings={
            "device": args.device,
            "interpreter": str(interpreter),
            "tf32": record.get("tf32"),
            "torch_threads": record.get("torch_threads", args.threads),
            "record_threshold": args.record_threshold,
            "gpu_guard_mode": args.gpu_guard,
            "gpu_guard_profile": args.gpu_guard_profile,
            "expected_peak_vram_bytes": args.expected_peak_vram_bytes,
            "allowed_gpu_neighbour_pids": list(args.allow_gpu_neighbour),
            "detector_dir": str(detector_dir),
            "raw_root": str(args.raw_root),
        },
        frames=frames,
        detection_coverage=coverage,
        versions=record.get("versions", {}),
        timing=_timing(record),
        per_frame_seconds=record.get("per_frame_seconds", {}),
        gpu={
            "peak_vram_bytes": record.get("gpu_peak_vram_bytes"),
            "peak_allocated_bytes": record.get("gpu_peak_allocated_bytes"),
            "cuda_available": record.get("versions", {}).get("cuda_available"),
            "guard": record.get("gpu_guard"),
        },
        interpolation=None,
        inputs=inputs,
        battle=fs_common.git_revision(repository_root),
        mmdetection={
            "path": str(detector_dir / "mmdetection"),
            **fs_common.git_revision(detector_dir / "mmdetection"),
        },
        outputs={
            "detections": {view: f"{view}.jsonl" for view in views},
            "frames": "frames.json",
            "worker_result": "worker_result.json",
            "worker_log": "worker.log",
        },
        notes=notes,
        host=platform.node(),
        created_at=utc_now(),
    )


def write_frames_json(
    output: Path, *, trial: str, views: list[str], per_view_frames: dict[str, list[int]]
) -> None:
    union = sorted({frame for frames in per_view_frames.values() for frame in frames})
    fs_common.write_json(
        output / "frames.json",
        {
            "trial": trial,
            "views": views,
            "frames": union,
            "frames_per_view": per_view_frames,
        },
    )


def run_interpolate_directory(
    directory: Path, iou_threshold: float = INTERPOLATION_IOU
) -> dict[str, int]:
    """Fill the skipped frames of every `<view>.jsonl` in `directory`; update the manifest."""
    manifest = read_manifest(directory)
    if manifest.state != "succeeded":
        raise RuntimeError(f"manifest state is {manifest.state}: {manifest.reason}")
    filled_per_view: dict[str, int] = {}
    for view in manifest.views:
        path = directory / f"{view}.jsonl"
        rows, filled = interpolate_rows(load_jsonl(path), iou_threshold)
        write_jsonl(path, rows)
        filled_per_view[view] = filled
    manifest.interpolation = {
        "method": "linear box and score interpolation between same-class detections",
        "association": (
            f"greedy one-to-one IoU >= {iou_threshold:g} between consecutive detected frames"
        ),
        "iou_threshold": iou_threshold,
        "filled_rows_per_view": filled_per_view,
        "rule": "a class absent on either neighbouring detected frame is never invented",
        "applied_at": utc_now(),
    }
    if any(filled_per_view.values()):
        manifest.detection_coverage = "strided_interpolated"
    write_manifest(directory, manifest)
    return filled_per_view


def run_run(args: argparse.Namespace) -> int:
    repository_root = Path(args.repository_root).resolve()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        print(f"{output} exists and is not empty; pass --overwrite", file=sys.stderr)
        return 2
    output.mkdir(parents=True, exist_ok=True)
    views = parse_views(args.views)
    per_view_frames, coverage = frames_per_view(
        views,
        start=args.start,
        count=args.count,
        explicit=args.frames,
        stride_fpv=args.stride_fpv,
        stride_fixed=args.stride_fixed,
    )
    raw_root = Path(args.raw_root)
    if not raw_root.is_absolute():
        raw_root = repository_root / raw_root
    videos = {view: video_path(raw_root, args.trial, view) for view in views}
    interpreter = interpreter_for(Path(args.detector_dir), args.device, args.interpreter)
    write_frames_json(output, trial=args.trial, views=views, per_view_frames=per_view_frames)

    def finish(state: str, reason: str | None, record: dict[str, Any] | None) -> int:
        manifest = build_manifest(
            args=args,
            output=output,
            views=views,
            per_view_frames=per_view_frames,
            coverage=coverage,
            videos=videos,
            interpreter=interpreter,
            worker_record=record,
            state=state,
            reason=reason,
            repository_root=repository_root,
        )
        write_manifest(output, manifest)
        if reason:
            print(f"{state}: {reason}", file=sys.stderr)
        return 0 if state == "succeeded" else 2

    if not interpreter.is_file():
        return finish(
            "blocked",
            f"detector interpreter for --device {args.device} does not exist: {interpreter}; "
            "run scripts/install_finebio_detector.sh"
            + (" with FINEBIO_CUDA=1" if args.device == "cuda" else ""),
            None,
        )
    command = worker_command(
        args,
        interpreter=interpreter,
        output=output,
        per_view_frames=per_view_frames,
        videos=videos,
    )
    (output / "worker_command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
    print(f"spawning {interpreter} for {len(views)} view(s), coverage {coverage}", flush=True)
    with (output / "worker.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            check=False,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=worker_environment(args.device),
            cwd=str(repository_root),
        )
    result_path = output / "worker_result.json"
    if not result_path.is_file():
        return finish(
            "failed", f"worker exited {completed.returncode} without worker_result.json", None
        )
    record = json.loads(result_path.read_text(encoding="utf-8"))
    state = str(record.get("state", "failed"))
    exit_code = finish(state, record.get("reason"), record)
    if exit_code == 0:
        timing = _timing(record)
        print(
            json.dumps(
                {
                    "state": state,
                    "coverage": coverage,
                    "frames": {v: len(f) for v, f in per_view_frames.items()},
                    "mean_ms_per_frame": timing["mean_ms_per_frame"],
                    "elapsed_seconds": timing["elapsed_seconds"],
                    "gpu_peak_vram_bytes": record.get("gpu_peak_vram_bytes"),
                },
                indent=2,
            )
        )
        if args.interpolate and coverage == "strided":
            filled = run_interpolate_directory(output, args.interpolation_iou)
            print(json.dumps({"interpolated_rows": filled}, indent=2))
    return exit_code


def run_interpolate(args: argparse.Namespace) -> int:
    filled = run_interpolate_directory(Path(args.directory), args.interpolation_iou)
    print(json.dumps({"interpolated_rows": filled}, indent=2))
    return 0


# --------------------------------------------------------------------------------------------
# CLI


def _add_gpu_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tf32",
        action="store_true",
        help="allow TF32 convolutions/matmuls on the GPU (~10%% faster, scores move up to ~0.03)",
    )
    parser.add_argument(
        "--allow-gpu-neighbour",
        type=int,
        action="append",
        default=[],
        metavar="PID",
        help="Let this GPU process (by PID) share the card; recorded in the manifest.",
    )
    parser.add_argument(
        "--gpu-guard",
        choices=gpu_guard.GUARD_MODES,
        default=gpu_guard.DEFAULT_GUARD_MODE,
        help="vram (default): classify neighbours and check headroom; strict: Sep 21 name rule.",
    )
    parser.add_argument(
        "--gpu-guard-profile",
        choices=tuple(gpu_guard.EXPECTED_PEAK_BYTES),
        default=gpu_guard.DEFAULT_PROFILE,
        help="Expected-peak profile for the vram guard (default unknown: 4 GiB, 6 GiB headroom).",
    )
    parser.add_argument(
        "--expected-peak-vram-bytes",
        type=int,
        default=None,
        help="Override the profile's expected peak VRAM (bytes).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="battle-finebio-detect",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="detect on raw frames of one trial (spawns the detector venv)")
    run.add_argument("--trial", required=True, help="e.g. P03_03_01")
    run.add_argument("--views", default=",".join(VIEWS), help="comma-separated subset of the six")
    run.add_argument("--start", type=int, default=None, help="first raw frame of the window")
    run.add_argument("--count", type=int, default=None, help="number of raw frames in the window")
    run.add_argument(
        "--frames",
        type=parse_frame_spec,
        default=None,
        help="explicit raw frames instead of a window: N, A-B (inclusive), A:B:S (range)",
    )
    run.add_argument("--stride-fpv", type=int, default=1, help="detect every n-th fpv frame")
    run.add_argument("--stride-fixed", type=int, default=1, help="detect every n-th fixed frame")
    run.add_argument("--model", choices=sorted(MODELS), default="dino")
    run.add_argument("--device", choices=DEVICES, default="cpu")
    run.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = torch default)")
    run.add_argument("--record-threshold", type=float, default=RECORD_THRESHOLD)
    run.add_argument("--output", type=Path, required=True, help="e.g. runs/<run>/detections")
    run.add_argument("--overwrite", action="store_true", help="replace a non-empty --output")
    run.add_argument("--run-id", default=None, help="default: the parent directory's name")
    run.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    run.add_argument("--detector-dir", type=Path, default=DEFAULT_DETECTOR_DIR)
    run.add_argument("--interpreter", default=None, help="override the detector venv interpreter")
    run.add_argument("--repository-root", type=Path, default=Path.cwd())
    run.add_argument(
        "--interpolate",
        action="store_true",
        help="after a strided run, fill the skipped frames (same as the interpolate subcommand)",
    )
    run.add_argument("--interpolation-iou", type=float, default=INTERPOLATION_IOU)
    _add_gpu_arguments(run)
    run.set_defaults(func=run_run)

    interpolate = sub.add_parser("interpolate", help="fill skipped frames of a strided run")
    interpolate.add_argument("--directory", type=Path, required=True, help="the run's --output")
    interpolate.add_argument("--interpolation-iou", type=float, default=INTERPOLATION_IOU)
    interpolate.set_defaults(func=run_interpolate)

    worker = sub.add_parser("worker", help="internal: MMDetection inference (detector venv)")
    worker.add_argument("--model", choices=sorted(MODELS), required=True)
    worker.add_argument("--device", required=True)
    worker.add_argument("--detector-dir", required=True)
    worker.add_argument("--output", required=True)
    worker.add_argument("--record-threshold", type=float, default=RECORD_THRESHOLD)
    worker.add_argument("--threads", type=int, default=0)
    worker.add_argument("--frames-file", required=True, help="the run's frames.json")
    worker.add_argument("--videos-json", required=True, help='{"view": "/path.mp4", ...}')
    _add_gpu_arguments(worker)
    worker.set_defaults(func=run_worker)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run" and args.frames is None and (args.start is None or args.count is None):
        parser.error("run needs --frames or both --start and --count")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
