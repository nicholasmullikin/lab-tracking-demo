"""Battle-owned Kineo runner: build the offline pipeline with our dataset cameras as GT.

Executed under Kineo's pixi environment from the Kineo checkout, never from the battle env:

    cd /home/nick/src/kineo && pixi run python \
        /home/nick/src/battle/scripts/kineo_multiview_runner.py \
        --config-file <run>/kineo_selfcal.yaml --sequence-name <name> \
        --gt-annotations-dir <run>/gt_annotations --runtime-json <run>/kineo_runtime.json \
        C10095=<run>/trims/C10095_first_minute_1800f.mp4 C10115=... [--validate-only]

It mirrors `kineo/demo/offline/demo.py` but (a) names each view after its camera, (b) passes
the `camera_intrinsics` / `camera_extrinsics` PKLs written by `battle-kineo-multiview prepare`
as `gt_annotations` (the stock CLI passes `{}`), which is what `transfer_gt_annotations` and
`rerun_export` read, and (c) records wall time and torch's peak GPU memory.  Nothing in Kineo
is modified.

`--validate-only` is a CPU dry run for the queue: it loads and resolves the YAML, imports every
stage class, instantiates every runtime config dataclass, checks constructor signatures and
referenced model files, fully instantiates only the stages that load no model, parses the GT
PKLs through Kineo's own annotation classes, and checks that every video opens with the same
frame count.  Run it with `CUDA_VISIBLE_DEVICES=""` so nothing can reach the GPU.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

MODEL_FREE_STAGES = {
    "TransferGroundTruthAnnotationsStage",
    "GlobalTimeResamplingStage",
    "KeypointsPairsSamplingStage",
    "SfMCameraExtrinsicsInitializationStage",
    "BundleAdjustmentSamplingStage",
    "BundleAdjustmentStage",
    "MVSTriangulationStage",
    "SMPLGlobalScaleEstimationStage",
    "GlobalScaleApplicationStage",
    "SceneReorientationStage",
    "BundleAdjustmentHistoryRerunExportStage",
    "ExportBvhStage",
    "RerunExportStage",
    "AnnotationsExportStage",
}
MODEL_PATH_KEYS = ("torchscript_model_path", "smpl_model_path", "joint_regressor_path")


def parse_views(specs: list[str]) -> list[tuple[str, str]]:
    views = []
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"video argument must be VIEW=PATH, got {spec!r}")
        view, path = spec.split("=", 1)
        views.append((view, path))
    return views


def load_gt_annotations(gt_dir: Path) -> dict:
    import pickle

    from kineo.annotations.camera_extrinsics import CameraExtrinsicsAnnotations
    from kineo.annotations.camera_intrinsics import CameraIntrinsicsAnnotations

    with (gt_dir / "camera_intrinsics.pkl").open("rb") as handle:
        intrinsics = CameraIntrinsicsAnnotations.from_dict(pickle.load(handle))
    with (gt_dir / "camera_extrinsics.pkl").open("rb") as handle:
        extrinsics = CameraExtrinsicsAnnotations.from_dict(pickle.load(handle))
    return {"camera_intrinsics": intrinsics, "camera_extrinsics": extrinsics}


def validate(config_file: str, gt_dir: Path, views: list[tuple[str, str]]) -> dict:
    import cv2
    import hydra
    import torch
    from kineo.pipeline.pipeline import PipelineStage
    from omegaconf import OmegaConf

    report: dict = {
        "cuda_visible": torch.cuda.is_available(),
        "torch": torch.__version__,
        "stages": [],
        "errors": [],
    }
    cfg = OmegaConf.load(config_file)
    cfg.shared_intrinsics = False
    cfg.use_cache = False
    OmegaConf.resolve(cfg)
    report["shared_intrinsics"] = bool(cfg.shared_intrinsics)
    report["output_root_dir"] = str(cfg.output_root_dir)
    for key, node in cfg.pipeline.stages.items():
        entry: dict = {"stage": key, "order": int(node.order)}
        try:
            cls = hydra.utils.get_class(node._target_)
            if not issubclass(cls, PipelineStage):
                raise TypeError(f"{cls.__name__} is not a PipelineStage")
            entry["class"] = cls.__name__
            kwargs = {k: v for k, v in node.items() if k != "_target_"}
            if "runtime_cfg" in kwargs:
                runtime_cfg = hydra.utils.instantiate(node.runtime_cfg)
                entry["runtime_cfg"] = type(runtime_cfg).__name__
                kwargs["runtime_cfg"] = runtime_cfg
            signature = inspect.signature(cls.__init__)
            parameters = signature.parameters
            accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in parameters.values())
            unknown = [k for k in kwargs if k not in parameters and not accepts_kwargs]
            missing = [
                name
                for name, p in parameters.items()
                if name != "self"
                and p.default is p.empty
                and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
                and name not in kwargs
            ]
            if unknown or missing:
                raise TypeError(f"constructor mismatch: unknown {unknown}, missing {missing}")
            for path_key in MODEL_PATH_KEYS:
                if path_key in kwargs and not Path(str(kwargs[path_key])).exists():
                    raise FileNotFoundError(f"{path_key}={kwargs[path_key]} does not exist")
            if cls.__name__ in MODEL_FREE_STAGES:
                stage = hydra.utils.instantiate(node)
                entry["instantiated"] = True
                entry["name"] = stage.name
            else:
                entry["instantiated"] = False
                entry["reason"] = "loads a model in __init__; signature and paths checked only"
        except Exception as error:  # noqa: BLE001 - every failure belongs in the report
            entry["error"] = f"{type(error).__name__}: {error}"
            report["errors"].append(f"{key}: {entry['error']}")
        report["stages"].append(entry)

    try:
        gt = load_gt_annotations(gt_dir)
        report["gt_views"] = gt["camera_extrinsics"].views_ids
        report["gt_intrinsics_views"] = gt["camera_intrinsics"].views_ids
        if set(report["gt_views"]) != {view for view, _ in views}:
            raise ValueError(f"GT views {report['gt_views']} != argv views {[v for v, _ in views]}")
        first = gt["camera_intrinsics"].first_or_default()
        report["gt_distortion_model"] = first.distortion_model.value
        report["gt_resolution_hw"] = list(first.resolution_hw)
    except Exception as error:  # noqa: BLE001
        report["errors"].append(f"gt_annotations: {type(error).__name__}: {error}")

    counts = {}
    for view, path in views:
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            report["errors"].append(f"{view}: cannot open {path}")
            continue
        counts[view] = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
    report["video_frame_counts"] = counts
    if counts and len(set(counts.values())) != 1:
        report["errors"].append(f"frame counts differ: {counts}")
    report["ok"] = not report["errors"]
    return report


def run(
    config_file: str,
    sequence_name: str,
    gt_dir: Path,
    views: list[tuple[str, str]],
    batch_size: int | None,
    runtime_json: Path | None,
) -> None:
    import torch
    from kineo.datasets.keypoints_sequence_dataset import ViewInput
    from kineo.io.frame_sequence_loader import VideoLoader
    from kineo.pipeline.pipeline import Pipeline
    from omegaconf import OmegaConf

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Torch {torch.__version__} on {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)
        torch.cuda.reset_peak_memory_stats(device)

    cfg = OmegaConf.load(config_file)
    if batch_size is not None:
        cfg.batch_size = batch_size
    cfg.shared_intrinsics = False
    cfg.use_cache = False

    started = time.perf_counter()
    pipeline = Pipeline.build_pipeline_from_config(cfg, device)
    built = time.perf_counter()
    view_inputs = [
        ViewInput(
            view_id=view,
            frame_loader=VideoLoader(video_path=path, device=device),
            audio_loader=None,
        )
        for view, path in views
    ]
    frame_counts = {v["view_id"]: int(v["frame_loader"].n_frames) for v in view_inputs}
    print(f"Views: {frame_counts}", flush=True)
    gt_annotations = load_gt_annotations(gt_dir)
    print(f"GT cameras: {gt_annotations['camera_extrinsics'].views_ids}", flush=True)

    pipeline.run(
        sequence_name=sequence_name,
        views=view_inputs,
        annotations={},
        gt_annotations=gt_annotations,
    )
    finished = time.perf_counter()

    record = {
        "sequence_name": sequence_name,
        "config_file": config_file,
        "views": dict(views),
        "frame_counts": frame_counts,
        "device": str(device),
        "torch": torch.__version__,
        "elapsed_seconds": finished - started,
        "pipeline_build_seconds": built - started,
        "pipeline_run_seconds": finished - built,
        "gpu_peak_vram_bytes": (
            int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
        ),
        "gpu_peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cwd": os.getcwd(),
    }
    print(json.dumps(record), flush=True)
    if runtime_json is not None:
        runtime_json.parent.mkdir(parents=True, exist_ok=True)
        runtime_json.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--sequence-name", required=True)
    parser.add_argument("--gt-annotations-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--runtime-json", type=Path, default=None)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("videos", nargs="+", help="VIEW=PATH, view ids become Kineo view_ids")
    args = parser.parse_args()
    views = parse_views(args.videos)
    if args.validate_only:
        report = validate(args.config_file, args.gt_annotations_dir, views)
        print(json.dumps(report), flush=True)
        return 0 if report["ok"] else 1
    run(
        args.config_file,
        args.sequence_name,
        args.gt_annotations_dir,
        views,
        args.batch_size,
        args.runtime_json,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
