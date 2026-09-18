#!/usr/bin/env python
"""LM-EEC inference driver for the Track 7 keyframe pairs (runs under the LM-EEC venv).

Reads `<run_dir>/pairs.json` written by `battle-egoexo-correspondence prepare`, loads each
direction's checkpoint once, and for every object of every direction feeds the keyframes that
carry a query mask through `SAM2VideoPredictor.init_state` / `propagate_in_video` of
`sam2.sam2_correspondence_predictor`.  In the predictor's own vocabulary `ego_*` is the view
that carries the query mask and `exo_*` the view being predicted, whatever the cameras are.

Outputs, relative to the run directory:
  predictions/<direction>/<object>/<key>.png   predicted target-view mask (0/255, proxy size)
  predictions.json                              one row per pair with the model's own confidence
                                                (predicted IoU, object score) plus runtime facts

The predictor returns 480x480 logits (its `_get_orig_video_res_output1` resizes to the model
square, not to the frame), so the driver resizes the logits to the target frame's size itself.
Only stdlib, torch, numpy and Pillow are imported: this file never sees the battle package.

Usage:
  python scripts/lm_eec_driver.py --run-dir <run_dir> [--mode sequence|independent]
                                  [--device cuda] [--cpu-smoke]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) > 0


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").save(path)


def image_size_hw(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        width, height = image.size
    return height, width


def install_cpu_shim(torch) -> None:
    """Route the predictor's hard-coded `.to("cuda")` calls to the CPU (smoke runs only)."""
    original_to = torch.Tensor.to

    def patched_to(self, *args, **kwargs):
        args = tuple("cpu" if isinstance(a, str) and a.startswith("cuda") else a for a in args)
        if isinstance(kwargs.get("device"), str) and kwargs["device"].startswith("cuda"):
            kwargs["device"] = "cpu"
        return original_to(self, *args, **kwargs)

    torch.Tensor.to = patched_to  # type: ignore[method-assign]
    torch.cuda.synchronize = lambda *a, **k: None  # type: ignore[assignment]


def git_head(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--lm-eec-root", type=Path, default=Path("/home/nick/src/LM-EEC"))
    parser.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    parser.add_argument("--mode", choices=("sequence", "independent"), default="sequence")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-thresh", type=float, default=0.0)
    parser.add_argument(
        "--cpu-smoke",
        action="store_true",
        help="shim the model's hard-coded cuda calls so the whole path runs on CPU",
    )
    parser.add_argument(
        "--objects", nargs="*", default=None, help="restrict to these object names (smoke)"
    )
    parser.add_argument(
        "--max-keyframes", type=int, default=None, help="restrict to the first N keyframes (smoke)"
    )
    args = parser.parse_args()

    started = time.perf_counter()
    run_dir = args.run_dir.resolve()
    pairs = json.loads((run_dir / "pairs.json").read_text(encoding="utf-8"))
    os.chdir(args.lm_eec_root)  # Hydra resolves the config module relative to the package

    import torch

    if args.cpu_smoke:
        install_cpu_shim(torch)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; refusing to fall back silently", file=sys.stderr)
        return 3
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    from sam2.build_sam import build_sam2_video_predictor_ego

    keyframes = pairs["keyframes"]
    if args.max_keyframes is not None:
        keyframes = keyframes[: args.max_keyframes]
    predictions: list[dict] = []
    skipped: list[str] = []
    load_seconds: dict[str, float] = {}
    checkpoints: dict[str, str] = {}
    first_output: float | None = None

    for direction in pairs["directions"]:
        name = direction["direction"]
        if direction.get("skipped_reason"):
            skipped.append(f"{name}: {direction['skipped_reason']}")
            continue
        objects = [o for o in direction["objects"] if args.objects is None or o in args.objects]
        query_key = "exo_query_masks" if name == "exo_to_ego" else "ego_query_masks"
        checkpoint = Path(direction["checkpoint"])
        if not checkpoint.is_file():
            skipped.append(f"{name}: checkpoint missing at {checkpoint}")
            continue
        checkpoints[name] = f"{checkpoint} sha256 {sha256(checkpoint)}"
        t_load = time.perf_counter()
        predictor = build_sam2_video_predictor_ego(
            config_file=args.config,
            ckpt_path=str(checkpoint),
            device=str(device),
            hydra_overrides_extra=["++model.non_overlap_masks=false"],
        )
        load_seconds[name] = time.perf_counter() - t_load
        source_dir = run_dir / direction["source_frames_dir"]
        target_dir = run_dir / direction["target_frames_dir"]
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else torch.autocast(device_type="cpu", enabled=False)
        )

        for obj in objects:
            keyed = [
                (pair["key"], run_dir / pair[query_key][obj])
                for pair in keyframes
                if pair.get(query_key, {}).get(obj) is not None
            ]
            if not keyed:
                skipped.append(f"{name}/{obj}: no query masks on any keyframe")
                continue
            groups = [keyed] if args.mode == "sequence" else [[k] for k in keyed]
            for group in groups:
                keys = [k for k, _ in group]
                query_masks = {k: read_mask(p) for k, p in group}
                target_hw = image_size_hw(target_dir / f"{keys[0]}.jpg")
                with torch.inference_mode(), autocast:
                    state = predictor.init_state(
                        ego_video_path=str(source_dir),
                        exo_video_path=str(target_dir),
                        video_frames=keys,
                        offload_video_to_cpu=False,
                        async_loading_frames=False,
                    )
                    t_frame = time.perf_counter()
                    for frame_idx, _obj_ids, logits in predictor.propagate_in_video(
                        state, query_masks, keys
                    ):
                        key = keys[frame_idx]
                        logits = logits.float()
                        resized = torch.nn.functional.interpolate(
                            logits, size=target_hw, mode="bilinear", align_corners=False
                        )[0, 0]
                        mask = (resized > args.score_thresh).cpu().numpy()
                        out = state["exo_output_dict"]["non_cond_frame_outputs"][frame_idx]
                        iou = out.get("iou")
                        score_logit = out.get("object_score_logits")
                        iou_value = float(iou.reshape(-1)[0]) if iou is not None else None
                        score_value = (
                            float(score_logit.reshape(-1)[0]) if score_logit is not None else None
                        )
                        mask_path = run_dir / "predictions" / name / obj / f"{key}.png"
                        write_mask(mask_path, mask)
                        now = time.perf_counter()
                        if first_output is None:
                            first_output = now - started
                        predictions.append(
                            {
                                "schema_version": "1.0",
                                "direction": name,
                                "object": obj,
                                "key": key,
                                "mask_uri": mask_path.relative_to(run_dir).as_posix(),
                                "mask_area_px": int(mask.sum()),
                                "empty": not bool(mask.any()),
                                "iou_prediction": iou_value,
                                "object_score_logit": score_value,
                                "object_score": (
                                    float(torch.sigmoid(torch.tensor(score_value)))
                                    if score_value is not None
                                    else None
                                ),
                                "max_logit": float(resized.max()),
                                "seconds": now - t_frame,
                            }
                        )
                        t_frame = now
                        print(
                            f"{name} {obj} {key}: area {int(mask.sum())} px, iou_pred {iou_value}, "
                            f"score_logit {score_value}",
                            flush=True,
                        )
                del state
        del predictor
        if device.type == "cuda":
            torch.cuda.empty_cache()

    runtime = {
        "schema_version": "1.0",
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "mode": args.mode,
        "cpu_smoke": bool(args.cpu_smoke),
        "lm_eec_head": git_head(args.lm_eec_root),
        "model_load_seconds": load_seconds,
        "time_to_first_output_seconds": first_output,
        "total_seconds": time.perf_counter() - started,
        "gpu_peak_vram_reserved_bytes": (
            int(torch.cuda.max_memory_reserved()) if device.type == "cuda" else None
        ),
        "gpu_peak_vram_allocated_bytes": (
            int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None
        ),
    }
    payload = {
        "schema_version": "1.0",
        "manifest_kind": "egoexo_correspondence_predictions",
        "run_id": pairs["run_id"],
        # Python 3.10 venv: no datetime.UTC alias.
        "predicted_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),  # noqa: UP017
        "pairs_sha256": sha256(run_dir / "pairs.json"),
        "checkpoints": checkpoints,
        "predictions": predictions,
        "skipped": skipped,
        "runtime": runtime,
    }
    (run_dir / "predictions.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {len(predictions)} predictions to {run_dir / 'predictions.json'} in "
        f"{runtime['total_seconds']:.1f} s (first output {first_output}); skipped {len(skipped)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
