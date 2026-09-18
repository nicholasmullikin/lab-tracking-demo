"""CLIP embeddings plus Drop-DTW alignment for the wilor interpreter environment."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from time import perf_counter
from typing import Any

DROP_DTW_SOURCE = Path("/home/nick/src/Drop-DTW")


def _write_result(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--transcript-json", type=Path, required=True)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--keep-percentile", type=float, default=0.3)
    parser.add_argument("--openclip-checkpoint", type=Path, required=True)
    parser.add_argument("--openclip-cache-dir", type=Path)
    args = parser.parse_args()

    run_directory = args.run_directory
    run_directory.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    state = "succeeded"
    reason = ""

    try:
        import cv2
        import numpy as np
        import open_clip
        import torch
        from PIL import Image

        sys.path.insert(0, str(DROP_DTW_SOURCE))
        from dp.exact_dp import drop_dtw  # noqa: E402

        transcript = json.loads(args.transcript_json.read_text(encoding="utf-8"))
        steps = transcript["steps"]
        if not steps:
            raise RuntimeError("transcript contains no steps for the requested interval")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="openai",
            cache_dir=str(args.openclip_cache_dir) if args.openclip_cache_dir else None,
            device=device,
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        model.eval()

        capture = cv2.VideoCapture(str(args.video))
        if not capture.isOpened():
            raise RuntimeError(f"could not open video: {args.video}")
        video_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        frame_stride = max(int(round(video_fps / args.sample_fps)), 1)

        frame_features: list[list[float]] = []
        frame_indices: list[int] = []
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % frame_stride == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_tensor = preprocess(Image.fromarray(rgb)).unsqueeze(0).to(device)
                with torch.no_grad():
                    embedding = model.encode_image(image_tensor)
                    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                frame_features.append(embedding.squeeze(0).cpu().numpy().tolist())
                frame_indices.append(frame_index)
            frame_index += 1
        capture.release()

        text_features = []
        for step in steps:
            tokens = tokenizer([step["action"]]).to(device)
            with torch.no_grad():
                embedding = model.encode_text(tokens)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            text_features.append(embedding.squeeze(0).cpu().numpy())
        text_array = np.stack(text_features)
        video_array = np.stack([np.asarray(row) for row in frame_features])
        cost_matrix = 1.0 - (text_array @ video_array.T)
        drop_costs = np.quantile(cost_matrix, args.keep_percentile, axis=0)
        min_cost, _path, _x_dropped = drop_dtw(
            cost_matrix,
            drop_costs,
            exclusive=True,
            contiguous=True,
            return_labels=False,
        )
        labels = drop_dtw(
            cost_matrix,
            drop_costs,
            exclusive=True,
            contiguous=True,
            return_labels=True,
        )
        intervals = []
        for step_index, step in enumerate(steps):
            matched_indices = [
                frame_indices[video_index]
                for video_index, label in enumerate(labels)
                if int(label) == step_index + 1
            ]
            intervals.append(
                {
                    "action": step["action"],
                    "annotation_start_frame": step["annotation_start_frame"],
                    "annotation_end_frame": step["annotation_end_frame"],
                    "matched_analysis_frames": matched_indices,
                    "matched_seconds": [round(index / video_fps, 3) for index in matched_indices],
                }
            )
        artifact = {
            "transcript_source": transcript,
            "clip_model": "ViT-B-32/openai",
            "openclip_checkpoint": str(args.openclip_checkpoint),
            "sample_fps": args.sample_fps,
            "keep_percentile": args.keep_percentile,
            "alignment_cost": float(min_cost),
            "intervals": intervals,
            "frame_indices": frame_indices,
            "weak_supervision_note": (
                "Ordered text derives from Assembly101 coarse ground-truth annotations; "
                "alignment cost and intervals are exploratory weak supervision only."
            ),
        }
        (run_directory / "alignment.json").write_text(json.dumps(artifact, indent=2) + "\n")
    except Exception as error:  # noqa: BLE001
        state = "failed"
        reason = f"{type(error).__name__}: {error}"
        (run_directory / "worker.traceback.log").write_text(traceback.format_exc())

    _write_result(
        run_directory / "worker_result.json",
        {
            "state": state,
            "reason": reason,
            "elapsed_seconds": perf_counter() - started,
        },
    )


if __name__ == "__main__":
    main()
