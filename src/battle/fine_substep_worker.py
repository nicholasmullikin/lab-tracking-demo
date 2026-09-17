"""Pinned OpenCLIP crop scoring for the fine substep experiment."""

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
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--crop-manifest", type=Path, required=True)
    parser.add_argument("--label-contract", type=Path, required=True)
    parser.add_argument("--openclip-checkpoint", type=Path, required=True)
    parser.add_argument("--openclip-cache-dir", type=Path)
    parser.add_argument("--keep-percentile", type=float, default=0.3)
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

        contract_payload = json.loads(args.label_contract.read_text(encoding="utf-8"))
        if contract_payload.get("provenance_tag") != "agent_authored_visual_review":
            raise RuntimeError("label contract must be agent_authored_visual_review")
        substeps = contract_payload["substeps"]
        crop_rows = json.loads(args.crop_manifest.read_text(encoding="utf-8"))
        if not crop_rows:
            raise RuntimeError("crop manifest is empty")

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

        frame_embeddings: list[list[float]] = []
        frame_indices: list[int] = []
        for row in crop_rows:
            frame_index = int(row["analysis_frame_index"])
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"failed to decode frame {frame_index}")
            box = row["workspace_box"]
            crop = frame[box["y0"] : box["y1"], box["x0"] : box["x1"]]
            if crop.size == 0:
                raise RuntimeError(f"empty crop at frame {frame_index}")
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            image_tensor = preprocess(Image.fromarray(rgb)).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model.encode_image(image_tensor)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            frame_embeddings.append(embedding.squeeze(0).cpu().numpy().tolist())
            frame_indices.append(frame_index)
        capture.release()

        substep_prompt_scores: list[list[float]] = []
        substep_text_embeddings: list[np.ndarray] = []
        prompt_records: list[dict[str, object]] = []
        for substep in substeps:
            prompt_vectors = []
            for prompt in substep["prompts"]:
                tokens = tokenizer([prompt]).to(device)
                with torch.no_grad():
                    text_embedding = model.encode_text(tokens)
                    text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
                prompt_vectors.append(text_embedding.squeeze(0).cpu().numpy())
                prompt_records.append(
                    {"substep_id": substep["substep_id"], "prompt": prompt},
                )
            text_array = np.stack(prompt_vectors)
            video_array = np.stack([np.asarray(row) for row in frame_embeddings])
            cosine = text_array @ video_array.T
            max_over_prompts = cosine.max(axis=0)
            substep_prompt_scores.append(max_over_prompts.tolist())
            ensemble = text_array.mean(axis=0)
            ensemble = ensemble / np.linalg.norm(ensemble)
            substep_text_embeddings.append(ensemble)

        score_matrix = np.stack(substep_prompt_scores).T
        text_array = np.stack(substep_text_embeddings)
        video_array = np.stack([np.asarray(row) for row in frame_embeddings])
        cost_matrix = 1.0 - (text_array @ video_array.T)
        drop_costs = np.quantile(cost_matrix, args.keep_percentile, axis=0)
        drop_cost, _path, _x_dropped = drop_dtw(
            cost_matrix,
            drop_costs,
            exclusive=True,
            contiguous=True,
            return_labels=False,
        )
        drop_labels = drop_dtw(
            cost_matrix,
            drop_costs,
            exclusive=True,
            contiguous=True,
            return_labels=True,
        )

        payload = {
            "clip_model": "ViT-B-32/openai",
            "openclip_checkpoint": str(args.openclip_checkpoint),
            "frame_indices": frame_indices,
            "frame_embeddings": frame_embeddings,
            "clip_score_matrix": score_matrix.tolist(),
            "drop_dtw_cost": float(drop_cost),
            "drop_dtw_labels": [int(label) for label in drop_labels],
            "prompt_records": prompt_records,
        }
        (run_directory / "worker_scores.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
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
