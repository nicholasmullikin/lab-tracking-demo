"""Run bounded Kineo NLF-only headless inference and normalize its PKL outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from .external_smoke_import import import_kineo, sha256_file

DEFAULT_CONFIG = Path("configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_g2.json")
DEFAULT_KINEO_CONFIG = Path("configs/kineo_nlf_headless_only.yaml")
DEFAULT_KINEO_ROOT = Path("/home/nick/src/kineo")
DEFAULT_KINEO_REVISION = "03b36e31c79bd40bc8bb1ce4c9c08907140952dd"
DEFAULT_PROXY = Path("data/derived/assembly101/smoke_frames/focused_static_20s.mp4")
DEFAULT_SEQUENCE = "assembly101_focused_static_20s"
DEFAULT_SECONDS = 20.0
MAX_SECONDS = 60.0
NLF_CHECKPOINT = "checkpoints/nlf_l_multi_0.3.2.torchscript"
MOGE_MODEL = "Ruicheng/moge-2-vitl"
RTMLIB_MODEL_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "yolox_tiny_8xb8-300e_humanart-6f3252f9.zip"
)


def _git_fingerprint(repository_root: Path) -> dict[str, str]:
    head = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repository_root), "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    diff = subprocess.run(
        ["git", "-C", str(repository_root), "diff", "--stat"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    dirty_digest = hashlib.sha256(f"{status}\n{diff}".encode()).hexdigest()
    return {
        "head": head,
        "dirty": bool(status.strip()),
        "status_short": status,
        "diff_stat": diff,
        "dirty_fingerprint_sha256": dirty_digest,
    }


def _bounded_proxy(proxy_path: Path, output_path: Path, frame_count: int) -> Path:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(proxy_path),
            "-frames:v",
            str(frame_count),
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ],
        check=True,
    )
    return output_path


def _validate_pkls(native_root: Path) -> dict[str, object]:
    bboxes = pickle.loads((native_root / "bboxes_2d.pkl").read_bytes())
    keypoints = pickle.loads((native_root / "keypoints_2d.pkl").read_bytes())
    timings = pickle.loads((native_root / "stage_timings.pkl").read_bytes())
    intrinsics_path = native_root / "camera_intrinsics.pkl"
    intrinsics: dict[str, object] = {"annotations": []}
    if intrinsics_path.is_file():
        try:
            intrinsics = pickle.loads(intrinsics_path.read_bytes())
        except ModuleNotFoundError:
            intrinsics = {"annotations": [], "unpickle_requires_kineo_module": True}
    bbox_rows = bboxes["annotations"]
    keypoint_rows = keypoints["annotations"]
    if len(bbox_rows) != len(keypoint_rows):
        raise ValueError("bbox and keypoint annotation counts differ")
    frame_indices = sorted({int(row["frame_idx"]) for row in bbox_rows})
    xy = keypoint_rows[0]["xy"]
    names = keypoints["metadata"]["formats"][0]["keypoints_names"]
    finite_xy = sum(
        1
        for row in keypoint_rows
        for point in row["xy"][:55]
        if all(map(lambda v: abs(v) < 1e9, point))
    )
    return {
        "bbox_rows": len(bbox_rows),
        "keypoint_rows": len(keypoint_rows),
        "first_frame": frame_indices[0] if frame_indices else None,
        "last_frame": frame_indices[-1] if frame_indices else None,
        "distinct_frames": len(frame_indices),
        "nlf_joint_count": len(xy),
        "nlf_body_joint_count": 55,
        "body_joint_names_head": names[:5],
        "finite_body_xy_values": finite_xy,
        "stage_timings": timings["annotations"],
        "intrinsics_keys": list(intrinsics["annotations"][0].keys())
        if intrinsics.get("annotations")
        else [],
    }


def run_kineo_nlf(args: argparse.Namespace) -> Path:
    repository_root = args.repository_root.resolve()
    kineo_root = args.kineo_root.resolve()
    frame_count = round(args.seconds * 30)
    run_id = args.run_id or (
        f"kineo-nlf-headless-{int(args.seconds)}s-{datetime.now(UTC).strftime('%Y%m%dt%H%M%Sz')}"
    )
    proxy_path = (repository_root / args.proxy).resolve()
    bounded_proxy_dir = repository_root / "data/logs/kineo_cache/bounded_inputs"
    bounded_proxy_dir.mkdir(parents=True, exist_ok=True)
    bounded_proxy = bounded_proxy_dir / f"{args.sequence_name}_{frame_count}f.mp4"
    if not bounded_proxy.exists():
        _bounded_proxy(proxy_path, bounded_proxy, frame_count)

    git_state = _git_fingerprint(kineo_root)
    checkpoint_path = kineo_root / NLF_CHECKPOINT
    provenance = {
        "run_id": run_id,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "kineo_repository": str(kineo_root),
        "kineo_git": git_state,
        "battle_config": str(args.config),
        "kineo_config": str(args.kineo_config),
        "sequence_name": args.sequence_name,
        "proxy_uri": proxy_path.relative_to(repository_root).as_posix(),
        "proxy_sha256": sha256_file(proxy_path),
        "bounded_proxy_uri": bounded_proxy.relative_to(repository_root).as_posix(),
        "bounded_proxy_sha256": sha256_file(bounded_proxy),
        "frame_count": frame_count,
        "person_selection": {
            "method": "rtmlib_best_bbox_only",
            "description": (
                "Per frame keep highest detector-confidence person bbox (not largest area)."
            ),
            "frame_step": 5,
            "bbox_thr": 0.3,
            "nms_iou_thr": 0.65,
        },
        "model_identities": {
            "nlf_torchscript": {
                "path": NLF_CHECKPOINT,
                "sha256": sha256_file(checkpoint_path) if checkpoint_path.is_file() else None,
            },
            "moge_model": MOGE_MODEL,
            "rtmlib_bbox_model_url": RTMLIB_MODEL_URL,
        },
    }
    if not args.skip_inference:
        env = os.environ.copy()
        env["OUTPUT_ROOT_DIR"] = str(repository_root / "runs/kineo")
        env["CACHE_ROOT_DIR"] = str(repository_root / "data/logs/kineo_cache")
        command = [
            "pixi",
            "run",
            "python",
            "-m",
            "kineo.demo.offline.demo",
            "--config-file",
            str((repository_root / args.kineo_config).resolve()),
            "--sequence-name",
            args.sequence_name,
            "--shared-intrinsics",
            str(bounded_proxy),
        ]
        started = time.perf_counter()
        subprocess.run(command, cwd=kineo_root, env=env, check=True)
        provenance["inference_elapsed_seconds"] = time.perf_counter() - started

    native_root = (
        repository_root
        / "runs/kineo/infer_nlf_headless_only/offline_demo/annotations"
        / args.sequence_name
    )
    validation = _validate_pkls(native_root)

    import_args = argparse.Namespace(
        repository_root=repository_root,
        config=args.config,
        output_root=args.output_root,
        run_id=run_id,
        native=native_root.relative_to(repository_root),
        video=bounded_proxy.relative_to(repository_root),
        image=Path("data/derived/assembly101/smoke_frames/focused_static_frame0.jpg"),
    )
    manifest_path = import_kineo(import_args)
    run_directory = manifest_path.parent
    provenance["manifest_uri"] = manifest_path.relative_to(repository_root).as_posix()
    provenance["validation"] = validation
    provenance["classification"] = "kineo_nlf_only_partial"
    (run_directory / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    (run_directory / "pkl_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--kineo-config", type=Path, default=DEFAULT_KINEO_CONFIG)
    parser.add_argument("--kineo-root", type=Path, default=DEFAULT_KINEO_ROOT)
    parser.add_argument("--proxy", type=Path, default=DEFAULT_PROXY)
    parser.add_argument("--sequence-name", default=DEFAULT_SEQUENCE)
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--run-id")
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Normalize and export from existing PKLs without rerunning Kineo.",
    )
    args = parser.parse_args()
    if args.seconds <= 0 or args.seconds > MAX_SECONDS:
        raise SystemExit(f"--seconds must be within (0, {MAX_SECONDS}]")
    print(run_kineo_nlf(args), file=sys.stderr)


if __name__ == "__main__":
    main()
