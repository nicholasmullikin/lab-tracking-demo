"""Run the paired blue-pipette experiment on every native frame of all six videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import cv2

from .muggled_worker import MAX_SIDE_LENGTH
from .pipette_line_review import VARIANTS, choose_mask
from .pipette_video_metrics import aggregate, observe_view

CORRECTIONS = {
    "T1": (0, 500),
    "T2": (0, 100),
    "T3": (0, 300),
    "T4": (0, 200),
    "T5": (0,),
    "fpv": (0,),
}


def build_config(base: Path, checkout: Path, checkpoint: Path) -> dict:
    labels = json.loads((base / "every100-review/labels.json").read_text())
    ratings = json.loads((base / "point-prompt-study/ratings.json").read_text())
    sources = json.loads((base / "frame-150-review/sources.json").read_text())
    views = {}
    for view, corrections in CORRECTIONS.items():
        video = Path(sources[view]["raw_video"]).resolve()
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise ValueError(f"Cannot open {video}")
        record = {
            "video": str(video),
            "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": capture.get(cv2.CAP_PROP_FPS),
            "size_wh": [
                int(capture.get(k)) for k in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)
            ],
        }
        capture.release()
        groups, aliases, identities = {}, {}, {}
        for variant in VARIANTS:
            entries = {}
            for frame in corrections:
                path, provenance = choose_mask(base, labels, ratings, frame, view, variant)
                entries[str(frame)] = {
                    "path": str(path.resolve()),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "provenance": provenance,
                }
            identity = tuple(x["sha256"] for x in entries.values())
            if identity not in identities:
                group = f"g{len(groups)}"
                identities[identity] = group
                groups[group] = {"corrections": entries, "variants": []}
            group = identities[identity]
            aliases[variant] = group
            groups[group]["variants"].append(variant)
        views[view] = {**record, "groups": groups, "aliases": aliases}
    if len({(x["frame_count"], x["fps"]) for x in views.values()}) != 1:
        raise ValueError("Native video lengths and rates must match")
    return {
        "checkpoint": str(checkpoint.resolve()),
        "muggled_checkout": str(checkout.resolve()),
        "max_side_length": MAX_SIDE_LENGTH,
        "native_fps": views["T1"]["fps"],
        "sampling": "every native video frame",
        "views": views,
        "correction_policy": (
            "All views seeded at raw frame 0; matched control/study corrections only at "
            "the seven studied camera images; unassisted propagation thereafter."
        ),
        "corrections_by_view": {v: list(f) for v, f in CORRECTIONS.items()},
    }


def run_view(root: Path, view: str, model_python: Path) -> dict:
    directory = root / "native" / view
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "failed.json").unlink(missing_ok=True)
    command = [
        str(model_python),
        str(Path(__file__).with_name("pipette_video_worker.py")),
        "--config",
        str(root / "config.json"),
        "--view",
        view,
        "--out",
        str(directory),
        "--resume",
    ]
    start = perf_counter()
    with (directory / "worker.log").open("a") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    record = {"view": view, "exit_code": result.returncode, "wall_seconds": perf_counter() - start}
    if result.returncode:
        (directory / "failed.json").write_text(json.dumps(record))
        raise RuntimeError(record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=Path("runs/finebio-pipette-improvement-20260930")
    )
    parser.add_argument(
        "--model-python",
        type=Path,
        default=Path("/home/nick/.pyenv/versions/muggled_sam/bin/python"),
    )
    parser.add_argument("--muggled-checkout", type=Path, default=Path("/home/nick/src/muggled_sam"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/home/nick/src/muggled_sam/model_weights/sam3.1_multiplex.pt"),
    )
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    root = (args.base / "full-video-line-comparison").resolve()
    root.mkdir(parents=True, exist_ok=True)
    config = build_config(args.base.resolve(), args.muggled_checkout, args.checkpoint)
    config_path = root / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError(
            "Existing experiment config differs; archive its outputs before restarting"
        )
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    with (
        ThreadPoolExecutor(max_workers=6) as observers,
        ThreadPoolExecutor(max_workers=args.workers) as workers,
    ):
        obs = [observers.submit(observe_view, root, v) for v in CORRECTIONS]
        inference = [
            workers.submit(run_view, root, v, args.model_python)
            for v in ("T1", "T2", "T4", "T3", "fpv", "T5")
        ]
        results = []
        for future in as_completed(inference):
            results.append(future.result())
            (root / "worker-results.json").write_text(json.dumps(results, indent=2) + "\n")
        for future in obs:
            future.result()
    aggregate(root)
    from .pipette_video_agreement import evaluate
    from .pipette_video_review import export

    evaluate(root, args.base / "every100-review/labels.json")
    export(root)


if __name__ == "__main__":
    main()
