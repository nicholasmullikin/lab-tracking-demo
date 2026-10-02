"""Staged, fixed-slot pipette study using the existing multiplex worker.

All outputs are isolated from production and earlier experiments. Seed selections
must be reviewed before tracking; later corrective prompts are never supplied.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

ROOT = Path("runs/pipette-multiplex-20261002")
CHECKOUT = Path(os.environ.get("BATTLE_MUGGLED_CHECKOUT", str(Path.home() / "src/muggled_sam")))
MODEL_PYTHON = Path(
    os.environ.get(
        "BATTLE_MODEL_PYTHON", str(Path.home() / ".pyenv/versions/muggled_sam/bin/python")
    )
)
MODEL = CHECKOUT / "model_weights/sam3.1_multiplex.pt"
LABELS = ("blue_pipette", "yellow_pipette", "red_pipette", "multichannel_pipette")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def launch(command: list[str], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "worker_command.json", command)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(CHECKOUT), env.get("PYTHONPATH", "")))
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    with (directory / "worker.log").open("a") as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def replay(root: Path) -> None:
    command = json.loads((root / "recovered-command.json").read_text())
    directory = (root / "replay-T2").resolve()
    command[command.index("--run-directory") + 1] = str(directory)
    command[command.index("--max-frames") + 1] = "30"
    launch(command, directory)


def decode_seeds(root: Path) -> None:
    """Use the established full-frame calibration decoder and its review overlays."""
    import sys

    sys.path.insert(0, str(CHECKOUT))
    sys.path.insert(0, str(Path(__file__).parent))
    import cv2
    import torch
    from muggled_calibration_worker import _decode_rectangle
    from muggled_sam.make_sam import make_sam_from_state_dict

    prompts = json.loads((root / "seeds/prompts.json").read_text())
    results = root / "seeds/results"
    (results / "masks").mkdir(parents=True, exist_ok=True)
    model = make_sam_from_state_dict(MODEL).get_interactive_context()
    model.to(device="cuda", dtype=torch.bfloat16)

    class PointOnly:
        def encode_prompts(self, boxes, fg, bg):
            return model.encode_prompts([], fg, bg)

        def generate_masks(self, *args):
            return model.generate_masks(*args)

    manifest = {}
    with torch.inference_mode():
        for view, entries in prompts.items():
            image = cv2.imread(str(root / f"seeds/{view}-original.png"))
            h, w = image.shape[:2]
            encoded = model.encode_image(image, 1280, True)
            manifest[view] = {}
            for label, entry in entries.items():
                if entry.get("absent"):
                    manifest[view][label] = entry
                    continue
                result = _decode_rectangle(
                    interactive_model=PointOnly() if entry.get("points_only") else model,
                    encoded_image=encoded,
                    frame=image,
                    prompt_box=dict(zip(("x1", "y1", "x2", "y2"), entry["box"], strict=True)),
                    target_label=label,
                    candidate_id=f"{view}_{label}",
                    results_directory=results,
                    fg_points=[[x / w, y / h] for x, y in entry.get("fg", [])],
                    bg_points=[[x / w, y / h] for x, y in entry.get("bg", [])],
                    show_review=False,
                )
                manifest[view][label] = {"prompt": entry, "result": result}
                print(view, label, result["deterministic_best_candidate_index"], flush=True)
            write_json(root / "seeds/candidates.json", manifest)


def track(root: Path, frames: int, policy: str, views: list[str]) -> None:
    config = json.loads((root / "config.json").read_text())
    if not config.get("reviewed_seeds"):
        raise ValueError("Freeze reviewed seeds before propagation")
    if policy != "baseline" and not (root / "baseline/frozen.json").exists():
        raise ValueError("Freeze the reviewed plain baseline before policy experiments")
    if policy == "combined":
        review = json.loads((root / "policy-review.json").read_text())
        if not all(review.get(p, {}).get("qualified") for p in ("exclusivity", "memory")):
            raise ValueError("Both isolated memory policies must qualify before combination")
    for view in views:
        record = config["views"][view]
        directory = (root / policy / view).resolve()
        command = [
            str(MODEL_PYTHON),
            str(Path(__file__).with_name("muggled_worker.py").resolve()),
            "--run-directory",
            str(directory),
            "--video",
            record["video"],
            "--view-id",
            view,
            "--source-offset-seconds",
            "0",
            "--model",
            str(MODEL),
            "--max-frames",
            str(frames),
            "--max-side-length",
            "1280",
            "--analysis-fps",
            str(record["fps"]),
            "--concepts-json",
            json.dumps(LABELS),
            "--prompt-mode",
            "manual_seed_multiplexed_keyframes",
            "--multi-keyframe-schedule-json",
            json.dumps(record["schedule"], sort_keys=True),
            "--max-frame-memory",
            "4",
            "--prompt-memory-semantics",
            "append",
            "--max-prompt-memory",
            "32",
            "--mask-period-frames",
            "1",
            "--checkpoint-every",
            "300",
            "--checkpoint-at",
            str(frames - 1),
            "--gpu-guard",
            "vram",
            "--gpu-guard-profile",
            "sam3_1280",
        ]
        if policy in ("exclusivity", "combined"):
            command += ["--slot-exclusivity", "argmax"]
        if policy in ("memory", "combined"):
            command += ["--memory-gate", "on"]
        checkpoints = sorted((directory / "native/checkpoints").glob("*.pt"))
        if checkpoints:
            from .muggled_worker import parse_args, stream_identity

            old_command = json.loads((directory / "worker_command.json").read_text())
            if stream_identity(parse_args(old_command[2:]), LABELS) != stream_identity(
                parse_args(command[2:]), LABELS
            ):
                raise ValueError("Resume inputs differ from the saved stream")
            resume_frame = int(checkpoints[-1].stem[1:])
            records = directory / "observations.jsonl"
            lines = records.read_text().splitlines()
            if len(lines) >= frames:
                print(f"{view}: already has {len(lines)} frames", flush=True)
                continue
            # The worker appends and checkpoints precede the named frame. Archive
            # the tail before reprocessing it, so resumed observations stay unique.
            write_json(
                directory / f"stage-{len(lines)}-result.json",
                json.loads((directory / "worker_result.json").read_text()),
            )
            (directory / f"stage-{len(lines)}-tail.jsonl").write_text(
                "\n".join(lines[resume_frame:]) + "\n"
            )
            records.write_text("\n".join(lines[:resume_frame]) + "\n")
            command += ["--resume-from-checkpoint", str(checkpoints[-1])]
        elif (directory / "observations.jsonl").exists():
            raise ValueError(f"Existing stream has no resumable checkpoint: {directory}")
        launch(command, directory)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("replay", "decode-seeds", "track"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--views", nargs="+", default=["T1", "T2", "T3", "T4", "T5", "fpv"])
    parser.add_argument(
        "--policy", choices=("baseline", "exclusivity", "memory", "combined"), default="baseline"
    )
    args = parser.parse_args()
    if args.action == "replay":
        replay(args.root)
    elif args.action == "decode-seeds":
        launch(
            [
                str(MODEL_PYTHON),
                str(Path(__file__).resolve()),
                "decode-seeds-model",
                "--root",
                str(args.root.resolve()),
            ],
            args.root / "seeds",
        )
    else:
        track(args.root, args.frames, args.policy, args.views)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "decode-seeds-model":
        decode_seeds(Path(sys.argv[sys.argv.index("--root") + 1]))
    else:
        main()
