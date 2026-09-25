"""`battle-muggled-arms`: the SAM3 worker on a video window with prompts from outside.

The FineBio arms (plan `p4-arms`) run on raw or natively trimmed videos with detector boxes
and tracker re-projections as prompts; there is no Assembly101 G2 preprocessing manifest, no
fixed view vocabulary and no calibration workspace behind the schedule.  `battle-muggled-smoke`
assumes all three (its `--view` choices, `G2PreprocessingManifest`, the schedule loader's
calibration binding), so this sibling driver spawns the same worker with the same GPU guard,
records the same measurements and condition record, and writes a `MuggledSAMArmRunManifest`
that binds the run to its inputs by SHA-256 instead of by an approved config.

Two modes, one per worker mode:

* ``box-decode`` (arm b): a per-frame box stream (JSONL) in, masks out, no video memory.
* ``video-memory`` (arms c/d): a seed/correction payload in the worker's own schedule format
  (seeds by mask or `prompt_box`, per-slot `start_frame`, corrections by mask or box with
  `selected_by: detector_reseed | track_reproject`), the memory arms (`--prompt-memory-semantics
  append`), the Sep 18 policy flags and the tau hook (`--memory-write-min-score`).

Frame indices in the prompt files are analysis indices: frame 0 is `--start-frame` of the
video.  Nothing under `runs/` is committed (FineBio licence).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from . import fs_common, gpu_guard
from .digest_cache import sha256_file
from .fs_common import relative_uri, run_timestamp
from .muggled_smoke import (
    DEFAULT_MODEL,
    MUGGLED_SAM_PYTHON,
    MUGGLED_SAM_SOURCE,
    _external_revision,
    _flat_runtime_settings,
    _gpu_guard_settings,
    correction_memory_settings_from_args,
)
from .muggled_worker import (
    _corrections_by_frame,
    correction_memory_semantics,
    parse_box_stream,
    slot_start_frames,
)
from .observations import load_observations
from .schemas import (
    AdapterMetadata,
    ArtifactFingerprint,
    CorrectionMemorySettings,
    FrameRange,
    MethodState,
    MethodStatus,
    MuggledSAMArmRunManifest,
    MultiKeyframeCorrectionScheduleMetadata,
    RuntimeMeasurements,
    TrackerMemoryPolicy,
)

ADAPTER_VERSION = "0.1.0"
WORKER_PATH = Path(__file__).resolve().parent / "muggled_worker.py"
DEFAULT_MAX_SIDE_LENGTH = 1280
DEFAULT_MAX_FRAME_MEMORY = 4
# FineBio ships 30000/1001 fps video and the proxies keep it (plan p1-configs).
DEFAULT_ANALYSIS_FPS = 30000 / 1001
WORKER_FAILED_EXIT_CODE = 3
LICENCE_NOTE = (
    "FineBio is licensed for non-commercial research; frames, videos, masks and recordings "
    "derived from it stay under runs/ or data/ and are never committed or redistributed"
)


def make_run_id(mode: str, view_id: str, now: datetime | None = None, *, suffix: str = "") -> str:
    base = f"muggledsam-arm-{mode.replace('_', '-')}-{view_id.lower()}-{run_timestamp(now)}"
    return f"{base}-{suffix}" if suffix else base


def memory_policy_from_args(args: argparse.Namespace) -> TrackerMemoryPolicy:
    return TrackerMemoryPolicy(
        slot_exclusivity=getattr(args, "slot_exclusivity", "off"),
        memory_gate=getattr(args, "memory_gate", "off"),
        exclusivity_loser_logit=getattr(args, "exclusivity_loser_logit", -8.0),
        gate_min_object_score=getattr(args, "gate_min_object_score", 0.0),
        gate_min_iou=getattr(args, "gate_min_iou", 0.5),
        gate_max_contested_fraction=getattr(args, "gate_max_contested_fraction", 0.2),
        gate_area_band=tuple(getattr(args, "gate_area_band", (0.5, 2.0))),
        gate_area_history_frames=getattr(args, "gate_area_history_frames", 30),
        memory_write_min_score=getattr(args, "memory_write_min_score", None),
    )


def load_schedule_payload(
    schedule_path: Path, *, effective_memory_semantics: str
) -> tuple[dict[str, Any], tuple[str, ...], MultiKeyframeCorrectionScheduleMetadata]:
    """Read a worker schedule payload, resolve its mask paths and validate it as the worker will.

    The file may omit `memory_semantics` or declare the semantics the run is configured for;
    anything else is refused here rather than by the worker.  Relative `mask_path`s are taken
    against the file's directory.  Concepts are the seeds' targets in slot order.
    """
    payload = json.loads(schedule_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("seeds"), list):
        raise ValueError(f"{schedule_path} must be an object with a seeds list")
    declared = payload.get("memory_semantics")
    if declared is not None and declared != effective_memory_semantics:
        raise ValueError(
            f"{schedule_path} declares {declared!r}; this run is configured for "
            f"{effective_memory_semantics!r}"
        )
    seeds = sorted(payload["seeds"], key=lambda seed: int(seed["initial_multiplex_slot"]))
    if [int(seed["initial_multiplex_slot"]) for seed in seeds] != list(range(len(seeds))):
        raise ValueError("schedule seeds must fill multiplex slots 0..N-1 exactly once")
    concepts = tuple(str(seed["target"]) for seed in seeds)
    if len(set(concepts)) != len(concepts):
        raise ValueError("schedule seed targets must be distinct")
    corrections = list(payload.get("corrections", []))
    for entry in [*seeds, *corrections]:
        mask_path = entry.get("mask_path")
        if mask_path is not None:
            resolved = (schedule_path.parent / mask_path).resolve()
            if not resolved.is_file():
                raise ValueError(f"schedule mask does not exist: {resolved}")
            if entry.get("mask_sha256") is None:
                entry["mask_sha256"] = sha256_file(resolved)
            elif sha256_file(resolved) != entry["mask_sha256"]:
                raise ValueError(f"schedule mask SHA-256 changed: {resolved}")
            entry["mask_path"] = str(resolved)
    normalised = {
        "memory_semantics": effective_memory_semantics,
        "seeds": seeds,
        "corrections": sorted(
            corrections, key=lambda item: (int(item["frame_index"]), int(item["multiplex_slot"]))
        ),
    }
    grouped = _corrections_by_frame(
        normalised, concepts, memory_semantics=effective_memory_semantics
    )
    starts = slot_start_frames(normalised)
    later = normalised["corrections"]

    def frames_by_kind(kind: str) -> tuple[int, ...]:
        from_corrections = {
            int(item["frame_index"])
            for item in later
            if item.get("selected_by") == kind
            and (item.get("prompt_box") is not None or "prompt_box_xyxy_px" in item)
        }
        from_starts = {
            frame for slot, frame in starts.items() if seeds[slot].get("selected_by") == kind
        }
        return tuple(sorted(from_corrections | from_starts))

    schedule_fingerprint = ArtifactFingerprint(
        uri=str(schedule_path), sha256=sha256_file(schedule_path), source="measured"
    )
    metadata = MultiKeyframeCorrectionScheduleMetadata(
        schedule_fingerprint=schedule_fingerprint,
        # The payload is its own policy: there is no separate keyframe-budget file here.
        correction_policy_fingerprint=schedule_fingerprint,
        correction_memory_semantics=effective_memory_semantics,
        scheduled_correction_frame_indices=tuple(sorted(grouped)),
        agent_selected_correction_frame_indices=tuple(
            sorted(
                {int(item["frame_index"]) for item in later if item.get("selected_by") == "agent"}
            )
        ),
        detector_reseed_correction_frame_indices=frames_by_kind("detector_reseed"),
        track_reproject_correction_frame_indices=frames_by_kind("track_reproject"),
        slot_start_frames=tuple(sorted(starts.items())),
    )
    return normalised, concepts, metadata


def worker_command(
    args: argparse.Namespace,
    *,
    run_directory: Path,
    concepts: Sequence[str],
    schedule_payload: dict[str, Any] | None,
    memory_policy: TrackerMemoryPolicy,
    memory_settings: CorrectionMemorySettings,
) -> list[str]:
    """The worker command line for this run; pure, so a test can read it without a GPU."""
    command = [
        str(args.external_python),
        str(WORKER_PATH),
        "--run-directory",
        str(run_directory),
        "--video",
        str(Path(args.video).resolve()),
        "--view-id",
        args.view_id,
        "--source-offset-seconds",
        str(args.source_offset_seconds),
        "--model",
        str(Path(args.model).resolve()),
        "--max-frames",
        str(args.max_frames),
        "--max-side-length",
        str(args.max_side_length),
        "--analysis-fps",
        str(args.analysis_fps),
        "--start-frame",
        str(args.start_frame),
        "--concepts-json",
        json.dumps(list(concepts)),
        "--gpu-guard",
        args.gpu_guard,
    ]
    for pid in args.allow_gpu_neighbour:
        command.extend(["--allow-gpu-neighbour", str(int(pid))])
    profile, expected_peak = _gpu_guard_settings(args)
    if profile is not None:
        command.extend(["--gpu-guard-profile", profile])
    if expected_peak is not None:
        command.extend(["--expected-peak-vram-bytes", str(int(expected_peak))])
    if args.mode == "box_decode":
        command.extend(
            ["--prompt-mode", "box_stream", "--box-stream", str(Path(args.box_stream).resolve())]
        )
        if args.other_slots_as_negatives:
            command.append("--other-slots-as-negatives")
        return command
    assert schedule_payload is not None
    command.extend(
        [
            "--prompt-mode",
            "manual_seed_multiplexed_keyframes",
            "--multi-keyframe-schedule-json",
            json.dumps(schedule_payload, sort_keys=True),
            "--max-frame-memory",
            str(args.max_frame_memory),
            "--checkpoint-every",
            str(args.checkpoint_every),
        ]
    )
    if args.no_checkpoints:
        command.append("--no-checkpoints")
    if not memory_policy.is_default:
        command.extend(memory_policy.worker_arguments())
    command.extend(memory_settings.worker_arguments())
    return command


def run_worker(command: list[str], run_directory: Path) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    source_paths = [str(MUGGLED_SAM_SOURCE)]
    if existing := environment.get("PYTHONPATH"):
        source_paths.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(source_paths)
    completed = subprocess.run(
        command, capture_output=True, text=True, env=environment, check=False
    )
    (run_directory / "worker.stdout.log").write_text(completed.stdout)
    (run_directory / "worker.stderr.log").write_text(completed.stderr)
    result_path = run_directory / "worker_result.json"
    if not result_path.is_file():
        return {
            "state": "failed",
            "reason": (
                f"worker exited {completed.returncode} without worker_result.json; "
                "see worker.stdout.log and worker.stderr.log"
            ),
            "frames_processed": 0,
            "elapsed_seconds": 0.0,
            "time_to_first_usable_output_seconds": None,
            "gpu_peak_vram_bytes": None,
            "masks_written": 0,
            "known_unavailable_measures": [
                "time_to_first_usable_output_seconds",
                "gpu_peak_vram_bytes",
            ],
            "runtime_settings": {},
        }
    result = json.loads(result_path.read_text())
    result["worker_exit_code"] = completed.returncode
    return result


def run_arm(args: argparse.Namespace) -> Path:
    repository_root = Path.cwd().resolve()
    video_path = Path(args.video).resolve()
    model_path = Path(args.model).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"video does not exist: {video_path}")
    memory_settings = correction_memory_settings_from_args(args)
    memory_policy = memory_policy_from_args(args)
    schedule_payload: dict[str, Any] | None = None
    schedule_metadata: MultiKeyframeCorrectionScheduleMetadata | None = None
    if args.mode == "box_decode":
        prompt_path = Path(args.box_stream).resolve()
        _frames, concepts = parse_box_stream(prompt_path.read_text(encoding="utf-8"))
        if not concepts:
            raise ValueError(f"box stream {prompt_path} names no slots")
        condition_suffix = "neg" if args.other_slots_as_negatives else ""
    else:
        prompt_path = Path(args.schedule).resolve()
        schedule_payload, concepts, schedule_metadata = load_schedule_payload(
            prompt_path,
            effective_memory_semantics=correction_memory_semantics(
                prompt_memory_semantics=memory_settings.prompt_memory_semantics,
                keep_frame_memory_at_correction=memory_settings.keep_frame_memory_at_correction,
            ),
        )
        parts = [memory_policy.run_id_suffix()] if not memory_policy.is_default else []
        memory_suffix = memory_settings.run_id_suffix(
            reference_frame_memory_entries=DEFAULT_MAX_FRAME_MEMORY
        )
        if memory_suffix:
            parts.append(memory_suffix)
        condition_suffix = "-".join(parts)
    suffix_parts = [f"r{args.max_side_length}"]
    if condition_suffix:
        suffix_parts.append(condition_suffix)
    if args.run_id_suffix:
        suffix_parts.append(args.run_id_suffix)
    run_id = args.run_id or make_run_id(args.mode, args.view_id, suffix="-".join(suffix_parts))
    run_directory = (args.run_root / run_id).resolve()
    run_directory.mkdir(parents=True, exist_ok=False)

    command = worker_command(
        args,
        run_directory=run_directory,
        concepts=concepts,
        schedule_payload=schedule_payload,
        memory_policy=memory_policy,
        memory_settings=memory_settings,
    )
    fs_common.write_json(run_directory / "worker_command.json", command)
    result = run_worker(command, run_directory)
    state = MethodState(result["state"])
    observations_path = run_directory / "observations.jsonl"
    rows = 0
    observations_sha256 = None
    method_statuses = [
        MethodStatus(
            method_name=f"muggledsam-sam3-{args.mode.replace('_', '-')}",
            stage="objects",
            state=state,
            artifact_uri=(
                relative_uri(observations_path, repository_root)
                if observations_path.is_file()
                else None
            ),
            blocker=result.get("reason") if state is not MethodState.SUCCEEDED else None,
            measured_on=(
                f"analysis frames [0, {args.max_frames}) = source frames "
                f"[{args.start_frame}, {args.start_frame + args.max_frames}) of {video_path.name}; "
                "decoder / tracker self-reports only, nothing against ground truth"
            ),
        )
    ]
    if observations_path.is_file():
        rows = len(load_observations(observations_path))
        observations_sha256 = fs_common.sha256_file(observations_path)
    manifest = MuggledSAMArmRunManifest(
        manifest_kind="muggledsam_sam3_arm_run",
        run_id=run_id,
        mode=args.mode,
        view_id=args.view_id,
        video_fingerprint=ArtifactFingerprint(
            uri=relative_uri(video_path, repository_root),
            sha256=sha256_file(video_path),
            source="measured",
        ),
        prompt_fingerprint=ArtifactFingerprint(
            uri=relative_uri(prompt_path, repository_root),
            sha256=sha256_file(prompt_path),
            source="measured",
        ),
        model_fingerprint=ArtifactFingerprint(
            uri=str(model_path), sha256=sha256_file(model_path), source="measured"
        ),
        worker_fingerprint=ArtifactFingerprint(
            uri=relative_uri(WORKER_PATH, repository_root),
            sha256=sha256_file(WORKER_PATH),
            source="measured",
        ),
        adapter=AdapterMetadata(
            name="battle.muggled_arms",
            version=ADAPTER_VERSION,
            implementation_basis=(
                "MuggledSAM SAMV3p1InteractiveModel encode_prompts/generate_masks (box_decode); "
                "simple_examples/video_segmentation_multiplexed.py (video_memory)"
            ),
            external_source_uri=str(MUGGLED_SAM_SOURCE),
            external_revision=_external_revision(),
        ),
        start_frame=args.start_frame,
        requested_analysis_frame_range=FrameRange(
            start_frame=0, end_frame_exclusive=args.max_frames
        ),
        analysis_fps=args.analysis_fps,
        source_offset_seconds=args.source_offset_seconds,
        max_side_length=args.max_side_length,
        concepts=tuple(concepts),
        other_slots_as_negatives=bool(args.mode == "box_decode" and args.other_slots_as_negatives),
        memory_policy=memory_policy if args.mode == "video_memory" else None,
        memory_settings=memory_settings if args.mode == "video_memory" else None,
        multi_keyframe_corrections=schedule_metadata,
        stream_identity=result.get("runtime_settings", {}).get("stream_identity"),
        method_statuses=tuple(method_statuses),
        measurements=RuntimeMeasurements(
            elapsed_seconds=float(result["elapsed_seconds"]),
            time_to_first_usable_output_seconds=result.get("time_to_first_usable_output_seconds"),
            gpu_peak_vram_bytes=result.get("gpu_peak_vram_bytes"),
            known_unavailable_measures=tuple(result.get("known_unavailable_measures", [])),
        ),
        runtime_settings={
            "external_python": str(args.external_python),
            "model_path": str(model_path),
            "cuda_visible_devices": "0",
            "worker_exit_code": result.get("worker_exit_code"),
            **_flat_runtime_settings(result.get("runtime_settings", {})),
        },
        observations_uri=(
            relative_uri(observations_path, repository_root)
            if observations_path.is_file()
            else None
        ),
        observations_sha256=observations_sha256,
        observation_rows=rows,
        mask_artifact_uri=(
            relative_uri(run_directory / "masks", repository_root)
            if (run_directory / "masks").is_dir()
            else None
        ),
        mask_artifact_count=int(result.get("masks_written", 0)),
        licence_note=LICENCE_NOTE,
    )
    (run_directory / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
    return run_directory


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--video", type=Path, required=True, help="Raw or trimmed video.")
    parser.add_argument("--view-id", required=True, help="e.g. fpv, T1..T5.")
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--run-id", default=None, help="Exact run id; default is generated.")
    parser.add_argument("--run-id-suffix", default="", help="Appended to the generated run id.")
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Source frame that becomes analysis frame 0 (prompt files use analysis indices).",
    )
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--max-side-length", type=int, default=DEFAULT_MAX_SIDE_LENGTH)
    parser.add_argument(
        "--analysis-fps",
        type=float,
        default=DEFAULT_ANALYSIS_FPS,
        help="Frame rate used to map analysis frames to source seconds (FineBio: 30000/1001).",
    )
    parser.add_argument(
        "--source-offset-seconds",
        type=float,
        default=None,
        help="Source seconds of analysis frame 0; default start_frame / analysis_fps.",
    )
    parser.add_argument("--external-python", type=Path, default=MUGGLED_SAM_PYTHON)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--allow-gpu-neighbour", type=int, action="append", default=[], metavar="PID"
    )
    parser.add_argument(
        "--gpu-guard", choices=gpu_guard.GUARD_MODES, default=gpu_guard.DEFAULT_GUARD_MODE
    )
    parser.add_argument(
        "--gpu-guard-profile", choices=tuple(gpu_guard.EXPECTED_PEAK_BYTES), default=None
    )
    parser.add_argument("--expected-peak-vram-bytes", type=int, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the SAM3 worker on a video window with prompts from outside."
    )
    modes = parser.add_subparsers(dest="mode", required=True)

    box = modes.add_parser(
        "box-decode", help="Arm (b): per-frame box stream in, masks out, no video memory."
    )
    _add_common_arguments(box)
    box.add_argument("--box-stream", type=Path, required=True, help="Per-frame boxes, JSONL.")
    box.add_argument(
        "--other-slots-as-negatives",
        action="store_true",
        help="Other boxes' centres as background points for every prompt (off = plain decode).",
    )

    video = modes.add_parser(
        "video-memory",
        help="Arms (c)/(d): SAM3.1 multiplex video memory with a seed/correction payload.",
    )
    _add_common_arguments(video)
    video.add_argument(
        "--schedule",
        type=Path,
        required=True,
        help=(
            "Worker schedule payload: {seeds: [{target, initial_multiplex_slot, mask_path | "
            "prompt_box | prompt_box_xyxy_px, start_frame?, selected_by?}], corrections: "
            "[{frame_index, multiplex_slot, target, mask_path | prompt_box, selected_by}]}."
        ),
    )
    video.add_argument("--max-frame-memory", type=int, default=DEFAULT_MAX_FRAME_MEMORY)
    video.add_argument(
        "--prompt-memory-semantics", choices=("replace", "append"), default="replace"
    )
    video.add_argument("--max-prompt-memory", type=int, default=None)
    video.add_argument("--keep-frame-memory-at-correction", action="store_true")
    video.add_argument("--recent-first", action="store_true")
    video.add_argument("--checkpoint-every", type=int, default=0)
    video.add_argument("--no-checkpoints", action="store_true")
    video.add_argument("--slot-exclusivity", choices=("off", "argmax"), default="off")
    video.add_argument("--memory-gate", choices=("off", "on"), default="off")
    video.add_argument("--exclusivity-loser-logit", type=float, default=-8.0)
    video.add_argument("--gate-min-object-score", type=float, default=0.0)
    video.add_argument("--gate-min-iou", type=float, default=0.5)
    video.add_argument("--gate-max-contested-fraction", type=float, default=0.2)
    video.add_argument(
        "--gate-area-band",
        type=lambda value: tuple(float(item) for item in value.split(",")),
        default=(0.5, 2.0),
    )
    video.add_argument("--gate-area-history-frames", type=int, default=30)
    video.add_argument(
        "--memory-write-min-score",
        type=float,
        default=None,
        metavar="TAU",
        help="Skip the frame-memory write for a slot whose raw score is below TAU (the tau hook).",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.mode = args.mode.replace("-", "_")
    if args.start_frame < 0 or args.max_frames <= 0:
        parser.error("--start-frame must be >= 0 and --max-frames > 0")
    if args.source_offset_seconds is None:
        args.source_offset_seconds = args.start_frame / args.analysis_fps
    return args


def main() -> None:
    args = parse_args()
    run_directory = run_arm(args)
    manifest = MuggledSAMArmRunManifest.model_validate_json(
        (run_directory / "manifest.json").read_text(encoding="utf-8")
    )
    status = manifest.method_statuses[0]
    print(run_directory)
    if status.state is not MethodState.SUCCEEDED:
        print(f"worker {status.state.value}: {status.blocker}", file=sys.stderr)
        raise SystemExit(WORKER_FAILED_EXIT_CODE)


if __name__ == "__main__":
    main()
