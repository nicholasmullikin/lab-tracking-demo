"""Headless, agent-attributed later-frame SAM3 correction candidates.

The browser workspace records every acceptance as a human decision.  When an agent's visual
review (never a human) proposes an extra correction keyframe, this tool keeps that provenance
explicit: it derives a fresh calibration directory from a finalized one (so the original
fingerprinted manifest is never rewritten), decodes box prompts through the same isolated
MuggledSAM worker, tags every resulting candidate ``selected_by="agent"``, and finalizes an
augmented schedule whose new rows carry that tag.  It never seeds frame 0 and never tracks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from .muggled_calibration import (
    _write_manifest,
    finalize_correction_schedule,
    frame_reference,
    next_candidate_id,
)
from .muggled_calibration_web import WorkerClient, _normal_box, _normal_points
from .muggled_smoke import DEFAULT_MODEL, MUGGLED_SAM_PYTHON, MUGGLED_SAM_SOURCE, sha256_file
from .muggled_smoke import relative_uri as _relative_uri
from .schemas import (
    ArtifactFingerprint,
    MuggledSAMBoxCalibrationManifest,
    MuggledSAMCalibrationCandidate,
    MuggledSAMCalibrationWorkspace,
    PixelBox,
    PixelPoint,
)

AGENT_PROVENANCE = "agent_authored_visual_review"
SCHEDULE_NAME = "multi_keyframe_correction_schedule.json"
MANIFEST_NAME = "calibration_manifest.json"


def _load(calibration_dir: Path) -> tuple[Path, MuggledSAMBoxCalibrationManifest]:
    manifest_path = (calibration_dir / MANIFEST_NAME).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"calibration manifest is unavailable: {manifest_path}")
    return manifest_path, MuggledSAMBoxCalibrationManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )


def _link_tree(source: Path, destination: Path) -> int:
    """Hard-link (falling back to copy) every result artifact so URIs stay valid."""
    count = 0
    for path in sorted(source.rglob("*")):
        target = destination / path.relative_to(source)
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(path, target)
        except OSError:
            shutil.copy2(path, target)
        count += 1
    return count


def derive_calibration(*, source_dir: Path, output_dir: Path, repository_root: Path) -> Path:
    """Copy a finalized calibration into a new draft directory that records its origin."""
    source_manifest_path, manifest = _load(source_dir)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"derived calibration directory already exists: {output_dir}")
    if not manifest.final_correction_schedule_uri:
        raise ValueError("derive only from a calibration whose correction schedule is finalized")
    output_dir.mkdir(parents=True)
    _link_tree(source_dir / "results", output_dir / "results")
    derived = manifest.model_copy(
        update={
            "calibration_id": output_dir.name,
            "result_directory_uri": _relative_uri(output_dir / "results", repository_root),
            "final_proposal_uri": None,
            "final_correction_schedule_uri": None,
            "plan_revision": 0,
            "superseded_plans": (),
            "workspace": MuggledSAMCalibrationWorkspace(),
            "derived_from_calibration": ArtifactFingerprint(
                uri=_relative_uri(source_manifest_path, repository_root),
                sha256=sha256_file(source_manifest_path),
                source="measured",
            ),
        }
    )
    manifest_path = output_dir / MANIFEST_NAME
    _write_manifest(manifest_path, derived)
    return manifest_path


def parse_prompt(
    value: str,
) -> tuple[str, PixelBox, tuple[PixelPoint, ...], tuple[PixelPoint, ...]]:
    """Parse ``target=x1,y1,x2,y2[;fg=x,y[;fg=...]][;bg=x,y]``."""
    head, *options = value.split(";")
    target, _, box_text = head.partition("=")
    coordinates = [int(item) for item in box_text.split(",")]
    if not target or len(coordinates) != 4:
        raise ValueError(f"prompt must look like target=x1,y1,x2,y2: {value}")
    box = PixelBox(x1=coordinates[0], y1=coordinates[1], x2=coordinates[2], y2=coordinates[3])
    fg: list[PixelPoint] = []
    bg: list[PixelPoint] = []
    for option in options:
        kind, _, point_text = option.partition("=")
        x, y = (int(item) for item in point_text.split(","))
        if kind == "fg":
            fg.append(PixelPoint(x=x, y=y))
        elif kind == "bg":
            bg.append(PixelPoint(x=x, y=y))
        else:
            raise ValueError(f"unknown prompt option {kind!r} in {value}")
    return target.strip(), box, tuple(fg), tuple(bg)


def _worker(
    manifest: MuggledSAMBoxCalibrationManifest,
    *,
    repository_root: Path,
    output_dir: Path,
    external_python: Path,
    model: Path,
    device: str,
) -> WorkerClient:
    proxy_path = (repository_root / manifest.proxy.uri).resolve()
    if not proxy_path.is_file() or sha256_file(proxy_path) != manifest.proxy.sha256:
        raise ValueError("approved proxy is missing or its checksum no longer matches")
    if not external_python.is_file():
        raise ValueError(f"configured MuggledSAM interpreter does not exist: {external_python}")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(MUGGLED_SAM_SOURCE), environment["PYTHONPATH"]]
        if environment.get("PYTHONPATH")
        else [str(MUGGLED_SAM_SOURCE)]
    )
    worker_path = Path(__file__).with_name("muggled_calibration_worker.py")
    return WorkerClient(
        [
            str(external_python),
            str(worker_path),
            "--serve-jsonl",
            "--proxy",
            str(proxy_path),
            "--model",
            str(model.resolve()),
            "--results-directory",
            str(output_dir / "results"),
            "--device",
            device,
        ],
        environment=environment,
        stderr_path=output_dir / "agent_worker.stderr.log",
    )


def decode_candidates(
    *,
    calibration_dir: Path,
    frame_index: int,
    prompts: tuple[str, ...],
    repository_root: Path,
    external_python: Path = MUGGLED_SAM_PYTHON,
    model: Path = DEFAULT_MODEL,
    device: str = "cuda:0",
    decoder: Any | None = None,
) -> tuple[str, ...]:
    """Decode box prompts at one later frame and persist them as agent-authored candidates."""
    manifest_path, manifest = _load(calibration_dir)
    if manifest.final_correction_schedule_uri:
        raise ValueError("this calibration is finalized; derive a draft copy first")
    if frame_index <= 0:
        raise ValueError("agent corrections are later-frame only; frame 0 seeds stay human")
    frame = frame_reference(
        frame_index / manifest.proxy_fps,
        fps=manifest.proxy_fps,
        source_offset_seconds=manifest.source_offset_seconds,
        frame_count=manifest.proxy_frame_count,
    )
    parsed = [parse_prompt(item) for item in prompts]
    working = manifest
    requests: list[dict[str, Any]] = []
    for target, box, fg, bg in parsed:
        candidate_id = next_candidate_id(working, frame_index)
        # Reserve the ID so several prompts in one batch never collide.
        working = working.model_copy(
            update={
                "candidates": (
                    *working.candidates,
                    _placeholder(working, candidate_id, target, frame, box, fg, bg),
                )
            }
        )
        normalized_box = _normal_box(box, manifest)
        requests.append(
            {
                "box_id": f"p{frame_index:06d}-b{int(candidate_id.rsplit('b', 1)[1]):02d}",
                "candidate_id": candidate_id,
                "frame_index": frame_index,
                "pixel_box": box.model_dump(mode="json"),
                "intended_target": target,
                "boxes": [
                    [
                        [normalized_box.x, normalized_box.y],
                        [
                            normalized_box.x + normalized_box.width,
                            normalized_box.y + normalized_box.height,
                        ],
                    ]
                ],
                "fg_points": [[p.x, p.y] for p in _normal_points(fg, manifest)],
                "bg_points": [[p.x, p.y] for p in _normal_points(bg, manifest)],
            }
        )
    owns_decoder = decoder is None
    if decoder is None:
        decoder = _worker(
            manifest,
            repository_root=repository_root,
            output_dir=manifest_path.parent,
            external_python=external_python,
            model=model,
            device=device,
        )
    try:
        decoder.request("frame_preview", {"frame_index": frame_index})
        response = decoder.request("batch_decode", {"prompts": requests}, timeout=600)
    finally:
        if owns_decoder:
            decoder.close()
    decoded = {item["candidate_id"]: item["decoder_result"] for item in response["decoded"]}
    candidates = list(manifest.candidates)
    for (target, box, fg, bg), request in zip(parsed, requests, strict=True):
        candidates.append(
            MuggledSAMCalibrationCandidate(
                candidate_id=request["candidate_id"],
                intended_target=target,
                frame=frame,
                pixel_box=box,
                normalized_box=_normal_box(box, manifest),
                pixel_fg_points=fg,
                pixel_bg_points=bg,
                normalized_fg_points=_normal_points(fg, manifest),
                normalized_bg_points=_normal_points(bg, manifest),
                decoder_result=decoded[request["candidate_id"]],
                selected_by="agent",
            )
        )
    timestamps = tuple(sorted({*manifest.requested_proxy_timestamps_seconds, frame.proxy_seconds}))
    _write_manifest(
        manifest_path,
        manifest.model_copy(
            update={
                "candidates": tuple(candidates),
                "requested_proxy_timestamps_seconds": timestamps,
            }
        ),
    )
    return tuple(request["candidate_id"] for request in requests)


def load_decode_plan(path: Path) -> tuple[tuple[int, tuple[str, ...]], ...]:
    """Read an ordered list of `{frame, prompts}` groups to decode in one session."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("a decode plan must be a non-empty list of {frame, prompts} objects")
    plan: list[tuple[int, tuple[str, ...]]] = []
    for entry in payload:
        if not isinstance(entry, dict):
            raise ValueError(f"decode plan entries must be objects, got {type(entry).__name__}")
        frame = entry.get("frame")
        prompts = entry.get("prompts")
        if not isinstance(frame, int) or frame < 1:
            raise ValueError(f"decode plan frame must be an integer above zero, got {frame!r}")
        if not isinstance(prompts, list) or not prompts:
            raise ValueError(f"decode plan frame {frame} needs at least one prompt")
        plan.append((frame, tuple(str(prompt) for prompt in prompts)))
    return tuple(plan)


def decode_candidate_batch(
    *,
    calibration_dir: Path,
    plan: tuple[tuple[int, tuple[str, ...]], ...],
    repository_root: Path,
    external_python: Path = MUGGLED_SAM_PYTHON,
    model: Path = DEFAULT_MODEL,
    device: str = "cuda:0",
    decoder: Any | None = None,
) -> dict[int, tuple[str, ...]]:
    """Decode several frames' prompts against one warm worker.

    Each `decode` invocation otherwise reloads the multi-gigabyte checkpoint, which
    dominates a review loop that is really asking one model several small questions.
    """
    _, manifest = _load(calibration_dir)
    owns_decoder = decoder is None
    if decoder is None:
        decoder = _worker(
            manifest,
            repository_root=repository_root,
            output_dir=calibration_dir.resolve(),
            external_python=external_python,
            model=model,
            device=device,
        )
    decoded: dict[int, tuple[str, ...]] = {}
    try:
        for frame_index, prompts in plan:
            decoded[frame_index] = decode_candidates(
                calibration_dir=calibration_dir,
                frame_index=frame_index,
                prompts=prompts,
                repository_root=repository_root,
                decoder=decoder,
            )
    finally:
        if owns_decoder:
            decoder.close()
    return decoded


def _placeholder(
    manifest: MuggledSAMBoxCalibrationManifest,
    candidate_id: str,
    target: str,
    frame: Any,
    box: PixelBox,
    fg: tuple[PixelPoint, ...],
    bg: tuple[PixelPoint, ...],
) -> MuggledSAMCalibrationCandidate:
    """A schema-valid stand-in used only to reserve candidate IDs before decoding."""
    return MuggledSAMCalibrationCandidate(
        candidate_id=candidate_id,
        intended_target=target,
        frame=frame,
        pixel_box=box,
        normalized_box=_normal_box(box, manifest),
        decoder_result={
            "api": "muggledsam_sam3_interactive",
            "candidate_count": 1,
            "deterministic_best_candidate_index": 0,
            "candidates": [
                {
                    "candidate_index": 0,
                    "iou_score": 0.0,
                    "mask_uri": "results/masks/pending.png",
                    "is_deterministic_best": True,
                }
            ],
            "overlay_uri": "results/pending.png",
        },
        selected_by="agent",
    )


def accept_agent_candidate(
    *, calibration_dir: Path, candidate_id: str, candidate_index: int, rationale: str
) -> MuggledSAMCalibrationCandidate:
    """Record an agent visual-review acceptance as a later-frame correction candidate."""
    manifest_path, manifest = _load(calibration_dir)
    if manifest.final_correction_schedule_uri:
        raise ValueError("this calibration is finalized; derive a draft copy first")
    if not rationale.strip():
        raise ValueError("an agent acceptance needs a written visual-review rationale")
    candidates = list(manifest.candidates)
    candidate = next((item for item in candidates if item.candidate_id == candidate_id), None)
    if candidate is None:
        raise KeyError(f"decoded candidate does not exist: {candidate_id}")
    if candidate.selected_by != "agent":
        raise ValueError("only agent-decoded candidates may be accepted by this tool")
    candidates = [
        other.model_copy(
            update={
                "human_selected_candidate_index": None,
                "human_accepted": False,
                "selected_for_correction": False,
            }
        )
        if other.candidate_id != candidate_id
        and other.human_accepted
        and other.intended_target == candidate.intended_target
        and other.frame.analysis_frame_index == candidate.frame.analysis_frame_index
        else other
        for other in candidates
    ]
    accepted = candidate.model_copy(
        update={
            "human_selected_candidate_index": candidate_index,
            "human_accepted": True,
            "selected_for_correction": True,
            "selected_by": "agent",
        }
    )
    accepted = MuggledSAMCalibrationCandidate.model_validate(accepted.model_dump(mode="json"))
    candidates[[item.candidate_id for item in candidates].index(candidate_id)] = accepted
    _write_manifest(manifest_path, manifest.model_copy(update={"candidates": tuple(candidates)}))
    log_path = manifest_path.with_name("agent_acceptances.jsonl")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "candidate_id": candidate_id,
                    "candidate_index": candidate_index,
                    "intended_target": candidate.intended_target,
                    "analysis_frame_index": candidate.frame.analysis_frame_index,
                    "provenance": AGENT_PROVENANCE,
                    "rationale": rationale.strip(),
                }
            )
            + "\n"
        )
    return accepted


def finalize_agent_schedule(
    *,
    calibration_dir: Path,
    correction_policy_path: Path,
    manual_seed_target_config_path: Path,
    repository_root: Path,
) -> Path:
    """Write the augmented schedule from every retained seed and correction selection."""
    manifest_path, manifest = _load(calibration_dir)
    if manifest.final_correction_schedule_uri:
        raise ValueError("this calibration already has a finalized correction schedule")
    candidate_ids = tuple(
        candidate.candidate_id
        for candidate in manifest.candidates
        if not candidate.rejected
        and candidate.human_accepted
        and (candidate.selected_for_finalization or candidate.selected_for_correction)
    )
    if not any(
        candidate.selected_by == "agent"
        for candidate in manifest.candidates
        if candidate.candidate_id in candidate_ids
    ):
        raise ValueError("no agent-accepted correction candidate is selected")
    schedule_path = manifest_path.with_name(SCHEDULE_NAME)
    next_manifest = manifest.model_copy(
        update={
            "final_correction_schedule_uri": _relative_uri(schedule_path, repository_root),
            "plan_revision": 1,
        }
    )
    manifest_content = (next_manifest.model_dump_json(indent=2) + "\n").encode()
    finalize_correction_schedule(
        manifest_path=manifest_path,
        candidate_ids=candidate_ids,
        schedule_path=schedule_path,
        correction_policy_path=correction_policy_path,
        manual_seed_target_config_path=manual_seed_target_config_path,
        repository_root=repository_root,
        manifest=next_manifest,
        calibration_manifest_sha256=hashlib.sha256(manifest_content).hexdigest(),
    )
    manifest_path.write_bytes(manifest_content)
    return schedule_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)

    derive = commands.add_parser("derive", help="copy a finalized calibration into a draft")
    derive.add_argument("--source-calibration", type=Path, required=True)
    derive.add_argument("--output-dir", type=Path, required=True)

    decode = commands.add_parser("decode", help="decode agent box prompts at one later frame")
    decode.add_argument("--calibration", type=Path, required=True)
    decode.add_argument("--frame", type=int, required=True)
    decode.add_argument(
        "--prompt", action="append", required=True, help="target=x1,y1,x2,y2[;fg=x,y][;bg=x,y]"
    )
    decode.add_argument("--external-python", type=Path, default=MUGGLED_SAM_PYTHON)
    decode.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    decode.add_argument("--device", default="cuda:0")

    batch = commands.add_parser(
        "decode-batch", help="decode several frames' prompts against one warm worker"
    )
    batch.add_argument("--calibration", type=Path, required=True)
    batch.add_argument(
        "--plan",
        type=Path,
        required=True,
        help='JSON list of {"frame": int, "prompts": ["target=x1,y1,x2,y2", ...]} groups.',
    )
    batch.add_argument("--external-python", type=Path, default=MUGGLED_SAM_PYTHON)
    batch.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    batch.add_argument("--device", default="cuda:0")

    accept = commands.add_parser("accept", help="record an agent visual-review acceptance")
    accept.add_argument("--calibration", type=Path, required=True)
    accept.add_argument("--candidate-id", required=True)
    accept.add_argument("--index", type=int, required=True)
    accept.add_argument("--rationale", required=True)

    schedule = commands.add_parser("schedule", help="finalize the augmented schedule")
    schedule.add_argument("--calibration", type=Path, required=True)
    schedule.add_argument("--correction-policy", type=Path, required=True)
    schedule.add_argument("--manual-seed-target-config", type=Path, required=True)

    args = parser.parse_args()
    root = args.repository_root.resolve()
    if args.command == "derive":
        print(
            derive_calibration(
                source_dir=args.source_calibration, output_dir=args.output_dir, repository_root=root
            )
        )
    elif args.command == "decode":
        ids = decode_candidates(
            calibration_dir=args.calibration,
            frame_index=args.frame,
            prompts=tuple(args.prompt),
            repository_root=root,
            external_python=args.external_python,
            model=args.model,
            device=args.device,
        )
        print(json.dumps({"candidate_ids": list(ids)}))
    elif args.command == "decode-batch":
        decoded = decode_candidate_batch(
            calibration_dir=args.calibration,
            plan=load_decode_plan(args.plan),
            repository_root=root,
            external_python=args.external_python,
            model=args.model,
            device=args.device,
        )
        print(
            json.dumps(
                {str(frame): list(ids) for frame, ids in decoded.items()},
                sort_keys=True,
            )
        )
    elif args.command == "accept":
        accepted = accept_agent_candidate(
            calibration_dir=args.calibration,
            candidate_id=args.candidate_id,
            candidate_index=args.index,
            rationale=args.rationale,
        )
        print(accepted.model_dump_json(indent=2))
    else:
        print(
            finalize_agent_schedule(
                calibration_dir=args.calibration,
                correction_policy_path=args.correction_policy.resolve(),
                manual_seed_target_config_path=args.manual_seed_target_config.resolve(),
                repository_root=root,
            )
        )


if __name__ == "__main__":
    main()
