"""Prepare pending two-timestamp human QA records without running inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schemas import (
    ArtifactFingerprint,
    ClockName,
    FixedTimestampHumanQARecord,
    FrameRange,
    HumanQACheckpoint,
    HumanQACheckpointRole,
    HumanQADisposition,
    HumanQAEvidence,
    MethodState,
    RunManifest,
)

QA_PROTOCOL = "assembly101_easy_hard_source_timestamps_v1"
QA_SELECTION_RULE = "one_easy_manipulation_and_one_hard_or_occluded_manipulation"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def repository_relative_uri(path: Path, repository_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repository_root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"artifact must be inside the repository: {resolved}") from error


@dataclass(frozen=True)
class CompletedRun:
    manifest_path: Path
    run_directory: Path
    manifest: RunManifest
    run_profile: str
    method_id: str
    view_id: str
    analysis_frame_range: FrameRange
    config_fingerprint: ArtifactFingerprint
    source_fingerprint: ArtifactFingerprint
    proxy_fingerprint: ArtifactFingerprint


def _manifest_path(run_path: Path) -> Path:
    path = run_path.resolve()
    if path.is_dir():
        path = path / "manifest.json"
    if path.name != "manifest.json" or not path.is_file():
        raise ValueError(f"expected a run directory or manifest.json: {run_path}")
    return path


def load_completed_run(run_path: Path, repository_root: Path) -> CompletedRun:
    """Validate one explicit run and select its declared completed profile."""

    manifest_path = _manifest_path(run_path)
    repository_relative_uri(manifest_path, repository_root)
    manifest = RunManifest.model_validate_json(manifest_path.read_text())
    run_directory = manifest_path.parent
    if run_directory.name != manifest.run_id:
        raise ValueError("run directory name must match manifest run_id")

    object_statuses = [
        status
        for status in manifest.method_statuses
        if status.stage == "objects" and status.state is MethodState.SUCCEEDED
    ]
    if len(object_statuses) != 1:
        raise ValueError("human QA preparation requires exactly one succeeded objects method")

    profiles: list[tuple[str, Any]] = [
        ("g3_full_static_candidate", manifest.g3_candidate),
        ("g4_e4_60_second_candidate", manifest.e4_candidate),
        ("full_ego_manual_seed_multiplexed_baseline", manifest.full_ego_manual_seed),
    ]
    selected = [(name, metadata) for name, metadata in profiles if metadata is not None]
    if len(selected) != 1:
        raise ValueError(
            "human QA preparation supports one declared completed static, e4 candidate, "
            "or full ego run profile"
        )
    run_profile, metadata = selected[0]
    return CompletedRun(
        manifest_path=manifest_path,
        run_directory=run_directory,
        manifest=manifest,
        run_profile=run_profile,
        method_id=object_statuses[0].method_name,
        view_id=metadata.view_id,
        analysis_frame_range=metadata.requested_analysis_frame_range,
        config_fingerprint=metadata.config_fingerprint,
        source_fingerprint=metadata.source_fingerprint,
        proxy_fingerprint=metadata.proxy_fingerprint,
    )


def evenly_spaced_frame_indices(frame_range: FrameRange, count: int) -> tuple[int, ...]:
    """Include both endpoints while selecting a small deterministic review grid."""

    if count < 2:
        raise ValueError("candidate sheet requires at least two timestamps")
    if count > frame_range.frame_count:
        raise ValueError("candidate count cannot exceed the available analysis frames")
    span = frame_range.frame_count - 1
    denominator = count - 1
    return tuple(
        frame_range.start_frame + (index * span + denominator // 2) // denominator
        for index in range(count)
    )


def _synchronized_candidate_contract(
    runs: tuple[CompletedRun, CompletedRun],
) -> tuple[FrameRange, int, float, float]:
    first, second = runs
    if first.manifest.clip.source_name != second.manifest.clip.source_name:
        raise ValueError("candidate views must come from the same named source")
    if first.analysis_frame_range != second.analysis_frame_range:
        raise ValueError("candidate views must cover the same analysis frame range")
    first_fps = first.manifest.clip.timing.clocks.fps_for(ClockName.ANALYSIS)
    second_fps = second.manifest.clip.timing.clocks.fps_for(ClockName.ANALYSIS)
    first_mapping = next(
        mapping
        for mapping in first.manifest.clip.timing.mappings
        if mapping.clock is ClockName.ANALYSIS
    )
    second_mapping = next(
        mapping
        for mapping in second.manifest.clip.timing.mappings
        if mapping.clock is ClockName.ANALYSIS
    )
    if (
        first_fps != second_fps
        or first_mapping.source_offset_seconds != second_mapping.source_offset_seconds
        or first_mapping.scale != second_mapping.scale
    ):
        raise ValueError("candidate views must have identical analysis-to-source mappings")
    return (
        first.analysis_frame_range,
        first_fps,
        first_mapping.source_offset_seconds,
        first_mapping.scale,
    )


def _decode_selected_frames(proxy_path: Path, frame_indices: tuple[int, ...]) -> list[Any]:
    from PIL import Image

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0:s=x",
            str(proxy_path),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    if probe.returncode != 0:
        raise RuntimeError(f"could not probe proxy dimensions: {proxy_path}")
    try:
        width, height = (int(value) for value in probe.stdout.strip().split("x"))
    except ValueError as error:
        raise RuntimeError(f"invalid ffprobe dimensions for {proxy_path}") from error

    select = "+".join(f"eq(n\\,{frame_index})" for frame_index in frame_indices)
    decode = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(proxy_path),
            "-vf",
            f"select={select}",
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        capture_output=True,
        check=False,
    )
    if decode.returncode != 0:
        raise RuntimeError(
            f"could not decode candidate frames from {proxy_path}: "
            f"{decode.stderr.decode(errors='replace').strip()}"
        )
    frame_bytes = width * height * 3
    expected_bytes = frame_bytes * len(frame_indices)
    if len(decode.stdout) != expected_bytes:
        raise RuntimeError(
            f"decoded {len(decode.stdout)} bytes from {proxy_path}; expected {expected_bytes}"
        )
    return [
        Image.frombytes(
            "RGB",
            (width, height),
            decode.stdout[offset : offset + frame_bytes],
        )
        for offset in range(0, expected_bytes, frame_bytes)
    ]


def _letterbox(frame: Any, width: int, height: int) -> Any:
    from PIL import Image

    resized = frame.copy()
    resized.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (20, 20, 20))
    x = (width - resized.width) // 2
    y = (height - resized.height) // 2
    canvas.paste(resized, (x, y))
    return canvas


def render_synchronized_candidate_sheet(
    runs: tuple[CompletedRun, CompletedRun],
    repository_root: Path,
    output_path: Path,
    *,
    count: int = 12,
) -> tuple[Path, tuple[float, ...]]:
    """Render raw synchronized views only; no model outputs or semantic labels."""

    from PIL import Image, ImageDraw, ImageFont

    frame_range, analysis_fps, source_offset_seconds, scale = (
        _synchronized_candidate_contract(runs)
    )
    frame_indices = evenly_spaced_frame_indices(frame_range, count)
    relative_seconds = tuple(
        scale * (frame_index - frame_range.start_frame) / analysis_fps
        for frame_index in frame_indices
    )
    source_seconds = tuple(
        source_offset_seconds + scale * frame_index / analysis_fps
        for frame_index in frame_indices
    )

    frames_by_run = []
    for run in runs:
        _verify_fingerprinted_file(
            run.proxy_fingerprint, repository_root, label=f"{run.view_id} proxy"
        )
        frames_by_run.append(
            _decode_selected_frames(
                repository_root / run.proxy_fingerprint.uri,
                frame_indices,
            )
        )

    image_width = 540
    image_height = 304
    label_height = 38
    block_width = image_width * 2
    block_height = label_height + image_height
    columns = 2
    title_height = 64
    label_font = ImageFont.load_default(size=17)
    view_font = ImageFont.load_default(size=18)
    title_font = ImageFont.load_default(size=23)
    subtitle_font = ImageFont.load_default(size=17)
    blocks = []
    for candidate_offset, (frame_index, relative, source) in enumerate(
        zip(frame_indices, relative_seconds, source_seconds, strict=True)
    ):
        panels = []
        for run, decoded_frames in zip(runs, frames_by_run, strict=True):
            panel = _letterbox(decoded_frames[candidate_offset], image_width, image_height)
            draw = ImageDraw.Draw(panel)
            draw.rectangle((0, 0, image_width, 30), fill=(20, 20, 20))
            draw.text((10, 5), run.view_id, fill=(255, 255, 255), font=view_font)
            panels.append(panel)
        block = Image.new("RGB", (block_width, block_height), (238, 238, 238))
        block_draw = ImageDraw.Draw(block)
        block_draw.text(
            (10, 8),
            (
                f"Candidate {candidate_offset + 1:02d} | source={source:.6f}s | "
                f"clip=+{relative:.6f}s | analysis frame={frame_index}"
            ),
            fill=(20, 20, 20),
            font=label_font,
        )
        block.paste(panels[0], (0, label_height))
        block.paste(panels[1], (image_width, label_height))
        blocks.append(block)

    if len(blocks) % columns:
        blocks.append(Image.new("RGB", (block_width, block_height), (20, 20, 20)))

    sheet_width = block_width * columns
    sheet_height = title_height + block_height * (len(blocks) // columns)
    sheet = Image.new("RGB", (sheet_width, sheet_height), (35, 35, 35))
    sheet_draw = ImageDraw.Draw(sheet)
    sheet_draw.text(
        (16, 5),
        "SOURCE-TIMESTAMP CANDIDATES — RAW SYNCHRONIZED STATIC + EGO FRAMES",
        fill=(255, 255, 255),
        font=title_font,
    )
    sheet_draw.text(
        (16, 35),
        "Human selection aid only: choose one easy and one hard/occluded instant.",
        fill=(225, 225, 225),
        font=subtitle_font,
    )
    for index, block in enumerate(blocks):
        x = (index % columns) * block_width
        y = title_height + (index // columns) * block_height
        sheet.paste(block, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return output_path, source_seconds


def _checkpoint_indices(
    run: CompletedRun,
    source_timestamps: tuple[float, float],
) -> tuple[int, int]:
    analysis_fps = run.manifest.clip.timing.clocks.fps_for(ClockName.ANALYSIS)
    mapping = next(
        mapping
        for mapping in run.manifest.clip.timing.mappings
        if mapping.clock is ClockName.ANALYSIS
    )
    indices = []
    for source_seconds in source_timestamps:
        exact_index = (
            (source_seconds - mapping.source_offset_seconds) * analysis_fps / mapping.scale
        )
        analysis_frame_index = round(exact_index)
        if abs(exact_index - analysis_frame_index) > 1e-9:
            raise ValueError(
                f"source timestamp {source_seconds} does not map exactly to an analysis frame"
            )
        if not (
            run.analysis_frame_range.start_frame
            <= analysis_frame_index
            < run.analysis_frame_range.end_frame_exclusive
        ):
            raise ValueError(
                f"source timestamp {source_seconds} is outside run {run.manifest.run_id}"
            )
        indices.append(analysis_frame_index)
    if indices[0] == indices[1]:
        raise ValueError("easy and hard checkpoints must use distinct source timestamps")
    return indices[0], indices[1]


def _verify_fingerprinted_file(
    fingerprint: ArtifactFingerprint,
    repository_root: Path,
    *,
    label: str,
) -> None:
    path = repository_root / fingerprint.uri
    if not path.is_file():
        raise ValueError(f"{label} is unavailable: {fingerprint.uri}")
    if sha256_file(path) != fingerprint.sha256:
        raise ValueError(f"{label} fingerprint changed: {fingerprint.uri}")


def _preflight_output(path: Path, *, overwrite_pending: bool) -> None:
    if not path.exists():
        return
    try:
        existing = FixedTimestampHumanQARecord.model_validate_json(path.read_text())
    except ValueError as error:
        raise ValueError(f"refusing to overwrite an invalid existing QA record: {path}") from error
    has_human_decision = any(
        checkpoint.disposition is not HumanQADisposition.PENDING
        for checkpoint in existing.checkpoints
    )
    if has_human_decision:
        raise ValueError(f"refusing to overwrite a human-reviewed QA record: {path}")
    if not overwrite_pending:
        raise ValueError(f"pending QA record already exists (use --overwrite-pending): {path}")


def _render_model_overlay_contact_sheet(
    run: CompletedRun,
    repository_root: Path,
    evidence_path: Path,
    proxy_timestamps: tuple[float, float],
    source_offset_seconds: float,
) -> None:
    runtime_path = run.run_directory / "runtime_settings.json"
    if not runtime_path.is_file():
        raise ValueError(f"run runtime settings are unavailable: {runtime_path}")
    runtime = json.loads(runtime_path.read_text())
    external_python = Path(runtime.get("external_python", ""))
    if not external_python.is_file():
        raise ValueError(
            "run's recorded Python environment is unavailable for contact-sheet rendering: "
            f"{external_python}"
        )
    command = [
        str(external_python),
        str(Path(__file__).with_name("g3_contact_sheet.py")),
        "--run-directory",
        str(run.run_directory),
        "--repository-root",
        str(repository_root),
        "--timestamps",
        *(str(timestamp) for timestamp in proxy_timestamps),
        "--source-offset-seconds",
        str(source_offset_seconds),
        "--output",
        str(evidence_path),
    ]
    completed = subprocess.run(command, capture_output=True, check=False, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "could not render human QA evidence with the run's recorded environment: "
            f"{completed.stderr.strip()}"
        )


def _write_record(path: Path, record: FixedTimestampHumanQARecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = record.model_dump_json(indent=2) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as temporary:
        temporary.write(content)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def prepare_pending_record(
    run: CompletedRun,
    repository_root: Path,
    *,
    easy_source_seconds: float,
    hard_source_seconds: float,
) -> FixedTimestampHumanQARecord:
    """Generate bounded visual evidence and return an unattributed pending record."""

    source_timestamps = (easy_source_seconds, hard_source_seconds)
    analysis_indices = _checkpoint_indices(run, source_timestamps)
    _verify_fingerprinted_file(
        run.config_fingerprint, repository_root, label="run configuration"
    )

    evidence_path = (
        run.run_directory
        / "human_qa"
        / f"source-frames-{round(easy_source_seconds * 60)}-{round(hard_source_seconds * 60)}.png"
    )
    if not evidence_path.exists():
        mapping = next(
            mapping
            for mapping in run.manifest.clip.timing.mappings
            if mapping.clock is ClockName.ANALYSIS
        )
        proxy_timestamps = tuple(
            (timestamp - mapping.source_offset_seconds) / mapping.scale
            for timestamp in source_timestamps
        )
        _render_model_overlay_contact_sheet(
            run,
            repository_root,
            evidence_path,
            proxy_timestamps,
            mapping.source_offset_seconds,
        )
    if not evidence_path.is_file():
        raise ValueError(f"human QA evidence was not created: {evidence_path}")

    evidence = HumanQAEvidence(
        uri=repository_relative_uri(evidence_path, repository_root),
        sha256=sha256_file(evidence_path),
        artifact_kind="two_checkpoint_contact_sheet",
    )
    source_fps = run.manifest.clip.timing.clocks.fps_for(ClockName.SOURCE)
    checkpoints = tuple(
        HumanQACheckpoint(
            role=role,
            clock=ClockName.SOURCE,
            source_seconds=source_seconds,
            source_frame_index=round(source_seconds * source_fps),
            analysis_frame_index=analysis_frame_index,
            evidence=(evidence,),
        )
        for role, source_seconds, analysis_frame_index in zip(
            (HumanQACheckpointRole.EASY, HumanQACheckpointRole.HARD),
            source_timestamps,
            analysis_indices,
            strict=True,
        )
    )
    return FixedTimestampHumanQARecord(
        manifest_kind="fixed_timestamp_human_qa",
        qa_protocol=QA_PROTOCOL,
        selection_rule=QA_SELECTION_RULE,
        run_id=run.manifest.run_id,
        run_profile=run.run_profile,
        method_id=run.method_id,
        clip_id=run.manifest.clip.clip_id,
        view_id=run.view_id,
        run_manifest_fingerprint=ArtifactFingerprint(
            uri=repository_relative_uri(run.manifest_path, repository_root),
            sha256=sha256_file(run.manifest_path),
            source="measured",
        ),
        config_fingerprint=run.config_fingerprint,
        source_video_fingerprint=run.source_fingerprint,
        timing=run.manifest.clip.timing,
        run_analysis_frame_range=run.analysis_frame_range,
        checkpoints=checkpoints,
        overall_status=HumanQADisposition.PENDING,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare pending, evidence-fingerprinted human QA records from explicit completed runs."
        )
    )
    parser.add_argument(
        "runs", type=Path, nargs="+", help="Run directories or manifest.json files."
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--easy-source-seconds", type=float, required=True)
    parser.add_argument("--hard-source-seconds", type=float, required=True)
    parser.add_argument(
        "--overwrite-pending",
        action="store_true",
        help="Replace only an existing all-pending record; reviewed records are always refused.",
    )
    args = parser.parse_args()

    repository_root = args.repository_root.resolve()
    output_directory = args.output_directory.resolve()
    repository_relative_uri(output_directory, repository_root)
    completed_runs = [load_completed_run(path, repository_root) for path in args.runs]
    output_paths = [
        output_directory / f"{run.manifest.run_id}.human-qa.json" for run in completed_runs
    ]
    for path in output_paths:
        _preflight_output(path, overwrite_pending=args.overwrite_pending)

    records = [
        prepare_pending_record(
            run,
            repository_root,
            easy_source_seconds=args.easy_source_seconds,
            hard_source_seconds=args.hard_source_seconds,
        )
        for run in completed_runs
    ]
    for path, record in zip(output_paths, records, strict=True):
        _write_record(path, record)
        FixedTimestampHumanQARecord.model_validate_json(path.read_text())
        print(f"Wrote pending human QA record: {repository_relative_uri(path, repository_root)}")

    print(
        "Next manual action: inspect both source-time checkpoints in each contact sheet and "
        "record a human pass/flag/fail, optional note, reviewer identity, and timezone-aware time."
    )


def candidate_main() -> None:
    parser = argparse.ArgumentParser(
        description="Render raw synchronized source-timestamp candidates for human selection."
    )
    parser.add_argument(
        "runs",
        type=Path,
        nargs=2,
        metavar=("FIRST_RUN", "SECOND_RUN"),
        help="Exactly two completed run directories or manifest.json files.",
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/qa/assembly101_source_timestamp_candidates.png"),
    )
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repository_root = args.repository_root.resolve()
    output_path = args.output.resolve()
    output_uri = repository_relative_uri(output_path, repository_root)
    if output_uri.split("/", maxsplit=1)[0] not in {"artifacts", "runs"}:
        raise ValueError("candidate sheet output must stay under gitignored artifacts/ or runs/")
    if output_path.exists() and not args.overwrite:
        raise ValueError(f"candidate sheet already exists (use --overwrite): {output_path}")
    loaded = tuple(load_completed_run(path, repository_root) for path in args.runs)
    runs: tuple[CompletedRun, CompletedRun] = (loaded[0], loaded[1])
    path, source_timestamps = render_synchronized_candidate_sheet(
        runs, repository_root, output_path, count=args.count
    )
    print(f"Wrote synchronized candidate sheet: {path}")
    print(
        "Candidate source timestamps: "
        + ", ".join(f"{timestamp:.6f}" for timestamp in source_timestamps)
    )
    print("Next manual action: choose one easy and one hard/occluded source timestamp.")


if __name__ == "__main__":
    main()
