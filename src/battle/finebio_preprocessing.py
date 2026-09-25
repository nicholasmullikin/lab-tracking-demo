"""``battle-finebio-preprocess``: the FineBio window proxies and their manifest (p1-configs).

One trial window (raw frames ``[start, end)``, the same for all six views because the videos
share a frame count and are synchronised to +/-1 frame) is cut into six proxies with
`finebio_frames.proxy_ffmpeg_args` **unchanged**: native resolution (fixed 1920x1080, fpv
1920x1440), native 30000/1001 rate, no ``fps=`` filter, exact-frame trim on a full decode,
``-fps_mode passthrough``, libx264 crf 18. Proxy frame k is raw frame ``start + k``; the
manifest records ``frame_index_offset = start`` and every downstream stage adds it back.

For every proxy the tool verifies the counted frame count, probes the stream through
`media_probe` (the ``.probe.json`` sidecar the review builders read), hashes the proxy and
the raw video, and runs `finebio_frames.frame_index_contract` on three sample offsets
(first, middle, last frame): the mean absolute grey difference between proxy frame k and raw
frames ``start+k-1, start+k, start+k+1`` must have its minimum at 0. The fpv proxy is also
checked against the shipped per-frame pose (`proxy_marker_check`: the day's markers projected
through the pose of raw frame ``start+k`` onto the ArUco corners of proxy frame k, under
10 px). ``pose_length_check`` confirms the pose file has one entry per raw frame.

Outputs:

* ``<output>/<trial>_<view>_<start>-<end>.mp4`` (six proxies; under ``data/derived/finebio``,
  gitignored: FineBio is non-commercial research data);
* ``<output>/manifest.json``: `FineBioPreprocessingManifest` (kind ``finebio_preprocessing``);
* ``configs/finebio/cameras/<trial>_<start>-<end>.json``: the trial camera config with
  ``frame_index_offset = start`` (committed; numbers only);
* ``configs/clips/finebio_<trial>_<start>-<end>.json``: the clip config the downstream builders
  read (views, targets, window, proxies, camera configs, manifest path).

    uv run battle-finebio-preprocess --trial P03_03_01 --window-from configs/finebio/trials.json
    uv run battle-finebio-preprocess --trial P03_01_01 --start 1798 --end 2398
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from . import finebio_frames as ff
from . import media_probe
from .finebio_cameras import read_camera_config, write_camera_config
from .fs_common import relative_uri, sha256_file
from .multiview_schemas import FINEBIO_FIXED_VIEWS, FINEBIO_FPV_VIEW, FineBioCameraConfig
from .schemas import VersionedModel

VIEWS: tuple[str, ...] = (*FINEBIO_FIXED_VIEWS, FINEBIO_FPV_VIEW)
LICENCE = (
    "FineBio (aistairc/FineBio), non-commercial research use; the proxies under data/derived "
    "and every frame, video or mask derived from them stay out of the repository"
)
INTRINSICS_RESCALE = {"fixed": 0.5, "fpv": 0.48}
SOURCE_LICENSE_LINE = (
    "FineBio dataset terms: non-commercial research; nothing under data/ or runs/ is "
    "redistributed (docs/SOURCES.md)"
)
# The classes the tracker owns, the containers and landmarks it uses as volumes, and the
# hand classes used as occluders and probes (plan: Decisions taken, p2-seeds).
DEFAULT_TARGETS: tuple[str, ...] = (
    "cell_culture_plate",
    "blue_pipette",
    "yellow_pipette",
    "red_pipette",
    "8_channel_pipette",
    "50ml_tube",
    "15ml_tube",
    "micro_tube",
    "8_tube_stripes",
    "blue_tip_rack",
    "yellow_tip_rack",
    "red_tip_rack",
    "8_channel_tip_rack",
)
DEFAULT_CONTAINERS: tuple[str, ...] = (
    "centrifuge",
    "vortex_mixer",
    "pcr_machine",
    "micro_tube_rack",
    "50ml_tube_rack",
    "15ml_tube_rack",
    "8_tube_stripes_rack",
    "magnetic_rack",
    "trash_can",
)
DEFAULT_PROBES: tuple[str, ...] = ("left_hand", "right_hand")


# --------------------------------------------------------------------------- manifest


class ArtifactRef(VersionedModel):
    uri: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class FineBioProxyRecord(VersionedModel):
    """One view's proxy: what was cut, from what, with which command, and the checks."""

    view: str = Field(min_length=1)
    raw: ArtifactRef
    raw_frame_count: int = Field(ge=1)
    proxy: ArtifactRef
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    r_frame_rate: str = Field(min_length=1)
    avg_frame_rate: str = Field(min_length=1)
    counted_frames: int = Field(ge=1)
    probe_fps: int = Field(ge=1)
    ffmpeg_args: tuple[str, ...] = Field(min_length=1)
    encode_seconds: float = Field(ge=0)
    frame_index_contract: dict[str, Any]
    marker_check: dict[str, Any] | None = None


class FineBioPreprocessingManifest(VersionedModel):
    """The window proxies of one FineBio trial (no HF provenance: the data is local)."""

    manifest_kind: Literal["finebio_preprocessing"] = "finebio_preprocessing"
    trial: str = Field(min_length=1)
    recording_day: str = Field(pattern=r"^\d{6}$")
    window_start_frame: int = Field(ge=0)
    window_end_frame_exclusive: int = Field(gt=0)
    frame_count: int = Field(ge=1)
    # raw_frame = proxy_frame + frame_index_offset, for every view.
    frame_index_offset: int = Field(ge=0)
    fps: Literal["30000/1001"] = "30000/1001"
    views: dict[str, FineBioProxyRecord]
    proxy_recipe: dict[str, Any]
    camera_config: ArtifactRef
    window_camera_config: ArtifactRef
    intrinsics_rescale: dict[str, float]
    pose_length_check: dict[str, Any]
    window_source: dict[str, Any]
    clip_config_uri: str | None = None
    checks_passed: bool
    licence: str = LICENCE
    created: str = Field(min_length=1)
    command: str = Field(min_length=1)


def proxy_recipe_record() -> dict[str, Any]:
    template = ff.proxy_ffmpeg_args(Path("<raw>"), Path("<out>"), 0, 1)
    return {
        "function": "battle.finebio_frames.proxy_ffmpeg_args",
        "resolution": "native (fixed 1920x1080, fpv 1920x1440), no scale filter",
        "rate": f"native {ff.NATIVE_RATE}, no fps filter",
        "trim": "select='between(n,start,end)' on a full decode, setpts=N/FRAME_RATE/TB, "
        "-fps_mode passthrough, -frames:v count, no -ss",
        "codec": f"libx264 crf {ff.PROXY_CRF} preset medium yuv420p faststart, no audio",
        "template": template,
    }


# --------------------------------------------------------------------------- window


def window_from_trials(trials_path: Path, trial: str) -> dict[str, Any]:
    payload = json.loads(trials_path.read_text(encoding="utf-8"))
    for entry in payload["trials"]:
        if entry["trial"] == trial:
            return entry
    raise KeyError(f"{trial} is not in {trials_path}")


def proxy_name(trial: str, view: str, start: int, end: int) -> str:
    return f"{trial}_{view}_{start}-{end}.mp4"


def window_camera_config_path(trial: str, start: int, end: int) -> Path:
    return Path("configs/finebio/cameras") / f"{trial}_{start}-{end}.json"


def clip_config_path(trial: str, start: int, end: int) -> Path:
    return Path("configs/clips") / f"finebio_{trial}_{start}-{end}.json"


def sample_offsets(frame_count: int) -> list[int]:
    return sorted({0, frame_count // 2, frame_count - 1})


# --------------------------------------------------------------------------- build


def build_view(
    trial: str,
    view: str,
    start: int,
    count: int,
    output: Path,
    *,
    repository_root: Path,
    day: str,
    skip_existing: bool = False,
) -> FineBioProxyRecord:
    raw = ff.video_path(trial, view)
    out = output / proxy_name(trial, view, start, start + count)
    args = ff.proxy_ffmpeg_args(raw, out, start, count)
    began = time.monotonic()
    if skip_existing and out.exists():
        stream = ff.ffprobe_stream(out)
        counted = ff.counted_frames(out)
        if counted != count:
            raise RuntimeError(f"{out} has {counted} frames, expected {count}")
        info = {**stream, "counted_frames": counted}
    else:
        info = ff.build_proxy(raw, out, start, count)
    encode_seconds = time.monotonic() - began
    frames, fps, (width, height) = media_probe.video_info(out, verify=True)
    if frames != count:
        raise RuntimeError(f"media_probe counted {frames} frames in {out}, expected {count}")
    offsets = sample_offsets(count)
    contract = ff.frame_index_contract(raw, out, start, offsets)
    marker = ff.proxy_marker_check(trial, day, out, start, offsets) if view == "fpv" else None
    return FineBioProxyRecord(
        view=view,
        raw=ArtifactRef(uri=relative_uri(raw, repository_root), sha256=sha256_file(raw)),
        raw_frame_count=ff.raw_frame_count(raw),
        proxy=ArtifactRef(uri=relative_uri(out, repository_root), sha256=sha256_file(out)),
        width=width,
        height=height,
        r_frame_rate=str(info["r_frame_rate"]),
        avg_frame_rate=str(info["avg_frame_rate"]),
        counted_frames=int(info["counted_frames"]),
        probe_fps=fps,
        ffmpeg_args=tuple(args),
        encode_seconds=encode_seconds,
        frame_index_contract=contract,
        marker_check=marker,
    )


def checks_pass(record: FineBioProxyRecord) -> tuple[bool, list[str]]:
    problems = []
    if not record.frame_index_contract["all_minima_at_zero"]:
        problems.append(f"{record.view}: frame-index contract minimum is not at offset 0")
    if record.r_frame_rate != ff.NATIVE_RATE:
        problems.append(f"{record.view}: r_frame_rate {record.r_frame_rate} != {ff.NATIVE_RATE}")
    if record.marker_check is not None:
        rms = record.marker_check["median_corner_rms_px"]
        if rms is None:
            problems.append(f"{record.view}: no markers seen on the proxy samples")
        elif rms >= ff.PROXY_MARKER_MAX_RMS_PX:
            problems.append(f"{record.view}: proxy marker RMS {rms:.1f} px over the gate")
    return not problems, problems


def write_window_camera_config(
    config: FineBioCameraConfig, start: int, end: int, path: Path, *, source_uri: str
) -> FineBioCameraConfig:
    windowed = config.model_copy(
        update={
            "frame_index_offset": start,
            "provenance": {
                **config.provenance,
                "window": {
                    "start_frame": start,
                    "end_frame_exclusive": end,
                    "frame_index_offset": start,
                    "trial_camera_config": source_uri,
                    "note": "raw_frame = proxy_frame + frame_index_offset; poses unchanged",
                },
            },
        }
    )
    write_camera_config(windowed, path)
    return windowed


def clip_config(
    manifest: FineBioPreprocessingManifest,
    *,
    trial_entry: dict[str, Any] | None,
    trials_ref: ArtifactRef | None,
    targets: tuple[str, ...] = DEFAULT_TARGETS,
    containers: tuple[str, ...] = DEFAULT_CONTAINERS,
    probes: tuple[str, ...] = DEFAULT_PROBES,
) -> dict[str, Any]:
    start, end = manifest.window_start_frame, manifest.window_end_frame_exclusive
    window = f"{start}-{end}"
    entry = trial_entry or {}
    return {
        "schema_version": "1.0",
        "config_kind": "finebio_clip_config",
        "clip_id": f"finebio-{manifest.trial}-{window}",
        "source_name": "FineBio",
        "source_license": SOURCE_LICENSE_LINE,
        "trial": manifest.trial,
        "role": entry.get("role"),
        "room": entry.get("room"),
        "recording_day": manifest.recording_day,
        "fps": manifest.fps,
        "window": {
            "start_frame": start,
            "end_frame_exclusive": end,
            "frame_count": manifest.frame_count,
            "seconds": [round(start * 1001 / 30000, 2), round(end * 1001 / 30000, 2)],
        },
        "frame_index_offset": manifest.frame_index_offset,
        "views": list(VIEWS),
        "fixed_views": list(FINEBIO_FIXED_VIEWS),
        "fpv_view": FINEBIO_FPV_VIEW,
        "view_sizes": {v: [r.width, r.height] for v, r in manifest.views.items()},
        "targets": list(targets),
        "containers": list(containers),
        "probes": list(probes),
        "camera_config": manifest.camera_config.uri,
        "window_camera_config": manifest.window_camera_config.uri,
        "preprocessing_manifest": f"data/derived/finebio/{manifest.trial}/{window}/manifest.json",
        "proxies": {v: r.proxy.uri for v, r in manifest.views.items()},
        "proxy_sha256": {v: r.proxy.sha256 for v, r in manifest.views.items()},
        "annotated_frames_in_window": entry.get("annotated_frames_in_window"),
        "centrifuge_cycles_in_window": (entry.get("centrifuge") or {}).get("cycles_in_window"),
        "window_fpv_pose_valid_fraction": entry.get("window_fpv_pose_valid_fraction"),
        "trials_source": None if trials_ref is None else trials_ref.model_dump(),
        "downstream": {
            "detections": f"runs/finebio-detect-{manifest.trial}-{window}/ (GPU phase, "
            "battle-finebio-detect on the six proxies; frame_index raw = proxy + offset)",
            "rig": f"uv run battle-finebio-rig --config {manifest.window_camera_config.uri} "
            f"--detections runs/finebio-detect-{manifest.trial}-{window} "
            f"--frames {start}-{end - 1} --output runs/finebio-rig-{manifest.trial}-{window} "
            "--negative-control",
        },
        "licence": manifest.licence,
    }


def run(
    trial: str,
    start: int,
    end: int,
    *,
    output: Path,
    camera_config_path: Path,
    repository_root: Path,
    trials_path: Path | None,
    jobs: int,
    skip_existing: bool,
    command: str,
    write_configs: bool = True,
) -> tuple[FineBioPreprocessingManifest, dict[str, Any], list[str]]:
    if end <= start:
        raise ValueError("the window needs end > start")
    count = end - start
    config = read_camera_config(camera_config_path)
    if config.trial != trial:
        raise ValueError(f"{camera_config_path} is for {config.trial}, not {trial}")
    output.mkdir(parents=True, exist_ok=True)
    day = config.recording_day
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures = {
            view: pool.submit(
                build_view,
                trial,
                view,
                start,
                count,
                output,
                repository_root=repository_root,
                day=day,
                skip_existing=skip_existing,
            )
            for view in VIEWS
        }
        records = {view: future.result() for view, future in futures.items()}
    problems: list[str] = []
    for record in records.values():
        ok, found = checks_pass(record)
        problems.extend(found)
    pose_check = ff.pose_length_check(trial, VIEWS)
    if not pose_check["equal"]:
        problems.append("fpv pose length differs from the raw frame counts")
    window_config_path = repository_root / window_camera_config_path(trial, start, end)
    if write_configs:
        write_window_camera_config(
            config,
            start,
            end,
            window_config_path,
            source_uri=relative_uri(camera_config_path, repository_root),
        )
    else:
        write_window_camera_config(
            config,
            start,
            end,
            output / window_config_path.name,
            source_uri=relative_uri(camera_config_path, repository_root),
        )
        window_config_path = output / window_config_path.name
    trial_entry = None
    trials_ref = None
    if trials_path is not None:
        trial_entry = window_from_trials(trials_path, trial)
        trials_ref = ArtifactRef(
            uri=relative_uri(trials_path, repository_root), sha256=sha256_file(trials_path)
        )
    clip_uri = relative_uri(repository_root / clip_config_path(trial, start, end), repository_root)
    manifest = FineBioPreprocessingManifest(
        trial=trial,
        recording_day=day,
        window_start_frame=start,
        window_end_frame_exclusive=end,
        frame_count=count,
        frame_index_offset=start,
        views=records,
        proxy_recipe=proxy_recipe_record(),
        camera_config=ArtifactRef(
            uri=relative_uri(camera_config_path, repository_root),
            sha256=sha256_file(camera_config_path),
        ),
        window_camera_config=ArtifactRef(
            uri=relative_uri(window_config_path, repository_root),
            sha256=sha256_file(window_config_path),
        ),
        intrinsics_rescale=dict(INTRINSICS_RESCALE),
        pose_length_check=pose_check,
        window_source={
            "trials_json": None if trials_ref is None else trials_ref.model_dump(),
            "entry_window": None if trial_entry is None else trial_entry.get("window"),
            "requested": {"start_frame": start, "end_frame_exclusive": end},
        },
        clip_config_uri=clip_uri if write_configs else None,
        checks_passed=not problems,
        created=datetime.now(UTC).isoformat(timespec="seconds"),
        command=command,
    )
    (output / "manifest.json").write_text(manifest.model_dump_json(indent=1) + "\n")
    clip = clip_config(manifest, trial_entry=trial_entry, trials_ref=trials_ref)
    if write_configs and not problems:
        path = repository_root / clip_config_path(trial, start, end)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(clip, indent=2) + "\n")
    return manifest, clip, problems


def summary_lines(manifest: FineBioPreprocessingManifest, problems: list[str]) -> list[str]:
    lines = [
        f"{manifest.trial} raw [{manifest.window_start_frame}, "
        f"{manifest.window_end_frame_exclusive}) -> {manifest.frame_count} frames per view, "
        f"frame_index_offset {manifest.frame_index_offset}, day {manifest.recording_day}"
    ]
    for view, record in manifest.views.items():
        contract = record.frame_index_contract
        at_zero = [s["difference"]["0"] for s in contract["samples"]]
        off = contract["min_difference_off_zero"]
        line = (
            f"  {view}: {record.width}x{record.height} {record.r_frame_rate}, "
            f"{record.counted_frames} frames in {record.encode_seconds:.0f} s; contract at 0 = "
            + "/".join(f"{d:.2f}" for d in at_zero)
            + f" grey levels, min off-zero {off:.2f}, minima at 0: {contract['all_minima_at_zero']}"
        )
        if record.marker_check:
            rms = record.marker_check["median_corner_rms_px"]
            line += (
                f"; markers on {record.marker_check['frames_with_markers']}/3 samples, "
                f"median RMS {'n/a' if rms is None else f'{rms:.2f} px'}"
            )
        lines.append(line)
    pose = manifest.pose_length_check
    lines.append(
        f"  pose length {pose['pose_frames']} == raw frame counts: {pose['equal']} "
        f"(valid {pose['valid_pose_fraction']:.3f})"
    )
    lines.append(f"  checks passed: {manifest.checks_passed}")
    lines.extend(f"  PROBLEM: {p}" for p in problems)
    return lines


# --------------------------------------------------------------------------- cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="battle-finebio-preprocess",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--trial", required=True)
    parser.add_argument(
        "--window-from",
        type=Path,
        default=Path("configs/finebio/trials.json"),
        help="trials.json with the per-trial window (start/end raw frames, end exclusive)",
    )
    parser.add_argument("--start", type=int, default=None, help="raw start frame (overrides)")
    parser.add_argument("--end", type=int, default=None, help="raw end frame, exclusive")
    parser.add_argument(
        "--output", type=Path, default=None, help="data/derived/finebio/<trial>/<w>"
    )
    parser.add_argument("--camera-config", type=Path, default=None)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--skip-existing", action="store_true", help="reuse proxies already cut")
    parser.add_argument("--dry-run", action="store_true", help="print the ffmpeg commands only")
    parser.add_argument(
        "--no-configs",
        action="store_true",
        help="write the window camera config beside the proxies and no clip config",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path.cwd()
    trials_path = args.window_from if args.window_from and args.window_from.exists() else None
    start, end = args.start, args.end
    if start is None or end is None:
        if trials_path is None:
            raise SystemExit("give --start and --end, or a --window-from trials.json")
        window = window_from_trials(trials_path, args.trial)["window"]
        start = window["start"] if start is None else start
        end = window["end"] if end is None else end
    output = args.output or Path("data/derived/finebio") / args.trial / f"{start}-{end}"
    camera_config = args.camera_config or Path("configs/finebio/cameras") / f"{args.trial}.json"
    if args.dry_run:
        for view in VIEWS:
            raw = ff.video_path(args.trial, view)
            out = output / proxy_name(args.trial, view, start, end)
            print(subprocess.list2cmdline(ff.proxy_ffmpeg_args(raw, out, start, end - start)))
        return 0
    manifest, _, problems = run(
        args.trial,
        start,
        end,
        output=output,
        camera_config_path=camera_config,
        repository_root=root,
        trials_path=trials_path,
        jobs=args.jobs,
        skip_existing=args.skip_existing,
        command=" ".join(sys.argv),
        write_configs=not args.no_configs,
    )
    print("\n".join(summary_lines(manifest, problems)))
    print(f"manifest -> {output / 'manifest.json'}")
    if manifest.clip_config_uri and not problems:
        print(f"clip config -> {manifest.clip_config_uri}")
        print(f"window camera config -> {manifest.window_camera_config.uri}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
