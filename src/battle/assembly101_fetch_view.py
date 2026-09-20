"""Fetch one focused window of an Assembly101 recording view without downloading the file.

The pinned Hugging Face revision resolves to a signed CDN URL that answers HTTP Range
requests.  ffmpeg reads that URL through a local counting proxy, seeks to the focused
source interval and, in one decode, writes both a 60 fps trim at sensor resolution (for
clock-offset scans) and the standard 1280x720 CFR 30 fps analysis proxy, encoded exactly like
`scripts/create_assembly101_four_part_reassembly_proxies.sh`.  The proxy records every byte
range ffmpeg asked for and every byte it was given, so the acquisition report can state how
much of the 1.6-4 GB file was actually transferred.

Local sources (the recordings already on disk) go through the same trim/proxy path so every
view's window is produced by one recipe.  Nothing here reads poses, annotations or models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import threading
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Literal

from pydantic import Field

from . import media_probe
from .assembly101_pose_schemas import ASSEMBLY101_CITATION, ASSEMBLY101_LICENSE
from .assembly101_recordings import (
    ANALYSIS_FPS,
    DATASET_REPO,
    DATASET_REVISION,
    RECORDING_1,
    SOURCE_FPS,
    Assembly101Recording,
    get_recording,
)
from .schemas import VersionedModel

# Recording-1 constants, kept for every caller written before the recording registry existed.
RECORDING_ID = RECORDING_1.recording_id
RAW_ROOT = RECORDING_1.raw_root
LOCAL_RECORDINGS = RECORDING_1.local_recordings_root
DERIVED_ROOT = RECORDING_1.derived_root
REPORT_NAME = "static_views_focused_acquisition_report.md"
MANIFEST_NAME = "static_views_focused_acquisition_manifest.json"

FOCUSED_START_SECONDS = RECORDING_1.window_start_seconds
FOCUSED_DURATION_SECONDS = RECORDING_1.window_duration_seconds

STATIC_VIEWS: tuple[str, ...] = (
    "C10095",
    "C10115",
    "C10118",
    "C10119",
    "C10379",
    "C10390",
    "C10395",
    "C10404",
)
NEW_STATIC_VIEWS: tuple[str, ...] = tuple(view for view in STATIC_VIEWS if view != "C10379")
EGO_VIEWS: tuple[str, ...] = ("HMC_21110305", "HMC_21176623", "HMC_21176875", "HMC_21179183")
STATIC_PROXY_SCALE = "1280:720"
EGO_PROXY_SCALE = "954:720"
PROXY_ENCODER_ARGS: tuple[str, ...] = (
    "-an",
    "-c:v",
    "libx264",
    "-preset",
    "medium",
    "-crf",
    "18",
    "-pix_fmt",
    "yuv420p",
    "-movflags",
    "+faststart",
    "-fps_mode",
    "cfr",
)


def claim_boundaries(recording: Assembly101Recording = RECORDING_1) -> tuple[str, ...]:
    return (
        "These files are trimmed, re-encoded derivatives of the dataset's own recordings for "
        "local review only; the 60 fps trim is lossy (libx264 crf 18), not the original bytes.",
        f"Frame 0 of both outputs is source time {recording.window_start_seconds:.3f} s (raw "
        f"60 fps frame {recording.window_start_raw_frame}) by ffmpeg's accurate seek; the proxy "
        "keeps even source frames as in every earlier proxy.",
        "No pose, annotation or model was read to make them; CC BY-NC 4.0 attribution applies "
        "and nothing here backs an accuracy claim.",
    )


CLAIM_BOUNDARIES: tuple[str, ...] = claim_boundaries(RECORDING_1)


def video_name(view: str) -> str:
    """Dataset file stem for a camera: `C10095_rgb` or `HMC_21179183_mono10bit`."""
    if view.startswith("HMC_"):
        return f"{view}_mono10bit"
    return f"{view}_rgb"


def proxy_scale_for(view: str) -> str:
    return EGO_PROXY_SCALE if view.startswith("HMC_") else STATIC_PROXY_SCALE


def window_stem(view: str, start_seconds: float, duration_seconds: float) -> str:
    end = start_seconds + duration_seconds
    return f"{video_name(view)}_{start_seconds:.3f}-{end:.3f}"


def raw60_path(
    view: str,
    derived_root: Path | None = None,
    *,
    recording: Assembly101Recording = RECORDING_1,
) -> Path:
    root = derived_root if derived_root is not None else recording.derived_root
    return root / (
        window_stem(view, recording.window_start_seconds, recording.window_duration_seconds)
        + "_raw60.mp4"
    )


def proxy_path(
    view: str,
    derived_root: Path | None = None,
    *,
    recording: Assembly101Recording = RECORDING_1,
) -> Path:
    root = derived_root if derived_root is not None else recording.derived_root
    width, height = proxy_scale_for(view).split(":")
    return root / (
        window_stem(view, recording.window_start_seconds, recording.window_duration_seconds)
        + f"_{width}x{height}_{ANALYSIS_FPS}fps.mp4"
    )


class RangeRequestRecord(VersionedModel):
    """One HTTP request ffmpeg made through the counting proxy."""

    range_header: str | None
    upstream_status: int
    bytes_sent: int = Field(ge=0)


class EncodedWindowFile(VersionedModel):
    """One produced file with its measured facts."""

    uri: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: int = Field(gt=0)
    frame_count: int = Field(gt=0)
    duration_seconds: float = Field(gt=0)
    codec: str = Field(min_length=1)
    pixel_format: str = Field(min_length=1)


class ViewWindowAcquisition(VersionedModel):
    """Everything measured while producing one view's focused window."""

    manifest_kind: Literal["assembly101_view_window_acquisition"]
    recording_id: str = Field(min_length=1)
    view: str = Field(min_length=1)
    video_name: str = Field(min_length=1)
    source_kind: Literal["hf_range", "local_file"]
    source_uri: str = Field(min_length=1)
    hf_dataset: str | None = None
    hf_revision: str | None = None
    hf_size_bytes: int | None = None
    hf_etag: str | None = None
    start_seconds: float = Field(ge=0)
    duration_seconds: float = Field(gt=0)
    raw_frame_start: int = Field(ge=0)
    raw_frame_end_exclusive: int = Field(gt=0)
    range_requests: tuple[RangeRequestRecord, ...] = ()
    bytes_transferred: int = Field(ge=0)
    fraction_of_source_transferred: float | None = None
    raw60: EncodedWindowFile
    proxy: EncodedWindowFile
    proxy_built_here: bool
    ffmpeg_version: str = Field(min_length=1)
    ffmpeg_elapsed_seconds: float = Field(ge=0)
    generated_at_utc: str = Field(min_length=1)
    license: str = Field(min_length=1)
    citation: str = Field(min_length=1)
    claim_boundaries: tuple[str, ...] = Field(min_length=1)


class _CountingProxy:
    """Forward ffmpeg's Range requests to a signed URL and count what is delivered."""

    def __init__(self, upstream_url: str) -> None:
        self.upstream_url = upstream_url
        self.records: list[RangeRequestRecord] = []
        self._lock = threading.Lock()
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                return

            def do_GET(self) -> None:  # noqa: N802
                proxy._serve(self)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        )

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}/source.mp4"

    def __enter__(self) -> _CountingProxy:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()

    @property
    def bytes_sent(self) -> int:
        with self._lock:
            return sum(record.bytes_sent for record in self.records)

    def _serve(self, handler: BaseHTTPRequestHandler) -> None:
        range_header = handler.headers.get("Range")
        headers = {"Range": range_header} if range_header else {}
        request = urllib.request.Request(self.upstream_url, headers=headers)
        sent = 0
        status = 0
        try:
            with urllib.request.urlopen(request, timeout=120) as upstream:
                status = upstream.status
                handler.send_response(status)
                for name in ("Content-Type", "Content-Length", "Content-Range", "Accept-Ranges"):
                    value = upstream.headers.get(name)
                    if value is not None:
                        handler.send_header(name, value)
                handler.send_header("Connection", "close")
                handler.end_headers()
                while True:
                    chunk = upstream.read(1 << 20)
                    if not chunk:
                        break
                    handler.wfile.write(chunk)
                    sent += len(chunk)
        except urllib.error.HTTPError as error:
            status = error.code
            handler.send_response(error.code)
            handler.send_header("Connection", "close")
            handler.end_headers()
        except (BrokenPipeError, ConnectionResetError):
            # ffmpeg closes the socket after a seek; bytes already written were delivered.
            pass
        finally:
            handler.close_connection = True
            with self._lock:
                self.records.append(
                    RangeRequestRecord(
                        range_header=range_header, upstream_status=status, bytes_sent=sent
                    )
                )


def resolve_hf_source(
    view: str, recording: Assembly101Recording = RECORDING_1
) -> tuple[str, str, int, str]:
    """Return `(hf_path, signed_location, size, etag)` for a recording view at the pin."""
    from huggingface_hub import get_hf_file_metadata, get_token, hf_hub_url

    hf_path = f"recordings/{recording.recording_id}/{video_name(view)}.mp4"
    url = hf_hub_url(DATASET_REPO, hf_path, repo_type="dataset", revision=DATASET_REVISION)
    metadata = get_hf_file_metadata(url, token=get_token())
    if metadata.size is None or metadata.etag is None:
        raise RuntimeError(f"Hugging Face returned no size/etag for {hf_path}")
    return hf_path, metadata.location, int(metadata.size), metadata.etag


def ffmpeg_version() -> str:
    completed = subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True, text=True)
    return completed.stdout.splitlines()[0].strip()


def ffmpeg_window_command(
    source: str,
    *,
    raw60_output: Path,
    proxy_output: Path | None,
    proxy_scale: str,
    start_seconds: float = FOCUSED_START_SECONDS,
    duration_seconds: float = FOCUSED_DURATION_SECONDS,
) -> list[str]:
    """One decode, two encodes: the 60 fps trim and the 30 fps proxy (same filters as before).

    With `proxy_output=None` only the trim is written (the view's proxy already exists and
    its checksum is pinned in a clip config).
    """
    trim = f"[0:v:0]trim=duration={duration_seconds},setpts=PTS-STARTPTS"
    if proxy_output is None:
        filter_graph = f"{trim}[raw]"
        outputs = ["-map", "[raw]", *PROXY_ENCODER_ARGS, str(raw60_output)]
    else:
        filter_graph = (
            f"{trim},split=2[raw][pre];"
            f"[pre]fps={ANALYSIS_FPS}:round=near,scale={proxy_scale}:flags=lanczos,setsar=1[proxy]"
        )
        outputs = [
            "-map",
            "[raw]",
            *PROXY_ENCODER_ARGS,
            str(raw60_output),
            "-map",
            "[proxy]",
            *PROXY_ENCODER_ARGS,
            str(proxy_output),
        ]
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-n",
        "-ss",
        f"{start_seconds:.3f}",
        "-i",
        source,
        "-filter_complex",
        filter_graph,
        *outputs,
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stream_facts(path: Path) -> dict[str, str]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt,duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)["streams"][0]


def describe_file(path: Path, repository_root: Path) -> EncodedWindowFile:
    frame_count, fps, (width, height) = media_probe.video_info(path, verify=True)
    facts = _stream_facts(path)
    return EncodedWindowFile(
        uri=path.resolve().relative_to(repository_root.resolve()).as_posix(),
        size_bytes=path.stat().st_size,
        sha256=_sha256(path),
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_seconds=float(facts["duration"]),
        codec=facts["codec_name"],
        pixel_format=facts["pix_fmt"],
    )


def acquire_view_window(
    view: str,
    *,
    repository_root: Path,
    local_source: Path | None = None,
    overwrite: bool = False,
    recording: Assembly101Recording = RECORDING_1,
) -> ViewWindowAcquisition:
    """Produce the raw60 trim and the proxy for one view and return the measured record."""
    repository_root = repository_root.resolve()
    derived_root = repository_root / recording.derived_root
    derived_root.mkdir(parents=True, exist_ok=True)
    raw60 = repository_root / raw60_path(view, recording=recording)
    proxy = repository_root / proxy_path(view, recording=recording)
    start_seconds = recording.window_start_seconds
    duration_seconds = recording.window_duration_seconds
    boundaries = claim_boundaries(recording)
    if raw60.exists():
        if not overwrite:
            raise FileExistsError(f"{raw60} exists; pass --overwrite to rebuild it")
        raw60.unlink()
    # An existing proxy is pinned by checksum in a clip config: keep it, build only the trim.
    proxy_output: Path | None = None if proxy.exists() else proxy
    generated_at = datetime.now(UTC).isoformat()
    version = ffmpeg_version()
    started = time.monotonic()
    if local_source is not None:
        source_path = local_source.resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"local recording is unavailable: {source_path}")
        command = ffmpeg_window_command(
            str(source_path),
            raw60_output=raw60,
            proxy_output=proxy_output,
            proxy_scale=proxy_scale_for(view),
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
        )
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"ffmpeg failed for {view}: {completed.stderr.strip()}")
        elapsed = time.monotonic() - started
        source_size = source_path.stat().st_size
        return ViewWindowAcquisition(
            manifest_kind="assembly101_view_window_acquisition",
            recording_id=recording.recording_id,
            view=view,
            video_name=video_name(view),
            source_kind="local_file",
            source_uri=source_path.relative_to(repository_root).as_posix(),
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
            raw_frame_start=recording.window_start_raw_frame,
            raw_frame_end_exclusive=recording.window_end_raw_frame_exclusive,
            bytes_transferred=0,
            fraction_of_source_transferred=None if source_size == 0 else 0.0,
            raw60=describe_file(raw60, repository_root),
            proxy=describe_file(proxy, repository_root),
            proxy_built_here=proxy_output is not None,
            ffmpeg_version=version,
            ffmpeg_elapsed_seconds=elapsed,
            generated_at_utc=generated_at,
            license=ASSEMBLY101_LICENSE,
            citation=ASSEMBLY101_CITATION,
            claim_boundaries=boundaries,
        )
    hf_path, location, size, etag = resolve_hf_source(view, recording)
    with _CountingProxy(location) as counting:
        command = ffmpeg_window_command(
            counting.url,
            raw60_output=raw60,
            proxy_output=proxy_output,
            proxy_scale=proxy_scale_for(view),
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
        )
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"ffmpeg failed for {view}: {completed.stderr.strip()}")
        elapsed = time.monotonic() - started
        records = tuple(counting.records)
        transferred = counting.bytes_sent
    return ViewWindowAcquisition(
        manifest_kind="assembly101_view_window_acquisition",
        recording_id=recording.recording_id,
        view=view,
        video_name=video_name(view),
        source_kind="hf_range",
        source_uri=hf_path,
        hf_dataset=DATASET_REPO,
        hf_revision=DATASET_REVISION,
        hf_size_bytes=size,
        hf_etag=etag,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        raw_frame_start=recording.window_start_raw_frame,
        raw_frame_end_exclusive=recording.window_end_raw_frame_exclusive,
        range_requests=records,
        bytes_transferred=transferred,
        fraction_of_source_transferred=transferred / size,
        raw60=describe_file(raw60, repository_root),
        proxy=describe_file(proxy, repository_root),
        proxy_built_here=proxy_output is not None,
        ffmpeg_version=version,
        ffmpeg_elapsed_seconds=elapsed,
        generated_at_utc=generated_at,
        license=ASSEMBLY101_LICENSE,
        citation=ASSEMBLY101_CITATION,
        claim_boundaries=boundaries,
    )


def per_view_record_path(
    view: str, repository_root: Path, recording: Assembly101Recording = RECORDING_1
) -> Path:
    return repository_root / recording.acquisition_records_root / f"{view}.json"


def load_record(
    view: str, repository_root: Path, recording: Assembly101Recording = RECORDING_1
) -> ViewWindowAcquisition:
    return ViewWindowAcquisition.model_validate_json(
        per_view_record_path(view, repository_root, recording).read_text(encoding="utf-8")
    )


ALL_STATIC_CLIP_CONFIG = Path(
    "configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_all_static_g2.json"
)
SINGLE_VIEW_CLIP_CONFIG = Path(
    "configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_g2.json"
)


def static_view_id(view: str) -> str:
    return f"static-{view.lower()}"


def view_id_for(view: str) -> str:
    """Clip-config view id: `static-c10095` or `ego-hmc21179183`."""
    if view.startswith("HMC_"):
        return f"ego-{view.replace('_', '').lower()}"
    return static_view_id(view)


def _proxy_from_record(
    record: ViewWindowAcquisition, recording: Assembly101Recording, repository_root: Path
):
    """A `VideoProxy` for one acquired view; remote sources cite the HF path and LFS etag."""
    from .schemas import FrameRange, RawVideoSource, VideoDimensions, VideoProxy

    view_id = view_id_for(record.view)
    raw_range = FrameRange(
        start_frame=recording.window_start_raw_frame,
        end_frame_exclusive=recording.window_end_raw_frame_exclusive,
    )
    if record.source_kind == "hf_range":
        if record.hf_etag is None:
            raise ValueError(f"{record.view} carries no Hugging Face etag")
        raw_uri = f"hf://datasets/{DATASET_REPO}@{DATASET_REVISION}/{record.source_uri}"
        raw_checksum = record.hf_etag
    else:
        raw_uri = record.source_uri
        raw_checksum = _sha256(repository_root / raw_uri)
    raw_width, raw_height = (636, 480) if record.view.startswith("HMC_") else (1920, 1080)
    return VideoProxy(
        view_id=view_id,
        raw_source=RawVideoSource(
            view_id=view_id,
            raw_uri=raw_uri,
            checksum_sha256=raw_checksum,
            dimensions=VideoDimensions(width=raw_width, height=raw_height),
            fps=SOURCE_FPS,
            raw_frame_range=raw_range,
        ),
        proxy_uri=record.proxy.uri,
        checksum_sha256=record.proxy.sha256,
        dimensions=VideoDimensions(width=record.proxy.width, height=record.proxy.height),
        fps=record.proxy.fps,
        frame_count=record.proxy.frame_count,
        codec="h264",
        crf=18,
        preset="medium",
        pixel_format="yuv420p",
    )


def _timing(source_offset_seconds: float):
    from .schemas import ClockMapping, ClockName, ClockSet, ClockSpec, TimingModel

    clocks = ClockSet(
        clocks=(
            ClockSpec(name=ClockName.SOURCE, fps=SOURCE_FPS),
            ClockSpec(name=ClockName.ANALYSIS, fps=ANALYSIS_FPS),
            ClockSpec(name=ClockName.ANNOTATION, fps=30),
            ClockSpec(name=ClockName.POSE, fps=60),
        )
    )
    return TimingModel(
        clocks=clocks,
        mappings=tuple(
            ClockMapping(clock=spec.name, source_offset_seconds=source_offset_seconds, scale=1.0)
            for spec in clocks.clocks
        ),
    )


def build_clip_config(
    recording: Assembly101Recording,
    views: tuple[str, ...],
    *,
    repository_root: Path,
    clip_id: str,
    g1_scope: str,
    g2_scope: str,
    approved_by: str,
):
    """A `G2PreprocessingManifest` for `views` of `recording`, from the acquisition records.

    The clip asset is the recording's primary static view when it is among the views, else
    the first view.  Raw sources of remote views cite the `hf://` path at the pinned revision
    with the LFS etag (the file's SHA-256) as checksum.
    """
    from .schemas import (
        ClipGateApproval,
        ClipManifest,
        EncodedAssetInput,
        FrameRange,
        G2PreprocessingManifest,
        MethodState,
        TimeInterval,
    )

    repository_root = repository_root.resolve()
    proxies = tuple(
        _proxy_from_record(
            load_record(view, repository_root, recording), recording, repository_root
        )
        for view in views
    )
    primary = next(
        (p for p in proxies if p.view_id == view_id_for(recording.primary_static_view)),
        proxies[0],
    )
    start = recording.window_start_seconds
    end = recording.window_end_seconds
    manifest = G2PreprocessingManifest(
        manifest_kind="assembly101_g2_preprocessing",
        clip=ClipManifest(
            clip_id=clip_id,
            source_name="Assembly101",
            source_license=(
                "CC BY-NC 4.0; dataset terms and permitted use recorded in the local provenance "
                "ledger"
            ),
            asset=EncodedAssetInput(
                uri=primary.proxy_uri,
                media_type="video/mp4",
                checksum_sha256=primary.checksum_sha256,
            ),
            timing=_timing(start),
            source_duration_seconds=recording.window_duration_seconds,
            views=tuple(p.view_id for p in proxies),
            provenance_status=MethodState.APPROVED,
        ),
        hf_dataset=DATASET_REPO,
        hf_revision=DATASET_REVISION,
        provenance_ledger_uri=recording.acquisition_report.as_posix(),
        g1=ClipGateApproval(state="approved", approved_by=approved_by, scope=g1_scope),
        g2=ClipGateApproval(state="approved", approved_by=approved_by, scope=g2_scope),
        raw_timing=_timing(0.0),
        proxy_timing=_timing(start),
        source_interval=TimeInterval(start_seconds=start, end_seconds=end),
        raw_frame_range=FrameRange(
            start_frame=recording.window_start_raw_frame,
            end_frame_exclusive=recording.window_end_raw_frame_exclusive,
        ),
        analysis_frame_range=FrameRange(
            start_frame=round(start * ANALYSIS_FPS), end_frame_exclusive=round(end * ANALYSIS_FPS)
        ),
        proxy_frame_range=FrameRange(
            start_frame=0, end_frame_exclusive=recording.window_proxy_frame_count
        ),
        scaling_policy="preserve_aspect_ratio_height_720",
        proxies=proxies,
    )
    return G2PreprocessingManifest.model_validate(manifest.model_dump())


def _core_span_text(recording: Assembly101Recording) -> str:
    core_start, core_end = recording.core_proxy_frame_range
    return (
        f"source {recording.window_start_seconds:.3f}-{recording.window_end_seconds:.3f} s; the "
        f"{recording.core_duration_seconds:.0f} s analysis span is source "
        f"{recording.core_start_seconds:.3f}-"
        f"{recording.core_start_seconds + recording.core_duration_seconds:.3f} s = proxy frames "
        f"[{core_start}, {core_end})"
    )


def write_recording_clip_configs(
    recording: Assembly101Recording,
    *,
    repository_root: Path,
    approved_by: str = "user (Track C plan of Sep 20; automatic pipeline on a second recording)",
) -> tuple[Path, ...]:
    """Write the all-static config and one config per fetched ego view for a registry recording."""
    repository_root = repository_root.resolve()
    written: list[Path] = []
    slug = recording.label.replace("_", "-")
    static = build_clip_config(
        recording,
        recording.static_views,
        repository_root=repository_root,
        clip_id=f"assembly101-{slug}-four-part-reassembly-focused-all-static-g2",
        g1_scope=(
            f"all eight Assembly101 static views of recording {recording.recording_id} "
            f"(toy {recording.toy_id}, subject {recording.subject_id}), fetched over HTTP Range "
            f"for {_core_span_text(recording)}; see docs/SOURCES.md"
        ),
        g2_scope=(
            f"{recording.window_duration_seconds:.1f}-second four-part reassembly window on a "
            "recording the system has not seen: coarse actions attach interior / screw chassis / "
            "attach body inside the analysis span; seeds are to be automatic, not human"
        ),
        approved_by=approved_by,
    )
    target = repository_root / recording.all_static_clip_config
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(static.model_dump_json(indent=2) + "\n", encoding="utf-8")
    written.append(target)
    for view in recording.ego_views:
        ego_slug = view.lower().replace("_", "")
        ego = build_clip_config(
            recording,
            (view,),
            repository_root=repository_root,
            clip_id=f"assembly101-{slug}-four-part-reassembly-focused-ego-{ego_slug}-g2",
            g1_scope=(
                f"monochrome ego view {view} of recording {recording.recording_id}, fetched over "
                f"HTTP Range for {_core_span_text(recording)}; 954x720 proxy by the Track 0 recipe"
            ),
            g2_scope=(
                f"source-aligned {recording.window_duration_seconds:.1f}-second ego interval "
                "matching the all-static window of the same recording; ego views join only behind "
                "the pose gate"
            ),
            approved_by=approved_by,
        )
        target = repository_root / recording.ego_clip_config(view)
        target.write_text(ego.model_dump_json(indent=2) + "\n", encoding="utf-8")
        written.append(target)
    return tuple(written)


def write_all_static_clip_config(
    repository_root: Path,
    *,
    output: Path = ALL_STATIC_CLIP_CONFIG,
    template: Path = SINGLE_VIEW_CLIP_CONFIG,
) -> Path:
    """Derive the eight-static-view clip config from the approved single-view one.

    The single-view config is left untouched because several review builders assert
    `clip.views == ("static-c10379",)` on run manifests that embed it.  Proxy checksums come
    from the per-view acquisition records; a remote view's raw source is its Hugging Face path
    at the pinned revision and its checksum is the LFS etag, which is the file's SHA-256.
    Recording 1 only; other recordings use `write_recording_clip_configs`.
    """
    from .schemas import (
        G2PreprocessingManifest,
        RawVideoSource,
        VideoDimensions,
        VideoProxy,
    )

    repository_root = repository_root.resolve()
    base = G2PreprocessingManifest.model_validate_json(
        (repository_root / template).read_text(encoding="utf-8")
    )
    existing = {proxy.view_id: proxy for proxy in base.proxies}
    proxies: list[VideoProxy] = []
    for view in STATIC_VIEWS:
        view_id = static_view_id(view)
        if view_id in existing:
            proxies.append(existing[view_id])
            continue
        record = load_record(view, repository_root)
        if record.source_kind != "hf_range" or record.hf_etag is None:
            raise ValueError(f"{view} was not acquired from the Hugging Face pin")
        proxies.append(
            VideoProxy(
                view_id=view_id,
                raw_source=RawVideoSource(
                    view_id=view_id,
                    raw_uri=f"hf://datasets/{DATASET_REPO}@{DATASET_REVISION}/{record.source_uri}",
                    checksum_sha256=record.hf_etag,
                    dimensions=VideoDimensions(width=1920, height=1080),
                    fps=SOURCE_FPS,
                    raw_frame_range=base.raw_frame_range,
                ),
                proxy_uri=record.proxy.uri,
                checksum_sha256=record.proxy.sha256,
                dimensions=VideoDimensions(width=record.proxy.width, height=record.proxy.height),
                fps=record.proxy.fps,
                frame_count=record.proxy.frame_count,
                codec="h264",
                crf=18,
                preset="medium",
                pixel_format="yuv420p",
            )
        )
    clip = base.clip.model_copy(
        update={
            "clip_id": "assembly101-nusar-9033-four-part-reassembly-focused-all-static-g2",
            "views": tuple(proxy.view_id for proxy in proxies),
        }
    )
    manifest = base.model_copy(
        update={
            "clip": clip,
            "proxies": tuple(proxies),
            "g1": base.g1.model_copy(
                update={
                    "scope": (
                        "all eight Assembly101 static views of the pinned recording, focused "
                        "window only (source 294.000-386.700 s), fetched over HTTP Range; see "
                        "docs/SOURCES.md"
                    )
                }
            ),
        }
    )
    G2PreprocessingManifest.model_validate(manifest.model_dump())
    target = repository_root / output
    target.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return target


def write_acquisition_report(
    repository_root: Path, recording: Assembly101Recording = RECORDING_1
) -> tuple[Path, Path]:
    """Write the human report and the JSON manifest for every per-view record present."""
    repository_root = repository_root.resolve()
    all_views = (*recording.static_views, *recording.ego_views_on_hub)
    records = [
        load_record(view, repository_root, recording)
        for view in all_views
        if per_view_record_path(view, repository_root, recording).is_file()
    ]
    manifest_path = repository_root / recording.acquisition_manifest
    manifest_path.write_text(
        json.dumps(
            {
                "manifest_kind": "assembly101_focused_window_acquisition",
                "recording_id": recording.recording_id,
                "hf_dataset": DATASET_REPO,
                "hf_revision": DATASET_REVISION,
                "license": ASSEMBLY101_LICENSE,
                "citation": ASSEMBLY101_CITATION,
                "window": {
                    "start_seconds": recording.window_start_seconds,
                    "duration_seconds": recording.window_duration_seconds,
                    "raw_frame_range": [
                        recording.window_start_raw_frame,
                        recording.window_end_raw_frame_exclusive,
                    ],
                    "core_span_seconds": [
                        recording.core_start_seconds,
                        recording.core_start_seconds + recording.core_duration_seconds,
                    ],
                    "core_proxy_frame_range": list(recording.core_proxy_frame_range),
                },
                "claim_boundaries": claim_boundaries(recording),
                "views": [json.loads(record.model_dump_json()) for record in records],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Assembly101 focused-window acquisition: all static views and ego trims",
        "",
        f"Recording: `{recording.recording_id}` (toy {recording.toy_id}, subject "
        f"{recording.subject_id})",
        f"Dataset: Hugging Face `{DATASET_REPO}` @ revision `{DATASET_REVISION}` "
        f"({ASSEMBLY101_LICENSE}; {ASSEMBLY101_CITATION})",
        f"Generated: {datetime.now(UTC).isoformat()}. Machine-readable companion: "
        f"`{MANIFEST_NAME}`; per-view records under `static_views_focused_acquisition/`.",
        f"Window: {_core_span_text(recording)}.",
        "",
        "## Method",
        "",
        "- `battle-fetch-assembly101-view` resolves each recording's signed CDN URL for the pinned"
        " revision (`hf_hub_url` + `get_hf_file_metadata`; the token is read by"
        " `huggingface_hub.get_token()` and never printed) and points ffmpeg at a local counting"
        f" proxy that forwards its HTTP Range requests. ffmpeg seeks to source"
        f" {recording.window_start_seconds:.3f} s and, in one"
        " decode, writes a 60 fps trim at sensor resolution (`*_raw60.mp4`, for clock-offset scans)"
        " and the standard 1280x720 CFR-30 proxy with the same filter chain and encoder settings as"
        " `scripts/create_assembly101_four_part_reassembly_proxies.sh`.",
        "- The full 1.6-4 GB files were never downloaded; `bytes_transferred` counts the bytes the"
        " proxy delivered to ffmpeg (the moov atom at the head plus the mdat range for the window;"
        " a lower bound on network bytes by at most the socket buffer).",
        "- Recordings already on disk (C10379 and the four HMC cameras) went through the same"
        " ffmpeg recipe from the local file; their existing proxies were kept (checksums are pinned"
        " in clip configs) and only the trims were added, plus new 954x720 proxies for the three"
        " HMC cameras that had none at 294.000 s.",
        "- Frame alignment check on C10379 (local full file available): trim frame t equals raw"
        " frame 17640 + t (downscaled MAE minimum at d = 0 for t = 0..2) and proxy frame p equals"
        " trim frame 2p. The same ffmpeg seek/filter path was used for every view.",
        "- No pose, annotation or model was read. These are review derivatives under CC BY-NC 4.0;"
        " nothing here backs an accuracy claim.",
        "",
        "## Per-view results",
        "",
        "| view | source | HF size (B) | bytes transferred | fraction | ranges | "
        "raw60 frames/fps | raw60 size (B) | raw60 SHA-256 | proxy | proxy frames | "
        "proxy size (B) | proxy SHA-256 | ffmpeg s |",
        "|---|---|---:|---:|---:|---|---|---:|---|---|---:|---:|---|---:|",
    ]
    for record in records:
        ranges = "; ".join(
            f"`{r.range_header}` -> {r.bytes_sent:,} B" for r in record.range_requests
        )
        fraction = record.fraction_of_source_transferred
        fraction_text = "" if fraction is None else f"{fraction:.3f}"
        lines.append(
            f"| {record.view} | {record.source_kind} `{record.source_uri}` | "
            f"{record.hf_size_bytes or 0:,} | {record.bytes_transferred:,} | "
            f"{fraction_text} | "
            f"{ranges or 'local'} | {record.raw60.frame_count} @ {record.raw60.fps} | "
            f"{record.raw60.size_bytes:,} | `{record.raw60.sha256}` | "
            f"`{Path(record.proxy.uri).name}`{'' if record.proxy_built_here else ' (existing)'} | "
            f"{record.proxy.frame_count} | {record.proxy.size_bytes:,} | `{record.proxy.sha256}` | "
            f"{record.ffmpeg_elapsed_seconds:.0f} |"
        )
    total = sum(r.bytes_transferred for r in records)
    lines += [
        "",
        f"Total bytes transferred from the Hugging Face CDN: {total:,} B "
        f"({total / 1e6:,.0f} MB) for {sum(r.source_kind == 'hf_range' for r in records)} "
        "remote views.",
        "",
        "## Files",
        "",
        "- Trims: `data/derived/assembly101/<recording>/<view>_<start>-<end>_raw60.mp4`"
        f" (1920x1080 or 636x480, 60 fps, {recording.window_raw_frame_count:,} frames, libx264"
        " crf 18).",
        "- Proxies: `data/derived/assembly101/<recording>/<view>_<start>-<end>_1280x720_30fps.mp4`"
        f" (static) and `_954x720_30fps.mp4` (ego), {recording.window_proxy_frame_count:,} frames.",
        "- Tracked clip config for the eight static proxies: "
        f"`{recording.all_static_clip_config}`.",
    ]
    report_path = repository_root / recording.acquisition_report
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--view",
        action="append",
        default=[],
        help="Camera name, e.g. C10095 or HMC_21179183; repeatable.",
    )
    parser.add_argument(
        "--recording",
        help="Registry label or recording id (configs/assembly101/recordings.json); "
        "default: recording 1.",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Read the recording already under data/raw instead of the Hugging Face CDN.",
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="After fetching, write the acquisition report/manifest and the clip config(s).",
    )
    args = parser.parse_args()
    if not args.view and not args.write_report:
        parser.error("pass --view and/or --write-report")
    recording = get_recording(args.recording, args.repository_root)
    for view in args.view:
        local = None
        if args.local:
            local = (
                args.repository_root / recording.local_recordings_root / f"{video_name(view)}.mp4"
            )
        record = acquire_view_window(
            view,
            repository_root=args.repository_root,
            local_source=local,
            overwrite=args.overwrite,
            recording=recording,
        )
        output = per_view_record_path(view, args.repository_root.resolve(), recording)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
        print(
            f"{view}: {record.bytes_transferred:,} B transferred, raw60 "
            f"{record.raw60.frame_count} frames, proxy {record.proxy.frame_count} frames, "
            f"{record.ffmpeg_elapsed_seconds:.0f} s -> {output}"
        )
    if args.write_report:
        report, manifest = write_acquisition_report(args.repository_root, recording)
        print(f"wrote {report}\nwrote {manifest}")
        if recording.recording_id == RECORDING_ID:
            print(f"wrote {write_all_static_clip_config(args.repository_root)}")
        else:
            for path in write_recording_clip_configs(
                recording, repository_root=args.repository_root
            ):
                print(f"wrote {path}")


if __name__ == "__main__":
    main()
