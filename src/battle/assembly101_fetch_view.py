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
from .schemas import VersionedModel

RECORDING_ID = "nusar-2021_action_both_9033-c02a_9033_user_id_2021-02-04_140532"
DATASET_REPO = "cvml-nus/assembly101"
DATASET_REVISION = "bfc15ea5e3f0bc8f8c232af6c1b45aa137a9d967"
RAW_ROOT = Path("data/raw/assembly101") / RECORDING_ID
LOCAL_RECORDINGS = RAW_ROOT / "recordings" / RECORDING_ID
DERIVED_ROOT = Path("data/derived/assembly101") / RECORDING_ID
REPORT_NAME = "static_views_focused_acquisition_report.md"
MANIFEST_NAME = "static_views_focused_acquisition_manifest.json"

FOCUSED_START_SECONDS = 294.0
FOCUSED_DURATION_SECONDS = 92.7
SOURCE_FPS = 60
ANALYSIS_FPS = 30

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
CLAIM_BOUNDARIES: tuple[str, ...] = (
    "These files are trimmed, re-encoded derivatives of the dataset's own recordings for "
    "local review only; the 60 fps trim is lossy (libx264 crf 18), not the original bytes.",
    "Frame 0 of both outputs is source time 294.000 s (raw 60 fps frame 17640) by ffmpeg's "
    "accurate seek; the proxy keeps even source frames as in every earlier proxy.",
    "No pose, annotation or model was read to make them; CC BY-NC 4.0 attribution applies "
    "and nothing here backs an accuracy claim.",
)


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


def raw60_path(view: str, derived_root: Path = DERIVED_ROOT) -> Path:
    return derived_root / (
        window_stem(view, FOCUSED_START_SECONDS, FOCUSED_DURATION_SECONDS) + "_raw60.mp4"
    )


def proxy_path(view: str, derived_root: Path = DERIVED_ROOT) -> Path:
    width, height = proxy_scale_for(view).split(":")
    return derived_root / (
        window_stem(view, FOCUSED_START_SECONDS, FOCUSED_DURATION_SECONDS)
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


def resolve_hf_source(view: str) -> tuple[str, str, int, str]:
    """Return `(hf_path, signed_location, size, etag)` for a recording view at the pin."""
    from huggingface_hub import get_hf_file_metadata, get_token, hf_hub_url

    hf_path = f"recordings/{RECORDING_ID}/{video_name(view)}.mp4"
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
) -> ViewWindowAcquisition:
    """Produce the raw60 trim and the proxy for one view and return the measured record."""
    repository_root = repository_root.resolve()
    derived_root = repository_root / DERIVED_ROOT
    derived_root.mkdir(parents=True, exist_ok=True)
    raw60 = repository_root / raw60_path(view)
    proxy = repository_root / proxy_path(view)
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
        )
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"ffmpeg failed for {view}: {completed.stderr.strip()}")
        elapsed = time.monotonic() - started
        source_size = source_path.stat().st_size
        return ViewWindowAcquisition(
            manifest_kind="assembly101_view_window_acquisition",
            recording_id=RECORDING_ID,
            view=view,
            video_name=video_name(view),
            source_kind="local_file",
            source_uri=source_path.relative_to(repository_root).as_posix(),
            start_seconds=FOCUSED_START_SECONDS,
            duration_seconds=FOCUSED_DURATION_SECONDS,
            raw_frame_start=round(FOCUSED_START_SECONDS * SOURCE_FPS),
            raw_frame_end_exclusive=round(
                (FOCUSED_START_SECONDS + FOCUSED_DURATION_SECONDS) * SOURCE_FPS
            ),
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
            claim_boundaries=CLAIM_BOUNDARIES,
        )
    hf_path, location, size, etag = resolve_hf_source(view)
    with _CountingProxy(location) as counting:
        command = ffmpeg_window_command(
            counting.url,
            raw60_output=raw60,
            proxy_output=proxy_output,
            proxy_scale=proxy_scale_for(view),
        )
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"ffmpeg failed for {view}: {completed.stderr.strip()}")
        elapsed = time.monotonic() - started
        records = tuple(counting.records)
        transferred = counting.bytes_sent
    return ViewWindowAcquisition(
        manifest_kind="assembly101_view_window_acquisition",
        recording_id=RECORDING_ID,
        view=view,
        video_name=video_name(view),
        source_kind="hf_range",
        source_uri=hf_path,
        hf_dataset=DATASET_REPO,
        hf_revision=DATASET_REVISION,
        hf_size_bytes=size,
        hf_etag=etag,
        start_seconds=FOCUSED_START_SECONDS,
        duration_seconds=FOCUSED_DURATION_SECONDS,
        raw_frame_start=round(FOCUSED_START_SECONDS * SOURCE_FPS),
        raw_frame_end_exclusive=round(
            (FOCUSED_START_SECONDS + FOCUSED_DURATION_SECONDS) * SOURCE_FPS
        ),
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
        claim_boundaries=CLAIM_BOUNDARIES,
    )


def per_view_record_path(view: str, repository_root: Path) -> Path:
    return repository_root / RAW_ROOT / "static_views_focused_acquisition" / f"{view}.json"


def load_record(view: str, repository_root: Path) -> ViewWindowAcquisition:
    return ViewWindowAcquisition.model_validate_json(
        per_view_record_path(view, repository_root).read_text(encoding="utf-8")
    )


ALL_STATIC_CLIP_CONFIG = Path(
    "configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_all_static_g2.json"
)
SINGLE_VIEW_CLIP_CONFIG = Path(
    "configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_g2.json"
)


def static_view_id(view: str) -> str:
    return f"static-{view.lower()}"


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


def write_acquisition_report(repository_root: Path) -> tuple[Path, Path]:
    """Write the human report and the JSON manifest for every per-view record present."""
    repository_root = repository_root.resolve()
    records = [
        load_record(view, repository_root)
        for view in (*STATIC_VIEWS, *EGO_VIEWS)
        if per_view_record_path(view, repository_root).is_file()
    ]
    manifest_path = repository_root / RAW_ROOT / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(
            {
                "manifest_kind": "assembly101_focused_window_acquisition",
                "recording_id": RECORDING_ID,
                "hf_dataset": DATASET_REPO,
                "hf_revision": DATASET_REVISION,
                "license": ASSEMBLY101_LICENSE,
                "citation": ASSEMBLY101_CITATION,
                "window": {
                    "start_seconds": FOCUSED_START_SECONDS,
                    "duration_seconds": FOCUSED_DURATION_SECONDS,
                    "raw_frame_range": [
                        round(FOCUSED_START_SECONDS * SOURCE_FPS),
                        round((FOCUSED_START_SECONDS + FOCUSED_DURATION_SECONDS) * SOURCE_FPS),
                    ],
                },
                "claim_boundaries": CLAIM_BOUNDARIES,
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
        f"Recording: `{RECORDING_ID}`",
        f"Dataset: Hugging Face `{DATASET_REPO}` @ revision `{DATASET_REVISION}` "
        f"({ASSEMBLY101_LICENSE}; {ASSEMBLY101_CITATION})",
        f"Generated: {datetime.now(UTC).isoformat()}. Machine-readable companion: "
        f"`{MANIFEST_NAME}`; per-view records under `static_views_focused_acquisition/`.",
        "",
        "## Method",
        "",
        "- `battle-fetch-assembly101-view` resolves each recording's signed CDN URL for the pinned"
        " revision (`hf_hub_url` + `get_hf_file_metadata`; the token is read by"
        " `huggingface_hub.get_token()` and never printed) and points ffmpeg at a local counting"
        " proxy that forwards its HTTP Range requests. ffmpeg seeks to source 294.000 s and, in one"
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
        " (1920x1080 or 636x480, 60 fps, 5,562 frames, libx264 crf 18).",
        "- Proxies: `data/derived/assembly101/<recording>/<view>_<start>-<end>_1280x720_30fps.mp4`"
        " (static) and `_954x720_30fps.mp4` (ego), 2,781 frames.",
        f"- Tracked clip config for the eight static proxies: `{ALL_STATIC_CLIP_CONFIG}`.",
    ]
    report_path = repository_root / RAW_ROOT / REPORT_NAME
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
        "--local",
        action="store_true",
        help="Read the recording already under data/raw instead of the Hugging Face CDN.",
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="After fetching, write the acquisition report/manifest and the all-static config.",
    )
    args = parser.parse_args()
    if not args.view and not args.write_report:
        parser.error("pass --view and/or --write-report")
    for view in args.view:
        local = None
        if args.local:
            local = args.repository_root / LOCAL_RECORDINGS / f"{video_name(view)}.mp4"
        record = acquire_view_window(
            view,
            repository_root=args.repository_root,
            local_source=local,
            overwrite=args.overwrite,
        )
        output = per_view_record_path(view, args.repository_root.resolve())
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
        print(
            f"{view}: {record.bytes_transferred:,} B transferred, raw60 "
            f"{record.raw60.frame_count} frames, proxy {record.proxy.frame_count} frames, "
            f"{record.ffmpeg_elapsed_seconds:.0f} s -> {output}"
        )
    if args.write_report:
        report, manifest = write_acquisition_report(args.repository_root)
        config = write_all_static_clip_config(args.repository_root)
        print(f"wrote {report}\nwrote {manifest}\nwrote {config}")


if __name__ == "__main__":
    main()
