"""Selective acquisition of one recording's Assembly101 poses and annotations (network, no GPU).

Port of the Sep 17 ad-hoc scripts kept under recording 1's raw tree
(`selective_acquisition_scripts/extract_poses.py`, `stream_fg.py`) onto the recording
registry, so a second recording can be prepared with one command:

- the coarse labels file and the small lookup tables are downloaded whole (they are KB);
- the fine-grained split CSVs are streamed once and only this recording's rows are kept,
  with the full-file SHA-256 computed on the stream and compared with the LFS etag;
- the ten `AssemblyPoses.zip` members for the recording are range-extracted from the 72 GB
  archive (central directory read from the tail, one range per member, CRC-32 verified);
- the dataset's 2D landmarks for the fetched window are written to a compact `.npz` per
  view so the clock-offset scans and camera fits never re-read the 1 GB JSON.

Everything fetched is dataset context under CC BY-NC 4.0; nothing here is ground truth
for any method in this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
import zlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field

from .assembly101_pose_schemas import ASSEMBLY101_CITATION, ASSEMBLY101_LICENSE
from .assembly101_recordings import (
    DATASET_REPO,
    DATASET_REVISION,
    Assembly101Recording,
    get_recording,
)
from .schemas import VersionedModel
from .zip_range import RangeReader, ZipMember, list_remote_zip_members, read_remote_zip_member

POSES_ARCHIVE = "AssemblyPoses.zip"
FINE_GRAINED_SPLITS: tuple[str, ...] = ("train", "validation", "test")
SMALL_TABLES: tuple[str, ...] = (
    "annotations/fine-grained-annotations/actions.csv",
    "annotations/fine-grained-annotations/head_actions.txt",
    "annotations/coarse-annotations/actions.csv",
    "annotations/coarse-annotations/coarse_seq_views.txt",
)
CLAIM_BOUNDARIES: tuple[str, ...] = (
    "Dataset poses come from Assembly101's own multi-view tracker with a fixed-scale hand "
    "model; the fine-grained and coarse segments are the dataset's human annotations. All of "
    "it is external review context, not ground truth for any method compared here.",
    "Only this recording's members and rows were transferred; the 72 GB pose archive and the "
    "full split CSVs were never stored. CC BY-NC 4.0 attribution applies.",
)


class FetchedFile(VersionedModel):
    hf_path: str = Field(min_length=1)
    local_path: str = Field(min_length=1)
    hf_size_bytes: int | None = None
    hf_etag: str | None = None
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    etag_matches_sha256: bool | None = None


class StreamedSplit(VersionedModel):
    split: str
    hf_path: str
    hf_size_bytes: int = Field(ge=0)
    hf_etag: str = Field(min_length=1)
    streamed_bytes: int = Field(ge=0)
    full_file_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    etag_matches_sha256: bool
    total_rows: int = Field(ge=0)
    rows_for_recording: int = Field(ge=0)
    local_filtered_path: str
    local_filtered_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    seconds: float = Field(ge=0)


class ExtractedMember(VersionedModel):
    member: str
    compressed_size: int = Field(ge=0)
    uncompressed_size: int = Field(ge=0)
    compression_method: int
    crc32: str
    local_header_offset: int = Field(ge=0)
    local_path: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source: Literal["range-extract", "already-present"]
    seconds: float = Field(ge=0)


class Landmarks2DWindow(VersionedModel):
    local_path: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    size_bytes: int = Field(ge=0)
    pose_frame_range: tuple[int, int]
    frames: int = Field(ge=1)
    view_keys: tuple[str, ...]
    missing_pose_frames: int = Field(ge=0)


class PosesAcquisition(VersionedModel):
    manifest_kind: Literal["assembly101_poses_annotations_acquisition"]
    recording_id: str
    hf_dataset: str
    hf_revision: str
    generated_at_utc: str
    archive: str
    archive_size_bytes: int = Field(ge=0)
    archive_etag: str
    archive_member_count: int = Field(ge=0)
    members: tuple[ExtractedMember, ...]
    coarse_labels: FetchedFile | None
    small_tables: tuple[FetchedFile, ...]
    fine_grained: tuple[StreamedSplit, ...]
    fine_grained_split_with_rows: str | None
    landmarks2d_window: Landmarks2DWindow | None
    bytes_transferred: int = Field(ge=0)
    license: str
    citation: str
    claim_boundaries: tuple[str, ...]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _token_headers() -> dict[str, str]:
    from huggingface_hub import get_token

    token = get_token()
    return {"Authorization": f"Bearer {token}"} if token else {}


def _hf_url_and_metadata(hf_path: str) -> tuple[str, int | None, str | None]:
    from huggingface_hub import get_hf_file_metadata, get_token, hf_hub_url

    url = hf_hub_url(DATASET_REPO, hf_path, repo_type="dataset", revision=DATASET_REVISION)
    metadata = get_hf_file_metadata(url, token=get_token())
    return url, metadata.size, metadata.etag


def _etag_matches(etag: str | None, sha256: str, data: bytes | None = None) -> bool | None:
    """LFS etags are the file's SHA-256; small non-LFS files carry the git blob SHA-1, which
    is checked against `data` when given.  None means the etag could not be checked."""
    if etag is None:
        return None
    cleaned = etag.strip('"').lower()
    if len(cleaned) == 64:
        return cleaned == sha256
    if len(cleaned) == 40 and data is not None:
        blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()  # noqa: S324
        return cleaned == blob
    return None


def fetch_small_file(hf_path: str, target: Path, repository_root: Path) -> FetchedFile:
    url, size, etag = _hf_url_and_metadata(hf_path)
    with urllib.request.urlopen(
        urllib.request.Request(url, headers=_token_headers()), timeout=300
    ) as response:
        data = response.read()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    digest = _sha256_bytes(data)
    return FetchedFile(
        hf_path=hf_path,
        local_path=target.resolve().relative_to(repository_root).as_posix(),
        hf_size_bytes=size,
        hf_etag=etag,
        size_bytes=len(data),
        sha256=digest,
        etag_matches_sha256=_etag_matches(etag, digest, data),
    )


def stream_fine_grained_split(
    split: str, recording: Assembly101Recording, repository_root: Path
) -> StreamedSplit:
    """Stream one split CSV and keep only lines naming the recording (header preserved)."""
    hf_path = f"annotations/fine-grained-annotations/{split}.csv"
    url, size, etag = _hf_url_and_metadata(hf_path)
    if size is None or etag is None:
        raise RuntimeError(f"Hugging Face returned no size/etag for {hf_path}")
    needle = recording.recording_id.encode()
    digest = hashlib.sha256()
    total_rows = 0
    kept: list[bytes] = []
    header: bytes | None = None
    streamed = 0
    started = time.monotonic()
    with urllib.request.urlopen(
        urllib.request.Request(url, headers=_token_headers()), timeout=600
    ) as response:
        buffer = b""
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
            streamed += len(chunk)
            buffer += chunk
            lines = buffer.split(b"\n")
            buffer = lines.pop()
            for line in lines:
                if header is None:
                    header = line
                    continue
                if not line.strip():
                    continue
                total_rows += 1
                if needle in line:
                    kept.append(line)
        if buffer.strip():
            if header is None:
                header = buffer
            else:
                total_rows += 1
                if needle in buffer:
                    kept.append(buffer)
    if header is None:
        raise RuntimeError(f"{hf_path} was empty")
    target = (
        repository_root / recording.fine_grained_root / f"{split}__{recording.recording_id}.csv"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(header + b"\n" + b"\n".join(kept) + (b"\n" if kept else b""))
    full = digest.hexdigest()
    return StreamedSplit(
        split=split,
        hf_path=hf_path,
        hf_size_bytes=size,
        hf_etag=etag,
        streamed_bytes=streamed,
        full_file_sha256=full,
        etag_matches_sha256=bool(_etag_matches(etag, full)),
        total_rows=total_rows,
        rows_for_recording=len(kept),
        local_filtered_path=target.resolve().relative_to(repository_root).as_posix(),
        local_filtered_sha256=_sha256_bytes(target.read_bytes()),
        seconds=time.monotonic() - started,
    )


def extract_pose_members(
    recording: Assembly101Recording, repository_root: Path
) -> tuple[tuple[ExtractedMember, ...], int, str, int, int]:
    """Range-extract every archive member naming the recording; returns members and facts."""
    url, size, etag = _hf_url_and_metadata(POSES_ARCHIVE)
    if size is None or etag is None:
        raise RuntimeError("Hugging Face returned no size/etag for AssemblyPoses.zip")
    reader = RangeReader(url, size=size, headers=_token_headers())
    members = list_remote_zip_members(reader)
    wanted: list[ZipMember] = [m for m in members if recording.recording_id in m.filename]
    if not wanted:
        raise FileNotFoundError(f"{recording.recording_id} has no member in {POSES_ARCHIVE}")
    out_root = repository_root / recording.raw_root / "AssemblyPoses_selective"
    records: list[ExtractedMember] = []
    transferred = 0
    for member in sorted(wanted, key=lambda m: m.uncompressed_size):
        target = out_root / member.filename
        target.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        source: Literal["range-extract", "already-present"] = "range-extract"
        if target.exists() and zlib.crc32(target.read_bytes()) & 0xFFFFFFFF == member.crc32:
            source = "already-present"
            raw = target.read_bytes()
        else:
            raw = read_remote_zip_member(reader, member)
            target.write_bytes(raw)
            transferred += member.compressed_size
        records.append(
            ExtractedMember(
                member=member.filename,
                compressed_size=member.compressed_size,
                uncompressed_size=member.uncompressed_size,
                compression_method=member.compression_method,
                crc32=f"{member.crc32:08x}",
                local_header_offset=member.local_header_offset,
                local_path=target.resolve().relative_to(repository_root).as_posix(),
                sha256=_sha256_bytes(raw),
                source=source,
                seconds=time.monotonic() - started,
            )
        )
        print(
            f"  {member.filename.split('/')[-2]}: {member.uncompressed_size:,} B "
            f"({member.compressed_size:,} compressed) [{source}] {records[-1].seconds:.1f} s",
            flush=True,
        )
    return tuple(records), size, etag, len(members), transferred


def write_landmarks2d_window(
    recording: Assembly101Recording, repository_root: Path
) -> Landmarks2DWindow:
    """Compact the shipped 2D landmarks for the fetched window: (F, 2, 21, 2) float32 per view."""
    source = repository_root / recording.poses_member("landmarks2D")
    raw = json.loads(source.read_text(encoding="utf-8"))
    start = recording.window_start_raw_frame
    end = recording.window_end_raw_frame_exclusive
    count = end - start
    view_keys: set[str] = set()
    for frame in range(start, end):
        entry = raw.get(str(frame))
        if entry:
            view_keys.update(entry)
    arrays = {key: np.full((count, 2, 21, 2), np.nan, dtype=np.float32) for key in view_keys}
    missing = 0
    for row, frame in enumerate(range(start, end)):
        entry = raw.get(str(frame))
        if entry is None:
            missing += 1
            continue
        for key, hands in entry.items():
            for hand in ("0", "1"):
                joints = hands.get(hand)
                if joints is not None:
                    arrays[key][row, int(hand)] = np.asarray(joints, dtype=np.float32)
    del raw
    target = repository_root / recording.shipped_2d_window
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        frames=np.arange(start, end, dtype=np.int64),
        **{key.replace(":", "_"): array for key, array in arrays.items()},
    )
    data = target.read_bytes()
    return Landmarks2DWindow(
        local_path=target.resolve().relative_to(repository_root).as_posix(),
        sha256=_sha256_bytes(data),
        size_bytes=len(data),
        pose_frame_range=(start, end),
        frames=count,
        view_keys=tuple(sorted(key.replace(":", "_") for key in view_keys)),
        missing_pose_frames=missing,
    )


def acquisition_path(recording: Assembly101Recording) -> Path:
    return recording.raw_root / "poses_annotations_acquisition.json"


def acquire(
    recording: Assembly101Recording,
    *,
    repository_root: Path,
    skip_fine_grained: bool = False,
    skip_poses: bool = False,
) -> PosesAcquisition:
    repository_root = repository_root.resolve()
    transferred = 0
    print(f"{recording.label}: coarse labels and lookup tables", flush=True)
    coarse = fetch_small_file(
        f"annotations/coarse-annotations/coarse_labels/assembly_{recording.recording_id}.txt",
        repository_root / recording.coarse_labels_path,
        repository_root,
    )
    transferred += coarse.size_bytes
    tables: list[FetchedFile] = []
    for hf_path in SMALL_TABLES:
        target = repository_root / recording.raw_root / hf_path
        tables.append(fetch_small_file(hf_path, target, repository_root))
        transferred += tables[-1].size_bytes
    splits: list[StreamedSplit] = []
    if not skip_fine_grained:
        for split in FINE_GRAINED_SPLITS:
            print(f"{recording.label}: streaming fine-grained {split}.csv", flush=True)
            splits.append(stream_fine_grained_split(split, recording, repository_root))
            transferred += splits[-1].streamed_bytes
            print(
                f"  {split}: {splits[-1].total_rows:,} rows, {splits[-1].rows_for_recording:,} "
                f"kept, etag match {splits[-1].etag_matches_sha256}, {splits[-1].seconds:.0f} s",
                flush=True,
            )
        summary = {s.split: json.loads(s.model_dump_json()) for s in splits}
        (
            repository_root / recording.fine_grained_root / "streaming_filter_summary.json"
        ).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with_rows = [s.split for s in splits if s.rows_for_recording > 0]
    members: tuple[ExtractedMember, ...] = ()
    archive_size = 0
    archive_etag = ""
    member_count = 0
    window: Landmarks2DWindow | None = None
    if not skip_poses:
        print(f"{recording.label}: AssemblyPoses.zip members by HTTP range", flush=True)
        members, archive_size, archive_etag, member_count, member_bytes = extract_pose_members(
            recording, repository_root
        )
        transferred += member_bytes
        (
            repository_root / recording.raw_root / "AssemblyPoses_selective/members_manifest.json"
        ).write_text(
            json.dumps(
                {
                    "repo": DATASET_REPO,
                    "revision": DATASET_REVISION,
                    "archive": POSES_ARCHIVE,
                    "archive_size_bytes": archive_size,
                    "archive_etag": archive_etag,
                    "member_count_total": member_count,
                    "recording": recording.recording_id,
                    "members": [json.loads(m.model_dump_json()) for m in members],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"{recording.label}: writing the 2D landmark window", flush=True)
        window = write_landmarks2d_window(recording, repository_root)
    record = PosesAcquisition(
        manifest_kind="assembly101_poses_annotations_acquisition",
        recording_id=recording.recording_id,
        hf_dataset=DATASET_REPO,
        hf_revision=DATASET_REVISION,
        generated_at_utc=datetime.now(UTC).isoformat(),
        archive=POSES_ARCHIVE,
        archive_size_bytes=archive_size,
        archive_etag=archive_etag,
        archive_member_count=member_count,
        members=members,
        coarse_labels=coarse,
        small_tables=tuple(tables),
        fine_grained=tuple(splits),
        fine_grained_split_with_rows=with_rows[0] if len(with_rows) == 1 else None,
        landmarks2d_window=window,
        bytes_transferred=transferred,
        license=ASSEMBLY101_LICENSE,
        citation=ASSEMBLY101_CITATION,
        claim_boundaries=CLAIM_BOUNDARIES,
    )
    target = repository_root / acquisition_path(recording)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", required=True, help="registry label or recording id")
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--skip-fine-grained", action="store_true")
    parser.add_argument("--skip-poses", action="store_true")
    args = parser.parse_args()
    recording = get_recording(args.recording, args.repository_root)
    record = acquire(
        recording,
        repository_root=args.repository_root,
        skip_fine_grained=args.skip_fine_grained,
        skip_poses=args.skip_poses,
    )
    print(
        f"{recording.label}: {record.bytes_transferred:,} B transferred; "
        f"{len(record.members)} pose members; fine-grained split with rows: "
        f"{record.fine_grained_split_with_rows}; -> {acquisition_path(recording)}"
    )


if __name__ == "__main__":
    main()
