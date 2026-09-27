"""Probe official Assembly101 AssemblyPoses.zip via HTTP Range without full download."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import get_hf_file_metadata, get_token, hf_hub_url

from .zip_range import (
    RangeReader,
    ZipRangeError,
    filter_members,
    list_remote_zip_members,
    read_remote_zip_member,
)

DEFAULT_REPO = "cvml-nus/assembly101"
DEFAULT_REVISION = "bfc15ea5e3f0bc8f8c232af6c1b45aa137a9d967"
DEFAULT_ARCHIVE = "AssemblyPoses.zip"
DEFAULT_RECORDING = "nusar-2021_action_both_9033-c02a_9033_user_id_2021-02-04_140532"
DEFAULT_OUTPUT = Path("docs/archive/athena_hf_calibration_probe.json")
CALIBRATION_PATTERNS = (
    "camera_extrinsics",
    "camera_position",
    "intrinsic",
    "calib",
    "xf_transf",
    "timestamp",
)


def _auth_headers() -> dict[str, str]:
    token = get_token()
    return {"Authorization": f"Bearer {token}"} if token else {}


def resolve_archive_url(repo: str, revision: str, archive: str) -> tuple[str, int, str]:
    url = hf_hub_url(repo, archive, repo_type="dataset", revision=revision)
    metadata = get_hf_file_metadata(url, token=get_token())
    return url, metadata.size, metadata.etag


def probe_archive(
    *,
    repo: str,
    revision: str,
    archive: str,
    recording: str,
    output_path: Path,
    extract: bool,
    extract_dir: Path,
) -> dict[str, object]:
    url, size, etag = resolve_archive_url(repo, revision, archive)
    reader = RangeReader(url, size=size, headers=_auth_headers())
    members = list_remote_zip_members(reader)
    recording_members = [member for member in members if recording in member.filename]
    calibration_members = filter_members(members, CALIBRATION_PATTERNS)
    recording_calibration = [
        member for member in recording_members if member in calibration_members
    ]
    payload: dict[str, object] = {
        "probed_at_utc": datetime.now(UTC).isoformat(),
        "dataset_repo": repo,
        "dataset_revision": revision,
        "archive_member": archive,
        "archive_size_bytes": size,
        "archive_etag": etag,
        "archive_url_host": url.split("/")[2],
        "range_requests_supported": True,
        "member_count_total": len(members),
        "member_count_for_recording": len(recording_members),
        "member_count_calibration_like_total": len(calibration_members),
        "member_count_calibration_like_for_recording": len(recording_calibration),
        "recording_id": recording,
        "recording_members": [
            {
                "filename": member.filename,
                "compressed_size": member.compressed_size,
                "uncompressed_size": member.uncompressed_size,
                "compression_method": member.compression_method,
            }
            for member in recording_members
        ],
        "calibration_like_members_for_recording": [
            member.filename for member in recording_calibration
        ],
        "top_level_prefixes": sorted({member.filename.split("/", 1)[0] for member in members}),
    }
    if extract and recording_calibration:
        extract_dir.mkdir(parents=True, exist_ok=True)
        extracted: list[dict[str, object]] = []
        for member in recording_calibration:
            if member.uncompressed_size > 50 * 1024 * 1024:
                extracted.append(
                    {
                        "filename": member.filename,
                        "skipped": True,
                        "reason": "member exceeds 50 MiB selective-extract cap",
                    }
                )
                continue
            raw = read_remote_zip_member(reader, member)
            target = extract_dir / member.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            extracted.append(
                {
                    "filename": member.filename,
                    "bytes": len(raw),
                    "local_path": target.as_posix(),
                }
            )
        payload["extracted_members"] = extracted
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE)
    parser.add_argument("--recording", default=DEFAULT_RECORDING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--extract", action="store_true")
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=Path("data/raw/assembly101/athena_calibration_extract"),
    )
    args = parser.parse_args()
    try:
        payload = probe_archive(
            repo=args.repo,
            revision=args.revision,
            archive=args.archive,
            recording=args.recording,
            output_path=args.output,
            extract=args.extract,
            extract_dir=args.extract_dir,
        )
    except ZipRangeError as error:
        raise SystemExit(f"zip range probe failed: {error}") from error
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
