"""Frame counts and dimensions for local videos, remembered beside each file.

Counting frames exactly means decoding the whole stream: `ffprobe -count_frames` takes
seconds on a one-minute proxy, and the review builders ask for the same numbers on every
run. The answer is cached in a sidecar keyed by the video's size and modification time.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

VideoInfo = tuple[int, int, tuple[int, int]]


def _sidecar_path(video_path: Path) -> Path:
    return video_path.with_name(f"{video_path.name}.probe.json")


def _probe(video_path: Path) -> VideoInfo:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(completed.stdout)["streams"][0]
    numerator, denominator = stream["avg_frame_rate"].split("/")
    return (
        int(stream["nb_read_frames"]),
        round(int(numerator) / int(denominator)),
        (int(stream["width"]), int(stream["height"])),
    )


def video_info(video_path: Path, *, verify: bool = False) -> VideoInfo:
    """Return `(frame_count, fps, (width, height))`, decoding only when it must."""
    resolved = video_path.resolve()
    status = resolved.stat()
    sidecar = _sidecar_path(resolved)
    if not verify and sidecar.is_file():
        try:
            stored = json.loads(sidecar.read_text(encoding="utf-8"))
            if stored["size_bytes"] == status.st_size and stored["mtime_ns"] == status.st_mtime_ns:
                return (
                    int(stored["frame_count"]),
                    int(stored["fps"]),
                    (int(stored["width"]), int(stored["height"])),
                )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
    frame_count, fps, (width, height) = _probe(resolved)
    try:
        sidecar.write_text(
            json.dumps(
                {
                    "size_bytes": status.st_size,
                    "mtime_ns": status.st_mtime_ns,
                    "frame_count": frame_count,
                    "fps": fps,
                    "width": width,
                    "height": height,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    except OSError:
        pass
    return frame_count, fps, (width, height)


def video_frame_count(video_path: Path, *, verify: bool = False) -> int:
    return video_info(video_path, verify=verify)[0]
