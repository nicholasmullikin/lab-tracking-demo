"""Shared test helpers: artifact guards, a fast local server, and cached media.

The suite is tiered so the default run needs nothing but the repository: tests that
read ignored `data/`, `runs/`, or `models/` trees carry the `real_data` marker and
skip with a named path when that tree is absent.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

# `serve_forever` wakes up on this interval to notice `shutdown`, so the default
# half-second poll is charged to every server-backed test at teardown.
SERVER_POLL_INTERVAL_SECONDS = 0.01


def require_artifact(path: Path | str) -> Path:
    """Skip the calling test when an ignored, generated artifact is not present."""
    resolved = Path(path)
    if not resolved.exists():
        pytest.skip(f"{resolved} is not built in this checkout")
    return resolved


def require_executable(name: str, reason: str) -> str:
    """Skip the calling test when an external tool the check needs is unavailable."""
    executable = shutil.which(name)
    if executable is None:
        pytest.skip(f"{name} is required for {reason}")
    return executable


def serve(workspace: Any) -> tuple[ThreadingHTTPServer, str]:
    """Start a calibration workspace server that tears down without a poll delay."""
    from battle.muggled_calibration_web import make_handler

    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(workspace))
    threading.Thread(
        target=server.serve_forever,
        args=(SERVER_POLL_INTERVAL_SECONDS,),
        daemon=True,
    ).start()
    return server, f"http://127.0.0.1:{server.server_port}"


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Encode one tiny fixture clip per session instead of once per test."""
    require_executable("ffmpeg", "the synthetic fixture clip")
    path = tmp_path_factory.mktemp("synthetic-video") / "input.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=16x8:r=30",
            "-frames:v",
            "3",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    return path
