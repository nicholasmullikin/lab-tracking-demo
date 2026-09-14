#!/usr/bin/env python
"""Rebuild a run's Rerun recording from its stored manifest, without re-running inference.

The manifest and mask artifacts are the full record of what a run produced, so an
exporter change can be applied to completed runs without spending GPU time on them.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from battle.exporter import export_run
from battle.schemas import RunManifest


def _video_dimensions(video_path: Path) -> tuple[int, int]:
    """Read the dimensions from the recording's own video rather than trusting a config."""
    probed = subprocess.run(
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
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    width, height = probed.split("x")
    return int(width), int(height)


def reexport(run_directory: Path) -> Path:
    manifest = RunManifest.model_validate_json((run_directory / "manifest.json").read_text())
    videos = sorted(run_directory.glob("input_*f.mp4"))
    if len(videos) != 1:
        raise ValueError(f"expected exactly one input video in {run_directory}, found {videos}")
    return export_run(
        manifest,
        run_directory / "smoke.rrd",
        video_path=videos[0],
        video_dimensions=_video_dimensions(videos[0]),
        mask_artifact_root=run_directory,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    for run_directory in args.runs:
        print(f"Rewrote {reexport(run_directory)}")


if __name__ == "__main__":
    main()
