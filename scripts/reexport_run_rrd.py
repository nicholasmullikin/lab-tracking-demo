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


def _rerun_output_path(manifest: RunManifest, run_directory: Path) -> Path:
    """Use the profile's recorded RRD name, retaining smoke.rrd for older manifests."""
    metadata_profiles = (
        manifest.smoke,
        manifest.g3_candidate,
        manifest.e4_candidate,
        manifest.full_ego_manual_seed,
        manifest.four_part_pilot,
        manifest.four_part_full,
        manifest.four_part_focused,
        manifest.mediapipe_hands,
    )
    artifact_uris = [
        metadata.rerun_artifact_uri
        for metadata in metadata_profiles
        if metadata is not None and metadata.rerun_artifact_uri is not None
    ]
    if len(artifact_uris) > 1:
        raise ValueError(f"manifest declares multiple Rerun artifacts: {artifact_uris}")
    return run_directory / (Path(artifact_uris[0]).name if artifact_uris else "smoke.rrd")


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
    videos = sorted({*run_directory.glob("input_*f.mp4"), *run_directory.glob("input.mp4")})
    if len(videos) != 1:
        raise ValueError(f"expected exactly one input video in {run_directory}, found {videos}")
    return export_run(
        manifest,
        _rerun_output_path(manifest, run_directory),
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
