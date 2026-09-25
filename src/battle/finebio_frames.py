"""FineBio frames: raw video paths, the canonical window proxy, and the frame-index contract.

The contract every FineBio stage relies on (Sep 24, p0-contracts):

* **proxy frame k == raw frame start+k.** `proxy_ffmpeg_args(raw, out, start_frame,
  frame_count)` is the one proxy recipe for this phase: native resolution (fixed 1920x1080,
  fpv 1920x1440), native 30000/1001 rate, **no ``fps=`` filter**, exact-frame trim by
  ``select=between(n,start,end)`` on a full decode with ``setpts=N/FRAME_RATE/TB`` and
  ``-fps_mode passthrough`` (no dropped or duplicated frames), ``-frames:v frame_count``,
  libx264 crf 18, yuv420p, faststart, no audio. Lane B's ``finebio_preprocessing`` builds the
  real proxies with exactly these arguments; the rate and the trim are not negotiable
  because the shipped per-frame fpv pose is indexed by raw frame.
* **pose length == raw frame count.** ``fpv_poses(trial).rets.size`` equals the raw video's
  frame count (`pose_length_check`), so ``poses[start + k]`` is the pose of proxy frame k.
* **markers still fit on the proxy.** The fpv pose for raw frame start+k projects the day's
  markers onto the ArUco corners detected in proxy frame k to under 10 px
  (`proxy_marker_check`).

`frame_index_contract` measures the first point directly: the mean absolute grey difference
between proxy frame k and raw frames start+k-1, start+k, start+k+1. After a crf-18 re-encode
the difference at offset 0 is a couple of grey levels; on moving content a one-frame offset is
several times larger, so the minimum sitting at 0 is the proof.

Raw frames are read with cv2 by seeking (``CAP_PROP_POS_FRAMES``), the same access path the
preflight used for its detections and its fpv-pose check (0.9 px median), so the seek index
and the pose index are known to agree; the proxy is read sequentially from frame 0.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .finebio_cameras import (
    RAW,
    Camera,
    detect_markers,
    fpv_poses,
    intrinsics,
    marker_points,
    match_markers,
)

NATIVE_RATE = "30000/1001"
PROXY_CRF = 18
PROXY_MARKER_MAX_RMS_PX = 10.0


def video_path(trial: str, view: str) -> Path:
    if view == "fpv":
        return RAW / "finebio_videos_fpv_test/finebio_videos" / f"{trial}.mp4"
    return RAW / "finebio_videos_tpv_test/finebio_videos" / f"{trial}_{view}.mp4"


def read_frame(video: Path, index: int) -> np.ndarray:
    """One BGR frame by seek; raises when the frame cannot be read."""
    cap = cv2.VideoCapture(str(video))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, img = cap.read()
    finally:
        cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {index} of {video}")
    return img


def read_frames_sequential(video: Path, indices: list[int]) -> dict[int, np.ndarray]:
    """Frames at `indices` decoded from frame 0 without seeking (for short proxies)."""
    wanted = set(indices)
    out: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(str(video))
    try:
        index = 0
        last = max(wanted) if wanted else -1
        while index <= last:
            ok, img = cap.read()
            if not ok:
                break
            if index in wanted:
                out[index] = img
            index += 1
    finally:
        cap.release()
    missing = wanted - set(out)
    if missing:
        raise RuntimeError(f"could not read frames {sorted(missing)} of {video}")
    return out


def ffprobe_stream(video: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames,r_frame_rate,avg_frame_rate,width,height",
            "-of",
            "json",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)["streams"][0]


def raw_frame_count(video: Path) -> int:
    """Container frame count (``nb_frames``), falling back to a cv2 count when absent."""
    stream = ffprobe_stream(video)
    count = stream.get("nb_frames")
    if count not in (None, "N/A"):
        return int(count)
    cap = cv2.VideoCapture(str(video))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def counted_frames(video: Path) -> int:
    """Exact frame count by decoding the stream (``ffprobe -count_frames``)."""
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "json",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(json.loads(completed.stdout)["streams"][0]["nb_read_frames"])


# --------------------------------------------------------------------------- proxy recipe


def proxy_ffmpeg_args(raw: Path, out: Path, start_frame: int, frame_count: int) -> list[str]:
    """The canonical window proxy: raw frames ``start_frame .. start_frame+frame_count-1``
    become proxy frames ``0 .. frame_count-1`` at the native resolution and rate.

    Exact-frame trim by frame number on a full decode (no ``-ss``, whose seek is by
    timestamp), timestamps regenerated so the survivors are consecutive at the input rate,
    passthrough sync so nothing is dropped or duplicated, and ``-frames:v`` as the stop.
    """
    if start_frame < 0 or frame_count < 1:
        raise ValueError("start_frame must be >= 0 and frame_count >= 1")
    end_frame = start_frame + frame_count - 1
    return [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(raw),
        "-vf",
        f"select='between(n,{start_frame},{end_frame})',setpts=N/FRAME_RATE/TB",
        "-fps_mode",
        "passthrough",
        "-frames:v",
        str(frame_count),
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        str(PROXY_CRF),
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out),
    ]


def build_proxy(raw: Path, out: Path, start_frame: int, frame_count: int) -> dict[str, Any]:
    """Run the canonical recipe and verify the result has exactly `frame_count` frames at the
    native rate; returns the ffprobe stream fields plus the counted frames."""
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(proxy_ffmpeg_args(raw, out, start_frame, frame_count), check=True)
    stream = ffprobe_stream(out)
    counted = counted_frames(out)
    if counted != frame_count:
        raise RuntimeError(f"{out} has {counted} frames, expected {frame_count}")
    return {**stream, "counted_frames": counted, "start_frame": start_frame}


# --------------------------------------------------------------------------- the contract


def _grey(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


def mean_abs_grey_difference(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"frame shapes differ: {a.shape} vs {b.shape}")
    return float(np.abs(_grey(a) - _grey(b)).mean())


def frame_index_contract(
    raw_video: Path,
    proxy_video: Path,
    start_frame: int,
    sample_offsets: list[int],
    *,
    neighbour_offsets: tuple[int, ...] = (-1, 0, 1),
) -> dict[str, Any]:
    """Proxy frame k against raw frames start+k+d for d in `neighbour_offsets`.

    Returns ``{"start_frame", "samples": [{"proxy_frame", "raw_frame", "difference": {d:
    mean abs grey difference}, "argmin_offset"}], "all_minima_at_zero", "max_difference_at_zero",
    "min_difference_off_zero"}``.
    """
    proxy_frames = read_frames_sequential(proxy_video, sample_offsets)
    samples = []
    for k in sample_offsets:
        raw_index = start_frame + k
        differences: dict[int, float] = {}
        for d in neighbour_offsets:
            if raw_index + d < 0:
                continue
            differences[d] = mean_abs_grey_difference(
                proxy_frames[k], read_frame(raw_video, raw_index + d)
            )
        argmin = min(differences, key=differences.get)
        samples.append(
            {
                "proxy_frame": k,
                "raw_frame": raw_index,
                "difference": {str(d): value for d, value in differences.items()},
                "argmin_offset": argmin,
            }
        )
    off_zero = [
        value for sample in samples for d, value in sample["difference"].items() if d != "0"
    ]
    return {
        "start_frame": start_frame,
        "samples": samples,
        "all_minima_at_zero": all(s["argmin_offset"] == 0 for s in samples),
        "max_difference_at_zero": max(s["difference"]["0"] for s in samples),
        "min_difference_off_zero": min(off_zero) if off_zero else None,
    }


def pose_length_check(trial: str, views: tuple[str, ...] = ("fpv",)) -> dict[str, Any]:
    """``fpv_poses(trial).rets.size`` against each listed video's container frame count."""
    rets, _, _ = fpv_poses(trial)
    counts = {view: raw_frame_count(video_path(trial, view)) for view in views}
    return {
        "trial": trial,
        "pose_frames": int(rets.size),
        "valid_pose_fraction": float(rets.mean()),
        "video_frames": counts,
        "equal": all(count == rets.size for count in counts.values()),
    }


def proxy_marker_check(
    trial: str, day: str, proxy_video: Path, start_frame: int, sample_offsets: list[int]
) -> dict[str, Any]:
    """The shipped fpv pose of raw frame start+k projecting the day's markers onto the ArUco
    corners detected in proxy frame k; median corner RMS per sample and overall."""
    rets, rots, trans = fpv_poses(trial)
    K, dist, size = intrinsics("fpv")
    markers = marker_points(day)
    frames = read_frames_sequential(proxy_video, sample_offsets)
    samples = []
    for k in sample_offsets:
        raw_index = start_frame + k
        detected = detect_markers(frames[k])
        if not rets[raw_index] or not detected:
            samples.append(
                {
                    "proxy_frame": k,
                    "raw_frame": raw_index,
                    "pose_valid": bool(rets[raw_index]),
                    "markers": len(detected),
                    "median_corner_rms_px": None,
                }
            )
            continue
        cam = Camera("fpv", K, dist, rots[raw_index], trans[raw_index], size)
        projected = cam.project(markers.reshape(-1, 3)).reshape(-1, 4, 2)
        rms = [m[2] for m in match_markers(detected, projected)]
        samples.append(
            {
                "proxy_frame": k,
                "raw_frame": raw_index,
                "pose_valid": True,
                "markers": len(detected),
                "median_corner_rms_px": float(np.median(rms)),
            }
        )
    values = [s["median_corner_rms_px"] for s in samples if s["median_corner_rms_px"] is not None]
    return {
        "trial": trial,
        "day": day,
        "samples": samples,
        "frames_with_markers": len(values),
        "median_corner_rms_px": float(np.median(values)) if values else None,
        "max_corner_rms_px": float(np.max(values)) if values else None,
    }
