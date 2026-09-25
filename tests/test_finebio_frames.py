"""The FineBio frame-index contract: proxy frame k == raw frame start+k, pose length == raw
frame count, markers still fit on the proxy. The proxy recipe itself is checked on the
synthetic clip in the default tier; the contract needs the raw videos (`real_data`)."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import require_artifact, require_executable

from battle import finebio_frames as ff
from battle.finebio_cameras import POSES

TRIAL = "P03_01_01"
DAY = "221013"
START = 1798
COUNT = 30
SAMPLES = [0, 14, 29]


def test_proxy_ffmpeg_args_is_the_canonical_recipe(tmp_path: Path) -> None:
    args = ff.proxy_ffmpeg_args(Path("raw.mp4"), tmp_path / "p.mp4", 1798, 30)

    assert args[0] == "ffmpeg" and args[-1] == str(tmp_path / "p.mp4")
    filters = args[args.index("-vf") + 1]
    assert filters == "select='between(n,1798,1827)',setpts=N/FRAME_RATE/TB"
    assert "fps=" not in filters and "scale=" not in filters and "-ss" not in args
    assert args[args.index("-fps_mode") + 1] == "passthrough"
    assert args[args.index("-frames:v") + 1] == "30"
    assert args[args.index("-c:v") + 1] == "libx264"
    assert args[args.index("-crf") + 1] == "18"
    assert args[args.index("-pix_fmt") + 1] == "yuv420p"
    assert args[args.index("-movflags") + 1] == "+faststart"
    assert "-an" in args and "-r" not in args
    with pytest.raises(ValueError):
        ff.proxy_ffmpeg_args(Path("raw.mp4"), tmp_path / "p.mp4", -1, 30)
    with pytest.raises(ValueError):
        ff.proxy_ffmpeg_args(Path("raw.mp4"), tmp_path / "p.mp4", 0, 0)


def test_build_proxy_trims_exact_frames_on_the_synthetic_clip(
    tmp_path: Path, synthetic_video: Path
) -> None:
    require_executable("ffprobe", "counting proxy frames")

    info = ff.build_proxy(synthetic_video, tmp_path / "proxy.mp4", 1, 2)

    assert info["counted_frames"] == 2 and info["start_frame"] == 1
    assert (info["width"], info["height"]) == (16, 8)
    assert ff.raw_frame_count(synthetic_video) == 3
    assert ff.counted_frames(tmp_path / "proxy.mp4") == 2
    frames = ff.read_frames_sequential(tmp_path / "proxy.mp4", [0, 1])
    assert set(frames) == {0, 1} and frames[0].shape == (8, 16, 3)
    with pytest.raises(RuntimeError, match="expected 5"):
        ff.build_proxy(synthetic_video, tmp_path / "short.mp4", 1, 5)


@pytest.fixture(scope="module")
def fpv_proxy(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, dict]:
    require_executable("ffmpeg", "the FineBio window proxy")
    require_executable("ffprobe", "the FineBio window proxy")
    raw = require_artifact(ff.video_path(TRIAL, "fpv"))
    require_artifact(POSES / f"first_person_camera_poses/{TRIAL}.npz")
    out = tmp_path_factory.mktemp("finebio-proxy") / f"{TRIAL}_fpv_{START}_{COUNT}.mp4"
    info = ff.build_proxy(raw, out, START, COUNT)
    return raw, out, info


@pytest.mark.real_data
def test_proxy_keeps_the_native_resolution_and_rate(fpv_proxy) -> None:
    _, proxy, info = fpv_proxy

    assert info["counted_frames"] == COUNT
    assert (info["width"], info["height"]) == (1920, 1440)
    assert info["r_frame_rate"] == ff.NATIVE_RATE == "30000/1001"
    assert info["avg_frame_rate"] == "30000/1001"
    assert ff.raw_frame_count(proxy) == COUNT


@pytest.mark.real_data
def test_proxy_frame_k_is_raw_frame_start_plus_k(fpv_proxy) -> None:
    raw, proxy, _ = fpv_proxy

    report = ff.frame_index_contract(raw, proxy, START, SAMPLES)

    assert [s["proxy_frame"] for s in report["samples"]] == SAMPLES
    assert [s["raw_frame"] for s in report["samples"]] == [START + k for k in SAMPLES]
    assert report["all_minima_at_zero"], report
    # crf-18 re-encode noise on 8-bit grey; a one-frame offset is at least 1.3x larger on
    # every sample and several times larger where the head moves.
    assert report["max_difference_at_zero"] < 3.0, report
    for sample in report["samples"]:
        at_zero = sample["difference"]["0"]
        for offset in ("-1", "1"):
            assert sample["difference"][offset] > 1.3 * at_zero, sample


@pytest.mark.real_data
@pytest.mark.parametrize(("trial", "expected"), [("P03_01_01", 5032), ("P03_03_01", 8492)], ids=str)
def test_pose_length_equals_the_raw_frame_count(trial: str, expected: int) -> None:
    require_executable("ffprobe", "the raw frame count")
    require_artifact(POSES / f"first_person_camera_poses/{trial}.npz")
    for view in ("fpv", "T1", "T5"):
        require_artifact(ff.video_path(trial, view))

    report = ff.pose_length_check(trial, ("fpv", "T1", "T5"))

    assert report["pose_frames"] == expected
    assert report["video_frames"] == {"fpv": expected, "T1": expected, "T5": expected}
    assert report["equal"]
    assert report["valid_pose_fraction"] > 0.97


@pytest.mark.real_data
def test_markers_on_the_proxy_fit_the_shipped_pose_of_raw_frame_start_plus_k(fpv_proxy) -> None:
    _, proxy, _ = fpv_proxy
    require_artifact(POSES / f"third_person_camera_poses/{DAY}/params/marker_points.npy")

    report = ff.proxy_marker_check(TRIAL, DAY, proxy, START, SAMPLES)

    assert report["frames_with_markers"] >= 2, report
    assert report["median_corner_rms_px"] < ff.PROXY_MARKER_MAX_RMS_PX, report
    assert report["max_corner_rms_px"] < ff.PROXY_MARKER_MAX_RMS_PX, report
    for sample in report["samples"]:
        assert sample["raw_frame"] == START + sample["proxy_frame"]
