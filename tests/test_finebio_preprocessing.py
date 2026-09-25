"""``battle-finebio-preprocess`` (p1-configs): the manifest contract, the window and clip-config
helpers and the committed window configs in the default tier; the smoke-window build on the
raw videos as `real_data`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import require_artifact, require_executable

from battle import finebio_frames as ff
from battle import finebio_preprocessing as fp
from battle.finebio_cameras import POSES, read_camera_config

ROOT = Path(__file__).resolve().parents[1]
TRIALS = ROOT / "configs/finebio/trials.json"
SMOKE_CLIP = ROOT / "configs/clips/finebio_P03_01_01_1798-2398.json"
SMOKE_WINDOW_CONFIG = ROOT / "configs/finebio/cameras/P03_01_01_1798-2398.json"
SHA = "0" * 64


def _record(view: str, width: int = 1920, height: int = 1080) -> fp.FineBioProxyRecord:
    return fp.FineBioProxyRecord(
        view=view,
        raw=fp.ArtifactRef(uri=f"data/raw/{view}.mp4", sha256=SHA),
        raw_frame_count=5032,
        proxy=fp.ArtifactRef(uri=f"data/derived/{view}.mp4", sha256=SHA),
        width=width,
        height=height,
        r_frame_rate="30000/1001",
        avg_frame_rate="30000/1001",
        counted_frames=600,
        probe_fps=30,
        ffmpeg_args=tuple(ff.proxy_ffmpeg_args(Path("raw.mp4"), Path("out.mp4"), 1798, 600)),
        encode_seconds=1.0,
        frame_index_contract={
            "start_frame": 1798,
            "samples": [{"proxy_frame": 0, "raw_frame": 1798, "difference": {"0": 0.6}}],
            "all_minima_at_zero": True,
            "max_difference_at_zero": 0.6,
            "min_difference_off_zero": 1.2,
        },
        marker_check={"median_corner_rms_px": 1.0, "frames_with_markers": 3}
        if view == "fpv"
        else None,
    )


def _manifest() -> fp.FineBioPreprocessingManifest:
    return fp.FineBioPreprocessingManifest(
        trial="P03_01_01",
        recording_day="221013",
        window_start_frame=1798,
        window_end_frame_exclusive=2398,
        frame_count=600,
        frame_index_offset=1798,
        views={v: _record(v, height=1440 if v == "fpv" else 1080) for v in fp.VIEWS},
        proxy_recipe=fp.proxy_recipe_record(),
        camera_config=fp.ArtifactRef(uri="configs/finebio/cameras/P03_01_01.json", sha256=SHA),
        window_camera_config=fp.ArtifactRef(
            uri="configs/finebio/cameras/P03_01_01_1798-2398.json", sha256=SHA
        ),
        intrinsics_rescale=dict(fp.INTRINSICS_RESCALE),
        pose_length_check={"pose_frames": 5032, "equal": True, "valid_pose_fraction": 0.97},
        window_source={"requested": {"start_frame": 1798, "end_frame_exclusive": 2398}},
        clip_config_uri="configs/clips/finebio_P03_01_01_1798-2398.json",
        checks_passed=True,
        created="2026-09-25T00:00:00+00:00",
        command="test",
    )


def test_manifest_round_trips_and_records_the_recipe_verbatim() -> None:
    manifest = _manifest()

    again = fp.FineBioPreprocessingManifest.model_validate_json(manifest.model_dump_json())

    assert again == manifest and again.manifest_kind == "finebio_preprocessing"
    assert again.fps == "30000/1001" and again.frame_index_offset == again.window_start_frame
    args = list(again.views["T1"].ffmpeg_args)
    assert args[args.index("-vf") + 1] == "select='between(n,1798,2397)',setpts=N/FRAME_RATE/TB"
    assert "fps=" not in args[args.index("-vf") + 1] and "-ss" not in args
    assert again.proxy_recipe["function"] == "battle.finebio_frames.proxy_ffmpeg_args"
    assert again.proxy_recipe["template"][args.index("-fps_mode") + 1] == "passthrough"
    assert again.intrinsics_rescale == {"fixed": 0.5, "fpv": 0.48}
    assert "non-commercial" in again.licence
    with pytest.raises(ValueError):
        fp.FineBioPreprocessingManifest.model_validate(
            {**manifest.model_dump(), "manifest_kind": "assembly101_g2_preprocessing"}
        )


def test_checks_pass_flags_a_shifted_proxy_and_a_bad_marker_fit() -> None:
    good = _record("fpv", height=1440)
    assert fp.checks_pass(good) == (True, [])
    shifted = good.model_copy(
        update={"frame_index_contract": {**good.frame_index_contract, "all_minima_at_zero": False}}
    )
    ok, problems = fp.checks_pass(shifted)
    assert not ok and "minimum is not at offset 0" in problems[0]
    off = good.model_copy(update={"marker_check": {"median_corner_rms_px": 12.0}})
    assert not fp.checks_pass(off)[0]
    wrong_rate = good.model_copy(update={"r_frame_rate": "30/1"})
    assert not fp.checks_pass(wrong_rate)[0]


def test_window_helpers_read_trials_json_and_name_the_outputs() -> None:
    entry = fp.window_from_trials(TRIALS, "P03_03_01")
    assert entry["window"]["start"] == 600 and entry["window"]["end"] == 4200
    assert fp.window_from_trials(TRIALS, "P20_03_01")["room"] == 2
    with pytest.raises(KeyError):
        fp.window_from_trials(TRIALS, "P99_00_00")
    assert fp.proxy_name("P03_03_01", "T1", 600, 4200) == "P03_03_01_T1_600-4200.mp4"
    assert fp.window_camera_config_path("P03_03_01", 600, 4200) == Path(
        "configs/finebio/cameras/P03_03_01_600-4200.json"
    )
    assert fp.clip_config_path("P03_03_01", 600, 4200) == Path(
        "configs/clips/finebio_P03_03_01_600-4200.json"
    )
    assert fp.sample_offsets(600) == [0, 300, 599] and fp.sample_offsets(1) == [0]


def test_clip_config_carries_views_targets_window_and_downstream_commands() -> None:
    manifest = _manifest()
    entry = fp.window_from_trials(TRIALS, "P03_01_01")

    clip = fp.clip_config(
        manifest, trial_entry=entry, trials_ref=fp.ArtifactRef(uri="t.json", sha256=SHA)
    )

    assert clip["config_kind"] == "finebio_clip_config"
    assert clip["views"] == ["T1", "T2", "T3", "T4", "T5", "fpv"] and clip["fpv_view"] == "fpv"
    assert clip["targets"] == list(fp.DEFAULT_TARGETS) and "centrifuge" in clip["containers"]
    assert clip["window"] == {
        "start_frame": 1798,
        "end_frame_exclusive": 2398,
        "frame_count": 600,
        "seconds": [59.99, 80.01],
    }
    assert clip["frame_index_offset"] == 1798 and clip["role"] == "smoke"
    assert clip["view_sizes"]["fpv"] == [1920, 1440]
    assert clip["proxies"]["T5"] == "data/derived/T5.mp4"
    assert clip["window_camera_config"].endswith("P03_01_01_1798-2398.json")
    assert clip["downstream"]["rig"].startswith("uv run battle-finebio-rig --config ")
    assert "--frames 1798-2397" in clip["downstream"]["rig"]
    assert json.loads(json.dumps(clip)) == clip


def test_committed_smoke_window_config_and_clip_config_agree() -> None:
    clip = json.loads(SMOKE_CLIP.read_text())
    window = read_camera_config(SMOKE_WINDOW_CONFIG)
    trial = read_camera_config(ROOT / "configs/finebio/cameras/P03_01_01.json")

    assert window.frame_index_offset == 1798 == clip["frame_index_offset"]
    assert window.fixed == trial.fixed and window.fpv == trial.fpv
    assert window.provenance["window"]["end_frame_exclusive"] == 2398
    assert clip["window_camera_config"] == "configs/finebio/cameras/P03_01_01_1798-2398.json"
    assert clip["camera_config"] == "configs/finebio/cameras/P03_01_01.json"
    assert set(clip["proxies"]) == set(fp.VIEWS)
    assert all(len(sha) == 64 for sha in clip["proxy_sha256"].values())


@pytest.mark.real_data
def test_smoke_window_build_passes_the_frame_index_contract_in_every_view(tmp_path: Path) -> None:
    require_executable("ffmpeg", "the FineBio window proxies")
    require_executable("ffprobe", "the FineBio window proxies")
    for view in fp.VIEWS:
        require_artifact(ff.video_path("P03_01_01", view))
    require_artifact(POSES / "first_person_camera_poses/P03_01_01.npz")
    require_artifact(POSES / "third_person_camera_poses/221013/params/marker_points.npy")

    manifest, clip, problems = fp.run(
        "P03_01_01",
        1798,
        1858,
        output=tmp_path / "window",
        camera_config_path=ROOT / "configs/finebio/cameras/P03_01_01.json",
        repository_root=ROOT,
        trials_path=TRIALS,
        jobs=6,
        skip_existing=False,
        command="test",
        write_configs=False,
    )

    assert problems == [] and manifest.checks_passed
    assert manifest.frame_count == 60 and manifest.frame_index_offset == 1798
    assert set(manifest.views) == set(fp.VIEWS)
    for view, record in manifest.views.items():
        assert record.counted_frames == 60 and record.r_frame_rate == "30000/1001", view
        assert (record.width, record.height) == ((1920, 1440) if view == "fpv" else (1920, 1080))
        contract = record.frame_index_contract
        assert [s["proxy_frame"] for s in contract["samples"]] == [0, 30, 59]
        assert contract["all_minima_at_zero"] and contract["max_difference_at_zero"] < 3.0
        assert (tmp_path / "window" / record.proxy.uri.rsplit("/", 1)[-1]).exists()
    assert manifest.views["fpv"].marker_check["median_corner_rms_px"] < ff.PROXY_MARKER_MAX_RMS_PX
    assert manifest.pose_length_check["equal"]
    assert manifest.clip_config_uri is None and clip["frame_index_offset"] == 1798
    window = read_camera_config(tmp_path / "window" / "P03_01_01_1798-1858.json")
    assert window.frame_index_offset == 1798
    assert (tmp_path / "window" / "manifest.json").exists()
    assert manifest.window_source["entry_window"]["start"] == 1798
