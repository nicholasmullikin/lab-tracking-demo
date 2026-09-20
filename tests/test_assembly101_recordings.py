"""Recording registry: config resolution and clock-rule lookup keyed by recording (Track C1)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from conftest import require_artifact

from battle import assembly101_camera_fit as fit
from battle import assembly101_clock_offset as clock
from battle import assembly101_contact_sheet as sheet
from battle import assembly101_fetch_poses as poses
from battle import assembly101_fetch_view as fetch
from battle import assembly101_recordings as recordings
from battle.assembly101_pose_schemas import Assembly101CameraModel

REC2_LABEL = "nusar_9061"
REC2_ID = "nusar-2021_action_both_9061-c02a_9061_user_id_2021-02-09_141537"


@pytest.fixture(scope="module")
def rec2() -> recordings.Assembly101Recording:
    return recordings.get_recording(REC2_LABEL, Path.cwd())


# --- registry ---------------------------------------------------------------------------


def test_registry_resolves_by_label_and_id_and_defaults_to_recording_1(rec2) -> None:
    registry = recordings.load_registry(Path.cwd())
    assert registry.default == recordings.RECORDING_1
    assert registry.get("nusar_9033") == recordings.RECORDING_1
    assert registry.get(recordings.RECORDING_1.recording_id) == recordings.RECORDING_1
    assert registry.get(REC2_ID) == rec2 and rec2.label == REC2_LABEL
    assert recordings.get_recording(None) is recordings.RECORDING_1
    assert recordings.get_recording("nusar_9033") is recordings.RECORDING_1
    with pytest.raises(KeyError, match="no Assembly101 recording"):
        registry.get("nusar_0000")
    assert rec2.toy_id == recordings.RECORDING_1.toy_id == "c02a"
    assert rec2.subject_id != recordings.RECORDING_1.subject_id


def test_recording_1_record_reproduces_the_historical_constants() -> None:
    one = recordings.RECORDING_1
    assert one.window_start_raw_frame == 17640 and one.window_raw_frame_count == 5562
    assert one.window_proxy_frame_count == 2781 and one.annotation_start_frame == 8820
    assert one.core_proxy_frame_range == (0, 1800)
    assert one.shipped_2d_window.name == "assembly101_landmarks2D_60fps_frames_17640_23202.npz"
    assert one.fine_grained_csv().name.startswith("train__")
    assert Path(one.clock_rules_path) == clock.CLOCK_RULES_CONFIG
    assert Path(one.camera_config_root) == fit.CONFIG_ROOT


def test_recording_2_window_is_80_seconds_with_a_60_second_core(rec2) -> None:
    assert (rec2.window_start_seconds, rec2.window_duration_seconds) == (374.0, 80.0)
    assert (rec2.window_start_raw_frame, rec2.window_end_raw_frame_exclusive) == (22440, 27240)
    assert rec2.window_proxy_frame_count == 2400
    assert rec2.core_proxy_frame_range == (300, 2100)
    assert rec2.primary_static_view == "C10379"
    assert rec2.ego_views == ("HMC_21110305", "HMC_21179183")
    assert set(rec2.ego_views) <= set(rec2.ego_views_on_hub)


def test_registry_rejects_a_core_span_outside_the_window() -> None:
    data = recordings.RECORDING_1.model_dump()
    data["core_start_seconds"] = 300.0
    data["core_duration_seconds"] = 100.0
    with pytest.raises(ValueError, match="core span"):
        recordings.Assembly101Recording.model_validate(data)


# --- paths keyed by recording -----------------------------------------------------------


def test_fetch_paths_follow_the_recording_window(rec2) -> None:
    assert fetch.raw60_path("C10379").name == "C10379_rgb_294.000-386.700_raw60.mp4"
    assert fetch.raw60_path("C10379", recording=rec2).name == "C10379_rgb_374.000-454.000_raw60.mp4"
    proxy = fetch.proxy_path("HMC_21179183", recording=rec2)
    assert proxy.name == "HMC_21179183_mono10bit_374.000-454.000_954x720_30fps.mp4"
    assert proxy.parent == rec2.derived_root and REC2_ID in proxy.as_posix()
    assert (
        fetch.per_view_record_path("C10095", Path("/r"), rec2)
        == Path("/r") / rec2.acquisition_records_root / "C10095.json"
    )
    assert fetch.view_id_for("HMC_21179183") == "ego-hmc21179183"
    assert fetch.view_id_for("C10095") == "static-c10095"
    assert rec2.ego_clip_config("HMC_21110305").name == (
        "assembly101_nusar_9061_four_part_reassembly_focused_ego_hmc_21110305_g2.json"
    )


def test_camera_estimate_paths_are_per_recording(rec2) -> None:
    assert fit.camera_estimate_path("C10379") == Path(
        "configs/assembly101/c10379_camera_estimate.json"
    )
    assert fit.camera_estimate_path("C10379", recording=rec2) == Path(
        "configs/assembly101/nusar_9061/c10379_camera_estimate.json"
    )
    assert fit.camera_estimate_path("C10379", Path("x")) == Path("x/c10379_camera_estimate.json")


# --- clock rules keyed by recording -----------------------------------------------------


def test_clock_rules_load_by_recording_and_refuse_the_wrong_recording(rec2, tmp_path: Path) -> None:
    one = clock.load_clock_rules_for(recordings.RECORDING_1, Path.cwd())
    assert one.recording_id == recordings.RECORDING_1.recording_id
    assert one.rule("C10379").pose_offset_frames == 9
    two = clock.load_clock_rules_for(rec2, Path.cwd())
    assert two.recording_id == REC2_ID and two.window_start_pose_frame == 22440
    assert set(two.views) == set(rec2.all_views)
    for view in rec2.all_views:
        rule = two.rule(view)
        assert rule.proxy_start_raw_frame == 22440
        assert rule.pose_frame(0) == 22440 + rule.pose_offset_frames
    assert two.rule("C10379").pose_offset_frames != one.rule("C10379").pose_offset_frames
    # A recording-1 file offered as recording 2's rules is refused.
    swapped = rec2.model_copy(update={"clock_rules_path": recordings.RECORDING_1.clock_rules_path})
    with pytest.raises(ValueError, match="describes"):
        clock.load_clock_rules_for(swapped, Path.cwd())
    # Mixing scans of two recordings into one rule file is refused too.
    scans = tmp_path / "scans"
    scans.mkdir()
    shutil.copy(Path(recordings.RECORDING_1.clock_scan_root) / "C10095.json", scans / "C10095.json")
    with pytest.raises(ValueError, match="refusing to mix"):
        clock.write_clock_rules(
            tmp_path, scan_root=Path("scans"), output=Path("out.json"), recording=rec2
        )


def test_chunk_plan_keeps_the_sep18_chunks_and_splits_shorter_trims_in_thirds() -> None:
    assert clock.chunk_plan(5562) == ((0, 1800, 3600), 1800)
    assert clock.chunk_plan(4800) == ((0, 1600, 3200), 1600)
    with pytest.raises(ValueError):
        clock.chunk_plan(600)


def _curve(values, metric, chunk=0):
    return clock.summarize_curve(metric, chunk, chunk * 1600, values, [10] * len(values))


def _peaked(centre: float):
    import numpy as np

    return [30.0 + 10.0 * float(np.exp(-((o - centre) ** 2) / 2.0)) for o in clock.OFFSETS]


def test_decide_offset_drops_a_lone_disagreeing_metric_and_widens_the_uncertainty() -> None:
    curves = [
        _curve(_peaked(5.4), "skin_hit"),
        _curve(_peaked(1.5), "gradient"),
        _curve(_peaked(6.2), "motion"),
    ]
    decision = clock.decide_offset(curves)
    assert not decision.ambiguous and decision.chosen == 6
    assert decision.dropped_metric == "gradient" and decision.uncertainty >= 2
    assert "majority fallback" in decision.evidence
    # Two metrics only: no majority, still ambiguous.
    assert clock.decide_offset(curves[:2]).ambiguous
    # Three metrics spread evenly: no clear odd one out, still ambiguous.
    spread = [
        _curve(_peaked(0.0), "skin_hit"),
        _curve(_peaked(2.5), "gradient"),
        _curve(_peaked(5.0), "motion"),
    ]
    assert clock.decide_offset(spread).ambiguous
    # Agreement within the tolerance never triggers the fallback.
    agreeing = [_curve(_peaked(6.0), m) for m in ("skin_hit", "gradient", "motion")]
    tight = clock.decide_offset(agreeing)
    assert tight.dropped_metric is None and tight.uncertainty == 1


# --- tracked recording-2 assets ---------------------------------------------------------


def test_tracked_recording_2_camera_estimates_and_clip_configs_are_complete(rec2) -> None:
    from battle.schemas import G2PreprocessingManifest

    for view in rec2.all_views:
        path = fit.camera_estimate_path(view, recording=rec2)
        assert path.is_file(), path
        model = Assembly101CameraModel.model_validate_json(path.read_text())
        assert model.view_key == clock.view_key(view)
        if view.startswith("HMC_"):
            assert model.is_ego and model.distortion_model == "rational"
        else:
            assert model.distortion_model == "brown" and model.camera_to_world is not None
            assert model.shipped_extrinsics_rms_pixels is not None
            assert model.shipped_extrinsics_rms_pixels < 0.01
    static = G2PreprocessingManifest.model_validate_json(
        Path(rec2.all_static_clip_config).read_text()
    )
    assert static.clip.views == tuple(fetch.view_id_for(v) for v in rec2.static_views)
    assert static.source_interval.start_seconds == 374.0
    assert static.proxy_frame_range.end_frame_exclusive == 2400
    assert all(p.frame_count == 2400 and p.fps == 30 for p in static.proxies)
    assert all(p.raw_source.raw_uri.startswith("hf://datasets/") for p in static.proxies)
    assert all(REC2_ID in p.proxy_uri for p in static.proxies)
    for view in rec2.ego_views:
        ego = G2PreprocessingManifest.model_validate_json(rec2.ego_clip_config(view).read_text())
        assert ego.clip.views == (fetch.view_id_for(view),)
        assert ego.proxies[0].dimensions.width == 954
    selection = json.loads(
        Path("configs/assembly101/nusar_9061/recording_selection.json").read_text()
    )
    assert selection["chosen"]["recording_id"] == REC2_ID
    assert len(selection["rejected"]) >= 2


# --- small helpers ----------------------------------------------------------------------


def test_contact_sheet_frames_and_coarse_lookup() -> None:
    assert sheet.contact_sheet_frames(2400) == (0, 600, 1200, 1800, 2399)
    frames = sheet.contact_sheet_frames(2781)
    assert frames[0] == 0 and frames[-1] == 2780 and len(frames) == 5
    assert all(b > a for a, b in zip(frames[:-1], frames[1:], strict=True))
    labels = ((100, 200, "attach interior"), (200, 260, "screw chassis"))
    assert sheet.coarse_action_at(labels, 150) == "attach interior"
    assert sheet.coarse_action_at(labels, 200) == "screw chassis"
    assert sheet.coarse_action_at(labels, 260) is None


def test_etag_check_handles_lfs_sha256_and_git_blob_ids() -> None:
    data = b"hello\n"
    sha = "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"
    blob = "ce013625030ba8dba906f756967f9e9ca394464a"
    assert poses._etag_matches(sha, sha) is True
    assert poses._etag_matches('"' + sha + '"', sha) is True
    assert poses._etag_matches(blob, sha, data) is True
    assert poses._etag_matches(blob, sha) is None
    assert poses._etag_matches(None, sha) is None
    assert poses._etag_matches("0" * 64, sha) is False


# --- real data --------------------------------------------------------------------------


@pytest.mark.real_data
def test_recording_2_windows_scans_and_rig_exist(rec2) -> None:
    from battle.multiview_geometry import CameraRig

    for view in rec2.all_views:
        record = fetch.load_record(view, require_artifact(Path.cwd()), rec2)
        assert record.recording_id == REC2_ID and record.source_kind == "hf_range"
        assert record.raw60.frame_count == 4800 and record.proxy.frame_count == 2400
        assert require_artifact(record.proxy.uri).stat().st_size == record.proxy.size_bytes
        scan = clock.load_scan(require_artifact(clock.scan_path(view, Path(rec2.clock_scan_root))))
        assert scan.recording_id == REC2_ID and scan.frames_scanned == 4800
    rules = clock.load_clock_rules_for(rec2, Path.cwd())
    rig = CameraRig.load(require_artifact(Path.cwd()), recording=rec2)
    assert set(rig.views) == set(rec2.all_views)
    assert rig.pose_frame("C10379", 0) == rules.rule("C10379").pose_frame(0)
    report = json.loads(require_artifact(Path(rec2.rig_check_root) / "report.json").read_text())
    assert report["recording_id"] == REC2_ID
    assert all(
        p["rms_pixels"] < 0.01 for p in report["projection"] if not p["view"].startswith("HMC")
    )
    manifest = json.loads(require_artifact(Path(rec2.reference_root) / "manifest.json").read_text())
    assert manifest["recording_id"] == REC2_ID and manifest["frame_count"] == 2400
