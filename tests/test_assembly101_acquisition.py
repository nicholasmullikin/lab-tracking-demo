"""Focused-window acquisition, per-view clock offsets and per-view camera fits."""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import pytest
from conftest import require_artifact

from battle import assembly101_camera_fit as fit
from battle import assembly101_clock_offset as clock
from battle import assembly101_fetch_view as fetch
from battle.assembly101_pose_schemas import Assembly101CameraModel

# --- fetch -----------------------------------------------------------------------------


def test_view_names_and_output_paths_follow_the_existing_proxy_convention() -> None:
    assert fetch.video_name("C10095") == "C10095_rgb"
    assert fetch.video_name("HMC_21179183") == "HMC_21179183_mono10bit"
    assert fetch.proxy_path("C10095").name == "C10095_rgb_294.000-386.700_1280x720_30fps.mp4"
    assert (
        fetch.proxy_path("HMC_21179183").name
        == "HMC_21179183_mono10bit_294.000-386.700_954x720_30fps.mp4"
    )
    assert fetch.raw60_path("C10404").name == "C10404_rgb_294.000-386.700_raw60.mp4"
    assert fetch.static_view_id("C10404") == "static-c10404"


def test_ffmpeg_command_seeks_once_and_encodes_trim_and_proxy_with_the_standard_filters() -> None:
    command = fetch.ffmpeg_window_command(
        "http://x/source.mp4",
        raw60_output=Path("raw.mp4"),
        proxy_output=Path("proxy.mp4"),
        proxy_scale="1280:720",
    )
    assert command[: command.index("-i")][-2:] == ["-ss", "294.000"]
    graph = command[command.index("-filter_complex") + 1]
    assert graph.startswith("[0:v:0]trim=duration=92.7,setpts=PTS-STARTPTS,split=2[raw][pre]")
    assert "fps=30:round=near,scale=1280:720:flags=lanczos,setsar=1[proxy]" in graph
    assert command.count("libx264") == 2 and command.count("-fps_mode") == 2
    assert command[-1] == "proxy.mp4"
    trim_only = fetch.ffmpeg_window_command(
        "src.mp4", raw60_output=Path("raw.mp4"), proxy_output=None, proxy_scale="1280:720"
    )
    assert trim_only.count("libx264") == 1 and "split" not in " ".join(trim_only)


class _RangeUpstream(BaseHTTPRequestHandler):
    payload = bytes(range(256)) * 40  # 10,240 bytes

    def log_message(self, *_: object) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        header = self.headers.get("Range")
        start, end = 0, len(self.payload) - 1
        if header:
            first, _, last = header.removeprefix("bytes=").partition("-")
            start = int(first)
            if last:
                end = int(last)
        body = self.payload[start : end + 1]
        self.send_response(206 if header else 200)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Range", f"bytes {start}-{end}/{len(self.payload)}")
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self.wfile.write(body)


def test_counting_proxy_forwards_ranges_and_records_every_byte_delivered() -> None:
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _RangeUpstream)
    threading.Thread(target=upstream.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread = threading.Thread(
        target=upstream.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        url = f"http://127.0.0.1:{upstream.server_port}/file.bin"
        with fetch._CountingProxy(url) as proxy:
            request = urllib.request.Request(proxy.url, headers={"Range": "bytes=100-199"})
            with urllib.request.urlopen(request, timeout=10) as response:
                assert response.status == 206
                assert response.read() == _RangeUpstream.payload[100:200]
            with urllib.request.urlopen(proxy.url, timeout=10) as response:
                assert len(response.read()) == len(_RangeUpstream.payload)
            assert proxy.bytes_sent == 100 + len(_RangeUpstream.payload)
            assert [r.range_header for r in proxy.records] == ["bytes=100-199", None]
    finally:
        upstream.shutdown()
        upstream.server_close()


# --- clock offsets ---------------------------------------------------------------------


def _curve(values: list[float], metric: str = "gradient", chunk: int = 0) -> clock.OffsetCurve:
    return clock.summarize_curve(metric, chunk, chunk * 1800, values, [10] * len(values))


def _peaked(centre: float, width: float = 2.0, base: float = 30.0, height: float = 10.0):
    return [base + height * float(np.exp(-((o - centre) ** 2) / width)) for o in clock.OFFSETS]


def test_summarize_curve_finds_a_subframe_peak_and_rejects_edges_and_flat_curves() -> None:
    curve = _curve(_peaked(6.4))
    assert curve.peak_offset == 6 and curve.informative
    assert curve.peak_offset_subframe == pytest.approx(6.4, abs=0.05)
    ramp = _curve([30.0 + 0.5 * i for i in range(len(clock.OFFSETS))])
    assert ramp.peak_offset == clock.OFFSETS[-1] and not ramp.informative
    flat = _curve([30.0] * len(clock.OFFSETS))
    assert not flat.informative and flat.prominence == 0.0


def test_decide_offset_uses_metric_medians_and_marks_disagreement_ambiguous() -> None:
    curves = [
        _curve(_peaked(6.1), "skin_hit", 0),
        _curve(_peaked(5.9), "skin_hit", 1),
        _curve(_peaked(12.0), "skin_hit", 2),  # one odd chunk must not move the answer
        _curve(_peaked(6.3), "gradient", 0),
        _curve(_peaked(6.0), "motion", 0),
    ]
    decision = clock.decide_offset(curves)
    assert not decision.ambiguous and decision.chosen == 6 and decision.uncertainty == 1
    assert "outlier chunks: skin_hit/chunk2" in decision.evidence
    disagreeing = [_curve(_peaked(2.0), "skin_hit", 0), _curve(_peaked(9.0), "gradient", 0)]
    decision = clock.decide_offset(disagreeing)
    assert decision.ambiguous and decision.chosen is None
    assert clock.decide_offset([_curve([30.0] * len(clock.OFFSETS))]).ambiguous


def _synthetic_scan_inputs(tmp_path: Path, true_offset: int, frames: int = 240):
    """Moving textured 'fingertips' whose dataset landmarks are indexed `true_offset` ahead.

    Each joint is drawn as a small bright dot so both the gradient and the frame-difference
    metrics peak exactly when the sampled landmark sits on its dot.
    """
    width, height = 320, 240
    path = tmp_path / "trim.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 60, (width, height))
    rows = frames + 40
    landmarks = np.zeros((rows, 2, 21, 2), dtype=np.float64)
    confidences = np.ones((rows, 2), dtype=np.float64)
    confidences[:, 1] = 0.0
    centres = {}
    for row in range(rows):
        x = 40 + (row * 7) % 220
        y = 60 + (row * 3) % 120
        centres[row] = (x, y)
        landmarks[row, 0, :, 0] = x + np.linspace(-40, 40, 21)
        landmarks[row, 0, :, 1] = y + 9 * np.sin(np.linspace(0, 6, 21))
    for frame in range(frames):
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[:] = (40, 40, 40)
        for joint in range(21):
            x, y = landmarks[frame + true_offset, 0, joint]
            cv2.circle(image, (int(round(x)), int(round(y))), 3, (200, 210, 230), -1)
        writer.write(image)
    writer.release()
    return path, landmarks, confidences


@pytest.mark.parametrize("true_offset", [3, 11])
def test_scan_video_recovers_a_known_offset_from_a_synthetic_trim(
    tmp_path: Path, true_offset: int
) -> None:
    video, landmarks, confidences = _synthetic_scan_inputs(tmp_path, true_offset)
    curves, scanned = clock.scan_video(
        video,
        landmarks,
        confidences,
        rgb=False,
        velocity_floor=1.0,
        chunk_starts=(0,),
        chunk_frames=240,
    )
    assert scanned == 240
    decision = clock.decide_offset(curves)
    assert not decision.ambiguous
    assert decision.chosen == true_offset


def test_rescore_reproduces_a_decision_from_stored_curves(tmp_path: Path) -> None:
    from battle.schemas import ArtifactFingerprint

    curves = [_curve(_peaked(6.0), "gradient", 0), _curve(_peaked(6.2), "motion", 0)]
    fingerprint = ArtifactFingerprint(uri="x", sha256="0" * 64, source="measured")
    decision = clock.decide_offset(curves)
    scan = clock.Assembly101ClockOffsetScan(
        manifest_kind="assembly101_clock_offset_scan",
        recording_id="rec",
        view="C10095",
        view_key="C10095:rgb",
        video=fingerprint,
        shipped_2d_window=fingerprint,
        hand_confidences=fingerprint,
        frames_scanned=1,
        curves=tuple(curves),
        metric_medians={},
        chosen_offset_subframe=None,
        chosen_offset_frames=None,
        offset_uncertainty_frames=None,
        ambiguous=True,
        evidence="stale",
        clock_rule=None,
        runtime_seconds=0.0,
        time_to_first_output_seconds=0.0,
        claim_boundaries=("fixture",),
    )
    rescored = clock.rescore(scan)
    assert rescored.chosen_offset_frames == decision.chosen == 6
    assert rescored.clock_rule is not None and rescored.clock_rule.pose_frame(1) == 17640 + 2 + 6
    assert clock.view_key("HMC_21179183") == "21179183:mono10bit"
    assert clock.npz_key("C10095") == "C10095_rgb"


# --- camera model and fit --------------------------------------------------------------


def _legacy_camera_json() -> str:
    return json.dumps(
        {
            "schema_version": "1.0",
            "view_key": "C10379:rgb",
            "raw_image_size": [1920, 1080],
            "intrinsic_matrix": [[1250.0, 0.0, 954.0], [0.0, 1250.0, 528.0], [0.0, 0.0, 1.0]],
            "distortion": [-0.13, 0.12, 0.0, 0.0, -0.02],
            "camera_to_world": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, -800.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "provenance": "estimated_from_dataset_landmark_projection",
            "fit_rms_pixels": 0.0001,
            "fit_point_count": 10,
        }
    )


def test_camera_model_keeps_legacy_files_and_validates_new_models() -> None:
    legacy = Assembly101CameraModel.model_validate_json(_legacy_camera_json())
    assert legacy.distortion_model == "brown" and legacy.extrinsics_kind == "fixed"
    assert not legacy.is_ego
    ego = Assembly101CameraModel(
        view_key="21110305:mono10bit",
        raw_image_size=(636, 480),
        intrinsic_matrix=((189.0, 0.0, 313.0), (0.0, 189.0, 234.0), (0.0, 0.0, 1.0)),
        distortion=(1.4, 0.2, 0.0, 0.0, 0.002, 1.5, 0.5, 0.02),
        distortion_model="rational",
        extrinsics_kind="per_frame_ego",
        provenance="estimated_from_dataset_landmark_projection",
        fit_rms_pixels=0.03,
        fit_point_count=100,
    )
    assert ego.is_ego and ego.camera_to_world is None
    with pytest.raises(ValueError, match="coefficients"):
        ego.model_copy(update={"distortion": (1.0,) * 5}).model_validate(
            ego.model_copy(update={"distortion": (1.0,) * 5}).model_dump()
        )
    with pytest.raises(ValueError, match="camera_to_world"):
        Assembly101CameraModel.model_validate({**legacy.model_dump(), "camera_to_world": None})


def test_calibration_recovers_synthetic_brown_intrinsics_through_the_given_pose() -> None:
    rng = np.random.default_rng(3)
    intrinsic = np.array([[1240.0, 0.0, 950.0], [0.0, 1245.0, 530.0], [0.0, 0.0, 1.0]])
    distortion = np.array([-0.12, 0.10, 1e-4, 2e-4, -0.02])
    rotation, _ = cv2.Rodrigues(np.array([0.2, -0.1, 0.05]))
    translation = np.array([20.0, -30.0, 900.0])
    world_to_camera = np.eye(4)
    world_to_camera[:3, :3] = rotation
    world_to_camera[:3, 3] = translation
    objects, images, poses = [], [], []
    for _ in range(40):
        points = rng.uniform([-250, -150, -120], [250, 150, 120], size=(30, 3))
        projected, _ = cv2.projectPoints(
            points, cv2.Rodrigues(rotation)[0], translation, intrinsic, distortion
        )
        objects.append(points.reshape(-1, 1, 3).astype(np.float32))
        images.append(projected.reshape(-1, 1, 2).astype(np.float32))
        poses.append(world_to_camera)
    rms, fitted_k, fitted_d = fit._calibrate(
        objects, images, (1920, 1080), fit.STATIC_INITIAL_K, fit.BROWN_FLAGS
    )
    residual = fit.shipped_extrinsics_residuals(objects, images, poses, fitted_k, fitted_d[:5])
    assert rms < 0.05
    assert np.sqrt(np.mean(residual**2)) < 0.1
    assert np.allclose(fitted_k, intrinsic, atol=0.5)
    assert np.allclose(fitted_d[:5], distortion, atol=2e-3)


def test_camera_estimate_paths_use_lower_case_view_names() -> None:
    assert fit.camera_estimate_path("C10095").name == "c10095_camera_estimate.json"
    assert fit.camera_estimate_path("HMC_21179183").name == "hmc_21179183_camera_estimate.json"
    assert fit.raw_image_size("HMC_21179183") == (636, 480)


# --- real data -------------------------------------------------------------------------

ALL_VIEWS = (*fetch.STATIC_VIEWS, *fetch.EGO_VIEWS)


def test_checked_in_camera_estimates_cover_every_view_with_tiny_shipped_pose_residuals() -> None:
    for view in ALL_VIEWS:
        path = fit.camera_estimate_path(view)
        assert path.is_file(), path
        model = Assembly101CameraModel.model_validate_json(path.read_text())
        assert model.view_key == clock.view_key(view)
        if view.startswith("HMC_"):
            assert model.is_ego and model.distortion_model == "rational"
        else:
            assert model.distortion_model == "brown" and model.camera_to_world is not None
            assert model.shipped_extrinsics_rms_pixels is None or (
                model.shipped_extrinsics_rms_pixels < 0.01
            )


def test_checked_in_clock_rules_cover_every_view_and_keep_the_c10379_offset() -> None:
    rules = clock.load_clock_rules(Path.cwd())
    assert set(rules.views) == set(ALL_VIEWS)
    assert not any(entry.ambiguous for entry in rules.views.values())
    assert rules.rule("C10379").pose_offset_frames == 9
    for view in fetch.EGO_VIEWS:
        assert rules.rule(view).pose_offset_frames == 0
    for view in fetch.STATIC_VIEWS:
        assert 4 <= rules.rule(view).pose_offset_frames <= 10
    with pytest.raises(KeyError):
        rules.rule("C99999")


def test_all_static_clip_config_lists_eight_proxies_with_the_pinned_revision() -> None:
    from battle.schemas import G2PreprocessingManifest

    manifest = G2PreprocessingManifest.model_validate_json(fetch.ALL_STATIC_CLIP_CONFIG.read_text())
    assert manifest.clip.views == tuple(fetch.static_view_id(v) for v in fetch.STATIC_VIEWS)
    assert manifest.hf_revision == fetch.DATASET_REVISION
    for proxy in manifest.proxies:
        assert proxy.frame_count == 2781 and proxy.fps == 30
        if proxy.view_id != "static-c10379":
            assert proxy.raw_source.raw_uri.startswith("hf://datasets/cvml-nus/assembly101@")


@pytest.mark.real_data
def test_acquired_windows_exist_with_the_expected_frame_counts() -> None:
    for view in ALL_VIEWS:
        record = fetch.load_record(view, require_artifact(Path.cwd()))
        require_artifact(fetch.per_view_record_path(view, Path.cwd()))
        assert record.raw60.frame_count == 5562 and record.raw60.fps == 60
        assert record.proxy.frame_count == 2781 and record.proxy.fps == 30
        assert require_artifact(record.proxy.uri).stat().st_size == record.proxy.size_bytes
        assert require_artifact(record.raw60.uri).stat().st_size == record.raw60.size_bytes


@pytest.mark.real_data
def test_offset_scans_on_disk_match_the_tracked_rules() -> None:
    rules = clock.load_clock_rules(Path.cwd())
    for view in ALL_VIEWS:
        scan = clock.load_scan(require_artifact(clock.scan_path(view)))
        assert scan.clock_rule == rules.rule(view)
        assert scan.frames_scanned == 5400
