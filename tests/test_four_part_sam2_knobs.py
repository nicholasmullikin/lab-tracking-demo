"""SAM2 model/input-size/schedule knobs on the four-part driver and worker (no GPU)."""

from __future__ import annotations

import contextlib
from pathlib import Path

import pytest
from conftest import require_artifact
from pydantic import ValidationError

from battle import four_part_segmentation as driver
from battle import four_part_video_worker as worker
from battle.schemas import (
    ArtifactFingerprint,
    FourPartSegmentationRunMetadata,
    FourPartTargetInitialization,
    Sam2ArmSettings,
    VramExtrapolation,
    VramProbe,
)

LEGACY_TAIL = [
    "--sam2-root",
    "/home/nick/src/DAM4SAM",
    "--sam2-config",
    "sam21pp_hiera_t.yaml",
    "--checkpoint",
    "/home/nick/src/DAM4SAM/checkpoints/sam2.1_hiera_tiny.pt",
]


def _driver_args(*extra: str) -> object:
    return driver.build_parser().parse_args(["dam4sam", *extra])


def _worker_args(*extra: str) -> object:
    return worker.build_parser().parse_args(
        [
            "--method",
            "dam4sam",
            "--run-directory",
            "run",
            "--video",
            "in.mp4",
            "--contract",
            "c.json",
            "--source-offset-seconds",
            "294",
            *LEGACY_TAIL,
            *extra,
        ]
    )


def test_driver_defaults_reproduce_the_previous_dam4sam_worker_command() -> None:
    args = _driver_args()
    assert (args.sam2_model, args.input_size, args.multi_keyframe_correction_schedule) == (
        "tiny",
        1024,
        None,
    )
    assert args.add_correction_to_drm is False and args.smoke_correction_frame is None
    assert args.proxy_view_id is None and args.proxy_clip_config is None
    assert args.vram_probe_frames == "30,300"
    assert args.extrapolate_to_frames == 1800
    assert args.fail_if_extrapolated_vram_over_bytes == 12 * 1024**3
    assert driver.worker_flags(args, view_id=None, schedule=None) == []

    command = driver.worker_command(
        method="dam4sam",
        run_directory=Path("/r"),
        input_video=Path("/r/input.mp4"),
        worker_contract_path=Path("/r/reviewed_seed_contract.json"),
        source_offset_seconds=294.0,
        frame_count=1800,
        sam2_model=args.sam2_model,
        extra_flags=driver.worker_flags(args, view_id=None, schedule=None),
    )
    # The exact argv that produced runs/dam4sam-four-part-reviewed-seed-60s-20260918t005416z.
    assert command[2:] == [
        "--method",
        "dam4sam",
        "--run-directory",
        "/r",
        "--video",
        "/r/input.mp4",
        "--contract",
        "/r/reviewed_seed_contract.json",
        "--source-offset-seconds",
        "294.0",
        "--analysis-fps",
        "30",
        "--frame-count",
        "1800",
        *LEGACY_TAIL,
    ]


def test_driver_knobs_become_worker_flags_only_when_set() -> None:
    args = _driver_args(
        "--sam2-model",
        "large",
        "--input-size",
        "1536",
        "--add-correction-to-drm",
        "--vram-probe-frames",
        "10,100",
    )
    flags = driver.worker_flags(args, view_id="static-c10379-1080p", schedule=Path("/r/s.json"))
    assert flags == [
        "--view-id",
        "static-c10379-1080p",
        "--sam2-model",
        "large",
        "--input-size",
        "1536",
        "--multi-keyframe-correction-schedule",
        "/r/s.json",
        "--add-correction-to-drm",
        "--vram-probe-frames",
        "10,100",
    ]
    assert driver.method_sam2_files("dam4sam", "large") == (
        "sam21pp_hiera_l.yaml",
        Path("/home/nick/src/DAM4SAM/checkpoints/sam2.1_hiera_large.pt"),
    )
    assert driver.method_sam2_files("samurai", "large") == (
        "configs/samurai/sam2.1_hiera_l.yaml",
        Path("/home/nick/src/samurai/sam2/checkpoints/sam2.1_hiera_large.pt"),
    )
    # The offline arms get the large pair through --sam2-config/--checkpoint and refuse the
    # DAM4SAM-only --sam2-model knob (the first samurai-large queue job died on it).
    samurai_args = driver.build_parser().parse_args(["samurai", "--sam2-model", "large"])
    samurai_flags = driver.worker_flags(samurai_args, view_id=None, schedule=Path("/r/s.json"))
    assert samurai_flags == ["--multi-keyframe-correction-schedule", "/r/s.json"]
    samurai_command = driver.worker_command(
        method="samurai",
        run_directory=Path("/r/run"),
        input_video=Path("/r/input.mp4"),
        worker_contract_path=Path("/r/contract.json"),
        source_offset_seconds=294.0,
        frame_count=1800,
        sam2_model="large",
        extra_flags=samurai_flags,
    )
    assert "--sam2-model" not in samurai_command
    assert samurai_command[samurai_command.index("--checkpoint") + 1] == (
        "/home/nick/src/samurai/sam2/checkpoints/sam2.1_hiera_large.pt"
    )
    assert samurai_command[samurai_command.index("--sam2-config") + 1] == (
        "configs/samurai/sam2.1_hiera_l.yaml"
    )
    for method, spec in driver.METHODS.items():
        assert driver.method_sam2_files(method, "tiny") == (spec["config"], spec["checkpoint"])
    with pytest.raises(ValueError, match="no SAM2"):
        driver.method_sam2_files("dam4sam", "base_plus")
    with pytest.raises(SystemExit):
        _driver_args("--input-size", "1280")
    assert driver.FRAME_COUNT_CHOICES == (300, 600, 1800)
    assert _driver_args("--frame-count", "300").frame_count == 300


def test_worker_defaults_match_the_driver_defaults() -> None:
    args = _worker_args()
    assert args.sam2_model == worker.DEFAULT_SAM2_MODEL == driver.DEFAULT_SAM2_MODEL == "tiny"
    assert args.input_size == worker.DEFAULT_INPUT_SIZE == 1024
    assert args.multi_keyframe_correction_schedule is None
    assert args.add_correction_to_drm is False
    assert args.vram_probe_frames == driver.DEFAULT_VRAM_PROBE_FRAMES == "30,300"
    assert args.view_id == "static-c10379"
    assert worker.SUPPORTED_FRAME_COUNTS == (300, 600, 1800)
    with pytest.raises(SystemExit):
        _worker_args("--sam2-model", "base_plus")


def test_filter_corrections_bounds_the_schedule_and_rebases_only_for_a_smoke() -> None:
    corrections = [
        {"frame_index": 0, "target": "chassis"},
        {"frame_index": 327, "target": "chassis"},
        {"frame_index": 327, "target": "cabin"},
        {"frame_index": 900, "target": "interior"},
        {"frame_index": 1235, "target": "rear_body"},
    ]
    applied, dropped, source = driver.filter_corrections(corrections, frame_count=1800)
    assert [item["frame_index"] for item in applied] == [327, 327, 900, 1235]
    assert dropped == () and source is None

    applied, dropped, source = driver.filter_corrections(corrections, frame_count=600)
    assert [item["frame_index"] for item in applied] == [327, 327]
    assert dropped == (900, 1235)

    applied, dropped, source = driver.filter_corrections(corrections, frame_count=300)
    assert applied == [] and dropped == (327, 900, 1235)

    applied, dropped, source = driver.filter_corrections(
        corrections, frame_count=300, smoke_correction_frame=150
    )
    assert [(item["frame_index"], item["target"]) for item in applied] == [
        (150, "chassis"),
        (150, "cabin"),
    ]
    assert all(item["source_frame_index"] == 327 for item in applied)
    assert dropped == (327, 900, 1235) and source == 327
    with pytest.raises(ValueError, match="inside the run"):
        driver.filter_corrections(corrections, frame_count=300, smoke_correction_frame=300)
    with pytest.raises(ValueError, match="no later correction"):
        driver.filter_corrections(corrections[:1], frame_count=300, smoke_correction_frame=10)


def _fingerprint(uri: str) -> ArtifactFingerprint:
    return ArtifactFingerprint(uri=uri, sha256="a" * 64, source="measured")


def _metadata(
    frame_count: int, requested_seconds: float, **overrides
) -> FourPartSegmentationRunMetadata:
    targets = ("chassis", "interior", "rear_body", "cabin")
    payload = {
        "method_arm": "dam4sam",
        "requested_analysis_frame_range": {"start_frame": 0, "end_frame_exclusive": frame_count},
        "requested_seconds": requested_seconds,
        "target_order": targets,
        "source_fingerprint": _fingerprint("raw.mp4"),
        "proxy_fingerprint": _fingerprint("proxy.mp4"),
        "contract_fingerprint": _fingerprint("contract.json"),
        "reviewed_schedule_fingerprint": _fingerprint("schedule.json"),
        "adapter": {
            "name": "dam4sam",
            "version": "1.1.0",
            "implementation_basis": "shared predictor",
            "external_source_uri": "/home/nick/src/DAM4SAM",
            "external_revision": "9c95",
        },
        "runtime_settings": {"analysis_fps": 30, "frame_count": frame_count},
        "measurements": {"elapsed_seconds": 1.0},
        "observations_uri": "runs/x/observations.jsonl",
        "native_masks_uri": "runs/x/native/masks",
        "target_initializations": [
            FourPartTargetInitialization(
                target_id=target,
                source="reviewed_mask",
                state="succeeded",
                reviewed_mask_fingerprint=_fingerprint(f"{target}.png"),
            )
            for target in targets
        ],
        "drm_memory_additions": dict.fromkeys(targets, 0),
    }
    payload.update(overrides)
    return FourPartSegmentationRunMetadata.model_validate(payload)


def _sam2_settings(**overrides) -> Sam2ArmSettings:
    payload = {
        "sam2_model": "large",
        "sam2_checkpoint_fingerprint": _fingerprint("/home/nick/src/DAM4SAM/checkpoints/large.pt"),
        "sam2_checkpoint_sha256_pinned": True,
        "sam2_config_fingerprint": _fingerprint(
            "runs/x/sam2_config/sam21pp_hiera_l_image1536.yaml"
        ),
        "sam2_config_source": "battle_input_size_copy",
        "input_image_size": 1536,
        "shared_predictor": True,
        "correction_schedule_fingerprint": _fingerprint(
            "runs/s/multi_keyframe_correction_schedule.json"
        ),
        "scheduled_correction_frame_indices": (327, 900, 1172, 1235),
        "dropped_correction_frame_indices": (1800, 2700),
        "correction_api": "add_new_mask",
        "correction_timing": "mid_stream_after_track",
        "add_correction_to_drm": False,
    }
    payload.update(overrides)
    return Sam2ArmSettings.model_validate(payload)


def test_manifest_records_the_sam2_provenance_and_vram_projection() -> None:
    probes = (
        VramProbe(
            frames_processed=30, memory_allocated_bytes=3_000, max_memory_allocated_bytes=4_000
        ),
        VramProbe(
            frames_processed=300, memory_allocated_bytes=3_270, max_memory_allocated_bytes=4_300
        ),
    )
    extrapolation = VramExtrapolation(
        basis="linear_between_two_probes",
        from_frames=(30, 300),
        slope_bytes_per_frame=1.0,
        extrapolate_to_frames=1800,
        projected_allocated_bytes=4_770,
        projected_peak_bytes=5_800,
        limit_bytes=12 * 1024**3,
        within_limit=True,
    )
    metadata = _metadata(
        300,
        10.0,
        sam2_settings=_sam2_settings(),
        vram_probes=probes,
        vram_extrapolation=extrapolation,
    )
    dumped = metadata.model_dump(mode="json")
    assert dumped["sam2_settings"]["sam2_model"] == "large"
    assert dumped["sam2_settings"]["shared_predictor"] is True
    assert dumped["sam2_settings"]["input_image_size"] == 1536
    assert dumped["sam2_settings"]["correction_api"] == "add_new_mask"
    assert dumped["sam2_settings"]["scheduled_correction_frame_indices"] == [327, 900, 1172, 1235]
    assert dumped["sam2_settings"]["add_correction_to_drm"] is False
    assert dumped["sam2_settings"]["sam2_checkpoint_fingerprint"]["sha256"] == "a" * 64
    assert dumped["sam2_settings"]["sam2_config_fingerprint"]["uri"].endswith("_image1536.yaml")
    assert [probe["frames_processed"] for probe in dumped["vram_probes"]] == [30, 300]
    assert dumped["vram_extrapolation"]["within_limit"] is True
    # Older manifests without the block still validate; 10 s is only valid as 300 frames.
    assert _metadata(600, 20.0).sam2_settings is None
    with pytest.raises(ValidationError, match=r"\[0, 300\)"):
        _metadata(600, 10.0)
    with pytest.raises(ValidationError, match="probes it was computed from"):
        _metadata(300, 10.0, vram_probes=probes[:1], vram_extrapolation=extrapolation)
    with pytest.raises(ValidationError, match="within_limit"):
        extrapolation.model_copy(update={"within_limit": False}).model_validate(
            extrapolation.model_dump() | {"within_limit": False}
        )


def test_sam2_settings_validate_correction_and_smoke_provenance() -> None:
    assert _sam2_settings().correction_timing == "mid_stream_after_track"
    smoke = _sam2_settings(
        scheduled_correction_frame_indices=(150,),
        dropped_correction_frame_indices=(327, 900, 1172, 1235),
        smoke_correction_frame=150,
        smoke_correction_source_frame=327,
    )
    assert smoke.smoke_correction_source_frame == 327
    with pytest.raises(ValidationError, match="schedule, API and timing"):
        _sam2_settings(correction_api=None)
    with pytest.raises(ValidationError, match="applied and source frame"):
        _sam2_settings(smoke_correction_frame=150)
    with pytest.raises(ValidationError, match="exactly its rebased frame"):
        _sam2_settings(smoke_correction_frame=150, smoke_correction_source_frame=327)
    with pytest.raises(ValidationError, match="meaningless"):
        _sam2_settings(
            scheduled_correction_frame_indices=(),
            correction_schedule_fingerprint=None,
            correction_api=None,
            correction_timing=None,
            add_correction_to_drm=True,
        )
    seed_only = _sam2_settings(
        scheduled_correction_frame_indices=(),
        correction_schedule_fingerprint=None,
        correction_api=None,
        correction_timing=None,
    )
    assert seed_only.scheduled_correction_frame_indices == ()


PROXY_1080P_CONFIG = (
    "configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_1080p_g2.json"
)
SAM3_SCHEDULE = Path(
    "runs/muggledsam-sam3-four-part-focused-corrections-agent-swap-20260918t000947z/"
    "multi_keyframe_correction_schedule.json"
)


@pytest.mark.real_data
@pytest.mark.parametrize(
    ("frame_count", "smoke_frame", "expected_scheduled", "expected_dropped"),
    [
        (1800, None, [327, 900, 1172, 1235], [1800, 2700]),
        (300, 150, [150], [327, 900, 1172, 1235, 1800, 2700]),
    ],
)
def test_the_sam3_schedule_resolves_for_the_sam2_arms(
    tmp_path: Path,
    frame_count: int,
    smoke_frame: int | None,
    expected_scheduled: list[int],
    expected_dropped: list[int],
) -> None:
    from battle.four_part_contract import load_contract
    from battle.schemas import G2PreprocessingManifest

    root = Path.cwd()
    require_artifact(root / SAM3_SCHEDULE)
    contract = load_contract(root, driver.DEFAULT_CONTRACT)
    clip_config = root / "configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_g2.json"
    config = G2PreprocessingManifest.model_validate_json(clip_config.read_text())

    path, resolved = driver.resolve_correction_schedule(
        schedule_path=root / SAM3_SCHEDULE,
        repository_root=root,
        clip_config_path=clip_config,
        authoring_proxy=config.proxies[0],
        contract=contract,
        frame_count=frame_count,
        smoke_correction_frame=smoke_frame,
        destination=tmp_path / driver.RESOLVED_SCHEDULE_NAME,
    )

    assert path.is_file()
    assert resolved["scheduled_correction_frame_indices"] == expected_scheduled
    assert resolved["dropped_correction_frame_indices"] == expected_dropped
    assert resolved["correction_api"] == "add_new_mask"
    assert resolved["schedule_frame_zero_seeds_match_contract"] is True
    assert resolved["smoke_correction_source_frame"] == (327 if smoke_frame else None)
    for entry in resolved["corrections"]:
        assert Path(entry["mask_path"]).is_file()
        assert entry["target"] in ("chassis", "interior", "rear_body", "cabin")
    if smoke_frame is None:
        assert [item["frame_index"] for item in resolved["corrections"]] == [
            327,
            327,
            327,
            327,
            900,
            900,
            1172,
            1172,
            1235,
            1235,
            1235,
        ]


def _fake_worker(result_extra: dict[str, object]):
    """Stand in for the external worker: 300 empty observations plus a result payload."""
    import json
    import subprocess

    real_run = subprocess.run

    def fake_run(command, **kwargs):
        if "--run-directory" not in command:  # ffmpeg bounding the input video
            return real_run(command, **kwargs)
        run_directory = Path(command[command.index("--run-directory") + 1])
        frame_count = int(command[command.index("--frame-count") + 1])
        offset = float(command[command.index("--source-offset-seconds") + 1])
        with (run_directory / "observations.jsonl").open("w") as handle:
            for index in range(frame_count):
                handle.write(
                    json.dumps(
                        {
                            "view_id": "static-c10379",
                            "analysis_frame_index": index,
                            "source_seconds": offset + index / 30,
                            "objects": [],
                        }
                    )
                    + "\n"
                )
        (run_directory / "worker_result.json").write_text(
            json.dumps(
                {
                    "state": "succeeded",
                    "elapsed_seconds": 1.5,
                    "gpu_peak_vram_bytes": 6_000,
                    "initialization": {},
                    "drm_memory_additions": {
                        "chassis": 1,
                        "interior": 0,
                        "rear_body": 0,
                        "cabin": 0,
                    },
                    **result_extra,
                }
            )
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    return fake_run


@pytest.mark.real_data
@pytest.mark.slow
def test_driver_assembles_sam2_provenance_and_gates_on_the_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path.cwd()
    require_artifact(root / SAM3_SCHEDULE)
    require_artifact(root / "configs/four_part_segmentation_comparison.json")
    config_copy = tmp_path / "sam21pp_hiera_l_image1536.yaml"
    config_copy.write_text("model:\n  image_size: 1536\n")
    checkpoint = tmp_path / "large.pt"
    checkpoint.write_bytes(b"not a checkpoint")
    monkeypatch.setattr(
        driver.subprocess,
        "run",
        _fake_worker(
            {
                "sam2": {
                    "sam2_model": "large",
                    "sam2_checkpoint_uri": str(checkpoint),
                    "sam2_checkpoint_sha256": "b" * 64,
                    "sam2_checkpoint_sha256_pinned": True,
                    "sam2_config_uri": str(config_copy),
                    "sam2_config_sha256": "c" * 64,
                    "sam2_config_source": "battle_input_size_copy",
                    "input_image_size": 1536,
                    "shared_predictor": True,
                    "add_correction_to_drm": False,
                },
                "vram_probes": [
                    {
                        "frames_processed": 30,
                        "memory_allocated_bytes": 3 * 1024**3,
                        "max_memory_allocated_bytes": 5 * 1024**3,
                    },
                    {
                        "frames_processed": 300,
                        "memory_allocated_bytes": 3 * 1024**3 + 270 * 2**20,
                        "max_memory_allocated_bytes": 5 * 1024**3 + 270 * 2**20,
                    },
                ],
                "corrections_applied": [{"frame_index": 150, "target": "chassis"}],
                "correction_masks_resized_to_frame": False,
                "seed_masks_resized_to_frame": False,
            }
        ),
    )
    # Slope 1 MiB/frame -> peak 5.26 GiB + 1500 MiB = 6.73 GiB at 1800 frames.
    # Run artifacts must live under the repository root (manifest URIs are relative to it).
    output_root = Path("runs") / "_pytest_sam2_knobs" / tmp_path.name
    monkeypatch.setattr(driver, "_contact_sheet", lambda *args, **kwargs: None)
    try:
        _run_driver_and_assert(root, output_root)
    finally:
        import shutil

        shutil.rmtree(root / output_root, ignore_errors=True)
        with contextlib.suppress(OSError):
            (root / output_root).parent.rmdir()


def _run_driver_and_assert(root: Path, output_root: Path) -> None:
    args = _driver_args(
        "--repository-root",
        str(root),
        "--output-root",
        str(output_root),
        "--run-id",
        "smoke-large-1536",
        "--frame-count",
        "300",
        "--sam2-model",
        "large",
        "--input-size",
        "1536",
        "--multi-keyframe-correction-schedule",
        str(SAM3_SCHEDULE),
        "--smoke-correction-frame",
        "150",
    )
    manifest_path = driver.run(args)
    from battle.schemas import RunManifest

    manifest = RunManifest.model_validate_json(manifest_path.read_text())
    settings = manifest.four_part_segmentation.sam2_settings
    assert settings is not None
    assert (settings.sam2_model, settings.input_image_size, settings.shared_predictor) == (
        "large",
        1536,
        True,
    )
    assert settings.sam2_config_source == "battle_input_size_copy"
    assert settings.sam2_checkpoint_sha256_pinned is True
    assert settings.correction_api == "add_new_mask"
    assert settings.correction_timing == "mid_stream_after_track"
    assert settings.scheduled_correction_frame_indices == (150,)
    assert settings.smoke_correction_source_frame == 327
    assert settings.dropped_correction_frame_indices == (327, 900, 1172, 1235, 1800, 2700)
    assert settings.schedule_frame_zero_seeds_match_contract is True
    assert manifest.four_part_segmentation.requested_seconds == 10.0
    projection = manifest.four_part_segmentation.vram_extrapolation
    assert projection is not None and projection.within_limit
    assert projection.slope_bytes_per_frame == pytest.approx(2**20)
    assert projection.projected_peak_bytes == 5 * 1024**3 + 1770 * 2**20
    assert (manifest_path.parent / driver.RESOLVED_SCHEDULE_NAME).is_file()
    command = (manifest_path.parent / "worker_command.txt").read_text().split()
    assert command[command.index("--sam2-model") + 1] == "large"
    assert command[command.index("--input-size") + 1] == "1536"
    assert "--checkpoint" in command
    assert command[command.index("--checkpoint") + 1].endswith("sam2.1_hiera_large.pt")

    gated = _driver_args(
        "--repository-root",
        str(root),
        "--output-root",
        str(output_root),
        "--run-id",
        "smoke-gated",
        "--frame-count",
        "300",
        "--sam2-model",
        "large",
        "--fail-if-extrapolated-vram-over-bytes",
        str(6 * 1024**3),
    )
    with pytest.raises(driver.ExtrapolatedVramOverLimit, match="exceeds"):
        driver.run(gated)
    assert (root / output_root / "smoke-gated" / "manifest.json").is_file()

    proxy_config = Path(PROXY_1080P_CONFIG)
    if not (root / proxy_config).is_file():
        return
    on_1080p = _driver_args(
        "--repository-root",
        str(root),
        "--output-root",
        str(output_root),
        "--run-id",
        "smoke-1080p",
        "--frame-count",
        "300",
        "--sam2-model",
        "large",
        "--input-size",
        "1536",
        "--proxy-clip-config",
        str(proxy_config),
    )
    manifest = RunManifest.model_validate_json(driver.run(on_1080p).read_text())
    proxy_fingerprint = manifest.four_part_segmentation.proxy_fingerprint
    assert "1920x1080" in proxy_fingerprint.uri
    assert proxy_fingerprint.uri != manifest.four_part_segmentation.source_fingerprint.uri
    command = (root / output_root / "smoke-1080p" / "worker_command.txt").read_text().split()
    assert "--view-id" not in command, "the 1080p proxy keeps the static-c10379 view id"
    assert command[command.index("--input-size") + 1] == "1536"


def test_vram_projection_line_and_gate_threshold() -> None:
    extrapolation = VramExtrapolation(
        basis="linear_between_two_probes",
        from_frames=(30, 300),
        slope_bytes_per_frame=2.5 * 2**20,
        extrapolate_to_frames=1800,
        projected_allocated_bytes=10 * 1024**3,
        projected_peak_bytes=13 * 1024**3,
        limit_bytes=12 * 1024**3,
        within_limit=False,
    )
    line = driver.vram_projection_line(extrapolation, frame_count=300)
    assert "2.500 MiB/frame" in line and "peak 13.00 GiB" in line and "OVER limit" in line
    assert "already covered" not in line
    assert "already covered" in driver.vram_projection_line(extrapolation, frame_count=1800)
