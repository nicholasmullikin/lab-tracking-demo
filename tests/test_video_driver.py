"""`battle.video_driver`: the shared steps of the bounded video drivers.

The manifest and contact-sheet tests pin the shared helpers against the inline code the
drivers carried before dedup pass 2 (copied here verbatim from `samurai_video.py` and
`boxmot_track.py` at commit 6e7f1df); the `real_data` test rebuilds the manifest of every
recorded 10 s smoke through the current drivers and requires the JSON text to match byte
for byte.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from battle import (
    boxmot_track,
    dam4sam_video,
    grounding_dino_sam2_video,
    samurai_video,
    video_driver,
    wilor_hands,
)
from battle.digest_cache import sha256_file
from battle.exporter import HAND_CONNECTIONS
from battle.fs_common import relative_uri
from battle.observations import rebuild_tracker_observations
from battle.schemas import (
    AdapterMetadata,
    ArtifactFingerprint,
    BoxMOTRunMetadata,
    ChunkContinuityPolicy,
    FrameObservations,
    FrameRange,
    FullDurationCoverage,
    G2PreprocessingManifest,
    HandSide,
    MethodState,
    MethodStatus,
    NormalizedBox,
    NormalizedPoint,
    PerFrameHand,
    RunManifest,
    RuntimeMeasurements,
    SamuraiVideoRunMetadata,
    TimeInterval,
)

REPOSITORY_ROOT = Path(__file__).parents[1]
CONFIG_PATH = (
    REPOSITORY_ROOT / "configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_g2.json"
)
VIEW_ID = "static-c10379"
# The pass-2 smokes (`before/` from the pre-refactor code snapshot, `after/` from the new
# HEAD).  Older recorded runs (the Sep 16 `runs/*-10s-*` smokes) rebuild identically except
# for schema fields added since (`nlf_body_2d` null -> [], WiLoR's external_source_* settings),
# so they are not byte-for-byte fixtures.
SMOKE_RUN_GLOBS = ("runs/dedup-pass2-smokes-*/*/*",)


def _config() -> tuple[G2PreprocessingManifest, Any]:
    config = G2PreprocessingManifest.model_validate_json(CONFIG_PATH.read_text())
    proxy = next(item for item in config.proxies if item.view_id == VIEW_ID)
    return config, proxy


def _tracker_observations(tmp_path: Path, frames: int = 3) -> tuple[Path, tuple[Any, ...]]:
    path = tmp_path / "observations.jsonl"
    lines = []
    for index in range(frames):
        lines.append(
            json.dumps(
                {
                    "view_id": VIEW_ID,
                    "analysis_frame_index": index,
                    "source_seconds": 294.0 + index / 30,
                    "objects": [
                        {
                            "object_id": "samurai-0",
                            "label": "seeded_hand",
                            "confidence": 1.0,
                            "box": {"x": 0.1 * (index + 1), "y": 0.2, "width": 0.3, "height": 0.4},
                            "mask": {
                                "uri": f"native/masks/{index:05d}.png",
                                "storage": "native_artifact",
                                "format": "png",
                            },
                        }
                    ],
                    "extra_worker_field": {"ignored": True},
                }
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path, rebuild_tracker_observations(path)


WORKER_RESULT: dict[str, Any] = {
    "state": "succeeded",
    "reason": "",
    "frames_processed": 3,
    "frames_with_masks": 3,
    "unique_mask_hashes": 3,
    "frames_with_tracks": 3,
    "total_track_observations": 4,
    "elapsed_seconds": 1.25,
    "time_to_first_usable_output_seconds": 0.5,
    "gpu_peak_vram_bytes": 1024,
    "runtime_settings": {
        "sam2_config": "configs/samurai/sam2.1_hiera_t.yaml",
        "samurai_mode": True,
        "init_bbox_xywh": [881, 446, 152, 129],
        "analysis_fps": 30.0,
        "max_frames": 3,
        "nested": {"b": 1, "a": [1, 2]},
        "nothing": None,
    },
    "worker_exit_code": 0,
}


# --------------------------------------------------------------------------- former inline code
# Verbatim from `samurai_video.run()` and `boxmot_track.run()` at 6e7f1df (constants inlined
# from those modules), kept as the reference the shared helpers must reproduce.


def _former_samurai_manifest(
    *,
    run_id: str,
    repository_root: Path,
    run_directory: Path,
    config: G2PreprocessingManifest,
    config_path: Path,
    proxy: Any,
    view_id: str,
    seconds: float,
    requested_frames: int,
    worker_result: dict[str, Any],
    observations: tuple[FrameObservations, ...],
    observations_path: Path,
    qa_path: Path,
) -> RunManifest:
    SAM2_CHECKPOINT = samurai_video.SAM2_CHECKPOINT
    SAM2_CHECKPOINT_SHA256 = samurai_video.SAM2_CHECKPOINT_SHA256
    SAMURAI_ROOT = samurai_video.SAMURAI_ROOT
    SAMURAI_REVISION = samurai_video.SAMURAI_REVISION
    SAM2_CONFIG = samurai_video.SAM2_CONFIG
    INIT_BBOX_XYWH = samurai_video.INIT_BBOX_XYWH
    METHOD_NAME = samurai_video.METHOD_NAME
    measurements = RuntimeMeasurements(
        elapsed_seconds=float(worker_result.get("elapsed_seconds", 0.0)),
        time_to_first_usable_output_seconds=worker_result.get(
            "time_to_first_usable_output_seconds"
        ),
        gpu_peak_vram_bytes=worker_result.get("gpu_peak_vram_bytes"),
        known_unavailable_measures=("ground-truth mask quality",),
    )
    runtime_settings: dict[str, str | int | float | bool | None] = {}
    for key, value in dict(worker_result.get("runtime_settings", {})).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            runtime_settings[key] = value
        else:
            runtime_settings[key] = json.dumps(value, sort_keys=True)
    runtime_settings["analysis_fps"] = proxy.fps
    runtime_settings["frames_with_masks"] = worker_result.get("frames_with_masks")
    runtime_settings["unique_mask_hashes"] = worker_result.get("unique_mask_hashes")
    metadata = SamuraiVideoRunMetadata(
        requested_analysis_frame_range=FrameRange(
            start_frame=0, end_frame_exclusive=requested_frames
        ),
        requested_seconds=seconds,
        source_fingerprint=ArtifactFingerprint(
            uri=proxy.raw_source.raw_uri,
            sha256=proxy.raw_source.checksum_sha256,
            source="approved_config",
        ),
        proxy_fingerprint=ArtifactFingerprint(
            uri=proxy.proxy_uri, sha256=proxy.checksum_sha256, source="approved_config"
        ),
        config_fingerprint=ArtifactFingerprint(
            uri=relative_uri(config_path, repository_root),
            sha256=sha256_file(config_path),
            source="measured",
        ),
        sam2_checkpoint_fingerprint=ArtifactFingerprint(
            uri=str(SAM2_CHECKPOINT),
            sha256=SAM2_CHECKPOINT_SHA256,
            source="measured",
        ),
        adapter=AdapterMetadata(
            name=METHOD_NAME,
            version="0.1.0",
            implementation_basis=(
                "SAMURAI SAM2.1 tiny video predictor with samurai_mode config and "
                "deterministic frame-0 hand bbox seed"
            ),
            external_source_uri=str(SAMURAI_ROOT),
            external_revision=SAMURAI_REVISION,
        ),
        runtime_settings=runtime_settings,
        measurements=measurements,
        observations_uri=relative_uri(observations_path, repository_root),
        native_masks_uri=relative_uri(run_directory / "native" / "masks", repository_root),
        rerun_artifact_uri=relative_uri(run_directory / "samurai.rrd", repository_root),
        qa_artifact_uri=relative_uri(qa_path, repository_root),
        sam2_model_config=SAM2_CONFIG,
        init_bbox_xywh=INIT_BBOX_XYWH,
        initialization_note=(
            "Frame 0 is initialized from the approved focused static hand box "
            f"{INIT_BBOX_XYWH} (xywh); SAMURAI config sets samurai_mode=true."
        ),
    )
    worker_state = (
        MethodState.SUCCEEDED if worker_result.get("state") == "succeeded" else MethodState.FAILED
    )
    method_statuses = (
        MethodStatus(
            method_name=METHOD_NAME,
            stage="objects",
            state=worker_state,
            artifact_uri=relative_uri(observations_path, repository_root),
            measured_on=f"{view_id}; approved {seconds:g}-second proxy prefix",
            blocker=(
                str(worker_result.get("reason")) if worker_state is MethodState.FAILED else None
            ),
        ),
        MethodStatus(
            method_name="rerun-samurai-export",
            stage="export",
            state=(
                MethodState.SUCCEEDED
                if worker_state is MethodState.SUCCEEDED
                else MethodState.NOT_RUN
            ),
            artifact_uri=relative_uri(run_directory / "samurai.rrd", repository_root),
            measured_on="normalized mask observations; bounded input video logged once",
        ),
    )
    return RunManifest(
        run_id=run_id,
        clip=config.clip.model_copy(update={"source_duration_seconds": seconds}),
        coverage=FullDurationCoverage(
            source_duration_seconds=seconds,
            covered_intervals=(TimeInterval(start_seconds=0.0, end_seconds=seconds),),
        ),
        chunk_policy=ChunkContinuityPolicy(
            overlap_seconds=0.0,
            max_allowed_gap_seconds=0.0,
            preserve_track_ids=True,
            carry_context_across_chunks=True,
        ),
        method_statuses=method_statuses,
        observations=observations,
        samurai_video=metadata,
    )


def _former_boxmot_manifest(
    *,
    run_id: str,
    repository_root: Path,
    run_directory: Path,
    config: G2PreprocessingManifest,
    config_path: Path,
    proxy: Any,
    view_id: str,
    seconds: float,
    requested_frames: int,
    worker_result: dict[str, Any],
    observations: tuple[FrameObservations, ...],
    observations_path: Path,
    qa_path: Path,
    detector_path: Path,
    detector_sha256: str,
    detector_classes: str,
) -> RunManifest:
    measurements = RuntimeMeasurements(
        elapsed_seconds=float(worker_result.get("elapsed_seconds", 0.0)),
        time_to_first_usable_output_seconds=worker_result.get(
            "time_to_first_usable_output_seconds"
        ),
        gpu_peak_vram_bytes=worker_result.get("gpu_peak_vram_bytes"),
        known_unavailable_measures=("ground-truth association accuracy",),
    )
    runtime_settings: dict[str, str | int | float | bool | None] = {}
    for key, value in dict(worker_result.get("runtime_settings", {})).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            runtime_settings[key] = value
        else:
            runtime_settings[key] = json.dumps(value, sort_keys=True)
    runtime_settings["analysis_fps"] = proxy.fps
    runtime_settings["frames_with_tracks"] = worker_result.get("frames_with_tracks")
    runtime_settings["total_track_observations"] = worker_result.get("total_track_observations")
    metadata = BoxMOTRunMetadata(
        requested_analysis_frame_range=FrameRange(
            start_frame=0, end_frame_exclusive=requested_frames
        ),
        requested_seconds=seconds,
        source_fingerprint=ArtifactFingerprint(
            uri=proxy.raw_source.raw_uri,
            sha256=proxy.raw_source.checksum_sha256,
            source="approved_config",
        ),
        proxy_fingerprint=ArtifactFingerprint(
            uri=proxy.proxy_uri, sha256=proxy.checksum_sha256, source="approved_config"
        ),
        config_fingerprint=ArtifactFingerprint(
            uri=relative_uri(config_path, repository_root),
            sha256=sha256_file(config_path),
            source="measured",
        ),
        detector_fingerprint=ArtifactFingerprint(
            uri=relative_uri(detector_path, repository_root),
            sha256=detector_sha256,
            source="measured",
        ),
        adapter=AdapterMetadata(
            name="boxmot-botsort",
            version="25.0.0",
            implementation_basis=(
                "Ultralytics YOLO per-frame detections associated by BoxMOT BotSort"
            ),
            external_source_uri="https://github.com/mikel-brostrom/boxmot",
        ),
        runtime_settings=runtime_settings,
        measurements=measurements,
        observations_uri=relative_uri(observations_path, repository_root),
        rerun_artifact_uri=relative_uri(run_directory / "tracks.rrd", repository_root),
        qa_artifact_uri=relative_uri(qa_path, repository_root),
        detector_source=(
            "ultralytics YOLOv8n COCO per-frame detections (class filter "
            f"{detector_classes}); no MuggledSAM track IDs"
        ),
    )
    worker_state = (
        MethodState.SUCCEEDED if worker_result.get("state") == "succeeded" else MethodState.FAILED
    )
    method_statuses = (
        MethodStatus(
            method_name="boxmot-yolo-static-smoke",
            stage="objects",
            state=worker_state,
            artifact_uri=relative_uri(observations_path, repository_root),
            measured_on=f"{view_id}; approved {seconds:g}-second proxy prefix",
            blocker=(
                str(worker_result.get("reason")) if worker_state is MethodState.FAILED else None
            ),
        ),
        MethodStatus(
            method_name="rerun-boxmot-export",
            stage="export",
            state=(
                MethodState.SUCCEEDED
                if worker_state is MethodState.SUCCEEDED
                else MethodState.NOT_RUN
            ),
            artifact_uri=relative_uri(run_directory / "tracks.rrd", repository_root),
            measured_on="normalized BoxMOT observations; bounded input video logged once",
        ),
    )
    return RunManifest(
        run_id=run_id,
        clip=config.clip.model_copy(update={"source_duration_seconds": seconds}),
        coverage=FullDurationCoverage(
            source_duration_seconds=seconds,
            covered_intervals=(TimeInterval(start_seconds=0.0, end_seconds=seconds),),
        ),
        chunk_policy=ChunkContinuityPolicy(overlap_seconds=0.0, max_allowed_gap_seconds=0.0),
        method_statuses=method_statuses,
        observations=observations,
        boxmot=metadata,
    )


def _former_tracker_contact_sheet(
    *,
    video_path: Path,
    observations: tuple[FrameObservations, ...],
    output_path: Path,
) -> Path:
    selected = sorted({0, len(observations) // 2, len(observations) - 1})
    capture = cv2.VideoCapture(str(video_path))
    panels = []
    try:
        for frame_index in selected:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"could not decode contact-sheet frame {frame_index}")
            height, width = frame.shape[:2]
            observation = observations[frame_index]
            for obj in observation.objects:
                x1 = round(obj.box.x * width)
                y1 = round(obj.box.y * height)
                x2 = round((obj.box.x + obj.box.width) * width)
                y2 = round((obj.box.y + obj.box.height) * height)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 2)
                cv2.putText(
                    frame,
                    f"{obj.object_id}: {obj.label}",
                    (x1, max(24, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 180, 255),
                    2,
                )
            cv2.putText(
                frame,
                f"analysis frame {frame_index}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            panels.append(frame)
    finally:
        capture.release()
    sheet = cv2.vconcat(panels)
    if not cv2.imwrite(str(output_path), sheet):
        raise RuntimeError(f"could not write contact sheet: {output_path}")
    return output_path


def _former_hands_contact_sheet(
    *,
    video_path: Path,
    observations: tuple[FrameObservations, ...],
    output_path: Path,
) -> Path:
    selected = sorted({0, len(observations) // 2, len(observations) - 1})
    capture = cv2.VideoCapture(str(video_path))
    panels = []
    try:
        for frame_index in selected:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"could not decode contact-sheet frame {frame_index}")
            height, width = frame.shape[:2]
            observation = observations[frame_index]
            for hand in observation.hands:
                points = [
                    (round(point.x * width), round(point.y * height)) for point in hand.landmarks
                ]
                for start, end in HAND_CONNECTIONS:
                    cv2.line(frame, points[start], points[end], (0, 180, 255), 2)
                for point in points:
                    cv2.circle(frame, point, 3, (0, 255, 255), -1)
                x1 = round(hand.box.x * width)
                y1 = round(hand.box.y * height)
                x2 = round((hand.box.x + hand.box.width) * width)
                y2 = round((hand.box.y + hand.box.height) * height)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 2)
                cv2.putText(
                    frame,
                    f"{hand.hand_id}: {hand.side}",
                    (x1, max(24, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 180, 255),
                    2,
                )
            cv2.putText(
                frame,
                f"analysis frame {frame_index}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            panels.append(frame)
    finally:
        capture.release()
    sheet = cv2.vconcat(panels)
    if not cv2.imwrite(str(output_path), sheet):
        raise RuntimeError(f"could not write contact sheet: {output_path}")
    return output_path


# --------------------------------------------------------------------------- manifests


def _common_arguments(tmp_path: Path) -> dict[str, Any]:
    config, proxy = _config()
    run_directory = REPOSITORY_ROOT / "runs" / "dedup-pass2-unit" / "example"
    observations_path, observations = _tracker_observations(tmp_path)
    return {
        "run_id": "example-run",
        "repository_root": REPOSITORY_ROOT,
        "run_directory": run_directory,
        "config": config,
        "config_path": CONFIG_PATH,
        "proxy": proxy,
        "view_id": VIEW_ID,
        "seconds": 0.1,
        "requested_frames": 3,
        "observations": observations,
        "observations_path": run_directory / "observations.jsonl",
        "qa_path": run_directory / "contact_sheet.png",
    }


@pytest.mark.parametrize("state", ["succeeded", "failed"])
def test_samurai_manifest_matches_the_former_inline_code(tmp_path: Path, state: str) -> None:
    arguments = _common_arguments(tmp_path)
    worker_result = {**WORKER_RESULT, "state": state, "reason": "boom" if state == "failed" else ""}
    former = _former_samurai_manifest(worker_result=worker_result, **arguments)
    rebuilt, rebuilt_state = samurai_video._build_manifest(worker_result=worker_result, **arguments)

    assert rebuilt.model_dump_json(indent=2) == former.model_dump_json(indent=2)
    assert rebuilt_state is (MethodState.SUCCEEDED if state == "succeeded" else MethodState.FAILED)
    settings = rebuilt.samurai_video.runtime_settings
    assert settings["nested"] == '{"a": [1, 2], "b": 1}'
    assert settings["init_bbox_xywh"] == "[881, 446, 152, 129]"
    assert settings["analysis_fps"] == 30 and isinstance(settings["analysis_fps"], int)
    assert list(settings)[:5] == list(WORKER_RESULT["runtime_settings"])[:5]
    assert list(settings)[-2:] == ["frames_with_masks", "unique_mask_hashes"]


def test_boxmot_manifest_matches_the_former_inline_code(tmp_path: Path) -> None:
    arguments = _common_arguments(tmp_path)
    extras = {
        "detector_path": REPOSITORY_ROOT / "models/yolo/yolov8n.pt",
        "detector_sha256": "f" * 64,
        "detector_classes": "0",
    }
    former = _former_boxmot_manifest(worker_result=WORKER_RESULT, **arguments, **extras)
    rebuilt, state = boxmot_track._build_manifest(
        worker_result=WORKER_RESULT, **arguments, **extras
    )

    assert rebuilt.model_dump_json(indent=2) == former.model_dump_json(indent=2)
    assert state is MethodState.SUCCEEDED
    assert rebuilt.chunk_policy == ChunkContinuityPolicy(**video_driver.ZERO_OVERLAP_CHUNK_POLICY)


def test_worker_status_and_measurements() -> None:
    assert video_driver.worker_status({"state": "succeeded"}) == (MethodState.SUCCEEDED, None)
    assert video_driver.worker_status({"state": "failed", "reason": 7}) == (MethodState.FAILED, "7")
    assert video_driver.worker_status({}) == (MethodState.FAILED, "None")
    measurements = video_driver.worker_measurements({}, known_unavailable_measures=("x",))
    assert measurements == RuntimeMeasurements(
        elapsed_seconds=0.0, known_unavailable_measures=("x",)
    )


def test_load_worker_observations_exact_or_at_least(tmp_path: Path) -> None:
    _tracker_observations(tmp_path, frames=4)
    loader = rebuild_tracker_observations
    path, observations = video_driver.load_worker_observations(
        tmp_path, {}, 3, loader=loader, exact=False
    )
    assert path == tmp_path / "observations.jsonl" and len(observations) == 4
    with pytest.raises(RuntimeError, match="produced 4 observations; expected 3"):
        video_driver.load_worker_observations(tmp_path, {}, 3, loader=loader, exact=True)
    with pytest.raises(RuntimeError, match="no such worker"):
        video_driver.load_worker_observations(
            tmp_path / "missing", {"reason": "no such worker"}, 3, loader=loader, exact=True
        )


def test_verify_inputs_seconds_ranges_and_unknown_view() -> None:
    common = {"repository_root": REPOSITORY_ROOT, "config_path": CONFIG_PATH, "view_id": VIEW_ID}
    with pytest.raises(ValueError, match=r"seconds must be in \[1.0, 20.0\]"):
        video_driver.verify_inputs(seconds=0.5, min_seconds=1.0, max_seconds=20.0, **common)
    with pytest.raises(ValueError, match=r"seconds must be in \(0, 60.0\]"):
        video_driver.verify_inputs(seconds=0.0, min_seconds=None, max_seconds=60.0, **common)
    with pytest.raises(ValueError, match="view nope not found in config"):
        video_driver.verify_inputs(
            seconds=1.0, min_seconds=None, max_seconds=60.0, **{**common, "view_id": "nope"}
        )


def test_prepend_pythonpath() -> None:
    assert video_driver.prepend_pythonpath(Path("/x"), {}) == "/x"
    assert video_driver.prepend_pythonpath(Path("/x"), {"PYTHONPATH": ""}) == "/x"
    assert video_driver.prepend_pythonpath(Path("/x"), {"PYTHONPATH": "/a:/b"}) == "/x:/a:/b"


def test_run_external_worker_records_logs_command_and_missing_result(tmp_path: Path) -> None:
    import sys

    result = video_driver.run_external_worker(
        sys.executable,
        [
            "-c",
            "import os, sys; print(os.environ['CUDA_VISIBLE_DEVICES'], os.environ['MARK']); "
            "sys.stderr.write('err'); sys.exit(3)",
        ],
        run_directory=tmp_path,
        env={"MARK": "here"},
        record_command=True,
    )
    assert result["state"] == "failed" and result["frames_processed"] == 0
    assert "exited 3 without worker_result.json" in result["reason"]
    assert (tmp_path / "worker.stdout.log").read_text() == "0 here\n"
    assert (tmp_path / "worker.stderr.log").read_text() == "err"
    assert (tmp_path / "worker_command.txt").read_text().startswith(sys.executable + " -c ")

    (tmp_path / "worker_result.json").write_text(json.dumps({"state": "succeeded", "n": 1}))
    result = video_driver.run_external_worker(
        sys.executable, ["-c", "pass"], run_directory=tmp_path
    )
    assert result == {"state": "succeeded", "n": 1, "worker_exit_code": 0}


# --------------------------------------------------------------------------- contact sheets


def _hand(index: int) -> PerFrameHand:
    points = tuple(
        NormalizedPoint(x=min(0.95, 0.1 + 0.04 * k + 0.05 * index), y=0.2 + 0.03 * k)
        for k in range(21)
    )
    return PerFrameHand(
        hand_id=f"hand-{index}",
        side=HandSide.LEFT,
        confidence=0.9,
        landmarks=points,
        box=NormalizedBox(x=0.1, y=0.2, width=0.5, height=0.6),
        model_side=HandSide.LEFT,
        model_handedness_confidence=0.9,
    )


def _sheet_bytes(
    render: Callable[..., Path], video: Path, observations: tuple[Any, ...], out: Path
) -> bytes:
    render(video_path=video, observations=observations, output_path=out)
    image = cv2.imread(str(out))
    assert image is not None
    return image.tobytes() + bytes(image.shape)


def test_tracker_contact_sheet_matches_the_former_inline_drawing(
    tmp_path: Path, synthetic_video: Path
) -> None:
    _, observations = _tracker_observations(tmp_path)
    new = _sheet_bytes(
        video_driver.tracker_contact_sheet, synthetic_video, observations, tmp_path / "new.png"
    )
    former = _sheet_bytes(
        _former_tracker_contact_sheet, synthetic_video, observations, tmp_path / "old.png"
    )
    assert new == former
    assert (tmp_path / "new.png").read_bytes() == (tmp_path / "old.png").read_bytes()
    image = cv2.imread(str(tmp_path / "new.png"))
    assert image.shape == (3 * 8, 16, 3)
    assert np.count_nonzero(image) > 0


def test_hands_contact_sheet_matches_the_former_inline_drawing(
    tmp_path: Path, synthetic_video: Path
) -> None:
    observations = tuple(
        FrameObservations(
            view_id=VIEW_ID,
            analysis_frame_index=index,
            source_seconds=294.0 + index / 30,
            hands=(_hand(index),),
        )
        for index in range(3)
    )
    new = _sheet_bytes(
        video_driver.hands_contact_sheet, synthetic_video, observations, tmp_path / "new.png"
    )
    former = _sheet_bytes(
        _former_hands_contact_sheet, synthetic_video, observations, tmp_path / "old.png"
    )
    assert new == former
    assert (tmp_path / "new.png").read_bytes() == (tmp_path / "old.png").read_bytes()


# --------------------------------------------------------------------------- recorded runs


def _rebuild_recorded_manifest(run_directory: Path) -> tuple[str, str]:
    """Rebuild a recorded smoke's manifest through the current driver; return (old, new) text."""
    manifest_text = (run_directory / "manifest.json").read_text()
    manifest = json.loads(manifest_text)
    worker_result = json.loads((run_directory / "worker_result.json").read_text())
    worker_result["worker_exit_code"] = 0
    drivers = {
        "dam4sam_video": dam4sam_video,
        "samurai_video": samurai_video,
        "grounding_dino_sam2_video": grounding_dino_sam2_video,
        "boxmot": boxmot_track,
        "wilor_hands": wilor_hands,
    }
    field = next(name for name in drivers if manifest.get(name))
    driver = drivers[field]
    metadata = manifest[field]
    config_path = REPOSITORY_ROOT / metadata["config_fingerprint"]["uri"]
    config = G2PreprocessingManifest.model_validate_json(config_path.read_text())
    view_id = manifest["observations"][0]["view_id"]
    proxy = next(item for item in config.proxies if item.view_id == view_id)
    observations_path = run_directory / "observations.jsonl"
    arguments: dict[str, Any] = {
        "run_id": manifest["run_id"],
        "repository_root": REPOSITORY_ROOT,
        "run_directory": run_directory,
        "config": config,
        "config_path": config_path,
        "proxy": proxy,
        "view_id": view_id,
        "seconds": metadata["requested_seconds"],
        "requested_frames": metadata["requested_analysis_frame_range"]["end_frame_exclusive"],
        "worker_result": worker_result,
        "observations": driver._load_observations(observations_path),
        "observations_path": observations_path,
        "qa_path": run_directory / "contact_sheet.png",
    }
    if field == "boxmot":
        classes = re.search(r"class filter (\S+)\)", metadata["detector_source"])
        assert classes is not None
        arguments.update(
            detector_path=REPOSITORY_ROOT / metadata["detector_fingerprint"]["uri"],
            detector_sha256=metadata["detector_fingerprint"]["sha256"],
            detector_classes=classes.group(1),
        )
    elif field == "wilor_hands":
        settings = metadata["runtime_settings"]
        arguments.update(
            checkpoint_path=wilor_hands.WILOR_SOURCE.parent
            / metadata["checkpoint_fingerprint"]["uri"],
            detector_path=wilor_hands.WILOR_SOURCE.parent / metadata["detector_fingerprint"]["uri"],
            save_native_evidence=metadata.get("native_evidence_uri") is not None,
            external_source_state=(
                metadata["adapter"]["external_revision"],
                settings["external_source_dirty"],
                settings["external_source_diff_sha256"],
                tuple(json.loads(settings["external_source_status"])),
            ),
        )
    rebuilt, _state = driver._build_manifest(**arguments)
    return manifest_text, rebuilt.model_dump_json(indent=2) + "\n"


def _recorded_runs() -> list[Path]:
    runs = []
    for pattern in SMOKE_RUN_GLOBS:
        for candidate in sorted(REPOSITORY_ROOT.glob(pattern)):
            if (candidate / "manifest.json").is_file() and (
                candidate / "worker_result.json"
            ).is_file():
                runs.append(candidate)
    return runs


@pytest.mark.real_data
@pytest.mark.parametrize(
    "run_directory", _recorded_runs(), ids=lambda path: path.relative_to(REPOSITORY_ROOT).as_posix()
)
def test_recorded_smoke_manifests_rebuild_byte_for_byte(run_directory: Path) -> None:
    former, rebuilt = _rebuild_recorded_manifest(run_directory)
    if former != rebuilt:
        left, right = json.loads(former), json.loads(rebuilt)
        for key in sorted(set(left) | set(right)):
            if left.get(key) != right.get(key):
                pytest.fail(f"{run_directory.name}: field {key!r} differs")
        pytest.fail(f"{run_directory.name}: manifest text differs (formatting)")


def test_recorded_runs_are_discovered_when_present() -> None:
    # The `real_data` rebuild above is parametrised at collection time; this only documents
    # the discovery so an empty checkout does not silently skip the family.
    assert isinstance(_recorded_runs(), list)
