"""Default-tier checks for `battle-muggled-arms` and the schema hunks behind the FineBio arms.

No GPU and no model: the driver's worker command is read without launching it, the schedule
payload loader is exercised on temporary files, and the schemas are round-tripped.
"""

from __future__ import annotations

import json
from datetime import UTC
from pathlib import Path

import pytest

from battle import muggled_arms
from battle.muggled_arms import (
    load_schedule_payload,
    make_run_id,
    parse_args,
    worker_command,
)
from battle.schemas import (
    ArtifactFingerprint,
    CalibrationFrameReference,
    CorrectionMemorySettings,
    FrameObservations,
    MuggledSAMArmRunManifest,
    MuggledSAMMultiKeyframeCorrection,
    MultiKeyframeCorrectionScheduleMetadata,
    NormalizedBox,
    PerFrameObject,
    TrackerMemoryPolicy,
    TrackerSlotDiagnostic,
)

APPEND = "append_prompt_memory_and_reset_frame_memory"


def _files(tmp_path: Path) -> tuple[Path, Path, Path]:
    video = tmp_path / "P03_01_01.mp4"
    video.write_bytes(b"video")
    model = tmp_path / "sam3.pt"
    model.write_bytes(b"weights")
    stream = tmp_path / "boxes.jsonl"
    stream.write_text(
        json.dumps(
            {
                "frame_index": 0,
                "boxes": [
                    {"slot": 0, "label": "plate", "box_xyxy_px": [1, 2, 3, 4]},
                    {"slot": 1, "label": "tube", "box_xyxy_px": [5, 6, 7, 8]},
                ],
            }
        )
        + "\n"
    )
    return video, model, stream


# --- worker command ------------------------------------------------------------------------


def test_box_decode_command_names_the_stream_the_start_frame_and_no_memory(tmp_path: Path) -> None:
    video, model, stream = _files(tmp_path)
    args = parse_args(
        [
            "box-decode",
            "--video",
            str(video),
            "--view-id",
            "fpv",
            "--box-stream",
            str(stream),
            "--start-frame",
            "1798",
            "--max-frames",
            "20",
            "--model",
            str(model),
            "--allow-gpu-neighbour",
            "4242",
        ]
    )
    assert args.mode == "box_decode"
    assert args.source_offset_seconds == pytest.approx(1798 * 1001 / 30000)

    command = worker_command(
        args,
        run_directory=tmp_path / "run",
        concepts=("plate", "tube"),
        schedule_payload=None,
        memory_policy=TrackerMemoryPolicy(),
        memory_settings=CorrectionMemorySettings(max_prompt_memory_entries=1),
    )

    assert command[0] == str(muggled_arms.MUGGLED_SAM_PYTHON)
    assert command[1].endswith("muggled_worker.py")
    flags = dict(zip(command[2::2], command[3::2], strict=False))
    assert flags["--prompt-mode"] == "box_stream"
    assert flags["--box-stream"] == str(stream.resolve())
    assert flags["--start-frame"] == "1798"
    assert flags["--max-frames"] == "20"
    assert flags["--max-side-length"] == "1280"
    assert json.loads(flags["--concepts-json"]) == ["plate", "tube"]
    assert flags["--allow-gpu-neighbour"] == "4242"
    assert flags["--gpu-guard-profile"] == "sam3_1280"
    assert "--other-slots-as-negatives" not in command
    assert "--max-frame-memory" not in command and "--checkpoint-every" not in command
    assert "--memory-write-min-score" not in command


def test_box_decode_negatives_flag_passes_through(tmp_path: Path) -> None:
    video, model, stream = _files(tmp_path)
    args = parse_args(
        [
            "box-decode",
            "--video",
            str(video),
            "--view-id",
            "T2",
            "--box-stream",
            str(stream),
            "--other-slots-as-negatives",
            "--model",
            str(model),
        ]
    )
    command = worker_command(
        args,
        run_directory=tmp_path / "run",
        concepts=("plate",),
        schedule_payload=None,
        memory_policy=TrackerMemoryPolicy(),
        memory_settings=CorrectionMemorySettings(max_prompt_memory_entries=1),
    )
    assert command[-1] == "--other-slots-as-negatives"
    assert args.source_offset_seconds == 0.0


def test_video_memory_command_carries_the_payload_the_memory_arm_and_tau(tmp_path: Path) -> None:
    video, model, _ = _files(tmp_path)
    args = parse_args(
        [
            "video-memory",
            "--video",
            str(video),
            "--view-id",
            "fpv",
            "--schedule",
            str(tmp_path / "schedule.json"),
            "--prompt-memory-semantics",
            "append",
            "--memory-write-min-score",
            "0.5",
            "--checkpoint-every",
            "300",
            "--model",
            str(model),
        ]
    )
    policy = muggled_arms.memory_policy_from_args(args)
    settings = muggled_arms.correction_memory_settings_from_args(args)
    payload = {"memory_semantics": APPEND, "seeds": [], "corrections": []}

    command = worker_command(
        args,
        run_directory=tmp_path / "run",
        concepts=("plate",),
        schedule_payload=payload,
        memory_policy=policy,
        memory_settings=settings,
    )

    flags = dict(zip(command[2::2], command[3::2], strict=False))
    assert flags["--prompt-mode"] == "manual_seed_multiplexed_keyframes"
    assert json.loads(flags["--multi-keyframe-schedule-json"]) == payload
    assert flags["--prompt-memory-semantics"] == "append"
    assert flags["--max-prompt-memory"] == "32"
    assert flags["--memory-write-min-score"] == "0.5"
    assert flags["--memory-gate"] == "off"
    assert flags["--checkpoint-every"] == "300"
    assert flags["--max-frame-memory"] == "4"
    assert policy.run_id_suffix() == "tau0p5"
    assert settings.run_id_suffix() == "pm-append"


def test_run_ids_name_the_mode_the_view_and_the_condition() -> None:
    from datetime import datetime

    now = datetime(2026, 9, 24, 23, 0, 0, tzinfo=UTC)
    assert make_run_id("box_decode", "fpv", now) == "muggledsam-arm-box-decode-fpv-20260924t230000z"
    assert (
        make_run_id("video_memory", "T2", now, suffix="r1280-tau0p5-pm-append")
        == "muggledsam-arm-video-memory-t2-20260924t230000z-r1280-tau0p5-pm-append"
    )


# --- schedule payload loader ---------------------------------------------------------------


def _write_schedule(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "schedule.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_schedule_payload_is_normalised_hashed_and_summarised(tmp_path: Path) -> None:
    mask = tmp_path / "seed_plate.png"
    mask.write_bytes(b"\x89PNG not really")
    payload = {
        "seeds": [
            {
                "target": "tube",
                "initial_multiplex_slot": 1,
                "start_frame": 5,
                "prompt_box_xyxy_px": [1, 2, 3, 4],
                "selected_by": "detector_reseed",
            },
            {"target": "plate", "initial_multiplex_slot": 0, "mask_path": "seed_plate.png"},
        ],
        "corrections": [
            {
                "frame_index": 10,
                "multiplex_slot": 0,
                "target": "plate",
                "prompt_box": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2},
                "selected_by": "track_reproject",
            },
            {
                "frame_index": 7,
                "multiplex_slot": 0,
                "target": "plate",
                "prompt_box_xyxy_px": [1, 2, 3, 4],
                "selected_by": "detector_reseed",
            },
        ],
    }
    schedule = _write_schedule(tmp_path, payload)

    normalised, concepts, metadata = load_schedule_payload(
        schedule, effective_memory_semantics=APPEND
    )

    assert concepts == ("plate", "tube")
    assert normalised["memory_semantics"] == APPEND
    assert [seed["initial_multiplex_slot"] for seed in normalised["seeds"]] == [0, 1]
    assert normalised["seeds"][0]["mask_path"] == str(mask.resolve())
    assert len(normalised["seeds"][0]["mask_sha256"]) == 64
    assert [item["frame_index"] for item in normalised["corrections"]] == [7, 10]
    assert metadata.scheduled_correction_frame_indices == (5, 7, 10)
    assert metadata.detector_reseed_correction_frame_indices == (5, 7)
    assert metadata.track_reproject_correction_frame_indices == (10,)
    assert metadata.slot_start_frames == ((1, 5),)
    assert metadata.correction_memory_semantics == APPEND
    assert metadata.schedule_fingerprint.sha256 == metadata.correction_policy_fingerprint.sha256


def test_schedule_payload_refuses_other_semantics_missing_masks_and_bad_slots(
    tmp_path: Path,
) -> None:
    base = {"seeds": [{"target": "plate", "initial_multiplex_slot": 0}], "corrections": []}
    schedule = _write_schedule(tmp_path, {**base, "memory_semantics": APPEND})
    with pytest.raises(ValueError, match="configured for"):
        load_schedule_payload(
            schedule, effective_memory_semantics="replace_prompt_memory_and_reset_frame_memory"
        )
    schedule = _write_schedule(
        tmp_path,
        {"seeds": [{"target": "plate", "initial_multiplex_slot": 0, "mask_path": "no.png"}]},
    )
    with pytest.raises(ValueError, match="does not exist"):
        load_schedule_payload(schedule, effective_memory_semantics=APPEND)
    schedule = _write_schedule(
        tmp_path, {"seeds": [{"target": "plate", "initial_multiplex_slot": 1}]}
    )
    with pytest.raises(ValueError, match="slots 0..N-1"):
        load_schedule_payload(schedule, effective_memory_semantics=APPEND)
    schedule = _write_schedule(
        tmp_path,
        {
            "seeds": [
                {"target": "plate", "initial_multiplex_slot": 0},
                {"target": "plate", "initial_multiplex_slot": 1},
            ]
        },
    )
    with pytest.raises(ValueError, match="distinct"):
        load_schedule_payload(schedule, effective_memory_semantics=APPEND)
    mask = tmp_path / "m.png"
    mask.write_bytes(b"png")
    schedule = _write_schedule(
        tmp_path,
        {
            "seeds": [
                {
                    "target": "plate",
                    "initial_multiplex_slot": 0,
                    "mask_path": "m.png",
                    "mask_sha256": "0" * 64,
                }
            ]
        },
    )
    with pytest.raises(ValueError, match="SHA-256 changed"):
        load_schedule_payload(schedule, effective_memory_semantics=APPEND)


def test_parse_args_rejects_impossible_windows(tmp_path: Path) -> None:
    video, model, stream = _files(tmp_path)
    with pytest.raises(SystemExit):
        parse_args(
            [
                "box-decode",
                "--video",
                str(video),
                "--view-id",
                "fpv",
                "--box-stream",
                str(stream),
                "--max-frames",
                "0",
            ]
        )


# --- schemas ---------------------------------------------------------------------------------


def _frame(index: int) -> CalibrationFrameReference:
    seconds = index / 30.0
    return CalibrationFrameReference(
        analysis_frame_index=index,
        proxy_seconds=seconds,
        analysis_seconds=seconds,
        source_seconds=60.0 + seconds,
    )


def test_a_box_correction_carries_its_provenance_and_no_mask() -> None:
    box = NormalizedBox(x=0.1, y=0.2, width=0.3, height=0.4)
    for kind in ("detector_reseed", "track_reproject"):
        correction = MuggledSAMMultiKeyframeCorrection(
            selected_by=kind,
            target_id="plate",
            object_id="sam3-00",
            multiplex_slot=0,
            frame=_frame(12),
            prompt_box=box,
        )
        assert correction.is_box_prompt
        again = MuggledSAMMultiKeyframeCorrection.model_validate_json(correction.model_dump_json())
        assert again == correction
    with pytest.raises(ValueError, match="detector_reseed or track_reproject"):
        MuggledSAMMultiKeyframeCorrection(
            selected_by="human",
            target_id="plate",
            object_id="sam3-00",
            multiplex_slot=0,
            frame=_frame(12),
            prompt_box=box,
        )
    with pytest.raises(ValueError, match="no calibration candidate or mask"):
        MuggledSAMMultiKeyframeCorrection(
            selected_by="detector_reseed",
            candidate_id="t000012-b01",
            target_id="plate",
            object_id="sam3-00",
            multiplex_slot=0,
            frame=_frame(12),
            prompt_box=box,
        )


def test_mask_corrections_keep_their_contract() -> None:
    fingerprint = ArtifactFingerprint(uri="masks/m.png", sha256="a" * 64, source="measured")
    correction = MuggledSAMMultiKeyframeCorrection(
        candidate_id="t000012-b01",
        human_selected_candidate_index=2,
        target_id="plate",
        object_id="sam3-00",
        multiplex_slot=0,
        frame=_frame(12),
        calibration_mask_fingerprint=fingerprint,
    )
    assert not correction.is_box_prompt and correction.selected_by == "human"
    legacy = json.loads(correction.model_dump_json())
    legacy.pop("prompt_box")
    assert MuggledSAMMultiKeyframeCorrection.model_validate(legacy) == correction
    with pytest.raises(ValueError, match="names its candidate"):
        MuggledSAMMultiKeyframeCorrection(
            target_id="plate", object_id="sam3-00", multiplex_slot=0, frame=_frame(12)
        )
    with pytest.raises(ValueError, match="human or an agent"):
        MuggledSAMMultiKeyframeCorrection(
            candidate_id="t000012-b01",
            human_selected_candidate_index=2,
            selected_by="detector_reseed",
            target_id="plate",
            object_id="sam3-00",
            multiplex_slot=0,
            frame=_frame(12),
            calibration_mask_fingerprint=fingerprint,
        )
    with pytest.raises(ValueError, match="frame-0 seeds must be human"):
        MuggledSAMMultiKeyframeCorrection(
            candidate_id="t000000-b01",
            human_selected_candidate_index=0,
            selected_by="agent",
            target_id="plate",
            object_id="sam3-00",
            multiplex_slot=0,
            frame=_frame(0),
            calibration_mask_fingerprint=fingerprint,
        )


def test_tau_is_part_of_the_policy_schema_and_its_run_id() -> None:
    assert TrackerMemoryPolicy().is_default
    policy = TrackerMemoryPolicy(memory_write_min_score=0.5)
    assert not policy.is_default
    assert policy.run_id_suffix() == "tau0p5"
    assert policy.worker_arguments()[-2:] == ["--memory-write-min-score", "0.5"]
    assert "--memory-write-min-score" not in TrackerMemoryPolicy().worker_arguments()
    both = TrackerMemoryPolicy(memory_gate="on", memory_write_min_score=-1.5)
    assert both.run_id_suffix() == "gon-taum1p5"
    legacy = json.loads(TrackerMemoryPolicy().model_dump_json())
    legacy.pop("memory_write_min_score")
    assert TrackerMemoryPolicy.model_validate(legacy).is_default


def test_observation_rows_take_the_decoder_provenance_fields() -> None:
    row = FrameObservations.model_validate(
        {
            "schema_version": "1.0",
            "view_id": "fpv",
            "analysis_frame_index": 3,
            "source_seconds": 60.1,
            "objects": [
                {
                    "object_id": "sam3-00",
                    "label": "plate",
                    "confidence": 0.94,
                    "box": {"x": 0.4, "y": 0.3, "width": 0.2, "height": 0.2},
                    "mask": {
                        "uri": "masks/000003_00.png",
                        "storage": "external_artifact",
                        "format": "png",
                    },
                    "object_score": 0.94,
                    "iou_prediction": 0.94,
                    "prompt_box": {"x": 0.45, "y": 0.32, "width": 0.19, "height": 0.17},
                    "source": "sam3_decode",
                    "prompt_source": "finebio_dino",
                    "prompt_score": 0.42,
                }
            ],
            "hands": [],
            "tracker_diagnostics": [
                {
                    "schema_version": "1.0",
                    "object_id": "sam3-01",
                    "label": "tube",
                    "multiplex_slot": 1,
                    "object_score": -2.2,
                    "iou_prediction": None,
                    "active": False,
                    "corrected": False,
                    "memory_written": False,
                    "memory_gate_reason": "unseeded",
                },
                {
                    "schema_version": "1.0",
                    "object_id": "sam3-00",
                    "label": "plate",
                    "multiplex_slot": 0,
                    "object_score": 9.5,
                    "iou_prediction": 0.9,
                    "active": True,
                    "corrected": True,
                    "selected_by": "track_reproject",
                    "prompt_box": {"x": 0.45, "y": 0.32, "width": 0.19, "height": 0.17},
                    "prompt_decoder_iou": 0.91,
                    "decoder_candidate_index": 2,
                    "seed_start": True,
                },
            ],
        }
    )
    assert row.objects[0].source == "sam3_decode"
    assert row.tracker_diagnostics[0].memory_gate_reason == "unseeded"
    assert row.tracker_diagnostics[1].selected_by == "track_reproject"
    plain = PerFrameObject(
        object_id="x", label="y", confidence=0.5, box=NormalizedBox(x=0, y=0, width=1, height=1)
    )
    assert plain.prompt_box is None and plain.source is None
    with pytest.raises(ValueError):
        PerFrameObject(
            object_id="x",
            label="y",
            confidence=0.5,
            box=NormalizedBox(x=0, y=0, width=1, height=1),
            source="tracker",
        )
    with pytest.raises(ValueError):
        TrackerSlotDiagnostic(
            object_id="x",
            label="y",
            multiplex_slot=0,
            object_score=1.0,
            active=True,
            memory_gate_reason="skipped",
        )


def test_schedule_metadata_and_the_arm_manifest_round_trip(tmp_path: Path) -> None:
    fingerprint = ArtifactFingerprint(uri="x", sha256="b" * 64, source="measured")
    metadata = MultiKeyframeCorrectionScheduleMetadata(
        schedule_fingerprint=fingerprint,
        correction_policy_fingerprint=fingerprint,
        correction_memory_semantics=APPEND,
        scheduled_correction_frame_indices=(5, 10),
        detector_reseed_correction_frame_indices=(5,),
        track_reproject_correction_frame_indices=(10,),
        slot_start_frames=((3, 5),),
    )
    legacy = json.loads(metadata.model_dump_json())
    for key in (
        "detector_reseed_correction_frame_indices",
        "track_reproject_correction_frame_indices",
        "slot_start_frames",
    ):
        legacy.pop(key)
    assert MultiKeyframeCorrectionScheduleMetadata.model_validate(legacy).slot_start_frames == ()

    common = dict(
        manifest_kind="muggledsam_sam3_arm_run",
        run_id="muggledsam-arm-video-memory-fpv-20260924t230000z",
        view_id="fpv",
        video_fingerprint=fingerprint,
        prompt_fingerprint=fingerprint,
        model_fingerprint=fingerprint,
        worker_fingerprint=fingerprint,
        adapter={
            "name": "battle.muggled_arms",
            "version": "0.1.0",
            "implementation_basis": "x",
            "external_source_uri": "/x",
        },
        start_frame=1798,
        requested_analysis_frame_range={"start_frame": 0, "end_frame_exclusive": 20},
        analysis_fps=30000 / 1001,
        source_offset_seconds=60.0,
        max_side_length=1280,
        concepts=("plate",),
        method_statuses=[{"method_name": "m", "stage": "objects", "state": "succeeded"}],
        measurements={"elapsed_seconds": 8.9},
        runtime_settings={"mode": "video_memory"},
        observation_rows=20,
        mask_artifact_count=75,
        licence_note=muggled_arms.LICENCE_NOTE,
    )
    manifest = MuggledSAMArmRunManifest(
        mode="video_memory",
        memory_policy=TrackerMemoryPolicy(memory_write_min_score=0.5),
        memory_settings=CorrectionMemorySettings(
            prompt_memory_semantics="append", max_prompt_memory_entries=32
        ),
        multi_keyframe_corrections=metadata,
        stream_identity="c" * 64,
        **common,
    )
    assert MuggledSAMArmRunManifest.model_validate_json(manifest.model_dump_json()) == manifest
    box = MuggledSAMArmRunManifest(mode="box_decode", other_slots_as_negatives=True, **common)
    assert box.memory_policy is None
    with pytest.raises(ValueError, match="no video memory"):
        MuggledSAMArmRunManifest(mode="box_decode", memory_policy=TrackerMemoryPolicy(), **common)
    with pytest.raises(ValueError, match="box_decode setting"):
        MuggledSAMArmRunManifest(mode="video_memory", other_slots_as_negatives=True, **common)
