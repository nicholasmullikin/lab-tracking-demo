"""Pure-function checks for the SAM3 worker's correction memory-bank flags.

Nothing here loads a model or touches CUDA: the worker helpers are exercised on plain
deques and namespaces, and the driver's plumbing is checked by capturing the worker command
it would have launched.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import deque
from pathlib import Path

import pytest

from battle import muggled_smoke
from battle import muggled_worker as worker
from battle.muggled_smoke import (
    _run_worker,
    correction_memory_settings_from_args,
    filter_dropped_correction_frames,
    run_condition_suffix,
)
from battle.schemas import (
    ArtifactFingerprint,
    CorrectionMemorySettings,
    MultiKeyframeCorrectionScheduleMetadata,
    StreamContinuityPolicy,
    TrackerMemoryPolicy,
)

TODAY = "replace_prompt_memory_and_reset_frame_memory"
COMBINATIONS = (
    ("replace", False, "replace_prompt_memory_and_reset_frame_memory"),
    ("append", False, "append_prompt_memory_and_reset_frame_memory"),
    ("append", True, "append_prompt_memory_and_keep_frame_memory"),
    ("replace", True, "replace_prompt_memory_and_keep_frame_memory"),
)


def _worker_args(tmp_path: Path, **overrides: object) -> argparse.Namespace:
    video = tmp_path / "input.mp4"
    video.write_bytes(b"proxy")
    model = tmp_path / "sam3.pt"
    model.write_bytes(b"weights")
    defaults = {
        "video": str(video),
        "model": str(model),
        "view_id": "static-c10379",
        "source_offset_seconds": 294.0,
        "analysis_fps": 30.0,
        "max_side_length": 1280,
        "max_frame_memory": 4,
        "preprocessing": "original_bgr",
        "lower_percentile": 1.0,
        "upper_percentile": 99.0,
        "clahe_clip_limit": 2.0,
        "clahe_tile_grid_size": 8,
        "prompt_mode": "manual_seed_multiplexed_keyframes",
        "text_targets_json": None,
        "hybrid_initialization_json": None,
        "manual_box_json": None,
        "manual_seeds_json": None,
        "multi_keyframe_schedule_json": None,
        "resume_from_checkpoint": None,
    }
    return argparse.Namespace(**{**defaults, **overrides})


# --- semantics strings -----------------------------------------------------------------


@pytest.mark.parametrize(("semantics", "keep", "expected"), COMBINATIONS)
def test_the_four_semantics_strings_are_exact(semantics: str, keep: bool, expected: str) -> None:
    assert (
        worker.correction_memory_semantics(
            prompt_memory_semantics=semantics, keep_frame_memory_at_correction=keep
        )
        == expected
    )
    settings = CorrectionMemorySettings(
        prompt_memory_semantics=semantics,
        max_prompt_memory_entries=CorrectionMemorySettings.default_prompt_memory_entries(semantics),
        keep_frame_memory_at_correction=keep,
    )
    assert settings.correction_memory_semantics == expected, (
        "the driver's schema and the worker must name the same condition"
    )


def test_unknown_prompt_memory_semantics_are_refused() -> None:
    with pytest.raises(ValueError, match="unknown prompt memory semantics"):
        worker.correction_memory_semantics(
            prompt_memory_semantics="merge", keep_frame_memory_at_correction=False
        )
    with pytest.raises(ValueError, match="unknown prompt memory semantics"):
        worker.default_max_prompt_memory("merge")


# --- defaults and settings record -------------------------------------------------------


def test_default_prompt_memory_is_one_for_replace_and_thirty_two_for_append() -> None:
    assert worker.default_max_prompt_memory("replace") == 1
    assert worker.default_max_prompt_memory("append") == 32
    assert CorrectionMemorySettings.default_prompt_memory_entries("replace") == 1
    assert CorrectionMemorySettings.default_prompt_memory_entries("append") == 32


def test_a_namespace_without_the_flags_reads_as_the_pre_flag_condition() -> None:
    settings = worker.memory_settings_from_args(argparse.Namespace(max_frame_memory=4))

    assert settings == {
        "prompt_memory_semantics": "replace",
        "max_prompt_memory": 1,
        "max_frame_memory": 4,
        "keep_frame_memory_at_correction": False,
        "is_recent_first": False,
    }
    assert worker.memory_settings_are_default(settings)
    assert worker.IS_RECENT_FIRST is False, "the worker never passed is_recent_first before"


def test_append_defaults_to_a_thirty_two_entry_bank_and_replace_requires_one() -> None:
    appended = worker.memory_settings_from_args(
        argparse.Namespace(prompt_memory_semantics="append", max_prompt_memory=None)
    )
    assert appended["max_prompt_memory"] == 32
    assert not worker.memory_settings_are_default(appended)

    smaller = worker.memory_settings_from_args(
        argparse.Namespace(prompt_memory_semantics="append", max_prompt_memory=8)
    )
    assert smaller["max_prompt_memory"] == 8

    with pytest.raises(ValueError, match="exactly one prompt memory entry"):
        worker.memory_settings_from_args(
            argparse.Namespace(prompt_memory_semantics="replace", max_prompt_memory=5)
        )
    with pytest.raises(ValueError, match="at least 1"):
        worker.memory_settings_from_args(
            argparse.Namespace(prompt_memory_semantics="append", max_prompt_memory=0)
        )
    with pytest.raises(ValueError, match="exactly one prompt memory entry"):
        CorrectionMemorySettings(prompt_memory_semantics="replace", max_prompt_memory_entries=32)


def test_each_flag_alone_leaves_the_default_condition() -> None:
    for override in (
        {"prompt_memory_semantics": "append"},
        {"keep_frame_memory_at_correction": True},
        {"recent_first": True},
    ):
        settings = worker.memory_settings_from_args(argparse.Namespace(**override))
        assert not worker.memory_settings_are_default(settings), override


# --- frame-memory position encoding provenance ------------------------------------------


@pytest.mark.parametrize(
    ("entries", "expected"),
    ((1, "within_trained_range"), (4, "within_trained_range"), (6, "within_trained_range")),
)
def test_frame_memory_up_to_six_is_within_the_trained_range(entries: int, expected: str) -> None:
    assert worker.frame_memory_position_encoding(entries) == expected
    assert (
        CorrectionMemorySettings(
            max_prompt_memory_entries=1, max_frame_memory_entries=entries
        ).frame_memory_position_encoding
        == expected
    )


@pytest.mark.parametrize("entries", (7, 8, 16))
def test_frame_memory_beyond_six_is_tagged_as_clamped(entries: int) -> None:
    assert worker.frame_memory_position_encoding(entries) == "clamped_beyond_6"
    assert (
        CorrectionMemorySettings(
            max_prompt_memory_entries=1, max_frame_memory_entries=entries
        ).frame_memory_position_encoding
        == "clamped_beyond_6"
    )


def test_runtime_settings_record_the_memory_condition(tmp_path: Path) -> None:
    default = worker._runtime_settings(_worker_args(tmp_path), ("chassis",))

    assert default["correction_memory_semantics"] == TODAY
    assert default["max_prompt_memory_entries"] == 1
    assert default["max_frame_memory_entries"] == 4
    assert default["prompt_memory_semantics"] == "replace"
    assert default["keep_frame_memory_at_correction"] is False
    assert default["is_recent_first"] is False
    assert default["frame_memory_position_encoding"] == "within_trained_range"

    arm = worker._runtime_settings(
        _worker_args(
            tmp_path,
            prompt_memory_semantics="append",
            max_prompt_memory=None,
            keep_frame_memory_at_correction=True,
            recent_first=True,
            max_frame_memory=8,
        ),
        ("chassis",),
    )
    assert arm["correction_memory_semantics"] == "append_prompt_memory_and_keep_frame_memory"
    assert arm["max_prompt_memory_entries"] == 32
    assert arm["max_frame_memory_entries"] == 8
    assert arm["is_recent_first"] is True
    assert arm["frame_memory_position_encoding"] == "clamped_beyond_6"
    assert "reverses the temporal position encoding" in arm["is_recent_first_semantics"]


def test_a_schedule_payload_does_not_override_the_flag_derived_semantics(tmp_path: Path) -> None:
    schedule = {
        "seeds": [{"target": "chassis", "initial_multiplex_slot": 0}],
        "corrections": [],
        "memory_semantics": "append_prompt_memory_and_reset_frame_memory",
    }
    settings = worker._runtime_settings(
        _worker_args(
            tmp_path,
            multi_keyframe_schedule_json=json.dumps(schedule),
            prompt_memory_semantics="append",
            max_prompt_memory=None,
        ),
        ("chassis",),
    )
    assert settings["correction_memory_semantics"] == "append_prompt_memory_and_reset_frame_memory"
    assert settings["multi_keyframe_correction_schedule"]["memory_semantics"] == (
        "append_prompt_memory_and_reset_frame_memory"
    )


# --- deque construction ---------------------------------------------------------------


def test_replace_builds_a_one_entry_prompt_bank_and_append_a_thirty_two_entry_bank() -> None:
    prompts, frames = worker.build_memory_banks(
        prompt_memories=["seed"], frame_memories=[], max_prompt_memory=1, max_frame_memory=4
    )
    assert prompts.maxlen == 1 and frames.maxlen == 4
    assert list(prompts) == ["seed"] and not frames

    prompts, frames = worker.build_memory_banks(
        prompt_memories=["seed"], frame_memories=[], max_prompt_memory=32, max_frame_memory=6
    )
    assert prompts.maxlen == 32 and frames.maxlen == 6


def test_the_prompt_bank_evicts_its_oldest_entry_when_full() -> None:
    prompts, _ = worker.build_memory_banks(
        prompt_memories=["seed"], frame_memories=[], max_prompt_memory=3, max_frame_memory=4
    )
    for name in ("c1", "c2", "c3"):
        worker.apply_correction_to_memory(
            prompt_memories=prompts,
            frame_memories=deque(),
            correction_memory=name,
            prompt_memory_semantics="append",
        )
    assert list(prompts) == ["c1", "c2", "c3"], "the seed is the first to fall off"


def test_resumed_banks_keep_the_checkpointed_entries_in_order() -> None:
    prompts, frames = worker.build_memory_banks(
        prompt_memories=["seed", "c327"],
        frame_memories=["f1", "f2", "f3"],
        max_prompt_memory=32,
        max_frame_memory=4,
    )
    assert list(prompts) == ["seed", "c327"]
    assert list(frames) == ["f1", "f2", "f3"]


# --- correction behaviour -------------------------------------------------------------


def _banks() -> tuple[deque[str], deque[str]]:
    return deque(["seed"], maxlen=32), deque(["older", "latest"], maxlen=4)


def test_replace_and_reset_is_todays_behaviour() -> None:
    prompts, frames = _banks()
    worker.apply_correction_to_memory(
        prompt_memories=prompts, frame_memories=frames, correction_memory="c900"
    )
    assert list(prompts) == ["c900"]
    assert not frames


def test_append_and_reset_keeps_the_seed_and_clears_frame_memory() -> None:
    prompts, frames = _banks()
    worker.apply_correction_to_memory(
        prompt_memories=prompts,
        frame_memories=frames,
        correction_memory="c900",
        prompt_memory_semantics="append",
    )
    assert list(prompts) == ["seed", "c900"]
    assert not frames


def test_append_and_keep_touches_nothing_but_the_prompt_bank() -> None:
    prompts, frames = _banks()
    worker.apply_correction_to_memory(
        prompt_memories=prompts,
        frame_memories=frames,
        correction_memory="c900",
        prompt_memory_semantics="append",
        keep_frame_memory_at_correction=True,
    )
    assert list(prompts) == ["seed", "c900"]
    assert list(frames) == ["older", "latest"]


def test_replace_and_keep_swaps_the_prompt_but_keeps_frame_memory() -> None:
    prompts, frames = deque(["seed"], maxlen=1), deque(["older", "latest"], maxlen=4)
    worker.apply_correction_to_memory(
        prompt_memories=prompts,
        frame_memories=frames,
        correction_memory="c900",
        keep_frame_memory_at_correction=True,
    )
    assert list(prompts) == ["c900"]
    assert list(frames) == ["older", "latest"]


def test_correction_orchestration_honours_append_semantics() -> None:
    numpy = pytest.importorskip("numpy")
    prompt_memories = deque(["seed"], maxlen=32)
    frame_memories = deque(["older", "latest"], maxlen=4)
    predicted = numpy.zeros((2, 2, 3), dtype=bool)
    corrected = numpy.ones((2, 3), dtype=bool)

    rebased = worker._replace_prompt_memory_for_correction(
        predicted_source_masks=predicted,
        correction_masks_by_slot={1: corrected},
        prompt_memories=prompt_memories,
        frame_memories=frame_memories,
        encoded_frame="encoded",
        encode_prompt_memory_from_mask=lambda frame, masks: f"prompt-from-{frame}",
        prompt_memory_semantics="append",
        keep_frame_memory_at_correction=True,
    )

    assert rebased[1].all() and not rebased[0].any()
    assert list(prompt_memories) == ["seed", "prompt-from-encoded"]
    assert list(frame_memories) == ["older", "latest"]


def test_correction_orchestration_default_is_unchanged() -> None:
    numpy = pytest.importorskip("numpy")
    prompt_memories = deque(["seed"], maxlen=1)
    frame_memories = deque(["older"], maxlen=4)

    worker._replace_prompt_memory_for_correction(
        predicted_source_masks=numpy.zeros((1, 2, 2), dtype=bool),
        correction_masks_by_slot={0: numpy.ones((2, 2), dtype=bool)},
        prompt_memories=prompt_memories,
        frame_memories=frame_memories,
        encoded_frame="encoded",
        encode_prompt_memory_from_mask=lambda frame, masks: "replacement",
    )

    assert list(prompt_memories) == ["replacement"]
    assert not frame_memories


def test_the_worker_refuses_a_payload_declaring_other_semantics() -> None:
    payload = {
        "memory_semantics": TODAY,
        "seeds": [{"target": "chassis", "initial_multiplex_slot": 0}],
        "corrections": [{"frame_index": 900, "multiplex_slot": 0, "target": "chassis"}],
    }
    assert worker._corrections_by_frame(payload, ("chassis",)) == {900: payload["corrections"]}
    assert worker._corrections_by_frame(payload, ("chassis",), memory_semantics=TODAY)

    with pytest.raises(ValueError, match="configured for 'append_prompt_memory_and_reset"):
        worker._corrections_by_frame(
            payload,
            ("chassis",),
            memory_semantics="append_prompt_memory_and_reset_frame_memory",
        )


# --- checkpoint compatibility ---------------------------------------------------------


def _legacy_checkpoint() -> dict[str, object]:
    """What a checkpoint written before the memory flags existed holds."""
    return {"max_prompt_memory": 1, "max_frame_memory": 4}


def _default_settings(**overrides: object) -> dict[str, object]:
    return worker.memory_settings_from_args(argparse.Namespace(max_frame_memory=4, **overrides))


def test_a_legacy_checkpoint_resumes_into_a_default_run() -> None:
    worker.check_checkpoint_memory_settings(_legacy_checkpoint(), _default_settings())


def test_a_legacy_checkpoint_refuses_every_memory_arm() -> None:
    for overrides, key in (
        ({"prompt_memory_semantics": "append"}, "prompt_memory_semantics"),
        ({"keep_frame_memory_at_correction": True}, "keep_frame_memory_at_correction"),
        ({"recent_first": True}, "is_recent_first"),
    ):
        with pytest.raises(ValueError, match=f"written with {key}="):
            worker.check_checkpoint_memory_settings(
                _legacy_checkpoint(), _default_settings(**overrides)
            )
    with pytest.raises(
        ValueError, match="holds 4 frame-memory entries; this run is configured for 6"
    ):
        worker.check_checkpoint_memory_settings(
            _legacy_checkpoint(),
            worker.memory_settings_from_args(argparse.Namespace(max_frame_memory=6)),
        )


def test_a_memory_arm_checkpoint_resumes_only_into_the_same_arm() -> None:
    arm = _default_settings(prompt_memory_semantics="append", keep_frame_memory_at_correction=True)
    checkpoint = {
        **_legacy_checkpoint(),
        **{key: arm[key] for key in worker.LEGACY_CHECKPOINT_MEMORY_SETTINGS},
    }

    worker.check_checkpoint_memory_settings(checkpoint, arm)

    with pytest.raises(ValueError, match="written with prompt_memory_semantics='append'"):
        worker.check_checkpoint_memory_settings(checkpoint, _default_settings())
    with pytest.raises(ValueError, match="written with max_prompt_memory=32"):
        worker.check_checkpoint_memory_settings(
            checkpoint,
            _default_settings(
                prompt_memory_semantics="append",
                max_prompt_memory=8,
                keep_frame_memory_at_correction=True,
            ),
        )
    with pytest.raises(ValueError, match="written with keep_frame_memory_at_correction=True"):
        worker.check_checkpoint_memory_settings(
            checkpoint, _default_settings(prompt_memory_semantics="append")
        )


def test_only_a_non_default_memory_condition_changes_the_stream_identity(tmp_path: Path) -> None:
    baseline = worker.stream_identity(_worker_args(tmp_path), ("chassis",))

    assert (
        worker.stream_identity(
            _worker_args(tmp_path, prompt_memory_semantics="replace", recent_first=False),
            ("chassis",),
        )
        == baseline
    )
    for override in (
        {"prompt_memory_semantics": "append"},
        {"keep_frame_memory_at_correction": True},
        {"recent_first": True},
    ):
        assert worker.stream_identity(_worker_args(tmp_path, **override), ("chassis",)) != baseline


# --- worker command line --------------------------------------------------------------


def _worker_argv(*extra: str) -> list[str]:
    return [
        "--run-directory",
        "/tmp/run",
        "--video",
        "/tmp/in.mp4",
        "--view-id",
        "static-c10379",
        "--source-offset-seconds",
        "294",
        "--model",
        "/tmp/sam3.pt",
        *extra,
    ]


def test_the_worker_parser_defaults_reproduce_the_pre_flag_condition() -> None:
    args = worker.parse_args(_worker_argv())
    assert worker.memory_settings_are_default(worker.memory_settings_from_args(args))
    assert args.max_frame_memory == 4 and args.max_prompt_memory is None


def test_the_worker_parser_accepts_and_validates_the_memory_flags() -> None:
    args = worker.parse_args(
        _worker_argv(
            "--prompt-memory-semantics",
            "append",
            "--keep-frame-memory-at-correction",
            "--recent-first",
            "--max-frame-memory",
            "8",
        )
    )
    settings = worker.memory_settings_from_args(args)
    assert settings == {
        "prompt_memory_semantics": "append",
        "max_prompt_memory": 32,
        "max_frame_memory": 8,
        "keep_frame_memory_at_correction": True,
        "is_recent_first": True,
    }
    with pytest.raises(SystemExit):
        worker.parse_args(_worker_argv("--max-prompt-memory", "8"))
    with pytest.raises(SystemExit):
        worker.parse_args(_worker_argv("--max-frame-memory", "0"))


# --- driver plumbing ------------------------------------------------------------------


def test_driver_settings_are_read_from_absent_or_present_flags() -> None:
    default = correction_memory_settings_from_args(argparse.Namespace(max_frame_memory=4))
    assert default == CorrectionMemorySettings(max_prompt_memory_entries=1)
    assert default.is_default_except_frame_memory()
    assert default.worker_arguments() == []
    assert default.run_id_suffix(reference_frame_memory_entries=4) == ""

    arm = correction_memory_settings_from_args(
        argparse.Namespace(
            max_frame_memory=6,
            prompt_memory_semantics="append",
            max_prompt_memory=None,
            keep_frame_memory_at_correction=True,
            recent_first=True,
        )
    )
    assert arm.max_prompt_memory_entries == 32
    assert arm.correction_memory_semantics == "append_prompt_memory_and_keep_frame_memory"
    assert arm.worker_arguments() == [
        "--prompt-memory-semantics",
        "append",
        "--max-prompt-memory",
        "32",
        "--keep-frame-memory-at-correction",
        "--recent-first",
    ]
    assert (
        arm.run_id_suffix(reference_frame_memory_entries=4) == "fm6-pm-append-keepfm-recent-first"
    )
    assert arm.run_id_suffix() == "pm-append-keepfm-recent-first"


def test_run_condition_suffix_names_the_memory_arms() -> None:
    off = TrackerMemoryPolicy()
    common = dict(max_side_length=1280, reference_side_length=720, reference_frame_memory=4)

    def suffix(**kwargs: object) -> str:
        settings = CorrectionMemorySettings(
            max_prompt_memory_entries=CorrectionMemorySettings.default_prompt_memory_entries(
                str(kwargs.get("prompt_memory_semantics", "replace"))
            ),
            **kwargs,  # type: ignore[arg-type]
        )
        return run_condition_suffix(off, memory_settings=settings, **common)

    assert suffix() == "r1280"
    assert suffix(max_frame_memory_entries=6) == "r1280-fm6"
    assert suffix(max_frame_memory_entries=8) == "r1280-fm8"
    assert suffix(prompt_memory_semantics="append") == "r1280-pm-append"
    assert (
        suffix(prompt_memory_semantics="append", keep_frame_memory_at_correction=True)
        == "r1280-pm-append-keepfm"
    )
    assert (
        suffix(prompt_memory_semantics="append", max_frame_memory_entries=6)
        == "r1280-fm6-pm-append"
    )
    assert suffix(is_recent_first=True) == "r1280-recent-first"
    assert run_condition_suffix(off, dropped_correction_frames=(900,), **common) == "r1280-drop900"
    assert (
        run_condition_suffix(
            off, dropped_correction_frames=(1172, 900), frame_zero_seeds_only=True, **common
        )
        == "r1280-seed0"
    ), "seeds-only already drops every correction"
    # The existing arms keep their names.
    assert run_condition_suffix(off, max_side_length=720, reference_side_length=720) == ""


def test_the_driver_passes_only_non_default_memory_flags_to_the_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        commands.append(list(command))
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="")

    monkeypatch.setattr(muggled_smoke.subprocess, "run", fake_run)
    common = dict(
        external_python=tmp_path / "python",
        worker_path=tmp_path / "worker.py",
        run_directory=tmp_path,
        proxy_path=tmp_path / "in.mp4",
        view_id="static-c10379",
        source_offset_seconds=294.0,
        model_path=tmp_path / "sam3.pt",
        max_frames=1800,
        max_side_length=1280,
        analysis_fps=30.0,
    )

    result = _run_worker(
        max_frame_memory=4,
        memory_settings=CorrectionMemorySettings(max_prompt_memory_entries=1),
        **common,
    )
    assert result["state"] == "failed", "no worker_result.json was written by the fake"
    default_command = commands[-1]
    assert "--prompt-memory-semantics" not in default_command
    assert "--keep-frame-memory-at-correction" not in default_command
    assert "--recent-first" not in default_command
    assert default_command[default_command.index("--max-frame-memory") + 1] == "4"

    _run_worker(
        max_frame_memory=6,
        memory_settings=CorrectionMemorySettings(
            prompt_memory_semantics="append",
            max_prompt_memory_entries=32,
            max_frame_memory_entries=6,
            keep_frame_memory_at_correction=True,
            is_recent_first=True,
        ),
        **common,
    )
    arm_command = commands[-1]
    assert arm_command[arm_command.index("--max-frame-memory") + 1] == "6"
    assert arm_command[arm_command.index("--prompt-memory-semantics") + 1] == "append"
    assert arm_command[arm_command.index("--max-prompt-memory") + 1] == "32"
    assert "--keep-frame-memory-at-correction" in arm_command
    assert "--recent-first" in arm_command

    with pytest.raises(ValueError, match="disagree"):
        _run_worker(
            max_frame_memory=4,
            memory_settings=CorrectionMemorySettings(
                max_prompt_memory_entries=1, max_frame_memory_entries=6
            ),
            **common,
        )


def test_continuity_policy_records_the_prompt_bank_size() -> None:
    policy = StreamContinuityPolicy(
        max_prompt_memory_entries=32, max_frame_memory_entries=6, detected_object_limit=4
    )
    assert policy.max_prompt_memory_entries == 32


# --- drop-correction filter -------------------------------------------------------------


def test_dropping_a_scheduled_correction_frame_is_recorded() -> None:
    assert filter_dropped_correction_frames(
        scheduled_frames=frozenset({327, 900, 1172, 1235}), requested_drops=(900,)
    ) == (900,)
    assert filter_dropped_correction_frames(
        scheduled_frames=frozenset({327, 900, 1172, 1235}), requested_drops=(1172, 900, 900)
    ) == (900, 1172)
    assert (
        filter_dropped_correction_frames(scheduled_frames=frozenset({327}), requested_drops=())
        == ()
    )


def test_dropping_a_frame_the_schedule_does_not_correct_is_an_error() -> None:
    with pytest.raises(ValueError, match=r"does not correct: \[901\]"):
        filter_dropped_correction_frames(
            scheduled_frames=frozenset({327, 900, 1172, 1235}), requested_drops=(900, 901)
        )


def test_schedule_metadata_accepts_the_widened_semantics_and_checks_cli_drops() -> None:
    fingerprint = ArtifactFingerprint(uri="s.json", sha256="a" * 64, source="measured")
    metadata = MultiKeyframeCorrectionScheduleMetadata(
        schedule_fingerprint=fingerprint,
        correction_policy_fingerprint=fingerprint,
        correction_memory_semantics="append_prompt_memory_and_keep_frame_memory",
        scheduled_correction_frame_indices=(327, 1172, 1235),
        dropped_correction_frame_indices=(900, 1800, 2700),
        cli_dropped_correction_frame_indices=(900,),
    )
    assert metadata.cli_dropped_correction_frame_indices == (900,)

    with pytest.raises(ValueError, match="must be listed among the dropped"):
        MultiKeyframeCorrectionScheduleMetadata(
            schedule_fingerprint=fingerprint,
            correction_policy_fingerprint=fingerprint,
            correction_memory_semantics=TODAY,
            dropped_correction_frame_indices=(1800,),
            cli_dropped_correction_frame_indices=(900,),
        )
    with pytest.raises(ValueError):
        MultiKeyframeCorrectionScheduleMetadata(
            schedule_fingerprint=fingerprint,
            correction_policy_fingerprint=fingerprint,
            correction_memory_semantics="merge_prompt_memory",
        )


REFERENCE_SCHEDULE = Path(
    "runs/muggledsam-sam3-four-part-focused-corrections-agent-swap-20260918t000947z/"
    "multi_keyframe_correction_schedule.json"
)
FOCUSED_CONFIG = Path("configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_g2.json")


@pytest.mark.real_data
def test_the_reference_schedule_can_run_without_its_900_correction() -> None:
    from conftest import require_artifact

    from battle.muggled_smoke import _load_multi_keyframe_correction_schedule
    from battle.schemas import G2PreprocessingManifest

    root = Path.cwd()
    require_artifact(REFERENCE_SCHEDULE)
    proxy = next(
        item
        for item in G2PreprocessingManifest.model_validate_json(FOCUSED_CONFIG.read_text()).proxies
        if item.view_id == "static-c10379"
    )
    common = dict(
        schedule_path=(root / REFERENCE_SCHEDULE).resolve(),
        repository_root=root,
        config_path=(root / FOCUSED_CONFIG).resolve(),
        proxy=proxy,
        analysis_fps=30.0,
        max_frame_exclusive=1800,
        drop_out_of_range_corrections=True,
    )

    payload, metadata = _load_multi_keyframe_correction_schedule(
        **common,
        dropped_correction_frames=(900,),
        correction_memory_semantics="append_prompt_memory_and_reset_frame_memory",
    )

    assert metadata.scheduled_correction_frame_indices == (327, 1172, 1235)
    assert metadata.dropped_correction_frame_indices == (900, 1800, 2700)
    assert metadata.cli_dropped_correction_frame_indices == (900,)
    assert metadata.correction_memory_semantics == "append_prompt_memory_and_reset_frame_memory"
    assert payload["memory_semantics"] == "append_prompt_memory_and_reset_frame_memory"
    assert all(item["frame_index"] != 900 for item in payload["corrections"])
    assert {item["frame_index"] for item in payload["corrections"]} == {327, 1172, 1235}

    with pytest.raises(ValueError, match="does not correct"):
        _load_multi_keyframe_correction_schedule(**common, dropped_correction_frames=(901,))
