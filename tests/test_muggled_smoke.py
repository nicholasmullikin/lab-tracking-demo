from __future__ import annotations

import argparse
import json
import types
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

import pytest

from battle.muggled_calibration import build_manifest, finalize_correction_schedule
from battle.muggled_smoke import (
    CONCEPTS,
    E4_CANDIDATE_FRAMES,
    FOUR_PART_FOCUSED_FRAMES,
    FOUR_PART_FULL_FRAMES,
    FOUR_PART_PILOT_FRAMES,
    FULL_EGO_MANUAL_SEED_FRAMES,
    G3_STATIC_FRAMES,
    SMOKE_FRAMES,
    _load_hybrid_initialization,
    _load_hybrid_smoke_approval,
    _load_manual_seed_multiplex,
    _load_multi_keyframe_correction_schedule,
    _qa_method_name,
    _require_correction_frame_in_range,
    load_manual_seed_target_config,
    load_observations,
    load_text_target_config,
    make_run_id,
    require_e4_candidate_range,
    require_four_part_ego_focused_range,
    require_four_part_focused_range,
    require_four_part_full_range,
    require_four_part_pilot_range,
    require_full_ego_manual_seed_range,
    require_g3_static_range,
    require_smoke_range,
    selected_frame_budget,
    sha256_file,
)
from battle.muggled_worker import (
    BOX_COMPONENT_KEEP_FRACTION,
    _box_from_component_stats,
    _box_from_mask,
    _corrections_by_frame,
    _is_numpy_mask,
    _looks_like_model_process,
    _ordered_hybrid_initial_masks,
    _positive_frame_count,
    _replace_prompt_memory_for_correction,
    _validate_hybrid_initialization,
    _validate_manual_seed_slots,
    _validate_text_targets,
)
from battle.schemas import (
    ArtifactFingerprint,
    ChunkContinuityPolicy,
    G2PreprocessingManifest,
    MuggledSAMCalibrationCandidate,
    MuggledSAMEgoConditionConfig,
    MuggledSAMHybridInitializationConfig,
    MuggledSAMProposedTrackingPromptConfig,
    ProposedTrackingSeed,
)


def test_numpy_masks_use_the_opencv_adapter_despite_numpy_device_attribute() -> None:
    numpy = pytest.importorskip("numpy")
    mask = numpy.zeros((2, 2), dtype=bool)

    assert mask.device == "cpu"
    assert _is_numpy_mask(mask)


def test_worker_frame_count_parser_accepts_new_positive_budgets() -> None:
    assert _positive_frame_count("1800") == 1800
    assert _positive_frame_count("2781") == 2781
    with pytest.raises(argparse.ArgumentTypeError, match="must be positive"):
        _positive_frame_count("0")


def test_box_ignores_speckle_components_but_keeps_a_split_object() -> None:
    numpy = pytest.importorskip("numpy")
    # Rows follow cv2.connectedComponentsWithStats: left, top, width, height, area.
    hand = [100, 200, 40, 50, 1600]
    occluded_fingers = [150, 210, 20, 20, 400]  # 25% of the hand: a legitimate split.
    speck = [900, 700, 2, 2, 4]  # far-away speckle that used to balloon the box.
    frame_shape = (720, 954)

    box = _box_from_component_stats(numpy.array([hand, speck, occluded_fingers]), frame_shape)

    assert box == pytest.approx(
        {"x": 100 / 954, "y": 200 / 720, "width": 70 / 954, "height": 50 / 720}
    )
    assert 400 >= BOX_COMPONENT_KEEP_FRACTION * 1600
    assert 4 < BOX_COMPONENT_KEEP_FRACTION * 1600


def test_box_from_mask_covers_only_the_dominant_components() -> None:
    numpy = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    mask = numpy.zeros((72, 96), dtype=bool)
    mask[20:40, 10:30] = True  # 400 px object
    mask[22:32, 32:42] = True  # 100 px fragment separated by a 2 px gap: kept (25%)
    mask[60, 90] = True  # 1 px speck: dropped
    mask[5, 70:74] = True  # 4 px sliver: dropped

    box = _box_from_mask(mask)

    assert box == pytest.approx({"x": 10 / 96, "y": 20 / 72, "width": 32 / 96, "height": 20 / 72})
    assert _box_from_mask(numpy.zeros((72, 96), dtype=bool)) is None


def test_gpu_guard_tolerates_an_open_rerun_viewer_but_not_other_python_processes() -> None:
    def process(name: str) -> dict[str, str]:
        return {"pid": "1", "process_name": name, "memory": "623 MiB"}

    viewer = "/home/nick/src/battle/.venv/lib/python3.12/site-packages/rerun_sdk/rerun_cli/rerun"
    assert not _looks_like_model_process(process(viewer))
    assert not _looks_like_model_process(process("/usr/local/bin/rerun"))
    assert _looks_like_model_process(process("/home/nick/.pyenv/versions/muggled_sam/bin/python"))
    assert _looks_like_model_process(process("/usr/bin/ollama"))
    assert _looks_like_model_process(process("/opt/rerun-experiments/.venv/bin/python3"))


def test_worker_accepts_four_ordered_manual_multiplex_slots() -> None:
    concepts = ("left_hand", "right_hand", "yellow_toy_top", "black_toy_top_base")
    seed_records = [
        {"target": target, "initial_multiplex_slot": slot} for slot, target in enumerate(concepts)
    ]

    _validate_manual_seed_slots(seed_records, concepts)

    with pytest.raises(ValueError, match="one mask for each"):
        _validate_manual_seed_slots(seed_records[:3], concepts)


def test_static_text_target_mapping_preserves_common_output_labels() -> None:
    root = Path(__file__).parents[1]
    config = load_text_target_config(
        target_config_path=root / "configs/muggledsam_static_common_four_target_text.json",
        repository_root=root,
        g2_config_path=root / "configs/clips/assembly101_nusar_9033_g2.json",
        view_id="static-c10379",
    )

    assert tuple(target.output_label for target in config.targets) == (
        "left_hand",
        "right_hand",
        "yellow_toy_top",
        "black_toy_top_base",
    )
    assert tuple(target.text_prompt for target in config.targets) == (
        "left hand",
        "right hand",
        "yellow toy top",
        "black toy top base",
    )


def test_worker_rejects_ambiguous_text_target_mapping() -> None:
    targets = [
        {"output_label": "left_hand", "text_prompt": "hand"},
        {"output_label": "right_hand", "text_prompt": "hand"},
    ]

    with pytest.raises(ValueError, match="prompts must be distinct"):
        _validate_text_targets(targets, ("left_hand", "right_hand"))
    with pytest.raises(ValueError, match="ordered output concepts"):
        _validate_text_targets(targets[:1], ("left_hand", "right_hand"))


def test_correction_orchestration_replaces_prompt_and_resets_frame_history() -> None:
    numpy = pytest.importorskip("numpy")
    prompt_memories = deque(["old-prompt"], maxlen=1)
    frame_memories = deque(["older-frame", "latest-frame"], maxlen=4)
    predicted = numpy.zeros((4, 2, 3), dtype=bool)
    predicted[0, :, :] = True
    corrected = numpy.zeros((2, 3), dtype=bool)
    corrected[1, 2] = True
    calls: list[object] = []

    result = _replace_prompt_memory_for_correction(
        predicted_source_masks=predicted,
        correction_masks_by_slot={2: corrected},
        prompt_memories=prompt_memories,
        frame_memories=frame_memories,
        encoded_frame="encoded-frame",
        encode_prompt_memory_from_mask=lambda frame, masks: (
            calls.append((frame, masks.copy())) or "replacement-prompt"
        ),
    )

    assert result[2].tolist() == corrected.tolist()
    assert result[0].tolist() == predicted[0].tolist()
    assert list(prompt_memories) == ["replacement-prompt"]
    assert not frame_memories
    assert calls[0][0] == "encoded-frame"


def test_worker_rejects_ambiguous_correction_slot_assignments() -> None:
    with pytest.raises(ValueError, match="ambiguous"):
        _corrections_by_frame(
            {
                "memory_semantics": "replace_prompt_memory_and_reset_frame_memory",
                "seeds": [
                    {"target": "left_hand", "initial_multiplex_slot": 0},
                    {"target": "right_hand", "initial_multiplex_slot": 1},
                ],
                "corrections": [
                    {"frame_index": 30, "multiplex_slot": 0, "target": "left_hand"},
                    {"frame_index": 30, "multiplex_slot": 0, "target": "left_hand"},
                ],
            },
            ("left_hand", "right_hand"),
        )


def test_long_run_correction_frames_are_validated_against_the_active_budget() -> None:
    _require_correction_frame_in_range(360, max_frame_exclusive=600)

    with pytest.raises(ValueError, match=r"\[0, 300\)"):
        _require_correction_frame_in_range(360, max_frame_exclusive=300)


@pytest.mark.parametrize(
    ("static_focused", "ego_focused", "expected"),
    (
        (True, False, "four-part-static-focused-reassembly-qa"),
        (False, True, "four-part-ego-focused-reassembly-qa"),
    ),
)
def test_focused_qa_status_keeps_the_selected_view_name(
    static_focused: bool, ego_focused: bool, expected: str
) -> None:
    assert (
        _qa_method_name(
            hybrid=False,
            full_ego=False,
            four_part_pilot=False,
            four_part_full=False,
            four_part_static_focused=static_focused,
            four_part_ego_focused=ego_focused,
        )
        == expected
    )


def test_smoke_range_is_exactly_the_approved_first_ten_seconds() -> None:
    frame_range = require_smoke_range(start_frame=0, max_frames=SMOKE_FRAMES)

    assert frame_range.start_frame == 0
    assert frame_range.end_frame_exclusive == 300
    with pytest.raises(ValueError, match="first 10 seconds"):
        require_smoke_range(start_frame=1, max_frames=SMOKE_FRAMES)
    with pytest.raises(ValueError, match="first 10 seconds"):
        require_smoke_range(start_frame=0, max_frames=299)


def test_smoke_range_tracks_seconds_rather_than_frames_across_analysis_rates() -> None:
    """The approved bound is ten seconds, so its frame count follows the analysis clock."""
    assert require_smoke_range(0, 600, analysis_fps=60.0).end_frame_exclusive == 600

    with pytest.raises(ValueError, match="at 60 fps"):
        require_smoke_range(0, 300, analysis_fps=60.0)
    with pytest.raises(ValueError, match="at 30 fps"):
        require_smoke_range(0, 600, analysis_fps=30.0)


def test_focused_views_share_one_frame_budget_resolver() -> None:
    common = {
        "g3_full_static": False,
        "g4_e4_candidate": False,
        "full_ego_manual_seed": False,
        "four_part_static_pilot": False,
        "four_part_static_full": False,
    }
    static = types.SimpleNamespace(
        **common, four_part_static_focused=True, four_part_ego_focused=False
    )
    ego = types.SimpleNamespace(
        **common, four_part_static_focused=False, four_part_ego_focused=True
    )

    assert selected_frame_budget(static, 30.0) == FOUR_PART_FOCUSED_FRAMES
    assert selected_frame_budget(ego, 30.0) == FOUR_PART_FOCUSED_FRAMES


def test_g3_range_allows_only_the_approved_static_full_proxy() -> None:
    frame_range = require_g3_static_range(
        "static-c10379", start_frame=0, max_frames=G3_STATIC_FRAMES
    )

    assert frame_range.frame_count == 5400
    with pytest.raises(ValueError, match="only static"):
        require_g3_static_range("ego-hmc21110305", start_frame=0, max_frames=G3_STATIC_FRAMES)


def test_e4_candidate_range_allows_only_the_approved_sixty_seconds() -> None:
    frame_range = require_e4_candidate_range(
        "ego-hmc21179183", start_frame=0, max_frames=E4_CANDIDATE_FRAMES
    )

    assert frame_range.frame_count == 1800
    with pytest.raises(ValueError, match="only ego-hmc21179183"):
        require_e4_candidate_range("ego-hmc21110305", start_frame=0, max_frames=E4_CANDIDATE_FRAMES)


def test_four_part_pilot_range_allows_only_the_approved_static_window() -> None:
    frame_range = require_four_part_pilot_range(
        "static-c10379", start_frame=0, max_frames=FOUR_PART_PILOT_FRAMES
    )

    assert frame_range.frame_count == 600
    with pytest.raises(ValueError, match="only static-c10379"):
        require_four_part_pilot_range(
            "ego-hmc21179183", start_frame=0, max_frames=FOUR_PART_PILOT_FRAMES
        )
    with pytest.raises(ValueError, match=r"\[0, 600\)"):
        require_four_part_pilot_range("static-c10379", start_frame=0, max_frames=599)


def test_four_part_full_range_allows_only_the_complete_static_proxy() -> None:
    frame_range = require_four_part_full_range(
        "static-c10379", start_frame=0, max_frames=FOUR_PART_FULL_FRAMES
    )

    assert frame_range.frame_count == 5901
    with pytest.raises(ValueError, match="only static-c10379"):
        require_four_part_full_range(
            "ego-hmc21179183", start_frame=0, max_frames=FOUR_PART_FULL_FRAMES
        )
    with pytest.raises(ValueError, match=r"\[0, 5901\)"):
        require_four_part_full_range("static-c10379", start_frame=0, max_frames=5900)


def test_four_part_focused_range_allows_only_the_separated_to_assembled_proxy() -> None:
    frame_range = require_four_part_focused_range(
        "static-c10379", start_frame=0, max_frames=FOUR_PART_FOCUSED_FRAMES
    )

    assert frame_range.frame_count == 2781
    with pytest.raises(ValueError, match="only static-c10379"):
        require_four_part_focused_range(
            "ego-hmc21179183", start_frame=0, max_frames=FOUR_PART_FOCUSED_FRAMES
        )
    with pytest.raises(ValueError, match=r"\[0, 2781\)"):
        require_four_part_focused_range("static-c10379", start_frame=0, max_frames=2780)


def test_four_part_ego_focused_range_allows_only_the_aligned_monochrome_proxy() -> None:
    frame_range = require_four_part_ego_focused_range(
        "ego-hmc21110305", start_frame=0, max_frames=FOUR_PART_FOCUSED_FRAMES
    )

    assert frame_range.frame_count == 2781
    with pytest.raises(ValueError, match="only ego-hmc21110305"):
        require_four_part_ego_focused_range(
            "static-c10379", start_frame=0, max_frames=FOUR_PART_FOCUSED_FRAMES
        )


def test_full_ego_manual_seed_range_allows_only_the_approved_proxy() -> None:
    frame_range = require_full_ego_manual_seed_range(
        "ego-hmc21179183", start_frame=0, max_frames=FULL_EGO_MANUAL_SEED_FRAMES
    )

    assert frame_range.frame_count == 5400
    with pytest.raises(ValueError, match="only ego-hmc21179183"):
        require_full_ego_manual_seed_range(
            "ego-hmc21110305", start_frame=0, max_frames=FULL_EGO_MANUAL_SEED_FRAMES
        )


def test_streaming_policy_uses_no_chunks() -> None:
    policy = ChunkContinuityPolicy(overlap_seconds=0.0, max_allowed_gap_seconds=0.0)

    assert policy.chunk_duration_seconds is None
    assert policy.preserve_track_ids
    assert policy.carry_context_across_chunks


def test_worker_observations_are_validated_line_by_line(tmp_path: Path) -> None:
    observations_path = tmp_path / "observations.jsonl"
    observations_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "view_id": "static-c10379",
                "analysis_frame_index": 0,
                "source_seconds": 215.0,
                "objects": [
                    {
                        "object_id": "sam3-00",
                        "label": CONCEPTS[0],
                        "confidence": 0.9,
                        "box": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4},
                    }
                ],
                "hands": [],
            }
        )
        + "\n"
    )

    observations = load_observations(observations_path)

    assert len(observations) == 1
    assert observations[0].objects[0].label == "hand"


def test_worker_observation_contract_rejects_unbounded_boxes(tmp_path: Path) -> None:
    observations_path = tmp_path / "observations.jsonl"
    observations_path.write_text(
        '{"view_id":"static-c10379","analysis_frame_index":0,"source_seconds":215,'
        '"objects":[{"object_id":"sam3-00","label":"hand","confidence":0.9,'
        '"box":{"x":0.9,"y":0.2,"width":0.3,"height":0.4}}]}\n'
    )

    with pytest.raises(ValueError, match="invalid observation"):
        load_observations(observations_path)


def test_run_identifier_is_portable_lowercase() -> None:
    run_id = make_run_id(
        "ego-hmc21110305",
        now=datetime(2026, 9, 8, 22, 53, tzinfo=UTC),
    )

    assert run_id == "muggledsam-sam3-smoke-ego-hmc21110305-20260908t225300z"


def test_run_condition_suffix_names_only_what_departs_from_the_reference_condition() -> None:
    from battle.muggled_smoke import run_condition_suffix
    from battle.schemas import TrackerMemoryPolicy

    default = TrackerMemoryPolicy()
    arm = TrackerMemoryPolicy(slot_exclusivity="argmax", memory_gate="on")

    assert run_condition_suffix(default, max_side_length=720, reference_side_length=720) == ""
    assert run_condition_suffix(default, max_side_length=504) == ""
    assert run_condition_suffix(default, max_side_length=1008, reference_side_length=720) == "r1008"
    assert (
        run_condition_suffix(
            arm, max_side_length=1008, reference_side_length=720, frame_zero_seeds_only=True
        )
        == "xargmax-gon-r1008-seed0"
    )
    assert make_run_id(
        "static-c10379",
        now=datetime(2026, 9, 18, 12, 0, tzinfo=UTC),
        profile="four-part-static-focused-reassembly",
        suffix="gon",
    ).endswith("-static-c10379-20260918t120000z-gon")


def test_focused_profile_may_stop_at_the_first_minute() -> None:
    frame_range = require_four_part_focused_range("static-c10379", start_frame=0, max_frames=1800)
    assert frame_range.end_frame_exclusive == 1800
    with pytest.raises(ValueError, match="first-minute bound"):
        require_four_part_focused_range("static-c10379", start_frame=0, max_frames=1500)

    common = {
        "g3_full_static": False,
        "g4_e4_candidate": False,
        "full_ego_manual_seed": False,
        "four_part_static_pilot": False,
        "four_part_static_full": False,
        "four_part_ego_focused": False,
        "four_part_static_focused": True,
    }
    assert selected_frame_budget(types.SimpleNamespace(**common, max_frames=1800), 30.0) == 1800
    # Any other request falls back to the full budget and is refused by the range guard.
    assert selected_frame_budget(types.SimpleNamespace(**common, max_frames=1500), 30.0) == 2781


def test_ego_conditions_are_explicit_and_schema_valid() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "muggledsam_ego_conditions.json"

    config = MuggledSAMEgoConditionConfig.model_validate_json(config_path.read_text())

    assert [condition.display_label for condition in config.conditions] == [
        "CONTRAST-NORMALIZED",
        "MANUAL-SEED",
    ]
    assert config.conditions[0].concepts == ("hand",)
    assert config.conditions[1].manual_box_seed is not None


def test_new_manual_seed_target_config_is_bound_to_e4_and_has_four_distinct_targets() -> None:
    root = Path(__file__).parents[1]
    config = load_manual_seed_target_config(
        target_config_path=(
            root / "configs/"
            "muggledsam_e4_left_hand_right_hand_yellow_toy_top_black_toy_top_base_manual_seed.json"
        ),
        repository_root=root,
        g2_config_path=root / "configs/clips/assembly101_nusar_9033_ego_viewpoint_screen_g2.json",
        view_id="ego-hmc21179183",
    )

    assert config.targets == (
        "left_hand",
        "right_hand",
        "yellow_toy_top",
        "black_toy_top_base",
    )


def test_static_manual_seed_target_config_is_bound_to_one_canonical_label() -> None:
    root = Path(__file__).parents[1]
    config = load_manual_seed_target_config(
        target_config_path=root / "configs/muggledsam_static_black_toy_top_base_manual_seed.json",
        repository_root=root,
        g2_config_path=root / "configs/clips/assembly101_nusar_9033_g2.json",
        view_id="static-c10379",
    )

    assert config.targets == ("black_toy_top_base",)


def _hybrid_worker_payload() -> dict[str, object]:
    return {
        "targets": [
            {
                "output_label": label,
                "initial_multiplex_slot": slot,
                "initialization_source": source,
            }
            for slot, (label, source) in enumerate(
                (
                    ("left_hand", "text_prompt"),
                    ("right_hand", "text_prompt"),
                    ("yellow_toy_top", "text_prompt"),
                    ("black_toy_top_base", "human_reviewed_mask"),
                )
            )
        ],
        "text_targets": [
            {"output_label": "left_hand", "text_prompt": "left hand"},
            {"output_label": "right_hand", "text_prompt": "right hand"},
            {"output_label": "yellow_toy_top", "text_prompt": "yellow toy top"},
        ],
        "manual_seeds": [
            {
                "target": "black_toy_top_base",
                "initial_multiplex_slot": 3,
                "mask_path": "/fixture/mask.png",
                "mask_sha256": "0" * 64,
            }
        ],
    }


@pytest.mark.real_data
def test_static_hybrid_contract_loads_reviewed_mask_with_manifest_provenance() -> None:
    from conftest import require_artifact

    root = Path(__file__).parents[1]
    require_artifact(
        root / "runs/muggledsam-sam3-static-black-toy-top-base-calibration-20260914t022125z"
    )
    g2_path = root / "configs/clips/assembly101_nusar_9033_g2.json"
    g2 = G2PreprocessingManifest.model_validate_json(g2_path.read_text())
    proxy = next(item for item in g2.proxies if item.view_id == "static-c10379")

    payload, metadata, text_config = _load_hybrid_initialization(
        hybrid_config_path=root / "configs/muggledsam_static_aligned_hybrid.json",
        repository_root=root,
        config_path=g2_path,
        proxy=proxy,
    )

    assert [target.output_label for target in text_config.targets] == [
        "left_hand",
        "right_hand",
        "yellow_toy_top",
    ]
    assert [target["initial_multiplex_slot"] for target in payload["targets"]] == [0, 1, 2, 3]
    assert payload["manual_seeds"][0]["initial_multiplex_slot"] == 3
    persisted = metadata.model_dump(mode="json")
    assert [target["initialization_source"] for target in persisted["targets"]] == [
        "text_prompt",
        "text_prompt",
        "text_prompt",
        "human_reviewed_mask",
    ]
    assert (
        persisted["targets"][3]["source_fingerprint"]["sha256"]
        == (payload["manual_seeds"][0]["mask_sha256"])
    )
    assert persisted["ground_truth_accuracy_claim"] is False


@pytest.mark.real_data
def test_full_hybrid_approval_binds_exact_reviewed_smoke_evidence() -> None:
    from conftest import require_artifact

    root = Path(__file__).parents[1]
    require_artifact(
        root / "runs/muggledsam-sam3-static-black-toy-top-base-calibration-20260914t022125z"
    )
    g2_path = root / "configs/clips/assembly101_nusar_9033_g2.json"
    g2 = G2PreprocessingManifest.model_validate_json(g2_path.read_text())
    proxy = next(item for item in g2.proxies if item.view_id == "static-c10379")
    _, metadata, _ = _load_hybrid_initialization(
        hybrid_config_path=root / "configs/muggledsam_static_aligned_hybrid.json",
        repository_root=root,
        config_path=g2_path,
        proxy=proxy,
    )
    smoke_manifest = (
        root
        / "runs/muggledsam-sam3-smoke-hybrid-static-static-c10379-20260915t005256z"
        / "manifest.json"
    )

    approval = _load_hybrid_smoke_approval(
        approved_smoke_manifest_path=smoke_manifest,
        repository_root=root,
        hybrid_metadata=metadata,
        config_path=g2_path,
        proxy=proxy,
        max_side_length=504,
        max_frame_memory=4,
        approved_at=datetime(2026, 9, 15, 0, 53, tzinfo=UTC),
        approved_by="user",
        approval_statement="Looks good!",
    )

    assert approval.approved_smoke_manifest_fingerprint.uri.endswith(
        "20260915t005256z/manifest.json"
    )
    assert approval.approved_smoke_qa_fingerprint.sha256 == (
        "898b5d8057563d1bb0bca4cd82b9d105570fcd072ccde6cf6f585208214a6043"
    )
    with pytest.raises(ValueError, match="settings must exactly match"):
        _load_hybrid_smoke_approval(
            approved_smoke_manifest_path=smoke_manifest,
            repository_root=root,
            hybrid_metadata=metadata,
            config_path=g2_path,
            proxy=proxy,
            max_side_length=720,
            max_frame_memory=4,
            approved_at=datetime(2026, 9, 15, 0, 53, tzinfo=UTC),
            approved_by="user",
            approval_statement="Looks good!",
        )


def test_hybrid_contract_rejects_order_overlap_and_missing_manual_mask() -> None:
    concepts = (
        "left_hand",
        "right_hand",
        "yellow_toy_top",
        "black_toy_top_base",
    )
    valid = _hybrid_worker_payload()
    _validate_hybrid_initialization(valid, concepts)

    reordered = json.loads(json.dumps(valid))
    reordered["targets"][0], reordered["targets"][1] = (
        reordered["targets"][1],
        reordered["targets"][0],
    )
    with pytest.raises(ValueError, match="canonical labels"):
        _validate_hybrid_initialization(reordered, concepts)

    overlap = json.loads(json.dumps(valid))
    overlap["manual_seeds"][0]["target"] = "yellow_toy_top"
    with pytest.raises(ValueError, match="target-to-slot"):
        _validate_hybrid_initialization(overlap, concepts)

    with pytest.raises(ValueError, match="reviewed mask is unavailable"):
        _ordered_hybrid_initial_masks(valid["targets"], {}, {})


def test_hybrid_missing_text_detection_preserves_canonical_slot_identity() -> None:
    payload = _hybrid_worker_payload()
    left_mask, yellow_mask, black_mask = object(), object(), object()

    ordered = _ordered_hybrid_initial_masks(
        payload["targets"],
        {
            "left_hand": (left_mask, 0.8),
            "yellow_toy_top": (yellow_mask, 0.7),
        },
        {"black_toy_top_base": black_mask},
    )

    assert [(slot, label) for slot, label, *_ in ordered] == [
        (0, "left_hand"),
        (2, "yellow_toy_top"),
        (3, "black_toy_top_base"),
    ]
    assert all(label != "right_hand" for _, label, *_ in ordered)


def test_hybrid_loader_rejects_changed_fingerprint_and_view(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    g2_path = root / "configs/clips/assembly101_nusar_9033_g2.json"
    g2 = G2PreprocessingManifest.model_validate_json(g2_path.read_text())
    static_proxy = next(item for item in g2.proxies if item.view_id == "static-c10379")
    hybrid_path = root / "configs/muggledsam_static_aligned_hybrid.json"
    changed = MuggledSAMHybridInitializationConfig.model_validate_json(
        hybrid_path.read_text()
    ).model_copy(
        update={
            "manual_seed_proposal_fingerprint": ArtifactFingerprint(
                uri="missing/proposal.json",
                sha256="0" * 64,
                source="measured",
            )
        }
    )
    changed_path = tmp_path / "changed-hybrid.json"
    changed_path.write_text(changed.model_dump_json(indent=2) + "\n")
    with pytest.raises(ValueError, match="unavailable or changed"):
        _load_hybrid_initialization(
            hybrid_config_path=changed_path,
            repository_root=root,
            config_path=g2_path,
            proxy=static_proxy,
        )

    wrong_proxy = next(item for item in g2.proxies if item.view_id != "static-c10379")
    with pytest.raises(ValueError, match="view must match"):
        _load_hybrid_initialization(
            hybrid_config_path=hybrid_path,
            repository_root=root,
            config_path=g2_path,
            proxy=wrong_proxy,
        )


def test_four_target_manual_seed_initializes_four_ordered_multiplex_slots(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    g2_path = root / "configs/clips/assembly101_nusar_9033_ego_viewpoint_screen_g2.json"
    g2 = G2PreprocessingManifest.model_validate_json(g2_path.read_text())
    proxy = next(item for item in g2.proxies if item.view_id == "ego-hmc21179183")
    targets_and_ids = (
        ("left_hand", "t000000-b47"),
        ("right_hand", "t000000-b48"),
        ("yellow_toy_top", "t000000-b49"),
        ("black_toy_top_base", "t000000-b50"),
    )
    candidates = []
    for target, candidate_id in targets_and_ids:
        mask_uri = f"results/masks/{candidate_id}_candidate-00.png"
        mask_path = tmp_path / mask_uri
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.write_bytes(candidate_id.encode())
        candidates.append(
            MuggledSAMCalibrationCandidate.model_validate(
                {
                    "candidate_id": candidate_id,
                    "intended_target": target,
                    "frame": {
                        "analysis_frame_index": 0,
                        "proxy_seconds": 0.0,
                        "analysis_seconds": 0.0,
                        "source_seconds": 215.0,
                    },
                    "pixel_box": {"x1": 10, "y1": 20, "x2": 100, "y2": 120},
                    "normalized_box": {
                        "x": 10 / 954,
                        "y": 20 / 720,
                        "width": 90 / 954,
                        "height": 100 / 720,
                    },
                    "decoder_result": {
                        "api": "muggledsam_sam3_interactive",
                        "candidate_count": 1,
                        "deterministic_best_candidate_index": 0,
                        "candidates": [
                            {
                                "candidate_index": 0,
                                "iou_score": 0.5,
                                "mask_uri": mask_uri,
                                "is_deterministic_best": True,
                            }
                        ],
                        "overlay_uri": f"results/{candidate_id}_overlay.png",
                    },
                    "human_selected_candidate_index": 0,
                    "human_accepted": True,
                    "selected_for_finalization": True,
                }
            )
        )
    manifest = build_manifest(
        repository_root=root,
        config_path=g2_path,
        timestamps=(0.0,),
        result_directory=tmp_path / "results",
        calibration_id="muggledsam-sam3-e4-box-calibration-targets",
    ).model_copy(update={"candidates": tuple(candidates)})
    manifest_path = tmp_path / "calibration_manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n")
    proposal = MuggledSAMProposedTrackingPromptConfig(
        manifest_kind="muggledsam_sam3_proposed_tracking_prompt",
        authority="proposed_non_authoritative",
        calibration_manifest_uri=str(manifest_path),
        calibration_manifest_sha256=sha256_file(manifest_path),
        view_id="ego-hmc21179183",
        seeds=tuple(
            ProposedTrackingSeed(
                candidate_id=candidate.candidate_id,
                intended_target=candidate.intended_target,
                reference_frame=candidate.frame,
                pixel_box=candidate.pixel_box,
                normalized_box=candidate.normalized_box,
                decoder_best_candidate_index=0,
                human_selected_candidate_index=0,
            )
            for candidate in reversed(candidates)
        ),
        tracker_initialization_limitations=("fixture",),
    )
    proposal_path = tmp_path / "proposal.json"
    proposal_path.write_text(proposal.model_dump_json(indent=2) + "\n")

    payload, metadata = _load_manual_seed_multiplex(
        proposal_path=proposal_path,
        repository_root=root,
        config_path=g2_path,
        proxy=proxy,
        manual_seed_target_config_path=(
            root / "configs/"
            "muggledsam_e4_left_hand_right_hand_yellow_toy_top_black_toy_top_base_manual_seed.json"
        ),
    )

    assert [seed["candidate_id"] for seed in payload["seeds"]] == [
        "t000000-b47",
        "t000000-b48",
        "t000000-b49",
        "t000000-b50",
    ]
    assert [seed.intended_target for seed in metadata.seeds] == [
        "left_hand",
        "right_hand",
        "yellow_toy_top",
        "black_toy_top_base",
    ]
    assert [seed["initial_multiplex_slot"] for seed in payload["seeds"]] == [0, 1, 2, 3]

    for candidate in candidates:
        mask_path = tmp_path / candidate.decoder_result.candidates[0].mask_uri
        mask_path.write_bytes(candidate.candidate_id.encode())
    schedule_path = tmp_path / "multi_keyframe_correction_schedule.json"
    schedule = finalize_correction_schedule(
        manifest_path=manifest_path,
        candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
        schedule_path=schedule_path,
        correction_policy_path=(
            root / "configs/muggledsam_e4_four_target_keyframe_correction_policy.json"
        ),
        manual_seed_target_config_path=(
            root / "configs/"
            "muggledsam_e4_left_hand_right_hand_yellow_toy_top_black_toy_top_base_manual_seed.json"
        ),
        repository_root=root,
    )
    keyframe_payload, keyframe_metadata = _load_multi_keyframe_correction_schedule(
        schedule_path=schedule_path,
        repository_root=root,
        config_path=g2_path,
        proxy=proxy,
        analysis_fps=30.0,
    )

    assert [slot.target_id for slot in schedule.slots] == [
        targets_and_ids[index][0] for index in range(4)
    ]
    assert [seed["target"] for seed in keyframe_payload["seeds"]] == [
        "left_hand",
        "right_hand",
        "yellow_toy_top",
        "black_toy_top_base",
    ]
    assert keyframe_payload["corrections"] == []
    assert keyframe_metadata.scheduled_correction_frame_indices == ()

    tampered = json.loads(schedule_path.read_text())
    tampered["corrections"][0]["calibration_mask_fingerprint"]["sha256"] = "0" * 64
    schedule_path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="mask is unavailable or has changed"):
        _load_multi_keyframe_correction_schedule(
            schedule_path=schedule_path,
            repository_root=root,
            config_path=g2_path,
            proxy=proxy,
            analysis_fps=30.0,
        )


REFERENCE_SCHEDULE = Path(
    "runs/muggledsam-sam3-four-part-focused-corrections-agent-swap-20260918t000947z/"
    "multi_keyframe_correction_schedule.json"
)
FOCUSED_CONFIG = Path("configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_g2.json")


@pytest.mark.real_data
def test_bounded_focused_runs_drop_only_the_corrections_they_cannot_reach() -> None:
    from conftest import require_artifact

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
    )

    full_payload, full_metadata = _load_multi_keyframe_correction_schedule(
        **common, max_frame_exclusive=2781
    )
    bounded_payload, bounded_metadata = _load_multi_keyframe_correction_schedule(
        **common, max_frame_exclusive=1800, drop_out_of_range_corrections=True
    )
    seeds_only_payload, seeds_only_metadata = _load_multi_keyframe_correction_schedule(
        **common,
        max_frame_exclusive=1800,
        drop_out_of_range_corrections=True,
        frame_zero_seeds_only=True,
    )

    assert full_metadata.scheduled_correction_frame_indices == (327, 900, 1172, 1235, 1800, 2700)
    assert bounded_metadata.scheduled_correction_frame_indices == (327, 900, 1172, 1235)
    assert bounded_metadata.dropped_correction_frame_indices == (1800, 2700)
    assert bounded_metadata.agent_selected_correction_frame_indices == (1172,)
    assert bounded_payload["seeds"] == full_payload["seeds"]
    assert all(item["frame_index"] < 1800 for item in bounded_payload["corrections"])
    assert seeds_only_payload["corrections"] == []
    assert seeds_only_metadata.frame_zero_seeds_only is True
    assert seeds_only_metadata.dropped_correction_frame_indices == (
        327,
        900,
        1172,
        1235,
        1800,
        2700,
    )
    with pytest.raises(ValueError, match=r"permits only frames \[0, 1800\)"):
        _load_multi_keyframe_correction_schedule(**common, max_frame_exclusive=1800)


def test_correction_schedule_is_refused_on_a_clock_it_was_not_authored_against(
    tmp_path: Path,
) -> None:
    """Frame indices are clock-bound, so reinterpreting them would move every correction."""
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="authored against"):
        _load_multi_keyframe_correction_schedule(
            schedule_path=tmp_path / "absent_schedule.json",
            repository_root=root,
            config_path=root / "configs/clips/assembly101_nusar_9033_e4_60fps.json",
            proxy=types.SimpleNamespace(view_id="ego-hmc21179183"),
            analysis_fps=60.0,
        )
