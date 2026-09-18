from __future__ import annotations

from pathlib import Path

import pytest

from battle.muggled_calibration import (
    E4_VIEW_ID,
    build_manifest,
    finalize_correction_schedule,
    finalize_prompt,
    frame_reference,
    parse_timestamps,
)
from battle.muggled_calibration_worker import (
    _aspect_fit_geometry,
    _candidate_action,
    _candidate_grid_geometry,
    _collection_action,
    _decoder_pixel_center_to_source,
    _encode_muggledsam_prompts,
    _finalized_seed_review_records,
    _human_selection_caption,
    _muggledsam_prompt_arguments,
    _review_crop_bounds,
    _select_single_box,
    _timestamp_action,
)
from battle.schemas import MuggledSAMCalibrationCandidate


class _Frame:
    shape = (720, 954, 3)


def _candidate(
    *,
    selected_for_finalization: bool,
    frame_index: int = 300,
    human_accepted: bool = False,
    selected_for_correction: bool = False,
    candidate_id: str | None = None,
    intended_target: str = "hand",
) -> MuggledSAMCalibrationCandidate:
    candidate_id = candidate_id or f"t{frame_index:06d}-b01"
    return MuggledSAMCalibrationCandidate.model_validate(
        {
            "candidate_id": candidate_id,
            "intended_target": intended_target,
            "frame": {
                "analysis_frame_index": frame_index,
                "proxy_seconds": frame_index / 30,
                "analysis_seconds": frame_index / 30,
                "source_seconds": 215.0 + frame_index / 30,
            },
            "pixel_box": {"x1": 100, "y1": 120, "x2": 300, "y2": 400},
            "normalized_box": {
                "x": 100 / 954,
                "y": 120 / 720,
                "width": 200 / 954,
                "height": 280 / 720,
            },
            "decoder_result": {
                "api": "muggledsam_sam3_interactive",
                "candidate_count": 2,
                "deterministic_best_candidate_index": 1,
                "candidates": [
                    {
                        "candidate_index": 0,
                        "iou_score": 0.2,
                        "mask_uri": f"results/masks/{candidate_id}_candidate-00.png",
                        "is_deterministic_best": False,
                    },
                    {
                        "candidate_index": 1,
                        "iou_score": 0.8,
                        "mask_uri": f"results/masks/{candidate_id}_candidate-01.png",
                        "is_deterministic_best": True,
                    },
                ],
                "overlay_uri": f"results/{candidate_id}_overlay.png",
                "stability_score_available": False,
                "limitations": ["The image API exposes no stability score."],
            },
            "human_selected_candidate_index": 1 if human_accepted else None,
            "human_accepted": human_accepted,
            "selected_for_finalization": selected_for_finalization,
            "selected_for_correction": selected_for_correction,
        }
    )


def test_timestamp_mapping_persists_exact_proxy_and_source_times() -> None:
    reference = frame_reference(10.0, fps=30, source_offset_seconds=215.0, frame_count=5400)

    assert reference.analysis_frame_index == 300
    assert reference.proxy_seconds == 10.0
    assert reference.analysis_seconds == 10.0
    assert reference.source_seconds == 225.0
    with pytest.raises(ValueError, match="outside"):
        frame_reference(180.0, fps=30, source_offset_seconds=215.0, frame_count=5400)


def test_build_manifest_supports_static_frame_zero_workspace(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    manifest = build_manifest(
        repository_root=root,
        config_path=root / "configs/clips/assembly101_nusar_9033_g2.json",
        timestamps=(0.0,),
        result_directory=tmp_path / "results",
        calibration_id="static-black-base-calibration",
        view_id="static-c10379",
    )

    assert manifest.view_id == "static-c10379"
    assert manifest.requested_proxy_timestamps_seconds == (0.0,)
    assert manifest.proxy_dimensions.width == 1280
    assert manifest.proxy_dimensions.height == 720
    assert manifest.proxy_fps == 30
    assert manifest.source_offset_seconds == 215.0


def test_finalize_prompt_supports_static_manual_seed_workspace(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    candidate = _candidate(
        selected_for_finalization=True,
        frame_index=0,
        human_accepted=True,
        intended_target="black_toy_top_base",
    )
    candidate = candidate.model_copy(
        update={
            "normalized_box": candidate.normalized_box.model_copy(
                update={
                    "x": 100 / 1280,
                    "y": 120 / 720,
                    "width": 200 / 1280,
                    "height": 280 / 720,
                }
            )
        }
    )
    manifest = build_manifest(
        repository_root=root,
        config_path=root / "configs/clips/assembly101_nusar_9033_g2.json",
        timestamps=(0.0,),
        result_directory=tmp_path / "results",
        calibration_id="static-black-base-calibration",
        view_id="static-c10379",
    ).model_copy(update={"candidates": (candidate,)})
    manifest_path = tmp_path / "calibration_manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n")

    proposal = finalize_prompt(
        manifest_path=manifest_path,
        candidate_ids=("t000000-b01",),
        proposal_path=tmp_path / "proposed_tracking_prompt.json",
        repository_root=root,
        manual_seed_target_config_path=(
            root / "configs/muggledsam_static_black_toy_top_base_manual_seed.json"
        ),
    )

    assert proposal.view_id == "static-c10379"
    assert [seed.intended_target for seed in proposal.seeds] == ["black_toy_top_base"]


def test_timestamp_parser_preserves_user_order_and_refuses_duplicates() -> None:
    assert parse_timestamps("0, 10,30,50") == (0.0, 10.0, 30.0, 50.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        parse_timestamps("10,0")


def test_single_roi_selection_returns_one_box_and_always_closes_window() -> None:
    calls: list[object] = []
    frame = _Frame()

    def selector(title: str, frame: _Frame, **kwargs: object) -> tuple[int, int, int, int]:
        calls.extend((title, frame, kwargs))
        return (100, 120, 200, 280)

    assert _select_single_box(
        frame=frame,
        title="one box",
        selector=selector,
        close_window=lambda title: calls.append(("closed", title)),
    ) == {"x1": 100, "y1": 120, "x2": 300, "y2": 400}
    assert calls == [
        "one box",
        frame,
        {"showCrosshair": True, "fromCenter": False},
        ("closed", "one box"),
    ]


def test_single_roi_cancellation_and_terminal_controls_are_unambiguous() -> None:
    closed: list[str] = []

    assert (
        _select_single_box(
            frame=_Frame(),
            title="cancelled box",
            selector=lambda *_args, **_kwargs: (0, 0, 0, 0),
            close_window=closed.append,
        )
        is None
    )
    assert closed == ["cancelled box"]
    assert _timestamp_action("") == "draw"
    assert _timestamp_action("s") == "skip"
    assert _timestamp_action("q") == "finish"
    assert _timestamp_action("draw") is None
    assert _collection_action("d", has_boxes=False) is None
    assert _collection_action("d", has_boxes=True) == "decode"
    assert _collection_action("a", has_boxes=True) == "add"
    assert _collection_action("r", has_boxes=True) == "retry"
    assert _candidate_action("a") == "accept"
    assert _candidate_action("r") == "retry"
    assert _candidate_action("s") == "skip"
    assert _candidate_action("decode") is None


def test_decoder_grid_pixel_centers_map_to_full_source_pixel_extent() -> None:
    source_width, source_height = 954, 720
    decoder_width, decoder_height = 288, 288

    top_left = _decoder_pixel_center_to_source(
        0,
        0,
        decoder_width=decoder_width,
        decoder_height=decoder_height,
        source_width=source_width,
        source_height=source_height,
    )
    bottom_right = _decoder_pixel_center_to_source(
        decoder_width - 1,
        decoder_height - 1,
        decoder_width=decoder_width,
        decoder_height=decoder_height,
        source_width=source_width,
        source_height=source_height,
    )

    assert top_left == pytest.approx((954 / 576 - 0.5, 720 / 576 - 0.5))
    assert bottom_right == pytest.approx((954 - 954 / 576 - 0.5, 720 - 720 / 576 - 0.5))


def test_muggledsam_box_foreground_background_prompt_shapes_are_passed_unchanged() -> None:
    boxes, foreground, background = _muggledsam_prompt_arguments(
        prompt_box={"x1": 100, "y1": 120, "x2": 300, "y2": 400},
        width=954,
        height=720,
        boxes=[[[100 / 954, 120 / 720], [300 / 954, 400 / 720]]],
        fg_points=[[150 / 954, 200 / 720]],
        bg_points=[[350 / 954, 220 / 720]],
    )

    class SpyInteractiveModel:
        calls: list[object] = []

        def encode_prompts(self, *args: object) -> str:
            self.calls.append(args)
            return "encoded"

    model = SpyInteractiveModel()
    assert (
        _encode_muggledsam_prompts(model, boxes=boxes, fg_points=foreground, bg_points=background)
        == "encoded"
    )
    assert model.calls == [(boxes, foreground, background)]


def test_review_crop_unions_prompt_and_mask_with_clamped_padding() -> None:
    crop = _review_crop_bounds(
        prompt_box={"x1": 100, "y1": 120, "x2": 200, "y2": 220},
        mask_box={"x1": 50, "y1": 80, "x2": 400, "y2": 300},
        width=954,
        height=720,
    )

    assert crop == {"x1": 0, "y1": 10, "x2": 470, "y2": 370}
    assert crop["x1"] <= 50 < 400 <= crop["x2"]
    assert crop["y1"] <= 80 < 300 <= crop["y2"]


def test_aspect_fit_geometry_letterboxes_roi_without_distorting_it() -> None:
    scale, resized_width, resized_height, offset_x, offset_y = _aspect_fit_geometry(
        width=400,
        height=100,
        target_width=446,
        target_height=300,
    )

    assert scale == pytest.approx(1.115)
    assert (resized_width, resized_height) == (446, 112)
    assert (offset_x, offset_y) == (0, 94)
    assert resized_width / resized_height == pytest.approx(4.0, abs=0.03)


def test_four_decoder_candidates_use_a_two_by_two_contact_sheet_grid() -> None:
    assert _candidate_grid_geometry(4) == (2, 2, 1890, 858)


def test_finalized_review_renders_accepted_selection_and_preserves_exclusion() -> None:
    selected = _candidate(
        selected_for_finalization=True, frame_index=0, human_accepted=True
    ).model_dump(mode="json")
    excluded = _candidate(
        selected_for_finalization=False, frame_index=0, human_accepted=True
    ).model_dump(mode="json")
    excluded["candidate_id"] = "t000000-b02"
    excluded["intended_target"] = "right_hand"
    proposal = {
        "seeds": [
            {
                "candidate_id": selected["candidate_id"],
                "intended_target": selected["intended_target"],
                "human_selected_candidate_index": selected["human_selected_candidate_index"],
            }
        ]
    }

    records, preserved_exclusions = _finalized_seed_review_records(
        {"candidates": [selected, excluded]}, proposal
    )

    assert records == [(selected, proposal["seeds"][0])]
    assert (
        _human_selection_caption(
            selected["human_selected_candidate_index"], selected["human_accepted"]
        )
        == "Human selection: accepted candidate 1"
    )
    assert preserved_exclusions == [excluded]


def test_calibration_manifest_records_selected_e4_proxy_identity() -> None:
    root = Path(__file__).parents[1]
    config_path = root / "configs/clips/assembly101_nusar_9033_ego_viewpoint_screen_g2.json"
    manifest = build_manifest(
        repository_root=root,
        config_path=config_path,
        timestamps=(0.0, 10.0, 30.0, 50.0),
        result_directory=root / "runs/example/results",
        calibration_id="muggledsam-sam3-e4-box-calibration-test",
    )

    assert manifest.view_id == E4_VIEW_ID
    assert manifest.proxy_fps == 30
    assert manifest.proxy_frame_count == 5400
    assert (
        manifest.proxy.sha256 == "ce9ae52af7184c44faabb0a0dd5157eb1d4d69af6401674d157a5f2a3dde4007"
    )
    assert manifest.source_offset_seconds == 215.0


def test_candidate_result_requires_one_deterministic_best_mask() -> None:
    payload = _candidate(selected_for_finalization=False).model_dump(mode="json")
    payload["decoder_result"]["candidates"][1]["is_deterministic_best"] = False

    with pytest.raises(ValueError, match="deterministic best"):
        MuggledSAMCalibrationCandidate.model_validate(payload)


def test_legacy_candidate_without_rejection_state_defaults_to_active_review() -> None:
    payload = _candidate(selected_for_finalization=False).model_dump(mode="json")
    payload.pop("rejected")

    candidate = MuggledSAMCalibrationCandidate.model_validate(payload)

    assert candidate.rejected is False


def test_finalize_writes_non_authoritative_user_selected_prompt(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    config_path = root / "configs/clips/assembly101_nusar_9033_ego_viewpoint_screen_g2.json"
    manifest = build_manifest(
        repository_root=root,
        config_path=config_path,
        timestamps=(0.0,),
        result_directory=tmp_path / "results",
        calibration_id="muggledsam-sam3-e4-box-calibration-test",
    ).model_copy(
        update={
            "candidates": (
                _candidate(selected_for_finalization=True, frame_index=0, human_accepted=True),
            )
        }
    )
    manifest_path = tmp_path / "calibration_manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n")

    proposal = finalize_prompt(
        manifest_path=manifest_path,
        candidate_ids=("t000000-b01",),
        proposal_path=tmp_path / "proposed_tracking_prompt.json",
        repository_root=tmp_path,
    )

    assert proposal.authority == "proposed_non_authoritative"
    assert proposal.ground_truth_accuracy_claim is False
    assert proposal.seeds[0].intended_target == "hand"
    assert proposal.seeds[0].human_selected_candidate_index == 1


def test_target_config_finalization_requires_three_new_human_selected_masks(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    config_path = root / "configs/clips/assembly101_nusar_9033_ego_viewpoint_screen_g2.json"
    target_config_path = (
        root / "configs/muggledsam_e4_yellow_toy_top_black_toy_top_base_manual_seed.json"
    )
    targets = ("left_hand", "yellow_toy_top", "black_toy_top_base")
    candidates = []
    for candidate_id, target in zip(("t000000-b47", "t000000-b11", "t000000-b92"), targets):
        candidate = _candidate(
            selected_for_finalization=True, frame_index=0, human_accepted=True
        ).model_copy(
            update={
                "candidate_id": candidate_id,
                "intended_target": target,
            }
        )
        candidates.append(candidate)
    manifest = build_manifest(
        repository_root=root,
        config_path=config_path,
        timestamps=(0.0,),
        result_directory=tmp_path / "results",
        calibration_id="muggledsam-sam3-e4-box-calibration-targets",
    ).model_copy(update={"candidates": tuple(candidates)})
    manifest_path = tmp_path / "calibration_manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n")

    with pytest.raises(ValueError, match="exactly 3 distinct"):
        finalize_prompt(
            manifest_path=manifest_path,
            candidate_ids=("t000000-b47", "t000000-b11"),
            proposal_path=tmp_path / "missing.json",
            repository_root=root,
            manual_seed_target_config_path=target_config_path,
        )

    proposal = finalize_prompt(
        manifest_path=manifest_path,
        candidate_ids=("t000000-b92", "t000000-b47", "t000000-b11"),
        proposal_path=tmp_path / "proposed_tracking_prompt.json",
        repository_root=root,
        manual_seed_target_config_path=target_config_path,
    )

    assert [seed.intended_target for seed in proposal.seeds] == list(targets)
    assert proposal.manual_seed_target_config_fingerprint is not None
    assert proposal.manual_seed_target_config_fingerprint.uri == (
        "configs/muggledsam_e4_yellow_toy_top_black_toy_top_base_manual_seed.json"
    )


def test_four_target_config_finalization_requires_four_new_human_selected_masks(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[1]
    config_path = root / "configs/clips/assembly101_nusar_9033_ego_viewpoint_screen_g2.json"
    target_config_path = (
        root / "configs/"
        "muggledsam_e4_left_hand_right_hand_yellow_toy_top_black_toy_top_base_manual_seed.json"
    )
    targets = ("left_hand", "right_hand", "yellow_toy_top", "black_toy_top_base")
    candidates = tuple(
        _candidate(selected_for_finalization=True, frame_index=0, human_accepted=True).model_copy(
            update={"candidate_id": candidate_id, "intended_target": target}
        )
        for candidate_id, target in zip(
            ("t000000-b47", "t000000-b48", "t000000-b49", "t000000-b50"),
            targets,
            strict=True,
        )
    )
    manifest = build_manifest(
        repository_root=root,
        config_path=config_path,
        timestamps=(0.0,),
        result_directory=tmp_path / "results",
        calibration_id="muggledsam-sam3-e4-box-calibration-four-targets",
    ).model_copy(update={"candidates": candidates})
    manifest_path = tmp_path / "calibration_manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n")

    with pytest.raises(ValueError, match="exactly 4 distinct"):
        finalize_prompt(
            manifest_path=manifest_path,
            candidate_ids=tuple(candidate.candidate_id for candidate in candidates[:3]),
            proposal_path=tmp_path / "missing.json",
            repository_root=root,
            manual_seed_target_config_path=target_config_path,
        )

    proposal = finalize_prompt(
        manifest_path=manifest_path,
        candidate_ids=tuple(candidate.candidate_id for candidate in reversed(candidates)),
        proposal_path=tmp_path / "proposed_tracking_prompt.json",
        repository_root=root,
        manual_seed_target_config_path=target_config_path,
    )

    assert [seed.intended_target for seed in proposal.seeds] == list(targets)


def test_correction_schedule_requires_frame_zero_and_rejects_duplicate_slots(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[1]
    config_path = root / "configs/clips/assembly101_nusar_9033_ego_viewpoint_screen_g2.json"
    target_config_path = (
        root / "configs/"
        "muggledsam_e4_left_hand_right_hand_yellow_toy_top_black_toy_top_base_manual_seed.json"
    )
    policy_path = root / "configs/muggledsam_e4_four_target_keyframe_correction_policy.json"
    targets = ("left_hand", "right_hand", "yellow_toy_top", "black_toy_top_base")
    initial = tuple(
        _candidate(
            selected_for_finalization=True,
            frame_index=0,
            human_accepted=True,
            candidate_id=f"t000000-b{slot + 1:02d}",
            intended_target=target,
        )
        for slot, target in enumerate(targets)
    )
    later = _candidate(
        selected_for_finalization=False,
        frame_index=30,
        human_accepted=True,
        selected_for_correction=True,
        candidate_id="t000030-b01",
        intended_target="left_hand",
    )
    duplicate_later = _candidate(
        selected_for_finalization=False,
        frame_index=30,
        human_accepted=True,
        selected_for_correction=True,
        candidate_id="t000030-b02",
        intended_target="left_hand",
    )
    candidates = initial + (later, duplicate_later)
    for candidate in candidates:
        selected = next(
            item
            for item in candidate.decoder_result.candidates
            if item.candidate_index == candidate.human_selected_candidate_index
        )
        mask_path = tmp_path / selected.mask_uri
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.write_bytes(candidate.candidate_id.encode())
    manifest = build_manifest(
        repository_root=root,
        config_path=config_path,
        timestamps=(0.0, 1.0),
        result_directory=tmp_path / "results",
        calibration_id="muggledsam-sam3-e4-keyframe-test",
    ).model_copy(update={"candidates": candidates})
    manifest_path = tmp_path / "calibration_manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n")

    schedule = finalize_correction_schedule(
        manifest_path=manifest_path,
        candidate_ids=(
            tuple(candidate.candidate_id for candidate in initial) + (later.candidate_id,)
        ),
        schedule_path=tmp_path / "schedule.json",
        correction_policy_path=policy_path,
        manual_seed_target_config_path=target_config_path,
        repository_root=root,
    )

    assert [
        (item.frame.analysis_frame_index, item.multiplex_slot) for item in schedule.corrections
    ] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (30, 0),
    ]
    with pytest.raises(ValueError, match="ordered and unique"):
        finalize_correction_schedule(
            manifest_path=manifest_path,
            candidate_ids=tuple(candidate.candidate_id for candidate in initial)
            + (later.candidate_id, duplicate_later.candidate_id),
            schedule_path=tmp_path / "duplicate.json",
            correction_policy_path=policy_path,
            manual_seed_target_config_path=target_config_path,
            repository_root=root,
        )


def test_finalize_refuses_candidate_not_selected_by_user(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    config_path = root / "configs/clips/assembly101_nusar_9033_ego_viewpoint_screen_g2.json"
    manifest = build_manifest(
        repository_root=root,
        config_path=config_path,
        timestamps=(0.0,),
        result_directory=tmp_path / "results",
        calibration_id="muggledsam-sam3-e4-box-calibration-test",
    ).model_copy(
        update={
            "candidates": (
                _candidate(selected_for_finalization=False, frame_index=0, human_accepted=False),
            )
        }
    )
    manifest_path = tmp_path / "calibration_manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n")

    with pytest.raises(ValueError, match="human-accepted frame-0"):
        finalize_prompt(
            manifest_path=manifest_path,
            candidate_ids=("t000000-b01",),
            proposal_path=tmp_path / "proposed_tracking_prompt.json",
            repository_root=tmp_path,
        )
