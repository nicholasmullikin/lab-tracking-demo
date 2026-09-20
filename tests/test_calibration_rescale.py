"""Rescaling a calibration and its schedule onto a proxy of another size."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from battle import calibration_rescale
from battle.digest_cache import sha256_file
from battle.schemas import (
    G2PreprocessingManifest,
    MuggledSAMBoxCalibrationManifest,
    MuggledSAMMultiKeyframeCorrectionSchedule,
)

ROOT = Path(__file__).resolve().parents[1]
FOCUSED_CONFIG = ROOT / "configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_g2.json"
RAW_SHA = "450731ebbb50f46cf8279383e4737db6d76e967f23580d3555de1888b78a9db9"
FAKE_SHA = "ab" * 32


def _write_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return sha256_file(path)


def _g2_config(root: Path, name: str, *, width: int, height: int, raw_sha: str = RAW_SHA) -> Path:
    payload = json.loads(FOCUSED_CONFIG.read_text(encoding="utf-8"))
    proxy = payload["proxies"][0]
    proxy["proxy_uri"] = f"data/{name}.mp4"
    proxy["checksum_sha256"] = FAKE_SHA
    proxy["dimensions"] = {"width": width, "height": height}
    proxy["raw_source"]["checksum_sha256"] = raw_sha
    payload["clip"]["asset"]["uri"] = proxy["proxy_uri"]
    payload["clip"]["asset"]["checksum_sha256"] = FAKE_SHA
    payload["scaling_policy"] = (
        "preserve_aspect_ratio_height_1080"
        if height == 1080
        else "preserve_aspect_ratio_height_720"
    )
    path = root / f"configs/{name}.json"
    _write_json(path, payload)
    return path


def _frame(index: int) -> dict:
    return {
        "analysis_frame_index": index,
        "proxy_seconds": index / 30,
        "analysis_seconds": index / 30,
        "source_seconds": 294.0 + index / 30,
    }


def _mask(shape: tuple[int, int], x: int, y: int, size: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[y : y + size, x : x + size] = 255
    return mask


def _source_calibration(root: Path, config_path: Path) -> tuple[Path, Path]:
    """A two-candidate calibration (frame-0 seed for two targets, one later correction)."""
    workspace = root / "runs/source-cal"
    masks = workspace / "results/masks"
    masks.mkdir(parents=True)
    shape = (720, 1280)
    files = {
        "t000000-b01_candidate-00.png": _mask(shape, 100, 100, 40),
        "t000000-b01_candidate-01.png": _mask(shape, 100, 100, 20),
        "t000000-b02_candidate-00.png": _mask(shape, 300, 200, 30),
        "t000300-b03_candidate-00.png": _mask(shape, 111, 101, 41),
    }
    for name, mask in files.items():
        cv2.imwrite(str(masks / name), mask)
    (workspace / "results/overlay.png").write_bytes(b"png")

    def candidate(cid: str, target: str, frame: int, names: list[str], **flags: object) -> dict:
        return {
            "candidate_id": cid,
            "intended_target": target,
            "frame": _frame(frame),
            "pixel_box": {"x1": 101, "y1": 99, "x2": 141, "y2": 141},
            "normalized_box": {
                "x": 101 / 1280,
                "y": 99 / 720,
                "width": 40 / 1280,
                "height": 42 / 720,
            },
            "pixel_fg_points": [{"x": 121, "y": 121}],
            "normalized_fg_points": [{"x": 121 / 1280, "y": 121 / 720}],
            "decoder_result": {
                "api": "muggledsam_sam3_interactive",
                "candidate_count": len(names),
                "deterministic_best_candidate_index": 0,
                "candidates": [
                    {
                        "candidate_index": i,
                        "iou_score": 0.5,
                        "mask_uri": f"results/masks/{name}",
                        "is_deterministic_best": i == 0,
                    }
                    for i, name in enumerate(names)
                ],
                "overlay_uri": "results/overlay.png",
            },
            "human_selected_candidate_index": 0,
            "human_accepted": True,
            **flags,
        }

    manifest = {
        "manifest_kind": "muggledsam_sam3_box_calibration",
        "calibration_id": "source-cal",
        "base_g2_config": config_path.relative_to(root).as_posix(),
        "base_g2_config_sha256": sha256_file(config_path),
        "view_id": "static-c10379",
        "proxy": {"uri": "data/p720.mp4", "sha256": FAKE_SHA, "source": "approved_config"},
        "source": {"uri": "data/raw.mp4", "sha256": RAW_SHA, "source": "approved_config"},
        "proxy_dimensions": {"width": 1280, "height": 720},
        "proxy_fps": 30,
        "proxy_frame_count": 2781,
        "source_offset_seconds": 294.0,
        "tool_version": "test",
        "image_representation": "original_bgr",
        "prompt_api": "boxes_fg_points_bg_points",
        "requested_proxy_timestamps_seconds": [0.0, 10.0],
        "candidates": [
            candidate(
                "t000000-b01",
                "chassis",
                0,
                ["t000000-b01_candidate-00.png", "t000000-b01_candidate-01.png"],
                selected_for_finalization=True,
            ),
            candidate(
                "t000000-b02",
                "cabin",
                0,
                ["t000000-b02_candidate-00.png"],
                selected_for_finalization=True,
            ),
            candidate(
                "t000300-b03",
                "chassis",
                300,
                ["t000300-b03_candidate-00.png"],
                selected_for_correction=True,
            ),
        ],
        "result_directory_uri": "runs/source-cal/results",
    }
    manifest_path = workspace / "calibration_manifest.json"
    manifest_sha = _write_json(manifest_path, manifest)
    MuggledSAMBoxCalibrationManifest.model_validate_json(manifest_path.read_text())

    seed_config = {
        "manifest_kind": "muggledsam_sam3_manual_seed_targets",
        "config_id": "seeds-720",
        "base_g2_config": config_path.relative_to(root).as_posix(),
        "view_id": "static-c10379",
        "targets": ["chassis", "cabin"],
    }
    seed_sha = _write_json(root / "configs/seeds_720.json", seed_config)
    policy = {
        "manifest_kind": "muggledsam_sam3_multi_keyframe_correction_policy",
        "policy_id": "policy-720",
        "policy_version": "3",
        "view_id": "static-c10379",
        "manual_seed_target_config_fingerprint": {
            "uri": "configs/seeds_720.json",
            "sha256": seed_sha,
            "source": "measured",
        },
        "targets": ["chassis", "cabin"],
        "maximum_later_correction_keyframes_per_target": 6,
        "correction_memory_semantics": "replace_prompt_memory_and_reset_frame_memory",
    }
    policy_sha = _write_json(root / "configs/policy_720.json", policy)

    def correction(cid: str, target: str, slot: int, frame: int, name: str, by: str) -> dict:
        return {
            "candidate_id": cid,
            "human_selected_candidate_index": 0,
            "selected_by": by,
            "target_id": target,
            "object_id": f"sam3-{slot:02d}",
            "multiplex_slot": slot,
            "frame": _frame(frame),
            "calibration_mask_fingerprint": {
                "uri": f"runs/source-cal/results/masks/{name}",
                "sha256": sha256_file(masks / name),
                "source": "measured",
            },
        }

    schedule = {
        "manifest_kind": "muggledsam_sam3_multi_keyframe_correction_schedule",
        "authority": "proposed_non_authoritative",
        "schedule_version": "1",
        "view_id": "static-c10379",
        "calibration_manifest_fingerprint": {
            "uri": "runs/source-cal/calibration_manifest.json",
            "sha256": manifest_sha,
            "source": "measured",
        },
        "correction_policy_fingerprint": {
            "uri": "configs/policy_720.json",
            "sha256": policy_sha,
            "source": "measured",
        },
        "manual_seed_target_config_fingerprint": {
            "uri": "configs/seeds_720.json",
            "sha256": seed_sha,
            "source": "measured",
        },
        "correction_memory_semantics": "replace_prompt_memory_and_reset_frame_memory",
        "slots": [
            {"target_id": "chassis", "object_id": "sam3-00", "multiplex_slot": 0},
            {"target_id": "cabin", "object_id": "sam3-01", "multiplex_slot": 1},
        ],
        "corrections": [
            correction("t000000-b01", "chassis", 0, 0, "t000000-b01_candidate-00.png", "human"),
            correction("t000000-b02", "cabin", 1, 0, "t000000-b02_candidate-00.png", "human"),
            correction("t000300-b03", "chassis", 0, 300, "t000300-b03_candidate-00.png", "agent"),
        ],
    }
    schedule_path = workspace / "multi_keyframe_correction_schedule.json"
    _write_json(schedule_path, schedule)
    MuggledSAMMultiKeyframeCorrectionSchedule.model_validate_json(schedule_path.read_text())
    return manifest_path, schedule_path


def _target_configs(root: Path, target_config: Path) -> tuple[Path, Path]:
    seed_path = root / "configs/seeds_1080.json"
    seed_sha = _write_json(
        seed_path,
        {
            "manifest_kind": "muggledsam_sam3_manual_seed_targets",
            "config_id": "seeds-1080",
            "base_g2_config": target_config.relative_to(root).as_posix(),
            "view_id": "static-c10379",
            "targets": ["chassis", "cabin"],
        },
    )
    policy_path = root / "configs/policy_1080.json"
    _write_json(
        policy_path,
        {
            "manifest_kind": "muggledsam_sam3_multi_keyframe_correction_policy",
            "policy_id": "policy-1080",
            "policy_version": "3",
            "view_id": "static-c10379",
            "manual_seed_target_config_fingerprint": {
                "uri": "configs/seeds_1080.json",
                "sha256": seed_sha,
                "source": "measured",
            },
            "targets": ["chassis", "cabin"],
            "maximum_later_correction_keyframes_per_target": 6,
            "correction_memory_semantics": "replace_prompt_memory_and_reset_frame_memory",
        },
    )
    return seed_path, policy_path


def test_rescale_writes_nearest_masks_scaled_boxes_and_rebound_schedule(tmp_path: Path) -> None:
    source_config = _g2_config(tmp_path, "g2_720", width=1280, height=720)
    target_config = _g2_config(tmp_path, "g2_1080", width=1920, height=1080)
    manifest_path, schedule_path = _source_calibration(tmp_path, source_config)
    seed_path, policy_path = _target_configs(tmp_path, target_config)
    out = tmp_path / "runs/source-cal-rescaled"

    provenance = calibration_rescale.rescale_calibration(
        repository_root=tmp_path,
        calibration_manifest=manifest_path,
        schedule=schedule_path,
        target_g2_config=target_config,
        target_manual_seed_config=seed_path,
        target_correction_policy=policy_path,
        output_dir=out,
    )

    assert (provenance.scale_x, provenance.scale_y) == (1.5, 1.5)
    assert provenance.interpolation == "nearest" and len(provenance.masks) == 4
    for record in provenance.masks:
        assert record.target_area_px == pytest.approx(record.source_area_px * 2.25, rel=0.02)
        assert sha256_file(tmp_path / record.target_uri) == record.target_sha256
        resized = cv2.imread(str(tmp_path / record.target_uri), cv2.IMREAD_GRAYSCALE)
        assert resized.shape == (1080, 1920)
        assert set(np.unique(resized)) <= {0, 255}  # nearest keeps the mask binary

    derived = MuggledSAMBoxCalibrationManifest.model_validate_json(
        (out / "calibration_manifest.json").read_text()
    )
    assert derived.proxy_dimensions.width == 1920 and derived.proxy_dimensions.height == 1080
    assert derived.base_g2_config == "configs/g2_1080.json"
    assert derived.base_g2_config_sha256 == sha256_file(target_config)
    assert derived.derived_from_calibration is not None
    assert derived.derived_from_calibration.sha256 == sha256_file(manifest_path)
    assert derived.calibration_id == "source-cal-rescaled-1920x1080"
    first = derived.candidates[0]
    assert (first.pixel_box.x1, first.pixel_box.y1, first.pixel_box.x2, first.pixel_box.y2) == (
        152,
        148,
        212,
        212,
    )
    assert first.pixel_fg_points[0].x == 182 and first.pixel_fg_points[0].y == 182
    assert first.decoder_result.candidates[0].mask_uri == (
        "results/masks/t000000-b01_candidate-00.png"
    )
    # Review artefacts are not rescaled; their URIs point back at the source workspace.
    assert derived.candidates[0].decoder_result.overlay_uri.startswith("../source-cal/")
    assert derived.workspace.pending_boxes == ()
    assert derived.final_correction_schedule_uri == (
        "runs/source-cal-rescaled/multi_keyframe_correction_schedule.json"
    )

    schedule = MuggledSAMMultiKeyframeCorrectionSchedule.model_validate_json(
        (out / "multi_keyframe_correction_schedule.json").read_text()
    )
    assert schedule.calibration_manifest_fingerprint.sha256 == sha256_file(
        out / "calibration_manifest.json"
    )
    assert schedule.correction_policy_fingerprint.uri == "configs/policy_1080.json"
    assert schedule.manual_seed_target_config_fingerprint.uri == "configs/seeds_1080.json"
    for correction in schedule.corrections:
        path = tmp_path / correction.calibration_mask_fingerprint.uri
        assert path.parent == out / "results/masks"
        assert sha256_file(path) == correction.calibration_mask_fingerprint.sha256
    assert [c.frame.analysis_frame_index for c in schedule.corrections] == [0, 0, 300]
    assert [c.selected_by for c in schedule.corrections] == ["human", "human", "agent"]

    sidecar = calibration_rescale.CalibrationRescaleProvenance.model_validate_json(
        (out / "rescale_provenance.json").read_text()
    )
    assert sidecar.derived_schedule is not None
    assert sidecar.derived_schedule.sha256 == sha256_file(
        out / "multi_keyframe_correction_schedule.json"
    )
    assert any("nobody reviewed them" in line for line in sidecar.claim_boundaries)
    G2PreprocessingManifest.model_validate_json(target_config.read_text())


def test_rescale_refuses_same_size_other_recording_and_nonempty_output(tmp_path: Path) -> None:
    source_config = _g2_config(tmp_path, "g2_720", width=1280, height=720)
    manifest_path, schedule_path = _source_calibration(tmp_path, source_config)
    same_size = _g2_config(tmp_path, "g2_same", width=1280, height=720)
    with pytest.raises(ValueError, match="nothing to rescale"):
        calibration_rescale.rescale_calibration(
            repository_root=tmp_path,
            calibration_manifest=manifest_path,
            schedule=None,
            target_g2_config=same_size,
            target_manual_seed_config=None,
            target_correction_policy=None,
            output_dir=tmp_path / "runs/out-a",
        )
    other_recording = _g2_config(tmp_path, "g2_other", width=1920, height=1080, raw_sha="cd" * 32)
    with pytest.raises(ValueError, match="same raw recording"):
        calibration_rescale.rescale_calibration(
            repository_root=tmp_path,
            calibration_manifest=manifest_path,
            schedule=None,
            target_g2_config=other_recording,
            target_manual_seed_config=None,
            target_correction_policy=None,
            output_dir=tmp_path / "runs/out-b",
        )
    target_config = _g2_config(tmp_path, "g2_1080", width=1920, height=1080)
    with pytest.raises(ValueError, match="needs --target-manual-seed-config"):
        calibration_rescale.rescale_calibration(
            repository_root=tmp_path,
            calibration_manifest=manifest_path,
            schedule=schedule_path,
            target_g2_config=target_config,
            target_manual_seed_config=None,
            target_correction_policy=None,
            output_dir=tmp_path / "runs/out-c",
        )
    occupied = tmp_path / "runs/out-d"
    occupied.mkdir(parents=True)
    (occupied / "x").write_text("x")
    with pytest.raises(FileExistsError):
        calibration_rescale.rescale_calibration(
            repository_root=tmp_path,
            calibration_manifest=manifest_path,
            schedule=None,
            target_g2_config=target_config,
            target_manual_seed_config=None,
            target_correction_policy=None,
            output_dir=occupied,
        )


def test_committed_1080p_configs_chain_to_each_other() -> None:
    config = G2PreprocessingManifest.model_validate_json(
        (
            ROOT / "configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_1080p_g2.json"
        ).read_text()
    )
    focused = G2PreprocessingManifest.model_validate_json(FOCUSED_CONFIG.read_text())
    assert config.scaling_policy == "preserve_aspect_ratio_height_1080"
    proxy, base = config.proxies[0], focused.proxies[0]
    assert (proxy.dimensions.width, proxy.dimensions.height) == (1920, 1080)
    assert proxy.raw_source == base.raw_source
    assert proxy.fps == base.fps and proxy.frame_count == base.frame_count
    assert config.source_interval == focused.source_interval
    seed_path = (
        ROOT / "configs/muggledsam_static_four_part_reassembly_focused_manual_seed_1080p.json"
    )
    seed = json.loads(seed_path.read_text())
    assert seed["base_g2_config"] == (
        "configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_1080p_g2.json"
    )
    policy_name = "muggledsam_static_four_part_reassembly_focused_correction_policy_v3_1080p.json"
    policy = json.loads((ROOT / "configs" / policy_name).read_text())
    assert policy["manual_seed_target_config_fingerprint"]["sha256"] == sha256_file(seed_path)
