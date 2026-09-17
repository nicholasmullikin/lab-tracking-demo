from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from battle.kineo_nlf import _config_with_frame_step, _validate_pkls


def test_kineo_frame_step_override_preserves_other_configuration(tmp_path: Path) -> None:
    source = tmp_path / "source.yaml"
    target = tmp_path / "nested" / "override.yaml"
    source.write_text("batch_size: 32\nrtmlib_bbox_detection_frame_step: 5\n", encoding="utf-8")

    _config_with_frame_step(source, target, 1)

    assert target.read_text(encoding="utf-8") == (
        "batch_size: 32\nrtmlib_bbox_detection_frame_step: 1\n"
    )


def test_kineo_frame_step_override_requires_existing_setting(tmp_path: Path) -> None:
    source = tmp_path / "source.yaml"
    source.write_text("batch_size: 32\n", encoding="utf-8")

    with pytest.raises(ValueError, match="frame-step"):
        _config_with_frame_step(source, tmp_path / "target.yaml", 1)


def test_nlf_pkl_validation_accepts_fused_crop_import_shape(tmp_path: Path) -> None:
    names = [f"joint-{index}" for index in range(55)]
    (tmp_path / "bboxes_2d.pkl").write_bytes(
        pickle.dumps(
            {
                "annotations": [
                    {
                        "frame_idx": 4,
                        "subject_id": "subject_0",
                        "xyxy": [10.0, 20.0, 100.0, 200.0],
                        "score": 0.5,
                    }
                ]
            }
        )
    )
    (tmp_path / "keypoints_2d.pkl").write_bytes(
        pickle.dumps(
            {
                "metadata": {"formats": [{"keypoints_names": names}]},
                "annotations": [
                    {
                        "frame_idx": 4,
                        "subject_id": "subject_0",
                        "xy": [[10.0, 20.0]] * 55,
                        "scores": [0.8] * 55,
                    }
                ],
            }
        )
    )
    (tmp_path / "stage_timings.pkl").write_bytes(pickle.dumps({"annotations": []}))

    validation = _validate_pkls(tmp_path)

    assert validation["bbox_rows"] == validation["keypoint_rows"] == 1
    assert validation["finite_body_xy_values"] == 55
