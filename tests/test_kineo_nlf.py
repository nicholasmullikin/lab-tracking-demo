from __future__ import annotations

from pathlib import Path

import pytest

from battle.kineo_nlf import _config_with_frame_step


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
