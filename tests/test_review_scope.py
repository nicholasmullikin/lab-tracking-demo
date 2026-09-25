"""The review scope (views, targets) the Assembly101 builders used to assert as literals now
comes from a clip config with the Assembly101 values as defaults (p1-configs)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from battle import four_part_contract as contract
from battle.kineo_fusion import verify_source_alignment

ROOT = Path(__file__).resolve().parents[1]


def test_defaults_are_the_assembly101_view_and_the_four_parts() -> None:
    scope = contract.review_scope()

    assert scope.views == ("static-c10379",) == contract.DEFAULT_APPROVED_VIEWS
    assert scope.targets == ("chassis", "interior", "rear_body", "cabin") == contract.TARGETS
    assert scope.source.startswith("defaults")


def test_assembly101_clip_config_supplies_the_views_and_keeps_the_four_parts() -> None:
    focused = Path("configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_g2.json")
    scope = contract.review_scope(focused, repository_root=ROOT)
    assert scope.views == ("static-c10379",) and scope.targets == contract.TARGETS

    all_static = Path(
        "configs/clips/assembly101_nusar_9061_four_part_reassembly_focused_all_static_g2.json"
    )
    scope = contract.review_scope(all_static, repository_root=ROOT)
    assert len(scope.views) == 8 and "static-c10379" in scope.views
    assert scope.targets == contract.TARGETS


def test_finebio_clip_config_supplies_views_and_targets() -> None:
    scope = contract.review_scope(
        Path("configs/clips/finebio_P03_01_01_1798-2398.json"), repository_root=ROOT
    )

    assert scope.views == ("T1", "T2", "T3", "T4", "T5", "fpv")
    assert "cell_culture_plate" in scope.targets and "chassis" not in scope.targets


def test_review_scope_rejects_a_config_without_views(tmp_path: Path) -> None:
    path = tmp_path / "clip.json"
    path.write_text(json.dumps({"targets": ["a"]}))
    with pytest.raises(ValueError, match="names no views"):
        contract.review_scope(path)
    path.write_text(json.dumps({"views": [], "targets": ["a"]}))
    with pytest.raises(ValueError, match="empty"):
        contract.review_scope(path)


def test_source_alignment_reads_the_approved_views_from_the_scope() -> None:
    from battle.fixtures import synthetic_run_manifest

    manifest = synthetic_run_manifest()
    ego_and_static = manifest.clip.model_copy(
        update={"views": ("static-c10379", "ego-hmc21110305")}
    )
    both = manifest.model_copy(update={"clip": ego_and_static})

    with pytest.raises(ValueError, match="approved view"):
        verify_source_alignment(both, both)
    verify_source_alignment(both, both, approved_views=("static-c10379", "ego-hmc21110305"))
    static_only = manifest.model_copy(
        update={"clip": manifest.clip.model_copy(update={"views": ("static-c10379",)})}
    )
    verify_source_alignment(static_only, static_only)
