from __future__ import annotations

from pathlib import Path

import pytest

from battle.four_part_comparison import _coverage, _render
from battle.four_part_contract import TARGETS, load_contract
from battle.schemas import ArtifactFingerprint, FourPartTargetInitialization, FrameObservations


def test_reviewed_seed_contract_has_exact_fingerprinted_target_order() -> None:
    root = Path(__file__).parents[1]

    contract = load_contract(root, Path("configs/four_part_segmentation_comparison.json"))

    assert tuple(seed.target_id for seed in contract.seeds) == TARGETS
    assert tuple(seed.box_xyxy for seed in contract.seeds) == (
        (793, 468, 955, 549),
        (845, 254, 909, 313),
        (773, 572, 844, 628),
        (997, 504, 1179, 632),
    )


def test_missing_targets_are_coverage_gaps_not_persisted_masks() -> None:
    observations = {
        0: FrameObservations(
            view_id="static-c10379",
            analysis_frame_index=0,
            source_seconds=294.0,
        )
    }

    assert _coverage(observations) == dict.fromkeys(TARGETS, 0)


def test_synthetic_alignment_maps_every_frame_to_source_clock() -> None:
    observations = {
        index: FrameObservations(
            view_id="static-c10379",
            analysis_frame_index=index,
            source_seconds=294.0 + index / 30,
        )
        for index in (0, 300, 599)
    }

    assert [observation.source_seconds for observation in observations.values()] == [
        294.0,
        304.0,
        313.96666666666664,
    ]


def test_open_vocabulary_schema_rejects_a_reviewed_seed() -> None:
    with pytest.raises(ValueError, match="cannot claim a reviewed mask"):
        FourPartTargetInitialization(
            target_id="chassis",
            source="open_vocabulary_detection",
            state="succeeded",
            reviewed_mask_fingerprint=ArtifactFingerprint(
                uri="runs/example/mask.png", sha256="a" * 64, source="measured"
            ),
        )


def test_rrd_render_clears_a_missing_frame(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    logged: list[tuple[str, object]] = []
    monkeypatch.setattr(
        "battle.four_part_comparison.rr.log",
        lambda path, value, **_: logged.append((path, value)),
    )
    observation = FrameObservations(
        view_id="static-c10379",
        analysis_frame_index=9,
        source_seconds=294.3,
    )

    _render(
        root="world/fixture/four_part_segmentation_comparison",
        method_id="samurai",
        run_directory=tmp_path,
        observation=observation,
        dimensions=(1280, 720),
    )

    assert logged[0][0].endswith("/methods/samurai/render")
    assert logged[0][1].__class__.__name__ == "Clear"
