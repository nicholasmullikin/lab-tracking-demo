"""Side-by-side reading of two consensus + hull builds (label-free)."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import require_artifact

from battle import multiview_build_comparison as mbc

ROOT = Path(__file__).resolve().parents[1]


def test_merge_joins_touching_and_overlapping_intervals() -> None:
    assert mbc._merge([(585, 598), (599, 604), (598, 599), (643, 658), (650, 660)]) == (
        (585, 604),
        (643, 660),
    )
    assert mbc._merge([]) == ()


@pytest.mark.real_data
def test_comparison_of_the_sep_18_builds_reproduces_the_ledger_numbers(tmp_path: Path) -> None:
    before = require_artifact(ROOT / "runs/multiview-part-consensus-first-minute-with-e4")
    after = require_artifact(ROOT / "runs/multiview-part-consensus-first-minute-policy")
    hull_before = require_artifact(ROOT / "runs/multiview-visual-hull-first-minute-with-e4")
    hull_after = require_artifact(ROOT / "runs/multiview-visual-hull-first-minute-policy")
    comparison = mbc.compare_builds(
        repository_root=ROOT,
        consensus_before=before.relative_to(ROOT),
        consensus_after=after.relative_to(ROOT),
        hull_before=hull_before.relative_to(ROOT),
        hull_after=hull_after.relative_to(ROOT),
        label_before="with-e4",
        label_after="policy",
    )
    # Numbers the Sep 18 ledger entry recorded for the policy build.
    assert comparison.consensus_after.reference_contradicted["chassis"].frames == 228
    assert comparison.consensus_after.episodes_total == 99
    assert comparison.consensus_after.episodes_reference == 15
    assert comparison.hull_after is not None
    assert comparison.hull_after.episodes_total == 146
    assert comparison.hull_after.episodes_reference == 32
    assert comparison.hull_after.reference_median_iou["chassis"] == pytest.approx(0.396, abs=0.001)
    markdown = mbc.render_markdown(comparison)
    assert "| C10379 chassis contradicted by the majority |" in markdown
    assert "not accuracy" in markdown
    output = tmp_path / "summary.md"
    assert (
        mbc.main(
            [
                "--repository-root",
                str(ROOT),
                "--consensus-before",
                str(before.relative_to(ROOT)),
                "--consensus-after",
                str(after.relative_to(ROOT)),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert output.is_file() and output.with_suffix(".json").is_file()
