from __future__ import annotations

import pytest

from battle.build_phases import PhaseTimer
from battle.interaction_review_v4 import LAYERS, build_first_minute_review


def test_phases_accumulate_and_report_the_unmeasured_remainder() -> None:
    timer = PhaseTimer("build", enabled=False)

    with timer.phase("validate"):
        pass
    timer.start("export")
    timer.stop("export")
    with timer.phase("validate"):
        pass

    assert set(timer.elapsed) == {"validate", "export"}
    report = timer.report()
    assert report.startswith("build: total ")
    assert "validate" in report and "export" in report and "other" in report


def test_stopping_a_phase_that_never_started_is_ignored() -> None:
    timer = PhaseTimer("build", enabled=False)

    timer.stop("never-opened")

    assert timer.elapsed == {}


def test_a_disabled_timer_prints_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    PhaseTimer("build", enabled=False).print_report()
    assert capsys.readouterr().out == ""

    PhaseTimer("build", enabled=True).print_report()
    assert "build: total" in capsys.readouterr().out


def test_an_unknown_review_layer_is_refused_before_any_work(tmp_path) -> None:
    with pytest.raises(ValueError, match="unknown review layers"):
        build_first_minute_review(
            repository_root=tmp_path,
            output_root=tmp_path / "out",
            layers=("stabilized_wilor", "not-a-layer"),
        )


def test_every_named_layer_is_unique_and_non_empty() -> None:
    assert len(set(LAYERS)) == len(LAYERS)
    assert all(name and name.strip() == name for name in LAYERS)
