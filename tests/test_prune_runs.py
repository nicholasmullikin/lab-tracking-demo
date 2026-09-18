from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "prune_runs.py"
_spec = importlib.util.spec_from_file_location("prune_runs", MODULE_PATH)
assert _spec and _spec.loader
prune_runs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prune_runs)


def test_only_names_absent_from_tracked_text_are_unreferenced() -> None:
    names = {"run-a", "run-b", "run-c"}
    text = "See runs/run-a for the baseline, and runs/run-c/manifest.json."

    assert prune_runs.referenced_runs(text, names) == {"run-a", "run-c"}


def test_a_run_cited_only_by_another_runs_manifest_is_reported_separately(tmp_path: Path) -> None:
    for name in ("keeper", "citer", "orphan"):
        (tmp_path / name).mkdir()
    (tmp_path / "citer" / "manifest.json").write_text(
        '{"input": "runs/keeper/observations.jsonl"}', encoding="utf-8"
    )

    citations = prune_runs.referencing_manifests(tmp_path, {"keeper", "citer", "orphan"})

    assert citations["keeper"] == {"citer"}
    assert citations["orphan"] == set()
    # A manifest naming its own directory is not a citation from elsewhere.
    assert citations["citer"] == set()


def test_directory_bytes_sums_only_files(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "a.bin").write_bytes(b"12345")
    (tmp_path / "b.bin").write_bytes(b"678")

    assert prune_runs.directory_bytes(tmp_path) == 8
