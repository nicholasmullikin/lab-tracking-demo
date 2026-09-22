"""The shared argparse fragments and the output-directory rule."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from battle import cli_common


def test_repository_root_defaults_to_the_current_directory(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    parser = argparse.ArgumentParser()
    cli_common.add_repository_root(parser)
    assert parser.parse_args([]).repository_root == Path.cwd() == tmp_path
    assert parser.parse_args(["--repository-root", "/r"]).repository_root == Path("/r")
    explicit = argparse.ArgumentParser()
    cli_common.add_repository_root(explicit, default=Path("/elsewhere"))
    assert explicit.parse_args([]).repository_root == Path("/elsewhere")


def test_output_root_and_flags_parse_like_the_hand_written_options() -> None:
    parser = argparse.ArgumentParser()
    cli_common.add_output_root(parser, Path("runs/x"))
    cli_common.add_output_flags(parser)
    args = parser.parse_args([])
    assert (args.output_root, args.overwrite, args.quiet) == (Path("runs/x"), False, False)
    args = parser.parse_args(["--output-root", "runs/y", "--overwrite", "--quiet"])
    assert (args.output_root, args.overwrite, args.quiet) == (Path("runs/y"), True, True)
    derived = argparse.ArgumentParser()
    cli_common.add_output_root(derived, None, help="default: derived")
    cli_common.add_output_flags(derived, quiet=False, overwrite_help="Replace the run dir.")
    assert derived.parse_args([]).output_root is None
    assert not hasattr(derived.parse_args([]), "quiet")
    assert "Replace the run dir." in derived.format_help()


def test_open_output_directory_accepts_empty_refuses_occupied_unless_overwrite(
    tmp_path: Path,
) -> None:
    fresh = tmp_path / "a" / "b"
    assert cli_common.open_output_directory(fresh, overwrite=False) == fresh
    assert fresh.is_dir()
    assert cli_common.open_output_directory(fresh, overwrite=False) == fresh
    (fresh / "manifest.json").write_text("{}")
    with pytest.raises(FileExistsError, match="pass --overwrite to replace it"):
        cli_common.open_output_directory(fresh, overwrite=False)
    with pytest.raises(FileExistsError, match="^custom text$"):
        cli_common.open_output_directory(fresh, overwrite=False, message="custom text")
    assert cli_common.open_output_directory(fresh, overwrite=True) == fresh
    assert (fresh / "manifest.json").exists()
