"""argparse fragments and the output-directory rule the Battle CLIs share.

Every `main()` stays its own entry point (the argument sets differ on purpose); what moves
here is the handful of lines they all wrote the same way: the `--repository-root` option
defaulting to the current directory, the `--output-root` / `--overwrite` / `--quiet` trio of
the review builders, and the mkdir-or-refuse rule for a non-empty output directory.  The
error text of :func:`open_output_directory` is the one most builders already raised; a caller
whose message a test pins passes `message=`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

OVERWRITE_HELP = "Replace an existing package in --output-root."
QUIET_HELP = "Suppress the per-phase timing report."
REPOSITORY_ROOT_HELP = (
    "Checkout the configs/ and runs/ paths are relative to (default: the current directory)."
)


def add_repository_root(
    parser: argparse.ArgumentParser, *, default: Path | None = None
) -> argparse.Action:
    """`--repository-root PATH`, defaulting to the current working directory."""
    return parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path.cwd() if default is None else default,
        help=REPOSITORY_ROOT_HELP,
    )


def add_output_root(
    parser: argparse.ArgumentParser, default: Path | None, *, help: str | None = None
) -> argparse.Action:
    """`--output-root PATH` with the caller's default (None when the builder derives it)."""
    return parser.add_argument("--output-root", type=Path, default=default, help=help)


def add_output_flags(
    parser: argparse.ArgumentParser,
    *,
    overwrite_help: str = OVERWRITE_HELP,
    quiet: bool = True,
    quiet_help: str = QUIET_HELP,
) -> None:
    """`--overwrite` and (by default) `--quiet`, both store-true."""
    parser.add_argument("--overwrite", action="store_true", help=overwrite_help)
    if quiet:
        parser.add_argument("--quiet", action="store_true", help=quiet_help)


def open_output_directory(path: Path, *, overwrite: bool, message: str | None = None) -> Path:
    """Create `path` (parents included) and return it.

    An existing directory that already holds entries is refused with `FileExistsError`
    unless `overwrite`; an existing empty directory never needs the flag.  Nothing inside is
    deleted here: the caller replaces the files it owns.
    """
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            message if message is not None else f"{path} exists; pass --overwrite to replace it"
        )
    path.mkdir(parents=True, exist_ok=True)
    return path
