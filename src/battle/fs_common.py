"""Standard-library file helpers shared by the Battle package and its external workers.

Five small functions that were copied into some thirty modules: a chunked SHA-256, the JSON
writer, the run-id timestamp, the repository-relative URI written into every manifest and the
`{revision, dirty}` record of a git checkout.  Consolidating them changes no bytes: the digest
is the digest, the JSON formatting keeps each caller's `indent` / `sort_keys`, and
`relative_uri` keeps the two forms the copies had.

This module imports only the standard library and stays Python 3.10 compatible (no
`datetime.UTC`, no `match`): the DAM4SAM / SAMURAI / WiLoR / LM-EEC / FineBio interpreters are
3.10.  Like `gpu_guard`, it is imported both as `battle.fs_common` (drivers, builders, tests)
and as a sibling top-level module by the workers that run under those interpreters, where
`battle` is not installed::

    try:
        from . import fs_common
    except ImportError:
        sys.path.append(str(Path(__file__).resolve().parent))
        import fs_common

Battle-venv code that hashes files it may hash again should prefer
`digest_cache.sha256_file`, which remembers digests against size and mtime and calls
:func:`sha256_file` here for the uncached read.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RUN_TIMESTAMP_FORMAT = "%Y%m%dt%H%M%Sz"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Return the hex SHA-256 of a file read in `chunk`-byte pieces."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def write_json(
    path: Path,
    payload: Any,
    *,
    indent: int | None = 2,
    sort_keys: bool = False,
    atomic: bool = False,
) -> None:
    """Write `payload` as JSON with a trailing newline.

    `indent` and `sort_keys` are passed to `json.dumps` so each caller keeps the bytes it
    has always written.  With `atomic` the text goes to `<path>.tmp` first and is renamed
    into place, so a reader polling the file never sees a partial document.
    """
    text = json.dumps(payload, indent=indent, sort_keys=sort_keys) + "\n"
    if atomic:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
        return
    path.write_text(text, encoding="utf-8")


def run_timestamp(now: datetime | None = None) -> str:
    """UTC timestamp for run ids, e.g. ``20260922t041239z`` (lowercase separators)."""
    # `datetime.UTC` is 3.11+; the worker interpreters are 3.10.
    return (now or datetime.now(timezone.utc)).strftime(RUN_TIMESTAMP_FORMAT)  # noqa: UP017


def relative_uri(path: Path, repository_root: Path, *, resolve: bool = True) -> str:
    """Repository-relative POSIX URI for persisted metadata; the path itself when outside.

    With `resolve` (the default, the form fourteen former copies used) both paths are made
    absolute with symlinks followed before the comparison, and a path outside the repository
    is returned absolute.  `resolve=False` compares the paths exactly as given and returns the
    given path when it is not under the root; that is the form `muggled_smoke.relative_uri`
    has always written and the seed and score modules import.
    """
    if resolve:
        path = path.resolve()
        repository_root = repository_root.resolve()
    try:
        return path.relative_to(repository_root).as_posix()
    except ValueError:
        return path.as_posix()


def git_revision(path: Path) -> dict[str, Any]:
    """`{"revision": <HEAD sha or "">, "dirty": <any porcelain status line>}` of a checkout.

    The form the FineBio scripts record for the Battle, MuggledSAM and MMDetection trees; a
    directory that is not a repository yields an empty revision and `dirty=False` rather than
    an error (`check=False`, stdout only).
    """

    def git(*parts: str) -> str:
        return subprocess.run(
            ["git", "-C", str(path), *parts], check=False, capture_output=True, text=True
        ).stdout.strip()

    return {"revision": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
