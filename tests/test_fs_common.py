"""`battle.fs_common`: the stdlib-only helpers shared with the external workers."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from battle import digest_cache, fs_common

MODULE_PATH = Path(fs_common.__file__)


def test_module_imports_only_the_standard_library() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= sys.stdlib_module_names, imported - sys.stdlib_module_names


def test_module_parses_as_python_3_10_for_the_worker_interpreters() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"), feature_version=(3, 10))
    # `datetime.UTC` is the one 3.11+ name a 3.12 codebase reaches for by habit.
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "UTC" not in names | attributes | imported


@pytest.mark.parametrize(
    "interpreter",
    [
        Path("/home/nick/.pyenv/versions/samurai/bin/python"),
        Path("/home/nick/.pyenv/versions/muggled_sam/bin/python"),
    ],
)
def test_module_imports_under_a_worker_interpreter(interpreter: Path, tmp_path: Path) -> None:
    if not interpreter.is_file():
        pytest.skip(f"{interpreter} is not installed on this machine")
    script = (
        "import sys; sys.path.insert(0, sys.argv[1]); import fs_common; "
        "print(fs_common.run_timestamp(), fs_common.sha256_file(sys.argv[2]))"
    )
    blob = tmp_path / "blob.bin"
    blob.write_bytes(b"x" * 10)
    completed = subprocess.run(
        [str(interpreter), "-c", script, str(MODULE_PATH.parent), str(blob)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert completed.returncode == 0, completed.stderr
    stamp, digest = completed.stdout.split()
    assert len(stamp) == 16 and digest == hashlib.sha256(b"x" * 10).hexdigest()


def test_module_imports_as_a_sibling_without_the_battle_package(tmp_path: Path) -> None:
    script = (
        "import sys; sys.path.insert(0, sys.argv[1]); import fs_common; "
        "print(fs_common.run_timestamp.__name__)"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, str(MODULE_PATH.parent)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "run_timestamp"


def test_sha256_file_matches_hashlib_for_any_chunk_size(tmp_path: Path) -> None:
    path = tmp_path / "blob.bin"
    payload = bytes(range(256)) * 5000  # 1.28 MB, crosses the 1 MiB chunk boundary
    path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()

    assert fs_common.sha256_file(path) == expected
    assert fs_common.sha256_file(path, chunk=1 << 22) == expected
    assert fs_common.sha256_file(path, chunk=7) == expected
    assert digest_cache.sha256_file(path, verify=True) == expected


def test_write_json_keeps_each_callers_bytes(tmp_path: Path) -> None:
    payload = {"b": [1, 2], "a": {"y": None, "x": 1.5}}
    plain = tmp_path / "plain.json"
    sorted_keys = tmp_path / "sorted.json"

    fs_common.write_json(plain, payload)
    fs_common.write_json(sorted_keys, payload, sort_keys=True)

    assert plain.read_bytes() == (json.dumps(payload, indent=2) + "\n").encode("utf-8")
    assert sorted_keys.read_bytes() == (
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def test_write_json_atomic_replaces_through_a_tmp_sibling(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("old", encoding="utf-8")

    fs_common.write_json(path, {"k": 1}, atomic=True)

    assert json.loads(path.read_text(encoding="utf-8")) == {"k": 1}
    assert path.read_text(encoding="utf-8") == '{\n  "k": 1\n}\n'
    assert not path.with_suffix(".tmp").exists()


def test_run_timestamp_is_the_lowercase_compact_utc_form() -> None:
    now = datetime(2026, 9, 22, 4, 12, 39, tzinfo=UTC)

    assert fs_common.run_timestamp(now) == "20260922t041239z"
    # The two spellings the copies used produce the same string.
    assert fs_common.run_timestamp(now) == f"{now:%Y%m%dt%H%M%Sz}"
    assert fs_common.run_timestamp(now) == now.strftime("%Y%m%dT%H%M%SZ").lower()
    assert len(fs_common.run_timestamp()) == 16


def test_relative_uri_resolving_form(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "runs" / "x").mkdir(parents=True)
    inside = root / "runs" / "x" / "manifest.json"
    inside.write_text("{}", encoding="utf-8")
    outside = tmp_path / "elsewhere.json"
    outside.write_text("{}", encoding="utf-8")

    assert fs_common.relative_uri(inside, root) == "runs/x/manifest.json"
    assert fs_common.relative_uri(root / "runs" / ".." / "runs" / "x" / "manifest.json", root) == (
        "runs/x/manifest.json"
    )
    assert fs_common.relative_uri(outside, root) == outside.resolve().as_posix()


def test_relative_uri_unresolved_form_compares_paths_as_given(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    given = Path("runs/x/manifest.json")

    # As `muggled_smoke.relative_uri` always did: a relative path is returned as written.
    assert fs_common.relative_uri(given, root, resolve=False) == "runs/x/manifest.json"
    assert fs_common.relative_uri(root / given, root, resolve=False) == "runs/x/manifest.json"
    assert fs_common.relative_uri(tmp_path / "other.json", root, resolve=False) == (
        (tmp_path / "other.json").as_posix()
    )
