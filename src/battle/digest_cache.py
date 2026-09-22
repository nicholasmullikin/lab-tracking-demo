"""Content digests for local artifacts, remembered between runs.

Every builder re-hashes the same approved proxies, bounded videos, configs, and mask
trees on each invocation. Those inputs are large and almost always unchanged, so the
digest is cached against each file's size and modification time and recomputed when
either moves.

The cache is an optimisation, never an authority: `sha256_file(..., verify=True)` reads
the bytes again, and any builder that must prove an artifact is unchanged should use it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .fs_common import sha256_file as _hash_bytes

CACHE_RELATIVE_PATH = Path(".cache/battle/file-digests.json")
_MEMORY: dict[str, tuple[int, int, str]] = {}
_DISK_LOADED = False
_DISK_DIRTY = False


def _cache_path() -> Path:
    override = os.environ.get("BATTLE_DIGEST_CACHE")
    if override:
        return Path(override)
    return Path.cwd() / CACHE_RELATIVE_PATH


def _load_disk() -> None:
    global _DISK_LOADED
    if _DISK_LOADED:
        return
    _DISK_LOADED = True
    path = _cache_path()
    if not path.is_file():
        return
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    for key, entry in stored.items():
        if key in _MEMORY:
            continue
        try:
            _MEMORY[key] = (int(entry["size"]), int(entry["mtime_ns"]), str(entry["sha256"]))
        except (KeyError, TypeError, ValueError):
            continue


def save() -> None:
    """Persist newly measured digests, tolerating a read-only or racing filesystem."""
    global _DISK_DIRTY
    if not _DISK_DIRTY:
        return
    path = _cache_path()
    payload = {
        key: {"size": size, "mtime_ns": mtime_ns, "sha256": digest}
        for key, (size, mtime_ns, digest) in _MEMORY.items()
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, indent=0, sort_keys=True), encoding="utf-8")
        temporary.replace(path)
    except OSError:
        return
    _DISK_DIRTY = False


def sha256_file(path: Path, *, verify: bool = False) -> str:
    """Return a file's SHA-256, reusing a cached digest unless size or mtime moved."""
    global _DISK_DIRTY
    resolved = path.resolve()
    status = resolved.stat()
    key = str(resolved)
    if not verify:
        _load_disk()
        cached = _MEMORY.get(key)
        if cached is not None and cached[0] == status.st_size and cached[1] == status.st_mtime_ns:
            return cached[2]
    digest = _hash_bytes(resolved)
    _MEMORY[key] = (status.st_size, status.st_mtime_ns, digest)
    _DISK_DIRTY = True
    save()
    return digest


def clear() -> None:
    """Forget every remembered digest, including the on-disk copy for this session."""
    global _DISK_LOADED, _DISK_DIRTY
    _MEMORY.clear()
    _DISK_LOADED = False
    _DISK_DIRTY = False
