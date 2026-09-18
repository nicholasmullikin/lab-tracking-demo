from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from battle import digest_cache
from battle.muggled_smoke import _bounded_video_stamp_path, _reusable_bounded_video


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BATTLE_DIGEST_CACHE", str(tmp_path / "digests.json"))
    digest_cache.clear()
    yield
    digest_cache.clear()


def test_digest_matches_a_direct_hash_and_is_reused(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"first")
    expected = hashlib.sha256(b"first").hexdigest()

    assert digest_cache.sha256_file(path) == expected
    assert Path(os.environ["BATTLE_DIGEST_CACHE"]).is_file()

    # A cached hit must not read the bytes again, so unreadable content still resolves.
    path.chmod(0o000)
    try:
        assert digest_cache.sha256_file(path) == expected
    finally:
        path.chmod(0o644)


def test_a_rewritten_file_is_hashed_again(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"first")
    digest_cache.sha256_file(path)

    path.write_bytes(b"second changed length")
    assert digest_cache.sha256_file(path) == hashlib.sha256(b"second changed length").hexdigest()


def test_verify_ignores_the_cache(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"first")
    stale = digest_cache.sha256_file(path)
    status = path.stat()

    # Rewrite in place, restoring size and mtime so only re-reading can notice.
    path.write_bytes(b"secon")
    os.utime(path, ns=(status.st_atime_ns, status.st_mtime_ns))

    assert digest_cache.sha256_file(path) == stale
    assert digest_cache.sha256_file(path, verify=True) == hashlib.sha256(b"secon").hexdigest()


def test_the_cache_survives_a_new_session(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"persisted")
    expected = digest_cache.sha256_file(path)

    digest_cache.clear()
    path.chmod(0o000)
    try:
        assert digest_cache.sha256_file(path) == expected
    finally:
        path.chmod(0o644)


def test_bounded_video_is_reused_only_for_the_same_request(tmp_path: Path) -> None:
    proxy = tmp_path / "proxy.mp4"
    proxy.write_bytes(b"proxy bytes")
    output = tmp_path / "input_600f.mp4"
    output.write_bytes(b"trimmed")
    status = proxy.stat()
    _bounded_video_stamp_path(output).write_text(
        json.dumps(
            {
                "source": str(proxy.resolve()),
                "source_size_bytes": status.st_size,
                "source_mtime_ns": status.st_mtime_ns,
                "frame_count": 600,
            }
        ),
        encoding="utf-8",
    )

    assert _reusable_bounded_video(proxy, output, 600) is True
    assert _reusable_bounded_video(proxy, output, 300) is False

    proxy.write_bytes(b"different proxy bytes")
    assert _reusable_bounded_video(proxy, output, 600) is False


def test_a_missing_or_unreadable_stamp_forces_a_re_encode(tmp_path: Path) -> None:
    proxy = tmp_path / "proxy.mp4"
    proxy.write_bytes(b"proxy bytes")
    output = tmp_path / "input_600f.mp4"
    output.write_bytes(b"trimmed")

    assert _reusable_bounded_video(proxy, output, 600) is False

    _bounded_video_stamp_path(output).write_text("{not json", encoding="utf-8")
    assert _reusable_bounded_video(proxy, output, 600) is False
