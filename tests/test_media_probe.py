from __future__ import annotations

import json
from pathlib import Path

import pytest

from battle import media_probe


def _probe_result(_path: Path) -> media_probe.VideoInfo:
    return (1800, 30, (1280, 720))


def test_the_first_probe_is_cached_beside_the_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "input.mp4"
    video.write_bytes(b"video bytes")
    calls: list[Path] = []

    def counted(path: Path) -> media_probe.VideoInfo:
        calls.append(path)
        return _probe_result(path)

    monkeypatch.setattr(media_probe, "_probe", counted)

    assert media_probe.video_info(video) == (1800, 30, (1280, 720))
    assert media_probe.video_info(video) == (1800, 30, (1280, 720))
    assert media_probe.video_frame_count(video) == 1800
    assert len(calls) == 1

    sidecar = json.loads((tmp_path / "input.mp4.probe.json").read_text())
    assert sidecar["frame_count"] == 1800
    assert sidecar["size_bytes"] == video.stat().st_size


def test_a_rewritten_video_is_probed_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "input.mp4"
    video.write_bytes(b"video bytes")
    results = iter([(1800, 30, (1280, 720)), (600, 30, (1280, 720))])
    monkeypatch.setattr(media_probe, "_probe", lambda _path: next(results))

    assert media_probe.video_frame_count(video) == 1800
    video.write_bytes(b"a shorter clip")
    assert media_probe.video_frame_count(video) == 600


def test_verify_ignores_the_sidecar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "input.mp4"
    video.write_bytes(b"video bytes")
    calls: list[Path] = []
    monkeypatch.setattr(
        media_probe, "_probe", lambda path: (calls.append(path), _probe_result(path))[1]
    )

    media_probe.video_info(video)
    media_probe.video_info(video, verify=True)

    assert len(calls) == 2


def test_a_corrupt_sidecar_is_replaced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "input.mp4"
    video.write_bytes(b"video bytes")
    (tmp_path / "input.mp4.probe.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(media_probe, "_probe", _probe_result)

    assert media_probe.video_frame_count(video) == 1800
    assert json.loads((tmp_path / "input.mp4.probe.json").read_text())["fps"] == 30
