"""`battle.worker_common`: the workers' shared frame extraction and peak-VRAM read."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from battle import worker_common


def test_extract_frames_writes_indexed_jpegs(tmp_path: Path, synthetic_video: Path) -> None:
    frames = worker_common.extract_frames(synthetic_video, tmp_path / "frames", 3)

    assert [path.name for path in frames] == ["00000.jpg", "00001.jpg", "00002.jpg"]
    assert all(path.is_file() and path.stat().st_size > 0 for path in frames)


def test_extract_frames_exact_versus_at_least(tmp_path: Path, synthetic_video: Path) -> None:
    with pytest.raises(RuntimeError, match="video produced 3 frames; expected exactly 5"):
        worker_common.extract_frames(synthetic_video, tmp_path / "exact", 5)
    frames = worker_common.extract_frames(synthetic_video, tmp_path / "loose", 5, exact=False)
    assert len(frames) == 3
    assert len(worker_common.extract_frames(synthetic_video, tmp_path / "two", 2)) == 2


def test_extract_frames_unreadable_video(tmp_path: Path) -> None:
    missing = tmp_path / "missing.mp4"
    with pytest.raises(RuntimeError, match="could not open video"):
        worker_common.extract_frames(missing, tmp_path / "a", 3)
    # The four-part worker's form: no open check, the short decode is the error.
    with pytest.raises(RuntimeError, match="video produced 0 frames; expected exactly 3"):
        worker_common.extract_frames(missing, tmp_path / "b", 3, check_open=False)
    with pytest.raises(RuntimeError, match="video produced zero frames"):
        worker_common.extract_frames(missing, tmp_path / "c", 3, check_open=False, exact=False)


def test_cuda_peak_bytes_reads_max_memory_allocated(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []

    def max_memory_allocated(device: object = None) -> float:
        calls.append(device)
        return 42.0

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(max_memory_allocated=max_memory_allocated)),
    )
    assert worker_common.cuda_peak_bytes() == 42
    assert worker_common.cuda_peak_bytes("cuda:0") == 42
    assert calls == [None, "cuda:0"]


def test_module_imports_only_the_standard_library_at_top_level() -> None:
    source = Path(worker_common.__file__).read_text(encoding="utf-8")
    top_level_imports = [
        line for line in source.splitlines() if line.startswith(("import ", "from "))
    ]
    assert top_level_imports == ["from __future__ import annotations", "from pathlib import Path"]
