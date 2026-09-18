"""Device-bound check that a resumed tracker stream is not an approximation.

Run with `uv run pytest -m gpu`. It needs the CUDA device, the local SAM3 checkpoint,
and the approved static proxy, and it streams the fixed 300-frame smoke twice.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import require_artifact

from battle.muggled_smoke import DEFAULT_MODEL, MUGGLED_SAM_PYTHON

pytestmark = pytest.mark.gpu

RESUME_AT = 100


def _smoke(run_root: Path, *extra: str) -> Path:
    before = set(run_root.iterdir()) if run_root.is_dir() else set()
    completed = subprocess.run(
        [
            str(Path(sys.executable).with_name("battle-muggled-smoke")),
            "--view",
            "static-c10379",
            "--run-root",
            str(run_root),
            *extra,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Wrote MuggledSAM/SAM3 run" in completed.stdout, completed.stdout
    created = sorted(set(run_root.iterdir()) - before)
    assert len(created) == 1, created
    return created[0]


def _frames(run: Path) -> dict[int, dict]:
    rows = (run / "observations.jsonl").read_text(encoding="utf-8").splitlines()
    return {json.loads(row)["analysis_frame_index"]: json.loads(row) for row in rows if row.strip()}


def _mask_digests(run: Path, row: dict) -> tuple[str, ...]:
    return tuple(
        sorted(
            hashlib.sha256((run / item["mask"]["uri"]).read_bytes()).hexdigest()
            for item in row["objects"]
            if item.get("mask")
        )
    )


def test_a_resumed_stream_reproduces_the_continuous_one_exactly(tmp_path: Path) -> None:
    require_artifact(DEFAULT_MODEL)
    require_artifact(MUGGLED_SAM_PYTHON)

    baseline = _smoke(tmp_path, "--checkpoint-every", str(RESUME_AT))
    checkpoint = baseline / "native" / "checkpoints" / f"f{RESUME_AT:06d}.pt"
    assert checkpoint.is_file(), sorted((baseline / "native" / "checkpoints").iterdir())

    resumed = _smoke(tmp_path, "--resume-run", str(baseline), "--resume-at", str(RESUME_AT))

    original, replayed = _frames(baseline), _frames(resumed)
    assert set(original) == set(replayed)
    for frame in sorted(original):
        assert _mask_digests(baseline, original[frame]) == _mask_digests(
            resumed, replayed[frame]
        ), f"masks differ at frame {frame}"
        assert original[frame]["objects"] == replayed[frame]["objects"], (
            f"observations differ at frame {frame}"
        )

    continuity = json.loads((resumed / "manifest.json").read_text())["smoke"]["continuity"]
    assert continuity["mode"] == "checkpoint_resumed"
    assert continuity["resumed_at_frame"] == RESUME_AT
    assert continuity["resumed_from_run"].endswith(baseline.name)
