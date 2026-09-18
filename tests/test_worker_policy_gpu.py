"""Device-bound check that the tracker memory policy, switched off, changes nothing.

Run with `uv run pytest -m gpu tests/test_worker_policy_gpu.py`.  It streams the focused
C10379 first minute (policy off, 720 px, the reference correction schedule) and requires the
observation rows and mask PNG bytes of frames [0, 1800) to equal the reference run's.  Set
`BATTLE_POLICY_REGRESSION_RUN_ROOT` to a directory holding exactly one already-made run of that
condition to compare it instead of streaming again.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import require_artifact

from battle.muggled_smoke import DEFAULT_MODEL, MUGGLED_SAM_PYTHON

pytestmark = pytest.mark.gpu

REFERENCE_RUN = Path(
    "runs/muggledsam-sam3-four-part-static-focused-reassembly-static-c10379-20260918t001210z"
)
REFERENCE_SCHEDULE = Path(
    "runs/muggledsam-sam3-four-part-focused-corrections-agent-swap-20260918t000947z/"
    "multi_keyframe_correction_schedule.json"
)
FOCUSED_CONFIG = Path("configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_g2.json")
FIRST_MINUTE = 1800


def _rows(run: Path, limit: int) -> dict[int, str]:
    rows: dict[int, str] = {}
    for line in (run / "observations.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        frame = json.loads(line)["analysis_frame_index"]
        if frame < limit:
            rows[frame] = line
    return rows


def _policy_off_first_minute_run(tmp_path: Path) -> Path:
    configured = os.environ.get("BATTLE_POLICY_REGRESSION_RUN_ROOT")
    if configured:
        candidates = [path for path in Path(configured).iterdir() if path.is_dir()]
        assert len(candidates) == 1, candidates
        return candidates[0]
    require_artifact(DEFAULT_MODEL)
    require_artifact(MUGGLED_SAM_PYTHON)
    completed = subprocess.run(
        [
            str(Path(sys.executable).with_name("battle-muggled-smoke")),
            "--config",
            str(FOCUSED_CONFIG),
            "--view",
            "static-c10379",
            "--four-part-static-focused",
            "--max-frames",
            str(FIRST_MINUTE),
            "--max-side-length",
            "720",
            "--multi-keyframe-correction-schedule",
            str(REFERENCE_SCHEDULE),
            "--run-root",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Wrote MuggledSAM/SAM3 run" in completed.stdout, completed.stdout
    created = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert len(created) == 1, created
    return created[0]


def test_policy_off_reproduces_the_reference_first_minute_byte_for_byte(tmp_path: Path) -> None:
    require_artifact(REFERENCE_RUN / "observations.jsonl")
    require_artifact(REFERENCE_SCHEDULE)

    run = _policy_off_first_minute_run(tmp_path)

    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    metadata = manifest["four_part_focused"]
    assert metadata["requested_analysis_frame_range"]["end_frame_exclusive"] == FIRST_MINUTE
    assert metadata["multi_keyframe_corrections"]["dropped_correction_frame_indices"] == [
        1800,
        2700,
    ]
    policy = json.loads(metadata["runtime_settings"]["tracker_memory_policy"])
    assert policy["slot_exclusivity"] == "off" and policy["memory_gate"] == "off"

    expected, actual = _rows(REFERENCE_RUN, FIRST_MINUTE), _rows(run, FIRST_MINUTE)
    assert sorted(actual) == list(range(FIRST_MINUTE))
    differing = [frame for frame in range(FIRST_MINUTE) if expected[frame] != actual[frame]]
    assert differing == [], f"observation rows differ on {len(differing)} frames: {differing[:10]}"
    mask_differences = []
    for frame in range(FIRST_MINUTE):
        for item in json.loads(expected[frame])["objects"]:
            uri = item["mask"]["uri"]
            if (REFERENCE_RUN / uri).read_bytes() != (run / uri).read_bytes():
                mask_differences.append(uri)
    assert mask_differences == [], f"{len(mask_differences)} masks differ: {mask_differences[:10]}"
