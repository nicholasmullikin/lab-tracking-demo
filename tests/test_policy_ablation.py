from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from battle import policy_ablation as pa


def _write_run(
    root: Path,
    name: str,
    *,
    frames: int,
    shift: int = 0,
    gated_frames: set[int] | None = None,
    overlap_from: int | None = None,
) -> Path:
    """A tiny four-part run: squares that move right by `shift`; optional gate diagnostics."""
    run = root / name / f"muggledsam-sam3-four-part-static-focused-reassembly-{name}"
    (run / "masks").mkdir(parents=True)
    rows = []
    for frame in range(frames):
        objects = []
        diagnostics = []
        for slot, target in enumerate(pa.TARGETS):
            mask = np.zeros((24, 32), dtype=bool)
            x0 = 2 + 7 * slot + shift
            if overlap_from is not None and frame >= overlap_from and target == "interior":
                x0 = 2 + shift  # sits on the chassis square
            mask[4:10, x0 : x0 + 5] = True
            if target == "cabin" and frame == frames - 1:
                mask[:, :] = False
                mask[0:20, 0:30] = True  # a sudden growth for the area proxy
            uri = f"masks/{frame:06d}_{slot:02d}.png"
            Image.fromarray(mask.astype(np.uint8) * 255).save(run / uri)
            objects.append(
                {
                    "object_id": f"sam3-{slot:02d}",
                    "label": target,
                    "confidence": 1.0,
                    "box": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2},
                    "mask": {"uri": uri, "storage": "external_artifact", "format": "png"},
                }
            )
            if gated_frames is not None and frame > 0:
                gated = frame in gated_frames and target == "chassis"
                diagnostics.append(
                    {
                        "label": target,
                        "multiplex_slot": slot,
                        "object_score": 3.0,
                        "active": True,
                        "corrected": False,
                        "contested_fraction": 0.3 if gated else 0.0,
                        "memory_written": not gated,
                        "memory_gate_reason": "contested" if gated else "ok",
                    }
                )
        rows.append(
            {
                "analysis_frame_index": frame,
                "objects": objects,
                "tracker_diagnostics": diagnostics,
            }
        )
    (run / "observations.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "four_part_focused": {
                    "multi_keyframe_corrections": {
                        "scheduled_correction_frame_indices": [3],
                        "frame_zero_seeds_only": False,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (run / "worker_result.json").write_text(
        json.dumps(
            {
                "state": "succeeded",
                "frames_processed": frames,
                "masks_written": 4 * frames,
                "elapsed_seconds": 12.5,
                "time_to_first_usable_output_seconds": 1.5,
                "gpu_peak_vram_bytes": 2**31,
                "runtime_settings": {
                    "max_side_length": 1008,
                    "tracker_memory_policy": {"slot_exclusivity": "argmax", "memory_gate": "on"},
                },
            }
        ),
        encoding="utf-8",
    )
    return run


def test_area_jump_frames_flag_departures_from_the_rolling_median_only_briefly() -> None:
    steady = [100] * 40
    assert pa.area_jump_frames(steady) == []
    stepped = [100] * 35 + [300] * 40
    jumps = pa.area_jump_frames(stepped)
    assert jumps[0] == 35
    # After a window of the new size the median has moved and the trace settles.
    assert max(jumps) < 35 + pa.AREA_HISTORY
    assert pa.area_jump_frames([None] * 5 + [100] * 30 + [10]) == [35]


def test_window_bookkeeping() -> None:
    assert pa.in_any_window(600) and not pa.in_any_window(1000)
    counts = pa._window_counts([300, 600, 1100, 1500])
    assert counts == {"279-408": 1, "573-722": 1, "1020-1172": 1, "outside": 1}


def test_analysis_measures_disagreement_overlap_and_gating(tmp_path: Path) -> None:
    arms_root = tmp_path / "arms"
    reference_dir = _write_run(tmp_path / "ref", "reference", frames=12)
    identical = _write_run(arms_root, "same", frames=12, gated_frames={2, 3, 4})
    shifted = _write_run(arms_root, "shifted", frames=12, shift=2, overlap_from=6)

    assert pa.discover_arms(tmp_path) == {"same": identical, "shifted": shifted}
    reference = pa.load_run("reference", reference_dir, frame_count=12)
    same = pa.analyse_arm(pa.load_run("same", identical, frame_count=12), reference, frame_count=12)
    other = pa.analyse_arm(
        pa.load_run("shifted", shifted, frame_count=12), reference, frame_count=12
    )

    assert same["disagreement_vs_reference"]["chassis"]["mean_iou"] == 1.0
    assert same["episodes_vs_reference"]["count"] == 0
    assert same["pairwise_slot_iou"]["frames_with_any_pair_over_threshold"] == 0
    # A two-pixel shift of a five-pixel square: IoU 3/7 on every frame, one episode per part.
    assert other["disagreement_vs_reference"]["chassis"]["mean_iou"] == pytest.approx(3 / 7)
    assert other["episodes_vs_reference"]["count"] == 4
    assert other["pairwise_slot_iou"]["chassis_interior_frames"] == 6
    assert other["pairwise_slot_iou"]["frames_per_pair"] == {"chassis/interior": 6}
    assert other["area_jump_proxy"]["frames_per_part"]["cabin"] == 0  # history too short

    gate = pa.gate_statistics(pa.load_run("same", identical, frame_count=12), frame_count=12)
    chassis = gate["per_slot"]["chassis"]
    assert chassis["frames_gated"] == 3 and chassis["reasons"] == {"contested": 3, "ok": 8}
    assert chassis["gated_fraction"] == pytest.approx(3 / 11)
    assert gate["starved_slots"] == []
    assert gate["per_slot"]["cabin"]["frames_gated"] == 0
    untracked = pa.gate_statistics(reference, frame_count=12)
    assert untracked["per_slot"]["chassis"]["gated_fraction"] is None

    measures = pa.run_measures(identical)
    assert measures["mask_grid_side"] == 252 and measures["slot_exclusivity"] == "argmax"
    assert measures["scheduled_correction_frames"] == [3]


def test_summary_and_tables_round_trip(tmp_path: Path) -> None:
    reference_dir = _write_run(tmp_path / "ref", "reference", frames=8)
    _write_run(tmp_path / "root" / "arms", "arm-a", frames=8, gated_frames={1})

    summary = pa.summarize(
        repository_root=tmp_path,
        root=Path("root"),
        reference_directory=reference_dir.relative_to(tmp_path),
        frame_count=8,
    )

    written = json.loads((tmp_path / "root" / "summary.json").read_text(encoding="utf-8"))
    assert written["ground_truth_accuracy_claim"] is False
    assert written["arms"]["arm-a"]["disagreement_vs_reference"]["cabin"]["mean_iou"] == 1.0
    assert written["arms"]["arm-a"]["gate"]["per_slot"]["chassis"]["frames_gated"] == 1
    text = (tmp_path / "root" / "summary.md").read_text(encoding="utf-8")
    assert text == pa.markdown_tables(summary)
    assert "| arm-a | 1008 | x+g | full |" in text
    assert "## Pre-accuracy measures" in text
    assert "| 252 |" in text


def test_contact_sheets_render_every_arm_and_the_pair(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    reference_dir = _write_run(tmp_path / "ref", "reference", frames=6)
    _write_run(tmp_path / "root" / "arms", "arm-a", frames=6, shift=1, gated_frames={2})
    video = tmp_path / "proxy.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (32, 24))
    for _ in range(6):
        writer.write(np.full((24, 32, 3), 90, dtype=np.uint8))
    writer.release()

    written = pa.render_sheets(
        repository_root=tmp_path,
        root=Path("root"),
        reference_directory=reference_dir.relative_to(tmp_path),
        proxy_path=video.relative_to(tmp_path),
        best_arm="arm-a",
        frames=(1, 2, 4),
        tile_width=64,
    )

    names = sorted(path.name for path in written)
    assert names == ["arm-a.png", "best_vs_reference.png", "reference.png"]
    pair = cv2.imread(str(tmp_path / "root" / "best_vs_reference.png"))
    assert pair is not None and pair.shape[1] == 128
