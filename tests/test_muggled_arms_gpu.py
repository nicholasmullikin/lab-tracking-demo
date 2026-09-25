"""Device-bound smokes of the two FineBio worker modes on the preflight window of P03_01_01.

Run with ``uv run pytest -q -m gpu tests/test_muggled_arms_gpu.py``.  Both tests stream 20
frames of the raw fpv video from raw frame 1798 (the preflight's SAM3 frame) with the four
detector boxes the preflight seeded (`runs/preflight-finebio-20260924/sam3_fpv/sam3_preflight.json`
-> `track.fpv.seeds`): plate, blue pipette, centrifuge, 50 ml tube.  Outputs go to `tmp_path`;
nothing from the video is committed (FineBio licence).  IoU here is between a mask's bounding
box and the prompt box, i.e. agreement with the FineBio detector, not accuracy.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from conftest import require_artifact

from battle.muggled_smoke import DEFAULT_MODEL, MUGGLED_SAM_PYTHON
from battle.schemas import MuggledSAMArmRunManifest

pytestmark = pytest.mark.gpu

VIDEO = Path("data/raw/finebio/finebio_videos_fpv_test/finebio_videos/P03_01_01.mp4")
PREFLIGHT = Path("runs/preflight-finebio-20260924/sam3_fpv/sam3_preflight.json")
START_FRAME = 1798
FRAMES = 20
MIN_BBOX_IOU = 0.6


def _seeds() -> list[dict]:
    require_artifact(PREFLIGHT)
    report = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    seeds = report["track"]["fpv"]["seeds"]
    assert [seed["class"] for seed in seeds] == [
        "cell_culture_plate",
        "blue_pipette",
        "centrifuge",
        "50ml_tube",
    ]
    return seeds


def _require_gpu_inputs() -> None:
    require_artifact(VIDEO)
    require_artifact(DEFAULT_MODEL)
    require_artifact(MUGGLED_SAM_PYTHON)


def _run(arguments: list[str], run_root: Path) -> tuple[Path, MuggledSAMArmRunManifest]:
    completed = subprocess.run(
        [
            str(Path(sys.executable).with_name("battle-muggled-arms")),
            *arguments,
            "--run-root",
            str(run_root),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    created = [path for path in run_root.iterdir() if path.is_dir()]
    assert len(created) == 1, (created, completed.stdout, completed.stderr)
    run = created[0]
    manifest = MuggledSAMArmRunManifest.model_validate_json(
        (run / "manifest.json").read_text(encoding="utf-8")
    )
    assert completed.returncode == 0, (
        completed.stdout,
        completed.stderr,
        (run / "worker.stderr.log").read_text(encoding="utf-8")[-4000:],
    )
    assert manifest.method_statuses[0].state.value == "succeeded"
    return run, manifest


def _rows(run: Path) -> dict[int, dict]:
    rows = {}
    for line in (run / "observations.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["analysis_frame_index"]] = row
    return rows


def _mask_bbox_iou(run: Path, item: dict, prompt_px: list[float], shape: tuple[int, int]) -> float:
    mask = cv2.imread(str(run / item["mask"]["uri"]), cv2.IMREAD_GRAYSCALE)
    assert mask is not None and mask.shape == shape
    ys, xs = np.nonzero(mask > 0)
    assert len(xs), "empty mask"
    mb = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    x0, y0, x1, y1 = prompt_px
    inter = max(min(mb[2], x1) - max(mb[0], x0), 0) * max(min(mb[3], y1) - max(mb[1], y0), 0)
    union = (mb[2] - mb[0]) * (mb[3] - mb[1]) + (x1 - x0) * (y1 - y0) - inter
    return float(inter / union)


def test_box_decode_arm_masks_all_four_preflight_boxes_on_frame_zero(tmp_path: Path) -> None:
    _require_gpu_inputs()
    seeds = _seeds()
    stream = tmp_path / "boxes.jsonl"
    with stream.open("w", encoding="utf-8") as handle:
        for frame in range(FRAMES):
            handle.write(
                json.dumps(
                    {
                        "frame_index": frame,
                        "boxes": [
                            {
                                "slot": slot,
                                "label": seed["class"],
                                "box_xyxy_px": seed["box_px"],
                                "score": seed["score"],
                                "source": "finebio_dino",
                            }
                            for slot, seed in enumerate(seeds)
                        ],
                    }
                )
                + "\n"
            )
    run_root = tmp_path / "runs"
    run_root.mkdir()

    run, manifest = _run(
        [
            "box-decode",
            "--video",
            str(VIDEO),
            "--view-id",
            "fpv",
            "--box-stream",
            str(stream),
            "--start-frame",
            str(START_FRAME),
            "--max-frames",
            str(FRAMES),
            "--max-side-length",
            "1280",
        ],
        run_root,
    )

    assert manifest.mode == "box_decode"
    assert manifest.concepts == tuple(seed["class"] for seed in seeds)
    assert manifest.observation_rows == FRAMES
    assert manifest.stream_identity is not None
    rows = _rows(run)
    assert sorted(rows) == list(range(FRAMES))
    first = rows[0]
    assert [item["object_id"] for item in first["objects"]] == [
        f"sam3-{slot:02d}" for slot in range(4)
    ]
    shape = (1440, 1920)
    ious = {}
    for slot, item in enumerate(first["objects"]):
        assert item["source"] == "sam3_decode"
        assert item["prompt_box"]["width"] > 0
        assert 0.0 <= item["object_score"] <= 1.0
        assert (run / item["mask"]["uri"]).is_file()
        ious[item["label"]] = _mask_bbox_iou(run, item, seeds[slot]["box_px"], shape)
    assert all(value >= MIN_BBOX_IOU for value in ious.values()), ious
    timing = json.loads(manifest.runtime_settings["box_decode_timing_ms"])
    assert timing["per_prompted_frame"]["n"] == FRAMES
    assert timing["per_prompted_frame"]["median"] > 0
    assert timing["per_box"]["median"] > 0
    encode = timing["image_encode"]["median"]
    decode = timing["decode_all_boxes"]["median"]
    print(
        f"BOX_DECODE ms/frame median {timing['per_prompted_frame']['median']:.0f} "
        f"(encode {encode:.0f}, decode {decode:.0f}), ms/box {timing['per_box']['median']:.0f}, "
        f"bbox IoU {ious}, peak VRAM {manifest.measurements.gpu_peak_vram_bytes / 2**30:.2f} GiB"
    )


def test_video_memory_arm_takes_box_seeds_a_mid_stream_start_and_the_tau_hook(
    tmp_path: Path,
) -> None:
    _require_gpu_inputs()
    seeds = _seeds()
    tube = seeds[3]
    pipette = seeds[1]
    height, width = 1440, 1920
    px = pipette["box_px"]
    schedule = {
        "seeds": [
            {
                "target": seed["class"],
                "initial_multiplex_slot": slot,
                "prompt_box_xyxy_px": seed["box_px"],
                "selected_by": "detector",
            }
            for slot, seed in enumerate(seeds[:3])
        ]
        + [
            {
                "target": tube["class"],
                "initial_multiplex_slot": 3,
                "start_frame": 5,
                "prompt_box_xyxy_px": tube["box_px"],
                "selected_by": "detector_reseed",
            }
        ],
        "corrections": [
            {
                "frame_index": 10,
                "multiplex_slot": 1,
                "target": pipette["class"],
                "prompt_box": {
                    "x": px[0] / width,
                    "y": px[1] / height,
                    "width": (px[2] - px[0]) / width,
                    "height": (px[3] - px[1]) / height,
                },
                "selected_by": "track_reproject",
            }
        ],
    }
    schedule_path = tmp_path / "schedule.json"
    schedule_path.write_text(json.dumps(schedule, indent=1), encoding="utf-8")
    run_root = tmp_path / "runs"
    run_root.mkdir()

    run, manifest = _run(
        [
            "video-memory",
            "--video",
            str(VIDEO),
            "--view-id",
            "fpv",
            "--schedule",
            str(schedule_path),
            "--start-frame",
            str(START_FRAME),
            "--max-frames",
            str(FRAMES),
            "--max-side-length",
            "1280",
            "--prompt-memory-semantics",
            "append",
            "--memory-write-min-score",
            "0.5",
            "--no-checkpoints",
        ],
        run_root,
    )

    assert manifest.mode == "video_memory"
    assert "pm-append" in manifest.run_id and "tau0p5" in manifest.run_id
    assert manifest.memory_policy is not None
    assert manifest.memory_policy.memory_write_min_score == 0.5
    assert manifest.multi_keyframe_corrections is not None
    assert manifest.multi_keyframe_corrections.slot_start_frames == ((3, 5),)
    assert manifest.multi_keyframe_corrections.detector_reseed_correction_frame_indices == (5,)
    assert manifest.multi_keyframe_corrections.track_reproject_correction_frame_indices == (10,)
    rows = _rows(run)
    assert sorted(rows) == list(range(FRAMES))
    frame0 = {item["object_id"]: item for item in rows[0]["objects"]}
    assert set(frame0) == {"sam3-00", "sam3-01", "sam3-02"}, "the tube slot starts at frame 5"
    assert all(0.0 < item["confidence"] <= 1.0 for item in frame0.values())
    for frame in range(1, 5):
        tube_diag = next(d for d in rows[frame]["tracker_diagnostics"] if d["multiplex_slot"] == 3)
        assert tube_diag["active"] is False
        assert tube_diag["memory_written"] is False
        assert tube_diag["memory_gate_reason"] == "unseeded"
        assert "sam3-03" not in {item["object_id"] for item in rows[frame]["objects"]}
    seeded = next(d for d in rows[5]["tracker_diagnostics"] if d["multiplex_slot"] == 3)
    assert seeded["corrected"] is True and seeded["seed_start"] is True
    assert seeded["selected_by"] == "detector_reseed"
    assert seeded["prompt_decoder_iou"] > 0.5
    tube_row = next(item for item in rows[5]["objects"] if item["object_id"] == "sam3-03")
    assert tube_row["source"] == "sam3_decode" and tube_row["prompt_box"]["width"] > 0
    assert _mask_bbox_iou(run, tube_row, tube["box_px"], (height, width)) >= MIN_BBOX_IOU
    reprompted = next(d for d in rows[10]["tracker_diagnostics"] if d["multiplex_slot"] == 1)
    assert reprompted["corrected"] is True and reprompted["selected_by"] == "track_reproject"
    tube_active_after = sum(
        1
        for frame in range(6, FRAMES)
        if next(d for d in rows[frame]["tracker_diagnostics"] if d["multiplex_slot"] == 3)["active"]
    )
    assert tube_active_after >= 10, tube_active_after
    tau_reasons = {
        d["memory_gate_reason"]
        for frame in range(1, FRAMES)
        for d in rows[frame]["tracker_diagnostics"]
    }
    assert tau_reasons <= {"ok", "low_object_score", "unseeded", "corrected"}, tau_reasons
    print(
        f"VIDEO_MEMORY elapsed {manifest.measurements.elapsed_seconds:.1f}s, tube active on "
        f"{tube_active_after}/14 frames after its start, gate reasons {sorted(tau_reasons)}, "
        f"peak VRAM {manifest.measurements.gpu_peak_vram_bytes / 2**30:.2f} GiB"
    )
