"""`battle.exemplar_pool`: reference filtering, the decoder-source switch and the pool metrics."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from battle import exemplar_pool as pool
from battle import mask_cache


def _rect(box: tuple[int, int, int, int], shape=(40, 60)) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[box[1] : box[3], box[0] : box[2]] = True
    return mask


def test_human_seed_provenance_filter() -> None:
    assert pool.is_human_seed_provenance("agent_proposed_human_accepted")
    assert pool.is_human_seed_provenance("human_seed_frame0")
    assert not pool.is_human_seed_provenance("geometric_from_two_human_views")
    assert not pool.is_human_seed_provenance("geometric_seed_transfer")


def test_run_seed_references_keep_human_seeds_only(tmp_path: Path) -> None:
    run = tmp_path / "run"
    (run / "masks").mkdir(parents=True)
    objects = []
    for slot, target in enumerate(("chassis", "interior")):
        uri = f"masks/000000_{slot:02d}.png"
        (run / uri).write_bytes(mask_cache.encode_rgba_mask_png(_rect((2, 2, 10, 10)), (1, 2, 3)))
        objects.append({"label": target, "mask": {"uri": uri}})
    (run / "observations.jsonl").write_text(
        json.dumps({"analysis_frame_index": 0, "objects": objects}) + "\n", encoding="utf-8"
    )
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "four_part_multiview": {
                    "seed_provenance": "mixed_per_part",
                    "seeds": [
                        {"target": "chassis", "provenance": "agent_proposed_human_accepted"},
                        {"target": "interior", "provenance": "geometric_from_two_human_views"},
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    references = pool.run_seed_references(run, tmp_path)
    assert [r["target"] for r in references] == ["chassis"]
    assert references[0]["provenance"] == "run_seed:agent_proposed_human_accepted"
    assert references[0]["fingerprint"]["uri"] == "run/masks/000000_00.png"


def test_decoder_for_source_switch(tmp_path: Path) -> None:
    assert (
        pool.decoder_for_source(
            "image_decoder",
            proxy_path=tmp_path / "p.mp4",
            results_directory=tmp_path,
            stderr_path=tmp_path / "e.log",
            references=None,
        )
        is None
    )
    with pytest.raises(ValueError):
        pool.decoder_for_source(
            "exemplar_detector",
            proxy_path=tmp_path / "p.mp4",
            results_directory=tmp_path,
            stderr_path=tmp_path / "e.log",
            references=None,
        )
    with pytest.raises(ValueError):
        pool.decoder_for_source(
            "nope",
            proxy_path=tmp_path / "p.mp4",
            results_directory=tmp_path,
            stderr_path=tmp_path / "e.log",
            references=tmp_path / "r.json",
        )


def _cell(frame: int, candidates: list[pool.PoolCandidate]) -> pool.PoolCell:
    return pool.PoolCell(
        frame=frame,
        part="chassis",
        truth_area_px=100,
        expected_area_px=100.0,
        projected_radius_px=10.0,
        centroid_proxy_px=(50.0, 50.0),
        candidates=tuple(candidates),
    )


def _candidate(
    score: float,
    iou: float,
    *,
    area: int = 100,
    centroid=(52.0, 50.0),
    inside=True,
    pool_name="exemplar:same_view:posneg",
):
    return pool.PoolCandidate(
        pool=pool_name,
        key=f"{score}",
        score=score,
        area_px=area,
        centroid_px=centroid,
        iou_vs_truth=iou,
        inside_box=inside,
    )


def test_accept_from_pool_filters_then_ranks_by_score() -> None:
    good = _candidate(0.6, 0.8)
    higher_but_far = _candidate(0.9, 0.1, centroid=(90.0, 90.0))
    too_big = _candidate(0.95, 0.2, area=1000)
    outside = _candidate(0.99, 0.0, inside=False)
    choice = pool.accept_from_pool(
        [good, higher_but_far, too_big, outside],
        expected_area_px=100.0,
        centroid_proxy_px=(50.0, 50.0),
        projected_radius_px=10.0,
    )
    assert choice is good
    assert (
        pool.accept_from_pool(
            [too_big], expected_area_px=100.0, centroid_proxy_px=None, projected_radius_px=None
        )
        is None
    )
    assert (
        pool.accept_from_pool(
            [good],
            expected_area_px=100.0,
            centroid_proxy_px=None,
            projected_radius_px=None,
            min_score=0.7,
        )
        is None
    )


def test_pool_metrics_and_leave_frames_out() -> None:
    cells = [
        _cell(0, [_candidate(0.8, 0.9), _candidate(0.2, 0.3)]),
        _cell(1, [_candidate(0.7, 0.2)]),  # accepted and harmful
        _cell(2, [_candidate(0.1, 0.95)]),  # low score: abstains under a floor
        _cell(3, []),
        _cell(4, [_candidate(0.9, 0.7)]),
        _cell(5, [_candidate(0.85, 0.65)]),
    ]
    metrics = pool.pool_metrics(cells, "exemplar:same_view")
    assert metrics["cells"] == 6 and metrics["accepted"] == 5 and metrics["harm"] == 1
    assert metrics["oracle_cells_at_least_0.6"] == 4
    assert metrics["oracle_mean_iou"] == pytest.approx((0.9 + 0.2 + 0.95 + 0 + 0.7 + 0.65) / 6)
    assert metrics["accepted_mean_iou"] == pytest.approx((0.9 + 0.2 + 0.95 + 0.7 + 0.65) / 5)
    assert pool.pool_metrics(cells, "decoder")["accepted"] == 0
    held = pool.leave_frames_out_pool(
        cells, "exemplar:same_view", frames=[0, 1, 2, 3, 4, 5], held_out=2
    )
    assert held["folds"] == 3 and held["held_out_cells"] == 6
    assert held["held_out_accepted"] <= 5
    assert not pool.meets_bar({"held_out_mean_iou": 0.59, "held_out_harm_rate": 0.0})
    assert pool.meets_bar({"held_out_mean_iou": 0.61, "held_out_harm_rate": 0.1})
    assert not pool.meets_bar({"held_out_mean_iou": 0.7, "held_out_harm_rate": 0.2})


def test_exemplar_candidates_for_cell_reads_kept_masks(tmp_path: Path) -> None:
    (tmp_path / "detections").mkdir()
    truth = _rect((10, 10, 30, 30))
    hit = _rect((12, 10, 32, 30))
    miss = _rect((40, 0, 58, 20))
    cv2.imwrite(str(tmp_path / "detections" / "a.png"), hit.astype(np.uint8) * 255)
    cv2.imwrite(str(tmp_path / "detections" / "b.png"), miss.astype(np.uint8) * 255)
    rows = [
        {
            "frame": 5,
            "target": "chassis",
            "set": "same_view",
            "variant": "posneg",
            "reference_frames_used": [0, 327],
            "masks": [
                {
                    "index": 0,
                    "score": 0.9,
                    "area_px": int(hit.sum()),
                    "centroid_px": [21.5, 19.5],
                    "uri": "detections/a.png",
                },
                {
                    "index": 1,
                    "score": 0.4,
                    "area_px": int(miss.sum()),
                    "centroid_px": [48.5, 9.5],
                    "uri": "detections/b.png",
                },
            ],
        }
    ]
    candidates, used = pool.exemplar_candidates_for_cell(
        tmp_path,
        rows,
        frame=5,
        part="chassis",
        set_name="same_view",
        variant="posneg",
        truth_mask=truth,
        gate_box=(0, 0, 35, 35),
    )
    assert used == (0, 327) and len(candidates) == 2
    assert candidates[0].iou_vs_truth == pytest.approx(18 / 22) and candidates[0].inside_box
    assert candidates[1].iou_vs_truth == 0.0 and not candidates[1].inside_box
