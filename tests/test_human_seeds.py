from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from battle import human_seeds as hs
from battle.multiview_schemas import SeedCandidate, SeedPrompt, SeedTransferPart
from battle.schemas import ArtifactFingerprint, PixelBox


def _write_mask(path: Path, shape=(72, 96), box=(10, 20, 40, 60)) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    y0, x0, y1, x1 = box
    mask[y0:y1, x0:x1] = True
    cv2.imwrite(str(path), mask.astype(np.uint8) * 255)
    return mask


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _b3_part(root: Path) -> SeedTransferPart:
    previous = root / "b3_seed.png"
    _write_mask(previous, box=(12, 22, 40, 60))
    return SeedTransferPart(
        target="chassis",
        kind="part",
        status="accepted",
        expected_area_px=1000.0,
        prompts=(
            SeedPrompt(
                prompt_id="t000000-b01",
                target="chassis",
                variant="x",
                pixel_box=PixelBox(x1=0, y1=0, x2=10, y2=10),
            ),
        ),
        accepted=SeedCandidate(
            prompt_id="t000000-b01",
            candidate_index=0,
            mask=ArtifactFingerprint(uri="b3_seed.png", sha256=_sha(previous), source="measured"),
            decoder_iou_estimate=0.9,
            mask_area_px=int((40 - 12) * (60 - 22)),
            sanity_pass=True,
            acceptance_basis="plane_warp_iou",
        ),
    )


def _proposal(root: Path, view: str = "C10119") -> tuple[dict, Path]:
    proposal_dir = root / "proposals" / view / "chassis_f000000"
    proposal_dir.mkdir(parents=True)
    candidates = []
    for index, box in enumerate(((10, 20, 40, 60), (5, 15, 45, 65))):
        path = proposal_dir / f"candidate_{index:02d}.png"
        mask = _write_mask(path, box=box)
        candidates.append(
            {
                "strategy": f"s{index}",
                "mask_uri": path.name,
                "sha256": _sha(path),
                "area_px": int(mask.sum()),
                "decoder_iou_estimate": 0.8 - 0.1 * index,
                "is_accepted_seed": False,
            }
        )
    record = {
        "view": view,
        "part": "chassis",
        "analysis_frame_index": 0,
        "candidates": candidates,
        "human_decision": None,
    }
    (proposal_dir / "proposal.json").write_text(json.dumps(record))
    cell = {
        "view": view,
        "part": "chassis",
        "analysis_frame_index": 0,
        "proposal": (proposal_dir / "proposal.json").relative_to(root).as_posix(),
        "candidates": [
            {
                "index": i,
                "strategy": c["strategy"],
                "mask_sha256": c["sha256"],
                "area_px": c["area_px"],
            }
            for i, c in enumerate(candidates)
        ],
        "decision": "accept",
        "accepted_candidate": 1,
        "note": "the second one is the part",
    }
    return cell, proposal_dir


def test_human_accepted_part_takes_the_chosen_candidate_with_provenance(tmp_path: Path) -> None:
    cell, proposal_dir = _proposal(tmp_path)
    decisions = ArtifactFingerprint(uri="configs/qa/d.json", sha256="a" * 64, source="measured")
    part = hs.human_accepted_part(
        _b3_part(tmp_path), cell, repository_root=tmp_path, decisions=decisions
    )
    assert part.status == "accepted" and part.provenance == "agent_proposed_human_accepted"
    assert part.accepted is not None
    assert part.accepted.candidate_index == 1
    assert part.accepted.acceptance_basis == "human_accepted_proposal"
    assert part.accepted.mask.uri == "proposals/C10119/chassis_f000000/candidate_01.png"
    assert part.accepted.mask_area_px == 40 * 50
    assert part.accepted.area_ratio_vs_expected == pytest.approx(2.0)
    notes = " ".join(part.accepted.sanity_notes)
    assert "candidate 1 (s1)" in notes and "configs/qa/d.json" in notes
    assert "human note: the second one is the part" in notes
    assert "replaces the B3 seed" in notes and "IoU with it 0." in notes
    # The prompt id of the B3 seed is kept so the worker payload stays well-formed.
    assert part.accepted.prompt_id == "t000000-b01"

    # A decisions file that disagrees with the proposal on the candidate's hash is refused.
    cell["candidates"][1]["mask_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="disagree"):
        hs.human_accepted_part(
            _b3_part(tmp_path), cell, repository_root=tmp_path, decisions=decisions
        )


def test_acceptance_counts_and_table() -> None:
    cells = [
        {"view": "A", "part": "chassis", "decision": "accept"},
        {"view": "A", "part": "chassis", "decision": None},
        {"view": "B", "part": "interior", "decision": "reject"},
        {"view": "B", "part": "interior", "decision": "unsure"},
    ]
    by_part = hs.acceptance_counts(cells, "part")
    assert by_part["chassis"] == {"accept": 1, "reject": 0, "unsure": 0, "undecided": 1, "cells": 2}
    assert by_part["interior"] == {
        "accept": 0,
        "reject": 1,
        "unsure": 1,
        "undecided": 0,
        "cells": 2,
    }
    assert hs.acceptance_counts(cells, "view")["B"]["cells"] == 2


def test_interior_candidates_need_area_and_luminance_in_band(tmp_path: Path) -> None:
    view_dir = tmp_path / "interior" / "C10001"
    (view_dir / "results" / "masks").mkdir(parents=True)
    (view_dir / "results" / "frames").mkdir(parents=True)
    frame = np.zeros((72, 96, 3), dtype=np.uint8)
    frame[:, :48] = 30  # the dark part on the left
    frame[:, 48:] = 120  # the hand on the right
    cv2.imwrite(str(view_dir / "results" / "frames" / "frame-000000.jpg"), frame)
    part_mask = _write_mask(view_dir / "results" / "masks" / "c0.png", box=(10, 5, 40, 40))
    hand_mask = _write_mask(view_dir / "results" / "masks" / "c1.png", box=(10, 50, 40, 90))
    tiny = _write_mask(view_dir / "results" / "masks" / "c2.png", box=(10, 5, 12, 8))
    decoded = {
        "t000000-b01": {
            "candidates": [
                {"candidate_index": 0, "iou_score": 0.6, "mask_uri": "results/masks/c0.png"},
                {"candidate_index": 1, "iou_score": 0.9, "mask_uri": "results/masks/c1.png"},
                {"candidate_index": 2, "iou_score": 0.7, "mask_uri": "results/masks/c2.png"},
            ]
        }
    }
    item = hs.InteriorViewPlan(
        view="C10001",
        view_id="static-c10001",
        analysis_frame_index=0,
        pose_frame_index=0,
        is_ego=False,
        proxy=ArtifactFingerprint(uri="p.mp4", sha256="b" * 64, source="approved_config"),
        proxy_dimensions=(96, 72),
        projected_centre_px=(24.0, 25.0),
        half_size_px=15.0,
        depth_mm=800.0,
        expected_area_px=float(part_mask.sum()),
        prompt=SeedPrompt(
            prompt_id="t000000-b01",
            target="interior",
            variant="v",
            pixel_box=PixelBox(x1=0, y1=0, x2=48, y2=48),
        ),
    )
    candidates = hs._load_candidates(tmp_path, item, decoded, view_dir, human_luminance=31.0)
    by_index = {c.candidate_index: c for c, _ in candidates}
    # Sorted by decoder score: the hand scores highest but is 4x brighter than the human masks.
    assert [c.candidate_index for c, _ in candidates] == [1, 2, 0]
    assert by_index[1].in_area_band and not by_index[1].in_luminance_band
    assert by_index[1].luminance_ratio_vs_human == pytest.approx(120 / 31)
    assert by_index[0].eligible and by_index[0].median_luminance == 30.0
    assert not by_index[2].in_area_band and by_index[2].in_luminance_band
    assert [c.candidate_index for c, _ in candidates if c.eligible] == [0]
    assert hand_mask.sum() == tiny.sum() * 200
    # Without a human luminance reference the appearance rule is inert.
    inert = hs._load_candidates(tmp_path, item, decoded, view_dir, human_luminance=None)
    assert all(c.in_luminance_band for c, _ in inert)


def test_median_luminance_requires_a_matching_frame() -> None:
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    frame = np.full((4, 4, 3), 200, dtype=np.uint8)
    assert hs.median_luminance(frame, mask) == 200.0
    assert hs.median_luminance(None, mask) is None
    assert hs.median_luminance(np.zeros((5, 4, 3), dtype=np.uint8), mask) is None
    assert hs.median_luminance(frame, np.zeros((4, 4), dtype=bool)) is None
