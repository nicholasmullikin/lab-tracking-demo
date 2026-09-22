"""`battle.detector_scorecard_v2` on a synthetic run, anchor set and appearance pass."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from battle import detector_scorecard_v2 as v2
from battle import review_anchors
from battle.review_anchors import ReviewAnchorMask, ReviewAnchorMaskSet, ReviewAnchorProvenance
from battle.schemas import VideoDimensions

TARGETS = ("chassis", "interior", "rear_body", "cabin")
SHAPE = (48, 64)
FRAMES = (10, 20, 30)
FRAME_COUNT = 40


def _square(x0: int, y0: int, size: int = 8) -> np.ndarray:
    mask = np.zeros(SHAPE, dtype=bool)
    mask[y0 : y0 + size, x0 : x0 + size] = True
    return mask


def _write_mask(path: Path, mask: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)


def _anchor_set(root: Path) -> dict[tuple[int, str], np.ndarray | None]:
    anchors = []
    truth: dict[tuple[int, str], np.ndarray | None] = {}
    for frame in FRAMES:
        for column, target in enumerate(TARGETS):
            if frame == 30 and target == "rear_body":
                anchors.append(
                    ReviewAnchorMask(
                        analysis_frame_index=frame,
                        target=target,
                        expected_visible="visible",
                        state="hidden",
                        selected_by="human",
                    )
                )
                truth[(frame, target)] = None
                continue
            mask = _square(column * 14, 4)
            path = root / "anchors" / "masks" / f"f{frame:06d}_{target}.png"
            _write_mask(path, mask)
            truth[(frame, target)] = mask
            anchors.append(
                ReviewAnchorMask(
                    analysis_frame_index=frame,
                    target=target,
                    expected_visible="visible",
                    state="labeled",
                    mask_uri=path.relative_to(root).as_posix(),
                    mask_sha256=review_anchors._sha256_bytes(path.read_bytes()),
                    area_pixels=int(mask.sum()),
                    selected_by="human",
                )
            )
    counts = {"labeled": 0, "hidden": 0, "unlabeled": 0}
    for anchor in anchors:
        counts[anchor.state] += 1
    mask_set = ReviewAnchorMaskSet(
        manifest_kind="human_review_anchor_masks",
        config={"uri": "configs/qa/x.json", "sha256": "1" * 64, "source": "measured"},
        calibration_manifest={"uri": "runs/x/c.json", "sha256": "2" * 64, "source": "measured"},
        view_id="ego-hmc21179183",
        targets=TARGETS,
        mask_dimensions=VideoDimensions(width=SHAPE[1], height=SHAPE[0]),
        windows={"279-408": (15, 25)},
        anchors=tuple(anchors),
        counts=counts,  # type: ignore[arg-type]
        exported_at=datetime.now(UTC),
        claim_boundary=review_anchors.CLAIM_BOUNDARY,
        license=review_anchors.LICENSE,
        provenance=ReviewAnchorProvenance(),
    )
    (root / review_anchors.MASK_SET_NAME).write_text(mask_set.model_dump_json(indent=2))
    return truth


# Which cells the synthetic run gets wrong: (frame, target) -> shift in px (None = mask missing).
WRONG = {(20, "chassis"): 30, (30, "interior"): None}
HIDDEN_FP = (30, "rear_body")


def _run(root: Path) -> Path:
    run = root / "run"
    (run / "masks").mkdir(parents=True)
    rows = []
    for frame in range(FRAME_COUNT):
        objects = []
        for slot, target in enumerate(TARGETS):
            key = (frame, target)
            if key in WRONG and WRONG[key] is None:
                continue
            shift = WRONG.get(key, 0) or 0
            size = 8
            if target == "rear_body" and frame >= 28:
                shift, size = 20, 20  # the screwdriver: > 300 px where the human says hidden
            mask = _square(slot * 14 + shift, 4, size)
            uri = f"masks/{frame:06d}_{slot:02d}.png"
            _write_mask(run / uri, mask)
            objects.append(
                {
                    "label": target,
                    "mask": {"uri": uri},
                    "object_score": 8.0 - (2.0 if key in WRONG else 0.0),
                }
            )
        rows.append({"analysis_frame_index": frame, "objects": objects})
    (run / "observations.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (run / "manifest.json").write_text("{}")
    return run


def _pass(root: Path, *, reference_frame: int = 10) -> Path:
    """Embeddings: each part has its own direction; wrong cells drift toward another part.

    Detections: best_overlap_iou 0.9 on good cells, 0.1 on wrong ones, presence low on hidden.
    """
    out = root / "pass"
    (out / "native").mkdir(parents=True)
    rng = np.random.default_rng(0)
    directions = rng.normal(size=(4, 32))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    embeddings = np.zeros((FRAME_COUNT, 4, 32), dtype=np.float16)
    has_mask = np.ones((FRAME_COUNT, 4), dtype=bool)
    for frame in range(FRAME_COUNT):
        for column, target in enumerate(TARGETS):
            key = (frame, target)
            if key in WRONG and WRONG[key] is None:
                has_mask[frame, column] = False
                continue
            vector = directions[column] + 0.05 * rng.normal(size=32)
            if key in WRONG or (target == "rear_body" and frame >= 28):
                vector = directions[(column + 1) % 4] + 0.05 * rng.normal(size=32)
            embeddings[frame, column] = (vector / np.linalg.norm(vector)).astype(np.float16)
    refs = [
        ("same_view", reference_frame, t, "positive", directions[i]) for i, t in enumerate(TARGETS)
    ]
    refs.append(("same_view", 35, "rear_body", "distractor:screwdriver", directions[3]))
    np.savez(
        out / "native" / "embeddings.npz",
        frames=np.arange(FRAME_COUNT),
        targets=np.asarray(TARGETS),
        embeddings=embeddings,
        has_mask=has_mask,
        areas=np.full((FRAME_COUNT, 4), 64),
        reference_keys=np.asarray([f"{s}|{f}|{t}|{r}" for (s, f, t, r, _) in refs]),
        reference_sets=np.asarray([r[0] for r in refs]),
        reference_frames=np.asarray([r[1] for r in refs]),
        reference_targets=np.asarray([r[2] for r in refs]),
        reference_roles=np.asarray([r[3] for r in refs]),
        reference_embeddings=np.stack([r[4] for r in refs]).astype(np.float16),
    )
    (out / "manifest.json").write_text(
        json.dumps({"reference_sets": [{"name": "same_view", "view": "HMC_21179183"}]})
    )
    with (out / "detections.jsonl").open("w") as handle:
        for frame in range(FRAME_COUNT):
            for target in TARGETS:
                key = (frame, target)
                wrong = key in WRONG or (target == "rear_body" and frame >= 28)
                for variant in ("pos", "posneg"):
                    handle.write(
                        json.dumps(
                            {
                                "frame": frame,
                                "target": target,
                                "set": "same_view",
                                "reference_view": "HMC_21179183",
                                "variant": variant,
                                "reference_frames_used": [reference_frame]
                                if frame != reference_frame
                                else [],
                                "excluded_reference_frame": reference_frame
                                if frame == reference_frame
                                else None,
                                "tracked_area_px": 0
                                if (key in WRONG and WRONG[key] is None)
                                else 64,
                                "presence": 0.2
                                if (target == "rear_body" and frame >= 28)
                                else 0.95,
                                "top_score": 0.3 if wrong else 0.9,
                                "top_centroid_distance_px": 50.0 if wrong else 2.0,
                                "top_iou_tracked": None
                                if (key in WRONG and WRONG[key] is None)
                                else (0.1 if wrong else 0.9),
                                "best_overlap_iou": None
                                if (key in WRONG and WRONG[key] is None)
                                else (0.1 if wrong else 0.9),
                            }
                        )
                        + "\n"
                    )
    return out


@pytest.fixture
def scored(tmp_path: Path) -> tuple[v2.RecordScore, Path, Path]:
    _anchor_set(tmp_path / "anchors")
    run = _run(tmp_path)
    pass_dir = _pass(tmp_path)
    spec = v2.RecordSpec(
        name="e4-test",
        view="HMC_21179183",
        view_id="ego-hmc21179183",
        run=run,
        pass_dir=pass_dir,
        anchors=tmp_path / "anchors",
        hidden_class="out_of_frame",
        frame_count=FRAME_COUNT,
    )
    return v2.build_record(spec, repository_root=tmp_path), tmp_path, pass_dir


def test_record_truth_classes_and_detectors(scored) -> None:
    record, _, _ = scored
    cells = {(c["frame"], c["target"]): c for c in record.cells}
    assert len(cells) == 12
    assert cells[(20, "chassis")]["failed"] and cells[(30, "interior")]["failed"]
    assert (
        cells[(30, "rear_body")]["failed"] and cells[(30, "rear_body")]["anchor_state"] == "hidden"
    )
    assert cells[(30, "rear_body")]["failure_class"] == "out_of_frame"
    assert cells[(20, "chassis")]["failure_class"] == "occlusion_leak"
    assert cells[(10, "cabin")]["failure_class"] == "outside"
    assert record.reference_frames == (10,)
    assert cells[(10, "chassis")]["reference_frame"]
    # Leave-reference-out: the reference frame's own embeddings have no same-view reference left.
    assert cells[(10, "chassis")]["detectors"]["emb_self"] is None
    assert cells[(20, "chassis")]["detectors"]["emb_self"] is not None
    assert (
        cells[(20, "chassis")]["detectors"]["emb_swap"]
        > cells[(20, "cabin")]["detectors"]["emb_swap"]
    )
    assert cells[(20, "chassis")]["detectors"]["det_iou"] == pytest.approx(0.9)
    assert cells[(30, "interior")]["detectors"]["det_iou"] is None
    assert cells[(30, "rear_body")]["detectors"]["det_presence"] == pytest.approx(0.8)
    assert cells[(30, "rear_body")]["detectors"]["det_disagree"] == 1.0
    overall = {s.detector: s for s in record.scores if s.subset == "all"}
    assert overall["det_iou"].auroc == 1.0 and overall["det_iou"].undefined == 1
    assert overall["det_presence"].positives == 3
    assert overall["det_disagree"].auroc == 1.0
    assert overall["emb_self_xv"].cells == 0
    assert record.class_counts["out_of_frame"] == {"cells": 1, "failed": 1}
    assert len(record.top) == 3


def test_outputs_and_named_tests(scored) -> None:
    record, root, _ = scored
    tests = v2.named_tests(
        [record],
        [
            {
                "label": "hidden",
                "record": "e4-test",
                "target": "rear_body",
                "anchor_state": "hidden",
            },
            {"label": "missing record", "record": "nope", "target": "chassis"},
        ],
    )
    assert tests[0]["frame"] == 30 and tests[0]["failed"]
    assert "det_presence" in tests[0]["flagged_top_decile"]
    assert tests[1]["status"] == "record not scored"
    spec_path = root / "spec.json"
    spec_path.write_text("{}")
    path = v2.write_outputs(
        records=[record],
        output_dir=root / "out",
        repository_root=root,
        spec_path=spec_path,
        tests=tests,
    )
    text = path.read_text()
    assert "Named tests" in text and "`det_iou`" in text and "out_of_frame" in text
    payload = json.loads((root / "out" / "scorecard_v2.json").read_text())
    assert payload["records"][0]["class_counts"]["all"]["cells"] == 12
    assert payload["leave_one_record_out"] == []
    rows = (root / "out" / "confidence_v2_e4-test.jsonl").read_text().splitlines()
    assert len(rows) == FRAME_COUNT * 4
    first = json.loads(rows[0])
    assert set(first["detectors"]) == set(v2.ALL_DETECTORS)


def test_pooled_and_leave_one_record_out(scored, tmp_path: Path) -> None:
    record, root, pass_dir = scored
    other = v2.RecordSpec(
        name="e4-copy",
        view="HMC_21179183",
        view_id="ego-hmc21179183",
        run=record.spec.run,
        pass_dir=pass_dir,
        anchors=root / "anchors",
        hidden_class="out_of_frame",
        frame_count=FRAME_COUNT,
    )
    second = v2.build_record(other, repository_root=root)
    overall, per_class = v2.pooled_scores([record, second])
    det_iou = next(s for s in overall if s.detector == "det_iou")
    assert det_iou.positives == 4 and det_iou.auroc == 1.0
    assert set(per_class) == {"outside", "occlusion_leak", "out_of_frame"}
    loro = v2.leave_one_record_out([record, second])
    assert [item["held_out_record"] for item in loro] == ["e4-test", "e4-copy"]
    assert loro[0]["recall_floor"]["tp"] >= 2


def test_classify_cell_rules() -> None:
    windows = {"279-408": (279, 408), "573-722": (573, 722)}
    assert (
        v2.classify_cell(300, anchor_state="labeled", windows=windows, hidden_class=None)
        == "occlusion_leak"
    )
    assert (
        v2.classify_cell(600, anchor_state="labeled", windows=windows, hidden_class=None)
        == "rotation_swap"
    )
    assert (
        v2.classify_cell(900, anchor_state="labeled", windows=windows, hidden_class=None)
        == "outside"
    )
    assert (
        v2.classify_cell(1500, anchor_state="labeled", windows=windows, hidden_class=None)
        == "distractor"
    )
    assert (
        v2.classify_cell(1700, anchor_state="hidden", windows=windows, hidden_class=None)
        == "distractor"
    )
    assert (
        v2.classify_cell(304, anchor_state="hidden", windows=windows, hidden_class="out_of_frame")
        == "out_of_frame"
    )
