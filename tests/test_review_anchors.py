from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from battle import digest_cache, review_anchors
from battle.muggled_calibration import _write_manifest, build_manifest, frame_reference
from battle.review_anchors import (
    ReviewAnchorMask,
    ReviewAnchorMaskSet,
    ReviewAnchorProvenance,
    build_first_minute_config,
    export_anchor_masks,
    load_config,
    markdown_table,
    prepare_workspace,
    score_cell,
    score_runs,
)
from battle.schemas import (
    MuggledSAMBoxCalibrationManifest,
    MuggledSAMCalibrationCandidate,
    MuggledSAMCalibrationHiddenTarget,
    VideoDimensions,
)

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/qa/first_minute_review_anchors.json"
TARGETS = ("chassis", "interior", "rear_body", "cabin")


def _write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)


def _square(shape: tuple[int, int], x0: int, y0: int, size: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[y0 : y0 + size, x0 : x0 + size] = True
    return mask


# ---------------------------------------------------------------------------- configuration


def test_committed_anchor_config_is_the_thirteen_frame_list() -> None:
    config = load_config(CONFIG)
    # The builder leaves provenance null; the committed file carries the human's, filled in
    # after the Sep 19 labelling session. Everything else must still match the builder.
    assert config.model_copy(update={"provenance": ReviewAnchorProvenance()}) == (
        build_first_minute_config()
    )
    frames = tuple(frame.analysis_frame_index for frame in config.frames)
    assert frames == (300, 370, 400, 600, 650, 700, 900, 1050, 1100, 1150, 1200, 1500, 1700)
    assert config.targets == TARGETS
    for frame in config.frames:
        inside = 1024 <= frame.analysis_frame_index < 1172
        assert frame.expected_visible["interior"] == ("hidden_prompt" if inside else "visible")
        assert all(frame.expected_visible[t] == "visible" for t in TARGETS if t != "interior")
        assert frame.proxy_seconds == pytest.approx(frame.analysis_frame_index / 30)
        assert frame.source_seconds == pytest.approx(294 + frame.analysis_frame_index / 30)
    assert config.frames_in_window((279, 408)) == (300, 370, 400)
    assert config.frames_in_window((573, 722)) == (600, 650, 700)
    assert config.frames_in_window((1020, 1172)) == (1050, 1100, 1150)
    assert config.anchor_kind == "human_review_anchor"
    assert "not a dataset" in config.claim_boundary and "not ground truth" in config.claim_boundary
    assert "CC BY-NC 4.0" in config.license
    assert config.provenance.author and config.provenance.reviewed_at is not None
    assert config.provenance.tool == "battle-muggled-calibration-web"


def test_anchor_config_rejects_inconsistent_clocks_and_hidden_prompts() -> None:
    config = build_first_minute_config()
    payload = json.loads(config.model_dump_json())
    payload["frames"][1]["source_seconds"] += 1.0
    with pytest.raises(ValueError, match="source_seconds"):
        review_anchors.ReviewAnchorConfig.model_validate(payload)
    payload = json.loads(config.model_dump_json())
    payload["frames"][0]["expected_visible"]["interior"] = "hidden_prompt"
    with pytest.raises(ValueError, match="hidden_prompt"):
        review_anchors.ReviewAnchorConfig.model_validate(payload)


# ------------------------------------------------------------------------ cell-level scoring


def _anchor(state: str, frame: int = 300, target: str = "chassis") -> ReviewAnchorMask:
    if state == "labeled":
        return ReviewAnchorMask(
            analysis_frame_index=frame,
            target=target,
            expected_visible="visible",
            state="labeled",
            mask_uri="anchors/masks/x.png",
            mask_sha256="0" * 64,
            area_pixels=16,
            selected_by="human",
        )
    return ReviewAnchorMask(
        analysis_frame_index=frame,
        target=target,
        expected_visible="hidden_prompt" if state == "hidden" else "visible",
        state=state,  # type: ignore[arg-type]
    )


def test_score_cell_perfect_and_disjoint_masks() -> None:
    shape = (12, 12)
    anchor_mask = _square(shape, 0, 0, 4)
    perfect = score_cell(_anchor("labeled"), anchor_mask, anchor_mask.copy(), run_mask_uri="m")
    assert perfect.outcome == "scored"
    assert perfect.iou == 1.0 and perfect.area_ratio == 1.0
    assert perfect.anchor_area == perfect.run_area == 16
    disjoint = score_cell(
        _anchor("labeled"), anchor_mask, _square(shape, 6, 6, 4), run_mask_uri="m"
    )
    assert disjoint.outcome == "scored"
    assert disjoint.iou == 0.0 and disjoint.area_ratio == 1.0
    larger = score_cell(_anchor("labeled"), anchor_mask, _square(shape, 0, 0, 8), run_mask_uri="m")
    assert larger.iou == pytest.approx(16 / 64) and larger.area_ratio == pytest.approx(4.0)


def test_score_cell_missing_run_mask_is_zero_and_flagged() -> None:
    anchor_mask = _square((12, 12), 0, 0, 4)
    missing = score_cell(_anchor("labeled"), anchor_mask, None, run_mask_uri=None)
    assert missing.outcome == "run_mask_missing"
    assert missing.iou == 0.0 and missing.run_area == 0 and missing.area_ratio == 0.0
    empty = score_cell(
        _anchor("labeled"), anchor_mask, np.zeros((12, 12), dtype=bool), run_mask_uri="m"
    )
    assert empty.outcome == "run_mask_missing"


def test_score_cell_hidden_anchor_correct_and_false_positive() -> None:
    correct = score_cell(_anchor("hidden"), None, None, run_mask_uri=None)
    assert correct.outcome == "hidden_correct" and correct.iou is None and correct.run_area == 0
    false_positive = score_cell(
        _anchor("hidden"), None, _square((12, 12), 2, 2, 5), run_mask_uri="m"
    )
    assert false_positive.outcome == "hidden_false_positive"
    assert false_positive.iou is None and false_positive.run_area == 25


def test_score_cell_unlabeled_anchor_is_skipped_but_counted() -> None:
    skipped = score_cell(_anchor("unlabeled"), None, _square((12, 12), 0, 0, 2), run_mask_uri="m")
    assert skipped.outcome == "unlabeled_skipped" and skipped.iou is None


def test_score_cell_resizes_a_run_mask_of_another_size() -> None:
    anchor_mask = _square((12, 12), 0, 0, 6)
    run_mask = _square((6, 6), 0, 0, 3)
    scored = score_cell(_anchor("labeled"), anchor_mask, run_mask, run_mask_uri="m")
    assert scored.resized_run_mask is True and scored.iou == 1.0


# --------------------------------------------------------------- run-level scoring fixture


def _synthetic_mask_set(root: Path, shape: tuple[int, int]) -> tuple[ReviewAnchorMaskSet, dict]:
    """Two frames x four parts: labeled cells, one hidden cell, one unlabeled cell."""
    masks_directory = root / "anchors" / "masks"
    anchors: list[ReviewAnchorMask] = []
    truth: dict[tuple[int, str], np.ndarray | None] = {}
    for frame, x0 in ((300, 0), (1100, 20)):
        for column, target in enumerate(TARGETS):
            if frame == 1100 and target == "interior":
                anchors.append(_anchor("hidden", frame, target))
                truth[(frame, target)] = None
                continue
            if frame == 1100 and target == "cabin":
                anchors.append(_anchor("unlabeled", frame, target))
                truth[(frame, target)] = None
                continue
            mask = _square(shape, x0 + column * 10, 0, 6)
            path = masks_directory / f"f{frame:06d}_{target}.png"
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
        calibration_manifest={
            "uri": "runs/x/calibration_manifest.json",
            "sha256": "2" * 64,
            "source": "measured",
        },
        view_id="static-c10379",
        targets=TARGETS,
        mask_dimensions=VideoDimensions(width=shape[1], height=shape[0]),
        windows={"279-408": (279, 408), "1020-1172": (1020, 1172)},
        anchors=tuple(anchors),
        counts=counts,  # type: ignore[arg-type]
        exported_at=datetime.now(UTC),
        claim_boundary=review_anchors.CLAIM_BOUNDARY,
        license=review_anchors.LICENSE,
        provenance=ReviewAnchorProvenance(),
    )
    (root / review_anchors.MASK_SET_NAME).write_text(mask_set.model_dump_json(indent=2))
    return mask_set, truth


def _synthetic_run(
    root: Path, name: str, shape: tuple[int, int], masks: dict[tuple[int, str], np.ndarray | None]
) -> Path:
    run = root / name
    (run / "masks").mkdir(parents=True)
    rows = []
    for frame in sorted({frame for frame, _ in masks}):
        objects = []
        for slot, target in enumerate(TARGETS):
            mask = masks.get((frame, target))
            if mask is None:
                continue
            uri = f"masks/{frame:06d}_{slot:02d}.png"
            _write_mask(run / uri, mask)
            objects.append(
                {
                    "label": target,
                    "object_id": f"sam3-{slot:02d}",
                    "mask": {"format": "png", "storage": "external_artifact", "uri": uri},
                }
            )
        rows.append({"analysis_frame_index": frame, "objects": objects})
    (run / "observations.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    (run / "manifest.json").write_text("{}")
    return run


def test_score_runs_over_synthetic_arms_and_markdown_table(tmp_path: Path) -> None:
    shape = (32, 64)
    anchors_root = tmp_path / "anchors-workspace"
    anchors_root.mkdir()
    _, truth = _synthetic_mask_set(anchors_root, shape)
    perfect = _synthetic_run(
        tmp_path / "arms" / "perfect",
        "run-a",
        shape,
        {key: mask for key, mask in truth.items() if mask is not None},
    )
    del perfect
    leaky: dict[tuple[int, str], np.ndarray | None] = {
        key: mask for key, mask in truth.items() if mask is not None
    }
    leaky[(1100, "interior")] = _square(shape, 40, 10, 5)  # a mask where the human said hidden
    leaky[(300, "chassis")] = _square(shape, 0, 0, 3)  # a quarter of the human's square
    del leaky[(1100, "rear_body")]  # missing run mask
    _synthetic_run(tmp_path / "leaky", "run-b", shape, leaky)

    report = score_runs(
        anchors=anchors_root,
        runs=[str(tmp_path / "arms" / "perfect"), f"leaky={tmp_path / 'leaky' / 'run-b'}"],
        repository_root=tmp_path,
    )
    assert report.anchor_counts == {"labeled": 6, "hidden": 1, "unlabeled": 1}
    by_name = {run.run_name: run for run in report.runs}
    assert set(by_name) == {"perfect", "leaky"}

    good = by_name["perfect"]
    assert good.mean_iou == 1.0
    assert good.mean_iou_by_target == {t: 1.0 for t in TARGETS if t != "cabin"} | {"cabin": 1.0}
    # Both synthetic frames sit inside a window, so the outside mean has nothing to average.
    assert good.mean_iou_by_window == {"279-408": 1.0, "1020-1172": 1.0, "outside": None}
    assert good.counts == {
        "scored": 6,
        "run_mask_missing": 0,
        "hidden_correct": 1,
        "hidden_false_positive": 0,
        "unlabeled_skipped": 1,
    }

    bad = by_name["leaky"]
    assert bad.counts["hidden_false_positive"] == 1 and bad.hidden_false_positive_area == 25
    assert bad.counts["run_mask_missing"] == 1
    assert bad.counts["scored"] == 5 and bad.counts["unlabeled_skipped"] == 1
    assert bad.mean_iou_by_target["chassis"] == pytest.approx((9 / 36 + 1.0) / 2)
    assert bad.mean_iou_by_target["rear_body"] == pytest.approx(0.5)
    assert bad.mean_iou_by_window["279-408"] == pytest.approx((9 / 36 + 1 + 1 + 1) / 4)
    assert bad.mean_iou_by_window["1020-1172"] == pytest.approx((1 + 0.0) / 2)

    table = markdown_table(report)
    assert table.splitlines()[0].startswith("| arm | IoU chassis | IoU interior |")
    assert "| IoU 279-408 | IoU 1020-1172 | IoU outside |" in table.splitlines()[0]
    perfect_row = "| perfect | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | - | 0 |"
    assert (perfect_row + " 0 (0) | 6 / 1 |") in table
    assert "| leaky |" in table and "1 (25)" in table
    assert "6 labeled, 1 hidden, 1 unlabeled" in table

    payload = json.loads(report.model_dump_json())
    assert payload["manifest_kind"] == "human_review_anchor_iou"
    assert payload["anchor_kind"] == "human_review_anchor"

    # The contact sheet: rows = anchor frames, columns = anchors | reference | best arm.
    mask_set, anchors_root = review_anchors.load_mask_set(anchors_root)
    frames_bgr = {frame: np.full((*shape, 3), 90, dtype=np.uint8) for frame in mask_set.frames}
    sheet = review_anchors.render_anchor_sheet(
        report=report,
        mask_set=mask_set,
        anchors_root=anchors_root,
        frames_bgr=frames_bgr,
        reference_name="leaky",
        reference_directory=Path(by_name["leaky"].run_directory),
        best_name="perfect",
        best_directory=Path(by_name["perfect"].run_directory),
        output_path=tmp_path / "out" / "sheet.png",
        tile_width=128,
    )
    image = np.asarray(Image.open(sheet))
    assert image.shape[1] == 3 * 128
    assert image.shape[0] > 2 * 128 * shape[0] // shape[1]  # two rows plus the title band


def test_committed_human_record_is_signed_and_complete() -> None:
    record = review_anchors.ReviewAnchorHumanRecord.model_validate_json(
        (ROOT / review_anchors.DEFAULT_RECORD).read_text(encoding="utf-8")
    )
    assert record.author and record.reviewed_at is not None
    assert record.counts == {"labeled": 51, "hidden": 1, "unlabeled": 0}
    assert len(record.anchors) == 52
    hidden = [(a.analysis_frame_index, a.target) for a in record.anchors if a.state == "hidden"]
    assert hidden == [(1700, "rear_body")]
    assert all(a.mask_sha256 for a in record.anchors if a.state == "labeled")
    assert record.config.uri == "configs/qa/first_minute_review_anchors.json"
    assert "not a dataset" in record.claim_boundary
    # The one named failure case sits on the hidden cell; every other cell is unannotated.
    named = [
        (a.analysis_frame_index, a.target, a.failure_case) for a in record.anchors if a.failure_case
    ]
    assert named == [(1700, "rear_body", "distractor_confusion")]
    hidden_entry = next(a for a in record.anchors if a.state == "hidden")
    assert hidden_entry.note and "screwdriver" in hidden_entry.note
    assert record.notes and "distractor_confusion" in record.notes


def test_record_entry_failure_case_is_optional_and_snake_case() -> None:
    legacy = {"analysis_frame_index": 1700, "target": "rear_body", "state": "hidden"}
    entry = review_anchors.ReviewAnchorRecordEntry.model_validate(legacy)
    assert entry.failure_case is None and entry.note is None
    tagged = review_anchors.ReviewAnchorRecordEntry.model_validate(
        {**legacy, "failure_case": "distractor_confusion", "note": "screwdriver"}
    )
    assert tagged.failure_case == "distractor_confusion"
    with pytest.raises(ValueError):
        review_anchors.ReviewAnchorRecordEntry.model_validate(
            {**legacy, "failure_case": "Distractor Confusion"}
        )


# ---------------------------------------------------------------- workspace prepare / export


def test_prepare_workspace_configures_exactly_the_anchor_frames(tmp_path: Path) -> None:
    manifest_path = prepare_workspace(
        config_path=CONFIG, output_dir=tmp_path / "anchors", repository_root=ROOT
    )
    manifest = MuggledSAMBoxCalibrationManifest.model_validate_json(manifest_path.read_text())
    assert manifest.view_id == "static-c10379"
    assert [round(t * manifest.proxy_fps) for t in manifest.requested_proxy_timestamps_seconds] == [
        300,
        370,
        400,
        600,
        650,
        700,
        900,
        1050,
        1100,
        1150,
        1200,
        1500,
        1700,
    ]
    assert manifest.candidates == () and manifest.hidden_targets == ()
    session = review_anchors.load_session(tmp_path / "anchors")
    assert session.frame_zero_prepopulated is False
    assert session.config.uri == "configs/qa/first_minute_review_anchors.json"
    with pytest.raises(FileExistsError):
        prepare_workspace(config_path=CONFIG, output_dir=tmp_path / "anchors", repository_root=ROOT)


def _candidate(
    manifest: MuggledSAMBoxCalibrationManifest,
    frame: int,
    target: str,
    *,
    index: int,
    mask_uri: str,
    accepted: bool,
) -> dict:
    reference = frame_reference(
        frame / manifest.proxy_fps,
        fps=manifest.proxy_fps,
        source_offset_seconds=manifest.source_offset_seconds,
        frame_count=manifest.proxy_frame_count,
    )
    return {
        "candidate_id": f"t{frame:06d}-b01",
        "intended_target": target,
        "frame": reference.model_dump(mode="json"),
        "pixel_box": {"x1": 0, "y1": 0, "x2": 128, "y2": 72},
        "normalized_box": {"x": 0.0, "y": 0.0, "width": 0.1, "height": 0.1},
        "decoder_result": {
            "api": "muggledsam_sam3_interactive",
            "candidate_count": 2,
            "deterministic_best_candidate_index": 0,
            "candidates": [
                {
                    "candidate_index": i,
                    "iou_score": 0.5,
                    "mask_uri": mask_uri if i == index else f"results/masks/other-{i}.png",
                    "review_uri": f"results/review-{i}.png",
                    "is_deterministic_best": i == 0,
                }
                for i in range(2)
            ],
            "overlay_uri": "results/overlay.png",
        },
        "human_selected_candidate_index": index if accepted else None,
        "human_accepted": accepted,
    }


def test_export_writes_labeled_hidden_and_unlabeled_anchors_with_fingerprints(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "anchors"
    prepare_workspace(config_path=CONFIG, output_dir=workspace, repository_root=ROOT)
    manifest_path = workspace / "calibration_manifest.json"
    manifest = MuggledSAMBoxCalibrationManifest.model_validate_json(manifest_path.read_text())
    shape = (manifest.proxy_dimensions.height, manifest.proxy_dimensions.width)
    chosen = _square(shape, 100, 100, 40)
    _write_mask(workspace / "results/masks/t000300-b01_candidate-01.png", chosen)
    _write_mask(workspace / "results/masks/other-0.png", _square(shape, 0, 0, 5))
    hidden_frame = frame_reference(
        1100 / 30,
        fps=manifest.proxy_fps,
        source_offset_seconds=manifest.source_offset_seconds,
        frame_count=manifest.proxy_frame_count,
    )
    updated = manifest.model_copy(
        update={
            "candidates": (
                MuggledSAMCalibrationCandidate.model_validate(
                    _candidate(
                        manifest,
                        300,
                        "chassis",
                        index=1,
                        mask_uri="results/masks/t000300-b01_candidate-01.png",
                        accepted=True,
                    )
                ),
                MuggledSAMCalibrationCandidate.model_validate(
                    _candidate(
                        manifest,
                        370,
                        "chassis",
                        index=0,
                        mask_uri="results/masks/other-0.png",
                        accepted=False,
                    )
                ),
            ),
            "hidden_targets": (
                MuggledSAMCalibrationHiddenTarget(intended_target="interior", frame=hidden_frame),
            ),
        }
    )
    _write_manifest(
        manifest_path,
        MuggledSAMBoxCalibrationManifest.model_validate(updated.model_dump(mode="json")),
    )

    # A config with null provenance, as the committed file was before the human labelled.
    unsigned_config = tmp_path / "configs/qa/unsigned.json"
    unsigned_config.parent.mkdir(parents=True)
    unsigned_config.write_text(build_first_minute_config().model_dump_json(indent=2))
    record_path = tmp_path / "docs/qa/record.json"
    mask_set, mask_set_path, written_record = export_anchor_masks(
        workspace=workspace,
        repository_root=tmp_path,
        record_path=record_path,
        config_path=unsigned_config,
    )
    assert mask_set_path == workspace / "anchors/anchor_masks.json"
    assert mask_set.counts == {"labeled": 1, "hidden": 1, "unlabeled": 13 * 4 - 2}
    labeled = mask_set.anchor(300, "chassis")
    assert labeled.state == "labeled"
    assert labeled.mask_uri == "anchors/masks/f000300_chassis.png"
    assert labeled.area_pixels == 1600 and labeled.source_candidate_index == 1
    exported = workspace / labeled.mask_uri
    assert (
        exported.read_bytes()
        == (workspace / "results/masks/t000300-b01_candidate-01.png").read_bytes()
    )
    assert labeled.mask_sha256 == review_anchors._sha256_bytes(exported.read_bytes())
    assert mask_set.anchor(1100, "interior").state == "hidden"
    assert mask_set.anchor(1100, "interior").expected_visible == "hidden_prompt"
    assert mask_set.anchor(370, "chassis").state == "unlabeled"  # preview only, never accepted
    assert mask_set.provenance.author is None

    assert written_record == record_path
    record = review_anchors.ReviewAnchorHumanRecord.model_validate_json(record_path.read_text())
    assert record.author is None and record.reviewed_at is None
    assert record.counts == mask_set.counts
    assert record.mask_set.sha256 == digest_cache.sha256_file(mask_set_path)
    assert {entry.state for entry in record.anchors} == {"labeled", "hidden", "unlabeled"}
    assert "not a dataset" in record.claim_boundary

    # The exported set scores against a run that reproduces the human's mask exactly.
    run = _synthetic_run(tmp_path / "runs", "run-a", shape, {(300, "chassis"): chosen})
    report = score_runs(anchors=workspace, runs=[str(run)], repository_root=tmp_path)
    (scored,) = report.runs
    assert scored.mean_iou == 1.0 and scored.counts["hidden_correct"] == 1
    assert scored.counts["unlabeled_skipped"] == 50


REFERENCE_RUN = ROOT / (
    "runs/muggledsam-sam3-four-part-static-focused-reassembly-static-c10379-20260918t001210z"
)
ABLATION_ARMS = ROOT / "runs/sam3-policy-ablation-20260918/arms"


@pytest.mark.real_data
def test_scorer_reads_real_run_layouts_with_the_reference_as_its_own_anchor(
    tmp_path: Path,
) -> None:
    if not (REFERENCE_RUN / "observations.jsonl").is_file():
        pytest.skip("reference run is not on this machine")
    arm = ABLATION_ARMS / "xg-r720-sched"
    if not arm.is_dir():
        pytest.skip("policy ablation arms are not on this machine")
    frames = (300, 1100)
    uris = review_anchors.run_masks_at(REFERENCE_RUN, frames)
    anchors_root = tmp_path / "anchors"
    (anchors_root / "anchors" / "masks").mkdir(parents=True)
    anchors = []
    for frame in frames:
        for target in TARGETS:
            source = REFERENCE_RUN / uris[frame][target]
            destination = anchors_root / "anchors" / "masks" / f"f{frame:06d}_{target}.png"
            destination.write_bytes(source.read_bytes())
            anchors.append(
                ReviewAnchorMask(
                    analysis_frame_index=frame,
                    target=target,
                    expected_visible="visible",
                    state="labeled",
                    mask_uri=destination.relative_to(anchors_root).as_posix(),
                    mask_sha256=review_anchors._sha256_bytes(destination.read_bytes()),
                    area_pixels=int(review_anchors.mask_cache.decode_mask_png(destination).sum()),
                    selected_by="human",
                )
            )
    mask_set = ReviewAnchorMaskSet(
        manifest_kind="human_review_anchor_masks",
        config={"uri": "configs/qa/x.json", "sha256": "1" * 64, "source": "measured"},
        calibration_manifest={
            "uri": "runs/x/calibration_manifest.json",
            "sha256": "2" * 64,
            "source": "measured",
        },
        view_id="static-c10379",
        targets=TARGETS,
        mask_dimensions=VideoDimensions(width=1280, height=720),
        windows={"279-408": (279, 408), "1020-1172": (1020, 1172)},
        anchors=tuple(anchors),
        counts={"labeled": 8, "hidden": 0, "unlabeled": 0},
        exported_at=datetime.now(UTC),
        claim_boundary=review_anchors.CLAIM_BOUNDARY,
        license=review_anchors.LICENSE,
        provenance=ReviewAnchorProvenance(),
    )
    (anchors_root / review_anchors.MASK_SET_NAME).write_text(mask_set.model_dump_json())

    report = score_runs(
        anchors=anchors_root,
        runs=[str(arm), f"reference={REFERENCE_RUN}"],
        repository_root=ROOT,
    )
    by_name = {run.run_name: run for run in report.runs}
    assert by_name["reference"].mean_iou == 1.0
    assert by_name["reference"].counts["scored"] == 8
    candidate = by_name["xg-r720-sched"]
    assert candidate.counts["scored"] + candidate.counts["run_mask_missing"] == 8
    assert candidate.mean_iou is not None and 0.0 <= candidate.mean_iou <= 1.0
    assert all(not cell.resized_run_mask for cell in candidate.cells)


def test_build_manifest_and_hidden_targets_round_trip_through_the_schema(tmp_path: Path) -> None:
    manifest = build_manifest(
        repository_root=ROOT,
        config_path=ROOT
        / "configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_g2.json",
        timestamps=(10.0,),
        result_directory=tmp_path / "results",
        calibration_id="hidden-round-trip",
        view_id="static-c10379",
    )
    frame = frame_reference(
        10.0,
        fps=manifest.proxy_fps,
        source_offset_seconds=manifest.source_offset_seconds,
        frame_count=manifest.proxy_frame_count,
    )
    marked = manifest.model_copy(
        update={
            "hidden_targets": (
                MuggledSAMCalibrationHiddenTarget(intended_target="interior", frame=frame),
            )
        }
    )
    payload = json.loads(marked.model_dump_json())
    restored = MuggledSAMBoxCalibrationManifest.model_validate(payload)
    assert restored.hidden_targets[0].state == "hidden"
    assert restored.hidden_targets[0].marked_by == "human"
    payload["hidden_targets"].append(payload["hidden_targets"][0])
    with pytest.raises(ValueError, match="at most once"):
        MuggledSAMBoxCalibrationManifest.model_validate(payload)
    payload["hidden_targets"] = [payload["hidden_targets"][0]]
    payload["hidden_targets"][0]["frame"]["analysis_frame_index"] = 301
    with pytest.raises(ValueError, match="must match its analysis frame"):
        MuggledSAMBoxCalibrationManifest.model_validate(payload)
    # Old manifests without the field still load.
    del payload["hidden_targets"]
    assert MuggledSAMBoxCalibrationManifest.model_validate(payload).hidden_targets == ()
