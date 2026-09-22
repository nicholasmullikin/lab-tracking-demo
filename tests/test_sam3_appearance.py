"""`battle.sam3_appearance` with a numpy stub in place of the SAM3 model (no torch, no GPU)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from battle import mask_cache
from battle import sam3_appearance as app
from battle.schemas import MuggledSAMImageDecoderResult

WIDTH, HEIGHT = 96, 64
TARGETS = ("chassis", "interior", "rear_body", "cabin")
# One rectangle per target; the stub detector "finds" it (plus a decoy) on every frame.
PART_BOXES = {
    "chassis": (4, 4, 30, 30),
    "interior": (40, 6, 56, 22),
    "rear_body": (60, 30, 90, 60),
    "cabin": (6, 36, 34, 60),
}
DECOY_BOX = (70, 2, 94, 18)


def _rect(box: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    x1, y1, x2, y2 = box
    mask[y1:y2, x1:x2] = True
    return mask


class StubBackend:
    """Deterministic stand-in: tokens are the box list; detections are the part rectangles."""

    def __init__(self) -> None:
        self.detect_calls = 0
        self.channels = 16

    def encode(self, frame_bgr: np.ndarray) -> dict:
        return {"frame_hw": frame_bgr.shape[:2], "mean": float(frame_bgr.mean())}

    def feature_map_4x(self, encoding: dict) -> np.ndarray:
        rng = np.random.default_rng(int(encoding["mean"]) + 1)
        return rng.normal(size=(self.channels, 20, 20)).astype(np.float32)

    def exemplar_tokens(self, encoding, boxes_norm, negative_boxes_norm):
        rows = [("pos", tuple(map(tuple, b))) for b in boxes_norm]
        rows += [("neg", tuple(map(tuple, b))) for b in (negative_boxes_norm or [])]
        return {"tokens": rows}

    def concat_tokens(self, token_sets):
        return {"tokens": [t for s in token_sets for t in s["tokens"]]}

    def detect(self, encoding, token_sets, top_k):
        self.detect_calls += 1
        out = []
        for tokens in token_sets:
            positives = [t for kind, t in tokens["tokens"] if kind == "pos"]
            # The first positive box tells the stub which part is asked for.
            (x1, y1), (x2, y2) = positives[0]
            asked = min(
                PART_BOXES,
                key=lambda p: (
                    abs(PART_BOXES[p][0] / WIDTH - x1) + abs(PART_BOXES[p][1] / HEIGHT - y1)
                ),
            )
            masks = np.stack([_rect(PART_BOXES[asked]), _rect(DECOY_BOX)])[:top_k]
            scores = np.array([0.9, 0.3])[: masks.shape[0]]
            boxes = np.array(
                [
                    [[b[0] / WIDTH, b[1] / HEIGHT], [b[2] / WIDTH, b[3] / HEIGHT]]
                    for b in (PART_BOXES[asked], DECOY_BOX)
                ]
            )[: masks.shape[0]]
            out.append(app.Detections(scores=scores, boxes_norm=boxes, masks=masks, presence=0.8))
        return out

    def peak_vram_bytes(self):
        return None


def _write_video(path: Path, frames: int) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (WIDTH, HEIGHT))
    assert writer.isOpened()
    for index in range(frames):
        frame = np.full((HEIGHT, WIDTH, 3), 20 + 7 * index, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _write_run(root: Path, frames: int, *, drop: set[tuple[int, str]] = frozenset()) -> Path:
    (root / "masks").mkdir(parents=True)
    with (root / "observations.jsonl").open("w", encoding="utf-8") as handle:
        for frame in range(frames):
            objects = []
            for slot, target in enumerate(TARGETS):
                if (frame, target) in drop:
                    continue
                uri = f"masks/{frame:06d}_{slot:02d}.png"
                box = PART_BOXES[target]
                # The tracked mask drifts one pixel per frame so IoU with the reference varies.
                shifted = (box[0] + frame, box[1], box[2] + frame, box[3])
                (root / uri).write_bytes(
                    mask_cache.encode_rgba_mask_png(_rect(shifted), (10, 20, 30))
                )
                objects.append({"label": target, "mask": {"uri": uri}, "object_score": 5.0})
            handle.write(json.dumps({"analysis_frame_index": frame, "objects": objects}) + "\n")
    return root


def _write_references(path: Path, proxy: Path, masks_dir: Path, frames: list[int]) -> Path:
    masks_dir.mkdir(parents=True, exist_ok=True)
    references = []
    for frame in frames:
        for target in TARGETS:
            mask_path = masks_dir / f"f{frame:06d}_{target}.png"
            cv2.imwrite(str(mask_path), _rect(PART_BOXES[target]).astype(np.uint8) * 255)
            references.append(
                {
                    "frame": frame,
                    "target": target,
                    "mask_path": str(mask_path),
                    "provenance": "test",
                }
            )
    distractor_path = masks_dir / "distractor.png"
    cv2.imwrite(str(distractor_path), _rect(DECOY_BOX).astype(np.uint8) * 255)
    spec = {
        "schema": app.REFERENCE_SCHEMA,
        "view": "TEST",
        "sets": [
            {
                "name": "same_view",
                "view": "TEST",
                "proxy": str(proxy),
                "references": references,
                "distractors": [
                    {
                        "frame": frames[0],
                        "target": "rear_body",
                        "mask_path": str(distractor_path),
                        "label": "decoy",
                    }
                ],
            },
            {
                "name": "cross_view",
                "view": "OTHER",
                "proxy": str(proxy),
                "references": [r for r in references if r["frame"] == frames[0]],
            },
        ],
    }
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path


@pytest.fixture
def fixture(tmp_path: Path) -> dict:
    proxy = tmp_path / "proxy.mp4"
    _write_video(proxy, 6)
    run = _write_run(tmp_path / "run", 6, drop={(3, "interior")})
    references = _write_references(tmp_path / "refs.json", proxy, tmp_path / "refmasks", [0, 4])
    return {"proxy": proxy, "run": run, "references": references, "root": tmp_path}


def _pass_args(fixture: dict, output: Path, **overrides) -> argparse.Namespace:
    values = dict(
        command="pass",
        proxy=str(fixture["proxy"]),
        references=str(fixture["references"]),
        model=None,
        device="cpu",
        max_side_length=1280,
        top_k=2,
        gpu_guard="vram",
        allow_gpu_neighbour=[],
        run=str(fixture["run"]),
        output=str(output),
        frame_count=6,
        frames=None,
        keep_frame=[2, 4],
        target=None,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def test_pass_writes_embeddings_detections_and_leave_reference_out(fixture: dict) -> None:
    backend = StubBackend()
    output = app.run_pass(_pass_args(fixture, fixture["root"] / "out"), backend=backend)
    with np.load(output / "native" / "embeddings.npz") as archive:
        assert archive["embeddings"].shape == (6, 4, 16)
        assert archive["frames"].tolist() == list(range(6))
        has_mask = archive["has_mask"]
        assert has_mask.sum() == 6 * 4 - 1 and not has_mask[3, 1]
        norms = np.linalg.norm(archive["embeddings"][has_mask].astype(np.float32), axis=1)
        assert np.allclose(norms, 1.0, atol=1e-2)
        roles = archive["reference_roles"].tolist()
        assert roles.count("positive") == 8 + 4 and any(r.startswith("distractor:") for r in roles)
        assert archive["reference_embeddings"].shape[1] == 16
    rows = [json.loads(line) for line in (output / "detections.jsonl").read_text().splitlines()]
    # 6 frames x 4 targets x 2 variants x 2 sets.
    assert len(rows) == 6 * 4 * 2 * 2
    at_reference = [r for r in rows if r["frame"] == 4 and r["set"] == "same_view"]
    assert at_reference and all(r["excluded_reference_frame"] == 4 for r in at_reference)
    assert all(r["reference_frames_used"] == [0] for r in at_reference)
    cross = [r for r in rows if r["frame"] == 4 and r["set"] == "cross_view"]
    assert all(
        r["excluded_reference_frame"] is None and r["reference_frames_used"] == [0] for r in cross
    )
    elsewhere = [r for r in rows if r["frame"] == 2 and r["set"] == "same_view"]
    assert all(r["reference_frames_used"] == [0, 4] for r in elsewhere)
    chassis = next(
        r
        for r in rows
        if r["frame"] == 2
        and r["target"] == "chassis"
        and r["set"] == "same_view"
        and r["variant"] == "pos"
    )
    assert chassis["presence"] == pytest.approx(0.8)
    assert chassis["top_score"] == pytest.approx(0.9)
    # Tracked mask shifted by 2 px against a 26-px-wide reference rectangle.
    assert 0.8 < chassis["best_overlap_iou"] < 0.95
    assert chassis["top_centroid_distance_px"] == pytest.approx(2.0)
    missing = next(
        r
        for r in rows
        if r["frame"] == 3
        and r["target"] == "interior"
        and r["set"] == "same_view"
        and r["variant"] == "pos"
    )
    assert missing["tracked_area_px"] == 0 and missing["best_overlap_iou"] is None
    kept_text = (output / "detections_kept.jsonl").read_text()
    kept = [json.loads(line) for line in kept_text.splitlines()]
    assert {k["frame"] for k in kept} == {2, 4}
    assert all((output / m["uri"]).is_file() for k in kept for m in k["masks"])
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["schema"] == app.MANIFEST_SCHEMA and manifest["detection_rows"] == len(rows)
    assert backend.detect_calls == 6


def test_pass_explicit_frames_and_single_reference_leave_out(fixture: dict) -> None:
    spec = json.loads(Path(fixture["references"]).read_text())
    spec["sets"] = [spec["sets"][0]]
    spec["sets"][0]["references"] = [r for r in spec["sets"][0]["references"] if r["frame"] == 4]
    path = fixture["root"] / "single.json"
    path.write_text(json.dumps(spec))
    output = app.run_pass(
        _pass_args(
            fixture, fixture["root"] / "out2", references=str(path), frames=[1, 4], keep_frame=None
        ),
        backend=StubBackend(),
    )
    rows = [json.loads(line) for line in (output / "detections.jsonl").read_text().splitlines()]
    assert {r["frame"] for r in rows} == {1, 4}
    unavailable = [r for r in rows if r.get("unavailable")]
    assert len(unavailable) == 8 and all(r["frame"] == 4 for r in unavailable)
    with np.load(output / "native" / "embeddings.npz") as archive:
        assert archive["frames"].tolist() == [1, 4]


def test_detection_row_math() -> None:
    tracked = _rect((10, 10, 30, 30))
    masks = np.stack([_rect((12, 10, 32, 30)), _rect((60, 40, 80, 60))])
    detections = app.Detections(
        scores=np.array([0.2, 0.7]),
        boxes_norm=np.zeros((2, 2, 2)),
        masks=masks,
        presence=0.5,
    )
    row = app.detection_row(
        frame=7,
        target="chassis",
        set_name="same_view",
        reference_view="V",
        variant="pos",
        reference_frames=(0,),
        excluded_frame=None,
        tracked=tracked,
        detections=detections,
        width=WIDTH,
        height=HEIGHT,
    )
    assert row["top_score"] == pytest.approx(0.7)
    assert row["top_iou_tracked"] == 0.0
    assert row["best_overlap_index"] == 0 and row["best_overlap_iou"] == pytest.approx(18 / 22)
    assert row["top_centroid_distance_px"] == pytest.approx(np.hypot(50, 30))
    assert row["detections_above_0.5"] == 1


def test_pooling_and_boxes() -> None:
    feature = np.zeros((3, 8, 8), dtype=np.float32)
    feature[0, :4, :] = 1.0
    feature[1, 4:, :] = 1.0
    mask = np.zeros((64, 64), dtype=bool)
    mask[:32] = True
    vector = app.pooled_embedding(feature, app.pool_weights(mask, (8, 8)))
    assert (
        vector is not None and vector[0] == pytest.approx(1.0) and vector[1] == pytest.approx(0.0)
    )
    assert app.pooled_embedding(feature, np.zeros((8, 8), dtype=np.float32)) is None
    assert app.mask_box_px(_rect((4, 4, 30, 30))) == (4, 4, 30, 30)
    assert app.mask_box_px(np.zeros((4, 4), dtype=bool)) is None
    assert app.normalized_box((0, 0, 48, 32), WIDTH, HEIGHT) == [[0.0, 0.0], [0.5, 0.5]]
    assert app.mask_iou(_rect((0, 0, 10, 10)), _rect((0, 0, 10, 10))) == 1.0
    assert app.mask_iou(np.zeros((4, 4), bool), np.zeros((4, 4), bool)) is None


def test_gate_and_exemplar_decoder_result_validate(tmp_path: Path) -> None:
    detections = StubBackend().detect(
        {"frame_hw": (HEIGHT, WIDTH)},
        [{"tokens": [("pos", ((4 / WIDTH, 4 / HEIGHT), (30 / WIDTH, 30 / HEIGHT)))]}],
        5,
    )[0]
    box = {"x1": 0, "y1": 0, "x2": 40, "y2": 40}
    assert app.gate_detections(detections, box) == [0]
    result = app.exemplar_decoder_result(
        detections=detections,
        prompt_box=box,
        candidate_id="t000010-b01",
        results_directory=tmp_path / "results",
        frame=np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8),
        reference_frames=(0, 4),
        set_name="same_view",
        variant="posneg",
    )
    model = MuggledSAMImageDecoderResult.model_validate(result)
    assert model.api == app.EXEMPLAR_API and model.candidate_count == 1
    assert model.candidates[0].iou_score == pytest.approx(0.9)
    assert (tmp_path / "results" / model.candidates[0].mask_uri.removeprefix("results/")).is_file()
    assert (tmp_path / "results" / "t000010-b01_overlay.png").is_file()
    # Nothing inside the box: one empty mask so the acceptance rule rejects.
    empty = app.exemplar_decoder_result(
        detections=detections,
        prompt_box={"x1": 0, "y1": 60, "x2": 3, "y2": 63},
        candidate_id="t000010-b02",
        results_directory=tmp_path / "results",
        frame=None,
        reference_frames=(0,),
        set_name="same_view",
        variant="pos",
        write_overlay=False,
    )
    model = MuggledSAMImageDecoderResult.model_validate(empty)
    assert model.candidates[0].iou_score == 0.0 and app.EMPTY_MASK_LIMITATION in model.limitations
    assert not mask_cache.decode_mask_png(
        tmp_path / "results" / "masks" / "t000010-b02_exemplar-00.png"
    ).any()


def test_merge_decoder_results_reindexes_and_marks_best() -> None:
    image = {
        "candidates": [
            {
                "candidate_index": 0,
                "iou_score": 0.5,
                "mask_uri": "results/masks/a.png",
                "is_deterministic_best": True,
            },
        ],
        "limitations": ["image"],
    }
    exemplar = {
        "candidates": [
            {
                "candidate_index": 0,
                "iou_score": 0.9,
                "mask_uri": "results/masks/b.png",
                "is_deterministic_best": True,
            },
            {
                "candidate_index": 1,
                "iou_score": 0.2,
                "mask_uri": "results/masks/c.png",
                "is_deterministic_best": False,
            },
        ],
        "overlay_uri": "results/x_overlay.png",
        "limitations": ["exemplar"],
    }
    merged = app.merge_decoder_results(image, exemplar)
    assert [c["candidate_index"] for c in merged["candidates"]] == [0, 1, 2]
    assert merged["deterministic_best_candidate_index"] == 1
    assert [c["is_deterministic_best"] for c in merged["candidates"]] == [False, True, False]
    assert "image" in merged["limitations"] and "exemplar" in merged["limitations"]


def test_serve_runtime_batch_decode(fixture: dict) -> None:
    args = argparse.Namespace(
        proxy=str(fixture["proxy"]),
        references=str(fixture["references"]),
        model=None,
        device="cpu",
        max_side_length=1280,
        top_k=2,
        gpu_guard="vram",
        allow_gpu_neighbour=[],
        results_directory=str(fixture["root"] / "serve"),
        candidate_source="exemplar_detector",
        exemplar_set="same_view",
        exemplar_variant="posneg",
    )
    runtime = app.ServeRuntime(args, backend=StubBackend())
    try:
        assert runtime.handle("health", {})["ready"]
        preview = runtime.handle("frame_preview", {"frame_index": 2})
        assert (fixture["root"] / "serve" / preview["image_uri"].removeprefix("results/")).is_file()
        response = runtime.handle(
            "batch_decode",
            {
                "prompts": [
                    {
                        "box_id": "p000002-b01",
                        "candidate_id": "t000002-b01",
                        "frame_index": 2,
                        "pixel_box": {"x1": 0, "y1": 0, "x2": 36, "y2": 36},
                        "intended_target": "chassis",
                        "boxes": [[[0, 0], [0.4, 0.5]]],
                        "fg_points": [],
                        "bg_points": [],
                    },
                    {
                        "box_id": "p000004-b01",
                        "candidate_id": "t000004-b01",
                        "frame_index": 4,
                        "pixel_box": {"x1": 0, "y1": 0, "x2": 36, "y2": 36},
                        "intended_target": "chassis",
                        "boxes": [[[0, 0], [0.4, 0.5]]],
                        "fg_points": [],
                        "bg_points": [],
                    },
                ]
            },
        )
        decoded = response["decoded"]
        assert [d["candidate_id"] for d in decoded] == ["t000002-b01", "t000004-b01"]
        for item in decoded:
            model = MuggledSAMImageDecoderResult.model_validate(item["decoder_result"])
            assert model.api == app.EXEMPLAR_API
        # Frame 4 is a reference frame: its exemplars came from frame 0 only.
        assert "reference frames [0]" in decoded[1]["decoder_result"]["limitations"][0]
        assert "reference frames [0, 4]" in decoded[0]["decoder_result"]["limitations"][0]
        with pytest.raises(ValueError):
            runtime.handle("nope", {})
    finally:
        runtime.close()


def test_load_reference_spec_rejects_bad_schema(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema": "other", "view": "V", "sets": []}))
    with pytest.raises(ValueError):
        app.load_reference_spec(path)
    path.write_text(
        json.dumps(
            {
                "schema": app.REFERENCE_SCHEMA,
                "view": "V",
                "sets": [
                    {"name": "a", "view": "V", "proxy": "p", "references": []},
                    {"name": "a", "view": "W", "proxy": "p", "references": []},
                ],
            }
        )
    )
    with pytest.raises(ValueError):
        app.load_reference_spec(path)


def test_cli_parser_defaults() -> None:
    parser = app.build_parser()
    args = parser.parse_args(
        ["pass", "--proxy", "p", "--references", "r", "--run", "d", "--output", "o"]
    )
    assert args.top_k == app.DEFAULT_TOP_K and args.gpu_guard == "vram" and args.frame_count == 1800
    serve = parser.parse_args(
        [
            "serve-jsonl",
            "--proxy",
            "p",
            "--references",
            "r",
            "--results-directory",
            "x",
            "--candidate-source",
            "both",
        ]
    )
    assert serve.candidate_source == "both" and serve.exemplar_variant == "posneg"
