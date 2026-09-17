"""Fuse verified Kineo and BoxMOT person boxes, then rerun NLF on each crop."""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .external_smoke_import import import_kineo, sha256_file
from .kineo_nlf import _validate_pkls
from .schemas import (
    ClockName,
    FrameObservations,
    KineoBoxProvenance,
    NormalizedBox,
    RunManifest,
)

FRAME_COUNT = 600
WIDTH = 1280
HEIGHT = 720
CONSISTENCY_WINDOW = 15
TARGET_INFERENCE_GAP = 3
HARD_INFERENCE_GAP = 5
DEFAULT_NATIVE_RUN = Path("runs/kineo-nlf-headless-20s-frame-step-1-overnight-v2")
DEFAULT_BOXMOT_RUN = Path("runs/boxmot-yolo-static-20s-20260916t0445z")
DEFAULT_OUTPUT = Path("runs/kineo-nlf-fused-20s-overnight-v3")


@dataclass(frozen=True)
class FusionResult:
    provenance: tuple[KineoBoxProvenance, ...]
    counts: dict[str, int]


def _box_iou(left: NormalizedBox, right: NormalizedBox) -> float:
    x1 = max(left.x, right.x)
    y1 = max(left.y, right.y)
    x2 = min(left.x + left.width, right.x + right.width)
    y2 = min(left.y + left.height, right.y + right.height)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = left.width * left.height + right.width * right.height - intersection
    return intersection / union if union else 0.0


def _center_distance(left: NormalizedBox, right: NormalizedBox) -> float:
    return (
        ((left.x + left.width / 2 - right.x - right.width / 2) ** 2)
        + ((left.y + left.height / 2 - right.y - right.height / 2) ** 2)
    ) ** 0.5


def _interpolate(left: NormalizedBox, right: NormalizedBox, weight: float) -> NormalizedBox:
    return NormalizedBox(
        x=left.x * (1 - weight) + right.x * weight,
        y=left.y * (1 - weight) + right.y * weight,
        width=left.width * (1 - weight) + right.width * weight,
        height=left.height * (1 - weight) + right.height * weight,
    )


def _read_rows(path: Path) -> dict[int, FrameObservations]:
    rows = [
        FrameObservations.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    result = {row.analysis_frame_index: row for row in rows}
    if set(result) != set(range(FRAME_COUNT)):
        raise ValueError(f"{path} must retain exactly analysis frames [0,{FRAME_COUNT})")
    return result


def verify_source_alignment(native: RunManifest, boxmot: RunManifest) -> None:
    """Require the same source asset, clocks, view, and exact frame timestamp mapping."""

    if native.clip.clip_id != boxmot.clip.clip_id or native.clip.asset != boxmot.clip.asset:
        raise ValueError("Kineo and BoxMOT must declare the same source asset")
    if native.clip.timing != boxmot.clip.timing:
        raise ValueError("Kineo and BoxMOT must declare identical clock mappings")
    for manifest in (native, boxmot):
        if manifest.clip.views != ("static-c10379",):
            raise ValueError("fusion accepts only the approved static view")
        if manifest.clip.timing.clocks.fps_for(ClockName.ANALYSIS) != 30:
            raise ValueError("fusion requires the 30-fps analysis clock")
        for row in manifest.observations:
            expected = manifest.clip.timing.source_seconds_for_frame(
                ClockName.ANALYSIS, row.analysis_frame_index
            )
            if abs(row.source_seconds - expected) > 1e-6:
                raise ValueError(
                    f"{manifest.run_id} source mapping mismatch at {row.analysis_frame_index}"
                )


def _nearest_native(
    frame: int, native: dict[int, NormalizedBox]
) -> tuple[NormalizedBox | None, int | None]:
    candidates = [
        (abs(frame - index), index, box)
        for index, box in native.items()
        if abs(frame - index) <= CONSISTENCY_WINDOW
    ]
    if not candidates:
        return None, None
    _, index, box = min(candidates, key=lambda value: (value[0], value[1]))
    return box, index


def _consistent_with_native(
    candidate: NormalizedBox, native: dict[int, NormalizedBox], frame: int
) -> tuple[bool, float | None, float | None]:
    reference, _ = _nearest_native(frame, native)
    if reference is None:
        return False, None, None
    iou = _box_iou(candidate, reference)
    center = _center_distance(candidate, reference)
    area_ratio = max(
        candidate.width * candidate.height / (reference.width * reference.height),
        reference.width * reference.height / (candidate.width * candidate.height),
    )
    return iou >= 0.2 or (center <= 0.15 and area_ratio <= 2.0), iou, center


def fuse_person_boxes(
    native_rows: dict[int, FrameObservations], boxmot_rows: dict[int, FrameObservations]
) -> FusionResult:
    """Select native boxes, verified BoxMOT fallbacks, then bounded inferred residuals."""

    native_boxes = {
        frame: row.objects[0].box for frame, row in native_rows.items() if row.objects
    }
    raw: list[KineoBoxProvenance] = []
    for frame in range(FRAME_COUNT):
        native_box = native_boxes.get(frame)
        boxmot = boxmot_rows[frame].objects[0] if boxmot_rows[frame].objects else None
        if native_box is not None:
            iou = _box_iou(native_box, boxmot.box) if boxmot else None
            center = _center_distance(native_box, boxmot.box) if boxmot else None
            raw.append(
                KineoBoxProvenance(
                    analysis_frame_index=frame,
                    source="detected_native",
                    box=native_box,
                    native_box=native_box,
                    boxmot_box=boxmot.box if boxmot else None,
                    boxmot_object_id=boxmot.object_id if boxmot else None,
                    source_mapping_verified=True,
                    nearest_native_iou=iou,
                    nearest_native_center_distance=center,
                )
            )
        elif boxmot is not None:
            accepted, iou, center = _consistent_with_native(boxmot.box, native_boxes, frame)
            raw.append(
                KineoBoxProvenance(
                    analysis_frame_index=frame,
                    source="boxmot_fallback" if accepted else "missing",
                    box=boxmot.box if accepted else None,
                    boxmot_box=boxmot.box,
                    boxmot_object_id=boxmot.object_id,
                    source_mapping_verified=accepted,
                    nearest_native_iou=iou,
                    nearest_native_center_distance=center,
                )
            )
        else:
            raw.append(
                KineoBoxProvenance(
                    analysis_frame_index=frame, source="missing", source_mapping_verified=True
                )
            )

    fused = list(raw)
    cursor = 0
    while cursor < FRAME_COUNT:
        if fused[cursor].source != "missing":
            cursor += 1
            continue
        start = cursor
        while cursor < FRAME_COUNT and fused[cursor].source == "missing":
            cursor += 1
        end = cursor
        length = end - start
        left = fused[start - 1].box if start else None
        right = fused[end].box if end < FRAME_COUNT else None
        if length > HARD_INFERENCE_GAP or (left is None and right is None):
            continue
        if left is not None and right is not None:
            for frame in range(start, end):
                fused[frame] = KineoBoxProvenance(
                    analysis_frame_index=frame,
                    source="interpolated",
                    box=_interpolate(left, right, (frame - start + 1) / (length + 1)),
                    source_mapping_verified=True,
                    residual_gap_length=length,
                )
        elif length <= TARGET_INFERENCE_GAP:
            held = left if left is not None else right
            assert held is not None
            for frame in range(start, end):
                fused[frame] = KineoBoxProvenance(
                    analysis_frame_index=frame,
                    source="held",
                    box=held,
                    source_mapping_verified=True,
                    residual_gap_length=length,
                )
    counts = dict(sorted(Counter(item.source for item in fused).items()))
    return FusionResult(provenance=tuple(fused), counts=counts)


def _pixel_box(box: NormalizedBox) -> list[float]:
    return [
        box.x * WIDTH,
        box.y * HEIGHT,
        (box.x + box.width) * WIDTH,
        (box.y + box.height) * HEIGHT,
    ]


def _write_fused_native_input(
    native_pkl_root: Path, output_root: Path, fusion: FusionResult
) -> Path:
    output_root.mkdir(parents=True, exist_ok=False)
    template = pickle.loads((native_pkl_root / "bboxes_2d.pkl").read_bytes())
    annotations = [
        {
            "view_id": "static-c10379",
            "frame_idx": item.analysis_frame_index,
            "subject_id": "subject_0",
            "category_id": 0,
            "xyxy": _pixel_box(item.box),
            "score": 1.0 if item.source == "detected_native" else 0.5,
        }
        for item in fusion.provenance
        if item.box is not None
    ]
    (output_root / "bboxes_2d.pkl").write_bytes(
        pickle.dumps({"metadata": template["metadata"], "annotations": annotations})
    )
    return output_root / "bboxes_2d.pkl"


def _run_worker(
    *,
    native_root: Path,
    template_keypoints: Path,
    video: Path,
    output: Path,
    batch_size: int,
) -> float:
    command = [
        "pixi",
        "run",
        "python",
        str(Path(__file__).with_name("kineo_fusion_worker.py")),
        "--video",
        str(video),
        "--bboxes",
        str(native_root / "bboxes_2d.pkl"),
        "--template-keypoints",
        str(template_keypoints),
        "--output",
        str(output),
        "--batch-size",
        str(batch_size),
    ]
    started = time.perf_counter()
    subprocess.run(command, cwd="/home/nick/src/kineo", check=True)
    return time.perf_counter() - started


def run(args: argparse.Namespace) -> Path:
    root = args.repository_root.resolve()
    native_run = (root / args.native_run).resolve()
    boxmot_run = (root / args.boxmot_run).resolve()
    output = (root / args.output_root).resolve()
    if output.exists():
        raise FileExistsError(output)
    native_manifest = RunManifest.model_validate_json((native_run / "manifest.json").read_text())
    boxmot_manifest = RunManifest.model_validate_json((boxmot_run / "manifest.json").read_text())
    verify_source_alignment(native_manifest, boxmot_manifest)
    native_rows = _read_rows(native_run / "observations.jsonl")
    boxmot_rows = _read_rows(boxmot_run / "observations.jsonl")
    fusion = fuse_person_boxes(native_rows, boxmot_rows)
    native_pkl_root = (
        root / "runs/kineo/fused_native_inputs" / f"{output.name}-native"
    ).resolve()
    source_native = (
        root
        / "runs/kineo/infer_nlf_headless_only/offline_demo/annotations/"
        / args.native_sequence
    ).resolve()
    if not (source_native / "keypoints_2d.pkl").is_file():
        raise FileNotFoundError(source_native / "keypoints_2d.pkl")
    _write_fused_native_input(source_native, native_pkl_root, fusion)
    worker_seconds = _run_worker(
        native_root=native_pkl_root,
        template_keypoints=source_native / "keypoints_2d.pkl",
        video=(boxmot_run / "input.mp4").resolve(),
        output=native_pkl_root / "keypoints_2d.pkl",
        batch_size=args.batch_size,
    )
    (native_pkl_root / "stage_timings.pkl").write_bytes(
        pickle.dumps(
            {
                "annotations": [
                    {
                        "stage_name": "NLF SMPL Keypoints Detection (fused person crops)",
                        "stage_idx": 0,
                        "duration_seconds": worker_seconds,
                    }
                ]
            }
        )
    )
    import_args = argparse.Namespace(
        repository_root=root,
        config=args.config,
        output_root=args.output_root.parent,
        run_id=output.name,
        native=native_pkl_root.relative_to(root),
        video=(boxmot_run / "input.mp4").relative_to(root),
        image=Path("data/derived/assembly101/smoke_frames/focused_static_frame0.jpg"),
    )
    manifest_path = import_kineo(import_args)
    if manifest_path.parent != output:
        raise AssertionError("Kineo fusion import wrote an unexpected run directory")
    validation = _validate_pkls(native_pkl_root)
    payload = {
        "classification": "kineo_nlf_only_partial",
        "native_run": args.native_run.as_posix(),
        "boxmot_run": args.boxmot_run.as_posix(),
        "native_pkl_fingerprint": sha256_file(source_native / "bboxes_2d.pkl"),
        "boxmot_manifest_fingerprint": sha256_file(boxmot_run / "manifest.json"),
        "mapping_verified": True,
        "policy": {
            "native_primary": True,
            "boxmot_fallback": (
                "requires identical source/timing mapping and nearby native consistency"
            ),
            "target_inference_gap_frames": TARGET_INFERENCE_GAP,
            "hard_inference_gap_frames": HARD_INFERENCE_GAP,
            "long_gaps_are_missing": True,
        },
        "counts_by_provenance": fusion.counts,
        "nlf_elapsed_seconds": worker_seconds,
        "nlf_input_frames": sum(item.box is not None for item in fusion.provenance),
        "pkl_validation": validation,
        "residual_missing_ranges": _ranges(
            [item.analysis_frame_index for item in fusion.provenance if item.source == "missing"]
        ),
    }
    (output / "box_fusion.json").write_text(
        json.dumps([item.model_dump(mode="json") for item in fusion.provenance], indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "provenance.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (output / "pkl_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    return manifest_path


def _ranges(indices: list[int]) -> list[dict[str, int]]:
    result: list[dict[str, int]] = []
    for index in indices:
        if not result or index > result[-1]["end_frame"] + 1:
            result.append({"start_frame": index, "end_frame": index})
        else:
            result[-1]["end_frame"] = index
    for item in result:
        item["length"] = item["end_frame"] - item["start_frame"] + 1
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_g2.json"),
    )
    parser.add_argument("--native-run", type=Path, default=DEFAULT_NATIVE_RUN)
    parser.add_argument("--boxmot-run", type=Path, default=DEFAULT_BOXMOT_RUN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--native-sequence", default="assembly101_focused_static_20s_step1_overnight_v2"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 64:
        raise SystemExit("--batch-size must be in [1,64]")
    print(run(args))


if __name__ == "__main__":
    main()
