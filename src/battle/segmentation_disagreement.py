"""Per-frame, per-target mask IoU between two segmentation runs and their disagreement episodes.

This measures *cross-method disagreement*, not accuracy: neither run is ground truth, so a low
IoU only says the two methods drew different masks for the same label on the same frame. The
two runs must share the analysis frame index space (same proxy, same 30 FPS contract); each run
is read from its ``observations.jsonl`` and the mask PNGs those rows reference.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import mask_ops
from .mask_ops import decode_mask_png

DEFAULT_TARGETS = ("chassis", "interior", "rear_body", "cabin")
CLAIM = "cross-method disagreement between two segmentation runs; not ground-truth accuracy"


@dataclass(frozen=True)
class FrameTargetIoU:
    frame: int
    target: str
    iou: float | None
    status: str
    """One of ``both_present``, ``missing_in_a``, ``missing_in_b``, ``both_missing``."""
    area_a: int
    area_b: int


@dataclass(frozen=True)
class DisagreementEpisode:
    target: str
    start_frame: int
    end_frame_exclusive: int
    frame_count: int
    mean_iou: float
    minimum_iou: float
    missing_frames_a: int
    missing_frames_b: int


def mask_iou(a: np.ndarray | None, b: np.ndarray | None) -> tuple[float | None, str]:
    """IoU of two boolean masks; a mask missing on one side counts as full disagreement."""
    if a is None and b is None:
        return None, "both_missing"
    if a is None:
        return 0.0, "missing_in_a"
    if b is None:
        return 0.0, "missing_in_b"
    value = mask_ops.mask_iou(a, b, empty_union=None)
    if value is None:
        return None, "both_missing"
    return value, "both_present"


def load_mask_index(run_directory: Path) -> dict[int, dict[str, Path]]:
    """Map analysis frame -> label -> mask path from a run's ``observations.jsonl``."""
    observations = run_directory / "observations.jsonl"
    if not observations.is_file():
        raise FileNotFoundError(observations)
    index: dict[int, dict[str, Path]] = {}
    for line in observations.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        frame = int(row["analysis_frame_index"])
        entry = index.setdefault(frame, {})
        for item in row.get("objects", ()):
            mask = item.get("mask")
            if not mask or not mask.get("uri"):
                continue
            entry[str(item["label"])] = run_directory / str(mask["uri"])
    return index


def read_mask(path: Path | None) -> np.ndarray | None:
    if path is None or not path.is_file():
        return None
    mask = decode_mask_png(path)
    return mask if mask.any() else None


def per_frame_target_iou(
    index_a: Mapping[int, Mapping[str, Path]],
    index_b: Mapping[int, Mapping[str, Path]],
    *,
    frames: Iterable[int],
    targets: Sequence[str] = DEFAULT_TARGETS,
) -> list[FrameTargetIoU]:
    rows: list[FrameTargetIoU] = []
    for frame in frames:
        masks_a = index_a.get(frame, {})
        masks_b = index_b.get(frame, {})
        for target in targets:
            a = read_mask(masks_a.get(target))
            b = read_mask(masks_b.get(target))
            iou, status = mask_iou(a, b)
            rows.append(
                FrameTargetIoU(
                    frame=frame,
                    target=target,
                    iou=iou,
                    status=status,
                    area_a=int(a.sum()) if a is not None else 0,
                    area_b=int(b.sum()) if b is not None else 0,
                )
            )
    return rows


def disagreement_episodes(
    rows: Sequence[FrameTargetIoU],
    *,
    threshold: float = 0.5,
    minimum_frames: int = 5,
) -> list[DisagreementEpisode]:
    """Maximal runs of consecutive frames with IoU < threshold, at least ``minimum_frames`` long.

    Frames where both runs lack the target are neutral: they neither extend nor break a run.
    """
    episodes: list[DisagreementEpisode] = []
    by_target: dict[str, list[FrameTargetIoU]] = {}
    for row in rows:
        by_target.setdefault(row.target, []).append(row)
    for target, target_rows in by_target.items():
        ordered = sorted(target_rows, key=lambda item: item.frame)
        current: list[FrameTargetIoU] = []
        previous_frame: int | None = None

        def flush() -> None:
            if len(current) >= minimum_frames:
                ious = [row.iou for row in current if row.iou is not None]
                episodes.append(
                    DisagreementEpisode(
                        target=target,
                        start_frame=current[0].frame,
                        end_frame_exclusive=current[-1].frame + 1,
                        frame_count=len(current),
                        mean_iou=float(sum(ious) / len(ious)),
                        minimum_iou=float(min(ious)),
                        missing_frames_a=sum(row.status == "missing_in_a" for row in current),
                        missing_frames_b=sum(row.status == "missing_in_b" for row in current),
                    )
                )
            current.clear()

        for row in ordered:
            if previous_frame is not None and row.frame != previous_frame + 1:
                flush()
            previous_frame = row.frame
            if row.iou is None:
                continue
            if row.iou < threshold:
                current.append(row)
            else:
                flush()
        flush()
    episodes.sort(key=lambda item: (-item.frame_count, item.start_frame, item.target))
    return episodes


def summarize(rows: Sequence[FrameTargetIoU]) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for target in sorted({row.target for row in rows}):
        selected = [row for row in rows if row.target == target]
        ious = [row.iou for row in selected if row.iou is not None]
        summary[target] = {
            "frames": len(selected),
            "compared_frames": len(ious),
            "mean_iou": float(sum(ious) / len(ious)) if ious else float("nan"),
            "median_iou": float(np.median(ious)) if ious else float("nan"),
            "frames_below_0_5": sum(value < 0.5 for value in ious),
            "missing_in_a": sum(row.status == "missing_in_a" for row in selected),
            "missing_in_b": sum(row.status == "missing_in_b" for row in selected),
            "both_missing": sum(row.status == "both_missing" for row in selected),
        }
    return summary


def compare_runs(
    *,
    run_a: Path,
    run_b: Path,
    start_frame: int,
    end_frame_exclusive: int,
    targets: Sequence[str] = DEFAULT_TARGETS,
    threshold: float = 0.5,
    minimum_frames: int = 5,
    label_a: str = "a",
    label_b: str = "b",
) -> dict[str, object]:
    rows = per_frame_target_iou(
        load_mask_index(run_a),
        load_mask_index(run_b),
        frames=range(start_frame, end_frame_exclusive),
        targets=targets,
    )
    episodes = disagreement_episodes(rows, threshold=threshold, minimum_frames=minimum_frames)
    return {
        "schema_version": "1.0",
        "manifest_kind": "segmentation_cross_method_disagreement",
        "claim": CLAIM,
        "ground_truth_accuracy_claim": False,
        "run_a": {"label": label_a, "uri": run_a.as_posix()},
        "run_b": {"label": label_b, "uri": run_b.as_posix()},
        "frame_range": {"start_frame": start_frame, "end_frame_exclusive": end_frame_exclusive},
        "targets": list(targets),
        "episode_rule": {
            "iou_threshold": threshold,
            "minimum_consecutive_frames": minimum_frames,
            "missing_on_one_side": "iou 0.0",
            "missing_on_both_sides": "neutral (skipped)",
        },
        "summary": summarize(rows),
        "episodes": [asdict(item) for item in episodes],
        "rows": [asdict(item) for item in rows],
    }


def format_episodes(episodes: Sequence[Mapping[str, object]], limit: int = 20) -> str:
    lines = ["target      frames            n   mean   min  missA missB"]
    for item in list(episodes)[:limit]:
        lines.append(
            f"{item['target']:<10} [{item['start_frame']:5d},{item['end_frame_exclusive']:5d}) "
            f"{item['frame_count']:4d}  {item['mean_iou']:.3f} {item['minimum_iou']:.3f}  "
            f"{item['missing_frames_a']:4d}  {item['missing_frames_b']:4d}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--label-a", default="a")
    parser.add_argument("--label-b", default="b")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame-exclusive", type=int, required=True)
    parser.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGETS))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--minimum-frames", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare_runs(
        run_a=args.run_a,
        run_b=args.run_b,
        start_frame=args.start_frame,
        end_frame_exclusive=args.end_frame_exclusive,
        targets=tuple(args.targets),
        threshold=args.threshold,
        minimum_frames=args.minimum_frames,
        label_a=args.label_a,
        label_b=args.label_b,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(CLAIM)
    summary: dict[str, dict[str, float | int]] = report["summary"]  # type: ignore[assignment]
    episodes: list[dict[str, object]] = report["episodes"]  # type: ignore[assignment]
    for target, values in summary.items():
        missing = f"{values['missing_in_a']}/{values['missing_in_b']}"
        print(
            f"{target:<10} mean IoU {values['mean_iou']:.3f}  median {values['median_iou']:.3f}  "
            f"<0.5 on {values['frames_below_0_5']} frames  "
            f"missing {args.label_a}/{args.label_b}: {missing}"
        )
    print(f"{len(episodes)} disagreement episodes")
    print(format_episodes(episodes))
    print(args.output)


if __name__ == "__main__":
    main()
