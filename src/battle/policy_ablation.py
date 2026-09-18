"""Label-free comparison of SAM3 tracker-policy arms on the focused C10379 first minute.

Every arm is one `battle-muggled-smoke` run under `<root>/arms/<arm>/`.  For each arm this
module measures, against the human-corrected reference run and against itself:

- cross-run disagreement (per-part mask IoU with the reference, overall, inside the three
  human-reported failure windows and outside them, plus the disagreement episodes);
- pairwise slot overlap (frames where two of the arm's own part masks overlap by IoU > 0.3);
- area-trace proxies (frames where a part's area leaves [0.5, 2] x its rolling median);
- memory-gate statistics from the worker diagnostics (fraction of frames a slot was not
  memorised, reason histogram, starvation flag);
- the five pre-accuracy measures (coverage, runtime, time to first output, peak VRAM, mask
  grid side) and, optionally, the `battle-review-metrics` episodes with the arm's masks
  swapped in as the reference.

Every number is self-consistency or cross-run disagreement: the reference is itself wrong in
the windows it was chosen to expose, so nothing here is accuracy.  The module also renders
the anchor-frame contact sheets and the two-run Rerun recording the review reads from.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import mask_cache
from .segmentation_disagreement import (
    DEFAULT_TARGETS,
    FrameTargetIoU,
    disagreement_episodes,
    mask_iou,
)

FRAME_COUNT = 1800
ANALYSIS_FPS = 30.0
TARGETS: tuple[str, ...] = DEFAULT_TARGETS
ANCHOR_FRAMES = (300, 370, 400, 600, 650, 700, 900, 1050, 1100, 1150, 1200, 1500, 1700)
# Human-reported failure windows on the focused C10379 clock (analysis frames, half-open).
WINDOWS: dict[str, tuple[int, int]] = {
    "279-408": (279, 408),
    "573-722": (573, 722),
    "1020-1172": (1020, 1172),
}
PAIR_IOU_THRESHOLD = 0.3
AREA_BAND = (0.5, 2.0)
AREA_HISTORY = 30
STARVATION_FRACTION = 0.30
DEFAULT_ROOT = Path("runs/sam3-policy-ablation-20260918")
DEFAULT_REFERENCE = Path(
    "runs/muggledsam-sam3-four-part-static-focused-reassembly-static-c10379-20260918t001210z"
)
CLAIM = (
    "label-free: every number is cross-run disagreement with a reference that is itself wrong "
    "inside the compared windows, or the arm's self-consistency; none is accuracy"
)


@dataclass
class LoadedRun:
    name: str
    run_directory: Path
    rows: dict[int, dict] = field(default_factory=dict)
    cache: mask_cache.MaskCache = field(init=False)

    def __post_init__(self) -> None:
        self.cache = mask_cache.cache_for(self.run_directory)

    def mask(self, frame: int, target: str) -> np.ndarray | None:
        row = self.rows.get(frame)
        if row is None:
            return None
        for item in row.get("objects", ()):
            if item.get("label") == target and item.get("mask") and item["mask"].get("uri"):
                mask = self.cache.mask(item["mask"]["uri"])
                return mask if mask.any() else None
        return None

    def masks(self, frame: int) -> dict[str, np.ndarray | None]:
        return {target: self.mask(frame, target) for target in TARGETS}

    def diagnostics(self, frame: int) -> list[dict]:
        row = self.rows.get(frame)
        return list(row.get("tracker_diagnostics", ())) if row else []


def load_run(name: str, run_directory: Path, *, frame_count: int = FRAME_COUNT) -> LoadedRun:
    run = LoadedRun(name, run_directory.resolve())
    with (run.run_directory / "observations.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["analysis_frame_index"] < frame_count:
                run.rows[row["analysis_frame_index"]] = row
    return run


def discover_arms(root: Path) -> dict[str, Path]:
    """`<root>/arms/<arm>/<single run directory>` for every arm with a finished run."""
    arms: dict[str, Path] = {}
    arms_root = root / "arms"
    if not arms_root.is_dir():
        return arms
    for arm_directory in sorted(path for path in arms_root.iterdir() if path.is_dir()):
        runs = [
            path
            for path in arm_directory.iterdir()
            if path.is_dir() and (path / "manifest.json").is_file()
        ]
        if len(runs) == 1:
            arms[arm_directory.name] = runs[0]
    return arms


def in_window(frame: int, window: tuple[int, int]) -> bool:
    return window[0] <= frame < window[1]


def in_any_window(frame: int) -> bool:
    return any(in_window(frame, window) for window in WINDOWS.values())


def _mean(values: Iterable[float]) -> float | None:
    items = [value for value in values if value is not None]
    return float(np.mean(items)) if items else None


def run_measures(run_directory: Path) -> dict[str, object]:
    """The five pre-accuracy measures plus the run condition, read from the run's own files."""
    result = json.loads((run_directory / "worker_result.json").read_text(encoding="utf-8"))
    settings = result.get("runtime_settings", {})
    policy = settings.get("tracker_memory_policy", {})
    manifest = json.loads((run_directory / "manifest.json").read_text(encoding="utf-8"))
    metadata = manifest.get("four_part_focused") or manifest.get("four_part_multiview") or {}
    corrections = metadata.get("multi_keyframe_corrections") or {}
    side = int(settings.get("max_side_length", 0))
    return {
        "run_id": run_directory.name,
        "state": result.get("state"),
        "frames_processed": result.get("frames_processed"),
        "masks_written": result.get("masks_written"),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "time_to_first_usable_output_seconds": result.get("time_to_first_usable_output_seconds"),
        "gpu_peak_vram_bytes": result.get("gpu_peak_vram_bytes"),
        "max_side_length": side,
        "mask_grid_side": side // 4,
        "slot_exclusivity": policy.get("slot_exclusivity", "off"),
        "memory_gate": policy.get("memory_gate", "off"),
        "policy": policy,
        "scheduled_correction_frames": corrections.get("scheduled_correction_frame_indices", []),
        "frame_zero_seeds_only": bool(corrections.get("frame_zero_seeds_only", False)),
    }


def coverage(run: LoadedRun, *, frame_count: int = FRAME_COUNT) -> dict[str, int]:
    counts = dict.fromkeys(TARGETS, 0)
    for frame in range(frame_count):
        row = run.rows.get(frame)
        if row is None:
            continue
        for item in row.get("objects", ()):
            if item.get("label") in counts and item.get("mask"):
                counts[item["label"]] += 1
    return counts


def gate_statistics(run: LoadedRun, *, frame_count: int = FRAME_COUNT) -> dict[str, object]:
    """Fraction of tracked frames each slot was not memorised, and why."""
    per_slot: dict[str, dict[str, object]] = {}
    for target in TARGETS:
        frames = 0
        gated = 0
        reasons: dict[str, int] = {}
        contested: list[float] = []
        gated_in_windows = dict.fromkeys(WINDOWS, 0)
        for frame in range(1, frame_count):
            for entry in run.diagnostics(frame):
                if entry.get("label") != target or entry.get("corrected"):
                    continue
                if entry.get("memory_written") is None:
                    continue
                frames += 1
                reason = entry.get("memory_gate_reason") or "ok"
                reasons[reason] = reasons.get(reason, 0) + 1
                if entry.get("contested_fraction") is not None:
                    contested.append(float(entry["contested_fraction"]))
                if not entry["memory_written"]:
                    gated += 1
                    for name, window in WINDOWS.items():
                        if in_window(frame, window):
                            gated_in_windows[name] += 1
        fraction = gated / frames if frames else None
        per_slot[target] = {
            "frames_with_policy_diagnostics": frames,
            "frames_gated": gated,
            "gated_fraction": fraction,
            "starved": bool(fraction is not None and fraction > STARVATION_FRACTION),
            "reasons": dict(sorted(reasons.items())),
            "mean_contested_fraction": _mean(contested),
            "frames_contested_over_0_2": sum(value > 0.2 for value in contested),
            "gated_in_windows": gated_in_windows,
        }
    return {
        "per_slot": per_slot,
        "starved_slots": [target for target, stats in per_slot.items() if stats["starved"]],
    }


def area_jump_frames(areas: Sequence[int | None]) -> list[int]:
    """Frames where a part's area leaves the band around the median of its previous 30 areas.

    The window is every previous frame with a mask, trusted or not, so a real size change is
    flagged for at most a window's worth of frames and then becomes the new normal; this is a
    descriptive trace, unlike the worker's gate, which only advances on frames it trusted.
    """
    jumps: list[int] = []
    history: list[int] = []
    for frame, area in enumerate(areas):
        if area is None or area <= 0:
            continue
        if len(history) >= AREA_HISTORY:
            median = float(np.median(history[-AREA_HISTORY:]))
            ratio = area / median if median > 0 else float("inf")
            if not AREA_BAND[0] <= ratio <= AREA_BAND[1]:
                jumps.append(frame)
        history.append(area)
    return jumps


def _window_counts(frames: Iterable[int]) -> dict[str, int]:
    counts = {name: 0 for name in WINDOWS}
    counts["outside"] = 0
    for frame in frames:
        matched = False
        for name, window in WINDOWS.items():
            if in_window(frame, window):
                counts[name] += 1
                matched = True
        if not matched:
            counts["outside"] += 1
    return counts


def analyse_arm(
    arm: LoadedRun, reference: LoadedRun, *, frame_count: int = FRAME_COUNT
) -> dict[str, object]:
    """One pass over the frames: disagreement rows, pairwise slot IoU and area traces."""
    rows: list[FrameTargetIoU] = []
    pair_frames: dict[str, list[int]] = {}
    any_pair_frames: list[int] = []
    max_pair_iou: list[float] = []
    areas: dict[str, list[int | None]] = {target: [] for target in TARGETS}
    for frame in range(frame_count):
        arm_masks = arm.masks(frame)
        reference_masks = reference.masks(frame)
        for target in TARGETS:
            a, b = arm_masks[target], reference_masks[target]
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
            areas[target].append(int(a.sum()) if a is not None else None)
        frame_max = 0.0
        for index, first in enumerate(TARGETS):
            for second in TARGETS[index + 1 :]:
                left, right = arm_masks[first], arm_masks[second]
                if left is None or right is None:
                    continue
                union = np.logical_or(left, right).sum()
                overlap = float(np.logical_and(left, right).sum() / union) if union else 0.0
                frame_max = max(frame_max, overlap)
                if overlap > PAIR_IOU_THRESHOLD:
                    pair_frames.setdefault(f"{first}/{second}", []).append(frame)
        max_pair_iou.append(frame_max)
        if frame_max > PAIR_IOU_THRESHOLD:
            any_pair_frames.append(frame)

    disagreement: dict[str, dict[str, object]] = {}
    for target in TARGETS:
        selected = [row for row in rows if row.target == target]
        by_window = {
            name: _mean(row.iou for row in selected if in_window(row.frame, window))
            for name, window in WINDOWS.items()
        }
        disagreement[target] = {
            "mean_iou": _mean(row.iou for row in selected),
            "mean_iou_in_windows": by_window,
            "mean_iou_outside_windows": _mean(
                row.iou for row in selected if not in_any_window(row.frame)
            ),
            "frames_below_0_5": sum(1 for row in selected if row.iou is not None and row.iou < 0.5),
            "missing_in_arm": sum(row.status == "missing_in_a" for row in selected),
            "missing_in_reference": sum(row.status == "missing_in_b" for row in selected),
        }
    episodes = disagreement_episodes(rows)
    episode_records = [
        {
            "target": item.target,
            "start_frame": item.start_frame,
            "end_frame_exclusive": item.end_frame_exclusive,
            "frame_count": item.frame_count,
            "mean_iou": round(item.mean_iou, 4),
            "windows": sorted(
                name
                for name, window in WINDOWS.items()
                if item.start_frame < window[1] and item.end_frame_exclusive > window[0]
            ),
        }
        for item in episodes
    ]
    area_jumps = {target: area_jump_frames(trace) for target, trace in areas.items()}
    return {
        "disagreement_vs_reference": disagreement,
        "episodes_vs_reference": {
            "count": len(episodes),
            "count_outside_windows": sum(1 for item in episode_records if not item["windows"]),
            "frames": sum(item.frame_count for item in episodes),
            "items": episode_records,
        },
        "pairwise_slot_iou": {
            "threshold": PAIR_IOU_THRESHOLD,
            "frames_with_any_pair_over_threshold": len(any_pair_frames),
            "frames_by_window": _window_counts(any_pair_frames),
            "frames_per_pair": {pair: len(frames) for pair, frames in sorted(pair_frames.items())},
            "chassis_interior_frames": len(pair_frames.get("chassis/interior", [])),
            "max_pair_iou_overall": float(max(max_pair_iou)) if max_pair_iou else 0.0,
        },
        "area_jump_proxy": {
            "band": list(AREA_BAND),
            "history_frames": AREA_HISTORY,
            "frames_per_part": {target: len(frames) for target, frames in area_jumps.items()},
            "frames_by_window_per_part": {
                target: _window_counts(frames) for target, frames in area_jumps.items()
            },
        },
    }


def review_metric_episodes(
    repository_root: Path, run_directory: Path, output_root: Path
) -> dict[str, object]:
    """`battle-review-metrics` with the arm as the mask source; episode counts per window."""
    from .review_metrics import build_review_metrics

    metrics_path = build_review_metrics(
        repository_root=repository_root,
        output_root=output_root,
        write_rrd=False,
        overwrite=True,
        reference_run=run_directory,
    )
    triggers = json.loads(
        (metrics_path.parent / "review_triggers.json").read_text(encoding="utf-8")
    )
    families = {
        "swap": ("segmentation_identity_swap", "segmentation_label_crossing"),
        "leakage": ("appearance_leakage",),
        "growth": ("mask_growth_hand_capture_suspect", "mask_growth_unexplained"),
        "area_anomaly": ("mask_area_anomaly_vs_median",),
    }
    counts: dict[str, dict[str, int]] = {}
    for family, kinds in families.items():
        episodes = [item for item in triggers["episodes"] if item["episode_type"] in kinds]
        per_window = {
            name: sum(
                1
                for item in episodes
                if item["start_frame"] < window[1] and item["end_frame"] >= window[0]
            )
            for name, window in WINDOWS.items()
        }
        per_window["total"] = len(episodes)
        per_window["outside"] = sum(
            1
            for item in episodes
            if not any(
                item["start_frame"] < window[1] and item["end_frame"] >= window[0]
                for window in WINDOWS.values()
            )
        )
        counts[family] = per_window
    return {"metrics_uri": str(metrics_path), "episodes": counts}


def summarize(
    *,
    repository_root: Path,
    root: Path,
    reference_directory: Path,
    arms: Mapping[str, Path] | None = None,
    with_review_metrics: bool = False,
    frame_count: int = FRAME_COUNT,
) -> dict[str, object]:
    root = (repository_root / root).resolve()
    arms = dict(arms) if arms is not None else discover_arms(root)
    reference = load_run(
        "reference", repository_root / reference_directory, frame_count=frame_count
    )
    summary: dict[str, object] = {
        "schema_version": "1.0",
        "manifest_kind": "sam3_tracker_policy_ablation_summary",
        "claim": CLAIM,
        "ground_truth_accuracy_claim": False,
        "reference_run": str(reference_directory),
        "frame_count": frame_count,
        "windows": {name: list(window) for name, window in WINDOWS.items()},
        "arms": {},
    }
    for name, run_directory in arms.items():
        arm = load_run(name, run_directory, frame_count=frame_count)
        record: dict[str, object] = {
            "run_directory": str(run_directory.relative_to(repository_root))
            if run_directory.is_relative_to(repository_root)
            else str(run_directory),
            "measures": run_measures(run_directory),
            "coverage": coverage(arm, frame_count=frame_count),
            "gate": gate_statistics(arm, frame_count=frame_count),
            **analyse_arm(arm, reference, frame_count=frame_count),
        }
        if with_review_metrics:
            try:
                record["review_metrics"] = review_metric_episodes(
                    repository_root, run_directory, root / "analysis" / name / "review_metrics"
                )
            except Exception as error:  # noqa: BLE001 - recorded, never fatal for the table
                record["review_metrics"] = {"error": f"{type(error).__name__}: {error}"}
        summary["arms"][name] = record
        print(f"{name}: analysed")
    summary["reference"] = {
        "coverage": coverage(reference, frame_count=frame_count),
        "gate": gate_statistics(reference, frame_count=frame_count),
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    (root / "summary.md").write_text(markdown_tables(summary), encoding="utf-8")
    return summary


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_tables(summary: Mapping[str, object]) -> str:
    arms: Mapping[str, Mapping[str, object]] = summary["arms"]  # type: ignore[assignment]
    lines = ["## Arms x label-free metrics", ""]
    header = (
        "| arm | side | policy | sched | IoU ch | IoU in | IoU rb | IoU cab | "
        "IoU ch 279-408 | IoU ch 573-722 | IoU ch 1020-1172 | IoU in 1020-1172 | "
        "IoU outside (mean of parts) | episodes (outside) | pair>0.3 frames (ch/in) | "
        "area jumps ch/in | gated ch/in % |"
    )
    lines += [header, "|" + "---|" * (header.count("|") - 1)]
    for name, record in arms.items():
        measures = record["measures"]
        disagreement = record["disagreement_vs_reference"]
        episodes = record["episodes_vs_reference"]
        pairs = record["pairwise_slot_iou"]
        jumps = record["area_jump_proxy"]["frames_per_part"]
        gate = record["gate"]["per_slot"]
        outside = _mean(part["mean_iou_outside_windows"] for part in disagreement.values())
        policy = (
            "+".join(
                part
                for part, on in (
                    ("x", measures["slot_exclusivity"] != "off"),
                    ("g", measures["memory_gate"] != "off"),
                )
                if on
            )
            or "off"
        )
        gated = "/".join(
            _fmt(100 * gate[t]["gated_fraction"], 0)
            if gate[t]["gated_fraction"] is not None
            else "-"
            for t in ("chassis", "interior")
        )
        lines.append(
            f"| {name} | {measures['max_side_length']} | {policy} | "
            f"{'seed0' if measures['frame_zero_seeds_only'] else 'full'} | "
            + " | ".join(_fmt(disagreement[t]["mean_iou"]) for t in TARGETS)
            + f" | {_fmt(disagreement['chassis']['mean_iou_in_windows']['279-408'])}"
            f" | {_fmt(disagreement['chassis']['mean_iou_in_windows']['573-722'])}"
            f" | {_fmt(disagreement['chassis']['mean_iou_in_windows']['1020-1172'])}"
            f" | {_fmt(disagreement['interior']['mean_iou_in_windows']['1020-1172'])}"
            f" | {_fmt(outside)} | {episodes['count']} ({episodes['count_outside_windows']})"
            f" | {pairs['frames_with_any_pair_over_threshold']} "
            f"({pairs['chassis_interior_frames']})"
            f" | {jumps['chassis']}/{jumps['interior']} | {gated} |"
        )
    lines += ["", "## Pre-accuracy measures", ""]
    lines += [
        "| arm | frames | masks | coverage ch/in/rb/cab | runtime s | ttfu s | "
        "peak VRAM GiB | grid |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, record in arms.items():
        measures = record["measures"]
        cov = record["coverage"]
        vram = measures["gpu_peak_vram_bytes"]
        lines.append(
            f"| {name} | {measures['frames_processed']} | {measures['masks_written']} | "
            f"{cov['chassis']}/{cov['interior']}/{cov['rear_body']}/{cov['cabin']} | "
            f"{_fmt(measures['elapsed_seconds'], 1)} | "
            f"{_fmt(measures['time_to_first_usable_output_seconds'], 2)} | "
            f"{_fmt(vram / 2**30 if vram else None, 2)} | {measures['mask_grid_side']} |"
        )
    if any("review_metrics" in record for record in arms.values()):
        lines += ["", "## Review-metric episodes (arm masks as the reference)", ""]
        lines += [
            "| arm | swap total (279-408 / 573-722 / 1020-1172 / outside) | leakage | growth | "
            "area anomaly |",
            "|---|---|---|---|---|",
        ]
        for name, record in arms.items():
            metrics = record.get("review_metrics", {})
            episodes = metrics.get("episodes")
            if not episodes:
                lines.append(f"| {name} | {metrics.get('error', '-')} | | | |")
                continue

            def cell(family: str) -> str:
                item = episodes[family]
                return (
                    f"{item['total']} ({item['279-408']} / {item['573-722']} / "
                    f"{item['1020-1172']} / {item['outside']})"
                )

            lines.append(
                f"| {name} | {cell('swap')} | {cell('leakage')} | {cell('growth')} | "
                f"{cell('area_anomaly')} |"
            )
    lines += ["", f"Claim: {summary['claim']}.", ""]
    return "\n".join(lines)


# --- contact sheets -------------------------------------------------------------------------


def assembly_crop(
    reference: LoadedRun, frames: Sequence[int], *, dimensions: tuple[int, int], margin: int = 60
) -> tuple[int, int, int, int]:
    """One crop (x0, y0, x1, y1) covering every reference part mask on the anchor frames."""
    width, height = dimensions
    xs: list[int] = []
    ys: list[int] = []
    for frame in frames:
        for mask in reference.masks(frame).values():
            if mask is None:
                continue
            rows, cols = np.nonzero(mask)
            xs += [int(cols.min()), int(cols.max())]
            ys += [int(rows.min()), int(rows.max())]
    if not xs:
        return 0, 0, width, height
    x0, x1 = max(0, min(xs) - margin), min(width, max(xs) + margin)
    y0, y1 = max(0, min(ys) - margin), min(height, max(ys) + margin)
    # Keep a 16:9-ish tile so the sheets line up whatever the parts did.
    crop_width, crop_height = x1 - x0, y1 - y0
    if crop_width / crop_height > 16 / 9:
        needed = int(crop_width * 9 / 16) - crop_height
        y0, y1 = max(0, y0 - needed // 2), min(height, y1 + needed - needed // 2)
    else:
        needed = int(crop_height * 16 / 9) - crop_width
        x0, x1 = max(0, x0 - needed // 2), min(width, x1 + needed - needed // 2)
    return x0, y0, x1, y1


def decode_frames(video_path: Path, frames: Sequence[int]) -> dict[int, np.ndarray]:
    """Sequential decode (index-exact) of the requested frames from a long-GOP proxy."""
    wanted = set(frames)
    capture = cv2.VideoCapture(str(video_path))
    decoded: dict[int, np.ndarray] = {}
    index = 0
    try:
        while wanted and index <= max(wanted):
            ok, frame = capture.read()
            if not ok:
                break
            if index in wanted:
                decoded[index] = frame
                wanted.discard(index)
            index += 1
    finally:
        capture.release()
    if wanted:
        raise RuntimeError(f"proxy ended before frames {sorted(wanted)}")
    return decoded


def render_tile(
    frame_bgr: np.ndarray,
    masks: Mapping[str, np.ndarray | None],
    crop: tuple[int, int, int, int],
    *,
    tile_width: int,
    caption: str,
    gated: Sequence[str] = (),
) -> np.ndarray:
    x0, y0, x1, y1 = crop
    tile = frame_bgr[y0:y1, x0:x1].copy()
    overlay = tile.copy()
    for target in TARGETS:
        mask = masks.get(target)
        if mask is None:
            continue
        color = mask_cache.REVIEW_COLORS[target][::-1]  # RGB -> BGR
        region = mask[y0:y1, x0:x1]
        overlay[region] = (0.65 * overlay[region] + 0.35 * np.array(color)).astype(np.uint8)
        contours, _ = cv2.findContours(
            region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, color, 2)
    scale = tile_width / overlay.shape[1]
    resized = cv2.resize(
        overlay, (tile_width, int(round(overlay.shape[0] * scale))), interpolation=cv2.INTER_AREA
    )
    cv2.putText(
        resized, caption, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA
    )
    cv2.putText(
        resized, caption, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
    )
    if gated:
        text = "gated: " + ",".join(gated)
        cv2.putText(
            resized, text, (6, resized.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3
        )
        cv2.putText(
            resized,
            text,
            (6, resized.shape[0] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (80, 220, 255),
            1,
        )
    return resized


def _gated_parts(run: LoadedRun, frame: int) -> list[str]:
    return [
        entry["label"] for entry in run.diagnostics(frame) if entry.get("memory_written") is False
    ]


def _grid(tiles: Sequence[np.ndarray], columns: int, title: str) -> np.ndarray:
    tile_height, tile_width = tiles[0].shape[:2]
    rows = (len(tiles) + columns - 1) // columns
    header = 34
    canvas = np.full((header + rows * tile_height, columns * tile_width, 3), 24, dtype=np.uint8)
    cv2.putText(
        canvas, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 1, cv2.LINE_AA
    )
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        y = header + row * tile_height
        x = column * tile_width
        canvas[y : y + tile.shape[0], x : x + tile.shape[1]] = tile
    return canvas


def render_sheets(
    *,
    repository_root: Path,
    root: Path,
    reference_directory: Path,
    proxy_path: Path,
    arms: Mapping[str, Path] | None = None,
    best_arm: str | None = None,
    frames: Sequence[int] = ANCHOR_FRAMES,
    tile_width: int = 400,
) -> list[Path]:
    root = (repository_root / root).resolve()
    arms = dict(arms) if arms is not None else discover_arms(root)
    reference = load_run("reference", repository_root / reference_directory)
    decoded = decode_frames(repository_root / proxy_path, frames)
    sample = next(iter(decoded.values()))
    crop = assembly_crop(reference, frames, dimensions=(sample.shape[1], sample.shape[0]))
    sheets_root = root / "sheets"
    sheets_root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    runs = {"reference": reference}
    runs.update({name: load_run(name, path) for name, path in arms.items()})
    for name, run in runs.items():
        tiles = [
            render_tile(
                decoded[frame],
                run.masks(frame),
                crop,
                tile_width=tile_width,
                caption=f"f{frame}",
                gated=_gated_parts(run, frame),
            )
            for frame in frames
        ]
        title = f"{name}: {run.run_directory.name}"
        path = sheets_root / f"{name}.png"
        cv2.imwrite(str(path), _grid(tiles, 5, title))
        written.append(path)
    if best_arm is not None:
        best = runs[best_arm]
        rows = []
        for frame in frames:
            left = render_tile(
                decoded[frame],
                reference.masks(frame),
                crop,
                tile_width=tile_width,
                caption=f"f{frame} reference",
            )
            right = render_tile(
                decoded[frame],
                best.masks(frame),
                crop,
                tile_width=tile_width,
                caption=f"f{frame} {best_arm}",
                gated=_gated_parts(best, frame),
            )
            rows.append(np.hstack([left, right]))
        path = root / "best_vs_reference.png"
        cv2.imwrite(str(path), _grid(rows, 1, f"reference | {best_arm} (crop {crop})"))
        written.append(path)
    return written


# --- Rerun recording -------------------------------------------------------------------------


def export_comparison(
    *,
    repository_root: Path,
    output_path: Path,
    reference_directory: Path,
    arm_name: str,
    arm_directory: Path,
    video_path: Path,
    summary: Mapping[str, object] | None = None,
    frame_count: int = FRAME_COUNT,
) -> Path:
    """One recording: the bounded video once, both runs' part masks as toggleable layers."""
    import rerun as rr
    import rerun.blueprint as rrb

    reference = load_run("reference", repository_root / reference_directory)
    arm = load_run(arm_name, arm_directory)
    root = "world/sam3_policy_ablation"
    rr.init("battle-sam3-policy-ablation", recording_id=f"sam3-policy-{arm_name}")
    output_path = (repository_root / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(output_path))
    rr.log(f"{root}/source/video_asset", rr.AssetVideo(path=str(video_path)), static=True)
    if summary is not None:
        rr.log(
            f"{root}/metadata/summary",
            rr.TextDocument(markdown_tables(summary), media_type="text/markdown"),
            static=True,
        )
    rr.log(
        f"{root}/metadata/claim",
        rr.TextDocument(
            f"reference: {reference.run_directory.name}\n{arm_name}: {arm.run_directory.name}\n"
            f"{CLAIM}",
            media_type="text/plain",
        ),
        static=True,
    )
    layers = {"reference": reference, arm_name: arm}
    for frame in range(frame_count):
        rr.set_time("analysis_frame", sequence=frame)
        rr.set_time("analysis_time", duration=frame / ANALYSIS_FPS)
        rr.log(
            f"{root}/source/video",
            rr.VideoFrameReference(
                seconds=frame / ANALYSIS_FPS, video_reference=f"{root}/source/video_asset"
            ),
        )
        for layer, run in layers.items():
            row = run.rows.get(frame)
            for target in TARGETS:
                path = f"{root}/runs/{layer}/masks/{target}"
                item = (
                    next(
                        (
                            o
                            for o in row.get("objects", ())
                            if o.get("label") == target and o.get("mask")
                        ),
                        None,
                    )
                    if row
                    else None
                )
                if item is None:
                    rr.log(path, rr.Clear(recursive=False))
                    continue
                rr.log(
                    path,
                    rr.EncodedImage(
                        contents=run.cache.rgba_png(
                            item["mask"]["uri"], mask_cache.REVIEW_COLORS[target]
                        ),
                        media_type="image/png",
                        opacity=0.45,
                        draw_order=1.0,
                    ),
                )
            for entry in run.diagnostics(frame):
                label = entry.get("label")
                if label not in TARGETS:
                    continue
                rr.log(
                    f"{root}/diagnostics/{layer}/{label}/object_score",
                    rr.Scalars([float(entry["object_score"])]),
                )
                if entry.get("memory_written") is not None:
                    rr.log(
                        f"{root}/diagnostics/{layer}/{label}/memory_gated",
                        rr.Scalars([0.0 if entry["memory_written"] else 1.0]),
                    )
    views = [
        rrb.Spatial2DView(
            origin=root,
            name=layer,
            contents=("$origin/source/video", f"$origin/runs/{layer}/masks/**"),
        )
        for layer in layers
    ]
    views.append(
        rrb.Spatial2DView(
            origin=root,
            name="both (toggle layers)",
            contents=("$origin/source/video", "$origin/runs/**"),
        )
    )
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(*views),
                rrb.Horizontal(
                    rrb.TimeSeriesView(origin=f"{root}/diagnostics", name="object score / gated"),
                    rrb.TextDocumentView(origin=f"{root}/metadata/summary", name="metrics"),
                ),
                row_shares=[3, 2],
            ),
            rrb.TimePanel(timeline="analysis_time", fps=ANALYSIS_FPS),
            auto_layout=False,
            auto_views=False,
        )
    )
    rr.disconnect()
    return output_path


# --- CLI --------------------------------------------------------------------------------------


def _proxy_for(repository_root: Path, run_directory: Path) -> Path:
    manifest = json.loads((run_directory / "manifest.json").read_text(encoding="utf-8"))
    return repository_root / manifest["clip"]["asset"]["uri"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    subparsers = parser.add_subparsers(dest="command", required=True)
    summarize_parser = subparsers.add_parser("summarize", help="summary.json + summary.md")
    summarize_parser.add_argument("--arm", action="append", default=[], metavar="NAME=RUN_DIR")
    summarize_parser.add_argument("--review-metrics", action="store_true")
    sheets_parser = subparsers.add_parser("sheets", help="anchor-frame contact sheets")
    sheets_parser.add_argument("--best", help="arm for best_vs_reference.png")
    sheets_parser.add_argument("--arm", action="append", default=[], metavar="NAME=RUN_DIR")
    rrd_parser = subparsers.add_parser("rrd", help="best arm vs reference recording")
    rrd_parser.add_argument("--best", required=True)
    rrd_parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = (args.repository_root / args.root).resolve()
    explicit = (
        {item.split("=", 1)[0]: Path(item.split("=", 1)[1]).resolve() for item in args.arm}
        if getattr(args, "arm", None)
        else None
    )
    if args.command == "summarize":
        summary = summarize(
            repository_root=args.repository_root,
            root=args.root,
            reference_directory=args.reference,
            arms=explicit,
            with_review_metrics=args.review_metrics,
        )
        print(markdown_tables(summary))
        print(root / "summary.json")
    elif args.command == "sheets":
        proxy = _proxy_for(args.repository_root, args.repository_root / args.reference)
        for path in render_sheets(
            repository_root=args.repository_root,
            root=args.root,
            reference_directory=args.reference,
            proxy_path=proxy,
            arms=explicit,
            best_arm=args.best,
        ):
            print(path)
    elif args.command == "rrd":
        arms = discover_arms(root)
        arm_directory = arms[args.best]
        summary_path = root / "summary.json"
        summary = (
            json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else None
        )
        video = arm_directory / f"input_{FRAME_COUNT}f.mp4"
        path = export_comparison(
            repository_root=args.repository_root,
            output_path=args.output or (args.root / "best_vs_reference.rrd"),
            reference_directory=args.reference,
            arm_name=args.best,
            arm_directory=arm_directory,
            video_path=video,
            summary=summary,
        )
        print(path)
        print(f"rerun {path}")


if __name__ == "__main__":
    main()
