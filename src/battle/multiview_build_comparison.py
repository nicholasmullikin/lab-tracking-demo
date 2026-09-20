"""Compare two consensus + visual-hull builds of the same minute (e.g. 720 px vs 1280 px runs).

Both builders write typed manifests and a per-frame JSONL; this reads two of each and lays the
label-free numbers side by side: where the reference view (C10379) is contradicted by the
majority and for how many frames, per-view agreement, episode counts, hull coverage and the
hull-vs-mask IoU of the reference view overall and inside the human-reported windows. Every
number is disagreement between views of the same tracker, never accuracy; a change between
the two builds says the views moved relative to each other, not which side is right.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from statistics import median
from typing import Literal

from pydantic import Field

from .multiview_schemas import MultiviewConsensusManifest, VisualHullManifest
from .policy_ablation import WINDOWS
from .schemas import ArtifactFingerprint, VersionedModel

PARTS: tuple[str, ...] = ("chassis", "rear_body", "cabin")


class IntervalList(VersionedModel):
    intervals: tuple[tuple[int, int], ...]
    frames: int = Field(ge=0)


class ConsensusSide(VersionedModel):
    root: str
    manifest: ArtifactFingerprint
    reference_view: str
    views: tuple[str, ...]
    reference_contradicted: dict[str, IntervalList]
    reference_agreement: dict[str, float]
    per_view_agreement: dict[str, dict[str, float]]
    episodes_total: int = Field(ge=0)
    episodes_reference: int = Field(ge=0)
    runtime_seconds: float = Field(ge=0)


class HullSide(VersionedModel):
    root: str
    manifest: ArtifactFingerprint
    views_used: tuple[str, ...]
    frames_with_hull: dict[str, int]
    median_voxel_count: dict[str, float]
    reference_median_iou: dict[str, float | None]
    reference_window_median_iou: dict[str, dict[str, float | None]]
    reference_window_frames: dict[str, dict[str, int]]
    episodes_total: int = Field(ge=0)
    episodes_reference: int = Field(ge=0)
    reference_episodes: dict[str, tuple[tuple[int, int], ...]]
    runtime_seconds: float = Field(ge=0)


class MultiviewBuildComparison(VersionedModel):
    manifest_kind: Literal["multiview_build_comparison"]
    label_before: str
    label_after: str
    consensus_before: ConsensusSide
    consensus_after: ConsensusSide
    hull_before: HullSide | None
    hull_after: HullSide | None
    claim_boundaries: tuple[str, ...] = (
        "Consensus and hull numbers are disagreement between views of one tracker, not "
        "accuracy; a change between builds says the views moved relative to each other.",
        "Seeds on the non-reference views are agent-authored geometric transfers; only the "
        "reference view carries human frame-0 seeds and corrections.",
        "Assembly101 is CC BY-NC 4.0.",
    )


def _fingerprint(path: Path, repository_root: Path) -> ArtifactFingerprint:
    from .digest_cache import sha256_file

    return ArtifactFingerprint(
        uri=path.resolve().relative_to(repository_root.resolve()).as_posix(),
        sha256=sha256_file(path),
        source="measured",
    )


def _merge(intervals: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)


def summarize_consensus(root: Path, repository_root: Path) -> ConsensusSide:
    manifest_path = repository_root / root / "manifest.json"
    manifest = MultiviewConsensusManifest.model_validate_json(manifest_path.read_text())
    reference = manifest.reference_view
    contradicted: dict[str, IntervalList] = {}
    for part in PARTS:
        intervals = _merge(
            (start, end)
            for target, start, end in manifest.reference_contradiction_intervals
            if target == part
        )
        contradicted[part] = IntervalList(
            intervals=intervals, frames=sum(end - start for start, end in intervals)
        )
    agreement = {summary.target: summary.per_view_agreement for summary in manifest.summaries}
    return ConsensusSide(
        root=root.as_posix(),
        manifest=_fingerprint(manifest_path, repository_root),
        reference_view=reference,
        views=tuple(source.view for source in manifest.sources),
        reference_contradicted=contradicted,
        reference_agreement={
            part: agreement.get(part, {}).get(reference, float("nan")) for part in PARTS
        },
        per_view_agreement={part: agreement.get(part, {}) for part in PARTS},
        episodes_total=len(manifest.episodes),
        episodes_reference=sum(1 for e in manifest.episodes if e.view == reference),
        runtime_seconds=manifest.runtime_seconds,
    )


def summarize_hull(root: Path, repository_root: Path, reference: str) -> HullSide:
    manifest_path = repository_root / root / "manifest.json"
    manifest = VisualHullManifest.model_validate_json(manifest_path.read_text())
    window_values: dict[str, dict[str, list[float]]] = {
        part: {name: [] for name in WINDOWS} for part in PARTS
    }
    with (repository_root / manifest.per_frame_uri).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            frame = int(record["analysis_frame_index"])
            for part in PARTS:
                iou = record["parts"].get(part, {}).get("view_iou", {}).get(reference)
                if iou is None:
                    continue
                for name, (low, high) in WINDOWS.items():
                    if low <= frame < high:
                        window_values[part][name].append(float(iou))
    reference_median = {
        comparison.target: comparison.median_iou
        for comparison in manifest.view_comparisons
        if comparison.view == reference
    }
    reference_episodes = {
        part: _merge(
            (e.start_frame, e.end_frame_exclusive)
            for e in manifest.episodes
            if e.view == reference and e.target == part
        )
        for part in PARTS
    }
    return HullSide(
        root=root.as_posix(),
        manifest=_fingerprint(manifest_path, repository_root),
        views_used=manifest.views_used,
        frames_with_hull={s.target: s.frames_with_hull for s in manifest.part_summaries},
        median_voxel_count={s.target: s.median_voxel_count for s in manifest.part_summaries},
        reference_median_iou={part: reference_median.get(part) for part in PARTS},
        reference_window_median_iou={
            part: {
                name: (round(median(values), 3) if values else None)
                for name, values in windows.items()
            }
            for part, windows in window_values.items()
        },
        reference_window_frames={
            part: {name: len(values) for name, values in windows.items()}
            for part, windows in window_values.items()
        },
        episodes_total=len(manifest.episodes),
        episodes_reference=sum(1 for e in manifest.episodes if e.view == reference),
        reference_episodes=reference_episodes,
        runtime_seconds=manifest.runtime_seconds,
    )


def compare_builds(
    *,
    repository_root: Path,
    consensus_before: Path,
    consensus_after: Path,
    hull_before: Path | None,
    hull_after: Path | None,
    label_before: str,
    label_after: str,
) -> MultiviewBuildComparison:
    before = summarize_consensus(consensus_before, repository_root)
    after = summarize_consensus(consensus_after, repository_root)
    return MultiviewBuildComparison(
        manifest_kind="multiview_build_comparison",
        label_before=label_before,
        label_after=label_after,
        consensus_before=before,
        consensus_after=after,
        hull_before=(
            summarize_hull(hull_before, repository_root, before.reference_view)
            if hull_before is not None
            else None
        ),
        hull_after=(
            summarize_hull(hull_after, repository_root, after.reference_view)
            if hull_after is not None
            else None
        ),
    )


def _intervals(items: IntervalList) -> str:
    if not items.intervals:
        return "none (0 frames)"
    return " ".join(f"`[{s},{e})`" for s, e in items.intervals) + f" ({items.frames} frames)"


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_markdown(comparison: MultiviewBuildComparison) -> str:
    b, a = comparison.consensus_before, comparison.consensus_after
    lines = [
        f"# Multi-view builds: {comparison.label_before} vs {comparison.label_after}",
        "",
        "Label-free: every number is disagreement between views of the same tracker "
        "(agent-seeded on every view but the reference), not accuracy.",
        "",
        f"- consensus before: `{b.root}` ({', '.join(b.views)}; {b.runtime_seconds:.0f} s)",
        f"- consensus after: `{a.root}` ({', '.join(a.views)}; {a.runtime_seconds:.0f} s)",
    ]
    if comparison.hull_before and comparison.hull_after:
        hb, ha = comparison.hull_before, comparison.hull_after
        lines += [
            f"- hull before: `{hb.root}` ({hb.runtime_seconds:.0f} s)",
            f"- hull after: `{ha.root}` ({ha.runtime_seconds:.0f} s)",
        ]
    lines += [
        "",
        "## Consensus",
        "",
        f"| measure | {comparison.label_before} | {comparison.label_after} |",
        "|---|---|---|",
    ]
    for part in PARTS:
        lines.append(
            f"| {b.reference_view} {part} contradicted by the majority | "
            f"{_intervals(b.reference_contradicted[part])} | "
            f"{_intervals(a.reference_contradicted[part])} |"
        )
    lines.append(
        f"| {b.reference_view} agreement {' / '.join(PARTS)} | "
        + " / ".join(f"{b.reference_agreement[p]:.2f}" for p in PARTS)
        + " | "
        + " / ".join(f"{a.reference_agreement[p]:.2f}" for p in PARTS)
        + " |"
    )
    lines.append(
        f"| consensus episodes, all views ({b.reference_view}) | "
        f"{b.episodes_total} ({b.episodes_reference}) | "
        f"{a.episodes_total} ({a.episodes_reference}) |"
    )
    views = sorted(set(b.views) | set(a.views))
    lines += ["", "Per-view agreement (chassis / rear_body / cabin):", ""]
    lines += [f"| view | {comparison.label_before} | {comparison.label_after} |", "|---|---|---|"]
    for view in views:

        def cell(side: ConsensusSide) -> str:
            values = [side.per_view_agreement[p].get(view) for p in PARTS]
            if all(v is None for v in values):
                return "-"
            return " / ".join("-" if v is None else f"{v:.2f}" for v in values)

        lines.append(f"| {view} | {cell(b)} | {cell(a)} |")
    if comparison.hull_before and comparison.hull_after:
        hb, ha = comparison.hull_before, comparison.hull_after
        lines += [
            "",
            "## Visual hull",
            "",
            f"| measure | {comparison.label_before} | {comparison.label_after} |",
            "|---|---|---|",
            "| frames with hull "
            + " / ".join(PARTS)
            + " | "
            + " / ".join(str(hb.frames_with_hull.get(p, 0)) for p in PARTS)
            + " | "
            + " / ".join(str(ha.frames_with_hull.get(p, 0)) for p in PARTS)
            + " |",
            "| median voxels "
            + " / ".join(PARTS)
            + " | "
            + " / ".join(f"{hb.median_voxel_count.get(p, 0):.0f}" for p in PARTS)
            + " | "
            + " / ".join(f"{ha.median_voxel_count.get(p, 0):.0f}" for p in PARTS)
            + " |",
        ]
        for part in PARTS:
            lines.append(
                f"| hull-vs-mask median IoU, {b.reference_view} {part}, overall | "
                f"{_fmt(hb.reference_median_iou[part])} | {_fmt(ha.reference_median_iou[part])} |"
            )
            lines.append(
                f"| hull-vs-mask median IoU, {b.reference_view} {part}, "
                + " / ".join(WINDOWS)
                + " | "
                + " / ".join(_fmt(hb.reference_window_median_iou[part][w]) for w in WINDOWS)
                + " | "
                + " / ".join(_fmt(ha.reference_window_median_iou[part][w]) for w in WINDOWS)
                + " |"
            )
        lines.append(
            f"| hull episodes, all views ({b.reference_view}) | "
            f"{hb.episodes_total} ({hb.episodes_reference}) | "
            f"{ha.episodes_total} ({ha.episodes_reference}) |"
        )
        for part in PARTS:
            lines.append(
                f"| {b.reference_view} hull {part} episodes | "
                + (" ".join(f"`[{s},{e})`" for s, e in hb.reference_episodes[part]) or "none")
                + " | "
                + (" ".join(f"`[{s},{e})`" for s, e in ha.reference_episodes[part]) or "none")
                + " |"
            )
    lines += ["", "Claim boundaries:", ""]
    lines += [f"- {line}" for line in comparison.claim_boundaries]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--consensus-before", type=Path, required=True)
    parser.add_argument("--consensus-after", type=Path, required=True)
    parser.add_argument("--hull-before", type=Path, default=None)
    parser.add_argument("--hull-after", type=Path, default=None)
    parser.add_argument("--label-before", default="before")
    parser.add_argument("--label-after", default="after")
    parser.add_argument("--output", type=Path, required=True, help="markdown summary path")
    parser.add_argument("--json", type=Path, default=None, help="typed JSON (default beside md)")
    args = parser.parse_args(argv)
    comparison = compare_builds(
        repository_root=args.repository_root,
        consensus_before=args.consensus_before,
        consensus_after=args.consensus_after,
        hull_before=args.hull_before,
        hull_after=args.hull_after,
        label_before=args.label_before,
        label_after=args.label_after,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(comparison), encoding="utf-8")
    json_path = args.json or args.output.with_suffix(".json")
    json_path.write_text(comparison.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output} and {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
