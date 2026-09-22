"""Assemble a per-target ensemble review reference with explicit mask provenance.

The ensemble copies whole masks from the corrected SAM3 reference by default, substitutes the
DAM4SAM arm's mask for one target only inside explicit policy intervals when a rule fires and
the substitute passes sanity, and writes explicit empty masks over agent-labelled hidden
intervals.  Nothing is blended.  Every frame x target carries its provenance, every
substitution and every declined attempt is recorded with a rationale, and the output keeps the
observation/mask schema the interaction review consumes.  It is a review reference: cross-method
fallback is not accuracy and neither source run is ground truth.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import interaction_review as review
from .cli_common import (
    add_output_flags,
    add_output_root,
    add_repository_root,
    open_output_directory,
)
from .ensemble_reference_schemas import (
    PROVENANCE_CODES,
    EnsembleFrameProvenance,
    EnsembleProvenanceSidecar,
    EnsembleReferencePolicy,
    EnsembleSubstitutionRecord,
    EnsembleTargetPolicy,
    EnsembleTargetSummary,
)
from .exploratory_comparison import _file_fingerprint
from .four_part_contract import ANALYSIS_FPS, TARGETS
from .mask_ops import mask_area, mask_centroid, mask_iou
from .observations import object_for_label
from .schemas import (
    ArtifactFingerprint,
    EnsembleReferenceRunMetadata,
    EnsembleSourceRunReference,
    FrameObservations,
    FrameRange,
    FullDurationCoverage,
    MaskReference,
    MaskStorage,
    MethodState,
    MethodStatus,
    NormalizedBox,
    PerFrameObject,
    RunManifest,
    SegmentationValidityInterval,
    TimeInterval,
)

FRAME_COUNT = 1800
POLICY_PATH = Path("configs/ensemble_reference/first_minute_v1.json")
OUTPUT_ROOT = Path("runs/ensemble-reference-first-minute-v1")
# Same checked-in thresholds the review-metrics swap/area detectors use.
METRICS_CONFIG_PATH = Path("configs/review_metrics/first_minute_v1.json")
# The 60 s bounded proxy the v4 review embeds; used only to render evidence sheets.
VIDEO_PATH = Path("runs/wilor-hands-static-60s-overnight-v2/input.mp4")
MASK_DIR = "masks"
SHEET_DIR = "sheets"
PROVENANCE_NAME = "ensemble_provenance.json"
EPISODE_CHECK_NAME = "segmentation_episode_check.json"
DEFAULT_SHEET_RANGE = (1000, 1250)
SHEET_STRIDE = 5
PROVENANCE_COLORS: dict[str, tuple[int, int, int]] = {
    "sam3_corrected": (80, 160, 255),
    "dam4sam_fallback": (255, 80, 220),
    "hidden_agent_label": (255, 200, 40),
    "missing": (160, 160, 160),
}
PART_COLORS: dict[str, tuple[int, int, int]] = {
    "chassis": (70, 130, 255),
    "interior": (60, 210, 150),
    "rear_body": (255, 190, 45),
    "cabin": (255, 95, 100),
}
SEGMENTATION_EPISODE_TYPES = (
    "segmentation_identity_swap",
    "segmentation_label_crossing",
    "mask_area_anomaly_vs_median",
)


# --------------------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------------------


def cluster_intervals(frames: Iterable[int]) -> tuple[tuple[int, int], ...]:
    """Half-open `[start, end)` runs of consecutive frames."""
    runs: list[tuple[int, int]] = []
    start = previous = None
    for frame in sorted(set(frames)):
        if start is None or previous is None:
            start = previous = frame
        elif frame != previous + 1:
            runs.append((start, previous + 1))
            start = previous = frame
        else:
            previous = frame
    if start is not None and previous is not None:
        runs.append((start, previous + 1))
    return tuple(runs)


def fallback_intervals_from_contradictions(
    intervals: Iterable[tuple[int, int]],
    *,
    merge_gap_below_frames: int = 15,
    min_interval_frames: int = 10,
) -> tuple[tuple[int, int], ...]:
    """Label-free fallback intervals from consensus contradiction runs of one part.

    Sorted half-open runs are merged when the gap between them is shorter than
    `merge_gap_below_frames`, then runs shorter than `min_interval_frames` are dropped (merge
    first, so two short neighbours can survive together).  The rule is the whole selection:
    nothing about where a human anchor sits enters here.
    """
    merged: list[list[int]] = []
    for start, end in sorted((int(s), int(e)) for s, e in intervals):
        if end <= start:
            raise ValueError(f"empty or inverted interval [{start},{end})")
        if merged and start - merged[-1][1] < merge_gap_below_frames:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return tuple((s, e) for s, e in merged if e - s >= min_interval_frames)


def _r(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)


# --------------------------------------------------------------------------------------
# Decision engine
# --------------------------------------------------------------------------------------


@dataclass
class AcceptedMask:
    frame: int
    mask: np.ndarray
    centroid: tuple[float, float]
    area: int


@dataclass
class TargetState:
    """Rolling state for one target: the last sane accepted mask and accepted areas."""

    last_sane: AcceptedMask | None = None
    accepted_areas: deque[tuple[int, int]] = field(default_factory=deque)


@dataclass(frozen=True)
class Decision:
    mask: np.ndarray | None
    record: EnsembleFrameProvenance
    substitution: EnsembleSubstitutionRecord | None = None


class EnsembleDecider:
    """Pure per-frame decision logic; masks in, provenance out, no I/O."""

    def __init__(self, policy: EnsembleReferencePolicy) -> None:
        self.policy = policy
        self.states = {target.target_id: TargetState() for target in policy.targets}

    def rolling_median(self, target_id: str, frame: int) -> float | None:
        state = self.states[target_id]
        window = self.policy.rolling_median_window_frames
        recent = [area for f, area in state.accepted_areas if frame - window <= f < frame]
        if len(recent) < self.policy.rolling_median_min_samples:
            recent = [area for _, area in state.accepted_areas]
        return float(np.median(recent)) if recent else None

    def _continuity(
        self, target: EnsembleTargetPolicy, frame: int, mask: np.ndarray
    ) -> tuple[float | None, float | None, bool]:
        """IoU, centroid jump and pass/fail of `mask` against the last sane accepted mask."""
        last = self.states[target.target_id].last_sane
        if last is None:
            return None, None, True
        centroid = mask_centroid(mask)
        if centroid is None:
            return None, None, False
        iou = mask_iou(mask, last.mask)
        jump = math.hypot(centroid[0] - last.centroid[0], centroid[1] - last.centroid[1])
        allowance = target.sanity.continuity_allowance(frame - last.frame)
        passes = iou >= target.sanity.min_iou_with_last_accepted or jump <= allowance
        return iou, jump, passes

    def _rules(
        self,
        target: EnsembleTargetPolicy,
        frame: int,
        primary: np.ndarray | None,
        others: dict[str, np.ndarray | None],
        median: float | None,
    ) -> tuple[list[str], float | None, float | None]:
        rules = target.rules
        fired: list[str] = []
        if primary is None or not primary.any():
            if rules.substitute_when_primary_missing:
                fired.append("primary_missing")
            return fired, None, None
        area = mask_area(primary)
        if (
            rules.area_fraction_of_rolling_median_below is not None
            and median is not None
            and area < rules.area_fraction_of_rolling_median_below * median
        ):
            fired.append("area_below_rolling_median_fraction")
        if rules.other_target_iou_above is not None:
            overlaps = [
                mask_iou(primary, other)
                for other in others.values()
                if other is not None and other.any()
            ]
            if overlaps and max(overlaps) > rules.other_target_iou_above:
                fired.append("other_target_iou_above_threshold")
        iou, jump, passes = self._continuity(target, frame, primary)
        if rules.discontinuity_with_last_accepted and iou is not None and not passes:
            fired.append("discontinuous_with_last_accepted")
        return fired, iou, jump

    def _accept(self, target_id: str, frame: int, mask: np.ndarray) -> None:
        centroid = mask_centroid(mask)
        assert centroid is not None
        state = self.states[target_id]
        state.last_sane = AcceptedMask(frame, mask, centroid, mask_area(mask))
        state.accepted_areas.append((frame, mask_area(mask)))
        window = self.policy.rolling_median_window_frames
        while state.accepted_areas and state.accepted_areas[0][0] < frame - 2 * window:
            state.accepted_areas.popleft()

    def decide(
        self,
        frame: int,
        target_id: str,
        primary: np.ndarray | None,
        fallback: np.ndarray | None,
        other_primary: dict[str, np.ndarray | None],
    ) -> Decision:
        target = self.policy.target(target_id)
        eligible_frame = frame < self.policy.contact_eligible_through_frame and not (
            target.in_intervals(frame, target.not_contact_eligible_intervals)
        )
        primary_area = None if primary is None else mask_area(primary)
        fallback_area = None if fallback is None else mask_area(fallback)
        if target.in_intervals(frame, target.hidden_intervals):
            return Decision(
                None,
                EnsembleFrameProvenance(
                    analysis_frame_index=frame,
                    target_id=target_id,  # type: ignore[arg-type]
                    provenance="hidden_agent_label",
                    area_pixels=0,
                    inside_fallback_interval=False,
                    sane=False,
                    contact_eligible=False,
                    primary_area_pixels=primary_area,
                    fallback_area_pixels=fallback_area,
                    rationale="agent-authored hidden interval: explicit empty mask",
                ),
            )
        inside = target.in_intervals(frame, target.fallback_intervals)
        median = self.rolling_median(target_id, frame)
        fired, iou, jump = self._rules(target, frame, primary, other_primary, median)
        common = {
            "analysis_frame_index": frame,
            "target_id": target_id,
            "rules_fired": tuple(fired),
            "inside_fallback_interval": inside,
            "primary_area_pixels": primary_area,
            "fallback_area_pixels": fallback_area,
            "rolling_median_area_pixels": _r(median, 1),
        }
        primary_present = primary is not None and primary.any()
        if not inside or not fired:
            # Default: the corrected SAM3 mask stands. Outside fallback intervals rule firings
            # are diagnostics only; the human-reviewed primary stays authoritative.
            if not primary_present:
                return Decision(
                    None,
                    EnsembleFrameProvenance(
                        **common,
                        provenance="missing",
                        area_pixels=0,
                        sane=False,
                        contact_eligible=False,
                        rationale="primary mask missing and no fallback interval applies",
                    ),
                )
            assert primary is not None
            self._accept(target_id, frame, primary)
            return Decision(
                primary,
                EnsembleFrameProvenance(
                    **common,
                    provenance="sam3_corrected",
                    area_pixels=mask_area(primary),
                    sane=True,
                    contact_eligible=eligible_frame,
                    iou_with_last_accepted=_r(iou),
                    centroid_jump_pixels=_r(jump, 2),
                ),
            )
        # Inside a fallback interval with at least one rule firing: try the fallback mask.
        decline_reason: str | None = None
        f_iou = f_jump = None
        if fallback is None or not fallback.any():
            decline_reason = "fallback mask missing"
        else:
            low, high = target.sanity.area_fraction_bounds
            f_iou, f_jump, continuous = self._continuity(target, frame, fallback)
            if median is not None and not (low * median <= fallback_area <= high * median):
                decline_reason = (
                    f"fallback area {fallback_area} px outside [{low:.2f}, {high:.2f}] x "
                    f"rolling median {median:.0f} px"
                )
            elif not continuous:
                decline_reason = (
                    f"fallback discontinuous with the last sane accepted mask (IoU "
                    f"{f_iou:.2f}, centroid jump {f_jump:.1f} px)"
                )
        if decline_reason is None:
            assert fallback is not None
            self._accept(target_id, frame, fallback)
            rationale = (
                f"rules {', '.join(fired)} fired on the primary mask "
                f"({primary_area if primary_area is not None else 'missing'} px vs rolling "
                f"median {median:.0f} px)"
                if median is not None
                else f"rules {', '.join(fired)} fired on the primary mask"
            ) + f"; DAM4SAM mask {fallback_area} px passed the sanity bounds"
            record = EnsembleSubstitutionRecord(
                **{k: v for k, v in common.items() if k not in ("inside_fallback_interval",)},
                outcome="substituted",
                iou_with_last_accepted=_r(f_iou),
                centroid_jump_pixels=_r(f_jump, 2),
                rationale=rationale,
            )
            return Decision(
                fallback,
                EnsembleFrameProvenance(
                    **common,
                    provenance="dam4sam_fallback",
                    area_pixels=mask_area(fallback),
                    sane=True,
                    contact_eligible=eligible_frame,
                    iou_with_last_accepted=_r(f_iou),
                    centroid_jump_pixels=_r(f_jump, 2),
                    rationale=rationale,
                ),
                record,
            )
        rationale = f"rules {', '.join(fired)} fired but substitution declined: {decline_reason}"
        declined = EnsembleSubstitutionRecord(
            **{k: v for k, v in common.items() if k not in ("inside_fallback_interval",)},
            outcome="declined",
            iou_with_last_accepted=_r(f_iou),
            centroid_jump_pixels=_r(f_jump, 2),
            rationale=rationale,
        )
        if not primary_present:
            return Decision(
                None,
                EnsembleFrameProvenance(
                    **common,
                    provenance="missing",
                    area_pixels=0,
                    sane=False,
                    contact_eligible=False,
                    rationale=rationale + "; primary mask also missing",
                ),
                declined,
            )
        assert primary is not None
        return Decision(
            primary,
            EnsembleFrameProvenance(
                **common,
                provenance="sam3_corrected",
                area_pixels=mask_area(primary),
                sane=False,
                contact_eligible=False,
                iou_with_last_accepted=_r(iou),
                centroid_jump_pixels=_r(jump, 2),
                rationale=rationale + "; primary kept but flagged unsane",
            ),
            declined,
        )


# --------------------------------------------------------------------------------------
# Summaries and outputs
# --------------------------------------------------------------------------------------


def summarize_target(
    target_id: str,
    records: Sequence[EnsembleFrameProvenance],
    declined: Sequence[EnsembleSubstitutionRecord],
) -> EnsembleTargetSummary:
    rows = [r for r in records if r.target_id == target_id]
    counts = {name: 0 for name in PROVENANCE_CODES}
    for row in rows:
        counts[row.provenance] += 1
    return EnsembleTargetSummary(
        target_id=target_id,  # type: ignore[arg-type]
        provenance_counts=counts,
        substituted_intervals=cluster_intervals(
            r.analysis_frame_index for r in rows if r.provenance == "dam4sam_fallback"
        ),
        declined_intervals=cluster_intervals(
            d.analysis_frame_index for d in declined if d.target_id == target_id
        ),
        hidden_intervals=cluster_intervals(
            r.analysis_frame_index for r in rows if r.provenance == "hidden_agent_label"
        ),
        missing_intervals=cluster_intervals(
            r.analysis_frame_index for r in rows if r.provenance == "missing"
        ),
        contact_eligible_intervals=cluster_intervals(
            r.analysis_frame_index for r in rows if r.contact_eligible
        ),
        rule_firings_outside_fallback_intervals=sum(
            1 for r in rows if r.rules_fired and not r.inside_fallback_interval
        ),
    )


def load_policy(path: Path) -> EnsembleReferencePolicy:
    policy = EnsembleReferencePolicy.model_validate_json(path.read_text(encoding="utf-8"))
    if not policy.no_blend:
        raise ValueError("the ensemble policy must keep no_blend true")
    return policy


def contact_eligible_intervals_by_part(
    sidecar: EnsembleProvenanceSidecar,
) -> dict[str, tuple[tuple[int, int], ...]]:
    return {item.target_id: item.contact_eligible_intervals for item in sidecar.summaries}


def load_sidecar(run_directory: Path) -> EnsembleProvenanceSidecar:
    return EnsembleProvenanceSidecar.model_validate_json(
        (run_directory / PROVENANCE_NAME).read_text(encoding="utf-8")
    )


def _frame_class(
    record: EnsembleFrameProvenance, target: EnsembleTargetPolicy, through: int
) -> str:
    """Why a frame x target is or is not contact-eligible, as a stable class label."""
    if record.contact_eligible:
        return record.provenance
    if record.provenance == "hidden_agent_label":
        return "hidden"
    if record.provenance == "missing":
        return "missing"
    if record.analysis_frame_index >= through:
        return "late_boundary"
    if target.in_intervals(record.analysis_frame_index, target.not_contact_eligible_intervals):
        return "policy_ineligible"
    if not record.sane:
        return "declined_substitution"
    return "ineligible"


def validity_intervals_from_sidecar(
    sidecar: EnsembleProvenanceSidecar, policy: EnsembleReferencePolicy
) -> tuple[SegmentationValidityInterval, ...]:
    """Per-target contact-eligibility intervals with the policy's rationales attached."""
    through = policy.contact_eligible_through_frame
    rationales = {
        "sam3_corrected": (
            "Ensemble mask copied from the corrected SAM3 reference; no policy rule fired "
            "inside a fallback interval, so the human/agent-reviewed primary stands."
        ),
        "dam4sam_fallback": (
            "Ensemble mask copied whole from the DAM4SAM arm because a policy rule fired on the "
            "SAM3 mask and the substitute passed the area/continuity sanity bounds "
            "(cross-method fallback, not accuracy)."
        ),
        "hidden": "hidden",
        "missing": "No source run provides a mask for this target on these frames.",
        "late_boundary": policy.late_ineligibility_rationale,
        "policy_ineligible": "policy_ineligible",
        "declined_substitution": (
            "A policy rule fired on the SAM3 mask but the DAM4SAM substitute failed the sanity "
            "bounds; the SAM3 mask is kept for display and flagged unsane, so contact fields "
            "are invalid_mask."
        ),
        "ineligible": "Not contact-eligible under the ensemble policy.",
    }
    intervals: list[SegmentationValidityInterval] = []
    for target in policy.targets:
        rows = sorted(
            (r for r in sidecar.frames if r.target_id == target.target_id),
            key=lambda r: r.analysis_frame_index,
        )
        if [r.analysis_frame_index for r in rows] != list(range(sidecar.frame_count)):
            raise ValueError(f"sidecar must hold exactly one {target.target_id} row per frame")
        start = 0
        current = _frame_class(rows[0], target, through) if rows else None
        for index in range(1, len(rows) + 1):
            klass = _frame_class(rows[index], target, through) if index < len(rows) else None
            if klass == current:
                continue
            assert current is not None
            rationale = rationales[current]
            if current == "hidden":
                rationale = next(
                    item.rationale
                    for item in target.hidden_intervals
                    if item.start_frame <= start < item.end_frame_exclusive
                )
            elif current == "policy_ineligible":
                rationale = next(
                    item.rationale
                    for item in target.not_contact_eligible_intervals
                    if item.start_frame <= start < item.end_frame_exclusive
                )
            intervals.append(
                SegmentationValidityInterval(
                    start_frame=start,
                    end_frame_exclusive=index,
                    state=(
                        "contact_eligible"
                        if current in ("sam3_corrected", "dam4sam_fallback")
                        else "not_contact_eligible"
                    ),
                    rationale=rationale,
                    provenance="ensemble_reference_policy",
                    target_id=target.target_id,
                )
            )
            start, current = index, klass
    return tuple(intervals)


def _mask_box(mask: np.ndarray, dimensions: tuple[int, int]) -> NormalizedBox:
    ys, xs = np.nonzero(mask)
    width, height = dimensions
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return NormalizedBox(
        x=x0 / width, y=y0 / height, width=(x1 - x0) / width, height=(y1 - y0) / height
    )


def _write_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path, optimize=True)


def _source_object(observation: FrameObservations, part: str) -> PerFrameObject | None:
    return object_for_label(observation, part, require_mask=False)


def _run_reference(source: review.LoadedSource, label: str, repository_root: Path):
    return EnsembleSourceRunReference(
        label=label,  # type: ignore[arg-type]
        run_directory_uri=source.run_directory.relative_to(repository_root).as_posix(),
        manifest_fingerprint=_file_fingerprint(
            source.run_directory / "manifest.json", repository_root
        ),
        observations_fingerprint=_file_fingerprint(
            source.run_directory / "observations.jsonl", repository_root
        ),
    )


# --------------------------------------------------------------------------------------
# Sheets
# --------------------------------------------------------------------------------------


def _outline(image: Image.Image, mask: np.ndarray | None, color, scale, origin) -> None:
    if mask is None or not mask.any():
        return
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    draw = ImageDraw.Draw(image)
    for contour in contours:
        points = [((p[0][0] - origin[0]) * scale, (p[0][1] - origin[1]) * scale) for p in contour]
        if len(points) > 1:
            draw.line(points + [points[0]], fill=color, width=2)


def render_before_after_sheet(
    video_path: Path,
    frames: Sequence[int],
    before: dict[int, dict[str, np.ndarray | None]],
    after: dict[int, dict[str, np.ndarray | None]],
    provenance: dict[tuple[int, str], str],
    output_path: Path,
    *,
    crop: tuple[int, int, int, int] = (700, 300, 420, 340),
    scale: int = 2,
    columns: int = 3,
) -> None:
    """Paired tiles: corrected SAM3 (left) vs ensemble with provenance-coloured labels (right)."""
    ox, oy, w, h = crop
    capture = cv2.VideoCapture(str(video_path))
    tiles: list[Image.Image] = []
    try:
        for frame in frames:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f"could not decode frame {frame}")
            crop_rgb = cv2.cvtColor(bgr[oy : oy + h, ox : ox + w], cv2.COLOR_BGR2RGB)
            base = Image.fromarray(crop_rgb).resize((w * scale, h * scale), Image.NEAREST)
            pair = Image.new("RGB", (2 * w * scale + 4, h * scale + 18), (0, 0, 0))
            for column, (label, masks) in enumerate(
                (("before: corrected SAM3", before[frame]), ("after: ensemble", after[frame]))
            ):
                tile = base.copy()
                for part in TARGETS:
                    _outline(tile, masks.get(part), PART_COLORS[part], scale, (ox, oy))
                pair.paste(tile, (column * (w * scale + 4), 18))
            draw = ImageDraw.Draw(pair)
            labels = ", ".join(
                f"{part[:3]}={provenance[(frame, part)].split('_')[0]}"
                for part in TARGETS
                if provenance[(frame, part)] != "sam3_corrected"
            )
            draw.text(
                (4, 3), f"f{frame}  before: corrected SAM3 | after: ensemble", fill=(255, 255, 255)
            )
            if labels:
                draw.text((w * scale + 8, 3), labels, fill=PROVENANCE_COLORS["dam4sam_fallback"])
            tiles.append(pair)
    finally:
        capture.release()
    if not tiles:
        return
    tile_w, tile_h = tiles[0].size
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_w, rows * tile_h), (20, 20, 20))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * tile_w, (index // columns) * tile_h))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


# --------------------------------------------------------------------------------------
# Segmentation episode check (metrics-branch swap detector + area anomalies)
# --------------------------------------------------------------------------------------


def _metrics_module():
    """Imported lazily: review_metrics imports the v4 builder, which imports this module."""
    from . import review_metrics

    return review_metrics


def segmentation_episodes(geometries: Sequence[object], config: object) -> list[dict[str, object]]:
    metrics = _metrics_module()
    medians = metrics.part_area_medians(geometries)
    pairs = metrics.pair_metrics(geometries, medians, config)
    stats = metrics.part_stats(geometries, medians)
    drafts = [*metrics.swap_episodes(pairs, config), *metrics.area_anomaly_episodes(stats, config)]
    ranked = metrics.rank_episodes(drafts)
    return [
        item.model_dump(mode="json")
        for item in ranked
        if item.episode_type in SEGMENTATION_EPISODE_TYPES
    ]


def _episodes_overlapping(episodes: Sequence[dict[str, object]], start: int, end: int):
    return [
        e
        for e in episodes
        if int(e["start_frame"]) <= end and int(e["end_frame"]) >= start  # type: ignore[arg-type]
    ]


# --------------------------------------------------------------------------------------
# Builder
# --------------------------------------------------------------------------------------


def build_ensemble_reference(
    *,
    repository_root: Path,
    policy_path: Path = POLICY_PATH,
    output_root: Path = OUTPUT_ROOT,
    metrics_config_path: Path = METRICS_CONFIG_PATH,
    video: Path = VIDEO_PATH,
    overwrite: bool = False,
    sheet_range: tuple[int, int] = DEFAULT_SHEET_RANGE,
) -> Path:
    repository_root = repository_root.resolve()
    policy_file = (repository_root / policy_path).resolve()
    policy = load_policy(policy_file)
    primary = review._validate_run(
        review.SourceSpec(policy.primary.label, Path(policy.primary.run_directory)),
        repository_root,
        frame_count=FRAME_COUNT,
    )
    fallback = review._validate_run(
        review.SourceSpec(policy.fallback.label, Path(policy.fallback.run_directory)),
        repository_root,
        frame_count=FRAME_COUNT,
    )
    review._validate_shared_sources([primary, fallback])
    dimensions = review.DIMENSIONS
    root = open_output_directory((repository_root / output_root).resolve(), overwrite=overwrite)
    (root / MASK_DIR).mkdir(parents=True, exist_ok=True)
    (root / SHEET_DIR).mkdir(parents=True, exist_ok=True)

    metrics = _metrics_module()
    decider = EnsembleDecider(policy)
    records: list[EnsembleFrameProvenance] = []
    substitutions: list[EnsembleSubstitutionRecord] = []
    declined: list[EnsembleSubstitutionRecord] = []
    observations: list[FrameObservations] = []
    before_geometry: list[object] = []
    after_geometry: list[object] = []
    previous_before: dict[str, np.ndarray | None] | None = None
    previous_after: dict[str, np.ndarray | None] | None = None
    sheet_masks_before: dict[int, dict[str, np.ndarray | None]] = {}
    sheet_masks_after: dict[int, dict[str, np.ndarray | None]] = {}
    for frame in range(FRAME_COUNT):
        primary_obs = primary.observations[frame]
        fallback_obs = fallback.observations[frame]
        primary_masks = {
            part: review._mask_for_part(primary_obs, part, primary.run_directory, dimensions)
            for part in TARGETS
        }
        fallback_masks = {
            part: review._mask_for_part(fallback_obs, part, fallback.run_directory, dimensions)
            for part in TARGETS
        }
        chosen: dict[str, np.ndarray | None] = {}
        objects: list[PerFrameObject] = []
        for part in TARGETS:
            decision = decider.decide(
                frame,
                part,
                primary_masks[part],
                fallback_masks[part],
                {other: primary_masks[other] for other in TARGETS if other != part},
            )
            mask_path = root / MASK_DIR / f"{frame:06d}_{part}.png"
            mask = decision.mask
            if mask is None:
                mask = np.zeros((dimensions[1], dimensions[0]), dtype=bool)
            _write_mask(mask_path, mask)
            record = decision.record.model_copy(update={"mask_uri": f"{MASK_DIR}/{mask_path.name}"})
            records.append(record)
            if decision.substitution is not None:
                (
                    substitutions if decision.substitution.outcome == "substituted" else declined
                ).append(decision.substitution)
            chosen[part] = decision.mask
            if decision.mask is not None:
                source_obs = primary_obs if record.provenance == "sam3_corrected" else fallback_obs
                source_object = _source_object(source_obs, part)
                objects.append(
                    PerFrameObject(
                        object_id=f"ensemble-{part}",
                        label=part,
                        confidence=source_object.confidence if source_object else 1.0,
                        box=_mask_box(decision.mask, dimensions),
                        mask=MaskReference(
                            uri=record.mask_uri or "",
                            storage=MaskStorage.EXTERNAL_ARTIFACT,
                            format="png",
                        ),
                        object_score=source_object.object_score if source_object else None,
                        iou_prediction=source_object.iou_prediction if source_object else None,
                    )
                )
        observations.append(
            FrameObservations(
                view_id=primary_obs.view_id,
                analysis_frame_index=frame,
                source_seconds=primary_obs.source_seconds,
                objects=tuple(objects),
            )
        )
        before_geometry.append(metrics.frame_geometry(frame, primary_masks, previous_before))
        after_geometry.append(metrics.frame_geometry(frame, chosen, previous_after))
        previous_before, previous_after = primary_masks, chosen
        if sheet_range[0] <= frame < sheet_range[1] or any(
            r.provenance != "sam3_corrected" for r in records[-len(TARGETS) :]
        ):
            sheet_masks_before[frame] = primary_masks
            sheet_masks_after[frame] = chosen

    summaries = tuple(summarize_target(part, records, declined) for part in TARGETS)
    sidecar = EnsembleProvenanceSidecar(
        manifest_kind="ensemble_reference_provenance_v1",
        policy=_file_fingerprint(policy_file, repository_root),
        primary_manifest=_file_fingerprint(
            primary.run_directory / "manifest.json", repository_root
        ),
        fallback_manifest=_file_fingerprint(
            fallback.run_directory / "manifest.json", repository_root
        ),
        frame_count=FRAME_COUNT,
        claim_boundaries=policy.claim_boundaries,
        summaries=summaries,
        substitutions=tuple(substitutions),
        declined=tuple(declined),
        frames=tuple(records),
    )
    (root / PROVENANCE_NAME).write_text(sidecar.model_dump_json(indent=1) + "\n", encoding="utf-8")

    observations_path = root / "observations.jsonl"
    with observations_path.open("w", encoding="utf-8") as handle:
        for item in observations:
            handle.write(item.model_dump_json() + "\n")
    primary_metadata = review._metadata(primary.manifest)
    source_fingerprint = getattr(primary_metadata, "source_fingerprint")
    proxy_fingerprint = getattr(primary_metadata, "proxy_fingerprint")
    if not isinstance(source_fingerprint, ArtifactFingerprint) or not isinstance(
        proxy_fingerprint, ArtifactFingerprint
    ):
        raise ValueError("the primary run must declare source and proxy fingerprints")
    run_id = root.name
    manifest = RunManifest(
        run_id=run_id,
        clip=primary.manifest.clip,
        coverage=FullDurationCoverage(
            source_duration_seconds=FRAME_COUNT / ANALYSIS_FPS,
            covered_intervals=(
                TimeInterval(start_seconds=0.0, end_seconds=FRAME_COUNT / ANALYSIS_FPS),
            ),
        ),
        chunk_policy=primary.manifest.chunk_policy,
        method_statuses=(
            MethodStatus(
                method_name="ensemble-reference",
                stage="objects",
                state=MethodState.SUCCEEDED,
                artifact_uri=(output_root / "observations.jsonl").as_posix(),
                measured_on=(
                    "retained corrected SAM3 and DAM4SAM masks over frames [0,1800); no model ran"
                ),
            ),
        ),
        observations=tuple(observations),
        ensemble_reference=EnsembleReferenceRunMetadata(
            policy_fingerprint=_file_fingerprint(policy_file, repository_root),
            source_fingerprint=source_fingerprint,
            proxy_fingerprint=proxy_fingerprint,
            primary_run=_run_reference(primary, policy.primary.label, repository_root),
            fallback_run=_run_reference(fallback, policy.fallback.label, repository_root),
            requested_analysis_frame_range=FrameRange(
                start_frame=0, end_frame_exclusive=FRAME_COUNT
            ),
            view_id=primary.manifest.clip.views[0],
            target_order=TARGETS,  # type: ignore[arg-type]
            observations_uri=(output_root / "observations.jsonl").as_posix(),
            masks_uri=(output_root / MASK_DIR).as_posix(),
            provenance_uri=(output_root / PROVENANCE_NAME).as_posix(),
            provenance_counts={item.target_id: item.provenance_counts for item in summaries},
            claim_boundaries=policy.claim_boundaries,
        ),
    )
    (root / "manifest.json").write_text(manifest.model_dump_json(indent=1) + "\n", encoding="utf-8")

    # Segmentation swap / area-anomaly episodes before and after, with the metrics config.
    metrics_config = metrics.load_config((repository_root / metrics_config_path).resolve())
    before_episodes = segmentation_episodes(before_geometry, metrics_config)
    after_episodes = segmentation_episodes(after_geometry, metrics_config)
    check_window = (1089, 1173)
    episode_check = {
        "schema_version": "1.0",
        "manifest_kind": "ensemble_segmentation_episode_check_v1",
        "claim": (
            "label-free swap/crossing/area-anomaly episodes from the review-metrics detectors, "
            "computed on the corrected SAM3 reference (before) and the ensemble (after); not "
            "accuracy"
        ),
        "metrics_config": metrics.config_fingerprint(
            (repository_root / metrics_config_path).resolve(), repository_root
        ).model_dump(mode="json"),
        "check_window": {"start_frame": check_window[0], "end_frame": check_window[1]},
        "before": {
            "episode_count": len(before_episodes),
            "episodes_in_check_window": _episodes_overlapping(before_episodes, *check_window),
            "episodes": before_episodes,
        },
        "after": {
            "episode_count": len(after_episodes),
            "episodes_in_check_window": _episodes_overlapping(after_episodes, *check_window),
            "episodes": after_episodes,
        },
        "new_after_episodes": [
            e
            for e in after_episodes
            if not any(
                b["episode_type"] == e["episode_type"]
                and b["subjects"] == e["subjects"]
                and int(b["start_frame"]) <= int(e["end_frame"])  # type: ignore[arg-type]
                and int(b["end_frame"]) >= int(e["start_frame"])  # type: ignore[arg-type]
                for b in before_episodes
            )
        ],
    }
    (root / EPISODE_CHECK_NAME).write_text(
        json.dumps(episode_check, indent=1) + "\n", encoding="utf-8"
    )

    # Before/after sheets: the requested window plus every other substituted/hidden frame.
    video_path = (repository_root / video).resolve()
    frames_in_video, fps, video_dimensions = review._video_info(video_path)
    if (frames_in_video, fps, video_dimensions) != (FRAME_COUNT, ANALYSIS_FPS, dimensions):
        raise ValueError(f"sheet video must be the 1800-frame 30-fps bounded proxy: {video_path}")
    provenance_lookup = {(r.analysis_frame_index, r.target_id): r.provenance for r in records}
    window_frames = [
        f for f in range(sheet_range[0], sheet_range[1], SHEET_STRIDE) if f in sheet_masks_before
    ]
    render_before_after_sheet(
        video_path,
        window_frames,
        sheet_masks_before,
        sheet_masks_after,
        provenance_lookup,
        root / SHEET_DIR / f"before_after_{sheet_range[0]}_{sheet_range[1]}.png",
    )
    extra = sorted(f for f in sheet_masks_before if not (sheet_range[0] <= f < sheet_range[1]))
    for start, end in cluster_intervals(extra):
        frames = list(range(start, end, max(1, (end - start) // 15 or 1)))
        render_before_after_sheet(
            video_path,
            frames,
            sheet_masks_before,
            sheet_masks_after,
            provenance_lookup,
            root / SHEET_DIR / f"before_after_{start}_{end}.png",
        )
    return root / "manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_repository_root(parser)
    parser.add_argument("--policy", type=Path, default=POLICY_PATH)
    add_output_root(parser, OUTPUT_ROOT)
    parser.add_argument("--metrics-config", type=Path, default=METRICS_CONFIG_PATH)
    parser.add_argument("--video", type=Path, default=VIDEO_PATH)
    add_output_flags(parser, overwrite_help="Replace an existing run dir.", quiet=False)
    parser.add_argument("--sheet-start", type=int, default=DEFAULT_SHEET_RANGE[0])
    parser.add_argument("--sheet-end", type=int, default=DEFAULT_SHEET_RANGE[1])
    args = parser.parse_args()
    print(
        build_ensemble_reference(
            repository_root=args.repository_root,
            policy_path=args.policy,
            output_root=args.output_root,
            metrics_config_path=args.metrics_config,
            video=args.video,
            overwrite=args.overwrite,
            sheet_range=(args.sheet_start, args.sheet_end),
        )
    )


if __name__ == "__main__":
    main()
