"""Measure how far one camera's video lags the Assembly101 pose clock, per view.

The dataset's 2D landmarks are exact projections of its 3D hand poses, indexed by pose frame.
The videos are 60 fps too, but a camera's first frame need not coincide with pose frame 0
(the C10379 static video has 23 fewer frames than the pose stream and lags it by nine).  This
module ports the Sep 17 ad-hoc offset scans: for candidate offsets it samples image evidence
at the projected fingertips of moving hands and looks for the offset where the evidence peaks.

Three metrics, each evaluated on three 30 s chunks of the focused 60 fps trim:

- `skin_hit`: fraction of fingertip samples inside a YCrCb skin mask (RGB views only);
- `gradient`: Gaussian-smoothed Sobel gradient magnitude at the fingertips (edges of a hand);
- `motion`: absolute frame-to-frame difference at the fingertips (a moving hand).

Samples are velocity-gated and velocity-weighted so resting hands do not dilute the peak.  A
chunk whose curve is flat is reported but not counted; a view whose informative metrics
disagree by more than two frames is marked ambiguous and gets no clock rule.  The result is
a typed record plus an `Assembly101ClockRule`; both are review context, not a claim about
either the dataset or any method.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from pydantic import Field

from .assembly101_fetch_view import raw60_path, video_name
from .assembly101_pose_schemas import Assembly101ClockRule
from .assembly101_recordings import RECORDING_1, Assembly101Recording, get_recording
from .cli_common import add_output_root, add_repository_root
from .schemas import ArtifactFingerprint, VersionedModel, fingerprint

# Recording-1 constants, kept for every caller written before the recording registry existed.
POSES_ROOT = RECORDING_1.poses_root
SHIPPED_2D_WINDOW = RECORDING_1.shipped_2d_window
OUTPUT_ROOT = Path(RECORDING_1.clock_scan_root)
WINDOW_START_POSE_FRAME = RECORDING_1.window_start_raw_frame
WINDOW_FRAME_COUNT = RECORDING_1.window_raw_frame_count
CHUNK_FRAMES = 1800
CHUNK_STARTS: tuple[int, ...] = (0, 1800, 3600)


def chunk_plan(frame_count: int) -> tuple[tuple[int, ...], int]:
    """Three chunks per trim: the Sep 18 30 s chunks when the trim allows, else thirds."""
    if frame_count >= CHUNK_STARTS[-1] + CHUNK_FRAMES:
        return CHUNK_STARTS, CHUNK_FRAMES
    third = frame_count // 3
    if third < 300:
        raise ValueError(f"trim of {frame_count} frames is too short for three offset chunks")
    return (0, third, 2 * third), third


OFFSETS: tuple[int, ...] = tuple(range(-6, 16))
FINGERTIPS: tuple[int, ...] = (0, 1, 2, 3, 4)
CONFIDENCE_FLOOR = 0.7
STATIC_VELOCITY_FLOOR = 5.0
EGO_VELOCITY_FLOOR = 3.0
FLAT_PROMINENCE = 1.2
DISAGREEMENT_FRAMES = 2
OUTLIER_FRAMES = 2
SKIN_LOWER = (0, 135, 85)
SKIN_UPPER = (255, 180, 135)

MetricName = Literal["skin_hit", "gradient", "motion"]


def view_key(view: str) -> str:
    """Pose-file camera key: `C10095:rgb` or `21179183:mono10bit`."""
    if view.startswith("HMC_"):
        return f"{view[4:]}:mono10bit"
    return f"{view}:rgb"


def npz_key(view: str) -> str:
    return view_key(view).replace(":", "_")


def is_ego(view: str) -> bool:
    return view.startswith("HMC_")


class OffsetCurve(VersionedModel):
    """One metric on one chunk: value per candidate offset and where it peaks."""

    metric: MetricName
    chunk_index: int = Field(ge=0)
    chunk_start_trim_frame: int = Field(ge=0)
    chunk_start_pose_frame: int = Field(ge=0)
    offsets: tuple[int, ...] = Field(min_length=2)
    values: tuple[float, ...] = Field(min_length=2)
    sample_counts: tuple[int, ...] = Field(min_length=2)
    peak_offset: int
    peak_offset_subframe: float
    prominence: float
    informative: bool
    near_peak_offsets: tuple[int, ...] = Field(min_length=1)


class Assembly101ClockOffsetScan(VersionedModel):
    """Per-view result of the offset scan, with the rule it supports (if unambiguous)."""

    manifest_kind: Literal["assembly101_clock_offset_scan"]
    recording_id: str = Field(min_length=1)
    view: str = Field(min_length=1)
    view_key: str = Field(min_length=1)
    video: ArtifactFingerprint
    shipped_2d_window: ArtifactFingerprint
    hand_confidences: ArtifactFingerprint
    frames_scanned: int = Field(ge=0)
    curves: tuple[OffsetCurve, ...] = Field(min_length=1)
    metric_medians: dict[str, float]
    chosen_offset_subframe: float | None
    chosen_offset_frames: int | None
    offset_uncertainty_frames: int | None
    ambiguous: bool
    evidence: str = Field(min_length=1)
    clock_rule: Assembly101ClockRule | None
    dropped_metric: str | None = None
    runtime_seconds: float = Field(ge=0)
    time_to_first_output_seconds: float = Field(ge=0)
    claim_boundaries: tuple[str, ...] = Field(min_length=1)


CLAIM_BOUNDARIES: tuple[str, ...] = (
    "The offset is the lag of this camera's video behind the dataset pose clock, measured "
    "from image evidence at the dataset's own projected fingertips; it is review context for "
    "per-frame overlays and cross-view timing, not a statement about any method.",
    "Dataset poses and 2D landmarks are external context (CC BY-NC 4.0); no accuracy claim "
    "rests on them.",
)


class _Sampler:
    """Accumulate velocity-weighted samples per (metric, offset) for one chunk."""

    def __init__(self, metrics: tuple[MetricName, ...]) -> None:
        self.sums = {m: dict.fromkeys(OFFSETS, 0.0) for m in metrics}
        self.weights = {m: dict.fromkeys(OFFSETS, 0.0) for m in metrics}
        self.counts = {m: dict.fromkeys(OFFSETS, 0) for m in metrics}

    def add(self, metric: MetricName, offset: int, value: float, weight: float) -> None:
        self.sums[metric][offset] += value * weight
        self.weights[metric][offset] += weight
        self.counts[metric][offset] += 1

    def curve(self, metric: MetricName) -> tuple[list[float], list[int]]:
        values = [
            self.sums[metric][o] / self.weights[metric][o] if self.weights[metric][o] > 0 else 0.0
            for o in OFFSETS
        ]
        return values, [self.counts[metric][o] for o in OFFSETS]


def _frame_features(frame: np.ndarray, previous_gray: np.ndarray | None, *, rgb: bool):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float32)
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.GaussianBlur(cv2.magnitude(gx, gy), (7, 7), 0)
    motion = None
    if previous_gray is not None:
        motion = cv2.GaussianBlur(cv2.absdiff(gray, previous_gray), (5, 5), 0)
    skin = None
    if rgb:
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        skin = cv2.dilate(cv2.inRange(ycrcb, SKIN_LOWER, SKIN_UPPER), np.ones((5, 5), np.uint8))
    return gray, gradient, motion, skin


def summarize_curve(
    metric: MetricName,
    chunk_index: int,
    chunk_start: int,
    values: list[float],
    counts: list[int],
    *,
    window_start_pose_frame: int = WINDOW_START_POSE_FRAME,
) -> OffsetCurve:
    array = np.asarray(values, dtype=np.float64)
    best = int(np.argmax(array))
    subframe = float(OFFSETS[best])
    if 0 < best < len(array) - 1:
        # Parabola through the peak and its neighbours; the vertex is the sub-frame peak.
        left, centre, right = array[best - 1], array[best], array[best + 1]
        denominator = left - 2 * centre + right
        if denominator < 0:
            subframe += float(0.5 * (left - right) / denominator)
    spread = float(array.std())
    prominence = float((array.max() - np.median(array)) / spread) if spread > 0 else 0.0
    value_range = float(array.max() - array.min())
    near = [OFFSETS[i] for i in range(len(OFFSETS)) if array[i] >= array.max() - 0.1 * value_range]
    # A maximum on the edge of the scanned range is a ramp, not a peak.
    interior = 0 < best < len(array) - 1
    return OffsetCurve(
        metric=metric,
        chunk_index=chunk_index,
        chunk_start_trim_frame=chunk_start,
        chunk_start_pose_frame=window_start_pose_frame + chunk_start,
        offsets=OFFSETS,
        values=tuple(float(v) for v in values),
        sample_counts=tuple(int(c) for c in counts),
        peak_offset=OFFSETS[best],
        peak_offset_subframe=subframe,
        prominence=prominence,
        informative=interior and prominence >= FLAT_PROMINENCE and min(counts) > 0,
        near_peak_offsets=tuple(near),
    )


def scan_video(
    video_path: Path,
    landmarks: np.ndarray,
    confidences: np.ndarray,
    *,
    rgb: bool,
    velocity_floor: float,
    chunk_starts: tuple[int, ...] = CHUNK_STARTS,
    chunk_frames: int = CHUNK_FRAMES,
    window_start_pose_frame: int = WINDOW_START_POSE_FRAME,
) -> tuple[list[OffsetCurve], int]:
    """Evaluate every metric on every chunk; `landmarks` is (F, 2, 21, 2) raw pixels."""
    metrics: tuple[MetricName, ...] = (
        ("skin_hit", "gradient", "motion")
        if rgb
        else (
            "gradient",
            "motion",
        )
    )
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tips = list(FINGERTIPS)
    curves: list[OffsetCurve] = []
    frames_scanned = 0
    for chunk_index, chunk_start in enumerate(chunk_starts):
        capture.set(cv2.CAP_PROP_POS_FRAMES, chunk_start)
        sampler = _Sampler(metrics)
        previous_gray = None
        for local in range(chunk_frames):
            ok, frame = capture.read()
            if not ok:
                break
            frames_scanned += 1
            trim_frame = chunk_start + local
            gray, gradient, motion, skin = _frame_features(frame, previous_gray, rgb=rgb)
            previous_gray = gray
            for offset in OFFSETS:
                row = trim_frame + offset
                if row < 1 or row + 1 >= landmarks.shape[0]:
                    continue
                for hand in (0, 1):
                    if confidences[row, hand] < CONFIDENCE_FLOOR:
                        continue
                    points = landmarks[row, hand][tips]
                    velocity = float(
                        np.linalg.norm(
                            landmarks[row + 1, hand][tips] - landmarks[row - 1, hand][tips], axis=1
                        ).mean()
                        / 2
                    )
                    if not np.isfinite(velocity) or velocity < velocity_floor:
                        continue
                    if not np.all(np.isfinite(points)):
                        continue
                    xs = np.rint(points[:, 0]).astype(int)
                    ys = np.rint(points[:, 1]).astype(int)
                    inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
                    if inside.sum() < 3:
                        continue
                    xs, ys = xs[inside], ys[inside]
                    sampler.add("gradient", offset, float(gradient[ys, xs].mean()), velocity)
                    if motion is not None:
                        sampler.add("motion", offset, float(motion[ys, xs].mean()), velocity)
                    if skin is not None:
                        sampler.add("skin_hit", offset, float((skin[ys, xs] > 0).mean()), velocity)
        for metric in metrics:
            values, counts = sampler.curve(metric)
            curves.append(
                summarize_curve(
                    metric,
                    chunk_index,
                    chunk_start,
                    values,
                    counts,
                    window_start_pose_frame=window_start_pose_frame,
                )
            )
    capture.release()
    return curves, frames_scanned


class OffsetDecision(VersionedModel):
    chosen_subframe: float | None
    chosen: int | None
    uncertainty: int | None
    ambiguous: bool
    metric_medians: dict[str, float]
    evidence: str
    dropped_metric: str | None = None


# When three metrics vote and exactly one sits outside the agreement of the other two, that
# metric is dropped and the rule carries at least this uncertainty (Sep 20, recording 2).
MAJORITY_FALLBACK_MIN_UNCERTAINTY = 2


MAJORITY_TIE_FRAMES = 0.5


def _majority_metrics(medians: dict[str, float]) -> tuple[str, dict[str, float]] | None:
    """The metric whose removal leaves the other two agreeing best, if that choice is clear.

    Needs three voting metrics; the remaining pair must agree within `DISAGREEMENT_FRAMES`,
    and if two different drops both achieve that, the tighter pair wins only when it is
    tighter by more than `MAJORITY_TIE_FRAMES`.
    """
    if len(medians) < 3:
        return None
    candidates: list[tuple[float, str, dict[str, float]]] = []
    for dropped in medians:
        rest = {m: v for m, v in medians.items() if m != dropped}
        spread = max(rest.values()) - min(rest.values())
        if spread <= DISAGREEMENT_FRAMES:
            candidates.append((spread, dropped, rest))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    if len(candidates) > 1 and candidates[1][0] - candidates[0][0] <= MAJORITY_TIE_FRAMES:
        return None
    _, dropped, rest = candidates[0]
    return dropped, rest


def decide_offset(curves: list[OffsetCurve]) -> OffsetDecision:
    """Combine the per-metric medians of informative sub-frame peaks.

    Each metric votes with the median of its informative chunk peaks, so one odd chunk cannot
    move the answer.  The integer offset is the rounded mean of those medians; the uncertainty
    is the largest deviation of a metric median from it, never below one frame.  Metrics whose
    medians differ by more than `DISAGREEMENT_FRAMES` do not get a rule, unless three metrics
    voted and exactly one of them is the odd one out: then the two that agree decide, the
    dropped metric is named, and the uncertainty is at least
    `MAJORITY_FALLBACK_MIN_UNCERTAINTY`.  Chunks farther than `OUTLIER_FRAMES` from the answer
    are named in the evidence.
    """
    per_metric: dict[str, list[float]] = {}
    for curve in curves:
        if curve.informative:
            per_metric.setdefault(curve.metric, []).append(curve.peak_offset_subframe)
    medians = {metric: float(np.median(peaks)) for metric, peaks in per_metric.items()}
    flat = [f"{c.metric}/chunk{c.chunk_index}" for c in curves if not c.informative]
    peaks_text = "; ".join(
        f"{c.metric} chunk {c.chunk_index}: peak {c.peak_offset:+d} "
        f"(sub-frame {c.peak_offset_subframe:+.2f}, prominence {c.prominence:.1f}"
        f"{'' if c.informative else ', flat'})"
        for c in curves
    )
    if not medians:
        return OffsetDecision(
            chosen_subframe=None,
            chosen=None,
            uncertainty=None,
            ambiguous=True,
            metric_medians=medians,
            evidence=f"No informative chunk in any metric. {peaks_text}",
        )
    ordered = sorted(medians.values())
    dropped: str | None = None
    voting = medians
    minimum_uncertainty = 1
    if ordered[-1] - ordered[0] > DISAGREEMENT_FRAMES:
        majority = _majority_metrics(medians)
        if majority is None:
            return OffsetDecision(
                chosen_subframe=None,
                chosen=None,
                uncertainty=None,
                ambiguous=True,
                metric_medians=medians,
                evidence=(
                    f"Metric medians disagree by {ordered[-1] - ordered[0]:.1f} frames "
                    f"({', '.join(f'{m} {v:+.2f}' for m, v in medians.items())}); ambiguous. "
                    f"{peaks_text}"
                ),
            )
        dropped, voting = majority
        minimum_uncertainty = MAJORITY_FALLBACK_MIN_UNCERTAINTY
    mean = float(np.mean(list(voting.values())))
    chosen = int(round(mean))
    uncertainty = max(
        minimum_uncertainty,
        int(np.ceil(max(abs(m - chosen) for m in voting.values()) - 1e-9)),
    )
    informative_count = sum(len(per_metric[m]) for m in voting)
    outliers = [
        f"{c.metric}/chunk{c.chunk_index} ({c.peak_offset_subframe:+.2f})"
        for c in curves
        if c.informative
        and c.metric in voting
        and abs(c.peak_offset_subframe - chosen) > OUTLIER_FRAMES
    ]
    evidence = (
        f"Chosen {chosen:+d} pose frames (+-{uncertainty}; mean of metric medians {mean:+.2f}: "
        + ", ".join(f"{m} {v:+.2f}" for m, v in voting.items())
        + f") from {informative_count} informative chunk peaks over {len(voting)} metrics"
        + (
            f"; the {dropped} metric ({medians[dropped]:+.2f}) disagreed with the other two by "
            f"more than {DISAGREEMENT_FRAMES} frames and was dropped (majority fallback, "
            f"uncertainty floored at {MAJORITY_FALLBACK_MIN_UNCERTAINTY})"
            if dropped is not None
            else ""
        )
        + (f"; flat or edge chunks ignored: {', '.join(flat)}" if flat else "")
        + (f"; outlier chunks: {', '.join(outliers)}" if outliers else "")
        + f". {peaks_text}"
    )
    return OffsetDecision(
        chosen_subframe=mean,
        chosen=chosen,
        uncertainty=uncertainty,
        ambiguous=False,
        metric_medians=medians,
        evidence=evidence,
        dropped_metric=dropped,
    )


def _fingerprint(path: Path, repository_root: Path) -> ArtifactFingerprint:
    if not path.is_file():
        raise FileNotFoundError(f"clock offset input is unavailable: {path}")
    return fingerprint(path, repository_root)


def load_window_confidences(
    confidences_path: Path,
    window_start_pose_frame: int = WINDOW_START_POSE_FRAME,
    window_frame_count: int = WINDOW_FRAME_COUNT,
) -> np.ndarray:
    """(F, 2) dataset hand confidences for the window; a pose frame the file lacks is zero."""
    raw = json.loads(confidences_path.read_text(encoding="utf-8"))
    rows = []
    for k in range(window_start_pose_frame, window_start_pose_frame + window_frame_count):
        entry = raw.get(str(k))
        rows.append([entry["0"], entry["1"]] if entry is not None else [0.0, 0.0])
    return np.array(rows, dtype=np.float64)


def run_scan(
    view: str,
    *,
    repository_root: Path,
    video_path: Path | None = None,
    shipped_2d_window: Path | None = None,
    recording: Assembly101Recording = RECORDING_1,
) -> Assembly101ClockOffsetScan:
    repository_root = repository_root.resolve()
    started = time.monotonic()
    window_start = recording.window_start_raw_frame
    window_count = recording.window_raw_frame_count
    video = repository_root / (video_path or raw60_path(view, recording=recording))
    window = repository_root / (
        shipped_2d_window if shipped_2d_window is not None else recording.shipped_2d_window
    )
    confidences_path = repository_root / recording.poses_member("hand_confidences")
    with np.load(window) as archive:
        key = npz_key(view)
        if key not in archive.files:
            raise KeyError(f"{window} has no array {key}")
        landmarks = np.asarray(archive[key], dtype=np.float64)
        frames = np.asarray(archive["frames"])
    if int(frames[0]) != window_start or landmarks.shape[0] != window_count:
        raise ValueError(
            f"shipped 2D window does not cover pose frames "
            f"[{window_start}, {window_start + window_count})"
        )
    confidences = load_window_confidences(confidences_path, window_start, window_count)
    ego = is_ego(view)
    chunk_starts, chunk_frames = chunk_plan(window_count)
    curves, frames_scanned = scan_video(
        video,
        landmarks,
        confidences,
        rgb=not ego,
        velocity_floor=EGO_VELOCITY_FLOOR if ego else STATIC_VELOCITY_FLOOR,
        chunk_starts=chunk_starts,
        chunk_frames=chunk_frames,
        window_start_pose_frame=window_start,
    )
    first_output = time.monotonic() - started
    decision = decide_offset(curves)
    rule = None
    if decision.chosen is not None and decision.uncertainty is not None:
        rule = Assembly101ClockRule(
            view_key=view_key(view),
            proxy_start_raw_frame=window_start,
            raw_frames_per_proxy_frame=2,
            pose_offset_frames=decision.chosen,
            pose_fps=60,
            analysis_fps=30,
            offset_uncertainty_frames=decision.uncertainty,
            offset_evidence=decision.evidence,
        )
    return Assembly101ClockOffsetScan(
        manifest_kind="assembly101_clock_offset_scan",
        recording_id=recording.recording_id,
        view=view,
        view_key=view_key(view),
        video=_fingerprint(video, repository_root),
        shipped_2d_window=_fingerprint(window, repository_root),
        hand_confidences=_fingerprint(confidences_path, repository_root),
        frames_scanned=frames_scanned,
        curves=tuple(curves),
        metric_medians=decision.metric_medians,
        chosen_offset_subframe=decision.chosen_subframe,
        chosen_offset_frames=decision.chosen,
        offset_uncertainty_frames=decision.uncertainty,
        ambiguous=decision.ambiguous,
        evidence=decision.evidence,
        clock_rule=rule,
        dropped_metric=decision.dropped_metric,
        runtime_seconds=time.monotonic() - started,
        time_to_first_output_seconds=first_output,
        claim_boundaries=CLAIM_BOUNDARIES,
    )


def rescore(scan: Assembly101ClockOffsetScan) -> Assembly101ClockOffsetScan:
    """Re-derive peaks and the decision from stored curve values (the video is not re-read)."""
    window_start = scan.curves[0].chunk_start_pose_frame - scan.curves[0].chunk_start_trim_frame
    curves = [
        summarize_curve(
            curve.metric,
            curve.chunk_index,
            curve.chunk_start_trim_frame,
            list(curve.values),
            list(curve.sample_counts),
            window_start_pose_frame=window_start,
        )
        for curve in scan.curves
    ]
    decision = decide_offset(curves)
    rule = None
    if decision.chosen is not None and decision.uncertainty is not None:
        rule = Assembly101ClockRule(
            view_key=scan.view_key,
            proxy_start_raw_frame=window_start,
            raw_frames_per_proxy_frame=2,
            pose_offset_frames=decision.chosen,
            pose_fps=60,
            analysis_fps=30,
            offset_uncertainty_frames=decision.uncertainty,
            offset_evidence=decision.evidence,
        )
    return scan.model_copy(
        update={
            "curves": tuple(curves),
            "metric_medians": decision.metric_medians,
            "chosen_offset_subframe": decision.chosen_subframe,
            "chosen_offset_frames": decision.chosen,
            "offset_uncertainty_frames": decision.uncertainty,
            "ambiguous": decision.ambiguous,
            "evidence": decision.evidence,
            "clock_rule": rule,
            "dropped_metric": decision.dropped_metric,
        }
    )


def scan_path(view: str, output_root: Path = OUTPUT_ROOT) -> Path:
    return output_root / f"{view}.json"


def load_scan(path: Path) -> Assembly101ClockOffsetScan:
    return Assembly101ClockOffsetScan.model_validate_json(path.read_text(encoding="utf-8"))


class Assembly101ViewClock(VersionedModel):
    """One view's measured clock rule and the scan it came from (tracked summary)."""

    view: str = Field(min_length=1)
    view_key: str = Field(min_length=1)
    ambiguous: bool
    chosen_offset_subframe: float | None
    clock_rule: Assembly101ClockRule | None
    scan: ArtifactFingerprint
    dropped_metric: str | None = None


class Assembly101ClockRuleSet(VersionedModel):
    """Every scanned view's clock rule, checked in so the rig and the review builders agree."""

    manifest_kind: Literal["assembly101_clock_rules"]
    recording_id: str = Field(min_length=1)
    window_start_pose_frame: int = Field(ge=0)
    views: dict[str, Assembly101ViewClock]
    claim_boundaries: tuple[str, ...] = Field(min_length=1)

    def rule(self, view: str) -> Assembly101ClockRule:
        entry = self.views.get(view)
        if entry is None:
            raise KeyError(f"no clock rule was measured for view {view}")
        if entry.clock_rule is None:
            raise ValueError(f"view {view} has an ambiguous clock offset; no rule")
        return entry.clock_rule


CLOCK_RULES_CONFIG = Path(RECORDING_1.clock_rules_path)


def write_clock_rules(
    repository_root: Path,
    *,
    scan_root: Path | None = None,
    output: Path | None = None,
    recording: Assembly101Recording = RECORDING_1,
) -> Path:
    """Collect every scan of one recording into that recording's tracked clock-rule file."""
    repository_root = repository_root.resolve()
    scan_root = scan_root if scan_root is not None else Path(recording.clock_scan_root)
    output = output if output is not None else Path(recording.clock_rules_path)
    views: dict[str, Assembly101ViewClock] = {}
    for path in sorted((repository_root / scan_root).glob("*.json")):
        scan = load_scan(path)
        if scan.recording_id != recording.recording_id:
            raise ValueError(
                f"{path} scans {scan.recording_id}, not {recording.recording_id}; refusing to "
                "mix recordings in one clock-rule file"
            )
        views[scan.view] = Assembly101ViewClock(
            view=scan.view,
            view_key=scan.view_key,
            ambiguous=scan.ambiguous,
            chosen_offset_subframe=scan.chosen_offset_subframe,
            clock_rule=scan.clock_rule,
            scan=_fingerprint(path, repository_root),
            dropped_metric=scan.dropped_metric,
        )
    if not views:
        raise FileNotFoundError(f"no scans under {repository_root / scan_root}")
    rules = Assembly101ClockRuleSet(
        manifest_kind="assembly101_clock_rules",
        recording_id=recording.recording_id,
        window_start_pose_frame=recording.window_start_raw_frame,
        views=views,
        claim_boundaries=CLAIM_BOUNDARIES,
    )
    target = repository_root / output
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(rules.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return target


def load_clock_rules(
    repository_root: Path, path: Path = CLOCK_RULES_CONFIG
) -> Assembly101ClockRuleSet:
    return Assembly101ClockRuleSet.model_validate_json(
        (repository_root / path).read_text(encoding="utf-8")
    )


def load_clock_rules_for(
    recording: Assembly101Recording, repository_root: Path
) -> Assembly101ClockRuleSet:
    """The tracked clock rules of one recording, checked to be about that recording."""
    rules = load_clock_rules(repository_root, Path(recording.clock_rules_path))
    if rules.recording_id != recording.recording_id:
        raise ValueError(
            f"{recording.clock_rules_path} describes {rules.recording_id}, "
            f"not {recording.recording_id}"
        )
    if rules.window_start_pose_frame != recording.window_start_raw_frame:
        raise ValueError(
            f"{recording.clock_rules_path} starts at pose frame {rules.window_start_pose_frame}, "
            f"the registry window at {recording.window_start_raw_frame}"
        )
    return rules


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", action="append", default=[], help="e.g. C10095, HMC_21179183")
    parser.add_argument(
        "--recording",
        help="Registry label or recording id (configs/assembly101/recordings.json); "
        "default: recording 1.",
    )
    parser.add_argument(
        "--write-config",
        action="store_true",
        help="Collect every scan under the output root into the recording's clock-rule file.",
    )
    parser.add_argument(
        "--video", type=Path, help="60 fps trim to scan (default: the focused trim)"
    )
    add_repository_root(parser)
    add_output_root(parser, None, help="default: the recording's clock_scan_root")
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Recompute peaks and the decision from an existing scan without re-reading video.",
    )
    args = parser.parse_args()
    if not args.view and not args.write_config:
        parser.error("pass --view and/or --write-config")
    recording = get_recording(args.recording, args.repository_root)
    output_root = (
        args.output_root if args.output_root is not None else Path(recording.clock_scan_root)
    )
    for view in args.view:
        output = args.repository_root.resolve() / scan_path(view, output_root)
        if args.rescore:
            result = rescore(load_scan(output))
        else:
            result = run_scan(
                view,
                repository_root=args.repository_root,
                video_path=args.video,
                recording=recording,
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
        state = (
            "AMBIGUOUS"
            if result.ambiguous
            else (
                f"{result.chosen_offset_frames:+d} +-{result.offset_uncertainty_frames}"
                f" (sub-frame {result.chosen_offset_subframe:+.2f})"
            )
        )
        print(f"{view} ({video_name(view)}): {state}; {result.runtime_seconds:.0f} s -> {output}")
        for curve in result.curves:
            print(
                f"  {curve.metric:9s} chunk {curve.chunk_index}: peak {curve.peak_offset:+d}"
                f" (sub-frame {curve.peak_offset_subframe:+.2f})"
                f" prominence {curve.prominence:.2f}{'' if curve.informative else ' (flat)'}"
            )
    if args.write_config:
        target = write_clock_rules(args.repository_root, scan_root=output_root, recording=recording)
        rules = Assembly101ClockRuleSet.model_validate_json(target.read_text(encoding="utf-8"))
        print(f"wrote {target} with {len(rules.views)} views")


if __name__ == "__main__":
    main()
