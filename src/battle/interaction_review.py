"""Build the focused, inference-free hand-and-part review package.

The builder reads only already-normalized Battle artifacts.  It intentionally keeps
segmentation, hand pose, body context, worker boxes, and weak temporal supervision in
separate Rerun roots rather than turning them into one implied joint tracker.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from PIL import Image, ImageDraw

from .exploratory_comparison import (
    METHOD_COLORS,
    _drop_dtw_text,
    _file_fingerprint,
    _log_hands,
    _log_nlf_body,
    validate_artifact_fingerprint,
)
from .exporter import _rgba_mask_png
from .fine_substep_contract import load_contract as load_fine_substep_contract
from .fine_substep_contract import substep_for_frame
from .four_part_contract import ANALYSIS_FPS, FRAME_COUNT, TARGETS, load_contract
from .schemas import (
    ArtifactFingerprint,
    ClockName,
    FrameObservations,
    InteractionContactDiagnostic,
    InteractionContactEvent,
    InteractionHandDisagreement,
    InteractionReviewIndexManifest,
    InteractionReviewPinnedMoment,
    RunManifest,
    SegmentationReviewEpisode,
    SegmentationReviewTrigger,
    TimeInterval,
)

OUTPUT_ROOT = Path("runs/interaction-review-first-20s")
OUTPUT_NAME = "interaction_review.rrd"
INDEX_NAME = "interaction_review_index.json"
GUIDE_NAME = "review_guide.md"
CONTACT_SHEET_NAME = "pinned_moments_contact_sheet.png"
COMPARISON_ID = "interaction_review_first_20s"
DIMENSIONS = (1280, 720)
FIRST_MINUTE_FRAME_COUNT = 1800
CONTACT_THRESHOLD_PIXELS = 12.0
CONTACT_START_FRAMES = 2
CONTACT_END_FRAMES = 3
TIP_INDICES = (4, 8, 12, 16, 20)


@dataclass(frozen=True)
class SourceSpec:
    method_id: str
    run_directory: Path


@dataclass(frozen=True)
class LoadedSource:
    spec: SourceSpec
    run_directory: Path
    manifest: RunManifest
    observations: dict[int, FrameObservations]
    artifacts: tuple[ArtifactFingerprint, ...]


DEFAULT_SOURCES = {
    "mediapipe": SourceSpec(
        "mediapipe", Path("runs/mediapipe-hands-static-20s-fused-dedup-th035-20260916t0428z")
    ),
    "wilor": SourceSpec("wilor", Path("runs/wilor-hands-static-20s-audited-source-state")),
    # v5 rebuild of the 20 s layer with the same bounded low-confidence lane-continuation gate
    # as the first-minute layer; `runs/wilor-hands-stabilized-20s-overnight-v2-r3` is the
    # pre-gate layer the earlier v2/v3 packages were built from.
    "stabilized_wilor": SourceSpec("stabilized_wilor", Path("runs/wilor-hands-stabilized-20s-v5")),
    "boxmot": SourceSpec("boxmot", Path("runs/boxmot-yolo-static-20s-20260916t0445z")),
    "kineo": SourceSpec("kineo", Path("runs/kineo-nlf-fused-20s-overnight-v3")),
    "drop_dtw": SourceSpec("drop_dtw", Path("runs/drop-dtw-static-20s-pinned-openclip-rerun")),
}
FIRST_MINUTE_SOURCES = {
    "mediapipe": SourceSpec(
        "mediapipe", Path("runs/mediapipe-hands-static-60s-fused-dedup-th035-20260916t0430z")
    ),
    "wilor": SourceSpec("wilor", Path("runs/wilor-hands-static-60s-overnight-v2")),
    # v5 adds the bounded low-confidence lane-continuation gate for finger-only hands.
    "stabilized_wilor": SourceSpec("stabilized_wilor", Path("runs/wilor-hands-stabilized-60s-v5")),
    "boxmot": SourceSpec("boxmot", Path("runs/boxmot-yolo-static-60s-v4-postreboot")),
    "kineo": SourceSpec("kineo", Path("runs/kineo-nlf-fused-60s-v4")),
}
REFERENCE_SEGMENTATIONS = {
    "reviewed_seed_sam2_control": Path(
        "runs/reviewed-seed-sam2-control-four-part-20s-20260916t1005z"
    ),
    "baseline_sam3": Path(
        "runs/muggledsam-sam3-four-part-static-focused-reassembly-static-c10379-20260916t023700z"
    ),
}


def _metadata(manifest: RunManifest) -> object | None:
    for name in (
        "mediapipe_hands",
        "wilor_hands",
        "boxmot",
        "drop_dtw",
        "four_part_segmentation",
        "four_part_focused",
        "external_partial",
    ):
        if (value := getattr(manifest, name)) is not None:
            return value
    return None


def _validate_run(
    spec: SourceSpec,
    repository_root: Path,
    *,
    frame_count: int = FRAME_COUNT,
    require_every_frame: bool = True,
) -> LoadedSource:
    """Validate source/proxy fingerprints and exact source-time mapping before use."""
    run_directory = (repository_root / spec.run_directory).resolve()
    manifest_path = run_directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"review input manifest is unavailable: {manifest_path}")
    manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    if manifest.clip.views != ("static-c10379",):
        raise ValueError(f"{spec.method_id} must use only the approved static RGB view")
    if manifest.clip.timing.clocks.fps_for(ClockName.ANALYSIS) != ANALYSIS_FPS:
        raise ValueError(f"{spec.method_id} must use the 30-fps analysis clock")
    metadata = _metadata(manifest)
    if metadata is None:
        raise ValueError(f"{spec.method_id} does not have recognized method metadata")
    artifacts = [_file_fingerprint(manifest_path, repository_root)]
    source = getattr(metadata, "source_fingerprint", None)
    proxy = getattr(metadata, "proxy_fingerprint", None)
    for label, fingerprint in (("source", source), ("proxy", proxy)):
        if fingerprint is None:
            continue
        if not isinstance(fingerprint, ArtifactFingerprint):
            raise ValueError(f"{spec.method_id} has invalid {label} fingerprint")
        validate_artifact_fingerprint(
            fingerprint, repository_root, label=f"{spec.method_id} {label}"
        )
        artifacts.append(fingerprint)
    if source is None or proxy is None:
        requested = getattr(metadata, "requested_input_fingerprint", None)
        if not isinstance(requested, ArtifactFingerprint):
            raise ValueError(
                f"{spec.method_id} must declare source/proxy or a fingerprinted bounded input"
            )
        validate_artifact_fingerprint(
            requested, repository_root, label=f"{spec.method_id} bounded input"
        )
        artifacts.append(requested)
    for attribute in ("config_fingerprint", "requested_input_fingerprint"):
        if isinstance((fingerprint := getattr(metadata, attribute, None)), ArtifactFingerprint):
            if not fingerprint.uri.startswith("hf://"):
                validate_artifact_fingerprint(
                    fingerprint, repository_root, label=f"{spec.method_id} declared input"
                )
            artifacts.append(fingerprint)
    observations: dict[int, FrameObservations] = {}
    for observation in manifest.observations:
        index = observation.analysis_frame_index
        if index >= frame_count:
            continue
        if index in observations:
            raise ValueError(f"{spec.method_id} has a duplicate frame {index}")
        observations[index] = observation
    validate_review_observations(manifest, observations)
    if require_every_frame and set(observations) != set(range(frame_count)):
        raise ValueError(
            f"{spec.method_id} must retain one normalized row for every frame [0,{frame_count})"
        )
    observations_path = run_directory / "observations.jsonl"
    if observations_path.is_file():
        artifacts.append(_file_fingerprint(observations_path, repository_root))
    return LoadedSource(spec, run_directory, manifest, observations, tuple(artifacts))


def validate_review_observations(
    manifest: RunManifest, observations: dict[int, FrameObservations]
) -> None:
    """Reject retained rows whose source timestamp is not exact for the declared clock."""
    for index, observation in observations.items():
        expected = manifest.clip.timing.source_seconds_for_frame(ClockName.ANALYSIS, index)
        if abs(observation.source_seconds - expected) > 1e-6:
            raise ValueError(f"source timestamp mismatch at frame {index}")


def _validate_shared_sources(sources: list[LoadedSource]) -> None:
    first = sources[0].manifest
    raw = next(
        (
            getattr(_metadata(source.manifest), "source_fingerprint")
            for source in sources
            if getattr(_metadata(source.manifest), "source_fingerprint", None) is not None
        ),
        None,
    )
    if not isinstance(raw, ArtifactFingerprint):
        raise ValueError("at least one review input must declare the raw-source fingerprint")
    timing = first.clip.timing.model_dump(mode="json")
    for source in sources:
        if source.manifest.clip.clip_id != first.clip.clip_id:
            raise ValueError("review inputs must name one clip")
        if source.manifest.clip.timing.model_dump(mode="json") != timing:
            raise ValueError("review inputs must share the exact source clock")
        candidate = getattr(_metadata(source.manifest), "source_fingerprint", None)
        if candidate is not None and candidate != raw:
            raise ValueError("review inputs must share one source-video fingerprint")


def _video_info(video_path: Path) -> tuple[int, int, tuple[int, int]]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(completed.stdout)["streams"][0]
    numerator, denominator = stream["avg_frame_rate"].split("/")
    return (
        int(stream["nb_read_frames"]),
        round(int(numerator) / int(denominator)),
        (int(stream["width"]), int(stream["height"])),
    )


def _mask_for_part(
    observation: FrameObservations, part: str, run_directory: Path, dimensions: tuple[int, int]
) -> np.ndarray | None:
    item = next((item for item in observation.objects if item.label == part and item.mask), None)
    if item is None or item.mask is None:
        return None
    path = (run_directory / item.mask.uri).resolve()
    if not path.is_relative_to(run_directory) or not path.is_file():
        raise FileNotFoundError(f"reference mask is unavailable: {item.mask.uri}")
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L"), dtype=np.uint8) > 0
    if mask.shape != (dimensions[1], dimensions[0]):
        raise ValueError(f"reference mask does not match the source pixels: {path}")
    return mask


def _pixel(point: object, dimensions: tuple[int, int]) -> tuple[int, int]:
    width, height = dimensions
    return (
        min(width - 1, max(0, round(getattr(point, "x") * (width - 1)))),
        min(height - 1, max(0, round(getattr(point, "y") * (height - 1)))),
    )


def _distance_map(mask: np.ndarray) -> np.ndarray:
    """Distance in source pixels to the mask, with zero exactly for interior pixels."""
    if not mask.any():
        raise ValueError("reference part mask is empty")
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def contact_measurement(
    hand: object, mask: np.ndarray, dimensions: tuple[int, int]
) -> tuple[float, float, float, bool]:
    """Return palm, nearest-fingertip, aggregate distance, and exact inside status."""
    distance = _distance_map(mask)
    palm_x, palm_y = _pixel(getattr(hand, "landmarks")[0], dimensions)
    palm = float(distance[palm_y, palm_x])
    tips = [
        float(distance[y, x])
        for x, y in (_pixel(getattr(hand, "landmarks")[index], dimensions) for index in TIP_INDICES)
    ]
    fingertip = min(tips)
    minimum = min(palm, fingertip)
    return palm, fingertip, minimum, minimum == 0.0


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 0.0


def segmentation_review_triggers(
    reference: LoadedSource,
    control: LoadedSource | None,
    hands: LoadedSource,
    dimensions: tuple[int, int],
    *,
    frame_count: int = FRAME_COUNT,
) -> tuple[SegmentationReviewTrigger, ...]:
    """Emit auditable geometry triggers without deciding target semantics."""

    triggers: list[SegmentationReviewTrigger] = []
    previous: dict[str, np.ndarray] = {}
    for frame in range(frame_count):
        for part in TARGETS:
            current = _mask_for_part(
                reference.observations[frame], part, reference.run_directory, dimensions
            )
            other = (
                _mask_for_part(control.observations[frame], part, control.run_directory, dimensions)
                if control is not None
                else None
            )
            if current is None:
                continue
            prior = previous.get(part)
            if prior is not None:
                temporal_iou = _mask_iou(prior, current)
                if temporal_iou < 0.5:
                    triggers.append(
                        SegmentationReviewTrigger(
                            analysis_frame_index=frame,
                            target_id=part,
                            trigger_type="temporal_iou_lt_0_5",
                            value=temporal_iou,
                        )
                    )
                ratio = max(prior.sum(), current.sum()) / max(1, min(prior.sum(), current.sum()))
                if ratio > 2:
                    triggers.append(
                        SegmentationReviewTrigger(
                            analysis_frame_index=frame,
                            target_id=part,
                            trigger_type="area_ratio_gt_2",
                            value=float(ratio),
                        )
                    )
                old_y, old_x = np.nonzero(prior)
                new_y, new_x = np.nonzero(current)
                jump = float(np.hypot(new_x.mean() - old_x.mean(), new_y.mean() - old_y.mean()))
                if jump > 25:
                    triggers.append(
                        SegmentationReviewTrigger(
                            analysis_frame_index=frame,
                            target_id=part,
                            trigger_type="centroid_jump_gt_25px",
                            value=jump,
                        )
                    )
            if other is not None and hands.observations[frame].hands:
                cross_iou = _mask_iou(current, other)
                if cross_iou < 0.5:
                    triggers.append(
                        SegmentationReviewTrigger(
                            analysis_frame_index=frame,
                            target_id=part,
                            trigger_type="cross_method_iou_lt_0_5_during_hand_presence",
                            value=cross_iou,
                        )
                    )
            previous[part] = current
    return tuple(triggers)


def cluster_segmentation_triggers(
    triggers: tuple[SegmentationReviewTrigger, ...], max_frame_gap: int = 6
) -> tuple[SegmentationReviewEpisode, ...]:
    """Compact nearby same-part geometry triggers without discarding raw trigger rows."""

    if max_frame_gap < 0:
        raise ValueError("max_frame_gap must be non-negative")
    episodes: list[SegmentationReviewEpisode] = []
    by_target: dict[str, list[SegmentationReviewTrigger]] = defaultdict(list)
    for trigger in triggers:
        by_target[trigger.target_id].append(trigger)
    for target in TARGETS:
        group: list[SegmentationReviewTrigger] = []
        for trigger in sorted(
            by_target[target], key=lambda item: (item.analysis_frame_index, item.trigger_type)
        ):
            starts_new_episode = group and (
                trigger.analysis_frame_index > group[-1].analysis_frame_index + max_frame_gap
            )
            if starts_new_episode:
                episodes.append(
                    SegmentationReviewEpisode(
                        target_id=target,
                        start_frame=group[0].analysis_frame_index,
                        end_frame=group[-1].analysis_frame_index,
                        trigger_count=len(group),
                        trigger_types=tuple(sorted({item.trigger_type for item in group})),
                    )
                )
                group = []
            group.append(trigger)
        if group:
            episodes.append(
                SegmentationReviewEpisode(
                    target_id=target,
                    start_frame=group[0].analysis_frame_index,
                    end_frame=group[-1].analysis_frame_index,
                    trigger_count=len(group),
                    trigger_types=tuple(sorted({item.trigger_type for item in group})),
                )
            )
    return tuple(sorted(episodes, key=lambda item: (item.start_frame, item.target_id)))


def _spatial_lanes(
    observations: dict[int, FrameObservations],
    dimensions: tuple[int, int],
    *,
    frame_count: int = FRAME_COUNT,
) -> dict[tuple[int, int], str]:
    """Assign short-lived proximity lanes by adjacent-frame nearest wrist, never as identity."""
    result: dict[tuple[int, int], str] = {}
    active: dict[str, tuple[int, int]] = {}
    next_lane = 1
    for frame in range(frame_count):
        hands = observations[frame].hands
        available = dict(active)
        assignments: dict[int, str] = {}
        for index, hand in enumerate(hands):
            wrist = np.asarray(_pixel(hand.landmarks[0], dimensions), dtype=float)
            candidates = sorted(
                (
                    (float(np.linalg.norm(wrist - np.asarray(previous, dtype=float))), lane)
                    for lane, previous in available.items()
                ),
                key=lambda item: (item[0], item[1]),
            )
            if candidates:
                lane = candidates[0][1]
                del available[lane]
            else:
                lane = f"spatial-lane-{next_lane}"
                next_lane += 1
            assignments[index] = lane
        active.update(
            {
                assignments[index]: _pixel(hand.landmarks[0], dimensions)
                for index, hand in enumerate(hands)
            }
        )
        result.update({(frame, index): lane for index, lane in assignments.items()})
    return result


def debounce_contact(raw: list[bool | None]) -> list[bool | None]:
    """Debounce a raw candidate stream and clear state immediately on missing evidence."""
    output: list[bool | None] = []
    state = False
    true_run = false_run = 0
    for value in raw:
        if value is None:
            state = False
            true_run = false_run = 0
            output.append(None)
            continue
        if value:
            true_run += 1
            false_run = 0
            if true_run >= CONTACT_START_FRAMES:
                state = True
        else:
            false_run += 1
            true_run = 0
            if false_run >= CONTACT_END_FRAMES:
                state = False
        output.append(state)
    return output


def nearest_wrist_matches(
    mediapipe_hands: tuple[object, ...],
    wilor_hands: tuple[object, ...],
    dimensions: tuple[int, int],
) -> list[tuple[int | None, int | None]]:
    """Greedily pair only same-frame wrist locations; no cross-frame identity is inferred."""
    pairs: list[tuple[int | None, int | None]] = []
    remaining = set(range(len(wilor_hands)))
    for mp_index, hand in enumerate(mediapipe_hands):
        wrist = np.asarray(_pixel(getattr(hand, "landmarks")[0], dimensions), dtype=float)
        candidates = sorted(
            (
                (
                    float(
                        np.linalg.norm(
                            wrist
                            - np.asarray(
                                _pixel(getattr(wilor_hands[index], "landmarks")[0], dimensions),
                                dtype=float,
                            )
                        )
                    ),
                    index,
                )
                for index in remaining
            ),
            key=lambda item: (item[0], item[1]),
        )
        if candidates:
            _, wilor_index = candidates[0]
            remaining.remove(wilor_index)
            pairs.append((mp_index, wilor_index))
        else:
            pairs.append((mp_index, None))
    pairs.extend((None, index) for index in sorted(remaining))
    return pairs


def _disagreements(
    mediapipe: LoadedSource,
    wilor: LoadedSource,
    dimensions: tuple[int, int],
    *,
    frame_count: int = FRAME_COUNT,
) -> tuple[InteractionHandDisagreement, ...]:
    records: list[InteractionHandDisagreement] = []
    scale = np.asarray(dimensions, dtype=float)
    for frame in range(frame_count):
        mp_hands = mediapipe.observations[frame].hands
        wi_hands = wilor.observations[frame].hands
        for mp_index, wi_index in nearest_wrist_matches(mp_hands, wi_hands, dimensions):
            if mp_index is None:
                records.append(
                    InteractionHandDisagreement(
                        analysis_frame_index=frame,
                        assignment_state="wilor_only",
                        wilor_hand_id=wi_hands[wi_index].hand_id if wi_index is not None else None,
                    )
                )
            elif wi_index is None:
                records.append(
                    InteractionHandDisagreement(
                        analysis_frame_index=frame,
                        assignment_state="mediapipe_only",
                        mediapipe_hand_id=mp_hands[mp_index].hand_id,
                    )
                )
            else:
                mp = np.asarray([[point.x, point.y] for point in mp_hands[mp_index].landmarks])
                wi = np.asarray([[point.x, point.y] for point in wi_hands[wi_index].landmarks])
                distances = np.linalg.norm((mp - wi) * scale, axis=1)
                records.append(
                    InteractionHandDisagreement(
                        analysis_frame_index=frame,
                        assignment_state="matched",
                        mediapipe_hand_id=mp_hands[mp_index].hand_id,
                        wilor_hand_id=wi_hands[wi_index].hand_id,
                        mean_landmark_distance_pixels=float(distances.mean()),
                        max_landmark_distance_pixels=float(distances.max()),
                        handedness_disagrees=mp_hands[mp_index].side != wi_hands[wi_index].side,
                    )
                )
    return tuple(records)


def contact_eligible_frame(frame: int, intervals: tuple[tuple[int, int], ...]) -> bool:
    """True when `frame` lies inside any half-open contact-eligible interval."""
    return any(start <= frame < end for start, end in intervals)


def _contacts(
    mediapipe: LoadedSource,
    segmentation: LoadedSource,
    dimensions: tuple[int, int],
    *,
    frame_count: int = FRAME_COUNT,
    segmentation_contact_eligible_through: int | None = None,
    segmentation_contact_eligible_intervals: tuple[tuple[int, int], ...] | None = None,
) -> tuple[tuple[InteractionContactDiagnostic, ...], tuple[InteractionContactEvent, ...]]:
    """Measure hand-to-part geometry on contact-eligible frames only.

    Eligibility is either everything before `segmentation_contact_eligible_through` or the
    union of half-open `segmentation_contact_eligible_intervals`; every other frame yields
    explicit `invalid_mask` rows so no stale candidate can leak across an ineligible gap.
    """
    if segmentation_contact_eligible_intervals is None:
        segmentation_contact_eligible_intervals = (
            ((0, segmentation_contact_eligible_through),)
            if segmentation_contact_eligible_through is not None
            else ((0, frame_count),)
        )
    elif segmentation_contact_eligible_through is not None:
        raise ValueError("pass either a contact-eligible cutoff or explicit intervals, not both")

    def eligible(frame: int) -> bool:
        return contact_eligible_frame(frame, segmentation_contact_eligible_intervals)

    lanes = _spatial_lanes(mediapipe.observations, dimensions, frame_count=frame_count)
    lanes_seen = sorted(set(lanes.values()))
    raw: dict[tuple[str, str], list[bool | None]] = defaultdict(lambda: [None] * frame_count)
    draft: list[InteractionContactDiagnostic] = []
    for frame in range(frame_count):
        if not eligible(frame):
            for lane in lanes_seen:
                for part in TARGETS:
                    draft.append(
                        InteractionContactDiagnostic(
                            analysis_frame_index=frame,
                            hand_source_id=lane,
                            part_id=part,
                            observation_state="invalid_mask",
                        )
                    )
            continue
        masks = {
            part: _mask_for_part(
                segmentation.observations[frame], part, segmentation.run_directory, dimensions
            )
            for part in TARGETS
        }
        for hand_index, hand in enumerate(mediapipe.observations[frame].hands):
            lane = lanes[(frame, hand_index)]
            for part, mask in masks.items():
                key = (lane, part)
                if mask is None:
                    draft.append(
                        InteractionContactDiagnostic(
                            analysis_frame_index=frame,
                            hand_source_id=lane,
                            part_id=part,
                            observation_state="missing_mask",
                        )
                    )
                    continue
                palm, fingertip, minimum, inside = contact_measurement(hand, mask, dimensions)
                candidate = inside or minimum <= CONTACT_THRESHOLD_PIXELS
                raw[key][frame] = candidate
                draft.append(
                    InteractionContactDiagnostic(
                        analysis_frame_index=frame,
                        hand_source_id=lane,
                        part_id=part,
                        observation_state="observed",
                        palm_distance_pixels=palm,
                        fingertip_distance_pixels=fingertip,
                        minimum_distance_pixels=minimum,
                        inside_mask=inside,
                        raw_contact_candidate=candidate,
                        debounced_contact_candidate=False,
                    )
                )
    # Absence is separately represented for every lane that exists anywhere, never carried forward.
    for frame in range(frame_count):
        if not eligible(frame):
            continue
        present = {
            lanes[(frame, index)] for index in range(len(mediapipe.observations[frame].hands))
        }
        for lane in lanes_seen:
            if lane not in present:
                for part in TARGETS:
                    draft.append(
                        InteractionContactDiagnostic(
                            analysis_frame_index=frame,
                            hand_source_id=lane,
                            part_id=part,
                            observation_state="missing_hand",
                        )
                    )
    debounced = {key: debounce_contact(values) for key, values in raw.items()}
    records: list[InteractionContactDiagnostic] = []
    events: list[InteractionContactEvent] = []
    prior: dict[tuple[str, str], bool] = defaultdict(bool)
    for item in sorted(
        draft, key=lambda item: (item.analysis_frame_index, item.hand_source_id, item.part_id)
    ):
        key = (item.hand_source_id, item.part_id)
        if item.observation_state != "observed":
            prior[key] = False
            records.append(item)
            continue
        value = debounced[key][item.analysis_frame_index]
        records.append(item.model_copy(update={"debounced_contact_candidate": value}))
        if value != prior[key]:
            events.append(
                InteractionContactEvent(
                    analysis_frame_index=item.analysis_frame_index,
                    hand_source_id=item.hand_source_id,
                    part_id=item.part_id,
                    event_type="contact_candidate_start" if value else "contact_candidate_end",
                )
            )
        prior[key] = bool(value)
    return tuple(records), tuple(events)


def deterministic_pinned_moments(
    disagreements: tuple[InteractionHandDisagreement, ...],
    contacts: tuple[InteractionContactDiagnostic, ...],
    events: tuple[InteractionContactEvent, ...],
    kineo: LoadedSource,
    segmentation_triggers: tuple[SegmentationReviewTrigger, ...] = (),
    segmentation_episodes: tuple[SegmentationReviewEpisode, ...] = (),
    *,
    frame_count: int = FRAME_COUNT,
    source_start_seconds: float = 294.0,
) -> tuple[InteractionReviewPinnedMoment, ...]:
    """Select stable review bookmarks, with frame number as every tie-breaker."""
    categories: dict[int, list[str]] = {
        0: ["required_frame_0"],
        frame_count // 2: [f"required_frame_{frame_count // 2}"],
        frame_count - 1: [f"required_frame_{frame_count - 1}"],
    }
    rationale: dict[int, list[str]] = {
        0: ["Required boundary frame."],
        frame_count // 2: ["Required midpoint frame."],
        frame_count - 1: ["Required final frame."],
    }
    requested = [
        (88, "hand_transition_review", "Requested hand-transition review range begins."),
        (194, "segmentation_reference_review", "Requested early part-reference review range."),
        (224, "segmentation_reference_review", "Requested part-reference review range."),
        (250, "interaction_review", "Requested insertion/hand review range."),
        (380, "action_transition_review", "Requested pre-tool action review range."),
        (413, "action_transition_review", "Requested tool-onset action review range."),
        (500, "action_transition_review", "Requested fastening action review range."),
        (548, "segmentation_correction_review", "Requested correction-candidate review range."),
        (597, "late_occlusion_review", "Requested final occlusion review range."),
    ]
    if frame_count == FIRST_MINUTE_FRAME_COUNT:
        requested.extend(
            (
                (370, "segmentation_reference_review", "Human-noted transient interior growth."),
                (900, "mid_minute_review", "First-minute midpoint review."),
                (
                    1020,
                    "segmentation_swap_review",
                    "Measured onset of the human-reported chassis/interior swap.",
                ),
                (
                    1172,
                    "segmentation_correction_review",
                    "Agent-selected chassis/interior correction keyframe.",
                ),
                (1200, "late_segmentation_review", "Known later segmentation degradation review."),
                (1584, "late_hand_review", "Requested late WiLoR audit interval begins."),
                (1637, "late_hand_review", "Requested late WiLoR audit interval ends."),
            )
        )
    for frame, category, note in requested:
        if frame >= frame_count:
            continue
        categories.setdefault(frame, []).append(category)
        rationale.setdefault(frame, []).append(note)
    matched = [item for item in disagreements if item.assignment_state == "matched"]
    if matched:
        selected = min(
            matched,
            key=lambda item: (
                -float(item.mean_landmark_distance_pixels or 0),
                item.analysis_frame_index,
            ),
        )
        categories.setdefault(selected.analysis_frame_index, []).append("high_hand_disagreement")
        rationale.setdefault(selected.analysis_frame_index, []).append(
            "Highest matched 21-landmark mean pixel disagreement; frame breaks ties."
        )
    missing = [
        item for item in disagreements if item.assignment_state in ("mediapipe_only", "wilor_only")
    ]
    if missing:
        selected = min(missing, key=lambda item: item.analysis_frame_index)
        categories.setdefault(selected.analysis_frame_index, []).append("missing_hand")
        rationale.setdefault(selected.analysis_frame_index, []).append(
            "First same-frame one-method-only hand detection."
        )
    if events:
        selected = min(events, key=lambda item: item.analysis_frame_index)
        categories.setdefault(selected.analysis_frame_index, []).append("contact_transition")
        rationale.setdefault(selected.analysis_frame_index, []).append(
            "First debounced geometry-only contact-candidate transition."
        )
    else:
        nearest = [
            item
            for item in contacts
            if item.observation_state == "observed" and item.minimum_distance_pixels is not None
        ]
        if nearest:
            selected = min(
                nearest,
                key=lambda item: (item.minimum_distance_pixels or 0, item.analysis_frame_index),
            )
            categories.setdefault(selected.analysis_frame_index, []).append(
                "closest_contact_candidate"
            )
            rationale.setdefault(selected.analysis_frame_index, []).append(
                "No debounced transition; nearest observed hand-to-part geometry."
            )
    gaps = [frame for frame in range(frame_count) if not kineo.observations[frame].nlf_body_2d]
    if gaps:
        frame = gaps[0]
        categories.setdefault(frame, []).append("kineo_gap")
        rationale.setdefault(frame, []).append("First Kineo NLF body-output gap.")
    late_missing = [
        item.analysis_frame_index
        for item in disagreements
        if item.analysis_frame_index >= frame_count * 3 // 4 and item.assignment_state != "matched"
    ]
    if late_missing:
        frame = max(late_missing)
        categories.setdefault(frame, []).append("late_occlusion_or_missing")
        rationale.setdefault(frame, []).append(
            "Latest second-half one-method-only hand detection; inspect for occlusion."
        )
    if segmentation_triggers:
        selected = min(
            segmentation_triggers,
            key=lambda item: (item.analysis_frame_index, item.target_id, item.trigger_type),
        )
        categories.setdefault(selected.analysis_frame_index, []).append(
            "segmentation_review_trigger"
        )
        rationale.setdefault(selected.analysis_frame_index, []).append(
            f"First automatic {selected.trigger_type} trigger for {selected.target_id}."
        )
    for episode in segmentation_episodes:
        categories.setdefault(episode.start_frame, []).append("segmentation_review_episode")
        rationale.setdefault(episode.start_frame, []).append(
            f"{episode.target_id} trigger episode f{episode.start_frame}–f{episode.end_frame} "
            f"({episode.trigger_count} raw triggers; {', '.join(episode.trigger_types)})."
        )
    return tuple(
        InteractionReviewPinnedMoment(
            analysis_frame_index=frame,
            source_seconds=source_start_seconds + frame / ANALYSIS_FPS,
            categories=tuple(categories[frame]),
            rationale=" ".join(rationale[frame]),
        )
        for frame in sorted(categories)
    )


def _log_reference_masks(
    root: str, observation: FrameObservations, directory: Path, dimensions: tuple[int, int]
) -> None:
    masks_root = f"{root}/primary/reference_four_part_segmentation"
    rr.log(masks_root, rr.Clear(recursive=True))
    colors = {
        "chassis": (70, 130, 255),
        "interior": (60, 210, 150),
        "rear_body": (255, 190, 45),
        "cabin": (255, 95, 100),
    }
    for part in TARGETS:
        mask = _mask_for_part(observation, part, directory, dimensions)
        if mask is not None:
            rr.log(
                f"{masks_root}/{part}",
                rr.EncodedImage(
                    contents=_rgba_mask_png(mask, colors[part]),
                    media_type="image/png",
                    opacity=0.35,
                    draw_order=1.0,
                ),
            )


def _log_context_boxes(
    root: str, context: str, observation: FrameObservations, dimensions: tuple[int, int]
) -> None:
    path = f"{root}/contexts/{context}"
    if not observation.objects:
        rr.log(path, rr.Clear(recursive=True))
        return
    width, height = dimensions
    rr.log(
        f"{path}/person_boxes",
        rr.Boxes2D(
            mins=[[item.box.x * width, item.box.y * height] for item in observation.objects],
            sizes=[
                [item.box.width * width, item.box.height * height] for item in observation.objects
            ],
            labels=[f"{context}: {item.object_id}" for item in observation.objects],
            colors=[METHOD_COLORS["boxmot"]] * len(observation.objects),
        ),
    )


def _log_scalar_or_clear(path: str, value: float | None) -> None:
    if value is None:
        rr.log(path, rr.Clear(recursive=True))
    else:
        rr.log(path, rr.Scalars([value]))


def _log_diagnostics_frame(
    root: str,
    frame: int,
    contacts: tuple[InteractionContactDiagnostic, ...],
    disagreements: tuple[InteractionHandDisagreement, ...],
) -> None:
    contact_records = [item for item in contacts if item.analysis_frame_index == frame]
    for item in contact_records:
        path = f"{root}/diagnostics/contact/{item.hand_source_id}/{item.part_id}"
        _log_scalar_or_clear(f"{path}/minimum_distance_pixels", item.minimum_distance_pixels)
        _log_scalar_or_clear(
            f"{path}/raw_contact_candidate",
            None if item.raw_contact_candidate is None else float(item.raw_contact_candidate),
        )
        _log_scalar_or_clear(
            f"{path}/debounced_contact_candidate",
            None
            if item.debounced_contact_candidate is None
            else float(item.debounced_contact_candidate),
        )
    records = [item for item in disagreements if item.analysis_frame_index == frame]
    matched = [item for item in records if item.assignment_state == "matched"]
    mean = (
        sum(item.mean_landmark_distance_pixels or 0 for item in matched) / len(matched)
        if matched
        else None
    )
    maximum = max((item.max_landmark_distance_pixels or 0 for item in matched), default=None)
    _log_scalar_or_clear(f"{root}/diagnostics/hand_disagreement/mean_pixels", mean)
    _log_scalar_or_clear(f"{root}/diagnostics/hand_disagreement/max_pixels", maximum)
    rr.log(
        f"{root}/diagnostics/hand_disagreement/presence_disagreement_count",
        rr.Scalars([sum(item.assignment_state != "matched" for item in records)]),
    )
    rr.log(
        f"{root}/diagnostics/hand_disagreement/handedness_disagreement_count",
        rr.Scalars([sum(bool(item.handedness_disagrees) for item in matched)]),
    )


NAVIGATION_CURRENT = "metadata/navigation/current"
NAVIGATION_SUBSTEP_INDEX = "metadata/navigation/agent_substep_index"
NAVIGATION_COARSE_GT_INDEX = "metadata/navigation/coarse_gt_index"
REVIEW_NOTES = "metadata/review_notes"
# (relative entity path, panel title) pairs the 20 s blueprint shows as static documents.
FIRST_20S_STATIC_TEXT_PANELS = (
    ("metadata/drop_dtw", "Drop-DTW weak supervision"),
    ("metadata/agent_substeps", "Agent-authored substeps (contract)"),
)
CoarseGtSegment = tuple[str, int, int]


def coarse_gt_for_frame(
    segments: tuple[CoarseGtSegment, ...], frame: int
) -> tuple[int, CoarseGtSegment] | None:
    """Return the (segment index, segment) covering `frame`, if any."""
    for index, segment in enumerate(segments):
        if segment[1] <= frame < segment[2]:
            return index, segment
    return None


def _log_navigation_static(root: str) -> None:
    rr.log(
        f"{root}/{NAVIGATION_SUBSTEP_INDEX}",
        rr.SeriesLines(names="agent-authored substep S01-S11 (visual review, [0,600) only)"),
        static=True,
    )
    rr.log(
        f"{root}/{NAVIGATION_COARSE_GT_INDEX}",
        rr.SeriesLines(names="coarse Assembly101 GT segment index (weak supervision)"),
        static=True,
    )


def _log_navigation_frame(
    root: str,
    frame: int,
    *,
    source_seconds: float,
    substep: object | None,
    coarse_gt: tuple[int, CoarseGtSegment] | None,
    extra_lines: tuple[str, ...] = (),
) -> None:
    """Log the per-frame navigation document plus its step-index time series.

    The document is re-logged every frame so a `TextDocumentView` always shows the current
    agent-authored substep and coarse GT segment; step indices give the same information a
    visible shape on the time panel.  Neither is a prediction or an accuracy claim.
    """
    if substep is None:
        substep_line = "- agent-authored substep: none (outside the checked-in `[0,600)` labels)"
        rr.log(f"{root}/{NAVIGATION_SUBSTEP_INDEX}", rr.Clear(recursive=False))
    else:
        substep_line = (
            f"- agent-authored substep: `{substep.substep_id}` **{substep.label}** "
            f"(frames [{substep.start_frame},{substep.end_frame_exclusive}), confidence "
            f"{substep.confidence}, provenance `agent_authored_visual_review`)"
        )
        rr.log(f"{root}/{NAVIGATION_SUBSTEP_INDEX}", rr.Scalars([int(substep.substep_id[1:])]))
    if coarse_gt is None:
        gt_line = "- coarse Assembly101 GT: none declared for this frame"
        rr.log(f"{root}/{NAVIGATION_COARSE_GT_INDEX}", rr.Clear(recursive=False))
    else:
        gt_index, (action, start, end) = coarse_gt
        gt_line = (
            f"- coarse Assembly101 GT (weak supervision, not prediction): segment {gt_index} "
            f"**{action}** (frames [{start},{end}))"
        )
        rr.log(f"{root}/{NAVIGATION_COARSE_GT_INDEX}", rr.Scalars([gt_index]))
    body = "\n".join(
        [
            f"# Navigation — frame {frame} / source {source_seconds:.3f} s",
            "",
            substep_line,
            gt_line,
            *extra_lines,
        ]
    )
    rr.log(f"{root}/{NAVIGATION_CURRENT}", rr.TextDocument(body, media_type="text/markdown"))


def _blueprint(
    root: str,
    dimensions: tuple[int, int],
    *,
    static_text_panels: tuple[tuple[str, str], ...] = FIRST_20S_STATIC_TEXT_PANELS,
) -> rrb.Blueprint:
    primary = rrb.Spatial2DView(
        origin=root,
        name="Primary interaction: corrected SAM3 + stabilized WiLoR",
        contents=(
            "$origin/source/video",
            "$origin/primary/reference_four_part_segmentation/**",
            "$origin/primary/stabilized_wilor/render/hands/**",
        ),
        visual_bounds=rrb.VisualBounds2D(x_range=[0, dimensions[0]], y_range=[0, dimensions[1]]),
    )
    comparison = rrb.Horizontal(
        rrb.Spatial2DView(
            origin=root,
            name="Raw WiLoR 2D (orange; toggleable)",
            contents=("$origin/source/video", "$origin/comparison/wilor_2d/render/hands/**"),
            visual_bounds=rrb.VisualBounds2D(
                x_range=[0, dimensions[0]], y_range=[0, dimensions[1]]
            ),
        ),
        rrb.Spatial2DView(
            origin=root,
            name="MediaPipe 2D (blue; fallback evidence)",
            contents=("$origin/source/video", "$origin/comparison/mediapipe_2d/render/hands/**"),
            visual_bounds=rrb.VisualBounds2D(
                x_range=[0, dimensions[0]], y_range=[0, dimensions[1]]
            ),
        ),
    )
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                primary,
                rrb.Vertical(
                    rrb.TimeSeriesView(
                        origin=f"{root}/diagnostics/contact",
                        name="Hand-to-part distances and debounced candidates",
                        contents="$origin/**",
                    ),
                    rrb.TimeSeriesView(
                        origin=f"{root}/diagnostics/hand_disagreement",
                        name="MediaPipe vs WiLoR disagreement",
                        contents="$origin/**",
                    ),
                    rrb.TimeSeriesView(
                        origin=f"{root}/metadata/navigation",
                        name="Navigation: agent substep index + coarse GT segment",
                        contents=(
                            f"$origin/{NAVIGATION_SUBSTEP_INDEX.rsplit('/', 1)[1]}",
                            f"$origin/{NAVIGATION_COARSE_GT_INDEX.rsplit('/', 1)[1]}",
                        ),
                    ),
                ),
                column_shares=[3, 2],
            ),
            comparison,
            rrb.Horizontal(
                rrb.Spatial2DView(
                    origin=root,
                    name="Kineo NLF body context (partial)",
                    contents=(
                        "$origin/source/video",
                        "$origin/contexts/kineo_nlf_body_context/**",
                    ),
                    visual_bounds=rrb.VisualBounds2D(
                        x_range=[0, dimensions[0]], y_range=[0, dimensions[1]]
                    ),
                ),
                rrb.Spatial3DView(
                    origin=f"{root}/contexts/wilor_camera_relative_non_metric_3d",
                    name="WiLoR camera-relative non-metric 3D",
                    contents="$origin/**",
                ),
                rrb.TextDocumentView(origin=f"{root}/{REVIEW_NOTES}", name="Review guide"),
                rrb.TextDocumentView(
                    origin=f"{root}/{NAVIGATION_CURRENT}",
                    name="Current substep + coarse GT (per frame)",
                ),
                rrb.Tabs(
                    *[
                        rrb.TextDocumentView(origin=f"{root}/{path}", name=title)
                        for path, title in static_text_panels
                    ]
                ),
                column_shares=[2, 2, 2, 2, 2],
            ),
            row_shares=[4, 3, 2],
        ),
        rrb.TimePanel(timeline="analysis_time", fps=ANALYSIS_FPS),
        auto_layout=False,
        auto_views=False,
    )


def _make_contact_sheet(
    video_path: Path,
    output_path: Path,
    moments: tuple[InteractionReviewPinnedMoment, ...],
    reference: LoadedSource,
    stabilized_wilor: LoadedSource,
) -> None:
    """Compact labels-only contact sheet; it is a navigation aid, not visual validation."""
    capture = cv2.VideoCapture(str(video_path))
    thumbnails: list[Image.Image] = []
    for moment in moments:
        capture.set(cv2.CAP_PROP_POS_FRAMES, moment.analysis_frame_index)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"could not decode pinned frame {moment.analysis_frame_index}")
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((400, 225))
        draw = ImageDraw.Draw(image)
        for part in TARGETS:
            mask = _mask_for_part(
                reference.observations[moment.analysis_frame_index],
                part,
                reference.run_directory,
                DIMENSIONS,
            )
            if mask is not None:
                ys, xs = np.where(mask)
                draw.rectangle(
                    (
                        xs.min() * 400 / 1280,
                        ys.min() * 225 / 720,
                        xs.max() * 400 / 1280,
                        ys.max() * 225 / 720,
                    ),
                    outline=(255, 230, 50),
                    width=1,
                )
        for hand in stabilized_wilor.observations[moment.analysis_frame_index].hands:
            x, y = _pixel(hand.landmarks[0], DIMENSIONS)
            draw.ellipse(
                (x * 400 / 1280 - 3, y * 225 / 720 - 3, x * 400 / 1280 + 3, y * 225 / 720 + 3),
                fill=(65, 169, 245),
            )
        draw.rectangle((0, 0, 400, 31), fill=(0, 0, 0))
        draw.text(
            (4, 4),
            f"f{moment.analysis_frame_index} / {moment.source_seconds:.3f}s: "
            + ", ".join(moment.categories),
            fill=(255, 255, 255),
        )
        thumbnails.append(image)
    capture.release()
    columns = 3
    rows = (len(thumbnails) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 400, rows * 225), color=(20, 20, 20))
    for index, image in enumerate(thumbnails):
        sheet.paste(image, ((index % columns) * 400, (index // columns) * 225))
    sheet.save(output_path)


def _review_guide(
    rrd_path: Path,
    index: InteractionReviewIndexManifest,
    moments: tuple[InteractionReviewPinnedMoment, ...],
) -> str:
    event_count = len(index.contact_events)
    moment_lines = "\n".join(
        f"- frame {item.analysis_frame_index} ({item.source_seconds:.3f} s): "
        f"`{', '.join(item.categories)}` — {item.rationale}"
        for item in moments
    )
    return f"""# Interaction review guide

Open exactly this recording:

```bash
rerun {rrd_path.as_posix()}
```

The one embedded RGB asset covers analysis frames `[0,600)` / source 294.000–314.000 s at
30 FPS. Start with **Primary interaction: corrected SAM3 + stabilized WiLoR**, then compare the
stabilized default with raw WiLoR and blue MediaPipe evidence panels. Use the hand-to-part
time series to navigate
geometry-only contact candidates. The BoxMOT worker context is intentionally absent from
the default blueprint; enable `contexts/boxmot_worker_context` from the entity tree only
when person/occlusion context is useful. Kineo is a separate body-context panel and covers
{index.coverage["kineo_body_frames"]}/600 frames.

## Layer meanings and claim boundaries

- **Reference parts:** `{index.reference_segmentation_method}` four-part masks. This is a
  comparison-control/reference layer, not ground truth or validated physical attachment.
- **Segmentation triggers:** {len(index.segmentation_review_triggers)} automatic geometry
  triggers are retained in the index and clustered into {len(index.segmentation_review_episodes)}
  same-part review episodes/bookmarks. They compare adjacent corrected-SAM3 masks and the
  reviewed-seed SAM2 control, using
  temporal IoU <0.5, area ratio >2, centroid jump >25 px, or cross-method IoU <0.5 while a
  stabilized hand is present; none decides target semantics.
- **Stabilized WiLoR:** the default layer preserves raw WiLoR gaps and records a separate
  raw/smoothed/fallback/missing provenance artifact. MediaPipe is used only for
  confidence/shape/workspace-gated gaps of at most five frames.
- **MediaPipe/WiLoR:** same-frame spatial comparisons use nearest wrists only. IDs are
  frame-local method labels, never cross-method or persistent identity equivalence.
- **Contact candidates:** minimum of palm/wrist and five fingertips is inside a reference
  mask or within {CONTACT_THRESHOLD_PIXELS:.0f} source pixels. Starts require
  {CONTACT_START_FRAMES} observed frames; ends require {CONTACT_END_FRAMES}. Missing hand or
  mask produces missing values and clears the debounce state. {event_count} geometry-only
  transition(s) were derived.
- **BoxMOT:** independent YOLO/BotSort person context only; not part tracking/segmentation.
- **Kineo:** 2D NLF body joints and person boxes only; no hand articulation, SfM, metric 3D,
  or BVH claim.
- **WiLoR 3D:** camera-relative non-metric coordinates, displayed separately; never
  Assembly-world aligned.
- **Drop-DTW:** Assembly101 coarse GT transcript weak supervision/navigation only, not an
  action prediction. ATHENA is metadata-only because real intrinsics are absent.
- **Agent substeps:** the checked-in 11-step `agent_authored_visual_review` timeline is the
  primary action-navigation layer. Crop-CLIP/Drop-DTW model arms stay exploratory secondary
  evidence because their checkpoints/boundaries do not establish a material improvement.

## Deterministic review bookmarks

{moment_lines}

All human dispositions are pending. Use these bookmarks to decide semantic coherence,
possible hand/object contact, and whether any method output needs a human flag; do not
treat the derived geometry or disagreement scores as accuracy measures.
"""


def _drop_dtw_bookmarks(path: Path) -> dict[int, str]:
    """Return only declared weak-supervision sample bookmarks, keyed by analysis frame."""
    alignment = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(frame): str(interval["action"])
        for interval in alignment["intervals"]
        for frame in interval["matched_analysis_frames"]
    }


def output_paths(
    repository_root: Path, output_root: Path = OUTPUT_ROOT
) -> tuple[Path, Path, Path, Path]:
    root = (repository_root / output_root).resolve()
    return (
        root / OUTPUT_NAME,
        root / INDEX_NAME,
        root / GUIDE_NAME,
        root / CONTACT_SHEET_NAME,
    )


def build_interaction_review(
    *,
    repository_root: Path,
    output_root: Path = OUTPUT_ROOT,
    reference_segmentation_method: str = "baseline_sam3",
) -> Path:
    """Build and validate the review package without model inference."""
    if reference_segmentation_method not in REFERENCE_SEGMENTATIONS:
        raise ValueError(
            "reference segmentation must be reviewed_seed_sam2_control or baseline_sam3"
        )
    repository_root = repository_root.resolve()
    contract = load_contract(
        repository_root, Path("configs/four_part_segmentation_comparison.json")
    )
    sources = {
        name: _validate_run(spec, repository_root, require_every_frame=name != "drop_dtw")
        for name, spec in DEFAULT_SOURCES.items()
    }
    reference = _validate_run(
        SourceSpec(
            reference_segmentation_method, REFERENCE_SEGMENTATIONS[reference_segmentation_method]
        ),
        repository_root,
    )
    control = _validate_run(
        SourceSpec(
            "reviewed_seed_sam2_control",
            REFERENCE_SEGMENTATIONS["reviewed_seed_sam2_control"],
        ),
        repository_root,
    )
    _validate_shared_sources([*sources.values(), reference, control])
    if tuple(item.label for item in reference.observations[0].objects) != TARGETS:
        raise ValueError("reference segmentation must preserve the four-part target order")
    # The corrected SAM3 reference preserves external masks but not a duplicated bounded
    # video. The stabilized WiLoR derivative retains the exact common approved proxy.
    video_path = sources["wilor"].run_directory / "input.mp4"
    if not video_path.is_file():
        raise FileNotFoundError("reference segmentation must include the bounded input video")
    frames, fps, dimensions = _video_info(video_path)
    if (frames, fps, dimensions) != (FRAME_COUNT, ANALYSIS_FPS, DIMENSIONS):
        raise ValueError(
            f"expected one 1280x720 600-frame 30-fps video, got {(frames, fps, dimensions)}"
        )
    contacts, events = _contacts(sources["stabilized_wilor"], reference, dimensions)
    disagreements = _disagreements(sources["mediapipe"], sources["wilor"], dimensions)
    triggers = segmentation_review_triggers(
        reference, control, sources["stabilized_wilor"], dimensions
    )
    episodes = cluster_segmentation_triggers(triggers)
    moments = deterministic_pinned_moments(
        disagreements, contacts, events, sources["kineo"], triggers, episodes
    )
    rrd_path, index_path, guide_path, sheet_path = output_paths(repository_root, output_root)
    rrd_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts = [
        _file_fingerprint(contract.path, repository_root),
        _file_fingerprint(video_path, repository_root),
        *[item for source in [*sources.values(), reference, control] for item in source.artifacts],
    ]
    unique_artifacts = tuple({(item.uri, item.sha256): item for item in artifacts}.values())
    source_fingerprint = getattr(_metadata(reference.manifest), "source_fingerprint")
    assert isinstance(source_fingerprint, ArtifactFingerprint)
    index = InteractionReviewIndexManifest(
        manifest_kind="interaction_review_first_20s",
        comparison_id=COMPARISON_ID,
        source_video=source_fingerprint,
        bounded_video=_file_fingerprint(video_path, repository_root),
        frame_count=FRAME_COUNT,
        analysis_fps=ANALYSIS_FPS,
        source_interval=TimeInterval(start_seconds=294.0, end_seconds=314.0),
        reference_segmentation_method=reference_segmentation_method,
        reference_segmentation_manifest=_file_fingerprint(
            reference.run_directory / "manifest.json", repository_root
        ),
        input_artifacts=unique_artifacts,
        contact_heuristic=(
            "Raw candidate when wrist/palm or any of thumb/index/middle/ring/pinky tips is "
            f"inside the reference mask or within {CONTACT_THRESHOLD_PIXELS:.0f} source pixels; "
            f"{CONTACT_START_FRAMES}-frame start and {CONTACT_END_FRAMES}-frame end debounce; "
            "missing observations clear state rather than implying far-away."
        ),
        hand_matching_rule=(
            "Greedy same-frame nearest-wrist assignment; assignment is not persistent "
            "identity matching."
        ),
        coordinate_semantics=(
            "All 2D geometry maps normalized top-left image coordinates to 1280x720 source pixels.",
            "Reference masks remain inside the selected run's native artifact root.",
            "WiLoR 3D is camera-relative and non-metric, never Assembly-world aligned.",
        ),
        claim_boundaries=(
            "Contact and event names are geometry heuristics, not ground-truth touch/grasp labels.",
            "MediaPipe and WiLoR IDs are never equated across methods or time.",
            "BoxMOT is worker/occlusion context, not part tracking or segmentation.",
            (
                "Kineo panel is NLF-only 2D body context; no hand articulation, SfM, metric "
                "3D, or BVH."
            ),
            "Drop-DTW uses Assembly101 GT coarse transcript weak supervision, not prediction.",
            "ATHENA is metadata-only because real camera intrinsics are unavailable.",
        ),
        coverage={
            "reference_part_mask_frames": 600,
            "mediapipe_hand_frames": sum(
                bool(source.hands) for source in sources["mediapipe"].observations.values()
            ),
            "wilor_hand_frames": sum(
                bool(source.hands) for source in sources["wilor"].observations.values()
            ),
            "stabilized_wilor_hand_frames": sum(
                bool(source.hands) for source in sources["stabilized_wilor"].observations.values()
            ),
            "boxmot_person_frames": sum(
                bool(source.objects) for source in sources["boxmot"].observations.values()
            ),
            "kineo_body_frames": sum(
                bool(source.nlf_body_2d) for source in sources["kineo"].observations.values()
            ),
        },
        contact_diagnostics=contacts,
        hand_disagreements=disagreements,
        contact_events=events,
        segmentation_review_triggers=triggers,
        segmentation_review_episodes=episodes,
        pinned_moments=moments,
    )
    rr.init("battle-interaction-review", recording_id=COMPARISON_ID)
    rr.save(rrd_path)
    root = f"world/{reference.manifest.clip.clip_id}/interaction_review"
    rr.log(f"{root}/source/video_asset", rr.AssetVideo(path=video_path), static=True)
    rr.log(
        f"{root}/metadata/index",
        rr.TextDocument(index.model_dump_json(indent=2), media_type="application/json"),
        static=True,
    )
    trigger_counts = {
        frame: sum(item.analysis_frame_index == frame for item in triggers)
        for frame in range(FRAME_COUNT)
    }
    drop_path = sources["drop_dtw"].run_directory / "alignment.json"
    drop_text, drop_cost = _drop_dtw_text(drop_path)
    drop_bookmarks = _drop_dtw_bookmarks(drop_path)
    rr.log(
        f"{root}/metadata/drop_dtw",
        rr.TextDocument(drop_text, media_type="text/markdown"),
        static=True,
    )
    fine_contract = load_fine_substep_contract(
        repository_root
        / "configs/fine_substeps/assembly101_focused_static_first_20s_agent_labels.json"
    )
    rr.log(
        f"{root}/metadata/agent_substeps",
        rr.TextDocument(fine_contract.model_dump_json(indent=2), media_type="application/json"),
        static=True,
    )
    rr.log(
        f"{root}/metadata/athena_limitations",
        rr.TextDocument(
            "# ATHENA\nBlocked on missing Assembly101 camera intrinsics. "
            "Its fixture is not on this timeline.",
            media_type="text/markdown",
        ),
        static=True,
    )
    rr.log(
        f"{root}/contexts/wilor_camera_relative_non_metric_3d/frame",
        rr.TextDocument(
            "WiLoR native camera-relative, non-metric axes/frame. "
            "Not aligned to Assembly world coordinates.",
            media_type="text/markdown",
        ),
        static=True,
    )
    guide = _review_guide(rrd_path, index, moments)
    rr.log(
        f"{root}/{REVIEW_NOTES}",
        rr.TextDocument(guide, media_type="text/markdown"),
        static=True,
    )
    coarse_gt_segments: tuple[CoarseGtSegment, ...] = tuple(
        (anchor.action, anchor.proxy_start_frame, anchor.proxy_end_frame_exclusive)
        for anchor in fine_contract.coarse_gt_anchors
    )
    _log_navigation_static(root)
    for frame in range(FRAME_COUNT):
        time = frame / ANALYSIS_FPS
        rr.set_time("analysis_frame", sequence=frame)
        rr.set_time("analysis_time", duration=time)
        rr.set_time("source_time", duration=294.0 + time)
        rr.log(
            f"{root}/source/video",
            rr.VideoFrameReference(seconds=time, video_reference=f"{root}/source/video_asset"),
        )
        _log_reference_masks(
            root, reference.observations[frame], reference.run_directory, dimensions
        )
        _log_hands(
            f"{root}/primary/stabilized_wilor/render",
            sources["stabilized_wilor"].observations[frame],
            dimensions=dimensions,
            color=METHOD_COLORS["wilor"],
            include_3d=False,
        )
        _log_hands(
            f"{root}/comparison/wilor_2d/render",
            sources["wilor"].observations[frame],
            dimensions=dimensions,
            color=METHOD_COLORS["wilor"],
            include_3d=False,
        )
        _log_hands(
            f"{root}/comparison/mediapipe_2d/render",
            sources["mediapipe"].observations[frame],
            dimensions=dimensions,
            color=METHOD_COLORS["mediapipe"],
            include_3d=False,
        )
        _log_hands(
            f"{root}/contexts/wilor_camera_relative_non_metric_3d",
            sources["wilor"].observations[frame],
            dimensions=dimensions,
            color=METHOD_COLORS["wilor"],
            include_3d=True,
        )
        _log_context_boxes(
            root, "boxmot_worker_context", sources["boxmot"].observations[frame], dimensions
        )
        kineo_root = f"{root}/contexts/kineo_nlf_body_context"
        _log_context_boxes(
            root, "kineo_nlf_body_context", sources["kineo"].observations[frame], dimensions
        )
        _log_nlf_body(
            kineo_root,
            sources["kineo"].observations[frame],
            dimensions=dimensions,
            color=METHOD_COLORS["kineo_nlf"],
        )
        _log_diagnostics_frame(root, frame, contacts, disagreements)
        rr.log(
            f"{root}/diagnostics/segmentation_review_trigger/count",
            rr.Scalars([trigger_counts[frame]]),
        )
        if (action := drop_bookmarks.get(frame)) is not None:
            rr.log(
                f"{root}/metadata/drop_dtw_bookmarks",
                rr.TextLog(f"GT coarse transcript weak-supervision bookmark: {action}"),
            )
        agent_substep = substep_for_frame(fine_contract, frame)
        rr.log(
            f"{root}/metadata/agent_substeps/timeline",
            rr.TextLog(
                f"{agent_substep.substep_id}: {agent_substep.label} (agent_authored_visual_review)"
            ),
        )
        _log_navigation_frame(
            root,
            frame,
            source_seconds=294.0 + time,
            substep=agent_substep,
            coarse_gt=coarse_gt_for_frame(coarse_gt_segments, frame),
            extra_lines=(
                (f"- Drop-DTW weak-supervision bookmark at this frame: **{action}**",)
                if action is not None
                else ()
            ),
        )
        if frame == 0:
            rr.log(f"{root}/metadata/drop_dtw_alignment_cost", rr.Scalars([drop_cost]))
    rr.send_blueprint(_blueprint(root, dimensions))
    rr.disconnect()
    _make_contact_sheet(video_path, sheet_path, moments, reference, sources["stabilized_wilor"])
    guide_path.write_text(guide, encoding="utf-8")
    final = index.model_copy(
        update={
            "output_rrd": _file_fingerprint(rrd_path, repository_root),
            "review_guide": _file_fingerprint(guide_path, repository_root),
            "contact_sheet": _file_fingerprint(sheet_path, repository_root),
        }
    )
    index_path.write_text(final.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return rrd_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--reference-segmentation",
        choices=tuple(REFERENCE_SEGMENTATIONS),
        default="baseline_sam3",
        help="Reference part source; default is the corrected focused static SAM3 run.",
    )
    args = parser.parse_args()
    print(
        build_interaction_review(
            repository_root=args.repository_root,
            output_root=args.output_root,
            reference_segmentation_method=args.reference_segmentation,
        )
    )


if __name__ == "__main__":
    main()
