"""Build the first-minute, inference-free interaction review recording.

The v4 package deliberately stops geometric contact candidates at the documented late
segmentation validity boundary.  Its additional layers remain independently derived
context, never a joint tracker or an action prediction.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rerun as rr

from . import assembly101_reference as a101
from . import athena_hands_review, mask_cache, multiview_consensus, multiview_review
from . import ensemble_reference as ensemble
from . import interaction_review as review
from .assembly101_pose_schemas import (
    ASSEMBLY101_EDGES,
    ASSEMBLY101_JOINT_NAMES,
    Assembly101HandFrame,
    Assembly101ReferenceManifest,
)
from .build_phases import PhaseTimer
from .cli_common import add_output_flags, add_output_root, add_repository_root
from .ensemble_reference_schemas import (
    PROVENANCE_CODES,
    EnsembleProvenanceSidecar,
    EnsembleReferencePolicy,
)
from .exploratory_comparison import (
    _file_fingerprint,
    _log_hands,
    _log_nlf_body,
    validate_artifact_fingerprint,
)
from .fine_substep_contract import load_contract as load_fine_substep_contract
from .fine_substep_contract import substep_for_frame
from .four_part_contract import ANALYSIS_FPS, TARGETS
from .observations import object_for_label
from .rerun_logging import init_and_save, log_boxes_from_observation, log_rgba_mask
from .schemas import (
    ArtifactFingerprint,
    FrameObservations,
    InteractionReviewIndexManifest,
    SegmentationValidityInterval,
    TimeInterval,
)

FRAME_COUNT = 1800
SOURCE_START_SECONDS = 294.0
OUTPUT_ROOT = Path("runs/interaction-review-first-minute-v4")
OUTPUT_NAME = "interaction_review_first_minute_v4.rrd"
INDEX_NAME = "interaction_review_index.json"
GUIDE_NAME = "review_guide.md"
CONTACT_SHEET_NAME = "pinned_moments_contact_sheet.png"
# Files `battle-review-presets` and the recording merge add next to a package (`.rbl` presets,
# a merged `.rrd`, the preset check); `--overwrite` may replace them along with the package.
PACKAGE_SIDE_FILE_SUFFIXES = frozenset({".rbl", ".rrd", ".stdout", ".log"})
PRESET_CHECK_NAME = "presets_check.json"
FINE_LABELS = Path("configs/fine_substeps/assembly101_focused_static_first_20s_agent_labels.json")

# Corrected focused SAM3 rerun whose schedule adds an agent-selected chassis/interior
# correction at frame 1172 on top of the human-selected seeds and corrections. Frames before
# 1172 are bit-identical to the earlier reference run. It is the ensemble's primary source and
# remains available through `--reference` for a SAM3-only package.
CORRECTED_SAM3_REFERENCE_SEGMENTATION = Path(
    "runs/muggledsam-sam3-four-part-static-focused-reassembly-static-c10379-20260918t001210z"
)
# Default reference: the per-target ensemble built by `battle-build-ensemble-reference`
# (corrected SAM3 by default, DAM4SAM chassis fallback inside [1055,1172) where the policy
# rules fire and the substitute passes sanity, explicit hidden interior over [1024,1172)).
FIRST_MINUTE_REFERENCE_SEGMENTATION = ensemble.OUTPUT_ROOT
# Contact geometry is measured only where the corrected masks keep stable part identities.
# For a SAM3-only reference the human-reported chassis/interior swap was measured to start
# leaking at ~1020-1032 and the chassis label is lost or swapped until the agent-selected
# correction at 1172; frames from 1200 remain ineligible per the retained late-degradation
# review. With the ensemble reference eligibility is per target and comes from the
# provenance sidecar instead.
SWAP_ONSET_FRAME = 1020
SWAP_CORRECTION_FRAME = 1172
SEGMENTATION_CONTACT_ELIGIBLE_THROUGH = 1200
SEGMENTATION_CONTACT_ELIGIBLE_INTERVALS = (
    (0, SWAP_ONSET_FRAME),
    (SWAP_CORRECTION_FRAME, SEGMENTATION_CONTACT_ELIGIBLE_THROUGH),
)
COARSE_GT = (
    ("attach interior", 0, 318),
    ("screw chassis", 318, 1031),
    ("attach body", 1031, 1281),
    ("screw chassis", 1281, 1800),
)


STATIC_TEXT_PANELS = (
    ("metadata/fine_grained_gt", "Fine-grained Assembly101 GT (dataset annotation)"),
    ("metadata/coarse_gt", "Coarse Assembly101 GT (weak supervision)"),
    ("metadata/agent_substeps_first_20s", "Agent-authored substeps (contract, [0,600))"),
    ("metadata/drop_dtw", "Drop-DTW status"),
)
# Dataset reference window built by `battle-build-assembly101-reference`: the recording's
# own 60 fps hand poses and fine-grained labels resampled onto this proxy's analysis clock.
ASSEMBLY101_REFERENCE = a101.OUTPUT_ROOT
ASSEMBLY101_HAND_COLORS: dict[str, tuple[int, int, int]] = {
    "left": (255, 235, 130),
    "right": (140, 255, 235),
}
ASSEMBLY101_CAMERA_PLANE_MM = 300.0
DROP_DTW_STATUS = """# Drop-DTW weak supervision (first minute)

No Drop-DTW alignment was run for the first-minute window. The only Drop-DTW artifact is the
20 s pinned OpenCLIP alignment shown in `runs/interaction-review-first-20s`. In this package the
coarse Assembly101 GT transcript (same weak-supervision basis, no model alignment) drives the
per-frame navigation document and the coarse GT segment index time series. It is navigation
context only, never a prediction or an accuracy claim.
"""


HAND_LAYER_STATES = ("low_confidence_continuation", "fallback", "missing")
PROVENANCE_CODE_LEGEND = ", ".join(f"{code} {name}" for name, code in PROVENANCE_CODES.items())
PROVENANCE_DISPLAY_NAMES = {
    "missing": "missing",
    "sam3_corrected": "sam3 primary",
    "dam4sam_fallback": "dam4sam fallback",
    "hidden_agent_label": "hidden (agent label)",
}
APPLICATION_ID = "battle-interaction-review-v4"
RECORDING_ID = "interaction_review_first_minute_v4"
# Candidate segmentation arms (`--candidate-arm NAME=RUN_DIR`) are logged beside the reference
# so the same frame can be read across trackers; they are comparison evidence, never a
# second reference.  Human anchor masks are logged as one more arm on the 13 anchor frames.
CANDIDATE_SEGMENTATION_ROOT = review.CANDIDATE_SEGMENTATION_ROOT
CANDIDATE_AREA_SERIES = review.CANDIDATE_AREA_SERIES
HUMAN_ANCHOR_ARM = "human_anchors"
HUMAN_ANCHOR_OUTLINES = review.HUMAN_ANCHOR_OUTLINES
CONFIDENCE_SERIES = review.CONFIDENCE_SERIES
ANCHOR_SERIES = review.ANCHOR_SERIES
ANCHOR_LOG = f"{review.ANCHOR_SERIES}/log"
ANCHOR_CONFIG = Path("configs/qa/first_minute_review_anchors.json")
ANCHOR_MASKS = Path("runs/human-review-anchors-first-minute/anchors/anchor_masks.json")
CONFIDENCE_NAME = "confidence.jsonl"
ARM_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def _intervals_text(intervals: tuple[tuple[int, int], ...]) -> str:
    return ", ".join(f"[{start},{end})" for start, end in intervals) or "none"


def provenance_legend(sidecar: EnsembleProvenanceSidecar, policy: EnsembleReferencePolicy) -> str:
    """Legend built from the states the sidecar actually holds, plus the policy's named
    ineligible intervals (a `not_contact_eligible` interval is not a provenance state, so it
    is listed separately instead of a code no frame carries)."""
    counts: dict[str, int] = {}
    for summary in sidecar.summaries:
        for state, count in summary.provenance_counts.items():
            counts[state] = counts.get(state, 0) + int(count)
    codes = ", ".join(
        f"{code} {PROVENANCE_DISPLAY_NAMES.get(name, name)}"
        for name, code in PROVENANCE_CODES.items()
        if counts.get(name, 0) > 0
    )
    ineligible = [
        f"{target.target_id} [{item.start_frame},{item.end_frame_exclusive}) "
        f"not_contact_eligible{': ' + item.failure_case if item.failure_case else ''}"
        for target in policy.targets
        for item in target.not_contact_eligible_intervals
    ]
    return codes + (f"; {'; '.join(ineligible)}" if ineligible else "")


def _reference_provenance_static(entity: str, legend: str = PROVENANCE_CODE_LEGEND) -> None:
    for part in TARGETS:
        rr.log(
            f"{entity}/{review.REFERENCE_PROVENANCE_SERIES}/{part}",
            rr.SeriesLines(
                names=f"{part} mask provenance ({legend})",
                colors=[ensemble.PART_COLORS[part]],
            ),
            static=True,
        )


# -- candidate arms, confidence series, anchor marks -------------------------------------------


def parse_candidate_arm(spec: str) -> tuple[str, Path]:
    """`NAME=RUN_DIR` for `--candidate-arm`; the name becomes an entity path segment."""
    name, separator, run_directory = spec.partition("=")
    name = name.strip()
    if not separator or not name or not run_directory.strip():
        raise argparse.ArgumentTypeError(f"expected NAME=RUN_DIR, got {spec!r}")
    if not ARM_NAME_PATTERN.fullmatch(name):
        raise argparse.ArgumentTypeError(
            f"candidate arm name {name!r} must match {ARM_NAME_PATTERN.pattern}"
        )
    if name == HUMAN_ANCHOR_ARM:
        raise argparse.ArgumentTypeError(f"{HUMAN_ANCHOR_ARM!r} is reserved for the anchor masks")
    return name, Path(run_directory.strip())


def _candidate_arm_static(entity: str, name: str) -> None:
    for part in TARGETS:
        rr.log(
            f"{entity}/{CANDIDATE_AREA_SERIES}/{name}/{part}",
            rr.SeriesLines(names=f"{name} {part} area (px)", colors=[ensemble.PART_COLORS[part]]),
            static=True,
        )


def _log_candidate_arm_frame(
    entity: str, name: str, source: review.LoadedSource, frame: int
) -> dict[str, int]:
    """One RGBA cut-out per part under the arm's own root, plus the mask area series."""
    observation = source.observations.get(frame)
    cache = mask_cache.cache_for(source.run_directory)
    areas: dict[str, int] = {}
    for part in TARGETS:
        item = object_for_label(observation, part) if observation is not None else None
        mask_path = f"{entity}/{CANDIDATE_SEGMENTATION_ROOT}/{name}/{part}"
        area_path = f"{entity}/{CANDIDATE_AREA_SERIES}/{name}/{part}"
        if item is None or item.mask is None:
            rr.log(mask_path, rr.Clear(recursive=False))
            rr.log(area_path, rr.Clear(recursive=False))
            continue
        log_rgba_mask(
            mask_path, cache.rgba_png(item.mask.uri, ensemble.PART_COLORS[part]), opacity=0.35
        )
        area = int(np.count_nonzero(cache.mask(item.mask.uri)))
        areas[part] = area
        rr.log(area_path, rr.Scalars([float(area)]))
    return areas


@dataclass(frozen=True)
class ConfidenceRow:
    confidence: float
    abstain: bool
    is_anchor_frame: bool
    anchor_truth_failed: bool | None


def load_confidence_series(path: Path) -> dict[tuple[int, str], ConfidenceRow]:
    """`confidence.jsonl` of `battle-detector-scorecard`: one row per frame x part."""
    rows: dict[tuple[int, str], ConfidenceRow] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        key = (int(item["analysis_frame_index"]), str(item["target"]))
        if key in rows:
            raise ValueError(f"{path} repeats frame x part {key}")
        rows[key] = ConfidenceRow(
            confidence=float(item["confidence"]),
            abstain=bool(item["abstain"]),
            is_anchor_frame=bool(item.get("is_anchor_frame", False)),
            anchor_truth_failed=item.get("anchor_truth_failed"),
        )
    return rows


def _confidence_static(entity: str, run_label: str) -> None:
    for part in TARGETS:
        rr.log(
            f"{entity}/{CONFIDENCE_SERIES}/{part}/confidence",
            rr.SeriesLines(
                names=f"{part} detector confidence ({run_label}; 1 - combined suspicion)",
                colors=[ensemble.PART_COLORS[part]],
            ),
            static=True,
        )
        rr.log(
            f"{entity}/{CONFIDENCE_SERIES}/{part}/abstain",
            rr.SeriesPoints(
                names=f"{part} abstain (confidence at or below the R>=0.8 point)",
                colors=[ensemble.PART_COLORS[part]],
                markers="cross",
                marker_sizes=3.0,
            ),
            static=True,
        )


def _log_confidence_frame(
    entity: str, frame: int, rows: dict[tuple[int, str], ConfidenceRow]
) -> None:
    for part in TARGETS:
        row = rows.get((frame, part))
        review._log_scalar_or_clear(
            f"{entity}/{CONFIDENCE_SERIES}/{part}/confidence",
            None if row is None else row.confidence,
        )
        # Abstentions are logged as points only where they occur so the panel reads as marks.
        review._log_scalar_or_clear(
            f"{entity}/{CONFIDENCE_SERIES}/{part}/abstain",
            1.0 if row is not None and row.abstain else None,
        )


@dataclass(frozen=True)
class AnchorMarks:
    """Human review anchors: frames, per-cell state, and (when exported) the mask PNGs."""

    frames: tuple[int, ...]
    windows: dict[str, tuple[int, int]]
    states: dict[tuple[int, str], str]
    masks: dict[tuple[int, str], Path]
    config_path: Path
    mask_set_path: Path | None

    def window_for(self, frame: int) -> str | None:
        for name, (start, end) in self.windows.items():
            if start <= frame < end:
                return name
        return None


def load_anchor_marks(
    repository_root: Path, config_path: Path = ANCHOR_CONFIG, mask_set: Path | None = ANCHOR_MASKS
) -> AnchorMarks:
    config = json.loads((repository_root / config_path).read_text(encoding="utf-8"))
    frames = tuple(int(item["analysis_frame_index"]) for item in config["frames"])
    windows = {name: (int(a), int(b)) for name, (a, b) in config.get("windows", {}).items()}
    states: dict[tuple[int, str], str] = {}
    masks: dict[tuple[int, str], Path] = {}
    mask_set_path = None
    if mask_set is not None and (repository_root / mask_set).is_file():
        mask_set_path = repository_root / mask_set
        exported = json.loads(mask_set_path.read_text(encoding="utf-8"))
        for anchor in exported["anchors"]:
            key = (int(anchor["analysis_frame_index"]), str(anchor["target"]))
            states[key] = str(anchor["state"])
            if anchor.get("mask_uri"):
                masks[key] = mask_set_path.parent.parent / anchor["mask_uri"]
    return AnchorMarks(
        frames=frames,
        windows=windows,
        states=states,
        masks=masks,
        config_path=repository_root / config_path,
        mask_set_path=mask_set_path,
    )


def mask_outlines(mask: np.ndarray) -> list[list[list[float]]]:
    """Closed contour polylines (pixel coordinates) of a boolean mask."""
    import cv2

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    strips: list[list[list[float]]] = []
    for contour in contours:
        points = contour.reshape(-1, 2).astype(float).tolist()
        if len(points) >= 3:
            strips.append([*points, points[0]])
    return strips


def _anchor_static(entity: str, anchors: AnchorMarks, *, with_masks: bool) -> None:
    rr.log(
        f"{entity}/{ANCHOR_SERIES}/anchor_frame",
        rr.SeriesPoints(
            names=f"human anchor frame ({len(anchors.frames)} frames, C10379)",
            colors=[(255, 255, 255)],
            markers="diamond",
            marker_sizes=6.0,
        ),
        static=True,
    )
    rr.log(
        f"{entity}/{ANCHOR_SERIES}/failed_cells",
        rr.SeriesPoints(
            names="anchor cells the scored run fails (IoU < 0.5, or hidden with > 300 px)",
            colors=[(255, 80, 80)],
            markers="cross",
            marker_sizes=6.0,
        ),
        static=True,
    )
    if with_masks:
        _candidate_arm_static(entity, HUMAN_ANCHOR_ARM)


def _log_anchor_frame(
    entity: str,
    frame: int,
    anchors: AnchorMarks,
    confidence: dict[tuple[int, str], ConfidenceRow] | None,
) -> None:
    """Marks on the timeline at anchor frames, the anchor masks as outlines on the primary view
    and as one more arm; everything is cleared on the following frame so nothing lingers."""
    is_anchor = frame in anchors.frames
    was_anchor = (frame - 1) in anchors.frames
    if not is_anchor and not was_anchor:
        return
    if not is_anchor:
        rr.log(f"{entity}/{ANCHOR_SERIES}/anchor_frame", rr.Clear(recursive=False))
        rr.log(f"{entity}/{ANCHOR_SERIES}/failed_cells", rr.Clear(recursive=False))
        rr.log(f"{entity}/{HUMAN_ANCHOR_OUTLINES}", rr.Clear(recursive=True))
        rr.log(
            f"{entity}/{CANDIDATE_SEGMENTATION_ROOT}/{HUMAN_ANCHOR_ARM}", rr.Clear(recursive=True)
        )
        rr.log(f"{entity}/{CANDIDATE_AREA_SERIES}/{HUMAN_ANCHOR_ARM}", rr.Clear(recursive=True))
        return
    rr.log(f"{entity}/{ANCHOR_SERIES}/anchor_frame", rr.Scalars([1.0]))
    failed = [
        part
        for part in TARGETS
        if confidence is not None
        and (row := confidence.get((frame, part))) is not None
        and row.anchor_truth_failed
    ]
    review._log_scalar_or_clear(
        f"{entity}/{ANCHOR_SERIES}/failed_cells", float(len(failed)) if confidence else None
    )
    lines = []
    for part in TARGETS:
        state = anchors.states.get((frame, part), "unlabeled")
        verdict = ""
        if confidence is not None and (row := confidence.get((frame, part))) is not None:
            if row.anchor_truth_failed is True:
                verdict = " FAILED"
            elif row.anchor_truth_failed is False:
                verdict = " ok"
        lines.append(f"{part} {state}{verdict}")
        mask_path = anchors.masks.get((frame, part))
        if mask_path is None:
            continue
        mask = mask_cache.decode_mask_png(mask_path)
        color = ensemble.PART_COLORS[part]
        strips = mask_outlines(mask)
        if strips:
            rr.log(
                f"{entity}/{HUMAN_ANCHOR_OUTLINES}/{part}",
                rr.LineStrips2D(
                    strips,
                    colors=[color] * len(strips),
                    radii=1.5,
                    labels=[f"human anchor {part} f{frame}"],
                    draw_order=3.0,
                ),
            )
        log_rgba_mask(
            f"{entity}/{CANDIDATE_SEGMENTATION_ROOT}/{HUMAN_ANCHOR_ARM}/{part}",
            mask_cache.encode_rgba_mask_png(mask, color),
            opacity=0.35,
        )
        rr.log(
            f"{entity}/{CANDIDATE_AREA_SERIES}/{HUMAN_ANCHOR_ARM}/{part}",
            rr.Scalars([float(np.count_nonzero(mask))]),
        )
    window = anchors.window_for(frame)
    heading = f"anchor f{frame}" + (f" (window {window})" if window else "")
    rr.log(
        f"{entity}/{ANCHOR_LOG}",
        rr.TextLog(f"{heading}: " + "; ".join(lines), level="WARN" if failed else "INFO"),
    )


def _log_reference_provenance_frame(
    entity: str,
    frame: int,
    observation: FrameObservations,
    provenance: dict[tuple[int, str], str],
    eligibility: dict[str, tuple[tuple[int, int], ...]],
    dimensions: tuple[int, int],
) -> None:
    """Colour-coded provenance layer: a scalar series per part plus a toggleable 2D overlay.

    The overlay marks only frames whose mask did not come from the primary tracker (DAM4SAM
    fallback boxes around the copied mask; hidden intervals clear the overlay) so a reviewer
    can see at a glance which frames rest on cross-method fallback.
    """
    for part in TARGETS:
        state = provenance[(frame, part)]
        rr.log(
            f"{entity}/{review.REFERENCE_PROVENANCE_SERIES}/{part}",
            rr.Scalars([PROVENANCE_CODES[state]]),
        )
        rr.log(
            f"{entity}/diagnostics/segmentation_contact_eligible/{part}",
            rr.Scalars([float(review.contact_eligible_frame(frame, eligibility.get(part, ())))]),
        )
        overlay = f"{entity}/{review.REFERENCE_PROVENANCE_OVERLAY}/{part}"
        item = object_for_label(observation, part, require_mask=False)
        if state == "dam4sam_fallback" and item is not None:
            log_boxes_from_observation(
                overlay,
                [item],
                dimensions,
                labels=[f"{part}: {state}"],
                colors=[ensemble.PROVENANCE_COLORS[state]],
                draw_order=2.0,
            )
        else:
            rr.log(overlay, rr.Clear(recursive=False))


def _load_ensemble_context(
    reference: review.LoadedSource, repository_root: Path
) -> tuple[EnsembleProvenanceSidecar, EnsembleReferencePolicy] | None:
    """Sidecar and policy for an ensemble reference; None for a plain SAM3 reference."""
    metadata = reference.manifest.ensemble_reference
    if metadata is None:
        return None
    sidecar = ensemble.load_sidecar(reference.run_directory)
    policy_path = validate_artifact_fingerprint(
        metadata.policy_fingerprint, repository_root, label="ensemble policy"
    )
    if sidecar.policy != metadata.policy_fingerprint:
        raise ValueError("ensemble sidecar and manifest disagree on the policy fingerprint")
    return sidecar, ensemble.load_policy(policy_path)


def _hand_layer_state_counts(provenance_path: Path) -> dict[tuple[int, str], int]:
    """Per-frame counts of the stabilized layer's explicit non-primary hand states."""
    counts: dict[tuple[int, str], int] = {}
    for item in json.loads(provenance_path.read_text(encoding="utf-8")):
        key = (int(item["analysis_frame_index"]), str(item["state"]))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _output_paths(root: Path) -> tuple[Path, Path, Path, Path]:
    return (
        root / OUTPUT_NAME,
        root / INDEX_NAME,
        root / GUIDE_NAME,
        root / CONTACT_SHEET_NAME,
    )


def _coarse_gt_markdown() -> str:
    rows = "\n".join(
        f"| {index} | {action} | {start} | {end} | "
        f"{SOURCE_START_SECONDS + start / ANALYSIS_FPS:.3f}–"
        f"{SOURCE_START_SECONDS + end / ANALYSIS_FPS:.3f} s |"
        for index, (action, start, end) in enumerate(COARSE_GT)
    )
    return (
        "# Coarse Assembly101 GT over the first minute\n\n"
        "Weak navigation context from the dataset transcript, not a prediction and not a "
        "fine-grained label. Segment index is what the navigation time series plots.\n\n"
        "| index | action | start frame | end frame (excl.) | source |\n"
        "| --- | --- | --- | --- | --- |\n"
        f"{rows}\n"
    )


def _fine_gt_markdown(manifest: Assembly101ReferenceManifest) -> str:
    rows = "\n".join(
        f"| {index} | {segment.action} | {segment.proxy_start_frame} | "
        f"{segment.proxy_end_frame_exclusive} | "
        f"{SOURCE_START_SECONDS + segment.proxy_start_frame / ANALYSIS_FPS:.3f}–"
        f"{SOURCE_START_SECONDS + segment.proxy_end_frame_exclusive / ANALYSIS_FPS:.3f} s | "
        f"{segment.annotation_id}{' (clipped)' if segment.clipped_to_window else ''} |"
        for index, segment in enumerate(manifest.fine_segments)
    )
    rule = manifest.clock_rule
    return (
        "# Fine-grained Assembly101 GT over the first minute\n\n"
        f"Dataset human annotations (30 FPS annotation clock; proxy frame = annotation frame − "
        f"{manifest.annotation_start_frame}) for recording `{manifest.recording_id}` at "
        f"revision `{manifest.dataset_revision[:12]}`. Two-hand actions are annotated as "
        "overlapping segments. Navigation context, not a prediction; licence "
        f"{manifest.license}.\n\n"
        f"Static-view pose clock: `pose_frame = {rule.proxy_start_raw_frame} + "
        f"{rule.raw_frames_per_proxy_frame}·proxy_frame {rule.pose_offset_frames:+d}` "
        f"(±{rule.offset_uncertainty_frames} frame).\n\n"
        "| index | action | start frame | end frame (excl.) | source | annotation id |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        f"{rows}\n"
    )


def _log_assembly101_static(entity: str, manifest: Assembly101ReferenceManifest) -> None:
    """Camera estimate as a frustum in the world-mm view plus the diagnostics series names."""
    camera = manifest.camera
    pose = np.asarray(camera.camera_to_world, dtype=np.float64)
    camera_root = f"{entity}/{review.ASSEMBLY101_3D_ROOT}/camera/{camera.view_key.split(':')[0]}"
    rr.log(
        camera_root,
        rr.Transform3D(translation=pose[:3, 3], mat3x3=pose[:3, :3]),
        static=True,
    )
    rr.log(
        camera_root,
        rr.Pinhole(
            image_from_camera=np.asarray(camera.intrinsic_matrix, dtype=np.float64),
            resolution=list(camera.raw_image_size),
            camera_xyz=rr.ViewCoordinates.RDF,
            image_plane_distance=ASSEMBLY101_CAMERA_PLANE_MM,
        ),
        static=True,
    )
    for side, color in ASSEMBLY101_HAND_COLORS.items():
        rr.log(
            f"{entity}/{review.ASSEMBLY101_DIAGNOSTICS}/confidence/{side}",
            rr.SeriesLines(names=f"dataset {side} hand confidence", colors=[color]),
            static=True,
        )
        rr.log(
            f"{entity}/{review.ASSEMBLY101_DIAGNOSTICS}/wrist_distance_to_stabilized_wilor_pixels/{side}",
            rr.SeriesLines(
                names=f"dataset {side} wrist -> nearest stabilized WiLoR wrist (px)",
                colors=[color],
            ),
            static=True,
        )


def _log_assembly101_frame(
    entity: str,
    frame: Assembly101HandFrame,
    wilor_observation: FrameObservations,
    *,
    dimensions: tuple[int, int],
    draw_threshold: float,
) -> None:
    """Dataset hands in proxy pixels and world millimetres, plus per-side diagnostics.

    Hands below the draw threshold stay out of both spatial views but keep their confidence
    on the time panel.  The wrist distance compares against the nearest stabilized WiLoR
    wrist regardless of reported side because method handedness is not stable.
    """
    two_d_root = f"{entity}/{review.ASSEMBLY101_2D_ROOT}"
    three_d_root = f"{entity}/{review.ASSEMBLY101_3D_ROOT}/hands"
    width, height = dimensions
    wilor_wrists = np.array(
        [
            [hand.landmarks[0].x * width, hand.landmarks[0].y * height]
            for hand in wilor_observation.hands
        ]
    ).reshape(-1, 2)
    present = {hand.side: hand for hand in frame.hands}
    for side in ASSEMBLY101_HAND_COLORS:
        hand = present.get(side)
        review._log_scalar_or_clear(
            f"{entity}/{review.ASSEMBLY101_DIAGNOSTICS}/confidence/{side}",
            None if hand is None else hand.confidence,
        )
        distance = None
        if hand is not None and hand.confidence >= draw_threshold and len(wilor_wrists):
            wrist = np.array(a101.wrist_pixels(hand))
            distance = float(np.min(np.linalg.norm(wilor_wrists - wrist, axis=1)))
        review._log_scalar_or_clear(
            f"{entity}/{review.ASSEMBLY101_DIAGNOSTICS}/wrist_distance_to_stabilized_wilor_pixels/{side}",
            distance,
        )
    drawn = [hand for hand in frame.hands if hand.confidence >= draw_threshold]
    if not drawn:
        rr.log(two_d_root, rr.Clear(recursive=True))
        rr.log(three_d_root, rr.Clear(recursive=True))
        return
    points_2d: list[list[float]] = []
    strips_2d: list[list[list[float]]] = []
    points_3d: list[list[float]] = []
    strips_3d: list[list[list[float]]] = []
    labels: list[str] = []
    colors_points: list[tuple[int, int, int]] = []
    colors_strips: list[tuple[int, int, int]] = []
    for hand in drawn:
        color = ASSEMBLY101_HAND_COLORS[hand.side]
        pixels = [[p.x, p.y] for p in hand.joints_proxy_pixels]
        world = [[p.x, p.y, p.z] for p in hand.joints_world_mm]
        points_2d.extend(pixels)
        points_3d.extend(world)
        labels.extend(f"dataset {hand.side}: {name}" for name in ASSEMBLY101_JOINT_NAMES)
        colors_points.extend([color] * len(pixels))
        strips_2d.extend([[pixels[a], pixels[b]] for a, b in ASSEMBLY101_EDGES])
        strips_3d.extend([[world[a], world[b]] for a, b in ASSEMBLY101_EDGES])
        colors_strips.extend([color] * len(ASSEMBLY101_EDGES))
    rr.log(
        f"{two_d_root}/landmarks",
        rr.Points2D(points_2d, labels=labels, colors=colors_points, radii=2.5),
    )
    rr.log(f"{two_d_root}/skeletons", rr.LineStrips2D(strips_2d, colors=colors_strips, radii=1.5))
    rr.log(
        f"{two_d_root}/wrists",
        rr.Points2D(
            [list(a101.wrist_pixels(hand)) for hand in drawn],
            labels=[
                f"dataset {hand.side} ({hand.confidence:.2f}, "
                f"{hand.joints_inside_image}/21 in frame)"
                for hand in drawn
            ],
            colors=[ASSEMBLY101_HAND_COLORS[hand.side] for hand in drawn],
            radii=5.0,
        ),
    )
    rr.log(
        f"{three_d_root}/joints",
        rr.Points3D(points_3d, labels=labels, colors=colors_points, radii=4.0),
    )
    rr.log(f"{three_d_root}/skeletons", rr.LineStrips3D(strips_3d, colors=colors_strips, radii=2.0))


def _assembly101_coverage(reference: a101.LoadedAssembly101Reference) -> dict[str, int]:
    threshold = reference.manifest.draw_confidence_threshold
    frames = reference.frames.values()
    return {
        "assembly101_hand_frames": sum(bool(f.hands) for f in frames),
        "assembly101_drawn_hand_frames": sum(
            any(h.confidence >= threshold for h in f.hands) for f in frames
        ),
        "assembly101_both_hands_drawn_frames": sum(
            sum(h.confidence >= threshold for h in f.hands) == 2 for f in frames
        ),
        "assembly101_fine_segments": len(reference.segments),
    }


def _prepare_output_root(root: Path, *, overwrite: bool) -> None:
    """Create the output root; an existing empty directory is fine, existing files are not."""
    existing = sorted(root.iterdir()) if root.is_dir() else []
    if existing and not overwrite:
        raise FileExistsError(
            f"{root} already contains {len(existing)} entries; pass --overwrite to replace the "
            "package files or choose another --output-root"
        )
    for path in existing:
        if path.name in {
            OUTPUT_NAME,
            INDEX_NAME,
            GUIDE_NAME,
            CONTACT_SHEET_NAME,
            PRESET_CHECK_NAME,
        } or (path.suffix in PACKAGE_SIDE_FILE_SUFFIXES and path.is_file()):
            path.unlink()
        else:
            raise FileExistsError(f"{root} holds an unexpected entry {path.name}; not overwriting")
    root.mkdir(parents=True, exist_ok=True)


def _reference_section(
    index: InteractionReviewIndexManifest,
    eligibility: dict[str, tuple[tuple[int, int], ...]],
    legend: str = PROVENANCE_CODE_LEGEND,
) -> str:
    if index.reference_segmentation_method != "ensemble_reference":
        return """- Corrected SAM3 stays the default segmentation display. It is a visible-surface
  comparison layer, not semantic ground truth. Geometry/disagreement triggers are review
  prompts only.
- The reference run adds one agent-selected chassis/interior correction at frame 1172
  (schedule rows tagged `selected_by: agent`, provenance `agent_authored_visual_review`); all
  frame-0 seeds and every other correction remain the earlier human-selected masks, and
  frames before 1172 are bit-identical to the previous reference.
- `contact_eligible` covers only `[0,1020)` and `[1172,1200)`. Frames `[1020,1172)` hold the
  human-reported chassis/interior swap (interior label leaks over the chassis from ~1020,
  chassis label lost from ~1110, pure label swap 1167-1234 in the old run) and frames from
  1200 keep the earlier late-degradation decision. On ineligible frames masks stay visible
  with a warning but all contact fields are `invalid_mask`."""
    counts = index.reference_provenance_counts or {}
    per_part = "\n".join(
        f"  - `{part}`: eligible {_intervals_text(eligibility.get(part, ()))}; masks "
        + ", ".join(f"{count} {name}" for name, count in counts.get(part, {}).items() if count)
        for part in TARGETS
    )
    has_hidden = any(counts.get(part, {}).get("hidden_agent_label") for part in TARGETS)
    hidden_text = (
        ", and explicit\n  empty masks over agent-labelled hidden intervals"
        if has_hidden
        else " (this policy labels\n  no hidden interval)"
    )
    hidden_claim = (
        "; hidden intervals are\n  agent visibility labels pending human confirmation"
        if has_hidden
        else ""
    )
    return f"""- The default segmentation display is the **per-target ensemble review reference**
  (`battle-build-ensemble-reference`): the SAM3 primary by default, the DAM4SAM arm's whole
  mask substituted for one target only inside explicit policy intervals when a rule fires
  (area below a fraction of the rolling median, overlap with another target, or
  discontinuity with the last accepted mask) and the substitute passes sanity{hidden_text}.
  Masks are never blended. Cross-method fallback is **not** accuracy; neither source run is
  ground truth{hidden_claim}.
- **Provenance layer.** `{review.REFERENCE_PROVENANCE_SERIES}/<part>` plots a per-part code
  ({legend}) on the time panel and
  `{review.REFERENCE_PROVENANCE_OVERLAY}/<part>` draws a magenta box around every
  DAM4SAM-sourced mask in the primary view (toggle it off in the blueprint tree). The
  per-frame navigation document names each part's provenance and eligibility.
- `contact_eligible` is now **per target**: a mask is eligible only where it is SAM3- or
  DAM4SAM-sourced, passes sanity, lies before frame 1200, and is not inside an agent-labelled
  ineligible interval. Ineligible parts yield `invalid_mask` contact rows.
{per_part}"""


def _assembly101_section(
    index: InteractionReviewIndexManifest, manifest: Assembly101ReferenceManifest | None
) -> str:
    if manifest is None:
        return ""
    rule = manifest.clock_rule
    offset_ms = rule.pose_offset_frames / rule.pose_fps * 1000
    check = manifest.projection_check
    check_text = (
        f" Our projection of the dataset 3D reproduces its shipped 2D to {check.rms_pixels:.4f} px "
        f"RMS over {check.compared_points:,} points."
        if check is not None
        else ""
    )
    return f"""
## Assembly101 dataset reference (external context, not ground truth for these methods)

- **Fine-grained GT** replaces the agent-authored substep track as the primary label
  navigation: {len(manifest.fine_segments)} dataset segments cover the window (per-frame
  document, `fine_gt_index` series, and the fine-grained table tab). Overlapping segments are
  the dataset's separate two-hand labels. The agent substep contract stays as a tab and its
  series remains logged but is no longer on the navigation panel.
- **Dataset hand poses** (`{review.ASSEMBLY101_2D_ROOT}`, `{review.ASSEMBLY101_3D_ROOT}`)
  are the recording's own 60 fps multi-view tracker output: 21 joints per hand in a
  fixed-scale hand model, world-frame millimetres, drawn only at confidence >=
  {manifest.draw_confidence_threshold}. The 2D overlay is a projection through an
  **estimated** C10379 camera (Brown model fitted to the dataset's own 2D/3D pairs; fit RMS
  {manifest.camera.fit_rms_pixels:.1e} px), not an official calibration.{check_text}
- **Clock correction.** The static C10379 video lags the pose clock by {rule.pose_offset_frames}
  pose frames (~{offset_ms:.0f} ms, ±{rule.offset_uncertainty_frames}):
  `pose_frame = {rule.proxy_start_raw_frame} + {rule.raw_frames_per_proxy_frame}·proxy_frame
  {rule.pose_offset_frames:+d}`. Earlier static/ego comparisons assumed no offset.
- **Diagnostics** (`{review.ASSEMBLY101_DIAGNOSTICS}`): per-side dataset confidence and the
  pixel distance from each dataset wrist to the nearest stabilized WiLoR wrist. Distance is a
  disagreement measure between two imperfect sources, never an error of either. Coverage:
  dataset hands in {index.coverage.get("assembly101_hand_frames", 0)}/1800 frames, both hands
  drawn in {index.coverage.get("assembly101_both_hands_drawn_frames", 0)}.
- Licence {manifest.license}; {manifest.citation}.
"""


def _comparison_section(
    candidate_arms: tuple[str, ...],
    confidence_run: str | None,
    anchors: AnchorMarks | None,
) -> str:
    if not candidate_arms and confidence_run is None and anchors is None:
        return ""
    lines = [
        "",
        "## Segmentation comparison layers (evidence beside the reference, not a second reference)",
        "",
    ]
    if candidate_arms:
        names = ", ".join(f"`{name}`" for name in candidate_arms)
        lines.append(
            f"- **Candidate arms** ({names}): each arm's four part masks are logged under "
            f"`{CANDIDATE_SEGMENTATION_ROOT}/<arm>/<part>` at every frame and its per-part mask "
            f"area under `{CANDIDATE_AREA_SERIES}/<arm>/<part>`. Every arm is one tracker's "
            "output on the same C10379 proxy; the reference above is the only mask the contact "
            "geometry uses. No arm is ground truth."
        )
    if anchors is not None:
        frames = ", ".join(str(frame) for frame in anchors.frames)
        lines.append(
            f"- **Human anchors** ({len(anchors.frames)} frames: {frames}): at each anchor frame "
            "the human's accepted decoder masks are drawn as outlines on the primary view "
            f"(`{HUMAN_ANCHOR_OUTLINES}/<part>`) and as the arm `{HUMAN_ANCHOR_ARM}` under "
            f"`{CANDIDATE_SEGMENTATION_ROOT}`; `{ANCHOR_SERIES}/anchor_frame` marks the frame on "
            f"the time panel, `{ANCHOR_SERIES}/failed_cells` counts the cells the scored run "
            f"fails (anchor IoU < 0.5, or a hidden cell with > 300 px) and `{ANCHOR_LOG}` names "
            "them. The outlines exist only on the anchor frame itself (cleared on the next "
            "frame). Anchors are review evidence on 13 frames of one view, not ground truth."
        )
    if confidence_run is not None:
        lines.append(
            f"- **Detector confidence** (`{CONFIDENCE_SERIES}/<part>/confidence`, from "
            f"`{confidence_run}`): 1 - the combined suspicion of the top-3 label-free detectors "
            "(SAM3 object score, seed-area drift, area jump), per frame and part; "
            f"`{CONFIDENCE_SERIES}/<part>/abstain` marks frames whose confidence is at or below "
            "the in-sample R>=0.8 threshold. Held out, that threshold recalls 0.44-0.64 of the "
            "anchor failures at precision 0.36-0.62, so read the series as a ranker of "
            "suspicious frames, not a calibrated gate."
        )
    return "\n".join(lines) + "\n"


def _guide(
    index: InteractionReviewIndexManifest,
    rrd_path: Path,
    eligibility: dict[str, tuple[tuple[int, int], ...]] | None = None,
    assembly101: Assembly101ReferenceManifest | None = None,
    legend: str = PROVENANCE_CODE_LEGEND,
    candidate_arms: tuple[str, ...] = (),
    confidence_run: str | None = None,
    anchors: AnchorMarks | None = None,
) -> str:
    eligibility = eligibility or {part: SEGMENTATION_CONTACT_ELIGIBLE_INTERVALS for part in TARGETS}
    display = (
        "the per-target ensemble reference (SAM3 primary with provenance-tracked DAM4SAM fallback)"
        if index.reference_segmentation_method == "ensemble_reference"
        else "corrected focused SAM3"
    )
    comparison_section = _comparison_section(candidate_arms, confidence_run, anchors)
    return f"""# First-minute interaction review (v4)

Open exactly this recording:

```bash
rerun {rrd_path.as_posix()}
```

This recording embeds one RGB asset spanning analysis frames `[0,1800)` / source
294.000–354.000 s at 30 FPS. The default panel is {display} plus the
WiLoR-primary stabilized hand layer. Raw WiLoR, MediaPipe, BoxMOT person context, and
partial Kineo NLF body context are separate/toggleable evidence layers.

## Navigation panels

- **Review guide** (bottom row) is this document, logged at `metadata/review_notes`.
- **Current GT segments (per frame)** re-renders every frame from
  `metadata/navigation/current`: the active fine-grained Assembly101 segments (when the
  dataset reference is present), the coarse Assembly101 GT segment for `[0,1800)`, the
  agent-authored substep for `[0,600)`, and the per-part segmentation contact-eligibility
  state.
- **Navigation** (right column) plots the label segment indices on the time panel so
  transitions are visible as steps.
- The bottom-right tabs hold the fine-grained and coarse GT tables, the checked-in substep
  contract JSON, and the Drop-DTW status note (no Drop-DTW alignment exists for the first
  minute).
{_assembly101_section(index, assembly101)}{comparison_section}
## Validity and claim boundaries

{_reference_section(index, eligibility, legend)}
- Frame ~370: the interior mask briefly grows into the chassis and recedes on its own (human
  observation); no correction was attempted there.
- Stabilized WiLoR (v5) preserves raw/smoothed/low_confidence_continuation/fallback/missing
  per-hand provenance. `low_confidence_continuation` marks a real WiLoR detection with
  confidence in [0.35,0.55) accepted only while it continues a lane accepted within 5 frames,
  for at most 5 consecutive frames (finger-only views); nothing is held or extrapolated.
  MediaPipe is a short-gap fallback only; its IDs are neither identities nor action labels.
  The per-frame counts of these states are plotted in the disagreement time series.
- BoxMOT is independent YOLO/BotSort person context only. Kineo is NLF-only 2D body/crop
  context with source-aligned BoxMOT fallback and <=5-frame interpolation; it is not a hand
  method (human feedback compared it with WiLoR on hands: no Kineo hand output exists in
  this pipeline), nor metric 3D, SfM, or BVH.
- The checked-in 11-step agent-authored navigation timeline covers only `[0,600)`. Coarse
  Assembly101 GT over the full minute is weak navigation context, explicitly not prediction.
  No extra fine action labels were promoted beyond visually reviewed evidence.

## Coverage

- stabilized WiLoR hand frames: {index.coverage["stabilized_wilor_hand_frames"]}/1800
- BoxMOT person frames: {index.coverage["boxmot_person_frames"]}/1800
- Kineo NLF body frames: {index.coverage["kineo_body_frames"]}/1800
- geometry triggers / episodes: {len(index.segmentation_review_triggers)} /
  {len(index.segmentation_review_episodes)}

All human decisions remain pending. Treat coordinates and derived transitions as review aids,
not accuracy or interaction assertions.
"""


def _log_static_documents(
    entity: str,
    *,
    guide: str,
    fine_contract: object,
    assembly101: Assembly101ReferenceManifest | None = None,
) -> None:
    """Log every static document the blueprint's text panels reference."""
    rr.log(
        f"{entity}/metadata/fine_grained_gt",
        rr.TextDocument(
            _fine_gt_markdown(assembly101)
            if assembly101 is not None
            else "# Fine-grained Assembly101 GT\n\nNo dataset reference window was supplied "
            "to this build; run `battle-build-assembly101-reference` first.\n",
            media_type="text/markdown",
        ),
        static=True,
    )
    rr.log(
        f"{entity}/metadata/coarse_gt",
        rr.TextDocument(_coarse_gt_markdown(), media_type="text/markdown"),
        static=True,
    )
    rr.log(
        f"{entity}/metadata/agent_substeps_first_20s",
        rr.TextDocument(fine_contract.model_dump_json(indent=2), media_type="application/json"),
        static=True,
    )
    rr.log(
        f"{entity}/metadata/drop_dtw",
        rr.TextDocument(DROP_DTW_STATUS, media_type="text/markdown"),
        static=True,
    )
    rr.log(
        f"{entity}/{review.REVIEW_NOTES}",
        rr.TextDocument(guide, media_type="text/markdown"),
        static=True,
    )
    review._log_navigation_static(entity, fine_gt=assembly101 is not None)


def _sam3_validity_intervals() -> tuple[SegmentationValidityInterval, ...]:
    """Frame-level eligibility retained for a corrected-SAM3-only reference."""
    return (
        SegmentationValidityInterval(
            start_frame=0,
            end_frame_exclusive=SWAP_ONSET_FRAME,
            state="contact_eligible",
            rationale=(
                "Human review accepted the first ~40 s apart from the chassis/interior swap; "
                "the agent narrowed the end to the measured onset of the interior mask "
                "leaking onto the chassis (interior area doubles over frames 1020-1044)."
            ),
            provenance="agent_authored_visual_review",
        ),
        SegmentationValidityInterval(
            start_frame=SWAP_ONSET_FRAME,
            end_frame_exclusive=SWAP_CORRECTION_FRAME,
            state="not_contact_eligible",
            rationale=(
                "Human-reported chassis/interior identity swap: the interior label grows over "
                "the chassis body, the chassis label collapses (<600 px from 1110) and the "
                "grey interior block is not separately visible until ~1167, so no credible "
                "correction mask exists earlier."
            ),
            provenance="human_feedback_report",
        ),
        SegmentationValidityInterval(
            start_frame=SWAP_CORRECTION_FRAME,
            end_frame_exclusive=SEGMENTATION_CONTACT_ELIGIBLE_THROUGH,
            state="contact_eligible",
            rationale=(
                "Agent-selected frame-1172 chassis/interior correction (selected_by=agent in "
                "the schedule) restored the labels; new chassis vs old interior IoU 0.926 over "
                "1172-1234 and near-zero chassis/interior overlap."
            ),
            provenance="agent_authored_visual_review",
        ),
        SegmentationValidityInterval(
            start_frame=SEGMENTATION_CONTACT_ELIGIBLE_THROUGH,
            end_frame_exclusive=FRAME_COUNT,
            state="not_contact_eligible",
            rationale=(
                "Late corrected-SAM3 overlays visibly cease to track stable part semantics "
                "(rear-body label stays on the table piece while the body is attached); masks "
                "remain comparison evidence only."
            ),
            provenance="agent_authored_visual_review",
        ),
    )


LAYERS: tuple[str, ...] = (
    "reference_masks",
    "stabilized_wilor",
    "wilor_2d",
    "mediapipe_2d",
    "wilor_3d",
    "boxmot",
    "kineo",
    "diagnostics",
    "assembly101_hands",
    "athena_hands",
    "assembly101_multiview",
)
# Cross-view part consensus from `battle-build-multiview-part-consensus`; the layer is logged
# only when that run exists (Track 2 of the multicam pass).
MULTIVIEW_CONSENSUS = multiview_consensus.OUTPUT_ROOT
# Multi-view triangulated hands from `battle-athena-hands`; the layer is skipped when absent.
# Both arms are logged when present: the eight-view MediaPipe arm under `athena_hands` and the
# three-view WiLoR arm under `athena_hands_wilor` (its own entity path, same 3D view).
ATHENA_HANDS_RUNS: tuple[tuple[str, Path], ...] = (
    (athena_hands_review.DEFAULT_LABEL, athena_hands_review.DEFAULT_OUTPUT_ROOT),
    (athena_hands_review.WILOR_LABEL, athena_hands_review.WILOR_OUTPUT_ROOT),
)


def build_first_minute_review(
    *,
    repository_root: Path,
    output_root: Path = OUTPUT_ROOT,
    overwrite: bool = False,
    reference_run: Path = FIRST_MINUTE_REFERENCE_SEGMENTATION,
    assembly101_reference: Path | None = ASSEMBLY101_REFERENCE,
    multiview_consensus_root: Path | None = MULTIVIEW_CONSENSUS,
    verify_fingerprints: bool = False,
    layers: tuple[str, ...] = LAYERS,
    timer: PhaseTimer | None = None,
    candidate_arms: dict[str, Path] | None = None,
    confidence_root: Path | None = None,
    anchor_config: Path | None = ANCHOR_CONFIG,
    anchor_masks: Path | None = ANCHOR_MASKS,
    application_id: str = APPLICATION_ID,
    recording_id: str = RECORDING_ID,
) -> Path:
    """Build a 1,800-row review package from retained, source-aligned artifacts.

    `reference_run` is the ensemble reference by default; a corrected-SAM3 run directory
    (`CORRECTED_SAM3_REFERENCE_SEGMENTATION`) keeps the earlier frame-level eligibility.
    `assembly101_reference` is the dataset window from `battle-build-assembly101-reference`;
    `None` builds without the dataset layer and fine-grained labels.  `candidate_arms` maps
    display names to further C10379 segmentation runs logged beside the reference;
    `confidence_root` is a `battle-detector-scorecard` run holding `confidence.jsonl`;
    `anchor_config` / `anchor_masks` mark the human anchor frames (and draw their masks when
    the export exists).  `application_id` / `recording_id` let another recording (the
    multiview comparison) be merged into the same store.
    """

    unknown = tuple(name for name in layers if name not in LAYERS)
    if unknown:
        raise ValueError(f"unknown review layers {unknown}; choose from {LAYERS}")
    selected = frozenset(layers)
    candidate_arms = dict(candidate_arms or {})
    for name in candidate_arms:
        parse_candidate_arm(f"{name}=x")
    timer = timer or PhaseTimer("interaction review v4", enabled=False)
    timer.start("validate")
    repository_root = repository_root.resolve()
    arms = {
        name: review._validate_run(
            review.SourceSpec(name, run_directory),
            repository_root,
            frame_count=FRAME_COUNT,
            require_every_frame=False,
            verify_fingerprints=verify_fingerprints,
        )
        for name, run_directory in candidate_arms.items()
    }
    confidence_path = (
        (repository_root / confidence_root / CONFIDENCE_NAME).resolve()
        if confidence_root is not None
        else None
    )
    confidence = load_confidence_series(confidence_path) if confidence_path is not None else None
    anchors = (
        load_anchor_marks(repository_root, anchor_config, anchor_masks)
        if anchor_config is not None and (repository_root / anchor_config).is_file()
        else None
    )
    dataset = (
        a101.load_reference(
            assembly101_reference,
            repository_root,
            frame_count=FRAME_COUNT,
            verify=verify_fingerprints,
        )
        if assembly101_reference is not None
        else None
    )
    multiview = (
        multiview_review.MultiviewLayer(repository_root, multiview_consensus_root)
        if "assembly101_multiview" in selected
        and dataset is not None
        and multiview_consensus_root is not None
        and (
            repository_root / multiview_consensus_root / multiview_consensus.MANIFEST_NAME
        ).is_file()
        else None
    )
    sources = {
        name: review._validate_run(
            spec,
            repository_root,
            frame_count=FRAME_COUNT,
            verify_fingerprints=verify_fingerprints,
        )
        for name, spec in review.FIRST_MINUTE_SOURCES.items()
    }
    reference = review._validate_run(
        review.SourceSpec("reference_segmentation", reference_run),
        repository_root,
        frame_count=FRAME_COUNT,
        verify_fingerprints=verify_fingerprints,
    )
    review._validate_shared_sources([*sources.values(), reference, *arms.values()])
    ensemble_context = _load_ensemble_context(reference, repository_root)
    reference_method = "ensemble_reference" if ensemble_context else "baseline_sam3"
    legend = provenance_legend(*ensemble_context) if ensemble_context else PROVENANCE_CODE_LEGEND
    video_path = sources["wilor"].run_directory / "input.mp4"
    frames, fps, dimensions = review._video_info(video_path)
    if (frames, fps, dimensions) != (FRAME_COUNT, ANALYSIS_FPS, review.DIMENSIONS):
        raise ValueError(
            f"expected one 1280x720 1800-frame 30-fps video, got {(frames, fps, dimensions)}"
        )
    if ensemble_context is None:
        eligibility = {part: SEGMENTATION_CONTACT_ELIGIBLE_INTERVALS for part in TARGETS}
        provenance: dict[tuple[int, str], str] = {}
        reference_artifacts: list[ArtifactFingerprint] = []
        sidecar_fingerprint = None
        provenance_counts = None
    else:
        sidecar, policy = ensemble_context
        eligibility = ensemble.contact_eligible_intervals_by_part(sidecar)
        provenance = {(r.analysis_frame_index, r.target_id): r.provenance for r in sidecar.frames}
        metadata = reference.manifest.ensemble_reference
        assert metadata is not None
        for fingerprint in (
            metadata.primary_run.manifest_fingerprint,
            metadata.primary_run.observations_fingerprint,
            metadata.fallback_run.manifest_fingerprint,
            metadata.fallback_run.observations_fingerprint,
        ):
            validate_artifact_fingerprint(fingerprint, repository_root, label="ensemble source")
        sidecar_fingerprint = _file_fingerprint(
            reference.run_directory / ensemble.PROVENANCE_NAME, repository_root
        )
        reference_artifacts = [
            metadata.policy_fingerprint,
            metadata.primary_run.manifest_fingerprint,
            metadata.primary_run.observations_fingerprint,
            metadata.fallback_run.manifest_fingerprint,
            metadata.fallback_run.observations_fingerprint,
            sidecar_fingerprint,
        ]
        provenance_counts = {
            item.target_id: dict(item.provenance_counts) for item in sidecar.summaries
        }
    timer.stop("validate")
    timer.start("geometry")
    contacts, events = review._contacts(
        sources["stabilized_wilor"],
        reference,
        dimensions,
        frame_count=FRAME_COUNT,
        segmentation_contact_eligible_by_part=eligibility,
    )
    disagreements = review._disagreements(
        sources["mediapipe"], sources["wilor"], dimensions, frame_count=FRAME_COUNT
    )
    triggers = review.segmentation_review_triggers(
        reference, None, sources["stabilized_wilor"], dimensions, frame_count=FRAME_COUNT
    )
    episodes = review.cluster_segmentation_triggers(triggers)
    moments = review.deterministic_pinned_moments(
        disagreements,
        contacts,
        events,
        sources["kineo"],
        triggers,
        episodes,
        frame_count=FRAME_COUNT,
        source_start_seconds=SOURCE_START_SECONDS,
    )
    hand_states = _hand_layer_state_counts(
        sources["stabilized_wilor"].run_directory / "hand_provenance.json"
    )
    root = (repository_root / output_root).resolve()
    rrd_path, index_path, guide_path, sheet_path = _output_paths(root)
    _prepare_output_root(root, overwrite=overwrite)
    dataset_manifest_fingerprint = (
        _file_fingerprint(dataset.run_directory / a101.MANIFEST_NAME, repository_root)
        if dataset is not None
        else None
    )
    multiview_fingerprint = (
        _file_fingerprint(multiview.root / multiview_consensus.MANIFEST_NAME, repository_root)
        if multiview is not None
        else None
    )
    arm_fingerprints = {
        name: _file_fingerprint(source.run_directory / "manifest.json", repository_root)
        for name, source in arms.items()
    }
    confidence_fingerprint = (
        _file_fingerprint(confidence_path, repository_root) if confidence_path is not None else None
    )
    anchor_fingerprints = (
        [
            _file_fingerprint(anchors.config_path, repository_root),
            *(
                [_file_fingerprint(anchors.mask_set_path, repository_root)]
                if anchors.mask_set_path is not None
                else []
            ),
        ]
        if anchors is not None
        else []
    )
    artifacts = [
        _file_fingerprint(video_path, repository_root),
        *[
            item
            for source in [*sources.values(), reference, *arms.values()]
            for item in source.artifacts
        ],
        *reference_artifacts,
        *([confidence_fingerprint] if confidence_fingerprint is not None else []),
        *anchor_fingerprints,
        *(
            [dataset_manifest_fingerprint, *dataset.manifest.input_artifacts]
            if dataset is not None and dataset_manifest_fingerprint is not None
            else []
        ),
        *([multiview_fingerprint] if multiview_fingerprint is not None else []),
    ]
    unique = tuple({(item.uri, item.sha256): item for item in artifacts}.values())
    source_fingerprint = getattr(review._metadata(reference.manifest), "source_fingerprint")
    if not isinstance(source_fingerprint, ArtifactFingerprint):
        raise ValueError("reference segmentation must declare the source fingerprint")
    validity = (
        ensemble.validity_intervals_from_sidecar(*ensemble_context)
        if ensemble_context is not None
        else _sam3_validity_intervals()
    )
    eligibility_text = (
        "; ".join(f"{part} {_intervals_text(eligibility[part])}" for part in TARGETS)
        if ensemble_context is not None
        else "[0,1020) and [1172,1200)"
    )
    index = InteractionReviewIndexManifest(
        manifest_kind="interaction_review_first_minute_v4",
        comparison_id="interaction_review_first_minute_v4",
        source_video=source_fingerprint,
        bounded_video=_file_fingerprint(video_path, repository_root),
        frame_count=FRAME_COUNT,
        analysis_fps=ANALYSIS_FPS,
        source_interval=TimeInterval(start_seconds=SOURCE_START_SECONDS, end_seconds=354.0),
        reference_segmentation_method=reference_method,  # type: ignore[arg-type]
        reference_segmentation_manifest=_file_fingerprint(
            reference.run_directory / "manifest.json", repository_root
        ),
        reference_provenance_sidecar=sidecar_fingerprint,
        reference_provenance_counts=provenance_counts,
        assembly101_reference=dataset_manifest_fingerprint,
        multiview_consensus=multiview_fingerprint,
        input_artifacts=unique,
        contact_heuristic=(
            f"On contact-eligible frames per part only ({eligibility_text}): wrist/palm or "
            "fingertip inside a mask or <=12 pixels, with 2-frame start and 3-frame end "
            "debounce. Missing or invalid masks clear."
        ),
        hand_matching_rule="Greedy same-frame nearest-wrist assignment, never identity matching.",
        coordinate_semantics=(
            "2D geometry maps normalized image coordinates to 1280x720 source pixels.",
            "WiLoR 3D is camera-relative/non-metric and separate from the 2D review.",
            *(
                (
                    "Assembly101 dataset hands are world-frame millimetres from the dataset's "
                    "tracker; their 2D is a projection through an estimated C10379 camera "
                    "with the measured +9 pose-frame static clock offset.",
                )
                if dataset is not None
                else ()
            ),
        ),
        claim_boundaries=(
            "Contact candidates and segmentation triggers are not ground-truth interaction claims.",
            "BoxMOT is person context only; Kineo is partial NLF-only body context.",
            "Agent-authored labels and coarse GT are navigation aids, not predictions.",
            *(dataset.manifest.claim_boundaries if dataset is not None else ()),
            *(multiview.manifest.claim_boundaries if multiview is not None else ()),
        ),
        coverage={
            **(_assembly101_coverage(dataset) if dataset is not None else {}),
            **(multiview_review.v4_coverage(multiview) if multiview is not None else {}),
            "reference_part_mask_frames": FRAME_COUNT,
            "stabilized_low_confidence_continuation_instances": sum(
                count
                for (_, state), count in hand_states.items()
                if state == "low_confidence_continuation"
            ),
            "stabilized_missing_frames": sum(
                count for (_, state), count in hand_states.items() if state == "missing"
            ),
            "mediapipe_hand_frames": sum(
                bool(item.hands) for item in sources["mediapipe"].observations.values()
            ),
            "wilor_hand_frames": sum(
                bool(item.hands) for item in sources["wilor"].observations.values()
            ),
            "stabilized_wilor_hand_frames": sum(
                bool(item.hands) for item in sources["stabilized_wilor"].observations.values()
            ),
            "boxmot_person_frames": sum(
                bool(item.objects) for item in sources["boxmot"].observations.values()
            ),
            "kineo_body_frames": sum(
                bool(item.nlf_body_2d) for item in sources["kineo"].observations.values()
            ),
        },
        contact_diagnostics=contacts,
        hand_disagreements=disagreements,
        contact_events=events,
        segmentation_review_triggers=triggers,
        segmentation_review_episodes=episodes,
        segmentation_validity_intervals=validity,
        pinned_moments=moments,
        logged_layers=None if selected == frozenset(LAYERS) else tuple(sorted(selected)),
        candidate_arms=arm_fingerprints or None,
        confidence_series=confidence_fingerprint,
        anchor_marks=tuple(anchor_fingerprints) or None,
        application_id=application_id,
        recording_id=recording_id,
    )
    timer.stop("geometry")
    timer.start("export")
    fine_contract = load_fine_substep_contract(repository_root / FINE_LABELS)
    init_and_save(application_id, rrd_path, recording_id=recording_id)
    entity = f"world/{reference.manifest.clip.clip_id}/interaction_review_v4"
    rr.log(f"{entity}/source/video_asset", rr.AssetVideo(path=video_path), static=True)
    rr.log(
        f"{entity}/metadata/index",
        rr.TextDocument(index.model_dump_json(indent=2), media_type="application/json"),
        static=True,
    )
    dataset_manifest = dataset.manifest if dataset is not None else None
    guide = _guide(
        index,
        rrd_path,
        eligibility,
        dataset_manifest,
        legend=legend,
        candidate_arms=tuple(arms),
        confidence_run=confidence_root.as_posix() if confidence_root is not None else None,
        anchors=anchors,
    )
    _log_static_documents(
        entity, guide=guide, fine_contract=fine_contract, assembly101=dataset_manifest
    )
    if ensemble_context is not None:
        _reference_provenance_static(entity, legend)
    for name in arms:
        _candidate_arm_static(entity, name)
    if confidence is not None and confidence_root is not None:
        _confidence_static(entity, confidence_root.name)
    if anchors is not None:
        _anchor_static(entity, anchors, with_masks=bool(anchors.masks))
    if dataset is not None and "assembly101_hands" in selected:
        _log_assembly101_static(entity, dataset.manifest)
    athena_arms: list[tuple[str, athena_hands_review.LoadedAthenaHands]] = []
    if dataset is not None and "athena_hands" in selected:
        for label, run_root in ATHENA_HANDS_RUNS:
            loaded = athena_hands_review.load_run_if_present(repository_root / run_root)
            if loaded is not None:
                athena_arms.append((label, loaded))
    for label, athena in athena_arms:
        athena_hands_review.log_static(
            entity, athena, athena_hands_review.static_rig(repository_root, athena), label=label
        )
    if multiview is not None:
        multiview_review.v4_log_static(entity, multiview)
    trigger_counts = {
        frame: sum(item.analysis_frame_index == frame for item in triggers)
        for frame in range(FRAME_COUNT)
    }
    for frame in range(FRAME_COUNT):
        time = frame / ANALYSIS_FPS
        rr.set_time("analysis_frame", sequence=frame)
        rr.set_time("analysis_time", duration=time)
        rr.set_time("source_time", duration=SOURCE_START_SECONDS + time)
        rr.log(
            f"{entity}/source/video",
            rr.VideoFrameReference(seconds=time, video_reference=f"{entity}/source/video_asset"),
        )
        if "reference_masks" in selected:
            review._log_reference_masks(
                entity, reference.observations[frame], reference.run_directory, dimensions
            )
        for name, source in arms.items():
            _log_candidate_arm_frame(entity, name, source, frame)
        if confidence is not None:
            _log_confidence_frame(entity, frame, confidence)
        if anchors is not None:
            _log_anchor_frame(entity, frame, anchors, confidence)
        if "stabilized_wilor" in selected:
            _log_hands(
                f"{entity}/primary/stabilized_wilor/render",
                sources["stabilized_wilor"].observations[frame],
                dimensions=dimensions,
                color=review.METHOD_COLORS["wilor"],
                include_3d=False,
            )
        if "wilor_2d" in selected:
            _log_hands(
                f"{entity}/comparison/wilor_2d/render",
                sources["wilor"].observations[frame],
                dimensions=dimensions,
                color=review.METHOD_COLORS["wilor"],
                include_3d=False,
            )
        if "mediapipe_2d" in selected:
            _log_hands(
                f"{entity}/comparison/mediapipe_2d/render",
                sources["mediapipe"].observations[frame],
                dimensions=dimensions,
                color=review.METHOD_COLORS["mediapipe"],
                include_3d=False,
            )
        if "wilor_3d" in selected:
            _log_hands(
                f"{entity}/contexts/wilor_camera_relative_non_metric_3d",
                sources["wilor"].observations[frame],
                dimensions=dimensions,
                color=review.METHOD_COLORS["wilor"],
                include_3d=True,
                include_2d=False,
            )
        if "boxmot" in selected:
            review._log_context_boxes(
                entity, "boxmot_worker_context", sources["boxmot"].observations[frame], dimensions
            )
        if "kineo" in selected:
            kineo_path = f"{entity}/contexts/kineo_nlf_body_context"
            review._log_context_boxes(
                entity, "kineo_nlf_body_context", sources["kineo"].observations[frame], dimensions
            )
            _log_nlf_body(
                kineo_path,
                sources["kineo"].observations[frame],
                dimensions=dimensions,
                color=review.METHOD_COLORS["kineo_nlf"],
            )
        if "diagnostics" in selected:
            review._log_diagnostics_frame(entity, frame, contacts, disagreements)
        for label, athena in athena_arms:
            athena_hands_review.log_frame(entity, athena, frame, label=label)
        if multiview is not None:
            multiview_review.v4_log_frame(entity, multiview, frame)
        if dataset is not None and "assembly101_hands" in selected:
            _log_assembly101_frame(
                entity,
                dataset.frames[frame],
                sources["stabilized_wilor"].observations[frame],
                dimensions=dimensions,
                draw_threshold=dataset.manifest.draw_confidence_threshold,
            )
        rr.log(
            f"{entity}/diagnostics/segmentation_review_trigger/count",
            rr.Scalars([trigger_counts[frame]]),
        )
        eligible_parts = tuple(
            part
            for part in TARGETS
            if review.contact_eligible_frame(frame, eligibility.get(part, ()))
        )
        eligible = bool(eligible_parts)
        rr.log(
            f"{entity}/diagnostics/segmentation_contact_eligible",
            rr.Scalars([float(eligible)]),
        )
        if ensemble_context is not None:
            _log_reference_provenance_frame(
                entity, frame, reference.observations[frame], provenance, eligibility, dimensions
            )
        for state in HAND_LAYER_STATES:
            rr.log(
                f"{entity}/diagnostics/hand_disagreement/stabilized_{state}_count",
                rr.Scalars([hand_states.get((frame, state), 0)]),
            )
        substep = substep_for_frame(fine_contract, frame) if frame < 600 else None
        if substep is not None:
            rr.log(
                f"{entity}/metadata/agent_substeps_first_20s/timeline",
                rr.TextLog(f"{substep.substep_id}: {substep.label} (agent_authored_visual_review)"),
            )
        review._log_navigation_frame(
            entity,
            frame,
            source_seconds=SOURCE_START_SECONDS + time,
            substep=substep,
            coarse_gt=review.coarse_gt_for_frame(COARSE_GT, frame),
            fine_gt=(
                a101.fine_segments_for_frame(dataset.segments, frame)
                if dataset is not None
                else None
            ),
            extra_lines=(
                (
                    "- reference masks: "
                    + (
                        "`contact_eligible`"
                        if eligible
                        else "`not_contact_eligible` (visible for comparison only; contact "
                        "fields are `invalid_mask`)"
                    ),
                )
                if ensemble_context is None
                else tuple(
                    f"- `{part}` mask: `{provenance[(frame, part)]}`, "
                    + (
                        "`contact_eligible`"
                        if part in eligible_parts
                        else "`not_contact_eligible` (contact fields are `invalid_mask`)"
                    )
                    for part in TARGETS
                )
            )
            + (multiview.navigation_lines(frame) if multiview is not None else ()),
        )
    rr.send_blueprint(
        review._blueprint(
            entity,
            dimensions,
            static_text_panels=(
                (*STATIC_TEXT_PANELS, multiview_review.V4_TEXT_PANEL)
                if multiview is not None
                else STATIC_TEXT_PANELS
            ),
            reference_provenance=ensemble_context is not None,
            assembly101=dataset is not None and "assembly101_hands" in selected,
            multiview=multiview is not None,
            provenance_panel_name=f"Reference mask provenance per part ({legend})",
            candidate_arms=(
                *arms,
                *((HUMAN_ANCHOR_ARM,) if anchors is not None and anchors.masks else ()),
            ),
            confidence=confidence is not None,
            anchors=anchors is not None,
        )
    )
    rr.disconnect()
    sheet_moments = tuple(
        item
        for item in moments
        if item.analysis_frame_index in {0, 300, 599, 900, 1199, 1200, 1584, 1637, 1799}
    )
    review._make_contact_sheet(
        video_path, sheet_path, sheet_moments, reference, sources["stabilized_wilor"]
    )
    guide_path.write_text(guide, encoding="utf-8")
    index_path.write_text(
        index.model_copy(
            update={
                "output_rrd": _file_fingerprint(rrd_path, repository_root),
                "review_guide": _file_fingerprint(guide_path, repository_root),
                "contact_sheet": _file_fingerprint(sheet_path, repository_root),
            }
        ).model_dump_json(indent=2)
        + "\n",
        encoding="utf-8",
    )
    timer.stop("export")
    timer.print_report()
    return rrd_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_repository_root(parser)
    add_output_root(parser, OUTPUT_ROOT)
    add_output_flags(
        parser,
        overwrite_help=(
            "Replace an existing package in --output-root (an empty directory never needs it)."
        ),
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=FIRST_MINUTE_REFERENCE_SEGMENTATION,
        help=(
            "Reference segmentation run directory: the ensemble reference by default, or the "
            f"corrected SAM3 run {CORRECTED_SAM3_REFERENCE_SEGMENTATION} for a SAM3-only package."
        ),
    )
    parser.add_argument(
        "--assembly101-reference",
        type=Path,
        default=ASSEMBLY101_REFERENCE,
        help=(
            "Dataset reference window from battle-build-assembly101-reference (hand poses and "
            "fine-grained labels on this proxy's clock)."
        ),
    )
    parser.add_argument(
        "--no-assembly101-reference",
        action="store_true",
        help="Build without the dataset hand-pose layer and fine-grained labels.",
    )
    parser.add_argument(
        "--multiview-consensus",
        type=Path,
        default=MULTIVIEW_CONSENSUS,
        help=(
            "Cross-view part consensus root from battle-build-multiview-part-consensus for the "
            "assembly101_multiview layer; pick the build whose reference run is this package's "
            "primary segmentation. The layer is skipped when the root has no manifest."
        ),
    )
    parser.add_argument(
        "--verify-fingerprints",
        action="store_true",
        help="Re-read every input instead of trusting a digest cached against size and mtime.",
    )
    parser.add_argument(
        "--layers",
        default=",".join(LAYERS),
        help=(
            "Comma-separated layers to log for a narrowed iteration build; anything short of "
            f"the full set is recorded as logged_layers in the index. Choose from {LAYERS}."
        ),
    )
    parser.add_argument(
        "--candidate-arm",
        action="append",
        type=parse_candidate_arm,
        default=[],
        metavar="NAME=RUN_DIR",
        help=(
            "Log another C10379 segmentation run's four part masks beside the reference under "
            f"{CANDIDATE_SEGMENTATION_ROOT}/NAME (repeatable; about 15 MB per arm)."
        ),
    )
    parser.add_argument(
        "--confidence",
        type=Path,
        default=None,
        metavar="RUN_DIR",
        help=(
            f"battle-detector-scorecard run holding {CONFIDENCE_NAME}; logs per-part confidence "
            f"and abstain series under {CONFIDENCE_SERIES}."
        ),
    )
    parser.add_argument(
        "--anchor-config",
        type=Path,
        default=ANCHOR_CONFIG,
        help="Human review anchor config whose frames are marked on the timeline.",
    )
    parser.add_argument(
        "--anchor-masks",
        type=Path,
        default=ANCHOR_MASKS,
        help="Exported anchor mask set; when present the masks are drawn on the anchor frames.",
    )
    parser.add_argument("--no-anchors", action="store_true", help="Skip the anchor marks.")
    parser.add_argument("--application-id", default=APPLICATION_ID)
    parser.add_argument(
        "--recording-id",
        default=RECORDING_ID,
        help="Recording id; share it with the multiview recording to merge both into one file.",
    )
    args = parser.parse_args()
    candidate_arms: dict[str, Path] = {}
    for name, run_directory in args.candidate_arm:
        if name in candidate_arms:
            parser.error(f"candidate arm {name!r} given twice")
        candidate_arms[name] = run_directory
    print(
        build_first_minute_review(
            repository_root=args.repository_root,
            output_root=args.output_root,
            layers=tuple(name.strip() for name in args.layers.split(",") if name.strip()),
            timer=PhaseTimer("interaction review v4", enabled=not args.quiet),
            overwrite=args.overwrite,
            reference_run=args.reference,
            assembly101_reference=(
                None if args.no_assembly101_reference else args.assembly101_reference
            ),
            multiview_consensus_root=args.multiview_consensus,
            verify_fingerprints=args.verify_fingerprints,
            candidate_arms=candidate_arms,
            confidence_root=args.confidence,
            anchor_config=None if args.no_anchors else args.anchor_config,
            anchor_masks=None if args.no_anchors else args.anchor_masks,
            application_id=args.application_id,
            recording_id=args.recording_id,
        )
    )


if __name__ == "__main__":
    main()
