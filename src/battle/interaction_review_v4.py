"""Build the first-minute, inference-free interaction review recording.

The v4 package deliberately stops geometric contact candidates at the documented late
segmentation validity boundary.  Its additional layers remain independently derived
context, never a joint tracker or an action prediction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import rerun as rr

from . import interaction_review as review
from .exploratory_comparison import _file_fingerprint, _log_hands, _log_nlf_body
from .fine_substep_contract import load_contract as load_fine_substep_contract
from .fine_substep_contract import substep_for_frame
from .four_part_contract import ANALYSIS_FPS
from .schemas import (
    ArtifactFingerprint,
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
FINE_LABELS = Path("configs/fine_substeps/assembly101_focused_static_first_20s_agent_labels.json")

# Corrected focused SAM3 rerun whose schedule adds an agent-selected chassis/interior
# correction at frame 1172 on top of the human-selected seeds and corrections. Frames before
# 1172 are bit-identical to the earlier reference run.
FIRST_MINUTE_REFERENCE_SEGMENTATION = Path(
    "runs/muggledsam-sam3-four-part-static-focused-reassembly-static-c10379-20260918t001210z"
)
# Contact geometry is measured only where the corrected masks keep stable part identities.
# The human-reported chassis/interior swap was measured to start leaking at ~1020-1032 and
# the chassis label is lost or swapped until the agent-selected correction at 1172; frames
# from 1200 remain ineligible per the retained late-degradation review.
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
    ("metadata/coarse_gt", "Coarse Assembly101 GT (weak supervision)"),
    ("metadata/agent_substeps_first_20s", "Agent-authored substeps (contract, [0,600))"),
    ("metadata/drop_dtw", "Drop-DTW status"),
)
DROP_DTW_STATUS = """# Drop-DTW weak supervision (first minute)

No Drop-DTW alignment was run for the first-minute window. The only Drop-DTW artifact is the
20 s pinned OpenCLIP alignment shown in `runs/interaction-review-first-20s`. In this package the
coarse Assembly101 GT transcript (same weak-supervision basis, no model alignment) drives the
per-frame navigation document and the coarse GT segment index time series. It is navigation
context only, never a prediction or an accuracy claim.
"""


HAND_LAYER_STATES = ("low_confidence_continuation", "fallback", "missing")


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


def _prepare_output_root(root: Path, *, overwrite: bool) -> None:
    """Create the output root; an existing empty directory is fine, existing files are not."""
    existing = sorted(root.iterdir()) if root.is_dir() else []
    if existing and not overwrite:
        raise FileExistsError(
            f"{root} already contains {len(existing)} entries; pass --overwrite to replace the "
            "package files or choose another --output-root"
        )
    for path in existing:
        if path.name in {OUTPUT_NAME, INDEX_NAME, GUIDE_NAME, CONTACT_SHEET_NAME}:
            path.unlink()
        else:
            raise FileExistsError(f"{root} holds an unexpected entry {path.name}; not overwriting")
    root.mkdir(parents=True, exist_ok=True)


def _guide(index: InteractionReviewIndexManifest, rrd_path: Path) -> str:
    return f"""# First-minute interaction review (v4)

Open exactly this recording:

```bash
rerun {rrd_path.as_posix()}
```

This recording embeds one RGB asset spanning analysis frames `[0,1800)` / source
294.000–354.000 s at 30 FPS. The default panel is corrected focused SAM3 plus the
WiLoR-primary stabilized hand layer. Raw WiLoR, MediaPipe, BoxMOT person context, and
partial Kineo NLF body context are separate/toggleable evidence layers.

## Navigation panels

- **Review guide** (bottom row) is this document, logged at `metadata/review_notes`.
- **Current substep + coarse GT (per frame)** re-renders every frame from
  `metadata/navigation/current`: the agent-authored substep for `[0,600)`, the coarse
  Assembly101 GT segment for `[0,1800)`, and the segmentation contact-eligibility state.
- **Navigation: agent substep index + coarse GT segment** (right column) plots the same two
  step indices on the time panel so transitions are visible as steps.
- The bottom-right tabs hold the static coarse GT table, the checked-in substep contract
  JSON, and the Drop-DTW status note (no Drop-DTW alignment exists for the first minute).

## Validity and claim boundaries

- Corrected SAM3 stays the default segmentation display. It is a visible-surface comparison
  layer, not semantic ground truth. Geometry/disagreement triggers are review prompts only.
- The reference run adds one agent-selected chassis/interior correction at frame 1172
  (schedule rows tagged `selected_by: agent`, provenance `agent_authored_visual_review`); all
  frame-0 seeds and every other correction remain the earlier human-selected masks, and
  frames before 1172 are bit-identical to the previous reference.
- `contact_eligible` covers only `[0,1020)` and `[1172,1200)`. Frames `[1020,1172)` hold the
  human-reported chassis/interior swap (interior label leaks over the chassis from ~1020,
  chassis label lost from ~1110, pure label swap 1167-1234 in the old run) and frames from
  1200 keep the earlier late-degradation decision. On ineligible frames masks stay visible
  with a warning but all contact fields are `invalid_mask`.
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


def _log_static_documents(entity: str, *, guide: str, fine_contract: object) -> None:
    """Log every static document the blueprint's text panels reference."""
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
    review._log_navigation_static(entity)


def build_first_minute_review(
    *, repository_root: Path, output_root: Path = OUTPUT_ROOT, overwrite: bool = False
) -> Path:
    """Build a 1,800-row review package from retained, source-aligned artifacts."""

    repository_root = repository_root.resolve()
    sources = {
        name: review._validate_run(spec, repository_root, frame_count=FRAME_COUNT)
        for name, spec in review.FIRST_MINUTE_SOURCES.items()
    }
    reference = review._validate_run(
        review.SourceSpec("baseline_sam3", FIRST_MINUTE_REFERENCE_SEGMENTATION),
        repository_root,
        frame_count=FRAME_COUNT,
    )
    review._validate_shared_sources([*sources.values(), reference])
    video_path = sources["wilor"].run_directory / "input.mp4"
    frames, fps, dimensions = review._video_info(video_path)
    if (frames, fps, dimensions) != (FRAME_COUNT, ANALYSIS_FPS, review.DIMENSIONS):
        raise ValueError(
            f"expected one 1280x720 1800-frame 30-fps video, got {(frames, fps, dimensions)}"
        )
    contacts, events = review._contacts(
        sources["stabilized_wilor"],
        reference,
        dimensions,
        frame_count=FRAME_COUNT,
        segmentation_contact_eligible_intervals=SEGMENTATION_CONTACT_ELIGIBLE_INTERVALS,
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
    artifacts = [
        _file_fingerprint(video_path, repository_root),
        *[item for source in [*sources.values(), reference] for item in source.artifacts],
    ]
    unique = tuple({(item.uri, item.sha256): item for item in artifacts}.values())
    source_fingerprint = getattr(review._metadata(reference.manifest), "source_fingerprint")
    if not isinstance(source_fingerprint, ArtifactFingerprint):
        raise ValueError("reference segmentation must declare the source fingerprint")
    validity = (
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
    index = InteractionReviewIndexManifest(
        manifest_kind="interaction_review_first_minute_v4",
        comparison_id="interaction_review_first_minute_v4",
        source_video=source_fingerprint,
        bounded_video=_file_fingerprint(video_path, repository_root),
        frame_count=FRAME_COUNT,
        analysis_fps=ANALYSIS_FPS,
        source_interval=TimeInterval(start_seconds=SOURCE_START_SECONDS, end_seconds=354.0),
        reference_segmentation_method="baseline_sam3",
        reference_segmentation_manifest=_file_fingerprint(
            reference.run_directory / "manifest.json", repository_root
        ),
        input_artifacts=unique,
        contact_heuristic=(
            "On contact-eligible frames only ([0,1020) and [1172,1200)): wrist/palm or "
            "fingertip inside a mask or <=12 pixels, with 2-frame start and 3-frame end "
            "debounce. Missing or invalid masks clear."
        ),
        hand_matching_rule="Greedy same-frame nearest-wrist assignment, never identity matching.",
        coordinate_semantics=(
            "2D geometry maps normalized image coordinates to 1280x720 source pixels.",
            "WiLoR 3D is camera-relative/non-metric and separate from the 2D review.",
        ),
        claim_boundaries=(
            "Contact candidates and segmentation triggers are not ground-truth interaction claims.",
            "BoxMOT is person context only; Kineo is partial NLF-only body context.",
            "Agent-authored labels and coarse GT are navigation aids, not predictions.",
        ),
        coverage={
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
    )
    fine_contract = load_fine_substep_contract(repository_root / FINE_LABELS)
    rr.init("battle-interaction-review-v4", recording_id="interaction_review_first_minute_v4")
    rr.save(rrd_path)
    entity = f"world/{reference.manifest.clip.clip_id}/interaction_review_v4"
    rr.log(f"{entity}/source/video_asset", rr.AssetVideo(path=video_path), static=True)
    rr.log(
        f"{entity}/metadata/index",
        rr.TextDocument(index.model_dump_json(indent=2), media_type="application/json"),
        static=True,
    )
    _log_static_documents(entity, guide=_guide(index, rrd_path), fine_contract=fine_contract)
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
        review._log_reference_masks(
            entity, reference.observations[frame], reference.run_directory, dimensions
        )
        _log_hands(
            f"{entity}/primary/stabilized_wilor/render",
            sources["stabilized_wilor"].observations[frame],
            dimensions=dimensions,
            color=review.METHOD_COLORS["wilor"],
            include_3d=False,
        )
        _log_hands(
            f"{entity}/comparison/wilor_2d/render",
            sources["wilor"].observations[frame],
            dimensions=dimensions,
            color=review.METHOD_COLORS["wilor"],
            include_3d=False,
        )
        _log_hands(
            f"{entity}/comparison/mediapipe_2d/render",
            sources["mediapipe"].observations[frame],
            dimensions=dimensions,
            color=review.METHOD_COLORS["mediapipe"],
            include_3d=False,
        )
        _log_hands(
            f"{entity}/contexts/wilor_camera_relative_non_metric_3d",
            sources["wilor"].observations[frame],
            dimensions=dimensions,
            color=review.METHOD_COLORS["wilor"],
            include_3d=True,
        )
        review._log_context_boxes(
            entity, "boxmot_worker_context", sources["boxmot"].observations[frame], dimensions
        )
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
        review._log_diagnostics_frame(entity, frame, contacts, disagreements)
        rr.log(
            f"{entity}/diagnostics/segmentation_review_trigger/count",
            rr.Scalars([trigger_counts[frame]]),
        )
        eligible = review.contact_eligible_frame(frame, SEGMENTATION_CONTACT_ELIGIBLE_INTERVALS)
        rr.log(
            f"{entity}/diagnostics/segmentation_contact_eligible",
            rr.Scalars([float(eligible)]),
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
            extra_lines=(
                "- corrected SAM3 masks: "
                + (
                    "`contact_eligible`"
                    if eligible
                    else "`not_contact_eligible` (visible for comparison only; contact fields "
                    "are `invalid_mask`)"
                ),
            ),
        )
    rr.send_blueprint(review._blueprint(entity, dimensions, static_text_panels=STATIC_TEXT_PANELS))
    rr.disconnect()
    sheet_moments = tuple(
        item
        for item in moments
        if item.analysis_frame_index in {0, 300, 599, 900, 1199, 1200, 1584, 1637, 1799}
    )
    review._make_contact_sheet(
        video_path, sheet_path, sheet_moments, reference, sources["stabilized_wilor"]
    )
    guide_path.write_text(_guide(index, rrd_path), encoding="utf-8")
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
    return rrd_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing package in --output-root (an empty directory never needs it).",
    )
    args = parser.parse_args()
    print(
        build_first_minute_review(
            repository_root=args.repository_root,
            output_root=args.output_root,
            overwrite=args.overwrite,
        )
    )


if __name__ == "__main__":
    main()
