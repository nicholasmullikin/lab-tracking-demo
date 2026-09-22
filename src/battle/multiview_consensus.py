"""Cross-view part consensus for the first minute: triangulate, score, flag disagreement.

For every analysis frame and part, the mask centroids of every static view that tracks the
part are triangulated on the shared rig (equal-weight DLT with the 30 raw-px reprojection
filter, at least two views).  Each view then gets a per-frame *reprojection error*: zero when
the consensus point lands inside that view's mask, otherwise the pixel distance to the mask.
Runs of frames where one view sits more than a threshold away from the consensus become
*disagreement episodes*; where the human-corrected C10379 reference is the one contradicted
by a majority of the other views, the builder emits `multiview_disagreement` triggers and
*proposed* `not_contact_eligible` intervals.  Nothing is substituted or applied: the C10379
masks stay the reference and every number here is cross-view disagreement, not accuracy.

The dataset's own wrists are triangulated the same way as a sanity anchor for the rig.

CC BY-NC 4.0 attribution applies to the Assembly101 assets behind every view.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import mask_cache
from .assembly101_camera_fit import PoseMembers
from .assembly101_clock_offset import WINDOW_START_POSE_FRAME, is_ego, npz_key
from .assembly101_pose_schemas import ASSEMBLY101_WRIST_INDEX
from .assembly101_recordings import RECORDING_1, Assembly101Recording, get_recording
from .four_part_contract import TARGETS
from .fs_common import relative_uri
from .multiview_geometry import CameraRig
from .multiview_schemas import (
    MULTIVIEW_CLAIM_BOUNDARIES,
    ConsensusEpisodeRules,
    ConsensusViewSource,
    DisagreementEpisode,
    MultiviewConsensusManifest,
    PartConsensusSummary,
    ProposedValidityInterval,
    WristTriangulationCheck,
)
from .multiview_seed_transfer import REFERENCE_RUN, REFERENCE_VIEW, proxy_to_raw_scale, view_id_for
from .schemas import FrameObservations, RunManifest, fingerprint

OUTPUT_ROOT = Path("runs/multiview-part-consensus-first-minute")
PER_FRAME_NAME = "per_frame.jsonl"
MANIFEST_NAME = "manifest.json"
FRAME_COUNT = 1800
MULTIVIEW_RUN_GLOB = "muggledsam-sam3-four-part-multiview-first-minute-*"
RULES = ConsensusEpisodeRules(
    reprojection_filter_px=30.0,
    min_views=2,
    disagreement_threshold_px=40.0,
    min_episode_frames=5,
    ego_wrist_gate_px=10.0,
    description=(
        "Per frame and part, mask centroids (raw px) of every view with the part are "
        "triangulated (DLT, drop-worst reprojection filter at 30 raw px, >= 2 views). A view's "
        "error is 0 inside its mask, else the pixel distance to the mask. An episode is >= 5 "
        "consecutive frames with error > 40 raw px while a consensus exists; it contradicts the "
        "majority when the view was dropped from a consensus formed by >= 2 other views. An ego "
        "view joins only on frames where the dataset wrists reproject within 10 raw px of the "
        "shipped 2D through its per-frame pose."
    ),
)


@dataclass
class ViewRun:
    view: str
    run_directory: Path
    manifest: RunManifest
    observations: dict[int, FrameObservations]
    targets: tuple[str, ...]
    seed_provenance: str
    cache: mask_cache.MaskCache = field(init=False)

    def __post_init__(self) -> None:
        self.cache = mask_cache.cache_for(self.run_directory)

    def mask(self, frame: int, target: str) -> np.ndarray | None:
        observation = self.observations.get(frame)
        if observation is None:
            return None
        item = next((o for o in observation.objects if o.label == target and o.mask), None)
        if item is None or item.mask is None:
            return None
        return self.cache.mask(item.mask.uri)


def load_view_run(
    repository_root: Path, run_directory: Path, *, view: str, frame_count: int = FRAME_COUNT
) -> ViewRun:
    run_directory = (repository_root / run_directory).resolve()
    manifest = RunManifest.model_validate_json(
        (run_directory / "manifest.json").read_text(encoding="utf-8")
    )
    observations: dict[int, FrameObservations] = {}
    with (run_directory / "observations.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            observation = FrameObservations.model_validate_json(line)
            if observation.analysis_frame_index < frame_count:
                observations[observation.analysis_frame_index] = observation
    if manifest.four_part_multiview is not None:
        targets = manifest.four_part_multiview.concepts
        provenance = manifest.four_part_multiview.seed_provenance
    elif manifest.four_part_focused is not None:
        targets = manifest.four_part_focused.concepts
        provenance = "human_reviewed"
    else:
        raise ValueError(f"{run_directory} is not a four-part run")
    return ViewRun(view, run_directory, manifest, observations, tuple(targets), provenance)


def has_default_tracker_policy(runtime_settings: Mapping[str, object]) -> bool:
    """True for runs made before the tracker memory policy existed or with it switched off."""
    recorded = runtime_settings.get("tracker_memory_policy")
    if recorded is None:
        return True
    policy = json.loads(recorded) if isinstance(recorded, str) else dict(recorded)
    return (
        policy.get("slot_exclusivity", "off") == "off" and policy.get("memory_gate", "off") == "off"
    )


def discover_multiview_runs(
    repository_root: Path, runs_root: Path = Path("runs")
) -> dict[str, Path]:
    """Latest successful geometry-seeded first-minute run per view."""
    found: dict[str, Path] = {}
    for directory in sorted((repository_root / runs_root).glob(MULTIVIEW_RUN_GLOB)):
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        metadata = manifest.four_part_multiview
        if metadata is None:
            continue
        core = next((s for s in manifest.method_statuses if s.stage == "objects"), None)
        if core is None or core.state.value != "succeeded":
            continue
        if not has_default_tracker_policy(metadata.runtime_settings):
            # Policy arms are compared against the default runs, never silently swapped in;
            # they reach the combiner only through an explicit --view-run.
            continue
        view = metadata.view_id.split("-", 1)[1].upper()
        if view.startswith("HMC"):
            view = "HMC_" + view[3:]
        found[view] = directory.relative_to(repository_root)
    return found


def mask_centroid_raw(mask: np.ndarray, scale: float) -> np.ndarray | None:
    moments = cv2.moments(mask.astype(np.uint8), binaryImage=True)
    if moments["m00"] <= 0:
        return None
    return (
        np.array([moments["m10"] / moments["m00"] + 0.5, moments["m01"] / moments["m00"] + 0.5])
        * scale
    )


def distance_to_mask_px(mask: np.ndarray, pixel_proxy: np.ndarray) -> float:
    """0 inside the mask, else Euclidean pixel distance to the nearest mask pixel (proxy px)."""
    height, width = mask.shape
    x, y = int(np.floor(pixel_proxy[0])), int(np.floor(pixel_proxy[1]))
    if 0 <= x < width and 0 <= y < height and mask[y, x]:
        return 0.0
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return float("nan")
    # Direct nearest-pixel search on the (sparse) mask is cheaper than a full distance transform
    # when the query point is outside, which is the only case that reaches here.
    return float(
        np.sqrt(np.min((xs + 0.5 - pixel_proxy[0]) ** 2 + (ys + 0.5 - pixel_proxy[1]) ** 2))
    )


def episodes_from_errors(
    errors: Mapping[str, np.ndarray],
    consensus_exists: np.ndarray,
    dropped: Mapping[str, np.ndarray],
    used_counts: np.ndarray,
    *,
    target: str,
    threshold_px: float,
    min_frames: int,
    min_majority: int = 2,
) -> tuple[DisagreementEpisode, ...]:
    """Runs of >= `min_frames` frames with error > threshold while a consensus exists."""
    episodes: list[DisagreementEpisode] = []
    for view, error in errors.items():
        flagged = consensus_exists & np.isfinite(error) & (error > threshold_px)
        start = None
        for frame in range(flagged.size + 1):
            active = frame < flagged.size and flagged[frame]
            if active and start is None:
                start = frame
            elif not active and start is not None:
                if frame - start >= min_frames:
                    span = slice(start, frame)
                    contradicts = bool(
                        np.all(dropped[view][span]) and np.all(used_counts[span] >= min_majority)
                    )
                    episodes.append(
                        DisagreementEpisode(
                            view=view,
                            target=target,
                            start_frame=start,
                            end_frame_exclusive=frame,
                            max_reprojection_error_px=float(np.max(error[span])),
                            mean_reprojection_error_px=float(np.mean(error[span])),
                            contradicts_majority=contradicts,
                        )
                    )
                start = None
    return tuple(episodes)


def merge_intervals(intervals: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)


def wrist_triangulation_check(
    rig: CameraRig,
    landmarks2d: Mapping[str, np.ndarray],
    members: PoseMembers,
    views: tuple[str, ...],
    frames: Iterable[int],
    *,
    reference_view: str,
    per_view_clocks: bool,
    frame_offset: int = 0,
    window_start_pose_frame: int = WINDOW_START_POSE_FRAME,
) -> WristTriangulationCheck:
    """Triangulate the dataset wrists from the shipped per-view 2D and compare with its 3D.

    `per_view_clocks=False` reads every view at the reference view's pose frame (a pure rig
    check); `True` reads each view at the pose frame its own clock rule assigns to the analysis
    frame, so the residual also carries the hand motion over the inter-camera clock spread.
    """
    residuals: list[float] = []
    used_frames = 0
    for frame in frames:
        frame = frame + frame_offset
        reference_pose = rig.pose_frame(reference_view, frame)
        key = str(reference_pose)
        if key not in members.landmarks3d:
            continue
        observations: dict[str, np.ndarray] = {}
        world_rows: list[np.ndarray] = []
        hands = [hand for hand in (0, 1) if float(members.confidences[key][str(hand)]) >= 0.8]
        if not hands:
            continue
        for view in views:
            pose = rig.pose_frame(view, frame) if per_view_clocks else reference_pose
            row = pose - window_start_pose_frame
            if row < 0 or row >= landmarks2d[view].shape[0]:
                continue
            width, height = rig.image_size(view)
            pixels = np.stack(
                [landmarks2d[view][row, hand, ASSEMBLY101_WRIST_INDEX] for hand in hands]
            ).astype(np.float64)
            outside = (
                (pixels[:, 0] < 0)
                | (pixels[:, 0] >= width)
                | (pixels[:, 1] < 0)
                | (pixels[:, 1] >= height)
            )
            pixels[outside] = np.nan
            observations[view] = pixels
        if len(observations) < 2:
            continue
        world_rows = [
            np.asarray(members.landmarks3d[key][str(hand)], dtype=np.float64)[
                ASSEMBLY101_WRIST_INDEX
            ]
            for hand in hands
        ]
        result = rig.triangulate(observations)
        valid = ~np.isnan(result.points[:, 0])
        if valid.any():
            used_frames += 1
            residuals.extend(
                np.linalg.norm(result.points[valid] - np.stack(world_rows)[valid], axis=1).tolist()
            )
    stacked = np.asarray(residuals)
    return WristTriangulationCheck(
        frames=used_frames,
        points=int(stacked.size),
        median_residual_mm=float(np.median(stacked)) if stacked.size else 0.0,
        p95_residual_mm=float(np.quantile(stacked, 0.95)) if stacked.size else 0.0,
        max_residual_mm=float(stacked.max()) if stacked.size else 0.0,
    )


def ego_pose_gate(
    rig: CameraRig,
    landmarks2d: Mapping[str, np.ndarray],
    members: PoseMembers,
    view: str,
    frames: Iterable[int],
    *,
    threshold_px: float,
    frame_offset: int = 0,
    window_start_pose_frame: int = WINDOW_START_POSE_FRAME,
    frame_count: int = FRAME_COUNT,
) -> np.ndarray:
    """Frames where the dataset wrists, projected through the ego pose, match the shipped 2D."""
    ok = np.zeros(frame_count, dtype=bool)
    width, height = rig.image_size(view)
    for frame in frames:
        pose = rig.pose_frame(view, frame + frame_offset)
        key = str(pose)
        row = pose - window_start_pose_frame
        if key not in members.landmarks3d or pose not in rig.ego_poses or row < 0:
            continue
        errors: list[float] = []
        for hand in (0, 1):
            if float(members.confidences[key][str(hand)]) < 0.5:
                continue
            shipped = landmarks2d[view][row, hand, ASSEMBLY101_WRIST_INDEX]
            if not (0 <= shipped[0] < width and 0 <= shipped[1] < height):
                continue
            world = np.asarray(members.landmarks3d[key][str(hand)], dtype=np.float64)[
                ASSEMBLY101_WRIST_INDEX
            ]
            ours = rig.project(view, world.reshape(1, 3), pose)[0]
            errors.append(float(np.linalg.norm(ours - shipped)))
        ok[frame] = bool(errors) and max(errors) < threshold_px
    return ok


def build_consensus(
    repository_root: Path,
    *,
    output_root: Path = OUTPUT_ROOT,
    reference_run: Path = REFERENCE_RUN,
    view_runs: Mapping[str, Path] | None = None,
    ego_views: tuple[str, ...] = (),
    frame_count: int = FRAME_COUNT,
    rules: ConsensusEpisodeRules = RULES,
    exclude_views: tuple[str, ...] = (),
    recording: Assembly101Recording | None = None,
    analysis_frame_offset: int = 0,
) -> MultiviewConsensusManifest:
    """Build the consensus; `exclude_views` leaves views (even the reference) out entirely.

    Excluding the reference view gives a consensus formed by the other cameras only, so a
    re-prompt of that view is not scored against a consensus its own masks helped form.
    `recording` selects another recording's rig, clock rules and poses (default recording 1);
    `analysis_frame_offset` is the proxy frame the runs' frame 0 corresponds to when they ran
    on a proxy trimmed from a later frame (pose lookups add it; mask indices do not).
    """
    started = time.monotonic()
    repository_root = repository_root.resolve()
    recording = recording or RECORDING_1
    rig = CameraRig.load(repository_root, recording=recording)
    members = PoseMembers(repository_root, recording)
    window_start_pose_frame = recording.window_start_raw_frame
    with np.load(repository_root / recording.shipped_2d_window) as archive:
        landmarks2d = {
            view: np.asarray(archive[npz_key(view)], dtype=np.float64) for view in rig.views
        }
    runs: dict[str, ViewRun] = {}
    if REFERENCE_VIEW not in exclude_views:
        runs[REFERENCE_VIEW] = load_view_run(
            repository_root, reference_run, view=REFERENCE_VIEW, frame_count=frame_count
        )
    if view_runs is None and recording.recording_id != RECORDING_1.recording_id:
        raise ValueError("another recording's consensus needs explicit --view-run entries")
    for view, directory in (view_runs or discover_multiview_runs(repository_root)).items():
        if view in exclude_views:
            continue
        # Ego runs join only when asked for (moving cameras behind the wrist gate); the
        # canonical consensus is static-only.
        if is_ego(view) and view not in ego_views:
            continue
        runs[view] = load_view_run(repository_root, directory, view=view, frame_count=frame_count)
    views = tuple(runs)
    static_views = tuple(v for v in views if not is_ego(v))
    gates = {
        view: ego_pose_gate(
            rig,
            landmarks2d,
            members,
            view,
            range(frame_count),
            threshold_px=rules.ego_wrist_gate_px or 10.0,
            frame_offset=analysis_frame_offset,
            window_start_pose_frame=window_start_pose_frame,
            frame_count=frame_count,
        )
        for view in views
        if is_ego(view) and view in ego_views
    }
    frames = range(frame_count)

    # Per part: centroids (raw px) per view per frame, masks kept only per frame.
    consensus = {target: np.full((frame_count, 3), np.nan) for target in TARGETS}
    used_counts = {target: np.zeros(frame_count, dtype=np.int64) for target in TARGETS}
    errors = {target: {view: np.full(frame_count, np.nan) for view in views} for target in TARGETS}
    used = {
        target: {view: np.zeros(frame_count, dtype=bool) for view in views} for target in TARGETS
    }
    observed = {
        target: {view: np.zeros(frame_count, dtype=bool) for view in views} for target in TARGETS
    }
    per_frame_records: list[dict] = []
    for frame in frames:
        record: dict = {"analysis_frame_index": frame, "parts": {}}
        for target in TARGETS:
            centroids: dict[str, np.ndarray] = {}
            masks: dict[str, np.ndarray] = {}
            poses: dict[str, int] = {}
            for view, run in runs.items():
                if target not in run.targets:
                    continue
                if is_ego(view):
                    if view not in gates or not gates[view][frame]:
                        continue
                    poses[view] = rig.pose_frame(view, frame + analysis_frame_offset)
                mask = run.mask(frame, target)
                if mask is None:
                    continue
                centroid = mask_centroid_raw(mask, proxy_to_raw_scale(view))
                if centroid is None:
                    continue
                centroids[view] = centroid
                masks[view] = mask
                observed[target][view][frame] = True
            if len(centroids) < rules.min_views:
                record["parts"][target] = {"consensus_world_mm": None, "views_used": []}
                continue
            result = rig.triangulate(
                {view: centroids[view].reshape(1, 2) for view in centroids},
                pose_frame=poses or None,
                reproj_filter_px=rules.reprojection_filter_px,
                min_views=rules.min_views,
            )
            point = result.points[0]
            if np.isnan(point).any():
                record["parts"][target] = {"consensus_world_mm": None, "views_used": []}
                continue
            consensus[target][frame] = point
            used_views = [view for index, view in enumerate(result.views) if result.used[0, index]]
            used_counts[target][frame] = len(used_views)
            view_errors: dict[str, float] = {}
            for view in centroids:
                projected_raw = rig.project(view, point.reshape(1, 3), poses.get(view))[0]
                projected_proxy = projected_raw / proxy_to_raw_scale(view)
                distance = distance_to_mask_px(masks[view], projected_proxy) * proxy_to_raw_scale(
                    view
                )
                errors[target][view][frame] = distance
                used[target][view][frame] = view in used_views
                view_errors[view] = round(float(distance), 2)
            record["parts"][target] = {
                "consensus_world_mm": [round(float(v), 2) for v in point],
                "views_used": used_views,
                "view_error_px": view_errors,
                "height_above_table_mm": None,
            }
        per_frame_records.append(record)

    # Episodes, summaries, proposals.
    all_episodes: list[DisagreementEpisode] = []
    summaries: list[PartConsensusSummary] = []
    proposals: list[ProposedValidityInterval] = []
    contradiction_intervals: list[tuple[str, int, int]] = []
    for target in TARGETS:
        exists = ~np.isnan(consensus[target][:, 0])
        dropped = {view: observed[target][view] & ~used[target][view] for view in views}
        episodes = episodes_from_errors(
            errors[target],
            exists,
            dropped,
            used_counts[target],
            target=target,
            threshold_px=rules.disagreement_threshold_px,
            min_frames=rules.min_episode_frames,
            min_majority=rules.min_views,
        )
        all_episodes.extend(episodes)
        reference_contradicted = merge_intervals(
            (e.start_frame, e.end_frame_exclusive)
            for e in episodes
            if e.view == REFERENCE_VIEW and e.contradicts_majority
        )
        for start, end in reference_contradicted:
            contradiction_intervals.append((target, start, end))
            proposals.append(
                ProposedValidityInterval(
                    target=target,
                    start_frame=start,
                    end_frame_exclusive=end,
                    proposed_state="not_contact_eligible",
                    trigger="multiview_disagreement",
                    rationale=(
                        f"C10379 {target} mask sits > {rules.disagreement_threshold_px:.0f} raw px "
                        f"from a consensus formed by >= {rules.min_views} other static views for "
                        f"{end - start} consecutive frames; proposal only, not applied"
                    ),
                )
            )
        per_view_error = {}
        per_view_agreement = {}
        per_view_frames = {}
        for view in views:
            valid = exists & np.isfinite(errors[target][view])
            per_view_frames[view] = int(observed[target][view].sum())
            if valid.any():
                per_view_error[view] = float(np.mean(errors[target][view][valid]))
                per_view_agreement[view] = float(
                    np.mean(errors[target][view][valid] <= rules.disagreement_threshold_px)
                )
        summaries.append(
            PartConsensusSummary(
                target=target,
                frames_with_consensus=int(exists.sum()),
                mean_views_used=float(used_counts[target][exists].mean()) if exists.any() else 0.0,
                per_view_mean_error_px=per_view_error,
                per_view_agreement=per_view_agreement,
                per_view_frames_observed=per_view_frames,
                episodes=len(episodes),
                reference_contradicted_frames=int(sum(e - s for s, e in reference_contradicted)),
            )
        )

    wrists = wrist_triangulation_check(
        rig,
        landmarks2d,
        members,
        static_views,
        range(0, frame_count, 5),
        reference_view=REFERENCE_VIEW,
        per_view_clocks=False,
        frame_offset=analysis_frame_offset,
        window_start_pose_frame=window_start_pose_frame,
    )
    wrists_per_view_clock = wrist_triangulation_check(
        rig,
        landmarks2d,
        members,
        static_views,
        range(0, frame_count, 5),
        reference_view=REFERENCE_VIEW,
        per_view_clocks=True,
        frame_offset=analysis_frame_offset,
        window_start_pose_frame=window_start_pose_frame,
    )

    root = repository_root / output_root
    root.mkdir(parents=True, exist_ok=True)
    per_frame_path = root / PER_FRAME_NAME
    with per_frame_path.open("w", encoding="utf-8") as handle:
        for record in per_frame_records:
            handle.write(json.dumps(record) + "\n")
    np.savez_compressed(
        root / "consensus_points.npz",
        **{f"consensus/{t}": consensus[t] for t in TARGETS},
        **{f"used_counts/{t}": used_counts[t] for t in TARGETS},
        **{f"error/{t}/{v}": errors[t][v] for t in TARGETS for v in views},
        **{f"used/{t}/{v}": used[t][v] for t in TARGETS for v in views},
        **{f"observed/{t}/{v}": observed[t][v] for t in TARGETS for v in views},
        views=np.array(views),
    )
    sources = tuple(
        ConsensusViewSource(
            view=view,
            view_id=view_id_for(view),
            run_directory_uri=relative_uri(run.run_directory, repository_root),
            manifest=fingerprint(run.run_directory / "manifest.json", repository_root),
            observations=fingerprint(run.run_directory / "observations.jsonl", repository_root),
            is_ego=is_ego(view),
            seed_provenance=run.seed_provenance,  # type: ignore[arg-type]
            targets=run.targets,
            frame_count=len(run.observations),
            proxy_to_raw_scale=proxy_to_raw_scale(view),
        )
        for view, run in runs.items()
    )
    manifest = MultiviewConsensusManifest(
        manifest_kind="multiview_part_consensus",
        frame_count=frame_count,
        analysis_fps=30,
        reference_view=REFERENCE_VIEW,
        targets=TARGETS,
        sources=sources,
        rules=rules,
        per_frame_uri=relative_uri(per_frame_path, repository_root),
        per_frame_fingerprint=fingerprint(per_frame_path, repository_root),
        summaries=tuple(summaries),
        episodes=tuple(all_episodes),
        reference_contradiction_intervals=tuple(contradiction_intervals),
        proposed_validity_intervals=tuple(proposals),
        wrist_triangulation=wrists,
        ego_pose_gate={view: int(gate.sum()) for view, gate in gates.items()} or None,
        runtime_seconds=time.monotonic() - started,
        claim_boundaries=(
            *MULTIVIEW_CLAIM_BOUNDARIES,
            "Wrist check: same-pose-frame triangulation of the shipped per-view 2D "
            f"(median {wrists.median_residual_mm:.4f} mm) verifies the rig; with each view "
            "read on its own clock rule the median is "
            f"{wrists_per_view_clock.median_residual_mm:.2f} mm and reflects hand motion over "
            "the inter-camera clock spread, not rig error.",
            "The proposed not_contact_eligible intervals are proposals for human review; the "
            "reference masks are not modified.",
        ),
    )
    (root / MANIFEST_NAME).write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (root / "wrist_check_per_view_clocks.json").write_text(
        wrists_per_view_clock.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def load_consensus(path: Path) -> MultiviewConsensusManifest:
    return MultiviewConsensusManifest.model_validate_json(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--reference-run", type=Path, default=REFERENCE_RUN)
    parser.add_argument(
        "--view-run",
        action="append",
        default=[],
        metavar="VIEW=RUN_DIR",
        help="Explicit run per view; default discovers the latest multiview run per view.",
    )
    parser.add_argument(
        "--ego-view",
        action="append",
        default=[],
        help="Ego view (e.g. HMC_21179183) to add as a moving camera behind the wrist gate.",
    )
    parser.add_argument("--frame-count", type=int, default=FRAME_COUNT)
    parser.add_argument(
        "--exclude-view",
        action="append",
        default=[],
        metavar="VIEW",
        help=(
            "Leave this view out of the consensus entirely (repeatable). Excluding C10379 "
            "builds a consensus from the other cameras only, for re-prompting that view."
        ),
    )
    parser.add_argument(
        "--recording",
        default=None,
        help="registry label of the recording when it is not recording 1 (rig, clocks, poses)",
    )
    parser.add_argument(
        "--analysis-frame-offset",
        type=int,
        default=0,
        help="proxy frame that the runs' frame 0 shows (runs on a proxy trimmed from that frame)",
    )
    args = parser.parse_args()
    view_runs = (
        {item.split("=", 1)[0]: Path(item.split("=", 1)[1]) for item in args.view_run}
        if args.view_run
        else None
    )
    manifest = build_consensus(
        args.repository_root,
        output_root=args.output_root,
        reference_run=args.reference_run,
        view_runs=view_runs,
        ego_views=tuple(args.ego_view),
        frame_count=args.frame_count,
        exclude_views=tuple(args.exclude_view),
        recording=get_recording(args.recording, args.repository_root) if args.recording else None,
        analysis_frame_offset=args.analysis_frame_offset,
    )
    for summary in manifest.summaries:
        print(
            f"{summary.target}: consensus on {summary.frames_with_consensus} frames "
            f"(mean {summary.mean_views_used:.2f} views), {summary.episodes} episodes, "
            f"C10379 contradicted {summary.reference_contradicted_frames} frames; agreement "
            + ", ".join(f"{v} {a:.2f}" for v, a in summary.per_view_agreement.items())
        )
    for target, start, end in manifest.reference_contradiction_intervals:
        print(f"  C10379 {target} contradicted [{start},{end})")
    w = manifest.wrist_triangulation
    print(
        f"wrists (same pose frame): median {w.median_residual_mm:.4f} mm, p95 "
        f"{w.p95_residual_mm:.4f}, max {w.max_residual_mm:.3f} over {w.points} points"
    )
    print(f"{manifest.runtime_seconds:.1f} s -> {args.output_root}")


if __name__ == "__main__":
    main()
