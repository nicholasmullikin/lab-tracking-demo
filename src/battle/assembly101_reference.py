"""Resample the Assembly101 dataset hand poses and fine-grained labels onto one review window.

Reads the selectively acquired `AssemblyPoses.zip` members and fine-grained CSV rows for one
recording (ignored `data/raw/...`), maps every 30 FPS analysis frame of the focused static
proxy onto its 60 FPS pose frame with the measured per-camera clock offset, projects the
world-frame 3D joints through the estimated camera model, and writes a small typed window
under `runs/` that the review builders can consume without touching the raw archive again.

No model runs here.  The output is dataset context for review, never ground truth for the
methods compared in this repository.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from . import digest_cache
from .assembly101_pose_schemas import (
    ASSEMBLY101_CITATION,
    ASSEMBLY101_HAND_SIDES,
    ASSEMBLY101_WRIST_INDEX,
    Assembly101CameraModel,
    Assembly101ClockRule,
    Assembly101FineSegment,
    Assembly101Hand,
    Assembly101HandFrame,
    Assembly101Point2D,
    Assembly101Point3D,
    Assembly101ProjectionCheck,
    Assembly101ReferenceManifest,
)
from .assembly101_recordings import (
    DATASET_REVISION,
    RECORDING_1,
    Assembly101Recording,
    get_recording,
)
from .cli_common import add_output_root, add_repository_root
from .schemas import ArtifactFingerprint
from .schemas import fingerprint as measured_fingerprint

# Recording-1 constants, kept for every caller written before the recording registry existed.
RECORDING_ID = RECORDING_1.recording_id
POSES_ROOT = RECORDING_1.poses_root
CAMERA_ESTIMATE = Path("configs/assembly101/c10379_camera_estimate.json")
SHIPPED_2D_WINDOW = RECORDING_1.shipped_2d_window
OUTPUT_ROOT = Path(RECORDING_1.reference_root)
MANIFEST_NAME = "manifest.json"
HANDS_NAME = "hands.jsonl"

STATIC_VIEW_KEY = "C10379:rgb"
STATIC_VIDEO_NAME = "C10379_rgb.mp4"
PROXY_DIMENSIONS = (1280, 720)
FRAME_COUNT = 1800
ANALYSIS_FPS = 30
SOURCE_START_SECONDS = RECORDING_1.window_start_seconds
ANNOTATION_FPS = 30
ANNOTATION_START_FRAME = RECORDING_1.annotation_start_frame
PROXY_START_RAW_FRAME = RECORDING_1.window_start_raw_frame
DRAW_CONFIDENCE_THRESHOLD = 0.5

# Measured on this recording: the static C10379 video starts ~9 pose frames (~150 ms) after
# the pose clock; the ego HMC_21110305 video has no offset.  See the Sep 17 acquisition report.
STATIC_CLOCK_RULE = Assembly101ClockRule(
    view_key=STATIC_VIEW_KEY,
    proxy_start_raw_frame=PROXY_START_RAW_FRAME,
    raw_frames_per_proxy_frame=2,
    pose_offset_frames=9,
    pose_fps=60,
    analysis_fps=30,
    offset_uncertainty_frames=1,
    offset_evidence=(
        "Skin-mask hit rate and fingertip gradient magnitude on the raw 60 fps video both peak "
        "at +8..+9 pose frames in two of three 30 s chunks; the zoomed overlay at proxy frame "
        "800 lands on the raised hand only with +9. Static video has 55,516 frames against "
        "55,539 pose frames."
    ),
)
CLAIM_BOUNDARIES = (
    "Dataset hand poses come from Assembly101's own multi-view tracker with a fixed-scale hand "
    "model; they are external review context, not ground truth for any method here.",
    "Intrinsics are an estimate recovered from the dataset's 2D/3D landmark projection, not an "
    "official calibration file.",
    "Fine-grained segments are the dataset's human annotations at 30 FPS; the 9-frame static "
    "clock offset is below their resolution and is applied only to per-frame pose overlays.",
    "Nothing here backs an accuracy claim; CC BY-NC 4.0 attribution applies.",
)


def _fingerprint(path: Path, repository_root: Path, *, verify: bool = False) -> ArtifactFingerprint:
    if not path.is_file():
        raise FileNotFoundError(f"Assembly101 reference input is unavailable: {path}")
    return measured_fingerprint(path, repository_root, verify=verify)


def _poses_member(
    kind: str, repository_root: Path, recording: Assembly101Recording = RECORDING_1
) -> Path:
    return repository_root / recording.poses_member(kind)


def _view_key(view: str) -> str:
    return f"{view[4:]}:mono10bit" if view.startswith("HMC_") else f"{view}:rgb"


def _video_name(view: str) -> str:
    return f"{view}_mono10bit.mp4" if view.startswith("HMC_") else f"{view}_rgb.mp4"


def load_camera_estimate(path: Path) -> Assembly101CameraModel:
    return Assembly101CameraModel.model_validate_json(path.read_text(encoding="utf-8"))


def world_to_camera(
    camera: Assembly101CameraModel, camera_to_world: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Rodrigues rotation and translation for `cv2.projectPoints` from a camera-to-world pose.

    Static views carry their pose; ego views must be given this frame's pose from
    `camera_extrinsics_ego`.
    """
    if camera_to_world is None:
        if camera.camera_to_world is None:
            raise ValueError(f"{camera.view_key} has per-frame extrinsics; pass camera_to_world")
        camera_to_world = np.asarray(camera.camera_to_world, dtype=np.float64)
    extrinsic = np.linalg.inv(np.asarray(camera_to_world, dtype=np.float64))
    rotation, _ = cv2.Rodrigues(extrinsic[:3, :3])
    return rotation, extrinsic[:3, 3].copy()


def project_world_points(
    points_world_mm: np.ndarray,
    camera: Assembly101CameraModel,
    camera_to_world: np.ndarray | None = None,
) -> np.ndarray:
    """Project (N, 3) world-mm joints to raw sensor pixels of the camera's view."""
    rotation, translation = world_to_camera(camera, camera_to_world)
    projected, _ = cv2.projectPoints(
        np.asarray(points_world_mm, dtype=np.float64).reshape(-1, 1, 3),
        rotation,
        translation,
        np.asarray(camera.intrinsic_matrix, dtype=np.float64),
        np.asarray(camera.distortion, dtype=np.float64),
    )
    return projected.reshape(-1, 2)


def _proxy_scale(camera: Assembly101CameraModel, dimensions: tuple[int, int]) -> np.ndarray:
    raw_width, raw_height = camera.raw_image_size
    return np.array([dimensions[0] / raw_width, dimensions[1] / raw_height], dtype=np.float64)


def load_fine_segments(
    csv_path: Path,
    *,
    video_name: str = STATIC_VIDEO_NAME,
    annotation_start_frame: int = ANNOTATION_START_FRAME,
    frame_count: int = FRAME_COUNT,
) -> tuple[Assembly101FineSegment, ...]:
    """Fine-grained segments of one view overlapping the window, on both clocks.

    The segment set is identical across the 12 views; one view's rows are read so every
    segment appears once.  `end_frame` is treated as exclusive, matching the dataset's
    30 FPS frame-range convention.
    """
    window_end = annotation_start_frame + frame_count
    segments: list[Assembly101FineSegment] = []
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if not row["video"].endswith("/" + video_name):
                continue
            start = int(row["start_frame"])
            end = int(row["end_frame"])
            if end <= annotation_start_frame or start >= window_end:
                continue
            proxy_start = max(start, annotation_start_frame) - annotation_start_frame
            proxy_end = min(end, window_end) - annotation_start_frame
            segments.append(
                Assembly101FineSegment(
                    annotation_id=row["id"],
                    action_id=int(row["action_id"]),
                    verb=row["verb_cls"],
                    noun=row["noun_cls"],
                    action=row["action_cls"],
                    annotation_start_frame=start,
                    annotation_end_frame=end,
                    proxy_start_frame=proxy_start,
                    proxy_end_frame_exclusive=proxy_end,
                    clipped_to_window=start < annotation_start_frame or end > window_end,
                )
            )
    segments.sort(key=lambda item: (item.proxy_start_frame, item.annotation_id))
    return tuple(segments)


def fine_segments_for_frame(
    segments: tuple[Assembly101FineSegment, ...], frame: int
) -> tuple[tuple[int, Assembly101FineSegment], ...]:
    """Every (index, segment) active at `frame`; overlapping two-hand labels both appear."""
    return tuple(
        (index, segment)
        for index, segment in enumerate(segments)
        if segment.proxy_start_frame <= frame < segment.proxy_end_frame_exclusive
    )


def build_hand_frames(
    *,
    landmarks3d: dict[str, dict[str, list[list[float]]]],
    confidences: dict[str, dict[str, float]],
    timestamps: dict[str, float],
    camera: Assembly101CameraModel,
    clock_rule: Assembly101ClockRule,
    dimensions: tuple[int, int] = PROXY_DIMENSIONS,
    frame_count: int = FRAME_COUNT,
    source_start_seconds: float = SOURCE_START_SECONDS,
) -> tuple[Assembly101HandFrame, ...]:
    """One typed frame per analysis frame; hands with zero confidence are absent."""
    scale = _proxy_scale(camera, dimensions)
    width, height = dimensions
    frames: list[Assembly101HandFrame] = []
    for proxy_frame in range(frame_count):
        pose_frame = clock_rule.pose_frame(proxy_frame)
        key = str(pose_frame)
        if key not in landmarks3d:
            raise KeyError(f"pose frame {pose_frame} missing from landmarks3D")
        hands: list[Assembly101Hand] = []
        for hand_index in (0, 1):
            confidence = float(confidences[key][str(hand_index)])
            if confidence <= 0:
                continue
            world = np.asarray(landmarks3d[key][str(hand_index)], dtype=np.float64)
            if world.shape != (21, 3):
                raise ValueError(f"pose frame {pose_frame} hand {hand_index} is not 21x3")
            pixels = project_world_points(world, camera) * scale
            inside = int(
                np.count_nonzero(
                    (pixels[:, 0] >= 0)
                    & (pixels[:, 0] < width)
                    & (pixels[:, 1] >= 0)
                    & (pixels[:, 1] < height)
                )
            )
            hands.append(
                Assembly101Hand(
                    hand_index=hand_index,  # type: ignore[arg-type]
                    side=ASSEMBLY101_HAND_SIDES[hand_index],  # type: ignore[arg-type]
                    confidence=min(1.0, confidence),
                    joints_world_mm=tuple(
                        Assembly101Point3D(x=float(x), y=float(y), z=float(z)) for x, y, z in world
                    ),
                    joints_proxy_pixels=tuple(
                        Assembly101Point2D(x=float(x), y=float(y)) for x, y in pixels
                    ),
                    joints_inside_image=inside,
                )
            )
        frames.append(
            Assembly101HandFrame(
                analysis_frame_index=proxy_frame,
                pose_frame_index=pose_frame,
                pose_timestamp_seconds=float(timestamps[key]),
                source_seconds=source_start_seconds + proxy_frame / clock_rule.analysis_fps,
                hands=tuple(hands),
            )
        )
    return tuple(frames)


def projection_check(
    frames: tuple[Assembly101HandFrame, ...],
    shipped_2d_window: Path,
    *,
    camera: Assembly101CameraModel,
    dimensions: tuple[int, int],
    window_start_pose_frame: int,
    repository_root: Path,
) -> Assembly101ProjectionCheck:
    """Compare our projection against the dataset's shipped 2D landmarks for the window."""
    array_key = camera.view_key.replace(":", "_")
    with np.load(shipped_2d_window) as archive:
        if array_key not in archive.files:
            raise KeyError(f"{shipped_2d_window} has no array {array_key}")
        shipped = np.asarray(archive[array_key], dtype=np.float64)
    scale = _proxy_scale(camera, dimensions)
    residuals: list[np.ndarray] = []
    for frame in frames:
        row = frame.pose_frame_index - window_start_pose_frame
        if not 0 <= row < shipped.shape[0]:
            # A positive clock offset maps the last few proxy frames past the fetched 2D
            # window; those hands are still valid, they just cannot be checked here.
            continue
        for hand in frame.hands:
            ours = np.array([[p.x, p.y] for p in hand.joints_proxy_pixels])
            theirs = shipped[row, hand.hand_index] * scale
            residuals.append(np.linalg.norm(ours - theirs, axis=1))
    if not residuals:
        raise ValueError("no hands available for the projection check")
    stacked = np.concatenate(residuals)
    return Assembly101ProjectionCheck(
        compared_points=int(stacked.size),
        rms_pixels=float(np.sqrt(np.mean(stacked**2))),
        max_pixels=float(stacked.max()),
        shipped_2d_window=_fingerprint(shipped_2d_window, repository_root),
    )


def _coverage(frames: tuple[Assembly101HandFrame, ...]) -> dict[str, int]:
    return {
        "frames": len(frames),
        "frames_with_any_hand": sum(bool(f.hands) for f in frames),
        "frames_with_both_hands": sum(len(f.hands) == 2 for f in frames),
        "left_hand_frames": sum(any(h.side == "left" for h in f.hands) for f in frames),
        "right_hand_frames": sum(any(h.side == "right" for h in f.hands) for f in frames),
        f"hands_at_or_above_{DRAW_CONFIDENCE_THRESHOLD:.2f}": sum(
            h.confidence >= DRAW_CONFIDENCE_THRESHOLD for f in frames for h in f.hands
        ),
        "hands_fully_inside_image": sum(
            h.joints_inside_image == 21 for f in frames for h in f.hands
        ),
    }


def build_reference(
    *,
    repository_root: Path,
    output_root: Path | None = None,
    camera_estimate: Path | None = None,
    shipped_2d_window: Path | None | Literal["default"] = "default",
    overwrite: bool = False,
    recording: Assembly101Recording = RECORDING_1,
    view: str | None = None,
    clock_rule: Assembly101ClockRule | None = None,
    frame_count: int | None = None,
) -> Path:
    """Build the dataset-reference window for one static view of one recording.

    Recording 1 keeps its historical defaults (C10379, the first 1,800 frames, the checked-in
    +9 rule).  Any other recording takes its primary static view, its whole fetched window and
    the view's rule from its tracked clock-rule file unless told otherwise.
    `shipped_2d_window=None` skips the projection check.
    """
    from .assembly101_clock_offset import load_clock_rules_for

    repository_root = repository_root.resolve()
    is_recording_1 = recording.recording_id == RECORDING_1.recording_id
    view = view if view is not None else recording.primary_static_view
    if view.startswith("HMC_"):
        raise ValueError("the reference window is built for a static view (fixed extrinsics)")
    view_key_ = _view_key(view)
    output_root = output_root if output_root is not None else Path(recording.reference_root)
    if camera_estimate is None:
        camera_estimate = (
            CAMERA_ESTIMATE
            if is_recording_1 and view == "C10379"
            else Path(recording.camera_config_root) / f"{view.lower()}_camera_estimate.json"
        )
    if shipped_2d_window == "default":
        shipped_2d_window = recording.shipped_2d_window
    if frame_count is None:
        frame_count = FRAME_COUNT if is_recording_1 else recording.window_proxy_frame_count
    if clock_rule is None:
        if is_recording_1 and view == "C10379":
            clock_rule = STATIC_CLOCK_RULE
        else:
            clock_rule = load_clock_rules_for(recording, repository_root).rule(view)
    if clock_rule.view_key != view_key_:
        raise ValueError(f"clock rule is for {clock_rule.view_key}, expected {view_key_}")
    if clock_rule.proxy_start_raw_frame != recording.window_start_raw_frame:
        raise ValueError("clock rule's proxy start does not match the recording window")
    root = (repository_root / output_root).resolve()
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise FileExistsError(f"{root} is not empty; pass --overwrite")
    camera_path = repository_root / camera_estimate
    camera = load_camera_estimate(camera_path)
    if camera.view_key != view_key_:
        raise ValueError(f"camera estimate is for {camera.view_key}, expected {view_key_}")
    extrinsics_path = _poses_member("camera_extrinsics_fixed", repository_root, recording)
    shipped_extrinsics = json.loads(extrinsics_path.read_text(encoding="utf-8"))[view_key_]
    if not np.allclose(shipped_extrinsics, camera.camera_to_world, atol=1e-9):
        raise ValueError("camera estimate's camera_to_world differs from the dataset extrinsics")
    landmarks_path = _poses_member("landmarks3D", repository_root, recording)
    confidences_path = _poses_member("hand_confidences", repository_root, recording)
    timestamps_path = _poses_member("timestamp", repository_root, recording)
    csv_path = repository_root / recording.fine_grained_csv(repository_root)
    landmarks3d = json.loads(landmarks_path.read_text(encoding="utf-8"))
    confidences = json.loads(confidences_path.read_text(encoding="utf-8"))
    timestamps = json.loads(timestamps_path.read_text(encoding="utf-8"))
    frames = build_hand_frames(
        landmarks3d=landmarks3d,
        confidences=confidences,
        timestamps=timestamps,
        camera=camera,
        clock_rule=clock_rule,
        frame_count=frame_count,
        source_start_seconds=recording.window_start_seconds,
    )
    del landmarks3d
    segments = load_fine_segments(
        csv_path,
        video_name=_video_name(view),
        annotation_start_frame=recording.annotation_start_frame,
        frame_count=frame_count,
    )
    check = None
    if shipped_2d_window is not None:
        window_path = repository_root / shipped_2d_window
        if window_path.is_file():
            check = projection_check(
                frames,
                window_path,
                camera=camera,
                dimensions=PROXY_DIMENSIONS,
                window_start_pose_frame=recording.window_start_raw_frame,
                repository_root=repository_root,
            )
    root.mkdir(parents=True, exist_ok=True)
    hands_path = root / HANDS_NAME
    hands_path.write_text(
        "".join(frame.model_dump_json() + "\n" for frame in frames), encoding="utf-8"
    )
    manifest = Assembly101ReferenceManifest(
        manifest_kind="assembly101_reference_window",
        recording_id=recording.recording_id,
        dataset_revision=DATASET_REVISION,
        license="CC BY-NC 4.0",
        citation=ASSEMBLY101_CITATION,
        view_key=view_key_,
        proxy_dimensions=PROXY_DIMENSIONS,
        frame_count=frame_count,
        analysis_fps=30,
        source_start_seconds=recording.window_start_seconds,
        annotation_start_frame=recording.annotation_start_frame,
        clock_rule=clock_rule,
        camera=camera,
        draw_confidence_threshold=DRAW_CONFIDENCE_THRESHOLD,
        input_artifacts=tuple(
            _fingerprint(path, repository_root)
            for path in (
                camera_path,
                extrinsics_path,
                landmarks_path,
                confidences_path,
                timestamps_path,
                csv_path,
            )
        ),
        hands_path=HANDS_NAME,
        fine_segments=segments,
        projection_check=check,
        coverage=_coverage(frames),
        claim_boundaries=CLAIM_BOUNDARIES,
        hands_fingerprint=_fingerprint(hands_path, repository_root, verify=True),
    )
    (root / MANIFEST_NAME).write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return root


class LoadedAssembly101Reference:
    def __init__(
        self,
        run_directory: Path,
        manifest: Assembly101ReferenceManifest,
        frames: dict[int, Assembly101HandFrame],
    ) -> None:
        self.run_directory = run_directory
        self.manifest = manifest
        self.frames = frames

    @property
    def segments(self) -> tuple[Assembly101FineSegment, ...]:
        return self.manifest.fine_segments


def load_reference(
    run_directory: Path,
    repository_root: Path,
    *,
    frame_count: int = FRAME_COUNT,
    verify: bool = False,
) -> LoadedAssembly101Reference:
    """Load a derived window and refuse it if its inputs or its hands file changed."""
    repository_root = repository_root.resolve()
    run_directory = (repository_root / run_directory).resolve()
    manifest = Assembly101ReferenceManifest.model_validate_json(
        (run_directory / MANIFEST_NAME).read_text(encoding="utf-8")
    )
    if manifest.frame_count != frame_count:
        raise ValueError(f"reference window has {manifest.frame_count} frames, need {frame_count}")
    for fingerprint in (*manifest.input_artifacts, manifest.hands_fingerprint):
        if fingerprint is None:
            raise ValueError("reference manifest lacks its hands fingerprint")
        path = repository_root / fingerprint.uri
        if not path.is_file():
            raise FileNotFoundError(f"Assembly101 reference input is unavailable: {path}")
        actual = digest_cache.sha256_file(path, verify=verify)
        if actual != fingerprint.sha256:
            raise ValueError(f"Assembly101 reference fingerprint mismatch for {fingerprint.uri}")
    frames: dict[int, Assembly101HandFrame] = {}
    with (run_directory / manifest.hands_path).open(encoding="utf-8") as handle:
        for line in handle:
            frame = Assembly101HandFrame.model_validate_json(line)
            frames[frame.analysis_frame_index] = frame
    if sorted(frames) != list(range(frame_count)):
        raise ValueError("reference hands file does not cover every analysis frame exactly once")
    return LoadedAssembly101Reference(run_directory, manifest, frames)


def wrist_pixels(hand: Assembly101Hand) -> tuple[float, float]:
    point = hand.joints_proxy_pixels[ASSEMBLY101_WRIST_INDEX]
    return point.x, point.y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_repository_root(parser)
    parser.add_argument(
        "--recording",
        help="Registry label or recording id (configs/assembly101/recordings.json); "
        "default: recording 1.",
    )
    parser.add_argument(
        "--view", help="Static camera, e.g. C10379 (default: the recording's primary view)."
    )
    add_output_root(parser, None, help="default: the recording's reference_root")
    parser.add_argument(
        "--camera-estimate",
        type=Path,
        help="default: the recording's checked-in estimate for the view",
    )
    parser.add_argument(
        "--shipped-2d-window",
        type=Path,
        help="Optional npz of the dataset's own 2D landmarks for a projection residual check "
        "(default: the recording's window).",
    )
    parser.add_argument("--no-projection-check", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    recording = get_recording(args.recording, args.repository_root)
    shipped: Path | None | Literal["default"] = (
        None
        if args.no_projection_check
        else (args.shipped_2d_window if args.shipped_2d_window is not None else "default")
    )
    root = build_reference(
        repository_root=args.repository_root,
        output_root=args.output_root,
        camera_estimate=args.camera_estimate,
        shipped_2d_window=shipped,
        overwrite=args.overwrite,
        recording=recording,
        view=args.view,
    )
    manifest = Assembly101ReferenceManifest.model_validate_json(
        (root / MANIFEST_NAME).read_text(encoding="utf-8")
    )
    print(root)
    print(json.dumps(manifest.coverage, indent=2))
    if manifest.projection_check is not None:
        print(
            "projection check vs shipped 2D: rms "
            f"{manifest.projection_check.rms_pixels:.4f} px, max "
            f"{manifest.projection_check.max_pixels:.4f} px over "
            f"{manifest.projection_check.compared_points} points"
        )
    print(f"fine-grained segments in window: {len(manifest.fine_segments)}")


if __name__ == "__main__":
    main()
