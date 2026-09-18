"""Run a bounded, provenance-checked MediaPipe Hand Landmarker baseline."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import subprocess
import time
from collections import Counter, deque
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2

from .exporter import export_run
from .schemas import (
    AdapterMetadata,
    ArtifactFingerprint,
    ChunkContinuityPolicy,
    ClockName,
    EncodedAssetInput,
    FrameObservations,
    FrameRange,
    FullDurationCoverage,
    G2PreprocessingManifest,
    HandSide,
    MediaPipeHandsRunMetadata,
    MethodState,
    MethodStatus,
    NormalizedBox,
    NormalizedPoint,
    PerFrameHand,
    RunManifest,
    RuntimeMeasurements,
    TimeInterval,
    VideoProxy,
)

DEFAULT_CONFIG = Path("configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_g2.json")
DEFAULT_MODEL = Path("models/mediapipe/hand_landmarker_float16_v1.task")
DEFAULT_MODEL_URI = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
DEFAULT_MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
DEFAULT_VIEW_ID = "static-c10379"
DEFAULT_SECONDS = 20.0
MAX_SECONDS = 60.0
HANDEDNESS_VOTE_FRAMES = 15
TRACK_MAX_GAP_FRAMES = 15
TRACK_MAX_WRIST_DISTANCE = 0.2
HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)
NormalizedRoi = tuple[float, float, float, float]
HandCandidate = tuple[tuple[NormalizedPoint, ...], HandSide, float]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_uri(path: Path, repository_root: Path) -> str:
    return path.resolve().relative_to(repository_root.resolve()).as_posix()


def _side(value: str) -> HandSide:
    normalized = value.casefold()
    if normalized == "left":
        return HandSide.LEFT
    if normalized == "right":
        return HandSide.RIGHT
    return HandSide.UNKNOWN


def anatomical_side(model_side: HandSide, *, input_mirrored: bool) -> HandSide:
    """Correct MediaPipe's selfie-oriented handedness for an unmirrored camera."""
    if input_mirrored or model_side is HandSide.UNKNOWN:
        return model_side
    return HandSide.RIGHT if model_side is HandSide.LEFT else HandSide.LEFT


def parse_roi(value: str) -> NormalizedRoi:
    try:
        roi = tuple(float(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("ROI must contain four comma-separated numbers") from error
    if len(roi) != 4:
        raise argparse.ArgumentTypeError("ROI must be x,y,width,height")
    x, y, width, height = roi
    if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > 1 or y + height > 1:
        raise argparse.ArgumentTypeError("ROI must be a positive rectangle inside [0, 1]²")
    return x, y, width, height


def remap_landmarks(
    landmarks: tuple[NormalizedPoint, ...], roi: NormalizedRoi | None
) -> tuple[NormalizedPoint, ...]:
    if roi is None:
        return landmarks
    x, y, width, height = roi
    return tuple(
        NormalizedPoint(x=x + point.x * width, y=y + point.y * height) for point in landmarks
    )


def _prepare_inference_frame(
    rgb: Any, *, roi: NormalizedRoi | None, upscale: float
) -> tuple[Any, NormalizedRoi | None]:
    if roi is None:
        return rgb, None
    frame_height, frame_width = rgb.shape[:2]
    x, y, width, height = roi
    x1 = round(x * frame_width)
    y1 = round(y * frame_height)
    x2 = round((x + width) * frame_width)
    y2 = round((y + height) * frame_height)
    crop = rgb[y1:y2, x1:x2]
    if upscale != 1:
        crop = cv2.resize(
            crop,
            None,
            fx=upscale,
            fy=upscale,
            interpolation=cv2.INTER_CUBIC,
        )
    actual_roi = (
        x1 / frame_width,
        y1 / frame_height,
        (x2 - x1) / frame_width,
        (y2 - y1) / frame_height,
    )
    return crop, actual_roi


def _landmark_box(landmarks: tuple[NormalizedPoint, ...]) -> NormalizedBox:
    x1 = min(point.x for point in landmarks)
    y1 = min(point.y for point in landmarks)
    x2 = max(point.x for point in landmarks)
    y2 = max(point.y for point in landmarks)
    epsilon = 1e-6
    x1 = min(x1, 1.0 - epsilon)
    y1 = min(y1, 1.0 - epsilon)
    return NormalizedBox(
        x=x1,
        y=y1,
        width=min(max(x2 - x1, epsilon), 1.0 - x1),
        height=min(max(y2 - y1, epsilon), 1.0 - y1),
    )


def _box_iou(first: NormalizedBox, second: NormalizedBox) -> float:
    intersection_width = max(
        0.0, min(first.x + first.width, second.x + second.width) - max(first.x, second.x)
    )
    intersection_height = max(
        0.0, min(first.y + first.height, second.y + second.height) - max(first.y, second.y)
    )
    intersection = intersection_width * intersection_height
    union = first.width * first.height + second.width * second.height - intersection
    return intersection / union if union else 0.0


def fuse_hand_candidates(
    primary: tuple[HandCandidate, ...],
    supplemental: tuple[HandCandidate, ...],
) -> tuple[HandCandidate, ...]:
    """Preserve primary detections and fill missing slots from a second inference region."""
    selected: list[HandCandidate] = []
    for candidate in primary:
        points, _, _ = candidate
        wrist = points[0]
        box = _landmark_box(points)
        if any(
            (wrist.x - existing[0][0].x) ** 2 + (wrist.y - existing[0][0].y) ** 2 <= 0.06**2
            or _box_iou(box, _landmark_box(existing[0])) >= 0.8
            for existing in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) == 2:
            break
    if len(selected) == 2:
        return tuple(selected)
    for candidate in sorted(supplemental, key=lambda item: item[2], reverse=True):
        points, _, _ = candidate
        wrist = points[0]
        box = _landmark_box(points)
        if any(
            (wrist.x - existing[0][0].x) ** 2 + (wrist.y - existing[0][0].y) ** 2 <= 0.12**2
            or _box_iou(box, _landmark_box(existing[0])) >= 0.5
            for existing in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= 2:
            break
    return tuple(selected)


def canonical_hand_ids(
    assignments: tuple[tuple[str, HandSide], ...],
) -> tuple[tuple[str, HandSide], ...]:
    """Expose frame-local detection IDs without claiming identity through occlusion."""
    return tuple(
        (f"hand-detection-{index}", side) for index, (_, side) in enumerate(assignments, start=1)
    )


@dataclass
class _HandTrack:
    track_id: int
    wrist: tuple[float, float]
    last_frame: int
    side_votes: deque[HandSide] = field(
        default_factory=lambda: deque(maxlen=HANDEDNESS_VOTE_FRAMES)
    )

    def update(self, *, wrist: tuple[float, float], frame_index: int, side: HandSide) -> HandSide:
        self.wrist = wrist
        self.last_frame = frame_index
        self.side_votes.append(side)
        counts = Counter(self.side_votes)
        best_count = max(counts.values())
        tied = {candidate for candidate, count in counts.items() if count == best_count}
        return next(candidate for candidate in reversed(self.side_votes) if candidate in tied)


class HandTrackAssigner:
    """Attach deterministic short-lived IDs and temporally voted handedness."""

    def __init__(self) -> None:
        self._tracks: dict[int, _HandTrack] = {}
        self._next_track_id = 1

    def assign(
        self,
        *,
        frame_index: int,
        wrists: tuple[tuple[float, float], ...],
        sides: tuple[HandSide, ...],
    ) -> tuple[tuple[str, HandSide], ...]:
        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if frame_index - track.last_frame <= TRACK_MAX_GAP_FRAMES
        }
        available = set(self._tracks)
        assignments: list[tuple[str, HandSide]] = []
        for wrist, side in zip(wrists, sides, strict=True):
            candidates = sorted(
                (
                    ((wrist[0] - track.wrist[0]) ** 2 + (wrist[1] - track.wrist[1]) ** 2, track_id)
                    for track_id, track in self._tracks.items()
                    if track_id in available
                )
            )
            track_id = (
                candidates[0][1]
                if candidates and candidates[0][0] <= TRACK_MAX_WRIST_DISTANCE**2
                else self._next_track_id
            )
            if track_id == self._next_track_id:
                self._next_track_id += 1
                self._tracks[track_id] = _HandTrack(
                    track_id=track_id, wrist=wrist, last_frame=frame_index
                )
            available.discard(track_id)
            voted_side = self._tracks[track_id].update(
                wrist=wrist, frame_index=frame_index, side=side
            )
            assignments.append((f"hand-{track_id}", voted_side))
        return tuple(assignments)


def _verify_inputs(
    *,
    repository_root: Path,
    config_path: Path,
    model_path: Path,
    view_id: str,
    seconds: float,
) -> tuple[G2PreprocessingManifest, VideoProxy, Path, int]:
    if not 0 < seconds <= MAX_SECONDS:
        raise ValueError(f"seconds must be in (0, {MAX_SECONDS}]")
    config = G2PreprocessingManifest.model_validate_json(config_path.read_text())
    proxies = {proxy.view_id: proxy for proxy in config.proxies}
    if view_id != DEFAULT_VIEW_ID or view_id not in proxies:
        raise ValueError("MediaPipe primary baseline requires the approved static-c10379 view")
    proxy = proxies[view_id]
    requested_frames = round(seconds * proxy.fps)
    if requested_frames > proxy.frame_count:
        raise ValueError("requested run exceeds the approved proxy")
    proxy_path = repository_root / proxy.proxy_uri
    if sha256_file(proxy_path) != proxy.checksum_sha256:
        raise ValueError("proxy checksum differs from the approved G2 configuration")
    if sha256_file(model_path) != DEFAULT_MODEL_SHA256:
        raise ValueError("Hand Landmarker model differs from the pinned float16 v1 asset")
    return config, proxy, proxy_path, requested_frames


def _detect(
    *,
    proxy_path: Path,
    model_path: Path,
    view_id: str,
    frame_count: int,
    fps: int,
    source_offset_seconds: float,
    input_mirrored: bool,
    roi: NormalizedRoi | None,
    roi_upscale: float,
    include_full_frame: bool,
    min_detection_confidence: float,
    min_presence_confidence: float,
    min_tracking_confidence: float,
) -> tuple[tuple[FrameObservations, ...], RuntimeMeasurements]:
    import mediapipe as mp  # Deferred: its audio tasks pull in sounddevice.

    start = time.perf_counter()
    first_output_seconds: float | None = None
    observations: list[FrameObservations] = []
    assigner = HandTrackAssigner()
    capture = cv2.VideoCapture(str(proxy_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open proxy video: {proxy_path}")
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=min_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    inference_regions: tuple[NormalizedRoi | None, ...] = (
        (None, roi) if roi is not None and include_full_frame else (roi,)
    )
    try:
        with ExitStack() as stack:
            landmarkers = tuple(
                stack.enter_context(mp.tasks.vision.HandLandmarker.create_from_options(options))
                for _ in inference_regions
            )
            for frame_index in range(frame_count):
                ok, bgr = capture.read()
                if not ok:
                    raise RuntimeError(f"video decode ended before frame {frame_index}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                candidate_sets: list[tuple[HandCandidate, ...]] = []
                for landmarker, inference_roi in zip(landmarkers, inference_regions, strict=True):
                    inference_frame, actual_roi = _prepare_inference_frame(
                        rgb, roi=inference_roi, upscale=roi_upscale
                    )
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=inference_frame)
                    result = landmarker.detect_for_video(image, round(frame_index * 1000 / fps))
                    point_sets = tuple(
                        remap_landmarks(
                            tuple(
                                NormalizedPoint(
                                    x=min(max(float(point.x), 0.0), 1.0),
                                    y=min(max(float(point.y), 0.0), 1.0),
                                )
                                for point in landmarks
                            ),
                            actual_roi,
                        )
                        for landmarks in result.hand_landmarks
                    )
                    model_sides = tuple(
                        _side(categories[0].category_name) if categories else HandSide.UNKNOWN
                        for categories in result.handedness
                    )
                    confidences = tuple(
                        float(categories[0].score) if categories else 0.0
                        for categories in result.handedness
                    )
                    candidate_sets.append(
                        tuple(zip(point_sets, model_sides, confidences, strict=True))
                    )
                fused = fuse_hand_candidates(
                    candidate_sets[0],
                    tuple(
                        candidate
                        for candidate_set in candidate_sets[1:]
                        for candidate in candidate_set
                    ),
                )
                point_sets = tuple(candidate[0] for candidate in fused)
                model_sides = tuple(candidate[1] for candidate in fused)
                confidences = tuple(candidate[2] for candidate in fused)
                corrected_sides = tuple(
                    anatomical_side(side, input_mirrored=input_mirrored) for side in model_sides
                )
                assigned = canonical_hand_ids(
                    assigner.assign(
                        frame_index=frame_index,
                        wrists=tuple((points[0].x, points[0].y) for points in point_sets),
                        sides=corrected_sides,
                    )
                )
                hands = tuple(
                    PerFrameHand(
                        hand_id=hand_id,
                        side=voted_side,
                        confidence=confidence,
                        landmarks=points,
                        box=_landmark_box(points),
                        model_side=model_side,
                        model_handedness_confidence=confidence,
                    )
                    for points, model_side, confidence, (hand_id, voted_side) in zip(
                        point_sets, model_sides, confidences, assigned, strict=True
                    )
                )
                observations.append(
                    FrameObservations(
                        view_id=view_id,
                        analysis_frame_index=frame_index,
                        source_seconds=source_offset_seconds + frame_index / fps,
                        hands=hands,
                    )
                )
                if first_output_seconds is None:
                    first_output_seconds = time.perf_counter() - start
    finally:
        capture.release()
    return tuple(observations), RuntimeMeasurements(
        elapsed_seconds=time.perf_counter() - start,
        time_to_first_usable_output_seconds=first_output_seconds,
        gpu_peak_vram_bytes=0,
        known_unavailable_measures=("ground-truth hand-pose accuracy",),
    )


def _write_observations(path: Path, observations: tuple[FrameObservations, ...]) -> None:
    path.write_text("".join(observation.model_dump_json() + "\n" for observation in observations))


def _bounded_video(proxy_path: Path, output_path: Path, frame_count: int) -> Path:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(proxy_path),
            "-frames:v",
            str(frame_count),
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ],
        check=True,
    )
    return output_path


def _contact_sheet(
    *,
    video_path: Path,
    observations: tuple[FrameObservations, ...],
    output_path: Path,
) -> Path:
    selected = sorted({0, len(observations) // 2, len(observations) - 1})
    capture = cv2.VideoCapture(str(video_path))
    panels = []
    try:
        for frame_index in selected:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"could not decode contact-sheet frame {frame_index}")
            height, width = frame.shape[:2]
            observation = observations[frame_index]
            for hand in observation.hands:
                points = [
                    (round(point.x * width), round(point.y * height)) for point in hand.landmarks
                ]
                for start, end in HAND_CONNECTIONS:
                    cv2.line(frame, points[start], points[end], (0, 180, 255), 2)
                for point in points:
                    cv2.circle(frame, point, 3, (0, 255, 255), -1)
                x1 = round(hand.box.x * width)
                y1 = round(hand.box.y * height)
                x2 = round((hand.box.x + hand.box.width) * width)
                y2 = round((hand.box.y + hand.box.height) * height)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 2)
                cv2.putText(
                    frame,
                    f"{hand.hand_id}: {hand.side}",
                    (x1, max(24, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 180, 255),
                    2,
                )
            cv2.putText(
                frame,
                f"analysis frame {frame_index}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            panels.append(frame)
    finally:
        capture.release()
    sheet = cv2.vconcat(panels)
    if not cv2.imwrite(str(output_path), sheet):
        raise RuntimeError(f"could not write contact sheet: {output_path}")
    return output_path


def run(args: argparse.Namespace) -> Path:
    repository_root = args.repository_root.resolve()
    config_path = (repository_root / args.config).resolve()
    model_path = (repository_root / args.model).resolve()
    config, proxy, proxy_path, requested_frames = _verify_inputs(
        repository_root=repository_root,
        config_path=config_path,
        model_path=model_path,
        view_id=args.view,
        seconds=args.seconds,
    )
    run_id = (
        args.run_id
        or f"mediapipe-hands-static-{args.seconds:g}s-{datetime.now(UTC):%Y%m%dt%H%M%Sz}"
    )
    run_directory = (repository_root / args.output_root / run_id).resolve()
    run_directory.mkdir(parents=True, exist_ok=False)
    observations, measurements = _detect(
        proxy_path=proxy_path,
        model_path=model_path,
        view_id=args.view,
        frame_count=requested_frames,
        fps=proxy.fps,
        source_offset_seconds=config.proxy_timing.source_seconds_for_frame(ClockName.ANALYSIS, 0),
        input_mirrored=args.input_mirrored,
        roi=args.roi,
        roi_upscale=args.roi_upscale,
        include_full_frame=args.include_full_frame,
        min_detection_confidence=args.min_detection_confidence,
        min_presence_confidence=args.min_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )
    observations_path = run_directory / "observations.jsonl"
    video_path = _bounded_video(proxy_path, run_directory / "input.mp4", requested_frames)
    qa_path = _contact_sheet(
        video_path=video_path,
        observations=observations,
        output_path=run_directory / "contact_sheet.png",
    )
    _write_observations(observations_path, observations)
    requested_range = FrameRange(start_frame=0, end_frame_exclusive=requested_frames)
    runtime_settings: dict[str, str | int | float | bool | None] = {
        "analysis_fps": proxy.fps,
        "num_hands": 2,
        "input_mirrored": args.input_mirrored,
        "roi": ",".join(str(value) for value in args.roi) if args.roi else None,
        "roi_upscale": args.roi_upscale,
        "include_full_frame": args.include_full_frame,
        "handedness_vote_frames": HANDEDNESS_VOTE_FRAMES,
        "track_max_gap_frames": TRACK_MAX_GAP_FRAMES,
        "track_max_wrist_distance": TRACK_MAX_WRIST_DISTANCE,
        "public_hand_id_policy": "frame-local detector order; no persistent identity claim",
        "min_hand_detection_confidence": args.min_detection_confidence,
        "min_hand_presence_confidence": args.min_presence_confidence,
        "min_tracking_confidence": args.min_tracking_confidence,
    }
    metadata = MediaPipeHandsRunMetadata(
        requested_analysis_frame_range=requested_range,
        requested_seconds=args.seconds,
        source_fingerprint=ArtifactFingerprint(
            uri=proxy.raw_source.raw_uri,
            sha256=proxy.raw_source.checksum_sha256,
            source="approved_config",
        ),
        proxy_fingerprint=ArtifactFingerprint(
            uri=proxy.proxy_uri, sha256=proxy.checksum_sha256, source="approved_config"
        ),
        config_fingerprint=ArtifactFingerprint(
            uri=relative_uri(config_path, repository_root),
            sha256=sha256_file(config_path),
            source="measured",
        ),
        model_fingerprint=ArtifactFingerprint(
            uri=DEFAULT_MODEL_URI, sha256=DEFAULT_MODEL_SHA256, source="measured"
        ),
        adapter=AdapterMetadata(
            name="mediapipe-hand-landmarker",
            version=importlib.import_module("mediapipe").__version__,
            implementation_basis="MediaPipe Tasks Hand Landmarker VIDEO mode",
            external_source_uri="https://github.com/google-ai-edge/mediapipe",
        ),
        runtime_settings=runtime_settings,
        measurements=measurements,
        observations_uri=relative_uri(observations_path, repository_root),
        rerun_artifact_uri=relative_uri(run_directory / "hands.rrd", repository_root),
        qa_artifact_uri=relative_uri(qa_path, repository_root),
    )
    method_statuses = (
        MethodStatus(
            method_name="mediapipe-hand-landmarker-static-baseline",
            stage="pose",
            state=MethodState.SUCCEEDED,
            artifact_uri=relative_uri(observations_path, repository_root),
            measured_on=f"{args.view}; approved {args.seconds:g}-second proxy prefix",
        ),
        MethodStatus(
            method_name="rerun-mediapipe-hands-export",
            stage="export",
            state=MethodState.SUCCEEDED,
            artifact_uri=relative_uri(run_directory / "hands.rrd", repository_root),
            measured_on="normalized MediaPipe observations; bounded input video logged once",
        ),
    )
    manifest = RunManifest(
        run_id=run_id,
        clip=config.clip.model_copy(update={"source_duration_seconds": args.seconds}),
        coverage=FullDurationCoverage(
            source_duration_seconds=args.seconds,
            covered_intervals=(TimeInterval(start_seconds=0.0, end_seconds=args.seconds),),
        ),
        chunk_policy=ChunkContinuityPolicy(overlap_seconds=0.0, max_allowed_gap_seconds=0.0),
        method_statuses=method_statuses,
        observations=observations,
        mediapipe_hands=metadata,
    )
    manifest_path = run_directory / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n")
    export_run(
        manifest,
        run_directory / "hands.rrd",
        video_path=video_path,
        video_dimensions=(proxy.dimensions.width, proxy.dimensions.height),
        asset_reference=EncodedAssetInput(
            uri=proxy.proxy_uri,
            media_type="video/mp4",
            checksum_sha256=proxy.checksum_sha256,
        ),
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--view", default=DEFAULT_VIEW_ID)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--run-id")
    parser.add_argument("--input-mirrored", action="store_true")
    parser.add_argument(
        "--roi",
        type=parse_roi,
        help="Normalized inference crop as x,y,width,height; outputs map to the full frame",
    )
    parser.add_argument("--roi-upscale", type=float, default=1.0)
    parser.add_argument(
        "--include-full-frame",
        action="store_true",
        help="Fuse full-frame and ROI detections; requires --roi",
    )
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-presence-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    args = parser.parse_args()
    if args.roi_upscale < 1:
        parser.error("--roi-upscale must be at least 1")
    if args.include_full_frame and args.roi is None:
        parser.error("--include-full-frame requires --roi")
    for name in (
        "min_detection_confidence",
        "min_presence_confidence",
        "min_tracking_confidence",
    ):
        if not 0 <= getattr(args, name) <= 1:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    print(run(args))


if __name__ == "__main__":
    main()
