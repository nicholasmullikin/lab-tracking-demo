"""Appearance-based, multi-view-checked seeding of a recording the system has never seen.

Recording 2 has no other-view masks to triangulate a seed from, so the localiser is
appearance-based: a DINOv2-small exemplar library is built from recording 1's human masks on
C10379 (the seed-search truth set), SAM3 image-decoder candidates are generated from a coarse
grid of box prompts over the table region in every static view at one seed frame, ranked per
part by exemplar similarity, and a part is *accepted* only when the top candidates of at
least three static views triangulate to one point (30 raw px filter) with sizes in band.

Honesty gate (plan B2/C3): on recording 1 the automatic seeding of `rear_body` and `cabin`
passed the 0.6 held-out gate (0.80 / 0.95) and `chassis` / `interior` did not (0.53 / 0.47).
Here every part is localised and checked, but only the passing parts are written as seeds;
the others are ranked *proposals* for the human. A part that fails consistency is "blocked",
never guessed.

Sub-commands: `window` (seed frame by dataset hand joints, trimmed proxies and a derived clip
config), `plan` (grid prompts), `decode` (GPU, one warm decoder per view), `accept` (CPU:
exemplars, ranking, consistency, seed manifests, proposals, report).

Claim boundary: every mask is agent-selected; acceptance means the cameras agree with each
other, not that the mask is right. CC BY-NC 4.0 applies to the dataset assets.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from pydantic import Field

from .assembly101_camera_fit import PoseMembers
from .assembly101_pose_schemas import ASSEMBLY101_HAND_SIDES
from .assembly101_recordings import Assembly101Recording, get_recording
from .digest_cache import sha256_file
from .four_part_contract import TARGETS
from .multiview_consensus import mask_centroid_raw, relative_uri
from .multiview_geometry import CameraRig, TablePlane
from .multiview_schemas import (
    EXEMPLAR_SEED_PROVENANCE,
    MultiviewSeedTransferManifest,
    SeedAcceptanceRules,
    SeedCandidate,
    SeedPrompt,
    SeedTransferPart,
)
from .multiview_seed_transfer import (
    load_mask,
    project_to_proxy,
    proxy_focal_px,
    proxy_to_raw_scale,
    square_box,
    view_id_for,
)
from .schemas import ArtifactFingerprint, G2PreprocessingManifest, PixelBox, VersionedModel
from .seed_search import DINOV2_SCRIPT, _crop_box

OUTPUT_ROOT = Path("runs/rec2-seed-proposals-20260920")
DEFAULT_RECORDING = "nusar_9061"
TRUTH_SET = Path("runs/seed-search-20260920/truth_set.json")
REC1_C10379_CONFIG = Path(
    "configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_g2.json"
)
SEED_FRAME_SEARCH = (300, 422)
SEED_FRAME_WINDOW = 5
GRID_SPACING_MM = 55.0
GRID_MARGIN_MM = 120.0
PART_RADII_MM = (32.0, 55.0)
BOX_MARGIN = 0.25
# Held-out IoU on recording 1's human masks (seed search, Sep 20): the gate that decides
# which parts may be *used* as seeds here; the others are proposals only.
REC1_HELD_OUT_IOU = {"chassis": 0.525, "interior": 0.465, "rear_body": 0.795, "cabin": 0.949}
SEED_GATE_IOU = 0.6
PASSING_PARTS = tuple(p for p in TARGETS if REC1_HELD_OUT_IOU[p] >= SEED_GATE_IOU)
CONSISTENCY_MIN_VIEWS = 3
CONSISTENCY_MAX_RAW_PX = 30.0
RADIUS_BAND = (0.5, 2.0)
PLAUSIBLE_RADIUS_MM = (18.0, 95.0)
TOP_K_PER_VIEW = 6
PROPOSALS_PER_VIEW = 3
MIN_PART_SEPARATION_MM = 60.0
DEDUPE_IOU = 0.85
MAX_CANDIDATES_PER_VIEW = 220
CLAIM_BOUNDARIES: tuple[str, ...] = (
    "Every seed and proposal is agent-selected (selected_by agent, provenance "
    "exemplar_multiview_consistency); no human reviewed any mask here.",
    "Exemplar similarity ranks SAM3 image-decoder candidates against one person's masks on "
    "16 frames of one view of another recording; it is not recognition and not accuracy.",
    "Acceptance means the top candidates of >= 3 static views triangulate to one point within "
    "30 raw px with sizes in band: the cameras agree with each other, not that the mask is right.",
    "Only parts whose automatic seeding passed the 0.6 held-out gate on recording 1 (rear_body, "
    "cabin) are written as seeds; chassis and interior are ranked proposals for the human.",
    "Assembly101 is CC BY-NC 4.0; attribution applies to every derived artifact.",
)


def _fingerprint(path: Path, repository_root: Path) -> ArtifactFingerprint:
    return ArtifactFingerprint(
        uri=relative_uri(path.resolve(), repository_root),
        sha256=sha256_file(path),
        source="measured",
    )


def _read_frame(video: Path, index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise RuntimeError(f"{video} has no frame {index}")
    return frame


# -- seed frame and window ------------------------------------------------------------------


class SeedFrameChoice(VersionedModel):
    recording: str
    reference_view: str
    search_range: tuple[int, int]
    seed_frame: int
    window_frames: int
    min_joint_height_mm: float
    rule: str
    per_frame_min_height_mm: dict[int, float]


def choose_seed_frame(
    rig: CameraRig,
    members: PoseMembers,
    plane: TablePlane,
    *,
    reference_view: str,
    search: tuple[int, int] = SEED_FRAME_SEARCH,
    window: int = SEED_FRAME_WINDOW,
    confidence_floor: float = 0.5,
) -> SeedFrameChoice:
    """The frame in `search` whose hands stay highest above the table over `window` frames.

    "Hands away" is measured, not assumed: for every candidate frame the lowest confident
    dataset hand joint above the fitted table plane is taken, the minimum over the window is
    the frame's score, and the highest score wins (a frame without confident hands scores as
    fully clear). The parts lie on the table, so high hands do not cover them.
    """
    per_frame: dict[int, float] = {}
    for frame in range(search[0], search[1]):
        key = str(rig.pose_frame(reference_view, frame))
        landmarks = members.landmarks3d.get(key)
        heights: list[float] = []
        if landmarks is not None:
            for hand in ASSEMBLY101_HAND_SIDES:
                if float(members.confidences[key][str(hand)]) < confidence_floor:
                    continue
                joints = np.asarray(landmarks[str(hand)], dtype=np.float64)
                heights.extend(plane.signed_distance(joints).tolist())
        per_frame[frame] = float(min(heights)) if heights else float("inf")
    best_frame, best_score = search[0], -np.inf
    for frame in range(search[0], search[1] - window + 1):
        score = min(per_frame[f] for f in range(frame, frame + window))
        if score > best_score:
            best_frame, best_score = frame, score
    return SeedFrameChoice(
        recording="",
        reference_view=reference_view,
        search_range=search,
        seed_frame=best_frame,
        window_frames=window,
        min_joint_height_mm=float(best_score if np.isfinite(best_score) else 1e6),
        rule=(
            f"argmax over frames f in [{search[0]}, {search[1] - window}] of the minimum, over "
            f"[f, f+{window}), of the lowest confident dataset hand joint's height above the "
            "fitted table plane; frames without confident hands count as clear"
        ),
        per_frame_min_height_mm={f: (h if np.isfinite(h) else 1e6) for f, h in per_frame.items()},
    )


def trimmed_proxy_command(
    source: Path, destination: Path, *, start_frame: int, count: int
) -> list[str]:
    """Frame-exact re-encode of proxy frames [start, start+count) with the proxy's own recipe."""
    return [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vf",
        f"select=between(n\\,{start_frame}\\,{start_frame + count - 1}),setpts=N/30/TB",
        "-r",
        "30",
        "-frames:v",
        str(count),
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(destination),
    ]


def _probe_frame_count(path: Path) -> int:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(completed.stdout.strip().splitlines()[0])


def write_seed_window(
    repository_root: Path,
    *,
    recording: Assembly101Recording,
    seed_frame: int,
    end_frame_exclusive: int,
    config_output: Path,
    derived_dir: Path,
) -> Path:
    """Trimmed proxies [seed_frame, end) for the static views and a clip config naming them.

    The tracker seeds at frame 0 of the video it is given, so a mid-window seed needs a
    proxy that starts there. Analysis frame `t` of the trimmed clip is proxy frame
    `seed_frame + t` of the recording's fetched window; the config's clock mappings carry the
    shifted source offset so timestamps stay in source seconds.
    """
    config_path = repository_root / recording.all_static_clip_config
    config = G2PreprocessingManifest.model_validate_json(config_path.read_text(encoding="utf-8"))
    count = end_frame_exclusive - seed_frame
    shift_seconds = seed_frame / 30.0
    derived_dir.mkdir(parents=True, exist_ok=True)
    proxies = []
    for proxy in config.proxies:
        source = repository_root / proxy.proxy_uri
        stem = source.stem.replace(
            f"{recording.window_start_seconds:.3f}-{recording.window_end_seconds:.3f}",
            f"seed{seed_frame}-{end_frame_exclusive}",
        )
        destination = derived_dir / f"{stem}.mp4"
        if not destination.is_file():
            subprocess.run(
                trimmed_proxy_command(source, destination, start_frame=seed_frame, count=count),
                check=True,
            )
        frames = _probe_frame_count(destination)
        if frames != count:
            raise RuntimeError(f"{destination} has {frames} frames, expected {count}")
        proxies.append(
            proxy.model_copy(
                update={
                    "proxy_uri": relative_uri(destination, repository_root),
                    "checksum_sha256": sha256_file(destination),
                    "frame_count": count,
                }
            )
        )

    def shifted(timing: Any) -> Any:
        return timing.model_copy(
            update={
                "mappings": tuple(
                    m.model_copy(
                        update={"source_offset_seconds": m.source_offset_seconds + shift_seconds}
                    )
                    for m in timing.mappings
                )
            }
        )

    clip = config.clip.model_copy(
        update={
            "clip_id": f"{config.clip.clip_id}-seed{seed_frame}",
            "asset": next(
                p for p in proxies if p.view_id == view_id_for(recording.primary_static_view)
            ).model_dump()
            and config.clip.asset.model_copy(
                update={
                    "uri": next(
                        p.proxy_uri
                        for p in proxies
                        if p.view_id == view_id_for(recording.primary_static_view)
                    ),
                    "checksum_sha256": next(
                        p.checksum_sha256
                        for p in proxies
                        if p.view_id == view_id_for(recording.primary_static_view)
                    ),
                }
            ),
            "timing": shifted(config.clip.timing),
            "source_duration_seconds": count / 30.0,
        }
    )
    raw_start = config.raw_frame_range.start_frame + 2 * seed_frame
    analysis_start = config.analysis_frame_range.start_frame + seed_frame
    derived = config.model_copy(
        update={
            "clip": clip,
            "proxy_timing": shifted(config.proxy_timing),
            "source_interval": config.source_interval.model_copy(
                update={
                    "start_seconds": config.source_interval.start_seconds + shift_seconds,
                    "end_seconds": config.source_interval.start_seconds
                    + shift_seconds
                    + count / 30.0,
                }
            ),
            "raw_frame_range": config.raw_frame_range.model_copy(
                update={"start_frame": raw_start, "end_frame_exclusive": raw_start + 2 * count}
            ),
            "analysis_frame_range": config.analysis_frame_range.model_copy(
                update={
                    "start_frame": analysis_start,
                    "end_frame_exclusive": analysis_start + count,
                }
            ),
            "proxy_frame_range": config.proxy_frame_range.model_copy(
                update={"start_frame": 0, "end_frame_exclusive": count}
            ),
            "proxies": tuple(proxies),
        }
    )
    g1 = derived.g1.model_copy(
        update={
            "scope": derived.g1.scope
            + f"; seed-window derivative: proxy frames [{seed_frame}, {end_frame_exclusive}) of "
            "that window re-encoded frame-exact (same libx264 crf 18 recipe) so a tracker "
            f"seeded at frame 0 of this clip is seeded at proxy frame {seed_frame} of the window"
        }
    )
    derived = derived.model_copy(update={"g1": g1})
    config_output.parent.mkdir(parents=True, exist_ok=True)
    config_output.write_text(derived.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return config_output


# -- grid plan -----------------------------------------------------------------------------


class GridPrompt(VersionedModel):
    prompt_id: str = Field(pattern=r"^t\d{6}-b\d{2,}$")
    view: str
    grid_index: int
    radius_mm: float
    world_point_mm: tuple[float, float, float]
    pixel_box: PixelBox


class ViewGridPlan(VersionedModel):
    view: str
    view_id: str
    proxy: ArtifactFingerprint
    proxy_dimensions: tuple[int, int]
    seed_frame: int
    prompts: tuple[GridPrompt, ...]


class ExemplarSeedPlan(VersionedModel):
    manifest_kind: Literal["exemplar_seed_plan"]
    recording: str
    recording_id: str
    clip_config: ArtifactFingerprint
    seed_frame: SeedFrameChoice
    table_plane: TablePlane
    grid_spacing_mm: float
    grid_margin_mm: float
    part_radii_mm: tuple[float, ...]
    box_margin: float
    grid_points_mm: tuple[tuple[float, float, float], ...]
    table_region_source: str
    views: tuple[ViewGridPlan, ...]
    prompt_count: int
    generated_at: datetime
    claim_boundaries: tuple[str, ...]


def fit_window_plane(
    rig: CameraRig, members: PoseMembers, recording: Assembly101Recording
) -> TablePlane:
    start = recording.window_start_raw_frame
    return rig.fit_table_plane(
        members.landmarks3d,
        members.confidences,
        pose_frames=range(start, start + recording.window_raw_frame_count),
    )


def table_grid(
    plane: TablePlane,
    members: PoseMembers,
    recording: Assembly101Recording,
    *,
    spacing_mm: float = GRID_SPACING_MM,
    margin_mm: float = GRID_MARGIN_MM,
    confidence_floor: float = 0.8,
    step: int = 10,
) -> tuple[np.ndarray, str]:
    """World points on the table plane covering where the hands worked during the window.

    The region is the bounding rectangle, in plane coordinates, of every confident dataset
    hand joint over the window dropped onto the plane, grown by `margin_mm`; the parts are
    laid out where the hands go.
    """
    normal = np.asarray(plane.normal)
    origin = np.asarray(plane.point_on_plane_mm)
    seed = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, seed)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    start = recording.window_start_raw_frame
    dropped: list[np.ndarray] = []
    for frame in range(start, start + recording.window_raw_frame_count, step):
        key = str(frame)
        landmarks = members.landmarks3d.get(key)
        if landmarks is None:
            continue
        for hand in ASSEMBLY101_HAND_SIDES:
            if float(members.confidences[key][str(hand)]) < confidence_floor:
                continue
            joints = np.asarray(landmarks[str(hand)], dtype=np.float64)
            joints = joints - plane.signed_distance(joints)[:, None] * normal[None, :]
            dropped.append(joints)
    if not dropped:
        raise ValueError("no confident hand joints in the window")
    points = np.concatenate(dropped)
    local = np.stack([(points - origin) @ u, (points - origin) @ v], axis=1)
    lo = local.min(axis=0) - margin_mm
    hi = local.max(axis=0) + margin_mm
    us = np.arange(lo[0], hi[0] + spacing_mm / 2, spacing_mm)
    vs = np.arange(lo[1], hi[1] + spacing_mm / 2, spacing_mm)
    grid = np.array([origin + a * u + b * v for a in us for b in vs])
    return grid, (
        f"bounding rectangle (plane coordinates) of {points.shape[0]} confident dataset hand "
        f"joints over the fetched window (every {step}th pose frame) dropped onto the table "
        f"plane, grown by {margin_mm:.0f} mm; {us.size} x {vs.size} points at {spacing_mm:.0f} mm"
    )


def plan_grid(
    repository_root: Path,
    *,
    recording: Assembly101Recording,
    output_dir: Path,
    seed_frame: int | None = None,
    views: Sequence[str] | None = None,
) -> Path:
    repository_root = repository_root.resolve()
    rig = CameraRig.load(repository_root, recording=recording)
    members = PoseMembers(repository_root, recording)
    plane = fit_window_plane(rig, members, recording)
    choice = choose_seed_frame(rig, members, plane, reference_view=recording.primary_static_view)
    choice = choice.model_copy(update={"recording": recording.label})
    if seed_frame is not None:
        choice = choice.model_copy(
            update={
                "seed_frame": seed_frame,
                "rule": choice.rule + f"; overridden on the command line to {seed_frame}",
            }
        )
    grid, region_source = table_grid(plane, members, recording)
    config_path = repository_root / recording.all_static_clip_config
    config = G2PreprocessingManifest.model_validate_json(config_path.read_text(encoding="utf-8"))
    proxies = {p.view_id: p for p in config.proxies}
    view_plans: list[ViewGridPlan] = []
    total = 0
    for view in views or recording.static_views:
        proxy = proxies[view_id_for(view)]
        shape = (proxy.dimensions.height, proxy.dimensions.width)
        counter = itertools.count(1)
        prompts: list[GridPrompt] = []
        centres = project_to_proxy(rig, view, grid)
        depths = rig.depth(view, grid)
        focal = proxy_focal_px(rig, view)
        for index, (centre, depth) in enumerate(zip(centres, depths, strict=True)):
            if np.isnan(centre).any() or depth <= 0:
                continue
            if not (0 <= centre[0] < shape[1] and 0 <= centre[1] < shape[0]):
                continue
            for radius in PART_RADII_MM:
                box = square_box(centre, radius * focal / depth, BOX_MARGIN, shape)
                if box is None:
                    continue
                prompts.append(
                    GridPrompt(
                        prompt_id=f"t{choice.seed_frame:06d}-b{next(counter):02d}",
                        view=view,
                        grid_index=index,
                        radius_mm=radius,
                        world_point_mm=tuple(float(x) for x in grid[index]),
                        pixel_box=box,
                    )
                )
        total += len(prompts)
        view_plans.append(
            ViewGridPlan(
                view=view,
                view_id=proxy.view_id,
                proxy=ArtifactFingerprint(
                    uri=proxy.proxy_uri, sha256=proxy.checksum_sha256, source="approved_config"
                ),
                proxy_dimensions=(proxy.dimensions.width, proxy.dimensions.height),
                seed_frame=choice.seed_frame,
                prompts=tuple(prompts),
            )
        )
    plan = ExemplarSeedPlan(
        manifest_kind="exemplar_seed_plan",
        recording=recording.label,
        recording_id=recording.recording_id,
        clip_config=_fingerprint(config_path, repository_root),
        seed_frame=choice,
        table_plane=plane,
        grid_spacing_mm=GRID_SPACING_MM,
        grid_margin_mm=GRID_MARGIN_MM,
        part_radii_mm=PART_RADII_MM,
        box_margin=BOX_MARGIN,
        grid_points_mm=tuple(tuple(float(x) for x in p) for p in grid),
        table_region_source=region_source,
        views=tuple(view_plans),
        prompt_count=total,
        generated_at=datetime.now(UTC),
        claim_boundaries=CLAIM_BOUNDARIES,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "seed_plan.json"
    path.write_text(plan.model_dump_json(indent=1) + "\n", encoding="utf-8")
    return path


def load_plan(path: Path) -> ExemplarSeedPlan:
    return ExemplarSeedPlan.model_validate_json(path.read_text(encoding="utf-8"))


# -- decode --------------------------------------------------------------------------------


def _decode_request(prompt: GridPrompt, shape: tuple[int, int], frame: int) -> dict[str, Any]:
    height, width = shape
    box = prompt.pixel_box
    return {
        "box_id": f"p{prompt.prompt_id[1:]}",
        "candidate_id": prompt.prompt_id,
        "frame_index": frame,
        "pixel_box": box.model_dump(mode="json"),
        "intended_target": "grid",
        "boxes": [[[box.x1 / width, box.y1 / height], [box.x2 / width, box.y2 / height]]],
        "fg_points": [],
        "bg_points": [],
    }


def run_decode(
    repository_root: Path, plan: ExemplarSeedPlan, *, output_dir: Path, device: str = "cuda:0"
) -> Path:
    """One warm image decoder per view; every grid prompt decoded at the seed frame."""
    from .seed_search import make_decoder

    started = time.monotonic()
    per_view: dict[str, Any] = {}
    for view_plan in plan.views:
        view_dir = output_dir / "decode" / view_plan.view
        results_dir = view_dir / "results"
        decoder = make_decoder(repository_root, view_plan.proxy, results_dir, device=device)
        view_started = time.monotonic()
        try:
            shape = (view_plan.proxy_dimensions[1], view_plan.proxy_dimensions[0])
            decoder.request("frame_preview", {"frame_index": view_plan.seed_frame}, timeout=600)
            decoded: dict[str, Any] = {}
            prompts = list(view_plan.prompts)
            for start in range(0, len(prompts), 40):
                chunk = prompts[start : start + 40]
                response = decoder.request(
                    "batch_decode",
                    {"prompts": [_decode_request(p, shape, view_plan.seed_frame) for p in chunk]},
                    timeout=1800,
                )
                for item in response["decoded"]:
                    decoded[item["candidate_id"]] = item["decoder_result"]
        finally:
            decoder.close()
        per_view[view_plan.view] = {
            "prompt_count": len(decoded),
            "elapsed_seconds": time.monotonic() - view_started,
            "results_dir": relative_uri(results_dir, repository_root),
            "decoded": decoded,
        }
        print(f"{view_plan.view}: {len(decoded)} prompts decoded", flush=True)
    payload = {
        "recording": plan.recording,
        "seed_frame": plan.seed_frame.seed_frame,
        "elapsed_seconds": time.monotonic() - started,
        "views": per_view,
    }
    path = output_dir / "decode_result.json"
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return path


# -- exemplars -----------------------------------------------------------------------------


def compute_embeddings(
    crops: list[dict[str, Any]], *, work_dir: Path, python: Path
) -> dict[str, np.ndarray]:
    """DINOv2-small pooled embeddings (CPU, the MuggledSAM interpreter); raises if unavailable."""
    if not crops:
        return {}
    work_dir.mkdir(parents=True, exist_ok=True)
    spec = work_dir / "dinov2_crops.json"
    spec.write_text(json.dumps({"crops": crops}), encoding="utf-8")
    script = work_dir / "dinov2_embed.py"
    script.write_text(DINOV2_SCRIPT, encoding="utf-8")
    output = work_dir / "dinov2_embeddings.npy"
    completed = subprocess.run(
        [str(python), str(script), str(spec), str(output)],
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        check=False,
    )
    (work_dir / "dinov2_embed.log").write_text(
        completed.stdout + completed.stderr, encoding="utf-8"
    )
    if completed.returncode != 0:
        raise RuntimeError(f"DINOv2 embedding failed; see {work_dir / 'dinov2_embed.log'}")
    loaded = np.load(output, allow_pickle=True).item()
    return {k: np.asarray(v, dtype=np.float64) for k, v in loaded.items()}


class ExemplarLibrary(VersionedModel):
    truth_set: ArtifactFingerprint
    source_view: str
    source_recording: str
    proxy: ArtifactFingerprint
    exemplars_per_part: dict[str, int]
    exemplar_masks: tuple[ArtifactFingerprint, ...]
    embedding_model: Literal["facebook/dinov2-small pooled, 224 px crops, pad 0.15"]


def build_exemplar_library(
    repository_root: Path, *, work_dir: Path, python: Path
) -> tuple[ExemplarLibrary, dict[str, np.ndarray]]:
    """Crops of every recording-1 human mask on C10379 (the seed-search truth set), embedded."""
    from .seed_search import load_truth_set

    truth_path = repository_root / TRUTH_SET
    truth = load_truth_set(truth_path)
    config = G2PreprocessingManifest.model_validate_json(
        (repository_root / REC1_C10379_CONFIG).read_text(encoding="utf-8")
    )
    proxy = next(p for p in config.proxies if p.view_id == "static-c10379")
    video = repository_root / proxy.proxy_uri
    frames_dir = work_dir / "exemplar_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    crops: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    masks: list[ArtifactFingerprint] = []
    frame_cache: dict[int, Path] = {}
    for entry in truth.positives("C10379"):
        if entry.mask is None:
            continue
        frame = entry.analysis_frame_index
        if frame not in frame_cache:
            image = _read_frame(video, frame)
            path = frames_dir / f"c10379_f{frame:06d}.png"
            cv2.imwrite(str(path), image)
            frame_cache[frame] = path
        mask = load_mask(repository_root / entry.mask.uri)
        box = _crop_box(mask)
        if box is None:
            continue
        key = f"{entry.part}|{frame}|{entry.source_candidate_id}"
        crops.append({"key": key, "image": str(frame_cache[frame]), "box": list(box)})
        counts[entry.part] = counts.get(entry.part, 0) + 1
        masks.append(entry.mask)
    embeddings = compute_embeddings(crops, work_dir=work_dir / "exemplar_embeddings", python=python)
    library = ExemplarLibrary(
        truth_set=_fingerprint(truth_path, repository_root),
        source_view="C10379",
        source_recording="nusar_9033",
        proxy=ArtifactFingerprint(
            uri=proxy.proxy_uri, sha256=proxy.checksum_sha256, source="approved_config"
        ),
        exemplars_per_part=counts,
        exemplar_masks=tuple(masks),
        embedding_model="facebook/dinov2-small pooled, 224 px crops, pad 0.15",
    )
    by_part: dict[str, list[np.ndarray]] = {}
    for key, vector in embeddings.items():
        by_part.setdefault(key.split("|", 1)[0], []).append(vector)
    return library, {part: np.stack(vectors) for part, vectors in by_part.items()}


# -- candidates, ranking, consistency -------------------------------------------------------


@dataclass
class Candidate:
    view: str
    prompt_id: str
    candidate_index: int
    mask_path: Path
    mask: np.ndarray
    decoder_iou: float
    area_px: int
    centroid_proxy: np.ndarray
    centroid_raw: np.ndarray
    depth_mm: float
    radius_mm: float
    similarity: dict[str, float]

    @property
    def key(self) -> str:
        return f"{self.prompt_id}#{self.candidate_index}"


def _centroid_depth(
    rig: CameraRig, plane: TablePlane, view: str, centroid_proxy: np.ndarray
) -> float:
    from .multiview_seed_transfer import lift_pixels_to_plane

    hit = lift_pixels_to_plane(
        rig, np.asarray(plane.normal), plane.offset_mm, view, centroid_proxy.reshape(1, 2)
    )[0]
    if np.isnan(hit).any():
        return float("nan")
    return float(rig.depth(view, hit.reshape(1, 3))[0])


def load_candidates(
    repository_root: Path,
    rig: CameraRig,
    plane: TablePlane,
    view_plan: ViewGridPlan,
    decoded: Mapping[str, Any],
    results_dir: Path,
) -> list[Candidate]:
    """Every decoded mask with a plausible on-table size, near-duplicates collapsed."""
    focal = proxy_focal_px(rig, view_plan.view)
    scale = proxy_to_raw_scale(view_plan.view)
    kept: list[Candidate] = []
    for prompt in view_plan.prompts:
        result = decoded.get(prompt.prompt_id)
        if result is None:
            continue
        for item in result["candidates"]:
            mask_path = (results_dir.parent / item["mask_uri"]).resolve()
            mask = load_mask(mask_path)
            area = int(mask.sum())
            if area < 40:
                continue
            centroid_raw = mask_centroid_raw(mask, scale)
            if centroid_raw is None:
                continue
            centroid_proxy = centroid_raw / scale
            depth = _centroid_depth(rig, plane, view_plan.view, centroid_proxy)
            if not np.isfinite(depth) or depth <= 0:
                continue
            radius = float(np.sqrt(area / np.pi) * depth / focal)
            if not (PLAUSIBLE_RADIUS_MM[0] <= radius <= PLAUSIBLE_RADIUS_MM[1]):
                continue
            kept.append(
                Candidate(
                    view=view_plan.view,
                    prompt_id=prompt.prompt_id,
                    candidate_index=int(item["candidate_index"]),
                    mask_path=mask_path,
                    mask=mask,
                    decoder_iou=float(item["iou_score"]),
                    area_px=area,
                    centroid_proxy=centroid_proxy,
                    centroid_raw=centroid_raw,
                    depth_mm=depth,
                    radius_mm=radius,
                    similarity={},
                )
            )
    kept.sort(key=lambda c: -c.decoder_iou)
    unique: list[Candidate] = []
    for candidate in kept:
        duplicate = False
        for other in unique:
            if np.linalg.norm(candidate.centroid_proxy - other.centroid_proxy) > 40:
                continue
            inter = np.logical_and(candidate.mask, other.mask).sum()
            union = np.logical_or(candidate.mask, other.mask).sum()
            if union and inter / union >= DEDUPE_IOU:
                duplicate = True
                break
        if not duplicate:
            unique.append(candidate)
        if len(unique) >= MAX_CANDIDATES_PER_VIEW:
            break
    return unique


def rank_by_exemplars(
    candidates: Sequence[Candidate],
    library: Mapping[str, np.ndarray],
    frame_image: Path,
    *,
    work_dir: Path,
    python: Path,
) -> None:
    """Fill `candidate.similarity[part]` = max cosine to that part's exemplars (in place)."""
    crops = []
    for candidate in candidates:
        box = _crop_box(candidate.mask)
        if box is None:
            continue
        crops.append({"key": candidate.key, "image": str(frame_image), "box": list(box)})
    embeddings = compute_embeddings(crops, work_dir=work_dir, python=python)
    for candidate in candidates:
        vector = embeddings.get(candidate.key)
        if vector is None:
            candidate.similarity = {part: -1.0 for part in library}
            continue
        candidate.similarity = {
            part: float(np.max(exemplars @ vector)) for part, exemplars in library.items()
        }


class ConsistencyResult(VersionedModel):
    part: str
    reached: bool
    reason: str
    world_point_mm: tuple[float, float, float] | None = None
    views_used: tuple[str, ...] = ()
    reprojection_px: dict[str, float] = Field(default_factory=dict)
    radius_mm: dict[str, float] = Field(default_factory=dict)
    median_radius_mm: float | None = None
    similarity: dict[str, float] = Field(default_factory=dict)
    margin: dict[str, float] = Field(default_factory=dict)
    chosen: dict[str, str] = Field(default_factory=dict)
    height_above_table_mm: float | None = None


def _margin(candidate: Candidate, part: str) -> float:
    others = [s for p, s in candidate.similarity.items() if p != part]
    return candidate.similarity.get(part, -1.0) - (max(others) if others else -1.0)


def find_consistent_part(
    rig: CameraRig,
    plane: TablePlane,
    part: str,
    per_view: Mapping[str, Sequence[Candidate]],
    *,
    excluded_points: Sequence[np.ndarray] = (),
    used_keys: set[str] | None = None,
    top_k: int = TOP_K_PER_VIEW,
) -> ConsistencyResult:
    """Best-supported triangulation of one part's top-K candidates across the static views.

    Every pair of candidates from two views proposes a 3D point; the views whose best candidate
    (highest part similarity among those reprojecting within 30 raw px) supports it are counted;
    the hypothesis with the most supporting views wins, ties broken by summed similarity. Needs
    >= 3 views, a point at least 60 mm from already-placed parts, and per-view radii within
    [0.5, 2.0] x their median.
    """
    used_keys = used_keys or set()
    shortlist: dict[str, list[Candidate]] = {}
    for view, candidates in per_view.items():
        ranked = sorted(
            (c for c in candidates if c.key not in used_keys and c.similarity.get(part, -1) > -1),
            key=lambda c: -c.similarity[part],
        )
        if ranked:
            shortlist[view] = ranked[:top_k]
    views = sorted(shortlist)
    if len(views) < CONSISTENCY_MIN_VIEWS:
        return ConsistencyResult(
            part=part,
            reached=False,
            reason=f"only {len(views)} views have ranked candidates (< {CONSISTENCY_MIN_VIEWS})",
        )
    best: ConsistencyResult | None = None
    best_score = (-1, -np.inf)
    for a, b in itertools.combinations(views, 2):
        for ca in shortlist[a]:
            for cb in shortlist[b]:
                result = rig.triangulate(
                    {a: ca.centroid_raw.reshape(1, 2), b: cb.centroid_raw.reshape(1, 2)},
                    reproj_filter_px=None,
                )
                point = result.points[0]
                if np.isnan(point).any():
                    continue
                if any(np.linalg.norm(point - p) < MIN_PART_SEPARATION_MM for p in excluded_points):
                    continue
                supporters: dict[str, Candidate] = {}
                errors: dict[str, float] = {}
                for view in views:
                    projected = rig.project(view, point.reshape(1, 3))[0]
                    choice = None
                    for candidate in shortlist[view]:
                        error = float(np.linalg.norm(projected - candidate.centroid_raw))
                        if error <= CONSISTENCY_MAX_RAW_PX and (
                            choice is None
                            or candidate.similarity[part] > choice[0].similarity[part]
                        ):
                            choice = (candidate, error)
                    if choice is not None:
                        supporters[view] = choice[0]
                        errors[view] = choice[1]
                if len(supporters) < CONSISTENCY_MIN_VIEWS:
                    continue
                refined = rig.triangulate(
                    {v: c.centroid_raw.reshape(1, 2) for v, c in supporters.items()},
                    reproj_filter_px=CONSISTENCY_MAX_RAW_PX,
                    min_views=CONSISTENCY_MIN_VIEWS,
                )
                refined_point = refined.points[0]
                if np.isnan(refined_point).any():
                    continue
                used = [v for v, u in zip(refined.views, refined.used[0], strict=True) if u]
                if len(used) < CONSISTENCY_MIN_VIEWS:
                    continue
                radii = {v: supporters[v].radius_mm for v in used}
                median_radius = float(np.median(list(radii.values())))
                in_band = [
                    v
                    for v in used
                    if RADIUS_BAND[0] * median_radius <= radii[v] <= RADIUS_BAND[1] * median_radius
                ]
                if len(in_band) < CONSISTENCY_MIN_VIEWS:
                    continue
                score = (len(in_band), sum(supporters[v].similarity[part] for v in in_band))
                if score > best_score:
                    best_score = score
                    reprojection = {
                        v: float(e)
                        for v, e in zip(refined.views, refined.reprojection_px[0], strict=True)
                        if v in in_band
                    }
                    best = ConsistencyResult(
                        part=part,
                        reached=True,
                        reason=(
                            f"{len(in_band)} static views agree within "
                            f"{CONSISTENCY_MAX_RAW_PX:.0f} raw px; radii in band"
                        ),
                        world_point_mm=tuple(float(x) for x in refined_point),
                        views_used=tuple(in_band),
                        reprojection_px=reprojection,
                        radius_mm={v: radii[v] for v in in_band},
                        median_radius_mm=median_radius,
                        similarity={v: supporters[v].similarity[part] for v in in_band},
                        margin={v: _margin(supporters[v], part) for v in in_band},
                        chosen={v: supporters[v].key for v in in_band},
                        height_above_table_mm=float(
                            plane.signed_distance(refined_point.reshape(1, 3))[0]
                        ),
                    )
    if best is None:
        return ConsistencyResult(
            part=part,
            reached=False,
            reason=(
                f"no 3D point is supported by >= {CONSISTENCY_MIN_VIEWS} views' top-{top_k} "
                f"candidates within {CONSISTENCY_MAX_RAW_PX:.0f} raw px with radii in band"
            ),
        )
    return best


# -- accept: manifests, proposals, report ---------------------------------------------------


class ViewSeedOutcome(VersionedModel):
    view: str
    candidates_considered: int
    seeded_parts: tuple[str, ...]
    proposal_parts: tuple[str, ...]
    seed_manifest: ArtifactFingerprint | None
    run_decision: str


class ExemplarSeedReport(VersionedModel):
    manifest_kind: Literal["exemplar_seed_report"]
    recording: str
    plan: ArtifactFingerprint
    decode_result: ArtifactFingerprint
    exemplar_library: ExemplarLibrary
    seed_frame: int
    rec1_held_out_iou: dict[str, float]
    seed_gate_iou: float
    parts_used_as_seeds: tuple[str, ...]
    consistency: tuple[ConsistencyResult, ...]
    views: tuple[ViewSeedOutcome, ...]
    elapsed_seconds: float
    generated_at: datetime
    claim_boundaries: tuple[str, ...]


def _overlay(
    frame_bgr: np.ndarray, masks: Sequence[tuple[str, np.ndarray]], box: PixelBox | None
) -> np.ndarray:
    palette = [(0, 200, 255), (255, 120, 0), (0, 255, 120), (255, 0, 200), (200, 200, 0)]
    out = frame_bgr.copy()
    for i, (label, mask) in enumerate(masks):
        colour = palette[i % len(palette)]
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, colour, 2)
        cv2.putText(
            out, label, (8, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA
        )
    if box is not None:
        cv2.rectangle(out, (box.x1, box.y1), (box.x2, box.y2), (255, 255, 255), 1)
    return out


def accept(
    repository_root: Path,
    *,
    plan_path: Path,
    decode_path: Path,
    output_dir: Path,
    python: Path,
    seed_window_config: Path | None = None,
) -> Path:
    """Rank, check consistency, write seed manifests (passing parts) and proposals."""
    started = time.monotonic()
    repository_root = repository_root.resolve()
    plan = load_plan(plan_path)
    decoded_all = json.loads(decode_path.read_text(encoding="utf-8"))
    recording = get_recording(plan.recording, repository_root)
    rig = CameraRig.load(repository_root, recording=recording)
    plane = plan.table_plane
    work_dir = output_dir / "exemplar"
    library_record, library = build_exemplar_library(
        repository_root, work_dir=work_dir, python=python
    )

    per_view: dict[str, list[Candidate]] = {}
    frame_images: dict[str, Path] = {}
    for view_plan in plan.views:
        entry = decoded_all["views"].get(view_plan.view)
        if entry is None:
            continue
        results_dir = repository_root / entry["results_dir"]
        candidates = load_candidates(
            repository_root, rig, plane, view_plan, entry["decoded"], results_dir
        )
        image = results_dir / "frames" / f"frame-{view_plan.seed_frame:06d}.jpg"
        if not image.is_file():
            image = work_dir / "frames" / f"{view_plan.view}_f{view_plan.seed_frame:06d}.png"
            image.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(image), _read_frame(repository_root / view_plan.proxy.uri, view_plan.seed_frame)
            )
        frame_images[view_plan.view] = image
        rank_by_exemplars(
            candidates, library, image, work_dir=work_dir / f"rank_{view_plan.view}", python=python
        )
        per_view[view_plan.view] = candidates
        print(f"{view_plan.view}: {len(candidates)} candidates ranked", flush=True)

    # Parts in order of recording-1 confidence; each placed part excludes its neighbourhood.
    order = sorted(TARGETS, key=lambda p: -REC1_HELD_OUT_IOU[p])
    results: dict[str, ConsistencyResult] = {}
    placed: list[np.ndarray] = []
    used_keys: set[str] = set()
    for part in order:
        result = find_consistent_part(
            rig, plane, part, per_view, excluded_points=placed, used_keys=used_keys
        )
        results[part] = result
        if result.reached and result.world_point_mm is not None:
            placed.append(np.asarray(result.world_point_mm))
            used_keys.update(result.chosen.values())
        print(
            f"{part}: {'consistent' if result.reached else 'BLOCKED'} - {result.reason}", flush=True
        )

    # Seed-window clip config (the run's proxies start at the seed frame) for the manifests.
    config_for_manifest = seed_window_config or (repository_root / recording.all_static_clip_config)
    window_config = G2PreprocessingManifest.model_validate_json(
        config_for_manifest.read_text(encoding="utf-8")
    )
    window_proxies = {p.view_id: p for p in window_config.proxies}
    rules = SeedAcceptanceRules(
        min_area_ratio=RADIUS_BAND[0] ** 2,
        max_area_ratio=RADIUS_BAND[1] ** 2,
        min_backprojection_iou=0.0,
        max_centroid_ray_distance_radii=1.0,
        min_hand_joints_inside_fraction=0.0,
        min_parts_to_run=2,
        description=(
            "Exemplar-ranked SAM3 candidates; a part is accepted when the top candidates of >= 3 "
            "static views triangulate within 30 raw px and each view's implied radius is within "
            "[0.5, 2.0] x the median; only parts that passed the 0.6 held-out gate on recording 1 "
            "(rear_body, cabin) are used as seeds, the rest are proposals"
        ),
    )
    outcomes: list[ViewSeedOutcome] = []
    proposals_root = output_dir / "proposals"
    seeds_root = output_dir / "seeds"
    for view, candidates in per_view.items():
        by_key = {c.key: c for c in candidates}
        parts: list[SeedTransferPart] = []
        seeded: list[str] = []
        proposal_parts: list[str] = []
        frame = cv2.imread(str(frame_images[view]))
        for part in TARGETS:
            result = results[part]
            chosen = by_key.get(result.chosen.get(view, "")) if result.reached else None
            ranked = sorted(
                (c for c in candidates if c.similarity.get(part, -1) > -1),
                key=lambda c: -c.similarity[part],
            )[:PROPOSALS_PER_VIEW]
            # Proposals for the human: top-3 by similarity, the consistent pick first if any.
            proposal_dir = proposals_root / view / part
            proposal_dir.mkdir(parents=True, exist_ok=True)
            listed = ([chosen] if chosen is not None else []) + [
                c for c in ranked if c is not chosen
            ]
            overlay_items = []
            records = []
            for rank, candidate in enumerate(listed[: PROPOSALS_PER_VIEW + 1]):
                destination = proposal_dir / f"candidate_{rank:02d}.png"
                destination.write_bytes(candidate.mask_path.read_bytes())
                margin = _margin(candidate, part)
                label = f"{rank}: sim {candidate.similarity[part]:.3f} margin {margin:+.3f}"
                if candidate is chosen:
                    label += " (consistent pick)"
                overlay_items.append((label, candidate.mask))
                records.append(
                    {
                        "rank": rank,
                        "candidate": candidate.key,
                        "mask": relative_uri(destination, repository_root),
                        "similarity": candidate.similarity,
                        "margin": _margin(candidate, part),
                        "area_px": candidate.area_px,
                        "radius_mm": candidate.radius_mm,
                        "consistent_pick": candidate is chosen,
                    }
                )
            cv2.imwrite(str(proposal_dir / "overlay.png"), _overlay(frame, overlay_items, None))
            (proposal_dir / "proposal.json").write_text(
                json.dumps(
                    {
                        "view": view,
                        "part": part,
                        "seed_frame": plan.seed_frame.seed_frame,
                        "used_as_seed": part in PASSING_PARTS and chosen is not None,
                        "rec1_held_out_iou": REC1_HELD_OUT_IOU[part],
                        "consistency": result.model_dump(mode="json"),
                        "candidates": records,
                        "human_decision": None,
                        "selected_by": "agent",
                        "provenance": EXEMPLAR_SEED_PROVENANCE,
                    },
                    indent=1,
                )
                + "\n",
                encoding="utf-8",
            )
            candidate_records = tuple(
                SeedCandidate(
                    prompt_id=c.prompt_id,
                    candidate_index=c.candidate_index,
                    mask=_fingerprint(c.mask_path, repository_root),
                    decoder_iou_estimate=c.decoder_iou,
                    mask_area_px=c.area_px,
                    sanity_pass=c is chosen,
                    acceptance_basis="exemplar_multiview_consistency" if c is chosen else None,
                    sanity_notes=(
                        f"exemplar similarity {c.similarity[part]:.3f}, "
                        f"margin {_margin(c, part):+.3f}",
                        f"implied radius {c.radius_mm:.0f} mm at depth {c.depth_mm:.0f} mm",
                    ),
                    consistency_views_used=result.views_used if c is chosen else None,
                    consistency_reprojection_px=(
                        result.reprojection_px.get(view) if c is chosen else None
                    ),
                )
                for c in listed[: PROPOSALS_PER_VIEW + 1]
            )
            if chosen is not None and part in PASSING_PARTS:
                status, reason = "accepted", None
                seeded.append(part)
            elif chosen is not None:
                status = "blocked"
                reason = (
                    f"consistent across {len(result.views_used)} views but {part}'s automatic "
                    f"seeding scored {REC1_HELD_OUT_IOU[part]:.3f} held-out on recording 1 "
                    f"(< {SEED_GATE_IOU}); proposal only"
                )
                proposal_parts.append(part)
            else:
                status = "blocked"
                reason = f"multi-view consistency not reached: {result.reason}"
                proposal_parts.append(part)
            parts.append(
                SeedTransferPart(
                    target=part,
                    kind="part",
                    status=status,  # type: ignore[arg-type]
                    blocked_reason=reason,
                    provenance=EXEMPLAR_SEED_PROVENANCE,
                    source_points_world_mm=(
                        (result.world_point_mm,) if result.world_point_mm is not None else ()
                    ),
                    triangulation_reprojection_px=result.reprojection_px,
                    height_above_table_mm=result.height_above_table_mm,
                    radius_mm=result.median_radius_mm,
                    projected_centroid_proxy_px=(
                        (float(chosen.centroid_proxy[0]), float(chosen.centroid_proxy[1]))
                        if chosen is not None
                        else None
                    ),
                    depth_mm=chosen.depth_mm if chosen is not None else None,
                    prompts=tuple(
                        SeedPrompt(
                            prompt_id=c.prompt_id,
                            target=part,
                            variant="grid_box",
                            pixel_box=next(
                                p.pixel_box
                                for vp in plan.views
                                if vp.view == view
                                for p in vp.prompts
                                if p.prompt_id == c.prompt_id
                            ),
                        )
                        for c in listed[: PROPOSALS_PER_VIEW + 1]
                    ),
                    candidates=candidate_records,
                    accepted=next((r for r in candidate_records if r.sanity_pass), None)
                    if status == "accepted"
                    else None,
                )
            )
        proxy = window_proxies[view_id_for(view)]
        run_decision = "run" if len(seeded) >= rules.min_parts_to_run else "skip"
        manifest = MultiviewSeedTransferManifest(
            manifest_kind="multiview_geometric_seed_transfer",
            view=view,
            view_id=view_id_for(view),
            analysis_frame_index=0,
            pose_frame_index=rig.pose_frame(view, plan.seed_frame.seed_frame),
            is_ego=False,
            reference_view=library_record.source_view,
            reference_run_manifest=library_record.truth_set,
            reference_seed_masks=library_record.exemplar_masks,
            secondary_reference_view=library_record.source_view,
            secondary_reference_pose_frame=0,
            secondary_reference_run_manifest=library_record.truth_set,
            secondary_reference_seed_masks=library_record.exemplar_masks[:1],
            transfer_method=(
                "exemplar_multiview_consistency: DINOv2-small exemplars of recording 1's human "
                "masks (the 'reference' fields name that truth set, not a run) rank SAM3 "
                "grid-box candidates at the seed frame; a part is accepted when >= 3 static "
                f"views triangulate within {CONSISTENCY_MAX_RAW_PX:.0f} raw px with radii in "
                f"band. analysis_frame_index 0 is proxy frame {plan.seed_frame.seed_frame} of "
                "the recording's fetched window (seed-window clip)."
            ),
            clip_config=_fingerprint(config_for_manifest, repository_root),
            proxy=ArtifactFingerprint(
                uri=proxy.proxy_uri, sha256=proxy.checksum_sha256, source="approved_config"
            ),
            proxy_dimensions=(proxy.dimensions.width, proxy.dimensions.height),
            proxy_to_raw_scale=proxy_to_raw_scale(view),
            table_plane=plane,
            rules=rules,
            parts=tuple(parts),
            decode_state="decoded",
            run_decision=run_decision,  # type: ignore[arg-type]
            run_decision_reason=(
                f"{len(seeded)} part(s) seeded ({', '.join(seeded) or 'none'}); "
                f"proposals only: {', '.join(proposal_parts) or 'none'}"
            ),
            provenance=EXEMPLAR_SEED_PROVENANCE,
            claim_boundaries=CLAIM_BOUNDARIES,
        )
        seed_dir = seeds_root / view
        seed_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = seed_dir / "seed_manifest.json"
        manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
        outcomes.append(
            ViewSeedOutcome(
                view=view,
                candidates_considered=len(candidates),
                seeded_parts=tuple(seeded),
                proposal_parts=tuple(proposal_parts),
                seed_manifest=_fingerprint(manifest_path, repository_root),
                run_decision=run_decision,
            )
        )

    report = ExemplarSeedReport(
        manifest_kind="exemplar_seed_report",
        recording=plan.recording,
        plan=_fingerprint(plan_path, repository_root),
        decode_result=_fingerprint(decode_path, repository_root),
        exemplar_library=library_record,
        seed_frame=plan.seed_frame.seed_frame,
        rec1_held_out_iou=REC1_HELD_OUT_IOU,
        seed_gate_iou=SEED_GATE_IOU,
        parts_used_as_seeds=PASSING_PARTS,
        consistency=tuple(results[p] for p in TARGETS),
        views=tuple(outcomes),
        elapsed_seconds=time.monotonic() - started,
        generated_at=datetime.now(UTC),
        claim_boundaries=CLAIM_BOUNDARIES,
    )
    path = output_dir / "seed_report.json"
    path.write_text(report.model_dump_json(indent=1) + "\n", encoding="utf-8")
    (output_dir / "seed_table.md").write_text(seed_table(report), encoding="utf-8")
    return path


def seed_table(report: ExemplarSeedReport) -> str:
    lines = [
        f"# Exemplar seeding on {report.recording}, seed frame {report.seed_frame} "
        "(agent-selected; review evidence, not ground truth)",
        "",
        "| part | rec-1 held-out IoU | gate | consistency | views | reprojection px | radius mm | "
        "similarity (min-max) | margin (min-max) | used as seed |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for result in report.consistency:
        gate = "pass" if report.rec1_held_out_iou[result.part] >= report.seed_gate_iou else "FAIL"
        sims = list(result.similarity.values())
        margins = list(result.margin.values())
        reproj = list(result.reprojection_px.values())
        radii = list(result.radius_mm.values())
        lines.append(
            f"| {result.part} | {report.rec1_held_out_iou[result.part]:.3f} | {gate} | "
            f"{'reached' if result.reached else 'BLOCKED'} | {len(result.views_used)} | "
            f"{(f'{min(reproj):.1f}-{max(reproj):.1f}' if reproj else '-')} | "
            f"{(f'{min(radii):.0f}-{max(radii):.0f}' if radii else '-')} | "
            f"{(f'{min(sims):.3f}-{max(sims):.3f}' if sims else '-')} | "
            f"{(f'{min(margins):+.3f}-{max(margins):+.3f}' if margins else '-')} | "
            f"{'yes' if result.reached and result.part in report.parts_used_as_seeds else 'no'} |"
        )
    lines += [
        "",
        "| view | candidates | seeded | proposals only | run decision |",
        "|---|---|---|---|---|",
    ]
    for view in report.views:
        lines.append(
            f"| {view.view} | {view.candidates_considered} | "
            f"{', '.join(view.seeded_parts) or '-'} | "
            f"{', '.join(view.proposal_parts) or '-'} | {view.run_decision} |"
        )
    lines += ["", *report.claim_boundaries, ""]
    return "\n".join(lines)


# -- CLI -----------------------------------------------------------------------------------


def main() -> None:
    from .muggled_smoke import MUGGLED_SAM_PYTHON

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--recording", default=DEFAULT_RECORDING)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="seed frame by hand joints, table grid prompts (CPU)")
    plan.add_argument("--seed-frame", type=int, default=None)
    plan.add_argument("--view", action="append", default=None)
    window = commands.add_parser(
        "window", help="trimmed proxies + derived clip config (CPU, ffmpeg)"
    )
    window.add_argument("--plan", type=Path, required=True)
    window.add_argument("--end-frame", type=int, required=True)
    window.add_argument("--config-output", type=Path, required=True)
    decode = commands.add_parser("decode", help="decode every grid prompt (GPU)")
    decode.add_argument("--plan", type=Path, required=True)
    decode.add_argument("--device", default="cuda:0")
    accept_cmd = commands.add_parser(
        "accept", help="rank, check consistency, write seeds/proposals"
    )
    accept_cmd.add_argument("--plan", type=Path, required=True)
    accept_cmd.add_argument("--decode-result", type=Path, default=None)
    accept_cmd.add_argument("--seed-window-config", type=Path, default=None)
    accept_cmd.add_argument("--exemplar-python", type=Path, default=MUGGLED_SAM_PYTHON)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    output = root / args.output_root
    recording = get_recording(args.recording, root)
    if args.command == "plan":
        path = plan_grid(
            root,
            recording=recording,
            output_dir=output,
            seed_frame=args.seed_frame,
            views=args.view,
        )
        loaded = load_plan(path)
        print(
            f"seed frame {loaded.seed_frame.seed_frame} (min joint height "
            f"{loaded.seed_frame.min_joint_height_mm:.0f} mm); {len(loaded.grid_points_mm)} grid "
            f"points; {loaded.prompt_count} prompts over {len(loaded.views)} views -> {path}"
        )
    elif args.command == "window":
        loaded = load_plan(args.plan)
        path = write_seed_window(
            root,
            recording=recording,
            seed_frame=loaded.seed_frame.seed_frame,
            end_frame_exclusive=args.end_frame,
            config_output=args.config_output.resolve(),
            derived_dir=root
            / recording.derived_root
            / f"seed_window_{loaded.seed_frame.seed_frame}",
        )
        print(f"-> {path}")
    elif args.command == "decode":
        loaded = load_plan(args.plan)
        path = run_decode(root, loaded, output_dir=output, device=args.device)
        print(f"-> {path}")
    else:
        path = accept(
            root,
            plan_path=args.plan.resolve(),
            decode_path=(args.decode_result or (output / "decode_result.json")).resolve(),
            output_dir=output,
            python=args.exemplar_python,
            seed_window_config=args.seed_window_config.resolve()
            if args.seed_window_config
            else None,
        )
        print((output / "seed_table.md").read_text(encoding="utf-8"))
        print(f"-> {path}")


if __name__ == "__main__":
    main()
