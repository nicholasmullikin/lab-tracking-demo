"""`battle-finebio-viewer`: the review recording of a FineBio trial window (`p5-viewer`).

One `.rrd` and three blueprint presets (`.rbl`), built from the outputs already on disk (the
tracking arms, the confidence and events passes, the rig check, the seeds, the detector pass,
the lane-B proxies); nothing is recomputed except the per-frame leave-one-view-out check the
slice module provides. The seven preflight cross-checks are standing entities on every build,
logged through `finebio_slice`'s functions:

1. `world/<view>/markers_projected` (marker reprojection per view, the fpv per frame);
2. `world/<view>/fpv_camera_centre` (the fpv camera centre in every fixed view);
3. `world/static_objects` + `world/<view>/static_reprojected` (static-object triangulation);
4. `world/left_hand`, `world/right_hand` + `world/<view>/<hand>_triangulated` (hands as probes);
5. `checks/clock_scan` (the rig's clock scan);
6. `checks/loo/<class>/<view>` (leave-one-view-out on the moving objects, per frame);
7. `checks/handoff/cell_culture_plate[_inside]` (fixed -> fpv hand-off residual per frame);

plus the negative control: camera 6's shipped pose as a second frustum (`world/T5_shipped`)
and a second marker reprojection set in T5 (`world/T5/markers_projected_shipped`), with the
rig's side-by-side numbers as text (`checks/negative_control`).

Presets: **World** (bench, markers, frusta, the moving fpv with its trail, every arm's 3D tracks
under `world/tracks/<arm>/<state>` with the default arm visible, trails, container volumes, the
centrifuge lid state, events and confidence series, the storyboard), **Cameras** (six tiles with
the proxy video as `AssetVideo` + `VideoFrameReference`, detector boxes, the default arm's SAM3
masks as RGBA cut-outs, track ids with abstain flags, seed boxes), **Evidence** (the cross-checks,
the negative control, scorecards, inventories, identity metrics, residual series). States are
entity paths (`world/tracks/<arm>/<state>`), so the tracker extensions' `held` / `contained`
rows render without a code change; `--tracks-dir tracks-ext` re-runs the build on them.

Storyboard frames are chosen from the data (the centrifuge cycle with the longest closed
containment, the longest fpv look-away from the plate, the six-view annotated frame for the
transparent plate, the largest confidence drop that a slot's mask-vs-box IoU confirms) and
marked on the timeline (`storyboard/marks`, `storyboard/index`, `storyboard/current`) and in
`storyboard.md`. Masks are logged every `--mask-every` frames (and on every storyboard frame) to
keep the recording near a gigabyte; the recording index records the strides. No viewer is
opened; `rerun rrd verify` / `stats` and the entity list are the checks.

FineBio is non-commercial research data: the recording holds video and masks and stays under
`runs/`; the `.rbl` presets hold only entity paths and are committed under `configs/rerun/`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from . import finebio_slice as slice_mod
from .finebio_arms import ClipWindow, load_clip
from .finebio_cameras import POSES, Camera, cameras_from_config, marker_points, read_camera_config
from .finebio_confidence import load_observation_signals
from .finebio_events import (
    PLATE_CLASS,
    Volume,
    container_volumes,
    cycles_in_window,
    lid_closed_at,
    lid_closed_intervals,
    median_box_sizes,
)
from .finebio_slice import (
    HAND_COLOURS,
    SliceSettings,
    class_colour,
    resolve_fpv_source,
    run_slice,
)
from .mask_cache import decode_mask_png, encode_rgba_mask_png
from .multiview_schemas import FINEBIO_FPV_VIEW, FineBioObservation
from .multiview_tracks import project

SCHEMA = "battle-finebio-viewer/1"
PRESETS = ("world", "cameras", "evidence")
RECORDING_NAME = "review.rrd"
DEFAULT_TRIALS = Path("configs/finebio/trials.json")
DEFAULT_PRESET_DIR = Path("configs/rerun")
FPS = 30000 / 1001
SECONDARY_STATES = ("coasting", "held", "contained", "single_view", "lost")
TRAIL_FRAMES = 30
TRAIL_STRIDE = 3
# Frames either side of the confidence-drop storyboard frame on which every arm's masks are logged.
STORY_MASK_HALO = 15
# The arms' known slot failures (the arms entry): preferred as the storyboard's confidence drop
# when they qualify, each within its own arm.
PREFERRED_FAILURE_SLOTS: dict[str, tuple[tuple[str, str], ...]] = {
    "c": (("T2", "50ml_tube#0"),),
    "d": (("T4", "50ml_tube#1"),),
}
LOO_CLASSES = ("cell_culture_plate", "blue_pipette", "left_hand", "right_hand")
CLAIM_BOUNDARY = (
    "Model output throughout: the FineBio DINO detector was trained on FineBio's own bench and "
    "cameras, so its boxes (and every mask prompted from them) are agreement between models, not "
    "accuracy; confidence ranks rows within an arm; events are geometry on 3D tracks; the human "
    "anchors of gate 2 (p6-anchors) are scored apart by battle-finebio-anchors and rank arms "
    "without making any of this accuracy."
)


# --------------------------------------------------------------------------- light rows


@dataclass(slots=True)
class TrackRow:
    frame: int
    track_id: str
    object_class: str
    position: tuple[float, float, float]
    uncertainty: float
    support_views: tuple[str, ...]
    support_slots: dict[str, str]
    state: str
    residual_px: dict[str, float]
    possibly_same_as: tuple[str, ...]
    container_id: str | None = None
    held_by: str | None = None
    group_size: int | None = None


def read_track_rows(path: Path) -> list[TrackRow]:
    """`tracks.jsonl` as light rows (four arms x 150k pydantic objects would not fit)."""
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append(
                TrackRow(
                    frame=int(r["frame_index"]),
                    track_id=r["track_id"],
                    object_class=r["object_class"],
                    position=tuple(float(x) for x in r["position_cm"]),
                    uncertainty=float(r.get("uncertainty_cm", 0.0)),
                    support_views=tuple(r.get("support_views", ())),
                    support_slots=dict(r.get("support_slots", {})),
                    state=r["state"],
                    residual_px={k: float(v) for k, v in (r.get("residual_px") or {}).items()},
                    possibly_same_as=tuple(r.get("possibly_same_as", ())),
                    container_id=r.get("container_id"),
                    held_by=r.get("held_by"),
                    group_size=r.get("group_size"),
                )
            )
    return rows


@dataclass
class ArmData:
    label: str
    arm_dir: Path
    tracks_dir: Path
    rows: list[TrackRow]
    by_frame: dict[int, list[TrackRow]]
    confidence: dict[tuple[str, int], tuple[float | None, bool]]
    confidence_md: str | None
    strip: dict[int, dict[str, Any]]
    episodes: list[dict[str, Any]]
    events_summary: dict[str, Any] | None
    events_md: str | None
    measures_md: str | None
    inventory_md: str | None
    identity: dict[str, Any] | None
    worker_runs: dict[str, Path]
    events_by_frame: dict[int, list[dict[str, Any]]]


def derived_dir_names(tracks_dir_name: str) -> tuple[str, str]:
    """The confidence and events output directories that belong to a tracks directory:
    `tracks` -> `confidence`, `events`; `tracks-ext` -> `confidence-ext`, `events-ext`."""
    tag = tracks_dir_name.removeprefix("tracks")
    return f"confidence{tag}", f"events{tag}"


def load_arm(label: str, arm_dir: Path, tracks_dir_name: str | None) -> ArmData:
    arm_dir = Path(arm_dir)
    if tracks_dir_name is not None:
        tracks_dir = arm_dir / tracks_dir_name
    elif (arm_dir / "tracks-ext" / "tracks.jsonl").is_file():
        tracks_dir = arm_dir / "tracks-ext"
    else:
        tracks_dir = arm_dir / "tracks"
    confidence_name, events_name = derived_dir_names(tracks_dir.name)
    confidence_dir = arm_dir / confidence_name
    events_dir = arm_dir / events_name
    rows = read_track_rows(tracks_dir / "tracks.jsonl")
    by_frame: dict[int, list[TrackRow]] = defaultdict(list)
    for r in rows:
        by_frame[r.frame].append(r)
    confidence: dict[tuple[str, int], tuple[float | None, bool]] = {}
    conf_path = confidence_dir / "confidence.jsonl"
    if conf_path.is_file():
        with conf_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    r = json.loads(line)
                    confidence[(r["track_id"], int(r["frame_index"]))] = (
                        r.get("confidence"),
                        bool(r.get("abstain", False)),
                    )
    strip: dict[int, dict[str, Any]] = {}
    strip_path = events_dir / "events_strip.jsonl"
    if strip_path.is_file():
        with strip_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    r = json.loads(line)
                    strip[int(r["frame_index"])] = r
    episodes = []
    ep_path = events_dir / "episodes.jsonl"
    if ep_path.is_file():
        episodes = [json.loads(line) for line in ep_path.read_text().splitlines() if line.strip()]
    events_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    ev_path = events_dir / "events.jsonl"
    if ev_path.is_file():
        for line in ev_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                events_by_frame[int(r["frame_index"])].append(r)
    worker_runs: dict[str, Path] = {}
    summary_path = arm_dir / "observations_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        worker_runs = {v: Path(p) for v, p in (summary.get("worker_runs") or {}).items()}

    def text(path: Path) -> str | None:
        return path.read_text(encoding="utf-8") if path.is_file() else None

    identity_path = tracks_dir / "identity_metrics.json"
    return ArmData(
        label=label,
        arm_dir=arm_dir,
        tracks_dir=tracks_dir,
        rows=rows,
        by_frame=dict(by_frame),
        confidence=confidence,
        confidence_md=text(confidence_dir / "confidence.md"),
        strip=strip,
        episodes=episodes,
        events_summary=(
            json.loads(text(events_dir / "events_summary.json") or "null")
            if (events_dir / "events_summary.json").is_file()
            else None
        ),
        events_md=text(events_dir / "events.md"),
        measures_md=text(arm_dir / "measures.md"),
        inventory_md=text(arm_dir / "occlusion_inventory.md"),
        identity=json.loads(text(identity_path) or "null") if identity_path.is_file() else None,
        worker_runs=worker_runs,
        events_by_frame=dict(events_by_frame),
    )


def parse_arm_dirs(spec: str) -> dict[str, Path]:
    """`a=<dir>,b=<dir>,...` -> {label: path}."""
    out: dict[str, Path] = {}
    for item in spec.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"--arm-dirs items are label=path, got {item!r}")
        label, path = item.split("=", 1)
        out[label.strip()] = Path(path.strip())
    if not out:
        raise ValueError("--arm-dirs names no arm")
    return out


# --------------------------------------------------------------------------- observations


@dataclass
class ViewObservations:
    """What the camera tiles draw per (view, frame): detector boxes and the SAM3 mask entries."""

    boxes: dict[tuple[str, int], list[tuple[str, float, tuple[float, float, float, float]]]]
    masks: dict[tuple[str, int], list[tuple[str, int, str]]]  # (slot label, mask index, class)
    plate_mask_views: dict[int, set[str]]
    pipette_boxes: dict[tuple[str, int], list[tuple[str, tuple[float, float]]]]


def load_view_observations(
    observations: Path, *, min_score: float = 0.3, frames: Iterable[int] | None = None
) -> ViewObservations:
    wanted = set(frames) if frames is not None else None
    boxes: dict[tuple[str, int], list] = defaultdict(list)
    masks: dict[tuple[str, int], list] = defaultdict(list)
    plate_views: dict[int, set[str]] = defaultdict(set)
    pipettes: dict[tuple[str, int], list] = defaultdict(list)
    with Path(observations).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            r = json.loads(line)
            frame = int(r["frame_index"])
            if wanted is not None and frame not in wanted:
                continue
            key = (r["view"], frame)
            if r["source"] == "detector":
                score = float(r.get("detector_score") or 0.0)
                if score < min_score:
                    continue
                box = tuple(float(x) for x in r["box_xyxy_px"])
                boxes[key].append((r["object_class"], score, box))
                if r["object_class"].endswith("_pipette"):
                    pipettes[key].append(
                        (r["object_class"], ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
                    )
            else:
                provenance = r.get("provenance") or {}
                object_id = str(provenance.get("object_id", ""))
                if object_id.startswith("sam3-") and provenance.get("mask") != "absent":
                    masks[key].append((r["slot"], int(object_id[5:]), r["object_class"]))
                    if r["object_class"] == PLATE_CLASS:
                        plate_views[frame].add(r["view"])
    return ViewObservations(dict(boxes), dict(masks), dict(plate_views), dict(pipettes))


def detector_rows_for_slice(
    observations: Path, classes: Sequence[str], *, min_score: float = 0.3
) -> list[FineBioObservation]:
    """The detector rows of a few classes as `FineBioObservation`s for the slice's LOO check."""
    wanted = set(classes)
    rows = []
    with Path(observations).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["source"] != "detector" or r["object_class"] not in wanted:
                continue
            if float(r.get("detector_score") or 0.0) < min_score:
                continue
            rows.append(FineBioObservation.model_validate(r))
    return rows


# --------------------------------------------------------------------------- storyboard


@dataclass
class StoryItem:
    index: int
    story: str
    title: str
    frame: int
    note: str
    entities: tuple[str, ...] = ()
    numbers: dict[str, Any] = field(default_factory=dict)

    def to_record(self, offset: int) -> dict[str, Any]:
        return {
            "index": self.index,
            "story": self.story,
            "title": self.title,
            "raw_frame": self.frame,
            "proxy_frame": self.frame - offset,
            "seconds": round(self.frame / FPS, 2),
            "note": self.note,
            "entities": list(self.entities),
            "numbers": self.numbers,
        }


def pick_centrifuge_story(arm: ArmData, cycles: Sequence[tuple[int, int]]) -> list[StoryItem]:
    """The cycle whose centrifuge `contained` episode spent the most frames inside while the
    lid was closed: entry, lid closes, inside while closed, lid opens, the id (or its
    successor) back."""
    table = (arm.events_summary or {}).get("centrifuge_cycles_vs_contained", [])
    best: tuple[int, dict[str, Any], dict[str, Any]] | None = None
    for cycle in table:
        for entry in cycle.get("contained_episodes", []):
            if entry["phase"] not in ("ended_while_closed", "through_the_cycle"):
                continue
            score = int(entry["inside_while_closed_frames"])
            if best is None or score > best[0]:
                best = (score, cycle, entry)
    if best is None:
        return []
    _, cycle, entry = best
    a, b = int(cycle["cycle"][0]), int(cycle["cycle"][1])
    track = entry["track_id"]
    end = int(entry["end_frame"])
    successors = entry.get("successors_after_opening", [])
    back_frame = int(successors[0]["frame_index"]) if successors else b + 2
    back_id = successors[0]["track_id"] if successors else track
    same = entry["phase"] == "through_the_cycle"
    entities = (
        f"world/tracks/{arm.label}/**",
        "world/containers/centrifuge",
        "world/containers/centrifuge_lid",
        f"events/{arm.label}/contained",
        "events/lid_closed",
    )
    nums = {
        "arm": arm.label,
        "track_id": track,
        "contained_start": entry["start_frame"],
        "contained_end": end,
        "end_reason": entry["end_reason"],
        "cycle": [a, b],
        "inside_while_closed_frames": entry["inside_while_closed_frames"],
        "successors": successors,
    }
    items = [
        StoryItem(
            0,
            "centrifuge",
            "a tube goes into the centrifuge",
            int(entry["start_frame"]),
            f"`{track}` ({entry['object_class']}) enters the centrifuge volume: `contained` "
            f"starts (arm {arm.label}).",
            entities,
            nums,
        ),
        StoryItem(
            0,
            "centrifuge",
            "the lid closes",
            a,
            f"lid state flips to closed at raw {a} (trials.json cycle [{a}, {b})); the tube is "
            "still `contained`.",
            entities,
            nums,
        ),
        StoryItem(
            0,
            "centrifuge",
            "inside while the lid is closed",
            min(a + max(1, (min(end, b - 1) - a) // 2), max(a, min(end, b - 1))),
            "every tile is empty for the tube; the 3D point sits inside the centrifuge box, "
            f"state `{'coasting' if not same else 'contained'}` on the core tracker rows "
            "(the extensions' `contained` state renders under `world/tracks/<arm>/contained`).",
            entities,
            nums,
        ),
        StoryItem(
            0,
            "centrifuge",
            "the lid opens",
            b,
            (
                f"the id `{track}` has survived the cycle."
                if same
                else f"`{track}` was lost at raw {end} ({entry['end_reason']}: the 30-frame coast "
                f"timeout is shorter than the {b - a}-frame closure); the volume is empty."
            ),
            entities,
            nums,
        ),
        StoryItem(
            0,
            "centrifuge",
            "the same id back" if same else "a successor id in the same place",
            back_frame,
            (
                f"`{track}` observed again inside the centrifuge."
                if same
                else f"`{back_id}` is born at raw {back_frame} where `{track}` was lost"
                + (
                    f" (possibly_same_as {', '.join(successors[0]['possibly_same_as'])})"
                    if successors and successors[0].get("possibly_same_as")
                    else ""
                )
                + "; identity across the closure is not claimed by the core tracker."
            ),
            entities,
            nums,
        ),
    ]
    return items


def pick_fpv_story(
    arm: ArmData,
    fpv_source,
    frames: Sequence[int],
    *,
    plate_class: str = PLATE_CLASS,
) -> list[StoryItem]:
    """The longest run of frames where the fpv pose is valid and the plate track projects
    outside the fpv image (the head camera looking away), bounded by frames where it is inside."""
    plate_by_frame: dict[int, TrackRow] = {}
    for f in frames:
        for r in arm.by_frame.get(f, []):
            if r.object_class == plate_class and r.state != "lost":
                plate_by_frame[f] = r
                break
    inside: dict[int, bool | None] = {}
    for f in frames:
        cam = fpv_source(f)
        row = plate_by_frame.get(f)
        if cam is None or row is None:
            inside[f] = None
            continue
        pixel = project(cam, np.asarray(row.position))
        inside[f] = bool(
            pixel is not None and 0 <= pixel[0] < cam.size[0] and 0 <= pixel[1] < cam.size[1]
        )
    best_run: tuple[int, int] | None = None
    start = None
    for f in frames:
        if inside[f] is False:
            if start is None:
                start = f
        else:
            if start is not None and inside[f] is True:
                if best_run is None or f - start > best_run[1] - best_run[0]:
                    best_run = (start, f)
            start = None
    if best_run is None:
        return []
    off, back = best_run
    before = off - 1
    ids = {plate_by_frame[f].track_id for f in range(before, back + 1) if f in plate_by_frame}
    reseeds = 0
    tracker_events = arm.tracks_dir / "events.jsonl"
    if tracker_events.is_file():
        for line in tracker_events.read_text().splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            if (
                e["kind"] in ("handoff_reseed", "detector_reseed")
                and off <= int(e["frame_index"]) < back
                and e.get("payload", {}).get("view") == FINEBIO_FPV_VIEW
                and e["track_id"] in ids
            ):
                reseeds += 1
    entities = (
        "world/fpv",
        "world/fpv_trail",
        f"world/tracks/{arm.label}/**",
        f"world/fpv/tracks/{arm.label}",
        "checks/handoff/**",
    )
    nums = {
        "arm": arm.label,
        "plate_track_ids": sorted(ids),
        "off_plate_frames": [off, back],
        "fpv_reseed_events_in_run": reseeds,
    }
    mid = off + (back - off) // 2
    return [
        StoryItem(
            0,
            "fpv",
            "the fpv still sees the plate",
            before,
            f"plate `{', '.join(sorted(ids))}` projects inside the fpv image; the fixed cameras "
            "and the fpv agree (hand-off residual series).",
            entities,
            nums,
        ),
        StoryItem(
            0,
            "fpv",
            "the fpv pans off the plate",
            mid,
            f"for raw frames [{off}, {back}) the plate projects outside the head camera; the id "
            f"is held by the fixed cameras alone ({reseeds} hand-off re-seed boxes for the fpv in "
            "the run: none are emitted while the projection is outside the image).",
            entities,
            nums,
        ),
        StoryItem(
            0,
            "fpv",
            "and back",
            back,
            "the plate is inside the fpv image again with the same id"
            + (
                " (one id across the run)." if len(ids) == 1 else f" ids in the run: {sorted(ids)}."
            ),
            entities,
            nums,
        ),
    ]


def pick_plate_story(
    observations: ViewObservations,
    arm: ArmData,
    fpv_source,
    frames: Sequence[int],
    *,
    views: Sequence[str],
    annotated: Sequence[int] = (),
) -> list[StoryItem]:
    """The transparent plate masked in all six views: the six-view annotated frame when it
    qualifies, else the qualifying frame where a pipette box sits nearest the plate in the fpv."""
    six = [f for f in frames if observations.plate_mask_views.get(f, set()) >= set(views)]
    if not six:
        return []
    choice = None
    reason = ""
    for f in annotated:
        if f in six:
            choice, reason = f, "the six-view annotated frame (the anchors' frame)"
            break
    nearest: tuple[float, int, str] | None = None
    for f in six:
        plate_row = next(
            (r for r in arm.by_frame.get(f, []) if r.object_class == PLATE_CLASS), None
        )
        cam = fpv_source(f)
        if plate_row is None or cam is None:
            continue
        pixel = project(cam, np.asarray(plate_row.position))
        if pixel is None:
            continue
        for cls, centre in observations.pipette_boxes.get((FINEBIO_FPV_VIEW, f), []):
            d = float(np.hypot(centre[0] - pixel[0], centre[1] - pixel[1]))
            if nearest is None or d < nearest[0]:
                nearest = (d, f, cls)
    if choice is None:
        if nearest is None:
            choice, reason = six[len(six) // 2], "a frame with the plate masked in six views"
        else:
            choice, reason = (
                nearest[1],
                (
                    f"the six-view frame with a pipette box nearest the plate in the fpv "
                    f"({nearest[2]}, {nearest[0]:.0f} px)"
                ),
            )
    entities = tuple(
        f"world/{v}/masks/{arm.label}/{slot_entity(PLATE_CLASS + '#0')}" for v in views
    ) + (f"world/*/tracks/{arm.label}",)
    return [
        StoryItem(
            0,
            "plate",
            "the transparent plate masked in six views",
            choice,
            f"{reason}; the plate has a SAM3 mask in all six views on {len(six)} of "
            f"{len(frames)} frames in arm {arm.label}"
            + (
                f"; nearest pipette box to the plate in the fpv at this frame: "
                f"{nearest[2]} ({nearest[0]:.0f} px)"
                if nearest is not None and nearest[1] == choice
                else ""
            )
            + ".",
            entities,
            {"arm": arm.label, "frames_with_six_plate_masks": len(six), "reason": reason},
        )
    ]


def pick_confidence_drop(
    arm: ArmData,
    signals: dict[tuple[str, int, str], dict[str, Any]],
    *,
    preferred_slots: Sequence[tuple[str, str]] = (),
    window: int = 15,
) -> list[StoryItem]:
    """The largest drop of a track's confidence (mean over `window` frames before vs after),
    shown to be a real failure by the supporting slot's mask-vs-box IoU falling with it.
    Tracks supported by a `preferred` (view, slot) are tried first."""
    by_track: dict[str, list[TrackRow]] = defaultdict(list)
    for r in arm.rows:
        by_track[r.track_id].append(r)
    candidates: list[tuple[float, str, int, str, str, float, float, float, float]] = []
    for track_id, rows in by_track.items():
        rows.sort(key=lambda r: r.frame)
        if len(rows) < 2 * window + 1:
            continue
        conf = np.array(
            [
                np.nan
                if arm.confidence.get((track_id, r.frame), (None, True))[0] is None
                else arm.confidence[(track_id, r.frame)][0]
                for r in rows
            ],
            dtype=np.float64,
        )
        if np.isfinite(conf).sum() < 2 * window:
            continue
        for i in range(window, len(rows) - window):
            before = conf[i - window : i]
            after = conf[i : i + window]
            if np.isfinite(before).sum() < window // 2 or np.isfinite(after).sum() < window // 2:
                continue
            drop = float(np.nanmean(before) - np.nanmean(after))
            if drop <= 0.15:
                continue
            # The views the track had in the frames before the drop: a real mask failure shows
            # as the slot's mask-vs-box IoU falling (a SAM3 row with no matching same-class box
            # at all is an IoU of 0: the mask's bbox overlaps no detector box of its class).
            views_before = {v for rr in rows[i - window : i] for v in rr.support_slots}
            for view in sorted(views_before):
                slot = next(
                    (
                        rr.support_slots[view]
                        for rr in reversed(rows[i - window : i])
                        if view in rr.support_slots
                    ),
                    "",
                )
                ib = _slot_ious(signals, rows[i - window : i], view)
                ia = _slot_ious(signals, rows[i : i + window], view)
                if len(ib) < 3 or len(ia) < 3:
                    continue
                iou_drop = float(np.mean(ib) - np.mean(ia))
                if iou_drop <= 0.2:
                    continue
                preferred = (view, slot) in preferred_slots
                candidates.append(
                    (
                        drop + (10.0 if preferred else 0.0) + iou_drop,
                        track_id,
                        rows[i].frame,
                        view,
                        slot,
                        float(np.nanmean(before)),
                        float(np.nanmean(after)),
                        float(np.mean(ib)),
                        float(np.mean(ia)),
                    )
                )
    if not candidates:
        return []
    candidates.sort(reverse=True)
    score, track_id, frame, view, slot, cb, ca, ib, ia = candidates[0]
    entities = (
        f"confidence/{arm.label}/tracks/{track_id}",
        f"world/{view}/masks/{arm.label}/{slot_entity(slot)}",
        f"world/{view}/tracks/{arm.label}",
        f"world/{view}/detector",
    )
    return [
        StoryItem(
            0,
            "confidence",
            "a confidence drop that is a real failure",
            frame,
            f"`{track_id}` (arm {arm.label}) falls from confidence {cb:.2f} to {ca:.2f} over "
            f"{window} frames either side of raw {frame}; its {view} slot `{slot}` mask-vs-box "
            f"IoU falls from {ib:.2f} to {ia:.2f}: the mask has left the detector's object.",
            entities,
            {
                "arm": arm.label,
                "track_id": track_id,
                "view": view,
                "slot": slot,
                "confidence_before": round(cb, 3),
                "confidence_after": round(ca, 3),
                "detector_box_iou_before": round(ib, 3),
                "detector_box_iou_after": round(ia, 3),
                "score": round(float(score), 3),
            },
        )
    ]


def _slot_ious(
    signals: dict[tuple[str, int, str], dict[str, Any]], rows: Sequence[TrackRow], view: str
) -> list[float]:
    """Mask-vs-box IoU of the rows' `view` slot; None on a SAM3 row counts as 0 (no same-class
    detector box overlaps the mask's bbox); detector rows carry no mask and are skipped."""
    out: list[float] = []
    for r in rows:
        slot = r.support_slots.get(view)
        if slot is None:
            continue
        sig = signals.get((view, r.frame, slot))
        if sig is None:
            continue
        iou = sig.get("detector_box_iou")
        if iou is None:
            if sig.get("source") in ("sam3_decode", "sam3_video"):
                out.append(0.0)
            continue
        out.append(float(iou))
    return out


def number_items(items: list[StoryItem]) -> list[StoryItem]:
    for i, item in enumerate(items, start=1):
        item.index = i
    return items


def storyboard_markdown(items: Sequence[StoryItem], *, offset: int, trial: str) -> str:
    lines = [
        f"# Storyboard, {trial} raw frames [{offset}, ...): frames marked on the `frame` timeline",
        "",
        "Raw frame = the shipped video's index; proxy frame = raw - "
        f"{offset} (the lane-B proxies and the worker masks); seconds from the trial start. "
        "Entities are the ones to look at in the recording (World / Cameras / Evidence presets).",
        "",
        "| # | story | title | raw | proxy | s | note |",
        "|---|---|---|---|---|---|---|",
    ]
    for it in items:
        lines.append(
            f"| {it.index} | {it.story} | {it.title} | {it.frame} | {it.frame - offset} | "
            f"{it.frame / FPS:.1f} | {it.note} |"
        )
    lines += ["", "Entities per item:", ""]
    for it in items:
        lines.append(f"- {it.index}: " + ", ".join(f"`{e}`" for e in it.entities))
    lines += ["", CLAIM_BOUNDARY, ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- logging


def slot_entity(slot: str) -> str:
    """A slot label as an entity path part: `cell_culture_plate#0` -> `cell_culture_plate-0`
    (`#` would be escaped in the path string)."""
    return slot.replace("#", "-")


def _dim(colour: Sequence[int], factor: float = 0.45) -> list[int]:
    return [int(128 + (c - 128) * factor) for c in colour]


def _state_colour(cls: str, state: str) -> list[int]:
    colour = class_colour(cls)
    return colour if state == "observed" else _dim(colour)


def _track_label(r: TrackRow, abstain: bool | None) -> str:
    label = r.track_id
    if r.state != "observed":
        label += f" [{r.state}]"
    if r.container_id:
        label += f" in {r.container_id}"
    if r.held_by:
        label += f" by {r.held_by}"
    if r.group_size:
        label += f" x{r.group_size}"
    if abstain:
        label += " !"
    return label


def _text(entity: str, text: str, *, markdown: bool = True) -> None:
    import rerun as rr

    rr.log(
        entity,
        rr.TextDocument(text, media_type=rr.MediaType.MARKDOWN if markdown else rr.MediaType.TEXT),
        static=True,
    )


def _send_scalars(entity: str, frames: Sequence[int], values: Sequence[float]) -> None:
    import rerun as rr

    frames_arr = np.asarray(frames, dtype=np.int64)
    rr.send_columns(
        entity,
        indexes=[
            rr.TimeColumn("frame", sequence=frames_arr),
            rr.TimeColumn("source_time", duration=frames_arr / FPS),
        ],
        columns=rr.Scalars.columns(scalars=np.asarray(values, dtype=np.float64)),
    )


def negative_control_text(rig: dict[str, Any]) -> str:
    block = rig.get("negative_control") or {}
    if not block:
        return "# Negative control\n\nNo `--negative-control` block in rig.json."
    lines = [
        "# Negative control: the shipped pose of every marker-PnP camera beside the pose in use",
        "",
        "A cross-check that never fails is not a check. Camera 6 (T5, top-down) does not fit its "
        "shipped extrinsics on this trial; the rig re-solves it from the bench markers. Below, "
        "the same seven-check numbers under both poses (`world/T5` is the marker-PnP frustum, "
        "`world/T5_shipped` the shipped one; `world/T5/markers_projected` green vs "
        "`markers_projected_shipped` red).",
        "",
        "| view | quantity | marker PnP | shipped |",
        "|---|---|---|---|",
    ]
    for view, nc in block.items():
        rms = nc.get("marker_rms_px", {})
        lines.append(
            f"| {view} | marker corner RMS px | {rms.get('marker_pnp', 0):.2f} | "
            f"{rms.get('shipped', 0):.2f} |"
        )
        centre = nc.get("centre_cm", {})
        lines.append(
            f"| {view} | camera centre cm | {centre.get('marker_pnp')} | {centre.get('shipped')} "
            f"(distance {centre.get('distance', 0):.2f} cm) |"
        )
        loo = nc.get("static_loo_px", {}).get("median", {})
        lines.append(
            f"| {view} | static LOO median px | {loo.get('marker_pnp', 0):.1f} | "
            f"{loo.get('shipped', 0):.1f} |"
        )
        for hand, values in nc.get("hands_px", {}).get("marker_pnp", {}).items():
            shipped = nc.get("hands_px", {}).get("shipped", {}).get(hand, {})
            lines.append(
                f"| {view} | {hand} residual median px | {values.get('median', 0):.1f} | "
                f"{shipped.get('median', 0):.1f} |"
            )
        per_object = nc.get("static_loo_px", {})
        pnp = per_object.get("marker_pnp", {})
        shipped_obj = per_object.get("shipped", {})
        for cls in pnp:
            lines.append(
                f"| {view} | LOO {cls} px | {pnp[cls]:.1f} | {shipped_obj.get(cls, 0):.1f} |"
            )
    return "\n".join(lines)


def clock_text(rig: dict[str, Any]) -> str:
    clock = rig.get("clock") or {}
    gates = (rig.get("gates") or {}).get("clock_offset_frames", {})
    lines = [
        "# Clock scan (this window)",
        "",
        "Best offset in frames of each fixed view against the other views' triangulation, "
        "-15..+15; a scan is informative when the best offset beats both +/-3 neighbours by 20%. "
        "Offsets inside +/-1 frame are reported and not applied.",
        "",
        "| view | class | best offset | residual at best px | residual at 0 px | informative | "
        "applied offset |",
        "|---|---|---|---|---|---|---|",
    ]
    for view, classes in clock.items():
        applied = gates.get(view, {}).get("offset_frames", 0)
        for cls, e in classes.items():
            lines.append(
                f"| {view} | {cls} | {e['best_offset']:+d} | {e['residual_px_at_best']:.1f} | "
                f"{e['residual_px_at_0']:.1f} | {'yes' if e.get('informative') else 'no'} | "
                f"{applied} |"
            )
    return "\n".join(lines)


def gates_text(rig: dict[str, Any]) -> str:
    gates = rig.get("gates") or {}
    inputs = gates.get("inputs", {})
    lines = [
        "# Gates (formulas evaluated on this window)",
        "",
        f"- association_px = {gates.get('association_px', 0):.1f} "
        f"({gates.get('formula', {}).get('association_px')}; static LOO median "
        f"{inputs.get('static_loo_median_px', 0):.2f} px over {inputs.get('static_loo_cells')} "
        "cells)",
        f"- handoff_px = {gates.get('handoff_px', 0):.1f} "
        f"({gates.get('formula', {}).get('handoff_px')}; per view p90 "
        + ", ".join(f"{v} {p:.1f}" for v, p in inputs.get("moving_loo_p90_px_by_view", {}).items())
        + ")",
        f"- floor {gates.get('floor_px')} px, cap {gates.get('cap_px')} px at "
        f"{gates.get('image_width_px')} px width",
        f"- birth: >= {gates.get('birth_min_fixed_views')} fixed views, or "
        f">= {gates.get('birth_fixed_views_with_fpv')} fixed views + the fpv with a valid pose",
        "",
        gates.get("formula", {}).get("floor_cap_rationale", ""),
    ]
    return "\n".join(lines)


def identity_text(arm: ArmData) -> str:
    m = arm.identity or {}
    if not m:
        return f"# Identity metrics, arm ({arm.label})\n\nnot available"
    lines = [
        f"# Identity metrics, arm ({arm.label})",
        "",
        f"tracks born {m.get('tracks_born')}, lost {m.get('tracks_lost')}, live at end "
        f"{m.get('tracks_live_at_end')}; re-acquisitions {m.get('reacquisitions')} (latency "
        f"median {(m.get('reacquisition_latency_frames') or {}).get('median')}); ambiguities "
        f"{m.get('ambiguities')}; fragmentation {m.get('fragmentation')}; slot disagreements "
        f"{m.get('slot_disagreements')}; id switches {m.get('id_switches')} "
        f"(reference `{m.get('id_switch_reference')}`).",
        "",
        "| class | born | fragmentation |",
        "|---|---|---|",
    ]
    for cls, e in sorted((m.get("per_class") or {}).items()):
        if isinstance(e, dict):
            lines.append(
                f"| {cls} | {e.get('tracks_born', e.get('born', '-'))} | "
                f"{e.get('fragmentation', '-')} |"
            )
    lines += ["", CLAIM_BOUNDARY]
    return "\n".join(lines)


@dataclass
class BuildInputs:
    clip: ClipWindow
    clip_doc: dict[str, Any]
    cams: dict[str, Camera]
    fpv_source: Any
    rig: dict[str, Any]
    arms: dict[str, ArmData]
    mask_arm: str
    secondary_mask_arm: str | None
    mask_every: int
    secondary_mask_every: int
    observations: dict[str, ViewObservations]
    volumes: list[Volume]
    lid_intervals: list[tuple[int, int]]
    cycles: list[tuple[int, int]]
    seeds: dict[str, Any] | None
    seeds_md: str | None
    scoreboard_md: str | None
    rig_md: str | None
    markers: np.ndarray | None
    shipped_t5: Camera | None
    storyboard: list[StoryItem]
    loo_rows: list[FineBioObservation]
    output: Path


@dataclass
class MaskSource:
    run_dir: Path
    offset: int
    cache: dict[str, bytes] = field(default_factory=dict)

    def png(self, frame: int, index: int, colour: tuple[int, int, int]) -> bytes | None:
        path = self.run_dir / "masks" / f"{frame - self.offset:06d}_{index:02d}.png"
        if not path.is_file():
            return None
        return encode_rgba_mask_png(decode_mask_png(path), colour)


def build_blueprint_world(inputs: BuildInputs):
    import rerun.blueprint as rrb

    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="world", name="World (cm, z down)"),
            rrb.Vertical(
                rrb.TimeSeriesView(origin="events", name="events strip (model output)"),
                rrb.TimeSeriesView(
                    origin=f"confidence/{inputs.mask_arm}",
                    name=f"confidence, arm {inputs.mask_arm}",
                ),
                rrb.Tabs(
                    rrb.TextDocumentView(origin="storyboard/current", name="storyboard"),
                    rrb.TextDocumentView(origin=f"events/{inputs.mask_arm}/active", name="events"),
                ),
            ),
            column_shares=[3, 2],
        ),
        rrb.TimePanel(timeline="frame", fps=FPS),
        collapse_panels=True,
    )


def log_recording(inputs: BuildInputs) -> dict[str, Any]:
    """Log everything; returns the index (entity counts, strides, timing)."""
    import rerun as rr

    from .rerun_logging import init_and_save, log_rgba_mask

    started = time.monotonic()
    clip = inputs.clip
    frames = list(clip.frames)
    offset = clip.start_frame
    fixed_views = list(inputs.cams)
    views = [*fixed_views, clip.fpv_view]
    rrd_path = inputs.output / RECORDING_NAME
    init_and_save(
        f"finebio-review-{clip.trial}", rrd_path, default_blueprint=build_blueprint_world(inputs)
    )
    # World frame, bench, markers, fixed frusta, cross-check 1 (fixed views).
    slice_mod.log_world_static(inputs.cams, inputs.markers)
    # Cross-check 3 from the rig's static points.
    static_pts = [np.asarray(s["point_cm"]) for s in inputs.rig.get("static", [])]
    static_labels = [f"{s['class']} {s['height_cm']:.1f} cm" for s in inputs.rig.get("static", [])]
    slice_mod.log_static_objects(static_pts, static_labels, inputs.cams)
    # Negative control: camera 6's shipped pose beside the marker-PnP one.
    if inputs.shipped_t5 is not None:
        slice_mod.log_camera_frustum(
            "world/T5_shipped", inputs.shipped_t5, image_plane_distance=15.0, static=True
        )
        if inputs.markers is not None:
            slice_mod.log_markers_projected(
                "world/T5/markers_projected_shipped",
                inputs.shipped_t5,
                inputs.markers,
                colour=[255, 60, 60],
                static=True,
            )
    # Cross-check 5 and the text evidence.
    slice_mod.log_clock_scan(clock_text(inputs.rig))
    _text("checks/negative_control", negative_control_text(inputs.rig))
    _text("checks/gates", gates_text(inputs.rig))
    if inputs.rig_md:
        _text("checks/rig", inputs.rig_md)
    if inputs.scoreboard_md:
        _text("checks/scoreboard", inputs.scoreboard_md)
    if inputs.seeds_md:
        _text("checks/seeds", inputs.seeds_md)
    _text("checks/claim_boundary", f"# Claim boundary\n\n{CLAIM_BOUNDARY}")
    for label, arm in inputs.arms.items():
        if arm.measures_md:
            _text(f"checks/measures/{label}", arm.measures_md)
        if arm.inventory_md:
            _text(f"checks/occlusion_inventory/{label}", arm.inventory_md)
        if arm.confidence_md:
            _text(f"checks/confidence/{label}", arm.confidence_md)
        if arm.events_md:
            _text(f"checks/events/{label}", arm.events_md)
        _text(f"checks/identity/{label}", identity_text(arm))
    # Storyboard.
    _text(
        "storyboard/guide",
        storyboard_markdown(inputs.storyboard, offset=offset, trial=clip.trial),
    )
    # Container volumes.
    for vol in inputs.volumes:
        rr.log(
            f"world/containers/{vol.name}",
            rr.Boxes3D(
                centers=[[vol.centre_xy[0], vol.centre_xy[1], (vol.z_top + vol.z_bottom) / 2]],
                half_sizes=[[vol.half_x_cm, vol.half_y_cm, (vol.z_bottom - vol.z_top) / 2]],
                labels=[vol.name],
                colors=[_dim(class_colour(vol.object_class), 0.7)],
            ),
            static=True,
        )
    # Videos: the proxies as assets (proxy frame k == raw offset + k), one frame reference per
    # frame in the loop below at the asset's own frame timestamps.
    video_frames: dict[str, int] = {}
    video_stamps: dict[str, np.ndarray] = {}
    for view in views:
        proxy = clip.proxies.get(view)
        if proxy is None or not Path(proxy).is_file():
            continue
        asset = rr.AssetVideo(path=str(proxy))
        rr.log(f"world/{view}/video_asset", asset, static=True)
        stamps = np.asarray(asset.read_frame_timestamps_nanos(), dtype=np.int64)
        video_stamps[view] = stamps
        video_frames[view] = int(min(len(stamps), len(frames)))
    # Per-frame scalar series through send_columns.
    _log_series(inputs, frames)
    # Cross-checks 6 and 7 per frame from the slice on the default arm's detector rows.
    _log_loo_and_handoff(inputs, frames)
    # Per-frame loop.
    story_frames = set()
    for it in inputs.storyboard:
        story_frames.add(it.frame)
        if it.story == "confidence":
            story_frames.update(range(it.frame - STORY_MASK_HALO, it.frame + STORY_MASK_HALO + 1))
    mask_sources = _mask_sources(inputs, offset)
    seed_boxes = _seed_boxes(inputs.seeds)
    trail: list[np.ndarray] = []
    history: dict[str, dict[str, deque]] = {label: {} for label in inputs.arms}
    seen_states: dict[str, set[str]] = {label: set() for label in inputs.arms}
    seen_2d: set[str] = set()
    hands_seen: set[str] = set()
    rig_hands = {
        hand: {int(e["frame"]): e for e in entries}
        for hand, entries in (inputs.rig.get("hands") or {}).items()
    }
    lid_state: bool | None = None
    mask_count = 0
    story_by_frame: dict[int, list[StoryItem]] = defaultdict(list)
    for it in inputs.storyboard:
        story_by_frame[it.frame].append(it)
    default_obs = inputs.observations[inputs.mask_arm]
    for frame in frames:
        rr.set_time("frame", sequence=frame)
        rr.set_time("source_time", duration=frame / FPS)
        for view, stamps in video_stamps.items():
            k = frame - offset
            if 0 <= k < len(stamps):
                rr.log(
                    f"world/{view}/video",
                    rr.VideoFrameReference(
                        nanoseconds=int(stamps[k]), video_reference=f"world/{view}/video_asset"
                    ),
                )
        fpv_cam = inputs.fpv_source(frame)
        all_cams = dict(inputs.cams)
        if fpv_cam is not None:
            all_cams[clip.fpv_view] = fpv_cam
            slice_mod.log_fpv_frame(fpv_cam, inputs.cams, inputs.markers, trail)
        else:
            slice_mod.clear_fpv_frame(inputs.cams)
        # Cross-check 4: the rig's per-frame hand triangulations.
        for hand, colour in HAND_COLOURS:
            entry = rig_hands.get(hand, {}).get(frame)
            if entry is None:
                if hand in hands_seen:
                    slice_mod.clear_hand_probe(hand, inputs.cams)
                continue
            hands_seen.add(hand)
            slice_mod.log_hand_probe(hand, entry["point_cm"], colour, inputs.cams)
        # Lid state.
        if inputs.lid_intervals:
            closed = lid_closed_at(inputs.lid_intervals, frame)
            if closed != lid_state:
                lid_state = closed
                centrifuge = next((v for v in inputs.volumes if v.name == "centrifuge"), None)
                if centrifuge is not None:
                    rr.log(
                        "world/containers/centrifuge_lid",
                        rr.Boxes3D(
                            centers=[
                                [centrifuge.centre_xy[0], centrifuge.centre_xy[1], centrifuge.z_top]
                            ],
                            half_sizes=[[centrifuge.half_x_cm, centrifuge.half_y_cm, 0.5]],
                            labels=["lid closed" if closed else "lid open"],
                            colors=[[230, 60, 60] if closed else [60, 200, 90]],
                            fill_mode=(
                                rr.components.FillMode.Solid
                                if closed
                                else rr.components.FillMode.MajorWireframe
                            ),
                        ),
                    )
        # Detector boxes per view (the default arm's detector rows).
        for view in views:
            boxes = default_obs.boxes.get((view, frame), [])
            entity = f"world/{view}/detector"
            if not boxes:
                if entity in seen_2d:
                    rr.log(entity, rr.Clear(recursive=False))
                continue
            seen_2d.add(entity)
            rr.log(
                entity,
                rr.Boxes2D(
                    array=np.array([b[2] for b in boxes]),
                    array_format=rr.Box2DFormat.XYXY,
                    labels=[f"{b[0]} {b[1]:.2f}" for b in boxes],
                    colors=[class_colour(b[0]) for b in boxes],
                    radii=1.0,
                ),
            )
            seeds_here = seed_boxes.get((view, frame))
            if seeds_here:
                rr.log(
                    f"world/{view}/seeds",
                    rr.Boxes2D(
                        array=np.array([s[1] for s in seeds_here]),
                        array_format=rr.Box2DFormat.XYXY,
                        labels=[f"seed {s[0]}" for s in seeds_here],
                        colors=[[255, 255, 255]],
                        radii=2.5,
                    ),
                )
        # Masks.
        for label, source_by_view in mask_sources.items():
            if label == inputs.mask_arm:
                every: int | None = inputs.mask_every
            elif label == inputs.secondary_mask_arm:
                every = inputs.secondary_mask_every
            else:
                every = None
            if (every is None or (frame - offset) % every != 0) and frame not in story_frames:
                continue
            obs = inputs.observations[label]
            for view, source in source_by_view.items():
                for slot, index, cls in obs.masks.get((view, frame), []):
                    png = source.png(frame, index, tuple(class_colour(cls)))
                    if png is None:
                        continue
                    log_rgba_mask(
                        f"world/{view}/masks/{label}/{slot_entity(slot)}", png, opacity=0.45
                    )
                    mask_count += 1
        # Tracks: 3D per arm and state, 2D per view, trails.
        for label, arm in inputs.arms.items():
            rows = [r for r in arm.by_frame.get(frame, []) if r.state != "lost"]
            by_state: dict[str, list[TrackRow]] = defaultdict(list)
            for r in rows:
                by_state[r.state].append(r)
            for state in set(by_state) | seen_states[label]:
                entity = f"world/tracks/{label}/{state}"
                state_rows = by_state.get(state, [])
                if not state_rows:
                    rr.log(entity, rr.Clear(recursive=False))
                    continue
                seen_states[label].add(state)
                rr.log(
                    entity,
                    rr.Points3D(
                        np.array([r.position for r in state_rows]),
                        colors=[_state_colour(r.object_class, r.state) for r in state_rows],
                        radii=[
                            float(np.clip(0.4 + 0.15 * r.uncertainty, 0.4, 2.5)) for r in state_rows
                        ],
                        labels=[
                            _track_label(
                                r, arm.confidence.get((r.track_id, frame), (None, None))[1]
                            )
                            for r in state_rows
                        ],
                    ),
                )
            # Trails for the default arm, 2D projections for the arms whose masks are logged:
            # the other arms' 3D points stay (toggle them in the World preset), which keeps
            # the recording near the size budget.
            if label == inputs.mask_arm:
                strips, strip_colours = [], []
                live_ids = set()
                for r in rows:
                    live_ids.add(r.track_id)
                    h = history[label].setdefault(r.track_id, deque(maxlen=TRAIL_FRAMES))
                    h.append((frame, np.asarray(r.position)))
                    pts = [p for i, (_, p) in enumerate(h) if (len(h) - 1 - i) % TRAIL_STRIDE == 0]
                    if len(pts) >= 2:
                        strips.append(np.array(pts))
                        strip_colours.append(_dim(class_colour(r.object_class), 0.8))
                for tid in [t for t in history[label] if t not in live_ids]:
                    del history[label][tid]
                trails_entity = f"world/tracks/{label}/trails"
                if strips:
                    rr.log(trails_entity, rr.LineStrips3D(strips, colors=strip_colours, radii=0.15))
                    seen_2d.add(trails_entity)
                elif trails_entity in seen_2d:
                    rr.log(trails_entity, rr.Clear(recursive=False))
            if label not in (inputs.mask_arm, inputs.secondary_mask_arm):
                continue
            for view in views:
                entity = f"world/{view}/tracks/{label}"
                cam = all_cams.get(view)
                positions, labels, colours = [], [], []
                for r in rows if cam is not None else []:
                    pixel = project(cam, np.asarray(r.position))
                    if pixel is None or not (
                        0 <= pixel[0] < cam.size[0] and 0 <= pixel[1] < cam.size[1]
                    ):
                        continue
                    positions.append(pixel)
                    labels.append(
                        _track_label(r, arm.confidence.get((r.track_id, frame), (None, None))[1])
                    )
                    colours.append(_state_colour(r.object_class, r.state))
                if positions:
                    rr.log(
                        entity,
                        rr.Points2D(np.array(positions), labels=labels, colors=colours, radii=5.0),
                    )
                    seen_2d.add(entity)
                elif entity in seen_2d:
                    rr.log(entity, rr.Clear(recursive=False))
        # Events: the default arm's active relations as text and the log of transitions.
        for label, arm in inputs.arms.items():
            for e in arm.events_by_frame.get(frame, []):
                payload = e.get("payload", {})
                rr.log(
                    f"events/{label}/log",
                    rr.TextLog(
                        f"{e['kind']} {payload.get('phase')}: {e['track_id']} "
                        f"({payload.get('object_class')}) -> {payload.get('target')}"
                        + (
                            f", {payload.get('duration_frames')} frames, "
                            f"{payload.get('end_reason')}"
                            if payload.get("phase") == "end"
                            else ""
                        ),
                        level="INFO" if payload.get("phase") == "start" else "DEBUG",
                    ),
                )
        strip_row = inputs.arms[inputs.mask_arm].strip.get(frame)
        if strip_row is not None:
            rr.log(
                f"events/{inputs.mask_arm}/active",
                rr.TextDocument(
                    _active_text(strip_row, frame, inputs.mask_arm),
                    media_type=rr.MediaType.MARKDOWN,
                ),
            )
        # Storyboard marks.
        for it in story_by_frame.get(frame, []):
            rr.log(
                "storyboard/marks",
                rr.TextLog(f"S{it.index} {it.story}: {it.title} (raw {frame})", level="INFO"),
            )
            rr.log("storyboard/index", rr.Scalars(float(it.index)))
            rr.log(
                "storyboard/current",
                rr.TextDocument(
                    f"# S{it.index}: {it.title}\n\nraw {frame}, proxy {frame - offset}, "
                    f"{frame / FPS:.1f} s ({it.story})\n\n{it.note}\n\n"
                    + "\n".join(f"- `{e}`" for e in it.entities),
                    media_type=rr.MediaType.MARKDOWN,
                ),
            )
    rr.disconnect()
    elapsed = time.monotonic() - started
    return {
        "recording": rrd_path.as_posix(),
        "size_bytes": rrd_path.stat().st_size,
        "frames": len(frames),
        "video_frames_referenced": video_frames,
        "masks_logged": mask_count,
        "mask_every": {inputs.mask_arm: inputs.mask_every},
        "secondary_mask_every": (
            {inputs.secondary_mask_arm: inputs.secondary_mask_every}
            if inputs.secondary_mask_arm
            else {}
        ),
        "build_seconds": round(elapsed, 1),
    }


def _active_text(strip_row: dict[str, Any], frame: int, arm: str) -> str:
    lines = [f"# Active relations, arm {arm}, raw {frame} (model output)", ""]
    lid = strip_row.get("lid_closed")
    if lid is not None:
        lines.append(f"- centrifuge lid: **{'closed' if lid else 'open'}**")
    for kind in ("contained", "held", "proximity"):
        items = strip_row.get(kind) or []
        if not items:
            continue
        for item in items:
            target = item.get("target") or item.get("hand")
            lines.append(f"- `{kind}`: {item['track_id']} ({item['object_class']}) -> {target}")
    if len(lines) == 2 or (len(lines) == 3 and lid is not None):
        lines.append("- no open episode")
    return "\n".join(lines)


def _mask_sources(inputs: BuildInputs, offset: int) -> dict[str, dict[str, MaskSource]]:
    """Every arm with worker masks on disk: the default arm at `mask_every`, the secondary arm
    at `secondary_mask_every`, the others on the storyboard frames only."""
    out: dict[str, dict[str, MaskSource]] = {}
    for label, arm in inputs.arms.items():
        sources = {}
        for view, run_dir in arm.worker_runs.items():
            if (run_dir / "masks").is_dir():
                sources[view] = MaskSource(run_dir, offset)
        if sources and label in inputs.observations:
            out[label] = sources
    return out


def _seed_boxes(
    seeds: dict[str, Any] | None,
) -> dict[tuple[str, int], list[tuple[str, tuple[float, float, float, float]]]]:
    out: dict[tuple[str, int], list] = defaultdict(list)
    if not seeds:
        return {}
    for view, block in (seeds.get("views") or {}).items():
        for slot in block.get("slots", []):
            seed = slot.get("seed") or {}
            if slot.get("status") == "accepted" and seed.get("frame") is not None:
                out[(view, int(seed["frame"]))].append(
                    (slot["label"], tuple(float(x) for x in seed["box"]))
                )
    return dict(out)


def _log_series(inputs: BuildInputs, frames: Sequence[int]) -> None:
    """Confidence per class and arm, abstain fraction, event counts, lid state, residuals per
    view, coasting-track counts; each one `send_columns` call."""
    frame_index = {f: i for i, f in enumerate(frames)}
    n = len(frames)
    if inputs.lid_intervals:
        _send_scalars(
            "events/lid_closed",
            frames,
            [1.0 if lid_closed_at(inputs.lid_intervals, f) else 0.0 for f in frames],
        )
    story_tracks = {
        (it.numbers.get("track_id"), it.numbers.get("arm"))
        for it in inputs.storyboard
        if it.numbers.get("track_id")
    }
    for label, arm in inputs.arms.items():
        per_class: dict[str, list[list[float]]] = defaultdict(lambda: [[] for _ in range(n)])
        abstain = np.zeros(n)
        total = np.zeros(n)
        coasting = np.zeros(n)
        residual: dict[str, list[list[float]]] = defaultdict(lambda: [[] for _ in range(n)])
        for r in arm.rows:
            i = frame_index.get(r.frame)
            if i is None or r.state == "lost":
                continue
            conf, abst = arm.confidence.get((r.track_id, r.frame), (None, None))
            if conf is not None:
                per_class[r.object_class][i].append(conf)
            if abst is not None:
                total[i] += 1
                abstain[i] += float(abst)
            if r.state == "coasting":
                coasting[i] += 1
            for view, px in r.residual_px.items():
                residual[view][i].append(px)
        for cls, lists in sorted(per_class.items()):
            values = np.array([np.mean(v) if v else np.nan for v in lists])
            keep = np.isfinite(values)
            if keep.any():
                _send_scalars(
                    f"confidence/{label}/{cls}", np.asarray(frames)[keep].tolist(), values[keep]
                )
        with np.errstate(invalid="ignore", divide="ignore"):
            fraction = np.where(total > 0, abstain / np.maximum(total, 1), np.nan)
        keep = np.isfinite(fraction)
        if keep.any():
            _send_scalars(
                f"confidence/{label}/abstain_fraction",
                np.asarray(frames)[keep].tolist(),
                fraction[keep],
            )
        _send_scalars(f"checks/occlusion/{label}/coasting_tracks", frames, coasting)
        for view, lists in sorted(residual.items()):
            values = np.array([np.median(v) if v else np.nan for v in lists])
            keep = np.isfinite(values)
            if keep.any():
                _send_scalars(
                    f"checks/residuals/{label}/{view}",
                    np.asarray(frames)[keep].tolist(),
                    values[keep],
                )
        for kind in ("contained", "held", "proximity"):
            counts = [float(len((arm.strip.get(f) or {}).get(kind) or [])) for f in frames]
            if arm.strip:
                _send_scalars(f"events/{label}/{kind}", frames, counts)
        for track_id, story_arm in story_tracks:
            if story_arm != label:
                continue
            series = [
                (f, arm.confidence[(track_id, f)][0])
                for f in frames
                if (track_id, f) in arm.confidence and arm.confidence[(track_id, f)][0] is not None
            ]
            if series:
                _send_scalars(
                    f"confidence/{label}/tracks/{track_id}",
                    [f for f, _ in series],
                    [c for _, c in series],
                )


def _log_loo_and_handoff(inputs: BuildInputs, frames: Sequence[int]) -> None:
    """Cross-checks 6 and 7 per frame: the slice's per-frame triangulation of the moving classes
    (LOO per view) and the plate's fixed -> fpv hand-off residual, from the rig's fpv block."""
    fpv_block = inputs.rig.get("fpv") or []
    handoff = [
        (int(e["frame"]), e["plate_fixed_to_fpv"])
        for e in fpv_block
        if e.get("plate_fixed_to_fpv")
        and e.get("plate_fixed_to_fpv", {}).get("residual_px") is not None
    ]
    if handoff:
        _send_scalars(
            f"checks/handoff/{PLATE_CLASS}",
            [f for f, _ in handoff],
            [float(h["residual_px"]) for _, h in handoff],
        )
        _send_scalars(
            f"checks/handoff/{PLATE_CLASS}_inside",
            [f for f, _ in handoff],
            [1.0 if h.get("inside_box") else 0.0 for _, h in handoff],
        )
    if not inputs.loo_rows:
        return
    settings = SliceSettings(fixed_views=tuple(inputs.cams), source_sets=("detector",))
    result = run_slice(inputs.loo_rows, inputs.cams, inputs.fpv_source, frames, settings)
    series: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for p in result.points:
        for view, value in p.loo_px.items():
            series[(p.object_class, view)].append((p.frame_index, value))
    for (cls, view), values in sorted(series.items()):
        _send_scalars(f"checks/loo/{cls}/{view}", [f for f, _ in values], [v for _, v in values])


# --------------------------------------------------------------------------- presets


def write_presets(inputs: BuildInputs, index: dict[str, Any]) -> dict[str, Any]:
    """The three `.rbl` presets beside the recording (and copies under `configs/rerun/`),
    validated against the recording's entity tree with `review_presets`' checker."""
    import rerun.blueprint as rrb

    from . import review_presets

    rrd_path = inputs.output / RECORDING_NAME
    facts = review_presets.read_recording_facts(rrd_path)
    paths = set(facts.entity_paths)
    fixed_views = list(inputs.cams)
    views = [*fixed_views, inputs.clip.fpv_view]
    arm = inputs.mask_arm
    other_arms = [a for a in inputs.arms if a != arm]

    def hidden(prefixes: Iterable[str]) -> dict[str, Any]:
        out = {}
        for prefix in prefixes:
            for path in paths:
                if path == f"/{prefix}" or path.startswith(f"/{prefix}/"):
                    out[path] = rrb.EntityBehavior(visible=False)
        return out

    def present(*candidates: str) -> list[str]:
        return [c for c in candidates if f"/{c}" in paths]

    time_panel = rrb.TimePanel(timeline="frame", fps=FPS)
    # World: the 3D view with every arm logged and the default arm visible.
    world_excludes = []
    for v in views:
        world_excludes += [f"- /world/{v}/video", f"- /world/{v}/detector", f"- /world/{v}/seeds"]
        world_excludes += [f"- /world/{v}/masks/**", f"- /world/{v}/tracks/**"]
    world_view = rrb.Spatial3DView(
        origin="world",
        name=f"World: cm, z down; tracks of arm ({arm}) shown, other arms hidden",
        contents=["+ /world/**", *world_excludes],
        overrides=hidden([f"world/tracks/{a}" for a in other_arms] + ["world/T5_shipped"]),
    )
    conf_series = rrb.TimeSeriesView(
        origin=f"confidence/{arm}",
        name=f"confidence per class, abstain fraction, storyboard tracks (arm {arm})",
    )
    events_series = rrb.TimeSeriesView(
        origin="events",
        name="events strip: active contained / held / proximity per arm, lid closed (model output)",
        contents=["+ /events/lid_closed"]
        + [f"+ /events/{a}/{k}" for a in inputs.arms for k in ("contained", "held", "proximity")],
        overrides=hidden([f"events/{a}" for a in other_arms]),
    )
    story_tabs = rrb.Tabs(
        *[
            rrb.TextDocumentView(origin=path, name=name)
            for path, name in (
                ("storyboard/current", "storyboard: current frame"),
                (f"events/{arm}/active", f"events active (arm {arm})"),
                ("storyboard/guide", "storyboard guide"),
            )
            if f"/{path}" in paths
        ],
        *(
            [rrb.TextLogView(origin="storyboard/marks", name="storyboard marks")]
            if "/storyboard/marks" in paths
            else []
        ),
        *(
            [rrb.TextLogView(origin=f"events/{arm}/log", name=f"event log (arm {arm})")]
            if f"/events/{arm}/log" in paths
            else []
        ),
    )
    world = rrb.Blueprint(
        rrb.Horizontal(
            world_view,
            rrb.Vertical(events_series, conf_series, story_tabs, row_shares=[1, 1, 1]),
            column_shares=[3, 2],
        ),
        time_panel,
        auto_layout=False,
        auto_views=False,
    )
    # Cameras: six tiles with the video, boxes, the default arm's masks and track ids.
    tiles = []
    for v in views:
        contents = [
            f"+ /{p}"
            for p in present(f"world/{v}/video", f"world/{v}/seeds", f"world/{v}/detector")
        ]
        contents += [
            f"+ /world/{v}/{sub}/**"
            for sub in ("masks", "tracks")
            if any(p.startswith(f"/world/{v}/{sub}/") for p in paths)
        ]
        size = inputs.clip_doc.get("view_sizes", {}).get(v, [1920, 1080])
        tiles.append(
            rrb.Spatial2DView(
                origin=f"world/{v}",
                name=f"{v}: video, detector boxes, SAM3 masks (arm {arm}), track ids (! = abstain)",
                contents=contents,
                overrides=hidden(
                    [f"world/{v}/masks/{a}" for a in other_arms]
                    + [f"world/{v}/tracks/{a}" for a in other_arms]
                ),
                visual_bounds=rrb.VisualBounds2D(x_range=[0, size[0]], y_range=[0, size[1]]),
            )
        )
    cameras = rrb.Blueprint(
        rrb.Vertical(
            rrb.Grid(*tiles, grid_columns=3),
            rrb.Horizontal(conf_series, events_series),
            row_shares=[4, 1],
        ),
        time_panel,
        auto_layout=False,
        auto_views=False,
    )
    # Evidence: the cross-checks, the negative control and the documents.
    evidence_excludes = []
    for v in views:
        evidence_excludes += [
            f"- /world/{v}/video",
            f"- /world/{v}/detector",
            f"- /world/{v}/seeds",
        ]
        evidence_excludes += [f"- /world/{v}/masks/**", f"- /world/{v}/tracks/**"]
    evidence_excludes += ["- /world/tracks/**", "- /world/containers/**"]
    rig_view = rrb.Spatial3DView(
        origin="world",
        name="Rig: frusta (T5 marker PnP and T5_shipped), static objects, hands, markers, fpv",
        contents=["+ /world/**", *evidence_excludes],
    )
    t5_contents = [
        f"+ /world/T5/{name}"
        for name in (
            "video",
            "markers_projected",
            "markers_projected_shipped",
            "static_reprojected",
            "fpv_camera_centre",
            "left_hand_triangulated",
            "right_hand_triangulated",
        )
        if f"/world/T5/{name}" in paths
    ]
    t5_view = rrb.Spatial2DView(
        origin="world/T5",
        name="T5 (camera 6): markers through the marker-PnP pose (green) and the shipped pose "
        "(red), static objects, hands, fpv centre",
        contents=t5_contents,
    )
    docs = [
        rrb.TextDocumentView(origin=path, name=name)
        for path, name in (
            ("checks/negative_control", "negative control"),
            ("checks/clock_scan", "clock scan"),
            ("checks/gates", "gates"),
            ("checks/scoreboard", "scoreboard"),
            *((f"checks/measures/{a}", f"measures ({a})") for a in inputs.arms),
            *(
                (f"checks/occlusion_inventory/{a}", f"occlusion inventory ({a})")
                for a in inputs.arms
            ),
            *((f"checks/identity/{a}", f"identity metrics ({a})") for a in inputs.arms),
            *((f"checks/confidence/{a}", f"confidence ({a})") for a in inputs.arms),
            *((f"checks/events/{a}", f"events ({a})") for a in inputs.arms),
            ("checks/seeds", "seeds"),
            ("checks/rig", "rig report"),
            ("checks/claim_boundary", "claim boundary"),
        )
        if f"/{path}" in paths
    ]
    series = []
    if any(p.startswith("/checks/loo/") for p in paths):
        series.append(
            rrb.TimeSeriesView(
                origin="checks/loo", name="leave-one-view-out residual px per class and view"
            )
        )
    if present(f"checks/handoff/{PLATE_CLASS}"):
        series.append(
            rrb.TimeSeriesView(
                origin="checks/handoff", name="fixed -> fpv hand-off px (plate) and inside flag"
            )
        )
    if any(p.startswith("/checks/residuals/") for p in paths):
        series.append(
            rrb.TimeSeriesView(
                origin="checks/residuals",
                name="cross-view residual px per arm and view (median over live tracks)",
                overrides=hidden([f"checks/residuals/{a}" for a in other_arms]),
            )
        )
    if any(p.startswith("/checks/occlusion/") for p in paths):
        series.append(rrb.TimeSeriesView(origin="checks/occlusion", name="coasting tracks per arm"))
    evidence = rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(rig_view, t5_view, rrb.Tabs(*docs), column_shares=[2, 2, 3]),
            rrb.Horizontal(*series) if series else rrb.Tabs(*docs[:1]),
            row_shares=[3, 2],
        ),
        time_panel,
        auto_layout=False,
        auto_views=False,
    )
    results = []
    written = {}
    for name, blueprint in (("world", world), ("cameras", cameras), ("evidence", evidence)):
        path = inputs.output / f"{name}.rbl"
        blueprint.save(facts.application_id, path)
        results.append(review_presets.check_preset(path, facts))
        written[name] = path.as_posix()
    report = {
        "recording": rrd_path.as_posix(),
        "application_id": facts.application_id,
        "recording_id": facts.recording_id,
        "entity_paths": len(facts.entity_paths),
        "presets": results,
        "ok": all(bool(r["ok"]) for r in results),
        "written": written,
        "open_command": (
            f"uv run rerun {rrd_path.as_posix()} {(inputs.output / 'world.rbl').as_posix()}"
        ),
    }
    (inputs.output / review_presets.PRESET_CHECK_NAME).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def entity_summary(rrd_path: Path) -> dict[str, Any]:
    from . import review_presets

    facts = review_presets.read_recording_facts(rrd_path)
    groups: Counter[str] = Counter()
    for path in facts.entity_paths:
        parts = path.strip("/").split("/")
        groups["/".join(parts[:2]) if len(parts) > 1 else parts[0]] += 1
    return {
        "entity_paths": len(facts.entity_paths),
        "application_id": facts.application_id,
        "recording_id": facts.recording_id,
        "by_group": dict(sorted(groups.items())),
        "paths": list(facts.entity_paths),
    }


def rerun_cli_check(rrd_path: Path, output: Path) -> dict[str, Any]:
    """`rerun rrd verify` and `rerun rrd stats` on the recording, outputs kept as text."""
    binary = shutil.which("rerun") or str(Path(sys.executable).with_name("rerun"))
    out: dict[str, Any] = {}
    for command in ("verify", "stats"):
        completed = subprocess.run(
            [binary, "rrd", command, str(rrd_path)], capture_output=True, text=True, check=False
        )
        (output / f"rrd_{command}.txt").write_text(completed.stdout + completed.stderr)
        out[command] = {"returncode": completed.returncode, "head": completed.stdout[:2000]}
    return out


# --------------------------------------------------------------------------- cli


def build(
    *,
    clip_path: Path,
    arm_dirs: dict[str, Path],
    rig_path: Path,
    seeds_dir: Path | None,
    output: Path,
    mask_arm: str = "b",
    secondary_mask_arm: str | None = "c",
    mask_every: int = 6,
    secondary_mask_every: int = 30,
    tracks_dir: str | None = None,
    trials_path: Path = DEFAULT_TRIALS,
    scoreboard_md: Path | None = None,
    preset_dir: Path | None = DEFAULT_PRESET_DIR,
    skip_video: bool = False,
    frames_limit: int | None = None,
    fpv_poses: Path | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    clip = load_clip(clip_path)
    clip_doc = json.loads(Path(clip_path).read_text(encoding="utf-8"))
    if frames_limit is not None:
        clip = replace(
            clip,
            end_frame_exclusive=min(clip.end_frame_exclusive, clip.start_frame + frames_limit),
        )
    if skip_video:
        clip = replace(clip, proxies={})
    config = read_camera_config(clip.camera_config)
    cams = cameras_from_config(config)
    fpv_source = resolve_fpv_source(config, fpv_poses)
    rig = json.loads(Path(rig_path).read_text(encoding="utf-8"))
    frames = list(clip.frames)
    arms = {label: load_arm(label, path, tracks_dir) for label, path in arm_dirs.items()}
    if mask_arm not in arms:
        mask_arm = next(iter(arms))
    if secondary_mask_arm not in arms:
        secondary_mask_arm = None
    observations = {
        label: load_view_observations(arm.arm_dir / "observations.jsonl", frames=frames)
        for label, arm in arms.items()
        if label in (mask_arm, secondary_mask_arm) or arm.worker_runs
    }
    default_arm = arms[mask_arm]
    box_sizes = median_box_sizes(default_arm.arm_dir / "observations.jsonl", clip.containers)
    volumes = container_volumes(rig, clip.containers, cams, box_sizes)
    trials = (
        json.loads(Path(trials_path).read_text(encoding="utf-8"))
        if Path(trials_path).is_file()
        else {}
    )
    intervals = lid_closed_intervals(trials, clip.trial)
    cycles = cycles_in_window(intervals, clip.start_frame, clip.end_frame_exclusive)
    seeds = None
    seeds_md = None
    if seeds_dir is not None and (Path(seeds_dir) / "seeds.json").is_file():
        seeds = json.loads((Path(seeds_dir) / "seeds.json").read_text(encoding="utf-8"))
        md = Path(seeds_dir) / "seeds.md"
        seeds_md = md.read_text(encoding="utf-8") if md.is_file() else None
    scoreboard_text = None
    if scoreboard_md is None:
        candidate = default_arm.arm_dir.parent / "scoreboard" / "scoreboard.md"
        scoreboard_md = candidate if candidate.is_file() else None
    if scoreboard_md is not None and Path(scoreboard_md).is_file():
        scoreboard_text = Path(scoreboard_md).read_text(encoding="utf-8")
    rig_md = Path(rig_path).with_name("rig.md")
    markers = None
    marker_file = (
        POSES / f"third_person_camera_poses/{config.recording_day}/params/marker_points.npy"
    )
    if marker_file.exists():
        markers = marker_points(config.recording_day)
    shipped_t5 = None
    t5 = config.fixed.get("T5")
    if t5 is not None and t5.provenance == "marker_pnp":
        from .finebio_cameras import fixed_camera

        try:
            shipped_t5 = fixed_camera(config.recording_day, t5.camera_id, "T5_shipped")
        except (FileNotFoundError, OSError):
            shipped_t5 = None
    # Storyboard.
    story: list[StoryItem] = []
    story += pick_centrifuge_story(default_arm, cycles)
    story += pick_fpv_story(default_arm, fpv_source, frames)
    story += pick_plate_story(
        observations[mask_arm],
        default_arm,
        fpv_source,
        frames,
        views=[*cams, clip.fpv_view],
        annotated=clip_doc.get("annotated_frames_in_window", {}).get("six_view", []),
    )
    # The confidence drop: every arm with masks is searched; the candidate with the largest
    # confidence drop + mask-vs-box IoU drop wins (the arms' known failures, when they qualify,
    # are preferred within their own arm).
    drop_items: list[StoryItem] = []
    for drop_arm, drop_data in arms.items():
        if not drop_data.worker_runs:
            continue
        drop_signals = load_observation_signals(
            drop_data.arm_dir / "observations.jsonl",
            {(v, r.frame, s) for r in drop_data.rows for v, s in r.support_slots.items()},
        )
        drop_items += pick_confidence_drop(
            drop_data, drop_signals, preferred_slots=PREFERRED_FAILURE_SLOTS.get(drop_arm, ())
        )
    if drop_items:
        story.append(max(drop_items, key=lambda it: it.numbers.get("score", 0.0)))
    story = number_items(story)
    loo_rows = detector_rows_for_slice(default_arm.arm_dir / "observations.jsonl", LOO_CLASSES)
    output.mkdir(parents=True, exist_ok=True)
    inputs = BuildInputs(
        clip=clip,
        clip_doc=clip_doc,
        cams=cams,
        fpv_source=fpv_source,
        rig=rig,
        arms=arms,
        mask_arm=mask_arm,
        secondary_mask_arm=secondary_mask_arm,
        mask_every=max(1, mask_every),
        secondary_mask_every=max(1, secondary_mask_every),
        observations=observations,
        volumes=volumes,
        lid_intervals=intervals,
        cycles=cycles,
        seeds=seeds,
        seeds_md=seeds_md,
        scoreboard_md=scoreboard_text,
        rig_md=rig_md.read_text(encoding="utf-8") if rig_md.is_file() else None,
        markers=markers,
        shipped_t5=shipped_t5,
        storyboard=story,
        loo_rows=loo_rows,
        output=output,
    )
    (output / "storyboard.md").write_text(
        storyboard_markdown(story, offset=clip.start_frame, trial=clip.trial), encoding="utf-8"
    )
    (output / "storyboard.json").write_text(
        json.dumps([it.to_record(clip.start_frame) for it in story], indent=1) + "\n",
        encoding="utf-8",
    )
    index = log_recording(inputs)
    presets = write_presets(inputs, index)
    if preset_dir is not None:
        Path(preset_dir).mkdir(parents=True, exist_ok=True)
        for name in PRESETS:
            shutil.copyfile(output / f"{name}.rbl", Path(preset_dir) / f"finebio_{name}.rbl")
        presets["copied_to"] = Path(preset_dir).as_posix()
    rrd_path = output / RECORDING_NAME
    summary = entity_summary(rrd_path)
    checks = rerun_cli_check(rrd_path, output)
    review_index = {
        "schema": SCHEMA,
        "trial": clip.trial,
        "clip_config": Path(clip_path).as_posix(),
        "window": [clip.start_frame, clip.end_frame_exclusive],
        "arms": {
            label: {"arm_dir": a.arm_dir.as_posix(), "tracks_dir": a.tracks_dir.as_posix()}
            for label, a in arms.items()
        },
        "mask_arm": mask_arm,
        "secondary_mask_arm": secondary_mask_arm,
        "recording": index,
        "entities": {k: v for k, v in summary.items() if k != "paths"},
        "presets": presets,
        "rerun_cli": checks,
        "storyboard": [it.to_record(clip.start_frame) for it in story],
        "containers": [v.to_record() for v in volumes],
        "cycles_in_window": [list(c) for c in cycles],
        "claim_boundary": CLAIM_BOUNDARY,
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }
    (output / "review_index.json").write_text(
        json.dumps(review_index, indent=1) + "\n", encoding="utf-8"
    )
    (output / "entity_paths.txt").write_text("\n".join(summary["paths"]) + "\n", encoding="utf-8")
    return review_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--clip-config", type=Path, required=True)
    parser.add_argument(
        "--arm-dirs", required=True, help="label=dir,label=dir,... (the arms to log)"
    )
    parser.add_argument("--rig", type=Path, required=True)
    parser.add_argument("--seeds", type=Path, default=None, help="seeds run dir (seeds.json)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask-arm", default="b", help="arm whose masks the tiles show by default")
    parser.add_argument(
        "--secondary-mask-arm", default="c", help="second arm whose masks are logged (sparser)"
    )
    parser.add_argument("--mask-every", type=int, default=6)
    parser.add_argument("--secondary-mask-every", type=int, default=30)
    parser.add_argument(
        "--tracks-dir",
        default=None,
        help="tracks directory name under each arm (default: tracks-ext when present, else "
        "tracks); the confidence and events outputs are read from confidence<tag> / "
        "events<tag> for tracks<tag>",
    )
    parser.add_argument("--trials", type=Path, default=DEFAULT_TRIALS)
    parser.add_argument("--scoreboard", type=Path, default=None, help="scoreboard.md to embed")
    parser.add_argument(
        "--preset-dir",
        default=str(DEFAULT_PRESET_DIR),
        help="where the committed preset copies go (finebio_<name>.rbl); '' to skip",
    )
    parser.add_argument("--skip-video", action="store_true", help="no AssetVideo (tests)")
    parser.add_argument("--frames", type=int, default=None, help="only the first N frames (tests)")
    parser.add_argument(
        "--fpv-poses",
        type=Path,
        default=None,
        help="fixture-format fpv_poses.json; default: the shipped pose file named in the config",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    index = build(
        clip_path=args.clip_config,
        arm_dirs=parse_arm_dirs(args.arm_dirs),
        rig_path=args.rig,
        seeds_dir=args.seeds,
        output=args.output,
        mask_arm=args.mask_arm,
        secondary_mask_arm=args.secondary_mask_arm or None,
        mask_every=args.mask_every,
        secondary_mask_every=args.secondary_mask_every,
        tracks_dir=args.tracks_dir,
        trials_path=args.trials,
        scoreboard_md=args.scoreboard,
        preset_dir=Path(args.preset_dir) if args.preset_dir else None,
        skip_video=args.skip_video,
        frames_limit=args.frames,
        fpv_poses=args.fpv_poses,
    )
    rec = index["recording"]
    print(
        f"{rec['recording']}: {rec['size_bytes'] / 1e6:.1f} MB, "
        f"{index['entities']['entity_paths']} entities, {rec['masks_logged']} masks, presets "
        f"ok={index['presets']['ok']}, verify rc={index['rerun_cli']['verify']['returncode']}, "
        f"{index['elapsed_seconds']} s"
    )
    return 0 if index["presets"]["ok"] and index["rerun_cli"]["verify"]["returncode"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
