"""Cached-mask geometry and native-rate review for the fixed-slot pipette study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import resource
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .finebio_colour import load_colour_settings, load_rig, sample_from_observation
from .finebio_observations import mask_axis_measurements
from .finebio_orientation import plunger_end_from_sample
from .pipette_line_review import cameras_at_raw_frame, clip_image_segment, json_value, panel
from .pipette_multiplex_run import LABELS, ROOT, write_json
from .pipette_video_metrics import frame_geometry, motion

VIEWS = ("T1", "T2", "T3", "T4", "T5", "fpv")
COLOURS = ((35, 155, 255), (255, 220, 30), (255, 55, 80), (205, 80, 255))


def geometry_inputs(root: Path, policy: str) -> dict:
    """Bind reusable geometry to its seeds, streams, calibration and cue implementation."""
    from .muggled_worker import parse_args, stream_identity

    paths = [root / "config.json", Path("configs/finebio/cameras/P03_03_01.json")]
    paths.append(Path("configs/finebio/pipettes.json"))
    paths.extend(
        Path(__file__).with_name(name + ".py")
        for name in (
            "finebio_colour", "finebio_observations", "finebio_orientation",
            "multiview_lines", "pipette_line_review", "pipette_video_metrics",
        )
    )
    commands = {
        v: json.loads((root / policy / v / "worker_command.json").read_text()) for v in VIEWS
    }
    return {
        "files": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "observation_implementation": hashlib.sha256(
            inspect.getsource(observation).encode()
        ).hexdigest(),
        "streams": {
            v: stream_identity(parse_args(command[2:]), LABELS) for v, command in commands.items()
        },
    }


def observation(directory: Path, image: np.ndarray, obj: dict | None, settings: dict) -> dict:
    mask = np.zeros(image.shape[:2], dtype=bool)
    path = None
    if obj is not None:
        path = directory / obj["mask"]["uri"]
        pixels = cv2.imread(str(path), 0)
        if pixels is None or pixels.shape != mask.shape:
            raise ValueError(f"Missing or incompatible mask: {path}")
        mask = pixels > 0
    axis = mask_axis_measurements(mask)
    ys, xs = np.nonzero(mask)
    sample = sample_from_observation(
        image, {"mask_axis_px": axis.axis_px, "mask_width_px": axis.width_px}, settings
    )
    return json_value(
        {
            "active": obj is not None,
            "mask": None if path is None else str(path.resolve()),
            "object_id": None if obj is None else obj["object_id"],
            "axis": asdict(axis),
            "centroid_px": None
            if not len(xs)
            else [float(xs.mean() + 0.5), float(ys.mean() + 0.5)],
            "pixels": int(mask.sum()),
            "colour": plunger_end_from_sample(sample, axis.width_px or 0),
        }
    )


def analyze(root: Path, policy: str, frames: int) -> None:
    config = json.loads((root / "config.json").read_text())
    folder = root / policy
    identity = geometry_inputs(root, policy)
    manifest = folder / "geometry-inputs.json"
    destination = folder / "geometry.jsonl"
    if manifest.exists():
        if json.loads(manifest.read_text()) != identity:
            raise ValueError("Geometry cache inputs changed; use a separate output directory")
    elif destination.exists():
        raise ValueError("Existing geometry has no input manifest; review before adopting it")
    else:
        write_json(manifest, identity)
    streams = {v: (folder / v / "observations.jsonl").open() for v in VIEWS}
    captures = {v: cv2.VideoCapture(config["views"][v]["video"]) for v in VIEWS}
    rig = load_rig(Path("configs/finebio/cameras/P03_03_01.json"))
    settings = load_colour_settings(Path("configs/finebio/pipettes.json"))
    sizes = {v: config["views"][v]["size_wh"] for v in VIEWS}
    cached = destination.read_text().splitlines() if destination.exists() else []
    if any(json.loads(line)["raw_frame"] != frame for frame, line in enumerate(cached)):
        raise ValueError("Duplicate or missing frame in cached geometry")
    if len(cached) > frames:
        raise ValueError("Cannot replace a longer geometry cache with a shorter stage")
    with destination.open("a") as output:
        for frame in range(frames):
            raw = {v: json.loads(streams[v].readline()) for v in VIEWS}
            for v in VIEWS:
                if raw[v]["analysis_frame_index"] != frame:
                    raise ValueError(f"Duplicate or missing native observation: {v}/{frame}")
            if frame < len(cached):
                for capture in captures.values():
                    if not capture.grab():
                        raise ValueError("Original video ended before cached geometry")
                continue
            rows = {}
            for v in VIEWS:
                ok, image = captures[v].read()
                if not ok:
                    raise ValueError(f"Video ended at {v}/{frame}")
                objects = {obj["label"]: obj for obj in raw[v]["objects"]}
                for slot, label in enumerate(LABELS):
                    if label in objects and objects[label]["object_id"] != f"sam3-{slot:02d}":
                        raise ValueError("Fixed slot identity changed")
                rows[v] = {
                    label: observation(folder / v, image, objects.get(label), settings)
                    for label in LABELS
                }
            blue = frame_geometry(
                {v: rows[v][LABELS[0]] for v in VIEWS},
                cameras_at_raw_frame(rig, frame, sizes),
            )
            output.write(
                json.dumps(
                    json_value({"raw_frame": frame, "views": rows, "blue": blue}), allow_nan=False
                )
                + "\n"
            )
            if frame % 300 == 0:
                output.flush()
                print(f"{policy}: geometry {frame}/{frames}", flush=True)
    for stream in streams.values():
        stream.close()
    for capture in captures.values():
        capture.release()
    metrics(folder, config["native_fps"])


def metrics(folder: Path, fps: float) -> None:
    buckets = {}
    history = {}
    lag = round(fps)
    for text in (folder / "geometry.jsonl").open():
        data = json.loads(text)
        frame, blue = data["raw_frame"], data["blue"]
        bucket = buckets.setdefault(math.floor(frame / fps), [])
        row = {
            "fit": blue["line"] is not None,
            "direction": blue["direction"]["tip_end"] is not None,
            "axis_views": sum(v["axis"]["axis_px"] is not None for v in blue["views"].values()),
            "active_views": sum(v["active"] for v in blue["views"].values()),
            "loo": [
                r["perpendicular_px"]
                for r in blue["loo"]
                if r["fitted"] and r["perpendicular_px"] is not None
            ],
        }
        if frame - lag in history:
            row.update(motion(blue, history[frame - lag], lag / fps))
        bucket.append(row)
        history[frame] = blue
        history.pop(frame - lag - 1, None)
    rows = []
    for second, observations in buckets.items():
        row = {
            "second": second,
            "frames": len(observations),
            "fit_fraction": np.mean([r["fit"] for r in observations]),
            "resolved_direction_fraction": np.mean([r["direction"] for r in observations]),
            "two_camera_axis_fraction": np.mean([r["axis_views"] >= 2 for r in observations]),
            "mean_active_views": np.mean([r["active_views"] for r in observations]),
        }
        residuals = [px for r in observations for px in r["loo"]]
        row["median_loo_px"] = float(np.median(residuals)) if residuals else None
        for key in ("midpoint_cm_s", "axis_deg_s", "directed_deg_s"):
            values = [r[key] for r in observations if r.get(key) is not None]
            row[key] = float(np.median(values)) if values else None
            row[key + "_samples"] = len(values)
        rows.append(json_value(row))
    write_json(folder / "per-second.json", rows)
    with (folder / "per-second.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def temporal(root: Path, policy: str) -> None:
    """Compare the existing accumulator/retrofit with identical cached colour/taper cues."""
    from .finebio_orientation import EpisodeAccumulator, OrientRecord, peak_signs

    selection = json.loads((root / "policy-review.json").read_text())
    if selection["selected_mask_policy"] != policy:
        raise ValueError("Select the mask policy before changing orientation")
    accumulator = EpisodeAccumulator()
    previous_vector = None
    previous_frame = None
    records, rows = [], []
    for text in (root / policy / "geometry.jsonl").open():
        data = json.loads(text)
        frame, blue = data["raw_frame"], data["blue"]
        row = {
            "raw_frame": frame,
            "baseline_tip_end": blue["direction"]["tip_end"],
            "online_tip_end": None,
            "retrofit_tip_end": None,
            "episode": None,
            "endpoint0_is_canonical_end0": None,
        }
        fitted = blue["line"]
        if fitted is not None and fitted["endpoints"] is not None:
            ends = np.array(fitted["endpoints"])
            vector = ends[1] - ends[0]
            if np.linalg.norm(vector) > 1e-6:
                if previous_frame is not None and frame > previous_frame + 1:
                    accumulator.resume()
                aligned = previous_vector is None or float(vector @ previous_vector) >= 0
                previous_vector = vector if aligned else -vector
                contribution = blue["direction"]["log_odds"] * (1 if aligned else -1)
                accumulator.add(
                    contribution,
                    elapsed_frames=1 if previous_frame is None else frame - previous_frame,
                )
                previous_frame = frame
                if accumulator.resolved and accumulator.sign:
                    canonical_tip = 0 if accumulator.sign > 0 else 1
                    row["online_tip_end"] = canonical_tip if aligned else 1 - canonical_tip
                row.update(
                    {
                        "episode": accumulator.episode,
                        "endpoint0_is_canonical_end0": aligned,
                        "log_odds": accumulator.log_odds,
                        "confidence": accumulator.confidence,
                        "contribution": contribution,
                    }
                )
                records.append(
                    OrientRecord("original-blue", frame, accumulator.episode, contribution, aligned)
                )
        rows.append(row)
    signs = peak_signs(records)
    counts = {name: 0 for name in ("baseline", "online", "retrofit")}
    flips = dict.fromkeys(counts, 0)
    previous = {}
    for row in rows:
        episode, aligned = row["episode"], row["endpoint0_is_canonical_end0"]
        if episode is not None:
            sign = signs["original-blue", episode]
            if sign:
                end = 0 if sign > 0 else 1
                row["retrofit_tip_end"] = end if aligned else 1 - end
            else:
                row["retrofit_tip_end"] = row["online_tip_end"]
        for name in counts:
            tip = row[f"{name}_tip_end"]
            if tip is None:
                previous.pop(name, None)
                continue
            counts[name] += 1
            canonical_tip = tip if aligned else 1 - tip
            old = previous.get(name)
            if old and old[0] == episode and old[1] == row["raw_frame"] - 1:
                flips[name] += old[2] != canonical_tip
            previous[name] = (episode, row["raw_frame"], canonical_tip)
    folder = root / policy
    with (folder / "orientation-temporal.jsonl").open("w") as output:
        for row in rows:
            output.write(json.dumps(row, allow_nan=False) + "\n")
    write_json(
        folder / "orientation-temporal-summary.json",
        {
            "frames": len(rows),
            "resolved_frames": counts,
            "resolved_fraction": {k: v / len(rows) for k, v in counts.items()},
            "adjacent_within_episode_sign_changes": flips,
            "cues": "Identical cached colour/taper votes; no new hand, gravity or tip-box cues",
            "endpoint_order": "Aligned across native frames before accumulation; SVD sign ignored",
            "ground_truth": False,
            "selection": "Pending visual review; fewer sign changes do not prove correct ends",
        },
    )


def export(root: Path, policy: str, start: int, frames: int) -> None:
    import rerun as rr
    import rerun.blueprint as b

    config = json.loads((root / "config.json").read_text())
    folder = root / policy
    captures = {v: cv2.VideoCapture(config["views"][v]["video"]) for v in VIEWS}
    for capture in captures.values():
        for _ in range(start):
            if not capture.grab():
                raise ValueError("Clip begins after video end")
    clips = folder / "clips"
    clips.mkdir(exist_ok=True)
    name = f"raw-{start:06d}-{start + frames - 1:06d}"
    rr.init("pipette-multiplex", recording_id=f"{policy}-{name}-explicit-empty-masks")
    rr.save(clips / f"{name}.rrd")
    rr.log(
        "/",
        rr.AnnotationContext(
            [
                rr.AnnotationInfo(id=0, color=(0, 0, 0, 0)),
                *[
                    rr.AnnotationInfo(id=i + 1, label=label, color=(*colour, 95))
                    for i, (label, colour) in enumerate(zip(LABELS, COLOURS, strict=True))
                ],
            ]
        ),
        static=True,
    )
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_DOWN, static=True)
    center = np.zeros(3)
    exported = 0
    empty_masks = {}
    review_frames = {start, start + frames - 1, 30, 100, 200, 300, 500, 600, 899}
    review_frames.update(
        math.ceil(second * config["native_fps"]) for second in (60, 120, 180, 240, 244, 271, 283)
    )
    with (folder / "geometry.jsonl").open() as source:
        for text in source:
            data = json.loads(text)
            frame = data["raw_frame"]
            if frame < start:
                continue
            if frame >= start + frames:
                break
            rr.set_time("raw_frame", sequence=frame)
            rr.set_time("seconds", duration=frame / config["native_fps"])
            sheets = []
            for view in VIEWS:
                ok, image = captures[view].read()
                if not ok:
                    raise ValueError("Native clip video ended early")
                origin = f"cameras/{view}"
                rr.log(
                    origin + "/original",
                    rr.Image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).compress(jpeg_quality=90),
                )
                overlay = image.copy()
                for slot, (label, colour) in enumerate(zip(LABELS, COLOURS, strict=True)):
                    row = data["views"][view][label]
                    path = origin + f"/{label}"
                    for child in ("mask", "axis", "label"):
                        rr.log(path + "/" + child, rr.Clear(recursive=False))
                    if row["mask"]:
                        mask = cv2.imread(row["mask"], 0) > 0
                        # Independent transparent layers preserve overlapping slots.
                        # PNG avoids 24 dense native rasters per viewer frame.
                        rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
                        rgba[mask] = (*colour[::-1], 95)
                        ok, encoded = cv2.imencode(".png", rgba, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                        if not ok:
                            raise RuntimeError("Could not encode review mask")
                        rr.log(
                            path + "/mask",
                            rr.EncodedImage(
                                contents=encoded.tobytes(),
                                media_type="image/png",
                                draw_order=slot + 1,
                            ),
                        )
                        overlay[mask] = (
                            overlay[mask] * 0.65 + np.array(colour[::-1]) * 0.35
                        ).astype("uint8")
                    else:
                        # Encoded-image textures can survive a Clear in the viewer.
                        # An explicit transparent image preserves an absent observation.
                        if view not in empty_masks:
                            empty = np.zeros((*image.shape[:2], 4), dtype=np.uint8)
                            ok, encoded = cv2.imencode(
                                ".png", empty, [cv2.IMWRITE_PNG_COMPRESSION, 1]
                            )
                            if not ok:
                                raise RuntimeError("Could not encode an absent review mask")
                            empty_masks[view] = encoded.tobytes()
                        rr.log(
                            path + "/mask",
                            rr.EncodedImage(
                                contents=empty_masks[view],
                                media_type="image/png",
                                draw_order=slot + 1,
                            ),
                        )
                    axis = row["axis"]["axis_px"]
                    if axis:
                        rr.log(path + "/axis", rr.LineStrips2D([axis], colors=colour, radii=1.5))
                        points = np.array(axis).round().astype(int)
                        cv2.line(overlay, tuple(points[0]), tuple(points[1]), colour[::-1], 2)
                    label_position = np.mean(axis, axis=0) if axis else row["centroid_px"]
                    if label_position is not None:
                        rr.log(
                            path + "/label",
                            rr.Points2D([label_position], labels=[label], colors=colour),
                        )
                projected = data["blue"]["views"][view].get("projected_world_endpoints_px")
                rr.log(origin + "/blue-reprojection", rr.Clear(recursive=True))
                if projected is not None:
                    clipped = clip_image_segment(projected, config["views"][view]["size_wh"])
                    if clipped is not None:
                        rr.log(
                            origin + "/blue-reprojection",
                            rr.LineStrips2D([clipped], colors=(40, 255, 70), radii=2),
                        )
                        p = clipped.round().astype(int)
                        cv2.line(overlay, tuple(p[0]), tuple(p[1]), (70, 255, 40), 2)
                if frame in review_frames:
                    sheets.append(
                        np.hstack(
                            [
                                panel(image, f"{view} original raw {frame}", (640, 420)),
                                panel(overlay, "All slots; green = blue 3D projection", (640, 420)),
                            ]
                        )
                    )
                    cv2.imwrite(str(clips / f"{frame:06d}-{view}-original.jpg"), image)
                    cv2.imwrite(str(clips / f"{frame:06d}-{view}-overlay.jpg"), overlay)
            for child in ("shaft", "dispensing"):
                rr.log("world/blue/" + child, rr.Clear(recursive=False))
            blue = data["blue"]
            if blue["line"] and blue["line"]["endpoints"]:
                ends = np.array(blue["line"]["endpoints"])
                center = ends.mean(axis=0)
                rr.log("world/blue/shaft", rr.LineStrips3D([ends], colors=COLOURS[0], radii=0.12))
                tip = blue["direction"]["tip_end"]
                if tip is not None:
                    rr.log(
                        "world/blue/dispensing",
                        rr.Arrows3D(
                            origins=[ends[1 - tip]],
                            vectors=[ends[tip] - ends[1 - tip]],
                            colors=(255, 80, 220),
                            radii=0.15,
                        ),
                    )
            if sheets:
                cv2.imwrite(str(clips / f"{frame:06d}-review.jpg"), np.vstack(sheets))
            exported += 1
    if exported != frames:
        raise ValueError(f"Only {exported}/{frames} geometry frames available")
    eye = b.EyeControls3D(position=center + [60, -60, -60], look_target=center, eye_up=[0, 0, -1])
    blueprint = b.Blueprint(
        b.Horizontal(
            b.Spatial3DView(origin="world", name="Original blue only", eye_controls=eye),
            b.Grid(
                *[b.Spatial2DView(origin=f"cameras/{v}", name=v) for v in VIEWS], grid_columns=2
            ),
            column_shares=[3, 7],
        ),
        b.TimePanel(timeline="raw_frame", state="expanded", play_state="paused"),
        b.SelectionPanel(state="hidden"),
        b.BlueprintPanel(state="collapsed"),
        auto_layout=False,
        auto_views=False,
    )
    rr.send_blueprint(blueprint)
    blueprint.save("pipette-multiplex", clips / f"{name}.rbl")
    rr.disconnect()
    for capture in captures.values():
        capture.release()
    write_json(
        clips / f"{name}.json",
        {
            "native_frames": exported,
            "fps": config["native_fps"],
            "seconds": frames / config["native_fps"],
            "original_logs_per_camera_frame": 1,
            "composite_image_logs": 0,
            "viewer_memory_limit": "8GB",
            "policy": policy,
        },
    )


def main() -> None:
    cv2.setNumThreads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("analyze", "export", "temporal"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--policy", default="baseline")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()
    started = time.perf_counter()
    succeeded = False
    try:
        if args.action == "analyze":
            analyze(args.root, args.policy, args.frames)
        elif args.action == "temporal":
            temporal(args.root, args.policy)
        else:
            export(args.root, args.policy, args.start, args.frames)
        succeeded = True
    finally:
        write_json(
            args.root / args.policy / "runtime"
            / f"{args.action}-{args.start:06d}-{args.frames:06d}.json",
            {
                "action": args.action,
                "succeeded": succeeded,
                "elapsed_seconds": time.perf_counter() - started,
                "cpu_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                "rss_source": "Linux getrusage process high-water mark",
                "requested_frames": None if args.action == "temporal" else args.frames,
                "start_raw_frame": args.start,
                "cached_masks_reused": True,
            },
        )


if __name__ == "__main__":
    main()
