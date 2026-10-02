"""Freeze a reviewed multiplex checkpoint into an inference-free Rerun demo."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tarfile
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as b

from .pipette_line_review import clip_image_segment
from .pipette_multiplex_review import COLOURS, VIEWS
from .pipette_multiplex_run import LABELS

APP = "battle-pipette-orientation-demo"
DISPLAY_LABELS = {
    "blue_pipette": "Blue pipette",
    "yellow_pipette": "Yellow pipette",
    "red_pipette": "Red pipette",
    "multichannel_pipette": "Multichannel pipette",
}


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def prefix_digest(path: Path, frames: int) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for index in range(frames):
            line = stream.readline()
            if not line.endswith(b"\n"):
                raise ValueError(f"Incomplete input: {path}, frame {index}")
            checksum.update(line)
    return checksum.hexdigest()


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def prefix(path: Path, frames: int, field: str) -> list[dict]:
    rows = []
    with path.open() as stream:
        for index in range(frames):
            line = stream.readline()
            if not line.endswith("\n"):
                raise ValueError(f"Incomplete input: {path}, frame {index}")
            row = json.loads(line)
            if row[field] != index:
                raise ValueError(f"Noncontiguous input: {path}, frame {index}")
            rows.append(row)
    return rows


def display_size(size: tuple[int, int], max_side: int) -> tuple[int, int]:
    factor = min(1.0, max_side / max(size))
    return tuple(max(1, round(value * factor)) for value in size)


def transform(points: object, size: tuple[int, int], shown: tuple[int, int]) -> np.ndarray:
    """Use the actual rounded display dimensions, including non-square source pixels."""
    return np.asarray(points, dtype=float) * (np.asarray(shown) / np.asarray(size))


def sampled_frames(frames: int, fps: float, hz: float = 2) -> set[int]:
    return {0, frames - 1} | {
        min(frames - 1, round(index * fps / hz)) for index in range(math.ceil(frames / fps * hz))
    }


def validate(source: Path, policy: str, frames: int) -> tuple[dict, list[dict], dict]:
    config = json.loads((source / "config.json").read_text())
    if not config.get("reviewed_seeds") or config["labels"] != list(LABELS):
        raise ValueError("A reviewed four-slot seed configuration is required")
    review_path = source / policy / f"review-{frames}.json"
    review = json.loads(review_path.read_text())
    if review["stage_frames"] != frames:
        raise ValueError("Review does not match the requested completed stage")
    for view in VIEWS:
        if frames > config["views"][view]["frame_count"]:
            raise ValueError(f"Requested stage exceeds {view} source")
        rows = prefix(source / policy / view / "observations.jsonl", frames, "analysis_frame_index")
        for row in rows:
            for obj in row["objects"]:
                if obj["label"] not in LABELS:
                    raise ValueError("Unexpected object label")
                if obj["object_id"] != f"sam3-{LABELS.index(obj['label']):02d}":
                    raise ValueError("Fixed slot identity changed")
    geometry = prefix(source / policy / "geometry.jsonl", frames, "raw_frame")
    for row in geometry:
        if set(row["views"]) != set(VIEWS):
            raise ValueError("Geometry is missing a camera")
        for view in VIEWS:
            if set(row["views"][view]) != set(LABELS):
                raise ValueError("Geometry is missing a pipette slot")
    return config, geometry, review


def log_world(blue: dict, tip: int | None) -> None:
    # Clear before each native frame so an abstention never inherits the previous arrow.
    rr.log("world/blue", rr.Clear(recursive=True))
    for leaf in ("shaft", "dispensing", "end"):
        rr.log(f"world/blue/{leaf}", rr.Clear(recursive=False))
    line = blue["line"]
    if line is None or line["endpoints"] is None:
        return
    ends = np.asarray(line["endpoints"])
    rr.log("world/blue/shaft", rr.LineStrips3D([ends], colors=COLOURS[0], radii=0.15))
    if tip is not None:
        rr.log(
            "world/blue/dispensing",
            rr.Arrows3D(
                origins=[ends[1 - tip]],
                vectors=[ends[tip] - ends[1 - tip]],
                colors=(255, 80, 220),
                radii=0.18,
            ),
        )
        rr.log("world/blue/end", rr.Points3D([ends[tip]], colors=(255, 255, 255), radii=0.3))


def log_camera(image: np.ndarray, rows: dict, view: str, tip: int | None) -> None:
    size = (image.shape[1], image.shape[0])
    shown = display_size(size, 960)
    image = cv2.resize(image, shown, interpolation=cv2.INTER_AREA)
    origin = f"cameras/{view}"
    rr.log(
        origin + "/original",
        rr.Image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).compress(jpeg_quality=80),
    )
    for slot, (label, colour) in enumerate(zip(LABELS, COLOURS, strict=True)):
        row = rows[label]
        path = origin + "/" + label
        rr.log(path, rr.Clear(recursive=True))
        for leaf in ("mask", "axis", "label"):
            rr.log(path + "/" + leaf, rr.Clear(recursive=False))
        rgba = np.zeros((shown[1], shown[0], 4), dtype=np.uint8)
        label_at = None
        if row["mask"]:
            pixels = cv2.imread(row["mask"], 0)
            if pixels is None or pixels.shape != (size[1], size[0]):
                raise ValueError(f"Missing or incompatible mask: {row['mask']}")
            mask = cv2.resize(pixels, shown, interpolation=cv2.INTER_NEAREST) > 0
            rgba[mask] = (*colour[::-1], 80)
            if mask.any():
                ys, xs = np.nonzero(mask)
                label_at = [float(xs.mean()), float(ys.mean())]
        # Explicit transparent pixels replace textures that can survive a viewer clear.
        ok, encoded = cv2.imencode(".png", rgba)
        if not ok:
            raise ValueError("PNG encoding failed")
        rr.log(
            path + "/mask",
            rr.EncodedImage(
                contents=encoded.tobytes(), media_type="image/png", draw_order=slot + 1
            ),
        )
        axis = row["axis"]["axis_px"]
        if axis is not None:
            points = transform(axis, size, shown)
            rr.log(path + "/axis", rr.LineStrips2D([points], colors=colour, radii=1.3))
            label_at = points.mean(axis=0)
        if label_at is not None:
            rr.log(
                path + "/label",
                rr.Points2D([label_at], labels=[DISPLAY_LABELS[label]], colors=colour),
            )
    rr.log(origin + "/projection", rr.Clear(recursive=True))
    for leaf in ("shaft", "dispensing"):
        rr.log(origin + "/projection/" + leaf, rr.Clear(recursive=False))
    projected = rows["blue_pipette"].get("projected_world_endpoints_px")
    if projected is not None:
        clipped = clip_image_segment(transform(projected, size, shown), shown)
        if clipped is not None:
            rr.log(
                origin + "/projection/shaft",
                rr.LineStrips2D([clipped], colors=(40, 255, 70), radii=2),
            )
        if tip is not None:
            ends = transform(projected, size, shown)
            # Do not move an offscreen physical endpoint onto the image border.
            end = ends[tip]
            if np.isfinite(ends).all() and np.all(end >= 0) and np.all(end < shown):
                rr.log(
                    origin + "/projection/dispensing",
                    rr.Points2D([end], colors=(255, 80, 220), radii=4),
                )


def layouts(
    center: np.ndarray,
    crops: dict[str, tuple],
    *,
    masks_visible: bool = True,
    preserve_time: bool = False,
) -> dict[str, b.Blueprint]:
    eye = b.EyeControls3D(position=center + [45, -55, -45], look_target=center, eye_up=[0, 0, -1])

    def world():
        return b.Spatial3DView(
            origin="world", name="Blue pipette · 3D shaft and dispensing end (cm)", eye_controls=eye
        )

    def camera(view, crop=False):
        bounds = None
        if crop:
            x0, y0, x1, y1 = crops[view]
            bounds = b.VisualBounds2D(x_range=[x0, x1], y_range=[y0, y1])
        return b.Spatial2DView(
            origin=f"cameras/{view}",
            name="First-person camera" if view == "fpv" else f"{view} · External camera",
            visual_bounds=bounds,
            contents=["$origin/**"]
            + ([] if masks_visible else [f"- /cameras/{view}/{label}/mask" for label in LABELS]),
        )

    orientation = b.Vertical(
        b.Horizontal(
            world(),
            b.Grid(*[camera(v, True) for v in ("T1", "T2", "T3", "fpv")], grid_columns=2),
            column_shares=[4, 6],
        ),
        b.TextDocumentView(origin="evidence/current", name="Orientation and direct review"),
        row_shares=[8, 2],
        name="Orientation",
    )
    cameras = b.Grid(*[camera(v) for v in VIEWS], grid_columns=3, name="Cameras")
    evidence = b.Vertical(
        b.Horizontal(
            b.TextDocumentView(origin="evidence/current", name="Frame evidence"),
            b.TextDocumentView(origin="evidence/guide", name="Five-minute route"),
        ),
        b.Horizontal(
            b.TimeSeriesView(origin="metrics/cues", name="Colour / taper log odds"),
            b.TimeSeriesView(origin="metrics/support", name="Axis cameras / resolved direction"),
            b.TimeSeriesView(
                origin="metrics/consistency", name="Dropped-camera residual · original px"
            ),
        ),
        name="Evidence",
    )
    tabs = b.Tabs(orientation, cameras, evidence, active_tab=0)
    layouts = {"demo": tabs, "orientation": orientation, "cameras": cameras, "evidence": evidence}
    return {
        name: b.Blueprint(
            layout,
            b.TimePanel(
                timeline="raw_frame" if preserve_time else "seconds",
                play_state=None if preserve_time else "paused",
                state="collapsed",
            ),
            b.SelectionPanel(state="hidden"),
            b.BlueprintPanel(state="collapsed"),
            auto_layout=False,
            auto_views=False,
        )
        for name, layout in layouts.items()
    }


def review_route(review: dict, frames: int) -> list[int]:
    candidates = [0, 100, 200, frames - 1]
    candidates += [int(frame) for frame in review.get("frames", {})]
    return sorted(set(frame for frame in candidates if 0 <= frame < frames))


def note_text(value: object) -> str:
    """Present structured camera review notes in readable language."""
    if isinstance(value, dict):
        return (
            "; ".join(
                f"{key}: {note_text(part)}"
                for key, part in value.items()
                if part is not None and part != [] and part != {}
            )
            or "None."
        )
    if isinstance(value, (list, tuple)):
        return ", ".join(note_text(part) for part in value) or "None."
    if isinstance(value, str) and value in LABELS:
        return value.replace("_pipette", "")
    return str(value)


def export(
    output: Path,
    config: dict,
    geometry: list[dict],
    review: dict,
    orientation: str,
    temporal: list[dict] | None,
    start: int,
    stop: int,
    native: bool,
    public_controls: bool = False,
) -> dict:
    fps = config["native_fps"]
    samples = (
        set(range(start, stop))
        if native
        else sampled_frames(stop, fps) | set(review_route(review, stop))
    )
    samples = {f for f in samples if start <= f < stop}
    captures = {v: cv2.VideoCapture(config["views"][v]["video"]) for v in VIEWS}
    for capture in captures.values():
        if not capture.isOpened():
            raise ValueError("Cannot open original video")
        if start:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    ends = [
        row["blue"]["line"]["endpoints"]
        for row in geometry[start:stop]
        if row["blue"]["line"] and row["blue"]["line"]["endpoints"]
    ]
    center = np.median(np.array(ends).reshape(-1, 3), axis=0) if ends else np.zeros(3)
    crops = {}
    for v in VIEWS:
        shown = display_size(tuple(config["views"][v]["size_wh"]), 960)
        axes = [r["views"][v]["blue_pipette"]["axis"]["axis_px"] for r in geometry[start:stop]]
        points = np.array([a for a in axes if a is not None]).reshape(-1, 2)
        if len(points):
            points = transform(points, tuple(config["views"][v]["size_wh"]), shown)
            low = np.maximum(0, np.quantile(points, 0.02, axis=0) - 55)
            high = np.minimum(shown, np.quantile(points, 0.98, axis=0) + 55)
            crops[v] = (*low, *high)
        else:
            crops[v] = (0, 0, *shown)
    presets = layouts(center, crops)
    recording_id = f"{output.parent.name}-{output.stem}"
    rr.init(APP, recording_id=recording_id, default_blueprint=presets["demo"])
    rr.save(output)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_DOWN, static=True)
    rr.log(
        "world/scale",
        rr.LineStrips3D([[[0, 0, 0], [10, 0, 0]]], colors=(160, 160, 160)),
        static=True,
    )
    rr.log("world/scale-label", rr.Points3D([[5, 0, 0]], labels=["10 cm"], radii=0), static=True)
    route = review_route(review, len(geometry))
    guide = (
        "# Pipette orientation\n\nBlue: 3D shaft and dispensing end. "
        "All four: camera masks and 2D axes.\n\n"
    )
    guide += "**Five-minute route**\n\n" + "\n".join(f"- Raw {f} · {f / fps:.2f} s" for f in route)
    guide += (
        "\n\nBlue line: fitted visible shaft. Magenta: resolved dispensing end. "
        "Green: reprojection. No arrow means unresolved.\n\n"
        "Fit availability and fixed slots do not establish correct identity. "
        f"Camera images are {'native-rate' if native else 'sampled at 2 Hz'}; "
        "world geometry is native-rate.\n\n"
        "Cue polarity follows the fitter's endpoint order; a sign change alone "
        "does not establish a physical flip.\n\n"
        "FineBio: Yagi et al., IJCV 2025. Research review."
    )
    rr.log("evidence/guide", rr.TextDocument(guide, media_type="text/markdown"), static=True)
    for f in range(start, stop):
        row = geometry[f]
        blue = row["blue"]
        tip = (
            blue["direction"]["tip_end"]
            if orientation == "baseline"
            else temporal[f][f"{orientation}_tip_end"]
        )
        rr.set_time("raw_frame", sequence=f)
        rr.set_time("seconds", duration=f / fps)
        log_world(blue, tip)
        contributions = blue["direction"]["contributions"]
        for cue in ("colour", "taper"):
            rr.log(f"metrics/cues/{cue}", rr.Scalars(contributions.get(cue, 0)))
        support = sum(r["axis"]["axis_px"] is not None for r in blue["views"].values())
        rr.log("metrics/support/axis-cameras", rr.Scalars(support))
        rr.log("metrics/support/resolved", rr.Scalars(int(tip is not None)))
        residuals = [
            r["perpendicular_px"]
            for r in blue["loo"]
            if r["fitted"] and r["perpendicular_px"] is not None
        ]
        if residuals:
            rr.log("metrics/consistency/median", rr.Scalars(float(np.median(residuals))))
        else:
            rr.log("metrics/consistency/median", rr.Clear(recursive=True))
        notes = review.get("frames", {}).get(str(f), {})
        status = "unresolved" if tip is None else "resolved estimate"
        text = (
            f"## Raw {f} · {f / fps:.2f} s\n\n**Direction: {status}** · "
            f"{orientation}; {support} axis cameras.\n\n"
        )
        text += (
            f"Instantaneous cues: {blue['direction'].get('reason') or 'colour/taper agree'}. "
            "Geometric support does not validate physical identity.\n\n"
        )
        if notes:
            text += "\n\n".join(
                f"**{k.replace('_', ' ')}:** {note_text(value)}" for k, value in notes.items()
            )
        else:
            text += "This frame has no direct visual review note. Consult the marked review frames."
        rr.log("evidence/current", rr.TextDocument(text, media_type="text/markdown"))
        for view, capture in captures.items():
            if f in samples:
                ok, image = capture.read()
                if not ok:
                    raise ValueError(f"Video ended early: {view}/{f}")
                log_camera(image, row["views"][view], view, tip)
            elif not capture.grab():
                raise ValueError(f"Video ended early: {view}/{f}")
    rr.send_blueprint(presets["demo"])
    for name, preset in presets.items():
        preset.save(APP, output.with_name(output.stem + f"-{name}.rbl"))
    if public_controls:
        for visible in (True, False):
            for name, preset in layouts(
                center, crops, masks_visible=visible, preserve_time=True
            ).items():
                if name != "demo":
                    suffix = "on" if visible else "off"
                    preset.save(APP, output.parent / f"view-{name}-masks-{suffix}.rbl")
    rr.disconnect()
    for capture in captures.values():
        capture.release()
    return {
        "recording": output.name,
        "recording_id": recording_id,
        "raw_start": start,
        "raw_stop_exclusive": stop,
        "geometry_frames": stop - start,
        "image_frames_per_camera": len(samples),
        "image_hz": fps if native else 2,
        "display_max_side": 960,
        "sha256": digest(output),
    }


def build(
    source: Path,
    policy: str,
    frames: int,
    output: Path,
    orientation: str = "baseline",
    overview_only: bool = False,
) -> None:
    if frames <= 0 or output.exists() or output.with_suffix(".tar.gz").exists():
        raise ValueError("Use a positive frame count and a new output directory")
    config, geometry, review = validate(source, policy, frames)
    temporal = None
    if orientation != "baseline":
        temporal = prefix(source / policy / "orientation-temporal.jsonl", frames, "raw_frame")
    output.mkdir(parents=True)
    shutil.copy2(Path(__file__), output / "exporter-source.py")
    write_json(output / "source-config.json", config)
    provenance = {}
    # Later stage reviews describe new intervals. Preserve earlier inspected frames too.
    history = sorted(
        (
            path
            for path in (source / policy).glob("review-*.json")
            if path.stem.removeprefix("review-").isdigit()
            and int(path.stem.removeprefix("review-")) <= frames
        ),
        key=lambda path: int(path.stem.removeprefix("review-")),
    )
    notes, origins = {}, {}
    for path in history:
        stage = json.loads(path.read_text())
        if stage["stage_frames"] != int(path.stem.removeprefix("review-")):
            raise ValueError(f"Review stage mismatch: {path}")
        target = output / "review-history" / path.name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(path, target)
        provenance[str(path)] = digest(path)
        masks = path.with_name(path.stem + "-mask-hashes.json")
        if masks.exists():
            for mask, expected in json.loads(masks.read_text()).items():
                if digest(Path(mask)) != expected:
                    raise ValueError(f"Reviewed mask changed: {mask}")
            shutil.copy2(masks, target.with_name(masks.name))
            provenance[str(masks)] = digest(masks)
        for frame, note in stage.get("frames", {}).items():
            if 0 <= int(frame) < frames:
                notes[frame], origins[frame] = note, path.name
    review = {**review, "frames": notes, "frame_review_sources": origins}
    write_json(output / "review.json", review)
    for path in [source / "config.json", source / policy / f"review-{frames}.json"]:
        provenance[str(path)] = digest(path)
    for view in VIEWS:
        path = source / policy / view / "observations.jsonl"
        provenance[str(path)] = {
            "prefix_frames": frames,
            "prefix_sha256": prefix_digest(path, frames),
        }
    with (output / "geometry.jsonl").open("w") as stream:
        for row in geometry:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    provenance[str(source / policy / "geometry.jsonl")] = {
        "prefix_frames": frames,
        "prefix_sha256": prefix_digest(source / policy / "geometry.jsonl", frames),
    }
    if temporal is not None:
        with (output / "orientation-temporal.jsonl").open("w") as stream:
            for row in temporal:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        path = source / policy / "orientation-temporal.jsonl"
        provenance[str(path)] = {
            "prefix_frames": frames,
            "prefix_sha256": prefix_digest(path, frames),
        }
    reviewed_masks = source / policy / f"review-{frames}-mask-hashes.json"
    if reviewed_masks.exists():
        for path, expected in json.loads(reviewed_masks.read_text()).items():
            if digest(Path(path)) != expected:
                raise ValueError(f"Reviewed mask changed: {path}")
        shutil.copy2(reviewed_masks, output / "reviewed-mask-hashes.json")
        provenance[str(reviewed_masks)] = digest(reviewed_masks)
    for name in ("protocol.json", "policy-review.json", "orientation-review.json"):
        if (source / name).exists():
            shutil.copy2(source / name, output / name)
            provenance[str(source / name)] = digest(source / name)
    for name in ("per-second.csv", "per-second.json", "orientation-temporal-summary.json"):
        if (source / policy / name).exists():
            shutil.copy2(source / policy / name, output / name)
    recordings = [
        export(
            output / "overview.rrd",
            config,
            geometry,
            review,
            orientation,
            temporal,
            0,
            frames,
            False,
        )
    ]
    if not overview_only:
        reviewed = review_route(review, frames)
        middle = (
            min(reviewed, key=lambda frame: abs(frame - (frames - 1) / 2))
            if reviewed
            else frames // 2
        )
        starts = sorted({0, max(0, min(frames - 300, middle - 150)), max(0, frames - 300)})
        for start in starts:
            stop = min(frames, start + 300)
            recordings.append(
                export(
                    output / f"native-{start:06d}-{stop - 1:06d}.rrd",
                    config,
                    geometry,
                    review,
                    orientation,
                    temporal,
                    start,
                    stop,
                    True,
                )
            )
    write_json(
        output / "manifest.json",
        {
            "schema": "battle-pipette-demo/1",
            "source": str(source.resolve()),
            "mask_policy": policy,
            "orientation_policy": orientation,
            "native_frames": frames,
            "seconds": frames / config["native_fps"],
            "fps": config["native_fps"],
            "rerun_version": rr.__version__,
            "exporter_source_sha256": digest(output / "exporter-source.py"),
            "review_route_raw_frames": review_route(review, frames),
            "source_hashes": provenance,
            "recordings": recordings,
            "ground_truth": False,
            "rebuild_dependencies": [
                "original videos in source-config.json",
                "mask paths in geometry.jsonl",
                "battle project environment; inference is not required",
            ],
            "open_dependencies": [
                "Rerun 0.37.1; source videos and masks are embedded in recordings"
            ],
        },
    )
    (output / "README.md").write_text(
        "# Saved pipette orientation demo\n\n"
        "Open `overview.rrd` with `overview-demo.rbl` in Rerun 0.37.1. "
        "The default tabs are Orientation, Cameras and Evidence. "
        "No source files or inference are needed to open recordings.\n\n"
        "The overview shows native-rate geometry and camera evidence "
        "sampled at two images per second. "
        "Native clips show every frame in their labeled intervals. "
        "All images are display-scaled; measurements retain original coordinates.\n\n"
        "Restore the archive into an empty directory. Verify `SHA256SUMS` before opening. "
        "Rebuilding requires the original videos and masks named "
        "in the saved configuration and geometry. "
        "No model weights are bundled. This is a local snapshot, not an off-machine backup.\n\n"
        "Source/configuration, policy, frame range, review route and hashes "
        "are recorded in `manifest.json`. "
        "Blue has 3D orientation; all four pipettes have camera masks and 2D axes. "
        "Unresolved and contaminated fits remain visible. FineBio: Yagi et al., IJCV 2025.\n"
    )
    archive = package_bundle(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "archive": str(archive),
                "frames": frames,
                "recordings": recordings,
            },
            indent=2,
        )
    )


def package_bundle(output: Path) -> Path:
    """Seal a local bundle, including review artifacts added before publication."""
    checksums = [
        (digest(p), p.relative_to(output))
        for p in sorted(output.rglob("*"))
        if p.is_file() and p != output / "SHA256SUMS"
    ]
    (output / "SHA256SUMS").write_text("".join(f"{sha}  {path}\n" for sha, path in checksums))
    archive = output.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz", compresslevel=1) as tar:
        tar.add(output, arcname=output.name)
    archive.with_suffix(archive.suffix + ".sha256").write_text(
        f"{digest(archive)}  {archive.name}\n"
    )
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--orientation", choices=("baseline", "online", "retrofit"), default="baseline"
    )
    parser.add_argument("--overview-only", action="store_true")
    args = parser.parse_args()
    build(args.source, args.policy, args.frames, args.output, args.orientation, args.overview_only)


if __name__ == "__main__":
    main()
