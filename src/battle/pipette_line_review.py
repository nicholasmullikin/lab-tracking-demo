"""Offline paired review of saved blue-pipette masks, camera axes and 3D lines.

Run with the project interpreter: python -m battle.pipette_line_review.
No segmentation inference, production policy changes or temporal propagation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .finebio_colour import load_colour_settings, load_rig, sample_from_observation
from .finebio_observations import mask_axis_measurements
from .finebio_orientation import (
    RESOLVE_LOG_ODDS,
    colour_cue,
    frame_log_odds,
    plunger_end_from_sample,
    taper_camera,
    taper_cue,
)
from .multiview_lines import AxisObs, fit_line, loo_residual

VIEWS = ("T1", "T2", "T3", "T4", "T5", "fpv")
VARIANTS = ("earlier-selected", "recipe-model-top", "recipe-visual-best")
CONDITION = "p2-body-shaft-n1"


def json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def choose_mask(
    base: Path, labels: dict, ratings: dict, frame: int, view: str, variant: str
) -> tuple[Path, dict]:
    previous = labels["frames"][str(frame)]["views"][view]
    case = f"f{frame:03d}-{view}"
    if variant != VARIANTS[0] and case in ratings["cases"]:
        reviewed = ratings["cases"][case]["conditions"][CONDITION]
        key = (
            "model_top_candidate_index"
            if variant == "recipe-model-top"
            else "selected_candidate_index"
        )
        index = reviewed[key]
        path = base / "point-prompt-study" / case / "results" / "masks"
        return path / f"{CONDITION}_candidate-{index:02d}.png", {
            "substituted": True,
            "condition": CONDITION,
            "candidate_index": index,
            "selection_policy": variant,
        }
    return base / "every100-review" / previous["mask_uri"], {
        "substituted": False,
        "selection": previous["selection"],
        "quality": previous["quality"],
        "review_note": previous["review_note"],
    }


def endpoint_correspondence(local: np.ndarray, projected: np.ndarray) -> tuple[int, int] | None:
    """Map local endpoints to world endpoint order; abstain on collapse or a tied pairing."""
    local = np.asarray(local, dtype=float)
    projected = np.asarray(projected, dtype=float)
    if not np.isfinite(projected).all() or np.linalg.norm(projected[1] - projected[0]) < 4:
        return None
    direct = float(np.linalg.norm(local - projected, axis=1).sum())
    reverse = float(np.linalg.norm(local - projected[::-1], axis=1).sum())
    if abs(direct - reverse) <= 1e-6 * max(1.0, direct, reverse):
        return None
    return (0, 1) if direct < reverse else (1, 0)


def conservative_direction(votes: list) -> dict:
    votes = [vote for vote in votes if vote is not None and vote.confidence > 0]
    odds, contributions = frame_log_odds(votes)
    reason = None
    if not votes:
        reason = "no_visible_colour_or_taper_cue"
    elif len({vote.tip_end for vote in votes}) > 1:
        reason = "colour_taper_disagreement"
    elif abs(odds) <= RESOLVE_LOG_ODDS:
        reason = "below_existing_resolution_threshold"
    return {
        "tip_end": None if reason else votes[0].tip_end,
        "reason": reason,
        "log_odds": odds,
        "contributions": contributions,
        "votes": [asdict(vote) for vote in votes],
        "policy": "colour/taper only; agreement required; no episode accumulation",
    }


def cameras_at_raw_frame(rig: Any, frame: int, sizes: dict) -> dict:
    cams = rig.at(frame)  # Intentionally raw, never frame minus the production clip offset.
    for view, cam in cams.items():
        if tuple(cam.size) != tuple(sizes[view]):
            raise ValueError(f"{view}: camera {cam.size} != original {sizes[view]}")
    return cams


def draw_overlay(
    image: np.ndarray, mask: np.ndarray, row: dict, projected: list | None, tip_end: int | None
) -> np.ndarray:
    out = image.copy()
    out[mask] = (out[mask] * 0.7 + np.array([255, 170, 20]) * 0.3).astype(np.uint8)
    ends = row["axis"]["axis_px"]
    if ends is not None:
        p = np.rint(ends).astype(int)
        cv2.line(out, tuple(p[0]), tuple(p[1]), (0, 220, 255), 3, cv2.LINE_AA)
        for i, point in enumerate(p):
            cv2.circle(out, tuple(point), 5, (0, 220, 255), -1)
            cv2.putText(out, f"L{i}", tuple(point + 7), 0, 0.5, (0, 220, 255), 1)
    if projected is not None:
        p = np.rint(projected).astype(int)
        cv2.line(out, tuple(p[0]), tuple(p[1]), (60, 255, 60), 3, cv2.LINE_AA)
        for i, point in enumerate(p):
            cv2.circle(out, tuple(point), 7, (255, 255, 255), 2)
            cv2.putText(out, f"W{i}", tuple(point + 9), 0, 0.6, (255, 255, 255), 2)
        if tip_end is not None:
            cv2.arrowedLine(
                out,
                tuple(p[1 - tip_end]),
                tuple(p[tip_end]),
                (255, 80, 230),
                3,
                cv2.LINE_AA,
                tipLength=0.12,
            )
    return out


def panel(image: np.ndarray, title: str, size: tuple[int, int] = (420, 380)) -> np.ndarray:
    w, h = size
    scale = min(w / image.shape[1], (h - 42) / image.shape[0])
    resized = cv2.resize(image, None, fx=scale, fy=scale)
    out = np.full((h, w, 3), 24, np.uint8)
    out[42 : 42 + resized.shape[0], : resized.shape[1]] = resized
    cv2.putText(out, title, (6, 22), 0, 0.48, (255, 255, 255), 1)
    return out


def run(base: Path, out: Path, camera_config: Path, pipettes: Path) -> dict:
    labels = json.loads((base / "every100-review/labels.json").read_text())
    ratings = json.loads((base / "point-prompt-study/ratings.json").read_text())
    manifest = json.loads((base / "point-prompt-study/manifest.json").read_text())
    rig = load_rig(camera_config)
    settings = load_colour_settings(pipettes)
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "battle-pipette-line-review/1",
        "ground_truth": False,
        "production_changed": False,
        "camera_config": str(camera_config.resolve()),
        "camera_config_sha256": hashlib.sha256(camera_config.read_bytes()).hexdigest(),
        "camera_calibration": json.loads(camera_config.read_text())["fixed"],
        "variants": {},
        "source_images": {},
        "source_masks": {},
    }
    cache = {}
    for variant in VARIANTS:
        frames = {}
        for frame in labels["source_frames"]:
            images = {
                v: cv2.imread(str(base / "every100-review" / str(frame) / f"{v}-original.png"))
                for v in VIEWS
            }
            if any(im is None for im in images.values()):
                raise ValueError(f"Missing original at raw frame {frame}")
            sizes = {v: (im.shape[1], im.shape[0]) for v, im in images.items()}
            cams = cameras_at_raw_frame(rig, frame, sizes)
            rows, pairs = {}, []
            for view, image in images.items():
                source = base / "every100-review" / str(frame) / f"{view}-original.png"
                sha = hashlib.sha256(source.read_bytes()).hexdigest()
                report["source_images"][f"{frame}/{view}"] = {
                    "path": str(source.resolve()),
                    "sha256": sha,
                    "size_wh": sizes[view],
                }
                case = f"f{frame:03d}-{view}"
                if case in manifest["cases"] and sha != manifest["cases"][case]["source_sha256"]:
                    raise ValueError(f"Study original hash mismatch: {case}")
                path, provenance = choose_mask(base, labels, ratings, frame, view, variant)
                path = path.resolve()
                if path not in cache:
                    pixels = cv2.imread(str(path), 0)
                    if pixels is None or pixels.shape != image.shape[:2]:
                        raise ValueError(f"Invalid native-size mask: {path}")
                    if not set(np.unique(pixels)) <= {0, 255}:
                        raise ValueError(f"Nonbinary mask: {path}")
                    mask = pixels > 0
                    axis = mask_axis_measurements(mask)
                    y, x = np.nonzero(mask)
                    centroid = (
                        None if not len(x) else [float(x.mean() + 0.5), float(y.mean() + 0.5)]
                    )
                    colour_sample = sample_from_observation(
                        image,
                        {"mask_axis_px": axis.axis_px, "mask_width_px": axis.width_px},
                        settings,
                    )
                    colour = plunger_end_from_sample(colour_sample, axis.width_px or 0)
                    cache[path] = (mask, axis, centroid, colour, colour_sample)
                mask, axis, centroid, colour, sample = cache[path]
                report["source_masks"][str(path.resolve())] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                row = {
                    "mask_path": str(path.resolve()),
                    "provenance": provenance,
                    "axis": asdict(axis),
                    "centroid_px": centroid,
                    "mask_pixels": int(mask.sum()),
                    "colour": colour,
                    "colour_sample": sample,
                    "camera_available": view in cams,
                }
                rows[view] = row
                if view in cams and centroid is not None:
                    pairs.append(
                        (
                            cams[view],
                            AxisObs(
                                view,
                                axis.axis_px,
                                centroid,
                                axis.elongation,
                                ends_px=axis.body_ends_px,
                            ),
                        )
                    )
            line = fit_line(pairs)
            colour_votes, taper_votes = [], []
            for view, row in rows.items():
                row["projected_world_endpoints_px"] = None
                row["axis_to_world_endpoint"] = None
                if line is None or line.endpoints is None or view not in cams:
                    continue
                cam = cams[view]
                depths = (line.endpoints @ cam.R.T + cam.tvec.reshape(3))[..., 2]
                if np.any(depths <= 0):
                    row["projection_reason"] = "world_endpoints_behind_camera"
                    continue
                projected = cam.project(line.endpoints)
                if not np.isfinite(projected).all():
                    continue
                row["projected_world_endpoints_px"] = projected
                row["world_endpoints_inside_image"] = [
                    bool(0 <= p[0] < sizes[view][0] and 0 <= p[1] < sizes[view][1])
                    for p in projected
                ]
                ends = row["axis"]["axis_px"]
                if ends is None:
                    continue
                mapping = endpoint_correspondence(np.array(ends), projected)
                row["axis_to_world_endpoint"] = mapping
                if mapping is None:
                    continue
                length = float(np.linalg.norm(np.array(ends)[1] - np.array(ends)[0]))
                colour = row["colour"]
                if colour and colour["tip_end"] is not None:
                    colour_votes.append((mapping[colour["tip_end"]], colour["confidence"], length))
                axis = row["axis"]
                taper = taper_camera(
                    end_widths=axis["end_widths_px"],
                    tip_side=axis["tip_side"],
                    wide_is_tip=False,
                    axis_residual_px=axis["residual_px"],
                    axis_to_track=mapping,
                )
                row["taper_world_vote"] = taper
                if taper:
                    taper_votes.append((taper[0], taper[1], length))
            direction = conservative_direction([colour_cue(colour_votes), taper_cue(taper_votes)])
            data = json_value(
                {
                    "raw_frame": frame,
                    "views": rows,
                    "line": None if line is None else asdict(line),
                    "fit_failure": None if line else "existing_fitter_no_valid_geometry",
                    "visible_length_cm": None if line is None else line.length_cm,
                    "direction": direction,
                    "loo": loo_residual(pairs),
                    "substitutions": sum(r["provenance"]["substituted"] for r in rows.values()),
                }
            )
            frames[str(frame)] = data
            directory = out / variant / str(frame)
            directory.mkdir(parents=True, exist_ok=True)
            for view, row in data["views"].items():
                mask = cache[Path(row["mask_path"])][0]
                overlay = draw_overlay(
                    images[view],
                    mask,
                    row,
                    row["projected_world_endpoints_px"],
                    direction["tip_end"],
                )
                cv2.imwrite(str(directory / f"{view}-overlay.jpg"), overlay)
            print(variant, frame, "fit", bool(line), "tip", direction["tip_end"], flush=True)
        report["variants"][variant] = frames
    report = json_value(report)
    (out / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (out / "summary.json").write_text(json.dumps(comparison_summary(report), indent=2) + "\n")
    export_sheets(base, out, report)
    export_rerun(base, out, report)
    return report


def comparison_summary(report: dict) -> list[dict]:
    def evaluable(frames: dict) -> dict:
        return {
            (int(frame), row["view"]): row
            for frame, data in frames.items()
            for row in data["loo"]
            if row["fitted"] and row["perpendicular_px"] is not None
        }

    baseline = evaluable(report["variants"][VARIANTS[0]])
    summary = []
    for variant, frames in report["variants"].items():
        cells = evaluable(frames)
        keys = sorted(cells.keys() & baseline.keys())
        differences = [cells[k]["perpendicular_px"] - baseline[k]["perpendicular_px"] for k in keys]
        summary.append(
            {
                "variant": variant,
                "fits": sum(data["line"] is not None for data in frames.values()),
                "resolved": sum(
                    data["direction"]["tip_end"] is not None for data in frames.values()
                ),
                "axis_cells": sum(
                    row["axis"]["axis_px"] is not None
                    for data in frames.values()
                    for row in data["views"].values()
                ),
                "loo_cells": len(cells),
                "paired_cells": len(keys),
                "paired_median_px": float(np.median([cells[k]["perpendicular_px"] for k in keys]))
                if keys
                else None,
                "paired_median_angle_deg": float(
                    np.median(
                        [cells[k]["angle_deg"] for k in keys if cells[k]["angle_deg"] is not None]
                    )
                )
                if any(cells[k]["angle_deg"] is not None for k in keys)
                else None,
                "improved_gt_0p1_px": sum(d < -0.1 for d in differences),
                "worsened_gt_0p1_px": sum(d > 0.1 for d in differences),
                "within_0p1_px": sum(abs(d) <= 0.1 for d in differences),
                "median_paired_change_px": float(np.median(differences)) if keys else None,
            }
        )
    return summary


def export_sheets(base: Path, out: Path, report: dict) -> None:
    for frame in report["variants"][VARIANTS[0]]:
        full_rows, crop_rows = [], []
        for view in VIEWS:
            original = cv2.imread(str(base / "every100-review" / frame / f"{view}-original.png"))
            images = [original] + [
                cv2.imread(str(out / var / frame / f"{view}-overlay.jpg")) for var in VARIANTS
            ]
            full_rows.append(
                np.hstack(
                    [
                        panel(im, f"{view} {name}")
                        for im, name in zip(images, ("original", *VARIANTS), strict=True)
                    ]
                )
            )
            masks = [
                cv2.imread(report["variants"][var][frame]["views"][view]["mask_path"], 0)
                for var in VARIANTS
            ]
            ys, xs = np.nonzero(np.logical_or.reduce([m > 0 for m in masks]))
            if len(xs):
                for variant in VARIANTS:
                    projected = report["variants"][variant][frame]["views"][view][
                        "projected_world_endpoints_px"
                    ]
                    if projected is not None:
                        projected = np.array(projected)
                        xs = np.append(xs, np.clip(projected[:, 0], 0, original.shape[1] - 1))
                        ys = np.append(ys, np.clip(projected[:, 1], 0, original.shape[0] - 1))
                x, y = max(0, xs.min() - 60), max(0, ys.min() - 60)
                x2, y2 = (
                    min(original.shape[1], xs.max() + 61),
                    min(original.shape[0], ys.max() + 61),
                )
                crops = [im[int(y) : int(y2), int(x) : int(x2)] for im in images]
            else:
                crops = images
            for variant, crop in zip(VARIANTS, crops[1:], strict=True):
                cv2.imwrite(str(out / variant / frame / f"{view}-crop.jpg"), crop)
            crop_rows.append(
                np.hstack(
                    [
                        panel(im, f"{view} {name}")
                        for im, name in zip(crops, ("original", *VARIANTS), strict=True)
                    ]
                )
            )
        cv2.imwrite(str(out / f"raw-{frame}-full.jpg"), np.vstack(full_rows))
        cv2.imwrite(str(out / f"raw-{frame}-crops.jpg"), np.vstack(crop_rows))
    payload = json.dumps(report).replace("</", "<\\/")
    page = """<!doctype html><meta charset="utf-8"><title>Pipette lines</title>
<style>
body{font:16px system-ui;background:#15212c;color:white;margin:24px}
img{max-width:100%}select{padding:8px}pre{white-space:pre-wrap}a{color:#8cd0ff}
</style>
<h1>Pipette camera and 3D line comparison</h1>
<p>Yellow: local mask axis. Green: projected 3D visible extent. White W0/W1: world endpoints.
Magenta arrow: dispensing direction, only when colour/taper agree. Cyan: mask.</p>
<p>Three mask sets: earlier selections; seven substitutions with recipe model-top;
optimistic visual-best substitutions.
No temporal tracking, length completion or measured accuracy.</p>
<select id="frame"></select><p id="notes"></p><img id="crops">
<details><summary>Full-frame comparisons</summary><img id="full"></details><pre id="stats"></pre>
<p><a href="README.md">Findings and methods</a> ·
<a href="results.json">Full provenance and metrics</a></p>
<script>
const D=PAYLOAD;
const S=document.getElementById('frame');
for(const f of Object.keys(D.variants['earlier-selected']))
  S.add(new Option('Raw '+f,f));
function update(){
  let f=S.value;
  document.getElementById('crops').src='raw-'+f+'-crops.jpg';
  document.getElementById('full').src='raw-'+f+'-full.jpg';
  document.getElementById('notes').textContent='Sparse sample at raw frame '+f+
    '; originals / earlier / recipe model-top / recipe visual-best.';
  document.getElementById('stats').textContent=JSON.stringify(
    Object.fromEntries(Object.entries(D.variants).map(([v,frames])=>[v,{
      line:frames[f].line,direction:frames[f].direction,loo:frames[f].loo
    }])),null,2);
}
S.onchange=update;update();
</script>"""
    (out / "review.html").write_text(page.replace("PAYLOAD", payload))


def export_rerun(base: Path, out: Path, report: dict) -> None:
    import rerun as rr
    import rerun.blueprint as b

    rr.init("finebio-pipette-line-comparison", recording_id="raw0-600-line-review-v1-20261001")
    rr.save(out / "pipette-lines.rrd")
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_DOWN, static=True)
    for frame in report["variants"][VARIANTS[0]]:
        rr.set_time("source_frame", sequence=int(frame))
        for variant, frames in report["variants"].items():
            data = frames[frame]
            line = data["line"]
            origin = f"world/{variant}"
            rr.log(origin, rr.Clear(recursive=True))
            if line and line["endpoints"] is not None:
                ends = np.array(line["endpoints"])
                colour = {
                    VARIANTS[0]: [255, 210, 30],
                    VARIANTS[1]: [70, 255, 70],
                    VARIANTS[2]: [255, 80, 230],
                }[variant]
                rr.log(origin + "/line", rr.LineStrips3D([ends], colors=colour, radii=0.12))
                rr.log(
                    origin + "/ends",
                    rr.Points3D(ends, colors=colour, radii=0.4, labels=["W0", "W1"]),
                )
                tip = data["direction"]["tip_end"]
                if tip is not None:
                    rr.log(
                        origin + "/direction",
                        rr.Arrows3D(
                            origins=[ends[1 - tip]],
                            vectors=[ends[tip] - ends[1 - tip]],
                            colors=colour,
                            radii=0.15,
                        ),
                    )
            for view in VIEWS:
                image = cv2.imread(str(out / variant / frame / f"{view}-overlay.jpg"))
                rr.log(
                    f"cameras/{variant}/{view}",
                    rr.Image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).compress(jpeg_quality=93),
                )
                crop = cv2.imread(str(out / variant / frame / f"{view}-crop.jpg"))
                crop = panel(crop, f"{view}: raw {frame}", size=(640, 480))
                rr.log(
                    f"crops/{variant}/{view}",
                    rr.Image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).compress(jpeg_quality=95),
                )
            length = data["visible_length_cm"]
            tip = data["direction"]["tip_end"]
            direction_text = f"toward W{tip}" if tip is not None else data["direction"]["reason"]
            text = [
                f"# Raw {frame}: {variant}",
                f"Visible extent: {length:.2f} cm" if length is not None else "No valid fit",
                f"Direction: {direction_text}",
                f"Substitutions: {data['substitutions']} / 6",
                "",
                "## Dropped-camera residuals",
                "",
                "| View | px | degrees |",
                "|---|---:|---:|",
            ]
            for row in data["loo"]:
                px, angle = row["perpendicular_px"], row["angle_deg"]
                text.append(
                    f"| {row['view']} | {px:.2f} | {angle:.2f} |"
                    if px is not None and angle is not None
                    else f"| {row['view']} | — | — |"
                )
            text += [
                "",
                "Colour/taper agree to draw arrows. No hand cue or temporal vote.",
                "Consistency residuals are not ground-truth accuracy.",
            ]
            rr.log(
                f"readout/{variant}",
                rr.TextDocument(
                    "\n\n".join(text[:5]) + "\n" + "\n".join(text[5:]),
                    media_type=rr.MediaType.MARKDOWN,
                ),
            )
        rr.log(
            "comparison",
            rr.Image(
                cv2.cvtColor(cv2.imread(str(out / f"raw-{frame}-crops.jpg")), cv2.COLOR_BGR2RGB)
            ).compress(jpeg_quality=95),
        )
        rr.log(
            "legend",
            rr.TextDocument(
                "Yellow local axis; green projected 3D visible extent; "
                "white world endpoint indices; magenta tip arrow only on cue agreement. "
                "World units: cm, z into bench. Three variants share all but seven masks. "
                "No length prior or temporal accumulation. Missing FPV poses excluded from 3D."
            ),
        )
    tabs = []
    endpoints = np.array(
        [
            x["line"]["endpoints"]
            for frames in report["variants"].values()
            for x in frames.values()
            if x["line"] and x["line"]["endpoints"] is not None
        ]
    ).reshape(-1, 3)
    target = (endpoints.min(axis=0) + endpoints.max(axis=0)) / 2 if len(endpoints) else np.zeros(3)
    span = max(35.0, float(np.ptp(endpoints, axis=0).max())) if len(endpoints) else 35.0
    eye = b.EyeControls3D(
        position=target + [span, -span, -span], look_target=target, eye_up=[0, 0, -1]
    )
    for variant in VARIANTS:
        tabs.append(
            b.Horizontal(
                b.Spatial3DView(origin="world", name="3D comparison", eye_controls=eye),
                b.Grid(
                    *[
                        b.Spatial2DView(origin=f"crops/{variant}/{view}", name=view)
                        for view in VIEWS
                    ],
                    grid_columns=3,
                ),
                b.TextDocumentView(origin=f"readout/{variant}", name="Evidence"),
                column_shares=[3, 6, 2],
                name=variant,
            )
        )
    tabs.append(b.Spatial2DView(origin="comparison", name="All before/after crops"))
    tabs.append(
        b.Grid(
            *[
                b.Spatial2DView(origin=f"cameras/recipe-model-top/{view}", name=view)
                for view in VIEWS
            ],
            grid_columns=3,
            name="Full-frame recipe model-top",
        )
    )
    bp = b.Blueprint(
        b.Vertical(
            b.Tabs(*tabs, active_tab=1), b.TextDocumentView(origin="legend"), row_shares=[12, 1]
        ),
        b.TimePanel(timeline="source_frame", state="collapsed", play_state="paused"),
        b.BlueprintPanel(state="collapsed"),
        b.SelectionPanel(state="hidden"),
        auto_layout=False,
        auto_views=False,
    )
    rr.send_blueprint(bp)
    bp.save("finebio-pipette-line-comparison", out / "pipette-lines.rbl")
    rr.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=Path("runs/finebio-pipette-improvement-20260930")
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/finebio-pipette-improvement-20260930/line-comparison"),
    )
    parser.add_argument(
        "--camera-config", type=Path, default=Path("configs/finebio/cameras/P03_03_01.json")
    )
    parser.add_argument("--pipettes", type=Path, default=Path("configs/finebio/pipettes.json"))
    args = parser.parse_args()
    run(args.base, args.out, args.camera_config, args.pipettes)


if __name__ == "__main__":
    main()
