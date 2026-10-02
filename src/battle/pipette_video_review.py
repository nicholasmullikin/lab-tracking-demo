"""Export full-video per-second plots, original/mask comparisons and a Rerun recording."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from .pipette_line_review import VARIANTS, VIEWS, draw_overlay, panel
from .pipette_video_metrics import paired_residuals

COLOURS = ("#ffd21e", "#46ff46", "#ff50e6")
METRICS = {
    "median_loo_px": "Consistency px",
    "median_paired_delta_px": "Paired change px",
    "fit_fraction": "Fit coverage",
    "direction_fraction": "Direction coverage",
    "mean_axis_views": "Usable camera axes",
    "1s_midpoint_cm_s_median": "Apparent cm/s",
    "1s_axis_deg_s_median": "Axis degrees/s",
    "1s_directed_deg_s_median": "Resolved direction degrees/s",
}


def display_frames(count: int, fps: float) -> dict[int, int]:
    """Sample bins containing native frames, avoiding a nonexistent image at the duration edge."""
    return {math.ceil(second * fps): second for second in range(math.floor((count - 1) / fps) + 1)}


def correction_counts(second: int, fps: float, count: int, schedule: dict[str, list[int]]) -> dict:
    """Mark native intervals spanning a reviewed-mask reset, which can create apparent motion."""
    frames = range(math.ceil(second * fps), min(count, math.ceil((second + 1) * fps)))
    events = [frame for times in schedule.values() for frame in times]
    unique = set(events)
    lag = round(fps)
    return {
        "correction_camera_events": sum(frame in frames for frame in events),
        "native_frames_with_correction": sum(frame in unique for frame in frames),
        "step_native_intervals_crossing_correction": sum(
            frame > 0 and frame in unique for frame in frames
        ),
        "1s_native_intervals_crossing_correction": sum(
            frame >= lag and any(frame - lag < event <= frame for event in unique)
            for frame in frames
        ),
    }


def shared_crop(image: np.ndarray, masks: list[np.ndarray]) -> tuple[slice, slice]:
    """Keep all variants in one crop, with enough context to expose adjacent-object leakage."""
    ys, xs = np.nonzero(np.logical_or.reduce(masks))
    if not len(xs):
        return slice(None), slice(None)
    return (
        slice(max(0, int(ys.min()) - 100), min(image.shape[0], int(ys.max()) + 101)),
        slice(max(0, int(xs.min()) - 100), min(image.shape[1], int(xs.max()) + 101)),
    )


def plots(root: Path, rows: list[dict], scope: str = "Full video") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("dark_background")
    fig, axes = plt.subplots(7, 1, figsize=(14, 21), sharex=True, constrained_layout=True)
    for variant, colour in zip(VARIANTS, COLOURS, strict=True):
        series = [x for x in rows if x["variant"] == variant]
        seconds = [x["second"] for x in series]
        for axis, key in zip(
            (axes[0], axes[1], axes[3], axes[4], axes[5], axes[6]),
            (
                "median_loo_px",
                "median_paired_delta_px",
                "mean_axis_views",
                "1s_midpoint_cm_s_median",
                "1s_axis_deg_s_median",
                "1s_directed_deg_s_median",
            ),
            strict=True,
        ):
            axis.plot(
                seconds,
                [np.nan if x[key] is None else x[key] for x in series],
                color=colour,
                label=variant,
                linewidth=1,
            )
        axes[2].plot(
            seconds,
            [x["fit_fraction"] for x in series],
            color=colour,
            linewidth=1,
            label=variant + " fit",
        )
        axes[2].plot(
            seconds,
            [x["direction_fraction"] for x in series],
            color=colour,
            linewidth=1,
            linestyle=":",
            label=variant + " direction",
        )
    for axis, label in zip(
        axes,
        (
            "Dropped-camera consistency (px)",
            "Paired change vs earlier masks (px; lower is better)",
            "Fraction of native frames",
            "Cameras with a usable local axis",
            "Apparent visible-midpoint speed (cm/s)",
            "Unoriented axis rate (degrees/s)",
            "Resolved direction rate (degrees/s)",
        ),
        strict=True,
    ):
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    axes[1].axhline(0, color="white", alpha=0.5, linewidth=1)
    axes[2].set_ylim(-0.03, 1.03)
    axes[3].set_ylim(-0.1, 6.1)
    axes[0].legend(loc="upper left", ncol=3)
    axes[2].legend(loc="lower left", ncol=2, fontsize=8)
    axes[-1].set_xlabel("Raw video time (seconds)")
    fig.suptitle(
        f"{scope}: native-rate blue-pipette comparison, summarized each second\n"
        "Consistency and apparent motion; no ground-truth accuracy or velocity"
    )
    fig.savefig(root / "per-second.png", dpi=150)
    fig.savefig(root / "per-second.svg")
    plt.close(fig)


def export(root: Path) -> None:
    import csv

    import rerun as rr
    import rerun.blueprint as b

    config = json.loads((root / "config.json").read_text())
    rows = json.loads((root / "per-second.json").read_text())
    fps = config["native_fps"]
    count = config["views"]["T1"]["frame_count"]
    schedule = config.get("corrections_by_view", {})
    for row in rows:
        row.update(correction_counts(row["second"], fps, count, schedule))
    (root / "per-second.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    with (root / "per-second.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    samples = display_frames(count, fps)
    captures = {v: cv2.VideoCapture(config["views"][v]["video"]) for v in VIEWS}
    full_video = all(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == count for cap in captures.values())
    scope = "Full video" if full_video else "Early segment"
    plots(root, rows, scope)
    sheets = root / "sheets"
    sheets.mkdir(exist_ok=True)
    rr.init("finebio-pipette-full-video", recording_id=f"raw0-{count - 1}-native-v1-20261002")
    rr.save(root / "pipette-full-video.rrd")
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_DOWN, static=True)
    for variant, colour in zip(VARIANTS, COLOURS, strict=True):
        rgb = tuple(int(colour[i : i + 2], 16) for i in (1, 3, 5))
        for key in METRICS:
            rr.log(
                f"metrics/{key}/{variant}",
                rr.SeriesLines(colors=[rgb], names=[variant]),
                static=True,
            )
    rr.log(
        "method",
        rr.TextDocument(
            "Every native frame analyzed in six cameras. "
            "Images and 3D lines displayed once per second. "
            "Cyan mask; yellow local axis; green projected visible extent; "
            "magenta dispensing arrow only on cue agreement. "
            "Earlier selections compared with model-top and optimistic visual-best study masks. "
            "Seeds at raw 0; matched studied correction events through raw 500; "
            "no later corrections. "
            "Motion includes changing visible extent, occlusion, mask drift and calibration error. "
            "Dropped-camera residuals measure consistency; they do not establish accuracy."
        ),
        static=True,
    )
    totals = {
        v: {
            "frames": 0,
            "fits": 0,
            "directed": 0,
            "loo": [],
            "paired_old": [],
            "paired_new": [],
            "delta": [],
        }
        for v in VARIANTS
    }
    first_ends = None
    with (root / "geometry.jsonl").open() as stream:
        for expected_frame, line in enumerate(stream):
            data = json.loads(line)
            frame = data["raw_frame"]
            if frame != expected_frame:
                raise ValueError(f"Geometry sequence expected {expected_frame}, got {frame}")
            for variant, geometry in data["variants"].items():
                total = totals[variant]
                total["frames"] += 1
                total["fits"] += int(geometry["line"] is not None)
                total["directed"] += int(geometry["direction"]["tip_end"] is not None)
                total["loo"].extend(
                    x["perpendicular_px"]
                    for x in geometry["loo"]
                    if x["fitted"] and x["perpendicular_px"] is not None
                )
                paired = paired_residuals(data["variants"][VARIANTS[0]], geometry)
                total["paired_old"].extend(paired["baseline_px"])
                total["paired_new"].extend(paired["revised_px"])
                total["delta"].extend(paired["delta_px"])
            if frame not in samples:
                for capture in captures.values():
                    if not capture.grab():
                        raise ValueError(f"Source ended at {frame}")
                continue
            second = samples[frame]
            rr.set_time("source_frame", sequence=frame)
            rr.set_time("seconds", duration=frame / fps)
            full_rows, crop_rows = [], []
            for view, capture in captures.items():
                ok, image = capture.read()
                if not ok:
                    raise ValueError(f"Source ended at {frame}/{view}")
                observations = [data["variants"][v]["views"][view] for v in VARIANTS]
                masks = [cv2.imread(x["mask"], 0) > 0 for x in observations]
                overlays = [
                    draw_overlay(
                        image,
                        mask,
                        observation,
                        observation["projected_world_endpoints_px"],
                        data["variants"][variant]["direction"]["tip_end"],
                    )
                    for mask, observation, variant in zip(
                        masks, observations, VARIANTS, strict=True
                    )
                ]
                crop = shared_crop(image, masks)
                images = [image, *overlays]
                titles = [f"{view} original {frame / fps:.2f}s", *[f"{view} {v}" for v in VARIANTS]]
                full_rows.append(
                    np.hstack([panel(im, title) for im, title in zip(images, titles, strict=True)])
                )
                crop_rows.append(
                    np.hstack(
                        [panel(im[crop], title) for im, title in zip(images, titles, strict=True)]
                    )
                )
                for variant, overlay in zip(VARIANTS, overlays, strict=True):
                    pair = np.hstack(
                        [
                            panel(image[crop], f"{view} original {frame / fps:.2f}s"),
                            panel(overlay[crop], f"{view} {variant}"),
                        ]
                    )
                    rr.log(
                        f"camera-pairs/{variant}/{view}",
                        rr.Image(cv2.cvtColor(pair, cv2.COLOR_BGR2RGB)).compress(jpeg_quality=94),
                    )
            full, cropped = np.vstack(full_rows), np.vstack(crop_rows)
            cv2.imwrite(str(sheets / f"{second:03d}-full.jpg"), full)
            cv2.imwrite(str(sheets / f"{second:03d}-crops.jpg"), cropped)
            rr.log(
                "comparison/full",
                rr.Image(cv2.cvtColor(full, cv2.COLOR_BGR2RGB)).compress(jpeg_quality=92),
            )
            rr.log(
                "comparison/crops",
                rr.Image(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)).compress(jpeg_quality=94),
            )
            for variant, colour in zip(VARIANTS, COLOURS, strict=True):
                origin = f"world/{variant}"
                rr.log(origin, rr.Clear(recursive=True))
                geometry = data["variants"][variant]
                fitted = geometry["line"]
                if fitted and fitted["endpoints"]:
                    ends = np.array(fitted["endpoints"])
                    if first_ends is None:
                        first_ends = ends
                    rgb = tuple(int(colour[i : i + 2], 16) for i in (1, 3, 5))
                    rr.log(
                        origin + "/visible_extent", rr.LineStrips3D([ends], colors=rgb, radii=0.12)
                    )
                    tip = geometry["direction"]["tip_end"]
                    if tip is not None:
                        rr.log(
                            origin + "/direction",
                            rr.Arrows3D(
                                origins=[ends[1 - tip]],
                                vectors=[ends[tip] - ends[1 - tip]],
                                colors=rgb,
                                radii=0.15,
                            ),
                        )
                row = next(x for x in rows if x["variant"] == variant and x["second"] == second)
                for key in METRICS:
                    value = row[key]
                    path = f"metrics/{key}/{variant}"
                    rr.log(path, rr.Clear(recursive=False) if value is None else rr.Scalars(value))
            if second % 30 == 0:
                print(f"Exported original/mask review {second}s / {count / fps:.2f}s", flush=True)
    for capture in captures.values():
        capture.release()
    if any(total["frames"] != count for total in totals.values()):
        raise ValueError("Geometry does not cover the complete configured video")
    center = first_ends.mean(axis=0) if first_ends is not None else np.zeros(3)
    eye = b.EyeControls3D(position=center + [60, -60, -60], look_target=center, eye_up=[0, 0, -1])
    bp = b.Blueprint(
        b.Tabs(
            *[
                b.Horizontal(
                    b.Spatial3DView(origin="world", eye_controls=eye, name="Three 3D variants"),
                    b.Grid(
                        *[
                            b.Spatial2DView(origin=f"camera-pairs/{variant}/{view}", name=view)
                            for view in VIEWS
                        ],
                        grid_columns=2,
                    ),
                    column_shares=[3, 7],
                    name=variant,
                )
                for variant in VARIANTS
            ],
            b.Horizontal(
                b.Spatial3DView(origin="world", eye_controls=eye, name="Three 3D variants"),
                b.Spatial2DView(origin="comparison/crops", name="Original / earlier / top / best"),
                column_shares=[3, 7],
                name="Cropped comparison",
            ),
            b.Spatial2DView(origin="comparison/full", name="Full originals and masks"),
            b.Grid(
                *[
                    b.TimeSeriesView(origin=f"metrics/{key}", name=title)
                    for key, title in METRICS.items()
                ],
                grid_columns=2,
                name="Per-second measurements",
            ),
            b.TextDocumentView(origin="method", name="Method and limits"),
            active_tab=1,
        ),
        b.TimePanel(timeline="seconds", state="expanded", play_state="paused"),
        b.BlueprintPanel(state="collapsed"),
        b.SelectionPanel(state="hidden"),
        auto_layout=False,
        auto_views=False,
    )
    rr.send_blueprint(bp)
    bp.save("finebio-pipette-full-video", root / "pipette-full-video.rbl")
    rr.disconnect()
    summary = []
    for variant, total in totals.items():
        summary.append(
            {
                "variant": variant,
                "native_frames": total["frames"],
                "fit_frames": total["fits"],
                "direction_frames": total["directed"],
                "fit_fraction": total["fits"] / total["frames"],
                "direction_fraction": total["directed"] / total["frames"],
                "loo_cells": len(total["loo"]),
                "paired_cells": len(total["delta"]),
                "paired_baseline_median_px": float(np.median(total["paired_old"]))
                if total["paired_old"]
                else None,
                "paired_revised_median_px": float(np.median(total["paired_new"]))
                if total["paired_new"]
                else None,
                "median_paired_delta_px": float(np.median(total["delta"]))
                if total["delta"]
                else None,
                "improved_gt_0p1_px": sum(x < -0.1 for x in total["delta"]),
                "worsened_gt_0p1_px": sum(x > 0.1 for x in total["delta"]),
            }
        )
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        f"# {scope}: blue-pipette comparison",
        "",
        "The configured native-rate segment is complete. Per-second consistency and "
        "apparent motion",
        "are measured for this segment, with missing fits and unresolved directions retained.",
        "",
        "| Variant | Native frames | Fit fraction | Directed fraction | Paired cells "
        "| Earlier median px | Revised median px | Median paired change px |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def format_value(value):
        return "—" if value is None else f"{value:.3f}"

    for result in summary:
        lines.append(
            f"| {result['variant']} | {result['native_frames']} | "
            f"{result['fit_fraction']:.3f} | {result['direction_fraction']:.3f} | "
            f"{result['paired_cells']} | {format_value(result['paired_baseline_median_px'])} | "
            f"{format_value(result['paired_revised_median_px'])} | "
            f"{format_value(result['median_paired_delta_px'])} |"
        )
    lines += [
        "",
        "Measured against: earlier selected masks on matching raw-frame/camera residual cells.",
        "A negative paired change means better dropped-camera consistency.",
        "",
        f"Recording: raw frames 0–{count - 1}, {fps:.8f} fps, {count / fps:.6f} "
        f"seconds, six cameras.",
        f"Display: {len(samples)} images per camera, sampled once per second. "
        f"Inference and metrics use every native frame.",
        "",
        "## Methods",
        "",
        "All cameras begin with verified masks at raw frame 0. Corrections occur only at studied",
        "camera images: T1 at 500, T2 at 100, T3 at 300, T4 at 200. T5 and FPV have "
        "no later corrections.",
        "The control receives earlier masks at exactly the same times as the study variants.",
        "Model-top and optimistic visual-best use the two foreground body/shaft "
        "points plus one background-point recipe.",
        "The remaining video propagates without additional prompts.",
        "",
        f"SAM3.1 uses a {config['max_side_length']}-pixel image side, "
        "one-object independent memory banks, "
        "one prompt slot and four frame slots.",
        "Identical complete correction schedules share inference. Different schedules "
        "share image features,",
        "while keeping their prompt and frame histories separate. Corrections replace "
        "prompt memory and clear frame memory.",
        "Full-resolution masks feed the existing axis extraction, camera calibration, "
        "3D line and colour/taper policies.",
        "FPV poses are read at the raw frame index. Invalid poses are excluded from 3D geometry.",
        "",
        "## Per-second measurements",
        "",
        "CSV bins are floor(raw_frame / native_fps), including the final partial second.",
        "Consistency deltas match camera residuals within each native frame before "
        "summarizing per second.",
        "Motion uses both consecutive frames and a 30-frame lag, with actual elapsed "
        "times of 1/fps and 30/fps.",
        "Position rates measure the fitted visible midpoint in cm/s. Axis rates use "
        "the unoriented line in degrees/s.",
        "Directed rates require resolved arrows at both frames. Arbitrary line-vector "
        "sign changes do not count as motion.",
        "Each rate includes valid sample counts and its median and 90th percentile. "
        "Missing measurements remain blank.",
        "Adjacent-second consistency and coverage changes are also included.",
        "Correction-event counts mark native intervals spanning a reviewed-mask reset. "
        "These counts include intervals without valid motion samples; rates remain unfiltered.",
        "",
        "## Interpretation",
        "",
        "Residuals measure multi-camera consistency, not segmentation or tip accuracy.",
        "Apparent midpoint movement also includes occlusion, changing visible extent, "
        "drift and calibration error.",
        "Lower residuals with fewer surviving camera observations do not establish improvement.",
        "Visual-best is a reviewed selection bound, not an automatic production policy.",
        "Direct inspection notes, when present, are in visual-review.json and "
        "early-six-camera-review/visual-review.json.",
        "Full-video sheets and Rerun support further direct inspection; generation "
        "alone is not a visual quality judgment.",
        "",
        "## Outputs",
        "",
        "- [Per-second CSV](per-second.csv)",
        "- [Per-second plot](per-second.png)",
        "- [Original/mask review](review.html)",
        "- [Native-frame summary](summary.json)",
        "- Rerun: pipette-full-video.rrd plus pipette-full-video.rbl",
        "",
        "Reproduce from the repository root with `.venv/bin/python -m battle.pipette_video_run`.",
        "Verified inputs and independent memory checkpoints are recorded in "
        "config.json and native/*/state.pt.",
    ]
    (root / "README.md").write_text("\n".join(lines) + "\n")
    (root / "review.html").write_text(
        """<!doctype html><meta charset="utf-8"><title>Full pipette video</title>
<style>body{font:16px system-ui;
background:#15212c;
color:white;
margin:24px}img{max-width:100%}a{color:#8cd0ff}select{padding:8px}pre{white-space:pre-wrap}</style>
<h1>SCOPE: native-rate blue-pipette comparison</h1><p>All native frames in this segment analyzed;
 images displayed once per second. Columns: original, earlier masks, recipe model-top, recipe
visual-best. Rows: six cameras.</p>

<p>Cyan mask;
 yellow local axis;
 green projected 3D visible extent;
 magenta dispensing arrow on cue agreement. Consistency and apparent motion are not
ground-truth accuracy or velocity.</p>

<select id="second"></select><img id="crop">
<details><summary>Full-frame originals and masks</summary><img id="full"></details><pre
id="metrics"></pre>
<img src="per-second.png"><p><a href="per-second.csv">Per-second CSV</a> · <a
href="summary.json">Native-frame summary</a> · <a href="README.md">Methods and findings</a></p>

<script>const DATA=PAYLOAD;
 const S=document.getElementById('second');

for(let s=0;
s<SECONDS;
s++)S.add(new Option(s+' seconds',s));

function update(){
let s=+S.value;
let f=String(s).padStart(3,'0');
document.getElementById('crop').src='sheets/'+f+'-crops.jpg';
document.getElementById('full').src='sheets/'+f+'-full.jpg';
document.getElementById('metrics').textContent=JSON.stringify(DATA.filter(x=>x.second===s),null,2)}
S.onchange=update;
update();
</script>""".replace("PAYLOAD", json.dumps(rows))
        .replace("SECONDS", str(len(samples)))
        .replace("SCOPE", scope)
    )
    (root / "export-complete.json").write_text(
        json.dumps(
            {
                "native_frames": count,
                "seconds": count / fps,
                "display_samples": len(samples),
                "full_video": full_video,
            }
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("runs/finebio-pipette-improvement-20260930/full-video-line-comparison"),
    )
    export(parser.parse_args().root)


if __name__ == "__main__":
    main()
