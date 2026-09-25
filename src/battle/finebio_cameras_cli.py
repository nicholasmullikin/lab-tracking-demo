"""``battle-finebio-cameras``: the per-trial FineBio camera solve (plan todo p0-cameras).

``solve`` is the preflight's ``mapping`` subcommand generalised: ArUco ``DICT_6X6_50`` on N
frames per fixed view, marker association by projected centroid with the best cyclic corner
order, every (recording day, camera id) shipped pose ranked by median corner RMS, the day
chosen by the summed residual over the five best cameras, one PnP per camera against the
chosen day's markers, and the pose decision per view: shipped where the chosen day's shipped
pose fits within `SHIPPED_POSE_MAX_RMS_PX`, else ``marker_pnp`` where the PnP fits within the
same bound, else **dropped** (the plan's stop rule; a camera is never faked). The fpv keeps the
shipped per-frame pose with the validity gate parameters recorded, and its marker fit is
measured on every ``--fpv-step``-th frame.

Outputs: the committable `FineBioCameraConfig` JSON (numbers only) and an evidence directory
under ``runs/`` (gitignored: FineBio is non-commercial research data and the overlays are
video frames) with ``mapping.json`` / ``mapping.md``, ``fpv_pose.json`` / ``fpv_pose.md``, one
marker overlay per fixed view and a copy of the config.

    uv run battle-finebio-cameras solve --trial P20_03_01 --seconds 30,60,90 \
        --output configs/finebio/cameras/P20_03_01.json \
        --evidence runs/finebio-cameras-P20_03_01-20260925/

Regression: solving P03_01_01 with ``--seconds 30,60,90`` reproduces the preflight's camera
ids, day, residuals and PnP centres (``tests/fixtures/finebio_preflight/rig_reference.json``)
and the committed ``configs/finebio/cameras/P03_01_01.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import finebio_cameras as fc
from .finebio_frames import read_frame, video_path
from .multiview_schemas import FineBioCameraConfig

DEFAULT_SECONDS = "30,60,90"
DEFAULT_FPV_STEP = 250


# --------------------------------------------------------------------------- helpers


def parse_seconds(text: str) -> list[float]:
    values = [float(token) for token in text.split(",") if token.strip()]
    if not values:
        raise ValueError("--seconds needs at least one value")
    return values


def parse_frames(text: str) -> list[int]:
    """Raw frame indices from ``N``, ``A-B`` (inclusive) and ``A:B:S`` tokens."""
    frames: set[int] = set()
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            first, last, step = (int(v) for v in token.split(":"))
            if step <= 0:
                raise ValueError(f"expected A:B:S with S > 0, got {token!r}")
            frames.update(range(first, last, step))
        elif "-" in token:
            first, last = (int(v) for v in token.split("-", 1))
            frames.update(range(first, last + 1))
        else:
            frames.add(int(token))
    if not frames or min(frames) < 0:
        raise ValueError("frame indices must be >= 0 and non-empty")
    return sorted(frames)


def default_evidence_dir(trial: str, when: datetime | None = None) -> Path:
    stamp = (when or datetime.now(UTC)).strftime("%Y%m%d")
    return Path("runs") / f"finebio-cameras-{trial}-{stamp}"


def draw_markers(
    img: np.ndarray,
    detected: dict[int, np.ndarray],
    matches: list[tuple[int, int, float, np.ndarray]],
    label: str,
    *,
    colour: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    vis = img.copy()
    for marker_id, corners in detected.items():
        cv2.polylines(vis, [corners.astype(np.int32)], True, (0, 0, 255), 2)
        cv2.putText(vis, f"id{marker_id}", tuple(corners[0].astype(int)), 0, 0.7, (0, 0, 255), 2)
    for _, _, rms, proj in matches:
        cv2.polylines(vis, [proj.astype(np.int32)], True, colour, 2)
        cv2.putText(vis, f"{rms:.1f}px", tuple((proj[2] + (4, 18)).astype(int)), 0, 0.7, colour, 2)
    cv2.putText(vis, label, (10, 30), 0, 0.9, (255, 255, 0), 2)
    return vis


# --------------------------------------------------------------------------- report text


def mapping_markdown(report: dict[str, Any], config: FineBioCameraConfig) -> str:
    frames = report["frames"]
    seconds = report.get("seconds")
    where = f"{seconds} s" if seconds else f"raw frames {frames}"
    lines = [
        f"# Camera solve, {report['trial']}",
        "",
        f"Frames at {where} (raw {frames}); ArUco DICT_6X6_50; residual = median corner RMS "
        "over detected markers, best cyclic corner order; PnP against the chosen day's markers.",
        "",
        "| view | camera | best day (px) | chosen day shipped (px) | PnP (px) "
        "| PnP vs shipped (cm) | markers/frame | pose in use |",
        "|---|---|---|---|---|---|---|---|",
    ]
    dropped = config.provenance.get("dropped_views", {})
    for view, info in report["views"].items():
        best = info["best"]
        chosen = info["chosen_day"]
        pnp = chosen["pnp"]
        if view in config.fixed:
            decision = config.fixed[view].provenance
        else:
            decision = "**dropped**"
        shipped_text = (
            "n/a"
            if chosen["median_corner_rms_px"] is None
            else f"{chosen['median_corner_rms_px']:.1f}"
        )
        pnp_text = (
            "n/a" if pnp is None else f"{pnp['corner_rms_px']:.2f} ({pnp['corner_count']} corners)"
        )
        distance_text = (
            "n/a" if chosen["pnp_vs_shipped_cm"] is None else f"{chosen['pnp_vs_shipped_cm']:.2f}"
        )
        lines.append(
            f"| {view} | {best['camera_id']} | {best['day']} {best['median_corner_rms_px']:.1f} | "
            f"{shipped_text} | {pnp_text} | {distance_text} | "
            f"{info['markers_detected_per_frame']} | {decision} |"
        )
    ranked = report["day_decision"]["by_summed_residual"]
    lines += [
        "",
        "Day by summed residual over the five chosen cameras: "
        + ", ".join(f"{d} {r:.1f}" for d, r in ranked[:4])
        + f" -> **{report['day_decision']['chosen']}** "
        f"(weighted vote {report['day_decision']['weighted_vote']}).",
        "Camera ids form a permutation: "
        f"**{report['camera_permutation_ok']}** "
        f"({ {v: i['best']['camera_id'] for v, i in report['views'].items()} }).",
        f"Pose rule: shipped kept where the chosen day's median corner RMS <= "
        f"{config.provenance['shipped_pose_max_rms_px']} px, else marker PnP where its RMS <= "
        f"{config.provenance['pnp_drop_over_px']} px, else dropped.",
    ]
    crosscheck = report["day_decision"].get("fpv_marker_rms_by_day")
    if crosscheck:
        lines.append(
            "Second witness, the shipped fpv pose against each candidate day's markers "
            "(median corner RMS): "
            + ", ".join(
                f"{d} {'n/a' if r is None else f'{r["median"]:.1f} px'}"
                for d, r in crosscheck.items()
            )
            + "."
        )
    if dropped:
        lines.append(
            "Dropped views: " + "; ".join(f"{v}: {d['reason']}" for v, d in dropped.items()) + "."
        )
    for view, other in report["views"].items():
        runner = other["runner_up_other_camera"]
        others = ", ".join(
            f"{o['day']} {o['median_corner_rms_px']:.1f}"
            for o in other["same_camera_other_days"][:3]
        )
        lines.append(
            f"- {view}: best other camera cam{runner['camera_id']} "
            f"{runner['median_corner_rms_px']:.1f} px; same camera other days: {others}."
        )
    return "\n".join(lines) + "\n"


def fpv_markdown(check: dict[str, Any]) -> str:
    if not check["corner_rms_px"]:
        return "no markers detected in the fpv frames checked\n"
    rms = check["corner_rms_px"]
    vel = check["velocity_gate"]
    text = (
        f"fpv pose vs ArUco ({check['trial']}, day {check['day']}, every {check['step']}th frame): "
        f"{check['frames_with_markers']}/{check['frames_checked']} checked frames had markers; "
        f"corner RMS median {rms['median']:.2f} px, p90 {rms['p90']:.2f}, max {rms['max']:.1f}; "
        f">{check['residual_gate_px']:.0f} px on {check['frames_over_gate']}"
    )
    if check["pnp_vs_shipped_cm"]:
        pnp = check["pnp_vs_shipped_cm"]
        text += (
            f"; marker-PnP centre vs shipped median {pnp['median']:.2f} cm, p90 {pnp['p90']:.2f}, "
            f"max {pnp['max']:.1f} (n={pnp['n']})"
        )
    text += (
        f"; velocity gate: {vel['pairs_over_gate']}/{vel['consecutive_valid_pairs']} consecutive "
        f"valid pairs step more than the gate (p99 step {vel['step_cm_p99']:.2f} cm)."
    )
    return text + "\n"


# --------------------------------------------------------------------------- solve


def solve_trial(
    trial: str,
    *,
    frames: list[int],
    seconds: list[float] | None,
    evidence: Path,
    output: Path,
    fpv_step: int,
    overlays: bool = True,
    pnp_over_px: float = fc.SHIPPED_POSE_MAX_RMS_PX,
    day_candidates: int = 3,
    command: str | None = None,
) -> tuple[dict[str, Any], FineBioCameraConfig, dict[str, Any] | None]:
    """Run the solve for one trial; writes the evidence directory and the config."""
    evidence.mkdir(parents=True, exist_ok=True)

    def read(view: str, frame: int) -> np.ndarray:
        return read_frame(video_path(trial, view), frame)

    report, kept = fc.solve_mapping(trial, frames, read, seconds=seconds)
    day = report["day_decision"]["chosen"]
    fpv_check = None
    provenance: dict[str, Any] = {
        "source": f"{evidence}/mapping.json (battle-finebio-cameras, {datetime.now(UTC).date()})",
        "command": command or " ".join(sys.argv),
    }
    if fpv_step > 0:

        def read_fpv(frame: int) -> np.ndarray:
            return read_frame(video_path(trial, "fpv"), frame)

        fpv_check = fc.fpv_pose_check(trial, day, read_fpv, step=fpv_step)
        provenance["fpv_pose_check"] = {k: v for k, v in fpv_check.items() if k != "rows"}
        # Second witness for the day: the shipped fpv pose is the authors' marker PnP, so it
        # fits only the marker layout of the day it was computed against.
        candidates = [d for d, _ in report["day_decision"]["by_summed_residual"][:day_candidates]]
        crosscheck = {}
        for candidate in candidates:
            check = fc.fpv_pose_check(trial, candidate, read_fpv, step=max(fpv_step * 6, 300))
            crosscheck[candidate] = None if not check["corner_rms_px"] else check["corner_rms_px"]
        report["day_decision"]["fpv_marker_rms_by_day"] = crosscheck
        provenance["fpv_day_crosscheck"] = crosscheck
    config = fc.camera_config_from_mapping(
        report, trial, pnp_over_px=pnp_over_px, drop_over_px=pnp_over_px, provenance=provenance
    )
    (evidence / "mapping.json").write_text(json.dumps(report, indent=1) + "\n")
    (evidence / "mapping.md").write_text(mapping_markdown(report, config))
    if fpv_check is not None:
        (evidence / "fpv_pose.json").write_text(json.dumps(fpv_check, indent=1) + "\n")
        (evidence / "fpv_pose.md").write_text(fpv_markdown(fpv_check))
    if overlays:
        write_overlays(trial, report, config, kept, evidence)
    fc.write_camera_config(config, output)
    fc.write_camera_config(config, evidence / "camera_config.json")
    return report, config, fpv_check


def write_overlays(
    trial: str,
    report: dict[str, Any],
    config: FineBioCameraConfig,
    kept: dict[str, list[tuple[int, np.ndarray, dict[int, np.ndarray]]]],
    evidence: Path,
) -> None:
    """First solved frame per fixed view: ArUco (red), the chosen day's shipped pose (green)
    and, where the pose in use is the marker PnP, that pose (cyan)."""
    day = report["day_decision"]["chosen"]
    markers = fc.marker_points(day).reshape(-1, 3)
    cameras = fc.cameras_from_config(config)
    for view, frames in kept.items():
        if not frames:
            continue
        frame, img, detected = frames[0]
        camera_id = report["views"][view]["best"]["camera_id"]
        shipped = fc.fixed_camera(day, camera_id, view)
        proj = shipped.project(markers).reshape(-1, 4, 2)
        matches = fc.match_markers(detected, proj) if detected else []
        state = config.fixed[view].provenance if view in config.fixed else "dropped"
        label = (
            f"{trial} {view} f{frame}: shipped cam{camera_id}/{day} (green) vs ArUco (red); {state}"
        )
        vis = draw_markers(img, detected, matches, label)
        if state == "marker_pnp":
            proj_pnp = cameras[view].project(markers).reshape(-1, 4, 2)
            cyan = (255, 255, 0)
            for _, _, rms, quad in fc.match_markers(detected, proj_pnp):
                cv2.polylines(vis, [quad.astype(np.int32)], True, cyan, 2)
                anchor = tuple((quad[0] + (4, -8)).astype(int))
                cv2.putText(vis, f"pnp {rms:.1f}px", anchor, 0, 0.6, cyan, 2)
        cv2.imwrite(str(evidence / f"{view}_markers.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])


def summary_lines(config: FineBioCameraConfig) -> list[str]:
    lines = [f"{config.trial}: day {config.recording_day}"]
    views = config.provenance.get("views", {})
    for view, cam in config.fixed.items():
        extra = views.get(view, {})
        distance = extra.get("pnp_vs_shipped_cm")
        lines.append(
            f"  {view} = cam{cam.camera_id} {cam.provenance}: in use "
            f"{cam.marker_fit_residual_px:.2f} px, shipped "
            f"{cam.shipped_marker_residual_px:.2f} px, PnP vs shipped "
            + ("n/a" if distance is None else f"{distance:.2f} cm")
        )
    for view, info in config.provenance.get("dropped_views", {}).items():
        lines.append(f"  {view} = cam{info['camera_id']} DROPPED: {info['reason']}")
    fpv = config.provenance.get("fpv_pose_check")
    if fpv and fpv.get("corner_rms_px"):
        lines.append(
            f"  fpv: shipped pose valid {config.fpv.valid_pose_fraction:.3f}; marker RMS median "
            f"{fpv['corner_rms_px']['median']:.2f} px on {fpv['frames_with_markers']}/"
            f"{fpv['frames_checked']} frames, {fpv['frames_over_gate']} over the "
            f"{fpv['residual_gate_px']:.0f} px gate"
        )
    crosscheck = config.provenance.get("fpv_day_crosscheck")
    if crosscheck:
        lines.append(
            "  day cross-check (fpv marker RMS): "
            + ", ".join(
                f"{d} {'n/a' if r is None else f'{r["median"]:.1f} px'}"
                for d, r in crosscheck.items()
            )
        )
    return lines


# --------------------------------------------------------------------------- cli


def cmd_solve(args: argparse.Namespace) -> int:
    seconds = None if args.frames else parse_seconds(args.seconds)
    frames = parse_frames(args.frames) if args.frames else [fc.seconds_to_frame(s) for s in seconds]
    evidence = args.evidence or default_evidence_dir(args.trial)
    output = args.output or Path("configs/finebio/cameras") / f"{args.trial}.json"
    _, config, _ = solve_trial(
        args.trial,
        frames=frames,
        seconds=seconds,
        evidence=evidence,
        output=output,
        fpv_step=args.fpv_step,
        overlays=not args.no_overlays,
        pnp_over_px=args.max_rms_px,
    )
    print("\n".join(summary_lines(config)))
    print(f"config -> {output}\nevidence -> {evidence}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    config = fc.read_camera_config(args.config)
    print("\n".join(summary_lines(config)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="battle-finebio-cameras",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("solve", help="solve one trial's cameras from ArUco markers")
    p.add_argument("--trial", required=True)
    p.add_argument("--seconds", default=DEFAULT_SECONDS, help="comma-separated times (s)")
    p.add_argument("--frames", default=None, help="raw frames N,A-B,A:B:S (overrides --seconds)")
    p.add_argument("--output", type=Path, default=None, help="config JSON (default configs/...)")
    p.add_argument("--evidence", type=Path, default=None, help="evidence dir (default runs/...)")
    p.add_argument("--fpv-step", type=int, default=DEFAULT_FPV_STEP, help="0 skips the fpv check")
    p.add_argument("--max-rms-px", type=float, default=fc.SHIPPED_POSE_MAX_RMS_PX)
    p.add_argument("--no-overlays", action="store_true")
    p.set_defaults(func=cmd_solve)
    p = sub.add_parser("show", help="print a camera config's decisions")
    p.add_argument("--config", type=Path, required=True)
    p.set_defaults(func=cmd_show)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
