"""Render review-only G3 contact sheets from completed normalized smoke artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TARGET_TIMES_SECONDS = (0.0, 5.0, 299 / 30)
ANALYSIS_FPS = 30
PANEL_WIDTH = 600
HEADER_HEIGHT = 46
LEGEND_HEIGHT = 128
VIEW_HEADER_HEIGHT = 34
SCREEN_FOOTER_HEIGHT = 50
COLORS_BGR = ((80, 200, 80), (60, 190, 255), (220, 120, 255))


def review_frame_indices(timestamps: tuple[float, ...] = TARGET_TIMES_SECONDS) -> tuple[int, ...]:
    """Map the fixed review timestamps to their nearest valid 30 FPS frame."""
    return tuple(round(timestamp * ANALYSIS_FPS) for timestamp in timestamps)


def _load_observations(path: Path) -> dict[int, dict[str, Any]]:
    observations: dict[int, dict[str, Any]] = {}
    with path.open() as file:
        for line in file:
            observation = json.loads(line)
            observations[int(observation["analysis_frame_index"])] = observation
    return observations


def _load_frame(capture: Any, frame_index: int) -> Any:
    capture.set(1, frame_index)  # cv2.CAP_PROP_POS_FRAMES without importing cv2 at module load.
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"could not decode proxy frame {frame_index}")
    return frame


def _mask_path(object_: dict[str, Any], run_directory: Path) -> Path | None:
    mask = object_.get("mask")
    return run_directory / mask["uri"] if mask else None


def _overlay_observation(
    frame: Any, observation: dict[str, Any], run_directory: Path
) -> tuple[Any, int]:
    import cv2

    height, width = frame.shape[:2]
    masks_applied = 0
    for object_index, object_ in enumerate(observation["objects"]):
        color = COLORS_BGR[object_index % len(COLORS_BGR)]
        mask_path = _mask_path(object_, run_directory)
        if mask_path and mask_path.is_file():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                if mask.shape != frame.shape[:2]:
                    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                overlay = frame.copy()
                overlay[mask > 0] = color
                frame = cv2.addWeighted(frame, 0.60, overlay, 0.40, 0)
                masks_applied += 1
        box = object_["box"]
        x1, y1 = round(box["x"] * width), round(box["y"] * height)
        x2 = round((box["x"] + box["width"]) * width)
        y2 = round((box["y"] + box["height"]) * height)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{object_['label']} | {object_['object_id']}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )
    return frame, masks_applied


def _panel(frame: Any, frame_index: int, masks_applied: int) -> Any:
    import cv2

    height, width = frame.shape[:2]
    panel_height = round(height * PANEL_WIDTH / width)
    panel = cv2.resize(frame, (PANEL_WIDTH, panel_height))
    mask_note = (
        f"{masks_applied} external mask(s)" if masks_applied else "boxes only; 5 FPS mask cadence"
    )
    cv2.rectangle(panel, (0, 0), (PANEL_WIDTH, 28), (20, 20, 20), -1)
    cv2.putText(
        panel,
        f"t={frame_index / ANALYSIS_FPS:.3f}s | analysis frame {frame_index} | {mask_note}",
        (8, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return panel


def _legend(initial: dict[str, Any], concepts: list[str], width: int) -> Any:
    import cv2
    import numpy as np

    legend = np.full((LEGEND_HEIGHT, width, 3), 245, dtype=np.uint8)
    initial_ids = {object_["label"]: object_["object_id"] for object_ in initial["objects"]}
    cv2.putText(
        legend,
        "Factual legend — target concept | returned track ID | first-frame status",
        (12, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    for row, concept in enumerate(concepts):
        track_id = initial_ids.get(concept, "—")
        status = (
            "present at initialization" if concept in initial_ids else "absent at initialization"
        )
        cv2.putText(
            legend,
            f"{concept} | {track_id} | {status}",
            (16, 53 + row * 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    return legend


def _view_panels(
    run_directory: Path, repository_root: Path, frame_indices: tuple[int, ...]
) -> tuple[dict[str, Any], list[Any]]:
    """Load a completed smoke run and render only its fixed review frames."""
    import cv2

    runtime = json.loads((run_directory / "runtime_settings.json").read_text())
    observations = _load_observations(run_directory / "observations.jsonl")
    if any(index not in observations for index in frame_indices):
        raise ValueError(
            f"required review frames missing from {run_directory / 'observations.jsonl'}"
        )
    proxy_path = repository_root / runtime["proxy"]
    capture = cv2.VideoCapture(str(proxy_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open approved proxy: {proxy_path}")
    try:
        panels = []
        for frame_index in frame_indices:
            frame = _load_frame(capture, frame_index)
            rendered, masks_applied = _overlay_observation(
                frame, observations[frame_index], run_directory
            )
            panels.append(_panel(rendered, frame_index, masks_applied))
    finally:
        capture.release()
    return runtime, panels


def render_viewpoint_screen_contact_sheet(
    run_directories: dict[str, Path], repository_root: Path, output_path: Path
) -> Path:
    """Render the fixed four-view screen without invoking or judging a model."""
    import cv2
    import numpy as np

    expected_views = (
        "ego-hmc21176875",
        "ego-hmc21176623",
        "ego-hmc21110305",
        "ego-hmc21179183",
    )
    if tuple(run_directories) != expected_views:
        raise ValueError(f"viewpoint screen must contain exactly {expected_views}")

    frame_indices = review_frame_indices()
    rows: list[Any] = []
    for view_id in expected_views:
        runtime, panels = _view_panels(run_directories[view_id], repository_root, frame_indices)
        if runtime["view_id"] != view_id:
            raise ValueError(f"run directory for {view_id} identifies {runtime['view_id']}")
        pane_height = max(panel.shape[0] for panel in panels)
        padded = [
            cv2.copyMakeBorder(
                panel,
                0,
                pane_height - panel.shape[0],
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
            for panel in panels
        ]
        initial = _load_observations(run_directories[view_id] / "observations.jsonl")[0]
        initial_labels = ", ".join(item["label"] for item in initial["objects"]) or "none"
        view_header = np.full(
            (VIEW_HEADER_HEIGHT, PANEL_WIDTH * len(padded), 3), 35, dtype=np.uint8
        )
        cv2.putText(
            view_header,
            f"{view_id.upper()} | initial text-concept outputs: {initial_labels}",
            (12, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        rows.append(np.vstack((view_header, np.hstack(padded))))

    width = PANEL_WIDTH * len(frame_indices)
    title = np.full((HEADER_HEIGHT, width, 3), 35, dtype=np.uint8)
    cv2.putText(
        title,
        "G3 EGO VIEWPOINT SCREEN — COMPLETED 10-SECOND SAM3 SMOKES ONLY",
        (14, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    footer = np.full((SCREEN_FOOTER_HEIGHT, width, 3), 245, dtype=np.uint8)
    cv2.putText(
        footer,
        "Review-only overlays at 0.000, 5.000, and 9.967 s; "
        "no ground truth or visual accuracy claim.",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = np.vstack((title, *rows, footer))
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"could not write contact sheet: {output_path}")
    return output_path


def render_contact_sheet(
    run_directory: Path,
    repository_root: Path,
    *,
    timestamps: tuple[float, ...] = TARGET_TIMES_SECONDS,
) -> Path:
    """Create one contact sheet without loading a model or modifying run outputs."""
    import cv2
    import numpy as np

    runtime = json.loads((run_directory / "runtime_settings.json").read_text())
    observations = _load_observations(run_directory / "observations.jsonl")
    frame_indices = review_frame_indices(timestamps)
    if any(index not in observations for index in frame_indices):
        raise ValueError(
            f"required review frames missing from {run_directory / 'observations.jsonl'}"
        )
    proxy_path = repository_root / runtime["proxy"]
    capture = cv2.VideoCapture(str(proxy_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open approved proxy: {proxy_path}")
    panels = []
    try:
        for frame_index in frame_indices:
            frame = _load_frame(capture, frame_index)
            rendered, masks_applied = _overlay_observation(
                frame, observations[frame_index], run_directory
            )
            panels.append(_panel(rendered, frame_index, masks_applied))
    finally:
        capture.release()
    pane_height = max(panel.shape[0] for panel in panels)
    padded = [
        cv2.copyMakeBorder(
            panel,
            0,
            pane_height - panel.shape[0],
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        for panel in panels
    ]
    header = np.full((HEADER_HEIGHT, PANEL_WIDTH * len(padded), 3), 35, dtype=np.uint8)
    title = (
        f"G3 FULL STATIC QA — {runtime['view_id'].upper()}"
        if runtime.get("run_profile") == "g3_full_static_candidate"
        else f"G4 E4 CANDIDATE QA — {runtime['view_id'].upper()}"
        if runtime.get("run_profile") == "g4_e4_60_second_candidate"
        else f"FULL EGO MANUAL-SEED MULTIPLEXED QA — {runtime['view_id'].upper()}"
        if runtime.get("run_profile") == "full_ego_manual_seed_multiplexed_baseline"
        else f"G3 HUMAN REVIEW — {runtime['view_id'].upper()} — COMPLETED SAM3 SMOKE ONLY"
    )
    cv2.putText(
        header,
        title,
        (14, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    image = np.vstack(
        (header, np.hstack(padded), _legend(observations[0], runtime["concepts"], header.shape[1]))
    )
    output_directory = run_directory / "g3_review"
    output_directory.mkdir(exist_ok=True)
    suffix = (
        "full_run_qa"
        if runtime.get("run_profile") == "g3_full_static_candidate"
        else "g4_e4_candidate_qa"
        if runtime.get("run_profile") == "g4_e4_60_second_candidate"
        else "full_ego_manual_seed_multiplexed_qa"
        if runtime.get("run_profile") == "full_ego_manual_seed_multiplexed_baseline"
        else "contact_sheet"
    )
    output_path = output_directory / f"{runtime['view_id']}_{suffix}.png"
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"could not write contact sheet: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a model-free G3 review contact sheet.")
    parser.add_argument("--run-directory", type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--timestamps", type=float, nargs="+", default=list(TARGET_TIMES_SECONDS))
    parser.add_argument(
        "--viewpoint-screen-run",
        action="append",
        metavar="VIEW=RUN_DIRECTORY",
        help="Provide all four fixed ego-view screen runs; cannot be combined with --timestamps.",
    )
    parser.add_argument("--output", type=Path, help="Required with --viewpoint-screen-run.")
    args = parser.parse_args()
    if args.viewpoint_screen_run:
        if args.timestamps != list(TARGET_TIMES_SECONDS) or args.output is None:
            parser.error("viewpoint screen requires --output and the fixed default timestamps")
        try:
            view_runs = {
                entry.split("=", maxsplit=1)[0]: Path(entry.split("=", maxsplit=1)[1]).resolve()
                for entry in args.viewpoint_screen_run
            }
        except IndexError:
            parser.error("--viewpoint-screen-run must use VIEW=RUN_DIRECTORY")
        output = render_viewpoint_screen_contact_sheet(
            view_runs, args.repository_root.resolve(), args.output.resolve()
        )
        print(f"Wrote G3 viewpoint screen contact sheet: {output}")
        return
    if args.run_directory is None:
        parser.error("--run-directory is required unless --viewpoint-screen-run is used")
    output = render_contact_sheet(
        args.run_directory.resolve(),
        args.repository_root.resolve(),
        timestamps=tuple(args.timestamps),
    )
    print(f"Wrote G3 review contact sheet: {output}")


if __name__ == "__main__":
    main()
