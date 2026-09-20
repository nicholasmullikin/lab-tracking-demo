"""Contact sheet of every fetched view of one Assembly101 recording (review only, no model).

One row per view, five columns: proxy frame 0 and four frames evenly spaced over the rest of
the fetched window.  Each panel is labelled with the view, the proxy frame, the source time
and the dataset's coarse action active at that time, so a human or a seed-transfer worker can
see where the parts sit in every camera before any run starts.  Nothing here is a claim about
any method; the coarse labels are the dataset's (CC BY-NC 4.0).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .assembly101_fetch_view import proxy_path
from .assembly101_recordings import (
    ANNOTATION_FPS,
    RECORDING_1,
    Assembly101Recording,
    get_recording,
)

PANEL_WIDTH = 480
LABEL_HEIGHT = 30
TITLE_HEIGHT = 44
COLUMNS = 5


def contact_sheet_frames(frame_count: int, columns: int = COLUMNS) -> tuple[int, ...]:
    """Frame 0 plus `columns - 1` frames at multiples of `frame_count / (columns - 1)`, the
    last one clamped to the final frame."""
    if frame_count < columns:
        raise ValueError(f"need at least {columns} frames, got {frame_count}")
    step = frame_count / (columns - 1)
    return tuple(min(frame_count - 1, round(i * step)) for i in range(columns))


def load_coarse_labels(path: Path) -> tuple[tuple[int, int, str], ...]:
    """(start, end_exclusive, action) rows of a coarse labels file, 30 fps annotation frames."""
    rows: list[tuple[int, int, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3 or not parts[0].strip():
            continue
        rows.append((int(parts[0]), int(parts[1]), parts[2].strip()))
    return tuple(rows)


def coarse_action_at(labels: tuple[tuple[int, int, str], ...], annotation_frame: int) -> str | None:
    for start, end, action in labels:
        if start <= annotation_frame < end:
            return action
    return None


def contact_sheet_path(recording: Assembly101Recording) -> Path:
    return recording.derived_root / (
        f"contact_sheet_{recording.window_start_seconds:.3f}-{recording.window_end_seconds:.3f}.png"
    )


def _read_frame(capture: cv2.VideoCapture, index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"could not decode proxy frame {index}")
    return frame


def _label(panel: np.ndarray, text: str) -> np.ndarray:
    bar = np.full((LABEL_HEIGHT, panel.shape[1], 3), 20, dtype=np.uint8)
    cv2.putText(bar, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 245, 245), 1, cv2.LINE_AA)
    return np.vstack((bar, panel))


def render_contact_sheet(
    recording: Assembly101Recording,
    *,
    repository_root: Path,
    output: Path | None = None,
    views: tuple[str, ...] | None = None,
) -> Path:
    repository_root = repository_root.resolve()
    views = views if views is not None else recording.all_views
    labels_path = repository_root / recording.coarse_labels_path
    labels = load_coarse_labels(labels_path) if labels_path.is_file() else ()
    core_start, core_end = recording.core_proxy_frame_range
    rows: list[np.ndarray] = []
    frames: tuple[int, ...] | None = None
    for view in views:
        path = repository_root / proxy_path(view, recording=recording)
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise FileNotFoundError(f"cannot open proxy {path}")
        try:
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if frames is None:
                frames = contact_sheet_frames(count)
            panels: list[np.ndarray] = []
            for index in frames:
                frame = _read_frame(capture, index)
                height, width = frame.shape[:2]
                panel = cv2.resize(frame, (PANEL_WIDTH, round(height * PANEL_WIDTH / width)))
                source_seconds = recording.window_start_seconds + index / 30
                action = coarse_action_at(labels, round(source_seconds * ANNOTATION_FPS))
                core = "core" if core_start <= index < core_end else "margin"
                text = f"{view} f{index} {source_seconds:.2f}s [{core}] {action or '-'}"
                panels.append(_label(panel, text))
        finally:
            capture.release()
        tallest = max(p.shape[0] for p in panels)
        padded = [
            cv2.copyMakeBorder(p, 0, tallest - p.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=0)
            for p in panels
        ]
        rows.append(np.hstack(padded))
    widest = max(r.shape[1] for r in rows)
    rows = [
        cv2.copyMakeBorder(r, 0, 0, 0, widest - r.shape[1], cv2.BORDER_CONSTANT, value=0)
        for r in rows
    ]
    title = np.full((TITLE_HEIGHT, widest, 3), 35, dtype=np.uint8)
    cv2.putText(
        title,
        f"{recording.recording_id}  window {recording.window_start_seconds:.3f}-"
        f"{recording.window_end_seconds:.3f} s  core proxy frames [{core_start}, {core_end})  "
        "review only; coarse labels are the dataset's (CC BY-NC 4.0)",
        (10, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    image = np.vstack((title, *rows))
    target = repository_root / (output if output is not None else contact_sheet_path(recording))
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), image):
        raise RuntimeError(f"could not write {target}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", help="registry label or recording id; default recording 1")
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, help="default: data/derived/.../contact_sheet_*.png")
    parser.add_argument("--view", action="append", help="subset of views; default all fetched")
    args = parser.parse_args()
    recording = (
        get_recording(args.recording, args.repository_root) if args.recording else RECORDING_1
    )
    target = render_contact_sheet(
        recording,
        repository_root=args.repository_root,
        output=args.output,
        views=tuple(args.view) if args.view else None,
    )
    print(target)


if __name__ == "__main__":
    main()
