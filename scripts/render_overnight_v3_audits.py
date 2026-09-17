"""Render deterministic raw/stabilized/fallback and mask/contact audit sheets."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from battle.exporter import HAND_CONNECTIONS
from battle.schemas import FrameObservations

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs/interaction-review-overnight-v3/audits"
RAW = ROOT / "runs/wilor-hands-static-20s-audited-source-state/observations.jsonl"
STABILIZED = ROOT / "runs/wilor-hands-stabilized-20s-overnight-v2-r3/observations.jsonl"
HAND_PROVENANCE = ROOT / "runs/wilor-hands-stabilized-20s-overnight-v2-r3/hand_provenance.json"
MASKS = (
    ROOT / "runs/muggledsam-sam3-four-part-static-focused-reassembly-static-c10379-20260916t023700z"
)
MASK_ROWS = MASKS / "observations.jsonl"
VIDEO = ROOT / "runs/wilor-hands-static-20s-audited-source-state/input.mp4"
INDEX = ROOT / "runs/interaction-review-overnight-v3/interaction_review_index.json"

WILOR_GROUPS = {
    "wilor_088_089": range(88, 90),
    "wilor_194_195": range(194, 196),
    "wilor_224_226": range(224, 227),
    "wilor_250_261": range(250, 262),
    "wilor_380_417": range(380, 418),
    "wilor_413_466": range(413, 467),
    "wilor_500_545": range(500, 546),
    "wilor_597_599": range(597, 600),
}
SEGMENTATION_GROUPS = {
    "segmentation_090_092": range(90, 93),
    "segmentation_323_328": range(323, 329),
    "segmentation_341": range(341, 342),
    "segmentation_414": range(414, 415),
    "segmentation_479_481": range(479, 482),
    "segmentation_496_520": range(496, 521),
    "segmentation_548_550": range(548, 551),
    "segmentation_595_599": range(595, 600),
}
MASK_COLORS = {
    "chassis": (70, 130, 255),
    "interior": (60, 210, 150),
    "rear_body": (255, 190, 45),
    "cabin": (255, 95, 100),
}


def _rows(path: Path) -> dict[int, FrameObservations]:
    return {
        row.analysis_frame_index: row
        for row in (
            FrameObservations.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        )
    }


def _frame(capture: cv2.VideoCapture, index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, image = capture.read()
    if not ok:
        raise RuntimeError(f"could not decode frame {index}")
    return cv2.resize(image, (320, 180), interpolation=cv2.INTER_AREA)


def _draw_hands(
    image: np.ndarray, observation: FrameObservations, color: tuple[int, int, int]
) -> None:
    for hand in observation.hands:
        points = [
            (round(point.x * 319), round(point.y * 179))
            for point in hand.landmarks
        ]
        for first, second in HAND_CONNECTIONS:
            cv2.line(image, points[first], points[second], color, 1, cv2.LINE_AA)
        cv2.circle(image, points[0], 3, color, -1, cv2.LINE_AA)


def _label(image: np.ndarray, text: str) -> np.ndarray:
    cv2.rectangle(image, (0, 0), (320, 20), (0, 0, 0), -1)
    cv2.putText(image, text, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    return image


def _grid(cells: list[np.ndarray], columns: int = 2) -> np.ndarray:
    blank = np.zeros_like(cells[0])
    cells += [blank] * ((-len(cells)) % columns)
    return cv2.vconcat(
        [cv2.hconcat(cells[index : index + columns]) for index in range(0, len(cells), columns)]
    )


def render_wilor_audits() -> None:
    raw = _rows(RAW)
    stabilized = _rows(STABILIZED)
    fallback_by_frame: dict[int, set[str]] = {}
    for item in json.loads(HAND_PROVENANCE.read_text(encoding="utf-8")):
        if item["source"] in {"mediapipe", "mediapipe_fallback"}:
            fallback_by_frame.setdefault(item["analysis_frame_index"], set()).add(
                item["output_hand_id"]
            )
    capture = cv2.VideoCapture(str(VIDEO))
    try:
        for name, frames in WILOR_GROUPS.items():
            cells: list[np.ndarray] = []
            for frame in frames:
                raw_panel = _frame(capture, frame)
                stabilized_panel = _frame(capture, frame)
                fallback_panel = _frame(capture, frame)
                _draw_hands(raw_panel, raw[frame], (0, 140, 255))
                _draw_hands(stabilized_panel, stabilized[frame], (255, 210, 50))
                fallback_hands = tuple(
                    hand
                    for hand in stabilized[frame].hands
                    if hand.hand_id in fallback_by_frame.get(frame, set())
                )
                _draw_hands(
                    fallback_panel,
                    stabilized[frame].model_copy(update={"hands": fallback_hands}),
                    (0, 0, 255),
                )
                cell = cv2.hconcat(
                    [
                        _label(raw_panel, f"f{frame} raw WiLoR"),
                        _label(stabilized_panel, f"f{frame} stabilized"),
                        _label(
                            fallback_panel,
                            f"f{frame} MediaPipe fallback={len(fallback_hands)}",
                        ),
                    ]
                )
                cells.append(cell)
            cv2.imwrite(str(OUTPUT / f"{name}.png"), _grid(cells))
    finally:
        capture.release()


def _mask_overlay(image: np.ndarray, observation: FrameObservations) -> np.ndarray:
    result = image.copy()
    for object_ in observation.objects:
        if object_.mask is None:
            continue
        with Image.open(MASKS / object_.mask.uri) as source:
            mask = np.asarray(source.convert("L").resize((320, 180))) > 0
        color = np.asarray(MASK_COLORS[object_.label], dtype=np.uint8)
        result[mask] = (0.62 * result[mask] + 0.38 * color).astype(np.uint8)
    return result


def render_segmentation_audits() -> None:
    rows = _rows(MASK_ROWS)
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    contacts: dict[int, list[str]] = {}
    for item in index["contact_diagnostics"]:
        if item["raw_contact_candidate"] and item["observation_state"] == "observed":
            contacts.setdefault(item["analysis_frame_index"], []).append(
                f"{item['hand_source_id']}→{item['part_id']}"
            )
    capture = cv2.VideoCapture(str(VIDEO))
    try:
        for name, frames in SEGMENTATION_GROUPS.items():
            cells: list[np.ndarray] = []
            for frame in frames:
                source = _frame(capture, frame)
                primary = _mask_overlay(source, rows[frame])
                diagnostic = primary.copy()
                candidates = ", ".join(contacts.get(frame, ())) or "none"
                cv2.rectangle(diagnostic, (0, 20), (320, 42), (0, 0, 0), -1)
                cv2.putText(
                    diagnostic,
                    f"raw contact: {candidates[:43]}",
                    (4, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 255, 255),
                    1,
                )
                cells.append(
                    cv2.hconcat(
                        [
                            _label(primary, f"f{frame} corrected SAM3 primary"),
                            _label(diagnostic, f"f{frame} + contact candidates"),
                        ]
                    )
                )
            cv2.imwrite(str(OUTPUT / f"{name}.png"), _grid(cells))
    finally:
        capture.release()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    render_wilor_audits()
    render_segmentation_audits()


if __name__ == "__main__":
    main()
