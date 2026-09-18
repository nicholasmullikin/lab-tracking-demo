"""OpenCV-operated, image-only decoder worker for ``battle.muggled_calibration``.

This deliberately has no Battle/Pydantic imports: it runs in MuggledSAM's separately
managed interpreter, and its output is validated by the Battle parent process.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n")
    temporary_path.replace(path)


def _read_frame(capture: Any, frame_index: int) -> Any:
    capture.set(1, frame_index)  # cv2.CAP_PROP_POS_FRAMES; kept numeric for minimal imports.
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"could not decode requested proxy frame {frame_index}")
    return frame


def _pixel_box(roi: Any, width: int, height: int) -> dict[str, int] | None:
    x, y, roi_width, roi_height = (int(value) for value in roi)
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(width, x + roi_width), min(height, y + roi_height)
    if x2 <= x1 or y2 <= y1:
        return None
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _select_single_box(
    *,
    frame: Any,
    title: str,
    selector: Callable[..., Any],
    close_window: Callable[[str], None],
) -> dict[str, int] | None:
    """Collect exactly one ROI, using injected HighGUI operations for testability.

    ``cv2.selectROI`` returns when Enter or Space accepts one rectangle, while ``c``
    returns an empty ROI. Calling it once per rectangle avoids the different finish
    control used by OpenCV's multi-ROI dialog.
    """
    try:
        roi = selector(title, frame, showCrosshair=True, fromCenter=False)
    finally:
        close_window(title)
    return _pixel_box(roi, frame.shape[1], frame.shape[0])


def _collection_action(value: str, *, has_boxes: bool) -> str | None:
    """Parse the explicit terminal action between single-ROI GUI invocations."""
    actions = {
        "a": "add",
        "add": "add",
        "r": "retry",
        "retry": "retry",
        "s": "skip",
        "skip": "skip",
    }
    if has_boxes:
        actions.update({"d": "decode", "decode": "decode"})
    return actions.get(value.strip().lower())


def _candidate_action(value: str) -> str | None:
    """Parse the terminal decision after a decoder review window closes."""
    return {
        "a": "accept",
        "accept": "accept",
        "r": "retry",
        "retry": "retry",
        "s": "skip",
        "skip": "skip",
    }.get(value.strip().lower())


def _timestamp_action(value: str) -> str | None:
    """Parse the initial terminal decision for a timestamp."""
    return {
        "": "draw",
        "s": "skip",
        "skip": "skip",
        "q": "finish",
        "quit": "finish",
    }.get(value.strip().lower())


def _normalized_box(box: dict[str, int], width: int, height: int) -> dict[str, float]:
    return {
        "x": box["x1"] / width,
        "y": box["y1"] / height,
        "width": (box["x2"] - box["x1"]) / width,
        "height": (box["y2"] - box["y1"]) / height,
    }


def _muggledsam_prompt_arguments(
    *,
    prompt_box: dict[str, int],
    width: int,
    height: int,
    boxes: list[list[list[float]]] | None = None,
    fg_points: list[list[float]] | None = None,
    bg_points: list[list[float]] | None = None,
) -> tuple[list[list[list[float]]], list[list[float]], list[list[float]]]:
    """Return the exact normalized list shapes accepted by MuggledSAM's image API."""
    if boxes is None:
        normalized_box = _normalized_box(prompt_box, width, height)
        boxes = [
            [
                [normalized_box["x"], normalized_box["y"]],
                [
                    normalized_box["x"] + normalized_box["width"],
                    normalized_box["y"] + normalized_box["height"],
                ],
            ]
        ]
    fg_points = [] if fg_points is None else fg_points
    bg_points = [] if bg_points is None else bg_points
    if (
        len(boxes) != 1
        or any(len(box) != 2 or any(len(point) != 2 for point in box) for box in boxes)
        or any(len(point) != 2 for point in fg_points)
        or any(len(point) != 2 for point in bg_points)
    ):
        raise ValueError("batch_decode requires boxes as Nx2x2 and fg_points/bg_points as Nx2")
    values = [
        coordinate
        for collection in (boxes, fg_points, bg_points)
        for item in collection
        for point in (item if collection is boxes else [item])
        for coordinate in point
    ]
    if any(not isinstance(value, (int, float)) or not 0 <= value <= 1 for value in values):
        raise ValueError("MuggledSAM prompt coordinates must be normalized numbers in [0, 1]")
    return boxes, fg_points, bg_points


def _encode_muggledsam_prompts(
    interactive_model: Any,
    *,
    boxes: list[list[list[float]]],
    fg_points: list[list[float]],
    bg_points: list[list[float]],
) -> Any:
    """Pass box, positive, and negative inputs unchanged to MuggledSAM."""
    return interactive_model.encode_prompts(boxes, fg_points, bg_points)


def _predicted_box(mask: Any) -> dict[str, float] | None:
    import numpy as np

    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    height, width = mask.shape
    return {
        "x": float(xs.min() / width),
        "y": float(ys.min() / height),
        "width": float((xs.max() + 1 - xs.min()) / width),
        "height": float((ys.max() + 1 - ys.min()) / height),
    }


def _decoder_pixel_center_to_source(
    decoder_x: float,
    decoder_y: float,
    *,
    decoder_width: int,
    decoder_height: int,
    source_width: int,
    source_height: int,
) -> tuple[float, float]:
    """Map a decoder-grid pixel center to source-pixel coordinates.

    MuggledSAM's square decoder output covers the complete square-preprocessed image.
    The image encoder's square sizing stretches the complete source extent into that
    square, so the inverse map independently scales x and y back to the original
    source dimensions.  The half-pixel terms match ``align_corners=False`` resizing.
    """
    return (
        (decoder_x + 0.5) * source_width / decoder_width - 0.5,
        (decoder_y + 0.5) * source_height / decoder_height - 0.5,
    )


def _source_mask_from_decoder_logits(mask_logits: Any, source_shape: tuple[int, int]) -> Any:
    """Resample one decoder logit plane onto the complete source-frame extent."""
    import torch.nn.functional as functional

    source_height, source_width = source_shape
    return functional.interpolate(
        mask_logits,
        size=(source_height, source_width),
        mode="bilinear",
        align_corners=False,
    )


def _mask_diagnostics(mask: Any, prompt_box: dict[str, int]) -> tuple[int, float, int, float]:
    """Return mask area and the explicitly defined portion contained by the prompt."""
    import numpy as np

    mask_pixels = int(np.count_nonzero(mask))
    height, width = mask.shape
    prompt_mask = mask[prompt_box["y1"] : prompt_box["y2"], prompt_box["x1"] : prompt_box["x2"]]
    pixels_in_prompt = int(np.count_nonzero(prompt_mask))
    return (
        mask_pixels,
        mask_pixels / (height * width),
        pixels_in_prompt,
        pixels_in_prompt / mask_pixels if mask_pixels else 0.0,
    )


def _source_box_from_mask(mask: Any) -> dict[str, int] | None:
    """Return source-pixel mask bounds with an exclusive lower-right corner."""
    import numpy as np

    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return {
        "x1": int(xs.min()),
        "y1": int(ys.min()),
        "x2": int(xs.max() + 1),
        "y2": int(ys.max() + 1),
    }


def _review_crop_bounds(
    *,
    prompt_box: dict[str, int],
    mask_box: dict[str, int] | None,
    width: int,
    height: int,
    padding_fraction: float = 0.20,
    minimum_padding_pixels: int = 24,
) -> dict[str, int]:
    """Center an aspect-preserving review crop on the prompt/mask union."""
    boxes = (prompt_box,) if mask_box is None else (prompt_box, mask_box)
    x1 = min(box["x1"] for box in boxes)
    y1 = min(box["y1"] for box in boxes)
    x2 = max(box["x2"] for box in boxes)
    y2 = max(box["y2"] for box in boxes)
    padding = max(minimum_padding_pixels, round(max(x2 - x1, y2 - y1) * padding_fraction))
    return {
        "x1": max(0, x1 - padding),
        "y1": max(0, y1 - padding),
        "x2": min(width, x2 + padding),
        "y2": min(height, y2 + padding),
    }


def _draw_box(image: Any, box: dict[str, int], color: tuple[int, int, int], label: str) -> None:
    import cv2

    cv2.rectangle(image, (box["x1"], box["y1"]), (box["x2"], box["y2"]), color, 2)
    cv2.putText(
        image,
        label,
        (box["x1"], max(22, box["y1"] - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def _draw_normalized_box(
    image: Any, box: dict[str, float] | None, color: tuple[int, int, int]
) -> None:
    import cv2

    if box is None:
        return
    height, width = image.shape[:2]
    x1, y1 = round(box["x"] * width), round(box["y"] * height)
    x2 = round((box["x"] + box["width"]) * width)
    y2 = round((box["y"] + box["height"]) * height)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)


def _aspect_fit_geometry(
    *,
    width: int,
    height: int,
    target_width: int,
    target_height: int,
) -> tuple[float, int, int, int, int]:
    """Calculate a letterboxed, aspect-preserving display placement."""
    scale = min(target_width / width, target_height / height)
    resized_width, resized_height = round(width * scale), round(height * scale)
    return (
        scale,
        resized_width,
        resized_height,
        (target_width - resized_width) // 2,
        (target_height - resized_height) // 2,
    )


def _fit_aspect(image: Any, target_width: int, target_height: int) -> tuple[Any, float, int, int]:
    """Fit an image within a fixed display area without changing its aspect ratio."""
    import cv2
    import numpy as np

    height, width = image.shape[:2]
    scale, resized_width, resized_height, offset_x, offset_y = _aspect_fit_geometry(
        width=width,
        height=height,
        target_width=target_width,
        target_height=target_height,
    )
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    canvas = np.full((target_height, target_width, 3), (24, 24, 24), dtype=np.uint8)
    canvas[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = resized
    return canvas, scale, offset_x, offset_y


def _render_review_view(
    *,
    frame: Any,
    mask: Any,
    prompt_box: dict[str, int],
    crop_box: dict[str, int],
    target_width: int,
    target_height: int,
) -> Any:
    """Render a source or ROI view with nearest-mask fill and a crisp source contour."""
    import cv2
    import numpy as np

    x1, y1, x2, y2 = (crop_box[key] for key in ("x1", "y1", "x2", "y2"))
    source = frame[y1:y2, x1:x2]
    source_mask = mask[y1:y2, x1:x2]
    highlighted = source.copy()
    mask_pixels = source_mask > 0
    mask_color = np.array((255, 220, 0), dtype=np.uint8)  # Cyan over monochrome video.
    highlighted[mask_pixels] = (
        highlighted[mask_pixels].astype(np.float32) * 0.62 + mask_color * 0.38
    ).astype(np.uint8)
    view, scale, offset_x, offset_y = _fit_aspect(highlighted, target_width, target_height)
    display_mask = cv2.resize(
        source_mask,
        (round(source.shape[1] * scale), round(source.shape[0] * scale)),
        interpolation=cv2.INTER_NEAREST,
    )
    contours, _ = cv2.findContours(display_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_offset = (offset_x, offset_y)
    cv2.drawContours(view, contours, -1, (255, 0, 255), 2, offset=contour_offset)

    display_box = {
        "x1": round((prompt_box["x1"] - x1) * scale) + offset_x,
        "y1": round((prompt_box["y1"] - y1) * scale) + offset_y,
        "x2": round((prompt_box["x2"] - x1) * scale) + offset_x,
        "y2": round((prompt_box["y2"] - y1) * scale) + offset_y,
    }
    _draw_box(view, display_box, (0, 190, 255), "prompt")
    return view


def _candidate_review_panel(
    *,
    frame: Any,
    mask: Any,
    prompt_box: dict[str, int],
    score: float,
    index: int,
    best_index: int,
    target_label: str,
    human_selected_candidate_index: int | None = None,
    human_accepted: bool = False,
) -> Any:
    """Build one side-by-side context/ROI panel for a decoder candidate."""
    import cv2
    import numpy as np

    human_selection = _human_selection_caption(human_selected_candidate_index, human_accepted)
    panel_width, panel_height = 940, 424
    view_width, view_height = 446, 300
    panel = np.full((panel_height, panel_width, 3), (30, 30, 30), dtype=np.uint8)
    mask_pixels, mask_fraction, in_prompt, in_prompt_fraction = _mask_diagnostics(mask, prompt_box)
    status = "DECODER TIEBREAK BEST" if index == best_index else "DECODER ALTERNATIVE"
    cv2.putText(
        panel,
        f"Candidate {index} ({target_label})  |  {status}  |  {human_selection}",
        (14, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (80, 255, 255) if index == best_index else (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        (
            f"Decoder IoU estimate (not accuracy): {score:.4f}  |  "
            f"mask: {mask_pixels:,} px ({mask_fraction:.2%})  |  "
            f"in prompt: {in_prompt:,} px / mask ({in_prompt_fraction:.2%})"
        ),
        (14, 51),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (225, 225, 225),
        1,
        cv2.LINE_AA,
    )
    full_crop = {"x1": 0, "y1": 0, "x2": frame.shape[1], "y2": frame.shape[0]}
    roi_crop = _review_crop_bounds(
        prompt_box=prompt_box,
        mask_box=_source_box_from_mask(mask),
        width=frame.shape[1],
        height=frame.shape[0],
    )
    panel[76 : 76 + view_height, 14 : 14 + view_width] = _render_review_view(
        frame=frame,
        mask=mask,
        prompt_box=prompt_box,
        crop_box=full_crop,
        target_width=view_width,
        target_height=view_height,
    )
    panel[76 : 76 + view_height, 480 : 480 + view_width] = _render_review_view(
        frame=frame,
        mask=mask,
        prompt_box=prompt_box,
        crop_box=roi_crop,
        target_width=view_width,
        target_height=view_height,
    )
    cv2.putText(
        panel,
        "CONTEXT: source frame, cyan mask, magenta contour, amber prompt",
        (14, 400),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (225, 225, 225),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "ROI: prompt/mask union + 20% (min 24 px), aspect preserved",
        (480, 400),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (225, 225, 225),
        1,
        cv2.LINE_AA,
    )
    return panel


def _human_selection_caption(
    human_selected_candidate_index: int | None, human_accepted: bool
) -> str:
    """Describe only the explicit human-selection state stored in the manifest."""
    if human_accepted != (human_selected_candidate_index is not None):
        raise ValueError("human acceptance requires exactly one selected decoder candidate")
    if human_selected_candidate_index is None:
        return "Human selection: pending"
    return f"Human selection: accepted candidate {human_selected_candidate_index}"


def _candidate_grid_geometry(
    candidate_count: int, *, panel_width: int = 940, panel_height: int = 424, gutter: int = 10
) -> tuple[int, int, int, int]:
    """Return the two-column contact-sheet geometry for decoder candidates."""
    columns = min(2, candidate_count)
    rows = (candidate_count + 1) // 2
    return (
        rows,
        columns,
        columns * panel_width + (columns - 1) * gutter,
        rows * panel_height + (rows - 1) * gutter,
    )


def _make_overlay(
    *,
    frame: Any,
    prompt_box: dict[str, int],
    masks: list[Any],
    scores: list[float],
    predicted_boxes: list[dict[str, float] | None],
    best_index: int,
    target_label: str,
) -> Any:
    import numpy as np

    del predicted_boxes  # Bounds are derived from the source mask to make spill visible.
    panels = [
        _candidate_review_panel(
            frame=frame,
            mask=mask,
            prompt_box=prompt_box,
            score=scores[index],
            index=index,
            best_index=best_index,
            target_label=target_label,
        )
        for index, mask in enumerate(masks)
    ]
    rows, _columns, sheet_width, sheet_height = _candidate_grid_geometry(len(panels))
    sheet = np.full((sheet_height, sheet_width, 3), (16, 16, 16), dtype=np.uint8)
    for index, panel in enumerate(panels):
        row, column = divmod(index, 2)
        y, x = row * 434, column * 950
        sheet[y : y + panel.shape[0], x : x + panel.shape[1]] = panel
    return sheet


def _sha256_file(path: Path) -> str:
    """Return a content hash without loading a potentially large artifact at once."""
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finalized_seed_review_records(
    manifest: dict[str, Any], proposal: dict[str, Any]
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], list[dict[str, Any]]]:
    """Verify and return proposal-backed selections plus preserved exclusions."""
    candidates = {str(candidate["candidate_id"]): candidate for candidate in manifest["candidates"]}
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    selected_ids: set[str] = set()
    for seed in proposal["seeds"]:
        candidate_id = str(seed["candidate_id"])
        if candidate_id in selected_ids:
            raise ValueError(f"proposal contains duplicate candidate ID: {candidate_id}")
        selected_ids.add(candidate_id)
        candidate = candidates.get(candidate_id)
        if candidate is None:
            raise ValueError(f"proposal references absent calibration candidate: {candidate_id}")
        selected_index = candidate.get("human_selected_candidate_index")
        if (
            candidate.get("human_accepted") is not True
            or candidate.get("selected_for_finalization") is not True
            or candidate["frame"]["analysis_frame_index"] != 0
            or not isinstance(selected_index, int)
            or selected_index != seed.get("human_selected_candidate_index")
        ):
            raise ValueError(
                "proposal seed must match a human-accepted, finalization-eligible "
                f"frame-0 calibration selection: {candidate_id}"
            )
        if candidate["intended_target"] != seed.get("intended_target"):
            raise ValueError(
                f"proposal target does not match calibration candidate: {candidate_id}"
            )
        selected.append((candidate, seed))
    if not selected:
        raise ValueError("proposal must contain at least one selected seed")
    excluded = [
        candidate
        for candidate in manifest["candidates"]
        if candidate.get("human_accepted") is True
        and candidate.get("human_selected_candidate_index") is not None
        and candidate["candidate_id"] not in selected_ids
    ]
    return selected, excluded


def render_final_selected_seed_review(
    *, manifest_path: Path, proposal_path: Path, output_path: Path
) -> Path:
    """Render a provenance-checked sheet from finalized proposal selections only."""
    import cv2
    import numpy as np

    manifest = json.loads(manifest_path.read_text())
    proposal = json.loads(proposal_path.read_text())
    if proposal.get("calibration_manifest_sha256") != _sha256_file(manifest_path):
        raise ValueError("proposal does not match the current calibration manifest fingerprint")
    selected, excluded = _finalized_seed_review_records(manifest, proposal)
    panel_width, panel_height = 940, 424
    header_height, card_header_height, gutter = 102, 28, 12
    footer_height = max(48, 24 * len(excluded) + 24)
    sheet_height = (
        header_height + len(selected) * (card_header_height + panel_height + gutter) + footer_height
    )
    sheet = np.full((sheet_height, panel_width, 3), (16, 16, 16), dtype=np.uint8)
    cv2.putText(
        sheet,
        f"FINALIZED E4 SEED REVIEW — {len(selected)} selected frame-0 masks",
        (14, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        (80, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        sheet,
        "Each displayed mask is an explicit human-accepted selection; decoder IoU is not accuracy.",
        (14, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (225, 225, 225),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        sheet,
        "Only proposal seeds marked eligible in the manifest are displayed.",
        (14, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (225, 225, 225),
        1,
        cv2.LINE_AA,
    )
    results_directory = manifest_path.parent / "results"
    for position, (candidate, seed) in enumerate(selected):
        selected_index = seed["human_selected_candidate_index"]
        decoder_candidate = next(
            item
            for item in candidate["decoder_result"]["candidates"]
            if item["candidate_index"] == selected_index
        )
        frame_index = candidate["frame"]["analysis_frame_index"]
        frame_path = results_directory / "frames" / f"frame-{frame_index:06d}.jpg"
        mask_path = manifest_path.parent / decoder_candidate["mask_uri"]
        frame = cv2.imread(str(frame_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            raise RuntimeError(f"missing saved source preview for final review: {frame_path}")
        if mask is None:
            raise RuntimeError(f"missing saved selected mask for final review: {mask_path}")
        panel = _candidate_review_panel(
            frame=frame,
            mask=mask,
            prompt_box=candidate["pixel_box"],
            score=float(decoder_candidate["iou_score"]),
            index=selected_index,
            best_index=candidate["decoder_result"]["deterministic_best_candidate_index"],
            target_label=candidate["intended_target"],
            human_selected_candidate_index=selected_index,
            human_accepted=True,
        )
        y = header_height + position * (card_header_height + panel_height + gutter)
        cv2.putText(
            sheet,
            (
                f"{candidate['candidate_id']}  |  {candidate['intended_target']}  |  "
                f"human-accepted candidate {selected_index}"
            ),
            (14, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (80, 255, 255),
            1,
            cv2.LINE_AA,
        )
        sheet[y + card_header_height : y + card_header_height + panel_height] = panel
    footer_y = header_height + len(selected) * (card_header_height + panel_height + gutter)
    if excluded:
        cv2.putText(
            sheet,
            "EXCLUDED (preserved human-accepted selections, not finalization-eligible):",
            (14, footer_y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (80, 255, 255),
            1,
            cv2.LINE_AA,
        )
        for position, candidate in enumerate(excluded, start=1):
            cv2.putText(
                sheet,
                (
                    f"{candidate['candidate_id']}  |  {candidate['intended_target']}  |  "
                    f"human-accepted candidate {candidate['human_selected_candidate_index']}  |  "
                    "not selected for finalization"
                ),
                (14, footer_y + 18 + position * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (225, 225, 225),
                1,
                cv2.LINE_AA,
            )
    else:
        cv2.putText(
            sheet,
            "No additional human-accepted selections were excluded from finalization.",
            (14, footer_y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (225, 225, 225),
            1,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), sheet):
        raise RuntimeError(f"could not write finalized seed review: {output_path}")
    return output_path


def _display_review(title: str, overlay: Any, saved_path: Path) -> None:
    import cv2

    height, width = overlay.shape[:2]
    scale = min(1.0, 1900 / width, 1000 / height)
    display = (
        overlay
        if scale == 1.0
        else cv2.resize(
            overlay,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    )
    cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(title, display.shape[1], display.shape[0])
    cv2.imshow(title, display)
    print(
        f"Review contact sheet was saved before this decision: {saved_path}. "
        "It remains available if you skip or redraw this box. "
        "Press any key in the window to continue."
    )
    cv2.waitKey(0)
    cv2.destroyWindow(title)


def _decode_rectangle(
    *,
    interactive_model: Any,
    encoded_image: Any,
    frame: Any,
    prompt_box: dict[str, int],
    target_label: str,
    candidate_id: str,
    results_directory: Path,
    boxes: list[list[list[float]]] | None = None,
    fg_points: list[list[float]] | None = None,
    bg_points: list[list[float]] | None = None,
    show_review: bool = True,
) -> dict[str, Any]:
    import cv2

    height, width = frame.shape[:2]
    boxes, fg_points, bg_points = _muggledsam_prompt_arguments(
        prompt_box=prompt_box,
        width=width,
        height=height,
        boxes=boxes,
        fg_points=fg_points,
        bg_points=bg_points,
    )
    prompts = _encode_muggledsam_prompts(
        interactive_model,
        boxes=boxes,
        fg_points=fg_points,
        bg_points=bg_points,
    )
    mask_logits, iou_predictions = interactive_model.generate_masks(encoded_image, prompts)
    scores = [float(value) for value in iou_predictions[0].detach().float().cpu().tolist()]
    best_index = max(range(len(scores)), key=lambda index: (scores[index], -index))
    candidate_records = []
    full_resolution_masks = []
    predicted_boxes = []
    for index in range(len(scores)):
        full_resolution_logits = _source_mask_from_decoder_logits(
            mask_logits[:, [index]],
            (height, width),
        )
        mask = ((full_resolution_logits > 0).byte() * 255).cpu().numpy().squeeze()
        mask_path = results_directory / "masks" / f"{candidate_id}_candidate-{index:02d}.png"
        if not cv2.imwrite(str(mask_path), mask):
            raise RuntimeError(f"could not write calibration mask: {mask_path}")
        predicted_box = _predicted_box(mask)
        review_path = results_directory / f"{candidate_id}_candidate-{index:02d}_review.png"
        review = _candidate_review_panel(
            frame=frame,
            mask=mask,
            prompt_box=prompt_box,
            score=scores[index],
            index=index,
            best_index=best_index,
            target_label=target_label,
        )
        if not cv2.imwrite(str(review_path), review):
            raise RuntimeError(f"could not write candidate review: {review_path}")
        full_resolution_masks.append(mask)
        predicted_boxes.append(predicted_box)
        candidate_records.append(
            {
                "candidate_index": index,
                "iou_score": scores[index],
                "predicted_box": predicted_box,
                "mask_uri": f"results/masks/{mask_path.name}",
                "review_uri": f"results/{review_path.name}",
                "is_deterministic_best": index == best_index,
            }
        )
    overlay = _make_overlay(
        frame=frame,
        prompt_box=prompt_box,
        masks=full_resolution_masks,
        scores=scores,
        predicted_boxes=predicted_boxes,
        best_index=best_index,
        target_label=target_label,
    )
    overlay_path = results_directory / f"{candidate_id}_overlay.png"
    if not cv2.imwrite(str(overlay_path), overlay):
        raise RuntimeError(f"could not write calibration overlay: {overlay_path}")
    if show_review:
        _display_review(f"SAM3 image-only review: {candidate_id}", overlay, overlay_path)
    return {
        "api": "muggledsam_sam3_interactive",
        "candidate_count": len(candidate_records),
        "deterministic_best_candidate_index": best_index,
        "candidates": candidate_records,
        "overlay_uri": f"results/{overlay_path.name}",
        "stability_score_available": False,
        "limitations": [
            "MuggledSAM interactive generate_masks exposes IoU estimates but no stability score.",
            "The decoder emitted all returned mask candidates; deterministic best is maximum "
            "IoU estimate with lowest candidate index used to break ties.",
            "Image-decoder IoU estimates are model outputs, not ground-truth accuracy.",
        ],
    }


def _next_candidate_id(candidates: list[dict[str, Any]], frame_index: int) -> str:
    prefix = f"t{frame_index:06d}-b"
    indices = [
        int(candidate["candidate_id"].removeprefix(prefix))
        for candidate in candidates
        if candidate["candidate_id"].startswith(prefix)
    ]
    return f"{prefix}{max(indices, default=0) + 1:02d}"


def _draw_one_box(frame: Any, timestamp_seconds: float) -> dict[str, int] | None:
    """Open one ROI dialog and return after one accept or cancellation."""
    import cv2

    print(
        "\nROI window: drag exactly one rectangle, then press SPACE or ENTER to accept it. "
        "Press c to cancel that rectangle."
    )
    return _select_single_box(
        frame=frame,
        title=f"Draw one box @ {timestamp_seconds:.3f}s",
        selector=cv2.selectROI,
        close_window=cv2.destroyWindow,
    )


def _collect_boxes(frame: Any, timestamp_seconds: float) -> list[dict[str, int]] | None:
    """Collect boxes with terminal decisions between one-rectangle GUI windows.

    Returns ``None`` when the user skips the timestamp and otherwise returns the
    current, non-empty batch to decode.  No OpenCV window remains open while stdin is
    read, avoiding competing GUI and terminal controls.
    """
    boxes: list[dict[str, int]] = []
    while True:
        box = _draw_one_box(frame, timestamp_seconds)
        if box is None:
            print("No rectangle was added.")
        else:
            boxes.append(box)
            print(f"Added box {len(boxes)}: {box}.")

        while True:
            if boxes:
                prompt = (
                    "Next action: [a]dd another box, [d]ecode current boxes, "
                    "[r]etry (clear current boxes), or [s]kip timestamp: "
                )
            else:
                prompt = "Next action: [r]etry selection or [s]kip timestamp: "
            action = _collection_action(input(prompt), has_boxes=bool(boxes))
            if action is None:
                print("Enter one of the displayed actions.")
                continue
            if action == "add":
                break
            if action == "decode":
                return boxes
            if action == "retry":
                boxes.clear()
                break
            return None


def _prompt_box_after_review(
    *,
    frame: Any,
    timestamp_seconds: float,
    box: dict[str, int],
    interactive_model: Any,
    encoded_image: Any,
    frame_index: int,
    source_offset_seconds: float,
    manifest: dict[str, Any],
    results_directory: Path,
    manifest_path: Path,
) -> None:
    """Label, decode, review, and optionally persist one selected prompt box."""
    while True:
        target_label = input("Target label (hand / yellow toy body / toy wheel / custom): ").strip()
        while not target_label:
            target_label = input("Target label cannot be empty; enter a label: ").strip()
        candidate_id = _next_candidate_id(manifest["candidates"], frame_index)
        decoder_result = _decode_rectangle(
            interactive_model=interactive_model,
            encoded_image=encoded_image,
            frame=frame,
            prompt_box=box,
            target_label=target_label,
            candidate_id=candidate_id,
            results_directory=results_directory,
        )
        while True:
            action = _candidate_action(
                input(
                    "Contact sheet is saved regardless. After review: [a]ccept, "
                    "[r]etry this box, or [s]kip this box: "
                )
            )
            if action is not None:
                break
            print("Enter a, r, or s.")
        if action == "skip":
            return
        if action == "retry":
            replacement = _draw_one_box(frame, timestamp_seconds)
            if replacement is None:
                print("Retry cancelled; skipped this box.")
                return
            box = replacement
            continue
        candidate_indices = [
            candidate["candidate_index"] for candidate in decoder_result["candidates"]
        ]
        while True:
            selected_index_text = input(
                f"Human-selected mask candidate ({', '.join(map(str, candidate_indices))}): "
            ).strip()
            try:
                human_selected_candidate_index = int(selected_index_text)
            except ValueError:
                human_selected_candidate_index = -1
            if human_selected_candidate_index in candidate_indices:
                break
            print("Enter one of the displayed candidate indices.")
        selected_for_finalization = False
        if frame_index == 0:
            selected_for_finalization = (
                input(
                    "Explicitly mark this human-selected frame-0 mask eligible for "
                    "--finalize? [y/N]: "
                )
                .strip()
                .lower()
                == "y"
            )
        else:
            print("Later-frame masks are decoder checks only and cannot be finalized.")
        manifest["candidates"].append(
            {
                "candidate_id": candidate_id,
                "intended_target": target_label,
                "frame": {
                    "analysis_frame_index": frame_index,
                    "proxy_seconds": frame_index / manifest["proxy_fps"],
                    "analysis_seconds": frame_index / manifest["proxy_fps"],
                    "source_seconds": source_offset_seconds + frame_index / manifest["proxy_fps"],
                },
                "pixel_box": box,
                "normalized_box": _normalized_box(box, frame.shape[1], frame.shape[0]),
                "decoder_result": decoder_result,
                "human_selected_candidate_index": human_selected_candidate_index,
                "human_accepted": True,
                "selected_for_finalization": selected_for_finalization,
            }
        )
        _write_json(manifest_path, manifest)
        print(f"Accepted {candidate_id} and updated {manifest_path}.")
        return


def _select_for_timestamp(
    *,
    capture: Any,
    manifest: dict[str, Any],
    timestamp: float,
    interactive_model: Any,
    results_directory: Path,
    manifest_path: Path,
) -> bool:
    fps = manifest["proxy_fps"]
    frame_index = round(timestamp * fps)
    if frame_index >= manifest["proxy_frame_count"]:
        raise ValueError(f"timestamp {timestamp} is outside this proxy")
    frame = _read_frame(capture, frame_index)
    source_offset_seconds = manifest["source_offset_seconds"]
    print(
        f"\nTimestamp request {timestamp:.3f}s -> proxy frame {frame_index} "
        f"({frame_index / fps:.3f}s, source {source_offset_seconds + frame_index / fps:.3f}s)."
    )
    while True:
        action = _timestamp_action(
            input("Press Enter to draw boxes, 's' to skip, or 'q' to finish this session: ")
        )
        if action is not None:
            break
        print("Press Enter, s, or q.")
    if action == "finish":
        return False
    if action == "skip":
        return True
    boxes = _collect_boxes(frame, frame_index / fps)
    if boxes is None:
        return True
    encoded_image = interactive_model.encode_image(frame, None, True)
    for box in boxes:
        _prompt_box_after_review(
            frame=frame,
            timestamp_seconds=frame_index / fps,
            box=box,
            interactive_model=interactive_model,
            encoded_image=encoded_image,
            frame_index=frame_index,
            source_offset_seconds=source_offset_seconds,
            manifest=manifest,
            results_directory=results_directory,
            manifest_path=manifest_path,
        )
    return True


def _review_saved(manifest: dict[str, Any], manifest_path: Path) -> None:
    import cv2

    while True:
        saved = manifest["candidates"]
        print("\nAccepted candidates:")
        for candidate in saved:
            print(
                f"  {candidate['candidate_id']}: {candidate['intended_target']} "
                f"@ frame {candidate['frame']['analysis_frame_index']} "
                f"(finalize={candidate['selected_for_finalization']})"
            )
        command = input(
            "Saved review: 'review ID' (open sheet), 'clear ID', 'toggle ID', or Enter to finish: "
        ).strip()
        if not command:
            return
        action, *values = command.split(maxsplit=1)
        candidate_id = values[0] if values else ""
        candidate = next(
            (item for item in manifest["candidates"] if item["candidate_id"] == candidate_id), None
        )
        if candidate is None:
            print("Enter a listed candidate ID.")
            continue
        if action == "review":
            overlay_path = manifest_path.parent / candidate["decoder_result"]["overlay_uri"]
            overlay = cv2.imread(str(overlay_path))
            if overlay is None:
                print(f"Missing overlay: {overlay_path}")
            else:
                _display_review(f"Saved review: {candidate_id}", overlay, overlay_path)
        elif action == "clear":
            manifest["candidates"].remove(candidate)
            _write_json(manifest_path, manifest)
            print(f"Cleared {candidate_id}; result files remain for audit.")
        elif action == "toggle":
            candidate["selected_for_finalization"] = not candidate["selected_for_finalization"]
            _write_json(manifest_path, manifest)
            print(f"Finalize eligibility is now {candidate['selected_for_finalization']}.")
        else:
            print("Allowed actions: review, clear, toggle.")


def run(args: argparse.Namespace) -> int:
    import cv2
    import torch
    from muggled_sam.make_sam import make_sam_from_state_dict

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"SAM3 checkpoint is unavailable: {model_path}")
    if not torch.cuda.is_available() and "cuda" in args.device:
        raise RuntimeError("CUDA is unavailable for the configured image decoder")
    capture = cv2.VideoCapture(args.proxy)
    if not capture.isOpened():
        raise RuntimeError(f"could not open approved e4 proxy: {args.proxy}")
    try:
        print(f"Loading MuggledSAM image decoder on {args.device}...")
        core = make_sam_from_state_dict(model_path)
        interactive_model = core.get_interactive_context()
        dtype = torch.bfloat16 if "cuda" in args.device else torch.float32
        interactive_model.to(device=args.device, dtype=dtype)
        results_directory = manifest_path.parent / "results"
        timestamps = tuple(json.loads(args.timestamps_json))
        for timestamp in timestamps:
            keep_calibrating = _select_for_timestamp(
                capture=capture,
                manifest=manifest,
                timestamp=float(timestamp),
                interactive_model=interactive_model,
                results_directory=results_directory,
                manifest_path=manifest_path,
            )
            if not keep_calibrating:
                break
        _review_saved(manifest, manifest_path)
        return 0
    finally:
        capture.release()
        cv2.destroyAllWindows()


class _ProtocolRuntime:
    """Long-lived isolated decoder used by Battle's localhost workspace."""

    def __init__(self, args: argparse.Namespace) -> None:
        import cv2
        import torch
        from muggled_sam.make_sam import make_sam_from_state_dict

        self.cv2 = cv2
        if not Path(args.model).is_file():
            raise FileNotFoundError(f"SAM3 checkpoint is unavailable: {args.model}")
        if not torch.cuda.is_available() and "cuda" in args.device:
            raise RuntimeError("CUDA is unavailable for the configured image decoder")
        self.capture = cv2.VideoCapture(args.proxy)
        if not self.capture.isOpened():
            raise RuntimeError(f"could not open approved e4 proxy: {args.proxy}")
        self.results_directory = Path(args.results_directory)
        (self.results_directory / "masks").mkdir(parents=True, exist_ok=True)
        (self.results_directory / "frames").mkdir(parents=True, exist_ok=True)
        core = make_sam_from_state_dict(Path(args.model))
        self.model = core.get_interactive_context()
        self.model.to(
            device=args.device, dtype=torch.bfloat16 if "cuda" in args.device else torch.float32
        )
        self.encoded_frames: dict[int, tuple[Any, Any]] = {}

    def close(self) -> None:
        self.capture.release()

    def _frame_and_encoding(self, frame_index: int) -> tuple[Any, Any]:
        cached = self.encoded_frames.get(frame_index)
        if cached is None:
            frame = _read_frame(self.capture, frame_index)
            cached = (frame, self.model.encode_image(frame, None, True))
            self.encoded_frames[frame_index] = cached
        return cached

    def handle(self, command: str, payload: dict[str, Any]) -> dict[str, Any]:
        if command == "health":
            return {"ready": True}
        if command == "frame_preview":
            frame_index = int(payload["frame_index"])
            frame = _read_frame(self.capture, frame_index)
            frame_path = self.results_directory / "frames" / f"frame-{frame_index:06d}.jpg"
            if not self.cv2.imwrite(str(frame_path), frame):
                raise RuntimeError(f"could not write source preview: {frame_path}")
            return {"frame_index": frame_index, "image_uri": f"results/frames/{frame_path.name}"}
        if command == "batch_decode":
            decoded: list[dict[str, Any]] = []
            for prompt in payload["prompts"]:
                frame_index = int(prompt["frame_index"])
                frame, encoded_image = self._frame_and_encoding(frame_index)
                decoded.append(
                    {
                        "box_id": prompt["box_id"],
                        "candidate_id": prompt["candidate_id"],
                        "decoder_result": _decode_rectangle(
                            interactive_model=self.model,
                            encoded_image=encoded_image,
                            frame=frame,
                            prompt_box=prompt["pixel_box"],
                            target_label=str(prompt["intended_target"]),
                            candidate_id=prompt["candidate_id"],
                            results_directory=self.results_directory,
                            boxes=prompt.get("boxes"),
                            fg_points=prompt.get("fg_points"),
                            bg_points=prompt.get("bg_points"),
                            show_review=False,
                        ),
                    }
                )
            return {"decoded": decoded}
        if command == "render_final_selected_seed_review":
            output_path = render_final_selected_seed_review(
                manifest_path=Path(str(payload["manifest_path"])),
                proposal_path=Path(str(payload["proposal_path"])),
                output_path=Path(str(payload["output_path"])),
            )
            return {"output_path": str(output_path)}
        raise ValueError(f"unknown protocol command: {command}")


def serve_jsonl(args: argparse.Namespace) -> int:
    """Serve newline-delimited requests, emitting only correlated JSON on stdout."""
    runtime: _ProtocolRuntime | None = None
    try:
        # Third-party initialization must not corrupt the JSON stdout channel.
        with contextlib.redirect_stdout(sys.stderr):
            runtime = _ProtocolRuntime(args)
        for line in sys.stdin:
            request_id = ""
            try:
                request = json.loads(line)
                request_id = str(request["request_id"])
                command = str(request["command"])
                if command == "shutdown":
                    response = {"request_id": request_id, "ok": True, "result": {"stopped": True}}
                    print(json.dumps(response), flush=True)
                    break
                result = runtime.handle(command, dict(request.get("payload", {})))
                response = {"request_id": request_id, "ok": True, "result": result}
            except (KeyError, TypeError, ValueError, RuntimeError, OSError) as error:
                response = {"request_id": request_id, "ok": False, "error": str(error)}
            print(json.dumps(response), flush=True)
        return 0
    finally:
        if runtime is not None:
            runtime.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenCV UI for image-only MuggledSAM box calibration."
    )
    parser.add_argument("--manifest")
    parser.add_argument("--proxy")
    parser.add_argument("--model")
    parser.add_argument("--timestamps-json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--serve-jsonl", action="store_true")
    parser.add_argument("--results-directory", type=Path)
    parser.add_argument("--proposal", type=Path)
    parser.add_argument("--render-final-selected-seed-review", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.render_final_selected_seed_review:
            if args.manifest is None or args.proposal is None or args.output is None:
                parser.error(
                    "--render-final-selected-seed-review requires --manifest, --proposal, "
                    "and --output"
                )
            render_final_selected_seed_review(
                manifest_path=Path(args.manifest),
                proposal_path=args.proposal,
                output_path=args.output,
            )
            return
        if args.serve_jsonl:
            if args.results_directory is None:
                parser.error("--serve-jsonl requires --results-directory")
            if args.proxy is None or args.model is None:
                parser.error("--serve-jsonl requires --proxy and --model")
            raise SystemExit(serve_jsonl(args))
        if (
            args.manifest is None
            or args.timestamps_json is None
            or args.proxy is None
            or args.model is None
        ):
            parser.error(
                "--manifest, --proxy, --model, and --timestamps-json are required outside "
                "--serve-jsonl"
            )
        raise SystemExit(run(args))
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
