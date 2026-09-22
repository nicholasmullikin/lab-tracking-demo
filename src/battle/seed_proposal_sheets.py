"""Contact sheets for the seed-search accept/reject proposals, one PNG per view.

`battle-seed-search` leaves, per view, the three cells with the largest cross-strategy
disagreement (`runs/seed-search-20260920/proposals/<VIEW>/<part>_f<frame>/`), each with its
distinct candidate masks and a `proposal.json` whose `human_decision` is null.  The full-frame
overlays it writes are hard to judge because the parts are a few dozen pixels wide, so this
tool renders, per cell, the frame with every candidate outlined plus one zoomed crop per
candidate (filled), and lists strategy, area and the decoder's own IoU estimate.  It also writes
a decisions template the human fills in.  Nothing here changes a seed or a run; the candidates
are agent decoder outputs awaiting a human accept/reject.  CC BY-NC 4.0 applies to the frames.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import mask_cache
from .cli_common import add_output_root, add_repository_root
from .multiview_consensus import MANIFEST_NAME, load_consensus
from .multiview_consensus import OUTPUT_ROOT as CONSENSUS_ROOT
from .multiview_review import PROPOSAL_COLORS, PROPOSALS_ROOT, ProposalCell, load_proposals

OUTPUT_ROOT = Path("runs/labeling-sessions-20260920/proposal_sheets")
TEMPLATE_PATH = Path("configs/qa/seed_proposal_decisions.template.json")
CROP_SIZE = 320
CROP_MARGIN = 0.6
FULL_WIDTH = 640


def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for name in ("DejaVuSans.ttf", "DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _read_frame(video: Path, frame: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = capture.read()
    finally:
        capture.release()
    if not ok:
        raise RuntimeError(f"could not decode frame {frame} of {video}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _crop_box(masks: list[np.ndarray], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Square crop around the union of the candidate masks, grown by `CROP_MARGIN`."""
    height, width = shape
    union = np.zeros(shape, dtype=bool)
    for mask in masks:
        union |= mask
    ys, xs = np.where(union)
    if ys.size == 0:
        return 0, 0, width, height
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    side = max(xs.max() - xs.min(), ys.max() - ys.min(), 40) * (1 + 2 * CROP_MARGIN)
    side = min(side, min(width, height))
    x0 = int(np.clip(cx - side / 2, 0, width - side))
    y0 = int(np.clip(cy - side / 2, 0, height - side))
    return x0, y0, int(x0 + side), int(y0 + side)


def _draw_outline(draw: ImageDraw.ImageDraw, mask: np.ndarray, color, scale: float, offset=(0, 0)):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for contour in contours:
        points = [
            ((x - offset[0]) * scale, (y - offset[1]) * scale) for x, y in contour.reshape(-1, 2)
        ]
        if len(points) >= 2:
            draw.line([*points, points[0]], fill=color, width=2)


def _cell_row(cell: ProposalCell, frame_rgb: np.ndarray, record: dict) -> Image.Image:
    masks = [mask_cache.decode_mask_png(path) for _, path in cell.candidates]
    height, width = frame_rgb.shape[:2]
    full_scale = FULL_WIDTH / width
    full = Image.fromarray(frame_rgb).resize((FULL_WIDTH, int(height * full_scale)))
    draw = ImageDraw.Draw(full)
    for index, mask in enumerate(masks):
        _draw_outline(draw, mask, PROPOSAL_COLORS[index % len(PROPOSAL_COLORS)], full_scale)
    x0, y0, x1, y1 = _crop_box(masks, (height, width))
    draw.rectangle(
        (x0 * full_scale, y0 * full_scale, x1 * full_scale, y1 * full_scale),
        outline=(255, 255, 255),
        width=1,
    )
    crop_scale = CROP_SIZE / (x1 - x0)
    crops: list[Image.Image] = []
    font = _font(13)
    for index, (mask, candidate) in enumerate(zip(masks, record["candidates"], strict=True)):
        crop = Image.fromarray(frame_rgb[y0:y1, x0:x1]).resize((CROP_SIZE, CROP_SIZE))
        color = PROPOSAL_COLORS[index % len(PROPOSAL_COLORS)]
        fill = Image.new("RGBA", crop.size, (*color, 0))
        alpha = Image.fromarray(mask[y0:y1, x0:x1].astype(np.uint8) * 90).resize(
            (CROP_SIZE, CROP_SIZE), Image.NEAREST
        )
        fill.putalpha(alpha)
        crop = Image.alpha_composite(crop.convert("RGBA"), fill).convert("RGB")
        crop_draw = ImageDraw.Draw(crop)
        _draw_outline(crop_draw, mask, color, crop_scale, offset=(x0, y0))
        caption = Image.new("RGB", (CROP_SIZE, 44), (0, 0, 0))
        text = ImageDraw.Draw(caption)
        text.text(
            (4, 2), f"candidate {index:02d}: {candidate['strategy']}", fill=color, font=_font(11)
        )
        estimate = candidate["decoder_iou_estimate"]
        text.text(
            (4, 22),
            f"area {candidate['area_px']} px, decoder IoU est. {estimate:.2f}",
            fill=(220, 220, 220),
            font=font,
        )
        tile = Image.new("RGB", (CROP_SIZE, CROP_SIZE + 44), (0, 0, 0))
        tile.paste(crop, (0, 0))
        tile.paste(caption, (0, CROP_SIZE))
        crops.append(tile)
    header_height = 48
    row_height = max(full.height, CROP_SIZE + 44) + header_height
    row = Image.new(
        "RGB", (FULL_WIDTH + 8 + len(crops) * (CROP_SIZE + 8), row_height), (20, 20, 20)
    )
    head = ImageDraw.Draw(row)
    head.text(
        (4, 4),
        f"{cell.view}  {cell.part}  frame {cell.frame}  (cross-strategy disagreement "
        f"{cell.disagreement:.2f}; {record['decision_for_b3']})",
        fill=(255, 255, 255),
        font=_font(16),
    )
    head.text((4, 26), record["reason"][:150], fill=(180, 180, 180), font=_font(12))
    row.paste(full, (0, header_height))
    for index, tile in enumerate(crops):
        row.paste(tile, (FULL_WIDTH + 8 + index * (CROP_SIZE + 8), header_height))
    return row


def write_view_sheet(view: str, cells: list[ProposalCell], video: Path, output: Path) -> Path:
    rows = []
    for cell in cells:
        record = json.loads((cell.directory / "proposal.json").read_text(encoding="utf-8"))
        rows.append(_cell_row(cell, _read_frame(video, cell.frame), record))
    width = max(row.width for row in rows)
    title_height = 40
    sheet = Image.new("RGB", (width, title_height + sum(row.height + 6 for row in rows)), (0, 0, 0))
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (4, 6),
        f"{view}: seed-search proposals, accept one candidate per cell or reject all "
        f"(decisions -> {TEMPLATE_PATH.name}); candidates are agent decoder outputs, not seeds",
        fill=(255, 255, 255),
        font=_font(16),
    )
    y = title_height
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height + 6
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return output


def decisions_template(
    proposals: dict[str, list[ProposalCell]], sheets_root: Path, repository_root: Path
) -> dict:
    cells = []
    for view, view_cells in sorted(proposals.items()):
        for cell in view_cells:
            record = json.loads((cell.directory / "proposal.json").read_text(encoding="utf-8"))
            proposal = cell.directory / "proposal.json"
            if proposal.is_relative_to(repository_root):
                proposal = proposal.relative_to(repository_root)
            cells.append(
                {
                    "view": view,
                    "part": cell.part,
                    "analysis_frame_index": cell.frame,
                    "proposal": proposal.as_posix(),
                    "contact_sheet": (sheets_root / f"{view}.png").as_posix(),
                    "candidates": [
                        {
                            "index": index,
                            "strategy": item["strategy"],
                            "mask_sha256": item["sha256"],
                            "area_px": item["area_px"],
                        }
                        for index, item in enumerate(record["candidates"])
                    ],
                    "decision": None,
                    "accepted_candidate": None,
                    "note": "",
                }
            )
    return {
        "schema_version": "1.0",
        "manifest_kind": "seed_proposal_decisions",
        "author": None,
        "author_type": "human",
        "reviewed_at": None,
        "instructions": (
            "For each cell set decision to 'accept' with accepted_candidate = the candidate index "
            "that is the part (and only the part), or 'reject' with accepted_candidate null when "
            "no candidate is; 'unsure' is allowed. Fill author and reviewed_at (UTC ISO 8601). The "
            "agent then copies each decision into the proposal.json human_decision field and "
            "records the acceptance rate per view and part; nothing is re-run without your say."
        ),
        "decision_vocabulary": ["accept", "reject", "unsure"],
        "generated_at": datetime.now(UTC).isoformat(),
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_repository_root(parser)
    parser.add_argument("--proposals-root", type=Path, default=PROPOSALS_ROOT)
    parser.add_argument(
        "--consensus-root",
        type=Path,
        default=CONSENSUS_ROOT,
        help="Consensus root whose sources name each view's run directory (for the proxy video).",
    )
    add_output_root(parser, OUTPUT_ROOT)
    parser.add_argument("--template", type=Path, default=TEMPLATE_PATH)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    proposals = load_proposals(root, args.proposals_root)
    if not proposals:
        raise SystemExit(f"no proposals under {args.proposals_root}")
    manifest = load_consensus(root / args.consensus_root / MANIFEST_NAME)
    run_by_view = {source.view: root / source.run_directory_uri for source in manifest.sources}
    for view, cells in sorted(proposals.items()):
        run_directory = run_by_view.get(view)
        if run_directory is None:
            print(f"{view}: no run in {args.consensus_root}; skipped")
            continue
        video = next(iter(sorted(run_directory.glob("input_*f.mp4"))), None)
        if video is None:
            print(f"{view}: no bounded proxy video in {run_directory}; skipped")
            continue
        output = write_view_sheet(view, cells, video, root / args.output_root / f"{view}.png")
        print(f"{output.relative_to(root)} ({len(cells)} cells)")
    template = decisions_template(proposals, args.output_root, root)
    (root / args.template).write_text(json.dumps(template, indent=2) + "\n", encoding="utf-8")
    print(f"{args.template} ({len(template['cells'])} cells)")


if __name__ == "__main__":
    main()
