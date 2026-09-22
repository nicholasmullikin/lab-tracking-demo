"""Grid layout for the review contact sheets (seed, exemplar, acceptance, dataset, proposals).

Each sheet drawer keeps its own cell content (the crop, the outlines, the caption burnt into
the tile); what they shared was the layout: cells in rows of `columns`, every cell in a row
padded at the bottom to the row's tallest cell, every row padded on the right to the widest
row, an optional label bar above each cell and an optional title bar above the grid.  The
padding is the constant fill each drawer used (black), so a sheet whose rows are all full
comes out pixel for pixel as before; a short last row is filled the way `np.zeros_like`
filled it.  Arrays are OpenCV images (H x W x 3, uint8, any channel order).
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

LABEL_HEIGHT = 30
TITLE_HEIGHT = 44
FONT = cv2.FONT_HERSHEY_SIMPLEX
Color = tuple[int, int, int]


def text_bar(
    width: int,
    text: str,
    *,
    height: int,
    background: int,
    color: Color = (245, 245, 245),
    font_scale: float = 0.42,
    thickness: int = 1,
    origin: tuple[int, int] = (6, 20),
) -> np.ndarray:
    """A `height x width` strip of `background` grey with `text` drawn once (anti-aliased)."""
    bar = np.full((height, width, 3), background, dtype=np.uint8)
    cv2.putText(bar, text, origin, FONT, font_scale, color, thickness, cv2.LINE_AA)
    return bar


def label_bar(width: int, text: str, *, height: int = LABEL_HEIGHT) -> np.ndarray:
    """The dark label strip the dataset contact sheet puts above each panel."""
    return text_bar(width, text, height=height, background=20)


def title_bar(width: int, text: str, *, height: int = TITLE_HEIGHT) -> np.ndarray:
    """The lighter title strip above a whole sheet."""
    return text_bar(
        width,
        text,
        height=height,
        background=35,
        color=(255, 255, 255),
        font_scale=0.6,
        origin=(10, 29),
    )


def _pad(image: np.ndarray, *, bottom: int = 0, right: int = 0, value: int = 0) -> np.ndarray:
    if not bottom and not right:
        return image
    # A scalar `value` would fill only the first channel.
    return cv2.copyMakeBorder(
        image, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=(value, value, value)
    )


def render_grid(
    cells: Sequence[np.ndarray],
    *,
    columns: int,
    labels: Sequence[str] | None = None,
    title: str | None = None,
    gap: int = 0,
    fill: int = 0,
) -> np.ndarray:
    """Lay `cells` out in rows of `columns`; see the module docstring for the padding rule.

    `labels` (one per cell) puts a :func:`label_bar` above each cell, `title` a
    :func:`title_bar` above the grid, `gap` pixels of `fill` between cells and rows.
    """
    if not cells:
        raise ValueError("no cells to render")
    if columns < 1:
        raise ValueError("columns must be at least 1")
    if labels is not None and len(labels) != len(cells):
        raise ValueError(f"{len(labels)} labels for {len(cells)} cells")
    items = [
        np.vstack((label_bar(cell.shape[1], text), cell)) if labels is not None else cell
        for cell, text in zip(cells, labels if labels is not None else [None] * len(cells))
    ]
    rows: list[np.ndarray] = []
    for start in range(0, len(items), columns):
        row_cells = items[start : start + columns]
        tallest = max(cell.shape[0] for cell in row_cells)
        padded = [_pad(cell, bottom=tallest - cell.shape[0], value=fill) for cell in row_cells]
        if gap:
            spacer = np.full((tallest, gap, 3), fill, dtype=np.uint8)
            padded = [part for cell in padded for part in (cell, spacer)][:-1]
        rows.append(np.hstack(padded))
    widest = max(row.shape[1] for row in rows)
    rows = [_pad(row, right=widest - row.shape[1], value=fill) for row in rows]
    if gap:
        spacer = np.full((gap, widest, 3), fill, dtype=np.uint8)
        rows = [part for row in rows for part in (row, spacer)][:-1]
    if title is not None:
        rows.insert(0, title_bar(widest, title))
    return np.vstack(rows)
