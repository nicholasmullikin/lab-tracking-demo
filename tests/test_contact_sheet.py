"""The shared grid layout reproduces the padding rules of the former per-sheet loops."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from battle import contact_sheet


def _cell(height: int, width: int, value: int) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def test_full_rows_of_equal_cells_are_a_plain_hstack_vstack() -> None:
    cells = [_cell(10, 20, v) for v in (1, 2, 3, 4, 5, 6)]
    grid = contact_sheet.render_grid(cells, columns=3)
    expected = np.vstack([np.hstack(cells[:3]), np.hstack(cells[3:])])
    assert np.array_equal(grid, expected)


def test_short_last_row_is_filled_like_the_former_zeros_like_padding() -> None:
    tiles = [_cell(10, 20, v) for v in (1, 2, 3)]
    grid = contact_sheet.render_grid(tiles, columns=2)
    rows = [np.hstack(tiles[:2]), np.hstack([tiles[2], np.zeros_like(tiles[0])])]
    assert np.array_equal(grid, np.vstack(rows))


def test_rows_of_different_widths_are_padded_right_and_cells_bottom() -> None:
    cells = [_cell(10, 20, 1), _cell(14, 30, 2), _cell(8, 12, 3)]
    grid = contact_sheet.render_grid(cells, columns=2)
    # Row 1: 50 wide, 14 tall (cell 1 padded at the bottom); row 2: 12 wide padded to 50.
    assert grid.shape == (22, 50, 3)
    assert np.all(grid[:10, :20] == 1) and np.all(grid[10:14, :20] == 0)
    assert np.all(grid[:14, 20:50] == 2)
    assert np.all(grid[14:, :12] == 3) and np.all(grid[14:, 12:] == 0)
    # The seed-search loop padded with np.pad; the dataset sheet with copyMakeBorder: same bytes.
    former = [
        np.hstack(
            [cv2.copyMakeBorder(cells[0], 0, 4, 0, 0, cv2.BORDER_CONSTANT, value=0), cells[1]]
        )
    ]
    former.append(np.pad(cells[2], ((0, 0), (0, 50 - 12), (0, 0))))
    assert np.array_equal(grid, np.vstack(former))


def test_labels_and_title_add_the_dataset_sheet_bars() -> None:
    cells = [_cell(10, 40, 7), _cell(10, 40, 8)]
    grid = contact_sheet.render_grid(cells, columns=2, labels=["a", "b"], title="t")
    assert grid.shape == (contact_sheet.TITLE_HEIGHT + contact_sheet.LABEL_HEIGHT + 10, 80, 3)
    title = grid[: contact_sheet.TITLE_HEIGHT]
    assert title.min() == 35 and title.max() == 255  # grey strip with white text
    label = grid[
        contact_sheet.TITLE_HEIGHT : contact_sheet.TITLE_HEIGHT + contact_sheet.LABEL_HEIGHT, :40
    ]
    assert label.min() == 20 and label.max() > 200  # dark strip with anti-aliased light text
    assert np.array_equal(label, contact_sheet.label_bar(40, "a")) and np.array_equal(
        title, contact_sheet.title_bar(80, "t")
    )
    body = grid[contact_sheet.TITLE_HEIGHT + contact_sheet.LABEL_HEIGHT :]
    assert np.all(body[:, :40] == 7) and np.all(body[:, 40:] == 8)


def test_gap_separates_cells_and_rows_without_a_trailing_strip() -> None:
    cells = [_cell(4, 6, 1), _cell(4, 6, 2), _cell(4, 6, 3)]
    grid = contact_sheet.render_grid(cells, columns=2, gap=2, fill=9)
    assert grid.shape == (4 + 2 + 4, 6 + 2 + 6, 3)
    assert np.all(grid[:4, 6:8] == 9) and np.all(grid[4:6] == 9)
    assert np.all(grid[6:, :6] == 3) and np.all(grid[6:, 8:] == 9)


def test_render_grid_rejects_empty_input_bad_columns_and_mismatched_labels() -> None:
    with pytest.raises(ValueError, match="no cells"):
        contact_sheet.render_grid([], columns=2)
    with pytest.raises(ValueError, match="columns"):
        contact_sheet.render_grid([_cell(2, 2, 1)], columns=0)
    with pytest.raises(ValueError, match="labels"):
        contact_sheet.render_grid([_cell(2, 2, 1)], columns=1, labels=["a", "b"])
