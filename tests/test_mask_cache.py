from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from battle import mask_cache


def _write_mask(path: Path, *, rows: slice, columns: slice, shape=(8, 12)) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[rows, columns] = True
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask * 255).astype(np.uint8)).save(path)
    return mask


def test_cache_decodes_each_mask_once_per_run(tmp_path: Path) -> None:
    expected = _write_mask(tmp_path / "masks/000000_00.png", rows=slice(1, 4), columns=slice(2, 6))
    cache = mask_cache.MaskCache(tmp_path)

    first = cache.mask("masks/000000_00.png")
    second = cache.mask("masks/000000_00.png")

    assert np.array_equal(first, expected)
    assert np.array_equal(second, expected)
    assert cache.png_decodes == 1


def test_sidecar_serves_masks_and_cutouts_without_touching_the_png(tmp_path: Path) -> None:
    uri = "masks/000000_00.png"
    expected = _write_mask(tmp_path / uri, rows=slice(0, 2), columns=slice(0, 3))
    color = (70, 130, 255)
    direct = mask_cache.encode_rgba_mask_png(expected, color)

    mask_cache.write_sidecar(tmp_path, {uri: (color,)})
    cache = mask_cache.MaskCache(tmp_path)

    assert np.array_equal(cache.mask(uri), expected)
    assert cache.rgba_png(uri, color) == direct
    assert cache.png_decodes == 0
    assert cache.rgba_encodes == 0
    assert cache.sidecar_hits == 2


def test_a_rewritten_mask_invalidates_its_sidecar_entry(tmp_path: Path) -> None:
    uri = "masks/000000_00.png"
    _write_mask(tmp_path / uri, rows=slice(0, 2), columns=slice(0, 3))
    mask_cache.write_sidecar(tmp_path, {uri: (mask_cache.REVIEW_COLORS["chassis"],)})

    rewritten = _write_mask(tmp_path / uri, rows=slice(4, 7), columns=slice(5, 11))
    cache = mask_cache.MaskCache(tmp_path)

    assert np.array_equal(cache.mask(uri), rewritten)
    assert cache.png_decodes == 1
    assert cache.sidecar_hits == 0
    chassis = mask_cache.REVIEW_COLORS["chassis"]
    assert cache.rgba_png(uri, chassis) == mask_cache.encode_rgba_mask_png(rewritten, chassis)
    assert cache.rgba_encodes == 1


def test_a_mask_outside_its_run_directory_is_refused(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    _write_mask(tmp_path / "outside.png", rows=slice(0, 1), columns=slice(0, 1))
    cache = mask_cache.MaskCache(run)

    with pytest.raises(ValueError, match="escapes its run directory"):
        cache.mask("../outside.png")


def test_bounded_memory_evicts_the_least_recently_used_mask(tmp_path: Path) -> None:
    uris = tuple(f"masks/{index:06d}_00.png" for index in range(4))
    for index, uri in enumerate(uris):
        _write_mask(tmp_path / uri, rows=slice(0, 1 + index), columns=slice(0, 2))
    cache = mask_cache.MaskCache(tmp_path, memory_masks=2)

    for uri in uris:
        cache.mask(uri)
    assert cache.png_decodes == 4

    cache.mask(uris[-1])
    assert cache.png_decodes == 4
    cache.mask(uris[0])
    assert cache.png_decodes == 5


def test_cache_for_shares_one_instance_per_run_directory(tmp_path: Path) -> None:
    mask_cache.clear_caches()
    try:
        assert mask_cache.cache_for(tmp_path) is mask_cache.cache_for(tmp_path)
        assert mask_cache.cache_for(tmp_path) is not mask_cache.cache_for(tmp_path / "other")
    finally:
        mask_cache.clear_caches()
