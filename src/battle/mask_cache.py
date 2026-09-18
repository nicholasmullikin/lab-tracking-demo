"""One decoded-mask cache shared by the geometry passes and the Rerun exporters.

A review package walks the same masks several times: once for contact geometry, once
for the segmentation triggers, and once more to encode the RGBA cut-outs the viewer
draws. Decoding those PNGs again on every pass, and re-encoding cut-outs that depend
only on the mask bytes and a fixed colour, is the bulk of a rebuild.

This module keeps a bounded in-memory cache per run directory and can persist one
sidecar per run holding the bit-packed masks and the encoded cut-outs. The sidecar
records each source PNG's size and modification time, so a stale entry is detected and
the PNG is read instead of trusted.
"""

from __future__ import annotations

import argparse
import io
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image

SIDECAR_RELATIVE_PATH = Path("native/mask_cache.npz")
MEMORY_MASK_LIMIT = 48
MEMORY_RGBA_LIMIT = 256
Color = tuple[int, int, int]


def _color_key(color: Color) -> str:
    red, green, blue = color
    return f"{red:02x}{green:02x}{blue:02x}"


def _entry_key(uri: str) -> str:
    return uri.replace("/", "\x1c")


def _stamp(path: Path) -> np.ndarray:
    status = path.stat()
    return np.array([status.st_size, status.st_mtime_ns], dtype=np.int64)


def decode_mask_png(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) > 0


def encode_rgba_mask_png(mask: np.ndarray, color: Color) -> bytes:
    """Encode a cut-out: the object colour where the mask is set, transparent elsewhere.

    Level 1 deflate is both faster and roughly nine hundred times smaller than storing
    the raw frame, because a cut-out is one flat region on a transparent field.
    """
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[mask] = (*color, 255)
    buffer = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG", compress_level=1)
    return buffer.getvalue()


class MaskCache:
    """Decoded masks and encoded cut-outs for one run directory."""

    def __init__(
        self,
        run_directory: Path,
        *,
        memory_masks: int = MEMORY_MASK_LIMIT,
        memory_rgba: int = MEMORY_RGBA_LIMIT,
    ) -> None:
        self.run_directory = run_directory.resolve()
        self._memory_masks = memory_masks
        self._memory_rgba = memory_rgba
        self._masks: OrderedDict[str, np.ndarray] = OrderedDict()
        self._rgba: OrderedDict[tuple[str, str], bytes] = OrderedDict()
        self._sidecar: dict[str, np.ndarray] | None = None
        self._sidecar_loaded = False
        self.png_decodes = 0
        self.rgba_encodes = 0
        self.sidecar_hits = 0

    @property
    def sidecar_path(self) -> Path:
        return self.run_directory / SIDECAR_RELATIVE_PATH

    def _sidecar_entries(self) -> dict[str, np.ndarray] | None:
        if not self._sidecar_loaded:
            self._sidecar_loaded = True
            path = self.sidecar_path
            if path.is_file():
                # NpzFile members decompress on access, so this stays lazy per mask.
                self._sidecar = np.load(path)  # type: ignore[assignment]
        return self._sidecar

    def _resolved(self, uri: str) -> Path:
        path = (self.run_directory / uri).resolve()
        if not path.is_relative_to(self.run_directory):
            raise ValueError(f"mask reference escapes its run directory: {uri}")
        if not path.is_file():
            raise FileNotFoundError(f"referenced mask is unavailable: {path}")
        return path

    def _fresh_sidecar_member(self, uri: str, name: str, path: Path) -> np.ndarray | None:
        entries = self._sidecar_entries()
        if entries is None:
            return None
        key = _entry_key(uri)
        stamp_name = f"stamp/{key}"
        if name not in entries or stamp_name not in entries:
            return None
        if not np.array_equal(entries[stamp_name], _stamp(path)):
            return None
        return entries[name]

    def mask(self, uri: str) -> np.ndarray:
        """Return the decoded boolean mask for one run-relative URI."""
        cached = self._masks.get(uri)
        if cached is not None:
            self._masks.move_to_end(uri)
            return cached
        path = self._resolved(uri)
        key = _entry_key(uri)
        packed = self._fresh_sidecar_member(uri, f"bits/{key}", path)
        if packed is not None:
            entries = self._sidecar_entries()
            assert entries is not None
            height, width = (int(value) for value in entries[f"shape/{key}"])
            mask = np.unpackbits(packed, count=height * width).reshape(height, width).astype(bool)
            self.sidecar_hits += 1
        else:
            mask = decode_mask_png(path)
            self.png_decodes += 1
        self._masks[uri] = mask
        while len(self._masks) > self._memory_masks:
            self._masks.popitem(last=False)
        return mask

    def rgba_png(self, uri: str, color: Color) -> bytes:
        """Return the encoded RGBA cut-out for one mask and display colour."""
        memory_key = (uri, _color_key(color))
        cached = self._rgba.get(memory_key)
        if cached is not None:
            self._rgba.move_to_end(memory_key)
            return cached
        path = self._resolved(uri)
        stored = self._fresh_sidecar_member(
            uri, f"rgba/{_color_key(color)}/{_entry_key(uri)}", path
        )
        if stored is not None:
            contents = stored.tobytes()
            self.sidecar_hits += 1
        else:
            contents = encode_rgba_mask_png(self.mask(uri), color)
            self.rgba_encodes += 1
        self._rgba[memory_key] = contents
        while len(self._rgba) > self._memory_rgba:
            self._rgba.popitem(last=False)
        return contents

    def stats(self) -> dict[str, int | bool]:
        return {
            "png_decodes": self.png_decodes,
            "rgba_encodes": self.rgba_encodes,
            "sidecar_hits": self.sidecar_hits,
            "sidecar_present": self.sidecar_path.is_file(),
        }


_CACHES: dict[Path, MaskCache] = {}


def cache_for(run_directory: Path) -> MaskCache:
    """Reuse one cache per run directory so separate passes share decoded masks."""
    resolved = run_directory.resolve()
    cache = _CACHES.get(resolved)
    if cache is None:
        cache = MaskCache(resolved)
        _CACHES[resolved] = cache
    return cache


def clear_caches() -> None:
    _CACHES.clear()


def write_sidecar(
    run_directory: Path,
    colors_by_uri: dict[str, tuple[Color, ...]],
) -> Path:
    """Pack every named mask, and the cut-outs each one is drawn with, into one archive.

    The archive is deflated because a bit-packed mask is mostly zeros, and only the
    colours a mask is actually logged with are encoded: packing the whole palette for
    every mask multiplies the cut-out payload by the number of targets for no gain.
    """
    run_directory = run_directory.resolve()
    payload: dict[str, np.ndarray] = {}
    for uri, colors in sorted(colors_by_uri.items()):
        path = (run_directory / uri).resolve()
        if not path.is_relative_to(run_directory):
            raise ValueError(f"mask reference escapes its run directory: {uri}")
        if not path.is_file():
            raise FileNotFoundError(f"referenced mask is unavailable: {path}")
        mask = decode_mask_png(path)
        key = _entry_key(uri)
        payload[f"bits/{key}"] = np.packbits(mask)
        payload[f"shape/{key}"] = np.array(mask.shape, dtype=np.int64)
        payload[f"stamp/{key}"] = _stamp(path)
        for color in colors:
            payload[f"rgba/{_color_key(color)}/{key}"] = np.frombuffer(
                encode_rgba_mask_png(mask, color), dtype=np.uint8
            )
    output = run_directory / SIDECAR_RELATIVE_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    clear_caches()
    return output


REVIEW_COLORS: dict[str, Color] = {
    "chassis": (70, 130, 255),
    "interior": (60, 210, 150),
    "rear_body": (255, 190, 45),
    "cabin": (255, 95, 100),
}


def logged_colors_by_uri(run_directory: Path) -> dict[str, tuple[Color, ...]]:
    """Map each referenced mask to the colours the review and export palettes give it.

    Both palettes are derived from the run itself: the review package colours a mask by
    its target label, and the exporter colours it by the object's stable class id.
    """
    from .exporter import _object_annotations
    from .schemas import FrameObservations, RunManifest

    observations = tuple(
        FrameObservations.model_validate_json(line)
        for line in (run_directory / "observations.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    manifest_path = run_directory / "manifest.json"
    exporter_colors: dict[str, Color] = {}
    if manifest_path.is_file():
        manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        for view_annotations in _object_annotations(manifest).values():
            for object_id, (_, _, color) in view_annotations.items():
                exporter_colors[object_id] = color

    colors_by_uri: dict[str, tuple[Color, ...]] = {}
    for observation in observations:
        for item in observation.objects:
            if item.mask is None:
                continue
            colors = {
                color
                for color in (REVIEW_COLORS.get(item.label), exporter_colors.get(item.object_id))
                if color is not None
            }
            colors_by_uri[item.mask.uri] = tuple(sorted(colors))
    return colors_by_uri


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument(
        "--masks-only",
        action="store_true",
        help="Pack the masks without any encoded cut-outs.",
    )
    args = parser.parse_args()
    colors_by_uri = logged_colors_by_uri(args.run_directory)
    if args.masks_only:
        colors_by_uri = dict.fromkeys(colors_by_uri, ())
    output = write_sidecar(args.run_directory, colors_by_uri)
    print(f"{output} ({output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
