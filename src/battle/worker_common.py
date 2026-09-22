"""Helpers shared by the external-interpreter video workers.

The DAM4SAM, SAMURAI, Grounding DINO + SAM2 and four-part workers each decoded the bounded
proxy into `native/frames/00000.jpg ...` with the same cv2 loop and read the same
`torch.cuda.max_memory_allocated` for their `gpu_peak_vram_bytes`.  This module holds both
once.  The workers run under their own pyenv interpreters (samurai / grounded_sam2 are Python
3.10) where the Battle package is not installed, so like `fs_common` and `gpu_guard` it
imports only the standard library at module level (cv2 and torch are imported inside the
functions, as the workers already do) and stays 3.10 compatible; it is imported as
`battle.worker_common` by the package and as a sibling module by the workers::

    try:
        from . import worker_common
    except ImportError:
        sys.path.append(str(Path(__file__).resolve().parent))
        import worker_common

The four `_extract_frames` copies differed in two behaviours, kept here as parameters and
documented per former copy:

| former copy                        | `isOpened()` check | short video            | returned |
|------------------------------------|--------------------|------------------------|----------|
| `dam4sam_video_worker`             | yes                | error unless exact     | paths    |
| `samurai_video_worker`             | yes                | error unless exact     | count    |
| `grounding_dino_sam2_video_worker` | yes                | error only on 0 frames | count    |
| `four_part_video_worker`           | no                 | error unless exact     | paths    |

:func:`extract_frames` always returns the paths (the count callers take `len()`); the error
texts are unified to the "video produced N frames" form three of the copies shared.
"""

from __future__ import annotations

from pathlib import Path


def extract_frames(
    video: Path,
    directory: Path,
    frame_count: int,
    *,
    exact: bool = True,
    check_open: bool = True,
) -> list[Path]:
    """Decode the first `frame_count` frames of `video` into `directory/<index:05d>.jpg`.

    Frames are written with cv2's default JPEG quality, as every worker did.  With
    `check_open` a video cv2 cannot open is an error before decoding (the four-part worker
    let it surface as a zero-frame result).  With `exact` (default) fewer decodable frames
    than requested is an error; otherwise only a video that yields no frame at all is.
    """
    # The two error branches are the former copies' texts; "for bounded smoke" (SAMURAI) and
    # "decoded N frames, expected M" (four-part) collapse onto the DAM4SAM wording.
    import cv2

    directory.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    if check_open and not capture.isOpened():
        raise RuntimeError(f"could not open video: {video}")
    frames: list[Path] = []
    while len(frames) < frame_count:
        ok, frame = capture.read()
        if not ok:
            break
        path = directory / f"{len(frames):05d}.jpg"
        cv2.imwrite(str(path), frame)
        frames.append(path)
    capture.release()
    if exact:
        if len(frames) != frame_count:
            raise RuntimeError(
                f"video produced {len(frames)} frames; expected exactly {frame_count}"
            )
    elif not frames:
        raise RuntimeError(f"video produced zero frames: {video}")
    return frames


def cuda_peak_bytes(device: object = None) -> int:
    """This process's peak CUDA allocation so far (`torch.cuda.max_memory_allocated`).

    The number every worker records as `gpu_peak_vram_bytes`: bytes allocated by the caching
    allocator for this process on `device` (the current device when None), not the card's
    reserved or total usage, so a run beside a neighbour still reports its own peak.
    """
    import torch

    return int(torch.cuda.max_memory_allocated(device))
