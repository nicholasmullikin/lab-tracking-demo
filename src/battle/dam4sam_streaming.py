"""Battle-owned DAM4SAM streaming wrapper: one SAM2 predictor shared by four trackers.

The upstream `DAM4SAMTracker` (external checkout, never modified here) builds its own
SAM2 model in `__init__`, so four targets meant four models plus four growing inference
states; the 60 s tiny run peaked at 7.3 GiB.  This module hands one already-built
`predictor` to every tracker, each keeping its own `inference_state`, and adds three
run conditions the fairness comparison needs:

* `--sam2-model tiny|large`: checkpoint + yaml pair from the DAM4SAM checkout, SHA-256
  pinned here like the tiny one in `dam4sam_video.py`;
* `--input-size`: `tracker.input_image_size` plus a battle-owned copy of the yaml with
  `image_size` replaced, written into the run directory and fingerprinted;
* `correct(image, mask)`: a mid-stream conditioning frame through
  `predictor.add_new_mask`, so the SAM2 arms can consume the same correction schedule
  as the SAM3 reference.

Sharing is safe because the only predictor-level mutable state DAM4SAM relies on is
`predictor.curr_out`, written by `propagate_in_video` and read by `add_to_drm` inside the
same `track()` call; the worker runs the four trackers strictly one after another per
frame, so no other tracker's propagate can interleave.

Everything torch- or dam4sam-specific is imported lazily: the module must import in the
battle venv (no torch) for unit tests that drive `correct()` with a stub predictor, and
as a top-level module (`import dam4sam_streaming`) inside the external worker process.
"""

from __future__ import annotations

import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_INPUT_IMAGE_SIZE = 1024
SUPPORTED_INPUT_IMAGE_SIZES = (1024, 1536)
# Same file `dam4sam_video.SAM2_CHECKPOINT_SHA256` pins; repeated here because this module
# must import without the battle package (the worker runs in the samurai pyenv).
SAM2_TINY_CHECKPOINT_SHA256 = "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69"
SAM2_LARGE_CHECKPOINT_SHA256 = "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"
SAM2_MODELS: dict[str, dict[str, str]] = {
    "tiny": {
        "checkpoint": "checkpoints/sam2.1_hiera_tiny.pt",
        "config": "sam21pp_hiera_t.yaml",
        "tracker_name": "sam21pp-T",
        "sha256": SAM2_TINY_CHECKPOINT_SHA256,
    },
    "large": {
        "checkpoint": "checkpoints/sam2.1_hiera_large.pt",
        "config": "sam21pp_hiera_l.yaml",
        "tracker_name": "sam21pp-L",
        "sha256": SAM2_LARGE_CHECKPOINT_SHA256,
    },
}
IMAGE_SIZE_LINE = re.compile(r"^(\s*image_size:\s*)(\d+)(\s*(#.*)?)$", re.MULTILINE)
CORRECTION_API = "add_new_mask"
# ImageNet normalisation constants DAM4SAMTracker.__init__ would have set.
IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Sam2ModelSpec:
    model: str
    checkpoint: Path
    config_name: str
    config_path: Path
    checkpoint_sha256: str
    tracker_name: str


def resolve_sam2_model(sam2_root: Path, model: str) -> Sam2ModelSpec:
    """Map `tiny|large` onto the DAM4SAM checkout's checkpoint and yaml."""
    if model not in SAM2_MODELS:
        raise ValueError(f"unknown SAM2 model {model!r}; expected one of {tuple(SAM2_MODELS)}")
    entry = SAM2_MODELS[model]
    root = Path(sam2_root)
    return Sam2ModelSpec(
        model=model,
        checkpoint=root / entry["checkpoint"],
        config_name=entry["config"],
        config_path=root / "sam2" / entry["config"],
        checkpoint_sha256=entry["sha256"],
        tracker_name=entry["tracker_name"],
    )


def verify_checkpoint(spec: Sam2ModelSpec) -> str:
    """Return the measured SHA-256 after checking it against the pinned one."""
    if not spec.checkpoint.is_file():
        raise FileNotFoundError(spec.checkpoint)
    measured = sha256_file(spec.checkpoint)
    if measured != spec.checkpoint_sha256:
        raise ValueError(
            f"SAM2 {spec.model} checkpoint checksum mismatch for {spec.checkpoint}: "
            f"{measured} != pinned {spec.checkpoint_sha256}"
        )
    return measured


def replace_image_size(yaml_text: str, input_size: int) -> str:
    """Rewrite the single `image_size:` line of a SAM2 yaml; refuse anything ambiguous."""
    matches = IMAGE_SIZE_LINE.findall(yaml_text)
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one image_size line in the SAM2 yaml, found {len(matches)}"
        )
    return IMAGE_SIZE_LINE.sub(
        lambda match: f"{match.group(1)}{input_size}{match.group(3)}", yaml_text
    )


def write_input_size_config(
    source_config: Path, input_size: int, destination_directory: Path
) -> Path:
    """Write the battle-owned yaml copy with `image_size` set to `input_size`."""
    if input_size not in SUPPORTED_INPUT_IMAGE_SIZES:
        raise ValueError(
            f"input size {input_size} is not one of {SUPPORTED_INPUT_IMAGE_SIZES}; every SAM2 "
            "stride (16 for the backbone, 4 for the low-res masks) must divide it"
        )
    source = Path(source_config)
    destination_directory = Path(destination_directory)
    destination_directory.mkdir(parents=True, exist_ok=True)
    destination = destination_directory / f"{source.stem}_image{input_size}{source.suffix}"
    destination.write_text(
        replace_image_size(source.read_text(encoding="utf-8"), input_size), encoding="utf-8"
    )
    return destination


def config_provenance(path: Path, *, source: str) -> dict[str, Any]:
    return {"uri": str(Path(path).resolve()), "sha256": sha256_file(path), "source": source}


def build_shared_predictor(
    sam2_root: Path,
    model: str,
    input_size: int,
    config_directory: Path,
    *,
    device: str = "cuda:0",
) -> tuple[Any, dict[str, Any]]:
    """Build one SAM2 video predictor for all four trackers and return its provenance.

    For `input_size == 1024` the checkout's own yaml is loaded through the hydra search
    path `sam2/__init__.py` registers.  Otherwise a battle-owned copy with `image_size`
    replaced is written into `config_directory` and hydra is re-pointed at that directory
    (hydra composes by config *name*, so the copy cannot simply be passed as a path).
    """
    root = Path(sam2_root)
    spec = resolve_sam2_model(root, model)
    measured_sha = verify_checkpoint(spec)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from sam2.build_sam import build_sam2_video_predictor

    if input_size == DEFAULT_INPUT_IMAGE_SIZE:
        config_name = spec.config_name
        config_path = spec.config_path
        config_source = "checkout"
    else:
        copied = write_input_size_config(spec.config_path, input_size, config_directory)
        from hydra import initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        GlobalHydra.instance().clear()
        initialize_config_dir(config_dir=str(copied.parent.resolve()), version_base="1.2")
        config_name = copied.name
        config_path = copied
        config_source = "battle_input_size_copy"
    predictor = build_sam2_video_predictor(config_name, str(spec.checkpoint), device=device)
    if int(getattr(predictor, "image_size", input_size)) != int(input_size):
        raise RuntimeError(
            f"predictor image_size {predictor.image_size} does not match input size {input_size}"
        )
    provenance = {
        "sam2_model": model,
        "tracker_name": spec.tracker_name,
        "sam2_checkpoint_uri": str(spec.checkpoint),
        "sam2_checkpoint_sha256": measured_sha,
        "sam2_checkpoint_sha256_pinned": True,
        "sam2_config_uri": str(config_path.resolve()),
        "sam2_config_sha256": sha256_file(config_path),
        "sam2_config_source": config_source,
        "input_image_size": int(input_size),
        "shared_predictor": True,
    }
    return predictor, provenance


def logits_to_mask(logits: Any) -> np.ndarray:
    """Threshold `[0, 0]` of the video-resolution logits exactly as DAM4SAM does."""
    first = logits[0, 0]
    if hasattr(first, "detach"):
        first = first.detach().float().cpu().numpy()
    return (np.asarray(first) > 0).astype(np.uint8)


def fit_mask(mask: Any, frame_shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize of a binary mask to `(height, width)` when needed."""
    binary = np.asarray(mask).astype(bool)
    if binary.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {binary.shape}")
    if binary.shape == tuple(frame_shape):
        return binary
    from PIL import Image

    height, width = frame_shape
    resized = Image.fromarray(binary.astype(np.uint8) * 255).resize(
        (int(width), int(height)), Image.NEAREST
    )
    return np.asarray(resized, dtype=np.uint8) > 0


def read_correction_mask(
    path: str | Path, expected_sha256: str, frame_shape: tuple[int, int]
) -> tuple[np.ndarray, bool]:
    """Load one schedule mask as the SAM3 worker does (hash, grayscale > 0, non-empty).

    The SAM3 worker refuses a mask whose size differs from the frame; here a size mismatch
    is resized nearest-neighbour instead, because the 1536 arm runs on a 1080p proxy while
    the schedule was authored on the 1280x720 one.  The second value says whether that
    happened so the manifest can record it.
    """
    from PIL import Image

    mask_path = Path(path)
    if sha256_file(mask_path) != expected_sha256:
        raise ValueError(f"selected calibration mask SHA-256 changed: {mask_path}")
    with Image.open(mask_path) as image:
        binary = np.asarray(image.convert("L"), dtype=np.uint8) > 0
    if not bool(binary.any()):
        raise ValueError(f"selected calibration mask is empty: {mask_path}")
    resized = binary.shape != tuple(frame_shape)
    return fit_mask(binary, frame_shape), resized


class SharedPredictorMixin:
    """`DAM4SAMTracker` behaviour on a caller-supplied predictor, plus `correct()`.

    Mixed in ahead of the upstream class by `shared_predictor_tracker_class`, so
    `initialize`, `track`, `_prepare_image` and the DRM logic stay upstream's verbatim.
    """

    predictor: Any
    input_image_size: int
    add_correction_to_drm: bool
    correction_frames: list[int]

    def __init__(
        self,
        predictor: Any,
        *,
        input_image_size: int = DEFAULT_INPUT_IMAGE_SIZE,
        add_correction_to_drm: bool = False,
    ) -> None:
        # Deliberately no super().__init__(): upstream's would build a second SAM2 model.
        self.predictor = predictor
        self.input_image_size = int(input_image_size)
        self.add_correction_to_drm = bool(add_correction_to_drm)
        self.correction_frames = []
        self.tracking_times = []
        self.img_mean, self.img_std = _normalization_tensors()

    def correct(self, image: Any, mask: Any) -> dict[str, Any]:
        """Condition the current frame on `mask` via `add_new_mask`; return its mask.

        Called after `track()` on the same frame, so `frame_index` is left alone.  The
        frame is re-staged in `inference_state["images"]`, `add_new_mask` runs with
        `is_init_cond_frame=False` (the frame is already tracked) and, because the yaml
        sets `add_all_frames_to_correct_as_cond`, stores the corrected output as a SAM2
        conditioning frame, which is exactly what DAM4SAM's DRM is made of.  The returned
        logits are thresholded as this frame's mask and replace the size `track()` had
        just recorded for it, so the DRM ratio still sees one size per frame.

        `add_correction_to_drm` only decides whether DAM4SAM's own bookkeeping treats the
        correction as a DRM addition (`last_added`, hence the 5-frame throttle and the
        worker's `drm_memory_additions` count).  It never calls `predictor.add_to_drm`,
        which would overwrite the corrected output with the stale `predictor.curr_out`.
        """
        if not hasattr(self, "inference_state"):
            raise RuntimeError("correct() requires initialize() first")
        binary = fit_mask(mask, (self.img_height, self.img_width))
        if not bool(binary.any()):
            raise ValueError(f"empty correction mask at frame {self.frame_index}")
        prepared = self._prepare_image(image)
        if hasattr(prepared, "dim") and prepared.dim() == 3:
            prepared = prepared.unsqueeze(0)
        images = self.inference_state["images"]
        images[self.frame_index] = prepared
        try:
            _, _, out_mask_logits = self.predictor.add_new_mask(
                inference_state=self.inference_state,
                frame_idx=self.frame_index,
                obj_id=0,
                mask=binary,
            )
        finally:
            images.pop(self.frame_index, None)
        pred_mask = logits_to_mask(out_mask_logits)
        n_pixels = int((pred_mask == 1).sum())
        if self.frame_index > 0 and len(self.object_sizes) >= self.frame_index:
            self.object_sizes[-1] = n_pixels
        else:
            self.object_sizes.append(n_pixels)
        self.correction_frames.append(int(self.frame_index))
        if self.add_correction_to_drm:
            self.last_added = self.frame_index
        return {"pred_mask": pred_mask}


def _normalization_tensors() -> tuple[Any, Any]:
    try:
        import torch
    except ImportError:  # battle venv unit tests drive a stub base without torch
        return None, None
    mean = torch.tensor(IMG_MEAN, dtype=torch.float32)[:, None, None]
    std = torch.tensor(IMG_STD, dtype=torch.float32)[:, None, None]
    return mean, std


def shared_predictor_tracker_class(base: type | None = None) -> type:
    """Return `SharedPredictorDAM4SAMTracker`, importing the upstream base lazily."""
    if base is None:
        from dam4sam_tracker import DAM4SAMTracker as base  # noqa: N813

    class SharedPredictorDAM4SAMTracker(SharedPredictorMixin, base):  # type: ignore[misc,valid-type]
        pass

    return SharedPredictorDAM4SAMTracker


def cuda_memory_probe(frames_processed: int) -> dict[str, int]:
    import torch

    return {
        "frames_processed": int(frames_processed),
        "memory_allocated_bytes": int(torch.cuda.memory_allocated()),
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "memory_reserved_bytes": int(torch.cuda.memory_reserved()),
    }


def parse_probe_frames(text: str) -> tuple[int, ...]:
    """`"30,300"` -> `(30, 300)`; counts of frames processed, so 300 is valid in a 300-frame run."""
    values = tuple(sorted({int(item) for item in str(text).split(",") if item.strip()}))
    if any(value < 1 for value in values):
        raise ValueError("VRAM probe frame counts must be positive")
    return values


def merge_probes(probe_lists: list[list[dict[str, int]]]) -> list[dict[str, int]]:
    """Per frames-processed maximum over independent streams (SAMURAI runs one per target)."""
    merged: dict[int, dict[str, int]] = {}
    for probes in probe_lists:
        for probe in probes:
            key = int(probe["frames_processed"])
            current = merged.setdefault(key, dict(probe))
            for field in (
                "memory_allocated_bytes",
                "max_memory_allocated_bytes",
                "memory_reserved_bytes",
            ):
                if field in probe:
                    current[field] = max(int(current.get(field, 0)), int(probe[field]))
    return [merged[key] for key in sorted(merged)]


def extrapolate_vram(
    probes: list[dict[str, int]], *, to_frames: int, limit_bytes: int
) -> dict[str, Any]:
    """Linear projection of the two probes' allocated-memory slope to `to_frames`.

    Peak is projected as the last probe's `max_memory_allocated` plus the slope times the
    remaining frames: the per-frame memory bank grows linearly, the weights and the
    encoder's transient peak do not.
    """
    if len(probes) < 2:
        raise ValueError("VRAM extrapolation needs two probes")
    ordered = sorted(probes, key=lambda probe: int(probe["frames_processed"]))
    first, last = ordered[0], ordered[-1]
    span = int(last["frames_processed"]) - int(first["frames_processed"])
    if span <= 0:
        raise ValueError("VRAM probes must be at distinct frame counts")
    slope = (int(last["memory_allocated_bytes"]) - int(first["memory_allocated_bytes"])) / span
    remaining = max(0, int(to_frames) - int(last["frames_processed"]))
    projected_allocated = int(round(int(last["memory_allocated_bytes"]) + slope * remaining))
    projected_peak = int(
        round(int(last["max_memory_allocated_bytes"]) + max(slope, 0.0) * remaining)
    )
    return {
        "basis": "linear_between_two_probes",
        "from_frames": (int(first["frames_processed"]), int(last["frames_processed"])),
        "slope_bytes_per_frame": float(slope),
        "extrapolate_to_frames": int(to_frames),
        "projected_allocated_bytes": max(0, projected_allocated),
        "projected_peak_bytes": max(0, projected_peak),
        "limit_bytes": int(limit_bytes),
        "within_limit": projected_peak <= int(limit_bytes),
    }


def corrections_by_frame(corrections: list[dict[str, Any]]) -> dict[int, dict[str, dict[str, Any]]]:
    """Index resolved schedule entries as `{frame_index: {target: entry}}`."""
    grouped: dict[int, dict[str, dict[str, Any]]] = {}
    for entry in corrections:
        frame_index = int(entry["frame_index"])
        target = str(entry["target"])
        if frame_index <= 0:
            raise ValueError("later corrections must be at positive frame indices")
        if target in grouped.setdefault(frame_index, {}):
            raise ValueError(f"duplicate correction for {target} at frame {frame_index}")
        grouped[frame_index][target] = entry
    return grouped
