"""Display-only image enhancement for the calibration canvas.

Every operator here exists to help a human read a monochrome egocentric frame. None
of it is ever sent to the SAM3 decoder, written to an extracted frame, mask, overlay,
or review artifact, hashed into provenance, or persisted into a calibration manifest,
proposal, or correction schedule. ``render_view`` copies its input and returns new
arrays; the caller keeps decoding from the original source pixels.

Edge and corner operators call VIGRA (https://ukoethe.github.io/vigra/) directly. VIGRA
is not on PyPI and cannot come from ``uv.lock``: it is built from source and installed
into the project virtual environment (see ``docs/vigra-build.md``). When it is absent
those operators are reported as unavailable instead of being silently approximated, so
a reader can never mistake a substitute for VIGRA's own result. Brightness, contrast,
and adaptive threshold are plain NumPy because VIGRA does not provide them.
"""

from __future__ import annotations

import math
import os
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

Stage = Literal["tone", "base", "edge", "corner"]
STAGE_ORDER: tuple[Stage, ...] = ("tone", "base", "edge", "corner")
STAGE_LABELS: dict[str, str] = {
    "tone": "1 · Tone",
    "base": "2 · Base image",
    "edge": "3 · Edge detectors",
    "corner": "4 · Corner and junction detectors",
}
MAXIMUM_CORNER_MARKERS = 400
OPERATOR_CACHE_ENTRIES = 64
# Working resolutions offered for interactive tuning. Every divisor above 1 runs the
# real VIGRA operator on a smaller image, which is a genuine loss of detail, so the UI
# names the working resolution instead of quietly shrinking behind a reader's back.
RESOLUTION_DIVISORS: tuple[int, ...] = (1, 2, 3, 4)
RESOLUTION_LABELS: dict[int, str] = {
    1: "Full resolution",
    2: "Half (faster, coarser)",
    3: "Third (faster, coarser)",
    4: "Quarter (fastest, coarsest)",
}

try:  # pragma: no cover - exercised by whichever environment runs the tests
    import vigra as _vigra

    VIGRA_IMPORT_ERROR: str | None = None
except Exception as error:  # pragma: no cover - import failure path
    _vigra = None
    VIGRA_IMPORT_ERROR = f"{type(error).__name__}: {error}"


def vigra_version() -> str | None:
    return str(_vigra.version) if _vigra is not None else None


def vigra_unavailable_reason() -> str | None:
    """Explain, once, why VIGRA-backed operators cannot run in this environment."""
    if _vigra is not None:
        return None
    return (
        f"VIGRA is not importable ({VIGRA_IMPORT_ERROR}); VIGRA-backed aids are "
        "disabled. Build it with docs/vigra-build.md."
    )


class ViewFilterError(ValueError):
    """A view-aid request that cannot be served, e.g. a VIGRA operator without VIGRA."""


@dataclass(frozen=True)
class ViewFilterParameter:
    id: str
    label: str
    minimum: float
    maximum: float
    step: float
    default: float

    def clamp(self, value: object) -> float:
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return self.default
        if not math.isfinite(number):
            return self.default
        steps = round((number - self.minimum) / self.step)
        snapped = self.minimum + steps * self.step
        return round(min(self.maximum, max(self.minimum, snapped)), 6)

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "step": self.step,
            "default": self.default,
        }


@dataclass(frozen=True)
class ViewFilterChoice:
    id: str
    label: str
    options: tuple[str, ...]
    default: str

    def clamp(self, value: object) -> str:
        return value if isinstance(value, str) and value in self.options else self.default

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "options": list(self.options),
            "default": self.default,
        }


@dataclass(frozen=True)
class ViewFilterOperator:
    id: str
    stage: Stage
    label: str
    backend: Literal["vigra", "vigra+numpy", "numpy"]
    provenance: str
    color: str | None = None
    parameters: tuple[ViewFilterParameter, ...] = ()
    choices: tuple[ViewFilterChoice, ...] = ()

    @property
    def requires_vigra(self) -> bool:
        return self.backend in ("vigra", "vigra+numpy")

    def defaults(self) -> dict[str, Any]:
        values: dict[str, Any] = {"enabled": False}
        for parameter in self.parameters:
            values[parameter.id] = parameter.default
        for choice in self.choices:
            values[choice.id] = choice.default
        return values

    def normalize(self, raw: object) -> dict[str, Any]:
        given = raw if isinstance(raw, Mapping) else {}
        values: dict[str, Any] = {"enabled": given.get("enabled") is True}
        for parameter in self.parameters:
            values[parameter.id] = parameter.clamp(given.get(parameter.id, parameter.default))
        for choice in self.choices:
            values[choice.id] = choice.clamp(given.get(choice.id, choice.default))
        return values

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "stage": self.stage,
            "label": self.label,
            "backend": self.backend,
            "provenance": self.provenance,
            "requires_vigra": self.requires_vigra,
            "available": not self.requires_vigra or _vigra is not None,
            "color": self.color,
            "parameters": [parameter.describe() for parameter in self.parameters],
            "choices": [choice.describe() for choice in self.choices],
        }


def _scale(default: float = 1.2) -> ViewFilterParameter:
    return ViewFilterParameter("scale", "Scale σ", 0.5, 5.0, 0.1, default)


def _relative_threshold(default: float = 0.1) -> ViewFilterParameter:
    return ViewFilterParameter("threshold", "Relative threshold", 0.01, 0.9, 0.01, default)


# Fixed composition order. Stages always run tone -> base -> edge -> corner, and
# operators inside a stage always run in this listed order, so toggling controls in any
# sequence produces the same image.
VIEW_FILTER_PIPELINE: tuple[ViewFilterOperator, ...] = (
    ViewFilterOperator(
        id="brightness",
        stage="tone",
        label="Brightness",
        backend="numpy",
        provenance="Not a VIGRA operator; additive luminance offset.",
        parameters=(ViewFilterParameter("amount", "Amount", -100, 100, 1, 0),),
    ),
    ViewFilterOperator(
        id="contrast",
        stage="tone",
        label="Contrast",
        backend="numpy",
        provenance="Not a VIGRA operator; gain about mid-grey.",
        parameters=(ViewFilterParameter("amount", "Amount", -100, 100, 1, 0),),
    ),
    ViewFilterOperator(
        id="adaptive_threshold",
        stage="base",
        label="Adaptive threshold",
        backend="numpy",
        provenance=(
            "Not a VIGRA operator; NumPy local-background threshold "
            "(mean = integral image, gaussian = vigra.filters.gaussianSmoothing "
            "when available)."
        ),
        parameters=(
            ViewFilterParameter("block_size", "Block size (odd)", 3, 99, 2, 25),
            ViewFilterParameter("constant", "Constant C", -40, 40, 1, 7),
        ),
        choices=(ViewFilterChoice("method", "Method", ("mean", "gaussian"), "mean"),),
    ),
    ViewFilterOperator(
        id="canny",
        stage="edge",
        label="Canny edges",
        backend="vigra",
        color="#fb923c",
        provenance="vigra.analysis.cannyEdgeImageWithThinning / cannyEdgeImage",
        parameters=(
            _scale(1.2),
            ViewFilterParameter("threshold", "Gradient threshold", 0.5, 50.0, 0.5, 4.0),
        ),
        choices=(ViewFilterChoice("method", "Variant", ("thinned", "plain"), "thinned"),),
    ),
    ViewFilterOperator(
        id="zero_crossings",
        stage="edge",
        label="Zero crossings (LoG)",
        backend="vigra+numpy",
        color="#38bdf8",
        provenance=(
            "vigra.filters.laplacianOfGaussian; crossings marked in NumPy at whole "
            "pixels, not VIGRA's sub-pixel ones"
        ),
        parameters=(
            _scale(1.6),
            ViewFilterParameter("threshold", "Jump threshold", 0.0, 40.0, 0.5, 2.0),
        ),
    ),
    ViewFilterOperator(
        id="shen_castan",
        stage="edge",
        label="Shen–Castan (ISEF)",
        backend="vigra",
        color="#c084fc",
        provenance="vigra.analysis.shenCastanEdgeImage",
        parameters=(
            _scale(1.0),
            ViewFilterParameter("threshold", "Gradient threshold", 0.5, 50.0, 0.5, 4.0),
        ),
    ),
    ViewFilterOperator(
        id="boundary_tensor",
        stage="edge",
        label="Boundary tensor energy",
        backend="vigra",
        color="#facc15",
        provenance="vigra.filters.boundaryTensor2D trace, thresholded on its maximum",
        parameters=(_scale(1.5), _relative_threshold(0.1)),
    ),
    ViewFilterOperator(
        id="corner_response",
        stage="corner",
        label="Corner response function (Harris)",
        backend="vigra",
        color="#f43f5e",
        provenance="vigra.analysis.cornernessHarris",
        parameters=(_scale(1.2), _relative_threshold(0.1)),
    ),
    ViewFilterOperator(
        id="beaudet",
        stage="corner",
        label="Beaudet",
        backend="vigra",
        color="#34d399",
        provenance="vigra.analysis.cornernessBeaudet, markers on |response|",
        parameters=(_scale(1.4), _relative_threshold(0.1)),
    ),
    ViewFilterOperator(
        id="rohr",
        stage="corner",
        label="Rohr",
        backend="vigra",
        color="#60a5fa",
        provenance="vigra.analysis.cornernessRohr",
        parameters=(_scale(1.2), _relative_threshold(0.1)),
    ),
    ViewFilterOperator(
        id="foerstner",
        stage="corner",
        label="Förstner",
        backend="vigra",
        color="#fbbf24",
        provenance="vigra.analysis.cornernessFoerstner",
        parameters=(_scale(1.2), _relative_threshold(0.1)),
    ),
    ViewFilterOperator(
        id="tensor_junction",
        stage="corner",
        label="Tensor corner/junction",
        backend="vigra",
        color="#f472b6",
        provenance="vigra.analysis.cornernessBoundaryTensor",
        parameters=(_scale(1.5), _relative_threshold(0.1)),
    ),
)

OPERATORS_BY_ID: dict[str, ViewFilterOperator] = {
    operator.id: operator for operator in VIEW_FILTER_PIPELINE
}


def default_settings() -> dict[str, dict[str, Any]]:
    return {operator.id: operator.defaults() for operator in VIEW_FILTER_PIPELINE}


def normalize_settings(raw: object) -> dict[str, dict[str, Any]]:
    """Clamp, snap, and order every control so unknown input cannot reach an operator."""
    given = raw if isinstance(raw, Mapping) else {}
    return {
        operator.id: operator.normalize(given.get(operator.id)) for operator in VIEW_FILTER_PIPELINE
    }


def enabled_operators(settings: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the enabled operator ids in fixed pipeline order, never request order."""
    normalized = normalize_settings(settings)
    return tuple(
        operator.id for operator in VIEW_FILTER_PIPELINE if normalized[operator.id]["enabled"]
    )


def is_identity(settings: Mapping[str, Any]) -> bool:
    return not enabled_operators(settings)


def settings_key(settings: Mapping[str, Any]) -> str:
    """Return a stable cache key; equal settings always produce the same string."""
    normalized = normalize_settings(settings)
    parts = []
    for operator in VIEW_FILTER_PIPELINE:
        values = normalized[operator.id]
        rendered = ",".join(f"{key}={values[key]}" for key in sorted(values))
        parts.append(f"{operator.id}:{rendered}")
    return "|".join(parts)


def normalize_resolution_divisor(value: object) -> int:
    """Snap a requested working resolution onto an offered one, defaulting to full."""
    try:
        divisor = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 1
    return divisor if divisor in RESOLUTION_DIVISORS else 1


def describe_pipeline() -> dict[str, Any]:
    return {
        "vigra_available": _vigra is not None,
        "vigra_version": vigra_version(),
        "unavailable_reason": vigra_unavailable_reason(),
        "stage_order": list(STAGE_ORDER),
        "stage_labels": dict(STAGE_LABELS),
        "maximum_corner_markers": MAXIMUM_CORNER_MARKERS,
        "resolution_divisors": [
            {"divisor": divisor, "label": RESOLUTION_LABELS[divisor]}
            for divisor in RESOLUTION_DIVISORS
        ],
        "operators": [operator.describe() for operator in VIEW_FILTER_PIPELINE],
    }


def _require_vigra(operator: ViewFilterOperator) -> Any:
    if _vigra is None:
        raise ViewFilterError(f"{operator.label} needs VIGRA. {vigra_unavailable_reason()}")
    return _vigra


def _luminance(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float32)
    channels = image[:, :, :3].astype(np.float32)
    return (
        0.299 * channels[:, :, 0] + 0.587 * channels[:, :, 1] + 0.114 * channels[:, :, 2]
    ).astype(np.float32)


def _numpy_gaussian(gray: np.ndarray, sigma: float) -> np.ndarray:
    radius = max(1, int(math.ceil(sigma * 3)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(offsets**2) / (2 * sigma * sigma))
    kernel /= kernel.sum()
    padded = np.pad(gray, ((0, 0), (radius, radius)), mode="reflect")
    rows = sum(
        kernel[index] * padded[:, index : index + gray.shape[1]] for index in range(kernel.size)
    )
    padded = np.pad(rows, ((radius, radius), (0, 0)), mode="reflect")
    return sum(
        kernel[index] * padded[index : index + gray.shape[0], :] for index in range(kernel.size)
    ).astype(np.float32)


def _gaussian(gray: np.ndarray, sigma: float) -> np.ndarray:
    if _vigra is None:
        return _numpy_gaussian(gray, sigma)
    tagged = _vigra.taggedView(np.ascontiguousarray(gray, dtype=np.float32), "yx")
    return np.asarray(_vigra.filters.gaussianSmoothing(tagged, float(sigma)), dtype=np.float32)


def _box_mean(gray: np.ndarray, block_size: int) -> np.ndarray:
    radius = (block_size - 1) // 2
    sums = np.zeros((gray.shape[0] + 1, gray.shape[1] + 1), dtype=np.float64)
    sums[1:, 1:] = gray.cumsum(axis=0).cumsum(axis=1)
    height, width = gray.shape
    rows = np.arange(height)
    columns = np.arange(width)
    top = np.clip(rows - radius, 0, height)
    bottom = np.clip(rows + radius + 1, 0, height)
    left = np.clip(columns - radius, 0, width)
    right = np.clip(columns + radius + 1, 0, width)
    total = (
        sums[np.ix_(bottom, right)]
        - sums[np.ix_(top, right)]
        - sums[np.ix_(bottom, left)]
        + sums[np.ix_(top, left)]
    )
    area = np.outer(bottom - top, right - left).astype(np.float64)
    return (total / np.maximum(area, 1.0)).astype(np.float32)


def _adaptive_threshold(gray: np.ndarray, options: Mapping[str, Any]) -> np.ndarray:
    block_size = int(options["block_size"])
    if block_size % 2 == 0:
        block_size += 1
    background = (
        _gaussian(gray, max(0.5, block_size / 6.0))
        if options["method"] == "gaussian"
        else _box_mean(gray, block_size)
    )
    return np.where(gray > background - float(options["constant"]), 255.0, 0.0).astype(np.float32)


def _tagged(gray: np.ndarray) -> Any:
    assert _vigra is not None
    return _vigra.taggedView(np.ascontiguousarray(gray, dtype=np.float32), "yx")


def _edge_mask(
    operator: ViewFilterOperator, gray: np.ndarray, options: Mapping[str, Any]
) -> np.ndarray:
    vigra = _require_vigra(operator)
    tagged = _tagged(gray)
    scale = float(options["scale"])
    if operator.id == "canny":
        threshold = float(options["threshold"])
        edges = (
            vigra.analysis.cannyEdgeImageWithThinning(tagged, scale, threshold, 1)
            if options["method"] == "thinned"
            else vigra.analysis.cannyEdgeImage(tagged, scale, threshold, 1)
        )
        return np.asarray(edges) > 0
    if operator.id == "shen_castan":
        edges = vigra.analysis.shenCastanEdgeImage(tagged, scale, float(options["threshold"]), 1)
        return np.asarray(edges) > 0
    if operator.id == "zero_crossings":
        laplacian = np.asarray(vigra.filters.laplacianOfGaussian(tagged, scale), dtype=np.float32)
        jump = float(options["threshold"])
        mask = np.zeros(laplacian.shape, dtype=bool)
        right = laplacian[:, :-1] * laplacian[:, 1:] < 0
        right &= np.abs(laplacian[:, :-1] - laplacian[:, 1:]) >= jump
        mask[:, :-1] |= right
        down = laplacian[:-1, :] * laplacian[1:, :] < 0
        down &= np.abs(laplacian[:-1, :] - laplacian[1:, :]) >= jump
        mask[:-1, :] |= down
        return mask
    tensor = np.asarray(vigra.filters.boundaryTensor2D(tagged, scale), dtype=np.float32)
    trace = tensor[:, :, 0] + tensor[:, :, 2]
    peak = float(np.max(trace)) if trace.size else 0.0
    if peak <= 0:
        return np.zeros(trace.shape, dtype=bool)
    return trace >= peak * float(options["threshold"])


_CORNERNESS = {
    "corner_response": "cornernessHarris",
    "beaudet": "cornernessBeaudet",
    "rohr": "cornernessRohr",
    "foerstner": "cornernessFoerstner",
    "tensor_junction": "cornernessBoundaryTensor",
}


def _corner_points(
    operator: ViewFilterOperator, gray: np.ndarray, options: Mapping[str, Any]
) -> list[dict[str, float]]:
    vigra = _require_vigra(operator)
    function = getattr(vigra.analysis, _CORNERNESS[operator.id])
    response = np.asarray(function(_tagged(gray), float(options["scale"])), dtype=np.float32)
    if operator.id == "beaudet":
        response = np.abs(response)
    peak = float(np.max(response)) if response.size else 0.0
    if peak <= 0:
        return []
    neighbourhood = np.full(response.shape, -np.inf, dtype=np.float32)
    for row in (-1, 0, 1):
        for column in (-1, 0, 1):
            if row == 0 and column == 0:
                continue
            shifted = np.roll(np.roll(response, row, axis=0), column, axis=1)
            neighbourhood = np.maximum(neighbourhood, shifted)
    interior = np.zeros(response.shape, dtype=bool)
    interior[1:-1, 1:-1] = True
    selected = interior & (response >= peak * float(options["threshold"]))
    selected &= response >= neighbourhood
    rows, columns = np.nonzero(selected)
    scores = response[rows, columns] / peak
    order = np.argsort(-scores)[:MAXIMUM_CORNER_MARKERS]
    return [
        {
            "x": float(columns[index]) + 0.5,
            "y": float(rows[index]) + 0.5,
            "score": round(float(scores[index]), 4),
        }
        for index in order
    ]


def _rgb(color: str) -> tuple[int, int, int]:
    return (int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16))


def _operator_key(operator: ViewFilterOperator, options: Mapping[str, Any]) -> str:
    return f"{operator.id}:" + ",".join(
        f"{name}={options[name]}" for name in sorted(options) if name != "enabled"
    )


class ViewOperatorCache:
    """Memoize one operator's own result per frame, tone state, and parameter set.

    The previous cache was keyed on the whole enabled stack, so flipping one checkbox
    recomputed all twelve operators. Keying per operator means the eleven a reader did
    not touch are reused untouched. Every entry is a display-only derived array; none
    of it is persisted, hashed, or forwarded to the decoder. The entry bound keeps the
    worst case — a long drag on an edge slider, whose masks are one bool per pixel —
    to a few tens of megabytes for a 954x720 frame.
    """

    def __init__(self, maximum: int = OPERATOR_CACHE_ENTRIES) -> None:
        self._maximum = maximum
        self._entries: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def fetch(self, key: str, compute: Callable[[], Any]) -> Any:
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                self.hits += 1
                return self._entries[key]
            self.misses += 1
        value = compute()
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self._maximum:
                self._entries.popitem(last=False)
        return value

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


def _memoize(cache: ViewOperatorCache | None, key: str, compute: Callable[[], Any]) -> Any:
    return compute() if cache is None else cache.fetch(key, compute)


_POOL_LOCK = threading.Lock()
_POOL: ThreadPoolExecutor | None = None


def _pool() -> ThreadPoolExecutor:
    """A small shared pool; VIGRA drops the GIL, so its operators genuinely overlap."""
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            workers = min(8, max(2, (os.cpu_count() or 2)))
            _POOL = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="view-aid")
        return _POOL


def _evaluate(jobs: list[tuple[str, Callable[[], Any]]]) -> dict[str, Any]:
    """Run independent operators concurrently, then hand results back by id.

    Composition still happens in fixed pipeline order, so the image does not depend on
    which operator happened to finish first.
    """
    if len(jobs) <= 1:
        return {name: compute() for name, compute in jobs}
    pool = _pool()
    futures = [(name, pool.submit(compute)) for name, compute in jobs]
    return {name: future.result() for name, future in futures}


@dataclass(frozen=True)
class ViewRender:
    pixels: np.ndarray
    applied: tuple[str, ...] = ()
    corners: tuple[dict[str, Any], ...] = ()
    backends: dict[str, str] = field(default_factory=dict)


def render_view(
    image: np.ndarray,
    settings: Mapping[str, Any],
    *,
    cache: ViewOperatorCache | None = None,
    token: str = "",
) -> ViewRender:
    """Build a display-only RGB image and corner markers from a source frame.

    ``image`` is never mutated and is never forwarded to the decoder or to disk. When a
    ``cache`` is supplied, ``token`` must identify the exact pixels in ``image`` so that
    per-operator results are only reused for the frame they were computed from.
    """
    normalized = normalize_settings(settings)
    gray = _luminance(np.asarray(image))
    applied: list[str] = []
    backends: dict[str, str] = {}

    tone = gray
    tone_parts = []
    if normalized["brightness"]["enabled"]:
        tone = np.clip(tone + float(normalized["brightness"]["amount"]), 0.0, 255.0)
        applied.append("brightness")
        backends["brightness"] = "numpy"
        tone_parts.append(f"brightness={normalized['brightness']['amount']}")
    if normalized["contrast"]["enabled"]:
        amount = float(normalized["contrast"]["amount"]) * 2.55
        factor = (259.0 * (amount + 255.0)) / (255.0 * (259.0 - amount))
        tone = np.clip(factor * (tone - 128.0) + 128.0, 0.0, 255.0)
        applied.append("contrast")
        backends["contrast"] = "numpy"
        tone_parts.append(f"contrast={normalized['contrast']['amount']}")
    tone = tone.astype(np.float32)
    # Tone feeds every later stage, so it belongs in the key of everything downstream.
    stem = f"{token}|{','.join(tone_parts)}"

    base = tone
    if normalized["adaptive_threshold"]["enabled"]:
        options = normalized["adaptive_threshold"]
        base = _memoize(
            cache,
            f"{stem}|{_operator_key(OPERATORS_BY_ID['adaptive_threshold'], options)}",
            lambda: _adaptive_threshold(tone, options),
        )
        applied.append("adaptive_threshold")
        backends["adaptive_threshold"] = "numpy"

    jobs: list[tuple[str, Callable[[], Any]]] = []
    for operator in VIEW_FILTER_PIPELINE:
        if operator.stage not in ("edge", "corner") or not normalized[operator.id]["enabled"]:
            continue
        options = normalized[operator.id]
        key = f"{stem}|{_operator_key(operator, options)}"
        compute = _edge_mask if operator.stage == "edge" else _corner_points
        jobs.append(
            (
                operator.id,
                lambda o=operator, p=options, k=key, c=compute: _memoize(
                    cache, k, lambda: c(o, tone, p)
                ),
            )
        )
    results = _evaluate(jobs)

    pixels = np.repeat(np.clip(base, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)
    for operator in VIEW_FILTER_PIPELINE:
        if operator.stage != "edge" or operator.id not in results:
            continue
        assert operator.color is not None
        pixels[results[operator.id]] = _rgb(operator.color)
        applied.append(operator.id)
        backends[operator.id] = operator.backend

    corners: list[dict[str, Any]] = []
    for operator in VIEW_FILTER_PIPELINE:
        if operator.stage != "corner" or operator.id not in results:
            continue
        corners.append({"id": operator.id, "color": operator.color, "points": results[operator.id]})
        applied.append(operator.id)
        backends[operator.id] = operator.backend

    return ViewRender(
        pixels=pixels,
        applied=tuple(applied),
        corners=tuple(corners),
        backends=backends,
    )


def unavailable_operator_ids(settings: Mapping[str, Any]) -> tuple[str, ...]:
    """Return enabled operator ids that cannot run because VIGRA is missing."""
    if _vigra is not None:
        return ()
    return tuple(
        operator_id
        for operator_id in enabled_operators(settings)
        if OPERATORS_BY_ID[operator_id].requires_vigra
    )
