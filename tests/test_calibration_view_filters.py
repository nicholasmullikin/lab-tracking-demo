"""Display-only view aids: parameter handling, ordering, and artifact immutability."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest
from conftest import require_executable
from conftest import serve as shared_serve
from PIL import Image
from test_muggled_calibration_web import make_workspace, post_json

from battle import calibration_view_filters as view_filters
from battle.calibration_view_filters import (
    OPERATORS_BY_ID,
    VIEW_FILTER_PIPELINE,
    ViewFilterError,
    default_settings,
    describe_pipeline,
    enabled_operators,
    is_identity,
    normalize_settings,
    render_view,
    settings_key,
    unavailable_operator_ids,
)

EXPECTED_ORDER = (
    "brightness",
    "contrast",
    "adaptive_threshold",
    "canny",
    "zero_crossings",
    "shen_castan",
    "boundary_tensor",
    "corner_response",
    "beaudet",
    "rohr",
    "foerstner",
    "tensor_junction",
)


def all_enabled() -> dict[str, dict[str, object]]:
    settings = default_settings()
    for entry in settings.values():
        entry["enabled"] = True
    return settings


def source_frame(seed: int = 11, size: tuple[int, int] = (96, 128)) -> np.ndarray:
    """A deterministic frame with structure, so detectors produce non-empty output."""
    height, width = size
    rows, columns = np.mgrid[0:height, 0:width]
    pattern = 120 + 90 * np.sin(rows / 7.0) * np.cos(columns / 5.0)
    pattern[20:50, 30:70] = 240
    noise = np.random.default_rng(seed).normal(0, 6, size=(height, width))
    grey = np.clip(pattern + noise, 0, 255).astype(np.uint8)
    return np.repeat(grey[:, :, None], 3, axis=2)


def write_fixture_frame(workspace, frame: np.ndarray) -> Path:
    path = workspace.manifest_path.parent / "results/frames/fixture.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame, mode="RGB").save(path, format="JPEG", quality=95)
    return path


def directory_fingerprint(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def serve(workspace):
    return shared_serve(workspace)


def test_pipeline_declares_every_requested_control_in_a_fixed_order() -> None:
    assert tuple(operator.id for operator in VIEW_FILTER_PIPELINE) == EXPECTED_ORDER
    assert [operator.stage for operator in VIEW_FILTER_PIPELINE] == (
        ["tone"] * 2 + ["base"] + ["edge"] * 4 + ["corner"] * 5
    )
    threshold = OPERATORS_BY_ID["adaptive_threshold"]
    assert [parameter.id for parameter in threshold.parameters] == ["block_size", "constant"]
    assert threshold.choices[0].options == ("mean", "gaussian")
    assert OPERATORS_BY_ID["canny"].choices[0].options == ("thinned", "plain")


def test_settings_are_clamped_snapped_and_defaulted() -> None:
    normalized = normalize_settings(
        {
            "brightness": {"enabled": True, "amount": 5000},
            "contrast": {"enabled": "yes", "amount": "not a number"},
            "adaptive_threshold": {"block_size": 26.4, "constant": -900, "method": "otsu"},
            "canny": {"scale": -3, "method": "thinned"},
            "unknown_operator": {"enabled": True},
        }
    )
    assert normalized["brightness"] == {"enabled": True, "amount": 100}
    # Only a real boolean True enables an operator; a truthy string must not.
    assert normalized["contrast"] == {"enabled": False, "amount": 0}
    assert normalized["adaptive_threshold"]["block_size"] == 27
    assert normalized["adaptive_threshold"]["block_size"] % 2 == 1
    assert normalized["adaptive_threshold"]["constant"] == -40
    assert normalized["adaptive_threshold"]["method"] == "mean"
    assert normalized["canny"]["scale"] == 0.5
    assert "unknown_operator" not in normalized
    assert set(normalized) == set(EXPECTED_ORDER)


def test_block_size_slider_can_only_produce_odd_values() -> None:
    parameter = next(
        item for item in OPERATORS_BY_ID["adaptive_threshold"].parameters if item.id == "block_size"
    )
    produced = {parameter.clamp(value) for value in range(0, 120)}
    assert produced == {float(value) for value in range(3, 100, 2)}


def test_enabled_operators_follow_pipeline_order_not_request_order() -> None:
    scrambled = {
        "tensor_junction": {"enabled": True},
        "brightness": {"enabled": True},
        "canny": {"enabled": True},
        "adaptive_threshold": {"enabled": True},
    }
    assert enabled_operators(scrambled) == (
        "brightness",
        "adaptive_threshold",
        "canny",
        "tensor_junction",
    )
    assert is_identity({}) is True
    assert is_identity(scrambled) is False


def test_settings_key_is_stable_and_independent_of_key_order() -> None:
    first = {"canny": {"enabled": True, "scale": 1.2, "threshold": 4.0, "method": "thinned"}}
    second = {"canny": {"method": "thinned", "threshold": 4.0, "scale": 1.2, "enabled": True}}
    assert settings_key(first) == settings_key(second)
    assert settings_key(first) != settings_key({"canny": {"enabled": True, "scale": 2.0}})
    # Out-of-range input normalizes onto the same key as its clamped equivalent.
    assert settings_key({"canny": {"enabled": True, "scale": 99}}) == settings_key(
        {"canny": {"enabled": True, "scale": 5.0}}
    )


def test_tone_and_threshold_operators_need_no_vigra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(view_filters, "_vigra", None)
    settings = default_settings()
    settings["brightness"].update({"enabled": True, "amount": 40})
    settings["contrast"].update({"enabled": True, "amount": 30})
    settings["adaptive_threshold"].update({"enabled": True, "method": "mean"})
    rendered = render_view(source_frame(), settings)
    assert rendered.applied == ("brightness", "contrast", "adaptive_threshold")
    assert set(np.unique(rendered.pixels)) <= {0, 255}
    assert unavailable_operator_ids(settings) == ()


def test_missing_vigra_reports_a_clear_error_instead_of_a_silent_substitute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(view_filters, "_vigra", None)
    monkeypatch.setattr(view_filters, "VIGRA_IMPORT_ERROR", "ModuleNotFoundError: vigra")
    settings = default_settings()
    settings["canny"]["enabled"] = True
    settings["corner_response"]["enabled"] = True
    assert unavailable_operator_ids(settings) == ("canny", "corner_response")
    description = describe_pipeline()
    assert description["vigra_available"] is False
    assert "docs/archive/vigra-build.md" in description["unavailable_reason"]
    assert [item["id"] for item in description["operators"] if not item["available"]] == [
        "canny",
        "zero_crossings",
        "shen_castan",
        "boundary_tensor",
        "corner_response",
        "beaudet",
        "rohr",
        "foerstner",
        "tensor_junction",
    ]
    with pytest.raises(ViewFilterError, match="needs VIGRA"):
        render_view(source_frame(), settings)


def test_brightness_and_contrast_compose_in_the_documented_order() -> None:
    levels = np.array([[20, 60, 100, 140, 180]], dtype=np.uint8)
    frame = np.repeat(levels[:, :, None], 3, axis=2)
    settings = default_settings()
    settings["brightness"].update({"enabled": True, "amount": 50})
    settings["contrast"].update({"enabled": True, "amount": 60})
    produced = render_view(frame, settings).pixels[:, :, 0].astype(np.float64)

    amount = 60 * 2.55
    factor = (259.0 * (amount + 255.0)) / (255.0 * (259.0 - amount))
    brightened = np.clip(levels.astype(np.float64) + 50, 0, 255)
    brightness_then_contrast = np.clip(factor * (brightened - 128.0) + 128.0, 0, 255)
    contrast_then_brightness = np.clip(
        np.clip(factor * (levels.astype(np.float64) - 128.0) + 128.0, 0, 255) + 50, 0, 255
    )

    assert np.array_equal(produced, brightness_then_contrast.astype(np.uint8))
    assert not np.array_equal(produced, contrast_then_brightness.astype(np.uint8))


def test_render_view_never_mutates_the_source_frame() -> None:
    pytest.importorskip("vigra")
    frame = source_frame()
    before = bytes(frame.tobytes())
    digest = hashlib.sha256(before).hexdigest()
    rendered = render_view(frame, all_enabled())
    assert rendered.applied == EXPECTED_ORDER
    assert frame.tobytes() == before
    assert hashlib.sha256(frame.tobytes()).hexdigest() == digest
    assert rendered.pixels is not frame
    assert not np.shares_memory(rendered.pixels, frame)


def test_read_only_source_arrays_are_accepted() -> None:
    pytest.importorskip("vigra")
    frame = source_frame()
    frame.flags.writeable = False
    assert render_view(frame, all_enabled()).pixels.shape == frame.shape


def test_vigra_backed_operators_call_vigra_itself() -> None:
    vigra = pytest.importorskip("vigra")
    frame = source_frame()
    grey = (0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]).astype(
        np.float32
    )
    tagged = vigra.taggedView(np.ascontiguousarray(grey), "yx")

    settings = default_settings()
    settings["canny"].update({"enabled": True, "scale": 1.2, "threshold": 4.0})
    rendered = render_view(frame, settings)
    expected = np.asarray(vigra.analysis.cannyEdgeImageWithThinning(tagged, 1.2, 4.0, 1)) > 0
    assert expected.any()
    painted = np.all(rendered.pixels == np.array([0xFB, 0x92, 0x3C], dtype=np.uint8), axis=2)
    assert np.array_equal(painted, expected)

    settings = default_settings()
    settings["corner_response"].update({"enabled": True, "scale": 1.2, "threshold": 0.1})
    corners = render_view(frame, settings).corners[0]
    response = np.asarray(vigra.analysis.cornernessHarris(tagged, 1.2))
    peak = float(response.max())
    for point in corners["points"]:
        assert response[int(point["y"]), int(point["x"])] >= peak * 0.1

    assert describe_pipeline()["vigra_version"] == str(vigra.version)
    backends = {operator.id: operator.backend for operator in VIEW_FILTER_PIPELINE}
    assert backends == {
        "brightness": "numpy",
        "contrast": "numpy",
        "adaptive_threshold": "numpy",
        "canny": "vigra",
        "zero_crossings": "vigra+numpy",
        "shen_castan": "vigra",
        "boundary_tensor": "vigra",
        "corner_response": "vigra",
        "beaudet": "vigra",
        "rohr": "vigra",
        "foerstner": "vigra",
        "tensor_junction": "vigra",
    }


def test_corner_markers_are_bounded_and_inside_the_frame() -> None:
    pytest.importorskip("vigra")
    frame = source_frame()
    settings = default_settings()
    for operator_id in ("corner_response", "beaudet", "rohr", "foerstner", "tensor_junction"):
        settings[operator_id].update({"enabled": True, "threshold": 0.01})
    rendered = render_view(frame, settings)
    assert [group["id"] for group in rendered.corners] == [
        "corner_response",
        "beaudet",
        "rohr",
        "foerstner",
        "tensor_junction",
    ]
    for group in rendered.corners:
        assert len(group["points"]) <= view_filters.MAXIMUM_CORNER_MARKERS
        for point in group["points"]:
            assert 0 <= point["x"] <= frame.shape[1]
            assert 0 <= point["y"] <= frame.shape[0]
            assert 0.0 <= point["score"] <= 1.0


def test_view_filters_endpoint_describes_every_control(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    server, base = serve(workspace)
    try:
        payload = json.loads(urllib.request.urlopen(f"{base}/api/view-filters").read())
        assert [item["id"] for item in payload["operators"]] == list(EXPECTED_ORDER)
        assert payload["stage_order"] == ["tone", "base", "edge", "corner"]
        for operator in payload["operators"]:
            assert operator["provenance"]
            assert operator["parameters"]
            if operator["backend"] == "numpy":
                assert "not a vigra operator" in operator["provenance"].lower()
            else:
                assert "vigra." in operator["provenance"]
    finally:
        server.shutdown()
        server.server_close()
        workspace.close()


def test_view_render_endpoint_returns_a_png_and_caches_per_frame_and_parameters(
    tmp_path: Path,
) -> None:
    pytest.importorskip("vigra")
    workspace = make_workspace(tmp_path)
    frame = source_frame()
    write_fixture_frame(workspace, frame)
    server, base = serve(workspace)
    try:
        settings = all_enabled()
        first = post_json(base, "/api/view-filters/render", {"timestamp": 0, "settings": settings})
        assert first["cached"] is False
        assert first["applied"] == list(EXPECTED_ORDER)
        decoded = Image.open(io.BytesIO(base64.b64decode(first["image_png_base64"])))
        assert decoded.format == "PNG"
        assert decoded.size == (frame.shape[1], frame.shape[0])

        again = post_json(base, "/api/view-filters/render", {"timestamp": 0, "settings": settings})
        assert again["cached"] is True
        assert again["image_png_base64"] == first["image_png_base64"]

        changed = dict(settings)
        changed["canny"] = {**settings["canny"], "scale": 3.0}
        third = post_json(base, "/api/view-filters/render", {"timestamp": 0, "settings": changed})
        assert third["cached"] is False
        assert third["image_png_base64"] != first["image_png_base64"]

        with pytest.raises(urllib.error.HTTPError) as empty:
            post_json(base, "/api/view-filters/render", {"timestamp": 0, "settings": {}})
        assert "enable at least one view aid" in empty.value.read().decode()
    finally:
        server.shutdown()
        server.server_close()
        workspace.close()


def test_per_operator_cache_reuses_the_operators_a_reader_did_not_touch() -> None:
    """Toggling one aid must not recompute the other eleven, nor change the result."""
    pytest.importorskip("vigra")
    frame = source_frame()
    settings = all_enabled()
    uncached = render_view(frame, settings)

    cache = view_filters.ViewOperatorCache()
    first = render_view(frame, settings, cache=cache, token="frame-0")
    assert np.array_equal(first.pixels, uncached.pixels)
    assert first.corners == uncached.corners
    assert first.applied == uncached.applied
    assert (cache.hits, cache.misses) == (0, 10)

    repeated = render_view(frame, settings, cache=cache, token="frame-0")
    assert np.array_equal(repeated.pixels, uncached.pixels)
    assert cache.misses == 10

    without_canny = {**settings, "canny": {**settings["canny"], "enabled": False}}
    toggled = render_view(frame, without_canny, cache=cache, token="frame-0")
    assert "canny" not in toggled.applied
    assert cache.misses == 10

    # A different frame shares no results, however identical the settings are.
    render_view(frame, settings, cache=cache, token="frame-1")
    assert cache.misses == 20


def test_cached_operators_are_not_reused_across_different_tone_settings() -> None:
    """Brightness feeds the detectors, so a toned frame cannot reuse an untoned result."""
    pytest.importorskip("vigra")
    frame = source_frame()
    settings = all_enabled()
    cache = view_filters.ViewOperatorCache()
    plain = render_view(frame, settings, cache=cache, token="frame-0")
    brightened = {**settings, "brightness": {**settings["brightness"], "amount": 60}}
    toned = render_view(frame, brightened, cache=cache, token="frame-0")
    assert cache.misses == 20
    assert not np.array_equal(plain.pixels, toned.pixels)
    assert np.array_equal(toned.pixels, render_view(frame, brightened).pixels)


def test_concurrent_operator_evaluation_keeps_the_fixed_composition_order() -> None:
    """Operators run in parallel but are composited in pipeline order, never finish order."""
    pytest.importorskip("vigra")
    frame = source_frame()
    settings = all_enabled()
    expected = render_view(frame, settings)
    for _ in range(4):
        repeated = render_view(frame, settings)
        assert np.array_equal(repeated.pixels, expected.pixels)
        assert repeated.applied == EXPECTED_ORDER
        assert [group["id"] for group in repeated.corners] == [
            group["id"] for group in expected.corners
        ]


def test_reduced_working_resolution_is_named_rather_than_applied_silently(
    tmp_path: Path,
) -> None:
    pytest.importorskip("vigra")
    assert view_filters.RESOLUTION_DIVISORS[0] == 1
    assert [view_filters.normalize_resolution_divisor(value) for value in (2, 4, 7, None, "3")] == [
        2,
        4,
        1,
        1,
        3,
    ]
    described = describe_pipeline()["resolution_divisors"]
    assert [item["divisor"] for item in described] == list(view_filters.RESOLUTION_DIVISORS)
    assert described[0]["label"] == "Full resolution"
    assert all(item["label"] for item in described)

    workspace = make_workspace(tmp_path)
    frame = source_frame()
    write_fixture_frame(workspace, frame)
    server, base = serve(workspace)
    try:
        settings = all_enabled()
        full = post_json(base, "/api/view-filters/render", {"timestamp": 0, "settings": settings})
        assert full["resolution_divisor"] == 1
        assert (full["width"], full["height"]) == (frame.shape[1], frame.shape[0])

        half = post_json(
            base,
            "/api/view-filters/render",
            {"timestamp": 0, "settings": settings, "resolution_divisor": 2},
        )
        assert half["resolution_divisor"] == 2
        assert (half["width"], half["height"]) == (frame.shape[1] // 2, frame.shape[0] // 2)
        decoded = Image.open(io.BytesIO(base64.b64decode(half["image_png_base64"])))
        assert decoded.size == (frame.shape[1] // 2, frame.shape[0] // 2)
        # Markers stay in full-frame coordinates so they land on the pixels they mark.
        for group in half["corners"]:
            for point in group["points"]:
                assert 0 <= point["x"] <= frame.shape[1]
                assert 0 <= point["y"] <= frame.shape[0]

        # An unoffered divisor falls back to full resolution instead of guessing.
        odd = post_json(
            base,
            "/api/view-filters/render",
            {"timestamp": 0, "settings": settings, "resolution_divisor": 9},
        )
        assert odd["resolution_divisor"] == 1
    finally:
        server.shutdown()
        server.server_close()
        workspace.close()


def test_cached_view_renders_follow_a_re_extracted_frame(tmp_path: Path) -> None:
    """Caching must never show yesterday's pixels for a frame that has been rewritten."""
    pytest.importorskip("vigra")
    workspace = make_workspace(tmp_path)
    write_fixture_frame(workspace, source_frame(seed=11))
    server, base = serve(workspace)
    try:
        settings = all_enabled()
        first = post_json(base, "/api/view-filters/render", {"timestamp": 0, "settings": settings})
        write_fixture_frame(workspace, source_frame(seed=29))
        again = post_json(base, "/api/view-filters/render", {"timestamp": 0, "settings": settings})
        assert again["cached"] is False
        assert again["image_png_base64"] != first["image_png_base64"]
    finally:
        server.shutdown()
        server.server_close()
        workspace.close()


def test_view_render_writes_nothing_into_the_run_directory(tmp_path: Path) -> None:
    pytest.importorskip("vigra")
    workspace = make_workspace(tmp_path)
    write_fixture_frame(workspace, source_frame())
    server, base = serve(workspace)
    try:
        before = directory_fingerprint(workspace.manifest_path.parent)
        manifest_before = json.dumps(workspace.manifest.model_dump(mode="json"), sort_keys=True)
        for scale in (1.0, 2.0, 3.0):
            settings = all_enabled()
            settings["canny"]["scale"] = scale
            post_json(base, "/api/view-filters/render", {"timestamp": 0, "settings": settings})
        assert directory_fingerprint(workspace.manifest_path.parent) == before
        assert (
            json.dumps(workspace.manifest.model_dump(mode="json"), sort_keys=True)
            == manifest_before
        )
        # The view settings live only in the browser; nothing names them in the manifest.
        assert not any(
            operator_id in manifest_before for operator_id in ("adaptive_threshold", "foerstner")
        )
        assert "view_aid" not in manifest_before and "view_filter" not in manifest_before
    finally:
        server.shutdown()
        server.server_close()
        workspace.close()


def test_enabled_view_aids_do_not_change_the_decode_payload_or_frame_bytes(
    tmp_path: Path,
) -> None:
    """The decoder must receive byte-identical prompts and pixels either way."""
    pytest.importorskip("vigra")
    prompts = [
        {
            "timestamp": 0,
            "intended_target": target,
            "pixel_box": {"x1": 10 + index, "y1": 20, "x2": 100, "y2": 120},
            "pixel_fg_points": [{"x": 40, "y": 50}],
            "pixel_bg_points": [{"x": 80, "y": 90}],
        }
        for index, target in enumerate(("left_hand", "right_hand"))
    ]

    def decode_with(directory: Path, render_views: bool) -> tuple[object, str]:
        workspace = make_workspace(directory)
        frame_path = write_fixture_frame(workspace, source_frame())
        frame_digest = hashlib.sha256(frame_path.read_bytes()).hexdigest()
        server, base = serve(workspace)
        try:
            if render_views:
                for scale in (1.0, 2.5):
                    settings = all_enabled()
                    settings["canny"]["scale"] = scale
                    post_json(
                        base, "/api/view-filters/render", {"timestamp": 0, "settings": settings}
                    )
            boxes = [workspace.add_or_update_prompt(dict(prompt)) for prompt in prompts]
            job_id = workspace.queue_decode([box["box_id"] for box in boxes])
            workspace.jobs[job_id]["future"].result(timeout=5)
            payload = workspace.decoder.last_batch_payload
            assert hashlib.sha256(frame_path.read_bytes()).hexdigest() == frame_digest
            return payload, frame_digest
        finally:
            server.shutdown()
            server.server_close()
            workspace.close()

    plain_payload, plain_digest = decode_with(tmp_path / "plain", render_views=False)
    filtered_payload, filtered_digest = decode_with(tmp_path / "filtered", render_views=True)

    assert json.dumps(filtered_payload, sort_keys=True) == json.dumps(plain_payload, sort_keys=True)
    assert filtered_digest == plain_digest
    serialized = json.dumps(filtered_payload)
    for operator_id in EXPECTED_ORDER:
        assert operator_id not in serialized
    assert "settings" not in serialized and "view" not in serialized


HEADLESS_VIEW_AID_SCRIPT = r"""
const vm = require("node:vm");
const base = process.argv[1];
const nativeFetch = globalThis.fetch;
const requests = [];

class Element {
  constructor(selector = "") {
    this.selector = selector;
    this.children = [];
    this.classList = {toggle() {}};
    this.dataset = {};
    this.style = {};
    this.value = "";
    this.textContent = "";
    this.checked = false;
    this.disabled = false;
    this.hidden = false;
    this.width = 954;
    this.height = 720;
  }
  addEventListener() {}
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  getBoundingClientRect() { return {left: 0, top: 0, width: 954, height: 720}; }
  setPointerCapture() {}
  getContext() {
    const element = this;
    return {
      arc() {}, beginPath() {}, clearRect() {}, drawImage() {}, fill() {}, fillRect() {},
      fillText() {}, moveTo() {}, lineTo() {},
      getImageData() {
        return {data: new Uint8ClampedArray(element.width * element.height * 4)};
      },
      putImageData() {}, restore() {}, save() {}, scale() {}, setTransform() {},
      stroke() {}, strokeRect() {}, translate() {},
    };
  }
  querySelectorAll() { return []; }
  descendants() {
    return this.children.flatMap((node) => (
      node instanceof Element ? [node, ...node.descendants()] : []
    ));
  }
}

const selectors = [
  "#canvas", "#status", "#labels", "#time-slider", "#timestamp", "#filmstrip",
  "#prompts", "#table", "#diff", "#candidates", "#eligible", "#empty",
  "#target-policy", "#custom-label", "#load-frame", "#delete", "#duplicate",
  "#decode", "#decode-selected", "#finalize-plan", "#prompt-mode", "#clear-points",
  "#point-guidance", "#active-candidate", "#toggle-rejected", "#mask-visible",
  "#mask-opacity", "#mask-opacity-value", "#view-aids-summary", "#view-aids-backend",
  "#view-aids-bypass", "#view-aids-reset", "#view-aid-stages", "#view-aids-timing",
  "#view-aid-resolution",
];
const elements = Object.fromEntries(selectors.map((selector) => [selector, new Element(selector)]));
globalThis.document = {
  activeElement: {tagName: "BODY"},
  addEventListener() {},
  createElement() { return new Element(); },
  createTextNode(value) { return String(value); },
  querySelector(selector) { return elements[selector]; },
};
globalThis.Image = class Image {
  constructor() { this.naturalWidth = 954; this.naturalHeight = 720; }
  set src(value) { this._src = value; setTimeout(() => this.onload && this.onload(), 0); }
  get src() { return this._src; }
};
globalThis.requestAnimationFrame = (callback) => setTimeout(callback, 0);
globalThis.fetch = (path, options) => {
  requests.push({path, body: options?.body ?? null});
  return nativeFetch(base + path, options);
};
const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

(async () => {
  const app = await (await nativeFetch(base + "/static/app.js")).text();
  vm.runInThisContext(app, {filename: "calibration-app.js"});
  await wait(200);

  const stages = elements["#view-aid-stages"];
  const cards = stages.descendants().filter((node) => node.className?.startsWith("view-aid "));
  if (cards.length !== 12) throw new Error(`expected 12 view-aid cards, got ${cards.length}`);
  const sliders = stages.descendants().filter((node) => node.type === "range");
  if (sliders.length < 12) throw new Error(`expected parameter sliders, got ${sliders.length}`);

  globalThis.setViewAid("canny", "enabled", true);
  globalThis.setViewAid("corner_response", "enabled", true);
  await wait(700);
  const renders = requests.filter((item) => item.path === "/api/view-filters/render");
  if (!renders.length) throw new Error("no display-only render was requested");
  const settings = JSON.parse(renders.at(-1).body).settings;
  if (!settings.canny.enabled || !settings.corner_response.enabled) {
    throw new Error("render request did not carry the enabled operators");
  }
  if (elements["#view-aids-summary"].textContent === "off") {
    throw new Error("summary did not report the enabled operators");
  }

  const resolution = elements["#view-aid-resolution"];
  if (!resolution.children.length) throw new Error("no working-resolution choices offered");
  if (resolution.value !== "1") throw new Error(`default resolution was ${resolution.value}`);

  // Dragging a slider must keep the very node under the pointer, not replace the panel.
  const scale = stages.descendants().find(
    (node) => node.type === "range" && Number(node.max) === 5
  );
  if (!scale) throw new Error("no scale slider was built");
  requests.length = 0;
  for (const value of ["1.5", "1.6", "1.7"]) { scale.value = value; scale.oninput(); }
  await wait(500);
  if (!stages.descendants().includes(scale)) {
    throw new Error("the slider being dragged was rebuilt out of the panel");
  }
  const dragRenders = requests.filter((item) => item.path === "/api/view-filters/render");
  if (dragRenders.length !== 1) {
    throw new Error(`slider drag issued ${dragRenders.length} renders, expected 1`);
  }

  // Brightness and contrast alone are pointwise, so nothing should reach the network.
  globalThis.resetViewAids();
  await wait(200);
  requests.length = 0;
  globalThis.setViewAid("brightness", "enabled", true);
  globalThis.setViewAid("brightness", "amount", 30);
  globalThis.setViewAid("contrast", "enabled", true);
  await wait(400);
  const toneRenders = requests.filter((item) => item.path === "/api/view-filters/render");
  if (toneRenders.length) {
    throw new Error(`tone-only aids asked the server ${toneRenders.length} time(s)`);
  }
  const toneReport = elements["#view-aids-timing"].textContent;
  if (!toneReport.includes("in browser")) {
    throw new Error(`tone report did not say where it ran: ${toneReport}`);
  }
  // Adding a detector puts the whole chain back on the server, tone included.
  globalThis.setViewAid("canny", "enabled", true);
  await wait(600);
  const chained = requests.filter((item) => item.path === "/api/view-filters/render");
  if (!chained.length) throw new Error("enabling a detector did not re-render on the server");
  const chainedSettings = JSON.parse(chained.at(-1).body).settings;
  if (!chainedSettings.brightness.enabled || !chainedSettings.canny.enabled) {
    throw new Error("server render lost the tone stage the detector reads");
  }
  globalThis.resetViewAids();
  await wait(200);

  requests.length = 0;
  await globalThis.decode(["p000000-b01"]);
  await wait(150);
  const decodes = requests.filter((item) => item.path === "/api/decode");
  if (decodes.length !== 1) throw new Error(`expected one decode request, got ${decodes.length}`);
  const decodeBody = decodes[0].body;
  if (decodeBody !== JSON.stringify({box_ids: ["p000000-b01"], live_preview: false})) {
    throw new Error(`decode body carried view state: ${decodeBody}`);
  }

  globalThis.resetViewAids();
  await wait(300);
  if (elements["#view-aids-summary"].textContent !== "off") {
    throw new Error("reset view did not clear the enabled operators");
  }
  if (elements["#view-aids-timing"].textContent !== "") {
    throw new Error("reset view did not clear the render report");
  }
})().catch((error) => {
  console.error(error.stack || error);
  process.exitCode = 1;
});
"""


@pytest.mark.slow
def test_headless_ui_renders_view_aids_without_touching_the_decode_request(
    tmp_path: Path,
) -> None:
    pytest.importorskip("vigra")
    node = require_executable("node", "the view-aid UI regression test")
    workspace = make_workspace(tmp_path)
    write_fixture_frame(workspace, source_frame())
    workspace.add_or_update_prompt(
        {
            "timestamp": 0,
            "intended_target": "left_hand",
            "pixel_box": {"x1": 10, "y1": 20, "x2": 100, "y2": 120},
        }
    )
    server, base = serve(workspace)
    try:
        completed = subprocess.run(
            [node, "-e", HEADLESS_VIEW_AID_SCRIPT, base],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        assert completed.returncode == 0, completed.stderr
        payload = workspace.decoder.last_batch_payload
        assert payload is not None
        assert set(payload["prompts"][0]) == {
            "box_id",
            "candidate_id",
            "frame_index",
            "pixel_box",
            "intended_target",
            "boxes",
            "fg_points",
            "bg_points",
        }
    finally:
        server.shutdown()
        server.server_close()
        workspace.close()


def test_calibration_ui_exposes_view_aid_controls_and_shortcuts() -> None:
    static = Path(__file__).parents[1] / "src/battle/static/calibration"
    markup = (static / "index.html").read_text()
    script = (static / "app.js").read_text()

    assert 'id="view-aids"' in markup
    assert 'id="view-aid-stages"' in markup
    assert 'id="view-aids-reset"' in markup
    assert 'id="view-aids-bypass"' in markup
    assert "never alter decoder input or provenance" in markup
    # One reset control, offered once.
    assert markup.count('id="view-aids-reset"') == 1
    # A reduced working resolution is a named choice with its cost spelled out.
    assert 'id="view-aid-resolution"' in markup
    assert "Lower resolutions are faster but show less detail" in markup

    assert '"/api/view-filters"' in script
    assert '"/api/view-filters/render"' in script
    assert 'VIEW_AID_SHORTCUTS = {v: "bypass", r: "reset"' in script
    assert "adaptive_threshold" in script and "corner_response" in script
    # Corner markers and the filtered base are drawn before the mask, prompt box and
    # point markers, so the existing overlays stay legible on top.
    assert script.index("drawCornerMarkers(aids.corners, scale)") < script.index(
        "const decoded = activeCandidate()"
    )
    # The browser never posts view settings to a prompt or decode route.
    for route in ('"/api/decode"', "`/api/prompts/${item.box_id}`"):
        segment = script[script.index(route) : script.index(route) + 400]
        assert "viewAids" not in segment
