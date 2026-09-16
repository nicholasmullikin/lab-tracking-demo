const canvas = document.querySelector("#canvas"), ctx = canvas.getContext("2d");
const defaultLabels = ["left_hand", "right_hand", "yellow_toy_body", "toy_wheel"];
const state = {
  manifest: null, frame: null, image: null, active: null, selected: null, drag: null,
  zoom: 1, pan: {x: 0, y: 0}, view: {width: 0, height: 0, pixelRatio: 1},
  label: defaultLabels[0], manualSeedTargets: [], activeCandidateId: null, plan: null,
  maskImages: new Map(), maskVersions: new Map(), pendingCandidateSelections: new Map(),
  maskVisible: true, maskOpacity: 0.38, correctionPolicy: null,
  showRejected: false, promptMode: "box",
  liveDecode: false, liveDecodeDelayMs: 450, liveDecodeQueuedBoxId: null, decodeInFlight: false,
  frameRequest: 0, refreshRequest: 0,
  viewFilters: null, viewAids: {}, viewAidsBypass: false, viewRender: null, viewRequest: null,
  viewResolutionDivisor: 1, viewTone: null, viewAbort: null,
};
/* Nodes for the view-aid cards, kept so a slider tick can update the two labels that
 * changed instead of rebuilding the panel underneath the pointer that is dragging it. */
const viewAidCards = new Map();
const viewAidStages = new Map();
const VIEW_AID_STORAGE_KEY = "battle.calibration.view-aids.v1";
const VIEW_AID_SHORTCUTS = {v: "bypass", r: "reset", t: "adaptive_threshold", e: "canny", c: "corner_response"};
const BACKEND_LABELS = {vigra: "VIGRA", "vigra+numpy": "partial", numpy: "NumPy"};
const BACKEND_CLASSES = {vigra: "vigra", "vigra+numpy": "partial", numpy: "numpy"};
const BACKEND_TITLES = {
  vigra: "Calls VIGRA directly.",
  "vigra+numpy": "VIGRA operator with one NumPy step.",
  numpy: "NumPy; VIGRA has no equivalent.",
};
const $ = (selector) => document.querySelector(selector);
const targetColors = ["#22d3ee", "#a78bfa", "#34d399", "#fb7185", "#f59e0b", "#60a5fa"];
const PANEL_STORAGE_KEY = "battle.calibration.panels.v1";
const LIVE_DECODE_STORAGE_KEY = "battle.calibration.live-decode.v1";
const LIVE_DECODE_DELAY_STORAGE_KEY = "battle.calibration.live-decode-delay.v1";
const LIVE_DECODE_DELAY_MIN_MS = 100;
const LIVE_DECODE_DELAY_MAX_MS = 5000;
let liveDecodeTimer = null;
/* The markup carries each section's fresh-profile default; a stored choice always wins.
 * Shortcuts keep working while a section is collapsed, so the only automatic expansion
 * is a live error. */
const panels = new Map();
const panelAttention = new Set();
function loadPanelPreferences() {
  try { return JSON.parse(globalThis.localStorage?.getItem(PANEL_STORAGE_KEY) || "{}") || {}; }
  catch { return {}; }
}
const panelPreferences = loadPanelPreferences();
function savePanelPreferences() {
  try { globalThis.localStorage?.setItem(PANEL_STORAGE_KEY, JSON.stringify(panelPreferences)); }
  catch { /* private-mode or storage-disabled browsers simply lose the preference */ }
}
function registerPanels(root = document) {
  for (const node of root?.querySelectorAll?.("details[data-panel]") || []) {
    const name = node.dataset?.panel;
    if (!name || panels.get(name) === node) continue;
    panels.set(name, node);
    const stored = panelPreferences[name];
    if (typeof stored === "boolean") node.open = stored;
    node.addEventListener?.("toggle", () => {
      panelPreferences[name] = node.open === true;
      savePanelPreferences();
    });
  }
}
function expandPanel(name) {
  const node = panels.get(name);
  if (!node || node.open) return;
  node.open = true;
  panelPreferences[name] = true;
  savePanelPreferences();
}
function setPanelSummary(name, text) {
  const badge = panels.get(name)?.querySelector?.(".panel-count");
  if (badge) badge.textContent = text;
}
function setPanelAttention(name, active) {
  panels.get(name)?.classList?.toggle("attention", active === true);
  if (!active) { panelAttention.delete(name); return; }
  if (panelAttention.has(name)) return;
  panelAttention.add(name);
  expandPanel(name);
}
async function api(path, options = {}) {
  const response = await fetch(path, {...options, headers: {"Content-Type": "application/json", ...(options.headers || {})}});
  const data = response.status === 204 ? {} : await response.json();
  if (!response.ok) throw new Error(data.error || `${response.status} ${response.statusText}`);
  return data;
}
function setStatus(message, error = false) { $("#status").textContent = message; $("#status").style.color = error ? "#fca5a5" : "#a7f3d0"; }
/* Fallback wording only; the server sends the same sentence it refuses requests with. */
const PLAN_LOCK_REASON = "Candidate review is locked because a tracking plan is finalized. "
  + "Reopen the plan for editing to change selections; the finalized artifacts are kept.";
function planLocked() { return state.plan?.finalized === true; }
function planLockReason() { return state.plan?.lock_reason || PLAN_LOCK_REASON; }
function modelSize() {
  const dimensions = state.manifest?.proxy_dimensions;
  return {
    width: dimensions?.width || state.image?.naturalWidth || canvas.width,
    height: dimensions?.height || state.image?.naturalHeight || canvas.height,
  };
}
function viewScale() {
  const {width, height} = modelSize();
  return Math.min(state.view.width / width, state.view.height / height) * state.zoom;
}
function screenPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return {x: event.clientX - rect.left, y: event.clientY - rect.top};
}
function world(point) {
  const scale = viewScale();
  return {x: (point.x - state.pan.x) / scale, y: (point.y - state.pan.y) / scale};
}
function resetView() {
  const {width, height} = modelSize();
  state.zoom = 1;
  const fit = viewScale();
  state.pan = {x: (state.view.width - width * fit) / 2, y: (state.view.height - height * fit) / 2};
}
function resizeCanvas({preserveCenter = false, reset = false} = {}) {
  const previous = state.view;
  const center = preserveCenter && previous.width && previous.height
    ? world({x: previous.width / 2, y: previous.height / 2})
    : null;
  const {width, height} = modelSize();
  canvas.style.aspectRatio = `${width} / ${height}`;
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const pixelRatio = globalThis.devicePixelRatio || 1;
  const backingWidth = Math.round(rect.width * pixelRatio);
  const backingHeight = Math.round(rect.height * pixelRatio);
  if (canvas.width !== backingWidth || canvas.height !== backingHeight) {
    canvas.width = backingWidth;
    canvas.height = backingHeight;
  }
  state.view = {width: rect.width, height: rect.height, pixelRatio};
  if (reset) {
    resetView();
  } else if (center) {
    const scale = viewScale();
    state.pan = {
      x: state.view.width / 2 - center.x * scale,
      y: state.view.height / 2 - center.y * scale,
    };
  }
}
function clamp(value, lower, upper) { return Math.max(lower, Math.min(upper, value)); }
function clampBox(box) {
  const {width, height} = modelSize();
  return {
    x1: clamp(Math.round(Math.min(box.x1, box.x2)), 0, width),
    y1: clamp(Math.round(Math.min(box.y1, box.y2)), 0, height),
    x2: clamp(Math.round(Math.max(box.x1, box.x2)), 0, width),
    y2: clamp(Math.round(Math.max(box.y1, box.y2)), 0, height),
  };
}
function moveBox(box, dx, dy) {
  const {width, height} = modelSize();
  const offsetX = clamp(Math.round(dx), -box.x1, width - box.x2);
  const offsetY = clamp(Math.round(dy), -box.y1, height - box.y2);
  return {x1: box.x1 + offsetX, y1: box.y1 + offsetY, x2: box.x2 + offsetX, y2: box.y2 + offsetY};
}
function pending() { return state.manifest.workspace.pending_boxes; }
function promptPoints(item, kind) {
  return item?.[kind === "foreground" ? "pixel_fg_points" : "pixel_bg_points"] || [];
}
function activePendingPrompt() {
  const selected = pending().find((item) => item.box_id === state.selected);
  if (selected && isOnActiveFrame(selected) && selected.intended_target === state.label) return selected;
  const matching = pending().filter(
    (item) => isOnActiveFrame(item) && item.intended_target === state.label
  );
  return matching.length === 1 ? matching[0] : null;
}
function syncDecodeControls() {
  const button = $("#decode");
  if (!button || !state.manifest) return;
  const hasPending = pending().some((item) => isOnActiveFrame(item));
  button.disabled = state.liveDecode || state.decodeInFlight || !hasPending;
  button.title = state.liveDecode
    ? "Disable live decode to run a manual frame batch."
    : state.decodeInFlight
      ? "Wait for the current decode to finish."
      : hasPending
        ? "Decode every pending prompt on this frame."
        : "No pending prompts on this frame.";
}
function targetColor(target) {
  const position = targetPosition(target);
  return targetColors[position === Number.MAX_SAFE_INTEGER ? 0 : position % targetColors.length];
}
function targetPosition(target) {
  const position = labels().indexOf(target);
  return position === -1 ? Number.MAX_SAFE_INTEGER : position;
}
function reviewedCandidates() {
  return state.manifest.candidates.filter((item) => state.showRejected || !item.rejected).sort((left, right) => (
    targetPosition(left.intended_target) - targetPosition(right.intended_target)
    || left.intended_target.localeCompare(right.intended_target)
    || left.frame.proxy_seconds - right.frame.proxy_seconds
    || left.candidate_id.localeCompare(right.candidate_id)
  ));
}
function activeTimestamp() {
  return state.active ?? state.manifest?.workspace.active_proxy_timestamp_seconds;
}
function isOnActiveFrame(item) {
  return Math.abs(item.frame.proxy_seconds - activeTimestamp()) <= 1e-9;
}
function activeCandidate() {
  const matching = state.manifest.candidates.filter(
    (item) => !item.rejected && item.intended_target === state.label && isOnActiveFrame(item)
  );
  return matching.find((item) => item.candidate_id === state.activeCandidateId)
    || matching.at(-1)
    || null;
}
function selectedCandidateIndex(item) {
  return state.pendingCandidateSelections.get(item.candidate_id)
    ?? item.human_selected_candidate_index
    ?? item.decoder_result.deterministic_best_candidate_index;
}
function selectedMask(item) {
  return item?.decoder_result.candidates.find(
    (candidate) => candidate.candidate_index === selectedCandidateIndex(item)
  ) || null;
}
function tintedBinaryMaskImage(image, color, opacity) {
  if (!image.naturalWidth || !image.naturalHeight) return image;
  const overlay = document.createElement("canvas");
  overlay.width = image.naturalWidth; overlay.height = image.naturalHeight;
  const overlayContext = overlay.getContext("2d", {willReadFrequently: true});
  if (!overlayContext?.getImageData) return image;
  overlayContext.drawImage(image, 0, 0);
  const pixels = overlayContext.getImageData(0, 0, overlay.width, overlay.height);
  const red = Number.parseInt(color.slice(1, 3), 16);
  const green = Number.parseInt(color.slice(3, 5), 16);
  const blue = Number.parseInt(color.slice(5, 7), 16);
  const alpha = Math.round(255 * opacity);
  for (let offset = 0; offset < pixels.data.length; offset += 4) {
    const positive = pixels.data[offset] || pixels.data[offset + 1] || pixels.data[offset + 2];
    pixels.data[offset] = positive ? red : 0;
    pixels.data[offset + 1] = positive ? green : 0;
    pixels.data[offset + 2] = positive ? blue : 0;
    pixels.data[offset + 3] = positive ? alpha : 0;
  }
  overlayContext.putImageData(pixels, 0, 0);
  return overlay;
}
function maskImage(candidate, color, opacity) {
  if (!candidate?.mask_uri) return null;
  let entry = state.maskImages.get(candidate.mask_uri);
  if (!entry) {
    const image = new Image();
    entry = {image, loaded: false, overlays: new Map()};
    state.maskImages.set(candidate.mask_uri, entry);
    image.onload = () => { entry.loaded = true; render(); };
    image.onerror = () => setStatus(`Could not load decoded mask: ${candidate.mask_uri}`, true);
    const version = state.maskVersions.get(candidate.mask_uri);
    image.src = `/artifacts/${candidate.mask_uri}${version ? `?v=${version}` : ""}`;
  }
  if (!entry.loaded) return null;
  const key = `${color}:${opacity}`;
  let overlay = entry.overlays.get(key);
  if (!overlay) {
    overlay = tintedBinaryMaskImage(entry.image, color, opacity);
    entry.overlays.set(key, overlay);
  }
  return overlay;
}
function hit(point) {
  const p = world(point), box = pending().findLast((item) => (
    isOnActiveFrame(item)
    && p.x >= item.pixel_box.x1 && p.x <= item.pixel_box.x2
    && p.y >= item.pixel_box.y1 && p.y <= item.pixel_box.y2
  ));
  if (!box) return null;
  const b = box.pixel_box, edge = 12 / viewScale();
  const handle = p.x > b.x2 - edge && p.y > b.y2 - edge ? "resize" : "move";
  return {box, handle, start: p, startBox: {...b}};
}
function hitPoint(point) {
  const p = world(point), radius = 12 / viewScale();
  const candidates = pending().filter(
    (item) => isOnActiveFrame(item) && item.intended_target === state.label
  );
  for (const item of candidates.toReversed()) {
    for (const kind of ["background", "foreground"]) {
      const points = promptPoints(item, kind);
      for (let index = points.length - 1; index >= 0; index -= 1) {
        const marker = points[index];
        if (Math.hypot(p.x - marker.x, p.y - marker.y) <= radius) return {item, kind, index};
      }
    }
  }
  return null;
}
function drawPoint(point, kind, scale, selected) {
  const radius = 7 / scale;
  const color = kind === "foreground" ? "#22c55e" : "#fb7185";
  ctx.save();
  ctx.lineWidth = 2 / scale;
  ctx.strokeStyle = selected ? "#fbbf24" : "#f8fafc";
  ctx.fillStyle = color;
  if (kind === "foreground") {
    ctx.beginPath(); ctx.arc(point.x, point.y, radius, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    ctx.strokeStyle = "#052e16";
    ctx.beginPath();
    ctx.moveTo(point.x - radius / 2, point.y); ctx.lineTo(point.x + radius / 2, point.y);
    ctx.moveTo(point.x, point.y - radius / 2); ctx.lineTo(point.x, point.y + radius / 2);
    ctx.stroke();
  } else {
    ctx.beginPath(); ctx.arc(point.x, point.y, radius, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    ctx.strokeStyle = "#4c0519";
    ctx.beginPath();
    ctx.moveTo(point.x - radius / 2, point.y - radius / 2); ctx.lineTo(point.x + radius / 2, point.y + radius / 2);
    ctx.moveTo(point.x + radius / 2, point.y - radius / 2); ctx.lineTo(point.x - radius / 2, point.y + radius / 2);
    ctx.stroke();
  }
  ctx.restore();
}
function viewAidOperators() { return state.viewFilters?.operators || []; }
function viewAidOperator(id) { return viewAidOperators().find((item) => item.id === id) || null; }
function defaultViewAids() {
  const settings = {};
  for (const operator of viewAidOperators()) {
    const entry = {enabled: false};
    for (const parameter of operator.parameters) entry[parameter.id] = parameter.default;
    for (const choice of operator.choices) entry[choice.id] = choice.default;
    settings[operator.id] = entry;
  }
  return settings;
}
function loadViewAids() {
  let stored = null;
  try { stored = JSON.parse(globalThis.localStorage?.getItem(VIEW_AID_STORAGE_KEY) || "null"); }
  catch { stored = null; }
  const defaults = defaultViewAids();
  state.viewAids = Object.fromEntries(Object.entries(defaults).map(([id, entry]) => (
    [id, {...entry, ...(stored?.operators?.[id] || {})}]
  )));
  state.viewAidsBypass = stored?.bypass === true;
  state.viewResolutionDivisor = viewResolutionDivisors().includes(stored?.resolution_divisor)
    ? stored.resolution_divisor
    : 1;
}
function saveViewAids() {
  // A browser-local view preference only; calibration manifests never carry these settings.
  try {
    globalThis.localStorage?.setItem(
      VIEW_AID_STORAGE_KEY,
      JSON.stringify({
        bypass: state.viewAidsBypass,
        operators: state.viewAids,
        resolution_divisor: state.viewResolutionDivisor,
      }),
    );
  } catch { /* private-mode or storage-disabled browsers simply lose the preference */ }
}
function viewResolutionDivisors() {
  return (state.viewFilters?.resolution_divisors || []).map((item) => item.divisor);
}
function enabledViewAids() {
  return viewAidOperators().filter((operator) => state.viewAids[operator.id]?.enabled);
}
/* Brightness and contrast are pointwise, so when nothing downstream depends on them the
 * browser applies them itself and the network stays out of the loop entirely. As soon as
 * a threshold, edge, or corner operator is on, the server runs the whole chain again:
 * those detectors read the toned image, and splitting the chain would change their input. */
function toneOnly() {
  const enabled = enabledViewAids();
  return enabled.length > 0 && enabled.every((operator) => operator.stage === "tone");
}
function viewAidKey() {
  const enabled = enabledViewAids().map((operator) => {
    const entry = state.viewAids[operator.id];
    return `${operator.id}:${Object.keys(entry).sort().map((key) => `${key}=${entry[key]}`).join(",")}`;
  });
  return `${state.frame}|${state.viewResolutionDivisor}|${enabled.join("|")}`;
}
/* Apply the tone stage in the browser with the same arithmetic the server uses:
 * Rec.601 luminance, an additive brightness offset, then the gain about mid-grey, each
 * clamped and truncated to 8 bits exactly as NumPy does. This is a display surface only
 * and the decoder still reads the untouched source frame. */
function toneCanvas() {
  const brightness = state.viewAids.brightness || {};
  const contrast = state.viewAids.contrast || {};
  const key = `${state.frame}|b=${brightness.enabled ? brightness.amount : "off"}`
    + `|c=${contrast.enabled ? contrast.amount : "off"}`;
  if (state.viewTone?.key === key) return state.viewTone.canvas;
  const source = state.image;
  if (!source?.naturalWidth) return null;
  const offscreen = document.createElement("canvas");
  offscreen.width = source.naturalWidth;
  offscreen.height = source.naturalHeight;
  const offscreenContext = offscreen.getContext("2d", {willReadFrequently: true});
  if (!offscreenContext?.getImageData) return null;
  offscreenContext.drawImage(source, 0, 0);
  const frame = offscreenContext.getImageData(0, 0, offscreen.width, offscreen.height);
  if (!frame?.data) return null;
  const amount = (contrast.enabled ? Number(contrast.amount) : 0) * 2.55;
  const factor = (259 * (amount + 255)) / (255 * (259 - amount));
  const offset = brightness.enabled ? Number(brightness.amount) : 0;
  const data = frame.data;
  for (let index = 0; index < data.length; index += 4) {
    let value = 0.299 * data[index] + 0.587 * data[index + 1] + 0.114 * data[index + 2];
    if (brightness.enabled) value = Math.min(255, Math.max(0, value + offset));
    if (contrast.enabled) value = Math.min(255, Math.max(0, factor * (value - 128) + 128));
    const toned = Math.trunc(value);
    data[index] = toned; data[index + 1] = toned; data[index + 2] = toned;
  }
  offscreenContext.putImageData(frame, 0, 0);
  state.viewTone = {key, canvas: offscreen};
  return offscreen;
}
/* Ask the server for a display-only rendering of the current frame. The response is
 * painted on the canvas and nothing else; prompts and decode requests are untouched. */
async function requestViewRender() {
  const timing = $("#view-aids-timing");
  if (state.viewAidsBypass || !enabledViewAids().length || state.active === null || state.active === undefined) {
    state.viewAbort?.abort();
    state.viewAbort = null;
    state.viewRequest = null;
    state.viewRender = null;
    if (timing) timing.textContent = "";
    setPanelAttention("view-aids", false);
    render();
    return;
  }
  const key = viewAidKey();
  if (state.viewRender?.key === key || state.viewRequest === key) return;
  if (toneOnly()) {
    state.viewAbort?.abort();
    state.viewAbort = null;
    state.viewRequest = null;
    const started = performance.now();
    const image = toneCanvas();
    if (image) {
      state.viewRender = {key, image, corners: [], backends: {}, divisor: 1};
      setPanelAttention("view-aids", false);
      if (timing) {
        timing.textContent = `${enabledViewAids().map((operator) => operator.id).join(", ")}`
          + ` · ${Math.round(performance.now() - started)} ms in browser · display only`;
      }
      render();
      return;
    }
  }
  state.viewAbort?.abort();
  const controller = new AbortController();
  state.viewAbort = controller;
  state.viewRequest = key;
  if (timing) timing.textContent = "Rendering…";
  try {
    const result = await api("/api/view-filters/render", {
      method: "POST",
      signal: controller.signal,
      body: JSON.stringify({
        timestamp: state.active,
        settings: state.viewAids,
        resolution_divisor: state.viewResolutionDivisor,
      }),
    });
    if (state.viewRequest !== key) return;
    const image = new Image();
    image.onload = () => {
      if (state.viewRequest !== key || key !== viewAidKey()) return;
      state.viewRender = {
        key, image, corners: result.corners, backends: result.backends,
        divisor: result.resolution_divisor || 1,
      };
      render();
    };
    image.src = `data:image/png;base64,${result.image_png_base64}`;
    setPanelAttention("view-aids", false);
    if (timing) {
      const working = result.resolution_divisor > 1
        ? ` · working ${result.width}×${result.height} (1/${result.resolution_divisor}), upscaled for display`
        : "";
      timing.textContent = `${result.applied.join(", ")} · ${result.milliseconds} ms${result.cached ? " (cached)" : ""}${working} · display only`;
    }
  } catch (error) {
    if (error.name === "AbortError") return;
    state.viewRender = null;
    if (timing) timing.textContent = `View aids unavailable: ${error.message}`;
    setPanelAttention("view-aids", true);
    setStatus(`View aids unavailable: ${error.message}`, true);
    render();
  } finally {
    if (state.viewRequest === key) state.viewRequest = null;
    if (state.viewAbort === controller) state.viewAbort = null;
  }
}
let viewRenderTimer = null;
/* Coalesce a slider drag into one request per pause, and keep the tone stage immediate
 * because it never leaves the browser. */
function scheduleViewRender() {
  if (viewRenderTimer) clearTimeout(viewRenderTimer);
  if (toneOnly()) { viewRenderTimer = null; requestViewRender(); return; }
  viewRenderTimer = setTimeout(() => { viewRenderTimer = null; requestViewRender(); }, 140);
}
function drawCornerMarkers(groups, scale) {
  const radius = 5 / scale;
  ctx.save();
  ctx.lineWidth = 1.6 / scale;
  for (const group of groups) {
    ctx.strokeStyle = group.color;
    for (const point of group.points) {
      ctx.beginPath();
      ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
      ctx.stroke();
    }
  }
  ctx.restore();
}
function viewAidNode(operator) {
  const card = document.createElement("div");
  const settings = state.viewAids[operator.id];
  const blocked = operator.requires_vigra && !operator.available;
  card.className = `view-aid ${settings.enabled && !blocked ? "on" : ""}${blocked ? " blocked" : ""}`;
  card.style.borderLeftColor = operator.color || "#475569";
  viewAidCards.set(operator.id, {card, blocked});
  const toggle = document.createElement("label"), check = document.createElement("input");
  check.type = "checkbox";
  check.checked = settings.enabled && !blocked;
  check.disabled = blocked;
  check.onchange = () => setViewAid(operator.id, "enabled", check.checked);
  viewAidCards.get(operator.id).check = check;
  const shortcut = Object.entries(VIEW_AID_SHORTCUTS).find(([, id]) => id === operator.id)?.[0];
  toggle.append(check, document.createTextNode(operator.label));
  if (shortcut) {
    const key = document.createElement("kbd");
    key.textContent = shortcut.toUpperCase();
    toggle.append(key);
  }
  const backend = document.createElement("small");
  backend.className = `fidelity ${BACKEND_CLASSES[operator.backend]}`;
  backend.title = BACKEND_TITLES[operator.backend];
  backend.textContent = blocked
    ? `${BACKEND_LABELS[operator.backend]} · unavailable`
    : BACKEND_LABELS[operator.backend];
  card.title = operator.provenance;
  card.append(toggle, backend);
  for (const choice of operator.choices) {
    const row = document.createElement("label"), select = document.createElement("select");
    row.className = "parameter";
    select.replaceChildren(...choice.options.map((name) => {
      const option = document.createElement("option");
      option.value = name; option.textContent = name;
      option.selected = settings[choice.id] === name;
      return option;
    }));
    select.value = settings[choice.id];
    select.disabled = blocked;
    select.onchange = () => setViewAid(operator.id, choice.id, select.value);
    row.append(document.createTextNode(choice.label), select);
    card.append(row);
  }
  for (const parameter of operator.parameters) {
    const row = document.createElement("label"), slider = document.createElement("input");
    const readout = document.createElement("output");
    row.className = "parameter";
    slider.type = "range";
    slider.min = parameter.minimum; slider.max = parameter.maximum; slider.step = parameter.step;
    slider.value = settings[parameter.id];
    slider.disabled = blocked;
    readout.textContent = String(settings[parameter.id]);
    slider.oninput = () => {
      readout.textContent = slider.value;
      setViewAid(operator.id, parameter.id, Number(slider.value));
    };
    row.append(document.createTextNode(parameter.label), readout, slider);
    card.append(row);
  }
  return card;
}
function renderViewAidControls() {
  const host = $("#view-aid-stages");
  if (!host || !state.viewFilters) return;
  viewAidCards.clear();
  viewAidStages.clear();
  host.replaceChildren(...state.viewFilters.stage_order.flatMap((stage) => {
    const operators = viewAidOperators().filter((item) => item.stage === stage);
    if (!operators.length) return [];
    const section = document.createElement("details");
    section.className = "view-aid-stage panel";
    section.dataset.panel = `view-aid-stage:${stage}`;
    const heading = document.createElement("summary");
    const grid = document.createElement("div");
    grid.className = "view-aid-grid";
    grid.append(...operators.map(viewAidNode));
    section.append(heading, grid);
    viewAidStages.set(stage, {heading, operators});
    return [section];
  }));
  registerPanels(host);
  const backend = $("#view-aids-backend");
  if (backend) {
    backend.textContent = state.viewFilters.vigra_available
      ? `VIGRA ${state.viewFilters.vigra_version} installed.`
      : state.viewFilters.unavailable_reason;
    backend.className = state.viewFilters.vigra_available ? "hint" : "hint warning";
  }
  renderResolutionControl();
  refreshViewAidChrome();
}
/* Update only the labels and classes a control change actually affects. Rebuilding the
 * whole panel used to destroy the very slider the pointer was dragging. */
function refreshViewAidChrome() {
  for (const [stage, entry] of viewAidStages) {
    const on = entry.operators.filter((item) => state.viewAids[item.id]?.enabled).length;
    entry.heading.textContent =
      `${state.viewFilters?.stage_labels?.[stage] || stage} · ${on}/${entry.operators.length} on`;
  }
  for (const [operatorId, entry] of viewAidCards) {
    const enabled = state.viewAids[operatorId]?.enabled === true && !entry.blocked;
    entry.card.className = `view-aid ${enabled ? "on" : ""}${entry.blocked ? " blocked" : ""}`;
    if (entry.check) entry.check.checked = enabled;
  }
  const bypass = $("#view-aids-bypass");
  if (bypass) bypass.checked = state.viewAidsBypass;
  const resolution = $("#view-aid-resolution");
  if (resolution) resolution.value = String(state.viewResolutionDivisor);
  const summary = $("#view-aids-summary");
  if (summary) {
    const enabled = enabledViewAids().map((operator) => operator.label);
    summary.textContent = !enabled.length
      ? "off"
      : state.viewAidsBypass
      ? `bypassed (${enabled.length} configured)`
      : enabled.join(", ");
  }
}
/* The working resolution is a real loss of detail in the VIGRA operators, so it is a
 * named control with the pixel size spelled out, never an invisible speed-up. */
function renderResolutionControl() {
  const select = $("#view-aid-resolution");
  if (!select || !state.viewFilters) return;
  const {width, height} = modelSize();
  select.replaceChildren(...(state.viewFilters.resolution_divisors || []).map((item) => {
    const option = document.createElement("option");
    option.value = String(item.divisor);
    const size = `${Math.max(1, Math.floor(width / item.divisor))}×`
      + `${Math.max(1, Math.floor(height / item.divisor))}`;
    option.textContent = `${item.label} · ${size}`;
    option.selected = item.divisor === state.viewResolutionDivisor;
    return option;
  }));
  select.value = String(state.viewResolutionDivisor);
  select.onchange = () => setViewResolutionDivisor(Number(select.value));
}
function setViewResolutionDivisor(divisor) {
  state.viewResolutionDivisor = viewResolutionDivisors().includes(divisor) ? divisor : 1;
  saveViewAids();
  refreshViewAidChrome();
  requestViewRender();
}
function setViewAid(operatorId, key, value) {
  if (!state.viewAids[operatorId]) return;
  state.viewAids[operatorId] = {...state.viewAids[operatorId], [key]: value};
  saveViewAids();
  refreshViewAidChrome();
  scheduleViewRender();
}
function toggleViewAid(operatorId) {
  const operator = viewAidOperator(operatorId);
  if (!operator || (operator.requires_vigra && !operator.available)) return;
  setViewAid(operatorId, "enabled", !state.viewAids[operatorId].enabled);
}
function setViewAidsBypass(bypass) {
  state.viewAidsBypass = bypass;
  saveViewAids();
  refreshViewAidChrome();
  requestViewRender();
}
function resetViewAids() {
  state.viewAids = defaultViewAids();
  state.viewAidsBypass = false;
  state.viewRender = null;
  state.viewRequest = null;
  state.viewTone = null;
  state.viewResolutionDivisor = 1;
  state.viewAbort?.abort();
  state.viewAbort = null;
  saveViewAids();
  renderViewAidControls();
  const timing = $("#view-aids-timing");
  if (timing) timing.textContent = "";
  if (state.image) resetView();
  render();
  setStatus("View reset: zoom, pan, and every view aid back to defaults.");
}
async function loadViewFilters() {
  try {
    state.viewFilters = await api("/api/view-filters");
  } catch (error) {
    state.viewFilters = null;
    setPanelAttention("view-aids", true);
    setStatus(`Could not load view-aid controls: ${error.message}`, true);
    return;
  }
  loadViewAids();
  renderViewAidControls();
  requestViewRender();
}
function render() {
  if (!state.image) return;
  const {width, height} = modelSize(), scale = viewScale();
  ctx.setTransform(state.view.pixelRatio, 0, 0, state.view.pixelRatio, 0, 0);
  ctx.clearRect(0, 0, state.view.width, state.view.height);
  ctx.save(); ctx.translate(state.pan.x, state.pan.y); ctx.scale(scale, scale);
  const aids = !state.viewAidsBypass && state.viewRender?.key === viewAidKey()
    ? state.viewRender
    : null;
  // A reduced working resolution is shown as the pixels it really produced, not as a
  // smoothed interpolation that would look like detail the operator never saw.
  const coarse = aids?.divisor > 1;
  if (coarse) ctx.imageSmoothingEnabled = false;
  ctx.drawImage(aids?.image || state.image, 0, 0, width, height);
  if (coarse) ctx.imageSmoothingEnabled = true;
  if (aids?.corners?.length) drawCornerMarkers(aids.corners, scale);
  const decoded = activeCandidate(), mask = selectedMask(decoded);
  const image = maskImage(mask, decoded ? targetColor(decoded.intended_target) : "#000000", state.maskOpacity);
  if (decoded && mask && image && state.maskVisible) {
    ctx.save();
    ctx.drawImage(image, 0, 0, width, height);
    ctx.restore();
  }
  const editablePrompt = activePendingPrompt();
  if (decoded && mask && !editablePrompt) {
    const b = decoded.pixel_box, color = targetColor(decoded.intended_target);
    ctx.strokeStyle = color; ctx.lineWidth = 3 / scale;
    ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
    ctx.fillStyle = color; ctx.font = `${16 / scale}px sans-serif`;
    ctx.fillText(
      `${decoded.intended_target} · mask ${mask.candidate_index + 1}`,
      b.x1 + 3 / scale, b.y1 - 5 / scale
    );
  }
  for (const item of pending().filter(isOnActiveFrame)) {
    const preview = state.drag?.box?.box_id === item.box_id ? state.drag.preview : null;
    const b = preview || item.pixel_box, selected = item.box_id === state.selected;
    ctx.strokeStyle = selected ? "#f59e0b" : targetColor(item.intended_target); ctx.lineWidth = 3 / scale;
    ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
    ctx.fillStyle = ctx.strokeStyle; ctx.font = `${16 / scale}px sans-serif`;
    ctx.fillText(item.intended_target, b.x1 + 3 / scale, b.y1 - 5 / scale);
    if (selected) { ctx.fillRect(b.x2 - 10 / scale, b.y2 - 10 / scale, 10 / scale, 10 / scale); }
    if (isOnActiveFrame(item) && item.intended_target === state.label) {
      for (const kind of ["foreground", "background"]) {
        promptPoints(item, kind).forEach((point, index) => {
          const preview = state.drag?.mode === "point"
            && state.drag.item.box_id === item.box_id
            && state.drag.kind === kind && state.drag.index === index
            ? state.drag.previewPoint : point;
          drawPoint(preview, kind, scale, selected);
        });
      }
    }
  }
  if (state.drag?.mode === "draw") {
    const {start, current} = state.drag; ctx.strokeStyle = "#fbbf24"; ctx.lineWidth = 2 / scale;
    ctx.strokeRect(start.x, start.y, current.x - start.x, current.y - start.y);
  }
  ctx.restore();
}
function labels() { return state.manualSeedTargets.length ? state.manualSeedTargets : defaultLabels; }
function renderLabelSummary() {
  setPanelSummary("label", `${state.label.replaceAll("_", " ")} · ${labels().length} target(s)`);
}
function selectLabel(label) {
  state.label = label;
  $("#labels").querySelectorAll("button").forEach((button) => {
    button.classList.toggle("active", button.dataset.label === label);
  });
  renderLabelSummary();
  renderActiveCandidate(); renderPointControls();
  render();
}
function renderLabels() {
  const available = labels();
  if (!available.includes(state.label)) state.label = available[0];
  $("#labels").replaceChildren(...available.map((label, index) => {
    const button = document.createElement("button");
    button.dataset.label = label;
    button.textContent = `${index + 1} ${label.replaceAll("_", " ")}`;
    button.className = label === state.label ? "active" : "";
    return button;
  }));
  renderLabelSummary();
}
function renderTargetPolicy(policy) {
  const help = policy
    ? `Frame 0 requires ${policy.required_target_count} distinct masks: ${labels().join(", ")}.`
    : "Custom labels are allowed; no named proposal policy is active.";
  const targetPolicy = $("#target-policy"), panel = $("#label-panel");
  if (targetPolicy) targetPolicy.textContent = "";
  if (panel) panel.title = help;
  const customLabel = $("#custom-label-control");
  if (customLabel) customLabel.hidden = Boolean(policy);
}
function setPromptMode(mode) {
  state.promptMode = mode;
  $("#prompt-mode")?.querySelectorAll("button").forEach((button) => {
    button.classList.toggle("active", button.dataset.promptMode === mode);
  });
  renderPointControls();
  render();
}
function renderPointControls() {
  const guidance = $("#point-guidance"), clear = $("#clear-points");
  if (!guidance || !clear) return;
  const prompt = activePendingPrompt();
  const names = {
    box: "Drag a tight box around the target.",
    foreground: "Click unambiguous target pixels.",
    background: "Click pixels the mask must exclude.",
  };
  const help = prompt
    ? `${names[state.promptMode]} ${prompt.pixel_fg_points.length} foreground and ${prompt.pixel_bg_points.length} background point(s) on ${prompt.intended_target.replaceAll("_", " ")}.`
    : `Select exactly one pending ${state.label.replaceAll("_", " ")} box on this frame before adding points.`;
  guidance.textContent = "";
  const panel = $("#prompt-clicks-panel");
  if (panel) panel.title = help;
  clear.disabled = !prompt || !(prompt.pixel_fg_points.length || prompt.pixel_bg_points.length);
  setPanelSummary("prompt-clicks", prompt
    ? `${state.promptMode} · +${prompt.pixel_fg_points.length}/−${prompt.pixel_bg_points.length}`
    : `${state.promptMode} · no prompt selected`);
}
function renderActiveCandidate() {
  const node = $("#active-candidate");
  if (!node) return;
  const item = activeCandidate(), mask = selectedMask(item);
  if (!item || !mask) {
    node.textContent = `No decoded mask for ${state.label.replaceAll("_", " ")} on this frame.`;
    node.title = "Decode a prompt for this target to preview its mask.";
    return;
  }
  const status = item.human_accepted ? "accepted" : "model-best preview";
  node.textContent = `${item.intended_target.replaceAll("_", " ")} · mask ${mask.candidate_index + 1}/${item.decoder_result.candidates.length} · ${status}`;
  node.title = "Keys 1–4 select and accept the corresponding mask.";
}
/* The lock is a workspace-wide mode, so it is announced above every panel rather than
 * inside the review section whose controls it disables. */
function renderPlanLock() {
  const banner = $("#plan-lock");
  const locked = planLocked();
  const finalize = $("#finalize-plan");
  if (finalize) {
    finalize.disabled = locked;
    if (locked) {
      finalize.title = "Already finalized. Reopen the plan for editing before writing a new revision.";
    }
  }
  if (!banner) return;
  banner.hidden = !locked;
  banner.replaceChildren();
  if (!locked) return;
  const plan = state.plan || {};
  const heading = document.createElement("strong");
  heading.textContent = `Tracking plan revision ${plan.plan_revision} is finalized — candidate review is locked.`;
  const reason = document.createElement("p");
  reason.textContent = planLockReason();
  const written = [plan.final_proposal_uri, plan.final_correction_schedule_uri].filter(Boolean);
  const artifacts = document.createElement("p");
  artifacts.className = "hint";
  artifacts.textContent = written.length ? `Written: ${written.join(" · ")}` : "";
  const reopen = document.createElement("button");
  reopen.textContent = "Reopen for editing";
  reopen.title = "Return this workspace to editable review. The finalized plan files stay on "
    + "disk as a superseded revision; finalizing again writes the next revision beside them.";
  reopen.onclick = () => reopenPlan();
  banner.append(heading, reason, artifacts, reopen);
}
async function reopenPlan() {
  try {
    const data = await api("/api/reopen-tracking-plan", {method: "POST", body: "{}"});
    const superseded = data.plan?.superseded_plans?.length || 0;
    await refresh();
    setStatus(`Reopened for editing; ${superseded} superseded plan revision(s) remain on disk unchanged.`);
  } catch (error) { setStatus(error.message, true); }
}
async function refresh() {
  const request = ++state.refreshRequest;
  const data = await api("/api/state");
  if (request !== state.refreshRequest) return;
  state.manifest = data.manifest; state.manualSeedTargets = data.manual_seed_targets; state.correctionPolicy = data.correction_policy;
  state.plan = data.plan || null;
  const selectedPrompt = pending().find((item) => (
    item.box_id === state.selected && isOnActiveFrame(item)
  ));
  const persistedPrompt = pending().find((item) => (
    item.box_id === state.manifest.workspace.selected_box_id && isOnActiveFrame(item)
  ));
  state.selected = selectedPrompt?.box_id || persistedPrompt?.box_id || null;
  renderLabels();
  renderTargetPolicy(data.manual_seed_target_policy);
  const finalizeButton = $("#finalize-plan");
  if (finalizeButton) {
    const policy = data.correction_policy;
    finalizeButton.title = policy
      ? `Writes the frame-0 proposal and correction schedule. At most ${policy.maximum_later_correction_keyframes_per_target} later keyframes per target.`
      : "Writes the selected frame-0 tracking proposal.";
  }
  renderPlanLock();
  $("#time-slider").max = Math.max(0, state.manifest.requested_proxy_timestamps_seconds.length - 1);
  $("#time-slider").value = Math.max(0, state.manifest.requested_proxy_timestamps_seconds.indexOf(state.active));
  $("#timestamp").value = state.active ?? state.manifest.workspace.active_proxy_timestamp_seconds;
  $("#filmstrip").replaceChildren(...state.manifest.requested_proxy_timestamps_seconds.map((time) => {
    const button = document.createElement("button"); button.textContent = `${time.toFixed(3)} s`;
    button.className = time === state.active ? "active" : ""; button.onclick = () => loadFrame(time); return button;
  }));
  const onFrame = pending().filter((p) => p.frame.proxy_seconds === state.active);
  const prompts = $("#prompts");
  if (prompts) prompts.replaceChildren(...onFrame.map(promptNode));
  syncDecodeControls();
  const rows = [...pending(), ...state.manifest.candidates];
  $("#table").replaceChildren(...rows.map(tableNode));
  $("#diff").replaceChildren(...data.last_diff.map((change) => { const li = document.createElement("li"); li.textContent = change; return li; }));
  const timestamps = state.manifest.requested_proxy_timestamps_seconds;
  const activeTime = activeTimestamp();
  setPanelSummary("frames", `${timestamps.length} frame${timestamps.length === 1 ? "" : "s"}${
    activeTime === null || activeTime === undefined ? "" : ` · active ${activeTime.toFixed(3)} s`}`);
  setPanelSummary("prompts", `${onFrame.length} on this frame · ${pending().length} total`);
  setPanelSummary("table", `${rows.length} row${rows.length === 1 ? "" : "s"}`);
  setPanelSummary("diff", `${data.last_diff.length} change${data.last_diff.length === 1 ? "" : "s"}`);
  renderCandidates(); renderEligible(); renderActiveCandidate(); renderPointControls(); render();
  const rejected = state.manifest.candidates.filter((item) => item.rejected).length;
  const toggle = $("#toggle-rejected");
  if (toggle) {
    toggle.hidden = rejected === 0 && !state.showRejected;
    toggle.textContent = state.showRejected
      ? `Hide rejected (${rejected})`
      : `Show rejected (${rejected})`;
  }
  setStatus(data.worker_online ? "Ready — edits autosave atomically." : "Static mode — decoder offline.", !data.worker_online);
}
function promptNode(item) {
  const node = document.createElement("div"); node.className = `prompt ${item.box_id === state.selected ? "selected" : ""}`;
  node.textContent = `${item.box_id} · ${item.intended_target} · ${item.stage} · ${item.pixel_box.x1},${item.pixel_box.y1}–${item.pixel_box.x2},${item.pixel_box.y2} · +${promptPoints(item, "foreground").length}/−${promptPoints(item, "background").length}`;
  node.onclick = () => { state.selected = item.box_id; state.label = item.intended_target; refresh(); }; return node;
}
function tableNode(item) {
  const isPending = "box_id" in item, frame = item.frame;
  const node = document.createElement("div"); node.className = "row";
  const b = item.pixel_box;
  const choice = isPending ? item.stage : item.rejected ? "rejected · artifacts retained"
    : item.human_accepted ? `mask ${item.human_selected_candidate_index}${item.selected_for_finalization ? " · eligible" : ""}` : "review pending";
  const points = `+${promptPoints(item, "foreground").length}/−${promptPoints(item, "background").length}`;
  node.textContent = `${frame.proxy_seconds.toFixed(3)}s · ${item.intended_target} · ${isPending ? item.box_id : item.candidate_id} · ${choice} · ${b.x1},${b.y1}–${b.x2},${b.y2} · ${points}`;
  return node;
}
/* Everything the candidate cards display. Drawing a box or moving a point refreshes the
 * whole workspace, and rebuilding these cards reloads every review thumbnail, so the
 * list is only rebuilt when something it actually shows has changed. */
function candidatesSignature() {
  return JSON.stringify([
    state.showRejected,
    state.activeCandidateId,
    activeCandidate()?.candidate_id ?? null,
    activeTimestamp() ?? null,
    state.label,
    labels(),
    pending().map((item) => [
      item.box_id, item.intended_target, item.frame.proxy_seconds, item.stage,
    ]),
    [...state.pendingCandidateSelections],
    planLocked(),
    state.manifest.candidates.map((item) => [
      item.candidate_id, item.intended_target, item.rejected, item.human_accepted,
      item.human_selected_candidate_index, item.selected_for_finalization,
      item.selected_for_correction,
      item.decoder_result.candidates.map((candidate) => candidate.candidate_index),
    ]),
  ]);
}
let candidatesRendered = null;
function renderCandidateStatusTable(host, locked, lockReason) {
  const table = document.createElement("table");
  table.className = "candidate-status-table";
  const heading = document.createElement("tr");
  const frameHeading = document.createElement("th");
  frameHeading.textContent = "Frame";
  heading.append(frameHeading);
  for (const target of labels()) {
    const cell = document.createElement("th");
    cell.textContent = target.replaceAll("_", " ");
    heading.append(cell);
  }
  table.append(heading);
  for (const time of state.manifest.requested_proxy_timestamps_seconds) {
    const row = document.createElement("tr");
    const frame = document.createElement("th");
    const frameIndex = Math.round(time * state.manifest.proxy_fps);
    frame.textContent = `${frameIndex} · ${time.toFixed(3)}s`;
    row.append(frame);
    for (const target of labels()) {
      const cell = document.createElement("td");
      const candidates = state.manifest.candidates.filter((item) => (
        item.intended_target === target && Math.abs(item.frame.proxy_seconds - time) <= 1e-9
      ));
      const accepted = candidates.find((item) => item.human_accepted && !item.rejected);
      const preview = candidates.filter((item) => !item.rejected).at(-1);
      const rejected = candidates.filter((item) => item.rejected).at(-1);
      const pendingPrompt = pending().find((item) => (
        item.intended_target === target && Math.abs(item.frame.proxy_seconds - time) <= 1e-9
      ));
      const item = accepted || preview || (state.showRejected ? rejected : null);
      const isInitial = frameIndex === 0;
      const scheduled = accepted && (
        isInitial ? accepted.selected_for_finalization : accepted.selected_for_correction
      );
      const done = Boolean(accepted);
      const status = accepted
        ? "✓ Done"
        : preview
          ? "◐ Preview"
          : pendingPrompt
            ? "○ Editing"
            : rejected && state.showRejected
              ? "× Rejected"
              : "○ Not done";
      const wrapper = document.createElement("span");
      wrapper.className = `status-cell ${done ? "done" : preview || accepted ? "preview" : ""}${
        state.label === target && Math.abs(activeTimestamp() - time) <= 1e-9 ? " active" : ""}`;
      const select = document.createElement("button");
      select.className = "status-main";
      select.textContent = status;
      select.title = accepted
        ? `Accepted${scheduled ? isInitial ? " as the initial mask" : " and scheduled as the correction" : ""}. Show it on the canvas.`
        : item
          ? "Show this target and candidate on the canvas."
        : "Show this target and frame on the canvas.";
      select.onclick = () => {
        state.label = target;
        state.selected = pendingPrompt?.box_id || null;
        state.activeCandidateId = item?.candidate_id || null;
        loadFrame(time);
      };
      wrapper.append(select);
      if (accepted) {
        const clear = document.createElement("button");
        clear.className = "status-clear";
        clear.textContent = "×";
        clear.title = locked ? lockReason : "Unaccept this mask.";
        clear.disabled = locked;
        clear.onclick = (event) => {
          event.stopPropagation();
          unacceptCandidate(accepted);
        };
        wrapper.append(clear);
      } else if (preview) {
        const clear = document.createElement("button");
        clear.className = "status-clear";
        clear.textContent = "×";
        clear.title = locked ? lockReason : "Discard this preview.";
        clear.disabled = locked;
        clear.onclick = (event) => {
          event.stopPropagation();
          if (pendingPrompt) deletePrompt(pendingPrompt.box_id);
          else rejectCandidate(preview);
        };
        wrapper.append(clear);
      } else if (rejected && state.showRejected) {
        const restore = document.createElement("button");
        restore.className = "status-clear";
        restore.textContent = "↶";
        restore.title = locked ? lockReason : "Restore this rejected candidate.";
        restore.disabled = locked;
        restore.onclick = (event) => {
          event.stopPropagation();
          restoreCandidate(rejected);
        };
        wrapper.append(restore);
      }
      cell.append(wrapper);
      row.append(cell);
    }
    table.append(row);
  }
  host.replaceChildren(table);
}
function renderCandidates() {
  const signature = candidatesSignature();
  if (signature === candidatesRendered) return;
  candidatesRendered = signature;
  const parent = $("#candidates"); parent.replaceChildren();
  const locked = planLocked(), lockReason = planLockReason();
  const items = reviewedCandidates();
  const active = state.manifest.candidates.filter((item) => !item.rejected);
  const completedCells = new Set(
    active
      .filter((item) => item.human_accepted)
      .map((item) => `${item.frame.analysis_frame_index}:${item.intended_target}`),
  ).size;
  const totalCells = state.manifest.requested_proxy_timestamps_seconds.length * labels().length;
  setPanelSummary(
    "candidates",
    `${completedCells}/${totalCells} done`,
  );
  const statusHost = $("#candidate-status");
  if (statusHost) {
    parent.replaceChildren();
    renderCandidateStatusTable(statusHost, locked, lockReason);
    return;
  }
  if (!items.length) {
    const empty = document.createElement("p");
    empty.className = "hint";
    empty.textContent = state.showRejected
      ? "No rejected candidates."
      : "No active decoded prompts yet. Decode all pending prompts on a selected frame to review every target group.";
    parent.append(empty);
    return;
  }
  for (const item of items) {
    const isActive = item.candidate_id === activeCandidate()?.candidate_id;
    const isInitial = item.frame.analysis_frame_index === 0;
    const card = document.createElement("article");
    card.className = `candidate ${item.human_accepted ? "accepted" : ""}${item.rejected ? " rejected" : ""}${isActive ? " active-viewport" : ""} ${isInitial ? "initial-mask" : "later-correction"}`;
    const heading = document.createElement("strong");
    const policyPosition = targetPosition(item.intended_target);
    const targetLabel = policyPosition === Number.MAX_SAFE_INTEGER
      ? item.intended_target.replaceAll("_", " ")
      : `Target ${policyPosition + 1}/${labels().length}: ${item.intended_target.replaceAll("_", " ")}`;
    heading.textContent = `${targetLabel} · ${item.candidate_id}`;
    const metadata = document.createElement("small");
    metadata.className = "candidate-metadata";
    metadata.textContent = `${item.live_preview ? "Live preview · updates in place · " : ""}Frame ${
      item.frame.analysis_frame_index} (${item.frame.proxy_seconds.toFixed(3)} s) · ${
      isInitial ? "initial mask · required" : "later correction · optional"} · ${
      item.decoder_result.candidates.length} mask option(s)`;
    card.title = item.rejected
      ? "Rejected; retained for provenance."
      : locked
        ? lockReason
        : isInitial
          ? "Choose one initial mask."
          : "Choose a correction only when needed.";
    const options = document.createElement("div");
    options.className = "candidate-options";
    card.append(heading, metadata, options);
    if (item.rejected) {
      for (const candidate of item.decoder_result.candidates) {
        if (!candidate.review_uri) continue;
        const review = document.createElement("a");
        review.className = "review-link";
        review.href = `/artifacts/${candidate.review_uri}`;
        review.target = "_blank";
        review.rel = "noopener";
        review.textContent = `Open retained mask ${candidate.candidate_index} review`;
        options.append(review);
      }
      const restore = document.createElement("button");
      restore.textContent = "Undo rejection";
      restore.disabled = locked;
      restore.title = locked
        ? lockReason
        : "Return this candidate to normal review without accepting it.";
      restore.onclick = () => restoreCandidate(item);
      card.append(restore);
      parent.append(card);
      continue;
    }
    for (const candidate of item.decoder_result.candidates) {
      const option = document.createElement("section");
      option.className = `mask-option ${candidate.is_deterministic_best ? "model-best" : ""}`;
      const label = document.createElement("label"), radio = document.createElement("input");
      radio.type = "radio"; radio.name = item.candidate_id; radio.value = candidate.candidate_index;
      radio.checked = selectedCandidateIndex(item) === candidate.candidate_index;
      radio.disabled = locked;
      if (locked) radio.title = lockReason;
      radio.onchange = () => accept(item, candidate.candidate_index);
      const keyHint = isActive && candidate.candidate_index < 4
        ? ` · key ${candidate.candidate_index + 1}`
        : "";
      label.append(radio, document.createTextNode(` mask ${candidate.candidate_index}${keyHint} · IoU ${candidate.iou_score.toFixed(4)}${candidate.is_deterministic_best ? " · model best" : ""}`));
      option.append(label);
      if (candidate.review_uri) {
        const review = document.createElement("a");
        review.className = "review-link";
        review.href = `/artifacts/${candidate.review_uri}`;
        review.target = "_blank";
        review.rel = "noopener";
        review.title = "Open full-size context and padded-crop review";
        const image = new Image();
        image.src = `/artifacts/${candidate.review_uri}`;
        image.alt = `Mask ${candidate.candidate_index}: full-frame context and padded crop`;
        image.loading = "lazy";
        review.append(image);
        option.append(review);
      }
      options.append(option);
    }
    /* One eligibility control per card, matching the only role this frame can play:
     * frame 0 carries the required initial masks, every later keyframe carries optional
     * corrections. The backend refuses the other combination, so offering it here only
     * produced a dead control labelled with frame-0 language on later keyframes. */
    const eligible = document.createElement("label"), check = document.createElement("input");
    check.type = "checkbox";
    check.checked = isInitial ? item.selected_for_finalization : item.selected_for_correction;
    check.disabled = locked || !item.human_accepted;
    check.title = locked
      ? lockReason
      : item.human_accepted
        ? isInitial
          ? "Required: every configured target needs one frame-0 initial mask."
          : "Optional: add this keyframe to the correction schedule for this target."
        : "Choose one mask above before including this candidate in the plan.";
    check.onchange = () => (isInitial
      ? accept(item, item.human_selected_candidate_index, check.checked)
      : accept(item, item.human_selected_candidate_index, false, check.checked));
    eligible.append(check, document.createTextNode(
      isInitial ? " include in Initial masks (required at frame 0)" : " include in Later corrections (optional)"
    ));
    card.append(eligible);
    const actions = document.createElement("div");
    actions.className = "candidate-actions";
    const reject = document.createElement("button");
    reject.textContent = "Reject / Remove";
    reject.title = locked
      ? lockReason
      : item.human_accepted
        ? "Unaccept this current choice before rejecting it."
        : "Remove from normal review and planning while retaining its mask artifacts.";
    reject.disabled = locked || item.human_accepted;
    reject.onclick = () => rejectCandidate(item);
    actions.append(reject);
    if (item.human_accepted) {
      const unaccept = document.createElement("button");
      unaccept.textContent = "Unaccept";
      unaccept.disabled = locked;
      unaccept.title = locked
        ? lockReason
        : "Clear this selection and planning eligibility; required before rejection.";
      unaccept.onclick = () => unacceptCandidate(item);
      actions.append(unaccept);
    }
    card.append(actions);
    parent.append(card);
  }
}
async function accept(
  item, candidateIndex, eligible = item.selected_for_finalization,
  correction = item.selected_for_correction,
) {
  if (planLocked()) return setStatus(planLockReason(), true);
  if (candidateIndex === null || candidateIndex === undefined) return setStatus("Select a mask before marking eligibility.", true);
  state.label = item.intended_target;
  state.activeCandidateId = item.candidate_id;
  state.pendingCandidateSelections.set(item.candidate_id, candidateIndex);
  renderLabels(); renderCandidates(); renderActiveCandidate(); render();
  try {
    await api(`/api/candidates/${item.candidate_id}/accept`, {method: "POST", body: JSON.stringify({candidate_index: candidateIndex, eligible, correction})});
    state.pendingCandidateSelections.delete(item.candidate_id);
    await refresh();
  } catch (error) {
    state.pendingCandidateSelections.delete(item.candidate_id);
    renderCandidates(); renderActiveCandidate(); render();
    setStatus(error.message, true);
  }
}
async function unacceptCandidate(item) {
  try {
    await api(`/api/candidates/${item.candidate_id}/unaccept`, {method: "POST", body: "{}"});
    setStatus(`Unaccepted ${item.candidate_id}; it can now be rejected or reselected.`);
    await refresh();
  } catch (error) { setStatus(error.message, true); }
}
async function rejectCandidate(item) {
  try {
    await api(`/api/candidates/${item.candidate_id}/reject`, {method: "POST", body: "{}"});
    if (state.activeCandidateId === item.candidate_id) state.activeCandidateId = null;
    setStatus(`Rejected ${item.candidate_id}; mask artifacts were retained.`);
    await refresh();
  } catch (error) { setStatus(error.message, true); }
}
async function restoreCandidate(item) {
  try {
    await api(`/api/candidates/${item.candidate_id}/restore`, {method: "POST", body: "{}"});
    setStatus(`Restored ${item.candidate_id} to normal review.`);
    await refresh();
  } catch (error) { setStatus(error.message, true); }
}
async function deletePrompt(boxId) {
  try {
    await api(`/api/prompts/${boxId}`, {method: "DELETE"});
    if (state.selected === boxId) state.selected = null;
    setStatus("Discarded editable prompt and live preview.");
    await refresh();
  } catch (error) { setStatus(error.message, true); }
}
function renderEligible() {
  const parent = $("#eligible"); parent.replaceChildren();
  const initial = state.manifest.candidates.filter((item) => !item.rejected && item.selected_for_finalization);
  const later = state.manifest.candidates.filter((item) => !item.rejected && item.selected_for_correction);
  const summary = document.createElement("p");
  summary.textContent = `Initial masks: ${initial.length}${initial.length ? ` (${initial.map((item) => item.intended_target.replaceAll("_", " ")).join(", ")})` : ""}.`;
  const corrections = document.createElement("p");
  const targetCount = new Set(later.map((item) => item.intended_target)).size;
  corrections.textContent = `Later corrections: ${later.length} keyframe${later.length === 1 ? "" : "s"}${later.length ? ` across ${targetCount} target${targetCount === 1 ? "" : "s"}` : ""}.`;
  parent.append(summary, corrections);
  const keyframe = activeTimestamp();
  if (keyframe) {
    const here = later.filter((item) => Math.abs(item.frame.proxy_seconds - keyframe) <= 1e-9);
    const active = document.createElement("p");
    active.className = "hint";
    active.textContent = `At the active ${keyframe.toFixed(3)} s keyframe: ${here.length
      ? here.map((item) => item.intended_target.replaceAll("_", " ")).join(", ")
      : "no corrections marked yet"}.`;
    parent.append(active);
  }
  const limit = state.correctionPolicy?.maximum_later_correction_keyframes_per_target;
  const overLimit = limit === undefined || limit === null ? [] : [...new Set(later.map(
    (item) => item.intended_target,
  ))].filter((target) => later.filter((item) => item.intended_target === target).length > limit);
  if (overLimit.length) {
    const warning = document.createElement("p");
    warning.className = "hint warning";
    warning.textContent = `Over the policy limit of ${limit} later keyframe${limit === 1 ? "" : "s"} per target: ${
      overLimit.map((target) => target.replaceAll("_", " ")).join(", ")}.`;
    parent.append(warning);
  }
  const superseded = state.plan?.superseded_plans || [];
  if (superseded.length) {
    const history = document.createElement("p");
    history.className = "hint";
    history.textContent = `Kept unchanged on disk: ${superseded.map(
      (plan) => `revision ${plan.plan_revision}`,
    ).join(", ")}. Finalizing writes revision ${(state.plan?.plan_revision || 0) + 1} beside them.`;
    parent.append(history);
  }
  const unpolicedCorrections = Boolean(later.length) && !state.correctionPolicy;
  if (unpolicedCorrections) {
    const warning = document.createElement("p");
    warning.className = "hint warning";
    warning.textContent = "Later corrections need a correction policy before they can be finalized.";
    parent.append(warning);
  }
  setPanelSummary(
    "finalize",
    `${initial.length} initial · ${later.length} correction${later.length === 1 ? "" : "s"}${unpolicedCorrections ? " · needs policy" : ""}${overLimit.length ? " · over limit" : ""}`,
  );
  setPanelAttention("finalize", unpolicedCorrections || Boolean(overLimit.length));
}
async function loadFrame(time = Number($("#timestamp").value)) {
  try {
    const request = ++state.frameRequest;
    const preserveView = state.image !== null;
    const selectedLabel = state.label;
    setStatus("Decoding selected source frame…"); state.active = time;
    const response = await api(`/api/frame?timestamp=${encodeURIComponent(time)}`);
    if (request !== state.frameRequest) return;
    await api("/api/workspace", {method: "POST", body: JSON.stringify({timestamp: time})});
    if (request !== state.frameRequest) return;
    const image = new Image(); image.onload = () => {
      if (request !== state.frameRequest) return;
      state.image = image; state.viewRender = null; state.viewRequest = null;
      state.label = selectedLabel;
      $("#empty").hidden = true;
      requestAnimationFrame(() => {
        resizeCanvas({preserveCenter: preserveView, reset: !preserveView});
        render(); refresh(); requestViewRender();
      });
    };
    image.src = `/artifacts/${response.image_uri}?v=${Date.now()}`; state.frame = response.frame_index;
  } catch (error) { setStatus(error.message, true); }
}
async function saveBox(box, id = null) {
  const existing = id
    ? pending().find((item) => item.box_id === id)
    : activePendingPrompt();
  const boxId = existing?.box_id || null;
  const intendedTarget = existing?.intended_target || state.label;
  const path = boxId ? `/api/prompts/${boxId}` : "/api/prompts";
  const method = boxId ? "PATCH" : "POST";
  await api(path, {method, body: JSON.stringify({timestamp: state.active, intended_target: intendedTarget, pixel_box: box})});
  await refresh();
  scheduleLiveDecode(boxId || activePendingPrompt()?.box_id);
}
function clampPoint(point) {
  const {width, height} = modelSize();
  return {x: clamp(Math.round(point.x), 0, width), y: clamp(Math.round(point.y), 0, height)};
}
async function savePromptPoints(item, foreground, background) {
  await api(`/api/prompts/${item.box_id}`, {
    method: "PATCH",
    body: JSON.stringify({
      timestamp: item.frame.proxy_seconds,
      intended_target: item.intended_target,
      pixel_box: item.pixel_box,
      pixel_fg_points: foreground,
      pixel_bg_points: background,
    }),
  });
  await refresh();
  scheduleLiveDecode(item.box_id);
}
canvas.addEventListener("contextmenu", (event) => event.preventDefault());
canvas.addEventListener("pointerdown", async (event) => {
  if (!state.image) return;
  const point = screenPoint(event);
  const marker = hitPoint(point);
  if (event.button === 2) {
    if (!marker) return;
    const foreground = [...promptPoints(marker.item, "foreground")];
    const background = [...promptPoints(marker.item, "background")];
    (marker.kind === "foreground" ? foreground : background).splice(marker.index, 1);
    try {
      await savePromptPoints(marker.item, foreground, background);
      setStatus(`Removed ${marker.kind} point from ${marker.item.intended_target.replaceAll("_", " ")}.`);
    } catch (error) { setStatus(error.message, true); }
    return;
  }
  canvas.setPointerCapture(event.pointerId);
  if (event.button === 1 || event.altKey || event.shiftKey) { state.drag = {mode: "pan", point}; return; }
  if (state.promptMode !== "box") {
    const item = activePendingPrompt();
    if (!item) {
      setStatus(`Select one pending ${state.label.replaceAll("_", " ")} box before adding points.`, true);
      return;
    }
    state.selected = item.box_id;
    if (marker?.item.box_id === item.box_id) {
      state.drag = {...marker, mode: "point", start: world(point), previewPoint: marker.item[
        marker.kind === "foreground" ? "pixel_fg_points" : "pixel_bg_points"
      ][marker.index]};
      render();
      return;
    }
    const foreground = [...promptPoints(item, "foreground")];
    const background = [...promptPoints(item, "background")];
    (state.promptMode === "foreground" ? foreground : background).push(clampPoint(world(point)));
    try {
      await savePromptPoints(item, foreground, background);
      setStatus(`Added ${state.promptMode} point to ${item.intended_target.replaceAll("_", " ")}.`);
    } catch (error) { setStatus(error.message, true); }
    return;
  }
  const found = hit(point); state.selected = found?.box.box_id || null;
  state.drag = found ? {...found, mode: found.handle} : {mode: "draw", start: world(point), current: world(point)}; render();
});
canvas.addEventListener("pointermove", (event) => {
  if (!state.drag) return; const p = screenPoint(event), w = world(p), d = state.drag;
  if (d.mode === "pan") { state.pan.x += p.x - d.point.x; state.pan.y += p.y - d.point.y; d.point = p; }
  else if (d.mode === "draw") d.current = w;
  else if (d.mode === "point") d.previewPoint = clampPoint(w);
  else {
    const dx = w.x - d.start.x, dy = w.y - d.start.y;
    d.preview = d.mode === "move"
      ? moveBox(d.startBox, dx, dy)
      : {
        ...d.startBox,
        x2: clamp(Math.round(w.x), d.startBox.x1, modelSize().width),
        y2: clamp(Math.round(w.y), d.startBox.y1, modelSize().height),
      };
  }
  render();
});
canvas.addEventListener("pointerup", async () => {
  const d = state.drag; state.drag = null; if (!d || d.mode === "pan") return;
  if (d.mode === "point") {
    const foreground = [...promptPoints(d.item, "foreground")];
    const background = [...promptPoints(d.item, "background")];
    (d.kind === "foreground" ? foreground : background)[d.index] = d.previewPoint;
    try {
      await savePromptPoints(d.item, foreground, background);
      setStatus(`Moved ${d.kind} point.`);
    } catch (error) { setStatus(error.message, true); }
    return;
  }
  if (d.mode !== "draw" && !d.preview) { render(); return; }
  try {
    const b = d.mode === "draw"
      ? clampBox({x1: d.start.x, y1: d.start.y, x2: d.current.x, y2: d.current.y})
      : d.preview || d.startBox;
    if (b.x2 - b.x1 > 3 && b.y2 - b.y1 > 3) await saveBox(b, d.box?.box_id); else await refresh();
  } catch (error) { setStatus(error.message, true); await refresh(); }
});
canvas.addEventListener("pointercancel", () => { state.drag = null; render(); refresh(); });
canvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  const p = screenPoint(event), before = world(p), factor = event.deltaY < 0 ? 1.15 : 1 / 1.15;
  state.zoom = Math.min(8, Math.max(.4, state.zoom * factor));
  const scale = viewScale();
  state.pan = {x: p.x - before.x * scale, y: p.y - before.y * scale};
  render();
}, {passive: false});
if (typeof ResizeObserver !== "undefined") {
  new ResizeObserver(() => {
    if (!state.image) return;
    resizeCanvas({preserveCenter: true});
    render();
  }).observe(canvas);
}
document.addEventListener("keydown", async (event) => {
  const focused = document.activeElement;
  if (["INPUT", "TEXTAREA", "SELECT"].includes(focused?.tagName) || focused?.isContentEditable) return;
  if (/^[1-4]$/.test(event.key)) {
    const item = activeCandidate();
    const candidateIndex = Number(event.key) - 1;
    if (item && item.decoder_result.candidates.some((candidate) => candidate.candidate_index === candidateIndex)) {
      event.preventDefault();
      const isInitial = item.frame.analysis_frame_index === 0;
      await accept(item, candidateIndex, isInitial, !isInitial);
      return;
    }
  }
  const pointModes = {b: "box", f: "foreground", n: "background"};
  if (pointModes[event.key.toLowerCase()]) {
    event.preventDefault();
    setPromptMode(pointModes[event.key.toLowerCase()]);
    return;
  }
  const viewAid = VIEW_AID_SHORTCUTS[event.key.toLowerCase()];
  if (viewAid && state.viewFilters) {
    event.preventDefault();
    if (viewAid === "bypass") setViewAidsBypass(!state.viewAidsBypass);
    else if (viewAid === "reset") resetViewAids();
    else toggleViewAid(viewAid);
    return;
  }
  if (event.key === "Delete" && state.selected) await deletePrompt(state.selected);
});
$("#labels").onclick = (event) => { if (event.target.dataset.label) selectLabel(event.target.dataset.label); };
$("#custom-label").onchange = (event) => {
  if (event.target.value.trim() && !state.manualSeedTargets.length) {
    selectLabel(event.target.value.trim());
  }
};
const promptMode = $("#prompt-mode");
if (promptMode) promptMode.onclick = (event) => {
  if (event.target.dataset.promptMode) setPromptMode(event.target.dataset.promptMode);
};
const clearPoints = $("#clear-points");
if (clearPoints) clearPoints.onclick = async () => {
  const item = activePendingPrompt();
  if (!item) return;
  try {
    await savePromptPoints(item, [], []);
    setStatus(`Cleared corrective points for ${item.intended_target.replaceAll("_", " ")}.`);
  } catch (error) { setStatus(error.message, true); }
};
$("#load-frame").onclick = () => loadFrame();
$("#time-slider").oninput = (event) => loadFrame(state.manifest.requested_proxy_timestamps_seconds[Number(event.target.value)]);
$("#delete").onclick = async () => { if (state.selected) await deletePrompt(state.selected); };
function scheduleLiveDecode(boxId) {
  if (!state.liveDecode || !boxId || state.plan?.finalized) return;
  state.liveDecodeQueuedBoxId = boxId;
  if (liveDecodeTimer !== null) clearTimeout(liveDecodeTimer);
  liveDecodeTimer = setTimeout(() => {
    liveDecodeTimer = null;
    const queuedBoxId = state.liveDecodeQueuedBoxId;
    if (!state.liveDecode || !queuedBoxId || state.plan?.finalized) return;
    if (state.decodeInFlight) {
      scheduleLiveDecode(queuedBoxId);
      return;
    }
    if (!pending().some((item) => item.box_id === queuedBoxId)) {
      state.liveDecodeQueuedBoxId = null;
      return;
    }
    state.liveDecodeQueuedBoxId = null;
    decode([queuedBoxId], true);
  }, state.liveDecodeDelayMs);
}
async function decode(boxIds, livePreview = false) {
  if (!boxIds.length) return;
  if (!livePreview && state.liveDecode) {
    setStatus("Disable live decode before running a manual frame batch.", true);
    return;
  }
  if (state.decodeInFlight) {
    if (livePreview) scheduleLiveDecode(boxIds.at(-1));
    else setStatus("A decoder job is already running; wait for it to finish.", true);
    return;
  }
  state.decodeInFlight = true;
  syncDecodeControls();
  try {
    const result = await api("/api/decode", {
      method:"POST", body:JSON.stringify({box_ids: boxIds, live_preview: livePreview}),
    });
    setStatus(`${livePreview ? "Live preview" : "Decoder"} job ${result.job_id} queued; workspace remains responsive.`);
    let polling = false;
    const timer = setInterval(async () => {
      if (polling) return;
      polling = true;
      try {
        const job = await api(`/api/jobs/${result.job_id}`);
        if (job.status === "succeeded" || job.status === "failed") {
          clearInterval(timer);
          state.decodeInFlight = false;
          syncDecodeControls();
          if (job.status === "succeeded" && job.candidate_ids?.length) {
            for (const item of state.manifest.candidates) {
              if (!job.candidate_ids.includes(item.candidate_id)) continue;
              for (const candidate of item.decoder_result.candidates) {
                state.maskImages.delete(candidate.mask_uri);
                state.maskVersions.set(candidate.mask_uri, result.job_id);
              }
            }
            state.activeCandidateId = job.candidate_ids.at(-1);
          }
          setStatus(
            job.status === "succeeded"
              ? `${livePreview ? "Live preview" : "Decoder batch"} completed.`
              : job.error,
            job.status === "failed",
          );
          await refresh();
          if (state.liveDecodeQueuedBoxId) scheduleLiveDecode(state.liveDecodeQueuedBoxId);
        }
      } catch (error) {
        clearInterval(timer);
        state.decodeInFlight = false;
        syncDecodeControls();
        setStatus(`Could not read decoder status: ${error.message}`, true);
        if (state.liveDecodeQueuedBoxId) scheduleLiveDecode(state.liveDecodeQueuedBoxId);
      } finally {
        polling = false;
      }
    }, 500);
    await refresh();
  } catch (error) {
    state.decodeInFlight = false;
    syncDecodeControls();
    setStatus(error.message, true);
  }
}
$("#decode").onclick = () => decode(pending().filter((item) => item.frame.proxy_seconds === state.active).map((item) => item.box_id));
const toggleRejected = $("#toggle-rejected");
if (toggleRejected) toggleRejected.onclick = () => {
  state.showRejected = !state.showRejected;
  renderCandidates();
  const rejected = state.manifest.candidates.filter((item) => item.rejected).length;
  toggleRejected.textContent = state.showRejected
    ? `Hide rejected (${rejected})`
    : `Show rejected (${rejected})`;
};
const maskVisible = $("#mask-visible"), maskOpacity = $("#mask-opacity");
if (maskVisible) maskVisible.onchange = () => { state.maskVisible = maskVisible.checked; render(); };
if (maskOpacity) maskOpacity.oninput = () => {
  state.maskOpacity = Number(maskOpacity.value) / 100;
  $("#mask-opacity-value").textContent = `${maskOpacity.value}%`;
  render();
};
const liveDecode = $("#live-decode"), liveDecodeDelay = $("#live-decode-delay");
if (liveDecode) {
  state.liveDecode = globalThis.localStorage?.getItem(LIVE_DECODE_STORAGE_KEY) === "true";
  liveDecode.checked = state.liveDecode;
  liveDecode.onchange = () => {
    state.liveDecode = liveDecode.checked;
    globalThis.localStorage?.setItem(LIVE_DECODE_STORAGE_KEY, String(state.liveDecode));
    if (!state.liveDecode && liveDecodeTimer !== null) {
      clearTimeout(liveDecodeTimer);
      liveDecodeTimer = null;
      state.liveDecodeQueuedBoxId = null;
    } else if (state.liveDecode) {
      scheduleLiveDecode(activePendingPrompt()?.box_id);
    }
    syncDecodeControls();
  };
}
if (liveDecodeDelay) {
  const storedDelay = Number(globalThis.localStorage?.getItem(LIVE_DECODE_DELAY_STORAGE_KEY));
  if (Number.isFinite(storedDelay)) {
    state.liveDecodeDelayMs = Math.min(
      LIVE_DECODE_DELAY_MAX_MS,
      Math.max(LIVE_DECODE_DELAY_MIN_MS, Math.round(storedDelay)),
    );
  }
  liveDecodeDelay.value = String(state.liveDecodeDelayMs);
  liveDecodeDelay.onchange = () => {
    state.liveDecodeDelayMs = Math.min(
      LIVE_DECODE_DELAY_MAX_MS,
      Math.max(LIVE_DECODE_DELAY_MIN_MS, Math.round(Number(liveDecodeDelay.value) || 450)),
    );
    liveDecodeDelay.value = String(state.liveDecodeDelayMs);
    globalThis.localStorage?.setItem(
      LIVE_DECODE_DELAY_STORAGE_KEY,
      String(state.liveDecodeDelayMs),
    );
    if (state.liveDecodeQueuedBoxId) scheduleLiveDecode(state.liveDecodeQueuedBoxId);
  };
}
const viewAidsBypass = $("#view-aids-bypass"), viewAidsReset = $("#view-aids-reset");
if (viewAidsBypass) viewAidsBypass.onchange = () => setViewAidsBypass(viewAidsBypass.checked);
if (viewAidsReset) viewAidsReset.onclick = () => resetViewAids();
$("#finalize-plan").onclick = async () => {
  try {
    const plan = await api("/api/finalize-tracking-plan", {method: "POST", body: "{}"});
    const targets = plan.proposal.seeds.map((seed) => seed.intended_target.replaceAll("_", " ")).join(", ");
    const later = plan.correction_schedule?.corrections.filter(
      (correction) => correction.frame.analysis_frame_index !== 0
    ) || [];
    const laterTargetCount = new Set(later.map((correction) => correction.target_id)).size;
    const schedule = `; ${later.length} later keyframe${later.length === 1 ? "" : "s"} across ${laterTargetCount} target${laterTargetCount === 1 ? "" : "s"}`;
    setStatus(`Finalized ${plan.proposal.seeds.length} initial mask(s) for ${targets}${schedule}; tracking was not launched.`);
    await refresh();
  } catch (error) { setStatus(error.message, true); }
};
registerPanels();
refresh()
  .then(() => loadFrame(activeTimestamp()))
  .then(() => loadViewFilters())
  .catch((error) => setStatus(error.message, true));
