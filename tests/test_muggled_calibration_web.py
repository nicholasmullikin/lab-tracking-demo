from __future__ import annotations

import argparse
import base64
import json
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from conftest import require_executable, serve

from battle.muggled_calibration import (
    build_manifest,
    finalize_prompt,
    load_correction_policy,
)
from battle.muggled_calibration_web import (
    CANDIDATE_REVIEW_LOCK_REASON,
    WorkerClient,
    Workspace,
)
from battle.muggled_calibration_web import make_workspace as create_workspace
from battle.muggled_smoke import load_manual_seed_target_config, sha256_file


class FixtureDecoder:
    """No-GPU decoder fixture that exercises persistence and queue state."""

    last_batch_payload: dict[str, object] | None = None
    candidate_count: int = 1

    def request(
        self, command: str, payload: dict[str, object], timeout: float = 120
    ) -> dict[str, object]:
        del timeout
        if command == "frame_preview":
            return {
                "frame_index": payload["frame_index"],
                "image_uri": "results/frames/fixture.jpg",
            }
        if command == "batch_decode":
            time.sleep(0.02)
            self.last_batch_payload = payload
            decoded = []
            for prompt in payload["prompts"]:
                candidate_id = prompt["candidate_id"]
                decoded.append(
                    {
                        "box_id": prompt["box_id"],
                        "candidate_id": candidate_id,
                        "decoder_result": {
                            "api": "muggledsam_sam3_interactive",
                            "candidate_count": self.candidate_count,
                            "deterministic_best_candidate_index": 0,
                            "candidates": [
                                {
                                    "candidate_index": index,
                                    "iou_score": 0.5 - index / 100,
                                    "mask_uri": (
                                        f"results/masks/{candidate_id}_candidate-{index:02d}.png"
                                    ),
                                    "review_uri": (
                                        f"results/{candidate_id}_candidate-{index:02d}_review.png"
                                    ),
                                    "is_deterministic_best": index == 0,
                                }
                                for index in range(self.candidate_count)
                            ],
                            "overlay_uri": f"results/{candidate_id}_overlay.png",
                        },
                    }
                )
            return {"decoded": decoded}
        if command == "render_final_selected_seed_review":
            return {"output_path": payload["output_path"]}
        raise RuntimeError(f"unexpected command {command}")

    def close(self) -> None:
        return


def make_workspace(tmp_path: Path) -> Workspace:
    root = Path(__file__).parents[1]
    manifest = build_manifest(
        repository_root=root,
        config_path=root / "configs/clips/assembly101_nusar_9033_ego_viewpoint_screen_g2.json",
        timestamps=(0.0, 10.0),
        result_directory=tmp_path / "results",
        calibration_id="muggledsam-sam3-e4-web-calibration-test",
    )
    manifest_path = tmp_path / "calibration_manifest.json"
    return Workspace(
        repository_root=root,
        manifest_path=manifest_path,
        manifest=manifest,
        decoder=FixtureDecoder(),
    )


def make_configured_workspace(tmp_path: Path) -> Workspace:
    workspace = make_workspace(tmp_path)
    root = workspace.repository_root
    target_config_path = (
        root / "configs/"
        "muggledsam_e4_left_hand_right_hand_yellow_toy_top_black_toy_top_base_manual_seed.json"
    )
    workspace.manual_seed_target_config = load_manual_seed_target_config(
        target_config_path=target_config_path,
        repository_root=root,
        g2_config_path=root / "configs/clips/assembly101_nusar_9033_ego_viewpoint_screen_g2.json",
        view_id=workspace.manifest.view_id,
    )
    workspace.manual_seed_target_config_path = target_config_path
    return workspace


def make_correction_workspace(tmp_path: Path) -> Workspace:
    workspace = make_configured_workspace(tmp_path)
    policy_path = (
        workspace.repository_root
        / "configs/muggledsam_e4_four_target_keyframe_correction_policy.json"
    )
    workspace.correction_policy = load_correction_policy(
        correction_policy_path=policy_path,
        manual_seed_target_config_path=workspace.manual_seed_target_config_path,
        repository_root=workspace.repository_root,
        g2_config_path=workspace.repository_root
        / "configs/clips/assembly101_nusar_9033_ego_viewpoint_screen_g2.json",
        view_id=workspace.manifest.view_id,
    )
    workspace.correction_policy_path = policy_path
    return workspace


def decode_candidates(workspace: Workspace, prompts: list[tuple[float, str]]) -> tuple[object, ...]:
    boxes = [
        workspace.add_or_update_prompt(
            {
                "timestamp": timestamp,
                "intended_target": target,
                "pixel_box": {"x1": 10 + index, "y1": 20, "x2": 100, "y2": 120},
            }
        )
        for index, (timestamp, target) in enumerate(prompts)
    ]
    job_id = workspace.queue_decode([box["box_id"] for box in boxes])
    workspace.jobs[job_id]["future"].result(timeout=2)
    return workspace.manifest.candidates


def test_live_preview_retains_prompt_and_replaces_logical_preview_with_immutable_artifacts(
    tmp_path: Path,
) -> None:
    workspace = make_workspace(tmp_path)
    prompt = workspace.add_or_update_prompt(
        {
            "timestamp": 0.0,
            "intended_target": "left_hand",
            "pixel_box": {"x1": 10, "y1": 20, "x2": 100, "y2": 120},
        }
    )

    first_job = workspace.queue_decode([prompt["box_id"]], live_preview=True)
    workspace.jobs[first_job]["future"].result(timeout=2)
    first_candidate = workspace.manifest.candidates[0]

    assert workspace.manifest.workspace.pending_boxes[0].stage == "pending"
    assert workspace.manifest.workspace.pending_boxes[0].box_id == prompt["box_id"]
    assert first_candidate.live_preview
    assert first_candidate.source_box_id == prompt["box_id"]

    second_job = workspace.queue_decode([prompt["box_id"]], live_preview=True)
    workspace.jobs[second_job]["future"].result(timeout=2)

    assert len(workspace.manifest.candidates) == 1
    second_candidate = workspace.manifest.candidates[0]
    assert second_candidate.candidate_id != first_candidate.candidate_id
    assert workspace.jobs[second_job]["candidate_ids"] == [
        workspace.manifest.candidates[0].candidate_id
    ]

    workspace.accept_candidate(second_candidate.candidate_id, 0, eligible=False)
    third_job = workspace.queue_decode([prompt["box_id"]], live_preview=True)
    workspace.jobs[third_job]["future"].result(timeout=2)

    assert len(workspace.manifest.candidates) == 2
    assert workspace.manifest.candidates[0].candidate_id == second_candidate.candidate_id
    assert workspace.manifest.candidates[0].human_accepted
    assert workspace.manifest.candidates[1].candidate_id != second_candidate.candidate_id


def test_one_editable_prompt_per_target_and_frame_updates_in_place(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    first = workspace.add_or_update_prompt(
        {
            "timestamp": 0.0,
            "intended_target": "left_hand",
            "pixel_box": {"x1": 10, "y1": 20, "x2": 100, "y2": 120},
        }
    )
    updated = workspace.add_or_update_prompt(
        {
            "timestamp": 0.0,
            "intended_target": "left_hand",
            "pixel_box": {"x1": 30, "y1": 40, "x2": 130, "y2": 140},
        }
    )

    assert updated["box_id"] == first["box_id"]
    assert len(workspace.manifest.workspace.pending_boxes) == 1
    assert workspace.manifest.workspace.pending_boxes[0].pixel_box.x1 == 30

    other_target = workspace.add_or_update_prompt(
        {
            "timestamp": 0.0,
            "intended_target": "right_hand",
            "pixel_box": {"x1": 10, "y1": 20, "x2": 100, "y2": 120},
        }
    )
    later_frame = workspace.add_or_update_prompt(
        {
            "timestamp": 10.0,
            "intended_target": "left_hand",
            "pixel_box": {"x1": 10, "y1": 20, "x2": 100, "y2": 120},
        }
    )

    assert other_target["box_id"] != first["box_id"]
    assert later_frame["box_id"] != first["box_id"]
    assert len(workspace.manifest.workspace.pending_boxes) == 3


def test_accepting_a_new_mask_replaces_the_prior_choice_for_that_target_frame(
    tmp_path: Path,
) -> None:
    workspace = make_workspace(tmp_path)
    first = decode_candidates(workspace, [(0.0, "left_hand")])[-1]
    second = decode_candidates(workspace, [(0.0, "left_hand")])[-1]

    workspace.accept_candidate(first.candidate_id, 0, eligible=True)
    workspace.accept_candidate(second.candidate_id, 0, eligible=True)

    by_id = {candidate.candidate_id: candidate for candidate in workspace.manifest.candidates}
    assert not by_id[first.candidate_id].human_accepted
    assert not by_id[first.candidate_id].selected_for_finalization
    assert by_id[second.candidate_id].human_accepted
    assert by_id[second.candidate_id].selected_for_finalization


def test_editing_an_accepted_live_prompt_invalidates_its_stale_choice(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    prompt = workspace.add_or_update_prompt(
        {
            "timestamp": 0.0,
            "intended_target": "left_hand",
            "pixel_box": {"x1": 10, "y1": 20, "x2": 100, "y2": 120},
        }
    )
    job_id = workspace.queue_decode([prompt["box_id"]], live_preview=True)
    workspace.jobs[job_id]["future"].result(timeout=2)
    candidate = workspace.manifest.candidates[0]
    workspace.accept_candidate(candidate.candidate_id, 0, eligible=True)

    workspace.add_or_update_prompt(
        {
            "timestamp": 0.0,
            "intended_target": "left_hand",
            "pixel_box": {"x1": 20, "y1": 30, "x2": 110, "y2": 130},
        },
        box_id=prompt["box_id"],
    )

    edited = workspace.manifest.candidates[0]
    assert not edited.human_accepted
    assert edited.human_selected_candidate_index is None
    assert not edited.selected_for_finalization


@pytest.mark.slow
def test_completed_live_decode_job_history_is_bounded(tmp_path: Path) -> None:
    """Queues 40 decode jobs through the fixture decoder's per-batch sleep."""
    workspace = make_workspace(tmp_path)
    prompt = workspace.add_or_update_prompt(
        {
            "timestamp": 0.0,
            "intended_target": "left_hand",
            "pixel_box": {"x1": 10, "y1": 20, "x2": 100, "y2": 120},
        }
    )

    for _ in range(40):
        job_id = workspace.queue_decode([prompt["box_id"]], live_preview=True)
        workspace.jobs[job_id]["future"].result(timeout=2)

    assert len(workspace.jobs) <= 32
    assert job_id in workspace.jobs


def test_tracking_plan_cannot_finalize_during_an_active_decode(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    workspace.jobs["active"] = {
        "status": "running",
        "box_ids": [],
        "live_preview": True,
    }

    with pytest.raises(ValueError, match="active decoder job"):
        workspace.finalize_tracking_plan()


def test_failed_live_decode_restores_editable_prompt_state(tmp_path: Path) -> None:
    class FailingDecoder(FixtureDecoder):
        def request(
            self, command: str, payload: dict[str, object], timeout: float = 120
        ) -> dict[str, object]:
            if command == "batch_decode":
                raise OSError("fixture decode failed")
            return super().request(command, payload, timeout)

    workspace = make_workspace(tmp_path)
    workspace.decoder = FailingDecoder()
    prompt = workspace.add_or_update_prompt(
        {
            "timestamp": 0.0,
            "intended_target": "left_hand",
            "pixel_box": {"x1": 10, "y1": 20, "x2": 100, "y2": 120},
        }
    )

    job_id = workspace.queue_decode([prompt["box_id"]], live_preview=True)
    workspace.jobs[job_id]["future"].result(timeout=2)

    assert workspace.jobs[job_id]["status"] == "failed"
    assert workspace.manifest.workspace.pending_boxes[0].stage == "pending"


def test_live_decode_discards_a_stale_prompt_revision(tmp_path: Path) -> None:
    class BlockingDecoder(FixtureDecoder):
        def __init__(self) -> None:
            self.started = threading.Event()
            self.release = threading.Event()

        def request(
            self, command: str, payload: dict[str, object], timeout: float = 120
        ) -> dict[str, object]:
            if command == "batch_decode":
                self.started.set()
                assert self.release.wait(timeout=2)
            return super().request(command, payload, timeout)

    workspace = make_workspace(tmp_path)
    decoder = BlockingDecoder()
    workspace.decoder = decoder
    prompt = workspace.add_or_update_prompt(
        {
            "timestamp": 0.0,
            "intended_target": "left_hand",
            "pixel_box": {"x1": 10, "y1": 20, "x2": 100, "y2": 120},
        }
    )
    job_id = workspace.queue_decode([prompt["box_id"]], live_preview=True)
    assert decoder.started.wait(timeout=2)

    workspace.add_or_update_prompt(
        {
            "timestamp": 0.0,
            "intended_target": "left_hand",
            "pixel_box": {"x1": 20, "y1": 30, "x2": 110, "y2": 130},
        },
        box_id=prompt["box_id"],
    )
    decoder.release.set()
    workspace.jobs[job_id]["future"].result(timeout=2)

    assert workspace.jobs[job_id]["stale_box_ids"] == [prompt["box_id"]]
    assert not workspace.manifest.candidates
    assert workspace.manifest.workspace.pending_boxes[0].stage == "pending"
    assert workspace.manifest.workspace.pending_boxes[0].pixel_box.x1 == 20


def test_deleting_an_editable_prompt_discards_its_unaccepted_live_preview(
    tmp_path: Path,
) -> None:
    workspace = make_workspace(tmp_path)
    prompt = workspace.add_or_update_prompt(
        {
            "timestamp": 0.0,
            "intended_target": "left_hand",
            "pixel_box": {"x1": 10, "y1": 20, "x2": 100, "y2": 120},
        }
    )
    job_id = workspace.queue_decode([prompt["box_id"]], live_preview=True)
    workspace.jobs[job_id]["future"].result(timeout=2)

    workspace.delete_prompt(prompt["box_id"])

    assert not workspace.manifest.workspace.pending_boxes
    assert not workspace.manifest.candidates


def write_selected_masks(workspace: Workspace) -> None:
    for candidate in workspace.manifest.candidates:
        selected = next(
            item
            for item in candidate.decoder_result.candidates
            if item.candidate_index == candidate.human_selected_candidate_index
        )
        path = workspace.manifest_path.parent / selected.mask_uri
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(candidate.candidate_id.encode())


def post_json(base: str, path: str, body: dict[str, object]) -> dict[str, object]:
    request = urllib.request.Request(
        f"{base}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return json.loads(urllib.request.urlopen(request).read())


def click_create_proposal_in_headless_dom(base: str) -> bool:
    """Run the served UI's proposal click handler against the fixture-only HTTP server."""
    node = shutil.which("node")
    if node is None:
        return False
    script = """
const vm = require("node:vm");
const base = process.argv[1];
const nativeFetch = globalThis.fetch;

class Element {
  constructor() {
    this.children = [];
    this.classList = {toggle() {}};
    this.dataset = {};
    this.style = {};
    this.value = "";
    this.checked = false;
  }
  addEventListener() {}
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  getContext() {
    return {
      clearRect() {}, drawImage() {}, fillRect() {}, fillText() {},
      restore() {}, save() {}, scale() {}, strokeRect() {}, translate() {},
    };
  }
  querySelectorAll(selector) {
    const descendants = (nodes) => nodes.flatMap((node) => (
      node instanceof Element ? [node, ...descendants(node.children)] : []
    ));
    return selector === "input:checked"
      ? descendants(this.children).filter((node) => node.checked)
      : [];
  }
}

const selectors = [
  "#canvas", "#status", "#labels", "#time-slider", "#timestamp", "#filmstrip",
  "#table", "#diff", "#candidate-status", "#eligible", "#empty", "#plan-lock",
  "#custom-label", "#custom-label-control", "#load-frame", "#delete", "#decode",
  "#finalize-plan", "#active-candidate", "#label-panel", "#prompt-clicks-panel",
      "#prompt-mode", "#clear-points", "#live-decode", "#live-decode-delay", "#frame-mode",
];
const elements = Object.fromEntries(selectors.map((selector) => [selector, new Element()]));
elements["#canvas"].width = 954;
elements["#canvas"].height = 720;
globalThis.document = {
  activeElement: {tagName: "BODY"},
  addEventListener() {},
  createElement() { return new Element(); },
  createTextNode(value) { return String(value); },
  querySelector(selector) { return elements[selector]; },
};
globalThis.Image = class Image {};
globalThis.fetch = (path, options) => nativeFetch(base + path, options);

(async () => {
  const app = await (await nativeFetch(base + "/static/app.js")).text();
  vm.runInThisContext(app, {filename: "calibration-app.js"});
  await new Promise((resolve) => setTimeout(resolve, 20));
  await elements["#finalize-plan"].onclick();
  if (elements["#status"].textContent !==
      "Ready — edits autosave atomically.") {
    throw new Error(`unexpected proposal status: ${elements["#status"].textContent}`);
  }
})().catch((error) => {
  console.error(error.stack || error);
  process.exitCode = 1;
});
"""
    completed = subprocess.run(
        [node, "-e", script, base],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    return True


def test_calibration_ui_exposes_one_tracking_plan_action() -> None:
    markup = (Path(__file__).parents[1] / "src/battle/static/calibration/index.html").read_text()

    assert "initial masks" in markup.lower()
    assert "later corrections" in markup.lower()
    assert markup.count("Finalize tracking plan") == 2
    assert "Create frame-0 proposal" not in markup
    assert "Create correction schedule" not in markup
    assert 'id="decode-selected"' not in markup
    assert 'id="duplicate"' not in markup
    assert 'id="toggle-rejected"' not in markup
    assert 'data-panel="prompts"' not in markup
    assert 'id="candidate-status"' in markup
    assert 'id="candidates"' not in markup
    assert 'id="live-decode"' in markup
    assert 'id="live-decode-delay"' in markup
    assert "Automatically preview" in markup
    assert "Wait time after the last edit" in markup
    assert 'id="mask-opacity" type="range" min="0" max="100"' in markup


def test_calibration_canvas_maps_css_and_dpr_to_model_pixels() -> None:
    """Exercise move and zoom without a browser or a real calibration workspace."""
    node = require_executable("node", "the calibration canvas regression test")
    root = Path(__file__).parents[1]
    script = r"""
const fs = require("node:fs");
const app = fs.readFileSync(process.argv[1], "utf8");
const listeners = {};
const calls = [];

class Element {
  constructor(selector = "") {
    this.selector = selector;
    this.children = [];
    this.classList = {toggle() {}};
    this.dataset = {};
    this.style = {};
    this.value = "";
    this.checked = false;
    this.hidden = false;
    this.width = 954;
    this.height = 720;
  }
  addEventListener(name, handler) { (listeners[this.selector] ||= {})[name] = handler; }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  getBoundingClientRect() { return {left: 50, top: 20, width: 960, height: 540}; }
  setPointerCapture() {}
  getContext() {
    return {
      clearRect() {}, drawImage() {}, fillRect() {}, fillText() {}, restore() {}, save() {},
      strokeRect() {}, translate() {},
      setTransform(...args) { calls.push(["transform", ...args]); },
      scale(...args) { calls.push(["scale", ...args]); },
    };
  }
  querySelectorAll(selector) {
    const descendants = (nodes) => nodes.flatMap((node) => (
      node instanceof Element ? [node, ...descendants(node.children)] : []
    ));
    return selector === "input:checked"
      ? descendants(this.children).filter((node) => node.checked)
      : [];
  }
}

const selectors = [
  "#canvas", "#status", "#labels", "#time-slider", "#timestamp", "#filmstrip",
  "#table", "#diff", "#candidate-status", "#eligible", "#empty", "#plan-lock",
  "#custom-label", "#custom-label-control", "#load-frame", "#delete", "#decode",
  "#finalize-plan", "#active-candidate", "#label-panel", "#prompt-clicks-panel",
      "#prompt-mode", "#clear-points", "#live-decode", "#live-decode-delay", "#frame-mode",
      "#enable-labeling",
];
const elements = Object.fromEntries(selectors.map((selector) => [selector, new Element(selector)]));
globalThis.document = {
  activeElement: {tagName: "BODY"},
  addEventListener() {},
  createElement() { return new Element(); },
  createTextNode(value) { return String(value); },
  querySelector(selector) { return elements[selector]; },
};
globalThis.devicePixelRatio = 2;
globalThis.requestAnimationFrame = (callback) => callback();
globalThis.Image = class Image {
  naturalWidth = 1920;
  naturalHeight = 1080;
  set src(_value) { this.onload(); }
};
const manifest = {
  proxy_dimensions: {width: 1920, height: 1080},
      proxy_fps: 30,
      proxy_frame_count: 5400,
  requested_proxy_timestamps_seconds: [0, 1],
  workspace: {
    active_proxy_timestamp_seconds: 0,
    pending_boxes: [{
      box_id: "p0", intended_target: "left_hand", stage: "pending",
      frame: {proxy_seconds: 0}, pixel_box: {x1: 100, y1: 100, x2: 300, y2: 300},
    }],
  },
  candidates: [],
};
globalThis.fetch = async (path, options = {}) => {
      if (path === "/api/state" || path === "/api/workspace") {
        if (path === "/api/workspace") calls.push(["workspace", JSON.parse(options.body)]);
    return {ok: true, status: 200, json: async () => ({
      manifest, manual_seed_targets: [], manual_seed_target_policy: null,
      last_diff: [], worker_online: true,
    })};
  }
  if (path.startsWith("/api/frame")) {
        const timestamp = Number(new URL(path, "http://localhost").searchParams.get("timestamp"));
        return {ok: true, status: 200, json: async () => ({
          frame_index: Math.round(timestamp * 30), image_uri: "frame.jpg",
        })};
  }
      if (path === "/api/calibration-frames") {
        const timestamp = JSON.parse(options.body).timestamp;
        const frameIndex = Math.round(timestamp * 30);
        manifest.requested_proxy_timestamps_seconds.push(frameIndex / 30);
        manifest.requested_proxy_timestamps_seconds.sort((left, right) => left - right);
        return {ok: true, status: 201, json: async () => ({
          frame_index: frameIndex, proxy_seconds: frameIndex / 30,
        })};
      }
  if (path.startsWith("/api/prompts")) {
    calls.push(["prompt", JSON.parse(options.body)]);
    return {ok: true, status: 201, json: async () => ({})};
  }
  throw new Error(`unexpected request: ${path}`);
};

const tick = () => new Promise((resolve) => setTimeout(resolve, 0));
(async () => {
  eval(app);
  await tick(); await tick(); await tick();
  const canvas = elements["#canvas"];
  if (canvas.width !== 1920 || canvas.height !== 1080) {
    throw new Error(
      `backing size was ${canvas.width}x${canvas.height}, expected DPR-scaled 1920x1080`
    );
  }
  const down = listeners["#canvas"].pointerdown;
  const move = listeners["#canvas"].pointermove;
  const up = listeners["#canvas"].pointerup;
  down({pointerId: 1, button: 0, clientX: 100, clientY: 70});
  move({clientX: 150, clientY: 70});
  move({clientX: 250, clientY: 70});
  await up({});
  const saved = calls.find((entry) => entry[0] === "prompt")?.[1]?.pixel_box;
  if (JSON.stringify(saved) !== JSON.stringify({x1: 400, y1: 100, x2: 600, y2: 300})) {
    throw new Error(`incorrect model-pixel move: ${JSON.stringify(saved)}`);
  }
  setPromptMode("foreground");
  await down({pointerId: 2, button: 0, clientX: 550, clientY: 270});
  const pointPayload = calls.filter((entry) => entry[0] === "prompt").at(-1)?.[1];
  if (JSON.stringify(pointPayload.pixel_fg_points) !== JSON.stringify([{x: 1000, y: 500}]) ||
      !Array.isArray(pointPayload.pixel_bg_points) || pointPayload.pixel_bg_points.length !== 0) {
    throw new Error(`incorrect model-pixel point: ${JSON.stringify(pointPayload)}`);
  }
  listeners["#canvas"].wheel({
    clientX: 530, clientY: 290, deltaY: -1, preventDefault() {},
  });
  if (!calls.some((entry) => entry[0] === "transform" && entry[1] === 2) ||
      !calls.some((entry) => entry[0] === "scale" && entry[1] === 0.575)) {
    throw new Error(`expected DPR transform and visible 1.15x zoom: ${JSON.stringify(calls)}`);
  }
  selectLabel("right_hand");
  calls.length = 0;
  await elements["#filmstrip"].children[1].onclick();
  await tick(); await tick();
  if (!calls.some((entry) => entry[0] === "scale" && entry[1] === 0.575)) {
    throw new Error(`frame switch reset zoom: ${JSON.stringify(calls)}`);
  }
  setPromptMode("box");
  down({pointerId: 3, button: 0, clientX: 100, clientY: 70});
  move({clientX: 200, clientY: 170});
  await up({});
  const switchedPrompt = calls.filter((entry) => entry[0] === "prompt").at(-1)?.[1];
  if (switchedPrompt?.intended_target !== "right_hand") {
    throw new Error(`frame switch reset label: ${JSON.stringify(switchedPrompt)}`);
  }
      const promptCount = calls.filter((entry) => entry[0] === "prompt").length;
      const workspaceCount = calls.filter((entry) => entry[0] === "workspace").length;
      elements["#time-slider"].value = "15";
      elements["#time-slider"].onchange({target: elements["#time-slider"]});
      await tick(); await tick();
      if (Number(elements["#timestamp"].value) !== 0.5) {
        throw new Error(`slider did not map frame 15 to 0.5 s: ${elements["#timestamp"].value}`);
      }
      if (elements["#frame-mode"].textContent !== "Browse only") {
        throw new Error("arbitrary slider frame was not marked browse-only");
      }
      await down({pointerId: 4, button: 0, clientX: 100, clientY: 70});
      move({clientX: 200, clientY: 170});
      await up({});
      if (calls.filter((entry) => entry[0] === "prompt").length !== promptCount) {
        throw new Error("browse-only frame created a prompt");
      }
      if (calls.filter((entry) => entry[0] === "workspace").length !== workspaceCount) {
        throw new Error("browse-only frame changed persisted workspace state");
      }
      if (elements["#enable-labeling"].hidden) {
        throw new Error("browse-only frame did not offer promotion to calibration");
      }
      await elements["#enable-labeling"].onclick();
      if (elements["#frame-mode"].textContent !== "Calibration frame") {
        throw new Error("promoted frame remained browse-only");
      }
      await down({pointerId: 5, button: 0, clientX: 100, clientY: 70});
      move({clientX: 200, clientY: 170});
      await up({});
      if (calls.filter((entry) => entry[0] === "prompt").length !== promptCount + 1) {
        throw new Error("promoted calibration frame did not allow a prompt");
      }
})().catch((error) => {
  console.error(error.stack || error);
  process.exitCode = 1;
});
"""
    completed = subprocess.run(
        [node, "-e", script, str(root / "src/battle/static/calibration/app.js")],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr


def test_point_prompts_persist_normalized_and_reach_the_decoder(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    try:
        prompt = workspace.add_or_update_prompt(
            {
                "timestamp": 0,
                "intended_target": "right_hand",
                "pixel_box": {"x1": 100, "y1": 120, "x2": 300, "y2": 400},
                "pixel_fg_points": [{"x": 150, "y": 200}],
                "pixel_bg_points": [{"x": 350, "y": 220}],
            }
        )
        persisted = json.loads(workspace.manifest_path.read_text())["workspace"]["pending_boxes"][0]
        assert persisted["pixel_fg_points"] == [{"schema_version": "1.0", "x": 150, "y": 200}]
        assert persisted["normalized_fg_points"][0]["x"] == pytest.approx(150 / 954)
        assert persisted["normalized_bg_points"][0]["y"] == pytest.approx(220 / 720)

        job_id = workspace.queue_decode([prompt["box_id"]])
        workspace.jobs[job_id]["future"].result(timeout=2)
        payload = workspace.decoder.last_batch_payload  # type: ignore[union-attr]
        assert payload == {
            "prompts": [
                {
                    "box_id": prompt["box_id"],
                    "candidate_id": "t000000-b01",
                    "frame_index": 0,
                    "pixel_box": {
                        "schema_version": "1.0",
                        "x1": 100,
                        "y1": 120,
                        "x2": 300,
                        "y2": 400,
                    },
                    "intended_target": "right_hand",
                    "boxes": [[[100 / 954, 120 / 720], [300 / 954, 400 / 720]]],
                    "fg_points": [[150 / 954, 200 / 720]],
                    "bg_points": [[350 / 954, 220 / 720]],
                }
            ]
        }
        candidate = workspace.manifest.candidates[0]
        assert candidate.pixel_bg_points[0].x == 350
        assert candidate.normalized_fg_points[0].y == pytest.approx(200 / 720)
    finally:
        workspace.close()


def test_point_prompt_provenance_survives_finalization_and_locks_rejection(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    try:
        prompt = workspace.add_or_update_prompt(
            {
                "timestamp": 0,
                "intended_target": "left_hand",
                "pixel_box": {"x1": 10, "y1": 20, "x2": 100, "y2": 120},
                "pixel_fg_points": [{"x": 40, "y": 50}],
                "pixel_bg_points": [{"x": 105, "y": 50}],
            }
        )
        job_id = workspace.queue_decode([prompt["box_id"]])
        workspace.jobs[job_id]["future"].result(timeout=2)
        candidate = workspace.manifest.candidates[0]
        workspace.accept_candidate(candidate.candidate_id, 0, eligible=True)

        proposal = workspace.create_proposal([candidate.candidate_id])

        assert proposal["seeds"][0]["pixel_fg_points"][0]["x"] == 40
        assert proposal["seeds"][0]["normalized_bg_points"][0]["x"] == pytest.approx(105 / 954)
        with pytest.raises(ValueError, match="candidate review is locked"):
            workspace.reject_candidate(candidate.candidate_id)
    finally:
        workspace.close()


def test_calibration_binary_mask_overlay_only_tints_positive_pixels() -> None:
    """A decoded grayscale zero pixel must not receive the selected target tint."""
    node = require_executable("node", "the calibration canvas regression test")
    root = Path(__file__).parents[1]
    script = r"""
const fs = require("node:fs");
const app = fs.readFileSync(process.argv[1], "utf8");
let writtenPixels = null;

class Element {
  constructor(selector = "") {
    this.selector = selector; this.children = []; this.dataset = {}; this.style = {};
    this.value = ""; this.checked = false; this.disabled = false; this.hidden = false;
    this.width = 2; this.height = 1; this.classList = {toggle() {}};
  }
  addEventListener() {}
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  getBoundingClientRect() { return {left: 0, top: 0, width: 2, height: 1}; }
  getContext() {
    return {
      clearRect() {}, drawImage() {}, fillRect() {}, fillText() {}, restore() {}, save() {},
      scale() {}, strokeRect() {}, translate() {}, setTransform() {},
      getImageData() {
        // Canvas decodes a grayscale PNG's zero and 255 samples as opaque RGB pixels.
        return {data: new Uint8ClampedArray([0, 0, 0, 255, 255, 255, 255, 255])};
      },
      putImageData(imageData) { writtenPixels = [...imageData.data]; },
    };
  }
  querySelectorAll() { return []; }
}
const selectors = [
  "#canvas", "#status", "#labels", "#time-slider", "#timestamp", "#filmstrip",
  "#prompts", "#table", "#diff", "#candidates", "#eligible", "#empty",
  "#target-policy", "#custom-label", "#load-frame", "#delete", "#duplicate",
  "#decode", "#decode-selected", "#finalize-plan",
];
const elements = Object.fromEntries(selectors.map((selector) => [selector, new Element(selector)]));
globalThis.document = {
  activeElement: {tagName: "BODY"},
  addEventListener() {},
  createElement(tag) { return new Element(tag); },
  createTextNode(value) { return String(value); },
  querySelector(selector) { return elements[selector]; },
};
globalThis.Image = class Image extends Element {
  naturalWidth = 2;
  naturalHeight = 1;
  set src(_value) { if (this.onload) this.onload(); }
};
globalThis.requestAnimationFrame = (callback) => callback();
const manifest = {
  proxy_dimensions: {width: 2, height: 1},
  proxy_fps: 30,
  proxy_frame_count: 300,
  requested_proxy_timestamps_seconds: [0],
  workspace: {active_proxy_timestamp_seconds: 0, pending_boxes: []},
  candidates: [],
};
globalThis.fetch = async (path) => {
  if (path === "/api/state" || path === "/api/workspace") {
    return {ok: true, status: 200, json: async () => ({
      manifest, manual_seed_targets: [], manual_seed_target_policy: null,
      correction_policy: null, last_diff: [], worker_online: true,
    })};
  }
  if (path.startsWith("/api/frame")) {
    return {ok: true, status: 200, json: async () => ({frame_index: 0, image_uri: "frame.jpg"})};
  }
  throw new Error(`unexpected request: ${path}`);
};
const tick = () => new Promise((resolve) => setTimeout(resolve, 0));
(async () => {
  eval(app);
  await tick(); await tick(); await tick();
  tintedBinaryMaskImage({naturalWidth: 2, naturalHeight: 1}, "#22d3ee", 0.38);
  const expected = [0, 0, 0, 0, 34, 211, 238, 97];
  if (JSON.stringify(writtenPixels) !== JSON.stringify(expected)) {
    throw new Error(`unexpected binary mask overlay: ${JSON.stringify(writtenPixels)}`);
  }
})().catch((error) => {
  console.error(error.stack || error);
  process.exitCode = 1;
});
"""
    completed = subprocess.run(
        [node, "-e", script, str(root / "src/battle/static/calibration/app.js")],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr


def test_calibration_canvas_switches_active_candidate_with_numeric_keys() -> None:
    """Render only the active mask and persist a numeric candidate acceptance."""
    node = require_executable("node", "the calibration canvas regression test")
    root = Path(__file__).parents[1]
    script = r"""
const fs = require("node:fs");
const app = fs.readFileSync(process.argv[1], "utf8");
const listeners = {};
const calls = [];

class Element {
  constructor(selector = "") {
    this.selector = selector; this.children = []; this.dataset = {}; this.style = {};
    this.value = ""; this.checked = false; this.disabled = false; this.hidden = false;
    this.classList = {toggle() {}};
  }
  addEventListener(name, handler) { (listeners[this.selector] ||= {})[name] = handler; }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  getBoundingClientRect() { return {left: 0, top: 0, width: 954, height: 720}; }
  setPointerCapture() {}
  getContext() {
    return {
      clearRect() {}, fillRect() {}, fillText() {}, restore() {}, save() {}, scale() {},
      strokeRect() {}, translate() {}, setTransform() {},
      drawImage(image) { calls.push(image._src); },
    };
  }
  querySelectorAll(selector) {
    return selector === "input:checked" ? this.children.filter((node) => node.checked) : [];
  }
}
const selectors = [
  "#canvas", "#status", "#labels", "#time-slider", "#timestamp", "#filmstrip",
  "#prompts", "#table", "#diff", "#candidates", "#eligible", "#empty",
  "#target-policy", "#custom-label", "#load-frame", "#delete", "#duplicate",
  "#decode", "#decode-selected", "#finalize-plan", "#active-candidate",
  "#mask-visible", "#mask-opacity", "#mask-opacity-value",
];
const elements = Object.fromEntries(selectors.map((selector) => [selector, new Element(selector)]));
elements["#canvas"].width = 954; elements["#canvas"].height = 720;
elements["#mask-visible"].checked = true; elements["#mask-opacity"].value = "38";
globalThis.document = {
  activeElement: {tagName: "BODY"},
  addEventListener(name, handler) { (listeners.document ||= {})[name] = handler; },
  createElement(tag) { return new Element(tag); },
  createTextNode(value) { return String(value); },
  querySelector(selector) { return elements[selector]; },
};
globalThis.requestAnimationFrame = (callback) => callback();
globalThis.Image = class Image extends Element {
  set src(value) { this._src = value; this.onload(); }
};
const candidate = {
  candidate_id: "t000000-b01", intended_target: "left_hand",
  frame: {analysis_frame_index: 0, proxy_seconds: 0},
  pixel_box: {x1: 20, y1: 30, x2: 220, y2: 240},
  human_selected_candidate_index: null, human_accepted: false,
  selected_for_finalization: false, selected_for_correction: false,
  decoder_result: {
    deterministic_best_candidate_index: 2,
    candidates: [0, 1, 2, 3].map((candidate_index) => ({
      candidate_index, iou_score: 0.5, mask_uri: `results/mask-${candidate_index}.png`,
      is_deterministic_best: candidate_index === 2,
    })),
  },
};
const manifest = {
  proxy_dimensions: {width: 954, height: 720}, proxy_fps: 30,
  requested_proxy_timestamps_seconds: [0],
  workspace: {active_proxy_timestamp_seconds: 0, pending_boxes: []},
  candidates: [candidate],
};
const accepted = [];
globalThis.fetch = async (path, options = {}) => {
  if (path === "/api/state" || path === "/api/workspace") {
    return {ok: true, status: 200, json: async () => ({
      manifest, manual_seed_targets: ["left_hand"], manual_seed_target_policy: null,
      correction_policy: null, last_diff: [], worker_online: true,
    })};
  }
  if (path.startsWith("/api/frame")) {
    return {ok: true, status: 200, json: async () => ({frame_index: 0, image_uri: "frame.jpg"})};
  }
  if (path.startsWith("/api/candidates/")) {
    const body = JSON.parse(options.body); accepted.push(body);
    candidate.human_selected_candidate_index = body.candidate_index;
    candidate.human_accepted = true;
    return {ok: true, status: 200, json: async () => ({})};
  }
  throw new Error(`unexpected request: ${path}`);
};
const tick = () => new Promise((resolve) => setTimeout(resolve, 0));
(async () => {
  eval(app);
  await tick(); await tick(); await tick();
  const maskCalls = () => calls.filter((path) => path?.includes("/mask-"));
  if (!calls.includes("/artifacts/results/mask-2.png") ||
      maskCalls().some((path) => !path.endsWith("mask-2.png"))) {
    throw new Error(`initial canvas did not render only model-best mask: ${JSON.stringify(calls)}`);
  }
  calls.length = 0;
  let prevented = false;
  await listeners.document.keydown({key: "2", preventDefault() { prevented = true; }});
  if (!prevented || accepted.length !== 1 || accepted[0].candidate_index !== 1 ||
      !calls.includes("/artifacts/results/mask-1.png") ||
      maskCalls().some((path) => !path.endsWith("mask-1.png"))) {
    throw new Error(`key candidate switch failed: ${JSON.stringify({accepted, calls, prevented})}`);
  }
  if (!elements["#active-candidate"].textContent.includes("mask 2/4 · accepted")) {
    throw new Error(
      `active candidate indicator missing: ${elements["#active-candidate"].textContent}`
    );
  }
  document.activeElement = {tagName: "INPUT"};
  await listeners.document.keydown({
    key: "3", preventDefault() { throw new Error("should not prevent"); },
  });
  if (accepted.length !== 1) throw new Error("numeric key changed a focused input");
})().catch((error) => {
  console.error(error.stack || error);
  process.exitCode = 1;
});
"""
    completed = subprocess.run(
        [node, "-e", script, str(root / "src/battle/static/calibration/app.js")],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr


def test_calibration_ui_rejects_hides_and_restores_a_candidate() -> None:
    node = require_executable("node", "the calibration review regression test")
    root = Path(__file__).parents[1]
    script = r"""
const fs = require("node:fs");
const app = fs.readFileSync(process.argv[1], "utf8");

class Element {
  constructor(selector = "") {
    this.selector = selector; this.children = []; this.dataset = {}; this.style = {};
    this.value = ""; this.checked = false; this.disabled = false; this.hidden = false;
    this.classList = {toggle() {}};
  }
  addEventListener() {}
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  getBoundingClientRect() { return {left: 0, top: 0, width: 954, height: 720}; }
  getContext() {
    return {
      clearRect() {}, drawImage() {}, fillRect() {}, fillText() {}, restore() {}, save() {},
      scale() {}, strokeRect() {}, translate() {}, setTransform() {},
    };
  }
  querySelectorAll() { return []; }
}
const selectors = [
  "#canvas", "#status", "#labels", "#time-slider", "#timestamp", "#filmstrip",
  "#table", "#diff", "#candidate-status", "#eligible", "#empty", "#plan-lock",
  "#custom-label", "#custom-label-control", "#load-frame", "#delete", "#decode",
  "#finalize-plan", "#active-candidate", "#label-panel", "#prompt-clicks-panel",
  "#prompt-mode", "#clear-points", "#live-decode", "#live-decode-delay",
  "#mask-visible", "#mask-opacity", "#mask-opacity-value",
];
const elements = Object.fromEntries(selectors.map((selector) => [selector, new Element(selector)]));
elements["#canvas"].width = 954; elements["#canvas"].height = 720;
elements["#mask-visible"].checked = true; elements["#mask-opacity"].value = "38";
globalThis.document = {
  activeElement: {tagName: "BODY"},
  addEventListener() {},
  createElement(tag) { return new Element(tag); },
  createTextNode(value) { return String(value); },
  querySelector(selector) { return elements[selector]; },
};
globalThis.requestAnimationFrame = (callback) => callback();
globalThis.Image = class Image extends Element {
  naturalWidth = 954; naturalHeight = 720;
  set src(value) { this._src = value; if (this.onload) this.onload(); }
};
const candidate = {
  candidate_id: "t000000-b01", intended_target: "left_hand",
  frame: {analysis_frame_index: 0, proxy_seconds: 0},
  pixel_box: {x1: 20, y1: 30, x2: 220, y2: 240},
  human_selected_candidate_index: null, human_accepted: false, rejected: false,
  selected_for_finalization: false, selected_for_correction: false,
  decoder_result: {
    deterministic_best_candidate_index: 0,
    candidates: [{
      candidate_index: 0, iou_score: 0.5, mask_uri: "results/mask.png",
      is_deterministic_best: true,
    }],
  },
};
const manifest = {
  proxy_dimensions: {width: 954, height: 720}, proxy_fps: 30,
  requested_proxy_timestamps_seconds: [0],
  workspace: {active_proxy_timestamp_seconds: 0, pending_boxes: []},
  candidates: [candidate],
};
const requests = [];
globalThis.fetch = async (path, options = {}) => {
  if (path === "/api/state" || path === "/api/workspace") {
    return {ok: true, status: 200, json: async () => ({
      manifest, manual_seed_targets: ["left_hand"], manual_seed_target_policy: null,
      correction_policy: null, last_diff: [], worker_online: true,
    })};
  }
  if (path.startsWith("/api/frame")) {
    return {ok: true, status: 200, json: async () => ({frame_index: 0, image_uri: "frame.jpg"})};
  }
  if (path.endsWith("/reject")) {
    requests.push(path); candidate.rejected = true;
    return {ok: true, status: 200, json: async () => ({manifest})};
  }
  if (path.endsWith("/restore")) {
    requests.push(path); candidate.rejected = false;
    return {ok: true, status: 200, json: async () => ({manifest})};
  }
  throw new Error(`unexpected request: ${path}`);
};
const tick = () => new Promise((resolve) => setTimeout(resolve, 0));
(async () => {
  eval(app);
  await tick(); await tick(); await tick();
  const statusCell = () => elements["#candidate-status"].children[0]
    .children[1].children[1].children[0];
  let wrapper = statusCell();
  const reject = wrapper.children[1];
  if (wrapper.children[0].textContent !== "◐ Preview"
      || reject.textContent !== "×" || reject.disabled) {
    throw new Error("preview cell lacks an enabled discard control");
  }
  await reject.onclick({stopPropagation() {}});
  wrapper = statusCell();
  if (wrapper.children[0].textContent !== "× Rejected") {
    throw new Error("rejected candidate is not visible in the status table");
  }
  const restore = wrapper.children[1];
  if (restore.textContent !== "↶") throw new Error("rejected cell lacks restore action");
  await restore.onclick({stopPropagation() {}});
  if (candidate.rejected || requests.length !== 2) {
    throw new Error(`reject/restore did not persist: ${JSON.stringify({candidate, requests})}`);
  }
})().catch((error) => {
  console.error(error.stack || error);
  process.exitCode = 1;
});
"""
    completed = subprocess.run(
        [node, "-e", script, str(root / "src/battle/static/calibration/app.js")],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr


def test_web_review_renders_all_four_configured_target_groups_headlessly() -> None:
    """Keep four decoded groups visible after the frame-0 review view loads."""
    node = require_executable("node", "the calibration review regression test")
    root = Path(__file__).parents[1]
    script = r"""
const fs = require("node:fs");
const app = fs.readFileSync(process.argv[1], "utf8");
const targets = ["left_hand", "right_hand", "yellow_toy_top", "black_toy_top_base"];

class Element {
  constructor(selector = "") {
    this.selector = selector; this.children = []; this.dataset = {}; this.style = {};
    this.value = ""; this.checked = false; this.disabled = false; this.hidden = false;
    this.classList = {toggle() {}};
  }
  addEventListener() {}
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  getBoundingClientRect() { return {left: 0, top: 0, width: 954, height: 720}; }
  getContext() {
    return {
      clearRect() {}, drawImage() {}, fillRect() {}, fillText() {}, restore() {}, save() {},
      scale() {}, strokeRect() {}, translate() {}, setTransform() {},
    };
  }
  querySelectorAll(selector) {
    const descendants = (nodes) => nodes.flatMap((node) => (
      node instanceof Element ? [node, ...descendants(node.children)] : []
    ));
    return selector === "input:checked"
      ? descendants(this.children).filter((node) => node.checked)
      : [];
  }
}
const selectors = [
  "#canvas", "#status", "#labels", "#time-slider", "#timestamp", "#filmstrip",
  "#table", "#diff", "#candidate-status", "#eligible", "#empty", "#plan-lock",
  "#custom-label", "#custom-label-control", "#load-frame", "#delete", "#decode",
  "#finalize-plan", "#active-candidate", "#label-panel", "#prompt-clicks-panel",
  "#prompt-mode", "#clear-points", "#live-decode", "#live-decode-delay",
];
const elements = Object.fromEntries(selectors.map((selector) => [selector, new Element(selector)]));
elements["#canvas"].width = 954; elements["#canvas"].height = 720;
globalThis.document = {
  activeElement: {tagName: "BODY"},
  addEventListener() {},
  createElement(tag) { return new Element(tag); },
  createTextNode(value) { return String(value); },
  querySelector(selector) { return elements[selector]; },
};
globalThis.Image = class Image extends Element {
  constructor() { super("img"); }
  set src(value) { this._src = value; if (this.onload) this.onload(); }
};
globalThis.requestAnimationFrame = (callback) => callback();
const manifest = {
  proxy_dimensions: {width: 954, height: 720}, proxy_fps: 30,
  requested_proxy_timestamps_seconds: [0, 10, 30, 50],
  workspace: {
    active_proxy_timestamp_seconds: 0,
    pending_boxes: targets.map((intended_target, index) => ({
      box_id: `p000000-b${String(index + 1).padStart(2, "0")}`, intended_target,
      stage: "pending", frame: {proxy_seconds: 0},
      pixel_box: {x1: 10, y1: 20, x2: 100, y2: 120},
    })),
  },
  candidates: targets.map((intended_target, index) => ({
    candidate_id: `t${String(index * 300).padStart(6, "0")}-b01`,
    intended_target,
    frame: {analysis_frame_index: index * 300, proxy_seconds: [0, 10, 30, 50][index]},
    pixel_box: {x1: 10, y1: 20, x2: 100, y2: 120},
    human_selected_candidate_index: index === 0 ? 1 : null,
    human_accepted: index === 0,
    selected_for_finalization: index === 0,
    decoder_result: {
      candidates: [0, 1].map((candidate_index) => ({
        candidate_index, iou_score: 0.5 + candidate_index / 10,
        review_uri: `results/${intended_target}-${candidate_index}.png`,
        is_deterministic_best: candidate_index === 1,
      })),
    },
  })),
};
const accepted = [];
const decodeRequests = [];
globalThis.fetch = async (path, options = {}) => {
  if (path === "/api/state" || path === "/api/workspace" || path.startsWith("/api/candidates/")) {
    if (path.startsWith("/api/candidates/")) accepted.push({path, body: JSON.parse(options.body)});
    return {ok: true, status: 200, json: async () => ({
      manifest, manual_seed_targets: targets,
      manual_seed_target_policy: {config_id: "four-target-policy", required_target_count: 4},
      last_diff: [], worker_online: true,
    })};
  }
  if (path === "/api/decode") {
    decodeRequests.push(JSON.parse(options.body).box_ids);
    return {ok: true, status: 202, json: async () => ({job_id: "four-targets"})};
  }
  if (path === "/api/jobs/four-targets") {
    return {ok: true, status: 200, json: async () => ({status: "succeeded"})};
  }
  if (path.startsWith("/api/frame")) {
    return {ok: true, status: 200, json: async () => ({frame_index: 0, image_uri: "frame.jpg"})};
  }
  throw new Error(`unexpected request: ${path}`);
};
globalThis.setInterval = (callback) => { Promise.resolve().then(callback); return 1; };
globalThis.clearInterval = () => {};
const tick = () => new Promise((resolve) => setTimeout(resolve, 0));
(async () => {
  eval(app);
  await tick(); await tick(); await tick();
  const table = elements["#candidate-status"].children[0];
  if (table.children.length !== 5) {
    throw new Error(`rendered ${table.children.length - 1} frame rows, expected 4`);
  }
  const headings = table.children[0].children.slice(1).map((cell) => cell.textContent);
  for (const [index, target] of targets.entries()) {
    if (headings[index] !== target.replaceAll("_", " ")) {
      throw new Error(`target column ${index + 1} label missing: ${headings[index]}`);
    }
  }
  const firstStatus = table.children[1].children[1].children[0].children[0];
  if (firstStatus.textContent !== "✓ Done") {
    throw new Error(`accepted target status missing: ${firstStatus.textContent}`);
  }
  await elements["#decode"].onclick();
  await tick();
  if (decodeRequests.length !== 1 || decodeRequests[0].length !== 4) {
    throw new Error(
      `primary decode did not queue all four prompts: ${JSON.stringify(decodeRequests)}`
    );
  }
})().catch((error) => {
  console.error(error.stack || error);
  process.exitCode = 1;
});
"""
    completed = subprocess.run(
        [node, "-e", script, str(root / "src/battle/static/calibration/app.js")],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr


def test_workspace_normalizes_uppercase_timestamp_output_directory(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    output_directory = tmp_path / "muggledsam-sam3-e4-web-calibration-left-hand-20260909T224835Z"
    args = argparse.Namespace(
        timestamps="0",
        output_dir=output_directory,
        run_root=tmp_path,
        resume=False,
        config=root / "configs/clips/assembly101_nusar_9033_ego_viewpoint_screen_g2.json",
        manual_seed_target_config=None,
        no_worker=True,
    )

    workspace = create_workspace(args, root)
    try:
        assert workspace.manifest.calibration_id == output_directory.name.lower()
        persisted = json.loads(workspace.manifest_path.read_text())
        assert persisted["calibration_id"] == output_directory.name.lower()
    finally:
        workspace.close()


def test_resume_preserves_promoted_frames_but_rejects_new_cli_frames(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    output_directory = tmp_path / "promoted-frame-workspace"
    args = argparse.Namespace(
        timestamps="0,10",
        output_dir=output_directory,
        run_root=tmp_path,
        resume=False,
        config=root / "configs/clips/assembly101_nusar_9033_ego_viewpoint_screen_g2.json",
        manual_seed_target_config=None,
        correction_policy=None,
        no_worker=True,
    )
    workspace = create_workspace(args, root)
    workspace.add_calibration_frame(5.0)
    workspace.close()

    args.resume = True
    resumed = create_workspace(args, root)
    try:
        assert resumed.manifest.requested_proxy_timestamps_seconds == (0.0, 5.0, 10.0)
    finally:
        resumed.close()

    args.timestamps = "0,7,10"
    with pytest.raises(ValueError, match="add frames in the workspace"):
        create_workspace(args, root)


def test_browse_only_frame_cannot_create_a_prompt(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    manifest_before = workspace.manifest
    try:
        with pytest.raises(ValueError, match="browse-only"):
            workspace.add_or_update_prompt(
                {
                    "timestamp": 5.0,
                    "intended_target": "left_hand",
                    "pixel_box": {"x1": 10, "y1": 20, "x2": 100, "y2": 120},
                }
            )
        with pytest.raises(ValueError, match="requested timestamps"):
            workspace.set_active_timestamp(5.0)

        assert workspace.manifest == manifest_before
        assert not workspace.manifest_path.exists()
        assert not workspace.manifest.workspace.pending_boxes
    finally:
        workspace.close()


def test_browse_only_frame_can_be_promoted_to_calibration(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    try:
        added = workspace.add_calibration_frame(3.23)

        assert added == {"frame_index": 97, "proxy_seconds": pytest.approx(97 / 30)}
        assert workspace.manifest.requested_proxy_timestamps_seconds == (
            0.0,
            pytest.approx(97 / 30),
            10.0,
        )
        assert workspace.add_calibration_frame(3.23) == added
        assert len(workspace.manifest.requested_proxy_timestamps_seconds) == 3
        prompt = workspace.add_or_update_prompt(
            {
                "timestamp": 3.23,
                "intended_target": "left_hand",
                "pixel_box": {"x1": 10, "y1": 20, "x2": 100, "y2": 120},
            }
        )
        assert prompt["frame"]["analysis_frame_index"] == 97
    finally:
        workspace.close()


def test_workspace_persists_batch_and_only_allows_human_frame_zero_proposals(
    tmp_path: Path,
) -> None:
    workspace = make_workspace(tmp_path)
    try:
        first = workspace.add_or_update_prompt(
            {
                "timestamp": 0,
                "intended_target": "left_hand",
                "pixel_box": {"x1": 10, "y1": 20, "x2": 100, "y2": 120},
            }
        )
        later = workspace.add_or_update_prompt(
            {
                "timestamp": 10,
                "intended_target": "right_hand",
                "pixel_box": {"x1": 20, "y1": 30, "x2": 110, "y2": 130},
            }
        )
        job_id = workspace.queue_decode([first["box_id"], later["box_id"]])
        workspace.jobs[job_id]["future"].result(timeout=2)
        assert workspace.jobs[job_id]["status"] == "succeeded"
        assert not workspace.manifest.workspace.pending_boxes
        assert workspace.last_diff

        first_candidate, later_candidate = workspace.manifest.candidates
        workspace.accept_candidate(first_candidate.candidate_id, 0, eligible=True)
        with pytest.raises(ValueError, match="only frame-0"):
            workspace.accept_candidate(later_candidate.candidate_id, 0, eligible=True)
        proposal = workspace.create_proposal([first_candidate.candidate_id])
        assert proposal["seeds"][0]["human_selected_candidate_index"] == 0
        persisted = json.loads((tmp_path / "calibration_manifest.json").read_text())
        assert persisted["candidates"][0]["human_accepted"] is True
    finally:
        workspace.close()


def test_http_rejection_persists_retains_artifact_excludes_proposals_and_can_undo(
    tmp_path: Path,
) -> None:
    workspace = make_workspace(tmp_path)
    server, base = serve(workspace)
    try:
        (candidate,) = decode_candidates(workspace, [(0, "left_hand")])
        post_json(
            base,
            f"/api/candidates/{candidate.candidate_id}/accept",
            {"candidate_index": 0, "eligible": True},
        )
        with pytest.raises(urllib.error.HTTPError) as protected:
            post_json(base, f"/api/candidates/{candidate.candidate_id}/reject", {})
        assert "unaccept the current accepted candidate" in protected.value.read().decode()

        post_json(base, f"/api/candidates/{candidate.candidate_id}/unaccept", {})
        mask_path = workspace.manifest_path.parent / candidate.decoder_result.candidates[0].mask_uri
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.write_bytes(b"provenance mask")
        post_json(base, f"/api/candidates/{candidate.candidate_id}/reject", {})

        persisted = json.loads(workspace.manifest_path.read_text())["candidates"][0]
        assert persisted["rejected"] is True
        assert persisted["human_accepted"] is False
        assert persisted["human_selected_candidate_index"] is None
        assert persisted["selected_for_finalization"] is False
        assert mask_path.read_bytes() == b"provenance mask"
        with pytest.raises(urllib.error.HTTPError) as excluded:
            post_json(base, "/api/proposal", {"candidate_ids": [candidate.candidate_id]})
        assert "non-rejected" in excluded.value.read().decode()
        with pytest.raises(urllib.error.HTTPError) as unavailable:
            post_json(
                base,
                f"/api/candidates/{candidate.candidate_id}/accept",
                {"candidate_index": 0, "eligible": True},
            )
        assert "restore a rejected candidate" in unavailable.value.read().decode()

        restored = post_json(base, f"/api/candidates/{candidate.candidate_id}/restore", {})
        assert restored["manifest"]["candidates"][0]["rejected"] is False
        post_json(
            base,
            f"/api/candidates/{candidate.candidate_id}/accept",
            {"candidate_index": 0, "eligible": True},
        )
        proposal = post_json(base, "/api/proposal", {"candidate_ids": [candidate.candidate_id]})
        assert proposal["seeds"][0]["candidate_id"] == candidate.candidate_id
    finally:
        server.shutdown()
        server.server_close()
        workspace.close()


def test_http_hidden_target_marks_persist_and_yield_to_accepted_masks(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    server, base = serve(workspace)
    try:
        with pytest.raises(urllib.error.HTTPError) as browse_only:
            post_json(base, "/api/hidden-targets", {"timestamp": 5.0, "intended_target": "x"})
        assert "browse-only" in browse_only.value.read().decode()

        marked = post_json(
            base, "/api/hidden-targets", {"timestamp": 10.0, "intended_target": "left_hand"}
        )
        (mark,) = marked["manifest"]["hidden_targets"]
        assert mark["intended_target"] == "left_hand"
        assert mark["frame"]["analysis_frame_index"] == 300
        assert mark["state"] == "hidden" and mark["marked_by"] == "human"
        persisted = json.loads(workspace.manifest_path.read_text())
        assert len(persisted["hidden_targets"]) == 1
        # Marking the same cell twice is idempotent, not a duplicate.
        again = post_json(
            base, "/api/hidden-targets", {"timestamp": 10.0, "intended_target": "left_hand"}
        )
        assert len(again["manifest"]["hidden_targets"]) == 1

        # Accepting a mask on the same cell supersedes the hidden mark.
        (candidate,) = decode_candidates(workspace, [(10.0, "left_hand")])
        accepted = post_json(
            base,
            f"/api/candidates/{candidate.candidate_id}/accept",
            {"candidate_index": 0, "eligible": False},
        )
        assert accepted["manifest"]["hidden_targets"] == []
        # ...and a hidden mark refuses while an accepted mask exists.
        with pytest.raises(urllib.error.HTTPError) as conflict:
            post_json(
                base, "/api/hidden-targets", {"timestamp": 10.0, "intended_target": "left_hand"}
            )
        assert "unaccept the accepted mask" in conflict.value.read().decode()

        post_json(base, f"/api/candidates/{candidate.candidate_id}/unaccept", {})
        post_json(base, "/api/hidden-targets", {"timestamp": 10.0, "intended_target": "left_hand"})
        cleared = post_json(
            base, "/api/hidden-targets/clear", {"timestamp": 10.0, "intended_target": "left_hand"}
        )
        assert cleared["manifest"]["hidden_targets"] == []
        with pytest.raises(urllib.error.HTTPError) as absent:
            post_json(
                base,
                "/api/hidden-targets/clear",
                {"timestamp": 10.0, "intended_target": "left_hand"},
            )
        assert "is not marked hidden" in absent.value.read().decode()
    finally:
        server.shutdown()
        server.server_close()
        workspace.close()


def test_resume_without_timestamps_keeps_the_persisted_frames(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    output_directory = tmp_path / "anchor-like-workspace"
    args = argparse.Namespace(
        timestamps="10,12.333333333,20",
        output_dir=output_directory,
        run_root=tmp_path,
        resume=False,
        config=root / "configs/clips/assembly101_nusar_9033_ego_viewpoint_screen_g2.json",
        manual_seed_target_config=None,
        correction_policy=None,
        no_worker=True,
    )
    create_workspace(args, root).close()
    args.resume = True
    args.timestamps = None
    resumed = create_workspace(args, root)
    try:
        assert [round(t * 30) for t in resumed.manifest.requested_proxy_timestamps_seconds] == [
            300,
            370,
            600,
        ]
    finally:
        resumed.close()
    fresh = argparse.Namespace(**{**vars(args), "resume": False, "output_dir": tmp_path / "d"})
    workspace = create_workspace(fresh, root)
    try:
        assert workspace.manifest.requested_proxy_timestamps_seconds == (0.0, 10.0, 30.0, 50.0)
    finally:
        workspace.close()


def test_rejected_candidate_causes_finalization_validation_without_writing_plan(
    tmp_path: Path,
) -> None:
    workspace = make_correction_workspace(tmp_path)
    targets = ("left_hand", "right_hand", "yellow_toy_top", "black_toy_top_base")
    try:
        candidates = decode_candidates(workspace, [(0, target) for target in targets])
        for candidate in candidates:
            workspace.accept_candidate(candidate.candidate_id, 0, eligible=True)
        workspace.unaccept_candidate(candidates[0].candidate_id)
        workspace.reject_candidate(candidates[0].candidate_id)
        before = workspace.manifest_path.read_bytes()

        with pytest.raises(ValueError, match="exactly 4 distinct human-selected frame-0 masks"):
            workspace.finalize_tracking_plan()

        assert workspace.manifest_path.read_bytes() == before
        assert not (tmp_path / "proposed_tracking_prompt.json").exists()
        assert not (tmp_path / "multi_keyframe_correction_schedule.json").exists()
    finally:
        workspace.close()


def test_http_rejected_candidate_cannot_create_correction_schedule(tmp_path: Path) -> None:
    workspace = make_correction_workspace(tmp_path)
    server, base = serve(workspace)
    targets = ("left_hand", "right_hand", "yellow_toy_top", "black_toy_top_base")
    try:
        candidates = decode_candidates(
            workspace, [(0, target) for target in targets] + [(10, "left_hand")]
        )
        for candidate in candidates[:-1]:
            workspace.accept_candidate(candidate.candidate_id, 0, eligible=True)
            selected_mask = candidate.decoder_result.candidates[0]
            mask_path = workspace.manifest_path.parent / selected_mask.mask_uri
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            mask_path.write_bytes(candidate.candidate_id.encode())
        workspace.reject_candidate(candidates[-1].candidate_id)
        manifest_before = workspace.manifest_path.read_bytes()

        with pytest.raises(urllib.error.HTTPError) as excluded:
            post_json(
                base,
                "/api/correction-schedule",
                {"candidate_ids": [candidate.candidate_id for candidate in candidates]},
            )

        assert "non-rejected" in excluded.value.read().decode()
        assert workspace.manifest_path.read_bytes() == manifest_before
        assert not (tmp_path / "multi_keyframe_correction_schedule.json").exists()
    finally:
        server.shutdown()
        server.server_close()
        workspace.close()


def test_completed_tracking_plan_locks_candidate_rejection_state(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    try:
        (candidate,) = decode_candidates(workspace, [(0, "left_hand")])
        workspace.accept_candidate(candidate.candidate_id, 0, eligible=True)
        workspace.finalize_tracking_plan()
        manifest_before = workspace.manifest_path.read_bytes()
        proposal_before = (tmp_path / "proposed_tracking_prompt.json").read_bytes()

        with pytest.raises(ValueError, match="candidate review is locked"):
            workspace.unaccept_candidate(candidate.candidate_id)
        with pytest.raises(ValueError, match="candidate review is locked"):
            workspace.add_or_update_prompt(
                {
                    "timestamp": 0.0,
                    "intended_target": "right_hand",
                    "pixel_box": {"x1": 10, "y1": 20, "x2": 100, "y2": 120},
                }
            )
        workspace.set_active_timestamp(10.0)

        assert workspace.manifest_path.read_bytes() == manifest_before
        assert (tmp_path / "proposed_tracking_prompt.json").read_bytes() == proposal_before
    finally:
        workspace.close()


def test_legacy_model_only_candidate_is_readable_but_cannot_finalize(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    try:
        prompt = workspace.add_or_update_prompt(
            {
                "timestamp": 0,
                "intended_target": "left_hand",
                "pixel_box": {"x1": 10, "y1": 20, "x2": 100, "y2": 120},
            }
        )
        job_id = workspace.queue_decode([prompt["box_id"]])
        workspace.jobs[job_id]["future"].result(timeout=2)
        legacy = workspace.manifest.model_dump(mode="json")
        candidate = legacy["candidates"][0]
        candidate.pop("human_selected_candidate_index")
        candidate.pop("human_accepted")
        candidate["selected_for_finalization"] = True
        path = tmp_path / "legacy.json"
        path.write_text(json.dumps(legacy))
        with pytest.raises(ValueError, match="human-accepted"):
            finalize_prompt(
                manifest_path=path,
                candidate_ids=[candidate["candidate_id"]],
                proposal_path=tmp_path / "proposal.json",
                repository_root=tmp_path,
            )
    finally:
        workspace.close()


def test_worker_client_uses_persistent_no_gpu_jsonl_fixture(tmp_path: Path) -> None:
    script = tmp_path / "fixture_worker.py"
    script.write_text(
        "import json, sys\n"
        "for line in sys.stdin:\n"
        " request=json.loads(line)\n"
        " result={'request': request['command']}\n"
        " response={'request_id':request['request_id'],'ok':True,'result':result}\n"
        " print(json.dumps(response),flush=True)\n"
        " if request['command']=='shutdown': break\n"
    )
    worker = WorkerClient(
        [sys.executable, str(script)], environment={}, stderr_path=tmp_path / "worker.log"
    )
    try:
        assert worker.request("health", {}) == {"request": "health"}
        assert worker.request("health", {}) == {"request": "health"}
    finally:
        worker.close()


def test_static_and_state_routes_are_available_without_a_model(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    server, base = serve(workspace)
    try:
        assert b"Rapid SAM3 calibration" in urllib.request.urlopen(f"{base}/").read()
        state = json.loads(urllib.request.urlopen(f"{base}/api/state").read())
        assert state["manifest"]["view_id"] == "ego-hmc21179183"
        request = urllib.request.Request(
            f"{base}/api/prompts",
            data=json.dumps(
                {
                    "timestamp": 0,
                    "intended_target": "left_hand",
                    "pixel_box": {"x1": 1, "y1": 2, "x2": 20, "y2": 30},
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        assert json.loads(urllib.request.urlopen(request).read())["box_id"] == "p000000-b01"
    finally:
        server.shutdown()
        server.server_close()
        workspace.close()


def test_no_worker_frame_preview_uses_local_ffmpeg_without_enabling_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = make_workspace(tmp_path)
    workspace.decoder = None
    server, base = serve(workspace)

    def fake_ffmpeg(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        assert command[0] == "ffmpeg"
        image_path = Path(command[-1])
        image_path.write_bytes(
            base64.b64decode(
                "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////"
                "2wBDAf//////////////////////////////////////////////////////////////////////////////////////wAARCAABAAEDASIAAhEBAxEB"
                "/8QAFQABAQAAAAAAAAAAAAAAAAAAAAf/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIQAxAAAAH/xAAUEAEAAAAAAAAAAAAAAAAAAAAA"
                "/9oACAEBAAEFAqf/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAEDAQE/AT//xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAECAQE/AT//"
                "xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAY/Aqf/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAE/IX//2gAMAwEAAgADAAAA"
                "EP/EABQRAQAAAAAAAAAAAAAAAAAAABD/2gAIAQMBAT8QH//EABQRAQAAAAAAAAAAAAAAAAAAABD/2gAIAQIBAT8QH//EABQQAQAAAAAAAA"
                "AAAAAAAAAAABD/2gAIAQEAAT8QH//Z"
            )
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("battle.muggled_calibration_web.subprocess.run", fake_ffmpeg)
    try:
        frame = json.loads(urllib.request.urlopen(f"{base}/api/frame?timestamp=0").read())
        assert frame == {"frame_index": 0, "image_uri": "results/frames/frame-000000.jpg"}
        image = urllib.request.urlopen(f"{base}/artifacts/{frame['image_uri']}").read()
        assert image.startswith(b"\xff\xd8")
        decode = urllib.request.Request(
            f"{base}/api/decode",
            data=json.dumps({"box_ids": []}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(decode)
        assert "decoder is offline (--no-worker)" in error.value.read().decode()
        state = json.loads(urllib.request.urlopen(f"{base}/api/state").read())
        assert state["worker_online"] is False
    finally:
        server.shutdown()
        server.server_close()
        workspace.close()


def test_main_closes_the_workspace_on_keyboard_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ClosingWorkspace:
        closed = False
        manifest_path = tmp_path / "run" / "calibration_manifest.json"

        def close(self) -> None:
            self.closed = True

    class InterruptingServer:
        server_port = 43210
        closed = False

        def __init__(self, _address: object, _handler: object) -> None:
            return

        def serve_forever(self) -> None:
            raise KeyboardInterrupt

        def server_close(self) -> None:
            type(self).closed = True

    workspace = ClosingWorkspace()
    monkeypatch.setattr(
        "battle.muggled_calibration_web.make_workspace",
        lambda _args, _root: workspace,
    )
    monkeypatch.setattr(
        "battle.muggled_calibration_web.ThreadingHTTPServer",
        InterruptingServer,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "battle-muggled-calibration-web",
            "--no-worker",
            "--output-dir",
            str(tmp_path / "run"),
        ],
    )

    __import__("battle.muggled_calibration_web", fromlist=["main"]).main()

    assert workspace.closed is True
    assert InterruptingServer.closed is True


def test_http_proposal_requires_human_accepted_frame_zero_candidate(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    server, base = serve(workspace)

    def request(path: str, body: dict[str, object]) -> dict[str, object]:
        value = urllib.request.Request(
            f"{base}{path}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return json.loads(urllib.request.urlopen(value).read())

    try:
        first = request(
            "/api/prompts",
            {
                "timestamp": 0,
                "intended_target": "left_hand",
                "pixel_box": {"x1": 10, "y1": 20, "x2": 100, "y2": 120},
            },
        )
        later = request(
            "/api/prompts",
            {
                "timestamp": 10,
                "intended_target": "right_hand",
                "pixel_box": {"x1": 20, "y1": 30, "x2": 110, "y2": 130},
            },
        )
        queued = request("/api/decode", {"box_ids": [first["box_id"], later["box_id"]]})
        workspace.jobs[queued["job_id"]]["future"].result(timeout=2)
        first_candidate, later_candidate = workspace.manifest.candidates

        with pytest.raises(urllib.error.HTTPError) as later_error:
            request(
                f"/api/candidates/{later_candidate.candidate_id}/accept",
                {"candidate_index": 0, "eligible": True},
            )
        assert "only frame-0" in json.loads(later_error.value.read())["error"]

        with pytest.raises(urllib.error.HTTPError) as unaccepted_error:
            request("/api/proposal", {"candidate_ids": [first_candidate.candidate_id]})
        assert "human-accepted frame-0" in json.loads(unaccepted_error.value.read())["error"]

        request(
            f"/api/candidates/{first_candidate.candidate_id}/accept",
            {"candidate_index": 0, "eligible": True},
        )
        proposal = request("/api/proposal", {"candidate_ids": [first_candidate.candidate_id]})
        assert proposal["seeds"][0]["candidate_id"] == first_candidate.candidate_id
    finally:
        server.shutdown()
        server.server_close()
        workspace.close()


def test_http_finalize_tracking_plan_with_only_initial_masks(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    server, base = serve(workspace)
    try:
        (candidate,) = decode_candidates(workspace, [(0, "left_hand")])
        workspace.accept_candidate(candidate.candidate_id, 0, eligible=True)

        plan = post_json(base, "/api/finalize-tracking-plan", {})

        assert plan["proposal"]["seeds"][0]["candidate_id"] == candidate.candidate_id
        assert plan["correction_schedule"] is None
        manifest_path = tmp_path / "calibration_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        assert (tmp_path / "proposed_tracking_prompt.json").is_file()
        assert not (tmp_path / "multi_keyframe_correction_schedule.json").exists()
        assert plan["proposal"]["calibration_manifest_sha256"] == sha256_file(manifest_path)
        assert manifest["final_proposal_uri"] == str(tmp_path / "proposed_tracking_prompt.json")
    finally:
        server.shutdown()
        server.server_close()
        workspace.close()


def test_http_finalize_tracking_plan_with_later_corrections(tmp_path: Path) -> None:
    workspace = make_correction_workspace(tmp_path)
    server, base = serve(workspace)
    targets = ("left_hand", "right_hand", "yellow_toy_top", "black_toy_top_base")
    try:
        candidates = decode_candidates(
            workspace, [(0, target) for target in targets] + [(10, "left_hand")]
        )
        for candidate in candidates[:-1]:
            workspace.accept_candidate(candidate.candidate_id, 0, eligible=True)
        workspace.accept_candidate(candidates[-1].candidate_id, 0, eligible=False, correction=True)
        write_selected_masks(workspace)

        plan = post_json(base, "/api/finalize-tracking-plan", {})

        assert len(plan["proposal"]["seeds"]) == len(targets)
        assert len(plan["correction_schedule"]["corrections"]) == len(targets) + 1
        manifest_path = tmp_path / "calibration_manifest.json"
        assert plan["proposal"]["calibration_manifest_sha256"] == sha256_file(manifest_path)
        assert plan["correction_schedule"]["calibration_manifest_fingerprint"]["sha256"] == (
            sha256_file(manifest_path)
        )
        assert (tmp_path / "multi_keyframe_correction_schedule.json").is_file()
    finally:
        server.shutdown()
        server.server_close()
        workspace.close()


def test_http_finalize_tracking_plan_rejects_missing_required_initial_target(
    tmp_path: Path,
) -> None:
    workspace = make_correction_workspace(tmp_path)
    server, base = serve(workspace)
    try:
        candidates = decode_candidates(
            workspace,
            [(0, target) for target in ("left_hand", "right_hand", "yellow_toy_top")],
        )
        for candidate in candidates:
            workspace.accept_candidate(candidate.candidate_id, 0, eligible=True)
        manifest_path = tmp_path / "calibration_manifest.json"
        before = manifest_path.read_bytes()

        with pytest.raises(urllib.error.HTTPError) as error:
            post_json(base, "/api/finalize-tracking-plan", {})

        assert "exactly 4 distinct human-selected frame-0 masks" in error.value.read().decode()
        assert manifest_path.read_bytes() == before
        assert not (tmp_path / "proposed_tracking_prompt.json").exists()
        assert not (tmp_path / "multi_keyframe_correction_schedule.json").exists()
    finally:
        server.shutdown()
        server.server_close()
        workspace.close()


def test_finalize_tracking_plan_does_not_commit_partial_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = make_correction_workspace(tmp_path)
    targets = ("left_hand", "right_hand", "yellow_toy_top", "black_toy_top_base")
    try:
        candidates = decode_candidates(workspace, [(0, target) for target in targets])
        for candidate in candidates:
            workspace.accept_candidate(candidate.candidate_id, 0, eligible=True)
        write_selected_masks(workspace)
        manifest_path = tmp_path / "calibration_manifest.json"
        before = manifest_path.read_bytes()

        def fail_schedule(**_kwargs: object) -> object:
            raise OSError("simulated schedule write failure")

        monkeypatch.setattr(
            "battle.muggled_calibration_web.finalize_correction_schedule", fail_schedule
        )
        with pytest.raises(OSError, match="simulated schedule write failure"):
            workspace.finalize_tracking_plan()

        assert manifest_path.read_bytes() == before
        assert not (tmp_path / "proposed_tracking_prompt.json").exists()
        assert not (tmp_path / "multi_keyframe_correction_schedule.json").exists()
    finally:
        workspace.close()


def test_web_create_proposal_uses_configured_target_policy_without_ui_shape_error(
    tmp_path: Path,
) -> None:
    """Click proposal with four checked configured-target inputs in a headless DOM."""
    workspace = make_configured_workspace(tmp_path)
    assert workspace.snapshot()["manual_seed_target_policy"] == {
        "config_id": "e4-left-hand-right-hand-yellow-toy-top-black-toy-top-base",
        "required_target_count": 4,
    }
    server, base = serve(workspace)

    def request(path: str, body: dict[str, object]) -> dict[str, object]:
        value = urllib.request.Request(
            f"{base}{path}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return json.loads(urllib.request.urlopen(value).read())

    try:
        targets = ("left_hand", "right_hand", "yellow_toy_top", "black_toy_top_base")
        prompts = [
            request(
                "/api/prompts",
                {
                    "timestamp": 0,
                    "intended_target": target,
                    "pixel_box": {"x1": 10 + index, "y1": 20, "x2": 100, "y2": 120},
                },
            )
            for index, target in enumerate(targets)
        ]
        queued = request("/api/decode", {"box_ids": [prompt["box_id"] for prompt in prompts]})
        workspace.jobs[queued["job_id"]]["future"].result(timeout=2)
        candidates_by_target = {
            candidate.intended_target: candidate for candidate in workspace.manifest.candidates
        }
        checked_candidate_ids = [
            candidates_by_target[target].candidate_id for target in reversed(targets)
        ]
        for candidate_id in checked_candidate_ids:
            request(
                f"/api/candidates/{candidate_id}/accept",
                {"candidate_index": 0, "eligible": True},
            )

        clicked = click_create_proposal_in_headless_dom(base)
        if not clicked:
            request("/api/proposal", {"candidate_ids": checked_candidate_ids})
        proposal = json.loads((tmp_path / "proposed_tracking_prompt.json").read_text())

        assert [seed["intended_target"] for seed in proposal["seeds"]] == list(targets)
        assert [seed["candidate_id"] for seed in proposal["seeds"]] == [
            candidates_by_target[target].candidate_id for target in targets
        ]
        assert (tmp_path / "proposed_tracking_prompt.json").is_file()
        persisted_manifest = tmp_path / "calibration_manifest.json"
        assert proposal["calibration_manifest_sha256"] == sha256_file(persisted_manifest)
        assert json.loads(persisted_manifest.read_text())["final_proposal_uri"] == (
            str(tmp_path / "proposed_tracking_prompt.json")
        )
    finally:
        server.shutdown()
        server.server_close()
        workspace.close()


EXPECTED_PANELS = (
    "frames",
    "label",
    "prompt-clicks",
    "view-aids",
    "candidates",
    "table",
    "diff",
    "finalize",
)
# The prompting workflow is open on a fresh profile; review and reporting stay collapsed.
DEFAULT_OPEN_PANELS = frozenset({"frames", "label", "prompt-clicks"})


def test_calibration_sections_are_collapsible_with_prompting_open_by_default() -> None:
    """Every workspace section is a disclosure carrying its default state in markup."""
    static = Path(__file__).parents[1] / "src/battle/static/calibration"
    markup = (static / "index.html").read_text()
    script = (static / "app.js").read_text()

    tags = re.findall(r"<details[^>]*>", markup)
    panels = {re.search(r'data-panel="([^"]+)"', tag).group(1): tag for tag in tags}

    assert set(panels) == set(EXPECTED_PANELS)
    assert markup.count("<summary>") == len(EXPECTED_PANELS)
    assert markup.count('class="panel-count"') == len(EXPECTED_PANELS)
    # Markup is the fresh-profile default; localStorage only overrides it.
    assert {
        panel for panel, tag in panels.items() if re.search(r"\sopen[\s>]", tag)
    } == DEFAULT_OPEN_PANELS
    assert "<h2>" not in markup
    assert "battle.calibration.panels.v1" in script


PANEL_STATE_SCRIPT = r"""
const fs = require("node:fs");
const app = fs.readFileSync(process.argv[1], "utf8");
const html = fs.readFileSync(process.argv[2], "utf8");
// Each section's fresh-profile default comes from the markup's own open attribute.
const panelDefaults = [...html.matchAll(/<details[^>]*>/g)].map((match) => [
  /data-panel="([^"]+)"/.exec(match[0])[1],
  /\sopen[\s>]/.test(match[0]),
]);

const stored = new Map();
globalThis.localStorage = {
  getItem: (key) => (stored.has(key) ? stored.get(key) : null),
  setItem: (key, value) => stored.set(key, String(value)),
};

class Element {
  constructor(selector = "") {
    this.selector = selector; this.children = []; this.dataset = {}; this.style = {};
    this.value = ""; this.checked = false; this.disabled = false; this.hidden = false;
    this.textContent = ""; this.width = 2; this.height = 1;
    this.classList = {toggle() {}};
  }
  addEventListener() {}
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  getBoundingClientRect() { return {left: 0, top: 0, width: 2, height: 1}; }
  setPointerCapture() {}
  getContext() {
    return {
      arc() {}, beginPath() {}, clearRect() {}, drawImage() {}, fill() {}, fillRect() {},
      fillText() {}, getImageData() { return null; }, lineTo() {}, moveTo() {},
      putImageData() {}, restore() {}, save() {}, scale() {}, setTransform() {},
      stroke() {}, strokeRect() {}, translate() {},
    };
  }
  querySelector() { return null; }
  querySelectorAll() { return []; }
}

class Panel extends Element {
  constructor(name, defaultOpen) {
    super(`details[${name}]`);
    this.dataset = {panel: name};
    this.badge = new Element(".panel-count");
    this.classes = new Set();
    this.classList = {
      toggle: (name, on) => { if (on) this.classes.add(name); else this.classes.delete(name); },
    };
    this.toggleHandlers = [];
    this._open = defaultOpen === true;
  }
  get open() { return this._open; }
  set open(value) {
    const next = value === true;
    if (next === this._open) return;
    this._open = next;
    for (const handler of this.toggleHandlers) handler();
  }
  addEventListener(name, handler) { if (name === "toggle") this.toggleHandlers.push(handler); }
  querySelector(selector) { return selector === ".panel-count" ? this.badge : null; }
}

const selectors = [
  "#canvas", "#status", "#labels", "#time-slider", "#timestamp", "#filmstrip",
  "#prompts", "#table", "#diff", "#candidates", "#eligible", "#empty",
  "#target-policy", "#custom-label", "#load-frame", "#delete", "#duplicate",
  "#decode", "#decode-selected", "#finalize-plan", "#prompt-mode", "#clear-points",
  "#point-guidance", "#active-candidate", "#toggle-rejected", "#mask-visible",
  "#mask-opacity", "#mask-opacity-value", "#view-aids-summary", "#view-aids-backend",
  "#view-aids-bypass", "#view-aids-reset", "#view-aid-stages", "#view-aids-timing",
];
const manifest = {
  proxy_dimensions: {width: 2, height: 1},
  proxy_fps: 30,
  proxy_frame_count: 300,
  requested_proxy_timestamps_seconds: [0],
  workspace: {active_proxy_timestamp_seconds: 0, pending_boxes: []},
  candidates: [],
};
globalThis.fetch = async (path) => {
  if (path === "/api/state" || path === "/api/workspace") {
    return {ok: true, status: 200, json: async () => ({
      manifest, manual_seed_targets: [], manual_seed_target_policy: null,
      correction_policy: null, last_diff: [], worker_online: true,
    })};
  }
  if (path.startsWith("/api/frame")) {
    return {ok: true, status: 200, json: async () => ({frame_index: 0, image_uri: "frame.jpg"})};
  }
  if (path === "/api/view-filters") {
    return {ok: true, status: 200, json: async () => ({
      operators: [], stage_order: [], stage_labels: {},
      vigra_available: true, vigra_version: "test", unavailable_reason: "",
    })};
  }
  throw new Error(`unexpected request: ${path}`);
};
globalThis.Image = class Image extends Element {
  naturalWidth = 2;
  naturalHeight = 1;
  set src(value) { this._src = value; if (this.onload) this.onload(); }
};
globalThis.requestAnimationFrame = (callback) => callback();
const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

let keydown = null;
function mountDocument() {
  const panels = Object.fromEntries(
    panelDefaults.map(([name, defaultOpen]) => [name, new Panel(name, defaultOpen)])
  );
  const elements = Object.fromEntries(
    selectors.map((selector) => [selector, new Element(selector)])
  );
  elements["#canvas"].width = 2; elements["#canvas"].height = 1;
  globalThis.document = {
    activeElement: {tagName: "BODY"},
    addEventListener(name, handler) { if (name === "keydown") keydown = handler; },
    createElement(tag) { return new Element(tag); },
    createTextNode(value) { return String(value); },
    querySelector(selector) { return elements[selector]; },
    querySelectorAll(selector) {
      return selector === "details[data-panel]" ? Object.values(panels) : [];
    },
  };
  return {panels, elements};
}

(async () => {
  const first = mountDocument();
  eval(app);
  await tick(); await tick(); await tick();

  const fresh = Object.values(first.panels).filter((panel) => panel.open)
    .map((panel) => panel.dataset.panel).sort().join(",");
  if (fresh !== "frames,label,prompt-clicks") {
    throw new Error(`unexpected fresh-profile sections: ${fresh}`);
  }
  for (const name of ["frames", "candidates", "finalize", "table"]) {
    if (!first.panels[name].badge.textContent) {
      throw new Error(`${name} header carries no summary`);
    }
  }
  if (!first.panels.frames.badge.textContent.includes("1 frame")) {
    throw new Error(`unexpected frame summary: ${first.panels.frames.badge.textContent}`);
  }
  if (!first.panels.candidates.badge.textContent.includes("0/4 done")) {
    throw new Error(`unexpected review summary: ${first.panels.candidates.badge.textContent}`);
  }

  // Collapsing a default-open section and opening a default-closed one both persist.
  first.panels["prompt-clicks"].open = false;
  first.panels.candidates.open = true;
  await tick();

  // A shortcut for a collapsed section still runs and updates that section's header.
  keydown({key: "f", preventDefault() {}});
  await tick();
  if (first.panels["prompt-clicks"].open) {
    throw new Error("a shortcut expanded a collapsed section");
  }
  if (!first.panels["prompt-clicks"].badge.textContent.startsWith("foreground")) {
    throw new Error(`shortcut did not reach the collapsed section: ${
      first.panels["prompt-clicks"].badge.textContent}`);
  }

  const second = mountDocument();
  eval(app);
  await tick(); await tick(); await tick();
  const restored = Object.values(second.panels).filter((panel) => panel.open)
    .map((panel) => panel.dataset.panel).sort().join(",");
  if (restored !== "candidates,frames,label") {
    throw new Error(`stored preferences were not honored: ${restored}`);
  }
})().catch((error) => {
  console.error(error.stack || error);
  process.exitCode = 1;
});
"""


def test_calibration_panels_apply_markup_defaults_and_remember_state() -> None:
    """Fresh profiles follow the markup defaults; a stored choice wins on the next load."""
    node = require_executable("node", "the calibration panel regression test")
    static = Path(__file__).parents[1] / "src/battle/static/calibration"
    completed = subprocess.run(
        [
            node,
            "-e",
            PANEL_STATE_SCRIPT,
            str(static / "app.js"),
            str(static / "index.html"),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr


def test_accept_candidate_allows_later_keyframes_and_guards_frame_zero_roles(
    tmp_path: Path,
) -> None:
    """A later keyframe accepts masks and corrections; only frame 0 is initialization."""
    workspace = make_correction_workspace(tmp_path)
    try:
        first, later = decode_candidates(workspace, [(0, "left_hand"), (10, "left_hand")])

        workspace.accept_candidate(later.candidate_id, 0, eligible=False)
        stored = {item.candidate_id: item for item in workspace.manifest.candidates}[
            later.candidate_id
        ]
        assert stored.human_accepted and stored.human_selected_candidate_index == 0
        assert not stored.selected_for_finalization and not stored.selected_for_correction

        workspace.accept_candidate(later.candidate_id, 0, eligible=False, correction=True)
        stored = {item.candidate_id: item for item in workspace.manifest.candidates}[
            later.candidate_id
        ]
        assert stored.selected_for_correction and not stored.selected_for_finalization

        with pytest.raises(ValueError, match="only frame-0 masks"):
            workspace.accept_candidate(later.candidate_id, 0, eligible=True)
        with pytest.raises(ValueError, match="only later-frame masks"):
            workspace.accept_candidate(first.candidate_id, 0, eligible=True, correction=True)
    finally:
        workspace.close()


class MultiMaskFixtureDecoder(FixtureDecoder):
    """Return four mask hypotheses so alternate-mask selection can be exercised."""

    candidate_count = 4


def test_alternate_mask_selection_changes_the_human_choice_on_any_keyframe(
    tmp_path: Path,
) -> None:
    """Keys 1-4 and the mask buttons must retarget the selection while a plan is a draft."""
    workspace = make_correction_workspace(tmp_path)
    workspace.decoder = MultiMaskFixtureDecoder()
    try:
        initial, later = decode_candidates(workspace, [(0, "left_hand"), (10, "left_hand")])

        workspace.accept_candidate(initial.candidate_id, 0, eligible=True)
        workspace.accept_candidate(initial.candidate_id, 2, eligible=True)
        stored = {item.candidate_id: item for item in workspace.manifest.candidates}
        assert stored[initial.candidate_id].human_selected_candidate_index == 2
        assert stored[initial.candidate_id].selected_for_finalization

        workspace.accept_candidate(later.candidate_id, 1, eligible=False, correction=True)
        workspace.accept_candidate(later.candidate_id, 3, eligible=False, correction=True)
        stored = {item.candidate_id: item for item in workspace.manifest.candidates}
        assert stored[later.candidate_id].human_selected_candidate_index == 3
        assert stored[later.candidate_id].selected_for_correction

        persisted = json.loads(workspace.manifest_path.read_text())["candidates"]
        assert {
            item["candidate_id"]: item["human_selected_candidate_index"] for item in persisted
        } == {initial.candidate_id: 2, later.candidate_id: 3}
    finally:
        workspace.close()


def test_finalized_plan_refuses_selection_until_it_is_reopened(tmp_path: Path) -> None:
    """The lock explains itself, survives the attempt intact, and reopening lifts it."""
    workspace = make_workspace(tmp_path)
    workspace.decoder = MultiMaskFixtureDecoder()
    server, base = serve(workspace)
    proposal_path = tmp_path / "proposed_tracking_prompt.json"
    try:
        (candidate,) = decode_candidates(workspace, [(0, "left_hand")])
        workspace.accept_candidate(candidate.candidate_id, 0, eligible=True)
        post_json(base, "/api/finalize-tracking-plan", {})
        first_revision = proposal_path.read_bytes()
        manifest_before = workspace.manifest_path.read_bytes()
        assert workspace.snapshot()["plan"] == {
            "finalized": True,
            "plan_revision": 1,
            "final_proposal_uri": str(proposal_path),
            "final_correction_schedule_uri": None,
            "superseded_plans": [],
            "lock_reason": CANDIDATE_REVIEW_LOCK_REASON,
        }

        with pytest.raises(urllib.error.HTTPError) as refused:
            post_json(
                base,
                f"/api/candidates/{candidate.candidate_id}/accept",
                {"candidate_index": 2, "eligible": True},
            )
        message = json.loads(refused.value.read())["error"]
        assert "candidate review is locked" in message
        assert "Reopen for editing" in message
        with pytest.raises(ValueError, match="candidate review is locked"):
            workspace.finalize_tracking_plan()
        assert proposal_path.read_bytes() == first_revision
        assert workspace.manifest_path.read_bytes() == manifest_before

        reopened = post_json(base, "/api/reopen-tracking-plan", {})

        assert reopened["plan"]["finalized"] is False
        assert reopened["plan"]["lock_reason"] is None
        assert reopened["plan"]["superseded_plans"] == [
            {
                "schema_version": reopened["plan"]["superseded_plans"][0]["schema_version"],
                "plan_revision": 1,
                "proposal_uri": str(proposal_path),
                "correction_schedule_uri": None,
            }
        ]
        assert proposal_path.read_bytes() == first_revision

        post_json(
            base,
            f"/api/candidates/{candidate.candidate_id}/accept",
            {"candidate_index": 2, "eligible": True},
        )
        assert workspace.manifest.candidates[0].human_selected_candidate_index == 2

        plan = post_json(base, "/api/finalize-tracking-plan", {})

        second_path = tmp_path / "proposed_tracking_prompt.r2.json"
        assert proposal_path.read_bytes() == first_revision
        assert second_path.is_file()
        assert plan["proposal"]["seeds"][0]["human_selected_candidate_index"] == 2
        assert plan["proposal"]["calibration_manifest_sha256"] == sha256_file(
            workspace.manifest_path
        )
        assert json.loads(second_path.read_text())["calibration_manifest_sha256"] == sha256_file(
            workspace.manifest_path
        )
        persisted = json.loads(workspace.manifest_path.read_text())
        assert persisted["final_proposal_uri"] == str(second_path)
        assert persisted["plan_revision"] == 2
        assert [plan["plan_revision"] for plan in persisted["superseded_plans"]] == [1]
    finally:
        server.shutdown()
        server.server_close()
        workspace.close()


def test_reopening_a_draft_workspace_is_refused(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    try:
        with pytest.raises(ValueError, match="already open for editing"):
            workspace.reopen_tracking_plan()
    finally:
        workspace.close()


def test_reopened_correction_plan_rehashes_both_artifacts(tmp_path: Path) -> None:
    """A re-finalized schedule must fingerprint the manifest it was actually written with."""
    workspace = make_correction_workspace(tmp_path)
    workspace.decoder = MultiMaskFixtureDecoder()
    targets = ("left_hand", "right_hand", "yellow_toy_top", "black_toy_top_base")
    try:
        candidates = decode_candidates(
            workspace, [(0, target) for target in targets] + [(10, "left_hand")]
        )
        for candidate in candidates[:-1]:
            workspace.accept_candidate(candidate.candidate_id, 0, eligible=True)
        workspace.accept_candidate(candidates[-1].candidate_id, 0, eligible=False, correction=True)
        write_selected_masks(workspace)
        workspace.finalize_tracking_plan()
        first_proposal = (tmp_path / "proposed_tracking_prompt.json").read_bytes()
        first_schedule = (tmp_path / "multi_keyframe_correction_schedule.json").read_bytes()

        workspace.reopen_tracking_plan()
        workspace.accept_candidate(candidates[0].candidate_id, 3, eligible=True)
        write_selected_masks(workspace)
        plan = workspace.finalize_tracking_plan()

        assert (tmp_path / "proposed_tracking_prompt.json").read_bytes() == first_proposal
        assert (tmp_path / "multi_keyframe_correction_schedule.json").read_bytes() == first_schedule
        manifest_sha256 = sha256_file(workspace.manifest_path)
        assert plan["proposal"]["calibration_manifest_sha256"] == manifest_sha256
        assert plan["correction_schedule"]["calibration_manifest_fingerprint"]["sha256"] == (
            manifest_sha256
        )
        schedule = json.loads((tmp_path / "multi_keyframe_correction_schedule.r2.json").read_text())
        assert schedule["calibration_manifest_fingerprint"]["sha256"] == manifest_sha256
        assert len(schedule["corrections"]) == len(targets) + 1
        persisted = json.loads(workspace.manifest_path.read_text())
        assert persisted["superseded_plans"][0] == {
            "schema_version": persisted["schema_version"],
            "plan_revision": 1,
            "proposal_uri": str(tmp_path / "proposed_tracking_prompt.json"),
            "correction_schedule_uri": str(tmp_path / "multi_keyframe_correction_schedule.json"),
        }
    finally:
        workspace.close()


def test_refinalizing_a_superseded_schedule_still_requires_its_policy(tmp_path: Path) -> None:
    """Reopening must not become a way to drop the correction policy from the plan."""
    workspace = make_correction_workspace(tmp_path)
    targets = ("left_hand", "right_hand", "yellow_toy_top", "black_toy_top_base")
    try:
        candidates = decode_candidates(workspace, [(0, target) for target in targets])
        for candidate in candidates:
            workspace.accept_candidate(candidate.candidate_id, 0, eligible=True)
        write_selected_masks(workspace)
        workspace.finalize_tracking_plan()
        workspace.reopen_tracking_plan()
        workspace.correction_policy_path = None

        with pytest.raises(ValueError, match="already wrote a correction schedule"):
            workspace.finalize_tracking_plan()

        assert not (tmp_path / "proposed_tracking_prompt.r2.json").exists()
    finally:
        workspace.close()


def test_calibration_ui_puts_the_plan_lock_above_every_collapsible_panel() -> None:
    static = Path(__file__).parents[1] / "src/battle/static/calibration"
    markup = (static / "index.html").read_text()

    assert markup.index('id="plan-lock"') < markup.index("<main>")
    assert "Reopen for editing" in (static / "app.js").read_text()
    assert ".plan-lock" in (static / "style.css").read_text()


def test_calibration_ui_offers_only_the_role_each_keyframe_can_play() -> None:
    """Later-keyframe cards expose corrections, never a dead frame-0 eligibility control."""
    node = require_executable("node", "the calibration review regression test")
    root = Path(__file__).parents[1]
    script = r"""
const fs = require("node:fs");
const app = fs.readFileSync(process.argv[1], "utf8");
const targets = ["left_hand", "right_hand"];

class Element {
  constructor(selector = "") {
    this.selector = selector; this.children = []; this.dataset = {}; this.style = {};
    this.value = ""; this.checked = false; this.disabled = false; this.hidden = false;
    this.title = ""; this.textContent = "";
    this.classList = {toggle() {}, add() {}, remove() {}};
  }
  addEventListener() {}
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  getBoundingClientRect() { return {left: 0, top: 0, width: 954, height: 720}; }
  getContext() {
    return {
      clearRect() {}, drawImage() {}, fillRect() {}, fillText() {}, restore() {}, save() {},
      scale() {}, strokeRect() {}, translate() {}, setTransform() {},
    };
  }
  querySelectorAll() { return []; }
  querySelector() { return null; }
}
const selectors = [
  "#canvas", "#status", "#labels", "#time-slider", "#timestamp", "#filmstrip",
  "#table", "#diff", "#candidate-status", "#eligible", "#empty", "#plan-lock",
  "#custom-label", "#custom-label-control", "#load-frame", "#delete", "#decode",
  "#finalize-plan", "#active-candidate", "#label-panel", "#prompt-clicks-panel",
  "#prompt-mode", "#clear-points", "#live-decode", "#live-decode-delay",
];
const elements = Object.fromEntries(selectors.map((s) => [s, new Element(s)]));
elements["#canvas"].width = 954; elements["#canvas"].height = 720;
globalThis.document = {
  activeElement: {tagName: "BODY"},
  addEventListener() {},
  createElement(tag) { return new Element(tag); },
  createTextNode(value) { return String(value); },
  querySelector(selector) { return elements[selector] || null; },
  querySelectorAll() { return []; },
};
globalThis.Image = class Image extends Element {
  constructor() { super("img"); }
  set src(value) { this._src = value; if (this.onload) this.onload(); }
};
globalThis.requestAnimationFrame = (callback) => callback();
/* Frame 0 holds one required initial mask per target; 10 s is a later correction
 * keyframe where only left_hand is visible and nothing has been reviewed yet. */
const makeCandidate = (target, slot, frameIndex, seconds, overrides) => ({
  candidate_id: `t${String(frameIndex).padStart(6, "0")}-b0${slot}`,
  intended_target: target,
  frame: {analysis_frame_index: frameIndex, proxy_seconds: seconds},
  pixel_box: {x1: 10, y1: 20, x2: 100, y2: 120},
  human_selected_candidate_index: null, human_accepted: false,
  selected_for_finalization: false, selected_for_correction: false, rejected: false,
  decoder_result: {
    deterministic_best_candidate_index: 1,
    candidates: [0, 1].map((candidate_index) => ({
      candidate_index, iou_score: 0.5 + candidate_index / 10,
      review_uri: `results/${target}-${frameIndex}-${candidate_index}.png`,
      is_deterministic_best: candidate_index === 1,
    })),
  },
  ...overrides,
});
const manifest = {
  proxy_dimensions: {width: 954, height: 720}, proxy_fps: 30,
  requested_proxy_timestamps_seconds: [0, 10],
  workspace: {active_proxy_timestamp_seconds: 10, pending_boxes: []},
  candidates: [
    ...targets.map((target, index) => makeCandidate(target, index + 1, 0, 0, {
      human_selected_candidate_index: 1, human_accepted: true,
      selected_for_finalization: true,
    })),
    makeCandidate("left_hand", 1, 300, 10, {}),
  ],
};
const posted = [];
globalThis.fetch = async (path, options = {}) => {
  if (path.startsWith("/api/candidates/")) {
    posted.push({path, body: JSON.parse(options.body)});
    return {ok: true, status: 200, json: async () => ({})};
  }
  if (path === "/api/state" || path === "/api/workspace") {
    return {ok: true, status: 200, json: async () => ({
      manifest, manual_seed_targets: targets,
      manual_seed_target_policy: {config_id: "two-target", required_target_count: 2},
      correction_policy: {maximum_later_correction_keyframes_per_target: 2},
      last_diff: [], worker_online: true,
    })};
  }
  if (path.startsWith("/api/frame")) {
    return {ok: true, status: 200, json: async () => ({frame_index: 300, image_uri: "f.jpg"})};
  }
  if (path.startsWith("/api/view-filters")) {
    return {ok: true, status: 200, json: async () => ({stages: [], backends: {}})};
  }
  throw new Error(`unexpected request: ${path}`);
};
globalThis.setInterval = () => 1;
globalThis.clearInterval = () => {};
const tick = () => new Promise((resolve) => setTimeout(resolve, 0));
const flatten = (node) => [node, ...node.children.flatMap(
  (child) => (child instanceof Element ? flatten(child) : [])
)];
const cardText = (card) => flatten(card).flatMap(
  (node) => [node.textContent, ...node.children.filter((c) => typeof c === "string")]
).join(" ");
const reload = async () => {
  elements["#timestamp"].value = 10;
  await elements["#load-frame"].onclick();
  await tick(); await tick();
};
(async () => {
  eval(app);
  await tick(); await tick(); await tick();
  await reload();
  let table = elements["#candidate-status"].children[0];
  if (table.children.length !== 3) {
    throw new Error("status table must contain both configured frames");
  }
  const initialStatus = table.children[1].children[1].children[0].children[0];
  const laterStatus = table.children[2].children[1].children[0].children[0];
  if (initialStatus.textContent !== "✓ Done" || !initialStatus.title.includes("initial mask")) {
    throw new Error(
      `frame-0 initial status is unclear: ${initialStatus.textContent} ${initialStatus.title}`
    );
  }
  if (laterStatus.textContent !== "◐ Preview") {
    throw new Error(`later unscheduled preview status is unclear: ${laterStatus.textContent}`);
  }

  // Once accepted and scheduled, the later cell names its correction role.
  manifest.candidates[2].human_accepted = true;
  manifest.candidates[2].human_selected_candidate_index = 0;
  manifest.candidates[2].selected_for_correction = true;
  await reload();
  table = elements["#candidate-status"].children[0];
  const scheduled = table.children[2].children[1].children[0].children[0];
  if (scheduled.textContent !== "✓ Done"
      || !scheduled.title.includes("scheduled as the correction")) {
    throw new Error(
      `later correction role is unclear: ${scheduled.textContent} ${scheduled.title}`
    );
  }
  const finalizeText = elements["#eligible"].children.map((n) => n.textContent).join(" ");
  if (!finalizeText.includes("10.000 s keyframe: left hand")) {
    throw new Error(`active-keyframe correction summary missing: ${finalizeText}`);
  }
})().catch((error) => {
  console.error(error.stack || error);
  process.exitCode = 1;
});
"""
    completed = subprocess.run(
        [node, "-e", script, str(root / "src/battle/static/calibration/app.js")],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr


def test_calibration_ui_shows_the_plan_lock_and_reopens_for_editing() -> None:
    """A finalized plan disables review with a reason; Reopen restores mask selection."""
    node = require_executable("node", "the calibration plan-lock regression test")
    root = Path(__file__).parents[1]
    script = r"""
const fs = require("node:fs");
const app = fs.readFileSync(process.argv[1], "utf8");
const lockReason = "candidate review is locked after finalizing a tracking plan; "
  + "completed plan artifacts remain unchanged. Choose Reopen for editing to amend the "
  + "plan; the finalized artifacts stay on disk as a superseded revision.";

class Element {
  constructor(selector = "") {
    this.selector = selector; this.children = []; this.dataset = {}; this.style = {};
    this.value = ""; this.checked = false; this.disabled = false; this.hidden = false;
    this.title = ""; this.textContent = "";
    this.classList = {toggle() {}, add() {}, remove() {}};
  }
  addEventListener() {}
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  getBoundingClientRect() { return {left: 0, top: 0, width: 954, height: 720}; }
  getContext() {
    return {
      clearRect() {}, drawImage() {}, fillRect() {}, fillText() {}, restore() {}, save() {},
      scale() {}, strokeRect() {}, translate() {}, setTransform() {},
    };
  }
  querySelectorAll() { return []; }
  querySelector() { return null; }
}
const selectors = [
  "#canvas", "#status", "#labels", "#time-slider", "#timestamp", "#filmstrip",
  "#table", "#diff", "#candidate-status", "#eligible", "#empty", "#plan-lock",
  "#custom-label", "#custom-label-control", "#load-frame", "#delete", "#decode",
  "#finalize-plan", "#active-candidate", "#label-panel", "#prompt-clicks-panel",
  "#prompt-mode", "#clear-points", "#live-decode", "#live-decode-delay",
];
const elements = Object.fromEntries(selectors.map((s) => [s, new Element(s)]));
elements["#canvas"].width = 954; elements["#canvas"].height = 720;
elements["#plan-lock"].hidden = true;
globalThis.document = {
  activeElement: {tagName: "BODY"},
  addEventListener() {},
  createElement(tag) { return new Element(tag); },
  createTextNode(value) { return String(value); },
  querySelector(selector) { return elements[selector] || null; },
  querySelectorAll() { return []; },
};
globalThis.Image = class Image extends Element {
  constructor() { super("img"); }
  set src(value) { this._src = value; if (this.onload) this.onload(); }
};
globalThis.requestAnimationFrame = (callback) => callback();
globalThis.setInterval = () => 1;
globalThis.clearInterval = () => {};
const manifest = {
  proxy_dimensions: {width: 954, height: 720},
      proxy_fps: 30,
  requested_proxy_timestamps_seconds: [0, 10],
  workspace: {active_proxy_timestamp_seconds: 0, pending_boxes: []},
  candidates: [{
    candidate_id: "t000000-b01", intended_target: "left_hand",
    frame: {analysis_frame_index: 0, proxy_seconds: 0},
    pixel_box: {x1: 10, y1: 20, x2: 100, y2: 120},
    human_selected_candidate_index: 0, human_accepted: true,
    selected_for_finalization: true, selected_for_correction: false, rejected: false,
    decoder_result: {
      deterministic_best_candidate_index: 0,
      candidates: [0, 1].map((candidate_index) => ({
        candidate_index, iou_score: 0.5 - candidate_index / 100,
        review_uri: `results/left_hand-${candidate_index}.png`,
        is_deterministic_best: candidate_index === 0,
      })),
    },
  }],
};
const plan = {
  finalized: true, plan_revision: 1,
  final_proposal_uri: "runs/e4/proposed_tracking_prompt.json",
  final_correction_schedule_uri: null, superseded_plans: [], lock_reason: lockReason,
};
const posted = [];
globalThis.fetch = async (path, options = {}) => {
  if (path === "/api/reopen-tracking-plan") {
    posted.push({path});
    plan.finalized = false; plan.lock_reason = null;
    plan.final_proposal_uri = null;
    plan.superseded_plans = [{
      plan_revision: 1, proposal_uri: "runs/e4/proposed_tracking_prompt.json",
      correction_schedule_uri: null,
    }];
    return {ok: true, status: 200, json: async () => ({plan})};
  }
  if (path.startsWith("/api/candidates/")) {
    posted.push({path, body: JSON.parse(options.body)});
    return {ok: true, status: 200, json: async () => ({})};
  }
  if (path === "/api/state" || path === "/api/workspace") {
    return {ok: true, status: 200, json: async () => ({
      manifest, manual_seed_targets: ["left_hand"],
      manual_seed_target_policy: {config_id: "one-target", required_target_count: 1},
      correction_policy: null, plan, last_diff: [], worker_online: true,
    })};
  }
  if (path.startsWith("/api/frame")) {
    return {ok: true, status: 200, json: async () => ({frame_index: 0, image_uri: "f.jpg"})};
  }
  if (path.startsWith("/api/view-filters")) {
    return {ok: true, status: 200, json: async () => ({stages: [], backends: {}})};
  }
  throw new Error(`unexpected request: ${path}`);
};
const tick = () => new Promise((resolve) => setTimeout(resolve, 0));
const flatten = (node) => [node, ...node.children.flatMap(
  (child) => (child instanceof Element ? flatten(child) : [])
)];
const card = () => elements["#candidates"].children[0];
const inputs = (type) => flatten(card()).filter((node) => node.type === type);
const bannerText = () => flatten(elements["#plan-lock"]).flatMap(
  (node) => [node.textContent, ...node.children.filter((child) => typeof child === "string")]
).join(" ");
const statusClear = () => elements["#candidate-status"].children[0]
  .children[1].children[1].children[0].children[1];
(async () => {
  eval(app);
  await tick(); await tick(); await tick(); await tick();

  // The lock is announced outside the review panel, with the server's own reason.
  if (elements["#plan-lock"].hidden) throw new Error("finalized plan left the lock banner hidden");
  if (!bannerText().includes("revision 1 is finalized")) {
    throw new Error(`lock banner does not name the revision: ${bannerText()}`);
  }
  if (!bannerText().includes("Reopen for editing to amend the plan")) {
    throw new Error(`lock banner does not carry the server reason: ${bannerText()}`);
  }
  if (!elements["#finalize-plan"].disabled) {
    throw new Error("a finalized plan left Finalize tracking plan enabled");
  }
  if (!statusClear().disabled || statusClear().title !== lockReason) {
    throw new Error("locked status-table action is not disabled with the server reason");
  }
  if (!elements["#live-decode"].disabled || !elements["#live-decode-delay"].disabled) {
    throw new Error("finalized plan left live-decode controls enabled");
  }

  const reopen = flatten(elements["#plan-lock"]).find(
    (node) => node.textContent === "Reopen for editing"
  );
  if (!reopen) throw new Error("the lock banner offers no reopen affordance");
  await reopen.onclick();
  await tick();
  if (posted[0].path !== "/api/reopen-tracking-plan") {
    throw new Error(`reopen did not reach the server: ${JSON.stringify(posted)}`);
  }
  if (!elements["#plan-lock"].hidden) throw new Error("reopened workspace kept the lock banner");
  if (elements["#finalize-plan"].disabled) {
    throw new Error("reopened workspace left Finalize tracking plan disabled");
  }
  const finalizeText = elements["#eligible"].children.map((node) => node.textContent).join(" ");
  if (!finalizeText.includes("Kept unchanged on disk: revision 1")) {
    throw new Error(`superseded revision is not reported: ${finalizeText}`);
  }

  if (statusClear().disabled) {
    throw new Error("reopened workspace left the status-table action disabled");
  }
  if (elements["#live-decode"].disabled || elements["#live-decode-delay"].disabled) {
    throw new Error("reopened workspace left live-decode controls disabled");
  }
})().catch((error) => {
  console.error(error.stack || error);
  process.exitCode = 1;
});
"""
    completed = subprocess.run(
        [node, "-e", script, str(root / "src/battle/static/calibration/app.js")],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
