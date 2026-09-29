"""`battle-finebio-tips` on synthetic rows, a fake frame source and the real P03 rig fixture."""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import tomllib
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import SERVER_POLL_INTERVAL_SECONDS
from finebio_fixtures import load_preflight_fixtures

from battle import finebio_tips as ft
from battle import fs_common
from battle.finebio_cameras import Camera

TIMESTAMP = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
PAGE_DATA = r'<script id="page-data" type="application/json">(.*?)</script>'
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------------------------
# a synthetic bench: blue at rest, yellow picked up, swung about and dipped, hands on it


def _det(view: str, frame: int, cls: str, box: list[float], *, pose_valid: bool = True) -> dict:
    return {
        "view": view,
        "frame_index": frame,
        "slot": f"{cls}#0",
        "object_class": cls,
        "detector_score": 0.9,
        "box_xyxy_px": box,
        "pose_valid": pose_valid,
        "source": "detector",
    }


def _sam(
    view: str,
    frame: int,
    cls: str,
    box: list[float],
    *,
    area: int,
    width: float,
    pose_valid: bool = True,
) -> dict:
    cx = (box[0] + box[2]) / 2
    return {
        **_det(view, frame, cls, box, pose_valid=pose_valid),
        "mask_bbox_px": box,
        "mask_centroid_px": [cx, (box[1] + box[3]) / 2],
        "mask_area_px": area,
        "mask_axis_px": [[cx, box[1]], [cx, box[3]]],
        "mask_elongation": 5.0,
        "mask_width_px": width,
        "source": "sam3_decode",
    }


def synthetic_rows(width: int = 1920, height: int = 1080, frames: int = 300) -> list[dict]:
    """300 frames in T4, T5 and the fpv. Blue stands still throughout. Yellow stands still
    until frame 100, swings sideways (fast enough to count as moving on most frames) until
    200 with a right hand on its plunger end, and dips its tip to the bench on 150-169. The
    fpv pose is invalid on 50-59. Red and the 8-channel never appear."""

    def box(x0: float, y0: float, x1: float, y1: float) -> list[float]:
        return [
            round(x0 * width, 1),
            round(y0 * height, 1),
            round(x1 * width, 1),
            round(y1 * height, 1),
        ]

    rows: list[dict] = []
    for f in range(frames):
        fpv_valid = not (50 <= f < 60)
        blue = box(0.05, 0.2, 0.08, 0.6)
        for view, area in (("T4", 4500), ("T5", 2500)):
            rows.append(_det(view, f, "blue_pipette", blue))
            rows.append(_sam(view, f, "blue_pipette", blue, area=area, width=30.0))
        if 100 <= f < 200:
            x = 0.3 + 0.25 * math.sin((f - 100) / 8)
            bottom = 0.95 if 150 <= f < 170 else 0.6
            hand = True
        else:
            x, bottom, hand = 0.3, 0.6, False
        yellow = box(x, 0.2, x + 0.03, bottom)
        hand_box = box(x - 0.05, 0.12, x + 0.08, 0.3)
        for view, area in (("T4", 5000), ("T5", 3000)):
            rows.append(_det(view, f, "yellow_pipette", yellow))
            rows.append(_sam(view, f, "yellow_pipette", yellow, area=area, width=30.0))
            if hand:
                rows.append(_det(view, f, "right_hand", hand_box))
        rows.append(_det("fpv", f, "yellow_pipette", yellow, pose_valid=fpv_valid))
        rows.append(
            _sam("fpv", f, "yellow_pipette", yellow, area=4000, width=15.0, pose_valid=fpv_valid)
        )
        if hand:
            rows.append(_det("fpv", f, "right_hand", hand_box, pose_valid=fpv_valid))
    return rows


def synthetic_clip(width: int, height: int, frames: int = 300) -> dict[str, Any]:
    return {
        "config_kind": "finebio_clip_config",
        "trial": "TEST",
        "window": {"start_frame": 0, "end_frame_exclusive": frames},
        "frame_index_offset": 0,
        "views": ["T4", "T5", "fpv"],
        "fixed_views": ["T4", "T5"],
        "view_sizes": {"T4": [width, height], "T5": [width, height], "fpv": [width, height]},
        "proxies": {},
        "window_camera_config": "configs/finebio/cameras/TEST.json",
    }


def fake_frames(width: int, height: int):
    def frames(view: str, raw: int) -> np.ndarray:
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[..., 0] = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
        image[..., 1] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
        image[..., 2] = (raw * 7) % 256
        return image

    return frames


def make_workspace(tmp_path: Path, *, width: int, height: int, frames: int = 3) -> Path:
    root = tmp_path / "tips"
    rows = synthetic_rows(width, height)
    observations = tmp_path / "observations.jsonl"
    observations.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    clip = synthetic_clip(width, height)
    clip_path = tmp_path / "clip.json"
    fs_common.write_json(clip_path, clip)
    tables = ft.build_tables(rows, window=(0, 300), fixed_views=clip["fixed_views"])
    params = ft.SelectionParams(frames=frames, spacing_frames=5)
    ft.build_workspace(
        clip=clip,
        clip_path=clip_path,
        observations=observations,
        tables=tables,
        params=params,
        output=root,
        frames=fake_frames(width, height),
        repository_root=tmp_path,
    )
    return root


def serve_app(app: ft.TipsApp) -> tuple[ThreadingHTTPServer, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), ft.make_tips_handler(app))
    threading.Thread(
        target=server.serve_forever, args=(SERVER_POLL_INTERVAL_SECONDS,), daemon=True
    ).start()
    return server, f"http://127.0.0.1:{server.server_port}"


@pytest.fixture
def served(tmp_path: Path):
    root = make_workspace(tmp_path, width=96, height=64)
    app = ft.TipsApp.open(workspace_dir=root, author="tester")
    server, base = serve_app(app)
    try:
        yield root, app, base
    finally:
        server.shutdown()
        server.server_close()


def get(base: str, path: str) -> tuple[int, bytes, str]:
    request = urllib.request.Request(base + path)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read(), response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as error:
        return error.code, error.read(), error.headers.get("Content-Type", "")


def post_json(base: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def post_error(base: str, path: str, payload: dict[str, Any]) -> tuple[int, str]:
    with pytest.raises(urllib.error.HTTPError) as failure:
        post_json(base, path, payload)
    return failure.value.code, json.loads(failure.value.read())["error"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------------------------
# frame states and the choice


def test_states_rest_held_low_and_transition_on_synthetic_rows() -> None:
    rows = synthetic_rows()
    tables = ft.build_tables(rows, window=(0, 300), fixed_views=("T4", "T5"))
    params = ft.SelectionParams(frames=9, spacing_frames=5)
    assert tables.views == ("fpv", "T4", "T5")
    assert tables.fpv_valid[55] is False and tables.fpv_valid[120] is True
    assert len(tables.hands[(120, "T4")]) == 1 and (20, "T4") not in tables.hands

    assert ft.moving_frames(tables, "blue_pipette", params) == {}
    moving = ft.moving_frames(tables, "yellow_pipette", params)
    assert moving and all(100 <= f <= 200 for f in moving)  # 200: the snap back to rest
    assert set(moving[120]) == {"T4", "T5"}

    states = ft.classify_frames(tables, params)
    assert states[20].state == "rest" and "no pipette box moved" in states[20].reason
    assert states[120].state == "held" and "yellow_pipette" in states[120].reason
    assert states[155].state == "low" and "box bottom" in states[155].reason
    assert states[90].state == "transition"
    assert {f for f, s in states.items() if s.state == "low"} == set(range(150, 170))

    # With line tracks the low state reads the resolved tip height instead of the box.
    heights = {(f, "yellow_pipette"): 3.0 for f in range(120, 125)}
    heights.update({(f, "yellow_pipette"): 12.0 for f in range(150, 170)})
    tracked = ft.classify_frames(tables, params, heights)
    assert {f for f, s in tracked.items() if s.state == "low"} == set(range(120, 125))
    assert "line-track tip 3.0 cm" in tracked[122].reason


def test_frame_cells_pick_the_largest_views_and_respect_the_fpv_pose() -> None:
    rows = synthetic_rows()
    tables = ft.build_tables(rows, window=(0, 300), fixed_views=("T4", "T5"))
    params = ft.SelectionParams()
    sizes = {"T4": (1920, 1080), "T5": (1920, 1080), "fpv": (1920, 1080)}
    cells = ft.frame_cells(120, tables, sizes, params)
    by_class = {}
    for cell in cells:
        by_class.setdefault(cell["class"], []).append(cell["view"])
    assert by_class == {"blue_pipette": ["T4", "T5"], "yellow_pipette": ["fpv", "T4", "T5"]}
    yellow_t4 = next(c for c in cells if c["class"] == "yellow_pipette" and c["view"] == "T4")
    # The hand sits on the plunger end at the top, so the tip is the bottom end of the axis.
    assert yellow_t4["suggested_tip_px"][1] == pytest.approx(0.6 * 1080, abs=0.1)
    assert "hand" in yellow_t4["tip_rule"] and yellow_t4["hand_boxes"] == 1
    assert yellow_t4["slot"] == 1 and yellow_t4["label"] == "yellow_pipette#0"
    blue_t4 = next(c for c in cells if c["class"] == "blue_pipette" and c["view"] == "T4")
    assert "no hand box" in blue_t4["tip_rule"]
    assert blue_t4["suggested_tip_px"][1] == pytest.approx(0.6 * 1080, abs=0.1)
    # Frame 55: the fpv pose is invalid, so yellow keeps its two fixed views.
    cells_55 = ft.frame_cells(55, tables, sizes, params)
    assert [c["view"] for c in cells_55 if c["class"] == "yellow_pipette"] == ["T4", "T5"]
    # Only two views per cell when asked; the largest masks win.
    two = ft.frame_cells(120, tables, sizes, ft.SelectionParams(views_per_cell=2))
    assert [c["view"] for c in two if c["class"] == "yellow_pipette"] == ["fpv", "T4"]
    # A pipette with fewer than min_views views is no cell at all.
    strict = ft.frame_cells(120, tables, {"T4": (1920, 1080)}, params)
    assert strict == []


def test_suggested_tip_rules_and_edge_test() -> None:
    axis = [[100.0, 100.0], [100.0, 500.0]]
    tip, rule = ft.suggested_tip(axis, [(60.0, 60.0, 140.0, 160.0)])
    assert tip == (100.0, 500.0) and "hand" in rule
    tip, rule = ft.suggested_tip(axis, [(60.0, 450.0, 140.0, 560.0)])
    assert tip == (100.0, 100.0)
    tip, rule = ft.suggested_tip(axis, [])
    assert tip == (100.0, 500.0) and "no hand box" in rule
    # A hand far from both ends (another pipette's, or idle) does not decide the end.
    tip, rule = ft.suggested_tip(axis, [(900.0, 60.0, 980.0, 160.0)])
    assert tip == (100.0, 500.0) and "no hand box" in rule
    # Both ends inside hand boxes (the palm deep on the lower end, the other glove's box
    # grazing the upper one): the end farther from the nearest hand centre leaves the palm
    # and is the tip side.
    tip, rule = ft.suggested_tip(axis, [(40.0, 300.0, 160.0, 560.0), (0.0, 90.0, 300.0, 400.0)])
    assert tip == (100.0, 100.0) and "hand-box centre" in rule
    assert ft.point_box_distance((0.0, 0.0), (10.0, 0.0, 20.0, 10.0)) == 10.0
    assert ft.point_box_distance((15.0, 5.0), (10.0, 0.0, 20.0, 10.0)) == 0.0
    assert ft.inside_image((5.0, 5.0), (100, 100), 6.0) is False
    assert ft.inside_image((6.0, 94.0), (100, 100), 6.0) is True


def test_select_frames_is_stratified_spaced_and_deterministic() -> None:
    rows = synthetic_rows()
    tables = ft.build_tables(rows, window=(0, 300), fixed_views=("T4", "T5"))
    params = ft.SelectionParams(frames=9, spacing_frames=5)
    states = ft.classify_frames(tables, params)
    sizes = {"T4": (1920, 1080), "T5": (1920, 1080), "fpv": (1920, 1080)}
    eligible = {f for f in range(300) if ft.frame_cells(f, tables, sizes, params)}
    assert eligible == set(range(300))
    chosen = ft.select_frames(states, eligible, params)
    assert chosen == ft.select_frames(states, eligible, params)
    frames = [c["raw_frame"] for c in chosen]
    assert frames == sorted(frames) and len(set(frames)) == 9
    by_state = {}
    for c in chosen:
        by_state[c["state"]] = by_state.get(c["state"], 0) + 1
    assert by_state == {"low": 3, "held": 3, "rest": 3}
    assert all(abs(a - b) >= 5 for a in frames for b in frames if a != b)
    assert all(states[c["raw_frame"]].state == c["state"] for c in chosen)
    assert all(c["reason"] for c in chosen)
    # A thin pool is topped up from the others: only two low frames fit ten frames apart.
    wide = ft.select_frames(states, eligible, ft.SelectionParams(frames=9, spacing_frames=10))
    counts = {}
    for c in wide:
        counts[c["state"]] = counts.get(c["state"], 0) + 1
    assert counts["low"] == 2 and sum(counts.values()) == 9
    # pick_spread: the quantile targets, spacing against what is already taken.
    assert ft.pick_spread(list(range(0, 100, 10)), 3, [], 5) == [0, 50, 90]
    assert ft.pick_spread([10, 12, 14], 2, [10], 5) == []
    assert ft.pick_spread([], 3, [], 5) == [] and ft.pick_spread([7], 1, [], 5) == [7]


# --------------------------------------------------------------------------------------------
# crop geometry and the click mapping


def test_zoom_geometry_and_click_round_trip() -> None:
    geometry = ft.zoom_geometry((1900.0, 1000.0), size=400, zoom=1, image_wh=(1920, 1080))
    assert geometry == {
        "offset": [1520, 680],
        "scale": 1.0,
        "source_size": [400, 400],
        "size": [400, 400],
    }
    assert ft.full_to_crop((1900.0, 1000.0), geometry) == (380.0, 320.0)
    assert ft.crop_to_full((380.0, 320.0), geometry) == (1900.0, 1000.0)

    doubled = ft.zoom_geometry((100.0, 50.0), size=400, zoom=2, image_wh=(1920, 1080))
    assert doubled["offset"] == [0, 0] and doubled["scale"] == 2.0
    assert doubled["source_size"] == [200, 200] and doubled["size"] == [400, 400]
    assert ft.full_to_crop((100.0, 50.0), doubled) == (200.0, 100.0)
    assert ft.crop_to_full((200.0, 100.0), doubled) == (100.0, 50.0)

    centred = ft.zoom_geometry((960.0, 540.0), size=400, zoom=1, image_wh=(1920, 1080))
    assert centred["offset"] == [760, 340]
    for point in ((960.0, 540.0), (761.5, 341.25), (1159.9, 739.9)):
        back = ft.crop_to_full(ft.full_to_crop(point, centred), centred)
        assert back == pytest.approx(point)

    tiny = ft.zoom_geometry((50.0, 30.0), size=400, zoom=1, image_wh=(96, 64))
    assert tiny["offset"] == [0, 0] and tiny["source_size"] == [96, 64]

    overview = ft.overview_geometry((1920, 1080), 960)
    assert overview["scale"] == 0.5 and overview["size"] == [960, 540]
    assert ft.crop_to_full(ft.full_to_crop((1234.0, 567.0), overview), overview) == (1234.0, 567.0)
    assert ft.overview_geometry((96, 64), 960)["scale"] == 1.0
    assert ft.zoom_factor(15.0) == 2 and ft.zoom_factor(30.0) == 1 and ft.zoom_factor(None) == 1


# --------------------------------------------------------------------------------------------
# the workspace and the click pages


def test_prepare_writes_cells_crops_template_and_readme(tmp_path: Path) -> None:
    root = make_workspace(tmp_path, width=96, height=64)
    workspace = ft.load_workspace(root)
    assert workspace["counts"]["frames"] == 3 and len(workspace["frames"]) == 3
    assert workspace["counts"]["cells"] == len(workspace["cells"]) >= 6
    # On this tiny image the dipped tip sits inside the edge margin, so no low frame is
    # eligible and the shortfall is topped up; a held frame carries only the active pipette,
    # a rest frame every pipette with two views.
    assert workspace["selection"]["eligible_by_state"]["low"] == 0
    states = [f["state"] for f in workspace["frames"]]
    assert "low" not in states and "held" in states and "rest" in states
    for entry in workspace["frames"]:
        classes = [s["class"] for s in entry["slots"]]
        if entry["state"] == "held":
            assert classes == ["yellow_pipette"]
        else:
            assert classes == ["blue_pipette", "yellow_pipette"]
    assert workspace["selection"]["active_only_when_moving"] is True
    counts = workspace["counts"]["cells_by_state"]
    assert counts["rest"] > counts["held"]
    assert workspace["identity_keys"] == ft.IDENTITY_KEYS
    for cell in workspace["cells"]:
        for crop in cell["crops"].values():
            assert (root / crop["uri"]).is_file()
            assert crop["size"][0] > 0 and crop["offset"] == [0, 0]  # the image is tiny
        assert cell["image_size"] == [96, 64] and cell["state"] in ft.STATES
        assert cell["suggested_tip_px"] and cell["tip_rule"]
    template = json.loads((root / ft.TEMPLATE_NAME).read_text())
    assert template["schema"] == f"{ft.SCHEMA}/decisions"
    assert all(c["tip_px"] is None and c["hidden"] is False for c in template["cells"])
    assert all("suggested_tip_px" not in c for c in template["cells"])
    readme = (root / "README.md").read_text()
    assert "very end of the white tip cone" in readme and "press `h`" in readme
    assert "plunger colour matches the class" in readme and "20 minutes" in readme
    assert "--tailscale" in readme
    assert (root / "crops").is_dir() and not (root / ft.RECORD_NAME).exists()


def test_index_and_frame_pages_show_cells_markers_and_keys(served) -> None:
    root, app, base = served
    status, body, ctype = get(base, "/")
    page = body.decode()
    assert status == 200 and ctype.startswith("text/html")
    total = app.record.progress()["total"]
    assert f"0 / {total}" in page
    for raw in app.frames:
        assert f'href="/frame/{raw}"' in page
    state = json.loads(get(base, "/api/state")[1])
    assert [f["url"] for f in state["frames"]] == [f"/frame/{raw}" for raw in app.frames]
    assert state["progress"]["decided"] == 0 and state["undo_depth"] == 0

    raw = app.frames[0]
    status, body, _ = get(base, f"/frame/{raw}")
    page = body.decode()
    assert status == 200
    cells = app.page_cells[raw]
    for cell in cells:
        assert f'id="cell-{cell["view"]}-{cell["slot"]}"' in page
        assert f"/files/{cell['crops']['zoom']['uri']}" in page
    assert page.count('class="marker suggested"') == 2 * len(cells)
    assert 'class="marker set"' not in page
    assert 'data-identity="blue_pipette"' in page and "<kbd>4</kbd>" in page
    assert "<kbd>u</kbd> undo" in page and "<kbd>h</kbd> hidden" in page
    assert "case 'u': undo();" in page and "case '1': case '2': case '3': case '4':" in page
    data = json.loads(re.search(PAGE_DATA, page, re.S).group(1))
    assert data["raw_frame"] == raw and data["position"] == 1 and data["pages"] == 3
    assert data["next"] == f"/frame/{app.frames[1]}" and data["prev"] is None
    assert len(data["cells"]) == len(cells)
    # Files: crops are served, anything else is not.
    zoom = cells[0]["crops"]["zoom"]["uri"]
    status, content, ctype = get(base, f"/files/{zoom}")
    assert status == 200 and ctype == "image/jpeg" and content[:2] == b"\xff\xd8"
    assert get(base, "/files/../clip.json")[0] == 404
    assert get(base, "/files/cells.json")[0] == 200  # JSON under the workspace is fine
    assert get(base, "/frame/999999")[0] == 404
    assert get(base, "/frame/abc")[0] == 404
    # The record was created from the template: nothing decided, the suggestion not copied.
    record = json.loads((root / ft.RECORD_NAME).read_text())
    assert record["author"] == "tester" and TIMESTAMP.match(record["updated_at"])
    assert all(c["tip_px"] is None and not c["hidden"] for c in record["cells"])
    assert all("suggested_tip_px" not in c for c in record["cells"])


def test_click_hidden_undo_identity_and_the_atomic_record(served) -> None:
    root, app, base = served
    template_sha = sha256(root / ft.TEMPLATE_NAME)
    raw = app.frames[0]
    cell = app.page_cells[raw][0]
    address = {"raw_frame": raw, "view": cell["view"], "slot": cell["slot"]}
    zoom = cell["crops"]["zoom"]

    body = post_json(base, "/api/cell", {**address, "click": {"crop": "zoom", "x": 10, "y": 20}})
    expected = ft.crop_to_full((10, 20), zoom)
    assert body["cell"]["tip_px"] == [round(expected[0], 1), round(expected[1], 1)]
    assert body["cell"]["clicked_in"] == "zoom" and body["cell"]["hidden"] is False
    assert body["markers"]["zoom"] == pytest.approx([10, 20])
    full = ft.full_to_crop(expected, cell["crops"]["full"])
    assert body["markers"]["full"] == pytest.approx([round(full[0], 1), round(full[1], 1)])
    assert body["suggested"]["zoom"] is not None
    assert body["page"]["decided"] == 1 and body["progress"]["decided"] == 1
    assert body["progress"]["clicked"] == 1 and TIMESTAMP.match(body["updated_at"])
    record = json.loads((root / ft.RECORD_NAME).read_text())
    stored = next(
        c for c in record["cells"] if c["view"] == cell["view"] and c["slot"] == cell["slot"]
    )
    assert stored["tip_px"] == body["cell"]["tip_px"]
    assert not (root / "decisions.tmp").exists()
    assert sha256(root / ft.TEMPLATE_NAME) == template_sha

    # A click on the whole frame maps through the overview's scale (1.0 on this tiny image).
    body = post_json(base, "/api/cell", {**address, "click": {"crop": "full", "x": 30, "y": 40}})
    assert body["cell"]["tip_px"] == [30.0, 40.0] and body["cell"]["clicked_in"] == "full"

    # The page now shows a set marker on that cell.
    page = get(base, f"/frame/{raw}")[1].decode()
    section = page[page.index(f'id="cell-{cell["view"]}-{cell["slot"]}"') :]
    section = section[: section.index("</figure>")]
    assert 'class="marker set"' in section and "tip set" in section

    # h: hidden clears the tip; u: undo brings the click back.
    body = post_json(base, "/api/cell", {**address, "hidden": True})
    assert body["cell"]["hidden"] is True and body["cell"]["tip_px"] is None
    assert body["progress"]["decided"] == 1 and body["progress"]["hidden"] == 1
    body = post_json(base, "/api/undo", {})
    assert len(body["cells"]) == 1
    assert body["cells"][0]["cell"]["tip_px"] == [30.0, 40.0]
    assert body["cells"][0]["cell"]["hidden"] is False
    assert body["undo_depth"] == 2 and body["progress"]["hidden"] == 0
    record = json.loads((root / ft.RECORD_NAME).read_text())
    stored = next(
        c for c in record["cells"] if c["view"] == cell["view"] and c["slot"] == cell["slot"]
    )
    assert stored["tip_px"] == [30.0, 40.0] and stored["hidden"] is False

    # x: clear leaves the cell undecided.
    body = post_json(base, "/api/cell", {**address, "clear": True})
    assert body["cell"]["tip_px"] is None and body["cell"]["hidden"] is False
    assert body["progress"]["decided"] == 0

    # Identity keys name the plunger colour on every view of that pipette on the frame.
    slot_cells = [c for c in app.page_cells[raw] if c["slot"] == cell["slot"]]
    body = post_json(
        base, "/api/identity", {"raw_frame": raw, "slot": cell["slot"], "instance_identity": "2"}
    )
    assert body["instance_identity"] == "yellow_pipette"
    assert len(body["cells"]) == len(slot_cells) >= 2
    assert all(p["cell"]["instance_identity"] == "yellow_pipette" for p in body["cells"])
    page = get(base, f"/frame/{raw}")[1].decode()
    assert 'data-identity="yellow_pipette" class="active"' in page
    record = json.loads((root / ft.RECORD_NAME).read_text())
    named = [c for c in record["cells"] if c["instance_identity"] == "yellow_pipette"]
    assert len(named) == len(slot_cells)
    # One undo takes the whole identity group back.
    body = post_json(base, "/api/undo", {})
    assert len(body["cells"]) == len(slot_cells)
    assert all(p["cell"]["instance_identity"] is None for p in body["cells"])
    body = post_json(
        base,
        "/api/identity",
        {"raw_frame": raw, "slot": cell["slot"], "instance_identity": "8_channel_pipette"},
    )
    assert body["instance_identity"] == "8_channel_pipette"

    # Author on the index page.
    body = post_json(base, "/api/author", {"author": " N. Reviewer "})
    assert body["author"] == "N. Reviewer"
    assert json.loads((root / ft.RECORD_NAME).read_text())["author"] == "N. Reviewer"


def test_invalid_changes_are_refused_and_nothing_is_written(served) -> None:
    root, app, base = served
    raw = app.frames[0]
    cell = app.page_cells[raw][0]
    address = {"raw_frame": raw, "view": cell["view"], "slot": cell["slot"]}
    before = sha256(root / ft.RECORD_NAME)
    code, error = post_error(
        base, "/api/cell", {**address, "click": {"crop": "zoom", "x": -500, "y": 0}}
    )
    assert code == 400 and "outside" in error
    code, error = post_error(
        base, "/api/cell", {**address, "click": {"crop": "nope", "x": 1, "y": 1}}
    )
    assert code == 400 and "crop" in error
    code, error = post_error(base, "/api/cell", {**address, "tip_px": [1e9, 0]})
    assert code == 400
    code, error = post_error(base, "/api/cell", {**address, "hidden": "yes"})
    assert code == 400 and "hidden" in error
    code, error = post_error(base, "/api/cell", {**address, "suggested_tip_px": [1, 1]})
    assert code == 400 and "unknown cell fields" in error
    code, error = post_error(base, "/api/cell", address)
    assert code == 400 and "nothing to change" in error
    code, error = post_error(base, "/api/cell", {"raw_frame": raw, "view": cell["view"]})
    assert code == 400 and "slot" in error
    code, error = post_error(base, "/api/cell", {**address, "raw_frame": 12345, "hidden": True})
    assert code == 404 and "no cell" in error
    code, error = post_error(
        base,
        "/api/identity",
        {"raw_frame": raw, "slot": cell["slot"], "instance_identity": "purple"},
    )
    assert code == 400 and "instance_identity" in error
    code, error = post_error(base, "/api/undo", {})
    assert code == 400 and "nothing to undo" in error
    assert sha256(root / ft.RECORD_NAME) == before
    # Failed validations leave no history behind either.
    assert app.record.history == []


def test_existing_record_is_reused_and_completed(tmp_path: Path) -> None:
    root = make_workspace(tmp_path, width=96, height=64)
    template = json.loads((root / ft.TEMPLATE_NAME).read_text())
    partial = dict(template)
    partial["cells"] = [dict(template["cells"][0], hidden=True, note="kept")]
    partial["author"] = "earlier"
    fs_common.write_json(root / ft.RECORD_NAME, partial)
    app = ft.TipsApp.open(workspace_dir=root, author=None)
    record = json.loads((root / ft.RECORD_NAME).read_text())
    assert record["author"] == "earlier"
    assert len(record["cells"]) == len(template["cells"])
    assert record["cells"][0]["hidden"] is True and record["cells"][0]["note"] == "kept"
    assert app.record.progress()["decided"] == 1
    elsewhere = tmp_path / "scratch" / "mine.json"
    ft.TipsApp.open(workspace_dir=root, record_path=elsewhere)
    assert json.loads(elsewhere.read_text())["schema"] == f"{ft.SCHEMA}/decisions"
    assert ft.load_record(None)["cells"] == []
    with pytest.raises(ValueError, match="decisions file"):
        ft.load_record(root / ft.CELLS_NAME)


# --------------------------------------------------------------------------------------------
# triangulation on the real rig and the tip-error scorer


def fixture_cameras() -> tuple[dict[str, Camera], int]:
    fixtures = load_preflight_fixtures()
    cams = fixtures.fixed_cameras()
    frame = next(f for f in sorted(fixtures.fpv_poses) if fixtures.fpv_poses[f]["valid"])
    fpv = fixtures.fpv_camera(frame)
    assert fpv is not None
    cams["fpv"] = fpv
    return cams, frame


def visible_views(cams: dict[str, Camera], point: np.ndarray) -> dict[str, tuple[float, float]]:
    out = {}
    for view, cam in cams.items():
        px = cam.project(point)[0]
        if 0 <= px[0] < cam.size[0] and 0 <= px[1] < cam.size[1]:
            out[view] = (float(px[0]), float(px[1]))
    return out


def _cell(raw: int, view: str, slot: int, cls: str, size: tuple[int, int]) -> dict[str, Any]:
    return {
        "raw_frame": raw,
        "proxy_frame": raw,
        "view": view,
        "slot": slot,
        "class": cls,
        "label": f"{cls}#0",
        "state": "held",
        "image_size": list(size),
        "mask_area_px": 1000,
        "suggested_tip_px": [1.0, 1.0],
    }


def test_clicks_triangulate_to_the_known_tip_on_the_real_rig() -> None:
    cams, frame = fixture_cameras()
    fixtures = load_preflight_fixtures()
    plate = next(s for s in fixtures.rig_reference["static"] if s["class"] == "centrifuge")
    tip = np.array(plate["point_cm"], dtype=np.float64) + np.array([0.0, 0.0, -2.0])
    pixels = visible_views(cams, tip)
    assert len(pixels) >= 3
    views = sorted(pixels)[:3]
    cells = [_cell(frame, v, 1, "yellow_pipette", cams[v].size) for v in views]
    cells += [_cell(frame, v, 2, "red_pipette", cams[v].size) for v in views[:2]]
    cells += [_cell(frame, v, 0, "blue_pipette", cams[v].size) for v in views[:2]]
    workspace = {"cells": cells}
    record = {
        "cells": [
            *[
                {
                    "raw_frame": frame,
                    "view": v,
                    "slot": 1,
                    "tip_px": list(pixels[v]),
                    "hidden": False,
                }
                for v in views
            ],
            # red: one click only -> dropped; blue: hidden everywhere but named -> identity only
            {"raw_frame": frame, "view": views[0], "slot": 2, "tip_px": list(pixels[views[0]])},
            {
                "raw_frame": frame,
                "view": views[0],
                "slot": 0,
                "hidden": True,
                "instance_identity": "blue_pipette",
            },
            {
                "raw_frame": frame,
                "view": views[1],
                "slot": 0,
                "hidden": True,
                "instance_identity": "blue_pipette",
            },
        ]
    }
    anchors, dropped = ft.triangulate_clicks(workspace, record, lambda _f: cams)
    with_tip = [a for a in anchors if a.tip_cm is not None]
    assert len(with_tip) == 1 and with_tip[0].slot == 1
    anchor = with_tip[0]
    assert np.linalg.norm(anchor.tip_cm - tip) < 0.05
    assert max(anchor.residual_px.values()) < 0.05 and anchor.views == sorted(
        views, key=ft.view_rank
    )
    assert anchor.dropped_view is None
    assert [d["slot"] for d in dropped] == [0, 2]
    assert "1 clicked view" in dropped[1]["reason"] and dropped[0]["reason"] == "no click"
    identity_only = [a for a in anchors if a.tip_cm is None]
    assert len(identity_only) == 1 and identity_only[0].identity == "blue_pipette"
    assert identity_only[0].hidden_views == views[:2]
    row = anchor.as_dict()
    assert row["tip_height_cm"] == pytest.approx(-tip[2], abs=0.05)

    # A mis-click 80 px off in one of three views is dropped by the residual gate.
    bad = dict(record)
    bad["cells"] = [dict(c) for c in record["cells"]]
    wrong = next(c for c in bad["cells"] if c["slot"] == 1 and c["view"] == views[0])
    wrong["tip_px"] = [wrong["tip_px"][0] + 80.0, wrong["tip_px"][1]]
    anchors, _ = ft.triangulate_clicks(workspace, bad, lambda _f: cams)
    gated = next(a for a in anchors if a.tip_cm is not None)
    assert gated.dropped_view == views[0]
    assert np.linalg.norm(gated.tip_cm - tip) < 0.1 and set(gated.residual_px) == set(views[1:])

    # Without a camera for a clicked view (fpv on an invalid-pose frame) the view is skipped.
    fixed_only = {v: c for v, c in cams.items() if v != "fpv"}
    anchors, dropped = ft.triangulate_clicks(workspace, record, lambda _f: fixed_only)
    if "fpv" in views:
        assert all("fpv" not in a.views for a in anchors)


def _write_tracks(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows))
    return path


def test_tip_error_against_line_tracks_and_the_point_baseline(tmp_path: Path) -> None:
    frame = 1800
    tip = np.array([20.0, 10.0, -3.0])
    direction = np.array([0.0, 0.0, -1.0])
    anchor = ft.TipAnchor(
        raw_frame=frame,
        slot=1,
        cls="yellow_pipette",
        state="low",
        identity="yellow_pipette",
        labels={"T4": "yellow_pipette#0", "T5": "yellow_pipette#0", "fpv": "yellow_pipette#0"},
        pixels={"T4": (1.0, 1.0), "T5": (2.0, 2.0)},
        tip_cm=tip,
        residual_px={"T4": 1.0, "T5": 2.0},
    )
    other = ft.TipAnchor(
        raw_frame=frame,
        slot=0,
        cls="blue_pipette",
        state="rest",
        identity="blue_pipette",
        labels={"T4": "blue_pipette#0"},
        pixels={"T4": (5.0, 5.0), "T5": (6.0, 6.0)},
        tip_cm=tip + np.array([30.0, 0.0, 0.0]),
        residual_px={"T4": 1.0, "T5": 1.0},
    )
    named_only = ft.TipAnchor(
        raw_frame=frame + 1,
        slot=2,
        cls="red_pipette",
        state="held",
        identity="red_pipette",
        labels={"T4": "red_pipette#0"},
        pixels={},
    )
    base = {
        "schema_version": "1.0",
        "frame_index": frame,
        "uncertainty_cm": 1.0,
        "state": "observed",
        "confidence": 0.9,
        "abstain": False,
    }
    (tmp_path / "lines").mkdir()
    resolved = _write_tracks(
        tmp_path / "lines" / "tracks.jsonl",
        [
            {
                **base,
                "track_id": "pipette-001",
                "object_class": "pipette",
                "position_cm": (tip + np.array([1.0, 0.0, 0.0]) + 11.5 * direction).tolist(),
                "endpoints_cm": [
                    (tip + np.array([1.0, 0.0, 0.0])).tolist(),
                    (tip + np.array([1.0, 0.0, 0.0]) + 23.0 * direction).tolist(),
                ],
                "direction": direction.tolist(),
                "tip_resolved": True,
                "observed_class": "yellow_pipette",
                "colour_identity": "yellow_pipette",
                "support_slots": {"T4": "yellow_pipette#0", "T5": "yellow_pipette#0"},
            },
            {
                **base,
                "track_id": "pipette-002",
                "object_class": "pipette",
                "position_cm": (tip + np.array([30.0, 0.0, 0.0]) + 11.5 * direction).tolist(),
                "endpoints_cm": [
                    (tip + np.array([30.0, 0.0, 0.0]) + 23.0 * direction).tolist(),
                    (tip + np.array([30.5, 0.0, 0.0])).tolist(),
                ],
                "direction": (-direction).tolist(),
                "tip_resolved": False,
                "observed_class": "blue_pipette",
                "colour_identity": None,
                "support_slots": {},
            },
            {
                **base,
                "frame_index": frame + 5,
                "track_id": "50ml_tube-003",
                "object_class": "50ml_tube",
                "position_cm": [0, 0, 0],
            },
        ],
    )
    points = _write_tracks(
        tmp_path / "points.jsonl",
        [
            {
                **base,
                "track_id": "yellow_pipette-007",
                "object_class": "yellow_pipette",
                "position_cm": (tip + 11.5 * direction).tolist(),
                "direction": direction.tolist(),
                "support_slots": {"T5": "yellow_pipette#0"},
            },
            {
                **base,
                "track_id": "blue_pipette-008",
                "object_class": "blue_pipette",
                "position_cm": (tip + np.array([30.0, 0.0, 0.0]) + 11.5 * direction).tolist(),
                "support_slots": {"T4": "blue_pipette#0"},
            },
        ],
    )
    anchors = [anchor, other, named_only]
    lines = ft.score_tracks_file("lines", resolved, anchors, repository_root=tmp_path)
    assert lines["anchors"] == 2 and lines["matched"] == 2 and lines["unmatched"] == 0
    yellow, blue, red = lines["cells"]
    assert yellow["track_id"] == "pipette-001" and yellow["match"] == "support_slots (2 views)"
    assert yellow["tip_error_cm"] == pytest.approx(1.0) and yellow["measure"] == "tip_endpoint"
    assert yellow["other_end_cm"] == pytest.approx(math.hypot(1.0, 23.0), abs=1e-3)
    assert blue["track_id"] == "pipette-002" and blue["match"].startswith("nearest endpoint")
    assert blue["tip_error_cm"] == pytest.approx(0.5)
    assert blue["measure"] == "nearer_endpoint_unresolved" and blue["tip_resolved"] is False
    assert red["track_id"] is None and red["tip_cm"] is None
    assert lines["tip_error"]["median_cm"] == pytest.approx(0.75)
    assert lines["tip_error"]["p90_cm"] == pytest.approx(0.95)
    assert lines["by_state"]["low"]["n"] == 1 and lines["by_state"]["rest"]["n"] == 1
    assert lines["by_state"]["held"]["n"] == 0 and lines["by_state"]["held"]["median_cm"] is None
    assert lines["median_under_threshold"] is True and lines["tips_unresolved"] == 1
    assert lines["measures"] == {"tip_endpoint": 1, "nearer_endpoint_unresolved": 1}
    identity = lines["identity"]
    assert identity["pipette_idf1"]["cells"] == 3 and identity["pipette_idf1"]["idtp"] == 2
    assert identity["pipette_idf1"]["cells_without_track"] == 1
    assert identity["agreement"]["colour_identity"] == {"agree": 1, "absent": 1}
    assert identity["agreement"]["observed_class"] == {"agree": 2}
    assert identity["agreement"]["track_class"] == {"absent": 2}

    point = ft.score_tracks_file("points", points, anchors, repository_root=tmp_path)
    yellow, blue, red = point["cells"]
    assert yellow["track_id"] == "yellow_pipette-007" and yellow["measure"] == "point_to_midpoint"
    assert yellow["tip_error_cm"] == pytest.approx(11.5) and yellow[
        "axis_distance_cm"
    ] == pytest.approx(0.0)
    assert blue["track_id"] == "blue_pipette-008" and blue["axis_distance_cm"] is None
    assert blue["match"] == "support_slots (1 views)"
    assert blue["tip_error_cm"] == pytest.approx(11.5) and blue["measure"] == "point_to_midpoint"
    assert point["median_under_threshold"] is False and point["point_tracks"] == 2
    # A point track's class is the detector's, so it can agree with the named colour.
    assert point["identity"]["agreement"]["track_class"] == {"agree": 2}
    assert point["identity"]["agreement"]["colour_identity"] == {"absent": 2}
    assert point["identity"]["pipette_idf1"]["idf1"] == pytest.approx(round(2 * 2 / (3 + 2), 4))

    # A one-view support match far from the tip yields to the track that is actually there;
    # with two agreeing views it holds, and with no nearer track it is kept and flagged.
    far_support = {
        **base,
        "track_id": "pipette-099",
        "object_class": "pipette",
        "position_cm": (tip + np.array([40.0, 0.0, 0.0])).tolist(),
        "endpoints_cm": [
            (tip + np.array([40.0, 0.0, 0.0])).tolist(),
            (tip + np.array([40.0, 0.0, -23.0])).tolist(),
        ],
        "tip_resolved": True,
        "support_slots": {"T4": "yellow_pipette#0"},
    }
    near_track = {
        **base,
        "track_id": "pipette-001",
        "object_class": "pipette",
        "position_cm": tip.tolist(),
        "endpoints_cm": [tip.tolist(), (tip + 23 * direction).tolist()],
        "tip_resolved": True,
        "support_slots": {},
    }
    row, how = ft.match_track(anchor, [far_support, near_track])
    assert row["track_id"] == "pipette-001" and how.startswith("nearest endpoint")
    assert "pipette-099 was 40.0 cm away" in how
    row, how = ft.match_track(anchor, [far_support])
    assert row["track_id"] == "pipette-099" and how == "support_slots (1 views, 40.0 cm away)"
    two_views = {
        **far_support,
        "support_slots": {"T4": "yellow_pipette#0", "T5": "yellow_pipette#0"},
    }
    row, how = ft.match_track(anchor, [two_views, near_track])
    assert row["track_id"] == "pipette-099" and how == "support_slots (2 views)"
    row, how = ft.match_track(named_only, [far_support, near_track])
    assert row is None and how == "no support match and no 3D tip"

    assert ft.parse_tracks(
        ["lines=a/tracks.jsonl", "runs/x/tracks-ext/tracks.jsonl", "b.jsonl"]
    ) == {
        "lines": Path("a/tracks.jsonl"),
        "tracks-ext": Path("runs/x/tracks-ext/tracks.jsonl"),
        "b": Path("b.jsonl"),
    }
    with pytest.raises(ValueError, match="two tracks files"):
        ft.parse_tracks(["a/tracks.jsonl", "b/a/tracks.jsonl"])


def test_score_and_export_on_a_served_workspace(tmp_path: Path) -> None:
    cams, _ = fixture_cameras()
    fixtures = load_preflight_fixtures()
    plate = next(s for s in fixtures.rig_reference["static"] if s["class"] == "centrifuge")
    tip = np.array(plate["point_cm"], dtype=np.float64) + np.array([0.0, 0.0, -2.0])
    pixels = visible_views(cams, tip)
    assert {"T4", "T5"} <= set(pixels)
    root = make_workspace(tmp_path, width=1920, height=1080)
    app = ft.TipsApp.open(workspace_dir=root, author="tester")
    workspace = ft.load_workspace(root)
    # A rest frame carries every pipette; a held or low one only the active pipette.
    raw = next(f["raw_frame"] for f in workspace["frames"] if f["state"] == "rest")
    yellow = [c for c in app.page_cells[raw] if c["class"] == "yellow_pipette"]
    assert len(yellow) >= 2
    clicked = 0
    clickable = {}
    for cell in yellow:
        px = pixels.get(cell["view"])
        if px is None or not ft.inside_image(px, cell["image_size"], 0.0):
            continue
        app.record.update(ft.cell_key(cell), {"tip_px": [px[0], px[1]]})
        clickable[cell["view"]] = px
        clicked += 1
    assert clicked >= 2
    app.record.set_identity(raw, 1, "2")
    blue = [c for c in app.page_cells[raw] if c["class"] == "blue_pipette"]
    for cell in blue:
        app.record.update(ft.cell_key(cell), {"hidden": True})
    app.record.set_identity(raw, 0, "1")

    direction = np.array([0.0, 0.0, -1.0])
    tracks = _write_tracks(
        tmp_path / "tracks.jsonl",
        [
            {
                "schema_version": "1.0",
                "frame_index": raw,
                "track_id": "pipette-001",
                "object_class": "pipette",
                "position_cm": (tip + 11.5 * direction).tolist(),
                "uncertainty_cm": 1.0,
                "state": "observed",
                "confidence": 0.9,
                "abstain": False,
                "endpoints_cm": [(tip + [0.5, 0, 0]).tolist(), (tip + 23 * direction).tolist()],
                "direction": direction.tolist(),
                "tip_resolved": True,
                "observed_class": "yellow_pipette",
                "colour_identity": "yellow_pipette",
                "support_slots": {c["view"]: c["label"] for c in yellow},
            }
        ],
    )
    report = ft.score_workspace(
        workspace_dir=root,
        record_path=root / ft.RECORD_NAME,
        tracks={"lines": tracks, "missing": tmp_path / "nope.jsonl"},
        cameras_at=lambda _frame: cams,
        repository_root=tmp_path,
    )
    assert report["anchors"]["count"] == 1 and report["anchors"]["by_class"] == {
        "yellow_pipette": 1
    }
    # Clicks are stored to 0.1 px, so the residual is that rounding, not the rig's.
    assert report["anchors"]["reprojection_residual_px"]["max"] < 0.3
    assert report["record_summary"]["clicked"] == clicked
    assert report["record_summary"]["hidden"] == len(blue)
    assert report["record"]["sha256"] == sha256(root / ft.RECORD_NAME)
    assert [a["tracks"] for a in report["arms"]] == ["lines"]
    assert report["missing_tracks"][0]["tracks"] == "missing"
    arm = report["arms"][0]
    assert arm["tip_error"]["median_cm"] == pytest.approx(0.5, abs=0.05)
    assert arm["median_under_threshold"] is True
    assert arm["identity"]["pipette_idf1"]["cells"] == 2  # yellow (tracked) and blue (named)
    markdown = ft.scoreboard_markdown(report)
    assert "# Tip scoreboard: 1 anchors" in markdown
    assert "| lines | 1 / 1 | 0.5" in markdown and "median < 2 cm" in markdown
    assert "Tracks file not found" in markdown and "## lines: per pipette-frame" in markdown
    assert "point_to_midpoint" in markdown

    # An empty record scores nothing and says so.
    empty = ft.score_workspace(
        workspace_dir=root,
        record_path=None,
        tracks={"lines": tracks},
        cameras_at=lambda _frame: cams,
        repository_root=tmp_path,
    )
    assert empty["anchors"]["count"] == 0 and empty["record"] is None
    assert "No anchor yet" in ft.scoreboard_markdown(empty)

    # Export: no pixels, the clicks in full-frame pixels, the decisions' SHA-256.
    output = tmp_path / "docs" / "qa" / "test-tip-clicks.human-record.json"
    doc = ft.export_record(
        workspace_dir=root,
        record_path=root / ft.RECORD_NAME,
        output=output,
        repository_root=tmp_path,
    )
    assert output.is_file() and json.loads(output.read_text()) == doc
    assert doc["schema"] == f"{ft.SCHEMA}/human-record" and doc["anchor_kind"] == ft.ANCHOR_KIND
    assert doc["author"] == "tester" and doc["reviewed_at"] is None
    assert doc["decisions"]["sha256"] == sha256(root / ft.RECORD_NAME)
    assert doc["workspace"] == "tips" and doc["cells_json"]["uri"] == "tips/cells.json"
    assert doc["counts"]["cells"] == len(workspace["cells"])
    assert doc["counts"]["clicked"] == clicked and doc["counts"]["hidden"] == len(blue)
    assert doc["counts"]["with_identity"] == len(yellow) + len(blue)
    clicks = {(c["view"], c["slot"]): c for c in doc["clicks"] if c["raw_frame"] == raw}
    for cell in yellow:
        entry = clicks[(cell["view"], cell["slot"])]
        px = clickable.get(cell["view"])
        if px is not None:
            assert entry["tip_px"] == [round(px[0], 1), round(px[1], 1)]
        assert entry["instance_identity"] == "yellow_pipette" and entry["hidden"] is False
    text = output.read_text()
    assert "suggested_tip_px" not in text and "crops/" not in text and ".jpg" not in text
    with pytest.raises(FileNotFoundError):
        ft.export_record(
            workspace_dir=root,
            record_path=tmp_path / "none.json",
            output=output,
            repository_root=tmp_path,
        )


# --------------------------------------------------------------------------------------------
# CLI


def test_cli_registration_defaults_and_bind_rules() -> None:
    pyproject = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text())
    assert pyproject["project"]["scripts"]["battle-finebio-tips"] == "battle.finebio_tips:main"
    parser = ft.build_parser()
    prep = parser.parse_args(
        ["prepare", "--observations", "obs", "--clip-config", "clip.json", "--output", "out"]
    )
    assert prep.frames == 30 and prep.spacing == 20 and prep.tracks is None
    assert prep.pipettes == ft.DEFAULT_PIPETTES_CONFIG
    serve = parser.parse_args(["serve", "--workspace", "ws"])
    assert serve.port == 8767 and serve.bind == "127.0.0.1" and serve.tailscale is False
    with pytest.raises(SystemExit):
        parser.parse_args(["serve", "--workspace", "ws", "--bind", "10.0.0.1", "--tailscale"])
    score = parser.parse_args(
        [
            "score",
            "--workspace",
            "ws",
            "--tracks",
            "a.jsonl",
            "--tracks",
            "b=b.jsonl",
            "--clip-config",
            "c",
        ]
    )
    assert score.tracks == ["a.jsonl", "b=b.jsonl"] and score.output is None
    with pytest.raises(SystemExit):
        parser.parse_args(["score", "--workspace", "ws", "--clip-config", "c"])
    export = parser.parse_args(["export", "--workspace", "ws", "--record", "r", "--output", "o"])
    assert export.record == Path("r")
    assert ft.validate_identity("3") == "red_pipette" and ft.validate_identity(None) is None
    assert ft.validate_identity("8_channel_pipette") == "8_channel_pipette"
    with pytest.raises(ValueError):
        ft.validate_identity("green")
    assert ft.pipette_length_cm(None) == ft.DEFAULT_LENGTH_CM
    committed = REPOSITORY_ROOT / "configs" / "finebio" / "pipettes.json"
    if committed.is_file():
        assert 15.0 < ft.pipette_length_cm(committed) < 30.0
