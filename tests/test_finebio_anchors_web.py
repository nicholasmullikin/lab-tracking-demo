"""`battle-finebio-anchors-web` on a two-frame-view, two-slot synthetic workspace."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from conftest import SERVER_POLL_INTERVAL_SECONDS
from PIL import Image

from battle import finebio_anchors as fa
from battle import fs_common
from battle.finebio_anchors_web import (
    App,
    Scorer,
    build_parser,
    make_handler,
    page_order,
    resolve_bind,
)

TIMESTAMP = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
PAGE_DATA = r'<script id="page-data" type="application/json">(.*?)</script>'


def _candidate(view: str, raw: int, slot: int, index: int, kind: str, dup: int | None = None):
    return {
        "index": index,
        "kind": kind,
        "mask_uri": fa.candidate_file(view, raw, slot, index),
        "prompt_box": [0.0, 0.0, 4.0, 4.0],
        "prompt_point": None,
        "decoder_iou_pred": 0.9 - index / 10,
        "mask_area_px": 1,
        "mask_bbox_px": [0, 0, 1, 1],
        "mask_bbox_iou_vs_box": 0.8,
        "duplicate_of": dup,
    }


def _cell(
    raw: int,
    view: str,
    slot: int,
    label: str,
    object_class: str,
    *,
    origin: str,
    box: list[float] | None,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "raw_frame": raw,
        "proxy_frame": raw - 600,
        "view": view,
        "slot": slot,
        "label": label,
        "class": object_class,
        "role": "container" if object_class == "centrifuge" else "object",
        "origin": origin,
        "in_cycle_neighbourhood": False,
        "reference_box": box,
        "arm_b_row": box is not None,
        "arm_b_detector_score": 0.77 if box is not None else None,
        "candidates": candidates,
        "sheet": f"sheets/f{raw:06d}_{view}.jpg",
    }


def make_workspace(tmp_path: Path) -> Path:
    """Two frame-views (916 fpv, six-view annotated; 637 T4, random) x two slots: a cell with
    c0 and a duplicate c1, a cell without candidates, a cell with c0 only, a cell with c0 and
    c2 (c1 missing). 1x1 white PNG masks; a sheet for 916 fpv only."""
    root = tmp_path / "anchors"
    cells = [
        _cell(
            916,
            "fpv",
            0,
            "centrifuge#0",
            "centrifuge",
            origin="six_view_annotated",
            box=[0.0, 0.0, 4.0, 4.0],
            candidates=[
                _candidate("fpv", 916, 0, 0, "arm_b_tight"),
                _candidate("fpv", 916, 0, 1, "margin", dup=0),
            ],
        ),
        _cell(
            916,
            "fpv",
            1,
            "50ml_tube#0",
            "50ml_tube",
            origin="six_view_annotated",
            box=None,
            candidates=[],
        ),
        _cell(
            637,
            "T4",
            0,
            "centrifuge#0",
            "centrifuge",
            origin="random",
            box=[1.0, 1.0, 3.0, 3.0],
            candidates=[_candidate("T4", 637, 0, 0, "arm_b_tight")],
        ),
        _cell(
            637,
            "T4",
            1,
            "50ml_tube#0",
            "50ml_tube",
            origin="random",
            box=[1.0, 1.0, 3.0, 3.0],
            candidates=[
                _candidate("T4", 637, 1, 0, "arm_b_tight"),
                _candidate("T4", 637, 1, 2, "box_point"),
            ],
        ),
    ]
    for cell in cells:
        for candidate in cell["candidates"]:
            path = root / candidate["mask_uri"]
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("L", (1, 1), 255).save(path)
    sheets = root / "sheets"
    sheets.mkdir(parents=True)
    Image.new("RGB", (1280, 50 + 2 * 318), (12, 12, 12)).save(sheets / "f000916_fpv.jpg")
    for name in ("f000916_fpv_overview.jpg", "f000637_T4_overview.jpg"):
        Image.new("RGB", (64, 48), (30, 30, 30)).save(sheets / name)
    (root / "worker.log").write_text("not served\n")
    workspace = {
        "schema": f"{fa.SCHEMA}/workspace",
        "config": {"uri": "configs/qa/test_anchors.json", "sha256": "0" * 64},
        "arm_b": "runs/arms-test/b-box-decode-arm",
        "worker_runs": {},
        "trial": "TEST",
        "frame_index_offset": 600,
        "decode": None,
        "cells": cells,
        "sheets": {"f000916_fpv": "sheets/f000916_fpv.jpg"},
        "counts": {"cells": len(cells)},
        "claim_boundary": fa.CLAIM_BOUNDARY,
        "licence": fa.LICENCE_NOTE,
    }
    fs_common.write_json(root / "workspace.json", workspace)
    template = fa.decisions_template(
        {"config_id": "test-anchors", "trial": "TEST"}, cells, workspace=root
    )
    fs_common.write_json(root / "decisions.template.json", template)
    return root


def serve_app(app: App) -> tuple[ThreadingHTTPServer, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    threading.Thread(
        target=server.serve_forever, args=(SERVER_POLL_INTERVAL_SECONDS,), daemon=True
    ).start()
    return server, f"http://127.0.0.1:{server.server_port}"


@pytest.fixture
def served(tmp_path: Path):
    root = make_workspace(tmp_path)
    app = App.open(workspace_dir=root, repository_root=tmp_path, author="tester", arms={})
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


def section(page: str, row_id: str) -> str:
    start = page.index(f'id="{row_id}"')
    end = page.index("</section>", start)
    return page[start:end]


# --------------------------------------------------------------------------------------------


def test_page_order_follows_the_brief() -> None:
    cells = [
        {"raw_frame": 4122, "view": "fpv", "origin": "random"},
        {"raw_frame": 1294, "view": "T4", "origin": "disagreement", "in_cycle_neighbourhood": 1},
        {"raw_frame": 916, "view": "T4", "origin": "six_view_annotated"},
        {"raw_frame": 916, "view": "fpv", "origin": "six_view_annotated"},
        {"raw_frame": 916, "view": "T1", "origin": "six_view_annotated"},
        {"raw_frame": 637, "view": "fpv", "origin": "random"},
        {"raw_frame": 1120, "view": "fpv", "origin": "disagreement", "in_cycle_neighbourhood": 1},
        {"raw_frame": 1521, "view": "fpv", "origin": "disagreement"},
    ]
    assert page_order(cells) == [
        (916, "fpv"),
        (916, "T1"),
        (916, "T4"),
        (1120, "fpv"),
        (1294, "T4"),
        (637, "fpv"),
        (1521, "fpv"),
        (4122, "fpv"),
    ]


def test_index_lists_pages_in_recommended_order_with_progress(served) -> None:
    root, app, base = served
    status, body, ctype = get(base, "/")
    page = body.decode()
    assert status == 200 and ctype.startswith("text/html")
    assert page.index('href="/frame/916/fpv"') < page.index('href="/frame/637/T4"')
    assert "0 / 4" in page
    assert 'id="author" value="tester"' in page
    assert "six-view frame" in page and "random" in page
    state = json.loads(get(base, "/api/state")[1])
    assert [p["url"] for p in state["pages"]] == ["/frame/916/fpv", "/frame/637/T4"]
    assert state["progress"] == {"labelled": 0, "total": 4}
    assert state["score"] == {"available": False, "running": False, "last": None}
    # The record was created from the template at start, nothing labelled, template untouched.
    record = json.loads((root / "decisions.json").read_text())
    assert record["schema"] == f"{fa.SCHEMA}/decisions"
    assert record["author"] == "tester"
    assert TIMESTAMP.match(record["updated_at"])
    assert all(cell["decision"] is None for cell in record["cells"])
    template = json.loads((root / "decisions.template.json").read_text())
    assert template["author"] is None and "updated_at" not in template


def test_frame_page_renders_rows_candidates_and_serves_the_images(served, tmp_path) -> None:
    root, app, base = served
    status, body, _ = get(base, "/frame/916/fpv")
    page = body.decode()
    assert status == 200
    assert 'id="row-0"' in page and 'id="row-1"' in page
    assert "centrifuge#0" in page and "50ml_tube#0" in page
    row0 = section(page, "row-0")
    assert 'data-candidates="[0, 1]"' in row0 and 'data-has-box="1"' in row0
    # Tiles are cut from the workspace's own sheet: column 0 is the reference crop, column
    # 1 + k the candidate k; the duplicate is greyed and captioned with its original.
    assert "background-position:-0px -50px" in row0
    assert "background-position:-256px -50px" in row0
    assert 'class="tile cand dup" data-index="1"' in row0 and "= c0" in row0
    assert "/files/candidates/fpv/f000916_s00_c0.png" in row0
    assert 'href="/files/sheets/f000916_fpv.jpg"' in page
    assert 'src="/files/sheets/f000916_fpv_overview.jpg"' in page
    for uri in (
        "/files/candidates/fpv/f000916_s00_c0.png",
        "/files/candidates/fpv/f000916_s00_c1.png",
        "/files/sheets/f000916_fpv_overview.jpg",
        "/files/sheets/f000916_fpv.jpg",
    ):
        status, content, ctype = get(base, uri)
        assert status == 200, uri
        assert ctype.startswith("image/") and content
    # Without a sheet the candidate PNGs themselves are the tiles; a missing index is absent.
    status, body, _ = get(base, "/frame/637/T4")
    page = body.decode()
    row1 = section(page, "row-1")
    assert 'data-candidates="[0, 2]"' in row1
    assert '<img src="/files/candidates/T4/f000637_s01_c2.png"' in row1
    assert "f000637_s01_c1.png" not in row1
    assert "background-position" not in page
    # Navigation follows the recommended order; the last page's "next" is the index.
    data = json.loads(re.search(PAGE_DATA, page, re.S).group(1))
    assert data["prev"] == "/frame/916/fpv" and data["next"] == "/"
    assert data["position"] == 2 and data["pages"] == 2
    # Only images, markdown and JSON under the workspace are served; nothing outside it.
    (tmp_path / "secret.json").write_text("{}")
    assert get(base, "/files/../secret.json")[0] == 404
    assert get(base, "/files/worker.log")[0] == 404
    assert get(base, "/files/candidates/fpv/missing.png")[0] == 404
    assert get(base, "/frame/999/fpv")[0] == 404
    assert get(base, "/frame/916/T9")[0] == 404


def test_post_decision_writes_the_record_atomically_and_leaves_the_template(served) -> None:
    root, app, base = served
    template = root / "decisions.template.json"
    before = sha256(template)
    record_path = root / "decisions.json"
    created_at = json.loads(record_path.read_text())["updated_at"]

    body = post_json(base, "/api/cell", {"raw_frame": 916, "view": "fpv", "slot": 0, "decision": 1})
    assert body["cell"]["decision"] == 1
    assert body["page"] == {"labelled": 1, "total": 2}
    assert body["progress"] == {"labelled": 1, "total": 4}
    assert TIMESTAMP.match(body["updated_at"]) and body["updated_at"] >= created_at

    record = json.loads(record_path.read_text())
    cell = next(c for c in record["cells"] if c["view"] == "fpv" and c["slot"] == 0)
    assert cell["decision"] == 1
    template_cell = next(
        c
        for c in json.loads(template.read_text())["cells"]
        if c["view"] == "fpv" and c["slot"] == 0
    )
    assert list(cell) == list(template_cell)
    assert record["author"] == "tester" and record["updated_at"] == body["updated_at"]
    assert set(record) >= set(json.loads(template.read_text()))
    assert not (root / "decisions.tmp").exists()
    assert sha256(template) == before

    # Digits arriving as strings and the words are accepted the same way; the page shows them.
    post_json(base, "/api/cell", {"raw_frame": 637, "view": "T4", "slot": 1, "decision": "2"})
    post_json(base, "/api/cell", {"raw_frame": 637, "view": "T4", "slot": 0, "decision": "box"})
    page = get(base, "/frame/637/T4")[1].decode()
    assert 'class="tile cand chosen" data-index="2"' in section(page, "row-1")
    assert 'data-decision="box" class="active"' in section(page, "row-0")
    assert "3 / 4" in get(base, "/")[1].decode()


def test_identity_note_and_clear(served) -> None:
    root, app, base = served
    address = {"raw_frame": 916, "view": "fpv", "slot": 0}
    body = post_json(base, "/api/cell", {**address, "instance_identity": "  tube_A "})
    assert body["cell"]["instance_identity"] == "tube_A"
    assert body["identities"] == ["centrifuge", "tube_A"]
    page = get(base, "/frame/637/T4")[1].decode()
    assert '<option value="tube_A"></option>' in page and '<option value="centrifuge">' in page
    body = post_json(base, "/api/cell", {**address, "note": "lid  half open"})
    assert body["cell"]["note"] == "lid half open"
    assert body["progress"]["labelled"] == 0  # identity and note do not label a cell

    post_json(base, "/api/cell", {**address, "decision": 0})
    assert json.loads(get(base, "/api/state")[1])["progress"]["labelled"] == 1
    body = post_json(base, "/api/cell", {**address, "decision": None})
    assert body["cell"]["decision"] is None and body["progress"]["labelled"] == 0
    record = json.loads((root / "decisions.json").read_text())
    cell = next(c for c in record["cells"] if c["view"] == "fpv" and c["slot"] == 0)
    assert cell["decision"] is None and cell["instance_identity"] == "tube_A"
    body = post_json(base, "/api/cell", {**address, "instance_identity": ""})
    assert body["cell"]["instance_identity"] is None
    assert body["identities"] == ["centrifuge"]

    body = post_json(base, "/api/author", {"author": " N. Reviewer "})
    assert body["author"] == "N. Reviewer"
    assert json.loads((root / "decisions.json").read_text())["author"] == "N. Reviewer"


def test_invalid_changes_are_refused_and_nothing_is_written(served) -> None:
    root, app, base = served
    record_path = root / "decisions.json"
    before = sha256(record_path)
    address = {"raw_frame": 916, "view": "fpv", "slot": 0}
    code, error = post_error(base, "/api/cell", {**address, "decision": 3})
    assert code == 400 and "does not exist" in error
    code, error = post_error(base, "/api/cell", {**address, "decision": True})
    assert code == 400
    code, error = post_error(base, "/api/cell", {**address, "decision": "maybe"})
    assert code == 400 and "hidden" in error
    code, error = post_error(base, "/api/cell", {**address, "sheet": "x"})
    assert code == 400 and "unknown cell fields" in error
    code, error = post_error(base, "/api/cell", address)
    assert code == 400 and "nothing to change" in error
    code, error = post_error(base, "/api/cell", {"raw_frame": 916, "view": "fpv"})
    assert code == 400 and "slot" in error
    code, error = post_error(
        base, "/api/cell", {"raw_frame": 1, "view": "fpv", "slot": 0, "note": "x"}
    )
    assert code == 404 and "no cell" in error
    code, error = post_error(base, "/api/score", {})
    assert code == 400 and "--arms" in error
    assert sha256(record_path) == before


def test_cell_without_candidates_offers_only_hidden_and_none_fits(served) -> None:
    root, app, base = served
    page = get(base, "/frame/916/fpv")[1].decode()
    row = section(page, "row-1")
    assert 'data-candidates="[]"' in row and 'data-has-box="0"' in row
    assert 'data-decision="hidden"' in row and 'data-decision="none_fits"' in row
    assert 'data-decision="box"' not in row
    assert "tile cand" not in row
    assert "no decoder candidates" in row
    address = {"raw_frame": 916, "view": "fpv", "slot": 1}
    code, error = post_error(base, "/api/cell", {**address, "decision": "box"})
    assert code == 400 and "reference box" in error
    code, error = post_error(base, "/api/cell", {**address, "decision": 0})
    assert code == 400
    assert post_json(base, "/api/cell", {**address, "decision": "hidden"})["cell"]["decision"] == (
        "hidden"
    )
    assert post_json(base, "/api/cell", {**address, "decision": "none_fits"})["cell"][
        "decision"
    ] == ("none_fits")


def test_served_record_loads_in_the_scorer(served) -> None:
    root, app, base = served
    post_json(base, "/api/cell", {"raw_frame": 916, "view": "fpv", "slot": 0, "decision": 1})
    post_json(base, "/api/cell", {"raw_frame": 916, "view": "fpv", "slot": 1, "decision": "hidden"})
    post_json(base, "/api/cell", {"raw_frame": 637, "view": "T4", "slot": 0, "decision": "box"})
    post_json(
        base,
        "/api/cell",
        {"raw_frame": 637, "view": "T4", "slot": 1, "decision": 0, "instance_identity": "tube_A"},
    )
    record = fa.load_record(root / "decisions.json")
    anchors = fa.anchors_from_record(fa.load_workspace(root), record, workspace_dir=root)
    assert [a.state for a in anchors] == ["mask", "hidden", "box", "mask"]
    assert anchors[0].candidate_index == 1 and anchors[0].mask is not None
    assert anchors[2].bbox == (1.0, 1.0, 3.0, 3.0)
    assert anchors[3].identity == "tube_A"
    summary = fa.record_summary(anchors)
    assert summary["labelled"] == 4 and summary["identities"] == ["centrifuge", "tube_A"]


def test_existing_record_is_reused_and_completed(tmp_path: Path) -> None:
    root = make_workspace(tmp_path)
    template = json.loads((root / "decisions.template.json").read_text())
    partial = dict(template)
    partial["cells"] = [dict(template["cells"][0], decision="hidden", note="kept")]
    partial["author"] = "earlier"
    fs_common.write_json(root / "decisions.json", partial)
    app = App.open(workspace_dir=root, repository_root=tmp_path, author=None, arms={})
    record = json.loads((root / "decisions.json").read_text())
    assert record["author"] == "earlier"
    assert len(record["cells"]) == 4 and record["cells"][0]["decision"] == "hidden"
    assert app.record.progress() == {"labelled": 1, "total": 4}
    # A --record elsewhere is created there from the template.
    elsewhere = tmp_path / "elsewhere" / "mine.json"
    App.open(workspace_dir=root, record_path=elsewhere, repository_root=tmp_path, arms={})
    assert json.loads(elsewhere.read_text())["schema"] == f"{fa.SCHEMA}/decisions"


def test_score_button_runs_the_command_and_reports_errors(tmp_path: Path) -> None:
    root = make_workspace(tmp_path)
    app = App.open(workspace_dir=root, repository_root=tmp_path, author="tester", arms={})
    output = root / "scoreboard"
    writer = (
        "import pathlib, sys; out = pathlib.Path(sys.argv[1]); out.mkdir(exist_ok=True); "
        "(out / 'anchor_scoreboard.md').write_text('# Anchor scoreboard: 1 / 4 cells labelled"
        "\\n\\n| arm | cells scored |\\n|---|---|\\n| b | 1 |\\n')"
    )
    app.scorer = Scorer(
        [sys.executable, "-c", writer, str(output)], output_dir=output, cwd=tmp_path
    )
    server, base = serve_app(app)
    try:
        page = get(base, "/")[1].decode()
        assert '<button id="score" ' in page and "disabled" not in page.split('id="score"')[1][:40]
        body = post_json(base, "/api/score", {})
        assert body["ok"] is True and "| b | 1 |" in body["markdown"]
        assert body["markdown_name"] == "anchor_scoreboard.md"
        status, content, ctype = get(base, "/files/scoreboard/anchor_scoreboard.md")
        assert status == 200 and ctype.startswith("text/markdown") and b"| b | 1 |" in content
        state = json.loads(get(base, "/api/state")[1])["score"]
        assert state["available"] and not state["running"] and state["last"]["ok"]

        app.scorer = Scorer(
            [sys.executable, "-c", "import sys; sys.stderr.write('bad arms\\n'); sys.exit(3)"],
            output_dir=output,
            cwd=tmp_path,
        )
        body = post_json(base, "/api/score", {})
        assert body["ok"] is False and "exited 3" in body["error"] and "bad arms" in body["error"]

        app.scorer = Scorer(
            [sys.executable, "-c", "import time; time.sleep(1.0)"], output_dir=output, cwd=tmp_path
        )
        results: list[Any] = []

        def run() -> None:
            try:
                results.append(post_json(base, "/api/score", {}))
            except urllib.error.HTTPError as error:
                results.append((error.code, json.loads(error.read())["error"]))

        threads = [threading.Thread(target=run) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        conflicts = [r for r in results if isinstance(r, tuple)]
        assert len(conflicts) == 1 and conflicts[0][0] == 409
        assert "already in progress" in conflicts[0][1]
    finally:
        server.shutdown()
        server.server_close()


def test_cli_defaults_and_bind_rules() -> None:
    args = build_parser().parse_args(["--workspace", "ws"])
    assert args.port == 8766 and args.bind == "127.0.0.1" and args.record is None
    assert args.tailscale is False and args.arms is None
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--workspace", "ws", "--bind", "10.0.0.1", "--tailscale"])
    assert resolve_bind("127.0.0.1", False) == ("127.0.0.1", None)
    with pytest.raises(ValueError, match="unauthenticated"):
        resolve_bind("0.0.0.0", False)
