"""`battle-finebio-anchors-web`: gate 2's anchor labelling as one page per frame-view.

The human's job at gate 2 is one glance and one keypress per row (a slot on a frame in a view):
which of the SAM3 image-decoder candidates is the object, or is it hidden. The static sheets
and `decisions.template.json` that `battle-finebio-anchors workspace` writes make that a JSON
edit per cell; this module serves the same workspace as pages instead::

    battle-finebio-anchors-web --workspace runs/finebio-anchors-<trial>-<date> \\
        [--record decisions.json] [--port 8766] [--tailscale | --bind 127.0.0.1]

* An index of the frame-views in the brief's order (the six-view frame in every view first,
  then the disagreement frames inside the centrifuge-cycle neighbourhoods, then the rest) with
  the labelled count per page and overall.
* A page per frame-view: the overview (every slot's box and number), then one row per slot
  with the reference-box crop and the candidate tiles cut from the workspace's own sheet
  (duplicates greyed with their `= cK`), the current decision, an `instance_identity` input
  with autocomplete over every identity already in the record, and a `note`.
* Keys: ``0``-``3`` choose a candidate, ``h`` hidden, ``b`` box, ``n`` none_fits, ``x`` clear,
  ``j``/``k`` (or the arrows) move between rows, ``]``/``[`` change page, ``a`` accepts c0 and
  advances (the fast path), ``i`` edits the identity, ``o`` toggles the overview size.
* Every change is written at once to the record (a temporary file renamed into place) with
  `author` and `updated_at`; the record keeps the template's schema, so `battle-finebio-anchors
  score` and `export` read it unchanged. The template itself is never modified.
* "Score now" runs `battle-finebio-anchors score` in a subprocess against the trial's arms and
  shows the resulting table.

Headless: the server never opens a browser and prints its URL. Loopback by default, or this
machine's Tailscale address with ``--tailscale``; there is no authentication. FineBio frames,
masks and sheets are served read-only from the workspace directory under `runs/` and never
leave it for the repository.
"""

from __future__ import annotations

import argparse
import getpass
import html
import json
import subprocess
import sys
import threading
import time
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from PIL import Image

from . import fs_common
from .finebio_anchors import (
    CANDIDATE_KINDS,
    DECISION_WORDS,
    VIEWS,
    cell_key,
    load_record,
    load_workspace,
    parse_arms,
)
from .muggled_calibration_web import (
    DEFAULT_HOST,
    WILDCARD_HOSTS,
    reachable_urls,
    resolve_tailscale_ipv4,
    tailscale_dns_name,
)

DEFAULT_PORT = 8766
TEMPLATE_NAME = "decisions.template.json"
RECORD_NAME = "decisions.json"
SCOREBOARD_DIRECTORY = "scoreboard"
SCORE_TIMEOUT_SECONDS = 1800.0
# The sheets `render_sheets` writes: a 50 px title, then rows of equal tiles, one column for the
# context crop and one per candidate kind. The tile size is read from the image itself.
SHEET_TITLE_HEIGHT = 50
SHEET_COLUMNS = 1 + len(CANDIDATE_KINDS)
SERVED_SUFFIXES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".md": "text/markdown; charset=utf-8",
    ".json": "application/json; charset=utf-8",
}
# Trial 1's arms, by their directory names beside arm (b)'s (the workspace records arm (b)).
ARM_DIRECTORIES = {
    "a": "a-boxes-only",
    "b": "b-box-decode-arm",
    "c": "c-video-memory-arm",
    "d": "d-video-memory-arm",
}
CELL_FIELDS = ("decision", "instance_identity", "note")
ORIGIN_LABELS = {
    "six_view_annotated": "six-view frame",
    "disagreement": "disagreement",
    "random": "random",
}

CellKey = tuple[int, str, int]


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def default_author() -> str | None:
    try:
        return getpass.getuser()
    except (KeyError, OSError):
        return None


# --------------------------------------------------------------------------------------------
# pages and the record


def page_order(cells: Iterable[dict[str, Any]]) -> list[tuple[int, str]]:
    """(raw frame, view) pages in the brief's order: the six-view annotated frame in every view
    first (fpv, then the fixed views), then the disagreement frames inside the centrifuge-cycle
    neighbourhoods, then the remaining frames by number."""
    frames: dict[int, dict[str, Any]] = {}
    for cell in cells:
        info = frames.setdefault(
            int(cell["raw_frame"]),
            {
                "origin": cell.get("origin"),
                "in_cycle": bool(cell.get("in_cycle_neighbourhood", False)),
                "views": set(),
            },
        )
        info["views"].add(str(cell["view"]))

    def tier(info: dict[str, Any]) -> int:
        if info["origin"] == "six_view_annotated":
            return 0
        return 1 if info["in_cycle"] else 2

    def view_rank(view: str) -> tuple[int, str]:
        return (VIEWS.index(view) if view in VIEWS else len(VIEWS), view)

    return [
        (raw, view)
        for raw, info in sorted(frames.items(), key=lambda item: (tier(item[1]), item[0]))
        for view in sorted(info["views"], key=view_rank)
    ]


def page_url(raw_frame: int, view: str) -> str:
    return f"/frame/{int(raw_frame)}/{view}"


def page_id(raw_frame: int, view: str) -> str:
    return f"f{int(raw_frame):06d}_{view}"


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("identity and note must be strings")
    return " ".join(value.split())


def validate_decision(value: Any, cell: dict[str, Any], workspace_cell: dict[str, Any]) -> Any:
    """A decision the scorer accepts for this cell: an existing candidate index, one of the
    words (`box` only where a reference box exists) or null."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("decision must be a candidate index, a word or null")
    if isinstance(value, str) and value.isdigit():
        value = int(value)
    if isinstance(value, int):
        if value not in [int(c) for c in cell.get("candidates", [])]:
            raise ValueError(
                f"candidate {value} does not exist for this cell (have {cell.get('candidates')})"
            )
        return value
    if value in DECISION_WORDS:
        if value == "box" and workspace_cell.get("reference_box") is None:
            raise ValueError("'box' needs a reference box; this cell has none (hidden / none_fits)")
        return str(value)
    raise ValueError(
        f"decision must be a candidate index, one of {list(DECISION_WORDS)} or null; got {value!r}"
    )


class Record:
    """Owner of the decisions file: loaded when it exists, else created from the template;
    every change is validated against the workspace's cells and the whole document is written
    atomically (temporary file, then rename) with `author` and `updated_at`."""

    def __init__(
        self,
        *,
        workspace_dir: Path,
        workspace: dict[str, Any],
        path: Path,
        author: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.lock = threading.RLock()
        self.workspace_cells = {cell_key(c): c for c in workspace["cells"]}
        template = json.loads((Path(workspace_dir) / TEMPLATE_NAME).read_text(encoding="utf-8"))
        template_cells = {cell_key(c): c for c in template["cells"]}
        if self.path.is_file():
            self.doc = load_record(self.path)
            created = False
        else:
            self.doc = template
            created = True
        self.doc.setdefault("cells", [])
        present = {cell_key(c) for c in self.doc["cells"]}
        added = 0
        for key, cell in template_cells.items():
            if key not in present:
                self.doc["cells"].append(dict(cell))
                added += 1
        self.cells: dict[CellKey, dict[str, Any]] = {cell_key(c): c for c in self.doc["cells"]}
        author_changed = bool(author) and self.doc.get("author") != author
        if author_changed:
            self.doc["author"] = author
        if created or added or author_changed:
            self._save()

    def _save(self) -> None:
        self.doc["updated_at"] = utc_now()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fs_common.write_json(self.path, self.doc, atomic=True)

    def cell(self, key: CellKey) -> dict[str, Any]:
        try:
            return self.cells[key]
        except KeyError:
            raise KeyError(f"no cell for frame {key[0]} {key[1]} slot {key[2]}") from None

    def update(self, key: CellKey, changes: dict[str, Any]) -> dict[str, Any]:
        unknown = sorted(set(changes) - set(CELL_FIELDS))
        if unknown:
            raise ValueError(f"unknown cell fields {unknown}; editable: {list(CELL_FIELDS)}")
        if not changes:
            raise ValueError("nothing to change")
        with self.lock:
            cell = self.cell(key)
            workspace_cell = self.workspace_cells.get(key, {})
            if "decision" in changes:
                cell["decision"] = validate_decision(changes["decision"], cell, workspace_cell)
            if "instance_identity" in changes:
                cell["instance_identity"] = clean_text(changes["instance_identity"]) or None
            if "note" in changes:
                cell["note"] = clean_text(changes["note"])
            self._save()
            return dict(cell)

    def set_author(self, author: Any) -> str | None:
        with self.lock:
            self.doc["author"] = clean_text(author) or None
            self._save()
            return self.doc["author"]

    def progress(self, keys: Iterable[CellKey] | None = None) -> dict[str, int]:
        with self.lock:
            cells = (
                list(self.cells.values())
                if keys is None
                else [self.cells[k] for k in keys if k in self.cells]
            )
            return {
                "labelled": sum(1 for c in cells if c.get("decision") is not None),
                "total": len(cells),
            }

    def identities(self) -> list[str]:
        with self.lock:
            return sorted(
                {
                    str(c["instance_identity"])
                    for c in self.cells.values()
                    if c.get("instance_identity")
                }
            )


# --------------------------------------------------------------------------------------------
# scoring


def default_arms(workspace: dict[str, Any], repository_root: Path) -> dict[str, Path]:
    """The trial's arm directories beside the workspace's arm (b), those that exist."""
    arm_b = Path(str(workspace.get("arm_b") or ""))
    if not str(arm_b):
        return {}
    if not arm_b.is_absolute():
        arm_b = repository_root / arm_b
    arms_run = arm_b.parent
    arms = {
        name: arms_run / directory
        for name, directory in ARM_DIRECTORIES.items()
        if (arms_run / directory).is_dir()
    }
    if "b" not in arms and arm_b.is_dir():
        arms["b"] = arm_b
    return dict(sorted(arms.items()))


def score_command(
    *, workspace_dir: Path, record_path: Path, arms: dict[str, Path], output_dir: Path
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "battle.finebio_anchors",
        "score",
        "--workspace",
        str(workspace_dir),
        "--record",
        str(record_path),
        "--arms",
        ",".join(f"{name}={path}" for name, path in arms.items()),
        "--output",
        str(output_dir),
    ]


class Scorer:
    """Runs the score command in a subprocess, one run at a time, and keeps the last outcome."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        output_dir: Path,
        cwd: Path,
        timeout: float = SCORE_TIMEOUT_SECONDS,
        label: str | None = None,
    ) -> None:
        self.command = list(command)
        self.output_dir = Path(output_dir)
        self.cwd = Path(cwd)
        self.timeout = timeout
        self.label = label or " ".join(self.command)
        self.last: dict[str, Any] | None = None
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self._lock.locked()

    def run(self) -> dict[str, Any]:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("a score run is already in progress")
        started = time.perf_counter()
        try:
            try:
                completed = subprocess.run(
                    self.command,
                    cwd=self.cwd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                outcome: dict[str, Any] = {
                    "ok": False,
                    "error": f"score did not finish within {self.timeout:g} s",
                }
            except OSError as error:
                outcome = {"ok": False, "error": f"could not start the score command: {error}"}
            else:
                markdown = self.output_dir / "anchor_scoreboard.md"
                if completed.returncode == 0 and markdown.is_file():
                    outcome = {
                        "ok": True,
                        "markdown": markdown.read_text(encoding="utf-8"),
                        "markdown_name": markdown.name,
                        "json_name": "anchor_scoreboard.json",
                    }
                else:
                    detail = completed.stderr.strip() or completed.stdout.strip()
                    outcome = {
                        "ok": False,
                        "error": (
                            f"score exited {completed.returncode}: {detail[-4000:]}"
                            if detail
                            else f"score exited {completed.returncode} without a scoreboard"
                        ),
                    }
            outcome["seconds"] = round(time.perf_counter() - started, 1)
            outcome["finished_at"] = utc_now()
            outcome["command"] = self.command
            self.last = outcome
            return outcome
        finally:
            self._lock.release()


# --------------------------------------------------------------------------------------------
# the application: pages as HTML, changes as JSON


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _fmt(value: Any) -> str:
    return "-" if value is None else f"{float(value):.2f}"


def _script_json(value: Any) -> str:
    """JSON for a `<script type="application/json">` body: raw text there, so only a closing
    tag needs escaping (the browser does not decode entities inside script elements)."""
    return json.dumps(value).replace("</", "<\\/")


def decision_text(decision: Any) -> str:
    if decision is None:
        return "unlabelled"
    if isinstance(decision, int) and not isinstance(decision, bool):
        return f"c{decision}"
    return str(decision)


class App:
    """One workspace directory, its record and the pages over them."""

    def __init__(
        self,
        *,
        workspace_dir: Path,
        workspace: dict[str, Any],
        record: Record,
        scorer: Scorer | None,
    ) -> None:
        self.workspace_dir = Path(workspace_dir).resolve()
        self.workspace = workspace
        self.record = record
        self.scorer = scorer
        self.pages = page_order(workspace["cells"])
        self.page_index = {page: i for i, page in enumerate(self.pages)}
        self.page_cells: dict[tuple[int, str], list[dict[str, Any]]] = {
            page: [] for page in self.pages
        }
        for cell in workspace["cells"]:
            self.page_cells[(int(cell["raw_frame"]), str(cell["view"]))].append(cell)
        for cells in self.page_cells.values():
            cells.sort(key=lambda c: int(c["slot"]))
        self._geometry: dict[str, tuple[int, int]] = {}
        self._geometry_lock = threading.Lock()

    @classmethod
    def open(
        cls,
        *,
        workspace_dir: Path,
        record_path: Path | None = None,
        arms: dict[str, Path] | None = None,
        author: str | None = None,
        repository_root: Path,
    ) -> App:
        workspace_dir = Path(workspace_dir)
        workspace = load_workspace(workspace_dir)
        record_path = Path(record_path) if record_path else workspace_dir / RECORD_NAME
        record = Record(
            workspace_dir=workspace_dir, workspace=workspace, path=record_path, author=author
        )
        if arms is None:
            arms = default_arms(workspace, repository_root)
        scorer = None
        if arms:
            output_dir = workspace_dir / SCOREBOARD_DIRECTORY
            scorer = Scorer(
                score_command(
                    workspace_dir=workspace_dir,
                    record_path=record_path,
                    arms=arms,
                    output_dir=output_dir,
                ),
                output_dir=output_dir,
                cwd=repository_root,
                label=", ".join(
                    f"{name}={fs_common.relative_uri(Path(path), repository_root)}"
                    for name, path in arms.items()
                ),
            )
        return cls(workspace_dir=workspace_dir, workspace=workspace, record=record, scorer=scorer)

    # ---- files

    def resolve_file(self, relative: str) -> Path | None:
        """A served file under the workspace, or None: images, markdown and JSON only, never a
        path that escapes the directory."""
        target = (self.workspace_dir / unquote(relative).lstrip("/")).resolve()
        if self.workspace_dir not in target.parents:
            return None
        if not target.is_file() or target.suffix.lower() not in SERVED_SUFFIXES:
            return None
        return target

    def sheet_geometry(self, sheet: str, rows: int) -> tuple[int, int] | None:
        """(tile width, tile height) of a sheet under the workspace, from the image size."""
        path = self.workspace_dir / sheet
        if not path.is_file() or rows <= 0:
            return None
        with self._geometry_lock:
            cached = self._geometry.get(sheet)
        if cached is None:
            with Image.open(path) as image:
                width, height = image.size
            cached = (width // SHEET_COLUMNS, max(1, (height - SHEET_TITLE_HEIGHT) // rows))
            with self._geometry_lock:
                self._geometry[sheet] = cached
        return cached

    # ---- state

    def page_progress(self, page: tuple[int, str]) -> dict[str, int]:
        return self.record.progress(cell_key(c) for c in self.page_cells[page])

    def state(self) -> dict[str, Any]:
        return {
            "workspace": str(self.workspace_dir),
            "record": str(self.record.path),
            "author": self.record.doc.get("author"),
            "updated_at": self.record.doc.get("updated_at"),
            "progress": self.record.progress(),
            "pages": [
                {
                    "raw_frame": raw,
                    "view": view,
                    "url": page_url(raw, view),
                    "progress": self.page_progress((raw, view)),
                }
                for raw, view in self.pages
            ],
            "identities": self.record.identities(),
            "score": {
                "available": self.scorer is not None,
                "running": self.scorer.running if self.scorer else False,
                "last": self.scorer.last if self.scorer else None,
            },
        }

    def update_cell(self, body: dict[str, Any]) -> dict[str, Any]:
        try:
            key: CellKey = (int(body["raw_frame"]), str(body["view"]), int(body["slot"]))
        except KeyError as error:
            raise ValueError(
                f"cell address needs raw_frame, view and slot (missing {error})"
            ) from None
        changes = {k: v for k, v in body.items() if k not in ("raw_frame", "view", "slot")}
        cell = self.record.update(key, changes)
        return {
            "cell": cell,
            "page": self.page_progress((key[0], key[1])),
            "progress": self.record.progress(),
            "identities": self.record.identities(),
            "updated_at": self.record.doc.get("updated_at"),
        }

    def set_author(self, body: dict[str, Any]) -> dict[str, Any]:
        author = self.record.set_author(body.get("author"))
        return {"author": author, "updated_at": self.record.doc.get("updated_at")}

    def score(self) -> dict[str, Any]:
        if self.scorer is None:
            raise ValueError("no arm directories to score against; start with --arms name=dir,...")
        return self.scorer.run()

    # ---- HTML

    def index_html(self) -> str:
        progress = self.record.progress()
        rows = []
        for i, (raw, view) in enumerate(self.pages, start=1):
            cells = self.page_cells[(raw, view)]
            first = cells[0] if cells else {}
            origin = ORIGIN_LABELS.get(str(first.get("origin")), str(first.get("origin")))
            if first.get("in_cycle_neighbourhood"):
                origin += ", centrifuge-cycle neighbourhood"
            page = self.page_progress((raw, view))
            done = " done" if page["total"] and page["labelled"] == page["total"] else ""
            rows.append(
                f'<tr class="page{done}"><td>{i}</td>'
                f'<td><a href="{page_url(raw, view)}">f{raw:06d} {_e(view)}</a></td>'
                f"<td>{int(first.get('proxy_frame', 0))}</td><td>{_e(origin)}</td>"
                f'<td class="progress">{page["labelled"]} / {page["total"]}</td></tr>'
            )
        score_html = self._score_html()
        author = self.record.doc.get("author") or ""
        return INDEX_TEMPLATE.format(
            style=STYLE,
            script=SCRIPT,
            trial=_e(self.workspace.get("trial", "")),
            labelled=progress["labelled"],
            total=progress["total"],
            record=_e(self.record.path),
            author=_e(author),
            updated=_e(self.record.doc.get("updated_at") or ""),
            score=score_html,
            rows="".join(rows),
            keys=KEYS_HTML,
            first_url=page_url(*self.pages[0]) if self.pages else "/",
            page_data=_script_json(
                {"kind": "index", "next": page_url(*self.pages[0]) if self.pages else None}
            ),
        )

    def _score_html(self) -> str:
        if self.scorer is None:
            return (
                '<button id="score" disabled title="no arm directories found; start with '
                '--arms name=dir,...">Score now</button> '
                '<span id="score-status" class="muted">no arms to score against</span>'
                '<div id="score-result"></div>'
            )
        return (
            f'<button id="score" title="{_e(self.scorer.label)}">Score now</button> '
            '<span id="score-status" class="muted">runs battle-finebio-anchors score '
            f"({_e(self.scorer.label)}); writes {SCOREBOARD_DIRECTORY}/</span>"
            '<div id="score-result"></div>'
        )

    def frame_html(self, raw: int, view: str) -> str:
        page = (int(raw), str(view))
        if page not in self.page_index:
            raise KeyError(f"no frame-view {raw} {view}")
        cells = self.page_cells[page]
        position = self.page_index[page]
        previous = page_url(*self.pages[position - 1]) if position > 0 else None
        following = page_url(*self.pages[position + 1]) if position + 1 < len(self.pages) else None
        overview = f"sheets/{page_id(*page)}_overview.jpg"
        has_overview = (self.workspace_dir / overview).is_file()
        sheet = (
            self.workspace.get("sheets", {}).get(page_id(*page)) or f"sheets/{page_id(*page)}.jpg"
        )
        geometry = self.sheet_geometry(sheet, len(cells))
        with self.record.lock:
            rows = "".join(
                self._row_html(i, cell, self.record.cells.get(cell_key(cell), {}), sheet, geometry)
                for i, cell in enumerate(cells)
            )
            identities = self.record.identities()
            page_progress = self.page_progress(page)
            progress = self.record.progress()
        first = cells[0] if cells else {}
        origin = ORIGIN_LABELS.get(str(first.get("origin")), str(first.get("origin")))
        data = {
            "kind": "frame",
            "page_id": page_id(*page),
            "raw_frame": page[0],
            "view": page[1],
            "prev": previous,
            "next": following or "/",
            "position": position + 1,
            "pages": len(self.pages),
        }
        return FRAME_TEMPLATE.format(
            style=STYLE,
            script=SCRIPT,
            trial=_e(self.workspace.get("trial", "")),
            raw=page[0],
            view=_e(page[1]),
            proxy=int(first.get("proxy_frame", 0)),
            origin=_e(origin),
            position=position + 1,
            pages=len(self.pages),
            prev_link=f'<a href="{previous}" title="[">&lsaquo; prev</a>'
            if previous
            else '<span class="muted">&lsaquo; prev</span>',
            next_link=f'<a href="{following}" title="]">next &rsaquo;</a>'
            if following
            else '<a href="/" title="]">index &rsaquo;</a>',
            page_labelled=page_progress["labelled"],
            page_total=page_progress["total"],
            labelled=progress["labelled"],
            total=progress["total"],
            overview=(
                f'<img id="overview" class="small" src="/files/{_e(overview)}" '
                'alt="frame overview with slot boxes" title="click or o: toggle size">'
                if has_overview
                else '<p class="muted">no overview image in this workspace</p>'
            ),
            sheet_link=(
                f'<a href="/files/{_e(sheet)}" target="_blank">static sheet</a>' if geometry else ""
            ),
            datalist="".join(f'<option value="{_e(i)}"></option>' for i in identities),
            rows=rows,
            keys=KEYS_HTML,
            score=self._score_html(),
            page_data=_script_json(data),
        )

    def _row_html(
        self,
        row_index: int,
        cell: dict[str, Any],
        record_cell: dict[str, Any],
        sheet: str,
        geometry: tuple[int, int] | None,
    ) -> str:
        decision = record_cell.get("decision")
        candidates = list(cell.get("candidates", []))
        indices = [int(c["index"]) for c in candidates]
        has_box = cell.get("reference_box") is not None
        labelled = " labelled" if decision is not None else ""
        chosen_word = decision if isinstance(decision, str) else None

        def word_button(word: str, key: str) -> str:
            active = " active" if chosen_word == word else ""
            return (
                f'<button data-decision="{word}" class="{active.strip()}" title="{key}">'
                f"{word} <kbd>{key}</kbd></button>"
            )

        buttons = [word_button("hidden", "h")]
        if has_box:
            buttons.append(word_button("box", "b"))
        buttons.append(word_button("none_fits", "n"))
        buttons.append('<button data-decision="clear" title="x">clear <kbd>x</kbd></button>')

        tiles = []
        if geometry is not None:
            tiles.append(
                self._sheet_tile(
                    sheet,
                    geometry,
                    row_index,
                    0,
                    classes="tile ref",
                    caption=(
                        f"reference box &middot; det {_fmt(cell.get('arm_b_detector_score'))}"
                        if has_box
                        else "no box on this frame: hidden / none_fits only"
                    ),
                    index=None,
                    link=None,
                )
            )
        elif candidates:
            c0 = next((c for c in candidates if int(c["index"]) == 0), candidates[0])
            tiles.append(
                f'<figure class="tile ref"><figcaption>reference (c{int(c0["index"])} mask)'
                f'</figcaption><img src="/files/{_e(c0["mask_uri"])}" alt="reference"></figure>'
            )
        for candidate in sorted(candidates, key=lambda c: int(c["index"])):
            index = int(candidate["index"])
            duplicate = candidate.get("duplicate_of")
            classes = "tile cand"
            if duplicate is not None:
                classes += " dup"
            if decision == index:
                classes += " chosen"
            caption = (
                f"<kbd>{index}</kbd> {_e(candidate.get('kind', ''))} &middot; dec "
                f"{_fmt(candidate.get('decoder_iou_pred'))} &middot; bbox "
                f"{_fmt(candidate.get('mask_bbox_iou_vs_box'))}"
            )
            if duplicate is not None:
                caption += f" &middot; <b>= c{int(duplicate)}</b>"
            link = f"/files/{candidate['mask_uri']}"
            if geometry is not None:
                tiles.append(
                    self._sheet_tile(
                        sheet,
                        geometry,
                        row_index,
                        1 + index,
                        classes=classes,
                        caption=caption,
                        index=index,
                        link=link,
                    )
                )
            else:
                tiles.append(
                    f'<figure class="{classes}" data-index="{index}"><figcaption>{caption}'
                    f'</figcaption><img src="{_e(link)}" alt="candidate {index}">'
                    f'<a class="mask" href="{_e(link)}" target="_blank">mask png</a></figure>'
                )
        if not candidates:
            tiles.append(
                '<p class="muted nocand">no decoder candidates: the slot had no detector box on '
                "this frame; <b>hidden</b> or <b>none_fits</b> (or leave unlabelled)</p>"
            )
        identity = record_cell.get("instance_identity") or ""
        note = record_cell.get("note") or ""
        return (
            f'<section class="row{labelled}" id="row-{int(cell["slot"])}" '
            f'data-slot="{int(cell["slot"])}" data-candidates="{_e(json.dumps(indices))}" '
            f'data-has-box="{int(has_box)}">'
            '<div class="meta">'
            f'<div class="title"><span class="slot">slot {int(cell["slot"])}</span> '
            f'<b class="label">{_e(cell["label"])}</b><br><span class="cls">{_e(cell["class"])}'
            f"{' &middot; ' + _e(cell['role']) if cell.get('role') else ''}</span></div>"
            f'<div class="state">decision: <span class="decision">{_e(decision_text(decision))}'
            "</span></div>"
            f'<div class="actions">{"".join(buttons)}</div>'
            f'<label><span>identity <kbd>i</kbd></span><input class="identity" '
            f'list="identities" value="{_e(identity)}" placeholder="tube_A, pipette_blue, ..." '
            'autocomplete="off"></label>'
            f'<label><span>note</span><input class="note" value="{_e(note)}" '
            'placeholder="mask on glove, ..." autocomplete="off"></label>'
            "</div>"
            f'<div class="tiles">{"".join(tiles)}</div>'
            "</section>"
        )

    def _sheet_tile(
        self,
        sheet: str,
        geometry: tuple[int, int],
        row_index: int,
        column: int,
        *,
        classes: str,
        caption: str,
        index: int | None,
        link: str | None,
    ) -> str:
        width, height = geometry
        x = column * width
        y = SHEET_TITLE_HEIGHT + row_index * height
        style = (
            f"background-image:url('/files/{_e(sheet)}');background-position:-{x}px -{y}px;"
            f"width:{width}px;height:{height}px"
        )
        data_index = f' data-index="{index}"' if index is not None else ""
        mask_link = (
            f'<a class="mask" href="{_e(link)}" target="_blank">mask png</a>' if link else ""
        )
        return (
            f'<figure class="{classes}"{data_index}><figcaption>{caption}</figcaption>'
            f'<div class="pic" style="{style}"></div>{mask_link}</figure>'
        )


# --------------------------------------------------------------------------------------------
# HTML, CSS and the page script (no external files, no frameworks)

STYLE = """
:root{color-scheme:dark}
body{margin:0;background:#121212;color:#ddd;font:14px/1.4 system-ui,sans-serif}
a{color:#8ab4f8}
header{position:sticky;top:0;z-index:2;background:#1c1c1c;border-bottom:1px solid #333;
  padding:8px 14px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
header h1{font-size:16px;margin:0}
main{padding:10px 14px}
.progress{font-variant-numeric:tabular-nums}
.muted{color:#999}
.keys{color:#aaa;font-size:12px;line-height:1.8}
kbd{background:#333;border:1px solid #555;border-radius:3px;padding:0 5px;
  font:12px ui-monospace,monospace}
#status{min-width:12em}
#status.error{color:#f66}
#status.ok{color:#9c9}
#overview{display:block;max-width:100%;cursor:zoom-out;margin-bottom:8px}
#overview.small{max-width:640px;cursor:zoom-in}
.row{display:flex;gap:12px;padding:8px;margin:8px 0;border:2px solid transparent;
  border-left:6px solid #555;background:#181818;border-radius:6px}
.row.labelled{border-left-color:#3c9}
.row.current{border-color:#fc3;background:#201d12}
.meta{width:270px;flex:none;display:flex;flex-direction:column;gap:6px}
.meta .title{font-size:15px}
.meta .cls{color:#aaa;font-size:12px}
.decision{font-weight:bold;color:#fc3}
.actions{display:flex;gap:4px;flex-wrap:wrap}
.tiles{display:flex;gap:8px;flex-wrap:wrap;align-items:flex-start}
.tile{margin:0;background:#000;border:2px solid #333;border-radius:4px;cursor:pointer;
  display:flex;flex-direction:column}
.tile.ref{cursor:default}
.tile.dup{opacity:.45}
.tile.chosen{border-color:#3c9;box-shadow:0 0 0 2px #3c9}
.tile figcaption{font-size:11px;padding:2px 4px;background:#222;white-space:nowrap}
.tile .pic{background-repeat:no-repeat;background-color:#000}
.tile img{display:block;width:256px;height:256px;object-fit:contain;
  image-rendering:pixelated;background:#000}
.tile .mask{font-size:11px;padding:1px 4px;color:#777}
.nocand{align-self:center}
button{background:#2a2a2a;color:#ddd;border:1px solid #555;border-radius:4px;padding:3px 8px;
  cursor:pointer}
button.active{background:#3c9;color:#000;border-color:#3c9}
button.active kbd{color:#000;background:#9fd}
button:disabled{opacity:.5;cursor:default}
label{display:flex;flex-direction:column;gap:2px;font-size:12px;color:#aaa}
input{background:#222;color:#ddd;border:1px solid #555;border-radius:4px;padding:3px 6px;
  font:14px system-ui,sans-serif}
table{border-collapse:collapse;margin:8px 0}
td,th{border:1px solid #333;padding:3px 10px;text-align:left}
tr.done td{color:#3c9}
#score-result table{font-size:12px}
#score-result pre{white-space:pre-wrap;color:#f66}
"""

KEYS_HTML = (
    '<div class="keys"><kbd>0</kbd>-<kbd>3</kbd> candidate &nbsp; <kbd>h</kbd> hidden &nbsp; '
    "<kbd>b</kbd> box &nbsp; <kbd>n</kbd> none_fits &nbsp; <kbd>x</kbd> clear &nbsp; "
    "<kbd>a</kbd> accept c0 and advance &nbsp; <kbd>j</kbd>/<kbd>k</kbd> row &nbsp; "
    "<kbd>]</kbd>/<kbd>[</kbd> page &nbsp; <kbd>i</kbd> identity (Enter/Esc leaves) &nbsp; "
    "<kbd>o</kbd> overview size &nbsp; <kbd>-</kbd>/<kbd>=</kbd> tile size</div>"
)

INDEX_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Anchors {trial}: {labelled} / {total}</title>
<style>{style}</style></head>
<body>
<header><h1>Review anchors {trial}</h1>
<span class="progress"><b id="global-progress">{labelled} / {total}</b> cells labelled</span>
<a href="{first_url}">start &rsaquo;</a>
<span id="status" class="muted"></span></header>
<main>
<p class="muted">record <code>{record}</code> &middot; author
<input id="author" value="{author}" placeholder="your name" autocomplete="off">
&middot; last write <span id="updated">{updated}</span></p>
<p>{score}</p>
<table><thead><tr><th>#</th><th>frame-view</th><th>proxy</th><th>why</th><th>labelled</th></tr>
</thead><tbody>{rows}</tbody></table>
{keys}
<p class="muted">One glance and one keypress per row: on a frame page <kbd>a</kbd> accepts arm
(b)'s own mask (c0) and moves to the next row; choose another candidate with its digit, or
<kbd>h</kbd> when the object is not visible. Every change is saved to the record at once.</p>
</main>
<script id="page-data" type="application/json">{page_data}</script>
<script>{script}</script>
</body></html>
"""

FRAME_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>f{raw:06d} {view}: {page_labelled} / {page_total}</title>
<style>{style}</style></head>
<body>
<header><a href="/">index</a> {prev_link}
<h1>f{raw:06d} {view} <span class="muted">(proxy {proxy}; {origin}; page {position} of {pages})
</span></h1> {next_link}
<span class="progress">page <b id="page-progress">{page_labelled} / {page_total}</b> &middot;
all <b id="global-progress">{labelled} / {total}</b></span>
<span id="status" class="muted"></span>
{sheet_link}</header>
<main>
{overview}
<datalist id="identities">{datalist}</datalist>
<div id="rows">{rows}</div>
{keys}
<p>{score}</p>
</main>
<script id="page-data" type="application/json">{page_data}</script>
<script>{script}</script>
</body></html>
"""

SCRIPT = r"""
(function () {
  'use strict';
  var data = JSON.parse(document.getElementById('page-data').textContent);
  var statusEl = document.getElementById('status');
  function status(text, kind) {
    if (!statusEl) return;
    statusEl.textContent = text;
    statusEl.className = kind || 'muted';
  }
  function esc(s) {
    return String(s).replace(/[&<>"]/g, function (c) {
      return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c];
    });
  }
  async function postJson(url, payload) {
    var res = await fetch(url, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload || {})
    });
    var body = null;
    try { body = await res.json(); } catch (e) { body = {error: res.statusText}; }
    if (!res.ok) throw new Error(body.error || res.statusText);
    return body;
  }

  // ---- author (index page)
  var author = document.getElementById('author');
  if (author) {
    author.addEventListener('change', function () {
      postJson('/api/author', {author: author.value}).then(function (body) {
        document.getElementById('updated').textContent = body.updated_at || '';
        status('author saved', 'ok');
      }).catch(function (e) { status(e.message, 'error'); });
    });
  }

  // ---- score button (both pages)
  var scoreButton = document.getElementById('score');
  var scoreStatus = document.getElementById('score-status');
  var scoreResult = document.getElementById('score-result');
  function tableHtml(rows) {
    var out = '<table><thead><tr>';
    rows[0].forEach(function (c) { out += '<th>' + esc(c) + '</th>'; });
    out += '</tr></thead><tbody>';
    rows.slice(1).forEach(function (r) {
      out += '<tr>';
      r.forEach(function (c) { out += '<td>' + esc(c) + '</td>'; });
      out += '</tr>';
    });
    return out + '</tbody></table>';
  }
  function renderMarkdown(text) {
    var out = [];
    var table = null;
    text.split('\n').forEach(function (line) {
      if (line.charAt(0) === '|') {
        var cells = line.split('|').slice(1, -1).map(function (s) { return s.trim(); });
        if (cells.every(function (c) { return /^-+$/.test(c); })) return;
        if (!table) table = [];
        table.push(cells);
        return;
      }
      if (table) { out.push(tableHtml(table)); table = null; }
      if (line.indexOf('# ') === 0) out.push('<h3>' + esc(line.slice(2)) + '</h3>');
      else if (line.trim()) out.push('<p class="muted">' + esc(line) + '</p>');
    });
    if (table) out.push(tableHtml(table));
    return out.join('');
  }
  if (scoreButton && !scoreButton.disabled) {
    scoreButton.addEventListener('click', async function () {
      scoreButton.disabled = true;
      scoreStatus.textContent = 'scoring... (reads four arms; tens of seconds)';
      scoreResult.innerHTML = '';
      try {
        var body = await postJson('/api/score', {});
        if (!body.ok) throw new Error(body.error || 'score failed');
        scoreStatus.textContent = 'scored in ' + body.seconds + ' s at ' + body.finished_at;
        scoreResult.innerHTML = renderMarkdown(body.markdown) +
          '<p><a href="/files/scoreboard/' + esc(body.markdown_name) + '" target="_blank">' +
          'anchor_scoreboard.md</a> &middot; <a href="/files/scoreboard/' +
          esc(body.json_name) + '" target="_blank">anchor_scoreboard.json</a></p>';
      } catch (e) {
        scoreStatus.textContent = 'score failed';
        var pre = document.createElement('pre');
        pre.textContent = e.message;
        scoreResult.appendChild(pre);
      } finally {
        scoreButton.disabled = false;
      }
    });
  }

  // ---- index page: ] goes to the first frame-view
  if (data.kind === 'index') {
    document.addEventListener('keydown', function (e) {
      var t = e.target;
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA')) {
        if (e.key === 'Enter' || e.key === 'Escape') { t.blur(); e.preventDefault(); }
        return;
      }
      if (e.key === ']' && data.next) { location.href = data.next; e.preventDefault(); }
    });
    return;
  }

  // ---- frame page
  var rows = Array.prototype.slice.call(document.querySelectorAll('.row'));
  if (!rows.length) return;
  var storageKey = 'finebio-anchors-row:' + data.page_id;
  var current = -1;
  function rowCandidates(row) { return JSON.parse(row.dataset.candidates || '[]'); }
  function setCurrent(i, scroll) {
    if (i < 0 || i >= rows.length) return false;
    if (current >= 0) rows[current].classList.remove('current');
    current = i;
    rows[current].classList.add('current');
    try { localStorage.setItem(storageKey, String(i)); } catch (e) { /* private mode */ }
    if (scroll !== false) rows[current].scrollIntoView({block: 'nearest', behavior: 'smooth'});
    return true;
  }
  var remembered = null;
  try { remembered = localStorage.getItem(storageKey); } catch (e) { remembered = null; }
  var start = remembered === null ? -1 : Number(remembered);
  if (!(start >= 0 && start < rows.length)) {
    start = rows.findIndex(function (r) { return !r.classList.contains('labelled'); });
    if (start < 0) start = 0;
  }
  setCurrent(start, false);

  function decisionText(d) {
    if (d === null || d === undefined) return 'unlabelled';
    if (typeof d === 'number') return 'c' + d;
    return String(d);
  }
  function applyCell(row, body) {
    var cell = body.cell;
    row.querySelector('.decision').textContent = decisionText(cell.decision);
    row.classList.toggle('labelled', cell.decision !== null);
    row.querySelectorAll('.cand').forEach(function (tile) {
      tile.classList.toggle('chosen', Number(tile.dataset.index) === cell.decision);
    });
    row.querySelectorAll('button[data-decision]').forEach(function (b) {
      b.classList.toggle('active', b.dataset.decision === cell.decision);
    });
    var identity = row.querySelector('.identity');
    if (document.activeElement !== identity) identity.value = cell.instance_identity || '';
    var note = row.querySelector('.note');
    if (document.activeElement !== note) note.value = cell.note || '';
    document.getElementById('page-progress').textContent =
      body.page.labelled + ' / ' + body.page.total;
    document.getElementById('global-progress').textContent =
      body.progress.labelled + ' / ' + body.progress.total;
    var list = document.getElementById('identities');
    list.innerHTML = body.identities.map(function (i) {
      return '<option value="' + esc(i) + '"></option>';
    }).join('');
    status('saved ' + (body.updated_at || ''), 'ok');
  }
  function post(row, changes) {
    var payload = {raw_frame: data.raw_frame, view: data.view, slot: Number(row.dataset.slot)};
    Object.keys(changes).forEach(function (k) { payload[k] = changes[k]; });
    return postJson('/api/cell', payload).then(function (body) {
      applyCell(row, body);
      return body;
    }).catch(function (e) { status(e.message, 'error'); return null; });
  }
  function decide(row, decision) {
    var candidates = rowCandidates(row);
    if (typeof decision === 'number' && candidates.indexOf(decision) < 0) {
      status('no candidate c' + decision + ' on this row (have ' +
        (candidates.length ? candidates.map(function (c) { return 'c' + c; }).join(' ') : 'none') +
        ')', 'error');
      return Promise.resolve(null);
    }
    if (decision === 'box' && row.dataset.hasBox !== '1') {
      status('no reference box on this row: hidden or none_fits', 'error');
      return Promise.resolve(null);
    }
    return post(row, {decision: decision});
  }
  function advance() {
    if (!setCurrent(current + 1)) status('last row; ] goes to the next page', 'muted');
  }
  var overview = document.getElementById('overview');
  function toggleOverview() { if (overview) overview.classList.toggle('small'); }
  if (overview) overview.addEventListener('click', toggleOverview);

  // Tile size: the sheet crops have fixed pixel geometry, so the row is zoomed as a whole.
  var zoomKey = 'finebio-anchors-zoom';
  var zoom = 1;
  try { zoom = Number(localStorage.getItem(zoomKey)) || 1; } catch (e) { zoom = 1; }
  function applyZoom() {
    document.querySelectorAll('.tiles').forEach(function (t) { t.style.zoom = String(zoom); });
  }
  function setZoom(value) {
    zoom = Math.min(1.5, Math.max(0.4, Math.round(value * 20) / 20));
    try { localStorage.setItem(zoomKey, String(zoom)); } catch (e) { /* private mode */ }
    applyZoom();
    status('tiles at ' + Math.round(zoom * 100) + '%', 'muted');
  }
  applyZoom();

  document.addEventListener('keydown', function (e) {
    var t = e.target;
    if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) {
      if (e.key === 'Enter' || e.key === 'Escape') { t.blur(); e.preventDefault(); }
      return;
    }
    if (e.altKey || e.ctrlKey || e.metaKey) return;
    var row = rows[current];
    switch (e.key) {
      case '0': case '1': case '2': case '3':
        decide(row, Number(e.key)); break;
      case 'h': decide(row, 'hidden'); break;
      case 'b': decide(row, 'box'); break;
      case 'n': decide(row, 'none_fits'); break;
      case 'x': decide(row, null); break;
      case 'a':
        if (rowCandidates(row).indexOf(0) < 0) {
          status('no c0 on this row: h (hidden) or n (none_fits), then j', 'error');
        } else {
          decide(row, 0).then(function (body) { if (body) advance(); });
        }
        break;
      case 'j': case 'ArrowDown': setCurrent(current + 1); break;
      case 'k': case 'ArrowUp': setCurrent(current - 1); break;
      case ']': if (data.next) location.href = data.next; break;
      case '[': if (data.prev) location.href = data.prev; break;
      case 'i': row.querySelector('.identity').focus(); break;
      case 'o': toggleOverview(); break;
      case '-': setZoom(zoom - 0.1); break;
      case '=': case '+': setZoom(zoom + 0.1); break;
      default: return;
    }
    e.preventDefault();
  });

  rows.forEach(function (row, i) {
    row.addEventListener('mousedown', function () { setCurrent(i, false); });
    row.querySelectorAll('.cand').forEach(function (tile) {
      tile.addEventListener('click', function (e) {
        if (e.target.closest('a')) return;
        setCurrent(i, false);
        decide(row, Number(tile.dataset.index));
      });
    });
    row.querySelectorAll('button[data-decision]').forEach(function (b) {
      b.addEventListener('click', function () {
        setCurrent(i, false);
        decide(row, b.dataset.decision === 'clear' ? null : b.dataset.decision);
      });
    });
    var identity = row.querySelector('.identity');
    identity.addEventListener('change', function () {
      post(row, {instance_identity: identity.value});
    });
    var note = row.querySelector('.note');
    note.addEventListener('change', function () { post(row, {note: note.value}); });
  });
})();
"""


# --------------------------------------------------------------------------------------------
# HTTP


def _send(handler: BaseHTTPRequestHandler, status: HTTPStatus, content: bytes, ctype: str) -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", ctype)
    handler.send_header("Content-Length", str(len(content)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(content)


def _json(handler: BaseHTTPRequestHandler, status: HTTPStatus, body: object) -> None:
    _send(handler, status, json.dumps(body).encode(), "application/json; charset=utf-8")


def _html(handler: BaseHTTPRequestHandler, text: str) -> None:
    _send(handler, HTTPStatus.OK, text.encode("utf-8"), "text/html; charset=utf-8")


def make_handler(app: App) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args: object) -> None:
            return

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            value = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(value, dict):
                raise ValueError("JSON body must be an object")
            return value

        def _file(self, relative: str) -> None:
            target = app.resolve_file(relative)
            if target is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            content = target.read_bytes()
            ctype = SERVED_SUFFIXES[target.suffix.lower()]
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(content)))
            # Sheets and masks never change while the server runs; the record and scoreboard do.
            self.send_header(
                "Cache-Control", "max-age=3600" if ctype.startswith("image/") else "no-store"
            )
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            try:
                if path == "/":
                    _html(self, app.index_html())
                elif path.startswith("/frame/"):
                    parts = path.removeprefix("/frame/").strip("/").split("/")
                    if len(parts) != 2 or not parts[0].isdigit():
                        self.send_error(HTTPStatus.NOT_FOUND)
                        return
                    _html(self, app.frame_html(int(parts[0]), unquote(parts[1])))
                elif path == "/api/state":
                    _json(self, HTTPStatus.OK, app.state())
                elif path.startswith("/files/"):
                    self._file(path.removeprefix("/files/"))
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
            except KeyError as error:
                _json(self, HTTPStatus.NOT_FOUND, {"error": str(error)})

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            try:
                body = self._body()
                if path == "/api/cell":
                    _json(self, HTTPStatus.OK, app.update_cell(body))
                elif path == "/api/author":
                    _json(self, HTTPStatus.OK, app.set_author(body))
                elif path == "/api/score":
                    _json(self, HTTPStatus.OK, app.score())
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
            except KeyError as error:
                _json(self, HTTPStatus.NOT_FOUND, {"error": str(error)})
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                _json(self, HTTPStatus.BAD_REQUEST, {"error": str(error)})
            except RuntimeError as error:
                _json(self, HTTPStatus.CONFLICT, {"error": str(error)})
            except OSError as error:
                _json(
                    self,
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": f"could not write the record: {error}"},
                )

    return Handler


# --------------------------------------------------------------------------------------------
# CLI


def resolve_bind(bind: str, tailscale: bool) -> tuple[str, str | None]:
    """The address to bind and the MagicDNS name to advertise; wildcards are refused because
    the workspace has no authentication."""
    if tailscale:
        return resolve_tailscale_ipv4(), tailscale_dns_name()
    if bind.strip() in WILDCARD_HOSTS:
        raise ValueError(
            f"refusing to bind every interface ({bind!r}): the workspace is unauthenticated and "
            "a POST edits the record; use --tailscale for the tailnet or an explicit --bind address"
        )
    return bind, None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Serve a battle-finebio-anchors workspace as pages: one keypress per row, the "
            "record saved on every change. Loopback by default, or --tailscale; headless."
        )
    )
    parser.add_argument("--workspace", type=Path, required=True, help="the anchors workspace")
    parser.add_argument(
        "--record",
        type=Path,
        default=None,
        help=f"decisions file (default <workspace>/{RECORD_NAME}; created from the template)",
    )
    parser.add_argument(
        "--arms",
        default=None,
        help=(
            "name=dir,... for Score now (default: the trial's a/b/c/d beside the workspace's "
            "arm (b))"
        ),
    )
    parser.add_argument(
        "--author", default=default_author(), help="written into the record (default: $USER)"
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="0 picks a free port")
    where = parser.add_mutually_exclusive_group()
    where.add_argument(
        "--bind", default=DEFAULT_HOST, help=f"address to bind (default {DEFAULT_HOST})"
    )
    where.add_argument(
        "--tailscale",
        action="store_true",
        help="bind this machine's Tailscale IPv4 (`tailscale ip -4`) for the tailnet",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        host, dns_name = resolve_bind(args.bind, args.tailscale)
        app = App.open(
            workspace_dir=args.workspace,
            record_path=args.record,
            arms=parse_arms(args.arms) if args.arms else None,
            author=args.author,
            repository_root=Path.cwd().resolve(),
        )
        server = ThreadingHTTPServer((host, args.port), make_handler(app))
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    urls = reachable_urls(host, server.server_port, dns_name)
    progress = app.record.progress()
    print(f"Anchor workspace: {urls[0]}", flush=True)
    for url in urls[1:]:
        print(f"Also reachable at: {url}", flush=True)
    if host != DEFAULT_HOST:
        print(
            "Bound beyond loopback: no authentication; any client that reaches this address "
            "can edit the record.",
            flush=True,
        )
    print(
        f"Record: {app.record.path} ({progress['labelled']} / {progress['total']} labelled; "
        f"{len(app.pages)} frame-views)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
