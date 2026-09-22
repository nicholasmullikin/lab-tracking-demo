"""Equivalence harness for the `src/battle` deduplication passes.

Every pass that consolidates helpers (hashes, relative URIs, fingerprints, mask ops,
observation loaders) must leave the persisted provenance byte-identical: the same SHA-256
digests, the same repository-relative URIs, the same manifest fields.  This script proves
it by rebuilding three CPU artifacts into `runs/dedup-equivalence/<label>/` before and after
a pass and deep-comparing the two trees:

1. `battle-build-multiview-part-consensus` on the r1280 pm-append inputs (~75 s);
2. `battle-anchor-iou` over the twenty first-minute arms of `runs/anchor-scoreboard-20260919`
   plus that run's own grouping script (seconds);
3. `battle-build-interaction-review-v4 --layers reference_masks --verify-fingerprints` into the
   scratch root (~20 s), then `rerun rrd verify` and `rerun rrd stats` on the recording.

`snapshot` also records the default pytest tier, `-m real_data`, and `ruff check` into
`summary.json` (skippable with `--skip-tests`).

`compare` walks both trees.  JSON / JSONL are parsed and compared after dropping volatile keys
(`generated_at`, `created_at`, `elapsed_seconds`, `duration_s`, `runtime_seconds`, `run_id`,
any key ending in `_at`, a `comparison_id` that embeds a timestamp) and after replacing the
scratch label inside every string, so `runs/dedup-equivalence/before/...` equals
`runs/dedup-equivalence/after/...`.  Markdown is compared as text with the same substitution.
`.npz` archives are compared array by array (the zip container embeds a write time).  PNGs and
every other file are compared byte for byte.

The Rerun `.rrd` is not byte-deterministic (measured on two builds of unchanged code: the
recording carries `RecordingInfo:start_time` and a `log_time` per row, the SDK batches rows
into chunks of varying size, and the blueprint store gets fresh view/container UUIDs), so it
is compared by bytes first and, when those differ, by a chunk-independent *content digest*:
for every entity path and column of the recording store the rows of all chunks are
concatenated, sorted on the recording's own timelines and hashed (`RowId`, `log_time` and
`RecordingInfo:start_time` dropped; string columns have the scratch label normalised); the
blueprint store is hashed as a UUID-normalised multiset of rows per entity kind
(`ViewportBlueprint:root_container`, a raw UUID reference, dropped).  `snapshot` writes that
digest beside the recording as `<name>.rrd.content.json`.

A `{uri, sha256}` fingerprint whose `uri` points into the scratch root (the index's own
recording, guide and contact sheet; the consensus manifest's `per_frame.jsonl`) is compared
as `<scratch-file>`: the file it names is compared by the walk above, and `compare` re-hashes
every such file on each side to prove the recorded digest still matches it (the review guide
embeds its own absolute path, so its digest legitimately differs between labels).  A
non-volatile difference exits 1 and prints the first twenty differing paths.

`compare-runs A B` (pass 2) compares two run directories of one GPU driver executed before and
after a refactor: every PNG under `native/masks/` or `masks/` byte for byte (reported
separately), then the same tree walk as `compare` with both run directories (absolute,
repository-relative and the run id itself) folded onto `<run>`, the per-execution
measurements `time_to_first_usable_output_seconds` and `gpu_peak_vram_bytes` treated as
volatile in addition, and `--replace OLD=NEW` for anything else that legitimately differs
(the two code-snapshot directories named in `worker_command.txt`).  Any mask difference or
non-volatile difference exits 1; `--report` writes the result as JSON.

Usage:

    uv run python scripts/dedup_equivalence.py snapshot --label before
    uv run python scripts/dedup_equivalence.py snapshot --label after
    uv run python scripts/dedup_equivalence.py compare
    uv run python scripts/dedup_equivalence.py compare-runs runs/x/before/<run> runs/x/after/<run>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCRATCH_ROOT = Path("runs/dedup-equivalence")
SCRATCH_PLACEHOLDER = "runs/dedup-equivalence/<label>"

VOLATILE_KEYS = frozenset(
    {
        "generated_at",
        "created_at",
        "elapsed_seconds",
        "duration_s",
        "runtime_seconds",
        "run_id",
    }
)
VOLATILE_KEY_SUFFIXES = ("_at",)
# `comparison_id` is dropped only when it embeds a timestamp; a fixed id (the review v4
# index uses `interaction_review_first_minute_v4`) is provenance and must match.
TIMESTAMP_PATTERN = re.compile(r"\d{8}t\d{6}z|\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}", re.IGNORECASE)
MAX_REPORTED_DIFFS = 20

CONSENSUS_REFERENCE = Path(
    "runs/sam3-memory-arms-20260919/arms/pm-append/"
    "muggledsam-sam3-four-part-static-focused-reassembly-static-c10379-20260920t033947z-r1280-pm-append"
)
CONSENSUS_VIEWS_ROOT = Path("runs/sam3-views-r1280-20260919/views")
CONSENSUS_VIEWS = (
    "C10095",
    "C10115",
    "C10118",
    "C10119",
    "C10390",
    "C10395",
    "C10404",
    "HMC_21179183",
)
CONSENSUS_EGO_VIEW = "HMC_21179183"
CONSENSUS_ROOT_FOR_REVIEW = Path("runs/multiview-part-consensus-first-minute-r1280-pm-append")

SCOREBOARD_REFERENCE_RUN = Path("runs/anchor-scoreboard-20260919")
SCOREBOARD_ANCHORS = Path("runs/human-review-anchors-first-minute")
_POLICY = Path("runs/sam3-policy-ablation-20260918/arms")
_MEMORY = Path("runs/sam3-memory-arms-20260919/arms")
_DAM4SAM = Path("runs/dam4sam-arms-20260919/arms")
SCOREBOARD_ARMS: tuple[tuple[str, Path], ...] = (
    (
        "reference",
        Path(
            "runs/muggledsam-sam3-four-part-static-focused-reassembly-static-c10379-20260918t001210z"
        ),
    ),
    ("ensemble-reference-v1", Path("runs/ensemble-reference-first-minute-v1")),
    ("ensemble-reference-v2", Path("runs/ensemble-reference-first-minute-v2")),
    ("off-r720-sched", _POLICY / "off-r720-sched"),
    ("off-r1008-sched", _POLICY / "off-r1008-sched"),
    ("off-r1280-sched", _POLICY / "off-r1280-sched"),
    ("xg-r1280-sched", _POLICY / "xg-r1280-sched"),
    ("off-r1920-sched", Path("runs/sam3-c10379-r1920-20260919/arms/off-r1920-sched")),
    ("dam4sam-60s", Path("runs/dam4sam-four-part-reviewed-seed-60s-20260918t005416z")),
    ("drop-900", _MEMORY / "drop-900"),
    ("fm6", _MEMORY / "fm6"),
    ("pm-append", _MEMORY / "pm-append"),
    ("pm-append-keepfm", _MEMORY / "pm-append-keepfm"),
    ("pm-append-fm6", _MEMORY / "pm-append-fm6"),
    ("fm8", _MEMORY / "fm8"),
    ("dam4sam-tiny-1024-sched-60s", _DAM4SAM / "dam4sam-tiny-1024-sched-60s"),
    ("dam4sam-large-1024-seed0-60s", _DAM4SAM / "dam4sam-large-1024-seed0-60s"),
    ("dam4sam-large-1024-sched-60s", _DAM4SAM / "dam4sam-large-1024-sched-60s"),
    ("samurai-large-1024-seed0-60s", _DAM4SAM / "samurai-large-1024-seed0-60s"),
)

REVIEW_RRD_NAME = "interaction_review_first_minute_v4.rrd"
STEPS = ("consensus", "scoreboard", "review", "tests", "real_data", "ruff")


# --------------------------------------------------------------------------- snapshot


def _env() -> dict[str, str]:
    env = dict(os.environ)
    # Nothing here needs a GPU and a foreign process may hold the card.
    env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def _run(
    argv: Sequence[str],
    *,
    cwd: Path,
    log: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(argv), flush=True)
    completed = subprocess.run(
        list(argv),
        cwd=cwd,
        env=_env(),
        text=True,
        capture_output=True,
    )
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if check and completed.returncode != 0:
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise SystemExit(f"{argv[0]} failed with exit code {completed.returncode}")
    return completed


def _discover_view_run(view: str) -> Path:
    candidates = sorted(p for p in (CONSENSUS_VIEWS_ROOT / view).iterdir() if p.is_dir())
    if len(candidates) != 1:
        raise SystemExit(f"expected exactly one run under {CONSENSUS_VIEWS_ROOT / view}")
    return candidates[0]


def snapshot_consensus(repository_root: Path, label_root: Path) -> dict[str, Any]:
    output_root = label_root / "consensus"
    argv = [
        "uv",
        "run",
        "battle-build-multiview-part-consensus",
        "--output-root",
        str(output_root),
        "--reference-run",
        str(CONSENSUS_REFERENCE),
        "--ego-view",
        CONSENSUS_EGO_VIEW,
    ]
    for view in CONSENSUS_VIEWS:
        argv += ["--view-run", f"{view}={_discover_view_run(view)}"]
    started = time.monotonic()
    _run(argv, cwd=repository_root, log=label_root / "consensus.build.log")
    return {"seconds": round(time.monotonic() - started, 1), "output_root": str(output_root)}


def snapshot_scoreboard(repository_root: Path, label_root: Path) -> dict[str, Any]:
    output_root = label_root / "scoreboard"
    output_root.mkdir(parents=True, exist_ok=True)
    argv = [
        "uv",
        "run",
        "battle-anchor-iou",
        "--anchors",
        str(SCOREBOARD_ANCHORS),
    ]
    for name, path in SCOREBOARD_ARMS:
        argv += ["--run", f"{name}={path}"]
    argv += [
        "--output",
        str(output_root / "anchor_iou.json"),
        "--markdown",
        str(output_root / "anchor_iou.scorer_order.md"),
        "--sheet",
        str(output_root / "anchors_vs_reference_vs_best.png"),
        "--sheet-reference",
        "reference",
        "--sheet-best",
        "pm-append",
    ]
    started = time.monotonic()
    _run(argv, cwd=repository_root, log=label_root / "scoreboard.build.log")
    _run(
        [
            "uv",
            "run",
            "python",
            str(SCOREBOARD_REFERENCE_RUN / "scoreboard_table.py"),
            str(output_root / "anchor_iou.json"),
            str(output_root / "anchor_iou.md"),
        ],
        cwd=repository_root,
        log=label_root / "scoreboard.table.log",
    )
    return {"seconds": round(time.monotonic() - started, 1), "output_root": str(output_root)}


def snapshot_review(repository_root: Path, label_root: Path) -> dict[str, Any]:
    output_root = label_root / "review_v4"
    argv = [
        "uv",
        "run",
        "battle-build-interaction-review-v4",
        "--output-root",
        str(output_root),
        "--layers",
        "reference_masks",
        "--verify-fingerprints",
        "--overwrite",
        "--multiview-consensus",
        str(CONSENSUS_ROOT_FOR_REVIEW),
        "--quiet",
    ]
    started = time.monotonic()
    _run(argv, cwd=repository_root, log=label_root / "review_v4.build.log")
    result: dict[str, Any] = {
        "seconds": round(time.monotonic() - started, 1),
        "output_root": str(output_root),
    }
    rrd = output_root / REVIEW_RRD_NAME
    digest = rrd_content_digest(rrd, labels=(label_root.name,))
    (output_root / f"{REVIEW_RRD_NAME}.content.json").write_text(
        json.dumps(digest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result["rrd_content_columns"] = len(digest)
    rerun = _rerun_executable(repository_root)
    if rerun is None:
        result["rrd_verify"] = "skipped: rerun CLI not found"
        print("rerun CLI not on PATH nor in the venv; skipping `rerun rrd verify`")
        return result
    verify = _run(
        [rerun, "rrd", "verify", str(rrd)],
        cwd=repository_root,
        log=label_root / "review_v4.rrd_verify.log",
        check=False,
    )
    result["rrd_verify"] = "ok" if verify.returncode == 0 else f"exit {verify.returncode}"
    # Stats (entity list, chunk and row counts, sizes) kept for a human reader; the chunk
    # counts vary between builds, so the file is a `.log` that `compare` skips.
    _run(
        [rerun, "rrd", "stats", str(rrd)],
        cwd=repository_root,
        log=label_root / "review_v4.rrd_stats.log",
        check=False,
    )
    return result


RRD_VOLATILE_COLUMNS = frozenset(
    {"rerun.controls.RowId", "log_time", "log_tick", "RecordingInfo:start_time"}
)
RRD_BLUEPRINT_REFERENCE_COLUMNS = frozenset({"ViewportBlueprint:root_container"})
UUID_PATTERN = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def _arrow_ipc_bytes(table: Any) -> bytes:
    import pyarrow as pa

    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def _is_string_like(arrow_type: Any) -> bool:
    import pyarrow as pa

    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return True
    if pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type):
        return _is_string_like(arrow_type.value_type)
    return False


def rrd_content_digest(
    path: Path, *, labels: Iterable[str] = (), replacements: Replacements = ()
) -> dict[str, str]:
    """Digest a recording's content independently of chunk layout, log time and blueprint ids.

    Returns ``{"recording:<entity>#<column>": "<sha256[:16]>/<rows>", "blueprint:<entity>":
    "<sha256[:16]>/<rows>"}``.  Recording columns are hashed after concatenating every chunk
    of the entity that shares the same column set and sorting on the recording's timelines;
    string columns are hashed from their Python values with the scratch labels normalised,
    everything else (scalars, blobs) from the Arrow IPC bytes of the combined column.
    """
    import hashlib
    from collections import defaultdict

    import pyarrow as pa
    from rerun.experimental import RrdReader

    labels = tuple(labels)
    reader = RrdReader(path)
    out: dict[str, str] = {}
    groups: dict[tuple[str, tuple[str, ...]], list[Any]] = defaultdict(list)
    for store in reader.recordings():
        for chunk in reader.stream(store=store):
            batch = chunk.to_record_batch()
            names = tuple(sorted(n for n in batch.schema.names if n not in RRD_VOLATILE_COLUMNS))
            groups[(chunk.entity_path, names)].append(
                pa.Table.from_batches([batch]).select(list(names))
            )
    for (entity, names), tables in sorted(groups.items()):
        table = pa.concat_tables(tables).combine_chunks()
        index_columns = [n for n in names if ":" not in n]
        if index_columns:
            table = table.sort_by([(n, "ascending") for n in index_columns])
        for name in names:
            column = table.column(name)
            if _is_string_like(column.type):
                payload = normalise_text(repr(column.to_pylist()), labels, replacements).encode(
                    "utf-8"
                )
            else:
                payload = _arrow_ipc_bytes(pa.table({name: column}).combine_chunks())
            digest = hashlib.sha256(payload).hexdigest()[:16]
            out[f"recording:{entity}#{name}"] = f"{digest}/{table.num_rows}"
    rows: dict[str, list[bytes]] = defaultdict(list)
    for store in reader.blueprints():
        for chunk in reader.stream(store=store):
            batch = chunk.to_record_batch()
            names = tuple(
                n
                for n in batch.schema.names
                if n not in RRD_VOLATILE_COLUMNS and n not in RRD_BLUEPRINT_REFERENCE_COLUMNS
            )
            table = pa.Table.from_batches([batch]).select(list(names))
            entity = UUID_PATTERN.sub("<uuid>", chunk.entity_path)
            for row in table.to_pylist():
                canonical = UUID_PATTERN.sub("<uuid>", repr(sorted(row.items())))
                rows[entity].append(hashlib.sha256(canonical.encode("utf-8")).digest())
    for entity, hashes in sorted(rows.items()):
        digest = hashlib.sha256(b"".join(sorted(hashes))).hexdigest()[:16]
        out[f"blueprint:{entity}"] = f"{digest}/{len(hashes)}"
    return out


def _rerun_executable(repository_root: Path) -> str | None:
    on_path = shutil.which("rerun")
    if on_path:
        return on_path
    in_venv = repository_root / ".venv" / "bin" / "rerun"
    if in_venv.is_file():
        return str(in_venv)
    return None


PYTEST_SUMMARY = re.compile(
    r"(\d+) (passed|failed|skipped|error|errors|xfailed|xpassed|deselected)"
)


def _pytest_counts(output: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in reversed(output.splitlines()):
        found = PYTEST_SUMMARY.findall(line)
        if found:
            for number, kind in found:
                counts[kind.rstrip("s") if kind == "errors" else kind] = int(number)
            break
    return counts


def snapshot_pytest(
    repository_root: Path, label_root: Path, *, marker: str | None
) -> dict[str, Any]:
    argv = ["uv", "run", "pytest", "-q", "-p", "no:cacheprovider"]
    if marker:
        argv += ["-m", marker]
    started = time.monotonic()
    name = f"pytest.{marker or 'default'}.log"
    completed = _run(argv, cwd=repository_root, log=label_root / name, check=False)
    return {
        "seconds": round(time.monotonic() - started, 1),
        "exit_code": completed.returncode,
        "counts": _pytest_counts(completed.stdout),
    }


def snapshot_ruff(repository_root: Path, label_root: Path) -> dict[str, Any]:
    completed = _run(
        ["uv", "run", "ruff", "check", "src", "tests", "scripts"],
        cwd=repository_root,
        log=label_root / "ruff.log",
        check=False,
    )
    return {"exit_code": completed.returncode, "ok": completed.returncode == 0}


def _git_head(repository_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=repository_root, text=True, capture_output=True
    )
    return completed.stdout.strip()


def snapshot(args: argparse.Namespace) -> None:
    repository_root: Path = args.repository_root.resolve()
    label_root = repository_root / SCRATCH_ROOT / args.label
    steps = set(args.only) if args.only else set(STEPS)
    if args.skip_tests:
        steps -= {"tests", "real_data", "ruff"}
    if label_root.exists():
        if not args.force:
            raise SystemExit(
                f"{label_root} exists; pass --force to rebuild it (the `before` snapshot is the "
                "pre-refactor reference and should normally be kept)"
            )
        for name in ("consensus", "scoreboard", "review_v4"):
            if name in steps or (name == "review_v4" and "review" in steps):
                shutil.rmtree(label_root / name, ignore_errors=True)
    label_root.mkdir(parents=True, exist_ok=True)
    summary_path = label_root / "summary.json"
    # A partial `--only` rebuild keeps the other steps' records.
    previous_steps: dict[str, Any] = {}
    if summary_path.is_file():
        previous_steps = json.loads(summary_path.read_text(encoding="utf-8")).get("steps", {})
    summary: dict[str, Any] = {
        "label": args.label,
        "git_head": _git_head(repository_root),
        "started_at": datetime.now(UTC).isoformat(),
        "steps": previous_steps,
    }
    if "consensus" in steps:
        summary["steps"]["consensus"] = snapshot_consensus(repository_root, label_root)
    if "scoreboard" in steps:
        summary["steps"]["scoreboard"] = snapshot_scoreboard(repository_root, label_root)
    if "review" in steps:
        summary["steps"]["review"] = snapshot_review(repository_root, label_root)
    if "tests" in steps:
        summary["steps"]["pytest_default"] = snapshot_pytest(
            repository_root, label_root, marker=None
        )
    if "real_data" in steps:
        summary["steps"]["pytest_real_data"] = snapshot_pytest(
            repository_root, label_root, marker="real_data"
        )
    if "ruff" in steps:
        summary["steps"]["ruff"] = snapshot_ruff(repository_root, label_root)
    summary["finished_at"] = datetime.now(UTC).isoformat()
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


# --------------------------------------------------------------------------- compare


Replacements = Sequence[tuple[str, str]]


def is_volatile_key(key: str, value: Any, extra_keys: frozenset[str] = frozenset()) -> bool:
    if key in VOLATILE_KEYS or key in extra_keys:
        return True
    if key.endswith(VOLATILE_KEY_SUFFIXES):
        return True
    if key == "comparison_id" and isinstance(value, str) and TIMESTAMP_PATTERN.search(value):
        return True
    return False


def normalise_text(text: str, labels: Iterable[str], replacements: Replacements = ()) -> str:
    """Replace `runs/dedup-equivalence/<label>` (whole path component) with a placeholder.

    `replacements` are literal `(old, new)` substitutions applied afterwards, longest `old`
    first; `compare-runs` uses them to fold two run directories onto one placeholder.
    """
    for old, new in sorted(replacements, key=lambda item: len(item[0]), reverse=True):
        if old:
            text = text.replace(old, new)
    names = sorted({label for label in labels if label}, key=len, reverse=True)
    if not names:
        return text
    pattern = re.compile(
        re.escape(SCRATCH_ROOT.as_posix()) + "/(?:" + "|".join(map(re.escape, names)) + r")(?=/|$)",
        re.MULTILINE,
    )
    return pattern.sub(SCRATCH_PLACEHOLDER, text)


def _in_scratch(uri: str, labels: Iterable[str]) -> bool:
    return normalise_text(uri, labels) != uri


def scratch_fingerprint_mismatches(root: Path, repository_root: Path, label: str) -> list[str]:
    """Check every `{uri, sha256}` that points into the scratch root against the file's bytes.

    Those digests legitimately differ between labels when the file embeds its own path (the
    review guide names the recording's absolute path), so `strip_volatile` neutralises them;
    this check keeps the builder honest by proving each one still matches its file.
    """
    import hashlib

    mismatches: list[str] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            uri = value.get("uri")
            digest = value.get("sha256")
            if isinstance(uri, str) and isinstance(digest, str) and _in_scratch(uri, (label,)):
                target = Path(uri) if Path(uri).is_absolute() else repository_root / uri
                if not target.is_file():
                    mismatches.append(f"{path}: {uri} missing")
                elif hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                    mismatches.append(f"{path}: sha256 does not match {uri}")
            for key, item in value.items():
                visit(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")

    for file in sorted(root.rglob("*.json")):
        if file.name in SKIP_NAMES:
            continue
        try:
            document = json.loads(file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        visit(document, file.relative_to(root).as_posix())
    return mismatches


def strip_volatile(
    value: Any,
    *,
    labels: Iterable[str] = (),
    drop_paths: frozenset[str] = frozenset(),
    extra_keys: frozenset[str] = frozenset(),
    replacements: Replacements = (),
    _path: str = "",
) -> Any:
    """Return `value` without volatile keys and with scratch labels normalised in strings.

    `drop_paths` names dotted JSON paths (e.g. ``output_rrd.sha256``) to drop as well; the
    caller uses it when the Rerun recording has been shown to differ only in its header.
    `extra_keys` are further volatile keys and `replacements` further literal string
    substitutions (see :func:`normalise_text`), both used by `compare-runs`.
    """
    labels = tuple(labels)
    options = {
        "labels": labels,
        "drop_paths": drop_paths,
        "extra_keys": extra_keys,
        "replacements": replacements,
    }
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        uri = value.get("uri")
        scratch_fingerprint = isinstance(uri, str) and _in_scratch(uri, labels)
        for key, item in value.items():
            path = f"{_path}.{key}" if _path else key
            if is_volatile_key(key, item, extra_keys) or path in drop_paths:
                continue
            if key == "sha256" and scratch_fingerprint:
                # The file itself is compared by the tree walk and the digest is checked
                # against the file by `scratch_fingerprint_mismatches`.
                out[key] = "<scratch-file>"
                continue
            out[key] = strip_volatile(item, _path=path, **options)
        return out
    if isinstance(value, list):
        return [strip_volatile(item, _path=f"{_path}[]", **options) for item in value]
    if isinstance(value, str):
        return normalise_text(value, labels, replacements)
    return value


def diff_json(left: Any, right: Any, path: str = "$") -> list[str]:
    """Return the JSON pointers at which two stripped documents differ (first level only)."""
    if type(left) is not type(right):
        return [f"{path}: {type(left).__name__} != {type(right).__name__}"]
    if isinstance(left, dict):
        diffs: list[str] = []
        for key in sorted(set(left) | set(right)):
            if key not in left:
                diffs.append(f"{path}.{key}: only in after")
            elif key not in right:
                diffs.append(f"{path}.{key}: only in before")
            else:
                diffs.extend(diff_json(left[key], right[key], f"{path}.{key}"))
        return diffs
    if isinstance(left, list):
        if len(left) != len(right):
            return [f"{path}: length {len(left)} != {len(right)}"]
        diffs = []
        for index, (a, b) in enumerate(zip(left, right)):
            diffs.extend(diff_json(a, b, f"{path}[{index}]"))
            if len(diffs) >= MAX_REPORTED_DIFFS:
                break
        return diffs
    if left != right:
        return [f"{path}: {left!r} != {right!r}"]
    return []


def _load_jsonl(path: Path) -> list[Any]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def compare_json_files(
    before: Path,
    after: Path,
    labels: Sequence[str],
    drop_paths: frozenset[str],
    *,
    extra_keys: frozenset[str] = frozenset(),
    replacements: Replacements = (),
) -> list[str]:
    if before.suffix == ".jsonl":
        left: Any = _load_jsonl(before)
        right: Any = _load_jsonl(after)
    else:
        left = json.loads(before.read_text(encoding="utf-8"))
        right = json.loads(after.read_text(encoding="utf-8"))
    options = {
        "labels": labels,
        "drop_paths": drop_paths,
        "extra_keys": extra_keys,
        "replacements": replacements,
    }
    return diff_json(strip_volatile(left, **options), strip_volatile(right, **options))


def compare_text_files(
    before: Path, after: Path, labels: Sequence[str], replacements: Replacements = ()
) -> list[str]:
    left = normalise_text(before.read_text(encoding="utf-8"), labels, replacements).splitlines()
    right = normalise_text(after.read_text(encoding="utf-8"), labels, replacements).splitlines()
    if left == right:
        return []
    diffs = []
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            diffs.append(f"line {index + 1}: {a[:80]!r} != {b[:80]!r}")
            if len(diffs) >= MAX_REPORTED_DIFFS:
                break
    if len(left) != len(right):
        diffs.append(f"line count {len(left)} != {len(right)}")
    return diffs


def compare_npz_files(before: Path, after: Path) -> list[str]:
    import numpy as np

    left = np.load(before)
    right = np.load(after)
    diffs = []
    for key in sorted(set(left.files) | set(right.files)):
        if key not in left.files or key not in right.files:
            diffs.append(f"array {key}: present on one side only")
            continue
        a, b = left[key], right[key]
        if a.shape != b.shape or a.dtype != b.dtype:
            diffs.append(f"array {key}: shape/dtype {a.shape}/{a.dtype} != {b.shape}/{b.dtype}")
        elif a.dtype.kind in "fc":
            if not np.array_equal(a, b, equal_nan=True):
                diffs.append(f"array {key}: values differ")
        elif not np.array_equal(a, b):
            diffs.append(f"array {key}: values differ")
    return diffs


def compare_bytes(before: Path, after: Path) -> list[str]:
    if before.read_bytes() == after.read_bytes():
        return []
    return [f"bytes differ ({before.stat().st_size} vs {after.stat().st_size} bytes)"]


def compare_rrd(
    before: Path, after: Path, labels: Sequence[str], replacements: Replacements = ()
) -> tuple[list[str], str]:
    """Compare a recording by bytes, falling back to the chunk-independent content digest."""
    if before.read_bytes() == after.read_bytes():
        return [], "bytes"
    left = rrd_content_digest(before, labels=labels, replacements=replacements)
    right = rrd_content_digest(after, labels=labels, replacements=replacements)
    diffs = []
    for key in sorted(set(left) | set(right)):
        if left.get(key) != right.get(key):
            diffs.append(f"{key}: {left.get(key)} != {right.get(key)}")
    return diffs, "content digest"


def _relative_files(root: Path) -> set[Path]:
    return {p.relative_to(root) for p in root.rglob("*") if p.is_file()}


SKIP_NAMES = {"summary.json"}
SKIP_SUFFIXES = {".log"}


def compare_trees(
    before_root: Path,
    after_root: Path,
    *,
    repository_root: Path,
    labels: Sequence[str],
    extra_keys: frozenset[str] = frozenset(),
    replacements: Replacements = (),
    fingerprint_self_check: bool = True,
) -> tuple[list[str], dict[str, Any]]:
    before_files = _relative_files(before_root)
    after_files = _relative_files(after_root)
    diffs: list[str] = []
    notes: dict[str, Any] = {"compared": 0, "rrd_method": None}
    for relative in sorted(before_files ^ after_files):
        if relative.name in SKIP_NAMES or relative.suffix in SKIP_SUFFIXES:
            continue
        side = "before" if relative in before_files else "after"
        diffs.append(f"{relative}: only in {side}")
    drop_paths: frozenset[str] = frozenset()
    shared = sorted(before_files & after_files)
    # Recordings first: whether their bytes match decides how their digests are compared.
    rrds = [p for p in shared if p.suffix == ".rrd"]
    for relative in rrds:
        result, method = compare_rrd(
            before_root / relative, after_root / relative, labels, replacements
        )
        notes["rrd_method"] = method
        notes["compared"] += 1
        if method != "bytes":
            notes["rrd_note"] = (
                "rrd bytes differ between snapshots (start time, log_time, chunk batching, "
                "blueprint UUIDs); compared by per-entity content digest; its sha256 in the "
                "index is a scratch-file fingerprint, checked against the file per side"
            )
        diffs.extend(f"{relative}: {d}" for d in result)
    if fingerprint_self_check:
        for side, root in (("before", before_root), ("after", after_root)):
            label = root.name
            for mismatch in scratch_fingerprint_mismatches(root, repository_root, label):
                diffs.append(f"{side} fingerprint self-check: {mismatch}")
    for relative in shared:
        if (
            relative.name in SKIP_NAMES
            or relative.suffix in SKIP_SUFFIXES
            or relative.suffix == ".rrd"
        ):
            continue
        before, after = before_root / relative, after_root / relative
        if relative.suffix in {".json", ".jsonl"}:
            result = compare_json_files(
                before,
                after,
                labels,
                drop_paths,
                extra_keys=extra_keys,
                replacements=replacements,
            )
        elif relative.suffix in {".md", ".txt"}:
            result = compare_text_files(before, after, labels, replacements)
        elif relative.suffix == ".npz":
            result = compare_npz_files(before, after)
        else:
            result = compare_bytes(before, after)
        notes["compared"] += 1
        diffs.extend(f"{relative}: {d}" for d in result)
    return diffs, notes


def _summary_counts(root: Path) -> dict[str, Any]:
    path = root / "summary.json"
    if not path.is_file():
        return {}
    summary = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, Any] = {"git_head": summary.get("git_head")}
    for step, value in summary.get("steps", {}).items():
        if step.startswith("pytest"):
            out[step] = {"exit_code": value.get("exit_code"), **value.get("counts", {})}
        elif step == "ruff":
            out[step] = value.get("ok")
        elif step == "review":
            out["rrd_verify"] = value.get("rrd_verify")
    return out


def compare(args: argparse.Namespace) -> None:
    repository_root: Path = args.repository_root.resolve()
    before_root = repository_root / SCRATCH_ROOT / args.before
    after_root = repository_root / SCRATCH_ROOT / args.after
    for root in (before_root, after_root):
        if not root.is_dir():
            raise SystemExit(f"missing snapshot {root}; run `snapshot --label {root.name}` first")
    diffs, notes = compare_trees(
        before_root, after_root, repository_root=repository_root, labels=(args.before, args.after)
    )
    print(f"before: {_summary_counts(before_root)}")
    print(f"after:  {_summary_counts(after_root)}")
    print(f"compared {notes['compared']} files; rrd compared by: {notes['rrd_method']}")
    if notes.get("rrd_note"):
        print(notes["rrd_note"])
    if diffs:
        print(f"NON-VOLATILE DIFFERENCES: {len(diffs)} (first {MAX_REPORTED_DIFFS})")
        for line in diffs[:MAX_REPORTED_DIFFS]:
            print(f"  {line}")
        raise SystemExit(1)
    print("EQUIVALENT: no non-volatile differences")


# --------------------------------------------------------------------------- compare-runs

# Per-run measurements that legitimately differ between two executions of the same code.
RUN_VOLATILE_KEYS = frozenset({"time_to_first_usable_output_seconds", "gpu_peak_vram_bytes"})
RUN_PLACEHOLDER = "<run>"
MASK_DIRECTORIES = ("native/masks", "masks")


def run_directory_replacements(
    run_directories: Sequence[Path], repository_root: Path
) -> list[tuple[str, str]]:
    """Substitutions folding each run directory (absolute and repository-relative) onto `<run>`.

    The run id is the directory name, so a URI such as `runs/x/<run-id>/observations.jsonl`
    and the `run_id` field both normalise, whichever side of the comparison they come from.
    """
    replacements: list[tuple[str, str]] = []
    for directory in run_directories:
        resolved = directory.resolve()
        replacements.append((resolved.as_posix(), RUN_PLACEHOLDER))
        try:
            replacements.append(
                (resolved.relative_to(repository_root.resolve()).as_posix(), RUN_PLACEHOLDER)
            )
        except ValueError:
            pass
        replacements.append((directory.name, RUN_PLACEHOLDER))
    return replacements


def compare_masks(before_root: Path, after_root: Path) -> dict[str, Any]:
    """Every PNG under the mask directories, by SHA-256; identical means equal names and bytes."""
    import hashlib

    def digests(root: Path) -> dict[str, str]:
        out: dict[str, str] = {}
        for directory in MASK_DIRECTORIES:
            base = root / directory
            if base.is_dir():
                for png in sorted(base.rglob("*.png")):
                    out[png.relative_to(root).as_posix()] = hashlib.sha256(
                        png.read_bytes()
                    ).hexdigest()
        return out

    left, right = digests(before_root), digests(after_root)
    differing = sorted(name for name in set(left) | set(right) if left.get(name) != right.get(name))
    return {
        "before_count": len(left),
        "after_count": len(right),
        "identical": not differing and len(left) == len(right),
        "differing": differing,
    }


def compare_runs(args: argparse.Namespace) -> None:
    """Compare two run directories of one driver: masks byte for byte, the rest as `compare`."""
    repository_root: Path = args.repository_root.resolve()
    before_root: Path = args.before.resolve()
    after_root: Path = args.after.resolve()
    for root in (before_root, after_root):
        if not root.is_dir():
            raise SystemExit(f"missing run directory {root}")
    replacements = run_directory_replacements([before_root, after_root], repository_root)
    for item in args.replace:
        old, separator, new = item.partition("=")
        if not separator or not old:
            raise SystemExit(f"--replace expects OLD=NEW, got {item!r}")
        replacements.append((old, new))
    masks = compare_masks(before_root, after_root)
    diffs, notes = compare_trees(
        before_root,
        after_root,
        repository_root=repository_root,
        labels=(),
        extra_keys=RUN_VOLATILE_KEYS,
        replacements=replacements,
        fingerprint_self_check=False,
    )
    manifest_diffs = [line for line in diffs if line.startswith("manifest.json:")]
    other_diffs = [line for line in diffs if not line.startswith("manifest.json:")]
    report = {
        "before": before_root.relative_to(repository_root).as_posix()
        if before_root.is_relative_to(repository_root)
        else str(before_root),
        "after": after_root.relative_to(repository_root).as_posix()
        if after_root.is_relative_to(repository_root)
        else str(after_root),
        "masks": masks,
        "manifest_equal_modulo_volatile": not manifest_diffs,
        "manifest_differences": manifest_diffs[:MAX_REPORTED_DIFFS],
        "other_differences": other_diffs[:MAX_REPORTED_DIFFS],
        "files_compared": notes["compared"],
        "rrd_method": notes["rrd_method"],
        "volatile_keys": sorted(VOLATILE_KEYS | RUN_VOLATILE_KEYS),
        "equivalent": not diffs and masks["identical"],
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"masks: {masks['before_count']} before / {masks['after_count']} after, "
        f"identical: {'yes' if masks['identical'] else 'NO'}"
    )
    print(f"manifest equal modulo volatile fields: {'yes' if not manifest_diffs else 'NO'}")
    print(f"compared {notes['compared']} files; rrd compared by: {notes['rrd_method']}")
    if masks["differing"]:
        print(f"DIFFERING MASKS: {len(masks['differing'])} (first {MAX_REPORTED_DIFFS})")
        for name in masks["differing"][:MAX_REPORTED_DIFFS]:
            print(f"  {name}")
    if diffs:
        print(f"NON-VOLATILE DIFFERENCES: {len(diffs)} (first {MAX_REPORTED_DIFFS})")
        for line in diffs[:MAX_REPORTED_DIFFS]:
            print(f"  {line}")
    if diffs or not masks["identical"]:
        raise SystemExit(1)
    print("EQUIVALENT: masks identical, no non-volatile differences")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="command", required=True)
    snap = sub.add_parser("snapshot", help="rebuild the three artifacts and record test status")
    snap.add_argument("--label", required=True, help="before | after | any scratch name")
    snap.add_argument("--only", nargs="*", choices=STEPS, help="run only these steps")
    snap.add_argument("--skip-tests", action="store_true", help="skip pytest and ruff")
    snap.add_argument("--force", action="store_true", help="rebuild into an existing label")
    snap.set_defaults(func=snapshot)
    comp = sub.add_parser("compare", help="deep-compare two snapshots ignoring volatile fields")
    comp.add_argument("--before", default="before")
    comp.add_argument("--after", default="after")
    comp.set_defaults(func=compare)
    runs = sub.add_parser(
        "compare-runs",
        help=(
            "compare two run directories of one driver: masks byte for byte, manifest and the "
            "other files modulo volatile fields with both run directories folded onto <run>"
        ),
    )
    runs.add_argument("before", type=Path)
    runs.add_argument("after", type=Path)
    runs.add_argument("--report", type=Path, default=None, help="write the comparison as JSON")
    runs.add_argument(
        "--replace",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help=(
            "further literal substitution applied to strings before comparing (repeatable), "
            "e.g. the two code-snapshot directories named in worker_command.txt"
        ),
    )
    runs.set_defaults(func=compare_runs)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
