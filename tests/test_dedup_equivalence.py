"""The dedup equivalence harness: volatile-key stripping and the tree diff on a fixture."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "dedup_equivalence.py"
_spec = importlib.util.spec_from_file_location("dedup_equivalence", MODULE_PATH)
assert _spec and _spec.loader
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)

LABELS = ("before", "after")


def test_strip_volatile_drops_named_keys_and_at_suffixes_and_normalises_labels() -> None:
    document = {
        "generated_at": "2026-09-22T00:00:00Z",
        "runtime_seconds": 73.7,
        "elapsed_seconds": 1,
        "duration_s": 2,
        "created_at": "x",
        "run_id": "abc",
        "exported_at": "2026-09-22T00:00:00Z",
        "nested": {"prepared_at": "x", "kept": 1, "uri": "runs/dedup-equivalence/before/a.json"},
        "list": [{"reviewed_at": "y", "value": 2}],
        "sha256": "deadbeef",
    }

    stripped = harness.strip_volatile(document, labels=LABELS)

    assert stripped == {
        "nested": {"kept": 1, "uri": "runs/dedup-equivalence/<label>/a.json"},
        "list": [{"value": 2}],
        "sha256": "deadbeef",
    }


def test_normalise_text_matches_whole_label_components_only() -> None:
    labels = ("before", "before-rrd-check")
    text = (
        "runs/dedup-equivalence/before/x runs/dedup-equivalence/before-rrd-check/y "
        "runs/dedup-equivalence/beforehand/z runs/dedup-equivalence/before"
    )

    assert harness.normalise_text(text, labels) == (
        "runs/dedup-equivalence/<label>/x runs/dedup-equivalence/<label>/y "
        "runs/dedup-equivalence/beforehand/z runs/dedup-equivalence/<label>"
    )


def test_scratch_file_fingerprints_compare_as_placeholders_but_other_digests_stay() -> None:
    document = {
        "review_guide": {"uri": "runs/dedup-equivalence/before/review_v4/guide.md", "sha256": "1"},
        "anchor_set": {"uri": "runs/human-review-anchors/anchor_masks.json", "sha256": "2"},
    }

    stripped = harness.strip_volatile(document, labels=LABELS)

    assert stripped["review_guide"] == {
        "uri": "runs/dedup-equivalence/<label>/review_v4/guide.md",
        "sha256": "<scratch-file>",
    }
    assert stripped["anchor_set"] == {
        "uri": "runs/human-review-anchors/anchor_masks.json",
        "sha256": "2",
    }


def test_scratch_fingerprint_self_check_reports_a_stale_or_missing_digest(tmp_path: Path) -> None:
    import hashlib

    root = tmp_path / "runs" / "dedup-equivalence" / "before"
    (root / "review_v4").mkdir(parents=True)
    guide = root / "review_v4" / "guide.md"
    guide.write_text("guide\n", encoding="utf-8")
    good = hashlib.sha256(guide.read_bytes()).hexdigest()
    (root / "review_v4" / "index.json").write_text(
        json.dumps(
            {
                "review_guide": {
                    "uri": "runs/dedup-equivalence/before/review_v4/guide.md",
                    "sha256": good,
                },
                "stale": {"uri": "runs/dedup-equivalence/before/review_v4/guide.md", "sha256": "0"},
                "gone": {"uri": "runs/dedup-equivalence/before/review_v4/nope.md", "sha256": "0"},
                "elsewhere": {"uri": "configs/x.json", "sha256": "0"},
            }
        ),
        encoding="utf-8",
    )

    mismatches = harness.scratch_fingerprint_mismatches(root, tmp_path, "before")

    assert mismatches == [
        "review_v4/index.json.stale: sha256 does not match "
        "runs/dedup-equivalence/before/review_v4/guide.md",
        "review_v4/index.json.gone: runs/dedup-equivalence/before/review_v4/nope.md missing",
    ]


def test_comparison_id_is_dropped_only_when_it_embeds_a_timestamp() -> None:
    fixed = harness.strip_volatile({"comparison_id": "interaction_review_first_minute_v4"})
    stamped = harness.strip_volatile({"comparison_id": "compare-20260922t010203z"})
    iso = harness.strip_volatile({"comparison_id": "run 2026-09-22T01:02:03"})

    assert fixed == {"comparison_id": "interaction_review_first_minute_v4"}
    assert stamped == {}
    assert iso == {}


def test_drop_paths_removes_a_dotted_field_but_keeps_its_siblings() -> None:
    document = {"output_rrd": {"uri": "a.rrd", "sha256": "1"}, "other": {"sha256": "2"}}

    stripped = harness.strip_volatile(document, drop_paths=frozenset({"output_rrd.sha256"}))

    assert stripped == {"output_rrd": {"uri": "a.rrd"}, "other": {"sha256": "2"}}


def test_diff_json_reports_the_pointer_of_each_difference() -> None:
    left = {"a": 1, "b": [1, 2, {"c": "x"}], "only_left": 0}
    right = {"a": 2, "b": [1, 2, {"c": "y"}], "only_right": 0}

    diffs = harness.diff_json(left, right)

    assert diffs == [
        "$.a: 1 != 2",
        "$.b[2].c: 'x' != 'y'",
        "$.only_left: only in before",
        "$.only_right: only in after",
    ]
    assert harness.diff_json([1, 2], [1]) == ["$: length 2 != 1"]
    assert harness.diff_json({"a": 1}, {"a": 1}) == []


def _write_tree(root: Path, *, label: str, value: int, sha: str, stamp: str) -> None:
    (root / "consensus").mkdir(parents=True)
    (root / "consensus" / "manifest.json").write_text(
        json.dumps(
            {
                "runtime_seconds": 10.0 if label == "before" else 99.0,
                "generated_at": stamp,
                "per_frame": {"uri": f"runs/dedup-equivalence/{label}/consensus/per_frame.jsonl"},
                "value": value,
                "digest": sha,
            }
        ),
        encoding="utf-8",
    )
    (root / "consensus" / "per_frame.jsonl").write_text(
        "\n".join(json.dumps({"frame": i, "created_at": stamp}) for i in range(3)) + "\n",
        encoding="utf-8",
    )
    (root / "consensus" / "summary.md").write_text(
        f"# Built into runs/dedup-equivalence/{label}/consensus\n\nvalue {value}\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        root / "consensus" / "points.npz", a=np.array([1.0, np.nan]), b=np.arange(3)
    )
    (root / "consensus" / "sheet.png").write_bytes(b"\x89PNG" + bytes([value]))
    (root / "consensus.build.log").write_text(f"took {value} s\n", encoding="utf-8")
    (root / "summary.json").write_text(json.dumps({"label": label}), encoding="utf-8")


def test_compare_trees_treats_timestamps_labels_and_logs_as_volatile(tmp_path: Path) -> None:
    before, after = tmp_path / "before", tmp_path / "after"
    _write_tree(before, label="before", value=7, sha="abc", stamp="2026-09-22T00:00:00Z")
    _write_tree(after, label="after", value=7, sha="abc", stamp="2026-09-22T01:00:00Z")

    diffs, notes = harness.compare_trees(before, after, repository_root=tmp_path, labels=LABELS)

    assert diffs == []
    assert notes["compared"] == 5  # manifest, per_frame, summary.md, points.npz, sheet.png
    assert notes["rrd_method"] is None


def test_compare_trees_reports_a_changed_digest_a_changed_table_and_changed_bytes(
    tmp_path: Path,
) -> None:
    before, after = tmp_path / "before", tmp_path / "after"
    _write_tree(before, label="before", value=7, sha="abc", stamp="t0")
    _write_tree(after, label="after", value=8, sha="abd", stamp="t1")

    diffs, _ = harness.compare_trees(before, after, repository_root=tmp_path, labels=LABELS)

    joined = "\n".join(diffs)
    assert "consensus/manifest.json: $.digest: 'abc' != 'abd'" in joined
    assert "consensus/manifest.json: $.value: 7 != 8" in joined
    assert "consensus/summary.md: line 3" in joined
    assert "consensus/sheet.png: bytes differ" in joined
    assert "per_frame.jsonl" not in joined
    assert "points.npz" not in joined
    assert ".log" not in joined


def test_compare_trees_reports_files_present_on_one_side_and_npz_array_changes(
    tmp_path: Path,
) -> None:
    before, after = tmp_path / "before", tmp_path / "after"
    _write_tree(before, label="before", value=7, sha="abc", stamp="t0")
    _write_tree(after, label="after", value=7, sha="abc", stamp="t0")
    (after / "consensus" / "extra.json").write_text("{}", encoding="utf-8")
    np.savez_compressed(after / "consensus" / "points.npz", a=np.array([1.0, 2.0]), b=np.arange(3))

    diffs, _ = harness.compare_trees(before, after, repository_root=tmp_path, labels=LABELS)

    assert "consensus/extra.json: only in after" in diffs
    assert "consensus/points.npz: array a: values differ" in diffs
    assert len(diffs) == 2


def _write_rrd(path: Path, *, text: str, chunked: bool) -> None:
    import rerun as rr

    stream = rr.RecordingStream("dedup-test", recording_id="fixed")
    stream.save(str(path))
    stream.log("notes", rr.TextDocument(text), static=True)
    for frame in range(6):
        stream.set_time("frame", sequence=frame)
        stream.log("series/value", rr.Scalars([float(frame) * 0.5]))
        if chunked and frame == 2:
            stream.flush()
    stream.disconnect()


def test_rrd_content_digest_ignores_log_time_chunking_and_scratch_labels(tmp_path: Path) -> None:
    pytest.importorskip("rerun")
    left, right = tmp_path / "left.rrd", tmp_path / "right.rrd"
    _write_rrd(left, text="see runs/dedup-equivalence/before/x", chunked=False)
    _write_rrd(right, text="see runs/dedup-equivalence/after/x", chunked=True)
    assert left.read_bytes() != right.read_bytes()

    digest_left = harness.rrd_content_digest(left, labels=LABELS)
    digest_right = harness.rrd_content_digest(right, labels=LABELS)

    assert digest_left == digest_right
    assert digest_left["recording:/series/value#Scalars:scalars"].endswith("/6")
    assert "recording:/notes#TextDocument:text" in digest_left
    assert not any(key.endswith("#log_time") for key in digest_left)

    diffs, method = harness.compare_rrd(left, right, LABELS)
    assert diffs == []
    assert method == "content digest"


def test_rrd_content_digest_reports_a_changed_value(tmp_path: Path) -> None:
    pytest.importorskip("rerun")
    left, right = tmp_path / "left.rrd", tmp_path / "right.rrd"
    _write_rrd(left, text="a", chunked=False)
    _write_rrd(right, text="b", chunked=False)

    diffs, _ = harness.compare_rrd(left, right, LABELS)

    assert len(diffs) == 1
    assert diffs[0].startswith("recording:/notes#TextDocument:text:")


def test_pytest_counts_reads_the_final_summary_line() -> None:
    output = "....\n646 passed, 12 skipped, 3 deselected in 40.12s\n"

    assert harness._pytest_counts(output) == {"passed": 646, "skipped": 12, "deselected": 3}
    assert harness._pytest_counts("2 failed, 1 passed, 1 error in 1s") == {
        "failed": 2,
        "passed": 1,
        "error": 1,
    }


@pytest.mark.parametrize(
    ("key", "value", "volatile"),
    [
        ("generated_at", "x", True),
        ("exported_at", "x", True),
        ("run_id", "x", True),
        ("sha256", "x", False),
        ("uri", "x", False),
        ("frame_count", 1800, False),
    ],
)
def test_is_volatile_key(key: str, value: object, volatile: bool) -> None:
    assert harness.is_volatile_key(key, value) is volatile
