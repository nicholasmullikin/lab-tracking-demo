"""Blueprint presets (`.rbl`) for the first-minute review package, and a check that every
view they declare points at entities the recording holds.

The human asked for one recording with a few layouts that switch every panel at once:
`segmentation.rbl` (reference beside the candidate arms, provenance, consensus contradiction,
detector confidence, anchor marks), `hands.rbl` (every hand source, no masks) and
`multiview.rbl` (the camera tiles with masks, the world-mm rig with hull voxels, the per-view
error series, the episode document).  Nothing here logs data; a preset only chooses which
logged entities are shown.  `battle-review-presets` reads the entity tree of the recording,
writes the three presets with the recording's application id, and validates each one by
reading it back and walking its view contents against that tree (`$origin` and `/**` globs
resolved; excluded expressions ignored).  A viewer was not opened by the agent.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import rerun.blueprint as rrb

from . import interaction_review as review
from . import multiview_review as multiview
from .four_part_contract import ANALYSIS_FPS, TARGETS
from .rerun_logging import time_series_view

PRESET_NAMES = ("segmentation", "hands", "multiview")
PRESET_CHECK_NAME = "presets_check.json"
COMBINED_RECORDING_NAME = "interaction_review_combined.rrd"
DIMENSIONS = review.DIMENSIONS
REVIEW_ROOT_SUFFIX = "/interaction_review_v4"


# -- recording facts -----------------------------------------------------------------------------


@dataclass(frozen=True)
class RecordingFacts:
    """What a review recording holds, read from its entity tree (no data is decoded)."""

    path: Path
    application_id: str
    recording_id: str
    entity_paths: tuple[str, ...]
    review_root: str | None
    multiview_root: str | None
    candidate_arms: tuple[str, ...] = ()
    multiview_views: tuple[str, ...] = ()
    athena_labels: tuple[str, ...] = ()
    flags: frozenset[str] = field(default_factory=frozenset)

    def has(self, relative: str, root: str | None = None) -> bool:
        base = self.review_root if root is None else root
        return base is not None and f"{base}/{relative}" in self.entity_paths

    def has_subtree(self, relative: str, root: str | None = None) -> bool:
        base = self.review_root if root is None else root
        if base is None:
            return False
        return query_matches(f"{base}/{relative}/**", self.entity_paths)

    def children(self, parent: str) -> tuple[str, ...]:
        prefix = parent.rstrip("/") + "/"
        names = {
            path[len(prefix) :].split("/", 1)[0]
            for path in self.entity_paths
            if path.startswith(prefix)
        }
        return tuple(sorted(names))


def read_recording_facts(rrd_path: Path) -> RecordingFacts:
    import rerun.experimental as experimental

    reader = experimental.RrdReader(str(rrd_path))
    recordings = reader.recordings()
    if len(recordings) != 1:
        raise ValueError(
            f"{rrd_path} holds {len(recordings)} recording stores; presets need exactly one"
        )
    store = recordings[0]
    entity_paths = tuple(
        sorted(str(path) for path in reader.store(store=store).schema().entity_paths())
    )
    review_root = next((path for path in entity_paths if path.endswith(REVIEW_ROOT_SUFFIX)), None)
    if review_root is None:
        candidates = {
            path[: path.index(REVIEW_ROOT_SUFFIX) + len(REVIEW_ROOT_SUFFIX)]
            for path in entity_paths
            if REVIEW_ROOT_SUFFIX + "/" in path
        }
        review_root = sorted(candidates)[0] if candidates else None
    multiview_root = (
        f"/{multiview.ENTITY_ROOT}"
        if any(path.startswith(f"/{multiview.ENTITY_ROOT}/") for path in entity_paths)
        else None
    )
    facts = RecordingFacts(
        path=rrd_path,
        application_id=str(store.application_id),
        recording_id=str(store.recording_id),
        entity_paths=entity_paths,
        review_root=review_root,
        multiview_root=multiview_root,
    )
    arms = (
        facts.children(f"{review_root}/{review.CANDIDATE_SEGMENTATION_ROOT}")
        if review_root is not None
        else ()
    )
    views = facts.children(f"{multiview_root}/views") if multiview_root is not None else ()
    athena = tuple(
        name
        for name in (
            facts.children(f"{review_root}/{review.ASSEMBLY101_3D_ROOT}")
            if review_root is not None
            else ()
        )
        if name.startswith("athena_hands")
    )
    flags = set()
    if review_root is not None:
        checks = {
            "provenance": f"{review.REFERENCE_PROVENANCE_SERIES}/chassis",
            "provenance_overlay": f"{review.REFERENCE_PROVENANCE_OVERLAY}/chassis",
            "anchor_outlines": f"{review.HUMAN_ANCHOR_OUTLINES}/chassis",
            "anchors": f"{review.ANCHOR_SERIES}/anchor_frame",
            "anchor_log": f"{review.ANCHOR_SERIES}/log",
            "confidence": f"{review.CONFIDENCE_SERIES}/chassis/confidence",
            "consensus": f"{review.MULTIVIEW_DIAGNOSTICS}/chassis/c10379_error_px",
            "multiview_document": multiview.V4_TEXT_PANEL[0],
            "assembly101_2d": f"{review.ASSEMBLY101_2D_ROOT}/skeletons",
            "assembly101_3d": f"{review.ASSEMBLY101_3D_ROOT}/hands/joints",
            "assembly101_series": f"{review.ASSEMBLY101_DIAGNOSTICS}/confidence/left",
            "wilor_3d": "contexts/wilor_camera_relative_non_metric_3d/camera_relative_3d/joints",
            "navigation": review.NAVIGATION_CURRENT,
            "fine_gt_index": review.NAVIGATION_FINE_GT_INDEX,
            "review_notes": review.REVIEW_NOTES,
        }
        flags.update(name for name, relative in checks.items() if facts.has(relative))
    if multiview_root is not None:
        if any(f"/hull/{target}" in path for path in entity_paths for target in TARGETS):
            flags.add("hull_voxels")
        if any("/hull_projection/" in path for path in entity_paths):
            flags.add("hull_projection")
        if facts.has(f"{multiview.ANCHOR_SERIES}/anchor_frame", multiview_root):
            flags.add("multiview_anchors")
        flags.update(
            f"anchor_view:{view}"
            for view in views
            if facts.has_subtree(f"views/{view}/human_anchor_outlines", multiview_root)
        )
        flags.update(
            f"proposal_view:{view}"
            for view in views
            if facts.has_subtree(f"views/{view}/proposals", multiview_root)
        )
    return RecordingFacts(
        path=rrd_path,
        application_id=facts.application_id,
        recording_id=facts.recording_id,
        entity_paths=entity_paths,
        review_root=review_root,
        multiview_root=multiview_root,
        candidate_arms=arms,
        multiview_views=views,
        athena_labels=athena,
        flags=frozenset(flags),
    )


# -- the three presets ---------------------------------------------------------------------------


def _bounds() -> rrb.VisualBounds2D:
    return rrb.VisualBounds2D(x_range=[0, DIMENSIONS[0]], y_range=[0, DIMENSIONS[1]])


def _time_panel() -> rrb.TimePanel:
    return rrb.TimePanel(timeline="analysis_time", fps=ANALYSIS_FPS)


def _require_review_root(facts: RecordingFacts) -> str:
    if facts.review_root is None:
        raise ValueError(f"{facts.path} holds no interaction-review entities")
    return facts.review_root.lstrip("/")


def segmentation_blueprint(facts: RecordingFacts) -> rrb.Blueprint:
    """Reference beside the candidate arms, with every segmentation-quality series."""
    root = _require_review_root(facts)
    primary = rrb.Spatial2DView(
        origin=root,
        name="Reference: ensemble v2 (sam3 primary / dam4sam fallback) + human anchor outlines",
        contents=(
            "$origin/source/video",
            "$origin/primary/reference_four_part_segmentation/**",
            *(
                (f"$origin/{review.REFERENCE_PROVENANCE_OVERLAY}/**",)
                if "provenance_overlay" in facts.flags
                else ()
            ),
            *(
                (f"$origin/{review.HUMAN_ANCHOR_OUTLINES}/**",)
                if "anchor_outlines" in facts.flags
                else ()
            ),
        ),
        visual_bounds=_bounds(),
    )
    arm_tiles = [review.candidate_arm_view(root, name, DIMENSIONS) for name in facts.candidate_arms]
    series: list[rrb.View] = []
    if "provenance" in facts.flags:
        series.append(
            time_series_view(
                f"{root}/{review.REFERENCE_PROVENANCE_SERIES}",
                "Reference provenance per part (1 sam3 primary, 2 dam4sam fallback)",
            )
        )
    if "consensus" in facts.flags:
        series.append(
            time_series_view(
                f"{root}/{review.MULTIVIEW_DIAGNOSTICS}",
                "Consensus contradiction: C10379 error vs consensus (raw px), views used",
            )
        )
    if "confidence" in facts.flags:
        series.append(
            time_series_view(
                f"{root}/{review.CONFIDENCE_SERIES}",
                "Detector confidence per part (1 - suspicion); crosses = abstain",
            )
        )
    if "anchors" in facts.flags:
        series.append(
            time_series_view(
                f"{root}/{review.ANCHOR_SERIES}",
                "Human anchor frames (diamonds) and failed cells (crosses)",
                ("$origin/anchor_frame", "$origin/failed_cells"),
            )
        )
    series.append(
        time_series_view(
            f"{root}/{review.CANDIDATE_AREA_SERIES}", "Candidate arms: mask area per part (px)"
        )
    )
    documents: list[rrb.View] = []
    if "anchor_log" in facts.flags:
        documents.append(
            rrb.TextLogView(origin=f"{root}/{review.ANCHOR_SERIES}/log", name="Anchor log")
        )
    if "navigation" in facts.flags:
        documents.append(
            rrb.TextDocumentView(
                origin=f"{root}/{review.NAVIGATION_CURRENT}", name="Current frame (per part)"
            )
        )
    if "review_notes" in facts.flags:
        documents.append(
            rrb.TextDocumentView(origin=f"{root}/{review.REVIEW_NOTES}", name="Review guide")
        )
    rows: list[rrb.Container | rrb.View] = [
        rrb.Horizontal(primary, rrb.Vertical(*series), column_shares=[3, 2])
    ]
    if arm_tiles:
        rows.append(rrb.Horizontal(*arm_tiles))
    if documents:
        rows.append(rrb.Horizontal(*documents))
    shares = [4, *([3] if arm_tiles else []), *([1] if documents else [])]
    return rrb.Blueprint(
        rrb.Vertical(*rows, row_shares=shares),
        _time_panel(),
        auto_layout=False,
        auto_views=False,
    )


def hands_blueprint(facts: RecordingFacts) -> rrb.Blueprint:
    """Every hand source, 2D and 3D, with the disagreement series; no masks."""
    root = _require_review_root(facts)

    def tile(name: str, *contents: str) -> rrb.Spatial2DView:
        return rrb.Spatial2DView(
            origin=root,
            name=name,
            contents=("$origin/source/video", *contents),
            visual_bounds=_bounds(),
        )

    tiles_2d = [
        tile(
            "Stabilized WiLoR (primary hand layer)",
            "$origin/primary/stabilized_wilor/render/hands/**",
        ),
        tile("Raw WiLoR 2D", "$origin/comparison/wilor_2d/render/hands/**"),
        tile("MediaPipe 2D (fallback evidence)", "$origin/comparison/mediapipe_2d/render/hands/**"),
    ]
    if "assembly101_2d" in facts.flags:
        tiles_2d.append(
            tile(
                "Assembly101 dataset hands 2D (projected 60 fps 3D, +9 frame static offset)",
                f"$origin/{review.ASSEMBLY101_2D_ROOT}/**",
            )
        )
    views_3d: list[rrb.View] = []
    if "assembly101_3d" in facts.flags:
        views_3d.append(
            rrb.Spatial3DView(
                origin=f"{root}/{review.ASSEMBLY101_3D_ROOT}",
                name="World mm: dataset hands, ATHENA triangulations, C10379 camera",
                contents=(
                    "$origin/**",
                    "- $origin/multiview_consensus",
                    "- $origin/hull/**",
                ),
            )
        )
    if "wilor_3d" in facts.flags:
        views_3d.append(
            rrb.Spatial3DView(
                origin=f"{root}/contexts/wilor_camera_relative_non_metric_3d",
                name="WiLoR camera-relative non-metric 3D",
                contents="$origin/camera_relative_3d/**",
            )
        )
    series: list[rrb.View] = [
        time_series_view(
            f"{root}/diagnostics/hand_disagreement",
            "MediaPipe vs WiLoR disagreement and stabilized-layer states",
        ),
    ]
    if "assembly101_series" in facts.flags:
        series.append(
            time_series_view(
                f"{root}/{review.ASSEMBLY101_DIAGNOSTICS}",
                "Dataset hand confidence, wrist distance to WiLoR, ATHENA disagreement (mm)",
                (
                    "$origin/confidence/**",
                    "$origin/wrist_distance_to_stabilized_wilor_pixels/**",
                    *(f"$origin/{label}/wrist_disagreement_mm/**" for label in facts.athena_labels),
                    *(f"$origin/{label}/contributing_views/**" for label in facts.athena_labels),
                ),
            )
        )
    if "confidence" in facts.flags:
        series.append(
            time_series_view(
                f"{root}/{review.CONFIDENCE_SERIES}",
                "Segmentation detector confidence (context for hand-part contact)",
            )
        )
    series.append(
        time_series_view(
            f"{root}/diagnostics/contact", "Hand-to-part distances and debounced contact candidates"
        )
    )
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(*tiles_2d),
            rrb.Horizontal(
                *views_3d, rrb.Vertical(*series), column_shares=[2] * len(views_3d) + [3]
            ),
            row_shares=[3, 3],
        ),
        _time_panel(),
        auto_layout=False,
        auto_views=False,
    )


def multiview_blueprint(facts: RecordingFacts) -> rrb.Blueprint:
    """Camera tiles with masks, the world-mm rig with hull voxels, per-view error, episodes."""
    if facts.multiview_root is None:
        raise ValueError(f"{facts.path} holds no multiview comparison entities")
    entity = facts.multiview_root.lstrip("/")
    anchor_view = next(
        (flag.split(":", 1)[1] for flag in sorted(facts.flags) if flag.startswith("anchor_view:")),
        None,
    )
    proposal_views = tuple(
        flag.split(":", 1)[1] for flag in sorted(facts.flags) if flag.startswith("proposal_view:")
    )
    grid = rrb.Grid(
        *[
            multiview.view_tile(
                entity,
                view,
                hull="hull_projection" in facts.flags,
                anchor_view=anchor_view,
                proposal_views=proposal_views,
            )
            for view in facts.multiview_views
        ],
        grid_columns=3 if len(facts.multiview_views) > 8 else 4,
    )
    tabs: list[rrb.View] = [
        time_series_view(
            f"{entity}/diagnostics/multiview/{target}",
            f"{target}: per-view error vs consensus (raw px)",
        )
        for target in TARGETS
    ]
    review_root = facts.review_root.lstrip("/") if facts.review_root else None
    if review_root is not None and "consensus" in facts.flags:
        tabs.insert(
            0,
            time_series_view(
                f"{review_root}/{review.MULTIVIEW_DIAGNOSTICS}",
                "C10379 contradiction: error vs consensus and views used (all parts)",
            ),
        )
    if review_root is not None and "anchors" in facts.flags:
        tabs.append(
            time_series_view(
                f"{review_root}/{review.ANCHOR_SERIES}",
                "Human anchor frames and failed cells (C10379)",
                ("$origin/anchor_frame", "$origin/failed_cells"),
            )
        )
    elif "multiview_anchors" in facts.flags:
        tabs.append(
            time_series_view(
                f"{entity}/{multiview.ANCHOR_SERIES}",
                "Human anchor frames (C10379)",
                "$origin/anchor_frame",
            )
        )
    documents: list[rrb.View] = [
        rrb.TextDocumentView(origin=f"{entity}/metadata/disagreement", name="Disagreement episodes")
    ]
    if review_root is not None and "multiview_document" in facts.flags:
        documents.append(
            rrb.TextDocumentView(
                origin=f"{review_root}/{multiview.V4_TEXT_PANEL[0]}",
                name="Episodes as seen by the review package",
            )
        )
    return rrb.Blueprint(
        rrb.Vertical(
            grid,
            rrb.Horizontal(
                rrb.Spatial3DView(
                    origin=f"{entity}/{multiview.WORLD_3D}",
                    name="World mm: consensus centroids, dataset hands, 8 static cameras"
                    + (", hull voxels (1 fps)" if "hull_voxels" in facts.flags else ""),
                    contents="$origin/**",
                ),
                rrb.Tabs(*tabs),
                rrb.Tabs(*documents),
                column_shares=[3, 3, 2],
            ),
            row_shares=[3, 2],
        ),
        _time_panel(),
        auto_layout=False,
        auto_views=False,
    )


PRESET_BUILDERS = {
    "segmentation": segmentation_blueprint,
    "hands": hands_blueprint,
    "multiview": multiview_blueprint,
}


# -- validation ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class BlueprintViewContents:
    name: str
    origin: str
    queries: tuple[str, ...]


def read_blueprint_views(rbl_path: Path) -> tuple[str, tuple[BlueprintViewContents, ...]]:
    """(application id, views) of a saved `.rbl`, read back through the SDK."""
    import rerun.experimental as experimental

    reader = experimental.RrdReader(str(rbl_path))
    blueprints = reader.blueprints()
    if len(blueprints) != 1:
        raise ValueError(f"{rbl_path} holds {len(blueprints)} blueprint stores; expected one")
    store = blueprints[0]
    names: dict[str, str] = {}
    origins: dict[str, str] = {}
    queries: dict[str, tuple[str, ...]] = {}
    for chunk in reader.stream(store=store):
        path = str(chunk.entity_path)
        if not path.startswith("/view/"):
            continue
        view_id = path.split("/")[2]
        batch = chunk.to_record_batch()
        columns = set(batch.schema.names)
        if "ViewContents:query" in columns:
            queries[view_id] = tuple(
                str(item) for row in batch.column("ViewContents:query").to_pylist() for item in row
            )
        if "ViewBlueprint:space_origin" in columns:
            origins[view_id] = str(batch.column("ViewBlueprint:space_origin").to_pylist()[-1][0])
        if "ViewBlueprint:display_name" in columns:
            names[view_id] = str(batch.column("ViewBlueprint:display_name").to_pylist()[-1][0])
    views = tuple(
        BlueprintViewContents(
            name=names.get(view_id, view_id),
            origin=origins.get(view_id, ""),
            queries=queries.get(view_id, ("$origin/**",)),
        )
        for view_id in sorted(set(names) | set(origins) | set(queries))
    )
    return str(store.application_id), views


def expand_query(origin: str, query: str) -> str:
    return query.replace("$origin", "/" + origin.strip("/")).replace("//", "/")


def query_matches(expression: str, entity_paths: tuple[str, ...]) -> bool:
    """`/a/b/**` matches `/a/b` and every descendant; anything else is an exact path."""
    if expression.endswith("/**"):
        base = expression[: -len("/**")]
        return any(path == base or path.startswith(base + "/") for path in entity_paths)
    return expression in entity_paths


def unmatched_queries(
    views: tuple[BlueprintViewContents, ...], entity_paths: tuple[str, ...]
) -> list[tuple[str, str]]:
    """(view name, expanded expression) for every inclusion that matches no logged entity."""
    missing: list[tuple[str, str]] = []
    for view in views:
        for query in view.queries:
            text = query.strip()
            if text.startswith("-"):
                continue
            if text.startswith("+"):
                text = text[1:].strip()
            expression = expand_query(view.origin, text)
            if not query_matches(expression, entity_paths):
                missing.append((view.name, expression))
    return missing


def check_preset(rbl_path: Path, facts: RecordingFacts) -> dict[str, object]:
    application_id, views = read_blueprint_views(rbl_path)
    missing = unmatched_queries(views, facts.entity_paths)
    return {
        "preset": rbl_path.name,
        "application_id": application_id,
        "application_id_matches": application_id == facts.application_id,
        "views": len(views),
        "queries": sum(len(view.queries) for view in views),
        "unmatched": [{"view": name, "expression": expression} for name, expression in missing],
        "ok": application_id == facts.application_id and not missing,
    }


def write_presets(
    rrd_path: Path, output_dir: Path, *, names: tuple[str, ...] = PRESET_NAMES
) -> dict[str, object]:
    """Write the presets for `rrd_path` into `output_dir` and validate each one."""
    facts = read_recording_facts(rrd_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name in names:
        path = output_dir / f"{name}.rbl"
        PRESET_BUILDERS[name](facts).save(facts.application_id, path)
        results.append(check_preset(path, facts))
    report = {
        "recording": rrd_path.as_posix(),
        "application_id": facts.application_id,
        "recording_id": facts.recording_id,
        "entity_paths": len(facts.entity_paths),
        "review_root": facts.review_root,
        "multiview_root": facts.multiview_root,
        "candidate_arms": list(facts.candidate_arms),
        "multiview_views": list(facts.multiview_views),
        "flags": sorted(facts.flags),
        "presets": results,
        "ok": all(bool(item["ok"]) for item in results),
        "open_command": " ".join(
            [
                "rerun",
                rrd_path.as_posix(),
                *[(output_dir / f"{name}.rbl").as_posix() for name in names[:1]],
            ]
        ),
    }
    (output_dir / PRESET_CHECK_NAME).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def merge_recordings(inputs: tuple[Path, ...], output: Path) -> Path:
    """`rerun rrd merge` of recordings that share one application/recording id, then the
    recording store alone re-written so the result carries no embedded layout: a merged file
    with a leftover blueprint and a `.rbl` on the command line would race for activation."""
    import rerun.experimental as experimental

    rerun_binary = Path(sys.executable).with_name("rerun")
    merged = output.with_name(output.name + ".merged.tmp")
    subprocess.run(
        [str(rerun_binary), "rrd", "merge", *[str(path) for path in inputs], "-o", str(merged)],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        reader = experimental.RrdReader(str(merged))
        recordings = reader.recordings()
        if len(recordings) != 1:
            raise ValueError(
                f"merge of {[path.name for path in inputs]} yielded {len(recordings)} recording "
                "stores; the inputs must share one application id and recording id"
            )
        store = recordings[0]
        reader.store(store=store).write_rrd(
            output, application_id=store.application_id, recording_id=store.recording_id
        )
    finally:
        merged.unlink(missing_ok=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rrd", type=Path, required=True, help="Recording the presets are for.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write the .rbl files (default: beside the recording).",
    )
    parser.add_argument(
        "--merge",
        type=Path,
        nargs="*",
        default=None,
        metavar="RRD",
        help=(
            "Merge these recordings (same application and recording id) into --rrd first "
            f"(written as {COMBINED_RECORDING_NAME} beside the first input unless --rrd names "
            "a new path); presets are then written for the merged file."
        ),
    )
    parser.add_argument("--names", default=",".join(PRESET_NAMES))
    args = parser.parse_args()
    rrd_path = args.rrd
    if args.merge:
        if rrd_path.exists() and rrd_path in args.merge:
            parser.error("--rrd must be the merge output, not one of its inputs")
        merge_recordings(tuple(args.merge), rrd_path)
        size_mb = rrd_path.stat().st_size / 1e6
        print(f"merged {len(args.merge)} recordings into {rrd_path} ({size_mb:.1f} MB)")
    output_dir = args.output_dir or rrd_path.parent
    names = tuple(name.strip() for name in args.names.split(",") if name.strip())
    report = write_presets(rrd_path, output_dir, names=names)
    for item in report["presets"]:  # type: ignore[union-attr]
        status = "ok" if item["ok"] else f"UNMATCHED {item['unmatched']}"
        counts = f"{item['views']} views, {item['queries']} queries"
        print(f"{output_dir / item['preset']}: {counts}, {status}")
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
