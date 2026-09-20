"""v6 review surface: candidate arms, confidence and anchor marks in the v4 builder, the
multiview anchor/proposal outlines, and the blueprint presets with their entity check."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rerun as rr
from PIL import Image

from battle import interaction_review as review
from battle import interaction_review_v4 as v4
from battle import multiview_review as multiview
from battle import review_presets as presets
from battle.four_part_contract import TARGETS

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# -- v4 builder options ---------------------------------------------------------------------


def test_candidate_arm_spec_parses_and_rejects_bad_names() -> None:
    assert v4.parse_candidate_arm("pm-append=runs/x/y") == ("pm-append", Path("runs/x/y"))
    for bad in ("nopath", "=runs/x", "bad name=runs/x", "a/b=runs/x", f"{v4.HUMAN_ANCHOR_ARM}=r"):
        with pytest.raises(Exception):
            v4.parse_candidate_arm(bad)


def test_provenance_legend_lists_only_states_present_and_named_ineligible_intervals() -> None:
    sidecar = SimpleNamespace(
        summaries=[
            SimpleNamespace(provenance_counts={"sam3_corrected": 1754, "dam4sam_fallback": 46}),
            SimpleNamespace(provenance_counts={"sam3_corrected": 1800, "hidden_agent_label": 0}),
        ]
    )
    policy = SimpleNamespace(
        targets=[
            SimpleNamespace(target_id="chassis", not_contact_eligible_intervals=()),
            SimpleNamespace(
                target_id="rear_body",
                not_contact_eligible_intervals=(
                    SimpleNamespace(
                        start_frame=1660,
                        end_frame_exclusive=1800,
                        failure_case="distractor_confusion",
                    ),
                ),
            ),
        ]
    )
    legend = v4.provenance_legend(sidecar, policy)  # type: ignore[arg-type]
    assert legend == (
        "1 sam3 primary, 2 dam4sam fallback; rear_body [1660,1800) not_contact_eligible: "
        "distractor_confusion"
    )
    assert "hidden" not in legend


def test_confidence_series_loads_rows_and_refuses_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "confidence.jsonl"
    rows = [
        {"analysis_frame_index": 0, "target": "chassis", "confidence": 0.9, "abstain": False},
        {
            "analysis_frame_index": 600,
            "target": "chassis",
            "confidence": 0.1,
            "abstain": True,
            "is_anchor_frame": True,
            "anchor_truth_failed": True,
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    loaded = v4.load_confidence_series(path)
    assert loaded[(600, "chassis")].abstain and loaded[(600, "chassis")].anchor_truth_failed
    assert loaded[(0, "chassis")].anchor_truth_failed is None
    path.write_text(json.dumps(rows[0]) + "\n" + json.dumps(rows[0]) + "\n")
    with pytest.raises(ValueError, match="repeats"):
        v4.load_confidence_series(path)


def _write_anchor_fixture(root: Path) -> tuple[Path, Path]:
    config = root / "configs/qa/anchors.json"
    config.parent.mkdir(parents=True)
    config.write_text(
        json.dumps(
            {
                "frames": [{"analysis_frame_index": 300}, {"analysis_frame_index": 1700}],
                "windows": {"279-408": [279, 408]},
            }
        )
    )
    workspace = root / "runs/anchors"
    (workspace / "anchors/masks").mkdir(parents=True)
    mask = np.zeros((720, 1280), dtype=np.uint8)
    mask[100:140, 200:260] = 255
    Image.fromarray(mask).save(workspace / "anchors/masks/f000300_chassis.png")
    (workspace / "anchors/anchor_masks.json").write_text(
        json.dumps(
            {
                "view_id": "static-c10379",
                "anchors": [
                    {
                        "analysis_frame_index": 300,
                        "target": "chassis",
                        "state": "labeled",
                        "mask_uri": "anchors/masks/f000300_chassis.png",
                    },
                    {"analysis_frame_index": 1700, "target": "rear_body", "state": "hidden"},
                ],
            }
        )
    )
    return config.relative_to(root), (workspace / "anchors/anchor_masks.json").relative_to(root)


def test_anchor_marks_log_on_anchor_frames_and_clear_on_the_next(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, mask_set = _write_anchor_fixture(tmp_path)
    anchors = v4.load_anchor_marks(tmp_path, config, mask_set)
    assert anchors.frames == (300, 1700)
    assert anchors.states[(1700, "rear_body")] == "hidden"
    assert (300, "chassis") in anchors.masks and (1700, "rear_body") not in anchors.masks
    assert anchors.window_for(300) == "279-408" and anchors.window_for(1700) is None
    strips = v4.mask_outlines(np.array(Image.open(anchors.masks[(300, "chassis")])) > 0)
    assert len(strips) == 1 and strips[0][0] == strips[0][-1]

    logged: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "battle.interaction_review_v4.rr.log",
        lambda path, value, **_: logged.append((path, value.__class__.__name__)),
    )
    monkeypatch.setattr(
        "battle.interaction_review.rr.log",
        lambda path, value, **_: logged.append((path, value.__class__.__name__)),
    )
    confidence = {
        (300, "chassis"): v4.ConfidenceRow(0.9, False, True, False),
        (300, "interior"): v4.ConfidenceRow(0.1, True, True, True),
    }
    v4._log_anchor_frame("e", 299, anchors, confidence)
    assert logged == []
    v4._log_anchor_frame("e", 300, anchors, confidence)
    kinds = dict(logged)
    assert kinds[f"e/{v4.ANCHOR_SERIES}/anchor_frame"] == "Scalars"
    assert kinds[f"e/{v4.ANCHOR_SERIES}/failed_cells"] == "Scalars"
    assert kinds[f"e/{v4.HUMAN_ANCHOR_OUTLINES}/chassis"] == "LineStrips2D"
    assert (
        kinds[f"e/{v4.CANDIDATE_SEGMENTATION_ROOT}/{v4.HUMAN_ANCHOR_ARM}/chassis"] == "EncodedImage"
    )
    assert kinds[f"e/{v4.ANCHOR_LOG}"] == "TextLog"
    logged.clear()
    v4._log_anchor_frame("e", 301, anchors, confidence)
    assert {kind for _, kind in logged} == {"Clear"}
    assert (f"e/{v4.HUMAN_ANCHOR_OUTLINES}", "Clear") in logged


def test_candidate_arm_frame_logs_masks_and_areas_or_clears(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    (run / "masks").mkdir(parents=True)
    mask = np.zeros((720, 1280), dtype=np.uint8)
    mask[0:10, 0:10] = 255
    Image.fromarray(mask).save(run / "masks/000000_chassis.png")
    item = SimpleNamespace(label="chassis", mask=SimpleNamespace(uri="masks/000000_chassis.png"))
    source = SimpleNamespace(run_directory=run, observations={0: SimpleNamespace(objects=[item])})
    logged: dict[str, object] = {}
    monkeypatch.setattr(
        "battle.interaction_review_v4.rr.log",
        lambda path, value, **_: logged.__setitem__(path, value),
    )
    areas = v4._log_candidate_arm_frame("e", "arm", source, 0)  # type: ignore[arg-type]
    assert areas == {"chassis": 100}
    assert (
        logged[f"e/{v4.CANDIDATE_SEGMENTATION_ROOT}/arm/chassis"].__class__.__name__
        == "EncodedImage"
    )
    assert logged[f"e/{v4.CANDIDATE_AREA_SERIES}/arm/chassis"].__class__.__name__ == "Scalars"
    assert logged[f"e/{v4.CANDIDATE_SEGMENTATION_ROOT}/arm/cabin"].__class__.__name__ == "Clear"
    logged.clear()
    assert v4._log_candidate_arm_frame("e", "arm", source, 5) == {}  # type: ignore[arg-type]
    assert all(value.__class__.__name__ == "Clear" for value in logged.values())


def test_v4_blueprint_with_arms_references_only_new_roots_the_builder_logs() -> None:
    root = "world/clip/interaction_review_v4"
    blueprint = review._blueprint(
        root,
        (1280, 720),
        reference_provenance=True,
        candidate_arms=("pm-append", v4.HUMAN_ANCHOR_ARM),
        confidence=True,
        anchors=True,
        provenance_panel_name="legend",
    )
    from rerun.blueprint.api import View

    def walk(node):
        if isinstance(node, View):
            yield node
            return
        for child in node.contents:
            yield from walk(child)

    views = list(walk(blueprint.root_container))
    names = {str(view.name) for view in views}
    assert "legend" in names
    assert "pm-append: four part masks (candidate arm)" in names
    queries = {
        str(item).replace("$origin", str(view.origin))
        for view in views
        if isinstance(view.contents, tuple)
        for item in view.contents
    }
    assert f"{root}/{review.CANDIDATE_SEGMENTATION_ROOT}/pm-append/**" in queries
    assert f"{root}/{review.HUMAN_ANCHOR_OUTLINES}/**" in queries
    assert f"{root}/{review.ANCHOR_SERIES}/anchor_frame" in queries
    origins = {str(view.origin) for view in views}
    assert f"{root}/{review.CONFIDENCE_SERIES}" in origins
    assert f"{root}/{review.CANDIDATE_AREA_SERIES}" in origins


# -- multiview outlines and proposals --------------------------------------------------------


def _write_proposal_fixture(root: Path) -> Path:
    base = root / "proposals"
    for view, part, frame, count in (("C10119", "chassis", 0, 2), ("C10119", "interior", 427, 1)):
        cell = base / view / f"{part}_f{frame:06d}"
        cell.mkdir(parents=True)
        candidates = []
        for index in range(count):
            mask = np.zeros((720, 1280), dtype=np.uint8)
            mask[50 + index * 20 : 80 + index * 20, 50:90] = 255
            Image.fromarray(mask).save(cell / f"candidate_{index:02d}.png")
            candidates.append({"strategy": f"s{index}", "mask_uri": f"candidate_{index:02d}.png"})
        (cell / "proposal.json").write_text(
            json.dumps(
                {
                    "view": view,
                    "part": part,
                    "analysis_frame_index": frame,
                    "disagreement_1_minus_mean_pairwise_iou": 0.3,
                    "candidates": candidates,
                    "human_decision": None,
                }
            )
        )
    return base.relative_to(root)


def test_proposals_outline_each_candidate_at_its_frame_and_clear_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _write_proposal_fixture(tmp_path)
    proposals = multiview.load_proposals(tmp_path, root)
    assert list(proposals) == ["C10119"]
    assert [(cell.part, cell.frame) for cell in proposals["C10119"]] == [
        ("chassis", 0),
        ("interior", 427),
    ]
    assert multiview.load_proposals(tmp_path, Path("missing")) == {}
    logged: list[tuple[str, object]] = []
    monkeypatch.setattr(
        "battle.multiview_review.rr.log", lambda path, value, **_: logged.append((path, value))
    )
    multiview.log_proposals_frame("v", 0, proposals["C10119"])
    assert [path for path, _ in logged] == ["v/proposals/chassis"]
    strips = logged[0][1]
    assert strips.__class__.__name__ == "LineStrips2D"
    logged.clear()
    multiview.log_proposals_frame("v", 1, proposals["C10119"])
    assert [(path, value.__class__.__name__) for path, value in logged] == [
        ("v/proposals/chassis", "Clear")
    ]
    logged.clear()
    multiview.log_proposals_frame("v", 100, proposals["C10119"])
    assert logged == []


def test_anchor_outlines_only_on_the_labelled_views_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, mask_set = _write_anchor_fixture(tmp_path)
    view, anchors = multiview.load_anchor_outlines(tmp_path, mask_set)
    assert view == "C10379" and sorted(anchors) == [300, 1700]
    assert "chassis" in anchors[300] and anchors[1700] == {}
    logged: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "battle.multiview_review.rr.log",
        lambda path, value, **_: logged.append((path, value.__class__.__name__)),
    )
    multiview.log_anchor_outlines_frame("root", "root/views/C10379", 300, anchors)
    assert ("root/views/C10379/human_anchor_outlines/chassis", "LineStrips2D") in logged
    assert (f"root/{multiview.ANCHOR_SERIES}/anchor_frame", "Scalars") in logged
    logged.clear()
    multiview.log_anchor_outlines_frame("root", "root/views/C10379", 301, anchors)
    assert ("root/views/C10379/human_anchor_outlines", "Clear") in logged
    logged.clear()
    multiview.log_anchor_outlines_frame("root", "root/views/C10379", 302, anchors)
    assert logged == []


def test_view_tiles_name_the_seed_provenance_and_include_outline_layers() -> None:
    reference = multiview.view_tile("e", "C10379", hull=True, anchor_view="C10379")
    other = multiview.view_tile("e", "C10119", hull=False, proposal_views=("C10119",))
    assert "human seeds" in str(reference.name) and "anchor outlines" in str(reference.name)
    assert "agent seeds, unreviewed" in str(other.name) and "proposals" in str(other.name)
    assert "$origin/human_anchor_outlines/**" in tuple(map(str, reference.contents))
    assert "$origin/proposals/**" in tuple(map(str, other.contents))
    assert "$origin/hull_projection/**" not in tuple(map(str, other.contents))


# -- presets and the entity check ----------------------------------------------------------


def test_query_resolution_handles_origin_globs_and_exclusions() -> None:
    paths = ("/a/b", "/a/b/c", "/a/d")
    assert presets.expand_query("a/b", "$origin/c") == "/a/b/c"
    assert presets.query_matches("/a/b/**", paths)
    assert presets.query_matches("/a/**", paths)
    assert presets.query_matches("/a/d", paths)
    assert not presets.query_matches("/a/x/**", paths)
    assert not presets.query_matches("/a/b/c/d", paths)
    views = (
        presets.BlueprintViewContents("ok", "a/b", ("$origin/**", "- $origin/zzz/**")),
        presets.BlueprintViewContents("bad", "a", ("$origin/d", "+ $origin/missing")),
    )
    assert presets.unmatched_queries(views, paths) == [("bad", "/a/missing")]


REVIEW_ROOT = "world/clip/interaction_review_v4"
MULTIVIEW_ROOT = multiview.ENTITY_ROOT


def _write_synthetic_recording(path: Path, *, with_multiview: bool = True) -> None:
    rr.init("battle-test-presets", recording_id="presets-test")
    rr.save(path)
    rr.set_time("analysis_frame", sequence=0)
    rr.set_time("analysis_time", duration=0.0)
    review_paths = [
        "source/video",
        "primary/reference_four_part_segmentation/chassis",
        f"{review.REFERENCE_PROVENANCE_OVERLAY}/chassis",
        f"{review.HUMAN_ANCHOR_OUTLINES}/chassis",
        f"{review.REFERENCE_PROVENANCE_SERIES}/chassis",
        f"{review.CANDIDATE_SEGMENTATION_ROOT}/armA/chassis",
        f"{review.CANDIDATE_SEGMENTATION_ROOT}/{v4.HUMAN_ANCHOR_ARM}/chassis",
        f"{review.CANDIDATE_AREA_SERIES}/armA/chassis",
        f"{review.CONFIDENCE_SERIES}/chassis/confidence",
        f"{review.CONFIDENCE_SERIES}/chassis/abstain",
        f"{review.ANCHOR_SERIES}/anchor_frame",
        f"{review.ANCHOR_SERIES}/failed_cells",
        f"{review.ANCHOR_SERIES}/log",
        f"{review.MULTIVIEW_DIAGNOSTICS}/chassis/c10379_error_px",
        multiview.V4_TEXT_PANEL[0],
        review.NAVIGATION_CURRENT,
        review.NAVIGATION_FINE_GT_INDEX,
        review.REVIEW_NOTES,
        "primary/stabilized_wilor/render/hands/x",
        "comparison/wilor_2d/render/hands/x",
        "comparison/mediapipe_2d/render/hands/x",
        f"{review.ASSEMBLY101_2D_ROOT}/skeletons",
        f"{review.ASSEMBLY101_3D_ROOT}/hands/joints",
        f"{review.ASSEMBLY101_3D_ROOT}/athena_hands/joints",
        f"{review.ASSEMBLY101_3D_ROOT}/multiview_consensus",
        f"{review.ASSEMBLY101_DIAGNOSTICS}/confidence/left",
        f"{review.ASSEMBLY101_DIAGNOSTICS}/wrist_distance_to_stabilized_wilor_pixels/left",
        f"{review.ASSEMBLY101_DIAGNOSTICS}/athena_hands/wrist_disagreement_mm/left",
        f"{review.ASSEMBLY101_DIAGNOSTICS}/athena_hands/contributing_views/left",
        "contexts/wilor_camera_relative_non_metric_3d/camera_relative_3d/joints",
        "diagnostics/hand_disagreement/mean_pixels",
        "diagnostics/contact/x/y",
    ]
    for relative in review_paths:
        rr.log(f"{REVIEW_ROOT}/{relative}", rr.Scalars([1.0]))
    if with_multiview:
        for view in ("C10379", "C10119"):
            for relative in (
                "video",
                "masks/chassis",
                "consensus_markers",
                "hull_projection/chassis",
            ):
                rr.log(f"{MULTIVIEW_ROOT}/views/{view}/{relative}", rr.Scalars([1.0]))
        rr.log(f"{MULTIVIEW_ROOT}/views/C10379/human_anchor_outlines/chassis", rr.Scalars([1.0]))
        rr.log(f"{MULTIVIEW_ROOT}/views/C10119/proposals/chassis", rr.Scalars([1.0]))
        rr.log(f"{MULTIVIEW_ROOT}/{multiview.WORLD_3D}/hull/chassis", rr.Scalars([1.0]))
        rr.log(f"{MULTIVIEW_ROOT}/{multiview.ANCHOR_SERIES}/anchor_frame", rr.Scalars([1.0]))
        for target in TARGETS:
            rr.log(f"{MULTIVIEW_ROOT}/diagnostics/multiview/{target}/C10379", rr.Scalars([1.0]))
        rr.log(f"{MULTIVIEW_ROOT}/metadata/disagreement", rr.TextDocument("x"), static=True)
    rr.disconnect()


def test_presets_are_written_read_back_and_resolve_against_the_recording(tmp_path: Path) -> None:
    rrd = tmp_path / "test.rrd"
    _write_synthetic_recording(rrd)
    facts = presets.read_recording_facts(rrd)
    assert facts.application_id == "battle-test-presets"
    assert facts.review_root == f"/{REVIEW_ROOT}"
    assert facts.multiview_root == f"/{MULTIVIEW_ROOT}"
    assert facts.candidate_arms == ("armA", v4.HUMAN_ANCHOR_ARM)
    assert facts.multiview_views == ("C10119", "C10379")
    assert {
        "confidence",
        "anchors",
        "hull_voxels",
        "anchor_view:C10379",
        "proposal_view:C10119",
    } <= facts.flags

    report = presets.write_presets(rrd, tmp_path / "out")
    assert report["ok"], report
    assert sorted(path.name for path in (tmp_path / "out").glob("*.rbl")) == sorted(
        f"{name}.rbl" for name in presets.PRESET_NAMES
    )
    check = json.loads((tmp_path / "out" / presets.PRESET_CHECK_NAME).read_text())
    assert [item["preset"] for item in check["presets"]] == [
        f"{n}.rbl" for n in presets.PRESET_NAMES
    ]
    application_id, views = presets.read_blueprint_views(tmp_path / "out/multiview.rbl")
    assert application_id == "battle-test-presets"
    names = {view.name for view in views}
    assert any("C10379" in name and "anchor outlines" in name for name in names), names
    assert any("C10119" in name and "proposals" in name for name in names), names


def test_preset_check_reports_entities_the_recording_lacks(tmp_path: Path) -> None:
    rrd = tmp_path / "test.rrd"
    _write_synthetic_recording(rrd, with_multiview=False)
    facts = presets.read_recording_facts(rrd)
    assert facts.multiview_root is None
    with pytest.raises(ValueError, match="multiview"):
        presets.multiview_blueprint(facts)
    import rerun.blueprint as rrb

    stray = rrb.Blueprint(
        rrb.Spatial2DView(
            origin=REVIEW_ROOT,
            name="stray",
            contents=("$origin/source/video", "$origin/not/logged/**"),
        )
    )
    stray.save(facts.application_id, tmp_path / "stray.rbl")
    result = presets.check_preset(tmp_path / "stray.rbl", facts)
    assert result["ok"] is False
    assert result["unmatched"] == [{"view": "stray", "expression": f"/{REVIEW_ROOT}/not/logged/**"}]


def test_merge_yields_one_store_without_an_embedded_blueprint(tmp_path: Path) -> None:
    import rerun.blueprint as rrb
    import rerun.experimental as experimental

    a, b = tmp_path / "a.rrd", tmp_path / "b.rrd"
    for path, entity in ((a, "x/one"), (b, "y/two")):
        rr.init("battle-test-merge", recording_id="merge-test")
        rr.save(path)
        rr.set_time("analysis_frame", sequence=0)
        rr.log(entity, rr.Scalars([1.0]))
        rr.send_blueprint(rrb.Blueprint(rrb.TimeSeriesView(origin=entity)))
        rr.disconnect()
    merged = presets.merge_recordings((a, b), tmp_path / "merged.rrd")
    reader = experimental.RrdReader(str(merged))
    assert [(s.application_id, s.recording_id) for s in reader.recordings()] == [
        ("battle-test-merge", "merge-test")
    ]
    assert reader.blueprints() == []
    assert {"/x/one", "/y/two"} <= set(reader.store().schema().entity_paths())
    assert not (tmp_path / "merged.rrd.merged.tmp").exists()


# -- the built v6 package (ignored artifacts) ----------------------------------------------


V6_ROOT = Path("runs/interaction-review-first-minute-v6")


@pytest.mark.real_data
def test_built_v6_package_presets_check_passes_and_recording_is_one_store() -> None:
    import rerun.experimental as experimental

    check_path = V6_ROOT / presets.PRESET_CHECK_NAME
    if not check_path.is_file():
        pytest.skip("v6 package not built")
    check = json.loads(check_path.read_text())
    assert check["ok"], check
    assert set(check["candidate_arms"]) >= {
        "pm-append",
        "dam4sam-large-1024-sched-60s",
        "ensemble-v1",
    }
    assert len(check["multiview_views"]) == 9
    reader = experimental.RrdReader(str(V6_ROOT / presets.COMBINED_RECORDING_NAME))
    assert len(reader.recordings()) == 1 and reader.blueprints() == []
    for name in presets.PRESET_NAMES:
        application_id, views = presets.read_blueprint_views(V6_ROOT / f"{name}.rbl")
        assert application_id == check["application_id"]
        assert views
