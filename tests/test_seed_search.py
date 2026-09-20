from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from battle import seed_search as ss
from battle.schemas import PixelBox, PixelPoint

ROOT = Path(__file__).parents[1]


def _prompt(prompt_id: str, margin: float, negatives: str, part: str = "cabin") -> ss.PromptSpec:
    return ss.PromptSpec(
        prompt_id=prompt_id,
        view="C10379",
        reference_frame=0,
        target_frame=0,
        part=part,
        margin=margin,
        negatives=negatives,
        pixel_box=PixelBox(x1=0, y1=0, x2=10, y2=10),
        sphere_source="other_views",
    )


def _candidate(prompt: ss.PromptSpec, index: int, area: int, score: float) -> ss.Candidate:
    mask = np.zeros((20, 20), dtype=bool)
    mask[: area // 20 + 1, : min(20, area)] = True
    return ss.Candidate(
        prompt=prompt,
        index=index,
        mask_uri=f"results/masks/{prompt.prompt_id}_candidate-{index:02d}.png",
        score=score,
        area=area,
        sha256=f"{index:064x}",
        mask=mask,
    )


def test_strategy_names_round_trip_and_grid_size() -> None:
    grid = ss.strategy_grid()
    assert len(grid) == len(ss.MARGINS) * len(ss.NEGATIVE_SETS) * len(ss.BOX_MODES) * len(ss.PICKS)
    for key in grid[:7]:
        assert ss.parse_strategy(key.name) == key
    assert ss.parse_strategy("m0.25|cabin_edge|two|highest_score").margin == 0.25


def test_pool_uses_the_margin_box_and_adds_the_wide_box_for_two() -> None:
    p_narrow = _prompt("t000000-b01", 0.15, "none")
    p_wide = _prompt("t000000-b02", 0.6, "none")
    p_rim = _prompt("t000000-b03", 0.15, "chassis_rim")
    candidates = {
        p_narrow.prompt_id: [_candidate(p_narrow, 0, 50, 0.9)],
        p_wide.prompt_id: [_candidate(p_wide, 0, 300, 0.7)],
        p_rim.prompt_id: [_candidate(p_rim, 0, 60, 0.95)],
    }
    prompts = [p_narrow, p_wide, p_rim]
    one = ss.pool_for_strategy(
        prompts, candidates, ss.StrategyKey(margin=0.15, negatives="none", boxes="one", pick="x")
    )
    assert [c.prompt.prompt_id for c in one] == ["t000000-b01"]
    two = ss.pool_for_strategy(
        prompts, candidates, ss.StrategyKey(margin=0.15, negatives="none", boxes="two", pick="x")
    )
    assert sorted(c.prompt.prompt_id for c in two) == ["t000000-b01", "t000000-b02"]
    rim = ss.pool_for_strategy(
        prompts,
        candidates,
        ss.StrategyKey(margin=0.15, negatives="chassis_rim", boxes="one", pick="x"),
    )
    assert [c.prompt.prompt_id for c in rim] == ["t000000-b03"]
    # A negative set that produced no distinct prompt falls back to the `none` prompt.
    fallback = ss.pool_for_strategy(
        prompts,
        candidates,
        ss.StrategyKey(margin=0.15, negatives="hand_joints", boxes="one", pick="x"),
    )
    assert [c.prompt.prompt_id for c in fallback] == ["t000000-b01"]


def test_pick_rules() -> None:
    prompt = _prompt("t000000-b01", 0.15, "none")
    pool = [
        _candidate(prompt, 0, 100, 0.8),
        _candidate(prompt, 1, 400, 0.6),
        _candidate(prompt, 2, 40, 0.9),
    ]
    assert ss.pick_candidate(pool, "highest_score").index == 2
    assert ss.pick_candidate(pool, "largest").index == 1
    assert ss.pick_candidate(pool, "smallest").index == 2
    assert ss.pick_candidate([], "largest") is None
    embeddings = {
        "cand:t000000-b01:0": np.array([1.0, 0.0]),
        "cand:t000000-b01:1": np.array([0.0, 1.0]),
        "cand:t000000-b01:2": np.array([0.7, 0.7]),
    }
    exemplars = [np.array([0.0, 1.0])]
    assert (
        ss.pick_candidate(pool, "exemplar", embeddings=embeddings, exemplars=exemplars).index == 1
    )
    assert ss.pick_candidate(pool, "exemplar", embeddings=None, exemplars=exemplars) is None


def _cell(frame: int, part: str, ious: dict[str, float], role: str = "positive") -> ss.CellResult:
    return ss.CellResult(
        reference_frame=frame,
        part=part,
        truth_role=role,  # type: ignore[arg-type]
        truth_area=100,
        sphere_source="other_views",
        iou=ious,
        exact_match={k: v >= 0.999 for k, v in ious.items()},
        chosen_area=dict.fromkeys(ious, 90),
        chosen_candidate=dict.fromkeys(ious, "t000000-b01:0"),
        prompt_count=2,
    )


def test_leave_frames_out_picks_the_fit_winner_and_scores_held_out_frames() -> None:
    names = ("A", "B")
    anchors = (10, 20, 30, 40)
    always = (0,)
    # A wins on every frame but 30, where B is far better; B wins on frame 30 only.
    cells = [
        _cell(0, "cabin", {"A": 0.9, "B": 0.5}),
        _cell(10, "cabin", {"A": 0.9, "B": 0.5}),
        _cell(20, "cabin", {"A": 0.8, "B": 0.6}),
        _cell(30, "cabin", {"A": 0.1, "B": 1.0}),
        _cell(40, "cabin", {"A": 0.7, "B": 0.6}),
        _cell(40, "cabin", {"A": 0.0, "B": 0.0}, role="hidden"),
    ]
    summary = ss.summarize_part("cabin", cells, names, anchors, always)
    assert summary.winner == "A"
    assert summary.runner_up == "B"
    assert summary.frames_scored == 5
    assert summary.winner_mean_iou_all_frames == pytest.approx((0.9 + 0.9 + 0.8 + 0.1 + 0.7) / 5)
    # Each anchor frame is held out in three of the four rotations; frame 0 never is.
    assert set(summary.held_out_per_frame) == {"10", "20", "30", "40"}
    # Rotations hold out {10,20,30}, {20,30,40}, {30,40,10}, {40,10,20}; the fitting sets are
    # {0,40}, {0,10}, {0,20}, {0,30}. Only the last one prefers B (0.75 vs 0.5), so frame 30
    # is always scored with A (0.1) and frame 10 once with B (0.5).
    assert summary.held_out_per_frame["30"] == pytest.approx(0.1)
    assert summary.held_out_per_frame["10"] == pytest.approx((0.9 + 0.9 + 0.5) / 3)
    assert summary.held_out_iou_min == pytest.approx(0.1)
    assert sorted(summary.winners_per_split.values()) == ["A", "A", "A", "B"]
    assert summary.winners_per_split["10,20,40"] == "B"
    assert summary.exact_match_fraction == 0.0
    assert summary.hidden_false_positive_px == {"40": 0}
    assert not summary.passes_transfer_gate


def test_transfer_gate_uses_held_out_mean() -> None:
    names = ("A",)
    cells = [_cell(f, "cabin", {"A": 0.95}) for f in (0, 10, 20, 30)]
    summary = ss.summarize_part("cabin", cells, names, (10, 20, 30), (0,))
    assert summary.passes_transfer_gate
    assert summary.held_out_iou_mean == pytest.approx(0.95)


def test_decode_request_normalises_to_the_proxy() -> None:
    prompt = _prompt("t000012-b03", 0.25, "other_centroids")
    prompt = prompt.model_copy(
        update={
            "pixel_box": PixelBox(x1=64, y1=36, x2=128, y2=72),
            "background_points": (PixelPoint(x=32, y=18),),
        }
    )
    request = ss.decode_request(prompt, (72, 128))
    assert request["boxes"] == [[[0.5, 0.5], [1.0, 1.0]]]
    assert request["bg_points"] == [[0.25, 0.25]]
    assert request["fg_points"] == []
    assert request["candidate_id"] == "t000012-b03" and request["box_id"] == "p000012-b03"


def test_boundary_points_and_crop_box() -> None:
    mask = np.zeros((40, 60), dtype=bool)
    mask[10:30, 20:50] = True
    points = ss.mask_boundary_points(mask, 4)
    assert points.shape == (4, 2)
    for x, y in points:
        assert 19.5 <= x <= 50.5 and 9.5 <= y <= 30.5
    assert ss.mask_boundary_points(np.zeros((4, 4), dtype=bool), 4).shape == (0, 2)
    box = ss._crop_box(mask)
    assert box is not None
    x0, y0, x1, y1 = box
    assert x0 <= 20 and x1 >= 50 and y0 <= 10 and y1 >= 30
    assert ss._crop_box(np.zeros((4, 4), dtype=bool)) is None


def test_points_inside_box_uses_the_proxy_bounds() -> None:
    box = PixelBox(x1=10, y1=10, x2=20, y2=20)
    points = np.array([[12.0, 12.0], [25.0, 12.0], [15.0, 19.0]])
    inside = ss._points_inside(points, box, (30, 30))
    assert [(p.x, p.y) for p in inside] == [(12, 12), (15, 19)]


def _write_mask(path: Path, area: int) -> None:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask.flat[:area] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(path)


def test_collect_calibration_separates_positives_negatives_and_agent_masks(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    frame = {
        "analysis_frame_index": 300,
        "proxy_seconds": 10.0,
        "analysis_seconds": 10.0,
        "source_seconds": 304.0,
    }

    def candidate(
        cid: str,
        part: str,
        chosen: int | None,
        *,
        accepted: bool,
        rejected: bool,
        by: str = "human",
        n: int = 3,
    ):
        options = []
        for i in range(n):
            uri = f"results/masks/{cid}_candidate-{i:02d}.png"
            _write_mask(workspace / uri, 4 + i)
            options.append({"candidate_index": i, "iou_score": 0.5 + 0.1 * i, "mask_uri": uri})
        return {
            "candidate_id": cid,
            "intended_target": part,
            "frame": frame,
            "decoder_result": {"candidates": options},
            "human_selected_candidate_index": chosen,
            "human_accepted": accepted,
            "rejected": rejected,
            "selected_by": by,
        }

    manifest = {
        "view_id": "static-c10379",
        "candidates": [
            candidate("t000300-b01", "chassis", 1, accepted=True, rejected=False),
            candidate("t000300-b02", "chassis", None, accepted=False, rejected=True),
            candidate("t000300-b03", "interior", 0, accepted=True, rejected=False, by="agent"),
            candidate("t000300-b04", "cabin", None, accepted=False, rejected=False),
        ],
    }
    (workspace / "calibration_manifest.json").write_text(json.dumps(manifest))

    class Rig:
        def pose_frame(self, view: str, frame: int) -> int:
            return 17649 + 2 * frame

    entries, excluded = ss._collect_calibration(
        Path("ws/calibration_manifest.json"),
        repository_root=tmp_path,
        rig=Rig(),  # type: ignore[arg-type]
        view="C10379",
        source="anchors",
        frame_filter=None,
    )
    roles = sorted((e.role, e.candidate_index) for e in entries)
    assert roles == [
        ("negative_rejected", 0),
        ("negative_rejected", 1),
        ("negative_rejected", 2),
        ("negative_unchosen", 0),
        ("negative_unchosen", 2),
        ("positive", 1),
    ]
    positive = next(e for e in entries if e.role == "positive")
    assert positive.area_px == 5 and positive.pose_frame_index == 17649 + 600
    assert [e.positive_candidate_id for e in entries if e.role == "negative_unchosen"] == [
        "t000300-b01",
        "t000300-b01",
    ]
    assert [(x.source_candidate_id, x.selected_by) for x in excluded] == [("t000300-b03", "agent")]


@pytest.mark.real_data
def test_written_truth_set_and_search_report_are_consistent() -> None:
    truth = ss.load_truth_set(ROOT / "runs/seed-search-20260920/truth_set.json")
    assert truth.counts["positive_c10379"] == 64 and truth.counts["positive_total"] == 68
    assert truth.counts["hidden"] == 1 and truth.counts["negative_rejected"] == 8
    assert all(e.selected_by == "human" for e in truth.entries)
    assert all(x.selected_by == "agent" or "first minute" in x.reason for x in truth.excluded)
    report = ss.SearchReport.model_validate_json(
        (ROOT / "runs/seed-search-20260920/search/search_report.json").read_text()
    )
    assert set(report.parts) == set(ss.TARGETS)
    assert (
        report.parts["cabin"].passes_transfer_gate
        and report.parts["rear_body"].passes_transfer_gate
    )
    assert not report.parts["chassis"].passes_transfer_gate
    assert not report.parts["interior"].passes_transfer_gate
    for cell in report.cells:
        if cell.part == "interior":
            assert cell.sphere_source == "self_prior"
        else:
            assert cell.sphere_source == "other_views"
    transfer = ss.TransferReport.model_validate_json(
        (ROOT / "runs/seed-search-20260920/transfer/transfer_report.json").read_text()
    )
    for view, parts in transfer.seeds_per_view.items():
        assert parts == {
            "chassis": "sep18_carried_over",
            "interior": "blocked",
            "rear_body": "seed_search",
            "cabin": "seed_search",
        }, view
