from __future__ import annotations

import json
from pathlib import Path

import pytest

from battle import anchor_frames_for_view as afv
from battle.assembly101_pose_schemas import Assembly101ClockRule
from battle.review_anchors import (
    ReviewAnchorConfig,
    build_first_minute_config,
    load_mask_set,
    view_paths,
)
from battle.schemas import ArtifactFingerprint

ROOT = Path(__file__).parents[1]


def _rule(offset: int, view_key: str = "X:rgb") -> Assembly101ClockRule:
    return Assembly101ClockRule(
        view_key=view_key,
        proxy_start_raw_frame=17640,
        raw_frames_per_proxy_frame=2,
        pose_offset_frames=offset,
        pose_fps=60,
        analysis_fps=30,
        offset_uncertainty_frames=1,
        offset_evidence="synthetic",
    )


def _fp(uri: str) -> ArtifactFingerprint:
    return ArtifactFingerprint(uri=uri, sha256="0" * 64, source="measured")


@pytest.mark.parametrize(
    ("target_offset", "shift", "residual"),
    [(9, 0, 0), (7, 1, 0), (6, 1, -1), (5, 2, 0), (0, 4, -1)],
)
def test_mapping_rounds_half_frames_down_and_records_the_residual(
    target_offset: int, shift: int, residual: int
) -> None:
    source, target = _rule(9), _rule(target_offset)
    for frame in (0, 300, 1700):
        mapped, got_residual = afv.mapped_frame(source, target, frame)
        assert mapped == frame + shift
        assert got_residual == residual
        # The mapped frame never shows a later instant than the source frame.
        assert target.pose_frame(mapped) <= source.pose_frame(frame)
        assert -1 <= got_residual <= 0
    assert afv.analysis_frame_shift(source, target) == shift


def test_mapping_refuses_frames_before_the_target_proxy() -> None:
    with pytest.raises(ValueError, match="precedes"):
        afv.mapped_frame(_rule(0), _rule(9), 0)


def test_view_config_maps_frames_windows_and_hidden_interval() -> None:
    source = build_first_minute_config()
    config = afv.build_view_config(
        source,
        view="C10115",
        source_rule=_rule(9),
        target_rule=_rule(6),
        clip_config="configs/clips/x.json",
        manual_seed_target_config="configs/x.json",
        source_config_fingerprint=_fp("configs/qa/first_minute_review_anchors.json"),
        clock_rules_fingerprint=_fp("configs/assembly101/clock_rules.json"),
        extra_frames=[1000, 300],
        extra_frames_source="runs/x/proposed_anchor_frames.json",
    )
    assert config.view_id == "static-c10115"
    assert config.frame_mapping is not None
    assert config.frame_mapping.analysis_frame_shift == 1
    assert not config.frame_mapping.extra_frames_pending
    mapped = [f for f in config.frames if f.origin == "mapped_anchor"]
    assert [f.analysis_frame_index for f in mapped] == [
        f.analysis_frame_index + 1 for f in source.frames
    ]
    assert all(f.residual_pose_frames == -1 for f in mapped)
    assert all(f.source_analysis_frame_index == f.analysis_frame_index - 1 for f in mapped)
    extras = [f for f in config.frames if f.origin == "extra_detector_selected"]
    # 300 duplicates an anchor frame and is dropped; 1000 is new and is 'visible' everywhere.
    assert [f.analysis_frame_index for f in extras] == [1001]
    assert set(extras[0].expected_visible.values()) == {"visible"}
    assert config.windows == {
        "279-408": (280, 409),
        "573-722": (574, 723),
        "1020-1172": (1021, 1173),
    }
    assert config.hidden_prompt_interval == (1025, 1173)
    for frame in config.frames:
        inside = 1025 <= frame.analysis_frame_index < 1173
        assert frame.expected_visible["interior"] == ("hidden_prompt" if inside else "visible")
        assert frame.proxy_seconds == pytest.approx(frame.analysis_frame_index / 30)
    assert config.workspace_timestamps is not None
    assert config.workspace_timestamps.split(",")[0] == f"{301 / 30:.6f}"
    # Round trip through the schema.
    ReviewAnchorConfig.model_validate_json(config.model_dump_json())


def test_view_config_says_when_extra_frames_are_pending() -> None:
    config = afv.build_view_config(
        build_first_minute_config(),
        view="C10119",
        source_rule=_rule(9),
        target_rule=_rule(7),
        clip_config="c",
        manual_seed_target_config="m",
        source_config_fingerprint=_fp("a"),
        clock_rules_fingerprint=_fp("b"),
        extra_frames_note="not yet",
    )
    assert config.frame_mapping is not None
    assert config.frame_mapping.extra_frames_pending
    assert config.frame_mapping.extra_frames_source is None
    assert len(config.frames) == 13


def test_read_extra_frames_accepts_the_common_shapes(tmp_path: Path) -> None:
    for payload in ([5, 3, 3], {"frames": [3, 5]}, [{"analysis_frame_index": 5}, {"frame": 3}]):
        path = tmp_path / "f.json"
        path.write_text(json.dumps(payload))
        assert afv.read_extra_frames(path) == {
            3: "extra_detector_selected",
            5: "extra_detector_selected",
        }
    # Track A's shape: records with a `selection`; random draws keep their own origin.
    path = tmp_path / "proposed.json"
    path.write_text(
        json.dumps(
            {
                "frames": [
                    {"analysis_frame_index": 220, "selection": "detector_ranked"},
                    {"analysis_frame_index": 40, "selection": "random"},
                    {"analysis_frame_index": 220, "selection": "random"},
                ]
            }
        )
    )
    assert afv.read_extra_frames(path) == {40: "extra_random", 220: "extra_detector_selected"}
    (tmp_path / "bad.json").write_text(json.dumps({"other": 1}))
    with pytest.raises(ValueError):
        afv.read_extra_frames(tmp_path / "bad.json")


def test_view_paths_keep_the_c10379_names_and_derive_the_rest() -> None:
    assert view_paths("static-c10379") == (
        Path("configs/qa/first_minute_review_anchors.json"),
        Path("runs/human-review-anchors-first-minute"),
        Path("docs/qa/first-minute-review-anchors.human-record.json"),
    )
    config, workspace, record = view_paths("static-c10119")
    assert config == Path("configs/qa/first_minute_review_anchors_static_c10119.json")
    assert workspace == Path("runs/human-review-anchors-first-minute-static-c10119")
    assert record == Path("docs/qa/first-minute-review-anchors-static-c10119.human-record.json")


def test_missing_record_for_a_view_fails_with_the_next_step(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no human review anchors exist for static-c10119"):
        load_mask_set(tmp_path / "nothing", view_id="static-c10119")


def test_committed_per_view_configs_match_the_clock_rules() -> None:
    """Every per-view config in configs/qa is what the mapper writes from the rules."""
    from battle.assembly101_clock_offset import load_clock_rules

    rules = load_clock_rules(ROOT, Path("configs/assembly101/clock_rules.json"))
    source_rule = rules.views["C10379"].clock_rule
    assert source_rule is not None
    for view in afv.DEFAULT_VIEWS:
        path = ROOT / view_paths(afv.view_id_for(view))[0]
        assert path.is_file(), path
        config = ReviewAnchorConfig.model_validate_json(path.read_text())
        rule = rules.views[view].clock_rule
        assert rule is not None
        assert config.frame_mapping is not None
        assert config.frame_mapping.analysis_frame_shift == afv.analysis_frame_shift(
            source_rule, rule
        )
        for frame in config.frames:
            if frame.origin != "mapped_anchor":
                continue
            assert frame.source_analysis_frame_index is not None
            mapped, residual = afv.mapped_frame(
                source_rule, rule, frame.source_analysis_frame_index
            )
            assert (frame.analysis_frame_index, frame.residual_pose_frames) == (mapped, residual)
        target_config = json.loads((ROOT / config.manual_seed_target_config).read_text())
        assert target_config["view_id"] == config.view_id
        assert target_config["base_g2_config"] == config.clip_config
