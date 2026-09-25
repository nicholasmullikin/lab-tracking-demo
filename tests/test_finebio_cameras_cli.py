"""``battle-finebio-cameras`` (p0-cameras): the pose decision rule, the day vote, the
committed per-trial configs, and the P03_01_01 regression against the preflight."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from conftest import require_artifact

from battle import finebio_cameras as fc
from battle import finebio_cameras_cli as cli

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs/finebio/cameras"
REFERENCE = ROOT / "tests/fixtures/finebio_preflight/rig_reference.json"
PREFLIGHT = ROOT / "runs/preflight-finebio-20260924"


def _info(shipped: float, pnp_rms: float | None, *, chosen: bool = True) -> dict:
    pnp = (
        None
        if pnp_rms is None
        else {"rvec": [0, 0, 0], "tvec": [0, 0, 90], "corner_rms_px": pnp_rms}
    )
    best = {"day": "221013", "camera_id": 4, "median_corner_rms_px": 2.0}
    if chosen:
        return {"best": best, "chosen_day": {"median_corner_rms_px": shipped, "pnp": pnp}}
    return {
        "best": best,
        "table": {"221013/cam4": shipped, "221109/cam4": 2.0},
        "pnp_pose": None if pnp is None else {"rvec": pnp["rvec"], "tvec": pnp["tvec"]},
        "pnp_corner_rms_px": pnp_rms,
    }


def test_pose_decision_keeps_shipped_within_the_gate_else_pnp_else_drops() -> None:
    gate = {"pnp_over_px": 10.0, "drop_over_px": 10.0}

    assert fc._pose_decision(_info(6.3, 0.8), "221013", **gate)[0] == "shipped"
    assert fc._pose_decision(_info(10.0, 0.8), "221013", **gate)[0] == "shipped"
    assert fc._pose_decision(_info(93.7, 0.7), "221013", **gate)[0] == "marker_pnp"
    assert fc._pose_decision(_info(30.0, 12.0), "221013", **gate)[0] == "dropped"
    assert fc._pose_decision(_info(30.0, None), "221013", **gate)[0] == "dropped"
    assert fc._pose_decision(_info(float("nan"), None), "221013", **gate)[0] == "dropped"


def test_pose_decision_gates_the_chosen_day_not_the_best_day_on_a_preflight_report() -> None:
    # The preflight mapping.json has no chosen_day block: the residual comes from its table
    # for the chosen day (T4 on P03: best day 221109 at 2.2 px, chosen day 221013 at 4.8 px).
    decision, shipped, pnp, pnp_rms = fc._pose_decision(
        _info(4.78, 0.76, chosen=False), "221013", pnp_over_px=10.0, drop_over_px=10.0
    )
    assert decision == "shipped" and shipped == pytest.approx(4.78)
    assert pnp is not None and pnp_rms == pytest.approx(0.76)
    decision, shipped, *_ = fc._pose_decision(
        _info(40.0, 0.76, chosen=False), "221013", pnp_over_px=10.0, drop_over_px=10.0
    )
    assert decision == "marker_pnp" and shipped == pytest.approx(40.0)


def test_decide_day_sums_the_chosen_cameras_residuals_over_days() -> None:
    views = {
        "T1": {
            "best": {"day": "A", "camera_id": 1, "median_corner_rms_px": 5.0},
            "same_camera_other_days": [{"day": "B", "median_corner_rms_px": 20.0}],
            "table": {"A/cam1": 5.0, "B/cam1": 20.0},
        },
        "T2": {
            "best": {"day": "B", "camera_id": 2, "median_corner_rms_px": 6.0},
            "same_camera_other_days": [{"day": "A", "median_corner_rms_px": 7.0}],
            "table": {"A/cam2": 7.0, "B/cam2": 6.0},
        },
    }

    decision = fc.decide_day(views, ["A", "B"])

    assert decision["chosen"] == "A"
    assert decision["by_summed_residual"] == [["A", 12.0], ["B", 26.0]]
    assert decision["weighted_vote"] == "A"


def test_cli_parsers_and_defaults() -> None:
    assert cli.parse_seconds("30,60,90") == [30.0, 60.0, 90.0]
    assert [fc.seconds_to_frame(s) for s in (30, 60, 90)] == [899, 1798, 2697]
    assert cli.parse_frames("5,1-3,10:16:5") == [1, 2, 3, 5, 10, 15]
    with pytest.raises(ValueError):
        cli.parse_frames("")
    with pytest.raises(ValueError):
        cli.parse_frames("1:10:0")
    assert str(cli.default_evidence_dir("P20_03_01")).startswith("runs/finebio-cameras-P20_03_01-")
    parser = cli.build_parser()
    args = parser.parse_args(["solve", "--trial", "P20_03_01", "--seconds", "30,60,90"])
    assert args.trial == "P20_03_01" and args.fpv_step == cli.DEFAULT_FPV_STEP
    assert args.func is cli.cmd_solve


@pytest.mark.parametrize("trial", ["P03_01_01", "P03_03_01"])
def test_committed_p03_configs_share_the_day_and_the_rig(trial: str) -> None:
    config = fc.read_camera_config(CONFIGS / f"{trial}.json")

    assert config.trial == trial and config.recording_day == "221013"
    assert {v: c.camera_id for v, c in config.fixed.items()} == {
        "T1": 1,
        "T2": 2,
        "T3": 3,
        "T4": 4,
        "T5": 6,
    }
    assert [c.provenance for c in config.fixed.values()] == ["shipped"] * 4 + ["marker_pnp"]
    assert all(c.marker_fit_residual_px < 10 for c in config.fixed.values())
    assert config.fixed["T5"].shipped_marker_residual_px > 90
    assert config.provenance["dropped_views"] == {}
    views = config.provenance["views"]
    # The residual recorded for a shipped pose is the chosen day's, not the best day's.
    assert views["T4"]["best_day"] == "221109"
    assert config.fixed["T4"].shipped_marker_residual_px == pytest.approx(
        views["T4"]["chosen_day_median_corner_rms_px"]
    )
    assert (
        config.fixed["T4"].shipped_marker_residual_px > views["T4"]["best_day_median_corner_rms_px"]
    )
    fpv = config.provenance["fpv_pose_check"]
    assert fpv["corner_rms_px"]["median"] < 1.5 and fpv["frames_over_gate"] <= 3
    assert "rows" not in fpv
    crosscheck = config.provenance["fpv_day_crosscheck"]
    assert min(crosscheck, key=lambda d: crosscheck[d]["median"]) == "221013"
    assert crosscheck["221013"]["median"] < 1.5
    assert all(r["median"] > 10 for d, r in crosscheck.items() if d != "221013")
    assert "\n".join(cli.summary_lines(config)).count("marker_pnp") == 1


def test_camera_6_agrees_between_the_two_p03_trials_solved_on_the_same_day() -> None:
    smoke = fc.cameras_from_config(fc.read_camera_config(CONFIGS / "P03_01_01.json"))
    trial = fc.cameras_from_config(fc.read_camera_config(CONFIGS / "P03_03_01.json"))

    for view in ("T1", "T2", "T3", "T4"):
        assert np.array_equal(smoke[view].rvec, trial[view].rvec), view
        assert np.array_equal(smoke[view].tvec, trial[view].tvec), view
    assert np.linalg.norm(smoke["T5"].centre - trial["T5"].centre) < 0.3
    bench = np.array([[0.0, 0.0, 0.0], [30.0, 20.0, 0.0], [-30.0, -20.0, 0.0]])
    assert np.linalg.norm(smoke["T5"].project(bench) - trial["T5"].project(bench), axis=1).max() < 1


def test_committed_p20_config_is_the_room_2_rig_on_day_221124() -> None:
    config = fc.read_camera_config(CONFIGS / "P20_03_01.json")

    assert config.recording_day == "221124" and config.frame_index_offset == 0
    assert {v: c.camera_id for v, c in config.fixed.items()} == {
        "T1": 1,
        "T2": 2,
        "T3": 3,
        "T4": 4,
        "T5": 6,
    }
    assert [c.provenance for c in config.fixed.values()] == ["marker_pnp"] * 4 + ["shipped"]
    assert all(c.marker_fit_residual_px < 10 for c in config.fixed.values())
    assert all(c.shipped_marker_residual_px > 10 for v, c in config.fixed.items() if v != "T5")
    assert config.provenance["dropped_views"] == {}
    cameras = fc.cameras_from_config(config)
    assert all(50 < -cam.centre[2] < 95 for cam in cameras.values())
    assert config.fpv.pose_frame_count == 6045 and config.fpv.valid_pose_fraction < 0.95
    crosscheck = config.provenance["fpv_day_crosscheck"]
    assert min(crosscheck, key=lambda d: crosscheck[d]["median"]) == "221124"
    assert crosscheck["221124"]["median"] < 2 and all(
        r["median"] > 20 for d, r in crosscheck.items() if d != "221124"
    )


@pytest.mark.real_data
def test_solving_p03_01_01_reproduces_the_preflight_and_the_committed_config() -> None:
    from battle.finebio_frames import read_frame, video_path

    require_artifact(fc.POSES / "third_person_camera_poses/221013")
    for view in fc.FIXED_VIEWS:
        require_artifact(video_path("P03_01_01", view))
    reference = json.loads(REFERENCE.read_text())
    frames = [fc.seconds_to_frame(s) for s in (30, 60, 90)]

    report, kept = fc.solve_mapping(
        "P03_01_01",
        frames,
        lambda v, f: read_frame(video_path("P03_01_01", v), f),
        seconds=[30, 60, 90],
    )

    assert report["day_decision"]["chosen"] == reference["day"]["chosen"] == "221013"
    for (day, total), (ref_day, ref_total) in zip(
        report["day_decision"]["by_summed_residual"][:3],
        reference["day"]["by_summed_residual_top3"],
    ):
        assert day == ref_day and total == pytest.approx(ref_total, abs=0.1)
    assert report["camera_permutation_ok"] and reference["camera_permutation_ok"]
    for view, ref in reference["views"].items():
        info = report["views"][view]
        assert info["best"]["camera_id"] == ref["camera_id"], view
        assert info["best"]["day"] == ref["day"], view
        assert info["best"]["median_corner_rms_px"] == pytest.approx(
            ref["shipped_median_corner_rms_px"], abs=0.1
        ), view
        assert info["pnp_corner_rms_px"] == pytest.approx(ref["pnp_corner_rms_px"], abs=0.1), view
        assert info["pnp_vs_shipped_cm"] == pytest.approx(ref["pnp_vs_shipped_cm"], abs=0.05), view
        assert info["markers_detected_per_frame"] == ref["markers_detected_per_frame"], view
        assert len(kept[view]) == 3
    if (PREFLIGHT / "mapping/mapping.json").exists():
        preflight = json.loads((PREFLIGHT / "mapping/mapping.json").read_text())
        for view, ref in preflight["views"].items():
            assert np.allclose(
                report["views"][view]["pnp_centre_cm"], ref["pnp_centre_cm"], atol=0.05
            ), view

    config = fc.camera_config_from_mapping(report, "P03_01_01")
    committed = fc.read_camera_config(CONFIGS / "P03_01_01.json")
    assert config.model_dump(exclude={"provenance"}) == committed.model_dump(exclude={"provenance"})
    assert config.provenance["views"] == committed.provenance["views"]
    assert [c.provenance for c in config.fixed.values()] == ["shipped"] * 4 + ["marker_pnp"]
