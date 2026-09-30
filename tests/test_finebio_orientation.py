"""The orientation vote: each cue, the log-odds scale, and the episode retrofit."""

from __future__ import annotations

import math

import numpy as np
import pytest

from battle.finebio_orientation import (
    CUE_COLOUR,
    CUE_GRAVITY,
    CUE_HAND,
    CUE_TAPER,
    CUE_TIP_BOX,
    DECAY_PER_FRAME,
    FLIP_MARGIN_LOG_ODDS,
    HAND_MARGIN_CM,
    HAND_REACH_CM,
    LOG_ODDS_SCALE,
    RESOLVE_LOG_ODDS,
    TAIL_CONFIDENCE,
    TIP_BOX_CONFIDENCE,
    CueVote,
    EpisodeAccumulator,
    OrientRecord,
    best_camera,
    colour_confidence,
    colour_cue,
    cue_agreement,
    frame_log_odds,
    gravity_cue,
    hand_cue,
    peak_signs,
    taper_camera,
    taper_cue,
    tip_box_cue,
)
from battle.finebio_stand import BENCH_UP

# A full shaft, so the length factor is 1 and the stated confidence is the contribution.
FULL = 400.0


def _ends(a, b) -> np.ndarray:
    return np.array([a, b], dtype=np.float64)


def _at_elevation(degrees: float) -> np.ndarray:
    """A unit segment whose upper end is endpoint 1. Up is BENCH_UP, so the upper end
    has the larger dot with it and the lower end (endpoint 0) is the tip."""
    angle = math.radians(degrees)
    # Direction from end 0 (lower) toward end 1 (upper) is +BENCH_UP * sin(angle)
    # plus a horizontal component.
    up = np.asarray(BENCH_UP, dtype=np.float64)
    horizontal = np.array([1.0, 0.0, 0.0])
    direction = math.cos(angle) * horizontal + math.sin(angle) * up
    return _ends([0.0, 0.0, 0.0], direction.tolist())


def test_the_scale_lets_one_tip_box_resolve_and_flip() -> None:
    """One number. A tip box at confidence 0.9 contributes LOG_ODDS_SCALE * 0.9.

    That contribution is above the resolve threshold, and from an accumulator parked
    at the opposite flip margin it still clears the margin the other way after one
    frame of decay.
    """
    contribution = LOG_ODDS_SCALE * TIP_BOX_CONFIDENCE
    assert contribution == pytest.approx(6.3)
    assert contribution > RESOLVE_LOG_ODDS
    assert contribution > FLIP_MARGIN_LOG_ODDS
    fresh = EpisodeAccumulator()
    fresh.add(contribution)
    assert fresh.resolved and fresh.sign == 1
    assert fresh.log_odds > FLIP_MARGIN_LOG_ODDS
    parked = EpisodeAccumulator()
    parked.log_odds = -FLIP_MARGIN_LOG_ODDS
    parked.sign = -1
    parked.add(contribution)
    assert parked.log_odds > FLIP_MARGIN_LOG_ODDS
    assert parked.sign == 1
    assert parked.resolved


def test_three_copies_of_a_wrong_taper_do_not_beat_one_colour_cue() -> None:
    """Three cameras saying the same wrong taper are one contribution, the best camera."""
    taper = taper_cue([(0, 0.8, FULL), (0, 0.8, FULL), (0, 0.8, FULL)])
    colour = colour_cue([(1, 0.9, FULL)])
    assert taper is not None and colour is not None
    assert taper.confidence == pytest.approx(0.8)
    total, per = frame_log_odds([taper, colour])
    assert total < 0
    assert abs(per[CUE_COLOUR]) > abs(per[CUE_TAPER])
    # Summing the three cameras would have let the taper win.
    assert 3 * 0.8 > 0.9


def test_three_different_cues_beat_one_strong_wrong_cue() -> None:
    wrong = CueVote(CUE_TIP_BOX, 0, TIP_BOX_CONFIDENCE)
    right = [
        CueVote(CUE_COLOUR, 1, 0.5),
        CueVote(CUE_TAPER, 1, 0.5),
        CueVote(CUE_HAND, 1, 0.5),
    ]
    total, per = frame_log_odds([wrong, *right])
    assert total < 0
    assert sum(abs(per[cue]) for cue in (CUE_COLOUR, CUE_TAPER, CUE_HAND)) > abs(per[CUE_TIP_BOX])


def test_a_short_shaft_votes_less_than_a_full_one() -> None:
    short = best_camera([(0, 1.0, 30.0)])
    full = best_camera([(1, 0.2, FULL)])
    assert short is not None and full is not None
    assert short[1] == pytest.approx(30.0 / FULL)
    assert full[1] > short[1]


def test_colour_confidence_is_the_hue_share_times_a_size_factor_clamped() -> None:
    assert colour_confidence(0.8, 25.0) == pytest.approx(0.8)
    assert colour_confidence(0.8, 10.0) == pytest.approx(0.8 * 10.0 / 25.0)
    assert colour_confidence(1.0, 100.0) == 1.0
    assert colour_confidence(1.2, 50.0) == 1.0
    assert 0.0 <= colour_confidence(2.0, 80.0) <= 1.0


def test_the_taper_names_the_narrow_end_and_skips_the_tail_on_the_8_channel() -> None:
    # Ratio 2.5 saturates the width confidence. Axis end 0 is wide.
    single = taper_camera(
        end_widths=(25.0, 10.0),
        tip_side=1,
        wide_is_tip=False,
        axis_residual_px=2.0,
        axis_to_track=(0, 1),
    )
    assert single == (1, 1.0)
    # The tail would name axis end 0. On the 8-channel it is not added, and the wide
    # end is the tip.
    eight = taper_camera(
        end_widths=(25.0, 10.0),
        tip_side=0,
        wide_is_tip=True,
        axis_residual_px=2.0,
        axis_to_track=(0, 1),
    )
    assert eight is not None
    assert eight[0] == 0
    assert eight[1] == pytest.approx(1.0)
    # A tail alone, no widths, on a single-channel class.
    tail_only = taper_camera(
        end_widths=None,
        tip_side=1,
        wide_is_tip=False,
        axis_residual_px=None,
        axis_to_track=(0, 1),
    )
    assert tail_only == (1, TAIL_CONFIDENCE)
    assert (
        taper_camera(
            end_widths=None,
            tip_side=1,
            wide_is_tip=True,
            axis_residual_px=None,
            axis_to_track=(0, 1),
        )
        is None
    )


def test_the_taper_abstains_when_the_mask_is_fragmented_or_the_signals_disagree() -> None:
    assert (
        taper_camera(
            end_widths=(25.0, 10.0),
            tip_side=1,
            wide_is_tip=False,
            axis_residual_px=25.1,
            axis_to_track=(0, 1),
        )
        is None
    )
    # Width says the narrow end (axis 1 -> track 1); the tail says axis 0 -> track 0.
    assert (
        taper_camera(
            end_widths=(25.0, 10.0),
            tip_side=0,
            wide_is_tip=False,
            axis_residual_px=1.0,
            axis_to_track=(0, 1),
        )
        is None
    )
    # Below the ratio gate, no width vote.
    assert (
        taper_camera(
            end_widths=(12.0, 10.0),
            tip_side=None,
            wide_is_tip=False,
            axis_residual_px=1.0,
            axis_to_track=(0, 1),
        )
        is None
    )


def test_the_hand_uses_the_15_cm_reach_and_the_3_cm_margin() -> None:
    ends = _ends([0.0, 0.0, 0.0], [30.0, 0.0, 0.0])
    on_the_butt = hand_cue(ends, [np.array([0.0, 0.0, 0.0])], both_ends_in_hand_box=False)
    assert on_the_butt is not None
    assert on_the_butt.tip_end == 1 and on_the_butt.confidence == pytest.approx(1.0)
    # 12 cm from end 0, 18 cm from end 1: inside the reach, margin met, confidence (15-12)/15.
    near = hand_cue(ends, [np.array([12.0, 0.0, 0.0])], both_ends_in_hand_box=False)
    assert near is not None
    assert near.tip_end == 1
    assert near.confidence == pytest.approx((HAND_REACH_CM - 12.0) / HAND_REACH_CM)
    # Difference under the 3 cm margin: no vote.
    assert hand_cue(ends, [np.array([14.0, 0.0, 0.0])], both_ends_in_hand_box=False) is None
    # Exactly at the reach, with the margin met (the other end is 25 cm away): confidence 0.
    longer = _ends([0.0, 0.0, 0.0], [40.0, 0.0, 0.0])
    assert (
        hand_cue(longer, [np.array([HAND_REACH_CM, 0.0, 0.0])], both_ends_in_hand_box=False) is None
    )
    # Beyond the reach of both ends.
    assert hand_cue(ends, [np.array([0.0, 100.0, 0.0])], both_ends_in_hand_box=False) is None
    # Both ends inside a hand box: zero, even with the hand on one end.
    assert hand_cue(ends, [np.array([0.0, 0.0, 0.0])], both_ends_in_hand_box=True) is None
    assert HAND_MARGIN_CM == 3.0 and HAND_REACH_CM == 15.0


def test_gravity_is_silent_under_45_degrees_and_names_the_lower_end() -> None:
    assert gravity_cue(_ends([0.0, 0.0, -1.0], [20.0, 0.0, -1.0])) is None
    assert gravity_cue(_at_elevation(30.0)) is None
    assert gravity_cue(_at_elevation(45.0)) is None
    above = gravity_cue(_at_elevation(46.0))
    assert above is not None and above.tip_end == 0
    assert above.confidence == pytest.approx((46.0 - 45.0) / 30.0)
    full = gravity_cue(_at_elevation(75.0))
    assert full is not None and full.confidence == pytest.approx(1.0) and full.tip_end == 0
    vertical = gravity_cue(_at_elevation(90.0))
    assert vertical is not None and vertical.confidence == pytest.approx(1.0)
    # One vertical frame resolves on its own: a single-view pipette standing up.
    total, per = frame_log_odds([vertical])
    acc = EpisodeAccumulator()
    acc.add(total)
    assert acc.resolved and acc.sign == 1
    assert per[CUE_GRAVITY] > RESOLVE_LOG_ODDS
    assert np.allclose(BENCH_UP, [0.0, 0.0, -1.0])


def test_five_frames_of_contrary_evidence_hold_and_thirty_yield() -> None:
    acc = EpisodeAccumulator()
    acc.add(LOG_ODDS_SCALE * TIP_BOX_CONFIDENCE)
    assert acc.sign == 1
    contrary = -1.0
    for _ in range(5):
        acc.add(contrary)
    assert acc.sign == 1
    for _ in range(25):
        acc.add(contrary)
    assert acc.sign == -1
    assert acc.log_odds < -FLIP_MARGIN_LOG_ODDS


def test_resume_carries_the_sign_and_the_flip_margin_still_applies() -> None:
    acc = EpisodeAccumulator()
    acc.add(LOG_ODDS_SCALE * TIP_BOX_CONFIDENCE)
    carried = acc.log_odds
    acc.resume()
    assert acc.episode == 1
    assert acc.log_odds == pytest.approx(carried)
    assert acc.sign == 1
    assert acc.decay_free == 0.0
    acc.add(-1.0)
    assert acc.sign == 1
    assert acc.log_odds == pytest.approx(carried * DECAY_PER_FRAME - 1.0)


def test_the_retrofit_writes_the_sign_at_the_peak_not_the_terminal_frame() -> None:
    """Sums 1, 2, -1. The peak is +2, so the episode's sign stays positive even though
    the terminal sum is negative."""
    records = [
        OrientRecord("pipette-1", frame, 0, contribution, True)
        for frame, contribution in ((0, 1.0), (1, 1.0), (2, -3.0))
    ]
    assert peak_signs(records)[("pipette-1", 0)] == 1
    # A peak that never clears 1.1 is not written back.
    weak = [OrientRecord("pipette-1", 0, 0, 0.4, True)]
    assert peak_signs(weak)[("pipette-1", 0)] == 0
    # A second episode does not inherit the first episode's decay-free sum.
    second = [
        *records,
        OrientRecord("pipette-1", 10, 1, -2.0, True),
    ]
    signs = peak_signs(second)
    assert signs[("pipette-1", 0)] == 1
    assert signs[("pipette-1", 1)] == -1


def test_cue_agreement_is_against_the_retrofit_sign() -> None:
    votes = {CUE_COLOUR: 2.0, CUE_TAPER: -1.0, CUE_HAND: 0.0, CUE_GRAVITY: 3.0}
    assert cue_agreement(votes, 1) == {
        CUE_COLOUR: "agree",
        CUE_TAPER: "differ",
        CUE_GRAVITY: "agree",
    }
    assert cue_agreement(votes, None) == {}


def test_an_ambiguous_plunger_sample_abstains_and_a_clear_one_is_clamped() -> None:
    from battle.finebio_orientation import plunger_end_from_sample

    assert plunger_end_from_sample(None, 30.0) is None
    ambiguous = plunger_end_from_sample(
        {"ambiguous": True, "confidence": 0.9, "tip_index": 1}, 40.0
    )
    assert ambiguous is not None and ambiguous["ambiguous"] is True and ambiguous["tip_end"] is None
    clear = plunger_end_from_sample({"ambiguous": False, "confidence": 0.8, "tip_index": 0}, 40.0)
    assert clear is not None and clear["tip_end"] == 0 and clear["confidence"] == pytest.approx(0.8)
    huge = plunger_end_from_sample({"ambiguous": False, "confidence": 1.0, "tip_index": 1}, 80.0)
    assert huge is not None and huge["confidence"] == 1.0


def test_one_contribution_per_cue_in_the_sum() -> None:
    total, per = frame_log_odds(
        [
            tip_box_cue([(0, FULL)]),
            colour_cue([(1, 0.4, FULL)]),
            gravity_cue(_at_elevation(90.0)),
        ]
    )
    assert set(per) == {CUE_TIP_BOX, CUE_COLOUR, CUE_GRAVITY}
    assert total == pytest.approx(sum(per.values()))


def test_retrofit_orders_all_rows_of_an_episode_without_changing_the_hold():
    from battle.multiview_schemas import Track3D
    from battle.multiview_tracks import orient_track_rows

    rows = [
        Track3D(
            frame_index=frame,
            track_id="pipette-1",
            object_class="pipette",
            position_cm=(0, 0, 0),
            uncertainty_cm=1,
            support_views=("T1",),
            state="held",
            confidence=0.8,
            abstain=False,
            endpoints_cm=((0, 0, 0), (20, 0, 0)),
            direction=(1, 0, 0),
            held_by="hand-1",
            tip_resolved=False,
        )
        for frame in range(3)
    ]
    records = [
        OrientRecord("pipette-1", frame, 0, value, True) for frame, value in enumerate([-2, -2, 1])
    ]
    oriented = orient_track_rows(rows, records)
    assert all(row.endpoints_cm == ((20, 0, 0), (0, 0, 0)) for row in oriented)
    assert all(row.direction == (-1, 0, 0) for row in oriented)
    assert all(row.tip_resolved and row.tip_basis == "vote" for row in oriented)
    assert all(row.state == "held" and row.held_by == "hand-1" for row in oriented)
    assert all(row.endpoints_cm == ((0, 0, 0), (20, 0, 0)) for row in rows)


def test_online_resolution_threshold_does_not_discard_the_carried_sign():
    from battle.multiview_tracks import MultiviewTracker, Track

    track = Track(
        track_id="pipette-1",
        object_class="pipette",
        position=np.zeros(3),
        uncertainty_cm=1,
        state="observed",
        born_frame=0,
        last_observed_frame=0,
        orient_log_odds=0.5,
        orient_sign=1,
    )
    fields = MultiviewTracker._orientation_row_fields(None, track, 0)
    assert fields["tip_resolved"] is False
    assert fields["tip_basis"] == "vote" and fields["orientation_online_sign"] == 1
    track.orient_log_odds = 2
    assert MultiviewTracker._orientation_row_fields(None, track, 0)["tip_resolved"] is True


def test_scoreboard_and_click_scorer_use_only_the_requested_sibling(tmp_path, monkeypatch):
    from battle import finebio_lines_scoreboard as board
    from battle import finebio_tips as tips

    directory = tmp_path / "v5a"
    directory.mkdir()
    online = directory / "tracks.jsonl"
    online.write_text("")
    assert board.tracks_path(directory) == online
    seen = []

    def read(path, frames):
        seen.append(path)
        return {}

    monkeypatch.setattr(tips, "read_track_rows", read)
    tips.score_tracks_file("online", online, [], repository_root=tmp_path)
    assert seen[-1] == online
    oriented = directory / "tracks_oriented.jsonl"
    oriented.write_text("")
    assert board.tracks_path(directory) == oriented
    tips.score_tracks_file("vote", online, [], repository_root=tmp_path)
    assert seen[-1] == oriented
    previous = tmp_path / "v4"
    previous.mkdir()
    old = previous / "tracks.jsonl"
    old.write_text("")
    assert board.tracks_path(previous) == old
    tips.score_tracks_file("previous", old, [], repository_root=tmp_path)
    assert seen[-1] == old


def test_orientation_scoreboard_counts_online_flips_against_the_retrofit():
    from battle.finebio_lines_scoreboard import orientation_summary
    from battle.multiview_schemas import Track3D

    base = Track3D(
        frame_index=0,
        track_id="pipette-1",
        object_class="pipette",
        position_cm=(0, 0, 0),
        uncertainty_cm=1,
        support_views=("T1",),
        state="observed",
        confidence=0.8,
        abstain=False,
        endpoints_cm=((0, 0, 0), (20, 0, 0)),
        orientation_episode=0,
        orientation_retrofit_sign=-1,
    )
    rows = [
        base.model_copy(
            update={
                "frame_index": frame,
                "orientation_online_sign": sign,
                "tip_confidence": 0.8,
                "tip_votes": {"colour": -2.0, "taper": 1.0},
            }
        )
        for frame, sign in enumerate([1, 1, -1])
    ]
    result = orientation_summary(rows)
    assert result["episodes"] == 1 and result["flips_within_episode"] == 1
    assert result["tip_confidence"]["median"] == 0.8
    assert result["cue_agreement"]["colour"] == {"agree": 3, "differ": 0, "agreement": 1.0}
    assert result["cue_agreement"]["taper"] == {"agree": 0, "differ": 3, "agreement": 0.0}
    assert orientation_summary([]) == {"enabled": False}


def test_the_online_half_life_counts_frames_without_observations():
    from battle.multiview_tracks import MultiviewTracker, Track

    acc = EpisodeAccumulator(log_odds=2, sign=1)
    acc.add(0, elapsed_frames=60)
    assert acc.log_odds == pytest.approx(1) and not acc.resolved
    assert acc.sign == 1
    track = Track(
        track_id="pipette-1",
        object_class="pipette",
        position=np.zeros(3),
        uncertainty_cm=1,
        state="held",
        born_frame=0,
        last_observed_frame=0,
        orient_log_odds=2,
        orient_sign=1,
        orient_voted_frame=0,
    )
    fields = MultiviewTracker._orientation_row_fields(None, track, 60)
    assert fields["tip_resolved"] is False
    assert fields["tip_confidence"] == pytest.approx(0.7311)
    assert fields["tip_votes"] is None and track.orient_log_odds == 2


def test_a_class_split_carries_the_clock_for_decay_through_a_gap():
    from battle.multiview_tracks import MultiviewTracker, Track, TrackerParams

    tracker = MultiviewTracker(
        {},
        lambda frame: None,
        TrackerParams(line_orientation="vote", line_classes=("blue_pipette",)),
    )
    track = Track(
        track_id="pipette-old",
        object_class="pipette",
        position=np.zeros(3),
        uncertainty_cm=1,
        state="observed",
        born_frame=0,
        last_observed_frame=10,
        orient_log_odds=2,
        orient_sign=1,
        orient_voted_frame=10,
    )
    new = tracker._class_split(track, "yellow_pipette", 10)
    assert new.orient_voted_frame == 10 and new.orient_sign == 1
    fields = tracker._orientation_row_fields(new, 70)
    assert fields["tip_resolved"] is False
    assert fields["tip_confidence"] == pytest.approx(0.7311)
