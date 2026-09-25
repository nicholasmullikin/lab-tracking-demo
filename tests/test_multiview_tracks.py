"""battle-multiview-tracks core (p3-tracker) on a synthetic rig with the P03 camera geometry,
and on the preflight fixtures against the slice's points and the preflight's residuals."""

from __future__ import annotations

import itertools
import json
from collections import defaultdict

import numpy as np
import pytest
from finebio_fixtures import FIXTURE_DIR, load_preflight_fixtures

from battle.finebio_slice import leave_one_out_residuals, run_slice
from battle.multiview_schemas import FineBioObservation, Track3D, TrackEvent, read_jsonl
from battle.multiview_tracks import (
    Gates,
    TrackerParams,
    hungarian,
    main,
    run_tracker,
    select_observations,
)

FIXED = ("T1", "T2", "T3", "T4", "T5")
WINDOW = list(range(1798, 1858))


@pytest.fixture(scope="module")
def fixtures():
    return load_preflight_fixtures()


@pytest.fixture(scope="module")
def rig(fixtures):
    cams = fixtures.fixed_cameras()
    fpv = fixtures.fpv_camera(1798)
    assert fpv is not None
    return cams, fpv


def _box_row(
    cams,
    view: str,
    frame: int,
    point,
    cls: str = "cell_culture_plate",
    *,
    size=(120.0, 80.0),
    offset=(0.0, 0.0),
    source: str = "detector",
    slot: str | None = None,
    score: float | None = 0.9,
    pose_valid: bool = True,
) -> FineBioObservation:
    pixel = cams[view].project(np.asarray(point, dtype=float))[0] + np.asarray(offset)
    w, h = size
    box = (
        float(pixel[0] - w / 2),
        float(pixel[1] - h / 2),
        float(pixel[0] + w / 2),
        float(pixel[1] + h / 2),
    )
    kwargs = dict(
        view=view,
        frame_index=frame,
        slot=slot or f"{cls}#0",
        object_class=cls,
        pose_valid=pose_valid,
        source=source,
    )
    if source == "detector":
        return FineBioObservation(box_xyxy_px=box, detector_score=score, **kwargs)
    return FineBioObservation(
        mask_bbox_px=box,
        mask_centroid_px=(float(pixel[0]), float(pixel[1])),
        mask_area_px=int(w * h / 2),
        sam3_object_score=0.8,
        detector_score=score,
        **kwargs,
    )


def _rows_for(cams, fpv, frames, point, views, cls="cell_culture_plate", **kw):
    all_cams = {**cams, "fpv": fpv}
    return [_box_row(all_cams, v, f, point, cls, **kw) for f in frames for v in views]


def _events(output, kind: str) -> list[TrackEvent]:
    return [e for e in output.events if e.kind == kind]


def _by_track(output) -> dict[str, list[Track3D]]:
    grouped: dict[str, list[Track3D]] = defaultdict(list)
    for row in output.rows:
        grouped[row.track_id].append(row)
    return dict(grouped)


P = TrackerParams(observation_source="detector", coast_timeout_frames=10, handoff_after_frames=3)


def test_hungarian_matches_brute_force() -> None:
    rng = np.random.default_rng(3)
    for n, m in ((3, 3), (2, 4), (4, 2), (5, 5)):
        cost = rng.random((n, m)) * 10
        pairs = hungarian(cost)
        assert len(pairs) == min(n, m)
        assert len({i for i, _ in pairs}) == len(pairs) == len({j for _, j in pairs})
        total = sum(cost[i, j] for i, j in pairs)
        best = (
            min(
                sum(cost[i, j] for i, j in zip(range(n), perm))
                for perm in itertools.permutations(range(m), n)
            )
            if n <= m
            else min(
                sum(cost[i, j] for i, j in zip(perm, range(m)))
                for perm in itertools.permutations(range(n), m)
            )
        )
        assert total == pytest.approx(best)
    assert hungarian(np.zeros((0, 3))) == []


def test_gates_from_rig_block_and_defaults() -> None:
    assert Gates() == Gates(30.0, 55.0, 3, 2)
    rig = {"gates": {"association_px": 24.5, "handoff_px": 48.0}}
    gates = Gates.from_rig(rig)
    assert gates.association_px == 24.5 and gates.handoff_px == 48.0
    assert gates.birth_min_fixed_views == 3 and gates.birth_fixed_views_with_fpv == 2
    assert Gates.from_rig({"association_px": 10}).association_px == 10


def test_birth_from_three_fixed_views(rig) -> None:
    cams, fpv = rig
    point = np.array([10.0, -5.0, -3.0])
    rows = _rows_for(cams, fpv, [0, 1, 2], point, ["T1", "T2", "T3"])
    out = run_tracker(rows, cams, lambda f: None, [0, 1, 2], P)
    tracks = _by_track(out)
    assert len(tracks) == 1
    rows_out = next(iter(tracks.values()))
    assert [r.state for r in rows_out] == ["observed"] * 3
    assert rows_out[0].support_views == ("T1", "T2", "T3")
    assert np.allclose(rows_out[-1].position_cm, point, atol=0.3)
    birth = _events(out, "birth")
    assert len(birth) == 1 and birth[0].payload["views"] == ["T1", "T2", "T3"]
    assert birth[0].payload["fpv_rule"] is False
    assert out.metrics["tracks_born"] == 1 and out.metrics["id_switches"] == 0
    assert all(r.confidence > 0.3 and not r.abstain for r in rows_out)


def test_birth_from_two_fixed_views_needs_a_valid_fpv(rig) -> None:
    cams, fpv = rig
    point = np.array([5.0, 20.0, -3.0])
    rows = _rows_for(cams, fpv, [0], point, ["T2", "T4", "fpv"])
    with_fpv = run_tracker(rows, cams, lambda f: fpv, [0], P)
    assert with_fpv.metrics["tracks_born"] == 1
    birth = _events(with_fpv, "birth")[0]
    assert birth.payload["fpv_rule"] is True and birth.payload["views"] == ["T2", "T4", "fpv"]
    assert np.allclose(with_fpv.rows[0].position_cm, point, atol=0.5)
    without_pose = run_tracker(rows, cams, lambda f: None, [0], P)
    assert without_pose.metrics["tracks_born"] == 0
    invalid_rows = [r for r in rows if r.view != "fpv"] + [
        _box_row({**cams, "fpv": fpv}, "fpv", 0, point, pose_valid=False)
    ]
    assert run_tracker(invalid_rows, cams, lambda f: fpv, [0], P).metrics["tracks_born"] == 0
    two_fixed_only = _rows_for(cams, fpv, [0], point, ["T2", "T4"])
    assert run_tracker(two_fixed_only, cams, lambda f: fpv, [0], P).metrics["tracks_born"] == 0


def test_false_pair_rejected_and_two_objects_assigned(rig) -> None:
    cams, fpv = rig
    a, b = np.array([10.0, -5.0, -3.0]), np.array([-20.0, 15.0, -3.0])
    # A is seen in T1 and T3, B in T2 and T4: no same-object pair exists.
    rows = _rows_for(cams, fpv, [0], a, ["T1", "T3"]) + _rows_for(cams, fpv, [0], b, ["T2", "T4"])
    assert run_tracker(rows, cams, lambda f: None, [0], P).metrics["tracks_born"] == 0
    # Both seen in T1..T4: pairwise assignment separates them and each is born once.
    rows = _rows_for(cams, fpv, [0, 1], a, FIXED[:4], slot="cell_culture_plate#0") + _rows_for(
        cams, fpv, [0, 1], b, FIXED[:4], slot="cell_culture_plate#1"
    )
    out = run_tracker(rows, cams, lambda f: None, [0, 1], P)
    assert out.metrics["tracks_born"] == 2
    positions = sorted(tuple(r.position_cm) for r in out.rows if r.frame_index == 1)
    assert np.allclose(positions[0], b, atol=0.3) and np.allclose(positions[1], a, atol=0.3)
    assert all(len(r.support_views) == 4 for r in out.rows)


def test_coasting_is_stationary_then_lost_and_a_new_id_is_born(rig) -> None:
    cams, fpv = rig
    point = np.array([10.0, -5.0, -3.0])
    frames = list(range(0, 30))
    rows = _rows_for(cams, fpv, [0, 1, 2, 3, 4], point, FIXED[:3])
    rows += _rows_for(cams, fpv, [25, 26], point, FIXED[:3])
    out = run_tracker(rows, cams, lambda f: None, frames, P)
    tracks = _by_track(out)
    first_id = next(t for t, rs in tracks.items() if rs[0].frame_index == 0)
    first = tracks[first_id]
    states = {r.frame_index: r.state for r in first}
    assert states[4] == "observed" and states[5] == "coasting"
    coasting = [r for r in first if r.state == "coasting"]
    assert all(r.position_cm == first[4].position_cm for r in coasting)
    assert all(r.abstain for r in coasting)
    uncertainties = [r.uncertainty_cm for r in coasting]
    assert uncertainties == sorted(uncertainties) and uncertainties[-1] > uncertainties[0]
    assert [r.frames_unobserved for r in coasting][:3] == [1, 2, 3]
    lost = _events(out, "lost")
    assert len(lost) == 1 and lost[0].track_id == first_id
    assert lost[0].frame_index == 4 + P.coast_timeout_frames + 1 == 15
    assert first[-1].state == "lost" and first[-1].frame_index == 15
    assert _events(out, "coasting")[0].frame_index == 5
    # After `lost`, the class is free: the return at 25 is a plain birth with a new id.
    second_id = next(t for t, rs in tracks.items() if rs[0].frame_index == 25)
    assert second_id != first_id and tracks[second_id][0].possibly_same_as == ()
    assert out.metrics["tracks_born"] == 2 and out.metrics["reacquisitions"] == 0
    assert out.metrics["per_class"]["cell_culture_plate"]["fragmentation"] == 1


def test_reacquisition_resumes_with_confirmation_and_refuses_without(rig) -> None:
    cams, fpv = rig
    point = np.array([10.0, -5.0, -3.0])
    frames = list(range(0, 9))
    seen = _rows_for(cams, fpv, [0, 1, 2], point, FIXED[:3])
    confirmed_return = _rows_for(cams, fpv, [7, 8], point, FIXED[:3])
    out = run_tracker(seen + confirmed_return, cams, lambda f: None, frames, P)
    assert out.metrics["tracks_born"] == 1 and out.metrics["reacquisitions"] == 1
    event = _events(out, "reacquired")[0]
    assert event.frame_index == 7 and event.payload["latency_frames"] == 5
    assert event.payload["confirmed_views"] == 3
    rows = _by_track(out)[event.track_id]
    assert {r.frame_index: r.state for r in rows}[7] == "observed"
    assert out.metrics["reacquisition_latency_frames"] == {"median": 5.0, "max": 5}

    params = TrackerParams(observation_source="auto", coast_timeout_frames=10)
    unconfirmed_return = _rows_for(
        cams, fpv, [7, 8], point, FIXED[:3], source="sam3_video", score=None
    )
    out = run_tracker(seen + unconfirmed_return, cams, lambda f: None, frames, params)
    assert out.metrics["reacquisitions"] == 0 and out.metrics["tracks_born"] == 2
    birth = _events(out, "birth")[1]
    old_id = _events(out, "birth")[0].track_id
    assert birth.payload["unconfirmed_reacquisition_of"] == old_id
    assert birth.payload["confirmed_views"] == 0
    new_rows = _by_track(out)[birth.track_id]
    assert new_rows[0].possibly_same_as == (old_id,) and new_rows[0].abstain is False
    assert _by_track(out)[old_id][-1].state == "coasting"

    # A SAM3 row is confirmed when a detector box of its class sits within the gate.
    confirmed_by_neighbour = unconfirmed_return + _rows_for(
        cams, fpv, [7, 8], point, FIXED[:3], offset=(6.0, 4.0), slot="cell_culture_plate#0"
    )
    out = run_tracker(seen + confirmed_by_neighbour, cams, lambda f: None, frames, params)
    assert out.metrics["reacquisitions"] == 1 and out.metrics["tracks_born"] == 1


def test_two_candidates_mean_nobody_resumes(rig) -> None:
    cams, fpv = rig
    point = np.array([10.0, -5.0, -3.0])
    other = point + np.array([1.5, 0.0, 0.0])
    frames = list(range(0, 8))
    seen = _rows_for(cams, fpv, [0, 1, 2], point, FIXED[:3])
    returns = _rows_for(cams, fpv, [7], point, FIXED[:3], slot="cell_culture_plate#0")
    returns += _rows_for(cams, fpv, [7], other, FIXED[:3], slot="cell_culture_plate#1")
    out = run_tracker(seen + returns, cams, lambda f: None, frames, P)
    assert out.metrics["reacquisitions"] == 0 and out.metrics["ambiguities"] == 1
    assert out.metrics["tracks_born"] == 3
    old_id = _events(out, "birth")[0].track_id
    ambiguous = _events(out, "ambiguous")
    assert len(ambiguous) == 2
    assert all(e.payload["coasting_tracks"] == [old_id] for e in ambiguous)
    new_ids = {e.track_id for e in ambiguous}
    for row in out.rows:
        if row.frame_index == 7 and row.track_id in new_ids:
            assert old_id in row.possibly_same_as
            # The two new tracks sit 1.5 cm apart: also flagged as near duplicates.
            assert (new_ids - {row.track_id}) <= set(row.possibly_same_as)
            assert row.abstain
    assert out.metrics["duplicate_pair_frames"] == 1


def test_handoff_reseed_emits_the_reprojected_box_or_the_detector_box(rig) -> None:
    cams, fpv = rig
    point = np.array([10.0, -5.0, -3.0])
    frames = list(range(0, 8))
    # A tighter stationary prior: the association gate settles near 38 px in these views and
    # the re-seed search gate near 63 px, so a box 48 px off is a re-seed, not a match.
    P = TrackerParams(
        observation_source="detector",
        coast_timeout_frames=10,
        handoff_after_frames=3,
        process_noise_cm=0.5,
    )
    rows = _rows_for(cams, fpv, [0], point, FIXED[:4], size=(90.0, 60.0))
    rows += _rows_for(cams, fpv, frames[1:], point, FIXED[:3], size=(90.0, 60.0))
    out = run_tracker(rows, cams, lambda f: None, frames, P)
    reseeds = _events(out, "handoff_reseed")
    # T4 saw the object at birth and then lost it; T5 never had it and is offered the object
    # with a default-sized box (the cross-camera hand-off case). Each fires once per cadence.
    assert [(e.frame_index, e.payload["view"]) for e in reseeds] == [(3, "T4"), (3, "T5")]
    assert reseeds[1].payload["box_from"] == "default"
    payload = reseeds[0].payload
    assert payload["provenance"] == "track_reproject"
    assert payload["box_from"] == "last_extent" and payload["frames_missing"] == 3
    expected = cams["T4"].project(point)[0]
    assert np.allclose(payload["point_px"], expected, atol=1.0)
    x0, y0, x1, y1 = payload["box_xyxy_px"]
    assert (x1 - x0, y1 - y0) == pytest.approx((90.0, 60.0))
    assert np.allclose([(x0 + x1) / 2, (y0 + y1) / 2], expected, atol=1.0)
    assert not _events(out, "detector_reseed")

    # A detector box of the class outside the association gate but inside the hand-off gate
    # in the missing view is offered instead, as a detector_reseed with its own box.
    rows_det = rows + _rows_for(cams, fpv, frames[1:], point, ["T4"], offset=(48.0, 0.0))
    out = run_tracker(rows_det, cams, lambda f: None, frames, P)
    det = _events(out, "detector_reseed")
    assert [(e.frame_index, e.payload["view"]) for e in det] == [(3, "T4")]
    assert det[0].payload["residual_px"] == pytest.approx(48, abs=1)
    assert det[0].payload["provenance"] == "detector_reseed"
    assert [e.payload["view"] for e in _events(out, "handoff_reseed")] == ["T5"]

    # A view that never had a valid pose or is off-image emits nothing.
    out = run_tracker(
        rows,
        cams,
        lambda f: None,
        frames,
        TrackerParams(observation_source="detector", handoff_after_frames=100),
    )
    assert not _events(out, "handoff_reseed")


def test_select_observations_source_rule_and_confirmation(rig) -> None:
    cams, fpv = rig
    point = np.array([10.0, -5.0, -3.0])
    rows = _rows_for(cams, fpv, [0], point, ["T1", "T2"])
    rows += _rows_for(cams, fpv, [0], point, ["T2"], source="sam3_video", score=None)
    rows += _rows_for(cams, fpv, [0], point, ["T3"], source="sam3_video", score=0.7)
    rows += _rows_for(cams, fpv, [0], point, ["T4"], source="sam3_video", score=None)
    rows += _rows_for(cams, fpv, [0], point, ["T4"], offset=(200.0, 0.0))
    auto = select_observations(0, rows, TrackerParams(observation_source="auto"))
    assert [o.source for o in auto.tracked["T1"]] == ["detector"]
    assert [o.source for o in auto.tracked["T2"]] == ["sam3_video"]
    assert auto.tracked["T2"][0].confirmed is True  # detector box within the gate
    assert auto.tracked["T3"][0].confirmed is True  # carries a detector score
    assert auto.tracked["T4"][0].confirmed is False  # detector box 200 px away
    assert len(auto.tracked["T4"]) == 1 and len(auto.detector["T4"]) == 1
    detector_only = select_observations(0, rows, TrackerParams(observation_source="detector"))
    assert set(detector_only.tracked) == {"T1", "T2", "T4"}
    sam3_only = select_observations(0, rows, TrackerParams(observation_source="sam3"))
    assert set(sam3_only.tracked) == {"T2", "T3", "T4"}


def test_preflight_fixtures_reproduce_the_slice_points_and_the_plate_track(fixtures) -> None:
    cams = fixtures.fixed_cameras()
    params = TrackerParams(observation_source="detector")
    out = run_tracker(fixtures.observations, cams, fixtures.fpv_camera, WINDOW, params)
    slice_result = run_slice(fixtures.observations, cams, fixtures.fpv_camera, WINDOW)
    tracks = _by_track(out)
    reference = fixtures.rig_reference
    distances = []
    for static in reference["static"]:
        cls = static["class"]
        slice_point = slice_result.median_point(cls)
        candidates = [rs for rs in tracks.values() if rs[0].object_class == cls and len(rs) >= 30]
        assert candidates, cls
        best = min(
            np.linalg.norm(np.median([r.position_cm for r in rs], axis=0) - slice_point)
            for rs in candidates
        )
        distances.append(best)
    assert np.median(distances) < 1.0 and max(distances) < 1.0
    # One plate id over the 60 consecutive frames, observed on every frame.
    plates = [rs for rs in tracks.values() if rs[0].object_class == "cell_culture_plate"]
    plate = max(plates, key=len)
    assert [r.frame_index for r in plate] == WINDOW
    assert all(r.state == "observed" for r in plate)
    assert np.median([len(r.support_views) for r in plate]) >= 5
    # Leave-one-view-out residuals of the plate's own support observations per view sit at
    # or under the preflight's 17-44 px medians (the tracker takes the nearest same-class
    # box inside its gate, the preflight the top-scoring one; in T2 that is a different box
    # with a smaller LOO).
    by_frame = fixtures.by_frame(source="detector")
    loo: dict[str, list[float]] = defaultdict(list)
    for row in plate:
        pixels = {}
        for view, slot in row.support_slots.items():
            if view == "fpv":
                continue
            obs = next(o for o in by_frame[row.frame_index][view] if o.slot == slot)
            pixels[view] = np.array(obs.point_px)
        if len(pixels) < 4:
            continue
        for view, value in leave_one_out_residuals(cams, pixels, list(pixels)).items():
            loo[view].append(value)
    for view in FIXED:
        expected = reference["loo_moving"]["cell_culture_plate"][view]["median_px"]
        assert 17 <= expected <= 44
        assert len(loo[view]) >= 50, view
        assert np.median(loo[view]) <= expected + 5.0, view
    assert max(np.median(loo[v]) for v in FIXED) > 17
    assert out.metrics["per_class"]["cell_culture_plate"]["tracks_born"] == 1
    assert out.metrics["id_switch_reference"] == "sam3_slots_proxy"


def test_cli_writes_the_four_outputs(tmp_path) -> None:
    out = tmp_path / "tracks"
    gates = tmp_path / "rig.json"
    gates.write_text(json.dumps({"gates": {"association_px": 32.0, "handoff_px": 50.0}}))
    assert (
        main(
            [
                "--fixtures",
                str(FIXTURE_DIR),
                "--output",
                str(out),
                "--frames",
                "1798:5",
                "--gates",
                str(gates),
                "--handoff-px",
                "58",
            ]
        )
        == 0
    )
    rows = list(read_jsonl(out / "tracks.jsonl", Track3D))
    events = list(read_jsonl(out / "events.jsonl", TrackEvent))
    assert rows and events and (out / "residuals.jsonl").stat().st_size > 0
    metrics = json.loads((out / "identity_metrics.json").read_text())
    assert metrics["params"]["gates"] == {
        "association_px": 32.0,
        "handoff_px": 58.0,
        "birth_min_fixed_views": 3,
        "birth_fixed_views_with_fpv": 2,
    }
    assert metrics["frames"] == {"count": 5, "first": 1798, "last": 1802}
    assert metrics["tracks_born"] == len({e.track_id for e in events if e.kind == "birth"})
    assert "cell_culture_plate" in metrics["residual_px_by_class_and_view"]
