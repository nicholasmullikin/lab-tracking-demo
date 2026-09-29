"""battle-multiview-tracks core (p3-tracker) on a synthetic rig with the P03 camera geometry,
and on the preflight fixtures against the slice's points and the preflight's residuals."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import defaultdict

import numpy as np
import pytest
from finebio_fixtures import FIXTURE_DIR, load_preflight_fixtures

from battle.finebio_slice import depth_cm, leave_one_out_residuals, run_slice
from battle.multiview_schemas import FineBioObservation, Track3D, TrackEvent, read_jsonl
from battle.multiview_tracks import (
    LINE_CLASS_SHORTHANDS,
    Gates,
    LinePrior,
    TrackerParams,
    build_container_volumes,
    convex_hull,
    hungarian,
    inside_convex_polygon,
    load_line_prior,
    main,
    parse_container_classes,
    parse_heights,
    parse_line_classes,
    ray_point,
    run_tracker,
    select_observations,
)

FIXED = ("T1", "T2", "T3", "T4", "T5")
WINDOW = list(range(1798, 1858))
# sha256 of the core's tracks / events / residuals on the fixtures window (1798:60) as written
# by the p3-tracker core before the extensions existed (commit ecd7b3d); the extensions are
# off by default and must not change a byte of them.
CORE_GOLDEN = {
    "auto": {
        "tracks": "85b0c365a66eeca4d6c9f5ca4ac1dae5f38dd00f62829a7eebb62e2da804651b",
        "events": "c736da64611c6c2f10c1008e1d4eb35ca8dee0528abebb74bcb405f149541021",
        "residuals": "240205cc631ea381f171e57b014a0c8cd49e3db054e71c514a6986085ad08c41",
    },
    "detector": {
        "tracks": "4af8027c4de9e7fcbc651bf38e3a1d32731c0075bca2862de183b8e8a97f83e8",
        "events": "3ee3b7fec79406d04d9f373f80fc0622d7c1eead5e0b6717d190b4f0425e4bca",
        "residuals": "ad5446f745cb9f75a85a431351b25a0f589541568f8ee7db6bb6b35b21c43ff0",
    },
}


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


# --------------------------------------------------------------------------- extensions


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_extensions_off_reproduce_the_core_byte_for_byte(tmp_path) -> None:
    """With every extension flag at its default the CLI writes the same bytes as the core did
    before the extensions existed (hashes recorded from the p3-tracker core on the fixtures),
    and no extension field appears in any row."""
    for source in ("auto", "detector"):
        out = tmp_path / source
        assert (
            main(
                [
                    "--fixtures",
                    str(FIXTURE_DIR),
                    "--output",
                    str(out),
                    "--frames",
                    "1798:60",
                    "--source",
                    source,
                ]
            )
            == 0
        )
        for name in ("tracks", "events", "residuals"):
            assert _sha256(out / f"{name}.jsonl") == CORE_GOLDEN[source][name], (source, name)
        text = (out / "tracks.jsonl").read_text()
        for key in ("group_size", "split_from", "container_id", "held_by"):
            assert f'"{key}"' not in text
        metrics = json.loads((out / "identity_metrics.json").read_text())
        assert metrics["extensions"]["enabled"] == {
            "motion_model": False,
            "containers": False,
            "group_tracks": False,
            "held": False,
        }
        assert metrics["params"]["extensions"]["motion_model"] is False
    assert not TrackerParams().any_extension


def _moving_rows(cams, fpv, frames, start, velocity, views, cls, **kw):
    rows = []
    for f in frames:
        point = np.asarray(start, dtype=float) + np.asarray(velocity, dtype=float) * (f - frames[0])
        rows += _rows_for(cams, fpv, [f], point, views, cls, **kw)
    return rows


def _oscillation(frames, centre, amplitude_cm=25.0, omega=0.25):
    """A raised object swung back and forth over the bench, starting at rest: peak speed
    `amplitude * omega` cm per frame (6.25 by default, 47-66 px per frame in T1..T3)."""
    return {
        f: np.asarray(centre, dtype=float)
        + np.array(
            [
                amplitude_cm * (1 - np.cos(omega * f)) - amplitude_cm,
                0.5 * amplitude_cm * (1 - np.cos(0.5 * omega * f)),
                3.0 * (1 - np.cos(omega * f)),
            ]
        )
        for f in frames
    }


def test_motion_model_keeps_a_mover_on_one_id_and_leaves_static_tracks_alone(rig) -> None:
    cams, fpv = rig
    frames = list(range(0, 60))
    static_point = np.array([-30.0, -5.0, -2.0])
    truth = _oscillation(frames, [0.0, -5.0, -8.0])
    rows = _rows_for(cams, fpv, frames, static_point, FIXED[:3], "cell_culture_plate")
    for f in frames:
        rows += _rows_for(cams, fpv, [f], truth[f], FIXED[:3], "blue_pipette")
    core = run_tracker(rows, cams, lambda f: None, frames, P)
    ext_params = TrackerParams(
        observation_source="detector",
        coast_timeout_frames=10,
        handoff_after_frames=3,
        motion_model=True,
    )
    ext = run_tracker(rows, cams, lambda f: None, frames, ext_params)
    # The core's stationary prior at a 30 px gate loses the swung object at its fast phases
    # (up to 6.25 cm per frame) and fragments it; the motion model keeps one id on every frame
    # within 1.2 cm of the truth, through the turning points.
    assert core.metrics["per_class"]["blue_pipette"]["tracks_born"] >= 4
    assert ext.metrics["per_class"]["blue_pipette"]["tracks_born"] == 1
    pipette = [r for r in ext.rows if r.object_class == "blue_pipette"]
    assert [r.frame_index for r in pipette] == frames
    assert all(r.state == "observed" for r in pipette)
    errors = [np.linalg.norm(np.array(r.position_cm) - truth[r.frame_index]) for r in pipette]
    assert max(errors) < 1.2 and np.median(errors) < 0.3
    motion = ext.metrics["extensions"]["motion_model"]
    assert motion["mover_tracks"] == 1 and motion["mover_tracks_by_class"] == {"blue_pipette": 1}
    assert motion["mover_frames"] > 40
    # The static plate is not a mover and its rows are unchanged to the last decimal.
    core_plate = [r for r in core.rows if r.object_class == "cell_culture_plate"]
    ext_plate = [r for r in ext.rows if r.object_class == "cell_culture_plate"]
    assert len(core_plate) == len(ext_plate) == len(frames)
    for a, b in zip(core_plate, ext_plate):
        assert a.position_cm == b.position_cm and a.uncertainty_cm == b.uncertainty_cm
        assert a.state == b.state and a.support_views == b.support_views
    assert np.linalg.norm(np.array(ext_plate[-1].position_cm) - static_point) < 0.1


def test_mover_single_view_update_follows_the_ray_and_the_static_case_does_not(rig) -> None:
    cams, fpv = rig
    frames = list(range(0, 30))
    start, velocity = np.array([10.0, -5.0, -6.0]), np.array([2.0, 0.5, 0.0])
    params = TrackerParams(
        observation_source="detector", coast_timeout_frames=10, motion_model=True
    )
    # Three views up to frame 19, then T1 alone: the mover keeps moving along T1's ray.
    rows = _moving_rows(cams, fpv, frames[:20], start, velocity, FIXED[:3], "blue_pipette")
    rows += _moving_rows(
        cams, fpv, frames[20:], start + velocity * 20, velocity, ["T1"], "blue_pipette"
    )
    out = run_tracker(rows, cams, lambda f: None, frames, params)
    pipette = {r.frame_index: r for r in out.rows if r.object_class == "blue_pipette"}
    assert out.metrics["per_class"]["blue_pipette"]["tracks_born"] == 1
    assert pipette[25].state == "single_view"
    truth_25 = start + velocity * 25
    pixel = cams["T1"].project(truth_25)[0]
    assert np.linalg.norm(cams["T1"].project(np.array(pipette[25].position_cm))[0] - pixel) < 3.0
    assert out.metrics["extensions"]["motion_model"]["single_view_ray_updates"] >= 5
    # ray_point itself: the closest point of the back-projected ray to a 3D point.
    near = np.array([12.0, -4.0, -6.0])
    on_ray = ray_point(cams["T2"], cams["T2"].project(near)[0], near + np.array([0.5, 0.5, 0.5]))
    assert np.linalg.norm(cams["T2"].project(on_ray)[0] - cams["T2"].project(near)[0]) < 0.5
    # A stationary object seen in one view keeps its position (the core's single_view rule).
    static = np.array([-30.0, -5.0, -2.0])
    rows = _rows_for(cams, fpv, frames[:5], static, FIXED[:3])
    rows += _rows_for(cams, fpv, frames[5:], static, ["T1"], offset=(40.0, 0.0))
    out = run_tracker(rows, cams, lambda f: None, frames, params)
    plate = [r for r in out.rows if r.object_class == "cell_culture_plate"]
    assert all(r.position_cm == plate[4].position_cm for r in plate[5:] if r.state == "single_view")


def _volume_box_rows(cams, frames, cls, centre_xy, half_xy, height, views):
    """Detector rows of a container class: the bounding box of its volume's projected corners
    in every view and frame (a stable box, as the bench objects give)."""
    cx, cy = centre_xy
    hx, hy = half_xy
    corners = np.array(
        [[x, y, z] for x in (cx - hx, cx + hx) for y in (cy - hy, cy + hy) for z in (0.0, -height)]
    )
    rows = []
    for view in views:
        pix = cams[view].project(corners)
        box = (
            float(pix[:, 0].min()),
            float(pix[:, 1].min()),
            float(pix[:, 0].max()),
            float(pix[:, 1].max()),
        )
        for f in frames:
            rows.append(
                FineBioObservation(
                    view=view,
                    frame_index=f,
                    slot=f"{cls}#0",
                    object_class=cls,
                    detector_score=0.9,
                    box_xyxy_px=box,
                    pose_valid=True,
                    source="detector",
                )
            )
    return rows


CONTAINER_XY, CONTAINER_HALF = (15.0, -15.0), (10.0, 10.0)


def test_container_volume_from_detector_boxes(rig) -> None:
    cams, _fpv = rig
    frames = list(range(0, 20))
    rows = _volume_box_rows(cams, frames, "centrifuge", CONTAINER_XY, CONTAINER_HALF, 12.0, FIXED)
    params = TrackerParams(containers=("centrifuge", "micro_tube_rack"))
    volumes, report = build_container_volumes(
        rows, cams, params.containers, params, {"centrifuge": (15.0, -15.0, -6.0)}
    )
    assert [v.container_id for v in volumes] == ["centrifuge"]
    assert "micro_tube_rack" in report["skipped"]
    (vol,) = volumes
    assert vol.height_cm == 12.0 and vol.views == FIXED and vol.rig_point_inside is True
    x0, y0, x1, y1 = vol.bounds_cm
    # The visual hull on the bench holds the true footprint and does not exceed it by much.
    assert x0 <= 5.0 and y0 <= -25.0 and x1 >= 25.0 and y1 >= -5.0
    assert (x1 - x0) < 32.0 and (y1 - y0) < 32.0
    assert vol.contains(np.array([15.0, -15.0, -5.0]))
    assert vol.contains(np.array([6.0, -24.0, 0.5]))  # just below the bench, within tolerance
    assert not vol.contains(np.array([15.0, -15.0, -13.0]))  # above the height
    assert not vol.contains(np.array([40.0, -15.0, -5.0]))
    # The projected volume polygon holds a point inside and not one far outside.
    inside_pixel = cams["T1"].project(np.array([15.0, -15.0, -5.0]))[0]
    outside_pixel = cams["T1"].project(np.array([-30.0, 10.0, -5.0]))[0]
    assert vol.pixel_inside(cams["T1"], inside_pixel)
    assert not vol.pixel_inside(cams["T1"], outside_pixel)
    hull = convex_hull(np.array([[0, 0], [2, 0], [1, 1], [2, 2], [0, 2], [1, 0.5]]))
    assert len(hull) == 4 and inside_convex_polygon(np.array([1.0, 1.0]), hull)
    assert not inside_convex_polygon(np.array([3.0, 1.0]), hull)
    assert parse_container_classes("centrifuge, vortex_mixer") == ("centrifuge", "vortex_mixer")
    heights = dict(parse_heights("centrifuge=14,foo=3"))
    assert (
        heights["centrifuge"] == 14.0
        and heights["foo"] == 3.0
        and heights["micro_tube_rack"] == 6.0
    )
    summary = vol.summary()
    assert summary["footprint_area_cm2"] == int(vol.cells.sum()) > 300


def test_contained_track_does_not_time_out_and_resumes_inside_the_volume(rig) -> None:
    cams, fpv = rig
    frames = list(range(0, 70))
    tube = np.array([15.0, -15.0, -5.0])  # inside the container, 5 cm above the bench
    away = np.array([-30.0, 5.0, -3.0])  # outside every container
    rows = _volume_box_rows(cams, frames, "centrifuge", CONTAINER_XY, CONTAINER_HALF, 12.0, FIXED)
    seen = frames[:10] + frames[60:]
    rows += _rows_for(cams, fpv, seen, tube, FIXED[:3], "50ml_tube", slot="50ml_tube#0")
    rows += _rows_for(cams, fpv, seen, away, FIXED[:3], "50ml_tube", slot="50ml_tube#1")
    core = run_tracker(rows, cams, lambda f: None, frames, P)
    assert core.metrics["per_class"]["50ml_tube"]["tracks_born"] == 4  # both lost and reborn
    params = TrackerParams(
        observation_source="detector",
        coast_timeout_frames=10,
        handoff_after_frames=3,
        containers=("centrifuge",),
    )
    ext = run_tracker(rows, cams, lambda f: None, frames, params)
    tracks = _by_track(ext)
    tubes = {tid: rs for tid, rs in tracks.items() if rs[0].object_class == "50ml_tube"}
    assert ext.metrics["per_class"]["50ml_tube"]["tracks_born"] == 3
    inside = next(rs for rs in tubes.values() if np.allclose(rs[0].position_cm, tube, atol=0.5))
    states = {r.frame_index: r.state for r in inside}
    assert states[9] == "observed" and states[10] == "contained" and states[59] == "contained"
    assert states[60] == "observed" and inside[-1].frame_index == frames[-1]
    assert all(r.container_id == "centrifuge" for r in inside if r.state == "contained")
    assert all(r.abstain for r in inside if r.state == "contained")
    contained_rows = [r for r in inside if r.state == "contained"]
    assert all(r.position_cm == inside[9].position_cm for r in contained_rows)
    events = {(e.kind, e.track_id) for e in ext.events}
    tid = inside[0].track_id
    assert ("contained", tid) in events and ("lost", tid) not in events
    reacquired = [e for e in _events(ext, "reacquired") if e.track_id == tid]
    assert len(reacquired) == 1 and reacquired[0].payload["from_state"] == "contained"
    assert reacquired[0].payload["latency_frames"] == 51
    # The tube outside every volume coasts and is lost as in the core.
    outside = next(rs for rs in tubes.values() if np.allclose(rs[0].position_cm, away, atol=0.5))
    assert outside[-1].state == "lost" and outside[-1].frame_index == 9 + 10 + 1
    ext_metrics = ext.metrics["extensions"]["contained"]
    assert ext_metrics["episodes"] == 1 and ext_metrics["by_container"] == {"centrifuge": 1}
    assert ext_metrics["reacquired"] == 1 and ext_metrics["live_at_end"] == 0
    assert ext_metrics["volume_report"]["built"][0]["container_id"] == "centrifuge"
    # A container is never contained in itself: the centrifuge's own track coasts.
    centrifuge = next(rs for rs in tracks.values() if rs[0].object_class == "centrifuge")
    assert all(r.state != "contained" for r in centrifuge)
    # With a contained timeout the ghost is bounded.
    bounded = run_tracker(
        rows,
        cams,
        lambda f: None,
        frames,
        TrackerParams(
            observation_source="detector",
            coast_timeout_frames=10,
            containers=("centrifuge",),
            contained_timeout_frames=20,
        ),
    )
    lost = [e for e in _events(bounded, "lost") if e.payload.get("from_state") == "contained"]
    assert len(lost) == 1 and lost[0].frame_index == 9 + 20 + 1


def test_group_track_in_a_footprint_with_a_split_and_a_gate_group(rig) -> None:
    cams, fpv = rig
    frames = list(range(0, 40))
    rows = _volume_box_rows(
        cams, frames, "micro_tube_rack", CONTAINER_XY, CONTAINER_HALF, 6.0, FIXED
    )
    # Four tubes 1.5 cm apart in the rack; the fourth leaves at frame 15 and sits 20 cm away.
    tubes = [np.array([12.0 + 1.5 * k, -15.0, -3.0]) for k in range(4)]
    for k, tube in enumerate(tubes[:3]):
        rows += _rows_for(
            cams, fpv, frames, tube, FIXED, "micro_tube", slot=f"micro_tube#{k}", size=(20, 30)
        )
    rows += _rows_for(
        cams, fpv, frames[:15], tubes[3], FIXED, "micro_tube", slot="micro_tube#3", size=(20, 30)
    )
    far = tubes[3] + np.array([0.0, 20.0, 0.0])
    rows += _rows_for(
        cams, fpv, frames[20:], far, FIXED, "micro_tube", slot="micro_tube#3", size=(20, 30)
    )
    params = TrackerParams(
        observation_source="detector",
        coast_timeout_frames=10,
        containers=("micro_tube_rack",),
        group_tracks=True,
    )
    core = run_tracker(rows, cams, lambda f: None, frames, P)
    ext = run_tracker(rows, cams, lambda f: None, frames, params)
    assert core.metrics["per_class"]["micro_tube"]["tracks_born"] >= 5
    assert core.metrics["duplicate_pair_frames"] > 50
    assert ext.metrics["per_class"]["micro_tube"]["tracks_born"] == 2
    assert ext.metrics["duplicate_pair_frames"] == 0 and ext.metrics["ambiguities"] == 0
    tracks = _by_track(ext)
    group_id = _events(ext, "group_formed")[0].track_id
    group = tracks[group_id]
    assert group[0].frame_index == 0 and group[0].group_size == 4
    assert group[0].container_id is None and group[0].state == "observed"
    assert all(r.group_size == 4 for r in group[:15])
    assert all(r.group_size == 3 for r in group[25:])
    assert group[-1].frame_index == frames[-1]
    x0, y0, x1, y1 = ext.metrics["extensions"]["contained"]["volumes"][0]["bounds_cm"]
    assert x0 <= group[0].position_cm[0] <= x1 and y0 <= group[0].position_cm[1] <= y1
    split = _events(ext, "group_split")
    assert len(split) == 1 and split[0].payload["split_from"] == group_id
    member = tracks[split[0].track_id]
    assert member[0].split_from == group_id and member[0].frame_index == 20
    assert np.allclose(member[0].position_cm, far, atol=0.5)
    assert member[-1].state == "observed" and member[-1].split_from == group_id
    groups = ext.metrics["extensions"]["group_tracks"]
    assert groups["footprint_groups"] == 1 and groups["splits"] == 1
    assert groups["max_group_size_by_class"] == {"micro_tube": 4}
    assert groups["observations_absorbed"] > 0
    # Two tubes 1.5 cm apart on the open bench: one candidate with a member count, no
    # near-duplicate ids (the gate rule, no container needed).
    pair_rows = []
    for k in range(2):
        pair_rows += _rows_for(
            cams,
            fpv,
            frames[:10],
            np.array([-20.0 + 1.5 * k, 5.0, -3.0]),
            FIXED[:3],
            "micro_tube",
            slot=f"micro_tube#{k}",
            size=(20, 30),
        )
    pair = run_tracker(
        pair_rows,
        cams,
        lambda f: None,
        frames[:10],
        TrackerParams(observation_source="detector", group_tracks=True),
    )
    assert pair.metrics["per_class"]["micro_tube"]["tracks_born"] == 1
    rows_out = pair.rows
    assert rows_out[0].group_size == 2 and all(r.group_size == 2 for r in rows_out)
    assert pair.metrics["extensions"]["group_tracks"]["candidates_merged"] == 1
    assert pair.metrics["duplicate_pair_frames"] == 0
    # A class outside `group_classes` is untouched by the group rule.
    plain = run_tracker(
        pair_rows,
        cams,
        lambda f: None,
        frames[:10],
        TrackerParams(observation_source="detector", group_tracks=True, group_classes=("pen",)),
    )
    assert plain.metrics["per_class"]["micro_tube"]["tracks_born"] == 2


def test_held_track_follows_the_hand_and_resumes(rig) -> None:
    cams, fpv = rig
    frames = list(range(0, 25))
    hand_start, hand_velocity = np.array([0.0, 0.0, -8.0]), np.array([1.0, 0.5, 0.0])
    tube_offset = np.array([2.0, 0.0, -1.0])
    rows = _moving_rows(
        cams, fpv, frames, hand_start, hand_velocity, FIXED[:4], "left_hand", size=(220, 220)
    )
    # The tube sits 2 cm from the hand, is seen up to frame 9, vanishes inside the hand box
    # while the hand keeps moving, and reappears at the hand's new position at frame 20.
    rows += _moving_rows(
        cams, fpv, frames[:10], hand_start + tube_offset, hand_velocity, FIXED[:3], "50ml_tube"
    )
    rows += _moving_rows(
        cams,
        fpv,
        frames[20:],
        hand_start + hand_velocity * 20 + tube_offset,
        hand_velocity,
        FIXED[:3],
        "50ml_tube",
    )
    core = run_tracker(rows, cams, lambda f: None, frames, P)
    assert core.metrics["per_class"]["50ml_tube"]["tracks_born"] == 2
    params = TrackerParams(
        observation_source="detector", coast_timeout_frames=15, handoff_after_frames=3, held=True
    )
    ext = run_tracker(rows, cams, lambda f: None, frames, params)
    assert ext.metrics["per_class"]["50ml_tube"]["tracks_born"] == 1
    tube = [r for r in ext.rows if r.object_class == "50ml_tube"]
    states = {r.frame_index: r.state for r in tube}
    assert states[9] == "observed" and states[10] == "held" and states[19] == "held"
    assert states[20] == "observed"
    hand_id = next(r.track_id for r in ext.rows if r.object_class == "left_hand")
    held_rows = [r for r in tube if r.state == "held"]
    assert all(r.held_by == hand_id for r in held_rows)
    # The held position moves with the hand (about 1.1 cm per frame) and lands near the truth.
    moved = np.linalg.norm(np.array(held_rows[-1].position_cm) - np.array(held_rows[0].position_cm))
    assert moved > 5.0
    truth_19 = hand_start + hand_velocity * 19 + tube_offset
    assert np.linalg.norm(np.array(held_rows[-1].position_cm) - truth_19) < 3.0
    held_event = [e for e in _events(ext, "held") if e.track_id == tube[0].track_id]
    assert len(held_event) == 1 and held_event[0].payload["hand_class"] == "left_hand"
    assert len(held_event[0].payload["views"]) >= 2
    reacquired = [e for e in _events(ext, "reacquired") if e.track_id == tube[0].track_id]
    assert len(reacquired) == 1 and reacquired[0].payload["from_state"] == "held"
    held_metrics = ext.metrics["extensions"]["held"]
    assert held_metrics["episodes"] == 1 and held_metrics["reacquired"] == 1
    assert held_metrics["by_hand"] == {"left_hand": 1}
    # Without a hand box in >= 2 views the same loss is plain coasting.
    no_hand = [r for r in rows if r.object_class != "left_hand"]
    plain = run_tracker(no_hand, cams, lambda f: None, frames, params)
    assert not _events(plain, "held") and _events(plain, "coasting")


# --------------------------------------------------------------------------- lines (p2)

PRIOR = load_line_prior("configs/finebio/pipettes.json")
LINE_P = dict(observation_source="auto", coast_timeout_frames=10, motion_model=True)


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _segment(mid, direction, length: float) -> tuple[np.ndarray, np.ndarray]:
    mid, direction = np.asarray(mid, dtype=float), _unit(direction)
    return mid - direction * length / 2, mid + direction * length / 2


def _axis_row(
    cams,
    view: str,
    frame: int,
    a,
    b,
    cls: str,
    *,
    rng,
    visible=(0.0, 1.0),
    width_cm: float = 2.0,
    width_scale: float = 1.0,
    residual_px: float = 5.0,
    noise_px: float = 0.5,
    slot: str | None = None,
) -> FineBioObservation:
    """A SAM3 row of the part `visible` of segment a->b as `test_multiview_lines` case (i)
    builds its views: the projected axis endpoints plus noise, the centroid of the visible
    part, the elongation and width a `width_cm` shaft shows at that depth."""
    cam = cams[view]
    lo, hi = visible
    p0, p1 = a + lo * (b - a), a + hi * (b - a)
    pixels = cam.project(np.stack([p0, p1])) + rng.normal(0.0, noise_px, (2, 2))
    mid = 0.5 * (p0 + p1)
    width_px = width_cm * cam.K[0, 0] / depth_cm(cam, mid) * width_scale
    length_px = float(np.linalg.norm(pixels[1] - pixels[0]))
    centroid = cam.project(mid)[0] + rng.normal(0.0, noise_px, 2)
    x0, y0 = pixels.min(axis=0) - width_px / 2
    x1, y1 = pixels.max(axis=0) + width_px / 2
    box = (float(x0), float(y0), float(x1), float(y1))
    return FineBioObservation(
        view=view,
        frame_index=frame,
        slot=slot or f"{cls}#0",
        object_class=cls,
        detector_score=0.8,
        box_xyxy_px=box,
        mask_bbox_px=box,
        mask_centroid_px=(float(centroid[0]), float(centroid[1])),
        mask_area_px=int(length_px * width_px),
        mask_axis_px=(
            (float(pixels[0, 0]), float(pixels[0, 1])),
            (float(pixels[1, 0]), float(pixels[1, 1])),
        ),
        mask_elongation=float(max(1.0, length_px / width_px)),
        mask_width_px=float(width_px),
        mask_axis_residual_px=float(residual_px),
        sam3_object_score=0.9,
        pose_valid=True,
        source="sam3_decode",
    )


def _endpoint_error(endpoints, a, b) -> float:
    e = np.asarray(endpoints, dtype=float)
    same = max(np.linalg.norm(e[0] - a), np.linalg.norm(e[1] - b))
    swapped = max(np.linalg.norm(e[0] - b), np.linalg.norm(e[1] - a))
    return float(min(same, swapped))


def _lines_metrics(out) -> dict:
    return out.metrics["extensions"]["lines"]


def test_line_classes_flag_parsing_and_prior() -> None:
    assert parse_line_classes("pipette") == LINE_CLASS_SHORTHANDS["pipette"]
    assert parse_line_classes("blue_pipette, red_pipette") == ("blue_pipette", "red_pipette")
    assert parse_line_classes(None) == () and parse_line_classes("") == ()
    assert TrackerParams().line_classes == () and not TrackerParams().any_extension
    assert TrackerParams(line_classes=("blue_pipette",)).any_extension
    assert PRIOR.length_cm == pytest.approx(23.21) and PRIOR.spread_cm == pytest.approx(2.49)
    # Class medians where the stand measured a class (the median over trials), else shared.
    assert PRIOR.length_for("blue_pipette") == pytest.approx(21.7)
    assert PRIOR.length_for("red_pipette") == pytest.approx(24.46)
    assert PRIOR.length_for("yellow_pipette") == pytest.approx(0.5 * (21.97 + 22.77))
    assert PRIOR.length_for("8_channel_pipette") == PRIOR.length_for(None) == PRIOR.length_cm


def test_b_moving_pipette_seen_as_different_portions_is_one_line_track(rig) -> None:
    """A blue pipette carried 2 cm per frame across the bench, re-gripped every 15 frames:
    for three frames every view sees the whole shaft, then T1 sees the top 45%, T4 the bottom
    45% and the head camera the tip, nothing else (`test_multiview_lines` case (i) portions).
    The line tracker keeps one id with a line on every multi-view frame; the point tracker
    on the same rows sits on the visible centroids, 6 cm off the shaft's middle, loses its
    3D support on most gripped frames and, without the motion model, fragments."""
    cams, fpv = rig
    all_cams = {**cams, "fpv": fpv}
    rng = np.random.default_rng(2)
    frames = list(range(60))
    length = PRIOR.length_for("blue_pipette")
    direction = _unit([0.3, 0.2, -1.0])

    def mid_at(f: int) -> np.ndarray:
        # Back and forth along y at x = 25 (inside all six frames): 2 cm per frame.
        phase = (2.0 * f) % 70.0
        return np.array([25.0, -10.0 + (phase if phase <= 35.0 else 70.0 - phase), -14.0])

    rows = []
    for f in frames:
        a, b = _segment(mid_at(f), direction, length)
        if f % 15 < 3:
            seen = {v: (0.0, 1.0) for v in ("T1", "T2", "T4", "fpv")}
        else:
            seen = {"T1": (0.0, 0.45), "T4": (0.55, 1.0), "fpv": (0.65, 1.0)}
        for v, vis in seen.items():
            rows.append(_axis_row(all_cams, v, f, a, b, "blue_pipette", rng=rng, visible=vis))
    gripped = [f for f in frames if f % 15 >= 3]
    line = run_tracker(
        rows, cams, lambda f: fpv, frames, TrackerParams(**LINE_P, line_classes=("blue_pipette",))
    )
    assert line.metrics["per_class"]["pipette"]["tracks_born"] == 1
    assert line.metrics["per_class"]["blue_pipette"]["tracks_born"] == 1
    pipette = [r for r in line.rows if r.object_class == "pipette"]
    assert [r.frame_index for r in pipette] == frames
    assert all(r.observed_class == "blue_pipette" for r in pipette)
    assert sum(1 for r in pipette if r.state == "observed") >= 55
    assert all(r.endpoints_cm is not None and r.direction is not None for r in pipette)
    errors = [
        _endpoint_error(r.endpoints_cm, *_segment(mid_at(r.frame_index), direction, length))
        for r in pipette
    ]
    assert np.median(errors) < 3.5 and max(errors) < 12.0
    mid_errors = [
        np.linalg.norm(np.array(r.position_cm) - mid_at(r.frame_index))
        for r in pipette
        if r.frame_index in gripped
    ]
    assert np.median(mid_errors) < 3.0
    lengths = [np.linalg.norm(np.subtract(*r.endpoints_cm)) for r in pipette]
    assert np.median(lengths) == pytest.approx(length, abs=0.5)
    metrics = _lines_metrics(line)
    assert metrics["enabled"] and metrics["line_births"] == 1 and metrics["point_births"] == 0
    assert metrics["frames"]["line_fraction"] == 1.0 and metrics["frames"]["point_fallback"] == 0
    # The gripped frames leave one plane (T1) and two rays to the tip: the prediction-aided
    # fit carries them; the full-view frames measure the direction again.
    assert metrics["frames"]["line_prediction_aided"] >= 20
    assert metrics["frames"]["line"] >= 55
    assert metrics["loo_residual_px"]["n"] > 0 and metrics["loo_residual_px"]["median"] < 3.0
    assert metrics["class_agreement"] == 1.0
    # The point tracker on the same rows: same params, no line classes.
    point = run_tracker(rows, cams, lambda f: fpv, frames, TrackerParams(**LINE_P))
    point_rows = [r for r in point.rows if r.object_class == "blue_pipette"]
    assert sum(1 for r in point_rows if r.state == "observed") < 45
    point_errors = [
        np.linalg.norm(np.array(r.position_cm) - mid_at(r.frame_index))
        for r in point_rows
        if r.frame_index in gripped
    ]
    assert np.median(point_errors) > 4.0
    stationary = run_tracker(
        rows,
        cams,
        lambda f: fpv,
        frames,
        TrackerParams(observation_source="auto", coast_timeout_frames=10),
    )
    assert stationary.metrics["per_class"]["blue_pipette"]["tracks_born"] >= 2
    # The row-level fields round-trip through JSON with the tracker's writer.
    text = pipette[10].model_dump_json(exclude_none=True)
    back = Track3D.model_validate_json(text)
    assert back.endpoints_cm == pipette[10].endpoints_cm and back.observed_class == "blue_pipette"


def test_b2_geometric_point_track_mover_seen_in_one_view_takes_the_ray_update(rig) -> None:
    """A pipette every camera sees as a compact blob (no axis in any view) is born as a point
    track of the geometric class; carried 3 cm a frame it becomes a mover, and when one view
    alone sees it the update is the core's lateral ray step, not the line one (the Sep 28
    trial-1 run hit the line path's endpoint assertion here)."""
    cams, _fpv = rig
    rng = np.random.default_rng(5)
    frames = list(range(16))
    rows = []
    for f in frames:
        mid = np.array([25.0, -10.0 + 3.0 * f, -14.0])
        views = ("T1", "T2", "T4") if f < 10 else ("T1",)
        for v in views:
            centroid = cams[v].project(mid)[0] + rng.normal(0.0, 0.5, 2)
            box = (
                float(centroid[0] - 12),
                float(centroid[1] - 12),
                float(centroid[0] + 12),
                float(centroid[1] + 12),
            )
            rows.append(
                FineBioObservation(
                    view=v,
                    frame_index=f,
                    slot="blue_pipette#0",
                    object_class="blue_pipette",
                    detector_score=0.8,
                    box_xyxy_px=box,
                    mask_bbox_px=box,
                    mask_centroid_px=(float(centroid[0]), float(centroid[1])),
                    mask_area_px=400,
                    mask_elongation=1.1,
                    mask_width_px=20.0,
                    sam3_object_score=0.9,
                    pose_valid=True,
                    source="sam3_decode",
                )
            )
    out = run_tracker(
        rows, cams, lambda f: None, frames, TrackerParams(**LINE_P, line_classes=("blue_pipette",))
    )
    pipette = [r for r in out.rows if r.object_class == "pipette"]
    assert pipette and all(r.endpoints_cm is None for r in pipette)
    assert _lines_metrics(out)["point_births"] == 1 and _lines_metrics(out)["line_births"] == 0
    single = [r for r in pipette if r.state == "single_view"]
    assert len(single) >= 3 and out.metrics["extensions"]["motion_model"]["single_view_ray_updates"]
    assert out.metrics["per_class"]["pipette"]["tracks_born"] == 1


def test_c_two_flat_parallel_pipettes_stay_two_tracks_and_a_merged_mask_is_dropped(rig) -> None:
    cams, _fpv = rig
    rng = np.random.default_rng(3)
    frames = list(range(20))
    la, lb = PRIOR.length_for("yellow_pipette"), PRIOR.length_for("red_pipette")
    # Flat on the bench (1 cm up), along x, 3 cm apart in y: the stand's spacing is 2.9 cm.
    a0, a1 = _segment([0.0, -10.0, -1.0], [1.0, 0.0, 0.0], la)
    b0, b1 = _segment([0.0, -7.0, -1.0], [1.0, 0.0, 0.0], lb)
    rows = []
    for f in frames:
        for v in FIXED:
            if f == 12 and v == "T3":
                # T3's yellow mask covers both pipettes: twice the width, 1.9x the extent, and
                # no separate red mask in that view.
                rows.append(
                    _axis_row(
                        cams,
                        v,
                        f,
                        a0,
                        a1,
                        "yellow_pipette",
                        rng=rng,
                        visible=(-0.45, 1.45),
                        width_scale=2.0,
                    )
                )
                continue
            rows.append(_axis_row(cams, v, f, a0, a1, "yellow_pipette", rng=rng))
            rows.append(_axis_row(cams, v, f, b0, b1, "red_pipette", rng=rng))
    params = TrackerParams(
        observation_source="auto",
        coast_timeout_frames=10,
        line_classes=("yellow_pipette", "red_pipette"),
    )
    out = run_tracker(rows, cams, lambda f: None, frames, params)
    assert out.metrics["per_class"]["pipette"]["tracks_born"] == 2
    assert out.metrics["per_class"]["yellow_pipette"]["tracks_born"] == 1
    assert out.metrics["per_class"]["red_pipette"]["tracks_born"] == 1
    # 3 cm apart is not a near duplicate for two lines (the stand's spacing).
    assert out.metrics["duplicate_pair_frames"] == 0 and out.metrics["ambiguities"] == 0
    assert all(r.possibly_same_as == () for r in out.rows)
    by_frame = defaultdict(dict)
    for r in out.rows:
        by_frame[r.frame_index][r.observed_class] = r
    for f in frames:
        assert set(by_frame[f]) == {"yellow_pipette", "red_pipette"}
        assert by_frame[f]["yellow_pipette"].state == "observed"
        assert by_frame[f]["red_pipette"].state == "observed"
    yellow_12 = by_frame[12]["yellow_pipette"]
    assert yellow_12.merged_views == ("T3",) and "T3" not in yellow_12.support_views
    assert yellow_12.support_views == ("T1", "T2", "T4", "T5")
    assert by_frame[11]["yellow_pipette"].merged_views is None
    assert by_frame[12]["red_pipette"].support_views == ("T1", "T2", "T4", "T5")
    assert by_frame[13]["yellow_pipette"].support_views == FIXED
    metrics = _lines_metrics(out)
    assert metrics["merged_views_by_view"] == {"T3": 1}
    assert metrics["frames"]["line_fraction"] == 1.0 and metrics["class_agreement"] == 1.0
    for f in (11, 12, 19):
        assert _endpoint_error(by_frame[f]["yellow_pipette"].endpoints_cm, a0, a1) < 0.5
        assert _endpoint_error(by_frame[f]["red_pipette"].endpoints_cm, b0, b1) < 0.5
    # The two lines lie flat: about 90 degrees from the bench normal.
    for r in out.rows:
        assert abs(np.degrees(np.arccos(abs(r.direction[2])))) > 88.0


def test_d_tip_and_butt_resolved_by_a_hand_track_and_held_at_the_butt(rig) -> None:
    cams, _fpv = rig
    rng = np.random.default_rng(4)
    frames = list(range(30))
    length = PRIOR.length_for("red_pipette")
    tip = np.array([-10.0, 5.0, -3.0])
    direction = _unit([0.5, 0.3, -1.0])
    butt = tip + direction * length
    hand = butt + np.array([0.0, 0.0, 1.0])  # the hand around the plunger end
    rows = []
    for f in frames:
        for v in FIXED[:4]:
            rows.append(_box_row(cams, v, f, hand, "right_hand", size=(220, 220)))
        if f < 15 or f >= 25:
            for v in FIXED:
                rows.append(_axis_row(cams, v, f, tip, butt, "red_pipette", rng=rng))
    params = TrackerParams(
        observation_source="auto",
        coast_timeout_frames=15,
        held=True,
        line_classes=("red_pipette",),
    )
    out = run_tracker(rows, cams, lambda f: None, frames, params)
    assert out.metrics["per_class"]["pipette"]["tracks_born"] == 1
    pipette = {r.frame_index: r for r in out.rows if r.object_class == "pipette"}
    assert set(pipette) == set(frames)
    # Frame 0 is born unresolved; from frame 1 the hand track at the butt names the ends, and
    # endpoints_cm[0] is the tip from then on.
    assert pipette[0].tip_resolved is False
    for f in frames[1:]:
        assert pipette[f].tip_resolved is True, f
        assert np.linalg.norm(np.array(pipette[f].endpoints_cm[0]) - tip) < 1.0, f
        assert np.linalg.norm(np.array(pipette[f].endpoints_cm[1]) - butt) < 1.0, f
        assert np.dot(pipette[f].direction, direction) > 0.99
    # Occluded from frame 15 with the hand box over it in four views: held, following the
    # hand at the butt; the segment keeps its direction and length; resumed at 25.
    assert pipette[14].state == "observed" and pipette[15].state == "held"
    assert pipette[24].state == "held" and pipette[25].state == "observed"
    hand_id = next(r.track_id for r in out.rows if r.object_class == "right_hand")
    assert all(pipette[f].held_by == hand_id for f in range(15, 25))
    held = out.metrics["extensions"]["held"]
    assert held["episodes"] == 1 and held["reacquired"] == 1
    metrics = _lines_metrics(out)
    assert metrics["tip_resolved_fraction"] == pytest.approx(29 / 30, abs=1e-4)
    assert metrics["tip_resolutions_by_basis"] == {"hand_track": 1}
    # Without the hand the ends stay in the order the geometry gave, unresolved.
    no_hand = [r for r in rows if r.object_class != "right_hand"]
    plain = run_tracker(no_hand, cams, lambda f: None, frames, params)
    plain_rows = [r for r in plain.rows if r.object_class == "pipette"]
    assert all(r.tip_resolved is False for r in plain_rows)
    assert _lines_metrics(plain)["tip_resolved_fraction"] == 0.0


def test_e_soft_prior_completes_a_truncated_extent_and_reports_the_deviation(rig) -> None:
    cams, _fpv = rig
    rng = np.random.default_rng(5)
    frames = list(range(10))
    # The two-state prior (Sep 29) completes a bare body to `bare_for`, the no-tip rest mode.
    length = PRIOR.bare_for("yellow_pipette")
    a, b = _segment([5.0, 0.0, -12.0], [0.2, 0.1, -1.0], length)
    # Every view sees the middle 60%: the visible extent is 0.6 of the prior, both ends
    # equally supported, so the prior completes both ends by half.
    rows = [
        _axis_row(cams, v, f, a, b, "yellow_pipette", rng=rng, visible=(0.2, 0.8))
        for f in frames
        for v in FIXED
    ]
    params = TrackerParams(observation_source="auto", line_classes=("yellow_pipette",))
    out = run_tracker(rows, cams, lambda f: None, frames, params)
    assert out.metrics["per_class"]["pipette"]["tracks_born"] == 1
    for r in out.rows:
        e = np.asarray(r.endpoints_cm)
        assert np.linalg.norm(e[1] - e[0]) == pytest.approx(length, abs=0.5)
        assert _endpoint_error(e, a, b) < 0.5
    metrics = _lines_metrics(out)
    assert metrics["length_cm"]["visible"]["median"] == pytest.approx(0.6 * length, abs=0.5)
    assert metrics["length_cm"]["deviation_from_prior"]["median"] == pytest.approx(
        0.4 * length, abs=0.5
    )
    assert metrics["frames"]["extended_by_prior"] == len(frames)
    assert _events(out, "birth")[0].payload["extended_by_prior"] is True
    # A hand at one end names the tip; the truncated extent is then completed at the tip
    # end, not split between both ends.
    hand_rows = [
        _box_row(cams, v, f, b, "left_hand", size=(220, 220)) for f in frames for v in FIXED[:4]
    ]
    held = run_tracker(
        rows + hand_rows,
        cams,
        lambda f: None,
        frames,
        TrackerParams(observation_source="auto", held=True, line_classes=("yellow_pipette",)),
    )
    resolved = [r for r in held.rows if r.object_class == "pipette" and r.tip_resolved]
    assert len(resolved) >= len(frames) - 1
    visible_butt_end = a + 0.8 * (b - a)
    # Born symmetric before the hand named the ends; the Kalman blend (gain about 0.8 a
    # frame) moves the state onto the tip-side completion within a few frames.
    for r in resolved:
        if r.frame_index < 4:
            continue
        e = np.asarray(r.endpoints_cm)
        assert np.linalg.norm(e[1] - e[0]) == pytest.approx(length, abs=0.5)
        # The butt end stays where the masks end (0.8), the tip is reconstructed L away.
        assert np.linalg.norm(e[1] - visible_butt_end) < 0.7, r.frame_index
        assert np.linalg.norm(e[0] - (visible_butt_end - _unit(b - a) * length)) < 0.7
    # A full extent within the spread is left as seen and not flagged.
    full = run_tracker(
        [_axis_row(cams, v, f, a, b, "yellow_pipette", rng=rng) for f in frames for v in FIXED],
        cams,
        lambda f: None,
        frames,
        params,
    )
    assert _lines_metrics(full)["frames"]["extended_by_prior"] == 0
    assert _lines_metrics(full)["length_cm"]["deviation_from_prior"]["median"] < 0.5


def test_f_class_votes_give_the_plurality_and_the_agreement(rig) -> None:
    cams, _fpv = rig
    rng = np.random.default_rng(6)
    frames = list(range(6))
    a, b = _segment([5.0, 0.0, -12.0], [0.2, 0.1, -1.0], 22.0)
    rows = [
        _axis_row(cams, v, f, a, b, cls, rng=rng)
        for f in frames
        for v, cls in (("T1", "blue_pipette"), ("T3", "blue_pipette"), ("T2", "red_pipette"))
    ]
    params = TrackerParams(observation_source="auto", line_classes=("blue_pipette", "red_pipette"))
    out = run_tracker(rows, cams, lambda f: None, frames, params)
    assert out.metrics["per_class"]["pipette"]["tracks_born"] == 1
    assert out.metrics["per_class"]["blue_pipette"]["tracks_born"] == 1
    assert "red_pipette" not in out.metrics["per_class"]
    rows_out = [r for r in out.rows if r.object_class == "pipette"]
    assert all(r.observed_class == "blue_pipette" for r in rows_out)
    assert all(r.support_views == ("T1", "T2", "T3") for r in rows_out)
    metrics = _lines_metrics(out)
    assert metrics["class_agreement"] == pytest.approx(2 / 3, abs=1e-4)
    (votes,) = metrics["class_votes_by_track"].values()
    assert votes == {"blue_pipette": 12, "red_pipette": 6}
    # A track id carries the geometric class, not a colour.
    assert rows_out[0].track_id.startswith("pipette-")


def test_g_track3d_line_fields_round_trip_and_old_rows_validate(tmp_path) -> None:
    row = Track3D(
        frame_index=3,
        track_id="pipette-001",
        object_class="pipette",
        position_cm=(1.0, 2.0, -3.0),
        uncertainty_cm=1.0,
        state="observed",
        confidence=0.8,
        abstain=False,
        direction=(0.0, 0.0, -1.0),
        endpoints_cm=((1.0, 2.0, 8.0), (1.0, 2.0, -14.0)),
        tip_resolved=True,
        observed_class="blue_pipette",
        line_residual_px={"T1": 0.4, "T4": 1.2},
        merged_views=("T3",),
        colour_identity="blue_pipette",
        colour_confidence=0.9,
    )
    back = Track3D.model_validate_json(row.model_dump_json(exclude_none=True))
    assert back == row
    old = {
        "schema_version": "1.0",
        "frame_index": 0,
        "track_id": "blue_pipette-001",
        "object_class": "blue_pipette",
        "position_cm": [0.0, 0.0, -1.0],
        "uncertainty_cm": 1.0,
        "state": "observed",
        "confidence": 0.5,
        "abstain": False,
    }
    old_row = Track3D.model_validate(old)
    assert old_row.direction is None and old_row.endpoints_cm is None
    assert old_row.observed_class is None and old_row.colour_identity is None
    assert old_row.merged_views is None and old_row.line_residual_px is None
    with pytest.raises(ValueError):
        Track3D.model_validate({**old, "colour_confidence": 1.5})
    # The CLI accepts the shorthand and writes the fields; with the flag off no line field
    # appears (the golden test above holds the bytes).
    out = tmp_path / "lines"
    assert (
        main(
            [
                "--fixtures",
                str(FIXTURE_DIR),
                "--output",
                str(out),
                "--frames",
                "1798:5",
                "--motion-model",
                "--held",
                "--line-classes",
                "pipette",
            ]
        )
        == 0
    )
    metrics = json.loads((out / "identity_metrics.json").read_text())
    assert metrics["params"]["extensions"]["line_classes"] == list(LINE_CLASS_SHORTHANDS["pipette"])
    lines = metrics["extensions"]["lines"]
    assert lines["enabled"] is True and lines["prior"]["length_cm"] == pytest.approx(23.21)
    assert metrics["extensions"]["enabled"] == {
        "motion_model": True,
        "containers": False,
        "group_tracks": False,
        "held": True,
    }
    rows = list(read_jsonl(out / "tracks.jsonl", Track3D))
    assert rows and all(r.object_class not in LINE_CLASS_SHORTHANDS["pipette"] for r in rows)
    geometric = [r for r in rows if r.object_class == "pipette"]
    assert geometric and all(
        r.observed_class in LINE_CLASS_SHORTHANDS["pipette"] for r in geometric
    )
    assert "pipette" in metrics["per_class"]


# --------------------------------------------------------------------------- Sep 29 defects
#
# Found on the first full run of --line-classes (runs/finebio-lines-*-20260928): every runaway
# extent began in a prediction-aided fit whose in-plane direction nobody checked, the held
# offset was measured from a stale prediction, and T2's blue slot joined the red pipette's
# track. Each test here fails on the Sep 28 tracker and passes on the corrected one.


def _blob_row(cams, view: str, frame: int, point, cls: str, *, rng, slot: str | None = None):
    """A compact SAM3 row (no usable axis): the view looks down the shaft."""
    centroid = cams[view].project(np.asarray(point, dtype=float))[0] + rng.normal(0.0, 0.5, 2)
    box = (
        float(centroid[0] - 12),
        float(centroid[1] - 12),
        float(centroid[0] + 12),
        float(centroid[1] + 12),
    )
    return FineBioObservation(
        view=view,
        frame_index=frame,
        slot=slot or f"{cls}#0",
        object_class=cls,
        detector_score=0.8,
        box_xyxy_px=box,
        mask_bbox_px=box,
        mask_centroid_px=(float(centroid[0]), float(centroid[1])),
        mask_area_px=400,
        mask_elongation=1.1,
        mask_width_px=20.0,
        sam3_object_score=0.9,
        pose_valid=True,
        source="sam3_decode",
    )


def _row_length(r) -> float:
    return float(np.linalg.norm(np.subtract(*r.endpoints_cm)))


def test_h_aided_fit_with_a_wrong_prediction_no_longer_writes_a_runaway_extent(rig) -> None:
    """The real failure (trial 1, pipette-047 at 1370, pipette-025 at 1105): a pipette
    pointing nearly at a camera (14 deg off its ray), then seen by that axis view plus one
    compact view only while it turns in the one way the axis view cannot see, within the
    plane through the camera and the shaft. The axis still lies on the predicted line's
    projection, so it is associated; the aided fit took the predicted direction, 60 deg from
    the observed axis within the plane, and the plane's endpoint rays met that near-grazing
    line four times a pipette away (257 and 292 cm written on the real rows, a 130 cm
    midpoint jump). Now the aided fit is rejected on its own plane's extent, the centroids
    carry the segment as a point, its length stays a pipette's and every step stays under
    the cap."""
    cams, _fpv = rig
    rng = np.random.default_rng(29)
    frames = list(range(18))
    length = PRIOR.length_for("blue_pipette")
    mid = np.array([25.0, -10.0, -14.0])
    ray = _unit(mid - cams["T4"].centre)
    across = _unit(np.cross(ray, [0.0, 0.0, 1.0]))
    before = _unit(np.cos(np.radians(14.0)) * ray + np.sin(np.radians(14.0)) * across)
    after = _unit(np.cos(np.radians(74.0)) * ray + np.sin(np.radians(74.0)) * across)
    rows = []
    for f in frames:
        direction = before if f < 6 else after
        a, b = _segment(mid, direction, length)
        if f < 6:
            for v in ("T1", "T2", "T4", "T5"):
                rows.append(_axis_row(cams, v, f, a, b, "blue_pipette", rng=rng))
        else:
            rows.append(_axis_row(cams, "T4", f, a, b, "blue_pipette", rng=rng))
            rows.append(_blob_row(cams, "T3", f, mid, "blue_pipette", rng=rng))
    params = TrackerParams(**LINE_P, line_classes=("blue_pipette",))
    out = run_tracker(rows, cams, lambda f: None, frames, params)
    pipette = [r for r in out.rows if r.object_class == "pipette"]
    assert out.metrics["per_class"]["pipette"]["tracks_born"] == 1
    assert [r.frame_index for r in pipette] == frames
    # The turned axis is still associated in T4 (it lies on the prediction's projection).
    assert all("T4" in r.support_views for r in pipette[6:])
    lengths = [_row_length(r) for r in pipette]
    assert max(lengths) <= 1.2 * length + 0.5, max(lengths)
    steps = [
        np.linalg.norm(np.subtract(b.position_cm, a.position_cm))
        for a, b in zip(pipette, pipette[1:])
        if a.state in ("observed", "single_view") and b.state in ("observed", "single_view")
    ]
    assert max(steps) <= params.line_max_step_cm + 1e-6, max(steps)
    metrics = _lines_metrics(out)
    assert metrics["frames"]["aided_rejected"] >= 6
    assert metrics["frames"]["line_prediction_aided"] == 0
    for r in pipette[6:]:
        assert r.state in ("observed", "single_view"), r.frame_index
        assert r.line_update in ("point", "predicted", "capped", "lateral"), r.frame_index
        assert r.extent_clamped is False


def test_i_two_axis_views_must_overlap_along_the_line_to_fit(rig) -> None:
    """Two planes always meet in a line, so their fit carries no residual: the first run
    accepted a T2 axis and a T4 axis of two different pipettes lying along one line when the
    union of their extents fitted a pipette. Births refused that pair (`_line_pair_cost`);
    updates now do too."""
    from battle.multiview_tracks import Obs, fit_line_members

    cams, _fpv = rig
    rng = np.random.default_rng(31)
    length = PRIOR.length_cm
    a, b = _segment([10.0, 0.0, -8.0], [1.0, 0.3, -0.1], length)

    def member(view: str, visible: tuple[float, float]) -> Obs:
        row = _axis_row(cams, view, 0, a, b, "yellow_pipette", rng=rng, visible=visible)
        return Obs(
            view=view,
            point=np.array(row.point_px, dtype=float),
            object_class="pipette",
            slot=row.slot,
            source=row.source,
            detector_score=row.detector_score,
            sam3_score=row.sam3_object_score,
            box=row.mask_bbox_px,
            confirmed=True,
            axis_px=row.mask_axis_px,
            elongation=row.mask_elongation,
            width_px=row.mask_width_px,
            axis_residual_px=row.mask_axis_residual_px,
            colour_class="yellow_pipette",
        )

    params = TrackerParams(observation_source="auto", line_classes=("yellow_pipette",))
    # T1 sees the first 40% of the shaft, T4 a stretch 35% further along the same line: the
    # union (1.15 L) fits under the merged factor, the gap (0.35 L = 8 cm) does not fit one
    # pipette.
    apart = {"T1": member("T1", (0.0, 0.4)), "T4": member("T4", (0.75, 1.15))}
    fit, merged = fit_line_members(apart, cams, params, length, PRIOR.spread_cm)
    assert fit is None and merged == ()
    # Two portions of one shaft that overlap or nearly touch fit as before.
    touching = {"T1": member("T1", (0.0, 0.4)), "T4": member("T4", (0.45, 0.85))}
    fit, _ = fit_line_members(touching, cams, params, length, PRIOR.spread_cm)
    assert fit is not None and not fit.clamped
    # The two portions span 85% of the shaft; the prior completes the rest by halves.
    assert fit.visible_length_cm == pytest.approx(0.85 * length, abs=1.0)
    assert np.linalg.norm(fit.endpoints[1] - fit.endpoints[0]) == pytest.approx(length, abs=0.5)
    assert _endpoint_error(fit.endpoints, a, b) < 2.5


def test_j_single_view_update_is_refused_when_the_view_looks_along_the_line(rig) -> None:
    """A single-view lateral update projects the segment onto the plane through the camera
    and the axis. When the predicted line runs along the camera's ray the plane fixes nothing
    along the shaft and the ends slide; when the predicted direction is near the plane's
    normal the projection collapses the segment. Both keep the prediction now."""
    from battle.multiview_lines import plane_from_axis
    from battle.multiview_tracks import MultiviewTracker, Track, _obs

    cams, _fpv = rig
    rng = np.random.default_rng(37)
    params = TrackerParams(**LINE_P, line_classes=("blue_pipette",))
    tracker = MultiviewTracker(cams, lambda f: None, params)
    mid = np.array([25.0, -10.0, -14.0])
    length = PRIOR.length_for("blue_pipette")
    # The observation: a shaft seen broadside in T2, 1 cm above the predicted midpoint.
    seen_a, seen_b = _segment(mid + np.array([0.0, 0.0, -1.0]), [0.0, 1.0, 0.0], length)
    obs = _obs(_axis_row(cams, "T2", 0, seen_a, seen_b, "blue_pipette", rng=rng))
    obs.colour_class, obs.object_class = obs.object_class, "pipette"

    def track(direction) -> Track:
        t = Track("pipette-001", "pipette", mid.copy(), 1.0, "observed", 0, 0, mover=True)
        t.set_endpoints(np.stack(_segment(mid, direction, length)))
        return t

    along = track(cams["T2"].centre - mid)
    kept = along.endpoints_cm.copy()
    assert tracker._lateral_line_update(along, cams["T2"], obs) is False
    assert np.allclose(along.endpoints_cm, kept)
    normal, _ = plane_from_axis(cams["T2"], np.asarray(obs.axis_px, dtype=float))
    crossing = track(normal)
    kept = crossing.endpoints_cm.copy()
    assert tracker._lateral_line_update(crossing, cams["T2"], obs) is False
    assert np.allclose(crossing.endpoints_cm, kept)
    broadside = track([0.0, 1.0, 0.0])
    assert tracker._lateral_line_update(broadside, cams["T2"], obs) is True
    # Moved onto the observed plane, length kept, and by about the 1 cm offset.
    feet = broadside.endpoints_cm @ normal - float(normal @ cams["T2"].centre)
    assert np.abs(feet).max() < 1e-6
    assert _row_length_of(broadside) == pytest.approx(length, abs=1e-6)
    assert 0.3 < np.linalg.norm(broadside.position - mid) < 1.5


def _row_length_of(t) -> float:
    return float(np.linalg.norm(t.endpoints_cm[1] - t.endpoints_cm[0]))


def test_k_midpoint_step_is_capped_and_the_row_demoted(rig) -> None:
    from battle.multiview_lines import Line3D
    from battle.multiview_tracks import LineFit, MultiviewTracker, Track

    cams, _fpv = rig
    params = TrackerParams(**LINE_P, line_classes=("blue_pipette",))
    tracker = MultiviewTracker(cams, lambda f: None, params)
    mid = np.array([25.0, -10.0, -14.0])
    length = PRIOR.length_for("blue_pipette")
    direction = _unit([0.3, 0.2, -1.0])
    t = Track("pipette-001", "pipette", mid.copy(), 1.0, "observed", 0, 0)
    t.set_endpoints(np.stack(_segment(mid, direction, length)))
    t.class_votes["blue_pipette"] = 5
    far = np.stack(_segment(mid + np.array([0.0, 40.0, 0.0]), direction, length))
    members = {"T1": None, "T4": None}  # only the view names are read here
    fit = LineFit(
        line=Line3D(point=far.mean(axis=0), direction=direction, endpoints=far),
        endpoints=far,
        members=members,
        residuals={"T1": 1.0, "T4": 1.0},
        merged_views=(),
        visible_length_cm=length,
        extended=False,
        plane_views=("T1", "T4"),
    )
    predicted = t.position.copy()
    tracker._apply_line_fit(t, fit, cams, 1, predicted=predicted, dt=1)
    # The gain (about 0.5) would have moved the midpoint 20 cm; the cap holds it at 15.
    assert np.linalg.norm(t.position - predicted) == pytest.approx(params.line_max_step_cm)
    assert t.line_update == "capped" and tracker.line_step_capped == 1
    assert t.uncertainty_cm == 1.0  # not shrunk: nothing was confirmed
    assert _row_length_of(t) == pytest.approx(length, abs=1e-6)
    # A plausible step passes untouched and shrinks the uncertainty.
    near = np.stack(_segment(mid + np.array([0.0, 2.0, 0.0]), direction, length))
    fit_near = LineFit(
        line=Line3D(point=near.mean(axis=0), direction=direction, endpoints=near),
        endpoints=near,
        members=members,
        residuals={"T1": 1.0, "T4": 1.0},
        merged_views=(),
        visible_length_cm=length,
        extended=False,
        plane_views=("T1", "T4"),
    )
    t2 = Track("pipette-002", "pipette", mid.copy(), 1.0, "observed", 0, 0)
    t2.set_endpoints(np.stack(_segment(mid, direction, length)))
    t2.class_votes["blue_pipette"] = 5
    tracker._apply_line_fit(t2, fit_near, cams, 1, predicted=t2.position.copy(), dt=1)
    assert t2.line_update == "fit" and t2.uncertainty_cm < 1.0
    assert tracker.line_step_capped == 1


def test_l_a_hold_is_refused_when_the_hand_was_never_near_the_butt(rig) -> None:
    """The first run measured the held offset from the coasted prediction at the moment
    support dropped, whatever the hand's distance: 12.7 / 19.9 cm butt-to-hand medians while
    held. A hand box over a pipette 20 cm from the hand track is an occlusion, not a hold."""
    cams, _fpv = rig
    rng = np.random.default_rng(41)
    frames = list(range(30))
    length = PRIOR.length_for("red_pipette")
    tip = np.array([-10.0, 5.0, -3.0])
    direction = _unit([0.5, 0.3, -1.0])
    butt = tip + direction * length
    params = TrackerParams(
        observation_source="auto",
        coast_timeout_frames=15,
        held=True,
        line_classes=("red_pipette",),
    )

    def run(hand: np.ndarray, box_px: float):
        rows = []
        for f in frames:
            for v in FIXED[:4]:
                rows.append(_box_row(cams, v, f, hand, "right_hand", size=(box_px, box_px)))
            if f < 15 or f >= 25:
                for v in FIXED:
                    rows.append(_axis_row(cams, v, f, tip, butt, "red_pipette", rng=rng))
        return run_tracker(rows, cams, lambda f: None, frames, params)

    # The hand 20 cm from the butt with a box wide enough to cover the pipette's projection.
    far = run(butt + np.array([0.0, 0.0, -20.0]), 700.0)
    pipette = {r.frame_index: r for r in far.rows if r.object_class == "pipette"}
    assert pipette[15].state == "coasting" and pipette[20].state == "coasting"
    assert far.metrics["extensions"]["held"]["episodes"] == 0
    assert _lines_metrics(far)["held"]["refused_hand_far_from_butt"] == 1
    assert not _events(far, "held")
    # The hand at the butt holds it as before, from the offset a localised frame measured,
    # and the rows say how far the butt and the tip sit from that hand.
    near = run(butt + np.array([0.0, 0.0, 1.0]), 220.0)
    pipette = {r.frame_index: r for r in near.rows if r.object_class == "pipette"}
    assert pipette[15].state == "held" and pipette[24].state == "held"
    assert near.metrics["extensions"]["held"]["episodes"] == 1
    assert _lines_metrics(near)["held"]["offset_from_a_localised_frame"] == 1
    for f in range(15, 25):
        assert pipette[f].butt_to_hand_cm == pytest.approx(1.0, abs=1.0), f
        assert pipette[f].tip_to_hand_cm == pytest.approx(length, abs=2.0), f
        assert pipette[f].tip_to_hand_cm > pipette[f].butt_to_hand_cm
    assert pipette[10].butt_to_hand_cm == pytest.approx(1.0, abs=1.0)


def test_m_class_veto_refuses_another_colour_while_a_same_class_mask_is_within_reach(
    rig,
) -> None:
    """Trial 1: T2's `blue_pipette#0` sat on the red pipette and joined the red track on six
    of the seven red disagreements, because the geometric class permitted it and the half-gate
    colour penalty did not outweigh a mask exactly on the line. A track whose plurality is
    decisive (>= 0.75 of the votes over >= 30 frames) now refuses the other colour while a
    same-class mask is within reach in that view; without one the colour never bars."""
    cams, _fpv = rig
    rng = np.random.default_rng(43)
    frames = list(range(70))
    length = PRIOR.length_for("red_pipette")
    a, b = _segment([0.0, -7.0, -1.0], [1.0, 0.0, 0.0], length)
    # The red mask in T2 drawn 2.5 cm off the shaft (a mask that caught the shadow), the blue
    # slot exactly on it.
    view = cams["T2"].centre - 0.5 * (a + b)
    across = _unit(np.cross(view, b - a)) * 2.5
    rows = []
    for f in frames:
        for v in FIXED:
            if v == "T2" and f >= 40:
                if f < 55:
                    rows.append(
                        _axis_row(cams, v, f, a + across, b + across, "red_pipette", rng=rng)
                    )
                rows.append(_axis_row(cams, v, f, a, b, "blue_pipette", rng=rng))
                continue
            rows.append(_axis_row(cams, v, f, a, b, "red_pipette", rng=rng))
    params = TrackerParams(observation_source="auto", line_classes=("red_pipette", "blue_pipette"))
    out = run_tracker(rows, cams, lambda f: None, frames, params)
    assert out.metrics["per_class"]["pipette"]["tracks_born"] == 1
    pipette = {r.frame_index: r for r in out.rows if r.object_class == "pipette"}
    for f in range(40, 55):
        assert pipette[f].support_slots.get("T2") == "red_pipette#0", f
    for f in range(55, 70):
        assert pipette[f].support_slots.get("T2") == "blue_pipette#0", f
    metrics = _lines_metrics(out)
    assert metrics["class_veto"]["observations_refused"] == 15
    assert metrics["class_veto"]["splits"] == 0
    assert all(r.observed_class == "red_pipette" for r in pipette.values())


def test_n_a_track_whose_votes_flip_to_another_colour_splits_into_a_new_id(rig) -> None:
    cams, _fpv = rig
    rng = np.random.default_rng(47)
    frames = list(range(140))
    length = PRIOR.length_cm
    a, b = _segment([0.0, -7.0, -1.0], [1.0, 0.0, 0.0], length)
    rows = []
    for f in frames:
        cls = "red_pipette" if f < 100 else "yellow_pipette"
        for v in FIXED:
            rows.append(_axis_row(cams, v, f, a, b, cls, rng=rng))
    params = TrackerParams(
        observation_source="auto", line_classes=("red_pipette", "yellow_pipette")
    )
    out = run_tracker(rows, cams, lambda f: None, frames, params)
    assert out.metrics["per_class"]["pipette"]["tracks_born"] == 2
    assert out.metrics["per_class"]["red_pipette"]["tracks_born"] == 1
    assert out.metrics["per_class"]["yellow_pipette"]["tracks_born"] == 1
    splits = _events(out, "class_split")
    assert len(splits) == 1 and _lines_metrics(out)["class_veto"]["splits"] == 1
    (split,) = splits
    # 60 voting frames, 5 votes each: the yellow share reaches 0.4 after 24 yellow frames.
    assert split.frame_index == 123
    assert split.payload["plurality_from"] == "red_pipette"
    assert split.payload["plurality_to"] == "yellow_pipette"
    old, new = split.payload["split_from"], split.track_id
    by_track = _by_track(out)
    assert by_track[old][-1].state == "lost" and by_track[old][-1].frame_index == 123
    assert by_track[new][0].frame_index == 123 and by_track[new][0].state == "observed"
    assert by_track[new][-1].observed_class == "yellow_pipette"
    assert by_track[old][-2].observed_class == "red_pipette"
    # The geometry carried over: same line on both rows of the split frame.
    assert np.allclose(by_track[old][-1].endpoints_cm, by_track[new][0].endpoints_cm)
    lost = [e for e in _events(out, "lost") if e.track_id == old]
    assert lost and lost[0].payload["split_to"] == new
    # The rows carry the new fields and round-trip through the schema.
    row = by_track[new][5]
    assert row.line_update == "fit" and row.extent_clamped is False
    back = Track3D.model_validate_json(row.model_dump_json(exclude_none=True))
    assert back == row


# --------------------------------------------------------------------------- Sep 29 disposable tips
#
# A disposable tip is attached to the pipette, picked from a tip rack and ejected into the
# trash; SAM3's body mask usually stops at the cone, so the axis "tip" is the body end, a bias
# of one tip length exactly when the tip matters. The detector's `*_tip` boxes carry the tip.
# Each test here fails on the v2 tracker and passes on this one.

TWO_STATE_PRIOR = LinePrior(
    length_cm=22.0,
    spread_cm=1.5,
    per_class={"blue_pipette": 22.0, "yellow_pipette": 22.0, "red_pipette": 22.0},
    source="test",
    bare={"blue_pipette": 22.0, "yellow_pipette": 22.0, "red_pipette": 22.0},
    tip={"blue_tip": 6.0, "yellow_tip": 5.0, "red_tip": 4.0},
)


def _tip_box_row(cams, view: str, frame: int, body_end, tip_end, cls: str, *, half_px=6.0):
    """A `*_tip` detector box around the projected disposable tip (body end to tip end)."""
    pixels = cams[view].project(np.stack([body_end, tip_end]))
    x0, y0 = pixels.min(axis=0) - half_px
    x1, y1 = pixels.max(axis=0) + half_px
    return FineBioObservation(
        view=view,
        frame_index=frame,
        slot=f"{cls}#0",
        object_class=cls,
        detector_score=0.7,
        box_xyxy_px=(float(x0), float(y0), float(x1), float(y1)),
        pose_valid=True,
        source="detector",
    )


def test_o_an_attached_tip_box_extends_the_extent_names_the_tip_and_sets_the_state(rig) -> None:
    """A blue pipette flat on the bench with a 6 cm disposable tip on. Four views see the
    body mask; T2 and T4 also see a `blue_tip` detector box on the body's end for the first
    25 frames, then the tip is gone. With the tip boxes attached the written segment reaches
    the tip (bare + tip), the tip end is `endpoints_cm[0]` on the box basis with no hand in
    sight, `tip_attached` turns on after five attached frames and off 15 frames after the
    last one, the tip class casts a vote, and the rows say which views attached. With the
    adapter off (the v2 tracker) none of that exists and the segment is the bare body."""
    cams, _fpv = rig
    rng = np.random.default_rng(53)
    frames = list(range(43))
    prior = TWO_STATE_PRIOR
    bare, tip_len = prior.bare_for("blue_pipette"), prior.tip_for("blue_tip")
    direction = _unit([1.0, 0.15, 0.0])
    butt = np.array([-12.0, -8.0, -1.0])
    body_end = butt + direction * bare
    tip_end = body_end + direction * tip_len
    rows = []
    for f in frames:
        for v in ("T1", "T2", "T4", "T5"):
            rows.append(_axis_row(cams, v, f, butt, body_end, "blue_pipette", rng=rng))
        if f < 25:
            for v in ("T2", "T4"):
                rows.append(_tip_box_row(cams, v, f, body_end, tip_end, "blue_tip"))
    params = TrackerParams(observation_source="auto", line_classes=("blue_pipette",))
    out = run_tracker(rows, cams, lambda f: None, frames, params, line_prior=prior)
    assert out.metrics["per_class"]["pipette"]["tracks_born"] == 1
    pipette = {r.frame_index: r for r in out.rows if r.object_class == "pipette"}
    assert set(pipette) == set(frames)
    for f in range(25):
        r = pipette[f]
        assert r.tip_attached_views == ("T2", "T4"), f
        assert r.tip_class == "blue_tip", f
        assert _row_length(r) == pytest.approx(bare + tip_len, abs=1.5), f
        if f >= 4:
            # The box names the tip once the state is on (not on one frame's box).
            assert r.tip_resolved is True and r.tip_basis == "tip_box", f
            assert np.linalg.norm(np.array(r.endpoints_cm[0]) - tip_end) < 1.5, f
            assert np.linalg.norm(np.array(r.endpoints_cm[1]) - butt) < 1.5, f
        else:
            assert r.tip_resolved is False, f
    # The state: None until five of eight frames carry a tip, True from the fifth frame.
    assert pipette[0].tip_attached is None and pipette[3].tip_attached is None
    assert all(pipette[f].tip_attached is True for f in range(4, 39))
    # No tip box from frame 25: the state holds the tip on the completed extent (the body
    # masks alone are one tip length short of the expected length) until 15 frames pass.
    for f in range(25, 39):
        assert pipette[f].tip_attached_views is None, f
        assert _row_length(r := pipette[f]) == pytest.approx(bare + tip_len, abs=1.5), f
        assert r.tip_class == "blue_tip"
    # The state is read after the frame's fit, so frame 39 still wrote the tipped length;
    # from frame 40 the expected length is the bare body and the blend follows within two.
    assert pipette[39].tip_attached is False and pipette[39].tip_class is None
    assert _row_length(pipette[41]) < bare + 0.5 * tip_len
    assert _row_length(pipette[42]) == pytest.approx(bare, abs=1.0)
    metrics = _lines_metrics(out)
    tips = metrics["tips"]
    assert tips["enabled"] and tips["line_frames_with_attached_tip"] == 25
    assert tips["attached_by_view"] == {"T2": 25, "T4": 25}
    assert tips["attached_by_mode"] == {"axis": 50} and tips["attached_by_tip_class"] == {
        "blue_tip": 50
    }
    assert tips["state_transitions"] == {"on": 1, "off": 1}
    assert tips["tip_class_votes"] == 50 and tips["tracks_ever_attached"] == 1
    assert metrics["tip_resolutions_by_basis"] == {"tip_box": 1}
    (votes,) = metrics["class_votes_by_track"].values()
    assert votes == {"blue_pipette": len(frames) * 4 + 50}
    # The rows round-trip through the schema with the new fields.
    back = Track3D.model_validate_json(pipette[10].model_dump_json(exclude_none=True))
    assert back == pipette[10]
    # The adapter off: the v2 tracker. No tip field, no tip vote, the bare body written.
    off = run_tracker(
        rows,
        cams,
        lambda f: None,
        frames,
        TrackerParams(
            observation_source="auto", line_classes=("blue_pipette",), line_tip_boxes=False
        ),
        line_prior=prior,
    )
    plain = [r for r in off.rows if r.object_class == "pipette"]
    assert plain and all(r.tip_attached is None and r.tip_attached_views is None for r in plain)
    assert all(r.tip_resolved is False and r.tip_class is None for r in plain)
    assert np.median([_row_length(r) for r in plain]) == pytest.approx(bare, abs=1.0)
    assert _lines_metrics(off)["tips"]["enabled"] is False
    (votes_off,) = _lines_metrics(off)["class_votes_by_track"].values()
    assert votes_off == {"blue_pipette": len(frames) * 4}


def test_o2_a_box_only_view_attaches_a_tip_on_the_track_s_projected_body(rig) -> None:
    """Trial 1's head camera: the blue pipette has no SAM3 slot there, so its observation is
    the detector box alone (no axis), while the `blue_tip` box is seen. Once the track is a
    line the tip attaches on the track's projected body (`track` mode): it counts for the
    state, the vote and the tip end, and changes no geometry."""
    cams, _fpv = rig
    rng = np.random.default_rng(59)
    frames = list(range(20))
    prior = TWO_STATE_PRIOR
    bare, tip_len = prior.bare_for("blue_pipette"), prior.tip_for("blue_tip")
    direction = _unit([1.0, 0.15, 0.0])
    butt = np.array([-12.0, -8.0, -1.0])
    body_end = butt + direction * bare
    tip_end = body_end + direction * tip_len
    rows = []
    for f in frames:
        for v in ("T2", "T4", "T5"):
            rows.append(_axis_row(cams, v, f, butt, body_end, "blue_pipette", rng=rng))
        # T1: a detector box over the body, no mask, and the tip box beyond the body end.
        rows.append(
            _box_row(cams, "T1", f, 0.5 * (butt + body_end), "blue_pipette", size=(260, 40))
        )
        rows.append(_tip_box_row(cams, "T1", f, body_end, tip_end, "blue_tip"))
    params = TrackerParams(observation_source="auto", line_classes=("blue_pipette",))
    out = run_tracker(rows, cams, lambda f: None, frames, params, line_prior=prior)
    assert out.metrics["per_class"]["pipette"]["tracks_born"] == 1
    pipette = {r.frame_index: r for r in out.rows if r.object_class == "pipette"}
    with_t1 = [f for f in frames if "T1" in pipette[f].support_views]
    assert len(with_t1) >= 15
    attached = [f for f in frames if pipette[f].tip_attached_views == ("T1",)]
    assert len(attached) >= 14 and attached[0] <= 2
    on = [f for f in attached if pipette[f].tip_attached is True]
    assert len(on) >= 10 and all(pipette[f].tip_basis == "tip_box" for f in on)
    # Born as the bare body and completed from the better-supported end until the box names
    # the tip; the first frames after the state turns on blend towards it, then sit on it.
    for f in on[2:]:
        assert np.linalg.norm(np.array(pipette[f].endpoints_cm[0]) - tip_end) < 1.0, f
        assert _row_length(pipette[f]) == pytest.approx(bare + tip_len, abs=1.0), f
    tips = _lines_metrics(out)["tips"]
    assert tips["attached_by_mode"].get("track", 0) >= 14 and tips["attached_by_view"] == {
        "T1": tips["attached_by_mode"]["track"]
    }
    assert pipette[frames[-1]].tip_attached is True


def _end_width_rows(cams, view, frame, a, b, cls, *, rng, wide_at, ratio=3.0):
    """An axis row whose mask is wide at the end nearer the 3D point `wide_at`."""
    row = _axis_row(cams, view, frame, a, b, cls, rng=rng)
    assert row.mask_axis_px is not None and row.mask_width_px is not None
    wide_px = cams[view].project(np.asarray(wide_at, dtype=float))[0]
    ends = np.asarray(row.mask_axis_px)
    wide_end = int(np.argmin(np.linalg.norm(ends - wide_px, axis=1)))
    narrow = float(row.mask_width_px)
    widths = [narrow, narrow]
    widths[wide_end] = narrow * ratio
    return row.model_copy(update={"mask_end_widths_px": (widths[0], widths[1])})


def test_q_the_width_profile_names_the_tip_by_the_class_rule_and_yields_to_a_hand(rig) -> None:
    """The click scorer's flips: every 8-channel anchor sat at the track's other end (the
    tracker called the plunger end the tip; the manifold with the tips is the other end),
    and the single-channel rest pipettes flipped when a hand box passed over one end. The
    mask's end widths decide with a class rule: the 8-channel's wide end is the manifold,
    the tip; a single-channel pipette's wide end is the grip, the butt. Without the widths
    (the v2 rows) a pipette with no hand near it stays unresolved; with them it is resolved
    on the `width` basis after the vote window's margin, and a hand track still overrules."""
    cams, _fpv = rig
    rng = np.random.default_rng(61)
    frames = list(range(12))
    length = PRIOR.length_for("8_channel_pipette")
    a, b = _segment([5.0, 2.0, -1.0], [1.0, 0.2, 0.0], length)  # flat, a -> b
    y0, y1 = _segment([5.0, -8.0, -1.0], [1.0, 0.2, 0.0], PRIOR.length_for("yellow_pipette"))
    rows = []
    for f in frames:
        for v in FIXED:
            # The 8-channel's manifold (wide) is at b; the yellow pipette's grip (wide) is at y1.
            rows.append(_end_width_rows(cams, v, f, a, b, "8_channel_pipette", rng=rng, wide_at=b))
            rows.append(_end_width_rows(cams, v, f, y0, y1, "yellow_pipette", rng=rng, wide_at=y1))
    params = TrackerParams(
        observation_source="auto", line_classes=("8_channel_pipette", "yellow_pipette")
    )
    out = run_tracker(rows, cams, lambda f: None, frames, params)
    assert out.metrics["per_class"]["pipette"]["tracks_born"] == 2
    by_class = defaultdict(dict)
    for r in out.rows:
        if r.object_class == "pipette":
            by_class[r.observed_class][r.frame_index] = r
    eight, yellow = by_class["8_channel_pipette"], by_class["yellow_pipette"]
    # Five views vote every frame: the margin of 5 is met on the first frame's votes.
    assert eight[0].tip_resolved is False  # born before any vote
    for f in frames[1:]:
        assert eight[f].tip_resolved is True and eight[f].tip_basis == "width", f
        assert np.linalg.norm(np.array(eight[f].endpoints_cm[0]) - b) < 1.0, f  # the manifold
        assert yellow[f].tip_resolved is True and yellow[f].tip_basis == "width", f
        assert np.linalg.norm(np.array(yellow[f].endpoints_cm[0]) - y0) < 1.0, f  # not the grip
    metrics = _lines_metrics(out)
    assert metrics["tip_resolutions_by_basis"] == {"width": 2}
    assert metrics["tip_basis_frames"]["width"] == 2 * (len(frames) - 1)
    assert metrics["width_basis"]["votes_cast"] == 2 * 5 * (len(frames) - 1)  # not at birth
    # The v2 rows (no end widths): nothing names the tip.
    plain = [r.model_copy(update={"mask_end_widths_px": None}) for r in rows]
    v2 = run_tracker(plain, cams, lambda f: None, frames, params)
    assert all(r.tip_resolved is False for r in v2.rows if r.object_class == "pipette")
    assert _lines_metrics(v2)["tip_basis_frames"] == {"unresolved": 2 * len(frames)}
    # A hand track at the yellow pipette's grip agrees with the widths and takes over as the
    # stronger basis; a hand track at the 8-channel's manifold end (a wrong reading the
    # widths would contradict) still wins: the width basis is a tie-breaker, not a veto.
    hand_rows = [
        *(
            _box_row(cams, v, f, y1 + np.array([0.0, 0.0, 1.0]), "right_hand", size=(220, 220))
            for f in frames
            for v in FIXED[:4]
        ),
        *(
            _box_row(cams, v, f, b + np.array([0.0, 0.0, 1.0]), "left_hand", size=(220, 220))
            for f in frames
            for v in FIXED[:4]
        ),
    ]
    held = run_tracker(
        rows + hand_rows,
        cams,
        lambda f: None,
        frames,
        TrackerParams(
            observation_source="auto",
            held=True,
            line_classes=("8_channel_pipette", "yellow_pipette"),
        ),
    )
    by_class = defaultdict(dict)
    for r in held.rows:
        if r.object_class == "pipette":
            by_class[r.observed_class][r.frame_index] = r
    for f in frames[2:]:
        assert by_class["yellow_pipette"][f].tip_basis == "hand_track", f
        assert np.linalg.norm(np.array(by_class["yellow_pipette"][f].endpoints_cm[0]) - y0) < 1.0
        assert by_class["8_channel_pipette"][f].tip_basis == "hand_track", f
        assert np.linalg.norm(np.array(by_class["8_channel_pipette"][f].endpoints_cm[0]) - a) < 1.0
    # The variant (`line_width_over_hand`): a decisive width vote outranks the hand, so the
    # 8-channel's tip goes back to the manifold and the row says so.
    variant = run_tracker(
        rows + hand_rows,
        cams,
        lambda f: None,
        frames,
        TrackerParams(
            observation_source="auto",
            held=True,
            line_classes=("8_channel_pipette", "yellow_pipette"),
            line_width_over_hand=True,
        ),
    )
    eight_v = {r.frame_index: r for r in variant.rows if r.observed_class == "8_channel_pipette"}
    for f in frames[2:]:
        assert eight_v[f].tip_basis == "width", f
        assert np.linalg.norm(np.array(eight_v[f].endpoints_cm[0]) - b) < 1.0, f


def test_p_two_state_prior_loads_and_the_old_file_shape_still_does(tmp_path) -> None:
    doc = {
        "length_cm": 23.2,
        "length_spread_cm": 2.5,
        "bare_length_cm": {"blue_pipette": 21.9, "yellow_pipette": 22.0},
        "tip_length_cm": {"blue_tip": 6.1, "yellow_tip": 5.2},
        "provenance": {"per_trial": {"P03": {"per_pipette_median_cm": {"yellow_pipette": 22.0}}}},
    }
    path = tmp_path / "pipettes.json"
    path.write_text(json.dumps(doc))
    prior = load_line_prior(path)
    assert prior.two_state
    assert prior.length_for("yellow_pipette") == pytest.approx(22.0)
    assert prior.bare_for("blue_pipette") == pytest.approx(21.9)
    assert prior.bare_for("red_pipette") == pytest.approx(23.2)  # falls back to length_for
    assert prior.tip_for("blue_tip") == pytest.approx(6.1)
    assert prior.tip_for("red_tip") == pytest.approx(0.5 * (6.1 + 5.2))  # median of the named
    assert prior.tip_class_for("blue_pipette") == "blue_tip"
    assert prior.expected_length("blue_pipette") == pytest.approx(21.9)
    assert prior.expected_length("blue_pipette", attached=True) == pytest.approx(28.0)
    assert prior.expected_length("blue_pipette", "yellow_tip", attached=True) == pytest.approx(27.1)
    assert prior.as_dict()["tip_length_cm"] == {"blue_tip": 6.1, "yellow_tip": 5.2}
    # The old file shape: one state, the tip length zero.
    old = {k: v for k, v in doc.items() if k not in ("bare_length_cm", "tip_length_cm")}
    path.write_text(json.dumps(old))
    single = load_line_prior(path)
    assert not single.two_state and single.tip_for("blue_tip") == 0.0
    assert single.expected_length("yellow_pipette", attached=True) == pytest.approx(22.0)
    assert single.bare_for("yellow_pipette") == single.length_for("yellow_pipette")
