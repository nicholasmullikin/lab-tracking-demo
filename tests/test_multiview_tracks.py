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

from battle.finebio_slice import leave_one_out_residuals, run_slice
from battle.multiview_schemas import FineBioObservation, Track3D, TrackEvent, read_jsonl
from battle.multiview_tracks import (
    Gates,
    TrackerParams,
    build_container_volumes,
    convex_hull,
    hungarian,
    inside_convex_polygon,
    main,
    parse_container_classes,
    parse_heights,
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
