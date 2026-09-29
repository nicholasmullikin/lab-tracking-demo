"""multiview_lines (p1-line-geometry) on the real P03 rig from the preflight fixtures, with
synthetic 30 cm segments projected into the cameras at 0.5 px noise: the fit, the extent and
length prior, the leave-one-camera-out residual and its two negative controls (camera 6 with a
6.4 cm pose error, one view one frame late), the line gate, and the box-centre baseline."""

from __future__ import annotations

import numpy as np
import pytest
from finebio_fixtures import load_preflight_fixtures

from battle.finebio_cameras import Camera
from battle.finebio_slice import depth_cm
from battle.multiview_lines import (
    AxisObs,
    Line3D,
    axis_residual,
    centroid_loo_residual,
    centroid_point,
    fit_line,
    is_elongated,
    line_cost,
    line_distance,
    line_endpoints,
    loo_residual,
    plane_from_axis,
    point_to_line_cm,
    ray_from_point,
    reproject_line,
)
from battle.multiview_tracks import ray_point

FIXED = ("T1", "T2", "T3", "T4", "T5")
NOISE_PX = 0.5
LENGTH_CM = 30.0
# The magnitude by which camera 6's shipped pose misses its marker PnP on P03 (cameras.json
# provenance: pnp_vs_shipped_cm 6.43).
SHIPPED_POSE_ERROR_CM = 6.4


@pytest.fixture(scope="module")
def rig():
    fixtures = load_preflight_fixtures()
    cams = fixtures.fixed_cameras()
    fpv = fixtures.fpv_camera(1798)
    assert fpv is not None
    return cams, fpv


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _segment(mid, direction, length: float = LENGTH_CM) -> tuple[np.ndarray, np.ndarray]:
    mid, direction = np.asarray(mid, dtype=float), _unit(direction)
    return mid - direction * length / 2, mid + direction * length / 2


# A pipette standing almost upright with its butt 3 cm above the bench, and one held at 45
# degrees with its middle 20 cm up.
VERTICAL = (
    np.array([5.0, -2.0, -3.0]),
    np.array([5.0, -2.0, -3.0]) + LENGTH_CM * _unit([0.05, 0.03, -1]),
)
TILTED = _segment([-5.0, 10.0, -20.0], [np.cos(np.pi / 4), 0.0, -np.sin(np.pi / 4)])
# Upright between 5 and 25 cm up at (2, 20): inside every fixed frame and the fpv's at 1798.
SIX_VIEW = _segment([2.0, 20.0, -10.0], [0.0, 0.0, -1.0])


def _observe(
    cam: Camera,
    view: str,
    a: np.ndarray,
    b: np.ndarray,
    rng: np.random.Generator,
    *,
    noise_px: float = NOISE_PX,
    width_cm: float = 2.0,
    visible: tuple[float, float] = (0.0, 1.0),
    elongation: float | None = None,
    shift: np.ndarray | None = None,
) -> AxisObs:
    """The view's observation of the part `visible` of segment a->b: projected axis
    endpoints plus noise, the centroid of the visible part plus noise, and the elongation a
    2 cm wide shaft would show at that depth (or the one given)."""
    if shift is not None:
        a, b = a + shift, b + shift
    lo, hi = visible
    p0, p1 = a + lo * (b - a), a + hi * (b - a)
    pixels = cam.project(np.stack([p0, p1]))
    mid = 0.5 * (p0 + p1)
    width_px = width_cm * cam.K[0, 0] / depth_cm(cam, mid)
    length_px = float(np.linalg.norm(pixels[1] - pixels[0]))
    elongation = max(1.0, length_px / width_px) if elongation is None else elongation
    return AxisObs(
        view,
        pixels + rng.normal(0.0, noise_px, pixels.shape),
        cam.project(mid)[0] + rng.normal(0.0, noise_px, 2),
        float(elongation),
    )


def _observations(cams, views, a, b, rng, **per_view_kwargs):
    """``[(camera, AxisObs)]`` for `views`; `per_view_kwargs[view]` are `_observe` kwargs."""
    return [(cams[v], _observe(cams[v], v, a, b, rng, **per_view_kwargs.get(v, {}))) for v in views]


def _direction_error_deg(line: Line3D, a, b) -> float:
    return float(np.degrees(np.arccos(min(1.0, abs(float(line.direction @ _unit(b - a)))))))


def _endpoint_error_cm(endpoints, a, b) -> float:
    """Worst endpoint error under the better of the two orderings."""
    e = np.asarray(endpoints)
    same = max(np.linalg.norm(e[0] - a), np.linalg.norm(e[1] - b))
    swapped = max(np.linalg.norm(e[0] - b), np.linalg.norm(e[1] - a))
    return float(min(same, swapped))


def _perturbed(cam: Camera, delta_cm: np.ndarray) -> Camera:
    """The same camera with its centre moved by `delta_cm` (world), rotation kept."""
    centre = cam.centre + np.asarray(delta_cm, dtype=float)
    return Camera(cam.name, cam.K, cam.dist, cam.rvec, -(cam.R @ centre), cam.size)


def _loo(obs, **kw) -> dict[str, dict]:
    return {r["view"]: r for r in loo_residual(obs, **kw)}


def test_plane_and_ray_follow_the_camera_conventions(rig) -> None:
    cams, fpv = rig
    a, b = SIX_VIEW
    for cam in (*cams.values(), fpv):
        pixels = cam.project(np.stack([a, b]))
        normal, d = plane_from_axis(cam, pixels)
        assert np.linalg.norm(normal) == pytest.approx(1.0)
        # The plane holds the camera centre and both true 3D endpoints (to a micron; the
        # undistortion is iterative).
        for x in (cam.centre, a, b):
            assert abs(normal @ x - d) < 1e-4
        analytic = _unit(np.cross(b - a, cam.centre - a))
        assert abs(abs(normal @ analytic) - 1.0) < 1e-9
        # The ray is `multiview_tracks.ray_point`'s: undistort, K^-1, R^T from the centre.
        origin, direction = ray_from_point(cam, pixels[0])
        assert np.allclose(origin, cam.centre)
        assert np.allclose(origin + ((a - origin) @ direction) * direction, a, atol=1e-4)
        assert np.allclose(ray_point(cam, pixels[0], a), a, atol=1e-4)
    with pytest.raises(ValueError):
        plane_from_axis(cams["T1"], np.array([[100.0, 100.0], [100.0, 100.0]]))


def test_a_vertical_segment_from_five_fixed_cameras(rig) -> None:
    cams, _ = rig
    a, b = VERTICAL
    rng = np.random.default_rng(1)
    obs = _observations(cams, FIXED, a, b, rng)
    line = fit_line(obs)
    assert line is not None
    assert line.support_views == FIXED and line.condition > 0.3
    assert _direction_error_deg(line, a, b) < 0.5
    assert line.endpoints is not None and _endpoint_error_cm(line.endpoints, a, b) < 0.5
    assert abs(line.length_cm - LENGTH_CM) < 0.5
    assert point_to_line_cm(0.5 * (a + b), line) < 0.2
    assert all(r.perpendicular_px < 1.5 for r in line.residuals.values())
    loo = _loo(obs)
    assert set(loo) == set(FIXED)
    assert all(r["fitted"] and r["perpendicular_px"] < 2.0 for r in loo.values())
    assert all(r["angle_deg"] < 2.0 for r in loo.values())
    # Twenty more noise draws: the direction bound holds on every draw, the endpoint and LOO
    # bounds in the median, with no draw past 1.5x them (the worst LOO is T4's, the camera
    # whose held-out fit is the least conditioned).
    directions, endpoints, worst_loo = [], [], []
    for seed in range(2, 22):
        draw = _observations(cams, FIXED, a, b, np.random.default_rng(seed))
        fitted = fit_line(draw)
        assert fitted is not None and fitted.endpoints is not None
        directions.append(_direction_error_deg(fitted, a, b))
        endpoints.append(_endpoint_error_cm(fitted.endpoints, a, b))
        worst_loo.append(max(r["perpendicular_px"] for r in loo_residual(draw)))
    assert max(directions) < 0.5
    assert np.median(endpoints) < 0.5 and max(endpoints) < 0.75
    assert np.median(worst_loo) < 2.0 and max(worst_loo) < 3.0


def test_b_tilted_segment_held_at_20_cm(rig) -> None:
    cams, _ = rig
    a, b = TILTED
    rng = np.random.default_rng(3)
    obs = _observations(cams, FIXED, a, b, rng)
    line = fit_line(obs)
    assert line is not None and line.plane_views == FIXED
    assert _direction_error_deg(line, a, b) < 0.5
    assert _endpoint_error_cm(line.endpoints, a, b) < 0.5
    assert -line.midpoint[2] == pytest.approx(20.0, abs=0.3)
    loo = _loo(obs)
    assert all(r["perpendicular_px"] < 2.0 and r["angle_deg"] < 1.0 for r in loo.values())


def test_six_cameras_with_the_wide_lens_fpv(rig) -> None:
    """The fpv (k1 = -0.26) joins: skipping the undistortion would cost tens of pixels."""
    cams, fpv = rig
    all_cams = {**cams, "fpv": fpv}
    a, b = SIX_VIEW
    rng = np.random.default_rng(4)
    obs = _observations(all_cams, (*FIXED, "fpv"), a, b, rng)
    assert is_elongated(obs[-1][1])
    line = fit_line(obs)
    assert line is not None and "fpv" in line.plane_views
    assert _direction_error_deg(line, a, b) < 0.5
    assert _endpoint_error_cm(line.endpoints, a, b) < 0.5
    loo = _loo(obs)
    assert loo["fpv"]["perpendicular_px"] < 2.0 and loo["fpv"]["angle_deg"] < 1.0
    assert all(r["perpendicular_px"] < 2.0 for r in loo.values())
    # The distortion the fpv's axis carries is far above the residual: undistorting first is
    # what makes the agreement, and `reproject_line` puts it back for drawing.
    raw_ends = obs[-1][1].endpoints_px
    assert np.linalg.norm(fpv.undistort(raw_ends) - raw_ends, axis=1).max() > 5.0
    x0, y0, x1, y1 = reproject_line(fpv, line)
    drawn = np.array([[x0, y0], [x1, y1]])
    assert _endpoint_error_cm(drawn, raw_ends[0], raw_ends[1]) < 3.0  # px here


def test_c_compact_view_is_a_ray_not_a_plane(rig) -> None:
    cams, _ = rig
    a, b = VERTICAL
    rng = np.random.default_rng(5)
    # T5 looks down the shaft: it reports endpoints too, but with elongation 1.2 they are
    # not an axis and the view must contribute its centroid's ray.
    obs = _observations(cams, FIXED, a, b, rng, T5={"elongation": 1.2})
    line = fit_line(obs)
    assert line is not None
    assert line.plane_views == ("T1", "T2", "T3", "T4") and line.ray_views == ("T5",)
    assert line.residuals["T5"].kind == "ray" and line.residuals["T5"].angle_deg is None
    assert line.residuals["T5"].perpendicular_px < 2.0
    assert _direction_error_deg(line, a, b) < 0.5
    assert _endpoint_error_cm(line.endpoints, a, b) < 0.5
    assert "T5" not in _loo(obs)  # no axis, nothing to hold out against
    # Two planes plus the ray: the ray takes part in the point solve.
    two = [pair for pair in obs if pair[1].view in ("T1", "T2", "T5")]
    line2 = fit_line(two)
    assert line2 is not None and line2.ray_views == ("T5",)
    assert point_to_line_cm(0.5 * (a + b), line2) < 0.3

    # One plane plus rays: rays to different portions of the shaft cross the plane at
    # different points and give the line; rays to the same centroid do not.
    parts = _observations(
        cams,
        ("T1", "T2", "T4"),
        a,
        b,
        rng,
        T2={"elongation": 1.0, "visible": (0.0, 0.4)},
        T4={"elongation": 1.0, "visible": (0.6, 1.0)},
    )
    one_plane = fit_line(parts)
    assert one_plane is not None
    assert one_plane.plane_views == ("T1",) and one_plane.ray_views == ("T2", "T4")
    assert _direction_error_deg(one_plane, a, b) < 1.0
    assert point_to_line_cm(0.5 * (a + b), one_plane) < 0.5
    assert 0 < one_plane.condition <= 1
    same_centroid = _observations(
        cams, ("T1", "T2", "T4"), a, b, rng, T2={"elongation": 1.0}, T4={"elongation": 1.0}
    )
    assert fit_line(same_centroid) is None
    # Rays only: the point tracker's job.
    rays_only = _observations(cams, FIXED, a, b, rng, **{v: {"elongation": 1.0} for v in FIXED})
    assert fit_line(rays_only) is None
    assert loo_residual(rays_only) == []


def test_d_two_planes_under_five_degrees_return_none(rig) -> None:
    cams, _ = rig
    # A horizontal pipette along x at 25 cm height between T1 and T2, whose centres sit 57 cm
    # up at nearly the same y: the two planes meet at about 3.9 degrees.
    a, b = np.array([-15.0, -17.0, -25.0]), np.array([15.0, -17.0, -25.0])
    n1, _ = plane_from_axis(cams["T1"], cams["T1"].project(np.stack([a, b])))
    n2, _ = plane_from_axis(cams["T2"], cams["T2"].project(np.stack([a, b])))
    pair_angle = np.degrees(np.arccos(abs(n1 @ n2)))
    assert 3.0 < pair_angle < 5.0
    rng = np.random.default_rng(6)
    two = _observations(cams, ("T1", "T2"), a, b, rng)
    assert fit_line(two) is None
    assert all(not r["fitted"] for r in loo_residual(two))
    # Lowering the gate accepts the pair; adding the top-down camera makes it a real fit.
    weak = fit_line(two, min_pair_angle_deg=2.0)
    assert weak is not None and weak.condition < 0.05
    three = two + _observations(cams, ("T5",), a, b, rng)
    line = fit_line(three)
    # T5 is on the same side of this pipette in y, so its plane meets the pair at only about
    # 20 degrees: usable, not ideal.
    assert line is not None and 0.1 < line.condition < 0.3
    assert _direction_error_deg(line, a, b) < 0.5


def test_e_length_prior_completes_a_truncated_extent_and_flags_merged(rig) -> None:
    cams, _ = rig
    a, b = TILTED
    rng = np.random.default_rng(7)
    # Two views see the lower 60% of the shaft, three see only 30%..60% (T4's 9 cm of it is
    # so foreshortened that it is compact and gives a ray): the 60% end is supported by every
    # axis view, the 0% end by two, so the prior extends past the 0% end.
    seen = {
        "T1": (0.0, 0.6),
        "T2": (0.0, 0.6),
        "T3": (0.3, 0.6),
        "T4": (0.3, 0.6),
        "T5": (0.3, 0.6),
    }
    obs = _observations(cams, FIXED, a, b, rng, **{v: {"visible": s} for v, s in seen.items()})
    line = fit_line(obs)
    assert line is not None and line.ray_views == ("T4",)
    assert line.length_cm == pytest.approx(0.6 * LENGTH_CM, abs=0.5)
    extent = line_endpoints(line, obs, LENGTH_CM)
    assert extent is not None
    assert set(extent.per_view) == {"T1", "T2", "T3", "T5"}
    assert extent.visible_length_cm == pytest.approx(0.6 * LENGTH_CM, abs=0.5)
    assert extent.length_cm == pytest.approx(LENGTH_CM, abs=1e-6)
    assert extent.extended and not extent.merged
    assert sorted(extent.support) == [2, 4]
    kept = 1 - (extent.extended_end or 0)
    seen_end = a + 0.6 * (b - a)
    assert np.linalg.norm(extent.endpoints[kept] - seen_end) < 0.5
    far = a - 0.4 * (b - a)  # the reconstructed end sits 12 cm past the unseen end
    assert np.linalg.norm(extent.endpoints[extent.extended_end] - far) < 0.5
    # Equal support extends both ends by half.
    obs_even = _observations(cams, FIXED, a, b, rng, **{v: {"visible": (0.2, 0.8)} for v in FIXED})
    even = line_endpoints(fit_line(obs_even), obs_even, LENGTH_CM)
    assert even.extended and even.extended_end is None
    assert _endpoint_error_cm(even.endpoints, a, b) < 0.5
    # A full extent within tolerance of the prior is completed but not flagged.
    obs_full = _observations(cams, FIXED, a, b, rng)
    full = line_endpoints(fit_line(obs_full), obs_full, LENGTH_CM)
    assert not full.extended and not full.merged
    assert full.length_cm == pytest.approx(LENGTH_CM, abs=0.5)
    # 1.5x the prior: two masks in one, flagged and left as seen.
    a_long, b_long = _segment(0.5 * (a + b), b - a, 1.5 * LENGTH_CM)
    obs_long = _observations(cams, FIXED, a_long, b_long, rng)
    line_long = fit_line(obs_long)
    merged = line_endpoints(line_long, obs_long, LENGTH_CM)
    assert merged.merged and not merged.extended
    assert merged.length_cm == pytest.approx(1.5 * LENGTH_CM, abs=0.6)
    # No axis view, no extent.
    compact = [(c, AxisObs(o.view, None, o.centroid_px)) for c, o in obs_long]
    assert line_endpoints(line_long, compact, LENGTH_CM) is None


def test_f_negative_control_shipped_pose_error_on_camera_6(rig) -> None:
    cams, _ = rig
    a, b = TILTED
    rng = np.random.default_rng(8)
    obs = _observations(cams, FIXED, a, b, rng)
    good = _loo(obs)
    # The observations are what the true camera saw; the geometry now believes a T5 (camera
    # 6) whose centre is 6.4 cm off, the size of the shipped pose's miss on P03.
    bad_t5 = _perturbed(cams["T5"], SHIPPED_POSE_ERROR_CM * _unit([1.0, 1.0, 0.0]))
    obs_bad = [(bad_t5 if o.view == "T5" else cam, o) for cam, o in obs]
    bad = _loo(obs_bad)
    ratio = bad["T5"]["perpendicular_px"] / good["T5"]["perpendicular_px"]
    assert good["T5"]["perpendicular_px"] < 2.0
    assert bad["T5"]["perpendicular_px"] > 20.0 and ratio > 10.0
    # Every other view's held-out fit is polluted by the wrong plane too.
    assert all(bad[v]["perpendicular_px"] > 5.0 for v in FIXED if v != "T5")
    # The in-sample fit sees it as well: T5's residual against the all-view line.
    line_bad = fit_line(obs_bad)
    assert line_bad is not None and line_bad.residuals["T5"].perpendicular_px > 10.0


def test_g_negative_control_one_view_one_frame_late(rig) -> None:
    cams, _ = rig
    a, b = TILTED
    rng = np.random.default_rng(9)
    good = _loo(_observations(cams, FIXED, a, b, rng))
    # T3's observation is of the pipette one frame later, moving 2 cm/frame across itself.
    across = 2.0 * _unit(np.cross(b - a, [0.0, 0.0, 1.0]))
    late = _observations(cams, FIXED, a, b, rng, T3={"shift": across})
    shifted = _loo(late)
    assert good["T3"]["perpendicular_px"] < 2.0
    assert shifted["T3"]["perpendicular_px"] > 10.0
    assert shifted["T3"]["perpendicular_px"] / good["T3"]["perpendicular_px"] > 10.0
    # Motion along the shaft moves nothing the line can see.
    along = _observations(cams, FIXED, a, b, rng, T3={"shift": 2.0 * _unit(b - a)})
    assert _loo(along)["T3"]["perpendicular_px"] < 2.0


def test_h_line_distance_and_cost() -> None:
    up = Line3D(
        point=[0.0, 0.0, -10.0], direction=[0.0, 0.0, -1.0], endpoints=[[0, 0, 0], [0, 0, -20]]
    )
    beside = Line3D(
        point=[3.0, 0.0, -5.0], direction=[0.0, 0.0, 1.0], endpoints=[[3, 0, 0], [3, 0, -20]]
    )
    angle, perpendicular, midpoint = line_distance(up, beside)
    assert angle == pytest.approx(0.0, abs=1e-9)
    assert perpendicular == pytest.approx(3.0) and midpoint == pytest.approx(3.0)
    assert line_cost(up, beside) == pytest.approx(3.0)
    crossing = Line3D(
        point=[0.0, 0.0, -10.0], direction=[np.sin(np.radians(30)), 0.0, -np.cos(np.radians(30))]
    )
    angle, perpendicular, midpoint = line_distance(up, crossing)
    assert angle == pytest.approx(30.0) and perpendicular == pytest.approx(0.0, abs=1e-9)
    assert midpoint == pytest.approx(0.0, abs=1e-9)
    assert line_cost(up, crossing) == pytest.approx(15.0 * np.sin(np.radians(30)))
    # Sign of the direction and order of the endpoints carry no meaning.
    flipped = Line3D(
        point=beside.point, direction=-beside.direction, endpoints=beside.endpoints[::-1]
    )
    assert line_distance(up, flipped) == pytest.approx((0.0, 3.0, 3.0), abs=1e-9)
    # A midpoint slid along the line costs `along_weight` per cm, nothing perpendicular.
    slid = Line3D(
        point=[0.0, 0.0, -30.0], direction=[0.0, 0.0, -1.0], endpoints=[[0, 0, -20], [0, 0, -40]]
    )
    angle, perpendicular, midpoint = line_distance(up, slid)
    assert (angle, perpendicular, midpoint) == pytest.approx((0.0, 0.0, 20.0), abs=1e-9)
    assert (
        line_cost(up, slid) == pytest.approx(10.0) and line_cost(up, slid, along_weight=0.0) == 0.0
    )
    assert point_to_line_cm([4.0, 0.0, -100.0], up) == pytest.approx(4.0)


def test_i_centroid_baseline_is_on_the_line_only_when_views_see_the_same_portion(rig) -> None:
    cams, _ = rig
    a, b = TILTED
    rng = np.random.default_rng(10)
    obs = _observations(cams, FIXED, a, b, rng)
    line = fit_line(obs)
    point = centroid_point(obs)
    assert point is not None
    assert point_to_line_cm(point, line) < 0.3
    assert np.linalg.norm(point - 0.5 * (a + b)) < 0.3
    assert centroid_point(obs[:1]) is None
    # Views see different portions of the shaft, as T1 (top third), T4 (bottom half) and the
    # head camera (tip) do on the bench: the box centres are different places on it.
    seen = {
        "T1": (0.0, 0.4),
        "T2": (0.5, 1.0),
        "T3": (0.2, 0.7),
        "T4": (0.6, 1.0),
        "T5": (0.0, 1.0),
    }
    parts = _observations(cams, FIXED, a, b, rng, **{v: {"visible": s} for v, s in seen.items()})
    line_parts = fit_line(parts)
    assert line_parts is not None and _direction_error_deg(line_parts, a, b) < 0.5
    assert _endpoint_error_cm(line_parts.endpoints, a, b) < 0.5
    line_loo = np.array([r["perpendicular_px"] for r in loo_residual(parts)])
    box = centroid_loo_residual(parts)
    assert [r["view"] for r in box] == list(FIXED) and all(r["fitted"] for r in box)
    box_distance = np.array([r["distance_px"] for r in box])
    box_perpendicular = np.array([r["perpendicular_px"] for r in box])
    assert line_loo.max() < 2.0
    assert np.median(box_distance) > 10 * np.median(line_loo)
    assert np.median(box_perpendicular) > np.median(line_loo)
    assert point_to_line_cm(centroid_point(parts), line) > 1.0
    # A missing axis falls back to the centroid target and reports no perpendicular.
    compact = [(c, AxisObs(o.view, None, o.centroid_px)) for c, o in parts]
    assert all(
        r["perpendicular_px"] is None and r["distance_px"] is not None
        for r in centroid_loo_residual(compact)
    )


def test_reproject_line_and_axis_residual_agree_with_the_camera(rig) -> None:
    cams, _ = rig
    a, b = VERTICAL
    truth = Line3D(point=0.5 * (a + b), direction=b - a, endpoints=[a, b])
    for view, cam in cams.items():
        x0, y0, x1, y1 = reproject_line(cam, truth)
        assert np.allclose([[x0, y0], [x1, y1]], cam.project(np.stack([a, b])))
        perpendicular, angle = axis_residual(cam, cam.project(np.stack([a, b])), truth)
        assert perpendicular < 1e-6 and angle < 1e-6
        assert truth.residuals == {}
    behind = Line3D(point=cams["T1"].centre - 10 * cams["T1"].R.T[:, 2], direction=[1, 0, 0])
    assert all(np.isnan(reproject_line(cams["T1"], behind)))
    assert np.isnan(axis_residual(cams["T1"], np.array([[0.0, 0.0], [10.0, 0.0]]), behind)[0])
