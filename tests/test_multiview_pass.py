"""Multi-view pass: geometric seed transfer, part consensus episodes, run metadata guards."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from conftest import require_artifact
from test_multiview_geometry import UP, _camera, _look_at, _rule

from battle import multiview_consensus as mvc
from battle import multiview_geometry as mvg
from battle import multiview_seed_transfer as st
from battle.muggled_smoke import require_four_part_multiview_range
from battle.multiview_schemas import (
    DisagreementEpisode,
    MultiviewSeedTransferManifest,
    SeedCandidate,
    SeedTransferPart,
)
from battle.schemas import (
    ArtifactFingerprint,
    FourPartMultiviewRunMetadata,
    FrameRange,
    GeometricSeedProvenance,
    PixelBox,
)


@pytest.fixture
def static_rig() -> mvg.CameraRig:
    target = np.array([0.0, 0.0, 0.0])
    poses = {
        "C10001": _look_at(np.array([600.0, -700.0, -900.0]), target, UP),
        "C10002": _look_at(np.array([-800.0, -650.0, -700.0]), target, UP),
        "C10003": _look_at(np.array([100.0, -900.0, 800.0]), target, UP),
    }
    cameras = {view: _camera(view, pose) for view, pose in poses.items()}
    rules = {view: _rule(f"{view}:rgb", 5) for view in poses}
    return mvg.CameraRig(cameras, rules, {})


def _plane_through(point: np.ndarray, normal: np.ndarray) -> tuple[np.ndarray, float]:
    normal = normal / np.linalg.norm(normal)
    return normal, float(normal @ point)


def test_plane_lift_and_warp_round_trip(static_rig: mvg.CameraRig) -> None:
    """A disc painted on the table in one view warps into another view onto the same 3D disc."""
    normal, offset = _plane_through(np.zeros(3), -UP)  # the table is y = 0, normal points up
    centre = np.array([40.0, 0.0, -30.0])
    # Paint the disc directly from geometry in the source view.
    shape = (720, 1280)
    source = np.zeros(shape, dtype=bool)
    grid_y, grid_x = np.mgrid[0:720, 0:1280]
    pixels = np.stack([grid_x.ravel() + 0.5, grid_y.ravel() + 0.5], axis=1)
    world = st.lift_pixels_to_plane(static_rig, normal, offset, "C10001", pixels)
    inside = np.linalg.norm(np.nan_to_num(world - centre, nan=1e9), axis=1) < 60.0
    source[grid_y.ravel()[inside], grid_x.ravel()[inside]] = True
    assert source.sum() > 200
    # Warp into the second view and compare with the disc painted there directly.
    world_2 = st.lift_pixels_to_plane(static_rig, normal, offset, "C10002", pixels)
    expected = np.zeros(shape, dtype=bool)
    inside_2 = np.linalg.norm(np.nan_to_num(world_2 - centre, nan=1e9), axis=1) < 60.0
    expected[grid_y.ravel()[inside_2], grid_x.ravel()[inside_2]] = True
    warped = st.warp_mask_via_plane(
        static_rig,
        normal,
        offset,
        source_view="C10001",
        source_mask=source,
        target_view="C10002",
        target_shape=shape,
        target_region=(0, 0, 1280, 720),
    )
    assert st.iou(warped, expected) > 0.9
    # The centroid ray of the warped disc passes near the disc centre (an oblique disc's image
    # centroid is not the projection of its centre, so "near" is half a radius).
    distance = st.ray_point_distance(static_rig, "C10002", st.mask_centroid(warped), centre, None)
    assert distance < 30.0


def test_project_to_proxy_marks_points_behind_the_camera(static_rig: mvg.CameraRig) -> None:
    pose = static_rig.camera_to_world("C10001")
    behind = pose[:3, 3] - 1000.0 * pose[:3, 2]
    projected = st.project_to_proxy(static_rig, "C10001", np.stack([np.zeros(3), behind]))
    assert np.isfinite(projected[0]).all()
    assert np.isnan(projected[1]).all()


def test_boxes_are_clamped_and_degenerate_boxes_are_refused() -> None:
    shape = (720, 1280)
    box = st.square_box(np.array([10.0, 700.0]), 50.0, 0.25, shape)
    assert box is not None and box.x1 == 0 and box.y2 == 720
    assert st.square_box(np.array([5000.0, 5000.0]), 50.0, 0.0, shape) is None
    assert st.bounding_box(np.array([[np.nan, np.nan], [1.0, 1.0]]), 0.1, shape) is None


def test_decode_requests_normalise_boxes_and_points() -> None:
    manifest_path = Path(__file__).parent / "fixtures" / "does_not_exist.json"
    assert not manifest_path.exists()
    part = SeedTransferPart(
        target="chassis",
        kind="part",
        status="blocked",
        blocked_reason="not decoded yet",
        prompts=(
            st.SeedPrompt(
                prompt_id="t000000-b01",
                target="chassis",
                variant="fixture",
                pixel_box=PixelBox(x1=128, y1=72, x2=256, y2=144),
                background_points=(st.PixelPoint(x=640, y=360),),
            ),
        ),
    )
    manifest = _seed_manifest(parts=(part,))
    (request,) = st.decode_requests(manifest)
    assert request["boxes"] == [[[0.1, 0.1], [0.2, 0.2]]]
    assert request["bg_points"] == [[0.5, 0.5]]
    assert request["candidate_id"] == "t000000-b01"


def _fingerprint(uri: str = "runs/x/mask.png") -> ArtifactFingerprint:
    return ArtifactFingerprint(uri=uri, sha256="0" * 64, source="measured")


def _seed_manifest(parts: tuple[SeedTransferPart, ...]) -> MultiviewSeedTransferManifest:
    plane = mvg.TablePlane(
        normal=(0.0, 1.0, 0.0),
        offset_mm=0.0,
        point_on_plane_mm=(0.0, 0.0, 0.0),
        down_vector=(0.0, -1.0, 0.0),
        down_vector_source="fixture",
        down_vector_alternatives_deg={},
        normal_vs_down_deg=0.0,
        fingertip_count=10,
        fitted_point_count=3,
        lowest_fraction=0.05,
        residual_rms_mm=1.0,
        residual_p95_mm=2.0,
        fingertips_below_plane_by_20mm_fraction=0.0,
        claim_boundary="fixture",
    )
    return MultiviewSeedTransferManifest(
        manifest_kind="multiview_geometric_seed_transfer",
        view="C10095",
        view_id="static-c10095",
        analysis_frame_index=0,
        pose_frame_index=17645,
        is_ego=False,
        reference_view="C10379",
        reference_run_manifest=_fingerprint(),
        reference_seed_masks=(_fingerprint(),),
        secondary_reference_view="HMC_21110305",
        secondary_reference_pose_frame=17640,
        secondary_reference_run_manifest=_fingerprint(),
        secondary_reference_seed_masks=(_fingerprint(),),
        transfer_method="fixture",
        clip_config=_fingerprint(),
        proxy=_fingerprint(),
        proxy_dimensions=(1280, 720),
        proxy_to_raw_scale=1.5,
        table_plane=plane,
        rules=st.RULES,
        parts=parts,
        decode_state="planned",
        run_decision="pending",
        run_decision_reason="fixture",
        claim_boundaries=("fixture",),
    )


def test_seed_part_status_requires_a_passing_accepted_candidate() -> None:
    passing = SeedCandidate(
        prompt_id="t000000-b01",
        candidate_index=0,
        mask=_fingerprint(),
        decoder_iou_estimate=0.9,
        mask_area_px=100,
        sanity_pass=True,
        acceptance_basis="plane_warp_iou",
    )
    SeedTransferPart(target="chassis", kind="part", status="accepted", accepted=passing)
    with pytest.raises(ValueError):
        SeedTransferPart(target="chassis", kind="part", status="accepted")
    with pytest.raises(ValueError):
        SeedTransferPart(
            target="chassis",
            kind="part",
            status="accepted",
            accepted=passing.model_copy(update={"sanity_pass": False}),
        )
    with pytest.raises(ValueError):
        SeedTransferPart(target="chassis", kind="part", status="blocked")


def test_multiview_run_range_permits_new_static_views_and_e4_only() -> None:
    assert require_four_part_multiview_range("static-c10095", 0, 1800).frame_count == 1800
    assert require_four_part_multiview_range("ego-hmc21179183", 0, 1800).frame_count == 1800
    for view, start, count in (
        ("static-c10379", 0, 1800),
        ("ego-hmc21110305", 0, 1800),
        ("static-c10095", 0, 2781),
        ("static-c10095", 10, 1800),
    ):
        with pytest.raises(ValueError):
            require_four_part_multiview_range(view, start, count)


def test_multiview_run_metadata_requires_ordered_seeds() -> None:
    def seed(target: str, slot: int) -> GeometricSeedProvenance:
        return GeometricSeedProvenance(
            target=target,
            multiplex_slot=slot,
            mask_fingerprint=_fingerprint(),
            prompt_id="t000000-b01",
            candidate_index=0,
        )

    common = dict(
        requested_analysis_frame_range=FrameRange(start_frame=0, end_frame_exclusive=1800),
        requested_seconds=60.0,
        view_id="static-c10095",
        source_fingerprint=_fingerprint(),
        proxy_fingerprint=_fingerprint(),
        config_fingerprint=_fingerprint(),
        adapter={
            "name": "battle.muggled_smoke",
            "version": "0.3.0",
            "implementation_basis": "fixture",
            "external_source_uri": "fixture",
        },
        continuity={
            "max_prompt_memory_entries": 1,
            "max_frame_memory_entries": 4,
            "detected_object_limit": 3,
        },
        runtime_settings={},
        measurements={"elapsed_seconds": 1.0},
        mask_artifact_count=0,
        seed_manifest_fingerprint=_fingerprint(),
    )
    metadata = FourPartMultiviewRunMetadata(
        concepts=("chassis", "rear_body", "cabin"),
        seeds=(seed("chassis", 0), seed("rear_body", 1), seed("cabin", 2)),
        blocked_targets={"interior": "seed_transfer_failed"},
        **common,
    )
    assert metadata.seed_provenance == "geometric_seed_transfer"
    assert all(item.selected_by == "agent" for item in metadata.seeds)
    with pytest.raises(ValueError):
        FourPartMultiviewRunMetadata(
            concepts=("chassis", "rear_body"),
            seeds=(seed("rear_body", 0), seed("chassis", 1)),
            **common,
        )
    with pytest.raises(ValueError):
        FourPartMultiviewRunMetadata(
            concepts=("chassis", "rear_body"),
            seeds=(seed("chassis", 0), seed("rear_body", 2)),
            **common,
        )


def test_disagreement_episodes_need_five_frames_and_a_majority_to_contradict() -> None:
    frames = 30
    exists = np.ones(frames, dtype=bool)
    exists[25:] = False
    errors = {
        "C10379": np.zeros(frames),
        "C10095": np.zeros(frames),
        "C10115": np.zeros(frames),
    }
    errors["C10379"][5:12] = 90.0  # seven frames: an episode
    errors["C10095"][14:17] = 90.0  # three frames: too short
    errors["C10115"][20:30] = 90.0  # runs past the end of the consensus: clipped to [20,25)
    dropped = {view: np.zeros(frames, dtype=bool) for view in errors}
    dropped["C10379"][5:12] = True
    dropped["C10115"][20:30] = True
    used = np.full(frames, 2)
    used[22] = 1  # one frame where the "majority" is a single view
    episodes = mvc.episodes_from_errors(
        errors, exists, dropped, used, target="chassis", threshold_px=40.0, min_frames=5
    )
    by_view = {episode.view: episode for episode in episodes}
    assert set(by_view) == {"C10379", "C10115"}
    reference = by_view["C10379"]
    assert (reference.start_frame, reference.end_frame_exclusive) == (5, 12)
    assert reference.contradicts_majority and reference.max_reprojection_error_px == 90.0
    other = by_view["C10115"]
    assert (other.start_frame, other.end_frame_exclusive) == (20, 25)
    assert not other.contradicts_majority
    assert all(isinstance(e, DisagreementEpisode) for e in episodes)
    assert mvc.merge_intervals([(5, 12), (10, 14), (20, 25)]) == ((5, 14), (20, 25))


def test_distance_to_mask_is_zero_inside_and_euclidean_outside() -> None:
    mask = np.zeros((20, 30), dtype=bool)
    mask[5:10, 10:15] = True
    assert mvc.distance_to_mask_px(mask, np.array([12.0, 7.0])) == 0.0
    outside = mvc.distance_to_mask_px(mask, np.array([14.5, 13.5]))
    assert outside == pytest.approx(4.0)
    centroid = mvc.mask_centroid_raw(mask, 1.5)
    assert centroid is not None and centroid == pytest.approx([12.5 * 1.5, 7.5 * 1.5])
    assert mvc.mask_centroid_raw(np.zeros((4, 4), dtype=bool), 1.0) is None


# -- real data -------------------------------------------------------------------------------


@pytest.mark.real_data
def test_seed_transfer_manifests_are_agent_authored_with_verified_masks() -> None:
    root = require_artifact(st.OUTPUT_ROOT)
    manifests = [st.load_manifest(path) for path in sorted(root.glob("*/seed_manifest.json"))]
    assert len(manifests) >= 7
    for manifest in manifests:
        assert manifest.selected_by == "agent"
        assert manifest.provenance == "geometric_seed_transfer"
        if not any(part.prompts for part in manifest.parts):
            # Nothing projected into the view at frame 0: skipped without spending a decode.
            assert manifest.run_decision == "skip"
            continue
        assert manifest.decode_state == "decoded"
        for part in manifest.parts:
            if part.accepted is None:
                assert part.status == "blocked" and part.blocked_reason
                continue
            path = Path(part.accepted.mask.uri)
            assert path.is_file()
            assert st.sha256_file(path) == part.accepted.mask.sha256
            ratio = part.accepted.area_ratio_vs_expected
            assert ratio is not None and 0.3 <= ratio <= 3.0
        if manifest.run_decision == "run":
            assert len(manifest.accepted_parts) >= manifest.rules.min_parts_to_run


def test_default_run_discovery_recognises_the_tracker_policy_as_a_run_condition() -> None:
    assert mvc.has_default_tracker_policy({"max_side_length": 720})
    assert mvc.has_default_tracker_policy(
        {"tracker_memory_policy": json.dumps({"slot_exclusivity": "off", "memory_gate": "off"})}
    )
    assert not mvc.has_default_tracker_policy(
        {"tracker_memory_policy": json.dumps({"slot_exclusivity": "argmax", "memory_gate": "off"})}
    )
    assert not mvc.has_default_tracker_policy(
        {"tracker_memory_policy": {"slot_exclusivity": "off", "memory_gate": "on"}}
    )


@pytest.mark.real_data
def test_multiview_runs_declare_agent_seeds_and_first_minute() -> None:
    runs = mvc.discover_multiview_runs(Path.cwd())
    if not runs:
        pytest.skip("no multiview first-minute runs built")
    for view, directory in runs.items():
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        metadata = manifest["four_part_multiview"]
        assert metadata["seed_provenance"] == "geometric_seed_transfer"
        assert metadata["requested_analysis_frame_range"]["end_frame_exclusive"] == 1800
        assert all(seed["selected_by"] == "agent" for seed in metadata["seeds"])
        assert "interior" in metadata["blocked_targets"] or "interior" in metadata["concepts"]
        settings = json.loads((directory / "runtime_settings.json").read_text(encoding="utf-8"))
        assert "agent" in settings["label"]


@pytest.mark.real_data
def test_consensus_manifest_has_static_majority_and_sub_millimetre_wrists() -> None:
    manifest = mvc.load_consensus(require_artifact(mvc.OUTPUT_ROOT / mvc.MANIFEST_NAME))
    assert manifest.reference_view == "C10379"
    assert len(manifest.sources) >= 6
    assert manifest.wrist_triangulation.median_residual_mm < 0.01
    for episode in manifest.episodes:
        assert episode.frame_count >= manifest.rules.min_episode_frames
    for proposal in manifest.proposed_validity_intervals:
        assert proposal.applied is False
        assert proposal.trigger == "multiview_disagreement"
    per_frame = Path(manifest.per_frame_uri)
    assert per_frame.is_file()
    assert sum(1 for _ in per_frame.open(encoding="utf-8")) == manifest.frame_count
