from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from battle.review_metrics import (
    CONFIG_PATH,
    PART_PAIRS,
    DraftEpisode,
    area_anomaly_episodes,
    band_membership,
    cluster_frames,
    config_fingerprint,
    contact_episodes,
    contact_intervals,
    contact_summary,
    finalize_appearance,
    frame_geometry,
    gap_proxies,
    gap_records,
    growth_episodes,
    growth_metrics_for_frame,
    hand_box_union,
    hand_gaps,
    jitter_by_phase,
    jitter_metrics,
    kineo_by_provenance,
    kineo_metrics,
    leakage_episodes,
    load_config,
    pair_metrics,
    part_area_medians,
    part_stats,
    rank_episodes,
    swap_episodes,
)
from battle.review_metrics_schemas import (
    ColorBand,
    HandGapFrameProxy,
    PartAppearanceFrameMetric,
    PartGrowthFrameMetric,
    PartPairFrameMetric,
    ReviewMetricsConfig,
)
from battle.schemas import (
    FrameObservations,
    ImageLandmark2D,
    InteractionContactDiagnostic,
    KineoBoxProvenance,
    NormalizedBox,
    NormalizedPoint,
    PerFrameHand,
    PerFrameNlfBody2D,
)

DIMS = (64, 48)


def _config(**overrides: object) -> ReviewMetricsConfig:
    payload = json.loads((Path(__file__).parents[1] / CONFIG_PATH).read_text())
    payload.update(overrides)
    return ReviewMetricsConfig.model_validate(payload)


def _blob(x0: int, y0: int, w: int, h: int) -> np.ndarray:
    mask = np.zeros((DIMS[1], DIMS[0]), dtype=bool)
    mask[y0 : y0 + h, x0 : x0 + w] = True
    return mask


def _steady_masks() -> dict[str, np.ndarray]:
    return {
        "chassis": _blob(4, 20, 20, 12),
        "interior": _blob(40, 4, 8, 6),
        "rear_body": _blob(4, 38, 10, 6),
        "cabin": _blob(44, 30, 14, 12),
    }


def _geometries(masks_by_frame: list[dict[str, np.ndarray | None]]):
    geometries = []
    previous = None
    for frame, masks in enumerate(masks_by_frame):
        geometries.append(frame_geometry(frame, masks, previous))
        previous = masks
    return geometries


def _hand(
    x: float, y: float, hand_id: str = "stabilized-lane-0", size: float = 0.1
) -> PerFrameHand:
    return PerFrameHand(
        hand_id=hand_id,
        side="right",
        confidence=0.9,
        landmarks=tuple(NormalizedPoint(x=x, y=y) for _ in range(21)),
        box=NormalizedBox(x=max(0, x - size / 2), y=max(0, y - size / 2), width=size, height=size),
        model_side="right",
        model_handedness_confidence=0.9,
    )


def _frame(index: int, hands: tuple[PerFrameHand, ...] = (), **extra: object) -> FrameObservations:
    return FrameObservations(
        view_id="static-c10379",
        analysis_frame_index=index,
        source_seconds=294 + index / 30,
        hands=hands,
        **extra,
    )


def test_cluster_frames_groups_hits_within_gap() -> None:
    assert cluster_frames([1, 2, 3, 10, 12, 30], 0) == [(1, 3), (10, 10), (12, 12), (30, 30)]
    assert cluster_frames([1, 2, 3, 10, 12, 30], 6) == [(1, 12), (30, 30)]
    with pytest.raises(ValueError):
        cluster_frames([1], -1)


def test_swap_detector_fires_on_label_exchange_and_absorption() -> None:
    config = _config()
    frames: list[dict[str, np.ndarray | None]] = []
    for frame in range(50):
        masks = _steady_masks()
        if 20 <= frame < 26:
            # Frames 20-25: the two labels exchange footprints outright.
            masks["chassis"], masks["interior"] = masks["interior"], masks["chassis"]
        if 40 <= frame:
            # Frames 40+: interior swells over the chassis while chassis collapses.
            masks["interior"] = _blob(2, 18, 24, 16)
            masks["chassis"] = _blob(12, 24, 2, 2)
        frames.append(masks)
    geometries = _geometries(frames)
    medians = part_area_medians(geometries)
    metrics = pair_metrics(geometries, medians, config)
    pair = [
        m
        for m in metrics
        if (m.part_a, m.part_b) == ("chassis", "interior") and m.state == "observed"
    ]
    by_frame = {m.analysis_frame_index: m for m in pair}
    assert by_frame[20].exchange_score == 1.0
    assert by_frame[26].exchange_score == 1.0  # swapping back is also an exchange
    assert by_frame[10].swap_score < config.swap_score_threshold
    assert by_frame[45].absorption_score >= config.swap_score_threshold
    assert by_frame[45].absorbing_part == "interior"
    episodes = swap_episodes(metrics, config)
    swaps = [e for e in episodes if e.episode_type == "segmentation_identity_swap"]
    ranges = {(e.start_frame, e.end_frame) for e in swaps if e.subjects == ("chassis", "interior")}
    assert (20, 26) in ranges
    assert any(start == 40 for start, _ in ranges)
    assert all(e.subjects != ("rear_body", "cabin") for e in swaps)
    assert len(PART_PAIRS) == 6


def test_label_crossing_heuristic_requires_trajectories_to_swap_sides() -> None:
    config = _config(crossing_window_frames=4)
    frames = []
    for frame in range(12):
        masks = _steady_masks()
        # Two similar blobs slide past each other horizontally; areas stay constant.
        masks["chassis"] = _blob(4 + 4 * frame, 20, 8, 8)
        masks["interior"] = _blob(48 - 4 * frame, 20, 8, 8)
        frames.append(masks)
    geometries = _geometries(frames)
    metrics = pair_metrics(geometries, part_area_medians(geometries), config)
    crossing_frames = [
        m.analysis_frame_index
        for m in metrics
        if (m.part_a, m.part_b) == ("chassis", "interior") and m.label_crossing
    ]
    assert crossing_frames and min(crossing_frames) >= 6
    assert not any(
        m.label_crossing for m in metrics if (m.part_a, m.part_b) == ("rear_body", "cabin")
    )
    episodes = [e for e in swap_episodes(metrics, config) if e.episode_type.endswith("crossing")]
    assert episodes and episodes[0].subjects == ("chassis", "interior")


def test_missing_masks_are_explicit_not_nan() -> None:
    frames = [_steady_masks(), {**_steady_masks(), "interior": None}]
    geometries = _geometries(frames)
    medians = part_area_medians(geometries)
    stats = part_stats(geometries, medians)
    missing = [s for s in stats if s.analysis_frame_index == 1 and s.part_id == "interior"]
    assert missing[0].state == "missing_mask" and missing[0].area_pixels is None
    pair = [
        m
        for m in pair_metrics(geometries, medians, _config())
        if m.analysis_frame_index == 1 and "interior" in (m.part_a, m.part_b)
    ]
    assert {m.state for m in pair} == {"missing_mask"}
    assert all(m.swap_score is None for m in pair)
    with pytest.raises(ValueError):
        PartPairFrameMetric(
            analysis_frame_index=1,
            part_a="chassis",
            part_b="interior",
            state="missing_mask",
            iou=0.1,
        )


def test_growth_classification_depends_on_hand_overlap() -> None:
    config = _config()
    before = _steady_masks()
    after = {**_steady_masks(), "interior": _blob(40, 4, 8, 12)}  # doubles
    geometries = _geometries([before, after])
    overlapping = hand_box_union((_hand(0.68, 0.15, size=0.2),), DIMS)
    far = hand_box_union((_hand(0.1, 0.9, size=0.1),), DIMS)
    with_hand = growth_metrics_for_frame(geometries[1], geometries[0], after, overlapping, config)
    without = growth_metrics_for_frame(geometries[1], geometries[0], after, far, config)
    interior_with = next(m for m in with_hand if m.part_id == "interior")
    interior_without = next(m for m in without if m.part_id == "interior")
    assert interior_with.growth_event and interior_with.growth_class == "hand_capture_suspect"
    assert interior_without.growth_event and interior_without.growth_class == "unexplained"
    assert interior_with.area_ratio_vs_previous == pytest.approx(2.0)
    chassis = next(m for m in with_hand if m.part_id == "chassis")
    assert chassis.growth_event is False and chassis.growth_class is None
    first = growth_metrics_for_frame(geometries[0], None, before, far, config)
    assert {m.state for m in first} == {"no_previous_mask"}
    episodes = growth_episodes(with_hand + without, config)
    assert {e.episode_type for e in episodes} == {
        "mask_growth_hand_capture_suspect",
        "mask_growth_unexplained",
    }
    with pytest.raises(ValueError, match="growth class"):
        PartGrowthFrameMetric(
            analysis_frame_index=1,
            part_id="interior",
            state="observed",
            area_ratio_vs_previous=1.0,
            centroid_velocity_pixels=0.0,
            hand_box_overlaps_mask=False,
            hand_overlap_fraction=0.0,
            growth_event=False,
            growth_class="unexplained",
        )


def test_slow_ramp_growth_and_sustained_area_anomaly() -> None:
    config = _config(growth_window_frames=5)
    frames = []
    for frame in range(40):
        masks = _steady_masks()
        # Interior grows 1 px per frame in height: never a 1.3x step, but 2x over the run.
        if frame >= 10:
            masks["interior"] = _blob(40, 4, 8, min(30, 6 + (frame - 10)))
        frames.append(masks)
    geometries = _geometries(frames)
    far = hand_box_union((_hand(0.1, 0.9, size=0.1),), DIMS)
    rows = []
    for frame in range(1, 40):
        rows.extend(
            growth_metrics_for_frame(
                geometries[frame],
                geometries[frame - 1],
                frames[frame],
                far,
                config,
                window=geometries[frame - 5] if frame >= 5 else None,
            )
        )
    interior = [m for m in rows if m.part_id == "interior"]
    assert all(m.area_ratio_vs_previous < config.growth_area_ratio_threshold for m in interior)
    assert interior[0].area_ratio_vs_window is None
    windowed = [m for m in interior if m.growth_event]
    assert windowed and all(m.growth_class == "unexplained" for m in windowed)
    assert min(m.analysis_frame_index for m in windowed) >= 10
    medians = part_area_medians(geometries)
    anomalies = area_anomaly_episodes(part_stats(geometries, medians), config)
    assert [(e.subjects, e.episode_type) for e in anomalies] == [
        (("interior",), "mask_area_anomaly_vs_median")
    ]
    assert anomalies[0].end_frame - anomalies[0].start_frame + 1 >= config.area_anomaly_min_frames
    assert "enlarged" in anomalies[0].rationale


def test_hand_gaps_reentry_jump_and_visibility_proxy() -> None:
    config = _config()
    observations = {}
    for frame in range(12):
        hands: tuple[PerFrameHand, ...] = ()
        if frame < 4:
            hands = (_hand(0.3, 0.5),)
        elif frame >= 9:
            hands = (_hand(0.6, 0.5),)  # re-enters far from the last known wrist
        observations[frame] = _frame(frame, hands)
    gaps = hand_gaps(observations, 12)
    assert gaps == [(4, 9)]
    skin = ColorBand(
        space="ycrcb",
        channel_low=(0, 130, 80),
        channel_high=(255, 170, 120),
        derived_from="fixture",
        sample_pixels=1,
    )
    skin_pixel = np.array([120, 150, 100], dtype=np.uint8)
    skin_crop = np.tile(skin_pixel, (9, 9, 1))
    empty_crop = np.zeros((9, 9, 3), dtype=np.uint8)
    samples = [
        SimpleNamespace(
            frame=frame,
            last_known_frame=3,
            hand_id="stabilized-lane-0",
            center=(19, 24),
            size=9,
            crop_ycrcb=skin_crop if frame < 7 else empty_crop,
            reference_ycrcb=skin_crop,
            background_ycrcb=skin_crop if frame == 6 else empty_crop,
            fingertips_in_frame=5,
        )
        for frame in range(4, 9)
    ]
    proxies = gap_proxies(samples, skin, config)
    # Frame 6 has a skin-like window but an equally skin-like background, so it is not a
    # suspect; frames 7-8 have no skin-like pixels at all.
    assert [p.visible_but_undetected_suspect for p in proxies] == [True, True, False, False, False]
    assert proxies[2].background_skin_fraction == 1.0
    assert band_membership(skin_crop, skin).all() and not band_membership(empty_crop, skin).any()
    records = gap_records(observations, gaps, proxies, DIMS)
    assert len(records) == 1
    record = records[0]
    assert (record.last_known_frame, record.reentry_frame, record.suspect_frames) == (3, 9, 2)
    assert record.reentry_jump_pixels == pytest.approx(0.3 * (DIMS[0] - 1), abs=1)
    assert record.reentry_jump_normalized > config.reentry_jump_normalized_threshold
    with pytest.raises(ValueError):
        HandGapFrameProxy(
            analysis_frame_index=4,
            last_known_frame=3,
            last_known_hand_id="h",
            window_center_x=1,
            window_center_y=1,
            window_size_pixels=3,
            skin_fraction_in_window=float("nan"),
            reference_skin_fraction=0.0,
            background_skin_fraction=0.0,
            fingertips_in_frame_count=0,
            visible_but_undetected_suspect=False,
        )


def test_jitter_is_normalized_by_hand_scale_and_grouped_by_phase() -> None:
    observations = {
        0: _frame(0, (_hand(0.30, 0.5, size=0.2),)),
        1: _frame(1, (_hand(0.31, 0.5, size=0.2),)),
        2: _frame(2),
        3: _frame(3, (_hand(0.31, 0.5, size=0.2),)),
    }
    provenance = {(0, "stabilized-lane-0"): "raw", (1, "stabilized-lane-0"): "smoothed"}
    jitter = jitter_metrics(observations, provenance, DIMS, 4)
    assert jitter[0].jitter_normalized is None
    assert jitter[1].provenance_state == "smoothed"
    scale = jitter[1].hand_scale_pixels
    assert jitter[1].jitter_normalized == pytest.approx(
        jitter[1].wrist_displacement_pixels / scale, rel=1e-3
    )
    assert jitter[2].provenance_state == "unknown" and jitter[2].jitter_normalized is None
    phases = jitter_by_phase(
        jitter,
        [("agent_substep", "S01", 0, 2), ("coarse_gt", "0:attach interior", 0, 4)],
        observations,
    )
    assert phases[0].sample_count == 1 and phases[0].missing_hand_frames == 0
    assert phases[1].sample_count == 1 and phases[1].missing_hand_frames == 1


def _diagnostic(frame: int, part: str, value: bool | None) -> InteractionContactDiagnostic:
    if value is None:
        return InteractionContactDiagnostic(
            analysis_frame_index=frame,
            hand_source_id="spatial-lane-1",
            part_id=part,
            observation_state="missing_hand",
        )
    return InteractionContactDiagnostic(
        analysis_frame_index=frame,
        hand_source_id="spatial-lane-1",
        part_id=part,
        observation_state="observed",
        palm_distance_pixels=0.0,
        fingertip_distance_pixels=0.0,
        minimum_distance_pixels=0.0,
        inside_mask=value,
        raw_contact_candidate=value,
        debounced_contact_candidate=value,
    )


def test_contact_intervals_durations_flicker_and_substep_consistency() -> None:
    diagnostics = []
    for frame in range(30):
        diagnostics.append(_diagnostic(frame, "cabin", 3 <= frame < 6))  # 3 frames -> flicker
        diagnostics.append(
            _diagnostic(frame, "chassis", 10 <= frame < 25 or frame >= 28)
        )  # 15 frames closed, then open-ended run
    diagnostics.append(_diagnostic(30, "chassis", None))
    intervals = contact_intervals(
        diagnostics,
        frame_count=31,
        substep_lookup={frame: "S01" for frame in range(31)},
        expected_parts={"S01": ("chassis", "interior")},
        coarse_gt=(("attach interior", 0, 31),),
        flicker_max_frames=5,
    )
    assert [(i.part_id, i.start_frame, i.duration_frames, i.closed) for i in intervals] == [
        ("cabin", 3, 3, True),
        ("chassis", 10, 15, True),
        ("chassis", 28, 2, False),
    ]
    assert [i.expected_for_substep for i in intervals] == [False, True, True]
    assert [i.flicker for i in intervals] == [True, False, True]
    summary = contact_summary(intervals)
    assert summary.sub_5_frame_count == 2
    assert summary.duration_max == 15 and summary.duration_min == 2
    assert summary.per_substep_unexpected == {"S01": 1}
    assert summary.duration_histogram["1-4"] == 2
    episodes = contact_episodes(intervals)
    assert sum(e.episode_type == "contact_unexpected_for_substep" for e in episodes) == 1
    assert sum(e.episode_type == "contact_flicker" for e in episodes) == 2


def _body(offset: float, confidence: float) -> PerFrameNlfBody2D:
    return PerFrameNlfBody2D(
        subject_id="subject_0",
        landmarks=tuple(
            ImageLandmark2D(name=f"j{i}", x=0.5 + offset, y=0.5, confidence=confidence)
            for i in range(3)
        ),
    )


def _provenance(frame: int, source: str) -> KineoBoxProvenance:
    box = NormalizedBox(x=0.2, y=0.1, width=0.4, height=0.5)
    if source == "missing":
        return KineoBoxProvenance(
            analysis_frame_index=frame, source="missing", source_mapping_verified=True
        )
    return KineoBoxProvenance(
        analysis_frame_index=frame,
        source=source,
        box=box,
        native_box=box if source == "detected_native" else None,
        boxmot_box=box if source == "boxmot_fallback" else None,
        source_mapping_verified=True,
        residual_gap_length=1 if source == "interpolated" else None,
    )


def test_kineo_metrics_group_jitter_and_confidence_by_provenance() -> None:
    sources = ["detected_native", "detected_native", "boxmot_fallback", "interpolated", "missing"]
    observations = {
        0: _frame(0, nlf_body_2d=(_body(0.0, 0.9),)),
        1: _frame(1, nlf_body_2d=(_body(0.01, 0.9),)),
        2: _frame(2, nlf_body_2d=(_body(0.05, 0.5),)),
        3: _frame(3, nlf_body_2d=(_body(0.05, 0.3),)),
        4: _frame(4),
    }
    provenance = {frame: _provenance(frame, source) for frame, source in enumerate(sources)}
    metrics = kineo_metrics(observations, provenance, DIMS, 5)
    assert metrics[0].joint_jitter_pixels is None and metrics[0].mean_joint_confidence == 0.9
    assert metrics[1].joint_jitter_pixels == pytest.approx(0.01 * DIMS[0], abs=0.01)
    assert metrics[4].body_present is False and metrics[4].mean_joint_confidence is None
    summary = {item.box_source: item for item in kineo_by_provenance(metrics)}
    assert summary["detected_native"].frame_count == 2
    assert summary["detected_native"].jitter_sample_count == 1
    assert summary["boxmot_fallback"].mean_joint_confidence == 0.5
    assert summary["interpolated"].median_jitter_pixels == pytest.approx(0.0, abs=1e-6)
    assert summary["missing"].body_frames == 0 and summary["missing"].mean_joint_confidence is None
    assert summary["held"].frame_count == 0


def test_rank_episodes_interleaves_types_by_within_type_score() -> None:
    drafts = [
        DraftEpisode("appearance_leakage", 0, 5, 2, 9.0, ("chassis",), "a"),
        DraftEpisode("appearance_leakage", 10, 15, 12, 8.0, ("chassis",), "b"),
        DraftEpisode("appearance_leakage", 20, 25, 22, 7.0, ("chassis",), "c"),
        DraftEpisode("segmentation_identity_swap", 30, 35, 32, 1.5, ("chassis", "interior"), "d"),
        DraftEpisode("segmentation_identity_swap", 40, 45, 42, 2.5, ("chassis", "interior"), "e"),
        DraftEpisode("contact_flicker", 50, 52, 50, 1.0, ("lane", "cabin"), "f"),
    ]
    episodes = rank_episodes(drafts)
    assert [e.rank for e in episodes] == list(range(1, 7))
    assert [e.rationale for e in episodes] == ["a", "e", "f", "b", "d", "c"]
    assert all(e.start_frame <= e.peak_frame <= e.end_frame for e in episodes)


def test_finalize_appearance_flags_outliers_not_steady_drift() -> None:
    config = _config()
    records = []
    for frame in range(60):
        distance = 40.0 if frame != 30 else 90.0  # persistently far from frame 0, one spike
        records.append(
            PartAppearanceFrameMetric(
                analysis_frame_index=frame,
                part_id="interior",
                state="observed",
                median_h=0.0,
                median_s=0.0,
                median_v=0.0,
                mean_l=0.0,
                mean_a=128.0,
                mean_b=128.0,
                lab_distance_from_frame0=distance + (frame % 3),
                lab_distance_robust_z=0.0,
                yellow_fraction=0.5 if frame == 45 else 0.0,
                hand_overlap_fraction=1.0,
                leakage_suspect=False,
            )
        )
    final = finalize_appearance(records, config)
    flagged = [item.analysis_frame_index for item in final if item.leakage_suspect]
    assert flagged == [30, 45]
    assert final[30].lab_distance_robust_z > config.appearance_robust_z_threshold
    episodes = leakage_episodes(final, config)
    assert [(e.start_frame, e.end_frame) for e in episodes] == [(30, 30), (45, 45)]
    assert all(e.score >= 1.0 for e in episodes)


def test_checked_in_config_is_agent_authored_and_covers_every_substep(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    config = load_config(root / CONFIG_PATH)
    assert config.provenance_tag == "agent_authored_assumption"
    assert [item.substep_id for item in config.expected_touched_parts] == [
        f"S{index:02d}" for index in range(1, 12)
    ]
    fingerprint = config_fingerprint(root / CONFIG_PATH, root)
    assert fingerprint.uri == CONFIG_PATH.as_posix()
    copied = tmp_path / "copy.json"
    copied.write_bytes((root / CONFIG_PATH).read_bytes())
    external = config_fingerprint(copied, root)
    assert external.uri == CONFIG_PATH.as_posix() and external.sha256 == fingerprint.sha256
