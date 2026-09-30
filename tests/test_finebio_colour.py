"""`battle-finebio-colour` (p2-colour-vote): the ring sampler on rendered bars, the vote
arithmetic, the hue-bin derivation, the annotate path with an injected frame provider and a
pinhole camera, and the config merge (no data/, no video, no GPU)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from battle.finebio_cameras import Camera
from battle.finebio_colour import (
    CLASS_COLOUR,
    COLOUR_FIELDS,
    COLOURS,
    DEFAULT_SETTINGS,
    HUE_BIN_COUNT,
    ColourSettings,
    ColourVote,
    FrameLookup,
    annotate_tracks,
    derive_hue_bins,
    index_by_slot,
    load_colour_settings,
    main,
    merge_colour_block,
    rank_views,
    ring_hue_sample,
    sample_from_observation,
    sample_track_end,
    sample_track_ends,
    sample_track_tip,
    select_rows,
)

GREY = (190, 190, 190)


def bin_centre_bgr(
    colour: str, settings: ColourSettings = DEFAULT_SETTINGS
) -> tuple[int, int, int]:
    """A fully saturated pixel at the centre of the colour's first hue bin."""
    lo, hi = settings.hue_bins[colour][0]
    hsv = np.array([[[int((lo + hi) / 2), 255, 255]]], dtype=np.uint8)
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def render_bar(
    *,
    p0: tuple[float, float],
    p1: tuple[float, float],
    width: float,
    ring_bgr: tuple[int, int, int] | None,
    ring_end: int,
    size: tuple[int, int] = (640, 480),
    ring_fraction_of_width: float = 0.35,
) -> np.ndarray:
    """A grey bar from p0 to p1 on black, with a band of colour at one end."""
    image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    a, b = np.array(p0, dtype=np.float64), np.array(p1, dtype=np.float64)
    cv2.line(
        image, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), GREY, int(width)
    )
    if ring_bgr is not None:
        end, other = (a, b) if ring_end == 0 else (b, a)
        d = (other - end) / np.linalg.norm(other - end)
        inner = end + ring_fraction_of_width * width * d
        cv2.line(
            image,
            tuple(np.round(end).astype(int)),
            tuple(np.round(inner).astype(int)),
            ring_bgr,
            int(width),
        )
    return image


def axis_row(p0, p1, width, object_class="yellow_pipette", view="T4", frame=600, slot=None):
    return {
        "view": view,
        "frame_index": frame,
        "slot": slot or f"{object_class}#0",
        "object_class": object_class,
        "mask_axis_px": [list(p0), list(p1)],
        "mask_width_px": width,
        "sam3_object_score": 0.9,
        "source": "sam3_decode",
    }


# --------------------------------------------------------------------------- sampler


@pytest.mark.parametrize("colour", COLOURS)
@pytest.mark.parametrize("ring_end", [0, 1])
def test_ring_colour_and_ring_end_are_read_from_a_rendered_bar(colour, ring_end):
    p0, p1, width = (100.0, 240.0), (400.0, 240.0), 40.0
    image = render_bar(
        p0=p0, p1=p1, width=width, ring_bgr=bin_centre_bgr(colour), ring_end=ring_end
    )
    sample = sample_from_observation(image, axis_row(p0, p1, width))
    assert sample is not None
    assert sample["colour"] == colour
    assert sample["confidence"] > 0.95
    assert sample["ring_index"] == ring_end
    # The finding: the ring is the plunger, so the tip is the other end.
    assert sample["tip_index"] == 1 - ring_end
    assert sample["ambiguous"] is False
    assert sample["end_samples"][1 - ring_end] is None
    assert sample["ring_fraction"] >= DEFAULT_SETTINGS.min_ring_fraction
    assert sample["weight"] > 0


def test_ring_hue_sample_reads_the_ring_only_from_the_ring_end():
    p0, p1, width = (100.0, 240.0), (400.0, 240.0), 40.0
    image = render_bar(p0=p0, p1=p1, width=width, ring_bgr=bin_centre_bgr("blue"), ring_end=0)
    inward = np.array([1.0, 0.0])
    at_ring = ring_hue_sample(image, p0, inward, width)
    at_other = ring_hue_sample(image, p1, -inward, width)
    assert at_ring is not None and at_ring["colour"] == "blue"
    assert at_ring["hist"]["blue"] == pytest.approx(1.0)
    assert at_ring["patch_px"] == pytest.approx(2 * 0.6 * width)
    assert at_other is None
    # A degenerate direction or a zero width cannot be sampled.
    assert ring_hue_sample(image, p0, (0.0, 0.0), width) is None
    assert ring_hue_sample(image, p0, inward, 0.0) is None


def test_hidden_ring_returns_none():
    p0, p1, width = (100.0, 240.0), (400.0, 240.0), 40.0
    image = render_bar(p0=p0, p1=p1, width=width, ring_bgr=None, ring_end=0)
    assert sample_from_observation(image, axis_row(p0, p1, width)) is None
    # A bright but unsaturated end (glare) is not a ring either.
    image = render_bar(p0=p0, p1=p1, width=width, ring_bgr=(255, 255, 255), ring_end=1)
    assert sample_from_observation(image, axis_row(p0, p1, width)) is None


def test_tiny_bar_returns_none():
    p0, p1, width = (100.0, 240.0), (160.0, 240.0), 6.0
    image = render_bar(p0=p0, p1=p1, width=width, ring_bgr=bin_centre_bgr("red"), ring_end=0)
    assert 2 * DEFAULT_SETTINGS.disc_radius_factor * width < DEFAULT_SETTINGS.min_patch_px
    assert sample_from_observation(image, axis_row(p0, p1, width)) is None
    # The same bar four times wider is readable.
    image = render_bar(
        p0=p0, p1=(340.0, 240.0), width=24.0, ring_bgr=bin_centre_bgr("red"), ring_end=0
    )
    assert sample_from_observation(image, axis_row(p0, (340.0, 240.0), 24.0))["colour"] == "red"


def test_out_of_frame_hue_and_row_without_axis_are_rejected():
    p0, p1, width = (5.0, 240.0), (300.0, 240.0), 40.0
    image = render_bar(p0=p0, p1=p1, width=width, ring_bgr=bin_centre_bgr("blue"), ring_end=0)
    # The disc at p0 is more than half outside the frame.
    assert ring_hue_sample(image, (-30.0, 240.0), (1.0, 0.0), width) is None
    assert sample_from_observation(image, {"view": "T4", "frame_index": 600}) is None


def test_skin_hue_falls_outside_every_bin_and_a_ring_of_two_colours_is_ambiguous():
    settings = DEFAULT_SETTINGS
    skin = np.array([[[12, 150, 200]]], dtype=np.uint8)
    skin_bgr = tuple(int(x) for x in cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)[0, 0])
    p0, p1, width = (100.0, 240.0), (400.0, 240.0), 40.0
    image = render_bar(p0=p0, p1=p1, width=width, ring_bgr=skin_bgr, ring_end=0)
    assert settings.colour_of_hue(12) is None
    assert sample_from_observation(image, axis_row(p0, p1, width)) is None
    both = render_bar(p0=p0, p1=p1, width=width, ring_bgr=bin_centre_bgr("yellow"), ring_end=0)
    cv2.line(both, (400, 240), (386, 240), bin_centre_bgr("red"), int(width))
    sample = sample_from_observation(both, axis_row(p0, p1, width))
    assert sample is not None and sample["ambiguous"] is True
    assert {s["colour"] for s in sample["end_samples"]} == {"yellow", "red"}


# --------------------------------------------------------------------------- vote


def _sample(colour: str, weight: float = 1.0, share: float = 1.0, view: str | None = None):
    hist = {c: (1 - share) / 2 for c in COLOURS}
    hist[colour] = share
    out = {"colour": colour, "confidence": share, "hist": hist, "weight": weight}
    if view:
        out["view"] = view
    return out


def test_colour_vote_identity_entropy_and_agreement():
    vote = ColourVote()
    assert vote.identity() == (None, 0.0)
    assert vote.entropy() == 0.0
    assert vote.agreement("blue_pipette") is None
    vote.add(_sample("blue", weight=2.0), end_index=0)
    vote.add(_sample("blue", weight=1.0), end_index=0)
    vote.add(_sample("yellow", weight=1.0), end_index=1)
    colour, confidence = vote.identity()
    assert colour == "blue" and confidence == pytest.approx(0.75)
    assert vote.entropy() == pytest.approx(-(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25)))
    assert vote.agreement("blue_pipette") == pytest.approx(2 / 3)
    assert vote.agreement("yellow_pipette") == pytest.approx(1 / 3)
    assert vote.agreement("red_pipette") == 0.0
    assert vote.agreement("cell_culture_plate") is None
    assert vote.ring_end() == (0, pytest.approx(2 / 3))
    uniform = ColourVote()
    for c in COLOURS:
        uniform.add(_sample(c))
    assert uniform.entropy() == pytest.approx(math.log2(3))
    assert uniform.identity()[1] == pytest.approx(1 / 3)
    as_dict = vote.as_dict()
    assert as_dict["identity"] == "blue" and as_dict["samples"] == 3
    assert as_dict["ring_end"] == 0 and as_dict["top_counts"] == {"blue": 2, "yellow": 1}


def test_colour_vote_weights_follow_patch_size_times_ring_fraction():
    heavy = _sample("red", weight=10.0, share=0.6)
    light = _sample("yellow", weight=1.0, share=1.0)
    vote = ColourVote()
    vote.add(heavy)
    vote.add(light)
    shares = vote.shares()
    assert shares["red"] == pytest.approx(6.0 / 11.0)
    assert shares["yellow"] == pytest.approx((2.0 + 1.0) / 11.0)
    assert vote.identity()[0] == "red"


# --------------------------------------------------------------------------- settings


def test_settings_roundtrip_and_hue_membership():
    settings = ColourSettings(
        hue_bins={
            "blue": ((100.0, 125.0),),
            "yellow": ((15.0, 35.0),),
            "red": ((165.0, 180.0), (0.0, 5.0)),
        },
        min_ring_fraction=0.05,
        ring_end="tip",
    )
    again = ColourSettings.from_dict(settings.as_dict())
    assert again == settings
    assert settings.colour_of_hue(2) == "red" and settings.colour_of_hue(170) == "red"
    assert settings.colour_of_hue(35) is None and settings.colour_of_hue(34.9) == "yellow"
    assert settings.tip_index_for_ring(0) == 0
    assert DEFAULT_SETTINGS.tip_index_for_ring(0) == 1
    index = settings.hue_colour_index(np.array([2.0, 20.0, 110.0, 90.0]))
    assert index.tolist() == [
        COLOURS.index("red"),
        COLOURS.index("yellow"),
        COLOURS.index("blue"),
        -1,
    ]


def test_load_colour_settings_falls_back_to_defaults(tmp_path: Path):
    assert load_colour_settings(None) is DEFAULT_SETTINGS
    assert load_colour_settings(tmp_path / "missing.json") is DEFAULT_SETTINGS
    path = tmp_path / "pipettes.json"
    path.write_text(json.dumps({"length_cm": 23.2}), encoding="utf-8")
    assert load_colour_settings(path) is DEFAULT_SETTINGS
    path.write_text(json.dumps({"length_cm": 23.2, "colour": {"min_value": 70}}), encoding="utf-8")
    loaded = load_colour_settings(path)
    assert loaded.min_value == 70 and loaded.hue_bins == DEFAULT_SETTINGS.hue_bins


def test_derive_hue_bins_from_peaked_histograms_keeps_them_apart():
    hists = {}
    for cls, peak_bin in (("blue_pipette", 22), ("yellow_pipette", 5), ("red_pipette", 33)):
        hist = np.zeros(HUE_BIN_COUNT)
        hist[peak_bin] = 0.6
        hist[peak_bin - 1] = 0.2
        hist[(peak_bin + 1) % HUE_BIN_COUNT] = 0.15
        hist[(peak_bin + 8) % HUE_BIN_COUNT] = 0.05  # a stray mode below the floor
        hists[cls] = hist.tolist()
    bins, modes = derive_hue_bins(hists)
    assert set(bins) == set(COLOURS)
    # Three bins round the peak (15 degrees) widened by 5 on each side.
    assert bins["blue"] == ((100.0, 125.0),)
    assert bins["yellow"] == ((15.0, 40.0),)
    assert bins["red"] == ((155.0, 180.0),)
    assert modes["red"]["peak_hue"] == 167.5 and modes["red"]["share_in_region"] == pytest.approx(
        0.95
    )
    # Two classes peaking next to each other are cut at the midpoint of their overlap.
    close = dict(hists)
    hist = np.zeros(HUE_BIN_COUNT)
    hist[8] = 1.0
    close["blue_pipette"] = hist.tolist()
    bins, _ = derive_hue_bins(close)
    y_lo, y_hi = bins["yellow"][0]
    b_lo, b_hi = bins["blue"][0]
    assert y_hi == b_lo and y_lo < y_hi < b_hi
    # A red mode at hue 0 wraps round the circle.
    wrap = dict(hists)
    hist = np.zeros(HUE_BIN_COUNT)
    hist[0] = 0.7
    hist[35] = 0.3
    wrap["red_pipette"] = hist.tolist()
    bins, _ = derive_hue_bins(wrap)
    assert bins["red"] == ((170.0, 180.0), (0.0, 10.0))


def test_select_rows_spreads_the_quota_over_views():
    rows = []
    for view, count in (("T2", 50), ("T4", 5), ("fpv", 50)):
        rows += [axis_row((0, 0), (10, 0), 20, view=view, frame=f) for f in range(count)]
    rows += [axis_row((0, 0), (10, 0), 20, object_class="red_pipette", frame=f) for f in range(3)]
    chosen = select_rows(rows, rows_per_class=30, seed=1)
    yellow = [r for r in chosen if r["object_class"] == "yellow_pipette"]
    assert len(yellow) == 30
    by_view = {v: sum(1 for r in yellow if r["view"] == v) for v in ("T2", "T4", "fpv")}
    assert by_view["T4"] == 5 and by_view["T2"] + by_view["fpv"] == 25
    assert sum(1 for r in chosen if r["object_class"] == "red_pipette") == 3
    assert select_rows(rows, rows_per_class=30, seed=1) == chosen


# --------------------------------------------------------------------------- 3D ends


def pinhole(name: str = "T1", size=(640, 480), tvec=(0.0, 0.0, 0.0)) -> Camera:
    K = np.array([[1000.0, 0.0, size[0] / 2], [0.0, 1000.0, size[1] / 2], [0.0, 0.0, 1.0]])
    return Camera(name, K, np.zeros(5), np.zeros(3), np.array(tvec, dtype=np.float64), size)


def render_track_frame(cam: Camera, endpoints_cm: np.ndarray, ring_end: int, colour: str):
    """The bar of a line track as `cam` sees it, with the ring at `endpoints_cm[ring_end]`."""
    px = cam.project(endpoints_cm)
    depth = float(endpoints_cm[0][2])
    width = float(cam.K[0, 0]) * DEFAULT_SETTINGS.pipette_diameter_cm / depth
    return render_bar(
        p0=tuple(px[0]),
        p1=tuple(px[1]),
        width=width,
        ring_bgr=bin_centre_bgr(colour),
        ring_end=ring_end,
        size=cam.size,
    )


def test_sample_track_ends_projects_both_ends_and_names_the_ring_end():
    cam = pinhole()
    ends = np.array([[-5.0, 0.0, 50.0], [5.0, 0.0, 50.0]])
    frames = {"T1": render_track_frame(cam, ends, ring_end=1, colour="red")}
    sample = sample_track_ends({"T1": cam}, frames, ends)
    assert sample is not None
    assert sample["colour"] == "red" and sample["ring_end"] == 1 and sample["tip_index"] == 0
    assert sample["view"] == "T1" and sample["views_tried"] == ["T1"]
    assert sample["expected_width_px"] == pytest.approx(50.0)
    assert np.allclose(sample["end_px"], cam.project(ends[1:])[0])
    # The plan's name is the same call.
    assert sample_track_tip is sample_track_end
    direct = sample_track_end({"T1": cam}, frames, ends[1], ends[0] - ends[1])
    assert direct is not None and direct["colour"] == "red"
    assert sample_track_end({"T1": cam}, frames, ends[0], ends[1] - ends[0]) is None
    # A per-view width overrides the projected diameter.
    wide = sample_track_end({"T1": cam}, frames, ends[1], ends[0] - ends[1], {"T1": 60.0})
    assert wide is not None and wide["expected_width_px"] == 60.0


def test_rank_views_prefers_the_nearest_camera_that_sees_the_end():
    near, far = pinhole("near"), pinhole("far", tvec=(0.0, 0.0, 100.0))
    behind = pinhole("behind", tvec=(0.0, 0.0, -100.0))
    off = pinhole("off", tvec=(200.0, 0.0, 0.0))
    end = np.array([0.0, 0.0, 50.0])
    ranked = rank_views(
        {"far": far, "near": near, "behind": behind, "off": off}, end, None, DEFAULT_SETTINGS
    )
    assert [v for v, _, _ in ranked] == ["near", "far"]
    assert ranked[0][2] > ranked[1][2]
    # A view whose frame is missing is skipped and the next one is tried.
    ends = np.array([[-5.0, 0.0, 50.0], [5.0, 0.0, 50.0]])
    frames = {"far": render_track_frame(far, ends, ring_end=0, colour="yellow")}
    sample = sample_track_ends({"near": near, "far": far}, frames, ends)
    assert sample is not None and sample["view"] == "far" and sample["colour"] == "yellow"


def test_frame_lookup_reads_each_view_once():
    calls = []

    def provider(view: str, frame: int):
        calls.append((view, frame))
        return None if view == "missing" else np.zeros((4, 4, 3), dtype=np.uint8)

    frames = FrameLookup(provider, 612)
    assert frames.get("T1") is not None and frames.get("T1") is not None
    assert frames.get("missing") is None and frames.get("missing") is None
    assert calls == [("T1", 612), ("missing", 612)]


# --------------------------------------------------------------------------- annotate


def test_annotate_two_row_fixture_with_fake_tracks_and_frames():
    cam = pinhole("T1")
    ends = np.array([[-5.0, 0.0, 50.0], [5.0, 0.0, 50.0]])
    line_rows = [
        {
            "frame_index": f,
            "track_id": "pipette-001",
            "object_class": "blue_pipette",
            "position_cm": [0.0, 0.0, 50.0],
            "endpoints_cm": ends.tolist(),
            "direction": [1.0, 0.0, 0.0],
            "state": "observed",
        }
        for f in (600, 601, 603)
    ]
    # A point-tracker row: no endpoints, a support slot naming an observation with an axis.
    p0, p1, width = (100.0, 300.0), (400.0, 300.0), 40.0
    point_row = {
        "frame_index": 600,
        "track_id": "red_pipette-007",
        "object_class": "red_pipette",
        "position_cm": [1.0, 2.0, 3.0],
        "support_slots": {"T4": "red_pipette#0", "T5": "red_pipette#0"},
        "state": "observed",
    }
    other = {"frame_index": 600, "track_id": "plate-001", "object_class": "cell_culture_plate"}
    observations = index_by_slot(
        [
            axis_row(p0, p1, width, object_class="red_pipette", view="T4", frame=600),
            {
                **axis_row(p0, p1, 10.0, object_class="red_pipette", view="T5", frame=600),
                "mask_axis_px": None,
            },
        ]
    )
    frames_seen = []
    line_frame = render_track_frame(cam, ends, ring_end=1, colour="blue")
    point_frame = render_bar(p0=p0, p1=p1, width=width, ring_bgr=bin_centre_bgr("red"), ring_end=0)

    def provider(view: str, frame: int):
        frames_seen.append((view, frame))
        if view == "T1":
            return line_frame
        if view == "T4":
            return point_frame
        return None

    rows = [*line_rows, point_row, other]
    annotated, summary = annotate_tracks(
        rows,
        cameras_for_frame=lambda frame: {"T1": cam},
        frame_provider=provider,
        observations=observations,
        every=3,
    )
    assert annotated is not None and len(annotated) == 5
    # Every third frame of the line track was sampled (600 and 603, not 601).
    assert sorted(f for v, f in frames_seen if v == "T1") == [600, 603]
    for row in line_rows:
        assert row["colour_identity"] == "blue"
        assert row["colour_confidence"] == pytest.approx(1.0)
        assert row["colour_entropy"] == 0.0
        assert row["colour_samples"] == 2
        assert row["colour_agreement"] == 1.0
        assert row["colour_ring_end"] == 1 and row["colour_tip_end"] == 0
        assert row["colour_ring_end_agreement"] == 1.0
    assert point_row["colour_identity"] == "red" and point_row["colour_samples"] == 1
    assert point_row["colour_agreement"] == 1.0
    assert point_row["colour_ring_end"] is None and point_row["colour_tip_end"] is None
    assert not any(k in other for k in COLOUR_FIELDS)
    assert summary["pipette-001"]["line_track"] is True
    assert summary["pipette-001"]["views"] == {"T1": 2}
    assert summary["red_pipette-007"]["line_track"] is False
    assert summary["red_pipette-007"]["agreement_with_class"] == 1.0


def test_annotate_without_evidence_leaves_identity_none():
    rows = [
        {"frame_index": 600, "track_id": "yellow_pipette-001", "object_class": "yellow_pipette"},
        {"frame_index": 603, "track_id": "yellow_pipette-001", "object_class": "yellow_pipette"},
    ]
    annotated, summary = annotate_tracks(
        rows, cameras_for_frame=None, frame_provider=lambda v, f: None, observations={}, every=1
    )
    assert all(r["colour_identity"] is None and r["colour_samples"] == 0 for r in annotated)
    assert all(r["colour_agreement"] is None for r in annotated)
    assert summary["yellow_pipette-001"]["identity"] is None


# --------------------------------------------------------------------------- config


def test_merge_colour_block_keeps_the_rest_of_the_config(tmp_path: Path):
    path = tmp_path / "pipettes.json"
    original = {"schema_version": "1.0", "length_cm": 23.21, "provenance": {"trial": "P03"}}
    path.write_text(json.dumps(original), encoding="utf-8")
    block = {**DEFAULT_SETTINGS.as_dict(), "provenance": {"trial": "P03_03_01"}}
    doc = merge_colour_block(path, block)
    again = json.loads(path.read_text(encoding="utf-8"))
    assert again == doc
    assert {k: again[k] for k in original} == original
    assert again["colour"]["ring_end"] == "plunger"
    assert load_colour_settings(path) == DEFAULT_SETTINGS
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_config_subcommand_merges_the_proposed_block(tmp_path: Path):
    calibration = tmp_path / "colour_calibration.json"
    block = {**DEFAULT_SETTINGS.as_dict(), "min_value": 65, "provenance": {"trial": "P03_03_01"}}
    calibration.write_text(json.dumps({"proposed_colour_block": block}), encoding="utf-8")
    pipettes = tmp_path / "pipettes.json"
    pipettes.write_text(json.dumps({"length_cm": 23.21}), encoding="utf-8")
    assert main(["config", "--calibration", str(calibration), "--pipettes", str(pipettes)]) == 0
    doc = json.loads(pipettes.read_text(encoding="utf-8"))
    assert doc["length_cm"] == 23.21 and doc["colour"]["min_value"] == 65
    assert load_colour_settings(pipettes).min_value == 65


def test_class_colour_table_covers_the_three_pipettes():
    assert set(CLASS_COLOUR.values()) == set(COLOURS)
    assert set(CLASS_COLOUR) == {"blue_pipette", "yellow_pipette", "red_pipette"}


def test_plunger_precompute_decodes_once_per_view_frame_and_keeps_source(tmp_path, monkeypatch):
    from battle import finebio_colour as colour
    from battle.finebio_orientation import load_plunger_ends

    rows = [
        {
            "view": view,
            "frame_index": frame,
            "slot": slot,
            "object_class": "blue_pipette",
            "source": "sam3_decode",
            "mask_axis_px": [[0, 0], [100, 0]],
            "mask_width_px": 25.0,
        }
        for view, frame, slot in [("T2", 1, "b"), ("T1", 2, "a"), ("T1", 1, "b"), ("T1", 1, "a")]
    ]
    source = tmp_path / "observations.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    before = source.read_bytes()
    reads = []

    def provider(view, frame):
        reads.append((view, frame))
        return np.zeros((10, 10, 3), dtype=np.uint8)

    def sample(image, row, settings):
        return {"ambiguous": row["slot"] == "b", "tip_index": 1, "confidence": 0.8}

    monkeypatch.setattr(colour, "sample_from_observation", sample)
    output = tmp_path / "plunger_ends.jsonl"
    summary = colour.write_plunger_ends(source, provider, output)
    assert reads == [("T1", 1), ("T1", 2), ("T2", 1)]
    assert summary["written"] == 4 and summary["ambiguous"] == 2
    assert source.read_bytes() == before
    assert not output.with_suffix(".jsonl.partial").exists()
    cached = load_plunger_ends(output)
    assert cached[("T1", 1, "a")] == (1, 0.8, False)
    assert cached[("T1", 1, "b")] == (None, 0.0, True)
    monkeypatch.setattr(colour, "ProxyFrameSource", lambda *a: pytest.fail("decoded cache again"))
    assert (
        colour.main(
            [
                "plunger-ends",
                "--observations",
                str(source),
                "--clip-config",
                str(tmp_path / "unused.json"),
                "--output",
                str(output),
            ]
        )
        == 0
    )
