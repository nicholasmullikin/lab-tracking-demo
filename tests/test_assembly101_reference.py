"""Assembly101 dataset reference window: clock rule, projection, labels, and the v4 layer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from conftest import require_artifact

from battle import assembly101_reference as a101
from battle.assembly101_pose_schemas import (
    ASSEMBLY101_EDGES,
    ASSEMBLY101_JOINT_NAMES,
    Assembly101CameraModel,
    Assembly101ClockRule,
    Assembly101FineSegment,
    Assembly101Hand,
    Assembly101HandFrame,
    Assembly101Point2D,
    Assembly101Point3D,
    Assembly101ReferenceManifest,
)
from battle.interaction_review import (
    NAVIGATION_FINE_GT_INDEX,
    NAVIGATION_SUBSTEP_INDEX,
    _blueprint,
    _log_navigation_frame,
)
from battle.interaction_review_v4 import (
    FINE_LABELS,
    STATIC_TEXT_PANELS,
    _log_assembly101_frame,
    _log_assembly101_static,
    _log_static_documents,
)
from battle.schemas import ArtifactFingerprint

CSV_HEADER = (
    "id,video,start_frame,end_frame,action_id,verb_id,noun_id,action_cls,verb_cls,noun_cls,"
    "toy_id,toy_name,is_shared,is_RGB\n"
)


def _camera(*, raw: tuple[int, int] = (1920, 1080)) -> Assembly101CameraModel:
    """A camera 1 m in front of the world origin looking down -Z... in world terms: the
    camera sits at z=-1000 looking along +Z with an identity rotation, so world points near
    the origin project close to the principal point."""
    return Assembly101CameraModel(
        view_key="C10379:rgb",
        raw_image_size=raw,
        intrinsic_matrix=((1000.0, 0.0, 960.0), (0.0, 1000.0, 540.0), (0.0, 0.0, 1.0)),
        distortion=(0.0, 0.0, 0.0, 0.0, 0.0),
        camera_to_world=(
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, -1000.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        provenance="estimated_from_dataset_landmark_projection",
        fit_rms_pixels=0.0,
        fit_point_count=1,
    )


def _rule(offset: int = 9) -> Assembly101ClockRule:
    return Assembly101ClockRule(
        view_key="C10379:rgb",
        proxy_start_raw_frame=100,
        raw_frames_per_proxy_frame=2,
        pose_offset_frames=offset,
        pose_fps=60,
        analysis_fps=30,
        offset_uncertainty_frames=1,
        offset_evidence="fixture",
    )


def test_clock_rule_maps_proxy_frames_onto_the_offset_pose_clock() -> None:
    rule = _rule(9)
    assert rule.pose_frame(0) == 109
    assert rule.pose_frame(1) == 111
    assert _rule(0).pose_frame(5) == 110
    with pytest.raises(ValueError):
        rule.pose_frame(-1)


def test_joint_graph_is_a_21_joint_connected_hand() -> None:
    assert len(ASSEMBLY101_JOINT_NAMES) == 21
    assert ASSEMBLY101_JOINT_NAMES[5] == "wrist"
    touched = {index for edge in ASSEMBLY101_EDGES for index in edge}
    assert touched == set(range(20)), "every joint except the palm centre is on an edge"


def test_projection_matches_a_hand_computed_pinhole_and_scales_to_the_proxy() -> None:
    camera = _camera()
    world = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 50.0, 500.0]])
    projected = a101.project_world_points(world, camera)
    # camera at z=-1000: depth of the origin is 1000 mm, so 100 mm -> 100 px.
    assert projected[0] == pytest.approx([960.0, 540.0])
    assert projected[1] == pytest.approx([1060.0, 540.0])
    assert projected[2] == pytest.approx([960.0, 540.0 + 1000 * 50 / 1500])


def _synthetic_inputs(frame_count: int, rule: Assembly101ClockRule):
    landmarks: dict[str, dict[str, list[list[float]]]] = {}
    confidences: dict[str, dict[str, float]] = {}
    timestamps: dict[str, float] = {}
    for proxy_frame in range(frame_count):
        key = str(rule.pose_frame(proxy_frame))
        base = np.zeros((21, 3))
        base[:, 0] = np.arange(21) * 10.0 + proxy_frame
        far = base.copy()
        far[:, 0] += 5000.0  # projects far outside the image
        landmarks[key] = {"0": base.tolist(), "1": far.tolist()}
        confidences[key] = {"0": 0.9, "1": 0.0 if proxy_frame == 1 else 0.7}
        timestamps[key] = 2451.8 + int(key) / 60
    return landmarks, confidences, timestamps


def test_build_hand_frames_drops_zero_confidence_hands_and_counts_inside_joints() -> None:
    rule = _rule(9)
    landmarks, confidences, timestamps = _synthetic_inputs(3, rule)
    frames = a101.build_hand_frames(
        landmarks3d=landmarks,
        confidences=confidences,
        timestamps=timestamps,
        camera=_camera(),
        clock_rule=rule,
        dimensions=(1280, 720),
        frame_count=3,
        source_start_seconds=294.0,
    )
    assert [frame.pose_frame_index for frame in frames] == [109, 111, 113]
    assert [len(frame.hands) for frame in frames] == [2, 1, 2]
    left = frames[0].hands[0]
    assert left.side == "left" and left.hand_index == 0
    assert left.joints_inside_image == 21
    # Raw 960 px -> proxy 640 px (x2/3) for the first joint.
    assert left.joints_proxy_pixels[0].x == pytest.approx(640.0)
    assert frames[2].hands[1].joints_inside_image == 0
    assert frames[1].source_seconds == pytest.approx(294.0 + 1 / 30)


def test_missing_pose_frame_is_an_error_not_a_silent_gap() -> None:
    rule = _rule(9)
    landmarks, confidences, timestamps = _synthetic_inputs(2, rule)
    with pytest.raises(KeyError, match="pose frame"):
        a101.build_hand_frames(
            landmarks3d=landmarks,
            confidences=confidences,
            timestamps=timestamps,
            camera=_camera(),
            clock_rule=rule,
            frame_count=3,
        )


def test_fine_segments_use_one_view_clip_to_the_window_and_keep_overlaps(tmp_path: Path) -> None:
    csv_path = tmp_path / "fg.csv"
    rid = "rec"
    static = f"{rid}/C10379_rgb.mp4"
    rows = [
        # id, video, start, end, action_id, verb_id, noun_id, action, verb, noun
        ("1", static, 90, 110, 8, 2, 17, "inspect diagram", "inspect", "diagram"),
        ("2", static, 100, 120, 42, 5, 10, "position interior", "position", "interior"),
        ("3", static, 105, 115, 4, 1, 2, "put down screwdriver", "put down", "screwdriver"),
        ("4", static, 130, 140, 3, 0, 2, "pick up screwdriver", "pick up", "screwdriver"),
        ("5", static, 20, 100, 0, 0, 1, "pick up screw", "pick up", "screw"),
        (
            "9",
            f"{rid}/C10404_rgb.mp4",
            100,
            120,
            42,
            5,
            10,
            "position interior",
            "position",
            "interior",
        ),
    ]
    csv_path.write_text(
        CSV_HEADER
        + "".join(
            f"{i},{v},{s:09d},{e:09d},{a:04d},{vb:04d},{n:04d},{ac},{vc},{nc},c02a,-,0,1\n"
            for i, v, s, e, a, vb, n, ac, vc, nc in rows
        )
    )
    segments = a101.load_fine_segments(csv_path, annotation_start_frame=100, frame_count=30)
    assert [s.annotation_id for s in segments] == ["1", "2", "3"]
    assert segments[0].proxy_start_frame == 0 and segments[0].proxy_end_frame_exclusive == 10
    assert segments[0].clipped_to_window is True
    assert segments[1].proxy_start_frame == 0 and segments[1].proxy_end_frame_exclusive == 20
    assert segments[1].clipped_to_window is False
    assert segments[2].proxy_start_frame == 5
    active = a101.fine_segments_for_frame(segments, 7)
    assert [index for index, _ in active] == [0, 1, 2]
    assert a101.fine_segments_for_frame(segments, 25) == ()


def test_fine_segment_rejects_empty_ranges() -> None:
    with pytest.raises(ValueError):
        Assembly101FineSegment(
            annotation_id="x",
            action_id=1,
            verb="v",
            noun="n",
            action="v n",
            annotation_start_frame=10,
            annotation_end_frame=10,
            proxy_start_frame=0,
            proxy_end_frame_exclusive=1,
            clipped_to_window=False,
        )


def _hand(side: str, x: float, confidence: float = 0.9) -> Assembly101Hand:
    index = 0 if side == "left" else 1
    return Assembly101Hand(
        hand_index=index,  # type: ignore[arg-type]
        side=side,  # type: ignore[arg-type]
        confidence=confidence,
        joints_world_mm=tuple(Assembly101Point3D(x=float(j), y=0.0, z=0.0) for j in range(21)),
        joints_proxy_pixels=tuple(Assembly101Point2D(x=x + j, y=100.0) for j in range(21)),
        joints_inside_image=21,
    )


def test_hand_side_must_match_the_dataset_hand_index() -> None:
    with pytest.raises(ValueError, match="disagree"):
        Assembly101Hand(
            hand_index=0,
            side="right",
            confidence=1.0,
            joints_world_mm=tuple(Assembly101Point3D(x=0, y=0, z=0) for _ in range(21)),
            joints_proxy_pixels=tuple(Assembly101Point2D(x=0, y=0) for _ in range(21)),
            joints_inside_image=0,
        )


def _fake_repository(tmp_path: Path, *, corrupt: bool = False) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    run = root / "runs/a101"
    run.mkdir(parents=True)
    source = root / "data/source.json"
    source.parent.mkdir(parents=True)
    source.write_text("{}")
    frames = [
        Assembly101HandFrame(
            analysis_frame_index=i,
            pose_frame_index=109 + 2 * i,
            pose_timestamp_seconds=1.0,
            source_seconds=294.0 + i / 30,
            hands=(_hand("left", 10.0),),
        )
        for i in range(2)
    ]
    hands = run / "hands.jsonl"
    hands.write_text("".join(f.model_dump_json() + "\n" for f in frames))

    def fingerprint(path: Path) -> ArtifactFingerprint:
        return ArtifactFingerprint(
            uri=path.relative_to(root).as_posix(),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            source="measured",
        )

    manifest = Assembly101ReferenceManifest(
        manifest_kind="assembly101_reference_window",
        recording_id="rec",
        dataset_revision="abc",
        license="CC BY-NC 4.0",
        citation="cite",
        view_key="C10379:rgb",
        proxy_dimensions=(1280, 720),
        frame_count=2,
        analysis_fps=30,
        source_start_seconds=294.0,
        annotation_start_frame=8820,
        clock_rule=_rule(9),
        camera=_camera(),
        draw_confidence_threshold=0.5,
        input_artifacts=(fingerprint(source),),
        hands_path="hands.jsonl",
        fine_segments=(),
        coverage={"frames": 2},
        claim_boundaries=("fixture",),
        hands_fingerprint=fingerprint(hands),
    )
    (run / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    if corrupt:
        hands.write_text(hands.read_text().replace("294.0", "294.5", 1))
    return root, Path("runs/a101")


def test_load_reference_reads_frames_and_refuses_changed_inputs(tmp_path: Path) -> None:
    root, run = _fake_repository(tmp_path)
    loaded = a101.load_reference(run, root, frame_count=2, verify=True)
    assert sorted(loaded.frames) == [0, 1]
    assert loaded.frames[1].pose_frame_index == 111
    with pytest.raises(ValueError, match="frames"):
        a101.load_reference(run, root, frame_count=3, verify=True)
    root, run = _fake_repository(tmp_path / "second", corrupt=True)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        a101.load_reference(run, root, frame_count=2, verify=True)


def _referenced(blueprint) -> set[str]:
    referenced: set[str] = set()

    def walk(node) -> None:
        for child in getattr(node, "contents", []) or []:
            walk(child)
        if hasattr(node, "root_container"):
            walk(node.root_container)

    def views(node):
        kind = type(node).__name__
        if kind.endswith("View"):
            yield node
        for child in getattr(node, "contents", []) or []:
            if not isinstance(child, str):
                yield from views(child)
        if hasattr(node, "root_container"):
            yield from views(node.root_container)

    for view in views(blueprint):
        kind = type(view).__name__
        if kind == "TextDocumentView":
            referenced.add(str(view.origin))
        elif kind == "TimeSeriesView" and isinstance(view.contents, tuple):
            for item in view.contents:
                referenced.add(str(item).replace("$origin", str(view.origin)))
    return referenced


def test_v4_blueprint_with_dataset_layer_references_only_logged_entities(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from types import SimpleNamespace

    from battle.fine_substep_contract import load_contract

    root_dir, run = _fake_repository(tmp_path)
    loaded = a101.load_reference(run, root_dir, frame_count=2)
    manifest = loaded.manifest
    root = "world/clip/interaction_review_v4"
    logged: set[str] = set()
    cleared: set[str] = set()

    def record(path, value, **_):
        (cleared if value.__class__.__name__ == "Clear" else logged).add(path)

    monkeypatch.setattr("battle.interaction_review.rr.log", record)
    monkeypatch.setattr("battle.interaction_review_v4.rr.log", record)
    contract = load_contract(FINE_LABELS)
    _log_static_documents(root, guide="# g", fine_contract=contract, assembly101=manifest)
    _log_assembly101_static(root, manifest)
    wilor = SimpleNamespace(
        hands=(SimpleNamespace(landmarks=(SimpleNamespace(x=0.01, y=0.14),) * 21),)
    )
    _log_assembly101_frame(
        root, loaded.frames[0], wilor, dimensions=(1280, 720), draw_threshold=0.5
    )
    segment = Assembly101FineSegment(
        annotation_id="1",
        action_id=1,
        verb="position",
        noun="interior",
        action="position interior",
        annotation_start_frame=8916,
        annotation_end_frame=9143,
        proxy_start_frame=96,
        proxy_end_frame_exclusive=323,
        clipped_to_window=False,
    )
    _log_navigation_frame(
        root, 100, source_seconds=297.3, substep=None, coarse_gt=None, fine_gt=((0, segment),)
    )
    referenced = _referenced(
        _blueprint(root, (1280, 720), static_text_panels=STATIC_TEXT_PANELS, assembly101=True)
    )
    assert f"{root}/{NAVIGATION_FINE_GT_INDEX}" in referenced
    assert f"{root}/{NAVIGATION_SUBSTEP_INDEX}" not in referenced
    assert f"{root}/metadata/fine_grained_gt" in referenced
    assert referenced <= logged, referenced - logged
    assert f"{root}/comparison/assembly101_hands_2d/skeletons" in logged
    assert f"{root}/contexts/assembly101_world_mm_3d/hands/joints" in logged
    assert f"{root}/contexts/assembly101_world_mm_3d/camera/C10379" in logged
    assert f"{root}/diagnostics/assembly101/confidence/left" in logged
    # No right hand this frame: its series clear rather than hold a stale value.
    assert f"{root}/diagnostics/assembly101/confidence/right" in cleared


def test_navigation_frame_without_fine_gt_clears_its_series_and_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logged: list[tuple[str, str, object]] = []
    monkeypatch.setattr(
        "battle.interaction_review.rr.log",
        lambda path, value, **_: logged.append((path, value.__class__.__name__, value)),
    )
    _log_navigation_frame(
        "world/x", 5, source_seconds=294.2, substep=None, coarse_gt=None, fine_gt=()
    )
    assert ("world/x/metadata/navigation/fine_gt_index", "Clear") in [(p, k) for p, k, _ in logged]
    document = next(v for p, k, v in logged if k == "TextDocument")
    assert "no segment covers this frame" in document.text.as_arrow_array().to_pylist()[0]


@pytest.mark.real_data
def test_built_reference_window_matches_the_acquisition_report() -> None:
    root = require_artifact(a101.OUTPUT_ROOT)
    manifest = Assembly101ReferenceManifest.model_validate_json(
        (root / a101.MANIFEST_NAME).read_text()
    )
    assert manifest.frame_count == 1800
    assert manifest.clock_rule.pose_offset_frames == 9
    assert manifest.clock_rule.pose_frame(0) == 17649
    assert len(manifest.fine_segments) == 27
    actions = {segment.action for segment in manifest.fine_segments}
    assert {"position interior", "screw chassis with screwdriver", "position rear body"} <= actions
    assert manifest.projection_check is not None
    assert manifest.projection_check.rms_pixels < 0.01
    assert manifest.coverage["frames_with_any_hand"] == 1800
    loaded = a101.load_reference(a101.OUTPUT_ROOT, Path.cwd())
    assert len(loaded.frames) == 1800


@pytest.mark.real_data
def test_built_v4_index_cites_the_dataset_reference() -> None:
    from battle.interaction_review_v4 import INDEX_NAME, OUTPUT_ROOT

    index = json.loads((require_artifact(OUTPUT_ROOT) / INDEX_NAME).read_text())
    assert index["assembly101_reference"] is not None
    assert index["assembly101_reference"]["uri"].endswith("manifest.json")
    assert index["coverage"]["assembly101_fine_segments"] == 27
