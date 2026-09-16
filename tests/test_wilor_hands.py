from __future__ import annotations

import json
from pathlib import Path

from battle.schemas import (
    CameraRelativePoint3D,
    FrameObservations,
    HandSide,
    NormalizedBox,
    NormalizedPoint,
    PerFrameHand,
    WiLoRHandsRunMetadata,
)
from battle.wilor_hands import _load_observations


def test_load_observations_maps_wilor_3d_joints(tmp_path: Path) -> None:
    observations_path = tmp_path / "observations.jsonl"
    observations_path.write_text(
        json.dumps(
            {
                "view_id": "static-c10379",
                "analysis_frame_index": 0,
                "source_seconds": 294.0,
                "hands": [
                    {
                        "hand_id": "hand-detection-1",
                        "side": "right",
                        "confidence": 0.91,
                        "landmarks": [{"x": 0.1, "y": 0.2} for _ in range(21)],
                        "box": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4},
                        "model_side": "right",
                        "model_handedness_confidence": 0.91,
                        "joints_3d_camera_relative": [
                            {"x": 0.01, "y": 0.02, "z": 0.03} for _ in range(21)
                        ],
                    }
                ],
            }
        )
        + "\n"
    )

    observations = _load_observations(observations_path)

    assert len(observations) == 1
    hand = observations[0].hands[0]
    assert hand.side is HandSide.RIGHT
    assert hand.joints_3d_camera_relative is not None
    assert len(hand.joints_3d_camera_relative) == 21
    assert hand.joints_3d_camera_relative[0] == CameraRelativePoint3D(x=0.01, y=0.02, z=0.03)


def test_wilor_metadata_requires_matching_frame_budget() -> None:
    from battle.schemas import (
        AdapterMetadata,
        ArtifactFingerprint,
        FrameRange,
        RuntimeMeasurements,
    )

    metadata = WiLoRHandsRunMetadata(
        requested_analysis_frame_range=FrameRange(start_frame=0, end_frame_exclusive=600),
        requested_seconds=20.0,
        source_fingerprint=ArtifactFingerprint(
            uri="data/raw/example.mp4", sha256="a" * 64, source="approved_config"
        ),
        proxy_fingerprint=ArtifactFingerprint(
            uri="data/derived/example.mp4", sha256="b" * 64, source="approved_config"
        ),
        config_fingerprint=ArtifactFingerprint(
            uri="configs/example.json", sha256="c" * 64, source="measured"
        ),
        checkpoint_fingerprint=ArtifactFingerprint(
            uri="models/wilor.ckpt", sha256="d" * 64, source="measured"
        ),
        detector_fingerprint=ArtifactFingerprint(
            uri="models/detector.pt", sha256="e" * 64, source="measured"
        ),
        adapter=AdapterMetadata(
            name="wilor-hand-pose",
            version="0.1.0",
            implementation_basis="fixture",
            external_source_uri="/home/nick/src/WiLoR",
        ),
        runtime_settings={"analysis_fps": 30},
        measurements=RuntimeMeasurements(elapsed_seconds=1.0),
        observations_uri="runs/example/observations.jsonl",
    )

    assert metadata.requested_analysis_frame_range.frame_count == 600


def test_frame_observations_accept_optional_3d_joints() -> None:
    landmarks = tuple(NormalizedPoint(x=0.1, y=0.2) for _ in range(21))
    joints = tuple(CameraRelativePoint3D(x=0.01, y=0.02, z=0.03) for _ in range(21))
    observation = FrameObservations(
        view_id="static-c10379",
        analysis_frame_index=0,
        source_seconds=294.0,
        hands=(
            PerFrameHand(
                hand_id="hand-detection-1",
                side=HandSide.LEFT,
                confidence=0.8,
                landmarks=landmarks,
                box=NormalizedBox(x=0.1, y=0.2, width=0.3, height=0.4),
                model_side=HandSide.LEFT,
                model_handedness_confidence=0.8,
                joints_3d_camera_relative=joints,
            ),
        ),
    )

    assert observation.hands[0].joints_3d_camera_relative == joints
