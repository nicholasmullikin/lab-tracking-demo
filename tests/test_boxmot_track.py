from __future__ import annotations

import json
from pathlib import Path

from battle.boxmot_track import _load_observations
from battle.boxmot_worker import _normalize_box
from battle.schemas import BoxMOTRunMetadata, NormalizedBox


def test_normalize_box_clamps_to_frame() -> None:
    box = _normalize_box(10, 20, 1300, 800, 1280, 720)

    assert box["x"] + box["width"] <= 1.0
    assert box["y"] + box["height"] <= 1.0


def test_load_observations_maps_boxmot_objects(tmp_path: Path) -> None:
    observations_path = tmp_path / "observations.jsonl"
    observations_path.write_text(
        json.dumps(
            {
                "view_id": "static-c10379",
                "analysis_frame_index": 0,
                "source_seconds": 294.0,
                "objects": [
                    {
                        "object_id": "boxmot-1",
                        "label": "person",
                        "confidence": 0.88,
                        "box": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4},
                    }
                ],
            }
        )
        + "\n"
    )

    observations = _load_observations(observations_path)

    assert len(observations) == 1
    assert observations[0].objects[0].object_id == "boxmot-1"
    assert observations[0].objects[0].box == NormalizedBox(x=0.1, y=0.2, width=0.3, height=0.4)


def test_boxmot_metadata_requires_detector_source_field() -> None:
    from battle.schemas import (
        AdapterMetadata,
        ArtifactFingerprint,
        FrameRange,
        RuntimeMeasurements,
    )

    metadata = BoxMOTRunMetadata(
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
        detector_fingerprint=ArtifactFingerprint(
            uri="models/yolo/yolov8n.pt", sha256="d" * 64, source="measured"
        ),
        adapter=AdapterMetadata(
            name="boxmot-botsort",
            version="25.0.0",
            implementation_basis="fixture",
            external_source_uri="https://github.com/mikel-brostrom/boxmot",
        ),
        runtime_settings={"analysis_fps": 30},
        measurements=RuntimeMeasurements(elapsed_seconds=1.0),
        observations_uri="runs/example/observations.jsonl",
        detector_source="ultralytics YOLOv8n per-frame detections",
    )

    assert "YOLOv8n" in metadata.detector_source
