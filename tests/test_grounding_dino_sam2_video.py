from __future__ import annotations

import json
from pathlib import Path

from battle.grounding_dino_sam2_video import _load_observations, _normalize_box
from battle.grounding_dino_sam2_video_worker import _normalize_box as worker_normalize_box
from battle.schemas import (
    AdapterMetadata,
    ArtifactFingerprint,
    FrameRange,
    GroundingDinoSam2VideoRunMetadata,
    MaskReference,
    MaskStorage,
    NormalizedBox,
    RuntimeMeasurements,
)


def test_normalize_box_clamps_to_frame() -> None:
    box = _normalize_box(10, 20, 1300, 800, 1280, 720)

    assert box.x + box.width <= 1.0
    assert box.y + box.height <= 1.0


def test_worker_normalize_box_matches_adapter() -> None:
    adapter_box = _normalize_box(881, 446, 1034, 576, 1280, 720)
    worker_box = worker_normalize_box(881, 446, 1034, 576, 1280, 720)

    assert adapter_box == NormalizedBox(**worker_box)


def test_load_observations_maps_masks(tmp_path: Path) -> None:
    observations_path = tmp_path / "observations.jsonl"
    observations_path.write_text(
        json.dumps(
            {
                "view_id": "static-c10379",
                "analysis_frame_index": 0,
                "source_seconds": 294.0,
                "objects": [
                    {
                        "object_id": "grounding-sam2-1",
                        "label": "hand",
                        "confidence": 0.953125,
                        "box": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4},
                        "mask": {
                            "uri": "native/masks/00000.png",
                            "storage": "native_artifact",
                            "format": "png",
                        },
                    }
                ],
            }
        )
        + "\n"
    )

    observations = _load_observations(observations_path)

    assert len(observations) == 1
    assert observations[0].objects[0].mask == MaskReference(
        uri="native/masks/00000.png",
        storage=MaskStorage.NATIVE_ARTIFACT,
        format="png",
    )


def test_grounding_dino_sam2_metadata_requires_prompt_and_config() -> None:
    metadata = GroundingDinoSam2VideoRunMetadata(
        requested_analysis_frame_range=FrameRange(start_frame=0, end_frame_exclusive=300),
        requested_seconds=10.0,
        source_fingerprint=ArtifactFingerprint(
            uri="data/raw/example.mp4", sha256="a" * 64, source="approved_config"
        ),
        proxy_fingerprint=ArtifactFingerprint(
            uri="data/derived/example.mp4", sha256="b" * 64, source="approved_config"
        ),
        config_fingerprint=ArtifactFingerprint(
            uri="configs/example.json", sha256="c" * 64, source="measured"
        ),
        grounding_model_fingerprint=ArtifactFingerprint(
            uri="hf://IDEA-Research/grounding-dino-tiny@rev",
            sha256="d" * 64,
            source="measured",
        ),
        sam2_checkpoint_fingerprint=ArtifactFingerprint(
            uri="/checkpoints/sam2.1_hiera_tiny.pt", sha256="e" * 64, source="measured"
        ),
        adapter=AdapterMetadata(
            name="transformers_grounding_dino_plus_sam2_video_smoke",
            version="0.1.0",
            implementation_basis="fixture",
            external_source_uri="/home/nick/src/Grounded-SAM-2",
            external_revision="deadbeef",
        ),
        runtime_settings={"analysis_fps": 30},
        measurements=RuntimeMeasurements(elapsed_seconds=1.0),
        observations_uri="runs/example/observations.jsonl",
        native_masks_uri="runs/example/native/masks",
        text_prompt="hand.",
        sam2_model_config="configs/sam2.1/sam2.1_hiera_t.yaml",
        initialization_note="fixture",
    )

    assert metadata.text_prompt == "hand."
