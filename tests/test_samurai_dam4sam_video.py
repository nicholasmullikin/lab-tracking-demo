from __future__ import annotations

import json
from pathlib import Path

from battle.dam4sam_video_worker import INIT_BBOX_XYWH as DAM4SAM_INIT_BBOX
from battle.dam4sam_video_worker import _normalize_box as dam4sam_normalize_box
from battle.samurai_video import INIT_BBOX_XYWH as SAMURAI_INIT_BBOX
from battle.samurai_video import _load_observations
from battle.samurai_video_worker import _normalize_box as samurai_normalize_box
from battle.schemas import (
    AdapterMetadata,
    ArtifactFingerprint,
    Dam4samVideoRunMetadata,
    FrameRange,
    MaskReference,
    MaskStorage,
    NormalizedBox,
    RuntimeMeasurements,
    SamuraiVideoRunMetadata,
)


def test_samurai_and_dam4sam_share_deterministic_hand_seed() -> None:
    assert SAMURAI_INIT_BBOX == (881, 446, 152, 129)
    assert DAM4SAM_INIT_BBOX == SAMURAI_INIT_BBOX


def test_samurai_worker_normalize_box_matches_adapter() -> None:
    adapter_box = samurai_normalize_box(881, 446, 1034, 576, 1280, 720)
    worker_box = dam4sam_normalize_box(881, 446, 1034, 576, 1280, 720)

    assert adapter_box == worker_box
    assert NormalizedBox(**adapter_box).width > 0


def test_load_samurai_observations_maps_masks(tmp_path: Path) -> None:
    observations_path = tmp_path / "observations.jsonl"
    observations_path.write_text(
        json.dumps(
            {
                "view_id": "static-c10379",
                "analysis_frame_index": 0,
                "source_seconds": 294.0,
                "objects": [
                    {
                        "object_id": "samurai-0",
                        "label": "seeded_hand",
                        "confidence": 1.0,
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


def test_samurai_metadata_requires_samurai_config_and_seed() -> None:
    metadata = SamuraiVideoRunMetadata(
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
        sam2_checkpoint_fingerprint=ArtifactFingerprint(
            uri="/checkpoints/sam2.1_hiera_tiny.pt", sha256="e" * 64, source="measured"
        ),
        adapter=AdapterMetadata(
            name="samurai_sam2_video_smoke",
            version="0.1.0",
            implementation_basis="fixture",
            external_source_uri="/home/nick/src/samurai",
            external_revision="deadbeef",
        ),
        runtime_settings={"analysis_fps": 30, "samurai_mode": True},
        measurements=RuntimeMeasurements(elapsed_seconds=1.0),
        observations_uri="runs/example/observations.jsonl",
        native_masks_uri="runs/example/native/masks",
        sam2_model_config="configs/samurai/sam2.1_hiera_t.yaml",
        init_bbox_xywh=(881, 446, 152, 129),
        initialization_note="fixture",
    )

    assert metadata.runtime_settings["samurai_mode"] is True


def test_dam4sam_metadata_requires_tracker_and_compatibility_note() -> None:
    metadata = Dam4samVideoRunMetadata(
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
        dam4sam_config_fingerprint=ArtifactFingerprint(
            uri="/home/nick/src/DAM4SAM/dam4sam_config.yaml",
            sha256="d" * 64,
            source="measured",
        ),
        sam2_checkpoint_fingerprint=ArtifactFingerprint(
            uri="/checkpoints/sam2.1_hiera_tiny.pt", sha256="e" * 64, source="measured"
        ),
        adapter=AdapterMetadata(
            name="dam4sam_video_smoke",
            version="0.1.0",
            implementation_basis="fixture",
            external_source_uri="/home/nick/src/DAM4SAM",
            external_revision="deadbeef",
        ),
        runtime_settings={"analysis_fps": 30, "tracker_name": "sam21pp-T"},
        measurements=RuntimeMeasurements(elapsed_seconds=1.0),
        observations_uri="runs/example/observations.jsonl",
        native_masks_uri="runs/example/native/masks",
        tracker_name="sam21pp-T",
        sam2_model_config="sam21pp_hiera_t.yaml",
        init_bbox_xywh=(881, 446, 152, 129),
        initialization_note="fixture",
        compatibility_note="fixture",
    )

    assert metadata.tracker_name == "sam21pp-T"
