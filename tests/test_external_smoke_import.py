from __future__ import annotations

from battle.external_smoke_import import _normalize_box
from battle.schemas import ExternalPartialRunMetadata


def test_normalize_external_box_clips_to_image_boundaries() -> None:
    box = _normalize_box(-10, 20, 1300, 800, 1280, 720)

    assert box.x == 0
    assert box.y > 0
    assert box.x + box.width == 1
    assert box.y + box.height == 1


def test_external_partial_contract_requires_native_provenance() -> None:
    payload = {
        "classification": "external_partial",
        "requested_input_fingerprint": {
            "uri": "data/input.mp4",
            "sha256": "a" * 64,
            "source": "measured",
        },
        "native_artifact_fingerprints": [
            {
                "uri": "runs/native-index.json",
                "sha256": "b" * 64,
                "source": "measured",
            }
        ],
        "adapter": {
            "name": "test-import",
            "version": "0.1.0",
            "implementation_basis": "fixture",
            "external_source_uri": "https://example.invalid/upstream",
            "external_revision": "deadbeef",
        },
        "reproduced_command": "upstream-command --input data/input.mp4",
        "decoded_frame_count": 2,
        "source_offset_seconds": 294.0,
        "normalized_artifact_uri": "runs/import/observations.jsonl",
        "limitations": ["external smoke only"],
    }

    metadata = ExternalPartialRunMetadata.model_validate(payload)

    assert metadata.classification == "external_partial"
    assert metadata.decoded_frame_count == 2
