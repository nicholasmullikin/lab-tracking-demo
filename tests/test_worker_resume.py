from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from battle.muggled_smoke import checkpoint_path_for, prepare_resume
from battle.schemas import ArtifactFingerprint, StreamContinuityPolicy


def _prior_run(tmp_path: Path, *, frames: int, checkpoint_at: int | None) -> Path:
    run = tmp_path / "prior"
    (run / "masks").mkdir(parents=True)
    mask = np.zeros((8, 12), dtype=np.uint8)
    mask[2:5, 3:7] = 255
    with (run / "observations.jsonl").open("w") as output:
        for frame in range(frames):
            uri = f"masks/{frame:06d}_00.png"
            Image.fromarray(mask).save(run / uri)
            output.write(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "view_id": "static-c10379",
                        "analysis_frame_index": frame,
                        "source_seconds": 294.0 + frame / 30,
                        "objects": [
                            {
                                "schema_version": "1.0",
                                "object_id": "sam3-00",
                                "label": "chassis",
                                "confidence": 1.0,
                                "box": {
                                    "schema_version": "1.0",
                                    "x": 0.25,
                                    "y": 0.25,
                                    "width": 0.33,
                                    "height": 0.37,
                                },
                                "mask": {
                                    "schema_version": "1.0",
                                    "uri": uri,
                                    "storage": "external_artifact",
                                    "format": "png",
                                },
                            }
                        ],
                        "hands": [],
                    }
                )
                + "\n"
            )
    if checkpoint_at is not None:
        path = checkpoint_path_for(run, checkpoint_at)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"tracker state")
    return run


def test_resume_copies_the_prefix_and_fingerprints_the_checkpoint(tmp_path: Path) -> None:
    prior = _prior_run(tmp_path, frames=10, checkpoint_at=6)
    run = tmp_path / "resumed"

    checkpoint, fingerprint = prepare_resume(
        prior_run=prior, resume_at=6, run_directory=run, repository_root=tmp_path
    )

    assert checkpoint == checkpoint_path_for(prior, 6)
    assert fingerprint.uri == "prior/native/checkpoints/f000006.pt"
    rows = (run / "observations.jsonl").read_text().splitlines()
    assert len(rows) == 6
    assert [json.loads(row)["analysis_frame_index"] for row in rows] == list(range(6))
    prior_rows = (prior / "observations.jsonl").read_text().splitlines()
    assert rows == prior_rows[:6]
    for frame in range(6):
        copied = run / f"masks/{frame:06d}_00.png"
        assert copied.read_bytes() == (prior / f"masks/{frame:06d}_00.png").read_bytes()
    assert not (run / "masks/000006_00.png").exists()


def test_a_missing_checkpoint_names_how_to_make_one(tmp_path: Path) -> None:
    prior = _prior_run(tmp_path, frames=10, checkpoint_at=None)

    with pytest.raises(ValueError, match="no tracker checkpoint for frame 6"):
        prepare_resume(
            prior_run=prior,
            resume_at=6,
            run_directory=tmp_path / "resumed",
            repository_root=tmp_path,
        )


def test_a_short_prior_run_cannot_be_resumed(tmp_path: Path) -> None:
    prior = _prior_run(tmp_path, frames=4, checkpoint_at=6)

    with pytest.raises(ValueError, match="one row for every earlier frame"):
        prepare_resume(
            prior_run=prior,
            resume_at=6,
            run_directory=tmp_path / "resumed",
            repository_root=tmp_path,
        )


def test_a_resumed_stream_must_declare_where_it_resumed() -> None:
    fingerprint = ArtifactFingerprint(uri="prior/f.pt", sha256="a" * 64, source="measured")

    policy = StreamContinuityPolicy(
        mode="checkpoint_resumed",
        max_prompt_memory_entries=1,
        max_frame_memory_entries=4,
        detected_object_limit=4,
        resumed_from_run="runs/prior",
        resumed_at_frame=1172,
        checkpoint_fingerprint=fingerprint,
    )
    assert policy.resumed_at_frame == 1172

    with pytest.raises(ValueError, match="must name its prior run"):
        StreamContinuityPolicy(
            mode="checkpoint_resumed",
            max_prompt_memory_entries=1,
            max_frame_memory_entries=4,
            detected_object_limit=4,
            resumed_at_frame=1172,
        )


def test_a_continuous_stream_cannot_claim_resume_provenance() -> None:
    with pytest.raises(ValueError, match="cannot claim resume provenance"):
        StreamContinuityPolicy(
            max_prompt_memory_entries=1,
            max_frame_memory_entries=4,
            detected_object_limit=4,
            resumed_at_frame=10,
        )
