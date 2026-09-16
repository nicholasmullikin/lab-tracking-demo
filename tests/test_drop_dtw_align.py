from __future__ import annotations

from pathlib import Path

from battle.assembly101_gt_transcript import parse_coarse_transcript


def test_parse_coarse_transcript_returns_overlapping_steps() -> None:
    labels_path = Path(
        "data/raw/assembly101/nusar-2021_action_both_9033-c02a_9033_user_id_2021-02-04_140532/"
        "annotations/coarse-annotations/coarse_labels/"
        "assembly_nusar-2021_action_both_9033-c02a_9033_user_id_2021-02-04_140532.txt"
    )

    steps = parse_coarse_transcript(
        labels_path,
        annotation_start_frame=int(294 * 30),
        annotation_end_frame=int((294 + 20) * 30),
    )

    assert len(steps) >= 1
    assert all(step.action for step in steps)
    assert all(step.overlap_frame_count > 0 for step in steps)
