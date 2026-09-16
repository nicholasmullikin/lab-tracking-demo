"""Build ordered Assembly101 coarse-action transcripts for weak-supervision alignment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TranscriptStep:
    action: str
    annotation_start_frame: int
    annotation_end_frame: int
    overlap_start_frame: int
    overlap_end_frame: int

    @property
    def overlap_frame_count(self) -> int:
        return self.overlap_end_frame - self.overlap_start_frame


def parse_coarse_transcript(
    labels_path: Path,
    *,
    annotation_start_frame: int,
    annotation_end_frame: int,
) -> tuple[TranscriptStep, ...]:
    """Return ordered coarse steps overlapping the requested annotation-frame interval."""
    if annotation_end_frame <= annotation_start_frame:
        raise ValueError("annotation interval must be positive")
    steps: list[TranscriptStep] = []
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        start_frame = int(parts[0])
        end_frame = int(parts[1])
        action = parts[2].strip()
        if end_frame <= annotation_start_frame or start_frame >= annotation_end_frame:
            continue
        overlap_start = max(start_frame, annotation_start_frame)
        overlap_end = min(end_frame, annotation_end_frame)
        if overlap_end <= overlap_start:
            continue
        steps.append(
            TranscriptStep(
                action=action,
                annotation_start_frame=start_frame,
                annotation_end_frame=end_frame,
                overlap_start_frame=overlap_start,
                overlap_end_frame=overlap_end,
            )
        )
    return tuple(steps)
