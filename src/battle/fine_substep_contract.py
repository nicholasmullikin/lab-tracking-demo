"""Load and validate agent-authored fine substep label contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .schemas import TimeInterval, VersionedModel

PROVENANCE_TAG = "agent_authored_visual_review"
FRAME_COUNT = 600
ANALYSIS_FPS = 30


class FineSubstepCoarseGtAnchor(VersionedModel):
    action: str = Field(min_length=1)
    annotation_start_frame: int = Field(ge=0)
    annotation_end_frame: int = Field(gt=0)
    proxy_start_frame: int = Field(ge=0, lt=FRAME_COUNT)
    proxy_end_frame_exclusive: int = Field(gt=0, le=FRAME_COUNT)


class FineSubstepCheckpoint(VersionedModel):
    analysis_frame_index: int = Field(ge=0, lt=FRAME_COUNT)
    source_seconds: float = Field(ge=0)
    note: str = Field(min_length=1)


class FineSubstepDefinition(VersionedModel):
    substep_id: str = Field(pattern=r"^S\d{2}$")
    label: str = Field(min_length=1)
    start_frame: int = Field(ge=0, lt=FRAME_COUNT)
    end_frame_exclusive: int = Field(gt=0, le=FRAME_COUNT)
    source_start_seconds: float = Field(ge=0)
    source_end_seconds_exclusive: float = Field(gt=0)
    confidence: Literal["high", "medium", "low"]
    coarse_gt_anchor: str = Field(min_length=1)
    evidence_frames: tuple[int, ...] = Field(min_length=1)
    prompts: tuple[str, ...] = Field(min_length=3, max_length=5)

    @model_validator(mode="after")
    def validate_interval(self) -> FineSubstepDefinition:
        if self.end_frame_exclusive <= self.start_frame:
            raise ValueError("substep interval must be positive")
        for frame in self.evidence_frames:
            if frame < self.start_frame or frame >= self.end_frame_exclusive:
                raise ValueError("evidence frame must fall inside substep interval")
        return self


class FineSubstepAgentLabelContract(VersionedModel):
    manifest_kind: Literal["fine_substep_agent_labels"]
    clip_id: str = Field(min_length=1)
    view_id: str = Field(min_length=1)
    provenance_tag: Literal["agent_authored_visual_review"]
    author_type: Literal["agent"]
    analysis_fps: Literal[30]
    frame_count: Literal[600]
    source_interval: TimeInterval
    claim_boundaries: tuple[str, ...] = Field(min_length=1)
    coarse_gt_anchors: tuple[FineSubstepCoarseGtAnchor, ...] = Field(min_length=1)
    checkpoints: tuple[FineSubstepCheckpoint, ...] = Field(min_length=1)
    substeps: tuple[FineSubstepDefinition, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_partition(self) -> FineSubstepAgentLabelContract:
        if self.provenance_tag != PROVENANCE_TAG:
            raise ValueError("fine substep labels must use agent_authored_visual_review provenance")
        ordered = sorted(self.substeps, key=lambda item: item.start_frame)
        if ordered[0].start_frame != 0:
            raise ValueError("substeps must start at frame 0")
        if ordered[-1].end_frame_exclusive != FRAME_COUNT:
            raise ValueError("substeps must end at frame 600")
        for left, right in zip(ordered, ordered[1:], strict=False):
            if left.end_frame_exclusive != right.start_frame:
                raise ValueError("substeps must partition [0,600) without gaps or overlap")
        return self.model_copy(update={"substeps": tuple(ordered)})


def load_contract(path: Path) -> FineSubstepAgentLabelContract:
    payload = json.loads(path.read_text(encoding="utf-8"))
    contract = FineSubstepAgentLabelContract.model_validate(payload)
    if contract.provenance_tag != PROVENANCE_TAG:
        raise ValueError("refusing non-agent-authored label contract")
    return contract


def substep_for_frame(
    contract: FineSubstepAgentLabelContract, frame_index: int
) -> FineSubstepDefinition:
    for substep in contract.substeps:
        if substep.start_frame <= frame_index < substep.end_frame_exclusive:
            return substep
    raise ValueError(f"frame {frame_index} is outside agent label partition")
