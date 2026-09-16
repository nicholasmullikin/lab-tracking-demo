"""Synthetic normalized fixtures; they contain no recordings or model outputs."""

from __future__ import annotations

from .schemas import (
    ChunkContinuityPolicy,
    ClipManifest,
    ClockMapping,
    ClockName,
    ClockSet,
    ClockSpec,
    EncodedAssetInput,
    FrameObservations,
    FullDurationCoverage,
    HandSide,
    MaskReference,
    MaskStorage,
    MethodState,
    MethodStatus,
    NormalizedBox,
    NormalizedPoint,
    PerFrameHand,
    PerFrameObject,
    RunManifest,
    TimeInterval,
    TimingModel,
)


def synthetic_timing() -> TimingModel:
    clocks = ClockSet(
        clocks=(
            ClockSpec(name=ClockName.SOURCE, fps=60),
            ClockSpec(name=ClockName.ANALYSIS, fps=30),
            ClockSpec(name=ClockName.ANNOTATION, fps=30),
            ClockSpec(name=ClockName.POSE, fps=60),
        )
    )
    return TimingModel(
        clocks=clocks,
        mappings=tuple(
            ClockMapping(clock=clock.name, source_offset_seconds=0.0, scale=1.0)
            for clock in clocks.clocks
        ),
    )


def synthetic_clip_manifest() -> ClipManifest:
    return ClipManifest(
        clip_id="synthetic-assembly101-0001",
        source_name="Assembly101 fixture placeholder",
        source_license="CC BY-NC 4.0 (verify before real use)",
        asset=EncodedAssetInput(
            uri="assets/synthetic-assembly101-0001.mp4",
            media_type="video/mp4",
        ),
        timing=synthetic_timing(),
        source_duration_seconds=3.0,
        views=("ego-01", "static-01"),
    )


def synthetic_run_manifest() -> RunManifest:
    clip = synthetic_clip_manifest()
    observations = tuple(
        FrameObservations(
            view_id="ego-01",
            analysis_frame_index=frame_index,
            source_seconds=clip.timing.source_seconds_for_frame(ClockName.ANALYSIS, frame_index),
            objects=(
                PerFrameObject(
                    object_id="tool-1",
                    label="synthetic screwdriver",
                    confidence=0.9,
                    box=NormalizedBox(x=0.1 + frame_index * 0.01, y=0.2, width=0.2, height=0.3),
                    mask=MaskReference(
                        uri=f"masks/{clip.clip_id}/ego-01/{frame_index:06d}.png",
                        storage=MaskStorage.EXTERNAL_ARTIFACT,
                        format="png",
                    ),
                ),
            ),
            hands=(
                PerFrameHand(
                    hand_id="hand-left-1",
                    side=HandSide.LEFT,
                    confidence=0.8,
                    landmarks=tuple(
                        NormalizedPoint(x=0.2 + index * 0.001, y=0.2 + index * 0.001)
                        for index in range(21)
                    ),
                    box=NormalizedBox(x=0.2, y=0.2, width=0.02, height=0.02),
                    model_side=HandSide.LEFT,
                    model_handedness_confidence=0.8,
                ),
            ),
        )
        for frame_index in range(3)
    )
    return RunManifest(
        run_id="synthetic-session-1",
        clip=clip,
        coverage=FullDurationCoverage(
            source_duration_seconds=clip.source_duration_seconds,
            covered_intervals=(TimeInterval(start_seconds=0.0, end_seconds=3.0),),
        ),
        chunk_policy=ChunkContinuityPolicy(
            chunk_duration_seconds=2.0,
            overlap_seconds=0.5,
            max_allowed_gap_seconds=0.0,
        ),
        method_statuses=(
            MethodStatus(
                method_name="fixture-rerun-export",
                stage="export",
                state=MethodState.SUCCEEDED,
                artifact_uri="artifacts/synthetic_fixture.rrd",
                measured_on="synthetic normalized fixture",
            ),
            MethodStatus(
                method_name="segmentation-adapter",
                stage="objects",
                state=MethodState.NOT_RUN,
                blocker="not implemented in session one",
            ),
        ),
        observations=observations,
    )
