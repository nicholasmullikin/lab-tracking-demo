"""Create a provenance-preserving, WiLoR-primary stabilized hand review layer."""

from __future__ import annotations

import argparse
from pathlib import Path

from .exporter import export_run
from .hand_stabilization import enrich_metrics, stabilize, write_result
from .schemas import EncodedAssetInput, FrameObservations, RunManifest

DEFAULT_WILOR = Path("runs/wilor-hands-static-20s-audited-source-state")
DEFAULT_MEDIAPIPE = Path("runs/mediapipe-hands-static-20s-fused-dedup-th035-20260916t0428z")
DEFAULT_PARTS = Path(
    "runs/muggledsam-sam3-four-part-static-focused-reassembly-static-c10379-20260916t023700z"
)
DEFAULT_OUTPUT = Path("runs/wilor-hands-stabilized-20s-overnight-v2")


def _observations(path: Path, *, frame_count: int) -> dict[int, FrameObservations]:
    rows = [
        FrameObservations.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    result = {item.analysis_frame_index: item for item in rows}
    if not set(range(frame_count)).issubset(result):
        raise ValueError(f"{path} must contain all frames [0,{frame_count})")
    return {index: result[index] for index in range(frame_count)}


def run(args: argparse.Namespace) -> Path:
    root = args.repository_root.resolve()
    source_dir = (root / args.wilor_run).resolve()
    mp_dir = (root / args.mediapipe_run).resolve()
    parts_dir = (root / args.parts_run).resolve()
    output_dir = (root / args.output_root).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    source_manifest = RunManifest.model_validate_json((source_dir / "manifest.json").read_text())
    frame_count = len(source_manifest.observations)
    if frame_count not in (600, 1800):
        raise ValueError(f"WiLoR review supports 600 or 1800 rows, got {frame_count}")
    result = stabilize(
        _observations(source_dir / "observations.jsonl", frame_count=frame_count),
        _observations(mp_dir / "observations.jsonl", frame_count=frame_count),
        _observations(parts_dir / "observations.jsonl", frame_count=frame_count),
        frame_count=frame_count,
    )
    metrics = enrich_metrics(
        result,
        _observations(source_dir / "observations.jsonl", frame_count=frame_count),
        frame_count=frame_count,
    )
    result = result.__class__(result.observations, result.provenance, metrics)
    write_result(result, output_dir)
    video = source_dir / "input.mp4"
    if not video.is_file():
        raise FileNotFoundError(video)
    metadata = source_manifest.wilor_hands
    if metadata is None:
        raise ValueError("source run must have WiLoR metadata")
    settings = dict(metadata.runtime_settings)
    settings.update(
        {
            "postprocessor": "WiLoR-primary deterministic stabilization",
            "postprocessor_provenance_uri": str(
                (output_dir / "hand_provenance.json").relative_to(root)
            ),
            "postprocessor_metrics_uri": str(
                (output_dir / "stabilization_metrics.json").relative_to(root)
            ),
            "postprocessor_policy": (
                "WiLoR confidence>=0.55, deduplicate; evidence-defined part-workspace "
                "gate; real WiLoR detections with confidence in [0.35,0.55) accepted only "
                "while continuing a lane accepted within 5 frames, for at most 5 consecutive "
                "frames, tagged low_confidence_continuation; MediaPipe confidence>=0.85 "
                "fallback only for <=5-frame WiLoR gaps; One-Euro wrist/palm smoothing "
                "(min_cutoff=1.5,beta=0.007,d_cutoff=1.0). "
                f"Applied independently across exactly {frame_count} source-aligned rows."
            ),
        }
    )
    manifest = source_manifest.model_copy(
        update={
            "run_id": output_dir.name,
            "observations": result.observations,
            "wilor_hands": metadata.model_copy(
                update={
                    "runtime_settings": settings,
                    "observations_uri": str((output_dir / "observations.jsonl").relative_to(root)),
                    "rerun_artifact_uri": str(
                        (output_dir / "hands_stabilized.rrd").relative_to(root)
                    ),
                    "qa_artifact_uri": str(
                        (output_dir / "stabilization_metrics.json").relative_to(root)
                    ),
                }
            ),
        }
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    export_run(
        manifest,
        output_dir / "hands_stabilized.rrd",
        video_path=video,
        video_dimensions=(1280, 720),
        asset_reference=EncodedAssetInput(
            uri=metadata.proxy_fingerprint.uri,
            media_type="video/mp4",
            checksum_sha256=metadata.proxy_fingerprint.sha256,
        ),
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--wilor-run", type=Path, default=DEFAULT_WILOR)
    parser.add_argument("--mediapipe-run", type=Path, default=DEFAULT_MEDIAPIPE)
    parser.add_argument("--parts-run", type=Path, default=DEFAULT_PARTS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(run(args))


if __name__ == "__main__":
    main()
