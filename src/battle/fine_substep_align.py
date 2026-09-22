"""Run bounded fine-substep crop CLIP alignment against agent-authored labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from PIL import Image, ImageDraw

from .cli_common import add_output_root, add_repository_root
from .drop_dtw_align import (
    DEFAULT_CONFIG,
    DEFAULT_OPENCLIP_CACHE,
    DEFAULT_OPENCLIP_CHECKPOINT,
    DROP_DTW_REVISION,
    DROP_DTW_SOURCE,
    OPENCLIP_REPOSITORY,
    OPENCLIP_REVISION,
    WILOR_PYTHON,
    sha256_file,
)
from .fine_substep_contract import (
    ANALYSIS_FPS,
    FRAME_COUNT,
    PROVENANCE_TAG,
    load_contract,
    substep_for_frame,
)
from .fine_substep_pipeline import (
    DIMENSIONS,
    build_crop_manifest,
    contact_bonus,
    evaluate_against_agent_labels,
    fuse_scores,
    labels_to_intervals,
    load_observations,
    monotonic_substep_dp,
    phase_condition_scores,
    predicted_substep_at_frame,
    sample_frame_indices,
    temporal_delta_features,
    write_crop_manifest,
)
from .fs_common import relative_uri, run_timestamp
from .rerun_logging import init_and_save, time_series_view
from .schemas import (
    AdapterMetadata,
    ArtifactFingerprint,
    ChunkContinuityPolicy,
    FineSubstepRunMetadata,
    FullDurationCoverage,
    G2PreprocessingManifest,
    MethodState,
    MethodStatus,
    RunManifest,
    RuntimeMeasurements,
    TimeInterval,
)
from .video_driver import bounded_video as _bounded_video
from .video_driver import prepend_pythonpath, run_external_worker

DEFAULT_LABEL_CONTRACT = Path(
    "configs/fine_substeps/assembly101_focused_static_first_20s_agent_labels.json"
)
DEFAULT_WILOR_RUN = Path("runs/wilor-hands-static-20s-audited-source-state")
DEFAULT_MEDIAPIPE_RUN = Path("runs/mediapipe-hands-static-20s-fused-dedup-th035-20260916t0428z")
DEFAULT_PARTS_RUN = Path(
    "runs/muggledsam-sam3-four-part-static-focused-reassembly-static-c10379-20260916t023700z"
)
DEFAULT_BASELINE_DROP_DTW = Path("runs/drop-dtw-static-20s-pinned-openclip-rerun")
DEFAULT_SECONDS = 20.0
DEFAULT_SAMPLE_FPS = 3.0
DEFAULT_WEIGHTS = {"motion": 0.12, "contact": 0.08}
ITERATION_WEIGHTS = {"motion": 0.18, "contact": 0.14}


def _run_worker(
    *,
    run_directory: Path,
    video_path: Path,
    crop_manifest: Path,
    label_contract: Path,
    checkpoint_path: Path,
    cache_dir: Path,
) -> dict[str, object]:
    worker_path = Path(__file__).with_name("fine_substep_worker.py")
    src_root = Path(__file__).resolve().parents[1]
    argv = [
        str(worker_path.resolve()),
        "--run-directory",
        str(run_directory.resolve()),
        "--video",
        str(video_path.resolve()),
        "--crop-manifest",
        str(crop_manifest.resolve()),
        "--label-contract",
        str(label_contract.resolve()),
        "--openclip-checkpoint",
        str(checkpoint_path.resolve()),
        "--openclip-cache-dir",
        str(cache_dir.resolve()),
    ]
    worker_result = run_external_worker(
        WILOR_PYTHON,
        argv,
        run_directory=run_directory,
        env={"HF_HUB_OFFLINE": "1", "PYTHONPATH": prepend_pythonpath(src_root)},
    )
    if worker_result.get("state") != "succeeded":
        raise RuntimeError(worker_result.get("reason", "fine substep worker failed"))
    return json.loads((run_directory / "worker_scores.json").read_text(encoding="utf-8"))


def _alignment_from_drop_dtw(
    contract,
    sample_frames: list[int],
    drop_labels: list[int],
) -> list[int]:
    return [max(0, min(len(contract.substeps) - 1, int(label) - 1)) for label in drop_labels]


def _build_evaluation_bundle(
    contract,
    *,
    sample_frames: list[int],
    monotonic_indices: list[int],
    drop_indices: list[int],
    baseline_alignment: dict[str, object],
) -> dict[str, object]:
    monotonic_eval = evaluate_against_agent_labels(
        contract,
        sample_frames=sample_frames,
        substep_indices=monotonic_indices,
        alignment_name="monotonic_dp",
    )
    drop_eval = evaluate_against_agent_labels(
        contract,
        sample_frames=sample_frames,
        substep_indices=drop_indices,
        alignment_name="drop_dtw_11_prototypes",
    )
    baseline_rows = []
    for checkpoint in contract.checkpoints:
        action = next(
            (
                interval["action"]
                for interval in baseline_alignment["intervals"]
                if checkpoint.analysis_frame_index in interval["matched_analysis_frames"]
            ),
            None,
        )
        agent = substep_for_frame(contract, checkpoint.analysis_frame_index)
        baseline_rows.append(
            {
                "analysis_frame_index": checkpoint.analysis_frame_index,
                "coarse_gt_action": action,
                "agent_substep_id": agent.substep_id,
                "agent_label": agent.label,
            }
        )
    return {
        "provenance_tag": PROVENANCE_TAG,
        "agent_review_note": (
            "Diagnostics compare predicted substeps to agent-authored labels only; "
            "not Assembly101 GT accuracy."
        ),
        "monotonic_dp": monotonic_eval,
        "drop_dtw_11_prototypes": drop_eval,
        "baseline_coarse_drop_dtw": {
            "alignment_cost": baseline_alignment["alignment_cost"],
            "intervals": baseline_alignment["intervals"],
            "checkpoint_rows": baseline_rows,
        },
    }


def _contact_sheet(
    *,
    video_path: Path,
    contract,
    crop_samples: list,
    output_path: Path,
    predicted_monotonic: list[int],
    sample_frames: list[int],
) -> None:
    evidence_frames = sorted(
        {frame for substep in contract.substeps for frame in substep.evidence_frames}
        | {checkpoint.analysis_frame_index for checkpoint in contract.checkpoints}
        | set(contract.substeps[index].start_frame for index in range(1, len(contract.substeps)))
    )
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video for contact sheet: {video_path}")
    panels: list[Image.Image] = []
    width, height = DIMENSIONS
    for frame_index in evidence_frames:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        sample = next(
            (item for item in crop_samples if item.analysis_frame_index == frame_index),
            None,
        )
        if sample is not None:
            box = sample.workspace_box
            draw.rectangle((box.x0, box.y0, box.x1, box.y1), outline=(0, 255, 255), width=3)
        agent = substep_for_frame(contract, frame_index)
        predicted = predicted_substep_at_frame(
            contract, sample_frames, predicted_monotonic, frame_index
        )
        header = (
            f"f{frame_index} ({frame_index / ANALYSIS_FPS + 294.0:.3f}s) "
            f"agent={agent.substep_id} pred={predicted.substep_id} "
            f"hand={getattr(sample, 'hand_cue_source', 'n/a')}"
        )
        banner = Image.new("RGB", (width, 36), (20, 20, 20))
        banner_draw = ImageDraw.Draw(banner)
        banner_draw.text((8, 8), header, fill=(255, 255, 255))
        combined = Image.new("RGB", (width, height + 36))
        combined.paste(banner, (0, 0))
        combined.paste(image, (0, 36))
        panels.append(combined)
    capture.release()
    if not panels:
        raise RuntimeError("contact sheet has no panels")
    cols = 4
    rows = (len(panels) + cols - 1) // cols
    sheet = Image.new("RGB", (width * cols, (height + 36) * rows), (30, 30, 30))
    for index, panel in enumerate(panels):
        x = (index % cols) * width
        y = (index // cols) * (height + 36)
        sheet.paste(panel, (x, y))
    sheet.save(output_path)


def _review_guide(
    *,
    contract,
    evaluation: dict[str, object],
    run_directory: Path,
    weights: dict[str, float],
    iteration: str,
) -> str:
    mono = evaluation["monotonic_dp"]
    drop = evaluation["drop_dtw_11_prototypes"]
    lines = [
        "# Fine substep review guide",
        "",
        f"Provenance: `{PROVENANCE_TAG}` — agent labels are not Assembly101 GT.",
        "",
        f"Run directory: `{run_directory}`",
        f"Iteration: `{iteration}`",
        f"Weights: motion={weights['motion']}, contact={weights['contact']}",
        "",
        "## Claim boundaries",
        "",
        "- Agent substeps are visual-review proposals only.",
        "- Coarse GT anchors are weak protocol references, not eval targets.",
        "- Missing WiLoR frames fall back to MediaPipe explicitly; missing masks stay explicit.",
        "",
        "## Diagnostics",
        "",
        (
            f"- Monotonic DP recovered {mono['recovered_substep_count']}/"
            f"{mono['target_substep_count']} substeps; collapsed={mono['collapsed']}; "
            f"checkpoint match rate={mono['checkpoint_accuracy']:.2f}"
        ),
        (
            f"- Drop-DTW(11) recovered {drop['recovered_substep_count']}/"
            f"{drop['target_substep_count']} substeps; collapsed={drop['collapsed']}; "
            f"checkpoint match rate={drop['checkpoint_accuracy']:.2f}"
        ),
        (
            "- Mean abs boundary offset (monotonic): "
            f"{mono['mean_abs_boundary_offset_frames']:.1f} frames"
        ),
        "",
        "## Checkpoints",
        "",
    ]
    for row in mono["checkpoints"]:
        lines.append(
            f"- f{row['analysis_frame_index']}: agent {row['agent_substep_id']} vs "
            f"monotonic {row['predicted_substep_id']} ({'match' if row['match'] else 'miss'})"
        )
    lines.extend(
        [
            "",
            "## Open in Rerun",
            "",
            f"```bash\nrerun {run_directory / 'fine_substep.rrd'}\n```",
        ]
    )
    return "\n".join(lines) + "\n"


def _export_rrd(
    *,
    output_path: Path,
    video_path: Path,
    contract,
    sample_frames: list[int],
    clip_scores: np.ndarray,
    fused_scores: np.ndarray,
    monotonic_indices: list[int],
    drop_indices: list[int],
    evaluation: dict[str, object],
    crop_samples: list,
    run_id: str,
) -> None:
    init_and_save(f"battle-{run_id}", output_path)
    root = "fine_substep"
    rr.log(f"{root}/metadata/provenance", rr.TextLog(PROVENANCE_TAG))
    rr.log(
        f"{root}/metadata/evaluation",
        rr.TextDocument(json.dumps(evaluation, indent=2), media_type="application/json"),
        static=True,
    )
    rr.log(f"{root}/source/video", rr.AssetVideo(path=str(video_path)), static=True)
    substep_ids = [substep.substep_id for substep in contract.substeps]
    for sample_index, frame_index in enumerate(sample_frames):
        rr.set_time("analysis", sequence=frame_index)
        rr.set_time("analysis_seconds", duration=frame_index / ANALYSIS_FPS)
        sample = next(item for item in crop_samples if item.analysis_frame_index == frame_index)
        box = sample.workspace_box
        rr.log(
            f"{root}/crops/workspace",
            rr.Boxes2D(
                mins=[[box.x0, box.y0]],
                sizes=[[box.x1 - box.x0, box.y1 - box.y0]],
                labels=[sample.hand_cue_source],
            ),
        )
        for substep_index, substep_id in enumerate(substep_ids):
            rr.log(
                f"{root}/scores/clip/{substep_id}",
                rr.Scalars([float(clip_scores[sample_index, substep_index])]),
            )
            rr.log(
                f"{root}/scores/fused/{substep_id}",
                rr.Scalars([float(fused_scores[sample_index, substep_index])]),
            )
        rr.log(
            f"{root}/alignment/monotonic",
            rr.TextLog(substep_ids[monotonic_indices[sample_index]]),
        )
        rr.log(
            f"{root}/alignment/drop_dtw",
            rr.TextLog(substep_ids[drop_indices[sample_index]]),
        )
    for substep in contract.substeps:
        rr.log(
            f"{root}/agent_labels/{substep.substep_id}/start_seconds",
            rr.Scalars([substep.start_frame / ANALYSIS_FPS]),
            static=True,
        )
        rr.log(
            f"{root}/agent_labels/{substep.substep_id}/end_seconds_exclusive",
            rr.Scalars([substep.end_frame_exclusive / ANALYSIS_FPS]),
            static=True,
        )
    for anchor in contract.coarse_gt_anchors:
        rr.log(
            f"{root}/coarse_gt/{anchor.action.replace(' ', '_')}/start_seconds",
            rr.Scalars([anchor.proxy_start_frame / ANALYSIS_FPS]),
            static=True,
        )
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial2DView(origin=f"{root}/source", name="Video"),
                time_series_view(f"{root}/scores/fused", "Fused scores"),
            ),
            time_series_view(f"{root}/scores/clip", "CLIP scores"),
        )
    )


def run_iteration(
    *,
    repository_root: Path,
    run_directory: Path,
    contract,
    config: G2PreprocessingManifest,
    proxy_path: Path,
    video_path: Path,
    crop_samples: list,
    sample_frames: list[int],
    weights: dict[str, float],
    iteration_name: str,
    label_contract_path: Path,
    checkpoint_path: Path,
    cache_dir: Path,
    baseline_alignment: dict[str, object],
) -> dict[str, object]:
    crop_manifest_path = run_directory / f"crop_manifest_{iteration_name}.json"
    write_crop_manifest(crop_manifest_path, crop_samples)
    worker_scores = _run_worker(
        run_directory=run_directory,
        video_path=video_path,
        crop_manifest=crop_manifest_path,
        label_contract=label_contract_path,
        checkpoint_path=checkpoint_path,
        cache_dir=cache_dir,
    )
    clip_scores = np.asarray(worker_scores["clip_score_matrix"], dtype=np.float32)
    tool_cues = np.asarray(worker_scores["yellow_tool_cues"], dtype=np.float32)
    embeddings = np.asarray(worker_scores["frame_embeddings"], dtype=np.float32)
    motion = temporal_delta_features(embeddings)
    contact = np.asarray(
        [contact_bonus(sample.contact_proximity) for sample in crop_samples],
        dtype=np.float32,
    )
    fused_scores = fuse_scores(
        clip_scores,
        motion=motion,
        contact=contact,
        weights=weights,
    )
    fused_scores = phase_condition_scores(
        fused_scores, sample_frames=sample_frames, tool_cues=tool_cues
    )
    monotonic_indices, monotonic_cost = monotonic_substep_dp(fused_scores, len(contract.substeps))
    drop_indices = _alignment_from_drop_dtw(
        contract, sample_frames, worker_scores["drop_dtw_labels"]
    )
    monotonic_intervals = labels_to_intervals(contract, sample_frames, monotonic_indices)
    drop_intervals = labels_to_intervals(contract, sample_frames, drop_indices)
    evaluation = _build_evaluation_bundle(
        contract,
        sample_frames=sample_frames,
        monotonic_indices=monotonic_indices,
        drop_indices=drop_indices,
        baseline_alignment=baseline_alignment,
    )
    evaluation["iteration"] = iteration_name
    evaluation["weights"] = weights
    evaluation["sample_fps"] = len(sample_frames) / (FRAME_COUNT / ANALYSIS_FPS)
    evaluation["sample_count"] = len(sample_frames)
    evaluation["monotonic_cost"] = monotonic_cost
    evaluation["drop_dtw_cost"] = worker_scores["drop_dtw_cost"]
    evaluation["phase_conditioning"] = (
        "Frame <345 applies a soft screwdriver-language penalty; frame >=345 uses "
        "a measured crop yellow-pixel cue. It does not encode agent substep boundaries."
    )
    scores_payload = {
        "sample_frames": sample_frames,
        "clip_score_matrix": clip_scores.tolist(),
        "fused_score_matrix": fused_scores.tolist(),
        "motion_features": motion.tolist(),
        "contact_features": contact.tolist(),
        "yellow_tool_cues": tool_cues.tolist(),
        "monotonic_indices": monotonic_indices,
        "drop_dtw_indices": drop_indices,
        "monotonic_intervals": monotonic_intervals,
        "drop_dtw_intervals": drop_intervals,
    }
    (run_directory / f"scores_{iteration_name}.json").write_text(
        json.dumps(scores_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_directory / f"evaluation_{iteration_name}.json").write_text(
        json.dumps(evaluation, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "evaluation": evaluation,
        "clip_scores": clip_scores,
        "fused_scores": fused_scores,
        "monotonic_indices": monotonic_indices,
        "drop_indices": drop_indices,
        "worker_scores": worker_scores,
    }


def run(args: argparse.Namespace) -> Path:
    repository_root = args.repository_root.resolve()
    config_path = (repository_root / args.config).resolve()
    label_contract_path = (repository_root / args.label_contract).resolve()
    checkpoint_path = args.openclip_checkpoint.resolve()
    config = G2PreprocessingManifest.model_validate_json(config_path.read_text())
    contract = load_contract(label_contract_path)
    proxy = next(item for item in config.proxies if item.view_id == args.view)
    if proxy is None:
        raise ValueError(f"view {args.view} not found in config")
    proxy_path = (repository_root / proxy.proxy_uri).resolve()
    if sha256_file(proxy_path) != proxy.checksum_sha256:
        raise ValueError("proxy checksum mismatch")

    wilor = load_observations((repository_root / args.wilor_run / "observations.jsonl").resolve())
    mediapipe = load_observations(
        (repository_root / args.mediapipe_run / "observations.jsonl").resolve()
    )
    parts = load_observations((repository_root / args.parts_run / "observations.jsonl").resolve())
    baseline_alignment = json.loads(
        (repository_root / args.baseline_drop_dtw / "alignment.json").read_text(encoding="utf-8")
    )

    requested_frames = min(round(args.seconds * proxy.fps), FRAME_COUNT)
    run_id = args.run_id or (f"fine-substep-static-{args.seconds:g}s-{run_timestamp()}")
    run_directory = (repository_root / args.output_root / run_id).resolve()
    run_directory.mkdir(parents=True, exist_ok=args.reuse_run_directory)
    video_path = run_directory / "input.mp4"
    if not video_path.is_file():
        _bounded_video(proxy_path, video_path, requested_frames)

    sample_frames = sample_frame_indices(sample_fps=args.sample_fps, frame_count=requested_frames)
    crop_samples = build_crop_manifest(
        sample_frames,
        wilor=wilor,
        mediapipe=mediapipe,
        parts=parts,
    )

    first = run_iteration(
        repository_root=repository_root,
        run_directory=run_directory,
        contract=contract,
        config=config,
        proxy_path=proxy_path,
        video_path=video_path,
        crop_samples=crop_samples,
        sample_frames=sample_frames,
        weights=DEFAULT_WEIGHTS,
        iteration_name="pass1",
        label_contract_path=label_contract_path,
        checkpoint_path=checkpoint_path,
        cache_dir=args.openclip_cache_dir.resolve(),
        baseline_alignment=baseline_alignment,
    )
    selected = first
    iteration_name = "pass1"
    if first["evaluation"]["monotonic_dp"]["collapsed"]:
        second = run_iteration(
            repository_root=repository_root,
            run_directory=run_directory,
            contract=contract,
            config=config,
            proxy_path=proxy_path,
            video_path=video_path,
            crop_samples=crop_samples,
            sample_frames=sample_frames,
            weights=ITERATION_WEIGHTS,
            iteration_name="pass2",
            label_contract_path=label_contract_path,
            checkpoint_path=checkpoint_path,
            cache_dir=args.openclip_cache_dir.resolve(),
            baseline_alignment=baseline_alignment,
        )
        if (
            second["evaluation"]["monotonic_dp"]["checkpoint_accuracy"]
            >= first["evaluation"]["monotonic_dp"]["checkpoint_accuracy"]
        ):
            selected = second
            iteration_name = "pass2"

    evaluation = selected["evaluation"]
    evaluation_path = run_directory / "evaluation.json"
    evaluation_path.write_text(json.dumps(evaluation, indent=2) + "\n", encoding="utf-8")
    scores_path = run_directory / "scores.json"
    scores_path.write_text(
        (run_directory / f"scores_{iteration_name}.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    rrd_path = run_directory / "fine_substep.rrd"
    _export_rrd(
        output_path=rrd_path,
        video_path=video_path,
        contract=contract,
        sample_frames=sample_frames,
        clip_scores=selected["clip_scores"],
        fused_scores=selected["fused_scores"],
        monotonic_indices=selected["monotonic_indices"],
        drop_indices=selected["drop_indices"],
        evaluation=evaluation,
        crop_samples=crop_samples,
        run_id=run_id,
    )
    sheet_path = run_directory / "boundary_contact_sheet.png"
    _contact_sheet(
        video_path=video_path,
        contract=contract,
        crop_samples=crop_samples,
        output_path=sheet_path,
        predicted_monotonic=selected["monotonic_indices"],
        sample_frames=sample_frames,
    )
    guide_path = run_directory / "review_guide.md"
    guide_path.write_text(
        _review_guide(
            contract=contract,
            evaluation=evaluation,
            run_directory=run_directory,
            weights=evaluation["weights"],
            iteration=iteration_name,
        ),
        encoding="utf-8",
    )

    metadata = FineSubstepRunMetadata(
        requested_seconds=args.seconds,
        source_fingerprint=ArtifactFingerprint(
            uri=proxy.raw_source.raw_uri,
            sha256=proxy.raw_source.checksum_sha256,
            source="approved_config",
        ),
        proxy_fingerprint=ArtifactFingerprint(
            uri=proxy.proxy_uri,
            sha256=proxy.checksum_sha256,
            source="approved_config",
        ),
        config_fingerprint=ArtifactFingerprint(
            uri=relative_uri(config_path, repository_root),
            sha256=sha256_file(config_path),
            source="measured",
        ),
        label_contract_fingerprint=ArtifactFingerprint(
            uri=relative_uri(label_contract_path, repository_root),
            sha256=sha256_file(label_contract_path),
            source="measured",
        ),
        openclip_checkpoint_fingerprint=ArtifactFingerprint(
            uri=(f"hf://{OPENCLIP_REPOSITORY}@{OPENCLIP_REVISION}/open_clip_model.safetensors"),
            sha256=sha256_file(checkpoint_path),
            source="measured",
        ),
        adapter=AdapterMetadata(
            name="fine-substep-crop-clip",
            version="0.1.0",
            implementation_basis=(
                "WiLoR-primary hand/workspace crops with baseline SAM3 part boxes, "
                "multi-prompt OpenCLIP ViT-B-32 scoring, and monotonic DP + Drop-DTW arms"
            ),
            external_source_uri=str(DROP_DTW_SOURCE),
            external_revision=DROP_DTW_REVISION,
        ),
        runtime_settings={
            "analysis_fps": proxy.fps,
            "sample_fps": args.sample_fps,
            "selected_iteration": iteration_name,
            "motion_weight": evaluation["weights"]["motion"],
            "contact_weight": evaluation["weights"]["contact"],
            "wilor_run": str(args.wilor_run),
            "mediapipe_run": str(args.mediapipe_run),
            "parts_run": str(args.parts_run),
            "parts_reference": args.parts_reference,
            "hand_cue_policy": "wilor_primary_mediapipe_fallback",
            "recovered_substeps": evaluation["monotonic_dp"]["recovered_substep_count"],
            "checkpoint_accuracy": evaluation["monotonic_dp"]["checkpoint_accuracy"],
        },
        measurements=RuntimeMeasurements(
            elapsed_seconds=float(
                json.loads((run_directory / "worker_result.json").read_text())["elapsed_seconds"]
            ),
            known_unavailable_measures=("benchmark accuracy against Assembly101 GT",),
        ),
        scores_uri=relative_uri(scores_path, repository_root),
        evaluation_uri=relative_uri(evaluation_path, repository_root),
        rerun_artifact_uri=relative_uri(rrd_path, repository_root),
        contact_sheet_uri=relative_uri(sheet_path, repository_root),
        review_guide_uri=relative_uri(guide_path, repository_root),
    )
    manifest = RunManifest(
        run_id=run_id,
        clip=config.clip.model_copy(update={"source_duration_seconds": args.seconds}),
        coverage=FullDurationCoverage(
            source_duration_seconds=args.seconds,
            covered_intervals=(TimeInterval(start_seconds=0.0, end_seconds=args.seconds),),
        ),
        chunk_policy=ChunkContinuityPolicy(overlap_seconds=0.0, max_allowed_gap_seconds=0.0),
        method_statuses=(
            MethodStatus(
                method_name="fine-substep-crop-clip",
                stage="alignment",
                state=MethodState.SUCCEEDED,
                artifact_uri=relative_uri(scores_path, repository_root),
                measured_on=(
                    f"{args.view}; agent labels {PROVENANCE_TAG}; {args.seconds:g}s prefix"
                ),
            ),
            MethodStatus(
                method_name="rerun-fine-substep-export",
                stage="export",
                state=MethodState.SUCCEEDED,
                artifact_uri=relative_uri(rrd_path, repository_root),
                measured_on="agent-review diagnostics and score timelines only",
            ),
        ),
        fine_substep=metadata,
    )
    manifest_path = run_directory / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_repository_root(parser)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--label-contract", type=Path, default=DEFAULT_LABEL_CONTRACT)
    parser.add_argument("--view", default="static-c10379")
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    parser.add_argument("--sample-fps", type=float, default=DEFAULT_SAMPLE_FPS)
    parser.add_argument("--wilor-run", type=Path, default=DEFAULT_WILOR_RUN)
    parser.add_argument("--mediapipe-run", type=Path, default=DEFAULT_MEDIAPIPE_RUN)
    parser.add_argument("--parts-run", type=Path, default=DEFAULT_PARTS_RUN)
    parser.add_argument(
        "--parts-reference",
        default="baseline_sam3",
        choices=["baseline_sam3", "reviewed_seed_sam2_control"],
    )
    parser.add_argument("--baseline-drop-dtw", type=Path, default=DEFAULT_BASELINE_DROP_DTW)
    parser.add_argument("--openclip-checkpoint", type=Path, default=DEFAULT_OPENCLIP_CHECKPOINT)
    parser.add_argument("--openclip-cache-dir", type=Path, default=DEFAULT_OPENCLIP_CACHE)
    add_output_root(parser, Path("runs"))
    parser.add_argument("--run-id")
    parser.add_argument("--reuse-run-directory", action="store_true")
    args = parser.parse_args()
    if args.parts_reference == "reviewed_seed_sam2_control":
        args.parts_run = Path("runs/reviewed-seed-sam2-control-four-part-20s-20260916t1005z")
    print(run(args))


if __name__ == "__main__":
    main()
