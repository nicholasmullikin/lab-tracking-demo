"""Build the inference-free focused static four-part mask comparison RRD."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import rerun as rr
import rerun.blueprint as rrb

from .exporter import _rgba_mask_png
from .four_part_contract import (
    ANALYSIS_FPS,
    FRAME_COUNT,
    TARGETS,
    load_contract,
    relative_uri,
    sha256_file,
)
from .mask_ops import decode_mask_png
from .schemas import (
    ArtifactFingerprint,
    FourPartSegmentationComparisonIndex,
    FourPartSegmentationComparisonMethod,
    FrameObservations,
    RunManifest,
    TimeInterval,
    fingerprint,
)

DEFAULT_OUTPUT_ROOT = Path("runs/four-part-segmentation-comparison")
DEFAULT_BASELINE = Path(
    "runs/muggledsam-sam3-four-part-static-focused-reassembly-static-c10379-20260916t023700z"
)
COLORS = {
    "baseline_sam3": (255, 170, 40),
    "grounding_dino_sam2_open_vocabulary": (235, 70, 70),
    "reviewed_seed_sam2_control": (70, 130, 255),
    "samurai": (70, 210, 160),
    "dam4sam": (185, 100, 255),
}


def _tree_fingerprint(path: Path, root: Path) -> ArtifactFingerprint:
    digest = hashlib.sha256()
    for child in sorted(path.rglob("*")):
        if child.is_file():
            digest.update(relative_uri(child, root).encode())
            digest.update(sha256_file(child).encode())
    return ArtifactFingerprint(
        uri=relative_uri(path, root), sha256=digest.hexdigest(), source="measured"
    )


def _load_run(
    run_directory: Path, frame_count: int = FRAME_COUNT
) -> tuple[RunManifest, dict[int, FrameObservations]]:
    manifest = RunManifest.model_validate_json((run_directory / "manifest.json").read_text())
    observations = {
        observation.analysis_frame_index: observation
        for observation in manifest.observations[:frame_count]
    }
    if len(observations) != frame_count or set(observations) != set(range(frame_count)):
        raise ValueError(f"{run_directory.name} does not have exactly frames [0,{frame_count})")
    expected = manifest.clip.timing.source_seconds_for_frame
    for index, observation in observations.items():
        if abs(observation.source_seconds - expected("analysis", index)) > 1e-6:
            raise ValueError(f"{run_directory.name} source timestamp mismatch at frame {index}")
    return manifest, observations


def _coverage(observations: dict[int, FrameObservations]) -> dict[str, int]:
    return {
        target: sum(
            any(item.label == target and item.mask for item in frame.objects)
            for frame in observations.values()
        )
        for target in TARGETS
    }


def _initialization_summary(manifest: RunManifest) -> tuple[str, ...]:
    if manifest.four_part_segmentation is None:
        return ("SAM3 baseline: reviewed masks plus correction schedule; not accuracy evidence.",)
    return tuple(
        f"{item.target_id}: {item.source} {item.state}"
        for item in manifest.four_part_segmentation.target_initializations
    )


def _render(
    *,
    root: str,
    method_id: str,
    run_directory: Path,
    observation: FrameObservations,
    dimensions: tuple[int, int],
) -> None:
    method_root = f"{root}/methods/{method_id}/render"
    rr.log(method_root, rr.Clear(recursive=True))
    color = COLORS[method_id]
    width, height = dimensions
    if not observation.objects:
        return
    mins, sizes, labels = [], [], []
    for item in observation.objects:
        mins.append([item.box.x * width, item.box.y * height])
        sizes.append([item.box.width * width, item.box.height * height])
        labels.append(f"{item.label} ({item.confidence:.2f})")
        if item.mask is None:
            continue
        mask_path = (run_directory / item.mask.uri).resolve()
        if not mask_path.is_relative_to(run_directory) or not mask_path.is_file():
            raise FileNotFoundError(f"missing mask {item.mask.uri}")
        mask = decode_mask_png(mask_path)
        if mask.shape != (height, width):
            raise ValueError(f"mask dimension mismatch: {mask_path}")
        rr.log(
            f"{method_root}/masks/{item.label}",
            rr.EncodedImage(
                contents=_rgba_mask_png(mask, color),
                media_type="image/png",
                opacity=0.38,
                draw_order=1.0,
            ),
        )
    rr.log(
        f"{method_root}/boxes",
        rr.Boxes2D(mins=mins, sizes=sizes, labels=labels, colors=[color] * len(mins)),
    )


def _blueprint(root: str, methods: list[str], dimensions: tuple[int, int]) -> rrb.Blueprint:
    views = tuple(
        rrb.Spatial2DView(
            origin=root,
            name=method.replace("_", " "),
            contents=("$origin/source/video", f"$origin/methods/{method}/render/**"),
            visual_bounds=rrb.VisualBounds2D(
                x_range=[0, dimensions[0]], y_range=[0, dimensions[1]]
            ),
        )
        for method in methods
    )
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(*views[:3]),
            rrb.Horizontal(*views[3:]),
            rrb.TextDocumentView(
                origin=f"{root}/metadata/index", name="Comparison contract and coverage"
            ),
            row_shares=[3, 3, 1],
        ),
        rrb.TimePanel(timeline="analysis_time", fps=ANALYSIS_FPS),
        auto_layout=False,
        auto_views=False,
    )


def build(
    *,
    repository_root: Path,
    baseline_directory: Path,
    arm_directories: dict[str, Path],
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    repository_root = repository_root.resolve()
    contract = load_contract(
        repository_root, Path("configs/four_part_segmentation_comparison.json")
    )
    loaded: dict[str, tuple[Path, RunManifest, dict[int, FrameObservations]]] = {}
    for method_id, directory in {"baseline_sam3": baseline_directory, **arm_directories}.items():
        directory = directory.resolve()
        manifest, observations = _load_run(directory)
        loaded[method_id] = directory, manifest, observations
    baseline_manifest = loaded["baseline_sam3"][1]
    video = next(
        directory / "input.mp4"
        for method, (directory, _, _) in loaded.items()
        if method != "baseline_sam3"
    )
    if not video.is_file():
        raise FileNotFoundError("a completed new arm must provide its 600-frame input.mp4")
    dimensions = (1280, 720)
    methods = []
    for method_id, (directory, manifest, observations) in loaded.items():
        masks = directory / ("masks" if method_id == "baseline_sam3" else "native/masks")
        methods.append(
            FourPartSegmentationComparisonMethod(
                method_id=method_id,
                display_name=method_id.replace("_", " "),
                run_manifest=fingerprint(directory / "manifest.json", repository_root),
                observations=fingerprint(directory / "observations.jsonl", repository_root),
                native_masks=_tree_fingerprint(masks, repository_root),
                initialization_summary=_initialization_summary(manifest),
                target_coverage=_coverage(observations),
            )
        )
    output_root = (repository_root / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rrd = output_root / "four_part_segmentation_comparison.rrd"
    index_path = output_root / "four_part_segmentation_comparison_index.json"
    index = FourPartSegmentationComparisonIndex(
        manifest_kind="four_part_segmentation_comparison",
        comparison_id="four_part_segmentation_comparison",
        contract_fingerprint=fingerprint(contract.path, repository_root),
        source_video=ArtifactFingerprint(
            uri="data/raw/assembly101/nusar-2021_action_both_9033-c02a_9033_user_id_2021-02-04_140532/recordings/nusar-2021_action_both_9033-c02a_9033_user_id_2021-02-04_140532/C10379_rgb.mp4",
            sha256="450731ebbb50f46cf8279383e4737db6d76e967f23580d3555de1888b78a9db9",
            source="approved_config",
        ),
        bounded_video=fingerprint(video, repository_root),
        frame_count=600,
        analysis_fps=30,
        source_interval=TimeInterval(start_seconds=294.0, end_seconds=314.0),
        methods=tuple(methods),
        prior_unified_comparison_uri="runs/exploratory-first-20s-comparison/exploratory_first_20s_comparison.rrd",
    )
    index_path.write_text(index.model_dump_json(indent=2) + "\n")
    root = f"world/{baseline_manifest.clip.clip_id}/four_part_segmentation_comparison"
    rr.init(
        "battle-four-part-segmentation-comparison", recording_id="four-part-segmentation-comparison"
    )
    rr.save(rrd)
    rr.log(f"{root}/source/video_asset", rr.AssetVideo(path=video), static=True)
    rr.log(
        f"{root}/metadata/index",
        rr.TextDocument(index.model_dump_json(indent=2), media_type="application/json"),
        static=True,
    )
    for frame_index in range(FRAME_COUNT):
        rr.set_time("analysis_frame", sequence=frame_index)
        rr.set_time("analysis_time", duration=frame_index / ANALYSIS_FPS)
        rr.set_time("source_time", duration=294.0 + frame_index / ANALYSIS_FPS)
        rr.log(
            f"{root}/source/video",
            rr.VideoFrameReference(
                seconds=frame_index / ANALYSIS_FPS, video_reference=f"{root}/source/video_asset"
            ),
        )
        for method_id, (directory, _, observations) in loaded.items():
            _render(
                root=root,
                method_id=method_id,
                run_directory=directory,
                observation=observations[frame_index],
                dimensions=dimensions,
            )
    rr.send_blueprint(_blueprint(root, list(loaded), dimensions))
    rr.disconnect()
    return rrd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    for method in (
        "grounding_dino_sam2_open_vocabulary",
        "reviewed_seed_sam2_control",
        "samurai",
        "dam4sam",
    ):
        parser.add_argument(f"--{method.replace('_', '-')}", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    arms = {
        method: getattr(args, method)
        for method in (
            "grounding_dino_sam2_open_vocabulary",
            "reviewed_seed_sam2_control",
            "samurai",
            "dam4sam",
        )
        if getattr(args, method) is not None
    }
    if not arms:
        raise SystemExit("supply at least one completed segmentation arm")
    print(
        build(
            repository_root=args.repository_root,
            baseline_directory=args.baseline,
            arm_directories=arms,
            output_root=args.output_root,
        )
    )


if __name__ == "__main__":
    main()
