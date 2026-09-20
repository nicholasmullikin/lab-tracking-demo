"""Carry a finalized calibration and its correction schedule onto a proxy of another size.

A calibration (`MuggledSAMBoxCalibrationManifest`) pins the proxy it was made on: every mask
PNG has the proxy's pixel size, every box and click is in proxy pixels, and the schedule and
the worker verify those fingerprints before a run starts. To run the same seeds and
corrections on a re-encode of the *same source instants* at another resolution (the 1080p
proxy beside the 720p one), this module writes a derived calibration whose masks are resized
(nearest-neighbour by default), whose boxes and clicks are scaled and re-normalised, and
whose fingerprints point at the target G2 configuration, then a derived schedule that names
the derived masks. Nothing is re-decoded and no human reviewed the resized masks: the
provenance sidecar says so, and the derived manifest records the calibration it came from.

The target proxy must be the same recording, the same interval start, the same frame rate
and frame count; only the pixel grid may differ. Later-frame corrections stay valid because
frame indices name the same instants on the same clock.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from pydantic import Field

from .digest_cache import sha256_file
from .schemas import (
    ArtifactFingerprint,
    G2PreprocessingManifest,
    MuggledSAMBoxCalibrationManifest,
    MuggledSAMCalibrationCandidate,
    MuggledSAMImageCandidate,
    MuggledSAMImageDecoderResult,
    MuggledSAMManualSeedTargetConfig,
    MuggledSAMMultiKeyframeCorrection,
    MuggledSAMMultiKeyframeCorrectionPolicy,
    MuggledSAMMultiKeyframeCorrectionSchedule,
    NormalizedBox,
    NormalizedPoint,
    PixelBox,
    PixelPoint,
    VersionedModel,
    VideoDimensions,
)

MANIFEST_NAME = "calibration_manifest.json"
SCHEDULE_NAME = "multi_keyframe_correction_schedule.json"
PROVENANCE_NAME = "rescale_provenance.json"
MASK_DIRECTORY = Path("results") / "masks"

Interpolation = Literal["nearest"]
INTERPOLATION_FLAGS: dict[str, int] = {"nearest": cv2.INTER_NEAREST}

CLAIM_BOUNDARIES: tuple[str, ...] = (
    "The rescaled masks are resampled copies of decoder masks a human (or, for later-frame "
    "corrections marked agent, an agent) chose on the source proxy; nobody reviewed them at "
    "the target resolution and their boundaries carry the source grid's block structure.",
    "Boxes and clicks are scaled and rounded to the target pixel grid; they are provenance for "
    "the masks, not prompts the run re-decodes.",
    "The target proxy shares the source recording, interval start, frame rate and frame count "
    "with the source proxy, so frame indices name the same instants; only the pixel grid "
    "differs.",
    "Nothing here is ground truth or an accuracy claim; CC BY-NC 4.0 covers the frames and "
    "every mask derived from them.",
)


class RescaledMaskRecord(VersionedModel):
    """One mask PNG before and after resampling."""

    candidate_id: str
    candidate_index: int = Field(ge=0)
    source_uri: str
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_area_px: int = Field(ge=0)
    target_uri: str
    target_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    target_area_px: int = Field(ge=0)


class CalibrationRescaleProvenance(VersionedModel):
    """Sidecar written beside the derived calibration and copied into runs that use it."""

    manifest_kind: Literal["muggledsam_calibration_rescale"]
    source_calibration_manifest: ArtifactFingerprint
    source_schedule: ArtifactFingerprint | None
    source_proxy: ArtifactFingerprint
    source_dimensions: VideoDimensions
    target_g2_config: ArtifactFingerprint
    target_proxy: ArtifactFingerprint
    target_dimensions: VideoDimensions
    scale_x: float = Field(gt=0)
    scale_y: float = Field(gt=0)
    interpolation: Interpolation
    derived_calibration_manifest: ArtifactFingerprint
    derived_schedule: ArtifactFingerprint | None
    masks: tuple[RescaledMaskRecord, ...]
    pending_boxes_dropped: int = Field(ge=0)
    runtime_seconds: float = Field(ge=0)
    claim_boundaries: tuple[str, ...] = CLAIM_BOUNDARIES


def _fingerprint(
    path: Path, repository_root: Path, *, source: Literal["approved_config", "measured"]
) -> ArtifactFingerprint:
    return ArtifactFingerprint(
        uri=path.resolve().relative_to(repository_root).as_posix(),
        sha256=sha256_file(path),
        source=source,
    )


def _scaled_box(box: PixelBox, scale_x: float, scale_y: float, dims: VideoDimensions) -> PixelBox:
    x1 = min(int(round(box.x1 * scale_x)), dims.width - 1)
    y1 = min(int(round(box.y1 * scale_y)), dims.height - 1)
    x2 = max(min(int(round(box.x2 * scale_x)), dims.width), x1 + 1)
    y2 = max(min(int(round(box.y2 * scale_y)), dims.height), y1 + 1)
    return PixelBox(x1=x1, y1=y1, x2=x2, y2=y2)


def _normalized_box(box: PixelBox, dims: VideoDimensions) -> NormalizedBox:
    return NormalizedBox(
        x=box.x1 / dims.width,
        y=box.y1 / dims.height,
        width=(box.x2 - box.x1) / dims.width,
        height=(box.y2 - box.y1) / dims.height,
    )


def _scaled_points(
    points: tuple[PixelPoint, ...], scale_x: float, scale_y: float, dims: VideoDimensions
) -> tuple[tuple[PixelPoint, ...], tuple[NormalizedPoint, ...]]:
    pixels = tuple(
        PixelPoint(
            x=min(int(round(point.x * scale_x)), dims.width),
            y=min(int(round(point.y * scale_y)), dims.height),
        )
        for point in points
    )
    normalized = tuple(
        NormalizedPoint(x=point.x / dims.width, y=point.y / dims.height) for point in pixels
    )
    return pixels, normalized


def _relative_to_output(path: Path, output_dir: Path) -> str:
    return Path(os.path.relpath(path.resolve(), output_dir.resolve())).as_posix()


def rescale_calibration(
    *,
    repository_root: Path,
    calibration_manifest: Path,
    schedule: Path | None,
    target_g2_config: Path,
    target_manual_seed_config: Path | None,
    target_correction_policy: Path | None,
    output_dir: Path,
    interpolation: Interpolation = "nearest",
) -> CalibrationRescaleProvenance:
    """Write the derived calibration (and schedule) under `output_dir`; return its provenance."""
    started = time.monotonic()
    repository_root = repository_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} exists and is not empty")
    source_manifest_path = calibration_manifest.resolve()
    source = MuggledSAMBoxCalibrationManifest.model_validate_json(
        source_manifest_path.read_text(encoding="utf-8")
    )
    target_config_path = target_g2_config.resolve()
    target_config = G2PreprocessingManifest.model_validate_json(
        target_config_path.read_text(encoding="utf-8")
    )
    target_proxy = next(
        (proxy for proxy in target_config.proxies if proxy.view_id == source.view_id), None
    )
    if target_proxy is None:
        raise ValueError(f"target G2 config has no proxy for view {source.view_id}")
    if target_proxy.raw_source.checksum_sha256 != source.source.sha256:
        raise ValueError("target proxy must come from the same raw recording as the calibration")
    if target_config.source_interval.start_seconds != source.source_offset_seconds:
        raise ValueError("target proxy must start at the calibration's source offset")
    if target_proxy.fps != source.proxy_fps or target_proxy.frame_count != source.proxy_frame_count:
        raise ValueError("target proxy must keep the calibration's frame rate and frame count")
    target_dims = target_proxy.dimensions
    source_dims = source.proxy_dimensions
    if (target_dims.width, target_dims.height) == (source_dims.width, source_dims.height):
        raise ValueError("target proxy has the calibration's own dimensions; nothing to rescale")
    scale_x = target_dims.width / source_dims.width
    scale_y = target_dims.height / source_dims.height

    schedule_model: MuggledSAMMultiKeyframeCorrectionSchedule | None = None
    if schedule is not None:
        schedule_model = MuggledSAMMultiKeyframeCorrectionSchedule.model_validate_json(
            schedule.resolve().read_text(encoding="utf-8")
        )
        if schedule_model.calibration_manifest_fingerprint.sha256 != sha256_file(
            source_manifest_path
        ):
            raise ValueError("schedule does not fingerprint the given calibration manifest")
        if target_manual_seed_config is None or target_correction_policy is None:
            raise ValueError(
                "a schedule needs --target-manual-seed-config and --target-correction-policy"
            )
        seed_config_path = target_manual_seed_config.resolve()
        seed_config = MuggledSAMManualSeedTargetConfig.model_validate_json(
            seed_config_path.read_text(encoding="utf-8")
        )
        if (repository_root / seed_config.base_g2_config).resolve() != target_config_path:
            raise ValueError("target manual-seed config must name the target G2 config")
        policy_path = target_correction_policy.resolve()
        policy = MuggledSAMMultiKeyframeCorrectionPolicy.model_validate_json(
            policy_path.read_text(encoding="utf-8")
        )
        if policy.manual_seed_target_config_fingerprint.sha256 != sha256_file(seed_config_path):
            raise ValueError("target correction policy must fingerprint the target seed config")
        source_policy = MuggledSAMMultiKeyframeCorrectionPolicy.model_validate_json(
            (repository_root / schedule_model.correction_policy_fingerprint.uri).read_text(
                encoding="utf-8"
            )
        )
        if (
            policy.targets != source_policy.targets
            or policy.view_id != source_policy.view_id
            or policy.correction_memory_semantics != source_policy.correction_memory_semantics
            or policy.maximum_later_correction_keyframes_per_target
            != source_policy.maximum_later_correction_keyframes_per_target
        ):
            raise ValueError("target correction policy must match the source policy's rules")

    mask_dir = output_dir / MASK_DIRECTORY
    mask_dir.mkdir(parents=True, exist_ok=True)
    flag = INTERPOLATION_FLAGS[interpolation]
    records: list[RescaledMaskRecord] = []
    new_mask_sha_by_uri: dict[str, tuple[str, str]] = {}
    candidates: list[MuggledSAMCalibrationCandidate] = []
    for candidate in source.candidates:
        image_candidates: list[MuggledSAMImageCandidate] = []
        for image_candidate in candidate.decoder_result.candidates:
            source_mask_path = (source_manifest_path.parent / image_candidate.mask_uri).resolve()
            mask = cv2.imread(str(source_mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != (source_dims.height, source_dims.width):
                raise ValueError(f"calibration mask is missing or mis-sized: {source_mask_path}")
            resized = cv2.resize(mask, (target_dims.width, target_dims.height), interpolation=flag)
            target_mask_path = mask_dir / source_mask_path.name
            if target_mask_path.exists():
                raise ValueError(f"duplicate mask file name: {source_mask_path.name}")
            if not cv2.imwrite(str(target_mask_path), resized):
                raise OSError(f"could not write {target_mask_path}")
            target_uri = target_mask_path.relative_to(repository_root).as_posix()
            target_sha = sha256_file(target_mask_path)
            source_uri = source_mask_path.relative_to(repository_root).as_posix()
            new_mask_sha_by_uri[source_uri] = (target_uri, target_sha)
            records.append(
                RescaledMaskRecord(
                    candidate_id=candidate.candidate_id,
                    candidate_index=image_candidate.candidate_index,
                    source_uri=source_uri,
                    source_sha256=sha256_file(source_mask_path),
                    source_area_px=int(np.count_nonzero(mask)),
                    target_uri=target_uri,
                    target_sha256=target_sha,
                    target_area_px=int(np.count_nonzero(resized)),
                )
            )
            review_uri = (
                _relative_to_output(
                    source_manifest_path.parent / image_candidate.review_uri, output_dir
                )
                if image_candidate.review_uri
                else None
            )
            image_candidates.append(
                image_candidate.model_copy(
                    update={
                        "mask_uri": target_mask_path.relative_to(output_dir).as_posix(),
                        "review_uri": review_uri,
                    }
                )
            )
        decoder_result = candidate.decoder_result.model_copy(
            update={
                "candidates": tuple(image_candidates),
                "overlay_uri": _relative_to_output(
                    source_manifest_path.parent / candidate.decoder_result.overlay_uri, output_dir
                ),
            }
        )
        pixel_box = _scaled_box(candidate.pixel_box, scale_x, scale_y, target_dims)
        fg_pixels, fg_normalized = _scaled_points(
            candidate.pixel_fg_points, scale_x, scale_y, target_dims
        )
        bg_pixels, bg_normalized = _scaled_points(
            candidate.pixel_bg_points, scale_x, scale_y, target_dims
        )
        candidates.append(
            MuggledSAMCalibrationCandidate.model_validate(
                {
                    **candidate.model_dump(mode="json"),
                    "pixel_box": pixel_box.model_dump(mode="json"),
                    "normalized_box": _normalized_box(pixel_box, target_dims).model_dump(
                        mode="json"
                    ),
                    "pixel_fg_points": [p.model_dump(mode="json") for p in fg_pixels],
                    "pixel_bg_points": [p.model_dump(mode="json") for p in bg_pixels],
                    "normalized_fg_points": [p.model_dump(mode="json") for p in fg_normalized],
                    "normalized_bg_points": [p.model_dump(mode="json") for p in bg_normalized],
                    "decoder_result": MuggledSAMImageDecoderResult.model_validate(
                        decoder_result.model_dump(mode="json")
                    ).model_dump(mode="json"),
                }
            )
        )

    derived_manifest_path = output_dir / MANIFEST_NAME
    derived_schedule_path = output_dir / SCHEDULE_NAME
    derived = MuggledSAMBoxCalibrationManifest.model_validate(
        {
            **source.model_dump(mode="json"),
            "calibration_id": (
                f"{source.calibration_id}-rescaled-{target_dims.width}x{target_dims.height}"
            ),
            "base_g2_config": target_config_path.relative_to(repository_root).as_posix(),
            "base_g2_config_sha256": sha256_file(target_config_path),
            "proxy": ArtifactFingerprint(
                uri=target_proxy.proxy_uri,
                sha256=target_proxy.checksum_sha256,
                source="approved_config",
            ).model_dump(mode="json"),
            "proxy_dimensions": target_dims.model_dump(mode="json"),
            "candidates": [c.model_dump(mode="json") for c in candidates],
            # Pending boxes are unfinished prompts; a derived calibration is never edited.
            "workspace": {
                "active_proxy_timestamp_seconds": (source.workspace.active_proxy_timestamp_seconds),
                "selected_box_id": None,
                "custom_label": "",
                "pending_boxes": [],
            },
            "result_directory_uri": (
                (output_dir / "results").relative_to(repository_root).as_posix()
            ),
            "final_proposal_uri": None,
            "final_correction_schedule_uri": (
                derived_schedule_path.relative_to(repository_root).as_posix()
                if schedule_model is not None
                else None
            ),
            "derived_from_calibration": _fingerprint(
                source_manifest_path, repository_root, source="measured"
            ).model_dump(mode="json"),
        }
    )
    derived_manifest_path.write_text(derived.model_dump_json(indent=2) + "\n", encoding="utf-8")
    derived_manifest_fingerprint = _fingerprint(
        derived_manifest_path, repository_root, source="measured"
    )

    derived_schedule_fingerprint: ArtifactFingerprint | None = None
    if schedule_model is not None:
        corrections: list[MuggledSAMMultiKeyframeCorrection] = []
        for correction in schedule_model.corrections:
            mapped = new_mask_sha_by_uri.get(correction.calibration_mask_fingerprint.uri)
            if mapped is None:
                raise ValueError(
                    "schedule mask is not part of the calibration: "
                    f"{correction.calibration_mask_fingerprint.uri}"
                )
            corrections.append(
                correction.model_copy(
                    update={
                        "calibration_mask_fingerprint": ArtifactFingerprint(
                            uri=mapped[0], sha256=mapped[1], source="measured"
                        )
                    }
                )
            )
        derived_schedule = MuggledSAMMultiKeyframeCorrectionSchedule.model_validate(
            {
                **schedule_model.model_dump(mode="json"),
                "calibration_manifest_fingerprint": derived_manifest_fingerprint.model_dump(
                    mode="json"
                ),
                "correction_policy_fingerprint": _fingerprint(
                    policy_path, repository_root, source="measured"
                ).model_dump(mode="json"),
                "manual_seed_target_config_fingerprint": _fingerprint(
                    seed_config_path, repository_root, source="measured"
                ).model_dump(mode="json"),
                "corrections": [c.model_dump(mode="json") for c in corrections],
            }
        )
        derived_schedule_path.write_text(
            derived_schedule.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        derived_schedule_fingerprint = _fingerprint(
            derived_schedule_path, repository_root, source="measured"
        )

    provenance = CalibrationRescaleProvenance(
        manifest_kind="muggledsam_calibration_rescale",
        source_calibration_manifest=_fingerprint(
            source_manifest_path, repository_root, source="measured"
        ),
        source_schedule=(
            _fingerprint(schedule.resolve(), repository_root, source="measured")
            if schedule is not None
            else None
        ),
        source_proxy=source.proxy,
        source_dimensions=source_dims,
        target_g2_config=_fingerprint(target_config_path, repository_root, source="measured"),
        target_proxy=ArtifactFingerprint(
            uri=target_proxy.proxy_uri,
            sha256=target_proxy.checksum_sha256,
            source="approved_config",
        ),
        target_dimensions=target_dims,
        scale_x=scale_x,
        scale_y=scale_y,
        interpolation=interpolation,
        derived_calibration_manifest=derived_manifest_fingerprint,
        derived_schedule=derived_schedule_fingerprint,
        masks=tuple(records),
        pending_boxes_dropped=len(source.workspace.pending_boxes),
        runtime_seconds=time.monotonic() - started,
    )
    (output_dir / PROVENANCE_NAME).write_text(
        provenance.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return provenance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, default=None)
    parser.add_argument("--target-g2-config", type=Path, required=True)
    parser.add_argument("--target-manual-seed-config", type=Path, default=None)
    parser.add_argument("--target-correction-policy", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interpolation", choices=tuple(INTERPOLATION_FLAGS), default="nearest")
    args = parser.parse_args(argv)
    provenance = rescale_calibration(
        repository_root=args.repository_root,
        calibration_manifest=args.calibration_manifest,
        schedule=args.schedule,
        target_g2_config=args.target_g2_config,
        target_manual_seed_config=args.target_manual_seed_config,
        target_correction_policy=args.target_correction_policy,
        output_dir=args.output_dir,
        interpolation=args.interpolation,
    )
    print(
        f"{len(provenance.masks)} masks rescaled x{provenance.scale_x:g}/{provenance.scale_y:g} "
        f"({provenance.interpolation}) -> {provenance.derived_calibration_manifest.uri}"
        + (
            f"; schedule -> {provenance.derived_schedule.uri}"
            if provenance.derived_schedule
            else ""
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
