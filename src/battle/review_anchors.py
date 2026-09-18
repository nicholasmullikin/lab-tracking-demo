"""Human review anchors for the focused C10379 first minute.

An anchor is one (frame, part) cell a human labels once in the calibration workspace:
either an accepted SAM3 image-decoder mask the human chose, or an explicit "hidden" mark
saying the part has no visible surface there. The anchors are review evidence for
scoring tracker arms against each other; they are not a dataset, not ground truth, and
carry no accuracy claim. This module owns four things:

- the typed anchor configuration (`configs/qa/first_minute_review_anchors.json`);
- preparing a calibration workspace whose calibration frames are exactly the anchors;
- exporting the workspace's accepted masks and hidden marks into a fingerprinted anchor
  mask set plus a human-record skeleton under `docs/qa/`;
- `battle-anchor-iou`, which scores any run directory against the accepted anchors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from pydantic import Field, model_validator

from . import mask_cache
from .muggled_calibration import TOOL_VERSION, _write_manifest, build_manifest
from .muggled_smoke import load_manual_seed_target_config, relative_uri, sha256_file
from .schemas import (
    ArtifactFingerprint,
    MuggledSAMBoxCalibrationManifest,
    VersionedModel,
    VideoDimensions,
)

ANCHOR_KIND = "human_review_anchor"
CLAIM_BOUNDARY = (
    "human_review_anchor masks are review evidence for scoring tracker arms against each "
    "other on a handful of frames; they are not a dataset, not ground truth, and support no "
    "accuracy claim. A human chose one SAM3 image-decoder mask per visible part, or marked "
    "the part hidden; the mask boundary is the decoder's, the choice is the human's."
)
LICENSE = (
    "CC BY-NC 4.0 (Assembly101 source frames); anchor masks and marks are derived from those "
    "frames and inherit the same non-commercial terms"
)
DEFAULT_CONFIG = Path("configs/qa/first_minute_review_anchors.json")
DEFAULT_WORKSPACE = Path("runs/human-review-anchors-first-minute")
DEFAULT_RECORD = Path("docs/qa/first-minute-review-anchors.human-record.json")
SESSION_NAME = "anchor_session.json"
MASK_SET_NAME = "anchors/anchor_masks.json"
MASK_DIRECTORY = "anchors/masks"

Visibility = Literal["visible", "hidden_prompt"]
AnchorState = Literal["labeled", "hidden", "unlabeled"]
CellOutcome = Literal[
    "scored",
    "run_mask_missing",
    "hidden_correct",
    "hidden_false_positive",
    "unlabeled_skipped",
]


# --------------------------------------------------------------------------- configuration


class ReviewAnchorProvenance(VersionedModel):
    """Who labelled and when; null until the human fills it in after the session."""

    author: str | None = None
    reviewed_at: datetime | None = None
    tool: str = "battle-muggled-calibration-web"
    notes: str | None = None


class ReviewAnchorFrame(VersionedModel):
    """One anchor frame on the focused clip's analysis clock."""

    analysis_frame_index: int = Field(ge=0)
    proxy_seconds: float = Field(ge=0)
    source_seconds: float = Field(ge=0)
    # `hidden_prompt` means the human must answer explicitly (draw a mask or mark hidden);
    # `visible` means one accepted mask is expected.
    expected_visible: dict[str, Visibility]
    note: str | None = None


class ReviewAnchorConfig(VersionedModel):
    """Typed anchor list for one clip view, with the claim boundary spelled out."""

    manifest_kind: Literal["human_review_anchor_config"]
    config_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    anchor_kind: Literal["human_review_anchor"] = ANCHOR_KIND
    clip_config: str = Field(min_length=1)
    view_id: str = Field(min_length=1)
    manual_seed_target_config: str = Field(min_length=1)
    analysis_fps: int = Field(gt=0)
    source_offset_seconds: float = Field(ge=0)
    targets: tuple[str, ...] = Field(min_length=1)
    frames: tuple[ReviewAnchorFrame, ...] = Field(min_length=1)
    # Human-reported failure windows (analysis frames, half-open) the per-window means use.
    windows: dict[str, tuple[int, int]]
    hidden_prompt_interval: tuple[int, int] | None = None
    claim_boundary: str = Field(min_length=1)
    license: str = Field(min_length=1)
    provenance: ReviewAnchorProvenance = Field(default_factory=ReviewAnchorProvenance)

    @model_validator(mode="after")
    def require_consistent_frames(self) -> ReviewAnchorConfig:
        indices = [frame.analysis_frame_index for frame in self.frames]
        if indices != sorted(set(indices)):
            raise ValueError("anchor frames must be increasing and unique")
        for frame in self.frames:
            expected = frame.analysis_frame_index / self.analysis_fps
            if abs(frame.proxy_seconds - expected) > 1e-9:
                raise ValueError(f"frame {frame.analysis_frame_index}: proxy_seconds mismatch")
            if abs(frame.source_seconds - (self.source_offset_seconds + expected)) > 1e-9:
                raise ValueError(f"frame {frame.analysis_frame_index}: source_seconds mismatch")
            if set(frame.expected_visible) != set(self.targets):
                raise ValueError(
                    f"frame {frame.analysis_frame_index}: expected_visible must name every target"
                )
            if self.hidden_prompt_interval is not None:
                low, high = self.hidden_prompt_interval
                inside = low <= frame.analysis_frame_index < high
                prompted = any(v == "hidden_prompt" for v in frame.expected_visible.values())
                if inside != prompted:
                    raise ValueError(
                        f"frame {frame.analysis_frame_index}: hidden_prompt must be used exactly "
                        "on frames inside hidden_prompt_interval"
                    )
        for name, (low, high) in self.windows.items():
            if low >= high:
                raise ValueError(f"window {name} must be a non-empty half-open interval")
        return self

    def frame(self, analysis_frame_index: int) -> ReviewAnchorFrame:
        for frame in self.frames:
            if frame.analysis_frame_index == analysis_frame_index:
                return frame
        raise KeyError(analysis_frame_index)

    def frames_in_window(self, window: tuple[int, int]) -> tuple[int, ...]:
        return tuple(
            frame.analysis_frame_index
            for frame in self.frames
            if window[0] <= frame.analysis_frame_index < window[1]
        )


def load_config(path: Path) -> ReviewAnchorConfig:
    return ReviewAnchorConfig.model_validate_json(path.read_text(encoding="utf-8"))


def build_first_minute_config() -> ReviewAnchorConfig:
    """The Sep 18 anchor list on the focused C10379 clock (analysis frames, 30 fps)."""
    fps, offset = 30, 294.0
    targets = ("chassis", "interior", "rear_body", "cabin")
    hidden_interval = (1024, 1172)
    frames = []
    for index in (300, 370, 400, 600, 650, 700, 900, 1050, 1100, 1150, 1200, 1500, 1700):
        inside = hidden_interval[0] <= index < hidden_interval[1]
        expected: dict[str, Visibility] = {
            target: ("hidden_prompt" if inside and target == "interior" else "visible")
            for target in targets
        }
        frames.append(
            ReviewAnchorFrame(
                analysis_frame_index=index,
                proxy_seconds=index / fps,
                source_seconds=offset + index / fps,
                expected_visible=expected,
                note=(
                    "interior may be occluded here; choose hidden or draw the visible surface"
                    if inside
                    else None
                ),
            )
        )
    return ReviewAnchorConfig(
        manifest_kind="human_review_anchor_config",
        config_id="first-minute-review-anchors-c10379",
        clip_config="configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_g2.json",
        view_id="static-c10379",
        manual_seed_target_config=(
            "configs/muggledsam_static_four_part_reassembly_focused_manual_seed.json"
        ),
        analysis_fps=fps,
        source_offset_seconds=offset,
        targets=targets,
        frames=tuple(frames),
        windows={"279-408": (279, 408), "573-722": (573, 722), "1020-1172": (1020, 1172)},
        hidden_prompt_interval=hidden_interval,
        claim_boundary=CLAIM_BOUNDARY,
        license=LICENSE,
    )


# ------------------------------------------------------------------------ session / export


class ReviewAnchorSession(VersionedModel):
    """Ties a prepared calibration workspace to the anchor configuration it serves."""

    manifest_kind: Literal["human_review_anchor_session"]
    anchor_kind: Literal["human_review_anchor"] = ANCHOR_KIND
    config: ArtifactFingerprint
    calibration_manifest_uri: str = Field(min_length=1)
    manual_seed_target_config: ArtifactFingerprint
    prepared_at: datetime
    frame_zero_prepopulated: bool = False
    frame_zero_provenance: str | None = None
    claim_boundary: str = Field(min_length=1)


class ReviewAnchorMask(VersionedModel):
    """One exported anchor cell: a fingerprinted mask, a hidden mark, or still unlabeled."""

    analysis_frame_index: int = Field(ge=0)
    target: str = Field(min_length=1)
    expected_visible: Visibility
    state: AnchorState
    mask_uri: str | None = None
    mask_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    area_pixels: int | None = Field(default=None, ge=0)
    source_candidate_id: str | None = None
    source_candidate_index: int | None = Field(default=None, ge=0)
    source_mask_uri: str | None = None
    selected_by: Literal["human"] | None = None

    @model_validator(mode="after")
    def require_state_fields(self) -> ReviewAnchorMask:
        has_mask = self.mask_uri is not None and self.mask_sha256 is not None
        if self.state == "labeled" and not (has_mask and self.area_pixels is not None):
            raise ValueError("a labeled anchor needs a mask uri, sha256 and area")
        if self.state != "labeled" and (has_mask or self.area_pixels is not None):
            raise ValueError("only labeled anchors carry a mask")
        return self


class ReviewAnchorMaskSet(VersionedModel):
    manifest_kind: Literal["human_review_anchor_masks"]
    anchor_kind: Literal["human_review_anchor"] = ANCHOR_KIND
    config: ArtifactFingerprint
    calibration_manifest: ArtifactFingerprint
    view_id: str = Field(min_length=1)
    targets: tuple[str, ...] = Field(min_length=1)
    mask_dimensions: VideoDimensions
    windows: dict[str, tuple[int, int]]
    anchors: tuple[ReviewAnchorMask, ...] = Field(min_length=1)
    counts: dict[AnchorState, int]
    exported_at: datetime
    claim_boundary: str = Field(min_length=1)
    license: str = Field(min_length=1)
    provenance: ReviewAnchorProvenance

    def anchor(self, frame: int, target: str) -> ReviewAnchorMask:
        for item in self.anchors:
            if item.analysis_frame_index == frame and item.target == target:
                return item
        raise KeyError((frame, target))

    @property
    def frames(self) -> tuple[int, ...]:
        return tuple(sorted({item.analysis_frame_index for item in self.anchors}))


class ReviewAnchorRecordEntry(VersionedModel):
    analysis_frame_index: int = Field(ge=0)
    target: str
    state: AnchorState
    mask_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


class ReviewAnchorHumanRecord(VersionedModel):
    """Committed, human-signed record of which anchors exist; the masks stay in runs/."""

    manifest_kind: Literal["human_review_anchor_record"]
    anchor_kind: Literal["human_review_anchor"] = ANCHOR_KIND
    author: str | None = None
    reviewed_at: datetime | None = None
    config: ArtifactFingerprint
    mask_set: ArtifactFingerprint
    calibration_manifest: ArtifactFingerprint
    counts: dict[AnchorState, int]
    anchors: tuple[ReviewAnchorRecordEntry, ...]
    claim_boundary: str
    license: str
    notes: str | None = None


def _fingerprint(path: Path, repository_root: Path) -> ArtifactFingerprint:
    return ArtifactFingerprint(
        uri=relative_uri(path.resolve(), repository_root),
        sha256=sha256_file(path),
        source="measured",
    )


def prepare_workspace(
    *,
    config_path: Path,
    output_dir: Path,
    repository_root: Path,
) -> Path:
    """Create a calibration workspace whose calibration frames are exactly the anchors.

    The workspace is a plain `battle-muggled-calibration-web` directory; nothing here is
    special-cased in the web tool. Frame 0 is not configured: the workspace only needs it
    to *finalize* a tracking plan, which this session never does, so every anchor frame is
    editable from the first load.
    """
    config = load_config(config_path)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"workspace already exists: {output_dir}")
    clip_config = (repository_root / config.clip_config).resolve()
    target_config_path = (repository_root / config.manual_seed_target_config).resolve()
    target_config = load_manual_seed_target_config(
        target_config_path=target_config_path,
        repository_root=repository_root,
        g2_config_path=clip_config,
        view_id=config.view_id,
    )
    if tuple(target_config.targets) != config.targets:
        raise ValueError("anchor targets must equal the manual-seed target policy order")
    output_dir.mkdir(parents=True)
    manifest = build_manifest(
        repository_root=repository_root,
        config_path=clip_config,
        timestamps=tuple(frame.proxy_seconds for frame in config.frames),
        result_directory=output_dir / "results",
        calibration_id=output_dir.name,
        view_id=config.view_id,
    ).model_copy(update={"tool_version": TOOL_VERSION})
    if manifest.proxy_fps != config.analysis_fps or (
        abs(manifest.source_offset_seconds - config.source_offset_seconds) > 1e-9
    ):
        raise ValueError("anchor clock does not match the clip configuration's proxy")
    manifest_path = output_dir / "calibration_manifest.json"
    _write_manifest(manifest_path, manifest)
    session = ReviewAnchorSession(
        manifest_kind="human_review_anchor_session",
        config=_fingerprint(config_path, repository_root),
        calibration_manifest_uri=relative_uri(manifest_path, repository_root),
        manual_seed_target_config=_fingerprint(target_config_path, repository_root),
        prepared_at=datetime.now(UTC),
        claim_boundary=config.claim_boundary,
    )
    (output_dir / SESSION_NAME).write_text(
        session.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return manifest_path


def load_session(workspace: Path) -> ReviewAnchorSession:
    return ReviewAnchorSession.model_validate_json(
        (workspace / SESSION_NAME).read_text(encoding="utf-8")
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def export_anchor_masks(
    *,
    workspace: Path,
    repository_root: Path,
    config_path: Path | None = None,
    record_path: Path | None = None,
) -> tuple[ReviewAnchorMaskSet, Path, Path | None]:
    """Turn the workspace's accepted masks and hidden marks into the anchor mask set.

    Every cell of anchor frame x target is written: labeled (mask copied byte-for-byte
    and fingerprinted), hidden, or unlabeled. Unlabeled cells are kept so a later scorer
    can count what is missing instead of silently averaging over fewer anchors.
    """
    workspace = workspace.resolve()
    session = load_session(workspace)
    if config_path is None:
        config_path = repository_root / session.config.uri
    config = load_config(config_path)
    manifest_path = workspace / "calibration_manifest.json"
    manifest = MuggledSAMBoxCalibrationManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    if manifest.view_id != config.view_id:
        raise ValueError("workspace view does not match the anchor configuration")
    configured = {
        round(seconds * manifest.proxy_fps)
        for seconds in manifest.requested_proxy_timestamps_seconds
    }
    missing = [
        frame.analysis_frame_index
        for frame in config.frames
        if frame.analysis_frame_index not in configured
    ]
    if missing:
        raise ValueError(f"workspace lacks anchor frames {missing}")
    mask_directory = workspace / MASK_DIRECTORY
    mask_directory.mkdir(parents=True, exist_ok=True)
    width, height = manifest.proxy_dimensions.width, manifest.proxy_dimensions.height
    hidden = {
        (mark.frame.analysis_frame_index, mark.intended_target) for mark in manifest.hidden_targets
    }
    anchors: list[ReviewAnchorMask] = []
    for frame in config.frames:
        index = frame.analysis_frame_index
        for target in config.targets:
            expected = frame.expected_visible[target]
            accepted = [
                c
                for c in manifest.candidates
                if c.human_accepted
                and not c.rejected
                and c.selected_by == "human"
                and c.intended_target == target
                and c.frame.analysis_frame_index == index
            ]
            if len(accepted) > 1:
                raise ValueError(f"frame {index} {target}: more than one accepted mask")
            if accepted:
                candidate = accepted[0]
                assert candidate.human_selected_candidate_index is not None
                option = next(
                    o
                    for o in candidate.decoder_result.candidates
                    if o.candidate_index == candidate.human_selected_candidate_index
                )
                source_path = (workspace / option.mask_uri).resolve()
                if not source_path.is_relative_to(workspace) or not source_path.is_file():
                    raise FileNotFoundError(f"accepted mask is unavailable: {source_path}")
                payload = source_path.read_bytes()
                mask = mask_cache.decode_mask_png(source_path)
                if mask.shape != (height, width):
                    raise ValueError(
                        f"frame {index} {target}: mask is {mask.shape[::-1]}, proxy is "
                        f"{width}x{height}"
                    )
                destination = mask_directory / f"f{index:06d}_{target}.png"
                destination.write_bytes(payload)
                anchors.append(
                    ReviewAnchorMask(
                        analysis_frame_index=index,
                        target=target,
                        expected_visible=expected,
                        state="labeled",
                        mask_uri=destination.relative_to(workspace).as_posix(),
                        mask_sha256=_sha256_bytes(payload),
                        area_pixels=int(mask.sum()),
                        source_candidate_id=candidate.candidate_id,
                        source_candidate_index=candidate.human_selected_candidate_index,
                        source_mask_uri=option.mask_uri,
                        selected_by="human",
                    )
                )
            elif (index, target) in hidden:
                anchors.append(
                    ReviewAnchorMask(
                        analysis_frame_index=index,
                        target=target,
                        expected_visible=expected,
                        state="hidden",
                        selected_by="human",
                    )
                )
            else:
                anchors.append(
                    ReviewAnchorMask(
                        analysis_frame_index=index,
                        target=target,
                        expected_visible=expected,
                        state="unlabeled",
                    )
                )
    counts: dict[AnchorState, int] = {"labeled": 0, "hidden": 0, "unlabeled": 0}
    for anchor in anchors:
        counts[anchor.state] += 1
    mask_set = ReviewAnchorMaskSet(
        manifest_kind="human_review_anchor_masks",
        config=_fingerprint(config_path, repository_root),
        calibration_manifest=_fingerprint(manifest_path, repository_root),
        view_id=config.view_id,
        targets=config.targets,
        mask_dimensions=VideoDimensions(width=width, height=height),
        windows=config.windows,
        anchors=tuple(anchors),
        counts=counts,
        exported_at=datetime.now(UTC),
        claim_boundary=config.claim_boundary,
        license=config.license,
        provenance=config.provenance,
    )
    mask_set_path = workspace / MASK_SET_NAME
    mask_set_path.write_text(mask_set.model_dump_json(indent=2) + "\n", encoding="utf-8")
    written_record: Path | None = None
    if record_path is not None:
        record = ReviewAnchorHumanRecord(
            manifest_kind="human_review_anchor_record",
            author=config.provenance.author,
            reviewed_at=config.provenance.reviewed_at,
            config=mask_set.config,
            mask_set=_fingerprint(mask_set_path, repository_root),
            calibration_manifest=mask_set.calibration_manifest,
            counts=counts,
            anchors=tuple(
                ReviewAnchorRecordEntry(
                    analysis_frame_index=a.analysis_frame_index,
                    target=a.target,
                    state=a.state,
                    mask_sha256=a.mask_sha256,
                )
                for a in anchors
            ),
            claim_boundary=config.claim_boundary,
            license=config.license,
            notes=(
                "Skeleton written by battle-anchor-export; author and reviewed_at stay null "
                "until the human who labelled fills them in."
                if config.provenance.author is None
                else None
            ),
        )
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
        written_record = record_path
    return mask_set, mask_set_path, written_record


def load_mask_set(anchors: Path) -> tuple[ReviewAnchorMaskSet, Path]:
    """Accept the workspace directory or the mask-set JSON itself."""
    path = anchors.resolve()
    if path.is_dir():
        path = path / MASK_SET_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"anchor mask set is unavailable: {path} (run battle-anchor-export first)"
        )
    root = path.parent.parent if path.parent.name == "anchors" else path.parent
    return ReviewAnchorMaskSet.model_validate_json(path.read_text(encoding="utf-8")), root


# ------------------------------------------------------------------------------ scoring


class AnchorCellScore(VersionedModel):
    analysis_frame_index: int = Field(ge=0)
    target: str
    anchor_state: AnchorState
    outcome: CellOutcome
    iou: float | None = Field(default=None, ge=0, le=1)
    anchor_area: int | None = Field(default=None, ge=0)
    run_area: int | None = Field(default=None, ge=0)
    # run area / anchor area; > 1 means the tracker's mask is larger than the human's.
    area_ratio: float | None = Field(default=None, ge=0)
    run_mask_uri: str | None = None
    resized_run_mask: bool = False


class AnchorRunScore(VersionedModel):
    run_name: str = Field(min_length=1)
    run_directory: str = Field(min_length=1)
    cells: tuple[AnchorCellScore, ...]
    mean_iou: float | None = None
    mean_iou_by_target: dict[str, float | None]
    mean_iou_by_window: dict[str, float | None]
    counts: dict[CellOutcome, int]
    hidden_false_positive_area: int = Field(ge=0)


class AnchorIoUReport(VersionedModel):
    manifest_kind: Literal["human_review_anchor_iou"]
    anchor_kind: Literal["human_review_anchor"] = ANCHOR_KIND
    anchor_set: ArtifactFingerprint
    targets: tuple[str, ...]
    windows: dict[str, tuple[int, int]]
    anchor_counts: dict[AnchorState, int]
    runs: tuple[AnchorRunScore, ...]
    generated_at: datetime
    claim_boundary: str


def _mean(values: Iterable[float | None]) -> float | None:
    items = [v for v in values if v is not None]
    return float(np.mean(items)) if items else None


def resolve_run_directory(argument: str) -> tuple[str, Path]:
    """`name=path`, a run directory, or an arm directory holding exactly one run."""
    name, separator, path_text = argument.partition("=")
    if not separator:
        name, path_text = "", argument
    path = Path(path_text).resolve()
    if not (path / "observations.jsonl").is_file():
        runs = (
            sorted(
                child
                for child in path.iterdir()
                if child.is_dir() and (child / "observations.jsonl").is_file()
            )
            if path.is_dir()
            else []
        )
        if len(runs) != 1:
            raise FileNotFoundError(
                f"{path} is neither a run directory nor an arm directory with one run"
            )
        return name or path.name, runs[0]
    return name or path.name, path


def run_masks_at(run_directory: Path, frames: Iterable[int]) -> dict[int, dict[str, str]]:
    """Frame -> label -> mask uri for the requested frames, read from observations.jsonl."""
    wanted = set(frames)
    found: dict[int, dict[str, str]] = {}
    with (run_directory / "observations.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            frame = int(row["analysis_frame_index"])
            if frame not in wanted:
                continue
            entry = found.setdefault(frame, {})
            for item in row.get("objects", ()):
                mask = item.get("mask") or {}
                if item.get("label") and mask.get("uri"):
                    entry[str(item["label"])] = str(mask["uri"])
    return found


def _resize_nearest(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    resized = image.resize((shape[1], shape[0]), Image.NEAREST)
    return np.asarray(resized) > 0


def score_cell(
    anchor: ReviewAnchorMask,
    anchor_mask: np.ndarray | None,
    run_mask: np.ndarray | None,
    *,
    run_mask_uri: str | None,
) -> AnchorCellScore:
    """Score one (frame, part) cell; pure so the fixtures can pin every branch."""
    base = {
        "analysis_frame_index": anchor.analysis_frame_index,
        "target": anchor.target,
        "anchor_state": anchor.state,
        "run_mask_uri": run_mask_uri,
    }
    run_present = run_mask is not None and bool(run_mask.any())
    run_area = int(run_mask.sum()) if run_present else 0
    if anchor.state == "unlabeled":
        return AnchorCellScore(**base, outcome="unlabeled_skipped", run_area=run_area)
    if anchor.state == "hidden":
        if run_present:
            return AnchorCellScore(**base, outcome="hidden_false_positive", run_area=run_area)
        return AnchorCellScore(**base, outcome="hidden_correct", run_area=0)
    assert anchor_mask is not None
    anchor_area = int(anchor_mask.sum())
    if not run_present:
        return AnchorCellScore(
            **base,
            outcome="run_mask_missing",
            iou=0.0,
            anchor_area=anchor_area,
            run_area=0,
            area_ratio=0.0,
        )
    assert run_mask is not None
    resized = False
    if run_mask.shape != anchor_mask.shape:
        run_mask = _resize_nearest(run_mask, anchor_mask.shape)
        resized = True
    union = int(np.logical_or(anchor_mask, run_mask).sum())
    intersection = int(np.logical_and(anchor_mask, run_mask).sum())
    return AnchorCellScore(
        **base,
        outcome="scored",
        iou=intersection / union if union else 0.0,
        anchor_area=anchor_area,
        run_area=run_area,
        area_ratio=(run_area / anchor_area) if anchor_area else None,
        resized_run_mask=resized,
    )


def score_run(
    *,
    run_name: str,
    run_directory: Path,
    mask_set: ReviewAnchorMaskSet,
    anchors_root: Path,
) -> AnchorRunScore:
    frames = mask_set.frames
    uris = run_masks_at(run_directory, frames)
    cache = mask_cache.cache_for(run_directory)
    cells: list[AnchorCellScore] = []
    for anchor in mask_set.anchors:
        anchor_mask = (
            mask_cache.decode_mask_png(anchors_root / anchor.mask_uri)
            if anchor.state == "labeled" and anchor.mask_uri is not None
            else None
        )
        uri = uris.get(anchor.analysis_frame_index, {}).get(anchor.target)
        run_mask = cache.mask(uri) if uri is not None else None
        cells.append(score_cell(anchor, anchor_mask, run_mask, run_mask_uri=uri))
    scored = [c for c in cells if c.iou is not None]
    counts: dict[CellOutcome, int] = {
        "scored": 0,
        "run_mask_missing": 0,
        "hidden_correct": 0,
        "hidden_false_positive": 0,
        "unlabeled_skipped": 0,
    }
    for cell in cells:
        counts[cell.outcome] += 1
    return AnchorRunScore(
        run_name=run_name,
        run_directory=str(run_directory),
        cells=tuple(cells),
        mean_iou=_mean(c.iou for c in scored),
        mean_iou_by_target={
            target: _mean(c.iou for c in scored if c.target == target)
            for target in mask_set.targets
        },
        mean_iou_by_window={
            name: _mean(c.iou for c in scored if window[0] <= c.analysis_frame_index < window[1])
            for name, window in mask_set.windows.items()
        },
        counts=counts,
        hidden_false_positive_area=sum(
            c.run_area or 0 for c in cells if c.outcome == "hidden_false_positive"
        ),
    )


def score_runs(
    *,
    anchors: Path,
    runs: Sequence[str],
    repository_root: Path,
) -> AnchorIoUReport:
    mask_set, anchors_root = load_mask_set(anchors)
    mask_set_path = anchors_root / MASK_SET_NAME
    scores = []
    for argument in runs:
        name, directory = resolve_run_directory(argument)
        scores.append(
            score_run(
                run_name=name,
                run_directory=directory,
                mask_set=mask_set,
                anchors_root=anchors_root,
            )
        )
    return AnchorIoUReport(
        manifest_kind="human_review_anchor_iou",
        anchor_set=_fingerprint(mask_set_path, repository_root),
        targets=mask_set.targets,
        windows=mask_set.windows,
        anchor_counts=mask_set.counts,
        runs=tuple(scores),
        generated_at=datetime.now(UTC),
        claim_boundary=mask_set.claim_boundary,
    )


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def markdown_table(report: AnchorIoUReport) -> str:
    """Rows = arms; columns = per-part mean IoU, overall, per-window means, then counts."""
    header = [
        "arm",
        *[f"IoU {t}" for t in report.targets],
        "IoU all",
        *[f"IoU {w}" for w in report.windows],
        "missing",
        "hidden FP (px)",
        "scored / unlabeled",
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for run in report.runs:
        row = [
            run.run_name,
            *[_fmt(run.mean_iou_by_target.get(t)) for t in report.targets],
            _fmt(run.mean_iou),
            *[_fmt(run.mean_iou_by_window.get(w)) for w in report.windows],
            str(run.counts["run_mask_missing"]),
            f"{run.counts['hidden_false_positive']} ({run.hidden_false_positive_area})",
            f"{run.counts['scored']} / {run.counts['unlabeled_skipped']}",
        ]
        lines.append("| " + " | ".join(row) + " |")
    counts = report.anchor_counts
    lines.append("")
    lines.append(
        f"Anchors: {counts.get('labeled', 0)} labeled, {counts.get('hidden', 0)} hidden, "
        f"{counts.get('unlabeled', 0)} unlabeled (skipped). Windows use the anchor frames "
        "inside each. " + report.claim_boundary
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------- CLIs


def _repository_root() -> Path:
    return Path.cwd().resolve()


def export_main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the human review-anchor labelling workspace, or export its accepted "
            "masks and hidden marks as a fingerprinted anchor mask set. Never labels."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare", help="create the calibration workspace with the anchor frames configured"
    )
    prepare.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    prepare.add_argument("--output-dir", type=Path, default=DEFAULT_WORKSPACE)
    export = commands.add_parser(
        "export", help="write anchors/anchor_masks.json and the docs/qa human-record skeleton"
    )
    export.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    export.add_argument("--config", type=Path, default=None)
    export.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    export.add_argument("--no-record", action="store_true")
    args = parser.parse_args()
    root = _repository_root()
    try:
        if args.command == "prepare":
            manifest_path = prepare_workspace(
                config_path=args.config.resolve(), output_dir=args.output_dir, repository_root=root
            )
            print(f"Prepared anchor workspace: {manifest_path.parent}")
            print(f"Manifest: {manifest_path}")
        else:
            mask_set, mask_set_path, record_path = export_anchor_masks(
                workspace=args.workspace,
                repository_root=root,
                config_path=args.config.resolve() if args.config else None,
                record_path=None if args.no_record else args.record.resolve(),
            )
            print(f"Anchor mask set: {mask_set_path}")
            print(f"Counts: {json.dumps(mask_set.counts)}")
            if record_path is not None:
                print(f"Human record skeleton: {record_path}")
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


def iou_main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score run directories against the human review anchors: IoU and area ratio per "
            "accepted mask, false positives on hidden parts, missing run masks flagged, "
            "unlabeled anchors skipped and counted. Review evidence, not accuracy."
        )
    )
    parser.add_argument("--anchors", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="[NAME=]PATH",
        help="a run directory or an arm directory holding exactly one run; repeatable",
    )
    parser.add_argument("--output", type=Path, required=True, help="typed JSON report")
    parser.add_argument(
        "--markdown", type=Path, default=None, help="table path (default: <output>.md)"
    )
    args = parser.parse_args()
    root = _repository_root()
    try:
        report = score_runs(anchors=args.anchors, runs=args.run, repository_root=root)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    table = markdown_table(report)
    markdown_path = args.markdown or args.output.with_suffix(".md")
    markdown_path.write_text(table, encoding="utf-8")
    print(table, end="")
    print(f"Report: {args.output}")
    print(f"Table: {markdown_path}")
