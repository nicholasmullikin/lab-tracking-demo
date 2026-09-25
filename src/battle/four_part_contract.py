"""Shared, review-bound inputs for the four-part segmentation comparison."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .digest_cache import sha256_file
from .fs_common import relative_uri

TARGETS = ("chassis", "interior", "rear_body", "cabin")
FRAME_COUNT = 600
ANALYSIS_FPS = 30
# The Assembly101 review scope: the one human-reviewed static RGB view and the four parts.
# Builders that used to assert these literals read them from a clip config when one is
# given (`review_scope`) and fall back to these defaults, so the Assembly101 builds are
# unchanged and a FineBio clip config (`views`, `targets`) can drive the same code.
DEFAULT_TARGETS = TARGETS
DEFAULT_APPROVED_VIEWS: tuple[str, ...] = ("static-c10379",)


@dataclass(frozen=True)
class ReviewScope:
    """The views a build accepts and the ordered targets it expects."""

    views: tuple[str, ...] = DEFAULT_APPROVED_VIEWS
    targets: tuple[str, ...] = DEFAULT_TARGETS
    source: str = "defaults (Assembly101)"


def review_scope(
    clip_config: Path | None = None, *, repository_root: Path | None = None
) -> ReviewScope:
    """Views and targets from a clip config, Assembly101 defaults when none is given.

    Accepts both shapes in ``configs/clips``: the Assembly101 preprocessing manifest
    (``clip.views``; no target list, so the four parts stay) and the FineBio clip config
    (top-level ``views`` and ``targets``).
    """
    if clip_config is None:
        return ReviewScope()
    path = Path(clip_config)
    if repository_root is not None and not path.is_absolute():
        path = repository_root / path
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "views" in payload:
        views = tuple(str(v) for v in payload["views"])
    elif isinstance(payload.get("clip"), dict) and "views" in payload["clip"]:
        views = tuple(str(v) for v in payload["clip"]["views"])
    else:
        raise ValueError(f"{path} names no views")
    targets = tuple(str(t) for t in payload.get("targets", DEFAULT_TARGETS))
    if not views or not targets:
        raise ValueError(f"{path} has an empty views or targets list")
    return ReviewScope(views=views, targets=targets, source=str(clip_config))


@dataclass(frozen=True)
class ReviewedSeed:
    target_id: str
    candidate_id: str
    mask_path: Path
    mask_sha256: str
    box_xyxy: tuple[int, int, int, int]


@dataclass(frozen=True)
class FourPartContract:
    path: Path
    fingerprint: str
    proxy_path: Path
    source_offset_seconds: float
    seeds: tuple[ReviewedSeed, ...]
    open_vocabulary_prompts: dict[str, tuple[str, ...]]
    baseline_run: Path


def _resolve(repository_root: Path, uri: str) -> Path:
    path = (repository_root / uri).resolve()
    if not path.is_relative_to(repository_root.resolve()):
        raise ValueError(f"contract URI escapes repository: {uri}")
    return path


def _require_fingerprint(repository_root: Path, payload: dict[str, Any], *, name: str) -> Path:
    path = _resolve(repository_root, str(payload["uri"]))
    expected = str(payload["sha256"])
    if not path.is_file():
        raise FileNotFoundError(f"{name} is unavailable: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{name} fingerprint mismatch: expected {expected}, got {actual}")
    return path


def _mask_box(path: Path) -> tuple[int, int, int, int]:
    with Image.open(path) as image:
        mask = image.convert("L")
        box = mask.getbbox()
    if box is None:
        raise ValueError(f"reviewed seed mask is empty: {path}")
    return tuple(int(value) for value in box)


def load_contract(repository_root: Path, contract_path: Path) -> FourPartContract:
    """Load only the reviewed frame-zero masks named by the checked-in contract."""
    repository_root = repository_root.resolve()
    path = _resolve(repository_root, str(contract_path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("manifest_kind") != "four_part_segmentation_comparison_contract":
        raise ValueError("unexpected four-part comparison contract kind")
    if tuple(payload.get("targets", ())) != TARGETS:
        raise ValueError(f"contract targets must be exactly ordered {TARGETS}")
    frame_range = payload.get("analysis_frame_range", {})
    if frame_range != {"start_frame": 0, "end_frame_exclusive": FRAME_COUNT}:
        raise ValueError("comparison contract must be exactly analysis frames [0, 600)")
    proxy_path = _require_fingerprint(
        repository_root, payload["focused_static_proxy_fingerprint"], name="focused static proxy"
    )
    _require_fingerprint(
        repository_root, payload["focused_static_config_fingerprint"], name="focused static config"
    )
    schedule_path = _require_fingerprint(
        repository_root,
        payload["reviewed_schedule_fingerprint"],
        name="reviewed correction schedule",
    )
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    schedule_seeds = {
        item["target_id"]: item
        for item in schedule["corrections"]
        if item["frame"]["analysis_frame_index"] == 0
    }
    declared_seeds = payload.get("reviewed_frame_zero_seeds", ())
    if [item.get("target_id") for item in declared_seeds] != list(TARGETS):
        raise ValueError("contract must name one ordered reviewed frame-zero seed per target")
    seeds: list[ReviewedSeed] = []
    for declared in declared_seeds:
        target = str(declared["target_id"])
        scheduled = schedule_seeds.get(target)
        if scheduled is None:
            raise ValueError(f"reviewed schedule has no frame-zero seed for {target}")
        scheduled_mask = scheduled["calibration_mask_fingerprint"]
        declared_mask = declared["mask_fingerprint"]
        if scheduled["candidate_id"] != declared["candidate_id"] or (
            scheduled_mask["uri"] != declared_mask["uri"]
            or scheduled_mask["sha256"] != declared_mask["sha256"]
        ):
            raise ValueError(f"contract seed differs from reviewed schedule for {target}")
        mask_path = _require_fingerprint(
            repository_root, declared["mask_fingerprint"], name=f"{target} reviewed seed"
        )
        actual_box = _mask_box(mask_path)
        expected_box = tuple(declared["derived_box_xyxy"])
        if actual_box != expected_box:
            raise ValueError(
                f"{target} derived mask box differs from contract: "
                f"expected {expected_box}, got {actual_box}"
            )
        seeds.append(
            ReviewedSeed(
                target,
                str(declared["candidate_id"]),
                mask_path,
                str(declared["mask_fingerprint"]["sha256"]),
                actual_box,
            )
        )
    prompts = payload.get("open_vocabulary_prompts", {})
    if tuple(prompts) != TARGETS or any(not prompts[target] for target in TARGETS):
        raise ValueError("contract must give prompts/synonyms for every ordered target")
    baseline = _require_fingerprint(
        repository_root,
        payload["baseline_sam3_manifest_fingerprint"],
        name="baseline SAM3 manifest",
    )
    return FourPartContract(
        path=path,
        fingerprint=sha256_file(path),
        proxy_path=proxy_path,
        source_offset_seconds=float(payload["source_interval"]["start_seconds"]),
        seeds=tuple(seeds),
        open_vocabulary_prompts={target: tuple(prompts[target]) for target in TARGETS},
        baseline_run=baseline.parent,
    )


def seed_manifest(contract: FourPartContract, repository_root: Path) -> dict[str, Any]:
    """Portable record injected into each new arm; no seed bytes are copied."""
    return {
        "contract_uri": relative_uri(contract.path, repository_root),
        "contract_sha256": contract.fingerprint,
        "targets": list(TARGETS),
        "reviewed_frame_zero_seeds": [
            {
                "target_id": seed.target_id,
                "candidate_id": seed.candidate_id,
                "mask_fingerprint": {
                    "uri": relative_uri(seed.mask_path, repository_root),
                    "sha256": seed.mask_sha256,
                    "source": "measured",
                },
                "derived_box_xyxy": list(seed.box_xyxy),
            }
            for seed in contract.seeds
        ],
    }
