"""Shared, review-bound inputs for the four-part segmentation comparison."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

TARGETS = ("chassis", "interior", "rear_body", "cabin")
FRAME_COUNT = 600
ANALYSIS_FPS = 30


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_uri(path: Path, repository_root: Path) -> str:
    return path.resolve().relative_to(repository_root.resolve()).as_posix()


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
