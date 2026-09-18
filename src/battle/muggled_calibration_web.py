"""Local browser workspace for selected-frame MuggledSAM box calibration only."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import queue
import subprocess
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

from .calibration_view_filters import (
    ViewFilterError,
    ViewOperatorCache,
    describe_pipeline,
    is_identity,
    normalize_resolution_divisor,
    normalize_settings,
    render_view,
    settings_key,
    unavailable_operator_ids,
)
from .muggled_calibration import (
    DEFAULT_CONFIG,
    DEFAULT_TIMESTAMPS_SECONDS,
    TOOL_VERSION,
    _write_manifest,
    build_manifest,
    calibration_id_from_output_directory,
    finalize_correction_schedule,
    finalize_prompt,
    frame_reference,
    load_correction_policy,
    next_pending_box_id,
    parse_timestamps,
)
from .muggled_smoke import (
    DEFAULT_MODEL,
    MUGGLED_SAM_PYTHON,
    MUGGLED_SAM_SOURCE,
    load_manual_seed_target_config,
    relative_uri,
    sha256_file,
)
from .schemas import (
    MuggledSAMBoxCalibrationManifest,
    MuggledSAMCalibrationCandidate,
    MuggledSAMCalibrationHiddenTarget,
    MuggledSAMCalibrationPendingBox,
    MuggledSAMManualSeedTargetConfig,
    MuggledSAMMultiKeyframeCorrectionPolicy,
    MuggledSAMSupersededTrackingPlan,
    NormalizedBox,
    NormalizedPoint,
    PixelBox,
    PixelPoint,
)

STATIC_DIRECTORY = Path(__file__).with_name("static") / "calibration"
VIEW_RENDER_CACHE_ENTRIES = 12
VIEW_SOURCE_CACHE_ENTRIES = 8
JOB_HISTORY_LIMIT = 32
# One sentence for both the refused request and the banner the browser shows, so the
# reason a control is disabled is worded identically wherever a human reads it.
CANDIDATE_REVIEW_LOCK_REASON = (
    "candidate review is locked after finalizing a tracking plan; "
    "completed plan artifacts remain unchanged. Choose Reopen for editing to amend the "
    "plan; the finalized artifacts stay on disk as a superseded revision."
)


class WorkerError(RuntimeError):
    """A structured failure returned by, or while reading, the external worker."""


class Decoder(Protocol):
    def request(
        self, command: str, payload: dict[str, Any], timeout: float = 120
    ) -> dict[str, Any]: ...

    def close(self) -> None: ...


class WorkerClient:
    """Correlated JSONL client that keeps the isolated model warm between requests."""

    def __init__(
        self, command: list[str], *, environment: dict[str, str], stderr_path: Path
    ) -> None:
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        self._stderr_file = stderr_path.open("a", encoding="utf-8")
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_file,
            text=True,
            bufsize=1,
            env=environment,
        )
        self._pending: dict[str, queue.Queue[dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self._reader = threading.Thread(target=self._drain_stdout, daemon=True)
        self._reader.start()

    def _drain_stdout(self) -> None:
        assert self._process.stdout is not None
        for line in self._process.stdout:
            try:
                response = json.loads(line)
                request_id = response["request_id"]
            except (TypeError, ValueError, KeyError) as error:
                self._fail_all(f"malformed worker response: {error}: {line.strip()}")
                continue
            with self._lock:
                waiter = self._pending.pop(str(request_id), None)
            if waiter is None:
                self._fail_all(f"mismatched or duplicate worker response: {request_id}")
            else:
                waiter.put(response)
        self._fail_all(f"worker exited with status {self._process.poll()}")

    def _fail_all(self, message: str) -> None:
        with self._lock:
            pending, self._pending = self._pending, {}
        for waiter in pending.values():
            waiter.put({"ok": False, "error": message})

    def request(
        self, command: str, payload: dict[str, Any], timeout: float = 120
    ) -> dict[str, Any]:
        request_id = uuid.uuid4().hex
        waiter: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        with self._lock:
            if self._process.poll() is not None:
                raise WorkerError(f"worker exited with status {self._process.returncode}")
            self._pending[request_id] = waiter
            assert self._process.stdin is not None
            try:
                self._process.stdin.write(
                    json.dumps({"request_id": request_id, "command": command, "payload": payload})
                    + "\n"
                )
                self._process.stdin.flush()
            except OSError as error:
                self._pending.pop(request_id, None)
                raise WorkerError(f"could not write decoder request: {error}") from error
        try:
            response = waiter.get(timeout=timeout)
        except queue.Empty as error:
            with self._lock:
                self._pending.pop(request_id, None)
            raise WorkerError(f"decoder timed out after {timeout:g}s") from error
        if not response.get("ok"):
            raise WorkerError(str(response.get("error", "worker returned an unknown error")))
        result = response.get("result")
        if not isinstance(result, dict):
            raise WorkerError("worker returned a non-object result")
        return result

    def close(self) -> None:
        if self._process.poll() is None:
            try:
                self.request("shutdown", {}, timeout=5)
            except WorkerError:
                self._process.terminate()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
        self._stderr_file.close()


def _normal_box(pixel_box: PixelBox, manifest: MuggledSAMBoxCalibrationManifest) -> NormalizedBox:
    width, height = manifest.proxy_dimensions.width, manifest.proxy_dimensions.height
    return NormalizedBox(
        x=pixel_box.x1 / width,
        y=pixel_box.y1 / height,
        width=(pixel_box.x2 - pixel_box.x1) / width,
        height=(pixel_box.y2 - pixel_box.y1) / height,
    )


def _normal_points(
    points: tuple[PixelPoint, ...], manifest: MuggledSAMBoxCalibrationManifest
) -> tuple[NormalizedPoint, ...]:
    """Convert source-image clicks to MuggledSAM's normalized point coordinates."""
    width, height = manifest.proxy_dimensions.width, manifest.proxy_dimensions.height
    if any(point.x > width or point.y > height for point in points):
        raise ValueError("prompt points must be inside the proxy dimensions")
    return tuple(NormalizedPoint(x=point.x / width, y=point.y / height) for point in points)


def _pixel_points(
    body: dict[str, Any], key: str, prior: tuple[PixelPoint, ...]
) -> tuple[PixelPoint, ...]:
    """Read an optional editable point list, retaining prior points on box-only updates."""
    if key not in body:
        return prior
    raw_points = body[key]
    if not isinstance(raw_points, list):
        raise ValueError(f"{key} must be a list of {{x, y}} point objects")
    return tuple(PixelPoint.model_validate(point) for point in raw_points)


def _muggledsam_prompt_payload(prompt: MuggledSAMCalibrationPendingBox) -> dict[str, Any]:
    """Return the exact list shapes consumed by ``encode_prompts(boxes, fg_points, bg_points)``."""
    box = prompt.normalized_box
    return {
        "boxes": [
            [
                [box.x, box.y],
                [box.x + box.width, box.y + box.height],
            ]
        ],
        "fg_points": [[point.x, point.y] for point in prompt.normalized_fg_points],
        "bg_points": [[point.x, point.y] for point in prompt.normalized_bg_points],
    }


def _same_prompt_revision(
    current: MuggledSAMCalibrationPendingBox | None,
    queued: MuggledSAMCalibrationPendingBox,
) -> bool:
    """Compare editable prompt content while ignoring queue-state bookkeeping."""
    return current is not None and current.model_copy(
        update={"stage": "pending"}
    ) == queued.model_copy(update={"stage": "pending"})


def _validate_manifest(
    manifest: MuggledSAMBoxCalibrationManifest,
) -> MuggledSAMBoxCalibrationManifest:
    """Force Pydantic validation after immutable model-copy updates."""
    return MuggledSAMBoxCalibrationManifest.model_validate(manifest.model_dump(mode="json"))


def manifest_diff(before: object, after: object, path: str = "") -> list[str]:
    """Return a concise audit diff; media paths are represented, never read."""
    if type(before) is not type(after):
        return [f"{path or '/'}: {before!r} → {after!r}"]
    if isinstance(before, dict):
        changes: list[str] = []
        for key in sorted(set(before) | set(after)):
            child = f"{path}/{key}"
            if key not in before:
                changes.append(f"{child}: added")
            elif key not in after:
                changes.append(f"{child}: deleted")
            else:
                changes.extend(manifest_diff(before[key], after[key], child))
        return changes
    if isinstance(before, list):
        if before == after:
            return []
        return [f"{path}: list changed ({len(before)} → {len(after)})"]
    return [] if before == after else [f"{path or '/'}: {before!r} → {after!r}"]


def _transactional_replace(contents: dict[Path, bytes]) -> None:
    """Replace related finalization artifacts, restoring prior files on a write failure."""
    temporary_paths: dict[Path, Path] = {}
    previous = {path: path.read_bytes() if path.exists() else None for path in contents}
    replaced: list[Path] = []
    try:
        for path, content in contents.items():
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
            ) as temporary:
                temporary.write(content)
                temporary_paths[path] = Path(temporary.name)
        for path, temporary_path in temporary_paths.items():
            temporary_path.replace(path)
            replaced.append(path)
    except OSError:
        for path in reversed(replaced):
            original = previous[path]
            if original is None:
                path.unlink(missing_ok=True)
            else:
                with tempfile.NamedTemporaryFile(
                    mode="wb", dir=path.parent, prefix=f".{path.name}.rollback.", delete=False
                ) as rollback:
                    rollback.write(original)
                    rollback_path = Path(rollback.name)
                rollback_path.replace(path)
        raise
    finally:
        for temporary_path in temporary_paths.values():
            temporary_path.unlink(missing_ok=True)


class Workspace:
    """Thread-safe manifest owner; decoding occurs outside HTTP request threads."""

    def __init__(
        self,
        *,
        repository_root: Path,
        manifest_path: Path,
        manifest: MuggledSAMBoxCalibrationManifest,
        decoder: Decoder | None,
        manual_seed_target_config: MuggledSAMManualSeedTargetConfig | None = None,
        manual_seed_target_config_path: Path | None = None,
        correction_policy: MuggledSAMMultiKeyframeCorrectionPolicy | None = None,
        correction_policy_path: Path | None = None,
    ) -> None:
        self.repository_root = repository_root
        self.manifest_path = manifest_path
        self.manifest = manifest
        self.decoder = decoder
        self.manual_seed_target_config = manual_seed_target_config
        self.manual_seed_target_config_path = manual_seed_target_config_path
        self.correction_policy = correction_policy
        self.correction_policy_path = correction_policy_path
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sam3-decode")
        self.jobs: dict[str, dict[str, Any]] = {}
        self.last_diff: list[str] = []
        # Display-only view renders are memoized in memory per frame and parameter set.
        # They are deliberately never written to disk: no run artifact may depend on,
        # or be altered by, a browser view preference.
        self.view_renders: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.view_operator_cache = ViewOperatorCache()
        # The source frame behind a view render is read-only and identical between
        # ticks of a slider, so it is decoded once rather than on every request.
        self.view_sources: OrderedDict[str, np.ndarray] = OrderedDict()
        self.view_frame_paths: dict[int, Path] = {}

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "manifest": self.manifest.model_dump(mode="json"),
                "worker_online": self.decoder is not None,
                "manual_seed_targets": (
                    list(self.manual_seed_target_config.targets)
                    if self.manual_seed_target_config is not None
                    else []
                ),
                "manual_seed_target_policy": (
                    {
                        "config_id": self.manual_seed_target_config.config_id,
                        "required_target_count": len(self.manual_seed_target_config.targets),
                    }
                    if self.manual_seed_target_config is not None
                    else None
                ),
                "correction_policy": (
                    {
                        "policy_id": self.correction_policy.policy_id,
                        "policy_version": self.correction_policy.policy_version,
                        "maximum_later_correction_keyframes_per_target": (
                            self.correction_policy.maximum_later_correction_keyframes_per_target
                        ),
                        "memory_semantics": self.correction_policy.correction_memory_semantics,
                    }
                    if self.correction_policy is not None
                    else None
                ),
                "last_diff": self.last_diff,
                "plan": {
                    "finalized": self._plan_is_finalized(),
                    "plan_revision": self._current_plan_revision(),
                    "final_proposal_uri": self.manifest.final_proposal_uri,
                    "final_correction_schedule_uri": (self.manifest.final_correction_schedule_uri),
                    "superseded_plans": [
                        plan.model_dump(mode="json") for plan in self.manifest.superseded_plans
                    ],
                    "lock_reason": (
                        CANDIDATE_REVIEW_LOCK_REASON if self._plan_is_finalized() else None
                    ),
                },
                "limits": {
                    "finalization": "Human-selected frame-0 masks initialize all slots; "
                    "later selected masks are optional corrections. This workspace never tracks.",
                    "diagnostics": "Decoder IoU and prompt overlap are diagnostics, not accuracy.",
                },
            }

    def _persist(self, next_manifest: MuggledSAMBoxCalibrationManifest) -> None:
        before = self.manifest.model_dump(mode="json")
        self.manifest = _validate_manifest(next_manifest)
        _write_manifest(self.manifest_path, self.manifest)
        self.last_diff = manifest_diff(before, self.manifest.model_dump(mode="json"))

    def _plan_is_finalized(self) -> bool:
        return (
            self.manifest.final_proposal_uri is not None
            or self.manifest.final_correction_schedule_uri is not None
        )

    def _require_draft_candidate_review(self) -> None:
        if self._plan_is_finalized():
            raise ValueError(CANDIDATE_REVIEW_LOCK_REASON)

    def _current_plan_revision(self) -> int:
        """Return the revision of the plan on disk, treating pre-revision runs as one."""
        return max(self.manifest.plan_revision, 1 if self._plan_is_finalized() else 0)

    def _prune_jobs(self) -> None:
        completed = [
            job_id for job_id, job in self.jobs.items() if job["status"] in {"succeeded", "failed"}
        ]
        remove_count = max(0, len(self.jobs) - JOB_HISTORY_LIMIT + 1)
        for job_id in completed[:remove_count]:
            self.jobs.pop(job_id, None)

    def _active_job_for_box(self, box_id: str) -> dict[str, Any] | None:
        return next(
            (
                job
                for job in self.jobs.values()
                if job["status"] in {"queued", "running"} and box_id in job["box_ids"]
            ),
            None,
        )

    def _require_no_active_decode(self) -> None:
        if any(job["status"] in {"queued", "running"} for job in self.jobs.values()):
            raise ValueError("wait for the active decoder job before finalizing")

    def _plan_paths(self, revision: int) -> tuple[Path, Path]:
        """Name the proposal and schedule files a given plan revision owns.

        Revision one keeps the historical unsuffixed names every existing run and reader
        already uses; later revisions land beside those files so a superseded plan is
        never rewritten.
        """
        suffix = "" if revision <= 1 else f".r{revision}"
        return (
            self.manifest_path.with_name(f"proposed_tracking_prompt{suffix}.json"),
            self.manifest_path.with_name(f"multi_keyframe_correction_schedule{suffix}.json"),
        )

    def reopen_tracking_plan(self) -> dict[str, Any]:
        """Return a finalized workspace to editable review without touching its artifacts.

        Only the manifest changes: the finalized proposal and schedule are recorded as a
        superseded revision and keep their bytes, and the next finalization writes its own
        revision rather than overwriting them.
        """
        with self.lock:
            if not self._plan_is_finalized():
                raise ValueError("this workspace is already open for editing")
            revision = self._current_plan_revision()
            superseded = (
                *self.manifest.superseded_plans,
                MuggledSAMSupersededTrackingPlan(
                    plan_revision=revision,
                    proposal_uri=self.manifest.final_proposal_uri,
                    correction_schedule_uri=self.manifest.final_correction_schedule_uri,
                ),
            )
            self._persist(
                self.manifest.model_copy(
                    update={
                        "final_proposal_uri": None,
                        "final_correction_schedule_uri": None,
                        "plan_revision": revision,
                        "superseded_plans": superseded,
                    }
                )
            )
            return self.snapshot()

    def add_or_update_prompt(
        self, body: dict[str, Any], box_id: str | None = None
    ) -> dict[str, Any]:
        with self.lock:
            self._require_draft_candidate_review()
            frame = frame_reference(
                float(body["timestamp"]),
                fps=self.manifest.proxy_fps,
                source_offset_seconds=self.manifest.source_offset_seconds,
                frame_count=self.manifest.proxy_frame_count,
            )
            if not self._is_requested_calibration_frame(frame.analysis_frame_index):
                raise ValueError(
                    "this frame is browse-only; prompts are limited to configured "
                    "calibration frames"
                )
            pixel_box = PixelBox.model_validate(body["pixel_box"])
            pending = list(self.manifest.workspace.pending_boxes)
            intended_target = str(body["intended_target"]).strip()
            if box_id is None:
                matching = [
                    item
                    for item in pending
                    if item.frame.analysis_frame_index == frame.analysis_frame_index
                    and item.intended_target == intended_target
                ]
                if len(matching) > 1:
                    raise ValueError(
                        "multiple editable prompts already exist for this target and frame"
                    )
                prior = matching[0] if matching else None
                box_id = prior.box_id if prior is not None else None
            else:
                prior = next((item for item in pending if item.box_id == box_id), None)
            if box_id is not None and prior is None:
                raise KeyError(f"pending prompt does not exist: {box_id}")
            active_job = self._active_job_for_box(box_id) if box_id is not None else None
            if active_job is not None and not active_job["live_preview"]:
                raise ValueError("wait for the manual prompt decode to finish")
            if any(
                item.box_id != box_id
                and item.frame.analysis_frame_index == frame.analysis_frame_index
                and item.intended_target == intended_target
                for item in pending
            ):
                raise ValueError("an editable prompt already exists for this target and frame")
            prompt = MuggledSAMCalibrationPendingBox(
                box_id=box_id or next_pending_box_id(self.manifest, frame.analysis_frame_index),
                intended_target=intended_target,
                frame=frame,
                pixel_box=pixel_box,
                normalized_box=_normal_box(pixel_box, self.manifest),
                pixel_fg_points=_pixel_points(
                    body, "pixel_fg_points", prior.pixel_fg_points if prior is not None else ()
                ),
                pixel_bg_points=_pixel_points(
                    body, "pixel_bg_points", prior.pixel_bg_points if prior is not None else ()
                ),
                stage=prior.stage if prior is not None else "pending",
            )
            prompt = prompt.model_copy(
                update={
                    "normalized_fg_points": _normal_points(prompt.pixel_fg_points, self.manifest),
                    "normalized_bg_points": _normal_points(prompt.pixel_bg_points, self.manifest),
                }
            )
            if prior is not None:
                pending[pending.index(prior)] = prompt
            else:
                pending.append(prompt)
            workspace = self.manifest.workspace.model_copy(
                update={"pending_boxes": tuple(pending), "selected_box_id": prompt.box_id}
            )
            candidates = tuple(
                candidate.model_copy(
                    update={
                        "human_selected_candidate_index": None,
                        "human_accepted": False,
                        "legacy_finalization_requested": False,
                        "selected_for_finalization": False,
                        "selected_for_correction": False,
                    }
                )
                if candidate.human_accepted
                and (
                    candidate.source_box_id == prompt.box_id
                    or (
                        candidate.frame.analysis_frame_index == prompt.frame.analysis_frame_index
                        and candidate.intended_target == prompt.intended_target
                    )
                )
                else candidate
                for candidate in self.manifest.candidates
            )
            self._persist(
                self.manifest.model_copy(update={"workspace": workspace, "candidates": candidates})
            )
            return prompt.model_dump(mode="json")

    def _is_requested_calibration_frame(self, frame_index: int) -> bool:
        return any(
            round(timestamp * self.manifest.proxy_fps) == frame_index
            for timestamp in self.manifest.requested_proxy_timestamps_seconds
        )

    def add_calibration_frame(self, timestamp: float) -> dict[str, Any]:
        """Promote one browse-only frame into the persisted calibration set."""
        with self.lock:
            self._require_draft_candidate_review()
            frame = frame_reference(
                timestamp,
                fps=self.manifest.proxy_fps,
                source_offset_seconds=self.manifest.source_offset_seconds,
                frame_count=self.manifest.proxy_frame_count,
            )
            if self._is_requested_calibration_frame(frame.analysis_frame_index):
                return {
                    "frame_index": frame.analysis_frame_index,
                    "proxy_seconds": frame.proxy_seconds,
                }
            timestamps = tuple(
                sorted(
                    {
                        *self.manifest.requested_proxy_timestamps_seconds,
                        frame.proxy_seconds,
                    }
                )
            )
            workspace = self.manifest.workspace.model_copy(
                update={
                    "active_proxy_timestamp_seconds": frame.proxy_seconds,
                    "selected_box_id": None,
                }
            )
            self._persist(
                self.manifest.model_copy(
                    update={
                        "requested_proxy_timestamps_seconds": timestamps,
                        "workspace": workspace,
                    }
                )
            )
            return {
                "frame_index": frame.analysis_frame_index,
                "proxy_seconds": frame.proxy_seconds,
            }

    def _hidden_cell(self, timestamp: float, intended_target: str) -> tuple[Any, str]:
        frame = frame_reference(
            timestamp,
            fps=self.manifest.proxy_fps,
            source_offset_seconds=self.manifest.source_offset_seconds,
            frame_count=self.manifest.proxy_frame_count,
        )
        if not self._is_requested_calibration_frame(frame.analysis_frame_index):
            raise ValueError(
                "this frame is browse-only; hidden marks are limited to configured "
                "calibration frames"
            )
        target = intended_target.strip()
        if not target:
            raise ValueError("intended_target is required")
        return frame, target

    def mark_target_hidden(self, timestamp: float, intended_target: str) -> dict[str, Any]:
        """Record that a human looked at this frame and found the target not visible.

        This is the explicit alternative to accepting a mask: the cell is reviewed, and
        the review says there is nothing to draw. It refuses while an accepted mask
        exists for the cell so the two statements can never coexist silently.
        """
        with self.lock:
            self._require_draft_candidate_review()
            frame, target = self._hidden_cell(timestamp, intended_target)
            if any(
                candidate.human_accepted
                and not candidate.rejected
                and candidate.intended_target == target
                and candidate.frame.analysis_frame_index == frame.analysis_frame_index
                for candidate in self.manifest.candidates
            ):
                raise ValueError("unaccept the accepted mask before marking this target hidden")
            mark = MuggledSAMCalibrationHiddenTarget(intended_target=target, frame=frame)
            others = tuple(
                item
                for item in self.manifest.hidden_targets
                if not (
                    item.intended_target == target
                    and item.frame.analysis_frame_index == frame.analysis_frame_index
                )
            )
            self._persist(self.manifest.model_copy(update={"hidden_targets": (*others, mark)}))
            return mark.model_dump(mode="json")

    def clear_hidden_target(self, timestamp: float, intended_target: str) -> None:
        with self.lock:
            self._require_draft_candidate_review()
            frame, target = self._hidden_cell(timestamp, intended_target)
            remaining = tuple(
                item
                for item in self.manifest.hidden_targets
                if not (
                    item.intended_target == target
                    and item.frame.analysis_frame_index == frame.analysis_frame_index
                )
            )
            if len(remaining) == len(self.manifest.hidden_targets):
                raise KeyError(
                    f"{target} is not marked hidden on frame {frame.analysis_frame_index}"
                )
            self._persist(self.manifest.model_copy(update={"hidden_targets": remaining}))

    def delete_prompt(self, box_id: str) -> None:
        with self.lock:
            self._require_draft_candidate_review()
            active_job = self._active_job_for_box(box_id)
            if active_job is not None and not active_job["live_preview"]:
                raise ValueError("wait for the manual prompt decode to finish")
            pending = tuple(
                item for item in self.manifest.workspace.pending_boxes if item.box_id != box_id
            )
            if len(pending) == len(self.manifest.workspace.pending_boxes):
                raise KeyError(f"pending prompt does not exist: {box_id}")
            selected = self.manifest.workspace.selected_box_id
            workspace = self.manifest.workspace.model_copy(
                update={
                    "pending_boxes": pending,
                    "selected_box_id": None if selected == box_id else selected,
                }
            )
            candidates = tuple(
                candidate
                for candidate in self.manifest.candidates
                if not (
                    candidate.live_preview
                    and candidate.source_box_id == box_id
                    and not candidate.human_accepted
                )
            )
            self._persist(
                self.manifest.model_copy(update={"workspace": workspace, "candidates": candidates})
            )

    def set_active_timestamp(self, timestamp: float) -> None:
        """Persist the current approved frame so an interrupted browser session resumes."""
        with self.lock:
            frame = frame_reference(
                timestamp,
                fps=self.manifest.proxy_fps,
                source_offset_seconds=self.manifest.source_offset_seconds,
                frame_count=self.manifest.proxy_frame_count,
            )
            if not self._is_requested_calibration_frame(frame.analysis_frame_index):
                raise ValueError("workspace timestamp must be one of the requested timestamps")
            if self._plan_is_finalized():
                return
            selected = next(
                (
                    prompt
                    for prompt in self.manifest.workspace.pending_boxes
                    if prompt.box_id == self.manifest.workspace.selected_box_id
                ),
                None,
            )
            workspace = self.manifest.workspace.model_copy(
                update={
                    "active_proxy_timestamp_seconds": frame.proxy_seconds,
                    "selected_box_id": (
                        selected.box_id
                        if selected is not None
                        and selected.frame.analysis_frame_index == frame.analysis_frame_index
                        else None
                    ),
                }
            )
            self._persist(self.manifest.model_copy(update={"workspace": workspace}))

    def frame_preview(self, frame_index: int) -> dict[str, Any]:
        """Return a source-frame JPEG without requiring the model worker."""
        if self.decoder is not None:
            return self.decoder.request("frame_preview", {"frame_index": frame_index})
        proxy_path = (self.repository_root / self.manifest.proxy.uri).resolve()
        if not proxy_path.is_file():
            raise WorkerError(f"approved e4 proxy is unavailable: {proxy_path}")
        frames_directory = self.manifest_path.parent / "results" / "frames"
        frames_directory.mkdir(parents=True, exist_ok=True)
        image_name = f"frame-{frame_index:06d}.jpg"
        image_path = frames_directory / image_name
        if not image_path.is_file():
            completed = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(proxy_path),
                    "-vf",
                    f"select=eq(n\\,{frame_index})",
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    str(image_path),
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
            if completed.returncode or not image_path.is_file():
                message = completed.stderr.strip() or "ffmpeg produced no image"
                raise WorkerError(f"could not extract source frame {frame_index}: {message}")
        return {"frame_index": frame_index, "image_uri": f"results/frames/{image_name}"}

    def _view_frame_path(self, frame_index: int) -> Path:
        """Resolve, and remember, the extracted frame a view render reads.

        Which file backs a frame index does not change during a session, so the lookup
        (a worker round trip when the decoder is online) happens once instead of on every
        slider tick. The file itself is still stat-ed on each render.
        """
        with self.lock:
            remembered = self.view_frame_paths.get(frame_index)
        if remembered is not None and remembered.is_file():
            return remembered
        preview = self.frame_preview(frame_index)
        image_path = (self.manifest_path.parent / str(preview["image_uri"])).resolve()
        run_root = self.manifest_path.parent.resolve()
        if run_root not in image_path.parents or not image_path.is_file():
            raise WorkerError(f"source frame image is unavailable: {image_path}")
        with self.lock:
            self.view_frame_paths[frame_index] = image_path
        return image_path

    def _view_source(self, image_path: Path, token: str, divisor: int) -> np.ndarray:
        """Return the read-only source pixels a view render works from.

        Decoding the frame and shrinking it are pure functions of the frame file and the
        chosen working resolution, so both are memoized: a slider drag should not pay for
        a JPEG decode on every tick.
        """
        with self.lock:
            cached = self.view_sources.get(token)
            if cached is not None:
                self.view_sources.move_to_end(token)
                return cached
        with Image.open(image_path) as opened:
            frame = opened.convert("RGB")
            if divisor > 1:
                frame = frame.resize(
                    (max(1, frame.width // divisor), max(1, frame.height // divisor)),
                    Image.BILINEAR,
                )
            source = np.asarray(frame)
        source.flags.writeable = False
        with self.lock:
            self.view_sources[token] = source
            while len(self.view_sources) > VIEW_SOURCE_CACHE_ENTRIES:
                self.view_sources.popitem(last=False)
        return source

    def render_frame_view(
        self, frame_index: int, settings: Any, resolution_divisor: object = 1
    ) -> dict[str, Any]:
        """Render display-only view aids for one frame without touching any artifact.

        The frame image is opened read-only, the enhanced result is returned as an
        in-memory PNG, and nothing about the request reaches the decoder, the manifest,
        or any stored mask, review, proposal, or correction schedule.
        """
        normalized = normalize_settings(settings)
        if is_identity(normalized):
            raise ViewFilterError("enable at least one view aid before rendering")
        missing = unavailable_operator_ids(normalized)
        if missing:
            raise ViewFilterError(
                f"these view aids need VIGRA and it is unavailable: {', '.join(missing)}"
            )
        divisor = normalize_resolution_divisor(resolution_divisor)
        image_path = self._view_frame_path(frame_index)
        stat = image_path.stat()
        # The token pins every cached result to these exact frame bytes, so a
        # re-extracted frame is never shown through a stale cache.
        token = f"{frame_index}:{divisor}:{stat.st_size}:{stat.st_mtime_ns}"
        key = f"{token}:{settings_key(normalized)}"
        with self.lock:
            cached = self.view_renders.get(key)
            if cached is not None:
                self.view_renders.move_to_end(key)
                return {**cached, "cached": True}
        started = time.perf_counter()
        source = self._view_source(image_path, token, divisor)
        rendered = render_view(source, normalized, cache=self.view_operator_cache, token=token)
        payload = {
            "frame_index": frame_index,
            "image_png_base64": _encode_view_png(rendered.pixels),
            "corners": [
                {**group, "points": _scaled_points(group["points"], divisor)}
                for group in rendered.corners
            ],
            "applied": list(rendered.applied),
            "backends": dict(rendered.backends),
            "resolution_divisor": divisor,
            "width": int(rendered.pixels.shape[1]),
            "height": int(rendered.pixels.shape[0]),
            "milliseconds": round((time.perf_counter() - started) * 1000),
        }
        with self.lock:
            self.view_renders[key] = payload
            while len(self.view_renders) > VIEW_RENDER_CACHE_ENTRIES:
                self.view_renders.popitem(last=False)
        return {**payload, "cached": False}

    def queue_decode(self, box_ids: list[str], *, live_preview: bool = False) -> str:
        if self.decoder is None:
            raise WorkerError(
                "decoder is offline (--no-worker); static workspace remains available"
            )
        with self.lock:
            self._require_draft_candidate_review()
            self._prune_jobs()
            pending = list(self.manifest.workspace.pending_boxes)
            chosen = [item for item in pending if item.box_id in set(box_ids)]
            if not chosen or len(chosen) != len(set(box_ids)):
                raise ValueError("select one or more existing pending prompts")
            if any(item.stage != "pending" for item in chosen):
                raise ValueError("wait for the queued prompt decode to finish")
            queued = [
                item.model_copy(update={"stage": "queued"}) if item in chosen else item
                for item in pending
            ]
            workspace = self.manifest.workspace.model_copy(update={"pending_boxes": tuple(queued)})
            self._persist(self.manifest.model_copy(update={"workspace": workspace}))
            job_id = uuid.uuid4().hex
            self.jobs[job_id] = {
                "status": "queued",
                "box_ids": box_ids,
                "live_preview": live_preview,
            }
            future = self.executor.submit(
                self._decode, job_id, tuple(chosen), live_preview=live_preview
            )
            self.jobs[job_id]["future"] = future
            return job_id

    def _decode(
        self,
        job_id: str,
        prompts: tuple[MuggledSAMCalibrationPendingBox, ...],
        *,
        live_preview: bool,
    ) -> None:
        try:
            with self.lock:
                self.jobs[job_id]["status"] = "running"
                next_number_by_frame: dict[int, int] = {}
                for candidate in self.manifest.candidates:
                    frame_index = candidate.frame.analysis_frame_index
                    suffix = candidate.candidate_id.rsplit("b", 1)[-1]
                    next_number_by_frame[frame_index] = max(
                        next_number_by_frame.get(frame_index, 0), int(suffix)
                    )
                requests = []
                for prompt in prompts:
                    frame_index = prompt.frame.analysis_frame_index
                    next_number_by_frame[frame_index] = next_number_by_frame.get(frame_index, 0) + 1
                    candidate_id = f"t{frame_index:06d}-b{next_number_by_frame[frame_index]:02d}"
                    requests.append(
                        {
                            "box_id": prompt.box_id,
                            "candidate_id": candidate_id,
                            "frame_index": frame_index,
                            "pixel_box": prompt.pixel_box.model_dump(mode="json"),
                            "intended_target": prompt.intended_target,
                            **_muggledsam_prompt_payload(prompt),
                        }
                    )
            assert self.decoder is not None
            response = self.decoder.request("batch_decode", {"prompts": requests})
            by_box = {item["box_id"]: item for item in response["decoded"]}
            with self.lock:
                current_by_id = {
                    prompt.box_id: prompt for prompt in self.manifest.workspace.pending_boxes
                }
                fresh_prompts = tuple(
                    prompt
                    for prompt in prompts
                    if _same_prompt_revision(current_by_id.get(prompt.box_id), prompt)
                )
                stale_box_ids = sorted(
                    prompt.box_id for prompt in prompts if prompt not in fresh_prompts
                )
                candidates = list(self.manifest.candidates)
                if live_preview:
                    source_box_ids = {prompt.box_id for prompt in fresh_prompts}
                    candidates = [
                        candidate
                        for candidate in candidates
                        if not (
                            candidate.live_preview
                            and candidate.source_box_id in source_box_ids
                            and not candidate.human_accepted
                        )
                    ]
                decoded_candidate_ids = []
                for prompt in fresh_prompts:
                    item = by_box[prompt.box_id]
                    decoded_candidate_ids.append(item["candidate_id"])
                    candidates.append(
                        MuggledSAMCalibrationCandidate(
                            candidate_id=item["candidate_id"],
                            intended_target=prompt.intended_target,
                            frame=prompt.frame,
                            pixel_box=prompt.pixel_box,
                            normalized_box=prompt.normalized_box,
                            pixel_fg_points=prompt.pixel_fg_points,
                            pixel_bg_points=prompt.pixel_bg_points,
                            normalized_fg_points=prompt.normalized_fg_points,
                            normalized_bg_points=prompt.normalized_bg_points,
                            decoder_result=item["decoder_result"],
                            source_box_id=prompt.box_id if live_preview else None,
                            live_preview=live_preview,
                        )
                    )
                removed_ids = {prompt.box_id for prompt in prompts}
                if live_preview:
                    workspace = self.manifest.workspace.model_copy(
                        update={
                            "pending_boxes": tuple(
                                item.model_copy(update={"stage": "pending"})
                                if item.box_id in removed_ids
                                else item
                                for item in self.manifest.workspace.pending_boxes
                            )
                        }
                    )
                else:
                    fresh_ids = {prompt.box_id for prompt in fresh_prompts}
                    workspace = self.manifest.workspace.model_copy(
                        update={
                            "pending_boxes": tuple(
                                item.model_copy(update={"stage": "pending"})
                                if item.box_id in removed_ids
                                else item
                                for item in self.manifest.workspace.pending_boxes
                                if item.box_id not in fresh_ids
                            ),
                            "selected_box_id": (
                                None
                                if self.manifest.workspace.selected_box_id in fresh_ids
                                else self.manifest.workspace.selected_box_id
                            ),
                        }
                    )
                self._persist(
                    self.manifest.model_copy(
                        update={"candidates": tuple(candidates), "workspace": workspace}
                    )
                )
                self.jobs[job_id].update(
                    {
                        "status": "succeeded",
                        "candidate_count": len(fresh_prompts),
                        "candidate_ids": decoded_candidate_ids,
                        "stale_box_ids": stale_box_ids,
                    }
                )
        except Exception as error:
            with self.lock:
                self.jobs[job_id].update({"status": "failed", "error": str(error)})
                failed_box_ids = set(self.jobs[job_id]["box_ids"])
                workspace = self.manifest.workspace.model_copy(
                    update={
                        "pending_boxes": tuple(
                            item.model_copy(update={"stage": "pending"})
                            if item.box_id in failed_box_ids
                            else item
                            for item in self.manifest.workspace.pending_boxes
                        )
                    }
                )
                self._persist(self.manifest.model_copy(update={"workspace": workspace}))

    def accept_candidate(
        self, candidate_id: str, index: int, eligible: bool, correction: bool | None = None
    ) -> None:
        with self.lock:
            self._require_draft_candidate_review()
            candidates = list(self.manifest.candidates)
            candidate = next(
                (item for item in candidates if item.candidate_id == candidate_id), None
            )
            if candidate is None:
                raise KeyError(f"decoded candidate does not exist: {candidate_id}")
            if candidate.rejected:
                raise ValueError("restore a rejected candidate before accepting it")
            if eligible and candidate.frame.analysis_frame_index != 0:
                raise ValueError("only frame-0 masks can be marked finalization eligible")
            if correction and candidate.frame.analysis_frame_index == 0:
                raise ValueError("only later-frame masks can be marked correction eligible")
            candidates = [
                other.model_copy(
                    update={
                        "human_selected_candidate_index": None,
                        "human_accepted": False,
                        "legacy_finalization_requested": False,
                        "selected_for_finalization": False,
                        "selected_for_correction": False,
                    }
                )
                if other.candidate_id != candidate_id
                and other.human_accepted
                and other.intended_target == candidate.intended_target
                and other.frame.analysis_frame_index == candidate.frame.analysis_frame_index
                else other
                for other in candidates
            ]
            if (
                correction
                and self.correction_policy is not None
                and self.correction_policy.maximum_later_correction_keyframes_per_target == 1
            ):
                candidates = [
                    other.model_copy(update={"selected_for_correction": False})
                    if other.candidate_id != candidate_id
                    and other.intended_target == candidate.intended_target
                    and other.selected_for_correction
                    else other
                    for other in candidates
                ]
            accepted = candidate.model_copy(
                update={
                    "human_selected_candidate_index": index,
                    "human_accepted": True,
                    "legacy_finalization_requested": False,
                    "selected_for_finalization": eligible,
                    "selected_for_correction": (
                        candidate.selected_for_correction if correction is None else correction
                    ),
                }
            )
            candidates[candidates.index(candidate)] = accepted
            # Accepting a mask is the human saying the part is visible here, which
            # supersedes any earlier hidden mark on the same cell.
            hidden_targets = tuple(
                item
                for item in self.manifest.hidden_targets
                if not (
                    item.intended_target == candidate.intended_target
                    and item.frame.analysis_frame_index == candidate.frame.analysis_frame_index
                )
            )
            self._persist(
                self.manifest.model_copy(
                    update={"candidates": tuple(candidates), "hidden_targets": hidden_targets}
                )
            )

    def unaccept_candidate(self, candidate_id: str) -> None:
        """Explicitly clear a human choice before a candidate can be rejected."""
        with self.lock:
            self._require_draft_candidate_review()
            candidates = list(self.manifest.candidates)
            candidate = next(
                (item for item in candidates if item.candidate_id == candidate_id), None
            )
            if candidate is None:
                raise KeyError(f"decoded candidate does not exist: {candidate_id}")
            if candidate.rejected:
                raise ValueError("restore a rejected candidate before changing its acceptance")
            unaccepted = candidate.model_copy(
                update={
                    "human_selected_candidate_index": None,
                    "human_accepted": False,
                    "legacy_finalization_requested": False,
                    "selected_for_finalization": False,
                    "selected_for_correction": False,
                }
            )
            candidates[candidates.index(candidate)] = unaccepted
            self._persist(self.manifest.model_copy(update={"candidates": tuple(candidates)}))

    def reject_candidate(self, candidate_id: str) -> None:
        """Hide a decoded candidate without deleting its mask artifact or provenance."""
        with self.lock:
            self._require_draft_candidate_review()
            candidates = list(self.manifest.candidates)
            candidate = next(
                (item for item in candidates if item.candidate_id == candidate_id), None
            )
            if candidate is None:
                raise KeyError(f"decoded candidate does not exist: {candidate_id}")
            if candidate.human_accepted:
                raise ValueError("unaccept the current accepted candidate before rejecting it")
            rejected = candidate.model_copy(
                update={
                    "legacy_finalization_requested": False,
                    "selected_for_finalization": False,
                    "selected_for_correction": False,
                    "rejected": True,
                }
            )
            candidates[candidates.index(candidate)] = rejected
            self._persist(self.manifest.model_copy(update={"candidates": tuple(candidates)}))

    def restore_candidate(self, candidate_id: str) -> None:
        """Return a rejected candidate to ordinary review without auto-accepting it."""
        with self.lock:
            self._require_draft_candidate_review()
            candidates = list(self.manifest.candidates)
            candidate = next(
                (item for item in candidates if item.candidate_id == candidate_id), None
            )
            if candidate is None:
                raise KeyError(f"decoded candidate does not exist: {candidate_id}")
            if not candidate.rejected:
                raise ValueError(f"decoded candidate is not rejected: {candidate_id}")
            candidates[candidates.index(candidate)] = candidate.model_copy(
                update={"rejected": False}
            )
            self._persist(self.manifest.model_copy(update={"candidates": tuple(candidates)}))

    def create_proposal(self, candidate_ids: list[str]) -> dict[str, Any]:
        with self.lock:
            self._require_draft_candidate_review()
            self._require_no_active_decode()
            if self.decoder is None:
                raise WorkerError(
                    "cannot render the finalized seed review while the decoder worker is offline"
                )
            revision = self._current_plan_revision() + 1
            proposal_path, _ = self._plan_paths(revision)
            next_manifest = _validate_manifest(
                self.manifest.model_copy(
                    update={
                        "final_proposal_uri": relative_uri(proposal_path, self.repository_root),
                        "plan_revision": revision,
                    }
                )
            )
            manifest_content = (next_manifest.model_dump_json(indent=2) + "\n").encode()
            manifest_sha256 = hashlib.sha256(manifest_content).hexdigest()
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=proposal_path.parent,
                prefix=f".{proposal_path.name}.",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
            try:
                proposal = finalize_prompt(
                    manifest_path=self.manifest_path,
                    candidate_ids=tuple(candidate_ids),
                    proposal_path=temporary_path,
                    repository_root=self.repository_root,
                    manual_seed_target_config_path=self.manual_seed_target_config_path,
                    manifest=next_manifest,
                    calibration_manifest_sha256=manifest_sha256,
                )
                before = self.manifest.model_dump(mode="json")
                _transactional_replace(
                    {
                        proposal_path: temporary_path.read_bytes(),
                        self.manifest_path: manifest_content,
                    }
                )
                self.manifest = next_manifest
                self.last_diff = manifest_diff(before, next_manifest.model_dump(mode="json"))
            finally:
                temporary_path.unlink(missing_ok=True)
            review_path = (
                self.manifest_path.parent
                / "results"
                / "final_selected_seed_review_contact_sheet.png"
            )
            self.decoder.request(
                "render_final_selected_seed_review",
                {
                    "manifest_path": str(self.manifest_path),
                    "proposal_path": str(proposal_path),
                    "output_path": str(review_path),
                },
            )
            return proposal.model_dump(mode="json")

    def create_correction_schedule(self, candidate_ids: list[str]) -> dict[str, Any]:
        with self.lock:
            self._require_draft_candidate_review()
            self._require_no_active_decode()
            if self.correction_policy_path is None or self.manual_seed_target_config_path is None:
                raise WorkerError(
                    "a --correction-policy and --manual-seed-target-config are required "
                    "to finalize a multi-keyframe schedule"
                )
            revision = self._current_plan_revision() + 1
            _, schedule_path = self._plan_paths(revision)
            next_manifest = _validate_manifest(
                self.manifest.model_copy(
                    update={
                        "final_correction_schedule_uri": relative_uri(
                            schedule_path, self.repository_root
                        ),
                        "plan_revision": revision,
                    }
                )
            )
            manifest_content = (next_manifest.model_dump_json(indent=2) + "\n").encode()
            manifest_sha256 = hashlib.sha256(manifest_content).hexdigest()
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=schedule_path.parent,
                prefix=f".{schedule_path.name}.",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
            try:
                schedule = finalize_correction_schedule(
                    manifest_path=self.manifest_path,
                    candidate_ids=tuple(candidate_ids),
                    schedule_path=temporary_path,
                    correction_policy_path=self.correction_policy_path,
                    manual_seed_target_config_path=self.manual_seed_target_config_path,
                    repository_root=self.repository_root,
                    manifest=next_manifest,
                    calibration_manifest_sha256=manifest_sha256,
                )
                before = self.manifest.model_dump(mode="json")
                _transactional_replace(
                    {
                        schedule_path: temporary_path.read_bytes(),
                        self.manifest_path: manifest_content,
                    }
                )
                self.manifest = next_manifest
                self.last_diff = manifest_diff(before, next_manifest.model_dump(mode="json"))
            finally:
                temporary_path.unlink(missing_ok=True)
            return schedule.model_dump(mode="json")

    def finalize_tracking_plan(self) -> dict[str, Any]:
        """Validate and commit the selected plan without exposing partial artifacts."""
        with self.lock:
            self._require_draft_candidate_review()
            self._require_no_active_decode()
            initial_ids = tuple(
                candidate.candidate_id
                for candidate in self.manifest.candidates
                if candidate.selected_for_finalization and not candidate.rejected
            )
            later_ids = tuple(
                candidate.candidate_id
                for candidate in self.manifest.candidates
                if candidate.selected_for_correction and not candidate.rejected
            )
            if later_ids and self.correction_policy_path is None:
                raise ValueError(
                    "Later corrections are selected, but this workspace has no correction policy."
                )
            scheduled_before = self.manifest.final_correction_schedule_uri is not None or any(
                plan.correction_schedule_uri is not None for plan in self.manifest.superseded_plans
            )
            if scheduled_before and self.correction_policy_path is None:
                raise ValueError(
                    "This calibration already wrote a correction schedule; restart this "
                    "workspace with its --correction-policy to keep the schedule "
                    "integrity-bound before finalizing again."
                )

            revision = self._current_plan_revision() + 1
            proposal_path, schedule_path = self._plan_paths(revision)
            updates: dict[str, object] = {
                "final_proposal_uri": relative_uri(proposal_path, self.repository_root),
                "plan_revision": revision,
            }
            if self.correction_policy_path is not None:
                updates["final_correction_schedule_uri"] = relative_uri(
                    schedule_path, self.repository_root
                )
            next_manifest = _validate_manifest(self.manifest.model_copy(update=updates))
            manifest_content = (next_manifest.model_dump_json(indent=2) + "\n").encode()
            manifest_sha256 = hashlib.sha256(manifest_content).hexdigest()

            temporary_paths: list[Path] = []
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=proposal_path.parent,
                    prefix=f".{proposal_path.name}.",
                    delete=False,
                ) as temporary:
                    proposal_temporary_path = Path(temporary.name)
                temporary_paths.append(proposal_temporary_path)
                proposal = finalize_prompt(
                    manifest_path=self.manifest_path,
                    candidate_ids=initial_ids,
                    proposal_path=proposal_temporary_path,
                    repository_root=self.repository_root,
                    manual_seed_target_config_path=self.manual_seed_target_config_path,
                    manifest=next_manifest,
                    calibration_manifest_sha256=manifest_sha256,
                )
                contents = {proposal_path: proposal_temporary_path.read_bytes()}
                schedule = None
                if self.correction_policy_path is not None:
                    if self.manual_seed_target_config_path is None:
                        raise WorkerError(
                            "a correction policy requires a manual seed target configuration"
                        )
                    with tempfile.NamedTemporaryFile(
                        mode="w",
                        encoding="utf-8",
                        dir=schedule_path.parent,
                        prefix=f".{schedule_path.name}.",
                        delete=False,
                    ) as temporary:
                        schedule_temporary_path = Path(temporary.name)
                    temporary_paths.append(schedule_temporary_path)
                    schedule = finalize_correction_schedule(
                        manifest_path=self.manifest_path,
                        candidate_ids=initial_ids + later_ids,
                        schedule_path=schedule_temporary_path,
                        correction_policy_path=self.correction_policy_path,
                        manual_seed_target_config_path=self.manual_seed_target_config_path,
                        repository_root=self.repository_root,
                        manifest=next_manifest,
                        calibration_manifest_sha256=manifest_sha256,
                    )
                    contents[schedule_path] = schedule_temporary_path.read_bytes()
                contents[self.manifest_path] = manifest_content
                before = self.manifest.model_dump(mode="json")
                _transactional_replace(contents)
                self.manifest = next_manifest
                self.last_diff = manifest_diff(before, next_manifest.model_dump(mode="json"))
                return {
                    "proposal": proposal.model_dump(mode="json"),
                    "correction_schedule": (
                        schedule.model_dump(mode="json") if schedule is not None else None
                    ),
                }
            finally:
                for temporary_path in temporary_paths:
                    temporary_path.unlink(missing_ok=True)

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
        if self.decoder is not None:
            self.decoder.close()


def _encode_view_png(pixels: np.ndarray) -> str:
    """Encode a display-only render as PNG, cheaply.

    Nothing is resampled or quantized, so the image a reader sees is exactly the one
    the operators produced. Only two things change relative to a default save: a render
    with no coloured edge overlay is stored as a single grey channel instead of three
    identical ones, and the deflate level is low because this never leaves localhost.
    """
    coloured = pixels[:, :, 0]
    monochrome = np.array_equal(coloured, pixels[:, :, 1]) and np.array_equal(
        coloured, pixels[:, :, 2]
    )
    image = (
        Image.fromarray(coloured, mode="L") if monochrome else Image.fromarray(pixels, mode="RGB")
    )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=1)
    return base64.b64encode(buffer.getvalue()).decode()


def _scaled_points(points: list[dict[str, Any]], divisor: int) -> list[dict[str, Any]]:
    """Lift marker coordinates from the working resolution back to frame coordinates."""
    if divisor == 1:
        return list(points)
    return [{**point, "x": point["x"] * divisor, "y": point["y"] * divisor} for point in points]


def _json(handler: BaseHTTPRequestHandler, status: HTTPStatus, body: object) -> None:
    encoded = json.dumps(body).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(encoded)))
    handler.end_headers()
    handler.wfile.write(encoded)


def make_handler(workspace: Workspace) -> type[BaseHTTPRequestHandler]:
    """Create a handler bound to one validated run directory."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args: object) -> None:
            return

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            value = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(value, dict):
                raise ValueError("JSON body must be an object")
            return value

        def _static(self, name: str) -> None:
            path = STATIC_DIRECTORY / name
            if not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            content_type = "text/javascript" if name.endswith(".js") else "text/css"
            if name.endswith(".html"):
                content_type = "text/html"
            content = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._static("index.html")
            elif parsed.path.startswith("/static/"):
                self._static(parsed.path.removeprefix("/static/"))
            elif parsed.path == "/api/state":
                _json(self, HTTPStatus.OK, workspace.snapshot())
            elif parsed.path == "/api/view-filters":
                _json(self, HTTPStatus.OK, describe_pipeline())
            elif parsed.path == "/api/diff":
                _json(self, HTTPStatus.OK, {"changes": workspace.last_diff})
            elif parsed.path.startswith("/api/jobs/"):
                job = workspace.jobs.get(parsed.path.rsplit("/", 1)[-1])
                if job is None:
                    _json(self, HTTPStatus.NOT_FOUND, {"error": "unknown job"})
                else:
                    _json(
                        self,
                        HTTPStatus.OK,
                        {key: value for key, value in job.items() if key != "future"},
                    )
            elif parsed.path == "/api/frame":
                try:
                    timestamp = float(parse_qs(parsed.query)["timestamp"][0])
                    frame = frame_reference(
                        timestamp,
                        fps=workspace.manifest.proxy_fps,
                        source_offset_seconds=workspace.manifest.source_offset_seconds,
                        frame_count=workspace.manifest.proxy_frame_count,
                    )
                    _json(self, HTTPStatus.OK, workspace.frame_preview(frame.analysis_frame_index))
                except (KeyError, ValueError, WorkerError, subprocess.TimeoutExpired) as error:
                    _json(self, HTTPStatus.BAD_REQUEST, {"error": str(error)})
            elif parsed.path.startswith("/artifacts/"):
                candidate = (
                    workspace.manifest_path.parent / parsed.path.removeprefix("/artifacts/")
                ).resolve()
                run_root = workspace.manifest_path.parent.resolve()
                if run_root not in candidate.parents or not candidate.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                content = candidate.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header(
                    "Content-Type", "image/png" if candidate.suffix == ".png" else "image/jpeg"
                )
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            try:
                body = self._body()
                if self.path == "/api/calibration-frames":
                    _json(
                        self,
                        HTTPStatus.CREATED,
                        workspace.add_calibration_frame(float(body["timestamp"])),
                    )
                elif self.path == "/api/prompts":
                    _json(self, HTTPStatus.CREATED, workspace.add_or_update_prompt(body))
                elif self.path == "/api/hidden-targets":
                    workspace.mark_target_hidden(
                        float(body["timestamp"]), str(body["intended_target"])
                    )
                    _json(self, HTTPStatus.CREATED, workspace.snapshot())
                elif self.path == "/api/hidden-targets/clear":
                    workspace.clear_hidden_target(
                        float(body["timestamp"]), str(body["intended_target"])
                    )
                    _json(self, HTTPStatus.OK, workspace.snapshot())
                elif self.path == "/api/workspace":
                    workspace.set_active_timestamp(float(body["timestamp"]))
                    _json(self, HTTPStatus.OK, workspace.snapshot())
                elif self.path == "/api/view-filters/render":
                    frame = frame_reference(
                        float(body["timestamp"]),
                        fps=workspace.manifest.proxy_fps,
                        source_offset_seconds=workspace.manifest.source_offset_seconds,
                        frame_count=workspace.manifest.proxy_frame_count,
                    )
                    _json(
                        self,
                        HTTPStatus.OK,
                        workspace.render_frame_view(
                            frame.analysis_frame_index,
                            body.get("settings"),
                            body.get("resolution_divisor", 1),
                        ),
                    )
                elif self.path == "/api/decode":
                    _json(
                        self,
                        HTTPStatus.ACCEPTED,
                        {
                            "job_id": workspace.queue_decode(
                                body["box_ids"],
                                live_preview=body.get("live_preview") is True,
                            )
                        },
                    )
                elif self.path.startswith("/api/candidates/") and self.path.endswith("/accept"):
                    workspace.accept_candidate(
                        self.path.split("/")[-2],
                        int(body["candidate_index"]),
                        bool(body["eligible"]),
                        (bool(body["correction"]) if "correction" in body else None),
                    )
                    _json(self, HTTPStatus.OK, workspace.snapshot())
                elif self.path.startswith("/api/candidates/") and self.path.endswith("/unaccept"):
                    workspace.unaccept_candidate(self.path.split("/")[-2])
                    _json(self, HTTPStatus.OK, workspace.snapshot())
                elif self.path.startswith("/api/candidates/") and self.path.endswith("/reject"):
                    workspace.reject_candidate(self.path.split("/")[-2])
                    _json(self, HTTPStatus.OK, workspace.snapshot())
                elif self.path.startswith("/api/candidates/") and self.path.endswith("/restore"):
                    workspace.restore_candidate(self.path.split("/")[-2])
                    _json(self, HTTPStatus.OK, workspace.snapshot())
                elif self.path == "/api/proposal":
                    _json(
                        self, HTTPStatus.CREATED, workspace.create_proposal(body["candidate_ids"])
                    )
                elif self.path == "/api/correction-schedule":
                    _json(
                        self,
                        HTTPStatus.CREATED,
                        workspace.create_correction_schedule(body["candidate_ids"]),
                    )
                elif self.path == "/api/finalize-tracking-plan":
                    _json(self, HTTPStatus.CREATED, workspace.finalize_tracking_plan())
                elif self.path == "/api/reopen-tracking-plan":
                    _json(self, HTTPStatus.OK, workspace.reopen_tracking_plan())
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
            except (
                KeyError,
                TypeError,
                ValueError,
                WorkerError,
                OSError,
                json.JSONDecodeError,
            ) as error:
                _json(self, HTTPStatus.BAD_REQUEST, {"error": str(error)})

        def do_PATCH(self) -> None:  # noqa: N802
            if not self.path.startswith("/api/prompts/"):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                _json(
                    self,
                    HTTPStatus.OK,
                    workspace.add_or_update_prompt(self._body(), self.path.rsplit("/", 1)[-1]),
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                _json(self, HTTPStatus.BAD_REQUEST, {"error": str(error)})

        def do_DELETE(self) -> None:  # noqa: N802
            try:
                if not self.path.startswith("/api/prompts/"):
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                workspace.delete_prompt(self.path.rsplit("/", 1)[-1])
                _json(self, HTTPStatus.NO_CONTENT, {})
            except KeyError as error:
                _json(self, HTTPStatus.NOT_FOUND, {"error": str(error)})

    return Handler


def make_workspace(args: argparse.Namespace, repository_root: Path) -> Workspace:
    view_id = getattr(args, "view", "ego-hmc21179183")
    output_directory = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (
            args.run_root / f"muggledsam-sam3-e4-web-calibration-{uuid.uuid4().hex[:12]}"
        ).resolve()
    )
    manifest_path = output_directory / "calibration_manifest.json"
    if args.resume:
        if not manifest_path.is_file():
            raise ValueError(f"--resume requires an existing calibration manifest: {manifest_path}")
        manifest = MuggledSAMBoxCalibrationManifest.model_validate_json(manifest_path.read_text())
        if manifest.base_g2_config_sha256 != sha256_file(args.config):
            raise ValueError("cannot resume: selected G2 configuration fingerprint changed")
        # A resumed workspace keeps the frames it persisted unless the caller names some.
        timestamps = (
            manifest.requested_proxy_timestamps_seconds
            if args.timestamps is None
            else parse_timestamps(args.timestamps)
        )
        configured_frames = {round(timestamp * manifest.proxy_fps) for timestamp in timestamps}
        persisted_frames = {
            round(timestamp * manifest.proxy_fps)
            for timestamp in manifest.requested_proxy_timestamps_seconds
        }
        if not configured_frames.issubset(persisted_frames):
            raise ValueError("cannot resume with new --timestamps; add frames in the workspace")
        if manifest.view_id != view_id:
            raise ValueError("cannot resume with a different --view")
    else:
        timestamps = parse_timestamps(
            ",".join(map(str, DEFAULT_TIMESTAMPS_SECONDS))
            if args.timestamps is None
            else args.timestamps
        )
        if output_directory.exists():
            raise ValueError(f"output directory exists; use --resume: {output_directory}")
        output_directory.mkdir(parents=True)
        manifest = build_manifest(
            repository_root=repository_root,
            config_path=args.config,
            timestamps=timestamps,
            result_directory=output_directory / "results",
            calibration_id=calibration_id_from_output_directory(output_directory),
            view_id=view_id,
        ).model_copy(update={"tool_version": TOOL_VERSION})
        _write_manifest(manifest_path, manifest)
    target_config_path = (
        getattr(args, "manual_seed_target_config", None).resolve()
        if getattr(args, "manual_seed_target_config", None) is not None
        else None
    )
    target_config = (
        load_manual_seed_target_config(
            target_config_path=target_config_path,
            repository_root=repository_root,
            g2_config_path=args.config.resolve(),
            view_id=manifest.view_id,
        )
        if target_config_path is not None
        else None
    )
    correction_policy_path = (
        getattr(args, "correction_policy", None).resolve()
        if getattr(args, "correction_policy", None) is not None
        else None
    )
    if correction_policy_path is not None and target_config_path is None:
        raise ValueError("--correction-policy requires --manual-seed-target-config")
    correction_policy = (
        load_correction_policy(
            correction_policy_path=correction_policy_path,
            manual_seed_target_config_path=target_config_path,
            repository_root=repository_root,
            g2_config_path=args.config.resolve(),
            view_id=manifest.view_id,
        )
        if correction_policy_path is not None and target_config_path is not None
        else None
    )
    decoder: Decoder | None = None
    if not args.no_worker:
        if not args.external_python.is_file():
            raise ValueError(
                f"configured MuggledSAM interpreter does not exist: {args.external_python}"
            )
        proxy_path = (repository_root / manifest.proxy.uri).resolve()
        if not proxy_path.is_file() or sha256_file(proxy_path) != manifest.proxy.sha256:
            raise ValueError("approved e4 proxy is missing or its checksum no longer matches")
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = "0"
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(MUGGLED_SAM_SOURCE), environment["PYTHONPATH"]]
            if environment.get("PYTHONPATH")
            else [str(MUGGLED_SAM_SOURCE)]
        )
        worker_path = Path(__file__).with_name("muggled_calibration_worker.py")
        decoder = WorkerClient(
            [
                str(args.external_python),
                str(worker_path),
                "--serve-jsonl",
                "--proxy",
                str(proxy_path),
                "--model",
                str(args.model.resolve()),
                "--results-directory",
                str(output_directory / "results"),
                "--device",
                args.device,
            ],
            environment=environment,
            stderr_path=output_directory / "worker.stderr.log",
        )
    return Workspace(
        repository_root=repository_root,
        manifest_path=manifest_path,
        manifest=manifest,
        decoder=decoder,
        manual_seed_target_config=target_config,
        manual_seed_target_config_path=target_config_path,
        correction_policy=correction_policy,
        correction_policy_path=correction_policy_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Serve a localhost-only selected-frame SAM3 calibration workspace; never tracks video."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--view",
        choices=("static-c10379", "ego-hmc21110305", "ego-hmc21179183"),
        default="ego-hmc21179183",
    )
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--timestamps",
        default=None,
        help=(
            "Comma-separated proxy seconds of the calibration frames "
            f"(default {','.join(map(str, DEFAULT_TIMESTAMPS_SECONDS))}; with --resume the "
            "persisted frames)."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--external-python", type=Path, default=MUGGLED_SAM_PYTHON)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--manual-seed-target-config",
        type=Path,
        help="Use this named target policy when creating a proposal.",
    )
    parser.add_argument(
        "--correction-policy",
        type=Path,
        help="Enable an integrity-bound multi-keyframe correction schedule for this target policy.",
    )
    parser.add_argument(
        "--no-worker", action="store_true", help="Serve static/state routes without decoder."
    )
    args = parser.parse_args()
    if args.host != "127.0.0.1":
        parser.error("this utility is deliberately loopback-only; --host must be 127.0.0.1")
    args.config = args.config.resolve()
    try:
        workspace = make_workspace(args, Path.cwd().resolve())
        server = ThreadingHTTPServer((args.host, args.port), make_handler(workspace))
        print(f"Rapid calibration workspace: http://{args.host}:{server.server_port}/")
        print(f"Manifest: {workspace.manifest_path}")
        try:
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
        finally:
            server.server_close()
            workspace.close()
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
