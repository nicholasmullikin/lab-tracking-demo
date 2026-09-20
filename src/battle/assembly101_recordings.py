"""Which Assembly101 recording and source window every Track 0 tool is talking about.

Until Sep 20 the acquisition, clock-offset, camera-fit, rig and reference modules carried
one recording (`nusar-2021_action_both_9033-c02a_9033_user_id_2021-02-04_140532`) and one
window (source 294.000-386.700 s) as module constants.  Track C needs the same tooling on a
second recording of the same toy by a different subject, so those facts now live in one
typed record per recording, checked in as `configs/assembly101/recordings.json`.  Every
module keeps its recording-1 constants and defaults, and takes a `recording=` argument for
anything else; nothing built on recording 1 changes.

A record only names dataset paths and a window; it makes no claim about the content of the
recording.  CC BY-NC 4.0 attribution applies to every asset it points at.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .schemas import VersionedModel

DATASET_REPO = "cvml-nus/assembly101"
DATASET_REVISION = "bfc15ea5e3f0bc8f8c232af6c1b45aa137a9d967"
SOURCE_FPS = 60
ANALYSIS_FPS = 30
ANNOTATION_FPS = 30
RECORDINGS_CONFIG = Path("configs/assembly101/recordings.json")

ALL_STATIC_VIEWS: tuple[str, ...] = (
    "C10095",
    "C10115",
    "C10118",
    "C10119",
    "C10379",
    "C10390",
    "C10395",
    "C10404",
)


class Assembly101Recording(VersionedModel):
    """One recording plus the one source window this repository works on.

    `window_*` is the fetched interval (trims and proxies cover all of it); `core_*` is the
    analysis span inside it that the per-recording clip and runs are about (recording 1: the
    first minute of the 92.7 s window; recording 2: 60 s inside an 80 s window with 10 s of
    margin each side).  Paths are repository-relative.
    """

    recording_id: str = Field(min_length=1)
    label: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    toy_id: str = Field(min_length=1)
    subject_id: str = Field(min_length=1)
    window_start_seconds: float = Field(ge=0)
    window_duration_seconds: float = Field(gt=0)
    core_start_seconds: float = Field(ge=0)
    core_duration_seconds: float = Field(gt=0)
    static_views: tuple[str, ...] = Field(min_length=1)
    ego_views: tuple[str, ...] = ()
    ego_views_on_hub: tuple[str, ...] = ()
    primary_static_view: str = Field(min_length=1)
    clock_rules_path: str = Field(min_length=1)
    camera_config_root: str = Field(min_length=1)
    clock_scan_root: str = Field(min_length=1)
    rig_check_root: str = Field(min_length=1)
    reference_root: str = Field(min_length=1)
    all_static_clip_config: str = Field(min_length=1)
    fine_grained_split: Literal["train", "validation", "test"] | None = None
    note: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_core_inside_window(self) -> Assembly101Recording:
        window_end = self.window_start_seconds + self.window_duration_seconds
        core_end = self.core_start_seconds + self.core_duration_seconds
        if (
            self.core_start_seconds < self.window_start_seconds - 1e-9
            or core_end > window_end + 1e-9
        ):
            raise ValueError("core span must lie inside the fetched window")
        if self.primary_static_view not in self.static_views:
            raise ValueError("primary static view must be one of the static views")
        if any(view not in self.ego_views_on_hub for view in self.ego_views):
            raise ValueError("every fetched ego view must be listed among the ego views on the Hub")
        return self

    # -- windows --------------------------------------------------------------------------

    @property
    def window_end_seconds(self) -> float:
        return self.window_start_seconds + self.window_duration_seconds

    @property
    def window_start_raw_frame(self) -> int:
        return round(self.window_start_seconds * SOURCE_FPS)

    @property
    def window_end_raw_frame_exclusive(self) -> int:
        return round(self.window_end_seconds * SOURCE_FPS)

    @property
    def window_raw_frame_count(self) -> int:
        return self.window_end_raw_frame_exclusive - self.window_start_raw_frame

    @property
    def window_proxy_frame_count(self) -> int:
        return round(self.window_duration_seconds * ANALYSIS_FPS)

    @property
    def annotation_start_frame(self) -> int:
        return round(self.window_start_seconds * ANNOTATION_FPS)

    @property
    def core_proxy_frame_range(self) -> tuple[int, int]:
        start = round((self.core_start_seconds - self.window_start_seconds) * ANALYSIS_FPS)
        return start, start + round(self.core_duration_seconds * ANALYSIS_FPS)

    # -- views ----------------------------------------------------------------------------

    @property
    def all_views(self) -> tuple[str, ...]:
        return (*self.static_views, *self.ego_views)

    # -- paths ----------------------------------------------------------------------------

    @property
    def raw_root(self) -> Path:
        return Path("data/raw/assembly101") / self.recording_id

    @property
    def derived_root(self) -> Path:
        return Path("data/derived/assembly101") / self.recording_id

    @property
    def local_recordings_root(self) -> Path:
        return self.raw_root / "recordings" / self.recording_id

    @property
    def poses_root(self) -> Path:
        return self.raw_root / "AssemblyPoses_selective/assembly101_camera_and_hand_poses"

    def poses_member(self, kind: str) -> Path:
        return self.poses_root / kind / f"{self.recording_id}.json"

    @property
    def fine_grained_root(self) -> Path:
        return self.raw_root / "annotations/fine-grained-annotations"

    def fine_grained_csv(self, repository_root: Path | None = None) -> Path:
        """The filtered split CSV holding this recording's rows.

        When the split is pinned in the registry the path is deterministic; otherwise the
        non-empty `<split>__<recording>.csv` under the raw tree is used (the streaming filter
        writes one file per split and only one carries rows).
        """
        if self.fine_grained_split is not None:
            return self.fine_grained_root / f"{self.fine_grained_split}__{self.recording_id}.csv"
        if repository_root is None:
            raise ValueError(f"{self.label}: fine-grained split unknown; pass repository_root")
        root = repository_root / self.fine_grained_root
        candidates = sorted(root.glob(f"*__{self.recording_id}.csv"))
        with_rows = [path for path in candidates if len(path.read_text().splitlines()) > 1]
        if len(with_rows) != 1:
            raise FileNotFoundError(
                f"{self.label}: expected exactly one non-empty fine-grained CSV under {root}, "
                f"found {[p.name for p in with_rows]}"
            )
        return self.fine_grained_root / with_rows[0].name

    @property
    def coarse_labels_path(self) -> Path:
        return (
            self.raw_root
            / "annotations/coarse-annotations/coarse_labels"
            / f"assembly_{self.recording_id}.txt"
        )

    @property
    def shipped_2d_window(self) -> Path:
        return self.derived_root / (
            "assembly101_landmarks2D_60fps_frames_"
            f"{self.window_start_raw_frame}_{self.window_end_raw_frame_exclusive}.npz"
        )

    @property
    def acquisition_records_root(self) -> Path:
        return self.raw_root / "static_views_focused_acquisition"

    @property
    def acquisition_report(self) -> Path:
        return self.raw_root / "static_views_focused_acquisition_report.md"

    @property
    def acquisition_manifest(self) -> Path:
        return self.raw_root / "static_views_focused_acquisition_manifest.json"

    def ego_clip_config(self, view: str) -> Path:
        return Path("configs/clips") / (
            f"assembly101_{self.label}_four_part_reassembly_focused_ego_{view.lower()}_g2.json"
        )


class Assembly101RecordingRegistry(VersionedModel):
    manifest_kind: Literal["assembly101_recordings"]
    hf_dataset: str = Field(min_length=1)
    hf_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    default_label: str = Field(min_length=1)
    recordings: tuple[Assembly101Recording, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_unique_labels(self) -> Assembly101RecordingRegistry:
        labels = [recording.label for recording in self.recordings]
        ids = [recording.recording_id for recording in self.recordings]
        if len(set(labels)) != len(labels) or len(set(ids)) != len(ids):
            raise ValueError("recording labels and ids must be unique")
        if self.default_label not in labels:
            raise ValueError("default_label must name a registered recording")
        return self

    def get(self, key: str) -> Assembly101Recording:
        """Look a recording up by label or by full recording id."""
        for recording in self.recordings:
            if key in (recording.label, recording.recording_id):
                return recording
        raise KeyError(
            f"no Assembly101 recording {key!r}; known: "
            + ", ".join(f"{r.label} ({r.recording_id})" for r in self.recordings)
        )

    @property
    def default(self) -> Assembly101Recording:
        return self.get(self.default_label)


# Recording 1 is also defined in code so the modules' historical constants and defaults do
# not depend on reading a config file at import time.
RECORDING_1 = Assembly101Recording(
    recording_id="nusar-2021_action_both_9033-c02a_9033_user_id_2021-02-04_140532",
    label="nusar_9033",
    toy_id="c02a",
    subject_id="9033",
    window_start_seconds=294.0,
    window_duration_seconds=92.7,
    core_start_seconds=294.0,
    core_duration_seconds=60.0,
    static_views=ALL_STATIC_VIEWS,
    ego_views=("HMC_21110305", "HMC_21176623", "HMC_21176875", "HMC_21179183"),
    ego_views_on_hub=("HMC_21110305", "HMC_21176623", "HMC_21176875", "HMC_21179183"),
    primary_static_view="C10379",
    clock_rules_path="configs/assembly101/clock_rules.json",
    camera_config_root="configs/assembly101",
    clock_scan_root="runs/assembly101-clock-offsets",
    rig_check_root="runs/assembly101-multiview-rig-check",
    reference_root="runs/assembly101-reference-first-minute-v1",
    all_static_clip_config=(
        "configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_all_static_g2.json"
    ),
    fine_grained_split="train",
    note=(
        "Recording 1 (Sep 15-19): focused four-part reassembly window, all twelve views; the "
        "first minute is the human-reviewed comparison window."
    ),
)


@lru_cache(maxsize=8)
def _load_registry(path: Path) -> Assembly101RecordingRegistry:
    return Assembly101RecordingRegistry.model_validate_json(path.read_text(encoding="utf-8"))


def load_registry(
    repository_root: Path | None = None, path: Path = RECORDINGS_CONFIG
) -> Assembly101RecordingRegistry:
    root = (repository_root or Path.cwd()).resolve()
    return _load_registry((root / path).resolve())


def get_recording(key: str | None, repository_root: Path | None = None) -> Assembly101Recording:
    """Resolve a CLI `--recording` value: None or the recording-1 label/id needs no file."""
    if key is None or key in (RECORDING_1.label, RECORDING_1.recording_id):
        return RECORDING_1
    return load_registry(repository_root).get(key)
