"""Loader for the committed FineBio preflight fixtures (`tests/fixtures/finebio_preflight`).

Lane C's tracker tests import this: `load_preflight_fixtures()` gives the P03_01_01
observations (detector boxes, SAM3 decode and video-memory rows in raw pixels and raw frame
indices), the trial's camera config and the preflight's reference numbers, all without the
ignored `data/` or `runs/` trees. See the fixture README for provenance and the licence note.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from battle.finebio_cameras import Camera, cameras_from_config
from battle.multiview_schemas import FineBioCameraConfig, FineBioObservation, read_jsonl

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "finebio_preflight"


@dataclass(frozen=True)
class PreflightFixtures:
    observations: tuple[FineBioObservation, ...]
    cameras: FineBioCameraConfig
    rig_reference: dict[str, Any]
    fpv_poses: dict[int, dict[str, Any]]

    @property
    def trial(self) -> str:
        return self.cameras.trial

    def fixed_cameras(self) -> dict[str, Camera]:
        return cameras_from_config(self.cameras)

    def fpv_camera(self, frame_index: int) -> Camera | None:
        """The shipped fpv pose at a raw frame of the window, None when missing or invalid."""
        pose = self.fpv_poses.get(frame_index)
        if not pose or not pose["valid"]:
            return None
        return Camera(
            "fpv",
            np.array(self.cameras.fpv.K, dtype=np.float64),
            np.array(self.cameras.fpv.distortion, dtype=np.float64),
            np.array(pose["rvec"], dtype=np.float64),
            np.array(pose["tvec"], dtype=np.float64),
            tuple(self.cameras.fpv.image_size),
        )

    @property
    def frames(self) -> tuple[int, ...]:
        return tuple(sorted({row.frame_index for row in self.observations}))

    @property
    def consecutive_frames(self) -> tuple[int, ...]:
        return tuple(self.rig_reference["frames"]["consecutive"])

    def rows(
        self,
        *,
        view: str | None = None,
        frame_index: int | None = None,
        source: str | None = None,
        object_class: str | None = None,
        min_detector_score: float = 0.0,
    ) -> tuple[FineBioObservation, ...]:
        return tuple(
            row
            for row in self.observations
            if (view is None or row.view == view)
            and (frame_index is None or row.frame_index == frame_index)
            and (source is None or row.source == source)
            and (object_class is None or row.object_class == object_class)
            and (row.detector_score or 0.0) >= min_detector_score
        )

    def by_frame(
        self, *, source: str = "detector"
    ) -> dict[int, dict[str, tuple[FineBioObservation, ...]]]:
        """``{frame_index: {view: rows}}`` for one source."""
        grouped: dict[int, dict[str, list[FineBioObservation]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for row in self.observations:
            if row.source == source:
                grouped[row.frame_index][row.view].append(row)
        return {
            frame: {view: tuple(rows) for view, rows in views.items()}
            for frame, views in sorted(grouped.items())
        }


def load_preflight_fixtures(directory: Path = FIXTURE_DIR) -> PreflightFixtures:
    poses = json.loads((directory / "fpv_poses.json").read_text(encoding="utf-8"))
    return PreflightFixtures(
        observations=tuple(read_jsonl(directory / "observations.jsonl", FineBioObservation)),
        cameras=FineBioCameraConfig.model_validate_json(
            (directory / "cameras.json").read_text(encoding="utf-8")
        ),
        rig_reference=json.loads((directory / "rig_reference.json").read_text(encoding="utf-8")),
        fpv_poses={int(frame): pose for frame, pose in poses["frames"].items()},
    )
