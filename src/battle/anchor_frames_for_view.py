"""Map the C10379 human review anchors onto another view's proxy frames.

The 13 anchor frames were chosen on the focused C10379 clip, whose analysis frame `p` shows
dataset pose frame `17649 + 2p` (clock rule offset +9). Another view's proxy frame showing the
same instant follows from that view's own clock rule
(`configs/assembly101/clock_rules.json`): `floor((pose - proxy_start_raw_frame -
pose_offset_frames) / 2)`, the half frame rounded *down* so the mapped frame is never later
than the C10379 frame, the same integer-timeline convention as the ego-exo correspondence
(`egoexo_correspondence.mapped_analysis_frame`). The residual (target pose frame minus
source pose frame, 0 or -1) is recorded per frame. Because every clock rule has the same
raw-frame step, the mapping is a constant analysis-frame shift per view.

Output: `configs/qa/first_minute_review_anchors_<view_id>.json` (the `ReviewAnchorConfig`
schema with `view_id` set, the mapped frames, windows and hidden-prompt interval, and a
`frame_mapping` record), a per-view manual-seed target config bound to the view's clip
config so the calibration workspace resolves the view, and the `--timestamps` string the
workspace accepts. Extra frames (Track A's detector-selected list) are appended when the
file exists; otherwise the config says they are pending.

Nothing here is a label: it prepares a labelling session. CC BY-NC 4.0 applies to the frames.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

from .assembly101_clock_offset import CLOCK_RULES_CONFIG, is_ego, load_clock_rules
from .assembly101_pose_schemas import Assembly101ClockRule
from .muggled_smoke import relative_uri, sha256_file
from .multiview_seed_transfer import ALL_STATIC_CONFIG, view_id_for
from .review_anchors import (
    SOURCE_MANUAL_SEED_TARGET_CONFIG,
    SOURCE_VIEW_ID,
    AnchorFrameMapping,
    FrameOrigin,
    ReviewAnchorConfig,
    ReviewAnchorFrame,
    ReviewAnchorProvenance,
    Visibility,
    load_config,
    view_paths,
)
from .schemas import ArtifactFingerprint, MuggledSAMManualSeedTargetConfig

SOURCE_VIEW = "C10379"
E4_VIEW = "HMC_21179183"
E3_VIEW = "HMC_21110305"
E4_CLIP_CONFIG = Path(
    "configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_ego_e4_g2.json"
)
E3_CLIP_CONFIG = Path(
    "configs/clips/assembly101_nusar_9033_four_part_reassembly_focused_ego_g2.json"
)
EGO_MANUAL_SEED_TARGET_CONFIG = Path(
    "configs/muggledsam_ego_four_part_reassembly_focused_manual_seed.json"
)
DEFAULT_VIEWS = ("C10095", "C10115", "C10118", "C10119", "C10390", "C10395", "C10404", E4_VIEW)
DEFAULT_EXTRA_FRAMES_GLOB = "runs/detector-scorecard-20260920/*/proposed_anchor_frames.json"
CONVENTION = (
    "target_frame = floor((source_pose_frame - target.proxy_start_raw_frame - "
    "target.pose_offset_frames) / target.raw_frames_per_proxy_frame); source_pose_frame = "
    "source_rule.pose_frame(source_frame); half frames round down so the mapped frame is never "
    "later than the C10379 frame; residual = target_pose_frame - source_pose_frame"
)


def clip_config_for(view: str) -> Path:
    if view == E4_VIEW:
        return E4_CLIP_CONFIG
    if view == E3_VIEW:
        return E3_CLIP_CONFIG
    if is_ego(view):
        raise ValueError(f"{view} has no four-part clip config; only e3 and e4 were prepared")
    return ALL_STATIC_CONFIG


def mapped_frame(
    source_rule: Assembly101ClockRule, target_rule: Assembly101ClockRule, source_frame: int
) -> tuple[int, int]:
    """(target analysis frame, residual pose frames) for one source analysis frame."""
    source_pose = source_rule.pose_frame(source_frame)
    numerator = source_pose - target_rule.proxy_start_raw_frame - target_rule.pose_offset_frames
    target_frame = numerator // target_rule.raw_frames_per_proxy_frame
    if target_frame < 0:
        raise ValueError(f"source frame {source_frame} precedes the target proxy")
    return target_frame, target_rule.pose_frame(target_frame) - source_pose


def analysis_frame_shift(
    source_rule: Assembly101ClockRule, target_rule: Assembly101ClockRule
) -> int:
    """The constant `target - source` shift the two rules imply (checked at frame 0)."""
    if (
        source_rule.raw_frames_per_proxy_frame != target_rule.raw_frames_per_proxy_frame
        or source_rule.proxy_start_raw_frame != target_rule.proxy_start_raw_frame
    ):
        raise ValueError("clock rules with different steps or starts do not shift by a constant")
    return mapped_frame(source_rule, target_rule, 0)[0]


def read_extra_frames(path: Path) -> list[int]:
    """Accept a bare list, `{"frames": [...]}`, `{"analysis_frame_indices": [...]}`, or a
    list of `{"analysis_frame_index": n}` records; the frames are on the C10379 clock."""
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("frames", "analysis_frame_indices", "proposed_anchor_frames"):
            if key in payload:
                payload = payload[key]
                break
        else:
            raise ValueError(f"{path}: no frame list under frames/analysis_frame_indices")
    if not isinstance(payload, list):
        raise ValueError(f"{path}: expected a list of frames")
    frames: list[int] = []
    for item in payload:
        if isinstance(item, dict):
            item = item.get("analysis_frame_index", item.get("frame"))
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise ValueError(f"{path}: frame entries must be non-negative integers")
        frames.append(item)
    return sorted(set(frames))


def build_view_config(
    source: ReviewAnchorConfig,
    *,
    view: str,
    source_rule: Assembly101ClockRule,
    target_rule: Assembly101ClockRule,
    clip_config: str,
    manual_seed_target_config: str,
    source_config_fingerprint: ArtifactFingerprint,
    clock_rules_fingerprint: ArtifactFingerprint,
    extra_frames: list[int] | None = None,
    extra_frames_source: str | None = None,
    extra_frames_note: str | None = None,
) -> ReviewAnchorConfig:
    """Every source frame, window bound and the hidden interval mapped to `view`'s clock."""
    view_id = view_id_for(view)
    shift = analysis_frame_shift(source_rule, target_rule)
    hidden = source.hidden_prompt_interval
    frames: list[ReviewAnchorFrame] = []
    seen: set[int] = set()

    def add(source_frame: int, origin: FrameOrigin, template: ReviewAnchorFrame | None) -> None:
        target, residual = mapped_frame(source_rule, target_rule, source_frame)
        if target in seen:
            return
        seen.add(target)
        inside = hidden is not None and hidden[0] <= source_frame < hidden[1]
        if template is not None:
            expected = dict(template.expected_visible)
            note = template.note
        else:
            expected = {
                t: ("hidden_prompt" if inside and t == "interior" else "visible")
                for t in source.targets
            }
            expected_typed: dict[str, Visibility] = expected  # type: ignore[assignment]
            expected = expected_typed
            note = (
                "interior may be occluded here; choose hidden or draw the visible surface"
                if inside
                else None
            )
        frames.append(
            ReviewAnchorFrame(
                analysis_frame_index=target,
                proxy_seconds=target / source.analysis_fps,
                source_seconds=source.source_offset_seconds + target / source.analysis_fps,
                expected_visible=expected,
                note=note,
                origin=origin,
                source_analysis_frame_index=source_frame,
                residual_pose_frames=residual,
            )
        )

    for frame in source.frames:
        add(frame.analysis_frame_index, "mapped_anchor", frame)
    for extra in extra_frames or []:
        add(extra, "extra_detector_selected", None)
    frames.sort(key=lambda f: f.analysis_frame_index)
    windows = {name: (low + shift, high + shift) for name, (low, high) in source.windows.items()}
    mapping = AnchorFrameMapping(
        source_config=source_config_fingerprint,
        source_view_id=source.view_id,
        source_view=SOURCE_VIEW,
        target_view=view,
        clock_rules=clock_rules_fingerprint,
        source_pose_offset_frames=source_rule.pose_offset_frames,
        target_pose_offset_frames=target_rule.pose_offset_frames,
        analysis_frame_shift=shift,
        convention=CONVENTION,
        extra_frames_source=extra_frames_source,
        extra_frames_pending=extra_frames_source is None,
        extra_frames_note=extra_frames_note,
    )
    return ReviewAnchorConfig(
        manifest_kind="human_review_anchor_config",
        config_id=f"first-minute-review-anchors-{view.replace('_', '').lower()}",
        clip_config=clip_config,
        view_id=view_id,
        manual_seed_target_config=manual_seed_target_config,
        analysis_fps=source.analysis_fps,
        source_offset_seconds=source.source_offset_seconds,
        targets=source.targets,
        frames=tuple(frames),
        windows=windows,
        hidden_prompt_interval=(None if hidden is None else (hidden[0] + shift, hidden[1] + shift)),
        claim_boundary=source.claim_boundary,
        license=source.license,
        provenance=ReviewAnchorProvenance(
            notes=(
                f"Frames mapped from {source.view_id} through the clock rules by "
                "battle-anchor-frames-for-view; nothing labelled yet."
            )
        ),
        frame_mapping=mapping,
        workspace_timestamps=",".join(f"{f.proxy_seconds:.6f}" for f in frames),
    )


def write_manual_seed_target_config(
    *, template: Path, view_id: str, clip_config: Path, output: Path, repository_root: Path
) -> Path:
    """A copy of the four-part target policy bound to `view_id` and its clip config."""
    source = MuggledSAMManualSeedTargetConfig.model_validate_json(
        (repository_root / template).read_text(encoding="utf-8")
    )
    derived = source.model_copy(
        update={
            "config_id": f"{source.config_id}-{view_id}",
            "base_g2_config": clip_config.as_posix(),
            "view_id": view_id,
        }
    )
    output = repository_root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(derived.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return output


def target_config_path_for(view: str) -> Path:
    view_id = view_id_for(view)
    template = (
        EGO_MANUAL_SEED_TARGET_CONFIG if is_ego(view) else Path(SOURCE_MANUAL_SEED_TARGET_CONFIG)
    )
    return template.with_name(f"{template.stem}_{view_id.replace('-', '_')}.json")


def _fingerprint(path: Path, repository_root: Path) -> ArtifactFingerprint:
    return ArtifactFingerprint(
        uri=relative_uri(path.resolve(), repository_root),
        sha256=sha256_file(path),
        source="measured",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-config", type=Path, default=None)
    parser.add_argument("--clock-rules", type=Path, default=CLOCK_RULES_CONFIG)
    parser.add_argument("--view", action="append", default=None, help="dataset view, e.g. C10119")
    parser.add_argument(
        "--extra-frames",
        default=DEFAULT_EXTRA_FRAMES_GLOB,
        help="glob of JSON frame lists on the C10379 clock (Track A's proposals); optional",
    )
    parser.add_argument("--no-extra-frames", action="store_true")
    args = parser.parse_args()
    root = args.repository_root.resolve()
    source_config_path = (args.source_config or view_paths(SOURCE_VIEW_ID)[0]).resolve()
    source = load_config(source_config_path)
    rules = load_clock_rules(root, args.clock_rules)
    source_rule = rules.views[SOURCE_VIEW].clock_rule
    assert source_rule is not None
    extra_frames: list[int] = []
    extra_source: str | None = None
    extra_note: str | None = None
    if not args.no_extra_frames:
        matches = sorted(glob.glob(str(root / args.extra_frames)))
        if matches:
            for match in matches:
                extra_frames.extend(read_extra_frames(Path(match)))
            extra_frames = sorted(set(extra_frames))
            extra_source = "; ".join(relative_uri(Path(m).resolve(), root) for m in matches)
        else:
            extra_note = (
                f"no file matched {args.extra_frames}; rerun battle-anchor-frames-for-view once "
                "Track A has written its proposed_anchor_frames.json to append those frames"
            )
    clock_fp = _fingerprint(root / args.clock_rules, root)
    source_fp = _fingerprint(source_config_path, root)
    for view in args.view or DEFAULT_VIEWS:
        entry = rules.views.get(view)
        if entry is None or entry.clock_rule is None:
            raise SystemExit(f"{view}: no measured clock rule in {args.clock_rules}")
        clip_config = clip_config_for(view)
        target_config = write_manual_seed_target_config(
            template=EGO_MANUAL_SEED_TARGET_CONFIG
            if is_ego(view)
            else Path(SOURCE_MANUAL_SEED_TARGET_CONFIG),
            view_id=view_id_for(view),
            clip_config=clip_config,
            output=target_config_path_for(view),
            repository_root=root,
        )
        config = build_view_config(
            source,
            view=view,
            source_rule=source_rule,
            target_rule=entry.clock_rule,
            clip_config=clip_config.as_posix(),
            manual_seed_target_config=relative_uri(target_config, root),
            source_config_fingerprint=source_fp,
            clock_rules_fingerprint=clock_fp,
            extra_frames=extra_frames,
            extra_frames_source=extra_source,
            extra_frames_note=extra_note,
        )
        output = root / view_paths(config.view_id)[0]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(config.model_dump_json(indent=2) + "\n", encoding="utf-8")
        assert config.frame_mapping is not None
        mapped = [f for f in config.frames if f.origin == "mapped_anchor"]
        residuals = sorted({f.residual_pose_frames for f in mapped})
        print(
            f"{view} ({config.view_id}): shift {config.frame_mapping.analysis_frame_shift:+d} "
            f"frames (offset {entry.clock_rule.pose_offset_frames:+d} vs "
            f"{source_rule.pose_offset_frames:+d}), residual {residuals} pose frames, "
            f"{len(mapped)} mapped + {len(config.frames) - len(mapped)} extra frames -> {output}"
        )
        print(f"  frames: {[f.analysis_frame_index for f in config.frames]}")
        print(f"  --timestamps {config.workspace_timestamps}")
        print(f"  target config: {target_config}")
    if extra_note:
        print(extra_note)


if __name__ == "__main__":
    main()
