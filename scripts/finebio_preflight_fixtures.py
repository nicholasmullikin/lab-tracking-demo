#!/usr/bin/env python3
"""Convert the Sep 24 FineBio preflight outputs into the committed test fixtures.

    uv run python scripts/finebio_preflight_fixtures.py \
        --preflight runs/preflight-finebio-20260924 --trial P03_01_01 \
        --output tests/fixtures/finebio_preflight

Writes ``observations.jsonl`` (FineBioObservation rows: detector boxes at score >= 0.3 on the
preflight's 78 raw frames in six views, the SAM3 box-prompt decode rows at encoder side 1280,
and the SAM3.1 video-memory track series in fpv, T2, T4, T5), ``cameras.json`` (the trial's
FineBioCameraConfig, identical to ``configs/finebio/cameras/<trial>.json``) and
``rig_reference.json`` (the preflight numbers a regression test reproduces). Numeric derived
data only: no frame, video, mask or recording is copied (FineBio, non-commercial research).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from battle.finebio_cameras import (
    RAW,
    camera_config_from_mapping,
    fpv_pose_path,
    fpv_poses,
    write_camera_config,
)
from battle.multiview_schemas import FineBioObservation, write_jsonl

VIEWS = ("fpv", "T1", "T2", "T3", "T4", "T5")
IMAGE_SIZE = {"fpv": (1920, 1440), **{v: (1920, 1080) for v in VIEWS[1:]}}
SAM3_VIEWS = ("fpv", "T2", "T4", "T5")
MIN_DETECTOR_SCORE = 0.3
DETECTOR_NAME = "finebio_dino"


def _box(values) -> tuple[float, float, float, float]:
    return tuple(round(float(v), 1) for v in values)


def _score(value) -> float | None:
    return None if value is None else round(float(value), 4)


def load_detections(root: Path) -> dict[str, dict[int, list[dict]]]:
    out: dict[str, dict[int, list[dict]]] = {}
    for view in VIEWS:
        per_frame: dict[int, list[dict]] = {}
        for line in (root / f"{view}.jsonl").read_text().splitlines():
            rec = json.loads(line)
            per_frame[int(rec["frame_index"])] = rec["detections"]
        out[view] = per_frame
    return out


def detector_rows(
    dets: dict[str, dict[int, list[dict]]], rets: np.ndarray
) -> list[FineBioObservation]:
    rows = []
    for view in VIEWS:
        for frame in sorted(dets[view]):
            kept = [d for d in dets[view][frame] if d["score"] >= MIN_DETECTOR_SCORE]
            rank: dict[str, int] = {}
            for det in sorted(kept, key=lambda d: -d["score"]):
                k = rank.get(det["class"], 0)
                rank[det["class"]] = k + 1
                rows.append(
                    FineBioObservation(
                        view=view,
                        frame_index=frame,
                        slot=f"{det['class']}#{k}",
                        object_class=det["class"],
                        detector_score=_score(det["score"]),
                        box_xyxy_px=_box(det["box_xyxy_px"]),
                        pose_valid=bool(rets[frame]) if view == "fpv" else True,
                        source="detector",
                    )
                )
    return rows


def matching_detector_box(dets: list[dict], cls: str, score: float | None):
    if score is None:
        return None
    for det in dets:
        if det["class"] == cls and abs(det["score"] - score) < 1e-6:
            return _box(det["box_xyxy_px"])
    return None


def sam3_rows(
    preflight: Path, dets: dict[str, dict[int, list[dict]]], rets: np.ndarray
) -> tuple[list[FineBioObservation], dict]:
    rows: list[FineBioObservation] = []
    notes: dict = {"decode": {}, "track": {}}
    for sam3_view in SAM3_VIEWS:
        report = json.loads((preflight / f"sam3_{sam3_view}/sam3_preflight.json").read_text())
        seed_frame = int(report["frame"])
        for entry in report["decode"]:
            if entry["encoder_side"] != 1280 or entry.get("mask_bbox_px") is None:
                continue
            view = entry["view"]
            width, height = IMAGE_SIZE[view]
            bbox = _box(entry["mask_bbox_px"])
            rows.append(
                FineBioObservation(
                    view=view,
                    frame_index=seed_frame,
                    slot=f"{entry['class']}#0",
                    object_class=entry["class"],
                    detector_score=_score(entry["detector_score"]),
                    box_xyxy_px=_box(entry["box_px"]),
                    mask_bbox_px=bbox,
                    mask_centroid_px=(
                        round((bbox[0] + bbox[2]) / 2, 1),
                        round((bbox[1] + bbox[3]) / 2, 1),
                    ),
                    mask_area_px=int(entry["mask_area_px"]),
                    sam3_object_score=None,
                    pose_valid=bool(rets[seed_frame]) if view == "fpv" else True,
                    source="sam3_decode",
                    provenance={
                        "encoder_side": 1280,
                        "decoder_iou_pred": _score(entry.get("decoder_iou_pred")),
                        "mask_bbox_iou_vs_prompt": _score(entry.get("mask_bbox_iou_vs_prompt")),
                    },
                )
            )
            notes["decode"].setdefault(view, []).append(entry["class"])
        for view, track in report["track"].items():
            width, height = IMAGE_SIZE[view]
            for cls, series in track["series"].items():
                lost = [r["frame"] for r in series if r.get("mask_bbox_px") is None]
                notes["track"][f"{view}/{cls}"] = {
                    "frames": len(series),
                    "first_frame": series[0]["frame"],
                    "last_frame": series[-1]["frame"],
                    "lost_frames_omitted": lost,
                    "seed_frame": seed_frame,
                }
                for rec in series:
                    if rec.get("mask_bbox_px") is None:
                        continue
                    frame = int(rec["frame"])
                    bbox = _box(rec["mask_bbox_px"])
                    score = rec.get("detector_score")
                    rows.append(
                        FineBioObservation(
                            view=view,
                            frame_index=frame,
                            slot=f"{cls}#0",
                            object_class=cls,
                            detector_score=_score(score),
                            box_xyxy_px=matching_detector_box(
                                dets[view].get(frame, []), cls, score
                            ),
                            mask_bbox_px=bbox,
                            mask_centroid_px=(
                                round((bbox[0] + bbox[2]) / 2, 1),
                                round((bbox[1] + bbox[3]) / 2, 1),
                            ),
                            mask_area_px=int(round(rec["area_frac"] * width * height)),
                            sam3_object_score=round(float(rec["object_score"]), 4),
                            pose_valid=bool(rets[frame]) if view == "fpv" else True,
                            source="sam3_video",
                            provenance={
                                "encoder_side": 1280,
                                "detector_box_iou": _score(rec.get("detector_box_iou")),
                            },
                        )
                    )
    return rows, notes


def _summary(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
    }


def rig_reference(preflight: Path, frames_meta: dict) -> dict:
    mapping = json.loads((preflight / "mapping/mapping.json").read_text())
    rig = json.loads((preflight / "rig/rig.json").read_text())
    fpv_pose = json.loads((preflight / "fpv_pose/fpv_pose.json").read_text())
    frames = frames_meta["frames"]
    consecutive = [f for f in frames if f + 1 in frames or f - 1 in frames]
    views = {}
    for view, info in mapping["views"].items():
        views[view] = {
            "camera_id": info["best"]["camera_id"],
            "day": info["best"]["day"],
            "shipped_median_corner_rms_px": info["best"]["median_corner_rms_px"],
            "pnp_corner_rms_px": info["pnp_corner_rms_px"],
            "pnp_vs_shipped_cm": info["pnp_vs_shipped_cm"],
            "provenance": rig["camera_provenance"][view],
            "markers_detected_per_frame": info["markers_detected_per_frame"],
        }
    static_loo = [r for row in rig["static"] for r in row["loo_px"].values()]
    hands = {}
    for cls, rows in rig["hands"].items():
        residuals = [r for row in rows for r in row["residual_px"].values()]
        hands[cls] = {
            "frames_with_3_views": len(rows),
            "consecutive_frames": len(consecutive),
            "residual_px": _summary(residuals) if residuals else None,
            "height_cm_median": float(np.median([-row["point_cm"][2] for row in rows]))
            if rows
            else None,
        }
    clock = {
        view: {
            cls: {
                "best_offset": entry["best_offset"],
                "residual_px_at_best": entry["residual_px_by_offset"][str(entry["best_offset"])],
                "residual_px_at_0": entry["residual_px_by_offset"].get("0"),
            }
            for cls, entry in per_view.items()
        }
        for view, per_view in rig["clock"].items()
    }
    plate = [r["plate_fixed_to_fpv"] for r in rig["fpv"] if "plate_fixed_to_fpv" in r]
    fpv = {
        "pose_valid_consecutive_frames": len(rig["fpv"]),
        "plate_fixed_to_fpv_px": {
            **_summary([p["residual_px"] for p in plate]),
            "inside_box_fraction": float(np.mean([p["inside_box"] for p in plate])),
        },
    }
    for cls in ("left_hand", "right_hand"):
        values = [r[f"{cls}_fixed_to_fpv_px"] for r in rig["fpv"] if f"{cls}_fixed_to_fpv_px" in r]
        fpv[f"{cls}_fixed_to_fpv_px"] = _summary(values) if values else None
    return {
        "trial": frames_meta["trial"],
        "preflight": "runs/preflight-finebio-20260924 (Sep 24, 2026)",
        "frames": {
            "all": frames,
            "consecutive": consecutive,
            "spaced": frames,
            "note": "the rig used every frame as 'spaced' for the static medians and the "
            "consecutive run for hands, clock, moving LOO and the fpv hand-off",
        },
        "day": {
            "chosen": mapping["day_decision"]["chosen"],
            "by_summed_residual_top3": mapping["day_decision"]["by_summed_residual"][:3],
        },
        "camera_permutation_ok": mapping["camera_permutation_ok"],
        "views": views,
        "static": rig["static"],
        "static_loo_px": _summary(static_loo),
        "hands": hands,
        "clock": clock,
        "loo_moving": rig["loo_moving"],
        "fpv": fpv,
        "fpv_pose": {k: v for k, v in fpv_pose.items() if k != "rows"},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, default=Path("runs/preflight-finebio-20260924"))
    parser.add_argument("--trial", default="P03_01_01")
    parser.add_argument("--output", type=Path, default=Path("tests/fixtures/finebio_preflight"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    frames_meta = json.loads((args.preflight / "detections/frames.json").read_text())
    assert frames_meta["trial"] == args.trial
    rets, rots, trans = fpv_poses(args.trial)
    dets = load_detections(args.preflight / "detections")
    rows = detector_rows(dets, rets)
    sam3, notes = sam3_rows(args.preflight, dets, rets)
    rows += sam3
    rows.sort(key=lambda r: (r.frame_index, VIEWS.index(r.view), r.source, r.slot))
    count = write_jsonl(rows, args.output / "observations.jsonl", compact=True)
    # The shipped per-frame fpv pose for every frame that has an observation, so the fixture
    # triangulates six views without data/ (invalid frames carry only the flag).
    poses = {
        str(frame): {"valid": True, "rvec": rots[frame].tolist(), "tvec": trans[frame].tolist()}
        if rets[frame]
        else {"valid": False}
        for frame in sorted({r.frame_index for r in rows})
    }
    (args.output / "fpv_poses.json").write_text(
        json.dumps(
            {
                "trial": args.trial,
                "source": str(fpv_pose_path(args.trial).relative_to(RAW)),
                "frames": poses,
            },
            indent=None,
            separators=(",", ":"),
        )
        + "\n"
    )

    mapping = json.loads((args.preflight / "mapping/mapping.json").read_text())
    config = camera_config_from_mapping(
        mapping,
        args.trial,
        provenance={
            "source": f"{args.preflight}/mapping/mapping.json (preflight of Sep 24, 2026)",
            "command": f"uv run python scripts/finebio_preflight.py --trial {args.trial} "
            "mapping --seconds 30,60,90",
        },
    )
    write_camera_config(config, args.output / "cameras.json")

    reference = rig_reference(args.preflight, frames_meta)
    reference["observations"] = {
        "rows": count,
        "by_source": {
            source: sum(r.source == source for r in rows)
            for source in ("detector", "sam3_decode", "sam3_video")
        },
        "min_detector_score": MIN_DETECTOR_SCORE,
        "detector": DETECTOR_NAME,
        "detector_run": f"{args.preflight}/detections",
        "sam3": notes,
    }
    (args.output / "rig_reference.json").write_text(json.dumps(reference, indent=1) + "\n")
    print(json.dumps(reference["observations"], indent=1))
    for path in sorted(args.output.iterdir()):
        print(f"{path.stat().st_size:>10} {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
