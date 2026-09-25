"""`battle-finebio-confidence`: per-track, per-frame confidence with abstention (`p5-confidence`).

For every `Track3D` row of one tracking arm the five label-free signals the plan names are
read off the arm's own files and combined by rank, the Assembly101 scorecard's idea
(`detector_scorecard.normalized_ranks` / `rank_average`): no signal has units another has,
so each is mapped to its percentile within the arm and the percentiles are averaged.

Signals per (track, frame), each aggregated over the track's support views of that frame:

* `detector_box_iou`: the SAM3 mask's bounding box against the detector box that prompted or
  matched it (`provenance.detector_box_iou` on the observation row; mean over views);
* `residual_px`: the tracker's cross-view reprojection residual (`residuals.jsonl`; median
  over views; lower is better, so it is negated before ranking);
* `sam3_object_score`: the worker's object score (mean over views). Its meaning differs per
  arm (the decoder's IoU prediction in arm (b), the tracker's presence logit in (c)/(d)),
  which is why it is only ever compared by rank *within* one arm;
* `detector_score`: the detector's score of the box on the row (mean over views);
* `support`: the number of views supporting the track this frame.

`confidence` = the mean of the defined signals' normalised ranks, in [0, 1]. **Abstain rule**:
a row abstains when `support < 2`, or when any of the five signals is undefined for the row
(no support view carries it: a boxes-only arm has neither a mask IoU nor a SAM3 score, so every
one of its rows abstains, by construction and on the record), or when its combined rank lies in
the bottom decile of the arm's rows. The reasons are written on the row.

DINO-vs-Deformable-DETR agreement (a same-class DDETR box at IoU >= 0.5 for the track's DINO
box in a view, at DDETR score >= 0.3) is kept as a per-row column and reported against the
combined confidence in the summary, marked **near-redundant**: the detector runs found it fires
on 5-10% of boxes, concentrated where the SAM3 score and the residual are already weakest.

Claim boundary: every signal is model output (the FineBio detector was trained on FineBio's
own bench and cameras; mask-vs-box IoU is agreement between two models); confidence here ranks
rows within an arm and is not a probability of being right. Nothing under `runs/` is committed.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field

from .detector_scorecard import normalized_ranks, rank_average
from .finebio_detect import box_iou
from .multiview_schemas import Track3D, read_jsonl, write_jsonl
from .schemas import VersionedModel

SCHEMA = "battle-finebio-confidence/1"
SIGNALS: tuple[str, ...] = (
    "detector_box_iou",
    "residual_px",
    "sam3_object_score",
    "detector_score",
    "support",
)
# Signals where a smaller value is better are negated before ranking.
LOWER_IS_BETTER: frozenset[str] = frozenset({"residual_px"})
MIN_SUPPORT = 2
BOTTOM_DECILE = 0.10
DDETR_MIN_SCORE = 0.3
DDETR_MATCH_IOU = 0.5
DEFAULT_TRACKS_DIRS = ("tracks-ext", "tracks")
ABSTAIN_RULE = (
    f"abstain when support < {MIN_SUPPORT}, or any of the five signals ({', '.join(SIGNALS)}) "
    "is undefined for the row (no support view carries it), or the combined rank is in the "
    f"bottom {int(BOTTOM_DECILE * 100)}% of the arm's rows"
)
CLAIM_BOUNDARY = (
    "Every signal is model output: the FineBio DINO detector was trained on FineBio's own "
    "objects and cameras, mask-vs-box IoU is agreement between two models, the SAM3 score is "
    "the worker's own. Confidence ranks rows within one arm; it is not a probability of being "
    "right and nothing here is scored against ground truth."
)

Key = tuple[str, int, str]  # (view, frame, slot)


class TrackConfidence(VersionedModel):
    """One track at one frame: the raw signals, their ranks in the arm, the combination."""

    frame_index: int = Field(ge=0)
    track_id: str = Field(min_length=1)
    object_class: str = Field(min_length=1)
    state: str = Field(min_length=1)
    support: int = Field(ge=0)
    signals: dict[str, float | None]
    signal_views: dict[str, int]
    ranks: dict[str, float | None]
    confidence: float | None = Field(default=None, ge=0, le=1)
    abstain: bool
    abstain_reasons: tuple[str, ...] = ()
    # Near-redundant column: fraction of support views whose DINO box has a same-class DDETR
    # partner at IoU >= 0.5 (None when no support view carries a box or no DDETR pass exists).
    ddetr_agreement: float | None = Field(default=None, ge=0, le=1)
    tracker_confidence: float = Field(ge=0, le=1)
    tracker_abstain: bool


# --------------------------------------------------------------------------- inputs


def resolve_tracks_dir(arm_dir: Path, tracks_dir: str | None) -> Path:
    """`--tracks-dir` when given, else `tracks-ext/` when present (the tracker extensions'
    output, same layout), else `tracks/`."""
    if tracks_dir is not None:
        path = Path(tracks_dir)
        return path if path.is_absolute() else arm_dir / path
    for name in DEFAULT_TRACKS_DIRS:
        if (arm_dir / name / "tracks.jsonl").is_file():
            return arm_dir / name
    raise FileNotFoundError(f"no tracks.jsonl under {arm_dir}/{{tracks-ext,tracks}}")


def load_tracks(tracks_dir: Path) -> list[Track3D]:
    return list(read_jsonl(tracks_dir / "tracks.jsonl", Track3D))


def load_residuals(tracks_dir: Path) -> dict[tuple[str, str, int], float]:
    """(track, view, frame) -> residual px from `residuals.jsonl` (plain JSON: 600k rows)."""
    out: dict[tuple[str, str, int], float] = {}
    path = tracks_dir / "residuals.jsonl"
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            out[(row["track_id"], row["view"], int(row["frame_index"]))] = float(row["residual_px"])
    return out


def support_keys(rows: Iterable[Track3D]) -> set[Key]:
    return {(view, r.frame_index, slot) for r in rows for view, slot in r.support_slots.items()}


def load_observation_signals(observations: Path, wanted: set[Key]) -> dict[Key, dict[str, Any]]:
    """The per-observation values the signals need, for the (view, frame, slot) keys the tracks
    used; everything else in the 800k-row file is skipped unparsed beyond `json.loads`."""
    out: dict[Key, dict[str, Any]] = {}
    with observations.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["view"], int(row["frame_index"]), row["slot"])
            if key not in wanted:
                continue
            provenance = row.get("provenance") or {}
            out[key] = {
                "object_class": row["object_class"],
                "detector_score": row.get("detector_score"),
                "sam3_object_score": row.get("sam3_object_score"),
                "detector_box_iou": provenance.get("detector_box_iou"),
                "box": row.get("box_xyxy_px") or row.get("mask_bbox_px"),
                "source": row.get("source"),
            }
    return out


def ddetr_agreement(
    ddetr_dir: Path | None,
    signals: dict[Key, dict[str, Any]],
    *,
    min_score: float = DDETR_MIN_SCORE,
    match_iou: float = DDETR_MATCH_IOU,
) -> dict[Key, bool]:
    """Per used observation: does a same-class Deformable DETR box at `min_score` overlap its
    box at IoU >= `match_iou` on the same view and frame. Streams one DDETR file per view."""
    out: dict[Key, bool] = {}
    if ddetr_dir is None or not Path(ddetr_dir).is_dir():
        return out
    by_view_frame: dict[str, dict[int, list[Key]]] = defaultdict(lambda: defaultdict(list))
    for key, sig in signals.items():
        if sig["box"] is not None:
            by_view_frame[key[0]][key[1]].append(key)
    for view, frames in by_view_frame.items():
        path = Path(ddetr_dir) / f"{view}.jsonl"
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                frame = int(row["frame_index"])
                keys = frames.get(frame)
                if not keys:
                    continue
                by_class: dict[str, list[list[float]]] = defaultdict(list)
                for det in row.get("detections", ()):
                    if float(det["score"]) >= min_score:
                        by_class[det["class"]].append(det["box_xyxy_px"])
                for key in keys:
                    sig = signals[key]
                    boxes = by_class.get(sig["object_class"], ())
                    out[key] = any(box_iou(sig["box"], b) >= match_iou for b in boxes)
    return out


# --------------------------------------------------------------------------- the combination


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _median(values: Sequence[float]) -> float | None:
    return float(np.median(values)) if values else None


def raw_signals(
    row: Track3D,
    signals: dict[Key, dict[str, Any]],
    residuals: dict[tuple[str, str, int], float],
    agreement: dict[Key, bool],
) -> tuple[dict[str, float | None], dict[str, int], float | None]:
    """The five aggregates for one row, how many views carried each, and the DDETR column."""
    ious, scores, dets, res, agree = [], [], [], [], []
    for view, slot in row.support_slots.items():
        key = (view, row.frame_index, slot)
        sig = signals.get(key)
        if sig is not None:
            if sig["detector_box_iou"] is not None:
                ious.append(float(sig["detector_box_iou"]))
            if sig["sam3_object_score"] is not None:
                scores.append(float(sig["sam3_object_score"]))
            if sig["detector_score"] is not None:
                dets.append(float(sig["detector_score"]))
            if key in agreement:
                agree.append(1.0 if agreement[key] else 0.0)
        r = residuals.get((row.track_id, view, row.frame_index))
        if r is None:
            r = row.residual_px.get(view)
        if r is not None:
            res.append(float(r))
    values = {
        "detector_box_iou": _mean(ious),
        "residual_px": _median(res),
        "sam3_object_score": _mean(scores),
        "detector_score": _mean(dets),
        "support": float(len(row.support_views)),
    }
    counts = {
        "detector_box_iou": len(ious),
        "residual_px": len(res),
        "sam3_object_score": len(scores),
        "detector_score": len(dets),
        "support": len(row.support_views),
    }
    return values, counts, _mean(agree)


def combine(
    rows: Sequence[Track3D],
    raw: Sequence[dict[str, float | None]],
    *,
    bottom_decile: float = BOTTOM_DECILE,
    min_support: int = MIN_SUPPORT,
) -> tuple[list[dict[str, float | None]], list[float | None], list[bool], list[tuple[str, ...]]]:
    """Rank every signal within the arm, average the ranks, apply the abstain rule."""
    matrix = {
        name: np.array(
            [np.nan if v[name] is None else float(v[name]) for v in raw], dtype=np.float64
        )
        for name in SIGNALS
    }
    for name in LOWER_IS_BETTER:
        matrix[name] = -matrix[name]
    normalized = {name: normalized_ranks(matrix[name]) for name in SIGNALS}
    combined = rank_average(normalized, list(SIGNALS)) if rows else np.array([])
    defined = combined[np.isfinite(combined)]
    threshold = float(np.percentile(defined, bottom_decile * 100)) if defined.size else None
    ranks_out: list[dict[str, float | None]] = []
    confidence: list[float | None] = []
    abstain: list[bool] = []
    reasons_out: list[tuple[str, ...]] = []
    for index, row in enumerate(rows):
        ranks_out.append(
            {
                name: (
                    None
                    if not np.isfinite(normalized[name][index])
                    else round(float(normalized[name][index]), 4)
                )
                for name in SIGNALS
            }
        )
        value = combined[index] if index < combined.size else np.nan
        reasons: list[str] = []
        if len(row.support_views) < min_support:
            reasons.append(f"support_lt_{min_support}")
        for name in SIGNALS:
            if raw[index][name] is None:
                reasons.append(f"missing:{name}")
        if np.isfinite(value) and threshold is not None and value <= threshold:
            reasons.append("bottom_decile")
        confidence.append(None if not np.isfinite(value) else round(float(value), 4))
        abstain.append(bool(reasons))
        reasons_out.append(tuple(reasons))
    return ranks_out, confidence, abstain, reasons_out


def score_arm(
    rows: Sequence[Track3D],
    signals: dict[Key, dict[str, Any]],
    residuals: dict[tuple[str, str, int], float],
    agreement: dict[Key, bool],
) -> list[TrackConfidence]:
    raw: list[dict[str, float | None]] = []
    counts: list[dict[str, int]] = []
    agree: list[float | None] = []
    for row in rows:
        values, n, a = raw_signals(row, signals, residuals, agreement)
        raw.append(values)
        counts.append(n)
        agree.append(a)
    ranks, confidence, abstain, reasons = combine(rows, raw)
    out = []
    for index, row in enumerate(rows):
        out.append(
            TrackConfidence(
                frame_index=row.frame_index,
                track_id=row.track_id,
                object_class=row.object_class,
                state=row.state,
                support=len(row.support_views),
                signals={
                    k: (None if v is None else round(float(v), 4)) for k, v in raw[index].items()
                },
                signal_views=counts[index],
                ranks=ranks[index],
                confidence=confidence[index],
                abstain=abstain[index],
                abstain_reasons=reasons[index],
                ddetr_agreement=None if agree[index] is None else round(agree[index], 4),
                tracker_confidence=row.confidence,
                tracker_abstain=row.abstain,
            )
        )
    return out


# --------------------------------------------------------------------------- summary


def _mean_or_none(values: Sequence[float]) -> float | None:
    return round(float(np.mean(values)), 4) if values else None


def _dist(values: Sequence[float]) -> dict[str, float | int | None]:
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "median": None, "p10": None, "p90": None}
    return {
        "n": int(arr.size),
        "median": round(float(np.median(arr)), 4),
        "p10": round(float(np.percentile(arr, 10)), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
    }


def summarise(
    rows: Sequence[TrackConfidence], *, arm: str, tracks_dir: Path, ddetr_dir: Path | None
) -> dict[str, Any]:
    per_class: dict[str, Any] = {}
    by_class: dict[str, list[TrackConfidence]] = defaultdict(list)
    for r in rows:
        by_class[r.object_class].append(r)
    for cls, items in sorted(by_class.items()):
        reasons = Counter(reason for r in items for reason in r.abstain_reasons)
        per_class[cls] = {
            "rows": len(items),
            "tracks": len({r.track_id for r in items}),
            "confidence": _dist([r.confidence for r in items]),
            "abstain_fraction": round(float(np.mean([r.abstain for r in items])), 4),
            "abstain_reasons": dict(reasons.most_common()),
            "ddetr_agreement_mean": _mean_or_none(
                [r.ddetr_agreement for r in items if r.ddetr_agreement is not None]
            ),
        }
    reasons_all = Counter(reason for r in rows for reason in r.abstain_reasons)
    defined = [r for r in rows if r.confidence is not None]
    # Rows that carry every signal: there the rule reduces to support and the bottom decile.
    complete = [r for r in rows if all(r.signals.get(name) is not None for name in SIGNALS)]
    # The near-redundancy check: DDETR disagreement against the combined confidence.
    with_agreement = [r for r in defined if r.ddetr_agreement is not None]
    redundancy: dict[str, Any] = {"rows_with_agreement": len(with_agreement)}
    if with_agreement:
        conf = np.array([r.confidence for r in with_agreement], dtype=np.float64)
        agree = np.array([r.ddetr_agreement for r in with_agreement], dtype=np.float64)
        abst = np.array([r.abstain for r in with_agreement])
        disagree = agree < 0.5
        quartiles = np.percentile(conf, [25, 50, 75])
        bins = np.digitize(conf, quartiles)
        redundancy.update(
            {
                "disagreement_fraction": round(float(disagree.mean()), 4),
                "disagreeing_rows_already_abstaining": (
                    round(float(abst[disagree].mean()), 4) if disagree.any() else None
                ),
                "agreement_by_confidence_quartile": [
                    round(float(agree[bins == q].mean()), 4) if (bins == q).any() else None
                    for q in range(4)
                ],
                "confidence_median_when_disagreeing": (
                    round(float(np.median(conf[disagree])), 4) if disagree.any() else None
                ),
                "confidence_median_when_agreeing": (
                    round(float(np.median(conf[~disagree])), 4) if (~disagree).any() else None
                ),
            }
        )
    return {
        "schema": SCHEMA,
        "arm": arm,
        "tracks_dir": tracks_dir.as_posix(),
        "ddetr_dir": None if ddetr_dir is None else Path(ddetr_dir).as_posix(),
        "rows": len(rows),
        "tracks": len({r.track_id for r in rows}),
        "signals": list(SIGNALS),
        "lower_is_better": sorted(LOWER_IS_BETTER),
        "abstain_rule": ABSTAIN_RULE,
        "confidence": _dist([r.confidence for r in rows]),
        "abstain_fraction": round(float(np.mean([r.abstain for r in rows])), 4) if rows else None,
        "abstain_fraction_observed_rows": (
            round(float(np.mean([r.abstain for r in rows if r.state == "observed"])), 4)
            if any(r.state == "observed" for r in rows)
            else None
        ),
        "rows_with_all_signals": len(complete),
        "abstain_fraction_rows_with_all_signals": (
            round(float(np.mean([r.abstain for r in complete])), 4) if complete else None
        ),
        "abstain_reasons": dict(reasons_all.most_common()),
        "per_class": per_class,
        "ddetr_near_redundant": redundancy,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def summary_markdown(summary: dict[str, Any]) -> str:
    conf = summary["confidence"]
    lines = [
        f"# Confidence, arm ({summary['arm']})",
        "",
        f"{summary['rows']} track rows on {summary['tracks']} tracks from "
        f"`{summary['tracks_dir']}`. Signals ranked within the arm: "
        + ", ".join(summary["signals"])
        + f" (lower is better: {', '.join(summary['lower_is_better'])}); confidence = mean of "
        "the defined ranks.",
        "",
        f"**Abstain rule:** {summary['abstain_rule']}.",
        "",
        f"Confidence median {_fmt(conf['median'])} (p10 {_fmt(conf['p10'])}, p90 "
        f"{_fmt(conf['p90'])}); **abstain fraction {_fmt(summary['abstain_fraction'])}** "
        f"(observed rows only {_fmt(summary['abstain_fraction_observed_rows'])}; on the "
        f"{summary['rows_with_all_signals']} rows that carry all five signals "
        f"{_fmt(summary['abstain_fraction_rows_with_all_signals'])}). Reasons: "
        + ", ".join(f"{k} {v}" for k, v in summary["abstain_reasons"].items()),
        "",
        "## Per class",
        "",
        "| class | rows | tracks | confidence median / p10 / p90 | abstain | top reasons | "
        "DDETR agreement |",
        "|---|---|---|---|---|---|---|",
    ]
    for cls, e in summary["per_class"].items():
        c = e["confidence"]
        top = ", ".join(f"{k} {v}" for k, v in list(e["abstain_reasons"].items())[:3])
        lines.append(
            f"| {cls} | {e['rows']} | {e['tracks']} | {_fmt(c['median'])} / {_fmt(c['p10'])} / "
            f"{_fmt(c['p90'])} | {_fmt(e['abstain_fraction'])} | {top} | "
            f"{_fmt(e['ddetr_agreement_mean'])} |"
        )
    red = summary["ddetr_near_redundant"]
    lines += ["", "## DINO vs Deformable DETR agreement (near-redundant column)", ""]
    if red.get("rows_with_agreement"):
        lines += [
            f"Rows with a DDETR check: {red['rows_with_agreement']}; disagreement (no same-class "
            f"DDETR box at IoU >= {DDETR_MATCH_IOU} in the majority of support views) on "
            f"{_fmt(red['disagreement_fraction'])} of them, of which "
            f"{_fmt(red['disagreeing_rows_already_abstaining'])} already abstain under the rule "
            "above. Agreement by confidence quartile (low to high): "
            + " / ".join(_fmt(x) for x in red["agreement_by_confidence_quartile"])
            + "; confidence median when disagreeing "
            f"{_fmt(red['confidence_median_when_disagreeing'])} vs "
            f"{_fmt(red['confidence_median_when_agreeing'])} when agreeing. The column is kept "
            "on every row and not folded into the combination.",
        ]
    else:
        lines.append("No Deformable DETR pass was given (`--ddetr`); the column is empty.")
    lines += ["", f"Claim boundary: {summary['claim_boundary']}", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- cli


def infer_arm_label(arm_dir: Path) -> str:
    name = arm_dir.name
    return name.split("-", 1)[0] if "-" in name and len(name.split("-", 1)[0]) == 1 else name


def run(
    arm_dir: Path,
    output: Path,
    *,
    tracks_dir: str | None = None,
    ddetr_dir: Path | None = None,
    arm: str | None = None,
) -> dict[str, Any]:
    arm_dir = Path(arm_dir)
    tracks_path = resolve_tracks_dir(arm_dir, tracks_dir)
    rows = load_tracks(tracks_path)
    wanted = support_keys(rows)
    signals = load_observation_signals(arm_dir / "observations.jsonl", wanted)
    residuals = load_residuals(tracks_path)
    agreement = ddetr_agreement(ddetr_dir, signals)
    scored = score_arm(rows, signals, residuals, agreement)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(scored, output / "confidence.jsonl", compact=True)
    summary = summarise(
        scored, arm=arm or infer_arm_label(arm_dir), tracks_dir=tracks_path, ddetr_dir=ddetr_dir
    )
    summary["arm_dir"] = arm_dir.as_posix()
    (output / "confidence_summary.json").write_text(
        json.dumps(summary, indent=1) + "\n", encoding="utf-8"
    )
    (output / "confidence.md").write_text(summary_markdown(summary), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--arm-dir", type=Path, required=True, help="one arm's directory")
    parser.add_argument(
        "--detections",
        type=Path,
        default=None,
        help="the DINO detections directory (recorded; the detector scores are read from the "
        "arm's observation rows); --ddetr defaults to its sibling `ddetr`",
    )
    parser.add_argument(
        "--ddetr",
        type=Path,
        default=None,
        help="Deformable DETR detections directory for the agreement column",
    )
    parser.add_argument(
        "--tracks-dir",
        default=None,
        help="tracks directory relative to --arm-dir (default: tracks-ext when present, else "
        "tracks)",
    )
    parser.add_argument("--arm", default=None, help="arm label for the report (default: inferred)")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ddetr = args.ddetr
    if ddetr is None and args.detections is not None:
        sibling = Path(args.detections).resolve().parent / "ddetr"
        ddetr = sibling if sibling.is_dir() else None
    summary = run(
        args.arm_dir, args.output, tracks_dir=args.tracks_dir, ddetr_dir=ddetr, arm=args.arm
    )
    print(
        f"arm {summary['arm']}: {summary['rows']} rows, confidence median "
        f"{_fmt(summary['confidence']['median'])}, abstain {_fmt(summary['abstain_fraction'])} "
        f"-> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
