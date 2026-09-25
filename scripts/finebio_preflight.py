#!/usr/bin/env python3
"""FineBio preflight: geometry checks on the shipped six-camera calibration.

Read-only on ``data/``; writes tables, overlays and one Rerun recording under
``runs/preflight-finebio-<date>/``. Battle interpreter (``uv run python``), CPU only.

Subcommands
-----------
mapping    T1..T5 -> camera id and recording day by ArUco marker reprojection, plus an
           independent PnP camera centre per view from the detected markers.
fpv-pose   Shipped first-person pose vs ArUco markers detected in the fpv frames.
rig        From a detector pass (``finebio_preflight_detect.py``): static-object
           triangulation with leave-one-view-out residuals, hand triangulation, a clock
           scan per view, leave-one-view-out for the moving objects, the fpv camera centre
           projected into the fixed views, and the fixed-view plate reprojected into the fpv.
rerun      One ``.rrd`` with the rig (frusta, table, markers, triangulated points, fpv
           trajectory) and per-view frames with marker overlays. No viewer is opened.

Units are the calibration's board units (checkerboard on the bench, read as centimetres:
marker squares come out 6 cm and the fixed cameras 0.8-0.95 m from the bench centre).
Board z points into the table, so "height above the bench" is ``-z``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from battle.multiview_geometry import dlt_triangulate  # noqa: E402

RAW = Path("/home/nick/src/battle/data/raw/finebio")
POSES = RAW / "misc/finebio_camera_poses"
FIXED_VIEWS = ("T1", "T2", "T3", "T4", "T5")
CAMERA_IDS = (1, 2, 3, 4, 6)
STATIC_CLASSES = (
    "centrifuge",
    "pcr_machine",
    "vortex_mixer",
    "trash_can",
    "8_tube_stripes_rack",
    "micro_tube_rack",
    "magnetic_rack",
    "blue_tip_rack",
    "red_tip_rack",
    "yellow_tip_rack",
    "8_channel_tip_rack",
)
MOVING_CLASSES = ("left_hand", "right_hand", "cell_culture_plate", "blue_pipette")
ARUCO_DICT = cv2.aruco.DICT_6X6_50


# --------------------------------------------------------------------------- calibration io


@dataclass(frozen=True)
class Camera:
    name: str
    K: np.ndarray  # (3, 3) for the shipped video resolution
    dist: np.ndarray  # (5,)
    rvec: np.ndarray  # (3,) world -> camera
    tvec: np.ndarray  # (3,)
    size: tuple[int, int]  # (w, h)

    @property
    def R(self) -> np.ndarray:
        return cv2.Rodrigues(self.rvec.reshape(3, 1))[0]

    @property
    def centre(self) -> np.ndarray:
        return (-self.R.T @ self.tvec.reshape(3, 1)).ravel()

    @property
    def projection(self) -> np.ndarray:
        return self.K @ np.hstack([self.R, self.tvec.reshape(3, 1)])

    def project(self, points: np.ndarray) -> np.ndarray:
        pts, _ = cv2.projectPoints(
            np.asarray(points, dtype=np.float64).reshape(-1, 1, 3),
            self.rvec.astype(np.float64),
            self.tvec.astype(np.float64),
            self.K,
            self.dist,
        )
        return pts.reshape(-1, 2)

    def undistort(self, pixels: np.ndarray) -> np.ndarray:
        pts = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.undistortPoints(pts, self.K, self.dist, P=self.K).reshape(-1, 2)


def intrinsics(kind: str) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Shipped chessboard intrinsics rescaled from the calibration resolution to the video."""
    if kind == "fpv":
        d = np.load(POSES / "intrinsic_parameters/gopro9_5_wide_4k_43_0.50.npz")
        sx, sy, size = 1920 / 4000, 1440 / 3000, (1920, 1440)
    else:
        d = np.load(POSES / "intrinsic_parameters/gopro9_6_linear_4k_169_0.50.npz")
        sx, sy, size = 1920 / 3840, 1080 / 2160, (1920, 1080)
    K = d["intrinsic_matrix"].astype(np.float64).copy()
    K[0, 0] *= sx
    K[0, 2] *= sx
    K[1, 1] *= sy
    K[1, 2] *= sy
    return K, d["distCoeff"].astype(np.float64).ravel(), size


def days() -> list[str]:
    return sorted(p.name for p in (POSES / "third_person_camera_poses").iterdir() if p.is_dir())


def fixed_camera(day: str, camera_id: int, name: str | None = None) -> Camera:
    d = np.load(POSES / f"third_person_camera_poses/{day}/extrinsics/{camera_id}_board.npz")
    K, dist, size = intrinsics("tpv")
    return Camera(
        name or f"cam{camera_id}",
        K,
        dist,
        d["r"].astype(np.float64).ravel(),
        d["t"].astype(np.float64).ravel(),
        size,
    )


def rig_cameras(
    mapping: dict, *, pnp_over_px: float = 10.0
) -> tuple[dict[str, Camera], dict[str, str]]:
    """Cameras for the five fixed views from a mapping report: the shipped extrinsics where
    they fit the markers within `pnp_over_px`, otherwise the marker-PnP pose (provenance
    recorded per view as ``shipped`` or ``marker_pnp``)."""
    day = mapping["day_decision"]["chosen"]
    cams, provenance = {}, {}
    for v in FIXED_VIEWS:
        info = mapping["views"][v]
        cam = fixed_camera(day, info["best"]["camera_id"], v)
        if info["best"]["median_corner_rms_px"] > pnp_over_px and info.get("pnp_pose"):
            cam = Camera(
                v,
                cam.K,
                cam.dist,
                np.array(info["pnp_pose"]["rvec"]),
                np.array(info["pnp_pose"]["tvec"]),
                cam.size,
            )
            provenance[v] = "marker_pnp"
        else:
            provenance[v] = "shipped"
        cams[v] = cam
    return cams, provenance


def marker_points(day: str) -> np.ndarray:
    pts = np.load(POSES / f"third_person_camera_poses/{day}/params/marker_points.npy")
    return pts.astype(np.float64).reshape(-1, 4, 3)


def fpv_poses(trial: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(POSES / f"first_person_camera_poses/{trial}.npz")
    return (
        d["rets"].astype(bool),
        d["rots"].reshape(-1, 3).astype(np.float64),
        d["trans"].reshape(-1, 3).astype(np.float64),
    )


def fpv_camera(trial: str, frame: int) -> Camera | None:
    rets, rots, trans = fpv_poses(trial)
    if frame >= len(rets) or not rets[frame]:
        return None
    K, dist, size = intrinsics("fpv")
    return Camera("fpv", K, dist, rots[frame], trans[frame], size)


def video_path(trial: str, view: str) -> Path:
    if view == "fpv":
        return RAW / "finebio_videos_fpv_test/finebio_videos" / f"{trial}.mp4"
    return RAW / "finebio_videos_tpv_test/finebio_videos" / f"{trial}_{view}.mp4"


def read_frame(trial: str, view: str, frame: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path(trial, view)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read {view} frame {frame} of {trial}")
    return img


# --------------------------------------------------------------------------- markers


def detect_markers(img: np.ndarray) -> dict[int, np.ndarray]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(ARUCO_DICT), cv2.aruco.DetectorParameters()
    )
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2).astype(np.float64) for i, c in zip(ids.ravel(), corners)}


def match_markers(
    detected: dict[int, np.ndarray], projected: np.ndarray
) -> list[tuple[int, int, float, np.ndarray]]:
    """For each detected marker: (aruco id, index into projected markers, corner RMS px,
    projected corners in the matching corner order). Nearest projected centroid wins; the
    corner order is the cyclic shift with the smallest RMS."""
    out = []
    centroids = projected.mean(axis=1)
    for marker_id, corners in detected.items():
        j = int(np.argmin(np.linalg.norm(centroids - corners.mean(axis=0), axis=1)))
        best = None
        for shift in range(4):
            rolled = np.roll(projected[j], shift, axis=0)
            rms = float(np.sqrt(((rolled - corners) ** 2).sum(axis=1).mean()))
            if best is None or rms < best[0]:
                best = (rms, rolled)
        out.append((marker_id, j, best[0], best[1]))
    return out


def draw_markers(
    img: np.ndarray, detected: dict[int, np.ndarray], matches, label: str
) -> np.ndarray:
    vis = img.copy()
    for marker_id, corners in detected.items():
        cv2.polylines(vis, [corners.astype(np.int32)], True, (0, 0, 255), 2)
        cv2.putText(vis, f"id{marker_id}", tuple(corners[0].astype(int)), 0, 0.7, (0, 0, 255), 2)
    for marker_id, _, rms, proj in matches:
        cv2.polylines(vis, [proj.astype(np.int32)], True, (0, 255, 0), 2)
        cv2.putText(
            vis, f"{rms:.1f}px", tuple((proj[2] + (4, 18)).astype(int)), 0, 0.7, (0, 255, 0), 2
        )
    cv2.putText(vis, label, (10, 30), 0, 0.9, (255, 255, 0), 2)
    return vis


# --------------------------------------------------------------------------- mapping


def cmd_mapping(args: argparse.Namespace) -> int:
    out = args.output / "mapping"
    out.mkdir(parents=True, exist_ok=True)
    seconds = [float(s) for s in args.seconds.split(",")]
    all_days = days()
    report: dict = {"trial": args.trial, "seconds": seconds, "views": {}}
    lines = [
        f"# Camera mapping, {args.trial}",
        "",
        f"Frames at {seconds} s; ArUco DICT_6X6_50; residual = median corner RMS over "
        "detected markers, best cyclic corner order.",
        "",
    ]
    day_votes: dict[str, float] = defaultdict(float)
    for view in FIXED_VIEWS:
        detections = []
        for s in seconds:
            frame = int(round(s * 30000 / 1001))
            img = read_frame(args.trial, view, frame)
            detections.append((frame, img, detect_markers(img)))
        table = {}
        for day in all_days:
            mp = marker_points(day)
            for cam_id in CAMERA_IDS:
                cam = fixed_camera(day, cam_id)
                rms_all = []
                for _, _, det in detections:
                    if not det:
                        continue
                    proj = cam.project(mp.reshape(-1, 3)).reshape(-1, 4, 2)
                    rms_all.extend(m[2] for m in match_markers(det, proj))
                table[(day, cam_id)] = float(np.median(rms_all)) if rms_all else float("nan")
        ranked = sorted(table.items(), key=lambda kv: kv[1])
        (best_day, best_cam), best_rms = ranked[0]
        runner = next(((d, c), r) for (d, c), r in ranked if c != best_cam)
        same_cam_other_day = [
            ((d, c), r) for (d, c), r in ranked if c == best_cam and d != best_day
        ]
        # Independent PnP from detected corners against the best day's marker points
        mp = marker_points(best_day)
        cam = fixed_camera(best_day, best_cam)
        obj, imgp = [], []
        for _, _, det in detections:
            proj = cam.project(mp.reshape(-1, 3)).reshape(-1, 4, 2)
            for marker_id, j, rms, rolled in match_markers(det, proj):
                # recover the corner order shift used for `rolled` to align 3D corners
                shift = next(
                    k for k in range(4) if np.allclose(np.roll(proj[j], k, axis=0), rolled)
                )
                obj.append(np.roll(mp[j], shift, axis=0))
                imgp.append(det[marker_id])
        pnp_centre, pnp_rms, pnp_pose = None, None, None
        if obj:
            obj_a = np.concatenate(obj).reshape(-1, 1, 3)
            img_a = np.concatenate(imgp).reshape(-1, 1, 2)
            ok, rvec, tvec = cv2.solvePnP(
                obj_a, img_a, cam.K, cam.dist, flags=cv2.SOLVEPNP_ITERATIVE
            )
            if ok:
                pnp = Camera("pnp", cam.K, cam.dist, rvec.ravel(), tvec.ravel(), cam.size)
                pnp_centre = pnp.centre
                pnp_rms = float(
                    np.sqrt(
                        ((pnp.project(obj_a.reshape(-1, 3)) - img_a.reshape(-1, 2)) ** 2)
                        .sum(1)
                        .mean()
                    )
                )
                pnp_pose = {
                    "rvec": rvec.ravel().tolist(),
                    "tvec": tvec.ravel().tolist(),
                    "corner_count": int(obj_a.shape[0]),
                }
        n_markers = [len(d) for _, _, d in detections]
        report["views"][view] = {
            "best": {"day": best_day, "camera_id": best_cam, "median_corner_rms_px": best_rms},
            "runner_up_other_camera": {
                "day": runner[0][0],
                "camera_id": runner[0][1],
                "median_corner_rms_px": runner[1],
            },
            "same_camera_other_days": [
                {"day": d, "median_corner_rms_px": r} for (d, _), r in same_cam_other_day
            ],
            "markers_detected_per_frame": n_markers,
            "shipped_centre_cm": cam.centre.round(2).tolist(),
            "pnp_centre_cm": None if pnp_centre is None else pnp_centre.round(2).tolist(),
            "pnp_vs_shipped_cm": None
            if pnp_centre is None
            else float(np.linalg.norm(pnp_centre - cam.centre)),
            "pnp_corner_rms_px": pnp_rms,
            "pnp_pose": pnp_pose,
            "table": {f"{d}/cam{c}": r for (d, c), r in ranked},
        }
        for (d, _), r in [((best_day, best_cam), best_rms)] + same_cam_other_day:
            day_votes[d] += 1.0 / max(r, 0.5)
        lines.append(
            f"- **{view}** -> camera **{best_cam}**, day **{best_day}**: {best_rms:.1f} px "
            f"(other camera best: cam{runner[0][1]} {runner[1]:.1f} px; same camera other days: "
            + ", ".join(f"{d} {r:.1f}" for (d, _), r in same_cam_other_day[:3])
            + f"); markers/frame {n_markers}; PnP centre vs shipped "
            + (
                f"{report['views'][view]['pnp_vs_shipped_cm']:.1f} cm (PnP RMS {pnp_rms:.1f} px)"
                if pnp_centre is not None
                else "n/a"
            )
        )
        # overlay for the first frame
        frame, img, det = detections[0]
        proj = cam.project(mp.reshape(-1, 3)).reshape(-1, 4, 2)
        vis = draw_markers(
            img,
            det,
            match_markers(det, proj),
            f"{args.trial} {view} f{frame}: shipped cam{best_cam}/{best_day} (green) "
            "vs ArUco (red)",
        )
        cv2.imwrite(str(out / f"{view}_markers.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
    best_day_overall = max(day_votes.items(), key=lambda kv: kv[1])[0]
    # Day decision: which day minimises the summed residual over the five best cameras
    day_sum = {}
    for day in all_days:
        vals = [
            report["views"][v]["table"].get(
                f"{day}/cam{report['views'][v]['best']['camera_id']}", np.nan
            )
            for v in FIXED_VIEWS
        ]
        day_sum[day] = float(np.nansum(vals)) if not all(np.isnan(vals)) else float("nan")
    ranked_days = sorted(day_sum.items(), key=lambda kv: kv[1])
    report["day_decision"] = {
        "by_summed_residual": ranked_days,
        "chosen": ranked_days[0][0],
        "weighted_vote": best_day_overall,
    }
    cams = [report["views"][v]["best"]["camera_id"] for v in FIXED_VIEWS]
    report["camera_permutation_ok"] = len(set(cams)) == 5
    lines += [
        "",
        "Day by summed residual over the five chosen cameras: "
        + ", ".join(f"{d} {r:.1f}" for d, r in ranked_days[:4])
        + f" -> **{ranked_days[0][0]}**.",
        "Camera ids form a permutation of (1,2,3,4,6): "
        f"**{report['camera_permutation_ok']}** ({dict(zip(FIXED_VIEWS, cams))}).",
        "",
    ]
    (out / "mapping.json").write_text(json.dumps(report, indent=1))
    (out / "mapping.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


# --------------------------------------------------------------------------- fpv pose


def cmd_fpv_pose(args: argparse.Namespace) -> int:
    out = args.output / "fpv_pose"
    out.mkdir(parents=True, exist_ok=True)
    rets, rots, trans = fpv_poses(args.trial)
    mp = marker_points(args.day)
    K, dist, size = intrinsics("fpv")
    frames = [f for f in range(0, len(rets), args.step) if rets[f]]
    cap = cv2.VideoCapture(str(video_path(args.trial, "fpv")))
    rows = []
    overlays = 0
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            break
        det = detect_markers(img)
        cam = Camera("fpv", K, dist, rots[f], trans[f], size)
        proj = cam.project(mp.reshape(-1, 3)).reshape(-1, 4, 2)
        matches = match_markers(det, proj) if det else []
        rms = [m[2] for m in matches]
        # Independent PnP from the detected markers (>= 2 markers) for a position comparison
        pnp_dist = None
        if len(matches) >= 2:
            obj, imgp = [], []
            for marker_id, j, _, rolled in matches:
                shift = next(
                    k for k in range(4) if np.allclose(np.roll(proj[j], k, axis=0), rolled)
                )
                obj.append(np.roll(mp[j], shift, axis=0))
                imgp.append(det[marker_id])
            ok2, rvec, tvec = cv2.solvePnP(
                np.concatenate(obj).reshape(-1, 1, 3),
                np.concatenate(imgp).reshape(-1, 1, 2),
                K,
                dist,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if ok2:
                pnp = Camera("pnp", K, dist, rvec.ravel(), tvec.ravel(), size)
                pnp_dist = float(np.linalg.norm(pnp.centre - cam.centre))
        rows.append(
            {
                "frame": f,
                "markers_detected": len(det),
                "median_corner_rms_px": float(np.median(rms)) if rms else None,
                "max_corner_rms_px": float(np.max(rms)) if rms else None,
                "pnp_vs_shipped_cm": pnp_dist,
                "height_cm": float(-cam.centre[2]),
            }
        )
        if det and overlays < args.overlays:
            vis = draw_markers(
                img, det, matches, f"{args.trial} fpv f{f}: shipped pose (green) vs ArUco (red)"
            )
            cv2.imwrite(str(out / f"fpv_f{f:05d}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
            overlays += 1
    cap.release()
    with_markers = [r for r in rows if r["median_corner_rms_px"] is not None]
    med = [r["median_corner_rms_px"] for r in with_markers]
    pnp = [r["pnp_vs_shipped_cm"] for r in rows if r["pnp_vs_shipped_cm"] is not None]
    summary = {
        "trial": args.trial,
        "day": args.day,
        "frames_checked": len(rows),
        "frames_with_markers": len(with_markers),
        "corner_rms_px": {
            "median": float(np.median(med)),
            "p90": float(np.percentile(med, 90)),
            "max": float(np.max(med)),
        }
        if med
        else None,
        "frames_over_20px": int(sum(m > 20 for m in med)),
        "frames_over_50px": int(sum(m > 50 for m in med)),
        "pnp_vs_shipped_cm": {
            "median": float(np.median(pnp)),
            "p90": float(np.percentile(pnp, 90)),
            "max": float(np.max(pnp)),
            "n": len(pnp),
        }
        if pnp
        else None,
        "rows": rows,
    }
    (out / "fpv_pose.json").write_text(json.dumps(summary, indent=1))
    msg = (
        (
            f"fpv pose vs ArUco ({args.trial}, day {args.day}): "
            f"{len(with_markers)}/{len(rows)} checked frames had markers; "
            f"corner RMS median {summary['corner_rms_px']['median']:.1f} px, "
            f"p90 {summary['corner_rms_px']['p90']:.1f}, "
            f"max {summary['corner_rms_px']['max']:.1f}; "
            f">20 px on {summary['frames_over_20px']}, >50 px on {summary['frames_over_50px']}; "
            + (
                "marker-PnP centre vs shipped: "
                f"median {summary['pnp_vs_shipped_cm']['median']:.1f} cm, "
                f"p90 {summary['pnp_vs_shipped_cm']['p90']:.1f}, "
                f"max {summary['pnp_vs_shipped_cm']['max']:.1f} "
                f"(n={summary['pnp_vs_shipped_cm']['n']})"
                if pnp
                else ""
            )
        )
        if med
        else "no markers detected in the fpv frames checked"
    )
    (out / "fpv_pose.md").write_text(msg + "\n")
    print(msg)
    return 0


# --------------------------------------------------------------------------- rig from detections


def load_detections(root: Path, views) -> dict[str, dict[int, list[dict]]]:
    out: dict[str, dict[int, list[dict]]] = {}
    for view in views:
        path = root / f"{view}.jsonl"
        if not path.exists():
            continue
        per_frame = {}
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            per_frame[int(rec["frame_index"])] = rec["detections"]
        out[view] = per_frame
    return out


def top_box(dets: list[dict], cls: str, min_score: float) -> dict | None:
    cands = [d for d in dets if d["class"] == cls and d["score"] >= min_score]
    return max(cands, key=lambda d: d["score"]) if cands else None


def centre(box: list[float]) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


def triangulate(cams: list[Camera], pixels: list[np.ndarray]) -> np.ndarray:
    pts = np.stack([c.undistort(p).reshape(2) for c, p in zip(cams, pixels)])[:, None, :]
    P = np.stack([c.projection for c in cams])
    return dlt_triangulate(pts, P)[0]


def cmd_rig(args: argparse.Namespace) -> int:
    out = args.output / "rig"
    out.mkdir(parents=True, exist_ok=True)
    mapping = json.loads((args.output / "mapping/mapping.json").read_text())
    day = mapping["day_decision"]["chosen"]
    cams, provenance = rig_cameras(mapping)
    dets = load_detections(args.detections, (*FIXED_VIEWS, "fpv"))
    frames_meta = json.loads((args.detections / "frames.json").read_text())
    frames = frames_meta["frames"]
    consecutive = [f for f in frames if f + 1 in frames or f - 1 in frames]
    spaced = frames
    report: dict = {
        "day": day,
        "camera_ids": {v: mapping["views"][v]["best"]["camera_id"] for v in FIXED_VIEWS},
        "camera_provenance": provenance,
    }
    lines = [
        f"# Rig checks, {frames_meta['trial']} (day {day})",
        "",
        "Cameras: "
        + ", ".join(f"{v}=cam{report['camera_ids'][v]} ({provenance[v]})" for v in FIXED_VIEWS),
        "",
    ]

    # ---- static objects: per-view median centre over spaced frames, all-view + LOO
    lines += [
        "## Static objects (median box centre over spaced frames, score >= 0.5)",
        "",
        "| class | views | height above bench (cm) | all-view residual px per view "
        "| LOO residual px per view |",
        "|---|---|---|---|---|",
    ]
    static_rows = []
    for cls in STATIC_CLASSES:
        per_view = {}
        for v in FIXED_VIEWS:
            cs = [
                centre(top_box(dets[v][f], cls, 0.5)["box_xyxy_px"])
                for f in spaced
                if f in dets.get(v, {}) and top_box(dets[v][f], cls, 0.5)
            ]
            if len(cs) >= max(3, len(spaced) // 3):
                per_view[v] = np.median(np.stack(cs), axis=0)
        if len(per_view) < 3:
            continue
        vs = list(per_view)
        X = triangulate([cams[v] for v in vs], [per_view[v] for v in vs])
        resid = {v: float(np.linalg.norm(cams[v].project(X)[0] - per_view[v])) for v in vs}
        loo = {}
        for v in vs:
            others = [u for u in vs if u != v]
            if len(others) < 2:
                continue
            Xo = triangulate([cams[u] for u in others], [per_view[u] for u in others])
            loo[v] = float(np.linalg.norm(cams[v].project(Xo)[0] - per_view[v]))
        static_rows.append(
            {
                "class": cls,
                "views": vs,
                "point_cm": X.round(2).tolist(),
                "height_cm": float(-X[2]),
                "residual_px": resid,
                "loo_px": loo,
            }
        )
        lines.append(
            f"| {cls} | {len(vs)} | {-X[2]:.1f} | "
            + " ".join(f"{v}:{r:.0f}" for v, r in resid.items())
            + " | "
            + " ".join(f"{v}:{r:.0f}" for v, r in loo.items())
            + " |"
        )
    report["static"] = static_rows
    all_loo = [r for row in static_rows for r in row["loo_px"].values()]
    lines += [
        "",
        f"Static LOO residual over {len(all_loo)} (object, view) cells: "
        f"median {np.median(all_loo):.1f} px, p90 {np.percentile(all_loo, 90):.1f} px."
        if all_loo
        else "no static objects triangulated",
        "",
    ]

    # ---- hands per consecutive frame
    lines += ["## Hands per frame (consecutive window, score >= 0.5, >= 3 views)", ""]
    hand_rows = {"left_hand": [], "right_hand": []}
    hand_points = {"left_hand": {}, "right_hand": {}}
    for cls in hand_rows:
        for f in consecutive:
            vs, px = [], []
            for v in FIXED_VIEWS:
                b = top_box(dets[v].get(f, []), cls, 0.5) if v in dets else None
                if b:
                    vs.append(v)
                    px.append(centre(b["box_xyxy_px"]))
            if len(vs) < 3:
                continue
            X = triangulate([cams[v] for v in vs], px)
            resid = {v: float(np.linalg.norm(cams[v].project(X)[0] - p)) for v, p in zip(vs, px)}
            hand_rows[cls].append(
                {"frame": f, "views": vs, "point_cm": X.round(2).tolist(), "residual_px": resid}
            )
            hand_points[cls][f] = X
        res = [r for row in hand_rows[cls] for r in row["residual_px"].values()]
        heights = [-row["point_cm"][2] for row in hand_rows[cls]]
        if res:
            lines.append(
                f"- **{cls}**: {len(hand_rows[cls])}/{len(consecutive)} frames with >= 3 views; "
                f"residual median {np.median(res):.1f} px, p90 {np.percentile(res, 90):.1f}; "
                f"height above bench median {np.median(heights):.1f} cm "
                f"(range {min(heights):.1f}..{max(heights):.1f})"
            )
    report["hands"] = hand_rows
    lines.append("")

    # ---- clock scan per view on moving classes
    lines += [
        "## Clock scan per view (-15..+15 frames; residual of the other views' triangulation "
        "reprojected into the shifted view)",
        "",
        "| view | class | best offset | residual at best (px) | residual at 0 (px) "
        "| residual at +/-3 (px) |",
        "|---|---|---|---|---|---|",
    ]
    clock = {}
    for v in FIXED_VIEWS:
        clock[v] = {}
        for cls in MOVING_CLASSES:
            per_offset = {}
            for k in range(-15, 16):
                res = []
                for f in consecutive:
                    if f + k not in dets.get(v, {}):
                        continue
                    bv = top_box(dets[v][f + k], cls, 0.4)
                    if not bv:
                        continue
                    others, px = [], []
                    for u in FIXED_VIEWS:
                        if u == v:
                            continue
                        b = top_box(dets[u].get(f, []), cls, 0.4)
                        if b:
                            others.append(u)
                            px.append(centre(b["box_xyxy_px"]))
                    if len(others) < 2:
                        continue
                    X = triangulate([cams[u] for u in others], px)
                    res.append(
                        float(np.linalg.norm(cams[v].project(X)[0] - centre(bv["box_xyxy_px"])))
                    )
                if len(res) >= 10:
                    per_offset[k] = float(np.median(res))
            if not per_offset:
                continue
            best_k = min(per_offset, key=per_offset.get)
            clock[v][cls] = {"best_offset": best_k, "residual_px_by_offset": per_offset}
            r3 = [per_offset.get(k) for k in (-3, 3)]
            lines.append(
                f"| {v} | {cls} | {best_k:+d} | {per_offset[best_k]:.1f} | "
                f"{per_offset.get(0, float('nan')):.1f} | "
                + "/".join("n/a" if r is None else f"{r:.1f}" for r in r3)
                + " |"
            )
    report["clock"] = clock
    lines.append("")

    # ---- LOO for moving objects per frame
    lines += [
        "## Leave-one-view-out, moving objects (consecutive window)",
        "",
        "| class | held-out view | frames | centre residual median px | p90 "
        "| inside held-out box |",
        "|---|---|---|---|---|---|",
    ]
    loo_moving = {}
    for cls in ("cell_culture_plate", "blue_pipette", "left_hand", "right_hand"):
        loo_moving[cls] = {}
        for v in FIXED_VIEWS:
            res, inside = [], []
            for f in consecutive:
                bv = top_box(dets[v].get(f, []), cls, 0.4) if v in dets else None
                if not bv:
                    continue
                others, px = [], []
                for u in FIXED_VIEWS:
                    if u == v:
                        continue
                    b = top_box(dets[u].get(f, []), cls, 0.4)
                    if b:
                        others.append(u)
                        px.append(centre(b["box_xyxy_px"]))
                if len(others) < 3:
                    continue
                X = triangulate([cams[u] for u in others], px)
                p = cams[v].project(X)[0]
                res.append(float(np.linalg.norm(p - centre(bv["box_xyxy_px"]))))
                x0, y0, x1, y1 = bv["box_xyxy_px"]
                inside.append(bool(x0 <= p[0] <= x1 and y0 <= p[1] <= y1))
            if res:
                loo_moving[cls][v] = {
                    "frames": len(res),
                    "median_px": float(np.median(res)),
                    "p90_px": float(np.percentile(res, 90)),
                    "inside_fraction": float(np.mean(inside)),
                }
                lines.append(
                    f"| {cls} | {v} | {len(res)} | {np.median(res):.1f} | "
                    f"{np.percentile(res, 90):.1f} | {np.mean(inside):.2f} |"
                )
    report["loo_moving"] = loo_moving
    lines.append("")

    # ---- fpv: camera centre into fixed views; fixed-view plate into the fpv
    lines += ["## First-person camera", ""]
    rets, rots, trans = fpv_poses(frames_meta["trial"])
    K_f, dist_f, size_f = intrinsics("fpv")
    fpv_rows = []
    for f in consecutive:
        if not rets[f]:
            continue
        cam_f = Camera("fpv", K_f, dist_f, rots[f], trans[f], size_f)
        row = {
            "frame": f,
            "fpv_centre_cm": cam_f.centre.round(2).tolist(),
            "centre_in_view_px": {
                v: cams[v].project(cam_f.centre)[0].round(1).tolist() for v in FIXED_VIEWS
            },
        }
        # plate: triangulate from fixed views, project into fpv, compare with fpv detector box
        others, px = [], []
        for u in FIXED_VIEWS:
            b = top_box(dets[u].get(f, []), "cell_culture_plate", 0.4)
            if b:
                others.append(u)
                px.append(centre(b["box_xyxy_px"]))
        bf = top_box(dets.get("fpv", {}).get(f, []), "cell_culture_plate", 0.4)
        if len(others) >= 3 and bf:
            X = triangulate([cams[u] for u in others], px)
            p = cam_f.project(X)[0]
            x0, y0, x1, y1 = bf["box_xyxy_px"]
            row["plate_fixed_to_fpv"] = {
                "projected_px": p.round(1).tolist(),
                "fpv_box_centre_px": centre(bf["box_xyxy_px"]).round(1).tolist(),
                "residual_px": float(np.linalg.norm(p - centre(bf["box_xyxy_px"]))),
                "inside_box": bool(x0 <= p[0] <= x1 and y0 <= p[1] <= y1),
                "fixed_views": others,
            }
        for cls in ("left_hand", "right_hand"):
            if f in hand_points[cls]:
                bf2 = top_box(dets.get("fpv", {}).get(f, []), cls, 0.4)
                if bf2:
                    p = cam_f.project(hand_points[cls][f])[0]
                    row[f"{cls}_fixed_to_fpv_px"] = float(
                        np.linalg.norm(p - centre(bf2["box_xyxy_px"]))
                    )
        fpv_rows.append(row)
    report["fpv"] = fpv_rows
    plate_res = [
        r["plate_fixed_to_fpv"]["residual_px"] for r in fpv_rows if "plate_fixed_to_fpv" in r
    ]
    plate_in = [
        r["plate_fixed_to_fpv"]["inside_box"] for r in fpv_rows if "plate_fixed_to_fpv" in r
    ]
    lines.append(f"- fpv pose valid on {len(fpv_rows)}/{len(consecutive)} consecutive frames.")
    if plate_res:
        lines.append(
            "- Plate triangulated from the fixed views and projected into the fpv (1920x1440): "
            f"residual vs the fpv detector box centre median {np.median(plate_res):.0f} px, "
            f"p90 {np.percentile(plate_res, 90):.0f}; inside the fpv box on "
            f"{np.mean(plate_in):.2f} of {len(plate_res)} frames."
        )
    for cls in ("left_hand", "right_hand"):
        hr = [r[f"{cls}_fixed_to_fpv_px"] for r in fpv_rows if f"{cls}_fixed_to_fpv_px" in r]
        if hr:
            lines.append(
                f"- {cls} (fixed-view triangulation -> fpv): residual median "
                f"{np.median(hr):.0f} px, p90 {np.percentile(hr, 90):.0f} over {len(hr)} frames."
            )
    # overlays: fpv camera centre drawn on T1/T3/T4 at the first valid consecutive frame
    if fpv_rows:
        f0 = fpv_rows[0]["frame"]
        for v in ("T1", "T3", "T4"):
            img = read_frame(frames_meta["trial"], v, f0)
            p = np.array(fpv_rows[0]["centre_in_view_px"][v])
            cv2.circle(img, tuple(p.astype(int)), 18, (0, 255, 255), 3)
            cv2.putText(
                img, f"fpv camera centre (shipped pose) f{f0}", (10, 30), 0, 0.9, (0, 255, 255), 2
            )
            for cls, col in (("left_hand", (255, 128, 0)), ("right_hand", (0, 128, 255))):
                if f0 in hand_points[cls]:
                    q = cams[v].project(hand_points[cls][f0])[0]
                    cv2.drawMarker(img, tuple(q.astype(int)), col, cv2.MARKER_CROSS, 30, 3)
            for row in static_rows:
                q = cams[v].project(np.array(row["point_cm"]))[0]
                cv2.drawMarker(
                    img, tuple(q.astype(int)), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 20, 2
                )
                cv2.putText(
                    img, row["class"], tuple((q + (8, -8)).astype(int)), 0, 0.5, (0, 255, 0), 1
                )
            cv2.imwrite(str(out / f"{v}_fpv_centre_f{f0}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    (out / "rig.json").write_text(json.dumps(report, indent=1))
    (out / "rig.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


# --------------------------------------------------------------------------- rerun


def cmd_rerun(args: argparse.Namespace) -> int:
    import rerun as rr
    import rerun.blueprint as rrb

    mapping = json.loads((args.output / "mapping/mapping.json").read_text())
    day = mapping["day_decision"]["chosen"]
    cams, _ = rig_cameras(mapping)
    rig = (
        json.loads((args.output / "rig/rig.json").read_text())
        if (args.output / "rig/rig.json").exists()
        else None
    )
    mp = marker_points(day)
    rets, rots, trans = fpv_poses(args.trial)
    K_f, dist_f, size_f = intrinsics("fpv")
    frames = [int(f) for f in args.frames.split(",")]

    rr.init(f"finebio-preflight-{args.trial}", spawn=False)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_DOWN, static=True)
    # bench outline at z=0 (cm), sized to the room-1 bench from the marker and camera layout
    bench = np.array(
        [[-75, -45, 0], [75, -45, 0], [75, 55, 0], [-75, 55, 0], [-75, -45, 0]], dtype=float
    )
    rr.log(
        "world/bench", rr.LineStrips3D([bench], colors=[[120, 120, 120]], radii=0.3), static=True
    )
    rr.log(
        "world/board_origin",
        rr.Points3D(
            [[0, 0, 0]], colors=[[255, 255, 255]], radii=0.8, labels=["checkerboard origin"]
        ),
        static=True,
    )
    rr.log(
        "world/markers",
        rr.LineStrips3D(
            [np.vstack([m, m[:1]]) for m in mp],
            colors=[[255, 60, 60]],
            radii=0.25,
            labels=[f"marker {i}" for i in range(len(mp))],
        ),
        static=True,
    )
    for v, cam in cams.items():
        rr.log(f"world/{v}", rr.Transform3D(translation=cam.centre, mat3x3=cam.R.T), static=True)
        rr.log(
            f"world/{v}",
            rr.Pinhole(
                image_from_camera=cam.K,
                resolution=list(cam.size),
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=15.0,
            ),
            static=True,
        )
    if rig:
        pts = np.array([r["point_cm"] for r in rig["static"]])
        rr.log(
            "world/static_objects",
            rr.Points3D(
                pts, colors=[[80, 220, 80]], radii=1.0, labels=[r["class"] for r in rig["static"]]
            ),
            static=True,
        )
    cap = {v: cv2.VideoCapture(str(video_path(args.trial, v))) for v in (*FIXED_VIEWS, "fpv")}
    fpv_trail = []
    hand_pts = (
        {
            cls: {int(r["frame"]): r["point_cm"] for r in rig["hands"][cls]}
            for cls in ("left_hand", "right_hand")
        }
        if rig
        else {}
    )
    dets = (
        load_detections(args.detections, (*FIXED_VIEWS, "fpv"))
        if args.detections and args.detections.exists()
        else {}
    )
    for f in frames:
        rr.set_time("frame", sequence=f)
        rr.set_time("source_time", duration=f * 1001 / 30000)
        for v in (*FIXED_VIEWS, "fpv"):
            cap[v].set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap[v].read()
            if not ok:
                continue
            ok_enc, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
            entity = f"world/{v}"
            if v == "fpv":
                if not rets[f]:
                    rr.log("world/fpv", rr.Clear(recursive=True))
                    continue
                cam = Camera("fpv", K_f, dist_f, rots[f], trans[f], size_f)
                rr.log("world/fpv", rr.Transform3D(translation=cam.centre, mat3x3=cam.R.T))
                rr.log(
                    "world/fpv",
                    rr.Pinhole(
                        image_from_camera=cam.K,
                        resolution=list(cam.size),
                        camera_xyz=rr.ViewCoordinates.RDF,
                        image_plane_distance=10.0,
                    ),
                )
                fpv_trail.append(cam.centre)
                rr.log(
                    "world/fpv_trail",
                    rr.LineStrips3D([np.array(fpv_trail)], colors=[[255, 200, 0]], radii=0.2),
                )
            else:
                cam = cams[v]
            rr.log(
                f"{entity}/image", rr.EncodedImage(contents=jpg.tobytes(), media_type="image/jpeg")
            )
            det = detect_markers(img)
            proj = cam.project(mp.reshape(-1, 3)).reshape(-1, 4, 2)
            rr.log(
                f"{entity}/markers_projected",
                rr.LineStrips2D(
                    [np.vstack([m, m[:1]]) for m in proj],
                    colors=[[60, 255, 60]],
                    radii=1.5,
                    labels=[f"m{i}" for i in range(len(proj))],
                ),
            )
            if det:
                rr.log(
                    f"{entity}/markers_detected",
                    rr.LineStrips2D(
                        [np.vstack([c, c[:1]]) for c in det.values()],
                        colors=[[255, 60, 60]],
                        radii=1.5,
                        labels=[f"aruco {i}" for i in det],
                    ),
                )
            if rig:
                pts3 = np.array([r["point_cm"] for r in rig["static"]])
                rr.log(
                    f"{entity}/static_reprojected",
                    rr.Points2D(
                        cam.project(pts3),
                        colors=[[80, 220, 80]],
                        radii=4,
                        labels=[r["class"] for r in rig["static"]],
                    ),
                )
                for cls, col in (("left_hand", [255, 128, 0]), ("right_hand", [0, 128, 255])):
                    if f in hand_pts.get(cls, {}):
                        rr.log(
                            f"{entity}/{cls}_triangulated",
                            rr.Points2D(
                                cam.project(np.array(hand_pts[cls][f])),
                                colors=[col],
                                radii=6,
                                labels=[cls],
                            ),
                        )
                    else:
                        rr.log(f"{entity}/{cls}_triangulated", rr.Clear(recursive=False))
                if v != "fpv" and rets[f]:
                    cam_f = Camera("fpv", K_f, dist_f, rots[f], trans[f], size_f)
                    rr.log(
                        f"{entity}/fpv_camera_centre",
                        rr.Points2D(
                            cam.project(cam_f.centre),
                            colors=[[255, 255, 0]],
                            radii=8,
                            labels=["fpv camera"],
                        ),
                    )
            if v in dets and f in dets[v]:
                boxes = [d for d in dets[v][f] if d["score"] >= 0.5]
                if boxes:
                    rr.log(
                        f"{entity}/detector",
                        rr.Boxes2D(
                            array=np.array([d["box_xyxy_px"] for d in boxes]),
                            array_format=rr.Box2DFormat.XYXY,
                            labels=[f"{d['class']} {d['score']:.2f}" for d in boxes],
                            colors=[[200, 200, 255]],
                        ),
                    )
        for cls, col in (("left_hand", [255, 128, 0]), ("right_hand", [0, 128, 255])):
            if f in hand_pts.get(cls, {}):
                rr.log(
                    f"world/{cls}",
                    rr.Points3D([hand_pts[cls][f]], colors=[col], radii=1.5, labels=[cls]),
                )
    for c in cap.values():
        c.release()
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="world", name="Rig (cm, z down)"),
            rrb.Grid(
                *[rrb.Spatial2DView(origin=f"world/{v}", name=v) for v in (*FIXED_VIEWS, "fpv")],
                grid_columns=3,
            ),
            column_shares=[1, 2],
        ),
        collapse_panels=True,
    )
    rr.send_blueprint(blueprint)
    rrd = args.output / "preflight.rrd"
    rr.save(str(rrd))
    print(f"wrote {rrd}")
    return 0


# --------------------------------------------------------------------------- cli


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", type=Path, default=Path("runs/preflight-finebio-20260924"))
    parser.add_argument("--trial", default="P03_01_01")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("mapping")
    p.add_argument("--seconds", default="30,60,90")
    p.set_defaults(func=cmd_mapping)
    p = sub.add_parser("fpv-pose")
    p.add_argument("--day", required=True)
    p.add_argument("--step", type=int, default=50)
    p.add_argument("--overlays", type=int, default=6)
    p.set_defaults(func=cmd_fpv_pose)
    p = sub.add_parser("rig")
    p.add_argument("--detections", type=Path, required=True)
    p.set_defaults(func=cmd_rig)
    p = sub.add_parser("rerun")
    p.add_argument("--detections", type=Path, default=None)
    p.add_argument("--frames", default="1798,1813,1828,1843,1857,1918,2038,2158,2278,2398")
    p.set_defaults(func=cmd_rerun)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
