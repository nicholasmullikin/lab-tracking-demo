"""`battle-finebio-detect`: frame selection, interpolation, manifest, GPU guard settings.

The default tier needs no data and no detector venv; the `real_data` test reproduces the
preflight's T5 rows on the CPU venv and diffs them against the recorded JSONL.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import require_artifact

from battle import finebio_detect as fd
from battle import gpu_guard

REPO = Path(__file__).resolve().parents[1]
PREFLIGHT = REPO / "runs" / "preflight-finebio-20260924" / "detections"


def det(class_name: str, score: float, box: list[float], class_id: int = 0) -> dict:
    return {"class": class_name, "class_id": class_id, "score": score, "box_xyxy_px": box}


def row(view: str, frame: int, detections: list[dict]) -> dict:
    return fd.detection_row(view, frame, (1080, 1920), detections)


# --------------------------------------------------------------------------------------------
# frames and views


def test_parse_views_canonical_order_and_rejects_unknown() -> None:
    assert fd.parse_views("T5,fpv,T1") == ["fpv", "T1", "T5"]
    assert fd.parse_views("fpv,fpv") == ["fpv"]
    with pytest.raises(ValueError, match="unknown view"):
        fd.parse_views("fpv,T6")
    with pytest.raises(ValueError, match="no views"):
        fd.parse_views(" , ")


def test_parse_frame_spec_ints_ranges_and_steps() -> None:
    assert fd.parse_frame_spec("5") == [5]
    assert fd.parse_frame_spec("1798-1802") == [1798, 1799, 1800, 1801, 1802]
    assert fd.parse_frame_spec("1798:1888:30") == [1798, 1828, 1858]
    # The preflight set: 60 consecutive frames plus every 30th over 60-80 s, de-duplicated.
    frames = fd.parse_frame_spec("1798-1857,1798:2398:30")
    assert len(frames) == 78 and frames[0] == 1798 and frames[-1] == 2368
    assert frames == sorted(set(frames))
    with pytest.raises(ValueError):
        fd.parse_frame_spec("10-5")
    with pytest.raises(ValueError):
        fd.parse_frame_spec("1:10")


def test_strided_frames_keep_the_last_window_frame() -> None:
    assert fd.strided_frames(600, 10, 1) == list(range(600, 610))
    assert fd.strided_frames(600, 10, 5) == [600, 605, 609]
    assert fd.strided_frames(600, 11, 5) == [600, 605, 610]
    with pytest.raises(ValueError):
        fd.strided_frames(0, 0, 1)


def test_frames_per_view_coverage() -> None:
    views = ["fpv", "T1"]
    per_view, coverage = fd.frames_per_view(
        views, start=100, count=7, explicit=None, stride_fpv=1, stride_fixed=1
    )
    assert coverage == "full" and per_view["fpv"] == per_view["T1"] == list(range(100, 107))
    per_view, coverage = fd.frames_per_view(
        views, start=100, count=7, explicit=None, stride_fpv=3, stride_fixed=5
    )
    assert coverage == "strided"
    assert per_view["fpv"] == [100, 103, 106] and per_view["T1"] == [100, 105, 106]
    per_view, coverage = fd.frames_per_view(
        views, start=None, count=None, explicit=[7, 3], stride_fpv=1, stride_fixed=1
    )
    assert coverage == "explicit" and per_view["T1"] == [7, 3]
    with pytest.raises(ValueError):
        fd.frames_per_view(
            views, start=None, count=None, explicit=None, stride_fpv=1, stride_fixed=1
        )


def test_video_path_layout() -> None:
    root = Path("/data")
    assert fd.video_path(root, "P03_03_01", "fpv") == Path(
        "/data/finebio_videos_fpv_test/finebio_videos/P03_03_01.mp4"
    )
    assert fd.video_path(root, "P03_03_01", "T4") == Path(
        "/data/finebio_videos_tpv_test/finebio_videos/P03_03_01_T4.mp4"
    )


# --------------------------------------------------------------------------------------------
# interpolation


def test_box_iou() -> None:
    assert fd.box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert fd.box_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)
    assert fd.box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert fd.box_iou([0, 0, 0, 0], [0, 0, 0, 0]) == 0.0


def test_associate_same_class_by_iou_one_to_one() -> None:
    left = [det("plate", 0.9, [0, 0, 100, 100]), det("tube", 0.8, [200, 200, 220, 220])]
    right = [
        det("tube", 0.7, [201, 201, 221, 221]),
        det("plate", 0.85, [2, 2, 102, 102]),
        det("plate", 0.5, [4, 4, 104, 104]),  # second plate: the first takes the best match
    ]
    pairs = fd.associate_detections(left, right, 0.3)
    # Descending IoU: the plate pair (0.92) before the tube pair (0.82); the second plate is
    # left over once the first has taken the best match.
    assert [(i, j) for i, j, _ in pairs] == [(0, 1), (1, 0)]
    assert [round(iou, 2) for _, _, iou in pairs] == [0.92, 0.82]


def test_associate_refuses_other_classes_and_moved_boxes() -> None:
    left = [det("plate", 0.9, [0, 0, 100, 100]), det("tube", 0.8, [0, 0, 20, 20])]
    right = [det("tube", 0.9, [0, 0, 100, 100]), det("plate", 0.9, [500, 500, 600, 600])]
    assert fd.associate_detections(left, right, 0.3) == []


def test_interpolate_between_fills_linear_boxes_and_scores() -> None:
    left = row("T1", 10, [det("plate", 0.8, [0, 0, 100, 100], 24)])
    right = row("T1", 14, [det("plate", 0.4, [40, 0, 140, 100], 24)])
    filled = fd.interpolate_between(left, right, 0.3)
    assert [r["frame_index"] for r in filled] == [11, 12, 13]
    assert all(r["interpolated"] is True and r["source_frames"] == [10, 14] for r in filled)
    assert all(r["view"] == "T1" and r["image_hw"] == [1080, 1920] for r in filled)
    mid = filled[1]["detections"][0]
    assert mid["box_xyxy_px"] == pytest.approx([20, 0, 120, 100])
    assert mid["score"] == pytest.approx(0.6)
    assert mid["class"] == "plate" and mid["class_id"] == 24
    quarter = filled[0]["detections"][0]
    assert quarter["box_xyxy_px"] == pytest.approx([10, 0, 110, 100])


def test_interpolate_never_invents_a_class_absent_on_either_side() -> None:
    left = row("T1", 0, [det("plate", 0.9, [0, 0, 100, 100]), det("tube", 0.9, [0, 0, 10, 10])])
    right = row("T1", 3, [det("plate", 0.9, [0, 0, 100, 100])])
    filled = fd.interpolate_between(left, right, 0.3)
    assert len(filled) == 2
    assert all([d["class"] for d in r["detections"]] == ["plate"] for r in filled)
    # Nothing at all when the two sides share no class.
    right_other = row("T1", 3, [det("pen", 0.9, [0, 0, 10, 10])])
    assert all(r["detections"] == [] for r in fd.interpolate_between(left, right_other, 0.3))
    # Adjacent detected frames leave nothing to fill.
    assert fd.interpolate_between(left, row("T1", 1, []), 0.3) == []


def test_interpolate_rows_is_idempotent_and_keeps_detected_rows_verbatim() -> None:
    a = row("T2", 0, [det("plate", 0.9, [0, 0, 100, 100])])
    b = row("T2", 5, [det("plate", 0.7, [10, 0, 110, 100])])
    c = row("T2", 6, [det("plate", 0.7, [10, 0, 110, 100])])
    rows, filled = fd.interpolate_rows([c, a, b], 0.3)
    assert filled == 4
    assert [r["frame_index"] for r in rows] == [0, 1, 2, 3, 4, 5, 6]
    assert rows[0] is a and rows[5] is b and rows[6] is c
    again, filled_again = fd.interpolate_rows(rows, 0.3)
    assert filled_again == 4
    assert [json.dumps(r, sort_keys=True) for r in again] == [
        json.dumps(r, sort_keys=True) for r in rows
    ]
    assert fd.interpolate_rows([], 0.3) == ([], 0)


# --------------------------------------------------------------------------------------------
# manifest


def make_manifest(**overrides) -> fd.DetectManifest:
    payload = {
        "state": "succeeded",
        "trial": "P03_03_01",
        "views": ["fpv", "T1"],
        "model": {"name": "dino", "weights_sha256": "0" * 64},
        "settings": {"device": "cuda", "tf32": False},
        "frames": {"window": {"start": 600, "count": 3600, "end_exclusive": 4200}},
        "detection_coverage": "full",
        "run_id": "run",
        "versions": {"torch": "2.13.0+cu130"},
        "per_frame_seconds": {"fpv": [0.05, 0.06], "T1": [0.05]},
        "gpu": {"peak_vram_bytes": 924844032, "guard": {"schema": gpu_guard.PROVENANCE_SCHEMA}},
    }
    payload.update(overrides)
    return fd.DetectManifest(**payload)


def test_manifest_round_trip_and_validation(tmp_path: Path) -> None:
    manifest = make_manifest()
    fd.write_manifest(tmp_path, manifest)
    text = (tmp_path / "manifest.json").read_text(encoding="utf-8")
    assert json.loads(text)["schema"] == fd.MANIFEST_SCHEMA
    assert fd.read_manifest(tmp_path) == manifest
    assert fd.DetectManifest.from_dict(manifest.as_dict()) == manifest
    with pytest.raises(ValueError, match="state"):
        make_manifest(state="running")
    with pytest.raises(ValueError, match="detection_coverage"):
        make_manifest(detection_coverage="partial")
    with pytest.raises(ValueError, match="model.name"):
        make_manifest(model={"name": "yolo"})
    with pytest.raises(ValueError, match="unknown views"):
        make_manifest(views=["T9"])
    with pytest.raises(ValueError, match="unexpected manifest keys"):
        fd.DetectManifest.from_dict({**manifest.as_dict(), "extra": 1})
    with pytest.raises(ValueError, match="not a"):
        fd.DetectManifest.from_dict({**manifest.as_dict(), "schema": "other/1"})


def test_summarize_and_timing() -> None:
    assert fd.summarize([]) is None
    summary = fd.summarize([0.3, 0.1, 0.2])
    assert summary == {
        "min": 0.1,
        "median": 0.2,
        "mean": pytest.approx(0.2),
        "max": 0.3,
        "count": 3,
    }
    timing = fd._timing({"per_frame_seconds": {"fpv": [0.1, 0.3], "T1": []}, "elapsed_seconds": 9})
    assert timing["per_view"]["fpv"]["mean_ms_per_frame"] == pytest.approx(200.0)
    assert timing["per_view"]["T1"]["mean_ms_per_frame"] is None
    assert timing["mean_ms_per_frame"] == pytest.approx(200.0)


# --------------------------------------------------------------------------------------------
# interpolate subcommand on a synthetic run directory


def test_interpolate_directory_updates_rows_and_manifest(tmp_path: Path) -> None:
    views = ["T1"]
    fd.write_frames_json(tmp_path, trial="P03_03_01", views=views, per_view_frames={"T1": [0, 5]})
    fd.write_jsonl(
        tmp_path / "T1.jsonl",
        [
            row("T1", 0, [det("plate", 0.9, [0, 0, 100, 100]), det("pen", 0.5, [0, 0, 5, 5])]),
            row("T1", 5, [det("plate", 0.8, [5, 0, 105, 100])]),
        ],
    )
    fd.write_manifest(tmp_path, make_manifest(views=views, detection_coverage="strided"))
    assert fd.main(["interpolate", "--directory", str(tmp_path)]) == 0
    rows = fd.load_jsonl(tmp_path / "T1.jsonl")
    assert [r["frame_index"] for r in rows] == [0, 1, 2, 3, 4, 5]
    assert [r["interpolated"] for r in rows] == [False, True, True, True, True, False]
    assert all(len(r["detections"]) == 1 for r in rows[1:5])  # the pen is never invented
    manifest = fd.read_manifest(tmp_path)
    assert manifest.detection_coverage == "strided_interpolated"
    assert manifest.interpolation["filled_rows_per_view"] == {"T1": 4}
    assert manifest.interpolation["iou_threshold"] == fd.INTERPOLATION_IOU
    frames = json.loads((tmp_path / "frames.json").read_text(encoding="utf-8"))
    assert frames["frames_per_view"] == {"T1": [0, 5]} and frames["frames"] == [0, 5]
    assert fd.read_frames_file(tmp_path / "frames.json") == {"T1": [0, 5]}


def test_interpolate_directory_refuses_a_failed_run(tmp_path: Path) -> None:
    fd.write_manifest(tmp_path, make_manifest(state="failed", views=["T1"]))
    with pytest.raises(RuntimeError, match="failed"):
        fd.run_interpolate_directory(tmp_path)


# --------------------------------------------------------------------------------------------
# driver: worker command, environment, blocked run


def run_args(tmp_path: Path, **overrides) -> argparse.Namespace:
    parser = fd.build_parser()
    argv = [
        "run",
        "--trial",
        "P03_03_01",
        "--views",
        "fpv,T5",
        "--start",
        "600",
        "--count",
        "10",
        "--device",
        "cuda",
        "--output",
        str(tmp_path / "detections"),
        "--detector-dir",
        str(tmp_path / "detector"),
        "--raw-root",
        str(tmp_path / "raw"),
        "--repository-root",
        str(tmp_path),
        "--allow-gpu-neighbour",
        "4242",
    ]
    for key, value in overrides.items():
        argv += [f"--{key.replace('_', '-')}", str(value)]
    return parser.parse_args(argv)


def test_worker_command_selects_interpreter_by_device(tmp_path: Path) -> None:
    args = run_args(tmp_path)
    detector = tmp_path / "detector"
    assert fd.interpreter_for(detector, "cuda", None) == detector / ".venv-cuda/bin/python"
    assert fd.interpreter_for(detector, "cpu", None) == detector / ".venv/bin/python"
    assert fd.interpreter_for(detector, "cpu", "/usr/bin/python3") == Path("/usr/bin/python3")
    output = tmp_path / "detections"
    command = fd.worker_command(
        args,
        interpreter=detector / ".venv-cuda/bin/python",
        output=output,
        per_view_frames={"fpv": [600], "T5": [600]},
        videos={"fpv": tmp_path / "a.mp4", "T5": tmp_path / "b.mp4"},
    )
    assert command[0] == str(detector / ".venv-cuda/bin/python")
    assert command[1] == str(Path(fd.__file__).resolve()) and command[2] == "worker"
    assert command[command.index("--device") + 1] == "cuda"
    assert command[command.index("--frames-file") + 1] == str((output / "frames.json").resolve())
    assert command[command.index("--allow-gpu-neighbour") + 1] == "4242"
    assert command[command.index("--gpu-guard-profile") + 1] == gpu_guard.DEFAULT_PROFILE
    assert "--tf32" not in command
    assert json.loads(command[command.index("--videos-json") + 1])["T5"] == str(
        (tmp_path / "b.mp4").resolve()
    )
    # The worker parses what the driver produced.
    worker = fd.build_parser().parse_args(command[2:])
    assert worker.command == "worker" and worker.allow_gpu_neighbour == [4242]
    assert fd.worker_environment("cpu")["CUDA_VISIBLE_DEVICES"] == ""
    assert fd.worker_environment("cuda")["CUDA_VISIBLE_DEVICES"] != ""


def test_run_is_blocked_without_the_detector_interpreter(tmp_path: Path, capsys) -> None:
    args = run_args(tmp_path)
    assert fd.run_run(args) == 2
    manifest = fd.read_manifest(tmp_path / "detections")
    assert manifest.state == "blocked"
    assert ".venv-cuda/bin/python" in (manifest.reason or "")
    assert "FINEBIO_CUDA=1" in (manifest.reason or "")
    assert manifest.detection_coverage == "full"
    assert manifest.frames["per_view"]["T5"] == {
        "stride": 1,
        "detected_count": 10,
        "first": 600,
        "last": 609,
    }
    assert manifest.settings["allowed_gpu_neighbour_pids"] == [4242]
    assert manifest.model["name"] == "dino" and manifest.model["weights_sha256"] is None
    frames = json.loads((tmp_path / "detections" / "frames.json").read_text(encoding="utf-8"))
    assert frames["frames_per_view"]["fpv"] == list(range(600, 610))
    assert "blocked" in capsys.readouterr().err


def test_run_refuses_a_non_empty_output_without_overwrite(tmp_path: Path) -> None:
    (tmp_path / "detections").mkdir()
    (tmp_path / "detections" / "old.txt").write_text("x", encoding="utf-8")
    assert fd.run_run(run_args(tmp_path)) == 2
    assert not (tmp_path / "detections" / "manifest.json").exists()


def test_main_requires_a_window_or_frames(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        fd.main(["run", "--trial", "P03_03_01", "--output", str(tmp_path / "d")])


def test_module_imports_without_the_battle_package(tmp_path: Path) -> None:
    """The worker path: the file run by an interpreter that has no `battle` installed."""
    script = (
        "import sys, runpy; sys.argv = ['finebio_detect.py', '--help']\n"
        f"runpy.run_path({str(Path(fd.__file__).resolve())!r}, run_name='__main__')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-S", "-c", script],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(tmp_path),
        env={"PATH": "/usr/bin:/bin"},
    )
    assert completed.returncode == 0, completed.stderr
    assert "worker" in completed.stdout


# --------------------------------------------------------------------------------------------
# GPU guard settings the detector runs with


def stub_probe(apps: str, cmdlines: dict[int, list[str]]) -> gpu_guard.GpuProbe:
    def nvidia_smi(arguments):
        if "--query-gpu=index,name,memory.total,memory.used" in arguments:
            return "0, NVIDIA GeForce RTX 5070 Ti, 16303, 2716\n"
        return apps

    return gpu_guard.GpuProbe(
        nvidia_smi=nvidia_smi, cmdline=lambda pid: cmdlines.get(pid), ppid=lambda pid: 1
    )


def test_detector_guard_tolerates_the_game_and_refuses_another_worker() -> None:
    game = stub_probe(
        "7478, /usr/bin/kwin_wayland, 144\n9001, C:\\\\game\\\\SYNTHETIK.exe, 1400\n",
        {
            7478: ["/usr/bin/kwin_wayland"],
            9001: ["Z:\\steamapps\\common\\Synthetik\\SYNTHETIK.exe"],
        },
    )
    decision = gpu_guard.evaluate(
        mode="vram", profile=gpu_guard.DEFAULT_PROFILE, probe=game, own_pid=123456
    )
    assert decision.accepted, decision.reasons
    assert {n.neighbour_class for n in decision.neighbours} == {"known_benign"}
    assert decision.headroom["required_mib"] == 6144  # 4 GiB x 1.5: the plan's >= 6 GB rule
    worker = stub_probe(
        "9002, python, 2600\n",
        {9002: ["/home/nick/.pyenv/versions/muggled_sam/bin/python", "muggled_worker.py"]},
    )
    refused = gpu_guard.evaluate(mode="vram", profile=gpu_guard.DEFAULT_PROFILE, probe=worker)
    assert not refused.accepted and "another Battle GPU worker" in refused.reason
    allowed = gpu_guard.evaluate(
        mode="vram", profile=gpu_guard.DEFAULT_PROFILE, probe=worker, allowed_pids=[9002]
    )
    assert allowed.accepted


# --------------------------------------------------------------------------------------------
# real data: the preflight T5 rows reproduced on the CPU venv


@pytest.mark.real_data
@pytest.mark.slow
def test_cpu_reproduces_preflight_t5_rows(tmp_path: Path) -> None:
    reference_path = require_artifact(PREFLIGHT / "T5.jsonl")
    require_artifact(fd.video_path(REPO / fd.DEFAULT_RAW_ROOT, "P03_01_01", "T5"))
    require_artifact(fd.DEFAULT_DETECTOR_DIR / fd.INTERPRETERS["cpu"])
    output = tmp_path / "detections"
    code = fd.main(
        [
            "run",
            "--trial",
            "P03_01_01",
            "--views",
            "T5",
            "--start",
            "1798",
            "--count",
            "5",
            "--device",
            "cpu",
            "--output",
            str(output),
            "--repository-root",
            str(REPO),
        ]
    )
    manifest = fd.read_manifest(output)
    assert code == 0 and manifest.state == "succeeded", manifest.reason
    assert manifest.detection_coverage == "full"
    assert manifest.versions["cuda_available"] is False
    reference = {r["frame_index"]: r for r in fd.load_jsonl(reference_path)}
    produced = {r["frame_index"]: r for r in fd.load_jsonl(output / "T5.jsonl")}
    assert sorted(produced) == list(range(1798, 1803))
    for frame in produced:
        ref = sorted(reference[frame]["detections"], key=lambda d: (-d["score"], d["class"]))
        got = sorted(produced[frame]["detections"], key=lambda d: (-d["score"], d["class"]))
        assert produced[frame]["image_hw"] == reference[frame]["image_hw"] == [1080, 1920]
        assert len(ref) == len(got)
        for a, b in zip(ref, got, strict=True):
            assert a["class"] == b["class"]
            assert b["class_id"] == fd.FINEBIO_CLASSES.index(a["class"])
            assert a["score"] == pytest.approx(b["score"], abs=1e-3)
            assert a["box_xyxy_px"] == pytest.approx(b["box_xyxy_px"], abs=1e-3)
