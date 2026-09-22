"""Overnight queue runner: serial execution, timeouts, GPU gate and stop-on-first-error."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from battle import gpu_guard
from battle import overnight_queue as queue


def _spec(*jobs: tuple[str, list[str], float]) -> queue.QueueSpec:
    return queue.QueueSpec(
        jobs=[
            queue.QueueJob(name=name, argv=argv, timeout_s=timeout, cwd=".")
            for name, argv, timeout in jobs
        ]
    )


def _healthy(_since: str) -> queue.GpuHealth:
    return queue.GpuHealth(ok=True, nvidia_smi_ok=True, detail="fake gpu")


KWIN = (8429, "/usr/bin/kwin_wayland", 144)
CALIBRATION = (2071175, "/home/nick/.pyenv/versions/muggled_sam/bin/python", 1198)
TRACKER = (2236310, "/home/nick/.pyenv/versions/muggled_sam/bin/python", 3354)
PLAYER = (2158886, "/usr/bin/python3", 389)
CMDLINES = {
    8429: ["/usr/bin/kwin_wayland"],
    2071175: [CALIBRATION[1], "/home/nick/src/battle/src/battle/muggled_calibration_worker.py"],
    2236310: [TRACKER[1], "/x/code-snapshot-259b4f7/src/battle/muggled_worker.py"],
    2158886: ["/usr/bin/python3", "/usr/bin/showtime", "clip.mp4"],
}


def _probe(*snapshots: list[tuple[int, str, int]], used_mib: int = 4200) -> gpu_guard.GpuProbe:
    """A stubbed card; each guard evaluation consumes the next snapshot (the last one repeats)."""
    calls = {"n": 0}

    def nvidia_smi(arguments):
        if arguments[0].startswith("--query-gpu="):
            return f"0, RTX 5070 Ti, 16303, {used_mib}\n"
        apps = snapshots[min(calls["n"], len(snapshots) - 1)]
        calls["n"] += 1
        return "".join(f"{pid}, {name}, {mib}\n" for pid, name, mib in apps)

    return gpu_guard.GpuProbe(
        nvidia_smi=nvidia_smi, cmdline=lambda pid: CMDLINES.get(pid), ppid=lambda _pid: None
    )


def _runner(spec: queue.QueueSpec, tmp_path: Path, **kwargs) -> queue.QueueRunner:
    kwargs.setdefault("gpu_check", _healthy)
    kwargs.setdefault("guard_probe", _probe([KWIN]))
    return queue.QueueRunner(spec, log_path=tmp_path / "q.log", **kwargs)


def _events(log_path: Path) -> list[dict]:
    return [json.loads(line) for line in log_path.read_text().splitlines()]


def _python(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def test_jobs_run_serially_and_every_event_is_logged(tmp_path: Path) -> None:
    marker = tmp_path / "order.txt"
    spec = _spec(
        ("first", _python(f"open({str(marker)!r}, 'a').write('a')"), 30),
        ("second", _python(f"open({str(marker)!r}, 'a').write('b')"), 30),
    )
    runner = _runner(spec, tmp_path)
    results = runner.run()
    assert [r.state for r in results] == ["succeeded", "succeeded"]
    assert marker.read_text() == "ab"
    events = [e["event"] for e in _events(tmp_path / "q.log")]
    assert events == [
        "queue_start",
        "gpu_check",
        "job_start",
        "job_end",
        "gpu_check",
        "job_start",
        "job_end",
        "queue_end",
    ]
    assert _events(tmp_path / "q.log")[-1]["stopped"] is False
    assert (tmp_path / "logs" / "00_first.log").is_file()
    # The guard's provenance rides on every gpu_check event.
    check = _events(tmp_path / "q.log")[1]
    assert check["ok"] is True and check["gpu_guard"]["schema"] == "battle-gpu-guard/1"
    assert check["gpu_guard"]["accepted"] is True
    assert [n["cmdline_basename"] for n in check["gpu_guard"]["neighbours"]] == ["kwin_wayland"]
    assert check["gpu_guard"]["expected_peak"]["profile"] == "unknown"
    assert _events(tmp_path / "q.log")[2]["gpu_profile"] == "unknown"


def test_first_failure_stops_the_queue_and_later_jobs_are_skipped(tmp_path: Path) -> None:
    spec = _spec(
        ("ok", _python("pass"), 30),
        ("bad", _python("raise SystemExit(3)"), 30),
        ("never", _python("pass"), 30),
    )
    results = _runner(spec, tmp_path).run()
    assert [r.state for r in results] == ["succeeded", "failed", "skipped"]
    assert results[1].exit_code == 3
    assert "bad failed" in results[2].detail
    end = _events(tmp_path / "q.log")[-1]
    assert end["stopped"] is True and end["summary"]["skipped"] == 1


def test_continue_on_failure_keeps_going(tmp_path: Path) -> None:
    spec = _spec(("bad", _python("raise SystemExit(1)"), 30), ("after", _python("pass"), 30))
    results = _runner(spec, tmp_path, continue_on_failure=True).run()
    assert [r.state for r in results] == ["failed", "succeeded"]


def test_timeout_kills_the_job_and_its_children(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(queue, "KILL_GRACE_SECONDS", 0.5)
    spec = _spec(
        ("slow", _python("import time; time.sleep(30)"), 0.5), ("next", _python("pass"), 30)
    )
    results = _runner(spec, tmp_path).run()
    assert results[0].state == "timed_out"
    assert results[0].duration_s < 10
    assert results[1].state == "skipped"


def test_timeout_after_the_recheck_still_kills_the_job(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(queue, "KILL_GRACE_SECONDS", 0.5)
    spec = _spec(("slow", _python("import time; time.sleep(30)"), 1.0))
    results = _runner(spec, tmp_path, recheck_after_s=0.2).run()
    assert results[0].state == "timed_out"
    assert results[0].duration_s < 10
    assert "gpu_recheck" in [e["event"] for e in _events(tmp_path / "q.log")]


def test_gpu_check_failure_blocks_the_job_and_stops_the_queue(tmp_path: Path) -> None:
    calls: list[str] = []

    def failing(since: str) -> queue.GpuHealth:
        calls.append(since)
        return queue.GpuHealth(
            ok=False, nvidia_smi_ok=True, kernel_errors=["NVRM: Xid (PCI:0000:01:00): 79"]
        )

    spec = _spec(("gpu-job", _python("pass"), 30), ("next", _python("pass"), 30))
    results = _runner(spec, tmp_path, gpu_check=failing, journal_since="2026-09-18 00:00:00").run()
    assert [r.state for r in results] == ["blocked_gpu", "skipped"]
    assert "Xid" in results[0].detail
    assert calls == ["2026-09-18 00:00:00"]
    assert not (tmp_path / "logs" / "00_gpu-job.log").exists()


def test_gpu_guard_refusal_blocks_the_job_and_is_logged_with_every_neighbour(
    tmp_path: Path,
) -> None:
    # Another tracker worker (from a code snapshot) holds the card; the calibration worker is
    # allowed on the queue command line.
    spec = _spec(("smoke", _python("pass"), 30), ("next", _python("pass"), 30))
    runner = _runner(
        spec,
        tmp_path,
        guard_probe=_probe([KWIN, CALIBRATION, TRACKER]),
        allow_gpu_neighbours=[2071175],
    )
    results = runner.run()
    assert [r.state for r in results] == ["blocked_gpu", "skipped"]
    assert results[0].detail.startswith("gpu guard refused: another Battle GPU worker is running")
    assert "pid 2236310 (muggled_worker.py, 3354 MiB)" in results[0].detail
    events = _events(tmp_path / "q.log")
    assert [e["event"] for e in events] == [
        "queue_start",
        "gpu_check",
        "job_blocked",
        "job_skipped",
        "queue_end",
    ]
    check = events[1]
    assert check["ok"] is False and check["nvidia_smi_ok"] is True
    guard = check["gpu_guard"]
    assert guard["accepted"] is False and guard["mode"] == "vram"
    assert guard["allowed_neighbour_pids"] == [2071175]
    assert [
        (n["pid"], n["class"], n["used_mib"], n["cmdline_basename"]) for n in guard["neighbours"]
    ] == [
        (8429, "known_benign", 144, "kwin_wayland"),
        (2071175, "own_repo_model", 1198, "muggled_calibration_worker.py"),
        (2236310, "own_repo_model", 3354, "muggled_worker.py"),
    ]
    assert guard["headroom"]["headroom_mib"] == 16303 - 4200 - 599 - 1677
    assert not (tmp_path / "logs" / "00_smoke.log").exists()


def test_job_argv_allow_gpu_neighbour_pids_and_profile_reach_the_queue_guard(
    tmp_path: Path,
) -> None:
    job = queue.QueueJob(
        name="reprompt-c10379",
        argv=[*_python("pass"), "--max-side-length", "1920", "--allow-gpu-neighbour", "2071175"],
        timeout_s=30,
        gpu_profile="sam3_1080p",
    )
    spec = queue.QueueSpec(jobs=[job])
    results = _runner(spec, tmp_path, guard_probe=_probe([KWIN, CALIBRATION])).run()
    assert [r.state for r in results] == ["succeeded"]
    check = _events(tmp_path / "q.log")[1]
    assert check["gpu_guard"]["accepted"] is True
    assert check["gpu_guard"]["allowed_neighbour_pids"] == [2071175]
    assert check["gpu_guard"]["expected_peak"] == {
        "bytes": int(3.4 * gpu_guard.GIB),
        "mib": 3482,
        "source": "profile_default",
        "profile": "sam3_1080p",
    }
    assert check["gpu_guard"]["headroom"]["required_mib"] == 5223
    # A job that would not fit is refused on headroom alone, with the arithmetic in the detail.
    tight = queue.QueueSpec(
        jobs=[job.model_copy(update={"expected_peak_vram_bytes": 9 * gpu_guard.GIB})]
    )
    results = _runner(tight, tmp_path / "tight", guard_probe=_probe([KWIN, CALIBRATION])).run()
    assert results[0].state == "blocked_gpu"
    assert (
        "VRAM headroom 11504 MiB (total 16303 - used 4200 + own 0 - reserved 599)"
        in results[0].detail
    )
    assert "required 13824 MiB (1.5 x expected peak 9216 MiB" in results[0].detail


def test_recheck_thirty_seconds_in_logs_a_new_neighbour_and_never_kills(tmp_path: Path) -> None:
    spec = _spec(("smoke", _python("import time; time.sleep(1.5)"), 30))
    # Pre-job: the compositor only. During the job: a video player opened.
    runner = _runner(
        spec, tmp_path, guard_probe=_probe([KWIN], [KWIN, PLAYER]), recheck_after_s=0.3
    )
    results = runner.run()
    assert [r.state for r in results] == ["succeeded"]
    events = _events(tmp_path / "q.log")
    assert [e["event"] for e in events] == [
        "queue_start",
        "gpu_check",
        "job_start",
        "gpu_recheck",
        "job_end",
        "queue_end",
    ]
    recheck = events[3]
    assert recheck["job"] == "smoke" and 0.2 <= recheck["seconds_after_start"] < 1.5
    assert [(n["pid"], n["class"], n["cmdline_basename"]) for n in recheck["new_neighbours"]] == [
        (2158886, "known_benign", "showtime")
    ]
    assert recheck["would_accept"] is True and recheck["reasons"] == []
    assert recheck["gpu_guard"]["schema"] == "battle-gpu-guard/1"
    # A job shorter than the re-check delay is never re-checked.
    quick = _spec(("quick", _python("pass"), 30))
    _runner(
        quick, tmp_path / "quick", guard_probe=_probe([KWIN], [KWIN, PLAYER]), recheck_after_s=5
    ).run()
    assert "gpu_recheck" not in [e["event"] for e in _events(tmp_path / "quick" / "q.log")]


def test_strict_guard_mode_in_the_queue_refuses_by_nvidia_smi_name(tmp_path: Path) -> None:
    spec = _spec(("smoke", _python("pass"), 30))
    results = _runner(
        spec, tmp_path, guard_probe=_probe([KWIN, PLAYER]), gpu_guard_mode="strict"
    ).run()
    assert results[0].state == "blocked_gpu"
    assert "concurrent GPU model process(es) detected" in results[0].detail
    assert _events(tmp_path / "q.log")[1]["gpu_guard"]["mode"] == "strict"
    # Guard off: the event carries no guard record and the same card runs.
    results = _runner(
        spec, tmp_path / "off", guard_probe=_probe([KWIN, PLAYER]), gpu_guard_mode=None
    ).run()
    assert results[0].state == "succeeded"
    assert _events(tmp_path / "off" / "q.log")[1]["gpu_guard"] is None


def test_spawn_failure_is_a_recorded_state_not_a_crash(tmp_path: Path) -> None:
    spec = _spec(("missing", ["/nonexistent/binary-xyz"], 30))
    results = _runner(spec, tmp_path).run()
    assert results[0].state == "spawn_failed"


def test_kernel_error_pattern_matches_nvrm_and_xid_lines_only() -> None:
    lines = [
        "kernel: NVRM: GPU at PCI:0000:01:00: GPU-abc",
        "kernel: NVRM: Xid (PCI:0000:01:00): 79, pid=1, GPU has fallen off the bus.",
        "kernel: usb 1-1: new high-speed USB device",
    ]
    assert [bool(queue.KERNEL_ERROR_PATTERN.search(line)) for line in lines] == [True, True, False]


def test_spec_round_trip_and_interpreter_prefix(tmp_path: Path) -> None:
    path = tmp_path / "jobs.json"
    path.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "name": "mediapipe-c10095",
                        "argv": ["battle-mediapipe-hands", "--view", "static-c10095"],
                        "cwd": "/home/nick/src/battle",
                        "timeout_s": 1800,
                        "interpreter": ["uv", "run"],
                        "env": {"CUDA_VISIBLE_DEVICES": "0"},
                    }
                ],
                "log_path": "runs/x/queue.log",
            }
        )
    )
    spec = queue.load_spec(path)
    assert spec.jobs[0].command == [
        "uv",
        "run",
        "battle-mediapipe-hands",
        "--view",
        "static-c10095",
    ]
    assert spec.log_path == "runs/x/queue.log"
    assert spec.jobs[0].gpu_profile is None and spec.jobs[0].guard_profile == "unknown"
    with pytest.raises(ValueError):
        queue.QueueJob(name="bad name", argv=["x"], timeout_s=1)
    with pytest.raises(ValueError, match="unknown gpu_profile"):
        queue.QueueJob(name="a", argv=["x"], timeout_s=1, gpu_profile="sam9")
    with pytest.raises(ValueError):
        queue.QueueJob(name="a", argv=["x"], timeout_s=1, expected_peak_vram_bytes=0)
    job = queue.QueueJob(name="a", argv=["battle-dam4sam-video"], timeout_s=1)
    assert job.guard_profile == "dam4sam_large"


def test_cli_dry_run_lists_jobs_without_running(tmp_path: Path, capsys) -> None:
    path = tmp_path / "jobs.json"
    path.write_text(json.dumps({"jobs": [{"name": "a", "argv": ["false"], "timeout_s": 5}]}))
    assert (
        queue.main([str(path), "--dry-run", "--gpu-guard", "strict", "--allow-gpu-neighbour", "7"])
        == 0
    )
    assert "00 a: false (cwd ., 5 s, gpu profile unknown)" in capsys.readouterr().out


def test_cli_runs_without_inhibit_and_gpu_check(tmp_path: Path) -> None:
    path = tmp_path / "jobs.json"
    path.write_text(
        json.dumps(
            {"jobs": [{"name": "a", "argv": [sys.executable, "-c", "pass"], "timeout_s": 30}]}
        )
    )
    log = tmp_path / "q.log"
    code = queue.main([str(path), "--no-inhibit", "--no-gpu-check", "--log", str(log)])
    assert code == 0
    assert _events(log)[1]["detail"] == "gpu check disabled"


# --------------------------------------------------------------------------- job writers

# The bytes `egoexo_correspondence.prepare` (indent 1) and `kineo_multiview.prepare` (indent 2)
# wrote before their `queue_job` copies were folded onto `overnight_queue.queue_job`.
EGOEXO_JOBS_TEXT = """{
 "jobs": [
  {
   "name": "lm-eec-egoexo-correspondence",
   "argv": [
    "/repo/scripts/lm_eec_driver.py",
    "--run-dir",
    "/repo/runs/x",
    "--mode",
    "sequence"
   ],
   "cwd": "/home/nick/src/LM-EEC",
   "timeout_s": 1800,
   "env": {
    "CUDA_VISIBLE_DEVICES": "0",
    "PYTHONUNBUFFERED": "1"
   },
   "interpreter": [
    "/home/nick/src/LM-EEC/.venv/bin/python"
   ]
  }
 ]
}
"""
KINEO_JOBS_TEXT = """{
  "jobs": [
    {
      "name": "kineo-multiview-known",
      "argv": [
        "pixi",
        "run",
        "python",
        "/repo/scripts/kineo_multiview_runner.py"
      ],
      "cwd": "/k",
      "timeout_s": 1800,
      "env": {
        "CUDA_VISIBLE_DEVICES": "0"
      },
      "interpreter": []
    }
  ]
}
"""


def test_queue_job_and_write_jobs_reproduce_both_callers_files_byte_for_byte(
    tmp_path: Path,
) -> None:
    egoexo = queue.job_list(
        queue.queue_job(
            name="lm-eec-egoexo-correspondence",
            argv=[
                "/repo/scripts/lm_eec_driver.py",
                "--run-dir",
                "/repo/runs/x",
                "--mode",
                "sequence",
            ],
            cwd=Path("/home/nick/src/LM-EEC"),
            timeout_s=1800,
            env={"CUDA_VISIBLE_DEVICES": "0", "PYTHONUNBUFFERED": "1"},
            interpreter=["/home/nick/src/LM-EEC/.venv/bin/python"],
        )
    )
    kineo = queue.job_list(
        queue.queue_job(
            name="kineo-multiview-known",
            argv=["pixi", "run", "python", "/repo/scripts/kineo_multiview_runner.py"],
            cwd="/k",
            timeout_s=1800,
            env={"CUDA_VISIBLE_DEVICES": "0"},
        )
    )
    spec = queue.write_jobs(tmp_path / "a" / "jobs_egoexo.json", egoexo, indent=1)
    assert spec.jobs[0].command[0] == "/home/nick/src/LM-EEC/.venv/bin/python"
    assert (tmp_path / "a" / "jobs_egoexo.json").read_text(encoding="utf-8") == EGOEXO_JOBS_TEXT
    queue.write_jobs(tmp_path / "jobs_kineo.json", kineo)
    assert (tmp_path / "jobs_kineo.json").read_text(encoding="utf-8") == KINEO_JOBS_TEXT
    assert queue.load_spec(tmp_path / "jobs_kineo.json").jobs[0].cwd == "/k"


def test_queue_job_carries_the_guard_fields_only_when_set_and_write_jobs_validates(
    tmp_path: Path,
) -> None:
    job = queue.queue_job(
        name="smoke",
        argv=["battle-dam4sam-video"],
        cwd=".",
        timeout_s=60,
        interpreter=["uv", "run"],
        expected_peak_vram_bytes=2 * 1024**3,
    )
    assert list(job) == [
        "name",
        "argv",
        "cwd",
        "timeout_s",
        "env",
        "interpreter",
        "expected_peak_vram_bytes",
    ]
    spec = queue.job_list(job, log_path=Path("runs/x/queue.log"))
    assert spec["log_path"] == "runs/x/queue.log"
    assert queue.write_jobs(tmp_path / "jobs.json", spec).jobs[0].expected_peak_vram_bytes == 2**31
    with pytest.raises(ValueError):
        queue.write_jobs(
            tmp_path / "bad.json",
            queue.job_list(queue.queue_job(name="bad name", argv=["x"], cwd=".", timeout_s=1)),
        )
    assert not (tmp_path / "bad.json").exists()


# --------------------------------------------------------------------------- code snapshot


def _git(repo: Path, *parts: str) -> str:
    import subprocess

    return subprocess.run(
        ["git", "-C", str(repo), *parts], check=True, capture_output=True, text=True
    ).stdout.strip()


def _fixture_repository(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "src" / "battle").mkdir(parents=True)
    (repo / "src" / "battle" / "worker.py").write_text("VERSION = 1\n", encoding="utf-8")
    (repo / "README.md").write_text("readme\n", encoding="utf-8")
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "-c", "user.email=t@example.com", "-c", "user.name=t", "add", ".")
    _git(repo, "-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-q", "-m", "one")
    return repo


def test_code_snapshot_archives_the_commit_records_it_and_is_idempotent(tmp_path: Path) -> None:
    repo = _fixture_repository(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    short = _git(repo, "rev-parse", "--short", "HEAD")
    # Edits outside the archived paths (the README, an untracked script) do not make it dirty.
    (repo / "README.md").write_text("edited\n", encoding="utf-8")
    (repo / "test.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    root = tmp_path / "runs" / "pass"
    snapshot = queue.code_snapshot(root, repository_root=repo)
    assert snapshot.directory == (root / f"code-snapshot-{short}").resolve()
    assert snapshot.sha == sha and snapshot.short_sha == short and not snapshot.dirty
    assert not snapshot.reused
    assert snapshot.pythonpath == snapshot.directory / "src"
    assert (snapshot.directory / "src" / "battle" / "worker.py").read_text() == "VERSION = 1\n"
    assert not (snapshot.directory / "README.md").exists()
    record = json.loads((snapshot.directory / "snapshot.json").read_text(encoding="utf-8"))
    assert record["sha"] == sha and record["paths"] == ["src/battle"] and record["dirty"] is False
    assert record["pythonpath"] == str(snapshot.pythonpath)
    again = queue.code_snapshot(root, repository_root=repo)
    assert again.reused and again.directory == snapshot.directory
    # Something else at the target is refused rather than overwritten.
    other = tmp_path / "runs" / "other"
    (other / f"code-snapshot-{short}").mkdir(parents=True)
    (other / f"code-snapshot-{short}" / "stray").write_text("x")
    with pytest.raises(SystemExit, match="not a snapshot"):
        queue.code_snapshot(other, repository_root=repo)


def test_code_snapshot_refuses_a_dirty_archived_tree_unless_allowed(tmp_path: Path) -> None:
    repo = _fixture_repository(tmp_path)
    (repo / "src" / "battle" / "worker.py").write_text("VERSION = 2\n", encoding="utf-8")
    root = tmp_path / "runs" / "pass"
    with pytest.raises(SystemExit, match="uncommitted changes under src/battle"):
        queue.code_snapshot(root, repository_root=repo)
    assert not root.exists()
    snapshot = queue.code_snapshot(root, repository_root=repo, allow_dirty=True)
    assert snapshot.dirty
    # The archive is of the commit, not the working tree.
    assert (snapshot.directory / "src" / "battle" / "worker.py").read_text() == "VERSION = 1\n"
    record = json.loads((snapshot.directory / "snapshot.json").read_text(encoding="utf-8"))
    assert record["dirty"] is True


def test_code_snapshot_cli_prints_the_pythonpath_line(tmp_path: Path, capsys) -> None:
    repo = _fixture_repository(tmp_path)
    root = tmp_path / "runs" / "pass"
    code = queue.code_snapshot_main([str(root), "--repository-root", str(repo), "--commit", "HEAD"])
    out = capsys.readouterr().out
    short = _git(repo, "rev-parse", "--short", "HEAD")
    assert code == 0
    assert f"# every job: env.PYTHONPATH={(root / f'code-snapshot-{short}').resolve()}/src" in out
    assert f"git archive {short} src/battle" in out
