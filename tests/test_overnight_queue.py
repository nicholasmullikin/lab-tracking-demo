"""Overnight queue runner: serial execution, timeouts, GPU gate and stop-on-first-error."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

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
    runner = queue.QueueRunner(spec, log_path=tmp_path / "q.log", gpu_check=_healthy)
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


def test_first_failure_stops_the_queue_and_later_jobs_are_skipped(tmp_path: Path) -> None:
    spec = _spec(
        ("ok", _python("pass"), 30),
        ("bad", _python("raise SystemExit(3)"), 30),
        ("never", _python("pass"), 30),
    )
    results = queue.QueueRunner(spec, log_path=tmp_path / "q.log", gpu_check=_healthy).run()
    assert [r.state for r in results] == ["succeeded", "failed", "skipped"]
    assert results[1].exit_code == 3
    assert "bad failed" in results[2].detail
    end = _events(tmp_path / "q.log")[-1]
    assert end["stopped"] is True and end["summary"]["skipped"] == 1


def test_continue_on_failure_keeps_going(tmp_path: Path) -> None:
    spec = _spec(("bad", _python("raise SystemExit(1)"), 30), ("after", _python("pass"), 30))
    results = queue.QueueRunner(
        spec, log_path=tmp_path / "q.log", gpu_check=_healthy, continue_on_failure=True
    ).run()
    assert [r.state for r in results] == ["failed", "succeeded"]


def test_timeout_kills_the_job_and_its_children(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(queue, "KILL_GRACE_SECONDS", 0.5)
    spec = _spec(
        ("slow", _python("import time; time.sleep(30)"), 0.5), ("next", _python("pass"), 30)
    )
    results = queue.QueueRunner(spec, log_path=tmp_path / "q.log", gpu_check=_healthy).run()
    assert results[0].state == "timed_out"
    assert results[0].duration_s < 10
    assert results[1].state == "skipped"


def test_gpu_check_failure_blocks_the_job_and_stops_the_queue(tmp_path: Path) -> None:
    calls: list[str] = []

    def failing(since: str) -> queue.GpuHealth:
        calls.append(since)
        return queue.GpuHealth(
            ok=False, nvidia_smi_ok=True, kernel_errors=["NVRM: Xid (PCI:0000:01:00): 79"]
        )

    spec = _spec(("gpu-job", _python("pass"), 30), ("next", _python("pass"), 30))
    results = queue.QueueRunner(
        spec, log_path=tmp_path / "q.log", gpu_check=failing, journal_since="2026-09-18 00:00:00"
    ).run()
    assert [r.state for r in results] == ["blocked_gpu", "skipped"]
    assert "Xid" in results[0].detail
    assert calls == ["2026-09-18 00:00:00"]
    assert not (tmp_path / "logs" / "00_gpu-job.log").exists()


def test_spawn_failure_is_a_recorded_state_not_a_crash(tmp_path: Path) -> None:
    spec = _spec(("missing", ["/nonexistent/binary-xyz"], 30))
    results = queue.QueueRunner(spec, log_path=tmp_path / "q.log", gpu_check=_healthy).run()
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
    with pytest.raises(ValueError):
        queue.QueueJob(name="bad name", argv=["x"], timeout_s=1)


def test_cli_dry_run_lists_jobs_without_running(tmp_path: Path, capsys) -> None:
    path = tmp_path / "jobs.json"
    path.write_text(json.dumps({"jobs": [{"name": "a", "argv": ["false"], "timeout_s": 5}]}))
    assert queue.main([str(path), "--dry-run"]) == 0
    assert "00 a: false" in capsys.readouterr().out


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
