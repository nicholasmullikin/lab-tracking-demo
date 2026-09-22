"""Run GPU jobs one at a time overnight, with a watchdog and a stop-on-first-error rule.

A job list (JSON) names each job's argv, working directory, timeout and optional environment
or interpreter prefix.  The queue re-executes itself under `systemd-inhibit` so the machine
does not sleep, then runs the jobs strictly serially.  Before every job it checks that
`nvidia-smi` answers, that the kernel log since the queue started carries no `NVRM` or
`Xid` line, and (through `battle.gpu_guard`) that the card has room for the job: every
compute process on the GPU is classified by its /proc cmdline and recorded in the
`gpu_check` event with the headroom arithmetic; another Battle worker not named by
`--allow-gpu-neighbour`, an unknown neighbour above 2 GiB or too little headroom for the
job's profile blocks the job.  Thirty seconds into each job the queue looks at the card once
more and logs (never kills) any neighbour that appeared.  A job that exceeds its timeout is
killed (whole process group).  Every event is appended as one JSON line to the queue log, and
the queue stops at the first GPU error or non-zero exit unless `--continue-on-failure` is
given.

Nothing here knows what the jobs do; it is plumbing for the overnight multicam pass.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import gpu_guard

DEFAULT_LOG = Path("runs/overnight-multicam-20260918/queue.log")
INHIBIT_MARKER = "BATTLE_QUEUE_INHIBITED"
INHIBIT_WHY = "battle overnight"
KERNEL_ERROR_PATTERN = re.compile(r"NVRM|Xid")
KILL_GRACE_SECONDS = 10.0
GUARD_RECHECK_SECONDS = 30.0

JobState = Literal["succeeded", "failed", "timed_out", "blocked_gpu", "skipped", "spawn_failed"]
JOB_STATES: tuple[JobState, ...] = (
    "succeeded",
    "failed",
    "timed_out",
    "blocked_gpu",
    "skipped",
    "spawn_failed",
)


class QueueJob(BaseModel):
    """One serial job; `interpreter` is prepended to `argv` (e.g. `["uv", "run"]`).

    `gpu_profile` names the expected-peak profile the GPU guard plans headroom for (default:
    inferred from the command line, `unknown` = 4 GiB when nothing matches);
    `expected_peak_vram_bytes` overrides the profile's number.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    argv: list[str] = Field(min_length=1)
    cwd: str = "."
    timeout_s: float = Field(gt=0)
    env: dict[str, str] = Field(default_factory=dict)
    interpreter: list[str] = Field(default_factory=list)
    gpu_profile: str | None = None
    expected_peak_vram_bytes: int | None = Field(default=None, gt=0)

    @field_validator("gpu_profile")
    @classmethod
    def _known_profile(cls, value: str | None) -> str | None:
        if value is not None and value not in gpu_guard.EXPECTED_PEAK_BYTES:
            raise ValueError(
                f"unknown gpu_profile {value!r}; known: {sorted(gpu_guard.EXPECTED_PEAK_BYTES)}"
            )
        return value

    @property
    def command(self) -> list[str]:
        return [*self.interpreter, *self.argv]

    @property
    def guard_profile(self) -> str:
        return self.gpu_profile or gpu_guard.infer_profile(self.command)


class QueueSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    jobs: list[QueueJob] = Field(min_length=1)
    log_path: str | None = None


class GpuHealth(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    nvidia_smi_ok: bool
    kernel_errors: list[str] = Field(default_factory=list)
    detail: str = ""
    # `battle.gpu_guard` provenance (schema battle-gpu-guard/1): every neighbour on the card,
    # the headroom arithmetic and the decision; None when the guard did not run.
    gpu_guard: dict[str, Any] | None = None


class JobResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    state: JobState
    exit_code: int | None = None
    duration_s: float = 0.0
    stdout_log: str | None = None
    detail: str = ""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def nvidia_smi_responds(timeout_s: float = 30.0) -> tuple[bool, str]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return False, "nvidia-smi is not on PATH"
    try:
        completed = subprocess.run(
            [executable, "--query-gpu=name,memory.used,temperature.gpu", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"nvidia-smi did not answer within {timeout_s:.0f} s"
    if completed.returncode != 0:
        return False, f"nvidia-smi exit {completed.returncode}: {completed.stderr.strip()[:200]}"
    return True, completed.stdout.strip()


def kernel_gpu_errors(since: str, timeout_s: float = 30.0) -> list[str]:
    """`journalctl -k --since <since>` lines mentioning NVRM or Xid (empty when clean)."""
    executable = shutil.which("journalctl")
    if executable is None:
        return []
    try:
        completed = subprocess.run(
            [executable, "-k", "--since", since, "--no-pager", "-o", "short-iso"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return [f"journalctl did not answer within {timeout_s:.0f} s"]
    return [line for line in completed.stdout.splitlines() if KERNEL_ERROR_PATTERN.search(line)]


def default_gpu_check(since: str) -> GpuHealth:
    smi_ok, smi_detail = nvidia_smi_responds()
    errors = kernel_gpu_errors(since)
    return GpuHealth(
        ok=smi_ok and not errors,
        nvidia_smi_ok=smi_ok,
        kernel_errors=errors,
        detail=smi_detail if smi_ok else smi_detail,
    )


def _terminate_group(process: subprocess.Popen[bytes]) -> None:
    for sig, wait in ((signal.SIGTERM, KILL_GRACE_SECONDS), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


class QueueRunner:
    def __init__(
        self,
        spec: QueueSpec,
        *,
        log_path: Path,
        continue_on_failure: bool = False,
        gpu_check: Callable[[str], GpuHealth] = default_gpu_check,
        journal_since: str | None = None,
        gpu_guard_mode: str | None = gpu_guard.DEFAULT_GUARD_MODE,
        allow_gpu_neighbours: Sequence[int] = (),
        guard_probe: gpu_guard.GpuProbe | None = None,
        recheck_after_s: float = GUARD_RECHECK_SECONDS,
    ) -> None:
        self.spec = spec
        self.log_path = log_path
        self.continue_on_failure = continue_on_failure
        self.gpu_check = gpu_check
        self.journal_since = journal_since or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # None switches the neighbour/headroom guard off (`--no-gpu-check`).
        self.gpu_guard_mode = gpu_guard_mode
        self.allow_gpu_neighbours = [int(pid) for pid in allow_gpu_neighbours]
        self.guard_probe = guard_probe
        self.recheck_after_s = recheck_after_s
        self.results: list[JobResult] = []
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.job_log_dir = self.log_path.parent / "logs"
        self.job_log_dir.mkdir(parents=True, exist_ok=True)

    def _log(self, event: str, **fields: object) -> None:
        record = {"time": utc_now(), "event": event, **fields}
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")

    def guard_decision(
        self, job: QueueJob, own_pid: int | None = None
    ) -> gpu_guard.GuardDecision | None:
        """Evaluate the GPU guard for `job`; PIDs named on the queue or in the job's own argv
        (`--allow-gpu-neighbour`) are tolerated.  None when the guard is switched off."""
        if self.gpu_guard_mode is None:
            return None
        return gpu_guard.evaluate(
            mode=self.gpu_guard_mode,
            allowed_pids=[*self.allow_gpu_neighbours, *gpu_guard.allowed_pids_in_argv(job.command)],
            profile=job.guard_profile,
            expected_peak_vram_bytes=job.expected_peak_vram_bytes,
            own_pid=own_pid,
            probe=self.guard_probe,
        )

    def _wait_with_recheck(
        self,
        process: subprocess.Popen[bytes],
        job: QueueJob,
        baseline: gpu_guard.GuardDecision | None,
        started: float,
    ) -> int:
        """Wait for the job; `recheck_after_s` in, look at the card again and log what changed.

        The job's own processes are excluded from the re-check and nothing is ever killed
        here: the event records which neighbours appeared since the pre-job check and
        whether the guard would still accept the job, for a later reader.
        """
        if baseline is None or self.recheck_after_s >= job.timeout_s:
            return process.wait(timeout=job.timeout_s)
        try:
            return process.wait(timeout=self.recheck_after_s)
        except subprocess.TimeoutExpired:
            pass
        recheck = self.guard_decision(job, own_pid=process.pid)
        if recheck is not None:
            appeared = gpu_guard.new_neighbours(baseline, recheck)
            self._log(
                "gpu_recheck",
                job=job.name,
                seconds_after_start=round(time.monotonic() - started, 1),
                new_neighbours=[neighbour.as_dict() for neighbour in appeared],
                would_accept=recheck.accepted,
                reasons=recheck.reasons,
                gpu_guard=recheck.as_provenance(),
            )
        remaining = job.timeout_s - (time.monotonic() - started)
        return process.wait(timeout=max(remaining, 0.0))

    def _run_job(
        self, index: int, job: QueueJob, baseline: gpu_guard.GuardDecision | None = None
    ) -> JobResult:
        stdout_log = self.job_log_dir / f"{index:02d}_{job.name}.log"
        env = {**os.environ, **job.env}
        started = time.monotonic()
        with stdout_log.open("ab") as handle:
            handle.write(f"# {utc_now()} {' '.join(job.command)}\n".encode())
            handle.flush()
            try:
                process = subprocess.Popen(
                    job.command,
                    cwd=job.cwd,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except OSError as error:
                return JobResult(
                    name=job.name,
                    state="spawn_failed",
                    duration_s=time.monotonic() - started,
                    stdout_log=str(stdout_log),
                    detail=str(error),
                )
            try:
                exit_code = self._wait_with_recheck(process, job, baseline, started)
            except subprocess.TimeoutExpired:
                _terminate_group(process)
                return JobResult(
                    name=job.name,
                    state="timed_out",
                    exit_code=process.returncode,
                    duration_s=time.monotonic() - started,
                    stdout_log=str(stdout_log),
                    detail=f"killed after {job.timeout_s:.0f} s",
                )
        return JobResult(
            name=job.name,
            state="succeeded" if exit_code == 0 else "failed",
            exit_code=exit_code,
            duration_s=time.monotonic() - started,
            stdout_log=str(stdout_log),
        )

    def run(self) -> list[JobResult]:
        self._log(
            "queue_start",
            jobs=[job.name for job in self.spec.jobs],
            continue_on_failure=self.continue_on_failure,
            journal_since=self.journal_since,
            pid=os.getpid(),
        )
        stopped_reason: str | None = None
        for index, job in enumerate(self.spec.jobs):
            if stopped_reason is not None:
                result = JobResult(name=job.name, state="skipped", detail=stopped_reason)
                self.results.append(result)
                self._log("job_skipped", job=job.name, reason=stopped_reason)
                continue
            health = self.gpu_check(self.journal_since)
            guard = self.guard_decision(job)
            if guard is not None:
                detail = health.detail
                if not guard.accepted:
                    detail = f"gpu guard refused: {guard.reason}"
                    if not health.ok:
                        detail = f"{health.detail}; {detail}" if health.detail else detail
                health = health.model_copy(
                    update={
                        "ok": health.ok and guard.accepted,
                        "detail": detail,
                        "gpu_guard": guard.as_provenance(),
                    }
                )
            self._log("gpu_check", job=job.name, **health.model_dump())
            if not health.ok:
                detail = health.detail or "; ".join(health.kernel_errors[:3])
                result = JobResult(name=job.name, state="blocked_gpu", detail=detail)
                self.results.append(result)
                self._log("job_blocked", job=job.name, detail=detail)
                stopped_reason = f"gpu check failed before {job.name}: {detail}"
                continue
            self._log(
                "job_start",
                job=job.name,
                command=job.command,
                cwd=job.cwd,
                timeout_s=job.timeout_s,
                gpu_profile=job.guard_profile if guard is not None else None,
            )
            result = self._run_job(index, job, guard)
            self.results.append(result)
            self._log("job_end", **result.model_dump())
            if result.state != "succeeded" and not self.continue_on_failure:
                stopped_reason = f"{job.name} {result.state}"
        self._log(
            "queue_end",
            stopped=stopped_reason is not None,
            reason=stopped_reason,
            summary={state: sum(r.state == state for r in self.results) for state in JOB_STATES},
        )
        return self.results


def load_spec(path: Path) -> QueueSpec:
    return QueueSpec.model_validate_json(path.read_text(encoding="utf-8"))


def reexec_under_inhibit(argv: list[str]) -> int:
    """Run this same command under systemd-inhibit; returns its exit code."""
    inhibit = shutil.which("systemd-inhibit")
    if inhibit is None:
        return -1
    env = {**os.environ, INHIBIT_MARKER: "1"}
    completed = subprocess.run(
        [
            inhibit,
            "--what=sleep:idle",
            f"--why={INHIBIT_WHY}",
            sys.executable,
            "-m",
            "battle.overnight_queue",
            *argv,
        ],
        env=env,
        check=False,
    )
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path, help="JSON job list")
    parser.add_argument("--log", type=Path, default=None, help=f"JSONL log (default {DEFAULT_LOG})")
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--no-inhibit", action="store_true", help="do not wrap in systemd-inhibit")
    parser.add_argument(
        "--no-gpu-check",
        action="store_true",
        help="skip the nvidia-smi/journal checks and the GPU guard (CPU-only lists)",
    )
    parser.add_argument(
        "--gpu-guard",
        choices=gpu_guard.GUARD_MODES,
        default=gpu_guard.DEFAULT_GUARD_MODE,
        help=(
            "GPU guard evaluated before every job and logged in the gpu_check event. strict: "
            "any model-like process by nvidia-smi name blocks (the Sep 21 worker logic); vram "
            "(default): neighbours classified by /proc cmdline, another Battle worker, an "
            "unknown neighbour above 2 GiB or headroom below 1.5 x the job's expected peak "
            "blocks."
        ),
    )
    parser.add_argument(
        "--allow-gpu-neighbour",
        type=int,
        action="append",
        default=[],
        metavar="PID",
        help=(
            "PID of a GPU process the queue's guard may tolerate for every job (repeatable); "
            "PIDs named by --allow-gpu-neighbour inside a job's own argv are tolerated for "
            "that job as well."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="list the jobs and exit")
    args = parser.parse_args(argv)
    spec = load_spec(args.spec)
    log_path = args.log or (Path(spec.log_path) if spec.log_path else DEFAULT_LOG)
    if args.dry_run:
        for index, job in enumerate(spec.jobs):
            print(
                f"{index:02d} {job.name}: {' '.join(job.command)} "
                f"(cwd {job.cwd}, {job.timeout_s:.0f} s, gpu profile {job.guard_profile})"
            )
        return 0
    if not args.no_inhibit and os.environ.get(INHIBIT_MARKER) != "1":
        code = reexec_under_inhibit(sys.argv[1:] if argv is None else argv)
        if code >= 0:
            return code
        print("systemd-inhibit unavailable; running without a sleep inhibitor", file=sys.stderr)
    gpu_check = (
        (lambda _since: GpuHealth(ok=True, nvidia_smi_ok=True, detail="gpu check disabled"))
        if args.no_gpu_check
        else default_gpu_check
    )
    runner = QueueRunner(
        spec,
        log_path=log_path,
        continue_on_failure=args.continue_on_failure,
        gpu_check=gpu_check,
        gpu_guard_mode=None if args.no_gpu_check else args.gpu_guard,
        allow_gpu_neighbours=args.allow_gpu_neighbour,
    )
    results = runner.run()
    for result in results:
        print(f"{result.name}: {result.state}" + (f" ({result.detail})" if result.detail else ""))
    return 0 if all(result.state == "succeeded" for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
