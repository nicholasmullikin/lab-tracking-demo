"""VRAM-aware GPU guard shared by the SAM3 worker, `battle-muggled-smoke` and the overnight queue.

The guard exists because this machine hard-crashed once under GPU load, so a run must not
start beside anything that could push the card past its memory.  Until Sep 21 the guard was
process-name based: any non-whitelisted "python-ish" process on the GPU refused the start.
That blocked runs beside a 389 MiB video player, a 1.4 GiB game or a browser that would all
have fit comfortably on the 16 GiB card, while the risk it should measure (two model
processes, or too little headroom) was not what it measured.

Two modes, chosen by ``--gpu-guard``:

``strict``
    The Sep 21 logic, byte for byte: :func:`looks_like_model_process` on nvidia-smi's process
    name (Rerun viewer tolerated), every model-like process blocks unless its PID was named
    with ``--allow-gpu-neighbour``.

``vram`` (default)
    Every compute process on the card is classified by its ``/proc/<pid>/cmdline`` (not by
    nvidia-smi's name, which reports a video player as ``python3``):

    * ``own_repo_model``: a Battle GPU worker (:data:`OWN_REPO_WORKER_BASENAMES`), including
      the calibration workspace worker because it holds a SAM3 model;
    * ``known_benign``: an allowlisted desktop compositor, browser, media player, game or the
      Rerun viewer (:data:`BENIGN_BASENAMES`, :data:`BENIGN_CMDLINE_MARKERS`), plus any process
      under :data:`SMALL_NEIGHBOUR_MIB` that is not ``own_repo_model``;
    * ``unknown``: everything else.

    The start is refused when another ``own_repo_model`` is running and its PID was not
    allowed, when an ``unknown`` neighbour holds more than :data:`UNKNOWN_NEIGHBOUR_LIMIT_MIB`,
    or when ``headroom = total - used - sum(neighbour reservations)`` is below
    :data:`HEADROOM_FACTOR` times the job's expected peak.  A neighbour's reservation is the
    growth it may still do on top of what it holds now (:data:`GROWTH_FRACTION` of its current
    usage: half again for a model process, nothing for a benign one).  The expected peak comes
    from ``--expected-peak-vram-bytes`` or from a profile default taken from recorded worker
    manifests (:data:`EXPECTED_PEAK_BYTES`).

Every neighbour (pid, class, MiB, cmdline basename), the headroom arithmetic and the decision
are returned as one provenance dictionary (:meth:`GuardDecision.as_provenance`) that the
worker stores under ``runtime_settings.gpu_guard`` and the queue in its ``gpu_check`` event,
so a later reader can see what shared the card.

This module imports only the standard library: the SAM3 worker runs it under the separately
managed MuggledSAM interpreter, where the Battle package is not installed.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

PROVENANCE_SCHEMA = "battle-gpu-guard/1"
GUARD_MODES = ("strict", "vram")
DEFAULT_GUARD_MODE = "vram"
MIB = 1024**2
GIB = 1024**3

# Decision constants of the vram mode.
HEADROOM_FACTOR = 1.5
SMALL_NEIGHBOUR_MIB = 512
UNKNOWN_NEIGHBOUR_LIMIT_MIB = 2048
NEIGHBOUR_CLASSES = ("own_repo_model", "known_benign", "unknown")
# Growth a neighbour may still do on top of what it holds when the guard looks: a model
# process allocates again when it decodes or grows its memory bank; a compositor, browser,
# player or game holds a roughly steady working set.
GROWTH_FRACTION: dict[str, float] = {"own_repo_model": 0.5, "unknown": 0.5, "known_benign": 0.0}

# Expected peak VRAM per job profile, from recorded `worker_result.json` manifests
# (`gpu_peak_vram_bytes`): SAM3 multiplex at max side 1280 2.39-2.65 GiB, at 1920 3.36 GiB,
# the four-part SAM2 worker 3.29 GiB, DAM4SAM large 6.80-6.85 GiB.
EXPECTED_PEAK_BYTES: dict[str, int] = {
    "sam3_1280": int(2.6 * GIB),
    "sam3_1080p": int(3.4 * GIB),
    "four_part": int(3.4 * GIB),
    "dam4sam_large": int(7.0 * GIB),
    "unknown": int(4.0 * GIB),
}
DEFAULT_PROFILE = "unknown"

# Battle GPU workers, matched by the basename of any command-line token, so a code-snapshot
# copy under runs/ matches as well as the working tree.
OWN_REPO_WORKER_BASENAMES = frozenset(
    {
        "muggled_worker.py",
        "muggled_calibration_worker.py",
        "four_part_video_worker.py",
        "dam4sam_video_worker.py",
        "samurai_video_worker.py",
        "grounding_dino_sam2_video_worker.py",
        "kineo_fusion_worker.py",
        "kineo_multiview_runner.py",
        "wilor_worker.py",
        "boxmot_worker.py",
        "finebio_sam3_smoke.py",
    }
)
# Desktop compositors, browsers, media players, games and launchers, the Rerun viewer.
BENIGN_BASENAMES = frozenset(
    {
        # compositors / display
        "kwin_wayland",
        "kwin_x11",
        "kwin",
        "plasmashell",
        "kscreenlocker_greet",
        "gnome-shell",
        "mutter",
        "Xorg",
        "Xwayland",
        "Hyprland",
        "sway",
        "weston",
        "gamescope",
        # browsers
        "firefox",
        "firefox-bin",
        "chrome",
        "chromium",
        "chromium-browser",
        "google-chrome",
        "brave",
        "brave-browser",
        "msedge",
        "vivaldi-bin",
        "epiphany",
        "WebKitWebProcess",
        # media players / image viewers
        "showtime",
        "mpv",
        "vlc",
        "totem",
        "celluloid",
        "haruna",
        "dragon",
        "smplayer",
        "kodi",
        "ffplay",
        "gwenview",
        "loupe",
        "eog",
        "obs",
        # games / launchers
        "steam",
        "steamwebhelper",
        "wine",
        "wine64",
        "wineserver",
        "wine-preloader",
        "wine64-preloader",
        "lutris",
        "heroic",
        # the Rerun viewer (a renderer, not a model)
        "rerun",
    }
)
# Substrings of the joined command line that mark a benign process whatever its basename.
BENIGN_CMDLINE_MARKERS = ("/rerun_sdk/rerun_cli/", "steamapps/", "steamapps\\")
_INTERPRETER_BASENAME = re.compile(r"^(python[\d.]*|python[\d.]*-bin|perl|node|ruby|sh|bash|zsh)$")
_MAX_ANCESTRY_STEPS = 64


# ------------------------------------------------------------------------------------------
# nvidia-smi and /proc access (injectable for tests)


def run_nvidia_smi(arguments: Sequence[str], timeout_s: float = 30.0) -> str | None:
    """Return nvidia-smi's stdout for `arguments`, or None when it is unavailable or fails."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    try:
        completed = subprocess.run(
            [executable, *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def read_cmdline(pid: int) -> list[str] | None:
    """`/proc/<pid>/cmdline` as a list, or None when the process is gone or unreadable."""
    try:
        with open(f"/proc/{int(pid)}/cmdline", "rb") as handle:
            data = handle.read()
    except OSError:
        return None
    tokens = data.split(b"\0")
    while tokens and not tokens[-1]:
        tokens.pop()
    return [token.decode("utf-8", errors="replace") for token in tokens]


def read_ppid(pid: int) -> int | None:
    """Parent PID from `/proc/<pid>/status`, or None when unreadable."""
    try:
        with open(f"/proc/{int(pid)}/status", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith("PPid:"):
                    return int(line.split(":", 1)[1].strip())
    except (OSError, ValueError):
        return None
    return None


@dataclass(frozen=True)
class GpuProbe:
    """The three system reads the guard makes; tests substitute stubs."""

    nvidia_smi: Callable[[Sequence[str]], str | None] = run_nvidia_smi
    cmdline: Callable[[int], list[str] | None] = read_cmdline
    ppid: Callable[[int], int | None] = read_ppid


DEFAULT_PROBE = GpuProbe()


# ------------------------------------------------------------------------------------------
# nvidia-smi parsing


@dataclass(frozen=True)
class GpuMemory:
    index: int
    name: str
    total_mib: int
    used_mib: int


@dataclass(frozen=True)
class ComputeApp:
    pid: int
    process_name: str
    used_mib: int | None


GPU_QUERY = ("--query-gpu=index,name,memory.total,memory.used", "--format=csv,noheader,nounits")
COMPUTE_APPS_QUERY = (
    "--query-compute-apps=pid,process_name,used_gpu_memory",
    "--format=csv,noheader,nounits",
)


def _int_or_none(text: str) -> int | None:
    text = text.strip()
    if not text or text.startswith("["):  # "[N/A]", "[Not Supported]"
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def parse_gpu_memory(csv_text: str | None) -> list[GpuMemory]:
    """Parse `--query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits`."""
    gpus: list[GpuMemory] = []
    for line in (csv_text or "").splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 4:
            continue
        index = _int_or_none(fields[0])
        total = _int_or_none(fields[-2])
        used = _int_or_none(fields[-1])
        if index is None or total is None or used is None:
            continue
        gpus.append(
            GpuMemory(index=index, name=", ".join(fields[1:-2]), total_mib=total, used_mib=used)
        )
    return gpus


def parse_compute_apps(csv_text: str | None) -> list[ComputeApp]:
    """Parse `--query-compute-apps=pid,process_name,used_gpu_memory` (csv, noheader, nounits)."""
    apps: list[ComputeApp] = []
    for line in (csv_text or "").splitlines():
        if not line.strip() or line.startswith("No running processes found"):
            continue
        pid_text, _, rest = line.partition(",")
        name, _, memory_text = rest.rpartition(",")
        pid = _int_or_none(pid_text)
        if pid is None or not name.strip():
            continue
        apps.append(
            ComputeApp(pid=pid, process_name=name.strip(), used_mib=_int_or_none(memory_text))
        )
    return apps


def select_gpu(gpus: Sequence[GpuMemory], visible_devices: str | None = None) -> GpuMemory | None:
    """The GPU a job will use: the first `CUDA_VISIBLE_DEVICES` entry when it is an index."""
    if not gpus:
        return None
    if visible_devices is None:
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible = visible_devices.split(",")[0].strip()
    if visible.isdigit():
        for gpu in gpus:
            if gpu.index == int(visible):
                return gpu
    return gpus[0]


# ------------------------------------------------------------------------------------------
# classification


def basename(token: str) -> str:
    """Last path component for POSIX and Windows (Wine/Proton) paths alike."""
    return re.split(r"[\\/]", token.strip())[-1]


def program_basename(cmdline: Sequence[str]) -> str | None:
    """Basename of the program a command line runs, looking past interpreter wrappers.

    `/usr/bin/python3 /usr/bin/showtime video.mp4` is `showtime`; `/usr/bin/kwin_wayland ...`
    is `kwin_wayland`; `S:\\steamapps\\common\\Synthetik\\SYNTHETIK.exe` is `SYNTHETIK.exe`.
    """
    if not cmdline:
        return None
    first = basename(cmdline[0])
    if not _INTERPRETER_BASENAME.match(first):
        return first
    for token in cmdline[1:]:
        if token.startswith("-"):
            continue
        return basename(token) or first
    return first


@dataclass(frozen=True)
class Neighbour:
    pid: int
    neighbour_class: str
    used_mib: int | None
    reserved_mib: int
    cmdline_basename: str | None
    process_name: str
    allowed_by_operator: bool
    why: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "class": self.neighbour_class,
            "used_mib": self.used_mib,
            "reserved_mib": self.reserved_mib,
            "cmdline_basename": self.cmdline_basename,
            "process_name": self.process_name,
            "allowed_by_operator": self.allowed_by_operator,
            "why": self.why,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Neighbour:
        return cls(
            pid=int(payload["pid"]),
            neighbour_class=str(payload["class"]),
            used_mib=None if payload.get("used_mib") is None else int(payload["used_mib"]),
            reserved_mib=int(payload["reserved_mib"]),
            cmdline_basename=payload.get("cmdline_basename"),
            process_name=str(payload.get("process_name", "")),
            allowed_by_operator=bool(payload.get("allowed_by_operator", False)),
            why=str(payload.get("why", "")),
        )


def classify_neighbour(
    app: ComputeApp, cmdline: Sequence[str] | None, allowed_pids: Iterable[int] = ()
) -> Neighbour:
    """Classify one compute process as own_repo_model, known_benign or unknown (see module doc)."""
    allowed = int(app.pid) in {int(pid) for pid in allowed_pids}
    program = program_basename(cmdline) if cmdline else None
    if cmdline:
        joined = " ".join(cmdline)
        own = [basename(token) for token in cmdline if basename(token) in OWN_REPO_WORKER_BASENAMES]
        if own:
            neighbour_class, why = "own_repo_model", f"cmdline names {own[0]}"
        elif program in BENIGN_BASENAMES:
            neighbour_class, why = "known_benign", f"allowlisted basename {program}"
        elif program is not None and program.lower().endswith(".exe"):
            neighbour_class, why = "known_benign", f"Windows program under Wine/Proton: {program}"
        elif any(marker in joined for marker in BENIGN_CMDLINE_MARKERS):
            marker = next(marker for marker in BENIGN_CMDLINE_MARKERS if marker in joined)
            neighbour_class, why = "known_benign", f"cmdline contains {marker!r}"
        elif app.used_mib is not None and app.used_mib < SMALL_NEIGHBOUR_MIB:
            neighbour_class, why = "known_benign", f"under {SMALL_NEIGHBOUR_MIB} MiB"
        else:
            neighbour_class, why = "unknown", "not a Battle worker, not allowlisted, not small"
    else:
        program = basename(app.process_name) or None
        if app.used_mib is not None and app.used_mib < SMALL_NEIGHBOUR_MIB:
            neighbour_class = "known_benign"
            why = f"cmdline unreadable; under {SMALL_NEIGHBOUR_MIB} MiB by nvidia-smi's name"
        else:
            neighbour_class, why = "unknown", "cmdline unreadable"
    reserved = (
        0
        if app.used_mib is None
        else int(math.ceil(app.used_mib * GROWTH_FRACTION[neighbour_class]))
    )
    return Neighbour(
        pid=int(app.pid),
        neighbour_class=neighbour_class,
        used_mib=app.used_mib,
        reserved_mib=reserved,
        cmdline_basename=program,
        process_name=app.process_name,
        allowed_by_operator=allowed,
        why=why,
    )


def is_descendant_of(pid: int, ancestor: int, ppid_of: Callable[[int], int | None]) -> bool:
    """True when `pid` is `ancestor` or has it in its parent chain (`/proc/<pid>/status`)."""
    current: int | None = int(pid)
    for _ in range(_MAX_ANCESTRY_STEPS):
        if current is None or current <= 0:
            return False
        if current == int(ancestor):
            return True
        current = ppid_of(current)
    return False


# ------------------------------------------------------------------------------------------
# strict mode (the Sep 21 logic, verbatim)


def looks_like_model_process(process: dict[str, str]) -> bool:
    """Flag GPU processes that could be running a model, tolerating the Rerun viewer.

    The viewer is a renderer, not a model, but it lives under a Python venv path
    (``.../site-packages/rerun_sdk/rerun_cli/rerun``) and is launched by a ``python``
    wrapper, so the name tokens alone would refuse to start beside an open recording.
    """
    name = process["process_name"].lower()
    if is_rerun_viewer(name):
        return False
    return any(token in name for token in ("python", "torch", "ollama", "llama", "vllm"))


def is_rerun_viewer(name: str) -> bool:
    executable = name.rsplit("/", 1)[-1]
    return executable == "rerun" or "/rerun_sdk/rerun_cli/" in name


def partition_neighbours_strict(
    gpu_processes: list[dict[str, str]], allowed_pids: Iterable[int]
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Split the model-like GPU processes into (blocking, allowed-by-operator).

    An operator may name specific PIDs (never process names) that are allowed to share the
    GPU, e.g. a human's calibration workspace worker that must not be stopped.
    """
    allowed = {str(int(pid)) for pid in allowed_pids}
    blocking: list[dict[str, str]] = []
    tolerated: list[dict[str, str]] = []
    for process in gpu_processes:
        if not looks_like_model_process(process):
            continue
        (tolerated if process["pid"] in allowed else blocking).append(process)
    return blocking, tolerated


def legacy_process_records(apps: Sequence[ComputeApp]) -> list[dict[str, str]]:
    """The `{pid, process_name, memory}` records the Sep 21 guard read from nvidia-smi."""
    return [
        {
            "pid": str(app.pid),
            "process_name": app.process_name,
            "memory": "[N/A]" if app.used_mib is None else f"{app.used_mib} MiB",
        }
        for app in apps
    ]


# ------------------------------------------------------------------------------------------
# profiles


def sam3_profile_for_side_length(max_side_length: int) -> str:
    """Profile of the SAM3 multiplex worker from its `--max-side-length`."""
    if max_side_length <= 1280:
        return "sam3_1280"
    if max_side_length <= 1920:
        return "sam3_1080p"
    return DEFAULT_PROFILE


def infer_profile(argv: Sequence[str]) -> str:
    """Guess a job's profile from its command line; `unknown` (4 GiB) when nothing matches."""
    tokens = [basename(token) for token in argv]
    if any(token in {"battle-dam4sam-video", "dam4sam_video_worker.py"} for token in tokens):
        return "dam4sam_large"
    if any(
        token in {"battle-four-part-segmentation", "four_part_video_worker.py"} for token in tokens
    ):
        return "four_part"
    if any(token in {"battle-muggled-smoke", "muggled_worker.py"} for token in tokens):
        side = 1280
        for position, token in enumerate(argv[:-1]):
            if token == "--max-side-length" and str(argv[position + 1]).isdigit():
                side = int(argv[position + 1])
        return sam3_profile_for_side_length(side)
    return DEFAULT_PROFILE


def expected_peak_bytes(profile: str | None, override: int | None = None) -> tuple[int, str, str]:
    """(bytes, source, profile) for a job: an explicit override wins over the profile default."""
    name = profile or DEFAULT_PROFILE
    if name not in EXPECTED_PEAK_BYTES:
        raise ValueError(
            f"unknown GPU guard profile {name!r}; known: {sorted(EXPECTED_PEAK_BYTES)}"
        )
    if override is not None:
        if override <= 0:
            raise ValueError("--expected-peak-vram-bytes must be positive")
        return int(override), "flag", name
    return EXPECTED_PEAK_BYTES[name], "profile_default", name


# ------------------------------------------------------------------------------------------
# decision


@dataclass
class GuardDecision:
    mode: str
    accepted: bool
    reasons: list[str]
    checked_at: str
    nvidia_smi_available: bool
    gpu: GpuMemory | None
    neighbours: list[Neighbour]
    own_processes: list[Neighbour]
    allowed_neighbour_pids: list[int]
    expected_peak_bytes: int
    expected_peak_source: str
    profile: str
    headroom: dict[str, Any] | None
    notes: list[str] = field(default_factory=list)

    @property
    def reason(self) -> str | None:
        return "; ".join(self.reasons) if self.reasons else None

    @property
    def neighbour_pids(self) -> set[int]:
        return {neighbour.pid for neighbour in self.neighbours}

    def as_provenance(self) -> dict[str, Any]:
        present = self.neighbour_pids | {own.pid for own in self.own_processes}
        return {
            "schema": PROVENANCE_SCHEMA,
            "mode": self.mode,
            "checked_at": self.checked_at,
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "nvidia_smi_available": self.nvidia_smi_available,
            "gpu": None
            if self.gpu is None
            else {
                "index": self.gpu.index,
                "name": self.gpu.name,
                "total_mib": self.gpu.total_mib,
                "used_mib": self.gpu.used_mib,
            },
            "allowed_neighbour_pids": list(self.allowed_neighbour_pids),
            "allowed_neighbour_pids_absent": [
                pid for pid in self.allowed_neighbour_pids if pid not in present
            ],
            "neighbours": [neighbour.as_dict() for neighbour in self.neighbours],
            "own_processes": [own.as_dict() for own in self.own_processes],
            "expected_peak": {
                "bytes": self.expected_peak_bytes,
                "mib": int(math.ceil(self.expected_peak_bytes / MIB)),
                "source": self.expected_peak_source,
                "profile": self.profile,
            },
            "headroom": None if self.headroom is None else dict(self.headroom),
            "rules": {
                "headroom_factor": HEADROOM_FACTOR,
                "small_neighbour_mib": SMALL_NEIGHBOUR_MIB,
                "unknown_neighbour_limit_mib": UNKNOWN_NEIGHBOUR_LIMIT_MIB,
                "growth_fraction": dict(GROWTH_FRACTION),
            },
            "notes": list(self.notes),
            "semantics": (
                "neighbours are the other compute processes on the card when the guard looked, "
                "classified by /proc cmdline; own_processes belong to the job itself and are "
                "excluded from the rules; headroom_mib = total - used + own - sum(reserved); "
                "the job may start when accepted is true. Wall-clock timing beside any "
                "neighbour may see contention; gpu_peak_vram_bytes stays this process's own."
            ),
        }

    @classmethod
    def from_provenance(cls, payload: dict[str, Any]) -> GuardDecision:
        if payload.get("schema") != PROVENANCE_SCHEMA:
            raise ValueError(f"not a {PROVENANCE_SCHEMA} record: {payload.get('schema')!r}")
        gpu = payload.get("gpu")
        peak = payload["expected_peak"]
        return cls(
            mode=str(payload["mode"]),
            accepted=bool(payload["accepted"]),
            reasons=[str(reason) for reason in payload.get("reasons", [])],
            checked_at=str(payload["checked_at"]),
            nvidia_smi_available=bool(payload["nvidia_smi_available"]),
            gpu=None
            if gpu is None
            else GpuMemory(
                index=int(gpu["index"]),
                name=str(gpu["name"]),
                total_mib=int(gpu["total_mib"]),
                used_mib=int(gpu["used_mib"]),
            ),
            neighbours=[Neighbour.from_dict(item) for item in payload.get("neighbours", [])],
            own_processes=[Neighbour.from_dict(item) for item in payload.get("own_processes", [])],
            allowed_neighbour_pids=[int(pid) for pid in payload.get("allowed_neighbour_pids", [])],
            expected_peak_bytes=int(peak["bytes"]),
            expected_peak_source=str(peak["source"]),
            profile=str(peak["profile"]),
            headroom=None if payload.get("headroom") is None else dict(payload["headroom"]),
            notes=[str(note) for note in payload.get("notes", [])],
        )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def describe(neighbour: Neighbour) -> str:
    size = "size unknown" if neighbour.used_mib is None else f"{neighbour.used_mib} MiB"
    return f"pid {neighbour.pid} ({neighbour.cmdline_basename or neighbour.process_name}, {size})"


def evaluate(
    *,
    mode: str = DEFAULT_GUARD_MODE,
    allowed_pids: Iterable[int] = (),
    profile: str | None = None,
    expected_peak_vram_bytes: int | None = None,
    own_pid: int | None = None,
    probe: GpuProbe | None = None,
    visible_devices: str | None = None,
    now: Callable[[], str] = _utc_now,
) -> GuardDecision:
    """Query the card once and decide whether a job with this profile may start.

    `own_pid`: the job's own process (the worker itself, or the queue's child); it and its
    descendants are reported under `own_processes` and left out of the rules, and their
    memory is added back into the headroom so a re-check 30 s into a job answers "would the
    guard still accept this job now".  `probe` defaults to the real nvidia-smi and /proc
    reads (:data:`DEFAULT_PROBE`, resolved at call time so tests can substitute it).
    """
    if mode not in GUARD_MODES:
        raise ValueError(f"unknown GPU guard mode {mode!r}; expected one of {GUARD_MODES}")
    if probe is None:
        probe = DEFAULT_PROBE
    allowed = list(dict.fromkeys(int(pid) for pid in allowed_pids))  # operator order, deduplicated
    peak_bytes, peak_source, profile_name = expected_peak_bytes(profile, expected_peak_vram_bytes)
    gpu_text = probe.nvidia_smi(list(GPU_QUERY))
    apps_text = probe.nvidia_smi(list(COMPUTE_APPS_QUERY))
    available = gpu_text is not None and apps_text is not None
    gpu = select_gpu(parse_gpu_memory(gpu_text), visible_devices)
    apps = parse_compute_apps(apps_text)

    neighbours: list[Neighbour] = []
    own_processes: list[Neighbour] = []
    for app in apps:
        classified = classify_neighbour(app, probe.cmdline(app.pid), allowed)
        if own_pid is not None and is_descendant_of(app.pid, own_pid, probe.ppid):
            own_processes.append(classified)
        else:
            neighbours.append(classified)

    reasons: list[str] = []
    notes: list[str] = []
    headroom: dict[str, Any] | None = None
    if not available:
        notes.append(
            "nvidia-smi unavailable or failed: neighbours and headroom were not evaluated; "
            "the CUDA availability check decides"
        )
    if mode == "strict":
        blocking, _tolerated = partition_neighbours_strict(legacy_process_records(apps), allowed)
        blocking = [
            record
            for record in blocking
            if int(record["pid"]) not in {own.pid for own in own_processes}
        ]
        if blocking:
            reasons.append(f"concurrent GPU model process(es) detected: {blocking}")
    else:
        foreign_workers = [
            neighbour
            for neighbour in neighbours
            if neighbour.neighbour_class == "own_repo_model" and not neighbour.allowed_by_operator
        ]
        if foreign_workers:
            reasons.append(
                "another Battle GPU worker is running: "
                + ", ".join(describe(neighbour) for neighbour in foreign_workers)
                + "; name its PID with --allow-gpu-neighbour only if the two must share the card"
            )
        large_unknown = [
            neighbour
            for neighbour in neighbours
            if neighbour.neighbour_class == "unknown"
            and neighbour.used_mib is not None
            and neighbour.used_mib > UNKNOWN_NEIGHBOUR_LIMIT_MIB
        ]
        if large_unknown:
            reasons.append(
                f"unknown GPU neighbour above {UNKNOWN_NEIGHBOUR_LIMIT_MIB} MiB: "
                + ", ".join(describe(neighbour) for neighbour in large_unknown)
            )
        if gpu is not None:
            own_mib = sum(own.used_mib or 0 for own in own_processes)
            reserved_mib = sum(neighbour.reserved_mib for neighbour in neighbours)
            headroom_mib = gpu.total_mib - gpu.used_mib + own_mib - reserved_mib
            expected_peak_mib = int(math.ceil(peak_bytes / MIB))
            required_mib = int(math.ceil(expected_peak_mib * HEADROOM_FACTOR))
            headroom = {
                "total_mib": gpu.total_mib,
                "used_mib": gpu.used_mib,
                "own_mib": own_mib,
                "reserved_mib": reserved_mib,
                "headroom_mib": headroom_mib,
                "expected_peak_mib": expected_peak_mib,
                "factor": HEADROOM_FACTOR,
                "required_mib": required_mib,
                "sufficient": headroom_mib >= required_mib,
                "formula": "headroom_mib = total - used + own - sum(reserved); "
                "required_mib = ceil(expected_peak_mib x factor)",
            }
            if headroom_mib < required_mib:
                reasons.append(
                    f"VRAM headroom {headroom_mib} MiB (total {gpu.total_mib} - used "
                    f"{gpu.used_mib} + own {own_mib} - reserved {reserved_mib}) is below the "
                    f"required {required_mib} MiB ({HEADROOM_FACTOR:g} x expected peak "
                    f"{headroom['expected_peak_mib']} MiB, profile {profile_name})"
                )
        elif available:
            notes.append("nvidia-smi reported no GPU memory line; headroom not evaluated")
    return GuardDecision(
        mode=mode,
        accepted=not reasons,
        reasons=reasons,
        checked_at=now(),
        nvidia_smi_available=available,
        gpu=gpu,
        neighbours=neighbours,
        own_processes=own_processes,
        allowed_neighbour_pids=allowed,
        expected_peak_bytes=peak_bytes,
        expected_peak_source=peak_source,
        profile=profile_name,
        headroom=headroom,
        notes=notes,
    )


def allowed_pids_in_argv(argv: Sequence[str]) -> list[int]:
    """PIDs named by `--allow-gpu-neighbour PID` (or `=PID`) anywhere on a command line."""
    pids: list[int] = []
    for position, token in enumerate(argv):
        if token == "--allow-gpu-neighbour" and position + 1 < len(argv):
            value = str(argv[position + 1])
        elif token.startswith("--allow-gpu-neighbour="):
            value = token.split("=", 1)[1]
        else:
            continue
        if value.strip().isdigit():
            pids.append(int(value))
    return pids


def new_neighbours(before: GuardDecision, after: GuardDecision) -> list[Neighbour]:
    """Neighbours present in `after` whose PID was not on the card in `before`."""
    seen = before.neighbour_pids | {own.pid for own in before.own_processes}
    return [neighbour for neighbour in after.neighbours if neighbour.pid not in seen]
