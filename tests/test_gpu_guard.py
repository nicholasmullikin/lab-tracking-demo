"""VRAM-aware GPU guard: classification, headroom, the four decisions, strict parity, provenance.

No GPU: nvidia-smi output and /proc cmdlines are stubbed with the processes that shared the
card on the night of Sep 21 (compositor, calibration worker, a Steam game, a tracker worker,
the `showtime` video player nvidia-smi reports as python3).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from battle import gpu_guard
from battle.gpu_guard import (
    ComputeApp,
    GpuProbe,
    GuardDecision,
    classify_neighbour,
    evaluate,
    looks_like_model_process,
)

TOTAL_MIB = 16303
CALIBRATION_PYTHON = "/home/nick/.pyenv/versions/muggled_sam/bin/python"
KWIN = (8429, "/usr/bin/kwin_wayland", 144)
CALIBRATION = (2071175, CALIBRATION_PYTHON, 1198)
GAME = (2202800, "S:\\steamapps\\common\\Synthetik\\SYNTHETIK.exe", 1460)
TRACKER = (2236310, CALIBRATION_PYTHON, 3354)
PLAYER = (2158886, "/usr/bin/python3", 389)
VIEWER = (
    9001,
    "/home/nick/src/battle/.venv/lib/python3.12/site-packages/rerun_sdk/rerun_cli/rerun",
    623,
)
OLLAMA = (77, "/usr/bin/ollama", 3072)

CMDLINES: dict[int, list[str]] = {
    8429: ["/usr/bin/kwin_wayland", "--wayland-fd", "7", "--socket", "wayland-0"],
    2071175: [
        CALIBRATION_PYTHON,
        "/home/nick/src/battle/src/battle/muggled_calibration_worker.py",
        "--serve-jsonl",
        "--proxy",
        "/home/nick/src/battle/data/derived/assembly101/proxy.mp4",
    ],
    2202800: ["S:\\steamapps\\common\\Synthetik\\SYNTHETIK.exe"],
    2236310: [
        CALIBRATION_PYTHON,
        "/home/nick/src/battle/runs/multiview-reprompt-20260921/code-snapshot-259b4f7/src/battle/"
        "muggled_worker.py",
        "--run-directory",
        "/home/nick/src/battle/runs/multiview-reprompt-20260921/C10379",
    ],
    2158886: ["/usr/bin/python3", "/usr/bin/showtime", "/home/nick/finebio/P03_01_01.mp4"],
    9001: [VIEWER[1], "runs/finebio-sam3-smoke-20260921/recording.rrd"],
    77: ["/usr/bin/ollama", "serve"],
}


def _probe(
    *,
    apps: Sequence[tuple[int, str, int | None]],
    used_mib: int,
    total_mib: int = TOTAL_MIB,
    cmdlines: dict[int, list[str]] | None = None,
    ppids: dict[int, int] | None = None,
    gpu_available: bool = True,
) -> GpuProbe:
    cmdlines = CMDLINES if cmdlines is None else cmdlines
    ppids = ppids or {}

    def nvidia_smi(arguments: Sequence[str]) -> str | None:
        if not gpu_available:
            return None
        if arguments[0].startswith("--query-gpu="):
            return f"0, NVIDIA GeForce RTX 5070 Ti, {total_mib}, {used_mib}\n"
        if not apps:
            return ""
        return "".join(
            f"{pid}, {name}, {'[N/A]' if mib is None else mib}\n" for pid, name, mib in apps
        )

    return GpuProbe(
        nvidia_smi=nvidia_smi,
        cmdline=lambda pid: cmdlines.get(pid),
        ppid=lambda pid: ppids.get(pid),
    )


def _decide(probe: GpuProbe, **kwargs) -> GuardDecision:
    kwargs.setdefault("profile", "sam3_1280")
    return evaluate(probe=probe, now=lambda: "2026-09-22T03:00:00+00:00", **kwargs)


def _classes(decision: GuardDecision) -> dict[int, str]:
    return {neighbour.pid: neighbour.neighbour_class for neighbour in decision.neighbours}


# --- classification -------------------------------------------------------------------------


def test_neighbours_are_classified_by_proc_cmdline_not_by_nvidia_smi_name() -> None:
    def classify(app: tuple[int, str, int | None]) -> gpu_guard.Neighbour:
        pid, name, mib = app
        return classify_neighbour(ComputeApp(pid, name, mib), CMDLINES.get(pid))

    kwin, calibration, game, tracker, player, viewer, ollama = map(
        classify, (KWIN, CALIBRATION, GAME, TRACKER, PLAYER, VIEWER, OLLAMA)
    )
    assert (kwin.neighbour_class, kwin.cmdline_basename) == ("known_benign", "kwin_wayland")
    # The calibration workspace worker holds a SAM3 model: a Battle worker even though nvidia-smi
    # only shows the interpreter path.
    assert calibration.neighbour_class == "own_repo_model"
    assert calibration.cmdline_basename == "muggled_calibration_worker.py"
    assert (game.neighbour_class, game.cmdline_basename) == ("known_benign", "SYNTHETIK.exe")
    assert game.reserved_mib == 0
    # The code-snapshot copy of the tracker worker matches by basename.
    assert tracker.neighbour_class == "own_repo_model"
    assert tracker.cmdline_basename == "muggled_worker.py"
    assert tracker.reserved_mib == 1677  # ceil(3354 x 0.5): a model process may grow
    # `showtime` is python3 to nvidia-smi; its cmdline names a media player.
    assert (player.neighbour_class, player.cmdline_basename) == ("known_benign", "showtime")
    assert viewer.neighbour_class == "known_benign" and viewer.cmdline_basename == "rerun"
    assert (ollama.neighbour_class, ollama.reserved_mib) == ("unknown", 1536)


def test_small_unknown_processes_are_benign_but_small_battle_workers_are_not() -> None:
    small = classify_neighbour(
        ComputeApp(5, "/opt/thing/bin/python3", 300), ["/opt/thing/bin/python3", "serve.py"]
    )
    assert small.neighbour_class == "known_benign" and "under 512 MiB" in small.why
    large = classify_neighbour(
        ComputeApp(6, "/opt/thing/bin/python3", 900), ["/opt/thing/bin/python3", "serve.py"]
    )
    assert large.neighbour_class == "unknown"
    resting_worker = classify_neighbour(
        ComputeApp(7, CALIBRATION_PYTHON, 200), [CALIBRATION_PYTHON, "wilor_worker.py"]
    )
    assert resting_worker.neighbour_class == "own_repo_model"
    # Unreadable cmdline (process gone, or not ours): size decides, the name is not trusted.
    unreadable_small = classify_neighbour(ComputeApp(8, "/usr/bin/python3", 100), None)
    unreadable_large = classify_neighbour(ComputeApp(9, "/usr/bin/python3", 2500), None)
    assert unreadable_small.neighbour_class == "known_benign"
    assert unreadable_large.neighbour_class == "unknown"
    assert "unreadable" in unreadable_large.why
    unsized = classify_neighbour(
        ComputeApp(10, "/usr/bin/python3", None), ["/usr/bin/python3", "x.py"]
    )
    assert unsized.neighbour_class == "unknown" and unsized.reserved_mib == 0


def test_program_basename_looks_past_interpreters_and_windows_paths() -> None:
    assert (
        gpu_guard.program_basename(["/usr/bin/python3", "/usr/bin/showtime", "a.mp4"]) == "showtime"
    )
    assert gpu_guard.program_basename(["/usr/bin/python3.12", "-u", "-m", "rerun"]) == "rerun"
    assert gpu_guard.program_basename(["S:\\steamapps\\common\\X\\GAME.exe"]) == "GAME.exe"
    assert gpu_guard.program_basename(["/usr/bin/kwin_wayland", "--x"]) == "kwin_wayland"
    assert gpu_guard.program_basename(["/usr/bin/python3"]) == "python3"
    assert gpu_guard.program_basename([]) is None


def test_nvidia_smi_csv_parsing_handles_commas_in_names_and_missing_memory() -> None:
    gpus = gpu_guard.parse_gpu_memory("0, NVIDIA GeForce RTX 5070 Ti, 16303, 7615\n")
    assert gpus == [gpu_guard.GpuMemory(0, "NVIDIA GeForce RTX 5070 Ti", 16303, 7615)]
    apps = gpu_guard.parse_compute_apps(
        "8429, /usr/bin/kwin_wayland, 144\n12, /opt/a, b/prog, [N/A]\nNo running processes found\n"
    )
    assert apps == [
        ComputeApp(8429, "/usr/bin/kwin_wayland", 144),
        ComputeApp(12, "/opt/a, b/prog", None),
    ]
    assert gpu_guard.parse_compute_apps(None) == [] and gpu_guard.parse_gpu_memory("") == []
    two = gpu_guard.parse_gpu_memory("0, A, 100, 1\n1, B, 200, 2\n")
    assert gpu_guard.select_gpu(two, visible_devices="1").name == "B"
    assert gpu_guard.select_gpu(two, visible_devices="").name == "A"


# --- headroom arithmetic --------------------------------------------------------------------


def test_headroom_is_total_minus_used_minus_neighbour_reservations() -> None:
    probe = _probe(apps=[KWIN, CALIBRATION, GAME], used_mib=7615)
    decision = _decide(probe, allowed_pids=[2071175])
    assert decision.accepted and decision.reasons == []
    headroom = decision.headroom
    assert headroom is not None
    # Only the model process reserves growth: ceil(1198 x 0.5) = 599; compositor and game 0.
    assert headroom["reserved_mib"] == 599
    assert headroom["headroom_mib"] == TOTAL_MIB - 7615 - 599 == 8089
    # sam3_1280 = 2.6 GiB = 2663 MiB; required = ceil(2663 x 1.5) = 3995.
    assert headroom["expected_peak_mib"] == 2663
    assert headroom["required_mib"] == 3995
    assert headroom["sufficient"] is True
    assert decision.expected_peak_source == "profile_default"
    assert decision.expected_peak_bytes == gpu_guard.EXPECTED_PEAK_BYTES["sam3_1280"]


def test_expected_peak_profiles_and_flag_override() -> None:
    gib = gpu_guard.GIB
    assert gpu_guard.EXPECTED_PEAK_BYTES == {
        "sam3_1280": int(2.6 * gib),
        "sam3_1080p": int(3.4 * gib),
        "four_part": int(3.4 * gib),
        "dam4sam_large": int(7.0 * gib),
        "unknown": int(4.0 * gib),
    }
    assert gpu_guard.expected_peak_bytes(None) == (int(4.0 * gib), "profile_default", "unknown")
    assert gpu_guard.expected_peak_bytes("sam3_1080p", 5 * gib) == (5 * gib, "flag", "sam3_1080p")
    with pytest.raises(ValueError, match="unknown GPU guard profile"):
        gpu_guard.expected_peak_bytes("sam9")
    with pytest.raises(ValueError, match="positive"):
        gpu_guard.expected_peak_bytes("sam3_1280", 0)
    assert gpu_guard.sam3_profile_for_side_length(504) == "sam3_1280"
    assert gpu_guard.sam3_profile_for_side_length(1280) == "sam3_1280"
    assert gpu_guard.sam3_profile_for_side_length(1920) == "sam3_1080p"
    assert gpu_guard.sam3_profile_for_side_length(2560) == "unknown"


def test_profile_is_inferred_from_a_queue_job_command_line() -> None:
    smoke = ["uv", "run", "battle-muggled-smoke", "--config", "c.json", "--max-side-length", "1280"]
    assert gpu_guard.infer_profile(smoke) == "sam3_1280"
    assert gpu_guard.infer_profile([*smoke[:-1], "1920"]) == "sam3_1080p"
    assert (
        gpu_guard.infer_profile(["uv", "run", "battle-muggled-smoke", "--view", "x"]) == "sam3_1280"
    )
    assert gpu_guard.infer_profile(["uv", "run", "battle-dam4sam-video", "--x"]) == "dam4sam_large"
    assert gpu_guard.infer_profile(["uv", "run", "battle-four-part-segmentation"]) == "four_part"
    assert (
        gpu_guard.infer_profile(["uv", "run", "battle-multiview-reprompt", "decode"]) == "unknown"
    )
    assert gpu_guard.allowed_pids_in_argv(
        [
            "x",
            "--allow-gpu-neighbour",
            "2071175",
            "--allow-gpu-neighbour=2158886",
            "--allow-gpu-neighbour",
        ]
    ) == [2071175, 2158886]


# --- the four decisions ----------------------------------------------------------------------


def test_another_battle_worker_refuses_unless_its_pid_is_allowed() -> None:
    probe = _probe(apps=[KWIN, CALIBRATION, GAME, TRACKER], used_mib=7615)
    refused = _decide(probe, allowed_pids=[2071175])
    assert not refused.accepted
    assert len(refused.reasons) == 1
    assert "another Battle GPU worker is running" in refused.reasons[0]
    assert "pid 2236310 (muggled_worker.py, 3354 MiB)" in refused.reasons[0]
    assert "2071175" not in refused.reasons[0], "the allowed calibration worker is not the reason"
    # The calibration workspace worker counts as a tracker: without its PID it refuses too.
    without = _decide(probe, allowed_pids=[2236310])
    assert not without.accepted and "muggled_calibration_worker.py" in without.reason
    # Both named: accepted, and the headroom charges both reservations.
    both = _decide(probe, allowed_pids=[2071175, 2236310])
    assert both.accepted
    assert both.headroom["reserved_mib"] == 599 + 1677
    assert both.headroom["headroom_mib"] == TOTAL_MIB - 7615 - 2276
    assert {n.pid for n in both.neighbours if n.allowed_by_operator} == {2071175, 2236310}


def test_short_headroom_refuses_and_says_the_arithmetic() -> None:
    # A 1080p run beside the calibration worker on a card with 12 GiB already in use.
    probe = _probe(apps=[KWIN, CALIBRATION], used_mib=12300)
    decision = _decide(probe, allowed_pids=[2071175], profile="sam3_1080p")
    assert not decision.accepted
    assert decision.headroom["headroom_mib"] == TOTAL_MIB - 12300 - 599 == 3404
    assert decision.headroom["required_mib"] == 5223  # ceil(3482 x 1.5)
    assert decision.headroom["sufficient"] is False
    [reason] = decision.reasons
    assert reason == (
        "VRAM headroom 3404 MiB (total 16303 - used 12300 + own 0 - reserved 599) is below the "
        "required 5223 MiB (1.5 x expected peak 3482 MiB, profile sam3_1080p)"
    )
    # The same card with a smaller job passes.
    assert _decide(
        probe, allowed_pids=[2071175], expected_peak_vram_bytes=2 * gpu_guard.GIB
    ).accepted


def test_a_large_unknown_neighbour_refuses_while_a_smaller_one_only_reserves() -> None:
    probe = _probe(apps=[KWIN, OLLAMA], used_mib=3400)
    decision = _decide(probe)
    assert not decision.accepted
    [reason] = decision.reasons
    assert reason == "unknown GPU neighbour above 2048 MiB: pid 77 (ollama, 3072 MiB)"
    smaller = _probe(apps=[KWIN, (77, "/usr/bin/ollama", 1500)], used_mib=1800)
    accepted = _decide(smaller)
    assert accepted.accepted
    assert _classes(accepted)[77] == "unknown"
    assert accepted.headroom["reserved_mib"] == 750


def test_benign_neighbours_that_blocked_the_old_guard_are_accepted() -> None:
    # The Sep 22 02:03 UTC case: the calibration worker (allowed), the video player nvidia-smi
    # calls python3, a Steam game and the compositor; all fit on the card.
    probe = _probe(apps=[KWIN, CALIBRATION, GAME, PLAYER, VIEWER], used_mib=4200)
    decision = _decide(probe, allowed_pids=[2071175])
    assert decision.accepted
    assert _classes(decision) == {
        8429: "known_benign",
        2071175: "own_repo_model",
        2202800: "known_benign",
        2158886: "known_benign",
        9001: "known_benign",
    }
    assert decision.headroom["reserved_mib"] == 599
    assert decision.headroom["headroom_mib"] == TOTAL_MIB - 4200 - 599
    # The strict rule refused the same card because of the player.
    strict = _decide(probe, allowed_pids=[2071175], mode="strict")
    assert not strict.accepted
    assert "2158886" in strict.reason and "showtime" not in strict.reason


def test_own_processes_are_excluded_from_the_rules_and_added_back_to_headroom() -> None:
    # A queue re-check 30 s into a job: the job's worker (pid 2236310) descends from the job
    # process (pid 400) through `uv run` (pid 450); the calibration worker does not.
    ppids = {2236310: 450, 450: 400, 400: 300, 300: 1, 2071175: 2000, 2000: 1}
    probe = _probe(apps=[KWIN, CALIBRATION, TRACKER], used_mib=7615, ppids=ppids)
    recheck = _decide(probe, allowed_pids=[2071175], own_pid=400)
    assert recheck.accepted
    assert [own.pid for own in recheck.own_processes] == [2236310]
    assert recheck.headroom["own_mib"] == 3354
    assert recheck.headroom["headroom_mib"] == TOTAL_MIB - 7615 + 3354 - 599
    before = _decide(_probe(apps=[KWIN, CALIBRATION], used_mib=4200), allowed_pids=[2071175])
    after_probe = _probe(apps=[KWIN, CALIBRATION, TRACKER, PLAYER], used_mib=8000, ppids=ppids)
    after = _decide(after_probe, allowed_pids=[2071175], own_pid=400)
    assert [n.pid for n in gpu_guard.new_neighbours(before, after)] == [2158886]
    assert not gpu_guard.is_descendant_of(2071175, 400, ppids.get)
    assert gpu_guard.is_descendant_of(400, 400, ppids.get)


def test_without_nvidia_smi_the_guard_records_that_it_could_not_look() -> None:
    decision = _decide(_probe(apps=[], used_mib=0, gpu_available=False))
    assert decision.accepted and decision.headroom is None
    assert decision.nvidia_smi_available is False
    assert decision.neighbours == [] and any(
        "nvidia-smi unavailable" in note for note in decision.notes
    )
    with pytest.raises(ValueError, match="unknown GPU guard mode"):
        _decide(_probe(apps=[], used_mib=0), mode="lenient")


# --- strict mode is the Sep 21 function ---------------------------------------------------------


def _legacy_looks_like_model_process(process: dict[str, str]) -> bool:
    """`muggled_worker._looks_like_model_process` as committed in 88ba96e, verbatim."""
    name = process["process_name"].lower()
    executable = name.rsplit("/", 1)[-1]
    if executable == "rerun" or "/rerun_sdk/rerun_cli/" in name:
        return False
    return any(token in name for token in ("python", "torch", "ollama", "llama", "vllm"))


def _legacy_partition(gpu_processes, allowed_pids):
    allowed = {str(int(pid)) for pid in allowed_pids}
    blocking, tolerated = [], []
    for process in gpu_processes:
        if not _legacy_looks_like_model_process(process):
            continue
        (tolerated if process["pid"] in allowed else blocking).append(process)
    return blocking, tolerated


def test_strict_mode_matches_the_sep_21_function_name_for_name() -> None:
    names = [
        KWIN[1], CALIBRATION_PYTHON, GAME[1], "/usr/bin/python3", VIEWER[1], "/usr/local/bin/rerun",
        "/usr/bin/ollama", "/opt/llama.cpp/server", "/opt/vllm/bin/vllm", "/usr/lib/libtorch-thing",
        "/opt/rerun-experiments/.venv/bin/python3", "/home/nick/.pyenv/versions/x/bin/Python",
        "", "rerun", "/x/rerun_sdk/rerun_cli/rerun-bin",
    ]  # fmt: skip
    records = [
        {"pid": str(index), "process_name": name, "memory": "1 MiB"}
        for index, name in enumerate(names)
    ]
    for record in records:
        assert looks_like_model_process(record) == _legacy_looks_like_model_process(record), record
    assert gpu_guard.partition_neighbours_strict(records, [1, 6]) == _legacy_partition(
        records, [1, 6]
    )
    assert gpu_guard.partition_neighbours_strict(records, []) == _legacy_partition(records, [])

    # The worker's public names still resolve to the same rule.
    from battle.muggled_worker import _looks_like_model_process, partition_gpu_neighbours

    for record in records:
        assert _looks_like_model_process(record) == _legacy_looks_like_model_process(record)
    assert partition_gpu_neighbours(records, [1]) == _legacy_partition(records, [1])


def test_worker_strict_mode_writes_the_sep_21_reason_and_record(monkeypatch) -> None:
    from battle.muggled_worker import build_parser, gpu_guard_decision

    base = [
        "--run-directory", "/tmp/x", "--video", "v.mp4", "--view-id", "static-c10119",
        "--source-offset-seconds", "0", "--model", "m.pt",
    ]  # fmt: skip
    processes = gpu_guard.legacy_process_records(
        [ComputeApp(*KWIN), ComputeApp(*CALIBRATION), ComputeApp(*PLAYER)]
    )
    strict = build_parser().parse_args(
        [*base, "--gpu-guard", "strict", "--allow-gpu-neighbour", "2071175"]
    )
    reason, record = gpu_guard_decision(strict, processes)
    player = {"pid": "2158886", "process_name": "/usr/bin/python3", "memory": "389 MiB"}
    assert reason == f"concurrent GPU model process(es) detected: {[player]}"
    assert record == {
        "allowed_neighbour_pids": [2071175],
        "tolerated_neighbours": [
            {"pid": "2071175", "process_name": CALIBRATION_PYTHON, "memory": "1198 MiB"}
        ],
        "semantics": (
            "the operator allowed these GPU processes (by PID) to run beside this worker; "
            "gpu_peak_vram_bytes stays this process's own allocation but wall-clock timing "
            "may see contention; the guard still refuses any other model process"
        ),
    }
    # Without operator-named PIDs strict mode writes no record at all, as before.
    assert (
        gpu_guard_decision(build_parser().parse_args([*base, "--gpu-guard", "strict"]), [])[1]
        is None
    )

    # The default is vram: the same card is accepted and the full provenance is recorded.
    monkeypatch.setattr(
        gpu_guard, "DEFAULT_PROBE", _probe(apps=[KWIN, CALIBRATION, PLAYER], used_mib=4200)
    )
    vram_args = build_parser().parse_args(
        [*base, "--allow-gpu-neighbour", "2071175", "--max-side-length", "1280"]
    )
    assert vram_args.gpu_guard == "vram" and vram_args.gpu_guard_profile is None
    reason, record = gpu_guard_decision(vram_args, processes)
    assert reason is None
    assert record["schema"] == "battle-gpu-guard/1" and record["accepted"] is True
    assert record["expected_peak"] == {
        "bytes": int(2.6 * gpu_guard.GIB),
        "mib": 2663,
        "source": "profile_default",
        "profile": "sam3_1280",
    }
    assert [n["class"] for n in record["neighbours"]] == [
        "known_benign",
        "own_repo_model",
        "known_benign",
    ]
    peak_args = build_parser().parse_args(
        [*base, "--gpu-guard-profile", "sam3_1080p", "--expected-peak-vram-bytes", "3000000000"]
    )
    _reason, record = gpu_guard_decision(peak_args, [])
    assert (
        record["expected_peak"]["source"] == "flag"
        and record["expected_peak"]["bytes"] == 3000000000
    )
    assert record["expected_peak"]["profile"] == "sam3_1080p"


# --- provenance ----------------------------------------------------------------------------------


def test_provenance_round_trips_through_json_and_names_every_neighbour() -> None:
    probe = _probe(apps=[KWIN, CALIBRATION, GAME, PLAYER, OLLAMA], used_mib=9000)
    decision = _decide(probe, allowed_pids=[2071175, 4242])
    payload = decision.as_provenance()
    assert set(payload) == {
        "schema", "mode", "checked_at", "accepted", "reasons", "nvidia_smi_available", "gpu",
        "allowed_neighbour_pids", "allowed_neighbour_pids_absent", "neighbours", "own_processes",
        "expected_peak", "headroom", "rules", "notes", "semantics",
    }  # fmt: skip
    assert payload["schema"] == "battle-gpu-guard/1" and payload["mode"] == "vram"
    assert payload["checked_at"] == "2026-09-22T03:00:00+00:00"
    assert payload["gpu"] == {
        "index": 0,
        "name": "NVIDIA GeForce RTX 5070 Ti",
        "total_mib": 16303,
        "used_mib": 9000,
    }
    assert payload["allowed_neighbour_pids"] == [2071175, 4242]
    assert payload["allowed_neighbour_pids_absent"] == [4242]
    assert [n["pid"] for n in payload["neighbours"]] == [8429, 2071175, 2202800, 2158886, 77]
    assert payload["neighbours"][1] == {
        "pid": 2071175,
        "class": "own_repo_model",
        "used_mib": 1198,
        "reserved_mib": 599,
        "cmdline_basename": "muggled_calibration_worker.py",
        "process_name": CALIBRATION_PYTHON,
        "allowed_by_operator": True,
        "why": "cmdline names muggled_calibration_worker.py",
    }
    assert payload["rules"] == {
        "headroom_factor": 1.5,
        "small_neighbour_mib": 512,
        "unknown_neighbour_limit_mib": 2048,
        "growth_fraction": {"own_repo_model": 0.5, "unknown": 0.5, "known_benign": 0.0},
    }
    assert payload["accepted"] is False and payload["reasons"] == [
        "unknown GPU neighbour above 2048 MiB: pid 77 (ollama, 3072 MiB)"
    ]
    restored = GuardDecision.from_provenance(json.loads(json.dumps(payload, sort_keys=True)))
    assert restored == decision
    assert restored.as_provenance() == payload
    with pytest.raises(ValueError, match="battle-gpu-guard/1"):
        GuardDecision.from_provenance({"schema": "other"})


def test_worker_and_smoke_parsers_take_the_guard_flags(tmp_path) -> None:
    from battle import muggled_smoke
    from battle.muggled_worker import build_parser

    base = [
        "--run-directory", str(tmp_path), "--video", "v.mp4", "--view-id", "static-c10119",
        "--source-offset-seconds", "0", "--model", "m.pt",
    ]  # fmt: skip
    worker = build_parser().parse_args(base)
    assert (worker.gpu_guard, worker.gpu_guard_profile, worker.expected_peak_vram_bytes) == (
        "vram",
        None,
        None,
    )
    with pytest.raises(SystemExit):
        build_parser().parse_args([*base, "--gpu-guard", "off"])

    smoke_args = argparse.Namespace(
        gpu_guard="vram",
        gpu_guard_profile=None,
        expected_peak_vram_bytes=None,
        max_side_length=1920,
    )
    assert muggled_smoke._gpu_guard_settings(smoke_args) == ("sam3_1080p", int(3.4 * gpu_guard.GIB))
    smoke_args.expected_peak_vram_bytes = 4_000_000_000
    assert muggled_smoke._gpu_guard_settings(smoke_args) == ("sam3_1080p", 4_000_000_000)
    smoke_args.gpu_guard = "strict"
    assert muggled_smoke._gpu_guard_settings(smoke_args) == (None, None)


def test_legacy_gpu_processes_are_the_sep_21_records_the_scripts_read() -> None:
    probe = _probe(apps=[KWIN, CALIBRATION, (4242, "/usr/bin/python3", None)], used_mib=3000)

    records = gpu_guard.legacy_gpu_processes(probe)

    assert records == [
        {"pid": "8429", "process_name": "/usr/bin/kwin_wayland", "memory": "144 MiB"},
        {"pid": "2071175", "process_name": CALIBRATION_PYTHON, "memory": "1198 MiB"},
        {"pid": "4242", "process_name": "/usr/bin/python3", "memory": "[N/A]"},
    ]
    blocking, tolerated = gpu_guard.partition_neighbours_strict(records, [2071175])
    assert [record["pid"] for record in blocking] == ["4242"]
    assert [record["pid"] for record in tolerated] == ["2071175"]
    assert gpu_guard.legacy_gpu_processes(_probe(apps=[], used_mib=0, gpu_available=False)) == []


def test_module_is_python_310_compatible_for_the_foreign_interpreters() -> None:
    import ast

    tree = ast.parse(Path(gpu_guard.__file__).read_text(encoding="utf-8"))
    imported_from_datetime = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "datetime"
        for alias in node.names
    }
    assert "UTC" not in imported_from_datetime
    assert not any(
        isinstance(node, ast.Attribute) and node.attr == "UTC" for node in ast.walk(tree)
    )
