"""Resume boundaries must preserve one observation per native frame."""

import json

import pytest

from battle import pipette_multiplex_run as run


def study(tmp_path, monkeypatch):
    schedule = {
        "seeds": [],
        "corrections": [],
        "memory_semantics": "append_prompt_memory_and_reset_frame_memory",
    }
    run.write_json(
        tmp_path / "config.json",
        {
            "reviewed_seeds": True,
            "views": {
                "T2": {"video": str(tmp_path / "video.mp4"), "fps": 30, "schedule": schedule}
            },
        },
    )
    commands = []

    def launch(command, directory):
        directory.mkdir(parents=True, exist_ok=True)
        run.write_json(directory / "worker_command.json", command)
        commands.append(command)

    monkeypatch.setattr(run, "launch", launch)
    run.track(tmp_path, 300, "baseline", ["T2"])
    directory = tmp_path / "baseline/T2"
    checkpoints = directory / "native/checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "f000299.pt").touch()
    (directory / "observations.jsonl").write_text(
        "".join(json.dumps({"analysis_frame_index": frame}) + "\n" for frame in range(300))
    )
    run.write_json(directory / "worker_result.json", {"state": "succeeded", "frames": 300})
    return directory, commands


def test_resume_reprocesses_checkpoint_frame_without_duplicate_and_archives_tail(
    tmp_path, monkeypatch
):
    directory, commands = study(tmp_path, monkeypatch)
    run.track(tmp_path, 900, "baseline", ["T2"])
    kept = [
        json.loads(line) for line in (directory / "observations.jsonl").read_text().splitlines()
    ]
    assert [row["analysis_frame_index"] for row in kept] == list(range(299))
    assert (
        json.loads((directory / "stage-300-tail.jsonl").read_text())["analysis_frame_index"] == 299
    )
    command = commands[-1]
    assert command[command.index("--resume-from-checkpoint") + 1].endswith("f000299.pt")


def test_incompatible_resume_preserves_existing_observations(tmp_path, monkeypatch):
    directory, commands = study(tmp_path, monkeypatch)
    observations = directory / "observations.jsonl"
    original = observations.read_bytes()
    config = json.loads((tmp_path / "config.json").read_text())
    config["views"]["T2"]["schedule"]["corrections"] = [{"frame_index": 100}]
    run.write_json(tmp_path / "config.json", config)
    with pytest.raises(ValueError, match="Resume inputs differ"):
        run.track(tmp_path, 900, "baseline", ["T2"])
    assert observations.read_bytes() == original
    assert len(commands) == 1


def test_completed_stage_is_not_replayed(tmp_path, monkeypatch):
    directory, commands = study(tmp_path, monkeypatch)
    original = (directory / "observations.jsonl").read_bytes()
    run.track(tmp_path, 300, "baseline", ["T2"])
    assert len(commands) == 1
    assert (directory / "observations.jsonl").read_bytes() == original


def test_geometry_rejects_changed_inputs_before_touching_cache(tmp_path, monkeypatch):
    from battle import pipette_multiplex_review as review

    run.write_json(tmp_path / "config.json", {})
    folder = tmp_path / "baseline"
    folder.mkdir()
    run.write_json(folder / "geometry-inputs.json", {"calibration": "old"})
    cache = folder / "geometry.jsonl"
    cache.write_text('{"raw_frame":0}\n')
    monkeypatch.setattr(review, "geometry_inputs", lambda *args: {"calibration": "changed"})
    with pytest.raises(ValueError, match="Geometry cache inputs changed"):
        review.analyze(tmp_path, "baseline", 1)
    assert cache.read_text() == '{"raw_frame":0}\n'


def test_temporal_vote_ignores_fit_endpoint_sign_and_preserves_missing_fit(tmp_path):
    from battle.pipette_multiplex_review import temporal

    run.write_json(tmp_path / "policy-review.json", {"selected_mask_policy": "baseline"})
    folder = tmp_path / "baseline"
    folder.mkdir()
    records = []
    for frame, reversed_ends in enumerate((False, True, None, True)):
        line = (
            None
            if reversed_ends is None
            else {
                "endpoints": [[1, 0, 0], [-1, 0, 0]] if reversed_ends else [[-1, 0, 0], [1, 0, 0]],
            }
        )
        records.append(
            {
                "raw_frame": frame,
                "blue": {
                    "line": line,
                    "direction": {
                        "tip_end": None if line is None else 0 if reversed_ends else 1,
                        "log_odds": 0 if line is None else 4 if reversed_ends else -4,
                    },
                },
            }
        )
    (folder / "geometry.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    temporal(tmp_path, "baseline")
    rows = [
        json.loads(row) for row in (folder / "orientation-temporal.jsonl").read_text().splitlines()
    ]
    assert [r["online_tip_end"] for r in rows] == [1, 0, None, 0]
    assert [r["retrofit_tip_end"] for r in rows] == [1, 0, None, 0]
    assert rows[3]["episode"] == 1
    summary = json.loads((folder / "orientation-temporal-summary.json").read_text())
    assert set(summary["adjacent_within_episode_sign_changes"].values()) == {0}
