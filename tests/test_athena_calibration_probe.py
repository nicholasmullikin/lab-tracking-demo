from __future__ import annotations

import json
from pathlib import Path

from battle.athena_calibration_probe import probe_archive


def test_probe_archive_writes_member_inventory(tmp_path: Path) -> None:
    output_path = tmp_path / "probe.json"
    payload = probe_archive(
        repo="cvml-nus/assembly101",
        revision="bfc15ea5e3f0bc8f8c232af6c1b45aa137a9d967",
        archive="AssemblyPoses.zip",
        recording="nusar-2021_action_both_9033-c02a_9033_user_id_2021-02-04_140532",
        output_path=output_path,
        extract=False,
        extract_dir=tmp_path / "extract",
    )
    assert output_path.is_file()
    assert payload["range_requests_supported"] is True
    assert payload["member_count_total"] > 0
    assert payload["member_count_calibration_like_for_recording"] >= 5
    calibration_names = payload["calibration_like_members_for_recording"]
    assert not any("intrinsic" in name.lower() for name in calibration_names)
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["archive_size_bytes"] > 70_000_000_000
