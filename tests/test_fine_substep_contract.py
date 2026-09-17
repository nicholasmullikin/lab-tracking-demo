from __future__ import annotations

from pathlib import Path

import pytest

from battle.fine_substep_contract import load_contract, substep_for_frame


def test_agent_label_contract_partitions_first_20s() -> None:
    contract = load_contract(
        Path("configs/fine_substeps/assembly101_focused_static_first_20s_agent_labels.json")
    )
    assert contract.provenance_tag == "agent_authored_visual_review"
    assert len(contract.substeps) == 11
    assert contract.substeps[0].substep_id == "S01"
    assert contract.substeps[-1].end_frame_exclusive == 600
    assert substep_for_frame(contract, 0).substep_id == "S01"
    assert substep_for_frame(contract, 404).substep_id == "S07"
    assert substep_for_frame(contract, 405).substep_id == "S08"


def test_agent_label_contract_rejects_bad_provenance(tmp_path: Path) -> None:
    payload = Path(
        "configs/fine_substeps/assembly101_focused_static_first_20s_agent_labels.json"
    ).read_text()
    bad = payload.replace("agent_authored_visual_review", "assembly101_gt")
    path = tmp_path / "bad.json"
    path.write_text(bad)
    with pytest.raises(Exception):
        load_contract(path)
