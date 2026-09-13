from __future__ import annotations

from battle.g3_contact_sheet import TARGET_TIMES_SECONDS, review_frame_indices


def test_g3_contact_sheet_uses_fixed_smoke_timestamps() -> None:
    assert TARGET_TIMES_SECONDS == (0.0, 5.0, 299 / 30)
    assert review_frame_indices() == (0, 150, 299)
