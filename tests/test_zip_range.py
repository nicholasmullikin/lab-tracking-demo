from __future__ import annotations

import io
import zipfile

from battle.zip_range import RangeReader, list_remote_zip_members, read_remote_zip_member


def _build_zip_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("alpha/hello.txt", "hello world")
        archive.writestr("beta/values.json", '{"value": 42}')
    return buffer.getvalue()


class _MemoryRangeReader(RangeReader):
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        super().__init__("memory://zip", size=len(payload))

    def read_range(self, start: int, end_exclusive: int) -> bytes:
        return self._payload[start:end_exclusive]


def test_list_remote_zip_members_reads_central_directory() -> None:
    payload = _build_zip_bytes()
    members = list_remote_zip_members(_MemoryRangeReader(payload))
    names = {member.filename for member in members}
    assert names == {"alpha/hello.txt", "beta/values.json"}


def test_read_remote_zip_member_decompresses_payload() -> None:
    payload = _build_zip_bytes()
    reader = _MemoryRangeReader(payload)
    member = next(
        member
        for member in list_remote_zip_members(reader)
        if member.filename.endswith("json")
    )
    raw = read_remote_zip_member(reader, member)
    assert raw == b'{"value": 42}'
