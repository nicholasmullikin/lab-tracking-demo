"""Read ZIP/ZIP64 central directories over HTTP Range without downloading archives."""

from __future__ import annotations

import struct
import zlib
from collections.abc import Iterable
from dataclasses import dataclass

EOCD_SIGNATURE = 0x06054B50
ZIP64_EOCD_LOCATOR_SIGNATURE = 0x07064B50
ZIP64_EOCD_SIGNATURE = 0x06064B50
CENTRAL_DIR_SIGNATURE = 0x02014B50
LOCAL_HEADER_SIGNATURE = 0x04034B50
MAX_COMMENT_LENGTH = 65_535
DEFAULT_TAIL_BYTES = 256 * 1024


@dataclass(frozen=True)
class ZipMember:
    filename: str
    compressed_size: int
    uncompressed_size: int
    compression_method: int
    local_header_offset: int
    crc32: int

    @property
    def data_offset(self) -> int:
        return self.local_header_offset


class ZipRangeError(RuntimeError):
    pass


class RangeReader:
    """Fetch byte ranges from a remote object that supports HTTP Range requests."""

    def __init__(self, url: str, *, size: int, headers: dict[str, str] | None = None) -> None:
        self.url = url
        self.size = size
        self.headers = headers or {}

    def read_range(self, start: int, end_exclusive: int) -> bytes:
        import urllib.request

        if start < 0 or end_exclusive > self.size or start >= end_exclusive:
            raise ZipRangeError(f"invalid range [{start}, {end_exclusive}) for size {self.size}")
        request = urllib.request.Request(
            self.url,
            headers={**self.headers, "Range": f"bytes={start}-{end_exclusive - 1}"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            status = getattr(response, "status", response.getcode())
            if status not in (200, 206):
                raise ZipRangeError(f"unexpected HTTP status {status} for range request")
            return response.read()


def _parse_eocd(data: bytes) -> tuple[int, int, int]:
    if len(data) < 22:
        raise ZipRangeError("EOCD record too short")
    (
        signature,
        _disk_no,
        _central_dir_disk,
        entries_on_disk,
        entries_total,
        central_size,
        central_offset,
    ) = struct.unpack_from("<IHHHHII", data, 0)
    if signature != EOCD_SIGNATURE:
        raise ZipRangeError("EOCD signature not found")
    comment_length = struct.unpack_from("<H", data, 20)[0]
    if len(data) < 22 + comment_length:
        raise ZipRangeError("EOCD comment truncated")
    return central_offset, entries_on_disk or entries_total, central_size


def _parse_zip64_eocd(data: bytes) -> tuple[int, int, int]:
    if len(data) < 56 or struct.unpack_from("<I", data, 0)[0] != ZIP64_EOCD_SIGNATURE:
        raise ZipRangeError("ZIP64 EOCD signature not found")
    (
        _signature,
        _record_size,
        _version_made,
        _version_needed,
        _disk_number,
        _central_dir_disk,
        entries_on_disk,
        entries_total,
        central_size,
        central_offset,
    ) = struct.unpack_from("<IQHHIIQQQQ", data, 0)
    return central_offset, entries_on_disk or entries_total, central_size


def _find_eocd(data: bytes) -> bytes:
    index = data.rfind(struct.pack("<I", EOCD_SIGNATURE))
    if index < 0:
        raise ZipRangeError("EOCD signature not found in tail window")
    comment_length = struct.unpack_from("<H", data, index + 20)[0]
    record_end = index + 22 + comment_length
    if record_end > len(data):
        raise ZipRangeError("EOCD record extends past downloaded tail window")
    return data[index:record_end]


def _central_directory_bounds(reader: RangeReader) -> tuple[int, int]:
    tail_size = min(reader.size, DEFAULT_TAIL_BYTES)
    tail = reader.read_range(reader.size - tail_size, reader.size)
    eocd = _find_eocd(tail)
    central_offset, entry_count, central_size = _parse_eocd(eocd)
    if central_offset == 0xFFFFFFFF or central_size == 0xFFFFFFFF or entry_count == 0xFFFF:
        locator_index = tail.rfind(struct.pack("<I", ZIP64_EOCD_LOCATOR_SIGNATURE))
        if locator_index < 0:
            raise ZipRangeError("ZIP64 locator missing for oversized archive")
        zip64_offset = struct.unpack_from("<Q", tail, locator_index + 8)[0]
        zip64_end = min(zip64_offset + 128, reader.size)
        zip64 = reader.read_range(zip64_offset, zip64_end)
        central_offset, entry_count, central_size = _parse_zip64_eocd(zip64)
    end = central_offset + central_size
    if central_offset < 0 or end > reader.size:
        raise ZipRangeError(
            f"central directory [{central_offset}, {end}) outside archive size {reader.size}"
        )
    return central_offset, end


def _decode_filename(raw_name: bytes, flag_bits: int) -> str:
    if flag_bits & 0x800:
        return raw_name.decode("utf-8")
    return raw_name.decode("cp437")


def _parse_central_directory(data: bytes) -> list[ZipMember]:
    members: list[ZipMember] = []
    offset = 0
    while offset + 4 <= len(data):
        signature = struct.unpack_from("<I", data, offset)[0]
        if signature != CENTRAL_DIR_SIGNATURE:
            break
        if offset + 46 > len(data):
            raise ZipRangeError("truncated central directory header")
        (
            _version_made,
            _version_needed,
            flag_bits,
            compression_method,
            _mod_time,
            _mod_date,
            crc32,
            compressed_size,
            uncompressed_size,
            filename_length,
            extra_length,
            comment_length,
            _disk_start,
            _internal_attr,
            _external_attr,
            local_header_offset,
        ) = struct.unpack_from("<HHHHHHIIIHHHHHII", data, offset + 4)
        name_start = offset + 46
        name_end = name_start + filename_length
        extra_end = name_end + extra_length
        record_end = extra_end + comment_length
        if record_end > len(data):
            raise ZipRangeError("truncated central directory filename/extra/comment")
        filename = _decode_filename(data[name_start:name_end], flag_bits)
        if (
            compressed_size == 0xFFFFFFFF
            or uncompressed_size == 0xFFFFFFFF
            or local_header_offset == 0xFFFFFFFF
        ):
            extra = data[name_end:extra_end]
            zip64_index = extra.find(b"\x01\x00")
            if zip64_index < 0:
                raise ZipRangeError(f"ZIP64 extra field missing for {filename}")
            zip64_data_size = struct.unpack_from("<H", extra, zip64_index + 2)[0]
            zip64_values = struct.unpack_from(
                f"<{'Q' * (zip64_data_size // 8)}",
                extra,
                zip64_index + 4,
            )
            value_index = 0
            if uncompressed_size == 0xFFFFFFFF:
                uncompressed_size = zip64_values[value_index]
                value_index += 1
            if compressed_size == 0xFFFFFFFF:
                compressed_size = zip64_values[value_index]
                value_index += 1
            if local_header_offset == 0xFFFFFFFF:
                local_header_offset = zip64_values[value_index]
                value_index += 1
        members.append(
            ZipMember(
                filename=filename,
                compressed_size=compressed_size,
                uncompressed_size=uncompressed_size,
                compression_method=compression_method,
                local_header_offset=local_header_offset,
                crc32=crc32,
            )
        )
        offset = record_end
    return members


def list_remote_zip_members(reader: RangeReader) -> list[ZipMember]:
    start, end = _central_directory_bounds(reader)
    return _parse_central_directory(reader.read_range(start, end))


def _local_header_data_offset(reader: RangeReader, member: ZipMember) -> int:
    header = reader.read_range(member.local_header_offset, member.local_header_offset + 30)
    if struct.unpack_from("<I", header, 0)[0] != LOCAL_HEADER_SIGNATURE:
        raise ZipRangeError(f"local header missing for {member.filename}")
    filename_length, extra_length = struct.unpack_from("<HH", header, 26)
    return member.local_header_offset + 30 + filename_length + extra_length


def read_remote_zip_member(reader: RangeReader, member: ZipMember) -> bytes:
    if member.compression_method not in (0, 8):
        raise ZipRangeError(
            f"unsupported compression method {member.compression_method} for {member.filename}"
        )
    data_start = _local_header_data_offset(reader, member)
    payload = reader.read_range(data_start, data_start + member.compressed_size)
    if member.compression_method == 0:
        raw = payload
    else:
        raw = zlib.decompress(payload, -zlib.MAX_WBITS)
    if len(raw) != member.uncompressed_size:
        raise ZipRangeError(
            f"decompressed size mismatch for {member.filename}: "
            f"expected {member.uncompressed_size}, got {len(raw)}"
        )
    verified = zlib.crc32(raw) & 0xFFFFFFFF
    if verified != member.crc32:
        raise ZipRangeError(f"CRC mismatch for {member.filename}")
    return raw


def filter_members(members: Iterable[ZipMember], patterns: Iterable[str]) -> list[ZipMember]:
    lowered = [pattern.lower() for pattern in patterns]
    return [
        member
        for member in members
        if any(pattern in member.filename.lower() for pattern in lowered)
    ]
