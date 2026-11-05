"""Minimal, dependency-free reader for classic libpcap files.

Only what the challenge needs: iterate packets, strip the link layer, IPv4 and
UDP headers, and yield UDP payloads. It replaces Scapy in the active pipeline
(Scapy is still used by the archived inspection scripts), which removes a heavy
dependency and makes truncated records explicit instead of a printed warning.

Format reference: https://wiki.wireshark.org/Development/LibpcapFileFormat
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# Magic numbers: microsecond / nanosecond resolution, both byte orders.
_MAGIC = {
    b"\xd4\xc3\xb2\xa1": ("<", 1e-6),
    b"\xa1\xb2\xc3\xd4": (">", 1e-6),
    b"\x4d\x3c\xb2\xa1": ("<", 1e-9),
    b"\xa1\xb2\x3c\x4d": (">", 1e-9),
}

LINKTYPE_ETHERNET = 1
LINKTYPE_RAW = 101
LINKTYPE_LINUX_SLL = 113
LINKTYPE_IPV4 = 228


@dataclass
class PcapStats:
    records: int = 0
    truncated: int = 0
    non_udp: int = 0
    udp: int = 0
    notes: list = field(default_factory=list)


@dataclass
class UdpPacket:
    index: int          # record index in file order (preserves temporal order)
    timestamp: float    # seconds since epoch from the record header
    src_port: int
    dst_port: int
    payload: bytes


def _ip_offset(linktype: int, frame: bytes) -> int | None:
    """Return the byte offset of the IPv4 header, or None if not IPv4."""
    if linktype == LINKTYPE_ETHERNET:
        if len(frame) < 14:
            return None
        offset, ethertype = 14, struct.unpack_from(">H", frame, 12)[0]
        while ethertype in (0x8100, 0x88A8) and len(frame) >= offset + 4:  # VLAN tags
            ethertype = struct.unpack_from(">H", frame, offset + 2)[0]
            offset += 4
        return offset if ethertype == 0x0800 else None
    if linktype == LINKTYPE_LINUX_SLL:
        if len(frame) < 16:
            return None
        return 16 if struct.unpack_from(">H", frame, 14)[0] == 0x0800 else None
    if linktype in (LINKTYPE_RAW, LINKTYPE_IPV4):
        return 0 if frame and (frame[0] >> 4) == 4 else None
    raise ValueError(f"Unsupported pcap link type {linktype}")


def _udp_payload(frame: bytes, ip: int):
    if len(frame) < ip + 20:
        return None
    ihl = (frame[ip] & 0x0F) * 4
    protocol = frame[ip + 9]
    if protocol != 17 or ihl < 20:
        return None
    udp = ip + ihl
    if len(frame) < udp + 8:
        return None
    src, dst, length = struct.unpack_from(">HHH", frame, udp)
    end = udp + length if length >= 8 else len(frame)
    return src, dst, frame[udp + 8:min(end, len(frame))]


def iter_udp_payloads(path: str | Path, stats: PcapStats | None = None) -> Iterator[UdpPacket]:
    """Yield UDP payloads in file order. Truncated records are counted and skipped."""
    stats = stats if stats is not None else PcapStats()
    with open(path, "rb") as handle:
        header = handle.read(24)
        if len(header) < 24:
            raise ValueError(f"{path}: not a pcap file (too short)")
        if header[:4] == b"\x0a\x0d\x0d\x0a":
            raise ValueError(f"{path}: pcapng is not supported; convert with editcap -F pcap")
        if header[:4] not in _MAGIC:
            raise ValueError(f"{path}: unknown pcap magic {header[:4].hex()}")
        endian, resolution = _MAGIC[header[:4]]
        linktype = struct.unpack_from(endian + "I", header, 20)[0] & 0x0FFFFFFF
        record_fmt = endian + "IIII"

        index = 0
        while True:
            record = handle.read(16)
            if not record:
                break
            if len(record) < 16:
                stats.truncated += 1
                stats.notes.append("partial record header at end of file")
                break
            ts_sec, ts_frac, caplen, origlen = struct.unpack(record_fmt, record)
            frame = handle.read(caplen)
            stats.records += 1
            index += 1
            if len(frame) < caplen or caplen < origlen:
                # Snap-length cut or file ended mid-record: payload is incomplete.
                stats.truncated += 1
                continue
            ip = _ip_offset(linktype, frame)
            parsed = _udp_payload(frame, ip) if ip is not None else None
            if parsed is None:
                stats.non_udp += 1
                continue
            stats.udp += 1
            src, dst, payload = parsed
            yield UdpPacket(index - 1, ts_sec + ts_frac * resolution, src, dst, payload)


def write_pcap(path: str | Path, frames: list[bytes], timestamps: list[float] | None = None) -> None:
    """Write Ethernet frames to a classic pcap (used only by the synthetic test fixture)."""
    timestamps = timestamps or [i * 0.03 for i in range(len(frames))]
    with open(path, "wb") as handle:
        handle.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, LINKTYPE_ETHERNET))
        for ts, frame in zip(timestamps, frames):
            sec = int(ts)
            usec = int(round((ts - sec) * 1e6))
            handle.write(struct.pack("<IIII", sec, usec, len(frame), len(frame)))
            handle.write(frame)


def build_udp_frame(payload: bytes, src_port: int = 5500, dst_port: int = 5500) -> bytes:
    """Ethernet/IPv4/UDP frame around ``payload`` (checksums left zero)."""
    udp = struct.pack(">HHHH", src_port, dst_port, 8 + len(payload), 0) + payload
    total = 20 + len(udp)
    ip = struct.pack(">BBHHHBBH4s4s", 0x45, 0, total, 0, 0, 64, 17, 0,
                     bytes([10, 10, 10, 10]), bytes([255, 255, 255, 255]))
    eth = b"\xff" * 6 + b"\x00\x11\x22\x33\x44\x55" + b"\x08\x00"
    return eth + ip + udp
